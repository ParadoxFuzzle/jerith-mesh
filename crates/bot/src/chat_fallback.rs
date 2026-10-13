//! Conversational fallback for @mentions and DMs.
//!
//! The framework only dispatches slash/prefix commands: a bare mention
//! followed by free text logs "didn't recognize command name" (poise), and
//! DMs without any command glyph are dropped entirely. This handler catches
//! both cases and routes the text through the same gateway chat path `/ask`
//! uses — so @mentioning the bot or DM'ing it gets a mesh-served reply, with
//! the same per-user history, cooldown gate and Jerith Guard screening.
//!
//! In-guild messages without a leading bot mention are ignored: the bot
//! should not answer every channel message. In DMs there is no prefix
//! medium, so plain text is treated as conversational input.
//!
//! Handler contract (poise 0.7): non-capturing fn over
//! `(FrameworkContext<Data, E>, &serenity::FullEvent) -> BoxFuture<Result<(), E>>`,
//! so the whole chat call is spawned off the dispatch loop as soon as the
//! cheap checks (author/gate/prompt) are done.

use poise::serenity_prelude as serenityall;

use crate::api;
use crate::commands::Data;

type Error = anyhow::Error;

/// Registered as the framework's `event_handler`.
pub async fn handle(
    ctx: poise::FrameworkContext<'_, Data, Error>,
    event: &serenityall::FullEvent,
) -> Result<(), Error> {
    let serenityall::FullEvent::Message { new_message: msg } = event else {
        return Ok(());
    };
    // Never react to other bots (incl. ourselves) or to attachments-only
    // messages (nothing textual to forward).
    if msg.author.bot || msg.content.is_empty() {
        return Ok(());
    }

    let bot_id = ctx.bot_id().get();
    let is_dm = msg.guild_id.is_none();

    let prompt = match strip_leading_bot_mention(&msg.content, bot_id) {
        Some(rest) => {
            let rest = rest.trim();
            if rest.is_empty() {
                // Bare "@bot" with no text: answer inline instead of sending
                // an empty prompt to the gateway.
                let _ = msg
                    .channel_id
                    .say(ctx.serenity_context, HINT)
                    .await;
                return Ok(());
            }
            // A leading mention that names a real command (@bot /ask hi,
            // @bot help, …) belongs to the framework: don't double-handle.
            if first_token_is_command(rest, &ctx.options().commands) {
                return Ok(());
            }
            Some(rest.to_string())
        }
        None => {
            if is_dm {
                // Bare DM text (no mention possible medium-less): fully
                // conversational, like the old agent stack behaved.
                Some(msg.content.trim().to_string())
            } else {
                // Guild message not addressed to us: stay quiet.
                None
            }
        }
    };
    let Some(prompt) = prompt else { return Ok(()) };

    let user = msg.author.id.get();
    let data = ctx.user_data().await;
    // Cooldown gate before spawning: cheap, sync, and prevents a backlog of
    // spawned chats when someone @mentions in rapid succession.
    if let Err(text) = data.state.gate_ask(user) {
        let _ = msg.channel_id.say(ctx.serenity_context, text).await;
        return Ok(());
    }

    let http_ctx = ctx.serenity_context.clone();
    let channel = msg.channel_id;
    let state = data.state.clone();
    let cfg = data.cfg.clone();
    tokio::spawn(async move {
        show_typing(&http_ctx, channel).await;
        let history = state.history(user);
        match api::chat(&cfg, &prompt, &history).await {
            Ok((reply, served_by)) => {
                state.push_turn(user, &prompt, &reply);
                let text = crate::commands::format_reply(&reply, served_by.as_deref());
                let _ = send_long(&http_ctx, channel, &text).await;
            }
            Err(e) => {
                tracing::warn!(error = %e, "conversational fallback chat failed");
                let _ = channel
                    .say(&http_ctx, "⚠️ couldn't reach the mesh just now — try `@bot ask <question>` in a moment.")
                    .await;
            }
        }
    });
    Ok(())
}

/// Shown when the bot is mentioned with no instruction at all.
const HINT: &str =
    "✋ Hi! Ask the mesh with `@bot ask <question>` or just DM me anything — try `@bot help` for commands.";

/// If `content` opens with this bot's mention (`<@id>` / `<@!id>`), return
/// the remainder; otherwise `None`.
pub fn strip_leading_bot_mention(content: &str, bot_user_id: u64) -> Option<&str> {
    for pat in [format!("<@{bot_user_id}>"), format!("<@!{bot_user_id}>")] {
        if let Some(rest) = content.strip_prefix(pat.as_str()) {
            return Some(rest);
        }
    }
    None
}

/// Whether the message fragment's first token names a registered command
/// (slash/prefix commands share their names + aliases here).
fn first_token_is_command(text: &str, commands: &[poise::Command<Data, Error>]) -> bool {
    let Some(tok) = first_token(text) else { return false };
    commands.iter().any(|c| {
        c.name.as_ref().eq_ignore_ascii_case(tok)
            || c.aliases.iter().any(|a| a.eq_ignore_ascii_case(tok))
    })
}

fn first_token(text: &str) -> Option<&str> {
    text.split_whitespace().next()
}

/// Typing indicator while the mesh thinks (best-effort).
async fn show_typing(http_ctx: &serenityall::Context, channel: serenityall::ChannelId) {
    let _ = channel.broadcast_typing(http_ctx).await;
}

/// Discord's per-message cap is 2000 chars — keep the tail (the answer end)
/// with the same `say_long` convention used by every command.
async fn send_long(
    http_ctx: &serenityall::Context,
    channel: serenityall::ChannelId,
    text: &str,
) -> Result<(), serenityall::Error> {
    let clipped = clip_tail(text, 1990);
    channel.say(http_ctx, clipped).await.map(|_| ())
}

pub(crate) fn clip_tail(text: &str, max: usize) -> String {
    if text.chars().count() > max {
        let skip = text.chars().count() - max + 3;
        format!("…{}", text.chars().skip(skip).collect::<String>())
    } else {
        text.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mention_strip_both_parenthesization_variants() {
        assert_eq!(strip_leading_bot_mention("<@42> hi", 42), Some(" hi"));
        assert_eq!(strip_leading_bot_mention("<@!42> hi", 42), Some(" hi"));
        assert_eq!(strip_leading_bot_mention("<@42>", 42), Some(""));
        assert_eq!(strip_leading_bot_mention("hi", 42), None);
        assert_eq!(strip_leading_bot_mention("<@43> hi", 42), None);
        assert_eq!(strip_leading_bot_mention("a <@42> b", 42), None);
    }

    #[test]
    fn first_token_extracts_command_word() {
        assert_eq!(first_token("ask hi"), Some("ask"));
        assert_eq!(first_token("This is a test"), Some("This"));
        assert_eq!(first_token("   "), None);
    }

    #[test]
    fn clip_tail_keeps_tail_with_ellipsis() {
        let long = "x".repeat(3000);
        let clipped = clip_tail(&long, 1990);
        // Same formula as commands::say_long: 3 ellipsis-ish chars net
        // consumed, so the total stays just under the 2000-char cap.
        assert_eq!(clipped.chars().count(), 1988);
        assert!(clipped.starts_with('…'));
        assert!(clipped.ends_with('x'));
        assert_eq!(clip_tail("short", 1990), "short");
    }
}
