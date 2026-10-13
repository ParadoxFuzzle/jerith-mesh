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

    // "Contacted" = the bot is mentioned anywhere in the message. The
    // serenity `mentions` list is authoritative (covers <@id> and <@!id>)
    // and arrives fully populated in guild channels even without the
    // privileged message-content intent. `leading` matters for hand-off:
    // a leading mention naming a real command belongs to the framework.
    let leading = strip_leading_bot_mention(&msg.content, bot_id).is_some();
    let contacted = leading
        || msg.mentions.iter().any(|u| u.id.get() == bot_id);

    let prompt = if !contacted {
        if is_dm {
            // Bare DM text (no mention possible medium-less): fully
            // conversational, like the old agent stack behaved.
            Some(msg.content.trim().to_string())
        } else {
            // Guild message not addressed to us: stay quiet. (Wider
            // "answer every channel message" would need Discord's
            // privileged Message Content intent — see the readme.)
            None
        }
    } else {
        // Mentioned anywhere in a DM or server: strip every bot-mention
        // token (leading or mid-sentence) and treat the remainder as the
        // prompt, so "check @Jerith status ok?" converses naturally.
        let cleaned = strip_bot_mentions(&msg.content, bot_id).trim().to_string();
        if cleaned.is_empty() {
            // Bare "@bot" with no text: answer inline instead of sending an
            // empty prompt to the gateway.
            let _ = msg
                .channel_id
                .say(ctx.serenity_context, HINT)
                .await;
            return Ok(());
        }
        // A LEADING mention that names a real command (@bot ask hi,
        // @bot help, …) belongs to the framework: don't double-handle.
        // Mid-sentence mentions are conversational by definition here —
        // poise never sees them (it only dispatches on a leading glyph).
        if leading && first_token_is_command(&cleaned, &ctx.options().commands) {
            return Ok(());
        }
        Some(cleaned)
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
    // Individual-user recognition: every turn keeps its speaker, so the
    // model (whose persona encodes the owner's id) can tell owner from
    // guests and address people by name across the conversation.
    let tagged = format!("{}{}", speaker_tag(&msg.author), prompt);
    tokio::spawn(async move {
        show_typing(&http_ctx, channel).await;
        let history = state.history(user);
        match api::chat(&cfg, &tagged, &history).await {
            Ok((reply, served_by)) => {
                state.push_turn(user, &tagged, &reply);
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

/// Remove every occurrence of this bot's mention tokens (both `<@id>` and
/// `<@!id>` spellings, anywhere in the text), leaving other users' mentions
/// and the surrounding text intact.
pub fn strip_bot_mentions(content: &str, bot_user_id: u64) -> String {
    let mut out = content.to_string();
    for pat in [format!("<@{bot_user_id}>"), format!("<@!{bot_user_id}>")] {
        out = out.replace(pat.as_str(), "");
    }
    out
}

/// Tag the caller's identity onto their message so the model sees WHO is
/// talking (the persona already encodes the owner's Discord id, giving it
/// the same individual-user awareness the OpenClaw gateway had). Kept in
/// history, so each stored turn carries its speaker across the conversation.
pub fn speaker_tag(user: &serenityall::User) -> String {
    let display = user
        .global_name
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or(&user.name);
    format!("[Discord user: {} (username {}, id {})]\n", display, user.name, user.id.get())
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

    #[test]
    fn strip_all_bot_mentions_mid_text_and_variants() {
        // Leading mention (the old focus) plus mid-sentence and trailing.
        assert_eq!(strip_bot_mentions("<@42> check the mesh", 42), " check the mesh");
        assert_eq!(strip_bot_mentions("hey <@42> what's up", 42), "hey  what's up");
        assert_eq!(strip_bot_mentions("ping <@!42>?", 42), "ping ?");
        assert_eq!(strip_bot_mentions("<@42><@!42> bounce", 42), " bounce");
        // Other users' mentions stay intact.
        assert_eq!(strip_bot_mentions("a <@7> b <@42> c", 42), "a <@7> b  c");
    }

    #[test]
    fn speaker_tag_carries_display_name_username_and_id() {
        let mut u = serenityall::User::default();
        u.id = serenityall::UserId::new(12345);
        u.name = "noodly_handle".into();
        u.global_name = Some("NoodlyPanda".into());
        assert_eq!(
            speaker_tag(&u),
            "[Discord user: NoodlyPanda (username noodly_handle, id 12345)]\n"
        );
        // Falls back to the username when there is no display name.
        u.global_name = None;
        assert!(speaker_tag(&u).starts_with("[Discord user: noodly_handle "));
    }
}
