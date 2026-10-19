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

use crate::commands::Data;

type Error = anyhow::Error;

/// Registered as the framework's `event_handler`.
pub async fn handle(
    ctx: poise::FrameworkContext<'_, Data, Error>,
    event: &serenityall::FullEvent,
) -> Result<(), Error> {
    // Approval-bridge button presses (FEATURES §15, approval.rs): a DM
    // card's Approve/Deny click resolves the pending id. Everything the
    // handler does is identity-checked inside `resolve_discord`: only
    // the configured owner id may resolve; strangers get a refusal row
    // typed on the card.
    if let serenityall::FullEvent::InteractionCreate { interaction, .. } = event
        && let serenityall::Interaction::Component(ci) = interaction
        && approval_press(ctx, ci).await
    {
        return Ok(());
    }
    let serenityall::FullEvent::Message { new_message: msg } = event else {
        return Ok(());
    };
    // Never react to other bots. But ANY attachment IS content: image-
    // only mentions/DMs are describable prompts (the 2026-10-17 deploy
    // dropped them), and file-only ones are sandbox-readable ones.
    let has_image_attachment = msg.attachments.iter().any(|a| {
        a.content_type
            .as_deref()
            .is_some_and(|t| t.starts_with("image/"))
    });
    let has_file_attachment = !msg.attachments.is_empty();
    if msg.author.bot || (msg.content.is_empty() && !has_file_attachment) {
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
    let contacted = leading || msg.mentions.iter().any(|u| u.id.get() == bot_id);

    // Tenor/GIF ack (§ gifs): a message that carries a gif AND reaches
    // the bot (a DM, or a mention) earns a deterministic emoji reaction.
    // A message that is ONLY gif links is acknowledged INSTEAD of
    // becoming a chat prompt — a bare tenor URL is not a question, and
    // the pipeline has no eyes for gif bytes anyway.
    let gif_attachments: Vec<(Option<&str>, &str)> = msg
        .attachments
        .iter()
        .map(|a| (a.content_type.as_deref(), a.filename.as_str()))
        .collect();
    let gif_embed_parts: Vec<String> = msg
        .embeds
        .iter()
        .flat_map(|e| {
            [
                e.url.as_deref().map(|s| s.to_string()),
                e.description.clone(),
                e.video.as_ref().map(|v| v.url.clone()),
                e.image.as_ref().map(|i| i.url.clone()),
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
        })
        .collect();
    let gif_parts: Vec<&str> = gif_embed_parts.iter().map(String::as_str).collect();
    let gif_hit = crate::gifs::message_is_gif(&msg.content, &gif_attachments, &gif_parts);
    if gif_hit && (contacted || is_dm) {
        let reaction = crate::gifs::pick_reaction(msg.id.get());
        match msg
            .react(
                ctx.serenity_context,
                serenityall::ReactionType::Unicode(reaction.to_string()),
            )
            .await
        {
            Ok(_) => {
                tracing::debug!(target: "bot", message_id = msg.id.get(), reaction, "gif recognized — reacted")
            }
            Err(e) => {
                tracing::debug!(target: "bot", error = %e, "gif reaction failed (missing perms is normal)")
            }
        }
        let gif_urls: Vec<&str> = msg
            .content
            .split_whitespace()
            .filter(|t| crate::gifs::is_gif_url(t))
            .collect();
        if crate::gifs::content_is_gif_only(&msg.content, &gif_urls) {
            let _ = msg
                .channel_id
                .say(
                    ctx.serenity_context,
                    format!("{reaction} nice GIF! Ask me for one anytime — `gif <topic>`"),
                )
                .await;
            return Ok(());
        }
    }

    let prompt = if !contacted {
        if is_dm {
            // Bare DM text (no mention possible medium-less): fully
            // conversational, like the old agent stack behaved. An
            // image-only DM (empty text, image attached) IS the prompt:
            // describe what was sent.
            let trimmed = msg.content.trim().to_string();
            if !trimmed.is_empty() {
                Some(trimmed)
            } else {
                (has_image_attachment || has_file_attachment)
                    .then(|| crate::converse::IMAGE_ONLY_PROMPT.to_string())
            }
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
            // Bare "@bot" with attachments: they ARE the instruction —
            // images describe, files sandbox-read. With neither: hint.
            if has_image_attachment || has_file_attachment {
                Some(crate::converse::IMAGE_ONLY_PROMPT.to_string())
            } else {
                // Bare "@bot" with no text: answer inline instead of
                // sending an empty prompt to the gateway.
                let _ = msg.channel_id.say(ctx.serenity_context, HINT).await;
                return Ok(());
            }
        } else {
            // A LEADING mention that names a real command (@bot ask hi,
            // @bot help, …) belongs to the framework: don't double-handle.
            // Mid-sentence mentions are conversational by definition here —
            // poise never sees them (it only dispatches on a leading glyph).
            if leading && first_token_is_command(&cleaned, &ctx.options().commands) {
                return Ok(());
            }
            Some(cleaned)
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
    // Attachments rode along: images get described through the gateway's
    // vision skill, other files get read INSIDE a disposable Docker
    // sandbox — and the bracketed context lands ahead of the prompt. The
    // model-visible text stays a String, so the pipeline needs no
    // multimodal plumbing at all.
    let attachments: Vec<crate::converse::AttSpec> = if contacted || is_dm {
        msg.attachments
            .iter()
            .map(|a| crate::converse::AttSpec {
                url: a.url.clone(),
                label: a.filename.clone(),
                kind: if a
                    .content_type
                    .as_deref()
                    .is_some_and(|t| t.starts_with("image/"))
                {
                    crate::converse::AttKind::Image
                } else {
                    crate::converse::AttKind::Other
                },
                id: Some(a.id.get()),
                size: Some(a.size as u64),
            })
            .collect()
    } else {
        Vec::new()
    };
    // Individual-user recognition: every turn keeps its speaker, so the
    // model (whose persona encodes the owner's id) can tell owner from
    // guests and address people by name across the conversation. The
    // tag is the transport's `speaker_prefix` contribution — the
    // pipeline itself applies it.
    let speaker_prefix = speaker_tag(&msg.author);
    // Privilege (owner-DM trust model, owner.rs): only a DM authored by
    // the configured owner admits the `[plugins] owner_only` skills (gcp
    // bridge) into this session's registry. Guild mentions — even the
    // owner's in-guild messages — are never privileged. Computed BEFORE
    // the spawn from Copy values: the spawned block must not borrow a
    // `FullEvent`-lifetime handle.
    let owner_privileged = is_dm
        && cfg
            .owner_discord_id
            .is_some_and(|id| id == msg.author.id.get());
    tokio::spawn(async move {
        show_typing(&http_ctx, channel).await;
        // Attachments first (downloads + vision/sandbox round-trips can
        // be slow; the typing indicator already covers the wait):
        // failures land as honest bracketed lines inside the prompt, never
        // a silent drop.
        let prompt =
            match crate::converse::attachments_context(&cfg, &state, &attachments, Some(&prompt))
                .await
            {
                Some(ctx) => format!("{ctx}\n{prompt}"),
                None => prompt,
            };
        // Live typing ([chat] stream, default on): the answer streams
        // through the gateway's SSE passthrough and edits into Discord
        // while tokens land. The sink degrades on its own — a failed
        // placeholder POST or a failed edit just silences it — so the
        // chat result path below is identically correct either way.
        // Buffered configs (stream = false) keep the plain one-shot send.
        let mut sink: Option<std::sync::Arc<dyn crate::converse::ConversationSink>> =
            if cfg.chat_stream {
                LiveSink::start(&http_ctx, channel)
                    .await
                    .map(|s| std::sync::Arc::new(s) as _)
            } else {
                None
            };
        // The whole pipeline (squeeze → speaker tag → chat → usage/history
        // bookkeeping → reply formatting → streamed delivery) has one
        // definition now (converse::run_session); this front-end only
        // chooses the sink and the error display.
        // Pre-session capture of the "live surface in play" bit: run_session
        // consumes the sink Option, so `sink.is_none()` afterwards no longer
        // means "no live surface". Testing it post-session was the live
        // 2026-10-17 09:55 double — every successful streamed chat re-posted
        // the same answer ~60 s later via send_long.
        let had_live_sink = sink.is_some();
        match crate::converse::run_session(
            &cfg,
            &state,
            crate::converse::Session {
                user,
                prompt,
                owner_privileged,
            },
            &speaker_prefix,
            &mut sink,
        )
        .await
        {
            Ok((text, _served_by)) => {
                // A live sink was already finished inside the pipeline;
                // buffered configs deliver here, as before — decided by
                // pre-session sink presence, not by the (now consumed)
                // Option.
                if crate::converse::buffered_delivery_needed(had_live_sink) {
                    send_long(&http_ctx, channel, &text).await;
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "conversational fallback chat failed");
                // Surface the actual cause (/ask already does — it propagates
                // the anyhow error). A generic "couldn't reach the mesh" hid
                // real distinctions: gateway-restart windows, a 401 from an
                // authed runtime lane, or a 90-s slow-answer timeout — each
                // needs a different fix from the operator, and "401" answers
                // "try again in a moment" with silence forever.
                let mut reason = e.to_string().replace('\n', "; ");
                if reason.chars().count() > 160 {
                    reason = format!(
                        "…{}",
                        reason
                            .chars()
                            .skip(reason.chars().count() - 160)
                            .collect::<String>()
                    );
                }
                let text = format!(
                    "⚠️ the mesh didn't answer just now — {reason}\ntry `@bot ask <question>` again in a moment."
                );
                if let Some(sink) = sink.take() {
                    sink.finish(text).await;
                } else {
                    send_long(&http_ctx, channel, &text).await;
                }
            }
        }
    });
    Ok(())
}

/// Shown when the bot is mentioned with no instruction at all.
const HINT: &str = "✋ Hi! Ask the mesh with `@bot ask <question>` or just DM me anything — try `@bot help` for commands.";

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
    let Some(tok) = first_token(text) else {
        return false;
    };
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
    format!(
        "[Discord user: {} (username {}, id {})]\n",
        display,
        user.name,
        user.id.get()
    )
}

fn first_token(text: &str) -> Option<&str> {
    text.split_whitespace().next()
}

/// Typing indicator while the mesh thinks (best-effort).
async fn show_typing(http_ctx: &serenityall::Context, channel: serenityall::ChannelId) {
    let _ = channel.broadcast_typing(http_ctx).await;
}

/// Live-typing sink for streamed @mention/DM chats: posts a placeholder
/// message once, then edits it while the gateway's SSE passthrough lands
/// content deltas (`LiveSink::start` → `tap` is the api layer's
/// per-fragment tap → `finish` replaces everything with the final formatted
/// reply and stops the edit ticker).
///
/// Edits are throttled to ≥ 2 s apart and ≥ 120 new chars — Discord both
/// rate-limits message edits and renders nothing from sub-token churn, so
/// tighter edits only trip the limiter. Every failure inside the sink is
/// swallowed (best-effort by contract): a deleted placeholder, a missing
/// permission, a Discord hiccup — the streamed chat itself continues, and
/// the streamed partial already shown is replaced wholesale by finish().
pub(crate) struct LiveSink {
    http_ctx: serenityall::Context,
    channel: serenityall::ChannelId,
    msg_id: serenityall::MessageId,
    pub(crate) state: std::sync::Arc<tokio::sync::Mutex<SinkState>>,
    ticker: tokio::task::JoinHandle<()>,
}

pub(crate) struct SinkState {
    text: String,
    /// Chars at the last successfully applied edit (throttle baseline).
    last_len: usize,
    last_edit: std::time::Instant,
    dirty: bool,
}

impl LiveSink {
    pub(crate) async fn start(
        http_ctx: &serenityall::Context,
        channel: serenityall::ChannelId,
    ) -> Option<LiveSink> {
        let placeholder = channel.say(http_ctx, "…").await.ok()?;
        tracing::info!(
            channel = %channel,
            message_id = %placeholder.id,
            "live sink: placeholder posted (one message per chat; the final edit below is the only other touch)"
        );
        let state = std::sync::Arc::new(tokio::sync::Mutex::new(SinkState {
            text: String::new(),
            last_len: 0,
            last_edit: std::time::Instant::now() - std::time::Duration::from_secs(10),
            dirty: false,
        }));
        // Ticker: apply pending edits every 2 s until finish() aborts us.
        let tick_state = state.clone();
        let ticker_ctx = http_ctx.clone();
        let ticker = tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(2));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                interval.tick().await;
                let mut s = tick_state.lock().await;
                if s.dirty
                    && s.text.chars().count().saturating_sub(s.last_len) >= 120
                    && s.last_edit.elapsed() >= std::time::Duration::from_secs(2)
                {
                    let shown = clip_tail(&s.text, 1990);
                    let msg_id = placeholder.id;
                    if channel
                        .edit_message(
                            &ticker_ctx,
                            msg_id,
                            poise::serenity_prelude::EditMessage::new().content(shown),
                        )
                        .await
                        .is_ok()
                    {
                        s.last_len = s.text.chars().count();
                        s.last_edit = std::time::Instant::now();
                        s.dirty = false;
                    }
                    // Stay dirty: the next tick or finish() retries. A
                    // deleted message fails every later edit — harmless
                    // noise, bounded by finish() aborting this task.
                }
            }
        });
        Some(LiveSink {
            http_ctx: http_ctx.clone(),
            channel,
            msg_id: placeholder.id,
            state,
            ticker,
        })
    }
}

/// The live-typing sink as the pipeline's seam (Phase 3b): fragments
/// flow through `on_event` (the same try-lock discipline the old
/// `tap_for` bridge used), the final formatted reply through `finish`
/// (which also aborts the edit ticker). The `Arc` keying exists because
/// the pipeline finishes the sink while its own per-request tap holds a
/// clone.
impl crate::converse::ConversationSink for LiveSink {
    fn on_event(&self, event: crate::api::LiveEvent<'_>) {
        if let Ok(mut s) = self.state.try_lock() {
            match event {
                crate::api::LiveEvent::RoundStart => {
                    s.text.clear();
                    s.dirty = true;
                }
                crate::api::LiveEvent::Text(frag) => {
                    if !frag.is_empty() {
                        s.text.push_str(frag);
                        s.dirty = true;
                    }
                }
            }
        }
    }

    fn finish(
        self: std::sync::Arc<Self>,
        text: String,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> {
        Box::pin(async move {
            self.ticker.abort();
            let clipped = clip_tail(&text, 1990);
            // Two identical messages = the resend path firing: the ticker
            // already typed the full answer into the placeholder, and ONE
            // failed final edit then posted the same text again as a NEW
            // message (the live 2026-10-16 19:08 "two identical answers"
            // incident shape). Retry the edit IN PLACE first — a transient
            // Discord 429/5xx clears in a second or two; only a persistent
            // failure (deleted placeholder, lost permission) deserves the
            // resend fallback.
            let mut delivered = false;
            for attempt in 1..=3u8 {
                let step = self
                    .channel
                    .edit_message(
                        &self.http_ctx,
                        self.msg_id,
                        poise::serenity_prelude::EditMessage::new().content(&clipped),
                    )
                    .await;
                match step {
                    Ok(_) => {
                        delivered = true;
                        tracing::info!(
                            channel = %self.channel,
                            message_id = %self.msg_id,
                            attempt,
                            "live sink: final edit applied in place (no resend)"
                        );
                        break;
                    }
                    Err(e) => {
                        if attempt < 3 {
                            tracing::warn!(
                                attempt,
                                error = %e,
                                "live sink final edit failed — retrying in place before any resend"
                            );
                            tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
                        } else {
                            tracing::warn!(error = %e, "live sink final edit failed 3× — resending (a second message appears when this fires)");
                        }
                    }
                }
            }
            if !delivered {
                send_long(&self.http_ctx, self.channel, &text).await;
            }
        })
    }
}

/// Discord's per-message cap is 2000 chars — keep the tail (the answer end)
/// with the same `say_long` convention used by every command.
/// Fire-and-forget: every caller already discards the error, so the helper
/// logs its own failure and returns nothing (a fat `serenity::Error` has no
/// business being smuggled through a Result nobody reads).
async fn send_long(http_ctx: &serenityall::Context, channel: serenityall::ChannelId, text: &str) {
    let clipped = clip_tail(text, 1990);
    if let Err(e) = channel.say(http_ctx, clipped).await {
        tracing::debug!(error = %e, "send_long: tail resend failed");
    }
}

pub(crate) fn clip_tail(text: &str, max: usize) -> String {
    if text.chars().count() > max {
        let skip = text.chars().count() - max + 3;
        format!("…{}", text.chars().skip(skip).collect::<String>())
    } else {
        text.to_string()
    }
}

// ---------------------------------------------------------------------------
// Approval-bridge button presses (FEATURES §15, approval.rs)
// ---------------------------------------------------------------------------

/// Serve one component interaction that belongs to the approval bridge.
/// Returns `Ok(true)` when the interaction was consumed (approval custom
/// id — answered in here, whether the presser was the owner or not), so
/// the caller can stop without poise treating it as an unowned event.
async fn approval_press(
    ctx: poise::FrameworkContext<'_, Data, Error>,
    ci: &serenityall::ComponentInteraction,
) -> bool {
    // Bridge buttons only; anything else is some other integration's.
    if crate::approval::parse_button(&ci.data.custom_id).is_none() {
        return false;
    }
    let user_id = ci.user.id.get();
    let cfg_owner = ctx.user_data().await.cfg.owner_discord_id;
    let is_owner = Some(user_id) == cfg_owner;
    let ack = |content: String| {
        ci.create_response(
            ctx.serenity_context,
            serenityall::CreateInteractionResponse::Message(
                serenityall::CreateInteractionResponseMessage::new()
                    .content(content)
                    .ephemeral(true),
            ),
        )
    };
    let custom_id = ci.data.custom_id.clone();
    let reply = if !is_owner {
        "That approval belongs to the bot owner — this card will not act on your press.".to_string()
    } else {
        match crate::approval::resolve_discord(user_id, &custom_id).await {
            Ok(line) => line,
            Err(e) => approval_error_line(&e.to_string()),
        }
    };
    if let Err(e) = ack(reply).await {
        tracing::warn!(error = %e, "approval button ack failed");
    }
    true
}
/// First line of an error string, clipped (card text discipline). Named
/// past `first_line` — chat_fallback already defines one for its own
/// reply clipping, and these two stay free to evolve separately.
use crate::approval::first_line as approval_error_line;

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
        assert_eq!(
            strip_bot_mentions("<@42> check the mesh", 42),
            " check the mesh"
        );
        assert_eq!(
            strip_bot_mentions("hey <@42> what's up", 42),
            "hey  what's up"
        );
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
