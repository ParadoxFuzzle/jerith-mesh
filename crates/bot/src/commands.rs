//! Slash commands. Thin wrappers: gate cooldowns → call `api::*` → format a
//! reply. Formatting is split from IO so tests can cover it without a
//! Discord connection.

use anyhow::Context as _;
use poise::serenity_prelude as serenityall;

use crate::api;
use crate::chat_fallback::clip_tail;
use crate::config::BotConfig;
use crate::state::BotState;

/// Data passed to every command.
pub struct Data {
    pub cfg: BotConfig,
    pub state: BotState,
}

type Error = anyhow::Error;
type Ctx<'a> = poise::Context<'a, Data, Error>;

fn user_id(ctx: &Ctx<'_>) -> u64 {
    ctx.author().id.get()
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

/// Show a file's sandbox-extracted content on demand.
#[poise::command(slash_command, broadcast_typing)]
pub async fn read(
    ctx: Ctx<'_>,
    #[description = "The file to read (attach it when you send this command)"]
    file: serenityall::Attachment,
) -> Result<(), Error> {
    let user = user_id(&ctx);
    if let Err(msg) = ctx.data().state.gate_cmd(user) {
        ctx.say(msg).await?;
        return Ok(());
    }
    ctx.defer().await?;
    // The same routing the chat context uses: images get the vision
    // skill's description, other files get the Docker sandbox read —
    // and the extraction cache means a re-sent file doesn't re-run the
    // container.
    let kind = if file
        .content_type
        .as_deref()
        .is_some_and(|t| t.starts_with("image/"))
    {
        crate::converse::AttKind::Image
    } else {
        crate::converse::AttKind::Other
    };
    let spec = crate::converse::AttSpec {
        url: file.url.clone(),
        label: file.filename.clone(),
        kind,
        id: Some(file.id.get()),
        size: Some(file.size as u64),
    };
    let block = match
        crate::converse::attachments_context(
            &ctx.data().cfg,
            &ctx.data().state,
            std::slice::from_ref(&spec),
            None,
        )
        .await
    {
        Some(b) => b,
        None => format!("- [{}] (no readable content)", file.filename),
    };
    // Body only (drop the block header), Discord-bounded, honest tail
    // when the extraction ran longer than one message can show.
    let body = block
        .lines()
        .skip(1)
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string();
    let (shown, note) = if body.chars().count() > 1600 {
        let total = body.chars().count();
        (
            body.chars().take(1600).collect::<String>(),
            format!("\n_(…truncated — {} chars extracted in total)_", total),
        )
    } else {
        (body, String::new())
    };
    ctx.say(format!("📄 {shown}{note}")).await?;
    Ok(())
}

/// Ask the mesh a question (inherits the gateway's mesh-peer fallback).
#[poise::command(slash_command, prefix_command, broadcast_typing)]
pub async fn ask(
    ctx: Ctx<'_>,
    #[description = "Your question"] prompt: String,
) -> Result<(), Error> {
    let user = user_id(&ctx);
    if let Err(msg) = ctx.data().state.gate_ask(user) {
        ctx.say(msg).await?;
        return Ok(());
    }
    ctx.defer().await?;

    let cfg = ctx.data().cfg.clone();
    // Individual-user recognition (same convention as the DM/@mention path):
    // every turn keeps its speaker so the model can tell owner from guests
    // and address people by name across the conversation. The tag is this
    // transport's `speaker_prefix`; the pipeline applies it host-side.
    let speaker_prefix = crate::chat_fallback::speaker_tag(ctx.author());
    // Streaming ([chat] stream, default on): /ask live-types the answer the
    // same way the @mention/DM fallback does — placeholder message, throttled
    // edits while tokens land, one final formatted replacement. The
    // placeholder lives in the command's channel (slash replies would
    // otherwise need the deferred-response edit path); `stream = false`
    // restores the buffered single-shot `say_long`.
    let mut sink: Option<std::sync::Arc<dyn crate::converse::ConversationSink>> = if cfg.chat_stream
    {
        crate::chat_fallback::LiveSink::start(&ctx.serenity_context().clone(), ctx.channel_id())
            .await
            .map(|s| std::sync::Arc::new(s) as _)
    } else {
        None
    };
    // The whole pipeline (squeeze → speaker tag → chat → usage/history
    // bookkeeping → reply formatting → streamed delivery) has one
    // definition now (converse::run_session); /ask's remaining
    // transport-specific bits: tally persistence on chat success, the
    // buffered `say_long`, and poise error propagation when no sink exists.
    // Pre-session capture of the "live surface in play" bit: run_session
    // consumes the sink Option, so `sink.is_none()` afterwards no longer
    // means "no live surface" — testing it post-session re-posted every
    // successful streamed chat (the live 2026-10-17 09:55 double).
    let had_live_sink = sink.is_some();
    let chat = crate::converse::run_session(
        &cfg,
        &ctx.data().state,
        crate::converse::Session { user, prompt },
        &speaker_prefix,
        &mut sink,
    )
    .await;
    match chat {
        Ok((text, _served_by)) => {
            // Persist after chats too (not just /mine wins) so token spend and
            // history survive restarts on chat-active bots.
            persist_tally_after_win(ctx).await;
            if crate::converse::buffered_delivery_needed(had_live_sink) {
                say_long(ctx, &text).await?;
            }
            // Sink-delivered: the pipeline already streamed + finished.
        }
        Err(e) => match sink.take() {
            Some(sink) => sink.finish(format!("⚠️ chat failed just now — {e}")).await,
            None => return Err(e),
        },
    }
    Ok(())
}

/// Clear your in-memory chat history with the bot.
#[poise::command(slash_command, prefix_command)]
pub async fn reset(ctx: Ctx<'_>) -> Result<(), Error> {
    let dropped = ctx.data().state.reset(user_id(&ctx));
    let text = if dropped == 0 {
        "🧠 no conversation history to clear.".to_string()
    } else {
        format!("🧠 cleared {dropped} turn(s) of conversation history.")
    };
    ctx.say(text).await?;
    Ok(())
}

/// Store a note in the node's private memory (never leaves the node).
#[poise::command(slash_command, prefix_command)]
pub async fn remember(
    ctx: Ctx<'_>,
    #[description = "Note to memorize"] text: String,
) -> Result<(), Error> {
    if let Err(msg) = ctx.data().state.gate_cmd(user_id(&ctx)) {
        ctx.say(msg).await?;
        return Ok(());
    }
    ctx.defer().await?;
    let (id, provider) = api::memory_embed(&ctx.data().cfg, &text).await?;
    ctx.say(format!("💾 stored note `{id}` ({provider} embeddings)."))
        .await?;
    Ok(())
}

/// Search the node's private memory.
#[poise::command(slash_command, prefix_command)]
pub async fn recall(
    ctx: Ctx<'_>,
    #[description = "What to search for"] query: String,
) -> Result<(), Error> {
    if let Err(msg) = ctx.data().state.gate_cmd(user_id(&ctx)) {
        ctx.say(msg).await?;
        return Ok(());
    }
    ctx.defer().await?;
    let hits = api::memory_search(&ctx.data().cfg, &query, 5).await?;
    ctx.say(format_recall(&query, &hits)).await?;
    Ok(())
}

/// Node + mesh status snapshot.
#[poise::command(slash_command, prefix_command)]
pub async fn status(ctx: Ctx<'_>) -> Result<(), Error> {
    if let Err(msg) = ctx.data().state.gate_cmd(user_id(&ctx)) {
        ctx.say(msg).await?;
        return Ok(());
    }
    ctx.defer().await?;
    let st = api::gateway_status(&ctx.data().cfg).await?;
    let your_tokens = ctx.data().state.tokens_used(user_id(&ctx));
    ctx.say(format_status_with_tokens(&st, Some(your_tokens)))
        .await?;
    Ok(())
}

/// List live peers in this model's mesh.
#[poise::command(slash_command, prefix_command)]
pub async fn peers(ctx: Ctx<'_>) -> Result<(), Error> {
    if let Err(msg) = ctx.data().state.gate_cmd(user_id(&ctx)) {
        ctx.say(msg).await?;
        return Ok(());
    }
    ctx.defer().await?;
    let st = api::gateway_status(&ctx.data().cfg).await?;
    ctx.say(format_peers(&st)).await?;
    Ok(())
}

/// List recent mail for business@/ai@ (owner-DM only).
// Inbox contents are the owner's eyes only: deliberately NOT in owner.rs
// `USER_COMMANDS` so the default-deny gate makes this owner-DM only
// (users elsewhere get the standard refusal). The long privacy rationale
// lives here because poise turns the FIRST doc paragraph into the slash
// description and Discord caps that at 100 chars.
#[poise::command(slash_command, prefix_command)]
pub async fn mail(
    ctx: Ctx<'_>,
    #[description = "Which inbox: business, ai, or empty for all"] inbox: Option<String>,
    #[description = "How many messages (default 10, max 25)"] count: Option<u16>,
) -> Result<(), Error> {
    let box_filter = inbox.as_deref().map(str::to_ascii_lowercase).and_then(|b| {
        // Only the two provisioned mailboxes are addressable; keeps typos
        // ("buiness") from returning a misleading "no mail for buiness@".
        (b == "business" || b == "ai").then_some(b)
    });
    let limit = count.unwrap_or(10).min(25) as usize;
    ctx.defer().await?;
    let messages = crate::mailbox::list_messages_in(
        &crate::mailbox::default_base_dir(),
        box_filter.as_deref(),
        limit,
    );
    ctx.say(crate::mailbox::format_listing(
        &messages,
        box_filter.as_deref(),
    ))
    .await?;
    Ok(())
}

/// Try to mine JER on the local JerithChain node (per-user cooldown).
#[poise::command(slash_command, prefix_command)]
pub async fn mine(ctx: Ctx<'_>) -> Result<(), Error> {
    let discord_id = user_id(&ctx);
    if let Err(msg) = ctx.data().state.gate_mine(discord_id) {
        ctx.say(msg).await?;
        return Ok(());
    }
    ctx.defer().await?;
    let channel = ctx.channel_id().get().to_string();
    let user = discord_id.to_string();
    let outcome = api::mine(&ctx.data().cfg, &user, &channel).await?;
    // Credit the bot-observed /leaderboard when the chain reports a reward.
    if outcome.won
        && let Some(uj) = outcome.reward_uj
    {
        ctx.data().state.record_mine_win(discord_id, uj);
        // FEATURES §8: /mine win as a hookable event (fire-and-forget).
        crate::hooks::global().emit(crate::hooks::Event::new(
            crate::hooks::EV_MINE_WON,
            serde_json::json!({ "discord_id": discord_id, "amount_uj": uj }),
        ));
    }
    // Optional: announce wins in a configured channel (fire-and-forget; a
    // missing/incorrect channel id must not fail the user's command).
    if let Some(announce_id) = ctx.data().cfg.announce_channel_id
        && let Some(text) = api::mine_announcement(&outcome, &user)
    {
        let http = ctx.serenity_context().http.clone();
        tokio::spawn(async move {
            let ch = serenityall::ChannelId::new(announce_id);
            if let Err(e) = ch.say(&http, &text).await {
                tracing::warn!(error = %e, channel = announce_id, "mine announcement failed");
            }
        });
    }
    persist_tally_after_win(ctx).await;
    ctx.say(outcome.message).await?;
    Ok(())
}

/// Save the tally to disk after a /mine (cheap: small JSON, atomic rename).
/// Best-effort: failures log a warning and don't affect the reply.
async fn persist_tally_after_win(ctx: Ctx<'_>) {
    // Snapshot the state Arc, then write in the background so the user's
    // reply isn't delayed by disk IO.
    let st = ctx.data().state.clone();
    tokio::spawn(async move {
        tokio::task::spawn_blocking(move || {
            let path = crate::state::tally_path();
            if let Err(e) = st.save_to_disk(&path) {
                tracing::warn!(error = %e, path = %path.display(), "failed to persist mine tally");
            }
        })
        .await
        .ok();
    });
}

/// JER mining leaderboard (chain totals when available).
#[poise::command(slash_command, prefix_command)]
pub async fn leaderboard(ctx: Ctx<'_>) -> Result<(), Error> {
    if let Err(msg) = ctx.data().state.gate_cmd(user_id(&ctx)) {
        ctx.say(msg).await?;
        return Ok(());
    }
    let viewer = user_id(&ctx);
    // Chain is authoritative; on failure use the bot's persisted tally.
    let token = crate::core_token().ok();
    let chain = match api::chain_leaderboard(&ctx.data().cfg, token, 10).await {
        Ok(entries) if !entries.is_empty() => {
            let rows: Vec<(u64, u64, u32)> = entries
                .into_iter()
                .map(|e| {
                    (
                        e.discord_id,
                        e.total_mined_uj,
                        e.wins.min(u32::MAX as u64) as u32,
                    )
                })
                .collect();
            ctx.say(format_leaderboard(&rows, viewer, "chain totals (lifetime)"))
                .await?;
            return Ok(());
        }
        Ok(_) => "chain reports no wins yet",
        Err(e) => {
            tracing::debug!(error = %e, "chain leaderboard unavailable, using bot tally");
            ""
        }
    };
    let rows = ctx.data().state.leaderboard(10);
    let source = if chain.is_empty() && rows.is_empty() {
        ""
    } else if chain.is_empty() {
        "bot-observed wins (chain unavailable)"
    } else {
        ""
    };
    ctx.say(format_leaderboard(&rows, viewer, source)).await?;
    Ok(())
}

/// Show your own JER balance on the local JerithChain node.
#[poise::command(slash_command, prefix_command)]
pub async fn myjercraft(ctx: Ctx<'_>) -> Result<(), Error> {
    if let Err(msg) = ctx.data().state.gate_cmd(user_id(&ctx)) {
        ctx.say(msg).await?;
        return Ok(());
    }
    ctx.defer().await?;
    let did = user_id(&ctx);
    let token = crate::core_token()?;
    let bal = api::my_balance(&ctx.data().cfg, &token, did).await?;
    ctx.say(format_my_balance(&bal, did)).await?;
    Ok(())
}

/// Briefing of the watched threads (decisions + mesh digest).
///
/// The full what-did-i-miss port of OpenClaw's skill: sweeps each
/// configured archiver channel's latest messages over Discord REST,
/// extracts the decision/tech-note lines and adds a 3-line mesh reasoning
/// digest. Read-only — no memory writes, no web round-trips.
#[poise::command(slash_command, prefix_command, broadcast_typing)]
pub async fn briefing(
    ctx: Ctx<'_>,
    #[description = "Hours to look back (1-72, default 24)"] hours: Option<i64>,
) -> Result<(), Error> {
    if let Err(msg) = ctx.data().state.gate_cmd(user_id(&ctx)) {
        ctx.say(msg).await?;
        return Ok(());
    }
    ctx.defer().await?;
    let hours = hours.unwrap_or(24).clamp(1, 72);
    let since_snowflake = discord_snowflake_for_hours_ago(hours);
    let cfg = ctx.data().cfg.clone();
    if cfg.archiver.channel_ids.is_empty() {
        ctx.say(
            "No archiver channels configured — add `[archiver] channel_ids` to bot.toml first.",
        )
        .await?;
        return Ok(());
    }
    let client = crate::archiver::discord_client();
    let mut lines: Vec<String> = Vec::new();
    for cid in &cfg.archiver.channel_ids {
        let msgs = match crate::archiver::fetch_messages(
            &client,
            &cfg.discord_token,
            *cid,
            Some(since_snowflake),
            80,
        )
        .await
        {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!(channel = cid, error = %e, "briefing channel fetch failed");
                continue;
            }
        };
        // Per-guild marker resolution needs the channel's owning guild.
        let (label, guild) = crate::archiver::thread_info(&client, &cfg.discord_token, *cid).await;
        let markers = crate::archiver::effective_markers(&cfg.archiver, guild);
        let notes = crate::archiver::extract_notes(
            &msgs,
            &markers.iter().map(String::as_str).collect::<Vec<_>>(),
            6,
            None,
        );
        if !notes.is_empty() {
            lines.push(format!("**{label}**"));
            lines.extend(notes.iter().take(4).map(|n| format!("- {n}")));
        }
    }
    if lines.is_empty() {
        ctx.say(format!("Nothing decision-shaped in the last {hours}h across the watched threads — quiet, or markers need tuning (see bot.toml [archiver] markers).")).await?;
        return Ok(());
    }
    // A short mesh digest on top (a direct chat call — no tools in this
    // path because the digest prompt is not a question, per the tool gate).
    let digest = match api::chat(
        &cfg,
        &format!(
            "You are briefing an operator. In at most 3 short lines, give the high-level state of this channel activity digest (no narration, no preamble):\n{}",
            lines.join("\n")
        ),
        &[],
    )
    .await
    {
        Ok((text, _)) if !crate::api::answer_is_degenerate(&text) => {
            format!("**Mesh digest:** {}\n\n", text.trim().lines().take(3).collect::<Vec<_>>().join(" "))
        }
        _ => String::new(),
    };
    ctx.say(clip_tail(&format!("{digest}{}", lines.join("\n")), 1900))
        .await?;
    Ok(())
}

/// Discord snowflake for `hours` ago (snowflake epoch 2015-01-01T00:00Z).
fn discord_snowflake_for_hours_ago(hours: i64) -> u64 {
    const SNOWFLAKE_EPOCH_MS: u64 = 1_420_070_400_000;
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(SNOWFLAKE_EPOCH_MS);
    let cutoff_ms = now_ms.saturating_sub((hours.max(0) as u64) * 3_600_000);
    cutoff_ms.saturating_sub(SNOWFLAKE_EPOCH_MS) << 22
}

/// Search the lore vault (sourced excerpts, no model in the loop).
#[poise::command(slash_command, prefix_command, broadcast_typing)]
pub async fn lore(
    ctx: Ctx<'_>,
    #[description = "Canon question or terms"] query: String,
) -> Result<(), Error> {
    if let Err(msg) = ctx.data().state.gate_cmd(user_id(&ctx)) {
        ctx.say(msg).await?;
        return Ok(());
    }
    ctx.defer().await?;
    let st = ctx.data().cfg.clone();
    let hits = lore_search(&st, &query, 4).await?;
    if hits.is_empty() {
        ctx.say(format!("No lore matched \"{query}\" — the vault is indexed node-side with exact terms; try a proper noun."))
            .await?;
        return Ok(());
    }
    let mut out = format!("📚 **Lore — {query}**\n");
    for h in hits.iter().take(3) {
        out.push_str(&format!(
            "\n**{}** · score {}\n> {}\n",
            h["id"].as_str().unwrap_or("?"),
            h["score"].as_u64().unwrap_or(0),
            h["excerpt"].as_str().unwrap_or("")
        ));
    }
    ctx.say(clip_tail(&out, 1900)).await?;
    Ok(())
}

/// Call the gateway's lore skill and return the raw hits array.
pub async fn lore_search(
    cfg: &BotConfig,
    query: &str,
    top_k: u64,
) -> Result<Vec<serde_json::Value>, Error> {
    let url = format!("{}/v1/skills/lore/search", cfg.gateway_url);
    let resp = jerith_mesh_core::http::mesh_http()
        .post(&url)
        .timeout(std::time::Duration::from_secs(30))
        .json(&serde_json::json!({ "query": query, "top_k": top_k }))
        .send()
        .await
        .with_context(|| format!("gateway unreachable at {url}"))?;
    let status = resp.status();
    let body: serde_json::Value = resp.json().await.context("lore: non-JSON body")?;
    if !status.is_success() {
        anyhow::bail!(
            "lore search failed: {}",
            body.get("error").and_then(|v| v.as_str()).unwrap_or("?")
        );
    }
    Ok(body
        .get("hits")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default())
}

/// Show bot help.
#[poise::command(slash_command, prefix_command)]
pub async fn help(ctx: Ctx<'_>) -> Result<(), Error> {
    ctx.say(HELP_TEXT).await?;
    Ok(())
}

/// Guild-admin moderation lever — owner-DM only (default-deny gate).
// Deliberately NOT in owner.rs's USER_COMMANDS — the default-deny gate
// then enforces owner-DM only, which is the user's rule that the bot
// never executes system-level changes outside an owner DM. Because the
// invocation must be a DM, the guild and member arrive as explicit
// arguments instead of being inferred from the calling channel.
#[poise::command(slash_command, prefix_command)]
pub async fn timeout(
    ctx: Ctx<'_>,
    #[description = "Guild (server) id that holds the member"] guild_id: u64,
    #[description = "Member to time out (or to clear), as a user id"] member_id: u64,
    #[description = "Timeout length in minutes (1–40320; Discord caps at 28 days)"] minutes: Option<
        u64,
    >,
    #[description = "Reason recorded in the guild's audit log"] reason: Option<String>,
    #[description = "Clear the existing timeout instead of setting one"] clear: Option<bool>,
) -> Result<(), Error> {
    let clear = clear.unwrap_or(false);
    let edit = if clear {
        serenityall::builder::EditMember::new().enable_communication()
    } else {
        let minutes = minutes.context("minutes is required unless clear = true")?;
        validate_timeout_minutes(minutes)?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .context("system clock before the unix epoch")?
            .as_secs() as i64;
        let until = serenityall::model::Timestamp::from_unix_timestamp(now + (minutes as i64) * 60)
            .context("timeout end is not a valid timestamp")?;
        serenityall::builder::EditMember::new().disable_communication_until_datetime(until)
    };
    let reason = reason.unwrap_or_else(|| "jerith-mesh owner moderation".to_string());
    let edit = edit.audit_log_reason(&reason);
    let http = ctx.serenity_context().http.clone();
    let member = serenityall::GuildId::new(guild_id)
        .edit_member(&http, serenityall::UserId::new(member_id), edit)
        .await
        .context("timeout failed — check the bot has Moderate Members in that guild and the ids are right")?;
    let what = if clear { "cleared" } else { "set" };
    ctx.say(timeout_reply(
        what,
        member_id,
        guild_id,
        member.communication_disabled_until,
    ))
    .await?;
    Ok(())
}

/// Discord caps member timeouts at 28 days (40320 min); validate before
/// the call so the owner gets a clear refusal instead of an HTTP 400.
fn validate_timeout_minutes(minutes: u64) -> Result<(), Error> {
    anyhow::ensure!(
        (1..=28 * 24 * 60).contains(&minutes),
        "minutes must be between 1 and 40320 (Discord's 28-day timeout cap)"
    );
    Ok(())
}

/// Owner-facing confirmation line (split for the unit test).
fn timeout_reply(
    action: &str,
    member_id: u64,
    guild_id: u64,
    until: Option<serenityall::model::Timestamp>,
) -> String {
    match until {
        Some(t) => format!(
            "✅ Timeout {action} for <@{member_id}> in guild {guild_id} — active until <t:{}:f>.",
            t.unix_timestamp()
        ),
        None => format!("✅ Timeout {action} for <@{member_id}> in guild {guild_id}."),
    }
}

// ---------------------------------------------------------------------------
// Formatting (unit-tested)
// ---------------------------------------------------------------------------

pub const HELP_TEXT: &str = "\
**jerith-mesh bot**
`/ask <prompt>` — chat with the mesh (mesh-peer fallback included; Jerith can search the web / read pages itself when the node serves the web skills)
`/reset` — clear your conversation history
`/remember <text>` — store a note in the node's private memory
`/recall <query>` — search the node's private memory
`/briefing [hours]` — what happened across the watched threads (decisions + mesh digest, OpenClaw what-did-i-miss port)
`/status` — node + mesh status snapshot
`/peers` — list live peers in the mesh
`/mine` — try to mine JER on the local JerithChain node
`/leaderboard` — JER mining leaderboard (top winners)
`/myjercraft` — show your own JER balance on the chain
`/timeout <guild_id> <member_id> [minutes] [clear]` — guild moderation, **owner-DM only**
`/help` — this message

System-level actions are **owner-DM only**: commands outside this list are
refused for everyone except direct messages from the bot owner's account
(OpenClaw trust model, default-deny for unclassified commands).";

/// Append attribution when a mesh peer (not the local runtime) served it.
pub fn format_reply(reply: &str, served_by: Option<&str>) -> String {
    match served_by {
        Some(node) => format!("{reply}\n\n— served by mesh peer `{node}`"),
        None => reply.to_string(),
    }
}

/// /myjercraft rendering: named after the `/balance` contract (exists flag
/// drives the no-wallet hint; totals in uj-exact JER formatting).
pub fn format_my_balance(bal: &api::MyBalance, viewer: u64) -> String {
    if !bal.exists {
        return format!(
            "💳 <@{viewer}> — no JerithChain wallet yet. Run `/mine` once and one is created for you (mine rewards credit it automatically)."
        );
    }
    let jer = bal.balance_uj as f64 / 1_000_000.0;
    let jer_txt = if jer == (jer as u64) as f64 {
        format!("{}", jer as u64)
    } else {
        format!("{jer:.6}")
    };
    let addr = bal
        .address
        .as_deref()
        .map(short_addr)
        .unwrap_or_else(|| "unknown".to_string());
    let available = bal
        .available_jer
        .map(|a| {
            let a_txt = if a == (a as u64) as f64 {
                format!("{}", a as u64)
            } else {
                format!("{a:.6}")
            };
            format!(" · available **{a_txt} JER**")
        })
        .unwrap_or_default();
    let opted = match bal.opted_in {
        Some(true) => "",
        Some(false) => {
            "\n⚠️ you're opted out — mine rewards go to the pool owner until you opt in."
        }
        None => "",
    };
    format!("💳 <@{viewer}> — **{jer_txt} JER**{available}\naddress: `{addr}`{opted}")
}

/// `JERDC9695D27…4312`-style short form (first 12 + last 4).
fn short_addr(addr: &str) -> String {
    let n = addr.chars().count();
    if n <= 16 {
        return addr.to_string();
    }
    let head: String = addr.chars().take(12).collect();
    let tail: String = addr.chars().skip(n - 4).collect();
    format!("{head}…{tail}")
}

/// /leaderboard rendering: medal for top 3, `<@id>` mentions, mono totals.
pub fn format_leaderboard(rows: &[(u64, u64, u32)], viewer: u64, source: &str) -> String {
    if rows.is_empty() {
        return "🏆 no /mine wins recorded yet — be the first! `/mine`".to_string();
    }
    let medals = ["🥇", "🥈", "🥉"];
    let title = if source.is_empty() {
        "🏆 **JER mining leaderboard**".to_string()
    } else {
        format!("🏆 **JER mining leaderboard** — *{source}*")
    };
    let mut out = format!("{title}\n");
    for (i, (id, jer_uj, wins)) in rows.iter().enumerate() {
        let jer = *jer_uj as f64 / 1_000_000.0;
        let jer_txt = if jer == (jer as u64) as f64 {
            format!("{}", jer as u64)
        } else {
            format!("{jer:.6}")
        };
        let you = if *id == viewer { " (you)" } else { "" };
        let medal = medals.get(i).copied().unwrap_or("▫️");
        out.push_str(&format!(
            "{medal} <@{id}>{you} — **{jer_txt} JER** ({wins} win{})\n",
            if *wins == 1 { "" } else { "s" }
        ));
    }
    out
}

/// One line per hit: `• <text snippet> (score)`.
pub fn format_recall(query: &str, hits: &[serde_json::Value]) -> String {
    if hits.is_empty() {
        return format!("🔍 nothing in memory for “{query}”.");
    }
    let mut out = format!("🔍 **{query}**:\n");
    for (i, hit) in hits.iter().take(5).enumerate() {
        let text = hit
            .get("text")
            .and_then(|t| t.as_str())
            .map(|t| t.chars().take(120).collect::<String>())
            .unwrap_or_else(|| "<no text>".to_string());
        let score = hit
            .get("score")
            .map(|s| s.to_string())
            .unwrap_or_else(|| "?".to_string());
        out.push_str(&format!("{}. {text} ({score})\n", i + 1));
    }
    out.trim_end().to_string()
}

/// Compact status summary.
/// Token-less convenience wrapper — tests only; the live /status handler
/// calls format_status_with_tokens directly (it carries the caller's own
/// wallet token count).
#[cfg(test)]
pub fn format_status(st: &serde_json::Value) -> String {
    format_status_with_tokens(st, None)
}

/// `/status` rendering with the caller's cumulative token spend (perf+
/// session-cost-tracker port view). `tokens: None` keeps the older shape
/// (tests).
pub fn format_status_with_tokens(st: &serde_json::Value, tokens: Option<u64>) -> String {
    let get = |k: &str| st.get(k).cloned().unwrap_or(serde_json::Value::Null);
    let meter = |k: &str| {
        st.pointer(&format!("/meters/{k}"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    };
    // Metrics line (per-model aggregates from the gateway meters).
    let metrics_line = st
        .pointer("/metrics/models")
        .and_then(|v| v.as_object())
        .map(|models| {
            let mut parts: Vec<String> = models
                .iter()
                .take(3)
                .map(|(id, m)| {
                    format!(
                        "{}: {} req · {}% ok · {} tok · {}ms avg",
                        id,
                        m.get("requests").and_then(|v| v.as_u64()).unwrap_or(0),
                        m.get("ok_percent").and_then(|v| v.as_u64()).unwrap_or(0),
                        m.get("total_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
                        m.get("latency_ms_mean")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0),
                    )
                })
                .collect();
            parts.sort();
            parts.join(" · ")
        })
        .unwrap_or_default();
    let metrics_section = if metrics_line.is_empty() {
        String::new()
    } else {
        format!("\nmetrics: {metrics_line}")
    };
    let you_line = tokens
        .map(|t| format!("\nyour token spend (this bot, cumulative): {t}"))
        .unwrap_or_default();
    // Discovery rotation (Phase 1): a compact server-count line — the
    // last-good server is the one discovery actually rode last tick, so
    // a failover event is visible here without printing full URLs.
    let discovery_section = st
        .pointer("/discovery")
        .and_then(|d| d.as_object())
        .map(|d| {
            let configured = d
                .get("configured")
                .and_then(|v| v.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            let learned = d
                .get("learned")
                .and_then(|v| v.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            let last_good = d
                .get("last_good")
                .and_then(|v| v.as_str())
                .unwrap_or("none");
            format!(
                "\ndiscovery: {configured} configured · {learned} learned · last-good {last_good}"
            )
        })
        .unwrap_or_default();
    format!(
        "📊 **node** `{}` · role `{}` · model `{}` ({})\n\
         peers: {} total / {} alive · contribution: {}%\n\
         served: {} tokens · {} expert activations · {} bytes relayed\n\
         uptime: {}s · mines: {} attempted / {} won{}{}{}",
        get("node_id").as_str().unwrap_or("?"),
        get("role").as_str().unwrap_or("?"),
        get("model").as_str().unwrap_or("(none)"),
        get("quant").as_str().unwrap_or("?"),
        get("peer_count").as_u64().unwrap_or(0),
        get("peers_alive").as_u64().unwrap_or(0),
        get("contribution_percent").as_u64().unwrap_or(0),
        meter("tokens_served"),
        meter("expert_activations"),
        meter("bytes_relayed"),
        meter("uptime_seconds"),
        meter("mines_attempted"),
        st.pointer("/meters/mines_won")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        you_line,
        metrics_section,
        discovery_section,
    )
}

/// Peers summary from the /status snapshot.
pub fn format_peers(st: &serde_json::Value) -> String {
    let count = st.get("peer_count").and_then(|v| v.as_u64()).unwrap_or(0);
    let alive = st.get("peers_alive").and_then(|v| v.as_u64()).unwrap_or(0);
    format!("🕸 mesh peers: {alive}/{count} alive")
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Discord caps messages at 2000 chars; keep the tail (the answer's end).
async fn say_long(ctx: Ctx<'_>, text: &str) -> anyhow::Result<()> {
    const MAX: usize = 1990;
    let clipped: String = if text.chars().count() > MAX {
        let skip = text.chars().count() - MAX + 3;
        format!("…{}", text.chars().skip(skip).collect::<String>())
    } else {
        text.to_string()
    };
    ctx.say(clipped).await?;
    Ok(())
}

// Keep serenity_prelude referenced even if only poise macros use it.
#[allow(unused_imports)]
use serenityall as _serenity;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn timeout_minutes_capped_at_discords_28_days() {
        assert!(validate_timeout_minutes(1).is_ok());
        assert!(validate_timeout_minutes(40320).is_ok());
        assert!(validate_timeout_minutes(0).is_err());
        assert!(validate_timeout_minutes(40321).is_err());
    }

    #[test]
    fn timeout_reply_mentions_member_guild_and_expiry() {
        let t = serenityall::model::Timestamp::from_unix_timestamp(1_900_000_000).unwrap();
        let s = timeout_reply("set", 42, 1337, Some(t));
        assert!(s.contains("<@42>"));
        assert!(s.contains("guild 1337"));
        assert!(s.contains("1900000000"));
        let cleared = timeout_reply("cleared", 42, 1337, None);
        assert!(cleared.contains("cleared"));
        assert!(!cleared.contains("until"));
    }

    #[test]
    fn reply_attribution_line() {
        assert_eq!(format_reply("hello", None), "hello");
        assert_eq!(
            format_reply("hello", Some("node-abc")),
            "hello\n\n— served by mesh peer `node-abc`"
        );
    }

    #[test]
    fn recall_empty_and_hits() {
        assert_eq!(format_recall("q", &[]), "🔍 nothing in memory for “q”.");
        let hits = vec![json!({"text": "note about grids", "score": 0.87})];
        let out = format_recall("grids", &hits);
        assert!(out.contains("note about grids"));
        assert!(out.contains("0.87"));
    }

    #[test]
    fn recall_truncates_long_text() {
        let hits = vec![json!({"text": "x".repeat(500), "score": 1})];
        let out = format_recall("q", &hits);
        assert!(out.chars().count() < 200);
    }

    #[test]
    fn status_formats_all_sections() {
        let st = json!({
            "node_id": "n1", "role": "hybrid", "model": "glm-5.3", "quant": "q4",
            "peer_count": 3, "peers_alive": 2, "contribution_percent": 40,
            "meters": {
                "tokens_served": 11, "expert_activations": 5,
                "bytes_relayed": 999, "uptime_seconds": 60,
                "mines_attempted": 2, "mines_won": 1,
            }
        });
        let out = format_status(&st);
        assert!(out.contains("`n1`"));
        assert!(out.contains("3 total / 2 alive"));
        assert!(out.contains("11 tokens"));
        assert!(out.contains("mines: 2 attempted / 1 won"));
        // No discovery block in the snapshot → no discovery line at all.
        assert!(!out.contains("discovery:"));
    }

    #[test]
    fn status_discovery_line_counts_and_last_good() {
        let st = json!({
            "node_id": "n1", "role": "hybrid", "model": "glm-5.3", "quant": "q4",
            "peer_count": 1, "peers_alive": 1, "contribution_percent": 10,
            "meters": { "tokens_served": 1, "uptime_seconds": 1 },
            "discovery": {
                "configured": ["https://primary.example.org"],
                "learned": ["https://a.example.org", "https://b.example.org"],
                "last_good": "https://a.example.org",
            }
        });
        let out = format_status(&st);
        assert!(
            out.contains("discovery: 1 configured · 2 learned · last-good https://a.example.org")
        );
    }

    #[test]
    fn leaderboard_empty_and_medals() {
        assert!(format_leaderboard(&[], 1, "").contains("no /mine wins"));
        let rows = vec![
            (10u64, 5_000_000u64, 1u32),
            (20, 15_000_000, 3),
            (30, 2_500_000, 2),
            (40, 1_000_000, 1),
        ];
        let out = format_leaderboard(&rows, 20, "chain totals (lifetime)");
        // Input order preserved (docs: sort caller-side); medal per position.
        assert!(out.contains("*chain totals (lifetime)*"));
        assert!(out.contains("🥇 <@10>"));
        assert!(out.contains("🥈 <@20>") && out.contains("(you)"));
        assert!(out.contains("🥉 <@30>"));
        assert!(out.contains("▫️ <@40>"));
        assert!(out.contains("**15 JER** (3 wins)"));
        assert!(out.contains("**2.500000 JER** (2 wins)"));
        assert!(out.contains("**5 JER** (1 win)"));
        assert!(out.contains("**1 JER** (1 win)"));
    }

    #[test]
    fn my_balance_formats() {
        let bal = crate::api::MyBalance {
            exists: false,
            balance_uj: 0,
            address: None,
            available_jer: None,
            opted_in: None,
        };
        assert!(format_my_balance(&bal, 5).contains("no JerithChain wallet"));

        let bal = crate::api::MyBalance {
            exists: true,
            balance_uj: 50_000_000,
            address: Some("JERDC9695D27A54B2130A5CF25FC97A4312".to_string()),
            available_jer: Some(50.0),
            opted_in: Some(true),
        };
        let out = format_my_balance(&bal, 5);
        assert!(out.contains("**50 JER**"));
        assert!(out.contains("available **50 JER**"));
        assert!(out.contains("JERDC9695D27…4312"));
        assert!(!out.contains("opted out"));

        let bal = crate::api::MyBalance {
            exists: true,
            balance_uj: 2_500_000,
            address: Some("JERDAB".to_string()),
            available_jer: Some(2.25),
            opted_in: Some(false),
        };
        let out = format_my_balance(&bal, 9);
        assert!(out.contains("**2.500000 JER**"));
        assert!(out.contains("available **2.250000 JER**"));
        assert!(out.contains("opted out"));
        // short addresses pass through
        assert!(out.contains("`JERDAB`"));
    }

    #[test]
    fn peers_summary() {
        assert_eq!(
            format_peers(&json!({"peer_count": 4, "peers_alive": 4})),
            "🕸 mesh peers: 4/4 alive"
        );
    }
}
