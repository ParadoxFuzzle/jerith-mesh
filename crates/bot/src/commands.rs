//! Slash commands. Thin wrappers: gate cooldowns → call `api::*` → format a
//! reply. Formatting is split from IO so tests can cover it without a
//! Discord connection.

use poise::serenity_prelude as serenityall;

use crate::api;
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

    let history = ctx.data().state.history(user);
    let cfg = ctx.data().cfg.clone();
    let (reply, served_by) = api::chat(&cfg, &prompt, &history).await?;
    ctx.data().state.push_turn(user, &prompt, &reply);
    let text = format_reply(&reply, served_by.as_deref());
    say_long(ctx, &text).await?;
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
    ctx.say(format_status(&st)).await?;
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
    if outcome.won {
        if let Some(uj) = outcome.reward_uj {
            ctx.data().state.record_mine_win(discord_id, uj);
        }
    }
    // Optional: announce wins in a configured channel (fire-and-forget; a
    // missing/incorrect channel id must not fail the user's command).
    if let Some(announce_id) = ctx.data().cfg.announce_channel_id {
        if let Some(text) = api::mine_announcement(&outcome, &user) {
            let http = ctx.serenity_context().http.clone();
            tokio::spawn(async move {
                let ch = serenityall::ChannelId::new(announce_id);
                if let Err(e) = ch.say(&http, &text).await {
                    tracing::warn!(error = %e, channel = announce_id, "mine announcement failed");
                }
            });
        }
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
                .map(|e| (e.discord_id, e.total_mined_uj, e.wins.min(u32::MAX as u64) as u32))
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

/// Show bot help.
#[poise::command(slash_command, prefix_command)]
pub async fn help(ctx: Ctx<'_>) -> Result<(), Error> {
    ctx.say(HELP_TEXT).await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Formatting (unit-tested)
// ---------------------------------------------------------------------------

pub const HELP_TEXT: &str = "\
**jerith-mesh bot**
`/ask <prompt>` — chat with the mesh (mesh-peer fallback included)
`/reset` — clear your conversation history
`/remember <text>` — store a note in the node's private memory
`/recall <query>` — search the node's private memory
`/status` — node + mesh status snapshot
`/peers` — list live peers in the mesh
`/mine` — try to mine JER on the local JerithChain node
`/leaderboard` — JER mining leaderboard (top winners)
`/help` — this message";

/// Append attribution when a mesh peer (not the local runtime) served it.
pub fn format_reply(reply: &str, served_by: Option<&str>) -> String {
    match served_by {
        Some(node) => format!("{reply}\n\n— served by mesh peer `{node}`"),
        None => reply.to_string(),
    }
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
pub fn format_status(st: &serde_json::Value) -> String {
    let get = |k: &str| st.get(k).cloned().unwrap_or(serde_json::Value::Null);
    let meter = |k: &str| {
        st.pointer(&format!("/meters/{k}"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    };
    format!(
        "📊 **node** `{}` · role `{}` · model `{}` ({})\n\
         peers: {} total / {} alive · contribution: {}%\n\
         served: {} tokens · {} expert activations · {} bytes relayed\n\
         uptime: {}s · mines: {} attempted / {} won",
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
    }

    #[test]
    fn leaderboard_empty_and_medals() {
        assert!(format_leaderboard(&[], 1, "").contains("no /mine wins"));
        let rows = vec![(10u64, 5_000_000u64, 1u32), (20, 15_000_000, 3), (30, 2_500_000, 2), (40, 1_000_000, 1)];
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
    fn peers_summary() {
        assert_eq!(
            format_peers(&json!({"peer_count": 4, "peers_alive": 4})),
            "🕸 mesh peers: 4/4 alive"
        );
    }
}
