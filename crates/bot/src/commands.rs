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
    let user = user_id(&ctx);
    if let Err(msg) = ctx.data().state.gate_mine(user) {
        ctx.say(msg).await?;
        return Ok(());
    }
    ctx.defer().await?;
    let channel = ctx.channel_id().get().to_string();
    let out = api::mine(&ctx.data().cfg, &user.to_string(), &channel).await?;
    ctx.say(out).await?;
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
`/help` — this message";

/// Append attribution when a mesh peer (not the local runtime) served it.
pub fn format_reply(reply: &str, served_by: Option<&str>) -> String {
    match served_by {
        Some(node) => format!("{reply}\n\n— served by mesh peer `{node}`"),
        None => reply.to_string(),
    }
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
    fn peers_summary() {
        assert_eq!(
            format_peers(&json!({"peer_count": 4, "peers_alive": 4})),
            "🕸 mesh peers: 4/4 alive"
        );
    }
}
