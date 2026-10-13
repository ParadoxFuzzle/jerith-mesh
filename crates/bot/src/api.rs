//! Gateway + chain HTTP calls the slash commands make.
//!
//! Kept as plain async functions (no poise types) so they are unit-testable
//! and reusable. Everything talks OpenAI-ish shapes to the gateway.

use anyhow::{Context as _, Result};
use serde_json::{json, Value};

const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(90);

/// Legacy placeholder some code paths still carry — only used if the
/// gateway's /status could not tell us the real runtime model.
const MODEL_PLACEHOLDER: &str = "jerith-mesh";

/// Resolved `(gateway_url, runtime_model)` pair, fetched once per process
/// (the bot's config is fixed at startup, so the URL never changes under
/// us). Sending the placeholder model verbatim to an Ollama/Colibri runtime
/// gets `model 'jerith-mesh' not found` — the node actually serves whatever
/// its config.advertised model is, exposed on /status as `model`.
static RESOLVED_MODEL: tokio::sync::OnceCell<(String, String)> =
    tokio::sync::OnceCell::const_new();

/// The model id chat requests should carry: the node's advertised runtime
/// model, or `None` when /status is unreachable (caller falls back to the
/// placeholder and lets the gateway error surface verbatim).
async fn runtime_model(cfg: &crate::config::BotConfig) -> Option<String> {
    let (url, model) = RESOLVED_MODEL
        .get_or_try_init(|| async {
            let url = format!("{}/status", cfg.gateway_url.trim_end_matches('/'));
            let resp = client()
                .get(&url)
                .send()
                .await
                .with_context(|| format!("/model status lookup failed at {url}"))?;
            let body: Value = resp.json().await.context("model status: non-JSON body")?;
            let model = model_from_status(&body)
                .with_context(|| format!("model status: no usable model field at {url}"))?;
            Ok::<(String, String), anyhow::Error>((cfg.gateway_url.clone(), model))
        })
        .await
        .ok()?;
    debug_assert_eq!(url, &cfg.gateway_url, "gateway url changed within process");
    Some(model.clone())
}

/// Pull the advertised model out of a gateway /status body: a non-empty
/// string `model` field (e.g. `{ "model": "qwen3:8b", ... }`).
fn model_from_status(body: &Value) -> Option<String> {
    let model = body.get("model")?.as_str()?.trim();
    (!model.is_empty()).then(|| model.to_string())
}

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(TIMEOUT)
        .build()
        .expect("static reqwest client config")
}

/// POST /v1/chat/completions on the gateway. Returns the assistant text plus
/// the `mesh_fallback` node id when the mesh (not the local runtime) served it.
/// Message assembly for every chat shape the bot produces (`/ask`, DM and
/// @mention fallbacks share this path). A persona system prompt goes first,
/// then the conversation history, then the caller's prompt. Pure fn — tests
/// cover ordering without a network or framework.
pub fn build_messages(
    history: &[Value],
    prompt: &str,
    persona: Option<&str>,
) -> Vec<Value> {
    let mut messages: Vec<Value> = Vec::with_capacity(history.len() + 2);
    if let Some(p) = persona.map(str::trim).filter(|p| !p.is_empty()) {
        messages.push(json!({"role": "system", "content": p}));
    }
    messages.extend(history.iter().cloned());
    messages.push(json!({"role": "user", "content": prompt}));
    messages
}

pub async fn chat(
    cfg: &crate::config::BotConfig,
    prompt: &str,
    history: &[Value],
) -> Result<(String, Option<String>)> {
    // Guard screening covers user-side content only: history + prompt. The
    // persona system prompt is operator-owned static config and is attached
    // AFTER screening — the Guard's PII redaction could otherwise mangle
    // the owner's own security rules (e.g. IDs defined in bot.toml / SOUL)
    // that the persona quotes verbatim.
    let mut messages = build_messages(history, prompt, None);

    // Jerith Guard: screen OUTBOUND content before any provider call
    // (defense in depth — the gateway screens too, but the bot token can
    // point at any operator-chosen gateway URL, and the reply in Discord
    // should cite the blocking rule directly without a round-trip).
    // PIIs (emails/phones) are redacted in place; SSNs/cards/credentials
    // and injection / malware-shape prompts hard-block.

    // Jerith Guard: screen OUTBOUND content before any provider call
    // (defense in depth — the gateway screens too, but the bot token can
    // point at any operator-chosen gateway URL, and the reply in Discord
    // should cite the blocking rule directly without a round-trip).
    // PIIs (emails/phones) are redacted in place; SSNs/cards/credentials
    // and injection / malware-shape prompts hard-block.
    if cfg.guard.enabled {
        match jerith_mesh_core::guard::Guard::build(&cfg.guard) {
            Ok(guard) => match guard.screen_messages(&messages) {
                Ok(Some(patched)) => messages = patched,
                Ok(None) => {}
                Err(verdict) => anyhow::bail!(
                    "🛡️ Jerith Guard blocked this request ({}). Edit the prompt and try again.",
                    verdict.block_reason()
                ),
            },
            Err(e) => {
                // A malformed [guard] in bot.toml must degrade to the
                // gateway's own screening, never to a failed bot.
                tracing::warn!(error = %e, "bot guard config invalid — screening skipped for this call");
            }
        }
    }

    // Persona goes first, after user-side screening (see above). The bot
    // answering "as Jerith" is exactly this message.
    let persona_first = cfg
        .system_prompt
        .as_deref()
        .map(|p| !p.trim().is_empty())
        .unwrap_or(false);
    if persona_first {
        messages.insert(
            0,
            json!({"role": "system", "content": cfg.system_prompt.clone().unwrap_or_default()}),
        );
    }

    // Live identity: resolve the tagged speaker against the jerith_people
    // registry (direct Postgres — same contract as the OpenClaw stack:
    // root@127.0.0.1:5432, password via ~/.pgpass). The freshest row is
    // attached as a second system message; an unreachable database or an
    // unknown speaker just means the chat proceeds without it.
    if let Some(pg) = cfg.postgres.as_ref().filter(|p| p.enabled) {
        if let Some(discord_id) = crate::people::extract_speaker_id(&prompt) {
            match crate::people::lookup(pg, &discord_id).await {
                Ok(Some(block)) => messages.insert(
                    if persona_first { 1 } else { 0 },
                    json!({"role": "system", "content": block}),
                ),
                Ok(None) => {}
                Err(e) => {
                    tracing::debug!(error = %e, "people-db lookup skipped — chatting without speaker context");
                }
            }
        }
    }

    let url = format!("{}/v1/chat/completions", cfg.gateway_url);
    let model = runtime_model(cfg).await.unwrap_or_else(|| MODEL_PLACEHOLDER.to_string());
    let resp = client()
        .post(&url)
        .json(&json!({
            "model": model,
            "messages": messages,
        }))
        .send()
        .await
        .with_context(|| format!("gateway unreachable at {url}"))?;

    let status = resp.status();
    let body: Value = resp.json().await.context("gateway returned non-JSON")?;
    if !status.is_success() {
        let err = body
            .get("error")
            .and_then(value_to_string)
            .unwrap_or_else(|| status.to_string());
        anyhow::bail!("gateway error {status}: {err}");
    }
    let content = body
        .pointer("/choices/0/message/content")
        .and_then(value_to_string)
        .unwrap_or_default();
    let served_by = body
        .get("mesh_fallback")
        .and_then(value_to_string)
        .filter(|s| !s.is_empty());
    Ok((content, served_by))
}

/// POST /v1/skills/memory/embed — store text in the node's private memory.
pub async fn memory_embed(cfg: &crate::config::BotConfig, text: &str) -> Result<(String, String)> {
    let url = format!("{}/v1/skills/memory/embed", cfg.gateway_url);
    let resp = client()
        .post(&url)
        .json(&json!({ "text": text }))
        .send()
        .await
        .with_context(|| format!("gateway unreachable at {url}"))?;
    let status = resp.status();
    let body: Value = resp.json().await.context("gateway returned non-JSON")?;
    if !status.is_success() {
        anyhow::bail!("memory.embed failed: {}", body_error(&body, status));
    }
    let id = body
        .get("id")
        .and_then(value_to_string)
        .unwrap_or_else(|| "unknown".to_string());
    let provider = body
        .get("provider")
        .and_then(value_to_string)
        .unwrap_or_else(|| "unknown".to_string());
    Ok((id, provider))
}

/// POST /v1/skills/memory/search — vector search over the node's memory.
pub async fn memory_search(
    cfg: &crate::config::BotConfig,
    query: &str,
    top_k: u64,
) -> Result<Vec<Value>> {
    let url = format!("{}/v1/skills/memory/search", cfg.gateway_url);
    let resp = client()
        .post(&url)
        .json(&json!({ "query": query, "top_k": top_k }))
        .send()
        .await
        .with_context(|| format!("gateway unreachable at {url}"))?;
    let status = resp.status();
    let body: Value = resp.json().await.context("gateway returned non-JSON")?;
    if !status.is_success() {
        anyhow::bail!("memory.search failed: {}", body_error(&body, status));
    }
    Ok(body
        .get("hits")
        .and_then(|h| h.as_array())
        .cloned()
        .unwrap_or_default())
}

/// GET /status on the gateway — node snapshot for /status and /peers.
pub async fn gateway_status(cfg: &crate::config::BotConfig) -> Result<Value> {
    let url = format!("{}/status", cfg.gateway_url);
    let resp = client()
        .get(&url)
        .send()
        .await
        .with_context(|| format!("gateway unreachable at {url}"))?;
    let status = resp.status();
    let body: Value = resp.json().await.context("gateway returned non-JSON")?;
    if !status.is_success() {
        anyhow::bail!("status failed: {}", body_error(&body, status));
    }
    Ok(body)
}

/// Outcome of a /mine attempt — enough to render both the direct reply and
/// an optional announcement post.
#[derive(Debug)]
pub struct MineOutcome {
    pub won: bool,
    pub message: String,
    pub amount: Option<String>,
    /// Reward in micro-JER as reported by the chain node (`reward_uj`), when
    /// it supplies it — used to credit the bot-side /leaderboard.
    pub reward_uj: Option<u64>,
}

/// POST /mine on the local JerithChain node (bearer-token auth via
/// keys/api.token or JERITH_TOKEN, same as the passive miner).
pub async fn mine(
    cfg: &crate::config::BotConfig,
    discord_id: &str,
    channel_id: &str,
) -> Result<MineOutcome> {
    let token = crate::core_token()?;
    let url = format!("{}/mine", cfg.chain_url);
    let message_id = format!("discord-bot:{discord_id}:{}", now_ms());
    let resp = client()
        .post(&url)
        .bearer_auth(token)
        .json(&json!({
            "discord_id": discord_id,
            "source": "discord",
            "message_id": message_id,
            "guild_id": "",
            "channel_id": channel_id,
        }))
        .send()
        .await
        .with_context(|| format!("JerithChain node unreachable at {url}"))?;
    let status = resp.status();
    let body: Value = resp.json().await.context("chain returned non-JSON")?;
    if !status.is_success() {
        anyhow::bail!("mine failed: {}", body_error(&body, status));
    }
    let won = body.get("ok").and_then(|o| o.as_bool()).unwrap_or(false);
    if won {
        // Prefer the exact reward_uj; fall back to parsing amount strings.
        let reward_uj = body
            .get("reward_uj")
            .and_then(|v| v.as_u64())
            .or_else(|| {
                body.get("reward_jer")
                    .and_then(|v| v.as_f64())
                    .map(|f| (f * 1_000_000.0) as u64)
            });
        let amount = reward_uj
            .map(|u| {
                let jer = u as f64 / 1_000_000.0;
                if jer == (jer as u64) as f64 {
                    format!("{} JER", jer as u64)
                } else {
                    format!("{jer:.6} JER")
                }
            })
            .or_else(|| {
                body.get("amount")
                    .and_then(value_to_string)
                    .filter(|a| !a.is_empty())
            });
        let message = match &amount {
            Some(a) => format!("⛏ mined {a}! 🎉"),
            None => "⛏ mined JER! 🎉".to_string(),
        };
        Ok(MineOutcome {
            won,
            message,
            amount,
            reward_uj,
        })
    } else {
        let reason = body
            .get("reason")
            .and_then(value_to_string)
            .unwrap_or_else(|| "not this time".to_string());
        Ok(MineOutcome {
            won,
            message: format!("⛏ mine attempt: {reason}"),
            amount: None,
            reward_uj: None,
        })
    }
}

/// One chain /mystate row (top user by total mined).
#[derive(Debug, Clone, PartialEq)]
pub struct ChainLbEntry {
    pub discord_id: u64,
    pub wins: u64,
    pub total_mined_uj: u64,
}

/// Chain /balance for one Discord user.
#[derive(Debug, Clone, PartialEq)]
pub struct MyBalance {
    pub exists: bool,
    pub balance_uj: u64,
    pub address: Option<String>,
    pub available_jer: Option<f64>,
    pub opted_in: Option<bool>,
}

/// GET /balance on the chain node for the caller's Discord id.
pub async fn my_balance(cfg: &crate::config::BotConfig, token: &str, discord_id: u64) -> Result<MyBalance> {
    let url = format!("{}/balance?discord_id={}", cfg.chain_url, discord_id);
    let resp = client()
        .get(&url)
        .bearer_auth(token)
        .send()
        .await
        .with_context(|| format!("chain node unreachable at {url}"))?;
    let status = resp.status();
    let body: Value = resp.json().await.context("chain returned non-JSON")?;
    if !status.is_success() {
        anyhow::bail!("balance failed: {}", body_error(&body, status));
    }
    let exists = body.get("exists").and_then(|v| v.as_bool()).unwrap_or(false);
    Ok(MyBalance {
        exists,
        balance_uj: body.get("balance_uj").and_then(|v| v.as_u64()).unwrap_or(0),
        address: body
            .get("address")
            .and_then(value_to_string),
        available_jer: body.get("available_jer").and_then(|v| v.as_f64()),
        opted_in: body.get("opted_in").and_then(|v| v.as_bool()),
    })
}

/// GET /mystate on the chain node — chain-authoritative leaderboard totals
/// (per-user lifetime, includes passive mining the bot never observed).
pub async fn chain_leaderboard(
    cfg: &crate::config::BotConfig,
    token: Option<String>,
    top: usize,
) -> Result<Vec<ChainLbEntry>> {
    let token = match token {
        Some(t) => t,
        None => return Err(anyhow::anyhow!("no chain api token")),
    };
    let url = format!("{}/mystate?limit={}", cfg.chain_url, top.min(200));
    let resp = client()
        .get(&url)
        .bearer_auth(token)
        .send()
        .await
        .with_context(|| format!("chain node unreachable at {url}"))?;
    let status = resp.status();
    let body: Value = resp.json().await.context("chain returned non-JSON")?;
    if !status.is_success() {
        anyhow::bail!("mystate failed: {}", body_error(&body, status));
    }
    let mut out = Vec::new();
    for e in body
        .get("entries")
        .and_then(|x| x.as_array())
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .take(top)
    {
        // Pass the raw string through; a non-numeric discord id just never
        // matches any Discord user later.
        let did: u64 = e
            .get("discord_id")
            .and_then(|v| v.as_str().map(|s| s.to_string()).or_else(|| v.as_u64().map(|n| n.to_string())))
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        if did == 0 {
            continue;
        }
        out.push(ChainLbEntry {
            discord_id: did,
            wins: e.get("wins").and_then(|v| v.as_u64()).unwrap_or(0),
            total_mined_uj: e.get("total_mined_uj").and_then(|v| v.as_u64()).unwrap_or(0),
        });
    }
    Ok(out)
}

/// Announcement text for a win in the configured channel, or `None` when not
/// a win. Formatted here so both the reply and the announcement stay in sync.
pub fn mine_announcement(outcome: &MineOutcome, user: &str) -> Option<String> {
    if !outcome.won {
        return None;
    }
    let amount = outcome
        .amount
        .as_deref()
        .map(|a| format!(" ({a})"))
        .unwrap_or_default();
    Some(format!("⛏ **{user}** mined JER{amount}! 🎉"))
}

fn body_error(body: &Value, status: reqwest::StatusCode) -> String {
    body.get("error")
        .and_then(value_to_string)
        .unwrap_or_else(|| status.to_string())
}

fn value_to_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Null => None,
        other => Some(other.to_string()),
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_messages_places_persona_first_without_dupes() {
        let history = vec![
            json!({"role": "user", "content": "hi"}),
            json!({"role": "assistant", "content": "hey"}),
        ];
        let out = build_messages(&history, "who are you?", Some("You are Jerith.\n"));
        assert_eq!(out.len(), 4);
        assert_eq!(out[0]["role"], json!("system"));
        assert_eq!(out[0]["content"], json!("You are Jerith."));
        assert_eq!(out[1]["role"], json!("user"));
        assert_eq!(out[2]["role"], json!("assistant"));
        assert_eq!(out[3]["role"], json!("user"));
        assert_eq!(out[3]["content"], json!("who are you?"));
    }

    #[test]
    fn build_messages_without_persona_skips_system() {
        let out = build_messages(&[], "asf", None);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["role"], json!("user"));
        // Whitespace-only persona is treated as absent.
        let out = build_messages(&[], "asf", Some("   "));
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn model_resolution_pulls_advertised_model() {
        assert_eq!(
            model_from_status(&json!({"model": "qwen3:8b", "node_id": "n"})),
            Some("qwen3:8b".to_string())
        );
        assert_eq!(model_from_status(&json!({"model": "  "})), None);
        assert_eq!(model_from_status(&json!({})), None);
        assert_eq!(model_from_status(&json!({"model": 42})), None);
    }

    #[test]
    fn my_balance_parsing_shapes() {
        // The render side lives in commands.rs; here pin the field extraction
        // from the two documented /balance response shapes.
        let existing = serde_json::json!({
            "exists": true, "address": "JERDC9695",
            "balance_uj": 50_000_000u64, "balance_jer": 50.0,
            "available_jer": 49.9, "opted_in": true
        });
        let got = MyBalance {
            exists: existing.get("exists").and_then(|v| v.as_bool()).unwrap_or(false),
            balance_uj: existing.get("balance_uj").and_then(|v| v.as_u64()).unwrap_or(0),
            address: existing.get("address").and_then(value_to_string),
            available_jer: existing.get("available_jer").and_then(|v| v.as_f64()),
            opted_in: existing.get("opted_in").and_then(|v| v.as_bool()),
        };
        assert!(got.exists);
        assert_eq!(got.balance_uj, 50_000_000);
        assert_eq!(got.address.as_deref(), Some("JERDC9695"));

        let none = serde_json::json!({"exists": false, "balance_uj": 0, "balance_jer": 0.0});
        let got2 = MyBalance {
            exists: none.get("exists").and_then(|v| v.as_bool()).unwrap_or(false),
            balance_uj: none.get("balance_uj").and_then(|v| v.as_u64()).unwrap_or(0),
            address: none.get("address").and_then(value_to_string),
            available_jer: none.get("available_jer").and_then(|v| v.as_f64()),
            opted_in: none.get("opted_in").and_then(|v| v.as_bool()),
        };
        assert!(!got2.exists);
        assert!(got2.address.is_none());
    }

    #[tokio::test]
    async fn my_balance_unreachable_is_err() {
        let cfg = crate::config::BotConfig::from_parts("http://127.0.0.1:3", "http://127.0.0.1:4");
        assert!(my_balance(&cfg, "t", 12345).await.is_err());
    }

    #[tokio::test]
    async fn chain_leaderboard_unreachable_is_err() {
        let cfg = crate::config::BotConfig::from_parts("http://127.0.0.1:3", "http://127.0.0.1:4");
        let res = chain_leaderboard(&cfg, Some("tok".to_string()), 10).await;
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn chain_leaderboard_requires_token() {
        let cfg = crate::config::BotConfig::from_parts("http://127.0.0.1:3", "http://127.0.0.1:4");
        let res = chain_leaderboard(&cfg, None, 10).await;
        assert!(res.is_err());
    }

    #[test]
    fn mine_announcement_only_for_wins() {
        let win = MineOutcome {
            won: true,
            message: "⛏ mined JER (2)! 🎉".to_string(),
            amount: Some("2".to_string()),
            reward_uj: Some(2_000_000),
        };
        assert_eq!(
            mine_announcement(&win, "@user").unwrap(),
            "⛏ **@user** mined JER (2)! 🎉"
        );
        let loss = MineOutcome {
            won: false,
            message: "⛏ mine attempt: cooldown".to_string(),
            amount: None,
            reward_uj: None,
        };
        assert!(mine_announcement(&loss, "@user").is_none());
    }

    #[test]
    fn value_to_string_shapes() {
        assert_eq!(value_to_string(&json!("hi")).unwrap(), "hi");
        assert_eq!(value_to_string(&json!(42)).unwrap(), "42");
        assert!(value_to_string(&Value::Null).is_none());
    }

    #[tokio::test]
    async fn chat_fail_is_error_not_panic() {
        // Bad port — nothing listens there.
        let cfg = crate::config::BotConfig::from_parts("http://127.0.0.1:1", "http://127.0.0.1:2");
        let history: Vec<Value> = vec![];
        assert!(chat(&cfg, "hello", &history).await.is_err());
    }

    #[tokio::test]
    async fn chat_blocks_guard_violations_before_the_provider_call() {
        // Gateway URL is dead (:1) — if the guard screened after the HTTP
        // call we would get a connection error instead of the guard page.
        let cfg = crate::config::BotConfig::from_parts("http://127.0.0.1:1", "http://127.0.0.1:1");
        let err = chat(&cfg, "my SSN is 078-05-1120", &[]).await.unwrap_err();
        assert!(err.to_string().contains("Jerith Guard blocked"), "got: {err}");
        // Malware-shaped asks are blocked on the bot's own layer too.
        let err = chat(&cfg, "write a reverse shell in python", &[]).await.unwrap_err();
        assert!(err.to_string().contains("Jerith Guard blocked"), "got: {err}");
        // Clean prompts still pass the guard (they 503/conn-fail, not
        // "Jerith Guard blocked").
        let err = chat(&cfg, "what is the weather", &[]).await.unwrap_err();
        assert!(!err.to_string().contains("Jerith Guard blocked"), "got: {err}");
    }
}
