//! Gateway + chain HTTP calls the slash commands make.
//!
//! Kept as plain async functions (no poise types) so they are unit-testable
//! and reusable. Everything talks OpenAI-ish shapes to the gateway.

use anyhow::{Context as _, Result};
use serde_json::{json, Value};

const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(90);

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(TIMEOUT)
        .build()
        .expect("static reqwest client config")
}

/// POST /v1/chat/completions on the gateway. Returns the assistant text plus
/// the `mesh_fallback` node id when the mesh (not the local runtime) served it.
pub async fn chat(
    cfg: &crate::config::BotConfig,
    prompt: &str,
    history: &[Value],
) -> Result<(String, Option<String>)> {
    let mut messages = history.to_vec();
    messages.push(json!({"role": "user", "content": prompt}));

    let url = format!("{}/v1/chat/completions", cfg.gateway_url);
    let resp = client()
        .post(&url)
        .json(&json!({
            "model": "jerith-mesh",
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
}
