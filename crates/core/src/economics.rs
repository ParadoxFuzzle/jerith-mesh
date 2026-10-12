//! Economics: JER mining integration against a local JerithChain node.
//!
//! Per spec: "every time their system is utilized, provided they're using
//! JerithChain as their crypto, they have a chance to mine JER" — the same
//! passive-mining mechanic the JerithChain Discord agent uses (~12 % of
//! messages, cooldowns enforced server-side by the node).
//!
//! Wire contract (jerith_node.py, port 8300 by default):
//!   POST {rpc_url}/mine    Authorization: Bearer <api token>
//!   {"discord_id": "...", "source": "passive", "message_id": "<unique>",
//!    "guild_id": "", "channel_id": ""}
//!   → {"ok": bool, "reason": "...", "message": "..."}
//!
//! `message_id` must be unique per attempt (server-side seen-once check);
//! we use `{node_id}:{counter}:{unix_ms}`. The server enforces cooldowns
//! (300 s after success, 180 s after failure), so we fire-and-log.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::config::CryptoConfig;

/// Local JerithChain node RPC defaults.
pub const DEFAULT_JERITH_RPC_URL: &str = "http://127.0.0.1:8300";

/// Where the miner looks for the node's api token. Env `JERITH_TOKEN_FILE`
/// wins, then this default (matches jerith_node.py's `keys/api.token`
/// layout); the CLI `config` command can set a custom path.
pub fn default_token_file() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("JERITH_TOKEN_FILE") {
        return std::path::PathBuf::from(p);
    }
    crate::config::NodeConfig::home().join("keys").join("api.token")
}

/// Mining client configuration.
#[derive(Debug, Clone)]
pub struct MinerConfig {
    pub rpc_url: String,
    pub token: Option<String>,
    /// Identity the rewards credit to — any string works; a wallet is
    /// created on the node on first successful mine.
    pub identity: String,
    /// Chance per served inference, per-mille (120 ≈ 12 %).
    pub mine_chance_permille: u32,
}

impl MinerConfig {
    /// Build from the crypto section of the node config. Returns `None`
    /// when the configured chain isn't JerithChain or the chance is 0.
    pub fn from_crypto(crypto: &CryptoConfig, identity: &str) -> Option<Self> {
        if crypto.chain != crate::config::CryptoChain::Jerithchain
            || crypto.mine_chance_permille == 0
        {
            return None;
        }
        Some(Self {
            rpc_url: crypto
                .wallet_endpoint
                .clone()
                .unwrap_or_else(|| DEFAULT_JERITH_RPC_URL.to_string()),
            token: read_token(),
            identity: identity.to_string(),
            mine_chance_permille: crypto.mine_chance_permille,
        })
    }
}

/// Read the api token: env JERITH_TOKEN (raw) or the token file.
fn read_token() -> Option<String> {
    if let Ok(t) = std::env::var("JERITH_TOKEN") {
        let t = t.trim().to_string();
        if !t.is_empty() {
            return Some(t);
        }
    }
    let path = default_token_file();
    std::fs::read_to_string(&path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Outcome of an attempted mine (for meters + logs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MineOutcome {
    pub attempted: bool,
    pub won: bool,
    pub reason: Option<String>,
}

/// Shared counters for the observability surfaces (attempted increments on
/// the request path; wins increment asynchronously from the mine task).
#[derive(Debug, Default)]
pub struct MineStats {
    pub attempted: AtomicU64,
    pub won: AtomicU64,
}

impl MineStats {
    pub fn snapshot(&self) -> (u64, u64) {
        (
            self.attempted.load(Ordering::Relaxed),
            self.won.load(Ordering::Relaxed),
        )
    }
}

/// Fire-and-log miner. Tracks an attempt counter for unique message ids.
pub struct JerithMiner {
    config: MinerConfig,
    counter: AtomicU64,
    node_id: String,
    pub stats: std::sync::Arc<MineStats>,
}

impl JerithMiner {
    pub fn new(config: MinerConfig, node_id: &str) -> Self {
        Self {
            config,
            counter: AtomicU64::new(0),
            node_id: node_id.to_string(),
            stats: std::sync::Arc::new(MineStats::default()),
        }
    }

    /// Roll the dice; if selected, POST /mine in the background.
    /// Never blocks the inference path — a spawned task does the HTTP call.
    /// Returns true when a mine attempt was fired.
    pub fn maybe_mine(&self) -> bool {
        if !self.roll() {
            return false;
        }
        self.stats.attempted.fetch_add(1, Ordering::Relaxed);
        let config = self.config.clone();
        let node_id = self.node_id.clone();
        let attempt = self.counter.fetch_add(1, Ordering::Relaxed);
        let stats = std::sync::Arc::clone(&self.stats);
        tokio::spawn(async move {
            let outcome = attempt_mine(&config, &node_id, attempt).await;
            match &outcome {
                Ok(o) if o.won => {
                    stats.won.fetch_add(1, Ordering::Relaxed);
                    tracing::info!(target: "economics", "mined JER 🎉");
                }
                Ok(o) => tracing::debug!(target: "economics", reason = ?o.reason, "mine attempt did not win"),
                Err(e) => tracing::debug!(target: "economics", error = %e, "mine attempt failed"),
            }
        });
        true
    }

    /// Per-mille probability roll.
    pub fn roll(&self) -> bool {
        if self.config.mine_chance_permille >= 1000 {
            return true;
        }
        rand::random::<u32>() % 1000 < self.config.mine_chance_permille
    }
}

/// Perform one mine attempt (testable, non-spawned variant).
pub async fn attempt_mine(
    config: &MinerConfig,
    node_id: &str,
    attempt: u64,
) -> anyhow::Result<MineOutcome> {
    let token = config.token.clone().ok_or_else(|| {
        anyhow::anyhow!(
            "no JerithChain api token — put it in {} or set JERITH_TOKEN",
            default_token_file().display()
        )
    })?;
    let message_id = format!(
        "{node_id}:{attempt}:{}",
        crate::bootstrap::now_ms()
    );
    let url = format!("{}/mine", config.rpc_url.trim_end_matches('/'));
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()?;
    let resp = client
        .post(&url)
        .bearer_auth(token)
        .json(&serde_json::json!({
            "discord_id": config.identity,
            "source": "passive",
            "message_id": message_id,
            "guild_id": "",
            "channel_id": "",
        }))
        .send()
        .await?
        .error_for_status()?;
    #[derive(Deserialize)]
    struct MineResp {
        ok: bool,
        #[serde(default)]
        reason: Option<String>,
    }
    let body: MineResp = resp.json().await?;
    Ok(MineOutcome {
        attempted: true,
        won: body.ok,
        reason: body.reason,
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CryptoChain, CryptoConfig};

    fn cfg(chance: u32) -> CryptoConfig {
        CryptoConfig {
            chain: CryptoChain::Jerithchain,
            mine_chance_permille: chance,
            ..CryptoConfig::default()
        }
    }

    #[test]
    fn miner_skips_non_jerithchain() {
        let crypto = CryptoConfig {
            chain: CryptoChain::Bitcoin,
            ..CryptoConfig::default()
        };
        assert!(MinerConfig::from_crypto(&crypto, "n1").is_none());
    }

    #[test]
    fn miner_skips_zero_chance() {
        assert!(MinerConfig::from_crypto(&cfg(0), "n1").is_none());
    }

    #[test]
    fn miner_uses_default_rpc_when_unset() {
        let m = MinerConfig::from_crypto(&cfg(120), "n1").unwrap();
        assert_eq!(m.rpc_url, DEFAULT_JERITH_RPC_URL);
        assert_eq!(m.identity, "n1");
    }

    #[test]
    fn roll_always_when_1000() {
        let m = JerithMiner::new(
            MinerConfig {
                rpc_url: DEFAULT_JERITH_RPC_URL.into(),
                token: None,
                identity: "n1".into(),
                mine_chance_permille: 1000,
            },
            "n1",
        );
        for _ in 0..50 {
            assert!(m.roll());
        }
    }

    #[test]
    fn roll_never_when_zero() {
        let m = JerithMiner::new(
            MinerConfig {
                rpc_url: DEFAULT_JERITH_RPC_URL.into(),
                token: None,
                identity: "n1".into(),
                mine_chance_permille: 0,
            },
            "n1",
        );
        for _ in 0..50 {
            assert!(!m.roll());
        }
    }

    #[test]
    fn roll_in_band_for_half_chance() {
        let m = JerithMiner::new(
            MinerConfig {
                rpc_url: DEFAULT_JERITH_RPC_URL.into(),
                token: None,
                identity: "n1".into(),
                mine_chance_permille: 500,
            },
            "n1",
        );
        let hits = (0..1000).filter(|_| m.roll()).count();
        assert!(
            (400..=600).contains(&hits),
            "500‰ chance should win ~500/1000 times, got {hits}"
        );
    }

    #[tokio::test]
    async fn attempt_mine_without_token_fails_clearly() {
        let config = MinerConfig {
            rpc_url: DEFAULT_JERITH_RPC_URL.into(),
            token: None,
            identity: "n1".into(),
            mine_chance_permille: 120,
        };
        let err = attempt_mine(&config, "n1", 1).await.unwrap_err();
        assert!(err.to_string().contains("api token"));
    }

    #[test]
    fn mine_outcome_serializes() {
        let o = MineOutcome {
            attempted: true,
            won: true,
            reason: None,
        };
        let json = serde_json::to_string(&o).unwrap();
        let back: MineOutcome = serde_json::from_str(&json).unwrap();
        assert!(back.won);
    }
}
