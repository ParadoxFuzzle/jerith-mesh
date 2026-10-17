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

use anyhow::Context as _;
use serde::{Deserialize, Serialize};

use crate::config::{CryptoChain, CryptoConfig};

/// Local JerithChain node RPC defaults.
pub const DEFAULT_JERITH_RPC_URL: &str = "http://127.0.0.1:8300";

/// Where the miner looks for the node's api token. Env `JERITH_TOKEN_FILE`
/// wins, then this default (matches jerith_node.py's `keys/api.token`
/// layout); the CLI `config` command can set a custom path.
pub fn default_token_file() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("JERITH_TOKEN_FILE") {
        return std::path::PathBuf::from(p);
    }
    crate::config::NodeConfig::home()
        .join("keys")
        .join("api.token")
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
/// Shared by the miner and the Phase 5 stake chain-wiring (stake.rs).
pub fn read_token() -> Option<String> {
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

/// Effective chain RPC base URL for a crypto config.
pub fn chain_rpc_url(crypto: &CryptoConfig) -> String {
    crypto
        .wallet_endpoint
        .clone()
        .unwrap_or_else(|| DEFAULT_JERITH_RPC_URL.to_string())
}

/// The typed §9 refusal for an optional chain: the config accepts the
/// label (bitcoin / ethereum / dogecoin / monero) but no miner/wallet
/// adapter exists yet. Returned wherever those would be built so the node
/// says WHY it is idle instead of presenting a unlabeled quiet minimum —
/// there is no fake mining stats for an unconfigured chain.
/// `None` for JerithChain (the real adapter) and `None` (nothing configured).
pub fn optional_chain_note(chain: CryptoChain) -> Option<&'static str> {
    match chain {
        CryptoChain::Bitcoin => Some(
            "bitcoin: config accepted, no miner/wallet adapter yet — mining and the wallet card are disabled (no fake stats)",
        ),
        CryptoChain::Ethereum => Some(
            "ethereum: config accepted, no miner/wallet adapter yet — mining and the wallet card are disabled (no fake stats)",
        ),
        CryptoChain::Dogecoin => Some(
            "dogecoin: config accepted, no miner/wallet adapter yet — mining and the wallet card are disabled (no fake stats)",
        ),
        CryptoChain::Monero => Some(
            "monero: config accepted, no miner/wallet adapter yet — mining and the wallet card are disabled (no fake stats)",
        ),
        CryptoChain::Jerithchain | CryptoChain::None => None,
    }
}

// ---------------------------------------------------------------------------
// Per-node wallet card (Phase 4): the node's own JER balance from the
// chain RPC, chain-truthful rather than locally-counted.
// ---------------------------------------------------------------------------

/// The wallet card's chain-RPC probe config: which RPC base, which token,
/// and which identity/address answers as "this node's wallet".
#[derive(Debug, Clone)]
pub struct WalletCard {
    /// Chain RPC base (config `[crypto] wallet_endpoint`, default :8300).
    pub rpc_url: String,
    /// The node api token — the chain node token-gates /balance.
    pub token: Option<String>,
    /// What the card shows as the wallet identity (address, else node id).
    pub identity: String,
    /// When `[crypto] wallet_address` is set, /balance is asked by ADDRESS;
    /// otherwise the mined identity string (the node id) is asked by
    /// discord_id (the chain keys wallets by identity strings — the miner
    /// credits the node id, so the node's plain wallet card is the same
    /// account).
    pub address: Option<String>,
}

impl WalletCard {
    /// Build from the crypto section: a card exists only on JerithChain.
    /// The payout address comes from config `[crypto] wallet_address`; the
    /// default wallet card follows the miner's credit identity (node id).
    pub fn from_crypto(crypto: &CryptoConfig, node_id: &str) -> Option<Self> {
        if crypto.chain != CryptoChain::Jerithchain {
            return None;
        }
        let address = crypto.wallet_address.clone().filter(|a| !a.trim().is_empty());        let identity = address.clone().unwrap_or_else(|| node_id.to_string());
        Some(Self {
            rpc_url: chain_rpc_url(crypto),
            token: read_token(),
            identity,
            address,
        })
    }

    /// The /balance query string for this card's wallet.
    fn balance_query(&self) -> String {
        match &self.address {
            Some(a) => format!("address={}", urlencode(a.trim())),
            // The chain keys wallets by identity strings; a node identity
            // behaves like any other account handle for this purpose.
            None => format!("discord_id={}", urlencode(&self.identity)),
        }
    }
}

/// Percent/URL-encode for the /balance query (identity strings and chain
/// addresses are plain hex/urlsafe today; encode anyway so a wallet
/// address with exotic bytes survives the query).
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            other => out.push_str(&format!("%{other:02X}")),
        }
    }
    out
}

/// What the chain node answered about one wallet (jerith_node.py /balance).
#[derive(Debug, Clone, Serialize)]
pub struct ChainBalance {
    /// False when the identity has no wallet yet (chain-node contract:
    /// `discord_id` mode returns `exists:false` shape; `address` mode 404s
    /// — both carry balance 0 here, NOT a transport error: reachability
    /// is a separate bit from wallet existence).
    pub exists: bool,
    pub address: Option<String>,
    pub balance_uj: u64,
    pub balance_jer: f64,
    pub available_jer: Option<f64>,
}

/// Token-authed live balance probe for one wallet card. HTTP 404 in
/// address mode is an unknown wallet, not a failure — reachability and
/// existence are separate (the dashboard must never show "—" just
/// because the wallet wasn't created yet).
pub async fn wallet_balance(card: &WalletCard) -> anyhow::Result<ChainBalance> {
    let Some(token) = card.token.as_ref() else {
        anyhow::bail!(
            "chain api token missing — set JERITH_TOKEN or {}",
            default_token_file().display()
        );
    };
    let token: &str = token;
    let url = format!(
        "{}/balance?{}",
        card.rpc_url.trim_end_matches('/'),
        card.balance_query()
    );
    let resp = reqwest::Client::builder()
        .use_rustls_tls()
        .timeout(Duration::from_secs(5))
        .build()?
        .get(&url)
        .bearer_auth(token)
        .send()
        .await
        .context("chain balance unreachable")?;
    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        // Address asked, no wallet under it yet (chain-node contract).
        return Ok(ChainBalance {
            exists: false,
            address: card.address.clone(),
            balance_uj: 0,
            balance_jer: 0.0,
            available_jer: None,
        });
    }
    let body: serde_json::Value = resp
        .error_for_status()
        .context("chain balance rejected")?
        .json()
        .await?;
    if body.get("exists").and_then(|v| v.as_bool()) == Some(false) {
        return Ok(ChainBalance {
            exists: false,
            address: card.address.clone().or_else(|| {
                body.get("address")
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
            }),
            balance_uj: 0,
            balance_jer: 0.0,
            available_jer: None,
        });
    }
    Ok(ChainBalance {
        exists: true,
        address: body
            .get("address")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .or_else(|| card.address.clone()),
        balance_uj: body
            .get("balance_uj")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        balance_jer: body
            .get("balance_jer")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0),
        available_jer: body
            .get("available_jer")
            .and_then(|v| v.as_f64()),
    })
}

/// GET /balance?discord_id=<identity> → µJER credited balance.
/// (The chain keys wallets by identity strings; a node identity behaves
/// like any other account handle for this purpose.)
pub async fn chain_balance_uj(rpc_url: &str, identity: &str) -> anyhow::Result<u64> {
    let url = format!(
        "{}/balance?discord_id={}",
        rpc_url.trim_end_matches('/'),
        identity
    );
    let client = reqwest::Client::builder()
        .use_rustls_tls()
        .timeout(Duration::from_secs(5))
        .build()?;
    let resp = client
        .get(&url)
        .send()
        .await
        .context("chain balance unreachable")?;
    let body: serde_json::Value = resp
        .error_for_status()
        .context("chain balance rejected")?
        .json()
        .await?;
    Ok(body.get("balance_uj").and_then(|v| v.as_u64()).unwrap_or(0))
}

// Outcome of an attempted mine (for meters + logs).
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
                Ok(o) => {
                    tracing::debug!(target: "economics", reason = ?o.reason, "mine attempt did not win")
                }
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
    let message_id = format!("{node_id}:{attempt}:{}", crate::bootstrap::now_ms());
    let url = format!("{}/mine", config.rpc_url.trim_end_matches('/'));
    let client = reqwest::Client::builder()
        .use_rustls_tls()
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
    use serde_json::json;

    #[test]
    fn optional_chain_note_types_the_refusal_honestly() {
        // Each optional chain gets its own named refusal…
        for (chain, label) in [
            (CryptoChain::Bitcoin, "bitcoin"),
            (CryptoChain::Ethereum, "ethereum"),
            (CryptoChain::Dogecoin, "dogecoin"),
            (CryptoChain::Monero, "monero"),
        ] {
            let note = optional_chain_note(chain)
                .unwrap_or_else(|| panic!("{label} must carry a typed refusal"));
            assert!(note.contains(label), "{label} note must name the chain");
            assert!(
                note.contains("disabled"),
                "{label} note must say what is off"
            );
        }
        // …and the real adapters stay silent (nothing to refuse).
        assert_eq!(optional_chain_note(CryptoChain::Jerithchain), None);
        assert_eq!(optional_chain_note(CryptoChain::None), None);
    }

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

    /// Stub chain node (jerith_node.py-shaped /balance): token-gated,
    /// and the tests read back what it was asked (query + auth) from a
    /// shared slot. 404 on `address-DNE`, full shape otherwise.
    async fn stub_chain(
        seen: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    ) -> String {
        use axum::response::IntoResponse as _;
        let seen_chat = seen.clone();
        let app = axum::Router::new().route(
            "/balance",
            axum::routing::get(move |uri: axum::http::Uri, headers: axum::http::HeaderMap| {
                let seen = seen_chat.clone();
                async move {
                    seen.lock().unwrap().replace(format!(
                        "{}|{}",
                        uri.query().unwrap_or(""),
                        headers.get("authorization").and_then(|v| v.to_str().ok()).unwrap_or("")
                    ));
                    if headers.get("authorization").and_then(|v| v.to_str().ok())
                        != Some("Bearer tok")
                    {
                        return axum::http::StatusCode::UNAUTHORIZED
                            .into_response();
                    }
                    let q = uri.query().unwrap_or("");
                    if q.contains("address=addr-broke") {
                        return axum::http::StatusCode::NOT_FOUND.into_response();
                    }
                    let bal: u64 = if q.contains("balance=55") { 5_500_000 } else { 1_234_567 };
                    axum::Json(json!({
                        "exists": true,
                        "address": "jer1abc",
                        "balance_uj": bal,
                        "balance_jer": bal as f64 / 1e6,
                        "available_jer": bal as f64 / 1e6 - 0.5,
                    }))
                    .into_response()
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}")
    }

    fn card(rpc: &str, address: Option<&str>) -> WalletCard {
        WalletCard {
            rpc_url: rpc.to_string(),
            token: Some("tok".into()),
            identity: address.unwrap_or("n1").to_string(),
            address: address.map(str::to_string),
        }
    }

    #[test]
    fn wallet_card_from_crypto_address_wins_node_id_falls_back() {
        let mut crypto = cfg(120);
        crypto.wallet_endpoint = Some("http://127.0.0.1:9999".into());
        // Default: the card follows the miner's credit identity (node id).
        let c = WalletCard::from_crypto(&crypto, "nodeid1").unwrap();
        assert_eq!(c.rpc_url, "http://127.0.0.1:9999");
        assert_eq!(c.identity, "nodeid1");
        assert_eq!(c.address, None);
        // A config payout address becomes the card's identity.
        crypto.wallet_address = Some("jer1pay".into());
        let c = WalletCard::from_crypto(&crypto, "nodeid1").unwrap();
        assert_eq!(c.identity, "jer1pay");
        assert_eq!(c.address.as_deref(), Some("jer1pay"));
        // Non-JerithChain: no card.
        crypto.chain = CryptoChain::Bitcoin;
        assert!(WalletCard::from_crypto(&crypto, "n1").is_none());
    }

    #[tokio::test]
    async fn wallet_balance_asks_address_mode_and_authenticates() {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(None));
        let rpc = stub_chain(seen.clone()).await;
        let b = wallet_balance(&card(&rpc, Some("jer1abc"))).await.unwrap();
        assert!(b.exists);
        assert_eq!(b.address.as_deref(), Some("jer1abc"));
        assert_eq!(b.balance_uj, 1_234_567);
        let q = seen.lock().unwrap().take().unwrap();
        assert!(
            q.starts_with("address=jer1abc|Bearer tok"),
            "must ask by ADDRESS with the bearer token, got {q}"
        );
    }

    #[tokio::test]
    async fn wallet_balance_asks_discord_id_mode_without_address() {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(None));
        let rpc = stub_chain(seen.clone()).await;
        wallet_balance(&card(&rpc, None)).await.unwrap();
        let q = seen.lock().unwrap().take().unwrap();
        assert!(
            q.starts_with("discord_id=n1|Bearer tok"),
            "default card asks by the miner's credit identity, got {q}"
        );
    }

    #[tokio::test]
    async fn wallet_balance_unknown_address_is_exists_false_not_error() {
        // A dedicated stub: 404 → the typed not-a-wallet answer.
        let app = axum::Router::new().route(
            "/balance",
            axum::routing::get(
                || async { axum::http::StatusCode::NOT_FOUND },
            ),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let b = wallet_balance(&card(&format!("http://{addr}"), Some("addr-broke")))
            .await
            .unwrap();
        assert!(!b.exists);
        assert_eq!(b.balance_uj, 0);
    }

    #[tokio::test]
    async fn wallet_balance_without_token_fails_config_clearly() {
        let mut c = card("http://127.0.0.1:1", None);
        c.token = None;
        let err = wallet_balance(&c).await.unwrap_err();
        assert!(err.to_string().contains("api token missing"), "{err}");
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
