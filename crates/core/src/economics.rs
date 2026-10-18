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
            "bitcoin: mining is not implemented for this chain and stays disabled (no fake stats); the wallet card probes your supplied wallet_endpoint live when configured",
        ),
        CryptoChain::Ethereum => Some(
            "ethereum: mining is not implemented for this chain and stays disabled (no fake stats); the wallet card probes your supplied wallet_endpoint live when configured",
        ),
        CryptoChain::Dogecoin => Some(
            "dogecoin: mining is not implemented for this chain and stays disabled (no fake stats); the wallet card probes your supplied wallet_endpoint live when configured",
        ),
        CryptoChain::Monero => Some(
            "monero: mining is not implemented for this chain and stays disabled (no fake stats); the wallet card probes your supplied wallet_endpoint live when configured",
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
    let resp = crate::http::mesh_http()
        .get(&url)
        .bearer_auth(token)
        .timeout(Duration::from_secs(5))
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

// ---------------------------------------------------------------------------
// Optional-chain wallet cards (§9, Phase 4): the wallet-card half of the
// Bitcoin / Ethereum / Dogecoin / Monero adapters. Mining on those chains
// stays honestly refused (the note above); the user's OWN node is probed
// for a chain-truthful balance, the same contract WalletCard serves for
// JER — no fake numbers, reachability carried separately.
// ---------------------------------------------------------------------------

/// Split a user-supplied endpoint `scheme://[user:pass@]host[:port][/path]`
/// into base URL + optional Basic-auth credentials (Bitcoin/Dogecoin Core
/// and monerod-style RPC auth embed in the URL — no new config fields).
/// Upper-case schemes accepted; anything without an http(s):// scheme is
/// None (a typed config error, never a silent guess).
pub fn split_endpoint(
    url: &str,
) -> Option<(String, Option<String>, Option<String>)> {
    let (scheme, rest) = url.split_once("://")?;
    if !(scheme.eq_ignore_ascii_case("http") || scheme.eq_ignore_ascii_case("https")) {
        return None;
    }
    let (creds, hostpart) = match rest.split_once('@') {
        Some((c, h)) => (Some(c), h),
        None => (None, rest),
    };
    let (user, pass) = match creds {
        Some(c) => match c.split_once(':') {
            Some((u, p)) => (Some(u.to_string()), Some(p.to_string())),
            None => (Some(c.to_string()), None),
        },
        None => (None, None),
    };
    Some((format!("{scheme}://{hostpart}"), user, pass))
}

/// The wallet-card probe config for one optional chain.
#[derive(Debug, Clone)]
pub struct OptionalChainWallet {
    pub chain: CryptoChain,
    /// User-supplied RPC base (may embed user:pass@ for Core-style auth).
    pub rpc_url: String,
    user: Option<String>,
    pass: Option<String>,
    /// Config `[crypto] wallet_address` — required for Ethereum; optional
    /// for Core chains (falls back to the wallet's own balance) and not
    /// meaningful for Monero (wallet-rpc is bound to its wallet file).
    pub address: Option<String>,
}

impl OptionalChainWallet {
    /// Build from the crypto section: exists when an optional chain has a
    /// wallet_endpoint worth probing. JerithChain has WalletCard; None
    /// chain has nothing configured.
    pub fn from_crypto(crypto: &CryptoConfig) -> Option<Self> {
        match crypto.chain {
            CryptoChain::Bitcoin
            | CryptoChain::Ethereum
            | CryptoChain::Dogecoin
            | CryptoChain::Monero => crypto.wallet_endpoint.as_ref().and_then(|url| {
                let (rpc_url, user, pass) = split_endpoint(url)?;
                Some(Self {
                    chain: crypto.chain,
                    rpc_url,
                    user,
                    pass,
                    address: crypto.wallet_address.clone().filter(|a| !a.trim().is_empty()),
                })
            }),
            CryptoChain::Jerithchain | CryptoChain::None => None,
        }
    }

    /// The human display unit for this chain's balance.
    pub fn unit(&self) -> &'static str {
        match self.chain {
            CryptoChain::Bitcoin => "BTC",
            CryptoChain::Ethereum => "ETH",
            CryptoChain::Dogecoin => "DOGE",
            CryptoChain::Monero => "XMR",
            _ => "?",
        }
    }

    /// The pooled mesh client (probe budget applied per request inside
    /// core_rpc/jsonrpc — the client is shared pool-wide now).
    fn http(&self) -> reqwest::Client {
        crate::http::mesh_http()
    }

    fn auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match (&self.user, &self.pass) {
            (Some(u), Some(p)) => req.basic_auth(u, Some(p)),
            (Some(u), None) => req.basic_auth(u, Some("")),
            _ => req,
        }
    }

    /// POST one Core-style JSON-RPC call (Bitcoin/Dogecoin), returning the
    /// `result` member. The RPC's own error member is surfaced as an Err —
    /// the chain said no, and the card must show that, not a fake zero.
    async fn core_rpc(&self, client: &reqwest::Client, method: &str, params: serde_json::Value) -> anyhow::Result<serde_json::Value> {
        let resp = self
            .auth(client.post(&self.rpc_url))
            .json(&serde_json::json!({
                "jsonrpc": "1.0",
                "id": "jerith",
                "method": method,
                "params": params,
            }))
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .context("chain RPC unreachable")?;
        let body: serde_json::Value = resp
            .error_for_status()
            .context("chain RPC rejected")?
            .json()
            .await
            .context("chain RPC did not answer JSON")?;
        if let Some(err) = body.get("error").filter(|e| !e.is_null()) {
            anyhow::bail!("chain RPC error: {err}");
        }
        body
            .get("result")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("chain RPC answer missing result"))
    }

    /// POST one EIP-1474 style JSON-RPC call (Ethereum, monero-wallet-rpc).
    async fn jsonrpc(&self, client: &reqwest::Client, method: &str, params: serde_json::Value) -> anyhow::Result<serde_json::Value> {
        let resp = self
            .auth(client.post(&self.rpc_url))
            .json(&serde_json::json!({
                "jsonrpc": "2.0",
                "id": "jerith",
                "method": method,
                "params": params,
            }))
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .context("chain RPC unreachable")?;
        let body: serde_json::Value = resp
            .error_for_status()
            .context("chain RPC rejected")?
            .json()
            .await
            .context("chain RPC did not answer JSON")?;
        if let Some(err) = body.get("error").filter(|e| !e.is_null()) {
            anyhow::bail!("chain RPC error: {err}");
        }
        body.get("result")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("chain RPC answer missing result"))
    }

    /// Chain-truthful balance for the address (or the wallet itself on
    /// Core/Monero). Succeeds with honest numbers or errs with the typed
    /// reason — the caller must never guess a balance.
    pub async fn balance(&self) -> anyhow::Result<OptionalBalance> {
        let unit = self.unit();
        let client = self.http();
        match self.chain {
            CryptoChain::Bitcoin | CryptoChain::Dogecoin => {
                // An address watches exactly that one; no address asks the
                // node wallet's own total (the operator's node, their truth).
                let (method, params) = match self.address.as_deref() {
                    Some(addr) => ("getreceivedbyaddress", serde_json::json!([addr, 1])),
                    None => ("getbalance", serde_json::json!([])),
                };
                let result = self.core_rpc(&client, method, params).await?;
                let amount = result
                    .as_f64()
                    .ok_or_else(|| anyhow::anyhow!("chain RPC returned a non-numeric balance"))?;
                Ok(OptionalBalance {
                    ok: true,
                    chain: self.chain,
                    unit,
                    amount,
                    raw: None,
                    address: self.address.clone(),
                    error: None,
                })
            }
            CryptoChain::Ethereum => {
                let addr = self.address.as_deref().ok_or_else(|| {
                    anyhow::anyhow!(
                        "ethereum probes need [crypto] wallet_address (eth_getBalance is address-keyed)"
                    )
                })?;
                if !addr.starts_with("0x") || addr.len() != 42 {
                    anyhow::bail!("wallet_address must be a 0x-prefixed 20-byte address");
                }
                let hex = self
                    .jsonrpc(&client, "eth_getBalance", serde_json::json!([addr, "latest"]))
                    .await?;
                let hexs = hex
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("eth_getBalance returned a non-string result"))?;
                let digits = hexs.strip_prefix("0x").unwrap_or(hexs);
                if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_hexdigit()) {
                    anyhow::bail!("eth_getBalance returned {hexs:?} — not hex");
                }
                let wei = u128::from_str_radix(digits, 16)
                    .map_err(|e| anyhow::anyhow!("eth_getBalance number too large: {e}"))?;
                Ok(OptionalBalance {
                    ok: true,
                    chain: self.chain,
                    unit,
                    amount: wei as f64 / 1e18,
                    raw: Some(wei.to_string()),
                    address: self.address.clone(),
                    error: None,
                })
            }
            CryptoChain::Monero => {
                let result = self.jsonrpc(&client, "get_balance", serde_json::json!({})).await?;
                let atomic = result
                    .get("balance")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| anyhow::anyhow!("monero get_balance missing numeric balance"))?;
                Ok(OptionalBalance {
                    ok: true,
                    chain: self.chain,
                    unit,
                    amount: atomic as f64 / 1e12,
                    raw: Some(atomic.to_string()),
                    address: None,
                    error: None,
                })
            }
            CryptoChain::Jerithchain | CryptoChain::None => {
                anyhow::bail!("not an optional chain")
            }
        }
    }
}

/// The probe outcome /status and the banner render: honest numbers, or the
/// typed reason the card is empty. `ok=false` never carries amounts.
#[derive(Debug, Clone, Serialize)]
pub struct OptionalBalance {
    /// False when the probe couldn't produce a trustworthy number.
    pub ok: bool,
    pub chain: CryptoChain,
    pub unit: &'static str,
    /// Human coin units (BTC/ETH/DOGE/XMR) when ok.
    pub amount: f64,
    /// Atomic units as a decimal string when derivable (wei / piconero);
    /// f64 display of 1e18-scale numbers loses digits, raw keeps them.
    pub raw: Option<String>,
    pub address: Option<String>,
    pub error: Option<String>,
}

/// Live probe wrapper for one optional-chain wallet card: Ok(numbers) or
/// Err(reason) — /status turns Err into `{ok:false, error}`.
pub async fn optional_chain_probe(
    card: &OptionalChainWallet,
) -> anyhow::Result<OptionalBalance> {
    card.balance().await
}

/// GET /balance?discord_id=<identity> → µJER credited balance.
/// (The chain keys wallets by identity strings; a node identity behaves
/// like any other account handle for this purpose.)
pub async fn chain_balance_uj(rpc_url: &str, identity: &str) -> anyhow::Result<u64> {    let url = format!(
        "{}/balance?discord_id={}",
        rpc_url.trim_end_matches('/'),
        identity
    );
    let resp = crate::http::mesh_http()
        .get(&url)
        .timeout(Duration::from_secs(5))
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
    let resp = crate::http::mesh_http()
        .post(&url)
        .bearer_auth(token)
        .timeout(Duration::from_secs(15))
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

    /// The optional-chain wallet card exists exactly when an http(s)
    /// endpoint is configured; Core-RPC credentials ride the URL, and the
    /// real chain configs still pass them through untouched.
    #[test]
    fn optional_wallet_from_crypto_and_split_endpoint() {
        // Plain endpoint, no creds, address carried.
        let crypto = CryptoConfig {
            chain: CryptoChain::Bitcoin,
            wallet_endpoint: Some("http://127.0.0.1:8332".to_string()),
            wallet_address: Some("bc1qtest".to_string()),
            ..CryptoConfig::default()
        };
        let w = OptionalChainWallet::from_crypto(&crypto).expect("endpoint configured");
        assert_eq!(w.rpc_url, "http://127.0.0.1:8332");
        assert_eq!(w.address.as_deref(), Some("bc1qtest"));
        assert_eq!(w.unit(), "BTC");

        // Core-RPC auth via user:pass@ (no new config fields needed).
        let crypto = CryptoConfig {
            chain: CryptoChain::Dogecoin,
            wallet_endpoint: Some("https://user:pass@pool.example:22555/wallet".to_string()),
            wallet_address: None,
            ..CryptoConfig::default()
        };
        let w = OptionalChainWallet::from_crypto(&crypto).expect("endpoint configured");
        assert_eq!(w.rpc_url, "https://pool.example:22555/wallet");
        assert!(matches!(&w.user, Some(u) if u == "user"));
        assert!(matches!(&w.pass, Some(p) if p == "pass"));
        assert_eq!(w.unit(), "DOGE");

        // JerithChain and no-endpoint stay None; a broken scheme refuses.
        let jer = CryptoConfig {
            chain: CryptoChain::Jerithchain,
            wallet_endpoint: Some("http://x".to_string()),
            ..CryptoConfig::default()
        };
        assert!(OptionalChainWallet::from_crypto(&jer).is_none());
        let none = CryptoConfig {
            chain: CryptoChain::Bitcoin,
            ..CryptoConfig::default()
        };
        assert!(OptionalChainWallet::from_crypto(&none).is_none());
        let bad = CryptoConfig {
            chain: CryptoChain::Bitcoin,
            wallet_endpoint: Some("ftp://nope".to_string()),
            ..CryptoConfig::default()
        };
        assert!(OptionalChainWallet::from_crypto(&bad).is_none());
    }

    /// Core-RPC happy path: an address watches getreceivedbyaddress; no
    /// address falls back to the wallet's own balance. The probe result is
    /// human coin units, and the RPC's OWN error member is an Err (no fake
    /// zero on a chain-side refusal).
    #[tokio::test]
    async fn optional_wallet_core_rpc_getreceived_and_getbalance() {
        use std::sync::Arc as Arcasier;
        let method_hits = Arcasier::new(std::sync::atomic::AtomicI64::new(-1));
        let auth_seen = Arcasier::new(std::sync::atomic::AtomicBool::new(false));
        let (hits, authh) = (method_hits.clone(), auth_seen.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let app = axum::Router::new().route(
                "/",
                axum::routing::post(
                    move |hdrs: axum::http::HeaderMap,
                          axum::Json(body): axum::Json<serde_json::Value>| {
                        let hits = hits.clone();
                        let authh = authh.clone();
                        async move {
                            // basic_auth for the URL-embedded creds.
                            let b64 = hdrs
                                .get("authorization")
                                .and_then(|v| v.to_str().ok())
                                .unwrap_or("")
                                .strip_prefix("Basic ")
                                .unwrap_or("")
                                .to_string();
                            authh.store(
                                b64 == "dXNlcjpwYXNz", // base64("user:pass")
                                std::sync::atomic::Ordering::SeqCst,
                            );
                            let m = body.get("method").and_then(|v| v.as_str()).unwrap_or("?");
                            hits.store(
                                match m {
                                    "getreceivedbyaddress" => 0,
                                    "getbalance" => 1,
                                    _ => 2,
                                },
                                std::sync::atomic::Ordering::SeqCst,
                            );
                            axum::Json(serde_json::json!({"result": 1.5, "error": null, "id": "x"}))
                        }
                    },
                ),
            );
            axum::serve(listener, app).await.unwrap();
        });

        let endpoint = format!("http://user:pass@{addr}");
        let crypto = CryptoConfig {
            chain: CryptoChain::Bitcoin,
            wallet_endpoint: Some(endpoint.clone()),
            wallet_address: Some("bc1qwatched".to_string()),
            ..CryptoConfig::default()
        };
        let w = OptionalChainWallet::from_crypto(&crypto).unwrap();
        let b = optional_chain_probe(&w).await.expect("probe ok");
        assert!(b.ok);
        assert_eq!(b.amount, 1.5);
        assert_eq!(b.unit, "BTC");
        assert_eq!(method_hits.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert!(auth_seen.load(std::sync::atomic::Ordering::SeqCst));

        // No address → the wallet's own getbalance.
        let crypto = CryptoConfig {
            chain: CryptoChain::Bitcoin,
            wallet_endpoint: Some(endpoint.clone()),
            wallet_address: None,
            ..CryptoConfig::default()
        };
        let w = OptionalChainWallet::from_crypto(&crypto).unwrap();
        let b = optional_chain_probe(&w).await.expect("probe ok");
        assert_eq!(b.amount, 1.5);
        assert_eq!(method_hits.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert!(b.address.is_none());

        // The chain refusing (error member) must reach the caller as Err —
        // Reach a hanging server: reuse the stub but ask an unreachable port.
        let w = OptionalChainWallet {
            chain: CryptoChain::Dogecoin,
            rpc_url: "http://127.0.0.1:1".to_string(),
            user: None,
            pass: None,
            address: None,
        };
        let e = optional_chain_probe(&w).await.expect_err("unreachable errs");
        assert!(e.to_string().contains("unreachable"), "{e}");
    }

    /// Ethereum: address-keyed eth_getBalance, hex wei → human ETH + raw
    /// decimal string; the missing/wrong address is a typed refusal.
    #[tokio::test]
    async fn optional_wallet_ethereum_hex_wei_and_typed_refusals() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let app = axum::Router::new().route(
                "/",
                axum::routing::post(
                    move |axum::Json(body): axum::Json<serde_json::Value>| async move {
                        assert_eq!(
                            body.get("method").and_then(|v| v.as_str()),
                            Some("eth_getBalance")
                        );
                        axum::Json(serde_json::json!({
                            "jsonrpc": "2.0",
                            "id": 1,
                            "result": "0x1bc16d674ec80000"
                        }))
                    },
                ),
            );
            axum::serve(listener, app).await.unwrap();
        });

        let endpoint = format!("http://{addr}");
        let addr_hex = "0x95222290dd7278aa3ddd389cc1e1d165cc4bafe5";
        let crypto = CryptoConfig {
            chain: CryptoChain::Ethereum,
            wallet_endpoint: Some(endpoint.clone()),
            wallet_address: Some(addr_hex.to_string()),
            ..CryptoConfig::default()
        };
        let w = OptionalChainWallet::from_crypto(&crypto).unwrap();
        let b = optional_chain_probe(&w).await.expect("probe ok");
        assert_eq!(b.raw.as_deref(), Some("2000000000000000000"));
        assert_eq!(b.amount, 2.0);
        assert_eq!(b.unit, "ETH");

        // No address → typed refusal, not an empty card.
        let crypto = CryptoConfig {
            chain: CryptoChain::Ethereum,
            wallet_endpoint: Some(endpoint.clone()),
            ..CryptoConfig::default()
        };
        let w = OptionalChainWallet::from_crypto(&crypto).unwrap();
        let e = optional_chain_probe(&w).await.expect_err("no address errs");
        assert!(e.to_string().contains("wallet_address"), "{e}");

        // Malformed address → typed refusal.
        let crypto = CryptoConfig {
            chain: CryptoChain::Ethereum,
            wallet_endpoint: Some(endpoint.clone()),
            wallet_address: Some("0xshort".to_string()),
            ..CryptoConfig::default()
        };
        let w = OptionalChainWallet::from_crypto(&crypto).unwrap();
        let e = optional_chain_probe(&w).await.expect_err("bad address errs");
        assert!(e.to_string().contains("0x-prefixed"), "{e}");
    }

    /// Monero: wallet-rpc get_balance, atomic piconero → human XMR + raw;
    /// a missing numeric balance is an honest Err, never a fake zero.
    #[tokio::test]
    async fn optional_wallet_monero_piconero() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let app = axum::Router::new().route(
                "/json_rpc",
                axum::routing::post(
                    move |axum::Json(body): axum::Json<serde_json::Value>| async move {
                        assert_eq!(
                            body.get("method").and_then(|v| v.as_str()),
                            Some("get_balance")
                        );
                        axum::Json(serde_json::json!({
                            "jsonrpc": "2.0",
                            "id": "0",
                            "result": {"balance": 1_500_000_000_000u64, "unlocked_balance": 1_500_000_000_000u64}
                        }))
                    },
                ),
            );
            axum::serve(listener, app).await.unwrap();
        });

        // The wallet file is address-agnostic — endpoint carries the path.
        let endpoint = format!("http://{addr}/json_rpc");
        let crypto = CryptoConfig {
            chain: CryptoChain::Monero,
            wallet_endpoint: Some(endpoint),
            wallet_address: None,
            ..CryptoConfig::default()
        };
        let w = OptionalChainWallet::from_crypto(&crypto).unwrap();
        let b = optional_chain_probe(&w).await.expect("probe ok");
        assert_eq!(b.amount, 1.5);
        assert_eq!(b.raw.as_deref(), Some("1500000000000"));
        assert_eq!(b.unit, "XMR");
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
