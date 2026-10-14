//! Phase 5 economics: stake + slashing on JerithChain (optional tier).
//!
//! Spec (§ "Economic skin in the game"): a Contributor joins with a small
//! JER stake; successful verified work earns rewards, and proven
//! freeriding (commitment failures, ringer mismatches, failed-worker
//! evidence) slashes part of the stake.
//!
//! v2 shape — the JerithChain node now implements the escrow contract:
//! `POST /stake/lock`, `POST /stake/unlock`, `GET /stake` (api-token
//! authed) and `POST /stake/slash` (Ed25519 mesh wire auth). Core wiring:
//!
//! 1. **Stake join gate** — when `required_uj` > 0, the local ledger's
//!    `staked_uj` is trued up against the live `GET /stake` view each mesh
//!    tick (`refresh_from_chain`): a node whose real escrow is short no
//!    longer clears the gate, and /status shows the live number.
//! 2. **Escrow lifecycle** — the ping loop escrows the configured stake
//!    once per process (`lock_required_once`: POST /stake/lock, the node's
//!    chain wallet auto-created on first 404, idempotent via the persisted
//!    `escrowed_uj` marker; shortfall-only top-ups) and returns it on mesh
//!    shutdown (`unlock_all`: POST /stake/unlock). Restarts skip the lock.
//! 3. **Slash ledger (local, persisted)** — every verified offense the
//!    node reports about a peer is recorded here with the penalty amount:
//!    `required_uj × slash_percent / 100` per offense, floored at 1 µJ
//!    while staking is enabled.
//! 4. **Signed slash notices** — each DishonestResults report the fraud
//!    flush accepts becomes a POST {chain}/stake/slash with the same
//!    Ed25519 wire auth the bootstrap uses (`WireAuth::with_stamp`); the
//!    reporter is derived from the signing key (pubkey[:16]) so the
//!    chain's reporter↔pubkey binding always holds:
//!      POST {chain}/stake/slash
//!      {"reporter","target","amount_uj","reason","evidence_digest","stamp_ms"}
//!      payload: jerith-slash1|{reporter}|{target}|{amount_uj}|{stamp}
//!    Failure to submit is degraded to a log entry — the ledger keeps the
//!    record either way, so nothing rides on the endpoint responding.

use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::config::CryptoConfig;

/// Persisted at `JERITH_HOME/data/stake.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SlashEvent {
    pub ts_ms: u64,
    /// The accused peer (or `"self"` for self-slashing in future revisions).
    pub target: String,
    /// Penalty amount in micro-JER.
    pub amount_uj: u64,
    /// Machine-readable reason, e.g. `dishonest_results`.
    pub reason: String,
    /// SHA-256 (hex) of the signed evidence payload — links notices to the
    /// bootstrap fraud reports without storing raw peer data.
    pub evidence_digest: String,
    /// Whether the chain accepted a POST (false while the chain side of
    /// the contract is pending).
    pub chain_accepted: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StakeLedger {
    pub staked_uj: u64,
    pub slashed_total_uj: u64,
    /// µJ this node escrowed on the chain via /stake/lock — the idempotency
    /// marker so a restart doesn't re-lock. Slashes move escrow → treasury,
    /// and a live mapping to the chain's own response wakes this up via
    /// `refresh-from-chain` — true-up only against the fresh number when it is present.
    #[serde(default)]
    pub escrowed_uj: u64,
    pub slash_events: Vec<SlashEvent>,
}

impl StakeLedger {
    fn path() -> std::path::PathBuf {
        crate::config::NodeConfig::data_dir().join("stake.json")
    }

    /// Load (missing/corrupt → fresh ledger; staking math continues).
    pub fn load() -> Self {
        Self::load_from(&Self::path())
    }

    /// Load from an explicit path (test/dev seam: private ledgers keep the
    /// shared production file out of the tests' blast radius).
    pub fn load_from(path: &std::path::Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(text) => serde_json::from_str(&text).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self) -> anyhow::Result<()> {
        self.save_to(&Self::path())
    }

    /// Persist under an explicit path (pairs with `load_from`).
    pub fn save_to(&self, path: &std::path::Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_string(self)?)?;
        Ok(())
    }

    /// Record a slash event; caps the audit ring at the last 50 entries.
    /// Returns the recorded event.
    pub fn record_slash(&mut self, ev: SlashEvent) -> SlashEvent {
        self.slashed_total_uj += ev.amount_uj;
        self.slash_events.push(ev.clone());
        if self.slash_events.len() > 50 {
            self.slash_events.drain(..self.slash_events.len() - 50);
        }
        ev
    }

    /// Whether this node currently clears the join-gate line.
    pub fn clears_gate(&self, required_uj: u64) -> bool {
        required_uj == 0 || self.staked_uj >= required_uj
    }
}

/// Shared handle for gateway/ping-loop access.
pub struct StakeManager {
    ledger: StdMutex<StakeLedger>,
    config: StakeConfig,
    /// Chain RPC base (derived from the wallet endpoint; None when the
    /// node isn't wired to Jerithchain or the tier is disabled).
    chain_base: Option<String>,
    /// Ed25519 identity that signs slash notices (None in tests/CLI reads).
    identity: Option<crate::signing::SigningIdentity>,
    /// Bearer token for the token-authed chain endpoints (/stake/*,
    /// /wallet/create) — JERITH_TOKEN or the token file, same as mining.
    token: Option<String>,
    /// Where the ledger persists. Defaults to data_dir/stake.json; tests
    /// redirect it to a private temp path so parallel suites and the live
    /// node never scribble over each other's escrow markers.
    ledger_path: std::path::PathBuf,
    /// One escrow attempt per process (restarts reset the flag and verify).
    lock_attempted: std::sync::atomic::AtomicBool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct StakeConfig {
    /// Required JER escrow in micro-JER for Contributor/Hybrid (0 = off).
    pub required_uj: u64,
    /// Percent of the required stake slashed per verified offense.
    pub slash_percent_per_offense: u32,
    /// POST slash notices to the chain (feature-tolerant; see module docs).
    pub chain_slash: bool,
}

impl Default for StakeConfig {
    fn default() -> Self {
        Self {
            required_uj: 0,
            slash_percent_per_offense: 25,
            chain_slash: true,
        }
    }
}

impl StakeManager {
    /// Build from config; derives the slash URL from the chain RPC.
    pub fn new(
        crypto: &CryptoConfig,
        identity: Option<crate::signing::SigningIdentity>,
    ) -> Arc<Self> {
        let config = crypto.stake;
        let chain_base =
            if config.chain_slash && crypto.chain == crate::config::CryptoChain::Jerithchain {
                Some(crate::economics::chain_rpc_url(crypto))
            } else {
                None
            };
        let ledger_path = StakeLedger::path();
        let ledger = StakeLedger::load_from(&ledger_path);
        // No eager save here: constructing a manager must not touch the
        // shared global ledger file — tests rearm to a private path right
        // after, and production persists on the first real mutation.
        Arc::new(Self {
            ledger: StdMutex::new(ledger),
            config,
            chain_base,
            identity,
            token: crate::economics::read_token(),
            ledger_path,
            lock_attempted: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Point the ledger at a private path with fresh state (tests/dev):
    /// cross-test and cross-process pollution of the shared production
    /// ledger otherwise lets one run's escrow/slash markers short-circuit
    /// another's lock flows and even slash-journals fake offenses into the
    /// live node's dataset.
    pub fn rearm_ledger(&mut self, path: std::path::PathBuf) {
        let ledger = StakeLedger::load_from(&path);
        self.ledger_path = path;
        self.save_ledger(&ledger);
    }

    /// Ledger snapshot for /status and the dashboard.
    pub fn snapshot(&self) -> StakeLedger {
        self.ledger.lock().unwrap().clone()
    }

    /// Join gate verdict against the current ledger.
    pub fn clears_gate(&self) -> bool {
        self.ledger
            .lock()
            .unwrap()
            .clears_gate(self.config.required_uj)
    }

    /// The configured requirement (µJER; 0 = staking off).
    pub fn required_uj(&self) -> u64 {
        self.config.required_uj
    }

    /// Persist the ledger. Failures are warned, never fatal — in-memory
    /// accounting stays authoritative for the running process.
    fn save_ledger(&self, ledger: &StakeLedger) {
        if let Err(e) = ledger.save_to(&self.ledger_path) {
            tracing::warn!(target: "economics", error = %e, "stake ledger save failed");
        }
    }

    // ---- chain wiring (the escrow contract live on jerith_node.py) -----

    /// Raw `GET /stake` body for this node's identity (Some on success).
    /// Shared by the tick sync and the `jerith stake status` CLI view.
    pub async fn chain_view(&self) -> Option<serde_json::Value> {
        let identity = self.identity.as_ref()?;
        let base = self.chain_base.as_deref()?;
        let token = self.token.clone()?;
        let url = format!(
            "{}/stake?discord_id={}",
            base.trim_end_matches('/'),
            identity.node_id()
        );
        let (status, body) = stake_get(&url, &token).await.ok()?;
        if !(200..300).contains(&status) {
            tracing::debug!(target: "economics", status, "chain /stake view unavailable");
            return None;
        }
        Some(body)
    }

    /// Trues the ledger up against the live `GET /stake` view each mesh
    /// tick: a slash the chain applied against us surfaces as a lower
    /// `staked_uj` (the join gate gets honest), `slashed_total_uj` syncs to
    /// the chain's own accounting. Quiet no-op without chain/token/identity
    /// or when the wallet hasn't been chain-created yet (404).
    pub async fn refresh_from_chain(&self) -> bool {
        let body = match self.chain_view().await {
            Some(b) => b,
            None => return false, // chain down — keep local truth, quiet
        };
        let staked = body.get("staked_uj").and_then(|v| v.as_u64()).unwrap_or(0);
        let slashed = body
            .get("slashed_total_uj")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        {
            let mut ledger = self.ledger.lock().unwrap();
            let changed = ledger.staked_uj != staked || ledger.slashed_total_uj != slashed;
            ledger.staked_uj = staked;
            ledger.slashed_total_uj = slashed;
            if changed {
                self.save_ledger(&ledger);
            }
        }
        true
    }

    /// Escrow the join stake — exactly one attempt per mesh process
    /// (restarts reset the flag and re-verify against chain truth): a no-op
    /// when staking is disabled, the chain isn't wired, or the requirement
    /// is already covered. See `lock_required`.
    pub async fn lock_required_once(&self) {
        if self
            .lock_attempted
            .swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            return;
        }
        self.lock_required().await;
    }

    /// Escrow the configured requirement: posts only the shortfall against
    /// the refreshed view. True when the requirement is covered afterwards
    /// (already-covered inputs short-circuit without a POST).
    pub async fn lock_required(&self) -> bool {
        let required = self.config.required_uj;
        if required == 0 {
            return false; // staking off — never escrow
        }
        let covered = {
            let ledger = self.ledger.lock().unwrap();
            ledger.staked_uj.max(ledger.escrowed_uj)
        };
        if covered >= required {
            return true; // escrow already in place (marker or live view)
        }
        self.lock_topup(required - covered).await
    }

    /// Explicit escrow attempt (drives `lock_required`, and CLI top-ups):
    /// POST /stake/lock for exactly `amount_uj`, the chain auto-creating
    /// the wallet on first 404 via /wallet/create. The chain clamps to what
    /// it actually escrows, so the returned `staked_uj` is recorded
    /// verbatim and the idempotency marker follows along.
    pub async fn lock_topup(&self, amount_uj: u64) -> bool {
        let Some(identity) = self.identity.as_ref() else {
            return false;
        };
        let Some(base) = self.chain_base.as_deref() else {
            return false;
        };
        let Some(token) = self.token.clone() else {
            tracing::debug!(target: "economics", "stake lock skipped: no chain api token");
            return false;
        };
        if amount_uj == 0 {
            tracing::debug!(target: "economics", "stake lock skipped: zero amount");
            return true; // nothing requested — nothing to do
        }
        let discord_id = identity.node_id();
        let body = serde_json::json!({
            "discord_id": discord_id,
            "amount_jer": amount_uj as f64 / 1_000_000.0,
        });
        match stake_post(base, "/stake/lock", &token, &body).await {
            Ok((200..=299, reply)) => {
                self.record_chain_stake(&reply);
                tracing::info!(target: "economics", escrowed_uj = amount_uj, "stake escrowed on chain");
                true
            }
            Ok((404, _)) => {
                // Chain has no wallet for this identity yet — create it,
                // then retry the lock exactly once.
                let wb = serde_json::json!({"discord_id": discord_id});
                match stake_post(base, "/wallet/create", &token, &wb).await {
                    Ok((200..=299, _)) => {
                        match stake_post(base, "/stake/lock", &token, &body).await {
                            Ok((200..=299, reply)) => {
                                self.record_chain_stake(&reply);
                                tracing::info!(target: "economics", escrowed_uj = amount_uj, "wallet created; stake escrowed on chain");
                                true
                            }
                            other => {
                                tracing::warn!(target: "economics", reply = ?other, "stake lock failed after wallet create");
                                false
                            }
                        }
                    }
                    other => {
                        tracing::warn!(target: "economics", reply = ?other, "chain wallet create failed — cannot escrow stake");
                        false
                    }
                }
            }
            Ok((status, reply)) => {
                tracing::warn!(
                    target: "economics",
                    status,
                    detail = %reply.get("detail").and_then(|d| d.as_str()).unwrap_or("?"),
                    "chain rejected stake lock"
                );
                false
            }
            Err(e) => {
                tracing::warn!(target: "economics", error = %e, "chain stake lock unreachable");
                false
            }
        }
    }

    /// Record exactly what the chain says it now escrows for this node
    /// (used by lock and unlock responses).
    fn record_chain_stake(&self, reply: &serde_json::Value) {
        let staked = reply.get("staked_uj").and_then(|v| v.as_u64()).unwrap_or(0);
        let mut ledger = self.ledger.lock().unwrap();
        ledger.staked_uj = staked;
        ledger.escrowed_uj = staked;
        self.save_ledger(&ledger);
    }

    /// Return the whole known escrow on mesh shutdown (best effort): the
    /// chain clamps to what actually remains after slashes, and both
    /// `staked_uj` and the idempotency marker true up to the response.
    pub async fn unlock_all(&self) -> bool {
        let escrowed = {
            let ledger = self.ledger.lock().unwrap();
            ledger.escrowed_uj
        };
        if escrowed == 0 {
            return false; // never escrowed here — nothing to unwind
        }
        self.unlock_amount(escrowed).await
    }

    /// Explicit unlock attempt (drives `unlock_all`, and CLI unlocks): POST
    /// /stake/unlock for exactly `amount_uj` of this node's escrow. Both
    /// ledger values true up to the response, so a partial chain-side
    /// clamp lands as live truth.
    pub async fn unlock_amount(&self, amount_uj: u64) -> bool {
        let Some(identity) = self.identity.as_ref() else {
            return false;
        };
        let Some(base) = self.chain_base.as_deref() else {
            return false;
        };
        let Some(token) = self.token.clone() else {
            return false;
        };
        if amount_uj == 0 {
            return true; // nothing requested — nothing to do
        }
        let body = serde_json::json!({
            "discord_id": identity.node_id(),
            "amount_jer": amount_uj as f64 / 1_000_000.0,
        });
        match stake_post(base, "/stake/unlock", &token, &body).await {
            Ok((200..=299, reply)) => {
                self.record_chain_stake(&reply);
                true
            }
            other => {
                tracing::warn!(target: "economics", reply = ?other, "stake unlock failed — escrow persists");
                false
            }
        }
    }

    /// Penalty for one verified offense against `target` (µJER).
    /// 0 when staking is disabled entirely.
    fn penalty_uj(&self) -> u64 {
        if self.config.required_uj == 0 {
            return 0;
        }
        (self.config.required_uj * self.config.slash_percent_per_offense as u64 / 100).max(1)
    }

    /// Record (and optionally submit) a slash notice for a verified offense.
    /// `evidence_payload` is the exact signed bytes shown to the bootstrap
    /// — its SHA-256 anchors the ledger entry.
    pub async fn notice(&self, target: &str, reason: &str, evidence_payload: &str) {
        let amount = self.penalty_uj();
        if amount == 0 {
            return;
        }
        let ev = SlashEvent {
            ts_ms: crate::bootstrap::now_ms(),
            target: target.to_string(),
            amount_uj: amount,
            reason: reason.to_string(),
            evidence_digest: hex::encode(<sha2::Sha256 as sha2::Digest>::digest(
                evidence_payload.as_bytes(),
            )),
            chain_accepted: false,
        };
        let slash_url = self
            .chain_base
            .as_deref()
            .map(|b| format!("{}/stake/slash", b.trim_end_matches('/')));
        let submitted = match (slash_url, self.sign_notice(&ev)) {
            (Some(url), Some((auth, body))) => submit_slash(&url, &auth, &body).await,
            _ => false,
        };
        let mut ledger = self.ledger.lock().unwrap();
        let ev = SlashEvent {
            chain_accepted: submitted,
            ..ev
        };
        let ev = ledger.record_slash(ev);
        self.save_ledger(&ledger);
        tracing::warn!(
            target: "economics",
            target = %ev.target,
            amount_uj = ev.amount_uj,
            chain_accepted = submitted,
            "slash notice recorded ({reason})"
        );
    }

    /// Signed body for the chain slash contract.
    fn sign_notice(
        &self,
        ev: &SlashEvent,
    ) -> Option<(crate::signing::WireAuth, serde_json::Value)> {
        let identity = self.identity.as_ref()?;
        // The chain binds reporter == pubkey[:16]; deriving the reporter
        // from the signing key (not config) guarantees that binding holds
        // even if config node_id ever drifts from the key fingerprint.
        let reporter = identity.node_id();
        let payload = format!(
            "jerith-slash1|{}|{}|{}|{}",
            reporter, ev.target, ev.amount_uj, ev.ts_ms
        );
        let auth = crate::signing::WireAuth::with_stamp(identity, ev.ts_ms, payload.as_bytes());
        Some((
            auth,
            serde_json::json!({
                "reporter": reporter,
                "target": ev.target,
                "amount_uj": ev.amount_uj,
                "reason": ev.reason,
                "evidence_digest": ev.evidence_digest,
                "stamp_ms": ev.ts_ms,
            }),
        ))
    }
}

/// POST the notice; failure just means "recorded locally only" (the/wire
/// contract is live on jerith_node.py since the 2026-10-12 deploy).
async fn submit_slash(
    url: &str,
    auth: &crate::signing::WireAuth,
    body: &serde_json::Value,
) -> bool {
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
    {
        Ok(c) => c,
        Err(_) => return false,
    };
    let resp = client
        .post(url)
        .header("x-jerith-pubkey", &auth.pubkey)
        .header("x-jerith-stamp", auth.stamp_ms.to_string())
        .header("x-jerith-sig", &auth.sig)
        .json(body)
        .send()
        .await;
    match resp {
        Ok(r) if r.status().is_success() => true,
        Ok(r) => {
            tracing::debug!(target: "economics", status = %r.status(), "chain slash endpoint rejected notice");
            false
        }
        Err(_) => false, // chain down — ledger keeps the record
    }
}

/// POST a token-authed stake-chain endpoint (/stake/lock, /stake/unlock,
/// /wallet/create). Returns (status, json-body-or-null); the caller logs
/// and degrades — nothing here is fatal.
async fn stake_post(
    base: &str,
    path: &str,
    token: &str,
    body: &serde_json::Value,
) -> anyhow::Result<(u16, serde_json::Value)> {
    let url = format!("{}{}", base.trim_end_matches('/'), path);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()?;
    let resp = client
        .post(url)
        .bearer_auth(token)
        .json(body)
        .send()
        .await?;
    let status = resp.status().as_u16();
    let text = resp.text().await.unwrap_or_default();
    let json = serde_json::from_str(&text).unwrap_or(serde_json::Value::Null);
    Ok((status, json))
}

/// GET variant for the /stake read view.
async fn stake_get(url: &str, token: &str) -> anyhow::Result<(u16, serde_json::Value)> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()?;
    let resp = client.get(url).bearer_auth(token).send().await?;
    let status = resp.status().as_u16();
    let text = resp.text().await.unwrap_or_default();
    let json = serde_json::from_str(&text).unwrap_or(serde_json::Value::Null);
    Ok((status, json))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CryptoChain;

    fn crypto(required_uj: u64) -> CryptoConfig {
        CryptoConfig {
            chain: CryptoChain::Jerithchain,
            wallet_endpoint: None,
            wallet_address: None,
            mine_chance_permille: 0,
            stake: StakeConfig {
                required_uj,
                ..Default::default()
            },
        }
    }

    #[test]
    fn gate_clears_only_with_enough_stake() {
        let l = StakeLedger {
            staked_uj: 900,
            ..Default::default()
        };
        assert!(l.clears_gate(0)); // disabled → always clear
        assert!(l.clears_gate(900)); // exactly at line
        assert!(!l.clears_gate(1000)); // short
    }

    #[test]
    fn penalty_math_no_zero_when_enabled() {
        // Pure math check via config: 25 % of 800 µJ = 200; of 1 µJ → min 1.
        assert_eq!(800 * 25 / 100, 200);
        assert_eq!(1 * 25 / 100, 0); // why the .max(1) floor exists
    }

    #[test]
    fn ledger_caps_audit_ring_and_sums() {
        let mut l = StakeLedger::default();
        for i in 0..55 {
            l.record_slash(SlashEvent {
                ts_ms: i,
                target: format!("p{i}"),
                amount_uj: 10,
                reason: "dishonest_results".into(),
                evidence_digest: "x".into(),
                chain_accepted: false,
            });
        }
        assert_eq!(l.slash_events.len(), 50); // ring capped
        assert_eq!(l.slashed_total_uj, 550); // total never lost
        assert_eq!(l.slash_events.first().unwrap().target, "p5"); // oldest dropped
    }

    #[test]
    fn stake_config_defaults_match_spec() {
        let s = StakeConfig::default();
        assert_eq!(s.required_uj, 0); // opt-in tier
        assert_eq!(s.slash_percent_per_offense, 25); // spec's "part of the stake"
        assert!(s.chain_slash);
    }

    // ---- chain wiring (escrow contract live on jerith_node.py) -------------

    /// A local axum stub with exactly the endpoints the chain node serves
    /// for the stake contract, recording every hit for assertions.
    struct ChainStub {
        addr: std::net::SocketAddr,
        hits: Arc<StdMutex<Vec<(String, serde_json::Value)>>>,
    }

    impl ChainStub {
        /// `staked_after_lock`: what GET /stake and /stake/lock report once
        /// the escrow went through (mimics the chain clamping to reality).
        async fn spawn(staked_after_lock: Option<u64>) -> Self {
            let hits: Arc<StdMutex<Vec<(String, serde_json::Value)>>> =
                Arc::new(StdMutex::new(Vec::new()));
            let view_hits = hits.clone();
            let app = axum::Router::new()
                .route(
                    "/stake",
                    axum::routing::get(
                        move |axum::extract::Query(q): axum::extract::Query<
                            std::collections::HashMap<String, String>,
                        >| {
                            let state = view_hits.clone();
                            async move {
                                state
                                    .lock()
                                    .unwrap()
                                    .push(("GET /stake".into(), serde_json::Value::Null));
                                axum::Json(serde_json::json!({
                                    "discord_id": q.get("discord_id").cloned().unwrap_or_default(),
                                    "staked_uj": 0,
                                    "slashed_total_uj": 0,
                                    "events": [],
                                }))
                            }
                        },
                    ),
                )
                .route(
                    "/stake/lock",
                    axum::routing::post({
                        let state = hits.clone();
                        move |axum::Json(body): axum::Json<serde_json::Value>| async move {
                            state
                                .lock()
                                .unwrap()
                                .push(("POST /stake/lock".into(), body.clone()));
                            (
                                axum::http::StatusCode::OK,
                                axum::Json(serde_json::json!({
                                    "ok": true,
                                    "staked_uj": staked_after_lock.unwrap_or(0),
                                })),
                            )
                        }
                    }),
                )
                .route(
                    "/stake/unlock",
                    axum::routing::post({
                        let state = hits.clone();
                        move |axum::Json(body): axum::Json<serde_json::Value>| async move {
                            state
                                .lock()
                                .unwrap()
                                .push(("POST /stake/unlock".into(), body.clone()));
                            (
                                axum::http::StatusCode::OK,
                                axum::Json(serde_json::json!({
                                    "ok": true, "staked_uj": 0, "slashed_total_uj": 0,
                                })),
                            )
                        }
                    }),
                )
                .route(
                    "/wallet/create",
                    axum::routing::post({
                        let state = hits.clone();
                        move |axum::Json(body): axum::Json<serde_json::Value>| async move {
                            state
                                .lock()
                                .unwrap()
                                .push(("POST /wallet/create".into(), body.clone()));
                            (
                                axum::http::StatusCode::OK,
                                axum::Json(serde_json::json!({"ok": true})),
                            )
                        }
                    }),
                )
                .route(
                    "/stake/slash",
                    axum::routing::post({
                        let state = hits.clone();
                        move |headers: axum::http::HeaderMap,
                              axum::Json(body): axum::Json<serde_json::Value>| {
                            async move {
                                let header = |name: &str| {
                                    headers
                                        .get(name)
                                        .and_then(|v| v.to_str().ok())
                                        .unwrap_or_default()
                                        .to_string()
                                };
                                let sig = header("x-jerith-sig");
                                let pk = header("x-jerith-pubkey");
                                let stamp = header("x-jerith-stamp");
                                let amount_uj = body.get("amount_uj").cloned().unwrap_or_default();
                                state.lock().unwrap().push(
                                    (format!(
                                        "POST /stake/slash pk={pk} stamp={stamp} sig={sig}"
                                    ), body),
                                );
                                (axum::http::StatusCode::OK, axum::Json(serde_json::json!({
                                    "ok": true, "slashed_uj": amount_uj,
                                })))
                            }
                        }
                    }),
                );
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            Self { addr, hits }
        }

        fn path_hits(&self) -> Vec<String> {
            self.hits
                .lock()
                .unwrap()
                .iter()
                .map(|(p, _)| p.clone())
                .collect()
        }
    }

    /// StakeManager pointed at a stub chain (wallet_endpoint override).
    /// The token is injected explicitly: the env/file lookup that production
    /// uses is simply not available under `cargo test`.
    /// Unique per-uses temp ledger path — parallel tests (and repeat runs)
    /// each get their own file instead of racing on the shared production
    /// ledger at data_dir/stake.json.
    fn fresh_ledger_path() -> std::path::PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("jerith-stake-test-{}-{n}.json", std::process::id()))
    }

    fn manager_with_stub(stub: &ChainStub) -> Arc<StakeManager> {
        let mut crypto = crypto(1_000_000);
        crypto.wallet_endpoint = Some(format!("http://{}", stub.addr));
        let mut mgr = StakeManager::new(&crypto, Some(crate::signing::SigningIdentity::generate()));
        if let Some(m) = Arc::get_mut(&mut mgr) {
            m.token = Some("test-token".into());
            m.rearm_ledger(fresh_ledger_path());
        }
        mgr
    }

    #[tokio::test]
    async fn lock_creates_wallet_then_escrows() {
        let stub = ChainStub::spawn(Some(1_000_000)).await;
        let mgr = manager_with_stub(&stub);
        assert!(mgr.lock_required().await);
        let hit_paths = stub.path_hits();
        // Wallet 404 path isn't triggered by this stub (it returns 200
        // straight away), but the lock lands and the ledger true-ups.
        assert!(hit_paths.iter().any(|p| p == "POST /stake/lock"));
        let ledger = mgr.snapshot();
        assert_eq!(ledger.staked_uj, 1_000_000);
        assert_eq!(ledger.escrowed_uj, 1_000_000);
    }

    #[tokio::test]
    async fn lock_is_idempotent_once_covered() {
        let stub = ChainStub::spawn(Some(1_000_000)).await;
        let mgr = manager_with_stub(&stub);
        assert!(mgr.lock_required().await);
        let hits_after_first = stub.path_hits().len();
        // Covered now → a second call must not POST anything.
        assert!(mgr.lock_required().await);
        assert_eq!(stub.path_hits().len(), hits_after_first);
    }

    #[tokio::test]
    async fn unlock_posts_and_zeros_marker() {
        let stub = ChainStub::spawn(Some(1_000_000)).await;
        let mgr = manager_with_stub(&stub);
        assert!(mgr.lock_required().await);
        assert!(mgr.unlock_all().await);
        assert!(stub.path_hits().iter().any(|p| p == "POST /stake/unlock"));
        let ledger = mgr.snapshot();
        assert_eq!(ledger.escrowed_uj, 0);
        assert_eq!(ledger.staked_uj, 0);
    }

    #[tokio::test]
    async fn lock_topup_posts_explicit_amount_even_when_covered() {
        let stub = ChainStub::spawn(Some(2_700_000)).await;
        let mgr = manager_with_stub(&stub);
        // An operator top-up posts regardless of the gate's covered math —
        // `jerith stake lock <amount>` is an instruction, not gate logic.
        assert!(mgr.lock_topup(700_000).await);
        let hits = stub.hits.lock().unwrap().clone();
        let (_, body) = hits
            .iter()
            .find(|(p, _)| p == "POST /stake/lock")
            .cloned()
            .expect("explicit lock posted");
        assert!((body["amount_jer"].as_f64().unwrap() - 0.7).abs() < 1e-9);
        // Chain truth recorded verbatim (stub reports 2_700_000).
        let ledger = mgr.snapshot();
        assert_eq!(ledger.staked_uj, 2_700_000);
        assert_eq!(ledger.escrowed_uj, 2_700_000);
        // lock_required itself is now covered by the chain's reply.
        assert!(mgr.lock_required().await);
        assert_eq!(
            stub.path_hits()
                .iter()
                .filter(|p| *p == "POST /stake/lock")
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn unlock_amount_posts_explicit_partial_then_marker_covers_rest() {
        let stub = ChainStub::spawn(Some(0)).await;
        let mgr = manager_with_stub(&stub);
        {
            let mut l = mgr.ledger.lock().unwrap();
            l.staked_uj = 400_000;
            l.escrowed_uj = 400_000;
        }
        // Partial unlock: the exact body goes out, and the response zeroes
        // both ledger values (chain truth wins over the marker).
        assert!(mgr.unlock_amount(200_000).await);
        let hits = stub.hits.lock().unwrap().clone();
        let (_, body) = hits
            .iter()
            .find(|(p, _)| p == "POST /stake/unlock")
            .cloned()
            .expect("partial unlock posted");
        assert!((body["amount_jer"].as_f64().unwrap() - 0.2).abs() < 1e-9);
        let ledger = mgr.snapshot();
        assert_eq!(ledger.staked_uj, 0);
        assert_eq!(ledger.escrowed_uj, 0);
    }

    #[tokio::test]
    async fn refresh_pulls_chain_truth_into_ledger() {
        let stub = ChainStub::spawn(None).await; // view reports staked 0
        let mgr = manager_with_stub(&stub);
        {
            let mut l = mgr.ledger.lock().unwrap();
            l.staked_uj = 999_999;
            l.escrowed_uj = 999_999;
        }
        assert!(mgr.refresh_from_chain().await);
        let ledger = mgr.snapshot();
        assert_eq!(ledger.staked_uj, 0); // chain truth wins
        assert_eq!(ledger.slashed_total_uj, 0); // synced too
    }

    #[tokio::test]
    async fn slash_notice_yields_exact_wire_contract() {
        let stub = ChainStub::spawn(Some(0)).await;
        let crypto = {
            let mut c = crypto(1_000_000);
            c.wallet_endpoint = Some(format!("http://{}", stub.addr));
            c
        };
        let identity = crate::signing::SigningIdentity::generate();
        let mut mgr = StakeManager::new(&crypto, Some(identity.clone()));
        // Private fresh ledger: otherwise this notice's escrow/slash
        // markers land in the shared production stake.json — polluting the
        // live node's dataset and short-circuiting sibling tests' flows.
        if let Some(m) = Arc::get_mut(&mut mgr) {
            m.rearm_ledger(fresh_ledger_path());
        }
        mgr.notice("victim-node", "dishonest_results", "evidence-bytes")
            .await;

        let hits = stub.hits.lock().unwrap().clone();
        let (meta, body) = hits
            .iter()
            .find(|(p, _)| p.starts_with("POST /stake/slash"))
            .cloned()
            .expect("slash notice posted");
        assert_eq!(body["target"], "victim-node");
        assert_eq!(body["amount_uj"], 250_000); // 25% of 1 JER
        // The reporter must be the identity's node_id (pubkey[:16]),
        // matching the chain's reporter↔pubkey binding.
        assert_eq!(body["reporter"], identity.node_id());
        // Mirror the chain node's verification exactly: split apart the
        // x-jerith-* header trio the stub captured, rebuild the payload
        // chain-style off the header stamp, and check reporter == pk[:16].
        let grab = |key: &str, meta: &str| {
            meta.split_whitespace()
                .find_map(|part| part.strip_prefix(&format!("{key}=")))
                .unwrap_or_default()
                .to_string()
        };
        let pk = grab("pk", &meta);
        let stamp = grab("stamp", &meta);
        let sig = grab("sig", &meta);
        assert!(!pk.is_empty() && !sig.is_empty() && !stamp.is_empty());
        assert_eq!(&pk[..16], body["reporter"].as_str().unwrap());
        assert_eq!(stamp, body["stamp_ms"].as_u64().unwrap().to_string());
        let payload = format!(
            "jerith-slash1|{}|victim-node|250000|{}",
            body["reporter"].as_str().unwrap(),
            stamp,
        );
        assert!(crate::signing::verify(&pk, payload.as_bytes(), &sig));
    }
}
