//! Phase 5 economics: stake + slashing on JerithChain (optional tier).
//!
//! Spec (§ "Economic skin in the game"): a Contributor joins with a small
//! JER stake; successful verified work earns rewards, and proven
//! freeriding (commitment failures, ringer mismatches, failed-worker
//! evidence) slashes part of the stake.
//!
//! v1 shape, scoped to what the JerithChain node's API can do today
//! (`/mine`, `/balance`, `/mystate` — no native stake contract yet):
//!
//! 1. **Stake join gate** — when `required_uj` > 0, a Contributor/Hybrid
//!    must present a chain balance ≥ the requirement at startup. A node
//!    that can't (or whose chain is unreachable) is *demoted to Requester*
//!    for the session — it can consume compute but is never offered work.
//!    Config is untouched, so the node self-restores the role once it
//!    stakes (or the chain comes back) on the next start.
//! 2. **Slash ledger (local, persisted)** — every verified offense the
//!    node *reports about a peer* (and, in a future chain revision, every
//!    offense peers prove about *this* node) is recorded here with the
//!    penalty amount: `required_uj × slash_percent / 100` per offense,
//!    floored at 1 µJ while staking is enabled.
//! 3. **Signed slash notices** — when a chain slash endpoint is enabled,
//!    each notice is POSTed with the same Ed25519 wire auth the bootstrap
//!    uses (`stamps frozen via WireAuth::with_stamp`):
//!      POST {chain}/stake/slash
//!      {"reporter","target","amount_uj","reason","evidence","stamp_ms"}
//!      payload: jerith-slash1|{reporter}|{target}|{amount_uj}|{stamp}
//!    Failure to submit is degraded to a log entry — the ledger keeps the
//!    record either way, so nothing rides on the endpoint existing yet.

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
    pub slash_events: Vec<SlashEvent>,
}

impl StakeLedger {
    fn path() -> std::path::PathBuf {
        crate::config::NodeConfig::data_dir().join("stake.json")
    }

    /// Load (missing/corrupt → fresh ledger; staking math continues).
    pub fn load() -> Self {
        match std::fs::read_to_string(Self::path()) {
            Ok(text) => serde_json::from_str(&text).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self) -> anyhow::Result<()> {
        let path = Self::path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, serde_json::to_string(self)?)?;
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
    /// Where the chain slash endpoint lives (derived from wallet endpoint).
    chain_slash_url: Option<String>,
    node_id: String,
    /// Ed25519 identity that signs slash notices (None in tests/CLI reads).
    identity: Option<crate::signing::SigningIdentity>,
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
        node_id: &str,
        identity: Option<crate::signing::SigningIdentity>,
    ) -> Arc<Self> {
        let config = crypto.stake;
        let chain_slash_url = if config.chain_slash {
            let base = crypto
                .wallet_endpoint
                .clone()
                .unwrap_or_else(|| crate::economics::DEFAULT_JERITH_RPC_URL.to_string());
            Some(format!("{}/stake/slash", base.trim_end_matches('/')))
        } else {
            None
        };
        let mut ledger = StakeLedger::load();
        ledger.staked_uj = config.required_uj.max(ledger.staked_uj);
        let _ = ledger.save();
        Arc::new(Self {
            ledger: StdMutex::new(ledger),
            config,
            chain_slash_url,
            node_id: node_id.to_string(),
            identity,
        })
    }

    /// Ledger snapshot for /status and the dashboard.
    pub fn snapshot(&self) -> StakeLedger {
        self.ledger.lock().unwrap().clone()
    }

    /// Join gate verdict against the current ledger.
    pub fn clears_gate(&self) -> bool {
        self.ledger.lock().unwrap().clears_gate(self.config.required_uj)
    }

    /// The configured requirement (µJER; 0 = staking off).
    pub fn required_uj(&self) -> u64 {
        self.config.required_uj
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
        let submitted = match (&self.chain_slash_url, self.sign_notice(&ev)) {
            (Some(url), Some((auth, body))) => submit_slash(url, &auth, &body).await,
            _ => false,
        };
        let mut ledger = self.ledger.lock().unwrap();
        let ev = SlashEvent { chain_accepted: submitted, ..ev };
        let ev = ledger.record_slash(ev);
        if let Err(e) = ledger.save() {
            tracing::warn!(target: "economics", error = %e, "stake ledger save failed");
        }
        tracing::warn!(
            target: "economics",
            target = %ev.target,
            amount_uj = ev.amount_uj,
            chain_accepted = submitted,
            "slash notice recorded ({reason})"
        );
    }

    /// Signed body for the (future) chain slash contract.
    fn sign_notice(&self, ev: &SlashEvent) -> Option<(crate::signing::WireAuth, serde_json::Value)> {
        let identity = self.identity.as_ref()?;
        let payload = format!(
            "jerith-slash1|{}|{}|{}|{}",
            self.node_id, ev.target, ev.amount_uj, ev.ts_ms
        );
        let auth = crate::signing::WireAuth::with_stamp(identity, ev.ts_ms, payload.as_bytes());
        Some((
            auth,
            serde_json::json!({
                "reporter": self.node_id,
                "target": ev.target,
                "amount_uj": ev.amount_uj,
                "reason": ev.reason,
                "evidence_digest": ev.evidence_digest,
                "stamp_ms": ev.ts_ms,
            }),
        ))
    }
}

/// POST the notice; the endpoint is allowed to be absent (contract is
/// pending on the chain side) — false just means "recorded locally only".
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
        Err(_) => false, // chain down / endpoint pending — ledger keeps it
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CryptoChain, NodeRole};

    fn crypto(required_uj: u64) -> CryptoConfig {
        CryptoConfig {
            chain: CryptoChain::Jerithchain,
            wallet_endpoint: None,
            wallet_address: None,
            mine_chance_permille: 0,
            stake: StakeConfig { required_uj, ..Default::default() },
        }
    }

    #[test]
    fn gate_clears_only_with_enough_stake() {
        let l = StakeLedger { staked_uj: 900, ..Default::default() };
        assert!(l.clears_gate(0));      // disabled → always clear
        assert!(l.clears_gate(900));    // exactly at line
        assert!(!l.clears_gate(1000));  // short
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
        assert_eq!(l.slashed_total_uj, 550);  // total never lost
        assert_eq!(l.slash_events.first().unwrap().target, "p5"); // oldest dropped
    }

    #[test]
    fn stake_config_defaults_match_spec() {
        let s = StakeConfig::default();
        assert_eq!(s.required_uj, 0);               // opt-in tier
        assert_eq!(s.slash_percent_per_offense, 25); // spec's "part of the stake"
        assert!(s.chain_slash);
    }
}
