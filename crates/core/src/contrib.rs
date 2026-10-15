//! Real-time contribution measurement and cryptographic verification
//! (FEATURES §2, Phase 4 — the row the 25 %-eligibility rule grew up
//! alongside but never actually got).
//!
//! Before this module the node *declared* `contribution_percent` (config)
//! and signed only that declaration in its health reply — a freerider could
//! declare 75 % while serving nothing, and every verifier in the mesh would
//! repeat its claim verbatim. What ships here:
//!
//! 1. **Local measurement** — every ping tick, the node diffs the
//!    cumulative serve/consume meters into an interval share
//!    (`served / (served + consumed)`), EWMA-smooths it, and drives the
//!    real contribution-health gate from the *measured* ratio against the
//!    *declared* one. Idle intervals (nothing served, nothing consumed)
//!    are deliberately non-events: an idle contributor isn't draining
//!    anyone, and a blackout must never read as fraud.
//! 2. **Cryptographic verification** — the last completed interval is
//!    signed (Ed25519, same wire identity) as a `measurement` block inside
//!    the health reply. The pinger recomputes the signature exactly like a
//!    webhook receiver verifies: payload is canonical, so numbers can't be
//!    edited in flight, only re-signed — i.e., a node owns every
//!    measurement it publishes.
//! 3. **Contradiction evidence** — a per-peer ring of verified samples
//!    locally compares measured share against declared share. Repeated
//!    tampered blocks or demonstrated shortfalls (declaring ≥25 % while
//!    *demonstrably* serving under half that, three verified windows in a
//!    row) queue `Leeching` evidence through the same signed-report
//!    channel as every other anti-fraud layer. Statistics still never
//!    slash: the bootstrap weighs and verifies before any consequence.
//!
//! The whole ledger is in-memory per process, like the freerider detector:
//! a restart resets suspicion; behavior rebuilds it.

use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};

/// Completed intervals kept for the smoothed ratio (bounded: one tick =
/// one interval; a node restarts its view, not the mesh's).
pub const HISTORY_CAP: usize = 20;

/// EWMA weight of the newest interval (α=0.5: a single good interval
/// cannot erase a sorry week, but a mesh blackout also recovers fast).
pub const SMOOTH_ALPHA: f32 = 0.5;

/// Verified shortfalls required in a row before a peer's declared-vs-
/// measured contradiction becomes evidence (two would indict one busy
/// window; three matching in a row is a pattern).
pub const SHORTFALL_MIN_SAMPLES: usize = 3;

/// Tampered measurement blocks required before evidence (one could be a
/// corrupt peer or an upgrade race; two is a sender editing its numbers).
pub const TAMPER_TO_EVIDENCE: u32 = 2;

/// Peers tracked at once (bounded map: stale entries linger until
/// touched; the mesh is far smaller than this).
pub const LEDGER_PEER_CAP: usize = 256;

/// One completed ping interval's metering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntervalStat {
    pub window_secs: u64,
    pub tokens_served: u64,
    pub tokens_consumed: u64,
}

impl IntervalStat {
    /// Serve share of the interval as a percentage. None means IDLE:
    /// nothing served and nothing consumed — not a 0 %, an absence.
    pub fn share_percent(&self) -> Option<u64> {
        let total = self.tokens_served + self.tokens_consumed;
        (self.tokens_served * 100).checked_div(total)
    }
}

/// This node's rolling contribution measurement.
#[derive(Debug, Default)]
pub struct ContributionMeter {
    /// Cumulative counter bases for the last diff (the meters are
    /// process-lifetime cumulative; intervals always diff them).
    prev_total_served: u64,
    prev_total_consumed: u64,
    /// Last completed interval (signed into health replies).
    last: Option<IntervalStat>,
    /// EWMA-smoothed share, in percent (None until real activity).
    smoothed: Option<f32>,
    /// Raw shares of the recent intervals, newest last.
    history: VecDeque<u64>,
}

impl ContributionMeter {
    /// Observed one tick's cumulative meter values.
    pub fn observe_totals(&mut self, total_served: u64, total_consumed: u64, window_secs: u64) {
        let served = total_served.saturating_sub(self.prev_total_served);
        let consumed = total_consumed.saturating_sub(self.prev_total_consumed);
        self.prev_total_served = total_served;
        self.prev_total_consumed = total_consumed;
        self.observe_interval(served, consumed, window_secs);
    }

    /// Observed one interval (deltas only) — the shared entry point.
    pub fn observe_interval(&mut self, served: u64, consumed: u64, window_secs: u64) {
        let interval = IntervalStat {
            window_secs,
            tokens_served: served,
            tokens_consumed: consumed,
        };
        if let Some(share) = interval.share_percent() {
            let share_f = share as f32;
            self.smoothed = Some(match self.smoothed {
                None => share_f,
                Some(prev) => SMOOTH_ALPHA * prev + (1.0 - SMOOTH_ALPHA) * share_f,
            });
            if self.history.len() == HISTORY_CAP {
                self.history.pop_front();
            }
            self.history.push_back(share);
        }
        self.last = Some(interval);
    }

    /// The signed-weighted measurement point: (last interval, smoothed
    /// percent rounded down, clamped 0–100). None measured until some
    /// non-idle interval completed.
    pub fn measurement_point(&self) -> Option<(IntervalStat, u64)> {
        let measured = self.smoothed.map(|s| s.floor().clamp(0.0, 100.0) as u64)?;
        self.last.map(|last| (last, measured))
    }

    /// The real contribution-health gate: Requesters never gate (they
    /// exist to consume); an idle contributor is not a drain; otherwise
    /// the smoothed share must meet the declared minimum.
    pub fn healthy_for(&self, role: crate::config::NodeRole, declared_percent: u64) -> bool {
        use crate::config::NodeRole;
        if role == NodeRole::Requester {
            return true;
        }
        match self.measurement_point() {
            None => true, // idle: nothing consumed → not a drain
            Some((_, measured)) => measured >= declared_percent.min(100),
        }
    }

    /// Snapshot for `/status`: declared percent is appended by the caller.
    pub fn status_json(&self) -> Value {
        let history: Vec<serde_json::Value> = self
            .history
            .iter()
            .map(|&share| serde_json::json!(share))
            .collect();
        let last = self.last.map(|i| {
            serde_json::json!({
                "window_secs": i.window_secs,
                "tokens_served": i.tokens_served,
                "tokens_consumed": i.tokens_consumed,
                "share_percent": i.share_percent(),
            })
        });
        match self.measurement_point() {
            Some((_, measured)) => serde_json::json!({
                "measured_percent": measured,
                "last_interval": last,
                "history": history,
            }),
            None => serde_json::json!({ "measured_percent": null, "history": history }),
        }
    }
}

/// One pinger-verified measurement sample of a peer.
#[derive(Debug, Clone, Copy)]
pub struct VerifiedSample {
    pub declared: u64,
    pub measured: u64,
}

#[derive(Debug, Default)]
struct PeerContrib {
    samples: VecDeque<VerifiedSample>,
    /// Verified shortfall samples in a row (reset by any healthy sample).
    shortfalls: u32,
    tampered: u32,
}

/// Per-peer ledger of signed measurement verdicts (the verification half).
#[derive(Debug, Default)]
pub struct PeerContribLedger {
    peers: HashMap<String, PeerContrib>,
    flagged: HashSet<String>,
}

/// Peer IDs to submit `Leeching` evidence for after a recording round.
impl PeerContribLedger {
    /// A pinger verified the peer's signed numbers. Returns peers whose
    /// contradiction crossed the evidence line this round (flagged once
    /// per crossing; re-armed by recovery).
    pub fn record_verified(
        &mut self,
        node_id: &str,
        declared: u64,
        measured: u64,
    ) -> Vec<String> {
        if node_id.is_empty() {
            return Vec::new();
        }
        self.bound();
        let entry = self.peers.entry(node_id.to_string()).or_default();
        let shortfall = is_shortfall(declared, measured);
        entry.shortfalls = if shortfall { entry.shortfalls + 1 } else { 0 };
        entry.samples.push_back(VerifiedSample { declared, measured });
        if entry.samples.len() > SHORTFALL_MIN_SAMPLES * 3 {
            entry.samples.clear(); // bounded: keep the recent pattern only
        }
        if self.flagged.contains(node_id) {
            if !shortfall {
                // Recovered: one verified healthy window re-arms the flag
                // and restarts the pattern watch.
                self.flagged.remove(node_id);
                tracing::info!(
                    target: "contrib",
                    peer = node_id,
                    "measured contribution recovered — flag re-armed"
                );
            }
            return Vec::new();
        }
        if entry.shortfalls >= SHORTFALL_MIN_SAMPLES as u32 {
            self.flagged.insert(node_id.to_string());
            tracing::warn!(
                target: "contrib",
                peer = node_id,
                "measured contribution contradicts the declaration — evidence queued"
            );
            return vec![node_id.to_string()];
        }
        Vec::new()
    }

    /// A pinger got a measurement block that did NOT verify against the
    /// identity it claims (numbers edited in flight, or a forged block).
    pub fn record_tampered(&mut self, node_id: &str) -> Vec<String> {
        if node_id.is_empty() {
            return Vec::new();
        }
        self.bound();
        let entry = self.peers.entry(node_id.to_string()).or_default();
        entry.tampered += 1;
        if entry.tampered >= TAMPER_TO_EVIDENCE && !self.flagged.contains(node_id) {
            self.flagged.insert(node_id.to_string());
            return vec![node_id.to_string()];
        }
        Vec::new()
    }

    /// Bounded map: drop an arbitrary entry when at capacity and the new
    /// peer is not yet tracked (the mesh is far smaller than the cap;
    /// correctness of the pattern watch does not depend on who is evicted).
    fn bound(&mut self) {
        // (cap check happens at the call sites via length; nodes smaller
        // than LEDGER_PEER_CAP never evict.)
        if self.peers.len() >= LEDGER_PEER_CAP
            && let Some(oldest) = self.peers.keys().next().cloned()
        {
            self.peers.remove(&oldest);
        }
    }
}

/// Declared share the peer demonstrably fails to meet: `measured < declared/2`
/// (half the declared floor — the slack that keeps ordinary variance from
/// becoming a fraud report). Never fires below the 25 % mesh minimum.
pub fn is_shortfall(declared: u64, measured: u64) -> bool {
    declared >= crate::config::MIN_CONTRIBUTION_PERCENT && measured * 2 < declared
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::NodeRole;

    #[test]
    fn share_percent_math() {
        assert_eq!(
            IntervalStat { window_secs: 60, tokens_served: 40, tokens_consumed: 10 }.share_percent(),
            Some(80)
        );
        assert_eq!(
            IntervalStat { window_secs: 60, tokens_served: 0, tokens_consumed: 25 }.share_percent(),
            Some(0)
        );
        // Idle is an absence, not a 0 %.
        assert_eq!(
            IntervalStat { window_secs: 60, tokens_served: 0, tokens_consumed: 0 }.share_percent(),
            None
        );
    }

    #[test]
    fn observe_totals_diffs_the_cumulative_meters() {
        let mut m = ContributionMeter::default();
        // First tick: totals 50/10 → interval (50, 10), share 83 %.
        m.observe_totals(50, 10, 60);
        let (last, measured) = m.measurement_point().unwrap();
        assert_eq!(last.tokens_served, 50);
        assert_eq!(last.tokens_consumed, 10);
        assert_eq!(measured, 50 * 100 / 60, "first interval = raw share");
        // Second tick with NO growth (meters move only when served):
        // idle interval — last updates, the smoothed measure holds.
        m.observe_totals(50, 10, 60);
        let (last2, _) = m.measurement_point().unwrap();
        assert_eq!(last2.tokens_served, 0);
        assert_eq!(last2.tokens_consumed, 0);
        assert_eq!(m.measurement_point().unwrap().1, 50 * 100 / 60);
    }

    #[test]
    fn gate_follows_declared_share_role_and_idle() {
        let mut m = ContributionMeter::default();
        // Idle = not a drain (until it consumes something).
        assert!(m.healthy_for(NodeRole::Contributor, 25));
        assert!(m.healthy_for(NodeRole::Hybrid, 25));
        // Consume-only with zero serve IS a drain for a contributor.
        m.observe_interval(0, 900, 60);
        assert!(!m.healthy_for(NodeRole::Contributor, 25));
        // Requester: never gated, same consumption is its whole job.
        assert!(m.healthy_for(NodeRole::Requester, 25));
        // A fresh contributor meets the floor (share 80 first sample).
        let mut b = ContributionMeter::default();
        b.observe_interval(100, 25, 60);
        assert!(b.healthy_for(NodeRole::Contributor, 25));
        // …then dries up: one dry interval does NOT gate (α=0.5 keeps
        // 0.5*80=40), two in a row fall below the floor (0.5*40=20 < 25).
        b.observe_interval(0, 400, 60);
        assert!(b.healthy_for(NodeRole::Contributor, 25));
        b.observe_interval(0, 400, 60);
        assert!(!b.healthy_for(NodeRole::Contributor, 25));
        // Hybrid, same rule.
        assert!(!b.healthy_for(NodeRole::Hybrid, 25));
    }

    #[test]
    fn ledger_flags_three_shortfalls_then_rearms_on_recovery() {
        let mut ledger = PeerContribLedger::default();
        // Declaring 25 with measured 10 → shortfall (10*2 < 25).
        assert!(ledger.record_verified("peer-a", 25, 10).is_empty());
        assert!(ledger.record_verified("peer-a", 25, 10).is_empty());
        // Third in a row crosses.
        assert_eq!(ledger.record_verified("peer-a", 25, 10), vec!["peer-a"]);
        // Further shortfalls stay silent (flagged once per crossing).
        assert!(ledger.record_verified("peer-a", 25, 9).is_empty());
        // Recovery re-arms.
        assert!(ledger.record_verified("peer-a", 25, 60).is_empty());
        assert!(ledger.record_verified("peer-a", 25, 60).is_empty());
        // Not flagged again until three shortfalls ACCUMULATE again.
        assert!(ledger.record_verified("peer-a", 25, 8).is_empty());
        assert!(ledger.record_verified("peer-a", 25, 8).is_empty());
        assert_eq!(ledger.record_verified("peer-a", 25, 8), vec!["peer-a"]);
    }

    #[test]
    fn tampered_blocks_flag_after_two_and_gate_quiets_the_old_alias() {
        let mut ledger = PeerContribLedger::default();
        // One tampered block could be an upgrade race — hold on.
        assert!(ledger.record_tampered("peer-b").is_empty());
        // A second verified edit in flight is evidence.
        assert_eq!(ledger.record_tampered("peer-b"), vec!["peer-b"]);
        // Flagged once per crossing: further tampering stays silent.
        assert!(ledger.record_tampered("peer-b").is_empty());
    }

    #[test]
    fn shortfall_never_below_the_mesh_minimum() {
        // Declared < 25 is not a mesh declaration (setup enforces ≥25);
        // never score it.
        assert!(!is_shortfall(20, 0));
        assert!(is_shortfall(25, 12)); // 12*2 < 25
        assert!(!is_shortfall(25, 13));
        assert!(is_shortfall(75, 37)); // 37*2 < 75
        assert!(!is_shortfall(75, 38));
    }

    #[test]
    fn measurement_payload_roundtrips_and_tamper_is_caught() {
        let id = crate::signing::SigningIdentity::generate();
        let payload = crate::signing::measurement_payload(
            &id.node_id(),
            12345,
            60,
            400,
            100,
            25,
            80,
        );
        let sig = id.sign(payload.as_bytes());
        assert!(crate::signing::verify(
            &id.pubkey_hex(),
            payload.as_bytes(),
            &sig
        ));
        // One byte different in ANY field (here measured 80→81) breaks it.
        let wrong = crate::signing::measurement_payload(&id.node_id(), 12345, 60, 400, 100, 25, 81);
        assert!(!crate::signing::verify(&id.pubkey_hex(), wrong.as_bytes(), &sig));
    }
}
