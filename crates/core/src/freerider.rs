//! Statistical freerider detection (Phase 4 — ARCHITECTURE.md §5, "Statistical
//! scoring" row).
//!
//! The other anti-fraud layers catch *proofs* (commitments, redundant replay,
//! ringer challenges). This layer catches the *pattern* of a node that keeps
//! its registry slot warm (healthy pings, honest ads) while being a net drain
//! on the mesh it claims to contribute to. Two conservative signals, both
//! measured from the coordinator side (the node that offers work via expert
//! dispatch):
//!
//! 1. **latency-vs-pool** — for one expert, a peer whose *successful* serve
//!    latency sits at a median far above the median of the OTHER peers
//!    serving the SAME expert in the same window (needs ≥ 3 comparison
//!    peers, ≥ 3 samples, ≥ 4× slower) erodes the mesh's throughput relative
//!    to what its claim implies.
//! 2. **selected-but-absent** — in a window where expert work demonstrably
//!    FLOWED (≥ 2 other peers answering ≥ 40 % of their selections), a peer
//!    selected ≥ 10 times that answered ≤ 25 % of those selections is
//!    soaking up plan slots other peers would have used.
//!
//! Honest-degradation rules are hard-wired:
//! - dispatches that end in an ERROR feed nothing — an all-honest-infra
//!   blackout dilutes EVERY peer's answer rate equally and must never be
//!   read as an outlier;
//! - dispatches where the LOCAL weights answered feed nothing — the remote
//!   peers were never actually needed;
//! - timeouts on a peer's side never appear here at all: only completed,
//!   peer-served runs feed samples;
//! - a peer that simply never appears in plans is never scored.
//!
//! Consequence discipline: crossing the flag threshold queues
//! `FraudKind::Leeching` **evidence** into the shared signed-report channel —
//! the bootstrap weighs it (score +20) and applies its own verification
//! before any stake consequence. Statistics never slash. The score decays
//! every evaluation (recovery by behavior); the flag is raised once and
//! re-armed only after the score falls below half the flag line. The whole
//! ledger is in-memory per process: a restart resets suspicion.
//!
//! Operators watch the running suspicion on `/status` (freerider section).

use serde_json::{json, Value};
use std::collections::{HashMap, HashSet, VecDeque};

/// Served samples kept per peer (bounded ring: a stale window must fade).
pub const SAMPLES_PER_PEER_CAP: usize = 128;

/// Distinct OTHER peers that must have served the same expert in-window
/// before a peer's latency is comparable at all.
pub const POOL_MIN_PEERS: usize = 3;

/// Peer median latency beyond this multiple of the pool median is a
/// latency-vs-pool signal. Deliberately lax for a heterogeneous mesh.
pub const LATENCY_FACTOR: f32 = 4.0;

/// Minimum distinct served samples for one (peer, expert) latency median.
pub const LATENCY_MIN_SAMPLES: usize = 3;

/// Selections before a peer's answer-rate is evaluated at all. Counted
/// WITHIN the recency window: a peer asked rarely is judged on recent asks,
/// not on process-lifetime history.
pub const ABSENT_MIN_SELECTIONS: usize = 10;

/// Pool-side context for the absent signal: the window must show real flow.
pub const POOL_ANSWER_RATE_FLOOR: f32 = 0.4;

/// A peer answering at most this share of its selections (with the pool
/// flowing) is soaking up plan slots.
pub const ABSENT_ANSWER_RATE_CEILING: f32 = 0.25;

/// Other peers (≥ 2) whose answer rate clears the floor — the pool context.
pub const POOL_RATE_MIN_PEERS: usize = 2;

/// Score added per evaluation when a signal fires (once per signal).
pub const WEIGHT_LATENCY: f32 = 18.0;
pub const WEIGHT_ABSENT: f32 = 10.0;

/// Effective recency window per peer: latency comparisons read only the
/// most recent W served samples. Without it a recovery would be invisible —
/// one bad week of samples would weigh forever against a healed peer.
pub const WINDOW_SAMPLES: usize = 12;

/// Evaluation decay: suspicion fades when behavior stops feeding it.
pub const DECAY: f32 = 0.93;



/// Score at which Leeching evidence is queued.
pub const FLAG_SCORE: f32 = 55.0;

/// Hysteresis: the flag re-arms only after the score falls back under half
/// the flag line (recovery is real, not one clean round).
pub const HYSTERESIS_FACTOR: f32 = 0.5;

#[derive(Debug, Clone)]
struct Serve {
    /// Absolute clock — kept for future windowing overhaul.
    _at_ms: u64,
    expert_id: String,
    latency_ms: u64,
}

#[derive(Debug, Default)]
struct PeerLedger {
    /// Successful serves (bounded ring, expert-tagged).
    served: VecDeque<Serve>,
    /// Selection ledger (bounded ring): true = answered this ask. Walk-
    /// order semantics — only the ranks the dispatch walk actually asked
    /// (everything up to and including the peer that served) count as
    /// asks; ranks BEHIND the server were never attempted and must never
    /// soak suspicion for not answering work nobody gave them.
    selections: VecDeque<bool>,
}

/// Per-peer statistical freerider ledger + evaluation. In-memory by design.
#[derive(Debug, Default)]
pub struct FreeriderDetector {
    peers: HashMap<String, PeerLedger>,
    scores: HashMap<String, f32>,
    /// Peers whose crossing already queued evidence (re-armed by hysteresis).
    flagged_once: HashSet<String>,
    /// Total dispatch runs (peer-served only) that fed the ledger.
    dispatches_observed: u64,
    /// Total flag events actually raised (≤ per-peer crossing count).
    flags_raised: u64,
}

impl FreeriderDetector {
    /// Feed one COMPLETED dispatch where a remote peer served the expert.
    /// `candidates` is the ordered remote plan (self excluded by the caller);
    /// every listed peer is ledgered as selected this round, `served_by`
    /// additionally as answered.
    pub fn feed_dispatch(
        &mut self,
        expert_id: &str,
        candidates: &[String],
        served_by: &str,
        latency_ms: u64,
    ) {
        self.dispatches_observed += 1;
        // Walk-order cut: the ranks behind `served_by` were never asked.
        let ask_upto = candidates
            .iter()
            .position(|c| c == served_by)
            .unwrap_or(candidates.len());
        let now_ms = crate::bootstrap::now_ms();
        for id in candidates.iter().take(ask_upto + 1) {
            if id.is_empty() {
                continue;
            }
            let answered = id == served_by;
            let led = self.peers.entry(id.clone()).or_default();
            if answered {
                if led.served.len() >= SAMPLES_PER_PEER_CAP {
                    led.served.pop_front();
                }
                led.served.push_back(Serve {
                    _at_ms: now_ms,
                    expert_id: expert_id.to_string(),
                    latency_ms,
                });
            }
            if led.selections.len() >= WINDOW_SAMPLES {
                led.selections.pop_front();
            }
            led.selections.push_back(answered);
        }
    }

    /// Run one evaluation round (decay + signals + hysteresis). Returns the
    /// node ids whose crossing should queue Leeching evidence.
    pub fn evaluate(&mut self) -> Vec<String> {
        // Decay + window roll-out first: suspicion and stale counters fade
        // when nothing feeds them.
        for s in self.scores.values_mut() {
            *s *= DECAY;
            if *s < 0.05 {
                *s = 0.0;
            }
        }
        // --- Signal 1: latency-vs-pool, per (peer, expert) ---------------
        let mut by_expert: HashMap<&str, HashMap<&String, Vec<u64>>> = HashMap::new();
        for (id, led) in &self.peers {
            // Recency window: only the most recent samples speak.
            let start = led.served.len().saturating_sub(WINDOW_SAMPLES);
            for s in led.served.iter().skip(start) {
                by_expert
                    .entry(s.expert_id.as_str())
                    .or_default()
                    .entry(id)
                    .or_default()
                    .push(s.latency_ms);
            }
        }
        for per_peer in by_expert.values() {
            // Per-peer median of successful serves for this expert.
            let per_median: HashMap<String, f32> = per_peer
                .iter()
                .filter(|(_, l)| l.len() >= LATENCY_MIN_SAMPLES)
                .filter_map(|(id, l)| median(l).map(|m| ((*id).clone(), m)))
                .collect();
            for (id, mine) in &per_median {
                // Comparison pool: OTHER peers on the same expert.
                let others: Vec<f32> = per_median
                    .iter()
                    .filter(|(oid, _)| oid != &id)
                    .map(|(_, m)| *m)
                    .collect();
                if others.len() < POOL_MIN_PEERS {
                    continue;
                }
                let pool = median_f32(others).unwrap_or(f32::MAX);
                if *mine > LATENCY_FACTOR * pool {
                    let s = self.scores.entry(id.clone()).or_default();
                    if *s < FLAG_SCORE {
                        *s += WEIGHT_LATENCY;
                    }
                }
            }
        }

        // --- Signal 2: selected-but-absent -------------------------------
        // Rates AFTER the window decay, so old selections fade out of the
        // judgment. Pool context: ≥ 2 OTHER peers answering ≥ 40 %.
        let rates: Vec<(String, f32)> = self
            .peers
            .iter()
            .filter(|(_, l)| l.selections.len() >= ABSENT_MIN_SELECTIONS)
            .map(|(id, l)| {
                let answered = l.selections.iter().filter(|a| **a).count();
                (id.clone(), answered as f32 / l.selections.len() as f32)
            })
            .collect();
        for (id, rate) in &rates {
            let other_flowing = rates
                .iter()
                .filter(|(oid, r)| oid != id && *r >= POOL_ANSWER_RATE_FLOOR)
                .count();
            if other_flowing < POOL_RATE_MIN_PEERS {
                continue;
            }
            if *rate <= ABSENT_ANSWER_RATE_CEILING {
                let s = self.scores.entry(id.clone()).or_default();
                if *s < FLAG_SCORE {
                    *s += WEIGHT_ABSENT;
                }
            }
        }

        // --- Flag crossing + hysteresis ----------------------------------
        let mut flagged: Vec<String> = Vec::new();
        for (id, s) in &self.scores {
            if *s >= FLAG_SCORE && !self.flagged_once.contains(id) {
                self.flagged_once.insert(id.clone());
                self.flags_raised += 1;
                flagged.push(id.clone());
            }
        }
        // Re-arm below the hysteresis line.
        let hysteresis = FLAG_SCORE * HYSTERESIS_FACTOR;
        self.flagged_once
            .retain(|id| self.scores.get(id).copied().unwrap_or(0.0) >= hysteresis);

        flagged.sort();
        flagged
    }

    /// Operator snapshot for /status.
    pub fn snapshot(&self) -> Value {
        let mut suspicious: Vec<(String, i32)> = self
            .scores
            .iter()
            .filter(|(_, s)| **s > 0.0)
            .map(|(id, s)| (id.clone(), (*s * 10.0) as i32))
            .collect();
        suspicious.sort_by_key(|b| std::cmp::Reverse(b.1));
        suspicious.truncate(5);
        json!({
            "dispatches_observed": self.dispatches_observed,
            "flags_raised": self.flags_raised,
            "suspicious": suspicious
                .iter()
                .map(|(id, s10)| json!({"node_id": id, "score_10x": s10}))
                .collect::<Vec<_>>(),
            "thresholds": {
                "latency_factor": LATENCY_FACTOR,
                "pool_min_peers": POOL_MIN_PEERS,
                "absent_min_selections": ABSENT_MIN_SELECTIONS,
                "flag_score_10x": (FLAG_SCORE * 10.0) as i32,
            },
        })
    }
}

fn median(lats: &[u64]) -> Option<f32> {
    if lats.is_empty() {
        return None;
    }
    let mut v: Vec<f32> = lats.iter().map(|x| *x as f32).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        Some(v[n / 2])
    } else {
        Some((v[n / 2 - 1] + v[n / 2]) / 2.0)
    }
}

fn median_f32(mut v: Vec<f32>) -> Option<f32> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(v[v.len() / 2])
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn median_f32(mut v: Vec<f32>) -> Option<f32> {
        if v.is_empty() {
            return None;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Some(v[v.len() / 2])
    }

    fn fed(d: &mut FreeriderDetector, expert: &str, candidates: &[&str], served: &str, ms: u64) {
        let cands: Vec<String> = candidates.iter().map(|s| s.to_string()).collect();
        d.feed_dispatch(expert, &cands, served, ms);
    }

    /// Signal 1 path: four fast servers frame one slow peer; repeated rounds
    /// of bad samples push the score past the flag line, honest peers stay
    /// clean.
    #[test]
    fn slow_peer_flags_after_repeated_latency_samples() {
        let mut d = FreeriderDetector::default();
        let ps: Vec<String> = (0..5).map(|i| format!("p{i}")).collect();
        let refs: Vec<&str> = ps.iter().map(|s| s.as_str()).collect();

        let mut ever_flagged = false;
        for _ in 0..8 {
            for id in &refs[..4] {
                fed(&mut d, "e", &refs, id, 10);
            }
            fed(&mut d, "e", &refs, "p4", 60);
            let flagged = d.evaluate();
            if flagged.iter().any(|x| x == "p4") {
                ever_flagged = true;
            }
        }
        assert!(ever_flagged, "slow peer flagged; scores={:?}", d.scores);
        // Honest peers carry (near-)no suspicion.
        for i in 0..4 {
            let s = d.scores.get(&format!("p{i}")).copied().unwrap_or(0.0);
            assert!(s < 1.0, "honest p{i} scored {s}");
        }
    }

    /// Guard: without a comparable pool there is NO latency signal — a lone
    /// slow peer on an expert nobody else served is unjudged.
    #[test]
    fn no_latency_signal_without_pool_context() {
        let mut d = FreeriderDetector::default();
        for _ in 0..5 {
            fed(&mut d, "e", &["peer-0", "peer-1"], "peer-0", 10);
            fed(&mut d, "e", &["peer-0", "peer-1"], "peer-1", 90);
        }
        let flagged = d.evaluate();
        assert!(!flagged.contains(&"peer-1".to_string()), "no pool ⟹ no flag");
        assert!(d.scores.get("peer-1").copied().unwrap_or(0.0) < 1.0);
    }

    /// Signal 2: a selected-but-absent peer soaks up plan slots while the
    /// pool demonstrably flows.
    #[test]
    fn selected_but_absent_flags_with_flowing_pool() {
        let mut d = FreeriderDetector::default();

        // Per round: the pool keeps flowing (peer-a answers twice), the
        // lazy peer keeps soaking up plan slots. Repeated rounds + decay
        // emulate the real cadence: the crossing is gradual, by design.
        // Per round: the pool keeps flowing (peer-a answers twice), the
        // lazy peer keeps soaking up plan slots. Repeated rounds + window
        // sliding emulate the real cadence: the crossing is gradual, by
        // design — evidence queues once 10 recent asks show the pattern.
        let mut ever_flagged = false;
        for _ in 0..20 {
            fed(&mut d, "e", &["peer-a", "peer-b"], "peer-a", 10);
            fed(&mut d, "e", &["peer-a", "peer-b"], "peer-b", 10);
            fed(&mut d, "e", &["peer-lazy", "peer-a"], "peer-a", 10);
            if d.evaluate()
                .iter()
                .any(|x| x == "peer-lazy")
            {
                ever_flagged = true;
            }
        }
        assert!(
            ever_flagged,
            "lazy peer flagged; rates={:?}",
            d.peers
                .iter()
                .map(|(k, v)| format!(
                    "{} answered={}/{} asked",
                    k,
                    v.selections.iter().filter(|a| **a).count(),
                    v.selections.len()
                ))
                .collect::<Vec<_>>()
        );
    }

    /// Honest refusals are invisible: nothing feeds on failed dispatches.
    #[test]
    fn failures_feed_nothing() {
        let mut d = FreeriderDetector::default();
        assert!(d.evaluate().is_empty());
        assert_eq!(d.dispatches_observed, 0);
    }

    /// Recovery: after a flag, sustained clean behavior decays the score
    /// under the hysteresis line, re-arming the crossing.
    #[test]
    fn decay_re_arms_after_recovery() {
        let mut d = FreeriderDetector::default();
        let ps: Vec<String> = (0..5).map(|i| format!("p{i}")).collect();
        let refs: Vec<&str> = ps.iter().map(|s| s.as_str()).collect();

        let mut ever_flagged = false;
        // 6 rounds ≈ score 18/(1−0.93·(window)) accumulates past 55.
        for _ in 0..6 {
            for id in &refs[..4] {
                fed(&mut d, "e", &refs, id, 10);
            }
            fed(&mut d, "e", &refs, "p4", 60);
            if d.evaluate().iter().any(|x| x == "p4") {
                ever_flagged = true;
            }
        }
        assert!(ever_flagged, "flag raised during rounds");
        assert!(d.flagged_once.contains("p4"));
        // Recovery phase: p4 starts answering like everyone else — its
        // recency window slides onto fast samples, both signals go silent,
        // and decay erodes the score below the hysteresis line.
        for _ in 0..10 {
            for id in &refs {
                fed(&mut d, "e", &refs, id, 10);
            }
            d.evaluate();
        }
        // Pure decay after genuine behavior change: 0.93^n → re-armed.
        for _ in 0..15 {
            d.evaluate();
        }
        assert!(
            !d.flagged_once.contains("p4"),
            "hysteresis re-arms after decay (score={:?})",
            d.scores.get("p4")
        );
    }

    /// Snapshot is operator-shaped: counters + bounded suspicious list.
    #[test]
    fn snapshot_carries_running_suspicion() {
        let mut d = FreeriderDetector::default();
        d.feed_dispatch("e", &["q1".to_string(), "q2".to_string()], "q1", 5);
        d.feed_dispatch("e", &["q1".to_string(), "q2".to_string()], "q2", 5);
        let _ = d.evaluate();
        let snap = d.snapshot();
        assert!(snap["dispatches_observed"].as_u64().unwrap() >= 2);
        assert!(snap["thresholds"].is_object());
        assert!(snap["suspicious"].is_array());
    }

    /// Medians behave on odd/even lengths and tolerate mixed values.
    #[test]
    fn median_helpers() {
        assert_eq!(median(&vec![3, 1, 2]), Some(2.0));
        // Even length interpolates (true median).
        assert_eq!(median(&vec![4, 1]), Some(2.5));
        assert_eq!(median(&vec![]), None);
        assert_eq!(median_f32(vec![]), None);
        assert_eq!(median_f32(vec![9.0, 7.0, 7.0]), Some(7.0));
    }
}
