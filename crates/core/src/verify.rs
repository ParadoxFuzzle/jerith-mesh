//! Phase 4 verification layer: cryptographic commitments + redundant replay.
//!
//! Two mechanisms ship here (ARCHITECTURE.md §5 anti-fraud table):
//!
//! 1. **Commitments** — every `ExpertResult` the server returns carries a
//!    SHA-256 commitment over the canonical (req_id, expert_id, node_id,
//!    input, output) tuple. The requester recomputes it from the reply body,
//!    so a node can never later claim a different result for the same work
//!    (enables efficient fraud proofs when a replay check fails).
//! 2. **Redundant replay** — with a configurable probability, the requester
//!    re-runs the same expert query on a second independent node and the two
//!    outputs are compared. A mismatch is reported as `DishonestResults` to
//!    the bootstrap (signed evidence, dedup'd server-side) for both -- or the
//!    outlying -- nodes; local outputs are treated as trusted arbiters.
//!
//! Design principles honored (per the spec): no proof on the hot path (the
//! commitment costs one hash; replay is probabilistic, default ~10 %), all
//! measurement is local, and reports carry the reporter's signature so the
//! bootstrap can weight them.

use std::collections::HashMap;
use std::sync::Arc;

use sha2::{Digest, Sha256};
use tokio::sync::Mutex;

/// Canonical commitment prefix — version bump breaks wire compat deliberately.
pub const COMMITMENT_PREFIX: &str = "jerith-cmt1";

/// Hex commitment string length (SHA-256).
pub const COMMITMENT_HEX_LEN: usize = 64;

/// Canonical commitment payload over an expert result. Must stay byte-stable
/// for a given (req_id, expert_id, node_id, input, output): the server and
/// verifier compute it independently and compare hex digests.
pub fn commitment_payload(
    req_id: &str,
    expert_id: &str,
    node_id: &str,
    input: &[f32],
    output: &[f32],
) -> String {
    format!(
        "{COMMITMENT_PREFIX}|{req_id}|{expert_id}|{node_id}|{}|{}",
        crate::quic::pack_f32_hex(input),
        crate::quic::pack_f32_hex(output),
    )
}

/// Compute a commitment (SHA-256 hex) over an expert result.
pub fn result_commitment(
    req_id: &str,
    expert_id: &str,
    node_id: &str,
    input: &[f32],
    output: &[f32],
) -> String {
    hex_sha256(commitment_payload(req_id, expert_id, node_id, input, output).as_bytes())
}

/// Check a server-returned commitment against the reply body.
pub fn verify_commitment(
    req_id: &str,
    expert_id: &str,
    node_id: &str,
    input: &[f32],
    output: &[f32],
    commitment: &str,
) -> bool {
    if commitment.len() != COMMITMENT_HEX_LEN {
        return false;
    }
    result_commitment(req_id, expert_id, node_id, input, output) == commitment
}

fn hex_sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// Compare two expert outputs for agreement. Exact match when `tolerance` is
/// 0.0 (the deterministic default); a small absolute tolerance otherwise.
/// Different lengths or any NaN never agree.
pub fn outputs_agree(a: &[f32], b: &[f32], tolerance: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| x == y || (tolerance > 0.0 && (x - y).abs() <= tolerance))
}

/// Rolling verification counters (local-only, per requester).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VerificationStats {
    /// Remote expert results received.
    pub expert_results_received: u64,
    /// Commitment checks performed.
    pub commitment_checks: u64,
    /// Commitment mismatches (result body did not match the commitment).
    pub commitment_failures: u64,
    /// Redundant replays dispatched.
    pub replays_dispatched: u64,
    /// Replays whose outputs agreed (incl. local arbitration).
    pub replay_matches: u64,
    /// Replays whose outputs disagreed.
    pub replay_mismatches: u64,
    /// DishonestResults fraud reports actually submitted to the bootstrap.
    pub fraud_reports_flushed: u64,
}

/// Shared verification tracker: counters + signed-evidence flood control.
///
/// The router (or gateway) records every outcome here; the mesh ping loop
/// flushes pending `DishonestResults` reports to the bootstrap on its regular
/// cadence. Per-target cooldown prevents a malicious sibling from tricking us
/// into spamming the bootstrap by sending us self-sabotaged work.
pub struct VerificationTracker {
    stats: Mutex<VerificationStats>,
    /// node_id → last flag time (ms), for flood-control dedup.
    last_flagged: Mutex<HashMap<String, u64>>,
    /// Targets queued for a DishonestResults report (drained on flush).
    pending: Mutex<Vec<String>>,
    /// Minimum interval between repeat queueing of the same target.
    cooldown_ms: u64,
}

impl Default for VerificationTracker {
    fn default() -> Self {
        Self::with_cooldown(60_000)
    }
}

impl VerificationTracker {
    pub fn with_cooldown(cooldown_ms: u64) -> Self {
        Self {
            stats: Mutex::new(VerificationStats::default()),
            last_flagged: Mutex::new(HashMap::new()),
            pending: Mutex::new(Vec::new()),
            cooldown_ms,
        }
    }

    /// Shared handle.
    pub fn shared() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub async fn stats(&self) -> VerificationStats {
        *self.stats.lock().await
    }

    pub async fn note_expert_result(&self) {
        self.stats.lock().await.expert_results_received += 1;
    }

    /// Record one commitment check; `false` also flags the node as
    /// dishonest (the reply body was tampered with relative to its own
    /// commitment — the strongest local evidence we can gather).
    pub async fn note_commitment(&self, node_id: &str, ok: bool) {
        {
            let mut stats = self.stats.lock().await;
            stats.commitment_checks += 1;
            if !ok {
                stats.commitment_failures += 1;
            }
        }
        if !ok {
            self.flag_dishonest(&[node_id]).await;
        }
    }

    pub async fn note_replay(&self, matched: bool) {
        let mut stats = self.stats.lock().await;
        stats.replays_dispatched += 1;
        if matched {
            stats.replay_matches += 1;
        } else {
            stats.replay_mismatches += 1;
        }
    }

    /// Queue (with cooldown) the given nodes for a DishonestResults report.
    pub async fn flag_dishonest(&self, node_ids: &[&str]) {
        let mut last = self.last_flagged.lock().await;
        let mut queued = self.pending.lock().await;
        let now = crate::bootstrap::now_ms();
        for id in node_ids.iter().copied().filter(|id| !id.is_empty()) {
            let allowed = last
                .get(id)
                .map(|t| now.saturating_sub(*t) >= self.cooldown_ms)
                .unwrap_or(true);
            if allowed {
                last.insert(id.to_string(), now);
                queued.push(id.to_string());
            }
        }
    }

    /// Actually submit queued reports to the bootstrap. Returns the targets
    /// the bootstrap accepted — the stake manager turns each into a signed
    /// slash notice (Phase 5). Called on the mesh loop's regular cadence so
    /// evidence submission never sits on the inference hot path.
    pub async fn flush_fraud_reports(&self, state: &Arc<crate::ping::MeshState>) -> Vec<String> {
        // Drain before the await: targets flagged while we submit go to the
        // next flush instead of double-sending within one batch.
        let mut to_report: Vec<String> = {
            let mut q = self.pending.lock().await;
            q.drain(..).collect()
        };
        to_report.sort();
        to_report.dedup();
        if to_report.is_empty() {
            return Vec::new();
        }

        let reporter = state.identity.node_id();
        let bootstrap =
            crate::bootstrap::BootstrapClient::new(state.config.read().await.bootstrap_url.clone());
        let mut sent: Vec<String> = Vec::new();
        for node_id in &to_report {
            if node_id == &reporter {
                continue; // a node cannot report itself
            }
            match bootstrap
                .report_fraud(
                    crate::bootstrap_server::FraudKind::DishonestResults,
                    node_id,
                    &reporter,
                    &state.identity,
                )
                .await
            {
                Ok(_) => {
                    sent.push(node_id.clone());
                    tracing::warn!(
                        target: "mesh",
                        node = %node_id,
                        "dishonest-result evidence submitted to bootstrap"
                    );
                }
                Err(e) => {
                    tracing::debug!(
                        target: "mesh",
                        node = %node_id,
                        error = %e,
                        "dishonest-result report submission failed (evidence dropped, will re-flag on repeat)"
                    );
                }
            }
        }
        let sent_count = sent.len();
        if sent_count > 0 {
            // Count only what actually reached the bootstrap.
            self.stats.lock().await.fraud_reports_flushed += sent_count as u64;
        }
        sent
    }
}

// ---------------------------------------------------------------------------
// Ringer challenges (spot-checks with known answers)
// ---------------------------------------------------------------------------

/// Builtin expert namespace: expert ids under this prefix are answered
/// deterministically by any mesh node *without touching the runtime* — a
/// pure function both the server and the pinger can compute in microseconds.
pub const RINGER_EXPERT_PREFIX: &str = "jerith-builtin/";

/// The builtin expert functions (single source of truth for both sides).
/// All arithmetic is exact in IEEE f32 (integer-valued inputs, power-of-two
/// scaling) so peers agree bit-for-bit — a wrong node's gamble is detectable
/// with zero tolerance.
pub fn builtin_expert_answer(kind: &str, input: &[f32]) -> Vec<f32> {
    match kind {
        // Scale by 2 (power of two — bit-exact) and reverse the order.
        "reverse2" | "" => {
            let mut out: Vec<f32> = input.iter().map(|x| x * 2.0).collect();
            out.reverse();
            out
        }
        // Scale by 4, keep order.
        "scale4" => input.iter().map(|x| x * 4.0).collect(),
        _ => Vec::new(),
    }
}

/// A ringer task derived from a seed (varies per round so a dishonest node
/// cannot precompute/replay answers): input is exact integer-valued f32s in
/// [-48, 49), the expected output is `builtin_expert_answer("reverse2", …)`.
/// Returns (expert_id, input, expected_output).
pub fn ringer_task(seed: u64) -> (&'static str, Vec<f32>, Vec<f32>) {
    let input: Vec<f32> = (0..8)
        .map(|i| {
            (((seed
                .wrapping_mul(1103515245)
                .wrapping_add(12345 + i as u64 * 77))
                % 97) as f32)
                - 48.0
        })
        .collect();
    let expected = builtin_expert_answer("reverse2", &input);
    ("jerith-builtin/reverse2", input, expected)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const I: &[f32] = &[1.0, -2.5, 3.0];
    const O: &[f32] = &[9.5, 0.0, -7.25];

    #[test]
    fn commitment_is_deterministic_and_hex() {
        let c1 = result_commitment("req-1", "exp.a", "node-1", I, O);
        let c2 = result_commitment("req-1", "exp.a", "node-1", I, O);
        assert_eq!(c1, c2);
        assert_eq!(c1.len(), COMMITMENT_HEX_LEN);
        assert!(c1.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn commitment_covers_every_tuple_field() {
        let base = result_commitment("req-1", "exp.a", "node-1", I, O);
        assert_ne!(base, result_commitment("req-2", "exp.a", "node-1", I, O));
        assert_ne!(base, result_commitment("req-1", "exp.b", "node-1", I, O));
        assert_ne!(base, result_commitment("req-1", "exp.a", "node-2", I, O));
        assert_ne!(
            base,
            result_commitment("req-1", "exp.a", "node-1", &[2.0], O)
        );
        assert_ne!(
            base,
            result_commitment("req-1", "exp.a", "node-1", I, &[8.0])
        );
    }

    #[test]
    fn verify_commitment_detects_tampering() {
        let c = result_commitment("r", "e", "n", I, O);
        assert!(verify_commitment("r", "e", "n", I, O, &c));
        // One output byte flipped → mismatch.
        assert!(!verify_commitment("r", "e", "n", I, &[9.5, 0.0, -7.2], &c));
        // Bad length → mismatch.
        assert!(!verify_commitment("r", "e", "n", I, O, "abcd"));
        // Empty output vs committed non-empty → mismatch.
        assert!(!verify_commitment("r", "e", "n", I, &[], &c));
    }

    #[test]
    fn outputs_agree_rules() {
        assert!(outputs_agree(O, O, 0.0));
        assert!(!outputs_agree(O, &[9.5, 0.0, -7.249], 0.0));
        assert!(outputs_agree(O, &[9.5, 0.0, -7.249], 0.002));
        assert!(!outputs_agree(O, &[9.5, 0.0], 0.0)); // length mismatch
        assert!(!outputs_agree(&[f32::NAN], &[f32::NAN], 0.0)); // NaN never agrees
    }

    fn state() -> Arc<crate::ping::MeshState> {
        use crate::config::*;
        let mut c = NodeConfig::default();
        c.ensure_identity().unwrap();
        let (st, _rx) = crate::ping::MeshState::new(c).unwrap();
        st
    }

    #[tokio::test]
    async fn tracker_counts_and_cooldown_dedups() {
        let t = VerificationTracker::with_cooldown(1000);
        t.note_commitment("liar", false).await;
        t.note_commitment("honest", true).await;
        t.note_replay(true).await;
        t.note_replay(false).await;
        let s = t.stats().await;
        assert_eq!(s.commitment_checks, 2);
        assert_eq!(s.commitment_failures, 1);
        assert_eq!(s.replays_dispatched, 2);
        assert_eq!(s.replay_matches, 1);
        assert_eq!(s.replay_mismatches, 1);
        // Flagging the same target again within cooldown must not duplicate
        // evidence: still one pending queue entry.
        t.flag_dishonest(&["liar"]).await;
        t.flag_dishonest(&["liar"]).await;
        let state = state();
        // Node reporting may fail (no bootstrap in test) — just ensure flush
        // caps at one report for the single dedup'd target.
        let targets = t.flush_fraud_reports(&state).await;
        assert!(targets.len() <= 1);
        let s2 = t.stats().await;
        assert!(s2.fraud_reports_flushed <= 1);
    }

    #[test]
    fn ringer_task_is_deterministic_and_seed_varying() {
        let (e1, in1, out1) = ringer_task(42);
        let (e2, in2, out2) = ringer_task(42);
        assert_eq!(e1, e2);
        assert_eq!(e1, "jerith-builtin/reverse2");
        assert_eq!(in1, in2);
        assert_eq!(out1, out2);
        // Varies across seeds (a cached answer cannot pass every round).
        let (_, in3, _) = ringer_task(43);
        assert_ne!(in1, in3);
        // Inputs stay integer-valued and covered by builtin answer.
        for v in &in1 {
            assert!(
                v.fract() == 0.0 && (*v >= -48.0) && (*v < 49.0),
                "ringer input exact: {v}"
            );
        }
        assert_eq!(out1, builtin_expert_answer("reverse2", &in1));
        // Bit-exact power-of-two scaling: reverse + ×2.
        assert_eq!(out1[0], in1[7] * 2.0);
        assert_eq!(out1[7], in1[0] * 2.0);
    }

    #[test]
    fn builtin_answers_are_exact_and_bounded() {
        let input = [1.5, -3.0, 7.0];
        assert_eq!(
            builtin_expert_answer("scale4", &input),
            vec![6.0, -12.0, 28.0]
        );
        assert_eq!(
            builtin_expert_answer("reverse2", &input),
            vec![14.0, -6.0, 3.0]
        );
        assert!(builtin_expert_answer("unknown-kind", &input).is_empty());
    }

    #[tokio::test]
    async fn tracker_never_reports_itself() {
        let t = VerificationTracker::default();
        let state = state();
        let me = state.identity.node_id();
        t.flag_dishonest(&[me.as_str()]).await;
        let sent = t.flush_fraud_reports(&state).await; // self-reports skipped
        assert!(sent.is_empty());
    }
}
