//! Multi-hop pipeline route planning (Phase 3 — FEATURES §13's first
//! milestone, M0).
//!
//! What already exists in earlier phases: experts as real weights
//! (`expert_store`), the deterministic layer-ownership ring
//! (`replication::ShardMap::layer_ownership`), and expert dispatch plans.
//! What a Petals-style pipeline ultimately needs is route planning over
//! *layer groups*: an activation walks a deterministic sequence of hops,
//! each owned by exactly one node from the ring, and the runtime later
//! ships residual streams along that route (M1+; see ARCHITECTURE.md).
//!
//! M0: pure route planning from the ring's full layer map —
//! no I/O, no runtime, no clocks: deterministic, unit-testable, and
//! naturally degradable (a single hop when only the local node is alive).
//!
//! M0 + M1 + M2 (this module): pure route planning from the ring's full layer
//! map, the route executor that walks it, and the degradation layer that
//! keeps routes alive when an owner dies mid-walk. Planning is I/O-free and
//! clock-free (deterministic, unit-testable, naturally degradable); the
//! executor is the M1 hop runtime over the same f32-vector wire contract
//! expert dispatch already uses, with the Phase 4 commitment recomputed
//! *per hop* (each hop's expert id names its layer run, so a tampering hop
//! is caught downstream exactly like a single-expert reply) and per-hop
//! deadlines derived from the owner's measured RTT.
//!
//! M2 adds the degradation layer: a dead hop's layers are re-hosted per the
//! ring's own rule (dropping the departed owner re-pins only its layers),
//! unrecoverable losses produce honest hops-short partial answers with a
//! typed reason (the router walks on, mirroring reject-continue), and
//! flapping owners are dampened by a freerider-style two-line hysteresis
//! so a bouncing node cannot chatter the route plan.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::time::Duration;

/// One pipeline hop: the node that owns this run of consecutive layers
/// (`None` = the local node's own slice), together with those layers in
/// ascending order. Hops are ordered by ascending layer index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineHop {
    /// Owner's node id; `None` marks the local node's own slice.
    pub node: Option<String>,
    /// The consecutive layers this hop executes (ascending).
    pub layers: Vec<usize>,
}

/// Build the pipeline route for the ring's full layer ownership map
/// (layers ascending — the caller's model walk order, which a BTreeMap
/// preserves). Consecutive layers with the same owner merge into one hop;
/// the local slice appears as `node: None`.
///
/// The map is the full one from
/// [`ShardMap::layer_ownership`](crate::replication::ShardMap::layer_ownership)
/// — restricting it to self-owned layers left every downstream planner's
/// remote path unreachable (the pre-M0 defect this module replaces with a
/// complete picture).
pub fn plan_route(self_node_id: &str, ownership: &BTreeMap<usize, String>) -> Vec<PipelineHop> {
    let mut hops: Vec<PipelineHop> = Vec::new();
    for (&layer, owner) in ownership {
        let node = if owner == self_node_id {
            None
        } else {
            Some(owner.clone())
        };
        match hops.last_mut() {
            Some(last) if last.node == node => last.layers.push(layer),
            _ => hops.push(PipelineHop {
                node,
                layers: vec![layer],
            }),
        }
    }
    hops
}

/// Stable per-hop expert id: the model id plus the hop's layer run as a
/// closed range. The commitment layer canonicalizes over
/// (req_id, expert_id, node_id, input, output), so a *deterministic expert
/// id per hop* is what makes per-hop commitments composable rather than
/// ad-hoc — the same expert-shaped tuple every single-expert reply uses.
pub fn hop_expert_id(model_id: &str, hop: &PipelineHop) -> String {
    let first = hop.layers.first().copied().unwrap_or(0);
    let last = hop.layers.last().copied().unwrap_or(first);
    format!("{model_id}#{first}-{last}")
}

/// Per-hop deadline: the dispatch base plus 3× the owner's measured RTT
/// (round-trip plus the run plus generous jitter headroom — the same
/// conservative multiplier the mesh uses for one-expert dispatch).
pub fn hop_timeout(base: Duration, rtt_ms: u64) -> Duration {
    let extra = rtt_ms
        .saturating_mul(3)
        .saturating_mul(1_000_000)
        .try_into()
        .unwrap_or(u64::MAX);
    base.saturating_add(Duration::from_nanos(extra))
}

/// Typed failure of a route walk: which hop failed, who owned it, which
/// layers it covered, why, and how much of the route completed before the
/// failure (M2's partial-answer semantics build on that count).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteError {
    pub hop_index: usize,
    pub node: Option<String>,
    pub layers: Vec<usize>,
    pub reason: String,
    pub hops_completed: usize,
}

impl std::fmt::Display for RouteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let owner = self.node.clone().unwrap_or_else(|| "local".into());
        write!(
            f,
            "pipeline hop {}/{owner} (layers {:?}) after {} completed hops: {}",
            self.hop_index, self.layers, self.hops_completed, self.reason
        )
    }
}

impl std::error::Error for RouteError {}

/// One hop attempt shared by both walks: local executor, remote executor
/// under its RTT-derived deadline, plus the Phase 4 commitment recomputed
/// against the hop's own expert id (a tampering hop reads exactly like a
/// single-expert commitment mismatch).
async fn attempt_one_hop(
    model_id: &str,
    req_id: &str,
    hop: &PipelineHop,
    activation: &[f32],
    local_run: &impl Fn(&[f32], &[usize]) -> anyhow::Result<Vec<f32>>,
    remote_run: &impl Fn(
        &str,
        &str,
        Vec<f32>,
        Duration,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<(Vec<f32>, Option<String>)>> + Send>>,
    rtt_of: &impl Fn(&str) -> u64,
    base_timeout: Duration,
) -> anyhow::Result<Vec<f32>> {
    let expert_id = hop_expert_id(model_id, hop);
    match &hop.node {
        None => local_run(activation, &hop.layers),
        Some(node) => {
            let activation = activation.to_vec();
            let (out, commitment) = remote_run(
                &expert_id,
                node,
                activation.clone(),
                hop_timeout(base_timeout, rtt_of(node)),
            )
            .await?;
            match commitment {
                Some(c)
                    if crate::verify::verify_commitment(
                        req_id,
                        &expert_id,
                        node,
                        &activation,
                        &out,
                        &c,
                    ) =>
                {
                    Ok(out)
                }
                Some(_) => {
                    Err(anyhow::anyhow!("commitment mismatch from hop {node}: body altered"))
                }
                None => Ok(out),
            }
        }
    }
}

/// Walk [`plan_route`]'s sequence hop-by-hop, carrying the residual stream:
/// each hop's output becomes the next hop's input. Composition mirrors the
/// existing single-expert contract —
///
/// - **Local hop** (`node: None`) runs [`crate::expert_store::Expert::forward`]
///   shape via `local_run`, which receives the current activation and the
///   hop's layers (real weights arrive with the Colibri-side layer split;
///   until then synthetic/moe experts fill the role).
/// - **Remote hop** runs `remote_run` — the same signed expert RPC — whose
///   `Duration` argument is [`hop_timeout`] from the owner's RTT (the
///   closure owns applying it, as `execute`'s dispatch closures already do).
///   It returns the output plus the server's Phase 4 commitment, which the
///   executor re-verifies here against (req_id, *per-hop* expert id, owner,
///   in, out): a hop's commitment names its own layer run, so a tampering
///   hop is caught downstream exactly like a single-expert reply.
/// - Any failure aborts the route with a typed [`RouteError`] — no silent
///   skipping. The degradation lives in
///   [`execute_route_resilient`], which reassigns and walks on instead.
pub async fn execute_route(
    model_id: &str,
    req_id: &str,
    route: &[PipelineHop],
    mut activation: Vec<f32>,
    local_run: impl Fn(&[f32], &[usize]) -> anyhow::Result<Vec<f32>>,
    remote_run: impl Fn(
        &str,
        &str,
        Vec<f32>,
        Duration,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<(Vec<f32>, Option<String>)>> + Send>>,
    rtt_of: impl Fn(&str) -> u64,
    base_timeout: Duration,
) -> Result<Vec<f32>, RouteError> {
    for (idx, hop) in route.iter().enumerate() {
        match attempt_one_hop(model_id, req_id, hop, &activation, &local_run, &remote_run, &rtt_of, base_timeout)
            .await
        {
            Ok(out) => activation = out,
            Err(e) => {
                return Err(RouteError {
                    hop_index: idx,
                    node: hop.node.clone(),
                    layers: hop.layers.clone(),
                    reason: e.to_string(),
                    hops_completed: idx,
                });
            }
        }
    }
    Ok(activation)
}

/// Cap on route surgery per walk: how many times a failed hop's layers may
/// be re-hosted before the executor gives up and degrades honestly. A
/// replacement that itself fails can be re-reassigned once more — but the
/// walk can never spin (the budget sums what the route plus a bounded
/// headroom allows).
pub const REASSIGN_BUDGET: usize = 12;

/// Layers an unrecovered hop left undone (honest hops-short answer).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DegradedHop {
    pub layers: Vec<usize>,
    pub node: Option<String>,
    pub reason: String,
}

/// Layers a dead owner's hop gave up, re-hosted via the ring's own rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReassignedHop {
    pub layers: Vec<usize>,
    pub was_node: Option<String>,
    pub reason: String,
}

/// Result of a perhaps-degraded route walk. Never an error unless the walk
/// truly cannot answer: a degraded walk returns the residual as far as the
/// chain carried it, in [`RouteResult::activation`], with the typing in
/// [`RouteResult::degraded`] (skipped) and [`RouteResult::reassigned`]
/// (re-hosted). Mirrors reject-continue — the router walks on.
#[derive(Debug, Clone)]
pub struct RouteResult {
    pub activation: Vec<f32>,
    pub hops_run: usize,
    pub degraded: Vec<DegradedHop>,
    pub reassigned: Vec<ReassignedHop>,
}

impl RouteResult {
    pub fn is_complete(&self) -> bool {
        self.degraded.is_empty()
    }
}

/// Availability hysteresis for route planning: the freerider module's
/// decay-then-flag shape, pointed at liveness instead of honesty. A node's
/// failure score rises on failures and decays on successes; crossing
/// [`HOP_EXCLUDE_SCORE`] excludes it from route planning, and it returns
/// only after decay below [`HOP_CLEAR_SCORE`] — one line to raise, a lower
/// one to clear, so a flapping peer cannot chatter the plan.
pub const HOP_EXCLUDE_SCORE: u32 = 3;
pub const HOP_CLEAR_SCORE: u32 = 1;

#[derive(Debug, Clone, Default)]
pub struct RouteStability {
    scores: HashMap<String, u32>,
    excluded: HashSet<String>,
}

impl RouteStability {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn note_failure(&mut self, node: &str) {
        let s = self.scores.entry(node.to_string()).or_insert(0);
        *s = s.saturating_add(1);
        if *s >= HOP_EXCLUDE_SCORE {
            self.excluded.insert(node.to_string());
        }
    }

    pub fn note_success(&mut self, node: &str) {
        if let Some(s) = self.scores.get_mut(node) {
            *s = s.saturating_sub(1);
            if *s <= HOP_CLEAR_SCORE {
                self.excluded.remove(node);
            }
        }
    }

    pub fn is_excluded(&self, node: &str) -> bool {
        self.excluded.contains(node)
    }

    /// Alive set minus excluded owners — never empty: a mesh whose every
    /// owner flapped still routes (better a flaky hop than no hop at all;
    /// the committed-replay layers catch a dishonest node separately).
    pub fn filter_alive(&self, alive: &[String]) -> Vec<String> {
        let kept: Vec<String> = alive
            .iter()
            .filter(|a| !self.excluded.contains(*a))
            .cloned()
            .collect();
        if kept.is_empty() {
            alive.to_vec()
        } else {
            kept
        }
    }
}

/// The resilient walk over [`plan_route`]'s sequence: like [`execute_route`]
/// but a failure need not kill the answer —
///
/// - **Dead-hop reassignment (the ring's own rule):** the failed hop's
///   layers go to `reassign(layers, dead_owner)`, which the caller wires to
///   [`ShardMap::layer_ownership`](crate::replication::ShardMap::layer_ownership)
///   over the alive set minus the dead owner — dropping a node re-pins
///   only its layers, never the whole route. The replacement sub-hops are
///   walked in order (front of the queue), bounded by [`REASSIGN_BUDGET`].
/// - **Honest parity between hops:** every failure and success feeds
///   `stability` (`note_*`), so repeated deaths exclude a flapping owner
///   from *future* route plans without this walk's outcomed changing.
/// - **Partial answers:** when `reassign` yields nothing (nobody alive to
///   take the layers, or the caller refuses), the hop's layers are skipped
///   and typed into [`RouteResult::degraded`] — the router walks on and the
///   caller receives the residual as far as the chain carried it, with the
///   lost layers named.
pub async fn execute_route_resilient(
    model_id: &str,
    req_id: &str,
    route: Vec<PipelineHop>,
    mut activation: Vec<f32>,
    local_run: impl Fn(&[f32], &[usize]) -> anyhow::Result<Vec<f32>>, 
    remote_run: impl Fn(
        &str,
        &str,
        Vec<f32>,
        Duration,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<(Vec<f32>, Option<String>)>> + Send>>,
    rtt_of: impl Fn(&str) -> u64,
    base_timeout: Duration,
    stability: &mut RouteStability,
    reassign: impl Fn(&[usize], Option<&str>) -> Option<Vec<PipelineHop>>,
) -> RouteResult {
    let mut pending: VecDeque<PipelineHop> = route.into();
    let mut budget = REASSIGN_BUDGET.saturating_add(pending.len());
    let mut degraded: Vec<DegradedHop> = Vec::new();
    let mut reassigned: Vec<ReassignedHop> = Vec::new();
    let mut hops_run: usize = 0;

    while let Some(hop) = pending.pop_front() {
        match attempt_one_hop(model_id, req_id, &hop, &activation, &local_run, &remote_run, &rtt_of, base_timeout)
            .await
        {
            Ok(out) => {
                activation = out;
                hops_run += 1;
                if let Some(node) = &hop.node {
                    stability.note_success(node);
                }
            }
            Err(e) => {
                let reason = e.to_string();
                if let Some(node) = &hop.node {
                    stability.note_failure(node);
                }
                let replacement = if budget > 0 {
                    reassign(&hop.layers, hop.node.as_deref())
                } else {
                    None
                };
                match replacement {
                    Some(sub) if !sub.is_empty() => {
                        budget -= 1;
                        reassigned.push(ReassignedHop {
                            layers: hop.layers.clone(),
                            was_node: hop.node.clone(),
                            reason: reason.clone(),
                        });
                        for replacement_hop in sub.into_iter().rev() {
                            pending.push_front(replacement_hop);
                        }
                    }
                    _ => {
                        degraded.push(DegradedHop {
                            layers: hop.layers.clone(),
                            node: hop.node.clone(),
                            reason: format!("skipped: {reason}"),
                        });
                    }
                }
            }
        }
    }
    RouteResult { activation, hops_run, degraded, reassigned }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_store::ExpertWeights;
    use crate::verify::{result_commitment, verify_commitment};
    use std::time::Duration;

    const BASE: Duration = Duration::from_millis(500);

    fn map(entries: &[(usize, &str)]) -> BTreeMap<usize, String> {
        entries
            .iter()
            .map(|(l, o)| (*l, o.to_string()))
            .collect()
    }

    #[test]
    fn consecutive_same_owner_layers_merge_into_one_hop() {
        let m = map(&[(0, "a"), (1, "a"), (2, "b"), (3, "b"), (4, "c")]);
        let route = plan_route("self", &m);
        assert_eq!(
            route,
            vec![
                PipelineHop {
                    node: Some("a".into()),
                    layers: vec![0, 1]
                },
                PipelineHop {
                    node: Some("b".into()),
                    layers: vec![2, 3]
                },
                PipelineHop {
                    node: Some("c".into()),
                    layers: vec![4]
                },
            ]
        );
    }

    #[test]
    fn local_slices_are_none_hops_and_merge() {
        let m = map(&[(0, "self"), (1, "self"), (2, "other")]);
        let route = plan_route("self", &m);
        assert_eq!(
            route,
            vec![
                PipelineHop {
                    node: None,
                    layers: vec![0, 1]
                },
                PipelineHop {
                    node: Some("other".into()),
                    layers: vec![2]
                },
            ]
        );
    }

    #[test]
    fn all_local_is_one_hop_and_empty_is_empty() {
        let m = map(&[(0, "self"), (1, "self")]);
        let route = plan_route("self", &m);
        assert_eq!(route.len(), 1);
        assert_eq!(route[0].node, None);
        assert!(plan_route("self", &BTreeMap::new()).is_empty());
    }

    #[test]
    fn remote_start_hands_off_immediately() {
        // A route this node does not start still yields the full ordered
        // walk — M1's executor decides whether the local node initiates it
        // or receives it from a peer.
        let m = map(&[(0, "a"), (1, "self"), (2, "a")]);
        let route = plan_route("self", &m);
        assert_eq!(
            route
                .iter()
                .map(|h| h.node.clone().unwrap_or_else(|| "self".into()))
                .collect::<Vec<_>>(),
            vec!["a", "self", "a"],
            "non-contiguous owners must NOT merge: {route:?}"
        );
        assert_eq!(route[0].node.as_deref(), Some("a"));
        assert_eq!(route[0].layers, vec![0]);
    }

    // ---- M1: route execution ----

    /// One hop of a synthetic route; every hop is one synthetic expert
    /// forward keyed by the per-hop expert id (the real layer split arrives
    /// with the Colibri-side work — the executor is weights-agnostic).
    fn run_hop(expert_id: &str, x: &[f32]) -> anyhow::Result<Vec<f32>> {
        ExpertWeights::synthetic(expert_id, 8, 16).forward(x)
    }

    /// The same deterministic expert run a live peer would execute, wrapped
    /// like [`remote_run`]'s contract and returning a REAL Phase 4
    /// commitment over (req_id, per-hop expert id, node, in, out).
    fn remote_ok(
        req_id: &'static str,
    ) -> impl Fn(
        &str,
        &str,
        Vec<f32>,
        Duration,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<(Vec<f32>, Option<String>)>> + Send>> {
        move |expert_id, node, input, _timeout| {
            let expert_id = expert_id.to_owned();
            let node = node.to_owned();
            let fut = async move {
                run_hop(&expert_id, &input).map(|out| {
                    let c = result_commitment(req_id, &expert_id, &node, &input, &out);
                    (out, Some(c))
                })
            };
            Box::pin(fut)
        }
    }

    #[tokio::test]
    async fn route_walks_all_hops_and_final_vector_matches_serial_run() {
        let model = "moe-v1";
        let req = "req-1";
        let m = map(&[(0, "self"), (1, "peer-a"), (2, "self"), (3, "peer-b")]);
        let route = plan_route("self", &m);
        assert_eq!(route.len(), 4, "alternating owners never merge");

        let input = vec![0.5_f32; 8];
        let out = execute_route(
            model,
            req,
            &route,
            input.clone(),
            |a, layers| {
                // The local slice is one layer-run expert.
                run_hop(&hop_expert_id(model, &PipelineHop { node: None, layers: layers.to_vec() }), a)
            },
            remote_ok(req),
            |_| 40, // 40 ms peer RTT
            BASE,
        )
        .await
        .expect("every hop alive");

        // Serial reference run: same expert ids, same order, chained.
        let mut expect = input;
        for hop in &route {
            expect = run_hop(&hop_expert_id(model, hop), &expect).unwrap();
        }
        assert_eq!(out, expect, "residual stream must chain hop outputs");
    }

    #[tokio::test]
    async fn tampered_remote_hop_is_caught_by_the_per_hop_commitment() {
        // Middle hop computes a commitment over the HONEST body, then
        // mutates the body it actually ships — the per-hop recomputation
        // must reject the mismatch and abort the route, naming the owner.
        let model = "moe-v1";
        let req = "req-2";
        let m = map(&[(0, "self"), (1, "peer-a"), (2, "peer-b")]);
        let route = plan_route("self", &m);

        let mut input = vec![0.25_f32; 8];
        input[0] = 1.0;
        let err = execute_route(
            model,
            req,
            &route,
            input,
            |a, layers| {
                run_hop(
                    &hop_expert_id(model, &PipelineHop { node: None, layers: layers.to_vec() }),
                    a,
                )
            },
            move |expert_id, node, activation, _t| {
                let expert_id = expert_id.to_owned();
                let node = node.to_owned();
                let tamper = node == "peer-a";
                Box::pin(async move {
                    let mut out = run_hop(&expert_id, &activation)?;
                    if tamper {
                        let honest = out.clone();
                        let c = result_commitment(req, &expert_id, &node, &activation, &honest);
                        out[1] += 0.5; // ship body ≠ committed body
                        return Ok((out, Some(c)));
                    }
                    let c = result_commitment(req, &expert_id, &node, &activation, &out);
                    Ok((out, Some(c)))
                })
            },
            |_| 20,
            BASE,
        )
        .await
        .expect_err("tampered hop aborts the route");
        let RouteError { hop_index, node, reason, hops_completed, .. } = &err;
        assert_eq!(*hop_index, 1, "the tampering middle hop faulted");
        assert_eq!(node.as_deref(), Some("peer-a"));
        assert_eq!(*hops_completed, 1, "local hop 0 completed first");
        assert!(reason.contains("commitment mismatch"), "typed reason: {reason}");
    }

    #[tokio::test]
    async fn honest_remote_hop_commitment_passes_and_tampering_fails() {
        let model = "x";
        let req = "r";
        let input = vec![0.5_f32; 8];
        let route = vec![PipelineHop {
            node: Some("peer-a".into()),
            layers: vec![7],
        }];
        // Honest case passes.
        let ok = execute_route(model, req, &route, input.clone(),
            |a, _| run_hop("fallback", a),
            remote_ok(req), |_| 15, BASE)
            .await.expect("honest commitment verifies");
        assert_eq!(ok, run_hop(&hop_expert_id(model, &route[0]), &input).unwrap());

        // Tampered output with commitment over the ORIGINAL body → mismatch.
        let variant: Vec<f32> = ok.iter().map(|v| v * 2.0).collect();
        let c = result_commitment(req, &hop_expert_id(model, &route[0]), "peer-a", &input, &ok);
        assert!(!verify_commitment(req, &hop_expert_id(model, &route[0]), "peer-a", &input, &variant, &c),
            "altered body must fail the per-hop commitment");
    }

    #[tokio::test]
    async fn remote_failure_aborts_with_typed_route_error() {
        let route = vec![
            PipelineHop { node: Some("a".into()), layers: vec![0] },
            PipelineHop { node: Some("b".into()), layers: vec![1] },
            PipelineHop { node: Some("c".into()), layers: vec![2] },
        ];
        let err = execute_route(
            "m", "req", &route, vec![0.0_f32; 4],
            |a, _| run_hop("m#local", a),
            |_e, node, _x, _t| {
                let dead = node == "b";
                Box::pin(async move {
                    if dead {
                        anyhow::bail!("peer unreachable")
                    }
                    Ok((vec![0.0_f32; 4], None))
                })
            },
            |_| 30, BASE,
        )
        .await.expect_err("dead hop aborts");
        let RouteError { hop_index, node, layers, reason, hops_completed } = &err;
        assert_eq!(*hop_index, 1);
        assert_eq!(node.as_deref(), Some("b"));
        assert_eq!(layers.as_slice(), &[1_usize][..]);
        assert_eq!(*hops_completed, 1, "hop 0 completed before the failure");
        assert_eq!(reason, "peer unreachable");
        assert!(err.to_string().contains("pipeline hop 1/b"), "typed display: {err}");
    }

    #[test]
    fn hop_timeout_accounts_for_rtt() {
        assert_eq!(hop_timeout(Duration::from_millis(100), 0), Duration::from_millis(100));
        assert_eq!(hop_timeout(Duration::from_millis(100), 40), Duration::from_millis(220));
        // Monotonic in RTT, and the saturating path never panics.
        let huge = hop_timeout(Duration::from_millis(100), u64::MAX);
        assert!(huge > Duration::from_millis(100));
        assert_eq!(huge, Duration::from_nanos(u64::MAX) + Duration::from_millis(100));
    }

    #[tokio::test]
    async fn empty_route_returns_input_unchanged() {
        assert_eq!(
            execute_route("m", "r", &[], vec![1.0_f32; 4],
                |_, _| unreachable!(),
                |_, _, _, _| Box::pin(async { unreachable!() }),
                |_| u64::MAX, BASE)
                .await.unwrap(),
            vec![1.0_f32; 4]
        );
    }

    // ---- M2: degradation + hysteresis ----

    /// Local executor used by the M2 resilience tests: keyed by the same
    /// per-hop expert id so replacement hops answer deterministically.
    fn local_of(model: &'static str) -> impl Fn(&[f32], &[usize]) -> anyhow::Result<Vec<f32>> {
        move |a, layers| {
            run_hop(
                &hop_expert_id(model, &PipelineHop { node: None, layers: layers.to_vec() }),
                a,
            )
        }
    }

    #[tokio::test]
    async fn dead_hop_layers_reassigned_by_the_ring_rule_and_route_completes() {
        let model = "moe-v2";
        let req = "req-m2-1";
        let route = vec![
            PipelineHop { node: None, layers: vec![0] },
            PipelineHop { node: Some("peer-a".into()), layers: vec![1, 2] },
            PipelineHop { node: Some("peer-b".into()), layers: vec![3] },
        ];

        // The caller's reassign closure IS the ring's own rule: layers re-
        // pinned over the alive set minus the dead owner, merged per owner.
        let reassign = |layers: &[usize], dead: Option<&str>| {
            assert_eq!(dead, Some("peer-a"), "only the dead owner is dropped");
            if layers == [1_usize, 2] {
                Some(vec![PipelineHop { node: Some("peer-b".into()), layers: vec![1, 2] }])
            } else {
                None
            }
        };
        let mut stability = RouteStability::new();

        let mut input = vec![0.5_f32; 8];
        input[3] = -0.25;
        let result = execute_route_resilient(
            model, req, route, input,
            local_of(model),
            move |expert_id, node, activation, _t| {
                let expert_id = expert_id.to_owned();
                let node = node.to_owned();
                let fut = async move {
                    if node == "peer-a" {
                        // The production closure honors the hop deadline from
                        // the QUIC dispatch; here the dead peer just fails.
                        anyhow::bail!("peer unreachable");
                    }
                    run_hop(&expert_id, &activation).map(|out| {
                        let c = result_commitment(req, &expert_id, &node, &activation, &out);
                        (out, Some(c))
                    })
                };
                Box::pin(fut)
            },
            |n| if n == "peer-a" { 5 } else { 40 },
            BASE,
            &mut stability,
            reassign,
        )
        .await;

        // Complete walk: the failed hop was RE-HOSTED, not skipped.
        assert!(result.degraded.is_empty(), "no honest losses: {:?}", result.degraded);
        assert_eq!(result.reassigned.len(), 1, "the dead hop was re-hosted");
        assert_eq!(result.reassigned[0].layers, vec![1, 2]);
        assert_eq!(result.reassigned[0].was_node.as_deref(), Some("peer-a"));
        // hops_run counts successful executions: the dead attempt is not one.
        assert_eq!(result.hops_run, 3, "local 0, replacement 1-2, tail 3");

        // Serial reference: layer 1-2 ran under the replacement id.
        let serial = vec![
            PipelineHop { node: None, layers: vec![0] },
            PipelineHop { node: Some("peer-b".into()), layers: vec![1, 2] },
            PipelineHop { node: Some("peer-b".into()), layers: vec![3] },
        ];
        let mut expect = vec![0.5_f32; 8];
        expect[3] = -0.25;
        for hop in &serial {
            expect = run_hop(&hop_expert_id(model, hop), &expect).unwrap();
        }
        assert_eq!(result.activation, expect, "reassigned walk must equal the re-planned serial walk");
    }

    #[tokio::test]
    async fn unrecoverable_hop_walks_on_with_an_honest_partial_answer() {
        let model = "moe-v3";
        let req = "req-m2-2";
        let route = vec![
            PipelineHop { node: None, layers: vec![0] },
            PipelineHop { node: Some("gone".into()), layers: vec![1] },
            PipelineHop { node: Some("peer-b".into()), layers: vec![2] },
        ];
        let mut stability = RouteStability::new();

        let input = vec![0.5_f32; 8];
        let result = execute_route_resilient(
            model, req, route, input,
            local_of(model),
            move |expert_id, node, activation, _t| {
                let expert_id = expert_id.to_owned();
                let node = node.to_owned();
                let fut = async move {
                    if node == "gone" {
                        anyhow::bail!("peer unreachable")
                    }
                    run_hop(&expert_id, &activation).map(|out| (out, None))
                };
                Box::pin(fut)
            },
            |_| 20,
            BASE,
            &mut stability,
            |_layers, _dead| None, // nobody alive to take the layers
        )
        .await;

        // The router walked ON past the lost hop:
        assert_eq!(result.hops_run, 2, "local hop 0 and peer-b hop 2 ran");
        assert_eq!(result.degraded.len(), 1, "exactly the lost layer typed");
        assert_eq!(result.degraded[0].layers, vec![1]);
        assert_eq!(result.degraded[0].node.as_deref(), Some("gone"));
        assert!(result.degraded[0].reason.contains("skipped"));
        assert!(result.degraded[0].reason.contains("peer unreachable"));
        assert!(!result.is_complete());

        // Honest hops-short: residual = local(0) then peer-b(2) — layer 1
        // skipped inside the chain deterministically, not silently dropped.
        let local0 = run_hop(
            &hop_expert_id(model, &PipelineHop { node: None, layers: vec![0] }),
            &vec![0.5_f32; 8],
        )
        .unwrap();
        let expect_b2 = run_hop(
            &hop_expert_id(model, &PipelineHop { node: Some("peer-b".into()), layers: vec![2] }),
            &local0,
        )
        .unwrap();
        assert_eq!(result.activation, expect_b2);
    }

    #[test]
    fn flapping_owner_is_dampened_by_two_line_hysteresis() {
        let mut stability = RouteStability::new();
        assert!(!stability.is_excluded("flapper"));
        stability.note_failure("flapper");
        stability.note_failure("flapper");
        stability.note_failure("flapper");
        assert!(stability.is_excluded("flapper"), "raise at {HOP_EXCLUDE_SCORE}");

        // One success only decays — no instant re-inclusion mid-flap.
        stability.note_success("flapper");
        assert!(stability.is_excluded("flapper"), "score 2 sits in the hysteresis band");
        stability.note_success("flapper");
        assert!(!stability.is_excluded("flapper"), "cleared below {HOP_CLEAR_SCORE}");

        // Unknown nodes are never excluded; healthy peers are untouched.
        assert!(!stability.is_excluded("stranger"));
        stability.note_success("stranger");
        assert!(!stability.is_excluded("stranger"));
    }

    #[test]
    fn filter_alive_drops_excluded_but_never_empties_the_set() {
        let mut stability = RouteStability::new();
        for _ in 0..3 {
            stability.note_failure("a");
        }
        let alive = vec!["a".to_string(), "b".to_string()];
        assert_eq!(stability.filter_alive(&alive), vec!["b".to_string()]);

        // Everyone flapped: route anyway (a isogenic flaky hop beats no hop).
        for _ in 0..3 {
            stability.note_failure("b");
        }
        assert_eq!(stability.filter_alive(&alive), alive, "never hand back an empty routing set");
    }

    #[tokio::test]
    async fn walk_failures_feed_the_stability_tracker() {
        let model = "moe-v4";
        let req = "req-m2-3";
        // Three hops all owned by the same dead node — the walk notes each.
        let route = vec![
            PipelineHop { node: Some("dead".into()), layers: vec![0] },
            PipelineHop { node: Some("dead".into()), layers: vec![1] },
            PipelineHop { node: Some("dead".into()), layers: vec![2] },
        ];
        let mut stability = RouteStability::new();
        let result = execute_route_resilient(
            model, req, route, vec![0.0_f32; 4],
            local_of(model),
            |_e, node, _x, _t| {
                let dead = node.to_owned();
                Box::pin(async move {
                    anyhow::ensure!(dead != "dead", "peer unreachable");
                    Ok::<_, anyhow::Error>((vec![0.0_f32; 4], None))
                })
            },
            |_| 20, BASE,
            &mut stability,
            |_l, _d| None,
        )
        .await;
        assert_eq!(result.degraded.len(), 3);
        assert!(stability.is_excluded("dead"), "3 walk failures must raise the flag");
    }
}
