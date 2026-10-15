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
//! M0 + M1 (this module): pure route planning from the ring's full layer map
//! and the route executor that walks it. Planning is I/O-free and
//! clock-free (deterministic, unit-testable, naturally degradable); the
//! executor is the M1 hop runtime over the same f32-vector wire contract
//! expert dispatch already uses, with the Phase 4 commitment recomputed
//! *per hop* (each hop's expert id names its layer run, so a tampering hop
//! is caught downstream exactly like a single-expert reply) and per-hop
//! deadlines derived from the owner's measured RTT.

use std::collections::BTreeMap;
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
///   skipping of a hop's layers (M2 replaces abort with ring-rule
///   reassignment; the count of completed hops is already recorded).
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
        let expert_id = hop_expert_id(model_id, hop);
        let outcome = match &hop.node {
            None => local_run(&activation, &hop.layers).map(|out| (out, None::<String>)),
            Some(node) => remote_run(&expert_id, node, activation.clone(), hop_timeout(base_timeout, rtt_of(node)))
                .await
                .and_then(|(out, commitment)| {
                    match commitment {
                        // The committing side is the mesh's tamper-evidence
                        // contract for one expert; compose it per hop.
                        Some(c) if crate::verify::verify_commitment(
                            req_id,
                            &expert_id,
                            node,
                            &activation,
                            &out,
                            &c,
                        ) => Ok((out, None)),
                        Some(_) => Err(anyhow::anyhow!(
                            "commitment mismatch from hop {node}: body altered"
                        )),
                        None => Ok((out, None)),
                    }
                }),
        };
        match outcome {
            Ok((out, _)) => activation = out,
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
}
