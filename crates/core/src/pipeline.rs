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
//! M0 (this module): pure route planning from the ring's full layer map —
//! no I/O, no runtime, no clocks: deterministic, unit-testable, and
//! naturally degradable (a single hop when only the local node is alive).
//!
//! Later milestones build on this shape:
//! - M1: hop execution over the existing QUIC activation contract, with
//!   per-hop timeouts and the per-hop commitment layer composed on.
//! - M2: degradation (skip/hop-abort semantics, flapping-peer hysteresis).
//! - M3: latency budgeting (hop deadlines proportional to RTT).

use std::collections::BTreeMap;

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
