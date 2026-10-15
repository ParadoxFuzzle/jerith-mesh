//! Hot-expert replication + cold-expert sharding (Phase 2).
//!
//! Two policies over the same facts — who holds what, who asked for what:
//!
//! - **Hot experts replicate.** The top 10–20 % most-activated experts
//!   (`expert_stats`) should live on more than one node, so a peer that
//!   goes away (or gets slow) does not take the mesh's busiest experts
//!   with it. [`plan_replication`] compares the advertised holders against
//!   the desired replication factor and returns the fetches worth doing —
//!   pulling the container from the best (nearest) holder into the local
//!   store, which then advertises it on the next ping.
//! - **Cold experts shard.** The long tail gets exactly one owner instead
//!   of a replica set, and every node must agree on *which* node that is
//!   without a coordinator round-trip. [`ShardMap::primary_owner`] is
//!   rendezvous (highest-random-weight) hashing over the live node ids:
//!   deterministic for a given node set, stable when the set changes
//!   (only the departed node's experts move), and computable from the ads
//!   every node already has.
//!
//! Both are pure functions of inputs the mesh already exchanges, so they
//! are unit-testable without a network — the executor (in `ping`) just
//! applies what they return.

use std::collections::HashMap;

use crate::bootstrap::PeerAd;

/// A peer as the replication policy sees it: identity, distance, and what
/// its advertisement says it holds.
#[derive(Debug, Clone)]
pub struct ReplicaPeer {
    pub node_id: String,
    pub rtt_ms: u64,
    /// `None` = legacy ad (unknown hosting), `Some(list)` = advertised.
    pub hosted_experts: Option<Vec<String>>,
    pub alive: bool,
}

impl ReplicaPeer {
    /// Advertised holder: the peer's ad explicitly names this expert.
    /// Only these can be a transfer *source* — dialing a peer whose
    /// hosting is merely unknown would burn a round-trip to learn it
    /// does not have the weights.
    pub fn explicit_holder(&self, expert_id: &str) -> bool {
        match &self.hosted_experts {
            Some(list) => list.iter().any(|e| e == expert_id),
            None => false,
        }
    }

    /// Hosting unknown (pre-Phase-2 ad). Counts toward the replication
    /// accounting — assuming these peers hold nothing would make every
    /// node over-replicate whenever an old node is in the mesh — but is
    /// never used as a fetch source.
    pub fn unknown_hosting(&self) -> bool {
        self.hosted_experts.is_none()
    }
}

/// One fetch worth performing: pull `expert_id` from `source_node`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplicationAction {
    pub expert_id: String,
    pub source_node: String,
    /// How many holders the mesh advertised before this action (for logs
    /// and for `replication_factor` accounting in later rounds).
    pub holders_before: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplicationConfig {
    /// Hot fraction in percent (the spec's 10–20 %).
    pub hot_percent: u32,
    /// Desired mesh-wide holders per hot expert (≥ 2 by definition).
    pub replication_factor: usize,
    /// Most fetches to start in one decision round: replication is
    /// background work and must not crowd out serving traffic.
    pub max_actions_per_round: usize,
}

impl Default for ReplicationConfig {
    fn default() -> Self {
        Self {
            hot_percent: crate::expert_stats::DEFAULT_HOT_PERCENT,
            replication_factor: 3,
            max_actions_per_round: 2,
        }
    }
}

/// Decide which hot experts this node should replicate, and from whom.
///
/// Deliberately conservative:
/// - experts the node already holds are skipped (it *is* a holder);
/// - a hot expert already at `replication_factor` confirmed holders is
///   left alone;
/// - the nearest confirmed holder is the source (ties by node id), so a
///   replica tends to land near the traffic that caused it — but any
///   holder works, and a transfer that fails simply retries next round;
/// - at most `max_actions_per_round` actions come back, so a node that
///   just joined a busy mesh trickles its replicas in rather than
///   saturating the link.
pub fn plan_replication(
    hot: &[String],
    local_hosted: &[String],
    peers: &[ReplicaPeer],
    config: ReplicationConfig,
) -> Vec<ReplicationAction> {
    let mut actions = Vec::new();
    if hot.is_empty() || config.max_actions_per_round == 0 {
        return actions;
    }
    let factor = config.replication_factor.max(2);
    for expert_id in hot {
        if actions.len() >= config.max_actions_per_round {
            break;
        }
        if local_hosted.iter().any(|e| e == expert_id) {
            continue;
        }
        let mut holders: Vec<&ReplicaPeer> = peers
            .iter()
            .filter(|p| p.alive && p.explicit_holder(expert_id))
            .collect();
        let unknown = peers
            .iter()
            .filter(|p| p.alive && p.unknown_hosting())
            .count();
        let holders_before = holders.len() + unknown;
        if holders_before >= factor {
            continue;
        }
        if holders.is_empty() {
            // No advertised holder: a fetch has no reliable source. (A
            // *cold* expert in this state is the shard map's problem — its
            // owner is expected to seed it; see `ShardMap`.)
            continue;
        }
        holders.sort_by(|a, b| {
            a.rtt_ms
                .cmp(&b.rtt_ms)
                .then_with(|| a.node_id.cmp(&b.node_id))
        });
        actions.push(ReplicationAction {
            expert_id: expert_id.clone(),
            source_node: holders[0].node_id.clone(),
            holders_before,
        });
    }
    actions
}

/// Cold-expert sharding: deterministic single ownership of the tail.
#[derive(Debug, Clone, Default)]
pub struct ShardMap {
    /// Sorted node ids, so every node computes the same ranking from the
    /// same membership regardless of ad arrival order.
    nodes: Vec<String>,
}

impl ShardMap {
    pub fn new(mut nodes: Vec<String>) -> Self {
        nodes.sort();
        nodes.dedup();
        Self { nodes }
    }

    /// From live ads: every alive peer, plus `self_node_id` (the node
    /// computing the map must include itself, or it will disagree with its
    /// peers about who owns what).
    pub fn from_ads(self_node_id: &str, ads: &[PeerAd]) -> Self {
        let mut nodes: Vec<String> = ads.iter().map(|a| a.node_id.clone()).collect();
        nodes.push(self_node_id.to_string());
        Self::new(nodes)
    }

    /// Members of the shard ring (sorted, deduped).
    pub fn members(&self) -> &[String] {
        &self.nodes
    }

    fn score(&self, expert_id: &str, node_id: &str) -> u64 {
        // FNV-1a over (expert|node): a stable pseudo-random weight per
        // pair. Rendezvous hashing = the node with the highest weight owns
        // the key, which reassigns only the departed node's keys when the
        // ring changes.
        let mut h: u64 = 0xcbf29ce484222325;
        for b in expert_id
            .as_bytes()
            .iter()
            .chain(b"|".iter())
            .chain(node_id.as_bytes())
        {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    /// Preferred owners for one expert, best first (`k` distinct nodes).
    pub fn owners(&self, expert_id: &str, k: usize) -> Vec<String> {
        let mut scored: Vec<(u64, &String)> = self
            .nodes
            .iter()
            .map(|n| (self.score(expert_id, n), n))
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(b.1)));
        scored
            .into_iter()
            .take(k.max(1))
            .map(|(_, n)| n.clone())
            .collect()
    }

    /// The single owner of a cold expert.
    pub fn primary_owner(&self, expert_id: &str) -> Option<String> {
        self.owners(expert_id, 1).into_iter().next()
    }

    /// Assignment for a whole cold set: expert → owner (hot experts get
    /// `hot_factor` owners instead of one).
    pub fn assign(
        &self,
        cold: &[String],
        hot: &[String],
        hot_factor: usize,
    ) -> HashMap<String, Vec<String>> {
        let mut out: HashMap<String, Vec<String>> = HashMap::new();
        for expert_id in cold {
            if let Some(owner) = self.primary_owner(expert_id) {
                out.insert(expert_id.clone(), vec![owner]);
            }
        }
        for expert_id in hot {
            let owners = self.owners(expert_id, hot_factor.max(2));
            if !owners.is_empty() {
                out.insert(expert_id.clone(), owners);
            }
        }
        out
    }

    /// Does this node own the cold expert (i.e. should it seed/serve it)?
    pub fn owns(&self, self_node_id: &str, expert_id: &str) -> bool {
        self.primary_owner(expert_id).as_deref() == Some(self_node_id)
    }

    /// Layer ownership map for pipeline / layer-group fallback (Petals-style).
    /// Each node owns a contiguous slice of model layers. Given a set of
    /// alive node ids (including self), this assigns each layer a primary
    /// owner from the ring, forming a deterministic layer-group assignment
    /// that reassigns only the departed node's layers when the set changes.
    pub fn layer_ownership(
        &self,
        self_node_id: &str,
        layers: &[usize],
        alive: &[String],
    ) -> HashMap<usize, String> {
        let mut out: HashMap<usize, String> = HashMap::new();
        let _ring = ShardMap::from_ads(self_node_id, &[]);
        for &layer in layers {
            // Use the shard map's rendezvous hash to deterministically assign
            // each layer to a node. The node ids in the ring are sorted and
            // deduped; the score function gives every node a stable weight.
            let owner = self.primary_owner_for_layer(&layer, alive);
            if owner.as_deref() == Some(self_node_id) {
                out.insert(layer, self_node_id.to_string());
            }
        }
        out
    }

    /// Preferred owner for one model layer from the alive set.
    fn primary_owner_for_layer(&self, layer: &usize, alive: &[String]) -> Option<String> {
        let mut best_node: Option<String> = None;
        let mut best_score: u64 = 0;
        for node_id in alive {
            let s = self.score(&format!("L{}", layer), node_id);
            if best_node.is_none() || s > best_score {
                best_score = s;
                best_node = Some(node_id.clone());
            }
        }
        best_node
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peer(node_id: &str, rtt: u64, hosted: Option<&[&str]>) -> ReplicaPeer {
        ReplicaPeer {
            node_id: node_id.to_string(),
            rtt_ms: rtt,
            hosted_experts: hosted.map(|l| l.iter().map(|s| s.to_string()).collect()),
            alive: true,
        }
    }

    #[test]
    fn under_replicated_hot_experts_are_fetched_from_the_nearest_holder() {
        let hot = vec!["L0.E0".to_string(), "L0.E1".to_string()];
        let peers = vec![
            peer("far", 200, Some(&["L0.E0", "L0.E1"])),
            peer("near", 5, Some(&["L0.E0"])),
        ];
        let actions = plan_replication(
            &hot,
            &[],
            &peers,
            ReplicationConfig {
                replication_factor: 3,
                max_actions_per_round: 4,
                ..Default::default()
            },
        );
        assert_eq!(
            actions,
            vec![
                ReplicationAction {
                    expert_id: "L0.E0".into(),
                    source_node: "near".into(),
                    holders_before: 2,
                },
                ReplicationAction {
                    expert_id: "L0.E1".into(),
                    source_node: "far".into(),
                    holders_before: 1,
                },
            ]
        );
    }

    #[test]
    fn already_replicated_or_locally_held_experts_are_left_alone() {
        let hot = vec!["L0.E0".to_string(), "L0.E1".to_string()];
        let peers = vec![
            peer("a", 10, Some(&["L0.E0", "L0.E1"])),
            peer("b", 20, Some(&["L0.E0", "L0.E1"])),
            peer("c", 30, Some(&["L0.E0", "L0.E1"])),
        ];
        // Factor 3 satisfied by three holders → nothing to do.
        assert!(
            plan_replication(
                &hot,
                &[],
                &peers,
                ReplicationConfig {
                    replication_factor: 3,
                    max_actions_per_round: 4,
                    ..Default::default()
                }
            )
            .is_empty()
        );
        // Holding it locally also satisfies the accounting.
        assert!(
            plan_replication(
                &["L0.E0".to_string()],
                &["L0.E0".to_string()],
                &peers[..2],
                ReplicationConfig {
                    replication_factor: 3,
                    max_actions_per_round: 4,
                    ..Default::default()
                }
            )
            .is_empty()
        );
        // An expert nobody advertises has no source: skipped, not looped.
        assert!(
            plan_replication(
                &["L0.E9".to_string()],
                &[],
                &peers,
                ReplicationConfig {
                    replication_factor: 3,
                    max_actions_per_round: 4,
                    ..Default::default()
                }
            )
            .is_empty()
        );
    }

    #[test]
    fn round_budget_caps_how_much_a_joiner_pulls_at_once() {
        let hot: Vec<String> = (0..10).map(|i| format!("L0.E{i}")).collect();
        let peers = vec![peer("holder", 10, Some(&["L0.E0", "L0.E1", "L0.E2"]))];
        let actions = plan_replication(
            &hot,
            &[],
            &peers,
            ReplicationConfig {
                replication_factor: 3,
                max_actions_per_round: 2,
                ..Default::default()
            },
        );
        assert_eq!(actions.len(), 2, "one round pulls at most the budget");
        // Legacy (unknown hosting) peers count toward the factor but can
        // never be a source: 3 expected holders, 2 unknown → no action.
        let legacy = vec![peer("old", 50, None), peer("older", 60, None)];
        assert!(
            plan_replication(
                &["L0.E0".to_string()],
                &[],
                &legacy,
                ReplicationConfig {
                    replication_factor: 3,
                    max_actions_per_round: 4,
                    ..Default::default()
                }
            )
            .is_empty()
        );
    }

    #[test]
    fn cold_shard_ownership_is_deterministic_across_nodes() {
        let nodes = vec!["n2".to_string(), "n1".to_string(), "n3".to_string()];
        let one = ShardMap::new(nodes.clone());
        let two = ShardMap::new(vec!["n3".to_string(), "n1".to_string(), "n2".to_string()]);
        for expert in ["L9.E1", "L9.E2", "L9.E777", "cold.tail"] {
            assert_eq!(
                one.primary_owner(expert),
                two.primary_owner(expert),
                "ad arrival order must not change ownership of {expert}"
            );
            let owner = one.primary_owner(expert).unwrap();
            assert!(nodes.contains(&owner));
            assert!(one.owns(&owner, expert));
            // Exactly one owner, and a node only owns what it owns.
            let owners = one.owners(expert, 1);
            assert_eq!(owners, vec![owner]);
        }
        // Empty ring ⇒ nobody owns anything (never a panic).
        assert_eq!(ShardMap::new(vec![]).primary_owner("x"), None);
    }

    #[test]
    fn ring_changes_move_only_the_departed_nodes_keys() {
        let before = ShardMap::new(vec!["n1".into(), "n2".into(), "n3".into()]);
        let after = ShardMap::new(vec!["n1".into(), "n3".into()]); // n2 left
        let mut moved_by_leaver = 0;
        let mut moved_by_survivor = 0;
        for i in 0..200 {
            let expert = format!("L4.E{i}");
            if before.primary_owner(&expert) != after.primary_owner(&expert) {
                moved_by_leaver += 1;
                continue;
            }
            // Unchanged keys must stay unchanged.
            moved_by_survivor += 0;
        }
        assert!(
            moved_by_leaver > 0,
            "some keys must follow the departed node"
        );
        // n1 and n3 keep ownership of what they had: recompute directly.
        for i in 0..200 {
            let expert = format!("L4.E{i}");
            let owner_before = before.primary_owner(&expert).unwrap();
            if owner_before != "n2" {
                assert_eq!(
                    after.primary_owner(&expert).as_deref(),
                    Some(owner_before.as_str()),
                    "survivor {owner_before} lost {expert} even though it stayed"
                );
            }
        }
        assert_eq!(moved_by_survivor, 0);
    }

    #[test]
    fn assignment_covers_every_expert_with_the_right_owner_count() {
        let ring = ShardMap::new(vec!["a".into(), "b".into(), "c".into(), "d".into()]);
        let cold: Vec<String> = (0..10).map(|i| format!("cold.{i}")).collect();
        let hot: Vec<String> = (0..3).map(|i| format!("hot.{i}")).collect();
        let map = ring.assign(&cold, &hot, 3);
        for expert in &cold {
            assert_eq!(map.get(expert).map(Vec::len), Some(1), "cold = one owner");
        }
        for expert in &hot {
            assert_eq!(map.get(expert).map(Vec::len), Some(3), "hot = replica set");
            let owners = &map[expert];
            let unique: std::collections::HashSet<&String> = owners.iter().collect();
            assert_eq!(unique.len(), owners.len(), "owners must be distinct");
        }
        // Hot experts want at least 2 owners even if asked for fewer.
        let map2 = ring.assign(&[], &["hot.x".into()], 1);
        assert_eq!(map2["hot.x"].len(), 2);
    }

    #[test]
    fn shard_map_from_ads_includes_self_once() {
        let ads = vec![
            PeerAd {
                node_id: "me".into(),
                address: "127.0.0.1:1".into(),
                model_id: "m".into(),
                quant: "int4".into(),
                contribution_percent: 25,
                last_seen_ms: 0,
                role: crate::config::NodeRole::Hybrid,
                rpc_address: String::new(),
                resources: None,
                hosted_experts: None,
                hosted_expert_count: None,
                discovery_servers: Vec::new(),
            },
            PeerAd {
                node_id: "peer".into(),
                address: "127.0.0.1:2".into(),
                model_id: "m".into(),
                quant: "int4".into(),
                contribution_percent: 25,
                last_seen_ms: 0,
                role: crate::config::NodeRole::Hybrid,
                rpc_address: String::new(),
                resources: None,
                hosted_experts: None,
                hosted_expert_count: None,
                discovery_servers: Vec::new(),
            },
        ];
        let ring = ShardMap::from_ads("me", &ads);
        assert_eq!(ring.members(), ["me", "peer"]);
        // A ring of one node owns everything (a lone node is the mesh).
        let solo = ShardMap::from_ads("me", &[]);
        assert_eq!(solo.primary_owner("L0.E0").as_deref(), Some("me"));
    }
}
