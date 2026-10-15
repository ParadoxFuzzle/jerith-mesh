//! Mesh runtime: health ping loop + contribution metering.
//!
//! Owns the long-running asyncio of the node:
//! - Refreshes peer list from bootstrap (model-homogeneous filter applied).
//! - Pings every connectable peer with a minimal-resource request,
//!   deprecating nodes that fail repeatedly and reporting them to the
//!   bootstrap so they're deprecated globally.
//! - Advertises ourselves to the bootstrap on the same cadence.
//! - Accumulates the four contribution meters (tokens, activations,
//!   bandwidth, uptime) and marks the node's own contribution state.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{Mutex, RwLock, watch};
use tokio::time::{MissedTickBehavior, interval};

use crate::bootstrap::{BootstrapClient, PeerAd, PeerRegistry, PeerState, now_ms};
use crate::config::{NodeConfig, NodeRole};

/// Shared Phase 4 verification tracker: counters + pending fraud evidence.
pub type SharedVerifyTracker = std::sync::Arc<crate::verify::VerificationTracker>;

/// Contribution meters accumulated between settlements.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Meters {
    /// Tokens generated for peers.
    pub tokens_served: u64,
    /// Expert loads/activations served for peers.
    pub expert_activations: u64,
    /// Bytes relayed for peers (in+out).
    pub bytes_relayed: u64,
    /// Seconds of healthy, pingable existence.
    pub uptime_seconds: u64,
    /// Tokens we consumed as a requester (for leech accounting).
    pub tokens_consumed: u64,
    /// JER mine attempts fired on served inference.
    pub mines_attempted: u64,
    /// JER mines that actually won a block reward.
    pub mines_won: u64,
}

impl Meters {
    pub fn snapshot_reset(&mut self) -> Self {
        std::mem::take(self)
    }
}

/// Shared runtime state visible to the gateway/web/CLI.
pub struct MeshState {
    pub config: RwLock<NodeConfig>,
    pub registry: Mutex<PeerRegistry>,
    pub meters: Mutex<Meters>,
    /// This node's Ed25519 wire identity (node_id = pubkey fingerprint).
    pub identity: crate::signing::SigningIdentity,
    /// Set when the contribution ratio dips below the minimum (rolling).
    pub contribution_healthy: watch::Sender<bool>,
    pub started_at: std::time::Instant,
    /// What THIS node can offer right now — refreshed at ping cadence
    /// (see hw::sample) and shipped inside our advertise (resource
    /// pooling, Phase 1): peers read it for placement + fleet census.
    pub resources: Mutex<Option<crate::hw::ResourceSample>>,
    /// Phase 4 verification counters; the ping loop flushes queued
    /// DishonestResults evidence to the bootstrap on its regular cadence.
    pub verify: SharedVerifyTracker,
    /// Statistical freerider detection (Phase 4): coordinator-side evidence
    /// only — the ledger rides the dispatch loop (feed_dispatch) and the
    /// evaluator runs on the ping tick, off the inference hot path.
    pub freerider:
        std::sync::Arc<tokio::sync::Mutex<crate::freerider::FreeriderDetector>>,
    /// Phase 5 stake + slashing handle (ledger persists across restarts).
    pub stake: Arc<crate::stake::StakeManager>,
    /// Phase 2 tensor transport: local expert weights (set by the node
    /// when `[experts]` resolves to a directory with experts). Drives the
    /// hosting advertisement, the router's local-first plan, and the
    /// `/status` store stats.
    pub experts: Mutex<Option<std::sync::Arc<crate::expert_store::ExpertStore>>>,
    /// Phase 2 replication: how often each expert was actually activated
    /// (served here, or dispatched from here). The hot/cold split and the
    /// shard ring both read it.
    pub expert_stats: crate::expert_stats::SharedExpertStats,
    /// Resource isolation (Phase 1): the kill-switch state the governor
    /// trips and everything mesh-facing consults before doing real work.
    pub isolate: std::sync::Arc<crate::isolate::IsolateState>,
    /// Per-peer block list (Phase 3): shared by the QUIC server's refusal
    /// gate, placement (mesh_view filters candidates) and `/status`. The
    /// ping tick re-syncs it from config, so edits land live.
    pub peer_block: std::sync::Arc<crate::ratelimit::BlockedPeers>,
    /// Bootstrap rotation (Phase 1 discovery redundancy): configured
    /// servers + peer-gossiped learned URLs + last-good bookkeeping.
    /// Persisted to `<data>/discovery-servers.json` when it grows.
    pub discovery:
        std::sync::Arc<tokio::sync::Mutex<crate::discovery::DiscoveryServers>>,
    /// Where those counters persist (see
    /// [`set_expert_stats`](Self::set_expert_stats)).
    expert_stats_path: std::sync::Mutex<std::path::PathBuf>,
}

impl MeshState {
    pub fn new(config: NodeConfig) -> anyhow::Result<(Arc<Self>, watch::Receiver<bool>)> {
        let (tx, rx) = watch::channel(true);
        let identity = config.identity()?;
        let stake = crate::stake::StakeManager::new(&config.crypto, Some(identity.clone()));
        // Built BEFORE `config` moves into the RwLock below.
        let peer_block = crate::ratelimit::BlockedPeers::from_list(&config.peers.blocked);
        // Discovery rotation: restore the learned set + last-good from the
        // persisted file, then layer the config's server list on top.
        let discovery = {
            let mut d = crate::discovery::DiscoveryServers::load(
                &crate::config::NodeConfig::data_dir().join("discovery-servers.json"),
            );
            d.configured = crate::discovery::sanitize(&config.discovery_seed_servers(), None);
            d
        };
        let state = Arc::new(Self {
            config: RwLock::new(config),
            registry: Mutex::new(PeerRegistry::new(None)),
            meters: Mutex::new(Meters::default()),
            identity,
            contribution_healthy: tx,
            started_at: std::time::Instant::now(),
            resources: Mutex::new(None),
            verify: crate::verify::VerificationTracker::shared(),
            stake,
            experts: Mutex::new(None),
            expert_stats: std::sync::Arc::new(std::sync::Mutex::new(
                crate::expert_stats::ExpertStats::load(&crate::expert_stats::default_stats_path()),
            )),
            expert_stats_path: std::sync::Mutex::new(crate::expert_stats::default_stats_path()),
            isolate: std::sync::Arc::new(crate::isolate::IsolateState::default()),
            peer_block,
            freerider: std::sync::Arc::new(tokio::sync::Mutex::new(
                crate::freerider::FreeriderDetector::default(),
            )),
            discovery: std::sync::Arc::new(tokio::sync::Mutex::new(discovery)),
        });
        Ok((state, rx))
    }

    /// Attach this node's expert weight store (Phase 2). Idempotent: the
    /// node calls it once at startup with the store it also hands to the
    /// QUIC handler.
    pub async fn set_expert_store(&self, store: std::sync::Arc<crate::expert_store::ExpertStore>) {
        *self.experts.lock().await = Some(store);
    }

    /// The attached store, if any.
    pub async fn expert_store(&self) -> Option<std::sync::Arc<crate::expert_store::ExpertStore>> {
        self.experts.lock().await.clone()
    }

    /// Count one expert activation (Phase 2 demand signal): served here or
    /// dispatched from here, either way the expert is hot for this node.
    pub fn record_activation(&self, expert_id: &str) {
        if let Ok(mut stats) = self.expert_stats.lock() {
            stats.record(expert_id);
        }
    }

    /// A copy of the activation counters.
    pub fn expert_stats_snapshot(&self) -> crate::expert_stats::ExpertStats {
        self.expert_stats
            .lock()
            .map(|s| s.clone())
            .unwrap_or_default()
    }

    /// Replace the demand counters and where they persist. Tests (and any
    /// embedding caller) use this to keep a run's numbers out of a live
    /// node's `<data>/expert-stats.json`.
    pub fn set_expert_stats(
        &self,
        stats: crate::expert_stats::ExpertStats,
        path: std::path::PathBuf,
    ) {
        if let Ok(mut current) = self.expert_stats.lock() {
            *current = stats;
        }
        if let Ok(mut current) = self.expert_stats_path.lock() {
            *current = path;
        }
    }

    /// Persist the counters (best-effort; statistics are advisory).
    pub fn save_expert_stats(&self) {
        let snapshot = self.expert_stats_snapshot();
        let path = self
            .expert_stats_path
            .lock()
            .map(|p| p.clone())
            .unwrap_or_else(|_| crate::expert_stats::default_stats_path());
        if let Err(e) = snapshot.save(&path) {
            tracing::warn!(target: "mesh", error = %e, "expert stats not persisted");
        }
    }

    /// Expert ids this node serves with real tensors (sorted; empty when
    /// no store is attached).
    pub async fn hosted_experts(&self) -> Vec<String> {
        match self.experts.lock().await.as_ref() {
            Some(store) => store.hosted(),
            None => Vec::new(),
        }
    }
}

/// Background replication loop (Phase 2): periodically ask "which hot
/// experts are under-replicated?" and pull the missing ones from a holder.
///
/// Separate from the ping loop on purpose — replication moves real bytes
/// and must never delay liveness, and a node with no store (or with
/// replication switched off) simply never spawns it.
pub struct ReplicaManager {
    shutdown_tx: watch::Sender<bool>,
    handle: tokio::task::JoinHandle<()>,
}

impl ReplicaManager {
    /// Spawn the loop. Call [`shutdown`](Self::shutdown) to stop it — or
    /// simply drop the manager: the loop notices the closed channel.
    pub fn spawn(
        state: Arc<MeshState>,
        chat: Arc<MeshFallbackChat>,
        interval: Duration,
        config: crate::replication::ReplicationConfig,
    ) -> Self {
        let (shutdown_tx, mut shutdown) = watch::channel(false);
        // Demand recorded since the last write. The signal has to survive a
        // restart even when *nothing* replicates (a lone node, or an
        // all-cold mesh), so activation counts are flushed on their own —
        // not only alongside a fetch. Read before the task starts, so an
        // activation that lands between `spawn` and the first poll is not
        // folded into the baseline and forgotten.
        let mut persisted_total = state.expert_stats_snapshot().total();
        let handle = tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            // First tick fires immediately; skipping it keeps boot quiet
            // (nothing is discovered yet anyway).
            ticker.tick().await;
            loop {
                tokio::select! {
                    _ = ticker.tick() => {
                        let actions = chat.replication_tick(&state, config).await;
                        let total = state.expert_stats_snapshot().total();
                        if !actions.is_empty() || total != persisted_total {
                            state.save_expert_stats();
                            persisted_total = total;
                        }
                    }
                    // A stop signal, or the manager going away (closed
                    // channel): both end the loop, after persisting the
                    // demand signal one last time. A closed channel must
                    // end it too — otherwise `changed()` returns instantly
                    // forever and the loop spins.
                    _ = shutdown.changed() => {
                        state.save_expert_stats();
                        break;
                    }
                }
            }
        });
        Self {
            shutdown_tx,
            handle,
        }
    }

    /// Stop the loop and wait (bounded) for its final stats flush. A
    /// transfer already in flight is given a moment, never waited on
    /// indefinitely — the tick's own timeout bounds it anyway.
    pub async fn shutdown(self) {
        let _ = self.shutdown_tx.send(true);
        let mut handle = self.handle;
        if tokio::time::timeout(Duration::from_secs(5), &mut handle)
            .await
            .is_err()
        {
            handle.abort();
        }
    }
}

/// Mesh-fallback chat handle shared with the gateway: owns the QUIC client
/// endpoint + expert router so `/v1/chat/completions` can fail over to
/// runtime-backed peers when the local runtime is unavailable.
pub struct MeshFallbackChat {
    pub endpoint: quinn::Endpoint,
    pub router: std::sync::Arc<crate::quic::ExpertRouter>,
    /// Live pins (node_id → rpc addr); replaced wholesale from the registry
    /// so the cert-pinning verifier always judges against current mesh state.
    pub pins: std::sync::Arc<tokio::sync::RwLock<crate::quic::PinMap>>,
}

impl MeshFallbackChat {
    /// Spawn with fresh pins from the registry: rpc address → node_id, so
    /// the QUIC transport's cert pinning verifier can prove "who answered".
    /// Pins are refreshed from the live registry before each dial batch via
    /// `set_pins`.
    pub fn spawn() -> anyhow::Result<Arc<Self>> {
        Self::spawn_with(crate::verify::VerificationTracker::shared(), 0)
    }

    /// Spawn sharing the mesh state's verification tracker so router-side
    /// commitment/replay outcomes feed the ping loop's evidence flush.
    /// `verify_percent` seeds the router's Phase 4 replay policy.
    pub fn spawn_with(
        verify: SharedVerifyTracker,
        verify_percent: u32,
    ) -> anyhow::Result<Arc<Self>> {
        let pins = std::sync::Arc::new(tokio::sync::RwLock::new(Default::default()));
        let endpoint = crate::quic::pinned_client_endpoint_dyn(pins.clone())
            .map_err(|e| anyhow::anyhow!("pinned client endpoint: {e}"))?;
        let router = std::sync::Arc::new(crate::quic::ExpertRouter::new_with_tracker(
            crate::quic::RouterOptions {
                nearby_rtt_ms: 50,
                dispatch_timeout: Duration::from_secs(30),
                // Phase 2: hybrid local + remote execution — kick the best
                // remote while local runs (latency hiding per the spec).
                speculative_remote: true,
                verify_percent,
                verify_tolerance: 0.0,
            },
            verify,
        ));
        Ok(Arc::new(Self {
            endpoint,
            router,
            pins,
        }))
    }

    /// Rebuild endpoint config pins from current registry rpc addresses.
    async fn refresh_pins(&self, state: &Arc<MeshState>) {
        let reg = state.registry.lock().await;
        let mut pins = self.pins.write().await;
        pins.clear();
        for ad in reg.all_ads() {
            if let Ok(addr) = ad.rpc_address.parse::<SocketAddr>() {
                pins.insert(ad.node_id.clone(), addr);
            }
        }
    }

    /// Build inputs for the router from live registry state: connectable
    /// peers (rtt, alive, advertised resources) + their advertised QUIC
    /// rpc addresses.
    async fn mesh_view(
        &self,
        state: &Arc<MeshState>,
        expert_id: Option<&str>,
    ) -> (
        Vec<(String, u64, bool, bool, Option<crate::hw::ResourceSample>)>,
        HashMap<String, SocketAddr>,
    ) {
        let reg = state.registry.lock().await;
        let peers: Vec<(String, u64, bool, bool, Option<crate::hw::ResourceSample>)> = reg
            .all_ads()
            .into_iter()
            .filter_map(|ad| {
                let health = reg.health_of(&ad.node_id)?;
                let alive = health.state != PeerState::Deprecated;
                let rtt = health.last_rtt_ms.unwrap_or(u64::MAX >> 2);
                // Expert dispatch only counts peers that advertise holding
                // THIS expert (legacy/unadvertised peers stay candidates).
                // Chat fallback is expert-agnostic, so it takes every alive
                // peer.
                let has_expert = match expert_id {
                    Some(id) => ad.hosts_expert(id),
                    None => true,
                };
                Some((
                    ad.node_id.clone(),
                    rtt,
                    has_expert,
                    alive,
                    ad.resources.clone(),
                ))
            })
            // Per-peer blocking (Phase 3): a blocked peer is never a
            // placement candidate — sending it work is exactly what it
            // was blocked to stop. Refusals still happen server-side if
            // a stale ad slips through; this keeps the router from even
            // building a plan around it.
            .filter(|(id, ..)| !state.peer_block.blocked(id))
            .collect();
        let addrs: HashMap<String, SocketAddr> = reg
            .all_ads()
            .into_iter()
            .filter(|ad| !ad.rpc_address.is_empty())
            // Replication dials live here too: a blocked peer is never a
            // weight source either way.
            .filter(|ad| !state.peer_block.blocked(&ad.node_id))
            .filter_map(|ad| {
                ad.rpc_address
                    .parse::<SocketAddr>()
                    .ok()
                    .map(|a| (ad.node_id.clone(), a))
            })
            .collect();
        (peers, addrs)
    }

    /// Dispatch one expert tensor query through the mesh (Phase 2 tensor
    /// transport) — the production entry point the local runtime uses to
    /// borrow an expert that lives on a peer.
    ///
    /// Plan: Local first when this node holds the weights, then peers that
    /// **advertise holding this expert** (resource-aware order), with the
    /// speculative pre-flight when local runs so a slow local attempt does
    /// not serialize a network leg behind it. Returns the activation
    /// vector plus the node that computed it (`None` = local).
    pub async fn dispatch_expert(
        &self,
        state: &Arc<MeshState>,
        expert_id: &str,
        input: &[f32],
    ) -> anyhow::Result<(Vec<f32>, Option<String>)> {
        // Kill switch (Phase 1): real dispatch into the mesh is the exact
        // workload the ceiling protects. A tripped switch refuses up front
        // — before demand is even recorded — so the plan is never built.
        if state.isolate.tripped() {
            anyhow::bail!(
                "kill switch tripped: {} — expert dispatch refused (local-only mode)",
                state
                    .isolate
                    .tripped_reason()
                    .unwrap_or_else(|| "resource ceiling breached".into())
            );
        }
        let (mut peers, addrs) = self.mesh_view(state, Some(expert_id)).await;
        // Cold-expert sharding (Phase 2): when nobody *advertises* this
        // expert, the deterministic shard ring says which node owns it. We
        // ask that owner rather than giving up — it either has the weights
        // (and answers) or honestly refuses, which is still better than
        // never trying. The ring is computed from the same ads every node
        // sees, so all nodes pick the same owner.
        // One registry read for both facts (ads + pings): the ring must be
        // computed from the same advertisement set every node sees, and the
        // lock is released before hashing or reading config.
        let (ads, rtts) = {
            let reg = state.registry.lock().await;
            let ads = reg.all_ads();
            let rtts: HashMap<String, u64> = ads
                .iter()
                .map(|ad| {
                    let rtt = reg
                        .health_of(&ad.node_id)
                        .and_then(|h| h.last_rtt_ms)
                        .unwrap_or(u64::MAX >> 2);
                    (ad.node_id.clone(), rtt)
                })
                .collect();
            (ads, rtts)
        };
        let advertised = ads.iter().any(|ad| ad.explicitly_hosts(expert_id));
        if !advertised {
            let own_id = state.config.read().await.node_id.clone();
            let ring = crate::replication::ShardMap::from_ads(&own_id, &ads);
            if let Some(owner) = ring.primary_owner(expert_id) {
                if owner != own_id && !peers.iter().any(|(id, ..)| *id == owner) {
                    let rtt = rtts.get(&owner).copied().unwrap_or(u64::MAX >> 2);
                    let resources = ads
                        .iter()
                        .find(|ad| ad.node_id == owner)
                        .and_then(|ad| ad.resources.clone());
                    tracing::debug!(
                        target: "mesh",
                        expert = %expert_id,
                        owner = %owner,
                        "no advertised holder — asking the cold-shard owner"
                    );
                    // Behind any advertised holder (there are none) and
                    // ahead of nothing: the owner is the candidate.
                    peers.push((owner, rtt, true, true, resources));
                }
            }
        }
        // Demand signal: this node wants this expert.
        state.record_activation(expert_id);
        self.refresh_pins(state).await;
        let req_id = format!("expert-{}-{}", expert_id, crate::bootstrap::now_ms());
        let store = state.expert_store().await;
        let local_store = store.clone();
        let local_run = move |x: &[f32]| match &local_store {
            Some(store) => store.run(expert_id, x),
            None => anyhow::bail!("no local expert weights attached"),
        };
        let mut completed = HashMap::new();
        // Attribution comes from the router: `served_by` is the node id
        // that actually replied, None when the local weights ran.
        let mut served_by = None;
        let dispatch_started = std::time::Instant::now();
        let out = self
            .router
            .execute_over_quic_speculative_attributed(
                &self.endpoint,
                &state.identity,
                &addrs,
                &req_id,
                expert_id,
                input,
                &peers,
                &mut completed,
                local_run,
                &mut served_by,
            )
            .await?;
        let dispatch_ms = dispatch_started.elapsed().as_millis() as u64;
        // Phase 4 freerider evidence: only COMPLETED peer-served runs feed
        // the ledger. Failed dispatches and local-served runs feed nothing —
        // honest blackouts and local wins must never read as outliers.
        if let Some(served) = served_by.as_ref() {
            let own_id = state.config.read().await.node_id.clone();
            if *served != own_id {
                let candidates: Vec<String> = peers
                    .iter()
                    .filter(|(id, ..)| *id != own_id)
                    .map(|(id, ..)| id.clone())
                    .collect();
                state
                    .freerider
                    .lock()
                    .await
                    .feed_dispatch(expert_id, &candidates, served, dispatch_ms);
            }
        }
        Ok((out, served_by))
    }

    /// One replication decision round (Phase 2): read demand, compare the
    /// mesh's advertised holders against the desired factor, and pull the
    /// missing hot experts from a holder that actually advertises them.
    ///
    /// Returns the actions performed (for logs/tests). Failures are
    /// per-expert and never fatal: the next round retries.
    pub async fn replication_tick(
        &self,
        state: &Arc<MeshState>,
        config: crate::replication::ReplicationConfig,
    ) -> Vec<crate::replication::ReplicationAction> {
        let Some(store) = state.expert_store().await else {
            return Vec::new();
        };
        // Kill switch: replication is discretionary mesh work — weight
        // fetches pause under a tripped switch and the next round retries.
        if state.isolate.tripped() {
            tracing::debug!(target: "mesh", "kill switch tripped — replication round skipped");
            return Vec::new();
        }
        let level = state.config.read().await.experts.replication_config();
        let stats = state.expert_stats_snapshot();
        let hot = stats.hot_set(config.hot_percent);
        if hot.is_empty() {
            return Vec::new();
        }
        // Addresses to dial: the expert-agnostic mesh view (chat/None), so
        // a peer is reachable for a transfer regardless of which experts it
        // advertises.
        let (_, addrs) = self.mesh_view(state, None).await;
        let reg = state.registry.lock().await;
        let ads = reg.all_ads();
        let replica_peers: Vec<crate::replication::ReplicaPeer> = ads
            .iter()
            .filter(|ad| {
                addrs.contains_key(&ad.node_id)
                    && reg
                        .health_of(&ad.node_id)
                        .map(|h| h.state != PeerState::Deprecated)
                        .unwrap_or(false)
            })
            .map(|ad| crate::replication::ReplicaPeer {
                node_id: ad.node_id.clone(),
                rtt_ms: reg
                    .health_of(&ad.node_id)
                    .and_then(|h| h.last_rtt_ms)
                    .unwrap_or(u64::MAX >> 2),
                hosted_experts: ad.hosted_experts.clone(),
                alive: true,
            })
            .collect();
        drop(reg);
        let local_hosted = store.hosted();
        let actions = crate::replication::plan_replication(
            &hot,
            &local_hosted,
            &replica_peers,
            crate::replication::ReplicationConfig {
                hot_percent: config.hot_percent,
                replication_factor: config.replication_factor,
                max_actions_per_round: level.max_actions_per_round,
            },
        );
        if actions.is_empty() {
            return actions;
        }
        self.refresh_pins(state).await;
        for action in &actions {
            let Some(addr) = addrs.get(&action.source_node).copied() else {
                continue;
            };
            match crate::quic::fetch_expert(
                &self.endpoint,
                addr,
                &action.source_node,
                &state.identity,
                &action.expert_id,
                Duration::from_secs(60),
            )
            .await
            {
                Ok(bytes) => match store.put_bytes(&bytes) {
                    Ok(weights) => tracing::info!(
                        target: "mesh",
                        expert = %action.expert_id,
                        source = %action.source_node,
                        bytes = bytes.len(),
                        held_by = action.holders_before,
                        "replicated hot expert ({} hidden × {} intermediate)",
                        weights.hidden,
                        weights.intermediate
                    ),
                    Err(e) => tracing::warn!(
                        target: "mesh",
                        expert = %action.expert_id,
                        error = %e,
                        "replicated expert rejected — checksum/format failure"
                    ),
                },
                Err(e) => tracing::warn!(
                    target: "mesh",
                    expert = %action.expert_id,
                    source = %action.source_node,
                    error = %e,
                    "hot-expert replication transfer failed (retried next round)"
                ),
            }
        }
        actions
    }

    /// Try local-strict chat (the gateway's existing forward); on failure
    /// fall back through the mesh. Returns content + which node answered
    /// (None = local).
    pub async fn chat_with_fallback(
        &self,
        state: &Arc<MeshState>,
        messages: &[serde_json::Value],
        local_try: impl FnOnce() -> anyhow::Result<String>,
    ) -> anyhow::Result<(String, Option<String>)> {
        req_id_tracker();
        let req_id = format!("chat-{}", crate::bootstrap::now_ms());
        let (peers, addrs) = self.mesh_view(state, None).await;
        self.refresh_pins(state).await;
        let local_attempt = local_try();
        let router = self.router.clone();
        let endpoint = self.endpoint.clone();
        let identity = state.identity.clone();
        let local_closed = local_attempt.is_err();
        match router
            .chat_over_quic(
                &endpoint,
                &identity,
                &addrs,
                &req_id,
                messages,
                &peers,
                || local_attempt,
            )
            .await
        {
            // The router reports which node actually answered: empty string =
            // local runtime, otherwise the node_id that served it (already
            // attribution-checked against the request's target on the wire).
            Ok((content, answered_by)) => {
                if local_closed && !answered_by.is_empty() {
                    Ok((content, Some(answered_by)))
                } else {
                    Ok((content, None))
                }
            }
            Err(e) => Err(e),
        }
    }
}

fn req_id_tracker() {}

/// Background loop handle.
pub struct MeshLoop {
    state: Arc<MeshState>,
    shutdown_tx: watch::Sender<bool>,
    shutdown: watch::Receiver<bool>,
    _handle: tokio::task::JoinHandle<()>,
}

impl MeshLoop {
    pub fn state(&self) -> Arc<MeshState> {
        Arc::clone(&self.state)
    }

    /// Signal the loop to stop cleanly, then abort the task.
    pub fn shutdown(self) {
        let _ = self.shutdown_tx.send(true);
        self._handle.abort();
    }

    /// Spawn the loop. Call `shutdown()` (or drop) to stop it.
    pub fn spawn(state: Arc<MeshState>, ping_interval: Duration) -> Self {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let st = Arc::clone(&state);
        let mut loop_rx = shutdown_rx.clone();
        let handle = tokio::spawn(async move {
            // No fixed bootstrap client: the rotation lives in
            // state.discovery (discovery.rs) — a dead primary degrades one
            // tick, then the loop dials the next server in the order.
            run_loop(st, ping_interval, &mut loop_rx).await;
        });
        Self {
            state,
            shutdown_tx,
            shutdown: shutdown_rx,
            _handle: handle,
        }
    }

    pub fn is_running(&self) -> bool {
        !self.shutdown.has_changed().map(|c| !c).unwrap_or(true)
    }
}

/// One complete loop iteration: refresh peers, ping all, advertise self.
/// Bootstrap legs run through the discovery rotation (discovery.rs): the
/// last-good server first, then the rest — one dead bootstrap degrades a
/// single tick instead of the process lifetime.
async fn run_loop(
    state: Arc<MeshState>,
    ping_interval: Duration,
    shutdown_rx: &mut watch::Receiver<bool>,
) {
    let mut tick = interval(ping_interval);
    tick.set_missed_tick_behavior(MissedTickBehavior::Skip);

    loop {
        if *shutdown_rx.borrow() {
            break;
        }

        let (model, my_ad_model) = {
            let cfg = state.config.read().await;
            match &cfg.model {
                Some(m) => (Some(m.clone()), (m.id.clone(), m.quant.clone())),
                None => (None, (String::new(), String::new())),
            }
        };

        // Hot-reload the per-peer block list: the only consumer outside
        // startup, and every other gate (QUIC server, placement, status)
        // reads the shared set — so a config.toml edit (or `jerith config
        // set`) takes effect within one tick, no restart.
        {
            let cfg = state.config.read().await;
            state.peer_block.set(&cfg.peers.blocked);
        }

        // Hot-reload the discovery rotation (Phase 1): the operator can
        // repoint the node at new bootstraps without a restart. The
        // learned set is state-owned (peer gossip) and persisted.
        {
            let cfg = state.config.read().await;
            let mut d = state.discovery.lock().await;
            d.configured = crate::discovery::sanitize(&cfg.discovery_seed_servers(), None);
        }

        // Initialize the registry's model gate on first iteration (or after
        // a model switch event, which cleared it).
        {
            let mut reg = state.registry.lock().await;
            let gate_matches =
                reg.model().map(|m| m.id.clone()) == model.as_ref().map(|m| m.id.clone());
            if !gate_matches {
                *reg = PeerRegistry::new(model.clone());
            }
        }

        // Phase 5 escrow wiring (live chain contract): sync the ledger to
        // the chain's /stake view (honest join gate), then escrow exactly
        // once per process. Both are quiet no-ops without the chain,
        // api token, or when staking is disabled (required_uj = 0).
        state.stake.refresh_from_chain().await;
        state.stake.lock_required_once().await;

        if let Some(m) = &model {
            // Signed peers-fetch: the bootstrap's rate lane attributes the
            // hit to OUR identity bucket (generous allowance for NAT-
            // packed deployments) instead of the per-IP anonymous floor.
            // Disseminated rotation (discovery.rs): first good server wins
            // and is remembered as the last-good for the following calls.
            let order = { state.discovery.lock().await.order(None) };
            // Borrows, not moves: try_each's closure is FnMut — a move of
            // the Arc/identity would fail the second attempt.
            let model_id = &m.id;
            let identity_ref = &state.identity;
            let fetched_leg = crate::discovery::try_each(&order, |url| {
                let client = BootstrapClient::new(url.to_string());
                async move {
                    client
                        .fetch_peers_signed(model_id, identity_ref)
                        .await
                }
            })
            .await;
            match fetched_leg {
                Ok((peers, used)) => {
                    {
                        let mut d = state.discovery.lock().await;
                        // Hostile-ad audit (F2): "answered" alone does not
                        // earn stickiness. A lying bootstrap can hoard the
                        // top slot with valid-but-EMPTY responses and starve
                        // discovery's peer view; an empty result clears the
                        // pin so the next tick diversifies from config order.
                        if peers.is_empty() {
                            d.clear_last_good();
                        } else {
                            d.note_good(&used);
                        }
                    }
                    let mut reg = state.registry.lock().await;
                    let fetched = peers.len();
                    for ad in peers {
                        reg.upsert(ad);
                    }
                    tracing::info!(
                        target: "mesh",
                        fetched,
                        bootstrap = %used,
                        "peer list refreshed from bootstrap"
                    );
                }
                Err(e) => {
                    // Bootstrap turned this node away (signature/expiry) or
                    // is unreachable (or every server failed) — must be
                    // visible, not silent.
                    tracing::warn!(
                        target: "mesh",
                        error = %e,
                        "bootstrap list refresh failed — degraded (local-only)"
                    );
                    let _ = state.contribution_healthy.send(false);
                }
            }

            // Gossip the discovery list (Phase 1): every fetched ad may
            // carry bootstrap servers its sender rotates through. Merging
            // keeps the rotation fresh with zero config; https-only +
            // capped like every other path into the list.
            {
                let ads: Vec<String> = {
                    let reg = state.registry.lock().await;
                    reg.all_ads()
                        .iter()
                        .flat_map(|ad| ad.discovery_servers.iter().cloned())
                        .collect()
                };
                let mut d = state.discovery.lock().await;
                if d.merge_learned(&ads) {
                    if let Err(e) = d.save() {
                        tracing::warn!(target: "mesh", error = %e, "discovery list persist failed");
                    } else {
                        tracing::info!(
                            target: "mesh",
                            learned = d.learned.len(),
                            "discovery list grew via peer gossip"
                        );
                    }
                }
            }

            // Ping every connectable peer — via the last-good bootstrap's
            // view: dead-server rotation happens at the fetch/advertise
            // legs; the ping pass rides the server that just answered.
            let ping_url = {
                let d = state.discovery.lock().await;
                d.last_good
                    .clone()
                    .or_else(|| d.order(None).first().cloned())
            };
            if let Some(url) = ping_url {
                ping_all(&state, &BootstrapClient::new(url)).await;
            }

            // Resource pooling (Phase 1): refresh OUR live sample right
            // before advertising so the mesh always placement-decides on
            // fresh data. Blocking bits (nvidia-smi exec, sysinfo refresh)
            // run off the loop thread.
            let sample = tokio::task::spawn_blocking(crate::hw::sample)
                .await
                .unwrap_or_default();
            *state.resources.lock().await = Some(sample);

            // Advertise ourselves (signed); the addresses point at our
            // routable interface (advertise_host override or the
            // default-route detection) so OTHER nodes can dial us directly
            // over QUIC — 127.0.0.1 ads made every remote dial a self-dial.
            let cfg = state.config.read().await;
            let advertise_host = cfg.advertised_host();
            // Phase 2 tensor transport: publish the experts actually held
            // (None when no store is attached — placement then keeps the
            // historical "assume hosted" behavior for this node).
            let (hosted_experts, hosted_expert_count) = match state.experts.lock().await.as_ref() {
                Some(store) => PeerAd::advertise_hosted_experts(&store.hosted()),
                None => (None, None),
            };
            let ad = PeerAd {
                node_id: cfg.node_id.clone(),
                address: format!("{advertise_host}:{}", cfg.gateway_port),
                rpc_address: format!("{advertise_host}:{}", cfg.effective_rpc_port()),
                model_id: my_ad_model.0,
                quant: my_ad_model.1,
                contribution_percent: cfg.contribution_percent,
                last_seen_ms: now_ms(),
                role: cfg.role,
                resources: state.resources.lock().await.clone(),
                hosted_experts,
                hosted_expert_count,
                // Gossip our rotation (Phase 1): capped, sanitized —
                // every peer that hears this ad can dial a working
                // bootstrap even if OUR primary dies.
                discovery_servers: {
                    let d = state.discovery.lock().await;
                    d.ad_servers(None)
                },
            };
            let identity_ref = &state.identity;
            let ad_ref = &ad;
            let advertise_order = state.discovery.lock().await.order(None);
            let advertise_leg = crate::discovery::try_each(&advertise_order, |url| {
                let client = BootstrapClient::new(url.to_string());
                async move { client.advertise(ad_ref, identity_ref).await }
            })
            .await;
            match advertise_leg {
                Ok(((), used)) => {
                    state.discovery.lock().await.note_good(&used);
                    tracing::debug!(target: "mesh", bootstrap = %used, "advertised to bootstrap");
                }
                Err(e) => {
                    // An advertise failure means the mesh can't see us — warn.
                    tracing::warn!(target: "mesh", error = %e, "advertise rejected/failed on every bootstrap");
                }
            }
        }

        // Phase 4: ringer spot-check (per config) against a random alive
        // peer, THEN (same cadence) flush queued DishonestResults evidence
        // to the bootstrap — never on the inference hot path.
        // Statistical freerider detection (Phase 4): evaluate the
        // coordinator ledger once per tick — pure in-memory work, no I/O on
        // the hot path. Crossings queue signed Leeching evidence through the
        // same flush cadence as every other fraud report; the config gate
        // is read per tick so `jerith config set` (or an edit) lands live.
        if state.config.read().await.freerider_enabled {
            let targets = state.freerider.lock().await.evaluate();
            if !targets.is_empty() {
                tracing::warn!(
                    target: "mesh",
                    targets = ?targets,
                    "statistical freerider evidence queued (Leeching)"
                );
                let refs: Vec<&str> = targets.iter().map(|s| s.as_str()).collect();
                state
                    .verify
                    .flag_fraud(crate::bootstrap_server::FraudKind::Leeching, &refs)
                    .await;
            }
        }
        maybe_spot_check_ringer(&state).await;
        let flushed_targets = state.verify.flush_fraud_reports(&state).await;
        if !flushed_targets.is_empty() {
            tracing::info!(
                target: "mesh",
                count = flushed_targets.len(),
                "verification evidence flushed"
            );
            // Phase 5: each bootstrap-accepted report becomes a signed slash
            // notice + ledger entry (no-op while staking is disabled).
            for target in &flushed_targets {
                let evidence = format!("dishonest-results|{}|{target}", state.identity.node_id());
                state
                    .stake
                    .notice(target, "dishonest_results", &evidence)
                    .await;
            }
        }

        // Uptime metering on every tick.
        {
            let mut meters = state.meters.lock().await;
            meters.uptime_seconds += ping_interval.as_secs();
            // Node contributes if it has served anything, or is a pure
            // Requester consuming modestly.
            let healthy = meters.tokens_served > 0
                || meters.expert_activations > 0
                || state.config.read().await.role == NodeRole::Requester;
            let _ = state.contribution_healthy.send(healthy);
        }

        tokio::select! {
            _ = tick.tick() => {}
            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() { break; }
            }
        }
    }
    // Phase 5: return the escrow on the way out — best effort, the chain
    // may be down (the persisted escrowed_uj marker keeps lock idempotent
    // across restarts either way).
    if state.stake.unlock_all().await {
        tracing::info!(target: "economics", "stake escrow returned on shutdown");
    }
    tracing::info!("mesh loop stopped");
}

/// Ping all connectable peers with a minimal-resource, signed request.
/// The reply must carry a valid Ed25519 health signature from the peer's
/// own key before we count it as alive — spoofed liveness is rejected.
async fn ping_all(state: &Arc<MeshState>, bootstrap: &BootstrapClient) {
    let targets: Vec<(String, String)> = {
        let reg = state.registry.lock().await;
        reg.connectable()
            .into_iter()
            .map(|(ad, _)| (ad.node_id.clone(), format!("http://{}/health", ad.address)))
            .collect()
    };
    if targets.is_empty() {
        return;
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(4))
        .build()
        .expect("reqwest client");
    let my_id = state.identity.node_id();

    let mut failed: Vec<String> = Vec::new();
    for (node_id, url) in targets {
        let started = std::time::Instant::now();
        let stamp = now_ms();
        let payload = crate::signing::ping_payload(&my_id, &node_id, stamp);
        // Freeze the stamp: the payload embeds it, so the header stamp and
        // the signed bytes must come from the SAME clock read (ms-boundary
        // races under load made pings intermittently rejected).
        let auth = crate::signing::WireAuth::with_stamp(&state.identity, stamp, payload.as_bytes());
        let result = client
            .get(&url)
            .header("x-jerith-pubkey", &auth.pubkey)
            .header("x-jerith-stamp", auth.stamp_ms.to_string())
            .header("x-jerith-sig", &auth.sig)
            .header("x-jerith-from", &my_id)
            .send()
            .await;
        match result {
            Ok(resp) if resp.status().is_success() => {
                // Verify the signed health reply attributes it to the peer.
                let rtt = started.elapsed().as_millis() as u64;
                let ok = verify_health_reply(&state, &node_id, resp).await;
                if ok {
                    let mut reg = state.registry.lock().await;
                    reg.record_ping_ok(&node_id, rtt);
                } else {
                    tracing::warn!(peer = %node_id, "health reply signature invalid — not counted alive");
                    failed.push(node_id);
                }
            }
            _ => {
                failed.push(node_id);
            }
        }
    }

    if !failed.is_empty() {
        let (report_evidence, my_id) = {
            let reg = state.registry.lock().await;
            let evidence: Vec<(String, u32)> = failed
                .iter()
                .filter_map(|id| {
                    let h = reg.health_of(id)?;
                    // Submit evidence only on the deprecating (3rd) failure.
                    (h.consecutive_failures == 3).then_some((id.clone(), h.consecutive_failures))
                })
                .collect();
            (evidence, state.identity.node_id())
        };
        let mut reg = state.registry.lock().await;
        for node_id in &failed {
            let was_alive = matches!(reg.state_of(node_id), Some(PeerState::Alive));
            reg.record_ping_fail(node_id);
            if was_alive && reg.state_of(node_id) == Some(PeerState::Deprecated) {
                tracing::warn!(node = %node_id, "peer deprecated");
                // Signed report (bootstrap now requires wire auth to count
                // eviction votes).
                let _ = bootstrap
                    .report_dead(node_id, &state.identity.node_id(), &state.identity)
                    .await;
            }
        }
        drop(reg);
        // Failed-worker evidence to the bootstrap (signed, dedup'd server
        // side). Only for transitions to Deprecated (3rd failure).
        for (node_id, fails) in report_evidence {
            if fails == 3 {
                match bootstrap
                    .report_fraud(
                        crate::bootstrap_server::FraudKind::FailedWorker,
                        &node_id,
                        &my_id,
                        &state.identity,
                    )
                    .await
                {
                    Ok(v) => tracing::info!(
                        target: "mesh",
                        node = %node_id,
                        reply = %v,
                        "failed-worker evidence submitted"
                    ),
                    Err(e) => tracing::debug!(
                        target: "mesh",
                        node = %node_id,
                        error = %e,
                        "fraud evidence submission failed"
                    ),
                }
            }
        }
    }
}

/// Phase 4 ringer spot-check: with `ringer_percent` probability per tick,
/// send a signed builtin known-answer expert query to one random alive peer
/// and grade the exact answer. A mismatch (or tampered commitment) queues
/// DishonestResults evidence — timeouts and infra failures do NOT, they
/// only say nothing. Runs off the inference hot path (loop cadence).
async fn maybe_spot_check_ringer(state: &Arc<MeshState>) {
    let percent = state.config.read().await.ringer_percent;
    if percent == 0 || rand::random::<u32>() % 100 >= percent.min(100) {
        return;
    }
    // Pick a random alive peer that offers QUIC work.
    let mut candidates: Vec<(String, SocketAddr)> = {
        let reg = state.registry.lock().await;
        reg.all_ads()
            .into_iter()
            .filter(|ad| ad.node_id != state.identity.node_id())
            .filter_map(|ad| {
                let alive = reg
                    .health_of(&ad.node_id)
                    .map(|h| h.state != PeerState::Deprecated)
                    .unwrap_or(false);
                if !alive {
                    return None;
                }
                ad.rpc_address
                    .parse::<SocketAddr>()
                    .ok()
                    .map(|a| (ad.node_id.clone(), a))
            })
            .collect()
    };
    if candidates.is_empty() {
        return;
    }
    let pick = rand::random::<usize>() % candidates.len();
    let (peer_id, addr) = candidates.swap_remove(pick);

    // Pin only this peer: cert must prove the node we mean to dial.
    let mut pins: crate::quic::PinMap = HashMap::new();
    pins.insert(peer_id.clone(), addr);
    let endpoint = match crate::quic::pinned_client_endpoint(std::sync::Arc::new(pins)) {
        Ok(e) => e,
        Err(e) => {
            tracing::debug!(target: "mesh", error = %e, "ringer endpoint unavailable");
            return;
        }
    };

    let seed = now_ms();
    let (expert_id, input, expected) = crate::verify::ringer_task(seed);
    let req_id = format!("ringer-{seed}");
    let mut req = crate::quic::NodeRpcRequest::ExpertQuery {
        req_id: req_id.clone(),
        expert_id: expert_id.to_string(),
        input: input.clone(),
        from_node_id: state.identity.node_id(),
        stamp_ms: 0,
        sig: String::new(),
    };
    crate::quic::sign_request(&mut req, &state.identity);
    let result = tokio::time::timeout(
        Duration::from_secs(6),
        crate::quic::call(
            &endpoint,
            addr,
            &peer_id,
            &req,
            &state.identity,
            Duration::from_secs(5),
        ),
    )
    .await;
    match result {
        Ok(Ok(crate::quic::NodeRpcReply::ExpertResult {
            req_id: rid,
            output,
            node_id: server_id,
            commitment,
        })) => {
            if server_id != peer_id || rid != req_id {
                // Attribution failure alone is not proof of dishonesty in
                // this path (could be glitched bootstrap entries); log it.
                tracing::warn!(target: "mesh", peer = %peer_id, got = %server_id, "ringer reply attribution mismatch");
                return;
            }
            // Same accounting as router-dispatched work (only after the
            // attribution check passes).
            state.verify.note_expert_result().await;
            if let Some(c) = &commitment {
                let ok = crate::verify::verify_commitment(
                    &req_id, expert_id, &server_id, &input, &output, c,
                );
                state.verify.note_commitment(&server_id, ok).await;
                if !ok {
                    return; // already flagged by note_commitment
                }
            }
            let agree = crate::verify::outputs_agree(&expected, &output, 0.0);
            state.verify.note_replay(agree).await;
            if agree {
                tracing::debug!(target: "mesh", peer = %peer_id, "ringer challenge passed");
            } else {
                state.verify.flag_dishonest(&[&peer_id]).await;
                tracing::warn!(
                    target: "mesh",
                    peer = %peer_id,
                    "ringer challenge FAILED — dishonest-result evidence queued"
                );
            }
        }
        Ok(Ok(crate::quic::NodeRpcReply::Error { req_id: _, message })) => {
            tracing::debug!(target: "mesh", peer = %peer_id, message = %message, "ringer rejected by peer");
        }
        Ok(Ok(other)) => {
            tracing::debug!(target: "mesh", peer = %peer_id, "unexpected ringer reply: {other:?}");
        }
        _ => {
            // Timeout / transport failure: not evidence of dishonesty.
            tracing::debug!(target: "mesh", peer = %peer_id, "ringer dispatch failed (infra)");
        }
    }
}

/// Verify a peer's signed health reply.
/// Body must include node_id, pubkey, stamp, sig, uptime_seconds,
/// contribution_percent — signature over the canonical health payload,
/// made by a key whose fingerprint equals the claimed node_id.
async fn verify_health_reply(
    _state: &Arc<MeshState>,
    expected_node_id: &str,
    resp: reqwest::Response,
) -> bool {
    #[derive(serde::Deserialize)]
    struct HealthReply {
        node_id: String,
        pubkey: String,
        stamp_ms: u64,
        sig: String,
        uptime_seconds: u64,
        contribution_percent: u64,
    }
    let Ok(reply) = resp.json::<HealthReply>().await else {
        return false;
    };
    if reply.node_id != expected_node_id {
        return false;
    }
    // The pubkey fingerprint must match the claimed identity.
    if reply.pubkey.len() != 64 || &reply.pubkey[..16] != reply.node_id {
        return false;
    }
    let payload = crate::signing::health_payload(
        &reply.node_id,
        reply.stamp_ms,
        reply.uptime_seconds,
        reply.contribution_percent,
    );
    crate::signing::verify(&reply.pubkey, payload.as_bytes(), &reply.sig)
        && now_ms().saturating_sub(reply.stamp_ms) <= crate::signing::STAMP_MAX_AGE_MS
}

/// Statistics about the most recent ping cycle, surfaced to the web UI.
#[derive(Debug, Clone, Copy, Default)]
pub struct PingStats {
    pub peers_pinged: u32,
    pub peers_alive: u32,
    pub peers_deprecated: u32,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bootstrap::DEPRECATE_AFTER_FAILED_PINGS;
    use crate::config::*;

    #[test]
    fn meters_snapshot_resets() {
        let mut m = Meters {
            tokens_served: 100,
            expert_activations: 7,
            bytes_relayed: 5_000,
            uptime_seconds: 60,
            tokens_consumed: 20,
            mines_attempted: 2,
            mines_won: 1,
        };
        let snapped = m.snapshot_reset();
        assert_eq!(snapped.tokens_served, 100);
        assert_eq!(m, Meters::default());
    }

    #[test]
    fn deprecate_count_matches_spec_threshold() {
        assert_eq!(DEPRECATE_AFTER_FAILED_PINGS, 3);
    }

    #[test]
    fn min_contribution_is_25() {
        assert_eq!(MIN_CONTRIBUTION_PERCENT, 25);
    }
}
