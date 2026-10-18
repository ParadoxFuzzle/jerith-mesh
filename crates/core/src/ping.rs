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

use crate::bootstrap::{BootstrapClient, FetchedPeers, PeerAd, PeerRegistry, PeerState, now_ms};
use crate::config::{NodeConfig, PrivacyLevel};
use serde::Serialize;

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
    /// Real-time contribution measurement (FEATURES §2): the node's own
    /// rolling measured serve share — per-tick diff of the cumulative
    /// meters, EWMA-smoothed, and signed into health replies.
    pub contrib: std::sync::Arc<tokio::sync::Mutex<crate::contrib::ContributionMeter>>,
    /// Per-peer verdicts over peers' SIGNED measurement blocks (the
    /// verification half): tampered blocks and demonstrated declared-vs-
    /// measured shortfalls queue Leeching evidence on the tick cadence.
    pub contrib_ledger: std::sync::Arc<tokio::sync::Mutex<crate::contrib::PeerContribLedger>>,
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
    pub freerider: std::sync::Arc<tokio::sync::Mutex<crate::freerider::FreeriderDetector>>,
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
    /// Phase 3 speculative prefetch guard: at most ONE background
    /// prefetch walk at a time (activations can arrive faster than a
    /// disk can read — the guard keeps prefetch polite; a skipped
    /// activation costs nothing, the next one restates demand).
    pub prefetching: std::sync::atomic::AtomicBool,
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
    pub discovery: std::sync::Arc<tokio::sync::Mutex<crate::discovery::DiscoveryServers>>,
    /// The live local memory store behind the memory.* skills — owned
    /// HERE (not in the gateway) so the config hot-reload can live-swap
    /// it without rebuilding the HTTP state. Boot swaps it in once the
    /// configured store opens; None → skills answer the honest
    /// "memory backend not configured".
    pub memory: tokio::sync::RwLock<Option<crate::memory::SharedMemory>>,
    /// Backoff for failed memory-store swaps during config reload: when
    /// the newly-configured store won't open (postgres down), skip the
    /// retry for this long so a dead neighbor doesn't spam every tick.
    memory_swap_backoff_until: std::sync::Mutex<Option<std::time::Instant>>,
    /// §2 model-switch machinery: a graceful mesh EXIT is latched (the
    /// `model` cold change in a reload plan) — the bootstrap got our
    /// signed /v1/leave, the advertise tick stops, and /status reports
    /// the pending restart instead of a wedge. One-shot: the executor
    /// fires once per process via compare_exchange.
    pub left_mesh_latch: std::sync::atomic::AtomicBool,
    /// Why the latch fired (typed journal: config model id we would serve
    /// after a restart), surfaced beside the latch in /status.
    pub pending_model: tokio::sync::RwLock<Option<String>>,
    /// §10 reputation gossip: bootstrap-verified score stamps we may carry
    /// inside ads. Newest-subject-per-stamp, capped; every stamp in here
    /// verified against an ORIGIN pin WE made from a configured/learned
    /// bootstrap's own published key — nothing unverified rides out.
    pub reputation_cache: std::sync::Arc<tokio::sync::Mutex<Vec<crate::signing::ReputationStamp>>>,
    /// Where those counters persist (see
    /// [`set_expert_stats`](Self::set_expert_stats)).
    expert_stats_path: std::sync::Mutex<std::path::PathBuf>,
    /// Pre-join mesh-compat verdict (FEATURES §12): the typed, readable
    /// result of comparing OUR model identity against the bootstrap's
    /// per-model/per-quant census, taken once on the loop's first
    /// successful bootstrap leg. "Healthy" nodes replace it with nothing
    /// interesting; the mismatch cases explain WHY the mesh looks empty
    /// (same model, wrong quant — N nodes could still take you).
    pub compat: tokio::sync::RwLock<Option<CompatVerdict>>,
}

/// One pre-join compat verdict over a local model identity + the live
/// bootstrap census (pure fn below drives it; see `compat_verdict`).
#[derive(Debug, Clone, Serialize)]
pub struct CompatVerdict {
    /// "healthy" | "wrong_quant" | "no_peers_on_model" | "bootstrap_unreachable"
    pub verdict: String,
    pub message: String,
    pub checked_at_ms: u64,
    /// Live peers running OUR exact model id (any quant).
    pub peers_on_model: u64,
    /// Of those, our exact quant (0 under wrong_quant).
    pub peers_same_quant: u64,
    /// Mismatch quant → count rows for our model (readable operator hint).
    pub other_quants: Vec<(String, u64)>,
    /// Live models we could join instead, when our model has no peers at
    /// all ("{model_id} ({count} nodes)" rows) — the honest answer to a
    ///"brand-new mesh?" suspicion.
    pub other_models: Vec<(String, u64)>,
}

/// Compare OUR identity against the census: determine if the same model, wrong quant
/// case, the live-alternatives case, or healthy — in that priority.
pub fn compat_verdict(
    local: &crate::models::ModelId,
    census: &[crate::bootstrap::ModelCensusRow],
) -> CompatVerdict {
    let mine = census
        .iter()
        .find(|row| row.model_id == local.id)
        .cloned()
        .unwrap_or_default();
    let peers_same_quant = mine.quants.get(&local.quant).copied().unwrap_or(0);
    let other_quants: Vec<(String, u64)> = mine
        .quants
        .iter()
        .filter(|(q, _)| q.as_str() != local.quant)
        .map(|(q, c)| (q.clone(), *c))
        .collect();
    let other_models: Vec<(String, u64)> = census
        .iter()
        .filter(|row| row.model_id != local.id && row.count > 0)
        .map(|row| (row.model_id.clone(), row.count))
        .collect();
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let mut v = CompatVerdict {
        verdict: "healthy".into(),
        message: format!(
            "mesh compatible — {} peer(s) on {} quant {}",
            peers_same_quant, local.id, local.quant
        ),
        checked_at_ms: now_ms,
        peers_on_model: mine.count,
        peers_same_quant,
        other_quants,
        other_models,
    };
    if peers_same_quant > 0 {
        return v;
    }
    if mine.count > 0 {
        // The model runs live on the mesh but NOT with our quant: peers
        // admit exact quant matches only, so this node would join an
        // empty registry. Say both halves readably.
        let list = v
            .other_quants
            .iter()
            .map(|(q, c)| format!("{q} ({c} node{})", if *c == 1 { "" } else { "s" }))
            .collect::<Vec<_>>()
            .join(", ");
        v.verdict = "wrong_quant".into();
        v.message = format!(
            "same model, wrong quant — the mesh runs {} on quant(s) [{list}]; you run '{}', which no peer admits (peers match id+quant exactly). `jerith config set model.quant <theirs>` is the fix",
            local.id, local.quant,
        );
        return v;
    }
    // No live peers on our model at all: either this mesh is brand-new,
    // or the model spelling drifts from the fleet. Surface what IS live
    // so the operator can pick a lane.
    v.verdict = "no_peers_on_model".into();
    if v.other_models.is_empty() {
        v.message = format!(
            "no live peers on {} yet — you ARE the mesh (brand-new network? check the bootstrap server if peers were expected)",
            local.id
        );
    } else {
        let alts = v
            .other_models
            .iter()
            .map(|(m, c)| format!("{m} ({c})"))
            .collect::<Vec<_>>()
            .join(", ");
        v.message = format!(
            "no live peers on {} — models live on this mesh right now: {alts}",
            local.id
        );
    }
    v
}

impl MeshState {
    pub fn new(config: NodeConfig) -> anyhow::Result<(Arc<Self>, watch::Receiver<bool>)> {
        let (tx, rx) = watch::channel(true);
        let identity = config.identity()?;
        let stake = crate::stake::StakeManager::new(
            &config.crypto,
            Some(identity.clone()),
            // The operator resolver half: /stake/* lives on the chain
            // node api (config `chain_url`), not the miner endpoint.
            Some(config.chain_url.trim().trim_end_matches('/').to_string())
                .filter(|u| !u.is_empty()),
        );
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
            contrib: std::sync::Arc::new(tokio::sync::Mutex::new(
                crate::contrib::ContributionMeter::default(),
            )),
            contrib_ledger: std::sync::Arc::new(tokio::sync::Mutex::new(
                crate::contrib::PeerContribLedger::default(),
            )),
            started_at: std::time::Instant::now(),
            resources: Mutex::new(None),
            verify: crate::verify::VerificationTracker::shared(),
            stake,
            experts: Mutex::new(None),
            expert_stats: std::sync::Arc::new(std::sync::Mutex::new(
                crate::expert_stats::ExpertStats::load(&crate::expert_stats::default_stats_path()),
            )),
            expert_stats_path: std::sync::Mutex::new(crate::expert_stats::default_stats_path()),
            prefetching: std::sync::atomic::AtomicBool::new(false),
            compat: tokio::sync::RwLock::new(None),
            isolate: std::sync::Arc::new(crate::isolate::IsolateState::default()),
            peer_block,
            freerider: std::sync::Arc::new(tokio::sync::Mutex::new(
                crate::freerider::FreeriderDetector::default(),
            )),
            discovery: std::sync::Arc::new(tokio::sync::Mutex::new(discovery)),
            memory: RwLock::new(None),
            memory_swap_backoff_until: std::sync::Mutex::new(None),
            left_mesh_latch: std::sync::atomic::AtomicBool::new(false),
            pending_model: tokio::sync::RwLock::new(None),
            reputation_cache: std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new())),
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

    /// Phase 3 speculative prefetch (FEATURES §13): the MoE pipeline
    /// walks layer by layer, so the layer AFTER the one just served is
    /// the natural prediction target. Warm the LRU with this node's
    /// most-demanded, locally-hosted experts of the following layer(s)
    /// while the requester still consumes the current layer's output.
    ///
    /// Guarded and honest by construction:
    /// - config knobs read live (`[experts].prefetch_*` — hot-reloadable);
    /// - one background walk at a time (the `prefetching` guard; the
    ///   skipped activation costs nothing, the next one restates demand);
    /// - only experts already hosted locally are touched — prefetch
    ///   warms the resident cache, it never pulls mesh weights;
    /// - the walk runs detached (spawned) so the serve path never waits
    ///   on a disk read — the call itself is sync, safe from both the
    ///   QUIC serve handler and the dispatch path.
    pub fn spawn_prefetch_after_serve(self: &Arc<Self>, served_expert: &str) {
        // One background walk at a time; activations arrive faster than a
        // disk can read, so the guard keeps prefetch polite (a skipped
        // activation costs nothing — the next one restates demand). The
        // knob check rides INSIDE the walk so a disabled config only
        // costs one atomic swap.
        if self
            .prefetching
            .swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            return; // a walk is already in flight; this activation restates demand
        }
        let state = Arc::clone(self);
        let served = served_expert.to_string();
        tokio::spawn(async move {
            let (enabled, per_layer, horizon) = {
                let cfg = state.config.read().await;
                (
                    cfg.experts.prefetch_enabled,
                    cfg.experts.prefetch_per_layer,
                    cfg.experts.prefetch_horizon,
                )
            };
            let reset = || {
                state
                    .prefetching
                    .store(false, std::sync::atomic::Ordering::SeqCst)
            };
            if !enabled || per_layer == 0 || horizon == 0 {
                reset();
                return;
            }
            let Some(store) = state.experts.lock().await.clone() else {
                reset();
                return; // no expert store attached (config-only node)
            };
            let candidates = {
                match state.expert_stats.lock() {
                    Ok(stats) => stats.prefetch_candidates(&served, horizon, per_layer),
                    Err(_) => Vec::new(), // poisoned lock: skip this walk
                }
            };
            for id in candidates {
                if !store.contains(&id) {
                    continue; // prefetch warms local weights only
                }
                match store.prefetch(&id) {
                    Ok(()) => tracing::debug!(
                        target: "mesh",
                        expert = %id,
                        after = %served,
                        "speculative prefetch: next-layer expert preloaded"
                    ),
                    Err(e) => tracing::debug!(
                        target: "mesh",
                        expert = %id,
                        error = %e,
                        "speculative prefetch load failed"
                    ),
                }
            }
            reset();
        });
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
    /// peers (rtt, alive, advertised resources, VERIFIED reputation (§10
    /// / Phase 5)) + their advertised QUIC rpc addresses.
    /// Dispatch-view wrapper: the (candidates, dial-addrs) pair without
    /// the detailed ad snapshot.
    async fn mesh_view(
        &self,
        state: &Arc<MeshState>,
        expert_id: Option<&str>,
    ) -> (Vec<crate::quic::MeshCandidate>, HashMap<String, SocketAddr>) {
        let (peers, addrs, _, _) = self.mesh_view_detailed(state, expert_id).await;
        (peers, addrs)
    }

    /// The FULL dispatch view in ONE registry lock: candidates, dial
    /// addresses, the raw ad snapshot and per-peer RTTs. The hot path
    /// (`dispatch_expert` + the cold-shard ring) used to acquire the
    /// global registry lock TWICE per dispatch and clone the entire ad
    /// set three times (once per view leg, once again for the ring) —
    /// every pass now reads one snapshot. Verified reputation (Phase 5)
    /// looks up the bounded pin-wall cache inside the same pass.
    async fn mesh_view_detailed(
        &self,
        state: &Arc<MeshState>,
        expert_id: Option<&str>,
    ) -> (
        Vec<crate::quic::MeshCandidate>,
        HashMap<String, SocketAddr>,
        Vec<PeerAd>,
        HashMap<String, u64>,
    ) {
        let verified = verified_scores(state).await;
        let (ads, rtts, peers, addrs) = {
            let reg = state.registry.lock().await;
            let ads: Vec<PeerAd> = reg.all_ads();
            let rtts: HashMap<String, u64> = ads
                .iter()
                .map(|ad| {
                    let rtt = reg.routing_rtt_of(&ad.node_id).unwrap_or(u64::MAX >> 2);
                    (ad.node_id.clone(), rtt)
                })
                .collect();
            let peers: Vec<crate::quic::MeshCandidate> = ads
                .iter()
                .filter_map(|ad| {
                    let health = reg.health_of(&ad.node_id)?;
                    let alive = health.state != PeerState::Deprecated;
                    // Routing-grade RTT: EWMA-smoothed when available, raw
                    // last sample before the second ping lands.
                    let rtt = health
                        .smoothed_rtt_ms
                        .or(health.last_rtt_ms)
                        .unwrap_or(u64::MAX >> 2);
                    // Expert dispatch only counts peers that advertise
                    // holding THIS expert (legacy/unadvertised peers stay
                    // candidates). Chat fallback is expert-agnostic, so it
                    // takes every alive peer.
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
                        verified.get(&ad.node_id).copied(),
                    ))
                })
                // Per-peer blocking (Phase 3): a blocked peer is never a
                // placement candidate — sending it work is exactly what it
                // was blocked to stop. Refusals still happen server-side if
                // a stale ad slips through; this keeps the router from even
                // building a plan around it.
                .filter(|(id, ..)| !state.peer_block.blocked(id))
                .collect();
            let addrs: HashMap<String, SocketAddr> = ads
                .iter()
                .filter(|ad| !ad.rpc_address.is_empty())
                // Replication dials live here too: a blocked peer is never
                // a weight source either way.
                .filter(|ad| !state.peer_block.blocked(&ad.node_id))
                .filter_map(|ad| {
                    ad.rpc_address
                        .parse::<SocketAddr>()
                        .ok()
                        .map(|a| (ad.node_id.clone(), a))
                })
                .collect();
            (ads, rtts, peers, addrs)
        };
        (peers, addrs, ads, rtts)
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
        // ONE registry lock for the whole view (candidates + dial addrs +
        // the ad snapshot the cold-shard ring needs — the second
        // registry pass this used to pay is gone).
        let (mut peers, addrs, ads, rtts) = self.mesh_view_detailed(state, Some(expert_id)).await;
        // Cold-expert sharding (Phase 2): when nobody *advertises* this
        // expert, the deterministic shard ring says which node owns it. We
        // ask that owner rather than giving up — it either has the weights
        // (and answers) or honestly refuses, which is still better than
        // never trying. The ring is computed from the same ads every node
        // sees, so all nodes pick the same owner.
        let advertised = ads.iter().any(|ad| ad.explicitly_hosts(expert_id));
        if !advertised {
            let own_id = state.config.read().await.node_id.clone();
            let ring = crate::replication::ShardMap::from_ads(&own_id, &ads);
            if let Some(owner) = ring.primary_owner(expert_id)
                && owner != own_id
                && !peers.iter().any(|(id, ..)| *id == owner)
            {
                let rtt = rtts.get(&owner).copied().unwrap_or(u64::MAX >> 2);
                let resources: Option<crate::hw::ResourceSample> = ads
                    .iter()
                    .find(|ad| ad.node_id == owner)
                    .and_then(|ad| ad.resources.clone());
                // Verified reputation rides the extended candidate too
                // (the owner is usually already in the plan from mesh_view,
                // in which case its score was already attached).
                let known_rep = peers
                    .iter()
                    .find(|(id, ..)| *id == owner)
                    .and_then(|(.., rep)| *rep);
                let rep = match known_rep {
                    Some(r) => Some(r),
                    None => verified_scores(state).await.get(&owner).copied(),
                };
                tracing::debug!(
                    target: "mesh",
                    expert = %expert_id,
                    owner = %owner,
                    "no advertised holder — asking the cold-shard owner"
                );
                // Behind any advertised holder (there are none) and
                // ahead of nothing: the owner is the candidate.
                peers.push((owner, rtt, true, true, resources, rep));
            }
        }
        // Demand signal: this node wants this expert.
        state.record_activation(expert_id);
        // Speculative prefetch (Phase 3): the requester already knows which
        // layer it is on — warm the next layer's demand-ranked locals now.
        state.spawn_prefetch_after_serve(expert_id);
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
                state.freerider.lock().await.feed_dispatch(
                    expert_id,
                    &candidates,
                    served,
                    dispatch_ms,
                );
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
                rtt_ms: reg.routing_rtt_of(&ad.node_id).unwrap_or(u64::MAX >> 2),
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
    ///
    /// §12 privacy tiers (hot-reloaded, read from state.config here at the
    /// dispatch choke point): `local_only` never leaves the node for a
    /// hop; `trusted_peers` keeps only the `[privacy] trusted_nodes`
    /// allowlist; `full_mesh` is every alive connectable peer. A tier-
    /// forbidden hop fails with a message that names the policy — a
    /// downed runtime under a strict tier reads as "refused by policy",
    /// never as a generic outage.
    pub async fn chat_with_fallback(
        &self,
        state: &Arc<MeshState>,
        messages: &[serde_json::Value],
        local_try: impl FnOnce() -> anyhow::Result<String>,
    ) -> anyhow::Result<(String, Option<String>)> {
        req_id_tracker();
        let req_id = format!("chat-{}", crate::bootstrap::now_ms());
        let tier = state.config.read().await.privacy.clone();
        let (peers, addrs) = {
            let (peers, addrs) = self.mesh_view(state, None).await;
            match tier.hop_allowlist() {
                Some(allow) => {
                    // local_only produces an empty allowlist: no candidate
                    // survives, so nothing can be dispatched.
                    (
                        peers
                            .into_iter()
                            .filter(|(id, ..)| allow.contains(&id.as_str()))
                            .collect::<Vec<_>>(),
                        addrs
                            .into_iter()
                            .filter(|(id, _)| allow.contains(&id.as_str()))
                            .collect(),
                    )
                }
                None => (peers, addrs),
            }
        };
        self.refresh_pins(state).await;
        // trusted_peers with nobody trusted (or local_only outright): the
        // local runtime still runs, and a local failure must say WHY no
        // hop was attempted instead of printing a bare outage.
        if peers.is_empty() {
            let policy_note = match tier.level {
                PrivacyLevel::LocalOnly => "no mesh hop attempted — privacy tier local-only forbids peer dispatch (config [privacy] level)".to_string(),
                PrivacyLevel::TrustedPeers => "no mesh hop attempted — privacy tier trusted-peers has no trusted nodes configured ([privacy] trusted_nodes)".to_string(),
                PrivacyLevel::FullMesh => "no mesh hop attempted — no alive connectable peers".to_string(),
            };
            tracing::info!(
                target: "mesh",
                level = ?tier.level,
                %policy_note
            );
            return match local_try() {
                Ok(text) => Ok((text, None)),
                Err(local_err) => {
                    anyhow::bail!("local runtime failed ({local_err}) — {policy_note}")
                }
            };
        }
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

/// What a config-file re-read wants applied to a running node (Phase 0.5
/// extension): hot fields land within the same tick, cold fields are
/// restart-only and warned about, and identity fields are refused
/// outright — a hot identity swap would be a forgery hazard on every
/// signed wire surface.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ConfigReloadPlan {
    /// Changed fields that apply right away (hot-reloadable contract).
    pub hot: Vec<&'static str>,
    /// Changed fields that only a restart can apply.
    pub cold: Vec<&'static str>,
    /// The memory triple changed — the executor live-swaps the store
    /// first and adopts the memory config fields only on success.
    pub memory_changed: bool,
    /// Identity changed — always refused with a loud ERROR.
    pub identity_violations: Vec<&'static str>,
}

/// Diff the config file against the running config. Pure and
/// unit-tested; [`reload_config_from_disk`] is the only side-effectful
/// part. The hot set is exactly the fields every use site re-reads from
/// `state.config` (contribution/ringer/verify/freerider gates, the
/// per-peer block list, the memory triple) — so updating the live
/// config here propagates on the next read.
pub fn plan_config_reload(disk: &NodeConfig, live: &NodeConfig) -> ConfigReloadPlan {
    let mut plan = ConfigReloadPlan::default();

    // Identity: never hot-applied, never quietly adopted.
    if disk.node_id != live.node_id {
        plan.identity_violations.push("node_id");
    }
    if disk.secret_key_hex != live.secret_key_hex {
        plan.identity_violations.push("secret_key");
    }

    // Hot keys.
    if disk.contribution_percent != live.contribution_percent {
        plan.hot.push("contribution_percent");
    }
    if disk.ringer_percent != live.ringer_percent {
        plan.hot.push("ringer_percent");
    }
    if disk.verify_percent != live.verify_percent {
        plan.hot.push("verify_percent");
    }
    if disk.freerider_enabled != live.freerider_enabled {
        plan.hot.push("freerider_enabled");
    }
    if disk.peers.blocked != live.peers.blocked {
        plan.hot.push("peers.blocked");
    }
    // §12 privacy tiers are policy the operator should be able to tighten
    // without a restart window (and widen the same way): the dispatch
    // choke point re-reads state.config each attempt.
    if disk.privacy != live.privacy {
        plan.hot.push("privacy");
    }
    // Phase 3 speculative prefetch knobs: read at every serve/dispatch
    // attempt, the same hot-path shape as privacy above.
    if disk.experts.prefetch_enabled != live.experts.prefetch_enabled
        || disk.experts.prefetch_per_layer != live.experts.prefetch_per_layer
        || disk.experts.prefetch_horizon != live.experts.prefetch_horizon
    {
        plan.hot.push("experts.prefetch");
    }
    let mem_key = |c: &NodeConfig| {
        (
            c.memory_backend,
            c.memory_dsn.clone(),
            c.specialists.embeddings_enabled,
            c.specialists.embedding_dim,
            c.specialists.embedding_model.clone(),
            c.specialists.ollama_url.clone(),
        )
    };
    if mem_key(disk) != mem_key(live) {
        plan.memory_changed = true;
    }

    // Cold keys: sockets and process-lifetime machinery.
    if disk.model != live.model {
        plan.cold.push("model");
    }
    if disk.role != live.role {
        plan.cold.push("role");
    }
    if disk.gateway_port != live.gateway_port {
        plan.cold.push("gateway_port");
    }
    if disk.rpc_port != live.rpc_port {
        plan.cold.push("rpc_port");
    }
    if disk.gateway_bind != live.gateway_bind {
        plan.cold.push("gateway_bind");
    }
    if disk.advertise_host != live.advertise_host {
        plan.cold.push("advertise_host");
    }
    if disk.runtime_url != live.runtime_url {
        plan.cold.push("runtime_url");
    }
    if disk.runtime_api_key != live.runtime_api_key {
        plan.cold.push("runtime_api_key");
    }
    // The gateway binds the model-router tiers at boot — all cold.
    if disk.runtime_tool_model != live.runtime_tool_model {
        plan.cold.push("runtime_tool_model");
    }
    if disk.runtime_vision_model != live.runtime_vision_model {
        plan.cold.push("runtime_vision_model");
    }
    if disk.runtime_deep_model != live.runtime_deep_model {
        plan.cold.push("runtime_deep_model");
    }
    if disk.runtime_primary_model != live.runtime_primary_model {
        plan.cold.push("runtime_primary_model");
    }
    if disk.bootstrap_url != live.bootstrap_url {
        plan.cold.push("bootstrap_url");
    }
    if format!("{:?}", disk.crypto) != format!("{:?}", live.crypto) {
        plan.cold.push("crypto");
    }
    plan
}

/// Re-read `config.toml` and apply the reload plan: hot fields this
/// tick, restart-required warnings for cold ones, and a live swap for
/// the memory backend — failed swaps keep the previous store serving
/// and never adopt the config fields, so `/status` never lies about the
/// store that is actually live.
async fn reload_config_from_disk(state: &Arc<MeshState>) {
    let disk = match crate::config::NodeConfig::load() {
        Ok(Some(c)) => c,
        Ok(None) => return,
        Err(e) => {
            tracing::warn!(
                error = %e,
                "config re-read failed — keeping the running config for this tick"
            );
            return;
        }
    };
    let plan = {
        let live = state.config.read().await;
        plan_config_reload(&disk, &live)
    };
    if plan.identity_violations.is_empty()
        && plan.hot.is_empty()
        && plan.cold.is_empty()
        && !plan.memory_changed
    {
        return;
    }
    if !plan.identity_violations.is_empty() {
        tracing::error!(
            fields = ?plan.identity_violations,
            "identity change in config.toml refused — restart the node to re-identify (a hot identity swap would forge every signed wire surface)"
        );
    }
    // Memory store swap first: adopt its config fields only when the new
    // store actually opened.
    let mut memory_adopted = true;
    if plan.memory_changed {
        let backoff_expired = match state.memory_swap_backoff_until.lock().unwrap().as_ref() {
            Some(t) => std::time::Instant::now() >= *t,
            None => true,
        };
        if !backoff_expired {
            tracing::debug!("memory swap retry delayed (previous open failed — backoff)");
            memory_adopted = false;
        } else {
            match crate::memory::MemoryHandle::open_for_config(&disk).await {
                Ok(store) => {
                    let to_desc = store.describe();
                    let from_desc = state.memory.read().await.as_ref().map(|m| m.describe());
                    *state.memory.write().await = Some(std::sync::Arc::new(store));
                    *state.memory_swap_backoff_until.lock().unwrap() = None;
                    tracing::info!(
                        from = ?from_desc,
                        to = %to_desc,
                        "memory store live-swapped (config hot-reload)"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "memory store swap failed — the previous store keeps serving; retrying with backoff"
                    );
                    *state.memory_swap_backoff_until.lock().unwrap() =
                        Some(std::time::Instant::now() + std::time::Duration::from_secs(60));
                    memory_adopted = false;
                }
            }
        }
    }
    // Adopt the hot fields (memory fields only when the swap succeeded).
    {
        let mut live = state.config.write().await;
        live.contribution_percent = disk.contribution_percent;
        live.ringer_percent = disk.ringer_percent;
        live.verify_percent = disk.verify_percent;
        live.freerider_enabled = disk.freerider_enabled;
        live.peers.blocked = disk.peers.blocked.clone();
        live.privacy = disk.privacy.clone();
        if memory_adopted {
            live.memory_backend = disk.memory_backend;
            live.memory_dsn = disk.memory_dsn.clone();
            live.specialists.embeddings_enabled = disk.specialists.embeddings_enabled;
            live.specialists.embedding_dim = disk.specialists.embedding_dim;
            live.specialists.embedding_model = disk.specialists.embedding_model.clone();
            live.specialists.ollama_url = disk.specialists.ollama_url.clone();
        }
    }
    for field in plan.hot {
        tracing::info!(field, "config hot-applied (took effect this tick)");
    }
    for field in &plan.cold {
        tracing::warn!(field, "config field changed — restart required to apply");
    }
    // §2 model-switch machinery: a model change is cold by deliberate
    // design (model switching is MESH RE-IDENTITY — routers, tier
    // bindings and the registry key off the boot-time model), but the
    // graceful HALF is live right now: signed /v1/leave retracts the
    // old-model ad instantly, the advertise latch closes, and /status
    // carries the pending restart.
    if plan.cold.contains(&"model") {
        let pending = disk
            .model
            .as_ref()
            .map(|m| format!("{} ({})", m.id, m.quant));
        graceful_model_exit(state, pending).await;
    }
}

/// The graceful half of the model-switch: retract our own advertisement
/// NOW (a stop then restart otherwise leaves the old ad up for the whole
/// 90 s TTL — a peer in the old mesh that no longer serves its model).
/// Idempotent per process: the first model-change tick fires the leave
/// and closes the advertise latch; later ticks see both latched.
async fn graceful_model_exit(state: &Arc<MeshState>, pending: Option<String>) {
    let first_change = !state
        .left_mesh_latch
        .swap(true, std::sync::atomic::Ordering::SeqCst);
    *state.pending_model.write().await = pending;
    if !first_change {
        return;
    }
    let order = { state.discovery.lock().await.order(None) };
    let node_id = { state.config.read().await.node_id.clone() };
    let node_id_ref = &node_id;
    let identity_ref = &state.identity;
    let legs = crate::discovery::try_each(&order, |url| {
        let client = BootstrapClient::new(url.to_string());
        async move { client.leave(node_id_ref, identity_ref).await }
    })
    .await;
    match legs {
        Ok((advertised_before, used)) => tracing::warn!(
            target: "mesh",
            bootstrap = %used,
            advertised_before,
            "model changed in config — LEFT the old mesh (signed /v1/leave, not re-advertising); restart `jerith run` to rejoin under the new model"
        ),
        Err(e) => tracing::warn!(
            target: "mesh",
            error = %e,
            "model changed — graceful leave could not reach any bootstrap; the old ad expires by TTL instead"
        ),
    }
}

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

        // Hot-reload the config FILE (Phase 0.5 extension): `jerith config
        // set` (or a raw edit) is picked up here — hot fields land this
        // tick, cold fields WARN about the restart, and a memory-backend
        // change live-swaps the store (migration guidance rides `config
        // set` and `jerith memory migrate` moves the rows safely).
        reload_config_from_disk(&state).await;

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
                async move { client.fetch_peers_signed_full(model_id, identity_ref).await }
            })
            .await;
            match fetched_leg {
                Ok((fetched, used)) => {
                    let FetchedPeers {
                        peers,
                        served_by: _,
                        bootstrap_key,
                        reputations,
                    } = fetched;
                    // §10 reputation trust surface: PIN the answering
                    // bootstrap's key and ingest its direct stamps.
                    pin_and_ingest(&state, &used, bootstrap_key.as_ref(), reputations).await;
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
                        // §10 gossip ingest: stamps carried INSIDE ads re-
                        // verify against OUR OWN pins (origin-bound); unknown
                        // origins, stale stamps, and forged sigs drop here,
                        // never into the cache. Self-stamps from the ad's
                        // own node are refused (no self-serving broadcast).
                        let _ = ingest_ad_gossip(&state, &ad).await;
                        reg.upsert(ad);
                    }
                    tracing::info!(
                        target: "mesh",
                        fetched,
                        bootstrap = %used,
                        "peer list refreshed from bootstrap"
                    );
                    // Pre-join mesh compat (FEATURES §12), once per
                    // process: if the registry admitted NOTHING so far,
                    // compare OUR identity against the bootstrap census
                    // so "mesh looks empty" always has a typed answer —
                    // especially "same model, wrong quant, N nodes
                    // could still take you". A healthy node overwrites
                    // nothing interesting; verdicts surface in /status.
                    // NOTE: the `reg` guard is held in this scope — use it
                    // instead of re-locking (a second registry lock on the
                    // same task deadlocks the loop AND /status's mutex).
                    if reg.is_empty() && state.compat.read().await.is_none() {
                        match BootstrapClient::new(used.clone())
                            .fetch_model_census()
                            .await
                        {
                            Ok(census) => {
                                let verdict = compat_verdict(m, &census);
                                if verdict.verdict != "healthy" {
                                    tracing::warn!(
                                        target: "mesh",
                                        verdict = %verdict.verdict,
                                        "{}",
                                        verdict.message
                                    );
                                } else {
                                    tracing::info!(
                                        target: "mesh",
                                        verdict = %verdict.verdict,
                                        "{}",
                                        verdict.message
                                    );
                                }
                                *state.compat.write().await = Some(verdict);
                            }
                            Err(e) => {
                                tracing::debug!(
                                    target: "mesh",
                                    error = %e,
                                    "mesh compat census probe failed (retry next tick)"
                                );
                            }
                        }
                    }
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
                // §10 reputation gossip: bootstrap-verified score stamps
                // (verified against OUR pins) ride inside the ad so peers
                // can learn configured-bootstrap scores without trusting
                // the advertiser. Newest first, capped like discovery.
                gossip_reputations: {
                    let cache = state.reputation_cache.lock().await;
                    (!cache.is_empty()).then(|| {
                        cache
                            .iter()
                            .take(crate::bootstrap::REPUTATION_GOSSIP_AD_MAX)
                            .cloned()
                            .collect()
                    })
                },
            };
            let identity_ref = &state.identity;
            let ad_ref = &ad;
            let advertise_order = state.discovery.lock().await.order(None);
            // §2 model-switch latch: a graceful exit means STOP re-adver-
            // tising the old identity — the signed /v1/leave already retrac-
            // ted it, and dialing again would un-exit. Restart joins the
            // new model's mesh fresh.
            let advertise_leg = if state
                .left_mesh_latch
                .load(std::sync::atomic::Ordering::SeqCst)
            {
                tracing::debug!(
                    target: "mesh",
                    "mesh exit latched — advertise skipped (restart re-joins under the new model)"
                );
                None
            } else {
                Some(
                    crate::discovery::try_each(&advertise_order, |url| {
                        let client = BootstrapClient::new(url.to_string());
                        async move { client.advertise(ad_ref, identity_ref).await }
                    })
                    .await,
                )
            };
            match advertise_leg {
                None => {}
                Some(Ok(((), used))) => {
                    state.discovery.lock().await.note_good(&used);
                    tracing::debug!(target: "mesh", bootstrap = %used, "advertised to bootstrap");
                }
                Some(Err(e)) => {
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

        // Uptime metering + real-time contribution measurement on every
        // tick (FEATURES §2): the meters are cumulative, so the tick diffs
        // them into an interval share (idle = absence, not 0 %), and the
        // gate now reads the MEASURED share against the DECLARED minimum
        // — claim-aside, the old rule was a token instead of a truth.
        {
            let (total_served, total_consumed, role, declared) = {
                let mut meters = state.meters.lock().await;
                meters.uptime_seconds += ping_interval.as_secs();
                let cfg = state.config.read().await;
                (
                    meters.tokens_served,
                    meters.tokens_consumed,
                    cfg.role,
                    cfg.contribution_percent,
                )
            };
            let mut contrib = state.contrib.lock().await;
            contrib.observe_totals(total_served, total_consumed, ping_interval.as_secs());
            let healthy = contrib.healthy_for(role, declared);
            drop(contrib);
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
/// Our own echoed ad is skipped: the bootstrap lists us back into the
/// peer set, and self-dialing just wastes the tick — worse, a busy
/// gateway that fails its own health check would rack up self-failures
/// toward its own deprecation line.
async fn ping_all(state: &Arc<MeshState>, bootstrap: &BootstrapClient) {
    let my_id = state.identity.node_id();
    let targets: Vec<(String, String)> = {
        let reg = state.registry.lock().await;
        reg.connectable()
            .into_iter()
            .filter(|(ad, _)| ad.node_id != my_id)
            .map(|(ad, _)| (ad.node_id.clone(), format!("http://{}/health", ad.address)))
            .collect()
    };
    if targets.is_empty() {
        return;
    }

    let client = crate::http::mesh_http();

    // Concurrent ping batch: the tick used to pay N x RTT serially
    // (fleet growth would have linearly stretched every tick before
    // any other work ran). Each leg is self-contained — own stamp,
    // own signed-send, own verify + evidence accounting — and the
    // post-batch deprecation machinery below only wants the
    // failures, so the join re-collects exactly those.
    let failed: Vec<String> = {
        let legs: Vec<_> = targets
            .into_iter()
            .map(|(node_id, url)| {
                let my_id = my_id.clone();
                let client = client.clone();
                async move { ping_one(state, bootstrap, &client, &my_id, node_id, url).await }
            })
            .collect();
        futures_util::future::join_all(legs)
            .await
            .into_iter()
            .flatten()
            .collect()
    };

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

/// One ping leg of the concurrent batch (see ping_all): sign, send,
/// verify the signed health reply, account evidence. Returns the
/// node_id for the post-batch failure machinery when the leg did
/// NOT count the peer alive; None when it did.
async fn ping_one(
    state: &Arc<MeshState>,
    bootstrap: &BootstrapClient,
    client: &reqwest::Client,
    my_id: &str,
    node_id: String,
    url: String,
) -> Option<String> {
    let started = std::time::Instant::now();
    let stamp = now_ms();
    let payload = crate::signing::ping_payload(my_id, &node_id, stamp);
    // Freeze the stamp: the payload embeds it, so the header stamp and
    // the signed bytes must come from the SAME clock read (ms-boundary
    // races under load made pings intermittently rejected).
    let auth = crate::signing::WireAuth::with_stamp(&state.identity, stamp, payload.as_bytes());
    let result = client
        .get(&url)
        .header("x-jerith-pubkey", &auth.pubkey)
        .header("x-jerith-stamp", auth.stamp_ms.to_string())
        .header("x-jerith-sig", &auth.sig)
        .header("x-jerith-from", my_id)
        .timeout(Duration::from_secs(4))
        .send()
        .await;
    match result {
        Ok(resp) if resp.status().is_success() => {
            // Verify the signed health reply attributes it to the peer.
            let rtt = started.elapsed().as_millis() as u64;
            let (ok, meas) = verify_health_reply(state, &node_id, resp).await;
            if ok {
                let mut reg = state.registry.lock().await;
                reg.record_ping_ok(&node_id, rtt);
                drop(reg);
                // FEATURES §2: tally the peer's signed measurement
                // verdicts — tampered blocks and demonstrated
                // declared-vs-measured shortfalls queue Leeching
                // evidence through the same flush cadence as every
                // other fraud layer (verified here locally, signed by
                // the peer's own key, so the bootstrap can weigh it).
                let (verified, targets): (Option<VerifiedMeasurement>, Vec<String>) = match meas {
                    Some(Ok(block)) => {
                        let flagged = state.contrib_ledger.lock().await.record_verified(
                            &node_id,
                            block.declared_percent,
                            block.measured_percent,
                        );
                        (Some(block), flagged)
                    }
                    Some(Err(())) => {
                        let flagged = state.contrib_ledger.lock().await.record_tampered(&node_id);
                        (None, flagged)
                    }
                    None => (None, Vec::new()),
                };
                if !targets.is_empty() {
                    let refs: Vec<&str> = targets.iter().map(String::as_str).collect();
                    state
                        .verify
                        .flag_fraud(crate::bootstrap_server::FraudKind::Leeching, &refs)
                        .await;
                }
                // §2 QUORUM: the verified block itself is FORWARDED to
                // the bootstrap under both signatures — the subject's
                // own signature anchors the figures (a biased pinger
                // cannot change them), ours attests it verified exactly
                // this block. Off the inference hot path (one POST per
                // verified ping); failures log at debug and count for
                // nothing else.
                if let Some(block) = &verified {
                    let obs = crate::bootstrap_server::MeasureObservation {
                        reporter: my_id.to_string(),
                        subject: block.subject.clone(),
                        subject_pubkey: block.subject_pubkey.clone(),
                        subject_sig: block.subject_sig.clone(),
                        block_stamp_ms: block.block_stamp_ms,
                        window_secs: block.window_secs,
                        tokens_served: block.tokens_served,
                        tokens_consumed: block.tokens_consumed,
                        declared_percent: block.declared_percent,
                        measured_percent: block.measured_percent,
                    };
                    match bootstrap.report_measure(&obs, &state.identity).await {
                        Ok(()) => tracing::debug!(
                            target: "mesh",
                            subject = %obs.subject,
                            "measurement block forwarded for the quorum tally"
                        ),
                        Err(e) => tracing::debug!(
                            target: "mesh",
                            error = %e,
                            "measurement quorum upload skipped (bootstrap unavailable?)"
                        ),
                    }
                }
            } else {
                tracing::warn!(peer = %node_id, "health reply signature invalid — not counted alive");
                return Some(node_id);
            }
        }
        _ => {
            return Some(node_id);
        }
    }

    None
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
///
/// FEATURES §2: when the reply carries a signed `measurement` block its
/// signature is re-verified the same way; invalid → tampered (evidence
/// path), absent → plain liveness only. Returns whether the peer counts
/// as alive, plus the verified (declared, measured) sample or a tamper
/// marker. The verification-phase check of the payload signature uses the
/// very identity whose key signed the whole reply.
/// A verified measurement block, field-for-field: everything the §2
/// quorum uploader forwards (the subject's own signature anchors it).
#[derive(Debug, Clone)]
pub struct VerifiedMeasurement {
    pub block_stamp_ms: u64,
    pub window_secs: u64,
    pub tokens_served: u64,
    pub tokens_consumed: u64,
    pub declared_percent: u64,
    pub measured_percent: u64,
    /// The subject's wire pubkey + the block signature, carried so the
    /// bootstrap can re-verify both signatures at tally time.
    pub subject: String,
    pub subject_pubkey: String,
    pub subject_sig: String,
}

async fn verify_health_reply(
    _state: &Arc<MeshState>,
    expected_node_id: &str,
    resp: reqwest::Response,
) -> (bool, Option<Result<VerifiedMeasurement, ()>>) {
    #[derive(serde::Deserialize)]
    struct MeasurementBlock {
        window_secs: u64,
        tokens_served: u64,
        tokens_consumed: u64,
        declared_percent: u64,
        measured_percent: u64,
        sig: String,
    }
    #[derive(serde::Deserialize)]
    struct HealthReply {
        node_id: String,
        pubkey: String,
        stamp_ms: u64,
        sig: String,
        uptime_seconds: u64,
        contribution_percent: u64,
        #[serde(default)]
        measurement: Option<MeasurementBlock>,
    }
    let Ok(reply) = resp.json::<HealthReply>().await else {
        return (false, None);
    };
    if reply.node_id != expected_node_id {
        return (false, None);
    }
    // The pubkey fingerprint must match the claimed identity.
    if reply.pubkey.len() != 64 || reply.pubkey[..16] != reply.node_id {
        return (false, None);
    }
    let payload = crate::signing::health_payload(
        &reply.node_id,
        reply.stamp_ms,
        reply.uptime_seconds,
        reply.contribution_percent,
    );
    let alive = crate::signing::verify(&reply.pubkey, payload.as_bytes(), &reply.sig)
        && now_ms().saturating_sub(reply.stamp_ms) <= crate::signing::STAMP_MAX_AGE_MS;
    let measurement = reply.measurement.map(|m| {
        if m.declared_percent != reply.contribution_percent {
            // A block declaring a different floor than the reply itself is
            // internally inconsistent — treat as tampered.
            return Err(());
        }
        let payload = crate::signing::measurement_payload(
            &reply.node_id,
            reply.stamp_ms,
            m.window_secs,
            m.tokens_served,
            m.tokens_consumed,
            m.declared_percent,
            m.measured_percent,
        );
        if crate::signing::verify(&reply.pubkey, payload.as_bytes(), &m.sig) {
            Ok(VerifiedMeasurement {
                block_stamp_ms: reply.stamp_ms,
                window_secs: m.window_secs,
                tokens_served: m.tokens_served,
                tokens_consumed: m.tokens_consumed,
                declared_percent: m.declared_percent,
                measured_percent: m.measured_percent,
                subject: reply.node_id.clone(),
                subject_pubkey: reply.pubkey.clone(),
                subject_sig: m.sig,
            })
        } else {
            Err(())
        }
    });
    (alive, measurement)
}

/// Verified reputation per subject (Phase 5 placement weighting): newest
/// stamp for each subject from OUR pin wall. Two lookups per entry — the
/// cache is already bounded, never a scan.
async fn verified_scores(state: &Arc<MeshState>) -> std::collections::HashMap<String, i32> {
    let cache = state.reputation_cache.lock().await.clone();
    cache.iter().map(|s| (s.subject.clone(), s.score)).collect()
}

/// §10 quotient — how old a reputation stamp may be and still ride
/// (direct fetches re-sign every response; gossip hops add minutes).
/// Old verdicts from a dead origin stop propagating here, not by policy
/// layers elsewhere.
pub const REPUTATION_STAMP_MAX_AGE_MS: u64 = 3_600_000;

/// Cap on the gossip cache carried inside ads (bounded like discovery).
pub const REPUTATION_CACHE_MAX: usize = crate::bootstrap::REPUTATION_GOSSIP_AD_MAX;

/// Verify one bootstrap-published stamp against THE origin's own key and
/// freshness. Returns the full wire stamp on success (origin fingerprint
/// comes from the key itself, never from the caller's word).
fn stamp_verified(
    origin: &crate::bootstrap::BootstrapKeyInfo,
    subject: &str,
    body: &crate::bootstrap::StampBody,
) -> Option<crate::signing::ReputationStamp> {
    // Score sanity: the floor lives server-side (REPUTATION_FLOOR); a
    // grossly out-of-range number is a broken/malicious origin — refuse.
    if !(-10_000..=100).contains(&body.score) {
        return None;
    }
    let stamp = crate::signing::ReputationStamp {
        origin: origin.node_id.clone(),
        subject: subject.to_string(),
        score: body.score,
        stamp_ms: body.stamp_ms,
        sig: body.sig.clone(),
    };
    if now_ms().saturating_sub(stamp.stamp_ms) > REPUTATION_STAMP_MAX_AGE_MS {
        return None;
    }
    if !stamp.verifies(&origin.pubkey) {
        return None;
    }
    Some(stamp)
}

/// Ingest verified stamps into the gossip cache: newest-per-subject wins,
/// whole set capped (`REPUTATION_CACHE_MAX`). Every stamp in here passed a
/// pin-or-direct verification upstream — the ad publisher sends nothing
/// unverifiable.
/// §10 pin + direct-ingest for ONE answered bootstrap: pin the key first
/// (trust-on-first-use on the configured/learned URL; a rotation replaces
/// the pin with a journal note), then ingest the response's direct stamps
/// (they verify against the key that published itself). Returns the
/// ingested count. Unit-testable against stub bootstraps.
async fn pin_and_ingest(
    state: &Arc<MeshState>,
    served_by: &str,
    bootstrap_key: Option<&crate::bootstrap::BootstrapKeyInfo>,
    reputations: std::collections::HashMap<String, crate::bootstrap::StampBody>,
) -> usize {
    let Some(key) = bootstrap_key else {
        return 0;
    };
    let rotated = {
        let mut d = state.discovery.lock().await;
        d.pin_key(served_by, &key.pubkey)
    };
    if rotated {
        // First sight or an actual rotation — either way an operator
        // should see it in the journal once per change.
        tracing::info!(
            target: "mesh",
            bootstrap = %served_by,
            pubkey = %key.pubkey,
            "bootstrap signing key pinned (reputation stamps verifiable from here)"
        );
    }
    let direct: Vec<crate::signing::ReputationStamp> = reputations
        .into_iter()
        .filter_map(|(subject, body)| stamp_verified(key, &subject, &body))
        .collect();
    let n = direct.len();
    ingest_verified_stamps(state, direct).await;
    n
}

/// §10 gossip ingest for ONE peer advertisement: every carried stamp must
/// (a) not be the ad owner's own subject (no self-serving broadcast),
/// (b) name an origin WE pinned, (c) verify against that pinned key, and
/// (d) be fresh. Returns the ingested count (journal-debug on refusal).
async fn ingest_ad_gossip(state: &Arc<MeshState>, ad: &PeerAd) -> usize {
    let Some(carried) = &ad.gossip_reputations else {
        return 0;
    };
    if carried.is_empty() {
        return 0;
    }
    let mut verified = Vec::new();
    {
        let d = state.discovery.lock().await;
        for stamp in carried {
            let pinned_ok = stamp.subject != ad.node_id
                && !stamp.subject.is_empty()
                && d.pubkey_for_fingerprint(&stamp.origin)
                    .is_some_and(|k| stamp.verifies(&k))
                && now_ms().saturating_sub(stamp.stamp_ms) <= REPUTATION_STAMP_MAX_AGE_MS;
            if pinned_ok {
                verified.push(stamp.clone());
            }
        }
    }
    if verified.len() < carried.len() {
        tracing::debug!(
            target: "mesh",
            from = %ad.node_id,
            dropped = carried.len() - verified.len(),
            "unverifiable reputation gossip dropped at the pin wall"
        );
    }
    let n = verified.len();
    ingest_verified_stamps(state, verified).await;
    n
}

async fn ingest_verified_stamps(
    state: &Arc<MeshState>,
    stamps: Vec<crate::signing::ReputationStamp>,
) {
    if stamps.is_empty() {
        return;
    }
    let mut cache = state.reputation_cache.lock().await;
    for stamp in stamps {
        cache.retain(|s| s.subject != stamp.subject || s.stamp_ms < stamp.stamp_ms);
        if !cache
            .iter()
            .any(|s| s.subject == stamp.subject && s.stamp_ms >= stamp.stamp_ms)
        {
            cache.push(stamp);
        }
    }
    cache.sort_by_key(|s| std::cmp::Reverse(s.stamp_ms));
    while cache.len() > REPUTATION_CACHE_MAX {
        cache.pop();
    }
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
    use crate::bootstrap::{DEPRECATE_AFTER_FAILED_PINGS, ModelCensusRow};
    use crate::config::*;

    fn census(rows: &[(&str, &str, u64)]) -> Vec<ModelCensusRow> {
        rows.iter()
            .map(|(m, q, c)| ModelCensusRow {
                model_id: m.to_string(),
                count: *c,
                quants: std::collections::BTreeMap::from([(q.to_string(), *c)]),
            })
            .fold(Vec::new(), |mut acc, row| {
                if let Some(existing) = acc.iter_mut().find(|r| r.model_id == row.model_id) {
                    existing.count += row.count;
                    for (k, v) in row.quants {
                        *existing.quants.entry(k).or_default() += v;
                    }
                } else {
                    acc.push(row);
                }
                acc
            })
    }

    fn local(id: &str, quant: &str) -> crate::models::ModelId {
        crate::models::ModelId {
            id: id.into(),
            quant: quant.into(),
            source: crate::models::ModelSource::Ollama { name: id.into() },
        }
    }

    #[test]
    fn compat_verdict_healthy_when_our_exact_quant_is_live() {
        let v = compat_verdict(
            &local("glm-5.3", "int4"),
            &census(&[
                ("glm-5.3", "int4", 3),
                ("glm-5.3", "fp8", 1),
                ("qwen3:8b", "default", 2),
            ]),
        );
        assert_eq!(v.verdict, "healthy", "{}", v.message);
        assert_eq!(v.peers_on_model, 4);
        assert_eq!(v.peers_same_quant, 3);
    }

    #[test]
    fn compat_verdict_wrong_quant_names_the_live_quants() {
        let v = compat_verdict(
            &local("glm-5.3", "default"),
            &census(&[("glm-5.3", "int4", 2), ("glm-5.3", "fp8", 1)]),
        );
        assert_eq!(v.verdict, "wrong_quant");
        assert!(
            v.message.contains("same model, wrong quant"),
            "{}",
            v.message
        );
        assert!(v.message.contains("int4 (2 nodes)"), "{}", v.message);
        assert!(v.message.contains("fp8 (1 node"), "{}", v.message);
        assert!(v.message.contains("'default'"), "{}", v.message);
        assert!(v.other_quants.iter().any(|(q, c)| q == "int4" && *c == 2));
    }

    #[test]
    fn compat_verdict_no_peers_names_live_alternatives() {
        let v = compat_verdict(
            &local("made-up", "default"),
            &census(&[("glm-5.3", "int4", 2)]),
        );
        assert_eq!(v.verdict, "no_peers_on_model");
        assert!(
            v.message.contains("models live on this mesh"),
            "{}",
            v.message
        );
        assert!(v.message.contains("glm-5.3 (2)"), "{}", v.message);
    }

    #[test]
    fn compat_verdict_brand_new_mesh_says_so() {
        let v = compat_verdict(&local("glm-5.3", "int4"), &census(&[]));
        assert_eq!(v.verdict, "no_peers_on_model");
        assert!(v.message.contains("you ARE the mesh"), "{}", v.message);
    }

    /// §10 reputation gossip never crosses the pin wall: one ad whose
    /// carried stamps exercise every rejection leg, then the acceptance
    /// leg. (a) a NOT-pinned origin's properly-signed stamp drops (the
    /// fingerprint is unknown at the receiver's pin table); (b) a
    /// self-stamp (subject = the ad owner praising itself) drops even
    /// fully signed and fresh; (c) a forged signature under a PINNED
    /// origin's fingerprint drops; (d) only the honestly-signed stamp
    /// about someone else under the pinned origin key lands in the
    /// gossip cache — the advertiser's word is never the reason.
    #[tokio::test]
    async fn reputation_gossip_drops_unpinned_forged_and_self_stamps() {
        let origin = crate::signing::SigningIdentity::generate();
        let forger = crate::signing::SigningIdentity::generate();
        let mut cfg = NodeConfig::default();
        cfg.ensure_identity().expect("test identity");
        let (state, _rx) = MeshState::new(cfg).expect("test mesh state");
        let subject = "subject-node";
        let now = now_ms();

        // A stamp's wires are: `origin` = the 16-hex fingerprint (what the
        // pin table indexes), `sig` = over a payload built from the FULL
        // origin pubkey (what verifies() rebuilds). Both conventions matter
        // — a fingerprint-signed payload MUST fail.
        let stamp_of = |signer: &crate::signing::SigningIdentity,
                        origin_pubkey: &str,
                        subj: &str,
                        score: i32,
                        stamp_ms: u64|
         -> crate::signing::ReputationStamp {
            let payload = crate::signing::reputation_payload(origin_pubkey, subj, score, stamp_ms);
            crate::signing::ReputationStamp {
                origin: origin_pubkey[..16].to_string(),
                subject: subj.to_string(),
                score,
                stamp_ms,
                sig: signer.sign(payload.as_bytes()),
            }
        };

        // Leg (a): origin not pinned yet — even a perfectly signed stamp
        // from the REAL origin must drop (fingerprint unknowable).
        let ad_owner = "advertiser-node";
        let mut ad = crate::bootstrap::PeerAd {
            node_id: ad_owner.into(),
            address: "127.0.0.1:9001".into(),
            model_id: "glm-5.3".into(),
            quant: "int4".into(),
            contribution_percent: 30,
            rpc_address: String::new(),
            last_seen_ms: now,
            role: crate::config::NodeRole::Hybrid,
            resources: None,
            hosted_experts: None,
            hosted_expert_count: None,
            discovery_servers: Vec::new(),
            gossip_reputations: Some(vec![stamp_of(
                &origin,
                &origin.pubkey_hex(),
                subject,
                42,
                now,
            )]),
        };
        assert_eq!(ingest_ad_gossip(&state, &ad).await, 0, "unpinned origin");
        assert!(state.reputation_cache.lock().await.is_empty());

        // Pin the origin (the operator's configured URL's bootstrap key).
        {
            let mut d = state.discovery.lock().await;
            d.pin_key("http://anchor.invalid", &origin.pubkey_hex());
        }

        // Leg (b): self-stamp — the ad owner certifying itself. Fully
        // signed and fresh, but it must never ride.
        let self_stamp = stamp_of(&origin, &origin.pubkey_hex(), ad_owner, 99, now);
        // Leg (c): forged — signed by the forger's KEY but wearing the
        // pinned origin's fingerprint (what a lazy attacker sends).
        let forged = stamp_of(&forger, &origin.pubkey_hex(), subject, 100, now);
        ad.gossip_reputations = Some(vec![self_stamp, forged.clone()]);
        assert_eq!(ingest_ad_gossip(&state, &ad).await, 0, "self + forged");
        assert!(state.reputation_cache.lock().await.is_empty());

        // Leg (d): the honest one — a real origin signature about a third
        // party. Both legs had better land now.
        let honest = stamp_of(&origin, &origin.pubkey_hex(), subject, 42, now);
        ad.gossip_reputations = Some(vec![honest.clone()]);
        assert_eq!(ingest_ad_gossip(&state, &ad).await, 1, "honest stamp lands");
        let cache = state.reputation_cache.lock().await;
        assert_eq!(cache.len(), 1);
        assert_eq!(cache[0], honest);
        assert_eq!(cache[0].score, 42);

        // The direct-fetch wall agrees: stamp_verified accepts the real
        // origin's key and rejects the forged body under the same subject.
        let key_info = crate::bootstrap::BootstrapKeyInfo {
            node_id: origin.node_id(),
            pubkey: origin.pubkey_hex(),
        };
        let body = crate::bootstrap::StampBody {
            score: 42,
            stamp_ms: now,
            sig: honest.sig.clone(),
        };
        assert!(stamp_verified(&key_info, subject, &body).is_some());
        // Fingerprint-only payload signing (a plausible mistake) must fail:
        let fp_signed = crate::signing::ReputationStamp {
            origin: origin.node_id(),
            subject: subject.to_string(),
            score: 42,
            stamp_ms: now,
            sig: origin.sign(
                crate::signing::reputation_payload(&origin.node_id(), subject, 42, now).as_bytes(),
            ),
        };
        assert!(!fp_signed.verifies(&origin.pubkey_hex()));
    }

    /// Phase 5 placement wiring: mesh_view attaches the VERIFIED score
    /// (from our own pin-wall cache) to the right peer, and nobody else's.
    #[tokio::test]
    async fn mesh_view_attaches_verified_reputation_per_peer() {
        use crate::config::NodeRole;
        let mut cfg = NodeConfig::default();
        cfg.ensure_identity().expect("test identity");
        let (state, _rx) = MeshState::new(cfg).expect("test mesh state");
        {
            let mut reg = PeerRegistry::new(Some(crate::models::ModelId {
                id: "glm-5.3".into(),
                quant: "int4".into(),
                source: crate::models::ModelSource::Curated,
            }));
            for id in ["scored-peer", "unscored-peer"] {
                reg.upsert(crate::bootstrap::PeerAd {
                    node_id: id.into(),
                    address: "127.0.0.1:9090".into(),
                    model_id: "glm-5.3".into(),
                    quant: "int4".into(),
                    contribution_percent: 30,
                    rpc_address: String::new(),
                    last_seen_ms: now_ms(),
                    role: NodeRole::Hybrid,
                    resources: None,
                    hosted_experts: None,
                    hosted_expert_count: None,
                    discovery_servers: Vec::new(),
                    gossip_reputations: None,
                });
                reg.record_ping_ok(id, 5);
            }
            *state.registry.lock().await = reg;
        }
        let stamp = crate::signing::ReputationStamp {
            origin: "47bb0d50e3de1009".to_string(),
            subject: "scored-peer".to_string(),
            score: -25,
            stamp_ms: now_ms(),
            sig: String::new(),
        };
        // The pin wall normally fills the cache; inject directly for the
        // wiring test (the wall itself is tested above).
        state.reputation_cache.lock().await.push(stamp);

        let router = MeshFallbackChat::spawn().expect("test router shell");
        let (peers, _addrs) = router.mesh_view(&state, None).await;
        let scored = peers
            .iter()
            .find(|(id, ..)| id == "scored-peer")
            .expect("peer present");
        assert_eq!(
            scored.5,
            Some(-25),
            "verified score attached to the right peer"
        );
        let unscored = peers
            .iter()
            .find(|(id, ..)| id == "unscored-peer")
            .expect("peer present");
        assert_eq!(unscored.5, None, "no stamp ⇒ no placement word at all");
    }

    /// §12 privacy tiers are policy: changing them is a HOT reload
    /// (binds within one tick), never a cold restart-only field.
    #[test]
    fn config_reload_plan_privacy_tiers_are_hot() {
        let live = NodeConfig::default();
        let mut disk = live.clone();
        disk.privacy.level = PrivacyLevel::LocalOnly;
        disk.privacy.trusted_nodes = vec!["aa".to_string()];
        let plan = plan_config_reload(&disk, &live);
        assert!(
            plan.hot.contains(&"privacy"),
            "tiers must bind within one tick"
        );
        assert!(plan.cold.is_empty());
        assert!(!plan.memory_changed);
        assert!(plan.identity_violations.is_empty());
        // An identical config stays silent.
        assert!(plan_config_reload(&disk, &disk) == ConfigReloadPlan::default());
    }

    /// The dispatch choke point: at local_only, a local runtime failure
    /// fails with the POLICY note (no peer hop was even attempted), not a
    /// generic outage message.
    #[tokio::test]
    async fn chat_fallback_never_hops_at_local_only() {
        let mut cfg = NodeConfig::default();
        cfg.ensure_identity().expect("test identity");
        cfg.privacy.level = PrivacyLevel::LocalOnly;
        let (state, _rx) = MeshState::new(cfg).expect("test mesh state");
        let mf = MeshFallbackChat::spawn().expect("test fallback chat");
        let err = mf
            .chat_with_fallback(
                &state,
                &[serde_json::json!({"role": "user", "content": "hi"})],
                || Err(anyhow::anyhow!("local runtime down")),
            )
            .await
            .expect_err("a failed local runtime must surface with the policy note");
        let msg = err.to_string();
        assert!(
            msg.contains("local-only forbids peer dispatch"),
            "typed refusal required: {msg}"
        );
        assert!(msg.contains("local runtime down"), "cause kept: {msg}");
    }

    /// trusted_peers with an EMPTY allowlist (fresh install): the honest
    /// answer names the un-actuated policy, and the local runtime still
    /// serves — hopping to unknown peers is exactly what the tier forbids.
    #[tokio::test]
    async fn chat_fallback_trusted_peers_without_allowlist_refuses_honestly() {
        let mut cfg = NodeConfig::default();
        cfg.ensure_identity().expect("test identity");
        cfg.privacy.level = PrivacyLevel::TrustedPeers;
        let (state, _rx) = MeshState::new(cfg).expect("test mesh state");
        let mf = MeshFallbackChat::spawn().expect("test fallback chat");
        let err = mf
            .chat_with_fallback(
                &state,
                &[serde_json::json!({"role": "user", "content": "hi"})],
                || Err(anyhow::anyhow!("local runtime down")),
            )
            .await
            .expect_err("a failed local runtime must surface with the policy note");
        let msg = err.to_string();
        assert!(msg.contains("no trusted nodes configured"), "{msg}");
        assert!(msg.contains("local runtime down"), "{msg}");
    }

    /// §2 model-switch machinery: the graceful half fires exactly once —
    /// a signed /v1/leave at the (stub) bootstrap, the advertise latch
    /// closes, /status's pending_model records the restart target; later
    /// ticks never re-fire it.
    #[tokio::test]
    async fn graceful_model_exit_leaves_once_and_latches() {
        use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
        let hits = std::sync::Arc::new(AtomicUsize::new(0));
        let hits_ref = hits.clone();
        let app = axum::Router::new().route(
            "/v1/leave",
            axum::routing::post(move |axum::Json(body): axum::Json<serde_json::Value>| {
                let hits = hits_ref.clone();
                async move {
                    hits.fetch_add(1, AtomicOrdering::SeqCst);
                    let _ = body;
                    axum::Json(serde_json::json!({"left": true}))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let mut cfg = NodeConfig::default();
        cfg.ensure_identity().expect("test identity");
        let (state, _rx) = MeshState::new(cfg).expect("test mesh state");
        {
            let mut d = state.discovery.lock().await;
            d.configured = vec![format!("http://{addr}")];
            d.last_good = None;
        }

        graceful_model_exit(&state, Some("newmodel (bogquant)".to_string())).await;
        assert_eq!(
            hits.load(AtomicOrdering::SeqCst),
            1,
            "exactly one leave fired"
        );
        assert!(
            state
                .left_mesh_latch
                .load(std::sync::atomic::Ordering::SeqCst),
            "advertise latch closed"
        );
        assert_eq!(
            state.pending_model.read().await.as_deref(),
            Some("newmodel (bogquant)"),
            "pending restart model recorded"
        );
        // Idempotence: the reload executor runs every tick — the second
        // invocation (any pending value) must not re-fire the leave.
        graceful_model_exit(&state, Some("newmodel (bogquant)".to_string())).await;
        assert_eq!(
            hits.load(AtomicOrdering::SeqCst),
            1,
            "latch made the leave one-shot"
        );
    }

    /// Phase 3 speculative prefetch: after a serve/dispatch of a layer-k
    /// expert, the state's background walk preloads the next layer's
    /// demand-ranked LOCAL experts into the LRU (config knobs respected;
    /// hosting honored; no mesh fetches).
    #[tokio::test]
    async fn prefetch_after_serve_warms_the_next_layer_demand_ranked() {
        let mut cfg = NodeConfig::default();
        cfg.ensure_identity().expect("test identity");
        let (state, _rx) = MeshState::new(cfg).expect("test mesh state");
        let dir = std::env::temp_dir().join(format!(
            "jerith-prefetch-test-{}-{}",
            std::process::id(),
            crate::bootstrap::now_ms()
        ));
        let store = Arc::new(crate::expert_store::ExpertStore::new(&dir, 16).expect("test store"));
        store
            .put(&crate::expert_store::ExpertWeights::synthetic(
                "L1.E5", 4, 3,
            ))
            .unwrap();
        state.set_expert_store(store.clone()).await;
        // Demand: layer 0 is being served now; layer 1 has been observed
        // before (the node hosts L1.E5, so earlier traffic recorded it).
        {
            let mut stats = state.expert_stats.lock().expect("stats");
            stats.record("L1.E5");
        }
        state.spawn_prefetch_after_serve("L0.E0");
        // The walk is spawned; give it a beat, then check the resident set.
        tokio::time::timeout(
            std::time::Duration::from_secs(2),
            tokio::time::sleep(std::time::Duration::from_millis(150)),
        )
        .await
        .expect("sleep");
        assert!(
            store.is_resident("L1.E5"),
            "the hosted next-layer demand candidate must be preloaded"
        );
        assert_eq!(store.stats().prefetches, 1); // two candidates
        // A hosted expert on an observed-but-colder layer-out position: with
        // per_layer = 2 L1.E5 is the only layer-1 candidate this node hosts.
        // Not-hosted candidates are skipped without touching the mesh.
        std::fs::remove_dir_all(&dir).ok();
        // The guard resets: a second walk is allowed afterwards.
        assert!(
            !state.prefetching.load(std::sync::atomic::Ordering::SeqCst),
            "the in-flight guard must clear after the walk"
        );
    }

    #[test]
    fn config_reload_plan_prefetch_knobs_are_hot() {
        let live = NodeConfig::default();
        let mut disk = NodeConfig::default();
        disk.experts.prefetch_enabled = false;
        let plan = plan_config_reload(&disk, &live);
        assert!(plan.hot.contains(&"experts.prefetch"));
        let mut disk = NodeConfig::default();
        disk.experts.prefetch_per_layer = 5;
        let plan = plan_config_reload(&disk, &live);
        assert!(plan.hot.contains(&"experts.prefetch"));
        // Identical configs stay silent.
        assert!(plan_config_reload(&live, &live).hot.is_empty());
    }

    #[test]
    fn config_reload_plan_classifies_fields() {
        let live = NodeConfig::default();
        let disk = NodeConfig {
            contribution_percent: live.contribution_percent + 5,
            bootstrap_url: "http://other-bootstrap/".to_string(),
            node_id: "someone-else".to_string(),
            ..NodeConfig::default()
        };
        let plan = plan_config_reload(&disk, &live);
        assert!(plan.hot.contains(&"contribution_percent"));
        assert_eq!(plan.cold, vec!["bootstrap_url"]);
        assert_eq!(plan.identity_violations, vec!["node_id"]);
        assert!(!plan.memory_changed);
    }

    #[test]
    fn config_reload_plan_detects_memory_triple_change() {
        let live = NodeConfig::default();
        let disk = NodeConfig {
            memory_backend: MemoryBackend::PostgresPgvector,
            memory_dsn: Some("postgres://u:p@127.0.0.1:5432/db?sslmode=disable".to_string()),
            ..NodeConfig::default()
        };
        let plan = plan_config_reload(&disk, &live);
        assert!(plan.memory_changed);
        assert!(plan.hot.is_empty(), "memory moves are not plain hot fields");
        assert!(plan.cold.is_empty());
        // Identical configs stay silent — the tick must not log noise.
        assert!(plan_config_reload(&live, &live) == ConfigReloadPlan::default());
    }

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

    /// One fleet peer for the drill below: a real axum server answering
    /// the SIGNED /health contract, after a fixed modeled RTT. Fresh stamp
    /// per request (the verifier stamps-age-checks the reply).
    async fn spawn_health_peer(rtt_ms: u64) -> (String /*node_id*/, String /*url*/) {
        let identity = crate::signing::SigningIdentity::generate();
        let node_id = identity.node_id();
        let pubkey = identity.pubkey_hex();
        // Handler copies: the route owns its own captures per request call.
        let (h_id, h_pub, h_identity) = (node_id.clone(), pubkey.clone(), identity.clone());
        let app = axum::Router::new().route(
            "/health",
            axum::routing::get(move || {
                let node_id = h_id.clone();
                let pubkey = h_pub.clone();
                let identity = h_identity.clone();
                async move {
                    tokio::time::sleep(Duration::from_millis(rtt_ms)).await;
                    let stamp = now_ms();
                    let uptime = 10u64;
                    let contribution = 40u64;
                    let payload =
                        crate::signing::health_payload(&node_id, stamp, uptime, contribution);
                    axum::Json(serde_json::json!({
                        "node_id": node_id,
                        "pubkey": pubkey,
                        "stamp_ms": stamp,
                        "sig": identity.sign(payload.as_bytes()),
                        "uptime_seconds": uptime,
                        "contribution_percent": contribution,
                        "measurement": serde_json::Value::Null,
                    }))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move { axum::serve(listener, app).await.expect("serve") });
        (node_id, format!("http://{addr}/health"))
    }

    fn median(v: &mut [u64]) -> u64 {
        v.sort_unstable();
        v[v.len() / 2]
    }

    /// Fleet load drill (release, `#[ignore]`d like the memory benches):
    /// quantify the concurrent ping batch (was N × RTT serial per tick),
    /// N peers. Real HTTP fleet shape: N signed /health stub servers, each
    /// adding a fixed modeled RTT (a LAN run gives ~0 ms, which makes the
    /// serial-vs-batch gap invisible; the delay models the WAN hop the
    /// change is about). The A/B runs the SHIPPED batch (`ping_all`) and
    /// the pre-optimizer serial round-robin (the very same `ping_one`
    /// legs awaited one at a time: own stamp, own verify, same client
    /// pool, same state) over the same targets, alternating order per
    /// pass, medians reported.
    ///
    ///   cargo test --release -p jerith-mesh-core \
    ///     -- --ignored fleet_ping_drill --nocapture
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "release-mode fleet load drill (dev numbers)"]
    async fn fleet_ping_drill_batch_vs_serial() {
        const PEERS: usize = 30;
        const MODELED_RTT_MS: u64 = 25;
        const PASSES: usize = 7;

        let mut cfg = NodeConfig::default();
        cfg.ensure_identity().expect("test identity");
        let (state, _rx) = MeshState::new(cfg).expect("mesh state");

        // The registry's model gate locks ads out until it has a model —
        // pin one and make every ad match (the shipped flow does this
        // when the node's model initializes the gate each tick).
        let drill_model = crate::models::ModelId {
            id: "drill-fleet".into(),
            quant: "drill".into(),
            source: crate::models::ModelSource::Curated,
        };
        let mut reg = PeerRegistry::new(Some(drill_model));
        let mut targets: Vec<(String, String)> = Vec::new();
        for _ in 0..PEERS {
            let (node_id, url) = spawn_health_peer(MODELED_RTT_MS).await;
            // Expose the same targets the shipped path derives from the
            // registry: host:port address + /health.
            let address = url
                .trim_start_matches("http://")
                .trim_end_matches("/health")
                .to_string();
            reg.upsert(crate::bootstrap::PeerAd {
                node_id: node_id.clone(),
                address: address.clone(),
                rpc_address: address,
                model_id: "drill-fleet".into(),
                quant: "drill".into(),
                contribution_percent: 40,
                last_seen_ms: now_ms(),
                role: crate::config::NodeRole::Hybrid,
                resources: None,
                hosted_experts: None,
                hosted_expert_count: None,
                discovery_servers: Vec::new(),
                gossip_reputations: None,
            });
            targets.push((node_id, url));
        }
        *state.registry.lock().await = reg;

        // Deliberately unreachable: every peer answers, so the post-batch
        // failure machinery (the only bootstrap caller in ping_all) must
        // never dial it — if the drill regresses into failure accounting,
        // the timing sample is polluted and the client connection errors
        // become the tell.
        let bootstrap = BootstrapClient::new("http://127.0.0.1:1");
        let client = crate::http::mesh_http();
        let my_id = state.identity.node_id();

        let mut batch_ms = Vec::new();
        let mut serial_ms = Vec::new();
        for pass in 0..PASSES {
            // Alternate the A/B order so pass position can't favor either.
            let targets_now: Vec<(String, String)> = {
                let reg = state.registry.lock().await;
                reg.connectable()
                    .into_iter()
                    .map(|(ad, _)| (ad.node_id.clone(), format!("http://{}/health", ad.address)))
                    .collect()
            };
            assert_eq!(targets_now.len(), PEERS, "fleet size drifted");

            if pass % 2 == 0 {
                let started = std::time::Instant::now();
                ping_all(&state, &bootstrap).await;
                batch_ms.push(started.elapsed().as_millis() as u64);

                let started = std::time::Instant::now();
                for (node_id, url) in targets_now {
                    let _ = ping_one(&state, &bootstrap, &client, &my_id, node_id, url).await;
                }
                serial_ms.push(started.elapsed().as_millis() as u64);
            } else {
                let started = std::time::Instant::now();
                for (node_id, url) in targets_now {
                    let _ = ping_one(&state, &bootstrap, &client, &my_id, node_id, url).await;
                }
                serial_ms.push(started.elapsed().as_millis() as u64);

                let started = std::time::Instant::now();
                ping_all(&state, &bootstrap).await;
                batch_ms.push(started.elapsed().as_millis() as u64);
            }
        }

        let bm = median(&mut batch_ms);
        let sm = median(&mut serial_ms);
        println!("fleet drill: {PEERS} peers, modeled RTT {MODELED_RTT_MS} ms, {PASSES} passes");
        println!("  batched (shipped):  median {bm} ms  (all samples: {batch_ms:?})");
        println!("  serial (old):       median {sm} ms  (all samples: {serial_ms:?})");
        println!("  tick saving: {} ms (-{}%)", sm - bm, 100 * (sm - bm) / sm);

        // The line the change was made on: a batched tick must cost about
        // ONE modeled RTT (plus overhead), not N of them.
        assert!(
            bm <= 4 * MODELED_RTT_MS,
            "batched tick must be ~one RTT, got {bm} ms"
        );
        // And it must not merely be a little faster: with N={PEERS}, the
        // serial tick is ~N×RTT. A >2× cut is the conservative half of
        // the expected ~{PEERS}×.
        assert!(
            sm >= 2 * bm,
            "serial must pay many RTTs, got serial {sm} vs batched {bm} ms"
        );
    }
}
