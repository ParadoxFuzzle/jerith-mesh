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

use tokio::sync::{watch, Mutex, RwLock};
use tokio::time::{interval, MissedTickBehavior};

use crate::bootstrap::{
    now_ms, BootstrapClient, PeerAd, PeerRegistry, PeerState,
};
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
    /// Phase 4 verification counters; the ping loop flushes queued
    /// DishonestResults evidence to the bootstrap on its regular cadence.
    pub verify: SharedVerifyTracker,
    /// Phase 5 stake + slashing handle (ledger persists across restarts).
    pub stake: Arc<crate::stake::StakeManager>,
}

impl MeshState {
    pub fn new(config: NodeConfig) -> anyhow::Result<(Arc<Self>, watch::Receiver<bool>)> {
        let (tx, rx) = watch::channel(true);
        let identity = config.identity()?;
        let stake = crate::stake::StakeManager::new(
            &config.crypto,
            Some(identity.clone()),
        );
        let state = Arc::new(Self {
            config: RwLock::new(config),
            registry: Mutex::new(PeerRegistry::new(None)),
            meters: Mutex::new(Meters::default()),
            identity,
            contribution_healthy: tx,
            started_at: std::time::Instant::now(),
            verify: crate::verify::VerificationTracker::shared(),
            stake,
        });
        Ok((state, rx))
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
                verify_percent,
                verify_tolerance: 0.0,
            },
            verify,
        ));
        Ok(Arc::new(Self { endpoint, router, pins }))
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
    /// peers (rtt, alive) + their advertised QUIC rpc addresses.
    async fn mesh_view(
        &self,
        state: &Arc<MeshState>,
    ) -> (Vec<(String, u64, bool, bool)>, HashMap<String, SocketAddr>) {
        let reg = state.registry.lock().await;
        let peers: Vec<(String, u64, bool, bool)> = reg
            .all_ads()
            .into_iter()
            .filter_map(|ad| {
                let health = reg.health_of(&ad.node_id)?;
                let alive = health.state != PeerState::Deprecated;
                let rtt = health.last_rtt_ms.unwrap_or(u64::MAX >> 2);
                Some((ad.node_id.clone(), rtt, true, alive))
            })
            .collect();
        let addrs: HashMap<String, SocketAddr> = reg
            .all_ads()
            .into_iter()
            .filter(|ad| !ad.rpc_address.is_empty())
            .filter_map(|ad| {
                ad.rpc_address
                    .parse::<SocketAddr>()
                    .ok()
                    .map(|a| (ad.node_id.clone(), a))
            })
            .collect();
        (peers, addrs)
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
        let (peers, addrs) = self.mesh_view(state).await;
        self.refresh_pins(state).await;
        let local_attempt = local_try();
        let router = self.router.clone();
        let endpoint = self.endpoint.clone();
        let identity = state.identity.clone();
        let local_closed = local_attempt.is_err();
        match router
            .chat_over_quic(&endpoint, &identity, &addrs, &req_id, messages, &peers, || {
                local_attempt
            })
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
            let bootstrap = BootstrapClient::new(st.config.read().await.bootstrap_url.clone());
            run_loop(st, bootstrap, ping_interval, &mut loop_rx).await;
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
async fn run_loop(
    state: Arc<MeshState>,
    bootstrap: BootstrapClient,
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

        // Initialize the registry's model gate on first iteration (or after
        // a model switch event, which cleared it).
        {
            let mut reg = state.registry.lock().await;
            let gate_matches = reg.model().map(|m| m.id.clone()) == model.as_ref().map(|m| m.id.clone());
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
            match bootstrap.fetch_peers(&m.id).await {
                Ok(peers) => {
                    let mut reg = state.registry.lock().await;
                    let fetched = peers.len();
                    for ad in peers {
                        reg.upsert(ad);
                    }
                    tracing::info!(
                        target: "mesh",
                        fetched,
                        "peer list refreshed from bootstrap"
                    );
                }
                Err(e) => {
                    // Bootstrap turned this node away (signature/expiry) or
                    // is unreachable — must be visible, not silent.
                    tracing::warn!(
                        target: "mesh",
                        error = %e,
                        "bootstrap list refresh failed — degraded (local-only)"
                    );
                    let _ = state.contribution_healthy.send(false);
                }
            }

            // Ping every connectable peer.
            ping_all(&state, &bootstrap).await;

            // Advertise ourselves (signed); rpc_address points at our
            // QUIC NodeRPC so peers can dispatch experts to us.
            let (ad, identity) = {
                let cfg = state.config.read().await;
                let ad = PeerAd {
                    node_id: cfg.node_id.clone(),
                    address: format!("127.0.0.1:{}", cfg.gateway_port),
                    rpc_address: format!("127.0.0.1:{}", cfg.effective_rpc_port()),
                    model_id: my_ad_model.0,
                    quant: my_ad_model.1,
                    contribution_percent: cfg.contribution_percent,
                    last_seen_ms: now_ms(),
                    role: cfg.role,
                };
                (ad, state.identity.clone())
            };
            if let Err(e) = bootstrap.advertise(&ad, &identity).await {
                // An advertise failure means the mesh can't see us — warn.
                tracing::warn!(target: "mesh", error = %e, "advertise rejected/failed");
            } else {
                tracing::debug!(target: "mesh", "advertised to bootstrap");
            }
        }

        // Phase 4: ringer spot-check (per config) against a random alive
        // peer, THEN (same cadence) flush queued DishonestResults evidence
        // to the bootstrap — never on the inference hot path.
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
                let evidence = format!(
                    "dishonest-results|{}|{target}",
                    state.identity.node_id()
                );
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
                ad.rpc_address.parse::<SocketAddr>().ok().map(|a| (ad.node_id.clone(), a))
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
                    &req_id,
                    expert_id,
                    &server_id,
                    &input,
                    &output,
                    c,
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
