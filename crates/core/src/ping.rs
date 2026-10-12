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

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{watch, Mutex, RwLock};
use tokio::time::{interval, MissedTickBehavior};

use crate::bootstrap::{
    now_ms, BootstrapClient, PeerAd, PeerRegistry, PeerState,
};
use crate::config::{NodeConfig, NodeRole};

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
}

impl MeshState {
    pub fn new(config: NodeConfig) -> anyhow::Result<(Arc<Self>, watch::Receiver<bool>)> {
        let (tx, rx) = watch::channel(true);
        let identity = config.identity()?;
        let state = Arc::new(Self {
            config: RwLock::new(config),
            registry: Mutex::new(PeerRegistry::new(None)),
            meters: Mutex::new(Meters::default()),
            identity,
            contribution_healthy: tx,
            started_at: std::time::Instant::now(),
        });
        Ok((state, rx))
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
        let auth = crate::signing::WireAuth::now(&state.identity, payload.as_bytes());
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
                let _ = bootstrap.report_dead(node_id).await;
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
    use crate::models::{ModelId, ModelSource};

    fn cfg(model_id: Option<&str>) -> NodeConfig {
        let mut c = NodeConfig::default();
        c.node_id = "self".into();
        c.ensure_identity().unwrap();
        if let Some(id) = model_id {
            c.model = Some(ModelId {
                id: id.into(),
                quant: "int4".into(),
                source: ModelSource::Curated,
            });
        }
        c
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
}
