//! Bootstrap discovery + model-homogeneous peer registry.
//!
//! Phase 0 wire format: the bootstrap node at `p2p.jerithai.com` serves
//! `GET /v1/peers?model=<model_id>` returning a JSON list of live nodes for
//! a given model. Nodes also POST their own advertisement there.
//!
//! The local registry only ever stores peers whose `model_id` (and quant)
//! match the local node's model — that is the *model-homogeneous rule*.

use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::Context as _;
use serde::{Deserialize, Serialize};

use crate::models::ModelId;

/// A peer advertisement as exchanged with the bootstrap node and peers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PeerAd {
    pub node_id: String,
    /// Host:port of the peer's HTTP gateway (health/status/API surface).
    pub address: String,
    pub model_id: String,
    pub quant: String,
    pub contribution_percent: u64,
    /// Unix ms of the peer's last self-reported heartbeat.
    pub last_seen_ms: u64,
    pub role: crate::config::NodeRole,
    /// Host:port of the peer's QUIC NodeRPC endpoint (expert dispatch).
    /// Serialization-defaulted for pre-RPC peers; empty string = absent.
    #[serde(default)]
    pub rpc_address: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerListResponse {
    pub peers: Vec<PeerAd>,
}

/// Running state for a peer in the local registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeerState {
    /// Advertised but not yet verified with a successful ping.
    Unknown,
    /// Last ping succeeded.
    Alive,
    /// Consecutive ping failures exceeded the deprecation threshold.
    Deprecated,
}

/// How many failed pings before a peer is marked Deprecated.
pub const DEPRECATE_AFTER_FAILED_PINGS: u32 = 3;

/// Peer health tracked locally.
#[derive(Debug, Clone)]
pub struct PeerHealth {
    pub state: PeerState,
    pub consecutive_failures: u32,
    pub last_rtt_ms: Option<u64>,
    pub last_ping_at: Option<Instant>,
    /// Whether the peer ever answered a ping successfully.
    pub ever_alive: bool,
}

/// Local in-memory registry of peers for THIS node's model only.
#[derive(Debug, Default)]
pub struct PeerRegistry {
    model: Option<ModelId>,
    peers: HashMap<String, PeerAd>,
    health: HashMap<String, PeerHealth>,
}

impl PeerRegistry {
    /// Read-only health lookup (for evidence decisions before mutation).
    pub fn health_of(&self, node_id: &str) -> Option<&PeerHealth> {
        self.health.get(node_id)
    }

    pub fn new(model: Option<ModelId>) -> Self {
        Self {
            model,
            peers: HashMap::new(),
            health: HashMap::new(),
        }
    }

    pub fn model(&self) -> Option<&ModelId> {
        self.model.as_ref()
    }

    /// The model-homogeneous gate: a peer is admitted only when its model id
    /// and quant match the local node's exactly. `None` model (unset) admits
    /// nothing since the mesh is defined by the model being run.
    pub fn admits(&self, ad: &PeerAd) -> bool {
        match &self.model {
            None => false,
            Some(m) => m.id == ad.model_id && m.quant == ad.quant,
        }
    }

    /// Insert or refresh a peer; returns false when rejected by the
    /// model-homogeneous rule (never stored, never connected to).
    pub fn upsert(&mut self, ad: PeerAd) -> bool {
        if !self.admits(&ad) {
            tracing::debug!(
                peer = %ad.node_id,
                model = %ad.model_id,
                "rejected peer: model-homogeneous mismatch"
            );
            return false;
        }
        let fresh = !self.peers.contains_key(&ad.node_id);
        let node_id = ad.node_id.clone();
        self.peers.insert(node_id.clone(), ad);
        if fresh {
            self.health.insert(
                node_id,
                PeerHealth {
                    state: PeerState::Unknown,
                    consecutive_failures: 0,
                    last_rtt_ms: None,
                    last_ping_at: None,
                    ever_alive: false,
                },
            );
        }
        true
    }

    /// Record a successful ping.
    pub fn record_ping_ok(&mut self, node_id: &str, rtt_ms: u64) {
        let h = self
            .health
            .entry(node_id.to_string())
            .or_insert_with(|| PeerHealth {
                state: PeerState::Unknown,
                consecutive_failures: 0,
                last_rtt_ms: None,
                last_ping_at: None,
                ever_alive: false,
            });
        h.state = PeerState::Alive;
        h.consecutive_failures = 0;
        h.last_rtt_ms = Some(rtt_ms);
        h.last_ping_at = Some(Instant::now());
        h.ever_alive = true;
    }

    /// Record a failed ping; flips to Deprecated after the threshold.
    pub fn record_ping_fail(&mut self, node_id: &str) {
        let Some(h) = self.health.get_mut(node_id) else {
            return;
        };
        h.consecutive_failures += 1;
        h.last_ping_at = Some(Instant::now());
        if h.consecutive_failures >= DEPRECATE_AFTER_FAILED_PINGS {
            h.state = PeerState::Deprecated;
        }
    }

    pub fn state_of(&self, node_id: &str) -> Option<PeerState> {
        self.health.get(node_id).map(|h| h.state)
    }

    pub fn rtt_of(&self, node_id: &str) -> Option<u64> {
        self.health.get(node_id).and_then(|h| h.last_rtt_ms)
    }

    /// All non-deprecated peers that are candidates for connection.
    pub fn connectable(&self) -> Vec<(&PeerAd, PeerHealth)> {
        self.peers
            .iter()
            .filter_map(|(id, ad)| self.health.get(id).map(|h| (ad, h.clone())))
            .filter(|(_, h)| h.state != PeerState::Deprecated)
            .collect()
    }

    pub fn all_ads(&self) -> Vec<PeerAd> {
        self.peers.values().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.peers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.peers.is_empty()
    }

    /// Swap the registry's model (model switch): per spec this means
    /// disconnecting everyone, so the registry clears itself.
    pub fn switch_model(&mut self, new_model: ModelId) {
        tracing::info!(
            from = ?self.model.as_ref().map(|m| m.id.clone()),
            to = %new_model.id,
            "model switch: disconnecting all peers"
        );
        self.model = Some(new_model);
        self.peers.clear();
        self.health.clear();
    }
}

/// HTTP client for the bootstrap node.
#[derive(Debug, Clone)]
pub struct BootstrapClient {
    base_url: String,
    inner: reqwest::Client,
}

impl BootstrapClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            inner: reqwest::Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .expect("reqwest client"),
        }
    }

    /// Fetch live peers for a model from the bootstrap node.
    pub async fn fetch_peers(&self, model: &str) -> anyhow::Result<Vec<PeerAd>> {
        let url = format!(
            "{}/v1/peers?model={}",
            self.base_url.trim_end_matches('/'),
            urlencode(model)
        );
        let resp = self
            .inner
            .get(&url)
            .send()
            .await
            .context("bootstrap fetch failed")?
            .error_for_status()
            .context("bootstrap returned an error")?;
        let parsed: PeerListResponse = resp
            .json()
            .await
            .context("bootstrap returned unparseable peer list")?;
        Ok(parsed.peers)
    }

    /// Signed variant of `fetch_peers`: the wire-auth headers put the hit
    /// in the bootstrap's peer-identity rate lane (per pubkey, generous
    /// budget) instead of its per-IP anonymous bucket — the allowance that
    /// lets large many-nodes-behind-one-IP deployments scale past the
    /// anonymous flood floor. Fails silently into the anonymous lane on
    /// the server if the signature doesn't verify (same result, tighter
    /// budget) — never a hard error path.
    pub async fn fetch_peers_signed(
        &self,
        model: &str,
        identity: &crate::signing::SigningIdentity,
    ) -> anyhow::Result<Vec<PeerAd>> {
        let url = format!(
            "{}/v1/peers?model={}",
            self.base_url.trim_end_matches('/'),
            urlencode(model)
        );
        let stamp = now_ms();
        let payload = crate::signing::peers_fetch_payload(stamp);
        let auth = crate::signing::WireAuth::with_stamp(identity, stamp, payload.as_bytes());
        let resp = self
            .inner
            .get(&url)
            .header("x-jerith-pubkey", &auth.pubkey)
            .header("x-jerith-stamp", auth.stamp_ms.to_string())
            .header("x-jerith-sig", &auth.sig)
            .send()
            .await
            .context("bootstrap fetch failed")?
            .error_for_status()
            .context("bootstrap returned an error")?;
        let parsed: PeerListResponse = resp
            .json()
            .await
            .context("bootstrap returned unparseable peer list")?;
        Ok(parsed.peers)
    }

    /// Advertise ourselves with the bootstrap node (heartbeat).
    /// The advertisement is Ed25519-signed over the canonical payload;
    /// the bootstrap rejects anything it cannot verify.
    pub async fn advertise(
        &self,
        ad: &PeerAd,
        identity: &crate::signing::SigningIdentity,
    ) -> anyhow::Result<()> {
        let url = format!("{}/v1/advertise", self.base_url.trim_end_matches('/'));
        let stamp = now_ms();
        let payload = crate::signing::advertise_payload(ad, stamp);
        // Stamp frozen (see signing::WireAuth::with_stamp — the payload
        // embeds the stamp; two clock reads race at ms boundaries).
        let auth = crate::signing::WireAuth::with_stamp(identity, stamp, payload.as_bytes());
        self.inner
            .put(&url)
            .json(ad)
            .header("x-jerith-pubkey", &auth.pubkey)
            .header("x-jerith-stamp", auth.stamp_ms.to_string())
            .header("x-jerith-sig", &auth.sig)
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }

    /// Report that a peer is dead (so the bootstrap deprecates it globally).
    /// Signed: the bootstrap now requires `jerith-dead1|stamp|reporter|target`
    /// wire-auth headers and refuses unsigned eviction votes (fail-closed).
    pub async fn report_dead(
        &self,
        node_id: &str,
        reporter_node_id: &str,
        identity: &crate::signing::SigningIdentity,
    ) -> anyhow::Result<()> {
        let stamp = now_ms();
        let payload = crate::signing::dead_payload(reporter_node_id, node_id, stamp);
        let auth = crate::signing::WireAuth::with_stamp(identity, stamp, payload.as_bytes());
        let url = format!("{}/v1/report-dead", self.base_url.trim_end_matches('/'));
        self.inner
            .post(&url)
            .json(&serde_json::json!({ "node_id": node_id, "reporter": reporter_node_id }))
            .header("x-jerith-pubkey", &auth.pubkey)
            .header("x-jerith-stamp", auth.stamp_ms.to_string())
            .header("x-jerith-sig", &auth.sig)
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }

    /// Submit fraud evidence about a peer, signed so the bootstrap can
    /// attribute (and someday weight) the report to the reporter.
    pub async fn report_fraud(
        &self,
        kind: crate::bootstrap_server::FraudKind,
        target_node_id: &str,
        reporter_node_id: &str,
        identity: &crate::signing::SigningIdentity,
    ) -> anyhow::Result<serde_json::Value> {
        if target_node_id == reporter_node_id {
            anyhow::bail!("a node cannot report itself");
        }
        let stamp = now_ms();
        let payload = format!(
            "jerith-fraud1|{stamp}|{}|{}|{}",
            match kind {
                crate::bootstrap_server::FraudKind::FailedWorker => "failed_worker",
                crate::bootstrap_server::FraudKind::Leeching => "leeching",
                crate::bootstrap_server::FraudKind::DishonestResults => "dishonest_results",
                crate::bootstrap_server::FraudKind::RateAbuse => "rate_abuse",
            },
            reporter_node_id,
            target_node_id,
        );
        // Stamp frozen with the payload (see signing::WireAuth::with_stamp).
        let auth = crate::signing::WireAuth::with_stamp(identity, stamp, payload.as_bytes());
        let url = format!("{}/v1/report-fraud", self.base_url.trim_end_matches('/'));
        let resp = self
            .inner
            .post(&url)
            .json(&serde_json::json!({
                "node_id": target_node_id,
                "reporter": reporter_node_id,
                "kind": kind,
            }))
            .header("x-jerith-pubkey", &auth.pubkey)
            .header("x-jerith-stamp", auth.stamp_ms.to_string())
            .header("x-jerith-sig", &auth.sig)
            .send()
            .await?
            .error_for_status()?;
        Ok(resp.json().await?)
    }

    /// Optional: live peer counts per model (used by setup UI).
    pub async fn model_counts(&self) -> anyhow::Result<Vec<(String, u64)>> {
        let url = format!("{}/v1/models/counts", self.base_url.trim_end_matches('/'));
        let resp = self
            .inner
            .get(&url)
            .send()
            .await
            .context("model counts fetch failed")?
            .error_for_status()
            .context("model counts request failed")?;
        let parsed: Vec<serde_json::Value> = resp.json().await?;
        Ok(parsed
            .into_iter()
            .filter_map(|v| {
                Some((
                    v.get("model_id")?.as_str()?.to_string(),
                    v.get("count")?.as_u64()?,
                ))
            })
            .collect())
    }
}

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' | '~' => out.push(c),
            _ => out.push_str(&format!("%{:02X}", c as u32)),
        }
    }
    out
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::NodeRole;
    use crate::models::{ModelId, ModelSource};

    fn model(id: &str, quant: &str) -> ModelId {
        ModelId {
            id: id.into(),
            quant: quant.into(),
            source: ModelSource::Curated,
        }
    }

    fn ad(node: &str, model_id: &str, quant: &str) -> PeerAd {
        PeerAd {
            node_id: node.into(),
            address: format!("127.0.0.1:{}", 9000 + node.len()),
            model_id: model_id.into(),
            quant: quant.into(),
            contribution_percent: 30,
            rpc_address: String::new(),
            last_seen_ms: now_ms(),
            role: NodeRole::Hybrid,
        }
    }

    #[test]
    fn homogeneous_rule_admits_exact_match_only() {
        let mut reg = PeerRegistry::new(Some(model("glm-5.3", "int4")));
        assert!(reg.upsert(ad("a", "glm-5.3", "int4")));
        assert!(!reg.upsert(ad("b", "glm-5.2", "int4")));
        assert!(!reg.upsert(ad("c", "glm-5.3", "fp16")));
        assert!(!reg.upsert(ad("d", "qwen3.8-flash-next", "int4")));
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn registry_without_model_admits_nothing() {
        let mut reg = PeerRegistry::new(None);
        assert!(!reg.upsert(ad("a", "glm-5.3", "int4")));
        assert!(reg.is_empty());
    }

    #[test]
    fn deprecated_after_three_failures_and_recovers() {
        let mut reg = PeerRegistry::new(Some(model("glm-5.3", "int4")));
        reg.upsert(ad("a", "glm-5.3", "int4"));
        reg.record_ping_ok("a", 25);
        assert_eq!(reg.state_of("a"), Some(PeerState::Alive));
        reg.record_ping_fail("a");
        reg.record_ping_fail("a");
        assert_eq!(reg.state_of("a"), Some(PeerState::Alive)); // below threshold
        reg.record_ping_fail("a");
        assert_eq!(reg.state_of("a"), Some(PeerState::Deprecated));
        // Node comes back online → back in the list.
        reg.record_ping_ok("a", 40);
        assert_eq!(reg.state_of("a"), Some(PeerState::Alive));
    }

    #[test]
    fn connectable_excludes_deprecated() {
        let mut reg = PeerRegistry::new(Some(model("glm-5.3", "int4")));
        reg.upsert(ad("a", "glm-5.3", "int4"));
        reg.upsert(ad("b", "glm-5.3", "int4"));
        for _ in 0..DEPRECATE_AFTER_FAILED_PINGS {
            reg.record_ping_fail("a");
        }
        let ids: Vec<String> = reg
            .connectable()
            .into_iter()
            .map(|(ad, _)| ad.node_id.clone())
            .collect();
        assert_eq!(ids, vec!["b".to_string()]);
    }

    #[test]
    fn rtt_recorded_on_success() {
        let mut reg = PeerRegistry::new(Some(model("glm-5.3", "int4")));
        reg.upsert(ad("a", "glm-5.3", "int4"));
        reg.record_ping_ok("a", 55);
        assert_eq!(reg.rtt_of("a"), Some(55));
    }

    #[test]
    fn switch_model_clears_everything() {
        let mut reg = PeerRegistry::new(Some(model("glm-5.3", "int4")));
        reg.upsert(ad("a", "glm-5.3", "int4"));
        reg.switch_model(model("qwen3.8-flash-next", "int4"));
        assert_eq!(reg.len(), 0);
        // And the old peer no longer matches the new homogeneous gate.
        assert!(!reg.upsert(ad("a", "glm-5.3", "int4")));
    }

    #[test]
    fn ping_fail_unknown_peer_is_noop() {
        let mut reg = PeerRegistry::new(Some(model("glm-5.3", "int4")));
        reg.record_ping_fail("ghost");
        assert!(reg.is_empty());
    }

    #[test]
    fn urlencode_encodes_specials() {
        assert_eq!(urlencode("glm-5.3"), "glm-5.3");
        assert_eq!(urlencode("a b/c"), "a%20b%2Fc");
    }
}
