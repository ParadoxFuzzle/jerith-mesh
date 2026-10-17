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
    /// Live resource offering (RAM / VRAM / GPU / CPU) for pooling:
    /// placement ordering + the fleet census read this. Wire-compatible
    /// with pre-pooling peers — omitted on serialize when unreported,
    /// defaulted on deserialize when absent, so old nodes and old
    /// bootstraps exchange ads unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resources: Option<crate::hw::ResourceSample>,
    /// Expert ids this node serves with **real tensors** (Phase 2 tensor
    /// transport) — capped at [`HOSTED_EXPERTS_AD_MAX`] entries, sorted.
    /// `None` = the peer does not advertise hosting (pre-Phase-2 node):
    /// placement keeps the historical "assume it has the expert" behavior
    /// so mixed meshes still work. `Some([])` = the peer advertises that it
    /// holds no experts, so expert placement skips it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hosted_experts: Option<Vec<String>>,
    /// Total number of experts held, when that exceeds the advertised list
    /// (the tail stays unlisted). Placement treats "listed ∪ unknown tail"
    /// as hosts, known-not-listed as not.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hosted_expert_count: Option<u64>,
    /// Bootstrap servers this node rotates through (Phase 1 discovery
    /// redundancy): gossip so the fleet's server list stays fresh without
    /// a coordinator. Sanitized by the sender (https, deduped, capped —
    /// `discovery::AD_MAX_SERVERS`); the receiver merges only https URLs
    /// from ads (plain http LAN bootstraps are config-only). Empty on
    /// pre-discovery ads — wire-compatible.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub discovery_servers: Vec<String>,
}

/// Cap on the advertised expert-id list: a long tail is represented by
/// `hosted_expert_count` instead of bloating every advertise.
pub const HOSTED_EXPERTS_AD_MAX: usize = 256;

/// One row of the bootstrap's per-model peer census (`GET
/// /v1/models/counts`): how many live nodes run the model, and how that
/// count splits over quants — the pre-join compat check's data source.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelCensusRow {
    pub model_id: String,
    pub count: u64,
    /// quant → peer count. BTree for stable, readable verdicts.
    pub quants: std::collections::BTreeMap<String, u64>,
}

impl PeerAd {
    /// Does this peer claim to host `expert_id`?
    ///
    /// - advertised list contains the id → yes
    /// - no advertisement at all (legacy peer) → yes (unknown, historical)
    /// - advertises a list, id absent, but the count says there is an
    ///   unlisted tail → yes (unknown tail)
    /// - advertises everything it holds and the id is not among it → no
    pub fn hosts_expert(&self, expert_id: &str) -> bool {
        match &self.hosted_experts {
            None => true,
            Some(list) => {
                if list.iter().any(|e| e == expert_id) {
                    return true;
                }
                self.hosted_expert_count
                    .map(|c| c as usize > list.len())
                    .unwrap_or(false)
            }
        }
    }

    /// Does this peer's ad *explicitly* name the expert? (Unlike
    /// [`hosts_expert`], an unadvertised/legacy peer is `false` here —
    /// placement may still try it, but replication never counts it as a
    /// confirmed holder or a transfer source.)
    pub fn explicitly_hosts(&self, expert_id: &str) -> bool {
        match &self.hosted_experts {
            Some(list) => list.iter().any(|e| e == expert_id),
            None => false,
        }
    }

    /// Build the advertisement fields from a node's hosted expert ids
    /// (sorted, capped; the count always carries the true total).
    pub fn advertise_hosted_experts(ids: &[String]) -> (Option<Vec<String>>, Option<u64>) {
        let mut sorted = ids.to_vec();
        sorted.sort();
        let count = sorted.len() as u64;
        sorted.truncate(HOSTED_EXPERTS_AD_MAX);
        (Some(sorted), Some(count))
    }
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
                .use_rustls_tls()
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

    /// GET /v1/models/counts — the live per-model peer census, with the
    /// per-quant breakdown that makes the pre-join compat check readable
    /// ("same model, wrong quant — N nodes could still take you").
    pub async fn fetch_model_census(&self) -> anyhow::Result<Vec<ModelCensusRow>> {
        let url = format!("{}/v1/models/counts", self.base_url.trim_end_matches('/'));
        let parsed: Vec<serde_json::Value> = self
            .inner
            .get(&url)
            .send()
            .await
            .context("bootstrap census fetch failed")?
            .error_for_status()
            .context("bootstrap census returned an error")?
            .json()
            .await
            .context("bootstrap census returned unparseable JSON")?;
        Ok(parsed
            .into_iter()
            .map(|row| ModelCensusRow {
                model_id: row
                    .get("model_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string(),
                count: row.get("count").and_then(|v| v.as_u64()).unwrap_or(0),
                quants: row
                    .get("quants")
                    .and_then(|v| v.as_array())
                    .map(|rows| {
                        rows.iter()
                            .filter_map(|r| {
                                Some((
                                    r.get("quant")?.as_str()?.to_string(),
                                    r.get("count").and_then(|v| v.as_u64()).unwrap_or(0),
                                ))
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
            })
            .collect())
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

    /// Gracefully retract our own advertisement (§2 model-switch/ exit
    /// machinery): signed — the bootstrap only removes entries the matching
    /// identity asks to remove. Idempotent on the server (`left:false`).
    pub async fn leave(
        &self,
        node_id: &str,
        identity: &crate::signing::SigningIdentity,
    ) -> anyhow::Result<bool> {
        let stamp = now_ms();
        let payload = crate::signing::leave_payload(node_id, stamp);
        let auth = crate::signing::WireAuth::with_stamp(identity, stamp, payload.as_bytes());
        let url = format!("{}/v1/leave", self.base_url.trim_end_matches('/'));
        let body: serde_json::Value = self
            .inner
            .post(&url)
            .json(&serde_json::json!({ "node_id": node_id }))
            .header("x-jerith-pubkey", &auth.pubkey)
            .header("x-jerith-stamp", auth.stamp_ms.to_string())
            .header("x-jerith-sig", &auth.sig)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        Ok(body.get("left").and_then(serde_json::Value::as_bool).unwrap_or(false))
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
            resources: None,
            hosted_experts: None,
            hosted_expert_count: None,
            discovery_servers: Vec::new(),
        }
    }

    #[test]
    fn hosted_expert_advertisement_drives_placement() {
        // Legacy peer: no advertisement → historical "assume hosted".
        let legacy = ad("legacy", "glm-5.3", "int4");
        assert!(legacy.hosts_expert("L3.E17"));

        // Advertising peer that holds nothing → expert placement skips it.
        let mut idle = ad("idle", "glm-5.3", "int4");
        let (list, count) = PeerAd::advertise_hosted_experts(&[]);
        idle.hosted_experts = list;
        idle.hosted_expert_count = count;
        assert!(!idle.hosts_expert("L3.E17"));

        // Advertising peer with a listed expert.
        let mut holder = ad("holder", "glm-5.3", "int4");
        let (list, count) = PeerAd::advertise_hosted_experts(&["L3.E17".into()]);
        holder.hosted_experts = list;
        holder.hosted_expert_count = count;
        assert!(holder.hosts_expert("L3.E17"));
        assert!(!holder.hosts_expert("L3.E18"));

        // Truncated advertisement with an unlisted tail → unknown, so the
        // peer stays a candidate rather than being wrongly excluded.
        let mut tail = ad("tail", "glm-5.3", "int4");
        let many: Vec<String> = (0..HOSTED_EXPERTS_AD_MAX + 5)
            .map(|i| format!("L0.E{i}"))
            .collect();
        let (list, count) = PeerAd::advertise_hosted_experts(&many);
        assert_eq!(list.as_ref().unwrap().len(), HOSTED_EXPERTS_AD_MAX);
        assert_eq!(count, Some(many.len() as u64));
        tail.hosted_experts = list;
        tail.hosted_expert_count = count;
        assert!(tail.hosts_expert("L0.E259"), "unlisted tail stays a host");
    }

    #[test]
    fn hosted_expert_fields_are_wire_compatible_with_pre_phase2_peers() {
        // An old ad JSON (no hosting fields) must parse as "unknown".
        let old = serde_json::json!({
            "node_id": "old-node",
            "address": "1.2.3.4:8900",
            "model_id": "glm-5.3",
            "quant": "int4",
            "contribution_percent": 40,
            "last_seen_ms": 1,
            "role": "hybrid",
            "rpc_address": "1.2.3.4:9000"
        });
        let parsed: PeerAd = serde_json::from_value(old).unwrap();
        assert_eq!(parsed.hosted_experts, None);
        assert!(
            parsed.hosts_expert("anything"),
            "legacy peer stays a candidate"
        );
        // And a peer with no hosting fields serializes without them.
        let out = serde_json::to_value(ad("n", "m", "int4")).unwrap();
        assert!(out.get("hosted_experts").is_none());
    }

    #[test]
    fn resources_field_is_wire_compatible_with_pre_pooling_peers() {
        // (1) an old ad WITHOUT the field deserializes (default None) and
        // re-serializes omitting it — old ↔ new nodes exchange ads unchanged
        let legacy = r#"{"node_id":"a","address":"127.0.0.1:1","model_id":"glm-5.3-flash","quant":"default","contribution_percent":25,"last_seen_ms":1,"role":"hybrid","rpc_address":""}"#;
        let ad: PeerAd = serde_json::from_str(legacy).expect("legacy ad parses");
        assert!(ad.resources.is_none());
        // Re-serialization omits the field entirely → old peers see
        // byte-identical shapes, no compat shim needed.
        assert!(
            serde_json::to_string(&ad)
                .map(|s| !s.contains("resources"))
                .unwrap_or(false)
        ); // (2) a new ad with a sample round-trips intact
        let full = serde_json::json!({
            "node_id": "b",
            "address": "127.0.0.1:2",
            "model_id": "glm-5.3-flash",
            "quant": "default",
            "contribution_percent": 40,
            "last_seen_ms": 2,
            "role": "hybrid",
            "rpc_address": "127.0.0.1:9000",
            "resources": {
                "ram_total_bytes": 32_000_000_000u64,
                "ram_available_bytes": 16_000_000_000u64,
                "swap_total_bytes": 0,
                "cpu_logical_cores": 8,
                "gpus": [{"name": "RTX 4060", "vram_total_bytes": 8_000_000_000u64,
                           "vram_used_bytes": 5_000_000_000u64, "vram_free_bytes": 3_000_000_000u64}],
                "sampled_at_ms": 99,
            }
        })
        .to_string();
        let ad2: PeerAd = serde_json::from_str(&full).expect("pooled ad parses");
        assert_eq!(
            ad2.resources.as_ref().unwrap().vram_free_total(),
            Some(3_000_000_000)
        );
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
