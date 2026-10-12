//! Bootstrap node server: the single-file trust anchor at p2p.jerithai.com.
//!
//! Endpoints (Phase 0 HTTP/JSON wire format):
//!   GET  /health              liveness for Cloudflare Tunnel checks
//!   GET  /v1/peers?model=     live peer list for a model (TTL-filtered)
//!   PUT  /v1/advertise        node self-advertisement (heartbeat)
//!   POST /v1/report-dead      peer-reported death → immediate deprecation
//!   GET  /v1/models/counts    live peer count per model (for the setup UI)
//!
//! State is in-memory with TTL expiry: an entry disappears after
//! `advertise_ttl` without a refresh, comes back on the next advertise,
//! and /v1/report-dead evicts immediately. Multiple independent bootstrap
//! nodes can run the same binary; Phase 1 adds gossip between them.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::bootstrap::{now_ms, PeerAd, PeerListResponse};

/// How long an advertisement stays live without refresh.
pub const DEFAULT_ADVERTISE_TTL_MS: u64 = 90_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapOptions {
    pub port: u16,
    pub advertise_ttl_ms: u64,
}

impl Default for BootstrapOptions {
    fn default() -> Self {
        Self {
            port: 8901,
            advertise_ttl_ms: DEFAULT_ADVERTISE_TTL_MS,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeadReport {
    node_id: String,
}

/// In-memory registry: node_id → (ad, last_refresh_ms, dead_votes).
#[derive(Default)]
struct Registry {
    entries: RwLock<HashMap<String, Entry>>,
}

#[derive(Debug, Clone)]
struct Entry {
    ad: PeerAd,
    last_refresh_ms: u64,
    /// Peer-submitted death reports; >= threshold evicts immediately.
    dead_votes: u32,
}

const DEAD_VOTES_TO_EVICTION: u32 = 2;

fn router(state: Arc<Registry>, ttl_ms: u64) -> Router {
    Router::new()
        .route("/health", get(bootstrap_health))
        .route("/v1/peers", get(peers))
        .route("/v1/advertise", put(advertise))
        .route("/v1/report-dead", post(report_dead))
        .route("/v1/models/counts", get(model_counts))
        .with_state((state, ttl_ms))
}

type Shared = (Arc<Registry>, u64);

async fn bootstrap_health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "ok": true, "service": "jerith-bootstrap" }))
}

/// Remove expired entries; returns a snapshot of live ones.
async fn sweep_live(registry: &Registry, ttl_ms: u64) -> Vec<Entry> {
    let now = now_ms();
    let mut entries = registry.entries.write().await;
    entries.retain(|_, e| now.saturating_sub(e.last_refresh_ms) < ttl_ms);
    entries.values().cloned().collect()
}

async fn peers(
    Query(params): Query<HashMap<String, String>>,
    State((registry, ttl_ms)): State<Shared>,
) -> Response {
    let want_model = params.get("model").cloned().unwrap_or_default();
    let live = sweep_live(&registry, ttl_ms).await;
    let peers: Vec<PeerAd> = live
        .into_iter()
        .filter(|e| want_model.is_empty() || e.ad.model_id == want_model)
        .map(|e| e.ad)
        .collect();
    (StatusCode::OK, Json(PeerListResponse { peers })).into_response()
}

async fn advertise(
    State((registry, ttl_ms)): State<Shared>,
    ad: Json<PeerAd>,
) -> Response {
    let mut ad = ad.0;
    // Never trust the client's own clock for expiry.
    ad.last_seen_ms = now_ms();
    if ad.node_id.is_empty() || ad.model_id.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "node_id and model_id are required"})),
        )
            .into_response();
    }
    let mut entries = registry.entries.write().await;
    entries.insert(
        ad.node_id.clone(),
        Entry {
            last_refresh_ms: now_ms(),
            dead_votes: 0,
            ad,
        },
    );
    let _ = ttl_ms;
    (StatusCode::OK, Json(serde_json::json!({"ok": true}))).into_response()
}

async fn report_dead(
    State((registry, _ttl_ms)): State<Shared>,
    Json(report): Json<DeadReport>,
) -> Response {
    let mut entries = registry.entries.write().await;
    match entries.get_mut(&report.node_id) {
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "unknown node"})),
        )
            .into_response(),
        Some(entry) => {
            entry.dead_votes += 1;
            if entry.dead_votes >= DEAD_VOTES_TO_EVICTION {
                entries.remove(&report.node_id);
                (StatusCode::OK, Json(serde_json::json!({"evicted": true}))).into_response()
            } else {
                (
                    StatusCode::OK,
                    Json(serde_json::json!({"evicted": false, "dead_votes": entry.dead_votes})),
                )
                    .into_response()
            }
        }
    }
}

async fn model_counts(State((registry, ttl_ms)): State<Shared>) -> Response {
    let live = sweep_live(&registry, ttl_ms).await;
    let mut counts: HashMap<String, u64> = HashMap::new();
    for e in live {
        *counts.entry(e.ad.model_id.clone()).or_default() += 1;
    }
    let mut out: Vec<serde_json::Value> = counts
        .into_iter()
        .map(|(model_id, count)| serde_json::json!({"model_id": model_id, "count": count}))
        .collect();
    out.sort_by_key(|v| {
        v.get("model_id")
            .and_then(|m| m.as_str())
            .unwrap_or_default()
            .to_string()
    });
    (StatusCode::OK, Json(out)).into_response()
}

/// Serve the bootstrap node until the process exits.
pub async fn serve(options: BootstrapOptions) -> anyhow::Result<()> {
    let registry = Arc::new(Registry::default());
    let app = router(registry, options.advertise_ttl_ms);
    let addr = SocketAddr::from(([0, 0, 0, 0], options.port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(port = options.port, "bootstrap node listening");
    axum::serve(listener, app).await?;
    Ok(())
}

/// Spawn the bootstrap node in-process (used by tests and by `jerith run`
/// when configured to co-host a bootstrap).
pub async fn spawn(options: BootstrapOptions) -> anyhow::Result<()> {
    tokio::spawn(async move {
        if let Err(e) = serve(options).await {
            tracing::error!(error = %e, "bootstrap server died");
        }
    });
    Ok(())
}

/// Unused-parameter silencer for Path (kept for future /v1/block/{height}).
#[allow(dead_code)]
async fn _path_used(_: Path<String>) {}

// ============================================================================
// Tests (in-process axum server via tokio)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use crate::config::NodeRole;
    use crate::models::{ModelId, ModelSource};

    fn ad(node: &str, model: &str) -> PeerAd {
        PeerAd {
            node_id: node.into(),
            address: format!("127.0.0.1:{}", 9000 + node.len()),
            model_id: model.into(),
            quant: "int4".into(),
            contribution_percent: 30,
            last_seen_ms: now_ms(),
            role: NodeRole::Hybrid,
        }
    }

    fn model(id: &str) -> ModelId {
        ModelId {
            id: id.into(),
            quant: "int4".into(),
            source: ModelSource::Curated,
        }
    }

    #[tokio::test]
    async fn advertise_then_list_and_counts() {
        let registry = Arc::new(Registry::default());
        let app = router(Arc::clone(&registry), DEFAULT_ADVERTISE_TTL_MS);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let base = format!("http://{addr}");
        let client = reqwest::Client::new();

        // Empty at first.
        let counts: Vec<serde_json::Value> =
            client.get(format!("{base}/v1/models/counts")).send().await.unwrap().json().await.unwrap();
        assert!(counts.is_empty());

        // Two glm-5.3 nodes, one qwen node.
        for (node, model) in [("a", "glm-5.3"), ("b", "glm-5.3"), ("c", "qwen3.8-flash-next")] {
            let r = client
                .put(format!("{base}/v1/advertise"))
                .json(&ad(node, model))
                .send()
                .await
                .unwrap();
            assert!(r.status().is_success());
        }

        // Peer list filtered by model: only same-model peers return.
        let list: PeerListResponse = client
            .get(format!("{base}/v1/peers?model=glm-5.3"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(list.peers.len(), 2);
        assert!(list.peers.iter().all(|p| p.model_id == "glm-5.3"));

        // Counts aggregate per model.
        let counts: Vec<serde_json::Value> =
            client.get(format!("{base}/v1/models/counts")).send().await.unwrap().json().await.unwrap();
        assert_eq!(counts.len(), 2);
        let glm = counts.iter().find(|c| c["model_id"] == "glm-5.3").unwrap();
        assert_eq!(glm["count"], 2);
    }

    #[tokio::test]
    async fn report_dead_evicts_after_threshold() {
        let registry = Arc::new(Registry::default());
        let app = router(Arc::clone(&registry), DEFAULT_ADVERTISE_TTL_MS);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let base = format!("http://{addr}");
        let client = reqwest::Client::new();

        client
            .put(format!("{base}/v1/advertise"))
            .json(&ad("dead-node", "glm-5.3"))
            .send()
            .await
            .unwrap();

        // First vote: not evicted.
        let r: serde_json::Value = client
            .post(format!("{base}/v1/report-dead"))
            .json(&serde_json::json!({"node_id": "dead-node"}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(r["evicted"], false);

        // Second vote: evicted.
        let r: serde_json::Value = client
            .post(format!("{base}/v1/report-dead"))
            .json(&serde_json::json!({"node_id": "dead-node"}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(r["evicted"], true);

        let list: PeerListResponse = client
            .get(format!("{base}/v1/peers?model=glm-5.3"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert!(list.peers.is_empty());
    }

    #[tokio::test]
    async fn ttl_expiry_drops_stale_entries() {
        let registry = Arc::new(Registry::default());
        let app = router(Arc::clone(&registry), 50); // 50 ms TTL
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let base = format!("http://{addr}");
        let client = reqwest::Client::new();

        client
            .put(format!("{base}/v1/advertise"))
            .json(&ad("ghost", "glm-5.3"))
            .send()
            .await
            .unwrap();
        let list: PeerListResponse = client
            .get(format!("{base}/v1/peers?model=glm-5.3"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(list.peers.len(), 1);

        tokio::time::sleep(Duration::from_millis(80)).await;
        let list: PeerListResponse = client
            .get(format!("{base}/v1/peers?model=glm-5.3"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert!(list.peers.is_empty(), "expired entry must drop from the list");
    }

    #[tokio::test]
    async fn advertise_rejects_missing_fields() {
        let registry = Arc::new(Registry::default());
        let app = router(Arc::clone(&registry), DEFAULT_ADVERTISE_TTL_MS);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let client = reqwest::Client::new();
        // Complete body but blank identity fields → handler must reject 400.
        let mut bad = serde_json::to_value(ad("", "glm-5.3")).unwrap();
        bad["model_id"] = serde_json::json!("");
        let r = client
            .put(format!("http://{}{}", addr, "/v1/advertise"))
            .json(&bad)
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::BAD_REQUEST);

        // Malformed body (missing required fields) → extractor rejects 422.
        let r = client
            .put(format!("http://{}{}", addr, "/v1/advertise"))
            .json(&serde_json::json!({"node_id": "x"}))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[test]
    fn default_options_match_spec_port() {
        let o = BootstrapOptions::default();
        assert_eq!(o.port, 8901);
        assert_eq!(o.advertise_ttl_ms, DEFAULT_ADVERTISE_TTL_MS);
    }

    #[test]
    fn model_roundtrip_is_preserved() {
        let m = model("glm-5.3");
        let json = serde_json::to_string(&m).unwrap();
        let back: ModelId = serde_json::from_str(&json).unwrap();
        assert_eq!(back, m);
    }
}
