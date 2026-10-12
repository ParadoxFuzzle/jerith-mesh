//! OpenAI-compatible local gateway + HTTP forwarder to the model runtime.
//!
//! Phase 0 jobs:
//! 1. Serve `POST /v1/chat/completions` and `GET /v1/models` — forwarding to
//!    the local Colibri/Ollama runtime (`runtime_url`, which itself exposes
//!    an OpenAI-compatible endpoint). Requests are metered; Phase 2 routes
//!    them to peer experts.
//! 2. Serve `GET /health` — the minimal-resource ping target other nodes
//!    hit to verify we're alive and contributing (adds <1 ms CPU).

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

use crate::economics::JerithMiner;
use crate::ping::MeshState;

pub struct Gateway {
    pub state: Arc<MeshState>,
    /// Base URL of the local Colibri/Ollama runtime.
    pub runtime_url: String,
    /// Optional JER miner (present when crypto chain = JerithChain).
    pub miner: Option<JerithMiner>,
}

impl Gateway {
    pub fn router(self) -> Router {
        let state = Arc::new(self);
        Router::new()
            .route("/health", get(health))
            .route("/status", get(status))
            .route("/v1/models", get(list_models))
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state)
    }
}

/// Minimal-resource ping target for other nodes. <1 ms CPU; reports only
/// identity + liveness data, never user data.
async fn health(State(gw): State<Arc<Gateway>>) -> Json<Value> {
    let cfg = gw.state.config.read().await;
    let meters = gw.state.meters.lock().await;
    Json(json!({
        "ok": true,
        "node_id": cfg.node_id,
        "model_id": cfg.model.as_ref().map(|m| m.id.clone()),
        "quant": cfg.model.as_ref().map(|m| m.quant.clone()),
        "contribution_percent": cfg.contribution_percent,
        "uptime_seconds": meters.uptime_seconds,
        "tokens_served": meters.tokens_served,
        "expert_activations": meters.expert_activations,
        "bytes_relayed": meters.bytes_relayed,
    }))
}

/// Human/CLI-facing status snapshot.
async fn status(State(gw): State<Arc<Gateway>>) -> Json<Value> {
    let cfg = gw.state.config.read().await;
    let reg = gw.state.registry.lock().await;
    let meters = gw.state.meters.lock().await;
    let alive = reg.connectable().len();
    Json(json!({
        "node_id": cfg.node_id,
        "role": cfg.role,
        "model": cfg.model.as_ref().map(|m| m.id.clone()),
        "quant": cfg.model.as_ref().map(|m| m.quant.clone()),
        "contribution_percent": cfg.contribution_percent,
        "peer_count": reg.len(),
        "peers_alive": alive,
        "meters": {
            "tokens_served": meters.tokens_served,
            "expert_activations": meters.expert_activations,
            "bytes_relayed": meters.bytes_relayed,
            "uptime_seconds": meters.uptime_seconds,
            "tokens_consumed": meters.tokens_consumed,
            "mines_attempted": meters.mines_attempted,
            "mines_won": gw
                .miner
                .as_ref()
                .map(|m| m.stats.snapshot().1)
                .unwrap_or(0),
        },
        "runtime_url": gw.runtime_url,
    }))
}

/// OpenAI-style model list (mirrors the runtime's list when reachable).
async fn list_models(State(gw): State<Arc<Gateway>>) -> Response {
    let url = format!("{}/v1/models", gw.runtime_url.trim_end_matches('/'));
    let client = reqwest::Client::new();
    match client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => {
            let body: Value = resp.json().await.unwrap_or_else(|_| json!({"data": []}));
            (StatusCode::OK, Json(body)).into_response()
        }
        Ok(resp) => (
            StatusCode::BAD_GATEWAY,
            Json(json!({"error": format!("runtime returned {}", resp.status())})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            Json(json!({"error": format!("local runtime unreachable: {e}")})),
        )
            .into_response(),
    }
}

/// Forward chat completions to the local runtime, metering the request.
/// Streaming responses are passed through byte-for-byte.
async fn chat_completions(
    State(gw): State<Arc<Gateway>>,
    Json(body): Json<Value>,
) -> Response {
    let url = format!(
        "{}/v1/chat/completions",
        gw.runtime_url.trim_end_matches('/')
    );
    let client = reqwest::Client::new();
    let payload = body.to_string();
    let send = client
        .post(&url)
        .header("content-type", "application/json")
        .body(payload)
        .send()
        .await;

    let resp = match send {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({
                    "error": format!("local runtime unreachable: {e}. Is colibri/ollama running?"),
                })),
            )
                .into_response()
        }
    };

    // Meter: Prometheus-style usage accounting when the runtime supplies it,
    // otherwise a conservative byte count as bytes_relayed.
    let is_stream = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(|v| v.contains("text/event-stream"))
        .unwrap_or(false);

    if is_stream {
        // Pass through; metering happens client-side in Phase 1+.
        let status = resp.status();
        let ct = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("text/event-stream")
            .to_string();
        let bytes = resp.bytes().await.unwrap_or_default();
        {
            let mut meters = gw.state.meters.lock().await;
            meters.bytes_relayed += bytes.len() as u64;
        }
        return (
            status,
            [( "content-type", ct.as_str() )],
            bytes,
        )
            .into_response();
    }

    let status = resp.status();
    let bytes = resp.bytes().await.unwrap_or_default();
    {
        let mut meters = gw.state.meters.lock().await;
        meters.bytes_relayed += bytes.len() as u64;
    }
    let parsed: Result<Value, _> = serde_json::from_slice(&bytes);
    if let Ok(body) = &parsed {
        // Tokens metered when usage is present.
        if let Some(usage) = body.get("usage") {
            let total = usage
                .get("total_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let mut meters = gw.state.meters.lock().await;
            meters.tokens_served += total;
        }
        // Expert activations: Phase 2 will count real expert traffic; for now
        // we count one activation per completion as a placeholder hook.
        // Spec: every served inference has a chance to mine JER.
        let fired_mine = gw.miner.as_ref().map(|m| m.maybe_mine()).unwrap_or(false);
        let mut meters = gw.state.meters.lock().await;
        meters.expert_activations += 1;
        if fired_mine {
            meters.mines_attempted += 1;
        }
    }
    (
        status,
        [("content-type", "application/json")],
        bytes,
    )
        .into_response()
}
