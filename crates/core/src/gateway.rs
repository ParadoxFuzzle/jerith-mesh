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
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

use crate::economics::JerithMiner;
use crate::memory::SharedMemory;
use crate::ping::MeshState;

pub struct Gateway {
    pub state: Arc<MeshState>,
    /// Base URL of the local Colibri/Ollama runtime.
    pub runtime_url: String,
    /// Optional JER miner (present when crypto chain = JerithChain).
    pub miner: Option<JerithMiner>,
    /// Local memory store backing the memory.* skills.
    pub memory: Option<SharedMemory>,
    /// Optional mesh-fallback chat handle (QUIC client endpoint + peer
    /// address map). Present when the node runs a server endpoint.
    pub mesh_fallback: Option<Arc<crate::ping::MeshFallbackChat>>,
}

impl Gateway {
    pub fn router(self) -> Router {
        let state = Arc::new(self);
        Router::new()
            .route("/health", get(health))
            .route("/status", get(status))
            .route("/v1/models", get(list_models))
            .route("/v1/chat/completions", post(chat_completions))
            .route(
                "/v1/skills/memory/embed",
                post(skill_memory_embed),
            )
            .route(
                "/v1/skills/memory/search",
                post(skill_memory_search),
            )
            .with_state(state)
    }
}

/// skill: memory.embed — embed + store text locally. Never leaves the node.
async fn skill_memory_embed(
    State(gw): State<Arc<Gateway>>,
    Json(body): Json<Value>,
) -> Response {
    let Some(memory) = &gw.memory else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "memory backend not configured"})),
        )
            .into_response();
    };
    let Some(text) = body.get("text").and_then(|t| t.as_str()) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "field 'text' is required"})),
        )
            .into_response();
    };
    match memory.embed_and_store(text).await {
        Ok((id, provider)) => (
            StatusCode::OK,
            Json(json!({"id": id, "provider": provider})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

/// skill: memory.search — local vector search over the local store.
async fn skill_memory_search(
    State(gw): State<Arc<Gateway>>,
    Json(body): Json<Value>,
) -> Response {
    let Some(memory) = &gw.memory else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "memory backend not configured"})),
        )
            .into_response();
    };
    let Some(query) = body.get("query").and_then(|q| q.as_str()) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "field 'query' is required"})),
        )
            .into_response();
    };
    let top_k = body.get("top_k").and_then(|k| k.as_u64()).unwrap_or(5) as usize;
    match memory.search(query, top_k).await {
        Ok(hits) => (StatusCode::OK, Json(json!({"hits": hits}))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

/// Minimal-resource ping target for other nodes. <1 ms CPU; reports only
/// identity + liveness data, never user data.
///
/// Wire auth: when the request carries signed ping headers they are
/// verified — a bad signature gets a 401 and must not count as liveness.
/// The reply itself is Ed25519-signed so the pinger can attribute it.
async fn health(
    State(gw): State<Arc<Gateway>>,
    headers: HeaderMap,
) -> Response {
    let cfg = gw.state.config.read().await;
    let meters = gw.state.meters.lock().await;

    // Verify the pinger's signature when headers are present.
    let has_auth = headers.contains_key("x-jerith-pubkey");
    if has_auth {
        let from = headers
            .get("x-jerith-from")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_string();
        let pubkey = headers
            .get("x-jerith-pubkey")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_string();
        let stamp = headers
            .get("x-jerith-stamp")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
        let sig = headers
            .get("x-jerith-sig")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_string();
        let payload = crate::signing::ping_payload(&from, &cfg.node_id, stamp);
        let auth = crate::signing::WireAuth {
            pubkey,
            stamp_ms: stamp,
            sig,
        };
        if auth.validate(payload.as_bytes(), &from).is_err() {
            return (
                StatusCode::UNAUTHORIZED,
                Json(json!({"error": "ping signature invalid"})),
            )
                .into_response();
        }
    }

    // Sign our reply so liveness can't be spoofed.
    let stamp = crate::bootstrap::now_ms();
    let payload = crate::signing::health_payload(
        &cfg.node_id,
        stamp,
        meters.uptime_seconds,
        cfg.contribution_percent,
    );
    let sig = gw.state.identity.sign(payload.as_bytes());
    let pubkey = gw.state.identity.pubkey_hex();

    Json(json!({
        "ok": true,
        "node_id": cfg.node_id,
        "pubkey": pubkey,
        "stamp_ms": stamp,
        "sig": sig,
        "model_id": cfg.model.as_ref().map(|m| m.id.clone()),
        "quant": cfg.model.as_ref().map(|m| m.quant.clone()),
        "contribution_percent": cfg.contribution_percent,
        "uptime_seconds": meters.uptime_seconds,
        "tokens_served": meters.tokens_served,
        "expert_activations": meters.expert_activations,
        "bytes_relayed": meters.bytes_relayed,
    }))
    .into_response()
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
        Err(local_err) => {
            // Mesh fallback: local runtime unreachable — try runtime-backed
            // peers over QUIC (lowest RTT first). The request's messages are
            // privacy-scoped by the caller config; we send them verbatim.
            if let Some(mf) = &gw.mesh_fallback {
                let Some(messages) = body.get("messages").and_then(|m| m.as_array()) else {
                    return (
                        StatusCode::BAD_GATEWAY,
                        Json(json!({
                            "error": format!("local runtime unreachable: {local_err} and no messages to forward"),
                            "mesh_fallback": Value::Null,
                        })),
                    )
                        .into_response();
                };
                let state = gw.state.clone();
                let msgs = messages.clone();
                let result = mf
                    .chat_with_fallback(&state, &msgs, || {
                        Err(anyhow::anyhow!("local down: {local_err}"))
                    })
                    .await;
                return match result {
                    Ok((content, served_by)) => {
                        {
                            let mut meters = gw.state.meters.lock().await;
                            meters.tokens_served += 1;
                            meters.expert_activations += 1;
                            meters.bytes_relayed += content.len() as u64;
                        }
                        if served_by.is_some() {
                            tracing::info!(
                                target: "mesh",
                                peer = ?served_by,
                                "chat completion served via mesh fallback"
                            );
                        }
                        // OpenAI-shape reply.
                        (
                            StatusCode::OK,
                            Json(json!({
                                "id": format!("mesh-{}", crate::bootstrap::now_ms()),
                                "object": "chat.completion",
                                "choices": [{
                                    "index": 0,
                                    "message": {"role": "assistant", "content": content},
                                    "finish_reason": "stop"
                                }],
                                "mesh_fallback": served_by,
                            })),
                        )
                            .into_response()
                    }
                    Err(e) => (
                        StatusCode::BAD_GATEWAY,
                        Json(json!({
                            "error": format!(
                                "local runtime unreachable: {local_err}; mesh fallback failed: {e}"
                            ),
                            "mesh_fallback": Value::Null,
                        })),
                    )
                        .into_response(),
                };
            }
            // Bot/API contract: a non-fallback, non-OK reply still carries
            // mesh_fallback (explicitly null) so clients always see the field.
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({
                    "error": format!("local runtime unreachable: {local_err}. Is colibri/ollama running?"),
                    "mesh_fallback": Value::Null,
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

#[cfg(test)]
mod tests {
    use super::*;
    use axum_test::TestServer;

    fn spawn_plain(runtime_url: &'static str, fallback: bool) -> TestServer {
        let mut cfg = crate::config::NodeConfig::default();
        cfg.runtime_url = runtime_url.to_string();
        cfg.ensure_identity().expect("test identity");
        let (state, _rx) = MeshState::new(cfg).expect("test mesh state");
        let gw = Gateway {
            state: state.clone(),
            runtime_url: runtime_url.to_string(),
            miner: None,
            memory: None,
            mesh_fallback: if fallback {
                Some(crate::ping::MeshFallbackChat::spawn().expect("test fallback"))
            } else {
                None
            },
        };
        TestServer::new(gw.router().into_make_service()).expect("test server")
    }

    #[tokio::test]
    async fn chat_without_runtime_or_fallback_is_clean_502() {
        // Nothing listens on :1; runtime dead, no mesh fallback configured.
        let server = spawn_plain("http://127.0.0.1:1", false);
        let res = server
            .post("/v1/chat/completions")
            .json(&json!({
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .await;
        assert_eq!(res.status_code(), axum::http::StatusCode::BAD_GATEWAY);
        let body: Value = res.json();
        assert!(
            body["error"]
                .as_str()
                .unwrap()
                .contains("local runtime unreachable")
        );
        // Contract: the field is always present, explicitly null here.
        assert_eq!(body.get("mesh_fallback"), Some(&Value::Null));
    }

    #[tokio::test]
    async fn chat_with_dead_fallback_peers_also_clean_502() {
        // Fallback configured but registry is empty — must still be a clean
        // 502 with mesh_fallback: null, never a 500.
        let server = spawn_plain("http://127.0.0.1:1", true);
        let res = server
            .post("/v1/chat/completions")
            .json(&json!({
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .await;
        assert_eq!(res.status_code(), axum::http::StatusCode::BAD_GATEWAY);
        let body: Value = res.json();
        assert!(
            body["error"]
                .as_str()
                .unwrap_or("")
                .contains("mesh fallback failed"),
            "body: {body}"
        );
        assert_eq!(body.get("mesh_fallback"), Some(&Value::Null));
    }
}
