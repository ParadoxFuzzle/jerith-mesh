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
use std::time::{Duration, Instant};

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Value, json};

use crate::economics::JerithMiner;

use crate::ping::MeshState;

/// How long the gateway waits for the local runtime's response headers
/// before treating it as unavailable. A runtime that accepts TCP but never
/// replies (hung process, wedged renderer) would otherwise hold the caller
/// open-endedly; past this bound the request flows into mesh fallback
/// exactly like a connection failure. Only the header wait is bounded — a
/// proper SSE body is streamed through without a byte deadline.
const RUNTIME_HEADERS_TIMEOUT: Duration = Duration::from_secs(180);

pub struct Gateway {
    pub state: Arc<MeshState>,
    /// Base URL of the local Colibri/Ollama runtime.
    pub runtime_url: String,
    /// Optional bearer key forwarded to the runtime (authed LiteLLM).
    /// None → runtime calls go out with no Authorization header.
    pub runtime_api_key: Option<String>,
    /// Optional runaway-generation cap for thinking models: when set, chat
    /// forwards to the runtime carry `max_tokens = N` UNLESS the caller set
    /// an explicit cap themselves (max_tokens / max_completion_tokens /
    /// num_predict — caller intent always wins). Motivation: a thinking
    /// model that loops unbounded wedges the chat queue behind it
    /// (deepseek-r1:8b live, 2026-10-14 — >6 min streaming without
    /// terminating on a one-line riddle).
    pub runtime_num_predict: Option<u64>,
    /// Optional JER miner (present when crypto chain = JerithChain).
    pub miner: Option<JerithMiner>,
    /// Optional mesh-fallback chat handle (QUIC client endpoint + peer
    /// address map). Present when the node runs a server endpoint.
    pub mesh_fallback: Option<Arc<crate::ping::MeshFallbackChat>>,
    /// Jerith Guard: outbound content filter screening every chat payload
    /// before it leaves the node (local forward AND mesh fallback).
    pub guard: Option<Arc<crate::guard::Guard>>,
    /// Model routing tiers (vision/deep). None → every request forwards
    /// with the caller's model verbatim.
    pub model_router: Option<Arc<crate::model_router::RouterModels>>,
    /// The advertised base model id (from `/status`); requests carrying
    /// this id — or a placeholder — count as "unpinned" and are eligible
    /// for rerouting to a tier.
    pub base_model: Option<String>,
    /// Web access skills (Brave search / URL fetch). None → disabled in
    /// config; present-but-keyless search answers with a setup hint.
    pub web: Option<Arc<crate::websearch::WebSearchService>>,
    /// Co-hosted JerithChain node RPC (config `chain_url`). Some → the
    /// /status snapshot carries a live `chain` section (read-only probe,
    /// short timeout, null when unreachable).
    pub chain_url: Option<String>,
    /// Perf/cost meters (model-performance-benchmark + session-cost-tracker
    /// ports): per-model latency/success/tokens from live traffic.
    pub metrics: Arc<crate::metrics::Metrics>,
}

impl Gateway {
    pub fn router(self) -> Router {
        let state = Arc::new(self);
        // Per-IP budget over ALL routes. The honest local caller (the bot +
        // web UI) uses a small fraction of this; the ceiling exists because
        // chat forwards a model round-trip and the skill endpoints hit
        // SQLite — one source shouldn't be able to run either hot.
        const GATEWAY_LIMIT_PER_MIN: u32 = 600;
        let limiter = Arc::new(crate::ratelimit::RateLimiter::new(
            GATEWAY_LIMIT_PER_MIN,
            Duration::from_secs(60),
        ));
        Router::new()
            .route("/health", get(health))
            .route("/status", get(status))
            .route("/v1/models", get(list_models))
            .route("/v1/chat/completions", post(chat_completions))
            .route("/v1/skills/memory/embed", post(skill_memory_embed))
            .route("/v1/skills/memory/search", post(skill_memory_search))
            .route("/v1/skills/memory/similar", post(skill_memory_similar))
            .route("/v1/skills/memory/stats", get(skill_memory_stats))
            .route("/v1/skills/lore/search", post(skill_lore_search))
            .route("/v1/skills/web/search", post(skill_web_search))
            .route("/v1/skills/web/fetch", post(skill_web_fetch))
            .route("/v1/skills/vision/describe", post(skill_vision_describe))
            .route("/v1/skills/vision/analyze", post(skill_vision_analyze))
            .route("/v1/metrics", get(metrics_handler))
            .with_state(state)
            .layer(axum::middleware::from_fn_with_state(
                limiter,
                crate::ratelimit::axum_guard,
            ))
    }
}

/// skill: memory.embed — embed + store text locally. Never leaves the node.
/// Input size is bounded: persistence-backed SQLite rows (plus whatever
/// embed provider round-trip the text takes) mean an unbounded POST here
/// would let one caller pin disk + sqlite locks. cap matches the archiver
/// note scale; larger callers must chunk.
const MEMORY_EMBED_TEXT_CAP_CHARS: usize = 64_000;

/// Cap on the serialized /v1/chat/completions payload we forward to the
/// local runtime (and mirror to mesh peers on fallback). The gateway is an
/// unauthenticated local-network surface; without a ceiling a single giant
/// body pins this node's memory, the runtime's, and every mesh relay hop.
/// Honest multimodal+history payloads land far below this; larger callers
/// must chunk.
const CHAT_FORWARD_CAP_BYTES: usize = 1_000_000;

async fn skill_memory_embed(State(gw): State<Arc<Gateway>>, Json(body): Json<Value>) -> Response {
    let Some(memory) = gw.state.memory.read().await.clone() else {
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
    let text = match text.chars().count() {
        n if n <= MEMORY_EMBED_TEXT_CAP_CHARS => text,
        _ => {
            crate::metrics::bump(&crate::metrics::OVERSIZE_MEMORY_REJECTS);
            return (
                StatusCode::PAYLOAD_TOO_LARGE,
                Json(json!({
                    "error": format!(
                        "text exceeds {MEMORY_EMBED_TEXT_CAP_CHARS} chars — chunk large documents"
                    )
                })),
            )
                .into_response();
        }
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
async fn skill_memory_search(State(gw): State<Arc<Gateway>>, Json(body): Json<Value>) -> Response {
    let Some(memory) = gw.state.memory.read().await.clone() else {
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
    // Query cap: the query is embedded via the provider round-trip (nomic
    // context is ~8k tokens anyway) — don't pay an unbounded one per hit.
    const SEARCH_QUERY_CAP_CHARS: usize = 10_000;
    if query.chars().count() > SEARCH_QUERY_CAP_CHARS {
        crate::metrics::bump(&crate::metrics::OVERSIZE_MEMORY_REJECTS);
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(json!({
                "error": format!(
                    "query exceeds {SEARCH_QUERY_CAP_CHARS} chars"
                )
            })),
        )
            .into_response();
    }
    // top_k clamp mirrors memory.similar: the response is fully
    // materialized, so a caller asking for top_k=1e9 would serialize the
    // whole store into one reply.
    let top_k = body
        .get("top_k")
        .and_then(|k| k.as_u64())
        .unwrap_or(5)
        .clamp(1, 25) as usize;
    match memory.search(query, top_k).await {
        Ok(hits) => (StatusCode::OK, Json(json!({"hits": hits}))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

/// skill: memory.similar — semantic-memories port: rank stored entries by
/// similarity to the entry with the given seed id. Body: {id, top_k?}.
async fn skill_memory_similar(State(gw): State<Arc<Gateway>>, Json(body): Json<Value>) -> Response {
    let Some(memory) = gw.state.memory.read().await.clone() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "memory backend not configured"})),
        )
            .into_response();
    };
    let Some(id) = body.get("id").and_then(Value::as_i64) else {
        return bad_request("field 'id' (integer) is required");
    };
    let top_k = body
        .get("top_k")
        .and_then(Value::as_u64)
        .unwrap_or(5)
        .clamp(1, 25) as usize;
    match memory.similar_to(id, top_k).await {
        Ok(hits) => (StatusCode::OK, Json(json!({"hits": hits}))).into_response(),
        Err(e) => (StatusCode::NOT_FOUND, Json(json!({"error": e.to_string()}))).into_response(),
    }
}

/// skill: memory.stats — health-dashboard port: store size/dims/providers.
async fn skill_memory_stats(State(gw): State<Arc<Gateway>>) -> Response {
    let Some(memory) = gw.state.memory.read().await.clone() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "memory backend not configured"})),
        )
            .into_response();
    };
    match memory.stats().await {
        Ok(stats) => (StatusCode::OK, Json(json!(stats))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

/// GET /v1/metrics — per-model perf/cost aggregates (benchmark + cost
/// tracker port): requests, mean/max latency, success %, token totals,
/// mesh attribution, plus the recent-completions ring.
async fn metrics_handler(State(gw): State<Arc<Gateway>>) -> impl IntoResponse {
    let snap = gw.metrics.snapshot();
    // Latency/ok helpers rendered for each model (Serialize keeps the
    // raw fields; these derived views ride alongside).
    let mut models = serde_json::Map::new();
    for (id, st) in &snap.models {
        models.insert(
            id.clone(),
            json!({
                "requests": st.requests,
                "ok": st.ok,
                "failed": st.failed,
                "ok_percent": st.ok_percent(),
                "latency_ms_mean": st.latency_ms_mean(),
                "latency_ms_max": st.latency_ms_max,
                "prompt_tokens": st.prompt_tokens,
                "completion_tokens": st.completion_tokens,
                "total_tokens": st.total_tokens,
                "mesh_served": st.mesh_served,
            }),
        );
    }
    Json(json!({
        "window_started_ms": snap.window_started_ms,
        "requests_total": snap.requests_total,
        "ok_total": snap.ok_total,
        "failed_total": snap.failed_total,
        "tokens_total": snap.tokens_total,
        "models": models,
        "recent": snap.recent,
    }))
}

/// skill: lore.search — void-lore-index port over the configured vault
/// (built on demand, ~/.openclaw borrow). Body: {query, top_k?}.
async fn skill_lore_search(State(_gw): State<Arc<Gateway>>, Json(body): Json<Value>) -> Response {
    let Some(query) = body
        .get("query")
        .or_else(|| body.get("q"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|q| !q.is_empty())
    else {
        return bad_request("field 'query' is required");
    };
    let k = body
        .get("top_k")
        .and_then(Value::as_u64)
        .unwrap_or(4)
        .clamp(1, 10) as usize;
    let Some(vault) = crate::lore::default_vault_dir() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "no lore vault — set $JERITH_LORE_VAULT or the [lore] vault_dir in config.toml"})),
        )
            .into_response();
    };
    let index = match tokio::task::spawn_blocking(move || crate::lore::build(&vault)).await {
        Ok(Ok(idx)) => idx,
        Ok(Err(e)) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": format!("lore index build failed: {e}")})),
            )
                .into_response();
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": format!("lore index task failed: {e}")})),
            )
                .into_response();
        }
    };
    let hits = index.search(query, k);
    (
        StatusCode::OK,
        Json(json!({
            "count": hits.len(),
            "entries": index.len(),
            "built_ms": index.built_ms(),
            "hits": hits,
        })),
    )
        .into_response()
}

fn web_disabled_response() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({
            "error": "web skills disabled — add `[websearch] enabled = true` to config.toml and restart",
        })),
    )
        .into_response()
}

fn bad_request(msg: &str) -> Response {
    (StatusCode::BAD_REQUEST, Json(json!({"error": msg}))).into_response()
}

/// skill: web.search — Brave Search API (web mode), OpenClaw brave-plugin
/// port. Optional body fields: count, country, search_lang, ui_lang,
/// freshness (day|week|month|year).
async fn skill_web_search(State(gw): State<Arc<Gateway>>, Json(body): Json<Value>) -> Response {
    let Some(web) = gw.web.clone() else {
        return web_disabled_response();
    };
    let Some(query) = body
        .get("query")
        .or_else(|| body.get("q"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|q| !q.is_empty())
    else {
        return bad_request("field 'query' is required");
    };
    let sq = crate::websearch::SearchQuery {
        query: query.to_string(),
        count: body.get("count").and_then(Value::as_u64).map(|c| c as u32),
        country: body
            .get("country")
            .and_then(Value::as_str)
            .map(str::to_string),
        search_lang: body
            .get("search_lang")
            .and_then(Value::as_str)
            .map(str::to_string),
        ui_lang: body
            .get("ui_lang")
            .and_then(Value::as_str)
            .map(str::to_string),
        freshness: body
            .get("freshness")
            .and_then(Value::as_str)
            .map(str::to_string),
    };
    match web.search(&sq).await {
        Ok(hits) => (
            StatusCode::OK,
            Json(json!({
                "provider": "brave",
                "mode": "web",
                "count": hits.len(),
                "results": hits,
            })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

/// skill: web.fetch — download a URL and return readable page text
/// (OpenClaw web_fetch port). Optional body field: max_chars.
async fn skill_web_fetch(State(gw): State<Arc<Gateway>>, Json(body): Json<Value>) -> Response {
    let Some(web) = gw.web.clone() else {
        return web_disabled_response();
    };
    let Some(url) = body
        .get("url")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|u| !u.is_empty())
    else {
        return bad_request("field 'url' is required");
    };
    let max_chars = body
        .get("max_chars")
        .and_then(Value::as_u64)
        .map(|c| c as usize)
        .filter(|c| *c > 0);
    match web.fetch_text(url, max_chars).await {
        Ok(page) => (StatusCode::OK, Json(json!(page))).into_response(),
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            Json(json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

/// skill: vision.describe — describe an image using the local vision model
async fn skill_vision_describe(
    State(gw): State<Arc<Gateway>>,
    Json(body): Json<Value>,
) -> Response {
    let Some(image) = body
        .get("image")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
    else {
        return bad_request("field 'image' is required");
    };
    let prompt = body
        .get("prompt")
        .and_then(|v| v.as_str())
        .unwrap_or("Describe this image")
        .to_string();
    vision_call(&gw, &prompt, &image).await
}

/// skill: vision.analyze — analyze an image with a custom prompt
async fn skill_vision_analyze(State(gw): State<Arc<Gateway>>, Json(body): Json<Value>) -> Response {
    let Some(image) = body
        .get("image")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
    else {
        return bad_request("field 'image' is required");
    };
    let Some(prompt) = body
        .get("prompt")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
    else {
        return bad_request("field 'prompt' is required");
    };
    vision_call(&gw, &prompt, &image).await
}

async fn vision_call(gw: &Arc<Gateway>, prompt: &str, image: &str) -> Response {
    let url = format!(
        "{}/v1/chat/completions",
        gw.runtime_url.trim_end_matches('/')
    );
    let client = reqwest::Client::builder()
        .use_rustls_tls()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(300))
        .build()
        .expect("vision client builder");
    let messages = json!([
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image}}
            ]
        }
    ]);
    let mut req_body = json!({
        "model": gw.base_model.as_deref().unwrap_or(""),
        "messages": messages,
        "max_tokens": 512
    });
    if let (Some(router), Some(msgs)) = (
        &gw.model_router,
        req_body.get("messages").and_then(|m| m.as_array()),
    ) {
        let requested = req_body.get("model").and_then(|m| m.as_str());
        // The vision skill never carries function definitions — no tool tier.
        if let Some(next) = crate::model_router::reroute_for_gateway(
            requested,
            gw.base_model.as_deref(),
            msgs,
            router,
            false,
        ) {
            req_body["model"] = serde_json::Value::String(next);
        }
    }
    let mut req = client.post(&url).json(&req_body);
    if let Some(k) = &gw.runtime_api_key {
        req = req.bearer_auth(k);
    }
    match req.send().await {
        Ok(resp) => {
            let status =
                StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            let body: Value = resp
                .json()
                .await
                .unwrap_or_else(|e| json!({"error": e.to_string()}));
            (status, Json(body)).into_response()
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
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
async fn health(State(gw): State<Arc<Gateway>>, headers: HeaderMap) -> Response {
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

    // Real-time measured contribution (FEATURES §2): the last COMPLETED
    // ping interval, signed with the SAME identity + stamp as the reply —
    // the pinger recomputes both signatures field-for-field, so a node
    // owns every number it publishes and cannot edit them in flight.
    let measurement = {
        let contrib = gw.state.contrib.lock().await;
        contrib.measurement_point().map(|(interval, measured)| {
            let payload = crate::signing::measurement_payload(
                &cfg.node_id,
                stamp,
                interval.window_secs,
                interval.tokens_served,
                interval.tokens_consumed,
                cfg.contribution_percent,
                measured,
            );
            Some(json!({
                "window_secs": interval.window_secs,
                "tokens_served": interval.tokens_served,
                "tokens_consumed": interval.tokens_consumed,
                "declared_percent": cfg.contribution_percent,
                "measured_percent": measured,
                "sig": gw.state.identity.sign(payload.as_bytes()),
            }))
        })
    };

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
        "measurement": measurement,
    }))
    .into_response()
}

/// Human/CLI-facing status snapshot.
async fn status(State(gw): State<Arc<Gateway>>) -> axum::response::Response {
    let cfg = gw.state.config.read().await;
    let reg = gw.state.registry.lock().await;
    let meters = gw.state.meters.lock().await;
    let alive = reg.connectable().len();
    // Hoisted: json! cannot host statement blocks.
    let verify = gw.state.verify.stats().await;
    let verify_value = json!({
        "expert_results_received": verify.expert_results_received,
        "commitment_checks": verify.commitment_checks,
        "commitment_failures": verify.commitment_failures,
        "replays_dispatched": verify.replays_dispatched,
        "replay_matches": verify.replay_matches,
        "replay_mismatches": verify.replay_mismatches,
        "fraud_reports_flushed": verify.fraud_reports_flushed,
    });
    let stake = gw.state.stake.snapshot();
    let stake_value = json!({
        "required_uj": gw.state.stake.required_uj(),
        "staked_uj": stake.staked_uj,
        "slashed_total_uj": stake.slashed_total_uj,
        "slash_events": stake.slash_events.len(),
        "gate_clear": gw.state.stake.clears_gate(),
    });
    let guard_value = gw
        .guard
        .as_ref()
        .map(|g| {
            let (screens, blocked, redacted, logged) = g.stats().snapshot();
            json!({
                "enabled": true,
                "screens": screens,
                "requests_blocked": blocked,
                "values_redacted": redacted,
                "detections_logged": logged,
            })
        })
        .unwrap_or(Value::Null);
    // Fleet resource census (resource pooling, Phase 1): our latest sample
    // + whatever connectable peers advertised inside their PeerAd. Unknown
    // amounts are excluded from sums (unknown ≠ zero — same rule as the
    // placement ordering).
    let own_sample = gw.state.resources.lock().await.clone();
    let mut samples: Vec<(String, crate::hw::ResourceSample)> = Vec::new();
    if let Some(own) = &own_sample {
        samples.push((cfg.node_id.clone(), own.clone()));
    }
    for (ad, _) in reg.connectable() {
        if let Some(res) = &ad.resources {
            samples.push((ad.node_id.clone(), res.clone()));
        }
    }
    let ram_total_bytes: u64 = samples.iter().map(|(_, s)| s.ram_total_bytes).sum();
    let ram_available_bytes: u64 = samples.iter().map(|(_, s)| s.ram_available_bytes).sum();
    let vram_free_known: Vec<u64> = samples
        .iter()
        .filter_map(|(_, s)| s.vram_free_total())
        .collect();
    let vram_total_known: Vec<u64> = samples
        .iter()
        .filter_map(|(_, s)| s.vram_total_total())
        .collect();
    let fleet_value = json!({
        "reporting_nodes": samples.len(),
        "ram_total_bytes": ram_total_bytes,
        "ram_available_bytes": ram_available_bytes,
        "vram_free_bytes": vram_free_known.first().map(|_| vram_free_known.iter().sum::<u64>()),
        "vram_total_bytes": vram_total_known.first().map(|_| vram_total_known.iter().sum::<u64>()),
        "nodes": samples
            .iter()
            .map(|(id, s)| json!({"node_id": id, "census": s.census()}))
            .collect::<Vec<_>>(),
    });
    // Phase 2 tensor transport: what this node serves from its expert
    // store, and how the on-demand cache is doing (hosted / resident /
    // hit-miss-load-eviction counters). Null when no store is attached.
    let experts_value = match gw.state.expert_store().await {
        Some(store) => {
            let s = store.stats();
            let hosted = store.hosted();
            // Phase 2 replication: the demand signal behind the hot/cold
            // split, and the shard ring's owner for the cold tail — so an
            // operator can see *why* this node holds what it holds.
            let stats = gw.state.expert_stats_snapshot();
            let percent = cfg.experts.hot_percent;
            let hot = stats.hot_set(percent);
            let cold = stats.cold_set(percent);
            let ring = crate::replication::ShardMap::from_ads(&cfg.node_id, &reg.all_ads());
            // Bounded sample: the tail can be long and /status is polled.
            let cold_placements: Vec<Value> = cold
                .iter()
                .take(16)
                .map(|id| json!({"expert_id": id, "cold_shard_owner": ring.primary_owner(id)}))
                .collect();
            let policy = cfg.experts.replication_config();
            json!({
                "dir": store.dir().display().to_string(),
                "hosted": hosted.len(),
                "hosted_ids": hosted,
                "cache_budget_bytes": store.cache_budget_bytes(),
                "resident_bytes": store.resident_bytes(),
                "hits": s.hits,
                "misses": s.misses,
                "loads": s.loads,
                "evictions": s.evictions,
                "load_errors": s.load_errors,
                // Demand signal + placement policy (Phase 2 replication).
                "activations_total": stats.total(),
                "activations_distinct": stats.distinct(),
                "counts": stats.snapshot()["counts"],
                "hot_percent": percent,
                "hot": hot,
                "cold": cold.len(),
                "cold_placements": cold_placements,
                "shard_ring_nodes": ring.members().len(),
                "replication": {
                    "enabled": cfg.experts.replication_enabled,
                    "holders_wanted": policy.replication_factor,
                    "interval_seconds": cfg.experts.replication_interval_secs,
                    "max_fetches_per_round": policy.max_actions_per_round,
                },
            })
        }
        None => Value::Null,
    };
    // Phase 1 resource isolation: the ceilings that arm the governor and
    // its live kill-switch state (policy lives in isolate.rs; this is
    // truth — snapshot() re-reads the switch at render time, so the
    // no-store cache header above matters here too).
    // Phase 2 cost estimation: bytes/min × 43 200 min/month → GB projection.
    let bw_rate = gw.state.isolate.bandwidth_rate_min();
    let bw_gib_month =
        bw_rate.map(|r| ((r as f64) * 60.0 * 24.0 * 30.0) / (1024.0 * 1024.0 * 1024.0));
    let resources_value = json!({
        "ram_ceiling_bytes": cfg.resources.ram_ceiling_bytes,
        "vram_ceiling_bytes": cfg.resources.vram_ceiling_bytes,
        "pin_cores": cfg.resources.pin_cores,
        "kill_switch": cfg.resources.kill_switch,
        "switch": crate::isolate::snapshot(&gw.state.isolate),
        "node_rss_bytes": crate::isolate::self_rss_bytes(),
        // Bandwidth caps (Phase 2): ceiling + the governor's live rate (0
        // while the window warms up) + cost estimation (GB/month).
        "bandwidth_ceiling_bytes_per_min": cfg.resources.bandwidth_ceiling_bytes_per_min,
        "bandwidth_rate_min": bw_rate,
        "bandwidth_est_gib_month": bw_gib_month,
    });
    // Phase 3 per-peer blocking: the list + how many refused RPCs it
    // yielded. Live truth from the shared BlockedPeers (hot-reloaded by
    // the ping tick, so edits show up here within one interval).
    let peer_block_value = json!({
        "blocked": gw.state.peer_block.list(),
        "refused_rpcs": gw.state.peer_block.refusals(),
    });
    // Discovery rotation (Phase 1): the live server list — config first,
    // learned-from-peers behind it — plus last-good and the ad payload,
    // so an operator can see WHERE discovery is failing when it does.
    let discovery_value = {
        let d = gw.state.discovery.lock().await;
        json!({
            "configured": d.configured,
            "learned": d.learned,
            "last_good": d.last_good,
            "ad_servers": d.ad_servers(None),
        })
    };
    // Real-time contribution measurement (FEATURES §2): declared vs
    // MEASURED serve share side by side, plus the gate verdict the watch
    // signal carries. The signed per-interval numbers ride /health for
    // peers to verify; this section is the operator's view.
    let contribution_value = {
        let contrib = gw.state.contrib.lock().await;
        let mut v = json!({
            "declared_percent": cfg.contribution_percent,
            "healthy": *gw.state.contribution_healthy.borrow(),
        });
        if let Some(obj) = v.as_object_mut()
            && let Value::Object(measured) = contrib.status_json()
        {
            for (k, val) in measured {
                obj.insert(k, val);
            }
        }
        v
    };
    // Statistical freerider detection (Phase 4): running coordinator-side
    // suspicion (decayed scores, evidence flags) — statistics, not verdicts;
    // the bootstrap weighs what we queue. Null when the phase is off.
    let freerider_value = if cfg.freerider_enabled {
        let d = gw.state.freerider.lock().await;
        d.snapshot()
    } else {
        Value::Null
    };
    // The store actually serving memory.* right now — live truth, unlike
    // config labels: the config hot-reload swaps it without a restart
    // (or keeps serving the previous store when the new one won't open).
    let memory_store_value = gw.state.memory.read().await.as_ref().map(|m| m.describe());
    let value = json!({
        "node_id": cfg.node_id,
        "role": cfg.role,
        "model": cfg.model.as_ref().map(|m| m.id.clone()),
        "quant": cfg.model.as_ref().map(|m| m.quant.clone()),
        "contribution_percent": cfg.contribution_percent,
        // Real-time contribution (FEATURES §2): measured share + gate.
        "contribution": contribution_value,
        // Local memory (FEATURES §6): the backend actually live-served.
        "memory_store": memory_store_value,
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
        // Runaway-generation cap state (armed via config `runtime_num_predict`;
        // the site dashboard surfaces it so an operator can see it live).
        "runtime_num_predict": gw.runtime_num_predict,
        // Resource pooling (Phase 1): live fleet census.
        "fleet_resources": fleet_value,
        // Resource isolation (Phase 1): ceilings + kill-switch state.
        "resources": resources_value,
        // Per-peer blocking (Phase 3): refused ids + refusal counter.
        "peer_block": peer_block_value,
        // Discovery rotation (Phase 1): configured + learned + last-good.
        "discovery": discovery_value,
        // Statistical freerider detection (Phase 4): running suspicion.
        "freerider": freerider_value,
        // Tensor transport (Phase 2): expert store state + cache counters.
        "experts": experts_value,
        "verify": verify_value,
        "guard": guard_value,
        "stake": stake_value,
        "web": json!({
            "enabled": gw.web.is_some(),
            "provider": "brave",
            "api_key_present": gw.web.as_ref().map(|w| w.has_api_key()).unwrap_or(false),
        }),
        "metrics": metrics_status(&gw.metrics.snapshot()),
        "security": serde_json::to_value(crate::metrics::security_snapshot())
            .expect("security counters serialize"),
        "chain": chain_probe(gw.chain_url.as_deref(), crate::economics::read_token().as_deref()).await,
    });
    // Live counters: a heuristically cached stale snapshot would make the
    // web dashboard (and any monitor) read lagged values.
    ([(header::CACHE_CONTROL, "no-store")], Json(value)).into_response()
}

/// Compact per-model view for the /status `metrics` section (full detail
/// lives on /v1/metrics).
fn metrics_status(snap: &crate::metrics::MetricsSnapshot) -> Value {
    let mut models = serde_json::Map::new();
    for (id, st) in &snap.models {
        models.insert(
            id.clone(),
            json!({
                "requests": st.requests,
                "ok_percent": st.ok_percent(),
                "latency_ms_mean": st.latency_ms_mean(),
                "total_tokens": st.total_tokens,
                "mesh_served": st.mesh_served,
            }),
        );
    }
    json!({
        "requests_total": snap.requests_total,
        "ok_total": snap.ok_total,
        "failed_total": snap.failed_total,
        "tokens_total": snap.tokens_total,
        "models": models,
    })
}

/// Live probe of the co-hosted JerithChain node's `/status` (read-only).
/// Every failure path degrades to Null — a downed chain must never take
/// the mesh status page with it.
async fn chain_probe(base: Option<&str>, token: Option<&str>) -> Value {
    let (Some(base), Some(token)) = (base, token) else {
        return Value::Null;
    };
    let base = base.trim().trim_end_matches('/');
    if base.is_empty() || token.trim().is_empty() {
        return Value::Null;
    }
    let url = format!("{base}/status");
    let resp = reqwest::Client::builder()
        .use_rustls_tls()
        .timeout(Duration::from_secs(3))
        .build()
        .ok()
        .map(|c| c.get(&url).bearer_auth(token.trim()).send());
    let Some(resp) = resp else { return Value::Null };
    match resp.await {
        Ok(r) => match r.json::<Value>().await {
            Ok(body) => json!({
                "reachable": true,
                "url": base,
                "height": body.pointer("/chain/height"),
                "tip_hash": body.pointer("/chain/tip_hash"),
                "circulating_jer": body.pointer("/chain/circulating_jer"),
                "max_supply_jer": body.pointer("/chain/max_supply_jer"),
                "wallets": body.pointer("/wallets"),
                "miners": body.pointer("/mining/miners"),
                "total_mined_uj": body.pointer("/mining/total_mined_uj"),
            }),
            Err(_) => json!({"reachable": false, "url": base, "error": "non-JSON chain response"}),
        },
        Err(e) => json!({"reachable": false, "url": base, "error": e.to_string()}),
    }
}

/// OpenAI-style model list (mirrors the runtime's list when reachable).
async fn list_models(State(gw): State<Arc<Gateway>>) -> Response {
    let url = format!("{}/v1/models", gw.runtime_url.trim_end_matches('/'));
    let client = crate::http::mesh_http();
    let mut req = client.get(&url);
    if let Some(k) = &gw.runtime_api_key {
        req = req.bearer_auth(k);
    }
    match req.send().await {
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

/// A guard block response (400) carrying the whole-detector report.
fn guard_block_response(verdict: crate::guard::GuardVerdict) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({
            "error": verdict.block_reason(),
            "guard": {
                "blocked": true,
                "detections": verdict.detections,
            },
            "mesh_fallback": Value::Null,
        })),
    )
        .into_response()
}

/// True when the chat runtime is this very machine (loopback/localhost):
/// forwarding there is not egress, so Jerith Guard's leave-the-node
/// screening is pointless there — agent tool-loop payloads carry
/// timestamps, scores and scores-as-floats that PII heuristics
/// false-positive on. The mesh-fallback edge screens instead (see
/// chat_completions), so peer dispatch is never unscreened.
fn runtime_is_loopback(runtime_url: &str) -> bool {
    let host = reqwest::Url::parse(runtime_url)
        .ok()
        .and_then(|u| u.host_str().map(str::to_string))
        .unwrap_or_default()
        .trim_matches(['[', ']'])
        .to_string();
    host == "localhost" || host == "::1" || host.starts_with("127.")
}

/// Forward chat completions to the local runtime, metering the request.
/// Streaming responses are passed through byte-for-byte.
async fn chat_completions(State(gw): State<Arc<Gateway>>, Json(mut body): Json<Value>) -> Response {
    // Jerith Guard screens at EGRESS, not at intake: content only needs
    // screening when something actually leaves the node. A loopback
    // runtime forward is not egress (the payload never leaves the
    // machine — and agent tool-loop payloads are full of numbers and
    // scores that false-positive the PII heuristics), so the screen
    // happens later, at the mesh-fallback edge, if peer dispatch ever
    // engages. Non-loopback runtimes keep the pre-forward screen.
    let loopback_runtime = runtime_is_loopback(&gw.runtime_url);
    if !loopback_runtime
        && let Some(guard) = &gw.guard
        && let Some(messages) = body.get("messages").and_then(|m| m.as_array())
    {
        match guard.screen_messages(messages) {
            Ok(Some(patched)) => {
                body["messages"] = Value::Array(patched);
                tracing::debug!(target: "guard", "outbound payload redacted before forwarding");
            }
            Ok(None) => {} // clean payload, pass touch-free
            Err(verdict) => return guard_block_response(verdict),
        }
    }
    let url = format!(
        "{}/v1/chat/completions",
        gw.runtime_url.trim_end_matches('/')
    );
    let request_started = Instant::now();
    let requested_model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    // Model routing: tool loops → tool tier (parse-reliable caller);
    // image content → vision tier; complex prompt → deep tier; basic
    // chat → caller's model untouched. Only *unpinned* requests
    // (placeholder / advertised base model) are reroutable, so
    // explicit model choices always win. On a tool-tier reroute the
    // payload also gets `think: false`: thinking models burn the whole
    // generation budget inside their reasoning channel before emitting
    // any tool call or text, and ollama then reports `content: ""` with
    // finish_reason stop at exactly the cap — the "runtime returned an
    // empty answer" family. Disabling thinking on the parse-reliability
    // tier (only) keeps the budget for actual calls; deep-tier and
    // pinned requests keep their thinking untouched. ollama honors the
    // field on /v1; LiteLLM drops unknown params (drop_params), so the
    // merge is inert everywhere else.
    if let (Some(router), Some(messages)) = (
        &gw.model_router,
        body.get("messages").and_then(|m| m.as_array()),
    ) {
        let requested = body.get("model").and_then(|m| m.as_str());
        let has_tools = crate::model_router::payload_has_tools(&body);
        if let Some(next) = crate::model_router::reroute_for_gateway(
            requested,
            gw.base_model.as_deref(),
            messages,
            router,
            has_tools,
        ) {
            tracing::info!(
                from = requested.unwrap_or("(none)"),
                to = %next,
                "model router: rerouted chat request"
            );
            if !router.tool.is_empty() && next == router.tool {
                body["think"] = serde_json::Value::Bool(false);
            }
            body["model"] = serde_json::Value::String(next);
        }
    }
    // A model-less BASIC chat (no tier engaged) would forward `model: null`
    // to the runtime — LiteLLM rejects that with 400 "Invalid model name
    // passed in model=None" (observed 2026-10-16 via a plain probe without
    // the field). Default to the advertised base model — the same id
    // /status serves — so any OpenAI-compatible client can omit `model`.
    if body
        .get("model")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .map(str::is_empty)
        .unwrap_or(true)
        && let Some(base) = gw
            .base_model
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
    {
        body["model"] = serde_json::Value::String(base.to_string());
    }
    // Runaway cap: bound generation for the runtime lane when configured
    // and the caller didn't express their own cap (see field docs).
    if let Some(cap) = gw.runtime_num_predict {
        let caller_capped = ["max_tokens", "max_completion_tokens", "num_predict"]
            .iter()
            .any(|k| body.get(*k).map(|v| !v.is_null()).unwrap_or(false));
        if !caller_capped {
            body["max_tokens"] = serde_json::json!(cap);
        }
    }
    let client = reqwest::Client::builder()
        .use_rustls_tls()
        // Distinguishes "host down" from "host up but wedged": a dead route
        // fails fast here, while a slow/hung runtime is caught by the
        // header-wait bound below instead.
        .connect_timeout(Duration::from_secs(10))
        .build()
        .expect("gateway runtime client builder");
    let payload = body.to_string();
    if payload.len() > CHAT_FORWARD_CAP_BYTES {
        crate::metrics::bump(&crate::metrics::OVERSIZE_CHAT_REJECTS);
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(json!({
                "error": format!(
                    "chat payload exceeds {CHAT_FORWARD_CAP_BYTES} bytes — chunk large conversations"
                ),
                "mesh_fallback": serde_json::Value::Null,
            })),
        )
            .into_response();
    }
    let mut send = client
        .post(&url)
        .header("content-type", "application/json")
        .body(payload);
    if let Some(k) = &gw.runtime_api_key {
        send = send.bearer_auth(k);
    }
    // Bound only the wait for response headers: on expiry the hung runtime
    // is treated as unreachable so mesh fallback can engage (see the match
    // arm below). Headers arrive => body/SSE streams unbounded.
    let send: Result<reqwest::Response, anyhow::Error> =
        match tokio::time::timeout(RUNTIME_HEADERS_TIMEOUT, send.send()).await {
            Ok(result) => result.map_err(anyhow::Error::from),
            Err(_) => Err(anyhow::anyhow!(
                "timed out after {}s waiting for runtime response headers (runtime hung?)",
                RUNTIME_HEADERS_TIMEOUT.as_secs()
            )),
        };

    let resp = match send {
        Ok(r) => r,
        Err(local_err) => {
            // Mesh fallback: local runtime unreachable — try runtime-backed
            // peers over QUIC (lowest RTT first). The request's messages are
            // privacy-scoped by the caller config; we send them verbatim.
            if let Some(mf) = &gw.mesh_fallback {
                // Phase 1 kill switch: mesh chat fallback is exactly the
                // "borrowed capacity" that ceilings bound. Tripped → reply
                // 502 with mesh_fallback: null (the field is never missing,
                // so clients see an honest refusal, not a hang).
                if gw.state.isolate.tripped() {
                    return (
                        StatusCode::BAD_GATEWAY,
                        Json(json!({
                            "error": format!(
                                "local runtime unreachable: {local_err}; mesh fallback refused — kill switch tripped ({})",
                                gw.state
                                    .isolate
                                    .tripped_reason()
                                    .unwrap_or_else(|| "resource ceiling breached".into())
                            ),
                            "mesh_fallback": Value::Null,
                        })),
                    )
                        .into_response();
                }
                // Egress edge — loopback-runtime lane: the pre-forward
                // screen was skipped at the top because a loopback forward
                // never leaves the node. Peer dispatch DOES leave it, so
                // screen here; fallback can never bypass Jerith Guard.
                // (Runs before the messages extraction: a redaction patch
                // must be on the array the dispatch actually sends.)
                if loopback_runtime
                    && let Some(guard) = &gw.guard
                    && let Some(msgs_to_screen) = body.get("messages").and_then(|m| m.as_array())
                {
                    match guard.screen_messages(msgs_to_screen) {
                        Ok(Some(patched)) => {
                            body["messages"] = Value::Array(patched);
                        }
                        Ok(None) => {} // clean payload
                        Err(verdict) => return guard_block_response(verdict),
                    }
                }
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
                // Streaming + dead local runtime: peers serve non-streaming;
                // wrap their single reply in a minimal SSE chunk shape so
                // stream clients work too, attributed via mesh_fallback.
                if body
                    .get("stream")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
                {
                    let state = gw.state.clone();
                    let msgs = messages.clone();
                    let result = mf
                        .chat_with_fallback(&state, &msgs, || {
                            Err(anyhow::anyhow!("local down: {local_err}"))
                        })
                        .await;
                    return match result {
                        Ok((content, served_by)) => {
                            let content = sanitize_runtime_text(&content);
                            {
                                let mut meters = gw.state.meters.lock().await;
                                meters.tokens_served += 1;
                                meters.expert_activations += 1;
                                meters.bytes_relayed += content.len() as u64;
                            }
                            // Perf meter: mesh-fallback stream wrap.
                            gw.metrics.record_simple(
                                &requested_model,
                                request_started,
                                200,
                                None,
                                served_by.as_deref(),
                            );
                            let id = format!("mesh-{}", crate::bootstrap::now_ms());
                            let chunk = json!({
                                "id": id,
                                "object": "chat.completion.chunk",
                                "choices": [{
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": content},
                                    "finish_reason": "stop"
                                }],
                                "mesh_fallback": served_by,
                            });
                            let sse = format!(
                                "data: {}\n\ndata: [DONE]\n\n",
                                serde_json::to_string(&chunk).unwrap_or_default()
                            );
                            (
                                StatusCode::OK,
                                [("content-type", "text/event-stream")],
                                sse,
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
                let state = gw.state.clone();
                let msgs = messages.clone();
                let result = mf
                    .chat_with_fallback(&state, &msgs, || {
                        Err(anyhow::anyhow!("local down: {local_err}"))
                    })
                    .await;
                return match result {
                    Ok((content, served_by)) => {
                        let content = sanitize_runtime_text(&content);
                        {
                            let mut meters = gw.state.meters.lock().await;
                            meters.tokens_served += 1;
                            meters.expert_activations += 1;
                            meters.bytes_relayed += content.len() as u64;
                        }
                        // Perf meter: mesh-fallback completion.
                        gw.metrics.record_simple(
                            &requested_model,
                            request_started,
                            200,
                            None,
                            served_by.as_deref(),
                        );
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
                .into_response();
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
        // Pass through CHUNK-BY-CHUNK: the old `resp.bytes().await` here
        // buffered the whole generation before sending a byte, silently
        // turning every stream client into a non-stream caller. SSE must
        // arrive as the runtime produces it — that is the entire point of
        // stream:true (bot live-typing reads it, 2026-10-14). Chunk
        // metering moves client-side along with the old Phase-1 note.
        let status = resp.status();
        let ct = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("text/event-stream")
            .to_string();
        return (
            status,
            [("content-type", ct.as_str())],
            axum::body::Body::from_stream(resp.bytes_stream()),
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
    // Display hygiene (buffered lane): some runtime models print HTML
    // break tags and reasoning-trace markers straight into visible text
    // (observed verbatim from deepseek-r1 via ollama: `<br>`, `</think>`).
    // Strip the TAGS, keep the words — artifact removal, not censorship.
    // The SSE passthrough above cannot do this reliably: a tag can straddle
    // a chunk boundary (~"<br" + ">"), so streamed clients get the raw
    // generation; the bot folds stream rounds before any final render and
    // can sanitize there in a follow-up.
    let bytes = match parsed {
        Ok(mut body) => {
            sanitize_choices(&mut body);
            serde_json::to_vec(&body)
                .map(Into::into)
                .unwrap_or_else(|_| bytes)
        }
        Err(_) => bytes,
    };
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
        // Perf/cost meters: one record per completion with the model the
        // request actually carried (post-routing) plus its usage.
        let effective_model = body
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or(&requested_model)
            .to_string();
        gw.metrics.record_simple(
            &effective_model,
            request_started,
            status.as_u16(),
            body.get("usage"),
            None,
        );
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
    (status, [("content-type", "application/json")], bytes).into_response()
}

/// Visual noise some runtime models print into otherwise-good text: HTML
/// break tags (which render as literal debris in chat UIs — what the model
/// meant was a line break) and reasoning-trace markers leaked outside the
/// dedicated field (observed verbatim: deepseek-r1 inlining `</think>` in
/// content). Tags go, words stay.
/// Visual noise some runtime models print into otherwise-good text: HTML
/// break tags (which render as literal debris in chat UIs — what the model
/// meant was a line break) and reasoning-trace markers leaked into content
/// (observed verbatim: deepseek-r1 inlining a close marker after prose).
/// Tags go, words stay. The scanner works token-wise rather than matching
/// whole tag literals: the markers arrive with stray whitespace variants
/// and the marker glyphs are risky to transcribe into source — the scanner
/// is robust to both.
fn sanitize_runtime_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.find('<') {
        out.push_str(&rest[..open]);
        let after_open = &rest[open + 1..];
        match after_open.find('>') {
            Some(close) => {
                let inner = &after_open[..close];
                let word: String = inner
                    .chars()
                    .filter(|c| !matches!(c, ' ' | '/' | '\t'))
                    .collect::<String>()
                    .to_lowercase();
                match word.as_str() {
                    "think" => {} // reasoning-trace marker: gone, words kept
                    "br" => out.push('\n'),
                    // Unknown tag: not our display-hygiene business — keep it.
                    _ => {
                        out.push('<');
                        out.push_str(inner);
                        out.push('>');
                    }
                }
                rest = &after_open[close + 1..];
            }
            None => {
                // Unterminated '<': not a tag, emit the tail as-is.
                out.push_str(&rest[open..]);
                rest = "";
            }
        }
    }
    out.push_str(rest);
    out.trim().to_string()
}

/// Sanitize every choice's assistant text content of a chat-completion
/// body, in place. Non-string content (null tool-call rounds, multimodal
/// arrays) is left untouched.
fn sanitize_choices(body: &mut Value) {
    let Some(choices) = body.get_mut("choices").and_then(Value::as_array_mut) else {
        return;
    };
    for choice in choices {
        let Some(message) = choice.get_mut("message") else {
            continue;
        };
        let Some(content) = message
            .get("content")
            .and_then(Value::as_str)
            .map(str::to_string)
        else {
            continue;
        };
        let cleaned = sanitize_runtime_text(&content);
        if cleaned != content {
            message["content"] = Value::String(cleaned);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum_test::TestServer;

    #[test]
    fn loopback_detection_covers_localhost_forms_and_refuses_remote() {
        for url in [
            "http://127.0.0.1:11434",
            "http://localhost:8900",
            "http://[::1]:8900",
            "http://127.0.0.1:4000/v1",
            "https://localhost/v1/chat/completions",
        ] {
            assert!(runtime_is_loopback(url), "{url} should count as loopback");
        }
        for url in [
            "https://api.openai.com/v1",
            "http://10.0.0.5:11434",
            "http://172.127.0.1:4000", // starts with 172, NOT the loopback /8
            "jerith-mesh",
        ] {
            assert!(
                !runtime_is_loopback(url),
                "{url} must NOT count as loopback"
            );
        }
    }

    fn spawn_plain(runtime_url: &'static str, fallback: bool) -> TestServer {
        let mut cfg = crate::config::NodeConfig {
            runtime_url: runtime_url.to_string(),
            ..crate::config::NodeConfig::default()
        };
        cfg.ensure_identity().expect("test identity");
        let (state, _rx) = MeshState::new(cfg).expect("test mesh state");
        let gw = Gateway {
            state: state.clone(),
            runtime_url: runtime_url.to_string(),
            runtime_api_key: None,
            runtime_num_predict: None,
            miner: None,
            mesh_fallback: if fallback {
                Some(crate::ping::MeshFallbackChat::spawn().expect("test fallback"))
            } else {
                None
            },
            guard: None,
            model_router: None,
            base_model: None,
            web: None,
            chain_url: None,
            metrics: Arc::new(crate::metrics::Metrics::new()),
        };
        TestServer::new(gw.router().into_make_service()).expect("test server")
    }

    /// Test server with the local memory backend armed (embed falls back
    /// to hashing when no ollama answers).
    fn spawn_with_memory() -> TestServer {
        let mut cfg = crate::config::NodeConfig {
            runtime_url: "http://127.0.0.1:1".to_string(),
            ..crate::config::NodeConfig::default()
        };
        cfg.ensure_identity().expect("test identity");
        let (state, _rx) = MeshState::new(cfg).expect("test mesh state");
        let memory = crate::memory::MemoryStore::open_in_memory(crate::memory::EmbedConfig {
            dim: 64,
            ..crate::memory::EmbedConfig::default()
        })
        .expect("test memory db");
        let memory = crate::memory::MemoryHandle::Sqlite(memory);
        // MeshState owns the live store (hot-reloadable); boot swaps it in
        // the same way. try_write is fine here: no readers exist yet.
        state
            .memory
            .try_write()
            .expect("no memory reader at test boot")
            .replace(std::sync::Arc::new(memory));
        let gw = Gateway {
            state,
            runtime_url: "http://127.0.0.1:1".to_string(),
            runtime_api_key: None,
            runtime_num_predict: None,
            miner: None,
            mesh_fallback: None,
            guard: None,
            model_router: None,
            base_model: None,
            web: None,
            chain_url: None,
            metrics: Arc::new(crate::metrics::Metrics::new()),
        };
        TestServer::new(gw.router().into_make_service()).expect("test server")
    }

    /// Test server with the default-configured Jerith Guard armed;
    /// `fallback` arms the mesh-fallback lane (the egress-edge tests).
    fn spawn_guarded(runtime_url: String, fallback: bool) -> TestServer {
        let runtime_url: &'static str = Box::leak(runtime_url.into_boxed_str());
        let mut cfg = crate::config::NodeConfig {
            runtime_url: runtime_url.to_string(),
            ..crate::config::NodeConfig::default()
        };
        cfg.ensure_identity().expect("test identity");
        let (state, _rx) = MeshState::new(cfg).expect("test mesh state");
        let guard = crate::guard::Guard::build(&crate::guard::GuardConfig::default())
            .expect("default guard config compiles");
        let gw = Gateway {
            state: state.clone(),
            runtime_url: runtime_url.to_string(),
            runtime_api_key: None,
            runtime_num_predict: None,
            miner: None,
            mesh_fallback: if fallback {
                Some(crate::ping::MeshFallbackChat::spawn().expect("test fallback"))
            } else {
                None
            },
            guard: Some(std::sync::Arc::new(guard)),
            model_router: None,
            base_model: None,
            web: None,
            chain_url: None,
            metrics: Arc::new(crate::metrics::Metrics::new()),
        };
        TestServer::new(gw.router().into_make_service()).expect("guarded test server")
    }

    /// This machine's NON-loopback address (the UDP connect-trick the
    /// config uses for advertise) — remote-egress tests point the runtime
    /// URL at it so the gateway treats the forward as true egress.
    fn nonloopback_ip_for_tests() -> String {
        let sock = std::net::UdpSocket::bind("0.0.0.0:0").expect("udp probe socket");
        sock.connect("8.8.8.8:80")
            .expect("a routable default route (the e2e requires one too)");
        let ip = sock
            .local_addr()
            .expect("probe local addr")
            .ip()
            .to_string();
        assert!(
            !ip.starts_with("127.") && ip != "::1",
            "detection must produce a non-loopback address, got {ip}"
        );
        ip
    }

    fn blocked_ssn_request() -> Value {
        json!({
            "model": "m",
            "messages": [{"role": "user", "content": "my SSN is 078-05-1120"}]
        })
    }

    #[tokio::test]
    async fn guard_blocks_nonloopback_runtime_requests_before_forwarding() {
        // REMOTE runtime (non-loopback): the payload leaves the node, so
        // the guard answers BEFORE any forward is attempted — a 400, not
        // the dead address's connection error.
        let server = spawn_guarded("http://10.255.255.1:9".into(), false);
        let res = server
            .post("/v1/chat/completions")
            .json(&blocked_ssn_request())
            .await;
        assert_eq!(res.status_code(), axum::http::StatusCode::BAD_REQUEST);
        let body: Value = res.json();
        assert!(
            body["error"]
                .as_str()
                .unwrap()
                .contains("blocked by jerith-guard"),
            "error must name the guard, got {body}"
        );
        // Detection report present, raw value never echoed back.
        assert_eq!(body["guard"]["blocked"], Value::Bool(true));
        assert!(!body.to_string().contains("078-05-1120"));
    }

    #[tokio::test]
    async fn guard_blocks_at_mesh_fallback_edge_when_runtime_is_loopback() {
        // Loopback runtime (screen correctly skipped at intake) + mesh
        // fallback armed: the local forward fails, and the FALLBACK edge
        // screens before any peer dispatch — blocked content still 400s
        // with the guard's report, never reaching a peer.
        let server = spawn_guarded("http://127.0.0.1:1".into(), true);
        let res = server
            .post("/v1/chat/completions")
            .json(&blocked_ssn_request())
            .await;
        assert_eq!(res.status_code(), axum::http::StatusCode::BAD_REQUEST);
        let body: Value = res.json();
        assert!(
            body["error"]
                .as_str()
                .unwrap()
                .contains("blocked by jerith-guard"),
            "expected the fallback-edge guard block, got {body}"
        );
        assert!(!body.to_string().contains("078-05-1120"));
    }

    #[tokio::test]
    async fn loopback_runtime_intake_is_not_screened_when_nothing_can_leave() {
        // Loopback runtime, NO fallback: nothing can ever leave the node,
        // so the guard is deliberately not consulted — the dead runtime's
        // failure surfaces as the familiar 502 (intake stays touch-free;
        // agent tool-loop payloads must not trip the PII heuristics).
        let server = spawn_guarded("http://127.0.0.1:1".into(), false);
        let res = server
            .post("/v1/chat/completions")
            .json(&blocked_ssn_request())
            .await;
        assert_eq!(res.status_code(), axum::http::StatusCode::BAD_GATEWAY);
    }

    #[tokio::test]
    async fn metrics_record_model_latency_and_tokens_per_completion() {
        // A runtime reply carrying usage (OpenAI shape) feeds the meters:
        // one request, its tokens, its latency band, per-model aggregate.
        let app = axum::Router::new().route(
            "/v1/chat/completions",
            axum::routing::post(|| async {
                axum::Json(json!({
                    "model": "qwen3:8b",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}]
                }))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let mut cfg = crate::config::NodeConfig {
            runtime_url: format!("http://{addr}"),
            ..crate::config::NodeConfig::default()
        };
        cfg.ensure_identity().expect("test identity");
        let (state, _rx) = MeshState::new(cfg.clone()).expect("test mesh state");
        let metrics = Arc::new(crate::metrics::Metrics::new());
        let gw = Gateway {
            state,
            runtime_url: cfg.runtime_url.clone(),
            runtime_api_key: None,
            runtime_num_predict: None,
            miner: None,
            mesh_fallback: None,
            guard: None,
            model_router: None,
            base_model: None,
            web: None,
            chain_url: None,
            metrics: metrics.clone(),
        };
        let server = TestServer::new(gw.router().into_make_service()).expect("metrics test server");
        // Drive the completion through the GATEWAY (which meters), with the
        // mock runtime behind it.
        let res = server
            .post("/v1/chat/completions")
            .json(&json!({"model": "x", "messages": [{"role": "user", "content": "hi"}]}))
            .await;
        assert!(res.status_code().is_success());
        let snap = metrics.snapshot();
        assert_eq!(snap.requests_total, 1);
        assert_eq!(snap.ok_total, 1);
        // Model recorded is the RESPONSE's model (qwen3:8b), not caller x.
        assert!(
            snap.models.contains_key("qwen3:8b"),
            "models: {:?}",
            snap.models.keys()
        );
        assert_eq!(snap.models["qwen3:8b"].total_tokens, 15);
        assert!(snap.models["qwen3:8b"].latency_ms_max < 60_000);
    }

    #[tokio::test]
    async fn guard_redacts_payload_before_it_reaches_a_remote_runtime() {
        // Recording runtime bound to this machine's NON-loopback address:
        // the gateway must treat the forward as egress and redact first.
        let received = std::sync::Arc::new(tokio::sync::Mutex::new(String::new()));
        let recv = received.clone();
        let runtime = axum::Router::new().route(
            "/v1/chat/completions",
            axum::routing::post(move |body: String| {
                let recv = recv.clone();
                async move {
                    *recv.lock().await = body;
                    axum::Json(json!({
                        "choices": [{"message": {"role": "assistant", "content": "ok"}}]
                    }))
                }
            }),
        );
        let ip = nonloopback_ip_for_tests();
        let listener = tokio::net::TcpListener::bind(format!("{ip}:0"))
            .await
            .expect("bind on the non-loopback address");
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, runtime).await.expect("runtime up") });

        let server = spawn_guarded(format!("http://{addr}"), false);
        let res = server
            .post("/v1/chat/completions")
            .json(&json!({
                "model": "m",
                "messages": [{"role": "user", "content": "ping bob@corp.example please"}]
            }))
            .await;
        assert_eq!(res.status_code(), axum::http::StatusCode::OK);
        let got = received.lock().await.clone();
        assert!(got.contains("[REDACTED_EMAIL]"), "runtime saw: {got}");
        assert!(!got.contains("bob@corp.example"), "runtime saw: {got}");
    }

    /// Recording runtime + gateway with the runaway cap armed: forwarded
    /// chats must carry max_tokens, and a caller's own cap must never be
    /// overridden (caller intent always wins).
    async fn assert_forwarded_body(gw_cap: Option<u64>, request: Value) -> serde_json::Value {
        let received = std::sync::Arc::new(tokio::sync::Mutex::new(String::new()));
        let recv = received.clone();
        let runtime = axum::Router::new().route(
            "/v1/chat/completions",
            axum::routing::post(move |body: String| {
                let recv = recv.clone();
                async move {
                    *recv.lock().await = body;
                    axum::Json(json!({
                        "choices": [{"message": {"role": "assistant", "content": "ok"}}]
                    }))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, runtime).await.expect("runtime up") });

        let mut cfg = crate::config::NodeConfig {
            runtime_url: format!("http://{addr}"),
            ..crate::config::NodeConfig::default()
        };
        cfg.ensure_identity().expect("test identity");
        let (state, _rx) = MeshState::new(cfg).expect("test mesh state");
        let gw = Gateway {
            state: state.clone(),
            runtime_url: format!("http://{addr}"),
            runtime_api_key: None,
            runtime_num_predict: gw_cap,
            miner: None,
            mesh_fallback: None,
            guard: None,
            model_router: None,
            base_model: None,
            web: None,
            chain_url: None,
            metrics: Arc::new(crate::metrics::Metrics::new()),
        };
        let server = TestServer::new(gw.router().into_make_service()).expect("test server");
        let res = server.post("/v1/chat/completions").json(&request).await;
        assert_eq!(res.status_code(), axum::http::StatusCode::OK);
        serde_json::from_str(&received.lock().await).expect("forwarded body json")
    }

    #[tokio::test]
    async fn num_predict_cap_caps_uncapped_forwards_only() {
        // Cap armed + caller cap-free → forwarded body carries max_tokens.
        let forwarded = assert_forwarded_body(
            Some(50),
            json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]}),
        )
        .await;
        assert_eq!(forwarded.get("max_tokens"), Some(&json!(50)));

        // Caller already capped (either OpenAI spelling) → untouched.
        let forwarded = assert_forwarded_body(
            Some(50),
            json!({"model": "m", "max_tokens": 999, "messages": [{"role": "user", "content": "hi"}]}),
        )
        .await;
        assert_eq!(forwarded.get("max_tokens"), Some(&json!(999)));

        // Gate disarmed → forwarded body carries no cap at all.
        let forwarded = assert_forwarded_body(
            None,
            json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]}),
        )
        .await;
        assert_eq!(forwarded.get("max_tokens"), None);
    }

    /// Tool/Deep tier reroutes: the tool tier's forwarded body must carry
    /// `think: false` (budget goes to calls/text, not the reasoning
    /// channel — the "empty answer at exactly the cap" degenerate), while
    /// the deep tier keeps thinking and basic chats keep the caller's
    /// model untouched with no think key at all.
    #[tokio::test]
    async fn tier_reroutes_forward_think_false_only_for_the_tool_tier() {
        let received = std::sync::Arc::new(tokio::sync::Mutex::new("".to_string()));
        let recv = received.clone();
        let runtime = axum::Router::new().route(
            "/v1/chat/completions",
            axum::routing::post(move |body: String| {
                let recv = recv.clone();
                async move {
                    *recv.lock().await = body;
                    axum::Json(json!({
                        "choices": [{"message": {"role": "assistant", "content": "ok"}}]
                    }))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, runtime).await.expect("runtime up") });
        let runtime_url = format!("http://{addr}");

        let mut cfg = crate::config::NodeConfig {
            runtime_url: runtime_url.clone(),
            ..crate::config::NodeConfig::default()
        };
        cfg.ensure_identity().expect("test identity");
        cfg.runtime_tool_model = Some("tier-tool:9b".to_string());
        cfg.runtime_deep_model = Some("tier-deep:9b".to_string());
        let (state, _rx) = MeshState::new(cfg.clone()).expect("test mesh state");
        let gw = Gateway {
            state: state.clone(),
            runtime_url: runtime_url.clone(),
            runtime_api_key: None,
            runtime_num_predict: None,
            miner: None,
            mesh_fallback: None,
            guard: None,
            model_router: Some(std::sync::Arc::new(
                crate::model_router::RouterModels::from_config(&cfg),
            )),
            base_model: None,
            web: None,
            chain_url: None,
            metrics: Arc::new(crate::metrics::Metrics::new()),
        };
        let server = TestServer::new(gw.router().into_make_service()).expect("test server");

        let tool_request = json!({
            "model": "jerith-mesh",
            "messages": [{"role": "user", "content": "search the notes"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "memory_search",
                    "description": "search",
                    "parameters": {"type": "object", "properties": {}}
                }
            }]
        });
        server
            .post("/v1/chat/completions")
            .json(&tool_request)
            .await;
        let forwarded: serde_json::Value = serde_json::from_str(&received.lock().await).unwrap();
        assert_eq!(forwarded["model"], json!("tier-tool:9b"), "tool reroute");
        assert_eq!(
            forwarded["think"],
            json!(false),
            "tool tier disables thinking"
        );

        let deep_request = json!({
            "model": "jerith-mesh",
            "messages": [{"role": "user", "content": "x".repeat(5000)}]
        });
        server
            .post("/v1/chat/completions")
            .json(&deep_request)
            .await;
        let forwarded: serde_json::Value = serde_json::from_str(&received.lock().await).unwrap();
        assert_eq!(forwarded["model"], json!("tier-deep:9b"), "deep reroute");
        assert_eq!(forwarded.get("think"), None, "deep tier keeps thinking");

        let basic = json!({
            "model": "jerith-mesh",
            "messages": [{"role": "user", "content": "hey"}]
        });
        server.post("/v1/chat/completions").json(&basic).await;
        let forwarded: serde_json::Value = serde_json::from_str(&received.lock().await).unwrap();
        assert_eq!(forwarded["model"], json!("jerith-mesh"), "basic untouched");
        assert_eq!(forwarded.get("think"), None);
    }

    #[tokio::test]
    async fn chat_oversized_payload_is_413_not_forwarded() {
        // Nothing listens on :1 — any success below would mean the payload
        // slipped past the cap toward the (dead) runtime.
        let server = spawn_plain("http://127.0.0.1:1", false);
        let pad = "x".repeat(CHAT_FORWARD_CAP_BYTES + 10);
        let res = server
            .post("/v1/chat/completions")
            .json(&json!({
                "model": "m",
                "messages": [{"role": "user", "content": pad}]
            }))
            .await;
        assert_eq!(res.status_code(), axum::http::StatusCode::PAYLOAD_TOO_LARGE);
        let body: Value = res.json();
        assert!(
            body["error"].as_str().unwrap().contains("exceeds"),
            "got: {body}"
        );
        assert_eq!(body.get("mesh_fallback"), Some(&Value::Null));
    }

    #[tokio::test]
    async fn memory_search_caps_top_k_and_query() {
        let server = spawn_with_memory();
        server
            .post("/v1/skills/memory/embed")
            .json(&json!({"text": "at least one row to search over"}))
            .await;

        // Absurd top_k: must neither panic nor materialize an oversized
        // reply — the 1..=25 clamp keeps the response shape bounded.
        let res = server
            .post("/v1/skills/memory/search")
            .json(&json!({"query": "row", "top_k": u64::MAX}))
            .await;
        assert_eq!(res.status_code(), axum::http::StatusCode::OK);
        let body: Value = res.json();
        assert!(body["hits"].as_array().unwrap().len() <= 25);

        // Query over the 10k-char cap → 413 before an embed round-trip.
        let big = "y".repeat(10_001);
        let res = server
            .post("/v1/skills/memory/search")
            .json(&json!({"query": big}))
            .await;
        assert_eq!(res.status_code(), axum::http::StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn status_surfaces_security_counters() {
        use std::sync::atomic::Ordering;

        let server = spawn_with_memory();
        let mem_before = crate::metrics::OVERSIZE_MEMORY_REJECTS.load(Ordering::Relaxed);
        // One oversize rejection so the /status section has a live,
        // nonzero breakdown value to read back.
        let big = "z".repeat(10_001);
        let res = server
            .post("/v1/skills/memory/search")
            .json(&json!({"query": big}))
            .await;
        assert_eq!(res.status_code(), axum::http::StatusCode::PAYLOAD_TOO_LARGE);
        assert!(
            crate::metrics::OVERSIZE_MEMORY_REJECTS.load(Ordering::Relaxed) > mem_before,
            "memory oversize counter must reflect the 413"
        );

        // The /status snapshot exposes the same process-local counters.
        let res = server.get("/status").await;
        assert_eq!(res.status_code(), axum::http::StatusCode::OK);
        let body: Value = res.json();
        let sec = body.get("security").expect("security section in /status");
        for key in [
            "rate_limited_429_total",
            "oversize_reject_total",
            "oversize_chat_rejects",
            "oversize_memory_rejects",
            "oversize_bootstrap_rejects",
        ] {
            assert!(
                sec[key].is_u64(),
                "security.{key} must be a u64 counter, got {sec}"
            );
        }
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
    async fn chat_sse_stream_with_dead_local_gets_mesh_sse_not_502() {
        // stream: true + dead local runtime + empty peer registry → the
        // fallback SSE shape when a peer can serve, or a clean 502 with
        // mesh_fallback null when none can. Either way never a raw 500.
        let server = spawn_plain("http://127.0.0.1:1", true);
        let res = server
            .post("/v1/chat/completions")
            .json(&json!({
                "model": "m",
                "stream": true,
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .await;
        let status = res.status_code();
        assert_eq!(status, axum::http::StatusCode::BAD_GATEWAY); // registry empty here
        let body: Value = res.json();
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

    #[tokio::test]
    async fn model_less_basic_chat_defaults_to_the_advertised_base_model() {
        // A client that omits `model` on a basic (tool-less) chat must get
        // the advertised base model substituted — not a forwarded null that
        // the runtime 400s on ("Invalid model name … model=None",
        // observed live 2026-10-16).
        let received = Arc::new(tokio::sync::Mutex::new(String::new()));
        let recv = received.clone();
        let runtime = axum::Router::new().route(
            "/v1/chat/completions",
            axum::routing::post(move |body: String| {
                let recv = recv.clone();
                async move {
                    *recv.lock().await = body;
                    axum::Json(json!({
                        "choices": [{"message": {"role": "assistant", "content": "ok"}}]
                    }))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, runtime).await.unwrap() });
        let runtime_url = format!("http://{addr}");

        let mut cfg = crate::config::NodeConfig {
            runtime_url: runtime_url.clone(),
            ..crate::config::NodeConfig::default()
        };
        cfg.ensure_identity().expect("test identity");
        let (state, _rx) = MeshState::new(cfg).expect("test mesh state");
        let gw = Gateway {
            state: state.clone(),
            runtime_url,
            runtime_api_key: None,
            runtime_num_predict: None,
            miner: None,
            mesh_fallback: None,
            guard: None,
            model_router: None,
            base_model: Some("base:7b".to_string()),
            web: None,
            chain_url: None,
            metrics: Arc::new(crate::metrics::Metrics::new()),
        };
        let server = TestServer::new(gw.router().into_make_service()).expect("test server");
        let res = server
            .post("/v1/chat/completions")
            .json(&json!({"messages": [{"role": "user", "content": "hi"}]}))
            .await;
        assert!(res.status_code().is_success());
        let forwarded: Value = serde_json::from_str(&received.lock().await).expect("runtime body");
        assert_eq!(forwarded["model"], json!("base:7b"));
    }

    #[tokio::test]
    async fn buffered_runtime_answers_are_sanitized_of_html_and_trace_tags() {
        // The runtime laces visible text with `<br>` debris and a leaked
        // `</think>` marker (observed verbatim from deepseek-r1 via ollama,
        // 2026-10-16): the gateway relays the words, not the artifacts.
        let runtime = axum::Router::new().route(
            "/v1/chat/completions",
            axum::routing::post(|| async {
                axum::Json(json!({
                    "choices": [{"message": {
                        "role": "assistant",
                        "content": "A<br>B\u{3C}think\u{3E}C</br>D<br />E"
                    }}],
                    "usage": {"total_tokens": 9}
                }))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, runtime).await.unwrap() });
        let runtime_url = format!("http://{addr}");

        let mut cfg = crate::config::NodeConfig {
            runtime_url: runtime_url.clone(),
            ..crate::config::NodeConfig::default()
        };
        cfg.ensure_identity().expect("test identity");
        let (state, _rx) = MeshState::new(cfg).expect("test mesh state");
        let gw = Gateway {
            state: state.clone(),
            runtime_url,
            runtime_api_key: None,
            runtime_num_predict: None,
            miner: None,
            mesh_fallback: None,
            guard: None,
            model_router: None,
            base_model: Some("m".to_string()),
            web: None,
            chain_url: None,
            metrics: Arc::new(crate::metrics::Metrics::new()),
        };
        let server = TestServer::new(gw.router().into_make_service()).expect("test server");
        let res = server
            .post("/v1/chat/completions")
            .json(&json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]}))
            .await;
        assert!(res.status_code().is_success());
        let body: Value = res.json();
        assert_eq!(
            body["choices"][0]["message"]["content"],
            json!("A\nBC\nD\nE")
        );
    }
}
