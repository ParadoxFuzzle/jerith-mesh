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

/// Speculative-decode chat lane state (FEATURES §1): admission-probe
/// outcome plus per-lane traffic stats. Honest shapes everywhere — the
/// engine owns the actual draft-and-verify (it has the logits); the mesh
/// records whether its hint was ACCEPTED and what each lane's measured
/// tokens/sec looks like, publishing statistics, never verdicts.
#[derive(Default)]
pub struct SpecLane {
    /// Admission probe: 0 = unprobed, 1 = admitted, 2 = refused,
    /// 3 = probe in flight. Once refused, the hint merge goes inert for
    /// the process lifetime (a real user request never becomes the
    /// Guinea pig for an experimental engine field).
    probe: std::sync::atomic::AtomicU8,
    /// Non-stream forward stats: hint lane (requests, completion tokens,
    /// wall ms) — fills only when usage telemetry comes back.
    hinted: tokio::sync::Mutex<SpecLaneStats>,
    /// Same fields for unhinted forwards on the same lane.
    plain: tokio::sync::Mutex<SpecLaneStats>,
}

#[derive(Default, Clone, Debug)]
pub struct SpecLaneStats {
    pub requests: u64,
    pub completion_tokens: u64,
    pub wall_ms: u64,
}

impl SpecLane {
    fn probe_state(&self) -> u8 {
        self.probe.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Record one completed non-stream forward.
    async fn record(&self, hinted: bool, tokens: u64, wall_ms: u64) {
        if tokens == 0 || wall_ms == 0 {
            return; // nothing honest to average over
        }
        let slot = if hinted { &self.hinted } else { &self.plain };
        let mut s = slot.lock().await;
        s.requests += 1;
        s.completion_tokens += tokens;
        s.wall_ms += wall_ms;
    }

    /// Tokens/sec per lane — None until at least one full datum landed.
    fn lane_tps(s: &SpecLaneStats) -> Option<f64> {
        if s.requests == 0 || s.wall_ms == 0 {
            None
        } else {
            Some(s.completion_tokens as f64 / (s.wall_ms as f64 / 1000.0))
        }
    }

    /// The /status card: config truth + admission + measured speeds.
    /// `speedup` is the hinted/plain ratio, published ONLY when both
    /// lanes have ≥ 5 measured requests (statistical honesty, never a
    /// one-request verdict).
    pub async fn status_value(&self, cfg: Option<&crate::config::SpecDecodeConfig>) -> Value {
        let probe_n = self.probe_state();
        let (h, p) = {
            let h = self.hinted.lock().await.clone();
            let p = self.plain.lock().await.clone();
            (h, p)
        };
        let h_tps = Self::lane_tps(&h);
        let p_tps = Self::lane_tps(&p);
        let speedup = match (h_tps, p_tps) {
            (Some(h_speed), Some(p_speed))
                if h.requests >= 5 && p.requests >= 5 && p_speed > 0.0 =>
            {
                serde_json::Value::from((h_speed / p_speed * 100.0).round() / 100.0)
            }
            _ => Value::Null,
        };
        json!({
            "enabled": cfg.is_some(),
            "num_speculative_tokens": cfg.map(|c| c.num_speculative_tokens).unwrap_or(0),
            "admission": match probe_n {
                0 => Value::Null,
                1 => serde_json::Value::from("accepted"),
                2 => serde_json::Value::from("refused"),
                _ => serde_json::Value::from("probing"),
            },
            "hinted": { "requests": h.requests, "tokens_per_sec": h_tps },
            "plain": { "requests": p.requests, "tokens_per_sec": p_tps },
            "speedup": speedup,
        })
    }
}

/// Pure hint merge: when the lane is live (enabled + admitted tokens),
/// stamp `spec_decode` onto the forward body (engines that know the
/// field consume it; drop_params proxies ignore it inertly). Returns
/// whether a hint rode this forward — the stat-recording key.
pub fn apply_spec_decode_hint(
    body: &mut Value,
    cfg: &crate::config::SpecDecodeConfig,
    admitted: bool,
) -> bool {
    if !admitted || cfg.num_speculative_tokens == 0 {
        return false;
    }
    body["spec_decode"] = json!({
        "num_speculative_tokens": cfg.num_speculative_tokens,
    });
    true
}

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
    /// Per-node JER wallet card (Phase 4): which chain RPC to ask and
    /// which identity/address answers for this node's wallet. Present when
    /// the crypto chain is JerithChain; /status then carries a live
    /// `wallet` section — chain-truthful balance rather than local wins.
    pub wallet: Option<std::sync::Arc<crate::economics::WalletCard>>,
    /// Optional-chain wallet card (§9, Phase 4): the user-supplied chain
    /// node's balance probe when chain ∈ {bitcoin, ethereum, dogecoin,
    /// monero} AND `wallet_endpoint` is configured. /status carries it in
    /// the crypto section as `optional_wallet` — mined balance by their
    /// own node, typed refusal on error, never a fake number.
    pub optional_wallet: Option<std::sync::Arc<crate::economics::OptionalChainWallet>>,
    /// Speculative-decode hint config (config `runtime_spec_decode`).
    /// None → the chat lane forwards untouched, exactly as before.
    pub spec_decode: Option<crate::config::SpecDecodeConfig>,
    /// Spec-lane admission + measured per-lane speeds (always present).
    pub spec_lane: Arc<SpecLane>,
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
            .route(
                "/v1/skills/writing/factcheck",
                post(skill_writing_factcheck),
            )
            .route("/v1/skills/writing/template", post(skill_writing_template))
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
    let mut text: String = match text.chars().count() {
        n if n <= MEMORY_EMBED_TEXT_CAP_CHARS => text.to_string(),
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
    // Secrets rule (operator, 2026-10-30): API keys, vault secrets, and
    // credential-shaped strings never land in memory. The same Guard that
    // screens egress screens the embed input: blocked-category hits (the
    // api-keys family in the shipped policy) REFUSE the write outright;
    // redact-category hits store the REDACTED text, never the raw secret.
    if let Some(guard) = &gw.guard {
        let probe = serde_json::json!({"role": "user", "content": text});
        match guard.screen_messages(std::slice::from_ref(&probe)) {
            Ok(Some(patched)) => {
                let redacted = patched
                    .first()
                    .and_then(|m| m.get("content").and_then(Value::as_str))
                    .unwrap_or(text.as_str());
                if redacted != text.as_str() {
                    tracing::warn!(
                        target: "guard",
                        "memory embed carried secret/PII-shaped text — storing the REDACTED version"
                    );
                    text = redacted.to_string();
                }
            }
            Ok(None) => {} // clean
            Err(verdict) => {
                return (
                    StatusCode::UNPROCESSABLE_ENTITY,
                    Json(json!({
                        "error": "memory embed refused",
                        "reason": verdict.block_reason(),
                        "why": "secrets/credentials never go into memory (guard API-key/password categories are policy-blocked)"
                    })),
                )
                    .into_response();
            }
        }
    }
    match memory.embed_and_store(&text).await {
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
    // Per-peer breakdown with the same derived views rendered per row.
    let peers: Vec<serde_json::Value> = snap
        .peers
        .iter()
        .map(|p| {
            json!({
                "peer": p.peer,
                "requests": p.requests,
                "ok": p.ok,
                "failed": p.failed,
                "ok_percent": p.ok_percent(),
                "latency_ms_mean": p.latency_ms_mean(),
                "latency_ms_max": p.latency_ms_max,
                "tokens_total": p.tokens_total,
            })
        })
        .collect();
    Json(json!({
        "window_started_ms": snap.window_started_ms,
        "requests_total": snap.requests_total,
        "ok_total": snap.ok_total,
        "failed_total": snap.failed_total,
        "tokens_total": snap.tokens_total,
        "models": models,
        "recent": snap.recent,
        "peers": peers,
        "history": snap.history,
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
        // Reliability (2026-10-30): the "which provider answered?" line —
        // provider label of the last served completion plus a stayed-local
        // verdict, so private-work callers check on-box vs off-box at a
        // glance.
        "served_by": {
            "last": gw.metrics.served_by(),
            "stayed_local": gw
                .metrics
                .served_by()
                .map(|s| s.starts_with("local:"))
                .unwrap_or(false),
        },
        "measurement": measurement,
    }))
    .into_response()
}

/// The fleet resource census (Phase 1): our live sample + whatever
/// connectable peers advertised inside their PeerAd. The bootstrap lists
/// OUR OWN advertisement back in the peer set, so a self-ad is a duplicate
/// of the live sample and is skipped — otherwise the census counts this
/// node twice and its RAM/VRAM inflates every fleet sum. Unknown amounts
/// are excluded from sums (unknown ≠ zero — the placement rule).
fn aggregate_fleet_census(
    own_id: &str,
    own: Option<crate::hw::ResourceSample>,
    peer_ads: impl IntoIterator<Item = (String, Option<crate::hw::ResourceSample>)>,
) -> serde_json::Value {
    let mut samples: Vec<(String, crate::hw::ResourceSample)> = Vec::new();
    if let Some(own) = own {
        samples.push((own_id.to_string(), own));
    }
    for (node_id, res) in peer_ads {
        if node_id == own_id {
            continue; // self-ad duplicate — see above
        }
        if let Some(res) = res {
            samples.push((node_id, res));
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
    json!({
        "reporting_nodes": samples.len(),
        "ram_total_bytes": ram_total_bytes,
        "ram_available_bytes": ram_available_bytes,
        "vram_free_bytes": vram_free_known.first().map(|_| vram_free_known.iter().sum::<u64>()),
        "vram_total_bytes": vram_total_known.first().map(|_| vram_total_known.iter().sum::<u64>()),
        "nodes": samples
            .iter()
            .map(|(id, s)| json!({"node_id": id, "census": s.census()}))
            .collect::<Vec<_>>(),
    })
}

/// The operator-facing peer roster (PeerRegistry → JSON). Every peer except
/// SELF — the bootstrap echoes our own advertisement back into the peer
/// set, and this node is not its own peer; counting it inflated
/// `peer_count`/`peers_alive` and would put a self-row in operator tools.
/// Sorted by node id so a dashboard isn't reordered by HashMap iteration.
fn peers_roster(reg: &crate::bootstrap::PeerRegistry, own_id: &str) -> serde_json::Value {
    let ads: Vec<crate::bootstrap::PeerAd> = reg.all_ads();
    let mut rows: Vec<&crate::bootstrap::PeerAd> =
        ads.iter().filter(|ad| ad.node_id != own_id).collect();
    rows.sort_by(|a, b| a.node_id.cmp(&b.node_id));
    let rows: Vec<_> = rows
        .iter()
        .map(|ad| {
            let health = reg.health_of(&ad.node_id);
            let state = health
                .map(|h| h.state)
                .unwrap_or(crate::bootstrap::PeerState::Unknown);
            let state = match state {
                crate::bootstrap::PeerState::Unknown => "unknown",
                crate::bootstrap::PeerState::Alive => "alive",
                crate::bootstrap::PeerState::Deprecated => "deprecated",
            };
            json!({
                "node_id": ad.node_id,
                "address": ad.address,
                "rpc_address": ad.rpc_address,
                "model": ad.model_id,
                "quant": ad.quant,
                "role": ad.role,
                "contribution_percent": ad.contribution_percent,
                "state": state,
                "last_rtt_ms": health.and_then(|h| h.last_rtt_ms),
                "smoothed_rtt_ms": health.and_then(|h| h.smoothed_rtt_ms),
                "consecutive_failures": health.map(|h| h.consecutive_failures).unwrap_or(0),
                "hosted_experts": ad.hosted_experts.clone().unwrap_or_default(),
            })
        })
        .collect();
    serde_json::Value::Array(rows)
}

/// Human/CLI-facing status snapshot.
async fn status(State(gw): State<Arc<Gateway>>) -> axum::response::Response {
    let cfg = gw.state.config.read().await;
    let reg = gw.state.registry.lock().await;
    let meters = gw.state.meters.lock().await;
    let roster = peers_roster(&reg, &cfg.node_id);
    let peer_count = roster.as_array().map(|a| a.len()).unwrap_or(0);
    let peers_alive = roster
        .as_array()
        .map(|a| a.iter().filter(|r| r["state"] != "deprecated").count())
        .unwrap_or(0);
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
        // Work-carrying RPCs typed-refused by the escrow join gate this
        // process. 0 whenever staking is disabled — the honest tell that
        // the refusal path exists only behind the Phase 5 tier.
        "gate_refusals": gw.state.stake.gate_refusals(),
    });
    // Probe hoist: the three live sections used to evaluate SERIALLY
    // inside json! — three TLS'd round trips measured 140-190 ms per
    // /status poll. Concurrent (tokio::join!) they cost the MAX of one
    // probe, and every probe rides the pooled shared client instead of
    // building a fresh TLS stack per request.
    let chain_token = crate::economics::read_token();
    let (bootstrap_value, chain_value, wallet_value) = tokio::join!(
        bootstrap_sybil_status(Some(cfg.bootstrap_url.as_str())),
        chain_probe(gw.chain_url.as_deref(), chain_token.as_deref()),
        wallet_status(gw.wallet.as_deref()),
    );
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
    let fleet_value = aggregate_fleet_census(
        &cfg.node_id,
        own_sample,
        reg.connectable()
            .into_iter()
            .map(|(ad, _)| (ad.node_id.clone(), ad.resources.clone())),
    );
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
                // Phase 3 speculative prefetch: warm-up actions + failures,
                // plus the knobs that drive them (hot-reloadable).
                "prefetches": s.prefetches,
                "prefetch_errors": s.prefetch_errors,
                "prefetch_enabled": cfg.experts.prefetch_enabled,
                "prefetch_per_layer": cfg.experts.prefetch_per_layer,
                "prefetch_horizon": cfg.experts.prefetch_horizon,
                // Demand signal + placement policy (Phase 2 replication).
                "activations_total": stats.total(),
                "activations_distinct": stats.distinct(),
                "counts": stats.snapshot()["counts"],
                "hot_percent": percent,
                "hot": hot,
                "cold": cold.len(),
                // §1 draft-expert scheduling: whether the first remote
                // hop leads by pure RTT (the draft must not wait behind
                // a slower, resource-richer peer).
                "draft_priority": cfg.experts.draft_priority,
                // §1 phase 2 (draft hot set): the always-resident tier —
                // the knob and the ids actually pinned right now (the
                // lowest-layer slice the replication round derives from
                // the hosted store when the knob is on).
                "draft_hot_percent": cfg.experts.draft_hot_percent,
                "draft_pinned": store.pinned_ids(),
                "pinned_resident_bytes": store.resident_bytes(),
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
    // §9 honest crypto surface: what is configured, what is actually live,
    // and — for an optional chain — the typed refusal for the un-
    // implemented mining half (no fake mining stats), plus the live wallet
    // card for the configured wallet_endpoint half.
    let optional_wallet_value = match gw.optional_wallet.as_ref() {
        Some(card) => match crate::economics::optional_chain_probe(card).await {
            Ok(b) => json!({
                "ok": true,
                "chain": b.chain,
                "unit": b.unit,
                "amount": b.amount,
                "raw": b.raw,
                "address": b.address,
            }),
            Err(e) => json!({"ok": false, "error": e.to_string()}),
        },
        None => Value::Null,
    };
    let crypto_value = json!({
        "chain": cfg.crypto.chain,
        "mining_enabled": gw.miner.is_some(),
        // §13 drift watch, miner half: attempts/wins are the existing
        // counters; the FAILURE STREAK is the silent-token-drift alarm
        // (a stale api token mined nothing for a long stretch with only
        // debug logs before). streak > 0 with a healthy chain = red flag.
        "mine_attempts_total": gw
            .miner
            .as_ref()
            .map(|m| m.stats.attempted.load(std::sync::atomic::Ordering::Relaxed))
            .unwrap_or(0),
        "mine_wins_total": gw
            .miner
            .as_ref()
            .map(|m| m.stats.won.load(std::sync::atomic::Ordering::Relaxed))
            .unwrap_or(0),
        "mine_failure_streak": gw
            .miner
            .as_ref()
            .map(|m| m.stats.failure_streak.load(std::sync::atomic::Ordering::Relaxed))
            .unwrap_or(0),
        "mine_failures_total": gw
            .miner
            .as_ref()
            .map(|m| m.stats.failures.load(std::sync::atomic::Ordering::Relaxed))
            .unwrap_or(0),
        "wallet_card": gw.wallet.is_some(),
        "optional_wallet": optional_wallet_value,
        "note": crate::economics::optional_chain_note(cfg.crypto.chain),
    });
    // §12 privacy tiers: the live policy (hot-reloaded) and how narrow it
    // currently makes peer placement — an operator auditing "where can my
    // prompts go" reads this section, not the config file.
    let privacy_value = {
        let trusted = cfg.privacy.hop_allowlist().map(|l| l.len());
        json!({
            "level": cfg.privacy.level,
            "trusted_nodes": cfg.privacy.trusted_nodes,
            // None = everything connectable is a hop candidate (full_mesh).
            "allowlist_size": trusted,
        })
    };
    // Phase 5 reputation-weighted placement surface (live): every VERIFIED
    // score stamp the router currently carries, with its dispatch tier.
    // Only pin-wall-verified stamps exist here — unverified word never
    // reaches placement, and the section makes that legible.
    let reputation_value = {
        let cache = gw.state.reputation_cache.lock().await;
        json!({
            "verified_subjects": cache.iter().map(|s| json!({
                "subject": s.subject,
                "score": s.score,
                "stamp_ms": s.stamp_ms,
                "origin": s.origin,
                "penalized": crate::quic::reputation_tier(Some(s.score)) == 1,
            })).collect::<Vec<_>>(),
        })
    };
    // §2 model-switch supervisor: whether this node has gracefully LEFT
    // the mesh (signed leave + advertise latch) while a leave is pending
    // confirmation, which model would be rejoined after it, and how many
    // in-process switches have completed (the latched window is seconds
    // nowadays — the counter makes the visits auditable).
    let mesh_exit_value = {
        let left = gw
            .state
            .left_mesh_latch
            .load(std::sync::atomic::Ordering::SeqCst);
        json!({
            "left_mesh": left,
            "pending_model": *gw.state.pending_model.read().await,
            "model_switches": gw
                .state
                .model_switches
                .load(std::sync::atomic::Ordering::SeqCst),
        })
    };
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
        "peer_count": peer_count,
        "peers_alive": peers_alive,
        "peers": roster,
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
        // Crypto surface (§9): configured chain vs what is actually live,
        // with the typed adapter refusal for optional chains.
        "crypto": crypto_value,
        // Privacy tiers (§12): live level + trusted-node allowlist size.
        "privacy": privacy_value,
        // Phase 5 reputation-weighted placement surface: which subjects the
        // router sees as VERIFIED (only pin-verified stamps live in the
        // cache) and who currently sits in the penalized dispatch tier
        // (negative verified score — placement preference, never refusal;
        // refusals are the stake tier's job).
        "reputation": reputation_value,
        // Model-switch (§2): graceful-exit latch + pending restart model.
        "mesh_exit": mesh_exit_value,
        // Runaway-generation cap state (armed via config `runtime_num_predict`;
        // the site dashboard surfaces it so an operator can see it live).
        "runtime_num_predict": gw.runtime_num_predict,
        // Speculative-decode lane (§1): admission + measured speeds.
        "spec_decode": gw.spec_lane.status_value(gw.spec_decode.as_ref()).await,
        // Resource pooling (Phase 1): live fleet census.
        "fleet_resources": fleet_value,
        // Resource isolation (Phase 1): ceilings + kill-switch state.
        "resources": resources_value,
        // Per-peer blocking (Phase 3): refused ids + refusal counter.
        "peer_block": peer_block_value,
        // Discovery rotation (Phase 1): configured + learned + last-good.
        "discovery": discovery_value,
        // Pre-join mesh compat (FEATURES §12): the typed verdict over OUR
        // model identity vs the bootstrap census (quants mismatch,
        // brand-new mesh). Null until the first probe answered.
        "compat": gw.state.compat.read().await.clone().map(|v| serde_json::to_value(v).expect("compat verdict serializes")).unwrap_or(Value::Null),
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
        // Probes run CONCURRENTLY (tokio::join!): /status used to pay
        // three serial TLS'd round trips (~140-190 ms measured); each
        // probe now also rides the pooled client instead of building a
        // fresh TLS stack per request. Parallel = max(single probe).
        // All three keep the failure contract: a downed dependency reads
        // null / {"reachable":false} and never takes /status down.
        "bootstrap": bootstrap_value,
        "chain": chain_value,
        // Per-node JER wallet card (Phase 4): live balance from the chain
        // RPC — the earnings view's chain-truthful side. Null when no
        // wallet card (non-JerithChain); {"reachable":false} reads like
        // chain_probe: a downed chain never takes /status down with it.
        "wallet": wallet_value,
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
/// §10 anti-sybil view from OUR configured bootstrap's /health: the
/// zero-weight fraud-report tally (`sybil.young_fraud_reports`) — the
/// operator-facing mirror of the weighting acting live. Same contract as
/// chain_probe: a downed bootstrap reads `{"reachable": false}` and
/// never takes /status down with it; unconfigured URL = null.
async fn bootstrap_sybil_status(base: Option<&str>) -> Value {
    let Some(base) = base.map(str::trim).filter(|b| !b.is_empty()) else {
        return Value::Null;
    };
    let base = base.trim_end_matches('/');
    let url = format!("{base}/health");
    match crate::http::probe_http().get(&url).send().await {
        Ok(r) if r.status().is_success() => match r.json::<Value>().await {
            Ok(body) => json!({
                "reachable": true,
                "url": base,
                "young_fraud_reports": body
                    .pointer("/sybil/young_fraud_reports")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0),
            }),
            Err(_) => json!({
                "reachable": false,
                "url": base,
                "error": "non-JSON bootstrap health",
            }),
        },
        Ok(r) => json!({
            "reachable": false,
            "url": base,
            "status": r.status().as_u16(),
        }),
        Err(_) => json!({
            "reachable": false,
            "url": base,
            "error": "bootstrap health unreachable",
        }),
    }
}

async fn chain_probe(base: Option<&str>, token: Option<&str>) -> Value {
    let (Some(base), Some(token)) = (base, token) else {
        return Value::Null;
    };
    let base = base.trim().trim_end_matches('/');
    if base.is_empty() || token.trim().is_empty() {
        return Value::Null;
    }
    let url = format!("{base}/status");
    match crate::http::probe_http()
        .get(&url)
        .bearer_auth(token.trim())
        .send()
        .await
    {
        Ok(r) => match r.json::<Value>().await {
            Ok(body) => {
                // §13 drift watch: every live probe feed the monotonicity
                // watch — a shrinking total_mined_uj/height is a chain
                // ledger event, surfaced here next to the numbers.
                let total_mined_uj = numeric_u64(body.pointer("/mining/total_mined_uj"));
                let height = numeric_u64(body.pointer("/chain/height"));
                let drift = crate::economics::ChainDriftWatch::global_probe(total_mined_uj, height);
                json!({
                    "reachable": true,
                    "url": base,
                    "height": body.pointer("/chain/height"),
                    "tip_hash": body.pointer("/chain/tip_hash"),
                    "circulating_jer": body.pointer("/chain/circulating_jer"),
                    "max_supply_jer": body.pointer("/chain/max_supply_jer"),
                    "wallets": body.pointer("/wallets"),
                    "miners": body.pointer("/mining/miners"),
                    "total_mined_uj": body.pointer("/mining/total_mined_uj"),
                    "drift": drift,
                })
            }
            Err(_) => json!({"reachable": false, "url": base, "error": "non-JSON chain response"}),
        },
        Err(e) => json!({"reachable": false, "url": base, "error": e.to_string()}),
    }
}

/// A JSON number as u64 — ints and integral floats both count (a chain
/// emitting 21.0 must parse as the height 21).
fn numeric_u64(v: Option<&Value>) -> Option<u64> {
    match v? {
        Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                Some(u)
            } else {
                n.as_f64().filter(|f| *f >= 0.0).map(|f| f as u64)
            }
        }
        _ => None,
    }
}

/// Live wallet-card probe: the node's own JER balance from the chain RPC
/// (token-authed /balance). Reachability and wallet existence are separate
/// bits — an un-created wallet shows 0-everything, NOT unreachable; a
/// missing api token is an honest config error in the section itself.
async fn wallet_status(card: Option<&crate::economics::WalletCard>) -> Value {
    let Some(card) = card else {
        return Value::Null;
    };
    match crate::economics::wallet_balance(card).await {
        Ok(b) => json!({
            "reachable": true,
            "identity": card.identity,
            "by_address": card.address.is_some(),
            "exists": b.exists,
            "address": b.address,
            "balance_uj": b.balance_uj,
            "balance_jer": b.balance_jer,
            "available_jer": b.available_jer,
        }),
        Err(e) => json!({
            "reachable": false,
            "identity": card.identity,
            "error": e.to_string(),
        }),
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
            let mut body: Value = resp.json().await.unwrap_or_else(|_| json!({"data": []}));
            annotate_models(&mut body, gw.base_model.as_deref());
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

/// Reliability (2026-10-30): which-provider classification for a served
/// completion. "local:{model}" = the node's own hosted id (text stayed
/// on-box, modulo the local runtime), "mesh:{peer}" = a mesh peer answered
/// (left the box), "runtime:{model}" = a runtime-tier id — with LiteLLM
/// tiers these are the shapes that can proxy off-box. The label rides the
/// x-jerith-served-by header of chat completions and the /health
/// served_by line, so private-work callers can check on-box vs off-box at
/// a glance instead of discovering the route after the fact.
pub fn classify_served_by(base_model: Option<&str>, model: &str) -> String {
    if base_model.is_some_and(|b| b == model) {
        format!("local:{model}")
    } else if model.starts_with("mesh:") {
        model.to_string()
    } else {
        format!("runtime:{model}")
    }
}

/// Attach `x-jerith-served-by` to a built chat-completion response (every
/// return path that knows what answered goes through here).
fn attach_served_header(label: &str, resp: Response) -> Response {
    let mut resp = resp;
    let value = axum::http::HeaderValue::from_str(label)
        .unwrap_or_else(|_| axum::http::HeaderValue::from_static("unknown"));
    resp.headers_mut().insert("x-jerith-served-by", value);
    resp
}

/// Annotate a runtime /v1/models listing with per-row `provider` labels
/// (same classification as x-jerith-served-by) plus a top-level
/// `local_model` echo — the operator's "which of these stay on-box?"
/// view for private-work planning.
fn annotate_models(body: &mut Value, base_model: Option<&str>) {
    if let Some(rows) = body.get_mut("data").and_then(Value::as_array_mut) {
        for row in rows {
            if let Some(id) = row.get("id").and_then(Value::as_str) {
                let label = classify_served_by(base_model, id);
                let provider = label.split(':').next().unwrap_or("runtime");
                row["provider"] = Value::String(provider.to_string());
            }
        }
    }
    if let Some(base) = base_model {
        body["local_model"] = Value::String(base.to_string());
    }
}

/// skill: writing.factcheck — the fact-check pass for sell/publish drafts
/// (reliability 2026-10-30). ONE call to the node's OWN hosted model
/// (never a tier reroute, never mesh fallback): handed CANON + DRAFT, it
/// flags conflicting or likely-invented numbers, dates, people, places
/// and titles — it never rewrites. Canon is auto-loaded so every draft
/// is checked against the operator's register even when the caller
/// forgets to paste it (see `default_factcheck_canon_path`); the
/// response's `canon_source` always says what the draft was checked
/// against. Refused loudly when the node hosts no local model: on a
/// runtime-proxy-only node this skill would ship the draft off-box,
/// which is the exact failure it exists to prevent.
const FACTCHECK_DRAFT_CAP_CHARS: usize = 30_000;
const FACTCHECK_CANON_CAP_CHARS: usize = 60_000;

/// Default canon register for the fact-check pass: an explicit
/// `JERITH_FACTCHECK_CANON` file path first, then the operator-maintained
/// register in the node workspace (`<home>/workspace/COLONY-X-CANON.md`
/// — today: the Colony X canon). None → the pass runs draft-only until
/// the caller supplies `canon` in the body; a NONE verdict is never
/// fabricated from a missing file.
pub fn default_factcheck_canon_path() -> Option<std::path::PathBuf> {
    canon_register_path(
        std::env::var("JERITH_FACTCHECK_CANON").ok().as_deref(),
        std::path::Path::new(&crate::config::NodeConfig::home()),
    )
}

/// Pure form of the canon-register resolution (testable without touching
/// static env). Order: env hint → `<home>/workspace/COLONY-X-CANON.md`.
fn canon_register_path(
    env_override: Option<&str>,
    home: &std::path::Path,
) -> Option<std::path::PathBuf> {
    if let Some(p) = env_override.map(str::trim).filter(|p| !p.is_empty()) {
        let p = std::path::PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    let p = home.join("workspace").join("COLONY-X-CANON.md");
    p.is_file().then_some(p)
}

async fn skill_writing_factcheck(
    State(gw): State<Arc<Gateway>>,
    Json(body): Json<Value>,
) -> Response {
    let Some(base) = gw
        .base_model
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "error": "no local model declared ([model] unset) — fact-check refuses rather than leave the box",
            })),
        )
            .into_response();
    };
    let Some(draft) = body.get("draft").and_then(Value::as_str) else {
        return bad_request("field 'draft' is required");
    };
    if draft.chars().count() > FACTCHECK_DRAFT_CAP_CHARS {
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(json!({"error": format!("draft exceeds {FACTCHECK_DRAFT_CAP_CHARS} chars")})),
        )
            .into_response();
    }
    // Canon: an explicit `canon` body field wins; otherwise the
    // operator's default register is auto-loaded. A register that exists
    // but fails to READ is a loud 503, never a silent empty-canon CLEAN
    // (an empty canon would mint clean verdicts from nothing).
    let (canon, canon_source) = match resolve_writing_canon(&body, true, FACTCHECK_CANON_CAP_CHARS)
    {
        Ok(pair) => pair,
        Err(resp) => return *resp,
    };
    let system = "You are a strict fact-check reviewer. Compare the DRAFT only against the CANON \
        the user supplies above (no web access, no memory of anything else). List ONLY claims that \
        conflict with the canon or appear invented: numbers, percentages, dates, counts, named \
        people, places, episode titles. For each flagged claim output one bullet: the claim, why \
        it is suspect (conflicts-canon / no-canon-support / off-canon), and nothing else. Do NOT \
        rewrite or improve the draft. If nothing is suspect, output exactly: CLEAN";
    let user_text = if canon.is_empty() {
        format!("DRAFT:\n{draft}")
    } else {
        format!("CANON:\n{canon}\n\nDRAFT:\n{draft}")
    };
    let payload = json!({
        "model": base,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        "stream": false,
        "max_tokens": gw.runtime_num_predict.unwrap_or(1024),
    });
    let url = format!(
        "{}/v1/chat/completions",
        gw.runtime_url.trim_end_matches('/')
    );
    let mut req = crate::http::mesh_http().post(&url).json(&payload);
    if let Some(k) = &gw.runtime_api_key {
        req = req.bearer_auth(k);
    }
    let label = format!("local:{base}");
    match tokio::time::timeout(RUNTIME_HEADERS_TIMEOUT, req.send()).await {
        Ok(Ok(resp)) if resp.status().is_success() => {
            let value: Value = resp.json().await.unwrap_or(json!({}));
            let flags = value
                .pointer("/choices/0/message/content")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let flags = sanitize_runtime_text(flags);
            gw.metrics.note_served(label.clone());
            let reply = (
                StatusCode::OK,
                Json(json!({
                    "flags": flags,
                    "served_by": label,
                    "canon_source": canon_source,
                })),
            )
                .into_response();
            attach_served_header(&label, reply)
        }
        Ok(Ok(resp)) => (
            StatusCode::BAD_GATEWAY,
            Json(json!({"error": format!("runtime returned {}", resp.status())})),
        )
            .into_response(),
        Ok(Err(e)) => (
            StatusCode::BAD_GATEWAY,
            Json(json!({
                "error": format!(
                    "local runtime unreachable (fact-check uses ONLY the local lane): {e}"
                )
            })),
        )
            .into_response(),
        Err(_) => (
            StatusCode::BAD_GATEWAY,
            Json(json!({
                "error": "local runtime unreachable (fact-check uses ONLY the local lane): timed out waiting for response headers"
            })),
        )
            .into_response(),
    }
}

/// Canon resolution shared by the writing skills (factcheck + template
/// pack): an explicit `canon` body field wins; otherwise, when
/// `auto_default` allows it, the operator's default register is
/// auto-loaded — and a register that exists but cannot be READ is a
/// loud 503, never a silent empty-canon pass (an empty canon would mint
/// clean verdicts / grounded prose from nothing). `auto_default = false`
/// (non-narrative templates) never sniffs the workspace register.
fn resolve_writing_canon(
    body: &Value,
    auto_default: bool,
    canon_cap_chars: usize,
) -> Result<(String, String), Box<axum::response::Response>> {
    let mut canon_source = "none".to_string();
    let canon = match body.get("canon").and_then(Value::as_str) {
        Some(c) if !c.is_empty() => {
            canon_source = "body".to_string();
            c.to_string()
        }
        _ if auto_default => match default_factcheck_canon_path() {
            None => String::new(),
            Some(path) => match std::fs::read_to_string(&path) {
                Ok(text) => {
                    canon_source = format!("default:{}", path.display());
                    text
                }
                Err(e) => {
                    return Err(Box::new((
                        StatusCode::SERVICE_UNAVAILABLE,
                        axum::Json(json!({
                            "error": format!(
                                "default canon register unreadable at {}: {e} — fix the file or pass canon in the body",
                                path.display()
                            ),
                        })),
                    )
                        .into_response()));
                }
            },
        },
        _ => String::new(),
    };
    if canon.chars().count() > canon_cap_chars {
        return Err(Box::new(
            (
                StatusCode::PAYLOAD_TOO_LARGE,
                axum::Json(json!({"error": format!("canon exceeds {canon_cap_chars} chars")})),
            )
                .into_response(),
        ));
    }
    Ok((canon, canon_source))
}

/// skill: writing.template — creator/writing template pack (2026-10-30):
/// structured OUTPUTS, not freeform — short scripts, episode blurbs,
/// sponsor About copy and YouTube SEO packs. ONE call to the node's OWN
/// hosted model (same local-only contract as writing.factcheck: never a
/// tier reroute, never mesh fallback, refused loudly when no local model
/// is declared), parsed as a JSON object and VALIDATED against the
/// template's required field list before it is returned — a violating
/// fill costs the model exactly one corrective retry, then a loud 502,
/// never a silently half-filled pack. Narrative templates
/// (short_script / episode_blurb / youtube_seo) auto-load the operator's
/// canon register, so a fill cannot invent "73% of sightings" folklore
/// stats; the response's canon_source always names what grounded it.
const TEMPLATE_SUBJECT_CAP_CHARS: usize = 12_000;
const TEMPLATE_STYLE_CAP_CHARS: usize = 4_000;

struct TemplateSpec {
    name: &'static str,
    /// Narrative templates: the canon register rides the prompt (and is
    /// attributed in canon_source) unless body canon overrides.
    canon_grounded: bool,
    /// Required string fields the filled JSON must carry (validation).
    fields: &'static [&'static str],
    scaffold: &'static str,
}

const WRITING_TEMPLATES: [TemplateSpec; 4] = [
    TemplateSpec {
        name: "short_script",
        canon_grounded: true,
        fields: &["title", "hook", "script", "outro_cta"],
        scaffold: "You write short YouTube scripts for the Colony X series. Write a 60-90 second script about the SUBJECT using ONLY facts stated in the CANON above — no invented statistics, dates, cast members or episode titles. Keep everything SFW and audience-friendly. Output ONLY a JSON object with string fields: title (episode-style title), hook (first spoken line), script (the full narration), outro_cta (closing call to action). No markdown, no code fences.",
    },
    TemplateSpec {
        name: "episode_blurb",
        canon_grounded: true,
        fields: &["title", "blurb"],
        scaffold: "You write episode blurbs for the Colony X series, grounded strictly in the CANON above — never invent statistics, dates, cast members or episode titles. Keep everything SFW. Output ONLY a JSON object with string fields: title, blurb (2-4 sentences, spoiler-safe). No markdown, no code fences.",
    },
    TemplateSpec {
        name: "sponsor_about",
        canon_grounded: false,
        fields: &["tagline", "about", "reply_note"],
        scaffold: "You write sponsor-friendly channel About copy for a creator. Tone: warm, professional, concrete. Keep everything SFW. Output ONLY a JSON object with string fields: tagline (one line), about (a 4-6 sentence About section), reply_note (a short closing note the sponsor can reply to). No markdown, no code fences.",
    },
    TemplateSpec {
        name: "youtube_seo",
        canon_grounded: true,
        fields: &["title", "description", "tags", "chapters"],
        scaffold: "You write YouTube SEO packs, grounded strictly in the CANON above — never invent statistics, dates or episode titles. Output ONLY a JSON object with string fields: title (70 chars or fewer), description (upload description opening with a 2-sentence summary), tags (comma-separated keywords), chapters (one chapter line per row, '0:00 Opening' style). No markdown, no code fences.",
    },
];

fn writing_template(name: &str) -> Option<&'static TemplateSpec> {
    WRITING_TEMPLATES.iter().find(|t| t.name == name)
}

/// The model's answer, tolerated-fences edition: models love wrapping
/// JSON in ```json blocks — take the first-`{`..last-`}` slice.
fn extract_json_object(text: &str) -> Option<&'_ str> {
    let t = text.trim();
    let stripped = t
        .strip_prefix("```")
        .map(|r| r.trim().trim_start_matches("json").trim())
        .unwrap_or(t);
    let start = stripped.find('{')?;
    let end = stripped.rfind('}')?;
    (end >= start).then(|| &stripped[start..=end])
}

/// Validate a template fill: one JSON object carrying every required
/// field as a non-empty string. Extra fields are tolerated and returned
/// as-is — the required set is the contract, extras are the model's
/// decoration.
fn validate_template_fill(
    content: &str,
    spec: &TemplateSpec,
) -> Result<serde_json::Map<String, Value>, String> {
    let slice = extract_json_object(content)
        .ok_or_else(|| "no JSON object found in the answer".to_string())?;
    let value: Value = serde_json::from_str(slice).map_err(|e| format!("invalid JSON: {e}"))?;
    let map = value
        .as_object()
        .ok_or_else(|| "JSON answer is not an object".to_string())?;
    for field in spec.fields {
        let ok = map
            .get(*field)
            .and_then(Value::as_str)
            .map(|s| !s.trim().is_empty())
            .unwrap_or(false);
        if !ok {
            return Err(format!("missing or empty required field '{field}'"));
        }
    }
    Ok(map.clone())
}

async fn skill_writing_template(
    State(gw): State<Arc<Gateway>>,
    Json(body): Json<Value>,
) -> Response {
    let Some(base) = gw
        .base_model
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "error": "no local model declared ([model] unset) — template fills refuse rather than leave the box",
            })),
        )
            .into_response();
    };
    let Some(tpl_name) = body
        .get("template")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
    else {
        return bad_request("field 'template' is required");
    };
    let Some(spec) = writing_template(tpl_name) else {
        return bad_request(&format!(
            "unknown template '{tpl_name}' — available: {}",
            WRITING_TEMPLATES
                .iter()
                .map(|t| t.name)
                .collect::<Vec<_>>()
                .join(", ")
        ));
    };
    let Some(subject) = body
        .get("subject")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
    else {
        return bad_request("field 'subject' is required");
    };
    if subject.chars().count() > TEMPLATE_SUBJECT_CAP_CHARS {
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(json!({
                "error": format!("subject exceeds {TEMPLATE_SUBJECT_CAP_CHARS} chars")
            })),
        )
            .into_response();
    }
    let style = body
        .get("style")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if style.chars().count() > TEMPLATE_STYLE_CAP_CHARS {
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(json!({
                "error": format!("style exceeds {TEMPLATE_STYLE_CAP_CHARS} chars")
            })),
        )
            .into_response();
    }
    // Canon: narrative templates auto-load the operator's register (same
    // resolution + loud-unreadable rule as factcheck); only a body canon
    // or the register joins the prompt — never both.
    let (canon, canon_source) =
        match resolve_writing_canon(&body, spec.canon_grounded, FACTCHECK_CANON_CAP_CHARS) {
            Ok(pair) => pair,
            Err(resp) => return *resp,
        };
    let mut system = spec.scaffold.to_string();
    system.push_str(
        "\n\nEvery claim must be supported by the canon or the SUBJECT itself — omit rather than invent.",
    );
    let mut user_text = String::new();
    if !canon.is_empty() {
        user_text.push_str("CANON:\n");
        user_text.push_str(&canon);
        user_text.push_str("\n\n");
    }
    user_text.push_str("SUBJECT:\n");
    user_text.push_str(subject);
    if !style.trim().is_empty() {
        user_text.push_str("\n\nSTYLE GUIDANCE:\n");
        user_text.push_str(style);
    }
    let url = format!(
        "{}/v1/chat/completions",
        gw.runtime_url.trim_end_matches('/')
    );
    let label = format!("local:{base}");
    // Two local attempts max: small models often answer wrapped in
    // ```json fences (tolerated above) or drop a field (one corrective
    // retry); a second failure is a loud 502, never a half-filled pack.
    let mut parsed: Option<serde_json::Map<String, Value>> = None;
    let mut last_fail = String::from("no local attempt succeeded");
    for attempt in 0..2 {
        let mut messages = vec![
            json!({"role": "system", "content": system}),
            json!({"role": "user", "content": user_text}),
        ];
        if attempt > 0 {
            messages.push(json!({
                "role": "system",
                "content": format!(
                    "Your previous answer failed validation: {last_fail}. Try again — output ONLY the JSON object with string fields: {}.",
                    spec.fields.join(", ")
                ),
            }));
        }
        let payload = json!({
            "model": base,
            "messages": messages,
            "stream": false,
            "max_tokens": gw.runtime_num_predict.unwrap_or(1024),
        });
        let mut req = crate::http::mesh_http()
            .post(&url)
            .timeout(RUNTIME_HEADERS_TIMEOUT)
            .json(&payload);
        if let Some(k) = &gw.runtime_api_key {
            req = req.bearer_auth(k);
        }
        match req.send().await {
            Ok(resp) if resp.status().is_success() => {
                let value: Value = resp.json().await.unwrap_or(json!({}));
                let content = value
                    .pointer("/choices/0/message/content")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let content = sanitize_runtime_text(content);
                match validate_template_fill(&content, spec) {
                    Ok(fields) => {
                        parsed = Some(fields);
                        break;
                    }
                    Err(msg) => {
                        tracing::info!(attempt, fail = %msg, "template fill attempt failed validation");
                        last_fail = msg;
                    }
                }
            }
            Ok(resp) => last_fail = format!("runtime returned {}", resp.status()),
            Err(e) => last_fail = format!("local runtime unreachable: {e}"),
        }
    }
    let Some(fields) = parsed else {
        return (
            StatusCode::BAD_GATEWAY,
            Json(json!({
                "error": format!("template fill failed after 2 local attempts: {last_fail}"),
            })),
        )
            .into_response();
    };
    gw.metrics.note_served(label.clone());
    let reply = (
        StatusCode::OK,
        Json(json!({
            "template": spec.name,
            "fields": fields,
            "canon_source": canon_source,
            "served_by": label,
        })),
    )
        .into_response();
    attach_served_header(&label, reply)
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
        } else if crate::model_router::auto_tool_tier_merge(
            requested,
            gw.base_model.as_deref(),
            router,
            has_tools,
        ) {
            // Tool tier engaged with NO model rewrite (the tool pick equals
            // the caller's auto model — the pinned same-as-base shape). The
            // think:false merge is the tier's whole remaining effect: skip
            // it and the tool loop thinks, burns the budget in the reasoning
            // channel, and lands empty (the reproduced `{}` incident).
            let model_name = requested.unwrap_or("(none)").to_string();
            body["think"] = serde_json::Value::Bool(false);
            tracing::info!(
                model = %model_name,
                "model router: tool tier no-op rewrite — think:false merged in lane"
            );
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
    // Provider-attribution (reliability 2026-10-30): classify the id that
    // will answer this forward — the node's hosted base reads "local:",
    // anything else reads "runtime:" (LiteLLM tiers proxy upstream). The
    // label is recorded into /health's served_by and stamped on the reply
    // header only on success; a failed lane never claims a provider.
    let pending_label = body
        .get("model")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(|m| classify_served_by(gw.base_model.as_deref(), m));
    // Spec-decode lane (FEATURES §1): decide whether this forward
    // carries the speculative hint. The engine-side draft-and-verify owns
    // the real algorithm (it has the logits); the mesh's lane is honest
    // configure → probe → measure only:
    // - admitted (1): merge the hint;
    // - unprobed (0): spawn the ONE synthetic admission probe — a real
    //   user payload never faces an unproven engine field;
    // - probing (3): no hint this round; the probe result lands next;
    // - refused (2): inert for the process lifetime.
    let mut hinted = false;
    if let Some(sd) = &gw.spec_decode {
        match gw.spec_lane.probe_state() {
            1 => hinted = apply_spec_decode_hint(&mut body, sd, true),
            0 if gw
                .spec_lane
                .probe
                .compare_exchange(
                    0,
                    3,
                    std::sync::atomic::Ordering::SeqCst,
                    std::sync::atomic::Ordering::SeqCst,
                )
                .is_ok() =>
            {
                let probe_gw = gw.clone();
                let tokens = sd.num_speculative_tokens;
                tokio::spawn(async move { spec_admission_probe(&probe_gw, tokens).await });
            }
            _ => {} // probing or refused — no hint this round
        }
    }
    // Chat lane client: ONE pooled instance per process — every chat
    // message used to build a fresh TLS-pinned client (and socket pool,
    // and TCP connect) for the runtime lane. The fail-fast contract is
    // client-level: a dead route refuses at connect fast; a slow/hung
    // RUNTIME is caught by the header-wait bound below instead.
    static CHAT_LANE: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    let client = CHAT_LANE.get_or_init(|| {
        reqwest::Client::builder()
            .use_rustls_tls()
            .connect_timeout(Duration::from_secs(10))
            .build()
            .expect("gateway runtime client builder")
    });
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
                            // Reliability: a fallback answer LEFT the box — label it honestly.
                            let served_label = format!(
                                "mesh:{}",
                                served_by.as_deref().unwrap_or("(unknown peer)")
                            );
                            gw.metrics.note_served(served_label.clone());
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
                            let stream_reply = (
                                StatusCode::OK,
                                [("content-type", "text/event-stream")],
                                sse,
                            )
                                .into_response();
                            attach_served_header(&served_label, stream_reply)
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
                        // Reliability: a fallback answer LEFT the box — label it honestly.
                        let served_label =
                            format!("mesh:{}", served_by.as_deref().unwrap_or("(unknown peer)"));
                        gw.metrics.note_served(served_label.clone());
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
                        let json_reply = (
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
                            .into_response();
                        attach_served_header(&served_label, json_reply)
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
        // Reliability: streaming never reaches the parsed-JSON branch, so
        // the label pinned from the forwarded id is the honest bound here.
        if status.is_success()
            && let Some(label) = &pending_label
        {
            gw.metrics.note_served(label.clone());
        }
        let ct = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("text/event-stream")
            .to_string();
        let stream_reply = (
            status,
            [("content-type", ct.as_str())],
            axum::body::Body::from_stream(resp.bytes_stream()),
        )
            .into_response();
        return match (&pending_label, status.is_success()) {
            (Some(label), true) => attach_served_header(label, stream_reply),
            _ => stream_reply,
        };
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
    let mut served_label: Option<String> = None;
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
        // Reliability: the runtime's own reply is the exact-served truth —
        // label from it, not just the id we forwarded. Successes only.
        if status.is_success() {
            let label = classify_served_by(gw.base_model.as_deref(), &effective_model);
            gw.metrics.note_served(label.clone());
            served_label = Some(label);
        }
        gw.metrics.record_simple(
            &effective_model,
            request_started,
            status.as_u16(),
            body.get("usage"),
            None,
        );
        // Spec-decode lane stats (§1): recorded per forward, split by
        // whether the hint rode along. Inputs are honest-only — a record
        // lands only when status is 2xx AND usage carried completion
        // tokens; streams never reach here.
        if status.is_success()
            && let Some(usage) = body.get("usage")
            && let Some(tokens) = usage.get("completion_tokens").and_then(|v| v.as_u64())
            && tokens > 0
        {
            // Sub-millisecond local round-trips truncate to 0 ms;
            // ceiling at 1 ms so the datum is recorded (0 tokens/ms
            // would be a lie in the other direction via the record
            // guard silently dropping measured completions).
            let wall_ms = (request_started.elapsed().as_millis() as u64).max(1);
            gw.spec_lane.record(hinted, tokens, wall_ms).await;
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
    let json_reply = (status, [("content-type", "application/json")], bytes).into_response();
    match &served_label {
        Some(label) => attach_served_header(label, json_reply),
        None => json_reply,
    }
}

/// ONE synthetic admission probe for the spec-decode lane (FEATURES §1): a
/// minimal completion carrying the hint, then the same request without it —
/// so an unreachable runtime is never misread as a field refusal. Outcomes:
/// admitted (1) — runtime accepted the hinted request;
/// refused (2) — the field alone 4xxed while the clean one succeeded;
/// unprobed (0) — runtime down/misconfigured (lane retries on a later
/// request by re-arming from 3 → 0).
async fn spec_admission_probe(gw: &Arc<Gateway>, tokens: u32) {
    let Some(base) = gw
        .base_model
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    else {
        // No advertised model id to probe against — the lane stays
        // unprobed and therefore never hinted (honest absence).
        gw.spec_lane
            .probe
            .store(0, std::sync::atomic::Ordering::SeqCst);
        return;
    };
    let url = format!(
        "{}/v1/chat/completions",
        gw.runtime_url.trim_end_matches('/')
    );
    let client = crate::http::mesh_http();
    let hinted_body = json!({
        "model": base,
        "messages": [{"role": "user", "content": "."}],
        "max_tokens": 1,
        "spec_decode": {"num_speculative_tokens": tokens},
    });
    let mut clean_body = hinted_body.clone();
    if let Some(obj) = clean_body.as_object_mut() {
        obj.remove("spec_decode");
    }
    let auth = |r: reqwest::RequestBuilder| match &gw.runtime_api_key {
        Some(k) => r.bearer_auth(k),
        None => r,
    };
    let hinted_status = auth(client.post(&url).json(&hinted_body))
        .send()
        .await
        .map(|r| r.status());
    let clean_status = auth(client.post(&url).json(&clean_body))
        .send()
        .await
        .map(|r| r.status());
    let lane = &gw.spec_lane.probe;
    match (hinted_status, clean_status) {
        (Ok(h), _) if h.is_success() => {
            lane.store(1, std::sync::atomic::Ordering::SeqCst);
            tracing::info!(
                target: "spec",
                tokens,
                "spec-decode hint admitted by runtime — hinted lane live"
            );
        }
        (Ok(h), Ok(c)) if h.is_client_error() && c.is_success() => {
            lane.store(2, std::sync::atomic::Ordering::SeqCst);
            tracing::warn!(
                target: "spec",
                tokens,
                hint_status = h.as_u16(),
                clean_status = c.as_u16(),
                "runtime refused the spec-decode hint (healthy without it) — lane inert this process"
            );
        }
        _ => {
            lane.store(0, std::sync::atomic::Ordering::SeqCst);
            tracing::debug!(
                target: "spec",
                "spec-decode admission probe inconclusive (runtime down?) — unprobed, retries later"
            );
        }
    }
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
    fn classify_served_by_shapes_the_three_provider_labels() {
        // Hosted base id → local (stayed on-box).
        assert_eq!(
            classify_served_by(Some("qwen3:8b"), "qwen3:8b"),
            "local:qwen3:8b"
        );
        // A runtime tier id (or unknown downstream variant) → runtime:
        // these are the shapes that can proxy off-box.
        assert_eq!(
            classify_served_by(Some("qwen3:8b"), "laguna-m"),
            "runtime:laguna-m"
        );
        // Mesh fallback answers arrive pre-attributed and pass through.
        assert_eq!(
            classify_served_by(None, "mesh:peer-node-1"),
            "mesh:peer-node-1"
        );
    }

    #[test]
    fn annotate_models_labels_rows_and_names_the_local_model() {
        let mut body = json!({"data": [
            {"id": "qwen3:8b"},
            {"id": "laguna-m"},
            {"id": "minicpm-v4.6:1b"},
        ]});
        annotate_models(&mut body, Some("qwen3:8b"));
        assert_eq!(body["data"][0]["provider"], "local");
        assert_eq!(body["data"][1]["provider"], "runtime");
        assert_eq!(body["data"][2]["provider"], "runtime");
        assert_eq!(body["local_model"], "qwen3:8b");
        // No listed base → no local_model echo, rows still labeled runtime.
        let mut body2 = json!({"data": [{"id": "whatever:7b"}]});
        annotate_models(&mut body2, None);
        assert_eq!(body2["data"][0]["provider"], "runtime");
        assert!(body2.get("local_model").is_none());
    }

    fn fleet_sample(ram_total: u64, ram_avail: u64) -> crate::hw::ResourceSample {
        crate::hw::ResourceSample {
            ram_total_bytes: ram_total,
            ram_available_bytes: ram_avail,
            swap_total_bytes: 0,
            cpu_logical_cores: 4,
            gpus: Vec::new(),
            sampled_at_ms: 0,
        }
    }

    /// The bootstrap echoes our own advertisement back into the peer set:
    /// the census must count SELF once (the live sample), not twice, and
    /// the fleet sums must not double-book this machine.
    #[test]
    fn fleet_census_counts_self_once_and_skips_self_ad() {
        // self: 32/20 GiB; the echo of our own ad says 32/20 too; a real
        // peer: 8/6 GiB. Honest census = 40 total / 26 available.
        let own = Some(fleet_sample(
            32 * 1024 * 1024 * 1024,
            20 * 1024 * 1024 * 1024,
        ));
        let echo = Some(fleet_sample(
            32 * 1024 * 1024 * 1024,
            20 * 1024 * 1024 * 1024,
        ));
        let peer = Some(fleet_sample(8 * 1024 * 1024 * 1024, 6 * 1024 * 1024 * 1024));
        let ads = vec![
            ("self-node".to_string(), echo),
            ("jetson-peer".to_string(), peer),
        ];
        let v = aggregate_fleet_census("self-node", own, ads);
        assert_eq!(v["reporting_nodes"], 2, "self + one peer: {v}");
        let g: u64 = 1024 * 1024 * 1024;
        assert_eq!(v["ram_total_bytes"], 40u64 * g, "{v}");
        assert_eq!(v["ram_available_bytes"], 26u64 * g, "{v}");
        let ids: Vec<&str> = v["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n["node_id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, vec!["self-node", "jetson-peer"], "{v}");
    }

    /// The peer roster excludes the self-ad (the bootstrap echoes our own
    /// ad back — we are not our own peer), carries liveness + routing
    /// facts, and is sorted by node id for stable dashboards.
    #[test]
    fn peers_roster_excludes_self_and_sorts_by_node_id() {
        let mk = |node_id: &str| crate::bootstrap::PeerAd {
            node_id: node_id.into(),
            address: "127.0.0.1:9001".into(),
            rpc_address: "127.0.0.1:9101".into(),
            model_id: "qwen3:8b".into(),
            quant: "int4".into(),
            contribution_percent: 40,
            role: crate::config::NodeRole::Hybrid,
            last_seen_ms: 0,
            resources: None,
            hosted_experts: Some(vec!["exp.a".into()]),
            hosted_expert_count: Some(1),
            discovery_servers: Vec::new(),
            gossip_reputations: None,
        };
        let gate = crate::models::ModelId {
            id: "qwen3:8b".into(),
            quant: "int4".into(),
            source: crate::models::ModelSource::Curated,
        };
        let mut reg = crate::bootstrap::PeerRegistry::new(Some(gate.clone()));
        reg.upsert(mk("self-node"));
        reg.upsert(mk("zz-peer"));
        reg.upsert(mk("aa-peer"));
        reg.record_ping_ok("aa-peer", 12);
        reg.record_ping_fail("zz-peer");
        reg.record_ping_fail("zz-peer");
        reg.record_ping_fail("zz-peer");
        let v = peers_roster(&reg, "self-node");
        let rows = v.as_array().unwrap();
        let ids: Vec<&str> = rows
            .iter()
            .map(|r| r["node_id"].as_str().unwrap())
            .collect();
        assert_eq!(
            ids,
            vec!["aa-peer", "zz-peer"],
            "sorted, self excluded: {v}"
        );
        assert_eq!(rows[0]["state"], "alive");
        assert_eq!(rows[0]["last_rtt_ms"], 12);
        assert_eq!(rows[0]["smoothed_rtt_ms"], 12);
        assert_eq!(rows[0]["hosted_experts"][0], "exp.a");
        assert_eq!(rows[1]["state"], "deprecated");
        assert_eq!(rows[1]["consecutive_failures"], 3);
        // Untested ad without health entry → unknown, no invented rtt.
        let mut lone = crate::bootstrap::PeerRegistry::new(Some(gate));
        lone.upsert(mk("fresh-peer"));
        let v = peers_roster(&lone, "someone-else");
        assert_eq!(v[0]["state"], "unknown");
        assert!(v[0]["last_rtt_ms"].is_null());
        assert!(v[0]["smoothed_rtt_ms"].is_null());
    }

    /// Peers without advertised resources are census-absent (unknown ≠
    /// zero), same rule as placement.
    #[test]
    fn fleet_census_excludes_unknown_resource_ads() {
        let ads = vec![
            (
                "rich-peer".to_string(),
                Some(fleet_sample(
                    10 * 1024 * 1024 * 1024,
                    5 * 1024 * 1024 * 1024,
                )),
            ),
            ("shy-peer".to_string(), None),
        ];
        let v = aggregate_fleet_census("self-node", None, ads);
        assert_eq!(v["reporting_nodes"], 1, "{v}");
        assert_eq!(v["ram_available_bytes"], 5u64 * 1024 * 1024 * 1024, "{v}");
        assert_eq!(v["ram_total_bytes"], 10u64 * 1024 * 1024 * 1024, "{v}");
    }

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
            wallet: None,
            optional_wallet: None,
            metrics: Arc::new(crate::metrics::Metrics::new()),
            spec_decode: None,
            spec_lane: std::sync::Arc::new(SpecLane::default()),
        };
        TestServer::new(gw.router().into_make_service()).expect("test server")
    }

    /// The Phase-4 wallet card rides /status when built: chain-truthful
    /// balance surfaced beside the local mined-wins counters.
    #[tokio::test]
    async fn status_carries_the_live_wallet_card_balance() {
        // Stub chain node: token-gated /balance answering the card's query.
        let app = axum::Router::new().route(
            "/balance",
            axum::routing::get(
                |uri: axum::http::Uri, headers: axum::http::HeaderMap| async move {
                    if headers.get("authorization").and_then(|v| v.to_str().ok())
                        != Some("Bearer tok")
                    {
                        return axum::http::StatusCode::UNAUTHORIZED.into_response();
                    }
                    if uri.query().unwrap_or("").contains("address=jer1pay") {
                        return axum::Json(json!({
                            "exists": true,
                            "address": "jer1pay",
                            "balance_uj": 7_500_000,
                            "balance_jer": 7.5,
                            "available_jer": 7.0,
                        }))
                        .into_response();
                    }
                    axum::http::StatusCode::NOT_FOUND.into_response()
                },
            ),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let mut cfg = crate::config::NodeConfig {
            runtime_url: "http://127.0.0.1:1".to_string(),
            ..crate::config::NodeConfig::default()
        };
        cfg.ensure_identity().expect("test identity");
        let (state, _rx) = MeshState::new(cfg).expect("test mesh state");
        let card = crate::economics::WalletCard {
            rpc_url: format!("http://{addr}"),
            token: Some("tok".into()),
            identity: "jer1pay".into(),
            address: Some("jer1pay".into()),
        };
        let gw = Gateway {
            state: state.clone(),
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
            wallet: Some(Arc::new(card)),
            optional_wallet: None,
            metrics: Arc::new(crate::metrics::Metrics::new()),
            spec_decode: None,
            spec_lane: std::sync::Arc::new(SpecLane::default()),
        };
        let server = TestServer::new(gw.router().into_make_service()).expect("test server");
        let body: Value = server.get("/status").await.json();
        let w = body.get("wallet").expect("wallet section present");
        assert_eq!(w["reachable"], json!(true));
        assert_eq!(w["exists"], json!(true));
        assert_eq!(w["balance_uj"], json!(7_500_000));
        assert_eq!(w["balance_jer"], json!(7.5));
        assert_eq!(w["available_jer"], json!(7.0));
        assert_eq!(w["identity"], json!("jer1pay"));
        assert_eq!(w["by_address"], json!(true));
    }

    /// §9 honest crypto surface: an optional chain surfaces the typed
    /// adapter refusal in /status instead of a unlabeled quiet off.
    #[tokio::test]
    async fn status_types_the_optional_chain_refusal() {
        let mut cfg = crate::config::NodeConfig {
            runtime_url: "http://127.0.0.1:1".to_string(),
            ..crate::config::NodeConfig::default()
        };
        cfg.ensure_identity().expect("test identity");
        cfg.crypto.chain = crate::config::CryptoChain::Ethereum;
        let (state, _rx) = MeshState::new(cfg).expect("test mesh state");
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
            wallet: None,
            optional_wallet: None,
            metrics: Arc::new(crate::metrics::Metrics::new()),
            spec_decode: None,
            spec_lane: std::sync::Arc::new(SpecLane::default()),
        };
        let server = TestServer::new(gw.router().into_make_service()).expect("test server");
        let body: Value = server.get("/status").await.json();
        let c = body.get("crypto").expect("crypto section present");
        assert_eq!(c["chain"], json!("ethereum"));
        assert_eq!(c["mining_enabled"], json!(false));
        assert_eq!(c["wallet_card"], json!(false));
        let note = c["note"].as_str().expect("typed refusal present");
        assert!(note.contains("ethereum"), "note names the chain: {note}");
        assert!(note.contains("disabled"), "note says what is off: {note}");
        // No endpoint configured → the optional-card slot stays honestly
        // empty (Null), not a fake zero.
        assert_eq!(c["optional_wallet"], Value::Null);
    }

    /// §9 optional-chain wallet card, live: a configured endpoint's probe
    /// rides /status next to the (mining-halved) honest note — an
    /// unreachable node answers `{ok:false, error}`, never a fake balance.
    #[tokio::test]
    async fn status_carries_the_live_optional_wallet_card() {
        // Mock Ethereum RPC answering the address-keyed probe.
        let eth_hex = "0x4563918244F40000"; // 5 ETH in wei
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let rpc_addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let app = axum::Router::new().route(
                "/",
                axum::routing::post(move || async move {
                    axum::Json(serde_json::json!({
                        "jsonrpc": "2.0", "id": 1, "result": eth_hex
                    }))
                }),
            );
            axum::serve(listener, app).await.unwrap();
        });

        let cfg_addr = format!("http://{rpc_addr}");
        let mut cfg = crate::config::NodeConfig {
            runtime_url: "http://127.0.0.1:1".to_string(),
            ..crate::config::NodeConfig::default()
        };
        cfg.ensure_identity().expect("test identity");
        cfg.crypto.chain = crate::config::CryptoChain::Ethereum;
        cfg.crypto.wallet_endpoint = Some(cfg_addr.clone());
        cfg.crypto.wallet_address = Some("0x95222290dd7278aa3ddd389cc1e1d165cc4bafe5".to_string());
        let (state, _rx) = MeshState::new(cfg).expect("test mesh state");
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
            wallet: None,
            optional_wallet: crate::economics::OptionalChainWallet::from_crypto(
                &crate::config::CryptoConfig {
                    chain: crate::config::CryptoChain::Ethereum,
                    wallet_endpoint: Some(cfg_addr),
                    wallet_address: Some("0x95222290dd7278aa3ddd389cc1e1d165cc4bafe5".to_string()),
                    ..crate::config::CryptoConfig::default()
                },
            )
            .map(std::sync::Arc::new),
            metrics: Arc::new(crate::metrics::Metrics::new()),
            spec_decode: None,
            spec_lane: std::sync::Arc::new(SpecLane::default()),
        };
        let server = TestServer::new(gw.router().into_make_service()).expect("test server");

        // Healthy chain: the card carries real numbers.
        let body: Value = server.get("/status").await.json();
        let c = body.get("crypto").expect("crypto section present");
        let ow = c["optional_wallet"].as_object().expect("card present");
        assert_eq!(ow["ok"], json!(true), "{ow:?}");
        assert_eq!(ow["unit"], json!("ETH"));
        assert_eq!(ow["amount"], json!(5.0));
        assert_eq!(ow["raw"], json!("5000000000000000000"));
        // The honest note stays: mining is still not implemented for ETH.
        assert!(
            c["note"].as_str().unwrap_or("").contains("disabled"),
            "mining refusal kept next to the live card"
        );

        // Unreachable chain: typed error, no fake balance.
        let mut cfg = crate::config::NodeConfig {
            runtime_url: "http://127.0.0.1:1".to_string(),
            ..crate::config::NodeConfig::default()
        };
        cfg.ensure_identity().expect("test identity");
        cfg.crypto.chain = crate::config::CryptoChain::Ethereum;
        cfg.crypto.wallet_endpoint = Some("http://127.0.0.1:1".to_string());
        cfg.crypto.wallet_address = Some("0x95222290dd7278aa3ddd389cc1e1d165cc4bafe5".to_string());
        let (state, _rx) = MeshState::new(cfg).expect("test mesh state");
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
            wallet: None,
            optional_wallet: crate::economics::OptionalChainWallet::from_crypto(
                &crate::config::CryptoConfig {
                    chain: crate::config::CryptoChain::Ethereum,
                    wallet_endpoint: Some("http://127.0.0.1:1".to_string()),
                    wallet_address: Some("0x95222290dd7278aa3ddd389cc1e1d165cc4bafe5".to_string()),
                    ..crate::config::CryptoConfig::default()
                },
            )
            .map(std::sync::Arc::new),
            metrics: Arc::new(crate::metrics::Metrics::new()),
            spec_decode: None,
            spec_lane: std::sync::Arc::new(SpecLane::default()),
        };
        let server = TestServer::new(gw.router().into_make_service()).expect("test server");
        let body: Value = server.get("/status").await.json();
        let ow = body["crypto"]["optional_wallet"]
            .as_object()
            .expect("card present");
        assert_eq!(ow["ok"], json!(false));
        assert!(
            ow["error"]
                .as_str()
                .unwrap_or_default()
                .contains("unreachable"),
            "typed transport error: {ow:?}"
        );
        assert!(ow.get("amount").is_none(), "no fake balance on error");
    }

    /// Phase 5: the /status reputation section renders the router's VERIFIED
    /// score knowledge (pin-wall stamps only), tiered — the operator sees
    /// who sits penalized and why, without trusting anyone's word.
    #[tokio::test]
    async fn status_reputation_section_reports_verified_subjects() {
        let mut cfg = crate::config::NodeConfig {
            runtime_url: "http://127.0.0.1:1".to_string(),
            ..crate::config::NodeConfig::default()
        };
        cfg.ensure_identity().expect("test identity");
        let (state, _rx) = MeshState::new(cfg).expect("test mesh state");
        state
            .reputation_cache
            .lock()
            .await
            .push(crate::signing::ReputationStamp {
                origin: "47bb0d50e3de1009".to_string(),
                subject: "scored-node".to_string(),
                score: -25,
                stamp_ms: 1_234,
                sig: "sig".to_string(),
            });
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
            wallet: None,
            optional_wallet: None,
            metrics: Arc::new(crate::metrics::Metrics::new()),
            spec_decode: None,
            spec_lane: std::sync::Arc::new(SpecLane::default()),
        };
        let server = TestServer::new(gw.router().into_make_service()).expect("test server");
        let body: Value = server.get("/status").await.json();
        let rep = body["reputation"]["verified_subjects"]
            .as_array()
            .expect("verified subjects listed");
        assert_eq!(rep.len(), 1);
        assert_eq!(rep[0]["subject"], json!("scored-node"));
        assert_eq!(rep[0]["score"], json!(-25));
        assert_eq!(
            rep[0]["penalized"],
            json!(true),
            "negative ⇒ penalized tier"
        );
    }

    /// §12 privacy tiers ride /status: the live level + allowlist size
    /// (None = full_mesh, everything connectable is a hop candidate).
    #[tokio::test]
    async fn status_carries_the_live_privacy_tier() {
        let server = spawn_plain("http://127.0.0.1:1", false);
        let body: Value = server.get("/status").await.json();
        let p = body.get("privacy").expect("privacy section present");
        assert_eq!(p["level"], json!("full_mesh"));
        assert_eq!(p["allowlist_size"], Value::Null);
        assert_eq!(p["trusted_nodes"], json!([]));
    }

    /// The JerithChain side stays quiet — the typed refusal is only for
    /// chains whose adapter does not exist yet.
    #[tokio::test]
    async fn status_leaves_the_real_chain_unannotated() {
        let mut cfg = crate::config::NodeConfig {
            runtime_url: "http://127.0.0.1:1".to_string(),
            ..crate::config::NodeConfig::default()
        };
        cfg.ensure_identity().expect("test identity");
        let (state, _rx) = MeshState::new(cfg).expect("test mesh state");
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
            wallet: None,
            optional_wallet: None,
            metrics: Arc::new(crate::metrics::Metrics::new()),
            spec_decode: None,
            spec_lane: std::sync::Arc::new(SpecLane::default()),
        };
        let server = TestServer::new(gw.router().into_make_service()).expect("test server");
        let body: Value = server.get("/status").await.json();
        let c = body.get("crypto").expect("crypto section present");
        assert_eq!(c["chain"], json!("jerithchain"));
        assert_eq!(c["note"], Value::Null);
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
            wallet: None,
            optional_wallet: None,
            metrics: Arc::new(crate::metrics::Metrics::new()),
            spec_decode: None,
            spec_lane: std::sync::Arc::new(SpecLane::default()),
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
            wallet: None,
            optional_wallet: None,
            metrics: Arc::new(crate::metrics::Metrics::new()),
            spec_decode: None,
            spec_lane: std::sync::Arc::new(SpecLane::default()),
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
            wallet: None,
            optional_wallet: None,
            spec_decode: None,
            spec_lane: std::sync::Arc::new(SpecLane::default()),
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
            wallet: None,
            optional_wallet: None,
            metrics: Arc::new(crate::metrics::Metrics::new()),
            spec_decode: None,
            spec_lane: std::sync::Arc::new(SpecLane::default()),
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
            wallet: None,
            optional_wallet: None,
            metrics: Arc::new(crate::metrics::Metrics::new()),
            spec_decode: None,
            spec_lane: std::sync::Arc::new(SpecLane::default()),
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

    /// The PINNED SAME-AS-BASE tool tier: runtime_tool_model equals the
    /// caller's auto model, so the tier rewrite suppresses itself (no-op)
    /// — and the think:true default would send the tool loop thinking.
    /// The think:false merge must still fire (the reproduced empty-`{}`
    /// tool-loop incident, live 2026-10-16), while the model id is left
    /// exactly as the caller sent it.
    #[tokio::test]
    async fn tool_tier_no_op_rewrite_still_merges_think_false() {
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
        // The live pin: tool tier EQUALS the base model.
        cfg.runtime_tool_model = Some("qwen3:8b".to_string());
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
            base_model: Some("qwen3:8b".to_string()),
            web: None,
            chain_url: None,
            wallet: None,
            optional_wallet: None,
            metrics: Arc::new(crate::metrics::Metrics::new()),
            spec_decode: None,
            spec_lane: std::sync::Arc::new(SpecLane::default()),
        };
        let server = TestServer::new(gw.router().into_make_service()).expect("test server");

        // Auto request (the advertised base model id) carrying tools: the
        // tool tier engages, no rewrite is logged, and think:false lands.
        let tool_request = json!({
            "model": "qwen3:8b",
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
        let res = server
            .post("/v1/chat/completions")
            .json(&tool_request)
            .await;
        assert_eq!(res.status_code(), 200);
        let forwarded: serde_json::Value = serde_json::from_str(&received.lock().await).unwrap();
        assert_eq!(
            forwarded["model"],
            json!("qwen3:8b"),
            "no-op rewrite keeps the model id"
        );
        assert_eq!(
            forwarded["think"],
            json!(false),
            "tool tier disables thinking even without a model rewrite"
        );

        // Plain chat (no tools) on the same base model: untouched, no think
        // key — the caller's model is not tier-routed without a tier.
        let plain = json!({
            "model": "qwen3:8b",
            "messages": [{"role": "user", "content": "hey"}]
        });
        server.post("/v1/chat/completions").json(&plain).await;
        let forwarded: serde_json::Value = serde_json::from_str(&received.lock().await).unwrap();
        assert_eq!(forwarded["model"], json!("qwen3:8b"));
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

    /// §10 observability: /status mirrors OUR bootstrap's /health sybil
    /// tally (stub bootstrap answers nonzero), and degrades to
    /// reachable:false when the bootstrap is down — /status itself stays
    /// 200 either way (same contract as chain_probe).
    #[tokio::test]
    async fn status_mirrors_the_bootstrap_sybil_counter() {
        // Stub bootstrap /health with a nonzero sybil tally.
        async fn stub_health() -> axum::Json<Value> {
            axum::Json(json!({
                "ok": true,
                "sybil": {"young_fraud_reports": 7},
            }))
        }
        let chain_app = axum::Router::new().route("/health", axum::routing::get(stub_health));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, chain_app).await.unwrap() });

        let mut cfg = crate::config::NodeConfig {
            runtime_url: "http://127.0.0.1:1".to_string(),
            bootstrap_url: format!("http://{addr}"),
            ..crate::config::NodeConfig::default()
        };
        cfg.ensure_identity().expect("test identity");
        let (state, _rx) = MeshState::new(cfg).expect("test mesh state");
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
            wallet: None,
            optional_wallet: None,
            metrics: Arc::new(crate::metrics::Metrics::new()),
            spec_decode: None,
            spec_lane: std::sync::Arc::new(SpecLane::default()),
        };
        let server = TestServer::new(gw.router().into_make_service()).unwrap();

        let res = server.get("/status").await;
        assert_eq!(res.status_code(), axum::http::StatusCode::OK);
        let body: Value = res.json();
        let boot = body.get("bootstrap").expect("bootstrap section in /status");
        assert_eq!(
            boot["reachable"], true,
            "stub bootstrap must read reachable"
        );
        assert_eq!(
            boot["young_fraud_reports"], 7,
            "zero-weight tally must mirror the bootstrap: {boot}"
        );
    }

    /// Same mirror, bootstrap DOWN: /status stays 200 and the section
    /// reads reachable:false — a downed bootstrap never takes /status
    /// down with it.
    #[tokio::test]
    async fn status_survives_a_downed_bootstrap_probe() {
        let mut cfg = crate::config::NodeConfig {
            runtime_url: "http://127.0.0.1:1".to_string(),
            // Port 1 is unassigned — the connect fails fast.
            bootstrap_url: "http://127.0.0.1:1".to_string(),
            ..crate::config::NodeConfig::default()
        };
        cfg.ensure_identity().expect("test identity");
        let (state, _rx) = MeshState::new(cfg).expect("test mesh state");
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
            wallet: None,
            optional_wallet: None,
            metrics: Arc::new(crate::metrics::Metrics::new()),
            spec_decode: None,
            spec_lane: std::sync::Arc::new(SpecLane::default()),
        };
        let server = TestServer::new(gw.router().into_make_service()).unwrap();

        let res = server.get("/status").await;
        assert_eq!(res.status_code(), axum::http::StatusCode::OK);
        let body: Value = res.json();
        let boot = body.get("bootstrap").expect("bootstrap section in /status");
        assert_eq!(boot["reachable"], false, "down bootstrap reads {boot}");
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
            wallet: None,
            optional_wallet: None,
            metrics: Arc::new(crate::metrics::Metrics::new()),
            spec_decode: None,
            spec_lane: std::sync::Arc::new(SpecLane::default()),
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
            wallet: None,
            optional_wallet: None,
            metrics: Arc::new(crate::metrics::Metrics::new()),
            spec_decode: None,
            spec_lane: std::sync::Arc::new(SpecLane::default()),
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

    // --- spec-decode lane (FEATURES §1) --------------------------------------

    #[test]
    fn spec_hint_fit_merge_respects_admission_and_zero_budget() {
        let zero = crate::config::SpecDecodeConfig {
            num_speculative_tokens: 0,
        };
        let mut body = json!({"model": "m"});
        // Refused → inert (a user payload never carries the field).
        assert!(!apply_spec_decode_hint(&mut body, &zero, false));
        assert!(body.get("spec_decode").is_none());
        // Zero budget → deliberately inert even when admitted.
        assert!(!apply_spec_decode_hint(&mut body, &zero, true));
        // Admitted + budget → the shape the engine consumes.
        let cfg = crate::config::SpecDecodeConfig {
            num_speculative_tokens: 12,
        };
        assert!(apply_spec_decode_hint(&mut body, &cfg, true));
        assert_eq!(body["spec_decode"]["num_speculative_tokens"], json!(12));
    }

    #[tokio::test]
    async fn spec_lane_stats_publish_speedup_only_with_enough_data() {
        let lane = SpecLane::default();
        // Nothing measured → nulls everywhere.
        let v = lane.status_value(None).await;
        assert_eq!(v["enabled"], false);
        assert!(v["admission"].is_null());
        assert!(v["speedup"].is_null());
        // One lane measured, the other starved → ratio stays withheld.
        for _ in 0..9 {
            lane.record(true, 100, 100).await; // hinted: 1000 tok/s
        }
        assert!(lane.status_value(None).await["speedup"].is_null());
        for _ in 0..5 {
            lane.record(false, 50, 1000).await; // plain: 50 tok/s
        }
        let v = lane
            .status_value(Some(&crate::config::SpecDecodeConfig::default()))
            .await;
        assert_eq!(v["enabled"], true);
        // 1000/50 = 20× (only because both lanes now have ≥ 5 records).
        assert_eq!(v["speedup"], json!(20.0), "{v}");
        // A zero-completion record carries nothing honest into the stats.
        let before = {
            let s = lane.hinted.lock().await.clone();
            s.requests
        };
        lane.record(true, 0, 100).await;
        let after = { lane.hinted.lock().await.clone().requests };
        assert_eq!(before, after);
    }

    #[tokio::test]
    async fn spec_lane_probes_admits_hints_and_measures() {
        // Runtime: accepts everything; records bodies so we can prove the
        // hint rides forward 2+, never the unprobed forward 1.
        let bodies = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let landed = bodies.clone();
        let runtime = axum::Router::new().route(
            "/v1/chat/completions",
            axum::routing::post(move |body: String| {
                let landed = landed.clone();
                async move {
                    landed.lock().await.push(body);
                    axum::Json(json!({
                        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                        "usage": {"completion_tokens": 7}
                    }))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, runtime).await.unwrap() });

        let mut cfg = crate::config::NodeConfig {
            runtime_url: format!("http://{addr}"),
            ..crate::config::NodeConfig::default()
        };
        cfg.ensure_identity().expect("test identity");
        let (state, _rx) = MeshState::new(cfg).expect("test mesh state");
        let gw = Gateway {
            state,
            runtime_url: format!("http://{addr}"),
            runtime_api_key: None,
            runtime_num_predict: None,
            miner: None,
            mesh_fallback: None,
            guard: None,
            model_router: None,
            base_model: Some("spec-test-model".to_string()),
            web: None,
            chain_url: None,
            wallet: None,
            optional_wallet: None,
            spec_decode: Some(crate::config::SpecDecodeConfig {
                num_speculative_tokens: 4,
            }),
            spec_lane: std::sync::Arc::new(SpecLane::default()),
            metrics: Arc::new(crate::metrics::Metrics::new()),
        };
        let lane = gw.spec_lane.clone();
        let server = TestServer::new(gw.router().into_make_service()).expect("server");

        // Forward 1: unprobed → spawns the probe, carries NO hint.
        let res = server
            .post("/v1/chat/completions")
            .json(&json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]}))
            .await;
        assert!(res.status_code().is_success());
        // The probe lands asynchronously (one synthetic hinted round-trip).
        for _ in 0..50 {
            if lane.probe_state() == 1 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        assert_eq!(
            lane.probe_state(),
            1,
            "probe should admit against this mock"
        );
        // Forward 2+: admitted → the hint rides.
        let res = server
            .post("/v1/chat/completions")
            .json(&json!({"model": "m", "messages": [{"role": "user", "content": "again"}]}))
            .await;
        assert!(res.status_code().is_success());
        // Index order is racy (the probe can land before/after forward 1):
        // key on content instead. The "hi" forward ran BEFORE admission →
        // clean; the "again" forward ran after → hinted.
        let seen = bodies.lock().await;
        let hi = seen
            .iter()
            .find(|b| b.contains("hi"))
            .expect("forward 1 recorded");
        let again = seen
            .iter()
            .find(|b| b.contains("again"))
            .expect("forward 2 recorded");
        assert!(
            !hi.contains("spec_decode"),
            "unprobed forward stays clean: {hi}"
        );
        assert!(
            again.contains("spec_decode"),
            "admitted forward carries the hint: {again}"
        );
        // Stats landed on BOTH lanes (usage telemetry is present).
        let status: Value = server.get("/status").await.json();
        assert_eq!(status["spec_decode"]["admission"], "accepted");
        assert_eq!(status["spec_decode"]["num_speculative_tokens"], 4);
        assert_eq!(status["spec_decode"]["hinted"]["requests"], json!(1));
        assert_eq!(status["spec_decode"]["plain"]["requests"], json!(1));
        assert!(
            status["spec_decode"]["speedup"].is_null(),
            "one datum is not a verdict"
        );
    }

    #[tokio::test]
    async fn spec_lane_probe_refusal_gates_the_hint_off_user_traffic() {
        // Runtime that REFUSES unknown body fields (the drop_params-less
        // strict engine shape) but serves the clean request fine.
        let runtime = axum::Router::new().route(
            "/v1/chat/completions",
            axum::routing::post(move |body: String| async move {
                if body.contains("spec_decode") {
                    axum::http::StatusCode::BAD_REQUEST.into_response()
                } else {
                    axum::Json(json!({
                        "choices": [{"message": {"role": "assistant", "content": "ok"}}]
                    }))
                    .into_response()
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, runtime).await.unwrap() });

        let mut cfg = crate::config::NodeConfig {
            runtime_url: format!("http://{addr}"),
            ..crate::config::NodeConfig::default()
        };
        cfg.ensure_identity().expect("test identity");
        let (state, _rx) = MeshState::new(cfg).expect("test mesh state");
        let gw = Gateway {
            state,
            runtime_url: format!("http://{addr}"),
            runtime_api_key: None,
            runtime_num_predict: None,
            miner: None,
            mesh_fallback: None,
            guard: None,
            model_router: None,
            base_model: Some("spec-test-model".to_string()),
            web: None,
            chain_url: None,
            wallet: None,
            optional_wallet: None,
            spec_decode: Some(crate::config::SpecDecodeConfig::default()),
            spec_lane: std::sync::Arc::new(SpecLane::default()),
            metrics: Arc::new(crate::metrics::Metrics::new()),
        };
        let lane = gw.spec_lane.clone();
        let server = TestServer::new(gw.router().into_make_service()).expect("server");
        let _ = server
            .post("/v1/chat/completions")
            .json(&json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]}))
            .await;
        for _ in 0..50 {
            if lane.probe_state() == 2 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        assert_eq!(
            lane.probe_state(),
            2,
            "field-refusal must lock the lane inert"
        );
        // Real user traffic NEVER carries the field once refused.
        let status: Value = server.get("/status").await.json();
        assert_eq!(status["spec_decode"]["admission"], "refused");
        // And subsequent forwards stay clean (request #2 runs after the
        // probe locked the lane; the mock would have 400'd a hinted one
        // — success here proves the merge stayed off).
        let res = server
            .post("/v1/chat/completions")
            .json(&json!({"model": "m", "messages": [{"role": "user", "content": "again"}]}))
            .await;
        assert!(res.status_code().is_success());
    }

    #[test]
    fn factcheck_canon_register_resolution_prefers_env_then_workspace_default() {
        let dir = std::env::temp_dir().join(format!("jerith-fc-canon-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let override_file = dir.join("register.md");
        std::fs::write(&override_file, "# canon\n").unwrap();
        let home_plain = dir.join("home-without-register");
        std::fs::create_dir_all(&home_plain).unwrap();
        let home_with = dir.join("home-with-register");
        let default_reg = home_with.join("workspace").join("COLONY-X-CANON.md");
        std::fs::create_dir_all(home_with.join("workspace")).unwrap();
        std::fs::write(&default_reg, "# canon\n").unwrap();

        // A real override wins over whatever the home workspace holds.
        assert_eq!(
            canon_register_path(Some(override_file.to_str().unwrap()), &home_with).as_deref(),
            Some(override_file.as_path())
        );
        // A broken override hint falls through to the workspace default.
        let missing = dir.join("missing.md");
        assert_eq!(
            canon_register_path(Some(missing.to_str().unwrap()), &home_with).as_deref(),
            Some(default_reg.as_path())
        );
        // Whitespace-only is "not set" — same as absent, NOT a parse
        // error, so a stray export can't silently disable the register.
        assert_eq!(
            canon_register_path(Some("   "), &home_with).as_deref(),
            Some(default_reg.as_path())
        );
        // No override + no register → no default canon (never fake one).
        assert_eq!(canon_register_path(None, &home_plain), None);

        std::fs::remove_dir_all(&dir).ok();
    } // NOTE: mutates JERITH_FACTCHECK_CANON — a var unique to the
    // factcheck lane (no other test reads it); written and removed within
    // this single test, so ordering is guaranteed on its own thread.
    // (The broad registry of canon resolution rules is list-style;
    // combine rather than parallelize.)    /// Stub runtime + a handler lane for canon attribution: the shared
    /// setup — one stub runtime (per-case) + a canon register for the
    /// mixed tests.
    #[tokio::test]
    async fn factcheck_handler_resolves_canon_and_fails_loud_on_broken_register() {
        use std::os::unix::fs::PermissionsExt as _;

        let dir = std::env::temp_dir().join(format!("jerith-fc-handler-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let register = dir.join("register.md");
        std::fs::write(
            &register,
            "CANON row: Rafey is a pink husky-puppy. Rocky is a raccoon.\n",
        )
        .unwrap();

        // Stub local runtime: records what it was handed, answers CLEAN —
        // attribution (body vs registry) is proven by canon_source, and
        // the recorded request proves the canon actually flowed to the
        // model, not just into the response label.
        let bodies: std::sync::Arc<tokio::sync::Mutex<Vec<String>>> =
            std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let lock = bodies.clone();
        tokio::spawn(async move {
            axum::serve(
                listener,
                axum::Router::new().route(
                    "/v1/chat/completions",
                    axum::routing::post(move |body: String| {
                        let lock = lock.clone();
                        async move {
                            lock.lock().await.push(body);
                            axum::Json(json!({
                                "choices": [{"message": {
                                    "role": "assistant",
                                    "content": "CLEAN"
                                }}]
                            }))
                            .into_response()
                        }
                    }),
                ),
            )
            .await
            .ok();
        });

        let runtime_url = format!("http://{addr}");
        let runtime_url: &'static str = Box::leak(runtime_url.into_boxed_str());
        let mut cfg = crate::config::NodeConfig {
            runtime_url: runtime_url.to_string(),
            ..crate::config::NodeConfig::default()
        };
        cfg.ensure_identity().expect("test identity");
        let (state, _rx) = MeshState::new(cfg).expect("test mesh state");
        let gw = Gateway {
            state,
            runtime_url: runtime_url.to_string(),
            runtime_api_key: None,
            runtime_num_predict: None,
            miner: None,
            mesh_fallback: None,
            guard: None,
            model_router: None,
            base_model: Some("factcheck-test-model".to_string()),
            web: None,
            chain_url: None,
            wallet: None,
            optional_wallet: None,
            spec_decode: None,
            spec_lane: std::sync::Arc::new(SpecLane::default()),
            metrics: Arc::new(crate::metrics::Metrics::new()),
        };
        let server = TestServer::new(gw.router().into_make_service()).expect("server");

        // 1. Missing draft → 400 before any canon work.
        let res = server
            .post("/v1/skills/writing/factcheck")
            .json(&json!({}))
            .await;
        assert_eq!(res.status_code(), axum::http::StatusCode::BAD_REQUEST);

        // 2. Env register auto-loads when the caller omits canon: the
        // response ATTRIBUTES the register, and the stub received the
        // register content in the prompt (not just a postal label).
        unsafe {
            std::env::set_var("JERITH_FACTCHECK_CANON", register.display().to_string());
        }
        let res = server
            .post("/v1/skills/writing/factcheck")
            .json(&json!({"draft": "Rafey is 73% brown-fox now."}))
            .await;
        assert_eq!(res.status_code(), axum::http::StatusCode::OK);
        let body: Value = res.json();
        let canon_source = body["canon_source"].as_str().expect("canon_source present");
        assert_eq!(
            canon_source,
            format!("default:{}", register.display()),
            "register must be the attributed canon source"
        );
        assert_eq!(body["served_by"], "local:factcheck-test-model");
        assert_eq!(body["flags"], "CLEAN");
        // Scoped borrow: the stub's tokio::Mutex must be FREE again
        // before the next call, or the next stub invocation self-deadlocks
        // and the handler eats its full header timeout (live bug we hit).
        {
            let guard = bodies.lock().await;
            let handed = guard.last().expect("one runtime call");
            assert!(
                handed.contains("CANON row: Rafey is a pink husky-puppy."),
                "the register content must reach the model prompt"
            );
        }

        // 3. An explicit body canon OVERRIDES the file — and the file is
        // NOT forwarded (one canon source, declared honestly).
        let res = server
            .post("/v1/skills/writing/factcheck")
            .json(&json!({
                "draft": "Rocky is a raccoon.",
                "canon": "Finding Nemo is set in the Great Barrier Reef."
            }))
            .await;
        assert_eq!(res.status_code(), axum::http::StatusCode::OK);
        let body: Value = res.json();
        assert_eq!(body["canon_source"], "body");
        {
            let guard = bodies.lock().await;
            let handed = guard.last().expect("second call");
            assert!(
                handed.contains("Great Barrier Reef"),
                "body canon wins: {handed}"
            );
            assert!(
                !handed.contains("husky-puppy"),
                "the file register must NOT also ride along"
            );
        }

        // 4. A register that exists but cannot be READ is a loud 503,
        // never a silent empty-canon CLEAN.
        std::fs::set_permissions(&register, std::fs::Permissions::from_mode(0o000)).unwrap();
        let res = server
            .post("/v1/skills/writing/factcheck")
            .json(&json!({"draft": "still checked?"}))
            .await;
        assert_eq!(
            res.status_code(),
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "unreadable register must fail loud"
        );
        let body: Value = res.json();
        assert!(
            body["error"].as_str().unwrap().contains("unreadable"),
            "got: {body}"
        );
        std::fs::set_permissions(&register, std::fs::Permissions::from_mode(0o644)).unwrap();

        // 5. Draft over the cap → 413 before an embed round-trip.
        let big = "d".repeat(FACTCHECK_DRAFT_CAP_CHARS + 1);
        let res = server
            .post("/v1/skills/writing/factcheck")
            .json(&json!({"draft": big}))
            .await;
        assert_eq!(res.status_code(), axum::http::StatusCode::PAYLOAD_TOO_LARGE);

        unsafe {
            std::env::remove_var("JERITH_FACTCHECK_CANON");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The template-pack handler: contract lanes — routing/validation
    /// errors, ONE corrective retry then success, and the loud 502 when
    /// both local attempts answer non-JSON. Deliberately no env-var
    /// mutation: text canon rides the body (env auto-load is the shared
    /// `resolve_writing_canon` path, already covered handler-side by the
    /// factcheck test), so this test never races the factcheck test's
    /// JERITH_FACTCHECK_CANON window under parallel execution.
    #[tokio::test]
    async fn template_handler_validates_and_names_canon() {
        // Stub: call 1 answers fence-wrapped garbage-free fill missing a
        // field? NO — call 1 answers prose (parse failure provoking the
        // corrective retry), call 2 answers a fence-wrapped valid fill.
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let handed: Arc<tokio::sync::Mutex<Vec<String>>> =
            Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let calls1 = calls.clone();
        let handed1 = handed.clone();
        tokio::spawn(async move {
            axum::serve(
                listener,
                axum::Router::new().route(
                    "/v1/chat/completions",
                    axum::routing::post(move |body: String| {
                        let calls = calls1.clone();
                        let handed = handed1.clone();
                        async move {
                            let n = calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            handed.lock().await.push(body.clone());
                            let fill = if n == 0 {
                                // Prose: provokes the corrective retry.
                                "Sure — I'll write that shortly."
                            // The runtime payload never carries the template
                            // id (only scaffold text rides) — branch on the
                            // sponsor scaffold's distinctive field name.
                            } else if body.contains("reply_note") {
                                "```json\n{\"tagline\": \"One diner, infinite rifts.\", \"about\": \"About the Rift-R Ranch diner channel.\", \"reply_note\": \"Ask us anything about tiers.\"}\n```"
                            } else {
                                "```json\n{\"title\": \"Rift-Ranch Shift\", \"hook\": \"The diner sign buzzed on.\", \"blurb\": \"One shift at the Rift-R Ranch diner.\"}\n```"
                            };
                            axum::Json(json!({
                                "choices": [{"message": {"role": "assistant", "content": fill}}]
                            }))
                            .into_response()
                        }
                    }),
                ),
            )
            .await
            .ok();
        });
        let runtime_url = format!("http://{addr}");
        let runtime_url: &'static str = Box::leak(runtime_url.into_boxed_str());
        let mut cfg = crate::config::NodeConfig {
            runtime_url: runtime_url.to_string(),
            ..crate::config::NodeConfig::default()
        };
        cfg.ensure_identity().expect("test identity");
        let (state, _rx) = MeshState::new(cfg).expect("test mesh state");
        let gw = Gateway {
            state,
            runtime_url: runtime_url.to_string(),
            runtime_api_key: None,
            runtime_num_predict: None,
            miner: None,
            mesh_fallback: None,
            guard: None,
            model_router: None,
            base_model: Some("template-test-model".to_string()),
            web: None,
            chain_url: None,
            wallet: None,
            optional_wallet: None,
            spec_decode: None,
            spec_lane: std::sync::Arc::new(SpecLane::default()),
            metrics: Arc::new(crate::metrics::Metrics::new()),
        };
        let server = TestServer::new(gw.router().into_make_service()).expect("server");

        // 1. Unknown template → 400 with the available list.
        let res = server
            .post("/v1/skills/writing/template")
            .json(&json!({"template": "haiku", "subject": "x"}))
            .await;
        assert_eq!(res.status_code(), axum::http::StatusCode::BAD_REQUEST);
        let body: Value = res.json();
        assert!(
            body["error"]
                .as_str()
                .unwrap()
                .contains("unknown template 'haiku'"),
            "got: {body}"
        );
        // 2. Missing subject → 400.
        let res = server
            .post("/v1/skills/writing/template")
            .json(&json!({"template": "episode_blurb"}))
            .await;
        assert_eq!(res.status_code(), axum::http::StatusCode::BAD_REQUEST);
        // 3. Over-cap subject → 413 before any round-trip.
        let big = "s".repeat(TEMPLATE_SUBJECT_CAP_CHARS + 1);
        let res = server
            .post("/v1/skills/writing/template")
            .json(&json!({"template": "episode_blurb", "subject": big}))
            .await;
        assert_eq!(res.status_code(), axum::http::StatusCode::PAYLOAD_TOO_LARGE);

        // 4. Prose answer → one corrective retry → a VALID fill, with the
        // body canon attributed and the canon text actually in the model
        // prompt (both calls recorded).
        let res = server
            .post("/v1/skills/writing/template")
            .json(&json!({
                "template": "episode_blurb",
                "subject": "Rafey's first shift at the Rift-Ranch diner",
                "canon": "CANON row: Rafey is a pink husky-puppy."
            }))
            .await;
        assert_eq!(res.status_code(), axum::http::StatusCode::OK, "retry lane");
        let body: Value = res.json();
        assert_eq!(body["template"], "episode_blurb");
        assert_eq!(body["canon_source"], "body");
        assert_eq!(body["served_by"], "local:template-test-model");
        assert_eq!(body["fields"]["title"], "Rift-Ranch Shift");
        assert_eq!(
            body["fields"]["blurb"],
            "One shift at the Rift-R Ranch diner."
        );
        {
            let guard = handed.lock().await;
            assert_eq!(guard.len(), 2, "exactly ONE corrective retry");
            let first = &guard[0];
            assert!(first.contains("CANON row: Rafey is a pink husky-puppy."));
            let second = &guard[1];
            assert!(
                second.contains("failed validation: no JSON object found"),
                "the retry must carry the corrective nudge: {second}"
            );
        }

        // 5. Non-grounded template (sponsor_about), no canon → canon is
        // honestly "none" and NO CANON section is shipped to the model.
        let res = server
            .post("/v1/skills/writing/template")
            .json(&json!({
                "template": "sponsor_about",
                "subject": "the colony-x channel",
                "canon": "sponsor-visible tone rules: concrete, warm"
            }))
            .await;
        let body: Value = res.json();
        assert!(
            body.get("template").is_some(),
            "sponsor fill must succeed: {body}"
        );
        assert_eq!(body["canon_source"], "body");
        {
            let guard = handed.lock().await;
            assert!(
                guard.last().unwrap().contains("sponsor-visible tone rules"),
                "an explicit body canon rides for any template"
            );
        }
        let res = server
            .post("/v1/skills/writing/template")
            .json(&json!({"template": "sponsor_about", "subject": "again"}))
            .await;
        let body: Value = res.json();
        assert_eq!(body["canon_source"], "none");
        {
            let guard = handed.lock().await;
            assert!(
                !guard.last().unwrap().contains("CANON:"),
                "non-grounded + no canon must not ship a CANON section"
            );
        }
    }

    /// Both attempts answer non-JSON: loud 502 after exactly two local
    /// calls, never a quietly half-filled pack.
    #[tokio::test]
    async fn template_handler_fails_loud_when_both_fill_attempts_are_non_json() {
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let calls1 = calls.clone();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(
                listener,
                axum::Router::new().route(
                    "/v1/chat/completions",
                    axum::routing::post(move |_body: String| {
                        let calls = calls1.clone();
                        async move {
                            calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            axum::Json(json!({
                                "choices": [{"message": {"role": "assistant", "content": "no braces here"}}]
                            }))
                            .into_response()
                        }
                    }),
                ),
            )
            .await
            .ok();
        });
        let runtime_url = format!("http://{addr}");
        let mut cfg = crate::config::NodeConfig {
            runtime_url: runtime_url.clone(),
            ..crate::config::NodeConfig::default()
        };
        cfg.ensure_identity().expect("test identity");
        let (state, _rx) = MeshState::new(cfg).expect("test mesh state");
        let gw = Gateway {
            state,
            runtime_url,
            runtime_api_key: None,
            runtime_num_predict: None,
            miner: None,
            mesh_fallback: None,
            guard: None,
            model_router: None,
            base_model: Some("template-test-model".to_string()),
            web: None,
            chain_url: None,
            wallet: None,
            optional_wallet: None,
            spec_decode: None,
            spec_lane: std::sync::Arc::new(SpecLane::default()),
            metrics: Arc::new(crate::metrics::Metrics::new()),
        };
        let server = TestServer::new(gw.router().into_make_service()).expect("server");
        let res = server
            .post("/v1/skills/writing/template")
            .json(&json!({
                "template": "short_script",
                "subject": "the diner.Tables — keep it strict:         ep 2",
            }))
            .await;
        assert_eq!(res.status_code(), axum::http::StatusCode::BAD_GATEWAY);
        let body: Value = res.json();
        assert!(
            body["error"]
                .as_str()
                .unwrap()
                .contains("failed after 2 local attempts"),
            "got: {body}"
        );
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "exactly two local attempts, no silent pass-through"
        );
    }
}
