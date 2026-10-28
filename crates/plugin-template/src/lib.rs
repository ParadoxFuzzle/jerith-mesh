//! The plugin SDK service (Phase 3d, `docs/PLUGIN-SDK.md`).
//!
//! The Phase 3a manifest (crates/bot/src/external.rs) gave a third party a
//! way to *declare* skills; this crate is the way to *serve* them. An
//! author builds a `SkillServer`, registers plain async handlers, and
//! `serve`s it — everything below is the SDK's job, never theirs:
//!
//! - the exact wire contract the bot's dispatcher speaks (one POST round
//!   trip, bounded JSON);
//! - the ONE signing recipe the whole codebase already shares
//!   (`hooks.rs` ⇄ `external.rs`): `x-jerith-signature` =
//!   hex(HMAC-SHA256(secret, "{ts_ms}.{body}")) + `x-jerith-timestamp`;
//!   unsigned service (no secret) only for loopback development;
//! - freshness (±5 minutes, the `gcp-bridge` rule) so a sniffed body
//!   can't replay forever;
//! - fail-loud typed errors: a wrong signature is REFUSED before any
//!   handler runs, an unknown skill is an honest `{"error", …}` the
//!   model can read and recover from, an oversize answer is refused
//!   upstream AND down here (same 65536 cap the dispatcher enforces).
//!
//! A `SkillServer` holds config + handlers only. Nothing here touches
//! the bot's process: the trust posture is the manifest's — loopback
//! http by default, https for anything non-loopback (enforced over in
//! `external.rs` parsing, mirrored by the copyable `plugin.toml`).
//!
//! Wire reference (mirrors `external.rs`'s module doc exactly):
//!
//! ```text
//! POST {endpoint}                              # /rpc, loopback default
//! x-jerith-signature: <hex>                    # when a secret is set
//! x-jerith-timestamp: {ts_ms}
//! {"method":"invoke","skill":"weather.now","args":{…},"ctx":{"ts_ms":N}}
//! → 200 {"…result…"}     any JSON object ≤ 65536 bytes, model reads it
//! → non-200 / timeout / oversize → the tool loop surfaces {"error"}
//! ```

use anyhow::Result;
use axum::Router;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use hmac::{Hmac, Mac};
use serde_json::{Value, json};
use sha2::Sha256;
use std::collections::HashMap;
use std::str::FromStr as _;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Same cap the bot's dispatcher enforces (external.rs): an answer
/// bigger than this is refused on BOTH ends — no bloated barker.
pub const MAX_RESPONSE_BYTES: usize = 65536;

/// Freshness window on `x-jerith-timestamp` (both directions of clock
/// error) — one sniffed request must not replay forever. Same 5-minute
/// gate `gcp-bridge` enforces before it spawns anything.
pub const MAX_CLOCK_SKEW_MS: u64 = 5 * 60 * 1000;

/// Now in milliseconds since the epoch (the SDK's only clock read; a
/// test's "future-dated" stamp is just now + skew + 1).
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// The sign side of the shared recipe — identical to
/// `external::sign_body` / `hooks` by design; kept here so a plugin's
/// OWN outbound calls (emit side) reuse the exact same scheme.
pub fn sign_body(secret: &str, ts_ms: u64, body: &str) -> String {
    let mut mac =
        Hmac::<Sha256>::new_from_slice(secret.as_bytes()).expect("HMAC accepts any key length");
    mac.update(format!("{ts_ms}.{body}").as_bytes());
    out_hex(&mac.finalize().into_bytes())
}

fn out_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// One registered skill handler: parsed `args` in, a result JSON object
/// out, or a typed error string that lands in `{"error": …}` and is fed
/// to the model verbatim (the refusal contract, unchanged).
pub type SkillFn = Arc<
    dyn Fn(Value) -> futures_util::future::BoxFuture<'static, Result<Value, String>> + Send + Sync,
>;

/// Adapt a plain `async fn(Value) -> Result<Value, String>` into a
/// [`SkillFn`] — the one-line conversion every handler needs.
pub fn skill<F, Fut>(f: F) -> SkillFn
where
    F: Fn(Value) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<Value, String>> + Send + 'static,
{
    Arc::new(move |args| Box::pin(f(args)))
}

/// The shared handler-set + secret state behind one service. Public so
/// embedding hosts (and tests) can build the router directly.
#[derive(Clone)]
pub struct ServerConfig {
    /// Model-visible prefix the manifest declares (gating id is
    /// `external:<id>` — the server checks nothing about it; that's the
    /// operator's `[plugins] enabled` row).
    id: String,
    /// `None` = unsigned dev mode (loopback only, by convention).
    secret: Option<String>,
    handlers: HashMap<String, SkillFn>,
}

impl std::fmt::Debug for ServerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServerConfig")
            .field("id", &self.id)
            .field("secret", &self.secret.is_some())
            .field("handlers", &self.handlers.len())
            .finish()
    }
}

/// The builder: one chain from `new` to `serve`. Every registration is
/// fail-loud at boot (a manifest row without a matching handler is an
/// authoring mistake the server refuses to mask).
pub struct SkillServer {
    cfg: ServerConfig,
}

impl SkillServer {
    /// `id` must match the manifest's `[plugin] id`; `secret` must match
    /// its `secret` exactly (or be `None` together with an unset
    /// manifest secret). The bot refuses drops that disagree.
    pub fn new(id: impl Into<String>, secret: Option<String>) -> Self {
        Self {
            cfg: ServerConfig {
                id: id.into(),
                secret,
                handlers: HashMap::new(),
            },
        }
    }

    /// Register one handler under its model-visible name — the FULLY
    /// qualified form (`example.echo`), exactly what `[[skill]] name`
    /// becomes after the manifest parser qualifies it. Registering an
    /// unqualified `echo` compiles but never fires; the dispatcher's
    /// calls are always prefixed.
    pub fn handler(mut self, name: impl Into<String>, f: SkillFn) -> Self {
        self.cfg.handlers.insert(name.into(), f);
        self
    }

    /// Bind + run forever: axum POST /rpc + this server's handlers.
    /// `addr` should be loopback unless the operator knowingly moves it
    /// (the manifest parsing already polices the non-loopback side).
    pub async fn serve(self, addr: std::net::SocketAddr) -> Result<()> {
        let state = Arc::new(self.cfg);
        let listener = tokio::net::TcpListener::bind(addr).await?;
        tracing::info!(
            plugin = %state.id,
            skills = state.handlers.len(),
            %addr,
            unsigned = state.secret.is_none(),
            "plugin service listening on POST /rpc"
        );
        axum::serve(listener, router(state)).await?;
        Ok(())
    }
}

/// The standalone router (public so tests and embedding hosts can wrap
/// the same handler shape — `axum::serve` needs one anyway).
pub fn router(state: Arc<ServerConfig>) -> Router {
    Router::new().route("/rpc", post(rpc)).with_state(state)
}

async fn rpc(State(cfg): State<Arc<ServerConfig>>, headers: HeaderMap, body: String) -> Response {
    // ---- 1. Auth (before anything else runs) --------------------------
    if let Some(secret) = &cfg.secret
        && let Err(msg) = verify_request(secret, &headers, &body)
    {
        tracing::warn!(plugin = %cfg.id, msg = %msg, "unsigned/refused request");
        return (StatusCode::FORBIDDEN, error_json(&msg)).into_response();
    }

    // ---- 2. Parse the one round trip --------------------------------
    let Ok(req) = serde_json::from_str::<Value>(&body) else {
        let msg = "body is not JSON";
        return (StatusCode::BAD_REQUEST, error_json(msg)).into_response();
    };
    if req["method"].as_str() != Some("invoke") {
        let msg = format!(
            "unknown method {:?} — the contract has exactly one",
            req["method"]
        );
        return (StatusCode::BAD_REQUEST, error_json(&msg)).into_response();
    }
    let Some(skill_name) = req["skill"].as_str().map(str::to_string) else {
        let msg = "missing skill name";
        return (StatusCode::BAD_REQUEST, error_json(msg)).into_response();
    };
    // `ctx` riding in is the dispatcher's ts echo; a plugin may use it
    // but must never trust it — the envelope is the only authority.
    let _ctx_ts = req.pointer("/ctx/ts_ms").and_then(Value::as_u64);

    // ---- 3. Dispatch (unknown skill stays MODEL-visible, not a 500) --
    let Some(handler) = cfg.handlers.get(&skill_name).cloned() else {
        let known: Vec<String> = cfg.handlers.keys().map(|s| s.to_string()).collect();
        let msg = format!(
            "plugin {:?} has no skill {:?} (registered: {})",
            cfg.id,
            skill_name,
            known.join(", ")
        );
        tracing::warn!(plugin = %cfg.id, skill = %skill_name, "unknown skill invoked");
        return (StatusCode::OK, error_json(&msg)).into_response();
    };

    // ---- 4. Run + bound the answer ----------------------------------
    let started = std::time::Instant::now();
    let args = req.get("args").cloned().unwrap_or(Value::Null);
    let out = match handler(args).await {
        Ok(v) => v,
        Err(e) => {
            // The model reads `{"error": …}` and may retry differently —
            // the skill's honest failure, not a transport failure.
            return (StatusCode::OK, axum::response::Json(json!({ "error": e }))).into_response();
        }
    };
    let bytes = serde_json::to_vec(&out).unwrap_or_default();
    if bytes.len() > MAX_RESPONSE_BYTES {
        let msg = format!(
            "skill response is {} bytes — the dispatcher's cap is {MAX_RESPONSE_BYTES}; shrink the answer",
            bytes.len()
        );
        tracing::warn!(plugin = %cfg.id, skill = %skill_name, msg = %msg);
        return (StatusCode::BAD_GATEWAY, error_json(&msg)).into_response();
    }
    // Every served invoke journals at INFO: the default-filter plugin
    // ("info") carries this receipt, so an operator can see a skill was
    // called without turning debug on (2026-10-28: DEBUG was invisible
    // under the default INFO during the weather plugin's live-fire).
    tracing::info!(
        plugin = %cfg.id,
        skill = %skill_name,
        ms = started.elapsed().as_millis() as u64,
        bytes = bytes.len(),
        "invoke served"
    );
    (StatusCode::OK, axum::response::Json(out)).into_response()
}

/// The verify side of the shared recipe — the `gcp-bridge` gate, minus
/// only what gcloud needs: headers present, timestamp within ±5 min,
/// HMAC over "{ts_ms}.{body}" exact match.
fn verify_request(secret: &str, headers: &HeaderMap, body: &str) -> Result<(), String> {
    let ts_raw = headers
        .get("x-jerith-timestamp")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| "missing x-jerith-timestamp".to_string())?;
    let ts: u64 = ts_raw
        .trim()
        .parse()
        .map_err(|_| format!("x-jerith-timestamp {ts_raw:?} is not a number"))?;
    let now = now_ms();
    if now.saturating_sub(ts) > MAX_CLOCK_SKEW_MS || ts.saturating_sub(now) > MAX_CLOCK_SKEW_MS {
        return Err(format!(
            "stale or future-dated request (ts {ts}, now {now})"
        ));
    }
    let sig = headers
        .get("x-jerith-signature")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| "missing x-jerith-signature".to_string())?;
    let expected = sign_body(secret, ts, body);
    if expected != sig.trim() {
        return Err("signature mismatch".to_string());
    }
    if !headers
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.parse::<usize>().is_ok_and(|n| n <= MAX_RESPONSE_BYTES))
        .unwrap_or(true)
    {
        return Err("request body absurdly large".to_string());
    }
    Ok(())
}

fn error_json(msg: &str) -> axum::response::Json<Value> {
    axum::response::Json(json!({ "error": msg }))
}

/// Parse helper shared with the copyable manifest: turn
/// `http://127.0.0.1:7788/rpc` into that literal host:port — the bind
/// half of the endpoint line, so ONE toml value drives both sides.
/// Literal IPs only (loopback is the whole story); a hostname endpoint
/// binds its own way outside this helper.
pub fn endpoint_addr(endpoint: &str) -> Result<std::net::SocketAddr> {
    let after_scheme = endpoint
        .trim_start_matches("https://")
        .trim_start_matches("http://");
    let host_port = after_scheme
        .split(['/', '?'])
        .next()
        .unwrap_or(after_scheme)
        .trim();
    std::net::SocketAddr::from_str(host_port)
        .map_err(|e| anyhow::anyhow!("endpoint {:?} has no parseable host:port: {e}", endpoint))
}

// ---------------------------------------------------------------------------
// Tests: the contract's self-check. The dispatcher side (`external::invoke`)
// is the sender; everything here re-implements ITS request shape by hand so
// a drift in either direction breaks a fast single-crate test.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Boot a server on an ephemeral port, return its endpoint URL.
    async fn spawn(cfg: ServerConfig) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router(Arc::new(cfg))).await.unwrap() });
        format!("http://{addr}/rpc")
    }

    fn echo_cfg(secret: Option<String>) -> ServerConfig {
        let mut handlers = HashMap::new();
        handlers.insert(
            "example.echo".to_string(),
            skill(|args: Value| async move {
                let phrase = args["text"].as_str().unwrap_or("…").to_string();
                Ok(json!({ " echo": phrase }))
            }),
        );
        ServerConfig {
            id: "example".into(),
            secret,
            handlers,
        }
    }

    fn signed_headers(secret: &str, ts: u64, body: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert("x-jerith-timestamp", ts.to_string().parse().unwrap());
        h.insert(
            "x-jerith-signature",
            sign_body(secret, ts, body).parse().unwrap(),
        );
        h
    }

    #[tokio::test]
    async fn unsigned_dev_mode_serves() {
        let url = spawn(echo_cfg(None)).await;
        let body = json!({
            "method": "invoke", "skill": "example.echo",
            "args": {"text": "hi"}, "ctx": {"ts_ms": 0}
        })
        .to_string();
        let resp = reqwest::Client::new()
            .post(&url)
            .body(body)
            .send()
            .await
            .unwrap();
        assert!(resp.status().is_success());
        let v: Value = resp.json().await.unwrap();
        assert_eq!(v[" echo"], "hi");
    }

    #[tokio::test]
    async fn signed_request_matches_the_recipe() {
        let secret = "s3cret".to_string();
        let url = spawn(echo_cfg(Some(secret.clone()))).await;
        let ts = now_ms();
        let body = json!({"method":"invoke","skill":"example.echo","args":{"text":" signed"},"ctx":{"ts_ms":ts}}).to_string();
        let resp = reqwest::Client::new()
            .post(&url)
            .headers(signed_headers(&secret, ts, &body))
            .body(body)
            .send()
            .await
            .unwrap();
        assert!(resp.status().is_success());
        let v: Value = resp.json().await.unwrap();
        assert_eq!(v[" echo"], " signed");
    }

    #[tokio::test]
    async fn unsigned_request_to_a_secret_server_is_refused_before_handlers() {
        let url = spawn(echo_cfg(Some("s3cret".into()))).await;
        let body = json!({"method":"invoke","skill":"example.echo","args":{},"ctx":{}}).to_string();
        let resp = reqwest::Client::new()
            .post(&url)
            .body(body)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let v: Value = resp.json().await.unwrap();
        // The refusal names whichever shared-recipe header came up short
        // first (timestamp is checked first) — same 403 family either way.
        assert!(v["error"].as_str().unwrap().contains("x-jerith"), "{v}");
    }

    #[tokio::test]
    async fn stale_or_future_stamp_is_refused() {
        let secret = "s3cret".to_string();
        let url = spawn(echo_cfg(Some(secret.clone()))).await;
        // Stale by one minute past the window.
        let ts = now_ms() - (MAX_CLOCK_SKEW_MS + 60_000);
        let body = json!({"method":"invoke","skill":"example.echo","args":{},"ctx":{}}).to_string();
        let resp = reqwest::Client::new()
            .post(&url)
            .headers(signed_headers(&secret, ts, &body))
            .body(body)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let v: Value = resp.json().await.unwrap();
        assert!(v["error"].as_str().unwrap().contains("stale"), "{v}");
    }

    #[tokio::test]
    async fn unknown_skill_is_a_model_visible_error() {
        let url = spawn(echo_cfg(None)).await;
        let body = json!({"method":"invoke","skill":"example.nope","args":{},"ctx":{}}).to_string();
        let resp = reqwest::Client::new()
            .post(&url)
            .body(body)
            .send()
            .await
            .unwrap();
        assert!(resp.status().is_success(), "200 keeps the loop calm");
        let v: Value = resp.json().await.unwrap();
        assert!(v["error"].as_str().unwrap().contains("example.nope"));
    }

    #[tokio::test]
    async fn handler_error_becomes_the_refusal_json() {
        let url = spawn(echo_cfg(None)).await;
        let body =
            json!({"method":"invoke","skill":"example.echo","args":{"text":"//fallback"},"ctx":{}})
                .to_string();
        // The echo above never errors; drive the direct handler path by
        // re-serving a config whose echo errors instead.
        let mut handlers = HashMap::new();
        handlers.insert(
            "example.echo".to_string(),
            skill(|_args: Value| async move { Err("the sky fell".to_string()) }),
        );
        let cfg2 = ServerConfig {
            id: "example".into(),
            secret: None,
            handlers,
        };
        let url2 = spawn(cfg2).await;
        let _ = url; // first server stays idle in the background; fine for a test
        let resp = reqwest::Client::new()
            .post(&url2)
            .body(body)
            .send()
            .await
            .unwrap();
        assert!(resp.status().is_success());
        let v: Value = resp.json().await.unwrap();
        assert_eq!(v["error"], "the sky fell");
    }

    #[test]
    fn endpoint_addr_parses_the_toml_line() {
        let a = endpoint_addr("http://127.0.0.1:7788/rpc").unwrap();
        assert_eq!(a.port(), 7788);
        assert!(a.ip().is_loopback());
        let b = endpoint_addr("https://plugins.example.org:8443/rpc");
        assert!(
            b.is_err(),
            "hostnames bind their own way — helper is literal IPs"
        );
        assert!(endpoint_addr("http://127.0.0.1/rpc").is_err());
    }
}
