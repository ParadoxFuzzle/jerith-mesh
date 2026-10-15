//! Webhook / event hooks for plugin authors (Phase 3, FEATURES §8).
//!
//! The bot's plugin/skill system makes the model able to summon capabilities;
//! hooks let plugin authors (and operators) *observe* — the bot POSTs a
//! signed JSON event to operator-configured URLs, fire-and-forget, and never
//! lets delivery slow a chat reply down.
//!
//! Events (the stable set; `type` on the wire is snake_case):
//! - `skill_invoked`   — the model summoned a plugin skill (data: skill, ok,
//!   error when it failed). Raw arguments are NEVER included: webhook
//!   receivers are third parties, and the arguments can carry user text.
//! - `guard_violation` — Jerith Guard blocked an outbound prompt (data:
//!   reason — the public block reason, never the blocked text).
//! - `mine_won`        — a `/mine` win landed in the tally (data:
//!   discord_id, amount_uj).
//!
//! Delivery discipline:
//! - HMAC-SHA256 over `"{ts_ms}.{body}"` with the configured secret in
//!   `x-jerith-signature` (hex, `x-jerith-timestamp` rides along to let
//!   receivers bound the replay window).
//! - Bounded retries per URL (linear backoff: 0.5 s → 1 s → 2 s, default
//!   3 attempts) and a global in-flight cap, so a slow webhook can never
//!   wedge the bot.
//! - No URLs configured (the default) → the subsystem is a no-op; nothing
//!   is constructed, nothing is sent.
//! - URLs are operator config: `http(s)` only, no whitespace, and payload
//!   minimality is treated as a security property, not an optimization.

use hmac::{Hmac, Mac};
use serde_json::{json, Value};
use sha2::Sha256;
use std::sync::{Arc, OnceLock};

/// Maximum simultaneous deliveries (a flood of events cannot pile up
/// unbounded tasks against a slow receiver).
pub const MAX_IN_FLIGHT: usize = 8;

/// Stable event type names — plugin authors key receivers on these.
pub const EV_SKILL_INVOKED: &str = "skill_invoked";
pub const EV_GUARD_VIOLATION: &str = "guard_violation";
pub const EV_MINE_WON: &str = "mine_won";

/// `[hooks]` in bot.toml. Empty `urls` (the default) keeps the feature off.
#[derive(Debug, Clone, serde::Deserialize, PartialEq)]
pub struct HooksConfig {
    /// Webhook URLs (http/https). Delivered concurrently, signed.
    #[serde(default)]
    pub urls: Vec<String>,
    /// Shared secret — when set, requests carry an HMAC-SHA256 signature
    /// over `"{ts_ms}.{body}"` (hex, `x-jerith-signature`).
    #[serde(default)]
    pub secret: Option<String>,
    /// Per-request HTTP timeout (seconds, clamped 1–30).
    #[serde(default = "default_hook_timeout_secs")]
    pub timeout_secs: u64,
    /// Delivery attempts per URL (clamped 1–5).
    #[serde(default = "default_hook_retries")]
    pub max_retries: u32,
}

fn default_hook_timeout_secs() -> u64 {
    5
}

fn default_hook_retries() -> u32 {
    3
}

impl Default for HooksConfig {
    fn default() -> Self {
        Self {
            urls: Vec::new(),
            secret: None,
            timeout_secs: default_hook_timeout_secs(),
            max_retries: default_hook_retries(),
        }
    }
}

impl HooksConfig {
    /// Fail-loud before the bot starts: a mistyped URL would otherwise be
    /// discovered as a silent delivery failure on every event.
    pub fn validate(&self) -> anyhow::Result<()> {
        for url in &self.urls {
            let trimmed = url.trim();
            anyhow::ensure!(
                trimmed.starts_with("http://") || trimmed.starts_with("https://"),
                "[hooks] url {url:?} is not an http(s) endpoint"
            );
            anyhow::ensure!(
                !trimmed.contains(char::is_whitespace),
                "[hooks] url {url:?} contains whitespace"
            );
        }
        Ok(())
    }
}

/// One event: a stable type, a timestamp, and a minimal data payload.
#[derive(Debug, Clone)]
pub struct Event {
    pub event_type: String,
    pub ts_ms: u64,
    pub data: Value,
}

impl Event {
    pub fn new(event_type: &str, data: Value) -> Self {
        Self {
            event_type: event_type.to_string(),
            ts_ms: jerith_mesh_core::bootstrap::now_ms(),
            data,
        }
    }

    /// The signed wire body: `{"type": "…", "ts_ms": N, "data": …}`.
    pub fn body(&self) -> Value {
        json!({
            "type": self.event_type,
            "ts_ms": self.ts_ms,
            "data": self.data,
        })
    }

    /// HMAC-SHA256 over `"{ts_ms}.{body}"`, hex — the exact recomputation a
    /// receiver performs with the shared secret.
    pub fn signature(&self, secret: &str) -> String {
        let body = self.body().to_string();
        let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes())
            .expect("HMAC accepts any key length");
        mac.update(format!("{}.{}", self.ts_ms, body).as_bytes());
        hex_encode(&mac.finalize().into_bytes())
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The receiver's exact verification: HMAC-SHA256 over `"{ts}.{body}"`.
/// Used in tests to prove signatures recompute from (ts, body, secret).
#[cfg(test)]
pub fn verify_signature(ts_ms: u64, body: &str, secret: &str, sig_hex: &str) -> bool {
    let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes())
        .expect("HMAC accepts any key length");
    mac.update(format!("{ts_ms}.{body}").as_bytes());
    hex_encode(&mac.finalize().into_bytes()) == sig_hex
}

/// Delivery handle shared by the bot. Unconfigured → disabled: emit returns
/// immediately without constructing an HTTP client.
#[derive(Debug, Clone)]
pub struct Hooks {
    cfg: Arc<HooksConfig>,
    http: Option<reqwest::Client>,
    in_flight: Arc<tokio::sync::Semaphore>,
}

impl Default for Hooks {
    /// The default handle is disabled — no URLs, no client, no traffic.
    fn default() -> Self {
        Self::new(HooksConfig::default())
    }
}

impl Hooks {
    pub fn new(cfg: HooksConfig) -> Self {
        let enabled = !cfg.urls.is_empty();
        let timeout = std::time::Duration::from_secs(cfg.timeout_secs.clamp(1, 30));
        Self {
            cfg: Arc::new(cfg),
            http: enabled.then(|| {
                reqwest::Client::builder()
                    .timeout(timeout)
                    .build()
                    .expect("reqwest client for hooks")
            }),
            in_flight: Arc::new(tokio::sync::Semaphore::new(MAX_IN_FLIGHT)),
        }
    }

    pub fn enabled(&self) -> bool {
        !self.cfg.urls.is_empty()
    }

    /// Fire-and-forget delivery of one event to every configured URL.
    /// Spawns bounded tasks per URL and returns immediately — chat latency
    /// is never hooked on delivery, which is the entire point here.
    pub fn emit(&self, event: Event) {
        if !self.enabled() {
            return;
        }
        if tokio::runtime::Handle::try_current().is_err() {
            // Called outside the runtime (defensive): drop instead of panic.
            return;
        }
        let Some(http) = self.http.clone() else {
            return;
        };
        let secret = self.cfg.secret.clone();
        let sig = event.signature(secret.as_deref().unwrap_or(""));
        let body = event.body().to_string();
        let retries = self.cfg.max_retries.clamp(1, 5);
        let urls = self.cfg.urls.clone();
        let sema = self.in_flight.clone();
        tokio::spawn(async move {
            for url in urls {
                let permit_ok = sema
                    .clone()
                    .acquire_owned()
                    .await
                    .is_ok();
                if !permit_ok {
                    return; // shutdown: not worth retrying here
                }
                deliver(http.clone(), url, body.clone(), sig.clone(), retries).await;
            }
        });
    }
}

/// One URL's delivery with linear backoff. 2xx = delivered; 5xx and
/// transport errors retry up to `retries` times, then log and give up
/// (events are advisories, not transactions).
async fn deliver(http: reqwest::Client, url: String, body: String, sig: String, retries: u32) {
    for attempt in 1..=retries {
        let req = http
            .post(&url)
            .header("content-type", "application/json")
            // Both headers ride per attempt: the ts lets receivers bound the
            // replay window, the signature binds (ts at signature time,
            // body). Receivers verify against the timestamp they accept.
            .header("x-jerith-timestamp", jerith_mesh_core::bootstrap::now_ms().to_string())
            .header("x-jerith-signature", &sig)
            .body(body.clone());
        match req.send().await {
            Ok(resp) if resp.status().is_success() => {
                tracing::debug!(target: "hooks", url = %url, attempt, "event delivered");
                return;
            }
            Ok(resp) => {
                let status = resp.status();
                if status.is_server_error() && attempt < retries {
                    tokio::time::sleep(std::time::Duration::from_millis(500u64 << (attempt - 1)))
                        .await;
                    continue;
                }
                tracing::warn!(target: "hooks", url = %url, %status, attempt, "event delivery rejected (giving up)");
                return;
            }
            Err(e) => {
                if attempt < retries {
                    tokio::time::sleep(std::time::Duration::from_millis(500u64 << (attempt - 1)))
                        .await;
                    continue;
                }
                tracing::warn!(target: "hooks", url = %url, error = %e, attempt, "event delivery failed (giving up)");
                return;
            }
        }
    }
}

/// The bot-wide handle. Configured once at startup (`init`); every emit
/// site shares it (the bot is a single process — the codebase's established
/// global-static pattern, like RESOLVED_MODELS).
static HOOKS: OnceLock<Hooks> = OnceLock::new();

/// Install the config-derived handle. Called once from main after load.
pub fn init(cfg: HooksConfig) {
    let _ = HOOKS.set(Hooks::new(cfg));
}

/// The shared handle (a disabled one before init — emit sites need no nil
/// checks and tests can drive `Hooks::new` directly).
pub fn global() -> &'static Hooks {
    // A lazily built disabled handle (no URLs → no client → no traffic).
    static FALLBACK: OnceLock<Hooks> = OnceLock::new();
    HOOKS
        .get()
        .unwrap_or_else(|| FALLBACK.get_or_init(Hooks::default))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU16, AtomicU32, Ordering};
    use std::sync::Mutex;

    #[test]
    fn config_validation_rejects_bad_urls() {
        assert!(HooksConfig::default().validate().is_ok(), "empty = off");
        assert!(HooksConfig {
            urls: vec!["https://observer.example/hook".into()],
            ..HooksConfig::default()
        }
        .validate()
        .is_ok());
        assert!(HooksConfig {
            urls: vec!["ftp://nope".into()],
            ..HooksConfig::default()
        }
        .validate()
        .is_err());
        assert!(HooksConfig {
            urls: vec!["https://x y".into()],
            ..HooksConfig::default()
        }
        .validate()
        .is_err());
    }

    #[test]
    fn body_schema_and_hmac_signature_are_receiver_recomputable() {
        let e = Event::new(EV_MINE_WON, json!({ "discord_id": 42, "amount_uj": 5 }));
        let body = e.body().to_string();
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["type"], EV_MINE_WON);
        assert_eq!(v["ts_ms"], json!(e.ts_ms));
        assert_eq!(v["data"]["discord_id"], json!(42));

        let sig = e.signature("the-secret");
        assert!(
            verify_signature(e.ts_ms, &body, "the-secret", &sig),
            "the receiver's recomputation matches"
        );
        assert!(!verify_signature(e.ts_ms, &body, "wrong", &sig));
        // One flipped byte in the body breaks the signature.
        let flipped = body.replace("mine_won", "mine_won_shared");
        assert!(!verify_signature(e.ts_ms, &flipped, "the-secret", &sig));
    }

    /// Receipt mock: counts hits and RECORDS (body, signature); the fail
    /// knob makes the retry bound observable (5xx → retries).
    struct Receiver {
        status: AtomicU16,
        hits: AtomicU32,
        seen: Mutex<Vec<(String, String)>>,
    }

    async fn spawn_receiver(status: u16) -> (String, Arc<Receiver>) {
        let state = Arc::new(Receiver {
            status: AtomicU16::new(status),
            hits: AtomicU32::new(0),
            seen: Mutex::new(Vec::new()),
        });
        let state_for_app = state.clone();
        let app = axum::Router::new().route(
            "/hook",
            axum::routing::post(move |headers: axum::http::HeaderMap, body: String| {
                let st = state_for_app.clone();
                async move {
                    st.hits.fetch_add(1, Ordering::SeqCst);
                    st.seen.lock().unwrap().push((
                        body,
                        headers
                            .get("x-jerith-signature")
                            .and_then(|v| v.to_str().ok())
                            .unwrap_or("")
                            .to_string(),
                    ));
                    if st.status.load(Ordering::SeqCst) == 200 {
                        axum::http::StatusCode::OK
                    } else {
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        tokio::time::sleep(std::time::Duration::from_millis(60)).await;
        (format!("http://{addr}/hook"), state)
    }

    async fn wait_hits(r: &Receiver, want: u32) {
        for _ in 0..150 {
            if r.hits.load(Ordering::SeqCst) >= want {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        panic!(
            "receiver never reached {want} hits (got {}, seen {})",
            r.hits.load(Ordering::SeqCst),
            r.seen.lock().unwrap().len()
        );
    }

    #[tokio::test]
    async fn delivered_once_and_signature_verifies_at_the_receiver() {
        let (url, r) = spawn_receiver(200).await;
        let hooks = Hooks::new(HooksConfig {
            urls: vec![url],
            secret: Some("s3cr3t".into()),
            ..HooksConfig::default()
        });
        hooks.emit(Event::new(
            EV_SKILL_INVOKED,
            json!({ "skill": "web_search", "ok": true }),
        ));
        wait_hits(&r, 1).await;

        let (body, sig) = r.seen.lock().unwrap()[0].clone();
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["type"], EV_SKILL_INVOKED);
        assert_eq!(v["data"]["skill"], "web_search");
        assert_eq!(v["data"]["ok"], json!(true));
        // The receiver recomputes from the body's own ts + the shared secret.
        let ts = v["ts_ms"].as_u64().unwrap();
        assert!(
            verify_signature(ts, &body, "s3cr3t", &sig),
            "the signature ties (ts, body, secret) — tamper-proof"
        );
    }

    #[tokio::test]
    async fn five_never_delivers_retries_are_bounded() {
        let (url, r) = spawn_receiver(500).await;
        let hooks = Hooks::new(HooksConfig {
            urls: vec![url],
            secret: None,
            timeout_secs: 2,
            max_retries: 2,
        });
        hooks.emit(Event::new(
            EV_GUARD_VIOLATION,
            json!({ "reason": "blocked-by-rule" }),
        ));
        // Two attempts against a 500, then the chain gives up — never more.
        wait_hits(&r, 2).await;
        tokio::time::sleep(std::time::Duration::from_millis(800)).await;
        assert_eq!(
            r.hits.load(Ordering::SeqCst),
            2,
            "the retry bound holds (no unbounded hammering)"
        );
    }

    #[tokio::test]
    async fn disabled_hooks_are_a_noop() {
        let hooks = Hooks::default();
        assert!(!hooks.enabled());
        hooks.emit(Event::new(EV_SKILL_INVOKED, json!({})));
        // No HTTP client was even constructed — emit returned at once.
        assert!(hooks.http.is_none());
    }
}
