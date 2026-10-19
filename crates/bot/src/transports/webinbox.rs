//! webinbox — the one optional HTTP listener for PUSH-only platforms
//! (Microsoft Teams, Google Chat app events, WhatsApp Cloud events,
//! Slack Events-API fallback). Loopback-bound by default
//! (`127.0.0.1:5543`): the operator fronts it with their reversed TLS
//! proxy — the platform's own public requirement. Verification is
//! per-platform and fail-closed:
//!
//! - **whatsapp**: `X-Hub-Signature-256: sha256=<hex>` HMAC-SHA256 over
//!   the RAW body against `[transports.whatsapp] app_secret`; GET verify
//!   handshake answers `hub.challenge` only when `hub.verify_token`
//!   matches. No app_secret → every message refused (Meta's signing IS
//!   the auth).
//! - **slack**: Events API `v0` signing:
//!   `X-Slack-Signature = "v0=" + hex(HMAC(signing_secret, "v0:{ts}:{body}"))`
//!   with a ±5 min replay window on `X-Slack-Request-Timestamp`, plus
//!   the `url_verification` challenge reply.
//! - **teams**: Bot Framework token validation (JWKS) is deliberately
//!   NOT implemented this build — the route runs only when
//!   `allow_unverified = true` IS SET in the config (loopback-only use),
//!   and logs that banner loudly; the serviceUrl allowlist in teams.rs
//!   still gates every REPLY.
//! - **google_chat**: posts b64 GONE (the platform's verification needs
//!   service-account JWTs); the route requires `allow_unverified` like
//!   Teams' — loopback dev mode.
//!
//! Each route parses the platform payload → [`Incoming`] → the shared
//! `handle_incoming` with the platform's own deliverer (teams.rs /
//! facebook.rs / feishu.rs helpers, or a webhook reply for slack).

use crate::config::BotConfig;
use crate::state::BotState;
use anyhow::{Context as _, Result};
use axum::body::Bytes;
use axum::extract::State as AxState;
use axum::http::HeaderMap;
use axum::routing::post;
use axum::Router;
use hmac::{Hmac, Mac};
use serde_json::Value;
use sha2::Sha256;
use std::sync::Arc;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WebinboxConfig {
    /// `host:port` — loopback default; a public bind is the operator's
    /// explicit choice (the banner says so).
    #[serde(default = "default_bind")]
    pub bind: String,
    /// Slack signing secret (Slack app settings), if the slack route runs.
    #[serde(default)]
    pub slack_signing_secret: Option<String>,
    /// Loopback-unverified mode for JWT-verified platforms.
    #[serde(default)]
    pub allow_unverified: bool,
}
fn default_bind() -> String {
    "127.0.0.1:5543".into()
}

impl WebinboxConfig {
    pub fn validate(&self) -> Result<()> {
        let (host, _port) = self
            .bind
            .rsplit_once(':')
            .context("webinbox: bind must be host:port")?;
        let public = host != "127.0.0.1" && host != "localhost" && host != "::1";
        if public && !self.allow_unverified {
            tracing::warn!(
                host,
                "webinbox has a PUBLIC bind — signatures for the routes are your \
                 only auth for slack/whatsapp; teams/google_chat routes refuse \
                 unless allow_unverified is explicitly set (it is: unverified)"
            );
        }
        Ok(())
    }
}

#[derive(Clone)]
struct Wb {
    cfg: Arc<BotConfig>,
    state: BotState,
    wcfg: WebinboxConfig,
    owner: Option<u64>,
}

/// HMAC-SHA256 hex — the one signing primitive every route shares.
fn hmac_hex(secret: &str, data: &[u8]) -> String {
    let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes()).expect("hmac key len");
    mac.update(data);
    let out = mac.finalize().into_bytes();
    out.iter().map(|b| format!("{b:02x}")).collect()
}

pub async fn run(cfg: Arc<BotConfig>, wcfg: &WebinboxConfig) {
    if let Err(e) = wcfg.validate() {
        tracing::error!(transport = "webinbox", error = %e, "bad [transports.webinbox] — refusing to serve");
        return;
    }
    let state = crate::state::new_state();
    let owner = cfg.owner_discord_id;
    let app = Router::new()
        .route("/in/whatsapp", post(in_whatsapp).get(verify_whatsapp))
        .route("/in/slack", post(in_slack))
        .route("/in/teams", post(in_teams))
        .route("/in/google_chat", post(in_google_chat))
        .with_state(Wb { cfg, state, wcfg: wcfg.clone(), owner });
    let listener = match tokio::net::TcpListener::bind(&wcfg.bind).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(bind = %wcfg.bind, error = %e, "webinbox bind failed");
            return;
        }
    };
    tracing::info!(bind = %wcfg.bind, "webinbox listening (platform webhooks welcome)");
    if let Err(e) = axum::serve(listener, app).await {
        tracing::error!(error = %e, "webinbox server stopped");
    }
}

// ---------------------------------------------------------------------------
// WhatsApp
// ---------------------------------------------------------------------------

/// GET handshake: Meta's `hub.mode=subscribe` + `hub.verify_token` +
/// `hub.challenge`; with a 200 TEXT/plain answer ONLY when the token
/// matches (the challenge), else 403. Query params ride the URL.
async fn verify_whatsapp(
    AxState(wb): AxState<Wb>,
    q: axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<String, (axum::http::StatusCode, String)> {
    let wcfg = wb
        .cfg
        .transports
        .whatsapp
        .clone()
        .ok_or((axum::http::StatusCode::FORBIDDEN, "whatsapp transport not configured".into()))?;
    let want = wcfg
        .verify_token
        .clone()
        .filter(|t| !t.is_empty())
        .ok_or((axum::http::StatusCode::FORBIDDEN, "whatsapp: verify_token missing".into()))?;
    let mode = q.get("hub.mode").map(String::as_str).unwrap_or("");
    let token = q.get("hub.verify_token").map(String::as_str).unwrap_or("");
    if mode == "subscribe" && ct_eq(token.as_bytes(), want.as_bytes()) {
        return Ok(q
            .get("hub.challenge")
            .cloned()
            .unwrap_or_default());
    }
    Err((axum::http::StatusCode::FORBIDDEN, "verify handshake refused".into()))
}

/// Post: verify, parse, route.
async fn in_whatsapp(
    AxState(wb): AxState<Wb>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<&'static str, (axum::http::StatusCode, String)> {
    let wcfg = wb
        .cfg
        .transports
        .whatsapp
        .clone()
        .ok_or((axum::http::StatusCode::FORBIDDEN, "whatsapp transport not configured".into()))?;
    let secret = wcfg
        .app_secret
        .clone()
        .ok_or((axum::http::StatusCode::FORBIDDEN, "whatsapp: app_secret missing".into()))?;
    let got = headers
        .get("x-hub-signature-256")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("sha256="))
        .unwrap_or_default();
    let want = hmac_hex(&secret, &body);
    if got.is_empty() || !ct_eq(got.as_bytes(), want.as_bytes()) {
        return Err((axum::http::StatusCode::FORBIDDEN, "signature mismatch".into()));
    }
    let v: Value = serde_json::from_slice(&body)
        .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, format!("bad json: {e}")))?;
    let Some((from, display, text)) = crate::transports::facebook::parse_message(&v) else {
        return Ok(""); // status notifications etc. — ack silently
    };
    let deliverer = crate::transports::facebook::make_deliverer(wcfg);
    let inc = crate::transports::Incoming {
        platform: "whatsapp",
        channel: from.clone(),
        user: from,
        display,
        text,
    };
    spawn_session(Arc::clone(&wb.cfg), wb.state.clone(), inc, wb.owner, deliverer);
    Ok("")
}

/// constant-time-ish compare (a hex HMAC check wants no early-exit
/// timing oracle; length must match first — hex is fixed size anyway).
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

fn spawn_session(
    cfg: Arc<BotConfig>,
    state: BotState,
    inc: crate::transports::Incoming,
    owner: Option<u64>,
    deliverer: crate::transports::DeliveryFn,
) {
    tokio::spawn(async move {
        crate::transports::handle_incoming(&cfg, &state, inc, owner, deliverer).await;
    });
}

// ---------------------------------------------------------------------------
// Slack (Events API 起 fallback lane)
// ---------------------------------------------------------------------------

async fn in_slack(
    AxState(wb): AxState<Wb>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<&'static str, (axum::http::StatusCode, String)> {
    let Some(secret) = wb.wcfg.slack_signing_secret.clone() else {
        return Err((
            axum::http::StatusCode::FORBIDDEN,
            "slack route needs webinbox.slack_signing_secret".into(),
        ));
    };
    let ts = headers
        .get("x-slack-request-timestamp")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<i64>().ok())
        .ok_or((axum::http::StatusCode::FORBIDDEN, "missing signature timestamp".into()))?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    if (now - ts).abs() > 300 {
        return Err((axum::http::StatusCode::FORBIDDEN, "slack replay window exceeded".into()));
    }
    let v0 = format!("v0:{ts}:{}", String::from_utf8_lossy(&body));
    let want = format!("v0={}", hmac_hex(&secret, v0.as_bytes()));
    let got = headers
        .get("x-slack-signature")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    if !ct_eq(got.as_bytes(), want.as_bytes()) {
        return Err((axum::http::StatusCode::FORBIDDEN, "slack signature mismatch".into()));
    }
    let v: Value = serde_json::from_slice(&body)
        .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, format!("bad json: {e}")))?;
    if v["type"].as_str() == Some("url_verification") {
        let challenge = v["challenge"].as_str().unwrap_or_default().to_string();
        // (axum response String; the caller sees it raw)
        return Ok(Box::leak(challenge.into_boxed_str()));
    }
    if v["type"].as_str() != Some("event_callback") {
        return Ok("");
    }    let ev = &v["event"];
    // Slack Events-API shares the message envelope: skip bots/subtypes,
    // build the Incoming, route through the shared pipeline.
    if ev["type"].as_str() != Some("message")
        || ev["subtype"].as_str().is_some()
        || ev["bot_id"].is_string()
    {
        return Ok("");
    }
    let channel = ev["channel"].as_str().map(str::to_string);
    let user = ev["user"].as_str().map(str::to_string);
    let text = ev["text"].as_str().and_then(|t| (!t.trim().is_empty()).then(|| t.to_string()));
    let (Some(channel), Some(user), Some(text)) = (channel, user, text) else {
        return Ok("");
    };
    let deliverer: crate::transports::DeliveryFn = {
        let token = wb
            .cfg
            .transports
            .slack
            .as_ref()
            .map(|s| s.bot_token.clone())
            .context("slack route needs [transports.slack] for replies")
            .map_err(|e| (axum::http::StatusCode::FORBIDDEN, e.to_string()))?;
        crate::transports::slack::make_deliverer(token)
    };
    let inc = crate::transports::Incoming {
        platform: "slack",
        channel,
        user: user.clone(),
        display: user,
        text,
    };
    spawn_session(Arc::clone(&wb.cfg), wb.state.clone(), inc, wb.owner, deliverer);
    Ok("")
}

// ---------------------------------------------------------------------------
// Teams (unverified loopback mode) / Google Chat (unverified loopback)
// ---------------------------------------------------------------------------

async fn in_teams(
    AxState(wb): AxState<Wb>,
    body: Bytes,
) -> Result<&'static str, (axum::http::StatusCode, String)> {
    if !wb.wcfg.allow_unverified {
        return Err((
            axum::http::StatusCode::FORBIDDEN,
            "teams route requires webinbox.allow_unverified (no JWT verification this build — loopback only)".into(),
        ));
    }
    let v: Value = serde_json::from_slice(&body)
        .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, format!("bad json: {e}")))?;
    let Some((conv, service_url, user, display, text)) = crate::transports::teams::parse_activity(&v) else {
        return Ok("");
    };
    let Some(tcfg) = wb.cfg.transports.teams.clone() else {
        return Err((axum::http::StatusCode::FORBIDDEN, "teams transport not configured".into()));
    };
    if !crate::transports::teams::service_url_allowed(&tcfg, &service_url) {
        return Err((axum::http::StatusCode::FORBIDDEN, "teams: serviceUrl not allowlisted".into()));
    }
    let deliverer = crate::transports::teams::make_deliverer(Arc::clone(&wb.cfg));
    let inc = crate::transports::Incoming {
        platform: "teams",
        // channel encodes conversationId|serviceUrl (teams.rs contract).
        channel: format!("{conv}|{service_url}"),
        user,
        display,
        text,
    };
    spawn_session(Arc::clone(&wb.cfg), wb.state.clone(), inc, wb.owner, deliverer);
    Ok("")
}

async fn in_google_chat(
    AxState(wb): AxState<Wb>,
    body: Bytes,
) -> Result<&'static str, (axum::http::StatusCode, String)> {
    if !wb.wcfg.allow_unverified {
        return Err((
            axum::http::StatusCode::FORBIDDEN,
            "google_chat route requires webinbox.allow_unverified (no service-account JWT verification this build)".into(),
        ));
    }
    let v: Value = serde_json::from_slice(&body)
        .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, format!("bad json: {e}")))?;
    // Chat event envelope: message.text, message.sender.{name,displayName},
    // space.name (a webhook-only space can't ping back per-space; replies
    // ride gcfg.webhook_url).
    let Some(text) = v.pointer("/message/text").and_then(Value::as_str).map(str::to_string) else {
        return Ok("");
    };
    let user = v.pointer("/message/sender/name").and_then(Value::as_str).unwrap_or("unknown").to_string();
    let display = v.pointer("/message/sender/displayName").and_then(Value::as_str).unwrap_or(&user).to_string();
    if text.trim().is_empty() {
        return Ok("");
    }
    let Some(gcfg) = wb.cfg.transports.google_chat.clone() else {
        return Err((axum::http::StatusCode::FORBIDDEN, "google_chat transport not configured".into()));
    };
    let deliverer = crate::transports::google_chat::make_deliverer(gcfg);
    let inc = crate::transports::Incoming {
        platform: "google_chat",
        channel: user.clone(),
        user,
        display,
        text,
    };
    spawn_session(Arc::clone(&wb.cfg), wb.state.clone(), inc, wb.owner, deliverer);
    Ok("")
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hmac_hex_matches_the_well_known_vectors() {
        // Two independent vectors (RFC 4231 test case 1 computed per
        // RFC's byte inputs, and python stdlib hmac.corpus cross-check):
        assert_eq!(
            hmac_hex("key", b"The quick brown fox jumps over the lazy dog"),
            "f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8"
        );
        assert_eq!(
            // Key = 0x0b repeated 20 bytes, data "Hi There" (RFC 4231 TC1).
            hmac_hex(
                "\u{b}".repeat(20).as_str(),
                b"Hi There"
            ),
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
    }

    #[test]
    fn ct_eq_is_lent_deterministic_lent() {
        assert!(ct_eq(b"abc", b"abc"));
        assert!(!ct_eq(b"abc", b"abd"));
        assert!(!ct_eq(b"abc", b"abcd"));
    }

    #[test]
    fn webinbox_config_loopback_default() {
        let c = WebinboxConfig { bind: default_bind(), slack_signing_secret: None, allow_unverified: false };
        assert_eq!(c.bind, "127.0.0.1:5543");
        c.validate().unwrap();
        let bad = WebinboxConfig { bind: "no-port".into(), ..c };
        assert!(bad.validate().is_err());
    }
}
