//! Microsoft Teams transport — the Bot Framework REST surface. The
//! inbound direction is a *push* channel: Bot Framework delivers
//! activities to a public HTTPS endpoint registered in Azure; this
//! build runs no public listener by default, so ingestion routes
//! through `webinbox` (`[transports.webinbox]`, loopback-bound, fronted
//! by the operator's TLS proxy) — see webinbox.rs /in/teams.
//!
//! Outbound here: the Azure AD client-credentials token dance
//! (`login.microsoftonline.com/{tenant}/oauth2/v2.0/token`, scope
//! `https://api.botframework.com/.default`) is implemented and cached;
//! replies POST `{serviceUrl}/v3/conversations/{conversationId}/activities`
//! with that bearer. `serviceUrl` is inbound-derived (Bot Framework's
//! own field), validated against a config allowlist — a spoofed
//! serviceUrl cannot redirect the bot's channel API traffic.
//!
//! No `webinbox` configured = the runner logs its typed requirement and
//! exits Ok (dormant outbound-capable): never a silent half-transport.

use super::{chunk_utf8, DeliveryFn};
use crate::config::BotConfig;
use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamsConfig {
    pub app_id: String,
    pub app_secret: String,
    pub tenant_id: String,
    /// Allowed `serviceUrl` origins (Bot Framework's regional endpoints).
    /// Empty = refuse every send (fail-closed: the token is powerless
    /// without a trusted serviceUrl).
    #[serde(default)]
    pub service_url_origins: Vec<String>,
}
impl TeamsConfig {
    pub fn validate(&self) -> Result<()> {
        for (f, v) in [("app_id", &self.app_id), ("app_secret", &self.app_secret), ("tenant_id", &self.tenant_id)] {
            anyhow::ensure!(!v.trim().is_empty(), "teams: {f} required");
        }
        Ok(())
    }
}

/// Bot Framework token cache: client credentials, ~3500 s TTL from the
/// response (`expires_in`); refresh 5 min early. Mutex-carried — one
/// bot process, one cache.
async fn bearer(tcfg: &TeamsConfig) -> Result<String> {
    static CACHE: tokio::sync::Mutex<Option<(std::time::Instant, String)>> =
        tokio::sync::Mutex::const_new(None);
    let mut guard = CACHE.lock().await;
    if let Some((exp, tok)) = guard.as_ref() {
        if *exp > std::time::Instant::now() {
            return Ok(tok.clone());
        }
    }
    let url = format!("https://login.microsoftonline.com/{}/oauth2/v2.0/token", tcfg.tenant_id);
    let form: &[(&str, &str)] = &[
        ("client_id", &tcfg.app_id),
        ("client_secret", &tcfg.app_secret),
        ("grant_type", "client_credentials"),
        ("scope", "https://api.botframework.com/.default"),
    ];
    let resp = crate::api::client()
        .post(&url)
        .timeout(Duration::from_secs(20))
        .form(form)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("teams token: {e}"))?;
    let status = resp.status().as_u16();
    let v: Value = resp.json().await.unwrap_or(Value::Null);
    anyhow::ensure!(status == 200, "teams token failed ({status}): {v}");
    let tok = v["access_token"].as_str().context("teams token: missing access_token")?.to_string();
    let ttl = v["expires_in"].as_i64().unwrap_or(3500).clamp(60, 3500) as u64;
    *guard = Some((
        std::time::Instant::now() + Duration::from_secs(ttl.saturating_sub(300)),
        tok.clone(),
    ));
    Ok(tok)
}

/// serviceUrl allowlist check (did allowlisted origins only).
pub(crate) fn service_url_allowed(tcfg: &TeamsConfig, service_url: &str) -> bool {
    if tcfg.service_url_origins.is_empty() {
        return false;
    }
    tcfg.service_url_origins.iter().any(|o| {
        let prefix = format!("{}/", o.trim_end_matches('/'));
        service_url.starts_with(&prefix)
            || service_url.trim_end_matches('/') == o.trim_end_matches('/')
    })
}

/// Parse an inbound Bot Framework message activity (webinbox's caller).
/// Returns (conversationId, service_url, user_id, display, text).
pub fn parse_activity(v: &Value) -> Option<(String, String, String, String, String)> {
    if v["type"].as_str()? != "message" {
        return None;
    }
    let conv = v["conversation"]["id"].as_str()?.to_string();
    let service_url = v["serviceUrl"].as_str()?.to_string();
    let user = v["from"]["id"].as_str()?.to_string();
    if user.is_empty() {
        return None;
    }
    let display = v["from"]["name"].as_str().unwrap_or(&user).to_string();
    let text = v["text"].as_str()?.trim().to_string();
    if text.is_empty() {
        return None;
    }
    Some((conv, service_url, user, display, text))
}

/// The shared deliverer for Teams sessions (webinbox provisions it):
/// realistic chunk caps (Teams folds long posts into "see more" at
/// ~28 KB; answer chunks stay under 3500 for readability).
pub fn make_deliverer(cfg: Arc<BotConfig>) -> DeliveryFn {
    Arc::new(move |channel: &str, text: String| {
        let cfg = Arc::clone(&cfg);
        // channel encodes conversationId|serviceUrl (webinbox's route
        // packs them; the field is opaque to the dialogue core).
        let (conv, service_url) = channel
            .split_once('|')
            .map(|(c, s)| (c.to_string(), s.to_string()))
            .unwrap_or_else(|| (channel.to_string(), String::new()));
        Box::pin(async move {
            let tcfg = cfg
                .transports
                .teams
                .clone()
                .context("teams deliverer: config gone")?;
            anyhow::ensure!(
                service_url_allowed(&tcfg, &service_url),
                "teams: serviceUrl {service_url:?} not in [transports.teams] service_url_origins"
            );
            let token = bearer(&tcfg).await?;
            for chunk in chunk_utf8(&text, 3500) {
                let url = format!(
                    "{}/v3/conversations/{}/activities",
                    service_url.trim_end_matches('/'),
                    conv
                );
                let body = json!({ "type": "message", "text": chunk });
                let resp = crate::api::client()
                    .post(&url)
                    .bearer_auth(&token)
                    .json(&body)
                    .timeout(Duration::from_secs(20))
                    .send()
                    .await
                    .map_err(|e| anyhow::anyhow!("teams send: {e}"))?;
                let code = resp.status().as_u16();
                anyhow::ensure!(
                    (200..300).contains(&code),
                    "teams activity POST failed ({code}): {}",
                    resp.text().await.unwrap_or_default()
                );
            }
            Ok(())
        })
    })
}

/// Runner: outbound-capable, dormant until a webinbox ingests; the
/// typed log IS the operator's instruction, never a guess.
pub async fn run(_cfg: Arc<BotConfig>, _state: crate::state::BotState, tcfg: TeamsConfig, _owner: Option<u64>) -> Result<()> {
    tcfg.validate()?;
    tracing::info!(
        transport = "teams",
        "outbound lane armed (client-credentials token cache); INBOUND requires \
         [transports.webinbox] bound to a fronted public /in/teams endpoint — \
         start webinbox to receive; this runner stays dormant until then"
    );
    Ok(())
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> TeamsConfig {
        TeamsConfig {
            app_id: "app".into(),
            app_secret: "s".into(),
            tenant_id: "t".into(),
            service_url_origins: vec!["https://smba.trafficmanager.net/amer/".into()],
        }
    }

    #[test]
    fn config_and_serviceurl_gate() {
        cfg().validate().unwrap();
        let empty = TeamsConfig { app_id: "".into(), ..cfg() };
        assert!(empty.validate().is_err());
        assert!(service_url_allowed(&cfg(), "https://smba.trafficmanager.net/amer/v3/x"));
        assert!(!service_url_allowed(&cfg(), "https://evil.example.net/v3/x"));
        let no_origins = TeamsConfig { service_url_origins: vec![], ..cfg() };
        assert!(!service_url_allowed(&no_origins, "https://smba.trafficmanager.net/amer/v3/x"));
    }

    #[test]
    fn activity_parse_pulls_conversation_service_and_text() {
        let v = json!({
            "type": "message",
            "serviceUrl": "https://smba.trafficmanager.net/amer/",
            "conversation": { "id": "19:abc@thread.v2" },
            "from": { "id": "29:xyz", "name": "Noodly" },
            "text": "hey bot, what's the tally?"
        });
        let parsed = parse_activity(&v);
        assert!(matches!(parsed, Some((conv, _, user, _d, t)) if conv == "19:abc@thread.v2" && user == "29:xyz" && t.starts_with("hey bot")));
        // Non-message / empty text refused.
        assert_eq!(parse_activity(&json!({"type": "typing", "conversation": {"id": "c"}, "from": {"id": "u"}, "serviceUrl": "s"})), None);
        assert_eq!(parse_activity(&json!({"type": "message", "conversation": {"id": "c"}, "from": {"id": "u"}, "serviceUrl": "s", "text": "  "}), ), None);
    }
}
