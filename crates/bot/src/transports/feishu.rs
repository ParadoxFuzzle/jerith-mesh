//! Feishu / Lark transport — Server API over REST (outbound) with the
//! app's `tenant_access_token` (internal app: POST
//! `/open-apis/auth/v3/tenant_access_token/internal` {app_id, app_secret}
//! → {tenant_access_token, expire(seconds)}), messages POST
//! `/open-apis/im/v1/messages?receive_id_type=chat_id` where `content`
//! is a JSON *string* ({"text": "…"}) — the platform's own wrapping.
//!
//! Inbound: Feishu's event subscription is either a public callback URL
//! (verify_url handshake + AES-encrypted payloads) or the SDK-internal
//! WebSocket long-connection; the callback requires a public HTTPS
//! front and the long-connection wire protocol is SDK-private, so this
//! build ships OUTBOUND ONLY and says so — the runner logs the typed
//! requirement instead of guessing a wire format from outside the SDK.
//!
//! Domain: `feishu` (open.feishu.cn) vs `lark` (open.larksuite.com) —
//! one config word, both domains are the same API shape.

use super::{chunk_utf8, DeliveryFn};
use crate::config::BotConfig;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeishuConfig {
    pub app_id: String,
    pub app_secret: String,
    /// "feishu" (CN, open.feishu.cn) | "lark" (intl, open.larksuite.com).
    #[serde(default = "default_domain")]
    pub domain: String,
    /// Optional chat id (oc_…) for a proactive/reply lane usable even
    /// without an inbound event (e.g. ops notices into a dev chat).
    #[serde(default)]
    pub notify_chat_id: Option<String>,
}
fn default_domain() -> String {
    "feishu".into()
}

impl FeishuConfig {
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(!self.app_id.trim().is_empty(), "feishu: app_id required");
        anyhow::ensure!(!self.app_secret.trim().is_empty(), "feishu: app_secret required");
        anyhow::ensure!(
            matches!(self.domain.as_str(), "feishu" | "lark"),
            "feishu: domain must be \"feishu\" or \"lark\""
        );
        Ok(())
    }
    pub fn base(&self) -> String {
        match self.domain.as_str() {
            "lark" => "https://open.larksuite.com".into(),
            _ => "https://open.feishu.cn".into(),
        }
    }
}

/// Token cache (Mutex<Instant, token>) — refresh 5 min early.
pub(crate) async fn tenant_token(fcfg: &FeishuConfig) -> Result<String> {
    static CACHE: tokio::sync::Mutex<Option<(std::time::Instant, String)>> =
        tokio::sync::Mutex::const_new(None);
    let mut g = CACHE.lock().await;
    if let Some((exp, t)) = g.as_ref() {
        if *exp > std::time::Instant::now() {
            return Ok(t.clone());
        }
    }
    let url = format!("{}/open-apis/auth/v3/tenant_access_token/internal", fcfg.base());
    let (code, v) = super::http_json(
        reqwest::Method::POST,
        &url,
        None,
        Some(json!({ "app_id": fcfg.app_id, "app_secret": fcfg.app_secret })),
        Duration::from_secs(20),
    )
    .await?;
    anyhow::ensure!(
        v["code"].as_i64().unwrap_or(-1) == 0 && v["tenant_access_token"].is_string(),
        "feishu token failed ({code}): {v}"
    );
    let secs = v["expire"].as_i64().unwrap_or(3600).clamp(120, 7200);
    *g = Some((
        std::time::Instant::now() + Duration::from_secs((secs - 300).max(60) as u64),
        v["tenant_access_token"].as_str().unwrap().to_string(),
    ));
    Ok(g.as_ref().unwrap().1.clone())
}

/// Send one text to a chat_id (or user open_id via the configured type).
/// Dormant until a feishu inbound lane exists (see module doc) — the
/// allow keeps the compiled, token-checked send path ready for that day.
#[allow(dead_code)]
pub(crate) async fn send_text(fcfg: &FeishuConfig, receive_id: &str, text: &str) -> Result<()> {
    for chunk in chunk_utf8(text, 3500) {
        let url = format!(
            "{}/open-apis/im/v1/messages?receive_id_type=chat_id",
            fcfg.base()
        );
        let token = tenant_token(fcfg).await?;
        let body = json!({
            "receive_id": receive_id,
            "msg_type": "text",
            "content": serde_json::to_string(&json!({"text": chunk}))?,
        });
        let (code, v) = super::http_json(
            reqwest::Method::POST,
            &url,
            Some(&token),
            Some(body),
            Duration::from_secs(20),
        )
        .await?;
        anynf(v, code)?;
    }
    Ok(())
}

/// Feishu's two-layer error code check (http status + feishu code).
#[allow(dead_code)]
fn anynf(v: Value, code: u16) -> Result<()> {
    let fc = v["code"].as_i64().unwrap_or(0);
    anyhow::ensure!(
        (200..300).contains(&code) && fc == 0,
        "feishu message refused (http {code}, feishu code {fc}): {v}"
    );
    Ok(())
}

/// The deliverer used by any future webinbox callback lane (channel ==
/// receive id); present now so outbound-only users can drive replies.
#[allow(dead_code)]
pub fn make_deliverer(fcfg: FeishuConfig) -> DeliveryFn {
    Arc::new(move |receive_id: &str, text: String| {
        let fcfg = fcfg.clone();
        let receive_id = receive_id.to_string();
        Box::pin(async move { send_text(&fcfg, &receive_id, &text).await })
    })
}

pub async fn run(_cfg: Arc<BotConfig>, _state: crate::state::BotState, fcfg: FeishuConfig, _owner: Option<u64>) -> Result<()> {
    fcfg.validate()?;
    // Token round-trip as the boot check: wrong app credentials fail
    // HERE, loudly, not on the first reply years later.
    let t = tenant_token(&fcfg).await?;
    tracing::info!(
        transport = "feishu",
        domain = %fcfg.domain,
        token_len = t.len(),
        "tenant token ok; OUTBOUND lane armed — INBOUND needs a public callback \
         URL (verify_url + AES) or the SDK-internal WebSocket long-connection, \
         neither shipped in this build"
    );
    if let Some(cid) = &fcfg.notify_chat_id {
        tracing::info!(chat = %cid, "notify chat configured (send-only lane)");
    }
    Ok(())
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_and_domains() {
        let c = FeishuConfig {
            app_id: "cli_x".into(),
            app_secret: "s".into(),
            domain: "feishu".into(),
            notify_chat_id: None,
        };
        c.validate().unwrap();
        assert_eq!(c.base(), "https://open.feishu.cn");
        let lark = FeishuConfig { domain: "lark".into(), ..c };
        assert_eq!(lark.base(), "https://open.larksuite.com");
        let bad = FeishuConfig { domain: "slack".into(), ..lark };
        assert!(bad.validate().is_err());
    }
}
