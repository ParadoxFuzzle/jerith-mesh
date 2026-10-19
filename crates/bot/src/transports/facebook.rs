//! WhatsApp Cloud API transport. Outbound: Graph POST
//! `/{phone_number_id}/messages` with the messaging_product envelope and
//! the long-lived/short-lived access token. Inbound: Meta PUSHES onto a
//! public webhook (there is no polling surface), verified + signed —
//! that lane lives in `webinbox.rs /in/whatsapp`, which validates the
//! `X-Hub-Signature-256` HMAC-SHA256 over the raw body against
//! `app_secret` (Meta's standard app-webhook scheme) and the
//! `hub.verify_token` GET challenge. Both config knobs live here so
//! rotation touches one file.
//!
//! No `app_secret` configured → the webhook route refuses messages
//! (fail-closed: unsigned Meta callbacks are untrusted by definition).

use super::{chunk_utf8, DeliveryFn};
use crate::config::BotConfig;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhatsappConfig {
    pub phone_number_id: String,
    /// Meta access token (graph.facebook.com bearer).
    pub access_token: String,
    /// Webhook verify token (set in the Meta app's webhook config).
    #[serde(default)]
    pub verify_token: Option<String>,
    /// App secret — REQUIRED for webhook message verification.
    #[serde(default)]
    pub app_secret: Option<String>,
    /// Graph API version pin (today's stable).
    #[serde(default = "default_graph_version")]
    pub api_version: String,
}
fn default_graph_version() -> String {
    "v22.0".into()
}

impl WhatsappConfig {
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(!self.phone_number_id.trim().is_empty(), "whatsapp: phone_number_id required");
        anyhow::ensure!(!self.access_token.trim().is_empty(), "whatsapp: access_token required");
        anyhow::ensure!(
            self.app_secret.is_some(),
            "whatsapp: app_secret required — webhook messages are untrusted without signature verification"
        );
        anyhow::ensure!(
            self.api_version.starts_with('v'),
            "whatsapp: api_version must look like v22.0"
        );
        Ok(())
    }
}

/// Send plain text to a JID (E.164 with or without wa-address forms —
/// Graph accepts the bare number).
pub(crate) async fn send_text(wcfg: &WhatsappConfig, to: &str, text: &str) -> Result<()> {
    for chunk in chunk_utf8(text, 3500) {
        let url = format!(
            "https://graph.facebook.com/{}/{}",
            wcfg.api_version, wcfg.phone_number_id
        );
        let body = json!({
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to.trim(),
            "type": "text",
            "text": { "body": chunk }
        });
        let (code, v) = super::http_json(
            reqwest::Method::POST,
            &url,
            Some(&wcfg.access_token),
            Some(body),
            Duration::from_secs(20),
        )
        .await?;
        anyhow::ensure!(
            (200..300).contains(&code) && v["error"].is_null(),
            "whatsapp send refused ({code}): {v}"
        );
    }
    Ok(())
}

/// The deliverer (channel == recipient number for this relay).
pub fn make_deliverer(wcfg: WhatsappConfig) -> DeliveryFn {
    Arc::new(move |to: &str, text: String| {
        let wcfg = wcfg.clone();
        let to = to.to_string();
        Box::pin(async move { send_text(&wcfg, &to, &text).await })
    })
}

/// Parse one webhook notification body (Meta's nested shape, unit-tested:
/// entry → changes[0].value → messages[0] with from/text.body). Returns
/// None for statuses (delivery receipts), unsupported message types,
/// and pings.
pub fn parse_message(v: &Value) -> Option<(String, String, String)> {
    // (from, display, text)
    let entry = v["entry"].as_array()?.first()?;
    let value = &entry["changes"].as_array()?.first()?["value"];
    let msg = value["messages"].as_array()?.first()?;
    let from = msg["from"].as_str()?.to_string();
    if from.is_empty() {
        return None;
    }
    let profile = value
        .pointer("/contacts/0/profile/name")
        .and_then(Value::as_str)
        .unwrap_or(&from)
        .to_string();
    let text = msg["text"]["body"].as_str()?.trim().to_string();
    if text.is_empty() {
        return None;
    }
    Some((from, profile, text))
}

pub async fn run(_cfg: Arc<BotConfig>, _state: crate::state::BotState, wcfg: WhatsappConfig, _owner: Option<u64>) -> Result<()> {
    wcfg.validate()?;
    tracing::info!(
        transport = "whatsapp",
        phone = %wcfg.phone_number_id,
        "graph send lane armed; INBOUND rides [transports.webinbox] /in/whatsapp \
         (X-Hub-Signature-256 verified with app_secret, hub.verify_token GET handshake)"
    );
    Ok(())
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_message_walks_metas_nest() {
        let v = json!({
            "entry": [{
                "changes": [{
                    "value": {
                        "contacts": [{ "profile": { "name": "Noodly" }, "wa_id": "15559998888" }],
                        "messages": [{ "from": "15559998888", "type": "text", "text": { "body": "hey mesh" } }]
                    }
                }]
            }]
        });
        assert_eq!(
            parse_message(&v),
            Some(("15559998888".into(), "Noodly".into(), "hey mesh".into()))
        );
        // Status notifications (no messages array) → None.
        let st = json!({ "entry": [{ "changes": [{ "value": { "statuses": [{ "id": "x" }] } }] }] });
        assert_eq!(parse_message(&st), None);
    }

    #[test]
    fn config_demands_app_secret_for_webhook_trust() {
        let c = WhatsappConfig {
            phone_number_id: "123".into(),
            access_token: "tok".into(),
            verify_token: Some("v".into()),
            app_secret: Some("sec".into()),
            api_version: "v22.0".into(),
        };
        c.validate().unwrap();
        let no_secret = WhatsappConfig { app_secret: None, ..c };
        assert!(no_secret.validate().is_err());
    }
}
