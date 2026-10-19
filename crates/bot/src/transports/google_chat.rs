//! Google Chat transport — webhook relay. The pragmatic, zero-friction
//! send lane: a Space webhook URL (Chat → Space settings → Apps &
//! integrations → webhooks) POSTs `{ "text": … }` — no service account,
//! no JWT, no domain certificate. The inbound direction is Google's
//! push (Chat API app with a `/cases` endpoint or Pub/Sub), which needs
//! RS256 service-account signing not shipped here — the runner states
//! that typed requirement and stays dormant rather than pretending.
//!
//! Cap: Chat hard-fails posts past 4K; 3500 stays within the UI fold.

use super::{chunk_utf8, DeliveryFn};
use crate::config::BotConfig;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleChatConfig {
    /// The space webhook URL (https://chat.googleapis.com/v1/spaces/…/messages?key=…).
    pub webhook_url: String,
    /// Optional `threadKey` so replies thread in the space UI.
    #[serde(default)]
    pub thread_key: Option<String>,
}
impl GoogleChatConfig {
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.webhook_url.starts_with("https://chat.googleapis.com/v1/spaces/"),
            "google_chat: webhook_url must be a chat.googleapis.com space webhook"
        );
        Ok(())
    }
}

/// The relay deliverer (channel is decorative — a webhook always posts
/// to its one space; the dialogue core still needs a channel key, and
/// incoming tests use `"webhook-relay"`).
pub fn make_deliverer(gcfg: GoogleChatConfig) -> DeliveryFn {
    Arc::new(move |_channel: &str, text: String| {
        let gcfg = gcfg.clone();
        Box::pin(async move {
            for chunk in chunk_utf8(&text, 3500) {
                let mut body = serde_json::json!({ "text": chunk });
                if let Some(k) = &gcfg.thread_key {
                    body["thread"] = serde_json::json!({ "threadKey": k });
                }
                let resp = crate::api::client()
                    .post(&gcfg.webhook_url)
                    .timeout(Duration::from_secs(20))
                    .json(&body)
                    .send()
                    .await
                    .map_err(|e| anyhow::anyhow!("google chat post: {e}"))?;
                let code = resp.status().as_u16();
                anyhow::ensure!(
                    (200..300).contains(&code),
                    "google chat webhook refused ({code}): {}",
                    resp.text().await.unwrap_or_default()
                );
            }
            Ok(())
        })
    })
}

pub async fn run(_cfg: Arc<BotConfig>, _state: crate::state::BotState, gcfg: GoogleChatConfig, _owner: Option<u64>) -> Result<()> {
    gcfg.validate()?;
    tracing::info!(
        transport = "google_chat",
        "webhook relay armed; INBOUND requires a Chat API app with RS256 service-account \
         push (not shipped) — outbound-only until then. Use [transports.webinbox] for \
         any webhook-driven ingest platform support."
    );
    Ok(())
}

