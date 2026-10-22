//! Telegram transport — the Bot API long-poll (`getUpdates`) with
//! `sendMessage` replies. Token auth is the whole story (the BotFather
//! token rides the `/bot<token>/` path prefix — no OAuth, no webhook, no
//! public endpoint), the cloud equivalent of the Discord token lane.
//! Long polling keeps a single update authority: a second consumer (a
//! webhook this token may not have, another poller) 409-conflicts — that
//! shape is a typed, honest config-fatal refusal, not a retry storm.
//!
//! Respond mode ("mention" default): private chats always relay; group
//! messages relay only when they mention the bot username ("all" relays
//! every group too — opt-in via bot.toml).
//!
//! Split cap: 4000 (Telegram's hard message cap is 4096 chars; the shared
//! rune-safe chunker keeps under the UI fold).
//!
//! Token redaction: the auth is IN the URL path, so the adapter's own
//! error contexts never embed the URL — a failed poll says "telegram api
//! call failed", never the token.

use super::{DeliveryFn, Incoming, chunk_utf8, handle_incoming};
use crate::config::BotConfig;
use crate::state::BotState;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;

const DEFAULT_API_URL: &str = "https://api.telegram.org";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelegramConfig {
    /// API root override (tests point it at a local stub). Default: the
    /// real cloud api.
    #[serde(default = "default_api_url")]
    pub api_url: String,
    /// BotFather token (`123456:ABC…`).
    pub token: String,
    /// Bot username for group-mention detection (mention mode).
    pub bot_username: String,
    /// Restrict relay to these chat ids (as strings); empty = any chat.
    #[serde(default)]
    pub allowed_chats: Vec<String>,
    /// mention | all
    #[serde(default = "default_respond_to")]
    pub respond_to: String,
    /// getUpdates long-poll seconds (clamp 5..50; Telegram caps 50).
    #[serde(default = "default_poll_timeout")]
    pub poll_timeout_secs: u32,
}
fn default_api_url() -> String {
    DEFAULT_API_URL.into()
}
fn default_respond_to() -> String {
    "mention".into()
}
fn default_poll_timeout() -> u32 {
    25
}

impl TelegramConfig {
    pub fn validate(&mut self) -> Result<()> {
        self.api_url = self.api_url.trim_end_matches('/').to_string();
        anyhow::ensure!(
            self.api_url.starts_with("https://") || self.api_url.starts_with("http://"),
            "telegram: api_url must be http(s)"
        );
        anyhow::ensure!(
            !self.token.trim().is_empty(),
            "telegram: token required (a BotFather token)"
        );
        anyhow::ensure!(
            matches!(self.respond_to.as_str(), "mention" | "all"),
            "telegram: respond_to must be \"mention\" | \"all\""
        );
        if self.respond_to == "mention" {
            anyhow::ensure!(
                !self.bot_username.trim().is_empty(),
                "telegram: bot_username required in mention mode"
            );
        }
        self.poll_timeout_secs = self.poll_timeout_secs.clamp(5, 50);
        Ok(())
    }
    /// One concrete api call URL (the token rides the path).
    fn call_url(&self, method: &str) -> String {
        format!("{}/bot{}/{}", self.api_url, self.token, method)
    }
}

/// The adapter's own HTTP wrapper — same shared client pool as every
/// transport, but the error context is REDACTED: the token rides the URL
/// path, so a failed call may never journal the URL itself.
async fn tg_json(url: &str, body: Option<Value>, timeout: Duration) -> Result<(u16, Value)> {
    let mut req = crate::api::client().request(reqwest::Method::POST, url);
    req = req.timeout(timeout);
    if let Some(v) = body {
        req = req.json(&v);
    }
    let resp = req
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("telegram api call failed: {e}"))?;
    let status = resp.status().as_u16();
    let text = resp.text().await.unwrap_or_default();
    let parsed = serde_json::from_str::<Value>(&text).unwrap_or(Value::String(text.clone()));
    Ok((status, parsed))
}

/// One parsed Telegram message update → (chat_id, user_id, display, text,
/// is_group). `None` skips: non-message updates, bot echoes (is_bot), and
/// messages with no text (stickers/mediaONLY edits ride later, not now).
pub fn parse_message(update: &Value) -> Option<(String, String, String, String, bool)> {
    let msg = update.get("message")?;
    let chat_id = msg["chat"]["id"].as_i64()?.to_string();
    let from = msg.get("from")?;
    // Bot echoes: other bots talking to us are loops waiting to happen.
    if from.get("is_bot").and_then(Value::as_bool) == Some(true) {
        return None;
    }
    let user_id = from["id"].as_i64()?.to_string();
    let display = from["username"]
        .as_str()
        .or_else(|| from["first_name"].as_str())
        .unwrap_or(&user_id)
        .to_string();
    let text = msg["text"].as_str()?.trim().to_string();
    if text.is_empty() {
        return None;
    }
    let is_group = !matches!(msg["chat"]["type"].as_str(), Some("private"));
    Some((chat_id, user_id, display, text, is_group))
}

/// The offset AFTER a processed batch: max update_id + 1 (absent ids →
/// the caller keeps polling without an offset — honest from-zero start).
pub fn next_offset(updates: &[Value]) -> Option<u64> {
    updates
        .iter()
        .filter_map(|u| u["update_id"].as_i64())
        .max()
        .map(|m| (m + 1).max(0) as u64)
}

fn should_relay(cfg: &TelegramConfig, chat_id: &str, is_group: bool, text: &str) -> bool {
    if !cfg.allowed_chats.is_empty() && !cfg.allowed_chats.iter().any(|c| c == chat_id) {
        return false;
    }
    if !is_group {
        return true; // private chats always relay
    }
    cfg.respond_to == "all"
        || text
            .to_lowercase()
            .contains(&format!("@{}", cfg.bot_username.to_lowercase()))
}

async fn send_reply(tcfg: &TelegramConfig, chat_id: &str, text: &str) -> Result<()> {
    for chunk in chunk_utf8(text, 4000) {
        let (code, v) = tg_json(
            &tcfg.call_url("sendMessage"),
            Some(json!({ "chat_id": chat_id, "text": chunk })),
            Duration::from_secs(20),
        )
        .await?;
        anyhow::ensure!(code == 200, "telegram sendMessage failed ({code}): {v}");
    }
    Ok(())
}

pub async fn run(
    cfg: Arc<BotConfig>,
    state: BotState,
    mut tcfg: TelegramConfig,
    owner: Option<u64>,
) -> Result<()> {
    tcfg.validate()?;
    tracing::info!(
        transport = "telegram",
        respond_to = %tcfg.respond_to,
        allowed_chats = tcfg.allowed_chats.len(),
        "starting (getUpdates long-poll)"
    );

    let deliverer: DeliveryFn = {
        let tcfg = tcfg.clone();
        Arc::new(move |chat: &str, text: String| {
            let tcfg = tcfg.clone();
            let chat = chat.to_string();
            Box::pin(async move { send_reply(&tcfg, &chat, &text).await })
        })
    };

    let mut offset: Option<u64> = None;
    loop {
        // Long poll: pass the offset + the server-side wait; on success
        // advance the offset so a restart never replays old messages.
        let mut url = tcfg.call_url("getUpdates");
        if let Some(o) = offset {
            url.push_str(&format!("?offset={o}"));
        }
        let wait = tcfg.poll_timeout_secs;
        url.push_str(if offset.is_some() { "&" } else { "?" });
        url.push_str(&format!("timeout={wait}"));
        let res = tg_json(&url, None, Duration::from_secs(u64::from(wait) + 10)).await;
        let (code, v) = match res {
            Ok(pair) => pair,
            Err(e) => {
                // Transient network wobble: quieter reconnect inside the
                // adapter (supervision respawn is for config-fatals).
                tracing::warn!(transport = "telegram", error = %e, "getUpdates poll failed — retrying");
                tokio::time::sleep(Duration::from_secs(5)).await;
                continue;
            }
        };
        if code == 409 {
            anyhow::bail!(
                "telegram: 409 conflict — another getUpdates consumer (webhook or second poller) holds this token; fix the config, restart bot"
            );
        }
        anyhow::ensure!(code == 200, "telegram getUpdates failed ({code}): {v}");
        let updates = v
            .get("result")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for update in &updates {
            // Single-consumer rule (approval.rs): when this lane's token
            // is also the [approvals] token, this loop is the ONLY
            // getUpdates consumer — approval buttons/commands arrive here
            // and must be resolved, never relayed as chat.
            if cfg.approvals.telegram_bot_token.as_deref() == Some(tcfg.token.as_str())
                && crate::approval::handle_transport_update(&cfg, update).await
            {
                continue;
            }
            let Some((chat_id, user_id, display, text, is_group)) = parse_message(update) else {
                continue;
            };
            if !should_relay(&tcfg, &chat_id, is_group, &text) {
                continue;
            }
            let inc = Incoming {
                platform: "telegram",
                channel: chat_id,
                user: user_id,
                display,
                text,
            };
            let cfg2 = Arc::clone(&cfg);
            let state2 = Arc::clone(&state);
            let deliverer2 = Arc::clone(&deliverer);
            tokio::spawn(async move {
                handle_incoming(&cfg2, &state2, inc, owner, deliverer2).await;
            });
        }
        offset = next_offset(&updates).or(offset);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(chat_type: &str, chat_id: i64, uid: i64, user: &str, text: &str) -> Value {
        json!({
            "update_id": 77,
            "message": {
                "message_id": 1,
                "text": text,
                "chat": { "id": chat_id, "type": chat_type },
                "from": { "id": uid, "is_bot": false, "username": user }
            }
        })
    }

    #[test]
    fn parses_message_and_skips_bots_and_media() {
        let m = msg("private", 42, 7, "noodly", "hello mesh");
        assert_eq!(
            parse_message(&m).unwrap(),
            (
                "42".into(),
                "7".into(),
                "noodly".into(),
                "hello mesh".into(),
                false
            )
        );
        // Bot echo skipped.
        let mut bot = msg("private", 42, 99, "otherbot", "echo");
        bot["message"]["from"]["is_bot"] = json!(true);
        assert_eq!(parse_message(&bot), None);
        // Media-only (no text) skipped; non-message updates skipped.
        let mut media = msg("group", 42, 7, "noodly", "");
        media["message"].as_object_mut().unwrap().remove("text");
        assert_eq!(parse_message(&media), None);
        assert_eq!(parse_message(&json!({"update_id": 5})), None);
    }

    #[test]
    fn relay_gate_is_chat_scoped_and_mention_scoped() {
        let mut c = TelegramConfig {
            api_url: DEFAULT_API_URL.into(),
            token: "t".into(),
            bot_username: "meshybot".into(),
            allowed_chats: vec![],
            respond_to: "mention".into(),
            poll_timeout_secs: 25,
        };
        assert!(c.validate().is_ok());
        // Private always; group only on mention.
        assert!(should_relay(&c, "42", false, "anything"));
        assert!(should_relay(&c, "43", true, "@meshybot what's up"));
        assert!(should_relay(&c, "43", true, "ping @MESHYBOT please"));
        assert!(!should_relay(&c, "43", true, "no mention here"));
        // all relays groups regardless.
        let mut relay = c.clone();
        relay.respond_to = "all".into();
        assert!(relay.validate().is_ok());
        assert!(should_relay(&relay, "43", true, "anything"));
        // allowed_chats scoping shuts the rest down entirely.
        relay.allowed_chats = vec!["42".into()];
        assert!(should_relay(&relay, "42", true, "anything"));
        assert!(!should_relay(&relay, "43", false, "anything"));
    }

    #[test]
    fn offset_advances_to_max_plus_one_and_survives_absent_ids() {
        let batch = vec![
            json!({ "update_id": 10 }),
            json!({ "update_id": 12, "message": {} }),
            json!({ "no_id": true }),
        ];
        assert_eq!(next_offset(&batch), Some(13));
        assert_eq!(next_offset(&[json!({"no_id": true})]), None);
    }

    #[test]
    fn call_url_shape_and_validation() {
        let mut c = TelegramConfig {
            api_url: "http://127.0.0.1:9999/".into(),
            token: "1:tok".into(),
            bot_username: "b".into(),
            allowed_chats: vec![],
            respond_to: "mention".into(),
            poll_timeout_secs: 0,
        };
        c.validate().unwrap();
        assert_eq!(c.api_url, "http://127.0.0.1:9999");
        assert_eq!(
            c.call_url("getUpdates"),
            "http://127.0.0.1:9999/bot1:tok/getUpdates"
        );
        // Clamped poll window.
        assert_eq!(c.poll_timeout_secs, 5);
        let mut bogus = TelegramConfig {
            token: String::new(),
            ..c.clone()
        };
        assert!(bogus.validate().is_err());
    }
}
