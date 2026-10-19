//! Mattermost transport — the websocket Events API at `/api/v4/websocket`
//! (PAT bearer auth + authentication_challenge) with `posted` events
//! routed to the dialogue core; replies POST `/api/v4/posts`. Personal
//! access token auth is the whole story: no OAuth dance, no public
//! endpoint, no webhook — a PAT reads and writes everything a bot
//! account may, the mattermost-native equivalent of a Session cookie.
//!
//! Respond mode ("mention" default): only messages mentioning
//! `bot_username` relays. "all" relays the whole channel (busy team
//! channels get noisy — opt-in).
//!
//! Split cap: 4000 (MM's hard message cap is ~8000; keep under the UI
//! fold with the shared rune-safe chunker).

use super::{DeliveryFn, Incoming, chunk_utf8, handle_incoming, http_json};
use crate::config::BotConfig;
use crate::state::BotState;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MattermostConfig {
    /// https://chat.example.org (no trailing slash, no /api).
    pub site_url: String,
    /// Personal access token (Bot account → Settings → Security → PAT).
    pub token: String,
    /// Bot username for mention detection (mention mode).
    pub bot_username: String,
    /// Restrict relay to these channel ids; empty = every channel the
    /// bot account can see.
    #[serde(default)]
    pub channels: Vec<String>,
    /// mention | all
    #[serde(default = "default_respond_to")]
    pub respond_to: String,
}
fn default_respond_to() -> String {
    "mention".into()
}

impl MattermostConfig {
    pub fn validate(&mut self) -> Result<()> {
        let site = self.site_url.trim_end_matches('/').to_string();
        anyhow::ensure!(
            site.starts_with("https://") || site.starts_with("http://"),
            "mattermost: site_url must be http(s)"
        );
        self.site_url = site;
        anyhow::ensure!(!self.token.trim().is_empty(), "mattermost: token required");
        anyhow::ensure!(
            matches!(self.respond_to.as_str(), "mention" | "all"),
            "mattermost: respond_to must be \"mention\" | \"all\""
        );
        if self.respond_to == "mention" {
            anyhow::ensure!(
                !self.bot_username.trim().is_empty(),
                "mattermost: bot_username required in mention mode"
            );
        }
        Ok(())
    }
    fn ws_url(&self) -> String {
        let base = self.site_url.trim_end_matches('/');
        base.replacen("http", "ws", 1) + "/api/v4/websocket"
    }
}

/// Parse one `posted` event's nested post object (unit-tested shape).
pub fn parse_posted(ev: &Value) -> Option<(String, String, String, String)> {
    // (channel_id, user_id, display, message)
    if ev["event"].as_str()? != "posted" {
        return None;
    }
    let data = &ev["data"];
    let post: Value = serde_json::from_str(data["post"].as_str()?).ok()?;
    let channel = post["channel_id"].as_str()?.to_string();
    let user_id = post["user_id"].as_str()?.to_string();
    let message = post["message"].as_str()?.to_string();
    if message.trim().is_empty() {
        return None;
    }
    // Bot accounts carry "props.from_bot" — skip own echoes cheaply.
    if post["props"].get("from_bot").is_some() {
        return None;
    }
    let display = ev["data"]
        .get("sender_name")
        .and_then(Value::as_str)
        .unwrap_or(&user_id)
        .to_string();
    Some((channel, user_id, display, message))
}

fn should_relay(cfg: &MattermostConfig, channel: &str, user_id: &str, message: &str) -> bool {
    if !cfg.channels.is_empty() && !cfg.channels.iter().any(|c| c == channel) {
        return false;
    }
    if cfg.respond_to == "mention" {
        // @mention of the bot OR a direct "mm: " address prefix.
        message
            .to_lowercase()
            .contains(&format!("@{}", cfg.bot_username.to_lowercase()))
            || message.starts_with("mm: ")
    } else {
        !user_id.is_empty()
    }
}

/// One long session: ws connect, authentication_challenge, route events.
async fn session(
    cfg: &BotConfig,
    state: &BotState,
    mcfg: &MattermostConfig,
    owner: Option<u64>,
) -> Result<()> {
    let deliverer: DeliveryFn = {
        let token = mcfg.token.clone();
        let site = mcfg.site_url.clone();
        Arc::new(move |channel: &str, text: String| {
            let token = token.clone();
            let site = site.clone();
            let channel = channel.to_string();
            Box::pin(async move {
                for chunk in chunk_utf8(&text, 4000) {
                    let body = json!({ "channel_id": channel, "message": chunk });
                    let (code, v) = http_json(
                        reqwest::Method::POST,
                        &format!("{site}/api/v4/posts"),
                        Some(&token),
                        Some(body),
                        Duration::from_secs(20),
                    )
                    .await?;
                    anyhow::ensure!(
                        code == 200 || code == 201,
                        "mattermost post failed ({code}): {v}"
                    );
                }
                Ok(())
            })
        })
    };

    loop_session(
        mcfg,
        cfg,
        state,
        owner,
        Arc::clone(&deliverer),
        mcfg.ws_url(),
    )
    .await
}

/// (single reconnect loop with an honest exit — the supervisor backoff
/// only sees fatal problems)
async fn loop_session(
    mcfg: &MattermostConfig,
    cfg: &BotConfig,
    state: &BotState,
    owner: Option<u64>,
    deliverer: DeliveryFn,
    url: String,
) -> Result<()> {
    use futures_util::{SinkExt as _, StreamExt as _};
    use tokio_tungstenite::tungstenite::Message;

    let mut req =
        tokio_tungstenite::tungstenite::client::IntoClientRequest::into_client_request(&url)?;
    req.headers_mut()
        .insert("Authorization", format!("Bearer {}", mcfg.token).parse()?);
    let (ws, _resp) = tokio_tungstenite::connect_async(req).await?;
    let (mut write, mut read) = ws.split();

    // Identity: the challenge is what turns the ws into an authed feed.
    write
        .send(Message::text(json!({"seq": 1, "action": "authentication_challenge", "data": {"token": mcfg.token.clone()}}).to_string()))
        .await?;

    tracing::info!(transport = "mattermost", "websocket authed; routing events");
    while let Some(frame) = read.next().await {
        let msg = frame?;
        let Message::Text(raw) = msg else { continue };
        let env: Value = match serde_json::from_slice(raw.as_bytes()) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if env["status"].as_str() == Some("FAIL") {
            anyhow::bail!("mattermost ws auth/refusal: {env}");
        }
        if let Some((channel, user_id, display, message)) = parse_posted(&env) {
            if !should_relay(mcfg, &channel, &user_id, &message) {
                continue;
            }
            let inc = Incoming {
                platform: "mattermost",
                channel,
                user: user_id.clone(),
                display: display.clone(),
                text: message.clone(),
            };
            let cfg2 = cfg.clone();
            let state2 = Arc::clone(state);
            let deliverer2 = Arc::clone(&deliverer);
            tokio::spawn(async move {
                handle_incoming(&cfg2, &state2, inc, owner, deliverer2).await;
            });
        }
    }
    anyhow::bail!("mattermost websocket closed")
}

pub async fn run(
    cfg: Arc<BotConfig>,
    state: BotState,
    mut mcfg: MattermostConfig,
    owner: Option<u64>,
) -> Result<()> {
    mcfg.validate()?;
    tracing::info!(transport = "mattermost", site = %mcfg.site_url, "starting");
    session(&cfg, &state, &mcfg, owner).await
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn posted_event_parses_channel_user_message_and_skips_bots() {
        let ev = json!({
            "event": "posted",
            "data": { "post": "{\"id\":\"p1\",\"channel_id\":\"c1\",\"user_id\":\"u1\",\"message\":\"hello mesh\"}", "sender_name": "@noodly" }
        });
        assert_eq!(
            parse_posted(&ev),
            Some((
                "c1".into(),
                "u1".into(),
                "@noodly".into(),
                "hello mesh".into()
            ))
        );
        // Bot echo skipped.
        let bot = json!({
            "event": "posted",
            "data": { "post": "{\"id\":\"p2\",\"channel_id\":\"c1\",\"user_id\":\"bot\",\"message\":\"echo\",\"props\":{\"from_bot\":\"true\"}}" }
        });
        assert_eq!(parse_posted(&bot), None);
        assert_eq!(parse_posted(&json!({"event": "channel viewed"})), None);
    }

    #[test]
    fn relay_gate_is_channel_scoped_and_mention_scoped() {
        let mut c = MattermostConfig {
            site_url: "https://chat.example.org/".into(),
            token: "pat".into(),
            bot_username: "meshy".into(),
            channels: vec!["c1".into()],
            respond_to: "mention".into(),
        };
        assert!(c.validate().is_ok());
        assert_eq!(c.site_url, "https://chat.example.org");
        assert!(should_relay(&c, "c1", "u1", "@meshy what's up"));
        assert!(should_relay(&c, "c1", "u1", "could you do it @MESHY?"));
        assert!(!should_relay(&c, "c2", "u1", "@meshy wrong channel"));
        assert!(!should_relay(&c, "c1", "u1", "no mention here"));
        let mut relay = MattermostConfig {
            respond_to: "all".into(),
            ..c
        };
        assert!(relay.validate().is_ok());
        assert!(should_relay(&relay, "c1", "u1", "anything"));
    }

    #[test]
    fn ws_url_turns_http_into_ws_once() {
        let c = MattermostConfig {
            site_url: "https://chat.example.org".into(),
            token: "t".into(),
            bot_username: "b".into(),
            channels: vec![],
            respond_to: "all".into(),
        };
        assert_eq!(c.ws_url(), "wss://chat.example.org/api/v4/websocket");
    }
}
