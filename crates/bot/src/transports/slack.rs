//! Slack transport — Socket Mode (no open ports, mirrors the Telegram
//! approval poller's philosophy): a websocket pulled fresh from
//! `apps.connections.open` (app-level token), envelopes acknowledged,
//! `events_api` message events routed into the dialogue core; replies go
//! out through `chat.postMessage` (bot token). Reconnects internally;
//! a fatal config error surfaces as Err for the supervisor.
//!
//! Self-echo discipline: `auth.test` resolves the bot's own user id at
//! boot; messages authored by it never re-enter the pipeline (a Slack
//! relay loop is one missing filter away otherwise). `message_changed`
//! and other subtypes are skipped — only fresh text routes anywhere.
//!
//! Signature note: Socket Mode carries no message-signature layer (the
//! wss channel IS the auth), so no HMAC surface here — the webinbox.rs
//! Events-API lane remains documented for operators who prefer webhooks.

use super::{DeliveryFn, Incoming, chunk_utf8, handle_incoming, http_json};
use crate::config::BotConfig;
use crate::state::BotState;
use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlackConfig {
    /// App-level token: `xapp-...` (Socket Mode toggle in the app config).
    pub app_token: String,
    /// Bot token: `xoxb-...` (chat:write scope).
    pub bot_token: String,
}
impl SlackConfig {
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.app_token.starts_with("xapp-"),
            "slack: app_token must be an xapp-… token"
        );
        anyhow::ensure!(
            self.bot_token.starts_with("xoxb-"),
            "slack: bot_token must be an xoxb-… token"
        );
        Ok(())
    }
}

/// Shared chat.postMessage deliverer (Socket Mode session and the
/// webinbox Events-API lane both use it).
pub(crate) fn make_deliverer(bot_token: String) -> DeliveryFn {
    Arc::new(move |channel: &str, text: String| {
        let token = bot_token.clone();
        let channel = channel.to_string();
        Box::pin(async move {
            for chunk in chunk_utf8(&text, 3900) {
                let body = json!({ "channel": channel, "text": chunk });
                let (code, v) = http_json(
                    reqwest::Method::POST,
                    "https://slack.com/api/chat.postMessage",
                    Some(&token),
                    Some(body),
                    Duration::from_secs(20),
                )
                .await?;
                anyhow::ensure!(
                    code == 200 && v["ok"] == json!(true),
                    "slack chat.postMessage failed ({code}): {v}"
                );
            }
            Ok(())
        })
    })
}

/// One reconnect-cycle: resolve a fresh wss url from the app token,
/// connect, run until the socket dies; Ok(()) if a clean shutdown.
async fn session(
    cfg: &BotConfig,
    state: &BotState,
    scfg: &SlackConfig,
    owner: Option<u64>,
) -> Result<()> {
    let (code, v) = http_json(
        reqwest::Method::POST,
        "https://slack.com/api/apps.connections.open",
        Some(&scfg.app_token),
        None,
        Duration::from_secs(15),
    )
    .await?;
    anyhow::ensure!(
        code == 200 && v["ok"] == json!(true),
        "slack: connections.open failed ({code}): {v}"
    );
    let url = v["url"]
        .as_str()
        .context("slack: missing wss url in connections.open response")?
        .to_string();

    let (ws, _resp) = tokio_tungstenite::connect_async(&url).await?;
    let (mut write, mut read) = ws.split();

    // Own-user id once per session: the self-echo filter.
    let (_c, me) = http_json(
        reqwest::Method::POST,
        "https://slack.com/api/auth.test",
        Some(&scfg.bot_token),
        None,
        Duration::from_secs(15),
    )
    .await?;
    let me_id = me["user_id"].as_str().unwrap_or_default().to_string();

    // One deliverer per session (chat.postMessage chunked).
    let deliverer: DeliveryFn = {
        let token = scfg.bot_token.clone();
        Arc::new(move |channel: &str, text: String| {
            let token = token.clone();
            let channel = channel.to_string();
            Box::pin(async move {
                for chunk in chunk_utf8(&text, 3900) {
                    let body = json!({ "channel": channel, "text": chunk });
                    let (code, v) = http_json(
                        reqwest::Method::POST,
                        "https://slack.com/api/chat.postMessage",
                        Some(&token),
                        Some(body),
                        Duration::from_secs(20),
                    )
                    .await?;
                    anyhow::ensure!(
                        code == 200 && v["ok"] == json!(true),
                        "slack chat.postMessage failed ({code}): {v}"
                    );
                }
                Ok(())
            })
        })
    };

    use futures_util::{SinkExt as _, StreamExt as _};
    use tokio_tungstenite::tungstenite::Message;
    while let Some(frame) = read.next().await {
        let msg = frame?;
        let Message::Text(raw) = msg else { continue };
        let env: Value = match serde_json::from_slice(raw.as_bytes()) {
            Ok(v) => v,
            Err(_) => continue,
        };
        match env["type"].as_str() {
            // Ack EVERY envelope that carries an id — slack socket mode
            // disconnects ackless connections after 3 s.
            Some("events_api") | Some("interactive_message") | Some("slash_commands") => {
                if let Some(id) = env["envelope_id"].as_str() {
                    let ack = json!({ "type": env["type"].as_str().unwrap_or("events_api"), "envelope_id": id });
                    write.send(Message::text(ack.to_string())).await?;
                }
                if env["type"].as_str() == Some("events_api") {
                    handle_event(
                        cfg,
                        state,
                        &env["payload"]["event"],
                        &me_id,
                        owner,
                        Arc::clone(&deliverer),
                    );
                }
            }
            Some("hello") => tracing::info!(transport = "slack", "socket mode: hello"),
            Some("disconnect") => {
                anyhow::bail!("slack socket mode disconnect ({})", env["reason"]);
            }
            _ => {}
        }
    }
    anyhow::bail!("slack socket closed")
}

fn handle_event(
    cfg: &BotConfig,
    state: &BotState,
    ev: &Value,
    me_id: &str,
    owner: Option<u64>,
    deliverer: DeliveryFn,
) {
    // Message events only; bot-authored or subtyped events never route.
    if ev["type"].as_str() != Some("message") {
        return;
    }
    if ev["subtype"].as_str().is_some() {
        return;
    }
    if ev["bot_id"].is_string() || ev["user"].as_str() == Some(me_id) {
        return;
    }
    let (Some(channel), Some(user), Some(text)) = (
        ev["channel"].as_str().map(str::to_string),
        ev["user"].as_str().map(str::to_string),
        ev["text"].as_str().map(str::to_string),
    ) else {
        return;
    };
    if text.trim().is_empty() {
        return;
    }
    let inc = Incoming {
        platform: "slack",
        channel,
        user: user.clone(),
        // Slack gives no display name on message events; ids do fine.
        display: user,
        text,
    };
    let cfg2 = cfg.clone();
    let state2 = state.clone();
    tokio::spawn(async move {
        handle_incoming(&cfg2, &state2, inc, owner, deliverer).await;
    });
}

pub async fn run(
    cfg: Arc<BotConfig>,
    state: BotState,
    scfg: SlackConfig,
    owner: Option<u64>,
) -> Result<()> {
    scfg.validate()?;
    tracing::info!(transport = "slack", "socket mode starting");
    // Internal reconnect loop: a socket death is routine (slack rotates
    // gateways); the supervisor only sees fatal (config) errors.
    session(&cfg, &state, &scfg, owner).await
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cfg() -> SlackConfig {
        SlackConfig {
            app_token: "xapp-1-abc".into(),
            bot_token: "xoxb-123".into(),
        }
    }

    #[test]
    fn validate_enforces_token_shapes() {
        test_cfg().validate().unwrap();
        let bad = SlackConfig {
            app_token: "sekret".into(),
            ..test_cfg()
        };
        assert!(bad.validate().is_err());
        let bad2 = SlackConfig {
            bot_token: "xoxp-1".into(),
            ..test_cfg()
        };
        assert!(bad2.validate().is_err());
    }

    #[tokio::test]
    async fn event_router_skips_bots_subtypes_and_non_messages() {
        let cfg = std::sync::Arc::new(crate::config::BotConfig::from_parts(
            // Dead port: exactly one fast connection-refused round for the
            // human message — the test asserts the ROUTING decisions, not
            // the pipeline's answer text (a live-gateway port would make
            // the count racy on this machine's real gateway).
            "http://127.0.0.1:18900",
            "http://127.0.0.1:18322",
        ));
        // What IS asserted: exactly one route decision for the human
        // message; the bot-echo, message_changed, and reaction_added
        // events contribute ZERO additional entries.
        let state = crate::state::new_state();
        let routed = Arc::new(std::sync::Mutex::new(Vec::new()));
        let deliverer: DeliveryFn = {
            let routed = Arc::clone(&routed);
            Arc::new(move |_ch: &str, text: String| {
                let r = Arc::clone(&routed);
                Box::pin(async move {
                    r.lock().unwrap().push(text);
                    Ok(())
                })
            })
        };
        // A fresh, human message: routes.
        let ev = json!({"type":"message","channel":"C1","user":"U1","text":"hello there"});
        handle_event(&cfg, &state, &ev, "UBOT", None, Arc::clone(&deliverer));
        // Bot-echo of the same text: skipped.
        let echo = json!({"type":"message","channel":"C1","user":"UBOT","text":"hello there"});
        handle_event(&cfg, &state, &echo, "UBOT", None, Arc::clone(&deliverer));
        let sub = json!({"type":"message","channel":"C1","user":"U1","text":"edited","subtype":"message_changed"});
        handle_event(&cfg, &state, &sub, "UBOT", None, Arc::clone(&deliverer));
        let other = json!({"type":"reaction_added","channel":"C1","user":"U1"});
        handle_event(&cfg, &state, &other, "UBOT", None, deliverer);
        // handle_event spawned the router onto the test's runtime: a
        // yield lets it run before the count (current-thread tokio).
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let got = routed.lock().unwrap();
        assert_eq!(
            got.len(),
            1,
            "the human message routed exactly once; bot/subtype echo and reaction events routed ZERO extra times: {got:?}"
        );
    }
}
