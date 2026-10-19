//! Nextcloud Talk transport — OCS REST over the room's HTTP API.
//! Inbound: long-poll `GET /ocs/v2.php/apps/spreed/api/v1/room/{token}/messages`
//! with `lookIntoFuture=1&timeout=10` (Talk's own wait-for-new-message
//! mode) plus a `sinceMessageId` cursor; outbound: POST the same URL
//! with `{ message }`. Auth: app password (username + app password via
//! Basic auth) — no OAuth dance, and app passwords are revocable
//! per device, the minimal-permission story.
//!
//! Split cap: 3200 (Talk messages render fine at 4k but folds UI chat
//! bubbles aggressively; 3200 stays a readable single-post per chunk).

use super::{DeliveryFn, Incoming, chunk_utf8, handle_incoming, http_json};
use crate::config::BotConfig;
use crate::state::BotState;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NextcloudConfig {
    /// https://cloud.example.org (no trailing slash).
    pub site_url: String,
    pub username: String,
    /// App password (Settings → Security → create per-device).
    pub app_password: String,
    /// Room tokens to serve (the opaque code in the room URL); empty =
    /// none (Talk requires explicit rooms — a lurking participant is a
    /// silent bug, never a default).
    pub rooms: Vec<String>,
}
impl NextcloudConfig {
    pub fn validate(&mut self) -> Result<()> {
        let site = self.site_url.trim_end_matches('/').to_string();
        anyhow::ensure!(
            site.starts_with("https://")
                || site.starts_with("http://127.0.0.1")
                || site.starts_with("http://localhost"),
            "nextcloud: site_url must be https or loopback http"
        );
        self.site_url = site;
        anyhow::ensure!(
            !self.username.trim().is_empty(),
            "nextcloud: username required"
        );
        anyhow::ensure!(
            !self.app_password.trim().is_empty(),
            "nextcloud: app_password required"
        );
        anyhow::ensure!(
            !self.rooms.is_empty(),
            "nextcloud: rooms required non-empty (explicit participation)"
        );
        for r in &self.rooms {
            anyhow::ensure!(
                !r.trim().is_empty() && !r.contains('/'),
                "nextcloud: room token {r:?} malformed"
            );
        }
        Ok(())
    }
    fn basic(&self) -> String {
        use base64::Engine as _;
        base64::engine::general_purpose::STANDARD
            .encode(format!("{}:{}", self.username, self.app_password))
    }
}

/// OCS envelope unwrapping: `{ ocs: { meta: ..., data: ... } }`.
fn ocs_data(v: &Value) -> Option<&Value> {
    v.pointer("/ocs/data")
}

/// One poll window against one room: returns fresh messages
/// (id, actor_id, actor_display, message) NEWER than `since`.
pub async fn poll_room(
    scfg: &NextcloudConfig,
    room: &str,
    since: &str,
) -> Result<Vec<(String, String, String, String)>> {
    let url = format!(
        "{}/ocs/v2.php/apps/spreed/api/v1/room/{}/messages?lookIntoFuture=1&includeLastKnown=0&timeout=10{}",
        scfg.site_url,
        room,
        if since.is_empty() {
            String::new()
        } else {
            format!("&sinceMessageId={since}")
        }
    );
    let (code, v) = http_json(
        reqwest::Method::GET,
        &url,
        Some(&scfg.basic()),
        None,
        Duration::from_secs(20),
    )
    .await?;
    anyhow::ensure!(
        (200..300).contains(&code),
        "nextcloud poll room {room} failed ({code}): {v}"
    );
    let mut out = Vec::new();
    for m in ocs_data(&v)
        .and_then(|d| d.get("messages").and_then(Value::as_array).cloned())
        .unwrap_or_default()
    {
        // (id, actorId, actorDisplayName, "message" field)
        let Some(id) = m["id"]
            .as_i64()
            .map(|i| i.to_string())
            .or_else(|| m["id"].as_str().map(str::to_string))
        else {
            continue;
        };
        let actor = m["actorId"].as_str().unwrap_or_default().to_string();
        if actor.is_empty() || actor == scfg.username {
            continue; // own echo
        }
        let display = m["actorDisplayName"]
            .as_str()
            .filter(|s| !s.is_empty())
            .unwrap_or(&actor)
            .to_string();
        let text = m["message"].as_str().unwrap_or_default().to_string();
        if text.trim().is_empty() {
            continue;
        }
        // System messages (joins, calls) have a non-chat `messageType`.
        if m["messageType"].as_str().is_some_and(|t| t != "comment") {
            continue;
        }
        out.push((id, actor, display, text));
    }
    Ok(out)
}

pub async fn run(
    cfg: Arc<BotConfig>,
    state: BotState,
    mut ncfg: NextcloudConfig,
    owner: Option<u64>,
) -> Result<()> {
    ncfg.validate()?;
    tracing::info!(transport = "nextcloud", site = %ncfg.site_url, rooms = ncfg.rooms.len(), "starting");
    let deliverer: DeliveryFn = {
        let ncfg2 = ncfg.clone();
        Arc::new(move |room: &str, text: String| {
            let ncfg2 = ncfg2.clone();
            let room = room.to_string();
            Box::pin(async move {
                for chunk in chunk_utf8(&text, 3200) {
                    let url = format!(
                        "{}/ocs/v2.php/apps/spreed/api/v1/room/{}/messages",
                        ncfg2.site_url, room
                    );
                    let (code, v) = http_json(
                        reqwest::Method::POST,
                        &url,
                        Some(&ncfg2.basic()),
                        Some(json!({ "message": chunk })),
                        Duration::from_secs(20),
                    )
                    .await?;
                    anyhow::ensure!(
                        (200..300).contains(&code),
                        "nextcloud post to {room} failed ({code}): {v}"
                    );
                }
                Ok(())
            })
        })
    };

    // Per-room cursors; one poll pass round-robin every loop (long-poll
    // on the FIRST room only would starve others — each poll has an
    // internal ≤10 s wait, and rooms stay few by config).
    let mut cursors: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let rooms = ncfg.rooms.clone();
    let mut idx = 0usize;
    loop {
        let room = &rooms[idx % rooms.len()];
        idx += 1;
        let since = cursors.get(room).cloned().unwrap_or_default();
        match poll_room(&ncfg, room, &since).await {
            Ok(msgs) => {
                for (id, actor, display, text) in msgs {
                    cursors.insert(room.clone(), id.clone());
                    let inc = Incoming {
                        platform: "nextcloud",
                        channel: room.clone(),
                        user: actor.clone(),
                        display: display.clone(),
                        text,
                    };
                    handle_incoming(&cfg, &state, inc, owner, Arc::clone(&deliverer)).await;
                }
            }
            Err(e) => {
                // Transient errors just advance the cursor-less loop; a
                // hard failure (401/404) bubbles to the supervisor.
                let hard = e.to_string().contains("401") || e.to_string().contains("404");
                if hard {
                    anyhow::bail!("nextcloud fatal room poll: {e}");
                }
                tracing::debug!(error = %e, room, "nextcloud poll blip");
            }
        }
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_requires_https_or_loopback_and_rooms() {
        let mut ok = NextcloudConfig {
            site_url: "https://cloud.example.org/".into(),
            username: "meshy".into(),
            app_password: "app-pass".into(),
            rooms: vec!["abc123".into()],
        };
        ok.validate().unwrap();
        assert_eq!(ok.site_url, "https://cloud.example.org");
        let mut bad_site = NextcloudConfig {
            site_url: "http://evil.example.org".into(),
            ..ok.clone()
        };
        assert!(bad_site.validate().is_err());
        let mut no_rooms = NextcloudConfig {
            rooms: vec![],
            ..ok.clone()
        };
        assert!(no_rooms.validate().is_err());
        let mut bad_room = NextcloudConfig {
            rooms: vec!["a/b".into()],
            ..ok
        };
        assert!(bad_room.validate().is_err());
    }

    #[test]
    fn basic_auth_is_standard_base64() {
        use base64::Engine as _;
        let c = NextcloudConfig {
            site_url: "https://c.example.org".into(),
            username: "user".into(),
            app_password: "secret".into(),
            rooms: vec!["r".into()],
        };
        assert_eq!(
            c.basic(),
            base64::engine::general_purpose::STANDARD.encode("user:secret")
        );
    }
}
