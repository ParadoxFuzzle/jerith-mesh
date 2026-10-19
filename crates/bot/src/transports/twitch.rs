//! Twitch chat transport — IRC over WSS (`wss://irc-ws.chat.twitch.tv`),
//! the same RFC 2812 shape irc.rs parses with Twitch's tag prefix on
//! top. Anonymous (no token) join = read-only: the deliverer reports a
//! typed error when someone sends while no `oauth_token` is set — a
//! loudly failing relay beats a silently deaf one, same discipline as
//! the module doc's fail-loud clause.
//!
//! Twitch tags: the first line token may start `@`
//! (`@badge-info=…;display-name=Foo;…`); the display name falls back to
//! the login from the nick mask. PING/PONG uses Twitch's one-word form.
//! Outbound PRIVMSG splits at 480 chars with the shared chunker.

use super::{DeliveryFn, Incoming, chunk_utf8, handle_incoming};
use crate::config::BotConfig;
use crate::state::BotState;
use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwitchConfig {
    pub nick: String,
    /// User-access-token (oauth:) — REQUIRED to SEND; anonymous joins are
    /// read-only relays by Twitch design.
    #[serde(default)]
    pub oauth_token: Option<String>,
    pub channels: Vec<String>,
}
impl TwitchConfig {
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(!self.nick.trim().is_empty(), "twitch: nick required");
        anyhow::ensure!(
            !self.channels.is_empty(),
            "twitch: channels required non-empty"
        );
        for ch in &self.channels {
            anyhow::ensure!(
                !ch.starts_with('#'),
                "twitch: channel {ch:?} is a bare login (no '#') — the JOIN adds the prefix"
            );
        }
        Ok(())
    }
    fn token_raw(&self) -> Option<&str> {
        self.oauth_token
            .as_deref()
            .map(|t| t.trim())
            .filter(|t| !t.is_empty())
            .map(|t| t.trim_start_matches("oauth:"))
    }
}

/// One line of the protocol (shared parser, unit-tested below).
#[derive(Debug, PartialEq)]
pub enum IrcLine {
    Ping,
    PrivMsg {
        channel: String,
        login: String,
        display: String,
        text: String,
    },
    Join,
    Other,
}

pub fn parse_line(line: &str) -> IrcLine {
    // Split an optional tags token off the front.
    let (tags, body) = match line.strip_prefix('@') {
        Some(rest) => match rest.split_once(' ') {
            Some((t, b)) => (Some(t), b),
            None => return IrcLine::Other,
        },
        None => (None, line),
    };
    if body.starts_with("PING") {
        return IrcLine::Ping;
    }
    let Some(rest) = body.strip_prefix(':') else {
        return IrcLine::Other;
    };
    let Some((mask, after)) = rest.split_once(' ') else {
        return IrcLine::Other;
    };
    if after.starts_with("JOIN ") {
        return IrcLine::Join;
    }
    if let Some((cmd, target_and_text)) = after.split_once(' ')
        && cmd == "PRIVMSG"
        && let Some((target, payload)) = target_and_text.split_once(' ')
    {
        let text = payload.trim_start_matches(':').to_string();
        if !text.is_empty() {
            let login = mask.split('!').next().unwrap_or(mask).to_string();
            let display = tags
                .and_then(|t| {
                    t.split(';').find_map(|kv| {
                        kv.strip_prefix("display-name=")
                            .filter(|v| !v.is_empty())
                            .map(str::to_string)
                    })
                })
                .unwrap_or_else(|| login.clone());
            let channel = target.trim_start_matches('#').to_lowercase();
            return IrcLine::PrivMsg {
                channel,
                login,
                display,
                text,
            };
        }
    }
    IrcLine::Other
}

pub async fn run(
    cfg: Arc<BotConfig>,
    state: BotState,
    tcfg: TwitchConfig,
    owner: Option<u64>,
) -> Result<()> {
    tcfg.validate()
        .map_err(|e| e.context("twitch: bad config — fix and restart"))?;
    tracing::info!(
        transport = "twitch",
        channels = tcfg.channels.len(),
        send = tcfg.token_raw().is_some(),
        "connecting"
    );
    let url = "wss://irc-ws.chat.twitch.tv:443";
    let (ws, _) = tokio_tungstenite::connect_async(url).await?;
    let (mut write, mut read) = ws.split();
    use futures_util::{SinkExt as _, StreamExt as _};
    use tokio_tungstenite::tungstenite::Message;

    // ONE writer task owns the write half; everything outbound (login,
    // joins, PONGs, replies) rides an mpsc command — the websocket's
    // split halves are single-owner, so the deliverer hands lines over
    // the channel instead of the socket. Pacing lives HERE (twitch caps
    // ~20 msgs/30 s in a channel; a slow mesh answer never floods).
    let (out_tx, mut out_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    {
        let writer = async move {
            let mut gap = tokio::time::interval(Duration::from_millis(350));
            gap.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            while let Some(line) = out_rx.recv().await {
                if write
                    .send(Message::text(format!("{line}\r\n")))
                    .await
                    .is_err()
                {
                    break;
                }
                gap.tick().await;
            }
        };
        tokio::spawn(writer);
    }
    let push = |out_tx: &tokio::sync::mpsc::UnboundedSender<String>, line: &str| {
        let _ = out_tx.send(line.to_string());
    };

    push(&out_tx, "CAP REQ :tags");
    push(&out_tx, "PASS SCHMOOPIIE");
    push(&out_tx, &format!("NICK {}", tcfg.nick));
    push(&out_tx, &format!("USER {0} 8 * :{0}", tcfg.nick));
    for ch in &tcfg.channels {
        push(&out_tx, &format!("JOIN #{ch}"));
    }

    // Deliverer: PRIVMSG bundle only with a token; typed loud error
    // otherwise (a read-only relay that fails loudly, never silently).
    let deliverer: DeliveryFn = {
        let token = tcfg.token_raw().map(str::to_string);
        let out_tx = out_tx.clone();
        Arc::new(move |channel: &str, text: String| {
            let token = token.clone();
            let out_tx = out_tx.clone();
            let channel = format!("#{channel}");
            Box::pin(async move {
                token.context(
                    "twitch: cannot SEND — no oauth_token configured (read-only relay by design)",
                )?;
                for chunk in chunk_utf8(&text, 480) {
                    let _ = out_tx.send(format!("PRIVMSG {channel} :{chunk}"));
                }
                Ok(())
            })
        })
    };

    while let Some(frame) = read.next().await {
        let msg = frame?;
        let Message::Text(raw) = msg else { continue };
        for line in raw.split("\r\n").filter(|l| !l.is_empty()) {
            match parse_line(line) {
                IrcLine::Ping => push(&out_tx, "PONG :tmi.twitch.tv"),
                IrcLine::Join => {}
                IrcLine::PrivMsg {
                    channel,
                    login,
                    display,
                    text,
                } => {
                    let inc = Incoming {
                        platform: "twitch",
                        channel,
                        user: login.clone(),
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
                IrcLine::Other => {}
            }
        }
    }
    anyhow::bail!("twitch socket closed")
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_handles_tags_display_name_and_channel() {
        let line = "@badge-info=;display-name=MoodyGamer;room-id=123 :moodygamer!moodygamer@moodygamer.tmi.twitch.tv PRIVMSG #noodl :GG what's the tally";
        let parsed = parse_line(line);
        assert_eq!(
            parsed,
            IrcLine::PrivMsg {
                channel: "noodl".into(),
                login: "moodygamer".into(),
                display: "MoodyGamer".into(),
                text: "GG what's the tally".into()
            }
        );
    }

    #[test]
    fn parse_anon_privmsg_and_ping() {
        let l1 = ":anon!anon@anon.tmi.twitch.tv PRIVMSG #noodl :hello mesh";
        assert!(matches!(parse_line(l1), IrcLine::PrivMsg { display, .. } if display == "anon"));
        assert_eq!(parse_line("PING :tmi.twitch.tv"), IrcLine::Ping);
        assert_eq!(
            parse_line(":bot!bot@bot.tmi.twitch.tv JOIN #noodl"),
            IrcLine::Join
        );
        assert_eq!(parse_line("garbage"), IrcLine::Other);
    }

    #[test]
    fn config_requires_bare_channels_and_sends_loud_without_token() {
        let c = TwitchConfig {
            nick: "jerithbot".into(),
            oauth_token: None,
            channels: vec!["noodl".into()],
        };
        c.validate().unwrap();
        let bad = TwitchConfig {
            channels: vec!["#noodl".into()],
            ..c.clone()
        };
        assert!(bad.validate().is_err());
        assert!(c.token_raw().is_none());
        let with = TwitchConfig {
            oauth_token: Some("oauth:abcd".into()),
            ..c
        };
        assert_eq!(with.token_raw(), Some("abcd"));
    }
}
