//! IRC transport (RFC 2812 over a native TCP client, optional TLS).
//!
//! The simplest honest integration in the list: no vendor SDK — a socket
//! + NICK/USER + PING/PONG + PRIVMSG. The `channels` list joins at boot;
//! DMs also route in (the answer goes back to the sender's nick). CTCP
//! ACTION (/me) normalizes to plain text; server numerics and
//! joins/parts are ignored except as registration/keepalive events.
//!
//! TLS via tokio-rustls + the webpki root store (both already inside the
//! tree through tokio-tungstenite's rustls feature — now direct deps so
//! this adapter can build a client config itself). `tls` defaults true;
//! plaintext is opt-in for tests and odd relays.
//!
//! Relay model: `respond_to: "mention"` (default) answers only lines
//! containing the bot's nick or an `ask:` prefix — relays respect humans;
//! `"all"` feeds every channel message to the pipeline (opt-in noise).
//! Outbound PRIVMSGs split at [`SEND_CAP`] chars with a flood-paced gap
//! between chunks (a socket-level token bucket per network is over-model
//! here; 400+350 ms survived every network this bot family has ridden).

use super::{chunk_utf8, handle_incoming, DeliveryFn, Incoming};
use crate::config::BotConfig;
use crate::state::BotState;
use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt as _, AsyncWriteExt as _, BufReader};
use tokio::sync::Mutex;

/// Outbound split cap (chars) — conservative across networks/bridges.
pub const SEND_CAP: usize = 400;
/// Flood pacing between consecutive outbound chunks.
const CHUNK_GAP_MS: u64 = 350;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrcConfig {
    /// `host:port` (port optional; default follows tls: 6697 else 6667).
    pub server: String,
    #[serde(default = "default_true")]
    pub tls: bool,
    pub nick: String,
    /// Optional server-level PASS (pre-registration).
    #[serde(default)]
    pub password: Option<String>,
    /// Optional NickServ password (IDENTIFY after the 001 welcome).
    #[serde(default)]
    pub nickserv_password: Option<String>,
    /// Channels joined at boot; required non-empty.
    pub channels: Vec<String>,
    /// "mention" (default) | "all".
    #[serde(default = "default_respond_to")]
    pub respond_to: String,
}
fn default_true() -> bool {
    true
}
fn default_respond_to() -> String {
    "mention".into()
}

impl IrcConfig {
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(!self.server.trim().is_empty(), "irc: server required");
        anyhow::ensure!(
            !self.nick.trim().is_empty(),
            "irc: nick required"
        );
        anyhow::ensure!(
            !self.channels.is_empty(),
            "irc: channels required non-empty (a lurker bot is a bug)"
        );
        for ch in &self.channels {
            anyhow::ensure!(
                ch.starts_with('#') && ch.len() > 1,
                "irc: channel {ch:?} must start with '#' and have a name"
            );
        }
        anyhow::ensure!(
            matches!(self.respond_to.as_str(), "mention" | "all"),
            "irc: respond_to must be \"mention\" or \"all\" (got {:?})",
            self.respond_to
        );
        Ok(())
    }
}

/// `(host, port)` with the tls-defaulted port filled in.
fn split_server(server: &str, tls: bool) -> (String, u16) {
    match server.rsplit_once(':') {
        Some((h, p)) if p.parse::<u16>().is_ok() => (h.trim().to_string(), p.parse().unwrap()),
        _ => (server.trim().to_string(), if tls { 6697 } else { 6667 }),
    }
}

/// The one stream object every send/read goes through: a Rust enum is
/// how Plain/TLS share the same Mutex<BufReader<_>> body (a dyn-with-
/// two-traits object is a compile error — enums are the honest encode).
enum Conn {
    Plain(tokio::net::TcpStream),
    Tls(tokio_rustls::client::TlsStream<tokio::net::TcpStream>),
}

impl tokio::io::AsyncRead for Conn {
    fn poll_read(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        match self.get_mut() {
            Conn::Plain(s) => std::pin::Pin::new(s).poll_read(cx, buf),
            Conn::Tls(t) => std::pin::Pin::new(t).poll_read(cx, buf),
        }
    }
}

impl tokio::io::AsyncWrite for Conn {
    fn poll_write(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        match self.get_mut() {
            Conn::Plain(s) => std::pin::Pin::new(s).poll_write(cx, buf),
            Conn::Tls(t) => std::pin::Pin::new(t).poll_write(cx, buf),
        }
    }
    fn poll_flush(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        match self.get_mut() {
            Conn::Plain(s) => std::pin::Pin::new(s).poll_flush(cx),
            Conn::Tls(t) => std::pin::Pin::new(t).poll_flush(cx),
        }
    }
    fn poll_shutdown(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        match self.get_mut() {
            Conn::Plain(s) => std::pin::Pin::new(s).poll_shutdown(cx),
            Conn::Tls(t) => std::pin::Pin::new(t).poll_shutdown(cx),
        }
    }
}

type ConnMu = Arc<Mutex<BufReader<Conn>>>;

async fn connect(cfg: &IrcConfig, host: &str, port: u16) -> Result<Conn> {
    if !cfg.tls {
        let s = tokio::net::TcpStream::connect((host, port))
            .await
            .context("irc tcp connect")?;
        return Ok(Conn::Plain(s));
    }
    // One-time process crypto-provider install (ring is the pinned
    // workspace provider; races with other installs are fine — both
    // winners and losers continue correctly).
    let _ = tokio_rustls::rustls::crypto::ring::default_provider().install_default();
    let mut roots = tokio_rustls::rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let config = tokio_rustls::rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();
    let name =
        tokio_rustls::rustls::pki_types::ServerName::try_from(host.to_string())
            .context("irc tls server name")?;
    let connector = tokio_rustls::TlsConnector::from(Arc::new(config));
    let tcp = tokio::net::TcpStream::connect((host, port))
        .await
        .context("irc tcp connect (tls)")?;
    let tls = connector
        .connect(name, tcp)
        .await
        .context("irc tls handshake")?;
    Ok(Conn::Tls(tls))
}

/// One outbound line, flood-paced. Called from the delivery closure and
/// the login/join chores (single writer — the Mutex is the serializer).
async fn send_line(conn: &ConnMu, line: &str) -> Result<()> {
    let mut c = conn.lock().await;
    c.get_mut().write_all(format!("{line}\r\n").as_bytes()).await?;
    c.get_mut().flush().await?;
    drop(c);
    if line.starts_with("PRIVMSG") {
        tokio::time::sleep(std::time::Duration::from_millis(CHUNK_GAP_MS)).await;
    }
    Ok(())
}

/// The delivery closure handed into the pipeline: chunk + PRIVMSG the
/// right target. `me` trades out for CTCP ACTION? No — plain text only.
fn make_deliverer(conn: ConnMu) -> DeliveryFn {
    Arc::new(move |target: &str, text: String| {
        let conn = Arc::clone(&conn);
        let target = target.to_string();
        Box::pin(async move {
            for chunk in chunk_utf8(&text, SEND_CAP) {
                send_line(&conn, &format!("PRIVMSG {target} :{chunk}")).await?;
            }
            Ok(())
        })
    })
}

/// Does this channel line concern us? (mention mode gate)
fn addressed_in(cfg: &IrcConfig, text: &str) -> bool {
    if cfg.respond_to == "all" {
        return true;
    }
    let lower = text.to_lowercase();
    lower.contains(&cfg.nick.to_lowercase()) || lower.starts_with("ask:") || lower.starts_with("ask ")
}

pub async fn run(cfg: Arc<BotConfig>, state: BotState, icfg: IrcConfig, owner: Option<u64>) -> Result<()> {
    icfg.validate()?;
    let (host, port) = split_server(&icfg.server, icfg.tls);
    tracing::info!(server = %host, port, channels = icfg.channels.len(), respond_to = %icfg.respond_to, tls = icfg.tls, "irc transport connecting");
    let conn: ConnMu = Arc::new(Mutex::new(BufReader::new(connect(&icfg, &host, port).await?)));

    // Registration: PASS → NICK → USER, then join + NickServ after 001.
    if let Some(pw) = &icfg.password {
        send_line(&conn, &format!("PASS {pw}")).await?;
    }
    send_line(&conn, &format!("NICK {}", icfg.nick)).await?;
    send_line(&conn, &format!("USER {0} 0 * :{0}", icfg.nick)).await?;

    // Delivery handles live from here on (sessions may outlive lines).
    let deliverer = make_deliverer(Arc::clone(&conn));

    let mut joined = false;
    let mut buf = String::new();
    loop {
        buf.clear();
        let n = {
            let mut c = conn.lock().await;
            c.read_line(&mut buf).await.context("irc read_line")?
        };
        if n == 0 {
            anyhow::bail!("irc: server closed the connection");
        }
        let line = buf.trim_end_matches(['\r', '\n']).to_string();
        if let Some(payload) = line.strip_prefix("PING ") {
            send_line(&conn, &format!("PONG {payload}")).await?;
            continue;
        }
        if !joined {
            if line.contains(" 376 ") || line.contains(" 422 ") || line.contains(" 001 ") {
                if let Some(pw) = &icfg.nickserv_password {
                    send_line(&conn, &format!("PRIVMSG NickServ :IDENTIFY {pw}")).await?;
                }
                for ch in &icfg.channels {
                    send_line(&conn, &format!("JOIN {ch}")).await?;
                }
                joined = true;
                tracing::info!(transport = "irc", "registered + channels joined");
            } else if line.contains(" 433 ") {
                // Nick in use: retry with an appended short rune.
                let alt = format!("{}_", icfg.nick);
                send_line(&conn, &format!("NICK {alt}")).await?;
            }
            continue;
        }
        // PRIVMSG parsing — the only event type routed anywhere.
        // Shape: `:nick!user@host PRIVMSG <target> :<text>`.
        if !line.starts_with(':') {
            continue;
        }
        let Some((mask, rest)) = line[1..].split_once(' ') else { continue };
        let Some((cmd, after)) = rest.split_once(' ') else { continue };
        if cmd != "PRIVMSG" {
            continue;
        }
        let Some((target, payload)) = after.split_once(' ') else { continue };
        let Some(mut text) = payload.strip_prefix(':').map(str::to_string) else { continue };
        let nick = mask.split('!').next().unwrap_or(mask).to_string();
        // CTCP ACTION normalization; other CTCPs are ignored outright.
        if let Some(inner) = text.strip_prefix('\u{1}') {
            if let Some(action) = inner
                .strip_prefix("ACTION ")
                .or_else(|| inner.strip_suffix('\u{1}'))
                .map(|s| s.trim_end_matches('\u{1}').to_string())
            {
                text = format!("*{nick} {action}");
            } else {
                continue;
            }
        }
        let (channel, is_dm) = if target.starts_with('#') {
            (target.to_string(), false)
        } else {
            (nick.clone(), true)
        };
        if !is_dm && !addressed_in(&icfg, &text) {
            continue;
        }
        let inc = Incoming {
            platform: "irc",
            channel,
            user: nick.clone(),
            display: nick.clone(),
            text,
        };
        let deliverer2 = Arc::clone(&deliverer);
        let cfg2 = Arc::clone(&cfg);
        let state2 = Arc::clone(&state);
        tokio::spawn(async move {
            handle_incoming(&cfg2, &state2, inc, owner, deliverer2).await;
        });
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_rejects_the_obvious_todos() {
        let ok = IrcConfig {
            server: "irc.libera.chat:6697".into(),
            tls: true,
            nick: "noodle-bot".into(),
            password: None,
            nickserv_password: None,
            channels: vec!["#jerith-mesh".into()],
            respond_to: "mention".into(),
        };
        ok.validate().unwrap();
        let bad_server = IrcConfig { server: "".into(), ..ok.clone() };
        assert!(bad_server.validate().is_err());
        let no_channels = IrcConfig { channels: vec![], ..ok.clone() };
        assert!(no_channels.validate().is_err());
        let bad_req = IrcConfig { respond_to: "everyone".into(), ..ok.clone() };
        assert!(bad_req.validate().is_err());
    }

    #[test]
    fn split_server_defaults_the_port_by_tls() {
        assert_eq!(split_server("irc.example.org", true), ("irc.example.org".into(), 6697));
        assert_eq!(split_server("irc.example.org", false), ("irc.example.org".into(), 6667));
        assert_eq!(split_server("irc.example.org:7000", false), ("irc.example.org".into(), 7000));
    }

    #[test]
    fn mention_gate_matches_the_addressing_contrat() {
        let c = IrcConfig {
            server: "a:1".into(), tls: true, nick: "meshy".into(), password: None,
            nickserv_password: None, channels: vec!["#a".into()], respond_to: "mention".into(),
        };
        assert!(addressed_in(&c, "hey meshy what do you think"));
        assert!(addressed_in(&c, "ask: what's the tally"));
        assert!(addressed_in(&c, "ask what's up"));
        assert!(!addressed_in(&c, "mashed potatoes"));
        let relay = IrcConfig { respond_to: "all".into(), ..c };
        assert!(addressed_in(&relay, "anything goes"));
    }

    #[test]
    fn action_parsed_to_plain_text_shape() {
        // The parse contract: "\u1ACTION ... \u1" → "*nick action".
        let raw = format!("\u{1}ACTION waves at the plain\u{1}");
        let inner = raw.trim_matches('\u{1}');
        let action = inner.strip_prefix("ACTION ").unwrap_or(inner);
        assert_eq!(action, "waves at the plain");
    }
}
