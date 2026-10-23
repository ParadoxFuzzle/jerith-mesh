//! `jerith-house` — the house's own card, over the plugin SDK
//! (`crates/plugin-template`, docs/PLUGIN-SDK.md). Third real consumer
//! (example, mesh pet, house). The composition lives in `lib.rs`
//! (pure, fixture-tested); this binary owns the fetches:
//!
//! - **gateway `/status`** → `$JERITH_GATEWAY` (default
//!   `http://127.0.0.1:8900`), 3 s bound; unreachability is a card
//!   field, never a service error;
//! - **the bot unit** → `systemctl --user show $JERITH_BOT_UNIT`
//!   (default `jerith-bot`), properties bounded by name;
//! - **bot.toml** → `$JERITH_BOT_TOML` (default
//!   `~/.jerith-mesh/bot.toml`) — read ONLY to name the `[transports]`
//!   keys; the card never sees values;
//! - **the journal** → `journalctl --user -u $JERITH_BOT_UNIT` with
//!   `-p warning`, bounded count (`house.journal` takes
//!   `since_minutes` 5–1440, default 120);
//! - **the commit** → `git -C $JERITH_HOUSE_REPO rev-parse --short
//!   HEAD`, when the repo is locatable.
//!
//! Every fetch degrades to `None` for its section — the card answers
//! with what it knows, and says "unknown" where it doesn't.

use anyhow::{Context as _, Result};
use jerith_house_plugin::Sources;
use jerith_plugin_template::{SkillServer, endpoint_addr, skill};
use serde_json::{Value, json};
use std::time::Duration;

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| default.to_string())
}

/// `[plugin] endpoint` + `[plugin] secret` (None = unsigned dev mode)
/// straight out of the manifest — one file rules both halves.
fn read_manifest(path: &str) -> Result<(std::net::SocketAddr, Option<String>)> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("reading manifest {path}"))?;
    // Document parse (toml::Table), NOT `str::parse::<toml::Value>` —
    // a Value FromStr only accepts a single value, so a whole manifest
    // string comes back «unexpected content, expected nothing».
    let file: toml::Table =
        toml::from_str(&raw).with_context(|| format!("{path}: not valid TOML"))?;
    let plugin = file
        .get("plugin")
        .and_then(toml::Value::as_table)
        .context("manifest has no [plugin] table")?;
    let endpoint = plugin
        .get("endpoint")
        .and_then(toml::Value::as_str)
        .context("[plugin] endpoint missing")?;
    let addr = endpoint_addr(endpoint)?;
    let secret = plugin
        .get("secret")
        .and_then(toml::Value::as_str)
        .map(str::to_string);
    Ok((addr, secret))
}

fn home_dir() -> String {
    std::env::var("HOME").unwrap_or_else(|_| ".".to_string())
}

fn bot_toml_path() -> String {
    env_or(
        "JERITH_BOT_TOML",
        &format!("{}/.jerith-mesh/bot.toml", home_dir()),
    )
}

fn bot_unit() -> String {
    env_or("JERITH_BOT_UNIT", "jerith-bot")
}

fn house_repo() -> String {
    env_or("JERITH_HOUSE_REPO", ".")
}

/// Gateway `/status`, 3 s bound. Errors → `None` (the card field).
async fn fetch_gateway() -> Option<Value> {
    let url = env_or("JERITH_GATEWAY", "http://127.0.0.1:8900");
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .ok()?;
    let body: Value = client
        .get(format!("{url}/status"))
        .send()
        .await
        .ok()?
        .json()
        .await
        .ok()?;
    Some(body)
}

/// Bounded spawn of one host tool; non-zero exit or a runaway stdout
/// collapses to `None` (the card asks no questions a dead tool can't
/// answer in 64 KiB).
async fn capture(cmd: &mut tokio::process::Command) -> Option<String> {
    cmd.stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null());
    let out = cmd.output().await.ok()?;
    if !out.status.success() {
        return None;
    }
    if out.stdout.len() > 65_536 {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).to_string())
}

async fn fetch_systemctl() -> Option<String> {
    let unit = bot_unit();
    let mut cmd = tokio::process::Command::new("systemctl");
    cmd.arg("--user").arg("show").arg(&unit).args([
        "--property=ActiveState",
        "--property=ActiveEnterTimestamp",
        "--property=NRestarts",
    ]);
    capture(&mut cmd).await
}

/// Recent WARN/ERROR journal lines for the bot unit. Bounded from the
/// spawn: `-n 8` newest-first, `-o cat` strips metadata, `-p warning`
/// keeps the same level gate the briefing's ring uses.
async fn fetch_journal(since_minutes: u64) -> Option<Vec<String>> {
    let unit = bot_unit();
    let mut cmd = tokio::process::Command::new("journalctl");
    cmd.arg("--user")
        .arg("-u")
        .arg(&unit)
        .arg("-p")
        .arg("warning")
        .arg("--no-pager")
        .arg("-o")
        .arg("cat")
        .arg("-n")
        .arg("8")
        .arg(format!("--since=-{since_minutes}min"));
    let raw = capture(&mut cmd).await?;
    let lines: Vec<String> = raw
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(String::from)
        .collect();
    Some(lines)
}

async fn fetch_commit() -> Option<String> {
    let repo = house_repo();
    let mut cmd = tokio::process::Command::new("git");
    cmd.arg("-C")
        .arg(&repo)
        .arg("rev-parse")
        .arg("--short")
        .arg("HEAD");
    let raw = capture(&mut cmd).await?;
    let trimmed = raw.trim().to_string();
    (!trimmed.is_empty()).then_some(trimmed)
}

/// Read bot.toml raw; the VALUES never leave the process — lib composes
/// from table key names only. Absent/unreadable file → None (honest).
fn fetch_bot_toml() -> Option<String> {
    let raw = std::fs::read_to_string(bot_toml_path()).ok()?;
    (raw.len() <= 65_536).then_some(raw)
}

/// Gather every source concurrently, then hand the completed reads to
/// the pure composer. Slow tools are degraded, never waited past: each
/// spawn rides its own task and the longest possible wait is the one
/// `capture` round.
async fn status_card() -> Result<Value, String> {
    let gw = tokio::spawn(fetch_gateway());
    let sys = tokio::spawn(fetch_systemctl());
    let commit = tokio::spawn(fetch_commit());
    let bt = tokio::spawn(async { fetch_bot_toml() });
    let journal = tokio::spawn(fetch_journal(120));

    let gateway = gw.await.map_err(|e| format!("gateway join: {e}"))?;
    let systemctl = sys.await.map_err(|e| format!("systemctl join: {e}"))?;
    let commit = commit.await.map_err(|e| format!("git join: {e}"))?;
    let bot_toml = bt.await.map_err(|e| format!("bot.toml join: {e}"))?;
    let journal = journal.await.map_err(|e| format!("journalctl join: {e}"))?;

    Ok(jerith_house_plugin::compose(&Sources {
        gateway: gateway.as_ref(),
        systemctl: systemctl.as_deref(),
        bot_toml: bot_toml.as_deref(),
        journal: journal.as_deref(),
        commit: commit.as_deref(),
    }))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let manifest = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "plugin.toml".to_string());
    let (addr, secret) = read_manifest(&manifest)?;
    let gateway_show = env_or("JERITH_GATEWAY", "http://127.0.0.1:8900");
    tracing::info!(
        manifest,
        %addr,
        secret = secret.is_some(),
        gateway = %gateway_show,
        unit = %bot_unit(),
        "house plugin starting"
    );

    SkillServer::new("house", secret)
        .handler(
            "house.status",
            skill(|_args: Value| async move { status_card().await }),
        )
        .handler(
            "house.journal",
            skill(|args: Value| async move {
                // 5–1440 minutes: an hour-ish default, a day at most.
                let since = args["since_minutes"].as_u64().unwrap_or(120).clamp(5, 1440);
                let unit = bot_unit();
                match fetch_journal(since).await {
                    Some(lines) => Ok(json!({
                        "unit": unit,
                        "since_minutes": since,
                        "lines": lines,
                    })),
                    None => Err(format!(
                        "journalctl --user unreachable (unit {unit}) — the journal is not readable from here"
                    )),
                }
            }),
        )
        .serve(addr)
        .await
}
