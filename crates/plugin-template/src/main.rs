//! `jerith-plugin-example` — the template crate's runnable example,
//! and the fastest way to prove a live lane: run this binary, drop the
//! copyable `plugin.toml` (same directory) into
//! `~/.jerith-mesh/plugins.d/example/plugin.toml`, add
//! `external:example` to `[plugins] enabled` in bot.toml, restart the
//! bot, and ask the model to use the `example.*` skills in chat.
//!
//! The two demo skills are deliberately trivial:
//! - `example.echo` — args in, echoed object out (the "it's alive" test).
//! - `example.roll` — one dN die, args in (`sides`), honest ceiling
//!   (≤ 1000), seed from wall-clock nanos (proves a per-call compute;
//!   a real skill would hit its own data source here).
//!
//! Authoring story: copy `main.rs`, replace the demo handlers with real
//! logic, keep `plugin.toml` in sync (id + `[[skill]]` rows), ship. The
//! SDK handles the wire: signature verification, freshness, bounded
//! responses, typed errors.
//!
//! ONE source of truth for the endpoint line: the manifest. The binary
//! READS `plugin.toml` (`[plugin] endpoint` + `secret`) and binds that
//! exact address — manifest and service cannot drift silently.

use anyhow::{Context as _, Result};
use jerith_plugin_template::{SkillServer, endpoint_addr, skill};
use serde_json::{Value, json};
use std::time::{SystemTime, UNIX_EPOCH};

/// nanos since epoch — the demo die's seed (a real service would use
/// whatever entropy it trusts).
fn seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as u64
}

async fn echo(args: Value) -> Result<Value, String> {
    let text = args["text"].as_str().unwrap_or_default().trim();
    if text.is_empty() {
        return Err("args.text must be a non-empty string".to_string());
    }
    Ok(json!({ "echo": text }))
}

async fn roll(args: Value) -> Result<Value, String> {
    let sides = args["sides"].as_u64().unwrap_or(6);
    if sides == 0 || sides > 1000 {
        return Err("args.sides must be 1..=1000".to_string());
    }
    // xorshift on the clock seed: trivially honest, no rand dep needed.
    let mut s = seed() | 1;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    Ok(json!({ "sides": sides, "roll": (s % sides) + 1 }))
}

/// `[plugin] endpoint` + `[plugin] secret` (None = unsigned dev mode)
/// straight out of the manifest — one file rules both halves.
fn read_manifest(path: &str) -> Result<(std::net::SocketAddr, Option<String>)> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("reading manifest {path}"))?;
    let file: toml::Value = raw
        .parse()
        .with_context(|| format!("{path}: not valid TOML"))?;
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
    tracing::info!(manifest, %addr, secret = secret.is_some(), "example plugin starting");

    SkillServer::new("example", secret)
        .handler("example.echo", skill(echo))
        .handler("example.roll", skill(roll))
        .serve(addr)
        .await
}
