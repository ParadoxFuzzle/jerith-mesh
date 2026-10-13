//! Bot configuration: env-first with optional TOML file.
//!
//! Env vars:
//! - `JERITH_DISCORD_TOKEN` — Discord bot token (required to run).
//! - `JERITH_GATEWAY_URL` — mesh node gateway to talk to
//!   (default `http://127.0.0.1:8900`).
//! - `JERITH_CHAIN_URL` — local JerithChain node RPC for `/mine`
//!   (default `http://127.0.0.1:8322`).
//!
//! Optional TOML (`~/.jerith-mesh/bot.toml`): `discord_token`, `gateway_url`,
//! `chain_url` — env always wins over the file.

use std::path::PathBuf;

use anyhow::{Context as _, Result};

/// Name of the token env var the bot reads first.
pub const DISCORD_TOKEN_ENV: &str = "JERITH_DISCORD_TOKEN";
/// Gateway (mesh node HTTP API) base URL env var.
pub const GATEWAY_URL_ENV: &str = "JERITH_GATEWAY_URL";
/// JerithChain node RPC base URL env var (used by /mine).
pub const CHAIN_URL_ENV: &str = "JERITH_CHAIN_URL";
/// Discord channel id (decimal string) where /mine wins are announced.
pub const ANNOUNCE_CHANNEL_ENV: &str = "JERITH_ANNOUNCE_CHANNEL_ID";

/// Default gateway URL: the local node's gateway port.
pub const DEFAULT_GATEWAY_URL: &str = "http://127.0.0.1:8900";
/// Default JerithChain node RPC URL.
pub const DEFAULT_CHAIN_URL: &str = "http://127.0.0.1:8322";

#[derive(Debug, Clone)]
pub struct BotConfig {
    pub discord_token: String,
    pub gateway_url: String,
    pub chain_url: String,
    /// Optional Discord channel for /mine announcements (Set via
    /// `JERITH_ANNOUNCE_CHANNEL_ID` or bot.toml `announce_channel_id`).
    pub announce_channel_id: Option<u64>,
    /// Jerith Guard configuration for outbound screening of /ask prompts
    /// *before* the gateway/provider call (defense in depth: the gateway
    /// screens too). Optional `[guard]` table in bot.toml.
    pub guard: jerith_mesh_core::guard::GuardConfig,
}

impl BotConfig {
    /// Load from env, falling back to `~/.jerith-mesh/bot.toml`.
    pub fn load() -> Result<Self> {
        let path = token_file_path();
        let file = read_file_config(&path).unwrap_or_default();

        let token = std::env::var(DISCORD_TOKEN_ENV)
            .ok()
            .or(file.discord_token)
            .context(format!(
                "no Discord token — set {DISCORD_TOKEN_ENV} or put `discord_token = \"...\"` in {}",
                path.display()
            ))?;

        let gateway_url = std::env::var(GATEWAY_URL_ENV)
            .ok()
            .or(file.gateway_url)
            .unwrap_or_else(|| DEFAULT_GATEWAY_URL.to_string())
            .trim_end_matches('/')
            .to_string();

        let chain_url = std::env::var(CHAIN_URL_ENV)
            .ok()
            .or(file.chain_url)
            .unwrap_or_else(|| DEFAULT_CHAIN_URL.to_string())
            .trim_end_matches('/')
            .to_string();

        let announce_channel_id = std::env::var(ANNOUNCE_CHANNEL_ENV)
            .ok()
            .map(|v| v.trim().to_string())
            .or(file.announce_channel_id.map(|v| v.to_string()))
            .and_then(|v| v.trim().parse::<u64>().ok());

        Ok(Self {
            discord_token: token,
            gateway_url,
            chain_url,
            announce_channel_id,
            guard: file.guard.unwrap_or_default(),
        })
    }

    /// Build without requiring a token (tests / dry checks).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn from_parts(gateway_url: &str, chain_url: &str) -> Self {
        Self {
            discord_token: String::new(),
            gateway_url: gateway_url.trim_end_matches('/').to_string(),
            chain_url: chain_url.trim_end_matches('/').to_string(),
            announce_channel_id: None,
            guard: jerith_mesh_core::guard::GuardConfig::default(),
        }
    }
}

/// `~/.jerith-mesh/bot.toml`
pub fn token_file_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".jerith-mesh").join("bot.toml")
}

#[derive(Debug, Default, serde::Deserialize)]
struct FileConfig {
    discord_token: Option<String>,
    gateway_url: Option<String>,
    chain_url: Option<String>,
    /// Decimal channel id in TOML; env overrides.
    announce_channel_id: Option<u64>,
    /// Optional `[guard]` section — same shape as the node's config.toml.
    guard: Option<jerith_mesh_core::guard::GuardConfig>,
}

fn read_file_config(path: &PathBuf) -> Option<FileConfig> {
    let text = std::fs::read_to_string(path).ok()?;
    toml::from_str(&text).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_parts_trims_trailing_slash() {
        let cfg = BotConfig::from_parts("http://127.0.0.1:8900/", "http://127.0.0.1:8322/");
        assert_eq!(cfg.gateway_url, "http://127.0.0.1:8900");
        assert_eq!(cfg.chain_url, "http://127.0.0.1:8322");
    }

    #[test]
    fn file_parse_ignores_unknown_keys() {
        let fc: FileConfig = toml::from_str("discord_token = \"x\"\nfuture_key = 1\n").unwrap();
        assert_eq!(fc.discord_token.unwrap(), "x");
    }
}
