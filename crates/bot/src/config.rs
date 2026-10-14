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
//! `chain_url`, `system_prompt` — env always wins over the file. When no
//! `system_prompt` is configured anywhere, the bot auto-loads
//! `~/.jerith-mesh/persona.md` (the migrated agent persona, see SOUL.md's
//! lineage) if that file exists.

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
/// Inline persona/system prompt env var (wins over bot.toml + persona.md).
pub const PERSONA_ENV: &str = "JERITH_PERSONA";
/// Context budget for auto-loaded persona files (chars, not bytes).
/// The composed live persona (~10.4k) plus the known-users export (~5.3k)
/// both ride under this, and the runtime (qwen3:8b via LiteLLM/Ollama)
/// exposes a 262k context — 32k fences pathological files without ever
/// being the binding constraint.
pub const PERSONA_MAX_CHARS: usize = 32000;

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
    /// Operator-owned persona/system prompt — sent as the first message of
    /// every chat (this is where SOUL/IDENTITY/USER.md live now). Never
    /// passed through the Guard: it is static config, and its PII redaction
    /// could mangle the owner's own security rules quoted inside it.
    pub system_prompt: Option<String>,
    /// Optional `[postgres]` table — live jerith_people lookup per chat turn
    /// (OpenClaw-stack defaults; password resolved from ~/.pgpass). Degrades
    /// silently to no context when the DB is unreachable.
    pub postgres: Option<crate::people::PgConfig>,
    /// Web access via the gateway's web skills (OpenClaw migration): when
    /// enabled, chat requests declare `web_search` + `web_fetch` tools and
    /// executed tool calls are fed back into the conversation (OpenAI
    /// tool-calling protocol). Off → the bot never touches the web.
    pub web: WebToolsConfig,
    /// Plugin / skill gating (`[plugins]` in bot.toml, OpenClaw-style
    /// modular plugin architecture; see plugins.rs). The `enabled` list
    /// names the pre-packaged plugins that are switched on; everything the
    /// model can summon derives from it (web loop tools, memory.*, lore).
    pub plugins: PluginsConfig,
    /// The bot owner's Discord id (default-deny gate for system-level
    /// commands; see owner.rs). Resolved from `$JERITH_BOT_OWNER_ID`, then
    /// bot.toml `[owner] discord_id`, then the OpenClaw config's
    /// `commands.ownerAllowFrom` (read-only borrow, like the token).
    pub owner_discord_id: Option<u64>,
    /// Discord thread-archiver (operator-configured background ingestion;
    /// off unless `[archiver] enabled = true` and channels are listed).
    pub archiver: ArchiverConfig,
    /// Context squeeze (OpenClaw context-squeeze/juice-compact port):
    /// per-user history compaction before chats.
    pub squeeze: SqueezeConfig,
    /// Ops monitor (OpenClaw ops-monitor family port): periodic checks
    /// with a single owner DM when something fails. Reports only.
    pub monitor: MonitorConfig,
    /// Chat request timeout in seconds (`[chat] timeout_secs` in bot.toml,
    /// clamped 60–600). Default 240: the gateway bounds a non-stream local
    /// runtime at 180 s (header wait, then mesh fallback), so 240 s lets a
    /// local answer land with headroom, and also covers a slower mesh-fallback
    /// leg. The old hard 90 s killed legitimately deep qwen3 thinking answers.
    pub chat_timeout_secs: u64,
    /// Stream chat completions through the gateway's SSE passthrough and
    /// live-type the answer into Discord (live edits while tokens arrive).
    /// Default on; `[chat] stream = false` restores the old buffered
    /// single-shot reply. Degenerates to a buffered answer automatically
    /// whenever the gateway answers non-SSE (JSON error/fallback paths).
    pub chat_stream: bool,
}

/// Chat timeout default ([chat] timeout_secs), years after the 90 s
/// constant died on a slow thinking answer.
const DEFAULT_CHAT_TIMEOUT_SECS: u64 = 240;

impl BotConfig {
    /// Load from env, falling back to `~/.jerith-mesh/bot.toml`.
    pub fn load() -> Result<Self> {
        let path = token_file_path();
        let file = read_file_config(&path).unwrap_or_default();

        // Owner id resolved first — everything below partially-moves `file`.
        let owner_discord_id = Self::resolve_owner(&file);

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

        let system_prompt = std::env::var(PERSONA_ENV)
            .ok()
            .map(|s| s.trim().to_string())
            .or(file.system_prompt)
            .filter(|s| !s.trim().is_empty());

        Ok(Self {
            discord_token: token,
            gateway_url,
            chain_url,
            announce_channel_id,
            guard: file.guard.unwrap_or_default(),
            system_prompt,
            postgres: file.postgres,
            web: file.web.unwrap_or_default(),
            plugins: {
                let cfg = file.plugins.unwrap_or_default();
                crate::plugins::SkillRegistry::check_config_enabled(&cfg.enabled)
                    .map_err(|e| anyhow::anyhow!("{} (in {})", e, path.display()))?;
                cfg
            },
            owner_discord_id,
            archiver: file.archiver.unwrap_or_default(),
            squeeze: file.squeeze.unwrap_or_default(),
            monitor: file.monitor.unwrap_or_default(),
            chat_timeout_secs: file
                .chat
                .as_ref()
                .and_then(|c| c.timeout_secs)
                .unwrap_or(DEFAULT_CHAT_TIMEOUT_SECS)
                .clamp(60, 600),
            chat_stream: file.chat.as_ref().and_then(|c| c.stream).unwrap_or(true),
        })
    }

    /// Owner id resolution (env → file → OpenClaw borrow). A missing owner
    /// is allowed but leaves the default-deny gate closed for everyone:
    /// system-level commands stay refused until an owner is configured.
    fn resolve_owner(file: &FileConfig) -> Option<u64> {
        let from_env = std::env::var("JERITH_BOT_OWNER_ID")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok());
        let resolved = from_env
            .or(file.owner.as_ref().and_then(|o| o.discord_id))
            .or_else(|| {
                crate::owner::openclaw_config_file()
                    .and_then(|p| crate::owner::borrow_owner_from_openclaw(&p))
            });
        match resolved {
            Some(id) => {
                if from_env.is_none() {
                    tracing::info!(
                        owner_discord_id = id,
                        "bot owner id resolved (system-level commands gated)"
                    );
                }
                Some(id)
            }
            None => {
                tracing::warn!(
                    "no bot owner configured — system-level commands are default-deny for everyone. \
                     Set $JERITH_BOT_OWNER_ID or [owner] discord_id in bot.toml to open the gate to yourself"
                );
                None
            }
        }
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
            system_prompt: None,
            postgres: None,
            web: WebToolsConfig::default(),
            plugins: PluginsConfig::default(),
            owner_discord_id: None,
            archiver: ArchiverConfig::default(),
            squeeze: SqueezeConfig::default(),
            monitor: MonitorConfig::default(),
            chat_timeout_secs: DEFAULT_CHAT_TIMEOUT_SECS,
            chat_stream: true,
        }
    }
}

/// Context-squeeze settings (`[squeeze]` in bot.toml) — OpenClaw
/// context-squeeze port. On by default with a gentle budget; `enabled =
/// false` restores the plain 8-turn ring semantics verbatim.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct SqueezeConfig {
    /// Run compaction before chats.
    #[serde(default = "default_squeeze_on")]
    pub enabled: bool,
    /// Char budget for a user's history before compaction kicks in
    /// (roleplay/story threads blow past 8 short turns fast).
    #[serde(default = "default_squeeze_budget")]
    pub budget_chars: usize,
    /// How many oldest messages get collapsed into the mesh summary
    /// (stage 3), when even the passive squeeze isn't enough.
    #[serde(default = "default_squeeze_summarize_count")]
    pub summarize_count: usize,
    /// Minimum seconds between stage-3 summaries per user (a mesh round
    /// trip; pacing keeps it rare).
    #[serde(default = "default_squeeze_interval")]
    pub min_interval_secs: u64,
}

fn default_squeeze_on() -> bool {
    true
}
fn default_squeeze_budget() -> usize {
    4000
}
fn default_squeeze_summarize_count() -> usize {
    6
}
fn default_squeeze_interval() -> u64 {
    300
}

impl Default for SqueezeConfig {
    fn default() -> Self {
        Self {
            enabled: default_squeeze_on(),
            budget_chars: default_squeeze_budget(),
            summarize_count: default_squeeze_summarize_count(),
            min_interval_secs: default_squeeze_interval(),
        }
    }
}

/// Discord thread-archiver configuration (`[archiver]` in bot.toml) — port
/// of the OpenClaw thread-archiver skill's options. Disabled unless
/// explicitly enabled AND at least one channel id is listed.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ArchiverConfig {
    #[serde(default)]
    pub enabled: bool,
    /// Thread/channel ids (decimal) to archive — global/thread ids are
    /// ordinary Discord channels to the REST API.
    #[serde(default)]
    pub channel_ids: Vec<u64>,
    /// Sweep cadence in minutes (OpenClaw cron equivalent).
    #[serde(default = "default_archiver_interval")]
    pub interval_minutes: u64,
    /// Page size per channel fetch (Discord max 100).
    #[serde(default = "default_archiver_max_messages")]
    pub max_messages_per_sweep: u32,
    /// Note lines kept per channel per sweep.
    #[serde(default = "default_archiver_max_notes")]
    pub max_notes_per_sweep: usize,
    /// Optional upstream-parity outbound copy of the summary markdown.
    #[serde(default)]
    pub notes_webhook: Option<String>,
    /// Global decision/note marker override (case-insensitive substring
    /// match per line). Empty → the built-in defaults (see
    /// archiver::DEFAULT_MARKERS).
    #[serde(default)]
    pub markers: Vec<String>,
    /// Per-guild marker overrides, read from bot.toml as
    /// `[archiver.guild_markers.<guild_id>]` + `markers = [...]` — wins
    /// over `markers` for every channel in that guild, so different
    /// servers (language, topic) match their own phrasing.
    #[serde(default)]
    pub guild_markers: std::collections::HashMap<u64, GuildMarkerSet>,
}

/// The value of a `[archiver.guild_markers.<guild_id>]` table.
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct GuildMarkerSet {
    #[serde(default)]
    pub markers: Vec<String>,
}

fn default_archiver_interval() -> u64 {
    360
}
fn default_archiver_max_messages() -> u32 {
    50
}
fn default_archiver_max_notes() -> usize {
    40
}

impl Default for ArchiverConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            channel_ids: Vec::new(),
            interval_minutes: default_archiver_interval(),
            max_messages_per_sweep: default_archiver_max_messages(),
            max_notes_per_sweep: default_archiver_max_notes(),
            notes_webhook: None,
            markers: Vec::new(),
            guild_markers: std::collections::HashMap::new(),
        }
    }
}

/// Ops-monitor settings (`[monitor]` in bot.toml) — OpenClaw
/// ops-monitor family port. On by default (disk/temp/log checks are
/// free); the backup-freshness check only runs for listed dirs.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct MonitorConfig {
    /// Run the periodic checks + owner-DM reports.
    #[serde(default = "default_monitor_on")]
    pub enabled: bool,
    /// Sweep cadence in minutes.
    #[serde(default = "default_monitor_interval")]
    pub interval_minutes: u64,
    /// Data dirs whose newest file must be fresh (≤26h), else the
    /// backups check fails into the owner DM.
    #[serde(default)]
    pub backup_dirs: Vec<String>,
}

fn default_monitor_on() -> bool {
    true
}
fn default_monitor_interval() -> u64 {
    30
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            enabled: default_monitor_on(),
            interval_minutes: default_monitor_interval(),
            backup_dirs: Vec::new(),
        }
    }
}

/// Tool-loop toggle for the gateway's web skills. Defaults on — the bot
/// can only use the web when the node itself serves it and the runtime
/// model cooperates; unsupported runtimes degrade to toolless chats (see
/// api.rs graceful degradation).
#[derive(Debug, Clone, Copy, serde::Deserialize)]
pub struct WebToolsConfig {
    /// Declare web_search/web_fetch in chat requests.
    #[serde(default = "default_web_tools_on")]
    pub enabled: bool,
}

fn default_web_tools_on() -> bool {
    true
}

impl Default for WebToolsConfig {
    fn default() -> Self {
        Self {
            enabled: default_web_tools_on(),
        }
    }
}

/// `~/.jerith-mesh/data` — shared runtime data (ledger files,
/// thread-archive markdown). Home-relative like the other path helpers.
pub fn data_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".jerith-mesh").join("data")
}

/// `~/.jerith-mesh/bot.toml`
pub fn token_file_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".jerith-mesh").join("bot.toml")
}

/// `~/.jerith-mesh/persona.md` — the auto-loaded persona when no explicit
/// system_prompt is configured. Compose it from the migrated agent docs
/// (openclaw-migrated/workspace/{SOUL,IDENTITY,USER,USER_MODEL}.md).
pub fn persona_file_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".jerith-mesh").join("persona.md")
}

/// `~/.jerith-mesh/known-users.md` — live export of the `jerith_people`
/// PostgreSQL database (site cron, every 5 min). Appended to the persona
/// so the bot reads its people registry — the individual-user recognition
/// source the OpenClaw gateway used to provide.
pub fn known_users_file_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home)
        .join(".jerith-mesh")
        .join("known-users.md")
}

/// Cap a persona system prompt so an oversized persona file can never blow
/// the runtime's context or the gateway's metering. Keeps the head — the
/// identity canon is written first — and appends an explicit marker.
pub fn cap_system_prompt(text: &str, max_chars: usize) -> String {
    let t = text.trim();
    let n = t.chars().count();
    if n <= max_chars {
        return t.to_string();
    }
    let keep = max_chars.saturating_sub(60);
    let head: String = t.chars().take(keep).collect();
    format!(
        "{head}\n\n…(persona truncated at {max_chars} chars — persona.md is too long for the runtime budget)"
    )
}

#[derive(Debug, Default, serde::Deserialize)]
struct FileConfig {
    discord_token: Option<String>,
    gateway_url: Option<String>,
    chain_url: Option<String>,
    /// Decimal channel id in TOML; env overrides.
    announce_channel_id: Option<u64>,
    /// Inline persona/system prompt (TOML string); env `JERITH_PERSONA`
    /// wins; persona.md file is the last fallback (handled in main.rs).
    system_prompt: Option<String>,
    /// Optional `[postgres]` table — live jerith_people lookup per chat turn.
    postgres: Option<crate::people::PgConfig>,
    /// Optional `[guard]` section — same shape as the node's config.toml.
    guard: Option<jerith_mesh_core::guard::GuardConfig>,
    /// Optional `[web]` section — tool-loop toggle for the gateway's
    /// web skills (Brave search + fetch).
    web: Option<WebToolsConfig>,
    /// Optional `[owner]` table — the bot owner's Discord id, gating
    /// system-level commands (see owner.rs; borrowed from the OpenClaw
    /// config when missing, same as the token).
    owner: Option<OwnerTable>,
    /// Optional `[archiver]` table — Discord thread-archiver config
    /// (OpenClaw skill port, see archiver.rs).
    archiver: Option<ArchiverConfig>,
    /// Optional `[squeeze]` table — context-squeeze compaction settings
    /// (OpenClaw context-squeeze port, see squeeze.rs).
    squeeze: Option<SqueezeConfig>,
    /// Optional `[monitor]` table — ops-monitor settings (checks +
    /// owner-DM reports; see monitor.rs).
    monitor: Option<MonitorConfig>,
    /// Optional `[chat]` table — chat request timeout knob.
    chat: Option<ChatTable>,
    /// Optional `[plugins]` table — plugin/skill gating (see plugins.rs).
    plugins: Option<PluginsConfig>,
}

/// `[plugins] enabled = [...]` — names the pre-packaged plugins that
/// stay available. Defaults to every plugin on; unknown ids fail the
/// load (fail-loud config, like the rest of the bot).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PluginsConfig {
    #[serde(default = "default_plugins_enabled")]
    pub enabled: Vec<String>,
}

impl Default for PluginsConfig {
    /// All pre-packaged plugins on (empty would mean the bot forgets its
    /// own web tools — an uninstallable default).
    fn default() -> Self {
        Self {
            enabled: default_plugins_enabled(),
        }
    }
}

fn default_plugins_enabled() -> Vec<String> {
    crate::plugins::PREPACKAGED_PLUGINS
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// `[chat] timeout_secs = N` in bot.toml (60–600 clamp).
#[derive(Debug, Default, serde::Deserialize)]
struct ChatTable {
    timeout_secs: Option<u64>,
    /// Stream + live-type (see BotConfig::chat_stream). Default true.
    stream: Option<bool>,
}

/// `[owner] discord_id = <decimal>` in bot.toml.
#[derive(Debug, Default, serde::Deserialize)]
struct OwnerTable {
    discord_id: Option<u64>,
}

fn read_file_config(path: &PathBuf) -> Option<FileConfig> {
    let text = std::fs::read_to_string(path).ok()?;
    toml::from_str(&text).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persona_cap_keeps_head_and_marks_truncation() {
        let short = cap_system_prompt("  You are Jerith.\n ", 6000);
        assert_eq!(short, "You are Jerith.");

        let long: String = "a".repeat(5000) + "\n" + &"b".repeat(10_000);
        let capped = cap_system_prompt(&long, 6000);
        assert!(capped.contains("…(persona truncated at 6000 chars"));
        // 6000-cap minus the marker ⇒ head content preserved (identity first).
        assert!(capped.starts_with(&"a".repeat(50)));
        assert!(capped.chars().count() <= 6040);
    }

    #[test]
    fn from_parts_trims_trailing_slash() {
        let cfg = BotConfig::from_parts("http://127.0.0.1:8900/", "http://127.0.0.1:8322/");
        assert_eq!(cfg.gateway_url, "http://127.0.0.1:8900");
        assert_eq!(cfg.chain_url, "http://127.0.0.1:8322");
    }

    #[test]
    fn archiver_parses_per_guild_markers() {
        let fc: FileConfig = toml::from_str(
            "[archiver]\n\
             enabled = true\n\
             channel_ids = [7, 9]\n\
             markers = [\"decision:\"]\n\
             [archiver.guild_markers.4200]\n\
             markers = [\"bestemd\", \"lore-claim\"]\n",
        )
        .unwrap();
        let a = fc.archiver.expect("archiver section parsed");
        assert!(a.enabled);
        assert_eq!(a.channel_ids, vec![7, 9]);
        assert_eq!(a.markers, vec!["decision:".to_string()]);
        assert_eq!(
            a.guild_markers.get(&4200).map(|s| s.markers.clone()),
            Some(vec!["bestemd".to_string(), "lore-claim".to_string()])
        );
    }

    #[test]
    fn file_parse_ignores_unknown_keys() {
        let fc: FileConfig = toml::from_str("discord_token = \"x\"\nfuture_key = 1\n").unwrap();
        assert_eq!(fc.discord_token.unwrap(), "x");
    }

    #[test]
    fn chat_timeout_parses_and_clamps() {
        let in_range = toml::from_str::<FileConfig>("[chat]\ntimeout_secs = 300\n").unwrap();
        assert_eq!(in_range.chat.as_ref().unwrap().timeout_secs, Some(300));
        let clamped_hi = toml::from_str::<FileConfig>("[chat]\ntimeout_secs = 5000\n").unwrap();
        assert_eq!(clamped_hi.chat.as_ref().unwrap().timeout_secs, Some(5000)); // parse keeps it; clamp is at load()
        let none = toml::from_str::<FileConfig>("discord_token = \"x\"\n").unwrap();
        assert!(none.chat.is_none());
    }
}
