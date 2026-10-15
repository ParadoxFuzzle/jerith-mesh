//! Discord thread archiver — port of the OpenClaw `thread-archiver`
//! plugin-skill, with the extraction half the OpenClaw agent harness used
//! to provide now implemented deterministically.
//!
//! What it does (operator-configured, not command-triggered):
//!   1. On an interval, sweep each configured channel/thread id over
//!      Discord REST (`/channels/{id}/messages?after=<last>`), so only
//!      new messages since the last sweep are read (snowflake ledger).
//!   2. Extract decision/technical-note lines (case-insensitive markers:
//!      "decision:", "decided to", "agreed", "action item", …), tagged
//!      `author: line`, other bots' lines skipped.
//!   3. Publish the summary three ways: the memory embed via the
//!      gateway's `POST /v1/skills/memory/embed` (the jerith-native
//!      addition: `/recall` becomes thread-aware, which OpenClaw could
//!      never offer — its archive only lived as a webhook post); the
//!      optional upstream-parity POST to `notes_webhook` (204, same body
//!      shape as the Python skill); and a local markdown archive file in
//!      the upstream `_summarize` format.
//!   4. Only after the embed succeeds is the ledger advanced — a failed
//!      gateway call is retried on the next sweep and never loses notes.
//!
//! Outbound scope (deliberately narrow): the node's
//! own gateway and, when configured, the operator's webhook.
//! It performs no command, no system change, and writes only under
//! `~/.jerith-mesh/data/thread-archiver/` — so it does *not* go through
//! the owner-DM gate (owner.rs is for command-triggered system-level work,
//! while this is operator-configured background ingestion).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::{ArchiverConfig, BotConfig};

/// Discord REST base (v10 — the bare minimum these bot uses).
const DISCORD_API: &str = "https://discord.com/api/v10";

/// Default decision/note markers, mirroring the lines the OpenClaw agent
/// used to surface as "Decisions & notes" in its archive digests.
pub const DEFAULT_MARKERS: [&str; 9] = [
    "decision",
    "decided",
    "agree to",
    "agreed",
    "resolution",
    "conclusion",
    "action item",
    "todo",
    "next step",
];

/// Built-in decision/note markers as owned strings — the fallback when
/// neither the guild nor the global `[archiver] markers` configure a list.
pub fn default_markers() -> Vec<String> {
    DEFAULT_MARKERS.iter().map(|m| m.to_string()).collect()
}

/// Marker list in force for one channel: the channel's guild override
/// (`[archiver.guild_markers.<guild_id>]`) wins, then the global
/// `[archiver] markers`, then the built-ins. Blank/whitespace entries are
/// dropped in every tier.
pub fn effective_markers(arch: &ArchiverConfig, guild_id: Option<u64>) -> Vec<String> {
    let clean = |list: &[String]| -> Vec<String> {
        list.iter()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    };
    if let Some(gid) = guild_id
        && let Some(set) = arch.guild_markers.get(&gid) {
            let over = clean(&set.markers);
            if !over.is_empty() {
                return over;
            }
        }
    let global = clean(&arch.markers);
    if !global.is_empty() {
        return global;
    }
    default_markers()
}

/// Markdown header the OpenClaw `_summarize` produced; kept byte-for-byte
/// so an operator tracking the OpenClaw wiki sees the same shape.
pub fn render_summary(source_ref: &str, archived_at: &str, notes: &[&str]) -> String {
    let mut out = format!(
        "## Thread Archive - {source_ref}\n*Archived {archived_at}*\n\n### Decisions & notes\n\n"
    );
    if notes.is_empty() {
        out.push_str("  (no notes extracted)\n");
    } else {
        for line in notes {
            out.push_str("  ");
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// Extract note-worthy lines from a messages payload (Discord REST:
/// newest-first array of message objects). Every surviving line is tagged
/// `author: text`. Other bots' messages are skipped (they are downstream
/// representations, not primary notes) — EXCEPT the archiver's own bot
/// account, whose posts (announcements, test digests) are primary notes
/// the mesh owner wrote through the bot. Non-matching prose dropped, and
/// the batch capped so one sweep can never balloon the memory store.
pub fn extract_notes(
    messages: &[Value],
    markers: &[&str],
    max_notes: usize,
    self_bot_id: Option<u64>,
) -> Vec<String> {
    let needle_filter = |content: &str| -> bool {
        markers.iter().any(|m| {
            content
                .to_ascii_lowercase()
                .contains(&m.to_ascii_lowercase())
        })
    };
    let mut out: Vec<String> = Vec::new();
    // Discord gives newest-first; walk oldest→newest so a summary reads
    // chronologically like the thread did.
    for msg in messages.iter().rev() {
        if out.len() >= max_notes {
            break;
        }
        let author = msg
            .pointer("/author/global_name")
            .or_else(|| msg.pointer("/author/username"))
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let is_bot = msg
            .pointer("/author/bot")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let author_id = msg
            .pointer("/author/id")
            .and_then(Value::as_str)
            .and_then(|s| s.parse::<u64>().ok());
        let is_self = self_bot_id
            .map(|self_id| author_id == Some(self_id))
            .unwrap_or(false);
        if is_bot && !is_self {
            continue;
        }
        let content = msg
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default();
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || !needle_filter(line) {
                continue;
            }
            out.push(format!("{author}: {line}"));
            if out.len() >= max_notes {
                break;
            }
        }
    }
    out
}

/// Newest message snowflake in the batch (`None` on an empty batch) —
/// the value the ledger advances to after a successful publish.
pub fn newest_message_id(messages: &[Value]) -> Option<u64> {
    messages
        .iter()
        .filter_map(|m| m.get("id").and_then(Value::as_str))
        .filter_map(|s| s.parse::<u64>().ok())
        .max()
}

/// UTC timestamp in the OpenClaw format (`%Y-%m-%d %H:%M:%S`), no chrono
/// in the workspace: Howard Hinnant's civil-from-days, unit-tested below.
pub fn utc_timestamp_string(unix_ms: u64) -> String {
    let secs = unix_ms / 1000;
    let days = secs / 86_400;
    let secs_of_day = secs % 86_400;
    //   civil algorithms (Hinnant 2017).
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{y:04}-{m:02}-{d:02} {:02}:{:02}:{:02}",
        secs_of_day / 3600,
        (secs_of_day % 3600) / 60,
        secs_of_day % 60
    )
}

/// Ledger — the OpenClaw skill's `ledger.json` shape minus URL entries
/// (channels are configured in bot.toml here), plus the pointer the
/// sweep resumes from (`last_message_id`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Ledger {
    pub version: u32,
    pub entries: HashMap<String, LedgerEntry>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LedgerEntry {
    #[serde(default)]
    pub last_message_id: Option<u64>,
    #[serde(default)]
    pub last_archived_at: Option<String>,
    #[serde(default)]
    pub archives: u32,
}

impl Ledger {
    pub fn empty() -> Self {
        Self {
            version: 1,
            entries: HashMap::new(),
        }
    }

    pub fn load(dir: &Path) -> Self {
        let path = dir.join("ledger.json");
        match std::fs::read_to_string(&path) {
            Ok(text) => serde_json::from_str(&text).unwrap_or_else(|e| {
                tracing::warn!(error = %e, path = %path.display(), "thread-archiver ledger unreadable — starting fresh");
                Self::empty()
            }),
            Err(_) => Self::empty(),
        }
    }

    /// Atomic write (tmp + rename), mirroring the upstream `_write_ledger`.
    pub fn save(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir).context("ledger dir create")?;
        let path = dir.join("ledger.json");
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, {
            let mut s = serde_json::to_vec_pretty(self)?;
            s.push(b'\n');
            s
        })?;
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }
}

/// Discord REST client bits — a slim reqwest wrapper (the bot already
/// speaks OpenAI-ish JSON everywhere; here it's Discord's API).
pub(crate) fn discord_client() -> reqwest::Client {
    reqwest::Client::builder()
        .user_agent("jerith-mesh-bot/0.1 (thread-archiver)")
        .build()
        .expect("static reqwest client config")
}

/// Resolve `#{name} ({id})` for the summary header plus the owning guild
/// id (needed for per-guild marker overrides); falls back to the bare id.
pub(crate) async fn thread_info(
    client: &reqwest::Client,
    token: &str,
    cid: u64,
) -> (String, Option<u64>) {
    let url = format!("{DISCORD_API}/channels/{cid}");
    let info = match client
        .get(&url)
        .header("Authorization", format!("Bot {token}"))
        .send()
        .await
    {
        Ok(resp) => resp.json::<Value>().await.ok(),
        Err(_) => None,
    };
    match info {
        Some(v) => {
            let name = v.get("name").and_then(Value::as_str).unwrap_or_default();
            let guild = v
                .get("guild_id")
                .and_then(Value::as_str)
                .and_then(|s| s.parse::<u64>().ok());
            let label = if name.is_empty() {
                format!("channel {cid}")
            } else {
                format!("#{name} ({cid})")
            };
            (label, guild)
        }
        None => (format!("channel {cid}"), None),
    }
}

/// Fetch the messages after `after_id` (None → first page).
/// `pub(crate)` — /briefing reuses this REST plumbing.
pub(crate) async fn fetch_messages(
    client: &reqwest::Client,
    token: &str,
    cid: u64,
    after_id: Option<u64>,
    limit: u32,
) -> Result<Vec<Value>> {
    let mut url = format!("{DISCORD_API}/channels/{cid}/messages?limit={limit}");
    if let Some(after) = after_id {
        url.push_str(&format!("&after={after}"));
    }
    let resp = client
        .get(&url)
        .header("Authorization", format!("Bot {token}"))
        .send()
        .await
        .with_context(|| format!("thread sweep fetch failed for {cid}"))?;
    let status = resp.status();
    let body: Value = resp
        .json()
        .await
        .context("discord returned non-JSON messages")?;
    if !status.is_success() {
        anyhow::bail!(
            "discord messages fetch for {cid} failed: {} ({})",
            status,
            body.as_str().unwrap_or("?")
        );
    }
    Ok(body.as_array().cloned().unwrap_or_default())
}

/// One sweep = walk the configured channels, publish summaries that have
/// new primary notes. Public for tests? No — an integration sweep hits
/// Discord; unit tests cover the pure parts. Returns channels progressed.
/// This bot's Discord user id (`GET /users/@me`) — lets the extractor
/// keep the archiver's own posts while dropping every other bot's.
async fn self_bot_id(client: &reqwest::Client, token: &str) -> Option<u64> {
    let url = format!("{}/users/@me", DISCORD_API);
    let resp = client
        .get(&url)
        .header("Authorization", format!("Bot {token}"))
        .send()
        .await
        .ok()?;
    let v: Value = resp.json().await.ok()?;
    v.get("id")
        .and_then(Value::as_str)
        .and_then(|s| s.parse::<u64>().ok())
}

pub async fn sweep(cfg: &BotConfig) -> usize {
    let arch = cfg.archiver.clone();
    let dir = archiver_dir();
    let mut ledger = Ledger::load(&dir);
    let client = discord_client();
    let self_id = self_bot_id(&client, &cfg.discord_token).await;
    let mut progressed = 0usize;

    for cid in &arch.channel_ids {
        if let Err(e) = sweep_one(
            &client,
            cfg,
            &arch,
            &mut ledger,
            dir.as_path(),
            *cid,
            self_id,
        )
        .await
        {
            tracing::warn!(channel = cid, error = %e, "thread-archiver sweep skipped channel");
        } else {
            progressed += 1;
        }
        // Gentle pacing: one REST call-set per channel per sweep.
        tokio::time::sleep(Duration::from_millis(1200)).await;
    }
    if let Err(e) = ledger.save(&dir) {
        tracing::warn!(error = %e, "thread-archiver ledger save failed");
    }
    progressed
}

async fn sweep_one(
    client: &reqwest::Client,
    cfg: &BotConfig,
    arch: &ArchiverConfig,
    ledger: &mut Ledger,
    dir: &Path,
    cid: u64,
    self_bot_id: Option<u64>,
) -> Result<()> {
    let key = cid.to_string();
    let after = ledger.entries.get(&key).and_then(|e| e.last_message_id);
    let messages = fetch_messages(
        client,
        &cfg.discord_token,
        cid,
        after,
        arch.max_messages_per_sweep,
    )
    .await?;
    if messages.is_empty() {
        return Ok(()); // nothing new
    }
    // Guild-aware markers must resolve before extraction (and the channel
    // fetch doubles as the existence/permission probe).
    let (label, guild_id) = thread_info(client, &cfg.discord_token, cid).await;
    let markers_owned = effective_markers(arch, guild_id);
    let markers = markers_owned.iter().map(String::as_str).collect::<Vec<_>>();
    let notes = extract_notes(&messages, &markers, arch.max_notes_per_sweep, self_bot_id);
    let newest = match newest_message_id(&messages) {
        Some(id) => id,
        None => return Ok(()),
    };
    // No worthwhile notes: still advance the ledger (the prose was read).
    if notes.is_empty() {
        let entry = ledger.entries.entry(key).or_default();
        entry.last_message_id = Some(newest);
        return Ok(());
    }
    let now = utc_timestamp_string(bootstrap_now_ms());
    let summary = render_summary(
        &label,
        &now,
        &notes.iter().map(String::as_str).collect::<Vec<_>>(),
    );

    // 1. memory embed — the node's private, never-mesh-leaving store.
    let mut embedded = false;
    match crate::api::memory_embed(cfg, &summary).await {
        Ok((id, provider)) => {
            embedded = true;
            tracing::info!(channel = cid, id = %id, provider = %provider, "thread summary embedded into node memory");
        }
        Err(e) => {
            tracing::warn!(channel = cid, error = %e, "thread summary embed failed — ledger NOT advanced; retry next sweep");
        }
    }
    if !embedded {
        // Leave the ledger untouched so the same batch is re-processed.
        return Ok(());
    }

    // 2. webhook parity with the OpenClaw skill (optional).
    if let Some(webhook) = arch.notes_webhook.as_deref().filter(|w| !w.is_empty())
        && let Err(e) = post_webhook(client, webhook, &summary).await {
            tracing::warn!(channel = cid, error = %e, "notes webhook post failed (memory already carries the summary)");
        }

    // 3. local markdown archive.
    let archive_dir = dir.join("archives");
    let _ = std::fs::create_dir_all(&archive_dir);
    let file = archive_dir.join(format!(
        "{cid}-{}.md",
        now.chars()
            .filter(|c| c.is_ascii_digit())
            .collect::<String>()
    ));
    let _ = std::fs::write(&file, &summary);

    let entry = ledger.entries.entry(key).or_default();
    entry.last_message_id = Some(newest);
    entry.last_archived_at = Some(now);
    entry.archives += 1;
    Ok(())
}

async fn post_webhook(client: &reqwest::Client, webhook: &str, body: &str) -> Result<()> {
    let resp = client
        .post(webhook)
        .json(&serde_json::json!({ "content": body }))
        .send()
        .await
        .context("notes webhook unreachable")?;
    let status = resp.status();
    if !(200..300).contains(&status.as_u16()) {
        anyhow::bail!("notes webhook returned {status}");
    }
    Ok(())
}

/// Where the archiver mirrors OpenClaw's STATE_DIR — under the node's
/// data dir (`~/.jerith-mesh/data/thread-archiver/`).
pub fn archiver_dir() -> PathBuf {
    crate::config::data_dir().join("thread-archiver")
}

/// Loop body spawned once from main.rs.
pub async fn run_loop(cfg: BotConfig) {
    let interval = cfg.archiver.interval_minutes.max(1);
    tracing::info!(
        channels = cfg.archiver.channel_ids.len(),
        interval_minutes = interval,
        "thread archiver loop started (OpenClaw skill port)"
    );
    if cfg.archiver.channel_ids.is_empty() {
        tracing::warn!(
            "thread archiver enabled but no channels listed — add to bot.toml: [archiver] channel_ids = [<thread ids>]"
        );
    }
    loop {
        sweep(&cfg).await;
        tokio::time::sleep(Duration::from_secs(interval * 60)).await;
    }
}

fn bootstrap_now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn msg(id: &str, author_id: &str, author: &str, content: &str, bot: bool) -> Value {
        json!({
            "id": id,
            "content": content,
            "author": {"id": author_id, "username": author, "global_name": author, "bot": bot},
        })
    }

    #[test]
    fn marker_resolution_prefers_guild_then_global_then_defaults() {
        let mut arch = ArchiverConfig::default();
        // Untouched config → the built-in list.
        assert_eq!(effective_markers(&arch, None), default_markers());
        assert_eq!(effective_markers(&arch, Some(1)), default_markers());
        // Global override (blank entries dropped, trimmed).
        arch.markers = [" decided: ", " ", "lore-claim"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            effective_markers(&arch, None),
            vec!["decided:".to_string(), "lore-claim".to_string()]
        );
        // Guild override wins for that guild only.
        arch.guild_markers.insert(
            42,
            crate::config::GuildMarkerSet {
                markers: vec!["bestemt".to_string(), "lore-claim:".to_string()],
            },
        );
        assert_eq!(
            effective_markers(&arch, Some(42)),
            vec!["bestemt".to_string(), "lore-claim:".to_string()]
        );
        assert_eq!(
            effective_markers(&arch, Some(43)),
            vec!["decided:".to_string(), "lore-claim".to_string()]
        );
        // An empty/blank guild override falls back to the global list.
        arch.guild_markers.insert(
            43,
            crate::config::GuildMarkerSet {
                markers: vec![" ".to_string()],
            },
        );
        assert_eq!(
            effective_markers(&arch, Some(43)),
            vec!["decided:".to_string(), "lore-claim".to_string()]
        );
    }

    #[test]
    fn extraction_surfaces_decisions_and_notes_chronologically() {
        let batch = vec![
            msg(
                "1003",
                "55",
                "noodly",
                "Decision: we use sqlite for memory.",
                false,
            ),
            msg(
                "1002",
                "66",
                "guest",
                "Action item: Noodly migrates the ledger.",
                false,
            ),
            msg(
                "1001",
                "77",
                "HelperBot",
                "Decision: I also decide things.",
                true,
            ), // bot → dropped
            msg("1000", "55", "noodly", "How much is the fish?", false), // no marker
        ];
        let notes = extract_notes(&batch, &DEFAULT_MARKERS, 10, None);
        assert_eq!(notes.len(), 2);
        assert!(notes[0].starts_with("guest: Action item"));
        assert!(notes[1].starts_with("noodly: Decision"));
        assert!(!notes.join("\n").contains("HelperBot"));
        assert!(!notes.join("\n").contains("fish"));
    }

    #[test]
    fn extraction_caps_the_batch() {
        let batch: Vec<Value> = (0..40)
            .map(|i| {
                msg(
                    format!("{i:06}").as_str(),
                    "500",
                    "n",
                    "decision: yes",
                    false,
                )
            })
            .collect();
        assert_eq!(extract_notes(&batch, &DEFAULT_MARKERS, 5, None).len(), 5);
    }

    #[test]
    fn extraction_keeps_the_archivers_own_posts_only() {
        // A third-party bot's decision line is dropped; the archiver's own
        // bot account (self id = 777) is a primary-note author and kept.
        let batch = vec![
            msg(
                "1005",
                "777",
                "JS",
                "Decision: self-authored notes count.",
                true,
            ),
            msg(
                "1004",
                "888",
                "HelperBot",
                "Decision: not my business.",
                true,
            ),
        ];
        let kept = extract_notes(&batch, &DEFAULT_MARKERS, 10, Some(777));
        assert_eq!(kept.len(), 1);
        assert!(kept[0].starts_with("JS: Decision"));
        assert_eq!(extract_notes(&batch, &DEFAULT_MARKERS, 10, None).len(), 0);
    }

    #[test]
    fn render_matches_the_upstream_format() {
        let html = render_summary(
            "#kiln-plans (12345)",
            "2026-10-13 07:30:00",
            &["noodly: Decision: brick kiln"],
        );
        assert!(html.starts_with("## Thread Archive - #kiln-plans (12345)\n*Archived 2026-10-13 07:30:00*\n\n### Decisions & notes\n\n  noodly: Decision: brick kiln\n"));
        let empty = render_summary("#x (1)", "2026-10-13 07:30:00", &[]);
        assert!(empty.contains("(no notes extracted)"));
    }

    #[test]
    fn ledger_roundtrips_and_inherits_the_upstream_version() {
        let dir = std::env::temp_dir().join(format!("arch-test-{}", std::process::id()));
        let mut ledger = Ledger::empty();
        ledger.entries.insert(
            "42".to_string(),
            LedgerEntry {
                last_message_id: Some(999_000),
                last_archived_at: Some("2026-10-13 07:30:00".into()),
                archives: 1,
            },
        );
        ledger.save(&dir).unwrap();
        let loaded = Ledger::load(&dir);
        assert_eq!(loaded.version, 1); // OpenClaw skill's ledger version
        assert_eq!(loaded.entries["42"].last_message_id, Some(999_000));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupt_ledger_falls_back_to_empty() {
        let dir = std::env::temp_dir().join(format!("arch-test2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("ledger.json"), "{not json").unwrap();
        assert!(Ledger::load(&dir).entries.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn newest_snowflake_is_max() {
        let batch = vec![
            msg("999", "1", "a", "x", false),
            msg("1010", "1", "a", "y", false),
        ];
        assert_eq!(newest_message_id(&batch), Some(1010));
        assert_eq!(newest_message_id(&[]), None);
    }

    #[test]
    fn utc_format_hits_known_instants() {
        assert_eq!(utc_timestamp_string(0), "1970-01-01 00:00:00");
        // 2026-01-01 00:00Z = 1,767,225,600 (20,454 days) · +285 days to
        // Oct 13 00:00 = 1,791,849,600 · + 27,000 s for 07:30.
        assert_eq!(
            utc_timestamp_string(1_791_876_600 * 1000),
            "2026-10-13 07:30:00"
        );
        // 2000-02-29 12:00Z: Jan 1 2000 = 946,684,800 · +59 days (Feb 29
        // 00:00) · + 12 h → 951,825,600 (leap-day sanity).
        assert_eq!(
            utc_timestamp_string(951_825_600 * 1000),
            "2000-02-29 12:00:00"
        );
    }
}
