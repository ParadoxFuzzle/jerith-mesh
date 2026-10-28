//! The plugin-lane receipt log (docs/PLUGIN-CHANNEL.md's audit-chain
//! open question). Every CLI plugin-lane ACTION appends one JSON line
//! to `<node home>/plugin.json`:
//!
//! - `install` / `remove` — what landed (or left) plugins.d, with the
//!   bundle digest and the PIN that authorized it — including WHERE the
//!   pin came from, so a rotation (pin-rotate receipt) explains any
//!   change in what subsequent installs accept;
//! - `pin-rotate` — the trust-robot's own record: old key → new key
//!   per plugin id, the moment rotation lands;
//! - `channel-verify` — the freshness check itself, stamped with the
//!   index's `generated_ms`, the age OBSERVED at check time and the
//!   window it passed under.
//!
//! NOT a security boundary: the verify chain, the pin table and the
//! bot's fail-closed manifest gate remain the enforcement. This is the
//! operator's human-readable audit trail — the receipt log answers
//! "when, with which key, how fresh" after the fact. Signing the chain
//! (tamper-evident receipts) stays a deliberate open question: receipts
//! are 0600 operator-owned files, same trust posture as pins file.
//!
//! Appends only; a torn/missing line skips on read rather than
//! blackholing the file, but malformed LOGGED shapes should not occur —
//! validation happens before record, like everywhere else in the lane.

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::config::NodeConfig;

/// The receipt log lives at the node home root — siblings of
/// config.toml / bot.toml, NOT under `plugins/` (an id may legally be
/// "plugin.json"; the home root cannot collide with the bundle tree).
pub const RECEIPT_FILE: &str = "plugin.json";

pub fn receipts_path_under(home: &Path) -> PathBuf {
    home.join(RECEIPT_FILE)
}

pub fn receipts_path() -> PathBuf {
    receipts_path_under(&NodeConfig::home())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceiptEntry {
    /// Wall-clock append stamp (epoch ms).
    pub ts_ms: u64,
    /// `install` | `remove` | `pin-rotate` | `channel-verify`.
    pub action: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plugin: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Publisher pubkey in play (64-hex; install's signer / rotate's
    /// new pin / channel's publisher). Full value — a receipt should
    /// not make the operator go hunting for the other half.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub publisher: Option<String>,
    /// The bundle artifact digest (the same one `.sha256` + a channel
    /// row carry — the join key across all three surfaces).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bundle_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pin: Option<PinReceipt>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel: Option<ChannelReceipt>,
    /// Free-form specifics (trash path, row count, rotate edges).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// The pin that authorized an action, with its SOURCE — the audit
/// trail's answer to "was this bot.toml's table, the legacy pins file,
/// or a one-shot --pubkey".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PinReceipt {
    /// `flag` | `bot.toml` | `pins-file`.
    pub source: String,
    /// 64-hex pin applied.
    pub pin: String,
}

/// The freshness state a channel check ran under.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelReceipt {
    /// The channel source as the operator addressed it (url or dir).
    pub source: String,
    /// The index's own generated_ms (as published).
    pub generated_ms: u64,
    /// now − generated_ms at the moment of the check.
    pub observed_age_ms: u64,
    /// The window enforced during this check (0 would have meant the
    /// 24 h default — resolved and recorded as seen).
    pub max_age_ms: u64,
}

impl ReceiptEntry {
    /// Append the entry to the receipt log under an explicit home (one
    /// JSON line; the file is created 0600 and NEVER rewritten — an
    /// append-only trail is the whole point).
    pub fn record_under(&self, home: &Path) -> anyhow::Result<()> {
        let line = serde_json::to_string(self).map_err(|e| anyhow::anyhow!("receipt {e}"))?;
        let path = receipts_path_under(home);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        file.write_all(line.as_bytes())?;
        file.write_all(b"\n")?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            // The CREATE got default umask perms; ensure 0600 in place
            // (idempotent — an operation on an existing file never
            // loosens anything).
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
        }
        Ok(())
    }

    /// The real-home record (the CLI call).
    pub fn record(&self) -> anyhow::Result<()> {
        self.record_under(&NodeConfig::home())
    }
}

/// Read the log under an explicit home: entries in append order.
/// Unreadable lines are SKIPPED (a torn tail from a crash must not
/// blackhole an audit trail) — but only whole-line garbage counts; a
/// malformed KNOWN shape would not have been written.
pub fn read_under(home: &Path) -> Vec<ReceiptEntry> {
    let Ok(text) = std::fs::read_to_string(receipts_path_under(home)) else {
        return Vec::new();
    };
    text.lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

pub fn read() -> Vec<ReceiptEntry> {
    read_under(&NodeConfig::home())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("jerith-receipt-test-{}-{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("tmp dir");
        dir
    }

    fn entry(action: &str, plugin: &str, detail: &str) -> ReceiptEntry {
        ReceiptEntry {
            ts_ms: 1730000000000,
            action: action.into(),
            plugin: Some(plugin.into()),
            version: Some("0.1.1".into()),
            publisher: Some("a".repeat(64)),
            bundle_sha256: Some("b".repeat(64)),
            pin: Some(PinReceipt {
                source: "bot.toml".into(),
                pin: "c".repeat(64),
            }),
            channel: Some(ChannelReceipt {
                source: "https://example.com/channel.json".into(),
                generated_ms: 1729999999000,
                observed_age_ms: 10_000,
                max_age_ms: 86_400_000,
            }),
            detail: Some(detail.into()),
        }
    }

    #[test]
    fn append_read_roundtrip_is_ordered() {
        let home = tmp("append");
        entry("install", "weather", "first")
            .record_under(&home)
            .unwrap();
        entry("pin-rotate", "weather", "second")
            .record_under(&home)
            .unwrap();
        entry("remove", "weather", "third")
            .record_under(&home)
            .unwrap();
        let entries = read_under(&home);
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].action, "install");
        assert_eq!(entries[2].action, "remove");
        // Optional fields survive the JSONL round-trip.
        assert_eq!(entries[0].pin.as_ref().unwrap().source, "bot.toml");
        assert_eq!(entries[0].channel.as_ref().unwrap().observed_age_ms, 10_000);
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn missing_log_reads_empty_and_torn_lines_skip() {
        let home = tmp("missing");
        assert!(read_under(&home).is_empty());

        // A torn tail (power loss mid-append) must not blackhole the log.
        let home2 = tmp("torn");
        entry("install", "weather", "kept")
            .record_under(&home2)
            .unwrap();
        let path = receipts_path_under(&home2);
        let mut raw = std::fs::read_to_string(&path).unwrap();
        raw.push_str("{\"ts_ms\": 1, \"actio");
        std::fs::write(&path, raw).unwrap();
        let entries = read_under(&home2);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].detail.as_deref(), Some("kept"));
        let _ = std::fs::remove_dir_all(&home);
        let _ = std::fs::remove_dir_all(&home2);
    }
}
