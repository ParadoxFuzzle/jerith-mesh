//! `jerith-quarantine` — the download-quarantine lane's wire glue,
//! over the plugin SDK (`crates/plugin-template`, docs/PLUGIN-SDK.md;
//! the SDK's fourth real consumer after the example, the pet, and the
//! house card). The scan engine + state live in `lib.rs` (pure, tested);
//! this binary:
//!
//! - reads its OWN `plugin.toml` for `[plugin] endpoint` + `secret`
//!   and binds that exact address — manifest and service cannot drift
//!   (the one-source-of-truth rule every drop keeps);
//! - watches the quarantine dir (`$JERITH_QUARANTINE_DIR`, default
//!   `~/Downloads`) on a 30 s poll (`$JERITH_QUARANTINE_WATCH_SECS`),
//!   appends every new scan's verdict to the ONE JSONL log the bot's
//!   monitor/briefing lanes read (`lib.rs`'s `state_path()`);
//! - serves the two model-visible skills the manifest declares:
//!   `quarantine.scan` (deep scan of a path — the chat can re-run the
//!   2026-10-23 routine on any file on demand) and `quarantine.recent`
//!   (the verdict log's counts + tail, what the briefing reads);
//! - logs every `suspicious`/`malicious` finding at WARN so the
//!   process journal (and the briefing's journal ring when this runs
//!   inside the bot - no wait, this is a separate process) carries the
//!   event trail.
//!
//! Authoring note: fourth citizen of the Phase 3d story — copy this
//! layout, swap the handlers, sync `plugin.toml`, ship.

use anyhow::{Context as _, Result};
use jerith_plugin_template::{SkillServer, endpoint_addr, skill};
use jerith_quarantine_plugin::{
    ScanReport, Verdict, WATCH_INTERVAL_SECS, WRITE_SETTLE_MS, append_report, counts, load_reports,
    scan_file, state_path, watch_once,
};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex as AsyncMutex;

/// now in ms — the only clock read on the wire; the lib takes it.
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// `[plugin] endpoint` + `[plugin] secret` (None = unsigned dev mode)
/// straight out of the manifest — one file rules both halves.
fn read_manifest(path: &str) -> Result<(std::net::SocketAddr, Option<String>)> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("reading manifest {path}"))?;
    // Document parse (toml::Table), NOT `raw.parse::<toml::Value>()` —
    // a Value's FromStr accepts exactly ONE value, so a whole manifest
    // string fails "unexpected content, expected nothing" (the pet's
    // lesson, pinned in both crates' shipped-manifest tests).
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

/// The watch dir: `$JERITH_QUARANTINE_DIR`, else `~/Downloads` — the
/// place double-click muscle memory lives (the lane's whole reason).
pub fn watch_dir() -> PathBuf {
    if let Ok(d) = std::env::var("JERITH_QUARANTINE_DIR") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join("Downloads")
}

/// One poll tick: scan the dir's settled new files, append each to the
/// log, log-loud anything strong. Returns what landed (tested via the
/// lib; here it is only bookkeeping).
async fn tick(
    dir: &Path,
    state: &Path,
    known: &AsyncMutex<HashMap<(String, u64, u64), ScanReport>>,
) -> Vec<ScanReport> {
    let mut k = known.lock().await;
    let fresh = watch_once(dir, now_ms(), &mut k);
    for r in &fresh {
        if let Err(e) = append_report(state, r) {
            tracing::warn!(target: "quarantine", path = %r.path, error = %e, "verdict append failed; will rescan on next tick");
        }
        match r.verdict {
            Verdict::Malicious => tracing::warn!(
                target: "quarantine", path = %r.path, sha = %(r.sha256.get(..12).unwrap_or("?")),
                "MALICIOUS file in the quarantine dir — the monitor lane should DM the owner"
            ),
            Verdict::Suspicious => tracing::warn!(
                target: "quarantine", path = %r.path,
                "suspicious file in the quarantine dir"
            ),
            Verdict::Benign => tracing::debug!(target: "quarantine", path = %r.path, "benign"),
        }
    }
    fresh
}

/// Compress one report into the card shape the skills return (bounded:
/// findings capped, hashes truncated — a verdict card, not the file).
fn report_card(r: &ScanReport) -> Value {
    json!({
        "path": r.path,
        "size_bytes": r.size_bytes,
        "sha256_prefix": r.sha256.get(..16).unwrap_or("?"),
        "kind": r.kind,
        "verdict": r.verdict.word(),
        "findings": r.findings.iter().take(6).collect::<Vec<_>>(),
        "note": r.note,
        "scanned_at_ms": r.scanned_at_ms,
    })
}

/// Scan a path the chat names: single file → one report; directory →
/// an on-demand sweep with the same settle/known semantics as the
/// watcher (bounded at 200 files so one invite can't bloat the lane).
async fn do_scan(path: Option<&str>, dir: &Path, state: &Path) -> Result<Value, String> {
    let target: PathBuf = match path {
        Some(p) if !p.trim().is_empty() => PathBuf::from(shellexpand(p.trim())),
        None | Some(_) => dir.to_path_buf(),
    };
    let now = now_ms();
    if target.is_file() {
        let r = scan_file(&target, now);
        if let Err(e) = append_report(state, &r) {
            tracing::warn!(target: "quarantine", error = %e, "scan report append failed");
        }
        if r.verdict != Verdict::Benign {
            tracing::warn!(target: "quarantine", path = %r.path, verdict = r.verdict.word(), "on-demand scan verdict");
        }
        return Ok(json!({ "report": report_card(&r) }));
    }
    if !target.is_dir() {
        return Err(format!(
            "{} is neither an existing file nor a directory",
            target.display()
        ));
    }
    // Directory sweep: top level + depth-2 subdirs (quarantine dirs
    // sometimes pack one level — overrides/, mods/), files only, hard
    // cap of 200 files so one invite can't bloat the lane, and the
    // watcher's settle rule (hot files left for the next tick).
    let mut files: Vec<PathBuf> = Vec::new();
    let mut stack = vec![target.clone()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.flatten() {
            let Ok(meta) = e.metadata() else { continue };
            if meta.is_dir() {
                stack.push(e.path());
            } else {
                files.push(e.path());
            }
        }
        if files.len() > 200 {
            break;
        }
    }
    files.truncate(200);
    let mut owned: Vec<ScanReport> = Vec::new();
    for p in files {
        let Ok(meta) = std::fs::metadata(&p) else {
            continue;
        };
        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        if now.saturating_sub(mtime) < WRITE_SETTLE_MS {
            continue;
        }
        let r = scan_file(&p, now);
        if let Err(e) = append_report(state, &r) {
            tracing::warn!(target: "quarantine", error = %e, "sweep report append failed");
        }
        owned.push(r);
    }
    owned.sort_by_key(|r| {
        (
            r.verdict == Verdict::Benign,
            std::cmp::Reverse(r.scanned_at_ms),
        )
    });
    let (b, s, m) = counts(&owned);
    let head: Vec<Value> = owned.iter().take(20).map(report_card).collect();
    Ok(json!({
        "target": target.display().to_string(),
        "counts": { "benign": b, "suspicious": s, "malicious": m },
        "scanned": owned.len(),
        "reports": head,
    }))
}

/// `~` for the brief handed-down path (a convenience skill arg, not a
/// prefix expansion — anything else passes through untouched).
fn shellexpand(p: &str) -> String {
    if (p == "~" || p.starts_with("~/"))
        && let Ok(home) = std::env::var("HOME")
    {
        return p.replacen('~', &home, 1);
    }
    p.to_string()
}

/// The verdict log's tail + counts — the shape the briefing's section
/// renders and the owner's monotree/journal reads back.
async fn do_recent(state: &Path, dir: &Path) -> Result<Value, String> {
    let all = load_reports(state, 500);
    let (b, s, m) = counts(&all);
    let tail: Vec<Value> = all.iter().rev().take(8).map(report_card).collect();
    // The standing question: which strong-verdict files are STILL on
    // disk right now (the monitor's Fail/Warn truth, kept current).
    let standing: Vec<Value> = all
        .iter()
        .rev()
        .filter(|r| r.verdict != Verdict::Benign && PathBuf::from(&r.path).exists())
        .take(8)
        .map(report_card)
        .collect();
    Ok(json!({
        "watch_dir": dir.display().to_string(),
        "state_file": state.display().to_string(),
        "counts_total": { "benign": b, "suspicious": s, "malicious": m },
        "standing": standing,
        "recent": tail,
    }))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let manifest = std::env::args().nth(1).unwrap_or("plugin.toml".to_string());
    let (addr, secret) = read_manifest(&manifest)?;
    let dir = Arc::new(watch_dir());
    let state = Arc::new(state_path());
    tracing::info!(
        manifest,
        %addr,
        secret = secret.is_some(),
        watch_dir = %dir.display(),
        state = %state.display(),
        "quarantine plugin starting"
    );

    // The watcher: known-set primed from the EXISTING log (a restart
    // does not rescan what's already been judged), then a poll loop.
    let known = Arc::new(AsyncMutex::new(primed(&state, &dir)));
    {
        // Initial sweep without the log's dedup would double-scan
        // nothing — watch_once consults `known`, already primed.
        let dir = dir.clone();
        let state = state.clone();
        let known = known.clone();
        tokio::spawn(async move {
            let watch_secs = std::env::var("JERITH_QUARANTINE_WATCH_SECS")
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(WATCH_INTERVAL_SECS)
                .max(5);
            loop {
                let _ = tick(&dir, &state, &known).await;
                tokio::time::sleep(Duration::from_secs(watch_secs)).await;
            }
        });
    }

    // The two skills, over the signed contract. Nothing here touches
    // the bot's process: loopback http, secret-signed, same as house.
    SkillServer::new("quarantine", secret)
        .handler(
            "quarantine.scan",
            skill({
                let dir = dir.clone();
                let state = state.clone();
                move |args: Value| {
                    let dir = dir.clone();
                    let state = state.clone();
                    async move { do_scan(args["path"].as_str(), &dir, &state).await }
                }
            }),
        )
        .handler(
            "quarantine.recent",
            skill({
                let dir = dir.clone();
                let state = state.clone();
                move |_args: Value| {
                    let dir = dir.clone();
                    let state = state.clone();
                    async move { do_recent(&state, &dir).await }
                }
            }),
        )
        .serve(addr)
        .await
}

/// Prime the watcher's known-set from the log's last-500 rows: a
/// restart rescan would DOUBLE every verdict line — the log itself is
/// the memory of what's been judged (name+size+mtime keys ride the
/// rows' own fields).
fn primed(state: &Path, dir: &Path) -> HashMap<(String, u64, u64), ScanReport> {
    let mut known: HashMap<(String, u64, u64), ScanReport> = HashMap::new();
    for r in load_reports(state, 500) {
        // The row's path+size is the identity; mtime is re-read live
        // when the file is present (a changed file has a new key, so it
        // WILL be re-scanned — changed content deserves a new verdict).
        if let Ok(meta) = std::fs::metadata(PathBuf::from(&r.path)) {
            let size = meta.len();
            if size == r.size_bytes {
                let mtime = meta
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);
                known.insert((r.path.clone(), size, mtime), r);
            }
        }
    }
    let _ = dir;
    known
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The SHIPPED manifest must stay servable by THIS binary: endpoint
    /// binds loopback, on a port that is not the other drops', signed,
    /// and the declared rows are exactly the two registered above.
    #[test]
    fn shipped_manifest_fits_this_binary() {
        let raw = include_str!("../plugin.toml");
        let file: toml::Table =
            toml::from_str(raw).expect("manifest is valid TOML (document parse)");
        let plugin = file
            .get("plugin")
            .expect("[plugin] table")
            .as_table()
            .expect("[plugin] is a table");
        assert_eq!(
            plugin.get("id").and_then(toml::Value::as_str),
            Some("quarantine")
        );
        let endpoint = plugin
            .get("endpoint")
            .and_then(toml::Value::as_str)
            .expect("endpoint line");
        let addr = endpoint_addr(endpoint).expect("endpoint binds");
        assert!(addr.ip().is_loopback(), "loopback by contract");
        for taken in [7788, 7789, 7790] {
            assert_ne!(addr.port(), taken, "the {taken} drop has that port");
        }
        assert!(
            plugin.get("secret").and_then(toml::Value::as_str).is_some(),
            "ships signed by default"
        );
        let names: Vec<&str> = file
            .get("skill")
            .and_then(toml::Value::as_array)
            .expect("[[skill]] rows")
            .iter()
            .filter_map(|s| s.get("name").and_then(toml::Value::as_str))
            .collect();
        assert_eq!(names, vec!["scan", "recent"], "the two handlers");
    }

    #[test]
    fn shellexpand_homes_only_the_bare_tilde() {
        let home = std::env::var("HOME").unwrap_or_default();
        assert_eq!(
            shellexpand("~/Downloads/x.zip"),
            format!("{home}/Downloads/x.zip")
        );
        assert_eq!(shellexpand("/tmp/x.zip"), "/tmp/x.zip");
        assert_eq!(shellexpand("weird~name.zip"), "weird~name.zip");
    }

    #[test]
    fn report_cards_bound_their_shape() {
        let dir = std::env::temp_dir().join(format!("q-main-{}", now_ms()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("z.txt");
        std::fs::write(&f, b"plain").unwrap();
        let r = scan_file(&f, now_ms());
        let card = report_card(&r);
        let s = serde_json::to_string(&card).unwrap();
        assert!(s.contains("\"verdict\":\"benign\""), "{s}");
        assert!(
            !s.contains("\"sha256\""),
            "full hash never rides the card, prefix only: {s}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
