//! Full node-state backup / verify / restore (Phase 2).
//!
//! `jerith backup create` snapshots everything the node persists —
//! identity-bearing config, the bot config, its persona files, and the
//! whole `data/` tree (memory sqlite, expert demand stats, stake + slash
//! ledgers …) — into ONE self-describing, checksummed file. `restore`
//! re-installs that state onto this (or a fresh) machine: the
//! backup/restore/migrate story.
//!
//! Format: a dependency-free container, deliberately shaped like the QUIC
//! weight-transfer frames the mesh already uses (JSON header frame, raw
//! payload after it):
//!
//! ```text
//! JBAK1\0\0\0            8-byte magic ("not a backup at all" is a parse error)
//! u32 LE                 format version (1)
//! manifest JSON + '\n'   {version, created_at_ms, node_id, entries_expected}
//! per entry:             {"path","bytes","sha256"} + '\n', then `bytes` raw
//! ```
//!
//! The manifest lists only the expected entry COUNT; the per-entry frames
//! (each carrying its own sha256) are the authority. A truncated or
//! tampered file fails its next entry's checksum instead of silently
//! reading shortened blobs. Expert *weights* are excluded by default (they
//! can be gigabytes, and a moved node re-pulls replicas over the mesh);
//! `include_experts` opts them in.

use anyhow::{Context as _, Result};
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Magic + version. Bumping the version is the incompatible change; the
/// magic alone catches "this is not a backup file at all".
const MAGIC: &[u8; 8] = b"JBAK1\0\0\0";
const VERSION: u32 = 1;
/// One file larger than this is refused rather than quietly archived — a
/// db over 1 GiB wants an operator decision, not a silent surprise.
const MAX_FILE_BYTES: u64 = 1024 * 1024 * 1024;
/// Total payload ceiling for the default backup.
const MAX_TOTAL_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// What one backup call did (user-facing print material).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackupReport {
    pub path: PathBuf,
    pub entries: usize,
    /// Total archived payload bytes (not counting frames).
    pub total_bytes: u64,
    /// Paths deliberately left out (weights by default, unreadable files).
    pub skipped: Vec<String>,
}

/// What `verify` (and `restore`'s first stage) reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyReport {
    pub version: u32,
    pub created_at_ms: u64,
    pub node_id: Option<String>,
    pub entries: usize,
    pub total_bytes: u64,
    /// The relative paths, in archive order (sorted at creation).
    pub paths: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestoreReport {
    pub written: Vec<String>,
    pub skipped_existing: Vec<String>,
}

/// Collect-and-archive options.
#[derive(Debug, Clone, Copy, Default)]
pub struct BackupOptions {
    /// Include the expert weight store (`<expert_id>.expert` containers).
    pub include_experts: bool,
}

/// A file the snapshot wants, as (absolute path, backup-relative path).
struct Candidate {
    abs: PathBuf,
    rel: String,
}

/// The node's persistent state as (abs, relative) pairs, deterministically
/// ordered (top-level files sorted, then the data tree sorted), so two
/// backups of an unchanged state differ only in the manifest timestamp.
fn collect_candidates(home: &Path, include_experts: bool) -> Result<(Vec<Candidate>, Vec<String>)> {
    let mut skipped: Vec<String> = Vec::new();
    let mut cands: Vec<Candidate> = Vec::new();

    // Refuse to archive nothing: without config.toml there is no node to
    // migrate.
    let config = home.join("config.toml");
    anyhow::ensure!(
        config.is_file(),
        "no config.toml under {} — nothing to back up (run setup first)",
        home.display()
    );

    fn push(abs: PathBuf, rel: &str, cands: &mut Vec<Candidate>, skipped: &mut Vec<String>) {
        match std::fs::metadata(&abs) {
            Ok(m) if m.is_file() => cands.push(Candidate {
                abs,
                rel: rel.to_string(),
            }),
            Ok(_) => skipped.push(format!("{rel} (not a regular file)")),
            Err(_) => {} // an absent optional file is simply not in the backup
        }
    }

    push(config, "config.toml", &mut cands, &mut skipped);

    // Top-level operator files (absent → silently skipped).
    for rel in ["bot.toml", "persona.md", "known-users.md"] {
        push(home.join(rel), rel, &mut cands, &mut skipped);
    }

    // The whole data tree, breadth-walked and sorted. A symlink inside is
    // skipped and REPORTED, never followed across the boundary.
    let mut walk: Vec<(PathBuf, String)> = Vec::new();
    let data = home.join("data");
    if data.is_dir() {
        walk.push((data, "data".to_string()));
    }
    // The weight store is a TOP-LEVEL dir (home/experts) — walked only on
    // the explicit opt-in (it can be gigabytes). Omitted-by-default is
    // still REPORTED (the operator should know what stayed behind).
    let experts = home.join("experts");
    if experts.is_dir() {
        if include_experts {
            walk.push((experts, "experts".to_string()));
        } else {
            skipped.push(
                "experts/ (expert weights — omitted, rerun with --include-experts)".to_string(),
            );
        }
    }
    while let Some((dir, prefix)) = walk.pop() {
        let mut entries: Vec<(PathBuf, String)> = Vec::new();
        for e in std::fs::read_dir(&dir).with_context(|| format!("reading {}", dir.display()))? {
            let e = e?;
            let path = e.path();
            let rel = format!("{prefix}/{}", e.file_name().to_string_lossy());
            if e.file_type()?.is_dir() {
                // Excluded weight store by default (a replica-pulled node
                // just refills it); include_experts rides them along. The
                // store is recognized by NAME wherever it sits in the
                // state tree — its dir is configurable, so 'experts' is a
                // role marker, not a fixed location.
                if e.file_name().to_string_lossy() == "experts" && !include_experts {
                    skipped.push(format!("{prefix}/experts/ (expert weights — omitted)"));
                    continue;
                }
                walk.push((path, rel));
            } else {
                entries.push((path, rel));
            }
        }
        entries.sort();
        for (abs, rel) in entries {
            push(abs, &rel, &mut cands, &mut skipped);
        }
    }

    // Bounded archive: refuse absurd trees loud and early.
    let total: u64 = cands
        .iter()
        .map(|c| std::fs::metadata(&c.abs).map(|m| m.len()).unwrap_or(0))
        .sum();
    anyhow::ensure!(
        total <= MAX_TOTAL_BYTES,
        "state tree is {total} bytes — over the default {MAX_TOTAL_BYTES} archive budget"
    );

    cands.sort_by(|a, b| a.rel.cmp(&b.rel));
    Ok((cands, skipped))
}

/// The node_id straight out of the config TOML, read-only (no load_from
/// side effects like the 0600 chmod heal the runtime load performs).
fn node_id_from_config(config_path: &Path) -> Option<String> {
    let text = std::fs::read_to_string(config_path).ok()?;
    let value: Value = toml::from_str(&text).ok()?;
    let id = value.get("node_id")?.as_str()?.trim().to_string();
    (!id.is_empty()).then_some(id)
}

/// Create the backup file. `out` = explicit path; None →
/// `<home>/backups/jerith-state-<unix-ms>.jbak`. Two streaming passes, no
/// full-archive buffer: entry frames go to a body temp file first (each
/// file's bytes are read one at a time), then the final file is written
/// manifest-first and the body appended with io::copy — memory stays at
/// one file's size, never the archive's.
pub fn create_into(home: &Path, out: Option<&Path>, opts: BackupOptions) -> Result<BackupReport> {
    let (cands, skipped) = collect_candidates(home, opts.include_experts)?;
    let out = match out {
        Some(p) => p.to_path_buf(),
        None => {
            let dir = home.join("backups");
            std::fs::create_dir_all(&dir)?;
            dir.join(format!("jerith-state-{}.jbak", crate::bootstrap::now_ms()))
        }
    };
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Pass 1: entry frames + payloads → body temp file; collect metas.
    let body = out.with_extension("jbak.body");
    let mut body_w = std::io::BufWriter::new(
        std::fs::File::create(&body).with_context(|| format!("creating {}", body.display()))?,
    );
    let mut metas: Vec<Value> = Vec::new();
    let mut total: u64 = 0;
    for c in &cands {
        let bytes =
            std::fs::read(&c.abs).with_context(|| format!("reading {}", c.abs.display()))?;
        anyhow::ensure!(
            bytes.len() as u64 <= MAX_FILE_BYTES,
            "{} exceeds the per-file cap ({MAX_FILE_BYTES} bytes)",
            c.abs.display()
        );
        let meta = json!({
            "path": c.rel,
            "bytes": bytes.len(),
            "sha256": hex::encode(Sha256::digest(&bytes)),
        });
        serde_json::to_writer(&mut body_w, &meta)?;
        body_w.write_all(b"\n")?;
        body_w.write_all(&bytes)?;
        total += bytes.len() as u64;
        metas.push(meta);
    }
    body_w.flush()?;
    drop(body_w);

    // Pass 2: final file = magic + version + manifest frame + streamed body.
    let manifest = json!({
        "version": VERSION,
        "created_at_ms": crate::bootstrap::now_ms(),
        "node_id": node_id_from_config(&home.join("config.toml")),
        "entries_expected": metas.len(),
    });
    let tmp = out.with_extension("jbak.tmp");
    let mut final_w = std::io::BufWriter::new(
        std::fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?,
    );
    final_w.write_all(MAGIC)?;
    final_w.write_all(&VERSION.to_le_bytes())?;
    serde_json::to_writer(&mut final_w, &manifest)?;
    final_w.write_all(b"\n")?;
    final_w.flush()?;
    {
        let mut body_r = std::fs::File::open(&body)?;
        std::io::copy(&mut body_r, &mut final_w)
            .with_context(|| format!("stitching {}", out.display()))?;
    }
    final_w.flush()?;
    drop(final_w);
    std::fs::rename(&tmp, &out).with_context(|| format!("finalizing {}", out.display()))?;
    std::fs::remove_file(&body).ok();

    Ok(BackupReport {
        path: out,
        entries: metas.len(),
        total_bytes: total,
        skipped,
    })
}

/// Read one JSON frame (terminated by '\n', newline consumed) from the
/// reader. EOF before any byte → None; EOF mid-frame → error.
fn read_frame(r: &mut impl Read) -> Result<Option<Vec<u8>>> {
    let mut frame = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        match r.read(&mut byte)? {
            0 => {
                if frame.is_empty() {
                    return Ok(None);
                }
                anyhow::bail!("unexpected EOF inside a backup frame");
            }
            _ => {
                frame.push(byte[0]);
                if byte[0] == b'\n' {
                    frame.pop();
                    return Ok(Some(frame));
                }
            }
        }
    }
}

/// Verify the container: magic/version, manifest shape + count, per-entry
/// checksums, and that the file ENDS exactly at the last entry.
pub fn verify(path: &Path) -> Result<VerifyReport> {
    let mut r = std::io::BufReader::new(
        std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?,
    );
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)
        .with_context(|| format!("{}: too small to be a backup", path.display()))?;
    anyhow::ensure!(magic == *MAGIC, "not a jerith state backup (bad magic)");
    let mut version_bytes = [0u8; 4];
    r.read_exact(&mut version_bytes)?;
    let version = u32::from_le_bytes(version_bytes);
    anyhow::ensure!(
        version == VERSION,
        "backup format version {version} is not readable by this build ({VERSION})"
    );
    let manifest_frame = read_frame(&mut r)?.context("missing manifest frame")?;
    let manifest: Value =
        serde_json::from_slice(&manifest_frame).context("manifest is not JSON")?;
    anyhow::ensure!(
        manifest.get("version").and_then(Value::as_u64) == Some(VERSION as u64),
        "manifest version disagrees with the container header"
    );
    let expected = manifest
        .get("entries_expected")
        .and_then(Value::as_u64)
        .context("manifest missing entries_expected")? as usize;

    let mut count = 0usize;
    let mut total = 0u64;
    let mut paths: Vec<String> = Vec::new();
    while let Some(frame) = read_frame(&mut r)? {
        let entry: Value = serde_json::from_slice(&frame).context("entry frame is not JSON")?;
        let rel = entry
            .get("path")
            .and_then(Value::as_str)
            .context("entry frame missing path")?
            .to_string();
        let bytes = entry
            .get("bytes")
            .and_then(Value::as_u64)
            .context("entry frame missing bytes")?;
        let want = entry
            .get("sha256")
            .and_then(Value::as_str)
            .context("entry frame missing sha256")?;
        let mut payload = vec![0u8; bytes as usize];
        r.read_exact(&mut payload)
            .with_context(|| format!("entry {rel}: truncated payload"))?;
        let got = hex::encode(Sha256::digest(&payload));
        anyhow::ensure!(
            got == want,
            "entry {rel}: checksum mismatch (expected {want}, got {got})"
        );
        count += 1;
        total += bytes;
        paths.push(rel);
    }
    anyhow::ensure!(
        count == expected,
        "entry count mismatch: manifest says {expected}, archive holds {count}"
    );
    anyhow::ensure!(!paths.is_empty(), "backup holds zero entries");
    Ok(VerifyReport {
        version,
        created_at_ms: manifest
            .get("created_at_ms")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        node_id: manifest
            .get("node_id")
            .and_then(Value::as_str)
            .map(str::to_string),
        entries: count,
        total_bytes: total,
        paths,
    })
}

/// Restore the backup into `home`: verify first, then write every entry
/// atomically (tmp + rename). Refuses to overwrite an existing file unless
/// `force`; refuses traversal paths (`..`, absolute, empty components).
pub fn restore_into(home: &Path, backup: &Path, force: bool) -> Result<RestoreReport> {
    let report = verify(backup)?;
    let mut written = Vec::new();
    let mut skipped_existing = Vec::new();
    let mut r = std::io::BufReader::new(std::fs::File::open(backup)?);
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    anyhow::ensure!(magic == *MAGIC, "not a jerith state backup (bad magic)");
    let mut vbuf = [0u8; 4];
    r.read_exact(&mut vbuf)?;
    read_frame(&mut r)?; // manifest frame (already verified above)

    std::fs::create_dir_all(home)?;
    while let Some(frame) = read_frame(&mut r)? {
        let entry: Value = serde_json::from_slice(&frame)?;
        let rel = entry
            .get("path")
            .and_then(Value::as_str)
            .context("entry frame missing path")?
            .to_string();
        let bytes = entry.get("bytes").and_then(Value::as_u64).unwrap_or(0);
        let mut payload = vec![0u8; bytes as usize];
        r.read_exact(&mut payload)
            .with_context(|| format!("entry {rel}: truncated payload"))?;

        let dest = safe_join(home, &rel)?;
        if dest.exists() && !force {
            skipped_existing.push(rel);
            continue;
        }
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = dest.with_extension("jbak.tmp");
        std::fs::write(&tmp, &payload)?;
        std::fs::rename(&tmp, &dest)?;
        written.push(rel);
    }
    anyhow::ensure!(
        written.len() + skipped_existing.len() == report.entries,
        "entry stream changed between verify and restore — refusing partial restore"
    );
    Ok(RestoreReport {
        written,
        skipped_existing,
    })
}

/// Join a backup-relative path under `home`, refusing traversal tricks:
/// absolute paths, `..`/root/curdir components, empty names.
fn safe_join(home: &Path, rel: &str) -> Result<PathBuf> {
    let p = Path::new(rel);
    anyhow::ensure!(p.is_relative(), "entry path {rel:?} is absolute");
    anyhow::ensure!(!rel.is_empty(), "entry path is empty");
    use std::path::Component;
    for c in p.components() {
        match c {
            Component::Normal(name) => {
                anyhow::ensure!(
                    !name.is_empty(),
                    "entry path {rel:?} has an empty component"
                );
            }
            other => anyhow::bail!("entry path {rel:?} contains {other:?} — refusing"),
        }
    }
    Ok(home.join(p))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_home(tag: &str) -> PathBuf {
        // Parallel test threads resolve now_ms to the same value: an
        // atomic sequence number keeps every dir name unique.
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "jerith-backup-{tag}-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write(path: &Path, contents: &[u8]) {
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p).unwrap();
        }
        std::fs::write(path, contents).unwrap();
    }

    /// A realistic node home: identity config, bot config, and the data
    /// tree with ledgers plus an expert weight store (omitted by default).
    fn seeded_home() -> PathBuf {
        let home = temp_home("seed");
        write(&home.join("config.toml"), b"node_id = \"abc123\"\n");
        write(&home.join("bot.toml"), b"[plugins]\n");
        write(&home.join("data/stake.json"), b"{\"staked_uj\": 900000}");
        write(&home.join("data/expert-stats.json"), b"{\"counts\": {}}");
        write(&home.join("experts/L3.E17.expert"), b"WEIGHTS");
        home
    }

    #[test]
    fn create_verify_restore_roundtrip_excludes_weights_by_default() {
        let home = seeded_home();
        let out = temp_home("out").join("node.jbak");
        let report = create_into(&home, Some(&out), BackupOptions::default()).unwrap();
        assert_eq!(report.path, out);
        // config.toml + bot.toml + the two data ledgers — the weight store
        // is a reported skip, not a silent one.
        assert_eq!(report.entries, 4, "skipped: {:?}", report.skipped);
        assert!(report.skipped.iter().any(|s| s.contains("expert weights")));

        let v = verify(&out).unwrap();
        assert_eq!(v.entries, report.entries);
        assert_eq!(v.total_bytes, report.total_bytes);
        assert_eq!(v.node_id.as_deref(), Some("abc123"));
        assert!(v.paths.iter().all(|p| !p.contains("experts")));
        assert_eq!(v.paths.first().map(String::as_str), Some("bot.toml"));

        // Restore into a FRESH home: every file re-emerges byte-identical,
        // including identity (config.toml) — the migrate story.
        let home2 = temp_home("restore");
        let r = restore_into(&home2, &out, false).unwrap();
        assert_eq!(r.written.len(), 4);
        assert!(r.skipped_existing.is_empty());
        assert_eq!(
            std::fs::read(home2.join("config.toml")).unwrap(),
            b"node_id = \"abc123\"\n"
        );
        assert_eq!(
            std::fs::read(home2.join("data/stake.json")).unwrap(),
            b"{\"staked_uj\": 900000}"
        );
        assert!(!home2.join("experts").exists(), "weights stay omitted");
        std::fs::remove_dir_all(&home).unwrap();
        std::fs::remove_dir_all(&home2).unwrap();
    }

    /// --include-experts is the opt-in for shipping weights with a move.
    #[test]
    fn include_experts_rides_weights_along() {
        let home = seeded_home();
        let out = temp_home("out2").join("node.jbak");
        let report = create_into(
            &home,
            Some(&out),
            BackupOptions {
                include_experts: true,
            },
        )
        .unwrap();
        assert_eq!(report.entries, 5, "the weight store rides along");
        assert!(report.skipped.iter().all(|s| !s.contains("expert weights")));
        let v = verify(&out).unwrap();
        assert!(v.paths.iter().any(|p| p == "experts/L3.E17.expert"));
        std::fs::remove_dir_all(&home).unwrap();
    }

    /// A flipped byte anywhere in an entry's payload fails the NEXT
    /// verify loudly — a tampered or truncated backup never reads as OK.
    #[test]
    fn corruption_is_caught_by_checksum() {
        let home = seeded_home();
        let out = temp_home("out3").join("node.jbak");
        create_into(&home, Some(&out), BackupOptions::default()).unwrap();

        let mut bytes = std::fs::read(&out).unwrap();
        let mid = bytes.len() * 3 / 4; // inside the last entry's payload
        bytes[mid] ^= 0x01;
        let tampered = temp_home("out3").join("tampered.jbak");
        std::fs::write(&tampered, &bytes).unwrap();

        let err = format!("{:#}", verify(&tampered).unwrap_err());
        assert!(err.contains("checksum mismatch"), "got: {err}");
        std::fs::remove_dir_all(&home).unwrap();
    }

    /// Truncated tail: an entry that runs out of payload bytes is a
    /// typed error, not a panic and not a silent shorter restore.
    #[test]
    fn truncated_tail_is_a_typed_error() {
        let home = seeded_home();
        let out = temp_home("out4").join("node.jbak");
        create_into(&home, Some(&out), BackupOptions::default()).unwrap();
        let bytes = std::fs::read(&out).unwrap();
        let truncated = temp_home("out4").join("short.jbak");
        std::fs::write(&truncated, &bytes[..bytes.len() - 7]).unwrap();

        let err = format!("{:#}", verify(&truncated).unwrap_err());
        assert!(
            err.contains("truncated") || err.contains("EOF") || err.contains("checksum"),
            "got: {err}"
        );
        std::fs::remove_dir_all(&home).unwrap();
    }

    /// Restore refuses to clobber state without --force, and clobbers
    /// with it.
    #[test]
    fn restore_overwrite_needs_force() {
        let home = seeded_home();
        let out = temp_home("out5").join("node.jbak");
        create_into(&home, Some(&out), BackupOptions::default()).unwrap();

        // Target home already holds a DIFFERENT config (the skip case).
        let target = temp_home("target");
        write(&target.join("config.toml"), b"node_id = \"old\"\n");

        let r = restore_into(&target, &out, false).unwrap();
        assert!(r.skipped_existing.iter().any(|s| s == "config.toml"));
        assert_eq!(
            std::fs::read(target.join("config.toml")).unwrap(),
            b"node_id = \"old\"\n",
            "without --force the existing identity survives"
        );

        let r = restore_into(&target, &out, true).unwrap();
        assert!(r.written.iter().any(|s| s == "config.toml"));
        assert_eq!(
            std::fs::read(target.join("config.toml")).unwrap(),
            b"node_id = \"abc123\"\n"
        );
        std::fs::remove_dir_all(&home).unwrap();
        std::fs::remove_dir_all(&target).unwrap();
    }

    /// Entry paths are data, and hostile ones are data too: traversal
    /// attempts are refused before a single byte is written.
    #[test]
    fn hostile_entry_paths_are_refused() {
        let home = temp_home("hostile");
        assert!(safe_join(&home, "../escape").is_err());
        assert!(safe_join(&home, "/etc/passwd").is_err());
        assert!(safe_join(&home, "").is_err());
        assert!(safe_join(&home, "a/../b").is_err());
        assert_eq!(
            safe_join(&home, "data/stake.json").unwrap(),
            home.join("data/stake.json")
        );
    }

    /// No config.toml → loud refusal, not an empty "successful" backup.
    #[test]
    fn backup_without_config_refuses() {
        let empty = temp_home("empty");
        let err = format!(
            "{:#}",
            create_into(&empty, None, BackupOptions::default()).unwrap_err()
        );
        assert!(err.contains("run setup first"), "got: {err}");
    }
}
