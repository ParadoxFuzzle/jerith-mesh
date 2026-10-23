//! `jerith_quarantine_plugin` — the mesh house's download-quarantine
//! lane (the plugin half; the bot's monitor/briefing lanes read the
//! SAME verdict log, a documented cross-file contract — the state
//! path is `lib.rs`'s `state_path()`, mirrored in `monitor.rs`).
//!
//! Born of the 2026-10-23 hand-run: a CurseForge-style modpack arrived
//! by DM carrying an obfuscated jar-class dropper inside
//! `overrides/mods` — strings extracted, verdict typed, hashed,
//! refused. That inspection routine, standing:
//!
//! - **real archive reading** — a minimal ZIP central-directory
//!   reader (stored + raw-deflate entries, per-entry inflate cap —
//!   `miniz_oxide`, pure Rust, no provider ambiguity for the
//!   workspace's rustls/ring pin), so jar inspection is byte truth,
//!   not extension rumor;
//! - **the conviction marks** — obfuscator fingerprints, process
//!   spawn, network bootstrap, file plants: the constant-pool
//!   identifiers a dropper cannot hide inside an inflated class,
//!   however much its own literals are frizzled;
//! - **typed verdicts, honest rules**: `malicious` needs a
//!   dropper-shaped COMBINATION or a native executable smuggled into
//!   the archive; `suspicious` is a lone strong mark (obfuscated
//!   class, bare `ProcessBuilder`) that hasn't combo'd; `benign` =
//!   nothing strong. A network-only jar (update checkers) stays
//!   quiet — the lane never cries wolf;
//! - **a JSONL verdict log** (`~/.jerith-mesh/quarantine/
//!   verdicts.jsonl`, `$JERITH_QUARANTINE_STATE` override) —
//!   appends only, tolerant reads, capped; the ONE thing the bot's
//!   monitor check and the daily briefing read;
//! - **the watcher** — a poll of the quarantine dir (a still-being-
//!   written file is left for the next tick, never scanned half-full)
//!   that appends new verdicts; deletions are honored by the READER
//!   side (monitor re-checks file presence), never assumed here.
//!
//! Discipline like the pet and house: `lib.rs` is pure between the
//! edges (clock, dir, state path all injected), `main.rs` is wire
//! glue over the SDK's signed `/rpc` contract.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use sha2::{Digest, Sha256};

/// The scanner never reads forever: a file above this is reported
/// unscanned ("too large") rather than read whole — a cap-lock, not a
/// squeeze (the per-request-budget discipline, in library form).
pub const MAX_FILE_BYTES: u64 = 512 * 1024 * 1024;

/// Per-entry inflate cap: one archive member inflating past this is
/// recorded by name, never decoded (zip-bomb guard; 32 MiB is many
/// multiples of any real class or texture). The entry stays in the
/// walk with a typed `note`, and the archive verdict keeps the OTHER
/// entries' evidence.
pub const MAX_ENTRY_INFLATED: usize = 32 * 1024 * 1024;

/// A file younger than this is still being written — the watcher
/// leaves it for the next tick instead of scanning a half-file.
pub const WRITE_SETTLE_MS: u64 = 10_000;

/// How many verdict rows the JSONL read keeps (newest kept, oldest
/// dropped — a log, not a ledger).
pub const STATE_CAP: usize = 500;

/// Watch-loop cadence, seconds (main.rs's poll loop; overridable per
/// test through the loop function's own parameter).
pub const WATCH_INTERVAL_SECS: u64 = 30;

// ---------------------------------------------------------------------------
// Verdict + findings
// ---------------------------------------------------------------------------

/// The three-way verdict — the shapes the 2026-10-23 hand-run produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Benign,
    Suspicious,
    Malicious,
}

impl Verdict {
    pub fn word(self) -> &'static str {
        match self {
            Verdict::Benign => "benign",
            Verdict::Suspicious => "suspicious",
            Verdict::Malicious => "malicious",
        }
    }
}

/// One piece of evidence: the kind names the rule, the detail quotes
/// the marker or the entry name. Never a guess — each finding cites
/// what the bytes actually contained.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Finding {
    pub kind: String,
    pub detail: String,
}

fn finding(kind: &str, detail: impl Into<String>) -> Finding {
    Finding {
        kind: kind.to_string(),
        detail: detail.into(),
    }
}

/// One completed scan — the JSONL row + the card the skills return.
/// Deserialize rides along because the log is read back (by this crate
/// AND by the bot's monitor/briefing lanes).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanReport {
    pub path: String,
    pub size_bytes: u64,
    /// SHA-256 of the scanned file — the identification handle for
    /// comparing the sample against every other recipient (the reason
    /// the hand-run hashed the dropper before discarding it).
    pub sha256: String,
    /// Coarse type: `archive` (zip-family incl. jar/whl),
    /// `native-exe` (MZ/ELF/Mach-O at offset 0), `other`.
    pub kind: String,
    pub verdict: Verdict,
    pub findings: Vec<Finding>,
    pub scanned_at_ms: u64,
    /// Honest degradation, never a guess: `too large`, `unreadable`,
    /// `unreadable zip` … `None` when the scan completed clean.
    pub note: Option<String>,
}

// ---------------------------------------------------------------------------
// Marker tables — the constant-pool strings a dropper cannot hide
// ---------------------------------------------------------------------------

/// Obfuscator fingerprints (the 2026-10-23 sample was ZKM-packed).
pub const OBFUSCATOR_MARKS: &[&'static str] =
    &["ZKM", "Allatori", "Karakuri", "Stringer", "yGuard"];

/// Process spawn — `Runtime`/`ProcessBuilder` identifiers travel WHOLE
/// inside a class's constant pool, so an inflated class exposes them
/// as contiguous strings even when its own literals are noise.
pub const SPAWN_MARKS: &[&'static str] = &[
    "java/lang/ProcessBuilder",
    "java/lang/Runtime",
    "getRuntime",
];

/// Network bootstrap (`Socket`, `HttpURLConnection`, raw `java/net/URL`).
pub const NET_MARKS: &[&'static str] = &[
    "java/net/Socket",
    "java/net/HttpURLConnection",
    "java/net/URL",
];

/// File-plant / persistence — `setExecutable` and `FileOutputStream`
/// are the plant-and-stage half of every Java dropper chain.
pub const PLANT_MARKS: &[&'static str] = &["setExecutable", "java/io/FileOutputStream"];

/// Native-executable magic at file/entry offset 0. Java classes carry
/// the CAFEBABE magic too, but a scanned ENTRY keeps its own magic
/// check separate (see `entry_native`): a class is not a native exec.
pub fn native_magic(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"MZ") {
        Some("PE/MZ")
    } else if bytes.starts_with(b"\x7fELF") {
        Some("ELF")
    } else if bytes.starts_with(&[0xce, 0xfa, 0xed, 0xfe])
        || bytes.starts_with(&[0xcf, 0xfa, 0xed, 0xfe])
    {
        Some("Mach-O")
    } else {
        None
    }
}

/// Printable ASCII runs (the `strings`-equivalent): a dropper's Java
/// identifiers live here, contiguous, even after its own literals are
/// obfuscated to unreadable noise.
pub fn runs_of_printable(bytes: &[u8], min: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut start: Option<usize> = None;
    for (i, b) in bytes.iter().enumerate() {
        if (0x20..0x7f).contains(b) {
            if start.is_none() {
                start = Some(i);
            }
        } else if let Some(s) = start.take() {
            if i - s >= min {
                out.push(String::from_utf8_lossy(&bytes[s..i]).into_owned());
            }
        }
    }
    if let Some(s) = start {
        if bytes.len() - s >= min {
            out.push(String::from_utf8_lossy(&bytes[s..]).into_owned());
        }
    }
    out
}

/// Which of `marks` appear anywhere in the printable runs of `bytes`.
/// Substring containment across run bodies: `java/lang/Runtime` lives
/// inside constant-pool strings that run on past the identifier.
pub fn contains_marks(content: &[u8], marks: &[&'static str]) -> Vec<&'static str> {
    let mut hits: Vec<&'static str> = Vec::new();
    // min 3: obfuscator fingerprints (“ZKM”) are short tokens, longer
    // than a stray two-letter spill but shorter than API identifiers.
    for run in runs_of_printable(content, 3) {
        for m in marks {
            if run.contains(m) && !hits.contains(&m) {
                hits.push(m);
            }
        }
    }
    hits
}

// ---------------------------------------------------------------------------
// Verdict rules — the typed rule table the tests pin
// ---------------------------------------------------------------------------

/// The marks one scan collected (each list's members are dedup'd).
pub struct Marks {
    pub obf: Vec<&'static str>,
    pub spawn: Vec<&'static str>,
    pub net: Vec<&'static str>,
    pub plant: Vec<&'static str>,
}

impl Marks {
    /// The one place the verdict logic lives, so the tests pin logic,
    /// not scattered conditionals:
    /// - **malicious**: a native executable smuggled into the archive,
    ///   OR a dropper combo — obfuscation with spawn or plant, spawn
    ///   with plant, network with plant (the downloader triple);
    /// - **suspicious**: a lone strong mark (`obf`, `spawn`, `plant`)
    ///   that didn't combo — dead giveaways a legitimate mod has no
    ///   reason to carry ALONE;
    /// - **benign**: nothing strong. Network alone ≠ strong: update
    ///   checkers are the norm, and the lane never cries wolf.
    pub fn verdict(&self, native_in_archive: Option<&str>) -> (Verdict, Vec<Finding>) {
        let mut f = Vec::new();
        for (kind, hits) in [
            ("obfuscated-class", &self.obf),
            ("process-spawn", &self.spawn),
            ("network-bootstrap", &self.net),
            ("file-plant", &self.plant),
        ] {
            for h in hits {
                f.push(finding(kind, *h));
            }
        }
        if let Some(magic) = native_in_archive {
            f.insert(
                0,
                finding(
                    "native-in-archive",
                    format!("entry carries {magic} magic (a native executable smuggled inside)"),
                ),
            );
        }
        let dropper_combo = (!self.spawn.is_empty() && !self.plant.is_empty())
            || (!self.obf.is_empty() && !self.spawn.is_empty())
            || (!self.obf.is_empty() && !self.plant.is_empty())
            || (!self.net.is_empty() && !self.plant.is_empty());
        let verdict = if native_in_archive.is_some() || dropper_combo {
            Verdict::Malicious
        } else if !self.obf.is_empty() || !self.spawn.is_empty() || !self.plant.is_empty() {
            Verdict::Suspicious
        } else {
            Verdict::Benign
        };
        (verdict, f)
    }
}

fn dedup(xs: &[&'static str]) -> Vec<&'static str> {
    let mut out: Vec<&'static str> = Vec::new();
    for x in xs {
        if !out.contains(x) {
            out.push(x);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// The minimal ZIP reader (central directory; stored + raw-deflate)
// ---------------------------------------------------------------------------

/// One decoded archive member.
struct ZipEntry {
    name: String,
    content: Vec<u8>,
    /// The entry content's ORIGINAL magic from `entry_native` (a class
    /// file's CAFEBABE is not a native executable; MZ/ELF/Mach-O is).
    entry_native: Option<&'static str>,
    /// Honest per-entry degradation ("payload truncated",
    /// "deflate decode blocked", …), None when decoded clean.
    note: Option<String>,
}

fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}
fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// Read every entry by walking the ZIP central directory. Errors are
/// honest strings ("unreadable zip …") that become the report's note.
/// Nothing here decodes beyond `MAX_ENTRY_INFLATED`.
fn read_zip(data: &[u8]) -> Result<Vec<ZipEntry>, String> {
    // 1. Find the end-of-central-directory record: scan the tail
    //    (65_557 is the spec's whole comment allowance + the record).
    let eocd = {
        let tail_start = data.len().saturating_sub(65_557 + 22);
        let mut found = None;
        let mut i = data.len();
        while i > tail_start + 22 && i >= 22 {
            if u32_at(data, i - 22) == 0x0605_4b50 {
                found = Some(i - 22);
                break;
            }
            i -= 1;
        }
        found.ok_or_else(|| "no end-of-central-directory record found".to_string())?
    };
    let count = u16_at(data, eocd + 10) as usize;
    let mut offset = u32_at(data, eocd + 16) as usize;

    // 2. Walk each central-directory record (0x02014b50), jumping to
    //    the LOCAL header for the payload offset.
    let mut out = Vec::new();
    for n in 0..count {
        if offset + 46 > data.len() || u32_at(data, offset) != 0x0201_4b50 {
            return Err(format!("central directory corrupted at entry {n}"));
        }
        let method = u16_at(data, offset + 10);
        let csize = u32_at(data, offset + 20) as usize;
        let name_len = u16_at(data, offset + 28) as usize;
        let extra_len = u16_at(data, offset + 30) as usize;
        let comment_len = u16_at(data, offset + 32) as usize;
        let lho = u32_at(data, offset + 42) as usize;
        let name_start = offset + 46;
        if name_start + name_len > data.len() {
            return Err("central directory truncated mid-name".into());
        }
        let name = String::from_utf8_lossy(&data[name_start..name_start + name_len]).into_owned();
        offset = name_start + name_len + extra_len + comment_len;

        // 3. The local header repeats the payload geometry (and is the
        //    only place its inflight extra-field length is known).
        let mut bad = |why: &str| {
            out.push(ZipEntry {
                name: name.clone(),
                content: Vec::new(),
                entry_native: None,
                note: Some(why.to_string()),
            });
        };
        if lho + 30 > data.len() || u32_at(data, lho) != 0x0403_4b50 {
            bad("local header missing");
            continue;
        }
        let lh_name_len = u16_at(data, lho + 26) as usize;
        let lh_extra_len = u16_at(data, lho + 28) as usize;
        let payload_at = lho + 30 + lh_name_len + lh_extra_len;
        let Some(payload) = data
            .get(payload_at..payload_at.saturating_add(csize))
            .filter(|p| p.len() == csize)
        else {
            bad("payload truncated");
            continue;
        };
        let content = match method {
            0 => payload.to_vec(),
            8 => match miniz_oxide::inflate::decompress_to_vec_with_limit(
                payload,
                MAX_ENTRY_INFLATED,
            ) {
                Ok(c) => c,
                Err(_) => {
                    bad("deflate decode blocked (corrupt or past the per-entry cap)");
                    continue;
                }
            },
            m => {
                bad(&format!(
                    "entry method {m} unsupported (not stored/deflate)"
                ));
                continue;
            }
        };
        out.push(ZipEntry {
            entry_native: entry_native(&name, &content),
            name,
            content,
            note: None,
        });
    }
    Ok(out)
}

/// A class file (0xCAFEBABE at offset 0) is NOT a native executable —
/// that's javac's magic. Everything else `native_magic` knows is.
fn entry_native(_name: &str, content: &[u8]) -> Option<&'static str> {
    if content.starts_with(&[0xca, 0xfe, 0xba, 0xbe]) {
        return None;
    }
    native_magic(content)
}

/// ZIP magic at offset 0 (jar, zip, whl — all the pack formats).
fn is_zip(bytes: &[u8]) -> bool {
    bytes.starts_with(b"PK\x03\x04") || bytes.starts_with(b"PK\x05\x06")
}

fn hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------------------
// The scanner
// ---------------------------------------------------------------------------

/// Scan ONE file from disk. Every failure becomes an honest note on a
/// completed report — the lane never loses a file to silence.
pub fn scan_file(path: &Path, now_ms: u64) -> ScanReport {
    let mut r = ScanReport {
        path: path.display().to_string(),
        size_bytes: 0,
        sha256: String::new(),
        kind: "other".into(),
        verdict: Verdict::Benign,
        findings: Vec::new(),
        scanned_at_ms: now_ms,
        note: None,
    };
    let Ok(meta) = std::fs::metadata(path) else {
        r.note = Some("unreadable (no metadata)".into());
        return r;
    };
    let size = meta.len();
    r.size_bytes = size;
    if size > MAX_FILE_BYTES {
        r.note = Some(format!(
            "too large ({size} bytes > {MAX_FILE_BYTES}) — not scanned"
        ));
        return r;
    }
    let Ok(bytes) = std::fs::read(path) else {
        r.note = Some("unreadable (open failed)".into());
        return r;
    };
    r.sha256 = hex(&Sha256::digest(&bytes));
    r.kind = if is_zip(&bytes) {
        "archive".to_string()
    } else if native_magic(&bytes).is_some() {
        "native-exe".to_string()
    } else {
        "other".to_string()
    };

    // A bare native executable at file offset 0: verdict right there.
    if r.kind == "native-exe" {
        let (v, f) = Marks {
            obf: vec![],
            spawn: vec![],
            net: vec![],
            plant: vec![],
        }
        .verdict(Some(native_magic(&bytes).unwrap_or("?")));
        r.verdict = v;
        r.findings = f;
        return r;
    }
    if r.kind == "archive" {
        match read_zip(&bytes) {
            Ok(entries) => {
                let mut obf = Vec::new();
                let mut spawn = Vec::new();
                let mut net = Vec::new();
                let mut plant = Vec::new();
                let mut native: Option<&'static str> = None;
                let mut broken: Vec<String> = Vec::new();
                for e in &entries {
                    if let Some(m) = e.entry_native {
                        native.get_or_insert(m);
                    }
                    obf.extend(contains_marks(&e.content, OBFUSCATOR_MARKS));
                    spawn.extend(contains_marks(&e.content, SPAWN_MARKS));
                    net.extend(contains_marks(&e.content, NET_MARKS));
                    plant.extend(contains_marks(&e.content, PLANT_MARKS));
                    if let Some(note) = &e.note {
                        broken.push(format!("`{}`: {note}", e.name));
                    }
                }
                let (v, f) = Marks {
                    obf: dedup(&obf),
                    spawn: dedup(&spawn),
                    net: dedup(&net),
                    plant: dedup(&plant),
                }
                .verdict(native);
                r.verdict = v;
                r.findings = f;
                if !broken.is_empty() {
                    broken.truncate(3);
                    r.note = Some(format!("entries unreadable: {}", broken.join("; ")));
                }
            }
            Err(why) => r.note = Some(format!("unreadable zip ({why})")),
        }
    }
    r
}

// ---------------------------------------------------------------------------
// State: the JSONL verdict log (the bot's ONE read surface)
// ---------------------------------------------------------------------------

/// The verdict log. `$JERITH_QUARANTINE_STATE` wins; else
/// `$HOME/.jerith-mesh/quarantine/verdicts.jsonl` (the same rebase
/// conventions as every other lane's state file).
pub fn state_path() -> PathBuf {
    if let Ok(p) = std::env::var("JERITH_QUARANTINE_STATE") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home)
        .join(".jerith-mesh")
        .join("quarantine")
        .join("verdicts.jsonl")
}

/// Append one row (newest last); creates parent dirs.
pub fn append_report(path: &Path, r: &ScanReport) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let line = serde_json::to_string(r).map_err(std::io::Error::other)?;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    f.write_all(line.as_bytes())?;
    f.write_all(b"\n")?;
    Ok(())
}

/// Load + parse the log, newest kept, up to `cap` rows. Bad lines are
/// skipped, never fatal — an append-only file may be half-written by
/// a crash, and a log must not wedge its readers.
pub fn load_reports(path: &Path, cap: usize) -> Vec<ScanReport> {
    let Ok(raw) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut out: Vec<ScanReport> = raw
        .lines()
        .filter_map(|l| serde_json::from_str::<ScanReport>(l).ok())
        .collect();
    if out.len() > cap {
        out = out.split_off(out.len() - cap);
    }
    out
}

/// `(benign, suspicious, malicious)` over any report slice — the
/// monitor/briefing head line and the skills' `counts` block.
pub fn counts(reports: &[ScanReport]) -> (u64, u64, u64) {
    let mut c = (0u64, 0u64, 0u64);
    for r in reports {
        match r.verdict {
            Verdict::Benign => c.0 += 1,
            Verdict::Suspicious => c.1 += 1,
            Verdict::Malicious => c.2 += 1,
        }
    }
    c
}

// ---------------------------------------------------------------------------
// The watcher
// ---------------------------------------------------------------------------

/// One watch tick over `dir`. `known` keys already-scanned files
/// (full path + size + stable-mtime); returned is the fresh reports —
/// the caller owns appending them. Files younger than
/// `WRITE_SETTLE_MS` blink out of this tick (still being written).
/// Keys whose file no longer exists drop from `known` — a re-download
/// of the same filename re-scans.
pub fn watch_once(
    dir: &Path,
    now_ms: u64,
    known: &mut HashMap<(String, u64, u64), ScanReport>,
) -> Vec<ScanReport> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new(); // the dir vanished for this tick: quiet, honest
    };
    let mut fresh: Vec<ScanReport> = Vec::new();
    for e in rd.flatten() {
        let Ok(meta) = e.metadata() else { continue };
        if !meta.is_file() {
            continue;
        }
        let size = meta.len();
        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        if now_ms.saturating_sub(mtime) < WRITE_SETTLE_MS {
            continue; // still being written — next tick
        }
        let name = e.file_name().to_string_lossy().into_owned();
        let key = (dir.join(&name).display().to_string(), size, mtime);
        if known.contains_key(&key) {
            continue;
        }
        let report = scan_file(&dir.join(&name), now_ms);
        known.insert(key, report.clone());
        fresh.push(report);
    }
    known.retain(|k, _| Path::new(&k.0).exists());
    fresh
}

// ---------------------------------------------------------------------------
// Tests — the conviction discipline in miniature: real zip fixtures,
// real markers, the rule table pinned in both directions.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn now() -> u64 {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    // -- zip fixture writer (test-only): local headers + central dir --
    fn le16(v: u16, out: &mut Vec<u8>) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    fn le32(v: u32, out: &mut Vec<u8>) {
        out.extend_from_slice(&v.to_le_bytes());
    }

    /// Build a REAL zip: `entries` = (name, stored?, bytes); stored
    /// false = method 8 raw deflate (compressed via miniz here).
    fn build_zip(entries: &[(&str, bool, &[u8])]) -> Vec<u8> {
        let mut body = Vec::new();
        let mut cd = Vec::new();
        for (name, stored, data) in entries {
            let comp: Vec<u8> = if *stored {
                data.to_vec()
            } else {
                miniz_oxide::deflate::compress_to_vec(data, 8)
            };
            let method: u16 = if *stored { 0 } else { 8 };
            let lho = body.len() as u32;
            le32(0x0403_4b50, &mut body); // local header
            le16(20, &mut body); // needed version
            le16(0, &mut body); // flags
            le16(method, &mut body);
            le16(0, &mut body); // time
            le16(0, &mut body); // date
            le32(0x2144_89f4, &mut body); // crc (our reader doesn't check)
            le32(comp.len() as u32, &mut body);
            le32(data.len() as u32, &mut body);
            le16(name.len() as u16, &mut body);
            le16(0, &mut body); // extra
            body.extend_from_slice(name.as_bytes());
            body.extend_from_slice(&comp);

            le32(0x0201_4b50, &mut cd); // central record
            le16(20, &mut cd);
            le16(20, &mut cd);
            le16(0, &mut cd);
            le16(method, &mut cd);
            le16(0, &mut cd);
            le16(0, &mut cd);
            le32(0x2144_89f4, &mut cd);
            le32(comp.len() as u32, &mut cd);
            le32(data.len() as u32, &mut cd);
            le16(name.len() as u16, &mut cd);
            le16(0, &mut cd); // extra
            le16(0, &mut cd); // comment
            le16(0, &mut cd); // disk
            le16(0, &mut cd); // int attrs
            le32(0, &mut cd); // ext attrs
            le32(lho, &mut cd);
            cd.extend_from_slice(name.as_bytes());
        }
        let cd_off = body.len() as u32;
        body.extend_from_slice(&cd);
        let mut tail = Vec::new();
        le32(0x0605_4b50, &mut tail); // EOCD
        le16(0, &mut tail);
        le16(0, &mut tail);
        le16(entries.len() as u16, &mut tail);
        le16(entries.len() as u16, &mut tail);
        le32(cd.len() as u32, &mut tail);
        le32(cd_off, &mut tail);
        le16(0, &mut tail);
        body.extend_from_slice(&tail);
        body
    }

    /// Class constant-pool-shaped content: the identifiers live whole
    /// (the point — obfuscators frizzle literals, never the pool).
    fn class_with(mut marks: Vec<&str>) -> Vec<u8> {
        let mut b = vec![0xca, 0xfe, 0xba, 0xbe];
        marks.push("com/sneaky/Main");
        for s in marks {
            b.extend_from_slice(s.as_bytes());
            b.push(0x00);
        }
        b
    }

    /// The 2026-10-23 conviction shape in one class: obfuscation on
    /// the outside, spawn + download + plant inside the pool.
    fn dropper_class() -> Vec<u8> {
        class_with(vec![
            "ZKM",
            "java/lang/ProcessBuilder",
            "java/net/HttpURLConnection",
            "setExecutable",
        ])
    }

    /// Write a fixture to a real temp file (the scan edge is the disk),
    /// scan it, clean up, return the report.
    fn scan_fixture(bytes: &[u8], name: &str) -> ScanReport {
        let dir = std::env::temp_dir().join(format!("q-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(format!("{}-{}", now(), name));
        std::fs::write(&p, bytes).unwrap();
        let r = scan_file(&p, now());
        let _ = std::fs::remove_file(&p);
        r
    }

    // -- the reader itself (a real zip, both storage methods) --------

    #[test]
    fn zip_reader_walks_both_storage_methods() {
        let zip = build_zip(&[
            ("pack.mcmeta", true, b"{\"pack\":{}}" as &[u8]),
            ("com/example/Mod.class", false, &class_with(vec![])),
        ]);
        let entries = read_zip(&zip).expect("reads our own fixture");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "pack.mcmeta");
        assert_eq!(entries[0].content, b"{\"pack\":{}}");
        assert_eq!(entries[1].name, "com/example/Mod.class");
        assert!(
            entries[1].content.starts_with(&[0xca, 0xfe, 0xba, 0xbe]),
            "deflate round-trips the class bytes"
        );
        // The inflated pool string is findable; the deflated stream
        // alone would NOT carry it (that's why inspection must decode).
        assert_eq!(
            contains_marks(&entries[1].content, &["com/sneaky/Main"]).len(),
            1
        );
    }

    #[test]
    fn a_truncated_zip_is_an_honest_unreadable() {
        let zip = build_zip(&[("a", true, b"x" as &[u8])]);
        let cut = zip.len() / 2;
        let r = scan_fixture(&zip[..cut], "truncated.zip");
        assert!(
            r.note
                .as_deref()
                .unwrap_or_default()
                .starts_with("unreadable zip"),
            "{r:?}"
        );
        assert_eq!(r.verdict, Verdict::Benign, "unreadable ≠ guilty");
    }

    // -- verdict rules: each direction pinned --------------------------

    #[test]
    fn the_bettercraft_shape_is_malicious() {
        let zip = build_zip(&[("com/sneaky/Main.class", false, &dropper_class())]);
        let r = scan_fixture(&zip, "pack-overrides.java");
        assert_eq!(r.kind, "archive");
        assert_eq!(r.verdict, Verdict::Malicious, "{:?}", r.findings);
        let kinds: Vec<&str> = r.findings.iter().map(|f| f.kind.as_str()).collect();
        assert!(kinds.contains(&"obfuscated-class"), "{kinds:?}");
        assert!(kinds.contains(&"process-spawn"), "{kinds:?}");
        assert!(kinds.contains(&"file-plant"), "{kinds:?}");
        assert!(kinds.contains(&"network-bootstrap"), "{kinds:?}");
    }

    #[test]
    fn stored_native_dropper_in_the_pack_is_malicious() {
        // The smuggling channel: a plausible-looking .jar name wrapped
        // around native bytes, planted via overrides/mods.
        let zip = build_zip(&[(
            "serpents-2.0.1.jar",
            true,
            &b"MZ\x90\x00tralala-native-payload"[..],
        )]);
        let r = scan_fixture(&zip, "mods.zip");
        assert_eq!(r.verdict, Verdict::Malicious, "{:?}", r.findings);
        assert_eq!(r.findings[0].kind, "native-in-archive", "{:?}", r.findings);
    }

    #[test]
    fn a_class_is_never_called_native() {
        let zip = build_zip(&[("Mod.class", true, &[0xca, 0xfe, 0xba, 0xbe, 0x00, 0x01])]);
        let r = scan_fixture(&zip, "class-in-zip.java");
        assert!(
            !r.findings.iter().any(|f| f.kind == "native-in-archive"),
            "CAFEBABE is javac's magic, not Mach-O: {r:?}"
        );
    }

    #[test]
    fn bare_pe_file_is_malicious_native_exe() {
        let r = scan_fixture(b"MZ this is not a minecraft pack", "thing.bin");
        assert_eq!(r.kind, "native-exe");
        assert_eq!(r.verdict, Verdict::Malicious);
    }

    #[test]
    fn plain_net_only_update_checker_stays_benign() {
        let zip = build_zip(&[(
            "com/example/Updater.class",
            false,
            &class_with(vec!["java/net/URL"]),
        )]);
        let r = scan_fixture(&zip, "updater.java");
        assert_eq!(r.verdict, Verdict::Benign, "{:?}", r.findings);
    }

    #[test]
    fn lone_strong_marks_land_suspicious_never_malicious() {
        let cases: Vec<(&str, Vec<u8>)> = vec![
            (
                "obfuscated-but-quiet",
                build_zip(&[("x.class", false, &class_with(vec!["ZKM"]))]),
            ),
            (
                "bare-spawn",
                build_zip(&[(
                    "y.class",
                    false,
                    &class_with(vec!["java/lang/ProcessBuilder"]),
                )]),
            ),
            (
                "bare-plant",
                build_zip(&[("z.class", false, &class_with(vec!["setExecutable"]))]),
            ),
        ];
        for (name, zip) in cases {
            let r = scan_fixture(&zip, &format!("{name}.java"));
            assert_eq!(
                r.verdict,
                Verdict::Suspicious,
                "{name}: a lone mark is suspicion, not a verdict death sentence: {r:?}"
            );
        }
    }

    #[test]
    fn network_with_plant_is_the_downloader_triple() {
        let zip = build_zip(&[(
            "d.class",
            false,
            &class_with(vec!["java/net/HttpURLConnection", "setExecutable"]),
        )]);
        let r = scan_fixture(&zip, "downloader.java");
        assert_eq!(r.verdict, Verdict::Malicious, "{:?}", r.findings);
    }

    #[test]
    fn plain_text_stays_other_and_benign() {
        let r = scan_fixture(b"just a plain text file for the wizard", "note.txt");
        assert_eq!(r.kind, "other");
        assert_eq!(r.verdict, Verdict::Benign);
        assert!(r.note.is_none());
    }

    #[test]
    fn every_report_carries_a_stable_hash() {
        let a = scan_fixture(&build_zip(&[("a", true, b"xy" as &[u8])]), "h1.zip");
        let b = scan_fixture(&build_zip(&[("a", true, b"xy" as &[u8])]), "h2.zip");
        assert_eq!(a.sha256, b.sha256);
        assert_eq!(a.sha256.len(), 64);
    }

    #[test]
    fn jsonl_state_round_trips_and_survives_garbage() {
        let p = std::env::temp_dir().join(format!("q-log-{}.jsonl", now()));
        let _ = std::fs::remove_file(&p);
        let a = scan_fixture(&build_zip(&[("a", true, b"xy" as &[u8])]), "log.zip");
        append_report(&p, &a).unwrap();
        // A half-written crash line must not wedge the reader: slip one
        // mid-log the way an interrupted append would leave it.
        let prev = std::fs::read_to_string(&p).unwrap();
        let _ = std::fs::write(&p, format!("{}GARBAGE-LINE\n", prev));
        let mut late = a.clone();
        late.path = "re-scanned".into();
        append_report(&p, &late).unwrap();
        let all = load_reports(&p, STATE_CAP);
        assert_eq!(all.len(), 2, "a garbage line is skipped, not fatal");
        assert_eq!(all[1].path, "re-scanned");
        let (benign, suspicious, malicious) = counts(&all);
        assert_eq!((benign, suspicious, malicious), (2, 0, 0));
        assert_eq!(Verdict::Malicious.word(), "malicious");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn state_cap_keeps_the_newest_tail() {
        let p = std::env::temp_dir().join(format!("q-cap-{}.jsonl", now()));
        let _ = std::fs::remove_file(&p);
        let base = scan_fixture(b"tiny-category", "cap.bin");
        for i in 0..7 {
            let mut r = base.clone();
            r.path = format!("row-{i}");
            append_report(&p, &r).unwrap();
        }
        let rows = load_reports(&p, 5);
        assert_eq!(rows.len(), 5);
        assert_eq!(rows.iter().next().unwrap().path, "row-2");
        assert_eq!(rows.iter().last().unwrap().path, "row-6");
        let _ = std::fs::remove_file(&p);
    }

    // -- watcher --------------------------------------------------------

    /// Settle a file's mtime long ago so the watcher's settle-guard
    /// doesn't skip it mid-test (std's File::set_modified).
    fn settle_mtime(p: &Path) {
        let f = std::fs::File::options().write(true).open(p).unwrap();
        f.set_modified(SystemTime::UNIX_EPOCH).unwrap();
    }

    #[test]
    fn watcher_batches_new_files_and_knows_them_after() {
        let dir = std::env::temp_dir().join(format!("q-watch-{}", now()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("pack.zip");
        std::fs::write(&f, build_zip(&[("a", true, b"z" as &[u8])])).unwrap();
        settle_mtime(&f);

        let mut known = HashMap::<(String, u64, u64), ScanReport>::new();
        let fresh = watch_once(&dir, now(), &mut known);
        assert_eq!(fresh.len(), 1, "the settled file is scanned");
        assert_eq!(fresh[0].kind, "archive");
        assert!(
            watch_once(&dir, now(), &mut known).is_empty(),
            "the second tick already knows it"
        );
        std::fs::remove_file(&f).unwrap();
        // The next tick forgets what is gone from disk.
        let _ = watch_once(&dir, now(), &mut known);
        assert!(known.is_empty(), "deletion drops the scan state");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn still_being_written_files_wait_for_the_next_tick() {
        let dir = std::env::temp_dir().join(format!("q-hot-{}", now()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("in-progress.zip");
        std::fs::write(&f, build_zip(&[("a", true, b"z" as &[u8])])).unwrap();
        // mtime stays NOW (unsettled) — the settle-guard holds.
        let mut known = HashMap::new();
        assert!(
            watch_once(&dir, now(), &mut known).is_empty(),
            "a hot file is never scanned half-full"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // -- the marks primitives -------------------------------------------

    #[test]
    fn printable_runs_and_substring_marks() {
        let runs = runs_of_printable(b"com/sneaky/Main\x0010.0.0.1\x00", 4);
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0], "com/sneaky/Main");
        assert!(
            contains_marks(
                b"\x00java/lang/ProcessBuilder\x00",
                &["java/lang/ProcessBuilder"]
            )
            .len()
                == 1
        );
        assert!(
            contains_marks(b"\x00short\x00", &["java/lang/ProcessBuilder"]).is_empty(),
            "runs shorter than the mark carry nothing"
        );
    }
}
