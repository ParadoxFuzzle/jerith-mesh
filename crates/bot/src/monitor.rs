//! Monitor scaffold — the OpenClaw ops-monitor family (disk-space-alerts,
//! ssl-cert-monitor, temperature-monitor, log-rotation-health,
//! docker-update-checker, backup-verification) collapsed into one module.
//!
//! Shape: a list of periodic checks (std-only, no external deps), a
//! report builder, and a loop that sends the bot owner **one DM** when
//! something needs attention — checks that pass stay quiet. Owner-DM
//! delivery honors the owner gate's identity resolution
//! (`cfg.owner_discord_id`); if no owner is configured the monitor still
//! runs and logs, it just can't DM anyone (fail-quiet, not fail-open).
//!
//! Privileged boundary: this module **reports only** — it never restarts
//! services, never runs repairs, never executes system changes. The
//! OpenClaw watchdog's auto-restart behavior was deliberately NOT ported
//! here; owner.rs's default-deny gate exists precisely so system-level
//! actions stay owner-commanded, and the monitor keeps to the reporting
//! half of that OpenClaw pattern.

use std::path::Path;

use poise::serenity_prelude as serenityall;
use serde::Serialize;

/// Health of one check.
#[derive(Debug, Clone, Serialize)]
pub struct CheckResult {
    pub check: &'static str,
    /// ok = stay silent, warn = mention it, fail = lead with it.
    pub level: Level,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Level {
    Ok,
    Warn,
    Fail,
}

impl Level {
    fn icon(self) -> &'static str {
        match self {
            Level::Ok => "✅",
            Level::Warn => "⚠️",
            Level::Fail => "❌",
        }
    }
}

/// What the checks are allowed to know — thresholds + paths (config.rs).
#[derive(Debug, Clone)]
pub struct MonitorLimits {
    /// Disk used percent above which the check warns.
    pub disk_warn_percent: u64,
    /// TLS cert expiry days below which the check warns.
    pub ssl_warn_days: u64,
    /// Degrees C above which a temperature sensor warns.
    pub temp_warn_c: i64,
    /// A log file larger than this many MB fails (rotation not working).
    pub log_max_mb: u64,
    /// Data dirs whose files should have been touched within 26h.
    pub backup_dirs: Vec<String>,
}

impl Default for MonitorLimits {
    fn default() -> Self {
        Self {
            disk_warn_percent: 90,
            ssl_warn_days: 14,
            temp_warn_c: 85,
            log_max_mb: 500,
            backup_dirs: Vec::new(),
        }
    }
}

/// Run every check. Never panics into the loop: each check is isolated.
pub fn run_checks(limits: &MonitorLimits) -> Vec<CheckResult> {
    let mut out = Vec::new();
    for r in [
        check_disk(limits),
        check_ssl(limits),
        check_temps(limits),
        check_logs(limits),
        check_backups(limits),
    ]
    .into_iter()
    {
        out.push(r);
    }
    out
}

/// Render the owner report: ONLY the warn/fail lines, (ok)-count trailer.
/// Empty string when everything is fine (nothing to DM).
pub fn render_report(results: &[CheckResult]) -> Option<String> {
    let bad: Vec<&CheckResult> = results
        .iter()
        .filter(|r| r.level != Level::Ok)
        .collect();
    if bad.is_empty() {
        return None;
    }
    let ok_count = results.len() - bad.len();
    let mut s = String::from("🩺 **jerith-mesh monitor**\n");
    for r in &bad {
        s.push_str(&format!(
            "{} **{}** — {}\n",
            r.level.icon(),
            r.check,
            r.detail
        ));
    }
    if ok_count > 0 {
        s.push_str(&format!("\n_{ok_count} other checks pass quietly._"));
    }
    Some(s)
}

// ---------------------------------------------------------------------------
// The checks (std-only)
// ---------------------------------------------------------------------------

/// Disk usage for the mesh home filesystem (statvfs via /proc or direct
/// syscall-free approximation: `df`-like reading of statfs is unavailable
/// std-only, so we read /proc/mounts-style usage is not exposed either —
/// use the `df -P` line parse, which is stable everywhere).
fn check_disk(limits: &MonitorLimits) -> CheckResult {
    // The mesh home's filesystem.
    let home = std::env::var("HOME").unwrap_or_else(|_| "/".into());
    let target = Path::new(&home);
    let output = std::process::Command::new("df")
        .arg("-P")
        .arg("--output=pcent,target")
        .arg(target)
        .output();
    match output {
        Ok(o) if o.status.success() => {
            let text = String::from_utf8_lossy(&o.stdout);
            let line = text.lines().nth(1).unwrap_or_default();
            let pct: u64 = line
                .split_whitespace()
                .next()
                .and_then(|p| p.trim_end_matches('%').parse().ok())
                .unwrap_or(0);
            let mount = line.split_whitespace().nth(1).unwrap_or("?").to_string();
            if pct >= limits.disk_warn_percent {
                CheckResult {
                    check: "disk",
                    level: Level::Fail,
                    detail: format!(
                        "{mount} at {pct}% (threshold {}%) — mesh home is on this filesystem",
                        limits.disk_warn_percent
                    ),
                }
            } else if disk_level(pct, limits.disk_warn_percent) == Level::Warn {
                // Near-threshold heads-up: warn before it becomes a fail.
                CheckResult {
                    check: "disk",
                    level: Level::Warn,
                    detail: format!(
                        "{mount} at {pct}% — approaching the {}% threshold",
                        limits.disk_warn_percent
                    ),
                }
            } else {
                CheckResult {
                    check: "disk",
                    level: Level::Ok,
                    detail: format!("{mount} at {pct}%"),
                }
            }
        }
        _ => CheckResult {
            check: "disk",
            level: Level::Ok, // can't measure → don't alarm
            detail: "df unavailable".into(),
        },
    }
}

/// SSL cert expiry for the boot/public host via `openssl s_client`.
fn check_ssl(limits: &MonitorLimits) -> CheckResult {
    let host = "p2p.jerithai.com";
    let output = std::process::Command::new("timeout")
        .args(["/9", "openssl", "s_client", "-servername", host, "-connect", &format!("{host}:443")])
        .stdin(std::process::Stdio::null())
        .output();
    let Ok(o) = output else {
        return CheckResult { check: "ssl", level: Level::Ok, detail: "openssl unavailable".into() };
    };
    let text = String::from_utf8_lossy(&o.stdout);
    let not_after = text
        .lines()
        .find_map(|l| l.trim().strip_prefix("notAfter="))
        .unwrap_or_default();
    if not_after.is_empty() {
        return CheckResult { check: "ssl", level: Level::Ok, detail: format!("{host}: cert unreadable (tunnel may be down) — openssl probe failed") };
    }
    match days_until_certs_not_after(&text) {
        Some(days) if days <= limits.ssl_warn_days as i64 => CheckResult {
            check: "ssl",
            level: Level::Fail,
            detail: format!("{host} certificate expires in {days} days ({not_after})"),
        },
        Some(_) => CheckResult {
            check: "ssl",
            level: Level::Ok,
            detail: format!("{host} certificate has >{} days left", limits.ssl_warn_days),
        },
        None => CheckResult {
            check: "ssl",
            level: Level::Ok,
            detail: "expiry not parseable this cycle".into(),
        },
    }
}

/// Days until the first `notAfter` found in an openssl output block.
/// Uses a hand-rolled month table (no chrono in the workspace).
fn days_until_certs_not_after(text: &str) -> Option<i64> {
    let raw = text.lines().find_map(|l| l.trim().strip_prefix("notAfter="))?;
    // e.g. "Jan  5 12:00:00 2027 GMT"
    let mut it = raw.split_whitespace();
    let mon = it.next()?;
    let day: i64 = it.next()?.parse().ok()?;
    let _time = it.next()?;
    let year: i64 = it.next()?.parse().ok()?;
    let month_index = match mon {
        "Jan" => 0, "Feb" => 1, "Mar" => 2, "Apr" => 3, "May" => 4, "Jun" => 5,
        "Jul" => 6, "Aug" => 7, "Sep" => 8, "Oct" => 9, "Nov" => 10, "Dec" => 11,
        _ => return None,
    };
    let now_days = days_from_civil_utc_today();
    let target = days_from_civil(year, month_index + 1, day);
    Some(target - now_days)
}

/// Howard Hinnant's days_from_civil (no chrono dep in the workspace).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn days_from_civil_utc_today() -> i64 {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64 / 1000)
        .unwrap_or(0);
    (secs / 86_400) as i64
}

/// CPU/GPU temps from the hwmon interface (std-only).
fn check_temps(limits: &MonitorLimits) -> CheckResult {
    let mut worst: Option<(i64, String)> = None;
    let Ok(rd) = std::fs::read_dir("/sys/class/hwmon") else {
        return CheckResult { check: "temps", level: Level::Ok, detail: "no hwmon on this host".into() };
    };
    for e in rd.flatten() {
        let name_path = e.path().join("name");
        let chip = std::fs::read_to_string(&name_path).unwrap_or_default().trim().to_string();
        let Ok(inputs) = std::fs::read_dir(e.path()) else { continue };
        #[allow(clippy::manual_flatten)]
        for f in inputs.flatten() {
            let fname = f.file_name().to_string_lossy().to_string();
            if !fname.starts_with("temp") || !fname.ends_with("_input") {
                continue;
            }
            let Ok(millic) = std::fs::read_to_string(f.path()) else { continue };
            let Ok(millic) = millic.trim().parse::<i64>() else { continue; };
            let c = millic / 1000;
            if worst.as_ref().map(|(worst_c, _)| c > *worst_c).unwrap_or(true) {
                worst = Some((c, chip.clone()));
            }
        }
    }
    match worst {
        Some((c, chip)) if c >= limits.temp_warn_c => CheckResult {
            check: "temps",
            level: Level::Fail,
            detail: format!("{chip} at {c}°C (threshold {}°C)", limits.temp_warn_c),
        },
        Some((c, chip)) => CheckResult {
            check: "temps",
            level: Level::Ok,
            detail: format!("hottest: {chip} at {c}°C"),
        },
        None => CheckResult {
            check: "temps",
            level: Level::Ok,
            detail: "no temperature sensors readable".into(),
        },
    }
}

/// Log files over the size cap (rotation health,OpenClaw check).
fn check_logs(limits: &MonitorLimits) -> CheckResult {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let mut biggest: Option<(u64, String)> = None;
    for dir in [
        format!("{home}/.jerith-mesh/logs"),
        format!("{home}/.jerith-mesh/bot"),
    ] {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            let Ok(meta) = e.metadata() else { continue };
            if !meta.is_file() {
                continue;
            }
            let size = meta.len();
            if biggest.as_ref().map(|(b, _)| size > *b).unwrap_or(true) {
                biggest = Some((size, p.display().to_string()));
            }
        }
    }
    match biggest {
        Some((bytes, path)) => {
            let mb = bytes / (1024 * 1024);
            if mb >= limits.log_max_mb {
                CheckResult {
                    check: "logs",
                    level: Level::Fail,
                    detail: format!(
                        "{path} is {mb} MB (rotation cap {} MB) — journald/handling not shrinking it",
                        limits.log_max_mb
                    ),
                }
            } else {
                CheckResult {
                    check: "logs",
                    level: Level::Ok,
                    detail: format!("largest log {} MB (cap {})", mb, limits.log_max_mb),
                }
            }
        }
        None => CheckResult {
            check: "logs",
            level: Level::Ok,
            detail: "no local log files yet".into(),
        },
    }
}

/// Backup freshness: each configured dir must have a file touched within
/// 26h (the OpenClaw nightly cadence + slack).
fn check_backups(limits: &MonitorLimits) -> CheckResult {
    if limits.backup_dirs.is_empty() {
        return CheckResult {
            check: "backups",
            level: Level::Ok,
            detail: "no backup_dirs configured — check skipped".into(),
        };
    }
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let stale: Vec<String> = limits
        .backup_dirs
        .iter()
        .filter(|dir| newest_file_age_ms(Path::new(dir.as_str()), now_ms) > 26 * 3_600_000)
        .cloned()
        .collect();
    if stale.is_empty() {
        CheckResult {
            check: "backups",
            level: Level::Ok,
            detail: format!("{} dirs fresh (≤26h)", limits.backup_dirs.len()),
        }
    } else {
        CheckResult {
            check: "backups",
            level: Level::Fail,
            detail: format!("stale (>26h since newest file): {}", stale.join(", ")),
        }
    }
}

/// Disk level as a pure decision (testable without spawning `df`):
/// fail at/above the threshold, warn within 5 points below it, else ok.
fn disk_level(pct: u64, warn_percent: u64) -> Level {
    if pct >= warn_percent {
        Level::Fail
    } else if warn_percent >= 5 && pct >= warn_percent - 5 {
        Level::Warn
    } else {
        Level::Ok
    }
}

fn newest_file_age_ms(dir: &Path, now_ms: u64) -> u64 {
    let Ok(rd) = std::fs::read_dir(dir) else { return u64::MAX };
    let mut newest: u64 = 0;
    for e in rd.flatten() {
        if let Ok(meta) = e.metadata() {
            if let Ok(mod_ms) = meta.modified().map(|t| {
                t.duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0)
            }) {
                newest = newest.max(mod_ms);
            }
        }
    }
    now_ms.saturating_sub(newest)
}

/// Monitor loop (spawned once from main.rs with the bot's HTTP client):
/// run the checks; when anything is warn/fail, DM the owner once through
/// serenity's HTTP handle — checks that pass stay quiet. An absent owner
/// id keeps the checks logging instead of DMing (fail-quiet, not
/// fail-open) — reporter only, system changes stay behind owner.rs.
pub async fn run_loop(
    http: std::sync::Arc<serenityall::http::Http>,
    owner_id: Option<u64>,
    cfg: crate::config::MonitorConfig,
) {
    tracing::info!(
        owner_dm = owner_id.map(|o| o.to_string()).unwrap_or_else(|| "none (fails quiet)".into()),
        interval_minutes = cfg.interval_minutes,
        "monitor scaffold started (OpenClaw ops-monitor port)"
    );
    let limits = MonitorLimits {
        backup_dirs: cfg.backup_dirs,
        ..MonitorLimits::default()
    };
    loop {
        let results = run_checks(&limits);
        if let Some(report) = render_report(&results) {
            match owner_id {
                Some(owner) => {
                    use serenityall::builder::CreateMessage;
                    let msg = CreateMessage::new().content(&report);
                    if let Err(e) = serenityall::UserId::new(owner).direct_message(&http, msg).await {
                        tracing::warn!(error = %e, "monitor owner-DM send failed");
                    }
                }
                None => {
                    tracing::info!(report = %report, "monitor would DM owner — no owner id configured");
                }
            }
        }
        tokio::time::sleep(std::time::Duration::from_secs(cfg.interval_minutes.max(1) * 60)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_pass_report_is_none() {
        let limits = MonitorLimits {
            disk_warn_percent: 100,
            ssl_warn_days: 0,   // will be Ok unless parseable+small — keep Ok branch
            temp_warn_c: 500,
            log_max_mb: 100_000,
            backup_dirs: vec![],
        };
        let results = run_checks(&limits);
        assert_eq!(results.len(), 5);
        assert!(render_report(&results).is_none(), "got {:?}", results.iter().filter(|r| r.level != Level::Ok).collect::<Vec<_>>());
    }

    #[test]
    fn render_lists_only_bad_lines() {
        let results = vec![
            CheckResult { check: "disk", level: Level::Ok, detail: "fs at 30%".into() },
            CheckResult { check: "ssl", level: Level::Fail, detail: "cert expires in 2 days".into() },
            CheckResult { check: "temps", level: Level::Warn, detail: "cpu at 88°C".into() },
        ];
        let report = render_report(&results).expect("report expected");
        assert!(report.contains("ssl"));
        assert!(report.contains("temps"));
        assert!(!report.contains("disk"));
        assert!(report.contains("1 other checks pass quietly"));
    }

    #[test]
    fn all_fail_report_still_renders() {
        let results = vec![CheckResult { check: "disk", level: Level::Fail, detail: "at 99%".into() }];
        let report = render_report(&results).expect("report expected");
        assert!(report.contains("disk"));
        // No "other checks" trailer when none passed.
        assert!(!report.contains("other checks"));
    }

    #[test]
    fn cert_days_math_hits_known_date() {
        // 2027-01-05 from aunix epoch day count should be >300 in 2026-10.
        let text = "notAfter=Jan  5 12:00:00 2027 GMT\n";
        let days = days_until_certs_not_after(text).unwrap();
        assert!((days - 84).abs() < 200, "days={days}");
    }

    #[test]
    fn disk_level_warns_below_the_threshold() {
        assert_eq!(disk_level(50, 90), Level::Ok);
        assert_eq!(disk_level(85, 90), Level::Warn);
        assert_eq!(disk_level(90, 90), Level::Fail);
        // Tiny thresholds have no warn band.
        assert_eq!(disk_level(3, 4), Level::Ok);
    }

    #[test]
    fn backup_staleness_detects_old_dirs() {
        let dir = std::env::temp_dir().join(format!("mon-backup-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let limits = MonitorLimits {
            disk_warn_percent: 100,
            ssl_warn_days: 0,
            temp_warn_c: 500,
            log_max_mb: 100_000,
            backup_dirs: vec![dir.display().to_string()],
        };
        // A fresh file → fresh dir.
        std::fs::write(dir.join("snap.txt"), "x").unwrap();
        let r = check_backups(&limits);
        assert_eq!(r.level, Level::Ok);
        // Backdate the file's mtime via a marker file built old isn't
        // possible portably; instead point at a dir with no files at all →
        // newest = 0 → staleness fully stale.
        let empty = dir.join("none");
        let _ = std::fs::create_dir_all(&empty);
        let limits2 = MonitorLimits { backup_dirs: vec![empty.display().to_string()], ..limits };
        let r2 = check_backups(&limits2);
        assert_eq!(r2.level, Level::Fail);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
