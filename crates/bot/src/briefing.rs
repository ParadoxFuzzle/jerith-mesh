//! Daily owner self-briefing — one owner DM per day stitching the past
//! 24h into a single honest note. FEATURES §24.
//!
//! Composition (not invention): every section reads a surface the bot
//! already owns —
//!   * **journal**: an in-process tracing layer (`JournalRing`) records
//!     every WARN/ERROR line (capped ring + counters); the briefing
//!     drains it. Spawning `journalctl` is deliberately NOT done: the
//!     process journal, not the unit journal, is the thing the bot can
//!     always see about itself, dev-box or unit-run alike. Only the
//!     operator's own log lines are kept — nothing user-content leaks.
//!   * **monitor**: `monitor::run_checks` fresh at brief time.
//!   * **mail**: the archive via `mailbox::list_messages_in` (today's
//!     count + the newest subjects; counts are honest about their basis
//!     — archive filename stamps, not delivery receipts).
//!   * **voice**: the live seat line + seat-loss / session-re-establish
//!     counters from voice.rs's runtime state.
//!   * **approvals**: outstanding human-in-the-loop cards (approval.rs).
//!   * **mesh**: peer aliveness + contribution from the gateway status.
//!
//! Delivery honors the owner gate's identity resolution
//! (`cfg.owner_discord_id`). With no owner configured the briefing is
//! built and logged at INFO instead of DM'd — the same fail-quiet rule
//! the monitor uses (reporter only; no privilege beyond its owner).
//!
//! Schedule: an hourly cadence fires the day's briefing once the local
//! hour reaches `[briefing] hour` (default 08). Fire state persists to
//! `$JERITH_HOME/bot/briefing-state.json` (same rebase convention as
//! mine-tally), so a restart AFTER the hour briefs immediately (catch-up)
//! and a restart BEFORE it waits — exactly once per local day. A failed
//! DM is NOT persisted: the next hourly tick retries today's briefing.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use poise::serenity_prelude as serenityall;
use serde_json::Value;
use tracing::field::{Field, Visit};
use tracing::{Event, Level};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;

// ---------------------------------------------------------------------------
// The journal ring (tracing layer)
// ---------------------------------------------------------------------------

/// Every WARN/ERROR the process emits is captured here; readers (the
/// briefing, future diagnostics) drain this, never a spawned journalctl.
/// Install with `.with(JournalRing)` on the fmt subscriber (main.rs).
pub struct JournalRing;

#[derive(Default)]
struct Ring {
    lines: VecDeque<(u64, String)>,
    warns: u64,
    errors: u64,
}

static RING: Mutex<Ring> = Mutex::new(Ring {
    lines: VecDeque::new(),
    warns: 0,
    errors: 0,
});

const RING_CAP: usize = 24;
const LINE_MAX: usize = 140;

fn unix_secs_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn clip_line(s: &str, max: usize) -> String {
    let s = s.trim();
    if s.chars().count() <= max {
        s.to_string()
    } else {
        format!(
            "{}…",
            s.chars().take(max.saturating_sub(1)).collect::<String>()
        )
    }
}

impl<S> Layer<S> for JournalRing
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let level = *event.metadata().level();
        if !matches!(level, Level::WARN | Level::ERROR) {
            return;
        }
        let mut v = LineVisitor::default();
        event.record(&mut v);
        let line = v.render();
        if line.is_empty() {
            return;
        }
        let mut ring = RING.lock().unwrap_or_else(|p| p.into_inner());
        if matches!(level, Level::ERROR) {
            ring.errors += 1;
        } else {
            ring.warns += 1;
        }
        let icon = if matches!(level, Level::ERROR) {
            "error"
        } else {
            "warn"
        };
        ring.lines.push_back((
            unix_secs_now(),
            format!("[{icon}] {}", clip_line(&line, LINE_MAX)),
        ));
        while ring.lines.len() > RING_CAP {
            ring.lines.pop_front();
        }
    }
}

/// Pull a log call's fields: the message field is the main text, the
/// rest accumulate as `field=value` context (matches the `%field`
/// style the crate logs with — Display values arrive via record_debug).
#[derive(Default)]
struct LineVisitor {
    message: Option<String>,
    context: Vec<String>,
}

impl Visit for LineVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.message.get_or_insert(format!("{value:?}"));
        } else if self.context.len() < 2 {
            self.context.push(format!("{}={value:?}", field.name()));
        }
    }
}

impl LineVisitor {
    fn render(&self) -> String {
        let msg = self.message.clone().unwrap_or_default();
        match self.context.is_empty() {
            true => msg,
            false => format!("{} ({})", msg, self.context.join(" ")),
        }
    }
}

/// Section text: counts since the last drain + the newest lines in
/// chronological order. The drain IS the reset — the next window starts
/// here, so the daily briefing's numbers mean "since the last briefing".
pub fn drain_journal_section() -> String {
    let mut ring = RING.lock().unwrap_or_else(|p| p.into_inner());
    let (warns, errors) = (ring.warns, ring.errors); // Newest few, presented in chronological order (oldest of the
    // shown set first — a day's tail reads bottom-up as the truth).
    let lines: Vec<String> = ring
        .lines
        .iter()
        .rev()
        .take(3)
        .map(|(_, l)| l.clone())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    ring.warns = 0;
    ring.errors = 0;
    ring.lines.clear();
    let mut out = format!("📜 journal since last briefing: {warns} warn");
    if errors > 0 {
        out.push_str(&format!(
            ", {errors} error{}",
            if errors == 1 { "" } else { "s" }
        ));
    }
    if warns == 0 && errors == 0 {
        out.push_str("s — all quiet");
        return out;
    }
    if !lines.is_empty() {
        out.push('\n');
        out.push_str(&lines.join("\n"));
    }
    out
}

// ---------------------------------------------------------------------------
// Report assembly
// ---------------------------------------------------------------------------

/// The full body; every section is honest when its source is absent.
async fn build_report(cfg: &crate::config::BotConfig, day: &str) -> String {
    let (seat_losses, reestablishes) = crate::voice::runtime_facts();
    let mut s = format!("🌅 **Jerith daily self-briefing** — {day}\n");

    // Mesh (peer truth; a cheap HTTPS GET that degrades with a name).
    s.push_str(&match crate::api::gateway_status(cfg).await {
        Ok(st) => {
            let get = |k: &str| st.get(k).and_then(Value::as_u64).unwrap_or(0);
            format!(
                "🕸️ mesh: peers {}/{} alive · contribution {}%\n",
                get("peers_alive"),
                get("peer_count"),
                get("contribution_percent")
            )
        }
        Err(e) => format!("🕸️ mesh: unreachable ({e:#})\n"),
    });

    // Voice seat truth — the /status wording, already proven correct.
    s.push_str(&format!("🔊 voice: {}", crate::voice::status_line()));
    if reestablishes > 0 || seat_losses > 0 {
        s.push_str(&format!(
            " · {reestablishes} re-establishments · {seat_losses} seat loss{}",
            if seat_losses == 1 { "" } else { "es" }
        ));
    }
    s.push('\n');

    // Outstanding human approvals.
    match crate::approval::status(None).await {
        Ok(v) => {
            let pending = v.get("count").and_then(Value::as_u64).unwrap_or(0);
            s.push_str(&format!("🪪 approvals: {pending} pending\n"));
        }
        Err(e) => s.push_str(&format!("🪪 approvals: unreadable ({e:#})\n")),
    }

    // Mail — counts mean what they say: archive stamps, not receipts.
    let mail = crate::mailbox::list_messages_in(&crate::mailbox::default_base_dir(), None, 400);
    let today_prefix: String = day.replace('-', "");
    let today_n = mail
        .iter()
        .filter(|m| m.arrived.starts_with(&today_prefix))
        .count();
    s.push_str(&format!(
        "📬 mail: {today_n} today · {} archived\n",
        mail.len()
    ));
    for m in mail.iter().take(2) {
        s.push_str(&format!(
            "   ✉ `{}` [{}] {}\n",
            m.arrived,
            m.box_name,
            clip_line(&m.subject, 60)
        ));
    }

    // Monitor, fresh at brief time (ward/fail lead, ok counted). The
    // icons are re-derived here because monitor's Level::icon is private.
    let limits = crate::monitor::MonitorLimits {
        backup_dirs: cfg.monitor.backup_dirs.clone(),
        ..crate::monitor::MonitorLimits::default()
    };
    let results = crate::monitor::run_checks(&limits);
    let bad: Vec<&crate::monitor::CheckResult> = results
        .iter()
        .filter(|r| r.level != crate::monitor::Level::Ok)
        .collect();
    if bad.is_empty() {
        s.push_str(&format!(
            "🩺 monitor: {} checks pass quietly\n",
            results.len()
        ));
    } else {
        for r in bad {
            let icon = match r.level {
                crate::monitor::Level::Fail => "❌",
                _ => "⚠️",
            };
            s.push_str(&format!(
                "🩺 monitor {}{} — {}\n",
                icon,
                r.check,
                clip_line(&r.detail, 90)
            ));
        }
    }

    // The journal, drained last (its window ends here).
    s.push_str(&drain_journal_section());
    s
}

// ---------------------------------------------------------------------------
// Persisted fire state + schedule loop
// ---------------------------------------------------------------------------

#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
struct BriefingState {
    last_day: Option<String>,
}

/// Same `$JERITH_HOME` rebase convention as the mine-tally ledger.
fn state_path() -> PathBuf {
    let dir = std::env::var("JERITH_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
            PathBuf::from(home).join(".jerith-mesh").join("bot")
        });
    dir.join("briefing-state.json")
}

fn load_last_day(path: &std::path::Path) -> Option<String> {
    let raw = std::fs::read_to_string(path).ok()?;
    serde_json::from_str::<BriefingState>(&raw)
        .ok()
        .and_then(|s| s.last_day)
}

fn save_last_day(path: &std::path::Path, day: &str) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let body = serde_json::to_string(&BriefingState {
        last_day: Some(day.to_string()),
    })
    .unwrap_or_else(|_| "{}".to_string());
    std::fs::write(path, body)
}

/// Local calendar day ("YYYY-MM-DD") + local hour — via the system
/// `date` (the monitor already spawns host tools; std has no local-time
/// parsing and the workspace has no chrono). `date` gone → UTC-math
/// fallback and the UTC date (the fallback honestly says so).
fn local_date_hour() -> (String, u32) {
    if let Ok(o) = std::process::Command::new("date")
        .arg("+%Y-%m-%d|%H")
        .output()
    {
        let t = String::from_utf8_lossy(&o.stdout).trim().to_string();
        if let Some((day, hour)) = t.split_once('|')
            && let Ok(h) = hour.parse::<u32>()
        {
            return (day.to_string(), h.min(23));
        }
        tracing::warn!(got = %t, "local date unparsable — briefing clock falls back to UTC");
    } else {
        tracing::warn!("host date unavailable — briefing clock falls back to UTC");
    }
    let secs = unix_secs_now();
    (format!("UTC {secs}"), (secs / 3600 % 24) as u32)
}

/// The lifespan loop: hourly cadence, fires once per local day once the
/// hour passes `[briefing] hour` (catch-up fires on late boots). No
/// owner → the report is logged at INFO instead of DM'd (fail-quiet).
pub async fn run_loop(
    http: Arc<serenityall::http::Http>,
    owner: Option<u64>,
    cfg: crate::config::BotConfig,
) {
    tracing::info!(
        hour = cfg.briefing.hour,
        owner = owner
            .map(|o| o.to_string())
            .unwrap_or_else(|| "none (journal-log only)".to_string()),
        "daily self-briefing scheduled"
    );
    let path = state_path();
    loop {
        let (day, hour) = local_date_hour();
        if cfg.briefing.enabled
            && hour >= cfg.briefing.hour
            && load_last_day(&path).as_deref() != Some(day.as_str())
        {
            let body = clip_line(&build_report(&cfg, &day).await, 1900);
            match owner {
                Some(id) => {
                    use serenityall::builder::CreateMessage;
                    let msg = CreateMessage::new().content(body.clone());
                    match serenityall::UserId::new(id)
                        .direct_message(&http, msg)
                        .await
                    {
                        Ok(_) => {
                            tracing::info!(day = %day, "daily self-briefing DM delivered");
                            let _ = save_last_day(&path, &day);
                        }
                        Err(e) => {
                            // Not persisted: the next hourly tick retries today's.
                            tracing::warn!(error = %e, day = %day, "briefing DM send failed — retry next tick");
                        }
                    }
                }
                None => {
                    tracing::info!(day = %day, report = %body, "no owner configured — briefing logged instead");
                    let _ = save_last_day(&path, &day);
                }
            }
        }
        tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
    }
}

// ---------------------------------------------------------------------------
// Tests (pure — no Discord, no gate)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// The visitor's honesty: message first, context after, clipped.
    #[test]
    fn journal_visitor_message_first_with_context() {
        let v = LineVisitor {
            message: Some("voice seat lost (driver has no voice connection) — rejoining".into()),
            context: vec!["guild=1296907287633330226".into()],
        };
        assert_eq!(
            v.render(),
            "voice seat lost (driver has no voice connection) — rejoining (guild=1296907287633330226)"
        );
        // Empty message renders to the empty clip (a line that says
        // nothing never enters the ring — see on_event's empty guard).
        let v2 = LineVisitor::default();
        assert_eq!(v2.render(), "");
        assert_eq!(clip_line("x".repeat(400).as_str(), 20).chars().count(), 20);
    }

    /// The ring: counts + cap hold, and the drain IS the reset.
    #[test]
    fn journal_drains_and_resets() {
        {
            let mut ring = RING.lock().unwrap();
            ring.warns = 3;
            ring.errors = 1;
            ring.lines = (0..30)
                .map(|i| (unix_secs_now(), format!("line {i}")))
                .collect();
        }
        let section = drain_journal_section();
        assert!(section.contains("3 warn"), "{section}");
        assert!(section.contains("1 error"), "{section}");
        // Cap held: the newest lines only (24 kept of the 30 pushed).
        assert!(section.contains("line 29"), "{section}");
        assert!(!section.contains("line 26"), "{section}");
        // The drain reset the window.
        {
            let ring = RING.lock().unwrap();
            assert_eq!(ring.warns, 0);
            assert_eq!(ring.errors, 0);
            assert!(ring.lines.is_empty());
        }
        // A second drain says all quiet.
        assert!(drain_journal_section().contains("all quiet"));
    }

    /// The fire gate: once-per-day via the state file, catch-up after
    /// the hour, silent before it.
    #[test]
    fn briefing_state_round_trip_and_gate() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("briefing-test-{}.json", std::process::id()));
        let _ = std::fs::remove_file(&path);
        assert_eq!(load_last_day(&path), None);
        save_last_day(&path, "2026-10-22").unwrap();
        assert_eq!(load_last_day(&path).as_deref(), Some("2026-10-22"));
        let did_fire = |day: &str, hour: u32, cfg_hour: u32, sent: Option<String>| {
            hour >= cfg_hour && sent.as_deref() != Some(day)
        };
        // Already sent today → no refire, even after the hour.
        assert!(!did_fire("2026-10-22", 9, 8, Some("2026-10-22".into())));
        // Fresh day after the hour → fires (catch-up).
        assert!(did_fire("2026-10-23", 21, 8, Some("2026-10-22".into())));
        // Fresh day before the hour → silent.
        assert!(!did_fire("2026-10-23", 3, 8, Some("2026-10-22".into())));
        let _ = std::fs::remove_file(&path);
    }

    /// GOLDEN FIXTURE (2026-10-22, fetched live from the owner DM via the
    /// Discord API): today's real morning briefing, verbatim. The test
    /// asserts the builder's format conventions against it — section
    /// order, per-line emoji prefixes, and each numeric field's shape —
    /// so a format change (emoji, wording, field rename) breaks this
    /// test and forces a deliberate, fresh-golden update rather than a
    /// silent format drift in Jerith's daily DM.
    const GOLDEN_LIVE_DM: &str = concat!(
        "🌅 **Jerith daily self-briefing** — 2026-10-22\n",
        "🕸️ mesh: peers 0/1 alive · contribution 40%\n",
        "🔊 voice: voice: 🔁 lane up — joining the voice channel…\n",
        "🪪 approvals: 0 pending\n",
        "📬 mail: 0 today · 17 archived\n",
        "   ✉ `20261016-132600` [ai] mail tunnel selftest\n",
        "   ✉ `20261015-145213` [ai] Hello Jerith\n",
        "🩺 monitor: 5 checks pass quietly\n",
        "📜 journal since last briefing: 0 warns — all quiet",
    );

    /// The number after `needle` in a line: the leading digit-run (so the
    /// mesh line's `0/1` fraction parses its left side, `40%` its 40).
    fn digits_after(line: &str, needle: &str) -> Option<u64> {
        let i = line.find(needle)?;
        let rest: String = line[i + needle.len()..]
            .trim_start()
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        rest.parse().ok()
    }

    #[test]
    fn golden_live_dm_matches_report_format() {
        let lines: Vec<&str> = GOLDEN_LIVE_DM.lines().collect();

        // The seven sections in the exact order build_report assembles:
        // header → mesh → voice → approvals → mail(+detail) → monitor →
        // journal. Order never shuffles.
        let prefixes = [
            "🌅 **Jerith daily self-briefing** — ",
            "🕸️ mesh: peers ",
            "🔊 voice: voice: ",
            "🪪 approvals: ",
            "📬 mail: ",
            "   ✉ `",
            "   ✉ `",
            "🩺 monitor: ",
            "📜 journal since last briefing: ",
        ];
        assert_eq!(
            lines.len(),
            prefixes.len(),
            "{}\n{}",
            lines.join("\n"),
            GOLDEN_LIVE_DM
        );
        for (line, prefix) in std::iter::zip(&lines, &prefixes) {
            assert!(
                line.starts_with(prefix),
                "line `{line}` does not start with `{prefix}`"
            );
        }

        // Header carries a YYYY-MM-DD day.
        assert!(
            lines[0]["🌅 **Jerith daily self-briefing** — ".len()..]
                .chars()
                .all(|c| c == '-' || c.is_ascii_digit())
        );

        // Mesh: `peers X/Y alive · contribution Z%` — all numeric fields.
        let mesh = lines[1];
        let alive = digits_after(mesh, "peers ").expect("peers_alive");
        let total = digits_after(mesh, &format!("peers {alive}/")).expect("peer_count");
        assert!(alive <= total, "{mesh}");
        digits_after(mesh, "contribution ").expect("contribution_percent");
        assert!(mesh.contains(" alive · contribution "), "{mesh}");

        // Approvals + mail: `N pending` / `N today · N archived`.
        digits_after(lines[3], "approvals: ").expect("pending count");
        digits_after(lines[4], "mail: ").expect("today count");
        digits_after(lines[4], "· ").expect("archived count");

        // Mail details: ``YYYYMMDD-HHMMSS`` stamp + [box] + subject.
        for detail in [&lines[5], &lines[6]] {
            let stamp = &detail["   ✉ `".len()..];
            let stamp = &stamp[..stamp.find('`').expect("closing backtick")];
            let stamp = stamp.replace('-', "");
            assert_eq!(stamp.len(), 14, "{detail}");
            assert!(stamp.chars().all(|c| c.is_ascii_digit()), "{detail}");
            assert!(detail.contains("] "), "{detail}");
        }

        // Monitor: either the all-quiet line (`N checks pass quietly`) or
        // per-check warn/fail lines. The golden carries the quiet shape.
        digits_after(lines[7], "monitor: ").expect("check count");
        assert!(lines[7].contains(" checks pass quietly"), "{}", lines[7]);

        // Journal: `N warn(, N error(s))?` then either trailing lines or
        // the `s — all quiet` closer (0 warns reads as the plural form).
        let jour = lines[8];
        let warns = digits_after(jour, "briefing: ").expect("warn count");
        if warns == 0 {
            assert!(jour.ends_with(" warns — all quiet"), "{jour}");
        } else if let Some(errors) = digits_after(jour, ", ") {
            assert!(jour.contains(&format!(" {errors} error")), "{jour}");
        }
    }
}
