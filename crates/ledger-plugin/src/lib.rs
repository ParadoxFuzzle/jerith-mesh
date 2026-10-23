//! `jerith_ledger_plugin` — the task ledger / intentions lane (the
//! plugin half; the bot's daily briefing reads the lane through the
//! signed skill surface like every other drop — never the state file
//! directly. The state path is this file's `state_path()`, mirrored
//! nowhere: the contract is the skills).
//!
//! The shape of the thing: the owner (or the model on the owner's
//! behalf) voices a goal worth keeping — `ledger.add` makes it a typed
//! entry on disk; `ledger.done` closes it (kept, timestamped); `ledger.drop`
//! deletes it (a mistake, not an achievement — never archived); `ledger.list`
//! answers "what's open" with honest ages. The briefing renders the
//! open tail too, so a goal whispered at midnight comes back the next
//! morning instead of evaporating in scrollback.
//!
//! Discipline like the pet and house: `lib.rs` is pure between the
//! edges (clock and state path injected), `main.rs` is wire glue over
//! the SDK's signed `/rpc` contract.

use serde::{Deserialize, Serialize};
use std::io::Write as _;
use std::path::{Path, PathBuf};

/// Open intentions cap: a ledger of 200 unfinished thoughts is a
/// mood board, not a ledger — one more add refuses with an honest
/// typed error instead of growing forever.
pub const OPEN_CAP: usize = 200;

/// Closed entries kept in the file (newest kept, oldest dropped).
pub const ARCHIVE_CAP: usize = 500;

/// One text cap (a ledger entry is a goal, not a document).
pub const TEXT_CAP: usize = 200;

// ---------------------------------------------------------------------------
// State + entries
// ---------------------------------------------------------------------------

/// One intention. `closed_at_ms` set = done (kept as history); the
/// entry is open while it's `None`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Entry {
    /// Monotone id, stable across restarts — the handle the chat
    /// commands (`done 12` / `drop 12`) key on.
    pub id: u64,
    pub text: String,
    /// Optional one-line context, verbatim memory space like the pet's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    pub added_at_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub closed_at_ms: Option<u64>,
}

impl Entry {
    pub fn open(&self) -> bool {
        self.closed_at_ms.is_none()
    }
}

/// The whole file. `next_id` makes ids unique across saves; entries
/// keep insertion order (oldest first — ages render honestly).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Ledger {
    version: u32,
    next_id: u64,
    pub entries: Vec<Entry>,
}

#[derive(Debug, PartialEq)]
pub enum LedgerError {
    /// Text dedupe: the same sentence open twice is noise.
    Duplicate(u64),
    /// Blank text after trim — not a goal, an empty thought.
    Empty,
    Cap,
    UnknownId(u64),
}

impl Ledger {
    pub fn new() -> Self {
        Ledger {
            version: 1,
            next_id: 1,
            entries: Vec::new(),
        }
    }

    fn open_count(&self) -> usize {
        self.entries.iter().filter(|e| e.open()).count()
    }

    /// Add one intention. Duplicate-guard is on the OPEN text only
    /// (doing the same thing next week is normal; redoing an open one
    /// means the entry already says it).
    pub fn add(
        &mut self,
        text: &str,
        note: Option<String>,
        now_ms: u64,
    ) -> Result<Entry, LedgerError> {
        let text = text.trim();
        if text.is_empty() {
            return Err(LedgerError::Empty);
        }
        let text = if text.len() > TEXT_CAP {
            &text[..TEXT_CAP]
        } else {
            text
        };
        if let Some(e) = self.entries.iter().find(|e| e.open() && e.text == text) {
            return Err(LedgerError::Duplicate(e.id));
        }
        if self.open_count() >= OPEN_CAP {
            return Err(LedgerError::Cap);
        }
        let id = self.next_id;
        self.next_id += 1;
        let e = Entry {
            id,
            text: text.to_string(),
            note,
            added_at_ms: now_ms,
            closed_at_ms: None,
        };
        self.entries.push(e.clone());
        Ok(e)
    }

    fn by_id_mut(&mut self, id: u64) -> Option<&mut Entry> {
        self.entries.iter_mut().find(|e| e.id == id)
    }

    /// Close one: kept in place, timestamped (the archive is history,
    /// not a trash can — never edited again after this).
    pub fn done(&mut self, id: u64, now_ms: u64) -> Result<Entry, LedgerError> {
        match self.by_id_mut(id) {
            Some(e) if e.open() => {
                e.closed_at_ms = Some(now_ms);
                Ok(e.clone())
            }
            Some(e) => Err(LedgerError::Duplicate(e.id)), // already closed
            None => Err(LedgerError::UnknownId(id)),
        }
    }

    /// Delete one without archiving — the "that wasn't a goal, that
    /// was a typo" lane.
    pub fn drop(&mut self, id: u64) -> Result<Entry, LedgerError> {
        match self.entries.iter().position(|e| e.id == id) {
            Some(i) => Ok(self.entries.remove(i)),
            None => Err(LedgerError::UnknownId(id)),
        }
    }

    /// Open intentions, oldest first; the archive is trimmed on save.
    pub fn open_sorted(&self) -> Vec<&Entry> {
        let mut v: Vec<&Entry> = self.entries.iter().filter(|e| e.open()).collect();
        v.sort_by_key(|e| e.added_at_ms);
        v
    }

    /// Closed ones this civil local day (the briefing's "closed today").
    pub fn closed_on_day(&self, day: i64) -> usize {
        self.entries
            .iter()
            .filter(|e| {
                e.closed_at_ms
                    .map(|ms| civil_day_of(ms) == day)
                    .unwrap_or(false)
            })
            .count()
    }

    /// Archive trim on the way out: only CLOSED entries are capped
    /// (ARCHIVE_CAP newest kept); every OPEN entry survives regardless
    /// of age — a week-old intention is not archive debris.
    fn trimmed(&self) -> Vec<Entry> {
        let closed = self.entries.iter().filter(|e| !e.open()).count();
        let mut to_remove = closed.saturating_sub(ARCHIVE_CAP);
        let mut out: Vec<Entry> = Vec::with_capacity(self.entries.len());
        for e in &self.entries {
            if to_remove > 0 && !e.open() {
                to_remove -= 1;
                continue; // oldest closed ones fall off the end
            }
            out.push(e.clone());
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Persistence (atomic: temp + rename, a killed process can't halve the file)
// ---------------------------------------------------------------------------

/// Where the ledger lives: `$JERITH_LEDGER_STATE` override, else
/// `<$JERITH_HOME|~>/.jerith-mesh/ledger/state.json` — the plugins.d
/// drop holds the manifest, never state (the pet's convention).
pub fn state_path() -> PathBuf {
    if let Ok(p) = std::env::var("JERITH_LEDGER_STATE") {
        return PathBuf::from(p);
    }
    let home = std::env::var("JERITH_HOME").unwrap_or_else(|_| {
        format!(
            "{}/.jerith-mesh",
            std::env::var("HOME").unwrap_or_else(|_| ".".to_string())
        )
    });
    PathBuf::from(home).join("ledger").join("state.json")
}

pub fn load(path: &Path) -> Ledger {
    match std::fs::read_to_string(path) {
        Ok(raw) => serde_json::from_str::<Ledger>(&raw).unwrap_or_default(),
        Err(_) => Ledger::new(),
    }
}

pub fn save(path: &Path, ledger: &Ledger) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let body =
        serde_json::to_string_pretty(&ledger.trimmed_to_save()).map_err(std::io::Error::other)?;
    let tmp = path.with_extension("json.tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(body.as_bytes())?;
        f.write_all(b"\n")?;
        f.sync_all().ok();
    }
    std::fs::rename(&tmp, path)
}

impl Ledger {
    fn trimmed_to_save(&self) -> Ledger {
        Ledger {
            version: self.version,
            next_id: self.next_id,
            entries: self.trimmed(),
        }
    }
}

// ---------------------------------------------------------------------------
// Day math (Howard Hinnant's civil-from-days, the pet's six numbers)
// ---------------------------------------------------------------------------

/// Civil day index from a unix-ms stamp — the "today" key of
/// `closed_on_day` and the age renderer's whole-multiple basis.
pub fn civil_day_of(ms: u64) -> i64 {
    (ms / 1000 / 86_400) as i64
}

/// Human age: whole days when ≥2, else hours when ≥2, else minutes
/// (from 2 up), else "just now" — each tier only when it doesn't
/// render as "1".
pub fn age_word(now_ms: u64, then_ms: u64) -> String {
    let age = now_ms.saturating_sub(then_ms);
    if age >= 2 * 86_400_000 {
        format!("{} d", age / 86_400_000)
    } else if age >= 2 * 3_600_000 {
        format!("{} h", age / 3_600_000)
    } else if age >= 2 * 60_000 {
        format!("{} min", age / 60_000)
    } else {
        "just now".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HOUR: u64 = 3_600_000;
    const DAY: u64 = 86_400_000;

    #[test]
    fn add_done_drop_roundtrip() {
        let mut l = Ledger::new();
        let a = l.add("ship the node", None, 100 * DAY).unwrap();
        let b = l
            .add(
                "migrate bot.toml",
                Some("with the rotated secret".into()),
                100 * DAY + HOUR,
            )
            .unwrap();
        assert_eq!(a.id, 1); // ids monotone from 1
        assert_eq!(b.id, 2);
        assert_eq!(l.open_sorted().len(), 2);
        let closed = l.done(b.id, 101 * DAY).unwrap();
        assert!(!closed.open());
        assert_eq!(l.open_sorted().len(), 1);
        let dropped = l.drop(a.id).unwrap();
        assert_eq!(dropped.text, "ship the node");
        assert_eq!(l.open_sorted().len(), 0);
    }

    #[test]
    fn duplicate_open_text_refuses_named() {
        let mut l = Ledger::new();
        l.add("same goal", None, 0).unwrap();
        let dup = l.add("same goal  ", None, 1).unwrap_err();
        match dup {
            LedgerError::Duplicate(1) => {}
            other => panic!("expected a duplicate naming entry 1, got {other:?}"),
        }
        // But a ALREADY-CLOSED one can be re-added (redoing work is legit).
        l.done(1, DAY).unwrap();
        assert!(l.add("same goal", None, 2 * DAY).is_ok());
    }

    #[test]
    fn done_is_once_and_unknown_ids_type() {
        let mut l = Ledger::new();
        let e = l.add("x", None, 0).unwrap();
        l.done(e.id, DAY).unwrap();
        assert!(matches!(
            l.done(e.id, 2 * DAY),
            Err(LedgerError::Duplicate(_))
        ));
        assert_eq!(l.drop(999), Err(LedgerError::UnknownId(999)));
    }

    #[test]
    fn text_trim_and_empty() {
        let mut l = Ledger::new();
        let e = l.add(&"y".repeat(300), None, 0).unwrap();
        assert_eq!(e.text.len(), TEXT_CAP);
        assert!(
            matches!(l.add("   ", None, 0), Err(LedgerError::Empty)),
            "blank text is not a goal"
        );
    }

    #[test]
    fn open_cap_refuses() {
        let mut l = Ledger::new();
        for i in 0..OPEN_CAP {
            l.add(&format!("goal {i}"), None, i as u64).unwrap();
        }
        assert_eq!(l.add("one more", None, 9_999), Err(LedgerError::Cap));
        // Closing one reopens the lane.
        l.done(1, DAY).unwrap();
        assert!(l.add("one more", None, 10_000).is_ok());
    }

    #[test]
    fn save_load_roundtrip_and_archive_trim() {
        let dir = std::env::temp_dir().join(format!("ledger-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.json");
        let mut l = Ledger::new();
        l.add("persist me", Some("note rides too".into()), 0)
            .unwrap();

        // Push past the archive cap, close them all.
        for i in 0..(ARCHIVE_CAP as u64 + 3) {
            let e = l
                .add(&format!("old {i}"), None, (i * 60_000).saturating_sub(1))
                .unwrap();
            l.done(e.id, i * 60_000 + 1).unwrap();
        }
        save(&path, &l).unwrap();
        let back = load(&path);
        assert_eq!(back.open_sorted()[0].text, "persist me");
        assert_eq!(
            back.open_sorted()[0].note.as_deref(),
            Some("note rides too")
        );
        assert_eq!(
            back.entries.iter().filter(|e| !e.open()).count(),
            ARCHIVE_CAP,
            "closed entries capped on save"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn closed_on_day_uses_civil_days() {
        let mut l = Ledger::new();
        let e1 = l.add("a", None, 5 * DAY).unwrap();
        let e2 = l.add("b", None, 5 * DAY + HOUR).unwrap();
        l.done(e1.id, 5 * DAY + 2 * HOUR).unwrap(); // same civil day as closed
        l.done(e2.id, 6 * DAY + 2 * HOUR).unwrap(); // next civil day
        let e3 = l.add("c", None, 5 * DAY).unwrap();
        assert_eq!(l.closed_on_day(civil_day_of(5 * DAY)), 1);
        assert_eq!(l.closed_on_day(civil_day_of(6 * DAY)), 1);
        assert_eq!(l.closed_on_day(civil_day_of(0)), 0);
        let _ = e3;
    }

    #[test]
    fn age_words_degrade_honestly() {
        assert_eq!(age_word(0, 0), "just now");
        assert_eq!(age_word(90_000, 0), "just now");
        assert_eq!(age_word(90 * 60_000, 0), "90 min");
        assert_eq!(age_word(3 * HOUR, 0), "3 h");
        assert_eq!(age_word(5 * DAY + HOUR, 0), "5 d");
    }
}
