//! Per-user chat memory + per-user rate limiting.
//!
//! Everything is in-process (a ring of turns per Discord user); the bot
//! keeps no user data on disk beyond what the node's own memory skill
//! already stores.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

/// How many turns of per-user context to forward with /ask.
const MAX_HISTORY: usize = 8;
/// Per-user /ask cooldown.
const ASK_COOLDOWN: Duration = Duration::from_secs(5);
/// Per-user /mine cooldown (matches the chain-node side pacing).
const MINE_COOLDOWN: Duration = Duration::from_secs(180);
/// Per-user cooldown for the cheap commands combined (status/peers/recall/remember).
const CMD_COOLDOWN: Duration = Duration::from_secs(2);

#[derive(Debug, Default)]
struct UserState {
    history: Vec<Value>,
    last_ask: Option<Instant>,
    last_mine: Option<Instant>,
    last_cmd: Option<Instant>,
    /// Total JER this user has won via /mine (bot-observed; the chain node's
    /// own `total_mined` includes passive mining the bot never saw).
    jer_won: u64,
    mine_wins: u32,
    /// Pacing for context-squeeze stage 3 (mesh summaries).
    last_squeeze: Option<Instant>,
    /// Cumulative prompt+completion tokens across this user's chats
    /// (bot-observed, from gateway usage figures).
    tokens_used: u64,
}

/// Shared bot state: per-user memory + cooldowns.
#[derive(Debug, Default)]
pub struct BotStateInner {
    users: Mutex<HashMap<u64, UserState>>,
}

pub type BotState = Arc<BotStateInner>;

pub fn new_state() -> BotState {
    Arc::new(BotStateInner::default())
}

/// Result of the cooldown gate: `Err(message)` should be shown verbatim.
pub type GateResult = Result<(), String>;

fn gate(last: &mut Option<Instant>, cooldown: Duration, label: &str) -> GateResult {
    if let Some(t) = *last {
        let elapsed = t.elapsed();
        if elapsed < cooldown {
            let wait = cooldown - elapsed;
            return Err(format!(
                "{label} is on cooldown — try again in {}s",
                wait.as_secs().max(1)
            ));
        }
    }
    *last = Some(Instant::now());
    Ok(())
}

/// Where the tally is persisted: `<home>/mine-tally.json`, rebased if
/// `JERITH_HOME` moves — so /leaderboard survives bot restarts.
pub fn tally_path() -> PathBuf {
    let dir = std::env::var("JERITH_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
            PathBuf::from(home).join(".jerith-mesh").join("bot")
        });
    dir.join("mine-tally.json")
}

#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
struct PersistedTally {
    users: HashMap<u64, PersistedUser>,
}

#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
struct PersistedUser {
    jer_won: u64,
    mine_wins: u32,
    history: Vec<Value>,
    #[serde(default)]
    tokens_used: u64,
}

impl BotStateInner {
    /// Persist the mine tally (+ chat history for continuity) to disk.
    /// Best-effort: a failed write is logged in tests only, never fatal —
    /// the caller decides where failures matter.
    pub fn save_to_disk(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let users = self.users.lock().expect("user map lock");
        let mut persisted = PersistedTally::default();
        for (id, st) in users.iter() {
            persisted.users.insert(
                *id,
                PersistedUser {
                    jer_won: st.jer_won,
                    mine_wins: st.mine_wins,
                    history: st.history.clone(),
                    tokens_used: st.tokens_used,
                },
            );
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, serde_json::to_vec(&persisted)?)?;
        std::fs::rename(&tmp, path)?; // atomic on POSIX
        Ok(())
    }

    /// Load the persisted tally + history into memory (returns # users read).
    pub fn load_from_disk(&self, path: &std::path::Path) -> anyhow::Result<usize> {
        let raw = std::fs::read(path)?;
        let persisted: PersistedTally = serde_json::from_slice(&raw)?;
        let mut users = self.users.lock().expect("user map lock");
        let n = persisted.users.len();
        for (id, p) in persisted.users {
            let st = users.entry(id).or_default();
            st.jer_won = st.jer_won.max(p.jer_won);
            st.mine_wins = st.mine_wins.max(p.mine_wins);
            st.tokens_used = st.tokens_used.max(p.tokens_used);
            if st.history.is_empty() {
                st.history = p.history;
            }
        }
        Ok(n)
    }

    /// Record a /mine win: adds to the user's bot-observed total.
    pub fn record_mine_win(&self, user: u64, jer_amount_uj: u64) {
        let mut users = self.users.lock().expect("user map lock");
        let st = users.entry(user).or_default();
        st.jer_won += jer_amount_uj;
        st.mine_wins += 1;
    }

    /// /leaderboard rows: users with at least one bot-observed win, biggest
    /// total first, capped at `top`.
    pub fn leaderboard(&self, top: usize) -> Vec<(u64, u64, u32)> {
        let users = self.users.lock().expect("user map lock");
        let mut rows: Vec<(u64, u64, u32)> = users
            .iter()
            .filter(|(_, st)| st.mine_wins > 0)
            .map(|(id, st)| (*id, st.jer_won, st.mine_wins))
            .collect();
        rows.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.cmp(&a.2)).then(a.0.cmp(&b.0)));
        rows.truncate(top);
        rows
    }

    /// Check + set the per-user /ask cooldown.
    pub fn gate_ask(&self, user: u64) -> GateResult {
        let mut users = self.users.lock().expect("user map lock");
        let st = users.entry(user).or_default();
        gate(&mut st.last_ask, ASK_COOLDOWN, "/ask")
    }

    /// Check + set the per-user /mine cooldown.
    pub fn gate_mine(&self, user: u64) -> GateResult {
        let mut users = self.users.lock().expect("user map lock");
        let st = users.entry(user).or_default();
        gate(&mut st.last_mine, MINE_COOLDOWN, "/mine")
    }

    /// Check + set the shared cooldown for cheap commands.
    pub fn gate_cmd(&self, user: u64) -> GateResult {
        let mut users = self.users.lock().expect("user map lock");
        let st = users.entry(user).or_default();
        gate(&mut st.last_cmd, CMD_COOLDOWN, "this command")
    }

    /// Context-squeeze pacing: true at most once per `min_interval_secs`
    /// per user — bounds the mesh-summary round-trips stage 3 can spend.
    pub fn try_gate_squeeze(&self, user: u64, min_interval_secs: u64) -> bool {
        let mut users = self.users.lock().expect("user map lock");
        let st = users.entry(user).or_default();
        let now = Instant::now();
        if let Some(t) = st.last_squeeze {
            if now.duration_since(t) < Duration::from_secs(min_interval_secs.max(1)) {
                return false;
            }
        }
        st.last_squeeze = Some(now);
        true
    }

    // Per-user cumulative token accounting (perf/cost meters, bot side).
    pub fn record_usage(&self, user: u64, tokens: u64) {
        let mut users = self.users.lock().expect("user map lock");
        users.entry(user).or_default().tokens_used += tokens;
    }

    /// Total tokens recorded for a user (0 when none).
    pub fn tokens_used(&self, user: u64) -> u64 {
        let users = self.users.lock().expect("user map lock");
        users.get(&user).map(|st| st.tokens_used).unwrap_or(0)
    }

    /// Append a completed turn to the user's in-memory history and trim to
    /// `MAX_HISTORY` turns.
    pub fn push_turn(&self, user: u64, user_text: &str, assistant_text: &str) {
        let mut users = self.users.lock().expect("user map lock");
        let st = users.entry(user).or_default();
        st.history
            .push(json!({"role": "user", "content": user_text}));
        st.history
            .push(json!({"role": "assistant", "content": assistant_text}));
        let excess = st.history.len().saturating_sub(2 * MAX_HISTORY);
        if excess > 0 {
            st.history.drain(..excess);
        }
    }

    /// Snapshot of the user's history for the next /ask.
    pub fn history(&self, user: u64) -> Vec<Value> {
        let users = self.users.lock().expect("user map lock");
        users
            .get(&user)
            .map(|st| st.history.clone())
            .unwrap_or_default()
    }

    /// Context squeeze: atomically replace a user's stored history with a
    /// compacted version (squeeze.rs output). The next `history()` call
    /// serves the compacted ring; the raw turns are gone (dropped-by-design
    /// — the summary digest lives in the replacement), matching OpenClaw's
    /// compaction semantics. Cooldown bookkeeping is untouched.
    pub fn replace_history(&self, user: u64, squeezed: Vec<Value>) {
        let mut users = self.users.lock().expect("user map lock");
        if let Some(st) = users.get_mut(&user) {
            st.history = squeezed;
        }
    }

    /// Clear a user's in-memory chat history. Returns how many turns were dropped.
    pub fn reset(&self, user: u64) -> usize {
        let mut users = self.users.lock().expect("user map lock");
        let st = users.entry(user).or_default();
        let dropped = st.history.len() / 2;
        st.history.clear();
        dropped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> BotState {
        new_state()
    }

    #[test]
    fn ask_cooldown_blocks_then_expires() {
        let s = state();
        assert!(s.gate_ask(1).is_ok());
        assert!(s.gate_ask(1).is_err());
    }

    #[test]
    fn cooldowns_are_per_user() {
        let s = state();
        assert!(s.gate_ask(10).is_ok());
        assert!(s.gate_ask(11).is_ok());
        assert!(s.gate_ask(10).is_err());
    }

    #[test]
    fn mine_cooldown_matches_chain_pacing() {
        let s = state();
        assert!(s.gate_mine(7).is_ok());
        assert!(s.gate_mine(7).is_err());
    }

    #[test]
    fn cmd_cooldown_blocks_immediately() {
        let s = state();
        assert!(s.gate_cmd(9).is_ok());
        assert!(s.gate_cmd(9).is_err());
    }

    #[test]
    fn history_capped_and_reset_counts_turns() {
        let s = state();
        for i in 0..10 {
            s.push_turn(5, &format!("q{i}"), &format!("a{i}"));
        }
        // MAX_HISTORY=8 turns → 16 messages; the oldest two turns dropped.
        let h = s.history(5);
        assert_eq!(h.len(), 16);
        assert!(h[0].to_string().contains("q2"));
        assert_eq!(s.reset(5), 8); // capped at MAX_HISTORY turns
        assert!(s.history(5).is_empty());
    }

    #[test]
    fn mine_wins_accumulate_and_rank() {
        let s = state();
        s.record_mine_win(100, 2_000_000); // 2 JER
        s.record_mine_win(100, 1_000_000); // +1 JER
        s.record_mine_win(200, 5_000_000); // 5 JER
        let rows = s.leaderboard(10);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], (200, 5_000_000, 1));
        assert_eq!(rows[1], (100, 3_000_000, 2));
        // Cap works.
        assert!(s.leaderboard(1).len() == 1);
        // Users without wins never appear.
        assert!(!s.leaderboard(10).iter().any(|r| r.0 == 99));
    }

    #[test]
    fn tally_survives_save_load_roundtrip() {
        let dir = std::env::temp_dir().join(format!("jb-tally-{}", std::process::id()));
        let path = dir.join("mine-tally.json");
        let s = state();
        s.record_mine_win(111, 3_000_000);
        s.record_mine_win(111, 2_000_000);
        s.record_mine_win(222, 9_999);
        s.push_turn(111, "hi", "yo");
        s.save_to_disk(&path).expect("save");

        // Fresh state (like a restarted bot process).
        let s2 = state();
        let n = s2.load_from_disk(&path).expect("load");
        assert_eq!(n, 2);
        let rows = s2.leaderboard(10);
        assert_eq!(rows[0], (111, 5_000_000, 2));
        assert_eq!(rows[1], (222, 9_999, 1));
        // Chat history continuity preserved.
        assert_eq!(s2.history(111).len(), 2);
        // Merging is max() per field: a reload never *lowers* anything, a
        // stale-snapshot reload never erases a newer win, and a reload of
        // the same data is an exact no-op.
        let s3 = state();
        s3.load_from_disk(&path).unwrap();
        // New win adds to the running total (5M + 6M) and increments wins.
        s3.record_mine_win(111, 6_000_000);
        // A stale snapshot reload must not undo the newer win.
        s3.load_from_disk(&path).unwrap();
        assert_eq!(s3.leaderboard(10)[0], (111, 11_000_000, 3));
        // But a *below* snapshot must be ignored (no downgrade).
        let s4 = state();
        s4.record_mine_win(333, 7_000_000);
        s4.load_from_disk(&path).unwrap();
        assert_eq!(s4.leaderboard(10)[0], (333, 7_000_000, 1));

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn save_creates_parent_dirs() {
        let dir = std::env::temp_dir().join(format!("jb-dirs-{}", std::process::id()));
        let path = dir.join("a").join("b").join("tally.json");
        let s = state();
        s.record_mine_win(77, 1);
        s.save_to_disk(&path).expect("save");
        assert!(path.exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn leaderboard_empty() {
        let s = state();
        assert!(s.leaderboard(10).is_empty());
    }

    #[test]
    fn reset_on_unknown_user_is_zero() {
        let s = state();
        assert_eq!(s.reset(99), 0);
    }
}
