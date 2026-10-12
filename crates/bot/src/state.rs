//! Per-user chat memory + per-user rate limiting.
//!
//! Everything is in-process (a ring of turns per Discord user); the bot
//! keeps no user data on disk beyond what the node's own memory skill
//! already stores.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

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

impl BotStateInner {
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
    fn reset_on_unknown_user_is_zero() {
        let s = state();
        assert_eq!(s.reset(99), 0);
    }
}
