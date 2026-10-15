//! Context squeeze — OpenClaw's context-squeeze/juice-compact pattern,
//! ported to the mesh bot.
//!
//! The bot keeps a per-user chat history ring (`state.rs`, 8 turns / 16
//! messages). Long art/roleplay/story threads fill that ring with prose
//! nobody needs verbatim two hours later — but the ring cap alone just
//! drops the oldest turns wholesale, and a fresh model call re-reads
//! whatever remains at full length.
//!
//! This module does what OpenClaw's compaction plugin did, in three
//! escalating stages (cheapest first):
//!
//! 1. **hard-trim** each oversized message to `per_msg_max` chars
//!    (marker-noted), when the total is only mildly over budget;
//! 2. **keep-head-drop-tail**: oldest turns are dropped first (the ring
//!    already does this — squeeze just makes the *budget* the gate instead
//!    of the turn count);
//! 3. **mesh summary**: the oldest `summarize_count` turns are collapsed
//!    into one `role: system` digest via the node's own chat loop
//!    (toolless by construction — the summary prompt is not a question),
//!    seeded budget reclaimed. Summarization happens at most once per
//!    `min_interval` per user; a squeeze run is a pure in-ring operation
//!    when the budget is satisfied.
//!
//! Config (`[squeeze]` in bot.toml; see config.rs SqueezeConfig). Setting
//! `enabled = false` restores the plain 8-turn ring semantics verbatim.

use serde_json::{Value, json};

/// Per-message hard-trim ceiling (chars): a message longer than this in a
/// squeezed ring keeps its head and a marker.
pub const PER_MSG_MAX: usize = 600;

/// Marker appended to hard-trimmed message content.
const TRIM_MARKER: &str = "… [trimmed by context squeeze]";

/// Trim one message's content in place (keeps head). Returns true when a
/// trim happened.
fn trim_message(msg: &mut Value, max: usize) -> bool {
    let Some(content) = msg
        .get("content")
        .and_then(Value::as_str)
        .map(str::to_string)
    else {
        return false;
    };
    if content.chars().count() <= max {
        return false;
    }
    let kept: String = content.chars().take(max).collect();
    if let Some(obj) = msg.as_object_mut() {
        obj.insert("content".to_string(), json!(format!("{kept}{TRIM_MARKER}")));
    }
    true
}

/// Stage 1+2 (pure, unit-tested): bring a history under `budget_chars` by
/// dropping whole oldest turns first, then hard-trimming the remaining
/// oversized messages. Returns the squeezed history (never empty unless
/// the input was). No model call involved.
pub fn squeeze_passive(history: &[Value], budget_chars: usize) -> Vec<Value> {
    if total_chars(history) <= budget_chars {
        return history.to_vec();
    }
    let mut ring: Vec<Value> = history.to_vec();
    // Stage 2: drop oldest turns (pairs) while over budget and more than
    // one turn remains.
    while total_chars(&ring) > budget_chars && ring.len() > 2 {
        ring.drain(..2);
    }
    // Stage 1: hard-trim the biggest remaining messages until under budget
    // or everything is at the cap.
    let mut trimmed_any = true;
    while total_chars(&ring) > budget_chars && trimmed_any {
        trimmed_any = false;
        // Find the largest message above the per-message cap.
        let mut idx = None;
        let mut biggest = PER_MSG_MAX;
        for (i, m) in ring.iter().enumerate() {
            let len = m
                .get("content")
                .and_then(Value::as_str)
                .map(|c| c.chars().count())
                .unwrap_or(0);
            if len > biggest {
                biggest = len;
                idx = Some(i);
            }
        }
        if let Some(i) = idx
            && trim_message(&mut ring[i], PER_MSG_MAX) {
                trimmed_any = true;
            }
    }
    ring
}

fn total_chars(history: &[Value]) -> usize {
    history
        .iter()
        .filter_map(|m| m.get("content").and_then(Value::as_str))
        .map(|c| c.chars().count())
        .sum()
}

/// Stage 3 (async, called from the chat call sites): when the history is
/// still over budget after the passive pass, collapse the oldest
/// `summarize_count` messages into one mesh-written digest and store the
/// replacement back (state.replace_history) — the OpenClaw compaction
/// semantics: the raw turns are gone, the summary lives on.
///
/// Paced by `min_interval_secs` per user (a mesh round-trip per squeeze is
/// enough); on any summary failure the history is left as the passive pass
/// left it (or untouched, if passive already fit). Returns the history the
/// caller should send (passive result when no deep squeeze happened).
pub async fn squeeze_deep(
    state: &crate::state::BotState,
    cfg: &crate::config::BotConfig,
    user: u64,
    raw: &[Value],
) -> Vec<Value> {
    let sq = &cfg.squeeze;
    if !sq.enabled {
        return raw.to_vec();
    }
    let passive = squeeze_passive(raw, sq.budget_chars);
    if total_chars(&passive) <= sq.budget_chars || raw.len() < 4 {
        return passive;
    }
    // Pacing: at most one mesh summary per interval per user.
    if !state.try_gate_squeeze(user, sq.min_interval_secs) {
        return passive;
    }
    let n = sq
        .summarize_count
        .min(raw.len().saturating_sub(2)) // keep at least the newest turn
        .max(2);
    let (to_summarize, rest) = raw.split_at(n);
    let transcript = to_summarize
        .iter()
        .filter_map(|m| {
            let role = m.get("role").and_then(Value::as_str).unwrap_or("?");
            let content = m.get("content").and_then(Value::as_str).unwrap_or("");
            (!content.trim().is_empty()).then(|| format!("{role}: {content}"))
        })
        .collect::<Vec<_>>()
        .join("\n");
    let prompt = format!(
        "Compress this conversation excerpt into one dense paragraph. \
         Keep decisions, facts, names, numbers and open questions; drop pleasantries. \
         Output only the summary.\n\n{transcript}"
    );
    let summary = match crate::api::chat(cfg, &prompt, &[]).await {
        Ok((text, _))
            if !crate::api::answer_is_degenerate(&text)
                && text.trim().chars().count() < total_chars(to_summarize) =>
        {
            text.trim().to_string()
        }
        _ => {
            tracing::debug!(user, "squeeze summary unusable — keeping passive result");
            return passive;
        }
    };
    let mut replacement = vec![json!({
        "role": "system",
        "content": format!("[Earlier conversation, compacted by context squeeze]\n{}", summary.trim()),
    })];
    replacement.extend(rest.iter().cloned());
    state.replace_history(user, replacement.clone());
    tracing::info!(
        user,
        collapsed = n,
        before = total_chars(raw),
        after = total_chars(&replacement),
        "context squeeze: oldest turns collapsed into a mesh digest"
    );
    replacement
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, text: &str) -> Value {
        json!({"role": role, "content": text})
    }

    #[test]
    fn under_budget_history_is_touched_not_at_all() {
        let h = vec![msg("user", "hi"), msg("assistant", "hello")];
        let out = squeeze_passive(&h, 1000);
        assert_eq!(out, h);
    }

    #[test]
    fn over_budget_drops_oldest_turns_first() {
        let h: Vec<Value> = (0..6)
            .flat_map(|i| {
                vec![
                    msg("user", &format!("question {i} {}", "x".repeat(80))),
                    msg("assistant", &format!("answer {i} {}", "y".repeat(80))),
                ]
            })
            .collect();
        let out = squeeze_passive(&h, 500);
        // Oldest turns gone, newest pair intact.
        assert!(!out[0].to_string().contains("question 0"));
        assert!(out.last().unwrap().to_string().contains("answer 5"));
        assert!(total_chars(&out) <= 500 + PER_MSG_MAX);
    }

    #[test]
    fn single_giant_message_is_head_trimmed_not_dropped() {
        let h = vec![
            msg("user", "story time"),
            msg("assistant", &"z".repeat(2000)),
        ];
        let out = squeeze_passive(&h, 700);
        let last = out.last().unwrap();
        let content = last.get("content").and_then(Value::as_str).unwrap();
        assert!(content.starts_with("zzz"));
        assert!(content.ends_with(TRIM_MARKER));
        assert!(content.chars().count() <= PER_MSG_MAX + TRIM_MARKER.len());
    }
}
