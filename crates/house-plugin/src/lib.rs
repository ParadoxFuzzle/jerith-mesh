//! `jerith_house_plugin` — the mesh house's own state card.
//!
//! The third plugin drop over the SDK's signed `/rpc` contract
//! (`docs/PLUGIN-SDK.md`, after the example and the mesh pet). It
//! answers one question the chat gets asked all the time — *how are
//! you doing?* — by COMPOSING surfaces the house already owns, never
//! by inventing:
//!
//! - **gateway `/status`** — mesh truth (peers, contribution, model);
//!   unreachable is reported, not hidden;
//! - **the bot unit** — `systemctl --user show` facts (active, booted
//!   when, restart count); the tool absent → unknown, honestly typed;
//! - **bot.toml** — ONLY the `[transports]` table's key NAMES (the
//!   configured chat surfaces); no values are ever read or relayed —
//!   the manifest's loopback posture does not get to leak tokens;
//! - **the process journal** — recent WARN/ERROR lines (bounded count,
//!   clipped lines), when `journalctl --user` is reachable;
//! - **the commit** this build runs from, when the house repo is
//!   locatable.
//!
//! Every source is injected as an `Option` into [`compose`] — pure,
//! clock-free, and testable against real fixtures. A source that
//! cannot be reached leaves its section absent or honestly marked
//! unknown; no field is ever guessed. The one-word `word` verdict
//! ("steady" / "a bit loud" / "needs you") is derived, all numbers
//! come straight from the surfaces.

use serde_json::{Value, json};

/// Everything a card is composed from. Each field is a completed READ —
/// the fetch side (main.rs) owns spawning/IO, this side stays pure.
#[derive(Default)]
pub struct Sources<'a> {
    /// Parsed gateway `/status` body; `None` = the gateway did not
    /// answer.
    pub gateway: Option<&'a Value>,
    /// Raw `systemctl --user show jerith-bot` output; `None` = systemd
    /// not reachable (dev box, container, …).
    pub systemctl: Option<&'a str>,
    /// Raw `bot.toml`; `None` = the file isn't where the env points.
    pub bot_toml: Option<&'a str>,
    /// Recent WARN/ERROR journal lines (already bounded by the caller);
    /// `None` = the journal was not readable.
    pub journal: Option<&'a [String]>,
    /// Short commit hash of the tree this build ran from; `None` when
    /// the repo root wasn't locatable.
    pub commit: Option<&'a str>,
}

/// Clip a line for the card (same honesty as the briefing: keep the
/// head, mark the cut).
fn clip(s: &str, max: usize) -> String {
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

/// The chat-surface NAMES configured in `bot.toml` — table keys of
/// `[transports]` only. Values stay unread: a name carries no secret,
/// the table's contents carry tokens. An absent table or a file that
/// won't parse is an honest empty list with `transports_known: false`.
pub fn transport_names(raw: Option<&str>) -> (Vec<String>, bool) {
    let Some(raw) = raw else {
        return (Vec::new(), false);
    };
    let Ok(doc) = toml::from_str::<toml::Table>(raw) else {
        return (Vec::new(), false);
    };
    let Some(t) = doc.get("transports").and_then(toml::Value::as_table) else {
        return (Vec::new(), false); // no transports table: known-absent
    };
    (t.keys().cloned().collect(), true)
}

/// Bot-unit facts from `systemctl --user show` output (the caller asks
/// for ActiveState, ActiveEnterTimestamp, NRestarts).
pub fn bot_facts(raw: Option<&str>) -> Value {
    let Some(raw) = raw else {
        return json!({"active": "unknown"});
    };
    let mut st = String::new();
    let mut since = String::new();
    let mut restarts: u64 = 0;
    for line in raw.lines() {
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        match k {
            "ActiveState" => st = v.trim().to_string(),
            "ActiveEnterTimestamp" => since = v.trim().to_string(),
            "NRestarts" => restarts = v.trim().parse().unwrap_or(0),
            _ => {}
        }
    }
    json!({
        "active": if st.is_empty() { "unknown" } else { st.as_str() },
        "since": since,
        "restarts": restarts,
    })
}

/// Mesh numbers, straight from `/status` top-level keys (the same
/// fields the daily briefing reads). `gateway_up` is the report, not
/// an assumption.
pub fn mesh_numbers(gw: Option<&Value>) -> Value {
    let Some(gw) = gw else {
        return json!({"gateway_up": false});
    };
    let u = |k: &str| gw.get(k).and_then(Value::as_u64).unwrap_or(0);
    json!({
        "gateway_up": true,
        "node_id": gw.get("node_id").and_then(Value::as_str).unwrap_or("?"),
        "model": gw.get("model").and_then(Value::as_str).unwrap_or("?"),
        "peers_alive": u("peers_alive"),
        "peer_count": u("peer_count"),
        "contribution_percent": u("contribution_percent"),
        "contribution_healthy": gw
            .pointer("/contribution/healthy")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    })
}

/// Recent journal lines: newest kept, capped, clipped. `None` stays
/// absent (the card omits the field — journal readers deserve to know
/// there IS no journal access).
pub fn journal_clip(lines: Option<&[String]>, cap: usize) -> Option<Value> {
    let lines = lines?;
    let kept: Vec<String> = lines
        .iter()
        .rev()
        .take(cap)
        .rev()
        .map(|l| clip(l, 140))
        .collect();
    Some(json!(kept))
}

/// The considered one-word verdict: composed, not averaged —
/// - the bot unit down or the gateway unreachable → **needs you**;
/// - otherwise with recent warns on record → **a bit loud**;
/// - otherwise → **steady**.
pub fn word(bot: Option<&Value>, mesh: Option<&Value>, warns: Option<&[String]>) -> &'static str {
    let bot_ok = bot
        .map(|b| b.get("active").and_then(Value::as_str) == Some("active"))
        .unwrap_or(false);
    let mesh_ok = mesh.map(|m| m.get("gateway_up").and_then(Value::as_bool) != Some(false));
    if !bot_ok || !mesh_ok.unwrap_or(false) {
        return "needs you";
    }
    if warns.map(|w| !w.is_empty()).unwrap_or(false) {
        return "a bit loud";
    }
    "steady"
}

/// The full card — every section honest about its absence.
pub fn compose(s: &Sources) -> Value {
    let bot = Some(bot_facts(s.systemctl));
    let mesh = Some(mesh_numbers(s.gateway));
    let mut card = json!({
        "house": "jerith-mesh",
        "word": word(bot.as_ref(), mesh.as_ref(), s.journal),
        "bot": bot,
        "mesh": mesh,
    });
    let obj = card.as_object_mut().expect("card just built as object");
    if let Some(commit) = s.commit {
        obj.insert("commit".into(), json!(commit));
    }
    let (transports, known) = transport_names(s.bot_toml);
    obj.insert("transports".into(), json!(transports));
    obj.insert("transports_known".into(), json!(known));
    if let Some(j) = journal_clip(s.journal, 5) {
        obj.insert("recent_warns".into(), j);
    }
    card
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A REAL /status capture (2026-10-23, this house's gateway),
    /// reduced to the fields the card reads — the composition is
    /// pinned against actual voltage, not an invented shape.
    fn status_fixture() -> Value {
        json!({
            "model": "qwen3:8b",
            "node_id": "4353ea12d8100f45",
            "peers_alive": 1,
            "peer_count": 1,
            "contribution_percent": 40,
            "contribution": {"declared_percent": 40, "healthy": true},
        })
    }

    const BOT_TOML_FIXTURE: &str = r#"
[briefing]
hour = 8
[transports.telegram]
token = "redacted-token-never-lands-in-the-card"
[transports.irc]
server = "irc.example.net"
"#;

    const SYSTEMCTL_ACTIVE: &str = "\
ActiveState=active
ActiveEnterTimestamp=Tue 2026-10-23 02:04:30 EDT
NRestarts=0
";

    #[test]
    fn transports_report_names_and_never_values() {
        let (names, known) = transport_names(Some(BOT_TOML_FIXTURE));
        // toml::Table sorts its keys — deterministic order, tested as such.
        assert_eq!(names, vec!["irc", "telegram"]);
        assert!(known);
        // A file with no transports table: known-absent, not an error.
        let (names2, known2) = transport_names(Some("[briefing]\nhour = 8\n"));
        assert!(names2.is_empty() && !known2);
        // Unreadable / absent file: unchanged honesty.
        let (names3, known3) = transport_names(Some("not toml +++"));
        assert!(names3.is_empty() && !known3);
        assert!(!transport_names(None).1);
    }

    #[test]
    fn bot_facts_parse_the_show_output() {
        let f = bot_facts(Some(SYSTEMCTL_ACTIVE));
        assert_eq!(f["active"], "active");
        assert_eq!(f["since"], "Tue 2026-10-23 02:04:30 EDT");
        assert_eq!(f["restarts"], 0);
        // Missing tool: unknown, never guessed.
        assert_eq!(bot_facts(None)["active"], "unknown");
        let f2 = bot_facts(Some(""));
        assert_eq!(f2["active"], "unknown");
    }

    #[test]
    fn mesh_numbers_come_from_the_status_top_level() {
        let m = mesh_numbers(Some(&status_fixture()));
        assert_eq!(m["gateway_up"], true);
        assert_eq!(m["peers_alive"], 1);
        assert_eq!(m["peer_count"], 1);
        assert_eq!(m["contribution_percent"], 40);
        assert_eq!(m["contribution_healthy"], true);
        assert_eq!(m["model"], "qwen3:8b");
        assert_eq!(mesh_numbers(None)["gateway_up"], false);
    }

    #[test]
    fn journal_clip_keeps_newest_and_caps() {
        let lines: Vec<String> = (0..12).map(|i| format!("line {i}")).collect();
        let kept = journal_clip(Some(&lines), 5).unwrap();
        assert_eq!(kept.as_array().unwrap().len(), 5);
        assert_eq!(kept[0], "line 7");
        assert_eq!(kept[4], "line 11");
        assert!(journal_clip(None, 5).is_none());
        // Long lines clip with the ellipsis marker.
        let long = vec!["x".repeat(300)];
        assert!(
            journal_clip(Some(&long), 5).unwrap()[0]
                .as_str()
                .unwrap()
                .ends_with('…')
        );
    }

    #[test]
    fn the_word_is_honest_about_each_layer() {
        let mesh = mesh_numbers(Some(&status_fixture()));
        let bot_ok = bot_facts(Some(SYSTEMCTL_ACTIVE));
        let bot_down = bot_facts(Some("ActiveState=failed\n"));
        let bot_unk = bot_facts(None);
        let mesh_out = mesh_numbers(None);
        // All green → steady. A warn on record → a bit loud.
        assert_eq!(word(Some(&bot_ok), Some(&mesh), Some(&[])), "steady");
        assert_eq!(
            word(Some(&bot_ok), Some(&mesh), Some(&["warn".into()])),
            "a bit loud"
        );
        // Unit down, or gateway out, or either unknowable → needs you.
        assert_eq!(word(Some(&bot_down), Some(&mesh), Some(&[])), "needs you");
        assert_eq!(word(Some(&bot_ok), Some(&mesh_out), Some(&[])), "needs you");
        assert_eq!(word(Some(&bot_unk), Some(&mesh), Some(&[])), "needs you");
    }

    #[test]
    fn compose_full_stage_and_degraded_stage() {
        let gw = status_fixture();
        let journal = vec!["2026-10-23T07:04:34Z WARN briefings: retry next tick".to_string()];
        let full = compose(&Sources {
            gateway: Some(&gw),
            systemctl: Some(SYSTEMCTL_ACTIVE),
            bot_toml: Some(BOT_TOML_FIXTURE),
            journal: Some(&journal),
            commit: Some("5954770"),
        });
        assert_eq!(full["word"], "a bit loud");
        assert_eq!(full["commit"], "5954770");
        assert_eq!(full["bot"]["active"], "active");
        assert_eq!(full["mesh"]["model"], "qwen3:8b");
        assert_eq!(full["transports"], json!(["irc", "telegram"]));
        assert_eq!(full["recent_warns"].as_array().unwrap().len(), 1);

        // Every source dark: the card still speaks, and honestly.
        let dark = compose(&Sources::default());
        assert_eq!(dark["word"], "needs you");
        assert!(dark.get("commit").is_none());
        assert!(dark.get("recent_warns").is_none());
        assert_eq!(dark["transports"], json!([]));
        assert_eq!(dark["transports_known"], false);
        assert_eq!(dark["mesh"]["gateway_up"], false);
        assert_eq!(dark["bot"]["active"], "unknown");
        // Nothing about the house's internals ever exceeds the card's
        // honest bounds (a fixture-shaped sanity assertion, kept small).
        let full_text = serde_json::to_string(&full).unwrap();
        assert!(!full_text.contains("redacted-token"), "values never leak");
        assert!(full_text.len() < 4096, "card stays small: {full_text}");
    }
}
