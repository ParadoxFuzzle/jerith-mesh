//! `jerith-ledger` — the task ledger / intentions lane's wire glue,
//! over the plugin SDK (`crates/plugin-template`, docs/PLUGIN-SDK.md;
//! the SDK's fifth real consumer after the example, the pet, the
//! house card, and the quarantine scanner). The ledger engine + state
//! live in `lib.rs` (pure, tested); this binary:
//!
//! - reads its OWN `plugin.toml` for `[plugin] endpoint` + `secret`
//!   and binds that exact address — manifest and service cannot drift
//!   (the one-source-of-truth rule every drop keeps);
//! - serves the four model-visible skills the manifest declares:
//!   `ledger.add` (capture an intention — the model's standing
//!   instruction is to offer it whenever the owner voices a goal),
//!   `ledger.list` (what's open + today's closes; the daily briefing
//!   renders from this same card), `ledger.done` and `ledger.drop`
//!   (close or delete by id);
//! - every mutation saves atomically (temp + rename); a killed
//!   process can't halve the file.
//!
//! No watcher, no poll — unlike quarantine and pet this lane only
//! moves when asked: intentions don't decay on a clock, they decay
//! when the owner closes them.

use anyhow::{Context as _, Result};
use jerith_plugin_template::{SkillServer, endpoint_addr, skill};

use jerith_ledger_plugin::{LedgerError, age_word, civil_day_of, load, save, state_path};
use serde_json::{Value, json};
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

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

/// One entry → the card shape the skills return (bounded: text comes
/// back whole — a goal is short by construction).
fn entry_card(e: &jerith_ledger_plugin::Entry, now: u64) -> Value {
    let mut c = json!({
        "id": e.id,
        "text": e.text,
        "added": age_word(now, e.added_at_ms),
        "status": if e.open() { "open" } else { "done" },
    });
    if let Some(n) = &e.note {
        c["note"] = json!(n);
    }
    if let Some(closed) = e.closed_at_ms {
        c["closed"] = json!(age_word(now, closed));
    }
    c
}

/// The list card — the briefing renders from exactly this (+ capped).
fn do_list(state: &Path) -> Result<Value, String> {
    let now = now_ms();
    let l = load(state);
    let open: Vec<Value> = l
        .open_sorted()
        .iter()
        .take(8)
        .map(|e| entry_card(e, now))
        .collect();
    Ok(json!({
        "state_file": state.display().to_string(),
        "open_count": l.open_sorted().len(),
        "oldest_id": l.open_sorted().first().map(|e| e.id),
        "oldest_age": l.open_sorted().first().map(|e| age_word(now, e.added_at_ms)),
        "closed_today": l.closed_on_day(civil_day_of(now)),
        "open": open,
    }))
}

async fn do_add(state: &Path, args: &Value) -> Result<Value, String> {
    let task = args["task"].as_str().unwrap_or("");
    let note = args["note"].as_str().map(str::to_string);
    let mut l = load(state);
    match l.add(task, note, now_ms()) {
        Ok(e) => {
            if let Err(io) = save(state, &l) {
                tracing::warn!(target: "ledger", error = %io, "ledger save failed after add");
            }
            Ok(json!({
                "added": entry_card(&e, now_ms()),
                "open_count": l.open_sorted().len(),
            }))
        }
        Err(LedgerError::Duplicate(id)) => Err(format!(
            "already open as #{id} — close that one or reword this"
        )),
        Err(LedgerError::Empty) => {
            Err("an empty string is not a goal — what is it, in one line?".into())
        }
        Err(LedgerError::Cap) => Err(format!(
            "the ledger is at its {} open-entry cap — close a few before adding more",
            jerith_ledger_plugin::OPEN_CAP
        )),
        Err(e) => Err(format!("add failed: {e:?}")),
    }
}

async fn do_close(state: &Path, args: &Value, drop_instead: bool) -> Result<Value, String> {
    let id = match args["id"].as_u64() {
        Some(i) => i,
        None => return Err("an entry id is required (from `ledger.list`)".into()),
    };
    let mut l = load(state);
    let verb = if drop_instead { "dropped" } else { "closed" };
    let r = if drop_instead {
        l.drop(id)
    } else {
        l.done(id, now_ms())
    };
    match r {
        Ok(e) => {
            if let Err(io) = save(state, &l) {
                tracing::warn!(target: "ledger", error = %io, "ledger save failed after {verb}");
            }
            let card = entry_card(&e, now_ms());
            Ok(json!({
                verb: card,
                "open_count": l.open_sorted().len(),
            }))
        }
        Err(LedgerError::Duplicate(closed_id)) if !drop_instead => Err(format!(
            "#{closed_id} is already done — `ledger.list` shows what's open"
        )),
        Err(LedgerError::UnknownId(i)) => {
            Err(format!("no entry #{i} — ids come from `ledger.list`"))
        }
        Err(e) => Err(format!("{verb} failed: {e:?}")),
    }
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
    let state = Arc::new(state_path());
    tracing::info!(
        manifest,
        %addr,
        secret = secret.is_some(),
        state = %state.display(),
        "ledger plugin starting"
    );

    SkillServer::new("ledger", secret)
        .handler(
            "ledger.add",
            skill({
                let state = state.clone();
                move |args: Value| {
                    let state = state.clone();
                    async move { do_add(&state, &args).await }
                }
            }),
        )
        .handler(
            "ledger.list",
            skill({
                let state = state.clone();
                move |_args: Value| {
                    let state = state.clone();
                    async move { do_list(&state) }
                }
            }),
        )
        .handler(
            "ledger.done",
            skill({
                let state = state.clone();
                move |args: Value| {
                    let state = state.clone();
                    async move { do_close(&state, &args, false).await }
                }
            }),
        )
        .handler(
            "ledger.drop",
            skill({
                let state = state.clone();
                move |args: Value| {
                    let state = state.clone();
                    async move { do_close(&state, &args, true).await }
                }
            }),
        )
        .serve(addr)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use jerith_ledger_plugin::Ledger;

    /// The SHIPPED manifest must stay servable by THIS binary: endpoint
    /// binds loopback, on a port that is not the other drops', signed,
    /// and the declared rows are exactly the four registered above.
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
            Some("ledger")
        );
        let endpoint = plugin
            .get("endpoint")
            .and_then(toml::Value::as_str)
            .expect("endpoint line");
        let addr = endpoint_addr(endpoint).expect("endpoint binds");
        assert!(addr.ip().is_loopback(), "loopback by contract");
        for taken in [7788, 7789, 7790, 7791] {
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
        assert_eq!(
            names,
            vec!["add", "list", "done", "drop"],
            "the four handlers"
        );
    }

    /// The list card's pin: the briefing renders these exact keys.
    #[test]
    fn list_card_shape_is_pinned() {
        let dir = std::env::temp_dir().join(format!("l-main-{}", now_ms()));
        std::fs::create_dir_all(&dir).unwrap();
        let state = dir.join("state.json");
        let mut l = Ledger::new();
        l.add("a pinned goal", Some("why: tests".into()), 86_400_000)
            .unwrap();
        save(&state, &l).unwrap();
        let card: Value = do_list(&state).unwrap();
        let s = serde_json::to_string(&card).unwrap();
        for key in [
            "state_file",
            "open_count",
            "oldest_id",
            "oldest_age",
            "closed_today",
            "open",
        ] {
            assert!(card.get(key).is_some(), "the briefing keys on {key}: {s}");
        }
        assert_eq!(card["open"][0]["text"], "a pinned goal");
        assert!(s.contains("note"), "notes ride along: {s}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn close_errors_type_their_way() {
        let dir = std::env::temp_dir().join(format!("l-err-{}", now_ms()));
        std::fs::create_dir_all(&dir).unwrap();
        let state = dir.join("state.json");
        let hit = do_close(&state, &json!({}), true).await;
        assert!(hit.unwrap_err().contains("id is required"));
        std::fs::write(&state, "{}").unwrap();
        let miss = do_close(&state, &json!({"id": 7}), true).await;
        assert!(miss.unwrap_err().contains("no entry #7"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
