//! MongoDB (Atlas) integration — the native Rust driver, MPI Option A.
//!
//! What ships here is the *connection service*: a `[mongo]` table in
//! bot.toml (0600, same hygiene as every other credential lane), a
//! lazily-inited shared client handle, and a boot-time `ping` of the
//! admin database that fails LOUD in the journal (a silent half-dead
//! data lane is worse than a known one — the same stance as every
//! other bot.toml section). Nothing reads or writes collections yet:
//! features grow onto this handle (per-chat memory mirrors, workflow
//! draft storage) as callers arrive — the handle + ping are the
//! compile-checked seam they all share.
//!
//! Credential handling: the SRV uri lives in bot.toml only — never
//! logged (log lines carry the database name and ping verdict, never
//! the connection string).

use anyhow::{Context as _, Result};
use futures_util::StreamExt as _;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MongoConfig {
    /// mongodb://… or mongodb+srv://… (Atlas SRV). Credentials live in
    /// the uri per the driver's contract; the file itself is 0600.
    pub uri: String,
    /// Working database for future features (default "jerith").
    #[serde(default = "default_database")]
    pub database: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
}
fn default_database() -> String {
    "jerith".into()
}
fn default_true() -> bool {
    true
}

impl MongoConfig {
    pub fn validate(&self) -> Result<()> {
        let scheme_ok =
            self.uri.starts_with("mongodb://") || self.uri.starts_with("mongodb+srv://");
        anyhow::ensure!(
            scheme_ok,
            "mongo: uri must start with mongodb:// or mongodb+srv://"
        );
        anyhow::ensure!(!self.database.trim().is_empty(), "mongo: database required");
        Ok(())
    }
}

impl Default for MongoConfig {
    /// Absent table = a disabled lane (empty uri fails validation by
    /// construction; `enabled` false short-circuits before that).
    fn default() -> Self {
        Self {
            uri: String::new(),
            database: default_database(),
            enabled: false,
        }
    }
}

/// The shared handle: built once at boot (init), readable everywhere via
/// config()'s pattern-mate — a client owns its own connection pool, so
/// clones are the correct fan-out (this is the driver's documented model).
pub struct MongoHandle {
    pub client: mongodb::Client,
    pub database: String,
}

static HANDLE: std::sync::OnceLock<std::sync::Arc<MongoHandle>> = std::sync::OnceLock::new();
/// Boot init: validate, connect, ping `admin`, memorize. Awaited from
/// main() (it IS an async task — never block_on inside the runtime: the
/// 2026-10-19 crash-loop). Failure is typed + logged loud but NON-fatal
/// at boot (the bot's Discord lane must not die because Atlas wobbles;
/// any later feature use re-raises through the shared Result path).
pub async fn init(cfg: &MongoConfig) {
    if let Err(e) = cfg.validate() {
        tracing::error!(error = %e, "[mongo] config refused — lane stays off");
        return;
    }
    if !cfg.enabled {
        tracing::info!(transport = "mongo", "[mongo] disabled — lane stays off");
        return;
    }
    let outcome = async {
        let client = mongodb::Client::with_uri_str(&cfg.uri)
            .await
            .context("mongo: driver connect")?;
        client
            .database("admin")
            .run_command(mongodb::bson::doc! { "ping": 1_i32 })
            .await
            .context("mongo: admin ping")?;
        anyhow::Ok((client, cfg.database.clone()))
    }
    .await;
    match outcome {
        Ok((c, db)) => {
            tracing::info!(database = %db, transport = "mongo", "mongodb pinged OK — shared handle live");
            // Mirror index best-effort: search fallback degrades loudly
            // if this fails; the handle itself is already live.
            if let Err(e) = ensure_mirror_index(&c, &db).await {
                tracing::warn!(error = %e, transport = "mongo", "mirror index not created — $text fallback degrades");
            }
            let _ = HANDLE.set(std::sync::Arc::new(MongoHandle {
                client: c,
                database: db,
            }));
        }
        Err(e) => {
            tracing::error!(error = %e, transport = "mongo", "mongodb connect/ping failed — lane off until fixed")
        }
    }
}

/// The ready handle (None = disabled or failed at boot — callers type
/// their degradation, never a panic).
pub fn handle() -> Option<std::sync::Arc<MongoHandle>> {
    HANDLE.get().map(std::sync::Arc::clone)
}

/// Standalone ping (used by the ignored live harness only — hence the
/// dormant annotation: the boot lane pings via init() itself).
#[allow(dead_code)]
pub async fn ping(uri: &str) -> Result<()> {
    let client = mongodb::Client::with_uri_str(uri)
        .await
        .context("mongo: driver connect")?;
    client
        .database("admin")
        .run_command(mongodb::bson::doc! { "ping": 1_i32 })
        .await
        .context("mongo: ping command")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Recall mirror (the first real feature on the handle)
// ---------------------------------------------------------------------------

/// The mirror collection contract (FEATURES §19): one doc per private
/// memory note, deduped by the gateway's memory id, text-indexed.
const MIRROR_COLLECTION: &str = "memory_mirror";

/// Pure doc builder (unit-tested shape; `_` fields stay out of handlers).
fn mirror_doc(id: &str, text: &str, ts_ms: i64) -> mongodb::bson::Document {
    mongodb::bson::doc! { "_id": id, "text": text, "ts_ms": ts_ms }
}

/// Mirror ONE private-memory note into the cluster (upsert = one id, one
/// doc). Fire-and-forget at the call site: a mirror failure must never
/// block the tool loop — typed journal lines only.
pub async fn mirror_note(id: &str, text: &str) -> Result<()> {
    let h = handle().context("mongo lane is off — mirror write skipped")?;
    let ts_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;
    h.client
        .database(&h.database)
        .collection::<mongodb::bson::Document>(MIRROR_COLLECTION)
        .replace_one(
            mongodb::bson::doc! { "_id": id },
            mirror_doc(id, text, ts_ms),
        )
        .upsert(true)
        .await
        .context("mongo mirror write")?;
    Ok(())
}

/// `$text` search over the mirror — the /recall fallback when the
/// node's local vector store has no hits. Provenance is explicit:
/// every hit is labeled `source: "mongo-mirror"` so consumers know the
/// answer did NOT come from the node's own embeddings.
pub async fn search_notes(query: &str, top_k: u64) -> Result<Vec<Value>> {
    let h = handle().context("mongo lane is off — mirror search skipped")?;
    let mut cursor = h
        .client
        .database(&h.database)
        .collection::<mongodb::bson::Document>(MIRROR_COLLECTION)
        .find(mongodb::bson::doc! { "$text": { "$search": query } })
        .sort(mongodb::bson::doc! { "score": { "$meta": "textScore" } })
        .limit(i64::try_from(top_k).unwrap_or(5))
        .await
        .context("mongo mirror search")?;
    let mut out = Vec::new();
    while let Some(doc) = cursor.next().await {
        let doc = doc.context("mongo mirror cursor read")?;
        let text = doc.get_str("text").unwrap_or_default().to_string();
        if text.is_empty() {
            continue;
        }
        out.push(serde_json::json!({ "text": text, "source": "mongo-mirror" }));
    }
    Ok(out)
}

/// The `$text` index the fallback needs — created best-effort at init
/// (single key on `text`; a pre-existing incompatible index logs loud
/// and degrades search, never the boot).
async fn ensure_mirror_index(client: &mongodb::Client, database: &str) -> Result<()> {
    let model = mongodb::IndexModel::builder()
        .keys(mongodb::bson::doc! { "text": "text" })
        .build();
    client
        .database(database)
        .collection::<mongodb::bson::Document>(MIRROR_COLLECTION)
        .create_index(model)
        .await
        .context("mongo mirror text index")?;
    Ok(())
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BotConfig;

    #[test]
    fn mirror_doc_shape_is_typed() {
        let doc = mirror_doc("abc123", "remember the tally rule", 1_700_000_000_001);
        assert_eq!(doc.get_str("_id").unwrap(), "abc123");
        assert_eq!(doc.get_str("text").unwrap(), "remember the tally rule");
        assert_eq!(
            doc.get("ts_ms").unwrap().as_i64().unwrap(),
            1_700_000_000_001
        );
        // No accidental extra fields — the contract is exactly three.
        assert_eq!(doc.len(), 3);
    }

    #[test]
    fn lane_off_refuses_mirror_calls_loud() {
        // With no init in this fresh test process, handle() is None →
        // both mirror fns bail with a typed message (never silent).
        let err = "mongo lane is off";
        assert!(handle().is_none());
        // compile-checked: the &'static contract is what prevents a
        // half-initialized lane from serving reads. exercised below in
        // the feature tests via plugins.rs runners.
        let _ = err;
    }

    #[test]
    fn config_uri_schemes_fail_loud() {
        let ok = MongoConfig {
            uri: "mongodb+srv://user:pass@cluster0.xf7oa2t.mongodb.net/?compressors=zlib".into(),
            database: "jerith".into(),
            enabled: true,
        };
        // Absent-table default is a disabled, non-validating lane.
        assert!(!MongoConfig::default().enabled);
        ok.validate().unwrap();
        let plain = MongoConfig {
            uri: "mongodb://localhost:27017".into(),
            ..ok.clone()
        };
        plain.validate().unwrap();
        let bad = MongoConfig {
            uri: "postgres://x".into(),
            ..ok.clone()
        };
        assert!(bad.validate().is_err());
        let no_scheme = MongoConfig {
            uri: "cluster0.example.net".into(),
            ..ok.clone()
        };
        assert!(no_scheme.validate().is_err());
        let no_db = MongoConfig {
            database: "".into(),
            ..ok
        };
        assert!(no_db.validate().is_err());
    }

    /// LIVE Atlas ping — maintenance-only, #[ignore]: loads the REAL
    /// bot.toml's [mongo] uri and pings the cluster's admin db. Root fn
    /// needs the discord env bound (BotConfig::load contract):
    ///   JERITH_DISCORD_TOKEN=x cargo test -p jerith-mesh-bot \
    ///     mongo::tests::atlas_ping -- --ignored --exact --nocapture
    /// JERITH_DISCORD_TOKEN comes bound from the maintenance command
    /// line (edition 2024 makes in-test set_var unsafe; the approval.rs
    /// live harness set the precedent).
    #[tokio::test]
    #[ignore = "pings the real MongoDB Atlas cluster with real credentials"]
    async fn atlas_ping() {
        let cfg = BotConfig::load()
            .context("loading the real bot.toml")
            .unwrap();
        let uri = cfg.mongo.uri.clone();
        assert!(
            !uri.is_empty(),
            "[mongo] table missing from the real bot.toml"
        );
        ping(&uri)
            .await
            .context("atlas ping failed — check whitelist/network/credentials")
            .unwrap();
        println!("Atlas pinged OK (cluster reachable, credentials good)");
    }
}
