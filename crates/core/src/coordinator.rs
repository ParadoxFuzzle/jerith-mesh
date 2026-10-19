//! Mesh coordinator (FEATURES "Mesh Coordinator" abstraction layer): ONE
//! shared store that every autonomous surface speaks the same wire shape
//! to — the bot's named workers, remote mesh nodes, any future agent
//! process.
//!
//! Three surfaces, one SQLite database (`coordinator.db` in the node data
//! dir):
//!
//! - **workers** — named agent registry: who exists, which model answers
//!   for it (`ollama:qwen3:8b`, a gateway model id, …), its persona, and
//!   its heartbeat. Distinct named bots share the host because they share
//!   this registry: it is what makes them addressable by name.
//! - **handoff** (the dedicated hand-off channel) — a work queue: `from`
//!   posts a unit of work addressed to a named worker; the worker claims
//!   FIFO, solves it, and posts a result. Claims carry a lease; a worker
//!   that dies mid-job loses the lease (the job requeues), so work never
//!   strands on a vanished agent.
//! - **pool** (the shared memory pool) — namespaced key/value context
//!   blobs. Workers read/write context by convention instead of carrying
//!   it manually between conversations: a delegator stages context under
//!   a key and hands only the key name to the worker.
//!
//! Served on the node gateway under `/v1/coordinator/*` so bot workers
//! and (future) remote nodes reach the same store over plain HTTP. All
//! writes are volume-capped: a hostile or buggy agent cannot pin the
//! store with a giant body or a million keys.

use anyhow::Context as _;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::Path as FsPath;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Volume caps (chars) per surface. Generous for real work units, tiny
/// for abuse: one handoff body ≤ 8k chars, one pool blob ≤ 32k, plus row
/// ceilings on both tables.
pub const HANDOFF_BODY_CAP: usize = 8_000;
pub const HANDOFF_SUBJECT_CAP: usize = 200;
pub const HANDOFF_RESULT_CAP: usize = 32_000;
pub const POOL_VAL_CAP: usize = 32_000;
pub const POOL_KEY_CAP: usize = 200;
pub const PERSONA_CAP: usize = 6_000;
const MAX_HANDOFF_ROWS: i64 = 5_000;
const MAX_POOL_ROWS: i64 = 2_000;
const MAX_WORKERS: i64 = 64;

/// A claim lease: rows claimed and never finished for longer than this
/// requeue on the next claim — matching the observe-act cadence of the
/// bot's worker loop (3-minute solve budget, 5-minute lease).
pub const CLAIM_LEASE_MS: i64 = 300_000;

/// Default TTL for a handoff unit (24 h).
pub const HANDOFF_TTL_MS: i64 = 24 * 3600 * 1000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Worker {
    pub name: String,
    pub model: String,
    pub persona: String,
    pub status: String,
    pub updated_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Handoff {
    pub id: i64,
    pub from: String,
    pub to: String,
    pub subject: String,
    pub body: String,
    pub status: String,
    pub claimed_by: Option<String>,
    pub result: Option<String>,
    pub created_ms: i64,
    pub updated_ms: i64,
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// SQLite coordinator store (WAL + NORMAL — the same availability trade
/// memory.rs accepted for an append-mostly local cache).
pub struct CoordinatorStore {
    conn: Mutex<Connection>,
}

impl CoordinatorStore {
    /// Open (or create) the store at `path`. `":memory:"` keeps its
    /// documented rusqlite behavior — the unit tests use it.
    pub fn open(path: &FsPath) -> anyhow::Result<Self> {
        if path.to_string_lossy() != ":memory:" {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("creating {}", parent.display()))?;
            }
        }
        let conn = Connection::open(path)
            .with_context(|| format!("opening coordinator db {}", path.display()))?;
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             CREATE TABLE IF NOT EXISTS workers (
                 name       TEXT PRIMARY KEY,
                 model      TEXT NOT NULL,
                 persona    TEXT NOT NULL DEFAULT '',
                 status     TEXT NOT NULL DEFAULT 'offline',
                 updated_ms INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS handoffs (
                 id          INTEGER PRIMARY KEY AUTOINCREMENT,
                 from_worker TEXT NOT NULL,
                 to_worker   TEXT NOT NULL,
                 subject     TEXT NOT NULL,
                 body        TEXT NOT NULL,
                 status      TEXT NOT NULL DEFAULT 'created',
                 claimed_by  TEXT,
                 claimed_ms  INTEGER,
                 result      TEXT,
                 created_ms  INTEGER NOT NULL,
                 updated_ms  INTEGER NOT NULL,
                 expires_ms  INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_handoffs_target
                 ON handoffs(to_worker, status, created_ms);
             CREATE TABLE IF NOT EXISTS pool (
                 namespace  TEXT NOT NULL,
                 key        TEXT NOT NULL,
                 val        TEXT NOT NULL,
                 updated_ms INTEGER NOT NULL,
                 PRIMARY KEY (namespace, key)
             );",
        )
        .context("creating coordinator schema")?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    // ------------------------------------------------------------------
    // workers — the named agent registry
    // ------------------------------------------------------------------

    /// Register or update a named worker (rejoin refreshes persona/model).
    pub async fn worker_join(
        &self,
        name: &str,
        model: &str,
        persona: &str,
        status: &str,
    ) -> anyhow::Result<()> {
        let name = name.trim();
        anyhow::ensure!(
            !name.is_empty() && name.chars().count() <= 100,
            "coordinator: worker name must be 1..=100 chars"
        );
        anyhow::ensure!(
            !model.trim().is_empty(),
            "coordinator: worker model required"
        );
        let persona: String = persona.chars().take(PERSONA_CAP).collect();
        let status: String = status.chars().take(24).collect();
        let conn = self.conn.lock().await;
        let exists: Option<bool> = conn
            .query_row("SELECT 1 FROM workers WHERE name = ?1", [name], |_| Ok(true))
            .optional()?;
        if exists.is_none() {
            let count: i64 =
                conn.query_row("SELECT COUNT(*) FROM workers", [], |r| r.get(0))?;
            anyhow::ensure!(
                count < MAX_WORKERS,
                "coordinator: worker registry full ({MAX_WORKERS})"
            );
        }
        conn.execute(
            "INSERT INTO workers (name, model, persona, status, updated_ms)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(name) DO UPDATE SET
               model = excluded.model,
               persona = excluded.persona,
               status = excluded.status,
               updated_ms = excluded.updated_ms",
            rusqlite::params![name, model, persona, status, now_ms()],
        )
        .context("coordinator: worker_join")?;
        Ok(())
    }

    /// Remove a named worker and expire its backlog — no work stands
    /// addressed to a worker that no longer exists.
    pub async fn worker_remove(&self, name: &str) -> anyhow::Result<()> {
        let conn = self.conn.lock().await;
        conn.execute("DELETE FROM workers WHERE name = ?1", [name])
            .context("coordinator: worker_remove")?;
        conn.execute(
            "UPDATE handoffs SET status = 'expired', updated_ms = ?2
             WHERE to_worker = ?1 AND status IN ('created', 'claimed')",
            rusqlite::params![name, now_ms()],
        )
        .context("coordinator: worker_remove backlog")?;
        Ok(())
    }

    pub async fn status_update(&self, name: &str, status: &str) -> anyhow::Result<()> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE workers SET status = ?2, updated_ms = ?3 WHERE name = ?1",
            rusqlite::params![name, status, now_ms()],
        )
        .context("coordinator: status_update")?;
        Ok(())
    }

    pub async fn workers_list(&self) -> anyhow::Result<Vec<Worker>> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT name, model, persona, status, updated_ms FROM workers ORDER BY name",
        )?;
        let rows = stmt
            .query_map([], |r| {
                Ok(Worker {
                    name: r.get(0)?,
                    model: r.get(1)?,
                    persona: r.get(2)?,
                    status: r.get(3)?,
                    updated_ms: r.get(4)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    // ------------------------------------------------------------------
    // handoff — the dedicated hand-off channel
    // ------------------------------------------------------------------

    /// Post one unit of work onto the hand-off channel. `ttl_secs = 0`
    /// means the default TTL; a huge ttl is clamped to 90 days.
    pub async fn handoff_post(
        &self,
        from: &str,
        to: &str,
        subject: &str,
        body: &str,
        ttl_secs: u64,
    ) -> anyhow::Result<i64> {
        anyhow::ensure!(
            !to.trim().is_empty(),
            "coordinator: handoff needs a `to` worker"
        );
        let subject = subject.trim();
        anyhow::ensure!(!subject.is_empty(), "coordinator: handoff subject required");
        let subject: String = subject.chars().take(HANDOFF_SUBJECT_CAP).collect();
        let body = body.trim();
        anyhow::ensure!(!body.is_empty(), "coordinator: handoff body required");
        let body: String = body.chars().take(HANDOFF_BODY_CAP).collect();
        let ttl_ms = if ttl_secs == 0 {
            HANDOFF_TTL_MS
        } else {
            (ttl_secs.min(90 * 24 * 3600) as i64) * 1000
        };
        let t = now_ms();
        let conn = self.conn.lock().await;
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM handoffs WHERE status != 'done'", [], |r| r.get(0))?;
        anyhow::ensure!(
            count < MAX_HANDOFF_ROWS,
            "coordinator: handoff store full ({MAX_HANDOFF_ROWS} open rows)"
        );
        conn.execute(
            "INSERT INTO handoffs (from_worker, to_worker, subject, body, status, created_ms, updated_ms, expires_ms)
             VALUES (?1, ?2, ?3, ?4, 'created', ?5, ?5, ?6)",
            rusqlite::params![from, to, subject, body, t, t + ttl_ms],
        )
        .context("coordinator: handoff_post")?;
        Ok(conn.last_insert_rowid())
    }

    /// Requeue expired-claim leases and expire aged jobs, then FIFO-claim
    /// the oldest open handoff addressed to this worker. None = idle.
    pub async fn handoff_claim(&self, to: &str) -> anyhow::Result<Option<Handoff>> {
        let t = now_ms();
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE handoffs SET status = 'created', claimed_by = NULL, claimed_ms = NULL, updated_ms = ?1
             WHERE status = 'claimed' AND claimed_ms < ?2",
            rusqlite::params![t, t - CLAIM_LEASE_MS],
        )
        .context("coordinator: requeue expired leases")?;
        conn.execute(
            "UPDATE handoffs SET status = 'expired', updated_ms = ?1
             WHERE created_ms + expires_ms < ?1 AND status != 'done'",
            rusqlite::params![t],
        )
        .context("coordinator: expire aged handoffs")?;
        // Select-then-claim: a single UPDATE/subquery can't hand the row
        // id back (last_insert_rowid() tracks INSERTs only), so the id
        // comes out of an explicit SELECT. Single-process Mutex holds the
        // two steps atomic; the status='created' guard makes a re-run idempotent.
        let pick: Option<i64> = conn
            .query_row(
                "SELECT id FROM handoffs
                 WHERE to_worker = ?1 AND status = 'created'
                 ORDER BY created_ms ASC, id ASC LIMIT 1",
                [to],
                |r| r.get(0),
            )
            .optional()
            .context("coordinator: pick open handoff")?;
        let Some(id) = pick else {
            return Ok(None);
        };
        conn.execute(
            "UPDATE handoffs SET status = 'claimed', claimed_by = ?2, claimed_ms = ?3, updated_ms = ?3
             WHERE id = ?1 AND status = 'created'",
            rusqlite::params![id, to, t],
        )
        .context("coordinator: handoff_claim")?;
        if conn.changes() == 0 {
            // Lost a race (another claimer got the id between select and
            // update — only possible across processes on the same db).
            return Ok(None);
        }
        let h = conn
            .query_row(
                "SELECT id, from_worker, to_worker, subject, body, status, claimed_by, result, created_ms, updated_ms
                 FROM handoffs WHERE id = ?1",
                [id],
                row_to_handoff,
            )
            .optional()?
            .ok_or_else(|| anyhow::anyhow!("coordinator: claim lost the row"))?;
        Ok(Some(h))
    }

    /// Post a result for a claimed handoff. The claimer must still hold
    /// the lease (`claimed_by` matches) — a stale worker cannot overwrite
    /// work a takeover requeued.
    pub async fn handoff_finish(&self, id: i64, worker: &str, result: &str) -> anyhow::Result<()> {
        let result: String = result.trim().chars().take(HANDOFF_RESULT_CAP).collect();
        anyhow::ensure!(!result.is_empty(), "coordinator: handoff result required");
        let t = now_ms();
        let conn = self.conn.lock().await;
        let claimed_by: Option<String> = conn
            .query_row(
                "SELECT claimed_by FROM handoffs WHERE id = ?1 AND status = 'claimed'",
                [id],
                |r| r.get(0),
            )
            .optional()?;
        anyhow::ensure!(
            claimed_by.as_deref() == Some(worker),
            "coordinator: not the claimer of handoff {id}"
        );
        conn.execute(
            "UPDATE handoffs SET status = 'done', result = ?2, updated_ms = ?3
             WHERE id = ?1 AND status = 'claimed' AND claimed_by = ?4",
            rusqlite::params![id, result, t, worker],
        )
        .context("coordinator: handoff_finish")?;
        Ok(())
    }

    /// Handoffs addressed to a worker: `status = "open"` (created or
    /// claimed), an exact status word, or "all". Bounded page.
    pub async fn handoffs_for(
        &self,
        worker: &str,
        status: &str,
        limit: i64,
    ) -> anyhow::Result<Vec<Handoff>> {
        let limit = limit.clamp(1, 50);
        let conn = self.conn.lock().await;
        let (sql, params): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = match status {
            "open" => (
                "SELECT id, from_worker, to_worker, subject, body, status, claimed_by, result, created_ms, updated_ms
                 FROM handoffs WHERE to_worker = ?1 AND status IN ('created','claimed')
                 ORDER BY created_ms DESC LIMIT ?2"
                    .into(),
                vec![Box::new(worker.to_string()), Box::new(limit)],
            ),
            "all" => (
                "SELECT id, from_worker, to_worker, subject, body, status, claimed_by, result, created_ms, updated_ms
                 FROM handoffs WHERE to_worker = ?1 ORDER BY created_ms DESC LIMIT ?2"
                    .into(),
                vec![Box::new(worker.to_string()), Box::new(limit)],
            ),
            other => (
                "SELECT id, from_worker, to_worker, subject, body, status, claimed_by, result, created_ms, updated_ms
                 FROM handoffs WHERE to_worker = ?1 AND status = ?2 ORDER BY created_ms DESC LIMIT ?3"
                    .into(),
                vec![
                    Box::new(worker.to_string()),
                    Box::new(other.to_string()),
                    Box::new(limit),
                ],
            ),
        };
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(params.as_slice().iter().map(|p| p.as_ref()).collect::<Vec<_>>().as_slice(), row_to_handoff)?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Handoffs POSTED BY an actor (`from_worker`), newest first — the
    /// delegator's result-pickup view: `bot.delegate` then `bot.inbox`
    /// reads the finished rows' results back.
    pub async fn handoffs_from(&self, from: &str, limit: i64) -> anyhow::Result<Vec<Handoff>> {
        let limit = limit.clamp(1, 50);
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, from_worker, to_worker, subject, body, status, claimed_by, result, created_ms, updated_ms
             FROM handoffs WHERE from_worker = ?1 ORDER BY id DESC LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(rusqlite::params![from, limit], row_to_handoff)?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    // ------------------------------------------------------------------
    // pool — the shared memory pool
    // ------------------------------------------------------------------

    /// Upsert a namespaced context blob. At the row cap the oldest
    /// entries (never the one being upserted) are pruned.
    pub async fn pool_put(&self, ns: &str, key: &str, val: &str) -> anyhow::Result<()> {
        let ns = ns.trim();
        anyhow::ensure!(
            !ns.is_empty() && ns.chars().count() <= 100,
            "coordinator: pool namespace must be 1..=100 chars"
        );
        let key: String = key.trim().chars().take(POOL_KEY_CAP).collect();
        anyhow::ensure!(!key.is_empty(), "coordinator: pool key required");
        let val = val.trim();
        anyhow::ensure!(!val.is_empty(), "coordinator: pool val required");
        let val: String = val.chars().take(POOL_VAL_CAP).collect();
        let t = now_ms();
        let conn = self.conn.lock().await;
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM pool", [], |r| r.get(0))?;
        if count >= MAX_POOL_ROWS {
            conn.execute(
                "DELETE FROM pool WHERE rowid IN (
                   SELECT rowid FROM pool
                   WHERE NOT (namespace = ?1 AND key = ?2)
                   ORDER BY updated_ms ASC LIMIT ?3
                 )",
                rusqlite::params![ns, key, count - MAX_POOL_ROWS + 1],
            )
            .context("coordinator: pool prune")?;
        }
        conn.execute(
            "INSERT INTO pool (namespace, key, val, updated_ms) VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(namespace, key) DO UPDATE SET val = excluded.val, updated_ms = excluded.updated_ms",
            rusqlite::params![ns, key, val, t],
        )
        .context("coordinator: pool_put")?;
        Ok(())
    }

    pub async fn pool_get(&self, ns: &str, key: &str) -> anyhow::Result<Option<String>> {
        let conn = self.conn.lock().await;
        let v: Option<String> = conn
            .query_row(
                "SELECT val FROM pool WHERE namespace = ?1 AND key = ?2",
                rusqlite::params![ns, key],
                |r| r.get(0),
            )
            .optional()?;
        Ok(v)
    }

    pub async fn pool_keys(&self, ns: &str) -> anyhow::Result<Vec<String>> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT key FROM pool WHERE namespace = ?1 ORDER BY updated_ms DESC LIMIT 200",
        )?;
        let rows = stmt
            .query_map([ns], |r| r.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }
}

fn row_to_handoff(r: &rusqlite::Row<'_>) -> rusqlite::Result<Handoff> {
    Ok(Handoff {
        id: r.get(0)?,
        from: r.get(1)?,
        to: r.get(2)?,
        subject: r.get(3)?,
        body: r.get(4)?,
        status: r.get(5)?,
        claimed_by: r.get(6)?,
        result: r.get(7)?,
        created_ms: r.get(8)?,
        updated_ms: r.get(9)?,
    })
}

// ---------------------------------------------------------------------------
// Wire surface — the mesh coordinator abstraction over HTTP
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct WorkerJoinBody {
    name: String,
    model: String,
    #[serde(default)]
    persona: String,
    #[serde(default = "default_join_status")]
    status: String,
}

fn default_join_status() -> String {
    "idle".into()
}

#[derive(Deserialize)]
struct HandoffPostBody {
    from: String,
    to: String,
    subject: String,
    body: String,
    #[serde(default)]
    ttl_secs: u64,
}

#[derive(Deserialize)]
struct HandoffFinishBody {
    id: i64,
    worker: String,
    result: String,
}

#[derive(Deserialize)]
struct PoolPutBody {
    ns: String,
    key: String,
    val: String,
}

fn err_msg(e: impl std::fmt::Display) -> Response {
    (StatusCode::BAD_REQUEST, Json(json!({"error": e.to_string()}))).into_response()
}

fn list_of(rows: Vec<Handoff>) -> Response {
    (
        StatusCode::OK,
        Json(serde_json::to_value(rows).unwrap_or_default()),
    )
        .into_response()
}

/// The coordinator HTTP router — `.merge()`d into the gateway's router
/// when the node has a coordinator store configured.
pub fn router(store: Arc<CoordinatorStore>) -> Router {
    Router::new()
        .route("/v1/coordinator/workers/join", post(worker_join))
        .route("/v1/coordinator/workers/list", get(workers_list))
        .route("/v1/coordinator/workers/remove", post(worker_remove_h))
        .route("/v1/coordinator/handoff/post", post(handoff_post_h))
        .route("/v1/coordinator/handoff/claim", post(handoff_claim_h))
        .route("/v1/coordinator/handoff/finish", post(handoff_finish_h))
        .route("/v1/coordinator/handoff/for", post(handoff_for_h))
        .route("/v1/coordinator/handoff/from", post(handoff_from_h))
        .route("/v1/coordinator/pool/put", post(pool_put_h))
        .route("/v1/coordinator/pool/get", get(pool_get_h))
        .route("/v1/coordinator/pool/keys", get(pool_keys_h))
        .with_state(store)
}

async fn worker_join(
    State(store): State<Arc<CoordinatorStore>>,
    Json(b): Json<WorkerJoinBody>,
) -> Response {
    match store.worker_join(&b.name, &b.model, &b.persona, &b.status).await {
        Ok(_) => (StatusCode::OK, Json(json!({"ok": true}))).into_response(),
        Err(e) => err_msg(e),
    }
}

async fn workers_list(State(store): State<Arc<CoordinatorStore>>) -> Response {
    match store.workers_list().await {
        Ok(ws) => (StatusCode::OK, Json(json!({"workers": ws}))).into_response(),
        Err(e) => err_msg(e),
    }
}

async fn worker_remove_h(
    State(store): State<Arc<CoordinatorStore>>,
    Json(b): Json<serde_json::Value>,
) -> Response {
    let Some(name) = b.get("name").and_then(|v| v.as_str()) else {
        return err_msg("field 'name' is required");
    };
    match store.worker_remove(name).await {
        Ok(_) => (StatusCode::OK, Json(json!({"ok": true}))).into_response(),
        Err(e) => err_msg(e),
    }
}

async fn handoff_post_h(
    State(store): State<Arc<CoordinatorStore>>,
    Json(b): Json<HandoffPostBody>,
) -> Response {
    match store.handoff_post(&b.from, &b.to, &b.subject, &b.body, b.ttl_secs).await {
        Ok(id) => (StatusCode::OK, Json(json!({"id": id}))).into_response(),
        Err(e) => err_msg(e),
    }
}

async fn handoff_claim_h(
    State(store): State<Arc<CoordinatorStore>>,
    Json(b): Json<serde_json::Value>,
) -> Response {
    let Some(worker) = b.get("worker").and_then(|v| v.as_str()) else {
        return err_msg("field 'worker' is required");
    };
    match store.handoff_claim(worker).await {
        Ok(Some(h)) => (
            StatusCode::OK,
            Json(serde_json::to_value(h).unwrap_or_default()),
        )
            .into_response(),
        Ok(None) => (StatusCode::OK, Json(json!({"id": null}))).into_response(),
        Err(e) => err_msg(e),
    }
}

async fn handoff_finish_h(
    State(store): State<Arc<CoordinatorStore>>,
    Json(b): Json<HandoffFinishBody>,
) -> Response {
    match store.handoff_finish(b.id, &b.worker, &b.result).await {
        Ok(_) => (StatusCode::OK, Json(json!({"ok": true}))).into_response(),
        Err(e) => err_msg(e),
    }
}

async fn handoff_for_h(
    State(store): State<Arc<CoordinatorStore>>,
    Json(b): Json<serde_json::Value>,
) -> Response {
    let Some(worker) = b.get("worker").and_then(|v| v.as_str()) else {
        return err_msg("field 'worker' is required");
    };
    let status = b.get("status").and_then(|v| v.as_str()).unwrap_or("open");
    let limit = b.get("limit").and_then(|v| v.as_i64()).unwrap_or(20);
    match store.handoffs_for(worker, status, limit).await {
        Ok(rows) => list_of(rows),
        Err(e) => err_msg(e),
    }
}

async fn handoff_from_h(
    State(store): State<Arc<CoordinatorStore>>,
    Json(b): Json<serde_json::Value>,
) -> Response {
    let Some(from) = b.get("from").and_then(|v| v.as_str()) else {
        return err_msg("field 'from' is required");
    };
    let limit = b.get("limit").and_then(|v| v.as_i64()).unwrap_or(20);
    match store.handoffs_from(from, limit).await {
        Ok(rows) => list_of(rows),
        Err(e) => err_msg(e),
    }
}

async fn pool_put_h(
    State(store): State<Arc<CoordinatorStore>>,
    Json(b): Json<PoolPutBody>,
) -> Response {
    match store.pool_put(&b.ns, &b.key, &b.val).await {
        Ok(_) => (StatusCode::OK, Json(json!({"ok": true}))).into_response(),
        Err(e) => err_msg(e),
    }
}

async fn pool_get_h(
    State(store): State<Arc<CoordinatorStore>>,
    Query(q): Query<std::collections::HashMap<String, String>>,
) -> Response {
    let Some(ns) = q.get("ns") else {
        return err_msg("query param 'ns' required");
    };
    let Some(key) = q.get("key") else {
        return err_msg("query param 'key' required");
    };
    match store.pool_get(ns, key).await {
        Ok(Some(v)) => (StatusCode::OK, Json(json!({"val": v}))).into_response(),
        Ok(None) => (StatusCode::OK, Json(json!({"val": null}))).into_response(),
        Err(e) => err_msg(e),
    }
}

async fn pool_keys_h(
    State(store): State<Arc<CoordinatorStore>>,
    Query(q): Query<std::collections::HashMap<String, String>>,
) -> Response {
    let Some(ns) = q.get("ns") else {
        return err_msg("query param 'ns' required");
    };
    match store.pool_keys(ns).await {
        Ok(keys) => (StatusCode::OK, Json(json!({"keys": keys}))).into_response(),
        Err(e) => err_msg(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn store() -> CoordinatorStore {
        CoordinatorStore::open(FsPath::new(":memory:")).unwrap()
    }

    async fn post(store: &CoordinatorStore, from: &str, to: &str, subject: &str, body: &str) -> i64 {
        store.handoff_post(from, to, subject, body, 0).await.unwrap()
    }

    #[tokio::test]
    async fn worker_registry_upserts_and_lists() {
        let s = store();
        s.worker_join("alpha", "ollama:llama3.1:8b", "does arithmetic", "idle")
            .await
            .unwrap();
        s.worker_join("beta", "remote:gpt-4o-mini", "", "idle").await.unwrap();
        // Rejoin updates persona + never duplicates.
        s.worker_join("alpha", "ollama:llama3.1:8b", "now does MORE arithmetic", "idle")
            .await
            .unwrap();
        let ws = s.workers_list().await.unwrap();
        assert_eq!(ws.len(), 2, "no duplicate join");
        let alpha = &ws[0];
        assert_eq!(alpha.name, "alpha", "sorted by name");
        assert!(alpha.persona.contains("MORE"), "persona was updated");
    }

    #[tokio::test]
    async fn handoff_fifo_claim_finish_and_idle_claim() {
        let s = store();
        let a = post(&s, "main", "alpha", "first", "compute 2+2").await;
        let b = post(&s, "main", "alpha", "second", "compute 3+3").await;
        let h = s.handoff_claim("alpha").await.unwrap().unwrap();
        assert_eq!(h.id, a, "FIFO: oldest first");
        assert_eq!(h.status, "claimed");
        assert_eq!(h.claimed_by.as_deref(), Some("alpha"));
        // Wrong claimer refused, right claimer finishes.
        assert!(s.handoff_finish(a, "beta", "x").await.is_err());
        s.handoff_finish(a, "alpha", "4").await.unwrap();
        let next = s.handoff_claim("alpha").await.unwrap().unwrap();
        assert_eq!(next.id, b);
        s.handoff_finish(b, "alpha", "6").await.unwrap();
        assert!(s.handoff_claim("alpha").await.unwrap().is_none(), "idle claim");
        // A finished job is exactly the result the delegator reads back.
        let done = s.handoffs_for("alpha", "done", 10).await.unwrap();
        assert_eq!(done.len(), 2);
        assert!(done.iter().any(|h| h.id == a && h.result.as_deref() == Some("4")));
    }

    #[tokio::test]
    async fn expired_lease_requeues_and_never_strands_work() {
        let s = store();
        let id = post(&s, "main", "alpha", "vital", "unblock the pipe").await;
        // Force an ancient claim (the ghost-worker shape): the lease logic
        // must requeue it for the live claimer.
        {
            let conn = s.conn.lock().await;
            conn.execute(
                "UPDATE handoffs SET status = 'claimed', claimed_by = 'ghost', claimed_ms = 1, updated_ms = 1 WHERE id = ?1",
                rusqlite::params![id],
            )
            .unwrap();
        }
        let h = s.handoff_claim("alpha").await.unwrap().unwrap();
        assert_eq!(h.id, id);
        assert_eq!(h.claimed_by.as_deref(), Some("alpha"), "stale lease reclaimed");
        // A short-ttl job expires out of the queue entirely.
        let short = s
            .handoff_post("main", "alpha", "ephemeral", "skippable body", 1)
            .await
            .unwrap();
        {
            let conn = s.conn.lock().await;
            conn.execute(
                "UPDATE handoffs SET created_ms = created_ms - 10_000, expires_ms = 1 WHERE id = ?1",
                rusqlite::params![short],
            )
            .unwrap();
        }
        // While the valid lease on `vital` is held, the next claim is idle
        // (the short job expired;`vital` is claimed, not stranded).
        assert!(s.handoff_claim("alpha").await.unwrap().is_none(), "held lease not re-claimed");
        // The ghost shape again: expire `vital`'s lease → claimable.
        {
            let conn = s.conn.lock().await;
            conn.execute(
                "UPDATE handoffs SET claimed_ms = 1, updated_ms = 1 WHERE id = ?1",
                rusqlite::params![id],
            )
            .unwrap();
        }
        let h = s.handoff_claim("alpha").await.unwrap().unwrap();
        assert_eq!(h.id, id, "stale lease requeued, job not stranded");
        let all = s.handoffs_for("alpha", "all", 10).await.unwrap();
        assert!(
            all.iter().any(|h| h.id == short && h.status == "expired"),
            "expired status visible"
        );
    }

    #[tokio::test]
    async fn pool_is_namespaced_and_capped() {
        let s = store();
        s.pool_put("ctx:invoice", "vendor", "Acme Corp").await.unwrap();
        s.pool_put("ctx:invoice", "amount", "42").await.unwrap();
        s.pool_put("ctx:other", "vendor", "different context").await.unwrap();
        assert_eq!(
            s.pool_get("ctx:invoice", "vendor").await.unwrap().as_deref(),
            Some("Acme Corp")
        );
        assert_eq!(s.pool_get("ctx:invoice", "amount").await.unwrap().as_deref(), Some("42"));
        s.pool_put("ctx:invoice", "vendor", "AcmeUpdated").await.unwrap();
        assert_eq!(
            s.pool_get("ctx:invoice", "vendor").await.unwrap().as_deref(),
            Some("AcmeUpdated")
        );
        let keys = s.pool_keys("ctx:invoice").await.unwrap();
        assert_eq!(keys.len(), 2, "namespace's own keys listed");
        assert!(!keys.contains(&"different context".to_string()), "values never leak as keys");
        assert!(s.pool_put("ctx:invoice", "", "x").await.is_err(), "empty key refused");
        assert!(s.pool_put("", "k", "x").await.is_err(), "empty ns refused");
        assert!(s.pool_put("ns", "k", "").await.is_err(), "empty val refused");
    }

    #[tokio::test]
    async fn volume_caps_and_requirements_honest() {
        let s = store();
        assert!(s.handoff_post("main", "alpha", "", "body", 0).await.is_err(), "subject required");
        assert!(s.handoff_post("main", "alpha", "subj", "", 0).await.is_err(), "body required");
        assert!(s.handoff_post("main", "", "subj", "body", 0).await.is_err(), "to required");
        // Oversize truncates (documented cap semantics), never a 500.
        let big = "x".repeat(HANDOFF_BODY_CAP + 200);
        let id = s.handoff_post("main", "alpha", "trim", &big, 0).await.unwrap();
        {
            let conn = s.conn.lock().await;
            let len: i64 = conn
                .query_row("SELECT LENGTH(body) FROM handoffs WHERE id=?1", [id], |r| r.get(0))
                .unwrap();
            assert_eq!(len as usize, HANDOFF_BODY_CAP, "body truncated to cap");
        }
        // A removed worker expires its backlog — nothing stands open.
        s.worker_join("delta", "m", "", "idle").await.unwrap();
        post(&s, "main", "delta", "task", "for delta").await;
        s.worker_remove("delta").await.unwrap();
        assert!(s.handoff_claim("delta").await.unwrap().is_none());
        let ws = s.workers_list().await.unwrap();
        assert!(ws.iter().all(|w| w.name != "delta"), "removed worker leaves the registry");
    }

    /// HTTP surface: join → list → post → claim → finish → for → idle
    /// claim, plus the pool round-trip over GET query params.
    #[tokio::test]
    async fn http_wire_surface_roundtrips() {
        let dir = std::env::temp_dir().join(format!("jerith-coord-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let store = Arc::new(CoordinatorStore::open(&dir.join("coord.db")).unwrap());
        let t = axum_test::TestServer::new(router(store)).unwrap();

        let r = t
            .post("/v1/coordinator/workers/join")
            .json(&json!({
                "name": "plumber",
                "model": "ollama:agents/plumber:7b",
                "persona": "fixes pipes",
            }))
            .await;
        assert_eq!(r.status_code(), StatusCode::OK, "join accepted: {}", r.text());

        let r = t.get("/v1/coordinator/workers/list").await;
        assert!(r.text().contains("plumber"), "workers list carries the join");

        let r = t
            .post("/v1/coordinator/handoff/post")
            .json(&json!({
                "from": "main",
                "to": "plumber",
                "subject": "fix",
                "body": "the east wing leaks",
            }))
            .await;
        let posted: serde_json::Value = serde_json::from_str(&r.text()).unwrap();
        let id = posted["id"].as_i64().expect("post returns an id");

        let r = t
            .post("/v1/coordinator/handoff/claim")
            .json(&json!({"worker": "plumber"}))
            .await;
        let claimed: serde_json::Value = serde_json::from_str(&r.text()).unwrap();
        assert_eq!(claimed["id"], json!(id), "FIFO claim returns the open job");
        assert_eq!(claimed["status"], json!("claimed"));

        t.post("/v1/coordinator/handoff/finish")
            .json(&json!({"id": id, "worker": "plumber", "result": "welded"}))
            .await;

        let r = t
            .post("/v1/coordinator/handoff/for")
            .json(&json!({"worker": "plumber", "status": "done"}))
            .await;
        assert!(r.text().contains("welded"), "done list carries the result");

        // Idle claim with no open jobs → honest null, not an error.
        let r = t
            .post("/v1/coordinator/handoff/claim")
            .json(&json!({"worker": "plumber"}))
            .await;
        assert!(r.text().contains("\"id\":null"), "idle claim honest: {}", r.text());

        // Pool over the wire: put (POST), get (GET), namespaced.
        t.post("/v1/coordinator/pool/put")
            .json(&json!({"ns": "ctx:demo", "key": "vendor", "val": "Acme"}))
            .await;
        let r = t.get("/v1/coordinator/pool/get?ns=ctx:demo&key=vendor").await;
        assert!(r.text().contains("Acme"), "pool get: {}", r.text());
    }
}
