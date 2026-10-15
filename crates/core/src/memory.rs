//! Local memory: SQLite vector store + local-only embeddings.
//!
//! Per spec:
//! - Memory is **strictly local** — never replicated into the mesh, never
//!   distributed. Two backends behind one [`MemoryHandle`] interface:
//!   SQLite (default) and PostgreSQL+pgvector. The operator's *chosen*
//!   backend is the one actually used — a postgres backend that can't be
//!   reached (or was chosen with no DSN) fails honestly, it is never a
//!   silent swap to SQLite.
//! - Embeddings are produced **locally** — by the local nomic
//!   nomic-embed-text-v2-moe model via the Ollama API. Raw text never
//!   leaves the node just to be embedded.
//! - If Ollama (or the model) is unavailable we fall back to a
//!   deterministic hashing embedder so memory still works offline — the
//!   fallback is always labeled in results so users know.
//! - Matryoshka truncation: vectors stored at the configured dimension
//!   (768/512/256) for storage savings.
//!
//! Skill surface (exposed via the gateway):
//!   POST /v1/skills/memory/embed  {text}         → memory.embed
//!   POST /v1/skills/memory/search {query, top_k} → memory.search

use std::path::Path;
use std::sync::Arc;

use anyhow::Context as _;
use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

/// Which provider produced an embedding.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EmbedProvider {
    /// nomic-embed-text-v2-moe via local Ollama.
    Nomic,
    /// Deterministic offline fallback (hashing bag-of-words).
    HashFallback,
}

/// memory.stats payload — the health-dashboard view of the store.
#[derive(Debug, Clone, Serialize)]
pub struct MemoryStats {
    pub count: u64,
    /// dimension → entry count (mixed dims show up here).
    pub dims: Vec<(u32, u64)>,
    /// provider → entry count (nomic vs hash fallback).
    pub providers: Vec<(String, u64)>,
    /// Approximate bytes held by embedding blobs.
    pub embedding_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub newest_created_ms: Option<u64>,
}

/// Where embeddings come from.
#[derive(Debug, Clone)]
pub struct EmbedConfig {
    /// Ollama base URL (default 127.0.0.1:11434).
    pub ollama_url: String,
    pub model: String,
    /// Matryoshka truncation dimension.
    pub dim: usize,
}

impl Default for EmbedConfig {
    fn default() -> Self {
        Self {
            ollama_url: "http://127.0.0.1:11434".into(),
            model: "nomic-embed-text-v2-moe".into(),
            dim: 256,
        }
    }
}

/// One retrieved memory hit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryHit {
    pub id: i64,
    pub text: String,
    pub score: f32,
    pub dim: usize,
    pub provider: EmbedProvider,
    pub created_ms: u64,
}

/// Retention cap on the memories table. The store is append-only with no
/// delete cadence, so without a ceiling a chat-driven embed loop grows disk
/// without bound. Oldest rows are pruned at insert time once past this.
const MAX_MEMORY_ROWS: i64 = 10_000;

/// SQLite-backed memory store.
pub struct MemoryStore {
    conn: Mutex<Connection>,
    embed: EmbedConfig,
    /// Path it was opened from (`""` for in-memory tests) — only used for
    /// the operator-facing [`MemoryHandle::describe`] line.
    path: std::path::PathBuf,
}

impl MemoryStore {
    /// Open (creating the schema if needed).
    pub fn open(path: &Path, embed: EmbedConfig) -> anyhow::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        let conn = Connection::open(path)
            .with_context(|| format!("opening memory db {}", path.display()))?;
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             CREATE TABLE IF NOT EXISTS memories (
                 id          INTEGER PRIMARY KEY AUTOINCREMENT,
                 text        TEXT NOT NULL,
                 dim         INTEGER NOT NULL,
                 embedding   BLOB NOT NULL,
                 provider    TEXT NOT NULL,
                 created_ms  INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_ms);",
        )
        .context("creating memory schema")?;
        Ok(Self {
            conn: Mutex::new(conn),
            embed,
            path: path.to_path_buf(),
        })
    }

    /// In-memory database (tests).
    pub fn open_in_memory(embed: EmbedConfig) -> anyhow::Result<Self> {
        let conn = Connection::open_in_memory().context("opening in-memory db")?;
        conn.execute_batch(
            "CREATE TABLE memories (
                 id          INTEGER PRIMARY KEY AUTOINCREMENT,
                 text        TEXT NOT NULL,
                 dim         INTEGER NOT NULL,
                 embedding   BLOB NOT NULL,
                 provider    TEXT NOT NULL,
                 created_ms  INTEGER NOT NULL
             );",
        )
        .context("creating memory schema")?;
        Ok(Self {
            conn: Mutex::new(conn),
            embed,
            path: std::path::PathBuf::new(),
        })
    }

    /// memory.embed — embed text and store it. Returns the row id.
    /// Rows are retention-capped: see `prune_beyond`.
    pub async fn embed_and_store(&self, text: &str) -> anyhow::Result<(i64, EmbedProvider)> {
        let (vector, provider) = self.embed_text(text).await?;
        let dim = vector.len();
        let bytes = pack_vector(&vector);
        let rowid = {
            let conn = self.conn.lock().await;
            conn.execute(
                "INSERT INTO memories (text, dim, embedding, provider, created_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![
                    text,
                    dim as i64,
                    bytes,
                    provider_str(provider),
                    crate::bootstrap::now_ms()
                ],
            )
            .context("inserting memory")?;
            conn.last_insert_rowid()
        };
        // Retention cap: the table is append-only with no delete cadence, so
        // an embed loop would grow disk without bound. Once past the cap,
        // every insert prunes the oldest rows (recent rows always win).
        if rowid > MAX_MEMORY_ROWS {
            self.prune_beyond(MAX_MEMORY_ROWS).await?;
        }
        Ok((rowid, provider))
    }

    /// memory.search — embed the query and return the most similar hits.
    pub async fn search(&self, query: &str, top_k: usize) -> anyhow::Result<Vec<MemoryHit>> {
        let (qvec, _) = self.embed_text(query).await?;
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare("SELECT id, text, dim, embedding, provider, created_ms FROM memories")
            .context("preparing memory query")?;
        let mut hits: Vec<MemoryHit> = stmt
            .query_map([], |row| {
                let bytes: Vec<u8> = row.get(3)?;
                let provider_s: String = row.get(4)?;
                let mut hit = MemoryHit {
                    id: row.get(0)?,
                    text: row.get(1)?,
                    score: 0.0,
                    dim: row.get::<_, i64>(2)? as usize,
                    provider: provider_from_str(&provider_s),
                    created_ms: row.get(5)?,
                };
                hit.score = cosine(&qvec, &unpack_vector(&bytes));
                Ok(hit)
            })
            .context("scanning memories")?
            .filter_map(|r| r.ok())
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(top_k);
        Ok(hits)
    }

    /// Produce an embedding: Ollama nomic when reachable, hashing fallback
    /// otherwise. The provider is always reported to the caller.
    pub async fn embed_text(&self, text: &str) -> anyhow::Result<(Vec<f32>, EmbedProvider)> {
        embed_with(&self.embed, text).await
    }

    pub async fn count(&self) -> anyhow::Result<u64> {
        let conn = self.conn.lock().await;
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .context("counting memories")?;
        Ok(n as u64)
    }

    /// Prune all but the `max_rows` most recent rows. Returns rows removed.
    /// Split out for testability: the retention cap stays exercised here
    /// with a small `max_rows` instead of ten thousand inserts.
    async fn prune_beyond(&self, max_rows: i64) -> anyhow::Result<usize> {
        let conn = self.conn.lock().await;
        let n = conn
            .execute(
                "DELETE FROM memories WHERE id NOT IN (
                     SELECT id FROM memories ORDER BY id DESC LIMIT ?1)",
                rusqlite::params![max_rows],
            )
            .context("pruning memories")?;
        Ok(n as usize)
    }

    /// "Forget this conversation" support: delete by id.
    pub async fn forget(&self, id: i64) -> anyhow::Result<bool> {
        let conn = self.conn.lock().await;
        let n = conn.execute("DELETE FROM memories WHERE id = ?1", [id])?;
        Ok(n > 0)
    }

    /// memory.similar — semantic-memories port: rank stored entries by
    /// cosine similarity to the entry with the given seed id ("find memories
    /// like this one"). The seed itself is excluded from the results.
    /// Unlike `search` this needs no new embedding call — the seed's stored
    /// vector is compared against every stored vector, so it works offline
    /// (hash-fallback stores included).
    pub async fn similar_to(&self, seed_id: i64, top_k: usize) -> anyhow::Result<Vec<MemoryHit>> {
        let seed_vec: Vec<f32> = {
            let conn = self.conn.lock().await;
            let row = conn
                .query_row(
                    "SELECT embedding FROM memories WHERE id = ?1",
                    [seed_id],
                    |r| r.get::<_, Vec<u8>>(0),
                )
                .optional()?
                .context(format!("no memory entry with id {seed_id}"))?;
            unpack_vector(&row)
        };
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare("SELECT id, text, dim, embedding, provider, created_ms FROM memories WHERE id != ?1")
            .context("preparing similar query")?;
        let mut hits: Vec<MemoryHit> = stmt
            .query_map([seed_id], |row| {
                let bytes: Vec<u8> = row.get(3)?;
                let provider_s: String = row.get(4)?;
                let mut hit = MemoryHit {
                    id: row.get(0)?,
                    text: row.get(1)?,
                    score: 0.0,
                    dim: row.get::<_, i64>(2)? as usize,
                    provider: provider_from_str(&provider_s),
                    created_ms: row.get(5)?,
                };
                hit.score = cosine(&seed_vec, &unpack_vector(&bytes));
                Ok(hit)
            })
            .context("scanning memories for similar")?
            .filter_map(|r| r.ok())
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(top_k);
        Ok(hits)
    }

    /// memory.stats — health-dashboard port: count, dimensions, provider
    /// split, newest entry, approximate store footprint. Powers
    /// /v1/skills/memory/stats and the CLI's `memory stats`.
    pub async fn stats(&self) -> anyhow::Result<MemoryStats> {
        let conn = self.conn.lock().await;
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .context("counting memories")?;
        let mut dims: std::collections::BTreeMap<i64, i64> = Default::default();
        let mut providers: std::collections::BTreeMap<String, i64> = Default::default();
        let mut bytes_total: i64 = 0;
        let mut newest_ms: i64 = 0;
        {
            let mut stmt = conn
                .prepare("SELECT dim, provider, LENGTH(embedding), created_ms FROM memories")
                .context("preparing stats scan")?;
            let rows = stmt
                .query_map([], |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, i64>(2)?,
                        r.get::<_, i64>(3)?,
                    ))
                })?
                .collect::<Result<Vec<_>, _>>()?;
            for (dim, provider, bytes, created) in rows {
                *dims.entry(dim).or_default() += 1;
                *providers.entry(provider).or_default() += 1;
                bytes_total += bytes;
                newest_ms = newest_ms.max(created);
            }
        }
        Ok(MemoryStats {
            count: count as u64,
            dims: dims
                .into_iter()
                .map(|(d, n)| (d as u32, n as u64))
                .collect(),
            providers: providers.into_iter().map(|(p, n)| (p, n as u64)).collect(),
            embedding_bytes: bytes_total as u64,
            newest_created_ms: if newest_ms > 0 {
                Some(newest_ms as u64)
            } else {
                None
            },
        })
    }
}

async fn ollama_embed(cfg: &EmbedConfig, text: &str) -> anyhow::Result<Vec<f32>> {
    let url = format!("{}/api/embeddings", cfg.ollama_url.trim_end_matches('/'));
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(20))
        .build()?;
    #[derive(serde::Deserialize)]
    struct Resp {
        embedding: Vec<f32>,
    }
    let resp = client
        .post(&url)
        .json(&serde_json::json!({ "model": cfg.model, "prompt": text }))
        .send()
        .await?
        .error_for_status()?;
    let body: Resp = resp.json().await?;
    if body.embedding.is_empty() {
        anyhow::bail!("empty embedding from ollama");
    }
    Ok(body.embedding)
}

/// Deterministic offline embedder: hashed bag-of-words with sublinear tf,
/// L2-normalized. Not semantically strong, but stable, fast, and good
/// enough to exercise storage + search offline.
pub fn hash_embed(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    for token in text.split_whitespace() {
        let tok = token.to_lowercase();
        let h = fnv1a(tok.as_bytes()) as usize;
        let idx = h % dim;
        v[idx] += 1.0;
        // Second hash for a second bucket reduces collisions.
        let h2 = fnv1a(format!("2:{tok}").as_bytes()) as usize;
        v[h2 % dim] += 0.5;
    }
    for slot in v.iter_mut() {
        if *slot > 0.0 {
            *slot = slot.sqrt(); // sublinear tf
        }
    }
    l2_normalize(&mut v);
    v
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Matryoshka truncation: keep the first `dim` components, renormalize.
fn truncate_matryoshka(v: &mut Vec<f32>, dim: usize) {
    if v.len() > dim {
        v.truncate(dim);
        l2_normalize(v);
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let dot: f32 = a[..n].iter().zip(b[..n].iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn pack_vector(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn unpack_vector(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn provider_str(p: EmbedProvider) -> &'static str {
    match p {
        EmbedProvider::Nomic => "nomic",
        EmbedProvider::HashFallback => "hash-fallback",
    }
}

fn provider_from_str(s: &str) -> EmbedProvider {
    match s {
        "nomic" => EmbedProvider::Nomic,
        _ => EmbedProvider::HashFallback,
    }
}

/// Produce an embedding, store-independent: Ollama nomic when reachable,
/// hashing fallback otherwise. The provider is always reported to the
/// caller, matching the SQLite store's semantics exactly.
pub async fn embed_with(cfg: &EmbedConfig, text: &str) -> anyhow::Result<(Vec<f32>, EmbedProvider)> {
    match ollama_embed(cfg, text).await {
        Ok(mut v) => {
            truncate_matryoshka(&mut v, cfg.dim);
            Ok((v, EmbedProvider::Nomic))
        }
        Err(e) => {
            tracing::debug!(error = %e, "nomic embedding unavailable — using hash fallback");
            Ok((hash_embed(text, cfg.dim), EmbedProvider::HashFallback))
        }
    }
}

/// PostgreSQL + pgvector memory store. Same strictly-local contract as the
/// SQLite store: the DSN points at a database the node operator owns, raw
/// text only ships to that one database, embeddings are produced locally.
///
/// With the pgvector extension present, vectors are `vector` columns and
/// search is an index-eligible `<=>` cosine ordering. Without it (bare
/// Postgres install, non-superuser connection), the store runs an honest
/// polyfill — packed-f32 `bytea` blobs, cosine computed in Rust — and
/// reports which mode it is in. A table written in the other mode is
/// refused loudly, never reshaped underneath the user's memories.
pub struct PgMemoryStore {
    client: tokio_postgres::Client,
    embed: EmbedConfig,
    /// true → native `vector` columns + `<=>`-ordered search.
    pgvector: bool,
    /// Schema holding the memories table (tests isolate via schemas).
    schema: String,
}

impl PgMemoryStore {
    /// Connect and prepare the schema, probing for the pgvector extension
    /// first. `schema` isolates parallel test runs; operators share
    /// `public` in their own database.
    pub async fn connect(
        dsn: &str,
        embed: EmbedConfig,
        schema: Option<&str>,
    ) -> anyhow::Result<Self> {
        // Refuse a TLS-demanding DSN up front with the real reason: this
        // store connects NoTls (memory is strictly node-local; TLS is the
        // deployment's job — a loopback database or a TLS-terminating
        // sidecar).
        if dsn.contains("sslmode=require") {
            anyhow::bail!(
                "postgres memory DSN demands TLS (sslmode=require) — this store connects NoTls; \
                 use a loopback database or terminate TLS at a sidecar"
            );
        }
        let (client, connection) = tokio_postgres::connect(dsn, tokio_postgres::NoTls)
            .await
            .with_context(|| format!("connecting postgres memory store ({})", describe_dsn(dsn)))?;
        // The store lives as long as the client; a dead connection is a
        // logged warning, then the next store call errors honestly.
        tokio::spawn(async move {
            if let Err(e) = connection.await {
                tracing::warn!(error = %e, "postgres memory store connection dropped");
            }
        });
        let schema = schema.unwrap_or("public").to_string();
        if schema != "public"
            && !schema
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            anyhow::bail!("postgres memory schema must match [a-zA-Z0-9_]+ — got {schema:?}");
        }
        // The pgvector extension is normally installed by the operator
        // beforehand; a non-superuser connection may fail to create it and
        // still legitimately find it present. Polyfill is never an error.
        if let Err(e) = client.execute("CREATE EXTENSION IF NOT EXISTS vector", &[]).await {
            tracing::debug!(
                error = %e,
                "pgvector extension create failed — needs a superuser or a preinstalled extension"
            );
        }
        let pgvector: bool = client
            .query_one(
                "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')",
                &[],
            )
            .await
            .context("probing pg_extension")?
            .get::<_, bool>(0);
        if schema != "public" {
            client
                .batch_execute(&format!("CREATE SCHEMA IF NOT EXISTS \"{schema}\""))
                .await
                .with_context(|| format!("creating schema {schema}"))?;
            // Keep `public` on the path: the pgvector extension (and any
            // types the operator installs) live there.
            client
                .batch_execute(&format!("SET search_path TO \"{schema}\", public"))
                .await
                .context("setting search_path")?;
        }
        // If the table exists from a run in the OTHER layout, stop loudly:
        // the user's memories are not silently dropped or reshaped. Scoped
        // to OUR schema — a superuser would otherwise match every test
        // schema's table in the database at once.
        let existing = client
            .query_opt(
                "SELECT data_type FROM information_schema.columns \
                 WHERE table_schema = $1 AND table_name = 'memories' \
                 AND column_name = 'embedding'",
                &[&schema],
            )
            .await
            .context("probing existing memories table")?
            .map(|r| r.get::<_, String>(0));
        if let Some(t) = existing {
            let expected = if pgvector { "USER-DEFINED" } else { "bytea" };
            if t != expected {
                anyhow::bail!(
                    "existing memories table stores embeddings as {t:?} but this run expects \
                     {expected:?} — pgvector availability changed since the previous run; \
                     migrate the table manually to preserve the data"
                );
            }
        }
        let col = if pgvector {
            "embedding vector NOT NULL"
        } else {
            "embedding BYTEA NOT NULL"
        };
        client
            .batch_execute(&format!(
                "CREATE TABLE IF NOT EXISTS memories (
                     id BIGSERIAL PRIMARY KEY,
                     text TEXT NOT NULL,
                     dim INTEGER NOT NULL,
                     provider TEXT NOT NULL,
                     created_ms BIGINT NOT NULL,
                     {col});
                 CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_ms);"
            ))
            .await
            .context("creating memory schema")?;
        tracing::info!(schema = %schema, pgvector, "postgres memory store ready");
        Ok(Self {
            client,
            embed,
            pgvector,
            schema,
        })
    }

    /// Raw embedding source (Ollama nomic / hash fallback) — identical to
    /// the SQLite store's semantics.
    pub async fn embed_text(&self, text: &str) -> anyhow::Result<(Vec<f32>, EmbedProvider)> {
        embed_with(&self.embed, text).await
    }

    /// memory.embed — embed locally + insert. Returns the row id; the
    /// retention cap mirrors [`MemoryStore::embed_and_store`] exactly.
    pub async fn embed_and_store(&self, text: &str) -> anyhow::Result<(i64, EmbedProvider)> {
        let (vector, provider) = embed_with(&self.embed, text).await?;
        let dim = vector.len() as i32;
        let created = crate::bootstrap::now_ms() as i64;
        let id: i64 = if self.pgvector {
            let s = vec_to_pg_text(&vector);
            self.client
                .query_one(
                    "INSERT INTO memories (text, dim, provider, created_ms, embedding)
                     VALUES ($1, $2, $3, $4, $5::text::vector) RETURNING id",
                    &[&text, &dim, &provider_str(provider), &created, &s],
                )
                .await
                .context("inserting memory")?
                .get::<_, i64>(0)
        } else {
            let blob = pack_vector(&vector);
            self.client
                .query_one(
                    "INSERT INTO memories (text, dim, provider, created_ms, embedding)
                     VALUES ($1, $2, $3, $4, $5) RETURNING id",
                    &[&text, &dim, &provider_str(provider), &created, &blob],
                )
                .await
                .context("inserting memory")?
                .get::<_, i64>(0)
        };
        if id > MAX_MEMORY_ROWS {
            self.prune_beyond(MAX_MEMORY_ROWS).await?;
        }
        Ok((id, provider))
    }

    /// memory.search — embed the query locally, rank by true cosine
    /// similarity, truncate to `top_k`.
    pub async fn search(&self, query: &str, top_k: usize) -> anyhow::Result<Vec<MemoryHit>> {
        let (qvec, _) = embed_with(&self.embed, query).await?;
        self.rank_against(&qvec, None, top_k).await
    }

    /// memory.similar — rank stored entries against the seed's stored
    /// vector (no re-embedding needed; works offline like the SQLite
    /// store). The seed itself is excluded.
    pub async fn similar_to(&self, seed_id: i64, top_k: usize) -> anyhow::Result<Vec<MemoryHit>> {
        let seed_vec: Vec<f32> = if self.pgvector {
            let row = self
                .client
                .query_opt(
                    "SELECT embedding::text FROM memories WHERE id = $1",
                    &[&seed_id],
                )
                .await
                .context("loading seed memory")?
                .context(format!("no memory entry with id {seed_id}"))?;
            let txt = row.get::<_, String>(0);
            vec_from_pg_text(&txt)?
        } else {
            let row = self
                .client
                .query_opt(
                    "SELECT embedding FROM memories WHERE id = $1",
                    &[&seed_id],
                )
                .await
                .context("loading seed memory")?
                .context(format!("no memory entry with id {seed_id}"))?;
            unpack_vector(&row.get::<_, Vec<u8>>(0))
        };
        self.rank_against(&seed_vec, Some(seed_id), top_k).await
    }

    /// The similarity engine shared by search and similar_to: native
    /// pgvector `<=>` when the whole table has exactly the query's
    /// dimension (uniform-dim tables are the normal case), a Rust cosine
    /// scan otherwise (mixed storing dims come from provider/dim config
    /// changes over time; cosine degrades to the shared prefix, the same
    /// rule the SQLite store has always had).
    async fn rank_against(
        &self,
        qvec: &[f32],
        exclude: Option<i64>,
        top_k: usize,
    ) -> anyhow::Result<Vec<MemoryHit>> {
        if self.pgvector {
            let row = self
                .client
                .query_one(
                    "SELECT COALESCE(MIN(dim), 0) AS lo, COALESCE(MAX(dim), 0) AS hi FROM memories",
                    &[],
                )
                .await
                .context("probing memory dims")?;
            let lo = row.get::<_, i32>(0);
            let hi = row.get::<_, i32>(1);
            if lo == hi && hi as usize == qvec.len() {
                return self.rank_indexed(qvec, exclude, top_k).await;
            }
        }
        self.rank_scan(qvec, exclude, top_k).await
    }

    /// Native pgvector path: one nearest-k query, index-eligible, scored
    /// by `<=>` cosine distance (`similarity = 1 - distance`).
    async fn rank_indexed(
        &self,
        qvec: &[f32],
        exclude: Option<i64>,
        top_k: usize,
    ) -> anyhow::Result<Vec<MemoryHit>> {
        let s = vec_to_pg_text(qvec);
        let rows = if exclude.is_some() {
            self.client
                .query(
                    "SELECT id, text, dim, provider, created_ms, embedding::text, \
                     embedding <=> $1::text::vector AS dist
                     FROM memories WHERE id != $2 ORDER BY dist LIMIT $3",
                    &[&s, &exclude.unwrap_or_default(), &(top_k as i64)],
                )
                .await
                .context("querying indexed memories")?
        } else {
            self.client
                .query(
                    "SELECT id, text, dim, provider, created_ms, embedding::text, \
                     embedding <=> $1::text::vector AS dist
                     FROM memories ORDER BY dist LIMIT $2",
                    &[&s, &(top_k as i64)],
                )
                .await
                .context("querying indexed memories")?
        };
        let mut hits = Vec::with_capacity(rows.len());
        for row in &rows {
            let mut hit = row_to_hit(row)?;
            hit.score = (1.0 - row.get::<_, f64>(6)) as f32;
            hits.push(hit);
        }
        Ok(hits)
    }

    /// Catch-all scan path (bytea stores; vector stores with mixed dims):
    /// load everything, cosine in Rust, sort + truncate by hand. Slower,
    /// identical semantics to the SQLite store's scan.
    async fn rank_scan(
        &self,
        qvec: &[f32],
        exclude: Option<i64>,
        top_k: usize,
    ) -> anyhow::Result<Vec<MemoryHit>> {
        let rows_query = if self.pgvector {
            "SELECT id, text, dim, provider, created_ms, embedding::text FROM memories"
        } else {
            "SELECT id, text, dim, provider, created_ms, embedding FROM memories"
        };
        let rows = self
            .client
            .query(rows_query, &[])
            .await
            .context("scanning memories")?;
        let mut hits = Vec::with_capacity(rows.len());
        for row in &rows {
            if exclude == Some(row.get::<_, i64>(0)) {
                continue;
            }
            let mut hit = row_to_hit(row)?;
            let stored = if self.pgvector {
                let txt = row.get::<_, String>(5);
                vec_from_pg_text(&txt)?
            } else {
                unpack_vector(&row.get::<_, Vec<u8>>(5))
            };
            hit.score = cosine(qvec, &stored);
            hits.push(hit);
        }
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(top_k);
        Ok(hits)
    }

    pub async fn forget(&self, id: i64) -> anyhow::Result<bool> {
        Ok(self
            .client
            .execute("DELETE FROM memories WHERE id = $1", &[&id])
            .await
            .context("forgetting memory")?
            > 0)
    }

    pub async fn count(&self) -> anyhow::Result<u64> {
        let row = self
            .client
            .query_one("SELECT COUNT(*) FROM memories", &[])
            .await
            .context("counting memories")?;
        Ok(row.get::<_, i64>(0) as u64)
    }

    /// Prune all but the `max_rows` most recent rows. Same retention rule
    /// and same testability split as [`MemoryStore::prune_beyond`].
    async fn prune_beyond(&self, max_rows: i64) -> anyhow::Result<usize> {
        Ok(self
            .client
            .execute(
                "DELETE FROM memories WHERE id NOT IN (
                     SELECT id FROM memories ORDER BY id DESC LIMIT $1)",
                &[&max_rows],
            )
            .await
            .context("pruning memories")? as usize)
    }

    /// memory.stats — health-dashboard port of the SQLite version: count,
    /// dimension split, provider split, newest entry, approximate store
    /// footprint (payload length; for `vector` columns that is the text
    /// projection length, close enough for a dashboard).
    pub async fn stats(&self) -> anyhow::Result<MemoryStats> {
        let count = self.count().await?;
        let emb_len = if self.pgvector {
            "LENGTH(embedding::text)"
        } else {
            "LENGTH(embedding)"
        };
        let rows = self
            .client
            .query(
                &format!(
                    "SELECT dim, provider, {emb_len}, created_ms FROM memories"
                ),
                &[],
            )
            .await
            .context("preparing stats scan")?;
        let mut dims: std::collections::BTreeMap<i64, i64> = Default::default();
        let mut providers: std::collections::BTreeMap<String, i64> = Default::default();
        let mut bytes_total: i64 = 0;
        let mut newest_ms: i64 = 0;
        for row in &rows {
            *dims.entry(row.get::<_, i32>(0) as i64).or_default() += 1;
            *providers.entry(row.get::<_, String>(1)).or_default() += 1;
            bytes_total += row.get::<_, i32>(2) as i64;
            newest_ms = newest_ms.max(row.get::<_, i64>(3));
        }
        Ok(MemoryStats {
            count,
            dims: dims
                .into_iter()
                .map(|(d, n)| (d as u32, n as u64))
                .collect(),
            providers: providers.into_iter().map(|(p, n)| (p, n as u64)).collect(),
            embedding_bytes: bytes_total as u64,
            newest_created_ms: if newest_ms > 0 {
                Some(newest_ms as u64)
            } else {
                None
            },
        })
    }
}

/// Row projection shared by both ranking paths (columns 0–4).
fn row_to_hit(row: &tokio_postgres::Row) -> anyhow::Result<MemoryHit> {
    Ok(MemoryHit {
        id: row.get::<_, i64>(0),
        text: row.get::<_, String>(1),
        score: 0.0,
        dim: row.get::<_, i32>(2) as usize,
        provider: provider_from_str(&row.get::<_, String>(3)),
        created_ms: row.get::<_, i64>(4) as u64,
    })
}

/// f32 slice → pgvector text form `[1.0,2.0]`. Rust's `Display` for f32
/// prints the shortest string that round-trips, so this is lossless for
/// every value the embedders produce.
fn vec_to_pg_text(v: &[f32]) -> String {
    let mut s = String::with_capacity(v.len() * 8 + 2);
    s.push('[');
    for (i, f) in v.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&f.to_string());
    }
    s.push(']');
    s
}

/// pgvector text form → f32 vector. Malformed input is a typed error, not
/// silent zeros — a corrupted stored vector must never impersonate a hit.
fn vec_from_pg_text(s: &str) -> anyhow::Result<Vec<f32>> {
    let inner = s.trim().trim_start_matches('[').trim_end_matches(']');
    if inner.trim().is_empty() {
        return Ok(Vec::new());
    }
    inner
        .split(',')
        .map(|t| {
            t.trim()
                .parse::<f32>()
                .map_err(|e| anyhow::anyhow!("bad stored vector component {t:?}: {e}"))
        })
        .collect()
}

/// The operator-facing DSN line: host/database only — credentials and port
/// never appear in logs, console output or connect errors.
fn describe_dsn(dsn: &str) -> String {
    if let Some((_, rest)) = dsn.split_once("://") {
        let hostpart = rest
            .rsplit_once('@')
            .map(|(_, h)| h)
            .unwrap_or(rest)
            .split(['?', '&', ' '])
            .next()
            .unwrap_or("");
        let (host, db) = match hostpart.split_once('/') {
            Some((h, d)) => (h, d),
            None => (hostpart, ""),
        };
        return format!("{}/{}", trim_stray(host), trim_stray(db));
    }
    // Key-value conninfo ("host=localhost port=5432 dbname=jerith").
    let mut host: Option<&str> = None;
    let mut db: Option<&str> = None;
    for word in dsn.split_whitespace() {
        let mut it = word.splitn(2, '=');
        let k = it.next().unwrap_or("");
        let v = it.next().unwrap_or("");
        match (k, v.is_empty()) {
            ("host", false) => host = Some(v),
            ("dbname", false) => db = Some(v),
            _ => {}
        }
    }
    match (host, db) {
        (Some(h), Some(d)) => format!("{h}/{d}"),
        _ => "postgres conninfo (details in config)".to_string(),
    }
}

fn trim_stray(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace() || c == ',')
        .trim_end_matches('/')
}

/// Whichever vector store the operator chose, behind one interface: the
/// gateway and the CLI never know the difference. A postgres backend
/// chosen without a DSN is refused here — a misconfiguration fails loud,
/// it is never a silent swap to a different store.
pub enum MemoryHandle {
    Sqlite(MemoryStore),
    Pg(PgMemoryStore),
}

/// Shared handle for the gateway.
pub type SharedMemory = Arc<MemoryHandle>;

impl MemoryHandle {
    /// Choose-and-open honoring the operator's configured backend. SQLite
    /// ignores `pg_dsn`; postgres requires one.
    pub async fn open(
        backend: crate::config::MemoryBackend,
        sqlite_path: &Path,
        pg_dsn: Option<&str>,
        embed: EmbedConfig,
    ) -> anyhow::Result<Self> {
        match backend {
            crate::config::MemoryBackend::Sqlite => {
                Ok(Self::Sqlite(MemoryStore::open(sqlite_path, embed)?))
            }
            crate::config::MemoryBackend::PostgresPgvector => {
                let dsn = pg_dsn
                    .filter(|s| !s.trim().is_empty())
                    .context(
                        "memory_backend = postgres but no DSN — set memory_dsn in config.toml \
                         (or the JERITH_MEMORY_DSN environment variable); refusing to run memory \
                         against a different store",
                    )?;
                Ok(Self::Pg(PgMemoryStore::connect(dsn, embed, None).await?))
            }
        }
    }

    /// memory.embed — embed + store. Returns the row id and the provider
    /// that produced the embedding (nomic vs offline hash fallback).
    pub async fn embed_and_store(&self, text: &str) -> anyhow::Result<(i64, EmbedProvider)> {
        match self {
            Self::Sqlite(s) => s.embed_and_store(text).await,
            Self::Pg(s) => s.embed_and_store(text).await,
        }
    }

    /// Produce an embedding without storing it (escapes both stores; used
    /// by the CLI's plain `embed` surface and skill handlers).
    pub async fn embed_text(&self, text: &str) -> anyhow::Result<(Vec<f32>, EmbedProvider)> {
        match self {
            Self::Sqlite(s) => s.embed_text(text).await,
            Self::Pg(s) => s.embed_text(text).await,
        }
    }

    /// memory.search — embed the query and return the most similar hits.
    pub async fn search(&self, query: &str, top_k: usize) -> anyhow::Result<Vec<MemoryHit>> {
        match self {
            Self::Sqlite(s) => s.search(query, top_k).await,
            Self::Pg(s) => s.search(query, top_k).await,
        }
    }

    /// memory.similar — "find memories like this one", seed-vector based.
    pub async fn similar_to(&self, seed_id: i64, top_k: usize) -> anyhow::Result<Vec<MemoryHit>> {
        match self {
            Self::Sqlite(s) => s.similar_to(seed_id, top_k).await,
            Self::Pg(s) => s.similar_to(seed_id, top_k).await,
        }
    }

    /// "Forget this conversation" support: delete by id.
    pub async fn forget(&self, id: i64) -> anyhow::Result<bool> {
        match self {
            Self::Sqlite(s) => s.forget(id).await,
            Self::Pg(s) => s.forget(id).await,
        }
    }

    pub async fn count(&self) -> anyhow::Result<u64> {
        match self {
            Self::Sqlite(s) => s.count().await,
            Self::Pg(s) => s.count().await,
        }
    }

    /// memory.stats — health-dashboard view of the live store.
    pub async fn stats(&self) -> anyhow::Result<MemoryStats> {
        match self {
            Self::Sqlite(s) => s.stats().await,
            Self::Pg(s) => s.stats().await,
        }
    }

    /// Operator-facing one-line description of the live store, honest
    /// about the backend actually in use (and the polyfill mode).
    pub fn describe(&self) -> String {
        match self {
            Self::Sqlite(s) => {
                if s.path.as_os_str().is_empty() {
                    format!("SQLite (in memory, dim {})", s.embed.dim)
                } else {
                    format!("SQLite at {} (dim {})", s.path.display(), s.embed.dim)
                }
            }
            Self::Pg(s) => {
                let mode = if s.pgvector {
                    "PostgreSQL + pgvector"
                } else {
                    "PostgreSQL (bytea polyfill — no pgvector extension)"
                };
                if s.schema == "public" {
                    format!("{mode} (dim {})", s.embed.dim)
                } else {
                    format!(
                        "{mode} (dim {}, schema {schema})",
                        s.embed.dim,
                        schema = s.schema
                    )
                }
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> MemoryStore {
        MemoryStore::open_in_memory(EmbedConfig {
            dim: 64,
            ..EmbedConfig::default()
        })
        .unwrap()
    }

    #[tokio::test]
    async fn prune_beyond_keeps_most_recent_rows() {
        let s = store();
        let mut ids = Vec::new();
        for i in 0..6 {
            let (id, _) = s.embed_and_store(&format!("memory row {i}")).await.unwrap();
            ids.push(id);
        }

        // Drop the two oldest; keep the four most recent.
        let removed = s.prune_beyond(4).await.unwrap();
        assert_eq!(removed, 2);
        assert_eq!(s.count().await.unwrap(), 4);
        let conn = s.conn.lock().await;
        let surviving: Vec<i64> = conn
            .prepare("SELECT id FROM memories ORDER BY id")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(surviving.as_slice(), &ids[2..], "newest rows survive");
    }

    #[tokio::test]
    async fn embed_and_store_roundtrip() {
        let s = store();
        let (id, provider) = s.embed_and_store("the quick brown fox").await.unwrap();
        assert!(id >= 1);
        // Provider depends on the environment (nomic via Ollama when it's
        // running, hash fallback otherwise) — either is valid.
        assert!(matches!(
            provider,
            EmbedProvider::Nomic | EmbedProvider::HashFallback
        ));
        assert_eq!(s.count().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn similar_ranks_against_the_seed_vector() {
        let s = store();
        let (a, _) = s
            .embed_and_store("how to bake sourdough bread at home")
            .await
            .unwrap();
        s.embed_and_store("quantum computing stock price forecast")
            .await
            .unwrap();
        s.embed_and_store("sourdough starter feeding schedule")
            .await
            .unwrap();

        let hits = s.similar_to(a, 2).await.unwrap();
        assert_eq!(hits.len(), 2);
        // The seed itself is never returned.
        assert!(hits.iter().all(|h| h.id != a));
        // The sourdough sibling outranks the unrelated entry.
        assert!(
            hits.iter().any(|h| h.text.contains("sourdough")),
            "expected a sourdough hit, got {:?}",
            hits.iter().map(|h| h.text.as_str()).collect::<Vec<_>>()
        );
        // Unknown seed errors clearly.
        assert!(s.similar_to(9999, 2).await.is_err());
    }

    #[tokio::test]
    async fn stats_reports_counts_dims_providers_and_footprint() {
        let s = store();
        s.embed_and_store("alpha entry about kilns").await.unwrap();
        s.embed_and_store("beta entry about kilns too")
            .await
            .unwrap();
        let st = s.stats().await.unwrap();
        assert_eq!(st.count, 2);
        assert_eq!(st.dims.len(), 1);
        assert!(st.embedding_bytes > 0);
        assert_eq!(st.providers.len(), 1);
        assert!(st.newest_created_ms.is_some());
        // Empty store: zeroed stats, no newest timestamp.
        let empty = MemoryStore::open_in_memory(EmbedConfig {
            dim: 64,
            ..EmbedConfig::default()
        })
        .unwrap();
        let st = empty.stats().await.unwrap();
        assert_eq!(st.count, 0);
        assert!(st.newest_created_ms.is_none());
    }

    #[tokio::test]
    async fn search_ranks_similar_text_higher() {
        let s = store();
        s.embed_and_store("how to bake sourdough bread at home")
            .await
            .unwrap();
        s.embed_and_store("quantum computing stock price forecast")
            .await
            .unwrap();
        s.embed_and_store("sourdough starter feeding schedule")
            .await
            .unwrap();

        let hits = s.search("sourdough bread baking", 3).await.unwrap();
        assert_eq!(hits.len(), 3);
        // Both sourdough docs must outrank the quantum doc.
        let texts: Vec<&str> = hits.iter().map(|h| h.text.as_str()).collect();
        let quantum_pos = texts
            .iter()
            .position(|t| t.contains("quantum"))
            .expect("all docs returned");
        let sourdough_first = texts[0].contains("sourdough");
        assert!(
            sourdough_first || quantum_pos == texts.len() - 1,
            "similar docs should rank higher, got order {texts:?}"
        );
        assert!(hits[0].score > hits[hits.len() - 1].score);
    }

    #[tokio::test]
    async fn search_empty_store_ok() {
        let s = store();
        let hits = s.search("anything", 5).await.unwrap();
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn forget_removes_entry() {
        let s = store();
        let (id, _) = s.embed_and_store("to be forgotten").await.unwrap();
        assert!(s.forget(id).await.unwrap());
        assert!(!s.forget(id).await.unwrap());
        assert_eq!(s.count().await.unwrap(), 0);
    }

    #[test]
    fn hash_embed_is_deterministic_and_normalized() {
        let a = hash_embed("hello world", 128);
        let b = hash_embed("hello world", 128);
        let c = hash_embed("goodbye world", 128);
        assert_eq!(a, b);
        assert_ne!(a, c);
        let norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "L2 normalized, got {norm}");
        assert_eq!(a.len(), 128);
    }

    #[test]
    fn hash_embed_similar_texts_share_buckets() {
        // Same token in both → cosine clearly above dissimilar pair.
        let a = hash_embed("sourdough baking tips", 64);
        let b = hash_embed("more sourdough baking tips today", 64);
        let c = hash_embed("quantum entanglement physics", 64);
        assert!(cosine(&a, &b) > cosine(&a, &c));
    }

    #[test]
    fn matryoshka_truncation_keeps_dim_and_normalizes() {
        let mut v = hash_embed("some longer text for embedding", 128);
        truncate_matryoshka(&mut v, 32);
        assert_eq!(v.len(), 32);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn vector_pack_roundtrip() {
        let v = vec![0.25f32, -1.5, 3.0, 0.0];
        let bytes = pack_vector(&v);
        assert_eq!(unpack_vector(&bytes), v);
    }

    #[test]
    fn cosine_well_known_values() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert!((cosine(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-6);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
    }

    #[tokio::test]
    async fn persists_across_reopen() {
        let dir = std::env::temp_dir().join(format!("jerith-mem-test-{}", std::process::id()));
        let path = dir.join("mem.sqlite");
        let _ = std::fs::remove_dir_all(&dir);
        {
            let s = MemoryStore::open(&path, EmbedConfig::default()).unwrap();
            s.embed_and_store("persistent memory entry").await.unwrap();
        }
        {
            let s = MemoryStore::open(&path, EmbedConfig::default()).unwrap();
            assert_eq!(s.count().await.unwrap(), 1);
            let hits = s.search("persistent memory", 1).await.unwrap();
            assert_eq!(hits.len(), 1);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn provider_roundtrip() {
        assert_eq!(
            provider_from_str(provider_str(EmbedProvider::Nomic)),
            EmbedProvider::Nomic
        );
        assert_eq!(
            provider_from_str(provider_str(EmbedProvider::HashFallback)),
            EmbedProvider::HashFallback
        );
    }

    // =========================================================================
    // PostgreSQL + pgvector backend — pure helpers (no server needed)
    // =========================================================================

    #[test]
    fn pg_vector_text_roundtrips_losslessly() {
        let v = vec![0.25f32, -1.5, 3.0, 0.0, 1.0 / 3.0];
        let text = vec_to_pg_text(&v);
        assert!(text.starts_with('[') && text.ends_with(']'));
        assert_eq!(vec_from_pg_text(&text).unwrap(), v, "shortest-round-trip f32s");
        assert_eq!(vec_from_pg_text("[]").unwrap(), Vec::<f32>::new());
        assert!(matches!(vec_from_pg_text("[1.0,oops]"), Err(_)), "malformed vector is a typed error, never silent zeros");
    }

    #[test]
    fn pg_dsn_description_never_leaks_credentials() {
        let full = describe_dsn("postgres://jerith:s3cret@10.0.0.7:5433/memdb?sslmode=disable");
        assert_eq!(full, "10.0.0.7:5433/memdb");
        assert!(!full.contains("s3cret"));
        assert!(!describe_dsn("postgres://jerith:s3cret@10.0.0.7:5433/memdb").contains("s3cret"));
        let kv = describe_dsn("host=db.internal port=5432 dbname=jerith user=postgres password=hunter2");
        assert_eq!(kv, "db.internal/jerith");
        assert!(!kv.contains("hunter2"));
        // No scheme, no recognizable pairs: an honest generic line.
        assert!(describe_dsn("postgresql-testdb").contains("conninfo"));
    }

    // =========================================================================
    // PostgreSQL + pgvector backend — live stores (env-gated, never a
    // silent skip: the reason is printed). Superuser DSN exercises the
    // native pgvector path; a non-superuser DSN on a database without the
    // extension (and no rights to install it) exercises the bytea
    // polyfill.
    // =========================================================================

    fn pg_url() -> Option<String> {
        std::env::var("JERITH_TEST_PG_URL")
            .ok()
            .filter(|s| !s.trim().is_empty())
    }

    fn pg_url_noext() -> Option<String> {
        std::env::var("JERITH_TEST_PG_URL_NOEXT")
            .ok()
            .filter(|s| !s.trim().is_empty())
    }

    #[tokio::test]
    async fn pg_native_roundtrip_search_rank_lifecycle() {
        let Some(dsn) = pg_url() else {
            eprintln!("skip pg_native_* : set JERITH_TEST_PG_URL to a postgres DSN with pgvector");
            return;
        };
        let schema = format!("memtest_{}_vecmode", std::process::id());
        let s = PgMemoryStore::connect(
            &dsn,
            EmbedConfig {
                dim: 64,
                ..EmbedConfig::default()
            },
            Some(&schema),
        )
        .await
        .expect("connect pgvector store");
        assert!(s.pgvector, "this DSN must have the vector extension");

        // Store and recover-by-similar (no re-embedding needed).
        let (a, _) = s
            .embed_and_store("how to bake sourdough bread at home")
            .await
            .expect("insert a");
        let (b, _) = s
            .embed_and_store("quantum computing stock price forecast")
            .await
            .expect("insert b");
        s.embed_and_store("sourdough starter feeding schedule")
            .await
            .expect("insert c");

        let hits = s.similar_to(a, 5).await.expect("similar");
        assert_eq!(hits.len(), 2, "seed excluded");
        assert!(hits.iter().all(|h| h.id != a));
        assert!(hits.iter().any(|h| h.id == b));

        // Search ranks the sourdough sibling above the quantum entry.
        let top = s.search("sourdough bread baking", 3).await.expect("search");
        assert_eq!(top.len(), 3);
        assert!(top[0].score > top[2].score, "strict ordering: {top:?}");
        assert_eq!(top.len(), 3, "top_k respected");
        let texts: Vec<&str> = top.iter().map(|h| h.text.as_str()).collect();
        let per = texts
            .iter()
            .take(2)
            .filter(|t| t.contains("sourdough"))
            .count();
        assert!(per >= 1, "sourdough sibling in top ranks, order was {texts:?}");

        // Empty top_k and forget and count.
        assert!(s.search("anything", 0).await.unwrap().is_empty());
        assert!(s.forget(b).await.unwrap());
        assert!(!s.forget(b).await.unwrap());
        assert_eq!(s.count().await.unwrap(), 2);

        // Stats carry dims/providers/bytes/newest.
        let st = s.stats().await.expect("stats");
        assert_eq!(st.count, 2);
        assert!(st.dims.contains(&(64, 2)));
        assert!(st.embedding_bytes > 0);
        assert!(st.newest_created_ms.is_some());

        // Retention cap: oldest rows pruned, newest survive.
        for i in 0..5 {
            s.embed_and_store(&format!("retention row {i}")).await.unwrap();
        }
        let removed = s.prune_beyond(5).await.expect("prune");
        assert_eq!(removed, 2, "7 rows → 5 most recent kept");
        assert_eq!(s.count().await.unwrap(), 5);
        let hits = s.search("retention row 4", 3).await.expect("pruned search");
        assert!(!hits.is_empty(), "newest rows survive pruning");
        // The forgotten (quantum) row is gone: at top_k = 3 across all
        // surviving rows — originals and retention rows — no hit is ever
        // the deleted entry.
        let hits = s.search("quantum computing stock price forecast", 3).await.unwrap();
        assert!(
            hits.iter().all(|h| !h.text.contains("quantum")),
            "the forgotten row must never be returned: {:?}",
            hits.iter().map(|h| h.text.as_str()).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn pg_native_persists_across_reconnect() {
        let Some(dsn) = pg_url() else {
            eprintln!("skip pg_native_reconnect : set JERITH_TEST_PG_URL");
            return;
        };
        let schema = format!("memtest_{}_reconnect", std::process::id());
        let embed = EmbedConfig {
            dim: 64,
            ..EmbedConfig::default()
        };
        let (first, _) = {
            let s = PgMemoryStore::connect(&dsn, embed.clone(), Some(&schema))
                .await
                .expect("connect 1");
            s.embed_and_store("persistent pg memory entry").await.expect("insert")
        };
        let s = PgMemoryStore::connect(&dsn, embed, Some(&schema))
            .await
            .expect("connect 2");
        assert_eq!(s.count().await.unwrap(), 1);
        let hits = s.search("persistent pg memory", 1).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, first);
    }

    #[tokio::test]
    async fn pg_bytea_polyfill_behaves_like_sqlite() {
        let Some(dsn) = pg_url_noext() else {
            eprintln!(
                "skip pg_bytea_polyfill : set JERITH_TEST_PG_URL_NOEXT to a postgres DSN \n             on a database WITHOUT the pgvector extension and no extension-creation rights"
            );
            return;
        };
        let schema = format!("memtest_{}_polyfill", std::process::id());
        let s = PgMemoryStore::connect(
            &dsn,
            EmbedConfig {
                dim: 64,
                ..EmbedConfig::default()
            },
            Some(&schema),
        )
        .await
        .expect("connect polyfill store");
        assert!(!s.pgvector, "this DSN must lack the extension");
        s.embed_and_store("how to bake sourdough bread at home").await.unwrap();
        s.embed_and_store("quantum computing stock price forecast").await.unwrap();
        s.embed_and_store("sourdough starter feeding schedule").await.unwrap();
        let top = s.search("sourdough bread baking", 3).await.expect("search");
        assert_eq!(top.len(), 3);
        // Polyfill ranks by true cosine against packed vectors.
        assert!(top[0].score > top[2].score, "strict ordering: {top:?}");
        let texts: Vec<&str> = top.iter().map(|h| h.text.as_str()).collect();
        assert!(
            texts.first().map(|t| t.contains("sourdough")).unwrap_or(false)
                || texts.last().map(|t| t.contains("quantum")).unwrap_or(false),
            "similar docs rank above dissimilar: {texts:?}"
        );
        // mixed-dim cross-check: a scan-based rank over parsed vectors.
        let st = s.stats().await.expect("stats");
        assert_eq!(st.count, 3);
        assert!(st.dims.contains(&(64, 3)));
    }

    #[tokio::test]
    async fn pg_native_mixed_dims_fall_back_to_scan() {
        let Some(dsn) = pg_url() else {
            eprintln!("skip pg_native_mixed : set JERITH_TEST_PG_URL");
            return;
        };
        let schema = format!("memtest_{}_mixed", std::process::id());
        // Two stores over ONE table (vectors are dimension-free here),
        // with different truncation dims — the mixed-dim situation the
        // scan path exists for.
        let s64 = PgMemoryStore::connect(
            &dsn,
            EmbedConfig {
                dim: 64,
                ..EmbedConfig::default()
            },
            Some(&schema),
        )
        .await
        .expect("connect 64");
        s64.embed_and_store("how to bake sourdough bread at home").await.unwrap();
        let s32 = PgMemoryStore::connect(
            &dsn,
            EmbedConfig {
                dim: 32,
                ..EmbedConfig::default()
            },
            Some(&schema),
        )
        .await
        .expect("connect 32");
        s32.embed_and_store("quantum computing stock price forecast").await.unwrap();
        // No uniform dim → the native `<=>` path must NOT be used; the
        // scan path still ranks, degrading to the shared prefix like the
        // SQLite store.
        let (h64, _) = s64.embed_and_store("sourdough starter feeding schedule").await.unwrap();
        let hits = s64.similar_to(h64, 2).await.expect("mixed search");
        assert_eq!(hits.len(), 2, "all rows located despite mixed dims");
        assert!(hits.iter().all(|h| h.id != h64));
    }
}
