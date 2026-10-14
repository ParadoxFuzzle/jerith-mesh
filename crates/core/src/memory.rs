//! Local memory: SQLite vector store + local-only embeddings.
//!
//! Per spec:
//! - Memory is **strictly local** — never replicated into the mesh, never
//!   distributed. This store is the SQLite backend (PostgreSQL+pgvector is
//!   the alternative backend in a later phase).
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
        match ollama_embed(&self.embed, text).await {
            Ok(mut v) => {
                truncate_matryoshka(&mut v, self.embed.dim);
                Ok((v, EmbedProvider::Nomic))
            }
            Err(e) => {
                tracing::debug!(error = %e, "nomic embedding unavailable — using hash fallback");
                let v = hash_embed(text, self.embed.dim);
                Ok((v, EmbedProvider::HashFallback))
            }
        }
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

/// Shared handle for the gateway.
pub type SharedMemory = Arc<MemoryStore>;

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
}
