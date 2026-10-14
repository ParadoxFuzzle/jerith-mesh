//! Colibri-style on-demand expert weights: the local tensor primitive.
//!
//! `crates/core` has always routed expert *queries*; until now the code
//! that answered them re-encoded a text completion's hash instead of
//! running the expert's tensors (see `quic::RuntimeExpert::run_async`).
//! This module is the real thing, deliberately small and dependency-free:
//!
//! - **On disk**: one `<expert_id>.expert` file per expert — magic,
//!   dims, id, the gate/up/down f32 tensors, and an integrity checksum.
//!   `write_to` is how an importer will lay experts out (a GGUF/Colibri
//!   extractor later); the format is stable and self-describing.
//! - **On demand**: weights load lazily on first use and stay resident in
//!   a byte-budgeted LRU. Hot experts (the ones a routing distribution
//!   actually pins) stay in RAM; cold ones are evicted and re-read — the
//!   same "keep the hot set local, stream the tail" shape Colibri uses,
//!   just with plain files instead of a mmap'd quantized shard.
//! - **Compute**: one MoE expert is a SwiGLU MLP — `down(silu(gate·x) *
//!   (up·x))` — run over the activation vector that arrives on the wire.
//!   Output has the same width as input, which is what the
//!   `ExpertQuery`/`ExpertResult` contract already carries.
//!
//! Quantized layouts (Q4_K/Q8_0 dequant, mmap shards) are the next step;
//! the store boundary (`run(expert_id, input) -> Vec<f32>`) does not move.

use anyhow::{Context as _, Result};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// File magic + version. Bumping the version is the incompatible change;
/// the magic alone catches "this is not an expert file at all".
const MAGIC: &[u8; 8] = b"JEXP1\0\0\0";
const VERSION: u32 = 1;
/// Sanity bound on a single expert file: dims are u32 and an activation
/// vector on the wire is capped at 65 536 floats (`quic::MAX_EXPERT_INPUT_FLOATS`),
/// so a legit expert can be big — but not gigabytes-of-dimensions big.
const MAX_DIM: usize = 1 << 20;

/// One expert's weights. Row-major, little-endian f32 on disk and in RAM.
///
/// Shapes for a hidden size `h` and intermediate size `i` (MoE FFN):
/// - `gate`: `[i × h]`  (gate projection)
/// - `up`:   `[i × h]`  (up projection)
/// - `down`: `[h × i]`  (down projection)
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertWeights {
    pub id: String,
    pub hidden: usize,
    pub intermediate: usize,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
}

impl ExpertWeights {
    /// A deterministic synthetic expert — used by tests and by
    /// `jerith experts make` to stand one up for a mesh smoke test.
    /// Weights are a fixed LCG so two nodes compute identical numbers.
    pub fn synthetic(id: &str, hidden: usize, intermediate: usize) -> Self {
        fn lcg(seed: &mut u64) -> f32 {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let x = (*seed >> 33) as u32;
            (x as f32 / u32::MAX as f32) * 0.2 - 0.1
        }
        // Seed from the id: same id → identical weights on every node
        // (the property a mesh-wide reproducible expert test needs), and
        // different ids → different experts.
        let mut seed: u64 = 0x5eed;
        for b in id.as_bytes() {
            seed = seed.wrapping_mul(0x100000001b3) ^ (*b as u64);
        }
        let mut fill = |n: usize| (0..n).map(|_| lcg(&mut seed)).collect::<Vec<f32>>();
        Self {
            id: id.to_string(),
            hidden,
            intermediate,
            gate: fill(hidden * intermediate),
            up: fill(hidden * intermediate),
            down: fill(hidden * intermediate),
        }
    }

    /// Bytes this expert occupies resident (payload only; the id string
    /// and Vec headers are noise next to real MoE dimensions).
    pub fn bytes(&self) -> usize {
        (self.gate.len() + self.up.len() + self.down.len()) * std::mem::size_of::<f32>()
    }

    /// Validate internal consistency before use (a corrupted file must
    /// fail here, not by indexing panic in the forward pass).
    pub fn validate(&self) -> Result<()> {
        let expect = self
            .hidden
            .checked_mul(self.intermediate)
            .context("expert dimensions overflow")?;
        anyhow::ensure!(
            self.gate.len() == expect && self.up.len() == expect && self.down.len() == expect,
            "expert {}: tensor sizes {} {} {} do not match {expect}",
            self.id,
            self.gate.len(),
            self.up.len(),
            self.down.len()
        );
        Ok(())
    }

    /// SwiGLU forward: `down(silu(gate·x) * (up·x))`.
    ///
    /// `x` is the activation vector for one token (`hidden` wide); the
    /// result is the expert's contribution, also `hidden` wide.
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>> {
        anyhow::ensure!(
            x.len() == self.hidden,
            "expert {}: input has {} floats, expected hidden={}",
            self.id,
            x.len(),
            self.hidden
        );
        let h = self.hidden;
        let i = self.intermediate;
        // h = silu(gate·x) * (up·x), accumulated per intermediate unit.
        let mut act = vec![0.0f32; i];
        for (unit, slot) in act.iter_mut().enumerate() {
            let g = dot(&self.gate[unit * h..(unit + 1) * h], x);
            let u = dot(&self.up[unit * h..(unit + 1) * h], x);
            *slot = silu(g) * u;
        }
        // y = down·h — each output row of `down` is `i` wide.
        let mut y = vec![0.0f32; h];
        for (out, slot) in y.iter_mut().enumerate() {
            *slot = dot(&self.down[out * i..(out + 1) * i], &act);
        }
        Ok(y)
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// FNV-1a over the bytes preceding the checksum field.
fn checksum(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

impl ExpertWeights {
    /// Serialize into the on-disk container (see the module docs for the
    /// layout; the checksum covers everything before it).
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut out = Vec::with_capacity(64 + self.bytes());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(self.id.len() as u32).to_le_bytes());
        out.extend_from_slice(self.id.as_bytes());
        out.extend_from_slice(&(self.hidden as u32).to_le_bytes());
        out.extend_from_slice(&(self.intermediate as u32).to_le_bytes());
        for t in [&self.gate, &self.up, &self.down] {
            for v in t {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        let sum = checksum(&out);
        out.extend_from_slice(&sum.to_le_bytes());
        Ok(out)
    }

    /// Parse a container. Every failure mode is a typed error: bad magic,
    /// truncated body, dimensions that overflow, tensor-length mismatch,
    /// or checksum disagreement (bit rot / partial write).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut cur = 0usize;
        let take = |cur: &mut usize, n: usize| -> Result<&[u8]> {
            anyhow::ensure!(
                bytes.len() >= *cur + n,
                "expert file truncated: wanted {n} bytes at {cur}, file is {}",
                bytes.len()
            );
            let s = &bytes[*cur..*cur + n];
            *cur += n;
            Ok(s)
        };
        anyhow::ensure!(
            take(&mut cur, 8)? == MAGIC,
            "not an expert file (bad magic)"
        );
        let version = u32::from_le_bytes(take(&mut cur, 4)?.try_into().unwrap());
        anyhow::ensure!(
            version == VERSION,
            "expert file version {version} unsupported (this build reads {VERSION})"
        );
        let id_len = u32::from_le_bytes(take(&mut cur, 4)?.try_into().unwrap()) as usize;
        let id = String::from_utf8(take(&mut cur, id_len)?.to_vec())
            .context("expert id is not valid UTF-8")?;
        let hidden = u32::from_le_bytes(take(&mut cur, 4)?.try_into().unwrap()) as usize;
        let intermediate = u32::from_le_bytes(take(&mut cur, 4)?.try_into().unwrap()) as usize;
        anyhow::ensure!(
            hidden > 0 && intermediate > 0 && hidden <= MAX_DIM && intermediate <= MAX_DIM,
            "expert {id}: implausible dimensions {hidden}×{intermediate}"
        );
        let n = hidden
            .checked_mul(intermediate)
            .context("expert dimensions overflow")?;
        let read_tensor = |cur: &mut usize| -> Result<Vec<f32>> {
            let raw = take(cur, n * 4)?;
            Ok(raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect())
        };
        let gate = read_tensor(&mut cur)?;
        let up = read_tensor(&mut cur)?;
        let down = read_tensor(&mut cur)?;
        let sum = u64::from_le_bytes(take(&mut cur, 8)?.try_into().unwrap());
        anyhow::ensure!(
            checksum(&bytes[..cur - 8]) == sum,
            "expert {id}: checksum mismatch (corrupted or partially written file)"
        );
        let w = Self {
            id,
            hidden,
            intermediate,
            gate,
            up,
            down,
        };
        w.validate()?;
        Ok(w)
    }

    /// Write one expert container to `path` (atomically: temp + rename,
    /// so a crashed importer can never leave a half file that a later
    /// scan would list as hosted).
    pub fn write_to(&self, path: &Path) -> Result<()> {
        let bytes = self.to_bytes()?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)
                .with_context(|| format!("creating expert dir {}", dir.display()))?;
        }
        let tmp = path.with_extension("expert.tmp");
        std::fs::write(&tmp, &bytes)
            .with_context(|| format!("writing expert temp file {}", tmp.display()))?;
        std::fs::rename(&tmp, path)
            .with_context(|| format!("publishing expert file {}", path.display()))?;
        Ok(())
    }

    /// Read one expert container from `path`.
    pub fn read_from(path: &Path) -> Result<Self> {
        let bytes =
            std::fs::read(path).with_context(|| format!("reading expert {}", path.display()))?;
        Self::from_bytes(&bytes).with_context(|| format!("parsing expert {}", path.display()))
    }
}

/// Hit/miss/load counters — the observability surface for `/status` and
/// for judging whether the cache budget is right for the hot set.
#[derive(Debug, Default)]
pub struct StoreStats {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub loads: AtomicU64,
    pub evictions: AtomicU64,
    pub load_errors: AtomicU64,
}

/// Point-in-time copy of [`StoreStats`] (cheap to render).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StoreStatsSnapshot {
    pub hits: u64,
    pub misses: u64,
    pub loads: u64,
    pub evictions: u64,
    pub load_errors: u64,
}

struct CacheEntry {
    weights: Arc<ExpertWeights>,
    bytes: usize,
}

#[derive(Default)]
struct ExpertCache {
    entries: HashMap<String, CacheEntry>,
    /// LRU order: front = least recently used.
    order: VecDeque<String>,
    bytes: usize,
}

/// The on-demand expert store: `dir` holds `<expert_id>.expert` files,
/// `cache_mb` bounds how many resident bytes are kept.
pub struct ExpertStore {
    dir: PathBuf,
    cache_budget: usize,
    cache: Mutex<ExpertCache>,
    stats: StoreStats,
}

impl ExpertStore {
    /// Open (or create) a store rooted at `dir` with a resident budget in
    /// MiB. A zero budget means "never cache" (load every call) — the
    /// tests use it to prove caching is what serves the second call.
    pub fn new(dir: impl Into<PathBuf>, cache_mb: usize) -> Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating expert store dir {}", dir.display()))?;
        Ok(Self {
            dir,
            cache_budget: cache_mb.saturating_mul(1024 * 1024),
            cache: Mutex::new(ExpertCache::default()),
            stats: StoreStats::default(),
        })
    }

    /// `~/.jerith-mesh/experts` — the default store the node wires up.
    pub fn default_dir() -> PathBuf {
        crate::config::NodeConfig::data_dir()
            .parent()
            .map(|p| p.join("experts"))
            .unwrap_or_else(|| PathBuf::from("experts"))
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn cache_budget_bytes(&self) -> usize {
        self.cache_budget
    }

    fn path_for(&self, expert_id: &str) -> PathBuf {
        // Expert ids are mesh-supplied strings; keep them to a single path
        // component so a hostile id can't escape the store dir.
        let safe: String = expert_id
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        self.dir.join(format!("{safe}.expert"))
    }

    /// Expert ids this node holds — the filename stems (the header id is
    /// re-checked on load, so a misnamed file fails loudly rather than
    /// serving the wrong expert).
    pub fn hosted(&self) -> Vec<String> {
        let mut ids: Vec<String> = std::fs::read_dir(&self.dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                let p = e.path();
                if p.extension().and_then(|x| x.to_str()) != Some("expert") {
                    return None;
                }
                p.file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_string())
            })
            .collect();
        ids.sort();
        ids
    }

    /// Does this node hold the expert? One stat (resident check first, so
    /// the hot path never touches the filesystem). This is what lets a
    /// query for an unhosted expert answer "no" instead of fabricating
    /// activations — the router then walks to the next candidate.
    pub fn contains(&self, expert_id: &str) -> bool {
        self.is_resident(expert_id) || self.path_for(expert_id).is_file()
    }

    /// Is this expert resident right now (no disk touch)?
    pub fn is_resident(&self, expert_id: &str) -> bool {
        self.cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entries
            .contains_key(expert_id)
    }

    /// Resident bytes right now.
    pub fn resident_bytes(&self) -> usize {
        self.cache.lock().unwrap_or_else(|e| e.into_inner()).bytes
    }

    pub fn stats(&self) -> StoreStatsSnapshot {
        StoreStatsSnapshot {
            hits: self.stats.hits.load(Ordering::Relaxed),
            misses: self.stats.misses.load(Ordering::Relaxed),
            loads: self.stats.loads.load(Ordering::Relaxed),
            evictions: self.stats.evictions.load(Ordering::Relaxed),
            load_errors: self.stats.load_errors.load(Ordering::Relaxed),
        }
    }

    /// Load (or fetch from cache) one expert's weights.
    pub fn get(&self, expert_id: &str) -> Result<Arc<ExpertWeights>> {
        {
            let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            let hit = cache.entries.get(expert_id).map(|e| Arc::clone(&e.weights));
            if let Some(weights) = hit {
                // LRU touch: move to the back (most recently used).
                cache.order.retain(|id| id != expert_id);
                cache.order.push_back(expert_id.to_string());
                drop(cache);
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(weights);
            }
        }
        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        let path = self.path_for(expert_id);
        let weights = Arc::new(ExpertWeights::read_from(&path).inspect_err(|_| {
            self.stats.load_errors.fetch_add(1, Ordering::Relaxed);
        })?);
        anyhow::ensure!(
            weights.id == expert_id,
            "expert file {} holds id {:?}, not {expert_id:?} — refusing to serve the wrong expert",
            path.display(),
            weights.id
        );
        self.stats.loads.fetch_add(1, Ordering::Relaxed);
        // Oversized for the budget → serve it without caching rather than
        // thrash the whole resident set.
        if weights.bytes() > self.cache_budget {
            return Ok(weights);
        }
        let bytes = weights.bytes();
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        cache.entries.insert(
            expert_id.to_string(),
            CacheEntry {
                weights: Arc::clone(&weights),
                bytes,
            },
        );
        cache.order.retain(|id| id != expert_id);
        cache.order.push_back(expert_id.to_string());
        cache.bytes += bytes;
        while cache.bytes > self.cache_budget {
            let Some(victim) = cache.order.pop_front() else {
                break;
            };
            if let Some(entry) = cache.entries.remove(&victim) {
                cache.bytes = cache.bytes.saturating_sub(entry.bytes);
                self.stats.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }
        Ok(weights)
    }

    /// Store one expert (atomic write, then a fresh load path).
    pub fn put(&self, weights: &ExpertWeights) -> Result<()> {
        let path = self.path_for(&weights.id);
        weights.write_to(&path)?;
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        cache.entries.remove(&weights.id);
        cache.order.retain(|id| id != &weights.id);
        Ok(())
    }

    /// Raw container bytes for one expert (Phase 2 transfer: a peer
    /// pulling a replica). Reading the file rather than a resident copy
    /// keeps chunk serving allocation-cheap; the OS page cache makes the
    /// hot path cheap anyway.
    ///
    /// `offset`/`len` are clamped to the file, so a requester can never
    /// read past the end and an oversized ask is a short read rather than
    /// an error (the client detects completion by `total_bytes`).
    pub fn bytes_of(&self, expert_id: &str, offset: u64, len: u32) -> Result<(Vec<u8>, u64)> {
        let path = self.path_for(expert_id);
        let bytes =
            std::fs::read(&path).with_context(|| format!("reading expert {}", path.display()))?;
        let total = bytes.len() as u64;
        let start = offset.min(total) as usize;
        let end = (start + len as usize).min(bytes.len());
        Ok((bytes[start..end].to_vec(), total))
    }

    /// Ingest a received expert container (Phase 2 replication). The bytes
    /// are parsed **before** anything touches the store, so a truncated or
    /// tampered transfer is rejected by the container checksum and the
    /// published file is always a whole, valid expert.
    pub fn put_bytes(&self, bytes: &[u8]) -> Result<ExpertWeights> {
        let weights = ExpertWeights::from_bytes(bytes)?;
        self.put(&weights)?;
        Ok(weights)
    }

    /// Serve one expert query: load on demand, run the SwiGLU forward pass.
    pub fn run(&self, expert_id: &str, input: &[f32]) -> Result<Vec<f32>> {
        let weights = self.get(expert_id)?;
        weights.forward(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_store(budget_mb: usize) -> (PathBuf, ExpertStore) {
        static N: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "jerith-expert-store-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let store = ExpertStore::new(&dir, budget_mb).unwrap();
        (dir, store)
    }

    /// Reference SwiGLU written independently of the implementation:
    /// explicit loops, no shared helpers.
    fn reference_forward(w: &ExpertWeights, x: &[f32]) -> Vec<f32> {
        let mut act = vec![0.0f32; w.intermediate];
        for unit in 0..w.intermediate {
            let mut g = 0.0f32;
            let mut u = 0.0f32;
            for k in 0..w.hidden {
                let xk = x[k];
                g += w.gate[unit * w.hidden + k] * xk;
                u += w.up[unit * w.hidden + k] * xk;
            }
            let s = g / (1.0 + (-g).exp());
            act[unit] = s * u;
        }
        let mut y = vec![0.0f32; w.hidden];
        for out in 0..w.hidden {
            let mut acc = 0.0f32;
            for unit in 0..w.intermediate {
                acc += w.down[out * w.intermediate + unit] * act[unit];
            }
            y[out] = acc;
        }
        y
    }

    #[test]
    fn forward_matches_independent_reference() {
        let w = ExpertWeights::synthetic("L0.E1", 6, 4);
        let x = vec![0.5f32, -0.25, 1.0, 0.0, -1.5, 2.0];
        let got = w.forward(&x).unwrap();
        let want = reference_forward(&w, &x);
        assert_eq!(got.len(), want.len());
        for (g, wv) in got.iter().zip(&want) {
            assert!((g - wv).abs() < 1e-5, "got {g}, want {wv}");
        }
    }

    #[test]
    fn forward_is_deterministic_and_nonconstant_across_experts() {
        let a = ExpertWeights::synthetic("L0.E1", 8, 5);
        let b = ExpertWeights::synthetic("L0.E2", 8, 5);
        let x = vec![0.3f32; 8];
        assert_eq!(a.forward(&x).unwrap(), a.forward(&x).unwrap());
        assert_ne!(
            a.forward(&x).unwrap(),
            b.forward(&x).unwrap(),
            "distinct experts must compute distinct activations"
        );
    }

    #[test]
    fn forward_rejects_wrong_input_width() {
        let w = ExpertWeights::synthetic("L0.E1", 4, 3);
        let err = w.forward(&[1.0, 2.0]).unwrap_err().to_string();
        assert!(err.contains("expected hidden=4"), "got {err}");
    }

    #[test]
    fn roundtrip_through_disk_is_exact() {
        let (dir, store) = tmp_store(16);
        let w = ExpertWeights::synthetic("L3.E17", 5, 3);
        store.put(&w).unwrap();
        assert_eq!(store.hosted(), vec!["L3.E17".to_string()]);
        let loaded = store.get("L3.E17").unwrap();
        assert_eq!(*loaded, w);
        // And the served activation matches the in-memory reference.
        let x = vec![0.1f32, -0.2, 0.3, -0.4, 0.5];
        let served = store.run("L3.E17", &x).unwrap();
        assert_eq!(served, reference_forward(&w, &x));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn corrupt_and_mismatched_files_fail_loudly() {
        let (dir, store) = tmp_store(16);
        let w = ExpertWeights::synthetic("L1.E2", 4, 3);
        store.put(&w).unwrap();

        // Bit flip inside the payload → checksum mismatch.
        let path = store.dir().join("L1.E2.expert");
        let mut bytes = std::fs::read(&path).unwrap();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0x01;
        std::fs::write(&path, &bytes).unwrap();
        let err = format!("{:#}", store.get("L1.E2").unwrap_err());
        assert!(err.contains("checksum mismatch"), "got {err}");

        // Truncated file → typed truncation error, not a panic.
        std::fs::write(&path, &bytes[..30]).unwrap();
        let err = format!("{:#}", store.get("L1.E2").unwrap_err());
        assert!(err.contains("truncated"), "got {err}");

        // Wrong magic → refused before any parsing.
        std::fs::write(&path, b"not an expert file at all............").unwrap();
        let err = format!("{:#}", store.get("L1.E2").unwrap_err());
        assert!(err.contains("bad magic"), "got {err}");

        // Missing expert → error naming it (the router treats this as
        // "not hosted here" and moves to the next candidate).
        let err = format!("{:#}", store.get("L9.E9").unwrap_err());
        assert!(err.contains("L9.E9"), "got {err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn misnamed_file_is_refused_rather_than_served() {
        let (dir, store) = tmp_store(16);
        let w = ExpertWeights::synthetic("L1.E2", 4, 3);
        // Correct contents, wrong filename → stem claims a different id.
        w.write_to(&store.dir().join("L9.E9.expert")).unwrap();
        let err = format!("{:#}", store.get("L9.E9").unwrap_err());
        assert!(
            err.contains("refusing to serve the wrong expert"),
            "got {err}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn cache_serves_repeats_and_evicts_by_lru_under_budget() {
        // Each synthetic 4×3 expert is (4*3*3)*4 = 144 bytes of payload.
        let (dir, store) = tmp_store(0); // budget 0 → never cache
        for id in ["L0.E1", "L0.E2"] {
            store.put(&ExpertWeights::synthetic(id, 4, 3)).unwrap();
        }
        let x = vec![1.0f32; 4];
        store.run("L0.E1", &x).unwrap();
        store.run("L0.E1", &x).unwrap();
        let s = store.stats();
        assert_eq!((s.hits, s.loads), (0, 2), "zero budget must never cache");
        assert!(!store.is_resident("L0.E1"));

        // Now a budget that fits exactly one expert: repeated use of A
        // keeps it (hit), touching B evicts A.
        drop(store);
        let store = ExpertStore::new(&dir, 0).unwrap();
        let one = ExpertWeights::synthetic("L0.E1", 4, 3).bytes();
        let store = ExpertStore {
            dir: store.dir.clone(),
            cache_budget: one + 16,
            cache: Mutex::new(ExpertCache::default()),
            stats: StoreStats::default(),
        };
        store.run("L0.E1", &x).unwrap(); // load
        store.run("L0.E1", &x).unwrap(); // hit
        assert_eq!(store.stats().hits, 1);
        store.run("L0.E2", &x).unwrap(); // load → evicts E1
        assert_eq!(store.stats().evictions, 1);
        assert!(!store.is_resident("L0.E1"));
        assert!(store.is_resident("L0.E2"));
        assert!(store.resident_bytes() <= store.cache_budget_bytes());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn oversized_expert_is_served_without_caching() {
        let (dir, store) = tmp_store(0);
        let w = ExpertWeights::synthetic("big", 64, 64);
        store.put(&w).unwrap();
        let x = vec![0.5f32; 64];
        let out = store.run("big", &x).unwrap();
        assert_eq!(out.len(), 64);
        assert!(!store.is_resident("big"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn hostile_expert_id_cannot_escape_the_store_dir() {
        let (dir, store) = tmp_store(16);
        let evil = "../../etc/passwd";
        store.put(&ExpertWeights::synthetic(evil, 4, 3)).unwrap();
        // Every written path stays a direct child of the store dir —
        // separators are neutralised, so `..` cannot climb out.
        for entry in std::fs::read_dir(&dir).unwrap() {
            let p = entry.unwrap().path();
            assert_eq!(
                p.parent().unwrap(),
                dir.as_path(),
                "escaped: {}",
                p.display()
            );
        }
        // The same id still round-trips (its header id matches).
        let x = vec![0.5f32; 4];
        assert!(store.run(evil, &x).is_ok());

        // Sanitisation is lossy, so two ids can share a stem — the header
        // id check keeps that from ever serving the wrong expert.
        store.put(&ExpertWeights::synthetic("a/b", 4, 3)).unwrap();
        let err = format!("{:#}", store.get("a_b").unwrap_err());
        assert!(
            err.contains("refusing to serve the wrong expert"),
            "got {err}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn byte_ranges_are_clamped_and_put_bytes_round_trips() {
        let (dir, store) = tmp_store(16);
        let w = ExpertWeights::synthetic("L4.E5", 4, 3);
        let container = w.to_bytes().unwrap();
        store.put(&w).unwrap();

        // Whole file, a middle slice, and an over-long ask.
        let (all, total) = store.bytes_of("L4.E5", 0, container.len() as u32).unwrap();
        assert_eq!(total, container.len() as u64);
        assert_eq!(all, container);
        let (mid, _) = store.bytes_of("L4.E5", 16, 8).unwrap();
        assert_eq!(mid, container[16..24]);
        let (tail, _) = store.bytes_of("L4.E5", 8, u32::MAX).unwrap();
        assert_eq!(tail, container[8..], "oversized ask must be a short read");
        // Past the end is an empty read, not a panic or a leak.
        let (past, _) = store
            .bytes_of("L4.E5", container.len() as u64 + 99, 16)
            .unwrap();
        assert!(past.is_empty());

        // A fresh node ingests the transferred bytes and computes identically.
        let (dir2, store2) = tmp_store(16);
        let ingested = store2.put_bytes(&all).unwrap();
        assert_eq!(ingested, w);
        let x = vec![0.1f32, 0.2, -0.3, 0.4];
        assert_eq!(
            store2.run("L4.E5", &x).unwrap(),
            store.run("L4.E5", &x).unwrap()
        );
        // Truncated transfer → refused before anything is published.
        let err = format!("{:#}", store2.put_bytes(&all[..all.len() - 5]).unwrap_err());
        assert!(
            err.contains("truncated") || err.contains("checksum"),
            "got {err}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
        std::fs::remove_dir_all(&dir2).unwrap();
    }

    #[test]
    fn contains_distinguishes_hosted_from_unhosted() {
        let (dir, store) = tmp_store(16);
        store.put(&ExpertWeights::synthetic("L2.E3", 4, 3)).unwrap();
        assert!(store.contains("L2.E3"), "hosted expert must be found");
        assert!(!store.contains("L2.E4"), "unhosted expert must not be");
        assert!(
            !store.contains("../../nope"),
            "path traversal must not resolve"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn empty_store_reports_no_hosted_experts() {
        let (dir, store) = tmp_store(4);
        assert!(store.hosted().is_empty());
        assert_eq!(store.stats(), StoreStatsSnapshot::default());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
