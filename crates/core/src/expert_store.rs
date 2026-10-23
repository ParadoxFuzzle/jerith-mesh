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
/// the magic's first four bytes keep catching "this is not an expert file
/// at all" (the last byte carries the written version — see `MAGIC_V2`).
/// v1 files stay readable forever (older shards, `experts make`).
const MAGIC: &[u8; 8] = b"JEXP1\0\0\0";
const MAGIC_V2: &[u8; 8] = b"JEXP2\0\0\0";
const VERSION: u32 = 1;
const VERSION_V2: u32 = 2;
/// Sanity bound on a single expert file: dims are u32 and an activation
/// vector on the wire is capped at 65 536 floats (`quic::MAX_EXPERT_INPUT_FLOATS`),
/// so a legit expert can be big — but not gigabytes-of-dimensions big.
const MAX_DIM: usize = 1 << 20;

/// What the expert's tensors compute. v1 containers are all SwiGlu; v2
/// adds the plain-MLP kind real models ship (nomic-bert-moe family) —
/// two tensors, no gate, and the activation named inside the file rather
/// than guessed at serve time. A container NEVER fabricates a missing
/// tensor: an Mlp expert's `gate` stays `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertKind {
    /// `down(silu(gate·x) * (up·x))` — gate + up + down.
    SwiGlu,
    /// `act_tagged_down(up·x)` — up + down, activation tagged.
    Mlp,
}

impl ExpertKind {
    /// Tensor count the kind's on-disk layout carries (layout order below).
    fn tensor_count(self) -> usize {
        match self {
            ExpertKind::SwiGlu => 3,
            ExpertKind::Mlp => 2,
        }
    }
    fn tag(self) -> u8 {
        match self {
            ExpertKind::SwiGlu => 0,
            ExpertKind::Mlp => 1,
        }
    }
    fn from_tag(t: u8) -> Result<Self> {
        match t {
            0 => Ok(ExpertKind::SwiGlu),
            1 => Ok(ExpertKind::Mlp),
            other => anyhow::bail!("unknown expert kind tag {other} (this build reads 0/1)"),
        }
    }
}

/// Activation the expert's tensors were trained with — recorded IN the
/// v2 container (never inferred at serve time). Silu is also SwiGlu's
/// choice; Gelu is the bert-family MLP default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Silu,
    Gelu,
}

impl Activation {
    fn tag(self) -> u8 {
        match self {
            Activation::Silu => 0,
            Activation::Gelu => 1,
        }
    }
    fn from_tag(t: u8) -> Result<Self> {
        match t {
            0 => Ok(Activation::Silu),
            1 => Ok(Activation::Gelu),
            other => anyhow::bail!("unknown activation tag {other} (this build reads 0/1)"),
        }
    }
    /// Apply the activation the container names. Gelu is the standard
    /// tanh approximation (the one llama.cpp and vLLM serve by default;
    /// bit-differences vs the erf form are below fp16 quantization noise).
    pub fn apply(self, v: f32) -> f32 {
        match self {
            Activation::Silu => silu(v),
            Activation::Gelu => gelu(v),
        }
    }
}

/// One expert's weights. Row-major, little-endian f32 on disk and in RAM.
///
/// Shapes for a hidden size `h` and intermediate size `i` (MoE FFN):
/// - `gate`: `[i × h]`  (gate projection — SwiGlu kind only, `None` otherwise)
/// - `up`:   `[i × h]`  (up projection)
/// - `down`: `[h × i]`  (down projection)
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertWeights {
    pub id: String,
    pub hidden: usize,
    pub intermediate: usize,
    pub kind: ExpertKind,
    pub activation: Activation,
    pub gate: Option<Vec<f32>>,
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
            kind: ExpertKind::SwiGlu,
            activation: Activation::Silu,
            gate: Some(fill(hidden * intermediate)),
            up: fill(hidden * intermediate),
            down: fill(hidden * intermediate),
        }
    }

    /// A deterministic plain-MLP expert (v2 kind): up + down only, gated
    /// by the activation tag — used by the MLP-kind tests and any tooling
    /// that needs the second shape without importing a model. Same LCG
    /// seed rule as [`synthetic`](Self::synthetic).
    pub fn synthetic_mlp(
        id: &str,
        hidden: usize,
        intermediate: usize,
        activation: Activation,
    ) -> Self {
        Self {
            kind: ExpertKind::Mlp,
            activation,
            gate: None,
            ..Self::synthetic(id, hidden, intermediate)
        }
    }

    /// Bytes this expert occupies resident (payload only; the id string
    /// and Vec headers are noise next to real MoE dimensions).
    pub fn bytes(&self) -> usize {
        self.kind.tensor_count()
            * self.hidden.saturating_mul(self.intermediate)
            * std::mem::size_of::<f32>()
    }

    /// Validate internal consistency before use (a corrupted file must
    /// fail here, not by indexing panic in the forward pass). Kind-shape
    /// honesty is checked too: an Mlp container must not carry a gate
    /// tensor it will never use, and a SwiGlu must carry all three.
    pub fn validate(&self) -> Result<()> {
        let expect = self
            .hidden
            .checked_mul(self.intermediate)
            .context("expert dimensions overflow")?;
        match self.kind {
            ExpertKind::SwiGlu => {
                let gate_len = self.gate.as_ref().map_or(0, |g| g.len());
                anyhow::ensure!(
                    gate_len == expect && self.up.len() == expect && self.down.len() == expect,
                    "expert {}: tensor sizes {gate_len} {} {} do not match {expect}",
                    self.id,
                    self.up.len(),
                    self.down.len()
                );
            }
            ExpertKind::Mlp => {
                anyhow::ensure!(
                    self.gate.is_none(),
                    "expert {}: Mlp container carries a gate tensor — refuse (never a fake gate)",
                    self.id
                );
                anyhow::ensure!(
                    self.up.len() == expect && self.down.len() == expect,
                    "expert {}: tensor sizes {} {} do not match {expect}",
                    self.id,
                    self.up.len(),
                    self.down.len()
                );
            }
        }
        Ok(())
    }

    /// Forward pass, kind-aware:
    /// - SwiGlu: `down(silu(gate·x) * (up·x))`
    /// - Mlp: `down(act(up·x))`, `act` the tagged [`Activation`]
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
        // h-combined values per intermediate unit.
        let mut act = vec![0.0f32; i];
        match self.kind {
            ExpertKind::SwiGlu => {
                let gate = self.gate.as_ref().expect("validated SwiGlu carries a gate");
                for (unit, slot) in act.iter_mut().enumerate() {
                    let g = dot(&gate[unit * h..(unit + 1) * h], x);
                    let u = dot(&self.up[unit * h..(unit + 1) * h], x);
                    *slot = silu(g) * u;
                }
            }
            ExpertKind::Mlp => {
                for (unit, slot) in act.iter_mut().enumerate() {
                    let u = dot(&self.up[unit * h..(unit + 1) * h], x);
                    *slot = self.activation.apply(u);
                }
            }
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

/// The standard tanh GELU (llama.cpp/vLLM's default gelu) — exact-odd
/// function, max |diff| vs the erf form ~1e-3 relative at the fp16-scale
/// weights this format stores.
fn gelu(v: f32) -> f32 {
    const K: f32 = 0.797_884_6_f32; // √(2/π)
    let inner = v * (1.0 + 0.044_715_f32 * v * v) * K;
    0.5 * v * (1.0 + inner.tanh())
}
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
    /// layout; the checksum covers everything before it). SwiGlu experts
    /// write v1 (bit-compatible with older files), Mlp experts require v2.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let (magic, version) = match self.kind {
            ExpertKind::SwiGlu => (MAGIC, VERSION),
            ExpertKind::Mlp => (MAGIC_V2, VERSION_V2),
        };
        let mut out = Vec::with_capacity(64 + self.bytes());
        out.extend_from_slice(magic);
        out.extend_from_slice(&version.to_le_bytes());
        out.extend_from_slice(&(self.id.len() as u32).to_le_bytes());
        out.extend_from_slice(self.id.as_bytes());
        out.extend_from_slice(&(self.hidden as u32).to_le_bytes());
        out.extend_from_slice(&(self.intermediate as u32).to_le_bytes());
        if version == VERSION_V2 {
            out.push(self.kind.tag());
            out.push(self.activation.tag());
        }
        match self.kind {
            ExpertKind::SwiGlu => {
                let gate = self.gate.as_ref().expect("validated SwiGlu carries a gate");
                for t in [gate, &self.up, &self.down] {
                    for v in t {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
            ExpertKind::Mlp => {
                for t in [&self.up, &self.down] {
                    for v in t {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
        }
        let sum = checksum(&out);
        out.extend_from_slice(&sum.to_le_bytes());
        Ok(out)
    }

    /// Parse a container. Every failure mode is a typed error: bad magic,
    /// truncated body, dimensions that overflow, tensor-length mismatch,
    /// unknown kind/activation tags, or checksum disagreement (bit rot /
    /// partial write). v1 files parse as synthetic SwiGlu/Silu experts.
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
        // Header: the versioned 8-byte magic ("JEXP1\0\0\0" / "JEXP2\0\0\0"),
        // then the explicit version u32 the writer lays down after both —
        // v1 shards stay bit-readable forever.
        let magic8 = take(&mut cur, 8)?;
        let version = if magic8 == MAGIC {
            VERSION
        } else if magic8 == MAGIC_V2 {
            VERSION_V2
        } else {
            anyhow::bail!("not an expert file (bad magic)")
        };
        let version_field = u32::from_le_bytes(take(&mut cur, 4)?.try_into().unwrap());
        anyhow::ensure!(
            version_field == version,
            "expert file version {version_field} unsupported (this build reads {VERSION}–{VERSION_V2})"
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
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect())
        };
        let mut kind = ExpertKind::SwiGlu;
        let mut activation = Activation::Silu;
        if version == VERSION_V2 {
            let tags = take(&mut cur, 2)?;
            kind = ExpertKind::from_tag(tags[0])?;
            activation = Activation::from_tag(tags[1])?;
        }
        let gate = if version == VERSION {
            Some(read_tensor(&mut cur)?)
        } else {
            match kind {
                ExpertKind::SwiGlu => Some(read_tensor(&mut cur)?),
                ExpertKind::Mlp => None,
            }
        };
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
            kind,
            activation,
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

// ---------------------------------------------------------------------------
// Router containers — one MoE layer's real routing weights
// (`blk.<n>.ffn_gate_inp.weight` from the GGUF, the top-k scorer),
// separate from the expert files so the store's expert listing/passwords
// stay experts. `.router` files on disk, kind-1 wire, self-describing.
// ---------------------------------------------------------------------------

/// Router file magic + version.
const ROUTER_MAGIC: &[u8; 8] = b"JROUTE1\0";

/// One MoE layer's router: `[e × h]` row-major routing weights that score
/// each expert for a hidden-wide token (`rows·x` → `e` logits). `used`
/// records the model's top-k when the GGUF names it — advisory metadata
/// for walkers, never a serve-time assumption.
#[derive(Debug, Clone, PartialEq)]
pub struct RouterWeights {
    /// Layer id, e.g. `L1.router`.
    pub id: String,
    /// Expert count = output width.
    pub experts: usize,
    /// Hidden width = input width.
    pub hidden: usize,
    /// Top-k experts per token, when the source model declares it.
    pub used: Option<usize>,
    /// `[e × h]` row-major: row `e` scores expert `e`.
    pub rows: Vec<f32>,
}

impl RouterWeights {
    /// Deterministic synthetic router (same seed family as experts) so
    /// mesh smoke tests can pace router pulls without a model import.
    pub fn synthetic(id: &str, experts: usize, hidden: usize, used: Option<usize>) -> Self {
        fn lcg(seed: &mut u64) -> f32 {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let x = (*seed >> 33) as u32;
            (x as f32 / u32::MAX as f32) * 0.2 - 0.1
        }
        let mut seed: u64 = 0x5eed;
        for b in id.as_bytes() {
            seed = seed.wrapping_mul(0x100000001b3) ^ (*b as u64);
        }
        Self {
            id: id.to_string(),
            experts,
            hidden,
            used,
            rows: (0..experts * hidden).map(|_| lcg(&mut seed)).collect(),
        }
    }

    pub fn bytes(&self) -> usize {
        self.experts.saturating_mul(self.hidden) * std::mem::size_of::<f32>()
    }

    pub fn validate(&self) -> Result<()> {
        let expect = self
            .experts
            .checked_mul(self.hidden)
            .context("router dimensions overflow")?;
        anyhow::ensure!(
            self.experts > 0 && self.hidden > 0,
            "router {}: degenerate dimensions",
            self.id
        );
        anyhow::ensure!(
            self.rows.len() == expect,
            "router {}: rows {} do not match {expect}",
            self.id,
            self.rows.len()
        );
        Ok(())
    }

    /// Score one token: `rows·x` → `experts` logits (softmax is the
    /// caller's job — the raw scores are what top-k needs).
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>> {
        anyhow::ensure!(
            x.len() == self.hidden,
            "router {}: input has {} floats, expected hidden={}",
            self.id,
            x.len(),
            self.hidden
        );
        Ok(self
            .rows
            .chunks(self.hidden)
            .map(|row| dot(row, x))
            .collect())
    }

    /// Expert k's row (`[h]` slice of the routing matrix).
    pub fn row(&self, expert_k: usize) -> Result<&[f32]> {
        anyhow::ensure!(
            expert_k < self.experts,
            "router {}: expert index {expert_k} out of range ({})",
            self.id,
            self.experts
        );
        Ok(&self.rows[expert_k * self.hidden..(expert_k + 1) * self.hidden])
    }

    /// Serialize (same checksum rule as the expert containers).
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut out = Vec::with_capacity(48 + self.bytes());
        out.extend_from_slice(ROUTER_MAGIC);
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&(self.id.len() as u32).to_le_bytes());
        out.extend_from_slice(self.id.as_bytes());
        out.extend_from_slice(&(self.experts as u32).to_le_bytes());
        out.extend_from_slice(&(self.hidden as u32).to_le_bytes());
        // u32::MAX = "top-k unknown" (the source model never declared it).
        out.extend_from_slice(&(self.used.unwrap_or(u32::MAX as usize) as u32).to_le_bytes());
        for v in &self.rows {
            out.extend_from_slice(&v.to_le_bytes());
        }
        let sum = checksum(&out);
        out.extend_from_slice(&sum.to_le_bytes());
        Ok(out)
    }

    /// Parse a router container. Typed failures like the expert parser:
    /// magic, version, truncation, dims, checksum.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut cur = 0usize;
        let mut take = |n: usize| -> anyhow::Result<&[u8]> {
            anyhow::ensure!(
                bytes.len() >= cur + n,
                "router file truncated: wanted {n} bytes at {cur}, file is {}",
                bytes.len()
            );
            let s = &bytes[cur..cur + n];
            cur += n;
            Ok(s)
        };
        anyhow::ensure!(take(8)? == ROUTER_MAGIC, "not a router file (bad magic)");
        let version = u32::from_le_bytes(take(4)?.try_into().unwrap());
        anyhow::ensure!(
            version == 1,
            "router file version {version} unsupported (this build reads 1)"
        );
        let id_len = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
        let id =
            String::from_utf8(take(id_len)?.to_vec()).context("router id is not valid UTF-8")?;
        let experts = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
        let hidden = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
        let used_raw = u32::from_le_bytes(take(4)?.try_into().unwrap());
        let used = if used_raw == u32::MAX {
            None
        } else {
            Some(used_raw as usize)
        };
        anyhow::ensure!(
            experts > 0 && hidden > 0 && experts <= MAX_DIM && hidden <= MAX_DIM,
            "router {id}: implausible dimensions {experts}×{hidden}"
        );
        let expect = experts
            .checked_mul(hidden)
            .context("router dimensions overflow")?;
        let raw = take(expect * 4)?;
        let rows = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let sum = u64::from_le_bytes(take(8)?.try_into().unwrap());
        anyhow::ensure!(
            checksum(&bytes[..cur - 8]) == sum,
            "router {id}: checksum mismatch (corrupted or partially written file)"
        );
        let r = Self {
            id,
            experts,
            hidden,
            used,
            rows,
        };
        r.validate()?;
        Ok(r)
    }

    /// Atomic write, same temp+rename rule as the expert containers.
    pub fn write_to(&self, path: &Path) -> Result<()> {
        let bytes = self.to_bytes()?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)
                .with_context(|| format!("creating router dir {}", dir.display()))?;
        }
        let tmp = path.with_extension("router.tmp");
        std::fs::write(&tmp, &bytes)
            .with_context(|| format!("writing router temp file {}", tmp.display()))?;
        std::fs::rename(&tmp, path)
            .with_context(|| format!("publishing router file {}", path.display()))?;
        Ok(())
    }
}

/// Expert / router ids are mesh-supplied strings; keep them to a single
/// path component so a hostile id can't escape the store dir.
fn sanitize_id(id: &str) -> String {
    id.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
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
    /// Speculative preloads (Phase 3): one per `prefetch` that ended with
    /// the expert resident (already-resident prefetches count too —
    /// the request was served by the cache either way).
    pub prefetches: AtomicU64,
    /// Prefetch attempts whose load failed (unhosted/corrupt expert).
    pub prefetch_errors: AtomicU64,
}

/// Point-in-time copy of [`StoreStats`] (cheap to render).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StoreStatsSnapshot {
    pub hits: u64,
    pub misses: u64,
    pub loads: u64,
    pub evictions: u64,
    pub load_errors: u64,
    pub prefetches: u64,
    pub prefetch_errors: u64,
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
    /// §1 phase 2 (draft hot set): the ALWAYS-RESIDENT tier — recomputed
    /// from `[experts].draft_hot_percent` every replication tick. Pinned
    /// ids are never eviction victims (the governor's full shed keeps
    /// them too) and are eagerly loaded from disk when the pin lands, so
    /// a draft pass never pays first-touch disk.
    pinned: std::sync::Mutex<std::collections::HashSet<String>>,
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
            pinned: std::sync::Mutex::new(std::collections::HashSet::new()),
        })
    }

    /// §1 phase 2: REPLACE the pinned set with `ids` (the knob's current
    /// derivation — stale pins from an old percent fade out every tick)
    /// and eagerly load any pinned expert that is not resident yet. Load
    /// failures are honest no-ops (an unhosted/corrupt id stays pinned
    /// but never becomes resident; the next tick with a sane set fixes
    /// the roster). Returns how many pins are now resident.
    pub fn apply_draft_pins(&self, ids: &[String]) -> usize {
        *self.pinned.lock().unwrap_or_else(|e| e.into_inner()) = ids.iter().cloned().collect();
        let mut resident = 0usize;
        for id in ids {
            if self.is_resident(id) {
                resident += 1;
                continue;
            }
            if self.load_from_disk_and_cache(id).is_ok() {
                resident += 1;
            }
        }
        resident
    }

    /// The pinned ids (sorted — /status renders them).
    pub fn pinned_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self
            .pinned
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .cloned()
            .collect();
        ids.sort();
        ids
    }

    pub fn is_pinned(&self, expert_id: &str) -> bool {
        self.pinned
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(expert_id)
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
        self.dir.join(format!("{}.expert", sanitize_id(expert_id)))
    }

    fn router_path_for(&self, router_id: &str) -> PathBuf {
        self.dir.join(format!("{}.router", sanitize_id(router_id)))
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

    /// Shed resident experts until at most `max_bytes` stay hot (LRU order,
    /// coldest first). The resource governor calls this at the RAM soft
    /// line: the hot set is the node's biggest *discretionary* allocation,
    /// and every evicted expert is re-read on demand — shedding costs a
    /// disk hit, not correctness. Returns the bytes evicted.
    pub fn evict_to(&self, max_bytes: usize) -> usize {
        let mut shed = 0usize;
        {
            // Lock order fixed cache → pinned everywhere (the load path
            // holds the cache lock while consulting pins — a pinned→cache
            // take here would deadlock the two paths against each other).
            let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            let pinned = self.pinned.lock().unwrap_or_else(|e| e.into_inner());
            while cache.bytes > max_bytes {
                // Pinned experts are NOT victims: the draft tier survives
                // even the governor's full shed. When every resident id is
                // pinned the shed stops — pins may exceed the cache
                // budget, that is the operator's ask, stated loudly here.
                let Some(pos) = cache.order.iter().position(|id| !pinned.contains(id)) else {
                    break;
                };
                let victim = cache.order.remove(pos).expect("position just checked");
                if let Some(entry) = cache.entries.remove(&victim) {
                    cache.bytes = cache.bytes.saturating_sub(entry.bytes);
                    shed += entry.bytes;
                    self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        if shed > 0 {
            tracing::info!(
                target: "mesh",
                shed_bytes = shed,
                "resource governor: shed expert hot-set under memory pressure"
            );
        }
        shed
    }

    pub fn stats(&self) -> StoreStatsSnapshot {
        StoreStatsSnapshot {
            hits: self.stats.hits.load(Ordering::Relaxed),
            misses: self.stats.misses.load(Ordering::Relaxed),
            loads: self.stats.loads.load(Ordering::Relaxed),
            evictions: self.stats.evictions.load(Ordering::Relaxed),
            load_errors: self.stats.load_errors.load(Ordering::Relaxed),
            prefetches: self.stats.prefetches.load(Ordering::Relaxed),
            prefetch_errors: self.stats.prefetch_errors.load(Ordering::Relaxed),
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
        self.load_from_disk_and_cache(expert_id)
    }

    /// Shared disk-load path of `get` and `prefetch`: read the container,
    /// re-check the header id against the filename (a misnamed file never
    /// serves the wrong expert), count the load, and insert into the LRU
    /// (properly evicting when over budget; an expert bigger than the
    /// whole budget is returned uncached rather than thrashing the set).
    fn load_from_disk_and_cache(&self, expert_id: &str) -> Result<Arc<ExpertWeights>> {
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
        let pinned = self.pinned.lock().unwrap_or_else(|e| e.into_inner());
        while cache.bytes > self.cache_budget {
            // Same pin rule as `evict_to`: the draft tier is never an
            // over-budget victim; only pinned entries stop the shedding.
            let Some(pos) = cache.order.iter().position(|id| !pinned.contains(id)) else {
                break;
            };
            let victim = cache.order.remove(pos).expect("position just checked");
            if let Some(entry) = cache.entries.remove(&victim) {
                cache.bytes = cache.bytes.saturating_sub(entry.bytes);
                self.stats.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }
        Ok(weights)
    }

    /// Speculative prefetch (Phase 3): bring one expert into the LRU
    /// *without* a serving request behind it. The served hits/misses that
    /// measure the cache belong to serving — a prefetch is booked under
    /// its own counters, and a success later shows up as an honest cache
    /// hit when the real query arrives. Callers must already know the
    /// expert is hosted (prefetch warms, it never fetches mesh weights).
    pub fn prefetch(&self, expert_id: &str) -> Result<()> {
        if self.is_resident(expert_id) {
            // Nothing to load; still a completed prefetch action.
            self.stats.prefetches.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
        match self.load_from_disk_and_cache(expert_id) {
            Ok(_) => {
                self.stats.prefetches.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(e) => {
                self.stats.prefetch_errors.fetch_add(1, Ordering::Relaxed);
                Err(e)
            }
        }
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

    /// Router ids this node holds (`.router` filenames; never mixed into
    /// the expert listing — a router is not a forwardable expert).
    pub fn routers(&self) -> Vec<String> {
        let mut ids: Vec<String> = std::fs::read_dir(&self.dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                let p = e.path();
                if p.extension().and_then(|x| x.to_str()) != Some("router") {
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

    /// Store one router container (atomic write; routers are tiny — reads
    /// go to disk each time, no LRU entry).
    pub fn put_router(&self, router: &RouterWeights) -> Result<()> {
        router.write_to(&self.router_path_for(&router.id))
    }

    /// Read one router container (filename stem must match the header id,
    /// the same misname-refusal rule the expert loaders enforce).
    pub fn get_router(&self, router_id: &str) -> Result<RouterWeights> {
        let path = self.router_path_for(router_id);
        let bytes =
            std::fs::read(&path).with_context(|| format!("reading router {}", path.display()))?;
        let r = RouterWeights::from_bytes(&bytes)
            .with_context(|| format!("parsing router {}", path.display()))?;
        anyhow::ensure!(
            r.id == router_id,
            "router file {} holds id {:?}, not {router_id:?} — refusing to serve the wrong router",
            path.display(),
            r.id
        );
        Ok(r)
    }

    /// Does this node hold a router for this layer id (early boot / one-stat)?
    pub fn contains_router(&self, router_id: &str) -> bool {
        self.router_path_for(router_id).is_file()
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
    /// explicit loops over sliced rows, no shared helpers.
    fn reference_forward(w: &ExpertWeights, x: &[f32]) -> Vec<f32> {
        let mut act: Vec<f32> = Vec::with_capacity(w.intermediate);
        for (gate_row, up_row) in w
            .gate
            .as_ref()
            .unwrap()
            .chunks(w.hidden)
            .zip(w.up.chunks(w.hidden))
        {
            let mut g = 0.0f32;
            let mut u = 0.0f32;
            for (gr, xk) in gate_row.iter().zip(x) {
                g += gr * xk;
            }
            for (ur, xk) in up_row.iter().zip(x) {
                u += ur * xk;
            }
            let s = g / (1.0 + (-g).exp());
            act.push(s * u);
        }
        let mut y: Vec<f32> = Vec::with_capacity(w.hidden);
        for down_row in w.down.chunks(w.intermediate) {
            let mut acc = 0.0f32;
            for (d, a) in down_row.iter().zip(&act) {
                acc += d * a;
            }
            y.push(acc);
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
            pinned: Mutex::new(std::collections::HashSet::new()),
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
    fn draft_pins_survive_full_shed_and_budget_pressure_and_clear() {
        let (dir, store) = tmp_store(16);
        for id in ["L0.E0", "L0.E1", "L1.E0", "L9.E9"] {
            store.put(&ExpertWeights::synthetic(id, 4, 3)).unwrap();
        }
        // Pin the draft tier (the lowest-layer slice): the pin lands and
        // all four ids load EAGERLY — a draft pass never pays disk.
        let resident = store.apply_draft_pins(&["L0.E0".into(), "L0.E1".into()]);
        assert_eq!(resident, 2, "pinned ids load eagerly on apply");
        assert!(store.is_resident("L0.E0") && store.is_resident("L0.E1"));
        assert_eq!(store.pinned_ids(), vec!["L0.E0", "L0.E1"]);

        // Give the eviction some non-pinned residents to shed.
        store.run("L1.E0", &[1.0f32; 4]).unwrap();
        store.run("L9.E9", &[1.0f32; 4]).unwrap();
        assert!(store.is_resident("L1.E0") && store.is_resident("L9.E9"));

        // The resource governor's FULL shed keeps the pins resident.
        let shed = store.evict_to(0);
        assert!(shed > 0, "non-pinned resident bytes were shed");
        assert!(store.is_resident("L0.E0") && store.is_resident("L0.E1"));

        // Budget pressure mid-load behaves the same: an unpinned load
        // evicts LRU unpinned entries, never a pinned one.
        store.run("L1.E0", &[1.0f32; 4]).unwrap();
        store.run("L9.E9", &[1.0f32; 4]).unwrap();
        assert!(store.is_resident("L0.E0") && store.is_resident("L0.E1"));
        assert_eq!(
            store.pinned_ids(),
            vec!["L0.E0", "L0.E1"],
            "pins untouched by eviction"
        );

        // Knob off: the empty re-derivation fades the pins out.
        store.apply_draft_pins(&[]);
        assert!(store.pinned_ids().is_empty());
        store.evict_to(0);
        assert!(!store.is_resident("L0.E0"), "unpinned shed like any entry");
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

    #[test]
    fn prefetch_warms_the_lru_without_touching_serving_counters() {
        let (dir, store) = tmp_store(16);
        store.put(&ExpertWeights::synthetic("L0.E1", 4, 3)).unwrap();
        store.put(&ExpertWeights::synthetic("L1.E2", 4, 3)).unwrap();
        assert_eq!(store.stats(), StoreStatsSnapshot::default());

        // Prefetch: the expert is resident afterwards, but NO serving
        // counter moved — hits/misses belong to real queries.
        store.prefetch("L1.E2").unwrap();
        assert!(store.is_resident("L1.E2"));
        let s = store.stats();
        assert_eq!(s.prefetches, 1);
        assert_eq!((s.hits, s.misses), (0, 0));
        assert_eq!(s.loads, 1, "a prefetch on a cold expert is a disk load");

        // The warmed expert's first REAL query is an honest cache hit —
        // the prefetch paid the disk cost instead of the requester.
        assert_eq!(store.get("L1.E2").unwrap().id, "L1.E2");
        assert_eq!(store.stats().hits, 1);

        // Prefetch of an already-resident expert: counted, zero loads.
        store.prefetch("L1.E2").unwrap();
        assert_eq!(store.stats().prefetches, 2);
        assert_eq!(store.stats().loads, 1);

        // Prefetch of something unhosted: typed error + its own counter.
        // The shared load path records its load_errors too (exactly what
        // a get() of the same id would show) — but no hit, no miss, and
        // no serving-side counter pretends a query happened.
        let err = format!("{:#}", store.prefetch("L9.E9").unwrap_err());
        assert!(err.contains("L9.E9"), "got {err}");
        let s = store.stats();
        assert_eq!(s.prefetch_errors, 1);
        assert_eq!(s.misses, 0, "no serving miss was counted by prefetch");
        assert_eq!(s.hits, 1, "only the serve's own hit remains");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The resource governor's shedding hook: `evict_to` must drop the
    /// LRU-coldest experts until the resident budget holds, count real
    /// evictions, and leave hosted experts re-loadable from disk.
    #[test]
    fn evict_to_sheds_resident_hot_set_and_keeps_experts_hosted() {
        // Each synthetic 4×3 expert is 144 bytes of payload — one byte
        // over the first budget, so two can't stay resident together.
        let (dir, store) = tmp_store(0);
        for id in ["L0.E1", "L0.E2", "L0.E3"] {
            store.put(&ExpertWeights::synthetic(id, 4, 3)).unwrap();
        }
        // Big budget picture first: load all three via the cache path.
        drop(store);
        let store = ExpertStore::new(&dir, 1).unwrap(); // 1 MiB — fits all
        let x = vec![1.0f32; 4];
        for id in ["L0.E1", "L0.E2", "L0.E3"] {
            store.run(id, &x).unwrap();
        }
        assert_eq!(store.resident_bytes(), 3 * 144);

        // Shed to two experts: one eviction, bytes drop, hosted stays.
        let before = store.stats().evictions;
        let shed = store.evict_to(2 * 144);
        assert_eq!(shed, 144, "the single coldest expert was shed");
        assert_eq!(store.resident_bytes(), 2 * 144);
        assert_eq!(store.stats().evictions, before + 1);
        for id in ["L0.E1", "L0.E2", "L0.E3"] {
            assert!(
                store.contains(id),
                "shedding never unhosts {id} — it only drops the RAM copy"
            );
        }

        // Shed to zero: fully cold, still loadable (disk hit, not a miss
        // of the expert itself).
        let shed_all = store.evict_to(0);
        assert_eq!(shed_all, 2 * 144);
        assert_eq!(store.resident_bytes(), 0);
        for id in ["L0.E1", "L0.E2", "L0.E3"] {
            assert!(store.contains(id));
        }
        store.run("L0.E1", &x).unwrap();
        assert!(store.is_resident("L0.E1"), "re-read from disk works");

        // Idempotent: the re-read above re-resident E1 (144 B), so the
        // next shed evicts exactly it — and the one after is a no-op.
        assert_eq!(store.evict_to(0), 144);
        assert_eq!(store.evict_to(0), 0, "shedding an empty set is a no-op");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    // ---- v2: the plain-MLP kind + activation tag + router containers ----

    /// Independent MLP reference (explicit rows, no shared helpers).
    fn reference_mlp(w: &ExpertWeights, x: &[f32]) -> Vec<f32> {
        let act: Vec<f32> =
            w.up.chunks(w.hidden)
                .map(|up_row| up_row.iter().zip(x).map(|(u, k)| u * k).sum::<f32>())
                .map(|v| match w.activation {
                    Activation::Silu => v / (1.0 + (-v).exp()),
                    Activation::Gelu => {
                        let inner = v * (1.0 + 0.044_715_f32 * v * v) * 0.797_884_6_f32;
                        0.5 * v * (1.0 + inner.tanh())
                    }
                })
                .collect();
        w.down
            .chunks(w.intermediate)
            .map(|down_row| down_row.iter().zip(&act).map(|(d, a)| d * a).sum::<f32>())
            .collect()
    }

    #[test]
    fn mlp_experts_forward_match_the_tagged_activation() {
        let x = vec![0.4f32, -0.7, 1.1, 0.05];
        for activation in [Activation::Gelu, Activation::Silu] {
            let w = ExpertWeights::synthetic_mlp("L3.E1", 4, 3, activation);
            let got = w.forward(&x).unwrap();
            let want = reference_mlp(&w, &x);
            for (g, wv) in got.iter().zip(&want) {
                assert!((g - wv).abs() < 1e-5, "{activation:?}: {g} vs {wv}");
            }
            assert_eq!(got.len(), x.len(), "MLP output is hidden-wide");
        }
    }

    #[test]
    fn mlp_container_roundtrips_and_refuses_a_fabricated_gate() {
        let w = ExpertWeights::synthetic_mlp("L2.E7", 4, 3, Activation::Gelu);
        let bytes = w.to_bytes().unwrap();

        // v2 round-trip: kind, activation, and the two-tensor shape survive.
        let parsed = ExpertWeights::from_bytes(&bytes).unwrap();
        assert_eq!(parsed, w);
        assert_eq!(parsed.kind, ExpertKind::Mlp);
        assert_eq!(parsed.activation, Activation::Gelu);
        assert!(parsed.gate.is_none(), "MLP containers never fake a gate");

        // An Mlp container slapped a gate onto: refused at validate, not
        // served with shape-blind math.
        let mut forged = w.clone();
        forged.gate = Some(vec![0.0f32; 12]);
        let err = forged.validate().unwrap_err().to_string();
        assert!(err.contains("never a fake gate"), "got {err}");

        // Unknown kind/activation tags are typed refusals (the kind tag
        // sits right after magic(+8)+version(+4)+id_len(+4)+id(+5 for
        // "L2.E7")+hidden(+4)+intermediate(+4)).
        let tag_pos = 8 + 4 + 4 + "L2.E7".len() + 4 + 4;
        let mut bad_tag = bytes.clone();
        bad_tag[tag_pos] = 9;
        let err = ExpertWeights::from_bytes(&bad_tag).unwrap_err().to_string();
        assert!(err.contains("expert kind tag 9"), "got {err}");
    }

    #[test]
    fn v1_files_stay_readable_as_swiglu_experts() {
        // A v1 container (bit-identical to what `experts make` wrote
        // before v2 existed) parses as a SwiGlu/Silu expert and still
        // computes the v1 math.
        let w = ExpertWeights::synthetic("L0.E1", 4, 3);
        let bytes = w.to_bytes().unwrap();
        assert_eq!(&bytes[0..8], b"JEXP1\0\0\0");
        let parsed = ExpertWeights::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.kind, ExpertKind::SwiGlu);
        assert_eq!(parsed.activation, Activation::Silu);
        let x = vec![0.3f32, -0.2, 0.9, 0.4];
        assert_eq!(parsed.forward(&x).unwrap(), w.forward(&x).unwrap());
    }

    #[test]
    fn router_containers_roundtrip_score_and_refuse_mismatches() {
        let r = RouterWeights::synthetic("L1.router", 6, 4, Some(2));
        assert_eq!(r.bytes(), 6 * 4 * 4);
        let parsed = RouterWeights::from_bytes(&r.to_bytes().unwrap()).unwrap();
        assert_eq!(parsed, r, "v1 router container round-trips");

        // Scoring: rows·x, raw logits (softmax is the caller's job).
        let x = vec![0.5f32, -0.25, 1.0, 0.0];
        let scores = parsed.forward(&x).unwrap();
        assert_eq!(scores.len(), 6);
        for (k, got) in scores.iter().enumerate() {
            let row = parsed.row(k).unwrap();
            let want: f32 = row.iter().zip(&x).map(|(a, b)| a * b).sum();
            assert_eq!(*got, want, "row {k}");
        }
        assert!(parsed.row(6).is_err(), "out-of-range expert refused");
        assert!(
            RouterWeights::synthetic("no-topk", 3, 4, None)
                .used
                .is_none()
        );

        // Wrong input width is a typed refusal.
        let err = parsed.forward(&[1.0, 2.0]).unwrap_err().to_string();
        assert!(err.contains("expected hidden=4"), "got {err}");

        // Corrupt body → checksum mismatch; not-a-router → bad magic.
        let good = r.to_bytes().unwrap();
        let mut bit_flipped = good.clone();
        let mid = bit_flipped.len() / 2;
        bit_flipped[mid] ^= 0x01;
        let err = format!("{:#}", RouterWeights::from_bytes(&bit_flipped).unwrap_err());
        assert!(err.contains("checksum mismatch"), "got {err}");
        let err = format!(
            "{:#}",
            RouterWeights::from_bytes(b"nonsense at all____________").unwrap_err()
        );
        assert!(err.contains("bad magic"), "got {err}");
    }

    #[test]
    fn store_hosts_routers_beside_experts_without_mingling() {
        let (dir, store) = tmp_store(16);
        store.put(&ExpertWeights::synthetic("L0.E1", 4, 3)).unwrap();
        let router = RouterWeights::synthetic("L0.router", 4, 4, None);
        store.put_router(&router).unwrap();

        assert_eq!(store.hosted(), vec!["L0.E1".to_string()]);
        assert_eq!(store.routers(), vec!["L0.router".to_string()]);
        assert!(store.contains("L0.E1"));
        assert!(store.contains_router("L0.router"));
        assert!(!store.contains_router("L1.router"));
        assert!(!store.contains("L0.router"), "a router id is not an expert");

        // Round-trip through the store, with the misname refusal intact.
        let served = store.get_router("L0.router").unwrap();
        assert_eq!(served, router);
        router
            .write_to(&store.dir().join("L9.router.router"))
            .unwrap();
        let err = format!("{:#}", store.get_router("L9.router").unwrap_err());
        assert!(
            err.contains("refusing to serve the wrong router"),
            "got {err}"
        );
        let err = format!("{:#}", store.get_router("L1.router").unwrap_err());
        assert!(err.contains("L1.router"), "got {err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
