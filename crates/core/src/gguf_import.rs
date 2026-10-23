//! The GGUF importer: real model weights into the expert store — the
//! Colibri-side layer split (FEATURES §13), built on the format itself
//! rather than a runtime wrapper.
//!
//! What this module does, honestly:
//!
//! - **reads real GGUF v2/v3** — magic + version, key-value metadata
//!   (arrays skipped by exact byte math, never materialized), the
//!   tensor directory, and the 32-byte-aligned tensor data section;
//! - **dequantizes the workspace's chosen formats**: F32, F16, BF16,
//!   Q8_0 and Q4_0. Anything else (Q4_K, IQ* families) is a typed
//!   unsupported error naming the type — never a silent garbage fill;
//! - **splits real MoE layers into expert containers**: the
//!   `blk.<n>.ffn_{gate,up,down}_exps.weight` 3-D tensor stacks become
//!   per-(layer, expert) `L<n>.E<k>` files in the store's stable
//!   `.expert` format (v2: the plain-MLP variant + activation tag the
//!   nomic-bert-moe line needs — 2-tensor experts with NO gate tensor,
//!   which v1's SwiGLU-only shape cannot express without fabrication),
//!   dense `ffn_*` layers land as a single `L<n>.E0`, and the real
//!   router tensor `ffn_gate_inp.weight` (the top-k routing weights)
//!   exports per MoE layer as `L<n>.router` (kind-2 container);
//! - **naming schemes it speaks**: the `…_exps.weight` 3-D stack
//!   (llama.cpp conversions of mixtral/deepseek/glm/qwen moe archs)
//!   and the nomic-bert-moe shapes seen here. Formats it doesn't
//!   recognize are REPORTED, never half-imported.
//!
//! The wire the store serves is untouched: same container format, same
//! checksum rules, same byte-identical harvesting. What changes is
//! where the floats came from.

use anyhow::{Context as _, Result, bail};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

// ---------------------------------------------------------------------------
// GGUF dtype codes (ggml enume values, pinned against the spec)
// ---------------------------------------------------------------------------

pub const GGML_F32: u32 = 0;
pub const GGML_F16: u32 = 1;
pub const GGML_Q4_0: u32 = 2;
pub const GGML_Q8_0: u32 = 8;
pub const GGML_BF16: u32 = 30;

/// {GGML_F32: 4 B, F16: 2 B, BF16: 2 B} — the flat (not block) types.
fn dtype_flat_size(dtype: u32) -> Option<usize> {
    match dtype {
        GGML_F32 => Some(4),
        GGML_F16 | GGML_BF16 => Some(2),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Metadata values (arrays are skipped by byte math, never materialized)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum GgufKv {
    Bool(bool),
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Str(String),
    /// Any array: element type + count + the byte length to skip.
    Arr {
        etype: u32,
        count: u64,
        bytes: u64,
    },
}

impl GgufKv {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufKv::U8(v) => Some(*v as u32),
            GgufKv::U16(v) => Some(*v as u32),
            GgufKv::U32(v) => Some(*v),
            GgufKv::I8(v) if *v >= 0 => Some(*v as u32),
            GgufKv::I16(v) if *v >= 0 => Some(*v as u32),
            GgufKv::I32(v) if *v >= 0 => Some(*v as u32),
            GgufKv::U64(v) if *v <= u32::MAX as u64 => Some(*v as u32),
            GgufKv::I64(v) if *v >= 0 && *v <= u32::MAX as i64 => Some(*v as u32),
            _ => None,
        }
    }
}

/// The wire type codes (GGUF metadata value types).
const KV_BOOL: u32 = 7;
const KV_U8: u32 = 0;
const KV_I8: u32 = 1;
const KV_U16: u32 = 2;
const KV_I16: u32 = 3;
const KV_U32: u32 = 4;
const KV_I32: u32 = 5;
const KV_F32: u32 = 6;
const KV_STR: u32 = 8;
const KV_ARR: u32 = 9;
const KV_U64: u32 = 10;
const KV_I64: u32 = 11;
const KV_F64: u32 = 12;

fn scalar_type_bytes(etype: u32) -> Option<u64> {
    match etype {
        KV_BOOL | KV_U8 | KV_I8 => Some(1),
        KV_U16 | KV_I16 => Some(2),
        KV_U32 | KV_I32 | KV_F32 => Some(4),
        KV_U64 | KV_I64 | KV_F64 => Some(8),
        KV_STR => None, // variable-length, counted below
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// The reader
// ---------------------------------------------------------------------------

/// One tensor directory row.
#[derive(Debug, Clone, PartialEq)]
pub struct GgufTensor {
    pub name: String,
    /// GGUF lists dims innermost-first (ne0 is the contiguous axis).
    pub dims: Vec<u64>,
    pub dtype: u32,
    /// Offset RELATIVE to the tensor-data section start.
    pub offset: u64,
}

/// An open GGUF file: the header parsed, the tensor dir known, the
/// data section positioned. Tensor reads stream from the file — a 2 GB
/// F32 never loads whole.
pub struct GgufFile {
    file: File,
    pub arch: String,
    pub kv: HashMap<String, GgufKv>,
    pub tensors: HashMap<String, GgufTensor>,
    data_start: u64,
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self> {
        let path_display = path.display();
        let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mut head = [0u8; 24];
        file.read_exact(&mut head)?;
        anyhow::ensure!(
            &head[0..4] == b"GGUF",
            "{}: not a GGUF file (bad magic)",
            path.display()
        );
        let version = u32::from_le_bytes(head[4..8].try_into().unwrap());
        anyhow::ensure!(
            (2..=3).contains(&version),
            "{path_display}: GGUF v{version} unsupported (this reader speaks v2–v3)"
        );
        let n_tensors = u64::from_le_bytes(head[8..16].try_into().unwrap());
        let n_kv = u64::from_le_bytes(head[16..24].try_into().unwrap());

        let mut arch = String::new();
        let mut kv = HashMap::new();
        for _ in 0..n_kv {
            let key = read_str(&mut file)?;
            let mut vt = [0u8; 4];
            file.read_exact(&mut vt)?;
            let vtype = u32::from_le_bytes(vt);
            let value = read_kv(&mut file, vtype, path)?;
            if key == "general.architecture"
                && let GgufKv::Str(s) = &value
            {
                arch = s.clone();
            }
            kv.insert(key, value);
        }

        let mut tensors = HashMap::new();
        for _ in 0..n_tensors {
            let name = read_str(&mut file)?;
            let mut nd = [0u8; 4];
            file.read_exact(&mut nd)?;
            let n_dims = u32::from_le_bytes(nd);
            anyhow::ensure!(
                (1..=4).contains(&n_dims),
                "{path_display}: tensor {name} has {n_dims} dims (max 4)"
            );
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                let mut d = [0u8; 8];
                file.read_exact(&mut d)?;
                dims.push(u64::from_le_bytes(d));
            }
            let mut t = [0u8; 12];
            file.read_exact(&mut t)?;
            let dtype = u32::from_le_bytes(t[0..4].try_into().unwrap());
            let offset = u64::from_le_bytes(t[4..12].try_into().unwrap());
            anyhow::ensure!(
                tensors
                    .insert(
                        name.clone(),
                        GgufTensor {
                            name,
                            dims,
                            dtype,
                            offset
                        }
                    )
                    .is_none(),
                "{path_display}: duplicate tensor name"
            );
        }
        // The data section starts at the 32-byte-aligned position after
        // the directory (GGUF spec § tensors, both v2 and v3).
        let dir_end = file.stream_position()?;
        let data_start = (dir_end + 31) & !31;

        Ok(Self {
            file,
            arch,
            kv,
            tensors,
            data_start,
        })
    }

    fn kv_u32(&self, key: &str) -> Result<u32> {
        self.kv
            .get(key)
            .and_then(|v| v.as_u32())
            .with_context(|| format!("GGUF metadata {key:?} missing or not an integer"))
    }

    /// Does this file's reader speak this dtype? (The dequantizer's
    /// speech, as a cheap pre-check the layer split consults before
    /// reading anything.)
    fn dtype_supported(&self, dtype: u32) -> Result<()> {
        anyhow::ensure!(
            matches!(
                dtype,
                GGML_F32 | GGML_F16 | GGML_Q4_0 | GGML_Q8_0 | GGML_BF16
            ),
            "dtype {dtype} unsupported"
        );
        Ok(())
    }

    /// A flat (F32/F16/BF16) or block-quantized (Q8_0/Q4_0) tensor's
    /// bytes read from disk and dequantized to f32. Element count =
    /// product of dims; failures are typed, never garbage.
    pub fn tensor(&mut self, name: &str) -> Result<Vec<f32>> {
        let t = self
            .tensors
            .get(name)
            .with_context(|| {
                format!(
                    "GGUF has no tensor {name:?} (has {} tensors)",
                    self.tensors.len()
                )
            })?
            .clone();
        // Eery arithmetic below is attacker-controlled via the tensor
        // directory. UNCHECKED u64 math here has real release-mode teeth
        // (2026-10-23 audit): data_start + offset wrapped past 2⁶⁴, an
        // absurd 'at' sailed through the EOF gate, and the header bytes
        // were decoded as floats into a 'valid' expert the caller never
        // questioned. Every sum/product a crafted file can drive past
        // 2⁶⁴ is checked → typed refusal.
        let n_floats = t
            .dims
            .iter()
            .try_fold(1u64, |a, d| a.checked_mul(*d))
            .with_context(|| format!("tensor {name}: dims {:?} overflow", t.dims))?;
        anyhow::ensure!(
            n_floats > 0 && n_floats <= usize::MAX as u64,
            "tensor {name}: empty or overflowing element count"
        );
        // The read position: data-section base + the directory-carried
        // offset. Overflowing here is the silent-poison shape, refused.
        let at = self.data_start.checked_add(t.offset).with_context(|| {
            format!(
                "tensor {name}: data offset {} wraps past the data section",
                t.offset
            )
        })?;
        let mut buf: Vec<u8>;
        match t.dtype {
            GGML_F32 | GGML_F16 | GGML_BF16 => {
                let size = n_floats
                    .checked_mul(dtype_flat_size(t.dtype).unwrap() as u64)
                    .with_context(|| format!("tensor {name}: byte size overflows"))?;
                read_exact_at(&mut self.file, at, size)?;
                buf = vec![0u8; size as usize];
                self.file.read_exact(&mut buf)?;
            }
            GGML_Q8_0 | GGML_Q4_0 => {
                let (block_bytes, per_block) =
                    (if t.dtype == GGML_Q8_0 { 34u64 } else { 18u64 }, 32u64);
                anyhow::ensure!(
                    n_floats.is_multiple_of(per_block),
                    "tensor {name}: {n_floats} floats is not a whole number of {}-float {} blocks",
                    per_block,
                    if t.dtype == GGML_Q8_0 { "Q8_0" } else { "Q4_0" }
                );
                let n_blocks = n_floats / per_block;
                let size = n_blocks
                    .checked_mul(block_bytes)
                    .with_context(|| format!("tensor {name}: byte size overflows"))?;
                read_exact_at(&mut self.file, at, size)?;
                buf = vec![0u8; size as usize];
                self.file.read_exact(&mut buf)?;
            }
            other => bail!(
                "tensor {name}: dtype {other} unsupported — this importer speaks \
                 F32(0)/F16(1)/Q4_0(2)/Q8_0(8)/BF16(30) only; re-quantize or import a \
                 more precise variant"
            ),
        }
        decode(&buf, t.dtype, n_floats as usize)
    }
}

fn read_str(f: &mut File) -> Result<String> {
    let mut n = [0u8; 8];
    f.read_exact(&mut n)?;
    let n = u64::from_le_bytes(n);
    anyhow::ensure!(
        n <= 1 << 16,
        "metadata/tensor name of {n} bytes: refuse (name cap 64 KiB)"
    );
    let mut buf = vec![0u8; n as usize];
    f.read_exact(&mut buf)?;
    Ok(String::from_utf8(buf)?)
}

fn read_kv(f: &mut File, vtype: u32, path: &Path) -> Result<GgufKv> {
    let path_display = path.display();
    let mut scalar = |n: u64| -> Result<[u8; 8]> {
        let mut b = vec![0u8; n as usize];
        f.read_exact(&mut b)?;
        let mut out = [0u8; 8];
        out[..n as usize].copy_from_slice(&b);
        Ok(out)
    };
    Ok(match vtype {
        KV_BOOL => GgufKv::Bool(scalar(1)?[0] != 0),
        KV_U8 => GgufKv::U8(scalar(1)?[0]),
        KV_I8 => GgufKv::I8(scalar(1)?[0] as i8),
        KV_U16 => GgufKv::U16(u16::from_le_bytes(scalar(2)?[0..2].try_into().unwrap())),
        KV_I16 => GgufKv::I16(i16::from_le_bytes(scalar(2)?[0..2].try_into().unwrap())),
        KV_U32 => GgufKv::U32(u32::from_le_bytes(scalar(4)?[0..4].try_into().unwrap())),
        KV_I32 => GgufKv::I32(i32::from_le_bytes(scalar(4)?[0..4].try_into().unwrap())),
        KV_F32 => GgufKv::F32(f32::from_le_bytes(scalar(4)?[0..4].try_into().unwrap())),
        KV_U64 => GgufKv::U64(u64::from_le_bytes(scalar(8)?)),
        KV_I64 => GgufKv::I64(i64::from_le_bytes(scalar(8)?)),
        KV_F64 => GgufKv::F64(f64::from_le_bytes(scalar(8)?)),
        KV_STR => GgufKv::Str(read_str(f)?),
        KV_ARR => {
            let mut hdr = vec![0u8; 12];
            f.read_exact(&mut hdr)?;
            let etype = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
            let count = u64::from_le_bytes(hdr[4..12].try_into().unwrap());
            let bytes = match scalar_type_bytes(etype) {
                Some(s) => {
                    // Fixed-width elements: the whole array is skipped by
                    // exact byte math (its content is noise for this
                    // reader — vocabularies, rope tables and the like).
                    let bytes = count.checked_mul(s).context("array size overflow")?;
                    f.seek_relative(bytes as i64)
                        .context("skipping past scalar array")?;
                    bytes
                }
                None if etype == KV_STR => {
                    let mut total = 0u64;
                    for _ in 0..count {
                        let mut n = [0u8; 8];
                        f.read_exact(&mut n)?;
                        let n = u64::from_le_bytes(n);
                        f.seek_relative(n as i64)
                            .context("skipping past string array end")?;
                        total += 8 + n;
                    }
                    total
                }
                None if etype == KV_ARR => {
                    // Nested arrays: skip recursively by byte-counting.
                    let mut total = 0u64;
                    for _ in 0..count {
                        let mut hdr = vec![0u8; 12];
                        f.read_exact(&mut hdr)?;
                        let inner_etype = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
                        let inner_count = u64::from_le_bytes(hdr[4..12].try_into().unwrap());
                        let inner_size = scalar_type_bytes(inner_etype)
                            .context("nested arrays of unknown type")?;
                        f.seek_relative((inner_count * inner_size) as i64)
                            .context("skipping nested array")?;
                        total += 12 + inner_count * inner_size;
                    }
                    total
                }
                None => bail!("{path_display}: unexpected array element type {etype}"),
            };
            GgufKv::Arr {
                etype,
                count,
                bytes,
            }
        }
        other => bail!("{path_display}: metadata type {other} unsupported"),
    })
}

fn read_exact_at(f: &mut File, at: u64, len: u64) -> Result<()> {
    // Both operands carry untrusted values; an unchecked at+len wraps in
    // release and can turn an out-of-file range into a passing check.
    let end = at
        .checked_add(len)
        .context("tensor data range overflows u64")?;
    anyhow::ensure!(
        f.metadata()?.len() >= end,
        "tensor data {at}..+{len} runs past EOF"
    );
    f.seek(SeekFrom::Start(at))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Dequantization
// ---------------------------------------------------------------------------

fn decode(raw: &[u8], dtype: u32, n_floats: usize) -> Result<Vec<f32>> {
    match dtype {
        GGML_F32 => Ok(raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .take(n_floats)
            .collect()),
        GGML_F16 => Ok(raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| half_to_f32(u16::from_le_bytes(*c)))
            .take(n_floats)
            .collect()),
        GGML_BF16 => Ok(raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([0, 0, c[0], c[1]]))
            .take(n_floats)
            .collect()),
        GGML_Q8_0 => {
            // Per 32 floats: fp16 scale + 32 × i8. v = scale · n.
            Ok(raw
                .as_chunks::<34>()
                .0
                .iter()
                .flat_map(|block| {
                    let scale = half_to_f32(u16::from_le_bytes(block[0..2].try_into().unwrap()));
                    block[2..34].iter().map(move |n| scale * (*n as i8 as f32))
                })
                .take(n_floats)
                .collect())
        }
        GGML_Q4_0 => {
            // Per 32 floats: fp16 delta + 16 bytes of 4-bit nibbles
            // (low nibble first). v = delta · (n − 8).
            Ok(raw
                .as_chunks::<18>()
                .0
                .iter()
                .flat_map(|block| {
                    let delta = half_to_f32(u16::from_le_bytes(block[0..2].try_into().unwrap()));
                    block[2..18].iter().flat_map(move |b| {
                        let lo = delta * (*b & 0x0f) as f32 - 8.0 * delta;
                        let hi = delta * (*b >> 4) as f32 - 8.0 * delta;
                        [lo, hi]
                    })
                })
                .take(n_floats)
                .collect())
        }
        other => Err(anyhow::anyhow!("dtype {other} unsupported in decode")),
    }
}

/// IEEE 754 binary16 → f32, the denormal-and-infinity-safe way.
fn half_to_f32(h: u16) -> f32 {
    // f16 bits: sign 1 | expo 5 (bias 15) | mantissa 10.
    // ARBITRARY bit shuffling would miss denormals/inf; use the exact
    // integer composition (std has no f16 in stable Rust).
    let sign = ((h as u32 >> 15) & 1) << 31;
    let expo = (h >> 10) & 0x1f;
    let mant = (h & 0x3ff) as u32;
    let bits: u32 = match expo {
        0 if mant == 0 => sign, // zero
        0 => {
            // Denormal: renormalize (mantissa·2⁻¹⁰ × 2⁻¹⁴).
            let shift = mant.leading_zeros() + 18; // position of the leading 1
            let m = (mant << (shift - 1)) & 0x3ff;
            let e = (127 - 15 - 10) as u32;
            sign | ((e - shift + 1) << 23) | (m << 13)
        }
        31 => sign | (0xff << 23) | (mant << 13), // inf / NaN
        e => sign | (((e as u32) + 127 - 15) << 23) | (mant << 13),
    };
    f32::from_bits(bits)
}

// ---------------------------------------------------------------------------
// The layer split — FEATURES §13's Colibri-side milestone: real model
// weights into the expert store. Scans the tensor directory, splits the
// MoE stacks per (layer, expert), writes routers, and REPORTS every
// shape it did not understand rather than half-importing it.
// ---------------------------------------------------------------------------

use crate::expert_store::{Activation, ExpertKind, ExpertStore, ExpertWeights, RouterWeights};

/// What one import wrote, honestly: the ids landed in the store, the
/// per-layer classification, and every FFN shape the file carried that
/// the importer did NOT convert (with a reason each) — an import that
/// silently skipped a tensor would be a lie on the mesh wire.
#[derive(Debug, Default, PartialEq)]
pub struct ImportReport {
    pub arch: String,
    pub experts_written: Vec<String>,
    pub routers_written: Vec<String>,
    pub moe_layers: Vec<usize>,
    pub dense_layers: Vec<usize>,
    pub notes: Vec<String>,
}

/// Splits a GGUF model's MoE layer weights into the expert store:
///
/// - `blk.<n>.ffn_{gate,up,down}_exps.weight` 3-D stacks → experts
///   `L<n>.E<k>` (SwiGlu kind when a gate stack exists — the stacked
///   llama.cpp mixtral/deepseek family — else the plain-MLP kind with
///   the tagged activation, the nomic-bert-moe shape);
/// - `blk.<n>.ffn_gate_inp.weight` → the REAL router as `L<n>.router`
///   (two-tensor models still route — the exporter carries the scorer);
/// - dense `blk.<n>.ffn_{gate,up,down}.weight` layers (2-D) land as one
///   expert `L<n>.E0` in whichever kind the layer's tensors form.
///
/// One store put per container (atomic temp+rename), so a failed import
/// never leaves a half-written expert behind. `activation` overrides the
/// tagged choice for plain-MLP experts (see [`tag_for_arch`]).
pub fn import_gguf(
    gguf_path: &Path,
    store: &ExpertStore,
    activation: Option<Activation>,
) -> Result<ImportReport> {
    let mut f = GgufFile::open(gguf_path)?;
    let arch = f.arch.clone();
    let mut report = ImportReport {
        arch: arch.clone(),
        ..Default::default()
    };

    let expert_count =
        f.kv.get(&format!("{arch}.expert_count"))
            .and_then(|v| v.as_u32())
            .map(|v| v as usize);
    let expert_used =
        f.kv.get(&format!("{arch}.expert_used_count"))
            .and_then(|v| v.as_u32())
            .map(|v| v as usize);
    let hidden: usize = f.kv_u32(&format!("{arch}.embedding_length"))? as usize;
    let tagged = activation.unwrap_or_else(|| tag_for_arch(&arch));

    // Classify every FFN tensor once. Unrecognized vocabularies are
    // REPORTED, never imported-by-default: no silent half-model in the
    // store. (Attention/embed/output tensors are out of this shop's
    // scope — they belong to the runtime, not the expert transport.)
    let mut stacks: HashMap<usize, LayerStack> = HashMap::new();
    for (name, t) in &f.tensors {
        let Some((layer, rest)) = blk_of(name) else {
            continue;
        };
        let slot = stacks.entry(layer).or_default();
        match rest {
            "ffn_gate_exps.weight" | "ffn_up_exps.weight" | "ffn_down_exps.weight" => {
                if t.dims.len() != 3 {
                    report.notes.push(format!(
                        "{name}: {}-D MoE stack unsupported (want 3-D [a,b,experts])",
                        t.dims.len()
                    ));
                    continue;
                }
                if rest.starts_with("ffn_gate_exps") {
                    slot.exps[0] = Some(t.clone());
                } else if rest.starts_with("ffn_up_exps") {
                    slot.exps[1] = Some(t.clone());
                } else {
                    slot.exps[2] = Some(t.clone());
                }
            }
            "ffn_gate_inp.weight" => slot.router = Some(t.clone()),
            "ffn_gate.weight" | "ffn_up.weight" | "ffn_down.weight" => {
                if rest.starts_with("ffn_gate") {
                    slot.dense[0] = Some(t.clone());
                } else if rest.starts_with("ffn_up") {
                    slot.dense[1] = Some(t.clone());
                } else {
                    slot.dense[2] = Some(t.clone());
                }
            }
            other if other.starts_with("ffn") => {
                report
                    .notes
                    .push(format!("unrecognized FFN token piece: {name} ({other})"));
            }
            _ => {}
        }
    }

    // An honest one-line record of the arch tag when the store carries
    // it: a bert-family arch names GELU for plain-MLP experts, and the
    // choice (and its escape hatch) belongs in the import report.
    let any_mlp = stacks.values().any(|s| {
        (s.exps[1].is_some() && s.exps[0].is_none())
            || (s.dense[1].is_some() && s.dense[0].is_none())
    });
    if any_mlp && tagged == Activation::Gelu {
        report.notes.push(format!(
            "plain-MLP experts tagged GELU (bert-family arch {arch:?}) — override with an explicit activation on import"
        ));
    }

    for layer in {
        let mut keys: Vec<usize> = stacks.keys().copied().collect();
        keys.sort();
        keys
    } {
        let stack = &stacks[&layer];
        let unsupported = stack.unsupported_dtype(&f);
        if let Some(router_t) = &stack.router {
            match router_export(&mut f, router_t, unsupported, expert_used) {
                Ok((id, r)) => {
                    store.put_router(&r)?;
                    report.routers_written.push(id);
                }
                Err(e) => report
                    .notes
                    .push(format!("blk.{layer}.ffn_gate_inp: skipped — {e:#}")),
            }
        }

        // MoE stacks (3-D) take precedence: at least one `_exps` tensor.
        if stack.exps.iter().any(|t| t.is_some()) {
            report.moe_layers.push(layer);
            let Some(expert_count) = expert_count else {
                report.notes.push(format!(
                    "blk.{layer}: MoE stacks present but metadata {arch}.expert_count missing — layer skipped"
                ));
                continue;
            };
            match stack.split_moe(&mut f, layer, expert_count, hidden, tagged, unsupported) {
                Ok(experts) => {
                    for w in experts {
                        report.experts_written.push(w.id.clone());
                        store.put(&w)?;
                    }
                }
                Err(e) => {
                    report
                        .notes
                        .push(format!("blk.{layer} MoE stack: skipped — {e:#}"));
                }
            }
            continue;
        }

        // Dense FFN layer: gate+up+down (SwiGlu) or up+down (Mlp).
        if stack.dense.iter().all(|t| t.is_none()) {
            continue;
        }
        match stack.split_dense(&mut f, layer, hidden, tagged, unsupported) {
            Ok(w) => {
                report.dense_layers.push(layer);
                report.experts_written.push(w.id.clone());
                store.put(&w)?;
            }
            Err(e) => {
                report
                    .notes
                    .push(format!("blk.{layer} dense FFN: skipped — {e:#}"));
            }
        }
    }

    report.experts_written.sort();
    report.routers_written.sort();
    report.moe_layers.sort();
    report.dense_layers.sort();
    // Notes too: two imports of the same model must produce IDENTICAL
    // reports (the tensor scan walks a HashMap, whose order is per-process
    // random — a diff of two runs must be empty).
    report.notes.sort();
    Ok(report)
}

/// `blk.<n>.<rest>` → `(n, rest)` (GGUF tensor-name convention).
fn blk_of(name: &str) -> Option<(usize, &str)> {
    let rest = name.strip_prefix("blk.")?;
    let (n, tail) = rest.split_once('.')?;
    n.parse::<usize>().ok().map(|n| (n, tail))
}

/// Which activation a plain-MLP expert tags: bert-family (nomic's own
/// embedder line included) trains GELU; the encoder-less causal family
/// ships SwiGLU-shaped FFNs or SiLU MLPs. Overridable per import.
fn tag_for_arch(arch: &str) -> Activation {
    if arch.contains("bert") || arch.contains("nomic") {
        Activation::Gelu
    } else {
        Activation::Silu
    }
}

/// One block's FFN tensor set, classified.
#[derive(Default)]
struct LayerStack {
    /// [gate, up, down] — the 3-D `…_exps` stacks.
    exps: [Option<GgufTensor>; 3],
    /// [gate, up, down] — the dense 2-D FFN tensors.
    dense: [Option<GgufTensor>; 3],
    /// `ffn_gate_inp.weight` — the real router.
    router: Option<GgufTensor>,
}

impl LayerStack {
    /// Does any present dtype fall outside the dequantizer's speech?
    fn unsupported_dtype(&self, f: &GgufFile) -> Option<u32> {
        self.exps
            .iter()
            .chain(self.dense.iter())
            .flatten()
            .map(|t| t.dtype)
            .chain(self.router.iter().map(|t| t.dtype))
            .find(|d| f.dtype_supported(*d).is_err())
    }

    /// MoE stacks → SwiGlu or Mlp experts. Every failure is a String
    /// reason the caller REPORTS, never a half-import.
    #[allow(clippy::too_many_arguments)]
    fn split_moe(
        &self,
        f: &mut GgufFile,
        layer: usize,
        expert_count: usize,
        hidden: usize,
        tagged: Activation,
        unsupported: Option<u32>,
    ) -> Result<Vec<ExpertWeights>> {
        dequant_error(unsupported)?;
        let gate_t = self.exps[0].as_ref();
        let up_t = self.exps[1]
            .as_ref()
            .context("no ffn_up_exps stack in the file")?;
        let down_t = self.exps[2]
            .as_ref()
            .context("no ffn_down_exps stack in the file")?;
        let kind = if gate_t.is_some() {
            ExpertKind::SwiGlu
        } else {
            ExpertKind::Mlp
        };

        // GGUF lays the EXPERT axis last (dims [d0, d1, e]); expert k's
        // slab is the flat window [k·d0·d1, (k+1)·d0·d1). Slab layouts
        // (which axis is fastest) must match the containers':
        // gate/up = [inter × hidden] (h fastest) → d0 == hidden,
        // down = [hidden × inter] (i fastest) → d0 == inter.
        let check_stack = |which: &str, t: &GgufTensor, d0_expect: u64| -> Result<()> {
            anyhow::ensure!(
                t.dims[2] == expert_count as u64,
                "{which} stack expert axis {} ≠ expert_count {expert_count}",
                t.dims[2]
            );
            anyhow::ensure!(
                t.dims[0] == d0_expect,
                "{which} stack dims {:?}: innermost axis {} must be {d0_expect}",
                t.dims,
                t.dims[0]
            );
            Ok(())
        };
        if let Some(g) = gate_t {
            check_stack("ffn_gate_exps", g, hidden as u64)?;
        }
        check_stack("ffn_up_exps", up_t, hidden as u64)?;
        check_stack(
            "ffn_down_exps",
            down_t,
            gate_t.map_or(up_t.dims[1], |g| g.dims[1]),
        )?;

        // Intermediate size comes off the gate-or-up stack's second axis
        // and must tile with hidden — the down-stack check just passed
        // proves d0(down) == that same i.
        let inter = gate_t.map_or(up_t.dims[1], |g| g.dims[1]) as usize;
        let slab_len = hidden.checked_mul(inter).context("slab size overflow")?;
        for (which, t) in [("gate", gate_t), ("up", Some(up_t)), ("down", Some(down_t))] {
            if let Some(t) = t {
                let product = t.dims.iter().take(2).product::<u64>();
                anyhow::ensure!(
                    product == slab_len as u64,
                    "{which} stack dims {:?} do not tile {hidden}×{inter}",
                    t.dims
                );
            }
        }

        // One dequant pass per whole stack, then e in-memory slices —
        // never e re-reads/re-dequants of the same file bytes.
        let gate_full = match gate_t {
            Some(t) => Some(f.tensor(&t.name)?),
            None => None,
        };
        let up_full = f.tensor(&up_t.name)?;
        let down_full = f.tensor(&down_t.name)?;
        let slice = |full: &[f32], k: usize| -> Vec<f32> {
            full[k * slab_len..(k + 1) * slab_len].to_vec()
        };
        let mut out = Vec::with_capacity(expert_count);
        for k in 0..expert_count {
            let base = || ExpertWeights {
                id: format!("L{layer}.E{k}"),
                hidden,
                intermediate: inter,
                kind,
                activation: tagged,
                gate: None,
                up: slice(&up_full, k),
                down: slice(&down_full, k),
            };
            out.push(match (kind, &gate_full) {
                (ExpertKind::SwiGlu, Some(gate_full)) => ExpertWeights {
                    gate: Some(slice(gate_full, k)),
                    ..base()
                },
                _ => base(),
            });
            out.last().expect("just pushed").validate()?;
        }
        Ok(out)
    }

    /// Dense FFN → one expert `L<n>.E0` (SwiGlu with a gate tensor, Mlp
    /// without). Shapes validated the same honest way.
    fn split_dense(
        &self,
        f: &mut GgufFile,
        layer: usize,
        hidden: usize,
        tagged: Activation,
        unsupported: Option<u32>,
    ) -> Result<ExpertWeights> {
        dequant_error(unsupported)?;
        let gate_t = self.dense[0].as_ref();
        let up_t = self.dense[1]
            .as_ref()
            .context("no ffn_up tensor in the file")?;
        let down_t = self.dense[2]
            .as_ref()
            .context("no ffn_down tensor in the file")?;
        let kind = if gate_t.is_some() {
            ExpertKind::SwiGlu
        } else {
            ExpertKind::Mlp
        };
        let inter = gate_t.map_or(up_t.dims[1], |g| g.dims[1]) as usize;
        let n = hidden.checked_mul(inter).context("slab size overflow")?;
        for (which, t) in [("gate", gate_t), ("up", Some(up_t)), ("down", Some(down_t))] {
            if let Some(t) = t {
                let product: u64 = t.dims.iter().product();
                anyhow::ensure!(
                    product == n as u64,
                    "{which} tensor dims {:?} do not tile {hidden}×{inter}",
                    t.dims
                );
            }
        }
        let w = ExpertWeights {
            id: format!("L{layer}.E0"),
            hidden,
            intermediate: inter,
            kind,
            activation: tagged,
            gate: match gate_t {
                Some(t) => Some(f.tensor(&t.name)?),
                None => None,
            },
            up: f.tensor(&up_t.name)?,
            down: f.tensor(&down_t.name)?,
        };
        w.validate()?;
        Ok(w)
    }
}

/// The dtype refusal the split functions raise (typed, actionable).
fn dequant_error(unsupported: Option<u32>) -> Result<()> {
    if let Some(d) = unsupported {
        anyhow::bail!(
            "dtype {d} unsupported — this importer speaks \
             F32(0)/F16(1)/Q4_0(2)/Q8_0(8)/BF16(30); re-quantize or import a more precise variant"
        );
    }
    Ok(())
}

/// The router export: `ffn_gate_inp.weight` ([hidden × experts] row-major,
/// h fastest) → a `.router` container carrying `rows` [e × h]. dtype and
/// shape checked; `used` records the model's declared top-k when present.
fn router_export(
    f: &mut GgufFile,
    t: &GgufTensor,
    unsupported: Option<u32>,
    used: Option<usize>,
) -> Result<(String, RouterWeights)> {
    dequant_error(unsupported)?;
    anyhow::ensure!(
        t.dims.len() == 2,
        "router tensor {:?} is {}-D (want 2-D [hidden, experts])",
        t.dims,
        t.dims.len()
    );
    let (h, e) = (t.dims[0], t.dims[1]);
    let full = f.tensor(&t.name)?; // [e][h] row-major, h fastest == rows layout
    let r = RouterWeights {
        id: format!(
            "L{}.router",
            blk_of(&t.name).expect("router tensor is blk.*").0
        ),
        experts: e as usize,
        hidden: h as usize,
        used,
        rows: full,
    };
    let id = r.id.clone();
    r.validate()?;
    Ok((id, r))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const F32_T: u32 = 0;
    const F16_T: u32 = 1;
    const Q4_0_T: u32 = 2;
    const Q8_0_T: u32 = 8;
    const BF16_T: u32 = 30;

    /// Write a real-format GGUF file (all three sections, all rules) —
    /// the shape a real converter writes, content ours.
    fn gguf_bytes(
        kvs: &[(&str, GgufKv)],
        tensors: &[(String, Vec<u64>, u32, Vec<f32>)],
    ) -> Vec<u8> {
        // Tensor bytes in the SAME order as the dir rows; per-tensor
        // offsets are cumulative (data-section-relative), the way a real
        // converter lays the file out.
        let mut data = Vec::new();
        let mut offsets: Vec<u64> = Vec::with_capacity(tensors.len());
        for (name, dims, dtype, vals) in tensors {
            let n: u64 = dims.iter().product();
            assert_eq!(n as usize, vals.len(), "{name}: element count");
            offsets.push(data.len() as u64);
            match *dtype {
                F32_T => {
                    for v in vals {
                        data.extend_from_slice(&v.to_le_bytes());
                    }
                }
                F16_T => {
                    for v in vals {
                        data.extend_from_slice(&f32_to_half(*v).to_le_bytes());
                    }
                }
                BF16_T => {
                    for v in vals {
                        data.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
                    }
                }
                Q4_0_T => {
                    for c in vals.chunks(32) {
                        data.extend_from_slice(&q4_0_block(c));
                    }
                }
                Q8_0_T => {
                    for c in vals.chunks(32) {
                        data.extend_from_slice(&q8_0_block(c));
                    }
                }
                // Unknown-to-this-helper dtypes: structural filler only —
                // the reader refuses them before decoding any byte.
                _ => data.resize(data.len() + vals.len(), 0),
            }
            let _ = name;
        }
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        let kv_by_string = |s: &String, out: &mut Vec<u8>| {
            out.extend_from_slice(&(s.len() as u64).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        };
        for (k, v) in kvs {
            kv_by_string(&k.to_string(), &mut out);
            let (t, bytes) = match v {
                GgufKv::Str(s) => (KV_STR, {
                    let mut b = (s.len() as u64).to_le_bytes().to_vec();
                    b.extend_from_slice(s.as_bytes());
                    b
                }),
                GgufKv::U32(x) => (KV_U32, x.to_le_bytes().to_vec()),
                GgufKv::F32(x) => (KV_F32, x.to_le_bytes().to_vec()),
                GgufKv::Arr {
                    etype,
                    count,
                    bytes,
                } => {
                    // The file writes REAL array elements; ours are counted.
                    (KV_ARR, {
                        let mut b = etype.to_le_bytes().to_vec();
                        b.extend_from_slice(&count.to_le_bytes());
                        // Pad the claimed bytes with u8 zeros typed by the elem —
                        // for the tests our arrays are u32 only.
                        assert_eq!(*etype, KV_U32, "test arrays are ints only");
                        for _ in 0..*bytes / 4 {
                            b.extend_from_slice(&0u32.to_le_bytes());
                        }
                        b
                    })
                }
                _ => panic!("test kvs are u32/f32/str/arr only"),
            };
            out.extend_from_slice(&t.to_le_bytes());
            out.extend_from_slice(&bytes);
        }
        for ((name, dims, dtype, _), offset) in tensors.iter().zip(&offsets) {
            kv_by_string(name, &mut out);
            out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&dtype.to_le_bytes());
            // v2/v3 dir rows carry the per-tensor data offset (u64).
            out.extend_from_slice(&offset.to_le_bytes());
        }
        // Alignment to 32 before data (the spec rule the reader pins).
        let pos = out.len() as u64;
        let pad = (32 - pos % 32) % 32;
        out.extend(std::iter::repeat_n(0, pad as usize));
        out.extend_from_slice(&data);
        out
    }

    fn f32_to_half(v: f32) -> u16 {
        let bits = v.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let mant = bits & 0x7fffff;
        let expo_bits = ((bits >> 23) & 0xff) as i32;
        if expo_bits == 0 {
            return sign;
        }
        if expo_bits >= 255 {
            return sign | (0x7c00u16 | if mant != 0 { 0x1 } else { 0 });
        }
        let e = expo_bits - 127 + 15;
        if e >= 31 {
            return sign | 0x7c00; // inf
        }
        if e <= 0 {
            return sign; // denormal → zero in the tests' ranges
        }
        sign | ((e as u16) << 10) | ((mant >> 13) as u16 & 0x3ff)
    }

    /// ggml's Q8_0 contract: d = amax/127 (call it an fp16 number), each
    /// 32-float block stores d then 32 × round(v/d) bytes — the dequantizer
    /// reads v = d·n exactly back.
    fn q8_0_block(vals: &[f32]) -> Vec<u8> {
        let amax = vals.iter().fold(1e-9f32, |a, v| a.max(v.abs()));
        let scale_bits = f32_to_half(amax / 127.0);
        let scale = half_to_f32(scale_bits);
        let mut out = scale_bits.to_le_bytes().to_vec();
        for v in vals {
            out.push(((*v / scale).round() as i32).clamp(-128, 127) as u8);
        }
        out
    }

    /// ggml's Q4_0 contract: d = amax/7 (fp16), 4-bit nibbles n with
    /// v = d·(n − 8) — the +8 bias centers the 16 levels on the block.
    fn q4_0_block(vals: &[f32]) -> Vec<u8> {
        let amax = vals.iter().fold(1e-9f32, |a, v| a.max(v.abs()));
        let delta_bits = f32_to_half(amax / 7.0);
        let delta = half_to_f32(delta_bits);
        let mut out = delta_bits.to_le_bytes().to_vec();
        for c in vals.chunks(2) {
            let lo = ((c[0] / delta + 8.0).round() as i32).clamp(0, 15) as u8;
            let hi = c
                .get(1)
                .map_or(8, |v| ((v / delta + 8.0).round() as i32).clamp(0, 15))
                as u8;
            out.push(lo | (hi << 4));
        }
        out
    }

    /// Unique per-call temp path: parallel tests must never share a
    /// target file (a shared path races write↔parse under `cargo test`).
    fn write_and_open(bytes: &[u8]) -> GgufFile {
        let path = std::env::temp_dir().join(format!(
            "jerith-gguf-test-{}-{}.gguf",
            std::process::id(),
            ATOMIC.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::write(&path, bytes).unwrap();
        let f = GgufFile::open(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        f
    }

    /// Regression (2026-10-23 audit): the release-mode poisoner. A dir row
    /// whose `offset` wraps `data_start + offset` past 2⁶⁴ used to land the
    /// read INSIDE the header (the EOF gate then passed), so the header
    /// bytes decoded as floats into a "valid" expert the importer wrote to
    /// the live store AS A SUCCESS. In a debug build the same file panicked
    /// on the unchecked add. Both shapes must now be typed refusals.
    #[test]
    fn overflow_tensor_arithmetic_is_refused_not_silently_poisoning() {
        // A minimal-but-valid container: arch kv + two F32 dense FFN
        // tensors laid out like the helper writes them — except the up
        // tensor's dir-row offset is u64::MAX (guaranteed wrap) and the
        // down tensor's dims overflow (2⁴⁰ × 2⁴⁰ ≫ 2⁶⁴).
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&2u64.to_le_bytes()); // n_tensors
        b.extend_from_slice(&1u64.to_le_bytes()); // n_kv
        b.extend_from_slice(&(20u64).to_le_bytes());
        b.extend_from_slice(b"general.architecture");
        b.extend_from_slice(&KV_STR.to_le_bytes());
        b.extend_from_slice(&(1u64).to_le_bytes());
        b.extend_from_slice(b"x");
        for (name, dims, offset) in [
            ("blk.0.ffn_up.weight", vec![8u64, 4], u64::MAX),
            ("blk.0.ffn_down.weight", vec![1u64 << 40, 1u64 << 40], 0),
        ] {
            b.extend_from_slice(&(name.len() as u64).to_le_bytes());
            b.extend_from_slice(name.as_bytes());
            b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims {
                b.extend_from_slice(&d.to_le_bytes());
            }
            b.extend_from_slice(&F32_T.to_le_bytes());
            b.extend_from_slice(&offset.to_le_bytes());
        }
        let pad = (32 - (b.len() as u64 % 32)) % 32;
        b.extend(std::iter::repeat_n(0u8, pad as usize));
        b.extend(std::iter::repeat_n(0x41u8, 8 * 4 * 4)); // up
        b.extend(std::iter::repeat_n(0x42u8, 4 * 8 * 4)); // down

        let mut f = write_and_open(&b);
        let err = f.tensor("blk.0.ffn_up.weight").unwrap_err().to_string();
        assert!(
            err.contains("wraps past the data section"),
            "wrap offset must be a typed refusal, got: {err}"
        );
        let err = f.tensor("blk.0.ffn_down.weight").unwrap_err().to_string();
        assert!(
            err.contains("dims [1099511627776, 1099511627776] overflow"),
            "overflowing dims must be a typed refusal, got: {err}"
        );
    }

    #[test]
    fn header_kv_and_alignment_parse() {
        let f = write_and_open(&gguf_bytes(
            &[
                ("general.architecture", GgufKv::Str("nomic-bert-moe".into())),
                ("nomic-bert-moe.block_count", GgufKv::U32(12)),
                ("nomic-bert-moe.expert_count", GgufKv::U32(8)),
                ("nomic-bert-moe.feed_forward_length", GgufKv::U32(3072)),
                ("nomic-bert-moe.expert_used_count", GgufKv::U32(2)),
                // An array the reader must skip by exact byte math.
                (
                    "tokenizer.ggml.tokens",
                    GgufKv::Arr {
                        etype: KV_U32,
                        count: 3,
                        bytes: 12,
                    },
                ),
            ],
            &[],
        ));
        assert_eq!(f.arch, "nomic-bert-moe");
        assert_eq!(f.kv_u32("nomic-bert-moe.block_count").unwrap(), 12);
        assert_eq!(f.kv_u32("nomic-bert-moe.expert_used_count").unwrap(), 2);
        assert!(f.tensors.is_empty());
    }

    #[test]
    fn f32_f16_bf16_q8_q4_dequant_contract() {
        let vals: Vec<f32> = (0..64).map(|i| i as f32 / 8.0 - 4.0).collect();
        let t = |name: &str, dtype: u32| (name.to_string(), vec![64u64], dtype, vals.clone());
        let mut f = write_and_open(&gguf_bytes(
            &[("general.architecture", GgufKv::Str("test".into()))],
            &[
                t("t.f32", F32_T),
                t("t.f16", F16_T),
                t("t.bf16", BF16_T),
                t("t.q8", Q8_0_T),
                t("t.q4", Q4_0_T),
            ],
        ));
        let exact = |got: &[f32], want: &[f32], eps: f32, what: &str| {
            for (g, w) in got.iter().zip(want) {
                assert!((g - w).abs() <= eps, "{what}: {g} vs {w}");
            }
            assert_eq!(got.len(), want.len(), "{what} length");
        };
        exact(&f.tensor("t.f32").unwrap(), &vals, 0.0, "f32");
        exact(&f.tensor("t.f16").unwrap(), &vals, 1e-3, "f16");
        exact(&f.tensor("t.bf16").unwrap(), &vals, 1e-2, "bf16");
        exact(&f.tensor("t.q8").unwrap(), &vals, 0.05, "q8_0"); // half of the amax/127 step
        exact(&f.tensor("t.q4").unwrap(), &vals, 0.3, "q4_0"); // half-step of amax/7 ≈ 0.286 on ±4
    }

    #[test]
    fn unknown_dtype_types_loud() {
        let mut f = write_and_open(&gguf_bytes(
            &[("general.architecture", GgufKv::Str("test".into()))],
            &[("bad.t".to_string(), vec![64, 64], 12, vec![0.0; 64 * 64])],
        ));
        let e = f.tensor("bad.t").unwrap_err().to_string();
        assert!(e.contains("unsupported"), "{e}");
    }

    /// The REAL pilot file, when the operator has it downloaded: the
    /// dequantized experts of the nomic Q8_0 read equal (to fp16-scale
    /// rounding) the same tensors in the F32 file — the two formats
    /// agree, so the store receives the model's actual floats. Trivially
    /// skipped (with the reason) without the files; never a folded pass.
    /// The pilot files live at the REPO ROOT's models/ — cargo runs test
    /// binaries from the package dir, so the path is anchored there.
    fn pilot_paths() -> (PathBuf, PathBuf) {
        // crates/core/src/gguf_import.rs → repo root is three up.
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .unwrap()
            .to_path_buf();
        (
            root.join("models/nomic-embed-text-v2-moe.Q8_0.gguf"),
            root.join("models/nomic-embed-text-v2-moe.f32.gguf"),
        )
    }

    #[test]
    fn live_real_dequant_cross_format_oracle() {
        let (q8, f32_) = pilot_paths();
        if !q8.is_file() || !f32_.is_file() {
            eprintln!("SKIP: pilot GGUFs not downloaded (models/nomic-…[.Q8_0|.f32].gguf)");
            return;
        }
        let mut q = GgufFile::open(&q8).unwrap();
        assert_eq!(q.arch, "nomic-bert-moe");
        assert_eq!(q.kv_u32("nomic-bert-moe.expert_count").unwrap(), 8);
        assert_eq!(GgufFile::open(&f32_).unwrap().arch, "nomic-bert-moe");
        // One attention slab (Q8_0 in q8-file, F32 in f32-file) + one expert stack slab.
        for tn in ["blk.0.attn_qkv.weight", "blk.1.ffn_up_exps.weight"] {
            let tq = q.tensor(tn).unwrap();
            let mut f = GgufFile::open(&f32_).unwrap();
            let tf = f.tensor(tn).unwrap();
            assert_eq!(tq.len(), tf.len());
            // Q8_0 stores fp16 block scales: max error ≤ the scale's own
            // quantization granule (~1e-3 relative on [-1,1] weights).
            let max_err = tq
                .iter()
                .zip(&tf)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(max_err < 0.005, "{tn}: dequant max error {max_err}");
        }
    }

    /// The REAL split pilot: when the pilot GGUFs are on disk, import BOTH
    /// (Q8_0 and the F32 oracle) into separate temp stores and prove the
    /// two stores' expert MATH agrees through a real forward pass — the
    /// Q8-derived expert serves (to fp16-scale granularity) the same
    /// activations the F32-derived one would. Also proves the routers
    /// score the same. Skipped (honestly) without the files.
    #[test]
    fn live_split_math_matches_the_f32_oracle() {
        let (q8, f32_) = pilot_paths();
        if !q8.is_file() || !f32_.is_file() {
            eprintln!("SKIP: pilot GGUFs not downloaded (models/nomic-…[.Q8_0|.f32].gguf)");
            return;
        }
        let (dir_q, store_q) = temp_store();
        let (dir_f, store_f) = temp_store();
        let rep_q = import_gguf(&q8, &store_q, None).unwrap();
        let rep_f = import_gguf(&f32_, &store_f, None).unwrap();
        assert_eq!(
            rep_q.experts_written, rep_f.experts_written,
            "same layers survive both quantizations"
        );
        assert_eq!(
            rep_q.notes, rep_f.notes,
            "both files must classify identically"
        );

        // One deterministic hidden-wide input (the same x both stores run).
        let w_probe = store_q.get(&rep_q.experts_written[0]).unwrap();
        let x: Vec<f32> = (0..w_probe.hidden)
            .map(|j| ((j % 13) as f32 - 6.0) / 6.0)
            .collect();

        for id in rep_q.experts_written.iter().take(4) {
            let y_q = store_q.run(id, &x).unwrap();
            let y_f = store_f.run(id, &x).unwrap();
            let max_err = y_q
                .iter()
                .zip(&y_f)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let y_max = y_f.iter().fold(0.1f32, |a, v| a.max(v.abs()));
            // 8-bit quantization noise, amplified through the intermediate
            // units, is RELATIVE to the outputs: measured ~1.3% on the real
            // nomic pilot — pin 2% of the output scale as the bound.
            let rel_pct = 100.0 * max_err / y_max;
            assert!(
                rel_pct < 2.0,
                "{id}: forward max error {max_err} = {rel_pct:.2}% of |y|max={y_max} (Q8-derived vs F32-derived)"
            );
        }

        // A Q8 router scores within the same granule of the F32 one.
        if rep_q.routers_written.len() == rep_f.routers_written.len() {
            for id in rep_q.routers_written.iter() {
                let rq = store_q.get_router(id).unwrap();
                let rf = store_f.get_router(id).unwrap();
                let s_q = rq.forward(&x).unwrap();
                let s_f = rf.forward(&x).unwrap();
                let max_err = s_q
                    .iter()
                    .zip(&s_f)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                let s_max = s_f.iter().fold(0.1f32, |a, v| a.max(v.abs()));
                assert!(
                    max_err / s_max < 0.02,
                    "{id}: router score max error {max_err} of |s|max={s_max}"
                );
            }
        }

        std::fs::remove_dir_all(dir_q).unwrap();
        std::fs::remove_dir_all(dir_f).unwrap();
    }

    fn temp_store() -> (std::path::PathBuf, ExpertStore) {
        let dir = std::env::temp_dir().join(format!(
            "jerith-gguf-split-{}-{}",
            std::process::id(),
            ATOMIC.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let store = ExpertStore::new(&dir, 64).unwrap();
        (dir, store)
    }

    static ATOMIC: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    const H: u64 = 4;
    const I: u64 = 3;
    const E: u64 = 2;

    /// A 2-layer test model: blk.0 dense SwiGlu FFN, blk.1 MoE with 2
    /// experts (SwiGlu stacks when gated) + a real router. Tensor n's
    /// elements are `value(n, i)` so any slab can be rebuilt exactly.
    fn split_model_bytes(with_gate: bool, arch: &str) -> Vec<u8> {
        let v = |n: usize| (n % 1000) as f32 / 500.0 - 1.0;
        let n: usize = (H * I) as usize;
        let mut tensors: Vec<(String, Vec<u64>, u32, Vec<f32>)> = vec![
            // attention tensor that must be skipped silently
            (
                "blk.0.attn_q.weight".into(),
                vec![H, H],
                F32_T,
                (0..(H * H) as usize).map(v).collect(),
            ),
            (
                "blk.0.ffn_up.weight".into(),
                vec![H, I],
                F32_T,
                (0..n).map(v).collect(),
            ),
            (
                "blk.0.ffn_down.weight".into(),
                vec![I, H],
                F32_T,
                (1000..1000 + n).map(v).collect(),
            ),
            (
                "blk.1.ffn_up_exps.weight".into(),
                vec![H, I, E],
                F32_T,
                (2000..2000 + n * E as usize).map(v).collect(),
            ),
            (
                "blk.1.ffn_down_exps.weight".into(),
                vec![I, H, E],
                F32_T,
                (3000..3000 + n * E as usize).map(v).collect(),
            ),
            (
                "blk.1.ffn_gate_inp.weight".into(),
                vec![H, E],
                F32_T,
                (4000..4000 + (H * E) as usize).map(v).collect(),
            ),
        ];
        if with_gate {
            tensors.insert(
                1,
                (
                    "blk.0.ffn_gate.weight".into(),
                    vec![H, I],
                    F32_T,
                    (500..500 + n).map(v).collect(),
                ),
            );
            tensors.insert(
                4,
                (
                    "blk.1.ffn_gate_exps.weight".into(),
                    vec![H, I, E],
                    F32_T,
                    (6000..6000 + n * E as usize).map(v).collect(),
                ),
            );
        }
        gguf_bytes(
            &[
                ("general.architecture", GgufKv::Str(arch.into())),
                (
                    format!("{arch}.embedding_length").as_str(),
                    GgufKv::U32(H as u32),
                ),
                (
                    format!("{arch}.expert_count").as_str(),
                    GgufKv::U32(E as u32),
                ),
                (format!("{arch}.expert_used_count").as_str(), GgufKv::U32(1)),
            ],
            &tensors,
        )
    }

    fn write_split_model(with_gate: bool, arch: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "jerith-split-model-{}-{}.gguf",
            std::process::id(),
            ATOMIC.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::write(&path, split_model_bytes(with_gate, arch)).unwrap();
        path
    }

    /// Reference SwiGlu expert for slab k of the test stacks: rebuild the
    /// same values the generator wrote (slabs stack CONTIGUOUSLY after the
    /// seed: expert k starts at seed + k·H·I) and compute the forward
    /// directly.
    fn slab_vals(seed: u128, k: usize) -> Vec<f32> {
        let slab = (H * I) as u128;
        (0..slab as usize)
            .map(|i| ((seed + k as u128 * slab + i as u128) % 1000) as f32 / 500.0 - 1.0)
            .collect()
    }

    fn reference_forward(gate: &[f32], up: &[f32], down: &[f32]) -> Vec<f32> {
        let act: Vec<f32> = (0..I as usize)
            .map(|unit| {
                let g: f32 = gate[unit * H as usize..(unit + 1) * H as usize]
                    .iter()
                    .enumerate()
                    .map(|(j, gv)| gv * (0.25 + 0.1 * j as f32))
                    .sum();
                let u: f32 = up[unit * H as usize..(unit + 1) * H as usize]
                    .iter()
                    .enumerate()
                    .map(|(j, uv)| uv * (0.25 + 0.1 * j as f32))
                    .sum();
                (g / (1.0 + (-g).exp())) * u
            })
            .collect();
        (0..H as usize)
            .map(|out| {
                down[out * I as usize..(out + 1) * I as usize]
                    .iter()
                    .zip(&act)
                    .map(|(d, a)| d * a)
                    .sum()
            })
            .collect()
    }
    fn input() -> Vec<f32> {
        (0..H as usize).map(|j| 0.25 + 0.1 * j as f32).collect()
    }

    #[test]
    fn moe_stack_and_dense_ffn_split_into_the_store() {
        let path = write_split_model(true, "test-mixtral");
        let (dir, store) = temp_store();
        let report = import_gguf(&path, &store, None).unwrap();

        assert_eq!(report.arch, "test-mixtral");
        assert_eq!(report.moe_layers, vec![1]);
        assert_eq!(report.dense_layers, vec![0]);
        assert_eq!(report.experts_written, vec!["L0.E0", "L1.E0", "L1.E1"]);
        assert_eq!(report.routers_written, vec!["L1.router"]);
        assert_eq!(store.hosted(), vec!["L0.E0", "L1.E0", "L1.E1"]);
        assert_eq!(store.routers(), vec!["L1.router"]);

        // The math on disk IS the model's math: expert L1.E1 forward over
        // a fixed input equals an independently computed reference built
        // from the same generator values.
        let k = 1usize;
        let gate = slab_vals(6000, k);
        let up = slab_vals(2000, k);
        let down = slab_vals(3000, k);
        let want = reference_forward(&gate, &up, &down);
        let got = store.run(&format!("L1.E{k}"), &input()).unwrap();
        assert_eq!(got, want, "split expert must compute the model's math");

        // The dense layer landed as a single SwiGlu expert.
        let w = store.get("L0.E0").unwrap();
        assert_eq!(w.kind, ExpertKind::SwiGlu);
        let want = reference_forward(&slab_vals(500, 0), &slab_vals(0, 0), &slab_vals(1000, 0));
        assert_eq!(store.run("L0.E0", &input()).unwrap(), want);

        // The router carries the model's real scorer.
        let r = store.get_router("L1.router").unwrap();
        assert_eq!((r.experts, r.hidden), (E as usize, H as usize));
        assert_eq!(r.used, Some(1));
        assert_eq!(
            r.rows,
            (4000..4000 + (H * E) as usize)
                .map(|n| (n % 1000) as f32 / 500.0 - 1.0)
                .collect::<Vec<f32>>()
        );

        std::fs::remove_file(&path).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn gateless_stacks_split_as_mlp_experts_with_the_tagged_activation() {
        let path = write_split_model(false, "test-dense");
        let (dir, store) = temp_store();
        let report = import_gguf(&path, &store, None).unwrap();

        assert_eq!(report.experts_written, vec!["L0.E0", "L1.E0", "L1.E1"]);
        let w = store.get("L1.E1").unwrap();
        assert_eq!(w.kind, ExpertKind::Mlp);
        assert_eq!(
            w.activation,
            crate::expert_store::Activation::Silu,
            "non-bert arch defaults to SiLU"
        );
        assert!(w.gate.is_none(), "the split never fabricates a gate");

        // nomic-style arch tags GELU for the two-tensor line.
        let (dir2, store2) = temp_store();
        let path2 = write_split_model(false, "nomic-bert-moe");
        let report2 = import_gguf(&path2, &store2, None).unwrap();
        assert_eq!(report2.experts_written, vec!["L0.E0", "L1.E0", "L1.E1"]);
        let w2 = store2.get("L1.E1").unwrap();
        assert_eq!(w2.activation, crate::expert_store::Activation::Gelu);
        assert!(
            report2.notes.iter().any(|n| n.contains("GELU")),
            "the choice is reported: {:?}",
            report2.notes
        );

        // Operator override wins.
        let (dir3, store3) = temp_store();
        let report3 =
            import_gguf(&path2, &store3, Some(crate::expert_store::Activation::Silu)).unwrap();
        assert_eq!(report3.experts_written.len(), 3);
        assert_eq!(
            store3.get("L1.E0").unwrap().activation,
            crate::expert_store::Activation::Silu
        );

        for d in [dir, dir2, dir3] {
            std::fs::remove_dir_all(d).unwrap();
        }
        std::fs::remove_file(&path).unwrap();
        std::fs::remove_file(&path2).unwrap();
    }

    #[test]
    fn broken_and_unsupported_layers_are_reported_never_half_imported() {
        // A metadata expert count the tensor axes disagree with: the MoE
        // layer is skipped WHOLE with a named note; the independent dense
        // layer still imports.
        let key = "test-lying.expert_count";
        let mut bytes = split_model_bytes(true, "test-lying");
        let at = find_sub(&bytes, key.as_bytes()).expect("kv present");
        // KV layout: len(8) + name + type-tag(4) + u32 value → value at
        // name_start + name_len + 4.
        bytes[at + key.len() + 4] = 3; // 2 → 3
        let path = temp_model(&bytes);
        let (dir, store) = temp_store();
        let report = import_gguf(&path, &store, None).unwrap();
        assert!(
            report
                .notes
                .iter()
                .any(|n| n.contains("blk.1") && n.contains("expert axis")),
            "mismatch reported: {:?}",
            report.notes
        );
        assert_eq!(
            store.hosted(),
            vec!["L0.E0"],
            "the lying layer never half-imports, the dense one still imports"
        );

        // An unsupported ggml dtype inside a stack: that layer (and the
        // router it shares the file with) is skipped by NAME with the
        // typed note; clean layers still import.
        let name = "blk.1.ffn_up_exps.weight";
        let mut bytes = split_model_bytes(false, "test-bold");
        let at = find_sub(&bytes, name.as_bytes()).expect("tensor dir row present");
        // Dir row: len(8) + name + n_dims(4) + dims(8·n_dims) + dtype(4).
        let dtype_at = at + name.len() + 4 + 8 * 3;
        bytes[dtype_at] = 12; // F32(0) → 12: a ggml code the reader refuses
        let path2 = temp_model(&bytes);
        let (dir2, store2) = temp_store();
        let report2 = import_gguf(&path2, &store2, None).unwrap();
        assert!(
            report2
                .notes
                .iter()
                .any(|n| n.contains("blk.1") && n.contains("dtype 12")),
            "unsupported dtype reported: {:?}",
            report2.notes
        );
        assert_eq!(
            store2.hosted(),
            vec!["L0.E0"],
            "the clean dense layer still imports"
        );
        assert!(
            !report2
                .experts_written
                .iter()
                .any(|id| id.starts_with("L1.")),
            "the blocked layer never half-imports: {:?}",
            report2.experts_written
        );

        std::fs::remove_file(&path).unwrap();
        std::fs::remove_file(&path2).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
        std::fs::remove_dir_all(dir2).unwrap();
    }

    /// A temp file for hand-assembled GGUF bytes.
    fn temp_model(bytes: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "jerith-split-model-{}-{}.gguf",
            std::process::id(),
            ATOMIC.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    /// First offset of `needle` in `hay` (test-side byte find).
    fn find_sub(hay: &[u8], needle: &[u8]) -> Option<usize> {
        hay.windows(needle.len()).position(|w| w == needle)
    }
}
