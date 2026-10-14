//! Curated model catalog + Colibri-style capability detection.
//!
//! For each known model we track the *resident* portion (always in RAM) plus
//! a minimum expert cache, mirroring how colibri recommends models: the
//! resident part + a small expert cache must fit in the machine's RAM for
//! the model to be runnable locally.
//!
//! These numbers are conservative placeholders for Phase 0 — they're meant
//! to be generated from actual benchmark data in later phases. When the
//! Colibri CLI is available, its `--list` output supersedes this table.

use serde::{Deserialize, Serialize};

use crate::hw::{GpuVendor, HardwareReport};

/// A model in the curated catalog or a user-provided model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ModelId {
    /// Canonical id used in the mesh (e.g. "glm-5.3", "qwen3.8-flash-next").
    pub id: String,
    /// Quantization tag; nodes only peer when this matches exactly.
    pub quant: String,
    /// Optional user path for GGUF/safetensors outside the catalog.
    pub source: ModelSource,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ModelSource {
    /// Curated Colibri-style engine.
    Curated,
    /// Explicit checkpoint path (GGUF / safetensors).
    Path { path: String },
    /// A model available through Ollama.
    Ollama { name: String },
    /// A model to be fetched from HuggingFace.
    HuggingFace { repo: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    pub id: String,
    pub display_name: String,
    /// Bytes of the model that must stay resident in RAM.
    pub resident_ram_bytes: u64,
    /// Minimum free RAM needed for the expert cache beyond residency.
    pub min_expert_cache_bytes: u64,
    /// Total disk footprint including all experts.
    pub disk_bytes: u64,
    /// Whether the model is a vision or image-generation engine.
    pub modality: ModelModality,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelModality {
    Text,
    Vision,
    ImageGen,
}

/// Runtime capability verdict for a model on a specific machine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFit {
    pub model: ModelSpec,
    pub fits: bool,
    pub reason: String,
}

/// The curated default model list from the spec.
pub fn curated() -> Vec<ModelSpec> {
    use ModelModality::{ImageGen, Text};
    // Numbers are placeholders to be replaced by benchmark-derived data.
    let models = [
        ("glm-5.2", "GLM-5.2", 10.5, 2.5, 45.0, Text),
        ("glm-5.3", "GLM-5.3", 10.5, 2.5, 45.0, Text),
        ("glm-5.3-flash", "GLM-5.3-Flash", 8.0, 2.0, 22.0, Text),
        (
            "deepseek-v4-flash",
            "DeepSeek-V4-Flash",
            7.5,
            2.0,
            22.0,
            Text,
        ),
        (
            "deepseek-v4.1-flash",
            "DeepSeek-V4.1-Flash",
            8.5,
            2.5,
            26.0,
            Text,
        ),
        ("kimi-k3", "Kimi K3", 12.0, 3.0, 55.0, Text),
        (
            "qwen3.8-flash-next",
            "Qwen3.8-Flash-Next",
            6.0,
            1.5,
            16.0,
            Text,
        ),
        ("qwen-image-2.1", "Qwen-Image-2.1", 5.0, 2.0, 25.0, ImageGen),
    ];
    models
        .into_iter()
        .map(
            |(id, name, res_gib, cache_gib, disk_gib, modality)| ModelSpec {
                id: id.to_string(),
                display_name: name.to_string(),
                resident_ram_bytes: (res_gib * 1024.0 * 1024.0 * 1024.0) as u64,
                min_expert_cache_bytes: (cache_gib * 1024.0 * 1024.0 * 1024.0) as u64,
                disk_bytes: (disk_gib * 1024.0 * 1024.0 * 1024.0) as u64,
                modality,
            },
        )
        .collect()
}

/// How much RAM this machine can dedicate to a resident model while still
/// leaving room for OS + the mesh overhead. Deliberately conservative.
pub fn usable_ram_bytes(hw: &HardwareReport) -> u64 {
    // Leave 1.5 GiB for the OS + daemon, or 10 % of RAM, whichever is larger.
    let reserve = (1_500 * 1024 * 1024).max(hw.ram_total_bytes / 10);
    hw.ram_available_bytes.saturating_sub(reserve)
}

/// Extra headroom a discrete GPU grants (usable VRAM directly speeds up
/// expert handling in colibri-style engines).
pub fn gpu_boost_bytes(hw: &HardwareReport) -> u64 {
    hw.gpus
        .iter()
        .filter(|g| g.vendor != GpuVendor::Other && g.vram_bytes > 0)
        .map(|g| g.vram_bytes / 2) // half of VRAM helps with expert cache
        .sum()
}

/// Determine which curated models fit on this machine.
pub fn detect_capable_models(hw: &HardwareReport) -> Vec<ModelFit> {
    let usable = usable_ram_bytes(hw);
    let boost = gpu_boost_bytes(hw);
    curated()
        .into_iter()
        .map(|m| fit_one(&m, usable, boost, hw))
        .collect()
}

fn fit_one(m: &ModelSpec, usable: u64, boost: u64, hw: &HardwareReport) -> ModelFit {
    let need = m.resident_ram_bytes + m.min_expert_cache_bytes;
    let effective = usable + boost;
    if need <= effective {
        ModelFit {
            model: m.clone(),
            fits: true,
            reason: format!(
                "resident {:.1} GiB + cache {:.1} GiB fits in effective {:.1} GiB",
                m.resident_ram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                m.min_expert_cache_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                effective as f64 / (1024.0 * 1024.0 * 1024.0)
            ),
        }
    } else {
        ModelFit {
            model: m.clone(),
            fits: false,
            reason: format!(
                "needs ~{:.1} GiB resident+cache but only {:.1} GiB effective available ({} total RAM, {} usable, {} GPU boost)",
                need as f64 / (1024.0 * 1024.0 * 1024.0),
                effective as f64 / (1024.0 * 1024.0 * 1024.0),
                format_bytes(hw.ram_total_bytes),
                format_bytes(usable),
                format_bytes(boost)
            ),
        }
    }
}

pub fn format_bytes(b: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    let b = b as f64;
    if b >= GIB {
        format!("{:.1} GiB", b / GIB)
    } else {
        format!("{:.0} MiB", b / MIB)
    }
}

/// Find the largest curated model this machine can run (used as the
/// "recommended" default during setup).
pub fn recommended_model(hw: &HardwareReport) -> Option<ModelSpec> {
    detect_capable_models(hw)
        .into_iter()
        .filter(|f| f.fits && f.model.modality == ModelModality::Text)
        .max_by_key(|f| f.model.resident_ram_bytes)
        .map(|f| f.model)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hw::GpuInfo;

    fn fake_hw(ram_total: u64, ram_avail: u64, gpu: Option<GpuInfo>) -> HardwareReport {
        let mut hw = HardwareReport {
            os: crate::hw::OsKind::Linux,
            os_release: "TestOS".into(),
            cpu_model: "Test CPU".into(),
            cpu_logical_cores: 8,
            cpu_physical_cores: 4,
            ram_total_bytes: ram_total,
            ram_available_bytes: ram_avail,
            swap_total_bytes: 0,
            gpus: gpu.map(|g| vec![g]).unwrap_or_default(),
            kernel: "test".into(),
            arch: "x86_64".into(),
            tool_output: None,
            tool_used: None,
        };
        // Force usable RAM calculation to be deterministic.
        hw.ram_available_bytes = ram_avail;
        hw
    }

    #[test]
    fn curated_contains_all_spec_models() {
        let ids: Vec<String> = curated().into_iter().map(|m| m.id).collect();
        for want in [
            "glm-5.2",
            "glm-5.3",
            "glm-5.3-flash",
            "deepseek-v4-flash",
            "deepseek-v4.1-flash",
            "kimi-k3",
            "qwen3.8-flash-next",
            "qwen-image-2.1",
        ] {
            assert!(
                ids.iter().any(|id| id == want),
                "missing curated model {want}"
            );
        }
    }

    #[test]
    fn big_machine_fits_most_models() {
        let hw = fake_hw(
            64 * 1024 * 1024 * 1024,
            60 * 1024 * 1024 * 1024,
            Some(GpuInfo {
                vendor: GpuVendor::Nvidia,
                name: "24GB".into(),
                vram_bytes: 24 * 1024 * 1024 * 1024,
            }),
        );
        let fits = detect_capable_models(&hw);
        let yes = fits.iter().filter(|f| f.fits).count();
        assert!(
            yes >= 6,
            "expected most models to fit on big machine, got {yes}"
        );
    }

    #[test]
    fn small_machine_fits_at_least_one_model() {
        // 16 GiB total with 12 GiB free: smallest curated model (~7.5 GiB
        // resident+cache) should fit, larger ones should not.
        let hw = fake_hw(16 * 1024 * 1024 * 1024, 12 * 1024 * 1024 * 1024, None);
        let fits = detect_capable_models(&hw);
        assert!(
            fits.iter().any(|f| f.fits),
            "smallest model should fit with 12 GiB free"
        );
        assert!(
            fits.iter().filter(|f| f.fits).count() <= 4,
            "only the smaller models should fit"
        );
    }

    #[test]
    fn tiny_machine_fits_at_least_something() {
        // 4 GiB free, no GPU — nothing in the catalog should fit but the
        // detector must not panic and must explain itself.
        let hw = fake_hw(4 * 1024 * 1024 * 1024, 3 * 1024 * 1024 * 1024, None);
        let fits = detect_capable_models(&hw);
        assert!(fits.iter().all(|f| !f.fits));
        assert!(fits.iter().all(|f| !f.reason.is_empty()));
    }

    #[test]
    fn gpu_boost_increases_effective_capacity() {
        let no_gpu = fake_hw(16 * 1024 * 1024 * 1024, 14 * 1024 * 1024 * 1024, None);
        let with_gpu = fake_hw(
            16 * 1024 * 1024 * 1024,
            14 * 1024 * 1024 * 1024,
            Some(GpuInfo {
                vendor: GpuVendor::Nvidia,
                name: "8GB".into(),
                vram_bytes: 8 * 1024 * 1024 * 1024,
            }),
        );
        assert!(
            usable_ram_bytes(&with_gpu) + gpu_boost_bytes(&with_gpu)
                > usable_ram_bytes(&no_gpu) + gpu_boost_bytes(&no_gpu)
        );
    }

    #[test]
    fn recommended_is_largest_fitting_text_model() {
        let hw = fake_hw(32 * 1024 * 1024 * 1024, 28 * 1024 * 1024 * 1024, None);
        let rec = recommended_model(&hw).expect("recommended should exist");
        assert_eq!(rec.modality, ModelModality::Text);
        let all_fits = detect_capable_models(&hw);
        let bigger_fitting = all_fits
            .iter()
            .filter(|f| f.fits && f.model.modality == ModelModality::Text)
            .any(|f| f.model.resident_ram_bytes > rec.resident_ram_bytes);
        assert!(
            !bigger_fitting,
            "recommended should be the largest fitting model"
        );
    }

    #[test]
    fn format_bytes_human_readable() {
        assert_eq!(format_bytes(0), "0 MiB");
        assert_eq!(format_bytes(512 * 1024 * 1024), "512 MiB");
        assert_eq!(format_bytes(2 * 1024 * 1024 * 1024), "2.0 GiB");
        assert_eq!(format_bytes(7_500_000_000), "7.0 GiB");
    }
}
