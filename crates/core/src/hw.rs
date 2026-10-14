//! Hardware probing with OS-specific inventory tools plus a Rust/sysinfo fallback.
//!
//! Per spec:
//! - Linux   → `inxi -Fxz`
//! - Windows → `systeminfo`
//! - macOS   → `system_profiler SPHardwareDataType`
//!
//! Each external tool is optional — if it is missing or fails, we still
//! produce a complete report from `sysinfo` so startup never blocks.

use std::process::Command;

use serde::{Deserialize, Serialize};
use sysinfo::System;

/// Raw hardware inventory for a node.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HardwareReport {
    pub os: OsKind,
    /// Human-readable OS description (e.g. "Ubuntu 24.04 LTS").
    pub os_release: String,
    pub cpu_model: String,
    pub cpu_logical_cores: u32,
    pub cpu_physical_cores: u32,
    /// Total RAM in bytes.
    pub ram_total_bytes: u64,
    /// Available RAM in bytes (snapshot).
    pub ram_available_bytes: u64,
    /// Total swap in bytes (0 when none).
    pub swap_total_bytes: u64,
    pub gpus: Vec<GpuInfo>,
    pub kernel: String,
    pub arch: String,
    /// Output of the OS-specific tool, when available.
    pub tool_output: Option<String>,
    /// Which tool produced `tool_output`.
    pub tool_used: Option<ToolKind>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OsKind {
    Linux,
    Windows,
    MacOs,
    Other,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolKind {
    Inxi,
    Systeminfo,
    SystemProfiler,
}

impl ToolKind {
    pub fn name(self) -> &'static str {
        match self {
            ToolKind::Inxi => "inxi",
            ToolKind::Systeminfo => "systeminfo",
            ToolKind::SystemProfiler => "system_profiler",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GpuInfo {
    pub vendor: GpuVendor,
    pub name: String,
    /// VRAM in bytes; 0 when unknown (e.g. integrated GPUs).
    pub vram_bytes: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Other,
}

/// Probe the local machine, preferring the OS-specific inventory tool.
pub fn probe() -> HardwareReport {
    let mut sys = System::new_all();
    sys.refresh_all();

    let os = detect_os();
    let (tool_used, tool_output) = run_os_tool(os);

    let gpus = probe_gpus();

    HardwareReport {
        os,
        os_release: os_release(&sys),
        cpu_model: cpu_model(&sys),
        cpu_logical_cores: sys.cpus().len() as u32,
        cpu_physical_cores: System::physical_core_count().unwrap_or(1) as u32,
        ram_total_bytes: sys.total_memory(),
        ram_available_bytes: sys.available_memory(),
        swap_total_bytes: sys.total_swap(),
        gpus,
        kernel: System::kernel_version().unwrap_or_else(|| "unknown".into()),
        arch: std::env::consts::ARCH.to_string(),
        tool_output,
        tool_used,
    }
}

fn detect_os() -> OsKind {
    match std::env::consts::OS {
        "linux" => OsKind::Linux,
        "windows" => OsKind::Windows,
        "macos" => OsKind::MacOs,
        _ => OsKind::Other,
    }
}

/// Run the OS's preferred inventory tool per spec; returns raw output.
fn run_os_tool(os: OsKind) -> (Option<ToolKind>, Option<String>) {
    let kind = match os {
        OsKind::Linux => Some(ToolKind::Inxi),
        OsKind::Windows => Some(ToolKind::Systeminfo),
        OsKind::MacOs => Some(ToolKind::SystemProfiler),
        OsKind::Other => None,
    };
    let Some(kind) = kind else {
        return (None, None);
    };
    let mut cmd = match os {
        OsKind::Linux => {
            let mut c = Command::new("inxi");
            c.args(["-Fxz"]);
            c
        }
        OsKind::Windows => Command::new("systeminfo"),
        OsKind::MacOs => {
            let mut c = Command::new("system_profiler");
            c.arg("SPHardwareDataType");
            c
        }
        OsKind::Other => return (None, None),
    };
    match cmd.output() {
        Ok(out) if out.status.success() => {
            let text = String::from_utf8_lossy(&out.stdout).to_string();
            (Some(kind), Some(text))
        }
        _ => (None, None),
    }
}

fn cpu_model(sys: &System) -> String {
    sys.cpus()
        .first()
        .map(|c| c.name().to_string())
        .unwrap_or_else(|| "Unknown CPU".into())
}

fn os_release(_sys: &System) -> String {
    // Linux: /etc/os-release PRETTY_NAME; macOS/Windows: sysinfo name.
    if let Ok(content) = std::fs::read_to_string("/etc/os-release") {
        for line in content.lines() {
            if let Some(v) = line.strip_prefix("PRETTY_NAME=") {
                return v.trim_matches('"').to_string();
            }
        }
    }
    System::name().unwrap_or_else(|| std::env::consts::OS.to_string())
}

/// Enumerate GPUs. Uses `nvidia-smi` on Linux/Windows for NVIDIA and falls
/// back to PCI IDs from sysfs on Linux; integrated Intel/AMD GPUs activate
/// the "Other" vendor path with unknown VRAM.
fn probe_gpus() -> Vec<GpuInfo> {
    let mut gpus = Vec::new();

    // NVIDIA via nvidia-smi (name + total VRAM in MiB).
    let mut smi = Command::new("nvidia-smi");
    smi.args([
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ]);
    if let Ok(out) = smi.output() {
        if out.status.success() {
            for line in String::from_utf8_lossy(&out.stdout).lines() {
                let fields: Vec<&str> = line.split(',').map(str::trim).collect();
                if fields.len() == 2 {
                    let vram_mib: u64 = fields[1].parse().unwrap_or(0);
                    gpus.push(GpuInfo {
                        vendor: GpuVendor::Nvidia,
                        name: fields[0].to_string(),
                        vram_bytes: vram_mib * 1024 * 1024,
                    });
                }
            }
        }
    }

    if gpus.is_empty() {
        // Linux sysfs PCI class probe: anything in VGA (0x03) / 3D (0x02) class.
        if let Ok(entries) = std::fs::read_dir("/sys/bus/pci/devices") {
            for entry in entries.flatten() {
                let path = entry.path();
                let Ok(class) = std::fs::read_to_string(path.join("class")) else {
                    continue;
                };
                let class = class.trim();
                if !(class.starts_with("0x03") || class.starts_with("0x02")) {
                    continue;
                }
                let vendor_id = std::fs::read_to_string(path.join("vendor"))
                    .map(|v| v.trim().to_string())
                    .unwrap_or_default();
                let dev_id = std::fs::read_to_string(path.join("device"))
                    .map(|v| v.trim().to_string())
                    .unwrap_or_default();
                let vendor = match vendor_id.as_str() {
                    "0x10de" => GpuVendor::Nvidia,
                    "0x1002" | "0x1022" => GpuVendor::Amd,
                    "0x8086" => GpuVendor::Intel,
                    "0x106b" => GpuVendor::Apple,
                    _ => GpuVendor::Other,
                };
                gpus.push(GpuInfo {
                    vendor,
                    name: format!("PCI {vendor_id}:{dev_id}"),
                    vram_bytes: 0,
                });
            }
        }
    }

    if gpus.is_empty() {
        // Apple Silicon: unified memory; note it as a single "Apple" GPU.
        if cfg!(target_os = "macos") {
            gpus.push(GpuInfo {
                vendor: GpuVendor::Apple,
                name: "Apple Silicon (unified memory)".into(),
                vram_bytes: 0,
            });
        }
    }

    gpus
}

/// VRAM in bytes summed over discrete GPUs (0 when none found).
pub fn total_discrete_vram(hw: &HardwareReport) -> u64 {
    hw.gpus.iter().map(|g| g.vram_bytes).sum()
}

// ============================================================================
// Live resource sampling (resource pooling, Phase 1)
// ============================================================================

/// Live VRAM picture of one GPU (from nvidia-smi). Every field optional:
/// a GPU that can't report absence of drivers/vendors gracefully degrades
/// to "unknown" instead of inventing zeros that would rank peers wrong.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct GpuSample {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vram_total_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vram_used_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vram_free_bytes: Option<u64>,
}

/// What this node can offer the mesh RIGHT NOW — the resource-pooling
/// counterpart of the setup-time `HardwareReport`, refreshed at ping
/// cadence and shipped inside the signed `PeerAd`. Placement (mesh-fallback
/// remote ordering) and the fleet census on /status both read this.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ResourceSample {
    pub ram_total_bytes: u64,
    /// Available RAM (snapshot) — the honest figure: caches can be
    /// reclaimed, but a node loading a model sees this shrink first.
    pub ram_available_bytes: u64,
    pub swap_total_bytes: u64,
    pub cpu_logical_cores: u32,
    pub gpus: Vec<GpuSample>,
    /// Unix ms when the snapshot was taken.
    pub sampled_at_ms: u64,
}

impl ResourceSample {
    /// Free VRAM summed over GPUs that report it. None when NO GPU reports
    /// (unknown ≠ zero: an unknown peer must not rank below a busy one).
    pub fn vram_free_total(&self) -> Option<u64> {
        let known: Vec<_> = self.gpus.iter().filter_map(|g| g.vram_free_bytes).collect();
        (!known.is_empty()).then(|| known.into_iter().sum())
    }

    /// Total VRAM summed over GPUs that report it (None = unknown).
    pub fn vram_total_total(&self) -> Option<u64> {
        let known: Vec<_> = self
            .gpus
            .iter()
            .filter_map(|g| g.vram_total_bytes)
            .collect();
        (!known.is_empty()).then(|| known.into_iter().sum())
    }

    /// One-line census: "16.4/31.8 GiB RAM · RTX 4060 6.7/7.6 GiB VRAM free".
    pub fn census(&self) -> String {
        let mut s = format!(
            "{:.1}/{:.1} GiB RAM",
            self.ram_available_bytes as f64 / GIB as f64,
            self.ram_total_bytes as f64 / GIB as f64
        );
        if let (Some(free), Some(total)) = (self.vram_free_total(), self.vram_total_total()) {
            s.push_str(&format!(
                " · {:.1}/{:.1} GiB VRAM free",
                free as f64 / GIB as f64,
                total as f64 / GIB as f64
            ));
        }
        s
    }
}

const GIB: u64 = 1024 * 1024 * 1024;

/// Parse `nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free
/// --format=csv,noheader,nounits` rows (name[, values in MiB]). Malformed
/// or empty output yields an empty vec — never an invented GPU.
fn parse_nvidia_smi_query(out: &str) -> Vec<GpuSample> {
    out.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|line| {
            let mut parts = line.split(',').map(str::trim);
            let name = parts.next()?.to_string();
            if name.is_empty() {
                return None;
            }
            let mut vram = [None; 3];
            for slot in vram.iter_mut() {
                *slot = parts
                    .next()
                    .and_then(|v| v.parse::<f64>().ok())
                    .map(|mib| (mib * 1024.0 * 1024.0) as u64);
            }
            // A garbage line (no reads at all) is not a GPU — never invent
            // one just because some config comments found their way in.
            if vram.iter().all(|v| v.is_none()) {
                return None;
            }
            // nounits MiB → bytes; [total, used, free]
            Some(GpuSample {
                name,
                vram_total_bytes: vram[0],
                vram_used_bytes: vram[1],
                vram_free_bytes: vram[2],
            })
        })
        .collect()
}

/// Shared sysinfo System: `System::new_all()` walks every process, which
/// is silly at ping cadence — one instance, memory/CPU refresh only.
fn shared_system() -> &'static std::sync::Mutex<System> {
    use std::sync::OnceLock;
    static SYS: OnceLock<std::sync::Mutex<System>> = OnceLock::new();
    SYS.get_or_init(|| std::sync::Mutex::new(System::new()))
}

/// Sample live resources. Cheap at ping cadence (§ placement refresh);
/// call from `spawn_blocking` when an async loop drives it.
/// GPU VRAM comes from nvidia-smi when present (any vendor without a
/// CLI there simply reports unknown); RAM/CPU always present.
pub fn sample() -> ResourceSample {
    let now = crate::bootstrap::now_ms();
    let (ram_total, ram_available, swap_total, cores) = {
        let mut sys = shared_system().lock().expect("sysinfo lock");
        sys.refresh_memory();
        sys.refresh_cpu_all();
        (
            sys.total_memory(),
            sys.available_memory(),
            sys.total_swap(),
            sys.cpus().len() as u32,
        )
    };
    let gpus = std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=name,memory.total,memory.used,memory.free")
        .arg("--format=csv,noheader,nounits")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .map(|out| parse_nvidia_smi_query(&out))
        .unwrap_or_default();
    ResourceSample {
        ram_total_bytes: ram_total,
        ram_available_bytes: ram_available,
        swap_total_bytes: swap_total,
        cpu_logical_cores: cores,
        gpus,
        sampled_at_ms: now,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_returns_plausible_live_values() {
        let s = sample();
        assert!(s.sampled_at_ms > 0, "timestamp set");
        assert!(s.ram_total_bytes >= 64 * 1024 * 1024, "live RAM seen");
        assert!(s.ram_available_bytes <= s.ram_total_bytes);
        // Census is human-readable either way (GPU or GPU-less host).
        assert!(s.census().contains("GiB RAM"));
    }

    #[test]
    fn nvidia_smi_query_rows_parse_to_gpu_samples() {
        let out = "NVIDIA GeForce RTX 4060, 8188, 4096, 4092\n";
        let gpus = parse_nvidia_smi_query(out);
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].name, "NVIDIA GeForce RTX 4060");
        assert_eq!(gpus[0].vram_total_bytes, Some(8188 * 1024 * 1024));
        assert_eq!(gpus[0].vram_used_bytes, Some(4096 * 1024 * 1024));
        assert_eq!(gpus[0].vram_free_bytes, Some(4092 * 1024 * 1024));
    }

    #[test]
    fn nvidia_smi_query_absent_or_garbage_is_not_a_gpu() {
        // driver missing → empty; garbage rows → ignored, never invented
        assert!(parse_nvidia_smi_query("").is_empty());
        assert!(parse_nvidia_smi_query("no drivers\n").is_empty());
        // two GPUs parse side by side (multi-GPU hosts)
        let two = parse_nvidia_smi_query("A, 100, 10, 90\nB, 200, 50, 150\n");
        assert_eq!(two.len(), 2);
        assert_eq!(two[1].vram_free_bytes, Some(150 * 1024 * 1024));
    }

    #[test]
    fn vram_totals_are_unknown_when_no_gpu_reports() {
        // unknown ≠ zero: an unreported host must not rank below a busy one
        let none = ResourceSample::default();
        assert_eq!(none.vram_free_total(), None);
        assert_eq!(none.vram_total_total(), None);
        let egpu = ResourceSample {
            gpus: vec![GpuSample {
                name: "RTX".into(),
                vram_total_bytes: Some(8 * GIB),
                vram_used_bytes: Some(6 * GIB),
                vram_free_bytes: Some(2 * GIB),
            }],
            ..Default::default()
        };
        assert_eq!(egpu.vram_free_total(), Some(2 * GIB));
        assert_eq!(egpu.vram_total_total(), Some(8 * GIB));
        assert!(egpu.census().contains("2.0/8.0 GiB VRAM free"));
    }

    #[test]
    fn probe_returns_a_complete_report() {
        let hw = probe();
        assert!(hw.cpu_logical_cores >= 1);
        assert!(hw.ram_total_bytes >= 64 * 1024 * 1024, "at least 64 MB RAM");
        assert!(hw.ram_available_bytes <= hw.ram_total_bytes);
        assert!(!hw.arch.is_empty());
        assert!(matches!(
            hw.os,
            OsKind::Linux | OsKind::Windows | OsKind::MacOs | OsKind::Other
        ));
    }

    #[test]
    fn probe_os_release_is_not_empty() {
        let hw = probe();
        assert!(!hw.os_release.is_empty());
    }

    #[test]
    fn serializes_round_trip() {
        let hw = probe();
        let json = serde_json::to_string(&hw).unwrap();
        let back: HardwareReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back, hw);
    }

    #[test]
    fn tool_used_only_when_output_present() {
        let hw = probe();
        if hw.tool_used.is_some() {
            assert!(hw.tool_output.is_some());
        }
        // Missing tool output is fine — must not block startup.
    }

    #[test]
    fn total_discrete_vram_sums() {
        let hw = HardwareReport {
            os: OsKind::Linux,
            os_release: "test".into(),
            cpu_model: "test".into(),
            cpu_logical_cores: 8,
            cpu_physical_cores: 4,
            ram_total_bytes: 16_000_000_000,
            ram_available_bytes: 8_000_000_000,
            swap_total_bytes: 0,
            gpus: vec![
                GpuInfo {
                    vendor: GpuVendor::Nvidia,
                    name: "RTX 4090".into(),
                    vram_bytes: 24 * 1024 * 1024 * 1024,
                },
                GpuInfo {
                    vendor: GpuVendor::Other,
                    name: "integrated".into(),
                    vram_bytes: 0,
                },
            ],
            kernel: "test".into(),
            arch: "x86_64".into(),
            tool_output: None,
            tool_used: None,
        };
        assert_eq!(total_discrete_vram(&hw), 24 * 1024 * 1024 * 1024);
    }
}
