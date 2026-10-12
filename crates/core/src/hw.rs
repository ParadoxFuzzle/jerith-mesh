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
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

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
