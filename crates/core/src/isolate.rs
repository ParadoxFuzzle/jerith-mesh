//! Resource isolation + emergency kill switch (Phase 1).
//!
//! The `[resources]` config expresses what this node is willing to give
//! the mesh — hard-bound. [`ResourceIsolationConfig`] carries the knobs;
//! this module enforces them:
//!
//! - **Ceilings**: a background governor samples this node's own RSS and
//!   the GPUs' VRAM usage every `interval`. Cross a ceiling's *soft* line
//!   (90 %) → shed the expert hot-set cache (the one big discretionary
//!   allocation; everything evicted is re-read on demand). Cross the
//!   *hard* line (100 %) with `kill_switch` on → **trip**: expert
//!   serving/dispatch, mesh chat fallback and replication are refused
//!   until usage recedes under the soft line again (hysteresis, so the
//!   switch does not flap at the boundary). Recovery is automatic.
//! - **Kill switch**: local chat, the gateway and the WebUI stay up — the
//!   operator's own access surface is never killed; only mesh *work* is.
//! - **CPU pinning**: `pin_cores > 0` restricts the process to the first
//!   N logical cores (Linux, best-effort `sched_setaffinity`).
//!
//! Deliberately honest about what it cannot see: without nvidia-smi the
//! VRAM branch is skipped (unknown ≠ zero), and a governor with no
//! ceilings configured is a no-op — the legacy behavior.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use serde_json::{Value, json};

use crate::config::ResourceIsolationConfig;
use crate::hw::ResourceSample;

/// One usage sample the governor judges. Built from a sysinfo process
/// probe (RAM) plus the last ping-cycle [`ResourceSample`] (VRAM).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Usage {
    /// This process' resident set size — the honest "our footprint".
    pub rss_bytes: u64,
    /// VRAM used summed over GPUs that report it (None = unknown: without
    /// nvidia-smi the VRAM branch is skipped entirely).
    pub vram_used_bytes: Option<u64>,
    /// Mesh-relayed bytes per rolling minute (None = window too young to
    /// measure: unknown ≠ zero, same rule as VRAM).
    pub bandwidth_bytes_per_min: Option<u64>,
}

/// The governor's verdict for one sample. Pure, hysteresis-aware: `tripped`
/// in, a named transition out — the caller diffs against live state.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// All clear.
    Nominal,
    /// A ceiling's soft line was crossed (or a hard line with the switch
    /// off): shed the expert hot-set cache and keep mesh work on.
    Shed(String),
    /// Hard line crossed with the switch on: trip mesh work off.
    Trip(String),
    /// Still tripped: usage sits between the soft and hard lines.
    RemainTripped(String),
    /// Usage receded under every soft line: disengage the switch.
    Recover,
}

fn soft_line(ceiling: u64) -> u64 {
    ceiling * ResourceIsolationConfig::SOFT_SHARE_PCT / 100
}

/// Judge one usage sample against the ceilings. This is the whole policy —
/// everything else (sampling, state, effectors) is plumbing around it.
pub fn decide(usage: &Usage, cfg: &ResourceIsolationConfig, tripped: bool) -> Verdict {
    if tripped {
        // Recovery only under BOTH soft lines — hysteresis, so a sample
        // hovering at the boundary does not flap the switch.
        let ram_ok = cfg
            .ram_ceiling_bytes
            .is_none_or(|c| usage.rss_bytes < soft_line(c));
        let vram_ok = match (cfg.vram_ceiling_bytes, usage.vram_used_bytes) {
            (Some(c), Some(used)) => used < soft_line(c),
            // No ceiling or no readings → that branch cannot hold a trip.
            _ => true,
        };
        let bandwidth_ok = match (
            cfg.bandwidth_ceiling_bytes_per_min,
            usage.bandwidth_bytes_per_min,
        ) {
            (Some(c), Some(rate)) => rate < soft_line(c),
            // No ceiling or a too-young measurement window → cannot hold a trip.
            _ => true,
        };
        let recover_reason = cfg
            .ram_ceiling_bytes
            .map(|c| {
                format!(
                    "RAM {:.1}/{:.1} GiB under the soft line",
                    usage.rss_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                    c as f64 / (1024.0 * 1024.0 * 1024.0)
                )
            })
            .unwrap_or_default();
        if ram_ok && vram_ok && bandwidth_ok {
            return Verdict::Recover;
        }
        return Verdict::RemainTripped(recover_reason);
    }

    // Hard breaches first (highest priority action), then soft.
    let mut hard: Option<String> = None;
    let mut soft: Option<String> = None;
    if let Some(ram) = cfg.ram_ceiling_bytes {
        if usage.rss_bytes >= ram {
            hard = Some(format!(
                "node RSS {:.1} GiB reached the RAM ceiling {:.1} GiB",
                usage.rss_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                ram as f64 / (1024.0 * 1024.0 * 1024.0)
            ));
        } else if usage.rss_bytes >= soft_line(ram) {
            soft = Some(format!(
                "node RSS {:.1}/{:.1} GiB crossed the soft line",
                usage.rss_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                ram as f64 / (1024.0 * 1024.0 * 1024.0)
            ));
        }
    }
    if let (Some(vram), Some(used)) = (cfg.vram_ceiling_bytes, usage.vram_used_bytes) {
        if used >= vram {
            hard = Some(format!(
                "GPU VRAM used {:.1}/{:.1} GiB reached the ceiling",
                used as f64 / (1024.0 * 1024.0 * 1024.0),
                vram as f64 / (1024.0 * 1024.0 * 1024.0)
            ));
        } else if soft.is_none() && used >= soft_line(vram) {
            soft = Some(format!(
                "GPU VRAM used {:.1}/{:.1} GiB crossed the soft line",
                used as f64 / (1024.0 * 1024.0 * 1024.0),
                vram as f64 / (1024.0 * 1024.0 * 1024.0)
            ));
        }
    }
    if let (Some(bw_ceiling), Some(rate)) = (
        cfg.bandwidth_ceiling_bytes_per_min,
        usage.bandwidth_bytes_per_min,
    ) {
        if rate >= bw_ceiling {
            hard = Some(format!(
                "relayed bandwidth {:.1} MiB/min reached the ceiling {:.1} MiB/min",
                rate as f64 / (1024.0 * 1024.0),
                bw_ceiling as f64 / (1024.0 * 1024.0)
            ));
        } else if soft.is_none() && rate >= soft_line(bw_ceiling) {
            soft = Some(format!(
                "relayed bandwidth {:.1}/{:.1} MiB/min crossed the soft line",
                rate as f64 / (1024.0 * 1024.0),
                bw_ceiling as f64 / (1024.0 * 1024.0)
            ));
        }
    }
    if let Some(reason) = hard {
        return if cfg.kill_switch {
            Verdict::Trip(reason)
        } else {
            // The switch is off: shed what we can, never take work away.
            Verdict::Shed(reason)
        };
    }
    match soft {
        Some(reason) => Verdict::Shed(reason),
        None => Verdict::Nominal,
    }
}

/// Kill-switch runtime state: one per node, shared by everything that
/// might (a) refuse work or (b) report the switch on `/status`.
#[derive(Debug, Default)]
pub struct IsolateState {
    tripped: AtomicBool,
    /// Why the switch tripped (cleared on recovery).
    reason: std::sync::Mutex<Option<String>>,
    trips: AtomicU64,
    recovers: AtomicU64,
    /// How many soft-line cache sheds the governor performed.
    cache_sheds: AtomicU64,
    /// Unix ms of the last trip (0 = never).
    tripped_since_ms: AtomicU64,
    /// Latest measured relayed bandwidth (bytes/min). 0 = window too young
    /// to measure (unknown ≠ zero, same rule as VRAM).
    bandwidth_rate_min: AtomicU64,
}

impl IsolateState {
    /// Is mesh work currently refused?
    pub fn tripped(&self) -> bool {
        self.tripped.load(Ordering::Relaxed)
    }

    /// The reason recorded at trip time (None when not tripped).
    pub fn tripped_reason(&self) -> Option<String> {
        self.reason
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Trip the switch. Idempotent: the first trip wins (its reason is the
    /// root cause). Returns true on the transition, false if already tripped.
    pub fn record_trip(&self, reason: &str) -> bool {
        let first = !self.tripped.swap(true, Ordering::SeqCst);
        if first {
            self.trips.fetch_add(1, Ordering::Relaxed);
            self.tripped_since_ms
                .store(crate::bootstrap::now_ms(), Ordering::Relaxed);
            *self.reason.lock().unwrap_or_else(|e| e.into_inner()) = Some(reason.to_string());
        }
        first
    }

    /// Disengage the switch. Returns true on the transition. The recorded
    /// reason is cleared — the switch reports tripped-or-not, live.
    pub fn record_recover(&self) -> bool {
        let was = self.tripped.swap(false, Ordering::SeqCst);
        if was {
            self.recovers.fetch_add(1, Ordering::Relaxed);
            self.tripped_since_ms.store(0, Ordering::Relaxed);
            *self.reason.lock().unwrap_or_else(|e| e.into_inner()) = None;
        }
        was
    }

    /// Count one soft-line cache shed (governor bookkeeping).
    pub fn count_shed(&self) {
        self.cache_sheds.fetch_add(1, Ordering::Relaxed);
    }

    /// Publication point for the governor's latest bandwidth measurement
    /// (bytes/min; 0 = window too young). `/status` reads this live.
    pub fn record_bandwidth_rate(&self, bytes_per_min: Option<u64>) {
        self.bandwidth_rate_min
            .store(bytes_per_min.unwrap_or(0), Ordering::Relaxed);
    }

    /// The latest measured relayed bandwidth (None = unmeasured yet).
    pub fn bandwidth_rate_min(&self) -> Option<u64> {
        match self.bandwidth_rate_min.load(Ordering::Relaxed) {
            0 => None,
            r => Some(r),
        }
    }
}

/// Status view for `/status` (governor counters + live switch state).
pub fn snapshot(iso: &IsolateState) -> Value {
    json!({
        "tripped": iso.tripped(),
        "reason": iso.tripped_reason(),
        "trips": iso.trips.load(Ordering::Relaxed),
        "recovers": iso.recovers.load(Ordering::Relaxed),
        "cache_sheds": iso.cache_sheds.load(Ordering::Relaxed),
        "tripped_since_ms": iso.tripped_since_ms.load(Ordering::Relaxed),
        "bandwidth_rate_min": iso.bandwidth_rate_min.load(Ordering::Relaxed),
    })
}

/// This process' RSS via the shared sysinfo instance (refresh only this
/// pid — a full process walk at cadence is what `hw::sample` avoids).
pub fn self_rss_bytes() -> u64 {
    use sysinfo::{Pid, ProcessesToUpdate, System};
    let mut sys = System::new();
    let pid = Pid::from_u32(std::process::id());
    sys.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
    sys.process(pid).map(|p| p.memory()).unwrap_or_default()
}

/// Best-effort CPU pinning: restrict this process to the first
/// `pin_cores` logical cores. Linux-only via `sched_setaffinity`; other
/// OSes log and continue (pinning is an optimization, never a gate).
/// Returns whether an affinity mask was actually applied.
pub fn apply_pinning(pin_cores: u32) -> bool {
    if pin_cores == 0 {
        return false;
    }
    #[cfg(target_os = "linux")]
    {
        // cpu_set_t holds 1024 bits; config validation caps pin_cores there.
        let n = pin_cores.min(1024);
        let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        unsafe {
            libc::CPU_ZERO(&mut set);
            for i in 0..n {
                libc::CPU_SET(i as usize, &mut set);
            }
            if libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) != 0 {
                tracing::warn!(
                    error = %std::io::Error::last_os_error(),
                    "[resources] sched_setaffinity failed — running unpinned"
                );
                return false;
            }
        }
        tracing::info!("[resources] pinned process to the first {n} logical core(s)");
        true
    }
    #[cfg(not(target_os = "linux"))]
    {
        tracing::warn!(
            "[resources] pin_cores = {} unsupported on {} — ignoring (Linux-only)",
            pin_cores,
            std::env::consts::OS
        );
        false
    }
}

/// Extract the governor's usage from this node's latest ping-cycle sample
/// (VRAM only; RSS is probed fresh — the ping sample is stale for us).
fn usage_from_sample(
    rss_bytes: u64,
    sample: &Option<ResourceSample>,
    bandwidth_bytes_per_min: Option<u64>,
) -> Usage {
    let vram = sample.as_ref().and_then(|s| {
        s.gpus
            .iter()
            .filter_map(|g| g.vram_used_bytes)
            .reduce(|a, b| a + b)
    });
    Usage {
        rss_bytes,
        vram_used_bytes: vram,
        bandwidth_bytes_per_min,
    }
}

/// Spawn the governor loop: one task, ticking `interval`, judging fresh
/// samples against the live `[resources]` config and acting on verdicts.
/// No ceilings configured → the loop idles (legacy behavior, zero cost).
pub fn spawn_governor(
    state: Arc<crate::ping::MeshState>,
    interval: Duration,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let standby = Duration::from_secs(5).max(interval);
        // Bandwidth window: consecutive (ts, counter) snapshots of the
        // relayed-bytes meter. The rate is only published once the oldest
        // spanning sample is ≥ 45 s old — a fresh node reports unknown,
        // never a fake zero.
        let mut bw_window: Vec<(std::time::Instant, u64)> = Vec::new();
        loop {
            let (cfg, iso) = {
                let c = state.config.read().await;
                (c.resources.clone(), state.isolate.clone())
            };
            // Armed at all? Ceiling-free + no VRAM ceiling = pure no-op.
            if cfg.ram_ceiling_bytes.is_none()
                && cfg.vram_ceiling_bytes.is_none()
                && cfg.bandwidth_ceiling_bytes_per_min.is_none()
            {
                tokio::time::sleep(standby).await;
                continue;
            }
            // Measure the relayed-bytes rate against the counter window.
            let now_counter = state.meters.lock().await.bytes_relayed;
            let now = std::time::Instant::now();
            if bw_window.last().map(|(t, _)| now - *t) >= Some(interval) || bw_window.is_empty() {
                if bw_window.len() >= 8 {
                    bw_window.remove(0);
                }
                bw_window.push((now, now_counter));
            }
            let bandwidth_rate = bw_window.first().and_then(|(old_ts, old_counter)| {
                let secs = now - *old_ts;
                if secs.as_secs() >= 45 {
                    Some(
                        ((now_counter.saturating_sub(*old_counter) as f64)
                            / (secs.as_secs_f64() / 60.0)) as u64,
                    )
                } else {
                    None
                }
            });
            iso.record_bandwidth_rate(bandwidth_rate);
            let usage = {
                let sample = state.resources.lock().await;
                usage_from_sample(self_rss_bytes(), &sample, bandwidth_rate)
            };
            apply_verdict(&state, &iso, &cfg, decide(&usage, &cfg, iso.tripped())).await;
            tokio::time::sleep(interval).await;
        }
    })
}

/// Effect one [`Verdict`] against live state (transition logging here, so
/// `decide` itself stays pure and testable without a node). `cfg` is
/// consulted only through the verdict it produced.
async fn apply_verdict(
    state: &Arc<crate::ping::MeshState>,
    iso: &Arc<IsolateState>,
    _cfg: &ResourceIsolationConfig,
    verdict: Verdict,
) {
    match verdict {
        Verdict::Nominal => {}
        Verdict::Shed(reason) => {
            if let Some(store) = state.expert_store().await {
                // Shed to half the configured hot-set, not to zero: keep
                // the truly hot experts resident while relieving pressure.
                store.evict_to(store.cache_budget_bytes() / 2);
                iso.count_shed();
            }
            tracing::warn!(target: "mesh", "{reason} — expert hot set shed (mesh work continues)");
        }
        Verdict::Trip(reason) => {
            if iso.record_trip(&reason) {
                tracing::error!(
                    target: "mesh",
                    "{reason} — KILL SWITCH TRIPPED: expert serving/dispatch, mesh chat fallback and replication are refused until usage recedes (local-only mode)"
                );
            }
        }
        Verdict::RemainTripped(_) => {
            // Stay tripped; no log spam per tick.
        }
        Verdict::Recover => {
            if iso.record_recover() {
                tracing::warn!(
                    target: "mesh",
                    "usage receded under the soft line — kill switch disengaged, mesh work resumed"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ResourceIsolationConfig;

    const GIB: u64 = 1024 * 1024 * 1024;

    fn cfg(
        ram_ceiling: Option<u64>,
        vram_ceiling: Option<u64>,
        kill: bool,
    ) -> ResourceIsolationConfig {
        ResourceIsolationConfig {
            ram_ceiling_bytes: ram_ceiling,
            vram_ceiling_bytes: vram_ceiling,
            bandwidth_ceiling_bytes_per_min: None,
            pin_cores: 0,
            kill_switch: kill,
        }
    }

    #[test]
    fn nominal_when_under_soft_line() {
        let c = cfg(Some(10 * GIB), None, true);
        let u = Usage {
            rss_bytes: 5 * GIB,
            vram_used_bytes: None,
            bandwidth_bytes_per_min: None,
        };
        assert_eq!(decide(&u, &c, false), Verdict::Nominal);
    }

    #[test]
    fn soft_line_sheds_cache_but_keeps_mesh_work_on() {
        let c = cfg(Some(10 * GIB), None, true);
        // 9.5 GiB ≥ 90 % of 10 GiB, hard line not reached.
        let u = Usage {
            rss_bytes: 19 * GIB / 2,
            vram_used_bytes: None,
            bandwidth_bytes_per_min: None,
        };
        let v = decide(&u, &c, false);
        assert!(matches!(v, Verdict::Shed(_)), "got {v:?}");
    }

    #[test]
    fn hard_line_with_switch_on_trips() {
        let c = cfg(Some(10 * GIB), None, true);
        let u = Usage {
            rss_bytes: 10 * GIB,
            vram_used_bytes: None,
            bandwidth_bytes_per_min: None,
        };
        assert!(matches!(decide(&u, &c, false), Verdict::Trip(_)));
    }

    #[test]
    fn hard_line_with_switch_off_only_sheds() {
        let c = cfg(Some(10 * GIB), None, false);
        let u = Usage {
            rss_bytes: 11 * GIB,
            vram_used_bytes: None,
            bandwidth_bytes_per_min: None,
        };
        assert!(matches!(decide(&u, &c, false), Verdict::Shed(_)));
    }

    #[test]
    fn no_ceiling_is_a_noop() {
        let c = cfg(None, None, true);
        let u = Usage {
            rss_bytes: u64::MAX / 2,
            vram_used_bytes: None,
            bandwidth_bytes_per_min: None,
        };
        assert_eq!(decide(&u, &c, false), Verdict::Nominal);
    }

    #[test]
    fn vram_unknown_is_skipped_not_zero() {
        // No nvidia-smi: VRAM unknown must NOT count as zero (which would
        // let a VRAM ceiling trip); the branch is skipped entirely.
        let c = cfg(None, Some(2 * GIB), true);
        let u = Usage {
            rss_bytes: 0,
            vram_used_bytes: None,
            bandwidth_bytes_per_min: None,
        };
        assert_eq!(decide(&u, &c, false), Verdict::Nominal);
    }

    #[test]
    fn vram_hard_line_trips_and_soft_line_sheds() {
        let c = cfg(None, Some(8 * GIB), true);
        let soft = Usage {
            rss_bytes: 0,
            vram_used_bytes: Some(39 * GIB / 5),
            bandwidth_bytes_per_min: None,
        };
        assert!(matches!(decide(&soft, &c, false), Verdict::Shed(_)));
        let hard = Usage {
            rss_bytes: 0,
            vram_used_bytes: Some(8 * GIB),
            bandwidth_bytes_per_min: None,
        };
        assert!(matches!(decide(&hard, &c, false), Verdict::Trip(_)));
    }

    #[test]
    fn tripped_only_recovers_under_the_soft_line() {
        let c = cfg(Some(10 * GIB), None, true);
        let hard = Usage {
            rss_bytes: 10 * GIB,
            vram_used_bytes: None,
            bandwidth_bytes_per_min: None,
        };
        let between = Usage {
            rss_bytes: 9 * GIB,
            vram_used_bytes: None,
            bandwidth_bytes_per_min: None,
        };
        let under = Usage {
            rss_bytes: (9 * GIB) - 1,
            vram_used_bytes: None,
            bandwidth_bytes_per_min: None,
        };
        assert!(matches!(decide(&hard, &c, true), Verdict::RemainTripped(_)));
        assert!(matches!(
            decide(&between, &c, true),
            Verdict::RemainTripped(_)
        ));
        assert_eq!(decide(&under, &c, true), Verdict::Recover);
    }

    #[test]
    fn any_remaining_soft_breach_holds_recover() {
        // RAM under its soft line but VRAM still over: stay tripped.
        let c = cfg(Some(10 * GIB), Some(8 * GIB), true);
        let u = Usage {
            rss_bytes: GIB,
            vram_used_bytes: Some(39 * GIB / 5),
            bandwidth_bytes_per_min: None,
        };
        assert!(matches!(decide(&u, &c, true), Verdict::RemainTripped(_)));
    }

    #[test]
    fn kill_switch_state_transitions_are_idempotent() {
        let iso = IsolateState::default();
        assert!(!iso.tripped());
        assert!(iso.record_trip("RSS over ceiling"));
        assert!(
            !iso.record_trip("RSS over ceiling again"),
            "first reason wins"
        );
        assert!(iso.tripped());
        assert_eq!(iso.tripped_reason().as_deref(), Some("RSS over ceiling"));
        assert!(iso.record_recover());
        assert!(!iso.record_recover(), "recover is also transition-only");
        assert!(!iso.tripped());
        assert_eq!(iso.tripped_reason(), None);
        assert_eq!(iso.trips.load(Ordering::Relaxed), 1);
        assert_eq!(iso.recovers.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn usage_from_sample_sums_only_reporting_gpus() {
        let sample = ResourceSample {
            gpus: vec![
                crate::hw::GpuSample {
                    name: "A".into(),
                    vram_total_bytes: Some(8 * GIB),
                    vram_used_bytes: Some(2 * GIB),
                    vram_free_bytes: Some(6 * GIB),
                },
                crate::hw::GpuSample {
                    name: "B".into(),
                    vram_used_bytes: Some(3 * GIB),
                    ..Default::default()
                },
            ],
            ..ResourceSample::default()
        };
        let u = usage_from_sample(GIB, &Some(sample), Some(100 * 1024 * 1024));
        assert_eq!(u.rss_bytes, GIB);
        assert_eq!(u.vram_used_bytes, Some(5 * GIB));
        assert_eq!(u.bandwidth_bytes_per_min, Some(100 * 1024 * 1024));
        // Unknown VRAM stays None, never 0. Same for an unmeasured window.
        let u2 = usage_from_sample(0, &None, None);
        assert_eq!(u2.vram_used_bytes, None);
        assert_eq!(u2.bandwidth_bytes_per_min, None);
    }

    // ------------------------------------------------------------------
    // Bandwidth caps (Phase 2): soft/hard lines on relayed MiB/min. The
    // unknown-rate rule mirrors VRAM: a too-young window is skipped (None),
    // never treated as zero traffic.
    // ------------------------------------------------------------------

    const MIBMIN: u64 = 1024 * 1024;

    fn bw_usage(rate: Option<u64>) -> Usage {
        Usage {
            rss_bytes: 0,
            vram_used_bytes: None,
            bandwidth_bytes_per_min: rate,
        }
    }

    #[test]
    fn bandwidth_hard_line_trips_and_soft_line_sheds() {
        let c = ResourceIsolationConfig {
            bandwidth_ceiling_bytes_per_min: Some(1024 * MIBMIN),
            ..ResourceIsolationConfig::default()
        };
        // Hard breach: 1 GiB/min reached.
        assert!(
            matches!(
                decide(&bw_usage(Some(1024 * MIBMIN)), &c, false),
                Verdict::Trip(_)
            ),
            "hard bandwidth breach trips"
        );
        // Soft breach: 95 % of the ceiling.
        assert!(
            matches!(
                decide(&bw_usage(Some(970 * MIBMIN)), &c, false),
                Verdict::Shed(_)
            ),
            "soft bandwidth breach sheds"
        );
        // Under the soft line: nominal.
        assert_eq!(
            decide(&bw_usage(Some(500 * MIBMIN)), &c, false),
            Verdict::Nominal
        );
    }

    #[test]
    fn bandwidth_unknown_rate_is_skipped_not_zero() {
        let c = ResourceIsolationConfig {
            bandwidth_ceiling_bytes_per_min: Some(64 * MIBMIN),
            ..ResourceIsolationConfig::default()
        };
        // A too-young measurement window (None) cannot trip: unknown ≠ 0.
        assert_eq!(decide(&bw_usage(None), &c, false), Verdict::Nominal);
    }

    #[test]
    fn bandwidth_breach_holds_recover_until_under_soft_line() {
        let c = ResourceIsolationConfig {
            bandwidth_ceiling_bytes_per_min: Some(1024 * MIBMIN),
            ..ResourceIsolationConfig::default()
        };
        // Tripped by bandwidth; still over the soft line → stay tripped.
        assert!(matches!(
            decide(&bw_usage(Some(950 * MIBMIN)), &c, true),
            Verdict::RemainTripped(_)
        ));
        // Receded under the soft line → recover.
        assert_eq!(
            decide(&bw_usage(Some(500 * MIBMIN)), &c, true),
            Verdict::Recover
        );
        // A too-young window can no longer hold the trip either.
        assert_eq!(decide(&bw_usage(None), &c, true), Verdict::Recover);
    }

    #[test]
    fn bandwidth_rate_publication_roundtrips_through_state() {
        let iso = IsolateState::default();
        assert_eq!(iso.bandwidth_rate_min(), None, "unmeasured at boot");
        iso.record_bandwidth_rate(Some(300 * MIBMIN));
        assert_eq!(iso.bandwidth_rate_min(), Some(300 * MIBMIN));
        iso.record_bandwidth_rate(None);
        assert_eq!(iso.bandwidth_rate_min(), None, "unknown ≠ zero");
        let snap = snapshot(&iso);
        assert_eq!(snap["bandwidth_rate_min"], serde_json::json!(0));
    }
}
