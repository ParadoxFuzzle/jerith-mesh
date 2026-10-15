//! Perf/cost meters — model-performance-benchmark + session-cost-tracker
//! ports, built on the gateway's existing counters.
//!
//! What OpenClaw's two skills did (latency/success per model; real token
//! counts per session) now lives in the mesh node one layer lower:
//!
//! - **Per-request capture** (gateway `chat_completions` completion paths):
//!   model id, latency, status, prompt/completion/total tokens (from the
//!   runtime's `usage` when present), served-by (local vs mesh peer).
//! - **Aggregates** ([`Metrics::snapshot`]): per-model request/latency/
//!   token totals plus global success rate — the ranking inputs the
//!   OpenClaw skill computed from trajectory logs, here from live traffic.
//! - **Exposure**: `GET /v1/metrics` (full JSON) and a `metrics` section
//!   in `/status` — both cheap reads over a ring + a BTreeMap.
//!
//! The ring is capped: a node that serves thousands of requests keeps a
//! bounded window (the aggregates carry the totals). Per-user accounting
//! is bot-side (`state.record_usage`), fed from this same `usage` data.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use serde::Serialize;

/// Ring cap: recent completions retained verbatim (oldest dropped).
const RING_CAP: usize = 500;

// ---- Security-reject counters (process-local, monotonic) ----
//
// 429s from the shared rate limiter and size-bound rejections (chat
// forward cap, memory caps, bootstrap ad-field caps + registry cap), so
// ops can read attack pressure straight off /status instead of
// grepping journal logs. Statics rather than state on `Gateway`
// because: (a) rejection sites span two processes (the node gateway
// and the standalone bootstrap), (b) several bump sites have no
// Gateway handle, and (c) they are process-local by definition — the
// node's /status and the bootstrap's /health each report their own
// process's counters. Relaxed ordering: these are observability
// counters, exact cross-thread sequencing is irrelevant.

pub static RATE_LIMITED_429_TOTAL: AtomicU64 = AtomicU64::new(0);
/// Chat payloads over `CHAT_FORWARD_CAP_BYTES` (gateway).
pub static OVERSIZE_CHAT_REJECTS: AtomicU64 = AtomicU64::new(0);
/// Memory embed text / search query over their char caps (gateway).
pub static OVERSIZE_MEMORY_REJECTS: AtomicU64 = AtomicU64::new(0);
/// Bootstrap ad-field length caps + registry-entry-cap refusals.
pub static OVERSIZE_BOOTSTRAP_REJECTS: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, Serialize)]
pub struct SecurityCounters {
    pub rate_limited_429_total: u64,
    pub oversize_reject_total: u64,
    pub oversize_chat_rejects: u64,
    pub oversize_memory_rejects: u64,
    pub oversize_bootstrap_rejects: u64,
}

/// Bump one counter (relaxed ordering fine for observability).
pub fn bump(counter: &AtomicU64) {
    counter.fetch_add(1, Ordering::Relaxed);
}

fn peek(counter: &AtomicU64) -> u64 {
    counter.load(Ordering::Relaxed)
}

/// Current security-reject counters, for the /status `security` section
/// and the bootstrap /health `security` section.
pub fn security_snapshot() -> SecurityCounters {
    let chat = peek(&OVERSIZE_CHAT_REJECTS);
    let memory = peek(&OVERSIZE_MEMORY_REJECTS);
    let bootstrap = peek(&OVERSIZE_BOOTSTRAP_REJECTS);
    SecurityCounters {
        rate_limited_429_total: peek(&RATE_LIMITED_429_TOTAL),
        oversize_reject_total: chat + memory + bootstrap,
        oversize_chat_rejects: chat,
        oversize_memory_rejects: memory,
        oversize_bootstrap_rejects: bootstrap,
    }
}

/// One completed (or failed) chat completion, as the gateway saw it.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionRecord {
    /// Model id the request was (re)routed to.
    pub model: String,
    /// Wall-clock milliseconds for the whole completion exchange.
    pub latency_ms: u64,
    /// Terminal HTTP-ish status this path returned (200/502/…).
    pub status: u16,
    /// Prompt tokens from `usage` (0 when the runtime omits it).
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    /// `None` → local runtime; Some → mesh peer that served it.
    pub served_by: Option<String>,
    /// Epoch ms (ordering aid for clients).
    pub at_ms: u64,
}

/// Per-model aggregates — the benchmark ranking view.
#[derive(Debug, Default, Clone, Serialize)]
pub struct ModelStats {
    pub requests: u64,
    pub ok: u64,
    pub failed: u64,
    pub latency_ms_total: u64,
    pub latency_ms_max: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    /// Served by mesh peers rather than the local runtime.
    pub mesh_served: u64,
}

impl ModelStats {
    /// Mean latency in ms (0 when no requests).
    pub fn latency_ms_mean(&self) -> u64 {
        self.latency_ms_total.checked_div(self.requests).unwrap_or(0)
    }
    /// Success ratio in percent (0 when no requests).
    pub fn ok_percent(&self) -> u64 {
        (self.ok * 100).checked_div(self.requests).unwrap_or(0)
    }
}

/// Full snapshot — the /v1/metrics body.
#[derive(Debug, Default, Clone, Serialize)]
pub struct MetricsSnapshot {
    pub requests_total: u64,
    pub ok_total: u64,
    pub failed_total: u64,
    pub tokens_total: u64,
    /// Per-model stats, sorted by model id.
    pub models: BTreeMap<String, ModelStats>,
    /// Most recent completions, oldest first (ring window).
    pub recent: Vec<CompletionRecord>,
    /// When the metering window opened (node start).
    pub window_started_ms: u64,
}

/// Shared meter handle (inside the Gateway).
#[derive(Debug)]
pub struct Metrics {
    inner: Mutex<MetricsInner>,
}

#[derive(Debug, Default)]
struct MetricsInner {
    ring: Vec<CompletionRecord>,
    models: BTreeMap<String, ModelStats>,
    requests_total: u64,
    ok_total: u64,
    failed_total: u64,
    tokens_total: u64,
    /// When the metering window opened (node start), epoch ms.
    window_started_ms: u64,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(MetricsInner {
                window_started_ms: crate::bootstrap::now_ms(),
                ..Default::default()
            }),
        }
    }

    /// Record one finished completion (success or failure). Cheap: one
    /// push + one map entry update under a short lock.
    pub fn record(&self, rec: CompletionRecord) {
        let mut inner = self.inner.lock().expect("metrics lock");
        inner.requests_total += 1;
        if rec.status < 400 {
            inner.ok_total += 1;
        } else {
            inner.failed_total += 1;
        }
        inner.tokens_total += rec.total_tokens;
        let entry = inner.models.entry(rec.model.clone()).or_default();
        entry.requests += 1;
        if rec.status < 400 {
            entry.ok += 1;
        } else {
            entry.failed += 1;
        }
        entry.latency_ms_total += rec.latency_ms;
        entry.latency_ms_max = entry.latency_ms_max.max(rec.latency_ms);
        entry.prompt_tokens += rec.prompt_tokens;
        entry.completion_tokens += rec.completion_tokens;
        entry.total_tokens += rec.total_tokens;
        if rec.served_by.is_some() {
            entry.mesh_served += 1;
        }
        inner.ring.push(rec);
        if inner.ring.len() > RING_CAP {
            let excess = inner.ring.len() - RING_CAP;
            inner.ring.drain(..excess);
        }
    }

    /// One record constructed from the parts the gateway has on hand.
    pub fn record_simple(
        &self,
        model: &str,
        started: Instant,
        status: u16,
        usage: Option<&serde_json::Value>,
        served_by: Option<&str>,
    ) {
        let (p, c, t) = usage
            .map(|u| {
                (
                    u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
                    u.get("completion_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0),
                    u.get("total_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
                )
            })
            .unwrap_or((0, 0, 0));
        self.record(CompletionRecord {
            model: model.to_string(),
            latency_ms: started.elapsed().as_millis() as u64,
            status,
            prompt_tokens: p,
            completion_tokens: c,
            total_tokens: if t > 0 { t } else { p + c },
            served_by: served_by.map(str::to_string),
            at_ms: crate::bootstrap::now_ms(),
        });
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        let inner = self.inner.lock().expect("metrics lock");
        MetricsSnapshot {
            requests_total: inner.requests_total,
            ok_total: inner.ok_total,
            failed_total: inner.failed_total,
            tokens_total: inner.tokens_total,
            models: inner.models.clone(),
            recent: inner.ring.clone(),
            window_started_ms: inner.window_started_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(model: &str, status: u16, latency_ms: u64, total_tokens: u64) -> CompletionRecord {
        CompletionRecord {
            model: model.to_string(),
            latency_ms,
            status,
            prompt_tokens: total_tokens / 2,
            completion_tokens: total_tokens - total_tokens / 2,
            total_tokens,
            served_by: None,
            at_ms: 0,
        }
    }

    #[test]
    fn aggregates_per_model_and_globally() {
        let m = Metrics::new();
        m.record(rec("qwen3:8b", 200, 100, 50));
        m.record(rec("qwen3:8b", 200, 300, 70));
        m.record(rec("glm:7b", 502, 30, 0));
        let snap = m.snapshot();
        assert_eq!(snap.requests_total, 3);
        assert_eq!(snap.ok_total, 2);
        assert_eq!(snap.failed_total, 1);
        assert_eq!(snap.tokens_total, 120);
        let q = &snap.models["qwen3:8b"];
        assert_eq!(q.requests, 2);
        assert_eq!(q.latency_ms_mean(), 200);
        assert_eq!(q.latency_ms_max, 300);
        assert_eq!(q.ok_percent(), 100);
        assert_eq!(q.total_tokens, 120);
        let g = &snap.models["glm:7b"];
        assert_eq!(g.ok_percent(), 0);
        // Ring window capped and ordered oldest-first.
        assert_eq!(snap.recent.len(), 3);
    }

    #[test]
    fn ring_drops_oldest_beyond_cap() {
        let m = Metrics::new();
        for i in 0..(RING_CAP + 10) {
            m.record(rec("m", 200, 1, i as u64));
        }
        let snap = m.snapshot();
        assert_eq!(snap.recent.len(), RING_CAP);
        assert_eq!(snap.requests_total, (RING_CAP + 10) as u64);
        // Oldest dropped: first retained is the 11th record.
        assert_eq!(snap.recent[0].total_tokens, 10);
    }

    #[test]
    fn record_simple_pulls_usage_fields_and_mesh_attribution() {
        let m = Metrics::new();
        let started = Instant::now();
        m.record_simple(
            "qwen3:8b",
            started,
            200,
            Some(&serde_json::json!({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})),
            Some("peer-node"),
        );
        let snap = m.snapshot();
        assert_eq!(snap.models["qwen3:8b"].mesh_served, 1);
        assert_eq!(snap.models["qwen3:8b"].total_tokens, 15);
        // Zero-token usage (runtime without usage figures) still counts as
        // a completed request.
        m.record_simple("qwen3:8b", started, 200, None, None);
        assert_eq!(snap_chain(&m).models["qwen3:8b"].requests, 2);
    }

    fn snap_chain(m: &Metrics) -> MetricsSnapshot {
        m.snapshot()
    }
}
