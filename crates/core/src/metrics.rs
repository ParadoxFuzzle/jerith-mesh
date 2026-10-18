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

/// History bucket width: the trend view aggregates completed completions
/// into fixed five-minute buckets (epoch-aligned) so the dashboard can draw
/// load/latency over time from a bounded structure instead of re-reading
/// the (shorter) record ring.
const BUCKET_MS: u64 = 300_000;
/// Retained buckets: 5 min × 120 = 10 h of history, oldest dropped.
const HISTORY_CAP: usize = 120;
/// Cap on the per-peer breakdown. The peer registry is already bounded,
/// but a hostile alias storm must not balloon /v1/metrics: beyond the cap
/// an unknown peer folds into a single `_other` row.
const PEER_CAP: usize = 128;

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
        self.latency_ms_total
            .checked_div(self.requests)
            .unwrap_or(0)
    }
    /// Success ratio in percent (0 when no requests).
    pub fn ok_percent(&self) -> u64 {
        (self.ok * 100).checked_div(self.requests).unwrap_or(0)
    }
}

/// Per-peer served aggregates — who actually answered the mesh for us.
/// Same shape as the per-model view, keyed by the node id in
/// `CompletionRecord::served_by`, so ops can see a slow or flaky peer
/// without grepping ring records.
#[derive(Debug, Default, Clone, Serialize)]
pub struct PeerStats {
    pub peer: String,
    pub requests: u64,
    pub ok: u64,
    pub failed: u64,
    pub latency_ms_total: u64,
    pub latency_ms_max: u64,
    pub tokens_total: u64,
}

impl PeerStats {
    /// Mean latency in ms (0 when no requests).
    pub fn latency_ms_mean(&self) -> u64 {
        self.latency_ms_total
            .checked_div(self.requests)
            .unwrap_or(0)
    }
    /// Success ratio in percent (0 when no requests).
    pub fn ok_percent(&self) -> u64 {
        (self.ok * 100).checked_div(self.requests).unwrap_or(0)
    }
}

/// One time slice (BUCKET_MS wide) for the trend view: counts, latency
/// and the mesh share of the completions that landed inside it.
#[derive(Debug, Default, Clone, Serialize)]
pub struct HistoryBucket {
    /// Epoch ms of the bucket start (multiple of BUCKET_MS).
    pub started_ms: u64,
    pub requests: u64,
    pub ok: u64,
    pub latency_ms_total: u64,
    pub latency_ms_max: u64,
    /// Completions served by mesh peers (vs the local runtime).
    pub mesh_served: u64,
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
    /// M3 pipeline-route counters (see “PipelineMetrics”).
    pub pipeline: PipelineMetrics,
    /// Per-peer served aggregates (mesh-chat attribution), oldest record
    /// order aside: sorted by node id (BTreeMap iteration order).
    pub peers: Vec<PeerStats>,
    /// 5-minute trend buckets, oldest first (bounded to HISTORY_CAP).
    pub history: Vec<HistoryBucket>,
}

/// Pipeline route counters (M3, FEATURES §13): planned and timed route
/// walks plus the hops that overran their measured-demand budget. The
/// fields are live as soon as a producer reports — routing has no live
/// invocation path yet (pending the Colibri-side transport), so until
/// then these read zero and the dashboard shows “no route data yet”.
#[derive(Debug, Default, Clone, Serialize, PartialEq, Eq)]
pub struct PipelineMetrics {
    pub routes_total: u64,
    pub over_budget_hops: u64,
    /// Duration of the most recent walk (sum of its attempt timings).
    pub last_route_ms: u64,
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
    /// Per-peer served breakdown (bounded: see PEER_CAP).
    peers: BTreeMap<String, PeerStats>,
    /// 5-minute trend buckets, oldest first (bounded: see HISTORY_CAP).
    history: Vec<HistoryBucket>,
    requests_total: u64,
    ok_total: u64,
    failed_total: u64,
    tokens_total: u64,
    /// When the metering window opened (node start), epoch ms.
    window_started_ms: u64,
    /// M3 pipeline-route counters.
    pipeline_routes_total: u64,
    pipeline_over_budget_hops: u64,
    pipeline_last_route_ms: u64,
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
        let ok = rec.status < 400;
        if ok {
            inner.ok_total += 1;
        } else {
            inner.failed_total += 1;
        }
        inner.tokens_total += rec.total_tokens;
        let entry = inner.models.entry(rec.model.clone()).or_default();
        entry.requests += 1;
        if ok {
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
        // Per-peer breakdown (bounded). Beyond the cap an unseen peer folds
        // into `_other` — one honest overflow row instead of unbounded map
        // growth from a hostile alias storm.
        if let Some(peer) = rec.served_by.as_deref() {
            let key = if inner.peers.contains_key(peer) || inner.peers.len() < PEER_CAP {
                peer.to_string()
            } else {
                "_other".to_string()
            };
            let p = inner.peers.entry(key.clone()).or_default();
            p.peer = key;
            p.requests += 1;
            if ok {
                p.ok += 1;
            } else {
                p.failed += 1;
            }
            p.latency_ms_total += rec.latency_ms;
            p.latency_ms_max = p.latency_ms_max.max(rec.latency_ms);
            p.tokens_total += rec.total_tokens;
        }
        // Trend buckets (bounded). Records carry a monotonic epoch stamp in
        // normal operation; a stamp older than the newest bucket (clock
        // rewound mid-run) skips the bucket update rather than inserting
        // into the middle of the ring.
        let idx = rec.at_ms / BUCKET_MS;
        match inner.history.last_mut() {
            // The record lands inside the newest bucket: update in place.
            Some(b) if b.started_ms == idx * BUCKET_MS => {
                b.requests += 1;
                b.ok += u64::from(ok);
                b.latency_ms_total += rec.latency_ms;
                b.latency_ms_max = b.latency_ms_max.max(rec.latency_ms);
                b.mesh_served += u64::from(rec.served_by.is_some());
            }
            // Older than the newest bucket — skip, never insert mid-ring.
            Some(b) if b.started_ms > idx * BUCKET_MS => {}
            // First bucket ever, or a newer epoch: open a new bucket.
            _ => {
                inner.history.push(HistoryBucket {
                    started_ms: idx * BUCKET_MS,
                    requests: 1,
                    ok: u64::from(ok),
                    latency_ms_total: rec.latency_ms,
                    latency_ms_max: rec.latency_ms,
                    mesh_served: u64::from(rec.served_by.is_some()),
                });
                if inner.history.len() > HISTORY_CAP {
                    let excess = inner.history.len() - HISTORY_CAP;
                    inner.history.drain(..excess);
                }
            }
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

    /// Record one finished pipeline route walk (M3): its total attempt
    /// latency and how many hops overran their priced budget.
    pub fn record_pipeline_route(&self, walk_ms: u64, over_budget_hops: usize) {
        let mut inner = self.inner.lock().expect("metrics lock");
        inner.pipeline_routes_total += 1;
        inner.pipeline_over_budget_hops += over_budget_hops as u64;
        inner.pipeline_last_route_ms = walk_ms;
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
            pipeline: PipelineMetrics {
                routes_total: inner.pipeline_routes_total,
                over_budget_hops: inner.pipeline_over_budget_hops,
                last_route_ms: inner.pipeline_last_route_ms,
            },
            peers: inner.peers.values().cloned().collect(),
            history: inner.history.clone(),
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

    fn rec_at(
        model: &str,
        status: u16,
        latency_ms: u64,
        peer: Option<&str>,
        at_ms: u64,
    ) -> CompletionRecord {
        CompletionRecord {
            model: model.to_string(),
            latency_ms,
            status,
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            served_by: peer.map(str::to_string),
            at_ms,
        }
    }

    #[test]
    fn per_peer_breakdown_attributes_latency_successfully() {
        let m = Metrics::new();
        m.record(rec_at("m", 200, 100, Some("peer-aa"), 1_000));
        m.record(rec_at("m", 200, 300, Some("peer-aa"), 2_000));
        m.record(rec_at("m", 502, 40, Some("peer-zz"), 3_000));
        // Local-runtime completion: never a peer row.
        m.record(rec_at("m", 200, 50, None, 4_000));
        let snap = m.snapshot();
        assert_eq!(snap.peers.len(), 2, "sorted by node id");
        let (aa, zz) = (&snap.peers[0], &snap.peers[1]);
        assert_eq!(aa.peer, "peer-aa");
        assert_eq!(aa.requests, 2);
        assert_eq!(aa.latency_ms_mean(), 200);
        assert_eq!(aa.latency_ms_max, 300);
        assert_eq!(aa.ok_percent(), 100);
        assert_eq!(zz.peer, "peer-zz");
        assert_eq!(zz.requests, 1);
        assert_eq!(zz.ok_percent(), 0);
    }

    #[test]
    fn peer_breakdown_overflow_folds_into_other() {
        let m = Metrics::new();
        for i in 0..(PEER_CAP + 3) {
            m.record(rec_at("m", 200, 10, Some(&format!("peer-{i:03}")), 1_000));
        }
        let snap = m.snapshot();
        assert_eq!(snap.peers.len(), PEER_CAP + 1, "cap + _other row");
        // '_' (0x5F) sorts before 'p' (0x70): the overflow row comes FIRST
        // in the id-sorted vector, so look it up rather than take last().
        let other = snap
            .peers
            .iter()
            .find(|p| p.peer == "_other")
            .expect("_other row exists");
        assert_eq!(other.requests, 3, "peers beyond the cap fold here");
        // A known peer stored before the cap keeps filling its own row.
        m.record(rec_at("m", 200, 10, Some("peer-000"), 1_100));
        let snap = m.snapshot();
        assert_eq!(snap.peers.first().unwrap().peer, "_other");
        let zero = snap
            .peers
            .iter()
            .find(|p| p.peer == "peer-000")
            .expect("pre-cap peer row");
        assert_eq!(zero.requests, 2);
    }

    #[test]
    fn history_buckets_aggregate_by_epoch_slice_and_bound() {
        let m = Metrics::new();
        // Same 5-min bucket: merged in place.
        m.record(rec_at("m", 200, 100, Some("p"), BUCKET_MS + 1));
        m.record(rec_at("m", 200, 300, None, BUCKET_MS + 2));
        // Next bucket: new entry; mesh attribution carried.
        m.record(rec_at("m", 502, 50, Some("p"), BUCKET_MS * 2 + 1));
        let snap = m.snapshot();
        assert_eq!(snap.history.len(), 2);
        let (b0, b1) = (&snap.history[0], &snap.history[1]);
        assert_eq!(b0.started_ms, BUCKET_MS);
        assert_eq!(b0.requests, 2);
        assert_eq!(b0.ok, 2);
        assert_eq!(b0.latency_ms_max, 300);
        assert_eq!(b0.mesh_served, 1);
        assert_eq!(b1.started_ms, BUCKET_MS * 2);
        assert_eq!(b1.ok, 0);
        // Bounded: the bound loop opens BUCKET epochs *40..*(40+241) on
        // top of the two earlier buckets; only the newest HISTORY_CAP
        // survive (oldest dropped, 5-min each).
        for i in 0..(HISTORY_CAP * 2 + 1) {
            m.record(rec_at(
                "m",
                200,
                10,
                None,
                BUCKET_MS * u64::try_from(40 + i).unwrap(),
            ));
        }
        let snap = m.snapshot();
        assert_eq!(snap.history.len(), HISTORY_CAP);
        assert_eq!(
            snap.history.first().unwrap().started_ms,
            BUCKET_MS * u64::try_from(40 + HISTORY_CAP + 1).unwrap()
        );
    }

    #[test]
    fn history_bucket_skips_a_clock_rewound_record() {
        let m = Metrics::new();
        m.record(rec_at("m", 200, 100, None, BUCKET_MS * 5));
        // A stamp suddenly in the past must not disturb the newest bucket
        // (monotonic clock contract; a rewound stamp is a broken client),
        // and must not insert mid-ring either.
        m.record(rec_at("m", 200, 100, None, 3));
        let snap = m.snapshot();
        assert_eq!(snap.history.len(), 1);
        assert_eq!(snap.history[0].requests, 1);
        assert_eq!(snap.history[0].started_ms, BUCKET_MS * 5);
        // The record itself still lands in the aggregates + ring.
        assert_eq!(snap.requests_total, 2);
        assert_eq!(snap.recent.len(), 2);
    }

    #[test]
    fn pipeline_route_recorder_updates_the_snapshot() {
        let m = Metrics::new();
        m.record_pipeline_route(1_250, 2);
        m.record_pipeline_route(750, 0);
        let snap = m.snapshot();
        assert_eq!(snap.pipeline.routes_total, 2);
        assert_eq!(snap.pipeline.over_budget_hops, 2);
        assert_eq!(snap.pipeline.last_route_ms, 750);
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
