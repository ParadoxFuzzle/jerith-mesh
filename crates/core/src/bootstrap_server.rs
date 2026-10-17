//! Bootstrap node server: the single-file trust anchor at p2p.jerithai.com.
//!
//! Endpoints (Phase 0 HTTP/JSON wire format):
//!   GET  /health              liveness for Cloudflare Tunnel checks
//!   GET  /v1/peers?model=     live peer list for a model (TTL-filtered)
//!   PUT  /v1/advertise        node self-advertisement (heartbeat)
//!   POST /v1/report-dead      peer-reported death → immediate deprecation
//!   GET  /v1/models/counts    live peer count per model (for the setup UI)
//!
//! State is in-memory with TTL expiry: an entry disappears after
//! `advertise_ttl` without a refresh, comes back on the next advertise,
//! and /v1/report-dead evicts immediately. Multiple independent bootstrap
//! nodes can run the same binary; Phase 1 adds gossip between them.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::{Path, Query, Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::bootstrap::{BootstrapKeyInfo, PeerAd, PeerListResponse, now_ms};
use std::sync::OnceLock;

/// How long an advertisement stays live without refresh.
pub const DEFAULT_ADVERTISE_TTL_MS: u64 = 90_000;

/// Anonymous-read budget (GET /v1/peers, /v1/models/counts, /health): per
/// client IP per 60s window. A peer's own refresh cadence polls far below
/// this; the cap exists to keep one source from running endless sweeps
/// (every read re-tenders + scores the whole live set).
const ANON_READ_LIMIT_PER_MIN: u32 = 240;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapOptions {
    pub port: u16,
    pub advertise_ttl_ms: u64,
}

impl Default for BootstrapOptions {
    fn default() -> Self {
        Self {
            port: 8901,
            advertise_ttl_ms: DEFAULT_ADVERTISE_TTL_MS,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeadReport {
    node_id: String,
    /// Signed attribution: who saw the node dead. Verified server-side —
    /// an unauthenticated eviction vote is an internet-reachable
    /// take-down of any peer (a node can also self-report honestly, so
    /// this endpoint carries the same wire-auth contract as advertise).
    reporter: String,
}

/// In-memory registry: node_id → (ad, last_refresh_ms, dead_votes, score).
struct Registry {
    entries: RwLock<HashMap<String, Entry>>,
    /// Maximum live node identities retained (NEW ads beyond this are
    /// refused with 503; refreshes always succeed).
    max_entries: usize,
    /// §2 contribution quorum: contributors' forward-verified measurement
    /// blocks, tallied per subject (see `MeasureTally`). Bounded per the
    /// same registry cap discipline as `entries`.
    measures: RwLock<MeasureTally>,
}

impl Default for Registry {
    fn default() -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            max_entries: MAX_REGISTRY_ENTRIES,
            measures: RwLock::new(MeasureTally::default()),
        }
    }
}

/// ONE forwarded measurement block: the subject-signed figures plus who
/// attested delivering them. Reports dedupe on (reporter, block_stamp).
#[derive(Debug, Clone)]
struct MeasureRecord {
    reporter: String,
    block_stamp_ms: u64,
    window_secs: u64,
    measured_percent: u64,
}

/// §2 quorum constants. The subject's OWN signature anchors the figures;
/// what quorum adds is CORROBORATION — how many distinct peers observed
/// and forwarded the same self-signed numbers. Statistics, not verdicts:
/// a single report is "unquorumed", never a suspicion.
const MEASURE_TALLY_RING_PER_SUBJECT: usize = 32;
/// Measured-share spread (percentage points) across distinct reporters'
/// most-recent blocks above which the subject is flagged INCONSISTENT
/// (each asker received different self-signed numbers).
const MEASURE_CONSISTENCY_SLACK_PERCENT: u64 = 10;
/// Distinct reporters required before the aggregate says anything at all.
const MEASURE_QUORUM_MIN_REPORTERS: usize = 2;

/// The per-subject tally ring + aggregate view. Bounded: a ring of
/// `MEASURE_TALLY_RING_PER_SUBJECT` per subject, subjects bounded by the
/// live registry (sweep drops subjects no longer advertised).
#[derive(Debug, Default)]
struct MeasureTally {
    subjects: HashMap<String, Vec<MeasureRecord>>,
    uploads_accepted: u64,
    uploads_duped: u64,
}

impl MeasureTally {
    /// Record one (verified, deduped) observation. Returns the ring depth
    /// after the insert so the handler can report it.
    fn record(&mut self, subject: &str, rec: MeasureRecord) -> (usize, bool) {
        let ring = self.subjects.entry(subject.to_string()).or_default();
        if ring
            .iter()
            .any(|r| r.reporter == rec.reporter && r.block_stamp_ms == rec.block_stamp_ms)
        {
            self.uploads_duped += 1;
            return (ring.len(), false);
        }
        ring.push(rec);
        // Bounded ring: drop the OLDEST half when full (a full flush each
        // time keeps the newest window intact for downstream aggregation).
        while ring.len() >= MEASURE_TALLY_RING_PER_SUBJECT {
            ring.remove(0);
        }
        self.uploads_accepted += 1;
        (ring.len(), true)
    }

    /// The aggregate for one subject over its RING-recency horizon
    /// (`3 × window_secs` of the newest block): distinct reporters,
    /// measured spread, and — past the quorum floor — the consistency
    /// verdict (`None` = unquorumed: one reporter is never a suspicion).
    fn quorum_view(&self, subject: &str) -> Option<QuorumViewShared> {
        let ring = self.subjects.get(subject)?;
        let newest = ring.iter().map(|r| r.block_stamp_ms).max()?;
        let window = ring
            .iter()
            .filter(|r| r.block_stamp_ms == newest)
            .min_by_key(|r| r.window_secs)
            .map(|r| r.window_secs)
            .unwrap_or(60)
            .max(1);
        let horizon_ms = window.saturating_mul(3_000);
        let recent: Vec<&MeasureRecord> = ring
            .iter()
            .filter(|r| r.block_stamp_ms.saturating_add(horizon_ms) >= newest)
            .collect();
        let mut reporters: Vec<&str> = recent.iter().map(|r| r.reporter.as_str()).collect();
        reporters.sort_unstable();
        reporters.dedup();
        let measureds: Vec<u64> = recent.iter().map(|r| r.measured_percent).collect();
        let spread = measureds.iter().max().copied().unwrap_or(0)
            - measureds.iter().min().copied().unwrap_or(0);
        Some(QuorumViewShared {
            reports: recent.len(),
            distinct_reporters: reporters.len(),
            measured_median: median(measureds),
            spread_percent: spread,
            consistent: if reporters.len() < MEASURE_QUORUM_MIN_REPORTERS {
                None
            } else {
                Some(spread <= MEASURE_CONSISTENCY_SLACK_PERCENT)
            },
        })
    }

    /// The serialized form for `/v1/peers`: one entry per subject with a
    /// quorum view (subjects with no recent records stay absent).
    fn shared_json(&self) -> serde_json::Map<String, serde_json::Value> {
        let mut out = serde_json::Map::new();
        for subject in self.subjects.keys() {
            if let Some(v) = self.quorum_view(subject) {
                out.insert(
                    subject.clone(),
                    serde_json::json!({
                        "reports": v.reports,
                        "distinct_reporters": v.distinct_reporters,
                        "measured_median": v.measured_median,
                        "spread_percent": v.spread_percent,
                        "consistent": v.consistent,
                    }),
                );
            }
        }
        out
    }
}

fn median(mut v: Vec<u64>) -> u64 {
    if v.is_empty() {
        return 0;
    }
    v.sort_unstable();
    v[v.len() / 2]
}

/// The aggregate view (server-side shape before serializing).
#[derive(Debug, Clone, PartialEq, Eq)]
struct QuorumViewShared {
    reports: usize,
    distinct_reporters: usize,
    measured_median: u64,
    spread_percent: u64,
    consistent: Option<bool>,
}

/// Field-length sanity on an incoming advertisement. Length-only checks —
/// semantic validation (signature, identity binding) happens later in the
/// handler; this gate runs BEFORE any crypto work so oversized junk is
/// refused without spending CPU. The `Response` error is heap-boxed so the
/// `Result` itself stays register-sized (`result_large_err`).
fn validate_ad_fields(ad: &PeerAd) -> Result<(), Box<Response>> {
    let too_long = |field: &str, cap| {
        crate::metrics::bump(&crate::metrics::OVERSIZE_BOOTSTRAP_REJECTS);
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": format!("field '{field}' exceeds {cap} chars")
            })),
        )
            .into_response()
    };
    if ad.node_id.chars().count() > AD_NODE_ID_CAP_CHARS {
        return Err(too_long("node_id", AD_NODE_ID_CAP_CHARS).into());
    }
    if ad.model_id.chars().count() > AD_MODEL_CAP_CHARS {
        return Err(too_long("model_id", AD_MODEL_CAP_CHARS).into());
    }
    if ad.quant.chars().count() > AD_QUANT_CAP_CHARS {
        return Err(too_long("quant", AD_QUANT_CAP_CHARS).into());
    }
    if ad.address.chars().count() > AD_ADDRESS_CAP_CHARS {
        return Err(too_long("address", AD_ADDRESS_CAP_CHARS).into());
    }
    if ad.rpc_address.chars().count() > AD_ADDRESS_CAP_CHARS {
        return Err(too_long("rpc_address", AD_ADDRESS_CAP_CHARS).into());
    }
    if ad.contribution_percent > 100 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "field 'contribution_percent' must be 0-100"
            })),
        )
            .into_response()
            .into());
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct Entry {
    ad: PeerAd,
    last_refresh_ms: u64,
    /// Peer-submitted death reports; >= threshold evicts immediately.
    dead_votes: u32,
    /// Reputation: 0 at join, decremented by corroborated fraud reports,
    /// recovered by daily time-decay (+2/day) toward 0.
    score: i32,
    /// node_id of the reporting peer → (kind, last_report_ms) already held.
    fraud_reports: HashMap<String, (FraudKind, u64)>,
}

const DEAD_VOTES_TO_EVICTION: u32 = 2;

/// Wire-format sanity caps. Identity creation is free, so without these an
/// identity holder could stuff megabyte-sized string fields into the
/// registry — every entry is re-serialized into each /v1/peers response,
/// scaling both memory and bandwidth with one oversized ad.
const AD_NODE_ID_CAP_CHARS: usize = 64;
const AD_MODEL_CAP_CHARS: usize = 64;
const AD_QUANT_CAP_CHARS: usize = 32;
const AD_ADDRESS_CAP_CHARS: usize = 128;
/// Total registry entries retained. Refreshes of existing node ids are
/// exempt — only NEW identities count against the cap.
const MAX_REGISTRY_ENTRIES: usize = 20_000;

// --- Fraud scoring (peer-reported evidence + decay) -------------------

/// Absolute floor — combined penalties cannot push below this.
const REPUTATION_FLOOR: i32 = -100;
/// Reports beyond this many distinct peers count double (corroboration).
const MIN_CORROBORATORS: u32 = 2;

/// Fraud reasons — peers report observed failures, not opinions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum FraudKind {
    /// Signed peer claimed jobs and failed/never returned results.
    FailedWorker,
    /// Advertised resources but customarily declines real dispatches.
    Leeching,
    /// Results failed redundant replay comparison.
    DishonestResults,
    /// Rate abuse / resource exhaustion.
    RateAbuse,
}

fn penalty_for(kind: &FraudKind) -> i32 {
    match kind {
        FraudKind::FailedWorker => 30,
        FraudKind::Leeching => 20,
        FraudKind::DishonestResults => 40,
        FraudKind::RateAbuse => 10,
    }
}

/// Reputation floor: below this the halt-filter in `peers()` withholds the
/// entry from peer lists (though it can be restored by behavior).
pub const HALT_SCORE_THRESHOLD: i32 = -20;

fn score_of(e: &Entry) -> i32 {
    e.score
}

#[allow(dead_code)]
fn entry_default_fields(ad: PeerAd) -> Entry {
    Entry {
        ad,
        last_refresh_ms: now_ms(),
        dead_votes: 0,
        score: 0,
        fraud_reports: HashMap::new(),
    }
}

fn router(state: Arc<Registry>, ttl_ms: u64) -> Router {
    router_with_limiters(
        state,
        ttl_ms,
        default_anon_limiter(),
        Arc::new(crate::ratelimit::RateLimiter::new(
            AUTH_READ_LIMIT_PER_MIN,
            Duration::from_secs(60),
        )),
    )
}

fn default_anon_limiter() -> Arc<crate::ratelimit::RateLimiter> {
    Arc::new(crate::ratelimit::RateLimiter::new(
        ANON_READ_LIMIT_PER_MIN,
        Duration::from_secs(60),
    ))
}

/// Per-identity allowance for signed peers-fetches: one legit node on any
/// IP costs the relay 12 req/min (fetch + advertise on the 10 s tick);
/// NAT deployments pack many nodes behind one IP, and the anonymous
/// per-IP floor (240/min) would cap them at ~20 nodes. A signed fetch
/// pays the fingerprint's OWN bucket instead — 120× its legit rate — so
/// any number of nodes scales while keeping anonymous (unsigned) floods
/// locked to the tight floor. Signature verification is ~60µs per hit.
const AUTH_READ_LIMIT_PER_MIN: u32 = 1_440;

#[derive(Clone)]
pub struct BootstrapLimiters {
    pub anon: Arc<crate::ratelimit::RateLimiter>,
    pub identity: Arc<crate::ratelimit::RateLimiter>,
}

/// Two-lane guard: a `/v1/peers` fetch carrying wire-auth headers that
/// VERIFY against `peers-req1|{stamp}` pays the identity bucket; anything
/// else (no headers, bad signature, other routes) pays its IP bucket.
pub async fn limiters_guard(
    State(l): State<BootstrapLimiters>,
    req: Request<Body>,
    next: Next,
) -> Response {
    let mut key = crate::ratelimit::client_ip(&req).map(crate::ratelimit::ip_key);
    let mut lane = l.anon.clone();
    let path = req.uri().path();
    let signed = ["x-jerith-pubkey", "x-jerith-stamp", "x-jerith-sig"]
        .iter()
        .all(|h| req.headers().get(*h).is_some());
    if path == "/v1/peers" && signed {
        let hdr = |name: &str| {
            req.headers()
                .get(name)
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string())
                .unwrap_or_default()
        };
        let auth = crate::signing::WireAuth {
            pubkey: hdr("x-jerith-pubkey"),
            stamp_ms: hdr("x-jerith-stamp").parse().unwrap_or(0),
            sig: hdr("x-jerith-sig"),
        };
        let payload = crate::signing::peers_fetch_payload(auth.stamp_ms);
        // expected_node_id = the presented pubkey's fingerprint: the
        // signature must be BY the key that sponsors the bucket.
        let expected = auth.pubkey[..16.min(auth.pubkey.len())].to_string();
        if auth.validate(payload.as_bytes(), &expected).is_ok() && !auth.pubkey.is_empty() {
            key = Some(crate::ratelimit::identity_key(&auth.pubkey));
            lane = l.identity.clone();
        }
    }
    crate::ratelimit::enforce(&lane.clone(), key, req, next).await
}

/// Split-out so tests can mount a tiny-budget limiter and reach 429s.
fn router_with_limiters(
    state: Arc<Registry>,
    ttl_ms: u64,
    anon: Arc<crate::ratelimit::RateLimiter>,
    identity: Arc<crate::ratelimit::RateLimiter>,
) -> Router {
    Router::new()
        .route("/health", get(bootstrap_health))
        .route("/v1/peers", get(peers))
        .route("/v1/advertise", put(advertise))
        .route("/v1/report-dead", post(report_dead))
        .route("/v1/report-fraud", post(report_fraud))
        .route("/v1/report-measure", post(report_measure))
        .route("/v1/leave", post(leave))
        .route("/v1/models/counts", get(model_counts))
        .with_state((state, ttl_ms))
        .layer(axum::middleware::from_fn_with_state(
            BootstrapLimiters { anon, identity },
            limiters_guard,
        ))
}

type Shared = (Arc<Registry>, u64);

async fn bootstrap_health() -> Json<serde_json::Value> {
    // `security` carries this process's reject counters (429s from the
    // limiter + oversize ad/registry refusals). Additive-only fields:
    // existing health consumers may ignore unknown keys.
    Json(serde_json::json!({
        "ok": true,
        "service": "jerith-bootstrap",
        "security": serde_json::to_value(crate::metrics::security_snapshot())
            .expect("security counters serialize"),
    }))
}

/// attitude decay: +2 reputation per full day since the last report, capped at 0.
fn apply_recovery(entry: &mut Entry, now_ms_val: u64) {
    if entry.fraud_reports.is_empty() {
        entry.score = entry.score.min(0);
        return;
    }
    let last_report = entry
        .fraud_reports
        .values()
        .map(|(_, ts)| *ts)
        .max()
        .unwrap_or(now_ms_val);
    let days = (now_ms_val.saturating_sub(last_report) / 86_400_000) as i32;
    let recovered = (entry.score + 2 * days).min(0);
    entry.score = recovered.max(REPUTATION_FLOOR);
}

/// Remove expired entries; returns a snapshot of live, recovery-applied ones.
async fn sweep_live_scored(registry: &Registry, ttl_ms: u64) -> Vec<Entry> {
    let now = now_ms();
    let mut entries = registry.entries.write().await;
    entries.retain(|_, e| now.saturating_sub(e.last_refresh_ms) < ttl_ms);
    let live: Vec<Entry> = entries
        .values_mut()
        .map(|e| {
            apply_recovery(e, now);
            e.clone()
        })
        .collect();
    live
}

/// One background-sweep pass: drop entries stale beyond `ttl_ms`.
/// Returns how many rows were removed (for the sweeper's log line).
async fn sweep_expired(registry: &Registry, ttl_ms: u64) -> usize {
    let now = now_ms();
    let mut w = registry.entries.write().await;
    let before = w.len();
    w.retain(|_, e| now.saturating_sub(e.last_refresh_ms) < ttl_ms);
    before - w.len()
}

/// Remove expired entries; returns a snapshot of live ones.
async fn sweep_live(registry: &Registry, ttl_ms: u64) -> Vec<Entry> {
    let now = now_ms();
    let mut entries = registry.entries.write().await;
    entries.retain(|_, e| now.saturating_sub(e.last_refresh_ms) < ttl_ms);
    entries.values().cloned().collect()
}

/// §10: the bootstrap's signing identity. Persisted hex secret in the
/// shared data dir (0600) so the key — and with it every receiver's pin —
/// survives restarts; generated on first boot. Used to sign the additive
/// `/v1/peers` reputation stamps; unloaded until `serve()` runs, so
/// library use of the registry (tests, in-process spawns) carries no key
/// obligations and degrades to a stampless list.
static REPUTATION_ID: OnceLock<crate::signing::SigningIdentity> = OnceLock::new();

fn init_reputation_identity() -> Option<crate::signing::SigningIdentity> {
    let dir = crate::config::NodeConfig::data_dir();
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("bootstrap-reputation.hex");
    let existing = std::fs::read_to_string(&path).ok();
    if let Some(hexstr) = existing {
        let bytes = hex::decode(hexstr.trim()).ok()?;
        crate::signing::SigningIdentity::from_secret_bytes(&bytes).ok()
    } else {
        let id = crate::signing::SigningIdentity::generate();
        let wrote = std::fs::write(&path, hex::encode(id.secret_bytes()));
        if wrote.is_ok() {
            crate::config::NodeConfig::restrict_file_permissions(&path);
        }
        Some(id)
    }
}

async fn peers(
    Query(params): Query<HashMap<String, String>>,
    State((registry, ttl_ms)): State<Shared>,
) -> Response {
    let want_model = params.get("model").cloned().unwrap_or_default();
    let live = sweep_live_scored(&registry, ttl_ms).await;
    // Halt-filtered: bins below HALT_SCORE_THRESHOLD are withheld — a
    // persistent non-contributor stops receiving mesh traffic but can come
    // back via adverts once its behavior corrects.
    // Restructured once (§10 + §2 additive fields build together): the
    // filtered live entries are materialized BEFORE the owning iterator
    // runs so both the ad list and the reputation stamps come from one
    // consistent snapshot.
    let filtered: Vec<&Entry> = live
        .iter()
        .filter(|e| want_model.is_empty() || e.ad.model_id == want_model)
        .filter(|e| score_of(e) > HALT_SCORE_THRESHOLD)
        .collect();
    let peers: Vec<PeerAd> = filtered.iter().map(|e| e.ad.clone()).collect();
    // §2 contribution quorum (additive wire field): the aggregate per peer
    // (distinct corroborating reporters / spread / consistency) — statistics
    // for operators and bots, never a verdict.
    let measurements = registry.measures.read().await.shared_json();
    // §10 signed reputation stamps (additive wire field): the bootstrap
    // signs each returned peer's CURRENT score so the figures can travel
    // node-to-node inside ads (receivers verify against the pin they made
    // from `bootstrap` here). Score range [REPUTATION_FLOOR, 0]:
    // fraud-decremented, time-recovered; 0 = neutral.
    let bootstrap = REPUTATION_ID
        .get()
        .map(|id| BootstrapKeyInfo {
            node_id: id.node_id(),
            pubkey: id.pubkey_hex(),
        });
    let reputations = match REPUTATION_ID.get() {
        Some(id) => {
            let mut m = std::collections::HashMap::new();
            for e in &filtered {
                let stamp = now_ms();
                let payload = crate::signing::reputation_payload(
                    &id.node_id(),
                    &e.ad.node_id,
                    e.score,
                    stamp,
                );
                m.insert(
                    e.ad.node_id.clone(),
                    crate::bootstrap::StampBody {
                        score: e.score,
                        stamp_ms: stamp,
                        sig: id.sign(payload.as_bytes()),
                    },
                );
            }
            m
        }
        None => Default::default(),
    };
    (
        StatusCode::OK,
        Json(PeerListResponse {
            peers,
            measurements,
            bootstrap,
            reputations,
        }),
    )
        .into_response()
}

async fn advertise(
    State((registry, ttl_ms)): State<Shared>,
    headers: axum::http::HeaderMap,
    ad: Json<PeerAd>,
) -> Response {
    let mut ad = ad.0;
    if ad.node_id.is_empty() || ad.model_id.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "node_id and model_id are required"})),
        )
            .into_response();
    }
    if let Err(resp) = validate_ad_fields(&ad) {
        return *resp;
    }

    // --- Wire auth: Ed25519 signature over the canonical payload ---
    let (pubkey, stamp, sig) = match (
        headers.get("x-jerith-pubkey").and_then(|v| v.to_str().ok()),
        headers.get("x-jerith-stamp").and_then(|v| v.to_str().ok()),
        headers.get("x-jerith-sig").and_then(|v| v.to_str().ok()),
    ) {
        (Some(p), Some(s), Some(g)) => (p.to_string(), s.to_string(), g.to_string()),
        _ => {
            return (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"error": "missing wire-auth headers"})),
            )
                .into_response();
        }
    };
    let Ok(stamp_ms) = stamp.parse::<u64>() else {
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({"error": "bad stamp"})),
        )
            .into_response();
    };
    let auth = crate::signing::WireAuth {
        pubkey,
        stamp_ms,
        sig,
    };
    // The client sends last_seen_ms in the payload; the server re-pins it
    // for expiry after verification.
    let client_seen = ad.last_seen_ms;
    let payload = crate::signing::advertise_payload(&ad, stamp_ms);
    if let Err(e) = auth.validate(payload.as_bytes(), &ad.node_id) {
        // Transitional compat: accept a signature over the v1 payload
        // (canonical shape before rpc_address was added) so a rolling
        // upgrade of mixed-version nodes doesn't 401-storm. Deprecation
        // window documented in docs/UPGRADE.md.
        let legacy_ad = crate::signing::legacy_ad_excerpt(&ad, stamp_ms);
        if auth.validate(legacy_ad.as_bytes(), &ad.node_id).is_ok() {
            tracing::info!(
                node = %ad.node_id,
                "accepted legacy-format advertisement — upgrade recommended (see docs/UPGRADE.md)"
            );
        } else {
            tracing::warn!(node = %ad.node_id, error = ?e, "rejected forged/stale advertisement");
            return (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"error": format!("wire auth failed: {e:?}")})),
            )
                .into_response();
        }
    }
    // Signature valid: never trust the client's clock for expiry.
    ad.last_seen_ms = client_seen.max(now_ms().saturating_sub(1_000));

    let mut entries = registry.entries.write().await;
    // Registry capacity: keypair generation is free, so an uncapped entry
    // map lets one client pin memory and inflate /v1/peers. Existing ids
    // always refresh; only NEW identities are refused once full.
    if !entries.contains_key(&ad.node_id) && entries.len() >= registry.max_entries {
        crate::metrics::bump(&crate::metrics::OVERSIZE_BOOTSTRAP_REJECTS);
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "registry at capacity — refreshes only"})),
        )
            .into_response();
    }
    // Preserve score/history across re-advertise (identity is stable).
    let entry = entries
        .remove(&ad.node_id)
        .unwrap_or_else(|| entry_default_fields(ad.clone()));
    entries.insert(
        ad.node_id.clone(),
        Entry {
            last_refresh_ms: now_ms(),
            ..entry
        }
        .with_ad(ad),
    );
    let _ = ttl_ms;
    (StatusCode::OK, Json(serde_json::json!({"ok": true}))).into_response()
}

async fn report_dead(
    State((registry, _ttl_ms)): State<Shared>,
    headers: axum::http::HeaderMap,
    Json(report): Json<DeadReport>,
) -> Response {
    // Wire auth: the reporter signs `jerith-dead1|stamp|reporter|target`
    // and the bootstrap checks identity/freshness/signature before the
    // vote counts. Unsigned reports are refused (fail-closed) — otherwise
    // anyone on the internet could evict any live peer with a handful of
    // identical POSTs (DEAD_VOTES_TO_EVICTION-threshold doS).
    let stamp = match headers
        .get("x-jerith-stamp")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
    {
        Some(s) => s,
        None => return unauthorized("missing or malformed stamp"),
    };
    if let Err(resp) = verify_report_auth(
        &headers,
        &report.reporter,
        &crate::signing::dead_payload(&report.reporter, &report.node_id, stamp),
    ) {
        return *resp;
    }
    let mut entries = registry.entries.write().await;
    match entries.get_mut(&report.node_id) {
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "unknown node"})),
        )
            .into_response(),
        Some(entry) => {
            entry.dead_votes += 1;
            if entry.dead_votes >= DEAD_VOTES_TO_EVICTION {
                entries.remove(&report.node_id);
                (StatusCode::OK, Json(serde_json::json!({"evicted": true}))).into_response()
            } else {
                (
                    StatusCode::OK,
                    Json(serde_json::json!({"evicted": false, "dead_votes": entry.dead_votes})),
                )
                    .into_response()
            }
        }
    }
}

impl Entry {
    /// Replace the advertised payload while keeping score/history.
    fn with_ad(mut self, ad: PeerAd) -> Self {
        self.ad = ad;
        self
    }
}

/// Shared verification for the report endpoints: pull the wire-auth
/// triple from headers, rebuild the caller-supplied canonical payload,
/// and validate identity + stamp + signature against the REPORTER's
/// node id (the report targets a third party — attribution is what
/// makes dedupe/corroboration meaningful). Errors surface as a mounted
/// 401 response.
fn verify_report_auth(
    headers: &axum::http::HeaderMap,
    reporter_node_id: &str,
    expected_payload: &str,
) -> Result<(), Box<Response>> {
    let (pubkey, stamp, sig) = match (
        headers.get("x-jerith-pubkey").and_then(|v| v.to_str().ok()),
        headers.get("x-jerith-stamp").and_then(|v| v.to_str().ok()),
        headers.get("x-jerith-sig").and_then(|v| v.to_str().ok()),
    ) {
        (Some(p), Some(s), Some(g)) => (p.to_string(), s.to_string(), g.to_string()),
        _ => return Err(unauthorized("missing wire-auth headers").into()),
    };
    let stamp_ms = match stamp.parse::<u64>() {
        Ok(s) => s,
        Err(_) => return Err(unauthorized("bad stamp").into()),
    };
    let auth = crate::signing::WireAuth {
        pubkey,
        stamp_ms,
        sig,
    };
    if let Err(e) = auth.validate(expected_payload.as_bytes(), reporter_node_id) {
        tracing::warn!(reporter = %reporter_node_id, error = ?e, "rejected unsigned/forged report");
        return Err(unauthorized(&format!("wire auth failed: {e:?}")).into());
    }
    Ok(())
}

fn unauthorized(msg: &str) -> Response {
    (
        StatusCode::UNAUTHORIZED,
        Json(serde_json::json!({"error": msg})),
    )
        .into_response()
}

/// Peer-submitted fraud evidence. Deduplicated per (reporter, target);
/// corroborated by ≥2 distinct reporters counting double.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FraudReport {
    /// Node being accused.
    pub node_id: String,
    /// Node filing the report (must not equal node_id).
    pub reporter: String,
    /// Observed failure kind.
    pub kind: FraudKind,
}

/// §2 contribution quorum: a pinger forwards a peer's SIGNED measurement
/// block, attesting to the exact figures it verified. The subject's own
/// signature anchors the numbers (a biased pinger cannot change them); the
/// reporter's attestation binds WHO corroborated the block. Accept requires
/// BOTH signatures to hold server-side. Statistics, not verdicts: the tally
/// is corroborated evidence for the reputation/stake layers, never an
/// auto-slash (no-statistics-slash boundary kept).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasureObservation {
    pub reporter: String,
    pub subject: String,
    /// The subject's wire pubkey — its fingerprint must equal `subject`
    /// and it must verify the block's own signature.
    pub subject_pubkey: String,
    /// The subject's own signature over its measurement block (travels in
    /// the body: the REPORTER authenticates via headers, the subject via
    /// this field).
    pub subject_sig: String,
    pub block_stamp_ms: u64,
    pub window_secs: u64,
    pub tokens_served: u64,
    pub tokens_consumed: u64,
    pub declared_percent: u64,
    pub measured_percent: u64,
}

/// Forwarded blocks may lag their health reply by a ping cycle or two.
const MEASURE_BLOCK_MAX_AGE_MS: u64 = 300_000;

async fn report_measure(
    State((registry, _ttl_ms)): State<Shared>,
    headers: axum::http::HeaderMap,
    Json(obs): Json<MeasureObservation>,
) -> Response {
    // Sanity caps (registry discipline): identities well-formed, figures
    // impossible to fake into overflow/div-zero paths downstream.
    if obs.subject.is_empty()
        || obs.reporter.is_empty()
        || obs.subject == obs.reporter
        || obs.subject_pubkey.len() != 64
        || obs.subject_pubkey[..16] != obs.subject
        || obs.subject_sig.is_empty()
        || obs.block_stamp_ms == 0
        || obs.window_secs == 0
        || obs.window_secs > 3_600
        || obs.tokens_served > 10_000_000_000_000
        || obs.tokens_consumed > 10_000_000_000_000
        || obs.declared_percent > 1_000
        || obs.measured_percent > 1_000
        || now_ms().saturating_sub(obs.block_stamp_ms) > MEASURE_BLOCK_MAX_AGE_MS
    {
        crate::metrics::bump(&crate::metrics::OVERSIZE_BOOTSTRAP_REJECTS);
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "malformed measurement observation"})),
        )
            .into_response();
    }
    // Subject's own signature anchors the figures (fingerprint already
    // bound above; stamp freshness checked by the cap above).
    let subject_payload = crate::signing::measurement_payload(
        &obs.subject,
        obs.block_stamp_ms,
        obs.window_secs,
        obs.tokens_served,
        obs.tokens_consumed,
        obs.declared_percent,
        obs.measured_percent,
    );
    if !crate::signing::verify(&obs.subject_pubkey, subject_payload.as_bytes(), &obs.subject_sig) {
        tracing::warn!(subject = %obs.subject, "tampered/forged subject block refused");
        return unauthorized("subject block signature invalid");
    }
    // Reporter attestation over the EXACT block it claims to have seen.
    let reporter_stamp = match headers
        .get("x-jerith-stamp")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
    {
        Some(s) => s,
        None => return unauthorized("missing or malformed stamp"),
    };
    let reporter_payload = crate::signing::measure_attest_payload(
        &obs.reporter,
        &obs.subject,
        obs.block_stamp_ms,
        obs.window_secs,
        obs.tokens_served,
        obs.tokens_consumed,
        obs.declared_percent,
        obs.measured_percent,
        reporter_stamp,
    );
    if let Err(resp) = verify_report_auth(&headers, &obs.reporter, &reporter_payload) {
        return *resp;
    }
    let (depth, fresh) = {
        let mut tally = registry.measures.write().await;
        tally.record(
            &obs.subject,
            MeasureRecord {
                reporter: obs.reporter.clone(),
                block_stamp_ms: obs.block_stamp_ms,
                window_secs: obs.window_secs,
                measured_percent: obs.measured_percent,
            },
        )
    };
    (StatusCode::OK, Json(serde_json::json!({"tallied": fresh, "ring": depth}))).into_response()
}

/// Graceful self-retraction (§2 model-switch machinery): a node leaves
/// the mesh IMMEDIATELY instead of waiting its ad TTL out (90 s of a peer
/// that no longer serves the mesh's model). Signed fail-closed like every
/// other registry write: the leaver must BE the identity of the entry —
/// node_id ↔ pubkey is bound the same way as advertise — so nobody can
/// scrape /v1/peers and undeclare somebody else (a doS on the mesh).
/// Idempotent: leaving twice answers `left:false`, never an error.
async fn leave(
    State((registry, _ttl_ms)): State<Shared>,
    headers: axum::http::HeaderMap,
    Json(body): Json<LeaveNotice>,
) -> Response {
    if body.node_id.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "node_id required"})),
        )
            .into_response();
    }
    let stamp = match headers
        .get("x-jerith-stamp")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
    {
        Some(s) => s,
        None => return unauthorized("missing or malformed stamp"),
    };
    if let Err(resp) = verify_report_auth(
        &headers,
        &body.node_id,
        &crate::signing::leave_payload(&body.node_id, stamp),
    ) {
        return *resp;
    }
    let mut entries = registry.entries.write().await;
    let existed = entries.remove(&body.node_id).is_some();
    tracing::info!(
        node = %body.node_id,
        existed,
        "graceful mesh leave applied (model switch or clean shutdown)"
    );
    (StatusCode::OK, Json(serde_json::json!({"left": existed}))).into_response()
}

/// What `/v1/leave` carries: only the leaver's own identity (the
/// signature does the trusting).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaveNotice {
    pub node_id: String,
}

async fn report_fraud(
    State((registry, _ttl_ms)): State<Shared>,
    headers: axum::http::HeaderMap,
    Json(rep): Json<FraudReport>,
) -> Response {
    if rep.node_id == rep.reporter || rep.reporter.is_empty() || rep.node_id.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "reporter must differ from node_id"})),
        )
            .into_response();
    }
    if !matches!(
        rep.kind,
        FraudKind::FailedWorker
            | FraudKind::Leeching
            | FraudKind::DishonestResults
            | FraudKind::RateAbuse
    ) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "unknown fraud kind"})),
        )
            .into_response();
    }
    // Wire auth: the client already signs `jerith-fraud1|{stamp}|{kind}|{reporter}|{target}`
    // (bootstrap.rs report_fraud) — the server now actually verifies it.
    // Without this check anyone on the internet could anonymous-slash any
    // peer's reputation (each distinct reporter name costs the target a
    // full penalty, and attacker-chosen names are free).
    let stamp = match headers
        .get("x-jerith-stamp")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
    {
        Some(s) => s,
        None => return unauthorized("missing or malformed stamp"),
    };
    let payload = format!(
        "jerith-fraud1|{stamp}|{}|{}|{}",
        match rep.kind {
            FraudKind::FailedWorker => "failed_worker",
            FraudKind::Leeching => "leeching",
            FraudKind::DishonestResults => "dishonest_results",
            FraudKind::RateAbuse => "rate_abuse",
        },
        rep.reporter,
        rep.node_id,
    );
    if let Err(resp) = verify_report_auth(&headers, &rep.reporter, &payload) {
        return *resp;
    }
    let mut entries = registry.entries.write().await;
    let Some(entry) = entries.get_mut(&rep.node_id) else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "unknown node"})),
        )
            .into_response();
    };

    apply_recovery(entry, now_ms());
    let now = now_ms();

    // Was this kind already reported by someone else before this report?
    let kind_already_seen = entry.fraud_reports.values().any(|(k, _)| k == &rep.kind);
    // Count of distinct reporters who already reported this kind.
    let prior_kind_reporters = entry
        .fraud_reports
        .values()
        .filter(|(k, _)| k == &rep.kind)
        .count() as u32;

    let already_reported_by_this_reporter = entry.fraud_reports.contains_key(&rep.reporter);

    if already_reported_by_this_reporter {
        // First claim by a reporter wins; later reports only refresh the
        // timestamp (cheap report-flood defense). No penalty change.
        if let Some(record) = entry.fraud_reports.get_mut(&rep.reporter) {
            record.1 = now;
        }
    } else if !kind_already_seen {
        // New (reporter, kind) pair, kind not seen before: single penalty.
        entry
            .fraud_reports
            .insert(rep.reporter.clone(), (rep.kind.clone(), now));
        entry.score = (entry.score - penalty_for(&rep.kind)).max(REPUTATION_FLOOR);
    } else {
        // Corroboration: this kind was already reported by someone else.
        // Penalize each additional distinct reporter (≥MIN_CORROBORATORS).
        entry
            .fraud_reports
            .insert(rep.reporter.clone(), (rep.kind.clone(), now));
        if prior_kind_reporters + 1 >= MIN_CORROBORATORS {
            entry.score = (entry.score - penalty_for(&rep.kind)).max(REPUTATION_FLOOR);
        }
    }
    let score = entry.score;
    drop(entries);
    tracing::warn!(
        node = %rep.node_id,
        reporter = %rep.reporter,
        kind = ?rep.kind,
        score,
        "fraud report recorded"
    );
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "recorded": true,
            "score": score,
            "halted": score <= HALT_SCORE_THRESHOLD,
        })),
    )
        .into_response()
}

async fn model_counts(State((registry, ttl_ms)): State<Shared>) -> Response {
    let live = sweep_live(&registry, ttl_ms).await;
    // Per-model count plus the per-quant breakdown — the mesh-compat
    // check (FEATURES §12) reads this to tell an operator "same model,
    // wrong quant — N nodes run quants [...]".
    let mut counts: HashMap<String, u64> = HashMap::new();
    let mut quants: HashMap<String, HashMap<String, u64>> = HashMap::new();
    for e in live {
        *counts.entry(e.ad.model_id.clone()).or_default() += 1;
        *quants
            .entry(e.ad.model_id.clone())
            .or_default()
            .entry(e.ad.quant.clone())
            .or_default() += 1;
    }
    let mut out: Vec<serde_json::Value> = counts
        .into_iter()
        .map(|(model_id, count)| {
            let mut qs: Vec<(String, u64)> = quants
                .remove(&model_id)
                .unwrap_or_default()
                .into_iter()
                .collect();
            qs.sort();
            let quant_rows: Vec<serde_json::Value> = qs
                .into_iter()
                .map(|(q, c)| serde_json::json!({"quant": q, "count": c}))
                .collect();
            serde_json::json!({
                "model_id": model_id,
                "count": count,
                "quants": quant_rows,
            })
        })
        .collect();
    out.sort_by_key(|v| {
        v.get("model_id")
            .and_then(|m| m.as_str())
            .unwrap_or_default()
            .to_string()
    });
    (StatusCode::OK, Json(out)).into_response()
}

/// Serve the bootstrap node until the process exits.
pub async fn serve(options: BootstrapOptions) -> anyhow::Result<()> {
    let registry = Arc::new(Registry::default());
    let app = router(registry.clone(), options.advertise_ttl_ms);
    // §10: load-or-generate the reputation signing identity FIRST — a
    // failure downgrades to stamps-off (registry still runs) rather than
    // taking the discovery node down.
    match init_reputation_identity() {
        Some(id) => {
            let _ = REPUTATION_ID.set(id);
        }
        None => tracing::warn!(
            "reputation signing identity unavailable — /v1/peers runs without score stamps"
        ),
    }
    // Background TTL sweeper: expiry otherwise only happens while a read
    // endpoint is polled, so an idle bootstrap would retain every stale
    // advertisement forever. Half-TTL cadence, floored at 15s.
    let sweeper = registry;
    let ttl_ms = options.advertise_ttl_ms;
    tokio::spawn(async move {
        let interval = Duration::from_millis((ttl_ms / 2).clamp(15_000, 30_000));
        loop {
            tokio::time::sleep(interval).await;
            let dropped = sweep_expired(&sweeper, ttl_ms).await;
            if dropped > 0 {
                tracing::debug!(dropped, "registry sweep dropped expired ads");
            }
        }
    });
    let addr = SocketAddr::from(([0, 0, 0, 0], options.port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(port = options.port, "bootstrap node listening");
    // Peer connect info is required for the per-IP rate limiter's fallback
    // attribution (headers take precedence when present, e.g. behind the
    // tunnel).
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;
    Ok(())
}

/// Spawn the bootstrap node in-process (used by tests and by `jerith run`
/// when configured to co-host a bootstrap).
pub async fn spawn(options: BootstrapOptions) -> anyhow::Result<()> {
    tokio::spawn(async move {
        if let Err(e) = serve(options).await {
            tracing::error!(error = %e, "bootstrap server died");
        }
    });
    Ok(())
}

/// Unused-parameter silencer for Path (kept for future /v1/block/{height}).
#[allow(dead_code)]
async fn _path_used(_: Path<String>) {}

// ============================================================================
// Tests (in-process axum server via tokio)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::NodeRole;
    use crate::models::{ModelId, ModelSource};
    use std::time::Duration;

    /// A peer ad backed by a real signing identity (node_id = fingerprint).
    fn make_peer(model: &str) -> (crate::signing::SigningIdentity, PeerAd) {
        let id = crate::signing::SigningIdentity::generate();
        let a = PeerAd {
            node_id: id.node_id(),
            address: "127.0.0.1:9000".into(),
            model_id: model.into(),
            quant: "int4".into(),
            contribution_percent: 30,
            rpc_address: String::new(),
            last_seen_ms: now_ms(),
            role: NodeRole::Hybrid,
            resources: None,
            hosted_experts: None,
            hosted_expert_count: None,
            discovery_servers: Vec::new(),
            gossip_reputations: None,
        };
        (id, a)
    }

    fn model(id: &str) -> ModelId {
        ModelId {
            id: id.into(),
            quant: "int4".into(),
            source: ModelSource::Curated,
        }
    }

    #[tokio::test]
    async fn advertise_then_list_and_counts() {
        let registry = Arc::new(Registry::default());
        let app = router(Arc::clone(&registry), DEFAULT_ADVERTISE_TTL_MS);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let base = format!("http://{addr}");
        let client = crate::http::mesh_http();

        // Empty at first.
        let counts: Vec<serde_json::Value> = client
            .get(format!("{base}/v1/models/counts"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert!(counts.is_empty());

        // Two glm-5.3 nodes, one qwen node — all signed by real identities.
        for model in ["glm-5.3", "glm-5.3", "qwen3.8-flash-next"] {
            let (id, a) = make_peer(model);
            let r = client
                .put(format!("{base}/v1/advertise"))
                .headers(signed_headers(&id, &a))
                .json(&a)
                .send()
                .await
                .unwrap();
            assert!(r.status().is_success(), "signed ad must be accepted");
        }

        // Peer list filtered by model: only same-model peers return.
        let list: PeerListResponse = client
            .get(format!("{base}/v1/peers?model=glm-5.3"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(list.peers.len(), 2);
        assert!(list.peers.iter().all(|p| p.model_id == "glm-5.3"));

        // Counts aggregate per model.
        let counts: Vec<serde_json::Value> = client
            .get(format!("{base}/v1/models/counts"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(counts.len(), 2);
        let glm = counts.iter().find(|c| c["model_id"] == "glm-5.3").unwrap();
        assert_eq!(glm["count"], 2);
        assert_eq!(glm["quants"][0]["quant"], "int4");
        assert_eq!(glm["quants"][0]["count"], 2);
    }

    /// The compat census the mesh-compat check consumes: quant counts MIX
    /// (two nodes int4, one bogus-default quant "fp8") and the row shape
    /// `BootstrapClient::fetch_model_census` parses.
    #[tokio::test]
    async fn model_counts_split_by_quant() {
        let registry = Arc::new(Registry::default());
        let app = router(Arc::clone(&registry), DEFAULT_ADVERTISE_TTL_MS);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let base = format!("http://{addr}");
        let client = crate::http::mesh_http();
        for (model, quant) in [
            ("glm-5.3", "int4"),
            ("glm-5.3", "int4"),
            ("glm-5.3", "fp8"),
        ] {
            let (mut id, mut a) = make_peer(model);
            a.quant = quant.into();
            let r = client
                .put(format!("{base}/v1/advertise"))
                .headers(signed_headers(&id, &a))
                .json(&a)
                .send()
                .await
                .unwrap();
            assert!(r.status().is_success());
            let _ = &mut id;
        }
        // Client-side parse: the typed census row carries both quants.
        let rows = crate::bootstrap::BootstrapClient::new(base)
            .fetch_model_census()
            .await
            .unwrap();
        let glm = rows.iter().find(|r| r.model_id == "glm-5.3").unwrap();
        assert_eq!(glm.count, 3);
        assert_eq!(glm.quants.get("int4"), Some(&2));
        assert_eq!(glm.quants.get("fp8"), Some(&1));
    }

    #[tokio::test]
    async fn report_dead_evicts_after_threshold() {
        let registry = Arc::new(Registry::default());
        let app = router(Arc::clone(&registry), DEFAULT_ADVERTISE_TTL_MS);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let base = format!("http://{addr}");
        let client = crate::http::mesh_http();

        let (id, a) = make_peer("glm-5.3");
        client
            .put(format!("{base}/v1/advertise"))
            .headers(signed_headers(&id, &a))
            .json(&a)
            .send()
            .await
            .unwrap();

        // First vote (signed by a different node's identity): not evicted.
        let (r_id, _ra) = make_peer("glm-5.3");
        let _ = _ra; // reporter identity comes from make_peer; r_id signs
        let reporter = r_id.node_id();
        let r: serde_json::Value = client
            .post(format!("{base}/v1/report-dead"))
            .headers(dead_report_headers(&r_id, &reporter, &a.node_id))
            .json(&serde_json::json!({"node_id": a.node_id, "reporter": reporter}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(r["evicted"], false, "first signed vote must not evict");

        // Second vote: evicted.
        let r: serde_json::Value = client
            .post(format!("{base}/v1/report-dead"))
            .headers(dead_report_headers(&r_id, &reporter, &a.node_id))
            .json(&serde_json::json!({"node_id": a.node_id, "reporter": reporter}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(r["evicted"], true);

        let list: PeerListResponse = client
            .get(format!("{base}/v1/peers?model=glm-5.3"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert!(list.peers.is_empty());
    }

    #[tokio::test]
    async fn ttl_expiry_drops_stale_entries() {
        let registry = Arc::new(Registry::default());
        let app = router(Arc::clone(&registry), 50); // 50 ms TTL
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let base = format!("http://{addr}");
        let client = crate::http::mesh_http();

        let (id, a) = make_peer("glm-5.3");
        client
            .put(format!("{base}/v1/advertise"))
            .headers(signed_headers(&id, &a))
            .json(&a)
            .send()
            .await
            .unwrap();
        let list: PeerListResponse = client
            .get(format!("{base}/v1/peers?model=glm-5.3"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(list.peers.len(), 1);

        tokio::time::sleep(Duration::from_millis(80)).await;
        let list: PeerListResponse = client
            .get(format!("{base}/v1/peers?model=glm-5.3"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert!(
            list.peers.is_empty(),
            "expired entry must drop from the list"
        );
    }

    /// Regression test for the ms-boundary stamp race: rapid freshly-signed
    /// advertises must ALL succeed. The pre-fix pattern (payload built with
    /// one now_ms(), auth stamped with a second now_ms()) rejected ~1-in-many
    /// requests whenever two clock reads straddled a millisecond, which
    /// surfaced as the flaky reputation test under parallel load.
    #[tokio::test]
    async fn advertise_survives_rapid_freshly_signed_requests() {
        let registry = Arc::new(Registry::default());
        let app = router(Arc::clone(&registry), DEFAULT_ADVERTISE_TTL_MS);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let base = format!("http://{addr}");
        let client = crate::http::mesh_http();
        let id = crate::signing::SigningIdentity::generate();

        const ROUNDS: usize = 50;
        for round in 0..ROUNDS {
            let a = PeerAd {
                node_id: id.node_id(),
                address: "127.0.0.1:9000".into(),
                model_id: "glm-5.3".into(),
                quant: "int4".into(),
                contribution_percent: 30,
                rpc_address: String::new(),
                last_seen_ms: now_ms(),
                role: NodeRole::Hybrid,
                resources: None,
                hosted_experts: None,
                hosted_expert_count: None,
                discovery_servers: Vec::new(),
                gossip_reputations: None,
            };
            let r = client
                .put(format!("{base}/v1/advertise"))
                .headers(signed_headers(&id, &a))
                .json(&a)
                .send()
                .await
                .unwrap();
            assert!(
                r.status().is_success(),
                "round {round}/{ROUNDS}: freshly-signed advertise was rejected ({}) — stamp race regression",
                r.status()
            );
        }
    }

    /// §2 model-switch machinery: a node may retract ITS OWN ad instantly
    /// (signed, fail-closed, idempotent) — and nobody else's.
    #[tokio::test]
    async fn leave_retracts_own_entry_idempotent_and_unforgeable() {
        let registry = Arc::new(Registry::default());
        let app = router(Arc::clone(&registry), DEFAULT_ADVERTISE_TTL_MS);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let client = crate::http::mesh_http();
        let ad_url = format!("http://{addr}/v1/advertise");
        let url = format!("http://{addr}/v1/leave");

        // Headers for one leave, with the stamped payload and the header
        // stamp frozen to the SAME clock read (avoids the 1 ms cross-tick
        // flake the two-clock pattern has under load).
        let leave_headers = |id: &crate::signing::SigningIdentity,
                             node_id: &str| {
            let stamp = now_ms();
            let payload = crate::signing::leave_payload(node_id, stamp);
            let auth =
                crate::signing::WireAuth::with_stamp(id, stamp, payload.as_bytes());
            let mut h = axum::http::HeaderMap::new();
            h.insert("x-jerith-pubkey", auth.pubkey.parse().unwrap());
            h.insert("x-jerith-stamp", auth.stamp_ms.to_string().parse().unwrap());
            h.insert("x-jerith-sig", auth.sig.parse().unwrap());
            h
        };

        // Two real nodes on the wire (signature-bound just like prod).
        let (id, ad) = make_peer("glm-5.3");
        let (blind_owner, blind_ad) = make_peer("qwen3.8-flash-next");
        for (idn, adn) in [(&id, &ad), (&blind_owner, &blind_ad)] {
            let r = client
                .put(&ad_url)
                .headers(signed_headers(idn, adn))
                .json(adn)
                .send()
                .await
                .unwrap();
            assert!(r.status().is_success(), "seed ad accepted");
        }
        assert_eq!(registry.entries.read().await.len(), 2);

        // The identity that owns the entry leaves: accepted and gone.
        let r = client
            .post(&url)
            .headers(leave_headers(&id, &ad.node_id))
            .json(&serde_json::json!({"node_id": ad.node_id}))
            .send()
            .await
            .unwrap();
        assert!(r.status().is_success(), "own leave accepted");
        let body: serde_json::Value = r.json().await.unwrap();
        assert_eq!(body["left"], serde_json::json!(true), "the ad existed");
        assert_eq!(registry.entries.read().await.len(), 1);

        // Idempotent: a second leave answers left:false, never an error.
        let r = client
            .post(&url)
            .headers(leave_headers(&id, &ad.node_id))
            .json(&serde_json::json!({"node_id": ad.node_id}))
            .send()
            .await
            .unwrap();
        let body: serde_json::Value = r.json().await.unwrap();
        assert_eq!(body["left"], serde_json::json!(false));

        // Unforgeable: a THIRD party cannot undeclare somebody else —
        // 401 (identity binding), and the victim's entry survives.
        let attacker = crate::signing::SigningIdentity::generate();
        let r = client
            .post(&url)
            .headers(leave_headers(&attacker, &blind_ad.node_id))
            .json(&serde_json::json!({"node_id": blind_ad.node_id}))
            .send()
            .await
            .unwrap();
        assert_eq!(
            r.status(),
            axum::http::StatusCode::UNAUTHORIZED,
            "forged leave refused"
        );
        assert_eq!(registry.entries.read().await.len(), 1, "victim survives");
    }

    /// §2 contribution quorum — the tally accepts a double-signed block
    /// (subject's own sig anchors the figures, reporter attests delivery),
    /// dedupes per (reporter, block), aggregates distinct-reporter
    /// consistency into /v1/peers, and refuses tampered/forged
    /// observations. Statistics, not verdicts: one report stays unquorumed.
    #[tokio::test]
    async fn measure_quorum_tallies_aggregates_and_refuses() {
        let registry = Arc::new(Registry::default());
        let app = router(Arc::clone(&registry), DEFAULT_ADVERTISE_TTL_MS);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let client = crate::http::mesh_http();
        let url = format!("http://{addr}/v1/report-measure");
        let peers_url = format!("http://{addr}/v1/peers");

        let subject = crate::signing::SigningIdentity::generate();
        let reporter = crate::signing::SigningIdentity::generate();
        let (obs1, headers1) = make_observation(&subject, &reporter, 60, 400, 100, 25, 80);
        let r = client
            .post(&url)
            .headers(headers1)
            .json(&obs1)
            .send()
            .await
            .unwrap();
        assert!(r.status().is_success(), "double-signed observation accepted");
        let body: serde_json::Value = r.json().await.unwrap();
        assert_eq!(body["tallied"], serde_json::json!(true));

        // One report alone is UNQUORUMED — statistics, not a verdict.
        let body: serde_json::Value = client.get(&peers_url).send().await.unwrap().json().await.unwrap();
        let q = body["measurements"][&obs1.subject].as_object().expect("aggregate present");
        assert_eq!(q["distinct_reporters"], serde_json::json!(1));
        assert_eq!(q["consistent"], serde_json::Value::Null);

        // A SECOND reporter forwarding its own verified copy of the same
        // self-signed block: quorum reached, consistent.
        let reporter2 = crate::signing::SigningIdentity::generate();
        let (obs2, headers2) = make_observation(&subject, &reporter2, 60, 400, 100, 25, 80);
        // Each helper call is a fresh block (new stamp) — the quorum view
        // groups by RECENCY HORIZON (3× window), not identical stamps, so
        // two ping cycles' blocks still corroborate the same subject.
        let r = client
            .post(&url)
            .headers(headers2)
            .json(&obs2)
            .send()
            .await
            .unwrap();
        assert!(r.status().is_success());
        let body: serde_json::Value = client.get(&peers_url).send().await.unwrap().json().await.unwrap();
        let q = body["measurements"][&obs1.subject].as_object().expect("aggregate present");
        assert_eq!(q["distinct_reporters"], serde_json::json!(2));
        assert_eq!(q["consistent"], serde_json::json!(true));

        // Dedupe: the SAME reporter forwarding the SAME (reporter,
        // block_stamp) pair is tallied:false, ring unchanged. Frozen-stamp: the
        // attestation embeds the SAME clock read the header carries.
        let stamp = now_ms();
        let attest = crate::signing::measure_attest_payload(
            &obs1.reporter,
            &obs1.subject,
            obs1.block_stamp_ms,
            obs1.window_secs,
            obs1.tokens_served,
            obs1.tokens_consumed,
            obs1.declared_percent,
            obs1.measured_percent,
            stamp,
        );
        let auth = crate::signing::WireAuth::with_stamp(&reporter, stamp, attest.as_bytes());
        let mut h = axum::http::HeaderMap::new();
        h.insert("x-jerith-pubkey", auth.pubkey.parse().unwrap());
        h.insert("x-jerith-stamp", auth.stamp_ms.to_string().parse().unwrap());
        h.insert("x-jerith-sig", auth.sig.parse().unwrap());
        let r = client
            .post(&url)
            .headers(h)
            .json(&obs1)
            .send()
            .await
            .unwrap();
        assert!(r.status().is_success());
        let body: serde_json::Value = client.get(&peers_url).send().await.unwrap().json().await.unwrap();
        let q = body["measurements"][&obs1.subject].as_object().expect("aggregate present");
        assert_eq!(q["distinct_reporters"], serde_json::json!(2), "deduped reporter");

        // TAMPERED figures: flip the measured share without re-signing the
        // subject block → 401, subject's signature catches it.
        let (mut obs_t, headers_dup) = make_observation(&subject, &reporter, 60, 400, 100, 25, 55);
        obs_t.declared_percent = 25;
        obs_t.measured_percent = 99; // NOT what the subject signed
        let _ = headers_dup;
        // Sign the attestation over the FALSIFIED set (a lying collab) —
        // the subject's own sig must still refuse it.
        let stamp = now_ms();
        let falsified_attest = crate::signing::measure_attest_payload(
            &obs_t.reporter,
            &obs_t.subject,
            obs_t.block_stamp_ms,
            obs_t.window_secs,
            obs_t.tokens_served,
            obs_t.tokens_consumed,
            obs_t.declared_percent,
            obs_t.measured_percent,
            stamp,
        );
        let auth = crate::signing::WireAuth::with_stamp(&reporter, stamp, falsified_attest.as_bytes());
        let mut h = axum::http::HeaderMap::new();
        h.insert("x-jerith-pubkey", auth.pubkey.parse().unwrap());
        h.insert("x-jerith-stamp", auth.stamp_ms.to_string().parse().unwrap());
        h.insert("x-jerith-sig", auth.sig.parse().unwrap());
        let r = client.post(&url).headers(h).json(&obs_t).send().await.unwrap();
        assert_eq!(r.status(), axum::http::StatusCode::UNAUTHORIZED, "tampered figures refused");

        // FORGED reporter: headers signed by a DIFFERENT key than the
        // claimed reporter → identity binding refuses.
        let attacker = crate::signing::SigningIdentity::generate();
        let stamp = now_ms();
        let attest = crate::signing::measure_attest_payload(
            &obs2.reporter,
            &obs2.subject,
            obs2.block_stamp_ms,
            obs2.window_secs,
            obs2.tokens_served,
            obs2.tokens_consumed,
            obs2.declared_percent,
            obs2.measured_percent,
            stamp,
        );
        let auth = crate::signing::WireAuth::with_stamp(&attacker, stamp, attest.as_bytes());
        let mut h = axum::http::HeaderMap::new();
        h.insert("x-jerith-pubkey", auth.pubkey.parse().unwrap());
        h.insert("x-jerith-stamp", auth.stamp_ms.to_string().parse().unwrap());
        h.insert("x-jerith-sig", auth.sig.parse().unwrap());
        let r = client.post(&url).headers(h).json(&obs2).send().await.unwrap();
        assert_eq!(r.status(), axum::http::StatusCode::UNAUTHORIZED, "forged reporter refused");
    }

    /// Builds a well-formed observation: the SUBJECT signs its own
    /// measurement block (the figures' anchor), the REPORTER signs the
    /// attestation headers (delivery corroboration).
    fn make_observation(
        subject: &crate::signing::SigningIdentity,
        reporter: &crate::signing::SigningIdentity,
        window_secs: u64,
        served: u64,
        consumed: u64,
        declared: u64,
        measured: u64,
    ) -> (MeasureObservation, axum::http::HeaderMap) {
        let block_stamp = now_ms();
        let subject_payload = crate::signing::measurement_payload(
            &subject.node_id(),
            block_stamp,
            window_secs,
            served,
            consumed,
            declared,
            measured,
        );
        let obs = MeasureObservation {
            reporter: reporter.node_id(),
            subject: subject.node_id(),
            subject_pubkey: subject.pubkey_hex(),
            subject_sig: subject.sign(subject_payload.as_bytes()),
            block_stamp_ms: block_stamp,
            window_secs,
            tokens_served: served,
            tokens_consumed: consumed,
            declared_percent: declared,
            measured_percent: measured,
        };
        let stamp = now_ms();
        let attest = crate::signing::measure_attest_payload(
            &obs.reporter,
            &obs.subject,
            obs.block_stamp_ms,
            window_secs,
            served,
            consumed,
            declared,
            measured,
            stamp,
        );
        let headers = {
            let auth =
                crate::signing::WireAuth::with_stamp(reporter, stamp, attest.as_bytes());
            let mut h = axum::http::HeaderMap::new();
            h.insert("x-jerith-pubkey", auth.pubkey.parse().unwrap());
            h.insert("x-jerith-stamp", auth.stamp_ms.to_string().parse().unwrap());
            h.insert("x-jerith-sig", auth.sig.parse().unwrap());
            h
        };
        (obs, headers)
    }

    fn signed_headers(id: &crate::signing::SigningIdentity, a: &PeerAd) -> axum::http::HeaderMap {
        let stamp = crate::bootstrap::now_ms();
        let payload = crate::signing::advertise_payload(a, stamp);
        // Stamp frozen: the flaky predecessor used WireAuth::now, which
        // re-read the clock and signed a payload whose embedded stamp could
        // differ from the header stamp by 1 ms under load → intermittent
        // 401s. See signing::WireAuth::with_stamp.
        let auth = crate::signing::WireAuth::with_stamp(id, stamp, payload.as_bytes());
        let mut h = axum::http::HeaderMap::new();
        h.insert("x-jerith-pubkey", auth.pubkey.parse().unwrap());
        h.insert("x-jerith-stamp", auth.stamp_ms.to_string().parse().unwrap());
        h.insert("x-jerith-sig", auth.sig.parse().unwrap());
        h
    }

    /// Signed headers for the report endpoints (payload built by caller):
    /// `report_signed_headers(&id, reporter, target, is_fraud...)` signs
    /// the dead-report canonical payload; fraud tests build their own
    /// `jerith-fraud1` payload via `signed_headers_for`.
    fn signed_headers_for(
        id: &crate::signing::SigningIdentity,
        payload: &str,
    ) -> axum::http::HeaderMap {
        let stamp = crate::bootstrap::now_ms();
        let auth = crate::signing::WireAuth::with_stamp(id, stamp, payload.as_bytes());
        let mut h = axum::http::HeaderMap::new();
        h.insert("x-jerith-pubkey", auth.pubkey.parse().unwrap());
        h.insert("x-jerith-stamp", auth.stamp_ms.to_string().parse().unwrap());
        h.insert("x-jerith-sig", auth.sig.parse().unwrap());
        h
    }

    fn dead_report_headers(
        id: &crate::signing::SigningIdentity,
        reporter: &str,
        target: &str,
    ) -> axum::http::HeaderMap {
        let stamp = crate::bootstrap::now_ms();
        let payload = crate::signing::dead_payload(reporter, target, stamp);
        signed_headers_for(id, &payload)
    }

    #[tokio::test]
    async fn advertise_requires_valid_signature() {
        let registry = Arc::new(Registry::default());
        let app = router(Arc::clone(&registry), DEFAULT_ADVERTISE_TTL_MS);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let client = crate::http::mesh_http();
        let url = format!("http://{addr}/v1/advertise");

        // Real identity, correctly signed → accepted and listed.
        let (id, mut good) = make_peer("glm-5.3");
        let r = client
            .put(&url)
            .headers(signed_headers(&id, &good))
            .json(&good)
            .send()
            .await
            .unwrap();
        assert!(r.status().is_success(), "valid ad must be accepted");
        let list: PeerListResponse = client
            .get(format!("http://{addr}/v1/peers?model=glm-5.3"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(list.peers.len(), 1);

        // Forged: attacker signs a claim to the victim's node_id → 401.
        let attacker = crate::signing::SigningIdentity::generate();
        good.node_id = id.node_id();
        let stamp = crate::bootstrap::now_ms();
        let payload = crate::signing::advertise_payload(&good, stamp);
        let forged = crate::signing::WireAuth {
            pubkey: id.pubkey_hex(), // victim's key claimed
            stamp_ms: stamp,
            sig: attacker.sign(payload.as_bytes()),
        };
        let mut h = axum::http::HeaderMap::new();
        h.insert("x-jerith-pubkey", forged.pubkey.parse().unwrap());
        h.insert(
            "x-jerith-stamp",
            forged.stamp_ms.to_string().parse().unwrap(),
        );
        h.insert("x-jerith-sig", forged.sig.parse().unwrap());
        let r = client
            .put(&url)
            .headers(h)
            .json(&good)
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::UNAUTHORIZED);

        // Missing headers entirely → 401.
        let r = client.put(&url).json(&good).send().await.unwrap();
        assert_eq!(r.status(), StatusCode::UNAUTHORIZED);

        // Stale stamp (replay) → 401.
        let old = crate::bootstrap::now_ms() - crate::signing::STAMP_MAX_AGE_MS - 5_000;
        let payload = crate::signing::advertise_payload(&good, old);
        let mut h = axum::http::HeaderMap::new();
        h.insert("x-jerith-pubkey", id.pubkey_hex().parse().unwrap());
        h.insert("x-jerith-stamp", old.to_string().parse().unwrap());
        h.insert("x-jerith-sig", id.sign(payload.as_bytes()).parse().unwrap());
        let r = client
            .put(&url)
            .headers(h)
            .json(&good)
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn advertise_rejects_missing_fields() {
        let registry = Arc::new(Registry::default());
        let app = router(Arc::clone(&registry), DEFAULT_ADVERTISE_TTL_MS);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let client = crate::http::mesh_http();
        // Complete body but blank identity fields → handler must reject 400
        // (field validation happens before wire auth).
        let mut bad = serde_json::to_value(make_peer("glm-5.3").1).unwrap();
        bad["node_id"] = serde_json::json!("");
        bad["model_id"] = serde_json::json!("");
        let r = client
            .put(format!("http://{}{}", addr, "/v1/advertise"))
            .json(&bad)
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::BAD_REQUEST);

        // Malformed body (missing required fields) → extractor rejects 422.
        let r = client
            .put(format!("http://{}{}", addr, "/v1/advertise"))
            .json(&serde_json::json!({"node_id": "x"}))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn advertise_rejects_oversized_fields() {
        let registry = Arc::new(Registry::default());
        let app = router(Arc::clone(&registry), DEFAULT_ADVERTISE_TTL_MS);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let client = crate::http::mesh_http();
        let base = format!("http://{addr}");

        // Every string field over its sanity cap → 400, rejected BEFORE any
        // signature work (no headers needed: validation precedes wire auth).
        // Each rejection also bumps the process-local oversize counter.
        let before =
            crate::metrics::OVERSIZE_BOOTSTRAP_REJECTS.load(std::sync::atomic::Ordering::Relaxed);
        let bad = serde_json::to_value(make_peer("glm-5.3").1).unwrap();
        for field in ["node_id", "model_id", "quant", "address", "rpc_address"] {
            let mut candidate = bad.clone();
            candidate[field] = serde_json::json!("x".repeat(2_000));
            let r = client
                .put(format!("{base}/v1/advertise"))
                .json(&candidate)
                .send()
                .await
                .unwrap();
            assert_eq!(r.status(), StatusCode::BAD_REQUEST, "field {field}");
        }
        // Parallel test streams share the counter: a delta >= 5 proves OUR
        // five rejections each bumped it (monotonic counter).
        assert!(
            crate::metrics::OVERSIZE_BOOTSTRAP_REJECTS.load(std::sync::atomic::Ordering::Relaxed)
                >= before + 5,
            "five field-cap rejections must bump the oversize counter"
        );

        // /health surfaces this process's security counters (additive
        // fields; shape check only).
        let health: serde_json::Value = client
            .get(format!("{base}/health"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let sec = health.get("security").expect("security section in /health");
        for key in [
            "rate_limited_429_total",
            "oversize_reject_total",
            "oversize_bootstrap_rejects",
        ] {
            assert!(
                sec[key].is_u64(),
                "security.{key} must be a u64 counter, got {sec}"
            );
        }

        // contribution_percent is only meaningful 0-100.
        let mut pct = serde_json::to_value(make_peer("glm-5.3").1).unwrap();
        pct["contribution_percent"] = serde_json::json!(150);
        let r = client
            .put(format!("{base}/v1/advertise"))
            .json(&pct)
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::BAD_REQUEST);

        // Positive control: an untouched ad is still accepted.
        let (id, good) = make_peer("glm-5.3");
        let r = client
            .put(format!("{base}/v1/advertise"))
            .headers(signed_headers(&id, &good))
            .json(&good)
            .send()
            .await
            .unwrap();
        assert!(r.status().is_success(), "untouched ad must be accepted");
    }

    #[tokio::test]
    async fn advertise_caps_new_entries_but_allows_refresh() {
        // Capacity-bound registry of ONE entry for a test-scale probe.
        let registry = Arc::new(Registry {
            measures: Default::default(),
            entries: RwLock::new(HashMap::new()),
            max_entries: 1,
        });
        let app = router(Arc::clone(&registry), DEFAULT_ADVERTISE_TTL_MS);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let client = crate::http::mesh_http();
        let url = format!("http://{addr}/v1/advertise");

        // First identity: accepted.
        let (id1, ad1) = make_peer("glm-5.3");
        let r = client
            .put(&url)
            .headers(signed_headers(&id1, &ad1))
            .json(&ad1)
            .send()
            .await
            .unwrap();
        assert!(r.status().is_success(), "first ad accepted");

        // Refresh of the SAME id: never counts against the cap.
        let r = client
            .put(&url)
            .headers(signed_headers(&id1, &ad1))
            .json(&ad1)
            .send()
            .await
            .unwrap();
        assert!(r.status().is_success(), "refresh must bypass the cap");

        // A NEW identity beyond the cap: refused with 503.
        let (id2, ad2) = make_peer("qwen3.8-flash-next");
        let r = client
            .put(&url)
            .headers(signed_headers(&id2, &ad2))
            .json(&ad2)
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body: serde_json::Value = r.json().await.unwrap();
        assert!(body["error"].as_str().unwrap().contains("at capacity"));

        // The registry-cap refusal is size-bounding too: it bumps the
        // same process-local oversize counter (delta >= 1; the earlier
        // accepted/refresh ads must not have bumped it).
        assert!(
            crate::metrics::OVERSIZE_BOOTSTRAP_REJECTS.load(std::sync::atomic::Ordering::Relaxed)
                >= 1,
            "registry-cap refusal must bump the oversize counter"
        );
    }

    #[tokio::test]
    async fn sweep_expired_drops_stale_only() {
        let registry = Arc::new(Registry::default());
        let (_, ad_fresh) = make_peer("glm-5.3");
        let fresh_id = ad_fresh.node_id.clone();
        registry.entries.write().await.insert(
            fresh_id.clone(),
            Entry {
                ad: ad_fresh,
                last_refresh_ms: now_ms(),
                dead_votes: 0,
                score: 0,
                fraud_reports: HashMap::new(),
            },
        );
        let (_, ad_stale) = make_peer("qwen3.8-flash-next");
        let stale_id = ad_stale.node_id.clone();
        registry.entries.write().await.insert(
            stale_id.clone(),
            Entry {
                ad: ad_stale,
                last_refresh_ms: now_ms().saturating_sub(DEFAULT_ADVERTISE_TTL_MS + 1_000),
                dead_votes: 0,
                score: 0,
                fraud_reports: HashMap::new(),
            },
        );

        let dropped = sweep_expired(&registry, DEFAULT_ADVERTISE_TTL_MS).await;
        assert_eq!(dropped, 1);
        let entries = registry.entries.read().await;
        assert!(entries.contains_key(&fresh_id), "fresh entry kept");
        assert!(!entries.contains_key(&stale_id), "stale entry dropped");
    }

    #[tokio::test]
    async fn anonymous_reads_rate_limited_per_ip() {
        use crate::ratelimit::RateLimiter;
        let registry = Arc::new(Registry::default());
        // Budget of 3/60s for a test-scale probe.
        let app = router_with_limiters(
            Arc::clone(&registry),
            DEFAULT_ADVERTISE_TTL_MS,
            Arc::new(RateLimiter::new(3, Duration::from_secs(60))),
            Arc::new(RateLimiter::new(999, Duration::from_secs(60))),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        // Connect info must be wired like production serve() does — the
        // limiter's socket-fallback attribution depends on it.
        tokio::spawn(async move {
            axum::serve(
                listener,
                app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
            )
            .await
            .unwrap();
        });
        let client = crate::http::mesh_http();
        let url = format!("http://{addr}/v1/models/counts");

        for _ in 0..3 {
            assert_eq!(
                client.get(&url).send().await.unwrap().status(),
                StatusCode::OK
            );
        }
        let r = client.get(&url).send().await.unwrap();
        assert_eq!(r.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(
            r.headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .is_some(),
            "429 must carry Retry-After"
        );
        let body: serde_json::Value = r.json().await.unwrap();
        assert!(body["error"].as_str().unwrap().contains("rate limit"));
    }

    /// Shared bootstrap-lane mount: app on a real listener wired with
    /// connect info (the anon lane's attribution needs it).
    async fn mount_with_limiters(
        registry: &Arc<Registry>,
        anon: Arc<crate::ratelimit::RateLimiter>,
        identity: Arc<crate::ratelimit::RateLimiter>,
    ) -> (reqwest::Client, String) {
        let app = router_with_limiters(
            Arc::clone(registry),
            DEFAULT_ADVERTISE_TTL_MS,
            anon,
            identity,
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(
                listener,
                app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
            )
            .await
            .unwrap();
        });
        (
            crate::http::mesh_http(),
            format!("http://{addr}/v1/peers?model=lane-test"),
        )
    }

    #[tokio::test]
    async fn signed_peers_fetches_get_the_identity_lane() {
        use crate::ratelimit::RateLimiter;
        let registry = Arc::new(Registry::default());
        // The anon floor is tiny (3/60s); the identity lane is generous.
        let (client, url) = mount_with_limiters(
            &registry,
            Arc::new(RateLimiter::new(3, Duration::from_secs(60))),
            Arc::new(RateLimiter::new(999, Duration::from_secs(60))),
        )
        .await;
        let identity = crate::signing::SigningIdentity::generate();

        // Unsigned (or keyless): the anonymous floor trips at its cap.
        let mut anonymous_429s = 0;
        for _ in 0..6 {
            let status = client.get(&url).send().await.unwrap().status();
            if status == StatusCode::TOO_MANY_REQUESTS {
                anonymous_429s += 1;
            }
        }
        assert_eq!(anonymous_429s, 3, "anon floor must trip at its cap");

        // Signed: the identity lane owns the budget — all 10 served even
        // though the same IP's anonymous bucket is long exhausted.
        for i in 0..10 {
            let stamp = now_ms();
            let headers =
                signed_headers_for(&identity, &crate::signing::peers_fetch_payload(stamp));
            let r = client.get(&url).headers(headers).send().await.unwrap();
            assert_eq!(
                r.status(),
                StatusCode::OK,
                "signed fetch #{i} must ride the identity lane"
            );
        }
    }

    #[tokio::test]
    async fn forged_signature_gets_the_anonymous_lane() {
        use crate::ratelimit::RateLimiter;
        let registry = Arc::new(Registry::default());
        let (client, url) = mount_with_limiters(
            &registry,
            Arc::new(RateLimiter::new(3, Duration::from_secs(60))),
            Arc::new(RateLimiter::new(999, Duration::from_secs(60))),
        )
        .await;
        let identity = crate::signing::SigningIdentity::generate();

        // Wire-auth headers present but signing a DIFFERENT payload —
        // the lane keys off `peers-req1|{stamp}` verbatim, so a forged
        // signature degrades to the anonymous floor, not the allowance.
        let mut forged_429s = 0;
        for i in 0..6 {
            let wrong_payload = format!("not-the-peers-lane|{}|{i}", now_ms());
            let headers = signed_headers_for(&identity, &wrong_payload);
            let status = client
                .get(&url)
                .headers(headers)
                .send()
                .await
                .unwrap()
                .status();
            if status == StatusCode::TOO_MANY_REQUESTS {
                forged_429s += 1;
            }
        }
        assert_eq!(
            forged_429s, 3,
            "forged signatures must ride the anon floor (3 capped)"
        );
    }

    async fn seed_live_peer(
        _registry: &Arc<Registry>,
        base: &str,
        client: &reqwest::Client,
        identity: &crate::signing::SigningIdentity,
    ) -> String {
        let a = PeerAd {
            node_id: identity.node_id(),
            address: "127.0.0.1:9000".into(),
            model_id: "glm-5.3".into(),
            quant: "int4".into(),
            contribution_percent: 30,
            rpc_address: String::new(),
            last_seen_ms: now_ms(),
            role: NodeRole::Hybrid,
            resources: None,
            hosted_experts: None,
            hosted_expert_count: None,
            discovery_servers: Vec::new(),
            gossip_reputations: None,
        };
        let r = client
            .put(format!("{base}/v1/advertise"))
            .headers(signed_headers(identity, &a))
            .json(&a)
            .send()
            .await
            .unwrap();
        assert!(r.status().is_success());
        a.node_id
    }

    /// POST a fraud report with the reporter's wire-auth headers. The
    /// server now verifies signatures, so tests (and the real client in
    /// bootstrap.rs) must sign `jerith-fraud1|stamp|{kind}|{reporter}|{target}`.
    async fn post_fraud(
        client: &reqwest::Client,
        base: &str,
        reporter_id: &crate::signing::SigningIdentity,
        kind_str: &str,
        target: &str,
    ) -> reqwest::Response {
        let reporter_name = reporter_id.node_id();
        let stamp = crate::bootstrap::now_ms();
        let payload = format!("jerith-fraud1|{stamp}|{kind_str}|{reporter_name}|{target}");
        let auth = crate::signing::WireAuth::with_stamp(reporter_id, stamp, payload.as_bytes());
        client
            .post(format!("{base}/v1/report-fraud"))
            .header("x-jerith-pubkey", auth.pubkey.clone())
            .header("x-jerith-stamp", auth.stamp_ms.to_string())
            .header("x-jerith-sig", auth.sig.clone())
            .json(&serde_json::json!({
                "node_id": target,
                "reporter": reporter_name,
                "kind": kind_str,
            }))
            .send()
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn fraud_reports_penalize_and_halt() {
        let registry = Arc::new(Registry::default());
        let app = router(Arc::clone(&registry), DEFAULT_ADVERTISE_TTL_MS);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let base = format!("http://{addr}");
        let client = crate::http::mesh_http();

        let accused = crate::signing::SigningIdentity::generate();
        let r1 = crate::signing::SigningIdentity::generate();
        let r2 = crate::signing::SigningIdentity::generate();
        let target = seed_live_peer(&registry, &base, &client, &accused).await;

        async fn post(
            client: &reqwest::Client,
            base: &str,
            reporter_sig: &crate::signing::SigningIdentity,
            kind_str: &str,
            target: &str,
        ) -> reqwest::Response {
            post_fraud(client, base, reporter_sig, kind_str, target).await
        }

        // Self-report rejected.
        let r = client
            .post(format!("{base}/v1/report-fraud"))
            .json(&serde_json::json!({
                "node_id": target,
                "reporter": target,
                "kind": "failed_worker",
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::BAD_REQUEST);

        // First report: -30 → already past the halt threshold (-20): a
        // single corroborated-or-not failed-worker signal stops mesh
        // dispatches immediately.
        let v: serde_json::Value = post(&client, &base, &r1, "failed_worker", &target)
            .await
            .json()
            .await
            .unwrap();
        assert_eq!(v["score"], -30);
        assert_eq!(v["halted"], true);

        // Same reporter repeats: deduped, no double penalty.
        let v: serde_json::Value = post(&client, &base, &r1, "failed_worker", &target)
            .await
            .json()
            .await
            .unwrap();
        assert_eq!(v["score"], -30);

        // Second distinct reporter (corroboration): another -30 → -60 → halted.
        let v: serde_json::Value = post(&client, &base, &r2, "failed_worker", &target)
            .await
            .json()
            .await
            .unwrap();
        assert_eq!(v["score"], -60);
        assert_eq!(v["halted"], true);

        // Halted node no longer appears in the public peer list.
        let list: PeerListResponse = client
            .get(format!("{base}/v1/peers?model=glm-5.3"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert!(list.peers.is_empty(), "halted peer must be withheld");

        // Different fraud kind adds a separate penalty once.
        let r3 = crate::signing::SigningIdentity::generate();
        let v: serde_json::Value = post_fraud(&client, &base, &r3, "dishonest_results", &target)
            .await
            .json()
            .await
            .unwrap();
        assert_eq!(v["score"], -100); // -60 - 40, at the -25... floor check below

        // Floor: beyond it never goes.
        let v2: serde_json::Value = post_fraud(&client, &base, &r2, "rate_abuse", &target)
            .await
            .json()
            .await
            .unwrap();
        assert_eq!(v2["score"], -100); // capped at REPUTATION_FLOOR
    }

    #[tokio::test]
    async fn reputation_survives_readvertise_and_decays() {
        let registry = Arc::new(Registry::default());
        let app = router(Arc::clone(&registry), DEFAULT_ADVERTISE_TTL_MS);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let base = format!("http://{addr}");
        let client = crate::http::mesh_http();

        let accused = crate::signing::SigningIdentity::generate();
        let reporter = crate::signing::SigningIdentity::generate();
        let target = seed_live_peer(&registry, &base, &client, &accused).await;

        // One fraud report → -30.
        let _: serde_json::Value = post_fraud(&client, &base, &reporter, "leeching", &target)
            .await
            .json()
            .await
            .unwrap();

        // Node re-advertises (e.g., restart) → score/history must persist.
        let a = PeerAd {
            node_id: target.clone(),
            address: "127.0.0.1:9000".into(),
            model_id: "glm-5.3".into(),
            quant: "int4".into(),
            contribution_percent: 30,
            rpc_address: String::new(),
            last_seen_ms: now_ms(),
            role: NodeRole::Hybrid,
            resources: None,
            hosted_experts: None,
            hosted_expert_count: None,
            discovery_servers: Vec::new(),
            gossip_reputations: None,
        };
        // Re-sign with the accused's identity (same key, same node_id).
        let r = client
            .put(format!("{base}/v1/advertise"))
            .headers(signed_headers(&accused, &a))
            .json(&a)
            .send()
            .await
            .unwrap();
        assert!(r.status().is_success());

        // Another report from a distinct peer: coroutine applies -20 on top
        // of -20 (decay is 0 immediately) → -40.
        let corroborator = crate::signing::SigningIdentity::generate();
        let v: serde_json::Value = post_fraud(&client, &base, &corroborator, "leeching", &target)
            .await
            .json()
            .await
            .unwrap();
        assert_eq!(v["score"], -40);

        // Old reporter repeats after re-advertise: still deduped (history
        // carried across).
        let v: serde_json::Value = post_fraud(&client, &base, &reporter, "leeching", &target)
            .await
            .json()
            .await
            .unwrap();
        assert_eq!(v["score"], -40);
    }

    #[test]
    fn reputation_recovery_calculates_decayed_scores() {
        let mut e = Entry {
            ad: PeerAd {
                node_id: "x".into(),
                address: String::new(),
                model_id: String::new(),
                quant: String::new(),
                contribution_percent: 30,
                rpc_address: String::new(),
                last_seen_ms: 0,
                role: NodeRole::Hybrid,
                resources: None,
                hosted_experts: None,
                hosted_expert_count: None,
                discovery_servers: Vec::new(),
                gossip_reputations: None,
            },
            last_refresh_ms: 0,
            dead_votes: 0,
            score: -60,
            fraud_reports: HashMap::new(),
        };
        // No reports → pinned at or below 0.
        apply_recovery(&mut e, now_ms());
        assert_eq!(e.score, -60);

        // A report 2 full days old → -60 + 4 = -56.
        e.fraud_reports.insert(
            "r1".into(),
            (FraudKind::Leeching, now_ms() - 2 * 86_400_000),
        );
        apply_recovery(&mut e, now_ms());
        assert_eq!(e.score, -56);

        // Long recovery: 60 days → capped at 0.
        e.fraud_reports.insert(
            "r1".into(),
            (FraudKind::Leeching, now_ms() - 60 * 86_400_000),
        );
        apply_recovery(&mut e, now_ms());
        assert_eq!(e.score, 0);

        // Floor respected.
        e.score = -30;
        e.fraud_reports
            .insert("r2".into(), (FraudKind::RateAbuse, now_ms()));
        // apply_recovery would try to add 0 days → stays.
        apply_recovery(&mut e, now_ms());
        assert_eq!(e.score, -30);
        // Compile-time sanity: the halt line never descends below the floor.
        const _: () = assert!(HALT_SCORE_THRESHOLD > REPUTATION_FLOOR); // -20 > -100
    }

    #[test]
    fn legacy_ad_payload_matches_old_canonical_shape() {
        // v1 canonical (pre-rpc_address): ends with the role string, no
        // trailing segment for rpc_address.
        let id = crate::signing::SigningIdentity::generate();
        let mut a = PeerAd {
            node_id: id.node_id(),
            address: "127.0.0.1:9000".into(),
            model_id: "glm-5.3".into(),
            quant: "int4".into(),
            contribution_percent: 30,
            rpc_address: "127.0.0.1:9001".into(),
            last_seen_ms: 1_000,
            role: NodeRole::Hybrid,
            resources: None,
            hosted_experts: None,
            hosted_expert_count: None,
            discovery_servers: Vec::new(),
            gossip_reputations: None,
        };
        // Two stamps: a fixed one for shape determinism, a fresh one for
        // the wire-auth freshness checks.
        let stamp = 7_000_000;
        let fresh_stamp = crate::bootstrap::now_ms();
        let legacy = crate::signing::legacy_ad_excerpt(&a, stamp);
        assert!(
            !legacy.contains("127.0.0.1:9001"),
            "legacy payload must exclude rpc_address"
        );
        assert!(legacy.ends_with("hybrid"), "legacy payload ends at role");
        // And a legacy-issued signature validates against it (fresh stamp).
        let legacy_fresh = crate::signing::legacy_ad_excerpt(&a, fresh_stamp);
        let sig = id.sign(legacy_fresh.as_bytes());
        let auth = crate::signing::WireAuth {
            pubkey: id.pubkey_hex(),
            stamp_ms: fresh_stamp,
            sig,
        };
        assert!(auth.validate(legacy_fresh.as_bytes(), &a.node_id).is_ok());
        // While the same signature does NOT validate over the new payload.
        let modern = crate::signing::advertise_payload(&a, fresh_stamp);
        assert_ne!(modern, legacy_fresh);

        // rpc_address change alone must not break legacy verification
        // (the field was added after the legacy epoch).
        a.rpc_address = "z".into();
        let legacy2 = crate::signing::legacy_ad_excerpt(&a, stamp);
        assert_eq!(legacy, legacy2, "sanity: builder is deterministic");
    }

    #[test]
    fn default_options_match_spec_port() {
        let o = BootstrapOptions::default();
        assert_eq!(o.port, 8901);
        assert_eq!(o.advertise_ttl_ms, DEFAULT_ADVERTISE_TTL_MS);
    }

    #[test]
    fn model_roundtrip_is_preserved() {
        let m = model("glm-5.3");
        let json = serde_json::to_string(&m).unwrap();
        let back: ModelId = serde_json::from_str(&json).unwrap();
        assert_eq!(back, m);
    }
}
