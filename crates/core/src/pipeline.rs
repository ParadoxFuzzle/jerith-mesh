//! Multi-hop pipeline route planning (Phase 3 — FEATURES §13's first
//! milestone, M0).
//!
//! What already exists in earlier phases: experts as real weights
//! (`expert_store`), the deterministic layer-ownership ring
//! (`replication::ShardMap::layer_ownership`), and expert dispatch plans.
//! What a Petals-style pipeline ultimately needs is route planning over
//! *layer groups*: an activation walks a deterministic sequence of hops,
//! each owned by exactly one node from the ring, and the runtime later
//! ships residual streams along that route (M1+; see ARCHITECTURE.md).
//!
//! M0: pure route planning from the ring's full layer map —
//! no I/O, no runtime, no clocks: deterministic, unit-testable, and
//! naturally degradable (a single hop when only the local node is alive).
//!
//! M0 + M1 + M2 (this module): pure route planning from the ring's full layer
//! map, the route executor that walks it, and the degradation layer that
//! keeps routes alive when an owner dies mid-walk. Planning is I/O-free and
//! clock-free (deterministic, unit-testable, naturally degradable); the
//! executor is the M1 hop runtime over the same f32-vector wire contract
//! expert dispatch already uses, with the Phase 4 commitment recomputed
//! *per hop* (each hop's expert id names its layer run, so a tampering hop
//! is caught downstream exactly like a single-expert reply) and per-hop
//! deadlines derived from the owner's measured RTT.
//!
//! M2 adds the degradation layer: a dead hop's layers are re-hosted per the
//! ring's own rule (dropping the departed owner re-pins only its layers),
//! unrecoverable losses produce honest hops-short partial answers with a
//! typed reason (the router walks on, mirroring reject-continue), and
//! flapping owners are dampened by a freerider-style two-line hysteresis
//! so a bouncing node cannot chatter the route plan.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::time::Duration;

/// One pipeline hop: the node that owns this run of consecutive layers
/// (`None` = the local node's own slice), together with those layers in
/// ascending order. Hops are ordered by ascending layer index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineHop {
    /// Owner's node id; `None` marks the local node's own slice.
    pub node: Option<String>,
    /// The consecutive layers this hop executes (ascending).
    pub layers: Vec<usize>,
}

/// Build the pipeline route for the ring's full layer ownership map
/// (layers ascending — the caller's model walk order, which a BTreeMap
/// preserves). Consecutive layers with the same owner merge into one hop;
/// the local slice appears as `node: None`.
///
/// The map is the full one from
/// [`ShardMap::layer_ownership`](crate::replication::ShardMap::layer_ownership)
/// — restricting it to self-owned layers left every downstream planner's
/// remote path unreachable (the pre-M0 defect this module replaces with a
/// complete picture).
pub fn plan_route(self_node_id: &str, ownership: &BTreeMap<usize, String>) -> Vec<PipelineHop> {
    let mut hops: Vec<PipelineHop> = Vec::new();
    for (&layer, owner) in ownership {
        let node = if owner == self_node_id {
            None
        } else {
            Some(owner.clone())
        };
        match hops.last_mut() {
            Some(last) if last.node == node => last.layers.push(layer),
            _ => hops.push(PipelineHop {
                node,
                layers: vec![layer],
            }),
        }
    }
    hops
}

/// Stable per-hop expert id: the model id plus the hop's layer run as a
/// closed range. The commitment layer canonicalizes over
/// (req_id, expert_id, node_id, input, output), so a *deterministic expert
/// id per hop* is what makes per-hop commitments composable rather than
/// ad-hoc — the same expert-shaped tuple every single-expert reply uses.
pub fn hop_expert_id(model_id: &str, hop: &PipelineHop) -> String {
    let first = hop.layers.first().copied().unwrap_or(0);
    let last = hop.layers.last().copied().unwrap_or(first);
    format!("{model_id}#{first}-{last}")
}

/// Per-hop deadline: the dispatch base plus 3× the owner's measured RTT
/// (round-trip plus the run plus generous jitter headroom — the same
/// conservative multiplier the mesh uses for one-expert dispatch).
pub fn hop_timeout(base: Duration, rtt_ms: u64) -> Duration {
    let extra = rtt_ms.saturating_mul(3).saturating_mul(1_000_000);
    base.saturating_add(Duration::from_nanos(extra))
}

// ---- M3 — latency budgeting: the walk is timed per attempt, and the
// budgets it is held to are priced from the measured-activation ordering.

/// One attempted hop's measured timing — all attempts, successes AND
/// failures (a failed attempt's dead wait is as real to the route's
/// latency as a slow success).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttemptTiming {
    pub node: Option<String>,
    pub layers: Vec<usize>,
    pub ms: u64,
    pub ok: bool,
}

/// A hop that overran the budget priced for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OverBudgetHop {
    pub node: Option<String>,
    pub layers: Vec<usize>,
    pub measured_ms: u64,
    pub budget_ms: u64,
}

/// One hop's priced slice of a route walk budget: its RTT-floored
/// deadline plus its share of whatever total budget remains, weighted
/// by measured demand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HopBudget {
    pub node: Option<String>,
    pub layers: Vec<usize>,
    pub demand_weight: u64,
    pub budget_ms: u64,
}

/// Price a planned route against the measured-activation ordering, M3's
/// contract with the demand signal: `demand_of` is the caller-wired
/// view over [`crate::expert_stats::ExpertStats`] (per-hop measured
/// activations — the same oracle prefetching will use, no new source of
/// truth). Every hop first gets its RTT-floored deadline
/// ([`hop_timeout`] — the M1 deadline IS the floor); whatever budget
/// remains after all floors is spread across hops in proportion to
/// measured demand, so hot hops win headroom while cold hops still get
/// their floor (nobody starves the walk). Ties in demand keep the
/// route's original order — stable sort, deterministic u128 arithmetic.
/// Hops return SORTED BY DEMAND DESCENDING: the measured-activation
/// order is the pricing contract, and callers walking the priced order
/// are walking demand itself.
pub fn plan_route_budget(
    route: &[PipelineHop],
    demand_of: impl Fn(&PipelineHop) -> u64,
    rtt_of: impl Fn(&str) -> u64,
    base: Duration,
    total_budget_ms: u64,
) -> Vec<HopBudget> {
    let mut priced: Vec<HopBudget> = route
        .iter()
        .map(|hop| {
            let floor_ms = match &hop.node {
                Some(node) => hop_timeout(base, rtt_of(node)).as_millis() as u64,
                None => base.as_millis() as u64,
            };
            HopBudget {
                node: hop.node.clone(),
                layers: hop.layers.clone(),
                demand_weight: demand_of(hop),
                budget_ms: floor_ms,
            }
        })
        .collect();
    // Stable sort: ties in demand keep the route's original order.
    priced.sort_by_key(|b| std::cmp::Reverse(b.demand_weight));
    let floors_sum: u128 = priced.iter().map(|b| b.budget_ms as u128).sum();
    let remaining = (total_budget_ms as u128).saturating_sub(floors_sum);
    let demand_sum: u128 = priced.iter().map(|b| b.demand_weight as u128).sum();
    let hop_count = priced.len();
    for b in priced.iter_mut() {
        // Distribute by demand share; with no measured demand at all the
        // shares degenerate to an even split (checked_div's None branch).
        let extra = remaining
            .checked_mul(b.demand_weight as u128)
            .unwrap_or(0)
            .checked_div(demand_sum)
            .unwrap_or(remaining / hop_count as u128);
        b.budget_ms = (b.budget_ms as u128 + extra) as u64;
    }
    priced
}

/// Typed failure of a route walk: which hop failed, who owned it, which
/// layers it covered, why, and how much of the route completed before the
/// failure (M2's partial-answer semantics build on that count).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteError {
    pub hop_index: usize,
    pub node: Option<String>,
    pub layers: Vec<usize>,
    pub reason: String,
    pub hops_completed: usize,
}

impl std::fmt::Display for RouteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let owner = self.node.clone().unwrap_or_else(|| "local".into());
        write!(
            f,
            "pipeline hop {}/{owner} (layers {:?}) after {} completed hops: {}",
            self.hop_index, self.layers, self.hops_completed, self.reason
        )
    }
}

impl std::error::Error for RouteError {}

/// One hop attempt shared by both walks: local executor, remote executor
/// under its RTT-derived deadline, plus the Phase 4 commitment recomputed
/// against the hop's own expert id (a tampering hop reads exactly like a
/// single-expert commitment mismatch).
// Parameter list mirrors the single-expert dispatch contract: two identity
// strings, the closures, and the clock inputs travel together.
#[allow(clippy::too_many_arguments)]
async fn attempt_one_hop(
    model_id: &str,
    req_id: &str,
    hop: &PipelineHop,
    activation: &[f32],
    local_run: &impl Fn(&[f32], &[usize]) -> anyhow::Result<Vec<f32>>,
    remote_run: &impl Fn(
        &str,
        &str,
        Vec<f32>,
        Duration,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = anyhow::Result<(Vec<f32>, Option<String>)>> + Send>,
    >,
    rtt_of: &impl Fn(&str) -> u64,
    base_timeout: Duration,
) -> anyhow::Result<Vec<f32>> {
    let expert_id = hop_expert_id(model_id, hop);
    match &hop.node {
        None => local_run(activation, &hop.layers),
        Some(node) => {
            let activation = activation.to_vec();
            let (out, commitment) = remote_run(
                &expert_id,
                node,
                activation.clone(),
                hop_timeout(base_timeout, rtt_of(node)),
            )
            .await?;
            match commitment {
                Some(c)
                    if crate::verify::verify_commitment(
                        req_id,
                        &expert_id,
                        node,
                        &activation,
                        &out,
                        &c,
                    ) =>
                {
                    Ok(out)
                }
                Some(_) => Err(anyhow::anyhow!(
                    "commitment mismatch from hop {node}: body altered"
                )),
                None => Ok(out),
            }
        }
    }
}

/// Walk [`plan_route`]'s sequence hop-by-hop, carrying the residual stream:
/// each hop's output becomes the next hop's input. Composition mirrors the
/// existing single-expert contract —
///
/// - **Local hop** (`node: None`) runs [`crate::expert_store::Expert::forward`]
///   shape via `local_run`, which receives the current activation and the
///   hop's layers (real weights arrive with the Colibri-side layer split;
///   until then synthetic/moe experts fill the role).
/// - **Remote hop** runs `remote_run` — the same signed expert RPC — whose
///   `Duration` argument is [`hop_timeout`] from the owner's RTT (the
///   closure owns applying it, as `execute`'s dispatch closures already do).
///   It returns the output plus the server's Phase 4 commitment, which the
///   executor re-verifies here against (req_id, *per-hop* expert id, owner,
///   in, out): a hop's commitment names its own layer run, so a tampering
///   hop is caught downstream exactly like a single-expert reply.
/// - Any failure aborts the route with a typed [`RouteError`] — no silent
///   skipping. The degradation lives in
///   [`execute_route_resilient`], which reassigns and walks on instead.
#[allow(clippy::too_many_arguments)]
pub async fn execute_route(
    model_id: &str,
    req_id: &str,
    route: &[PipelineHop],
    mut activation: Vec<f32>,
    local_run: impl Fn(&[f32], &[usize]) -> anyhow::Result<Vec<f32>>,
    remote_run: impl Fn(
        &str,
        &str,
        Vec<f32>,
        Duration,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = anyhow::Result<(Vec<f32>, Option<String>)>> + Send>,
    >,
    rtt_of: impl Fn(&str) -> u64,
    base_timeout: Duration,
) -> Result<Vec<f32>, RouteError> {
    for (idx, hop) in route.iter().enumerate() {
        match attempt_one_hop(
            model_id,
            req_id,
            hop,
            &activation,
            &local_run,
            &remote_run,
            &rtt_of,
            base_timeout,
        )
        .await
        {
            Ok(out) => activation = out,
            Err(e) => {
                return Err(RouteError {
                    hop_index: idx,
                    node: hop.node.clone(),
                    layers: hop.layers.clone(),
                    reason: e.to_string(),
                    hops_completed: idx,
                });
            }
        }
    }
    Ok(activation)
}

/// Cap on route surgery per walk: how many times a failed hop's layers may
/// be re-hosted before the executor gives up and degrades honestly. A
/// replacement that itself fails can be re-reassigned once more — but the
/// walk can never spin (the budget sums what the route plus a bounded
/// headroom allows).
pub const REASSIGN_BUDGET: usize = 12;

/// Layers an unrecovered hop left undone (honest hops-short answer).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DegradedHop {
    pub layers: Vec<usize>,
    pub node: Option<String>,
    pub reason: String,
}

/// Layers a dead owner's hop gave up, re-hosted via the ring's own rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReassignedHop {
    pub layers: Vec<usize>,
    pub was_node: Option<String>,
    pub reason: String,
}

/// Result of a perhaps-degraded route walk. Never an error unless the walk
/// truly cannot answer: a degraded walk returns the residual as far as the
/// chain carried it, in [`RouteResult::activation`], with the typing in
/// [`RouteResult::degraded`] (skipped) and [`RouteResult::reassigned`]
/// (re-hosted). Mirrors reject-continue — the router walks on.
#[derive(Debug, Clone)]
pub struct RouteResult {
    pub activation: Vec<f32>,
    pub hops_run: usize,
    pub degraded: Vec<DegradedHop>,
    pub reassigned: Vec<ReassignedHop>,
    /// Per-attempt measured timings (M3), in walk order — successes and
    /// failures alike; the failure timings count against the walk's
    /// latency as much as a slow success.
    pub attempts: Vec<AttemptTiming>,
    /// Hops (M3) that overran the budget priced for them, typed with the
    /// measured and budgeted durations. Budgets come from the caller's
    /// `budget_of` hook — `None` leaves this empty.
    pub over_budget: Vec<OverBudgetHop>,
    /// M4 enforcement: true when the whole-walk budget ran out and the
    /// remaining hops were degraded instead of dialed (each named in
    /// [`Self::degraded`] with a `walk budget exhausted` reason). A walk
    /// that finishes inside the envelope — or has no envelope — is false.
    pub budget_exhausted: bool,
}

impl RouteResult {
    pub fn is_complete(&self) -> bool {
        self.degraded.is_empty()
    }

    /// The reason string every budget-degraded hop carries, so callers
    /// can type-filter without parsing formats.
    pub const BUDGET_EXHAUSTED_REASON: &'static str = "walk budget exhausted";
    /// The reason string every per-request-budget-degraded hop carries
    /// (§1). Type-filterable without parsing formats, like the M4 wall.
    pub const REQUEST_BUDGET_EXHAUSTED_REASON: &'static str = "per-request budget exhausted";
}

/// §1 per-request expert budget: work-unit caps on what ONE top-level
/// request may spend on the mesh, enforced inline by the resilient walk
/// — the path speculative replays will reuse, so the budget exists before
/// real speculative runs do (the feature note's exact gap: a huge
/// speculative run cannot starve other askers).
///
/// Wall-clock budgets (M4's whole-walk envelope) only bite when dials are
/// SLOW; these caps bite when a single request issues more WORK than a
/// mesh share should carry — an unbounded speculative replay loop, a
/// poisoned plan with endless reassignment churn, or a request moving
/// wire-scale payloads hop by hop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestBudget {
    /// Max REMOTE dial attempts (fault reassignments included — every
    /// reassignment sub-hop is a fresh dial). `None` = no cap in play.
    pub max_remote_attempts: Option<usize>,
    /// Max wire payload (bytes, counted exactly as the walk ships them:
    /// 4 bytes per f32 element, in AND out) one request may move.
    /// `None` = no cap in play.
    pub max_wire_bytes: Option<u64>,
}

/// Cap on remote attempts per request (SPECULATIVE_RUN headroom: the
/// honest layer walk is one attempt per hop, and a full reassignment
/// storm fits `REASSIGN_BUDGET` + route length — far below this; the cap
/// exists for unbounded speculative replay, never to squeeze honest
/// walks).
pub const DEFAULT_MAX_REMOTE_ATTEMPTS: usize = 64;

/// Cap on wire bytes per request (honest MoE activations — d_model-sized
/// f32 vectors over a bounded hop count — sit orders of magnitude below;
/// the cap is a flood guard, not a squeeze).
pub const DEFAULT_REQUEST_WIRE_BYTES: u64 = 256 * 1024 * 1024;

impl Default for RequestBudget {
    fn default() -> Self {
        Self {
            max_remote_attempts: Some(DEFAULT_MAX_REMOTE_ATTEMPTS),
            max_wire_bytes: Some(DEFAULT_REQUEST_WIRE_BYTES),
        }
    }
}

/// Availability hysteresis for route planning: the freerider module's
/// decay-then-flag shape, pointed at liveness instead of honesty. A node's
/// failure score rises on failures and decays on successes; crossing
/// [`HOP_EXCLUDE_SCORE`] excludes it from route planning, and it returns
/// only after decay below [`HOP_CLEAR_SCORE`] — one line to raise, a lower
/// one to clear, so a flapping peer cannot chatter the plan.
pub const HOP_EXCLUDE_SCORE: u32 = 3;
pub const HOP_CLEAR_SCORE: u32 = 1;

#[derive(Debug, Clone, Default)]
pub struct RouteStability {
    scores: HashMap<String, u32>,
    excluded: HashSet<String>,
}

impl RouteStability {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn note_failure(&mut self, node: &str) {
        let s = self.scores.entry(node.to_string()).or_insert(0);
        *s = s.saturating_add(1);
        if *s >= HOP_EXCLUDE_SCORE {
            self.excluded.insert(node.to_string());
        }
    }

    pub fn note_success(&mut self, node: &str) {
        if let Some(s) = self.scores.get_mut(node) {
            *s = s.saturating_sub(1);
            if *s <= HOP_CLEAR_SCORE {
                self.excluded.remove(node);
            }
        }
    }

    pub fn is_excluded(&self, node: &str) -> bool {
        self.excluded.contains(node)
    }

    /// Alive set minus excluded owners — never empty: a mesh whose every
    /// owner flapped still routes (better a flaky hop than no hop at all;
    /// the committed-replay layers catch a dishonest node separately).
    pub fn filter_alive(&self, alive: &[String]) -> Vec<String> {
        let kept: Vec<String> = alive
            .iter()
            .filter(|a| !self.excluded.contains(*a))
            .cloned()
            .collect();
        if kept.is_empty() {
            alive.to_vec()
        } else {
            kept
        }
    }
}

/// The resilient walk over [`plan_route`]'s sequence: like [`execute_route`]
/// but a failure need not kill the answer —
///
/// - **Dead-hop reassignment (the ring's own rule):** the failed hop's
///   layers go to `reassign(layers, dead_owner)`, which the caller wires to
///   [`ShardMap::layer_ownership`](crate::replication::ShardMap::layer_ownership)
///   over the alive set minus the dead owner — dropping a node re-pins
///   only its layers, never the whole route. The replacement sub-hops are
///   walked in order (front of the queue), bounded by [`REASSIGN_BUDGET`].
/// - **Honest parity between hops:** every failure and success feeds
///   `stability` (`note_*`), so repeated deaths exclude a flapping owner
///   from *future* route plans without this walk's outcomed changing.
/// - **Partial answers:** when `reassign` yields nothing (nobody alive to
///   take the layers, or the caller refuses), the hop's layers are skipped
///   and typed into [`RouteResult::degraded`] — the router walks on and the
///   caller receives the residual as far as the chain carried it, with the
///   lost layers named.
#[allow(clippy::too_many_arguments)]
pub async fn execute_route_resilient(
    model_id: &str,
    req_id: &str,
    route: Vec<PipelineHop>,
    mut activation: Vec<f32>,
    local_run: impl Fn(&[f32], &[usize]) -> anyhow::Result<Vec<f32>>,
    remote_run: impl Fn(
        &str,
        &str,
        Vec<f32>,
        Duration,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = anyhow::Result<(Vec<f32>, Option<String>)>> + Send>,
    >,
    rtt_of: impl Fn(&str) -> u64,
    base_timeout: Duration,
    // M3: the caller-wired per-hop budget hook (ms). A hop that overruns
    // it is typed into RouteResult::over_budget and the walk continues —
    // budgeting reports, never clamps (M3 scopes the enforcement to
    // observability; reordering/preemption is later roadmap). `None`
    // disables the check silently.
    budget_of: Option<impl Fn(&PipelineHop) -> u64>,
    // M4 enforcement: the WHOLE-WALK envelope (ms, wall-clock from walk
    // start). Past it, every remaining hop is degraded, NOT dialed — one
    // huge speculative run may not starve other askers with unbounded
    // mesh dials, and a partially-walked route returns the honest
    // hops-short answer the degraded walk already carries. The envelope
    // never interrupts an in-flight dial (the current hop always lands,
    // success or failure, its timing recorded); enforcement only refuses
    // to START new ones. `None` disables (no wall — each dial's own
    // deadline still bounds it).
    walk_budget_ms: Option<u64>,
    // §1 per-request expert budget (work units). Enforced like the wall:
    // never interrupts an in-flight attempt, refuses to START new REMOTE
    // ones past the caps and degrades the remaining REMOTE hops by name
    // with [`RouteResult::REQUEST_BUDGET_EXHAUSTED_REASON`] — local hops
    // stay free (the caps bound what one request spends on the MESH, not
    // what it may compute locally). `None` disables entirely; the walk
    // keeps [`RequestBudget::default`]-shaped generosity available to
    // callers that pass `Some`.
    request_budget: Option<&RequestBudget>,
    stability: &mut RouteStability,
    reassign: impl Fn(&[usize], Option<&str>) -> Option<Vec<PipelineHop>>,
) -> RouteResult {
    let walk_started = std::time::Instant::now();
    let mut pending: VecDeque<PipelineHop> = route.into();
    let mut reassign_budget = REASSIGN_BUDGET.saturating_add(pending.len());
    let mut degraded: Vec<DegradedHop> = Vec::new();
    let mut reassigned: Vec<ReassignedHop> = Vec::new();
    let mut attempts: Vec<AttemptTiming> = Vec::new();
    let mut over_budget: Vec<OverBudgetHop> = Vec::new();
    let mut hops_run: usize = 0;
    let mut budget_exhausted = false;
    // §1 per-request budget ledger: remote dials started, wire bytes
    // actually shipped (4 bytes per f32, in + out). Only REMOTE attempts
    // accrue — local hops are not mesh spend. Caps are normalized to ≥1
    // so a zero/absent cap never disables the guard by accident.
    let mut remote_attempts: usize = 0;
    let mut wire_bytes: u64 = 0;
    let caps = request_budget.map(
        |RequestBudget {
             max_remote_attempts,
             max_wire_bytes,
         }| {
            (
                max_remote_attempts.map(|n| n.max(1)),
                max_wire_bytes.map(|b| b.max(1)),
            )
        },
    );

    loop {
        // Enforcement (M4): past the whole-walk envelope, degrade every
        // remaining hop by name instead of dialing — the walk returns the
        // residual with the loss typed, exactly like a reassignment that
        // found no alive owner.
        if let Some(walk_budget_ms) = walk_budget_ms
            && !pending.is_empty()
            && walk_started.elapsed().as_millis() as u64 > walk_budget_ms
        {
            let overrun_ms = walk_started.elapsed().as_millis() as u64;
            while let Some(remaining) = pending.pop_front() {
                degraded.push(DegradedHop {
                    layers: remaining.layers.clone(),
                    node: remaining.node.clone(),
                    reason: format!(
                        "skipped: {} after {overrun_ms}ms of {walk_budget_ms}ms envelope",
                        RouteResult::BUDGET_EXHAUSTED_REASON
                    ),
                });
            }
            budget_exhausted = true;
            break;
        }
        // §1 per-request budget enforcement: past either cap, refuse to
        // START new remote attempts — degrade the remaining REMOTE hops
        // by name (locals stay free and keep walking). Mirrors the M4
        // wall: in-flight work always lands, nothing is abandoned; with
        // no remote hops left the guard is inert (the walk finishes on
        // local work without spending further mesh units).
        if let Some((max_hops, max_bytes)) = &caps
            && (max_hops.is_some_and(|h| remote_attempts >= h)
                || max_bytes.is_some_and(|b| wire_bytes >= b))
            && pending.iter().any(|h| h.node.is_some())
        {
            budget_exhausted = true;
            let mut kept: VecDeque<PipelineHop> = VecDeque::with_capacity(pending.len());
            while let Some(remaining) = pending.pop_front() {
                if remaining.node.is_some() {
                    degraded.push(DegradedHop {
                        layers: remaining.layers.clone(),
                        node: remaining.node.clone(),
                        reason: format!(
                            "skipped: {} after {remote_attempts} remote attempts / {wire_bytes}B of caps",
                            RouteResult::REQUEST_BUDGET_EXHAUSTED_REASON
                        ),
                    });
                } else {
                    kept.push_back(remaining);
                }
            }
            pending = kept;
        }
        let Some(hop) = pending.pop_front() else {
            break;
        };
        let remote = hop.node.is_some();
        if remote {
            remote_attempts += 1;
            wire_bytes = wire_bytes.saturating_add((activation.len() * 4) as u64);
        }
        let started = std::time::Instant::now();
        let outcome = attempt_one_hop(
            model_id,
            req_id,
            &hop,
            &activation,
            &local_run,
            &remote_run,
            &rtt_of,
            base_timeout,
        )
        .await;
        let elapsed_ms = started.elapsed().as_millis() as u64;
        let hop_budget_ms = budget_of.as_ref().map(|f| f(&hop));
        match outcome {
            Ok(out) => {
                attempts.push(AttemptTiming {
                    node: hop.node.clone(),
                    layers: hop.layers.clone(),
                    ms: elapsed_ms,
                    ok: true,
                });
                if let Some(budget_ms) = hop_budget_ms
                    && elapsed_ms > budget_ms
                {
                    over_budget.push(OverBudgetHop {
                        node: hop.node.clone(),
                        layers: hop.layers.clone(),
                        measured_ms: elapsed_ms,
                        budget_ms,
                    });
                }
                if remote {
                    wire_bytes = wire_bytes.saturating_add((out.len() * 4) as u64);
                }
                activation = out;
                hops_run += 1;
                if let Some(node) = &hop.node {
                    stability.note_success(node);
                }
            }
            Err(e) => {
                attempts.push(AttemptTiming {
                    node: hop.node.clone(),
                    layers: hop.layers.clone(),
                    ms: elapsed_ms,
                    ok: false,
                });
                let reason = e.to_string();
                if let Some(node) = &hop.node {
                    stability.note_failure(node);
                }
                let replacement = if reassign_budget > 0 {
                    reassign(&hop.layers, hop.node.as_deref())
                } else {
                    None
                };
                match replacement {
                    Some(sub) if !sub.is_empty() => {
                        reassign_budget -= 1;
                        reassigned.push(ReassignedHop {
                            layers: hop.layers.clone(),
                            was_node: hop.node.clone(),
                            reason: reason.clone(),
                        });
                        for replacement_hop in sub.into_iter().rev() {
                            pending.push_front(replacement_hop);
                        }
                    }
                    _ => {
                        degraded.push(DegradedHop {
                            layers: hop.layers.clone(),
                            node: hop.node.clone(),
                            reason: format!("skipped: {reason}"),
                        });
                    }
                }
            }
        }
    }
    RouteResult {
        activation,
        hops_run,
        degraded,
        reassigned,
        attempts,
        over_budget,
        budget_exhausted,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_store::ExpertWeights;
    use crate::verify::{result_commitment, verify_commitment};
    use std::time::Duration;

    const BASE: Duration = Duration::from_millis(500);

    fn map(entries: &[(usize, &str)]) -> BTreeMap<usize, String> {
        entries.iter().map(|(l, o)| (*l, o.to_string())).collect()
    }

    #[test]
    fn consecutive_same_owner_layers_merge_into_one_hop() {
        let m = map(&[(0, "a"), (1, "a"), (2, "b"), (3, "b"), (4, "c")]);
        let route = plan_route("self", &m);
        assert_eq!(
            route,
            vec![
                PipelineHop {
                    node: Some("a".into()),
                    layers: vec![0, 1]
                },
                PipelineHop {
                    node: Some("b".into()),
                    layers: vec![2, 3]
                },
                PipelineHop {
                    node: Some("c".into()),
                    layers: vec![4]
                },
            ]
        );
    }

    #[test]
    fn local_slices_are_none_hops_and_merge() {
        let m = map(&[(0, "self"), (1, "self"), (2, "other")]);
        let route = plan_route("self", &m);
        assert_eq!(
            route,
            vec![
                PipelineHop {
                    node: None,
                    layers: vec![0, 1]
                },
                PipelineHop {
                    node: Some("other".into()),
                    layers: vec![2]
                },
            ]
        );
    }

    #[test]
    fn all_local_is_one_hop_and_empty_is_empty() {
        let m = map(&[(0, "self"), (1, "self")]);
        let route = plan_route("self", &m);
        assert_eq!(route.len(), 1);
        assert_eq!(route[0].node, None);
        assert!(plan_route("self", &BTreeMap::new()).is_empty());
    }

    #[test]
    fn remote_start_hands_off_immediately() {
        // A route this node does not start still yields the full ordered
        // walk — M1's executor decides whether the local node initiates it
        // or receives it from a peer.
        let m = map(&[(0, "a"), (1, "self"), (2, "a")]);
        let route = plan_route("self", &m);
        assert_eq!(
            route
                .iter()
                .map(|h| h.node.clone().unwrap_or_else(|| "self".into()))
                .collect::<Vec<_>>(),
            vec!["a", "self", "a"],
            "non-contiguous owners must NOT merge: {route:?}"
        );
        assert_eq!(route[0].node.as_deref(), Some("a"));
        assert_eq!(route[0].layers, vec![0]);
    }

    // ---- M1: route execution ----

    /// One hop of a synthetic route; every hop is one synthetic expert
    /// forward keyed by the per-hop expert id (the real layer split arrives
    /// with the Colibri-side work — the executor is weights-agnostic).
    fn run_hop(expert_id: &str, x: &[f32]) -> anyhow::Result<Vec<f32>> {
        ExpertWeights::synthetic(expert_id, 8, 16).forward(x)
    }

    /// The remote-hop future shape the walker's adapters return.
    type RemoteHopFut = std::pin::Pin<
        Box<dyn std::future::Future<Output = anyhow::Result<(Vec<f32>, Option<String>)>> + Send>,
    >;

    /// The same deterministic expert run a live peer would execute, wrapped
    /// like [`remote_run`]'s contract and returning a REAL Phase 4
    /// commitment over (req_id, per-hop expert id, node, in, out).
    fn remote_ok(req_id: &'static str) -> impl Fn(&str, &str, Vec<f32>, Duration) -> RemoteHopFut {
        move |expert_id, node, input, _timeout| {
            let expert_id = expert_id.to_owned();
            let node = node.to_owned();
            let fut = async move {
                run_hop(&expert_id, &input).map(|out| {
                    let c = result_commitment(req_id, &expert_id, &node, &input, &out);
                    (out, Some(c))
                })
            };
            Box::pin(fut)
        }
    }

    #[tokio::test]
    async fn route_walks_all_hops_and_final_vector_matches_serial_run() {
        let model = "moe-v1";
        let req = "req-1";
        let m = map(&[(0, "self"), (1, "peer-a"), (2, "self"), (3, "peer-b")]);
        let route = plan_route("self", &m);
        assert_eq!(route.len(), 4, "alternating owners never merge");

        let input = vec![0.5_f32; 8];
        let out = execute_route(
            model,
            req,
            &route,
            input.clone(),
            |a, layers| {
                // The local slice is one layer-run expert.
                run_hop(
                    &hop_expert_id(
                        model,
                        &PipelineHop {
                            node: None,
                            layers: layers.to_vec(),
                        },
                    ),
                    a,
                )
            },
            remote_ok(req),
            |_| 40, // 40 ms peer RTT
            BASE,
        )
        .await
        .expect("every hop alive");

        // Serial reference run: same expert ids, same order, chained.
        let mut expect = input;
        for hop in &route {
            expect = run_hop(&hop_expert_id(model, hop), &expect).unwrap();
        }
        assert_eq!(out, expect, "residual stream must chain hop outputs");
    }

    #[tokio::test]
    async fn tampered_remote_hop_is_caught_by_the_per_hop_commitment() {
        // Middle hop computes a commitment over the HONEST body, then
        // mutates the body it actually ships — the per-hop recomputation
        // must reject the mismatch and abort the route, naming the owner.
        let model = "moe-v1";
        let req = "req-2";
        let m = map(&[(0, "self"), (1, "peer-a"), (2, "peer-b")]);
        let route = plan_route("self", &m);

        let mut input = vec![0.25_f32; 8];
        input[0] = 1.0;
        let err = execute_route(
            model,
            req,
            &route,
            input,
            |a, layers| {
                run_hop(
                    &hop_expert_id(
                        model,
                        &PipelineHop {
                            node: None,
                            layers: layers.to_vec(),
                        },
                    ),
                    a,
                )
            },
            move |expert_id, node, activation, _t| {
                let expert_id = expert_id.to_owned();
                let node = node.to_owned();
                let tamper = node == "peer-a";
                Box::pin(async move {
                    let mut out = run_hop(&expert_id, &activation)?;
                    if tamper {
                        let honest = out.clone();
                        let c = result_commitment(req, &expert_id, &node, &activation, &honest);
                        out[1] += 0.5; // ship body ≠ committed body
                        return Ok((out, Some(c)));
                    }
                    let c = result_commitment(req, &expert_id, &node, &activation, &out);
                    Ok((out, Some(c)))
                })
            },
            |_| 20,
            BASE,
        )
        .await
        .expect_err("tampered hop aborts the route");
        let RouteError {
            hop_index,
            node,
            reason,
            hops_completed,
            ..
        } = &err;
        assert_eq!(*hop_index, 1, "the tampering middle hop faulted");
        assert_eq!(node.as_deref(), Some("peer-a"));
        assert_eq!(*hops_completed, 1, "local hop 0 completed first");
        assert!(
            reason.contains("commitment mismatch"),
            "typed reason: {reason}"
        );
    }

    #[tokio::test]
    async fn honest_remote_hop_commitment_passes_and_tampering_fails() {
        let model = "x";
        let req = "r";
        let input = vec![0.5_f32; 8];
        let route = vec![PipelineHop {
            node: Some("peer-a".into()),
            layers: vec![7],
        }];
        // Honest case passes.
        let ok = execute_route(
            model,
            req,
            &route,
            input.clone(),
            |a, _| run_hop("fallback", a),
            remote_ok(req),
            |_| 15,
            BASE,
        )
        .await
        .expect("honest commitment verifies");
        assert_eq!(
            ok,
            run_hop(&hop_expert_id(model, &route[0]), &input).unwrap()
        );

        // Tampered output with commitment over the ORIGINAL body → mismatch.
        let variant: Vec<f32> = ok.iter().map(|v| v * 2.0).collect();
        let c = result_commitment(req, &hop_expert_id(model, &route[0]), "peer-a", &input, &ok);
        assert!(
            !verify_commitment(
                req,
                &hop_expert_id(model, &route[0]),
                "peer-a",
                &input,
                &variant,
                &c
            ),
            "altered body must fail the per-hop commitment"
        );
    }

    #[tokio::test]
    async fn remote_failure_aborts_with_typed_route_error() {
        let route = vec![
            PipelineHop {
                node: Some("a".into()),
                layers: vec![0],
            },
            PipelineHop {
                node: Some("b".into()),
                layers: vec![1],
            },
            PipelineHop {
                node: Some("c".into()),
                layers: vec![2],
            },
        ];
        let err = execute_route(
            "m",
            "req",
            &route,
            vec![0.0_f32; 4],
            |a, _| run_hop("m#local", a),
            |_e, node, _x, _t| {
                let dead = node == "b";
                Box::pin(async move {
                    if dead {
                        anyhow::bail!("peer unreachable")
                    }
                    Ok((vec![0.0_f32; 4], None))
                })
            },
            |_| 30,
            BASE,
        )
        .await
        .expect_err("dead hop aborts");
        let RouteError {
            hop_index,
            node,
            layers,
            reason,
            hops_completed,
        } = &err;
        assert_eq!(*hop_index, 1);
        assert_eq!(node.as_deref(), Some("b"));
        assert_eq!(layers.as_slice(), &[1_usize][..]);
        assert_eq!(*hops_completed, 1, "hop 0 completed before the failure");
        assert_eq!(reason, "peer unreachable");
        assert!(
            err.to_string().contains("pipeline hop 1/b"),
            "typed display: {err}"
        );
    }

    #[test]
    fn hop_timeout_accounts_for_rtt() {
        assert_eq!(
            hop_timeout(Duration::from_millis(100), 0),
            Duration::from_millis(100)
        );
        assert_eq!(
            hop_timeout(Duration::from_millis(100), 40),
            Duration::from_millis(220)
        );
        // Monotonic in RTT, and the saturating path never panics.
        let huge = hop_timeout(Duration::from_millis(100), u64::MAX);
        assert!(huge > Duration::from_millis(100));
        assert_eq!(
            huge,
            Duration::from_nanos(u64::MAX) + Duration::from_millis(100)
        );
    }

    #[tokio::test]
    async fn empty_route_returns_input_unchanged() {
        assert_eq!(
            execute_route(
                "m",
                "r",
                &[],
                vec![1.0_f32; 4],
                |_, _| unreachable!(),
                |_, _, _, _| Box::pin(async { unreachable!() }),
                |_| u64::MAX,
                BASE
            )
            .await
            .unwrap(),
            vec![1.0_f32; 4]
        );
    }

    // ---- M2: degradation + hysteresis ----

    /// Local executor used by the M2 resilience tests: keyed by the same
    /// per-hop expert id so replacement hops answer deterministically.
    fn local_of(model: &'static str) -> impl Fn(&[f32], &[usize]) -> anyhow::Result<Vec<f32>> {
        move |a, layers| {
            run_hop(
                &hop_expert_id(
                    model,
                    &PipelineHop {
                        node: None,
                        layers: layers.to_vec(),
                    },
                ),
                a,
            )
        }
    }

    #[tokio::test]
    async fn dead_hop_layers_reassigned_by_the_ring_rule_and_route_completes() {
        let model = "moe-v2";
        let req = "req-m2-1";
        let route = vec![
            PipelineHop {
                node: None,
                layers: vec![0],
            },
            PipelineHop {
                node: Some("peer-a".into()),
                layers: vec![1, 2],
            },
            PipelineHop {
                node: Some("peer-b".into()),
                layers: vec![3],
            },
        ];

        // The caller's reassign closure IS the ring's own rule: layers re-
        // pinned over the alive set minus the dead owner, merged per owner.
        let reassign = |layers: &[usize], dead: Option<&str>| {
            assert_eq!(dead, Some("peer-a"), "only the dead owner is dropped");
            if layers == [1_usize, 2] {
                Some(vec![PipelineHop {
                    node: Some("peer-b".into()),
                    layers: vec![1, 2],
                }])
            } else {
                None
            }
        };
        let mut stability = RouteStability::new();

        let mut input = vec![0.5_f32; 8];
        input[3] = -0.25;
        let result = execute_route_resilient(
            model,
            req,
            route,
            input,
            local_of(model),
            move |expert_id, node, activation, _t| {
                let expert_id = expert_id.to_owned();
                let node = node.to_owned();
                let fut = async move {
                    if node == "peer-a" {
                        // The production closure honors the hop deadline from
                        // the QUIC dispatch; here the dead peer just fails.
                        anyhow::bail!("peer unreachable");
                    }
                    run_hop(&expert_id, &activation).map(|out| {
                        let c = result_commitment(req, &expert_id, &node, &activation, &out);
                        (out, Some(c))
                    })
                };
                Box::pin(fut)
            },
            |n| if n == "peer-a" { 5 } else { 40 },
            BASE,
            None::<fn(&PipelineHop) -> u64>,
            None,
            None,
            &mut stability,
            reassign,
        )
        .await;

        // Complete walk: the failed hop was RE-HOSTED, not skipped.
        assert!(
            result.degraded.is_empty(),
            "no honest losses: {:?}",
            result.degraded
        );
        assert_eq!(result.reassigned.len(), 1, "the dead hop was re-hosted");
        assert_eq!(result.reassigned[0].layers, vec![1, 2]);
        assert_eq!(result.reassigned[0].was_node.as_deref(), Some("peer-a"));
        // hops_run counts successful executions: the dead attempt is not one.
        assert_eq!(result.hops_run, 3, "local 0, replacement 1-2, tail 3");

        // Serial reference: layer 1-2 ran under the replacement id.
        let serial = vec![
            PipelineHop {
                node: None,
                layers: vec![0],
            },
            PipelineHop {
                node: Some("peer-b".into()),
                layers: vec![1, 2],
            },
            PipelineHop {
                node: Some("peer-b".into()),
                layers: vec![3],
            },
        ];
        let mut expect = vec![0.5_f32; 8];
        expect[3] = -0.25;
        for hop in &serial {
            expect = run_hop(&hop_expert_id(model, hop), &expect).unwrap();
        }
        assert_eq!(
            result.activation, expect,
            "reassigned walk must equal the re-planned serial walk"
        );
    }

    #[tokio::test]
    async fn unrecoverable_hop_walks_on_with_an_honest_partial_answer() {
        let model = "moe-v3";
        let req = "req-m2-2";
        let route = vec![
            PipelineHop {
                node: None,
                layers: vec![0],
            },
            PipelineHop {
                node: Some("gone".into()),
                layers: vec![1],
            },
            PipelineHop {
                node: Some("peer-b".into()),
                layers: vec![2],
            },
        ];
        let mut stability = RouteStability::new();

        let input = vec![0.5_f32; 8];
        let result = execute_route_resilient(
            model,
            req,
            route,
            input,
            local_of(model),
            move |expert_id, node, activation, _t| {
                let expert_id = expert_id.to_owned();
                let node = node.to_owned();
                let fut = async move {
                    if node == "gone" {
                        anyhow::bail!("peer unreachable")
                    }
                    run_hop(&expert_id, &activation).map(|out| (out, None))
                };
                Box::pin(fut)
            },
            |_| 20,
            BASE,
            None::<fn(&PipelineHop) -> u64>,
            None,
            None,
            &mut stability,
            |_layers, _dead| None, // nobody alive to take the layers
        )
        .await;

        // The router walked ON past the lost hop:
        assert_eq!(result.hops_run, 2, "local hop 0 and peer-b hop 2 ran");
        assert_eq!(result.degraded.len(), 1, "exactly the lost layer typed");
        assert_eq!(result.degraded[0].layers, vec![1]);
        assert_eq!(result.degraded[0].node.as_deref(), Some("gone"));
        assert!(result.degraded[0].reason.contains("skipped"));
        assert!(result.degraded[0].reason.contains("peer unreachable"));
        assert!(!result.is_complete());

        // Honest hops-short: residual = local(0) then peer-b(2) — layer 1
        // skipped inside the chain deterministically, not silently dropped.
        let local0 = run_hop(
            &hop_expert_id(
                model,
                &PipelineHop {
                    node: None,
                    layers: vec![0],
                },
            ),
            &[0.5_f32; 8],
        )
        .unwrap();
        let expect_b2 = run_hop(
            &hop_expert_id(
                model,
                &PipelineHop {
                    node: Some("peer-b".into()),
                    layers: vec![2],
                },
            ),
            &local0,
        )
        .unwrap();
        assert_eq!(result.activation, expect_b2);
    }

    #[test]
    fn flapping_owner_is_dampened_by_two_line_hysteresis() {
        let mut stability = RouteStability::new();
        assert!(!stability.is_excluded("flapper"));
        stability.note_failure("flapper");
        stability.note_failure("flapper");
        stability.note_failure("flapper");
        assert!(
            stability.is_excluded("flapper"),
            "raise at {HOP_EXCLUDE_SCORE}"
        );

        // One success only decays — no instant re-inclusion mid-flap.
        stability.note_success("flapper");
        assert!(
            stability.is_excluded("flapper"),
            "score 2 sits in the hysteresis band"
        );
        stability.note_success("flapper");
        assert!(
            !stability.is_excluded("flapper"),
            "cleared below {HOP_CLEAR_SCORE}"
        );

        // Unknown nodes are never excluded; healthy peers are untouched.
        assert!(!stability.is_excluded("stranger"));
        stability.note_success("stranger");
        assert!(!stability.is_excluded("stranger"));
    }

    #[test]
    fn filter_alive_drops_excluded_but_never_empties_the_set() {
        let mut stability = RouteStability::new();
        for _ in 0..3 {
            stability.note_failure("a");
        }
        let alive = vec!["a".to_string(), "b".to_string()];
        assert_eq!(stability.filter_alive(&alive), vec!["b".to_string()]);

        // Everyone flapped: route anyway (a isogenic flaky hop beats no hop).
        for _ in 0..3 {
            stability.note_failure("b");
        }
        assert_eq!(
            stability.filter_alive(&alive),
            alive,
            "never hand back an empty routing set"
        );
    }

    #[tokio::test]
    async fn walk_failures_feed_the_stability_tracker() {
        let model = "moe-v4";
        let req = "req-m2-3";
        // Three hops all owned by the same dead node — the walk notes each.
        let route = vec![
            PipelineHop {
                node: Some("dead".into()),
                layers: vec![0],
            },
            PipelineHop {
                node: Some("dead".into()),
                layers: vec![1],
            },
            PipelineHop {
                node: Some("dead".into()),
                layers: vec![2],
            },
        ];
        let mut stability = RouteStability::new();
        let result = execute_route_resilient(
            model,
            req,
            route,
            vec![0.0_f32; 4],
            local_of(model),
            |_e, node, _x, _t| {
                let dead = node.to_owned();
                Box::pin(async move {
                    anyhow::ensure!(dead != "dead", "peer unreachable");
                    Ok::<_, anyhow::Error>((vec![0.0_f32; 4], None))
                })
            },
            |_| 20,
            BASE,
            None::<fn(&PipelineHop) -> u64>,
            None,
            None,
            &mut stability,
            |_l, _d| None,
        )
        .await;
        assert_eq!(result.degraded.len(), 3);
        assert!(
            stability.is_excluded("dead"),
            "3 walk failures must raise the flag"
        );
    }

    // ---- M3: latency budgeting ----

    #[test]
    fn route_budget_orders_by_measured_demand_and_floors_at_rtt() {
        let route = vec![
            PipelineHop {
                node: Some("peer-a".into()),
                layers: vec![0],
            },
            PipelineHop {
                node: Some("peer-b".into()),
                layers: vec![1],
            },
            PipelineHop {
                node: None,
                layers: vec![2],
            },
        ];
        let budgets = plan_route_budget(
            &route,
            |hop| match hop.node.as_deref() {
                Some("peer-b") => 7,
                Some("peer-a") => 3,
                _ => 0,
            },
            |n| if n == "peer-a" { 400 } else { 20 },
            Duration::from_millis(100),
            10_000,
        );
        // The measured-activation ORDER is the pricing contract: most
        // demanded hop first; the tie between the two zero-demand hops
        // keeps the route's original order (peer-a before the local hop).
        assert_eq!(
            budgets
                .iter()
                .map(|b| b.layers.as_slice())
                .collect::<Vec<_>>(),
            vec![&[1_usize][..], &[0_usize][..], &[2_usize][..]]
        );
        // Exact arithmetic: floors a=1300 (100 + 3×400), b=160, local=100;
        // remaining 8440 split by demand 3:7:0 → 2532:5908:0.
        let by_layer = |layer: usize| {
            budgets
                .iter()
                .find(|b| b.layers == [layer][..])
                .unwrap()
                .budget_ms
        };
        assert_eq!(by_layer(1), 6_068, "hot hop wins the demand share");
        assert_eq!(by_layer(0), 3_832, "floored hop keeps its RTT deadline");
        assert_eq!(by_layer(2), 100, "zero-demand hop keeps only its floor");
    }

    #[test]
    fn equal_demand_splits_the_remaining_budget_evenly() {
        let route = vec![
            PipelineHop {
                node: None,
                layers: vec![0],
            },
            PipelineHop {
                node: Some("a".into()),
                layers: vec![1],
            },
            PipelineHop {
                node: Some("b".into()),
                layers: vec![2],
            },
        ];
        let budgets = plan_route_budget(
            &route,
            |_| 0,
            |_| 10, // floors: 100 + 130 + 130 = 360
            Duration::from_millis(100),
            1_360,
        );
        // remaining 1000 across 3 equal-demand hops = 333 each (integer
        // division; the 1 ms truncation is tolerance, never a giveaway):
        // local 433, remotes 463 each. Every hop keeps its RTT floor.
        assert_eq!(budgets[0].budget_ms, 433);
        assert_eq!(budgets[1].budget_ms, 463);
        assert_eq!(budgets[2].budget_ms, 463);
    }

    #[tokio::test]
    async fn over_budget_hops_are_typed_into_the_result() {
        let route = vec![
            PipelineHop {
                node: Some("slow".into()),
                layers: vec![0],
            },
            PipelineHop {
                node: Some("fast".into()),
                layers: vec![1],
            },
        ];
        let mut stability = RouteStability::new();
        let result = execute_route_resilient(
            "m",
            "req",
            route,
            vec![0.0_f32; 4],
            |_a, _l| Ok(vec![0.0_f32; 4]),
            move |_e, node, _x, _t| {
                let slow = node == "slow";
                Box::pin(async move {
                    if slow {
                        tokio::time::sleep(Duration::from_millis(30)).await;
                    }
                    Ok((vec![0.0_f32; 4], None))
                })
            },
            |_| 10,
            BASE,
            // Only the sleeper is priced against 1 ms; the unprioritized
            // fast hop gets u64::MAX so CI scheduling jitter can't flake it.
            Some(|hop: &PipelineHop| {
                if hop.node.as_deref() == Some("slow") {
                    1
                } else {
                    u64::MAX
                }
            }),
            None,
            None,
            &mut stability,
            |_l, _d| None,
        )
        .await;

        // The walk CONTINUED past the overrun (budgeting reports, never
        // clamps) and the overrun is typed with both durations.
        assert_eq!(result.hops_run, 2);
        assert_eq!(result.over_budget.len(), 1);
        assert_eq!(result.over_budget[0].node.as_deref(), Some("slow"));
        assert_eq!(result.over_budget[0].budget_ms, 1);
        assert!(
            result.over_budget[0].measured_ms > 20,
            "sleep(30ms) must measure above the 1 ms budget"
        );
        assert!(result.attempts.iter().all(|a| a.ok), "both hops ran clean");
        assert_eq!(result.attempts.len(), 2);
    }

    #[tokio::test]
    async fn attempts_are_timed_for_successes_and_failures_alike() {
        let route = vec![
            PipelineHop {
                node: None,
                layers: vec![0],
            },
            PipelineHop {
                node: Some("gone".into()),
                layers: vec![1],
            },
            PipelineHop {
                node: Some("peer-b".into()),
                layers: vec![2],
            },
        ];
        let mut stability = RouteStability::new();
        let result = execute_route_resilient(
            "m",
            "req",
            route,
            vec![0.0_f32; 4],
            |a, _l| Ok(a.to_vec()),
            move |_e, node, _x, _t| {
                let dead = node == "gone";
                Box::pin(async move {
                    if dead {
                        anyhow::bail!("peer unreachable")
                    }
                    Ok((vec![0.0_f32; 4], None))
                })
            },
            |_| 10,
            BASE,
            None::<fn(&PipelineHop) -> u64>,
            None,
            None,
            &mut stability,
            |_l, _d| None,
        )
        .await;

        // Attempts mirror the walk's real order, ok-flags honest: the
        // failed middle attempt keeps its (long-dead) wait on record.
        assert_eq!(result.attempts.len(), 3);
        assert!(result.attempts[0].ok);
        assert!(!result.attempts[1].ok);
        assert_eq!(result.attempts[1].node.as_deref(), Some("gone"));
        assert!(result.attempts[2].ok);
        assert_eq!(result.over_budget, Vec::new(), "no budgets → no overruns");
    }
}

/// M4 enforcement (Phase 3): the whole-walk budget is ENFORCED — a walk
/// that outruns its envelope degrades the remaining hops by name instead
/// of dialing them, and a walk that fits runs untouched.
#[cfg(test)]
mod budget_enforcement_tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const BASE: Duration = Duration::from_millis(500);

    /// The `remote_run` closure's boxed-future return, factored out so the
    /// test closures stay readable.
    type RunFut = std::pin::Pin<
        Box<dyn std::future::Future<Output = anyhow::Result<(Vec<f32>, Option<String>)>> + Send>,
    >;

    fn remote_route(n: usize) -> Vec<PipelineHop> {
        (0..n)
            .map(|i| PipelineHop {
                layers: vec![i],
                node: Some(format!("peer-{i}")),
            })
            .collect()
    }

    /// One huge run may not spend unbounded wall time: the envelope stops
    /// NEW dials (the in-flight hop always lands), and the tail is
    /// degraded with the shared, type-filterable reason.
    #[tokio::test]
    async fn walk_budget_degrades_the_tail_instead_of_dialing() {
        let mut stability = RouteStability::new();
        let dials = Arc::new(AtomicUsize::new(0));
        let route = remote_route(4);
        let remote_run = {
            let dials = dials.clone();
            move |_expert_id: &str, _node: &str, activation: Vec<f32>, _t: Duration| -> RunFut {
                let dials = dials.clone();
                Box::pin(async move {
                    dials.fetch_add(1, Ordering::SeqCst);
                    // Every dial takes ~150ms, against a walk envelope of
                    // ~250ms: honestly run, ~2 hops fit — never all four.
                    tokio::time::sleep(Duration::from_millis(150)).await;
                    Ok((activation, None))
                })
            }
        };
        let result = execute_route_resilient(
            "mod",
            "budget-walk",
            route,
            vec![0.5_f32; 8],
            |_, _| Ok(vec![0.5_f32; 8]),
            remote_run,
            |_| 1,
            BASE,
            None::<fn(&PipelineHop) -> u64>,
            Some(250),
            None,
            &mut stability,
            |_l, _d| None,
        )
        .await;

        // Enforcement fired: the tail was NOT dialed.
        assert!(result.budget_exhausted, "the envelope ran out mid-walk");
        assert!(
            result.hops_run >= 1,
            "at least the first hop landed inside the envelope"
        );
        assert_eq!(
            dials.load(Ordering::SeqCst),
            result.hops_run,
            "past the envelope no NEW dial may be issued"
        );
        let lost = 4 - result.hops_run;
        assert_eq!(
            result.degraded.len(),
            lost,
            "the tail was named, not dialed"
        );
        assert!(
            result
                .degraded
                .iter()
                .all(|d| d.reason.contains(RouteResult::BUDGET_EXHAUSTED_REASON)),
            "degraded with the budget reason: {:?}",
            result.degraded
        );
    }

    #[tokio::test]
    async fn walk_that_fits_its_envelope_runs_untouched() {
        let mut stability = RouteStability::new();
        let route = remote_route(4);
        let remote_run =
            move |_expert_id: &str, _node: &str, activation: Vec<f32>, _t: Duration| -> RunFut {
                Box::pin(async move {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    Ok((activation, None))
                })
            };
        let result = execute_route_resilient(
            "mod",
            "budget-fits",
            route,
            vec![0.5_f32; 8],
            |_, _| Ok(vec![0.5_f32; 8]),
            remote_run,
            |_| 1,
            BASE,
            None::<fn(&PipelineHop) -> u64>,
            Some(5_000),
            None,
            &mut stability,
            |_l, _d| None,
        )
        .await;
        assert!(!result.budget_exhausted);
        assert!(result.degraded.is_empty(), "{:?}", result.degraded);
        assert_eq!(
            result.hops_run, 4,
            "the whole route ran inside the envelope"
        );
    }

    /// §1 per-request budget: the hop cap stops the DIALS, not the
    /// latency — a walk that would issue more remote attempts than its
    /// cap degrades the tail with the typed reason. Deterministic (no
    /// timing involved: the cap is a count).
    #[tokio::test]
    async fn request_hop_cap_degrades_the_tail_without_dialing() {
        let mut stability = RouteStability::new();
        let dials = Arc::new(AtomicUsize::new(0));
        let route = remote_route(4);
        let remote_run = {
            let dials = dials.clone();
            move |_expert_id: &str, _node: &str, activation: Vec<f32>, _t: Duration| -> RunFut {
                let dials = dials.clone();
                Box::pin(async move {
                    dials.fetch_add(1, Ordering::SeqCst);
                    Ok((activation, None))
                })
            }
        };
        let budget = RequestBudget {
            max_remote_attempts: Some(2),
            max_wire_bytes: None,
        };
        let result = execute_route_resilient(
            "mod",
            "req-hop-cap",
            route,
            vec![0.5_f32; 8],
            |_, _| Ok(vec![0.5_f32; 8]),
            remote_run,
            |_| 1,
            BASE,
            None::<fn(&PipelineHop) -> u64>,
            None,
            Some(&budget),
            &mut stability,
            |_l, _d| None,
        )
        .await;

        assert!(result.budget_exhausted, "the hop cap ran out mid-walk");
        assert_eq!(
            dials.load(Ordering::SeqCst),
            2,
            "no dial may be issued past the attempt cap"
        );
        assert_eq!(result.hops_run, 2, "the two in-cap attempts landed");
        assert_eq!(result.degraded.len(), 2, "the tail was named, not dialed");
        assert!(
            result.degraded.iter().all(|d| d
                .reason
                .contains(RouteResult::REQUEST_BUDGET_EXHAUSTED_REASON)),
            "degraded with the per-request reason: {:?}",
            result.degraded
        );
    }

    /// §1 per-request budget: the wire-byte cap counts exactly what the
    /// walk ships (4 B per f32, in + out) — with 8-float activations that
    /// is 64 B per completed hop, so a 64 B cap finishes precisely one
    /// hop and stops the flood.
    #[tokio::test]
    async fn request_wire_cap_stops_the_flood() {
        let mut stability = RouteStability::new();
        let route = remote_route(4);
        let remote_run =
            move |_expert_id: &str, _node: &str, activation: Vec<f32>, _t: Duration| -> RunFut {
                Box::pin(async move {
                    // Output same size as the input: 8 floats = 32 B.
                    Ok((activation, None))
                })
            };
        let budget = RequestBudget {
            max_remote_attempts: None,
            max_wire_bytes: Some(64),
        };
        let result = execute_route_resilient(
            "mod",
            "req-wire-cap",
            route,
            vec![0.5_f32; 8],
            |_, _| Ok(vec![0.5_f32; 8]),
            remote_run,
            |_| 1,
            BASE,
            None::<fn(&PipelineHop) -> u64>,
            None,
            Some(&budget),
            &mut stability,
            |_l, _d| None,
        )
        .await;

        assert!(result.budget_exhausted, "the wire cap ran out mid-walk");
        assert_eq!(
            result.hops_run, 1,
            "64 B = exactly one 8-float hop (32 B in + 32 B out)"
        );
        assert_eq!(result.degraded.len(), 3);
        assert!(
            result.degraded[0].reason.contains("64B of caps"),
            "the reason names the byte ledger: {:?}",
            result.degraded[0].reason
        );
    }

    /// The DEFAULT budget exists to stop unbounded speculative replay,
    /// never to squeeze an honest walk: default caps fit a full
    /// REASSIGN_BUDGET storm plus a four-hop route with room to spare.
    #[tokio::test]
    async fn default_request_budget_never_squeezes_an_honest_walk() {
        assert_eq!(DEFAULT_MAX_REMOTE_ATTEMPTS, 64);
        assert_eq!(
            DEFAULT_REQUEST_WIRE_BYTES,
            256 * 1024 * 1024,
            "the flood guard is not a squeeze"
        );
        let mut stability = RouteStability::new();
        let route = remote_route(4);
        let remote_run =
            move |_expert_id: &str, _node: &str, activation: Vec<f32>, _t: Duration| -> RunFut {
                Box::pin(async move { Ok((activation, None)) })
            };
        let result = execute_route_resilient(
            "mod",
            "req-default",
            route,
            vec![0.5_f32; 8],
            |_, _| Ok(vec![0.5_f32; 8]),
            remote_run,
            |_| 1,
            BASE,
            None::<fn(&PipelineHop) -> u64>,
            None,
            Some(&RequestBudget::default()),
            &mut stability,
            |_l, _d| None,
        )
        .await;

        assert!(!result.budget_exhausted, "no cap trips on an honest walk");
        assert!(result.degraded.is_empty(), "{:?}", result.degraded);
        assert_eq!(result.hops_run, 4);
    }

    /// Local hops spend nothing on the mesh: past a tripped cap the walk
    /// still finishes its LOCAL hops and refuses only the remote tail
    /// (the caps bound mesh spend, not local compute).
    #[tokio::test]
    async fn tripped_cap_leaves_local_hops_free() {
        let mut stability = RouteStability::new();
        // Route: local, remote, remote, local, remote — the cap trips
        // after the two middle hops; the trailing local hop still runs,
        // while the trailing REMOTE hop is refused by name.
        let route = vec![
            PipelineHop {
                layers: vec![0],
                node: None,
            },
            PipelineHop {
                layers: vec![1],
                node: Some("peer-1".into()),
            },
            PipelineHop {
                layers: vec![2],
                node: Some("peer-2".into()),
            },
            PipelineHop {
                layers: vec![3],
                node: None,
            },
            PipelineHop {
                layers: vec![4],
                node: Some("peer-4".into()),
            },
        ];
        let remote_run =
            move |_expert_id: &str, _node: &str, activation: Vec<f32>, _t: Duration| -> RunFut {
                Box::pin(async move { Ok((activation, None)) })
            };
        let budget = RequestBudget {
            max_remote_attempts: Some(2),
            max_wire_bytes: None,
        };
        let result = execute_route_resilient(
            "mod",
            "req-local-free",
            route,
            vec![0.5_f32; 8],
            |_, _| Ok(vec![0.5_f32; 8]),
            remote_run,
            |_| 1,
            BASE,
            None::<fn(&PipelineHop) -> u64>,
            None,
            Some(&budget),
            &mut stability,
            |_l, _d| None,
        )
        .await;

        assert!(result.budget_exhausted, "the cap tripped mid-walk");
        assert_eq!(
            result.hops_run, 4,
            "local =1, the two remote hops, then the trailing local = 4"
        );
        assert_eq!(
            result.degraded.len(),
            1,
            "only the trailing remote was refused: {:?}",
            result.degraded
        );
        assert_eq!(
            result.degraded[0].node.as_deref(),
            Some("peer-4"),
            "the trailing remote was refused by name"
        );
        assert!(
            result.degraded[0]
                .reason
                .contains(RouteResult::REQUEST_BUDGET_EXHAUSTED_REASON)
        );
    }
}
