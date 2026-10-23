//! Expert activation statistics + hot/cold split (Phase 2 replication).
//!
//! The mesh's replication budget should follow *demand*, not guesses: the
//! spec calls for replicating the top 10–20 % most-activated experts and
//! sharding the long tail. That needs one cheap, honest input — how often
//! each expert is actually asked for.
//!
//! This module is that input, deliberately tiny:
//! - `record()` counts one activation (the serving path calls it for every
//!   expert query it answers, and the requester counts what it dispatches,
//!   so a node learns demand from both directions).
//! - `hot_set(percent)` ranks by count and takes the top slice (default
//!   15 %, clamped to the spec's 10–20 % band), deterministically
//!   tie-broken by expert id so two nodes with the same numbers agree.
//! - `snapshot()/load()` persist counts as JSON next to the expert store,
//!   because replication decisions that forget yesterday's traffic would
//!   oscillate.
//!
//! Counts are per-node observations, not authority: a node replicates what
//! *it* sees asked for, plus what its peers already replicate (the
//! replication factor is satisfied mesh-wide via the ads).

use anyhow::{Context as _, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Shared, internally-synchronised statistics: the serving path records
/// activations, the replication loop reads the hot set, both without
/// owning each other.
pub type SharedExpertStats = std::sync::Arc<std::sync::Mutex<ExpertStats>>;

/// Default hot fraction in percent — the middle of the spec's 10–20 %.
pub const DEFAULT_HOT_PERCENT: u32 = 15;
/// The spec's band. Values outside it are clamped rather than rejected:
/// config drift should not silently disable replication.
pub const MIN_HOT_PERCENT: u32 = 10;
pub const MAX_HOT_PERCENT: u32 = 20;

/// Sort key for the §1 DRAFT tier (phase 2, always-hot draft experts):
/// lowest layer first (the draft pass re-runs the front of the model
/// every decode step), then dense `E0` experts, then numbered experts;
/// ids that do not parse (`exp.drill.kat`-style synthetic names) sort
/// last alphabetically — so every node derives the same order from the
/// same numbers.
fn draft_key(id: &str) -> (u32, u8, u32, String) {
    let mut rank: u8 = 255;
    let mut layer_idx = u32::MAX;
    let mut expert_idx = u32::MAX;
    if let Some(rest) = id.strip_prefix('L')
        && let Some((layer, tail)) = rest.split_once('.')
        && let Ok(l) = layer.parse::<u32>()
    {
        layer_idx = l;
        if tail == "E0" {
            rank = 0;
        } else if let Some(m) = tail.strip_prefix('E')
            && let Ok(m) = m.parse::<u32>()
        {
            rank = 1;
            expert_idx = m;
        }
    }
    (layer_idx, rank, expert_idx, id.to_string())
}

/// The DRAFT tier: `percent` of the hosted experts (0 = off → empty),
/// drawn deterministically lowest-layer-first. Values above 50 are
/// clamped — a tier named "the always-hot slice" that could swallow the
/// whole store would just re-implement the cache budget by another name.
/// At least one expert whenever anything is hosted and the knob is on
/// (a fraction of a small store rounds up, mirroring `hot_set`).
pub fn draft_set(hosted: &[String], percent: u32) -> Vec<String> {
    if hosted.is_empty() || percent == 0 {
        return Vec::new();
    }
    let percent = percent.clamp(1, 50);
    let max_draft = (hosted.len() as u64 * percent as u64).div_ceil(100).max(1) as usize;
    let mut ordered: Vec<&String> = hosted.iter().collect();
    ordered.sort_by_key(|id| draft_key(id));
    ordered.into_iter().take(max_draft).cloned().collect()
}

/// Activation counts for the experts this node has seen requested.
#[derive(Debug, Default, Clone)]
pub struct ExpertStats {
    counts: HashMap<String, u64>,
    /// Total activations observed (sum of counts; kept so shares can be
    /// computed without re-summing a map on every decision).
    total: u64,
}

impl ExpertStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// Count one activation of `expert_id`.
    pub fn record(&mut self, expert_id: &str) {
        if expert_id.trim().is_empty() {
            return;
        }
        *self.counts.entry(expert_id.to_string()).or_insert(0) += 1;
        self.total += 1;
    }

    /// Activation count for one expert.
    pub fn count(&self, expert_id: &str) -> u64 {
        self.counts.get(expert_id).copied().unwrap_or(0)
    }

    /// Total activations observed.
    pub fn total(&self) -> u64 {
        self.total
    }

    /// How many distinct experts have been seen.
    pub fn distinct(&self) -> usize {
        self.counts.len()
    }

    /// Fraction of observed activations that went to this expert (0–1).
    pub fn share(&self, expert_id: &str) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.count(expert_id) as f64 / self.total as f64
    }

    /// All experts by descending count, ties broken by id ascending (so
    /// every node derives the same order from the same numbers).
    pub fn ranked(&self) -> Vec<(String, u64)> {
        let mut v: Vec<(String, u64)> =
            self.counts.iter().map(|(id, c)| (id.clone(), *c)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        v
    }

    /// The hot set: the top `percent` of the experts this node has seen
    /// activated, ranked by count — the spec's "top 10–20 % most-activated
    /// experts". The percent is clamped to that band, and at least one
    /// expert is hot whenever any activation has been seen (a fraction of
    /// a small set rounds up rather than to zero, so a young node still
    /// knows what to replicate first).
    ///
    /// The set is a fraction of *distinct observed experts*, not an
    /// activation-mass cover: the replication budget is about how many
    /// weight containers a node holds, and mass would let one dominant
    /// expert (55 % of traffic) stop the set at a single id.
    pub fn hot_set(&self, percent: u32) -> Vec<String> {
        if self.total == 0 {
            return Vec::new();
        }
        let percent = percent.clamp(MIN_HOT_PERCENT, MAX_HOT_PERCENT);
        let max_hot = (self.counts.len() as u64 * percent as u64).div_ceil(100);
        let max_hot = max_hot.max(1) as usize;
        self.ranked()
            .into_iter()
            .take(max_hot)
            .map(|(id, _)| id)
            .collect()
    }

    /// The cold remainder: everything observed that is not hot. These are
    /// the experts worth *sharding* rather than replicating.
    pub fn cold_set(&self, percent: u32) -> Vec<String> {
        let hot: std::collections::HashSet<String> = self.hot_set(percent).into_iter().collect();
        self.ranked()
            .into_iter()
            .map(|(id, _)| id)
            .filter(|id| !hot.contains(id))
            .collect()
    }

    /// Decide where an expert belongs, for logging/telemetry: hot experts
    /// get replicated (factor ≥ 2), cold ones get a single shard owner.
    pub fn classify(
        &self,
        expert_id: &str,
        percent: u32,
        replication_factor: usize,
    ) -> ExpertClass {
        if self.hot_set(percent).iter().any(|id| id == expert_id) {
            ExpertClass::Hot {
                desired_replicas: replication_factor.max(2),
            }
        } else {
            ExpertClass::Cold
        }
    }

    /// JSON snapshot (id → count) for persistence.
    pub fn snapshot(&self) -> serde_json::Value {
        let mut object = serde_json::Map::new();
        for (id, count) in self.ranked() {
            object.insert(id, serde_json::json!(count));
        }
        serde_json::json!({
            "total": self.total,
            "counts": object,
        })
    }

    /// Restore from [`snapshot`](Self::snapshot); unknown shapes degrade to
    /// "no statistics" rather than failing a node boot.
    pub fn from_snapshot(value: &serde_json::Value) -> Self {
        let mut stats = Self::new();
        if let Some(counts) = value.get("counts").and_then(|c| c.as_object()) {
            for (id, count) in counts {
                if let Some(n) = count.as_u64() {
                    stats.counts.insert(id.clone(), n);
                    stats.total += n;
                }
            }
        }
        // The counts are the source of truth: shares must never exceed 1,
        // so a hand-edited (smaller) total cannot shrink the denominator.
        if let Some(recorded) = value.get("total").and_then(|t| t.as_u64()) {
            stats.total = stats.total.max(recorded);
        }
        stats
    }

    /// Load from a JSON file; a missing file is simply "no statistics yet",
    /// and a corrupt one is logged and ignored (stats are advisory).
    pub fn load(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(text) => match serde_json::from_str::<serde_json::Value>(&text) {
                Ok(value) => Self::from_snapshot(&value),
                Err(e) => {
                    tracing::warn!(error = %e, path = %path.display(), "expert stats unreadable — starting fresh");
                    Self::new()
                }
            },
            Err(_) => Self::new(),
        }
    }

    /// Persist to `path` (parent dirs created). Best-effort: statistics
    /// must never break serving, so callers log and continue on error.
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        let text = serde_json::to_string_pretty(&self.snapshot())?;
        std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }
}

/// What the mesh should do with an expert, given local demand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertClass {
    /// Replicate to `desired_replicas` holders mesh-wide.
    Hot { desired_replicas: usize },
    /// Keep a single shard owner (cheaper than replicating the tail).
    Cold,
}

/// The layer index of a conventional `L<k>.E<n>` expert id. Ids that do
/// not carry the shape (builtin/ringer ids, hostile strings) have no
/// layer and never participate in prefetching.
pub fn expert_layer(expert_id: &str) -> Option<u32> {
    let rest = expert_id.strip_prefix('L')?;
    let dot = rest.find('.')?;
    rest[..dot].parse::<u32>().ok()
}

impl ExpertStats {
    /// Phase 3 speculative prefetch (FEATURES §13): the activation
    /// pipeline walks the model layer by layer, so the layer *after* the
    /// one just served is the natural prediction target. This returns
    /// the observed experts of layers `layer+1 ..= layer+horizon`, each
    /// ranked by demand (desc, ties by id — the same deterministic order
    /// every node derives), at most `limit` per layer. The caller decides
    /// which of these this node actually hosts — prefetch warms the
    /// local cache, it never pulls mesh weights.
    pub fn prefetch_candidates(
        &self,
        served_expert: &str,
        horizon: u32,
        limit: usize,
    ) -> Vec<String> {
        if horizon == 0 || limit == 0 {
            return Vec::new();
        }
        let start = match expert_layer(served_expert) {
            Some(l) => l,
            None => return Vec::new(),
        };
        let ranked = self.ranked(); // count desc, id asc
        let mut out = Vec::new();
        for l in start.saturating_add(1)..=start.saturating_add(horizon) {
            out.extend(
                ranked
                    .iter()
                    .filter(|(id, _)| expert_layer(id) == Some(l))
                    .take(limit)
                    .map(|(id, _)| id.clone()),
            );
        }
        out
    }
}

/// `~/.jerith-mesh/data/expert-stats.json` — beside the memory db, so a
/// node's whole local state lives under one directory.
pub fn default_stats_path() -> PathBuf {
    crate::config::NodeConfig::data_dir().join("expert-stats.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_path(tag: &str) -> PathBuf {
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        std::env::temp_dir().join(format!(
            "jerith-expert-stats-{}-{tag}-{}.json",
            std::process::id(),
            N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ))
    }

    /// §1 phase 2 (draft hot set): the tier is deterministic,
    /// lowest-layer-first, dense-E0 before numbered experts, non-layer
    /// synthetic ids last; the percent math mirrors hot_set (rounds up
    /// to ≥1) and clamps over-asks at 50 %.
    #[test]
    fn draft_set_is_layer_first_deterministic_and_bounded() {
        let hosted: Vec<String> = vec![
            "exp.drill.kat".into(), // synthetic: sorts last
            "L1.E2".into(),
            "L0.E1".into(),
            "L10.E0".into(), // numeric layer order, not lexicographic
            "L0.E0".into(),  // dense first within its layer
            "L1.E0".into(),
            "L2.E5".into(),
        ];
        // Off = empty; nothing at all hosted = empty.
        assert!(draft_set(&hosted, 0).is_empty());
        assert!(draft_set(&[], 15).is_empty());

        // 20 % of 7 → ceil 2: the two lowest-layer experts, dense E0
        // before the numbered one.
        assert_eq!(draft_set(&hosted, 20), vec!["L0.E0", "L0.E1"]);
        // 50 % of 7 → ceil 4 (and clamped 50 stays 50).
        assert_eq!(
            draft_set(&hosted, 50),
            vec!["L0.E0", "L0.E1", "L1.E0", "L1.E2"]
        );
        assert_eq!(draft_set(&hosted, 97), draft_set(&hosted, 50));
        // Full layer traversal order: L2 before L10 (numeric), the
        // synthetic name dead last.
        assert_eq!(
            draft_set(&hosted, 200).len(),
            draft_set(&hosted, 50).len(),
            "clamped to 50 % — never a whole-store swallow"
        );
        let all = {
            let mut v = draft_set(&hosted, 50);
            // the remainder in key order for the ordering assertion below
            v.extend(["L10.E0".into(), "L2.E5".into(), "exp.drill.kat".into()]);
            v
        };
        assert_eq!(
            draft_key("L2.E5"),
            (2, 1, 5, "L2.E5".into()),
            "numeric layer, numbered-expert rank"
        );
        assert!(draft_key("L2.E5") < draft_key("L10.E0"), "L2 < L10");
        assert!(
            draft_key("L10.E0") < draft_key("exp.drill.kat"),
            "synthetic last"
        );

        // Round-up: one hosted expert, 15 % → still hot.
        assert_eq!(draft_set(&["L0.E0".to_string()], 15), vec!["L0.E0"]);
        let _ = all; // ordering key assertions carry the proof
    }

    #[test]
    fn counts_and_shares_track_activations() {
        let mut s = ExpertStats::new();
        assert_eq!(s.total(), 0);
        assert!(s.hot_set(DEFAULT_HOT_PERCENT).is_empty());
        for _ in 0..8 {
            s.record("L1.E1");
        }
        for _ in 0..2 {
            s.record("L1.E2");
        }
        assert_eq!(s.total(), 10);
        assert_eq!(s.distinct(), 2);
        assert_eq!(s.count("L1.E1"), 8);
        assert!((s.share("L1.E1") - 0.8).abs() < 1e-9);
        assert!((s.share("L1.E2") - 0.2).abs() < 1e-9);
        // Unknown experts are zero, never an error.
        assert_eq!(s.count("nope"), 0);
        assert_eq!(s.share("nope"), 0.0);
        // Empty ids are ignored (a malformed request must not pollute stats).
        s.record("   ");
        assert_eq!(s.total(), 10);
    }

    #[test]
    fn hot_set_is_the_demand_mass_and_is_deterministic() {
        let mut s = ExpertStats::new();
        // 20 experts, Zipf-ish: E0 gets 50, E1 25, E2 15, the rest 1 each.
        for _ in 0..50 {
            s.record("L0.E0");
        }
        for _ in 0..25 {
            s.record("L0.E1");
        }
        for _ in 0..15 {
            s.record("L0.E2");
        }
        for i in 3..20 {
            s.record(&format!("L0.E{i}"));
        }
        let hot15 = s.hot_set(15);
        // 15 % of 20 distinct experts = 3 replicas max; those are the top 3.
        assert_eq!(hot15, vec!["L0.E0", "L0.E1", "L0.E2"]);
        // Deterministic across calls and across independently built copies.
        assert_eq!(hot15, s.hot_set(15));
        assert_eq!(hot15, ExpertStats::from_snapshot(&s.snapshot()).hot_set(15));

        // The band is enforced: 1 % clamps to 10 %, 90 % to 20 %.
        assert_eq!(s.hot_set(1), s.hot_set(MIN_HOT_PERCENT));
        assert_eq!(s.hot_set(90), s.hot_set(MAX_HOT_PERCENT));

        // Cold set is exactly the complement, in demand order.
        let cold = s.cold_set(15);
        assert_eq!(cold.len(), 17);
        assert!(cold.contains(&"L0.E19".to_string()));
        assert!(!cold.contains(&"L0.E0".to_string()));

        // Classification: hot experts want replicas, cold ones a single shard.
        assert_eq!(
            s.classify("L0.E0", 15, 3),
            ExpertClass::Hot {
                desired_replicas: 3
            }
        );
        assert_eq!(s.classify("L0.E19", 15, 3), ExpertClass::Cold);
        // A replication factor below 2 is nonsense — hot means ≥2 holders.
        assert_eq!(
            s.classify("L0.E0", 15, 1),
            ExpertClass::Hot {
                desired_replicas: 2
            }
        );
    }

    #[test]
    fn flat_demand_still_bounds_the_hot_set() {
        let mut s = ExpertStats::new();
        for i in 0..100 {
            s.record(&format!("L0.E{i:02}"));
        }
        // All equal: mass never accumulates past the cap, so the budget
        // (20 % of 100 = 20) is what decides.
        let hot = s.hot_set(20);
        assert_eq!(hot.len(), 20);
        assert_eq!(hot[0], "L0.E00", "ties break by id ascending");
        assert_eq!(s.cold_set(20).len(), 80);
    }

    #[test]
    fn stats_persist_and_reload_identically() {
        let path = tmp_path("persist");
        let mut s = ExpertStats::new();
        for _ in 0..7 {
            s.record("L5.E1");
        }
        s.record("L5.E9");
        s.save(&path).unwrap();
        let loaded = ExpertStats::load(&path);
        assert_eq!(loaded.total(), 8);
        assert_eq!(loaded.ranked(), s.ranked());
        assert_eq!(loaded.hot_set(15), s.hot_set(15));
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn expert_layer_parses_the_conventional_shape() {
        assert_eq!(expert_layer("L3.E17"), Some(3));
        assert_eq!(expert_layer("L0.E0"), Some(0));
        assert_eq!(expert_layer("L12.E100"), Some(12));
        // Everything else: None, never a panic — builtin/ringer ids and
        // hostile strings carry no layer and stay out of prefetching.
        assert_eq!(expert_layer("jerith-builtin/reverse2"), None);
        assert_eq!(expert_layer("E17"), None);
        assert_eq!(expert_layer("L..E1"), None);
        assert_eq!(expert_layer("L-1.E1"), None);
        assert_eq!(expert_layer(""), None);
    }

    #[test]
    fn prefetch_candidates_rank_the_next_layers_by_demand() {
        let mut s = ExpertStats::new();
        // Layer 0..3 demand: L2.E1 hottest, then L1.E2, L1.E1, tail.
        for _ in 0..6 {
            s.record("L2.E1");
        }
        for _ in 0..4 {
            s.record("L1.E2");
        }
        for _ in 0..3 {
            s.record("L1.E1");
        }
        for _ in 0..2 {
            s.record("L3.E9");
        }
        s.record("L0.E0");
        // Serving a layer-2 expert looks exactly one layer ahead: the two
        // observed layer-3 experts, demand-ranked, tie-broken by id.
        assert_eq!(s.prefetch_candidates("L2.E0", 1, 10), vec!["L3.E9"]);
        // Layer-0 expert, horizon 1: the top TWO layer-1 experts per layer
        // limit.
        assert_eq!(s.prefetch_candidates("L0.E0", 1, 2), vec!["L1.E2", "L1.E1"]);
        // Wider horizon walks further: layer 1 then layer 2, ranked each.
        assert_eq!(
            s.prefetch_candidates("L0.E0", 2, 10),
            vec!["L1.E2", "L1.E1", "L2.E1"]
        );
        // Layer limit truncates honestly.
        assert_eq!(s.prefetch_candidates("L0.E0", 1, 1), vec!["L1.E2"]);
        // A non-layer id has no prediction target.
        assert!(
            s.prefetch_candidates("jerith-builtin/reverse2", 1, 2)
                .is_empty()
        );
        // Zero knobs are an instant empty answer.
        assert!(s.prefetch_candidates("L0.E0", 0, 2).is_empty());
        assert!(s.prefetch_candidates("L0.E0", 1, 0).is_empty());
        // Empty stats likewise.
        let empty = ExpertStats::new();
        assert!(empty.prefetch_candidates("L0.E0", 1, 2).is_empty());
    }

    #[test]
    fn missing_or_corrupt_stats_never_fail_a_boot() {
        let missing = tmp_path("missing");
        assert_eq!(ExpertStats::load(&missing).total(), 0);
        let corrupt = tmp_path("corrupt");
        std::fs::write(&corrupt, "{not json").unwrap();
        assert_eq!(ExpertStats::load(&corrupt).total(), 0);
        // A hand-edited total below the real sum cannot inflate shares.
        let skewed = ExpertStats::from_snapshot(&serde_json::json!({
            "total": 1,
            "counts": {"a": 5}
        }));
        assert_eq!(skewed.total(), 5);
        assert!((skewed.share("a") - 1.0).abs() < 1e-9);
        std::fs::remove_file(&corrupt).unwrap();
    }
}
