//! "Recommended meshes for your hardware" (Phase 1, FEATURES §11).
//!
//! The setup flow already knows two things that answer "which mesh should I
//! join?": what this machine can actually run (`models::detect_capable_models`)
//! and which curated models currently have live peers (the bootstrap's
//! `/v1/models/counts`). This module joins them into a deterministic,
//! explainable ranking.
//!
//! Ranking order (documented so operators can argue with it):
//! 1. **Joinable beats empty** — a fine machine that would seed an empty mesh
//!    loses to a smaller machine that can join a live mesh today (the point
//!    of the mesh is finding peers; a lonely pick keeps you alone).
//! 2. **Bigger live meshes first** — among joinable candidates, more peers
//!    online wins (more experts and capacity live there).
//! 3. **Curated order as the stable tiebreak** — `detect_capable_models`
//!    returns models in curated order; equal-score candidates keep it, so
//!    suggestions never flip-flop between runs.
//!
//! Pure function, no I/O: the caller fetches counts and passes them in (an
//! unreachable bootstrap is just `&[]` — suggestions degrade to the hardware
//! pick, labelled as seed-me-ism).

use crate::models::ModelFit;

/// One recommended mesh: the model to run and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshSuggestion {
    pub model_id: String,
    pub peers_online: u64,
    pub rationale: String,
}

/// Rank capable models by live peer counts. `counts` is (model_id, online
/// nodes); missing ids count as zero peers. Not fitting models never appear.
pub fn recommended_meshes(fits: &[ModelFit], counts: &[(String, u64)]) -> Vec<MeshSuggestion> {
    let peers_of = |id: &str| {
        counts
            .iter()
            .find(|(cid, _)| cid == id)
            .map(|(_, c)| *c)
            .unwrap_or(0)
    };

    let mut out: Vec<(usize, MeshSuggestion)> = Vec::new();
    for (rank, fit) in fits.iter().enumerate() {
        if !fit.fits {
            continue;
        }
        let peers_online = peers_of(&fit.model.id);
        let rationale = if peers_online > 0 {
            format!(
                "{} node(s) live in this mesh — joining puts you to work today",
                peers_online
            )
        } else {
            "no peers online yet — you would be seeding this mesh".to_string()
        };
        out.push((
            rank,
            MeshSuggestion {
                model_id: fit.model.id.clone(),
                peers_online,
                rationale,
            },
        ));
    }
    out.sort_by(|a, b| b.1.peers_online.cmp(&a.1.peers_online).then(a.0.cmp(&b.0)));
    out.into_iter().map(|(_, s)| s).collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ModelModality, ModelSpec};

    fn spec(id: &str) -> ModelSpec {
        ModelSpec {
            id: id.to_string(),
            display_name: id.to_string(),
            resident_ram_bytes: 10_000_000_000,
            min_expert_cache_bytes: 2_000_000_000,
            disk_bytes: 45_000_000_000,
            modality: ModelModality::Text,
        }
    }

    fn fit(id: &str, fits: bool) -> ModelFit {
        ModelFit {
            model: spec(id),
            fits,
            reason: if fits { "runs fine" } else { "too big" }.to_string(),
        }
    }

    #[test]
    fn joinable_beats_empty_even_when_ranked_lower() {
        let fits = vec![fit("big-model", true), fit("small-model", true)];
        // The big model is the better fit (rank 0) but has NO live peers;
        // the smaller one has a live mesh.
        let counts = vec![("small-model".to_string(), 3u64)];
        let sug = recommended_meshes(&fits, &counts);
        assert_eq!(sug[0].model_id, "small-model");
        assert_eq!(sug[0].peers_online, 3);
        assert!(sug[0].rationale.contains("3 node(s) live"));

        // The seed-me case still appears, below, with the honest rationale.
        assert_eq!(sug[1].model_id, "big-model");
        assert_eq!(sug[1].peers_online, 0);
        assert!(sug[1].rationale.contains("seeding"));
    }

    #[test]
    fn bigger_live_meshes_rank_first() {
        let fits = vec![fit("m1", true), fit("m2", true)];
        let counts = vec![("m1".to_string(), 7u64), ("m2".to_string(), 12u64)];
        let sug = recommended_meshes(&fits, &counts);
        assert_eq!(sug[0].model_id, "m2");
        assert_eq!(sug[1].model_id, "m1");
    }

    #[test]
    fn curated_order_is_the_stable_tiebreak() {
        let fits = vec![fit("a-first", true), fit("b-second", true)];
        // No counts at all (unreachable bootstrap): both are seed picks and
        // keep curated order — suggestions never flip-flop between runs.
        let sug = recommended_meshes(&fits, &[]);
        assert_eq!(sug[0].model_id, "a-first");
        assert_eq!(sug[1].model_id, "b-second");
        assert_eq!(sug[0].peers_online, 0);
    }

    #[test]
    fn non_fitting_models_never_appear() {
        let fits = vec![fit("too-big", false), fit("right-size", true)];
        let counts = vec![("too-big".to_string(), 99u64)];
        let sug = recommended_meshes(&fits, &counts);
        assert_eq!(sug.len(), 1);
        assert_eq!(
            sug[0].model_id, "right-size",
            "a machine can't join a mesh its hardware can't run"
        );
    }

    #[test]
    fn missing_counts_entry_counts_as_zero_peers() {
        let fits = vec![fit("m", true)];
        let sug = recommended_meshes(&fits, &[("other-model".to_string(), 5u64)]);
        assert_eq!(sug[0].model_id, "m");
        assert_eq!(sug[0].peers_online, 0);
    }
}
