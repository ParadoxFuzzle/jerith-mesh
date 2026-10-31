//! Voluntary model sharing: which local models this node offers the mesh,
//! and what each model can actually do.
//!
//! Two halves, both deliberately conservative:
//!
//! 1. **Capability detection** (`detect`) labels a runtime model id with
//!    what it can serve — text, vision, video, embedding, image
//!    generation. Labels come from the id itself (the runtime's own tag:
//!    `moondream:latest`, `minicpm-v4.6:1b`, `nomic-embed-text`), not from
//!    a downloaded manifest, so a node announces nothing it has not
//!    actually pulled. Families we know to differ are distinguished:
//!    MiniCPM-V sees images **and** video, Moondream sees images only.
//! 2. **Opt-in sharing** (`SharingConfig`): an EMPTY list by default. A
//!    model is offered to the mesh only after the operator switches that
//!    exact id ON (dashboard toggle or `jerith config set sharing.models`).
//!    There is no "share everything" mode and no default-on switch.
//!
//! The advertisement built from this (`SharingConfig::offer`) rides the
//! peer ad, so nodes without vision/video/embedding can see who has it and
//! route their work there instead of answering blind.

use serde::{Deserialize, Serialize};

/// Cap on shared ids per node — a typo loop or a bloated list must not
/// bloat every advertisement (and every bootstrap row) forever.
pub const SHARED_MODELS_MAX: usize = 32;

/// What a model can serve. Serialized lowercase (`"text"`, `"vision"`, …)
/// so it travels in peer ads as stable JSON words.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    Text,
    Vision,
    /// Temporal understanding: frames of a video, not just one image.
    Video,
    Embedding,
    ImageGen,
}

impl Capability {
    pub fn as_str(self) -> &'static str {
        match self {
            Capability::Text => "text",
            Capability::Vision => "vision",
            Capability::Video => "video",
            Capability::Embedding => "embedding",
            Capability::ImageGen => "image_gen",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "text" => Some(Capability::Text),
            "vision" => Some(Capability::Vision),
            "video" => Some(Capability::Video),
            "embedding" => Some(Capability::Embedding),
            "image_gen" | "imagegen" => Some(Capability::ImageGen),
            _ => None,
        }
    }

    /// Canonical display/serialization order — the same order everywhere so
    /// ads, dashboards and tests never disagree about a capability set.
    pub fn all() -> [Capability; 5] {
        [
            Capability::Text,
            Capability::Vision,
            Capability::Video,
            Capability::Embedding,
            Capability::ImageGen,
        ]
    }
}

/// Embedding-only families. Checked first: `nomic-embed-text` must read as
/// an embedding model, never as a chat model with vision.
const EMBED_MARKERS: &[&str] = &[
    "embed",
    "bge-",
    "bge:",
    "bge-m",
    "nomic",
    "mxbai",
    "all-minilm",
    "minilm",
    "gte-",
    "e5-",
    "-e5",
    "jina-embed",
];

/// Local image-generation families (they produce images; they do not read
/// chat prompts as a text model would).
const IMAGEGEN_MARKERS: &[&str] = &[
    "flux",
    "sdxl",
    "stable-diffusion",
    "qwen-image",
    "dall-e",
    "playground-v",
];

/// Vision-capable families. Includes the plain `vision` / `-vl` / `:vl`
/// tokens so an operator's own tag (e.g. `my-org/vision-tune:latest`)
/// still labels correctly.
const VISION_MARKERS: &[&str] = &[
    "moondream",
    "llava",
    "bakllava",
    "minicpm-v",
    "minicpm-o",
    "qwen-vl",
    "qwen2-vl",
    "qwen2.5-vl",
    "qwen3-vl",
    "gemma3",
    "llama3.2-vision",
    "llama4",
    "pixtral",
    "internvl",
    "cogvlm",
    "smolvlm",
    "granite3.2-vision",
    "phi-4-multimodal",
    "mistral-small3",
    "vision",
    "-vl",
    ":vl",
    "vl:",
];

/// Families with temporal (video) understanding. MiniCPM-V and Qwen-VL
/// handle video frames; Moondream does not — the distinction the dashboard
/// has to make visible, so a requester with a video picks the right node.
const VIDEO_MARKERS: &[&str] = &[
    "minicpm-v",
    "minicpm-o",
    "qwen2-vl",
    "qwen2.5-vl",
    "qwen3-vl",
    "llava-video",
    "video-llava",
    "video",
];

fn has_any(hay: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| hay.contains(n))
}

/// Detect what one runtime model id can serve. Deterministic and offline:
/// the id is the only input (an id we do not recognize keeps the safe
/// `text` label and nothing more).
pub fn detect(model_id: &str) -> Vec<Capability> {
    let id = model_id.trim().to_ascii_lowercase();
    if id.is_empty() {
        return Vec::new();
    }
    let embedding = has_any(&id, EMBED_MARKERS);
    let imagegen = !embedding && has_any(&id, IMAGEGEN_MARKERS);
    let video = !embedding && !imagegen && has_any(&id, VIDEO_MARKERS);
    let vision = !embedding && !imagegen && (video || has_any(&id, VISION_MARKERS));
    Capability::all()
        .into_iter()
        .filter(|c| match c {
            Capability::Text => !embedding && !imagegen,
            Capability::Vision => vision,
            Capability::Video => video,
            Capability::Embedding => embedding,
            Capability::ImageGen => imagegen,
        })
        .collect()
}

/// Capability labels (canonical order) for one id, as wire/dashboard strings.
pub fn labels(model_id: &str) -> Vec<String> {
    detect(model_id)
        .into_iter()
        .map(|c| c.as_str().to_string())
        .collect()
}

/// `[sharing]` in config.toml — the models this operator opted to serve.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SharingConfig {
    /// Exact runtime model ids offered to the mesh. EMPTY = share nothing
    /// (the default: every model is opted out until switched ON).
    #[serde(default)]
    pub models: Vec<String>,
}

/// What this node publishes in its peer ad about shared models.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Offer {
    pub models: Vec<String>,
    /// Union of the shared models' capabilities, canonical order, deduped.
    pub capabilities: Vec<String>,
}

impl SharingConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.models.len() <= SHARED_MODELS_MAX,
            "[sharing] models: at most {SHARED_MODELS_MAX} ids (got {})",
            self.models.len()
        );
        for id in &self.models {
            anyhow::ensure!(
                !id.trim().is_empty(),
                "[sharing] models: blank entries are not model ids"
            );
            anyhow::ensure!(
                !id.chars().any(char::is_whitespace),
                "[sharing] models: {id:?} carries whitespace — model ids are single tokens"
            );
            anyhow::ensure!(
                id.len() <= 128,
                "[sharing] models: {id:?} is too long to be a model id"
            );
        }
        Ok(())
    }

    /// Deduplicated ids in operator order (whitespace trimmed). Used before
    /// advertising and before saving so a duplicate toggle cannot advertise
    /// the same model twice.
    pub fn normalized(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for id in &self.models {
            let id = id.trim();
            if !id.is_empty() && !out.iter().any(|x| x == id) {
                out.push(id.to_string());
            }
        }
        out
    }

    pub fn shares(&self, model_id: &str) -> bool {
        self.normalized().iter().any(|m| m == model_id)
    }

    /// Build the advertisement. Empty config → empty offer (nothing shared).
    pub fn offer(&self) -> Offer {
        let mut models = self.normalized();
        let mut seen: Vec<Capability> = Vec::new();
        for id in &models {
            for cap in detect(id) {
                if !seen.contains(&cap) {
                    seen.push(cap);
                }
            }
        }
        models.sort();
        let capabilities = Capability::all()
            .into_iter()
            .filter(|c| seen.contains(c))
            .map(|c| c.as_str().to_string())
            .collect();
        Offer {
            models,
            capabilities,
        }
    }
}

/// Validate a dashboard/CLI request to replace the shared set. Mirrors
/// `SharingConfig::validate` on the candidate list, so a refused write
/// never half-applies.
pub fn validate_models(models: &[String]) -> anyhow::Result<Vec<String>> {
    // Trim first: a dashboard text field/user paste may carry edges. Only
    // INTERNAL whitespace stays a refusal (a model id is one token).
    let cfg = SharingConfig {
        models: models.iter().map(|m| m.trim().to_string()).collect(),
    };
    cfg.validate()?;
    Ok(cfg.normalized())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_families_keep_their_real_differences() {
        // The operator's example: Moondream and MiniCPM both see images,
        // but only MiniCPM-V handles video.
        assert_eq!(labels("moondream:latest"), vec!["text", "vision"]);
        assert_eq!(labels("moondream2"), vec!["text", "vision"]);
        assert_eq!(labels("minicpm-v4.6:1b"), vec!["text", "vision", "video"]);
        assert_eq!(labels("qwen2.5-vl:7b"), vec!["text", "vision", "video"]);
        // Embedding models are labeled embedding, never chat/vision.
        assert_eq!(labels("nomic-embed-text"), vec!["embedding"]);
        assert_eq!(labels("bge-m3"), vec!["embedding"]);
        // Image generation is its own lane.
        assert_eq!(labels("flux.1-schnell"), vec!["image_gen"]);
        // Unknown tags stay honest: text only, no invented capability.
        assert_eq!(labels("qwen3:8b"), vec!["text"]);
        assert!(labels("   ").is_empty());
    }

    #[test]
    fn sharing_is_opt_in_and_empty_by_default() {
        let cfg = SharingConfig::default();
        assert!(cfg.models.is_empty(), "no model is shared until opted in");
        assert_eq!(cfg.offer(), Offer::default());
        assert!(!cfg.shares("moondream:latest"));
    }

    #[test]
    fn offer_dedupes_models_and_unions_capabilities_in_canonical_order() {
        let cfg = SharingConfig {
            models: vec![
                "minicpm-v4.6:1b".into(),
                "moondream:latest".into(),
                "minicpm-v4.6:1b".into(),
                "nomic-embed-text".into(),
            ],
        };
        let offer = cfg.offer();
        assert_eq!(
            offer.models,
            vec!["minicpm-v4.6:1b", "moondream:latest", "nomic-embed-text"],
            "sorted, deduped ids"
        );
        assert_eq!(
            offer.capabilities,
            vec!["text", "vision", "video", "embedding"]
        );
    }

    #[test]
    fn validate_refuses_blank_whitespace_and_oversized_lists() {
        assert!(
            SharingConfig {
                models: vec!["".into()]
            }
            .validate()
            .is_err()
        );
        assert!(
            SharingConfig {
                models: vec!["two words".into()]
            }
            .validate()
            .is_err()
        );
        let too_many = SharingConfig {
            models: (0..SHARED_MODELS_MAX + 1)
                .map(|i| format!("m{i}"))
                .collect(),
        };
        assert!(too_many.validate().is_err());
        let ok = validate_models(&[" moondream:latest ".into(), "moondream:latest".into()]);
        assert_eq!(ok.unwrap(), vec!["moondream:latest"]);
    }

    #[test]
    fn capability_words_roundtrip() {
        for cap in Capability::all() {
            assert_eq!(Capability::parse(cap.as_str()), Some(cap));
        }
        assert_eq!(Capability::parse("nope"), None);
    }
}
