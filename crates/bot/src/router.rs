//! Multi-model task swapping via Rust enums (FEATURES §17).
//!
//! A fresh lineup of tiny local models rides the same box (qwen3:0.6b,
//! bonsai-2:2b, tev1:0.8b, nemotron-3-nano:4b, qwen3:4b). Routing every
//! prompt to one model wastes them: sorting intent is a sub-second 0.6B
//! job, code generation wants a 4B. This module is the
//! zero-overhead, type-safe router:
//!
//! - one direct Ollama `/api/chat` call against the router model with
//!   **the answer's JSON schema passed as `format`** — the sampler
//!   physically cannot emit malformed output (the ollama-rs contract,
//!   here over plain reqwest; no fragile parse-back layers);
//! - the constrained answer parses into a plain Rust enum
//!   ([`TaskKind`]) — unknown classes are a compile-time impossibility;
//! - [`lane_model`] maps the class to the specialized model id for the
//!   local runtime lane. Only local lanes are ever touched: the router
//!   never picks a cloud/remote model, escalation stays explicit.
//!
//! Taxonomy (the module's whole "thinking budget"):
//! - `code_generation` — writing/refactoring code → the 4B coder lane.
//! - `data_extraction` — parse/structure/pull-out-fields → the 8B lane.
//! - `cloud_escalation` — something genuinely beyond local reasoning →
//!   the 8B lane (deep-r1 is busy; the gateway's own tiering decides
//!   more) while an `escalation note` rides the reason string.
//! - `intent_routing` — chat/lore/ops/small talk → no swap (the
//!   runtime's default tier is already correct for it).
//!
//! Failure discipline: the router is an OPTIMIZER, never a dependency.
//! Ollama down, schema parse failure, or a timeout → classify() errs and
//! the caller falls through to the runtime's normal model choice. A
//! degraded router degrades to yesterday's behaviour, never to a
//! refused chat.

use crate::config::BotConfig;
use anyhow::{Context as _, Result};
use serde_json::{Value, json};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Config ([router] in bot.toml)
// ---------------------------------------------------------------------------

/// `[router]` in bot.toml — the local task-enum router. Default off:
/// opt-in ([router] enabled = true) after an operator has eyeballed the
/// lane map; a mistuned router must never be a silent production
/// surprise. Frequencies: the classification call is bounded (10 s
/// timeout in classify, 12 s overall in maybe_lane_model) and failures
/// degrade to no swap.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RouterConfig {
    /// Master switch (default false — see module docs).
    #[serde(default)]
    pub enabled: bool,
    /// The tiny classifier model (fresh local lineup: qwen3:0.6b is the
    /// sweet spot — sub-second, schema-capable).
    #[serde(default = "default_router_model")]
    pub model: String,
    /// Ollama base (the node's local runtime).
    #[serde(default = "default_ollama_url")]
    pub ollama_url: String,
    /// Lane model map. Only local ollama ids are honored (see
    /// is_local_lane); remote ids are refused by lane_model.
    #[serde(default)]
    pub lanes: RouterLanes,
}

/// Task → model map ([router.lanes] in bot.toml).
#[derive(Debug, Clone, serde::Deserialize, Default)]
pub struct RouterLanes {
    #[serde(default = "default_lane_code")]
    pub code_generation: String,
    #[serde(default = "default_lane_data")]
    pub data_extraction: String,
    #[serde(default = "default_lane_cloud")]
    pub cloud_escalation: String,
}

fn default_router_model() -> String {
    "qwen3:0.6b".to_string()
}
fn default_ollama_url() -> String {
    "http://127.0.0.1:11434".to_string()
}
fn default_lane_code() -> String {
    "qwen3:4b".to_string()
}
fn default_lane_data() -> String {
    "qwen3:8b".to_string()
}
fn default_lane_cloud() -> String {
    "qwen3:8b".to_string()
}

impl Default for RouterConfig {
    /// Plain defaults with the router OFF and fresh-local-model lanes
    /// (classification stays inactive until the operator opts in). The
    /// lanes are built from the serde default fns explicitly — a derived
    /// Default on RouterLanes would hand back empty strings, silently
    /// disabling every lane.
    fn default() -> Self {
        Self {
            enabled: false,
            model: default_router_model(),
            ollama_url: default_ollama_url(),
            lanes: RouterLanes {
                code_generation: default_lane_code(),
                data_extraction: default_lane_data(),
                cloud_escalation: default_lane_cloud(),
            },
        }
    }
}

/// Task classes the constrained router can emit — the enum the lex
/// answer maps onto. Serialized form matches the schema's enum strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskKind {
    CodeGeneration,
    DataExtraction,
    CloudEscalation,
    IntentRouting,
}

impl TaskKind {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "code_generation" => Some(Self::CodeGeneration),
            "data_extraction" => Some(Self::DataExtraction),
            "cloud_escalation" => Some(Self::CloudEscalation),
            "intent_routing" => Some(Self::IntentRouting),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CodeGeneration => "code_generation",
            Self::DataExtraction => "data_extraction",
            Self::CloudEscalation => "cloud_escalation",
            Self::IntentRouting => "intent_routing",
        }
    }
}

/// One classified prompt: the class, the constrained router's own
/// one-line reason (useful in traces), and the answer schema version.
#[derive(Debug, Clone)]
pub struct Classification {
    pub kind: TaskKind,
    pub reason: String,
}

/// The answer's grammar, handed to Ollama as `format` — and consumed by
/// serde Deserialize. These two stay one shape on purpose: whatever the
/// sampler can pick is exactly what this struct accepts. (A function,
/// not a const — the json! macro is runtime-evaluated.)
fn router_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "task": {"type": "string", "enum": [
                "code_generation", "data_extraction", "cloud_escalation", "intent_routing"
            ]},
            "reason": {"type": "string"}
        },
        "required": ["task"],
        "additionalProperties": false
    })
}

/// The constrained answer's Rust twin.
#[derive(Debug, serde::Deserialize)]
struct RouterAnswer {
    task: String,
    #[serde(default)]
    reason: String,
}

/// Classify one user prompt (fast path: direct Ollama, schema-frozen).
/// Long prompts are truncated — the router needs the head of the
/// message, not all of it.
pub async fn classify(cfg: &BotConfig, prompt: &str) -> Result<Classification> {
    let router_cfg = &cfg.router;
    let url = format!("{}/api/chat", router_cfg.ollama_url.trim_end_matches('/'));
    let model = router_cfg.model.trim();
    anyhow::ensure!(!model.is_empty(), "router model is empty");

    let head: String = prompt.chars().take(600).collect();
    let body = json!({
        "model": model,
        "stream": false,
        // The fresh tiny lineup is thinking-capable: with think left on,
        // qwen3:0.6b spends the entire num_predict budget inside
        // `thinking` and emits an empty `content` (live 2026-10-19). The
        // classifier's reason is one clause — thinking is pure overhead
        // here. `think: false` returns straight to the constrained
        // content.
        "think": false,
        "format": router_schema(),
        "options": {"num_predict": 96, "temperature": 0.0},
        "messages": [
            {"role": "system", "content": "Classify the user message into exactly one task class. code_generation = writing or changing code or config files; data_extraction = parsing, structuring or pulling facts out of provided data; cloud_escalation = needs far more reasoning than a small local model; everything else = intent_routing."},
            {"role": "user", "content": head}
        ]
    });
    let resp = crate::api::client()
        .post(&url)
        .json(&body)
        .timeout(Duration::from_secs(10))
        .send()
        .await
        .with_context(|| format!("router unreachable at {url}"))?;
    anyhow::ensure!(resp.status().is_success(), "router call failed: {}", resp.status());
    let body: Value = resp.json().await.context("router returned non-JSON")?;
    let answer: RouterAnswer = serde_json::from_value(
        body.get("message").cloned().ok_or_else(|| anyhow::anyhow!("router response has no message"))?,
    )
    .context("router answer did not satisfy its schema")?;
    let kind = TaskKind::from_str(&answer.task)
        .ok_or_else(|| anyhow::anyhow!("router emitted an unknown class {:?}", answer.task))?;
    Ok(Classification {
        kind,
        reason: answer.reason,
    })
}

/// The specialized local model id for a class, or None when the class
/// should keep whatever the runtime picked already (chat/intent → the
/// default tier stays). Cloud/remote lanes are never returned.
pub fn lane_model(kind: TaskKind, cfg: &BotConfig) -> Option<String> {
    let lanes = &cfg.router.lanes;
    let model = match kind {
        TaskKind::CodeGeneration => &lanes.code_generation,
        TaskKind::DataExtraction => &lanes.data_extraction,
        TaskKind::CloudEscalation => &lanes.cloud_escalation,
        // The runtime's default model IS the intent lane — swapping it
        // would just add latency to exactly the messages that want none.
        TaskKind::IntentRouting => return None,
    };
    let model = model.trim();
    (!model.is_empty()).then(|| model.to_string()).filter(|m| is_local_lane(m))
}

/// Only local lanes drive the swap (the safe swap surface): BARE model
/// ids — the runtime's own litellm model_name vocabulary (`qwen3:4b`),
/// learned the hard way live 2026-10-17 (`model: ollama:qwen3:8b` → 400,
/// bare `qwen3:8b` served). Any provider-prefixed id (`openrouter/…`,
/// `gemini/…`, `ollama/…`) is refused outright: a swapped lane must be a
/// model the local runtime already serves under a bare name.
fn is_local_lane(model: &str) -> bool {
    !model.contains('/')
}

/// The chat path's integration point: classify + lane when the router is
/// enabled, else pass through. Errors degrade to None (no swap).
pub async fn maybe_lane_model(cfg: &BotConfig, prompt: &str) -> Option<String> {
    if !cfg.router.enabled {
        return None;
    }
    let classification = tokio::time::timeout(Duration::from_secs(12), classify(cfg, prompt))
        .await
        .ok()?
        .ok()?;
    let plan = lane_model(classification.kind, cfg);
    if let Some(lane) = &plan {
        tracing::debug!(
            lane = %lane,
            class = classification.kind.as_str(),
            reason = %classification.reason,
            "task router classification landed"
        );
    }
    plan
}

// ---------------------------------------------------------------------------
// Tests (taxonomy, lane map, degraded-path discipline)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn router_cfg() -> crate::config::BotConfig {
        let mut cfg = crate::config::BotConfig::from_parts("http://gateway", "http://chain");
        cfg.router = RouterConfig::default();
        cfg.router.enabled = true;
        cfg
    }

    fn cfg_off() -> crate::config::BotConfig {
        let mut cfg = router_cfg();
        cfg.router.enabled = false;
        cfg
    }

    #[test]
    fn taxonomy_round_trips() {
        for kind in [
            TaskKind::CodeGeneration,
            TaskKind::DataExtraction,
            TaskKind::CloudEscalation,
            TaskKind::IntentRouting,
        ] {
            assert_eq!(TaskKind::from_str(kind.as_str()), Some(kind));
        }
        assert_eq!(TaskKind::from_str("something_else"), None);
    }

    #[test]
    fn lanes_only_swap_local_ids() {
        let cfg = router_cfg();
        assert_eq!(
            lane_model(TaskKind::CodeGeneration, &cfg).as_deref(),
            Some("qwen3:4b")
        );
        assert_eq!(
            lane_model(TaskKind::DataExtraction, &cfg).as_deref(),
            Some("qwen3:8b")
        );
        // Intent keeps the runtime default.
        assert_eq!(lane_model(TaskKind::IntentRouting, &cfg), None);
        // A remote-looking lane url is refused even if configured wrong.
        let mut cfg = router_cfg();
        cfg.router.lanes.code_generation = "openrouter/mistral".into();
        assert_eq!(lane_model(TaskKind::CodeGeneration, &cfg), None);
        // Provider-qualified ids never serve as lanes either — the
        // runtime only answers bare model_name ids (live-learned:
        // `model: ollama:qwen3:8b` 400s, bare `qwen3:8b` serves).
        cfg.router.lanes.code_generation = "ollama/deepseek-r1:8b".into();
        assert_eq!(lane_model(TaskKind::CodeGeneration, &cfg), None);
    }

    #[tokio::test]
    async fn disabled_router_never_calls() {
        let cfg = cfg_off();
        assert_eq!(maybe_lane_model(&cfg, "fix this bug").await, None);
    }

    #[tokio::test]
    async fn dead_ollama_degrades_to_no_swap() {
        let mut cfg = router_cfg();
        // A port on this box's LAN that is closed (the runtime on 11434 is
        // real, so probe an unreachable one instead).
        cfg.router.ollama_url = "http://127.0.0.1:9".into();
        assert_eq!(maybe_lane_model(&cfg, "refactor this module").await, None);
    }

    #[test]
    fn schema_froze_the_grammar() {
        // The real call is proven against a live ollama elsewhere; here:
        // the schema, the struct, and the enum must agree shape-wise.
        let schema = router_schema();
        let props = schema["properties"]["task"]["enum"]
            .as_array()
            .expect("enum inside router schema");
        for v in props {
            let s = v.as_str().unwrap();
            assert!(TaskKind::from_str(s).is_some(), "schema enum {s} unparseable");
        }
        assert!(schema["required"].as_array().unwrap().contains(&json!("task")));
        // The struct parses anything the sample-level grammar allows onto
        // "task" as a string — the ENUM parse is where an off-vocabulary
        // answer is caught (the format contract means ollama never sends
        // one, but classify() must still be the hard wall).
        let sneaky: RouterAnswer =
            serde_json::from_value(json!({"task": "mystery"})).expect("string parses");
        assert_eq!(TaskKind::from_str(&sneaky.task), None, "unknown class maps to nothing");
        let good: RouterAnswer = serde_json::from_value(json!({"task": "code_generation"})).unwrap();
        assert_eq!(TaskKind::from_str(&good.task), Some(TaskKind::CodeGeneration));
    }
}
