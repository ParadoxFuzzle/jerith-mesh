//! Per-request model routing above the runtime (LiteLLM/Ollama).
//!
//! The runtime serves three tiers; the caller (bot, peer, CLI) mostly
//! always requests the advertised base model. This router re-classifies
//! each chat request:
//!
//! - **image content** → the vision model (small, cheap, tuned for it);
//! - **complex prompt** — long context, many turns, reasoning keywords,
//!   or an explicit `[deep]` tag → the deep model (e.g. qwen3:30b);
//! - **everything else** → the caller's model, untouched ("basic chats
//!   go to the smaller models").
//!
//! Deliberately *pinned* models (a caller asking for `or-free`, an
//! embedding model, or any specific non-base id) are never overridden —
//! rerouting is reserved for the auto-resolved base model the bot and
//! peers send, so explicit choices always win.

/// Vision model: small, cheap, image-tuned ( interchangeable via config).
pub const DEFAULT_VISION_MODEL: &str = "minicpm-v4.6:1b";
/// Deep model: the reasoning tier for complex prompts. Deliberately the
/// same 8B by default: the 30B was measured to pin ~100% CPU for minutes
/// on the typical operator box (a machine the owner actually uses), so
/// raising the tier is an explicit per-node config choice, not a default.
pub const DEFAULT_DEEP_MODEL: &str = "qwen3:8b";

/// Total text across the conversation beyond which a prompt counts as
/// "complex" (roughly: more than a screenful of live content).
const DEEP_MIN_CHARS: usize = 4000;
/// User turns beyond which the conversation itself is complex context.
const DEEP_MIN_USER_TURNS: usize = 8;

/// Case-insensitive markers of a task the small models routinely fumble.
const COMPLEX_MARKERS: [&str; 14] = [
    "[deep]",
    "step by step",
    "step-by-step",
    "in depth",
    "in-depth",
    "deep dive",
    "explain the",
    "explain how",
    "explain why",
    "architecture",
    "refactor",
    "trade-offs",
    "tradeoffs",
    "comprehensive",
];

/// The routing tiers (`tool`/`vision`/`deep`) — set once from node
/// config; the base tier is whoever the caller asked for. An empty
/// string disables a tier.
#[derive(Debug, Clone)]
pub struct RouterModels {
    pub tool: String,
    pub vision: String,
    pub deep: String,
}

impl Default for RouterModels {
    fn default() -> Self {
        Self {
            // Tool tier starts DISABLED at the router level: operators
            // pin it (setup default / config) once their runtime hosts
            // the pick — an unpinned default would reroute tool requests
            // to a model the runtime may not serve.
            tool: String::new(),
            vision: DEFAULT_VISION_MODEL.to_string(),
            deep: DEFAULT_DEEP_MODEL.to_string(),
        }
    }
}

/// Build the router tiers from a node config (the run path; the gateway
/// must honor what setup/config wrote, not fall back to the defaults).
impl RouterModels {
    pub fn from_config(cfg: &crate::config::NodeConfig) -> Self {
        Self {
            tool: cfg.runtime_tool_model.clone().unwrap_or_default(),
            vision: cfg.runtime_vision_model.clone().unwrap_or_default(),
            deep: cfg.runtime_deep_model.clone().unwrap_or_default(),
        }
    }
}

/// Extract the plain text of `content`, handling both the string form and
/// the multimodal array form (`[{"type": "text", "text": ...}, ...]`).
fn content_text(content: &serde_json::Value) -> String {
    match content {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// True when any message carries image content (multimodal array with an
/// `image_url` part).
pub fn has_image_content(messages: &[serde_json::Value]) -> bool {
    messages.iter().any(|m| {
        m.get("content").is_some_and(|c| match c {
            serde_json::Value::Array(parts) => parts.iter().any(|p| {
                p.get("type").and_then(|t| t.as_str()) == Some("image_url")
                    || p.get("image_url").is_some()
            }),
            _ => false,
        })
    })
}

/// The complexity verdict for a conversation: explicit `[deep]` tag, the
/// marker phrases, sheer size, or many turns.
pub fn is_complex(messages: &[serde_json::Value]) -> bool {
    let mut total_chars = 0usize;
    let mut user_turns = 0usize;
    let mut last_user_text = String::new();
    for m in messages {
        if m.get("role").and_then(|r| r.as_str()) == Some("user") {
            user_turns += 1;
            last_user_text = content_text(m.get("content").unwrap_or(&serde_json::Value::Null));
        }
        total_chars += content_text(m.get("content").unwrap_or(&serde_json::Value::Null))
            .chars()
            .count();
    }
    let lowered = last_user_text.to_lowercase();
    COMPLEX_MARKERS.iter().any(|mk| lowered.contains(mk))
        || total_chars >= DEEP_MIN_CHARS
        || user_turns >= DEEP_MIN_USER_TURNS
}

/// True when a chat request payload carries function definitions (an
/// agent tool loop). Such requests go to the `tool` tier: tool-calling
/// reliability is a parse-format property, not a depth property.
pub fn payload_has_tools(body: &serde_json::Value) -> bool {
    body.get("tools")
        .and_then(|t| t.as_array())
        .is_some_and(|t| !t.is_empty())
}

/// Decide the model for a chat round-trip. `base` is the caller-requested
/// model, `has_tools` whether the payload carries function definitions.
/// Returns the override (`Some(tool|vision|deep)`) or `None` = "keep the
/// caller's model" (the overwhelmingly common basic-chat outcome).
pub fn route(
    messages: &[serde_json::Value],
    base: Option<&str>,
    has_tools: bool,
) -> Option<&'static str> {
    let base = base?;
    if base.is_empty() {
        return None;
    }
    if has_image_content(messages) {
        return Some("vision");
    }
    if has_tools {
        return Some("tool");
    }
    if is_complex(messages) {
        return Some("deep");
    }
    None
}

/// Gateway-level policy: reroute ONLY unpinned requests — the caller sent
/// nothing, the bot's "jerith-mesh" placeholder, or the advertised base
/// model resolved from /status. Every other id is a deliberate choice and
/// passes through untouched. `has_tools` marks agent tool-loop payloads
/// (they want the `tool` tier ahead of the thinking rules).
pub fn reroute_for_gateway(
    requested: Option<&str>,
    base_model: Option<&str>,
    messages: &[serde_json::Value],
    tiers: &RouterModels,
    has_tools: bool,
) -> Option<String> {
    let requested = requested.unwrap_or_default().trim();
    let is_auto = requested.is_empty()
        || requested == "jerith-mesh"
        || base_model.is_some_and(|b| requested == b);
    if !is_auto {
        return None;
    }
    // Eligibility anchor: the requested model when present, else the
    // advertised base — so a caller who omitted `model` entirely still
    // gets tier routing instead of falling to the runtime default-blind.
    let anchor = if requested.is_empty() {
        base_model.unwrap_or_default()
    } else {
        requested
    };
    let next = match route(messages, Some(anchor), has_tools) {
        Some("tool") => Some(tiers.tool.clone()),
        Some("vision") => Some(tiers.vision.clone()),
        Some("deep") => Some(tiers.deep.clone()),
        _ => None,
    };
    // A tier resolving to the model the caller already sent is a no-op —
    // suppress the rewrite (and the misleading reroute journal line).
    next.filter(|n| !n.is_empty() && *n != requested)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tiers() -> RouterModels {
        RouterModels::default()
    }

    #[test]
    fn basic_chat_keeps_callers_model() {
        let msgs = vec![
            json!({"role": "system", "content": "you are jerith"}),
            json!({"role": "user", "content": "what time is mine? /mine"}),
        ];
        assert_eq!(route(&msgs, Some("qwen3:8b"), false), None);
        // Gateway policy: base-model request stays as-is when no tier wants it.
        assert_eq!(
            reroute_for_gateway(Some("qwen3:8b"), Some("qwen3:8b"), &msgs, &tiers(), false),
            None
        );
    }

    #[test]
    fn image_content_routes_to_vision() {
        let msgs = vec![json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "what is in this screenshot?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
            ]
        })];
        assert_eq!(
            reroute_for_gateway(Some("qwen3:8b"), Some("qwen3:8b"), &msgs, &tiers(), false),
            Some("minicpm-v4.6:1b".to_string())
        );
    }

    #[test]
    fn tool_payloads_route_to_the_tool_tier() {
        let msgs = vec![json!({"role": "user", "content": "search memory for the drill notes"})];
        let tiers = RouterModels {
            tool: "nemotron-3-nano:4b".into(),
            ..upgraded_tiers()
        };
        // A tool-loop request reads as "tool": the tool prompt is NOT his
        // reasoning choice — the payload format is. Even a complex-looking
        // prompt keeps tool-calling reliability first.
        let complex = [json!(
            {"role": "user", "content": format!("Refactor the module [deep] {}", "y".repeat(DEEP_MIN_CHARS))}
        )];
        assert_eq!(
            reroute_for_gateway(Some("qwen3:8b"), Some("qwen3:8b"), &complex, &tiers, true),
            Some("nemotron-3-nano:4b".to_string())
        );
        // Vision still wins over tools: an image-bearing tool call goes to
        // the vision model.
        let image_tools = vec![json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "caption this"},
                {"type": "image_url", "image_url": {"url": "https://x/img.png"}}
            ]
        })];
        assert_eq!(
            reroute_for_gateway(
                Some("qwen3:8b"),
                Some("qwen3:8b"),
                &image_tools,
                &tiers,
                true
            ),
            Some("minicpm-v4.6:1b".to_string())
        );
        // Disabled (empty) tool tier: the same request falls through to the
        // deep rules — a suppressed no-op when deep resolves to the base.
        let tiers_no_tool = RouterModels {
            tool: String::new(),
            ..upgraded_tiers()
        };
        assert_eq!(
            reroute_for_gateway(
                Some("qwen3:8b"),
                Some("qwen3:8b"),
                &msgs,
                &tiers_no_tool,
                true
            ),
            None
        );
    }

    #[test]
    fn payload_has_tools_needs_a_nonempty_function_list() {
        assert!(!payload_has_tools(&json!({"a": 1})));
        assert!(!payload_has_tools(&json!({"tools": []})));
        assert!(!payload_has_tools(&json!({"tools": "not-a-list"})));
        assert!(payload_has_tools(&json!({
            "tools": [{"type": "function", "function": {"name": "f"}}]
        })));
    }

    /// Tiers whose deep model differs from the base — the realistic
    /// upgraded-node configuration worth testing reroute behavior for.
    fn upgraded_tiers() -> RouterModels {
        RouterModels {
            deep: "qwen3-large-test".to_string(),
            ..RouterModels::default()
        }
    }

    #[test]
    fn long_conversation_routes_to_deep() {
        let msgs = vec![json!({"role": "user", "content": "x".repeat(DEEP_MIN_CHARS)})];
        assert_eq!(
            reroute_for_gateway(
                Some("qwen3:8b"),
                Some("qwen3:8b"),
                &msgs,
                &upgraded_tiers(),
                false
            ),
            Some("qwen3-large-test".to_string())
        );
    }

    #[test]
    fn deep_tier_equal_to_base_is_a_suppressed_no_op() {
        // Default config: deep == base (the 30B was dropped — it pinned
        // whole CPU cores). Complex prompts then keep the caller's model
        // with NO rewrite (and no misleading journal line).
        let msgs = vec![json!({"role": "user", "content": "x".repeat(DEEP_MIN_CHARS)})];
        assert_eq!(
            reroute_for_gateway(Some("qwen3:8b"), Some("qwen3:8b"), &msgs, &tiers(), false),
            None
        );
    }

    #[test]
    fn marker_phrase_routes_to_deep() {
        let msgs = vec![json!(
            {"role": "user", "content": "Explain the mesh verification pipeline in depth"}
        )];
        assert!(is_complex(&msgs));
        assert_eq!(
            reroute_for_gateway(
                Some("qwen3:8b"),
                Some("qwen3:8b"),
                &msgs,
                &upgraded_tiers(),
                false
            ),
            Some("qwen3-large-test".to_string())
        );
    }

    #[test]
    fn explicit_deep_tag_routes_to_deep_and_is_case_insensitive() {
        let msgs = vec![json!(
            {"role": "user", "content": "compare mentors [DEEP]"}
        )];
        assert_eq!(
            reroute_for_gateway(
                Some("qwen3:8b"),
                Some("qwen3:8b"),
                &msgs,
                &upgraded_tiers(),
                false
            ),
            Some("qwen3-large-test".to_string())
        );
    }

    #[test]
    fn many_user_turns_route_to_deep() {
        let msgs: Vec<serde_json::Value> = (0..DEEP_MIN_USER_TURNS)
            .map(|i| json!({"role": "user", "content": format!("question {i}")}))
            .collect();
        assert_eq!(
            reroute_for_gateway(
                Some("qwen3:8b"),
                Some("qwen3:8b"),
                &msgs,
                &upgraded_tiers(),
                false
            ),
            Some("qwen3-large-test".to_string())
        );
    }

    #[test]
    fn pinned_models_pass_through_untouched() {
        let deepish = vec![json!(
            {"role": "user", "content": format!("explain the architecture {}", "y".repeat(DEEP_MIN_CHARS))}
        )];
        let image = vec![json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "caption this"},
                {"type": "image_url", "image_url": {"url": "https://x/img.png"}}
            ]
        })];
        for msgs in [deepish, image] {
            assert_eq!(
                reroute_for_gateway(Some("or-free"), Some("qwen3:8b"), &msgs, &tiers(), false),
                None,
                "a caller who explicitly pinned a model is never overridden"
            );
        }
    }

    #[test]
    fn placeholder_model_is_eligible_for_rerouting() {
        let complex = vec![json!(
            {"role": "user", "content": "Refactor the whole module [deep]"}
        )];
        assert_eq!(
            reroute_for_gateway(
                Some("jerith-mesh"),
                Some("qwen3:8b"),
                &complex,
                &upgraded_tiers(),
                false
            ),
            Some("qwen3-large-test".to_string())
        );
        assert_eq!(
            reroute_for_gateway(None, Some("qwen3:8b"), &complex, &upgraded_tiers(), false),
            Some("qwen3-large-test".to_string())
        );
    }

    #[test]
    fn empty_base_string_disables_routing() {
        let msgs = vec![json!({"role": "user", "content": "anything"})];
        assert_eq!(route(&msgs, None, false), None);
        assert_eq!(route(&msgs, Some(""), false), None);
    }

    #[test]
    fn no_image_mention_in_plain_string_content() {
        // The word "image_url" inside plain text must NOT look multimodal.
        let msgs = vec![json!(
            {"role": "user", "content": "how do I embed an image_url in markdown?"}
        )];
        assert!(!has_image_content(&msgs));
        assert_eq!(
            reroute_for_gateway(Some("qwen3:8b"), Some("qwen3:8b"), &msgs, &tiers(), false),
            None
        );
    }
}
