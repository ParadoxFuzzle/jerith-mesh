//! Plugin / skill SDK (OpenClaw-style modular plugin architecture port).
//!
//! The bot's model-facing capabilities are organized as **plugins** owning
//! dot-named **skills** (`web_search`, `memory.embed`, …). A skill is an
//! OpenAI tool declaration (schema + executor) the chat tool loop can
//! invoke mid-answer; a plugin is the unit owners enable/disable in
//! bot.toml. Everything an author needs:
//!
//! ```toml
//! # ~/.jerith-mesh/bot.toml — omit the section for the default set
//! [plugins]
//! enabled = ["websearch", "memory", "lore", "squeeze"]
//! ```
//!
//! A plugin author adds one `SkillSpec` entry with a runner closing over
//! the gateway config — the tool loop's `tools` array, the whitelist
//! behind the content-intent slop path, and the `[plugins]` gating all
//! derive from that single table. Bot-side plugins (like `squeeze`)
//! register **zero** skills: they are lifecycle stages, not model tools,
//! but stay in the config surface so `enabled = [...]` describes the whole
//! bot in one place.

use anyhow::Context as _;
use serde_json::{Value, json};
use std::future::Future;
use std::pin::Pin;

/// Boxed executor future for one model-requested skill call.
type BoxedSkillFut<'a> = Pin<Box<dyn Future<Output = anyhow::Result<Value>> + Send + 'a>>;

/// A skill executor: gateway config + parsed tool arguments in, a JSON
/// value out (fed back into the tool loop as the tool round content).
type SkillRunner =
    for<'args> fn(&'args crate::config::BotConfig, &'args Value) -> BoxedSkillFut<'args>;

/// One callable skill inside a plugin.
#[derive(Debug, Clone)]
pub struct SkillSpec {
    /// OpenClaw-style dot-or-bare name the model sees (`web_search`,
    /// `memory.embed`, `lore.search`).
    pub name: &'static str,
    /// Owning plugin id (`websearch`, `memory`, `lore`).
    pub plugin: &'static str,
    /// One-line description shown in the tool declaration.
    pub description: &'static str,
    /// JSON-schema `parameters` object for the OpenAI tools array.
    pub parameters: Value,
    /// Executor against the gateway.
    pub run: SkillRunner,
}

impl SkillSpec {
    /// The OpenAI tool declaration this skill turns into.
    fn tool_def(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        })
    }
}

/// Pre-packaged plugin ids (`[plugins] enabled = [...]` in bot.toml).
pub const PREPACKAGED_PLUGINS: [&str; 5] = ["websearch", "memory", "lore", "squeeze", "vision"];

/// All pre-packaged model-facing skills. The bot-side `squeeze` plugin
/// intentionally has no entries (it registers no skills).
fn standard_skills() -> Vec<SkillSpec> {
    vec![
        SkillSpec {
            name: "web_search",
            plugin: "websearch",
            description: "Search the public web (Brave Search API) and return ranked results with title, url, description and page age. Use when the answer depends on external or recent information.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "count": {"type": "integer", "minimum": 1, "maximum": 20, "description": "How many results (default 5)"},
                    "freshness": {"type": "string", "enum": ["day", "week", "month", "year"], "description": "Time filter"}
                },
                "required": ["query"]
            }),
            run: web_search_runner,
        },
        SkillSpec {
            name: "web_fetch",
            plugin: "websearch",
            description: "Download an http(s):// URL and return the page's readable text (HTML stripped, length-capped, private targets refused). Use after web_search to read a specific page.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "http(s):// URL to read"},
                    "max_chars": {"type": "integer", "minimum": 200, "maximum": 16000, "description": "Text cap in characters (default 8000)"}
                },
                "required": ["url"]
            }),
            run: web_fetch_runner,
        },
        SkillSpec {
            name: "memory.embed",
            plugin: "memory",
            description: "Store a short note in the node's private memory (SQLite + nomic embeddings). Never leaves the node. Use when the user asks to remember something.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The note to store"}
                },
                "required": ["text"]
            }),
            run: memory_embed_runner,
        },
        SkillSpec {
            name: "memory.search",
            plugin: "memory",
            description: "Vector-search the node's private memory for notes relevant to a query.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Semantic search query"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "description": "How many hits (default 5)"}
                },
                "required": ["query"]
            }),
            run: memory_search_runner,
        },
        SkillSpec {
            name: "lore.search",
            plugin: "lore",
            description: "Search the node's lore index (community canon / long-term knowledge).",
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Lore query"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "description": "How many hits (default 5)"}
                },
                "required": ["query"]
            }),
            run: lore_search_runner,
        },
        SkillSpec {
            name: "vision.describe",
            plugin: "vision",
            description: "Describe or analyze an image using the local vision model. Returns a text description of the image.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": "Base64-encoded image data or data URL (data:image/...)"},
                    "prompt": {"type": "string", "description": "Optional prompt to guide the description (default: 'Describe this image')"}
                },
                "required": ["image"]
            }),
            run: vision_describe_runner,
        },
        SkillSpec {
            name: "vision.analyze",
            plugin: "vision",
            description: "Analyze an image for objects, text, or specific details using the local vision model.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": "Base64-encoded image data or data URL (data:image/...)"},
                    "prompt": {"type": "string", "description": "Analysis prompt (e.g., 'What text is in this image?')"}
                },
                "required": ["image", "prompt"]
            }),
            run: vision_analyze_runner,
        },
    ]
}

// --- Runners: thin argument adapters over the gateway skill calls -------

fn web_search_runner<'args>(
    cfg: &'args crate::config::BotConfig,
    args: &'args Value,
) -> BoxedSkillFut<'args> {
    Box::pin(async move {
        crate::api::web_search_skill(
            cfg,
            args.get("query")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            args.get("count").and_then(Value::as_u64),
            args.get("freshness").and_then(Value::as_str),
        )
        .await
    })
}

fn web_fetch_runner<'args>(
    cfg: &'args crate::config::BotConfig,
    args: &'args Value,
) -> BoxedSkillFut<'args> {
    Box::pin(async move {
        let cap = args
            .get("max_chars")
            .and_then(Value::as_u64)
            .map(|c| c as usize)
            .unwrap_or(8000)
            .clamp(200, 16000);
        crate::api::web_fetch_skill(
            cfg,
            args.get("url").and_then(Value::as_str).unwrap_or_default(),
            Some(cap),
        )
        .await
    })
}

fn memory_embed_runner<'args>(
    cfg: &'args crate::config::BotConfig,
    args: &'args Value,
) -> BoxedSkillFut<'args> {
    Box::pin(async move {
        let text = args.get("text").and_then(Value::as_str).unwrap_or_default();
        anyhow::ensure!(!text.trim().is_empty(), "memory.embed: empty text");
        let (id, provider) = crate::api::memory_embed(cfg, text).await?;
        Ok(json!({ "id": id, "provider": provider, "ok": true }))
    })
}

fn memory_search_runner<'args>(
    cfg: &'args crate::config::BotConfig,
    args: &'args Value,
) -> BoxedSkillFut<'args> {
    Box::pin(async move {
        let query = args
            .get("query")
            .and_then(Value::as_str)
            .unwrap_or_default();
        anyhow::ensure!(!query.trim().is_empty(), "memory.search: empty query");
        let top_k = args
            .get("top_k")
            .and_then(Value::as_u64)
            .unwrap_or(5)
            .clamp(1, 20);
        let results = crate::api::memory_search(cfg, query, top_k).await?;
        Ok(json!({ "results": results, "count": results.len() }))
    })
}

fn lore_search_runner<'args>(
    cfg: &'args crate::config::BotConfig,
    args: &'args Value,
) -> BoxedSkillFut<'args> {
    Box::pin(async move {
        let query = args
            .get("query")
            .and_then(Value::as_str)
            .unwrap_or_default();
        anyhow::ensure!(!query.trim().is_empty(), "lore.search: empty query");
        let top_k = args
            .get("top_k")
            .and_then(Value::as_u64)
            .unwrap_or(5)
            .clamp(1, 20);
        let url = format!("{}/v1/skills/lore/search", cfg.gateway_url);
        let resp = crate::api::client()
            .post(&url)
            .json(&json!({ "query": query, "top_k": top_k }))
            .send()
            .await
            .with_context(|| format!("gateway unreachable at {url}"))?;
        let status = resp.status();
        let body: Value = resp.json().await.context("gateway returned non-JSON")?;
        if !status.is_success() {
            anyhow::bail!("lore.search failed: {}", status);
        }
        Ok(body)
    })
}

fn vision_describe_runner<'args>(
    cfg: &'args crate::config::BotConfig,
    args: &'args Value,
) -> BoxedSkillFut<'args> {
    Box::pin(async move {
        let image = args.get("image").and_then(Value::as_str).unwrap_or_default();
        anyhow::ensure!(!image.trim().is_empty(), "vision.describe: empty image");
        let prompt = args.get("prompt").and_then(Value::as_str);
        vision_call(cfg, image, prompt).await
    })
}

fn vision_analyze_runner<'args>(
    cfg: &'args crate::config::BotConfig,
    args: &'args Value,
) -> BoxedSkillFut<'args> {
    Box::pin(async move {
        let image = args.get("image").and_then(Value::as_str).unwrap_or_default();
        anyhow::ensure!(!image.trim().is_empty(), "vision.analyze: empty image");
        let prompt = args.get("prompt").and_then(Value::as_str);
        let prompt = prompt.ok_or_else(|| anyhow::anyhow!("vision.analyze: prompt required"))?;
        vision_call(cfg, image, Some(prompt)).await
    })
}

async fn vision_call(
    cfg: &crate::config::BotConfig,
    image: &str,
    prompt: Option<&str>,
) -> anyhow::Result<Value> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()?;
    let url = format!("{}/v1/skills/vision/describe", cfg.gateway_url);
    let body = json!({
        "image": image,
        "prompt": prompt.unwrap_or("Describe this image"),
    });
    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .context("calling gateway /v1/skills/vision/describe")?;
    let status = resp.status();
    let body: Value = resp.json().await.context("reading gateway vision response")?;
    if !status.is_success() {
        anyhow::bail!("vision failed: {}", status);
    }
    Ok(body)
}

/// The registry: every skill the bot could have, plus the enabled-plugin
/// gate from `[plugins]` in bot.toml. All model-facing callables flow
/// through here — tool declarations, the tool whitelist, and dispatch.
#[derive(Debug, Clone)]
pub struct SkillRegistry {
    skills: Vec<SkillSpec>,
    /// Enabled plugin ids; also the dispatch whitelist.
    enabled_plugins: std::collections::BTreeSet<String>,
}

impl SkillRegistry {
    /// The standard set with only `enabled_plugins` switched on.
    pub fn standard(enabled_plugins: impl IntoIterator<Item = impl AsRef<str>>) -> Self {
        Self::from_skills(standard_skills(), enabled_plugins)
    }

    /// Registry from explicit skills — the extension point plugin authors
    /// (and tests) use instead of touching `standard_skills()`.
    pub fn from_skills(
        skills: Vec<SkillSpec>,
        enabled_plugins: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> Self {
        Self {
            skills,
            enabled_plugins: enabled_plugins
                .into_iter()
                .map(|p| p.as_ref().to_string())
                .collect(),
        }
    }

    /// Validate a `[plugins] enabled` list against the pre-packaged ids.
    /// Unknown names return an error naming the culprit — mistyped feature
    /// flags should fail loud, not sit silently off.
    pub fn check_config_enabled(entries: &[String]) -> anyhow::Result<()> {
        for e in entries {
            if !PREPACKAGED_PLUGINS.contains(&e.as_str()) {
                anyhow::bail!(
                    "unknown plugin `{e}` in [plugins] enabled — known: {}",
                    PREPACKAGED_PLUGINS.join(", ")
                );
            }
        }
        Ok(())
    }

    fn enabled(&self, skill: &SkillSpec) -> bool {
        self.enabled_plugins.contains(skill.plugin)
    }

    /// OpenAI tool declarations for every enabled skill — what the chat
    /// payload's `tools` array gets.
    pub fn tool_defs(&self) -> Vec<Value> {
        self.skills
            .iter()
            .filter(|s| self.enabled(s))
            .map(SkillSpec::tool_def)
            .collect()
    }

    /// Names of enabled skills — the dispatch whitelist (the model can
    /// never summon a disabled or unknown tool).
    pub fn enabled_skill_names(&self) -> Vec<&'static str> {
        self.skills
            .iter()
            .filter(|s| self.enabled(s))
            .map(|s| s.name)
            .collect()
    }

    /// Execute a model-requested skill by name. Unknown or disabled
    /// skills are refused (the tool loop surfaces the error to the model,
    /// which can recover; the chat itself never fails on it).
    pub async fn dispatch(
        &self,
        cfg: &crate::config::BotConfig,
        name: &str,
        args: &Value,
    ) -> anyhow::Result<Value> {
        let spec = self
            .skills
            .iter()
            .find(|s| s.name == name && self.enabled(s))
            .ok_or_else(|| anyhow::anyhow!("skill {name:?} unknown or disabled"))?;
        (spec.run)(cfg, args).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A test skill that echoes its arguments — proves the registry routes
    /// through the runner without any network.
    fn echo_skill(name: &'static str, plugin: &'static str) -> SkillSpec {
        SkillSpec {
            name,
            plugin,
            description: "echo args",
            parameters: json!({"type": "object"}),
            run: |_cfg: &crate::config::BotConfig, args: &Value| {
                Box::pin(async move { Ok(args.clone()) })
            },
        }
    }

    fn test_cfg() -> crate::config::BotConfig {
        crate::config::BotConfig {
            discord_token: "test".into(),
            gateway_url: "http://127.0.0.1:1".into(),
            chain_url: "http://127.0.0.1:1".into(),
            announce_channel_id: None,
            guard: Default::default(),
            system_prompt: None,
            postgres: None,
            web: Default::default(),
            plugins: Default::default(),
            owner_discord_id: None,
            archiver: Default::default(),
            squeeze: Default::default(),
            monitor: Default::default(),
            chat_timeout_secs: 240,
            chat_stream: false,
        }
    }

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    #[test]
    fn standard_registry_lists_all_prepackaged_skills() {
        let r = SkillRegistry::standard(PREPACKAGED_PLUGINS);
        let names = r.enabled_skill_names();
        for expect in [
            "web_search",
            "web_fetch",
            "memory.embed",
            "memory.search",
            "lore.search",
            "vision.describe",
            "vision.analyze",
        ] {
            assert!(names.contains(&expect), "missing {expect}");
        }
        // 7 model-facing skills; squeeze is bot-side (no tool def anywhere).
        assert_eq!(names.len(), 7);
        assert!(!names.iter().any(|n| n.contains("squeeze")));
    }

    #[test]
    fn gating_removes_disabled_plugin_skills() {
        let r = SkillRegistry::standard(["websearch"]);
        assert_eq!(r.enabled_skill_names(), vec!["web_search", "web_fetch"]);
        let defs = r.tool_defs();
        assert!(defs.iter().all(|d| !d.to_string().contains("memory.")));
    }

    #[test]
    fn empty_gate_is_all_off() {
        let r = SkillRegistry::standard(Vec::<String>::new());
        assert!(r.tool_defs().is_empty());
        assert!(r.enabled_skill_names().is_empty());
    }

    #[test]
    fn dispatch_routes_by_name_and_refuses_unknown_or_disabled() {
        let enabled = SkillRegistry::from_skills(
            vec![
                echo_skill("debug.echo_a", "a"),
                echo_skill("debug.echo_b", "b"),
            ],
            ["a"],
        );
        let args = json!({"x": 1});

        let ok = rt().block_on(enabled.dispatch(&test_cfg(), "debug.echo_a", &args));
        assert_eq!(ok.unwrap(), args);

        // Right shape, plugin disabled → refused.
        let err = rt().block_on(enabled.dispatch(&test_cfg(), "debug.echo_b", &args));
        assert_err_mentions(&err, "unknown or disabled");

        // Unknown name → refused.
        let err = rt().block_on(enabled.dispatch(&test_cfg(), "debug.nope", &args));
        assert_err_mentions(&err, "unknown or disabled");
    }

    fn assert_err_mentions(err: &anyhow::Result<Value>, needle: &str) {
        let msg = err.as_ref().unwrap_err().to_string();
        assert!(msg.contains(needle), "error `{msg}` lacks `{needle}`");
    }

    #[test]
    fn unknown_plugin_id_in_config_fails_loud() {
        assert!(
            SkillRegistry::check_config_enabled(&["websearch".into(), "squeeze".into()]).is_ok()
        );
        let err = SkillRegistry::check_config_enabled(&["websearch_typo".into()])
            .expect_err("unknown plugin must fail loud");
        let msg = err.to_string();
        assert!(msg.contains("websearch_typo"));
        assert!(msg.contains("known:"));
    }

    #[test]
    fn tool_defs_are_openai_shaped() {
        let r = SkillRegistry::standard(["memory"]);
        let defs = r.tool_defs();
        assert_eq!(defs.len(), 2);
        for d in defs {
            assert_eq!(d["type"], "function");
            let f = &d["function"];
            assert!(f["name"].as_str().unwrap().starts_with("memory."));
            assert!(f["parameters"]["properties"].is_object());
        }
    }
}
