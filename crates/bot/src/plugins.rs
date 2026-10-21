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

/// Pre-packaged plugin ids (`[plugins] enabled = [...]` in bot.toml).
/// Bare ids gate in-tree plugins; external drops gate as `external:<id>`.
pub const PREPACKAGED_PLUGINS: [&str; 7] = [
    "websearch",
    "memory",
    "lore",
    "squeeze",
    "vision",
    "approvals",
    "pipeline",
];

/// Plugins whose skills may only join a session's tool registry when the
/// session is an owner DM (owner-DM-only by design; also enforced twice —
/// at config load and at dispatch):
/// - `gcp` — external drop; gcloud exec under bot-held credentials.
/// - `coordinator` — multi-bot workers: spawning named agents + the
///   hand-off channel + the shared pool (host compute + mesh identities).
/// - `workflows` — browser record/replay (records the OWNER's browsing
///   sessions and drives the bot's own Chrome).
///
///   Chat is reachable by every Discord user, so a plain `[plugins]
///   enabled` listing of ANY of these ids — bare or `external:<id>` — is
///   refused at load. The registry refuses dispatch of their skills from a
///   registry not constructed in owner-privileged mode (the refusal
///   contract).
pub const OWNER_ONLY_PLUGINS: [&str; 3] = ["gcp", "coordinator", "workflows"];

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

/// The WIRE name a tool declaration carries: OpenAI-compatible providers
/// validate tool names against `[a-zA-Z0-9_-]+` and REJECT anything else
/// (live 2026-10-20: litellm's OpenRouter leg 400'd every tool-attached
/// chat with `Function at index 2 has an invalid name: "memory.embed"`,
/// hammering the router's fallbacks for seconds before the chat reached
/// any working leg — tools-attached answers sat at the placeholder until
/// a late single-shot leg delivered, the "still not streaming" report).
/// Dots and colons fold to underscores. The mapping is a registry
/// invariant: `dispatch` resolves wire names back to the real skill
/// (see [`SkillRegistry::dispatch`]), and [`Self::tool_defs`] emits ONLY
/// wire-safe names — pin by test.
pub fn wire_tool_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

impl SkillSpec {
    /// The OpenAI tool declaration this skill turns into — with the
    /// WIRE name (provider-validation-safe), never the dot-notation
    /// internal name.
    fn tool_def(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": wire_tool_name(self.name),
                "description": self.description,
                "parameters": self.parameters,
            }
        })
    }
}

/// All pre-packaged model-facing skills. The bot-side `squeeze` plugin
/// intentionally has no entries (it registers no skills).
fn standard_skills() -> Vec<SkillSpec> {
    let mut v = vec![
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
    ];
    // Multi-bot coordination (named workers + hand-off + shared pool).
    // Owner-only: joined at owner-DM sessions via the registry's
    // owner_mode gate — the plugin id never rides [plugins] enabled.
    v.extend(crate::coordinator::skills());
    // Workflow recording (owner-driven browse-and-record playbook).
    // Owner-only: never rides [plugins] enabled.
    v.extend(crate::workflows::skills());
    // Human-in-the-loop approval bridge (FEATURES §15): the model pauses
    // before critical side effects while the owner keeps the trigger.
    v.extend(crate::approval::skills());
    // Rust-native macro-tools (FEATURES §17.2): one model verb wrapping a
    // compiled sub-pipeline (scrape → strip → chunk → embed → summarize).
    v.extend(crate::pipeline::skills());
    v
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
        // Mongo recall mirror (mongo.rs, FEATURES §19): the private note
        // also lands in the cluster deduped by id — fire-and-forget, a
        // mirror failure NEVER blocks the tool loop (typed journal only).
        if crate::mongo::handle().is_some() {
            let id2 = id.clone();
            let text = text.to_string();
            tokio::spawn(async move {
                if let Err(e) = crate::mongo::mirror_note(&id2, &text).await {
                    tracing::warn!(error = %e, "memory mirror write skipped");
                }
            });
        }
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
        // Local first: /recall is the node's own embeddings. Empty local
        // hits fall through to the mongo mirror (provenance-labeled) so
        // a rebuilt/gapped node still answers from what it once knew.
        if results.is_empty() {
            match crate::mongo::search_notes(query, top_k).await {
                Ok(mirrored) if !mirrored.is_empty() => {
                    return Ok(json!({
                        "results": mirrored,
                        "count": mirrored.len(),
                        "fallback": "mongo-mirror",
                    }));
                }
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!(error = %e, "mongo mirror fallback failed — plain empty result stands");
                }
            }
        }
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
        let image = args
            .get("image")
            .and_then(Value::as_str)
            .unwrap_or_default();
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
        let image = args
            .get("image")
            .and_then(Value::as_str)
            .unwrap_or_default();
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
        .use_rustls_tls()
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
    let body: Value = resp
        .json()
        .await
        .context("reading gateway vision response")?;
    if !status.is_success() {
        anyhow::bail!("vision failed: {}", status);
    }
    Ok(body)
}

/// One manifest-declared skill joined into the registry (Phase 3a,
/// `docs/PLUGIN-SDK.md`): same public shape as an in-tree `SkillSpec`
/// (tool def, whitelist, refusal contract) but its executor is the
/// out-of-process HTTP dispatcher in `external.rs`, keyed by the drop
/// itself.
#[derive(Debug, Clone)]
pub struct ExternalSkillEntry {
    /// Qualified model-visible name (`weather.now`).
    pub name: String,
    /// Gating id in `[plugins] enabled` (`external:weather`).
    pub gate: String,
    pub description: String,
    pub parameters: Value,
    /// Endpoint + secret + timeout for the dispatcher.
    pub plugin: crate::external::ExternalPlugin,
}

/// The registry: every skill the bot could have, plus the enabled-plugin
/// gate from `[plugins]` in bot.toml. All model-facing callables flow
/// through here — tool declarations, the tool whitelist, and dispatch.
/// In-tree skills live in `skills`; manifest-declared external plugins
/// join into `external` (gated by `external:<id>` ids) without touching
/// the in-tree table.
#[derive(Debug, Clone)]
pub struct SkillRegistry {
    skills: Vec<SkillSpec>,
    external: Vec<ExternalSkillEntry>,
    /// Enabled plugin ids; also the dispatch whitelist.
    enabled_plugins: std::collections::BTreeSet<String>,
    /// Owner-privileged construction (an owner-DM chat session): the only
    /// mode in which owner-only external skills (`gcp.*`) are advertised
    /// and dispatchable. Every other surface — guild chats, guest DMs,
    /// the tool loop of another user — builds the registry WITHOUT this
    /// flag, so an owner-only skill is refused exactly like a disabled
    /// one by [`Self::dispatch`] and never reaches a tool declaration.
    owner_mode: bool,
}

/// The owner-only plugin id behind a gate id or plug-in name, if any —
/// both forms (`external:gcp` for drops, bare `coordinator` for in-tree
/// plugins) resolve through the same [`OWNER_ONLY_PLUGINS`] set.
fn owner_only_id_of(gate_or_plugin: &str) -> Option<&str> {
    let id = gate_or_plugin
        .strip_prefix("external:")
        .unwrap_or(gate_or_plugin);
    OWNER_ONLY_PLUGINS.contains(&id).then_some(id)
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
            external: Vec::new(),
            enabled_plugins: enabled_plugins
                .into_iter()
                .map(|p| p.as_ref().to_string())
                .collect(),
            owner_mode: false,
        }
    }

    /// Mark this registry as an owner-DM session (the privileged
    /// construction): owner-only external gates joined via [`Self::join`]
    /// become live. Non-owner surfaces never call this.
    pub fn owner_privileged(mut self) -> Self {
        self.owner_mode = true;
        self
    }

    /// Join extra external gating ids into the enabled set (the
    /// owner-only plugins list, joined ONLY by the privileged path above
    /// — config validation refuses to ship them in the plain `enabled`).
    pub fn join(mut self, ids: &[String]) -> Self {
        self.enabled_plugins.extend(ids.iter().cloned());
        self
    }

    /// Join parsed external-plugin drops (Phase 3a): each manifest skill
    /// becomes a registry entry gated by its `external:<id>` id. Name
    /// collisions with in-tree skills fail loud here too — a drop that
    /// shadows a built-in must be fixed, not silently ranked.
    pub fn with_external(
        mut self,
        plugins: Vec<crate::external::ExternalPlugin>,
    ) -> anyhow::Result<Self> {
        for plugin in plugins {
            let gate = plugin.gate_id();
            for skill in &plugin.skills {
                anyhow::ensure!(
                    !skill.name.is_empty(),
                    "external:{}: empty skill name",
                    plugin.id
                );
                anyhow::ensure!(
                    !self.skills.iter().any(|s| s.name == skill.name),
                    "external skill {name:?} collides with a built-in skill",
                    name = skill.name
                );
                anyhow::ensure!(
                    !gate.is_empty(),
                    "external:{}: missing plugin id",
                    plugin.id
                );
                self.external.push(ExternalSkillEntry {
                    name: skill.name.clone(),
                    gate: gate.clone(),
                    description: skill.description.clone(),
                    parameters: skill.parameters.clone(),
                    plugin: plugin.clone(),
                });
            }
        }
        Ok(self)
    }

    fn external_enabled(&self, e: &ExternalSkillEntry) -> bool {
        // Owner-only gates are dead unless this registry was built in
        // owner-privileged mode — even a bug that leaked the owner-only
        // id into a non-privileged enabled set cannot invoke them.
        if owner_only_id_of(&e.gate).is_some() && !self.owner_mode {
            return false;
        }
        self.enabled_plugins.contains(&e.gate)
    }

    /// Validate a `[plugins] enabled` list against the pre-packaged ids
    /// and the parsed external drops (`external:<id>` entries must have a
    /// matching drop). Unknown names return an error naming the culprit —
    /// mistyped feature flags (either kind) should fail loud, not sit
    /// silently off. Drops scanned from a missing `plugins.d` make any
    /// `external:*` entry unknown → fail loud.
    pub fn check_config_enabled(
        entries: &[String],
        external: &[crate::external::ExternalPlugin],
    ) -> anyhow::Result<()> {
        for e in entries {
            // Owner-only plugins may never ride the shared gate: chat is
            // reachable by every Discord user, and these skills execute
            // host-level commands under the bot's own credentials.
            if e.strip_prefix("external:")
                .is_some_and(|id| OWNER_ONLY_PLUGINS.contains(&id))
            {
                anyhow::bail!(
                    "plugin `{e}` is owner-only — it may never be listed in [plugins] enabled; \
                     use [plugins] owner_only = [\"{e}\"] (its skills join the registry only \
                     for owner-DM sessions)"
                );
            }
            if let Some(drop_id) = e.strip_prefix("external:") {
                let known = if external.is_empty() {
                    "none".to_string()
                } else {
                    external
                        .iter()
                        .map(|p| p.id.clone())
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                anyhow::ensure!(
                    external.iter().any(|p| p.id == drop_id),
                    "unknown plugin `{e}` in [plugins] enabled — no drop with id \
                     {drop_id:?} in plugins.d (known external: {known})",
                );
            } else if !PREPACKAGED_PLUGINS.contains(&e.as_str()) {
                // In-tree owner-only plugins (coordinator, workflows) are
                // joined at owner-DM sessions automatically — they never
                // ride the shared enabled list, by config or by typo.
                if owner_only_id_of(e).is_some() {
                    anyhow::bail!(
                        "plugin `{e}` is owner-only — it may never be listed in [plugins] \
                         enabled; its skills join the registry only for owner-DM sessions"
                    );
                }
                anyhow::bail!(
                    "unknown plugin `{e}` in [plugins] enabled — known: {} (external drops \
                     must be gated as `external:<id>`)",
                    PREPACKAGED_PLUGINS.join(", ")
                );
            }
        }
        Ok(())
    }

    /// Validate the `[plugins] owner_only` list: every entry must be an
    /// `external:<id>` form naming an existing drop, and (checked by
    /// [`Self::check_config_enabled`] for the `enabled` list) an
    /// owner-only id may not appear in both lists.
    pub fn check_owner_only(
        entries: &[String],
        external: &[crate::external::ExternalPlugin],
    ) -> anyhow::Result<()> {
        for e in entries {
            let Some(drop_id) = e.strip_prefix("external:") else {
                anyhow::bail!(
                    "invalid owner-only plugin `{e}` — owner-only entries must be external \
                     drops gated as `external:<id>`"
                );
            };
            anyhow::ensure!(
                external.iter().any(|p| p.id == drop_id),
                "unknown owner-only plugin `{e}` — no drop with id {drop_id:?} in plugins.d"
            );
        }
        Ok(())
    }

    /// In-tree name-collision gate for external drops (checked at config
    /// load, before any registry is built): a manifest skill must not
    /// shadow a built-in — the drop author fixes their name, the bot
    /// doesn't play ranking games with its own skills.
    pub fn check_external_collisions(
        external: &[crate::external::ExternalPlugin],
    ) -> anyhow::Result<()> {
        for plugin in external {
            for skill in &plugin.skills {
                anyhow::ensure!(
                    !standard_skills().iter().any(|s| s.name == skill.name),
                    "external skill {name:?} collides with a built-in skill — \
                     rename it in the manifest (names are plugin-qualified unless \
                     they shadow a built-in)",
                    name = skill.name
                );
            }
        }
        Ok(())
    }

    fn enabled(&self, skill: &SkillSpec) -> bool {
        // Owner-only in-tree plugins (coordinator, workflows) are dead
        // unless this registry was built in owner-privileged mode — the
        // same boundary external drops go through.
        if owner_only_id_of(skill.plugin).is_some() && !self.owner_mode {
            return false;
        }
        self.enabled_plugins.contains(skill.plugin)
    }

    /// OpenAI tool declarations for every enabled skill — what the chat
    /// payload's `tools` array gets. Joined-in external skills shape
    /// identically (same function-declaration JSON) so the model can't
    /// tell the difference.
    pub fn tool_defs(&self) -> Vec<Value> {
        self.skills
            .iter()
            .filter(|s| self.enabled(s))
            .map(SkillSpec::tool_def)
            .chain(
                self.external
                    .iter()
                    .filter(|e| self.external_enabled(e))
                    .map(|e| {
                        json!({
                            "type": "function",
                            "function": {
                                "name": wire_tool_name(&e.name),
                                "description": e.description,
                                "parameters": e.parameters,
                            }
                        })
                    }),
            )
            .collect()
    }

    /// Names of enabled skills — the dispatch whitelist (the model can
    /// never summon a disabled or unknown tool). In-tree names come back
    /// first, then the enabled external ones.
    pub fn enabled_skill_names(&self) -> Vec<String> {
        self.skills
            .iter()
            .filter(|s| self.enabled(s))
            .map(|s| s.name.to_string())
            .chain(
                self.external
                    .iter()
                    .filter(|e| self.external_enabled(e))
                    .map(|e| e.name.clone()),
            )
            .collect()
    }

    /// Execute a model-requested skill by name. Unknown or disabled
    /// skills are refused (the tool loop surfaces the error to the model,
    /// which can recover; the chat itself never fails on it). External
    /// entries dispatch over HTTP with their own gating, refusals and
    /// typed errors identical to the in-tree contract.
    pub async fn dispatch(
        &self,
        cfg: &crate::config::BotConfig,
        name: &str,
        args: &Value,
    ) -> anyhow::Result<Value> {
        // The model can only ever name the WIRE form (tool declarations
        // carry nothing else) — resolve dots/colons back to the real
        // skill. Exact-name match first keeps bare names one lookup away.
        let enabled = self.enabled_skill_names();
        let resolved = enabled
            .iter()
            .find(|n| n.as_str() == name)
            .cloned()
            .or_else(|| enabled.iter().find(|n| wire_tool_name(n) == name).cloned())
            .unwrap_or_else(|| name.to_string());
        if let Some(spec) = self
            .skills
            .iter()
            .find(|s| s.name == resolved && self.enabled(s))
        {
            return (spec.run)(cfg, args).await;
        }
        if let Some(entry) = self
            .external
            .iter()
            .find(|e| e.name == resolved && self.external_enabled(e))
        {
            return crate::external::invoke(&entry.plugin, &entry.name, args).await;
        }
        anyhow::bail!("skill {name:?} unknown or disabled")
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
            hooks: Default::default(),
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
            external_plugins: Vec::new(),
            chat_timeout_secs: 240,
            chat_stream: false,
            chat_max_tool_rounds: 2,
            sandbox: crate::config::SandboxConfig::default(),
            video: Default::default(),
            audio: Default::default(),
            memory: Default::default(),
            coordinator: crate::config::CoordinatorBotConfig::default(),
            workflows: crate::config::WorkflowsConfig::default(),
            approvals: crate::approval::ApprovalConfig::default(),
            router: crate::router::RouterConfig::default(),
            transports: crate::transports::TransportsConfig::default(),
            mongo: crate::mongo::MongoConfig::default(),
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
            "approval.request",
            "approval.status",
            "approval.cancel",
            "pipeline.scrape_summarize",
            "pipeline.refactor_module",
            "pipeline.cloud_escalate",
        ] {
            assert!(names.iter().any(|n| n == expect), "missing {expect}");
        }
        // 13 model-facing skills; squeeze is bot-side (no tool def
        // anywhere); coordinator/workflows never appear (owner-only gate).
        assert_eq!(names.len(), 13);
        assert!(!names.iter().any(|n| n.contains("squeeze")));
        assert!(!names.iter().any(|n| n.starts_with("workflow.")));
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

    /// The docs/PLUGINS.md worked example, registry half: the same spec
    /// shape (stubbed upstream — no network in tests), wired through
    /// `from_skills`, proves the author path end-to-end at the registry
    /// layer. The transport half is proven bot-wide by the mock-gateway
    /// tool-loop test in api.rs.
    fn weather_example_spec() -> SkillSpec {
        SkillSpec {
            name: "weather.now",
            plugin: "weather",
            description: "Current temperature and conditions for a place. Use when asked about the weather.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "place": {"type": "string", "description": "The place to look up"}
                },
                "required": ["place"]
            }),
            run: |_cfg: &crate::config::BotConfig, args: &Value| {
                // Stands in for the doc example's open-meteo fetch: same
                // JSON-out shape (`provider` self-label included).
                Box::pin(async move {
                    let place = args
                        .get("place")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    anyhow::ensure!(!place.trim().is_empty(), "weather.now: empty place");
                    Ok(json!({ "provider": "open-meteo", "temperature_c": 21.5 }))
                })
            },
        }
    }

    #[test]
    fn worked_example_weather_skill_is_routable_and_gated() {
        let args = json!({"place": "Praha"});

        // Gated on: the derived OpenAI tool def is well-shaped, the
        // whitelist admits the skill, dispatch routes it to the runner.
        let on = SkillRegistry::from_skills(vec![weather_example_spec()], ["weather"]);
        let defs = on.tool_defs();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0]["type"], "function");
        // Wire name (2026-10-20 provider validation); dispatch resolves
        // both the wire and the real form back to the dot-named skill.
        assert_eq!(defs[0]["function"]["name"], "weather_now");
        let served_wire = rt().block_on(on.dispatch(&test_cfg(), "weather_now", &args));
        assert_eq!(
            served_wire.unwrap(),
            json!({"provider": "open-meteo", "temperature_c": 21.5}),
            "the model can only ever name the wire form — it must dispatch"
        );
        assert_eq!(
            defs[0]["function"]["parameters"]["required"],
            json!(["place"])
        );
        assert_eq!(
            on.enabled_skill_names(),
            vec!["weather.now"],
            "the whitelist is derived from the same table"
        );
        let served = rt().block_on(on.dispatch(&test_cfg(), "weather.now", &args));
        assert_eq!(
            served.unwrap(),
            json!({"provider": "open-meteo", "temperature_c": 21.5})
        );

        // Gated off: no tool def reaches the model, and a dispatch
        // attempt is refused in the model-recoverable shape — the
        // refusal contract, applied to a third-party plugin id.
        let off = SkillRegistry::from_skills(vec![weather_example_spec()], Vec::<String>::new());
        assert!(off.tool_defs().is_empty());
        let err = rt().block_on(off.dispatch(&test_cfg(), "weather.now", &args));
        assert_err_mentions(&err, "unknown or disabled");

        // Argument validation survives the example shape too: an empty
        // required field is a typed runner error, not a panic.
        let empty = rt().block_on(on.dispatch(&test_cfg(), "weather.now", &json!({"place": "  "})));
        assert_err_mentions(&empty, "empty place");
    }

    #[test]
    fn unknown_plugin_id_in_config_fails_loud() {
        assert!(
            SkillRegistry::check_config_enabled(&["websearch".into(), "squeeze".into()], &[],)
                .is_ok()
        );
        let err = SkillRegistry::check_config_enabled(&["websearch_typo".into()], &[])
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
            // 2026-10-20: the wire name is the SANITIZED form — dots fold
            // to underscores (OpenAI/OpenRouter tool-name validation only
            // allows a-zA-Z0-9_-; dotted names 400'd every tool-attached
            // chat through litellm's remote legs). Dispatch resolves them
            // back; the internal dot names stay the registry identity.
            assert!(
                f["name"].as_str().unwrap().starts_with("memory_"),
                "wire name must be the sanitized form: {:?}",
                f["name"]
            );
            assert!(f["parameters"]["properties"].is_object());
        }
    }

    /// The WIRE-name contract: tool declarations are validated by
    /// OpenAI-compatible providers against `[a-zA-Z0-9_-]+` — every
    /// emitted def must fit, with dots/colons folded and bare names
    /// untouched. `dispatch` resolves the wire form back to the real
    /// skill (the model can only ever name the wire form).
    #[test]
    fn wire_tool_names_are_provider_safe_and_resolve_back() {
        assert_eq!(wire_tool_name("memory.embed"), "memory_embed");
        assert_eq!(wire_tool_name("external:gcp"), "external_gcp");
        assert_eq!(wire_tool_name("web_search"), "web_search");
        let r = SkillRegistry::standard(PREPACKAGED_PLUGINS);
        for d in r.tool_defs() {
            let name = d["function"]["name"].as_str().unwrap();
            assert!(
                name.chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
                "provider-unsafe wire tool name: {name:?}"
            );
        }
        // The model calls the WIRE form; the registry resolves the real
        // skill. The memory plugin's seen-log proves the dot-name ran.
        let names = r.enabled_skill_names();
        assert!(
            names.iter().any(|n| n == "memory.embed"),
            "the internal dot name stays the registry identity"
        );
        let wire = wire_tool_name("memory.embed");
        assert_eq!(wire, "memory_embed");
        // Resolve by round-trip through the same matcher dispatch uses.
        assert!(
            names
                .iter()
                .find(|n| wire_tool_name(n) == wire)
                .is_some_and(|n| n == "memory.embed")
        );
        // An exact real name also resolves (bare callers).
        assert!(
            names
                .iter()
                .any(|n| n.as_str() == wire_tool_name("web_search"))
        );
    }

    /// Phase 3a registry-level proof: an external skill joined via
    /// `with_external` dispatches through `registry.dispatch` (the tool
    /// loop's actual call path) to a real local HTTP stub — tool schema
    /// shape, namespaced gating, the HTTP hop, and the refusal contract
    /// all in one pass. invoke()'s own tests cover the wire details;
    /// this one covers the JOIN.
    #[tokio::test]
    async fn external_skill_dispatches_through_the_registry() {
        let outcome = serde_json::json!({"provider": "open-meteo", "temperature_c": 21.5});
        // The stub lives on THIS test's runtime — spawning it into a
        // throwaway block_on runtime dies with that runtime and every
        // later dispatch hits "connection refused".
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let serve_outcome = outcome.clone();
        tokio::spawn(async move {
            let app = axum::Router::new().route(
                "/rpc",
                axum::routing::post(move || {
                    let body = serve_outcome.clone();
                    async move { axum::Json(body) }
                }),
            );
            axum::serve(listener, app).await.unwrap();
        });
        let make_plugin = |endpoint: String| crate::external::ExternalPlugin {
            id: "weather".into(),
            endpoint,
            secret: None,
            timeout_secs: 5,
            skills: vec![crate::external::ExternalSkill {
                name: "weather.now".into(),
                description: "current conditions".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {"place": {"type": "string"}},
                    "required": ["place"]
                }),
            }],
        };

        // Gated ON through the namespaced id — the way bot.toml says it.
        let reg = SkillRegistry::standard(["external:weather"])
            .with_external(vec![make_plugin(format!("http://{addr}/rpc"))])
            .expect("drop joins cleanly");
        let names = reg.enabled_skill_names();
        assert!(
            names.iter().any(|n| n == "weather.now"),
            "whitelist: {names:?}"
        );
        // The tool def reaches the chat payload shaped right — with the
        // WIRE name (2026-10-20 provider validation; dispatch resolves
        // back to the dot name).
        let defs = reg.tool_defs();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0]["function"]["name"], "weather_now");
        // Dispatch — the registry → HTTP hop the tool loop exercises.
        let hit = reg
            .dispatch(&test_cfg(), "weather.now", &json!({"place": "Praha"}))
            .await;
        assert_eq!(hit.unwrap(), outcome);

        // The plugin NOT in [plugins] enabled → refusal, no HTTP call.
        let off = SkillRegistry::standard(Vec::<String>::new())
            .with_external(vec![make_plugin(format!("http://{addr}/rpc"))])
            .unwrap();
        let refused = off
            .dispatch(&test_cfg(), "weather.now", &json!({"place": "Praha"}))
            .await;
        assert_err_mentions(&refused, "unknown or disabled");
    }

    /// An owner-only drop (id in OWNER_ONLY_PLUGINS) with args: a gcp
    /// bridge — proves the owner-mode hard gate: without `owner_privileged`,
    /// the skills are INVISIBLE (no tool defs) and refused at dispatch
    /// even if the gating id somehow rode `enabled`; with it, the skills
    /// advertise and dispatch.
    fn make_gcp_plugin(endpoint: String) -> crate::external::ExternalPlugin {
        crate::external::ExternalPlugin {
            id: "gcp".into(),
            endpoint,
            secret: None,
            timeout_secs: 5,
            skills: vec![crate::external::ExternalSkill {
                name: "gcp.status".into(),
                description: "project + auth status".into(),
                parameters: json!({"type": "object", "properties": {}}),
            }],
        }
    }

    #[tokio::test]
    async fn owner_only_skills_refused_without_privileged_mode() {
        // A deliberately MISROuted config: the owner-only gate id in the
        // plain enabled list would be caught by check_config_enabled at
        // load — this test builds the registry that bug would produce, to
        // prove the second, dispatch-level gate holds regardless.
        let reg = SkillRegistry::standard(["external:gcp"])
            .with_external(vec![make_gcp_plugin("http://127.0.0.1:1/rpc".into())])
            .expect("drop joins");
        // No tool def reaches the model on any non-owner surface.
        assert!(reg.tool_defs().is_empty());
        // And an invoke is refused like any disabled skill.
        let refused = reg.dispatch(&test_cfg(), "gcp.status", &json!({})).await;
        assert_err_mentions(&refused, "unknown or disabled");
    }

    #[tokio::test]
    async fn owner_privileged_mode_admits_owner_only_skills() {
        let reg = SkillRegistry::standard(Vec::<String>::new())
            .join(&["external:gcp".to_string()])
            .owner_privileged()
            .with_external(vec![make_gcp_plugin("http://127.0.0.1:1/rpc".into())])
            .expect("drop joins");
        let defs = reg.tool_defs();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0]["function"]["name"], "gcp_status");
        // Dispatch reaches the (dead-endpoint) dispatcher — the typed
        // error proves routing through invoke(), not the refusal path.
        let err = reg
            .dispatch(&test_cfg(), "gcp.status", &json!({}))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("external:gcp"), "routed: {err}");
    }

    #[test]
    fn owner_only_id_may_not_ride_the_plain_enabled_list() {
        let exterior = make_gcp_plugin("https://10.9.9.9:12345/rpc".into());
        let err = SkillRegistry::check_config_enabled(&["external:gcp".into()], &[exterior])
            .expect_err("owner-only id in [plugins] enabled must fail loud");
        let msg = err.to_string();
        assert!(msg.contains("owner-only"), "names the boundary: {msg}");
        assert!(msg.contains("owner_only"), "points at the right key: {msg}");
    }

    #[test]
    fn owner_only_list_validates_against_drops() {
        let exterior = make_gcp_plugin("https://10.9.9.9:12345/rpc".into());
        assert!(SkillRegistry::check_owner_only(&["external:gcp".into()], &[exterior]).is_ok());
        // Bare id: not the external-drop form.
        let err = SkillRegistry::check_owner_only(&["gcp".into()], &[]) // id never matches anyway
            .expect_err("bare id must fail");
        assert!(err.to_string().contains("external:"));
        // External form but no such drop in plugins.d → fail loud.
        let err = SkillRegistry::check_owner_only(&["external:gcp".into()], &[])
            .expect_err("missing drop must fail loud");
        assert!(err.to_string().contains("plugins.d"));
        // Unknown plugin family entirely.
        let err = SkillRegistry::check_owner_only(&["external:weather".into()], &[])
            .expect_err("unknown drop must fail loud");
        assert!(err.to_string().contains("external:weather"));
    }
}
