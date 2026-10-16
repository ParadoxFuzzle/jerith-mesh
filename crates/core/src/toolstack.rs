//! The AI tool stack: optional companions the installer can set up, plus
//! the small-model tier catalog the setup flow offers.
//!
//! Two halves live here:
//!
//! 1. **Stack tools** — Ollama, LiteLLM, LangChain, LangGraph, LangSmith
//!    and PostgreSQL+pgvector. Each knows how to *detect* itself (a fast
//!    version probe) and how it would be *installed* per OS (concrete
//!    shell commands the node CLI runs — with the operator's explicit
//!    consent — never silently).
//! 2. **Small models** — the curated Ollama tags for the three local
//!    tiers the mesh routes chats across: *thinking* (used tools and
//!    reasoning), *vision* (image content) and *embedding* (local
//!    memory). Users pick per tier during `jerith setup`; the picks are
//!    written into config (`runtime_deep_model`, `runtime_vision_model`,
//!    `specialists.embedding_model`).
//!
//! Detection is deliberately pure and cheap (one version probe each) so
//! it is safe to call from the interactive setup prompt.

use crate::hw::OsKind;

// ============================================================================
// Small-model tier catalog
// ============================================================================

/// The four local routing tiers a pick applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelTier {
    /// Tool-calling tier (mesh `tool` tier): any chat request that
    /// carries function definitions. Deliberately separate from the
    /// thinking tier: tool-calling is a *format-reliability* property,
    /// not a depth property (qwen3:8b is a fine thinker whose tool
    /// parsing chokes while thinking; nemotron-3-nano:4b is 3/3 clean).
    Tool,
    /// Tools + reasoning tier (mesh `deep` tier; routed on complex prompts).
    Thinking,
    /// Image-content tier (mesh `vision` tier).
    Vision,
    /// Local memory embeddings (nomic-family Matryoshka models; other
    /// embedding models work but lose the truncation savings).
    Embedding,
}

impl ModelTier {
    pub fn label(self) -> &'static str {
        match self {
            ModelTier::Tool => "tool-calling",
            ModelTier::Thinking => "thinking / tools",
            ModelTier::Vision => "vision",
            ModelTier::Embedding => "embedding",
        }
    }

    pub fn config_field(self) -> &'static str {
        match self {
            ModelTier::Tool => "runtime_tool_model",
            ModelTier::Thinking => "runtime_deep_model",
            ModelTier::Vision => "runtime_vision_model",
            ModelTier::Embedding => "specialists.embedding_model",
        }
    }
}

/// One curated small model in the setup-flow picker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmallModel {
    /// Ollama tag (`jerith stack pull` and `ollama pull` accept it as-is).
    pub name: &'static str,
    pub display: &'static str,
    pub tier: ModelTier,
    /// Rough on-disk size the pull will fetch.
    pub size_hint: &'static str,
    /// One-line rationale shown in the picker.
    pub note: &'static str,
}

/// Thinking/tools tier. qwen3:8b is the mesh default (see
/// `model_router::DEFAULT_DEEP_MODEL`); the rest are user-pickable swaps.
const THINKING: &[SmallModel] = &[
    SmallModel {
        name: "qwen3:8b",
        display: "Qwen3 8B",
        tier: ModelTier::Thinking,
        size_hint: "~5 GiB",
        note: "mesh default — solid tools + reasoning balance",
    },
    SmallModel {
        name: "qwen3:4b",
        display: "Qwen3 4B",
        tier: ModelTier::Thinking,
        size_hint: "~2.6 GiB",
        note: "smaller footprint for tighter machines",
    },
    SmallModel {
        name: "deepseek-r1:8b",
        display: "DeepSeek-R1 8B",
        tier: ModelTier::Thinking,
        size_hint: "~5 GiB",
        note: "reasoning-first; sets runtime_num_predict to cap loops",
    },
];

/// Tool-calling tier: requests that carry function definitions. The
/// nemotron family is the proven-reliable local tool caller (measured
/// 3/3 clean tool_call parses on the live runtime, 2026-10-16, vs
/// qwen3:8b's parse failures whenever it is also thinking).
const TOOL: &[SmallModel] = &[
    SmallModel {
        name: "nemotron-3-nano:4b",
        display: "Nemotron-3-Nano 4B",
        tier: ModelTier::Tool,
        size_hint: "~2.6 GiB",
        note: "tool-calling tuned, very light — the mesh tool-tier default",
    },
    SmallModel {
        name: "qwen3:8b",
        display: "Qwen3 8B",
        tier: ModelTier::Tool,
        size_hint: "~5 GiB",
        note: "deeper tool reasoning — unreliable parses while thinking",
    },
];

/// Vision tier. minicpm-v4.6:1b is the mesh default
/// (`model_router::DEFAULT_VISION_MODEL`).
const VISION: &[SmallModel] = &[
    SmallModel {
        name: "minicpm-v4.6:1b",
        display: "MiniCPM-V 4.6 1B",
        tier: ModelTier::Vision,
        size_hint: "~1.5 GiB",
        note: "mesh default — tiny and fast at captions/OCR",
    },
    SmallModel {
        name: "qwen3-vl:4b",
        display: "Qwen3-VL 4B",
        tier: ModelTier::Vision,
        size_hint: "~3.4 GiB",
        note: "stronger document/chart reading, also thinks",
    },
    SmallModel {
        name: "moondream2",
        display: "Moondream 2",
        tier: ModelTier::Vision,
        size_hint: "~1.7 GiB",
        note: "minimal VQA alternative",
    },
];

/// Embedding tier, consumed by local memory. Matryoshka-capable models
/// honor `specialists.embedding_dim` truncation; others embed at full
/// width.
const EMBEDDING: &[SmallModel] = &[
    SmallModel {
        name: "nomic-embed-text-v2-moe",
        display: "Nomic Embed Text v2 MoE",
        tier: ModelTier::Embedding,
        size_hint: "~0.3 GiB",
        note: "mesh default — Matryoshka (768/512/256)",
    },
    SmallModel {
        name: "nomic-embed-text",
        display: "Nomic Embed Text (v1)",
        tier: ModelTier::Embedding,
        size_hint: "~0.27 GiB",
        note: "older, fixed 768-dim fallback pick",
    },
    SmallModel {
        name: "bge-m3",
        display: "BGE-M3",
        tier: ModelTier::Embedding,
        size_hint: "~1.2 GiB",
        note: "multilingual; full-width only (no truncation)",
    },
];

/// All curated small models for a tier, in picker order.
pub fn small_models(tier: ModelTier) -> &'static [SmallModel] {
    match tier {
        ModelTier::Tool => TOOL,
        ModelTier::Thinking => THINKING,
        ModelTier::Vision => VISION,
        ModelTier::Embedding => EMBEDDING,
    }
}

/// The mesh default pick for a tier (must belong to that tier's list).
pub fn default_small_model(tier: ModelTier) -> &'static SmallModel {
    small_models(tier)
        .first()
        .expect("every tier's catalog is non-empty")
}

/// Look up a pick by Ollama tag across the whole catalog (`qwen3:8b`,
/// `nomic-embed-text-v2-moe`, …). Free-form names outside the catalog are
/// still accepted by setup — this is only for validation + display.
pub fn find_small_model(name: &str) -> Option<&'static SmallModel> {
    let name = normalize(name);
    [
        ModelTier::Tool,
        ModelTier::Thinking,
        ModelTier::Vision,
        ModelTier::Embedding,
    ]
    .into_iter()
    .find_map(|tier| {
        small_models(tier)
            .iter()
            .find(|m| normalize(m.name) == name)
    })
}

fn normalize(name: &str) -> String {
    // `nomic-embed-text-v2-moe:latest` and `nomic-embed-text-v2-moe` are
    // the same Ollama model.
    let t = name.trim().to_lowercase();
    t.strip_suffix(":latest").unwrap_or(&t).to_string()
}

// ============================================================================
// Stack tools
// ============================================================================

/// An optional companion the installer may set up for mesh users.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StackTool {
    /// Local model runner — hosts the thinking/vision/embedding tags above
    /// and (or via LiteLLM) fronts the node's OpenAI-compatible runtime.
    Ollama,
    /// OpenAI-compatible proxy in front of any provider/runtime; also a
    /// drop-in value for `runtime_url` when steering many backends.
    LiteLLM,
    /// Orchestration framework; consumes the node's gateway as an
    /// OpenAI-compatible endpoint (see integrations/langchain/).
    LangChain,
    /// Graph/oracle orchestration on LangChain's runtime — the agent loop
    /// example in integrations/langchain/ builds on it.
    LangGraph,
    /// Tracing/observability for LangChain and LangGraph runs (optional
    /// LANGCHAIN/LANGSMITH env keys; nothing phones home otherwise).
    LangSmith,
    /// Server + pgvector extension — the node's postgres memory backend.
    Postgres,
}

impl StackTool {
    pub const ALL: &'static [StackTool] = &[
        StackTool::Ollama,
        StackTool::LiteLLM,
        StackTool::LangChain,
        StackTool::LangGraph,
        StackTool::LangSmith,
        StackTool::Postgres,
    ];

    /// CLI/slug name (`jerith stack install <name>`).
    pub fn name(self) -> &'static str {
        match self {
            StackTool::Ollama => "ollama",
            StackTool::LiteLLM => "litellm",
            StackTool::LangChain => "langchain",
            StackTool::LangGraph => "langgraph",
            StackTool::LangSmith => "langsmith",
            StackTool::Postgres => "postgres",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            StackTool::Ollama => "local model runner (hosts the small-model tiers)",
            StackTool::LiteLLM => "OpenAI-compatible proxy over any backend",
            StackTool::LangChain => "agent/orchestration framework (uses the local gateway)",
            StackTool::LangGraph => "LangChain's graph-based agent runtime",
            StackTool::LangSmith => "tracing + evals for LangChain/LangGraph",
            StackTool::Postgres => "postgresql server + pgvector (memory backend)",
        }
    }

    pub fn from_name(name: &str) -> Option<StackTool> {
        let name = name.trim().to_lowercase();
        // Accept the slug, the de-hyphenated form, and the common long
        // aliases operators actually type ("postgresql" for the database).
        StackTool::ALL.iter().copied().find(|t| {
            t.name() == name
                || t.name().replace('-', "") == name
                || (t == &StackTool::Postgres && name == "postgresql")
                || (t == &StackTool::LangChain && name == "lang-chain")
                || (t == &StackTool::LangGraph && name == "lang-graph")
                || (t == &StackTool::LangSmith && name == "lang-smith")
        })
    }

    /// The pip-installed distribution names (empty for native-only tools).
    /// Everything pip lands in the managed tools venv, never the system
    /// Python.
    pub fn pip_packages(self) -> &'static [&'static str] {
        match self {
            StackTool::LiteLLM => &["litellm"],
            StackTool::LangChain => &["langchain", "langchain-community", "langchain-openai"],
            StackTool::LangGraph => &["langgraph"],
            StackTool::LangSmith => &["langsmith"],
            _ => &[],
        }
    }

    /// Whether this tool needs the shared pip tools venv.
    pub fn uses_tools_venv(self) -> bool {
        !self.pip_packages().is_empty()
    }

    /// Fast probe: run and read output (never fails the caller — returns
    /// the raw result for the CLI to interpret).
    pub fn detect_cmd(self) -> (&'static str, &'static [&'static str]) {
        // Named constants: the returned slice must be 'static.
        const PY_CHAIN: &[&str] = &["-c", "__import__('langchain');print('ok')"];
        const PY_GRAPH: &[&str] = &["-c", "__import__('langgraph');print('ok')"];
        const PY_SMITH: &[&str] = &["-c", "__import__('langsmith');print('ok')"];
        match self {
            StackTool::Ollama => ("ollama", &["--version"]),
            StackTool::LiteLLM => ("litellm", &["--version"]),
            StackTool::LangChain => ("python3", PY_CHAIN),
            StackTool::LangGraph => ("python3", PY_GRAPH),
            StackTool::LangSmith => ("python3", PY_SMITH),
            StackTool::Postgres => ("psql", &["--version"]),
        }
    }

    /// Concrete install plan for an OS: shell commands run in order.
    pub fn install_plan(self, os: OsKind) -> Vec<String> {
        match self {
            StackTool::Ollama => match os {
                OsKind::Linux => vec!["curl -fsSL https://ollama.com/install.sh | sh".into()],
                OsKind::MacOs => vec![
                    "brew install ollama".into(),
                    "brew services start ollama".into(),
                ],
                OsKind::Windows => vec!["winget install --id Ollama.Ollama -e".into()],
                _ => vec![],
            },
            StackTool::Postgres => match os {
                OsKind::Linux => vec![
                    "sudo apt-get update".into(),
                    "sudo apt-get install -y postgresql postgresql-contrib postgresql-pgvector"
                        .into(),
                ],
                OsKind::MacOs => vec![
                    "brew install postgresql@16 pgvector".into(),
                    "brew services start postgresql@16".into(),
                    "createdb jerith 2>/dev/null || true".into(),
                ],
                OsKind::Windows => vec![
                    "winget install --id PostgreSQL.PostgreSQL -e".into(),
                    "rem then: psql -U postgres -c \"CREATE EXTENSION IF NOT EXISTS vector\""
                        .into(),
                ],
                _ => vec![],
            },
            StackTool::LiteLLM
            | StackTool::LangChain
            | StackTool::LangGraph
            | StackTool::LangSmith => {
                // pip family: the node CLI materializes the shared venv
                // first (see `venv_plan`), then the packages here. Like the
                // native tools, an unsupported OS refuses (empty plan).
                match os {
                    OsKind::Other => vec![],
                    _ => {
                        let python = Self::venv_python_expr(os);
                        self.pip_packages()
                            .iter()
                            .map(|p| format!("{python} -m pip install {p}"))
                            .collect()
                    }
                }
            }
        }
    }

    /// The venv python path expression for `os` — written for the shell
    /// the installer executes steps through (`sh -c` on unix, `cmd /C` on
    /// Windows), so the CLI can expand it with JERITH_HOME exported.
    pub fn venv_python_expr(os: OsKind) -> &'static str {
        match os {
            OsKind::Windows => "%JERITH_HOME%\\tools-venv\\Scripts\\python.exe",
            _ => "\"$JERITH_HOME/tools-venv/bin/python\"",
        }
    }

    /// One-time setup the pip tools need before their first package:
    /// create + upgrade the shared managed venv under the node home.
    pub fn venv_plan(os: OsKind) -> Vec<String> {
        let dir_expr = match os {
            OsKind::Windows => r"%JERITH_HOME%	ools-venv",
            _ => "\"$JERITH_HOME/tools-venv\"",
        };
        let python = "python3";
        vec![
            format!("{python} -m venv {dir_expr}"),
            format!(
                "{} -m pip install --upgrade pip",
                Self::venv_python_expr(os)
            ),
        ]
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiers_have_nonempty_catalogs_with_unique_names() {
        for tier in [
            ModelTier::Tool,
            ModelTier::Thinking,
            ModelTier::Vision,
            ModelTier::Embedding,
        ] {
            let models = small_models(tier);
            assert!(!models.is_empty(), "{tier:?} catalog empty");
            let mut seen = std::collections::HashSet::new();
            for m in models {
                assert!(seen.insert(m.name), "duplicate name {}", m.name);
                assert_eq!(m.tier, tier, "{} in wrong tier list", m.name);
                assert!(!m.display.is_empty() && !m.note.is_empty());
            }
        }
    }

    #[test]
    fn defaults_are_first_of_each_tier_and_match_mesh_defaults() {
        // Tool tier default is the measured-reliable tool caller.
        assert_eq!(
            default_small_model(ModelTier::Tool).name,
            "nemotron-3-nano:4b"
        );
        assert_eq!(default_small_model(ModelTier::Thinking).name, "qwen3:8b");
        assert_eq!(
            default_small_model(ModelTier::Vision).name,
            crate::model_router::DEFAULT_VISION_MODEL
        );
        assert_eq!(
            default_small_model(ModelTier::Embedding).name,
            "nomic-embed-text-v2-moe"
        );
        // And the thinking default matches the mesh's deep default.
        assert_eq!(
            default_small_model(ModelTier::Thinking).name,
            crate::model_router::DEFAULT_DEEP_MODEL
        );
    }

    #[test]
    fn find_small_model_normalizes_latest_suffix() {
        assert_eq!(
            find_small_model("nomic-embed-text-v2-moe:latest")
                .unwrap()
                .name,
            "nomic-embed-text-v2-moe"
        );
        assert_eq!(find_small_model("QWEN3:8B ").unwrap().name, "qwen3:8b");
        assert!(find_small_model("not-a-model:70b").is_none());
    }

    #[test]
    fn user_requested_small_models_all_present() {
        for name in [
            "deepseek-r1:8b",
            "qwen3:8b",
            "minicpm-v4.6:1b",
            "nomic-embed-text-v2-moe",
            "nemotron-3-nano:4b",
            "qwen3:4b",
            "qwen3-vl:4b",
            "moondream2",
        ] {
            assert!(find_small_model(name).is_some(), "missing {name}");
        }
    }

    #[test]
    fn stack_tools_parse_from_slugs() {
        for t in StackTool::ALL {
            assert_eq!(StackTool::from_name(t.name()), Some(*t));
            assert!(!t.description().is_empty());
        }
        assert_eq!(
            StackTool::from_name("postgresql"),
            Some(StackTool::Postgres)
        );
        assert_eq!(StackTool::from_name("nope"), None);
    }

    #[test]
    fn pip_tools_needs_venv_native_do_not() {
        assert!(StackTool::LangChain.uses_tools_venv());
        assert!(StackTool::LangGraph.uses_tools_venv());
        assert!(StackTool::LangSmith.uses_tools_venv());
        assert!(StackTool::LiteLLM.uses_tools_venv());
        assert!(!StackTool::Ollama.uses_tools_venv());
        assert!(!StackTool::Postgres.uses_tools_venv());
    }

    #[test]
    fn install_plans_are_per_os_and_nonempty() {
        for t in StackTool::ALL {
            for os in [OsKind::Linux, OsKind::MacOs, OsKind::Windows] {
                let plan = t.install_plan(os);
                assert!(!plan.is_empty(), "{:?} on {os:?} empty", t);
                for cmd in &plan {
                    assert!(!cmd.trim().is_empty(), "blank step in {:?}/{os:?}", t);
                }
            }
            // Unknown OS yields an empty plan — the CLI must refuse.
            assert!(t.install_plan(OsKind::Other).is_empty());
        }
        // The framework trio shares the venv plan's python.
        let lc = StackTool::LangChain.install_plan(OsKind::Linux);
        assert!(lc.iter().all(|c| c.contains("tools-venv")));
        // And every pip plan runs through the venv's python, never the
        // system interpreter.
        let lcw = StackTool::LangChain.install_plan(OsKind::Windows);
        assert!(lcw.iter().all(|c| c.contains("Scripts\\python.exe")));
        assert!(StackTool::Ollama.install_plan(OsKind::Linux)[0].contains("ollama"));
        // Postgres plan must cover the pgvector extension.
        let pg = StackTool::Postgres.install_plan(OsKind::MacOs).join(" ");
        assert!(pg.contains("pgvector"));
    }
}
