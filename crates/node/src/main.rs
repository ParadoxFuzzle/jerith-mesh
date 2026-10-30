//! `jerith` — the jerith-mesh node CLI.
//!
//! Commands:
//!   setup       guided onboarding (model → memory → contribution % → wallets)
//!   run         start the node (discovery + pings + gateway + web UI)
//!   status      machine + mesh status snapshot
//!   peers       list peers in this model's mesh
//!   model       show or switch the active model (disconnects all peers)
//!   config      get/set config values (contribution_percent, etc.)
//!   stake       show or drive the stake escrow (status | lock | unlock)

use std::io::{BufRead, Write as _};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context as _, bail};
use clap::{Parser, Subcommand};
use jerith_mesh_core::bootstrap::BootstrapClient;
use jerith_mesh_core::config::{
    MIN_CONTRIBUTION_PERCENT, MemoryBackend, NodeConfig, NodeRole, SetupChoices, default_model,
};
use jerith_mesh_core::gateway::Gateway;
use jerith_mesh_core::hw;
use jerith_mesh_core::models::{ModelId, ModelSource, detect_capable_models, recommended_model};
use jerith_mesh_core::ping::MeshState;
use jerith_mesh_core::web;
use serde_json::json;

#[derive(Parser)]
#[command(
    name = "jerith",
    about = "jerith-mesh node — P2P compute mesh for MoE LLMs",
    version,
    propagate_version = true
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

/// All `jerith setup` arguments — a boxed struct variant keeps `Cmd`
/// small (the documented clippy pattern for large argument sets).
#[derive(clap::Args)]
struct SetupArgs {
    /// Model id (curated) e.g. glm-5.3, or "huggingface:<repo>",
    /// "ollama:<name>", "path:<file>".
    #[arg(long)]
    model: Option<String>,
    /// "sqlite" or "postgres"
    #[arg(long)]
    memory: Option<String>,
    /// Postgres DSN (memory_backend postgres only):
    /// "postgres://user:pass@127.0.0.1:5432/db" — also readable from
    /// JERITH_MEMORY_DSN.
    #[arg(long = "memory-dsn")]
    memory_dsn: Option<String>,
    /// Contribution percent (>= 25).
    #[arg(long)]
    contribution: Option<u64>,
    /// contributor | requester | hybrid
    #[arg(long)]
    role: Option<String>,
    /// jerithchain | bitcoin | ethereum | dogecoin | monero | none
    #[arg(long)]
    crypto: Option<String>,
    /// Override the bootstrap node URL.
    #[arg(long)]
    bootstrap: Option<String>,
    /// Skip the live peer-count fetch during model listing.
    #[arg(long, default_value_t = false)]
    no_wait: bool,
    /// Disable the local vision specialist (MiniCPM-V; enabled by
    /// default — the interactive prompt's default as well).
    #[arg(long, default_value_t = false)]
    no_vision: bool,
    /// Disable the local embedding specialist (nomic text vectors;
    /// enabled by default — the interactive prompt's default as well).
    #[arg(long, default_value_t = false)]
    no_embeddings: bool,
    /// Ollama tag for the tool-calling tier (agent tool loops — chosen
    /// for parse RELIABILITY, not depth). Default: nemotron-3-nano:4b.
    /// "" disables the tier.
    #[arg(long = "tool-model")]
    tool_model: Option<String>,
    /// Ollama tag for the thinking/tools tier (deep tier on complex
    /// prompts). Default: qwen3:8b. "" disables the tier.
    #[arg(long = "deep-model")]
    deep_model: Option<String>,
    /// Ollama tag for the vision tier (image content). Default:
    /// minicpm-v4.6:1b. "" disables the tier.
    #[arg(long = "vision-model")]
    vision_model: Option<String>,
    /// Ollama tag memory embeds with. Default: nomic-embed-text-v2-moe.
    #[arg(long = "embedding-model")]
    embedding_model: Option<String>,
    /// OpenAI-compatible runtime base URL (scheme://host:port — the
    /// gateway appends /v1/…). Default: ollama when present, else the
    /// curated engine on :8000.
    #[arg(long = "runtime-url")]
    runtime_url: Option<String>,
    /// Install stack tools non-interactively (comma names, or "all").
    #[arg(long = "with")]
    with_stack: Option<String>,
    /// Print every step the stack install would run instead of doing
    /// anything (installs, model pulls). Nothing is executed.
    #[arg(long, default_value_t = false)]
    print_mode: bool,
}

#[derive(Subcommand)]
enum Cmd {
    /// Guided setup flow (interactive when on a TTY, flags otherwise).
    Setup(#[command(flatten)] Box<SetupArgs>),
    /// Run the node until Ctrl+C.
    Run {
        /// Override the config's model for this session (switches mesh).
        #[arg(long)]
        model: Option<String>,
    },
    /// Run a bootstrap node (the p2p.jerithai.com-style trust anchor).
    Bootstrap {
        #[arg(long, default_value_t = 8901)]
        port: u16,
        /// Advertisement TTL in milliseconds.
        #[arg(long, default_value_t = jerith_mesh_core::bootstrap_server::DEFAULT_ADVERTISE_TTL_MS)]
        ttl_ms: u64,
    },
    /// Print a status snapshot.
    Status,
    /// List peers in the current model's mesh.
    Peers,
    /// Show or switch the active model.
    Model {
        /// New model to switch to (disconnects from all current peers).
        new_model: Option<String>,
    },
    /// Read or modify config values.
    Config {
        /// e.g. contribution_percent=40, gateway_port=8900
        assignments: Vec<String>,
    },
    /// Local memory skills (strictly node-private).
    Memory {
        #[command(subcommand)]
        cmd: MemoryCmd,
    },
    /// Show or drive the stake escrow (Phase 5 tier).
    Stake {
        #[command(subcommand)]
        cmd: StakeCmd,
    },
    /// Introspect the co-hosted JerithChain node (read-only).
    Chain {
        #[command(subcommand)]
        cmd: ChainCmd,
    },
    /// Search the markdown lore vault (void-lore-index port).
    Lore {
        /// Canon question / terms.
        query: String,
        #[arg(long, default_value_t = 4)]
        top_k: usize,
    },
    /// Local expert weight store (Phase 2 tensor transport).
    Experts {
        #[command(subcommand)]
        cmd: ExpertsCmd,
    },
    /// Full node-state backup / verify / restore (Phase 2).
    Backup {
        #[command(subcommand)]
        cmd: BackupCmd,
    },
    /// Optional AI tool stack (ollama, litellm, langchain, langgraph,
    /// langsmith, postgres) + the small-model tier catalog.
    Stack {
        #[command(subcommand)]
        cmd: StackCmd,
    },
    /// Signed plugin bundle channel (docs/PLUGIN-CHANNEL.md) — install,
    /// verify and remove `.jbundle` drops without a source checkout.
    Plugin {
        #[command(subcommand)]
        cmd: PluginCmd,
    },
}

#[derive(Subcommand)]
enum PluginCmd {
    /// Install a signed bundle (.jbundle) from a local path or https URL:
    /// sha256 → publisher signature → pin → platform → node-contract →
    /// inner binary hash, THEN extract + stage the drop. A fresh HMAC
    /// secret is generated box-side on every install (the shipped one is
    /// never kept); enabling is always a manual bot.toml edit.
    Install {
        /// Local .jbundle path (.sha256/.sig companions beside it) or an https URL.
        source: String,
        /// Bypass the downgrade/equal-version guard (never the verify chain).
        #[arg(long, default_value_t = false)]
        force: bool,
        /// Receipt-log gate override (docs/PLUGIN-CHANNEL.md): proceed
        /// when the receipt log shows Broken/Truncated evidence. A
        /// crash-recovery escape — the receipt this install appends
        /// re-syncs the head witness — never a way to hide a break.
        #[arg(long, default_value_t = false)]
        receipt_allow_broken: bool,
        /// Publisher pubkey hex (default: [plugins.trusted_publishers]
        /// in bot.toml, then the legacy keys/plugin-pins.json).
        #[arg(long)]
        pubkey: Option<String>,
        /// FRESHNESS-PINNED install: the bundle must be a row of the
        /// VERIFIED channel pair at this source (an https channel.json
        /// URL or a local dir holding the pair). The full verify chain
        /// still runs; the row match (by artifact sha256) additionally
        /// binds the artifact to the index — and the receipt log
        /// records the freshness the row was checked under.
        #[arg(long)]
        channel: Option<String>,
        /// Freshness window ms for --channel installs (0 = the 24 h
        /// default) — the same knob `plugin search` and `plugin channel
        /// verify` take. An operator without the signing seed cannot
        /// "maintain" an aging first-party index; widening the window
        /// here is the honest out — the receipt records the window the
        /// row was actually checked under.
        #[arg(long, default_value_t = 0)]
        max_age_ms: u64,
    },
    /// Verify a bundle end to end without installing anything.
    Verify {
        /// The .jbundle file (.sha256/.sig companions beside it).
        path: PathBuf,
        /// Publisher pubkey hex (default: the pins file).
        #[arg(long)]
        pubkey: Option<String>,
    },
    /// Move a plugin's live drop out to plugins.d/.trash (never deleted;
    /// installed bundles and downloads stay for the audit trail).
    Remove {
        /// Plugin id (the drop dir name under plugins.d/).
        id: String,
    },
    /// Signed channel index (Phase 4): publish a channel.json + sig
    /// pair over a dist dir, or fetch + verify one. The index NAMES
    /// bundles; installing still runs the full verify chain — the
    /// channel shortens the list step, it never replaces the checks.
    Channel {
        #[command(subcommand)]
        cmd: ChannelCmd,
    },
    /// Build a signed .jbundle from a prepared staging dir — the
    /// publisher's shape in docs/PLUGIN-CHANNEL.md: `bundle.toml`,
    /// `plugin.toml`, `README.md`, `bin/<binary>`. Writes the bundle
    /// plus its `.sha256` and `.sig` companions. The signing key is the
    /// 32-byte hex seed file, 0600 and gitignored per the key model in
    /// docs/PLUGIN-CHANNEL.md, and `bundle.toml`'s `publisher` field
    /// MUST equal that key's pubkey.
    Build {
        /// Prepared staging dir.
        source: PathBuf,
        /// Output .jbundle path (companions land beside it).
        out: PathBuf,
        /// Signing key: 32-byte hex seed file. Default:
        /// keys/plugin-releases.hex (JERITH_BUNDLE_KEY overrides).
        #[arg(long)]
        key: Option<PathBuf>,
    },
    /// Search a VERIFIED signed channel index (Phase 4's human lane):
    /// fetch + verify the pair first (key pinning + freshness — the
    /// full channel checks), then filter rows by substring. Rows name
    /// artifacts; installing one still runs the full bundle verify
    /// chain — the channel shortens the find, never the trust.
    Search {
        /// Match substring — case-insensitive over id, version, os/arch
        /// and url. Omitted = list every row.
        query: Option<String>,
        /// Channel source: an https channel.json URL (the sig rides
        /// `….sig`) or a local directory holding the pair. Default:
        /// $JERITH_CHANNEL_URL.
        #[arg(long)]
        channel: Option<String>,
        /// Pinned publisher anchor (64-hex). Default: --anchor >
        /// JERITH_CHANNEL_ANCHOR > keys/plugin-releases.pub.
        #[arg(long)]
        anchor: Option<String>,
        /// Freshness window ms (0 = the 24 h default).
        #[arg(long, default_value_t = 0)]
        max_age_ms: u64,
    },
    /// Operator trust over publishers (Phase 3, docs/PLUGIN-CHANNEL.md
    /// key model): show the per-plugin-id pins across bot.toml's
    /// [plugins.trusted_publishers] (the primary trust root) and the
    /// legacy pins file, and rotate an id's key with zero trust-lag.
    Trust {
        #[command(subcommand)]
        cmd: TrustCmd,
    },
    /// Show the plugin-lane receipt log (<home>/plugin.json): installs,
    /// removals, pin rotations and channel verifications, each with the
    /// pin (and its SOURCE) that authorized it and the channel
    /// freshness observed at check time. Append-only, 0600.
    Receipts {
        /// Tail length (newest entries last).
        #[arg(long, default_value_t = 20)]
        count: usize,
        /// Print the whole log instead of the tail.
        #[arg(long, default_value_t = false)]
        all: bool,
        /// One-line verdict, scriptable: exit 0 = chain intact, exit 1
        /// = Broken/Truncated. Mutually exclusive with the human
        /// printout flags (docs/PLUGIN-CHANNEL.md receipt verdicts).
        #[arg(long, default_value_t = false, conflicts_with_all = ["all", "count"])]
        verdict: bool,
    },
}

#[derive(Subcommand)]
enum TrustCmd {
    /// List the pinned publisher keys per plugin id and where each pin
    /// lives ([plugins.trusted_publishers] in bot.toml vs the legacy
    /// keys/plugin-pins.json).
    Show,
    /// Re-pin <id> to --new <64-hex pubkey> in bot.toml's
    /// [plugins.trusted_publishers]. No trust-lag: installs signed by
    /// the OLD key are refused the moment rotation lands; the next
    /// install verifies against the new pin (or nothing installs).
    Rotate {
        /// Plugin id (the pin row's key).
        id: String,
        /// The new Ed25519 publisher pubkey, exactly 64 hex chars.
        #[arg(long)]
        new: String,
    },
}

#[derive(Subcommand)]
enum ChannelCmd {
    /// Build + sign a publisher's channel.json (+ .sig) from the signed
    /// bundle triples in a directory (default: dist/). Every triple
    /// there (id-version-os-arch.jbundle with .sha256 + .sig beside it)
    /// becomes a row; the anchor-of-record is dist's own key.
    Build {
        /// Directory holding the *-.jbundle triples (default: dist).
        dir: Option<PathBuf>,
        /// https prefix for each row's artifact URL (e.g. a release's
        /// .../releases/latest/download).
        #[arg(long)]
        url_base: String,
        /// Signing key: 32-byte hex seed file (JERITH_BUNDLE_KEY / the
        /// bundle-build default apply).
        #[arg(long)]
        key: Option<PathBuf>,
    },
    /// Fetch (https) or read (local dir) a channel pair and verify it:
    /// key pinning (--anchor; default the committed
    /// keys/plugin-releases.pub in the repo checkout, because the
    /// first-party channel outranks no source the operator hasn't
    /// pinned), freshness window, then row parity.
    Verify {
        /// A channel.json URL (https) a directory containing the pair.
        source: String,
        /// Pinned publisher anchor: 64-hex pubkey. Default:
        /// keys/plugin-releases.pub (JERITH_CHANNEL_ANCHOR overrides).
        #[arg(long)]
        anchor: Option<String>,
        /// Freshness window ms (0 = the 24 h default).
        #[arg(long, default_value_t = 0)]
        max_age_ms: u64,
    },
}

#[derive(Subcommand)]
enum BackupCmd {
    /// Snapshot the node's state into one checksummed file (config + bot
    /// config + persona files + the whole data tree; expert weights are
    /// omitted unless --include-experts — a moved node re-pulls replicas).
    Create {
        /// Explicit output path (default <home>/backups/jerith-state-<ms>.jbak).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Also archive the expert weight store (may be gigabytes).
        #[arg(long, default_value_t = false)]
        include_experts: bool,
    },
    /// Check a backup's integrity end to end (magic, manifest, checksums,
    /// exact EOF) without touching any state.
    Verify {
        /// The .jbak file to check.
        path: PathBuf,
    },
    /// Re-install a verified backup's state under this node's home dir.
    /// Existing files are left alone unless --force. Copying a backup to
    /// another machine (export JERITH_HOME first) and restoring is the
    /// migrate story.
    Restore {
        /// The .jbak file to restore from.
        path: PathBuf,
        /// Overwrite files that already exist (default: skip them).
        #[arg(long, default_value_t = false)]
        force: bool,
    },
}

/// The `--activation` override for `jerith experts import`: auto follows
/// the arch tag (bert-family → gelu, else silu); the rest force one tag.
#[derive(clap::ValueEnum, Clone, Copy)]
enum ImportActivation {
    Auto,
    Silu,
    Gelu,
}

impl ImportActivation {
    fn into_core(self) -> Option<jerith_mesh_core::expert_store::Activation> {
        use jerith_mesh_core::expert_store::Activation;
        match self {
            Self::Auto => None,
            Self::Silu => Some(Activation::Silu),
            Self::Gelu => Some(Activation::Gelu),
        }
    }
}

#[derive(Subcommand)]
enum ExpertsCmd {
    /// List the experts this node serves, plus cache/store stats.
    List,
    /// Write a synthetic expert into the store (the same deterministic
    /// weights every node computes, so it is safe for mesh smoke tests).
    Make {
        /// Expert id, e.g. L3.E17.
        id: String,
        #[arg(long, default_value_t = 8)]
        hidden: usize,
        #[arg(long, default_value_t = 6)]
        intermediate: usize,
    },
    /// Run one local expert forward pass on an input vector (offline check
    /// that the weights and the math work before peers dispatch to us).
    Run {
        /// Expert id.
        id: String,
        /// Comma-separated input floats (must be `hidden` wide).
        #[arg(long, value_delimiter = ',')]
        input: Vec<f32>,
    },
    /// Import a GGUF model file: split its MoE stacks / dense FFN layers
    /// into expert containers + routers (FEATURES §13, the Colibri-side
    /// layer split). Every un-importable layer is REPORTED by name — the
    /// store never keeps a half-written expert.
    Import {
        /// Path to the .gguf file (F32/F16/BF16/Q4_0/Q8_0).
        path: PathBuf,
        /// Force the activation tagging for plain-MLP experts
        /// (default: bert-family archs → gelu, everything else → silu).
        #[arg(long, value_enum, default_value_t = ImportActivation::Auto)]
        activation: ImportActivation,
    },
}

#[derive(Subcommand)]
enum ChainCmd {
    /// GET /status — height, supply, wallet counts, mining + security stats.
    Status,
    /// Wallet table straight from the chain's SQLite ledger (read-only).
    Wallets {
        #[arg(long, default_value_t = 25)]
        limit: u32,
        /// Sort by balance instead of address.
        #[arg(long, default_value_t = false)]
        by_balance: bool,
    },
    /// GET /balance for one wallet (Discord id or chain address).
    Balance {
        /// Discord id string or chain address.
        who: String,
    },
    /// GET /mystate — mining leaderboard (top by total mined).
    Miners {
        #[arg(long, default_value_t = 100)]
        limit: u32,
    },
    /// GET /block/{height}.
    Block { height: u64 },
    /// GET /tx/{txid}.
    Tx { txid: String },
}

#[derive(Subcommand)]
enum MemoryCmd {
    /// Embed + store a text entry.
    Add { text: String },
    /// Vector-search local memory.
    Search {
        query: String,
        #[arg(long, default_value_t = 5)]
        top_k: usize,
    },
    /// Number of stored entries.
    Count,
    /// Delete an entry by id.
    Forget { id: i64 },
    /// Entries similar to a seed entry (no re-embedding needed, works offline).
    Similar {
        /// The seed entry's id (from `memory search` / `memory add`).
        id: i64,
        #[arg(long, default_value_t = 5)]
        top_k: usize,
    },
    /// Store health: count, dimensions, providers, footprint.
    Stats,
    /// Copy every row from the previously-configured store into the
    /// store the config now names (migration guidance direction — the
    /// inverse of the switch `jerith config set memory_backend` made).
    /// Vectors are copied verbatim (no re-embedding); duplicates by text
    /// are skipped, so re-running is idempotent.
    Migrate {
        /// Empty the source store after a verified copy.
        #[arg(long)]
        drop_source: bool,
    },
}

#[derive(Subcommand)]
enum StakeCmd {
    /// Fetch the live chain escrow view and true-up the local ledger.
    Status,
    /// Escrow to the chain. Without an amount: top up the shortfall towards
    /// the configured `[crypto.stake] required_uj` (the run loop does this
    /// automatically once per process; this commands it by hand).
    Lock {
        /// Amount in JER (fractional, e.g. 2.5).
        amount_jer: Option<f64>,
    },
    /// Return escrow to the wallet. Without an amount: everything this
    /// node's marker says it escrowed.
    Unlock {
        /// Amount in JER (fractional, e.g. 2.5).
        amount_jer: Option<f64>,
    },
}

/// `jerith stack …` subcommands.
#[derive(Subcommand)]
enum StackCmd {
    /// List the stack tools and the small-model tier catalog.
    List {
        /// Show only the model catalog (not the tools).
        #[arg(long, default_value_t = false)]
        models: bool,
    },
    /// Check which stack tools are already available on this machine.
    Status,
    /// Install one tool with its official installer (each step is echoed
    /// before it runs — running the command IS the consent).
    Install { tool: String },
    /// Print an install plan instead of running it.
    InstallPlan { tool: String },
    /// Pull a small model into local ollama (tier tags accepted).
    Pull { model: String },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(run(cli.cmd))
}

async fn run(cmd: Cmd) -> anyhow::Result<()> {
    match cmd {
        Cmd::Setup(args) => setup_flow(*args),
        Cmd::Run { model } => run_node(model).await,
        Cmd::Bootstrap { port, ttl_ms } => {
            // §10 anti-sybil live half: the bootstrap weights reporter
            // establishment by registry age OR visible locked stake —
            // pass the chain base when this node runs on Jerithchain; a
            // broken/absent config degrades to the age rule alone. The
            // resolver is the same one the `chain`/`stake` CLI commands
            // use ($JERITH_CHAIN_URL → config `chain_url` → the
            // co-hosted :8322) — /stake lives on the chain NODE api, not
            // on the miner-facing wallet endpoint chain_rpc_url picks.
            let chain_base = jerith_mesh_core::config::NodeConfig::load()
                .ok()
                .flatten()
                .filter(|cfg| {
                    cfg.crypto.chain == jerith_mesh_core::config::CryptoChain::Jerithchain
                })
                .map(|cfg| chain_base(Some(&cfg)));
            jerith_mesh_core::bootstrap_server::serve(
                jerith_mesh_core::bootstrap_server::BootstrapOptions {
                    port,
                    advertise_ttl_ms: ttl_ms,
                    chain_base,
                },
            )
            .await
        }
        Cmd::Status => status_cmd().await,
        Cmd::Peers => peers_cmd().await,
        Cmd::Model { new_model } => model_cmd(new_model).await,
        Cmd::Config { assignments } => config_cmd(assignments).await,
        Cmd::Memory { cmd } => memory_cmd(cmd).await,
        Cmd::Stake { cmd } => stake_cmd(cmd).await,
        Cmd::Chain { cmd } => chain_cmd(cmd).await,
        Cmd::Lore { query, top_k } => lore_cmd(&query, top_k).await,
        Cmd::Experts { cmd } => experts_cmd(cmd).await,
        Cmd::Backup { cmd } => backup_cmd(cmd).await,
        Cmd::Stack { cmd } => stack_cmd(cmd),
        Cmd::Plugin { cmd } => plugin_cmd(cmd).await,
    }
}

/// `jerith lore <query>` — sourced canon answers from the lore vault
/// (void-lore-index port; vault default is the OpenClaw borrow).
/// `jerith experts …` — the local expert weight store (Phase 2 tensor
/// transport). Reads the same `[experts]` config the node serves from.
async fn experts_cmd(cmd: ExpertsCmd) -> anyhow::Result<()> {
    use jerith_mesh_core::expert_store::{ExpertStore, ExpertWeights};

    let cfg = jerith_mesh_core::config::NodeConfig::load()?.unwrap_or_default();
    let store = ExpertStore::new(cfg.experts.effective_dir(), cfg.experts.cache_mb)?;
    match cmd {
        ExpertsCmd::List => {
            let hosted = store.hosted();
            println!("expert store: {}", store.dir().display());
            println!(
                "cache budget: {} MiB · resident: {:.2} MiB · tensor-only serving: {}",
                store.cache_budget_bytes() / (1024 * 1024),
                store.resident_bytes() as f64 / (1024.0 * 1024.0),
                cfg.experts.tensors_only
            );
            if hosted.is_empty() {
                println!("hosted: (none) — add one with `jerith experts make <id>`");
            } else {
                println!("hosted ({}):", hosted.len());
                for id in &hosted {
                    let marker = if store.is_resident(id) {
                        "resident"
                    } else {
                        "on disk "
                    };
                    println!("  {id}  [{marker}]");
                }
            }
        }
        ExpertsCmd::Make {
            id,
            hidden,
            intermediate,
        } => {
            anyhow::ensure!(
                hidden > 0 && intermediate > 0,
                "hidden and intermediate must be positive"
            );
            let weights = ExpertWeights::synthetic(&id, hidden, intermediate);
            let bytes = weights.bytes();
            store.put(&weights)?;
            // Prove the file just written parses and computes before
            // announcing success.
            let served = store.run(&id, &vec![0.0f32; hidden])?;
            anyhow::ensure!(
                served.len() == hidden,
                "self-check failed: forward returned {} floats, expected {hidden}",
                served.len()
            );
            println!(
                "wrote expert {id} ({hidden}→{intermediate}→{hidden}, {:.1} KiB) to {}",
                bytes as f64 / 1024.0,
                store.dir().join(format!("{id}.expert")).display()
            );
        }
        ExpertsCmd::Import { path, activation } => {
            let report =
                jerith_mesh_core::gguf_import::import_gguf(&path, &store, activation.into_core())?;
            println!("arch: {}", report.arch);
            println!(
                "moe layers: {} · dense layers: {}",
                report
                    .moe_layers
                    .iter()
                    .map(|l| format!("blk.{l}"))
                    .collect::<Vec<_>>()
                    .join(", "),
                report
                    .dense_layers
                    .iter()
                    .map(|l| format!("blk.{l}"))
                    .collect::<Vec<_>>()
                    .join(", "),
            );
            println!(
                "experts written ({}): {}",
                report.experts_written.len(),
                report.experts_written.join(", ")
            );
            println!("routers written ({})", report.routers_written.len());
            if !report.notes.is_empty() {
                println!("notes ({}):", report.notes.len());
                for n in &report.notes {
                    println!("  ⚠ {n}");
                }
            }
        }
        ExpertsCmd::Run { id, input } => {
            let out = store.run(&id, &input)?;
            println!(
                "{id} ← [{}]",
                input
                    .iter()
                    .map(|f| format!("{f:.5}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            println!(
                "     → [{}]",
                out.iter()
                    .map(|f| format!("{f:.5}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            let s = store.stats();
            println!(
                "     store: {} hosted, {} resident bytes, {} hits / {} misses / {} loads",
                store.hosted().len(),
                store.resident_bytes(),
                s.hits,
                s.misses,
                s.loads
            );
        }
    }
    Ok(())
}

async fn lore_cmd(query: &str, top_k: usize) -> anyhow::Result<()> {
    let vault = jerith_mesh_core::lore::default_vault_dir()
        .context("no lore vault — set $JERITH_LORE_VAULT to a directory of .md files")?;
    let index = jerith_mesh_core::lore::build(&vault)?;
    let hits = index.search(query, top_k);
    if hits.is_empty() {
        println!(
            "no lore matched \"{query}\" ({} entries indexed)",
            index.len()
        );
        return Ok(());
    }
    println!("{} entries indexed · top {}:\n", index.len(), hits.len());
    for h in hits {
        println!("📌 {}", h.id);
        if let Some(t) = &h.heading {
            println!("   section: {t}");
        }
        println!("   score {} · tags: {}", h.score, h.tags.join(", "));
        println!("   {}\n", h.excerpt);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// JerithChain node integration — read-only introspection of the co-hosted
// chain (`jerith-node-dev` on :8322). The mesh stack shares the chain api
// token (economics::read_token: $JERITH_TOKEN → ~/.jerith-mesh/keys/api.token)
// and, for the wallet table, opens the chain's SQLite ledger read-only.
// ---------------------------------------------------------------------------

/// Chain RPC base: $JERITH_CHAIN_URL override → config `chain_url` → the
/// co-hosted default (127.0.0.1:8322).
fn chain_base(cfg: Option<&NodeConfig>) -> String {
    if let Ok(u) = std::env::var("JERITH_CHAIN_URL") {
        let u = u.trim().trim_end_matches('/').to_string();
        if !u.is_empty() {
            return u;
        }
    }
    match cfg {
        Some(c) if !c.chain_url.trim().is_empty() => {
            c.chain_url.trim().trim_end_matches('/').to_string()
        }
        _ => "http://127.0.0.1:8322".to_string(),
    }
}

/// Chain SQLite ledger path: config `chain_data_dir` → $JERITH_CHAIN_DATA
/// → the co-hosted jerithchain repo layout (this machine's convention,
/// matching deploy/start_jerith_node.sh's cwd). Always opened read-only.
fn chain_ledger_path(cfg: Option<&NodeConfig>) -> anyhow::Result<std::path::PathBuf> {
    let from_dir = |dir: &std::path::Path| dir.join("jerith.db");
    let configured = cfg
        .and_then(|c| c.chain_data_dir.clone())
        .filter(|s| !s.trim().is_empty())
        .map(|s| from_dir(std::path::Path::new(s.trim())));
    if let Some(p) = configured {
        anyhow::ensure!(p.exists(), "chain ledger not found at {}", p.display());
        return Ok(p);
    }
    if let Ok(d) = std::env::var("JERITH_CHAIN_DATA") {
        let d = d.trim().to_string();
        if !d.is_empty() {
            let p = from_dir(std::path::Path::new(&d));
            anyhow::ensure!(p.exists(), "chain ledger not found at {}", p.display());
            return Ok(p);
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        let p = from_dir(&std::path::Path::new(&home).join("Documents/Projects/jerithchain/data"));
        if p.exists() {
            return Ok(p);
        }
    }
    bail!(
        "chain ledger not found — set `chain_data_dir = \"<jerithchain repo>/data\"` \\
         in config.toml or $JERITH_CHAIN_DATA"
    )
}

/// GET one of the chain's token-authed JSON endpoints and return the body.
async fn chain_get(
    _cfg: Option<&NodeConfig>,
    base: &str,
    path: &str,
    query: &[(&str, &str)],
) -> anyhow::Result<serde_json::Value> {
    let token = jerith_mesh_core::economics::read_token().context(
        "no chain api token — stage ~/.jerith-mesh/keys/api.token (see docs/RUNBOOK.md) \\
         or set $JERITH_TOKEN",
    )?;
    let mut url = format!("{base}{path}");
    if !query.is_empty() {
        url.push('?');
        url.push_str(
            &query
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join("&"),
        );
    }
    let resp = jerith_mesh_core::http::mesh_http()
        .get(&url)
        .bearer_auth(token)
        .timeout(Duration::from_secs(15))
        .send()
        .await
        .with_context(|| format!("chain node unreachable at {url}"))?;
    let status = resp.status();
    let body: serde_json::Value = resp.json().await.context("chain node returned non-JSON")?;
    if !status.is_success() {
        anyhow::bail!(
            "chain {path} failed ({status}): {}",
            body.get("detail").and_then(|d| d.as_str()).unwrap_or("")
        );
    }
    Ok(body)
}

/// One row of the chain's wallet table (balances ⟕ users ⟕ mining_state).
#[derive(Debug, Clone, serde::Serialize)]
struct ChainWalletRow {
    address: String,
    balance_uj: i64,
    locked_uj: i64,
    nonce: i64,
    discord_id: Option<u64>,
    total_mined_uj: i64,
}

/// Read the wallet table out of the chain ledger. Duck-typed to the live
/// schema (balances ⟕ users.discord_id ⟕ mining_state); missing links stay
/// None. Connection is expected read-only (mode=ro).
fn load_chain_wallets(
    conn: &rusqlite::Connection,
    limit: u32,
    by_balance: bool,
) -> anyhow::Result<Vec<ChainWalletRow>> {
    let mut stmt = conn.prepare(
        "SELECT b.address, b.balance, b.nonce, b.locked, u.discord_id, COALESCE(m.total_mined, 0)
         FROM balances b
         LEFT JOIN users u ON u.address = b.address
         LEFT JOIN mining_state m ON m.discord_id = u.discord_id",
    )?;
    let mut rows: Vec<ChainWalletRow> = stmt
        .query_map([], |r| {
            let address: String = r.get(0)?;
            let balance_uj: i64 = r.get(1)?;
            let nonce: i64 = r.get(2)?;
            let locked_uj: i64 = r.get(3)?;
            let discord_id: Option<String> = r.get(4)?;
            let total_mined_uj: i64 = r.get(5)?;
            Ok(ChainWalletRow {
                address,
                balance_uj,
                locked_uj,
                nonce,
                discord_id: discord_id.as_deref().and_then(|s| s.parse::<u64>().ok()),
                total_mined_uj,
            })
        })?
        .collect::<Result<_, _>>()?;
    if by_balance {
        rows.sort_by_key(|row| std::cmp::Reverse(row.balance_uj));
    } else {
        rows.sort_by_key(|row| row.address.clone());
    }
    rows.truncate((limit.max(1) as usize).min(500));
    Ok(rows)
}
/// `jerith backup create|verify|restore` — the full node-state
/// backup/restore/migrate story (Phase 2). The container format lives in
/// `core::backup`; this is the human surface: one line per report field.
async fn backup_cmd(cmd: BackupCmd) -> anyhow::Result<()> {
    use jerith_mesh_core::backup::{self, BackupOptions};

    let home = jerith_mesh_core::config::NodeConfig::home();
    match cmd {
        BackupCmd::Create {
            out,
            include_experts,
        } => {
            let report =
                backup::create_into(&home, out.as_deref(), BackupOptions { include_experts })?;
            println!(
                "backup written: {} ({} files, {:.2} MiB)",
                report.path.display(),
                report.entries,
                report.total_bytes as f64 / (1024.0 * 1024.0)
            );
            if !report.skipped.is_empty() {
                println!("skipped ({}):", report.skipped.len());
                for s in &report.skipped {
                    println!("  - {s}");
                }
            }
            println!("note: config.toml carries the node identity key — keep the backup private");
        }
        BackupCmd::Verify { path } => {
            let report = backup::verify(&path)?;
            println!(
                "backup OK: {} files, {:.2} MiB, format v{}",
                report.entries,
                report.total_bytes as f64 / (1024.0 * 1024.0),
                report.version
            );
            println!(
                "from node {} at {}",
                report.node_id.unwrap_or_else(|| "(unknown)".into()),
                report.created_at_ms
            );
        }
        BackupCmd::Restore { path, force } => {
            let report = backup::restore_into(&home, &path, force)?;
            println!(
                "restored {} files under {}",
                report.written.len(),
                home.display()
            );
            if !report.skipped_existing.is_empty() {
                println!(
                    "skipped {} existing file(s) (rerun with --force to overwrite)",
                    report.skipped_existing.len()
                );
                for s in &report.skipped_existing {
                    println!("  - {s}");
                }
            }
            println!("next: jerith run (config.toml restored with this node's identity)");
        }
    }
    Ok(())
}

async fn chain_cmd(cmd: ChainCmd) -> anyhow::Result<()> {
    let cfg = jerith_mesh_core::config::NodeConfig::load()?.unwrap_or_default();
    match cmd {
        ChainCmd::Status => {
            let body = chain_get(Some(&cfg), &chain_base(Some(&cfg)), "/status", &[]).await?;
            let s = serde_json::to_string_pretty(&body)?;
            println!("{s}");
        }
        ChainCmd::Wallets { limit, by_balance } => {
            let db = chain_ledger_path(Some(&cfg))?;
            // Read-only open: a chain node writing WAL entries cannot be
            // corrupted by our reads.
            let uri = format!(
                "file:{}?mode=ro",
                db.display().to_string().replace('?', "%3F")
            );
            let conn = rusqlite::Connection::open_with_flags(
                &uri,
                rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_URI,
            )
            .with_context(|| format!("opening chain ledger {db:?}"))?;
            let rows = load_chain_wallets(&conn, limit, by_balance)?;
            println!(
                "{:<52} {:>14} {:>12} {:>12} {:>12} discord",
                "address", "balance JER", "locked", "nonce", "mined"
            );
            for r in rows {
                println!(
                    "{:<52} {:>14.6} {:>12.6} {:>12} {:>12.6} {}",
                    r.address,
                    r.balance_uj as f64 / 1e6,
                    r.locked_uj as f64 / 1e6,
                    r.nonce,
                    r.total_mined_uj as f64 / 1e6,
                    r.discord_id
                        .map(|d| d.to_string())
                        .unwrap_or_else(|| "—".to_string()),
                );
            }
        }
        ChainCmd::Balance { who } => {
            let is_discord_id = who.chars().all(|c| c.is_ascii_digit());
            let query = if is_discord_id {
                vec![("discord_id", who.as_str())]
            } else {
                vec![("address", who.as_str())]
            };
            let body = chain_get(Some(&cfg), &chain_base(Some(&cfg)), "/balance", &query).await?;
            let s = serde_json::to_string_pretty(&body)?;
            println!("{s}");
        }
        ChainCmd::Miners { limit } => {
            let body = chain_get(
                Some(&cfg),
                &chain_base(Some(&cfg)),
                "/mystate",
                &[("limit", limit.to_string().as_str())],
            )
            .await?;
            let s = serde_json::to_string_pretty(&body)?;
            println!("{s}");
        }
        ChainCmd::Block { height } => {
            let body = chain_get(
                Some(&cfg),
                &chain_base(Some(&cfg)),
                &format!("/block/{height}"),
                &[],
            )
            .await?;
            let s = serde_json::to_string_pretty(&body)?;
            println!("{s}");
        }
        ChainCmd::Tx { txid } => {
            let body = chain_get(
                Some(&cfg),
                &chain_base(Some(&cfg)),
                &format!("/tx/{txid}"),
                &[],
            )
            .await?;
            let s = serde_json::to_string_pretty(&body)?;
            println!("{s}");
        }
    }
    Ok(())
}

async fn memory_cmd(cmd: MemoryCmd) -> anyhow::Result<()> {
    use jerith_mesh_core::memory::MemoryHandle;
    let cfg = NodeConfig::load()?.context("no config — run setup first")?;
    // Same choose-and-open rules as the run path: configured backend,
    // honest failure (never a different store) when it is unusable.
    let store = MemoryHandle::open_for_config(&cfg).await?;
    match cmd {
        MemoryCmd::Add { text } => {
            let (id, provider) = store.embed_and_store(&text).await?;
            println!("stored #{id} (provider: {provider:?})");
        }
        MemoryCmd::Search { query, top_k } => {
            let hits = store.search(&query, top_k).await?;
            if hits.is_empty() {
                println!("no hits");
            }
            for h in hits {
                println!(
                    "#{:<4} {:>6.3}  {}",
                    h.id,
                    h.score,
                    h.text.chars().take(80).collect::<String>()
                );
            }
        }
        MemoryCmd::Count => {
            println!("{} entries", store.count().await?);
        }
        MemoryCmd::Forget { id } => {
            if store.forget(id).await? {
                println!("forgot #{id}");
            } else {
                println!("#{id} not found");
            }
        }
        MemoryCmd::Similar { id, top_k } => {
            let hits = store.similar_to(id, top_k).await?;
            if hits.is_empty() {
                println!("no other entries stored yet");
            }
            for h in hits {
                println!(
                    "#{:<4} {:>6.3}  {}",
                    h.id,
                    h.score,
                    h.text.chars().take(80).collect::<String>()
                );
            }
        }
        MemoryCmd::Stats => {
            let st = store.stats().await?;
            println!("entries: {}", st.count);
            for (dim, n) in &st.dims {
                println!("dim {dim}: {n}");
            }
            for (p, n) in &st.providers {
                println!("provider {p}: {n}");
            }
            println!("embedding bytes: {}", st.embedding_bytes);
            if let Some(ms) = st.newest_created_ms {
                println!("newest entry: {} ms epoch", ms);
            }
        }
        MemoryCmd::Migrate { drop_source } => {
            // Direction: from the store the config LEFT to the store it
            // now names. Postgres→sqlite reads the DSN from config (the
            // old postgres the node used); sqlite→postgres needs the DSN.
            let source = match cfg.memory_backend {
                MemoryBackend::PostgresPgvector => {
                    MemoryHandle::open(
                        MemoryBackend::Sqlite,
                        &NodeConfig::memory_db_path(),
                        None,
                        jerith_mesh_core::memory::EmbedConfig {
                            dim: cfg.specialists.embedding_dim as usize,
                            ..Default::default()
                        },
                        cfg.memory_max_rows,
                    )
                    .await?
                }
                MemoryBackend::Sqlite => {
                    let dsn = cfg.effective_memory_dsn().context(
                        "migrating FROM postgres needs the OLD memory_dsn in config.toml \
                         (or JERITH_MEMORY_DSN) — set it to the postgres the node used before",
                    )?;
                    MemoryHandle::open(
                        MemoryBackend::PostgresPgvector,
                        &NodeConfig::memory_db_path(),
                        Some(&dsn),
                        jerith_mesh_core::memory::EmbedConfig {
                            dim: cfg.specialists.embedding_dim as usize,
                            ..Default::default()
                        },
                        cfg.memory_max_rows,
                    )
                    .await?
                }
            };
            let src_count = source.count().await?;
            if src_count == 0 {
                println!(
                    "source store is empty — nothing to migrate\n\
                     (direction: FROM the store the config has LEFT; \
                     target = the now-configured store)"
                );
                return Ok(());
            }
            println!("migrating {} source rows → {}", src_count, store.describe());
            let report = store.copy_rows_from(&source).await.context(
                "migration copy — the target store keeps whatever rows landed, rerun is idempotent",
            )?;
            let target_count = store.count().await?;
            println!(
                "copied: {} · already present (skipped): {} · target now holds {} entries",
                report.copied, report.skipped_existing, target_count
            );
            if drop_source {
                let wiped = source.wipe().await?;
                println!("source emptied ({wiped} rows removed) — migration complete");
            } else {
                println!(
                    "source left intact — verify the target, then rerun with \
                     --drop-source to empty it"
                );
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// stake (Phase 5 escrow tier)
// ---------------------------------------------------------------------------

/// Amount → µJER with a positive guard.
fn jer_to_uj(amount_jer: f64) -> anyhow::Result<u64> {
    if amount_jer <= 0.0 {
        bail!("amount must be positive (JER)");
    }
    Ok((amount_jer * 1_000_000.0).round() as u64)
}

async fn stake_cmd(cmd: StakeCmd) -> anyhow::Result<()> {
    let cfg = NodeConfig::load()?.context("no config — run setup first")?;
    if cfg.crypto.chain != jerith_mesh_core::config::CryptoChain::Jerithchain {
        bail!(
            "staking rides jerithchain only — set crypto = \"jerithchain\" (and JERITH_TOKEN or keys/api.token) first"
        );
    }
    let identity = cfg.identity()?;
    let stake = jerith_mesh_core::stake::StakeManager::new(
        &cfg.crypto,
        Some(identity),
        // Same operator resolver as `chain_base(Some(&cfg))`: /stake/*
        // lives on the chain node's api (:8322 co-hosted), not the
        // miner-facing wallet endpoint.
        Some(chain_base(Some(&cfg))),
    );
    match cmd {
        StakeCmd::Status => {
            stake.refresh_from_chain().await;
            match stake.chain_view().await {
                Some(v) => {
                    let staked = v["staked_uj"].as_u64().unwrap_or(0);
                    println!(
                        "chain: staked {staked} µJ ({:.3} JER) | slashed {} µJ | escrow wallet {}",
                        staked as f64 / 1e6,
                        v["slashed_total_uj"].as_u64().unwrap_or(0),
                        v["escrow_address"].as_str().unwrap_or("?"),
                    );
                }
                None => println!(
                    "chain /stake view unavailable (chain down, wallet absent, or token) — showing local ledger"
                ),
            }
            let snap = stake.snapshot();
            println!(
                "local: staked {} µJ | escrowed {} µJ | slashed {} µJ | required {} µJ | gate_clear {}",
                snap.staked_uj,
                snap.escrowed_uj,
                snap.slashed_total_uj,
                stake.required_uj(),
                stake.clears_gate(),
            );
        }
        StakeCmd::Lock { amount_jer } => {
            stake.refresh_from_chain().await;
            let uj = match amount_jer {
                Some(a) => jer_to_uj(a)?,
                None => {
                    let required = stake.required_uj();
                    if required == 0 {
                        bail!(
                            "[crypto.stake] required_uj is 0 — enable the tier or pass an explicit amount"
                        );
                    }
                    let snap = stake.snapshot();
                    required.saturating_sub(snap.staked_uj.max(snap.escrowed_uj))
                }
            };
            if uj == 0 {
                println!("requirement already covered — nothing to lock");
                return Ok(());
            }
            println!("locking {uj} µJ ({:.3} JER)…", uj as f64 / 1e6);
            if stake.lock_topup(uj).await {
                println!(
                    "✓ escrow live: staked {} µJ | gate_clear {}",
                    stake.snapshot().staked_uj,
                    stake.clears_gate()
                );
            } else {
                bail!("lock failed — the chain's answer is in the log");
            }
        }
        StakeCmd::Unlock { amount_jer } => {
            stake.refresh_from_chain().await;
            let uj = match amount_jer {
                Some(a) => jer_to_uj(a)?,
                None => stake.snapshot().escrowed_uj,
            };
            if uj == 0 {
                println!("nothing marked as escrowed locally — nothing to return");
                return Ok(());
            }
            println!("unlocking {uj} µJ ({:.3} JER)…", uj as f64 / 1e6);
            if stake.unlock_amount(uj).await {
                println!(
                    "✓ unlocked; chain now reports staked {} µJ",
                    stake.snapshot().staked_uj
                );
            } else {
                bail!("unlock failed — the chain's answer is in the log");
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// setup
// ---------------------------------------------------------------------------

fn is_tty() -> bool {
    // Cheap heuristic: stdin is a pipe when run from another program.
    std::env::var("JERITH_NONINTERACTIVE").is_err() && check_tty()
}

fn check_tty() -> bool {
    #[cfg(unix)]
    {
        libc_isatty()
    }
    #[cfg(not(unix))]
    {
        true
    }
}

#[cfg(unix)]
fn libc_isatty() -> bool {
    // Direct libc isatty via std is unavailable; use a safe approximation
    // with /dev/tty openability.
    std::fs::File::open("/dev/tty").is_ok()
}

fn prompt(question: &str, default: &str) -> anyhow::Result<String> {
    println!("{question} [{default}]");
    print!("> ");
    std::io::stdout().flush()?;
    let mut line = String::new();
    std::io::stdin()
        .lock()
        .read_line(&mut line)
        .context("reading input")?;
    let answer = line.trim();
    Ok(if answer.is_empty() {
        default.to_string()
    } else {
        answer.to_string()
    })
}

fn parse_model_spec(spec: &str) -> anyhow::Result<ModelId> {
    if let Some(stripped) = spec.strip_prefix("huggingface:") {
        return Ok(ModelId {
            id: stripped.to_string(),
            quant: "int4".into(),
            source: ModelSource::HuggingFace {
                repo: stripped.to_string(),
            },
        });
    }
    if let Some(stripped) = spec.strip_prefix("ollama:") {
        return Ok(ModelId {
            id: stripped.to_string(),
            quant: "int4".into(),
            source: ModelSource::Ollama {
                name: stripped.to_string(),
            },
        });
    }
    if let Some(stripped) = spec.strip_prefix("path:") {
        return Ok(ModelId {
            id: stripped.to_string(),
            quant: "int4".into(),
            source: ModelSource::Path {
                path: stripped.to_string(),
            },
        });
    }
    // Curated id.
    let curated = jerith_mesh_core::models::curated();
    if curated.iter().any(|m| m.id == spec) {
        return Ok(default_model(spec));
    }
    // Small-model tier tags (setup's picker catalog; qwen3:8b etc.) are
    // ollama names served by the ollama/litellm runtime.
    if let Some(sm) = jerith_mesh_core::toolstack::find_small_model(spec) {
        return Ok(ModelId {
            id: sm.name.to_string(),
            quant: "int4".into(),
            source: ModelSource::Ollama {
                name: sm.name.to_string(),
            },
        });
    }
    bail!(
        "unknown model {spec:?}. Use a curated id (glm-5.2, glm-5.3, glm-5.3-flash, \
         deepseek-v4-flash, deepseek-v4.1-flash, kimi-k3, qwen3.8-flash-next, \
         qwen-image-2.1), a small-model tier tag (see `jerith stack list --models`), \
         or prefix: huggingface:<repo>, ollama:<name>, path:<file>"
    );
}

fn parse_memory(s: &str) -> anyhow::Result<MemoryBackend> {
    match s {
        "sqlite" => Ok(MemoryBackend::Sqlite),
        "postgres" => Ok(MemoryBackend::PostgresPgvector),
        other => bail!("unknown memory backend {other:?} (use sqlite or postgres)"),
    }
}

fn parse_role(s: &str) -> anyhow::Result<NodeRole> {
    match s {
        "contributor" => Ok(NodeRole::Contributor),
        "requester" => Ok(NodeRole::Requester),
        "hybrid" => Ok(NodeRole::Hybrid),
        other => bail!("unknown role {other:?} (use contributor, requester or hybrid)"),
    }
}

fn parse_crypto(s: &str) -> anyhow::Result<jerith_mesh_core::config::CryptoConfig> {
    use jerith_mesh_core::config::{CryptoChain, CryptoConfig};
    let chain = match s {
        "jerithchain" | "jer" => CryptoChain::Jerithchain,
        "bitcoin" | "btc" => CryptoChain::Bitcoin,
        "ethereum" | "eth" => CryptoChain::Ethereum,
        "dogecoin" | "doge" => CryptoChain::Dogecoin,
        "monero" | "xmr" => CryptoChain::Monero,
        "none" => CryptoChain::None,
        other => bail!("unknown chain {other:?}"),
    };
    Ok(CryptoConfig {
        chain,
        ..CryptoConfig::default()
    })
}

#[allow(clippy::too_many_arguments)]
fn setup_flow(a: SetupArgs) -> anyhow::Result<()> {
    let SetupArgs {
        model,
        memory,
        memory_dsn,
        contribution,
        role,
        crypto,
        bootstrap,
        no_wait,
        no_vision,
        no_embeddings,
        tool_model,
        deep_model,
        vision_model,
        embedding_model,
        runtime_url,
        with_stack,
        print_mode,
    } = a;
    println!("🦜 jerith-mesh setup");
    println!("───────────────────");

    // 1. Hardware scan (OS-specific tool + sysinfo fallback).
    let report = hw::probe();
    println!(
        "\n[1/5] Hardware: {} · {} · {} · {} · {} RAM{}\n",
        report.os_release,
        report.cpu_model,
        report.cpu_logical_cores,
        jerith_mesh_core::format_bytes(report.ram_total_bytes),
        match report.tool_used {
            Some(t) => t.name(),
            None => "sysinfo fallback",
        },
        if report.gpus.is_empty() {
            String::new()
        } else {
            format!(" · {} GPU(s)", report.gpus.len())
        }
    );

    // What can this machine actually run?
    let fits = detect_capable_models(&report);
    println!("\nModels this machine can run locally:");
    for fit in &fits {
        let marker = if fit.fits { "✓" } else { "✗" };
        println!("  {marker} {:<22} {}", fit.model.id, fit.reason);
    }
    if let Some(rec) = recommended_model(&report) {
        println!("\n  Recommended: {}", rec.id);
    }

    let interactive = is_tty();

    // 2. Model choice — optionally with live peer counts.
    let model_spec = if let Some(m) = model {
        m
    } else if interactive {
        let mut bootstrap_counts: Option<Vec<(String, u64)>> = None;
        if !no_wait {
            let url = bootstrap.clone().unwrap_or_default();
            let client = BootstrapClient::new(if url.is_empty() {
                NodeConfig::default().bootstrap_url.clone()
            } else {
                url
            });
            match futures_block_on(client.model_counts()) {
                Ok(counts) if !counts.is_empty() => {
                    println!("\nLive meshes (nodes currently sharing):");
                    for (id, count) in &counts {
                        println!("  {:<24} {} node(s)", id, count);
                    }
                    bootstrap_counts = Some(counts);
                }
                Ok(_) => {
                    println!("\n(bootstrap reachable but no live meshes yet)");
                }
                Err(_) => {
                    println!("\n(bootstrap unreachable — showing offline list)");
                }
            }
        }
        // Recommended meshes for your hardware (Phase 1): the capable-model
        // list joined with live peer counts — a small machine that can join
        // a live mesh beats a bigger one that would sit alone.
        let default_pick = {
            let sug = jerith_mesh_core::suggest::recommended_meshes(
                &fits,
                bootstrap_counts.as_deref().unwrap_or(&[]),
            );
            let pick = sug.first().map(|s| s.model_id.clone());
            if let Some(s) = sug.first() {
                println!(
                    "\n  Recommended mesh: {} ({} peer{} online) — {}",
                    s.model_id,
                    s.peers_online,
                    if s.peers_online == 1 { "" } else { "s" },
                    s.rationale
                );
            }
            pick.unwrap_or_else(|| "recommendation".to_string())
        };
        prompt(
            "\nModel to run? (id, or huggingface:<repo> / ollama:<name> / path:<file>)",
            &default_pick,
        )?
    } else {
        bail!("--model is required in non-interactive mode")
    };

    let model = if model_spec == "recommendation" {
        recommended_model(&report).context(
            "no model fits this machine — at least 8 GiB RAM needed for the smallest curated model",
        )?;
        let rec = recommended_model(&report).unwrap();
        default_model(&rec.id)
    } else {
        let mid = parse_model_spec(&model_spec)?;
        // Leech guard: warn if the machine can't realistically run it.
        if let Some(fit) = fits.iter().find(|f| f.model.id == mid.id)
            && !fit.fits
            && std::env::var("JERITH_OK_BEYOND_CAPACITY").is_err()
        {
            bail!(
                "this machine cannot run {id} locally: {reason}\n\
                     Passing capacity checks is required to prevent mesh leeching. \
                     Set JERITH_OK_BEYOND_CAPACITY=1 to override.",
                id = mid.id,
                reason = fit.reason
            );
        }
        mid
    };
    println!("\n[2/5] Model: {} (quant {})", model.id, model.quant);

    // 3. Memory backend.
    let memory_spec = match memory {
        Some(m) => m,
        None if interactive => prompt("\nMemory backend? (sqlite | postgres)", "sqlite")?,
        None => bail!("--memory is required in non-interactive mode"),
    };
    let memory_backend = parse_memory(&memory_spec)?;
    // A postgres backend is useless (and refused at boot) without the
    // connection string, so capture it here — flag, prompt, or env.
    let memory_dsn = match memory_backend {
        MemoryBackend::PostgresPgvector => {
            match memory_dsn {
                Some(d) if !d.trim().is_empty() => Some(d),
                None if interactive => {
                    let d = prompt(
                        "\nPostgres DSN for memory? (postgres://user:pass@host:5432/db, sslmode=disable)",
                        "postgres://postgres@127.0.0.1:5432/jerith?sslmode=disable",
                    )?;
                    Some(d)
                }
                // Flag missing (or blank): the deployment env may supply it.
                _ => std::env::var("JERITH_MEMORY_DSN")
                    .ok()
                    .filter(|d| !d.trim().is_empty()),
            }
        }
        MemoryBackend::Sqlite => None,
    };
    if memory_backend == MemoryBackend::PostgresPgvector && memory_dsn.is_none() {
        bail!(
            "--memory postgres needs --memory-dsn (or JERITH_MEMORY_DSN) — the DSN is stored \n         in config.toml's memory_dsn"
        );
    }
    println!("[3/5] Memory: {} (strictly local)", memory_backend.label());

    // 3b. Local specialists — the config defaults are both-on, so the
    // prompts mirror that and the flags override without a question.
    let on_off = |b: bool| if b { "on" } else { "off" };
    let vision_enabled = if no_vision {
        false
    } else if interactive {
        matches!(
            prompt("\nEnable the local vision specialist? (y/n)", "y")?
                .to_lowercase()
                .as_str(),
            "y" | "yes"
        )
    } else {
        true
    };
    let embeddings_enabled = if no_embeddings {
        false
    } else if interactive {
        matches!(
            prompt("\nEnable the local embedding specialist? (y/n)", "y")?
                .to_lowercase()
                .as_str(),
            "y" | "yes"
        )
    } else {
        true
    };
    // Tier model picks — the small-model catalog (toolstack). Thinking
    // routes complex prompts (runtime_deep_model), vision routes image
    // content (runtime_vision_model), embedding feeds local memory
    // (specialists.embedding_model). Curated tags or free-form
    // ollama:<name>; an empty pick ("") disables a tier. Flags win over
    // the prompt; defaults are the mesh's own routing defaults.
    use jerith_mesh_core::toolstack::{self, ModelTier, StackTool};
    let pick_tier = |tier: ModelTier, flag: &Option<String>| -> anyhow::Result<String> {
        if let Some(p) = flag {
            return validate_tier_pick(tier, p);
        }
        if !interactive {
            return Ok(toolstack::default_small_model(tier).name.to_string());
        }
        println!(
            "\n{} tier options (number, ollama tag, ollama:<name>, or \"\" to disable):",
            tier.label()
        );
        for (i, m) in toolstack::small_models(tier).iter().enumerate() {
            println!("  {}. {:<26} {:<10} {}", i + 1, m.name, m.size_hint, m.note);
        }
        let def = toolstack::default_small_model(tier).name;
        let answer = prompt(&format!("\nModel for the {} tier?", tier.label()), def)?;
        let pick = match answer.trim().parse::<usize>() {
            Ok(n) if (1..=toolstack::small_models(tier).len()).contains(&n) => {
                toolstack::small_models(tier)[n - 1].name.to_string()
            }
            _ => answer,
        };
        validate_tier_pick(tier, &pick)
    };
    let deep_pick = pick_tier(ModelTier::Thinking, &deep_model)?;
    // The tool tier is chosen for parse reliability, not depth — agent
    // tool loops reroute here ahead of the thinking rules.
    let tool_pick = pick_tier(ModelTier::Tool, &tool_model)?;
    // Vision tier is only kept when the vision specialist is enabled.
    let vision_pick = if vision_enabled {
        pick_tier(ModelTier::Vision, &vision_model)?
    } else {
        String::new()
    };
    // The embedding tag only matters when memory is actually on.
    let embedding_pick = if embeddings_enabled {
        pick_tier(ModelTier::Embedding, &embedding_model)?
    } else {
        String::new()
    };
    println!(
        "[3/5b] Local skills: vision {}, embeddings {} | tiers: tools {}, thinking {}, vision {}, embedding {} (excluded from the contribution accounting)",
        on_off(vision_enabled),
        on_off(embeddings_enabled),
        if tool_pick.is_empty() {
            "(off)"
        } else {
            tool_pick.as_str()
        },
        if deep_pick.is_empty() {
            "(off)"
        } else {
            deep_pick.as_str()
        },
        if vision_pick.is_empty() {
            "(off)"
        } else {
            vision_pick.as_str()
        },
        if embedding_pick.is_empty() {
            "(off)"
        } else {
            embedding_pick.as_str()
        }
    );

    // 4. Contribution % — min 25 hard-coded by spec.
    let contribution_pct = match contribution {
        Some(c) => c,
        None if interactive => {
            let s = prompt(
                "\nContribution percent? (minimum 25 to be eligible for the mesh)",
                "25",
            )?;
            s.parse().context("contribution must be a number")?
        }
        None => bail!("--contribution is required in non-interactive mode"),
    };
    if contribution_pct < MIN_CONTRIBUTION_PERCENT {
        bail!(
            "contribution {contribution_pct}% is below the minimum {MIN_CONTRIBUTION_PERCENT}% required to connect"
        );
    }
    println!("[4/5] Contribution: {contribution_pct}%");

    // 5. Role + crypto (optional).
    let role = match role {
        Some(r) => parse_role(&r)?,
        None if interactive => NodeRole::Hybrid,
        None => bail!("--role is required in non-interactive mode"),
    };
    let crypto = match crypto {
        Some(c) => parse_crypto(&c)?,
        None if interactive => {
            let s = prompt(
                "\nCrypto for earnings? (jerithchain | bitcoin | ethereum | dogecoin | monero | none)",
                "jerithchain",
            )?;
            parse_crypto(&s)?
        }
        None => jerith_mesh_core::config::CryptoConfig::default(),
    };

    // Runtime backstop: which OpenAI-compatible base URL serves the chat
    // lane. Ollama, once installed (stack tool), serves /v1/… itself — so
    // it becomes the default; otherwise the curated engine on :8000 (or a
    // LiteLLM proxy the operator names). The gateway appends /v1/… to
    // whatever lands here.
    let runtime_url = match runtime_url {
        Some(u) => Some(u),
        None if interactive => {
            let default = if detect_tool(StackTool::Ollama) {
                "http://127.0.0.1:11434 (ollama)"
            } else {
                "http://127.0.0.1:8000 (colibri/litellm)"
            };
            Some(prompt(
                "\nOpenAI-compatible runtime base URL? (scheme://host:port)",
                default,
            )?)
        }
        None => None,
    };
    // Accept a bare URL or an annotated answer like "url (ollama)".
    let runtime_url = runtime_url.map(|s| {
        s.split_whitespace()
            .next()
            .unwrap_or("http://127.0.0.1:8000")
            .to_string()
    });

    // Stack tools (--with all | tool,tool / --no-tools; interactive prompt
    // otherwise). Ollama is the host for the tier models; litellm + the
    // langchain family install into the node-managed tools venv; postgres
    // sets up the pgvector memory backend. Detection runs first so
    // already-present tools install as a harmless no-op.
    let with: Vec<StackTool> = if with_stack == Some("all".into()) {
        StackTool::ALL.to_vec()
    } else if with_stack.is_some() {
        let mut out = Vec::new();
        for n in with_stack.unwrap_or_default().split(',') {
            let t = StackTool::from_name(n)
                .with_context(|| format!("unknown stack tool {n:?} — see `jerith stack list`"))?;
            out.push(t);
        }
        out.sort();
        out.dedup();
        out
    } else if interactive {
        println!(
            "\nOptional AI tool stack (space/comma-separated numbers or names, empty = none):"
        );
        println!("  1. ollama     — hosts the tier models above locally");
        println!("  2. litellm    — OpenAI-compatible proxy over any backend");
        println!("  3. langchain  — agent framework using this node's gateway");
        println!("  4. langgraph  — LangChain's graph agent runtime");
        println!("  5. langsmith  — tracing + evals for LangChain/LangGraph");
        println!("  6. postgres   — PostgreSQL + pgvector (memory backend)");
        let answer = prompt("\nInstall which tools?", "")?;
        let mut out = Vec::new();
        for tok_str in answer.split([',', ' ']) {
            let tok_str = tok_str.trim();
            if tok_str.is_empty() {
                continue;
            }
            let tool = match tok_str.parse::<usize>() {
                Ok(n) if (1..=StackTool::ALL.len()).contains(&n) => StackTool::ALL[n - 1],
                _ => StackTool::from_name(tok_str).with_context(|| {
                    format!("unknown stack tool {tok_str:?} — see `jerith stack list`")
                })?,
            };
            out.push(tool);
        }
        out.sort();
        out.dedup();
        out
    } else {
        Vec::new()
    };

    let existing = NodeConfig::load()?;
    let mut cfg = SetupChoices {
        model,
        memory_backend,
        memory_dsn,
        contribution_percent: contribution_pct,
        role,
        crypto,
        specialists: jerith_mesh_core::config::SpecialistConfig {
            vision_enabled,
            embeddings_enabled,
            embedding_model: embedding_pick.clone(),
            ..Default::default()
        },
        bootstrap_url: bootstrap,
        gateway_port: NodeConfig::default().gateway_port,
        runtime_url,
    }
    .into_config(existing)?;
    // Tier picks land on the routing fields setup doesn't model — an
    // empty pick disables the tier (the gateway honors "" as "off").
    cfg.runtime_deep_model = if deep_pick.is_empty() {
        None
    } else {
        Some(deep_pick.clone())
    };
    cfg.runtime_vision_model = if vision_pick.is_empty() {
        None
    } else {
        Some(vision_pick.clone())
    };
    cfg.runtime_tool_model = if tool_pick.is_empty() {
        None
    } else {
        Some(tool_pick.clone())
    };
    if print_mode {
        println!(
            "   print-mode: config untouched — would write {} with the tier picks above",
            NodeConfig::config_path().display()
        );
    } else {
        cfg.save()?;
        // Touch the memory DB path so data dir exists.
        std::fs::create_dir_all(NodeConfig::data_dir())?;
    }

    // Materialize the chosen stack. --print-mode shows the exact plan and
    // touches nothing; otherwise every step is echoed before it runs
    // (running the command IS the consent) and failures stop setup with
    // the step's own error output.
    if !with.is_empty()
        || !deep_pick.is_empty()
        || !vision_pick.is_empty()
        || !embedding_pick.is_empty()
    {
        println!("\n[5/5] AI stack:");
    }
    let os = report.os;
    if !with.is_empty() {
        // Shared pip tools venv first — one plan, before any pip tool.
        if with.iter().any(|t| t.uses_tools_venv()) {
            for step in StackTool::venv_plan(os) {
                run_or_print(&step, print_mode, "tools-venv")?;
            }
        }
        for tool in &with {
            if detect_tool(*tool) {
                println!("   {:<10} already installed — skipping", tool.name());
                continue;
            }
            for step in tool.install_plan(os) {
                run_or_print(&step, print_mode, tool.name())?;
            }
        }
    }
    // Pull the tier models the operator just picked — only when ollama is
    // (already or just now) present and the pick is an ollama tag.
    let ollama_ready = detect_tool(StackTool::Ollama);
    if print_mode && !ollama_ready {
        for pick in [&tool_pick, &deep_pick, &vision_pick, &embedding_pick] {
            if !pick.is_empty() {
                println!("   $ ollama pull {pick}");
            }
        }
    } else if ollama_ready {
        for (tier, pick) in [
            (ModelTier::Tool, &tool_pick),
            (ModelTier::Thinking, &deep_pick),
            (ModelTier::Vision, &vision_pick),
            (ModelTier::Embedding, &embedding_pick),
        ] {
            if pick.is_empty() {
                continue;
            }
            if print_mode {
                println!("   $ ollama pull {pick}");
            } else {
                println!("   ollama     pulling {} ({})…", pick, tier.label());
                let pull = format!("ollama pull {pick}");
                run_or_print(&pull, false, "ollama")?;
            }
        }
    } else if !with.is_empty() {
        println!(
            "   note: ollama not available — pull the tier picks with `jerith stack pull <model>` after installing it"
        );
    }

    println!("\nDone. Config: {}", NodeConfig::config_path().display());
    println!("\nNext: ./jerith run   (or set up wallets with `jerith config`)");

    Ok(())
}

/// Validate a setup-flow tier pick: curated small-model tag, free-form
/// `ollama:<name>`, or `""` (disable the tier).
fn validate_tier_pick(
    tier: jerith_mesh_core::toolstack::ModelTier,
    pick: &str,
) -> anyhow::Result<String> {
    let pick = pick.trim();
    if pick.is_empty() {
        return Ok(String::new());
    }
    if let Some(name) = pick.strip_prefix("ollama:") {
        anyhow::ensure!(!name.trim().is_empty(), "empty ollama tag");
        return Ok(name.trim().to_string());
    }
    if let Some(m) = jerith_mesh_core::toolstack::find_small_model(pick) {
        if m.tier != tier {
            anyhow::bail!(
                "{} belongs to the {} tier, not {}",
                m.name,
                m.tier.label(),
                tier.label()
            );
        }
        return Ok(m.name.to_string());
    }
    anyhow::bail!(
        "unknown model {pick:?} for the {} tier — use a listed tag, ollama:<name>, or \"\" to disable",
        tier.label()
    )
}

/// Fast detect used inside setup (one version probe; failure = absent).
fn detect_tool(tool: jerith_mesh_core::toolstack::StackTool) -> bool {
    let (cmd, args) = tool.detect_cmd();
    std::process::Command::new(cmd)
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Echo then run one install step (bash on unix, cmd on Windows); the
/// child inherits stdout/stderr so installers stay interactive. Failure
/// aborts setup immediately with the step's own output as the diagnosis.
fn run_or_print(step: &str, print_mode: bool, label: &str) -> anyhow::Result<()> {
    if print_mode {
        println!("   $ {step}");
        return Ok(());
    }
    println!("   [{label}] $ {step}");
    let (program, args) = if cfg!(windows) {
        ("cmd", vec!["/C", step])
    } else {
        ("sh", vec!["-c", step])
    };
    let status = std::process::Command::new(program)
        .args(&args)
        .status()
        .with_context(|| format!("spawning {program} for [{label}]"))?;
    anyhow::ensure!(status.success(), "[{label}] step failed ({status}): {step}");
    Ok(())
}

// ---------------------------------------------------------------------------
// plugin bundle channel (docs/PLUGIN-CHANNEL.md Phase 2) — the human
// surface over core::pluginbundle: verify chain → (extract | stage).
// ---------------------------------------------------------------------------

/// Publisher pins file: `$JERITH_HOME|~/.jerith-mesh/keys/plugin-pins.json`
/// mapping plugin id →.ed25519 pubkey hex. LEGACY source (Phase 3's
/// `[plugins.trusted_publishers]` bot.toml table below is the primary
/// one now) — an absent file resolves as just "no legacy rows"; a
/// PRESENT but malformed file stays a loud failure (trust content is
/// load-bearing).
fn load_pins() -> anyhow::Result<std::collections::BTreeMap<String, String>> {
    let path = jerith_mesh_core::config::NodeConfig::home()
        .join("keys")
        .join("plugin-pins.json");
    if !path.is_file() {
        return Ok(std::collections::BTreeMap::new());
    }
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("reading publisher pins file {}", path.display()))?;
    serde_json::from_str(&raw)
        .with_context(|| format!("pins file {} is not valid JSON", path.display()))
}

/// Resolve the expected publisher for `id`: an explicit --pubkey, the
/// bot.toml trust row (primary), the legacy pins-file entry, or a
/// refusal that names the presented key (the human pastes it ONLY
/// after trusting the source). Returns the pin AND where it came from
/// — the source label rides the receipt log (audit trail).
fn resolve_pin(
    id: &str,
    pubkey: Option<&str>,
    presented: &str,
) -> anyhow::Result<(String, &'static str)> {
    if let Some(p) = pubkey {
        return Ok((p.trim().to_ascii_lowercase(), "flag"));
    }
    if let Some(p) = load_trusted_publishers()?.get(id) {
        return Ok((p.clone(), "bot.toml"));
    }
    if let Some(p) = load_pins()?.get(id) {
        return Ok((p.trim().to_ascii_lowercase(), "pins-file"));
    }
    let shown = presented.get(..8).unwrap_or(presented);
    anyhow::bail!(
        "no pin for plugin {id:?} — the bundle presents publisher key {shown}…\n\
         after trusting this source, paste into [plugins.trusted_publishers] in {}:\n\
         \n\
             \"{id}\" = \"{presented}\"\n\
         \n\
         (the legacy keys/plugin-pins.json file would also resolve — but bot.toml is \
         where trust lives now)",
        bot_toml_path().display()
    );
}

/// Does `[plugins].enabled` in bot.toml already gate `external:<id>`?
/// The install "next 1" step is wrong advice when the row exists (an
/// upgrade reinstall doesn't need it) — the two cases deserve different
/// guidance. `Err`-silent: after staging, a next-step check must never
/// fail the install (bot.toml was already parsed for the pin above).
fn bot_toml_gates(id: &str) -> anyhow::Result<bool> {
    let path = bot_toml_path();
    if !path.is_file() {
        return Ok(false);
    }
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    parse_enabled_gate(&text, id)
}

/// Is the conventional plugin unit (`jerith-<id>-plugin`) actually
/// active right now? `systemctl --user is-active` answers 0 only for
/// active — unknown units, inactive ones and a missing systemctl (mac
/// boxes) all read "not running", so the remove hint says so instead of
/// commanding a stop that was already done.
fn plugin_unit_active(id: &str) -> bool {
    let unit = format!("jerith-{id}-plugin");
    std::process::Command::new("systemctl")
        .args(["--user", "is-active", unit.as_str()])
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false)
}

fn parse_enabled_gate(text: &str, id: &str) -> anyhow::Result<bool> {
    use toml_edit::DocumentMut;
    let gate = format!("external:{id}");
    let doc: DocumentMut = text
        .parse()
        .map_err(|e| anyhow::anyhow!("not a TOML document ({e})"))?;
    let Some(enabled) = doc.get("plugins").and_then(|p| p.get("enabled")) else {
        return Ok(false);
    };
    Ok(enabled
        .as_array()
        .map(|a| a.iter().any(|v| v.as_str() == Some(gate.as_str())))
        .unwrap_or(false))
}

/// The operator bot.toml path — the trust table's home (docs/
/// PLUGIN-CHANNEL.md key model). Same file `[plugins] enabled` lives
/// in, so the pin list and the activation gate sit in one operator
/// surface.
fn bot_toml_path() -> PathBuf {
    jerith_mesh_core::config::NodeConfig::home().join("bot.toml")
}

/// A trust row's key: exactly 64 hex chars.
fn valid_pubkey_hex(s: &str) -> bool {
    s.len() == 64 && s.chars().all(|c| c.is_ascii_hexdigit())
}

fn normalize_key(s: &str) -> String {
    s.trim().to_ascii_lowercase()
}

/// `[plugins.trusted_publishers]` in bot.toml: the per-id pubkey pins
/// (Phase 3 of docs/PLUGIN-CHANNEL.md). Only `jerith` reads this — the
/// bot itself never fetches bundles — but the table IS the operator's
/// trust root, so a malformed row fails loud instead of being silently
/// skipped (a skipped row would silently downgrade to a refusal later,
/// and a trust table that half-works is worse than one that refuses).
/// No file / no section = an empty map: first-party installs resolve
/// through the committed anchor instead of a pin row.
fn load_trusted_publishers() -> anyhow::Result<std::collections::BTreeMap<String, String>> {
    let path = bot_toml_path();
    if !path.is_file() {
        return Ok(std::collections::BTreeMap::new());
    }
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    parse_trusted_publishers(&text)
        .with_context(|| format!("reading trust rows from {}", path.display()))
}

fn trust_pins_file_header(
    bot: &std::collections::BTreeMap<String, String>,
    pins: &std::collections::BTreeMap<String, String>,
) -> &'static str {
    if pins.is_empty() {
        // Empty/absent file: no legacy rows exist to migrate at all —
        // advertising rotation-migration here is already-done noise.
        "empty — no legacy rows to migrate; rotate pins land only in \
         bot.toml"
    } else if pins
        .iter()
        .all(|(id, key)| bot.get(id).is_some_and(|k| k.eq_ignore_ascii_case(key)))
    {
        "legacy file — every row mirrored in bot.toml, nothing left for a \
         rotate to migrate"
    } else {
        "legacy — a bot.toml row OVERRANKS it; a rotate migrates"
    }
}

/// The trust-table parse, file-text → id → 64-hex-lowercase map.
/// Test-separated from disk (and from the node's other bot.toml reads:
/// the bot's own loader is a different reader — this one only exists
/// inside `jerith`, and the bot never reads the table).
fn parse_trusted_publishers(
    text: &str,
) -> anyhow::Result<std::collections::BTreeMap<String, String>> {
    use toml_edit::{DocumentMut, Item, Value};
    let mut out = std::collections::BTreeMap::new();
    let doc: DocumentMut = text
        .parse()
        .map_err(|e| anyhow::anyhow!("not a TOML document ({e})"))?;
    let Some(tp) = doc.get("plugins").and_then(|p| p.get("trusted_publishers")) else {
        return Ok(out);
    };
    match tp {
        Item::Table(t) => {
            for (k, v) in t.iter() {
                let val = v.as_value().ok_or_else(|| {
                    anyhow::anyhow!("trusted_publishers[{k:?}] is not a pubkey string")
                })?;
                insert_trust_row(&mut out, k, val)?;
            }
        }
        Item::Value(Value::InlineTable(it)) => {
            for (k, v) in it.iter() {
                insert_trust_row(&mut out, k, v)?;
            }
        }
        _ => anyhow::bail!("[plugins.trusted_publishers] is not a table of id → pubkey strings"),
    }
    Ok(out)
}

/// One trust row: a plain-string Ed25519 pubkey, lowercased.
fn insert_trust_row(
    out: &mut std::collections::BTreeMap<String, String>,
    id: &str,
    v: &toml_edit::Value,
) -> anyhow::Result<()> {
    let toml_edit::Value::String(s) = v else {
        anyhow::bail!("trusted_publishers[{id:?}] is not a pubkey string");
    };
    let key = normalize_key(s.value());
    anyhow::ensure!(
        valid_pubkey_hex(&key),
        "trusted_publishers[{id:?}]: {:?} is not a 64-hex Ed25519 pubkey — \
         a trust row must be load-bearing or absent",
        s.value()
    );
    out.insert(id.to_string(), key);
    Ok(())
}

/// Rotate (or first-pin) `<id>` in bot.toml's
/// `[plugins.trusted_publishers]` WITHOUT disturbing anything else in
/// the file — comments, ordering and unrelated tables survive (that is
/// why this goes through toml_edit and not a string replace).
fn write_trusted_publishers(path: &Path, id: &str, key_hex: &str) -> anyhow::Result<PathBuf> {
    use toml_edit::{DocumentMut, InlineTable, Item, Table, Value, value};
    anyhow::ensure!(
        valid_pubkey_hex(key_hex),
        "{key_hex:?} is not a 64-hex Ed25519 pubkey"
    );
    let mut doc: DocumentMut = if path.is_file() {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        text.parse::<DocumentMut>()
            .map_err(|e| anyhow::anyhow!("{} is not a TOML document ({e})", path.display()))?
    } else {
        // A fresh trust file: [plugins.trusted_publishers] with the one
        // row, nothing else.
        DocumentMut::new()
    };
    if !doc.contains_key("plugins") {
        doc["plugins"] = Item::Table(Table::new());
    }
    let plugins = &mut doc["plugins"];
    if let Some(inline) = plugins.as_inline_table_mut() {
        // The operator wrote `[plugins]` as an inline table; the
        // file's own style wins — the pin lives in a nested inline.
        let sub = match inline.get_mut("trusted_publishers") {
            Some(v) => v.as_inline_table_mut().ok_or_else(|| {
                anyhow::anyhow!(
                    "[plugins] trusted_publishers is not a table of id → pubkey strings"
                )
            })?,
            None => {
                inline.insert("trusted_publishers", Value::InlineTable(InlineTable::new()));
                inline
                    .get_mut("trusted_publishers")
                    .and_then(|v| v.as_inline_table_mut())
                    .expect("just inserted")
            }
        };
        sub.insert(id, Value::from(key_hex));
    } else {
        let table = plugins.as_table_mut().ok_or_else(|| {
            anyhow::anyhow!(
                "[plugins] is neither a table nor an inline table in {}",
                path.display()
            )
        })?;
        match table
            .entry("trusted_publishers")
            .or_insert(Item::Table(Table::new()))
        {
            Item::Table(t) => {
                t[id] = value(key_hex);
            }
            Item::Value(Value::InlineTable(it)) => {
                it.insert(id, Value::from(key_hex));
            }
            _ => {
                anyhow::bail!("[plugins.trusted_publishers] is not a table of id → pubkey strings")
            }
        }
    }
    std::fs::write(path, doc.to_string()).with_context(|| format!("writing {}", path.display()))?;
    Ok(path.to_path_buf())
}

/// Fetch the bundle + both companions: a local path needs `.sha256`/`.sig`
/// files beside it; an https URL downloads all three (plain http is
/// refused; loopback plain http is a non-issue since this is a fetch of a
/// VERIFIED artifact, not signed-request carriage) into the cache dir.
async fn fetch_bundle(source: &str) -> anyhow::Result<(PathBuf, String, String)> {
    use jerith_mesh_core::pluginbundle::parse_sha256_file;
    if source.starts_with("http://") || source.starts_with("https://") {
        anyhow::ensure!(
            !source.starts_with("http://"),
            "plugin install: plain http is refused — fetch over https"
        );
        let client = jerith_mesh_core::http::mesh_http();
        let get = |url: String, cap: u64| {
            let client = client.clone();
            async move {
                let resp = client
                    .get(&url)
                    .timeout(Duration::from_secs(120))
                    .send()
                    .await
                    .with_context(|| format!("fetching {url}"))?;
                let resp = resp
                    .error_for_status()
                    .with_context(|| format!("fetching {url}"))?;
                let len = resp.content_length().unwrap_or_default();
                anyhow::ensure!(len <= cap, "{url}: {len} bytes — over the {cap}-byte cap");
                let bytes = resp
                    .bytes()
                    .await
                    .with_context(|| format!("reading {url}"))?;
                anyhow::ensure!(
                    bytes.len() as u64 <= cap,
                    "{url}: {} bytes — over the {cap}-byte cap",
                    bytes.len()
                );
                Ok::<_, anyhow::Error>(bytes)
            }
        };
        let cap = 256 * 1024 * 1024;
        let raw = get(source.to_string(), cap).await?;
        let sha_file = get(format!("{source}.sha256"), 4096).await?;
        let sig_file = get(format!("{source}.sig"), 4096).await?;
        let sha = parse_sha256_file(std::str::from_utf8(&sha_file)?)?;
        let sig = std::str::from_utf8(&sig_file)?.trim().to_string();
        let (id, version, _publisher) = jerith_mesh_core::pluginbundle::peek_identity(&raw)
            .context("downloaded bundle unreadable — cannot give it a cache name")?;
        let cache = jerith_mesh_core::config::NodeConfig::home()
            .join("plugins")
            .join("cache");
        std::fs::create_dir_all(&cache)?;
        let stem = format!(
            "{id}-{version}-{}-{}",
            std::env::consts::OS,
            std::env::consts::ARCH
        );
        let dest = cache.join(format!("{stem}.jbundle"));
        std::fs::write(&dest, &raw)?;
        std::fs::write(
            format!("{}.sha256", dest.display()),
            format!("{sha}  {stem}.jbundle\n"),
        )?;
        std::fs::write(format!("{}.sig", dest.display()), format!("{sig}\n"))?;
        println!("fetched → {}", dest.display());
        Ok((dest, sha, sig))
    } else {
        // Local path: companions must ride beside the bundle.
        let path = PathBuf::from(source);
        anyhow::ensure!(path.is_file(), "bundle path {} missing", path.display());
        let raw_sha = std::fs::read_to_string(format!("{}.sha256", path.display()))
            .with_context(|| format!("missing companion {}.sha256", path.display()))?;
        let sig = std::fs::read_to_string(format!("{}.sig", path.display()))
            .with_context(|| format!("missing companion {}.sig", path.display()))?
            .trim()
            .to_string();
        Ok((path, parse_sha256_file(&raw_sha)?, sig))
    }
}
/// One verify chain print (shared by install + verify).
fn run_verify(
    raw: &[u8],
    sha: &str,
    sig: &str,
    pin: &str,
) -> anyhow::Result<jerith_mesh_core::pluginbundle::VerifiedBundle> {
    let v = jerith_mesh_core::pluginbundle::verify_bytes(
        raw,
        &jerith_mesh_core::pluginbundle::VerifyOptions {
            expected_sha256: Some(sha),
            sig_hex: sig,
            pinned_pubkey: Some(pin),
            platform_os: std::env::consts::OS,
            platform_arch: std::env::consts::ARCH,
            node_contract_version: Some(env!("CARGO_PKG_VERSION")),
        },
    )?;
    let m = &v.meta;
    println!(
        "✓ verified: {} {} — publisher {}… · {}/{} · binary {} ({} bytes)",
        m.id,
        m.version,
        publisher_thumb(&m.publisher),
        m.os,
        m.arch,
        v.binary_name,
        v.binary.len()
    );
    Ok(v)
}

/// The publisher signing seed: explicit `--key`, $JERITH_BUNDLE_KEY, or
/// the repo-relative `keys/plugin-releases.hex` (0600, gitignored — the
/// first-party format; docs/PLUGIN-CHANNEL.md key model).
fn load_signing_seed(
    key: Option<&Path>,
) -> anyhow::Result<jerith_mesh_core::signing::SigningIdentity> {
    let key = key
        .map(Path::to_path_buf)
        .or_else(|| std::env::var("JERITH_BUNDLE_KEY").ok().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("keys/plugin-releases.hex"));
    let raw = std::fs::read_to_string(&key).with_context(|| {
        format!(
            "signing key {}: missing (generate the pair once; keep the hex file 0600 + gitignored)",
            key.display()
        )
    })?;
    jerith_mesh_core::signing::SigningIdentity::from_secret_hex(raw.trim())
        .with_context(|| format!("signing key {}: not a 32-byte hex seed", key.display()))
}

/// First 8 hex of a publisher key + last 4 — a thumbprint to display.
fn publisher_thumb(hex: &str) -> String {
    let n = hex.len();
    let head = hex.get(..8).unwrap_or(hex);
    let tail = n.checked_sub(4).and_then(|i| hex.get(i..)).unwrap_or("");
    format!("{head}…{tail}")
}

async fn plugin_cmd(cmd: PluginCmd) -> anyhow::Result<()> {
    use jerith_mesh_core::pluginbundle as pb;
    use jerith_mesh_core::pluginchannel as pc;
    use jerith_mesh_core::pluginreceipt as jr;
    use std::cmp::Ordering;
    match cmd {
        PluginCmd::Verify { path, pubkey } => {
            let (path, sha, sig) = fetch_bundle(&path.display().to_string()).await?;
            let raw = std::fs::read(&path)?;
            let (id, _version, presented) = pb::peek_identity(&raw)
                .context("cannot resolve a publisher pin for this bundle")?;
            let (pin, _pin_source) = resolve_pin(&id, pubkey.as_deref(), &presented)?;
            run_verify(&raw, &sha, &sig, &pin)?;
            // NOT receipted: verify's contract is "without touching disk"
            // (the picking-apart tool), and its failures already name
            // everything an install receipt would say.
            Ok(())
        }
        PluginCmd::Install {
            source,
            force,
            pubkey,
            channel,
            max_age_ms,
            receipt_allow_broken,
        } => {
            // GATE first (docs/PLUGIN-CHANNEL.md): trust loading refuses
            // to run while the receipt log's own witness questions it.
            receipt_gate_under(&NodeConfig::home(), receipt_allow_broken)?;
            let (bundle_path, sha, sig) = fetch_bundle(&source).await?;
            let raw = std::fs::read(&bundle_path)?;
            let (id, _version, presented) = pb::peek_identity(&raw)
                .context("cannot resolve a publisher pin for this bundle")?;
            let (pin, pin_source) = resolve_pin(&id, pubkey.as_deref(), &presented)?;
            let v = run_verify(&raw, &sha, &sig, &pin)?;
            let meta = &v.meta;

            // FRESHNESS-PINNED installs: --channel fetches + verifies a
            // channel pair and requires the bundle to BE a row of the
            // verified index (artifact-digest match). The receipt then
            // carries the freshness the row was checked under.
            let mut channel_receipt: Option<jr::ChannelReceipt> = None;
            if let Some(chsrc) = &channel {
                let (json, csig) = fetch_channel_pair(chsrc).await?;
                let anchor = resolve_channel_anchor(None)?;
                let index = pc::verify_channel(&json, &csig, &anchor, max_age_ms)?;
                let enforced = if max_age_ms == 0 {
                    pc::DEFAULT_MAX_AGE_MS
                } else {
                    max_age_ms
                };
                anyhow::ensure!(
                    index
                        .bundles
                        .iter()
                        .any(|e| e.sha256.eq_ignore_ascii_case(&sha)),
                    "--channel: the bundle ({sha}…) is NOT a row of the verified index — \
                     install it plainly, or fetch the artifact the index names"
                );
                let now = crate_now_ms();
                channel_receipt = Some(jr::ChannelReceipt {
                    source: chsrc.trim().to_string(),
                    generated_ms: index.generated_ms,
                    observed_age_ms: now.saturating_sub(index.generated_ms),
                    max_age_ms: enforced,
                });
            }

            // Downgrade / same-version guard: only an upgrade installs.
            let installed = pb::latest_installed(&meta.id)?;
            if let Some(cur) = installed {
                let ord = pb::version_cmp(&cur, &meta.version)?;
                anyhow::ensure!(
                    force || ord == Ordering::Less,
                    "plugin {}: installed {cur} is not older than {} — pass --force to \
                     reinstall (this rotates the HMAC secret and requires BOTH units to restart)",
                    meta.id,
                    meta.version
                );
            }

            let (vdir, drop_dir, _drop_manifest) = pb::install_paths(&meta.id, &meta.version);
            let _secret = pb::extract_installed(&v, &vdir)?;
            let staged = pb::stage_drop(&vdir, &drop_dir)?;
            println!("installed: {}", vdir.display());
            println!(
                "staged drop (fresh secret generated box-side): {}",
                staged.display()
            );
            println!();
            let gated = bot_toml_gates(&meta.id).unwrap_or(false);
            if gated {
                println!(
                    "next 1: SKIPPED — \"external:{}\" is already in [plugins].enabled in bot.toml",
                    meta.id
                );
            } else {
                println!(
                    "next 1: add \"external:{}\" to [plugins].enabled in {}",
                    meta.id,
                    jerith_mesh_core::config::NodeConfig::home()
                        .join("bot.toml")
                        .display()
                );
            }
            println!(
                "next 2: point the plugin unit at this binary:\n\n\
                 ExecStart = {vdir}/bin/{bin} {staged}\n",
                vdir = vdir.display(),
                bin = v.binary_name,
                staged = staged.display()
            );
            println!(
                "next 3: restart the plugin unit AND the bot — both halves read the \
                 manifest secret at process start"
            );
            jr::ReceiptEntry {
                ts_ms: crate_now_ms(),
                action: "install".into(),
                plugin: Some(meta.id.clone()),
                version: Some(meta.version.clone()),
                publisher: Some(meta.publisher.clone()),
                bundle_sha256: Some(v.tarball_sha256.clone()),
                pin: Some(jr::PinReceipt {
                    source: pin_source.into(),
                    pin: pin.clone(),
                }),
                channel: channel_receipt,
                detail: Some(format!("drop: {}", staged.display())),
                signer: None,
                prev: None,
                chain: None,
                sig: None,
            }
            .record()?;
            Ok(())
        }
        PluginCmd::Build { source, out, key } => {
            let identity = load_signing_seed(key.as_deref())?;
            let report = pb::build(&source, &out, &identity)?;
            println!(
                "built {}  ({} {} · publisher {}…)",
                report.bundle_path.display(),
                report.meta.id,
                report.meta.version,
                publisher_thumb(&report.meta.publisher),
            );
            println!("sha256: {}", report.tarball_sha256);
            println!(
                "companions: {} , {}",
                report.sha_path.display(),
                report.sig_path.display()
            );
            println!(
                "next: attach all three to the release (or host them), then \
                 'jerith plugin channel build' to list them in the signed index"
            );
            Ok(())
        }
        PluginCmd::Search {
            query,
            channel,
            anchor,
            max_age_ms,
        } => search_channel(query, channel, anchor, max_age_ms).await,
        PluginCmd::Trust { cmd } => trust_cmd(cmd).await,
        PluginCmd::Channel { cmd } => channel_cmd(cmd).await,
        PluginCmd::Receipts {
            count,
            all,
            verdict,
        } => receipts_cmd(count, all, verdict).await,
        PluginCmd::Remove { id } => {
            match pb::trash_drop(&id)? {
                Some(trash) => {
                    println!(
                        "drop moved to {} (not deleted — the audit trail holds; \
                         installed bundles stay too)",
                        trash.display()
                    );
                    // Same honesty rule as install's next steps: don't
                    // command edits that are already done. The gate row
                    // membership reads [plugins].enabled with the bot's
                    // exact semantics; the unit probe asks systemd whether
                    // the conventional plugin unit is ACTUALLY running.
                    let gated = bot_toml_gates(&id).unwrap_or(false);
                    if gated {
                        println!(
                            "next 1: remove the \"external:{id}\" row from \
                             [plugins].enabled in {}",
                            bot_toml_path().display()
                        );
                    } else {
                        println!(
                            "next 1: SKIPPED — no \"external:{id}\" row left \
                             in [plugins].enabled"
                        );
                    }
                    if plugin_unit_active(&id) {
                        println!(
                            "next 2: stop the running unit (systemctl --user stop \
                             jerith-{id}-plugin)"
                        );
                    } else {
                        println!(
                            "next 2: SKIPPED — no jerith-{id}-plugin unit is \
                             active (nothing to stop)"
                        );
                    }
                    jr::ReceiptEntry {
                        ts_ms: crate_now_ms(),
                        action: "remove".into(),
                        plugin: Some(id.clone()),
                        version: None,
                        publisher: None,
                        bundle_sha256: None,
                        pin: None,
                        channel: None,
                        detail: Some(format!("trash: {}", trash.display())),
                        signer: None,
                        prev: None,
                        chain: None,
                        sig: None,
                    }
                    .record()?;
                }
                None => {
                    println!(
                        "no live drop for {id:?} (nothing under plugins.d/{id}) — \
                         installed bundles were not touched"
                    );
                }
            }
            Ok(())
        }
    }
}

/// One search row's honest outcome, from the row's version and the
/// versions of that id already on disk: the printed tag plus whether
/// the row is actually installable (its exact version missing — not
/// merely "this id has something installed"). Test-separated from the
/// async search loop.
fn search_row_state(row_version: &str, installed: &[String]) -> (String, bool) {
    // `installed` is sort-ascending (the bundle helper sorts), so the
    // last entry is the latest.
    let latest = installed.last();
    let row_installed = installed.iter().any(|v| v == row_version);
    let tail = match latest {
        Some(l) if row_installed => format!("  · installed {l}"),
        Some(l) => format!("  · {l} installed, this row ({row_version}) is not"),
        None => String::new(),
    };
    (tail, !row_installed)
}

/// The channel pair from an https `channel.json` URL (the sig rides
/// `{.}.sig`; both downloads size-capped) or a local DIRECTORY holding
/// the pair. Shared by `channel verify` and `plugin search` (the
/// index is only ever touched AFTER verify_channel's full checks).
async fn fetch_channel_pair(source: &str) -> anyhow::Result<(String, String)> {
    if source.starts_with("https://") {
        anyhow::ensure!(
            !source.contains(char::is_whitespace),
            "channel: source url contains whitespace"
        );
        let client = jerith_mesh_core::http::mesh_http();
        let get = |url: String, cap: usize| {
            let client = client.clone();
            async move {
                let resp = client
                    .get(&url)
                    .send()
                    .await
                    .with_context(|| format!("fetching {url}"))?;
                let resp = resp
                    .error_for_status()
                    .with_context(|| format!("fetching {url}"))?;
                let bytes = resp
                    .bytes()
                    .await
                    .with_context(|| format!("reading {url}"))?;
                anyhow::ensure!(
                    bytes.len() <= cap,
                    "{url}: {} bytes — over the {cap}-byte cap",
                    bytes.len()
                );
                Ok::<_, anyhow::Error>(bytes)
            }
        };
        let json = get(source.to_string(), 256 * 1024).await?;
        let sigb = get(format!("{source}.sig"), 4096).await?;
        return Ok((
            String::from_utf8(json.to_vec())
                .map_err(|e| anyhow::anyhow!("channel.json is not utf-8 ({e})"))?,
            String::from_utf8(sigb.to_vec())
                .map_err(|e| anyhow::anyhow!("channel.json.sig is not utf-8 ({e})"))?,
        ));
    }
    let dir = PathBuf::from(source);
    anyhow::ensure!(
        dir.is_dir(),
        "channel: local source must be a DIRECTORY holding the {} pair (https urls are \
         the other form)",
        jerith_mesh_core::pluginchannel::CHANNEL_FILE
    );
    jerith_mesh_core::pluginchannel::load_channel_pair(&dir)
}

/// The pinned anchor cascade for the first-party default key: CLI
/// flag, then $JERITH_CHANNEL_ANCHOR, then the committed
/// keys/plugin-releases.pub (a repo checkout has the anchor in-tree).
/// Shared by `channel verify` and `plugin search`.
fn resolve_channel_anchor(anchor: Option<String>) -> anyhow::Result<String> {
    const NO_ANCHOR: &str = "channel: no anchor — pass --anchor <64-hex publisher pubkey> or set \
         JERITH_CHANNEL_ANCHOR (non-empty); verify runs only against a key the OPERATOR pinned";
    let found = anchor
        .or_else(|| std::env::var("JERITH_CHANNEL_ANCHOR").ok())
        .or_else(|| std::fs::read_to_string("keys/plugin-releases.pub").ok())
        .ok_or_else(|| anyhow::anyhow!(NO_ANCHOR))?;
    // Set-but-empty is the same as unset: an export like
    // `export ANCHOR=$(…) JERITH_CHANNEL_ANCHOR=$ANCHOR` expands $ANCHOR
    // before the first assignment lands (all words of a simple command
    // expand first), silently shipping an EMPTY anchor — which used to
    // surface far downstream as a bare publisher-vs-anchor mismatch
    // with a blank thumb. Fail fast, with the honest reason.
    let found = found.trim().to_string();
    anyhow::ensure!(!found.is_empty(), NO_ANCHOR);
    Ok(found)
}

/// `jerith plugin search [query] --channel <url|dir>` — the Phase 4
/// index's human lane. Verify FIRST (pinning + freshness, the full
/// channel checks), filter SECOND: an unverified index is never
/// searchable, and a row is a claim, not an install (the bundle
/// verify chain still runs on the artifact).
async fn search_channel(
    query: Option<String>,
    channel: Option<String>,
    anchor: Option<String>,
    max_age_ms: u64,
) -> anyhow::Result<()> {
    use jerith_mesh_core::pluginbundle as pb;
    use jerith_mesh_core::pluginchannel as pc;
    let source = channel
        .or_else(|| std::env::var("JERITH_CHANNEL_URL").ok())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "plugin search: no channel — pass --channel <https url | local dir> or set \
                 JERITH_CHANNEL_URL (an UNVERIFIED index is never searched)"
            )
        })?;
    let (json, sig) = fetch_channel_pair(&source).await?;
    let anchor = resolve_channel_anchor(anchor)?;
    let index = pc::verify_channel(&json, &sig, &anchor, max_age_ms)?;
    println!(
        "✓ channel verified: publisher {}… · generated {} ms ago · {} row(s)",
        {
            let p = &index.publisher;
            format!("{}…{}", &p[..8], &p[p.len() - 4..])
        },
        crate_now_ms().saturating_sub(index.generated_ms),
        index.bundles.len()
    );
    let q = query.unwrap_or_default().trim().to_lowercase();
    use jerith_mesh_core::pluginreceipt as jr;
    let mut matched = 0usize;
    // Rows whose exact version is already on disk: the honest basis
    // for the footer's "install any row" advice — an already-full slate
    // shouldn't command an install that is a no-op.
    let mut installable = 0usize;
    for e in &index.bundles {
        let hay = format!(
            "{}\x1f{}\x1f{}/{}\x1f{}",
            e.id, e.version, e.os, e.arch, e.url
        )
        .to_lowercase();
        if !q.is_empty() && !hay.contains(&q) {
            continue;
        }
        matched += 1;
        let installed = pb::installed_versions_under(&NodeConfig::home(), &e.id);
        let (tail, installable_row) = search_row_state(&e.version, &installed);
        if installable_row {
            installable += 1;
        }
        println!(
            "  {:<10} {:<8} {:<13} {}{}",
            e.id,
            e.version,
            format!("{}/{}", e.os, e.arch),
            e.url,
            tail,
        );
    }
    if matched == 0 {
        println!("  (no rows matched {q:?})");
    } else if installable == 0 {
        println!(
            "nothing to install — every matched row's exact version is already on disk \
             (reinstalling means passing that row's url to 'jerith plugin install')"
        );
    } else {
        println!(
            "install any unmatched row: jerith plugin install <url>  (the full verify chain \
             runs on the artifact regardless of the index)"
        );
    }
    jr::ReceiptEntry {
        ts_ms: crate_now_ms(),
        action: "channel-verify".into(),
        plugin: None,
        version: None,
        publisher: Some(index.publisher.clone()),
        bundle_sha256: None,
        pin: None,
        channel: Some(jr::ChannelReceipt {
            source: source.trim().to_string(),
            generated_ms: index.generated_ms,
            observed_age_ms: crate_now_ms().saturating_sub(index.generated_ms),
            max_age_ms: if max_age_ms == 0 {
                pc::DEFAULT_MAX_AGE_MS
            } else {
                max_age_ms
            },
        }),
        detail: Some(format!("search: {matched} row(s) matched {q:?}")),
        signer: None,
        prev: None,
        chain: None,
        sig: None,
    }
    .record()?;
    Ok(())
}

/// `jerith plugin trust …` — the operator's publisher trust table
/// (Phase 3, docs/PLUGIN-CHANNEL.md key model). The bot NEVER reads
/// these rows: trust lives in the CLI install lane, and the bot's
/// [plugins] code is untouched.
async fn trust_cmd(cmd: TrustCmd) -> anyhow::Result<()> {
    use jerith_mesh_core::pluginreceipt as jr;
    match cmd {
        TrustCmd::Show => {
            let bot = load_trusted_publishers()?;
            println!(
                "bot.toml [plugins.trusted_publishers] ({}): {} row(s)",
                bot_toml_path().display(),
                bot.len()
            );
            for (id, key) in &bot {
                println!("  {:<16} {}", id, publisher_thumb(key));
            }
            // Same already-done honesty as the other next-step hints:
            // when every pins row is mirrored into bot.toml, "a rotate
            // migrates" is advice for work already done — say so instead.
            let pin_header = match load_pins() {
                Ok(pins) => trust_pins_file_header(&bot, &pins),
                Err(_) => {
                    "empty — no legacy rows to migrate; rotate pins land \
                         only in bot.toml"
                }
            };
            println!(
                "pins file {} ({}):",
                jerith_mesh_core::config::NodeConfig::home()
                    .join("keys")
                    .join("plugin-pins.json")
                    .display(),
                pin_header
            );
            match load_pins() {
                Ok(pins) if !pins.is_empty() => {
                    for (id, key) in &pins {
                        match bot.get(id) {
                            Some(k) if k.eq_ignore_ascii_case(key) => {
                                println!(
                                    "  {:<16} {} (mirrored in bot.toml)",
                                    id,
                                    publisher_thumb(key)
                                );
                            }
                            _ => println!(
                                "  {:<16} {} (LEGACY — still resolves; rotate to migrate)",
                                id,
                                publisher_thumb(key)
                            ),
                        }
                    }
                }
                Ok(_) => println!("  (empty)"),
                Err(_) => println!("  (no file)"),
            }
            println!(
                "rotate: jerith plugin trust rotate <id> --new <64-hex pubkey>  (no \
                 trust-lag — the old key is refused from that moment)"
            );
            Ok(())
        }
        TrustCmd::Rotate { id, new } => {
            let id = id.trim();
            anyhow::ensure!(!id.is_empty(), "rotate: <id> required");
            let new_key = normalize_key(&new);
            anyhow::ensure!(
                valid_pubkey_hex(&new_key),
                "rotate: --new must be exactly 64 hex chars (the new Ed25519 publisher \
                 pubkey), got {new:?}"
            );
            let prior = load_trusted_publishers()?.get(id).cloned();
            let path = write_trusted_publishers(&bot_toml_path(), id, &new_key)?;
            match prior {
                Some(ref old) if old.eq_ignore_ascii_case(&new_key) => println!(
                    "rotated plugin {id:?}: pin already {}… in {} (rewritten in place)",
                    publisher_thumb(old),
                    path.display()
                ),
                Some(ref old) => println!(
                    "rotated plugin {id:?}: {}… → {}… in {} — NO trust-lag: installs signed \
                     by the old key are refused from this moment (a next install must \
                     verify against the new pin)",
                    publisher_thumb(old),
                    publisher_thumb(&new_key),
                    path.display()
                ),
                None => println!(
                    "pinned {id:?} (first row): {}… in {} — installs of this plugin now \
                     verify against this key",
                    publisher_thumb(&new_key),
                    path.display()
                ),
            }
            // A disagreeing legacy pins-file row is a stale trust
            // source once bot.toml holds the new pin — name it.
            if let Ok(pins) = load_pins()
                && let Some(old) = pins.get(id)
                && !normalize_key(old).eq_ignore_ascii_case(&new_key)
            {
                println!(
                    "note: the legacy pins file still maps {id:?} → {} — bot.toml \
                     now outranks it; remove that row to keep one source of truth",
                    publisher_thumb(old)
                );
            }
            // The rotation IS the audit event: what changed, for which id.
            jr::ReceiptEntry {
                ts_ms: crate_now_ms(),
                action: "pin-rotate".into(),
                plugin: Some(id.to_string()),
                version: None,
                publisher: Some(new_key.clone()),
                bundle_sha256: None,
                pin: None,
                channel: None,
                signer: None,
                prev: None,
                chain: None,
                sig: None,
                detail: Some(match &prior {
                    Some(old) => format!("old {old} → new {new_key}"),
                    None => format!("first pin: {new_key}"),
                }),
            }
            .record()?;
            Ok(())
        }
    }
}

/// `jerith plugin channel …` — the signed publisher index (Phase 4):
/// build + sign a pair over a dist dir, or fetch + verify a published
/// one. The index is a claim sheet: installing an entry still runs the
/// full bundle verify chain against the actual artifact.
async fn channel_cmd(cmd: ChannelCmd) -> anyhow::Result<()> {
    use jerith_mesh_core::pluginbundle as pb;
    use jerith_mesh_core::pluginchannel as pc;
    use jerith_mesh_core::pluginreceipt as jr;
    match cmd {
        ChannelCmd::Build { dir, url_base, key } => {
            let dir = dir.unwrap_or_else(|| PathBuf::from("dist"));
            anyhow::ensure!(
                url_base.starts_with("https://") && !url_base.contains(char::is_whitespace),
                "channel: --url-base must be https and whitespace-free (rows inherit it)"
            );
            let identity = load_signing_seed(key.as_deref())?;

            let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
                .with_context(|| format!("channel: reading {}", dir.display()))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "jbundle"))
                .collect();
            paths.sort();
            anyhow::ensure!(
                !paths.is_empty(),
                "channel: no .jbundle triples in {} — build the bundles first",
                dir.display()
            );

            let mut rows = Vec::new();
            for path in paths {
                let raw = std::fs::read(&path)?;
                let (id, version, publisher) = pb::peek_identity(&raw)
                    .with_context(|| format!("channel: reading {}", path.display()))?;
                anyhow::ensure!(
                    publisher == identity.pubkey_hex(),
                    "channel: {} was published by {}… but the signing key is {}… — a channel may only name its own publisher's bundles",
                    path.display(),
                    publisher_thumb(&publisher),
                    publisher_thumb(&identity.pubkey_hex())
                );
                let stem = path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .trim_end_matches(".jbundle")
                    .to_string();
                // <id>-<version>-<os>-<arch>: peel the known head, then
                // the tail's LAST dash separates os|arch (none of today's
                // os/arch tokens contain dashes: linux, macos, x86_64, aarch64).
                let tail = stem
                    .strip_prefix(&format!("{id}-{version}-"))
                    .ok_or_else(|| anyhow::anyhow!(
                        "channel: {} does not match the <id>-<version>-<os>-<arch>.jbundle name",
                        path.display()
                    ))?;
                let (os, arch) = tail.rsplit_once('-').ok_or_else(|| {
                    anyhow::anyhow!("channel: {} lacks an <os>-<arch> tail", path.display())
                })?;
                let sha_raw = std::fs::read_to_string(format!("{}.sha256", path.display()))
                    .with_context(|| {
                        format!("channel: missing companion for {}", path.display())
                    })?;
                let sha = pb::parse_sha256_file(&sha_raw)?;
                let url = format!("{url_base}/{}", stem);
                rows.push(pc::ChannelEntry {
                    id,
                    version,
                    os: os.to_string(),
                    arch: arch.to_string(),
                    url,
                    sha256: sha,
                    notes: String::new(),
                });
            }
            let row_count = rows.len();
            let (json, sig) = pc::build_channel(rows, &identity, crate_now_ms())?;
            let json_path = dir.join(pc::CHANNEL_FILE);
            let sig_path = dir.join(pc::CHANNEL_SIG_FILE);
            std::fs::write(&json_path, &json)?;
            std::fs::write(&sig_path, sig.trim())?;
            println!("channel: {} row(s)", row_count);
            println!("wrote {} + {}", json_path.display(), sig_path.display());
            println!(
                "next: upload BOTH files beside the bundles; operators verify via \
                 'jerith plugin channel verify <url> --anchor <pub hex>'"
            );
            Ok(())
        }
        ChannelCmd::Verify {
            source,
            anchor,
            max_age_ms,
        } => {
            let (json, sig) = fetch_channel_pair(&source).await?;
            let anchor = resolve_channel_anchor(anchor)?;
            let index = pc::verify_channel(&json, &sig, &anchor, max_age_ms)?;
            println!(
                "✓ channel verified: publisher {}… · generated {} ms ago · {} row(s)",
                publisher_thumb(&index.publisher),
                crate_now_ms().saturating_sub(index.generated_ms),
                index.bundles.len()
            );
            for e in &index.bundles {
                println!(
                    "  {:<10} {:<8} {:<13} {}",
                    e.id,
                    e.version,
                    format!("{}/{}", e.os, e.arch),
                    e.url
                );
            }
            println!(
                "install any row: jerith plugin install <url>  (the full verify \
                 chain runs against the artifact regardless of the index)"
            );
            jr::ReceiptEntry {
                ts_ms: crate_now_ms(),
                action: "channel-verify".into(),
                plugin: None,
                version: None,
                publisher: Some(index.publisher.clone()),
                bundle_sha256: None,
                pin: Some(jr::PinReceipt {
                    source: "anchor".into(),
                    pin: anchor.clone(),
                }),
                channel: Some(jr::ChannelReceipt {
                    source: source.trim().to_string(),
                    generated_ms: index.generated_ms,
                    observed_age_ms: crate_now_ms().saturating_sub(index.generated_ms),
                    max_age_ms: if max_age_ms == 0 {
                        pc::DEFAULT_MAX_AGE_MS
                    } else {
                        max_age_ms
                    },
                }),
                detail: Some(format!("{} row(s)", index.bundles.len())),
                signer: None,
                prev: None,
                chain: None,
                sig: None,
            }
            .record()?;
            Ok(())
        }
    }
}

/// `jerith plugin receipts` — read back the audit log, newest tail last.
/// `--verdict` is the scriptable form: one line, exit 0 only when the
/// chain stands (no human table, nothing else to parse).
async fn receipts_cmd(count: usize, all: bool, verdict: bool) -> anyhow::Result<()> {
    use jerith_mesh_core::pluginreceipt::{self as jr, LogIntegrity, read_verified};
    let log = read_verified();
    if verdict {
        // Ok = Intact, Err = Broken/Truncated (the one-liner carries the
        // same words the human printout uses).
        let line = receipt_verdict_line(&log);
        println!("{}", line.as_deref().unwrap_or_else(|e| e));
        if line.is_err() {
            std::process::exit(1);
        }
        return Ok(());
    }
    let entries = &log.entries;
    let path = jr::receipts_path();
    if entries.is_empty() {
        // An empty LOG but a NON-genesis head is not "no receipts yet"
        // — it is a wiped log whose witness file survived.
        if let LogIntegrity::Truncated { head, last } = &log.integrity {
            println!(
                "{} — no entries, but the chain head file expects {head}… and the log ends at {last}…",
                path.display()
            );
            println!("⚠ receipt log was TRUNCATED (wiped) — the head witness survived");
            return Ok(());
        }
        println!(
            "{} — no receipts yet (installs, removals, pin rotations \
                  and channel verifications append here)",
            path.display()
        );
        return Ok(());
    }
    let skip = if all {
        0
    } else {
        entries.len().saturating_sub(count)
    };
    println!(
        "{} — {} entr{} (showing {})",
        path.display(),
        entries.len(),
        if entries.len() == 1 { "y" } else { "ies" },
        entries.len() - skip
    );
    // The integrity verdict goes right under the header — the reader
    // decides per-line whether to trust what follows.
    match &log.integrity {
        LogIntegrity::Intact => {
            let sig_note = if log.signed > 0 {
                format!(
                    ", {} signed by the receipts key in {}",
                    log.signed,
                    jerith_mesh_core::pluginreceipt::RECEIPT_PUB_FILE
                )
            } else {
                String::new()
            };
            println!(
                "  chain intact: {} chained, hash-linked ✓ (head file in sync){sig_note}",
                log.chained
            );
        }
        LogIntegrity::Broken { at, why } => {
            println!("  ⚠ CHAIN BROKEN at entry #{at}: {why}");
            println!("        entries below are as-found on disk — treat as EVIDENCE, not truth");
        }
        LogIntegrity::Truncated { head, last } => {
            println!("  ⚠ LOG TRUNCATED: head file expects {head}… but the log ends at {last}…");
            println!(
                "        the tail was removed — entries below are as-found, treat as EVIDENCE, not truth"
            );
        }
    }
    println!();
    for e in &entries[skip..] {
        let subject = match (&e.plugin, &e.version) {
            (Some(p), Some(v)) => format!("{p} {v}"),
            (Some(p), None) => p.clone(),
            _ => "-".into(),
        };
        println!(
            "{:<13} {:<15} {:<24} {}",
            e.ts_ms,
            e.action,
            subject.trim_end(),
            e.detail.as_deref().unwrap_or("-")
        );
        if let Some(pin) = &e.pin {
            println!("               pin {}…  ({})", &pin.pin[..8], pin.source);
        }
        if let Some(c) = &e.channel {
            println!(
                "               channel {} — generated_ms {} (age {} ms ≤ {} ms window)",
                c.source, c.generated_ms, c.observed_age_ms, c.max_age_ms
            );
        }
        if let Some(pub_key) = &e.publisher {
            println!("               publisher {}…", &pub_key[..8]);
        }
    }
    Ok(())
}

/// The receipt-log gate (docs/PLUGIN-CHANNEL.md): trust LOADING refuses
/// to run on top of evidence that the receipt log was tampered with —
/// the log is the lane's own witness, and a Broken/Truncated verdict
/// means its testimony is in question. Removal paths (`plugin remove`,
/// a `trust rotate`) stay ungated: they are the operator's countermeasures
/// when something looks compromised, not attack surfaces, and gating
/// them would turn tampering evidence into a recovery deadlock.
/// `allow_override` (--receipt-allow-broken) is the crash-recovery
/// escape: a torn append can leave the head witness one receipt behind
/// the log, and the next receipt REWRITES the head — so overriding a
/// Truncated verdict self-heals. Overriding a Broken verdict does not
/// hide anything: the chain break stays in the log for every later
/// reader to see, and the override gets recorded as its own receipt.
fn receipt_gate_under(home: &Path, allow_override: bool) -> anyhow::Result<()> {
    use jerith_mesh_core::pluginreceipt::{LogIntegrity, read_verified_under};
    match read_verified_under(home).integrity {
        LogIntegrity::Intact => Ok(()),
        LogIntegrity::Truncated { head, last } => {
            anyhow::ensure!(
                allow_override,
                "receipt log: TRUNCATED (head witness {}… but the log's tail is {}…) — \
                 install REFUSED while its audit trail looks removed. \
                 Investigate first; if this immediately followed a crash mid-append, \
                 --receipt-allow-broken proceeds and the next receipt re-syncs the head",
                &head[..8],
                &last[..8]
            );
            println!(
                "⚠ receipt log: TRUNCATED — proceeding on override; the receipt just written re-syncs the head witness"
            );
            Ok(())
        }
        LogIntegrity::Broken { at, why } => {
            anyhow::ensure!(
                allow_override,
                "receipt log: CHAIN BROKEN at entry #{at} ({why}) — \
                 install REFUSED: do not load trust on top of tampering evidence. \
                 Investigate; --receipt-allow-broken overrides WITHOUT hiding the break"
            );
            println!(
                "⚠ receipt log: CHAIN BROKEN at entry #{at} ({why}) — proceeding on override; the break stays visible"
            );
            Ok(())
        }
    }
}

/// The `--verdict` one-liner for the receipt log — the scriptable form
/// of the same verdict `receipts_cmd` prints for humans. `Ok(line)` = the
/// chain stands; `Err(line)` = Broken/Truncated (the caller prints the
/// line and exits nonzero). Kept a pure function so the words are
/// unit-pinned without a real exit.
fn receipt_verdict_line(log: &jerith_mesh_core::pluginreceipt::Verdict) -> Result<String, String> {
    use jerith_mesh_core::pluginreceipt::{LogIntegrity, RECEIPT_PUB_FILE};
    match &log.integrity {
        LogIntegrity::Intact => {
            let sig_note = if log.signed > 0 {
                format!(", {} signed ({RECEIPT_PUB_FILE})", log.signed)
            } else {
                String::new()
            };
            Ok(format!(
                "chain intact: {} chained, hash-linked (head in sync){sig_note}",
                log.chained
            ))
        }
        LogIntegrity::Truncated { head, last } => Err(format!(
            "LOG TRUNCATED: head file expects {}… but the log ends at {}…",
            &head[..8],
            &last[..8]
        )),
        LogIntegrity::Broken { at, why } => Err(format!("CHAIN BROKEN at entry #{at}: {why}")),
    }
}

fn crate_now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod receipt_lane_tests {
    use super::*;

    fn gh(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("jerith-gate-test-{}-{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn receipt(detail: &str) -> jerith_mesh_core::pluginreceipt::ReceiptEntry {
        jerith_mesh_core::pluginreceipt::ReceiptEntry {
            ts_ms: crate_now_ms(),
            action: "test".into(),
            plugin: Some("weather".into()),
            version: None,
            publisher: None,
            bundle_sha256: None,
            pin: None,
            channel: None,
            detail: Some(detail.into()),
            signer: None,
            prev: None,
            chain: None,
            sig: None,
        }
    }

    #[test]
    fn anchor_resolver_treats_empty_and_blank_as_no_anchor() {
        for blank in [None, Some(String::new()), Some("   ".into())] {
            let err = resolve_channel_anchor(blank).unwrap_err().to_string();
            assert!(err.contains("no anchor"), "{err}");
        }
        let a = resolve_channel_anchor(Some("  ab\n".into())).unwrap();
        assert_eq!(a, "ab");
    }

    #[test]
    fn gate_allows_missing_or_empty_log() {
        let home = gh("gate-empty");
        receipt_gate_under(&home, false).unwrap();
        std::fs::create_dir_all(&home).unwrap();
        std::fs::write(
            jerith_mesh_core::pluginreceipt::receipts_path_under(&home),
            "",
        )
        .unwrap();
        receipt_gate_under(&home, false).unwrap();
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn gate_refuses_truncated_and_the_override_self_heals() {
        let home = gh("gate-trunc");
        receipt("one").record_under(&home).unwrap();
        receipt("two").record_under(&home).unwrap();
        // Remove the log's tail; head stays at the true last digest.
        let path = jerith_mesh_core::pluginreceipt::receipts_path_under(&home);
        let ls: Vec<String> = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect();
        assert_eq!(ls.len(), 2);
        std::fs::write(&path, format!("{}\n", ls[0])).unwrap();
        let err = receipt_gate_under(&home, false).unwrap_err().to_string();
        assert!(err.contains("TRUNCATED"), "{err}");
        assert!(err.contains("REFUSED"), "{err}");
        // Override proceeds… and the receipt it would append re-syncs
        // the head witness so the gate passes clean afterwards.
        receipt_gate_under(&home, true).unwrap();
        receipt("the override itself").record_under(&home).unwrap();
        receipt_gate_under(&home, false).unwrap();
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn gate_refuses_a_broken_chain_even_on_override_path_message() {
        let home = gh("gate-broken");
        receipt("one").record_under(&home).unwrap();
        receipt("two").record_under(&home).unwrap();
        // In-place edit of line 1 without touching its digest.
        let path = jerith_mesh_core::pluginreceipt::receipts_path_under(&home);
        let raw = std::fs::read_to_string(&path).unwrap();
        let tampered = raw.replace("two", "tampered");
        std::fs::write(&path, tampered).unwrap();
        let err = receipt_gate_under(&home, false).unwrap_err().to_string();
        assert!(err.contains("CHAIN BROKEN"), "{err}");
        assert!(err.contains("entry #1"), "{err}");
        // Override works but loudly; the log keeps the break.
        receipt_gate_under(&home, true).unwrap();
        let v = jerith_mesh_core::pluginreceipt::read_verified_under(&home);
        assert!(matches!(
            v.integrity,
            jerith_mesh_core::pluginreceipt::LogIntegrity::Broken { .. }
        ));
        let _ = std::fs::remove_dir_all(&home);
    }

    /// `--verdict`'s words: the three integrity shapes read exactly as
    /// the human printout does, so scripts and operators see one truth.
    #[test]
    fn verdict_line_tracks_the_integrity_verdict() {
        let home = gh("verdict-line");
        std::fs::create_dir_all(&home).unwrap();
        // Empty log → intact.
        std::fs::write(
            jerith_mesh_core::pluginreceipt::receipts_path_under(&home),
            "",
        )
        .unwrap();
        let log = jerith_mesh_core::pluginreceipt::read_verified_under(&home);
        let line = receipt_verdict_line(&log);
        assert!(
            line.as_deref().unwrap_or("err").contains("chain intact"),
            "{line:?}"
        );
        // In-place edit → CHAIN BROKEN with the entry number.
        receipt("one").record_under(&home).unwrap();
        receipt("two").record_under(&home).unwrap();
        let path = jerith_mesh_core::pluginreceipt::receipts_path_under(&home);
        let raw = std::fs::read_to_string(&path).unwrap();
        std::fs::write(&path, raw.replace("two", "tampered")).unwrap();
        let log = jerith_mesh_core::pluginreceipt::read_verified_under(&home);
        let line = receipt_verdict_line(&log).unwrap_err();
        assert!(line.contains("CHAIN BROKEN"), "{line}");
        assert!(line.contains("entry #1"), "{line}");
        // Tail removal → LOG TRUNCATED naming both digests.
        std::fs::write(&path, format!("{}\n", raw.lines().next().unwrap())).unwrap();
        let log = jerith_mesh_core::pluginreceipt::read_verified_under(&home);
        let line = receipt_verdict_line(&log).unwrap_err();
        assert!(line.contains("LOG TRUNCATED"), "{line}");
        let _ = std::fs::remove_dir_all(&home);
    }
}

/// `jerith stack …` — the optional AI companion stack + small-model
/// catalog. Install steps run each tool's own official installer, echoed
/// before execution; the CLI invocation itself is the consent.
fn stack_cmd(cmd: StackCmd) -> anyhow::Result<()> {
    use jerith_mesh_core::toolstack::{ModelTier, StackTool, small_models};
    let os = jerith_mesh_core::hw::probe().os;
    match cmd {
        StackCmd::List { models } => {
            if !models {
                println!("stack tools (`jerith stack install <name>`):");
                for t in StackTool::ALL {
                    let state = if detect_tool(*t) {
                        "present   "
                    } else {
                        "not found "
                    };
                    println!("  {:<10} {} {}", t.name(), state, t.description());
                }
            }
            let _ = small_models(ModelTier::Thinking).len();
            println!("\nsmall-model tiers (ollama tags — `jerith stack pull <tag>`):",);
            for tier in [ModelTier::Thinking, ModelTier::Vision, ModelTier::Embedding] {
                println!("  {} (config: {}):", tier.label(), tier.config_field());
                let def = jerith_mesh_core::toolstack::default_small_model(tier).name;
                for m in small_models(tier) {
                    let marker = if m.name == def {
                        " · mesh default"
                    } else {
                        ""
                    };
                    println!(
                        "    {:<26} {:<10} {}{}",
                        m.name, m.size_hint, m.note, marker
                    );
                }
            }
            Ok(())
        }
        StackCmd::Status => {
            for t in StackTool::ALL {
                let (cmd, args) = t.detect_cmd();
                println!(
                    "{:<10} {:<10} ({} {})",
                    t.name(),
                    if detect_tool(*t) {
                        "present"
                    } else {
                        "missing"
                    },
                    cmd,
                    args.join(" ")
                );
            }
            Ok(())
        }
        StackCmd::Install { ref tool } | StackCmd::InstallPlan { ref tool } => {
            let t = StackTool::from_name(tool).with_context(|| {
                format!(
                    "unknown stack tool {tool:?} — options: {}",
                    StackTool::ALL
                        .iter()
                        .map(|x| x.name())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })?;
            let steps = t.install_plan(os);
            anyhow::ensure!(
                !steps.is_empty(),
                "{}: no automated install for this OS — install it manually",
                t.name()
            );
            if let StackCmd::InstallPlan { .. } = cmd {
                println!(
                    "install plan for {} ({} step(s), not run):",
                    t.name(),
                    steps.len()
                );
                for s in &steps {
                    println!("  $ {s}");
                }
                return Ok(());
            }
            // pip tools share the node-managed venv — create it once.
            if t.uses_tools_venv() {
                let python = if cfg!(windows) {
                    r"Scripts\python.exe"
                } else {
                    "bin/python"
                };
                let venv_py = NodeConfig::home().join("tools-venv").join(python);
                if !venv_py.exists() {
                    for step in StackTool::venv_plan(os) {
                        shell_run(&step, "tools-venv")?;
                    }
                }
            }
            for step in &steps {
                shell_run(step, t.name())?;
            }
            println!("✓ {} installed", t.name());
            match t {
                StackTool::Ollama => println!(
                    "  point the node at it: jerith config set \n  runtime_url=http://127.0.0.1:11434"
                ),
                StackTool::LiteLLM => println!(
                    "  point the node at it: jerith config set runtime_url=http://127.0.0.1:4000"
                ),
                StackTool::LangChain | StackTool::LangGraph | StackTool::LangSmith => println!(
                    "  see integrations/langchain/ in the repo — the examples use this node's \n  gateway as an OpenAI-compatible endpoint (LangSmith: set LANGSMITH_TRACING=true \n  and note the LANGSMITH_API_KEY env vars; never commit them)"
                ),
                StackTool::Postgres => println!(
                    "  then: jerith config set memory_backend=postgres \\\n                     memory_dsn=postgres://postgres@127.0.0.1:5432/jerith\\?sslmode=disable"
                ),
            }
            Ok(())
        }
        StackCmd::Pull { model } => {
            let tag = model.trim().trim_start_matches("ollama:").to_string();
            anyhow::ensure!(!tag.is_empty(), "empty ollama tag");
            // Confirm the tag is servable before pulling (free-form names
            // are fine — ollama itself validates against its registry).
            shell_run(&format!("ollama pull {tag}"), "ollama")
        }
    }
}

/// Run one stack shell step (sh -c | cmd /C), output inherited, failure
/// loud. `label` is just the echoed context tag.
fn shell_run(step: &str, label: &str) -> anyhow::Result<()> {
    println!("  [{label}] $ {step}");
    let (program, args) = if cfg!(windows) {
        ("cmd", vec!["/C", step])
    } else {
        ("sh", vec!["-c", step])
    };
    let status = std::process::Command::new(program)
        .args(&args)
        .status()
        .with_context(|| format!("spawning {program} for [{label}]"))?;
    anyhow::ensure!(status.success(), "[{label}] failed: {step}");
    Ok(())
}

/// Small sync shim for the (async) bootstrap count fetch inside setup.
fn futures_block_on<F: std::future::Future>(fut: F) -> F::Output {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    rt.block_on(fut)
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

async fn run_node(switch_model: Option<String>) -> anyhow::Result<()> {
    let mut cfg = NodeConfig::load()?.context("no config found — run `jerith setup` first")?;
    if let Some(spec) = switch_model {
        let mid = parse_model_spec(&spec)?;
        cfg.set_model(mid);
        cfg.save()?;
    }
    cfg.validate()?;

    // Phase 5 stake join gate: a Contributor/Hybrid with `required_uj` > 0
    // must show the gate clearing, or it runs this session demoted to
    // Requester (can consume compute, never offered work). Config is not
    // modified — the node self-restores its role on the next start.
    //
    // The verdict input is the SAME chain truth the gate reads: escrowed
    // stake (the boot auto-escrow covers a funded wallet), or a wallet
    // balance that will fund it. The RPC is the operator resolver the
    // gate itself uses — $JERITH_CHAIN_URL → config `chain_url` → the
    // co-hosted :8322 — NOT the legacy miner-facing chain_rpc_url port,
    // whose /balance is a different ledger (the 2026-10-18 production
    // boot demoted hybrid → requester off that mismatch despite a live
    // 1 JER escrow).
    if cfg.crypto.stake.required_uj > 0 && cfg.role != NodeRole::Requester {
        let required = cfg.crypto.stake.required_uj;
        let rpc = chain_base(Some(&cfg));
        let stake = match cfg.identity() {
            Ok(id) => Some(jerith_mesh_core::stake::StakeManager::new(
                &cfg.crypto,
                Some(id),
                Some(rpc.clone()),
            )),
            Err(_) => None, // no usable identity — balance view still may probe
        };
        let staked = match stake.as_ref() {
            Some(s) => s
                .chain_view()
                .await
                .and_then(|v| v.get("staked_uj").and_then(|x| x.as_u64())),
            None => None,
        };
        // The balance view is only evidence when the escrow view cannot
        // clear the gate on its own (a funded wallet the boot escrow will
        // lock, or chain truth still syncing). A live escrow skips it —
        // no noise on the happy path.
        let balance = if staked.is_none_or(|s| s < required) {
            match jerith_mesh_core::economics::chain_balance_uj(&rpc, &cfg.node_id).await {
                Ok(b) => Some(b),
                Err(e) => {
                    println!(
                        "   stake       balance probe unavailable ({e}) — escrow view governs"
                    );
                    None
                }
            }
        } else {
            None
        };
        match (staked, balance) {
            (Some(s), _) if s >= required => {
                println!(
                    "   stake       escrow live: {} µJER staked (required {}) — gate clears",
                    s, required
                );
            }
            (staked, Some(b)) if b >= required => {
                println!(
                    "   stake       {} µJER wallet (required {}; escrow view: {:?}) — boot auto-escrow will land",
                    b, required, staked
                );
            }
            (staked, balance) => {
                println!(
                    "   stake       XXX insufficient: escrow {:?} / balance {:?} < required {} — demoting to requester for this session",
                    staked, balance, required
                );
                cfg.role = NodeRole::Requester;
            }
        }
    }

    println!("🦜 jerith-mesh starting");
    let model = cfg.model.clone().expect("validated");
    println!("   model       {} ({})", model.id, model.quant);
    println!("   node_id     {}", cfg.node_id);
    println!("   role        {}", role_label(cfg.role));
    println!(
        "   contribution {}% (min {})",
        cfg.contribution_percent, MIN_CONTRIBUTION_PERCENT
    );
    println!("   bootstrap   {}", cfg.bootstrap_url);
    println!("   crypto      {}", crypto_label(cfg.crypto.chain));
    println!("   gateway     http://127.0.0.1:{}", cfg.gateway_port);

    let (state, _health_watch) = MeshState::new(cfg.clone())?;

    // JER miner: active when crypto chain is JerithChain (spec default).
    let miner = jerith_mesh_core::economics::MinerConfig::from_crypto(&cfg.crypto, &cfg.node_id)
        .map(|mc| jerith_mesh_core::economics::JerithMiner::new(mc, &cfg.node_id));
    match &miner {
        Some(_) => {
            println!(
                "   economics   JER mining enabled ({}/1000 chance per served inference)",
                cfg.crypto.mine_chance_permille
            );
        }
        None => {
            // §9 typed refusal: a configured-but-unimplemented optional
            // chain says WHY it is off, in the banner as everywhere else.
            if let Some(note) = jerith_mesh_core::economics::optional_chain_note(cfg.crypto.chain) {
                println!("   economics   off — {note}");
            } else {
                println!("   economics   off");
            }
        }
    }

    // Local memory: the backend the operator chose is the one opened.
    // A postgres choice without a usable store fails honest and loud —
    // misconfiguration is a boot error (as with the resource ceilings);
    // an unreachable store degrades to "no memory" (warned, never a
    // silent swap). Embeddings stay local either way. The handle lives
    // in MeshState so the config hot-reload can live-swap it later.
    if cfg.specialists.embeddings_enabled {
        match jerith_mesh_core::memory::MemoryHandle::open_for_config(&cfg).await {
            Ok(store) => {
                println!("   memory      {}, local-only", store.describe());
                *state.memory.write().await = Some(std::sync::Arc::new(store));
            }
            Err(e) => {
                if cfg.memory_backend == MemoryBackend::PostgresPgvector
                    && cfg.effective_memory_dsn().is_none()
                {
                    return Err(e);
                }
                tracing::warn!(error = %e, "memory store unavailable");
                println!("   memory      unavailable ({e})");
            }
        }
    }

    // Phase 2 tensor transport: open the local expert weight store. This
    // node answers peers' ExpertQuery requests with the expert's real
    // SwiGLU tensors (on-demand load, byte-budgeted LRU) and advertises
    // what it holds so placement can route accordingly. Created even when
    // empty: an empty store advertises "holds nothing" honestly.
    let expert_store = match jerith_mesh_core::expert_store::ExpertStore::new(
        cfg.experts.effective_dir(),
        cfg.experts.cache_mb,
    ) {
        Ok(store) => {
            let hosted = store.hosted().len();
            println!(
                "   experts     {} ({} hosted, cache {} MiB){}",
                store.dir().display(),
                hosted,
                store.cache_budget_bytes() / (1024 * 1024),
                if cfg.experts.tensors_only {
                    ""
                } else {
                    " [legacy text stand-in enabled]"
                }
            );
            Some(std::sync::Arc::new(store))
        }
        Err(e) => {
            tracing::warn!(error = %e, "expert store unavailable — no tensor transport");
            None
        }
    };
    if let Some(store) = &expert_store {
        state.set_expert_store(store.clone()).await;
    }

    // Spawn the mesh loop (discovery + pings + meters).
    let mesh_loop = jerith_mesh_core::ping::MeshLoop::spawn(state.clone(), Duration::from_secs(10));
    // §13 graceful exit: SIGINT/SIGTERM stop the mesh loop and WAIT for
    // it — the stop arm returns the escrow on the way out, and the
    // await means the unlock completes BEFORE the process exits. The
    // gateway serve below never returns on its own, so this task is
    // the process's exit path (the abrupt-cut alternative — leaving a
    // marker behind for the next boot's chain-view recovery — stays
    // honest too; the SIGKILL drill proved it).
    tokio::spawn({
        let mesh_loop = mesh_loop;
        async move {
            // Either terminate signal stops the mesh loop and waits for
            // its graceful suffix (the escrow return) before exiting.
            #[cfg(unix)]
            {
                use tokio::signal::unix::{SignalKind, signal};
                let mut term = signal(SignalKind::terminate()).expect("SIGTERM handler");
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {},
                    _ = term.recv() => {},
                }
            }
            #[cfg(not(unix))]
            tokio::signal::ctrl_c().await.expect("ctrl_c handler");
            tracing::info!("graceful exit: stopping the mesh loop (escrow return runs inside)");
            mesh_loop.stop().await;
            tracing::info!("mesh loop stopped — exiting");
            std::process::exit(0);
        }
    });

    // Mesh fallback for chat: a QUIC client endpoint + router, wired into
    // the gateway so /v1/chat/completions can fail over to runtime-backed
    // peers when the local runtime is unavailable.
    let mesh_fallback = match jerith_mesh_core::ping::MeshFallbackChat::spawn_with_options(
        state.verify.clone(),
        // Full router knobs: Phase 4 replay policy + the Phase 3 async
        // chat-walk shape (hedged in-flight count, stagger, walk budget).
        jerith_mesh_core::quic::RouterOptions {
            nearby_rtt_ms: 50,
            dispatch_timeout: Duration::from_secs(30),
            speculative_remote: true,
            verify_tolerance: 0.0,
            verify_percent: cfg.verify_percent,
            chat_hedged_peers: cfg.chat_hedged_peers,
            chat_hedge_delay_ms: cfg.chat_hedge_delay_ms,
            chat_walk_budget_ms: cfg.chat_walk_budget_ms,
        },
    ) {
        Ok(mf) => {
            if let Some(store) = &expert_store {
                mf.router.attach_expert_store(store.clone()).await;
            }
            Some(mf)
        }
        Err(e) => {
            tracing::warn!(error = %e, "mesh chat fallback unavailable");
            None
        }
    };

    // Phase 1 resource isolation: CPU pinning (best-effort, Linux) + the
    // background governor enforcing RAM/VRAM ceilings with the kill switch.
    // Local chat + the WebUI NEVER stop; only mesh work is gated.
    if cfg.resources.pin_cores > 0 {
        jerith_mesh_core::isolate::apply_pinning(cfg.resources.pin_cores);
    }
    // Phase 3 per-peer blocking: quiet when the block list is empty (no
    // ids configured), visible when it is not — the operator set a policy.
    if !cfg.peers.blocked.is_empty() {
        println!(
            "   peers       block list: {} id(s) — inbound RPC refused, placement + replication never use them",
            cfg.peers.blocked.len()
        );
    }
    let governor = match (
        cfg.resources.ram_ceiling_bytes,
        cfg.resources.vram_ceiling_bytes,
    ) {
        (None, None) => {
            println!("   resources   no ceilings configured — governor idle");
            None
        }
        _ => {
            println!(
                "   resources   {}{}kill switch {} — governs mesh work only",
                cfg.resources
                    .ram_ceiling_bytes
                    .map(|b| format!(
                        "RAM ceiling {:.1} GiB ",
                        b as f64 / (1024.0 * 1024.0 * 1024.0)
                    ))
                    .unwrap_or_default(),
                cfg.resources
                    .vram_ceiling_bytes
                    .map(|b| format!(
                        "VRAM ceiling {:.1} GiB ",
                        b as f64 / (1024.0 * 1024.0 * 1024.0)
                    ))
                    .unwrap_or_default(),
                if cfg.resources.kill_switch {
                    "ARMED"
                } else {
                    "off"
                }
            );
            Some(jerith_mesh_core::isolate::spawn_governor(
                state.clone(),
                Duration::from_secs(10),
            ))
        }
    };
    let _ = &governor; // bound for the whole run; dropping it would stop the loop

    // Phase 2 replication: pull under-replicated hot experts from peers
    // that hold them. Runs only when a store is attached and the config
    // leaves it on; weight transfer itself stays available either way.
    //
    // Bound for the whole run on purpose: dropping the manager stops its
    // loop, so a temporary binding would spawn it and immediately end it.
    let _replicas = match (&expert_store, &mesh_fallback) {
        (Some(_), Some(fb)) if cfg.experts.replication_enabled => {
            let interval = Duration::from_secs(cfg.experts.replication_interval_secs.max(30));
            let policy = cfg.experts.replication_config();
            println!(
                "   replicas    hot {}% (factor {}) every {}s, ≤{} fetch/round",
                policy.hot_percent,
                policy.replication_factor,
                interval.as_secs(),
                policy.max_actions_per_round
            );
            Some(jerith_mesh_core::ping::ReplicaManager::spawn(
                state.clone(),
                fb.clone(),
                interval,
                policy,
            ))
        }
        _ => None,
    };

    // Jerith Guard: compiled outbound content filter (PII redaction,
    // prompt-injection / malware blocklists, do-not-exfiltrate rules).
    let guard = if cfg.guard.enabled {
        match jerith_mesh_core::guard::Guard::build(&cfg.guard) {
            Ok(g) => Some(std::sync::Arc::new(g)),
            Err(e) => {
                tracing::warn!(error = %e, "jerith-guard config invalid — guard DISABLED (outbound screening off)");
                None
            }
        }
    } else {
        None
    };

    // Web access skills (OpenClaw migration): Brave search + URL fetch.
    // Disabled in config, or a search call simply reports "missing Brave
    // API key" with the resolution order as the hint.
    let web = if cfg.websearch.enabled {
        let ws = cfg.websearch.clone();
        if let Some(_key) = ws.resolved_api_key().as_ref() {
            tracing::info!(provider = %ws.provider, "web skills enabled (brave search + url fetch)");
        } else {
            tracing::warn!(
                "web skills enabled but no Brave API key found — set [websearch] api_key/api_key_path, \
                 $JERITH_BRAVE_API_KEY, or copy the OpenClaw keyring to ~/.jerith-mesh/brave-key"
            );
        }
        let svc = jerith_mesh_core::websearch::WebSearchService::new(
            &ws.base_url,
            ws.resolved_api_key(),
            ws.default_count,
            ws.max_count,
            ws.fetch_max_chars,
            ws.timeout_secs,
        );
        Some(std::sync::Arc::new(svc))
    } else {
        None
    };

    // Gateway + web UI on one port.
    let gw = Gateway {
        state: state.clone(),
        runtime_url: cfg.runtime_url.clone(),
        runtime_api_key: cfg.runtime_api_key.clone(),
        runtime_num_predict: cfg.runtime_num_predict,
        spec_decode: cfg.runtime_spec_decode.clone(),
        spec_lane: std::sync::Arc::new(jerith_mesh_core::gateway::SpecLane::default()),
        miner,
        mesh_fallback,
        guard,
        // The gateway must honor what setup/config wrote for the tier
        // routing (previously it silently used the router defaults and
        // ignored `runtime_vision_model` / `runtime_deep_model`).
        model_router: Some(std::sync::Arc::new(
            jerith_mesh_core::model_router::RouterModels::from_config(&cfg),
        )),
        base_model: cfg.model.as_ref().map(|m| m.id.clone()),
        web,
        chain_url: Some(cfg.chain_url.clone()),
        // Per-node JER wallet card (Phase 4): present on JerithChain nodes;
        // shows config `wallet_address`'s balance or the miner's default
        // credit identity (the node id). /status probes it live.
        wallet: {
            // Operator chain base (chain_url → co-hosted :8322) so the
            // card probes the chain this node actually mines on.
            let card = jerith_mesh_core::economics::WalletCard::from_crypto(
                &cfg.crypto,
                &cfg.node_id,
                Some(&chain_base(Some(&cfg))),
            )
            .map(std::sync::Arc::new);
            if let Some(w) = &card {
                match jerith_mesh_core::economics::wallet_balance(w).await {
                    Ok(b) if b.exists => {
                        println!("   wallet      {} · {:.6} JER", w.identity, b.balance_jer)
                    }
                    Ok(_) => println!("   wallet      {} · no wallet on the chain yet", w.identity),
                    Err(e) => println!("   wallet      unreachable: {e:#}"),
                }
            }
            card
        },
        // Optional-chain wallet card (§9, Phase 4): probes the operator's
        // OWN chain node for a live balance (bitcoin/ethereum/dogecoin/
        // monero), the same chain-truthful contract as the JER card;
        // mining on these chains stays honestly refused via the note.
        optional_wallet: {
            let card = jerith_mesh_core::economics::OptionalChainWallet::from_crypto(&cfg.crypto)
                .map(std::sync::Arc::new);
            if let Some(w) = &card {
                match jerith_mesh_core::economics::optional_chain_probe(w).await {
                    Ok(b) => println!(
                        "   wallet(oc)  {} · {:.8} {}",
                        b.address.as_deref().unwrap_or("wallet"),
                        b.amount,
                        b.unit
                    ),
                    Err(e) => println!("   wallet(oc)  {} · ({e:#})", w.unit()),
                }
            }
            card
        },
        metrics: std::sync::Arc::new(jerith_mesh_core::metrics::Metrics::new()),
    };
    // Mesh coordinator (FEATURES — named worker registry + hand-off
    // channel + shared memory pool): the store lives beside the memory
    // db; `[coordinator] enabled` (default true) gates the HTTP surface,
    // which merges into the SAME port the gateway answers on — one
    // address for any agent process on the mesh to speak to.
    let mut app = axum::Router::new()
        .merge(gw.router())
        .merge(web::router(state.clone()));
    if cfg.coordinator.enabled {
        match jerith_mesh_core::coordinator::CoordinatorStore::open(
            &jerith_mesh_core::config::NodeConfig::data_dir().join("coordinator.db"),
        ) {
            Ok(store) => {
                app = app.merge(jerith_mesh_core::coordinator::router(std::sync::Arc::new(
                    store,
                )));
                tracing::info!(target: "gateway", "coordinator surface on (/v1/coordinator/*)");
            }
            Err(e) => tracing::warn!(
                target: "gateway",
                error = %e,
                "coordinator store unavailable — surface off this boot"
            ),
        }
    }

    let addr = format!("{}:{}", cfg.effective_gateway_bind(), cfg.gateway_port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    // Phase 1 + Phase 2: QUIC NodeRPC server. Peers dispatch expert
    // queries here. The server comes up when this node can serve *either*
    // chat (a reachable runtime on the mesh model) *or* real expert
    // tensors (a populated expert store) — a node that pools expert
    // weights serves them with no LLM runtime installed at all.
    let quic_port = cfg.effective_rpc_port();
    let rpc_addr: std::net::SocketAddr = format!("0.0.0.0:{quic_port}").parse()?;
    let model_name = cfg.model.as_ref().map(|m| m.id.clone()).unwrap_or_default();
    let runtime = match jerith_mesh_core::quic::RuntimeExpert::discover(
        &cfg.runtime_url,
        &model_name,
        cfg.runtime_api_key.as_deref(),
    )
    .await
    {
        Ok(mut handler) => {
            // Tier routing for peer-dispatched chat (same policy as the
            // gateway): tiers come from config; an empty string disables one.
            handler.vision_model = cfg.runtime_vision_model.clone();
            handler.deep_model = cfg.runtime_deep_model.clone();
            Some(std::sync::Arc::new(handler))
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                "runtime not servable — chat + legacy expert stand-in disabled"
            );
            None
        }
    };
    let hosting_experts = expert_store
        .as_ref()
        .map(|s| !s.hosted().is_empty())
        .unwrap_or(false);
    if runtime.is_some() || hosting_experts {
        let serving = jerith_mesh_core::quic::ExpertServing {
            store: expert_store.clone(),
            runtime,
            tensors_only: cfg.experts.tensors_only,
            // Demand signal: every expert this node actually runs feeds the
            // hot set that drives replication.
            stats: Some(state.expert_stats.clone()),
            // Kill switch: peer-facing serving closes when the governor trips.
            isolate: Some(state.isolate.clone()),
            // Phase 3 speculative prefetch: next-layer demand-ranked locals
            // preload into the LRU after every real tensor serve.
            prefetch: Some(state.clone()),
        };
        let banner = serving.describe();
        // Identity-bound QUIC cert: the transport cert key IS the node's
        // Ed25519 identity, letting peers pin node_id ↔ cert.
        let rpc_identity = cfg.identity()?;
        match jerith_mesh_core::quic::server_endpoint_for(&rpc_identity, rpc_addr) {
            Ok(endpoint) => {
                let (_rpc_tx, rpc_rx) = tokio::sync::watch::channel(false);
                let node_id = cfg.node_id.clone();
                tokio::spawn(async move {
                    let handler_handler: std::sync::Arc<dyn jerith_mesh_core::quic::ExpertHandler> =
                        std::sync::Arc::new(serving);
                    if let Err(e) = jerith_mesh_core::quic::serve_with_peer_block(
                        endpoint,
                        handler_handler,
                        node_id,
                        rpc_rx,
                        state.peer_block.clone(),
                        // Phase 5: the stake escrow join gate — while
                        // required_uj > 0 and the ledger doesn't clear,
                        // work-carrying RPC gets the honest typed refusal
                        // (pings keep flowing; see stake.rs §5).
                        Some(state.stake.clone()),
                    )
                    .await
                    {
                        tracing::error!(error = %e, "rpc server died");
                    }
                });
                println!(
                    "   node-rpc    quic://0.0.0.0:{quic_port} (expert dispatch: {banner}, model: {model_name})"
                );
            }
            Err(e) => tracing::warn!(error = %e, "quic rpc server unavailable"),
        }
    } else {
        println!(
            "   node-rpc    disabled (no runtime at {} and no hosted experts)",
            cfg.runtime_url
        );
    }

    println!("\nReady: web UI http://{addr}/ · OpenAI API http://{addr}/v1\nCtrl+C to stop.");
    // Connect info required for the gateway per-IP rate limiter's socket
    // fallback (headers take precedence when present, e.g. behind a proxy).
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .await?;
    Ok(())
}

fn role_label(r: NodeRole) -> &'static str {
    match r {
        NodeRole::Contributor => "contributor",
        NodeRole::Requester => "requester",
        NodeRole::Hybrid => "hybrid",
    }
}

fn crypto_label(c: jerith_mesh_core::config::CryptoChain) -> &'static str {
    use jerith_mesh_core::config::CryptoChain as C;
    match c {
        C::Jerithchain => "JerithChain (JER)",
        C::Bitcoin => "Bitcoin (user-managed)",
        C::Ethereum => "Ethereum (user-managed)",
        C::Dogecoin => "Dogecoin (user-managed)",
        C::Monero => "Monero (user-managed)",
        C::None => "off",
    }
}

// ---------------------------------------------------------------------------
// status / peers / model / config
// ---------------------------------------------------------------------------

async fn status_cmd() -> anyhow::Result<()> {
    let cfg = NodeConfig::load()?.context("no config — run setup first")?;
    let hw_report = hw::probe();
    let out = json!({
        "config": {
            "node_id": cfg.node_id,
            "model": cfg.model.as_ref().map(|m| m.id.clone()),
            "quant": cfg.model.as_ref().map(|m| m.quant.clone()),
            "role": role_label(cfg.role),
            "contribution_percent": cfg.contribution_percent,
            "memory_backend": cfg.memory_backend.label(),
        },
        "hardware": {
            "os_release": hw_report.os_release,
            "cpu": hw_report.cpu_model,
            "logical_cores": hw_report.cpu_logical_cores,
            "ram_total": hw_report.ram_total_bytes,
            "ram_available": hw_report.ram_available_bytes,
            "gpus": hw_report.gpus.len(),
        },
    });
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

async fn peers_cmd() -> anyhow::Result<()> {
    let cfg = NodeConfig::load()?.context("no config — run setup first")?;
    let model = cfg
        .model
        .clone()
        .context("no model configured — run setup")?;
    // Prefer the LOCAL node's live roster (ping-verified liveness + RTT +
    // hosted experts from /status); the bootstrap advertisement list is
    // the fallback for remote-node operators (or a node that is down —
    // its ad view is TTL-based, not ping-verified).
    let gateway = format!("http://127.0.0.1:{}", cfg.gateway_port);
    let resp = jerith_mesh_core::http::mesh_http()
        .get(format!("{gateway}/status"))
        .timeout(std::time::Duration::from_secs(4))
        .send()
        .await;
    let resp_ok = matches!(&resp, Ok(r) if r.status().is_success());
    if resp_ok {
        let resp = resp.unwrap();
        if let Ok(status) = resp.json::<serde_json::Value>().await {
            // No `peers` key = an older gateway binary without the
            // roster — fall through to the bootstrap advertisement view.
            let Some(peers) = status.get("peers").and_then(|p| p.as_array()).cloned() else {
                return bootstrap_peer_view(cfg, &model).await;
            };
            if peers.is_empty() {
                println!(
                    "No peers for {} {} (this node's view)",
                    model.id, model.quant
                );
                return Ok(());
            }
            println!("this node's roster (ping-verified): {gateway}/status");
            println!(
                "{:<22} {:<10} {:>13} {:<20} {:>4}  hosted",
                "node", "state", "rtt_ms", "model", "contrib"
            );
            for p in &peers {
                // Raw + EWMA-smoothed RTT. When they differ, show both:
                // the smoothed value is what placement orders by (an EWMA,
                // so one noisy ping cannot misroute mesh work), the raw is
                // the last single sample.
                let raw = p["last_rtt_ms"].as_u64();
                let smoothed = p["smoothed_rtt_ms"].as_u64();
                let rtt = match (raw, smoothed) {
                    (Some(r), Some(s)) if r != s => format!("{r} (sm {s})"),
                    (Some(r), _) => r.to_string(),
                    (None, Some(s)) => format!("sm {s}"),
                    (None, None) => "-".into(),
                };
                let hosted = p["hosted_experts"]
                    .as_array()
                    .map(|a| {
                        if a.is_empty() {
                            "-".into()
                        } else {
                            a.iter()
                                .filter_map(|e| e.as_str())
                                .collect::<Vec<_>>()
                                .join(",")
                        }
                    })
                    .unwrap_or_else(|| "-".into());
                println!(
                    "{:<22} {:<10} {:>13} {:<20} {:>3}%  {}",
                    p["node_id"].as_str().unwrap_or("?"),
                    p["state"].as_str().unwrap_or("?"),
                    rtt,
                    p["model"].as_str().unwrap_or("?"),
                    p["contribution_percent"].as_u64().unwrap_or(0),
                    hosted
                );
            }
            return Ok(());
        }
    }
    bootstrap_peer_view(cfg, &model).await
}

/// The bootstrap advertisement view: who is ADVERTISED in the model's
/// mesh (TTL-based, not ping-verified). The fallback for operators of a
/// remote node, or when the local gateway is down.
async fn bootstrap_peer_view(cfg: NodeConfig, model: &ModelId) -> anyhow::Result<()> {
    let client = BootstrapClient::new(&cfg.bootstrap_url);
    match client.fetch_peers(&model.id).await {
        Ok(peers) => {
            let same: Vec<_> = peers
                .into_iter()
                .filter(|p| p.model_id == model.id && p.quant == model.quant)
                .collect();
            if same.is_empty() {
                println!("No advertised peers for {} {}", model.id, model.quant);
            } else {
                println!(
                    "bootstrap advertisement view: {} (TTL-based, not ping-verified)",
                    cfg.bootstrap_url
                );
                println!(
                    "{:<22} {:<24} {:<8} {:>6}  role",
                    "node", "model", "quant", "contrib"
                );
                for p in same {
                    println!(
                        "{:<22} {:<24} {:<8} {:>4}%  {}",
                        p.node_id,
                        p.model_id,
                        p.quant,
                        p.contribution_percent,
                        role_label(p.role)
                    );
                }
            }
        }
        Err(e) => println!("bootstrap unreachable: {e}"),
    }
    Ok(())
}

async fn model_cmd(new_model: Option<String>) -> anyhow::Result<()> {
    let mut cfg = NodeConfig::load()?.context("no config — run setup first")?;
    match new_model {
        None => match &cfg.model {
            Some(m) => println!("{} ({}) — mesh:  peers in model mesh", m.id, m.quant),
            None => println!("no model set"),
        },
        Some(spec) => {
            let mid = parse_model_spec(&spec)?;
            let from = cfg.model.as_ref().map(|m| m.id.clone());
            println!(
                "Switching model {} → {}. Disconnecting all peers (model change forces a clean mesh exit)…",
                from.as_deref().unwrap_or("<none>"),
                mid.id
            );
            cfg.set_model(mid);
            cfg.save()?;
            println!("Saved. Restart `jerith run` to join the new model's mesh.");
        }
    }
    Ok(())
}

async fn config_cmd(assignments: Vec<String>) -> anyhow::Result<()> {
    let mut cfg = NodeConfig::load()?.context("no config — run setup first")?;
    // Documented shorthand: `jerith config set key=value …` is accepted
    // alongside the raw `jerith config key=value` form (the `set` token
    // is a no-op filter), so guides that say "config set" are true.
    let assignments: Vec<String> = assignments.into_iter().filter(|a| a != "set").collect();
    // Memory backend changes print store-migration guidance; capture
    // the pre-change state for the diff.
    let old = (
        cfg.memory_backend,
        cfg.memory_dsn.clone(),
        cfg.memory_max_rows,
    );
    if assignments.is_empty() {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "node_id": cfg.node_id,
                "model": cfg.model.as_ref().map(|m| m.id.clone()),
                "contrib_percent": cfg.contribution_percent,
                "gateway_port": cfg.gateway_port,
                "bootstrap_url": cfg.bootstrap_url,
                "crypto_chain": crypto_label(cfg.crypto.chain),
            }))?
        );
        return Ok(());
    }
    for a in assignments {
        let (k, v) = a.split_once('=').context("expected key=value")?;
        match k {
            "contribution_percent" => {
                let v: u64 = v.parse().context("contribution_percent must be a number")?;
                if v < MIN_CONTRIBUTION_PERCENT {
                    bail!("minimum is {MIN_CONTRIBUTION_PERCENT}%");
                }
                cfg.contribution_percent = v;
            }
            "gateway_port" => cfg.gateway_port = v.parse().context("bad port")?,
            "rpc_port" => {
                if v.eq_ignore_ascii_case("none") || v == "0" {
                    cfg.rpc_port = None;
                } else {
                    cfg.rpc_port = Some(v.parse().context("bad rpc_port")?);
                }
            }
            "runtime_url" => cfg.runtime_url = v.to_string(),
            "runtime_api_key" => cfg.runtime_api_key = Some(v.to_string()),
            "runtime_deep_model" | "runtime_vision_model" | "runtime_tool_model" => {
                let val = if v.eq_ignore_ascii_case("none") {
                    None
                } else {
                    Some(v.to_string())
                };
                match k {
                    "runtime_deep_model" => cfg.runtime_deep_model = val,
                    "runtime_vision_model" => cfg.runtime_vision_model = val,
                    _ => cfg.runtime_tool_model = val,
                }
            }
            "runtime_num_predict" => {
                cfg.runtime_num_predict = if v.eq_ignore_ascii_case("none") || v == "0" {
                    None
                } else {
                    Some(v.parse().context("runtime_num_predict must be a number")?)
                }
            }
            "embedding_model" => cfg.specialists.embedding_model = v.to_string(),
            "bootstrap_url" => cfg.bootstrap_url = v.to_string(),
            "memory_backend" => {
                cfg.memory_backend = parse_memory(v)?;
            }
            "memory_dsn" => {
                cfg.memory_dsn = if v.trim().is_empty() {
                    None
                } else {
                    Some(v.to_string())
                };
            }
            "memory_max_rows" => {
                let n: i64 = v
                    .parse()
                    .context("memory_max_rows must be a whole number")?;
                anyhow::ensure!(n >= 0, "memory_max_rows must be ≥ 0 (0 = no retention cap)");
                cfg.memory_max_rows = n;
            }
            other => bail!("unknown config key {other:?}"),
        }
    }
    // Final-state validation — assignments can fix each other up (e.g.
    // `memory_backend=postgres memory_dsn=...` in one command), so this
    // runs after the loop, never per-key: a postgres backend without a
    // DSN anywhere is refused, not saved.
    if cfg.memory_backend == MemoryBackend::PostgresPgvector && cfg.effective_memory_dsn().is_none()
    {
        bail!(
            "memory_backend = postgres but no DSN — add memory_dsn=postgres://user:pass@host:5432/db \
             to the same command (or set JERITH_MEMORY_DSN); refusing a backend the node could never open"
        );
    }
    let memory_changed = cfg.memory_backend != old.0 || cfg.memory_dsn != old.1;
    let cap_changed = cfg.memory_max_rows != old.2;
    cfg.save()?;
    println!("saved {}", NodeConfig::config_path().display());
    if cap_changed && !memory_changed {
        // A retention-cap change is not a backend change: it live-swaps
        // within one ping tick (the same memory-triple reload class), and
        // pruning follows the NEW cap on the next insert — rows are never
        // deleted for the old cap at swap time.
        let cap = if cfg.memory_max_rows > 0 {
            format!("{} rows", cfg.memory_max_rows)
        } else {
            "UNCAPPED (0)".to_string()
        };
        println!(
            "memory retention cap now {cap}. Applies within one ping tick \
             (store live-swap); pruning follows the new cap from the next insert — \
             nothing is deleted at swap time.\
             {}",
            if cfg.memory_max_rows == 0 && cfg.memory_backend == MemoryBackend::Sqlite {
                "\nNOTE: SQLite has no native vector index — search is a full exact scan, \
                 linear in rows."
            } else {
                ""
            }
        );
    }
    if memory_changed {
        // A live node applies this within one ping tick (the mesh loop
        // hot-reloads the config file); existing rows are not auto-moved
        // — see the guidance, printed here.
        println!(
            "{}",
            jerith_mesh_core::memory::migration_guidance(old.0.label(), cfg.memory_backend.label())
        );
    }
    Ok(())
}

/// `jerith chain …` — the read-only chain introspection commands.
#[cfg(test)]
mod chain_tests {
    use super::*;
    use rusqlite::Connection;

    #[test]
    fn chain_wallet_rows_join_and_sort() {
        let dir = std::env::temp_dir().join(format!("chain-cl-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let db = dir.join("jerith.db");
        let conn = Connection::open(&db).unwrap();
        conn.execute_batch(
            r#"
            CREATE TABLE balances (address TEXT PRIMARY KEY, balance INTEGER NOT NULL, nonce INTEGER NOT NULL, locked INTEGER NOT NULL);
            CREATE TABLE users (discord_id TEXT, address TEXT UNIQUE, vk_hex TEXT, enc_sk TEXT, wrapped_sk TEXT, password_stored INTEGER, totp_enc TEXT, totp_confirmed INTEGER, auto_sign_limit INTEGER, daily_spent INTEGER);
            CREATE TABLE mining_state (discord_id TEXT, last_success INTEGER, last_fail INTEGER, success_count INTEGER, fail_count INTEGER, total_mined INTEGER);
            INSERT INTO balances VALUES ('jer1aaa', 2500000, 4, 1000000);
            INSERT INTO balances VALUES ('jer1bbb', 12000000, 1, 0);
            INSERT INTO users (discord_id, address) VALUES ('1001', 'jer1bbb');
            INSERT INTO mining_state (discord_id, success_count, fail_count, total_mined) VALUES ('1001', 12, 1, 3000000);
            "#,
        )
        .unwrap();
        // Address-sort (default) LIMIT 1 keeps the lexicographically first.
        let rows = load_chain_wallets(&conn, 10, false).unwrap();
        assert_eq!(rows[0].address, "jer1aaa");
        // Balance-sort brings the bigger wallet first, with the joined
        // discord id + mined totals.
        let rows = load_chain_wallets(&conn, 10, true).unwrap();
        assert_eq!(rows[0].address, "jer1bbb");
        assert_eq!(rows[0].discord_id, Some(1001));
        assert_eq!(rows[0].total_mined_uj, 3_000_000);
        assert_eq!(rows[1].discord_id, None);
        // Limit applies.
        assert_eq!(load_chain_wallets(&conn, 1, true).unwrap().len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod trust_tests {
    use super::*;
    use std::path::Path;

    const KEY_A: &str = concat!(
        "aa11bb22", "cc33dd44", "ee55ff66", "00112233", "44556677", "8899aabb", "ccddeeff",
        "00001111"
    );
    const KEY_B: &str = concat!(
        "b10b0b0b", "c2c3c4c5", "d6d7d8d9", "e0e1e2e3", "f4f5f6f7", "08090a0b", "0c0d0e0f",
        "1a2b3c4d"
    );

    /// A lived-in file: comments, an existing enabled row, one pin.
    /// The pin is KEY_A in upper form — parse lowercases it back.
    fn comment_file() -> String {
        format!(
            "# my bot.toml — keep these comments intact\n\
             [plugins]\n\
             enabled = [\"websearch\", \"memory\"]\n\
             \n\
             [plugins.trusted_publishers]\n\
             \"weather\" = \"{}\"\n",
            KEY_A.to_uppercase()
        )
    }

    /// `[plugins].enabled` membership — the BOT's exact semantics
    /// (serde rows, no trimming): a space-padded row is an operator
    /// typo the bot itself would fail loud on, so the install guidance
    /// must not pretend it satisfies the gate.
    #[test]
    fn enabled_gate_membership() {
        let doc = comment_file();
        assert!(!parse_enabled_gate(&doc, "weather").unwrap());
        // In-tree plugins ride the enabled list as bare ids — the
        // external: prefix is drop-only, so a bare id never matches.
        assert!(!parse_enabled_gate(&doc, "websearch").unwrap());
        assert!(!parse_enabled_gate(&doc, "memory").unwrap());
        let doc2 = "[plugins]\nenabled = [\" external:weather \", \"websearch\"]\n".to_string();
        assert!(!parse_enabled_gate(&doc2, "weather").unwrap());
        let doc3 = "[plugins]\nenabled = [\"external:weatherx\"]\n";
        assert!(!parse_enabled_gate(doc3, "weather").unwrap());
        assert!(!parse_enabled_gate("", "weather").unwrap());
        assert!(
            !parse_enabled_gate("[plugins]\nowner_only = [\"external:pet\"]\n", "pet").unwrap(),
            "owner_only rows never satisfy the enabled gate"
        );
    }

    fn temp_file(name: &str, text: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("trust-t-{}-{}", name, std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join(name);
        std::fs::write(&path, text).unwrap();
        path
    }

    #[test]
    fn trusted_publishers_parse_normalizes_and_refuses_junk() {
        let pins = parse_trusted_publishers(&comment_file()).unwrap();
        assert_eq!(pins.get("weather").unwrap(), KEY_A); // lowercased
        assert_eq!(pins.len(), 1);

        // No section = an empty map (first-party boxes have no pin row).
        let empty = parse_trusted_publishers("[plugins]\nenabled = [\"x\"]\n").unwrap();
        assert!(empty.is_empty());

        // Not a table of rows (a string where the table belongs).
        assert!(parse_trusted_publishers("[plugins]\ntrusted_publishers = \"x\"\n").is_err());
        // Wrong length → loud, load-bearing row named.
        let short: String = KEY_A.chars().take(63).collect();
        assert!(
            parse_trusted_publishers(&format!(
                "[plugins.trusted_publishers]\n\"w\" = \"{short}\"\n"
            ))
            .is_err()
        );
        // Non-string row.
        assert!(parse_trusted_publishers("[plugins.trusted_publishers]\nw = 7\n").is_err());
    }

    #[test]
    fn rotate_rewrites_pin_and_preserves_the_rest_of_the_file() {
        let path = temp_file("rotate", &comment_file());
        write_trusted_publishers(&path, "weather", KEY_B).unwrap();
        write_trusted_publishers(&path, "ledger", KEY_A).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let pins = parse_trusted_publishers(&text).unwrap();
        assert_eq!(pins.get("weather").unwrap(), KEY_B);
        assert_eq!(pins.get("ledger").unwrap(), KEY_A);
        // The operator's edits outside the trust table survive verbatim.
        assert!(text.contains("# my bot.toml — keep these comments intact"));
        assert!(text.contains("enabled = [\"websearch\", \"memory\"]"));
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn rotate_creates_a_fresh_trust_file() {
        let path = temp_file("fresh", "");
        std::fs::remove_file(&path).unwrap();
        write_trusted_publishers(&path, "weather", KEY_A).unwrap();
        let pins = parse_trusted_publishers(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(pins.get("weather").unwrap(), KEY_A);
        assert_eq!(pins.len(), 1);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn rotate_survives_an_inline_plugins_table() {
        let path = temp_file("inline", "plugins = { enabled = [\"x\"] }\n");
        write_trusted_publishers(&path, "weather", KEY_A).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("enabled = [\"x\"]"));
        let pins = parse_trusted_publishers(&text).unwrap();
        assert_eq!(pins.get("weather").unwrap(), KEY_A);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    // Path import is used by the helpers' signatures, not by these
    // tests directly; keep the `use` from warning.
    #[test]
    fn path_alias_is_borrowed() {
        fn accepts(_p: &Path) {}
        accepts(Path::new("."));
    }

    // Already-done honesty for the search/trust next-step hints:
    // helpers feed PRINTFS, so their verdicts are pinned the way the
    // CLI prints them.
    #[test]
    fn search_row_state_honors_the_rows_own_version() {
        let have = |ins: &[&str], row: &str| {
            let sorted: Vec<String> = ins.iter().map(|s| s.to_string()).collect();
            // installed_versions_under sorts; the helper leans on it.
            let mut s = sorted;
            s.sort();
            search_row_state(row, &s)
        };
        // Exact version on disk → satisfied row, NOT installable.
        let (tail, installable) = have(&["0.1.3", "0.1.4"], "0.1.4");
        assert!(!installable);
        assert!(tail.contains("installed 0.1.4"), "{tail}");
        // Upgrade row (latest on disk is OLDER) — must not read as
        // satisfied; the tag names both sides honestly.
        let (tail, installable) = have(&["0.1.3"], "0.1.4");
        assert!(installable);
        assert!(
            tail.contains("0.1.3 installed") && tail.contains("this row (0.1.4) is not"),
            "{tail}"
        );
        // Nothing installed at all → silent tag, installable.
        let (tail, installable) = have(&[], "0.1.4");
        assert!(installable);
        assert!(tail.is_empty(), "{tail}");
        // Downgrade-style row: newer already installed, row version
        // missing — still installable, tag doesn't claim it's installed.
        let (tail, installable) = have(&["0.2.0"], "0.1.3");
        assert!(installable);
        assert!(tail.contains("0.2.0 installed"), "{tail}");
    }

    #[test]
    fn pins_header_says_rotate_migration_is_done_when_all_mirrored() {
        let mut bot = std::collections::BTreeMap::new();
        bot.insert("weather".into(), KEY_A.to_string());
        let mut pins = std::collections::BTreeMap::new();
        // Same key in a different case still counts as mirrored.
        pins.insert("weather".into(), KEY_A.to_uppercase());
        assert_eq!(
            trust_pins_file_header(&bot, &pins),
            "legacy file — every row mirrored in bot.toml, nothing left for a \
             rotate to migrate"
        );
        // One unmirrored row → the migrate advice stays.
        pins.insert("ledger".into(), KEY_B.to_string());
        assert_eq!(
            trust_pins_file_header(&bot, &pins),
            "legacy — a bot.toml row OVERRANKS it; a rotate migrates"
        );
        // Empty/absent pins file → the nothing-to-migrate wording, and
        // a dropped bot.toml row (pin gone) is NOT mirrored.
        assert_eq!(
            trust_pins_file_header(&bot, &std::collections::BTreeMap::new()),
            "empty — no legacy rows to migrate; rotate pins land only in \
             bot.toml"
        );
        let mut stale = std::collections::BTreeMap::new();
        stale.insert("weather".into(), KEY_A.to_string());
        assert_eq!(
            trust_pins_file_header(&std::collections::BTreeMap::new(), &stale),
            "legacy — a bot.toml row OVERRANKS it; a rotate migrates"
        );
    }
}
