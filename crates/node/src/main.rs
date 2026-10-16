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
use std::path::PathBuf;
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
            jerith_mesh_core::bootstrap_server::serve(
                jerith_mesh_core::bootstrap_server::BootstrapOptions {
                    port,
                    advertise_ttl_ms: ttl_ms,
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
    let client = reqwest::Client::builder()
        .use_rustls_tls()
        .timeout(Duration::from_secs(15))
        .build()?;
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
    let resp = client
        .get(&url)
        .bearer_auth(token)
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
    let stake = jerith_mesh_core::stake::StakeManager::new(&cfg.crypto, Some(identity));
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
        "[3/5b] Local skills: vision {}, embeddings {} | tiers: thinking {}, vision {}, embedding {} (excluded from the contribution accounting)",
        on_off(vision_enabled),
        on_off(embeddings_enabled),
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
        for pick in [&deep_pick, &vision_pick, &embedding_pick] {
            if !pick.is_empty() {
                println!("   $ ollama pull {pick}");
            }
        }
    } else if ollama_ready {
        for (tier, pick) in [
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
    // must show a chain balance ≥ the requirement, or it runs this session
    // demoted to Requester (can consume compute, never offered work). Config
    // is not modified — the node self-restores its role on the next start.
    if cfg.crypto.stake.required_uj > 0 && cfg.role != NodeRole::Requester {
        let rpc = jerith_mesh_core::economics::chain_rpc_url(&cfg.crypto);
        match jerith_mesh_core::economics::chain_balance_uj(&rpc, &cfg.node_id).await {
            Ok(balance) if balance >= cfg.crypto.stake.required_uj => {
                println!(
                    "   stake       {} µJER locked (required {})",
                    balance, cfg.crypto.stake.required_uj
                );
            }
            Ok(balance) => {
                println!(
                    "   stake       XXX insufficient: {} µJER < required {} — demoting to requester for this session",
                    balance, cfg.crypto.stake.required_uj
                );
                cfg.role = NodeRole::Requester;
            }
            Err(e) => {
                println!(
                    "   stake       chain balance unavailable ({e}) — demoting to requester for this session"
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
        None => println!("   economics   off"),
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
    let _loop_handle =
        jerith_mesh_core::ping::MeshLoop::spawn(state.clone(), Duration::from_secs(10));

    // Mesh fallback for chat: a QUIC client endpoint + router, wired into
    // the gateway so /v1/chat/completions can fail over to runtime-backed
    // peers when the local runtime is unavailable.
    let mesh_fallback = match jerith_mesh_core::ping::MeshFallbackChat::spawn_with(
        state.verify.clone(),
        // Phase 4 verification policy: replay probability from config.
        cfg.verify_percent,
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
        miner,
        mesh_fallback,
        guard,
        model_router: Some(std::sync::Arc::new(
            jerith_mesh_core::model_router::RouterModels::default(),
        )),
        base_model: cfg.model.as_ref().map(|m| m.id.clone()),
        web,
        chain_url: Some(cfg.chain_url.clone()),
        metrics: std::sync::Arc::new(jerith_mesh_core::metrics::Metrics::new()),
    };
    let app = axum::Router::new()
        .merge(gw.router())
        .merge(web::router(state.clone()));

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
    let Some(model) = cfg.model.as_ref() else {
        bail!("no model configured — run setup");
    };
    let client = BootstrapClient::new(&cfg.bootstrap_url);
    match client.fetch_peers(&model.id).await {
        Ok(peers) => {
            let same: Vec<_> = peers
                .into_iter()
                .filter(|p| p.model_id == model.id && p.quant == model.quant)
                .collect();
            if same.is_empty() {
                println!("No live peers for {} {}", model.id, model.quant);
            } else {
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
    // Memory backend changes print store-migration guidance; capture
    // the pre-change state for the diff.
    let old = (cfg.memory_backend, cfg.memory_dsn.clone());
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
            "runtime_deep_model" | "runtime_vision_model" => {
                let val = if v.eq_ignore_ascii_case("none") {
                    None
                } else {
                    Some(v.to_string())
                };
                if k == "runtime_deep_model" {
                    cfg.runtime_deep_model = val;
                } else {
                    cfg.runtime_vision_model = val;
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
    cfg.save()?;
    println!("saved {}", NodeConfig::config_path().display());
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
