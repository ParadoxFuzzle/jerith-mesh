//! `jerith` — the jerith-mesh node CLI.
//!
//! Commands:
//!   setup       guided onboarding (model → memory → contribution % → wallets)
//!   run         start the node (discovery + pings + gateway + web UI)
//!   status      machine + mesh status snapshot
//!   peers       list peers in this model's mesh
//!   model       show or switch the active model (disconnects all peers)
//!   config      get/set config values (contribution_percent, etc.)

use std::io::{BufRead, Write as _};
use std::time::Duration;

use anyhow::{bail, Context as _};
use clap::{Parser, Subcommand};
use jerith_mesh_core::bootstrap::BootstrapClient;
use jerith_mesh_core::config::{
    default_model, MemoryBackend, NodeConfig, NodeRole, SetupChoices, MIN_CONTRIBUTION_PERCENT,
};
use jerith_mesh_core::gateway::Gateway;
use jerith_mesh_core::hw;
use jerith_mesh_core::models::{detect_capable_models, recommended_model, ModelId, ModelSource};
use jerith_mesh_core::ping::MeshState;
use jerith_mesh_core::web;
use serde_json::json;

#[derive(Parser)]
#[command(
    name = "jerith",
    about = "jerith-mesh node — P2P compute mesh for MoE LLMs",
    version,
    propagate_version = true,
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Guided setup flow (interactive when on a TTY, flags otherwise).
    Setup {
        /// Model id (curated) e.g. glm-5.3, or "huggingface:<repo>",
        /// "ollama:<name>", "path:<file>".
        #[arg(long)]
        model: Option<String>,
        /// "sqlite" or "postgres"
        #[arg(long)]
        memory: Option<String>,
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
    },
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
}

#[derive(Subcommand)]
enum MemoryCmd {
    /// Embed + store a text entry.
    Add { text: String },
    /// Vector-search local memory.
    Search { query: String, #[arg(long, default_value_t = 5)] top_k: usize },
    /// Number of stored entries.
    Count,
    /// Delete an entry by id.
    Forget { id: i64 },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
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
        Cmd::Setup {
            model,
            memory,
            contribution,
            role,
            crypto,
            bootstrap,
            no_wait,
        } => setup_flow(
            model,
            memory,
            contribution,
            role,
            crypto,
            bootstrap,
            no_wait,
        ),
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
    }
}

async fn memory_cmd(cmd: MemoryCmd) -> anyhow::Result<()> {
    use jerith_mesh_core::memory::{EmbedConfig, MemoryStore};
    let cfg = NodeConfig::load()?.context("no config — run setup first")?;
    let embed_cfg = EmbedConfig {
        ollama_url: cfg.specialists.ollama_url.clone(),
        dim: cfg.specialists.embedding_dim as usize,
        ..Default::default()
    };
    let store = MemoryStore::open(&NodeConfig::memory_db_path(), embed_cfg)?;
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
    bail!(
        "unknown model {spec:?}. Use a curated id (glm-5.2, glm-5.3, glm-5.3-flash, \
         deepseek-v4-flash, deepseek-v4.1-flash, kimi-k3, qwen3.8-flash-next, \
         qwen-image-2.1) or prefix: huggingface:<repo>, ollama:<name>, path:<file>"
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

fn setup_flow(
    model: Option<String>,
    memory: Option<String>,
    contribution: Option<u64>,
    role: Option<String>,
    crypto: Option<String>,
    bootstrap: Option<String>,
    no_wait: bool,
) -> anyhow::Result<()> {
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
        let _ = bootstrap_counts;
        prompt("\nModel to run? (id, or huggingface:<repo> / ollama:<name> / path:<file>)", "recommendation")?
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
        if let Some(fit) = fits.iter().find(|f| f.model.id == mid.id) {
            if !fit.fits && !std::env::var("JERITH_OK_BEYOND_CAPACITY").is_ok() {
                bail!(
                    "this machine cannot run {id} locally: {reason}\n\
                     Passing capacity checks is required to prevent mesh leeching. \
                     Set JERITH_OK_BEYOND_CAPACITY=1 to override.",
                    id = mid.id,
                    reason = fit.reason
                );
            }
        }
        mid
    };
    println!("\n[2/5] Model: {} (quant {})", model.id, model.quant);

    // 3. Memory backend.
    let memory_spec = match memory {
        Some(m) => m,
        None if interactive => {
            prompt("\nMemory backend? (sqlite | postgres)", "sqlite")?
        }
        None => bail!("--memory is required in non-interactive mode"),
    };
    let memory_backend = parse_memory(&memory_spec)?;
    println!("[3/5] Memory: {} (strictly local)", memory_backend.label());

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
            let s = prompt("\nCrypto for earnings? (jerithchain | bitcoin | ethereum | dogecoin | monero | none)", "jerithchain")?;
            parse_crypto(&s)?
        }
        None => jerith_mesh_core::config::CryptoConfig::default(),
    };

    let choices = SetupChoices {
        model,
        memory_backend,
        contribution_percent: contribution_pct,
        role,
        crypto,
        specialists: Default::default(),
        bootstrap_url: bootstrap,
        gateway_port: NodeConfig::default().gateway_port,
        runtime_url: None,
    };

    let existing = NodeConfig::load()?;
    let cfg = choices.into_config(existing)?;
    cfg.save()?;
    // Touch the memory DB path so data dir exists.
    std::fs::create_dir_all(NodeConfig::data_dir())?;

    println!("\n[5/5] Done. Config: {}", NodeConfig::config_path().display());
    println!("\nNext: ./jerith run   (or set up wallets with `jerith config`)");

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
    let mut cfg = NodeConfig::load()?
        .context("no config found — run `jerith setup` first")?;
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
    println!("   memory      {}", cfg.memory_backend.label());
    println!(
        "   crypto      {}",
        crypto_label(cfg.crypto.chain)
    );
    println!("   gateway     http://127.0.0.1:{}", cfg.gateway_port);

    let (state, _health_watch) = MeshState::new(cfg.clone())?;

    // JER miner: active when crypto chain is JerithChain (spec default).
    let miner = jerith_mesh_core::economics::MinerConfig::from_crypto(
        &cfg.crypto,
        &cfg.node_id,
    )
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

    // Local memory (SQLite + local embeddings; strictly node-private).
    let memory = if cfg.specialists.embeddings_enabled {
        let embed_cfg = jerith_mesh_core::memory::EmbedConfig {
            ollama_url: cfg.specialists.ollama_url.clone(),
            dim: cfg.specialists.embedding_dim as usize,
            ..Default::default()
        };
        match jerith_mesh_core::memory::MemoryStore::open(&NodeConfig::memory_db_path(), embed_cfg)
        {
            Ok(store) => {
                println!(
                    "   memory      SQLite at {} (dim {}, local-only)",
                    NodeConfig::memory_db_path().display(),
                    cfg.specialists.embedding_dim
                );
                Some(std::sync::Arc::new(store))
            }
            Err(e) => {
                tracing::warn!(error = %e, "memory store unavailable");
                println!("   memory      unavailable ({e})");
                None
            }
        }
    } else {
        None
    };

    // Spawn the mesh loop (discovery + pings + meters).
    let _loop_handle = jerith_mesh_core::ping::MeshLoop::spawn(state.clone(), Duration::from_secs(10));

    // Mesh fallback for chat: a QUIC client endpoint + router, wired into
    // the gateway so /v1/chat/completions can fail over to runtime-backed
    // peers when the local runtime is unavailable.
    let mesh_fallback = match jerith_mesh_core::ping::MeshFallbackChat::spawn_with(
        state.verify.clone(),
        // Phase 4 verification policy: replay probability from config.
        cfg.verify_percent,
    ) {
        Ok(mf) => Some(mf),
        Err(e) => {
            tracing::warn!(error = %e, "mesh chat fallback unavailable");
            None
        }
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

    // Gateway + web UI on one port.
    let gw = Gateway {
        state: state.clone(),
        runtime_url: cfg.runtime_url.clone(),
        miner,
        memory: memory.clone(),
        mesh_fallback,
        guard,
    };
    let app = axum::Router::new()
        .merge(gw.router())
        .merge(web::router(state.clone()));

    let addr = format!("127.0.0.1:{}", cfg.gateway_port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    // Phase 1: QUIC NodeRPC server. Peers dispatch
    // expert queries here when the local runtime is reachable and serving
    // the mesh model; otherwise the node registers no RPC endpoint (it
    // stays discoverable but is not offered remote expert work).
    let quic_port = cfg.effective_rpc_port();
    let rpc_addr: std::net::SocketAddr = format!("0.0.0.0:{quic_port}").parse()?;
    let model_name = cfg
        .model
        .as_ref()
        .map(|m| m.id.clone())
        .unwrap_or_default();
    match jerith_mesh_core::quic::RuntimeExpert::discover(&cfg.runtime_url, &model_name).await {
        Ok(handler) => {
            // Identity-bound QUIC cert: the transport cert key IS the node's
            // Ed25519 identity, letting peers pin node_id ↔ cert.
            let rpc_identity = cfg.identity()?;
            match jerith_mesh_core::quic::server_endpoint_for(&rpc_identity, rpc_addr) {
                Ok(endpoint) => {
                    let (_rpc_tx, rpc_rx) = tokio::sync::watch::channel(false);
                    let node_id = cfg.node_id.clone();
                    tokio::spawn(async move {
                        let handler_handler: std::sync::Arc<dyn jerith_mesh_core::quic::ExpertHandler> =
                            std::sync::Arc::new(handler);
                        if let Err(e) = jerith_mesh_core::quic::serve(
                            endpoint,
                            handler_handler,
                            node_id,
                            rpc_rx,
                        )
                        .await
                        {
                            tracing::error!(error = %e, "rpc server died");
                        }
                    });
                    println!(
                        "   node-rpc    quic://0.0.0.0:{quic_port} (runtime expert dispatch, model: {model_name})"
                    );
                }
                Err(e) => tracing::warn!(error = %e, "quic rpc server unavailable"),
            }
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                "runtime not servable — remote expert dispatch disabled"
            );
            println!(
                "   node-rpc    disabled (runtime at {} is not serving this mesh model)",
                cfg.runtime_url
            );
        }
    }

    println!(
        "\nReady: web UI http://{addr}/ · OpenAI API http://{addr}/v1\nCtrl+C to stop."
    );
    axum::serve(listener, app).await?;
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
                    "{:<22} {:<24} {:<8} {:>6}  {}",
                    "node", "model", "quant", "contrib", "role"
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
        None => {
            match &cfg.model {
                Some(m) => println!("{} ({}) — mesh: {} peers in model mesh", m.id, m.quant, ""),
                None => println!("no model set"),
            }
        }
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
            "bootstrap_url" => cfg.bootstrap_url = v.to_string(),
            other => bail!("unknown config key {other:?}"),
        }
    }
    cfg.save()?;
    println!("saved {}", NodeConfig::config_path().display());
    Ok(())
}
