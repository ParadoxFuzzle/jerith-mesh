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
    Search { query: String, #[arg(long, default_value_t = 5)] top_k: usize },
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
        Cmd::Stake { cmd } => stake_cmd(cmd).await,
        Cmd::Chain { cmd } => chain_cmd(cmd).await,
        Cmd::Lore { query, top_k } => lore_cmd(&query, top_k).await,
    }
}

/// `jerith lore <query>` — sourced canon answers from the lore vault
/// (void-lore-index port; vault default is the OpenClaw borrow).
async fn lore_cmd(query: &str, top_k: usize) -> anyhow::Result<()> {
    let vault = jerith_mesh_core::lore::default_vault_dir().context(
        "no lore vault — set $JERITH_LORE_VAULT to a directory of .md files",
    )?;
    let index = jerith_mesh_core::lore::build(&vault)?;
    let hits = index.search(query, top_k);
    if hits.is_empty() {
        println!("no lore matched \"{query}\" ({} entries indexed)", index.len());
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
        Some(c) if !c.chain_url.trim().is_empty() => c.chain_url.trim().trim_end_matches('/').to_string(),
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
async fn chain_get(_cfg: Option<&NodeConfig>, base: &str, path: &str, query: &[(&str, &str)]) -> anyhow::Result<serde_json::Value> {
    let token = jerith_mesh_core::economics::read_token().context(
        "no chain api token — stage ~/.jerith-mesh/keys/api.token (see docs/RUNBOOK.md) \\
         or set $JERITH_TOKEN",
    )?;
    let client = reqwest::Client::builder()
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
    let body: serde_json::Value = resp
        .json()
        .await
        .context("chain node returned non-JSON")?;
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
                discord_id: discord_id
                    .as_deref()
                    .and_then(|s| s.parse::<u64>().ok()),
                total_mined_uj,
            })
        })?
        .collect::<Result<_, _>>()?;
    if by_balance {
        rows.sort_by(|a, b| b.balance_uj.cmp(&a.balance_uj));
    } else {
        rows.sort_by(|a, b| a.address.cmp(&b.address));
    }
    rows.truncate((limit.max(1) as usize).min(500));
    Ok(rows)
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
                db.display()
                    .to_string()
                    .replace('?', "%3F")
            );
            let conn = rusqlite::Connection::open_with_flags(
                &uri,
                rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY
                    | rusqlite::OpenFlags::SQLITE_OPEN_URI,
            )
            .with_context(|| format!("opening chain ledger {db:?}"))?;
            let rows = load_chain_wallets(&conn, limit, by_balance)?;
            println!(
                "{:<52} {:>14} {:>12} {:>12} {:>12} {}",
                "address", "balance JER", "locked", "nonce", "mined", "discord"
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
        bail!("staking rides jerithchain only — set crypto = \"jerithchain\" (and JERITH_TOKEN or keys/api.token) first");
    }
    let identity = cfg.identity()?;
    let stake =
        jerith_mesh_core::stake::StakeManager::new(&cfg.crypto, Some(identity));
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
        miner,
        memory: memory.clone(),
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
    match jerith_mesh_core::quic::RuntimeExpert::discover(
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
            "runtime_api_key" => cfg.runtime_api_key = Some(v.to_string()),
            "bootstrap_url" => cfg.bootstrap_url = v.to_string(),
            other => bail!("unknown config key {other:?}"),
        }
    }
    cfg.save()?;
    println!("saved {}", NodeConfig::config_path().display());
    Ok(())
}
