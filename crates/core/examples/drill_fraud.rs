//! Sandbox drill driver (§10 live drills) — NOT a production path.
//!
//! Drives the §10 loop against a REAL bootstrap and REAL chain from
//! scratch identities created for the drill (never a production node's
//! key): mint a drill identity, post a signed fraud report, or submit a
//! signed stake-slash notice the way `StakeManager::notice` does.
//!
//!   drill_fraud identity
//!   drill_fraud report --bootstrap URL --seed-hex HEX [--kind K] --target ID
//!   drill_fraud slash  --seed-hex HEX --target ID --required-uj U --pct P
//!                      [--reason STR] [--chain-base URL]
//!
//! Kinds: failed_worker | leeching | dishonest_results | rate_abuse.
//! The slash path uses the same wire contract as stake.rs (`jerith-slash1`,
//! reporter bound to the pubkey fingerprint), so the chain validates it
//! exactly as it would a notice from a running node.

use jerith_mesh_core::bootstrap::BootstrapClient;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KindArg {
    FailedWorker,
    Leeching,
    DishonestResults,
    RateAbuse,
}

impl From<KindArg> for jerith_mesh_core::bootstrap_server::FraudKind {
    fn from(k: KindArg) -> Self {
        use jerith_mesh_core::bootstrap_server::FraudKind as F;
        match k {
            KindArg::FailedWorker => F::FailedWorker,
            KindArg::Leeching => F::Leeching,
            KindArg::DishonestResults => F::DishonestResults,
            KindArg::RateAbuse => F::RateAbuse,
        }
    }
}

impl std::str::FromStr for KindArg {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "failed_worker" => Ok(Self::FailedWorker),
            "leeching" => Ok(Self::Leeching),
            "dishonest_results" => Ok(Self::DishonestResults),
            "rate_abuse" => Ok(Self::RateAbuse),
            other => Err(format!("unknown kind: {other}")),
        }
    }
}

struct Args {
    sub: Sub,
}

enum Sub {
    Identity,
    Report {
        bootstrap: String,
        seed_hex: String,
        kind: KindArg,
        target: String,
    },
    Slash {
        seed_hex: String,
        target: String,
        required_uj: u64,
        pct: u32,
        reason: String,
        chain_base: String,
    },
}

const USAGE: &str = "usage:\n  drill_fraud identity\n  drill_fraud report --bootstrap URL --seed-hex HEX [--kind K] --target ID\n  drill_fraud slash --seed-hex HEX --target ID --required-uj U --pct P [--reason STR] [--chain-base URL]\n";

fn parse_args() -> anyhow::Result<Args> {
    let mut it = std::env::args().skip(1);
    let sub = it.next().ok_or_else(|| anyhow::anyhow!("{USAGE}"))?;
    match sub.as_str() {
        "identity" => Ok(Args { sub: Sub::Identity }),
        "report" => {
            let mut flag = String::new();
            let mut pairs: Vec<(String, String)> = Vec::new();
            for a in it {
                if a.starts_with("--") {
                    flag = a.trim_start_matches("--").to_string();
                } else if flag.is_empty() {
                    return Err(anyhow::anyhow!("unexpected arg {a}\n{USAGE}"));
                } else {
                    pairs.push((std::mem::take(&mut flag), a));
                }
            }
            let get = |name: &str| -> anyhow::Result<String> {
                let mut found = None;
                for (k, v) in &pairs {
                    if k == name {
                        found = Some(v.clone());
                    }
                }
                found.ok_or_else(|| anyhow::anyhow!("missing --{name}\n{USAGE}"))
            };
            let bootstrap = get("bootstrap")?;
            let seed_hex = get("seed-hex")?;
            let target = get("target")?;
            let kind = get("kind").unwrap_or_else(|_| "leeching".into());
            let kind: KindArg = kind.parse().map_err(anyhow::Error::msg)?;
            Ok(Args {
                sub: Sub::Report {
                    bootstrap,
                    seed_hex,
                    kind,
                    target,
                },
            })
        }
        "slash" => {
            let mut flag = String::new();
            let mut pairs: Vec<(String, String)> = Vec::new();
            for a in it {
                if a.starts_with("--") {
                    flag = a.trim_start_matches("--").to_string();
                } else if flag.is_empty() {
                    return Err(anyhow::anyhow!("unexpected arg {a}\n{USAGE}"));
                } else {
                    pairs.push((std::mem::take(&mut flag), a));
                }
            }
            let get = |name: &str| -> anyhow::Result<String> {
                let mut found = None;
                for (k, v) in &pairs {
                    if k == name {
                        found = Some(v.clone());
                    }
                }
                found.ok_or_else(|| anyhow::anyhow!("missing --{name}\n{USAGE}"))
            };
            let seed_hex = get("seed-hex")?;
            let target = get("target")?;
            let required_uj = get("required-uj")?.parse()?;
            let pct = get("pct")?.parse()?;
            let reason = get("reason").unwrap_or_else(|_| "drill".into());
            let chain_base = get("chain-base").unwrap_or_else(|_| "http://127.0.0.1:8322".into());
            Ok(Args {
                sub: Sub::Slash {
                    seed_hex,
                    target,
                    required_uj,
                    pct,
                    reason,
                    chain_base,
                },
            })
        }
        other => Err(anyhow::anyhow!("unknown command {other}\n{USAGE}")),
    }
}

fn load_id(seed_hex: &str) -> anyhow::Result<jerith_mesh_core::signing::SigningIdentity> {
    let bytes = hex::decode(seed_hex)?;
    jerith_mesh_core::signing::SigningIdentity::from_secret_bytes(&bytes)
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let args = parse_args()?;
    match args.sub {
        Sub::Identity => {
            let id = jerith_mesh_core::signing::SigningIdentity::generate();
            println!("node_id={}", id.node_id());
            println!("pubkey_hex={}", id.pubkey_hex());
            println!("secret_hex={}", hex::encode(id.secret_bytes()));
        }
        Sub::Report {
            bootstrap,
            seed_hex,
            kind,
            target,
        } => {
            let id = load_id(&seed_hex)?;
            let client = BootstrapClient::new(&bootstrap);
            let resp = client
                .report_fraud(kind.into(), &target, &id.node_id(), &id)
                .await?;
            println!("{}", serde_json::to_string_pretty(&resp)?);
        }
        Sub::Slash {
            seed_hex,
            target,
            required_uj,
            pct,
            reason,
            chain_base,
        } => {
            use jerith_mesh_core::config::{CryptoChain, CryptoConfig};
            let id = load_id(&seed_hex)?;
            let mut crypto = CryptoConfig {
                chain: CryptoChain::Jerithchain,
                ..Default::default()
            };
            crypto.stake.required_uj = required_uj;
            crypto.stake.slash_percent_per_offense = pct;
            crypto.stake.chain_slash = true;
            let reporter_id_for_log = id.node_id();
            let mgr =
                jerith_mesh_core::stake::StakeManager::new(&crypto, Some(id), Some(chain_base.clone()));
            mgr.notice(&target, &reason, "drill-sandbox-evidence-payload").await;
            println!("slash notice submitted: reporter={reporter_id_for_log} target={target} amount={}µJ (probable)", required_uj * pct as u64 / 100);
            if let Some(v) = mgr.chain_view().await {
                println!("chain /stake view for target: {v}");
            } else {
                println!("chain /stake view unavailable (see node-side stake.json for the journal)");
            }
        }
    }
    Ok(())
}
