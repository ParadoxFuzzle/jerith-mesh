//! Multi-bot coordination — the bot's named workers over the mesh
//! coordinator (`crates/core/src/coordinator.rs`, served by every node's
//! gateway under `/v1/coordinator/*`).
//!
//! Distinct, named agents share this host (and this bot process) and hand
//! work to one another without manual context management:
//!
//! - **The registry** (`bot.workers`, `bot.worker.spawn`,
//!   `bot.worker.stop`): names workers and assigns each its own model —
//!   a local ollama weight tag (`ollama:qwen3:8b`) or any gateway-visible
//!   model id (the main bot's own model, a remote tier — whatever the
//!   node runtime routes).
//! - **The hand-off channel** (`bot.delegate` + claim loops): a
//!   delegator posts a unit of work addressed to a worker; the worker's
//!   loop claims FIFO, solves through the gateway with ITS model, and
//!   posts the result back. A worker's answer may end with ONE
//!   `HANDOFF> <worker> | <subject>` line: the loop then posts a
//!   follow-up whose subject carries the reserved `[chained]` marker.
//!   Chains are depth-1 — a chained task is never re-delegated — so
//!   workers pass sideways work to each other without a human in the
//!   loop and without a delegation ping-pong.
//! - **The shared pool** (`bot.pool.put/get/keys`): namespaced context
//!   blobs. Delegators stage context under a key and put ONLY the key
//!   name in the handoff body; the worker fetches it through the same
//!   surface.
//!
//! Owner-only plugin surface (it does NOT ride `[plugins] enabled`; it
//! joins owner-DM chat sessions only — spawning/steering named agents
//! spends host compute and hands out mesh identities). Worker loops run
//! inside this bot process (`start()` at boot, one claim task per active
//! registry worker), live entirely off the coordinator HTTP surface, and
//! self-terminate when their worker is removed from the registry.
//!
//! Config: `[coordinator] enabled = true` (bot.toml; default true).

use crate::config::BotConfig;
use serde_json::{Value, json};
use std::collections::{BTreeSet, HashMap};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use anyhow::Context as _;

/// Supervisor tick + worker claim-loop cadence.
const CLAIM_TICK: Duration = Duration::from_secs(3);

/// Solve wall clock (one gateway chat completion; the gateway bounds a
/// non-stream local runtime at 180 s before mesh fallback).
const SOLVE_TIMEOUT_SECS: u64 = 240;

/// Reserved subject prefix stamped on chained units — a claimed task
/// that arrived through a chain is never re-delegated (depth 1).
const CHAIN_MARKER: &str = "[chained]";

/// Cap on the answer snippet carried into a chained unit's body.
const CHAIN_SNIPPET_CHARS: usize = 2_000;

/// One named worker's routing + persona, mirroring the registry row.
#[derive(Debug, Clone)]
pub struct WorkerSpec {
    pub name: String,
    pub model: String,
    pub persona: String,
}

impl WorkerSpec {
    /// Addressable, safe identifier: ASCII alnum + `. _ -`, 1..=100.
    fn valid_name(name: &str) -> anyhow::Result<()> {
        anyhow::ensure!(
            !name.is_empty() && name.chars().count() <= 100,
            "worker name must be 1..=100 chars"
        );
        anyhow::ensure!(
            name.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')),
            "worker name must be [A-Za-z0-9._-] (it becomes an addressable identifier)"
        );
        Ok(())
    }

    fn from_args(args: &Value) -> anyhow::Result<WorkerSpec> {
        let name = require(args, "name")?;
        let model = require(args, "model")?;
        let persona = args
            .get("persona")
            .and_then(Value::as_str)
            .unwrap_or_default();
        Self::valid_name(&name)?;
        Ok(WorkerSpec {
            name,
            model,
            persona: persona.trim().chars().take(6_000).collect(),
        })
    }
}

fn require(args: &Value, field: &str) -> anyhow::Result<String> {
    args.get(field)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .with_context(|| format!("field {field:?} is required"))
}

// ---------------------------------------------------------------------------
// Claim-loop machinery
// ---------------------------------------------------------------------------

/// Running claim-loop tasks keyed by worker name; the supervisor spawns
/// and prunes entries (a loop exits on membership loss, the supervisor
/// observes the finished handle next tick).
fn loops() -> &'static Mutex<HashMap<String, tokio::task::JoinHandle<()>>> {
    static LOOPS: OnceLock<Mutex<HashMap<String, tokio::task::JoinHandle<()>>>> = OnceLock::new();
    LOOPS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Boot the coordinator side (idempotent): the supervisor keeps this
/// process's claim loops matched to the registry and tolerates gateway
/// blips forever.
pub fn start(cfg: BotConfig) {
    tokio::spawn(supervisor(cfg));
}

async fn supervisor(cfg: BotConfig) {
    let base = cfg.gateway_url.clone();
    tracing::info!(target: "bot", base = %base, "coordinator supervisor started");
    loop {
        tokio::time::sleep(CLAIM_TICK).await;
        let list = crate::api::client()
            .get(format!("{base}/v1/coordinator/workers/list"))
            .timeout(Duration::from_secs(15))
            .send()
            .await;
        let Ok(resp) = list else {
            continue;
        };
        if !resp.status().is_success() {
            tracing::debug!(target: "bot", status = %resp.status(), "coordinator tick refused");
            continue;
        }
        let v: Value = resp.json().await.unwrap_or(json!({}));
        let members: Vec<WorkerSpec> = v
            .get("workers")
            .and_then(Value::as_array)
            .map(|rows| {
                rows.iter()
                    .filter_map(|w| {
                        // "stopped" rows never run loops here (they are
                        // removed by bot.worker.stop anyway).
                        let status = w.get("status").and_then(Value::as_str)?;
                        if status == "stopped" {
                            return None;
                        }
                        Some(WorkerSpec {
                            name: w.get("name")?.as_str()?.to_string(),
                            model: w.get("model")?.as_str()?.to_string(),
                            persona: w
                                .get("persona")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        {
            let mut map = loops().lock().expect("loop map");
            map.retain(|_, h| !h.is_finished());
            let live: BTreeSet<String> = members.iter().map(|s| s.name.clone()).collect();
            let stale: Vec<String> = map
                .keys()
                .filter(|k| !live.contains(*k))
                .cloned()
                .collect();
            for name in stale {
                if let Some(h) = map.remove(&name) {
                    h.abort();
                }
            }
            for spec in members {
                if !map.contains_key(&spec.name) {
                    let cfg = cfg.clone();
                    let name = spec.name.clone();
                    let h = tokio::spawn(worker_loop(cfg, spec));
                    map.insert(name, h);
                }
            }
        }
    }
}

/// One worker's claim loop: heartbeat ⇄ membership check, then one
/// claim/solve/finish cycle per tick. Exits when the registry no longer
/// holds the worker (its backlog was expired by the store on removal).
async fn worker_loop(cfg: BotConfig, mut spec: WorkerSpec) {
    let base = cfg.gateway_url.clone();
    tracing::info!(target: "bot", worker = %spec.name, model = %spec.model, "worker claim loop started");
    loop {
        tokio::time::sleep(CLAIM_TICK).await;
        // Membership first: a removed worker's loop must stop (its
        // heartbeat join below would otherwise resurrect the row).
        let members = crate::api::client()
            .get(format!("{base}/v1/coordinator/workers/list"))
            .timeout(Duration::from_secs(15))
            .send()
            .await;
        // NOTE: spec refresh happens under the present-check below —
        // the registry is the SOURCE OF TRUTH for model/persona.
        if let Ok(resp) = members
            && resp.status().is_success()
        {
            let v: Value = resp.json().await.unwrap_or(json!({}));
            let mut present = false;
            for w in v.get("workers").and_then(Value::as_array).map(|a| a.as_slice()).unwrap_or(&[]) {
                if w.get("name").and_then(Value::as_str) != Some(spec.name.as_str()) {
                    continue;
                }
                present = true;
                // Registry is the source of truth: a rejoin with a new
                // model/persona updates THIS running loop on the next
                // tick (heartbeat mirrors the refreshed spec, so the row
                // no longer flips back to a stale model).
                let model = w
                    .get("model")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                if !model.is_empty() {
                    spec.model = model;
                }
                spec.persona = w
                    .get("persona")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .chars()
                    .take(6_000)
                    .collect();
            }
            if !present {
                tracing::info!(target: "bot", worker = %spec.name, "worker removed — claim loop exits");
                loops().lock().expect("loop map").remove(&spec.name);
                return;
            }
        }
        // Heartbeat (registry status shows the loop is alive).
        let _ = crate::api::client()
            .post(format!("{base}/v1/coordinator/workers/join"))
            .json(&json!({
                "name": spec.name, "model": spec.model,
                "persona": spec.persona, "status": "idle",
            }))
            .timeout(Duration::from_secs(15))
            .send()
            .await;
        if let Err(e) = run_one_iteration(&spec, &base).await {
            tracing::debug!(target: "bot", worker = %spec.name, error = %e, "claim tick skipped");
        }
    }
}

/// One claim→solve→finish(+chain) cycle for a worker against one
/// coordinator base URL. Idle (no claimable work) → Ok(false).
pub async fn run_one_iteration(spec: &WorkerSpec, base: &str) -> anyhow::Result<bool> {
    let client = crate::api::client();
    let claim_url = format!("{base}/v1/coordinator/handoff/claim");
    let resp = client
        .post(&claim_url)
        .json(&json!({"worker": spec.name}))
        .timeout(Duration::from_secs(15))
        .send()
        .await
        .with_context(|| format!("coordinator unreachable at {claim_url}"))?;
    if !resp.status().is_success() {
        anyhow::bail!("handoff/claim answered {}", resp.status());
    }
    let handoff: Value = resp.json().await.unwrap_or(json!({}));
    let Some(id) = handoff.get("id").and_then(Value::as_i64) else {
        return Ok(false); // idle tick
    };
    let subject = handoff
        .get("subject")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let body = handoff
        .get("body")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    tracing::info!(target: "bot", worker = %spec.name, id, subject = %subject, "handoff claimed");

    // Solve through the gateway with the WORKER's model. A failed solve
    // still finishes the handoff (with the error text) — work never
    // strands claimed-but-unfinished.
    let answer = match solve(base, spec, &subject, &body).await {
        Ok(a) => a,
        Err(e) => format!("⚠️ worker {} failed: {e:#}", spec.name),
    };
    client
        .post(format!("{base}/v1/coordinator/handoff/finish"))
        .json(&json!({"id": id, "worker": spec.name, "result": answer.clone()}))
        .timeout(Duration::from_secs(15))
        .send()
        .await
        .ok();

    // Depth-1 chain: follow the worker's directive ONLY for tasks that
    // did not themselves arrive via a chain.
    if !body.starts_with(CHAIN_MARKER)
        && let Some((to, next_subject)) = parse_handoff_directive(&answer)
    {
        let snippet: String = answer.chars().take(CHAIN_SNIPPET_CHARS).collect();
        client
            .post(format!("{base}/v1/coordinator/handoff/post"))
            .json(&json!({
                "from": spec.name,
                "to": to,
                "subject": format!("{CHAIN_MARKER} {next_subject}"),
                "body": format!("(chained by {} · original task: {snippet})", spec.name),
            }))
            .timeout(Duration::from_secs(15))
            .send()
            .await
            .ok();
        tracing::info!(target: "bot", worker = %spec.name, to = %to, "chained follow-up posted");
    }
    Ok(true)
}

/// The worker's solve: one bounded gateway chat completion with the
/// worker's model + persona, NO tools (workers solve; the main bot is
/// the conversational surface).
async fn solve(base: &str, spec: &WorkerSpec, subject: &str, task: &str) -> anyhow::Result<String> {
    let url = format!("{base}/v1/chat/completions");
    let persona = if spec.persona.is_empty() {
        "(none given)".to_string()
    } else {
        spec.persona.chars().take(6_000).collect()
    };
    let body = json!({
        "model": spec.model,
        "messages": [
            {
                "role": "system",
                "content": format!(
                    "You are worker `{name}` in a multi-agent mesh. Persona: {persona}\n\
                     Solve the delegated task in plain, complete prose. If part of the work \
                     is genuinely another worker's job, END with one line: \
                     `HANDOFF> <worker-name> | <subject>`.",
                    name = spec.name, persona = persona
                ),
            },
            {"role": "user", "content": format!("{subject}\n\n{task}")},
        ],
        "stream": false,
    });
    let resp = crate::api::client()
        .post(&url)
        .json(&body)
        .timeout(Duration::from_secs(SOLVE_TIMEOUT_SECS))
        .send()
        .await
        .with_context(|| format!("gateway unreachable at {url}"))?;
    let status = resp.status();
    let v: Value = resp.json().await.unwrap_or(json!({}));
    if !status.is_success() {
        let err = v.get("error").and_then(Value::as_str).unwrap_or("").to_string();
        anyhow::bail!("gateway {status}: {err}");
    }
    let answer = v
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    tracing::info!(target: "bot", worker = %spec.name, chars = answer.chars().count(), "handoff solved");
    Ok(answer)
}

/// `HANDOFF> <worker> | <subject>` — the LAST such line in an answer (if
/// any) names exactly one follow-up unit. Malformed matches are ignored;
/// surrounding prose is untouched.
fn parse_handoff_directive(answer: &str) -> Option<(String, String)> {
    let mut found = None;
    for line in answer.lines() {
        let line = line.trim_start();
        if let Some(rest) = line.strip_prefix("HANDOFF>").map(|r| r.trim()) {
            if let Some((worker, subject)) = rest.split_once('|') {
                let worker = worker.trim();
                let subject = subject.trim();
                if !worker.is_empty() && !subject.is_empty() {
                    found = Some((worker.to_string(), subject.to_string()));
                }
            }
        }
    }
    found
}

// ---------------------------------------------------------------------------
// The owner-only model-facing skill surface (plugin id `coordinator`)
// ---------------------------------------------------------------------------

type Fut<'a> = Pin<Box<dyn Future<Output = anyhow::Result<Value>> + Send + 'a>>;

async fn post_coordinator(base: &str, path: &str, body: Value) -> anyhow::Result<Value> {
    let url = format!("{base}{path}");
    let resp = crate::api::client()
        .post(&url)
        .json(&body)
        .timeout(Duration::from_secs(15))
        .send()
        .await
        .with_context(|| format!("gateway unreachable at {url}"))?;
    let status = resp.status();
    let v: Value = resp.json().await.unwrap_or(json!({}));
    if !status.is_success() {
        let err = v.get("error").and_then(Value::as_str).unwrap_or("").to_string();
        anyhow::bail!("coordinator {status}: {err}");
    }
    Ok(v)
}

async fn get_coordinator(base: &str, path: &str) -> anyhow::Result<Value> {
    let url = format!("{base}{path}");
    let resp = crate::api::client()
        .get(&url)
        .timeout(Duration::from_secs(15))
        .send()
        .await
        .with_context(|| format!("gateway unreachable at {url}"))?;
    let status = resp.status();
    let v: Value = resp.json().await.unwrap_or(json!({}));
    if !status.is_success() {
        anyhow::bail!("coordinator {status}");
    }
    Ok(v)
}

fn workers_runner<'a>(cfg: &'a BotConfig, _args: &'a Value) -> Fut<'a> {
    Box::pin(async move {
        let v = get_coordinator(&cfg.gateway_url, "/v1/coordinator/workers/list").await?;
        Ok(v)
    })
}

fn worker_spawn_runner<'a>(cfg: &'a BotConfig, args: &'a Value) -> Fut<'a> {
    Box::pin(async move {
        let spec = WorkerSpec::from_args(args)?;
        post_coordinator(
            &cfg.gateway_url,
            "/v1/coordinator/workers/join",
            json!({
                "name": spec.name, "model": spec.model,
                "persona": spec.persona, "status": "idle",
            }),
        )
        .await?;
        Ok(json!({
            "ok": true,
            "worker": spec.name,
            "model": spec.model,
            "note": "claim loop starts within ~3s; delegate with bot.delegate"
        }))
    })
}

fn worker_stop_runner<'a>(cfg: &'a BotConfig, args: &'a Value) -> Fut<'a> {
    Box::pin(async move {
        let name = require(args, "name")?;
        post_coordinator(
            &cfg.gateway_url,
            "/v1/coordinator/workers/remove",
            json!({"name": name}),
        )
        .await?;
        Ok(json!({"ok": true, "worker": name}))
    })
}

fn delegate_runner<'a>(cfg: &'a BotConfig, args: &'a Value) -> Fut<'a> {
    Box::pin(async move {
        let to = require(args, "to")?;
        let subject = require(args, "subject")?;
        let body = require(args, "body")?;
        let v = post_coordinator(
            &cfg.gateway_url,
            "/v1/coordinator/handoff/post",
            json!({"from": "main", "to": to, "subject": subject, "body": body}),
        )
        .await?;
        Ok(json!({"ok": true, "id": v.get("id").cloned().unwrap_or(Value::Null)}))
    })
}

fn inbox_runner<'a>(cfg: &'a BotConfig, args: &'a Value) -> Fut<'a> {
    Box::pin(async move {
        let limit = args.get("limit").and_then(Value::as_i64).unwrap_or(20);
        let v = post_coordinator(
            &cfg.gateway_url,
            "/v1/coordinator/handoff/from",
            json!({"from": "main", "limit": limit.clamp(1, 50)}),
        )
        .await?;
        Ok(v)
    })
}

fn pool_put_runner<'a>(cfg: &'a BotConfig, args: &'a Value) -> Fut<'a> {
    Box::pin(async move {
        let ns = require(args, "ns")?;
        let key = require(args, "key")?;
        let val = require(args, "val")?;
        post_coordinator(
            &cfg.gateway_url,
            "/v1/coordinator/pool/put",
            json!({"ns": ns, "key": key, "val": val}),
        )
        .await?;
        Ok(json!({"ok": true}))
    })
}

fn pool_get_runner<'a>(cfg: &'a BotConfig, args: &'a Value) -> Fut<'a> {
    Box::pin(async move {
        let ns = require(args, "ns")?;
        let key = require(args, "key")?;
        let v = get_coordinator(
            &cfg.gateway_url,
            &format!(
                "/v1/coordinator/pool/get?ns={}&key={}",
                urlencode(&ns),
                urlencode(&key)
            ),
        )
        .await?;
        Ok(json!({"val": v.get("val").cloned().unwrap_or(Value::Null)}))
    })
}

fn pool_keys_runner<'a>(cfg: &'a BotConfig, args: &'a Value) -> Fut<'a> {
    Box::pin(async move {
        let ns = require(args, "ns")?;
        let v = get_coordinator(
            &cfg.gateway_url,
            &format!("/v1/coordinator/pool/keys?ns={}", urlencode(&ns)),
        )
        .await?;
        Ok(v)
    })
}

/// Namespaces/keys ride the URL — percent-encode any non-URI-safe bytes.
fn urlencode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// The coordinator's model-facing skills (plugin id `coordinator`,
/// owner-DM-only gating handled by plugins.rs `OWNER_ONLY_PLUGINS`).
pub fn skills() -> Vec<crate::plugins::SkillSpec> {
    use crate::plugins::SkillSpec;
    vec![
        SkillSpec {
            name: "bot.workers",
            plugin: "coordinator",
            description: "List named bot workers with their model routing (local ollama weights or remote ids) and status.",
            parameters: json!({"type": "object"}),
            run: workers_runner,
        },
        SkillSpec {
            name: "bot.worker.spawn",
            plugin: "coordinator",
            description: "Create a NEW named bot worker with its own model routing (e.g. 'ollama:qwen3:8b' for a local weight, or a remote gateway model id) and optional persona. Never put secrets in the persona.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Addressable name [A-Za-z0-9._-]"},
                    "model": {"type": "string", "description": "Model id this worker solves with"},
                    "persona": {"type": "string", "description": "Optional brief operating persona"}
                },
                "required": ["name", "model"]
            }),
            run: worker_spawn_runner,
        },
        SkillSpec {
            name: "bot.worker.stop",
            plugin: "coordinator",
            description: "Remove a named bot worker: the registry row goes, its claim loop exits, and its open backlog expires.",
            parameters: json!({
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"]
            }),
            run: worker_stop_runner,
        },
        SkillSpec {
            name: "bot.delegate",
            plugin: "coordinator",
            description: "Hand work to a named worker on the dedicated hand-off channel: the worker claims, solves with its own model, and the result comes back (read via bot.inbox).",
            parameters: json!({
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Worker name"},
                    "subject": {"type": "string", "description": "Short task title"},
                    "body": {"type": "string", "description": "The task statement; reference pool keys instead of pasting big context"}
                },
                "required": ["to", "subject", "body"]
            }),
            run: delegate_runner,
        },
        SkillSpec {
            name: "bot.inbox",
            plugin: "coordinator",
            description: "Result pickup for handoffs this bot delegated: newest first, finished rows carry the worker's result.",
            parameters: json!({
                "type": "object",
                "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 50}},
                "required": []
            }),
            run: inbox_runner,
        },
        SkillSpec {
            name: "bot.pool.put",
            plugin: "coordinator",
            description: "Store context in the shared memory pool under a namespaced key; hand workers just the key name.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "ns": {"type": "string", "description": "e.g. ctx:invoice-2026"},
                    "key": {"type": "string"},
                    "val": {"type": "string", "description": "The context blob"}
                },
                "required": ["ns", "key", "val"]
            }),
            run: pool_put_runner,
        },
        SkillSpec {
            name: "bot.pool.get",
            plugin: "coordinator",
            description: "Read one shared-memory-pool blob by namespace + key.",
            parameters: json!({
                "type": "object",
                "properties": {"ns": {"type": "string"}, "key": {"type": "string"}},
                "required": ["ns", "key"]
            }),
            run: pool_get_runner,
        },
        SkillSpec {
            name: "bot.pool.keys",
            plugin: "coordinator",
            description: "List the shared-memory-pool keys in a namespace.",
            parameters: json!({
                "type": "object",
                "properties": {"ns": {"type": "string"}},
                "required": ["ns"]
            }),
            run: pool_keys_runner,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn directive_parsed_from_the_last_matching_line() {
        let (w, s) =
            parse_handoff_directive("Pipes fixed.\nHANDOFF> carpenter | rehang the door").unwrap();
        assert_eq!(w, "carpenter");
        assert_eq!(s, "rehang the door");
        assert!(parse_handoff_directive("all done, nothing further").is_none());
        assert!(parse_handoff_directive("HANDOFF> carpenter only").is_none());
        let (w, _) = parse_handoff_directive("HANDOFF> a | x\n\nHANDOFF> b | y").unwrap();
        assert_eq!(w, "b");
        let (w, _) =
            parse_handoff_directive("step 1...\n    HANDOFF> painter | walls").unwrap();
        assert_eq!(w, "painter");
    }

    #[test]
    fn spec_validation_is_typed() {
        let spec = WorkerSpec::from_args(&json!({
            "name": "plumber", "model": "ollama:qwen3:8b", "persona": "pipes",
        }))
        .unwrap();
        assert_eq!(spec.name, "plumber");
        let big = WorkerSpec::from_args(&json!({
            "name": "pl", "model": "m", "persona": "x".repeat(100_000),
        }))
        .unwrap();
        assert_eq!(big.persona.chars().count(), 6_000, "persona capped");

        assert!(WorkerSpec::from_args(&json!({"model": "m"})).is_err(), "name required");
        assert!(
            WorkerSpec::from_args(&json!({"name": "has space", "model": "m"})).is_err(),
            "charset enforced"
        );
        assert!(WorkerSpec::from_args(&json!({"name": "a"})).is_err(), "model required");
        assert!(WorkerSpec::valid_name("").is_err());
        assert!(WorkerSpec::valid_name("eval$(id)").is_err());
        assert!(WorkerSpec::valid_name(&"x".repeat(101)).is_err());
        assert!(WorkerSpec::valid_name("fix-it_01").is_ok());
    }

    #[test]
    fn urlencode_roundtrips_namespaces() {
        assert_eq!(urlencode("ctx:invoice"), "ctx%3Ainvoice");
        assert_eq!(urlencode("plain-ns_1.0"), "plain-ns_1.0");
        assert_eq!(urlencode("a b&c"), "a%20b%26c");
    }

    /// The full loop path against a real local HTTP stub: claim → solve
    /// (with the WORKER's model) → finish with the answer → chained
    /// follow-up posted with the depth-1 marker. One iteration of the
    /// production `run_one_iteration`, no shortcuts.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn claim_solve_finish_chain_through_http() {
        use axum::routing::{get as rget, post as rpost};
        let log: std::sync::Arc<Mutex<Vec<(String, Value)>>> =
            std::sync::Arc::new(Mutex::new(Vec::new()));

        let seen_solve = log.clone();
        let seen_finish = log.clone();
        let seen_post = log.clone();
        let app = axum::Router::new()
            .route(
                "/v1/chat/completions",
                rpost(move |axum::Json(body): axum::Json<Value>| {
                    let seen = seen_solve.clone();
                    async move {
                        seen.lock().unwrap().push(("solve".into(), body));
                        axum::Json(json!({
                            "choices": [{"message": {
                                "role": "assistant",
                                "content": "Welded the pipe.\nHANDOFF> inspector | check the weld"
                            }}]
                        }))
                    }
                }),
            )
            .route(
                "/v1/coordinator/handoff/claim",
                rpost(move || async {
                    axum::Json(json!({
                        "id": 5,
                        "from": "main",
                        "to": "welder",
                        "subject": "fix the pipe",
                        "body": "the east wing leaks",
                    }))
                }),
            )
            .route(
                "/v1/coordinator/handoff/finish",
                rpost(move |axum::Json(body): axum::Json<Value>| {
                    let seen = seen_finish.clone();
                    async move {
                        seen.lock().unwrap().push(("finish".into(), body));
                        axum::Json(json!({"ok": true}))
                    }
                }),
            )
            .route(
                "/v1/coordinator/handoff/post",
                rpost(move |axum::Json(body): axum::Json<Value>| {
                    let seen = seen_post.clone();
                    async move {
                        seen.lock().unwrap().push(("post".into(), body));
                        axum::Json(json!({"ok": true}))
                    }
                }),
            )
            .route(
                "/v1/coordinator/workers/list",
                rget(|| async { axum::Json(json!({"workers": []})) }),
            );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let spec = WorkerSpec {
            name: "welder".into(),
            model: "ollama:welder:7b".into(),
            persona: String::new(),
        };
        let claimed = run_one_iteration(&spec, &format!("http://{addr}"))
            .await
            .unwrap();
        assert!(claimed, "the stub had work to claim");
        // A second iteration also claims (the stub always answers 5) —
        // fine: the call returns per-iteration.

        let events = log.lock().unwrap();
        assert!(
            events.iter().any(|(k, v)| k == "finish"
                && v["id"].as_i64() == Some(5)
                && v["worker"] == json!("welder")
                && v["result"].as_str().unwrap().contains("Welded the pipe")),
            "finish carries the answer: {events:?}"
        );
        assert!(
            events.iter().any(|(k, v)| k == "post"
                && v["to"] == json!("inspector")
                && v["from"] == json!("welder")
                && v["subject"].as_str().unwrap_or("").starts_with(CHAIN_MARKER)),
            "chained post carries the depth-1 marker: {events:?}"
        );
        assert!(
            events.iter().any(|(k, v)| k == "solve" && v["model"] == json!("ollama:welder:7b")),
            "solve rides the worker's own model: {events:?}"
        );
    }
}
