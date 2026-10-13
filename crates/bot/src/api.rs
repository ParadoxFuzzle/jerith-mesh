//! Gateway + chain HTTP calls the slash commands make.
//!
//! Kept as plain async functions (no poise types) so they are unit-testable
//! and reusable. Everything talks OpenAI-ish shapes to the gateway.

use anyhow::{Context as _, Result};
use serde_json::{json, Value};

const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(90);

/// Legacy placeholder some code paths still carry — only used if the
/// gateway's /status could not tell us the real runtime model.
const MODEL_PLACEHOLDER: &str = "jerith-mesh";

/// Resolved runtime model per gateway URL, fetched once per URL per
/// process (a bot's config is fixed at startup; tests run several — the
/// per-URL map keeps them from poisoning each other). Sending the
/// placeholder model verbatim to an Ollama/Colibri runtime gets
/// `model 'jerith-mesh' not found` — the node actually serves whatever
/// its config.advertised model is, exposed on /status as `model`.
static RESOLVED_MODELS: tokio::sync::Mutex<
    Option<std::collections::HashMap<String, String>>,
> = tokio::sync::Mutex::const_new(None);

/// The model id chat requests should carry: the node's advertised runtime
/// model, or `None` when /status is unreachable (caller falls back to the
/// placeholder and lets the gateway error surface verbatim).
async fn runtime_model(cfg: &crate::config::BotConfig) -> Option<String> {
    let mut guard = RESOLVED_MODELS.lock().await;
    let map = guard.get_or_insert_with(Default::default);
    if let Some(model) = map.get(&cfg.gateway_url) {
        return Some(model.clone());
    }
    let url = format!("{}/status", cfg.gateway_url.trim_end_matches('/'));
    let resp = client()
        .get(&url)
        .send()
        .await
        .with_context(|| format!("/model status lookup failed at {url}"))
        .ok()?;
    let body: Value = resp.json().await.context("model status: non-JSON body").ok()?;
    let model = model_from_status(&body)?;
    map.insert(cfg.gateway_url.clone(), model.clone());
    Some(model)
}

/// Pull the advertised model out of a gateway /status body: a non-empty
/// string `model` field (e.g. `{ "model": "qwen3:8b", ... }`).
fn model_from_status(body: &Value) -> Option<String> {
    let model = body.get("model")?.as_str()?.trim();
    (!model.is_empty()).then(|| model.to_string())
}

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(TIMEOUT)
        .build()
        .expect("static reqwest client config")
}

/// POST /v1/chat/completions on the gateway. Returns the assistant text plus
/// the `mesh_fallback` node id when the mesh (not the local runtime) served it.
/// Message assembly for every chat shape the bot produces (`/ask`, DM and
/// @mention fallbacks share this path). A persona system prompt goes first,
/// then the conversation history, then the caller's prompt. Pure fn — tests
/// cover ordering without a network or framework.
pub fn build_messages(
    history: &[Value],
    prompt: &str,
    persona: Option<&str>,
) -> Vec<Value> {
    let mut messages: Vec<Value> = Vec::with_capacity(history.len() + 2);
    if let Some(p) = persona.map(str::trim).filter(|p| !p.is_empty()) {
        messages.push(json!({"role": "system", "content": p}));
    }
    messages.extend(history.iter().cloned());
    messages.push(json!({"role": "user", "content": prompt}));
    messages
}

pub async fn chat(
    cfg: &crate::config::BotConfig,
    prompt: &str,
    raw_history: &[Value],
) -> Result<(String, Option<String>)> {
    chat_with_usage(cfg, prompt, raw_history)
        .await
        .map(|(c, s, _t)| (c, s))
}

/// Chat with token accounting: returns `(reply, mesh_peer, tokens_used)` —
/// the runtime's `usage.total_tokens` summed over all rounds of the chat
/// (tool-loop when it engaged). 0 when the runtime omits figures.
pub async fn chat_with_usage(
    cfg: &crate::config::BotConfig,
    prompt: &str,
    raw_history: &[Value],
) -> Result<(String, Option<String>, u64)> {
    // Context squeeze (OpenClaw context-squeeze port): compact the history
    // before it rides the request when it blows the configured budget.
    // Pure pass first (drop-oldest-turns, head-trim) — the summary stage
    // needs the caller's state handle and runs separately (squeeze.rs docs).
    let history = if cfg.squeeze.enabled {
        crate::squeeze::squeeze_passive(raw_history, cfg.squeeze.budget_chars)
    } else {
        raw_history.to_vec()
    };
    // Guard screening covers user-side content only: history + prompt. The
    // persona system prompt is operator-owned static config and is attached
    // AFTER screening — the Guard's PII redaction could otherwise mangle
    // the owner's own security rules (e.g. IDs defined in bot.toml / SOUL)
    // that the persona quotes verbatim.
    let mut messages = build_messages(&history, prompt, None);

    // Jerith Guard: screen OUTBOUND content before any provider call
    // (defense in depth — the gateway screens too, but the bot token can
    // point at any operator-chosen gateway URL, and the reply in Discord
    // should cite the blocking rule directly without a round-trip).
    // PIIs (emails/phones) are redacted in place; SSNs/cards/credentials
    // and injection / malware-shape prompts hard-block.

    // Jerith Guard: screen OUTBOUND content before any provider call
    // (defense in depth — the gateway screens too, but the bot token can
    // point at any operator-chosen gateway URL, and the reply in Discord
    // should cite the blocking rule directly without a round-trip).
    // PIIs (emails/phones) are redacted in place; SSNs/cards/credentials
    // and injection / malware-shape prompts hard-block.
    if cfg.guard.enabled {
        match jerith_mesh_core::guard::Guard::build(&cfg.guard) {
            Ok(guard) => match guard.screen_messages(&messages) {
                Ok(Some(patched)) => messages = patched,
                Ok(None) => {}
                Err(verdict) => anyhow::bail!(
                    "🛡️ Jerith Guard blocked this request ({}). Edit the prompt and try again.",
                    verdict.block_reason()
                ),
            },
            Err(e) => {
                // A malformed [guard] in bot.toml must degrade to the
                // gateway's own screening, never to a failed bot.
                tracing::warn!(error = %e, "bot guard config invalid — screening skipped for this call");
            }
        }
    }

    // Persona goes first, after user-side screening (see above). The bot
    // answering "as Jerith" is exactly this message.
    let persona_first = cfg
        .system_prompt
        .as_deref()
        .map(|p| !p.trim().is_empty())
        .unwrap_or(false);
    if persona_first {
        messages.insert(
            0,
            json!({"role": "system", "content": cfg.system_prompt.clone().unwrap_or_default()}),
        );
    }

    // Live identity: resolve the tagged speaker against the jerith_people
    // registry (direct Postgres — same contract as the OpenClaw stack:
    // root@127.0.0.1:5432, password via ~/.pgpass). The freshest row is
    // attached as a second system message; an unreachable database or an
    // unknown speaker just means the chat proceeds without it.
    if let Some(pg) = cfg.postgres.as_ref().filter(|p| p.enabled) {
        if let Some(discord_id) = crate::people::extract_speaker_id(&prompt) {
            match crate::people::lookup(pg, &discord_id).await {
                Ok(Some(block)) => messages.insert(
                    if persona_first { 1 } else { 0 },
                    json!({"role": "system", "content": block}),
                ),
                Ok(None) => {}
                Err(e) => {
                    tracing::debug!(error = %e, "people-db lookup skipped — chatting without speaker context");
                }
            }
        }
    }

    let url = format!("{}/v1/chat/completions", cfg.gateway_url);
    let model = runtime_model(cfg).await.unwrap_or_else(|| MODEL_PLACEHOLDER.to_string());

    // Web access (OpenClaw stack migration): the chat request declares the
    // gateway's web skills as OpenAI tools, so a tool-capable runtime can
    // web_search / web_fetch mid-answer. Unsupported model/runtimes get
    // exactly one toolless retry below — web access degrades, never fails
    // the chat.
    const MAX_TOOL_ROUNDS: usize = 3;
    // Tokens accumulated across rounds (session-cost-tracker port input).
    let mut tokens_used: u64 = 0;
    // Tools only for prompts that read as information requests. A shared
    // story/paste (long, no question) must not make the model invent a
    // web_search for it — observed live: a user posted a short story and
    // the model web-searched a line out of it, then answered "{}".
    let mut tools_enabled = cfg.web.enabled && prompt_requests_web(prompt);

    let mut rounds = 0usize;
    loop {
        let mut payload = json!({
            "model": model,
            "messages": messages,
        });
        if tools_enabled {
            payload["tools"] = Value::Array(web_tool_defs());
        }
        let resp = client()
            .post(&url)
            .json(&payload)
            .send()
            .await
            .with_context(|| format!("gateway unreachable at {url}"))?;

        let status = resp.status();
        let body: Value = resp.json().await.context("gateway returned non-JSON")?;
        if !status.is_success() {
            // First-round failure with tools on: one toolless retry in case
            // the runtime rejects the tools array wholesale.
            if tools_enabled && rounds == 0 {
                tools_enabled = false;
                continue;
            }
            let err = body
                .get("error")
                .and_then(value_to_string)
                .unwrap_or_else(|| status.to_string());
            anyhow::bail!("gateway error {status}: {err}");
        }
        let mut tool_calls = body
            .pointer("/choices/0/message/tool_calls")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        // Slop tolerance (OpenClaw-era runtimes): some OpenAI-compatible
        // stacks — qwen3:8b via LiteLLM/Ollama on this vps — answer the
        // tool intent as plain JSON content instead of native tool_calls.
        // Detect `{"function_name": …, "arguments": …}` there and subject
        // it to the same loop (same mock unit tests cover both shapes).
        if tools_enabled && tool_calls.is_empty() && rounds < MAX_TOOL_ROUNDS {
            match content_tool_intent(&body) {
                Some((name, args)) => {
                    tool_calls = vec![json!({
                        "id": format!("iw{rounds}"),
                        "type": "function",
                        "function": {"name": name, "arguments": args.to_string()},
                    })];
                }
                None => {}
            }
        }
        // No tool use (or out of rounds): the plain chat answer — unless
        // it is degenerate ("{}", braces-only JSON, empty): one toolless
        // retry, then a graceful refusal, never braces rendered verbatim
        // into the chat (observed live, see the share incident above).
        if !tools_enabled || tool_calls.is_empty() || rounds >= MAX_TOOL_ROUNDS {
            let content = body
                .pointer("/choices/0/message/content")
                .and_then(value_to_string)
                .unwrap_or_default();
            let served_by = body
                .get("mesh_fallback")
                .and_then(value_to_string)
                .filter(|s| !s.is_empty());
            tokens_used += body
                .pointer("/usage/total_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            if answer_is_degenerate(&content) {
                if tools_enabled && rounds == 0 {
                    tools_enabled = false;
                    continue;
                }
                return Ok((
                    "⚠️ the runtime returned an empty answer — try rephrasing, or ask a direct question.".to_string(),
                    served_by,
                    tokens_used,
                ));
            }
            return Ok((content, served_by, tokens_used));
        }
        tokens_used += body
            .pointer("/usage/total_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);

        // Execute each tool call against the gateway skill endpoints and
        // append the results as `role: tool` messages for the next round.
        let assistant = body
            .pointer("/choices/0/message")
            .cloned()
            .unwrap_or(json!({"role": "assistant"}));
        messages.push(assistant);
        for tc in &tool_calls {
            let id = tc
                .get("id")
                .and_then(value_to_string)
                .unwrap_or_default();
            let name = tc
                .pointer("/function/name")
                .and_then(value_to_string)
                .unwrap_or_default();
            let raw_args = tc.pointer("/function/arguments").cloned().unwrap_or(Value::Null);
            let answer = run_web_tool(cfg, &name, &raw_args).await;
            messages.push(json!({
                "role": "tool",
                "tool_call_id": id,
                "name": name,
                "content": answer,
            }));
        }
        rounds += 1;
    }
}

/// OpenAI tool declarations for the node's web skills — parameter names
/// mirror the OpenClaw web_search/web_fetch tools the old agent had.
pub fn web_tool_defs() -> Vec<Value> {
    vec![
        json!({
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the public web (Brave Search API) and return ranked results with title, url, description and page age. Use when the answer depends on external or recent information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "count": {"type": "integer", "minimum": 1, "maximum": 20, "description": "How many results (default 5)"},
                        "freshness": {"type": "string", "enum": ["day", "week", "month", "year"], "description": "Time filter"}
                    },
                    "required": ["query"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "web_fetch",
                "description": "Download an http(s):// URL and return the page's readable text (HTML stripped, length-capped, private targets refused). Use after web_search to read a specific page.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "http(s):// URL to read"},
                        "max_chars": {"type": "integer", "minimum": 200, "maximum": 16000, "description": "Text cap in characters (default 8000)"}
                    },
                    "required": ["url"]
                }
            }
        }),
    ]
}

/// Parse Ollama/OpenAI tool-call arguments: per spec a JSON-encoded string,
/// but tolerate an inline object.
fn parse_tool_args(args: &Value) -> Value {
    match args {
        Value::String(s) => serde_json::from_str(s).unwrap_or_else(|_| json!({})),
        v => v.clone(),
    }
}

/// Tool names the content-intent path accepts — exactly the tool defs
/// declared to the runtime, so a model can never summon arbitrary tools.
const WEB_TOOL_NAMES: [&str; 2] = ["web_search", "web_fetch"];

/// Does this prompt plausibly want live web data? Questions do — decided
/// by a trailing "?" or an interrogative first word, deliberately NOT by
/// "any '?' anywhere": the live incident prompt quoted dialogue with
/// question marks inside a pasted story, which would otherwise have
/// tripped the tool path. Creative/transform openers (write/summarize/…)
/// and long shared prose (a story, lyrics, an article) go toolless so the
/// model answers in its own words instead of web-searching the text.
fn prompt_requests_web(prompt: &str) -> bool {
    let p = prompt.trim();
    if p.is_empty() {
        return false;
    }
    if p.ends_with('?') {
        return true;
    }
    let lower = p.to_ascii_lowercase();
    // Strip a leading speaker tag ("Noodly: …") before the prefix check.
    let body = lower
        .strip_prefix(|c: char| !c.is_whitespace())
        .and_then(|rest| rest.strip_prefix(':').map(|r| r.trim_start()))
        .unwrap_or(&lower);
    const NON_FACTUAL_OPENERS: [&str; 8] = [
        "write", "summarize", "summarise", "translate", "continue", "rewrite", "rank", "explain",
    ];
    if NON_FACTUAL_OPENERS.iter().any(|w| body.starts_with(w)) {
        return false;
    }
    const FACTUAL_OPENERS: [&str; 14] = [
        "who", "what", "when", "where", "why", "how", "which", "is", "are", "does", "do", "can",
        "should", "find",
    ];
    FACTUAL_OPENERS.iter().any(|w| body.starts_with(w)) || p.chars().count() < 200
}

/// An answer the model effectively failed to produce: empty, whitespace,
/// or a braces-only JSON stub ("{}" / null fields) — never shown to users.
pub fn answer_is_degenerate(content: &str) -> bool {
    let t = content.trim();
    if t.is_empty() {
        return true;
    }
    if t.starts_with('{') && t.ends_with('}') {
        if let Ok(v) = serde_json::from_str::<Value>(t) {
            let has_text = ["content", "text", "answer", "message"]
                .iter()
                .find_map(|k| v.get(k).and_then(Value::as_str))
                .map(|s| !s.trim().is_empty())
                .unwrap_or(false);
            return !has_text;
        }
    }
    false
}

/// Detect a model-declared tool intent hidden in the assistant's plain
/// content: `{"function_name": "web_search", "arguments": {…}}` (also
/// "tool"/"name" key variants, fenced code, and prose around the JSON).
/// Returns the function name and parsed arguments.
fn content_tool_intent(body: &Value) -> Option<(String, Value)> {
    let content = body
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)?;
    // Slice out the outermost JSON object even when the model mumbles
    // around it (```json fences, narration, or <think> preamble).
    let start = content.find('{')?;
    let end = content.rfind('}')?;
    if end <= start {
        return None;
    }
    let v: Value = serde_json::from_str(&content[start..=end]).ok()?;
    let name = v
        .get("function_name")
        .or_else(|| v.get("name"))
        .or_else(|| v.get("tool"))
        .and_then(Value::as_str)?;
    if !WEB_TOOL_NAMES.contains(&name) {
        return None;
    }
    let args = v
        .get("arguments")
        .cloned()
        .or_else(|| v.get("parameters").cloned())
        .unwrap_or(json!({}));
    Some((name.to_string(), parse_tool_args(&args)))
}

/// Execute one model-requested web tool against the gateway skills and
/// return the answer as the JSON string fed back to the model. Tool errors
/// come back as `{"error": …}` so the model can recover instead of the
/// whole chat failing.
async fn run_web_tool(cfg: &crate::config::BotConfig, name: &str, raw_args: &Value) -> String {
    let args = parse_tool_args(raw_args);
    let res: Result<Value> = match name {
        "web_search" => {
            web_search_skill(
                cfg,
                args.get("query").and_then(Value::as_str).unwrap_or_default(),
                args.get("count").and_then(Value::as_u64),
                args.get("freshness").and_then(Value::as_str),
            )
            .await
        }
        "web_fetch" => {
            let cap = args
                .get("max_chars")
                .and_then(Value::as_u64)
                .map(|c| c as usize)
                .unwrap_or(8000)
                .clamp(200, 16000);
            web_fetch_skill(
                cfg,
                args.get("url").and_then(Value::as_str).unwrap_or_default(),
                Some(cap),
            )
            .await
        }
        other => Err(anyhow::anyhow!("unknown function {other:?}")),
    };
    match res {
        Ok(v) => serde_json::to_string(&v).unwrap_or_else(|_| "{}".to_string()),
        Err(e) => json!({ "error": e.to_string() }).to_string(),
    }
}

/// POST /v1/skills/web/search on the gateway (Brave Search API, node side).
pub async fn web_search_skill(
    cfg: &crate::config::BotConfig,
    query: &str,
    count: Option<u64>,
    freshness: Option<&str>,
) -> Result<Value> {
    if query.trim().is_empty() {
        anyhow::bail!("empty search query");
    }
    let mut body = json!({ "query": query });
    if let Some(c) = count {
        body["count"] = json!(c);
    }
    if let Some(f) = freshness {
        body["freshness"] = json!(f);
    }
    post_gateway_skill(
        cfg,
        "/v1/skills/web/search",
        &body,
    )
    .await
}

/// POST /v1/skills/web/fetch on the gateway (URL download + text extract).
pub async fn web_fetch_skill(
    cfg: &crate::config::BotConfig,
    url: &str,
    max_chars: Option<usize>,
) -> Result<Value> {
    if url.trim().is_empty() {
        anyhow::bail!("empty url");
    }
    let mut body = json!({ "url": url });
    if let Some(m) = max_chars {
        body["max_chars"] = json!(m);
    }
    post_gateway_skill(cfg, "/v1/skills/web/fetch", &body).await
}

/// Shared body for the two web skill calls: gateway JSON POST + `error`
/// extraction on non-2xx.
async fn post_gateway_skill(
    cfg: &crate::config::BotConfig,
    path: &str,
    body: &Value,
) -> Result<Value> {
    let url = format!("{}{path}", cfg.gateway_url);
    let resp = client()
        .post(&url)
        .json(body)
        .send()
        .await
        .with_context(|| format!("gateway unreachable at {url}"))?;
    let status = resp.status();
    let body: Value = resp.json().await.context("gateway returned non-JSON")?;
    if !status.is_success() {
        anyhow::bail!("{path} failed: {}", body_error(&body, status));
    }
    Ok(body)
}

/// POST /v1/skills/memory/embed — store text in the node's private memory.
pub async fn memory_embed(cfg: &crate::config::BotConfig, text: &str) -> Result<(String, String)> {
    let url = format!("{}/v1/skills/memory/embed", cfg.gateway_url);
    let resp = client()
        .post(&url)
        .json(&json!({ "text": text }))
        .send()
        .await
        .with_context(|| format!("gateway unreachable at {url}"))?;
    let status = resp.status();
    let body: Value = resp.json().await.context("gateway returned non-JSON")?;
    if !status.is_success() {
        anyhow::bail!("memory.embed failed: {}", body_error(&body, status));
    }
    let id = body
        .get("id")
        .and_then(value_to_string)
        .unwrap_or_else(|| "unknown".to_string());
    let provider = body
        .get("provider")
        .and_then(value_to_string)
        .unwrap_or_else(|| "unknown".to_string());
    Ok((id, provider))
}

/// POST /v1/skills/memory/search — vector search over the node's memory.
pub async fn memory_search(
    cfg: &crate::config::BotConfig,
    query: &str,
    top_k: u64,
) -> Result<Vec<Value>> {
    let url = format!("{}/v1/skills/memory/search", cfg.gateway_url);
    let resp = client()
        .post(&url)
        .json(&json!({ "query": query, "top_k": top_k }))
        .send()
        .await
        .with_context(|| format!("gateway unreachable at {url}"))?;
    let status = resp.status();
    let body: Value = resp.json().await.context("gateway returned non-JSON")?;
    if !status.is_success() {
        anyhow::bail!("memory.search failed: {}", body_error(&body, status));
    }
    Ok(body
        .get("hits")
        .and_then(|h| h.as_array())
        .cloned()
        .unwrap_or_default())
}

/// GET /status on the gateway — node snapshot for /status and /peers.
pub async fn gateway_status(cfg: &crate::config::BotConfig) -> Result<Value> {
    let url = format!("{}/status", cfg.gateway_url);
    let resp = client()
        .get(&url)
        .send()
        .await
        .with_context(|| format!("gateway unreachable at {url}"))?;
    let status = resp.status();
    let body: Value = resp.json().await.context("gateway returned non-JSON")?;
    if !status.is_success() {
        anyhow::bail!("status failed: {}", body_error(&body, status));
    }
    Ok(body)
}

/// Outcome of a /mine attempt — enough to render both the direct reply and
/// an optional announcement post.
#[derive(Debug)]
pub struct MineOutcome {
    pub won: bool,
    pub message: String,
    pub amount: Option<String>,
    /// Reward in micro-JER as reported by the chain node (`reward_uj`), when
    /// it supplies it — used to credit the bot-side /leaderboard.
    pub reward_uj: Option<u64>,
}

/// POST /mine on the local JerithChain node (bearer-token auth via
/// keys/api.token or JERITH_TOKEN, same as the passive miner).
pub async fn mine(
    cfg: &crate::config::BotConfig,
    discord_id: &str,
    channel_id: &str,
) -> Result<MineOutcome> {
    let token = crate::core_token()?;
    let url = format!("{}/mine", cfg.chain_url);
    let message_id = format!("discord-bot:{discord_id}:{}", now_ms());
    let resp = client()
        .post(&url)
        .bearer_auth(token)
        .json(&json!({
            "discord_id": discord_id,
            "source": "discord",
            "message_id": message_id,
            "guild_id": "",
            "channel_id": channel_id,
        }))
        .send()
        .await
        .with_context(|| format!("JerithChain node unreachable at {url}"))?;
    let status = resp.status();
    let body: Value = resp.json().await.context("chain returned non-JSON")?;
    if !status.is_success() {
        anyhow::bail!("mine failed: {}", body_error(&body, status));
    }
    let won = body.get("ok").and_then(|o| o.as_bool()).unwrap_or(false);
    if won {
        // Prefer the exact reward_uj; fall back to parsing amount strings.
        let reward_uj = body
            .get("reward_uj")
            .and_then(|v| v.as_u64())
            .or_else(|| {
                body.get("reward_jer")
                    .and_then(|v| v.as_f64())
                    .map(|f| (f * 1_000_000.0) as u64)
            });
        let amount = reward_uj
            .map(|u| {
                let jer = u as f64 / 1_000_000.0;
                if jer == (jer as u64) as f64 {
                    format!("{} JER", jer as u64)
                } else {
                    format!("{jer:.6} JER")
                }
            })
            .or_else(|| {
                body.get("amount")
                    .and_then(value_to_string)
                    .filter(|a| !a.is_empty())
            });
        let message = match &amount {
            Some(a) => format!("⛏ mined {a}! 🎉"),
            None => "⛏ mined JER! 🎉".to_string(),
        };
        Ok(MineOutcome {
            won,
            message,
            amount,
            reward_uj,
        })
    } else {
        let reason = body
            .get("reason")
            .and_then(value_to_string)
            .unwrap_or_else(|| "not this time".to_string());
        Ok(MineOutcome {
            won,
            message: format!("⛏ mine attempt: {reason}"),
            amount: None,
            reward_uj: None,
        })
    }
}

/// One chain /mystate row (top user by total mined).
#[derive(Debug, Clone, PartialEq)]
pub struct ChainLbEntry {
    pub discord_id: u64,
    pub wins: u64,
    pub total_mined_uj: u64,
}

/// Chain /balance for one Discord user.
#[derive(Debug, Clone, PartialEq)]
pub struct MyBalance {
    pub exists: bool,
    pub balance_uj: u64,
    pub address: Option<String>,
    pub available_jer: Option<f64>,
    pub opted_in: Option<bool>,
}

/// GET /balance on the chain node for the caller's Discord id.
pub async fn my_balance(cfg: &crate::config::BotConfig, token: &str, discord_id: u64) -> Result<MyBalance> {
    let url = format!("{}/balance?discord_id={}", cfg.chain_url, discord_id);
    let resp = client()
        .get(&url)
        .bearer_auth(token)
        .send()
        .await
        .with_context(|| format!("chain node unreachable at {url}"))?;
    let status = resp.status();
    let body: Value = resp.json().await.context("chain returned non-JSON")?;
    if !status.is_success() {
        anyhow::bail!("balance failed: {}", body_error(&body, status));
    }
    let exists = body.get("exists").and_then(|v| v.as_bool()).unwrap_or(false);
    Ok(MyBalance {
        exists,
        balance_uj: body.get("balance_uj").and_then(|v| v.as_u64()).unwrap_or(0),
        address: body
            .get("address")
            .and_then(value_to_string),
        available_jer: body.get("available_jer").and_then(|v| v.as_f64()),
        opted_in: body.get("opted_in").and_then(|v| v.as_bool()),
    })
}

/// GET /mystate on the chain node — chain-authoritative leaderboard totals
/// (per-user lifetime, includes passive mining the bot never observed).
pub async fn chain_leaderboard(
    cfg: &crate::config::BotConfig,
    token: Option<String>,
    top: usize,
) -> Result<Vec<ChainLbEntry>> {
    let token = match token {
        Some(t) => t,
        None => return Err(anyhow::anyhow!("no chain api token")),
    };
    let url = format!("{}/mystate?limit={}", cfg.chain_url, top.min(200));
    let resp = client()
        .get(&url)
        .bearer_auth(token)
        .send()
        .await
        .with_context(|| format!("chain node unreachable at {url}"))?;
    let status = resp.status();
    let body: Value = resp.json().await.context("chain returned non-JSON")?;
    if !status.is_success() {
        anyhow::bail!("mystate failed: {}", body_error(&body, status));
    }
    let mut out = Vec::new();
    for e in body
        .get("entries")
        .and_then(|x| x.as_array())
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .take(top)
    {
        // Pass the raw string through; a non-numeric discord id just never
        // matches any Discord user later.
        let did: u64 = e
            .get("discord_id")
            .and_then(|v| v.as_str().map(|s| s.to_string()).or_else(|| v.as_u64().map(|n| n.to_string())))
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        if did == 0 {
            continue;
        }
        out.push(ChainLbEntry {
            discord_id: did,
            wins: e.get("wins").and_then(|v| v.as_u64()).unwrap_or(0),
            total_mined_uj: e.get("total_mined_uj").and_then(|v| v.as_u64()).unwrap_or(0),
        });
    }
    Ok(out)
}

/// Announcement text for a win in the configured channel, or `None` when not
/// a win. Formatted here so both the reply and the announcement stay in sync.
pub fn mine_announcement(outcome: &MineOutcome, user: &str) -> Option<String> {
    if !outcome.won {
        return None;
    }
    let amount = outcome
        .amount
        .as_deref()
        .map(|a| format!(" ({a})"))
        .unwrap_or_default();
    Some(format!("⛏ **{user}** mined JER{amount}! 🎉"))
}

fn body_error(body: &Value, status: reqwest::StatusCode) -> String {
    body.get("error")
        .and_then(value_to_string)
        .unwrap_or_else(|| status.to_string())
}

fn value_to_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Null => None,
        other => Some(other.to_string()),
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_messages_places_persona_first_without_dupes() {
        let history = vec![
            json!({"role": "user", "content": "hi"}),
            json!({"role": "assistant", "content": "hey"}),
        ];
        let out = build_messages(&history, "who are you?", Some("You are Jerith.\n"));
        assert_eq!(out.len(), 4);
        assert_eq!(out[0]["role"], json!("system"));
        assert_eq!(out[0]["content"], json!("You are Jerith."));
        assert_eq!(out[1]["role"], json!("user"));
        assert_eq!(out[2]["role"], json!("assistant"));
        assert_eq!(out[3]["role"], json!("user"));
        assert_eq!(out[3]["content"], json!("who are you?"));
    }

    #[test]
    fn build_messages_without_persona_skips_system() {
        let out = build_messages(&[], "asf", None);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["role"], json!("user"));
        // Whitespace-only persona is treated as absent.
        let out = build_messages(&[], "asf", Some("   "));
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn prompt_web_detection_matches_live_incident_shapes() {
        // The live incident: a pasted story share must NOT get tools.
        let story = "Since you remembered that Noodly likes vampires, here is a nice little short story in that vein (pun not intended).\n\nThe night was old and tired when he first noticed her. She sat at the far end of the club, in the shadows where the cigarette smoke pooled thickest. Dark hair, pale skin, eyes that caught the dim light and held it. \"Can I join you?\" he said, sliding onto the stool beside her. She smiled. Her teeth were sharp. \"I'm Elena,\" she said. He told her his name was Joe. He told her about his job, his apartment, the loneliness that had settled into his bones like sediment.";
        assert!(!prompt_requests_web(story));
        // Questions/requests do.
        assert!(prompt_requests_web("what is the latest jerithchain height?"));
        assert!(prompt_requests_web("search for P2P mesh papers"));
        assert!(prompt_requests_web("Noodly: how does isostatic pressing work?"));
        assert!(!prompt_requests_web("write me a haiku about kilns"));
        assert!(!prompt_requests_web("summarize this thread"));
        assert!(!prompt_requests_web(""));
        // Short non-question prompts still count as requests (lenient).
        assert!(prompt_requests_web("jerk chicken recipe"));
    }

    #[test]
    fn degenerate_answers_are_caught_and_real_answers_pass() {
        assert!(answer_is_degenerate(""));
        assert!(answer_is_degenerate("   \n"));
        assert!(answer_is_degenerate("{}"));
        assert!(answer_is_degenerate("{ \"tool\": \"web_search\" }"));
        assert!(!answer_is_degenerate("{ \"content\": \"A real answer.\" }"));
        assert!(!answer_is_degenerate("The kiln fires at 980 degrees."));
        assert!(!answer_is_degenerate("{} is an empty set, in plain prose."));
    }

    #[test]
    fn content_intent_shapes() {
        let intent = |content: &str| {
            content_tool_intent(&json!({
                "choices": [{"message": {"role": "assistant", "content": content}}]
            }))
        };
        // Canonical shape seen live on qwen3:8b via the LiteLLM/Ollama runtime.
        let (name, args) = intent(
            "{\"function_name\": \"web_search\", \"arguments\": {\"query\": \"kiln\"}}",
        )
        .unwrap();
        assert_eq!(name, "web_search");
        assert_eq!(args["query"], json!("kiln"));
        // Fenced + narrated + web_fetch + string-encoded arguments.
        let (name, args) = intent(
            "Sure, checking.\n```json\n{\"function_name\": \"web_fetch\", \"arguments\": \"{\\\"url\\\": \\\"https://x\\\"}\"}\n```\n",
        )
        .unwrap();
        assert_eq!(name, "web_fetch");
        assert_eq!(args["url"], json!("https://x"));
        // Unknown function names are ignored — only declared web tools.
        assert!(intent(
            "{\"function_name\": \"run_shell\", \"arguments\": {}}"
        )
        .is_none());
        // Not JSON at all → no intent.
        assert!(intent("I'll just answer directly.").is_none());
    }

    #[test]
    fn model_resolution_pulls_advertised_model() {
        assert_eq!(
            model_from_status(&json!({"model": "qwen3:8b", "node_id": "n"})),
            Some("qwen3:8b".to_string())
        );
        assert_eq!(model_from_status(&json!({"model": "  "})), None);
        assert_eq!(model_from_status(&json!({})), None);
        assert_eq!(model_from_status(&json!({"model": 42})), None);
    }

    #[test]
    fn my_balance_parsing_shapes() {
        // The render side lives in commands.rs; here pin the field extraction
        // from the two documented /balance response shapes.
        let existing = serde_json::json!({
            "exists": true, "address": "JERDC9695",
            "balance_uj": 50_000_000u64, "balance_jer": 50.0,
            "available_jer": 49.9, "opted_in": true
        });
        let got = MyBalance {
            exists: existing.get("exists").and_then(|v| v.as_bool()).unwrap_or(false),
            balance_uj: existing.get("balance_uj").and_then(|v| v.as_u64()).unwrap_or(0),
            address: existing.get("address").and_then(value_to_string),
            available_jer: existing.get("available_jer").and_then(|v| v.as_f64()),
            opted_in: existing.get("opted_in").and_then(|v| v.as_bool()),
        };
        assert!(got.exists);
        assert_eq!(got.balance_uj, 50_000_000);
        assert_eq!(got.address.as_deref(), Some("JERDC9695"));

        let none = serde_json::json!({"exists": false, "balance_uj": 0, "balance_jer": 0.0});
        let got2 = MyBalance {
            exists: none.get("exists").and_then(|v| v.as_bool()).unwrap_or(false),
            balance_uj: none.get("balance_uj").and_then(|v| v.as_u64()).unwrap_or(0),
            address: none.get("address").and_then(value_to_string),
            available_jer: none.get("available_jer").and_then(|v| v.as_f64()),
            opted_in: none.get("opted_in").and_then(|v| v.as_bool()),
        };
        assert!(!got2.exists);
        assert!(got2.address.is_none());
    }

    #[tokio::test]
    async fn my_balance_unreachable_is_err() {
        let cfg = crate::config::BotConfig::from_parts("http://127.0.0.1:3", "http://127.0.0.1:4");
        assert!(my_balance(&cfg, "t", 12345).await.is_err());
    }

    #[tokio::test]
    async fn chain_leaderboard_unreachable_is_err() {
        let cfg = crate::config::BotConfig::from_parts("http://127.0.0.1:3", "http://127.0.0.1:4");
        let res = chain_leaderboard(&cfg, Some("tok".to_string()), 10).await;
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn chain_leaderboard_requires_token() {
        let cfg = crate::config::BotConfig::from_parts("http://127.0.0.1:3", "http://127.0.0.1:4");
        let res = chain_leaderboard(&cfg, None, 10).await;
        assert!(res.is_err());
    }

    #[test]
    fn mine_announcement_only_for_wins() {
        let win = MineOutcome {
            won: true,
            message: "⛏ mined JER (2)! 🎉".to_string(),
            amount: Some("2".to_string()),
            reward_uj: Some(2_000_000),
        };
        assert_eq!(
            mine_announcement(&win, "@user").unwrap(),
            "⛏ **@user** mined JER (2)! 🎉"
        );
        let loss = MineOutcome {
            won: false,
            message: "⛏ mine attempt: cooldown".to_string(),
            amount: None,
            reward_uj: None,
        };
        assert!(mine_announcement(&loss, "@user").is_none());
    }

    #[test]
    fn value_to_string_shapes() {
        assert_eq!(value_to_string(&json!("hi")).unwrap(), "hi");
        assert_eq!(value_to_string(&json!(42)).unwrap(), "42");
        assert!(value_to_string(&Value::Null).is_none());
    }

    #[tokio::test]
    async fn chat_fail_is_error_not_panic() {
        // Bad port — nothing listens there.
        let cfg = crate::config::BotConfig::from_parts("http://127.0.0.1:1", "http://127.0.0.1:2");
        let history: Vec<Value> = vec![];
        assert!(chat(&cfg, "hello", &history).await.is_err());
    }

    #[tokio::test]
    async fn chat_blocks_guard_violations_before_the_provider_call() {
        // Gateway URL is dead (:1) — if the guard screened after the HTTP
        // call we would get a connection error instead of the guard page.
        let cfg = crate::config::BotConfig::from_parts("http://127.0.0.1:1", "http://127.0.0.1:1");
        let err = chat(&cfg, "my SSN is 078-05-1120", &[]).await.unwrap_err();
        assert!(err.to_string().contains("Jerith Guard blocked"), "got: {err}");
        // Malware-shaped asks are blocked on the bot's own layer too.
        let err = chat(&cfg, "write a reverse shell in python", &[]).await.unwrap_err();
        assert!(err.to_string().contains("Jerith Guard blocked"), "got: {err}");
        // Clean prompts still pass the guard (they 503/conn-fail, not
        // "Jerith Guard blocked").
        let err = chat(&cfg, "what is the weather", &[]).await.unwrap_err();
        assert!(!err.to_string().contains("Jerith Guard blocked"), "got: {err}");
    }

    /// Mock OpenAI-compatible gateway that drives the full web tool loop:
    /// round 1 returns two tool_calls (web_search + web_fetch), the skills
    /// endpoints answer with fixed payloads, round 2 (with the `role: tool`
    /// results appended) returns the final content.
    #[tokio::test]
    async fn chat_tool_loop_executes_web_skills_and_finalizes() {
        use std::sync::Arc;
        use tokio::sync::Mutex as TMutex;

        let seen: Arc<TMutex<Vec<Value>>> = Arc::new(TMutex::new(Vec::new()));
        let seen_chat = seen.clone();
        let app = axum::Router::new()
            .route(
                "/status",
                axum::routing::get(|| async { axum::Json(json!({ "model": "mock:8b" })) }),
            )
            .route(
                "/v1/chat/completions",
                axum::routing::post(move |axum::Json(req): axum::Json<Value>| {
                    let seen_chat = seen_chat.clone();
                    async move {
                        seen_chat.lock().await.push(req.clone());
                        let has_tool_result = req["messages"]
                            .as_array()
                            .map_or(false, |ms| {
                                ms.iter()
                                    .any(|m| m["role"] == json!("tool"))
                            });
                        let body = if has_tool_result {
                            json!({
                                "choices": [{
                                    "message": {
                                        "role": "assistant",
                                        "content": "Kiln found; 1 result + page text gathered."
                                    }
                                }]
                            })
                        } else {
                            json!({
                                "choices": [{
                                    "message": {
                                        "role": "assistant",
                                        "content": Value::Null,
                                        "tool_calls": [
                                            {"id": "c1", "type": "function", "function": {
                                                "name": "web_search",
                                                "arguments": "{\"query\": \"isopress kiln\", \"count\": 2}"
                                            }},
                                            {"id": "c2", "type": "function", "function": {
                                                "name": "web_fetch",
                                                "arguments": "{\"url\": \"https://example.com/kiln\"}"
                                            }}
                                        ]
                                    }
                                }]
                            })
                        };
                        (axum::http::StatusCode::OK, axum::Json(body))
                    }
                }),
            )
            .route(
                "/v1/skills/web/search",
                axum::routing::post(|| async {
                    axum::Json(json!({
                        "provider": "brave",
                        "mode": "web",
                        "count": 1,
                        "results": [
                            {"title": "K", "url": "https://example.com/kiln", "description": "d"}
                        ]
                    }))
                }),
            )
            .route(
                "/v1/skills/web/fetch",
                axum::routing::post(|| async {
                    axum::Json(json!({
                        "url": "https://example.com/kiln",
                        "title": "Kiln",
                        "text": "kiln spec text",
                        "truncated": false,
                        "status": 200
                    }))
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let mut cfg =
            crate::config::BotConfig::from_parts(&format!("http://{addr}"), "http://127.0.0.1:1");
        cfg.web.enabled = true;

        let (reply, served_by) = chat(&cfg, "find the kiln", &[]).await.expect("tool-loop chat");
        assert!(reply.contains("Kiln found"), "got {reply}");
        assert!(served_by.is_none());

        let seen = seen.lock().await;
        assert!(seen.len() >= 2, "two chat rounds expected: {seen:?}");
        assert!(seen[0].get("tools").is_some(), "round 1 must declare web tools");
        let tool_round = seen
            .iter()
            .find(|r| {
                r["messages"]
                    .as_array()
                    .map_or(false, |ms| ms.iter().any(|m| m["role"] == json!("tool")))
            })
            .expect("a round carrying tool results");
        let tool_msgs: Vec<&Value> = tool_round["messages"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|m| m["role"] == json!("tool"))
            .collect();
        assert_eq!(tool_msgs.len(), 2, "both tool calls answered: {tool_msgs:?}");
        let both = tool_msgs
            .iter()
            .map(|m| m["content"].as_str().unwrap_or_default())
            .collect::<String>();
        assert!(
            both.contains("\"title\":\"K\"") && both.contains("kiln spec text"),
            "search + fetch results flowed back: {both}"
        );
    }

    /// Runtime answers tool intents as plain JSON content (qwen3:8b via
    /// LiteLLM/Ollama): the loop must accept and execute them too.
    #[tokio::test]
    async fn chat_tool_loop_accepts_json_intent_from_runtime() {
        let app = axum::Router::new()
            .route(
                "/status",
                axum::routing::get(|| async { axum::Json(json!({ "model": "mock:8b" })) }),
            )
            .route(
                "/v1/chat/completions",
                axum::routing::post(move |axum::Json(req): axum::Json<Value>| async move {
                    let has_tool_result = req["messages"].as_array().map_or(false, |ms| {
                        ms.iter().any(|m| m["role"] == json!("tool"))
                    });
                    if has_tool_result {
                        axum::Json(json!({
                            "choices": [{"message": {
                                "role": "assistant",
                                "content": "Answer from the gathered results."
                            }}]
                        }))
                    } else {
                        // Real observed output shape (content IS the intent).
                        axum::Json(json!({
                            "choices": [{"message": {
                                "role": "assistant",
                                "content": "{\"function_name\": \"web_search\", \"arguments\": {\"query\": \"kiln\"}}"
                            }}]
                        }))
                    }
                }),
            )
            .route(
                "/v1/skills/web/search",
                axum::routing::post(|| async {
                    axum::Json(json!({"results": [{"title": "K", "url": "https://k"}]}))
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let mut cfg =
            crate::config::BotConfig::from_parts(&format!("http://{addr}"), "http://127.0.0.1:1");
        cfg.web.enabled = true;
        let (reply, _) = chat(&cfg, "find kiln", &[]).await.expect("intent loop chat");
        assert_eq!(reply, "Answer from the gathered results.");
    }

    /// A runtime that rejects the tools array wholesale is retried once
    /// toolless instead of failing the chat.
    #[tokio::test]
    async fn chat_tool_loop_degrades_to_toolless_on_runtime_rejection() {
        use axum::response::IntoResponse as _;
        let app = axum::Router::new()
            .route(
                "/status",
                axum::routing::get(|| async { axum::Json(json!({ "model": "mock:8b" })) }),
            )
            .route(
                "/v1/chat/completions",
                axum::routing::post(move |axum::Json(req): axum::Json<Value>| async move {
                    if req.get("tools").is_some() {
                        (axum::http::StatusCode::BAD_REQUEST, axum::Json(json!(
                            {"error": "model not supported: tools"})))
                            .into_response()
                    } else {
                        (
                            axum::http::StatusCode::OK,
                            axum::Json(json!({
                                "choices": [{"message": {"role": "assistant", "content": "plain no-tools answer"}}]
                            })),
                        )
                            .into_response()
                    }
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let mut cfg =
            crate::config::BotConfig::from_parts(&format!("http://{addr}"), "http://127.0.0.1:1");
        cfg.web.enabled = true;
        let (reply, _served_by) = chat(&cfg, "hi there", &[]).await.expect("toolless fallback");
        assert_eq!(reply, "plain no-tools answer");
    }
}
