//! Gateway + chain HTTP calls the slash commands make.
//!
//! Kept as plain async functions (no poise types) so they are unit-testable
//! and reusable. Everything talks OpenAI-ish shapes to the gateway.

use anyhow::{Context as _, Result};
use serde_json::{Value, json};

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
static RESOLVED_MODELS: tokio::sync::Mutex<Option<std::collections::HashMap<String, String>>> =
    tokio::sync::Mutex::const_new(None);

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
    let body: Value = resp
        .json()
        .await
        .context("model status: non-JSON body")
        .ok()?;
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

pub(crate) fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .use_rustls_tls()
        .timeout(TIMEOUT)
        .build()
        .expect("static reqwest client config")
}

// ---------------------------------------------------------------------------
// Streamed chat rounds (gateway SSE passthrough)
// ---------------------------------------------------------------------------

/// Per-round event for live taps (live-typing sinks). Fired in order:
/// `RoundStart` at each chat round ('self-retries on degenerate answers
/// included), then `Text` fragments of that round's SSE content.
pub enum LiveEvent<'a> {
    /// A new chat round begins — sinks should CLEAR whatever the previous
    /// round typed: web-search tool rounds type nothing (native tool_call
    /// deltas have no content), but content-JSON tool intent slop does
    /// appear briefly; a RoundStart on the answer round erases it so the
    /// user sees only the real answer typed out.
    RoundStart,
    /// A content fragment of the current round.
    Text(&'a str),
}

/// Type of the live tap passed to `chat_with_usage_live`. HRTB over the
/// fragment lifetime — the api folds fragments across multiple rounds,
/// each with its own borrow.
pub type LiveTap = dyn for<'a> Fn(LiveEvent<'a>) + Send + Sync;

/// Per-round accumulation for a streamed chat completion.
#[derive(Default)]
struct RoundAcc {
    content: String,
    /// OpenAI-shape tool calls accumulated from streamed fragments
    /// (delta.tool_calls pieces arrive index-keyed, name once + argument
    /// fragments incrementally).
    tool_calls: Vec<Value>,
    finish_reason: Option<String>,
    usage_total: u64,
    mesh_fallback: Option<String>,
    /// Set when the SSE stream broke mid-generation but real content had
    /// already arrived: the partial answer is kept (annotated) instead of
    /// failing the whole chat.
    cut: Option<String>,
}

impl RoundAcc {
    /// Materialize the same JSON shape the buffered path produces, so the
    /// loop's single set of downstream logic (tool intent, degenerate
    /// check, usage extraction, mesh_fallback attribution) stays the one
    /// source of truth regardless of transport.
    fn into_pseudo_body(self) -> Value {
        let mut content = self.content.clone();
        if let Some(cut) = &self.cut
            && !content.trim().is_empty()
        {
            content.push_str(&format!("\n\n_(stream cut: {cut})_"));
        }
        let mut message = json!({"role": "assistant", "content": content});
        if !self.tool_calls.is_empty() {
            message["tool_calls"] = Value::Array(self.tool_calls.clone());
        }
        let mut body = json!({
            "choices": [{
                "message": message,
                "finish_reason": self.finish_reason.unwrap_or_else(|| "stop".to_string()),
            }]
        });
        if self.usage_total > 0 {
            body["usage"] = json!({"total_tokens": self.usage_total});
        }
        if let Some(mf) = &self.mesh_fallback {
            body["mesh_fallback"] = json!(mf);
        }
        body
    }
}

/// Fold one `data: {…}` SSE line into the accumulator (pure — unit tests
/// cover the chunk shapes litellm/ollama and the gateway's mesh-fallback
/// wrap actually emit).
fn note_sse_chunk(acc: &mut RoundAcc, line: &str) {
    let trimmed = line.trim();
    let Some(data) = trimmed.strip_prefix("data:") else {
        return; // SSE comments, event: rows, blank separators
    };
    let data = data.trim();
    if data == "[DONE]" || data.is_empty() {
        return;
    }
    let Ok(j) = serde_json::from_str::<Value>(data) else {
        return; // keepalives / non-JSON heartbeats never kill a stream
    };
    if let Some(mf) = j
        .get("mesh_fallback")
        .and_then(value_to_string)
        .filter(|s| !s.is_empty())
    {
        acc.mesh_fallback = Some(mf);
    }
    if let Some(total) = j.pointer("/usage/total_tokens").and_then(Value::as_u64) {
        acc.usage_total = total; // trailing usage chunk carries the round total
    }
    let Some(choice) = j.pointer("/choices/0") else {
        return; // usage-only chunks have no choice
    };
    if let Some(fr) = choice
        .get("finish_reason")
        .and_then(value_to_string)
        .filter(|s| !s.is_empty())
    {
        acc.finish_reason = Some(fr);
    }
    if let Some(c) = choice
        .pointer("/delta/content")
        .and_then(value_to_string)
        .filter(|s| !s.is_empty())
    {
        acc.content.push_str(&c);
    }
    if let Some(frags) = choice
        .pointer("/delta/tool_calls")
        .and_then(Value::as_array)
    {
        for f in frags {
            let idx = f.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
            while acc.tool_calls.len() <= idx {
                acc.tool_calls.push(json!({
                    "id": String::new(),
                    "type": "function",
                    "function": {"name": String::new(), "arguments": String::new()},
                }));
            }
            let slot = acc.tool_calls.get_mut(idx).expect("index-grown above");
            if let Some(id) = f
                .get("id")
                .and_then(value_to_string)
                .filter(|s| !s.is_empty())
            {
                slot["id"] = json!(id);
            }
            if let Some(name) = f
                .pointer("/function/name")
                .and_then(value_to_string)
                .filter(|s| !s.is_empty())
            {
                slot["function"]["name"] = json!(name);
            }
            if let Some(args) = f
                .pointer("/function/arguments")
                .and_then(value_to_string)
                .filter(|s| !s.is_empty())
            {
                let prefix = slot["function"]["arguments"].as_str().unwrap_or("");
                slot["function"]["arguments"] = json!(format!("{prefix}{args}"));
            }
        }
    }
}

/// One streamed chat round: POST with `stream: true`, fold OpenAI deltas
/// as the gateway passes them through (the live tap fires with each new
/// content fragment), and return the same `(status, body)` tuple the
/// buffered path produces. Plain-JSON answers (error rows, non-SSE stacks)
/// parse exactly the buffered way. Transport breaks MID-stream with real
/// content already folded degrade to a cut-marked partial instead of
/// failing the chat; an empty side of the same failure still errors.
async fn fetch_streamed_round(
    url: &str,
    payload: &Value,
    timeout: std::time::Duration,
    live: Option<&LiveTap>,
) -> Result<(reqwest::StatusCode, Value)> {
    use futures_util::StreamExt;

    let mut payload = payload.clone();
    payload["stream"] = json!(true);
    payload["stream_options"] = json!({"include_usage": true});

    let resp = client()
        .post(url)
        .timeout(timeout)
        .json(&payload)
        .send()
        .await
        .with_context(|| format!("gateway unreachable at {url}"))?;
    let status = resp.status();

    let is_sse = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.contains("text/event-stream"))
        .unwrap_or(false);
    if !is_sse {
        // Plain JSON: parse exactly like the buffered path (errors and
        // non-streaming stacks included) and let the shared loop decide.
        let body: Value = resp.json().await.context("gateway returned non-JSON")?;
        return Ok((status, body));
    }

    let mut acc = RoundAcc::default();
    let mut bytes = resp.bytes_stream();
    let mut buf = String::new();
    while let Some(chunk) = bytes.next().await {
        match chunk {
            Ok(b) => {
                buf.push_str(&String::from_utf8_lossy(&b));
                while let Some(nl) = buf.find('\n') {
                    let line: String = buf.drain(..=nl).collect();
                    let before = acc.content.len();
                    note_sse_chunk(&mut acc, &line);
                    if let Some(tap) = live {
                        let fresh = &acc.content[before.min(acc.content.len())..];
                        if !fresh.is_empty() {
                            tap(LiveEvent::Text(fresh));
                        }
                    }
                }
            }
            Err(e) => {
                if !acc.content.trim().is_empty() {
                    acc.cut = Some(e.to_string());
                    break;
                }
                return Err(anyhow::Error::from(e))
                    .context(format!("gateway stream broke at {url}"));
            }
        }
    }
    Ok((status, acc.into_pseudo_body()))
}

/// POST /v1/chat/completions on the gateway. Returns the assistant text plus
/// the `mesh_fallback` node id when the mesh (not the local runtime) served it.
/// Message assembly for every chat shape the bot produces (`/ask`, DM and
/// @mention fallbacks share this path). A persona system prompt goes first,
/// then the conversation history, then the caller's prompt. Pure fn — tests
/// cover ordering without a network or framework.
pub fn build_messages(history: &[Value], prompt: &str, persona: Option<&str>) -> Vec<Value> {
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
    chat_with_usage_live(cfg, prompt, raw_history, None).await
}

/// Chat with token accounting AND a live tap: when `[chat] stream` is on
/// (the default), every round streams through the gateway's SSE
/// passthrough and `live` fires with a `RoundStart` marker then each new
/// content fragment — the @mention/DM path and `/ask` live-type the reply
/// into Discord while tokens land, web-search tool rounds included
/// (RoundStart clears the tool-exchange round's text before the answer
/// round types). `None` (monitors/cost accounting) keeps the buffered reply.
pub async fn chat_with_usage_live(
    cfg: &crate::config::BotConfig,
    prompt: &str,
    raw_history: &[Value],
    live: Option<&LiveTap>,
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
                Err(verdict) => {
                    // FEATURES §8: observers hear the block reason — never the
                    // blocked content itself.
                    crate::hooks::global().emit(crate::hooks::Event::new(
                        crate::hooks::EV_GUARD_VIOLATION,
                        json!({ "reason": verdict.block_reason() }),
                    ));
                    anyhow::bail!(
                        "🛡️ Jerith Guard blocked this request ({}). Edit the prompt and try again.",
                        verdict.block_reason()
                    )
                }
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
    if let Some(pg) = cfg.postgres.as_ref().filter(|p| p.enabled)
        && let Some(discord_id) = crate::people::extract_speaker_id(prompt)
    {
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

    let url = format!("{}/v1/chat/completions", cfg.gateway_url);
    let model = runtime_model(cfg)
        .await
        .unwrap_or_else(|| MODEL_PLACEHOLDER.to_string());

    // Web access (OpenClaw stack migration): the chat request declares the
    // enabled plugins' skills as OpenAI tools, so a tool-capable runtime can
    // web_search / memory.embed / lore.search mid-answer (the plugin SDK's
    // registry decides what's summonable; see plugins.rs). Unsupported
    // model/runtimes get exactly one toolless retry below — tools degrade,
    // never fail the chat.
    let registry = std::sync::Arc::new(crate::plugins::SkillRegistry::standard(
        cfg.plugins.enabled.clone(),
    ));
    const MAX_TOOL_ROUNDS: usize = 3;
    // Tokens accumulated across rounds (session-cost-tracker port input).
    let mut tokens_used: u64 = 0;
    // Tools only for prompts that read as information requests. A shared
    // story/paste (long, no question) must not make the model invent a
    // web_search for it — observed live: a user posted a short story and
    // the model web-searched a line out of it, then answered "{}".
    let mut tools_enabled = cfg.web.enabled && prompt_requests_web(prompt);

    let mut rounds = 0usize;
    // Per-request bound ([chat] timeout_secs, default 240 s): slow thinking
    // answers are legit — the gateway bounds the local runtime's header wait
    // at 180 s and hands off to mesh fallback, so the bot's bound only needs
    // to outlast that with headroom for a fallback leg, never the 90 s the
    // old shared client constant imposed.
    let chat_timeout = std::time::Duration::from_secs(cfg.chat_timeout_secs);
    loop {
        // Round boundary for live taps: whatever the previous round typed
        // (tool-intent slop, nothing for native tool_call rounds) clears
        // here, before this round's fragments arrive.
        if let Some(tap) = live {
            tap(LiveEvent::RoundStart);
        }
        let mut payload = json!({
            "model": model,
            "messages": messages,
        });
        if tools_enabled {
            payload["tools"] = Value::Array(registry.tool_defs());
        }
        let (status, body) = if cfg.chat_stream {
            fetch_streamed_round(&url, &payload, chat_timeout, live).await?
        } else {
            let resp = client()
                .post(&url)
                .timeout(chat_timeout)
                .json(&payload)
                .send()
                .await
                .with_context(|| format!("gateway unreachable at {url}"))?;
            let status = resp.status();
            let body: Value = resp.json().await.context("gateway returned non-JSON")?;
            (status, body)
        };
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
        if tools_enabled
            && tool_calls.is_empty()
            && rounds < MAX_TOOL_ROUNDS
            && let Some((name, args)) = content_tool_intent(&body, &registry.enabled_skill_names())
        {
            tool_calls = vec![json!({
                "id": format!("iw{rounds}"),
                "type": "function",
                "function": {"name": name, "arguments": args.to_string()},
            })];
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
            // JSON envelope answers (the 5:45 live incident: a plain
            // question answered as `{"status": …, "message": "the prose"}`
            // rendered brace-wrapped into Discord): when the body carries a
            // known text key, surface the prose — keys are metadata. Runs
            // BEFORE the degenerate check, so an envelope whose text field
            // is empty still falls through to the retry/refusal gate.
            let content = match unwrap_json_text_envelope(&content) {
                Some(inner) => {
                    tracing::info!(
                        target: "bot",
                        envelope_len = content.len(),
                        unfolded_len = inner.len(),
                        "unwrapped JSON envelope answer into its text field"
                    );
                    inner
                }
                None => content,
            };
            if answer_is_degenerate(&content) {
                // Journal trace so an operator can confirm the gate fired
                // (the '{'-reply incident was invisible in the logs). Only
                // a short head of the content goes out — a debug aid, not a
                // byte dump of whatever the model emitted.
                tracing::warn!(
                    target: "bot",
                    degenerate_len = content.len(),
                    round = rounds,
                    tools_were_enabled = tools_enabled,
                    degenerate_head = %truncate_degenerate_head(&content),
                    "degenerate answer caught — one toolless retry, then the graceful refusal"
                );
                // One retry even when tools were never on (the bare-`{`
                // case above comes from the plain lane). When tools WERE on,
                // the retry runs toolless: a `{` opener usually means the
                // model was fumbling a tool intent it cannot really make.
                if rounds == 0 {
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
            let id = tc.get("id").and_then(value_to_string).unwrap_or_default();
            let name = tc
                .pointer("/function/name")
                .and_then(value_to_string)
                .unwrap_or_default();
            let raw_args = tc
                .pointer("/function/arguments")
                .cloned()
                .unwrap_or(Value::Null);
            let answer = run_tool(&registry, cfg, &name, &raw_args).await;
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

/// Parse Ollama/OpenAI tool-call arguments: per spec a JSON-encoded string,
/// but tolerate an inline object.
fn parse_tool_args(args: &Value) -> Value {
    match args {
        Value::String(s) => serde_json::from_str(s).unwrap_or_else(|_| json!({})),
        v => v.clone(),
    }
}

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
        "write",
        "summarize",
        "summarise",
        "translate",
        "continue",
        "rewrite",
        "rank",
        "explain",
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
/// Extract the prose from a JSON envelope answer, when the body carries a
/// known text key. Discord-facing chat wants prose: a plain question
/// answered as `{"status": …, "message": "the actual prose"}` renders
/// brace-wrapped and reads as a malfunction even when everything inside
/// is correct. Behavior: object with a non-empty string `message` /
/// `content` / `text` / `answer` field → that string; anything else
/// (non-object, no text key, non-string field, parse failure, brace-only
/// junk) → unchanged, and the degenerate gate handles its share.
pub fn unwrap_json_text_envelope(content: &str) -> Option<String> {
    let t = content.trim();
    if !(t.starts_with('{') && t.ends_with('}')) {
        return None;
    }
    json_text_field(t)
}

/// Parse a brace-bounded body and return its chat-prose field, if any —
/// the one key list the envelope unwrap AND the degenerate check share,
/// so a body the unwrap can salvage is never flagged degenerate (they can
/// never disagree about what "has prose" means).
fn json_text_field(t: &str) -> Option<String> {
    let v = serde_json::from_str::<Value>(t).ok()?;
    let obj = v.as_object()?;
    for key in ["message", "content", "text", "answer"] {
        if let Some(s) = obj.get(key).and_then(Value::as_str) {
            let s = s.trim();
            if !s.is_empty() {
                return Some(s.to_string());
            }
        }
    }
    None
}

/// Short, loggable head of a degenerate answer: bounded, single-line, and
/// safe to print in a journal next to user prompts.
fn truncate_degenerate_head(content: &str) -> String {
    let head: String = content.chars().take(24).collect();
    head.chars()
        .map(|c| {
            if c == '{' || c == '}' {
                c
            } else if c.is_whitespace() {
                ' '
            } else {
                '\u{25CF}' // redacted dot — the shape carries, the bytes don't
            }
        })
        .collect()
}

pub fn answer_is_degenerate(content: &str) -> bool {
    let t = content.trim();
    if t.is_empty() {
        return true;
    }
    // Braces-and-nothing-else is the live incident shape: observed verbatim
    // in Discord as  `{` — a bare opener (a JSON tool-intent the model
    // started and never finished, or a `{}`-style answer the stream cut
    // short). None of these are text; ALL of them rendered as garbage
    // because the old check required a CLOSE brace too (`starts_with('{')
    // && ends_with('}')`), so a lone `{` sailed through to the chat.
    if t.chars().all(|c| c == '{' || c == '}' || c.is_whitespace()) {
        return true;
    }
    if let Some(rest) = t.strip_prefix('{') {
        // Brace-led serde intent: a chat answer opens with `{` + a quoted
        // field (the 05:57 render: `{ "` opening an object the stream never
        // balanced, or `{…} trailing prose'). Renderable only by extracting
        // a text field; everything else is degenerate — retry, then the
        // graceful refusal. Legit prose that merely MENTIONS braces (e.g.
        // `{} is an empty set`) has no quoted field right after the brace
        // and passes through untouched.
        let opens_a_field = rest.trim_start().starts_with('"');
        if opens_a_field && json_text_field(t).is_none() {
            return true;
        }
    }
    // Parseable JSON object with no text payload inside (the original
    // incident: `{}` and `{"function_name": …}` intent slop that is not
    // text the user can read).
    if t.starts_with('{')
        && t.ends_with('}')
        && let Ok(v) = serde_json::from_str::<Value>(t)
    {
        let has_text = ["content", "text", "answer", "message"]
            .iter()
            .find_map(|k| v.get(k).and_then(Value::as_str))
            .map(|s| !s.trim().is_empty())
            .unwrap_or(false);
        return !has_text;
    }
    false
}

/// Detect a model-declared tool intent hidden in the assistant's plain
/// content: `{"function_name": "web_search", "arguments": {…}}` (also
/// "tool"/"name" key variants, fenced code, and prose around the JSON).
/// Returns the function name and parsed arguments. `allowed` is exactly
/// the registry's enabled skill names (`plugins.rs`) — a model can never
/// summon a disabled or unknown tool through here.
fn content_tool_intent(body: &Value, allowed: &[&str]) -> Option<(String, Value)> {
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
    if !allowed.contains(&name) {
        return None;
    }
    let args = v
        .get("arguments")
        .cloned()
        .or_else(|| v.get("parameters").cloned())
        .unwrap_or(json!({}));
    Some((name.to_string(), parse_tool_args(&args)))
}

/// Execute one model-requested skill through the plugin registry and
/// return the answer as the JSON string fed back to the model. Tool
/// errors come back as `{"error": …}` so the model can recover instead
/// of the whole chat failing.
async fn run_tool(
    registry: &crate::plugins::SkillRegistry,
    cfg: &crate::config::BotConfig,
    name: &str,
    raw_args: &Value,
) -> String {
    let args = parse_tool_args(raw_args);
    let outcome = registry.dispatch(cfg, name, &args).await;
    // FEATURES §8: receivers learn the skill name and whether it worked —
    // raw arguments stay local (user text can flow through them).
    crate::hooks::global().emit(crate::hooks::Event::new(
        crate::hooks::EV_SKILL_INVOKED,
        match &outcome {
            Ok(_) => json!({ "skill": name, "ok": true }),
            Err(e) => json!({ "skill": name, "ok": false, "error": e.to_string() }),
        },
    ));
    match outcome {
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
    post_gateway_skill(cfg, "/v1/skills/web/search", &body).await
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
        let reward_uj = body.get("reward_uj").and_then(|v| v.as_u64()).or_else(|| {
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
pub async fn my_balance(
    cfg: &crate::config::BotConfig,
    token: &str,
    discord_id: u64,
) -> Result<MyBalance> {
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
    let exists = body
        .get("exists")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    Ok(MyBalance {
        exists,
        balance_uj: body.get("balance_uj").and_then(|v| v.as_u64()).unwrap_or(0),
        address: body.get("address").and_then(value_to_string),
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
            .and_then(|v| {
                v.as_str()
                    .map(|s| s.to_string())
                    .or_else(|| v.as_u64().map(|n| n.to_string()))
            })
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        if did == 0 {
            continue;
        }
        out.push(ChainLbEntry {
            discord_id: did,
            wins: e.get("wins").and_then(|v| v.as_u64()).unwrap_or(0),
            total_mined_uj: e
                .get("total_mined_uj")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
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
        assert!(prompt_requests_web(
            "what is the latest jerithchain height?"
        ));
        assert!(prompt_requests_web("search for P2P mesh papers"));
        assert!(prompt_requests_web(
            "Noodly: how does isostatic pressing work?"
        ));
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
        // The live Discord incident shape: a BARE opener (no close brace —
        // the old check needed `{…}` on both ends, so this rendered into
        // the chat verbatim). Whitespace-padded and paired variants too.
        assert!(answer_is_degenerate("{"));
        assert!(answer_is_degenerate("{ "));
        assert!(answer_is_degenerate(" { } "));
        assert!(answer_is_degenerate("  {}  "));
        assert!(answer_is_degenerate("}}"));
        // The 05:57 render: an answer that OPENS with `{ "` — JSON intent
        // the stream never balanced (or balanced with trailing prose).
        // Degenerate either way; a chat never leads with a field list.
        assert!(answer_is_degenerate("{ \"status\": \"done\", \"sum\""));
        assert!(answer_is_degenerate(
            "{ \"status\": \"ok\" } here is what happened"
        ));
        assert!(answer_is_degenerate("{\n  \"note\": \"never closed\""));
        // Prose that merely MENTIONS braces stays legit — no quoted field
        // right after the opener.
        assert!(!answer_is_degenerate("{} is an empty set, in plain prose."));
    }

    /// The 5:45 live incident shape: envelope WITH prose in a text field —
    /// not degenerate, but brace-wrapped render is wrong too. Unwrapping
    /// surfaces the prose; everything else passes untouched (and any
    /// bracketless non-JSON stays playable as literal text).
    #[test]
    fn json_envelope_answers_unwrap_their_text_field() {
        let envelope = json!({
            "status": "no_recent_update",
            "message": "No recent information about Glenn Lee's status was found in the node's memory. Would you like me to check other sources?"
        });
        let raw = serde_json::to_string(&envelope).unwrap();
        assert_eq!(
            unwrap_json_text_envelope(&raw).as_deref(),
            Some(
                "No recent information about Glenn Lee's status was found in the node's memory. Would you like me to check other sources?"
            ),
            "the prose reaches the chat, never the braces"
        );

        // Whitespace/newline-padded — the incident rendered the `{` on its
        // own line, so the text brought leading structure. Pretty-printed
        // JSON is the realistic shape (the paste showed the brace, then
        // indented fields, one per line).
        let padded = serde_json::to_string_pretty(&envelope).unwrap();
        assert_eq!(
            unwrap_json_text_envelope(&padded),
            unwrap_json_text_envelope(&raw)
        );

        // Other text keys serve too; first key wins.
        assert_eq!(
            unwrap_json_text_envelope("{\"content\": \"plain\"}").as_deref(),
            Some("plain")
        );
        assert_eq!(
            unwrap_json_text_envelope("{ \"answer\": \"4 bars\" }").as_deref(),
            Some("4 bars")
        );

        // What stays untouched: not an object, no usable text field,
        // non-string field, unterminated JSON, and plain prose.
        assert_eq!(unwrap_json_text_envelope("{}"), None);
        assert_eq!(unwrap_json_text_envelope("{ \"status\": 42 }"), None);
        assert_eq!(unwrap_json_text_envelope("{ \"message\": \"\" }"), None);
        assert_eq!(unwrap_json_text_envelope("{"), None);
        assert_eq!(
            unwrap_json_text_envelope("The kiln fires at 980 degrees."),
            None
        );
        assert_eq!(unwrap_json_text_envelope(""), None);
        // Prose that MENTIONS braces is not an envelope.
        assert_eq!(
            unwrap_json_text_envelope("{} is an empty set, in plain prose."),
            None
        );
    }

    #[test]
    fn content_intent_shapes() {
        let prepackaged =
            crate::plugins::SkillRegistry::standard(crate::plugins::PREPACKAGED_PLUGINS);
        let allowed = prepackaged.enabled_skill_names();
        let intent = |content: &str| {
            content_tool_intent(
                &json!({
                    "choices": [{"message": {"role": "assistant", "content": content}}]
                }),
                &allowed,
            )
        };
        // Canonical shape seen live on qwen3:8b via the LiteLLM/Ollama runtime.
        let (name, args) =
            intent("{\"function_name\": \"web_search\", \"arguments\": {\"query\": \"kiln\"}}")
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
        assert!(intent("{\"function_name\": \"run_shell\", \"arguments\": {}}").is_none());
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
            exists: existing
                .get("exists")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            balance_uj: existing
                .get("balance_uj")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            address: existing.get("address").and_then(value_to_string),
            available_jer: existing.get("available_jer").and_then(|v| v.as_f64()),
            opted_in: existing.get("opted_in").and_then(|v| v.as_bool()),
        };
        assert!(got.exists);
        assert_eq!(got.balance_uj, 50_000_000);
        assert_eq!(got.address.as_deref(), Some("JERDC9695"));

        let none = serde_json::json!({"exists": false, "balance_uj": 0, "balance_jer": 0.0});
        let got2 = MyBalance {
            exists: none
                .get("exists")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
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
        assert!(
            err.to_string().contains("Jerith Guard blocked"),
            "got: {err}"
        );
        // Malware-shaped asks are blocked on the bot's own layer too.
        let err = chat(&cfg, "write a reverse shell in python", &[])
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("Jerith Guard blocked"),
            "got: {err}"
        );
        // Clean prompts still pass the guard (they 503/conn-fail, not
        // "Jerith Guard blocked").
        let err = chat(&cfg, "what is the weather", &[]).await.unwrap_err();
        assert!(
            !err.to_string().contains("Jerith Guard blocked"),
            "got: {err}"
        );
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
                            .is_some_and(|ms| {
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

        let (reply, served_by) = chat(&cfg, "find the kiln", &[])
            .await
            .expect("tool-loop chat");
        assert!(reply.contains("Kiln found"), "got {reply}");
        assert!(served_by.is_none());

        let seen = seen.lock().await;
        assert!(seen.len() >= 2, "two chat rounds expected: {seen:?}");
        assert!(
            seen[0].get("tools").is_some(),
            "round 1 must declare web tools"
        );
        let tool_round = seen
            .iter()
            .find(|r| {
                r["messages"]
                    .as_array()
                    .is_some_and(|ms| ms.iter().any(|m| m["role"] == json!("tool")))
            })
            .expect("a round carrying tool results");
        let tool_msgs: Vec<&Value> = tool_round["messages"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|m| m["role"] == json!("tool"))
            .collect();
        assert_eq!(
            tool_msgs.len(),
            2,
            "both tool calls answered: {tool_msgs:?}"
        );
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
                    let has_tool_result = req["messages"].as_array().is_some_and(|ms| {
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
        let (reply, _) = chat(&cfg, "find kiln", &[])
            .await
            .expect("intent loop chat");
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
        let (reply, _served_by) = chat(&cfg, "hi there", &[])
            .await
            .expect("toolless fallback");
        assert_eq!(reply, "plain no-tools answer");
    }

    // ---------------------------------------------------------------
    // Streamed chat rounds (gateway SSE passthrough)
    // ---------------------------------------------------------------

    #[test]
    fn sse_fold_content_finish_usage() {
        use super::{RoundAcc, note_sse_chunk};
        let mut acc = RoundAcc::default();
        for line in [
            r#"data: {"choices":[{"delta":{"role":"assistant","content":"Hel"}}]}"#,
            ": keepalive comment",
            r#"data: {"choices":[{"delta":{"content":"lo!"}}]}"#,
            r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
            r#"data: {"choices":[],"usage":{"total_tokens":23}}"#,
            "data: [DONE]",
        ] {
            note_sse_chunk(&mut acc, line);
        }
        assert_eq!(acc.content, "Hello!");
        assert_eq!(acc.finish_reason.as_deref(), Some("stop"));
        assert_eq!(acc.usage_total, 23);
        assert!(acc.tool_calls.is_empty());
        let body = acc.into_pseudo_body();
        assert_eq!(
            body.pointer("/choices/0/message/content").unwrap(),
            &json!("Hello!")
        );
        assert_eq!(body.pointer("/usage/total_tokens").unwrap(), &json!(23));
    }

    #[test]
    fn sse_fold_tool_call_fragments_by_index() {
        use super::{RoundAcc, note_sse_chunk};
        let mut acc = RoundAcc::default();
        for line in [
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"web_search","arguments":""}}]}}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"query\": \"kiln\"}"}}]}}]}"#,
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}",
        ] {
            note_sse_chunk(&mut acc, line);
        }
        assert_eq!(acc.tool_calls.len(), 1);
        assert_eq!(acc.tool_calls[0]["id"], json!("c1"));
        assert_eq!(acc.tool_calls[0]["function"]["name"], json!("web_search"));
        assert_eq!(
            acc.tool_calls[0]["function"]["arguments"],
            json!("{\"query\": \"kiln\"}")
        );
        assert_eq!(acc.finish_reason.as_deref(), Some("tool_calls"));
        // The pseudo body must feed the loop's native tool-call extraction.
        let body = acc.into_pseudo_body();
        assert_eq!(
            body.pointer("/choices/0/message/tool_calls/0/function/name")
                .unwrap(),
            &json!("web_search")
        );
    }

    #[test]
    fn sse_fold_mesh_fallback_wrap() {
        use super::{RoundAcc, note_sse_chunk};
        let mut acc = RoundAcc::default();
        note_sse_chunk(
            &mut acc,
            r#"data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"peer answer"},"finish_reason":"stop"}],"mesh_fallback":"node-xyz"}"#,
        );
        assert_eq!(acc.content, "peer answer");
        assert_eq!(acc.mesh_fallback.as_deref(), Some("node-xyz"));
        assert_eq!(
            acc.into_pseudo_body().get("mesh_fallback").unwrap(),
            &json!("node-xyz")
        );
    }

    #[tokio::test]
    async fn chat_streams_sse_end_to_end_and_types_live() {
        use std::sync::Arc;
        use tokio::sync::Mutex as TMutex;

        // Mock gateway: answers SSE with three content chunks (separated
        // in time so the live tap genuinely fires mid-generation).
        let seen: Arc<TMutex<Vec<String>>> = Arc::new(TMutex::new(Vec::new()));
        let seen_live = seen.clone();
        let app = axum::Router::new()
            .route(
                "/status",
                axum::routing::get(|| async { axum::Json(json!({ "model": "mock:8b" })) }),
            )
            .route(
                "/v1/chat/completions",
                axum::routing::post(move || {
                    let seen_live = seen_live.clone();
                    async move {
                        let body = "data: {\"choices\":[{\"delta\":{\"content\":\"live \"}}]}\n\n"
                            .to_owned()
                            + "data: {\"choices\":[{\"delta\":{\"content\":\"typed\"}}]}\n\n"
                            + "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
                            + "data: {\"choices\":[],\"usage\":{\"total_tokens\":7}}\n\n"
                            + "data: [DONE]\n\n";
                        seen_live.lock().await.push(body.clone());
                        (
                            [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
                            body,
                        )
                    }
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let cfg =
            crate::config::BotConfig::from_parts(&format!("http://{addr}"), "http://127.0.0.1:1");
        let fragments: Arc<std::sync::Mutex<Vec<String>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let tap_seen = fragments.clone();
        let live = move |ev: LiveEvent| match ev {
            LiveEvent::RoundStart => {
                tap_seen
                    .lock()
                    .expect("test tap")
                    .push("⇆round".to_string());
            }
            LiveEvent::Text(frag) => {
                tap_seen.lock().expect("test tap").push(frag.to_string());
            }
        };
        let (reply, served_by, tokens) = chat_with_usage_live(&cfg, "say hi", &[], Some(&live))
            .await
            .expect("streamed chat");
        assert_eq!(reply, "live typed");
        assert!(served_by.is_none());
        assert_eq!(tokens, 7);
        // The live tap fired per fragment — the live-typing contract.
        assert_eq!(
            *fragments.lock().expect("test tap"),
            vec![
                "⇆round".to_string(),
                "live ".to_string(),
                "typed".to_string()
            ]
        );
    }

    #[tokio::test]
    async fn chat_stream_config_off_falls_back_to_buffered() {
        use std::sync::Arc;

        // Same SSE gateway, but [chat] stream = false → buffered parsing
        // of a plain JSON mock; the live tap must never fire.
        let app = axum::Router::new()
            .route(
                "/status",
                axum::routing::get(|| async { axum::Json(json!({ "model": "mock:8b" })) }),
            )
            .route(
                "/v1/chat/completions",
                axum::routing::post(|| async {
                    axum::Json(json!({
                        "choices": [{"message": {"role": "assistant", "content": "buffered"}}],
                        "usage": {"total_tokens": 3}
                    }))
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let mut cfg =
            crate::config::BotConfig::from_parts(&format!("http://{addr}"), "http://127.0.0.1:1");
        cfg.chat_stream = false;
        let fired: Arc<std::sync::Mutex<Vec<String>>> = Arc::new(Default::default());
        let tap_seen = fired.clone();
        let live = move |ev: LiveEvent| {
            if let LiveEvent::Text(frag) = ev {
                tap_seen.lock().expect("test tap").push(frag.to_string());
            }
        };
        let (reply, _s, tokens) = chat_with_usage_live(&cfg, "say hi", &[], Some(&live))
            .await
            .expect("buffered chat");
        assert_eq!(reply, "buffered");
        assert_eq!(tokens, 3);
        assert!(fired.lock().expect("test tap").is_empty());
    }

    #[tokio::test]
    async fn chat_tool_sse_streams_answer_round_and_clears_the_first() {
        // Web-search-augmented chat over SSE: round 0 streams a native
        // tool_call (no content — nothing typed), the round-1 answer streams
        // with content after the tap has seen a RoundStart boundary. The
        // sink must clear between rounds so web-informed answers type live.
        use std::sync::Arc;

        let app = axum::Router::new()
            .route(
                "/status",
                axum::routing::get(|| async { axum::Json(json!({ "model": "mock:8b" })) }),
            )
            .route(
                "/v1/chat/completions",
                axum::routing::post(|axum::Json(req): axum::Json<Value>| async move {
                    let has_tool_result = req["messages"]
                        .as_array()
                        .is_some_and(|ms| ms.iter().any(|m| m["role"] == json!("tool")));
                    let body = if has_tool_result {
                        let a1 = json!({"choices":[{"delta":{"content":"answer "}}]});
                        let a2 = json!({"choices":[{"delta":{"content":"after web search."}}]});
                        let fin = json!({"choices":[{"delta":{},"finish_reason":"stop"}]});
                        let usage = json!({"choices":[],"usage":{"total_tokens":42}});
                        format!(
                            "data: {}\n\ndata: {}\n\ndata: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
                            a1, a2, fin, usage
                        )
                    } else {
                        let frag0 = json!({"index":0,"id":"c1","function":{"name":"web_search","arguments":""}});
                        let args = json!({"index":0,"function":{"arguments": json!({"query": "isopress kiln site"}).to_string()}});
                        let fin = json!({"choices":[{"delta":{},"finish_reason":"tool_calls"}]});
                        format!(
                            "data: {}\n\ndata: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
                            json!({"choices":[{"delta":{"tool_calls":[frag0]}}]}),
                            json!({"choices":[{"delta":{"tool_calls":[args]}}]}),
                            fin
                        )
                    };
                    (
                        [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
                        body,
                    )
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
                            {"title": "Isopress kiln", "url": "https://example.com/kiln", "snippet": "isopress kiln built in 1989"}
                        ]
                    }))
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let mut extra =
            crate::config::BotConfig::from_parts(&format!("http://{addr}"), "http://127.0.0.1:1");
        extra.web.enabled = true;

        let events: Arc<std::sync::Mutex<Vec<String>>> = Arc::new(Default::default());
        let tap_seen = events.clone();
        let live = move |ev: LiveEvent| {
            let mut guard = tap_seen.lock().expect("test tap");
            match ev {
                LiveEvent::RoundStart => guard.push("⇆round".to_string()),
                LiveEvent::Text(f) => guard.push(f.to_string()),
            }
        };
        let (reply, served_by, tokens) = chat_with_usage_live(
            &extra,
            "Where is the isopress kiln built?",
            &[],
            Some(&live),
        )
        .await
        .expect("tool-streamed chat");
        assert_eq!(reply, "answer after web search.");
        assert!(served_by.is_none());
        assert_eq!(tokens, 42);
        // Two round boundaries, zero text in the tool round, the answer
        // typed live — the tool-round streaming contract.
        assert_eq!(
            *events.lock().expect("test tap"),
            vec![
                "⇆round".to_string(),
                "⇆round".to_string(),
                "answer ".to_string(),
                "after web search.".to_string()
            ]
        );
    }
}
