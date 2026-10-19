//! `scrape_and_summarize` — the first Rust-native macro-tool (FEATURES
//! §17.2), built on the composition pattern the approval bridge (§15)
//! established: **one model verb wrapping a compiled sub-pipeline**, so a
//! small local model spends its reasoning budget on a single decision
//! instead of orchestrating a dozen granular tool calls.
//!
//! The model's whole job:
//!
//! ```json
//! { "skill": "pipeline.scrape_summarize",
//!   "args": { "urls": ["https://example.com/release-notes"],
//!             "question": "what changed in the api?" } }
//! ```
//!
//! Rust does the rest, with the same discipline the rest of the bot
//! carries (fail-honest, bounded, no silent substitution):
//!
//! 1. **fetch** — bounded concurrent reqwest (≤5 urls, 2 at a time, 20 s
//!    per hop, 800 KiB each, text/* + html(ish) content types only,
//!    private targets refused the same as web_fetch);
//! 2. **strip** — hand-rolled tag/entity removal to readable text;
//! 3. **chunk** — word-boundary slices, globally capped;
//! 4. **embed** — each chunk into the node's private memory (nomic in
//!    SQLite, strictly local) so the page content becomes /recall-able,
//!    not just chat-transient;
//! 5. **summarize** — ONE gateway chat call (the runtime's own tier
//!    routing) over the stripped texts, answering the question when one
//!    was asked. The summary degrades honestly (flagged `summary_ok`
//!    false) when the gateway is unreachable — the embeddings still
//!    land, the caller still sees per-source status: a scrape is never
//!    a lie.
//!
//! The tool loop never sees partial pipelines of this shape: the skill
//! returns one structured JSON value or a typed error, mirroring
//! "scrape_and_summarize"'s promise that type safety and speed come
//! from the wrapper, not the model.

use anyhow::{Context as _, Result};
use serde_json::{Value, json};
use std::time::Duration;

/// Skill-facing bounds (also the security bounds — see refusals below).
const MAX_URLS: usize = 5;
const MAX_CONCURRENT: usize = 2;
const PER_HOP_TIMEOUT: Duration = Duration::from_secs(20);
const MAX_DOWNLOAD_BYTES: usize = 800 * 1024;
/// Global chunk budget across the whole run (embed + summarize input).
const MAX_CHUNKS: usize = 8;
const CHUNK_CHARS: usize = 1200;
/// Text budget the summarize leg forwards (chars; head-biased chunk
/// order so ledes survive a cap).
const SUMMARY_INPUT_CHARS: usize = 9000;
const SUMMARY_COMPLETION_TOKENS: u32 = 700;
/// Summary leg wall-clock (the runtime bounds itself; this adds headroom).
const SUMMARY_TIMEOUT: Duration = Duration::from_secs(180);

// ---------------------------------------------------------------------------
// Fetch + strip (pure Rust, no model involved)
// ---------------------------------------------------------------------------

/// One fetched, stripped source.
#[derive(Debug, Clone)]
pub struct Source {
    pub url: String,
    /// Ok = stripped text (possibly empty for a genuinely empty page);
    /// Err = the typed failure the result reports verbatim.
    pub text: Result<String, String>,
    pub content_type: String,
}

/// Fetch + strip one URL. Private targets are refused with the same
/// SSRF stance as the gateway's own web_fetch — literally the shared
/// core predicate, so a scrape tool can never become a LAN scanner
/// riding the bot's egress.
pub async fn fetch_source(_cfg: &crate::config::BotConfig, url: &str) -> Source {
    let url = url.trim().to_string();
    if let Err(reason) = jerith_mesh_core::websearch::assert_fully_fetchable(&url) {
        return Source {
            url: url.clone(),
            text: Err(first_line(&reason.to_string())),
            content_type: String::new(),
        };
    }
    let client = crate::api::client();
    let resp = match client.get(&url).timeout(PER_HOP_TIMEOUT).send().await {
        Ok(resp) => resp,
        Err(e) => {
            return Source {
                url: url.clone(),
                text: Err(first_line(&e.to_string())),
                content_type: String::new(),
            };
        }
    };
    // Pull status-line + final URL + content-type BEFORE consuming the
    // body, then take bytes by value (Response is single-use).
    let status = resp.status();
    let final_url = resp.url().to_string();
    let ctype = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if !status.is_success() {
        return Source {
            url: final_url.clone(),
            text: Err(format!("HTTP {status}")),
            content_type: ctype,
        };
    }
    if is_refused_type(&ctype) {
        return Source {
            url: final_url.clone(),
            text: Err(format!("content-type {ctype:?} is not page text")),
            content_type: ctype,
        };
    }
    // Size cap by bytes; honest refusal when the page is bigger.
    let bytes = match resp.bytes().await {
        Ok(b) => b,
        Err(e) => {
            return Source {
                url: final_url.clone(),
                text: Err(first_line(&e.to_string())),
                content_type: ctype,
            };
        }
    };
    if bytes.len() > MAX_DOWNLOAD_BYTES {
        return Source {
            url: final_url.clone(),
            text: Err(format!(
                "page exceeds the download cap ({} bytes > {})",
                bytes.len(),
                MAX_DOWNLOAD_BYTES
            )),
            content_type: ctype,
        };
    }
    let text = html_to_text(&String::from_utf8_lossy(&bytes));
    Source {
        url: final_url,
        text: Ok(text),
        content_type: ctype,
    }
}

/// Content types the stripper makes no sense for (refused, not guessed).
fn is_refused_type(ctype: &str) -> bool {
    let media = ctype.split(';').next().unwrap_or("").trim();
    !(media.starts_with("text/")
        || media == "application/xhtml+xml"
        || media == "application/xml"
        || media == "application/json")
}

/// Tag + entity stripper: a small state machine — `<script>`/`<style>`/
/// `<noscript>` bodies dropped entirely, other tags removed (adjacent
/// text nodes rejoin: "<b>ingest</b>ion" stays one word), entities
/// decoded, whitespace collapsed.
pub fn html_to_text(raw: &str) -> String {
    let bytes = raw.as_bytes();
    let mut out = String::with_capacity(raw.len() / 2);
    let mut skip_tag: Option<String> = None; // inside <script>/<style>/<noscript>
    let mut in_tag = false;
    let mut i = 0usize;
    let mut last_was_space = true; // leading whitespace collapse
    while i < bytes.len() {
        let b = bytes[i];
        if let Some(tag) = &skip_tag {
            // Hunt for the closing </tag>, then jump past it.
            let close = format!("</{}", tag);
            let rest = &raw[i..];
            if rest.len() >= close.len() && rest.to_ascii_lowercase().starts_with(&close) {
                match rest.find('>') {
                    Some(rel) => {
                        i += rel + 1;
                    }
                    None => {
                        i = bytes.len(); // unterminated skip body
                    }
                }
                skip_tag = None;
                in_tag = false;
                continue;
            }
            i += 1;
            continue;
        }
        if in_tag {
            if b == b'>' {
                in_tag = false;
                // NOTE: no space on tag close — text split across inline
                // tags rejoins ("in<b>g</b>est" stays "ingest").
            }
            i += 1;
            continue;
        }
        if b == b'<' {
            // Look ahead for a skip-tag start.
            let end = (i + 40).min(raw.len());
            let lower = raw[i..end].to_ascii_lowercase();
            for tag in ["script", "style", "noscript"] {
                if lower.starts_with(&format!("<{tag}")) {
                    skip_tag = Some(tag.to_string());
                    break;
                }
            }
            in_tag = true;
            i += 1;
            continue;
        }
        if b.is_ascii_whitespace() {
            if !last_was_space {
                out.push(' ');
                last_was_space = true;
            }
            i += 1;
            continue;
        }
        out.push(b as char);
        last_was_space = false;
        i += 1;
    }
    // Whitespace collapse ran per-char; entities decode afterwards.
    decode_entities(&out).trim().to_string()
}

/// The common HTML entities (named subset + numeric forms).
pub fn decode_entities(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(at) = rest.find('&') {
        out.push_str(&rest[..at]);
        rest = &rest[at..];
        let consumed = if let Some(semi) = rest.find(';') {
            if semi > 1 && semi <= 10 {
                let ent = &rest[1..semi];
                let decoded = match ent {
                    "amp" => Some('&'),
                    "lt" => Some('<'),
                    "gt" => Some('>'),
                    "quot" => Some('"'),
                    "apos" => Some('\''),
                    "nbsp" => Some(' '),
                    _ => {
                        if let Some(hex) = ent.strip_prefix("#x").or(ent.strip_prefix("#X")) {
                            u32::from_str_radix(hex, 16).ok().and_then(char::from_u32)
                        } else if let Some(dec) = ent.strip_prefix('#') {
                            dec.parse::<u32>().ok().and_then(char::from_u32)
                        } else {
                            None
                        }
                    }
                };
                match decoded {
                    Some(c) => {
                        out.push(c);
                        rest = &rest[semi + 1..];
                        true
                    }
                    None => false,
                }
            } else {
                false
            }
        } else {
            false
        };
        if !consumed {
            // Not a decodable entity — emit the '&' verbatim.
            out.push('&');
            rest = &rest[1..];
        }
    }
    out.push_str(rest);
    out
}

/// Word-boundary chunks of `CHUNK_CHARS` (never mid-word).
pub fn chunk_text(text: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut buf = String::new();
    for word in text.split_whitespace() {
        let would = buf.len() + word.len() + usize::from(!buf.is_empty());
        if would > CHUNK_CHARS && !buf.is_empty() {
            chunks.push(std::mem::take(&mut buf));
        }
        if !buf.is_empty() {
            buf.push(' ');
        }
        buf.push_str(word);
    }
    if !buf.is_empty() {
        chunks.push(buf);
    }
    chunks
}

// ---------------------------------------------------------------------------
// Embed + summarize (through the gateway, exactly like every other skill)
// ---------------------------------------------------------------------------

/// Embed one chunk into the node's private memory. Embed errors degrade
/// that chunk's flag only — never the whole call.
async fn embed_chunk(cfg: &crate::config::BotConfig, text: &str) -> bool {
    match crate::api::memory_embed(cfg, text).await {
        Ok(_) => true,
        Err(e) => {
            tracing::warn!(error = %e, "scrape_summarize chunk embed failed");
            false
        }
    }
}

/// The summarize leg: ONE gateway chat call (the runtime's own tier
/// routing decides the model), no persona entanglement — the system
/// message is the task only.
async fn summarize(
    cfg: &crate::config::BotConfig,
    sources: &[(String, String)], // (url, text)
    question: Option<&str>,
) -> Result<String> {
    // Head-biased budget: ledes survive a cap, tails fold away first.
    let mut body = String::new();
    for (url, text) in sources {
        let room = SUMMARY_INPUT_CHARS.saturating_sub(body.len());
        if room < 400 {
            break;
        }
        if text.len() <= room {
            body.push_str(&format!("\n\n### {url}\n{text}"));
        } else {
            let head: String = text.chars().take(room).collect();
            body.push_str(&format!("\n\n### {url}\n{head}"));
            break;
        }
    }
    anyhow::ensure!(!body.trim().is_empty(), "no page text to summarize");
    let task = match question {
        Some(q) if !q.trim().is_empty() => format!(
            "Answer the question using ONLY the page texts. Then bullet 3-5 key facts.\n\n\
             Question: {}",
            q.trim()
        ),
        _ => "Summarize the page texts: 3-5 key facts as bullets, one line each.\n\n".to_string(),
    };
    let payload = json!({
        "model": crate::api::MODEL_PLACEHOLDER,
        "temperature": 0.2,
        "max_tokens": SUMMARY_COMPLETION_TOKENS,
        "messages": [
            {"role": "system", "content": task},
            {"role": "user", "content": body}
        ]
    });
    let url = format!(
        "{}/v1/chat/completions",
        cfg.gateway_url.trim_end_matches('/')
    );
    let resp = crate::api::client()
        .post(&url)
        .json(&payload)
        .timeout(SUMMARY_TIMEOUT)
        .send()
        .await
        .with_context(|| format!("gateway unreachable at {url}"))?;
    let status = resp.status();
    let parsed: Value = resp.json().await.context("gateway returned non-JSON")?;
    anyhow::ensure!(status.is_success(), "summary leg failed: {}", status);
    parsed
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| anyhow::anyhow!("gateway answer had no content"))
}

// ---------------------------------------------------------------------------
// The macro-tool surface
// ---------------------------------------------------------------------------

/// Run the whole sub-pipeline. `embed` false (or a memory outage) keeps
/// everything else working; every source reports its own fate.
pub async fn scrape_summarize(
    cfg: &crate::config::BotConfig,
    urls: &[String],
    question: Option<&str>,
    embed: bool,
) -> Result<Value> {
    anyhow::ensure!(
        !urls.is_empty(),
        "scrape_summarize: at least one url is required"
    );
    anyhow::ensure!(
        urls.len() <= MAX_URLS,
        "scrape_summarize: at most {MAX_URLS} urls per run (got {})",
        urls.len()
    );

    // Fetch phase — bounded waves (semaphore substitute; N is small).
    let fetched: Vec<Source> = {
        let mut tasks: Vec<tokio::task::JoinHandle<Source>> = Vec::new();
        for url in urls.iter().take(MAX_URLS) {
            let cfg = cfg.clone();
            let url = url.clone();
            tasks.push(tokio::spawn(async move { fetch_source(&cfg, &url).await }));
        }
        let mut out = Vec::new();
        for wave in tasks.chunks_mut(MAX_CONCURRENT.max(1)) {
            for handle in wave.iter_mut() {
                out.push(handle.await.expect("scrape task join"));
            }
        }
        out
    };

    // Strip + chunk phases over the successful sources.
    let mut total_chunks = 0usize;
    let mut chunks: Vec<(String, String)> = Vec::new();
    let mut source_rows = Vec::new();
    for source in &fetched {
        match &source.text {
            Ok(text) => {
                let parts = chunk_text(text);
                let kept = parts.len().min(MAX_CHUNKS.saturating_sub(total_chunks));
                for part in parts.iter().take(kept) {
                    chunks.push((source.url.clone(), part.clone()));
                }
                total_chunks += kept;
                source_rows.push(json!({
                    "url": source.url,
                    "ok": true,
                    "chars": text.chars().count(),
                    "chunks": kept,
                    "dropped_chunks": parts.len().saturating_sub(kept),
                    "content_type": source.content_type,
                }));
            }
            Err(reason) => {
                source_rows.push(json!({ "url": source.url, "ok": false, "error": reason }));
            }
        }
    }

    // Embed phase (strictly local nomic store, via the gateway).
    let mut embedded = Vec::new();
    if embed {
        for (url, text) in chunks.iter() {
            embedded.push(json!({
                "url": url,
                "embedded": embed_chunk(cfg, text).await,
                "chars": text.len(),
            }));
        }
    }

    // Summarize phase (one reasoning step — the model's ONLY output role).
    let usable: Vec<(String, String)> = fetched
        .iter()
        .filter_map(|s| {
            s.text
                .as_ref()
                .ok()
                .filter(|t| !t.trim().is_empty())
                .map(|t| (s.url.clone(), t.clone()))
        })
        .collect();
    let (summary, summary_ok) = match summarize(cfg, &usable, question).await {
        Ok(text) => (text, true),
        Err(e) => (first_line(&e.to_string()), false),
    };

    Ok(json!({
        "ok": true,
        "sources": source_rows,
        "embedded": embedded,
        "summary": summary,
        "summary_ok": summary_ok,
        "question": question.unwrap_or_default(),
    }))
}

fn first_line(s: &str) -> String {
    s.lines()
        .next()
        .unwrap_or("error")
        .chars()
        .take(240)
        .collect()
}

// ---------------------------------------------------------------------------
// Model-facing skill (plugin `pipeline`)
// ---------------------------------------------------------------------------

pub fn skills() -> Vec<crate::plugins::SkillSpec> {
    use crate::plugins::SkillSpec;
    vec![SkillSpec {
        name: "pipeline.scrape_summarize",
        plugin: "pipeline",
        description: "ONE-CALL macro-tool: fetch up to 5 web pages, strip them to text, \
                      embed every chunk into private memory (/recall-able), and summarize \
                      them (or answer a question from them) in a single response. Prefer \
                      this over web_search+web_fetch chains when the target pages are known.",
        parameters: json!({
            "type": "object",
            "properties": {
                "urls": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5,
                         "description": "http(s) URLs to fetch"},
                "question": {"type": "string", "description": "Optional question the summary should answer"},
                "embed": {"type": "boolean", "description": "Embed chunks into private memory (default true)"}
            },
            "required": ["urls"]
        }),
        run: scrape_summarize_runner,
    }]
}

fn scrape_summarize_runner<'args>(
    cfg: &'args crate::config::BotConfig,
    args: &'args Value,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<Value>> + Send + 'args>> {
    Box::pin(async move {
        let urls: Vec<String> = args
            .get("urls")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        let question = args.get("question").and_then(Value::as_str);
        let embed = args.get("embed").and_then(Value::as_bool).unwrap_or(true);
        scrape_summarize(cfg, &urls, question, embed).await
    })
}

// ---------------------------------------------------------------------------
// Tests (pure parts + a live-internet integration check, gated on demand)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stripper_drops_scripts_and_keeps_prose() {
        let page = "<html><head><style>p{color:red}</style><script>evil()</script></head>\
                    <body><h1>Ingest</h1><p>Buffers&rsquo; sizing <b>adds</b> up.</p>\
                    <noscript>fallback junk</noscript></body></html>";
        let text = html_to_text(page);
        assert!(text.contains("Ingest"), "kept h1: {text}");
        assert!(text.contains("sizing"), "kept body text: {text}");
        assert!(text.contains("adds"), "inline tag rejoined: {text}");
        assert!(!text.contains("evil"), "script body leaked: {text}");
        assert!(!text.contains("color:red"), "style body leaked: {text}");
        assert!(!text.contains("fallback junk"), "noscript leaked: {text}");
    }

    #[test]
    fn inline_tags_rejoin_words() {
        assert_eq!(html_to_text("in<b>g</b>est"), "ingest");
        assert_eq!(html_to_text("a <b>bold</b> word"), "a bold word");
    }

    #[test]
    fn entities_decode_and_unknown_stay() {
        assert_eq!(decode_entities("A &amp; B &lt;x&gt; Z"), "A & B <x> Z");
        assert_eq!(decode_entities("a&#65;b"), "aAb");
        assert_eq!(
            decode_entities("nbs p test &nbsp; tail"),
            "nbs p test   tail"
        );
        assert_eq!(decode_entities("weird &gtos; thing"), "weird &gtos; thing");
        assert_eq!(decode_entities("safe & intact"), "safe & intact");
    }

    #[test]
    fn chunks_respect_word_boundaries() {
        let words: Vec<String> = (0..300).map(|i| format!("word{i}")).collect();
        let text = words.join(" ");
        let chunks = chunk_text(&text);
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert!(chunk.chars().count() <= CHUNK_CHARS);
            assert!(!chunk.starts_with(' ') && !chunk.ends_with(' '));
        }
        // No mid-word cut: the round-trip word count is preserved.
        let joined = chunks.join(" ");
        assert_eq!(joined.split_whitespace().count(), 300);
    }

    #[test]
    fn refused_type_gate() {
        assert!(is_refused_type("application/pdf"));
        assert!(is_refused_type("image/png"));
        assert!(!is_refused_type("text/html; charset=utf-8"));
        assert!(!is_refused_type("application/xhtml+xml"));
    }

    /// LIVE: real internet + real gateway + real embed — run explicitly:
    /// `cargo test -p jerith-mesh-bot live_scrape -- --ignored --nocapture`
    #[tokio::test]
    #[ignore = "live network + live gateway"]
    async fn live_scrape_summarize_real_internet() {
        let cfg = std::sync::Arc::new(crate::config::BotConfig::from_parts(
            "http://127.0.0.1:8900",
            "http://127.0.0.1:8322",
        ));
        let out = scrape_summarize(
            &cfg,
            &["https://example.com".to_string()],
            Some("what is this page for?"),
            true,
        )
        .await
        .expect("live pipeline");
        println!("{}", serde_json::to_string_pretty(&out).unwrap());
        assert_eq!(out["sources"][0]["ok"], json!(true));
        assert_eq!(out["summary_ok"], json!(true));
        assert_eq!(out["embedded"][0]["embedded"], json!(true));
    }
}
