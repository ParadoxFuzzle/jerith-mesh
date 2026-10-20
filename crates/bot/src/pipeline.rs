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
// refactor_module — the second macro-tool (§17.2): parse → patch →
// build-check → report, one model verb wrapping a compiled sub-pipeline.
// ---------------------------------------------------------------------------

/// Skill-facing bounds. The patch leg rewrites the WHOLE module in one
/// completion, so the input caps keep the output physically renderable
/// within the completion budget (~3.2 k tokens of dense code).
pub const MAX_CODE_CHARS: usize = 16_000;
pub const MAX_INSTRUCTION_CHARS: usize = 2_000;
/// Output budget: a patched module larger than this is a truncation
/// symptom, never a refactor — refused, not truncated silently.
pub const MAX_PATCHED_CHARS: usize = 64_000;
const REF_COMPLETION_TOKENS: u32 = 3200;
const REF_TIMEOUT: Duration = Duration::from_secs(300);
/// Build-check wall-clock (a cold rustc on a small module needs seconds;
/// the slack is for dependency-starved crate contexts).
const BUILD_CHECK_TIMEOUT: Duration = Duration::from_secs(90);

/// Target language of a refactor run. `Other` gets the model patch leg
/// but never a misleading parse or toolchain verdict — unchecked fields
/// stay null, never false.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefLang {
    Rust,
    Python,
    Typescript,
    Javascript,
    Other,
}

impl RefLang {
    pub fn parse(s: &str) -> RefLang {
        match s.trim().to_ascii_lowercase().as_str() {
            "rust" | "rs" => RefLang::Rust,
            "python" | "py" => RefLang::Python,
            "typescript" | "ts" => RefLang::Typescript,
            "javascript" | "js" => RefLang::Javascript,
            _ => RefLang::Other,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            RefLang::Rust => "rust",
            RefLang::Python => "python",
            RefLang::Typescript => "typescript",
            RefLang::Javascript => "javascript",
            RefLang::Other => "other",
        }
    }
}

/// What the structural parse saw. `balanced` is `None` where the pass
/// does not apply (Other) — absence is honest, never folded into false.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ParseReport {
    pub checked: bool,
    pub balanced: Option<bool>,
    pub detail: String,
}

/// Structural balance scan: a stack of `()` `[]` `{}` walked with
/// language-aware string/comment states. This is a TRUNCATION lint, not
/// a front-end — it exists to refuse sending a half-module to the model
/// and to catch a model patch that cut off mid-block. Known limits are
/// documented per branch; when a limit makes a verdict unsafe, report
/// `balanced: None` (unchecked), never a false refusal.
pub fn structural_balance(code: &str, lang: RefLang) -> ParseReport {
    let mut stack: Vec<(u8, usize)> = Vec::new(); // (delimiter byte, line)
    let line_of = |pos: usize| 1 + code[..pos].matches('\n').count();
    let mut i = 0usize;
    let bytes = code.as_bytes();
    let close_of = |open: u8| match open {
        b'(' => b')',
        b'[' => b']',
        b'{' => b'}',
        _ => open,
    };
    // Helper: record a mismatch and stop. Expands to a `return` carrying
    // the whole verdict (not a break), so no label control flow escapes
    // macro hygiene.
    macro_rules! bail {
        ($why:expr) => {{
            return ParseReport {
                checked: true,
                balanced: Some(false),
                detail: $why,
            };
        }};
    }

    {
        match lang {
            RefLang::Other => {
                return ParseReport {
                    checked: false,
                    balanced: None,
                    detail: "language not structurally scanned".into(),
                };
            }
            RefLang::Rust => {
                while i < bytes.len() {
                    match bytes[i] {
                        b'/' if bytes.get(i + 1) == Some(&b'/') => {
                            i += 2;
                            while i < bytes.len() && bytes[i] != b'\n' {
                                i += 1;
                            }
                        }
                        b'/' if bytes.get(i + 1) == Some(&b'*') => {
                            // Rust block comments NEST.
                            let mut depth = 1usize;
                            i += 2;
                            while i + 1 < bytes.len() && depth > 0 {
                                if bytes[i] == b'/' && bytes[i + 1] == b'*' {
                                    depth += 1;
                                    i += 2;
                                } else if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                                    depth -= 1;
                                    i += 2;
                                } else {
                                    i += 1;
                                }
                            }
                            if depth > 0 {
                                bail!("unterminated /* block comment".to_string());
                            }
                        }
                        b'"' => {
                            i += 1;
                            while i < bytes.len() {
                                match bytes[i] {
                                    b'\\' => i += 2,
                                    b'"' => {
                                        i += 1;
                                        break;
                                    }
                                    _ => i += 1,
                                }
                            }
                            if i > bytes.len() || (i == bytes.len() && bytes.last() != Some(&b'"'))
                            {
                                bail!("unterminated string literal".to_string());
                            }
                        }
                        b'\'' => {
                            // Char literal vs lifetime: `'a'` / `'\n'` are
                            // literals; anything else is a lifetime and
                            // carries no delimiters to track.
                            if bytes.get(i + 1) == Some(&b'\\') {
                                if bytes.get(i + 3) == Some(&b'\'') {
                                    i += 4;
                                    continue;
                                }
                            } else if bytes.get(i + 2) == Some(&b'\'') {
                                i += 3;
                                continue;
                            }
                            // Lifetime — skip the tick, walk on.
                            i += 1;
                        }
                        b'r' | b'b' => {
                            // Raw strings r"…", r#"…"#, byte / byte-raw
                            // forms (r, b", br, and the # variants).
                            let mut j = i + 1;
                            if bytes[i] == b'b' && bytes.get(j) == Some(&b'r') {
                                j += 1;
                            }
                            let mut hashes = 0usize;
                            while bytes.get(j) == Some(&b'#') {
                                hashes += 1;
                                j += 1;
                            }
                            if bytes.get(j) == Some(&b'"') {
                                // Closing form: '"' + `hashes` '#'s.
                                j += 1;
                                let mut closed = None;
                                let mut k = j;
                                while k < bytes.len() {
                                    if bytes[k] == b'"'
                                        && (0..hashes).all(|h| bytes.get(k + 1 + h) == Some(&b'#'))
                                    {
                                        closed = Some(k + 1 + hashes);
                                        break;
                                    }
                                    k += 1;
                                }
                                match closed {
                                    Some(end) => {
                                        i = end;
                                        continue;
                                    }
                                    None => {
                                        bail!("unterminated raw string literal".to_string())
                                    }
                                }
                            }
                            // A plain identifier starting with r/b — walk on.
                            i += 1;
                        }
                        c @ (b'(' | b'[' | b'{') => {
                            stack.push((c, line_of(i)));
                            i += 1;
                        }
                        c @ (b')' | b']' | b'}') => match stack.pop() {
                            Some((open, _)) if close_of(open) == c => i += 1,
                            Some((open, open_line)) => bail!(format!(
                                "unbalanced: '{}' at line {line} closes '{}' opened at line {open_line}",
                                c as char,
                                open as char,
                                line = line_of(i)
                            )),
                            None => {
                                bail!(format!(
                                    "unbalanced: stray '{}' at line {}",
                                    c as char,
                                    line_of(i)
                                ))
                            }
                        },
                        _ => i += 1,
                    }
                }
            }
            RefLang::Python => {
                // Python: only delimiters are () [] {}; strings may be
                // single/double quoted (with escapes) or triple-quoted.
                let mut triple: Option<u8> = None; // which quote char opened a triple block
                let mut in_string: Option<u8> = None;
                while i < bytes.len() {
                    let b = bytes[i];
                    if triple.is_some() && b == b'\\' {
                        i += 2;
                        continue;
                    }
                    if triple.is_some() {
                        if b == triple.unwrap()
                            && bytes.get(i + 1) == Some(&triple.unwrap())
                            && bytes.get(i + 2) == Some(&triple.unwrap())
                        {
                            triple = None;
                            in_string = None;
                            i += 3;
                            continue;
                        }
                        i += 1;
                        continue;
                    }
                    if let Some(q) = in_string {
                        if b == b'\\' {
                            i += 2;
                            continue;
                        }
                        if b == q {
                            in_string = None;
                        }
                        i += 1;
                        continue;
                    }
                    // Triple-quote opener probe BEFORE the single-quote
                    // handlers (""" is three of the same bytes).
                    if (b == b'"' || b == b'\'')
                        && bytes.get(i + 1) == Some(&b)
                        && bytes.get(i + 2) == Some(&b)
                    {
                        triple = Some(b);
                        in_string = Some(b);
                        i += 3;
                        continue;
                    }
                    match b {
                        b'#' => {
                            while i < bytes.len() && bytes[i] != b'\n' {
                                i += 1;
                            }
                        }
                        b'"' | b'\'' => {
                            in_string = Some(b);
                            i += 1;
                        }
                        c @ (b'(' | b'[' | b'{') => {
                            stack.push((c, line_of(i)));
                            i += 1;
                        }
                        c @ (b')' | b']' | b'}') => match stack.pop() {
                            Some((open, _)) if close_of(open) == c => i += 1,
                            Some((open, open_line)) => bail!(format!(
                                "unbalanced: '{}' at line {line} closes '{}' opened at line {open_line}",
                                c as char,
                                open as char,
                                line = line_of(i)
                            )),
                            None => {
                                bail!(format!(
                                    "unbalanced: stray '{}' at line {}",
                                    c as char,
                                    line_of(i)
                                ))
                            }
                        },
                        _ => i += 1,
                    }
                }
                if triple.is_some() || in_string.is_some() {
                    bail!("unterminated string literal".to_string());
                }
            }
            RefLang::Typescript | RefLang::Javascript => {
                // ts/js: two comment kinds, three string kinds; template
                // literals, `${expr}` interpolations and paren blocks all
                // live on ONE stack — a b'`' marker marks template
                // content, a b'%' marker marks a `${` interpolation. The
                // current mode is simply the stack top, so nesting
                // composes: `${` → code mode → backtick → template again.
                // Known limit (documented): regex literals (`/…/`) are
                // not lexed — a regex containing an unpaired delimiter
                // could false-positive; the report says checked, and a
                // toolchain check is the real verdict where one exists.
                while i < bytes.len() {
                    if matches!(stack.last(), Some((b'`', _))) {
                        // Template-literal content: opaque until the
                        // backtick closes it or `${` opens code again.
                        match bytes[i] {
                            b'\\' => i += 2,
                            b'`' => {
                                stack.pop();
                                i += 1;
                            }
                            b'$' if bytes.get(i + 1) == Some(&b'{') => {
                                stack.push((b'%', line_of(i)));
                                i += 2;
                            }
                            _ => i += 1,
                        }
                        continue;
                    }
                    match bytes[i] {
                        b'/' if bytes.get(i + 1) == Some(&b'/') => {
                            while i < bytes.len() && bytes[i] != b'\n' {
                                i += 1;
                            }
                        }
                        b'/' if bytes.get(i + 1) == Some(&b'*') => {
                            i += 2;
                            while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/')
                            {
                                i += 1;
                            }
                            if i + 1 >= bytes.len() {
                                bail!("unterminated /* block comment".to_string());
                            }
                            i += 2;
                        }
                        q @ (b'"' | b'\'') => {
                            i += 1;
                            let mut closed = false;
                            while i < bytes.len() {
                                match bytes[i] {
                                    b'\\' => i += 2,
                                    c if c == q => {
                                        i += 1;
                                        closed = true;
                                        break;
                                    }
                                    _ => i += 1,
                                }
                            }
                            if !closed {
                                bail!("unterminated string literal".to_string());
                            }
                        }
                        b'`' => {
                            stack.push((b'`', line_of(i)));
                            i += 1;
                        }
                        c @ (b'(' | b'[' | b'{') => {
                            stack.push((c, line_of(i)));
                            i += 1;
                        }
                        c @ (b')' | b']' | b'}') => {
                            match stack.pop() {
                                Some((open, _)) if close_of(open) == c => i += 1,
                                // The `${` sentinel: its `}` is consumed by
                                // the pop; control returns to template
                                // content naturally (top falls back to the
                                // enclosing backtick marker, if any).
                                Some((open, _)) if open == b'%' && c == b'}' => i += 1,
                                Some((open, open_line)) => bail!(format!(
                                    "unbalanced: '{}' at line {line} closes '{}' opened at line {open_line}",
                                    c as char,
                                    open as char,
                                    line = line_of(i)
                                )),
                                None => {
                                    bail!(format!(
                                        "unbalanced: stray '{}' at line {}",
                                        c as char,
                                        line_of(i)
                                    ))
                                }
                            }
                        }
                        _ => i += 1,
                    }
                }
            }
        }
    }

    if let Some(&(open, open_line)) = stack.last() {
        return ParseReport {
            checked: true,
            balanced: Some(false),
            detail: format!(
                "unbalanced: '{}' opened at line {open_line} never closed",
                open as char
            ),
        };
    }
    ParseReport {
        checked: true,
        balanced: Some(true),
        detail: format!("balanced — {} delimiters matched", stack.len() * 2),
    }
}

/// First fenced code block's inner text (â° ```lang … ``` â°), falling back
/// to the whole trimmed body when the model emitted no fence.
pub fn extract_code_block(text: &str) -> &str {
    if let Some(start) = text.find("```") {
        let after = &text[start + 3..];
        let body = after.split_once('\n').map(|x| x.1).unwrap_or(after);
        if let Some(end) = body.find("```") {
            return body[..end].trim();
        }
        // Unterminated fence (truncation): the remainder is still code.
        return body.trim();
    }
    text.trim()
}

/// The patch leg: ONE gateway chat call. Same shape as the summarize
/// leg — the runtime's own tier routing decides the model, the system
/// message is the task only, and the model's ONLY output role is the
/// refactored source.
async fn refactor_patch(
    cfg: &crate::config::BotConfig,
    language: RefLang,
    instruction: &str,
    code: &str,
) -> Result<String> {
    let task = format!(
        "You are a refactoring assistant for {} code. Apply the instruction \
         to the module below. Return ONLY the complete refactored source in \
         one fenced code block — no commentary, no commentary-only preamble, \
         no diffs. Preserve the module's public behavior unless the \
         instruction says otherwise.",
        language.name()
    );
    let payload = json!({
        "model": crate::api::MODEL_PLACEHOLDER,
        "temperature": 0.1,
        "max_tokens": REF_COMPLETION_TOKENS,
        "messages": [
            {"role": "system", "content": task},
            {"role": "user", "content": format!("Instruction:\n{instruction}\n\nModule:\n\n{code}")}
        ]
    });
    let url = format!(
        "{}/v1/chat/completions",
        cfg.gateway_url.trim_end_matches('/')
    );
    let resp = crate::api::client()
        .post(&url)
        .json(&payload)
        .timeout(REF_TIMEOUT)
        .send()
        .await
        .with_context(|| format!("gateway unreachable at {url}"))?;
    let status = resp.status();
    let parsed: Value = resp.json().await.context("gateway returned non-JSON")?;
    anyhow::ensure!(status.is_success(), "refactor patch leg failed: {status}");
    let content = parsed
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("gateway answer had no content"))?
        .to_string();
    anyhow::ensure!(
        content.chars().count() <= MAX_PATCHED_CHARS,
        "patched module exceeds the output budget ({} chars > {MAX_PATCHED_CHARS}) — \
         likely truncated; refactor refused",
        content.chars().count()
    );
    Ok(content)
}

/// Toolchain build-check verdict. `checked: false` is an honest status
/// with the reason attached — never folded into a pass.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct BuildCheck {
    pub checked: bool,
    pub ok: bool,
    pub tool: &'static str,
    pub error: Option<String>,
}

/// Run the language's cheapest real checker on the patched module:
/// rust `rustc --emit=metadata` (full parse + typecheck, no codegen),
/// python `python3 -m py_compile` (bytecode compile, no execution).
/// Nothing here EXECUTES the module — metadata/bbytecode emission
/// cannot run top-level code. No checker for the language → honest
/// `checked: false`.
pub async fn build_check(language: RefLang, code: &str) -> BuildCheck {
    let (tool, argv): (&'static str, Vec<String>) = match language {
        RefLang::Rust => (
            "rustc",
            vec![
                "rustc".into(),
                "--edition".into(),
                "2024".into(),
                "--crate-type".into(),
                "lib".into(),
                "--emit".into(),
                "metadata".into(),
                "-o".into(),
            ],
        ),
        RefLang::Python => (
            "python3",
            vec!["python3".into(), "-m".into(), "py_compile".into()],
        ),
        RefLang::Typescript | RefLang::Javascript | RefLang::Other => {
            return BuildCheck {
                checked: false,
                ok: false,
                tool: "none",
                error: Some("no build-check tool for this language in the bot's kit".into()),
            };
        }
    };
    let ext = if language == RefLang::Rust {
        "rs"
    } else {
        "py"
    };
    // Unique staging dir per check: the temp file is the module under
    // test; everything is cleaned up best-effort on the way out.
    static CHECK_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "jerith-ref-{}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0),
        CHECK_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
    ));
    if let Err(e) = tokio::fs::create_dir_all(&dir).await {
        return BuildCheck {
            checked: false,
            ok: false,
            tool,
            error: Some(format!("staging dir create failed: {e}")),
        };
    }
    let file = dir.join(format!("module.{ext}"));
    let out = dir.join("out.rmeta");
    let fut = match tokio::fs::write(&file, code).await {
        Ok(_) => {
            let mut argv = argv.clone();
            if tool == "rustc" {
                argv.push(out.to_string_lossy().to_string());
                // The module itself is rustc's input file — forgetting it
                // makes the checker answer "no input filename given",
                // which the live harness catches.
                argv.push(file.to_string_lossy().to_string());
            } else {
                argv.push(file.to_string_lossy().to_string());
            }
            tokio::process::Command::new(argv[0].clone())
                .args(&argv[1..])
                .current_dir(&dir)
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .kill_on_drop(true)
                .output()
        }
        Err(e) => {
            let _ = tokio::fs::remove_dir_all(&dir).await;
            let code_ = e;
            return BuildCheck {
                checked: false,
                ok: false,
                tool,
                error: Some(format!("staging write failed: {code_}")),
            };
        }
    };
    // Pin before the timeout wrapper borrows it (`output()` returns a
    // raw future, not a JoinHandle — pinning is the piece that makes
    // `&mut`-polling legal).
    let mut fut = std::pin::pin!(fut);
    let result = match tokio::time::timeout(BUILD_CHECK_TIMEOUT, fut.as_mut()).await {
        Ok(inner) => inner,
        Err(_) => {
            let _ = tokio::fs::remove_dir_all(&dir).await;
            return BuildCheck {
                checked: false,
                ok: false,
                tool,
                error: Some(format!(
                    "{tool} check exceeded {BUILD_CHECK_TIMEOUT:?} — treated as unchecked"
                )),
            };
        }
    };
    let _ = tokio::fs::remove_dir_all(&dir).await;
    match result {
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            if output.status.success() {
                BuildCheck {
                    checked: true,
                    ok: true,
                    tool,
                    error: None,
                }
            } else {
                BuildCheck {
                    checked: true,
                    ok: false,
                    tool,
                    error: Some(first_line(&stderr)),
                }
            }
        }
        Err(e) => BuildCheck {
            checked: false,
            ok: false,
            tool,
            error: Some(first_line(&e.to_string())),
        },
    }
}

/// Run the whole sub-pipeline: parse → patch → re-scan → build-check →
/// report. Every stage degrades typed; a refused or failed stage never
/// produces a half-truth report — the JSON carries per-stage status.
pub async fn refactor_module(
    cfg: &crate::config::BotConfig,
    code: &str,
    language: RefLang,
    instruction: &str,
    check: bool,
) -> Result<Value> {
    let code = code.trim();
    anyhow::ensure!(!code.is_empty(), "refactor_module: code is required");
    anyhow::ensure!(
        code.chars().count() <= MAX_CODE_CHARS,
        "refactor_module: code exceeds the cap ({MAX_CODE_CHARS} chars; got {})",
        code.chars().count()
    );
    let instruction = instruction.trim();
    anyhow::ensure!(
        !instruction.is_empty(),
        "refactor_module: instruction is required"
    );
    anyhow::ensure!(
        instruction.chars().count() <= MAX_INSTRUCTION_CHARS,
        "refactor_module: instruction exceeds {MAX_INSTRUCTION_CHARS} chars"
    );

    // Parse phase — a structurally broken input never reaches the model.
    let pre = structural_balance(code, language);
    if pre.balanced == Some(false) {
        return Ok(json!({
            "ok": false,
            "refused": format!("input module failed the structural scan: {}", pre.detail),
            "parse": pre,
            "language": language.name(),
        }));
    }

    // Patch phase.
    let raw_patch = refactor_patch(cfg, language, instruction, code).await?;
    let patched = extract_code_block(&raw_patch);
    anyhow::ensure!(!patched.is_empty(), "patch leg produced no code");
    let post = structural_balance(patched, language);

    // Build-check phase (bounded, no-execution tooling only).
    let build = if check {
        build_check(language, patched).await
    } else {
        BuildCheck {
            checked: false,
            ok: false,
            tool: "none",
            error: Some("checks disabled by request".into()),
        }
    };

    Ok(json!({
        "ok": true,
        "language": language.name(),
        "instruction": instruction,
        "refused": Value::Null,
        "parse": { "before": pre, "after": post },
        "build": build,
        "lines": { "before": code.lines().count(), "after": patched.lines().count() },
        "chars": { "before": code.chars().count(), "after": patched.chars().count() },
        "code": patched,
    }))
}

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
    vec![
        SkillSpec {
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
        },
        SkillSpec {
            name: "pipeline.refactor_module",
            plugin: "pipeline",
            description: "ONE-CALL macro-tool: structurally validate a source module, apply a \
                      refactoring instruction through the gateway in one reasoning step, \
                      re-scan the patched module, and run the language's cheapest real \
                      build-check (rustc metadata / python bytecode compile — never \
                      execution). Returns the patched source plus per-stage status.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The complete module source to refactor"},
                    "instruction": {"type": "string", "description": "What the refactor should do"},
                    "language": {"type": "string", "enum": ["rust", "python", "typescript", "javascript", "other"],
                                 "description": "Module language (default rust)"},
                    "checks": {"type": "boolean", "description": "Run the build-check leg (default true)"}
                },
                "required": ["code", "instruction"]
            }),
            run: refactor_module_runner,
        },
    ]
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

fn refactor_module_runner<'args>(
    cfg: &'args crate::config::BotConfig,
    args: &'args Value,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<Value>> + Send + 'args>> {
    Box::pin(async move {
        let code = args.get("code").and_then(Value::as_str).unwrap_or_default();
        let instruction = args
            .get("instruction")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let language = args
            .get("language")
            .and_then(Value::as_str)
            .map(RefLang::parse)
            .unwrap_or(RefLang::Rust);
        let checks = args.get("checks").and_then(Value::as_bool).unwrap_or(true);
        refactor_module(cfg, code, language, instruction, checks).await
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

    // --- refactor_module: structural scanner ---------------------------------

    #[test]
    fn rust_scanner_accepts_strings_lifetimes_chars_raw_and_comments() {
        let ok = "// header comment\nfn f<'a>(x: &'a str) -> &'static str {\n\
                  let c = 'x';\nlet esc = \"a\\\"b{[( )]}\";\n\
                  let raw = r#\"quotes \"and\" braces { \"#;\n\
                  let bytes = b\"byte [str]\";\n\
                  /* block /* nested */ comment */\n\
                  match list { Some(v) => v.len(), None => 0 }\n\
                  }\n";
        let r = structural_balance(ok, RefLang::Rust);
        assert_eq!(r.balanced, Some(true), "detail: {}", r.detail);
    }

    #[test]
    fn rust_scanner_rejects_unbalanced_and_unterminated() {
        let r = structural_balance("fn f() { let x = [1, 2;", RefLang::Rust);
        assert_eq!(r.balanced, Some(false));
        assert!(r.detail.contains("never closed") || r.detail.contains("unbalanced"));
        let r = structural_balance("let s = \"unterminated", RefLang::Rust);
        assert_eq!(r.balanced, Some(false));
        assert!(r.detail.contains("string"));
        // A stray closer is refused, not tolerated.
        let r = structural_balance("fn ok() {} }", RefLang::Rust);
        assert_eq!(r.balanced, Some(false));
        assert!(r.detail.contains("stray"));
    }

    #[test]
    fn python_scanner_handles_triple_quotes_and_comments() {
        let ok = "# header\ndef f(x):\n    \"\"\"doc {[(\n    string)]}\"\"\"\n    d = {'a': [1, (2, 3)]}\n    return f'{x}'\n";
        let r = structural_balance(ok, RefLang::Python);
        assert_eq!(r.balanced, Some(true), "detail: {}", r.detail);
        let bad = "def f(\n    s = \"\"\"unterminated";
        assert_eq!(
            structural_balance(bad, RefLang::Python).balanced,
            Some(false)
        );
    }

    #[test]
    fn js_scanner_handles_templates_and_interpolations() {
        let ok = "// header\nconst s = 'str {not code)';\n\
                  const t = `template ${list.map(x => ({k: x}))} tail`;\n\
                  function f() { return [1, 2]; }\n";
        let r = structural_balance(ok, RefLang::Javascript);
        assert_eq!(r.balanced, Some(true), "detail: {}", r.detail);
        let bad = "const t = `unterminated ;\n"; // one backtick only
        assert_eq!(
            structural_balance(bad, RefLang::Typescript).balanced,
            Some(false)
        );
    }

    #[test]
    fn other_language_report_is_unchecked_not_false() {
        let r = structural_balance("{ un ( balanced", RefLang::Other);
        assert!(!r.checked);
        assert_eq!(r.balanced, None);
    }

    #[test]
    fn fence_extraction_takes_first_block_and_survives_no_fence() {
        assert_eq!(
            extract_code_block("```rust\nfn a() {}\n```\n```python\nx\n```"),
            "fn a() {}"
        );
        assert_eq!(extract_code_block("```\nbare block\n```"), "bare block");
        assert_eq!(extract_code_block("no fence at all\n"), "no fence at all");
        // Unterminated fence: remainder still comes out as code.
        assert_eq!(extract_code_block("```rust\nfn b() {}"), "fn b() {}");
    }

    #[test]
    fn refactor_caps_refuse_before_any_leg_runs() {
        let cfg = crate::config::BotConfig::from_parts("http://127.0.0.1:1", "http://127.0.0.1:1");
        let big = "x".repeat(MAX_CODE_CHARS + 1);
        let err = tokio_block(&cfg, big.as_str(), RefLang::Rust, "clean it up", true);
        assert!(err.contains("exceeds the cap"), "got: {err}");
        let err = tokio_block(&cfg, "fn a() {}", RefLang::Rust, "", true);
        assert!(err.contains("instruction is required"), "got: {err}");
    }

    /// Sync affordance for the cap refusals (current_thread runtime).
    fn tokio_block(
        cfg: &crate::config::BotConfig,
        code: &str,
        lang: RefLang,
        instruction: &str,
        check: bool,
    ) -> String {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async { refactor_module(cfg, code, lang, instruction, check).await })
            .unwrap_err()
            .to_string()
    }

    #[test]
    fn build_check_refuses_no_tool_language_honestly() {
        let b = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async { build_check(RefLang::Other, "irrelevant").await });
        assert!(!b.checked);
        assert_eq!(b.tool, "none");
        assert!(b.error.unwrap().contains("no build-check tool"));
    }

    /// LIVE: requires a real `rustc` on PATH and compiles a tiny module —
    /// maintenance-only, like the scrape live harness:
    /// `cargo test -p jerith-mesh-bot live_build_check -- --ignored --nocapture`
    #[tokio::test]
    #[ignore = "requires rustc on PATH"]
    async fn live_build_check_rustc_metadata_pass_and_fail() {
        let good = build_check(RefLang::Rust, "pub fn one() -> u32 { 1 }").await;
        assert!(good.checked, "rustc absent: {:?}", good.error);
        assert!(good.ok, "error: {:?}", good.error);
        let bad = build_check(RefLang::Rust, "pub fn one() -> u32 { \"no\" }").await;
        assert!(bad.checked && !bad.ok);
        assert!(
            bad.error
                .as_deref()
                .unwrap_or_default()
                .contains("mismatched types")
        );
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
