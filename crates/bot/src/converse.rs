//! The dialogue core (Phase 3b, `docs/PLUGIN-SDK.md`): the transport-
//! neutral conversation pipeline, one step upstream of the already-
//! transport-agnostic tool loop in `api.rs`.
//!
//! What a transport (today: Discord via poise/serenity; tomorrow:
//! Telegram, WhatsApp adapters) provides is a [`ConversationSink`] —
//! where fragments stream, where the final answer lands. What it must
//! NOT re-implement is the pipeline itself: squeeze, speaker tagging,
//! the streamed chat ladder with its tool rounds, usage accounting,
//! history persistence, reply formatting. That pipeline was duplicated
//! invariant-for-invariant across `/ask` and the @mention/DM fallback;
//! [`run_session`] is now its single definition and transports shrink
//! to thin front-ends.
//!
//! Deliberately transport-side (not absorbed here):
//! - the **cooldown gate** — refusals are displayed differently per
//!   transport and happen before any streaming surface exists;
//!   `gate_ask` is also allowance-consuming, so it must fire exactly
//!   once per request;
//! - the **error display** — `run_session` returns `Err` with the sink
//!   untouched so each transport finishes with its own wording and its
//!   own fallback send;
//! - post-chat transport side effects (e.g. `/ask`'s tally persistence).
//!
//! Nothing here imports serenity: fragments are best-effort, non-
//! blocking, sync; finals are the only awaited delivery. The in-memory
//! sink below drives the session tests with a real gateway stub, so the
//! whole pipeline is provable end-to-end without a Discord connection.

use crate::api;
use crate::commands::format_reply;
use crate::config::BotConfig;
use crate::state::BotState;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// Where the pipeline's user-visible output goes. One object per chat
/// request; implementors decide streaming (Discord live-edits a
/// placeholder) or buffering (a plain final send).
pub trait ConversationSink: Send + Sync {
    /// A per-round stream event: `RoundStart` erases the previous
    /// round's fragment slop, `Text` is a fragment of the current round.
    /// SYNC and non-blocking by contract — the pipeline bridges this
    /// straight into `api.rs`'s sync `LiveTap`, so implementations must
    /// not await (buffer into local state; a ticker/spawn does the slow
    /// transporter part, exactly how the Discord live sink works).
    fn on_event(&self, event: api::LiveEvent<'_>);

    /// The final formatted reply (or a transport-chosen failure string)
    /// replaces the streamed surface. Consumes the sink handle (Arc):
    /// the pipeline finishes it at most once, and the Arc consumption is
    /// what lets the chat-time tap hold its own clone without lifetime
    /// gymnastics (the api.rs tap discipline, made structural).
    fn finish(self: Arc<Self>, text: String) -> Pin<Box<dyn Future<Output = ()> + Send>>;
}

/// What a transport hands the pipeline: the per-user routing key plus
/// the raw prompt (speaker-tagged inside `run_session` in the
/// model-visible shape the transport provides).
pub struct Session {
    pub user: u64,
    pub prompt: String,
}

/// How a transport decides post-session delivery when a chat returns
/// `Ok`: a live sink that was in play has already been finished by the
/// pipeline (the text is delivered); a session that never had one
/// (buffered config, or the placeholder POST failed) still needs a plain
/// send. The "was a sink in play" bit must be captured BEFORE the
/// session: run_session consumes the Option, so testing `sink.is_none()`
/// afterwards is true for a session that DID live-stream — the live
/// 2026-10-17 09:55 double (every successful streamed chat re-posted the
/// same answer ~60 s later) was exactly that mis-test.
pub fn buffered_delivery_needed(had_live_sink: bool) -> bool {
    !had_live_sink
}

/// Default question used when a message carries attachments but no usable
/// text (image-only mentions and DMs): the vision tier describes, the
/// chat model narrates.
pub const IMAGE_ONLY_PROMPT: &str = "What's in this image? Describe it";

/// Cap on image attachments processed per message: each walk costs one
/// download + one vision-tier round-trip; beyond a few, the answer gets
/// unreadable anyway. Files ride the sandbox's own `[sandbox]` caps.
pub const MAX_IMAGES_PER_MESSAGE: usize = 3;

/// What a transport hands over per attachment: the CDN url, display
/// label (the filename), the kind that routes it — images through
/// the gateway's vision skill, everything else through the Docker
/// sandbox ([sandbox] config, see sandbox.rs) — plus the Discord
/// attachment id and byte size feeding the extraction cache.
#[derive(Debug, Clone)]
pub struct AttSpec {
    pub url: String,
    pub label: String,
    pub kind: AttKind,
    pub id: Option<u64>,
    pub size: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttKind {
    Image,
    Other,
}

/// Pure assembly behind the transports' attachment context: render one
/// bracketed line per attachment — description/extracted text or an
/// honest failure (never a silent drop) — into one block the chat
/// model reads ahead of the prompt.
fn assemble_attachment_context(lines: &[(String, Result<String, String>)]) -> Option<String> {
    if lines.is_empty() {
        return None;
    }
    let mut out = String::from("[attachment context follows]\n");
    for (label, result) in lines {
        match result {
            Ok(desc) if !desc.trim().is_empty() => {
                out.push_str(&format!("- [{label}] {}\n", desc.trim()));
            }
            Ok(_) => out.push_str(&format!("- [{label}] (no readable content)\n")),
            Err(e) => {
                out.push_str(&format!("- [{label}] (could not read: {e})\n"));
            }
        }
    }
    out.truncate(out.trim_end().len());
    Some(out)
}

/// Transport-side attachment handling: images get described through the
/// gateway's vision skill, files get read inside the disposable sandbox
/// container, and the whole story assembles into the bracketed context
/// block the chat model reads ahead of the prompt. `None` = nothing to
/// say, no block.
pub async fn attachments_context(
    cfg: &crate::config::BotConfig,
    state: &crate::state::BotState,
    attachments: &[AttSpec],
    question: Option<&str>,
) -> Option<String> {
    if attachments.is_empty() {
        return None;
    }
    // The user's own text describes what they want from the images; a
    // plain describe when the message rides by on pixels alone.
    let describe_prompt = question
        .map(str::trim)
        .filter(|q| !q.is_empty())
        .unwrap_or(IMAGE_ONLY_PROMPT);
    // Results land in ATTACHMENT ORDER (lines mirror the iteration; a
    // sandbox slot is pre-filled with a placeholder that the batch run
    // overwrites once its value lands).
    let mut lines: Vec<(String, Result<String, String>)> = Vec::with_capacity(attachments.len());
    let mut file_batch = Vec::new();
    let mut batch_meta: Vec<(Option<u64>, Option<u64>)> = Vec::new(); // (id, size) per batch entry
    let mut batch_slots: Vec<usize> = Vec::new(); // line slot per batch entry
    let mut images_used = 0usize;
    for att in attachments {
        match att.kind {
            AttKind::Image => {
                images_used += 1;
                if images_used > MAX_IMAGES_PER_MESSAGE {
                    continue; // the walk budget ends at the cap
                }
                let entry = match api::fetch_vision_image(&att.url, &att.label).await {
                    Ok(api::VisionImage { label, data_url }) => {
                        match api::vision_describe_skill(cfg, describe_prompt, &data_url).await {
                            Ok(desc) => (label, Ok(desc)),
                            Err(e) => (label, Err(format!("{e:#}"))),
                        }
                    }
                    Err(e) => (att.label.clone(), Err(format!("{e:#}"))),
                };
                lines.push(entry);
            }
            AttKind::Other => {
                if let Some(hit) = state.sandbox_cache_get(att.id, &att.label, att.size) {
                    lines.push((att.label.clone(), hit));
                    continue;
                }
                batch_meta.push((att.id, att.size));
                batch_slots.push(lines.len());
                lines.push((att.label.clone(), Err("extraction pending".to_string())));
                file_batch.push(crate::sandbox::FileAttachment {
                    url: att.url.clone(),
                    label: att.label.clone(),
                });
            }
        }
    }
    // One sandbox container reads every un-cached file in the batch —
    // capped by [sandbox] max_files (message order kept), each outcome
    // an honest line. Successes feed the extraction cache; failures stay
    // re-runnable on the next ask.
    file_batch.truncate(cfg.sandbox.max_files);
    batch_meta.truncate(cfg.sandbox.max_files);
    batch_slots.truncate(cfg.sandbox.max_files);
    if !file_batch.is_empty() {
        let out = crate::sandbox::read_attachments(&cfg.sandbox, &file_batch).await;
        for (out_i, (slot, (label, val))) in batch_slots.iter().zip(out).enumerate() {
            lines[*slot].1 = val;
            if lines[*slot].1.is_ok() {
                let (id, size) = batch_meta[out_i];
                state.sandbox_cache_put(id, &label, size, lines[*slot].1.clone());
            }
        }
    }
    assemble_attachment_context(&lines)
}

/// One request → reply conversational pass, defined once (was duplicated
/// in `/ask` and the @mention fallback): squeeze (deep stage, may
/// persist) → speaker tag → streamed chat with the sink bridged into
/// the loop's tap → usage record → history persist → formatted reply →
/// sink finish. Returns the formatted text and who served it; on chat
/// failure the error propagates and the sink stays untouched.
pub async fn run_session(
    cfg: &BotConfig,
    state: &BotState,
    session: Session,
    speaker_prefix: &str,
    sink: &mut Option<Arc<dyn ConversationSink>>,
) -> anyhow::Result<(String, Option<String>)> {
    let Session { user, prompt } = session;

    let history0 = state.history(user);
    let history = if cfg.squeeze.enabled {
        crate::squeeze::squeeze_deep(state, cfg, user, &history0).await
    } else {
        history0
    };
    let prompt = format!("{speaker_prefix}{prompt}");

    // Bridge the sink into the loop's sync tap: fragments flow to the
    // transport's live-typing surface without the transport knowing
    // about SSE or rounds at all. The borrow lives only as long as the
    // chat call (the sink handle still sits in `sink` for finish() below).
    let tap: Option<Box<api::LiveTap>> = sink.as_ref().map(|s| {
        // The tap owns an Arc clone of the sink — 'static, no borrowed
        // lifetime to outlive the request (same discipline as the
        // LiveSink::tap_for bridge in chat_fallback.rs).
        let s = s.clone();
        Box::new(move |ev: api::LiveEvent<'_>| {
            s.on_event(ev);
        }) as Box<api::LiveTap>
    });
    let live: Option<&api::LiveTap> = tap.as_deref();

    let (reply, served_by, tokens) =
        api::chat_with_usage_live(cfg, &prompt, &history, live).await?;
    state.record_usage(user, tokens);
    state.push_turn(user, &prompt, &reply);
    let text = format_reply(&reply, served_by.as_deref());
    if let Some(sink) = sink.take() {
        sink.finish(text.clone()).await;
    }
    Ok((text, served_by))
}

/// In-memory sink (the test transport): records what a user would have
/// seen. Streaming folds fragment order; `streaming = false` ignores
/// fragments entirely — the buffered-config shape.
#[derive(Default, Clone)]
#[cfg(test)]
pub struct MemorySink {
    inner: Arc<std::sync::Mutex<SinkLog>>,
    streaming: bool,
}

#[derive(Default)]
#[cfg(test)]
struct SinkLog {
    fragments: Vec<String>,
    finished: Option<String>,
}

#[cfg(test)]
impl MemorySink {
    pub fn new(streaming: bool) -> Self {
        Self {
            inner: Arc::new(std::sync::Mutex::new(SinkLog::default())),
            streaming,
        }
    }

    /// Streamed-then-final assembled text in delivery order.
    pub fn assembled(&self) -> String {
        let log = self.inner.lock().unwrap();
        let mut out = log.fragments.concat();
        if let Some(fin) = &log.finished {
            out.push_str(fin);
        }
        out
    }

    /// Just the final line (what a buffered transport shows).
    pub fn final_text(&self) -> Option<String> {
        self.inner.lock().unwrap().finished.clone()
    }
}

#[cfg(test)]
impl ConversationSink for MemorySink {
    fn on_event(&self, event: api::LiveEvent<'_>) {
        // Buffered config: fragments never surface.
        if !self.streaming
            && let api::LiveEvent::Text(_) = event
        {
            return;
        }
        if let api::LiveEvent::Text(frag) = event {
            let mut log = self.inner.lock().unwrap();
            log.fragments.push(frag.to_string());
        }
    }

    fn finish(self: Arc<Self>, text: String) -> Pin<Box<dyn Future<Output = ()> + Send>> {
        Box::pin(async move {
            self.inner.lock().unwrap().finished = Some(text);
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use std::sync::Arc as StdArc;
    use tokio::sync::Mutex as TMutex;

    fn cfg() -> crate::config::BotConfig {
        crate::config::BotConfig::from_parts("http://127.0.0.1:1", "http://127.0.0.1:1")
    }

    /// Real gateway stub: `/status` for model detection, streaming SSE
    /// chat (two fragments then [DONE]), plus the non-streaming shape
    /// for the buffered-config test.
    async fn stub_gateway(seen: StdArc<TMutex<Vec<Value>>>) -> String {
        use axum::response::IntoResponse as _;
        let seen_chat = seen.clone();
        let app = axum::Router::new()
            .route(
                "/status",
                axum::routing::get(|| async { axum::Json(json!({ "model": "stub:8b" })) }),
            )
            .route(
                "/v1/chat/completions",
                axum::routing::post(move |axum::Json(req): axum::Json<Value>| {
                    let seen_chat = seen_chat.clone();
                    async move {
                        seen_chat.lock().await.push(req.clone());
                        let wants_stream = req
                            .get("stream")
                            .and_then(Value::as_bool)
                            .unwrap_or(false);
                        if wants_stream {
                            let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello \"}}]}\n\n\
                                      data: {\"choices\":[{\"delta\":{\"content\":\"mesh\"}}]}\n\n\
                                      data: [DONE]\n\n";
                            axum::response::Response::builder()
                                .header("content-type", "text/event-stream")
                                .body(axum::body::Body::from(sse))
                                .unwrap()
                        } else {
                            axum::response::Json(json!({
                                "choices": [{
                                    "message": {"role": "assistant", "content": "Hello mesh"},
                                    "mesh_fallback": Value::Null
                                }]
                            }))
                                .into_response()
                        }
                    }
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn pipeline_end_to_end_streams_and_persists() {
        let cfg_url = stub_gateway(StdArc::new(TMutex::new(Vec::new()))).await;
        let mut cfg = cfg();
        cfg.gateway_url = cfg_url;
        cfg.chat_stream = true;
        let state = crate::state::new_state();

        let sink = MemorySink::new(true);
        let mut boxed = Some(Arc::new(sink.clone()) as Arc<dyn ConversationSink>);
        let (text, served_by) = run_session(
            &cfg,
            &state,
            Session {
                user: 42,
                prompt: "ping".into(),
            },
            "[Discord user: tester (username tester, id 42)]\n",
            &mut boxed,
        )
        .await
        .unwrap();

        assert!(text.contains("Hello mesh"), "reply: {text}");
        assert_eq!(served_by, None, "the stub gateway answered locally");
        // The streamed surface assembled the fragments in order and then
        // the final replacement — live-typing, proven transport-side.
        let shown = sink.assembled();
        assert!(
            shown.starts_with("Hello mesh") && shown.contains(&format_reply("Hello mesh", None)),
            "assembly: {shown:?}"
        );
        assert!(boxed.is_none(), "run_session consumed the sink on success");
        // Pipeline persisted the turn (user + assistant) against the
        // tagged prompt — the model's speaker awareness rides history.
        let hist = state.history(42);
        assert_eq!(hist.len(), 2, "turn persisted: {hist:?}");
        let user_text = hist[0]["content"].as_str().unwrap();
        assert!(
            user_text.starts_with("[Discord user: tester"),
            "tagged: {user_text}"
        );
        assert_eq!(hist[1]["content"].as_str().unwrap(), "Hello mesh");
    }

    #[tokio::test]
    async fn buffered_config_shows_only_the_final() {
        let cfg_url = stub_gateway(StdArc::new(TMutex::new(Vec::new()))).await;
        let mut cfg = cfg();
        cfg.gateway_url = cfg_url;
        cfg.chat_stream = false; // buffered: the bot posts one message
        let state = crate::state::new_state();

        let sink = MemorySink::new(false);
        let mut boxed = Some(Arc::new(sink.clone()) as Arc<dyn ConversationSink>);
        let (text, _) = run_session(
            &cfg,
            &state,
            Session {
                user: 9,
                prompt: "ping".into(),
            },
            "",
            &mut boxed,
        )
        .await
        .unwrap();
        assert_eq!(sink.final_text().as_deref(), Some(text.as_str()));
        let hist = state.history(9);
        assert_eq!(hist.len(), 2);
        assert!(
            !text.contains("— served by"),
            "local answers carry no mesh attribution"
        );
    }

    /// The live 2026-10-17 09:55 double: every successful streamed chat
    /// re-posted the identical answer ~60 s later as a fresh Discord
    /// message. Front-ends decided buffered delivery by testing
    /// `sink.is_none()` AFTER run_session — always true, because the
    /// pipeline consumes the Option — so `send_long` fired on every Ok.
    /// The gate must read the PRE-session sink presence instead.
    #[tokio::test]
    async fn post_session_delivery_gate_reads_pre_session_sink_presence() {
        let cfg_url = stub_gateway(StdArc::new(TMutex::new(Vec::new()))).await;
        let mut cfg = cfg();
        cfg.gateway_url = cfg_url;
        cfg.chat_stream = true;
        let state = crate::state::new_state();

        // A live surface was in play when the session started.
        let mut boxed = Some(Arc::new(MemorySink::new(true)) as Arc<dyn ConversationSink>);
        let had_live_sink = boxed.is_some();
        run_session(
            &cfg,
            &state,
            Session {
                user: 7,
                prompt: "ping".into(),
            },
            "",
            &mut boxed,
        )
        .await
        .unwrap();
        // The pipeline consumed the sink — this is exactly why reading
        // `sink.is_none()` after the call is the bug shape:
        assert!(boxed.is_none(), "run_session consumed the sink");
        assert!(
            boxed.is_none() && had_live_sink,
            "post-session is_none() is true even though a live surface delivered — pre-session capture required"
        );
        // The rule: a consumed live surface means the text is already
        // delivered — a second (buffered) send would be the double.
        assert!(
            !buffered_delivery_needed(had_live_sink),
            "live-delivered session must not re-post"
        );
        // A session with no surface at all still needs the plain send.
        assert!(buffered_delivery_needed(false));
    }

    /// Attachment handling: the context line assembly — descriptions
    /// (*the user's question rides along as the skill prompt*) and
    /// sandbox text, honest failure lines for broken downloads/replies,
    /// and a nothing case.
    #[test]
    fn assemble_attachment_context_renders_each_line_honestly() {
        let lines = vec![
            (
                "kiln.png".to_string(),
                Ok::<String, String>(String::from("  a brick kiln, fired   ")),
            ),
            (
                "ghost.png".to_string(),
                Err(String::from("vision skill failed: 503")),
            ),
            ("mute.png".to_string(), Ok(String::from(""))),
            (
                "notes.txt".to_string(),
                Ok(String::from("(binary file, 42 bytes — no text extracted)")),
            ),
        ];
        let ctx = assemble_attachment_context(&lines).unwrap();
        assert!(
            ctx.starts_with("[attachment context follows]"),
            "header: {ctx}"
        );
        assert!(ctx.contains("- [kiln.png] a brick kiln, fired"), "{ctx}");
        assert!(
            ctx.contains("- [ghost.png] (could not read: vision skill failed: 503)"),
            "{ctx}"
        );
        assert!(ctx.contains("- [mute.png] (no readable content)"), "{ctx}");
        assert!(
            ctx.contains("- [notes.txt] (binary file, 42 bytes"),
            "{ctx}"
        );
        // No attachments, no block (and the transports skip the plainly).
        assert_eq!(assemble_attachment_context(&[]), None);
    }

    /// Transport-side attachment context end-to-end against a stub
    /// gateway: the image attachment URL downloads into a data URL, the
    /// user's question becomes the skill's prompt, and the broken
    /// attachment lands as the honest failure line instead of killing
    /// the answer.
    #[tokio::test]
    async fn attachments_context_describes_and_reports_failures() {
        let seen = StdArc::new(TMutex::new(Vec::new()));
        let seen_describe = seen.clone();
        let app = axum::Router::new()
            .route(
                "/img.png",
                axum::routing::get(|| async {
                    (
                        [("content-type", "image/png")],
                        axum::body::Bytes::from(vec![1u8, 2, 3, 4]),
                    )
                }),
            )
            .route(
                "/v1/skills/vision/describe",
                axum::routing::post(move |axum::Json(req): axum::Json<Value>| {
                    let seen_describe = seen_describe.clone();
                    async move {
                        seen_describe.lock().await.push(req.clone());
                        axum::Json(json!({
                            "choices": [{
                                "message": {"role": "assistant", "content": "a red brick kiln"}
                            }]
                        }))
                    }
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let mut cfg = cfg();
        cfg.gateway_url = format!("http://{addr}");

        let url = format!("http://{addr}/img.png");
        let state = crate::state::new_state();
        let ctx = attachments_context(
            &cfg,
            &state,
            &[
                AttSpec {
                    url: url.clone(),
                    label: "kiln.png".into(),
                    kind: AttKind::Image,
                    id: None,
                    size: None,
                },
                AttSpec {
                    url: "http://127.0.0.1:1/missing.png".into(),
                    label: "ghost.png".into(),
                    kind: AttKind::Image,
                    id: None,
                    size: None,
                },
            ],
            Some("what is it?"),
        )
        .await
        .unwrap();

        let ok_line = "- [kiln.png] a red brick kiln";
        assert!(
            ctx.contains("[attachment context follows]") && ctx.contains(ok_line),
            "ctx: {ctx}"
        );
        assert!(
            ctx.contains("- [ghost.png] (could not read: image download"),
            "{ctx}"
        );
        // The user's question — not the describe default — rode the
        // vision skill's prompt.
        let skill_reqs = seen.lock().await;
        assert_eq!(skill_reqs.len(), 1, "only the good image reached the skill");
        assert_eq!(skill_reqs[0]["prompt"], json!("what is it?"));
        assert!(
            skill_reqs[0]["image"]
                .as_str()
                .unwrap()
                .starts_with("data:image/png;base64,"),
            "bytes packed as a data url"
        );
    }

    /// File attachments ride the sandbox path: a text file and a binary
    /// blob through one container batch, honest lines both ways (host
    /// has docker; skips rather than fails where it doesn't).
    #[tokio::test]
    async fn attachments_context_reads_files_in_the_sandbox() {
        // Two gates: a reachable daemon AND the sandbox image built (CI has
        // Docker but never builds jerith-sandbox — exit 125 'repository does
        // not exist' is an environment gap, not a test failure).
        let probe = tokio::process::Command::new("docker")
            .arg("info")
            .output()
            .await;
        if !probe.map(|p| p.status.success()).unwrap_or(false) {
            eprintln!("docker not available — skipping live sandbox test");
            return;
        }
        let image = cfg().sandbox.image;
        if image.is_empty()
            || !tokio::process::Command::new("docker")
                .args(["image", "inspect", &image])
                .output()
                .await
                .map(|p| p.status.success())
                .unwrap_or(false)
        {
            eprintln!("sandbox image {image:?} not built here — skipping live sandbox test");
            return;
        }
        // a text attachment and a binary one, served by the stub gateway
        let app = axum::Router::new()
            .route(
                "/note.txt",
                axum::routing::get(|| async {
                    (
                        [("content-type", "text/plain")],
                        axum::body::Bytes::from_static(b"kiln schedule: fire at 980"),
                    )
                }),
            )
            .route(
                "/blob.bin",
                axum::routing::get(|| async {
                    (
                        [("content-type", "application/octet-stream")],
                        axum::body::Bytes::from(vec![0xFF, 0x00, 0xFE, 0x01]),
                    )
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let cfg = cfg();

        let state = crate::state::new_state();
        let ctx = attachments_context(
            &cfg,
            &state,
            &[
                AttSpec {
                    url: format!("http://{addr}/note.txt"),
                    label: "note.txt".into(),
                    kind: AttKind::Other,
                    id: Some(11),
                    size: Some(27),
                },
                AttSpec {
                    url: format!("http://{addr}/blob.bin"),
                    label: "blob.bin".into(),
                    kind: AttKind::Other,
                    id: Some(12),
                    size: Some(4),
                },
            ],
            Some("summarize what was sent"),
        )
        .await
        .unwrap();
        assert!(
            ctx.contains("- [note.txt] kiln schedule: fire at 980"),
            "sandbox text: {ctx}"
        );
        assert!(ctx.contains("- [blob.bin] (binary file"), "{ctx}");
        // The extraction landed in the cache: a SECOND pass with live
        // urls REMOVED (pointing nowhere) still returns the same text —
        // the note.txt id+name+size keys hit, re-forwarded files are free.
        let dead = format!("http://{addr}/removed-note.txt");
        let again = attachments_context(
            &cfg,
            &state,
            &[AttSpec {
                url: dead,
                label: "note.txt".into(),
                kind: AttKind::Other,
                id: Some(11),
                size: Some(27),
            }],
            Some("summarize what was sent"),
        )
        .await
        .unwrap();
        assert!(
            again.contains("- [note.txt] kiln schedule: fire at 980"),
            "cache hit: {again}"
        );
    }
}
