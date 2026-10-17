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
        run_session(&cfg, &state, Session { user: 7, prompt: "ping".into() }, "", &mut boxed)
            .await
            .unwrap();
        // The pipeline consumed the sink — this is exactly why reading
        // `sink.is_none()` after the call is the bug shape:
        assert!(boxed.is_none(), "run_session consumed the sink");
        assert!(
            boxed.is_none() == true && had_live_sink == true,
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
}
