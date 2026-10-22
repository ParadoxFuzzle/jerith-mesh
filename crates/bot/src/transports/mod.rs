//! External chat transports (docs/PLUGIN-SDK.md Role 2). Discord stays the
//! compiled-in first-class surface (poise/serenity); every other platform
//! here is a thin adapter of the SAME two lanes the SDK proposes:
//!
//!     inbound  = platform event → [`Incoming`] → the dialogue core
//!     outbound = final reply text → the platform's send API
//!
//! Nothing a transport does may re-implement the pipeline: ALL of them
//! route through `converse::run_session` (squeeze → speaker tag → chat →
//! bookkeeping → reply formatting), with the cooldown gate fired
//! transport-side exactly once per inbound message, per the documented
//! split. Each adapter is DORMANT until its `[transports.<platform>]`
//! table lands in bot.toml (no table = no task spawned, zero cost); a
//! typed startup refusal — never a silent guess — states what a platform
//! cannot offer this build (see gg.rs, teams.rs / google_chat.rs ingest).
//!
//! History/cooldown keys: platforms assign a stable u64 user key via
//! [`user_key`] (FNV-1a over `platform:user`), so per-user histories
//! never collide across transports even when raw ids look numerically
//! alike. Transports are guest surfaces by construction (the owner-DM
//! privilege model lives in Discord ids, owner.rs) — a platform config
//! may explicitly map helper identities to owner-equivalent treatment
//! later; none does yet, so `owner_privileged` is false everywhere.
//!
//! Chunking: each platform caps answers differently; [`chunk_utf8`]
//! splits on a char count WITHOUT splitting runes (a byte-splitter would
//! shred emoji and CJK). It is implemented once here and every sender
//! reuses it with its platform cap.

pub(crate) mod facebook;
pub(crate) mod feishu;
pub(crate) mod gg;
pub(crate) mod google_chat;
pub(crate) mod irc;
pub(crate) mod mattermost;
pub(crate) mod nextcloud;
pub(crate) mod signal;
pub(crate) mod slack;
pub(crate) mod teams;
pub(crate) mod telegram;
pub(crate) mod twitch;
pub(crate) mod webinbox;

use crate::config::BotConfig;
use crate::state::BotState;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// Shared outbound HTTP call for every HTTP-backend adapter: the api.rs
/// client (shared pool, timeouts respected), optional bearer, JSON in/out.
pub(crate) async fn http_json(
    method: reqwest::Method,
    url: &str,
    bearer: Option<&str>,
    body: Option<Value>,
    timeout: std::time::Duration,
) -> anyhow::Result<(u16, Value)> {
    let mut req = crate::api::client().request(method, url).timeout(timeout);
    if let Some(b) = bearer {
        req = req.bearer_auth(b);
    }
    if let Some(v) = body {
        req = req.json(&v);
    }
    let resp = req.send().await.with_context_(url)?;
    let status = resp.status().as_u16();
    let text = resp.text().await.unwrap_or_default();
    let parsed = serde_json::from_str::<Value>(&text).unwrap_or(Value::String(text.clone()));
    Ok((status, parsed))
}

/// tiny ctx shim (anyhow::Context path kept local to this module)
trait CtxShim {
    fn with_context_(self, url: &str) -> anyhow::Result<reqwest::Response>;
}
impl CtxShim for std::result::Result<reqwest::Response, reqwest::Error> {
    fn with_context_(self, url: &str) -> anyhow::Result<reqwest::Response> {
        self.map_err(|e| anyhow::anyhow!("http {url}: {e}"))
    }
}

/// Normalized inbound message — everything the dialogue core needs, and
/// nothing platform-specific leaks past this point.
#[derive(Debug, Clone)]
pub struct Incoming {
    pub platform: &'static str,
    /// Stable channel/group id (IRC channel, Slack channel, room token…).
    pub channel: String,
    /// Stable author id as a *string* (platform ids are heterogeneous).
    pub user: String,
    pub display: String,
    pub text: String,
}

/// The per-transport reply seam: deliver ONE reply chunk to ONE channel.
/// Transport-specific chunking happens INSIDE this closure (the sender
/// knows its cap); by the time the core hands text over it is whole.
pub type DeliveryFn = Arc<
    dyn Fn(&str, String) -> Pin<Box<dyn Future<Output = anyhow::Result<()>> + Send>> + Send + Sync,
>;

/// A stable per-user key for history + cooldown, collision-free across
/// transports (the platform tag is hashed IN — a numeric Telegram id can
/// never alias a Discord snowflake's lane).
pub fn user_key(platform: &str, user: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in format!("{platform}:{user}").as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x1_0000_001b);
    }
    h
}

/// Rune-safe chunking at a char cap: clean splits at newline (then
/// space, then the hard boundary), never mid-rune, never mid-emoji.
pub fn chunk_utf8(text: &str, max_chars: usize) -> Vec<String> {
    assert!(
        max_chars >= 16,
        "chunk cap {max_chars} is too small to be sane"
    );
    let mut out = Vec::new();
    let mut rest = text;
    loop {
        if rest.chars().count() <= max_chars {
            if !rest.is_empty() {
                out.push(rest.to_string());
            }
            return out;
        }
        let cut_at = rest
            .char_indices()
            .nth(max_chars)
            .map(|(i, _)| i)
            .unwrap_or(rest.len());
        // Walk back to a clean split point; prefer a newline, then a space.
        let window = &rest[..cut_at];
        let split = match window.rfind('\n') {
            Some(p) if p > 0 => p,
            Some(_) => cut_at,
            None => match window.rfind(' ') {
                Some(p) if p > 0 => p,
                Some(_) | None => cut_at,
            },
        };
        out.push(rest[..split].trim_end().to_string());
        let next = rest[split..].trim_start().to_string();
        if next.is_empty() {
            return out;
        }
        rest = Box::leak(next.into_boxed_str());
    }
}

/// Final-delivery sink: buffers the whole session, then hands the whole
/// text to the transport's delivery closure ONCE (the closure chunks).
/// Streamed fragments are a Discord-only affordance today; external
/// transports ship buffered — correct, cheap, one code path.
struct BufferedSink {
    inc: Incoming,
    deliver: DeliveryFn,
}

impl crate::converse::ConversationSink for BufferedSink {
    fn on_event(&self, _ev: crate::api::LiveEvent<'_>) {}
    fn finish(self: Arc<Self>, text: String) -> Pin<Box<dyn Future<Output = ()> + Send>>
    where
        Self: Sized + 'static,
    {
        let inc = self.inc.clone();
        let deliver = Arc::clone(&self.deliver);
        Box::pin(async move {
            if let Err(e) = deliver(&inc.channel, text).await {
                tracing::warn!(platform = inc.platform, channel = %inc.channel, error = %e, "transport final-send failed");
            }
        })
    }
}

/// One inbound message through the shared pipeline: gate once → session →
/// delivery. Transports only parse events and push (channel, user,
/// display, text) here — everything downstream is the core's job.
pub async fn handle_incoming(
    cfg: &BotConfig,
    state: &BotState,
    inc: Incoming,
    // Reserved for a future owner-identity mapping (see module doc):
    // transports are guest surfaces today, so the field is stored but
    // deliberately NOT wired to owner_privileged.
    _owner_id: Option<u64>,
    deliver: DeliveryFn,
) {
    let key = user_key(inc.platform, &inc.user);
    if let Err(msg) = state.gate_ask(key) {
        // Cooldown refusals answer in the transport's own voice (Discord
        // keeps its wording; here, terse + suppressed elsewhere).
        let note = format!("⏳ {msg}");
        if let Err(e) = deliver(&inc.channel, note).await {
            tracing::debug!(platform = inc.platform, error = %e, "cooldown notice undeliverable");
        }
        return;
    }
    let speaker = format!("[{}:{} @{}] ", inc.platform, inc.display, inc.channel);
    let text = inc.text.clone();
    let mut sink: Option<Arc<dyn crate::converse::ConversationSink>> =
        Some(Arc::new(BufferedSink {
            inc: inc.clone(),
            deliver: Arc::clone(&deliver),
        }));
    // Pre-session capture of the "live surface in play" bit — run_session
    // consumes the Option, so testing it afterwards is the classic
    // double-post mistake (converse.rs documents the 2026-10-17 live fire).
    let had_live_sink = sink.is_some();
    let res = crate::converse::run_session(
        cfg,
        state,
        crate::converse::Session {
            user: key,
            prompt: text,
            // Transports are guest surfaces: nothing maps a platforms
            // identity to the owner here (see module doc).
            owner_privileged: false,
        },
        &speaker,
        &mut sink,
    )
    .await;
    match res {
        Ok((reply, _served_by)) => {
            if crate::converse::buffered_delivery_needed(had_live_sink) {
                // A live sink was already finished inside the pipeline;
                // this fires only when no sink pre-existed.
                let _ = deliver(&inc.channel, reply).await;
            }
        }
        Err(e) => {
            tracing::warn!(platform = inc.platform, user = %inc.user, error = %e, "transport chat session failed");
            // The chat_fallback wording, transport-flavored: include the
            // real cause (gateway window, 401, slow-answer timeout are
            // different operator fixes) — never a blank "try again".
            let mut reason = e.to_string().replace('\n', "; ");
            if reason.chars().count() > 160 {
                let tail: String = reason.chars().skip(reason.chars().count() - 160).collect();
                reason = format!("…{tail}");
            }
            if let Err(e2) = deliver(
                &inc.channel,
                format!("⚠️ the mesh didn't answer just now — {reason}"),
            )
            .await
            {
                tracing::debug!(error = %e2, "transport error-send also failed");
            }
        }
    }
}

/// Launch every configured transport as its own supervised task. Called
/// from main() AFTER BotState exists: tasks are independent of the
/// Discord gateway (they hold cfg + state only), so a platform outage
/// never wobbles Discord, and vice versa.
pub async fn spawn_all(cfg: Arc<BotConfig>, state: BotState, owner_id: Option<u64>) {
    let run = |name: &'static str,
               fut_fut: Pin<Box<dyn Future<Output = anyhow::Result<()>> + Send>>| {
        supervise(name, fut_fut);
    };
    trait_cfg_dispatch(&cfg, &state, owner_id, &run);
    if let Some(w) = &cfg.transports.webinbox {
        let w = w.clone();
        let cfg2 = Arc::clone(&cfg);
        run(
            "webinbox",
            Box::pin(async move {
                webinbox::run(cfg2, &w).await;
                Ok(())
            }),
        );
    }
}

/// The spawn-seam shape: name + the transport task's own future. A
/// first-class alias keeps the dispatch signature readable.
type TransportRun = dyn Fn(&'static str, Pin<Box<dyn Future<Output = anyhow::Result<()>> + Send>>);

fn trait_cfg_dispatch(
    cfg: &Arc<BotConfig>,
    state: &BotState,
    owner_id: Option<u64>,
    run: &TransportRun,
) {
    macro_rules! spawn_if {
        ($field:ident, $adapter:ident, $name:literal) => {
            if let Some(c) = &cfg.transports.$field {
                let c = c.clone();
                let cfg2 = Arc::clone(cfg);
                let state2 = Arc::clone(state);
                let owner = owner_id;
                run(
                    $name,
                    Box::pin(async move { $adapter::run(cfg2, state2, c, owner).await }),
                );
            }
        };
    }
    spawn_if!(telegram, telegram, "telegram");
    spawn_if!(irc, irc, "irc");
    spawn_if!(slack, slack, "slack");
    spawn_if!(twitch, twitch, "twitch");
    spawn_if!(mattermost, mattermost, "mattermost");
    spawn_if!(signal, signal, "signal");
    spawn_if!(nextcloud, nextcloud, "nextcloud");
    spawn_if!(teams, teams, "teams");
    spawn_if!(google_chat, google_chat, "google_chat");
    spawn_if!(feishu, feishu, "feishu");
    spawn_if!(whatsapp, facebook, "whatsapp");
    spawn_if!(gg, gg, "gg");
}

/// Restart-supervision: transpORY tasks disconnect eventually (net wobble,
/// platform restarts). A failing task NEVER wedges anything else: exponential
/// backoff capped at 5 min, respawn forever; a normal exit (typed refusal) logs
/// and stops.
fn supervise(
    name: &'static str,
    mut first: Pin<Box<dyn Future<Output = anyhow::Result<()>> + Send>>,
) {
    tokio::spawn(async move {
        match first.as_mut().await {
            Ok(()) => {
                tracing::info!(transport = name, "transport exited normally");
            }
            Err(e) => {
                // Every adapter loops its own reconnects internally; an
                // Err escaping to supervision means a config-level fatal
                // (validation, auth refused at boot) — log loud and stop.
                tracing::error!(
                    transport = name,
                    error = %e,
                    "transport failed permanently — fix config, restart bot"
                );
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Config root ([transports] in bot.toml)
// ---------------------------------------------------------------------------

/// Every optional platform table. Absent table = dormant, free, zero net
/// cost (tokio spawn is skipped by table presence). Each platform's own
/// struct validates its required fields at LOAD (fail-loud, like every
/// other bot.toml section) — a typo never produces a silently dead task.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TransportsConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub telegram: Option<telegram::TelegramConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub irc: Option<irc::IrcConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slack: Option<slack::SlackConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub twitch: Option<twitch::TwitchConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mattermost: Option<mattermost::MattermostConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signal: Option<signal::SignalConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nextcloud: Option<nextcloud::NextcloudConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teams: Option<teams::TeamsConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub google_chat: Option<google_chat::GoogleChatConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feishu: Option<feishu::FeishuConfig>,
    /// `[transports.whatsapp]` (Cloud API) — lives in facebook.rs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub whatsapp: Option<facebook::WhatsappConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gg: Option<gg::GgConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub webinbox: Option<webinbox::WebinboxConfig>,
}

impl TransportsConfig {
    /// Sum of enabled platforms (used by main() to decide the spawn log).
    pub fn enabled_platforms(&self) -> Vec<&'static str> {
        let mut v = Vec::new();
        if self.telegram.is_some() {
            v.push("telegram");
        }
        if self.irc.is_some() {
            v.push("irc");
        }
        if self.slack.is_some() {
            v.push("slack");
        }
        if self.twitch.is_some() {
            v.push("twitch");
        }
        if self.mattermost.is_some() {
            v.push("mattermost");
        }
        if self.signal.is_some() {
            v.push("signal");
        }
        if self.nextcloud.is_some() {
            v.push("nextcloud");
        }
        if self.teams.is_some() {
            v.push("teams");
        }
        if self.google_chat.is_some() {
            v.push("google_chat");
        }
        if self.feishu.is_some() {
            v.push("feishu");
        }
        if self.whatsapp.is_some() {
            v.push("whatsapp");
        }
        if self.gg.is_some() {
            v.push("gg");
        }
        if self.webinbox.is_some() {
            v.push("webinbox");
        }
        v
    }
}

impl BotConfig {
    // (transports field lives directly on BotConfig; no helpers needed.)
    #[allow(dead_code)]
    /// See docs for the transports' pipeline contract; this impl block is
    /// intentionally empty (compile-checked marker only).
    fn _doc_marker(&self) {}
}
