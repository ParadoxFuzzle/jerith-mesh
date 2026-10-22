//! Human-in-the-loop approval bridge (FEATURES §15).
//!
//! The agent can carry a task right up to a critical side effect — a
//! payment API call, an email send, a purchase flow, a privileged
//! external-drop skill — park exactly that step as a pending approval,
//! and let its owner approve / correct / deny it from a chat client with
//! one action. Nothing privileged executes until someone answers.
//!
//! Flow (a deterministic Rust state machine — the local-first analogue
//! of a LangGraph `interrupt_before` checkpoint):
//!
//! 1. The model calls `approval.request` with a deferred action:
//!    - `skill` — an existing registry skill + its args (for example
//!      `workflow.run` over a recorded purchase playbook), or
//!    - `http_call` — a method/URL/body against an owner-allowlisted
//!      host (raw API payments, webhook triggers).
//!
//!    The request is persisted (0600 JSON, one file per request) and
//!    immediately answered to the model with a pending id — the tool
//!    loop never blocks on a human.
//! 2. Notifications fan out: a signed webhook event (hooks.rs — covers
//!    Slack bridges, ntfy, WhatsApp relays…), an owner DM in Discord
//!    with one-click **Approve / Deny** buttons, and a Telegram chat
//!    with an inline keyboard (poller — no open ports).
//! 3. The owner answers. Discord button presses and Telegram callbacks
//!    resolve the id (identity-checked: only the configured owner /
//!    Telegram chat may resolve). Correct-approvals patch the deferred
//!    args before execution.
//! 4. Resolution spawns execution of the deferred action through the
//!    same registry dispatch every chat uses; the result lands as a
//!    follow-up notification. Deny / expiry close the request without
//!    executing anything.
//!
//! Trust model (fail-closed, same discipline as OWNER_ONLY_PLUGINS):
//! - **Only the configured approver can resolve** — the bot owner's
//!   Discord id, or the configured Telegram chat id. Everyone else's
//!   clicks/messages are refused.
//! - **Deferred `skill` actions are down-scoped by construction**: the
//!   named skill must exist in the NON-owner-privileged registry of the
//!   current config. Owner-only drops (`external:gcp`, coordinator,
//!   workflows-as-plugin-gate) execute live in owner-DM sessions
//!   instead of deferring through this bridge — a guest-reachable
//!   deferred executor must never escalate beyond what a plain session
//!   can already do; the owner approving is the *send button*, not a
//!   privilege grant.
//! - **Deferred `http_call` actions execute only against allowlisted
//!   hosts** (`[approvals] http_call_allowlist`), https-only, size-
//!   bounded, method-restricted — the raw "execute an API payment"
//!   lane the owner chooses to expose. Validated twice: at deferral
//!   time and again at execution time.
//! - Request ids are unguessable 40-bit codes; resolution is one-shot
//!   and race-safe (the first resolution wins, later ones stable no-ops
//!   that never re-execute).
//! - Notification payloads never carry model args verbatim: the
//!   channels are relays (and webhook receivers are third parties),
//!   so they carry the summary + a coarse action description only.

use anyhow::{Context as _, Result};
use poise::serenity_prelude as serenityall;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

/// Local boxed-future alias for the skill-runner signatures (workflows.rs
/// pacing — plugins.rs's own helper is module-private).
type BoxFut<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

// ---------------------------------------------------------------------------
// Config ([approvals] in bot.toml)
// ---------------------------------------------------------------------------

/// `[approvals]` in bot.toml — the human-in-the-loop bridge. Enabled by
/// default (requests without any approver channel fail loudly per-chat,
/// they never route nowhere); each notification channel is opt-in.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ApprovalConfig {
    /// Master switch. False → `approval.request` says the bridge is off.
    #[serde(default = "default_approvals_enabled")]
    pub enabled: bool,
    /// Discord owner DM with Approve/Deny buttons (needs
    /// `[owner] discord_id`; see owner.rs). Default on.
    #[serde(default = "default_discord_owner_dm")]
    pub discord_owner_dm: bool,
    /// Telegram bot token (from @BotFather) — inline-keyboard approvals
    /// in the chat below. Requires `telegram_chat_id`.
    #[serde(default)]
    pub telegram_bot_token: Option<String>,
    /// Telegram chat id allowed to resolve (the owner's chat with the
    /// bot; from @userinfobot). Requires the token.
    #[serde(default)]
    pub telegram_chat_id: Option<i64>,
    /// How long a pending approval stays open before the sweeper expires
    /// it (clamped 1–2880 minutes at load).
    #[serde(default = "default_ttl_minutes")]
    pub ttl_minutes: u64,
    /// Http-call allowlist: exact hostname entries the deferred
    /// `http_call` lane may POST to (`https://` enforced — the scheme
    /// belongs in the URL, never in a host entry). Empty → refuse-only.
    #[serde(default)]
    pub http_call_allowlist: Vec<String>,
    /// Additional webhook hops for the notification itself, beyond the
    /// hooks.rs event stream (which already carries signed approval
    /// events to every `[hooks] urls` entry).
    #[serde(default)]
    pub notify_webhooks: Vec<String>,
    /// How long resolved records are kept for `approval.status`
    /// (clamped 1–168 h at load).
    #[serde(default = "default_prune_hours")]
    pub prune_resolved_hours: u64,
}

fn default_approvals_enabled() -> bool {
    true
}
fn default_discord_owner_dm() -> bool {
    true
}
fn default_ttl_minutes() -> u64 {
    60
}
fn default_prune_hours() -> u64 {
    48
}

impl Default for ApprovalConfig {
    fn default() -> Self {
        Self {
            enabled: default_approvals_enabled(),
            discord_owner_dm: default_discord_owner_dm(),
            telegram_bot_token: None,
            telegram_chat_id: None,
            ttl_minutes: default_ttl_minutes(),
            http_call_allowlist: Vec::new(),
            notify_webhooks: Vec::new(),
            prune_resolved_hours: default_prune_hours(),
        }
    }
}

impl ApprovalConfig {
    /// Fail-loud channel validation (hooks.rs discipline): a mistyped
    /// Telegram pair or a scheme inside an allowlist entry would surface
    /// as a silent delivery/exec failure on every request otherwise.
    pub fn validate(&self) -> anyhow::Result<()> {
        let tg_token = self.telegram_bot_token.as_deref().map(str::trim);
        if tg_token.is_some() != self.telegram_chat_id.is_some() {
            anyhow::bail!(
                "[approvals] telegram_bot_token and telegram_chat_id must be set together"
            );
        }
        for entry in &self.http_call_allowlist {
            let host = entry.trim();
            anyhow::ensure!(
                !host.contains(char::is_whitespace),
                "[approvals] http_call_allowlist entry {entry:?} contains whitespace"
            );
            anyhow::ensure!(
                !host.contains("://"),
                "[approvals] http_call_allowlist entry {entry:?} is a bare host (no scheme)"
            );
        }
        for url in &self.notify_webhooks {
            let trimmed = url.trim();
            anyhow::ensure!(
                trimmed.starts_with("http://") || trimmed.starts_with("https://"),
                "[approvals] notify_webhooks url {url:?} is not an http(s) endpoint"
            );
            anyhow::ensure!(
                !trimmed.contains(char::is_whitespace),
                "[approvals] notify_webhooks url {url:?} contains whitespace"
            );
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Deferred actions
// ---------------------------------------------------------------------------

/// The action that stays parked until an approver answers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DeferredAction {
    /// A registry skill (non-owner-only gate — module trust note) plus
    /// its exact args, executed through the same dispatch contract the
    /// chat tool loop uses.
    Skill {
        gate: String,
        skill: String,
        #[serde(default = "default_args")]
        args: Value,
    },
    /// A raw HTTP call against an allowlisted host (payments, send-mail
    /// webhooks, control-plane triggers). https only.
    HttpCall {
        #[serde(default = "default_http_method")]
        method: String,
        url: String,
        #[serde(default)]
        body: Option<Value>,
    },
}

fn default_args() -> Value {
    json!({})
}
fn default_http_method() -> String {
    "POST".to_string()
}

impl DeferredAction {
    /// The one-line description notification bodies show the approver.
    /// Args are NEVER printed verbatim: they can carry credentials,
    /// recipients or bank details, and the notification channels are
    /// relays, not secret keepers.
    pub fn describe(&self) -> String {
        match self {
            DeferredAction::Skill { gate, skill, .. } => {
                format!("run skill `{skill}` (plugin {gate})")
            }
            DeferredAction::HttpCall { method, url, .. } => {
                format!("HTTP {method} {url}")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Requests
// ---------------------------------------------------------------------------

/// Lifecycle state of a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// Parked, waiting for an approver.
    Pending,
    /// Approver accepted; the deferred action is queued for execution.
    Approved,
    /// Approver accepted with patched args.
    Corrected,
    /// Approver or requester refused; nothing executed.
    Cancelled,
    /// TTL sweeper expired the request; nothing executed.
    Expired,
    /// Executed; `result` carries the outcome summary.
    Done,
    /// Execution failed after approval; `result` carries the error.
    Failed,
}

/// One persisted request — the on-disk checkpoint of the paused state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    pub id: String,
    pub created_ms: u64,
    pub expires_ms: u64,
    /// Discord user that triggered the skill (0 = unknown; the skill
    /// runner contract carries no caller identity).
    pub requester_id: u64,
    /// Model-authored, human-facing summary of what is being approved.
    pub summary: String,
    pub action: DeferredAction,
    pub status: Status,
    /// Outcome summary filled at execution (Done / Failed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    /// Patched action when the owner chose the Correct path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corrected_action: Option<DeferredAction>,
}

fn now_ms() -> u64 {
    jerith_mesh_core::bootstrap::now_ms()
}

fn store_dir() -> PathBuf {
    // Test seam first: tests point the store at a scratch dir without
    // touching process env (edition 2024 makes set_var unsafe).
    #[cfg(test)]
    {
        let test_dir = TEST_STORE.lock().expect("test store lock").clone();
        if let Some(dir) = test_dir {
            return dir;
        }
    }
    let home = std::env::var("JERITH_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".to_string()))
                .join(".jerith-mesh")
        });
    home.join("bot").join("approvals")
}

fn request_path(id: &str) -> PathBuf {
    store_dir().join(format!("{id}.json"))
}

fn load_request(id: &str) -> Result<Option<Request>> {
    let path = request_path(id);
    let raw = match std::fs::read_to_string(&path) {
        Ok(raw) => raw,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    serde_json::from_str(&raw)
        .map(Some)
        .with_context(|| format!("parsing {}", path.display()))
}

/// Atomic write with owner-only permissions — the checkpoint must
/// survive a restart as exactly the state that was waiting.
fn save_request(req: &Request) -> Result<()> {
    let dir = store_dir();
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let path = request_path(&req.id);
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(req)?)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))?;
    }
    std::fs::rename(&tmp, &path)?; // atomic on POSIX
    Ok(())
}

/// 8-char unguessable id: one ~4.95-bit glyph per random byte over a
/// base32-style alphabet (≈ 40 bits total; no I/l/O/0/1 — the code may be
/// typed into a `correct` follow-up on a phone keyboard).
fn new_id() -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHJKMNPQRSTUVWXYZ23456789"; // 31 glyphs
    let mut bytes = [0u8; 8];
    rand::thread_rng().fill_bytes(&mut bytes);
    bytes
        .iter()
        .map(|b| ALPHABET[(b % 31) as usize] as char)
        .collect()
}

/// Concurrent pending approvals cap (a chatty guest cannot flood the
/// owner's DM with pending cards).
const MAX_PENDING: usize = 10;

// ---------------------------------------------------------------------------
// Runtime singletons (config + discord http set at bot start)
// ---------------------------------------------------------------------------

static CONFIG: OnceLock<Arc<crate::config::BotConfig>> = OnceLock::new();
static DISCORD_HTTP: OnceLock<Arc<serenityall::Http>> = OnceLock::new();
/// Candidate self-report dedupe (id seen once per process).
static SEEN_CANDIDATES: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<i64>>> =
    std::sync::OnceLock::new();

fn seen_candidates() -> std::sync::MutexGuard<'static, std::collections::HashSet<i64>> {
    SEEN_CANDIDATES
        .get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

// Test-only overrides: config + store can be swapped per-test without the
// process-env escape hatch (edition 2024).
#[cfg(test)]
static TEST_CONFIG: std::sync::Mutex<Option<Arc<crate::config::BotConfig>>> =
    std::sync::Mutex::new(None);
#[cfg(test)]
static TEST_STORE: std::sync::Mutex<Option<PathBuf>> = std::sync::Mutex::new(None);

/// Store the bot config (called from main before the client starts).
pub fn init(cfg: Arc<crate::config::BotConfig>) {
    let _ = CONFIG.set(cfg);
}

/// Store the Discord http handle once the serenity client is built (the
/// framework's setup stage). Notifications before this degrade
/// gracefully to telegram/webhook only.
pub fn set_discord_http(http: Arc<serenityall::Http>) {
    let _ = DISCORD_HTTP.set(http);
}

fn config() -> Result<&'static Arc<crate::config::BotConfig>> {
    #[cfg(test)]
    {
        if let Some(arc) = TEST_CONFIG.lock().expect("test config lock").clone() {
            // Leak a stable reference so the &'static contract holds for
            // the length of the test process (small, bounded config).
            return Ok(Box::leak(Box::new(arc)));
        }
    }
    CONFIG
        .get()
        .ok_or_else(|| anyhow::anyhow!("approval bridge not initialised (no bot config)"))
}

// ---------------------------------------------------------------------------
// Request + resolve (the state-machine transitions)
// ---------------------------------------------------------------------------

/// Create a pending approval and fan out notifications. Returns the
/// payload the model-facing skill hands back to the tool loop.
pub async fn request(summary: &str, action: DeferredAction, requester_id: u64) -> Result<Value> {
    let cfg = config()?;
    let acfg = &cfg.approvals;
    anyhow::ensure!(
        acfg.enabled,
        "approval bridge is disabled ([approvals] enabled = false)"
    );
    anyhow::ensure!(
        !summary.trim().is_empty(),
        "approval.request: summary is required (the approver reads this)"
    );
    anyhow::ensure!(
        has_approver_channel(cfg),
        "no approver channel configured — set [owner] discord_id or the telegram pair so \
         approvals have somewhere to land"
    );

    // Down-scoped deferred skills only (module trust note): the request-
    // time checks mirror the plain-session registry contract exactly.
    match &action {
        DeferredAction::Skill { gate, skill, args } => {
            anyhow::ensure!(
                !gate.trim().is_empty(),
                "approval.request: skill gate required"
            );
            anyhow::ensure!(
                args.as_object().is_some(),
                "approval.request: skill args must be an object"
            );
            let reg = crate::plugins::SkillRegistry::standard(cfg.plugins.enabled.clone())
                .with_external(cfg.external_plugins.clone())
                .context("building the request-time registry")?;
            anyhow::ensure!(
                reg.enabled_skill_names().iter().any(|n| n == skill),
                "deferred skill {skill:?} is not dispatchable for a plain session — owner-only \
                 actions run live in owner-DMs and cannot be deferred through this bridge"
            );
        }
        DeferredAction::HttpCall { method, url, body } => {
            validate_http_call(method, url, body.as_ref())?;
        }
    }

    anyhow::ensure!(
        pending_count()? < MAX_PENDING,
        "approval queue is full ({MAX_PENDING} pending) — resolve or wait before requesting again"
    );

    let ttl_ms = acfg.ttl_minutes * 60_000;
    let mut id = new_id();
    while load_request(&id)?.is_some() {
        id = new_id(); // 40-bit space; collisions vanish and are retried
    }
    let req = Request {
        id: id.clone(),
        created_ms: now_ms(),
        expires_ms: now_ms() + ttl_ms,
        requester_id,
        summary: summary.trim().to_string(),
        action: action.clone(),
        status: Status::Pending,
        result: None,
        corrected_action: None,
    };
    save_request(&req)?;

    // Fan out (best-effort per channel; the request stays valid even if
    // one relay is down — hooks enforces its own bounded retries).
    let card = notification_card(&req);
    emit_hook("approval_requested", &req);
    if acfg.discord_owner_dm
        && let Err(e) = notify_discord(&card, &req.id).await
    {
        tracing::warn!(error = %e, id = %id, "discord approval DM failed");
    }
    if acfg.telegram_chat_id.is_some()
        && let Err(e) = notify_telegram(&card, &req.id).await
    {
        tracing::warn!(error = %e, id = %id, "telegram approval notice failed");
    }
    for url in &acfg.notify_webhooks {
        if let Err(e) = post_webhook(url, &json!({"text": card})).await {
            tracing::warn!(error = %e, url, "approval notify webhook failed");
        }
    }

    Ok(json!({
        "id": id,
        "state": "pending",
        "expires_in_minutes": acfg.ttl_minutes,
        "action": req.action.describe(),
        "note": "Paused before the critical step. The owner was notified with Approve/Deny \
                 controls; execution resumes automatically on approval. Track with approval.status."
    }))
}

fn has_approver_channel(cfg: &crate::config::BotConfig) -> bool {
    let a = &cfg.approvals;
    (a.discord_owner_dm && cfg.owner_discord_id.is_some())
        || (a.telegram_bot_token.is_some() && a.telegram_chat_id.is_some())
}

/// `approval.status` payload: one request's view (by id) or the pending list.
pub async fn status(id: Option<&str>) -> Result<Value> {
    if let Some(id) = id {
        let req = load_request(id)?.ok_or_else(|| anyhow::anyhow!("approval {id:?} not found"))?;
        return Ok(request_row(&req));
    }
    let dir = store_dir();
    let mut rows = Vec::new();
    for sid in list_ids(&dir)? {
        if let Some(req) = load_request(&sid)?
            && req.status == Status::Pending
        {
            rows.push(request_row(&req));
        }
    }
    Ok(json!({ "pending": rows, "count": rows.len() }))
}

fn request_row(req: &Request) -> Value {
    let mut row = json!({
        "id": req.id,
        "status": req.status,
        "action": req.action.describe(),
        "summary": req.summary,
        "expires_ms": req.expires_ms,
    });
    if let Some(res) = &req.result {
        row["result"] = json!(res);
    }
    row
}

/// The requester (or the model on the user's behalf) cancels a still-
/// pending request. Any resolved state is a refusal — only a Pending
/// request can be cancelled, and resolution changes never re-execute.
pub async fn cancel(id: &str) -> Result<Value> {
    let Some(mut req) = load_request(id)? else {
        anyhow::bail!("approval {id:?} not found");
    };
    anyhow::ensure!(
        req.status == Status::Pending,
        "approval {id:?} is already resolved ({:?}) and cannot be cancelled",
        req.status
    );
    req.status = Status::Cancelled;
    req.result = Some("cancelled by the requester".to_string());
    save_request(&req)?;
    emit_hook("approval_resolved", &req);
    Ok(json!({ "id": req.id, "state": "cancelled" }))
}

/// One resolution decision. `corrected` carries the approver's patched
/// args for the Correct path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// Execute the stored action verbatim.
    Approve,
    /// Execute with the patched args.
    Correct,
    /// Refuse; nothing executes.
    Deny,
}

/// Identity-checked, one-shot resolution. A SECOND call on the same id
/// (double click, Discord/Telegram race) is a stable no-op returning the
/// already-resolved state — never an error, never a second execution.
/// Resolver identity is the CALLER's contract (chat_fallback for Discord
/// buttons verifies the owner id; the telegram poller verifies the chat
/// id) and is re-checked below against the config for defence in depth.
pub async fn resolve(
    resolver: ResolverId,
    id: &str,
    decision: Decision,
    corrected_args: Option<Value>,
) -> Result<Value> {
    let cfg = config()?;
    anyhow::ensure!(
        resolver.matches(cfg),
        "resolver is not the configured approver (owner DM or telegram chat only)"
    );
    let Some(mut req) = load_request(id)? else {
        anyhow::bail!("approval {id:?} not found");
    };
    match req.status {
        Status::Pending => {}
        Status::Expired => {
            return Ok(json!({ "id": id, "state": "expired", "note": "already closed by TTL" }));
        }
        other => {
            return Ok(json!({ "id": id, "state": other, "note": "already resolved; no re-run" }));
        }
    }
    if now_ms() > req.expires_ms {
        req.status = Status::Expired;
        req.result = Some("expired before resolution".to_string());
        save_request(&req)?;
        emit_hook("approval_resolved", &req);
        return Ok(json!({ "id": id, "state": "expired", "note": "expired before resolution" }));
    }

    match decision {
        Decision::Approve => {
            req.status = Status::Approved;
        }
        Decision::Correct => {
            let patch = corrected_args.ok_or_else(|| {
                anyhow::anyhow!("a corrected resolution requires the corrected args")
            })?;
            let action = patch_action(&req.action, &patch)?;
            req.corrected_action = Some(action);
            req.status = Status::Corrected;
        }
        Decision::Deny => {
            req.status = Status::Cancelled;
            req.result = Some("denied by the approver".to_string());
            save_request(&req)?;
            emit_hook("approval_resolved", &req);
            notify_resolution(cfg, &req, "denied — nothing was executed").await;
            return Ok(json!({ "id": id, "state": "cancelled", "note": "nothing was executed" }));
        }
    }
    save_request(&req)?;
    emit_hook("approval_resolved", &req);

    // Execution never rides the resolver's response path: interactions
    // must ack fast, and a workflow replay can legitimately take minutes.
    tokio::spawn(execute(id.to_string()));
    Ok(json!({
        "id": id,
        "state": "approved",
        "note": "executing the deferred action; the result lands as a follow-up notification"
    }))
}

/// Who resolved (already verified at the channel, re-verified here).
#[derive(Debug, Clone, Copy)]
pub enum ResolverId {
    Discord(u64),
    Telegram(i64),
}

impl ResolverId {
    fn matches(&self, cfg: &crate::config::BotConfig) -> bool {
        match self {
            ResolverId::Discord(id) => *id != 0 && Some(*id) == cfg.owner_discord_id,
            ResolverId::Telegram(chat) => Some(*chat) == cfg.approvals.telegram_chat_id,
        }
    }
}

/// The Approved / Corrected → execution transition's second half (spawned
/// off the resolver; doubles as the sweeper's repair path when a restart
/// orphaned an Approved/Corrected request). Loads, executes, persists the
/// terminal status and notifies.
async fn execute(id: String) {
    let Some(cfg) = CONFIG.get().cloned() else {
        return;
    };
    let Ok(Some(req)) = load_request(&id) else {
        return;
    };
    if !matches!(req.status, Status::Approved | Status::Corrected) {
        return; // another path finished it first
    }
    let action = req.corrected_action.as_ref().unwrap_or(&req.action);
    let outcome = match action {
        DeferredAction::Skill {
            gate: _gate,
            skill,
            args,
        } => execute_skill(&cfg, skill, args).await,
        DeferredAction::HttpCall { method, url, body } => {
            execute_http_call(&cfg, method, url, body.as_ref()).await
        }
    };
    let mut req = req;
    match outcome {
        Ok(desc) => {
            req.status = Status::Done;
            req.result = Some(desc);
        }
        Err(e) => {
            req.status = Status::Failed;
            req.result = Some(first_line(&e.to_string()));
        }
    }
    if let Err(e) = save_request(&req) {
        tracing::error!(error = %e, id, "persisting executed approval");
    }
    notify_resolution(&cfg, &req, req.result.as_deref().unwrap_or("done")).await;
}

async fn execute_skill(
    cfg: &Arc<crate::config::BotConfig>,
    skill: &str,
    args: &Value,
) -> Result<String> {
    // Owner-privileged registry (the approver IS the owner) — the
    // request-time down-scope check is what keeps guests from ever
    // reaching owner-only skills here; privileged execution at APPROVE
    // time means skills a plain session's stricter tool budget skipped
    // still work once a human said yes.
    let reg = crate::plugins::SkillRegistry::standard(cfg.plugins.enabled.clone());
    let reg = reg
        .with_external(cfg.external_plugins.clone())
        .context("building the execute-time registry")?;
    reg.dispatch(cfg, skill, args).await?;
    Ok(format!("skill {skill:?} executed"))
}

/// Allowlist + https + method + size validation. Runs at BOTH deferral
/// time and the moment the trigger is actually pulled — config could
/// have tightened in between, and the allowlist is the security wall.
fn validate_http_call(method: &str, url: &str, body: Option<&Value>) -> Result<()> {
    let cfg = config()?;
    let allowlist = &cfg.approvals.http_call_allowlist;
    anyhow::ensure!(
        !allowlist.is_empty(),
        "http_call is not permitted — [approvals] http_call_allowlist is empty"
    );
    let method = method.trim().to_ascii_uppercase();
    anyhow::ensure!(
        matches!(method.as_str(), "GET" | "POST" | "PUT" | "PATCH"),
        "http_call method {method:?} not permitted (GET/POST/PUT/PATCH)"
    );
    let trimmed = url.trim();
    anyhow::ensure!(
        trimmed.starts_with("https://"),
        "http_call url must be https (got {trimmed:?})"
    );
    let host = trimmed
        .trim_start_matches("https://")
        .split(['/', '?', ':'])
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    anyhow::ensure!(!host.is_empty(), "http_call url has no parseable host");
    anyhow::ensure!(
        allowlist
            .iter()
            .any(|a| a.trim().to_ascii_lowercase() == host),
        "http_call host {host:?} is not on the [approvals] http_call_allowlist"
    );
    if let Some(b) = body {
        let bytes = serde_json::to_vec(b)?.len();
        anyhow::ensure!(bytes <= 64 * 1024, "http_call body exceeds 64 KiB");
    }
    Ok(())
}

async fn execute_http_call(
    _cfg: &Arc<crate::config::BotConfig>,
    method: &str,
    url: &str,
    body: Option<&Value>,
) -> Result<String> {
    validate_http_call(method, url, body)?;
    let client = crate::api::client();
    let verb = reqwest::Method::from_bytes(method.trim().to_ascii_uppercase().as_bytes())
        .context("http_call method")?;
    let mut builder = client.request(verb.clone(), url.trim());
    if let Some(b) = body {
        builder = builder.json(b);
    }
    let resp = builder
        .timeout(Duration::from_secs(30))
        .send()
        .await
        .context("http_call request")?;
    let status = resp.status();
    // Deliberately minimal return: a payment API response is neither for
    // guest chats nor unbounded.
    let head = resp.text().await.unwrap_or_default();
    let snippet: String = head.chars().take(200).collect();
    Ok(format!(
        "HTTP {verb} {url} — status {status}; body[0..200]: {snippet}"
    ))
}

/// The Correct-path patch: Skill args are MERGED (the owner fixes the one
/// field they spotted; the rest of the recorded call survives), HttpCall
/// bodies are REPLACED (the owner re-authorizes the exact payload).
/// An HttpCall's method/url stay as deferred — changing the target is a
/// NEW request, never a silent redirect on an approval card.
fn patch_action(action: &DeferredAction, patch: &Value) -> Result<DeferredAction> {
    match action {
        DeferredAction::Skill { gate, skill, args } => {
            anyhow::ensure!(
                patch.as_object().is_some(),
                "the correct patch must be an object of skill args"
            );
            let mut merged = match args {
                Value::Object(map) => map.clone(),
                _ => serde_json::Map::new(),
            };
            if let Some(obj) = patch.as_object() {
                for (k, v) in obj {
                    merged.insert(k.clone(), v.clone());
                }
            }
            Ok(DeferredAction::Skill {
                gate: gate.clone(),
                skill: skill.clone(),
                args: Value::Object(merged),
            })
        }
        DeferredAction::HttpCall { method, url, .. } => {
            anyhow::ensure!(
                patch.as_object().is_some(),
                "the correct patch must be an object (it replaces the http_call body)"
            );
            Ok(DeferredAction::HttpCall {
                method: method.clone(),
                url: url.clone(),
                body: Some(patch.clone()),
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Notifications
// ---------------------------------------------------------------------------

/// The approver-facing card text. Model arguments never ride the wire.
pub fn notification_card(req: &Request) -> String {
    let mins = req.expires_ms.saturating_sub(now_ms()) / 60_000;
    format!(
        "**Approval needed — {id}**\n{summary}\n\nWill: {action}\nOpen for: {mins} min — \
         approve to resume, deny to cancel.",
        id = req.id,
        summary = req.summary,
        action = req.action.describe(),
    )
}

async fn dm_owner(http: &serenityall::Http, owner: u64, card: &str) -> Result<()> {
    let channel = serenityall::UserId::new(owner)
        .create_dm_channel(http)
        .await
        .context("opening the owner approval DM")?;
    channel
        .id
        .send_message(http, serenityall::CreateMessage::new().content(card))
        .await
        .context("sending the approval message")?;
    Ok(())
}

async fn notify_discord(card: &str, id: &str) -> Result<()> {
    let http = DISCORD_HTTP
        .get()
        .ok_or_else(|| anyhow::anyhow!("discord unreachable (bot not started yet)"))?;
    let cfg = config()?;
    let owner = cfg
        .owner_discord_id
        .ok_or_else(|| anyhow::anyhow!("no owner id for the approval DM"))?;
    let channel = serenityall::UserId::new(owner)
        .create_dm_channel(http)
        .await
        .context("opening the owner approval DM")?;
    let row = serenityall::CreateActionRow::Buttons(vec![
        serenityall::CreateButton::new(format!("ap:{id}:yes"))
            .label("Approve ✓")
            .style(serenityall::ButtonStyle::Success),
        serenityall::CreateButton::new(format!("ap:{id}:no"))
            .label("Deny ✕")
            .style(serenityall::ButtonStyle::Danger),
    ]);
    channel
        .id
        .send_message(
            http,
            serenityall::CreateMessage::new()
                .content(card)
                .components(vec![row]),
        )
        .await
        .context("sending the approval card")?;
    Ok(())
}

/// Telegram: the notification with an inline keyboard; the poller
/// (`telegram_poll`) resolves callbacks from the configured chat.
async fn notify_telegram(card: &str, id: &str) -> Result<()> {
    let cfg = config()?;
    let token = cfg
        .approvals
        .telegram_bot_token
        .as_deref()
        .unwrap_or_default();
    let chat = cfg.approvals.telegram_chat_id.unwrap_or_default();
    let body = json!({
        "chat_id": chat,
        "text": card,
        "reply_markup": {
            "inline_keyboard": [[
                {"text": "✔ Approve", "callback_data": format!("ap:{id}:yes")},
                {"text": "✕ Deny", "callback_data": format!("ap:{id}:no")}
            ]]
        }
    });
    let resp = crate::api::client()
        .post(format!("https://api.telegram.org/bot{token}/sendMessage"))
        .json(&body)
        .timeout(Duration::from_secs(15))
        .send()
        .await
        .context("telegram sendMessage")?;
    let _ = resp.json::<Value>().await.context("telegram reply")?;
    Ok(())
}

async fn post_webhook(url: &str, payload: &Value) -> Result<()> {
    crate::api::client()
        .post(url.trim())
        .json(payload)
        .timeout(Duration::from_secs(10))
        .send()
        .await
        .context("notify webhook")?
        .error_for_status()
        .context("notify webhook rejected the card")?;
    Ok(())
}

/// Resolution result → every resolved-into channel (submitted as a
/// follow-up, never as an edit of a dead message identity).
async fn notify_resolution(cfg: &Arc<crate::config::BotConfig>, req: &Request, outcome: &str) {
    let text = format!("**Approval {id}** — {outcome}", id = req.id);
    emit_hook("approval_resolved", req);
    if cfg.approvals.discord_owner_dm
        && let Some(http) = DISCORD_HTTP.get()
        && let Some(owner) = cfg.owner_discord_id
    {
        let _ = dm_owner(http, owner, &text).await;
    }
    if let Some(token) = cfg.approvals.telegram_bot_token.as_deref()
        && let Some(chat) = cfg.approvals.telegram_chat_id
    {
        send_telegram_text(token, chat, &text).await;
    }
    for url in &cfg.approvals.notify_webhooks {
        let _ = post_webhook(url, &json!({"text": text})).await;
    }
}

// ---------------------------------------------------------------------------
// Inbound resolutions: Discord buttons (chat_fallback calls) + Telegram poller
// ---------------------------------------------------------------------------

/// Custom-id grammar of the approval buttons:
/// `ap:<id>:yes` | `ap:<id>:no`. Returns None for non-approval ids so
/// the event handler ignores everything else untouched.
pub fn parse_button(custom_id: &str) -> Option<(String, bool)> {
    let rest = custom_id.strip_prefix("ap:")?;
    let (id, side) = rest.rsplit_once(':')?;
    // Request ids are always exactly 8 glyphs (see new_id) — anything
    // else is not one of this bridge's cards.
    if id.len() != 8 || !id.bytes().all(|b| b.is_ascii_alphanumeric()) {
        return None;
    }
    match side {
        "yes" => Some((id.to_string(), true)),
        "no" => Some((id.to_string(), false)),
        _ => None,
    }
}

/// A Discord button press from the owner: resolve + respond with the
/// state (used by chat_fallback; the caller owns the interaction ack).
pub async fn resolve_discord(user_id: u64, custom_id: &str) -> Result<String> {
    let Some((id, yes)) = parse_button(custom_id) else {
        anyhow::bail!("not an approval button");
    };
    let corrected = None; // v1: patch path arrives via Telegram text / future UI
    let decision = if yes {
        Decision::Approve
    } else {
        Decision::Deny
    };
    let value = resolve(ResolverId::Discord(user_id), &id, decision, corrected).await?;
    Ok(state_line(&value))
}

fn state_line(value: &Value) -> String {
    let state = value.get("state").and_then(Value::as_str).unwrap_or("?");
    let note = value.get("note").and_then(Value::as_str).unwrap_or("");
    if note.is_empty() {
        format!("Approval {state}.")
    } else {
        format!("Approval {state} — {note}")
    }
}

/// Whether the external chat transport owns this same bot token
/// (FEATURES §18): two getUpdates consumers on one token 409-conflict
/// forever, so when `[transports.telegram].token` equals the approvals
/// token the transport is the ONLY poller — this poller exits and the
/// transport hands approval updates back over [`handle_transport_update`].
pub fn transport_owns_token(cfg: &crate::config::BotConfig) -> bool {
    let Some(t) = &cfg.transports.telegram else {
        return false;
    };
    cfg.approvals.telegram_bot_token.as_deref() == Some(t.token.as_str())
}

/// The transport's update stream: does this update belong to the approval
/// bridge (an inline-keyboard press on an approval card, or an
/// `approve/deny` command from the configured chat)? Mirrors exactly the
/// shapes [`handle_telegram_update`] acts on — a false here means the
/// transport relays the update as ordinary chat.
fn is_approval_event(cfg: &crate::config::BotConfig, update: &Value) -> bool {
    if let Some(cb) = update.get("callback_query") {
        return cb
            .get("data")
            .and_then(Value::as_str)
            .is_some_and(|d| parse_button(d).is_some());
    }
    let Some(chat) = cfg.approvals.telegram_chat_id else {
        return false;
    };
    let Some(msg) = update.get("message") else {
        return false;
    };
    if msg.pointer("/from/id").and_then(Value::as_i64) != Some(chat) {
        return false;
    }
    let text = msg
        .get("text")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim();
    let Some((word, tail)) = text.split_once(char::is_whitespace) else {
        return false;
    };
    let id = tail.split_whitespace().next().unwrap_or("");
    matches!(word.to_ascii_lowercase().as_str(), "approve" | "deny") && id.len() == 8
}

/// Hand-off entry for the transport lane (FEATURES §18): one shared
/// getUpdates consumer — the transport feeds every update through here;
/// approval events are resolved (and acked) exactly as the old standalone
/// poller did, everything else returns false for ordinary chat relay.
pub async fn handle_transport_update(cfg: &Arc<crate::config::BotConfig>, update: &Value) -> bool {
    if !is_approval_event(cfg, update) {
        return false;
    }
    handle_telegram_update(cfg, update).await;
    true
}

/// Telegram long-poll loop: resolves inline-keyboard callbacks and text
/// commands (`approve <id>` / `deny <id>` / `approve <id> <json>` for the
/// Correct path) from the configured chat only. Runs until the process
/// dies; a dead network is absorbed (poll retries at the loop tick).
/// If the chat transport owns this same token (see
/// [`transport_owns_token`]) the transport is the poller instead — the
/// standalone loop exits so the token never has two consumers.
pub fn start_telegram_poller() {
    tokio::spawn(async move {
        let mut offset: i64 = 0;
        loop {
            let Ok(cfg) = config().cloned() else {
                tokio::time::sleep(Duration::from_secs(30)).await;
                continue;
            };
            if transport_owns_token(&cfg) {
                tracing::info!(
                    "approvals: chat transport owns this bot token — riding its poll stream"
                );
                return;
            }
            let Some(token) = cfg.approvals.telegram_bot_token.clone() else {
                return; // telegram not configured — poller exits cleanly
            };
            let url = format!("https://api.telegram.org/bot{token}/getUpdates");
            let resp = crate::api::client()
                .post(&url)
                .json(&json!({
                    "offset": offset,
                    "timeout": 25,
                    "allowed_updates": ["message", "callback_query"]
                }))
                .timeout(Duration::from_secs(35))
                .send()
                .await;
            let Ok(resp) = resp else {
                tokio::time::sleep(Duration::from_secs(10)).await;
                continue;
            };
            let body: Value = match resp.json().await {
                Ok(b) => b,
                Err(_) => {
                    tokio::time::sleep(Duration::from_secs(10)).await;
                    continue;
                }
            };
            // Confirm each handled update so offsets advance even across
            // restarts (getUpdates semantics: max-handled + 1).
            let updates = body
                .get("result")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            for update in updates {
                offset = update.get("update_id").and_then(Value::as_i64).unwrap_or(0) + 1;
                handle_telegram_update(&cfg, &update).await;
            }
        }
    });
}

async fn handle_telegram_update(cfg: &Arc<crate::config::BotConfig>, update: &Value) {
    // Self-report: a token configured WITHOUT the chat id logs every
    // sender once — DM the bot once, copy your id from the journal into
    // bot.toml, restart. No secrets in the log line, just ids and names.
    let Some(chat) = cfg.approvals.telegram_chat_id else {
        let (Some(from), Some(name)) = (
            update.pointer("/from/id").and_then(Value::as_i64),
            update
                .pointer("/from/first_name")
                .and_then(Value::as_str)
                .map(str::to_string),
        ) else {
            return;
        };
        if seen_candidates().insert(from) {
            tracing::info!(
                candidate_chat_id = from,
                name,
                "telegram candidate approver seen — set [approvals] telegram_chat_id \
                 to this id and restart to arm the channel"
            );
        }
        return;
    };
    // Inline-keyboard press.
    if let Some(cb) = update.get("callback_query") {
        let from = cb.pointer("/from/id").and_then(Value::as_i64).unwrap_or(0);
        let data = cb.get("data").and_then(Value::as_str).unwrap_or_default();
        let cb_id = cb.get("id").and_then(Value::as_str).unwrap_or_default();
        let (Some((id, yes)), true) = (parse_button(data), from == chat) else {
            return;
        };
        let outcome = resolve(ResolverId::Telegram(chat), &id, decision_from(yes), None).await;
        let text = match outcome {
            Ok(v) => state_line(&v),
            Err(e) => first_line(&e.to_string()),
        };
        ack_callback(&cfg.approvals.telegram_bot_token, cb_id, &text).await;
        return;
    }
    // Text command: `approve <id> [json]` / `deny <id>` — the json patch
    // on `approve` selects the Correct path.
    let Some(msg) = update.get("message") else {
        return;
    };
    let from = msg.pointer("/from/id").and_then(Value::as_i64).unwrap_or(0);
    let text = msg
        .get("text")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim();
    if from != chat {
        return;
    }
    let Some((word, tail)) = text.split_once(char::is_whitespace) else {
        return;
    };
    let (id, patch_json) = match tail.split_once(char::is_whitespace) {
        Some((id, p)) => (id.trim(), p.trim()),
        None => (tail.trim(), ""),
    };
    if id.len() != 8 {
        return;
    }
    let cmd = word.to_ascii_lowercase();
    if cmd != "approve" && cmd != "deny" {
        return;
    }
    let outcome = if cmd == "deny" {
        resolve(ResolverId::Telegram(chat), id, Decision::Deny, None).await
    } else if patch_json.is_empty() {
        resolve(ResolverId::Telegram(chat), id, Decision::Approve, None).await
    } else {
        match serde_json::from_str::<Value>(patch_json) {
            Ok(patch) => {
                resolve(
                    ResolverId::Telegram(chat),
                    id,
                    Decision::Correct,
                    Some(patch),
                )
                .await
            }
            Err(e) => Err(anyhow::anyhow!("the corrected args are not JSON: {e}")),
        }
    };
    let reply = match outcome {
        Ok(v) => state_line(&v),
        Err(e) => first_line(&e.to_string()),
    };
    send_telegram_text(&token_of(cfg), chat, &reply).await;
}

fn decision_from(yes: bool) -> Decision {
    if yes {
        Decision::Approve
    } else {
        Decision::Deny
    }
}

fn token_of(cfg: &Arc<crate::config::BotConfig>) -> String {
    cfg.approvals.telegram_bot_token.clone().unwrap_or_default()
}

async fn ack_callback(token: &Option<String>, cb_id: &str, text: &str) {
    let Some(token) = token else { return };
    let _ = crate::api::client()
        .post(format!(
            "https://api.telegram.org/bot{token}/answerCallbackQuery"
        ))
        .json(&json!({ "callback_query_id": cb_id, "text": text, "show_alert": true }))
        .timeout(Duration::from_secs(10))
        .send()
        .await;
}

async fn send_telegram_text(token: &str, chat: i64, text: &str) {
    if token.is_empty() {
        return;
    }
    let _ = crate::api::client()
        .post(format!("https://api.telegram.org/bot{token}/sendMessage"))
        .json(&json!({ "chat_id": chat, "text": text }))
        .timeout(Duration::from_secs(15))
        .send()
        .await;
}

// ---------------------------------------------------------------------------
// Sweeper: TTL expiry + record pruning
// ---------------------------------------------------------------------------

/// Periodic sweeper: expires still-Pending requests past their TTL
/// (notifying the channels so a stale card never dangles), executes
/// Approved/Corrected orphans (a restart between resolve and spawn), and
/// prunes resolved records past the keep window.
pub fn start_sweeper() {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(60)).await;
            let Ok(cfg) = config().cloned() else { continue };
            let Ok(ids) = list_ids(&store_dir()) else {
                continue;
            };
            for id in ids {
                let Ok(req_opt) = load_request(&id) else {
                    continue;
                };
                let Some(req) = req_opt else { continue };
                match req.status {
                    Status::Pending if now_ms() > req.expires_ms => {
                        let mut expired = req.clone();
                        expired.status = Status::Expired;
                        expired.result = Some("expired before resolution".to_string());
                        if save_request(&expired).is_ok() {
                            notify_resolution(&cfg, &expired, "expired — nothing was executed")
                                .await;
                        }
                    }
                    Status::Approved | Status::Corrected => {
                        // Orphan of a resolve that lost its spawn (restart
                        // race): re-drive the transition instead of letting
                        // an approved action sleep forever.
                        tokio::spawn(execute(id.clone()));
                    }
                    _ => {}
                }
                // Record pruning (resolved files past the keep window).
                let keep_ms = cfg.approvals.prune_resolved_hours * 3_600_000;
                if req.status != Status::Pending
                    && now_ms() > req.created_ms + keep_ms.max(3_600_000)
                {
                    let _ = std::fs::remove_file(request_path(&id));
                }
            }
        }
    });
}

fn list_ids(dir: &Path) -> Result<Vec<String>> {
    let mut ids = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(ids),
        Err(e) => return Err(e).context("listing the approval store"),
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if let Some(stem) = name.strip_suffix(".json")
            && stem.len() == 8
            && stem.bytes().all(|b| b.is_ascii_alphanumeric())
        {
            ids.push(stem.to_string());
        }
    }
    ids.sort();
    Ok(ids)
}

/// First line of an error string, clipped — the shared card-text
/// discipline (chat_fallback reuses it via alias).
pub(crate) fn first_line(s: &str) -> String {
    s.lines()
        .next()
        .unwrap_or("error")
        .chars()
        .take(240)
        .collect()
}

fn pending_count() -> Result<usize> {
    let dir = store_dir();
    let mut n = 0usize;
    for id in list_ids(&dir)? {
        if let Some(req) = load_request(&id)?
            && req.status == Status::Pending
            && now_ms() <= req.expires_ms
        {
            n += 1;
        }
    }
    Ok(n)
}

fn emit_hook(event: &str, req: &Request) {
    let event_type = if event == "approval_resolved" {
        crate::hooks::EV_APPROVAL_RESOLVED
    } else {
        crate::hooks::EV_APPROVAL_REQUESTED
    };
    crate::hooks::global().emit(crate::hooks::Event::new(
        event_type,
        json!({ "id": req.id, "action": req.action.describe(), "status": req.status }),
    ));
}

// ---------------------------------------------------------------------------
// Model-facing skills (plugin `approvals`)
// ---------------------------------------------------------------------------

/// Tool surface: request / status / cancel. The model learns to carry
/// tasks forward and stop before privileged edges; the human stays the
/// only one who pulls the trigger side effects need.
pub fn skills() -> Vec<crate::plugins::SkillSpec> {
    use crate::plugins::SkillSpec;
    vec![
        SkillSpec {
            name: "approval.request",
            plugin: "approvals",
            description: "Pause before a critical side effect (payment, send, purchase, \
                          privileged action) and ask the owner to approve. Give a clear human \
                          summary of exactly what will happen. Returns a pending id; execution \
                          resumes automatically when the owner approves.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Human-readable summary of what is being approved and why"},
                    "kind": {"type": "string", "enum": ["skill", "http_call"], "description": "Deferred action type"},
                    "skill": {"type": "string", "description": "Skill name (kind=skill): e.g. workflow.run"},
                    "gate": {"type": "string", "description": "Owning plugin id of the skill (kind=skill)"},
                    "args": {"type": "object", "description": "Skill arguments (kind=skill)"},
                    "method": {"type": "string", "enum": ["GET", "POST", "PUT", "PATCH"], "description": "HTTP method (kind=http_call, default POST)"},
                    "url": {"type": "string", "description": "https URL on an allowlisted host (kind=http_call)"},
                    "body": {"type": "object", "description": "JSON request body (kind=http_call)"}
                },
                "required": ["summary", "kind"]
            }),
            run: approval_request_runner,
        },
        SkillSpec {
            name: "approval.status",
            plugin: "approvals",
            description: "Check the state of pending approvals (all, or by id).",
            parameters: json!({
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Specific approval id"}
                }
            }),
            run: approval_status_runner,
        },
        SkillSpec {
            name: "approval.cancel",
            plugin: "approvals",
            description: "Withdraw a still-pending approval request (the requester changes \
                          their mind before the owner answers).",
            parameters: json!({
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Approval id to withdraw"}
                },
                "required": ["id"]
            }),
            run: approval_cancel_runner,
        },
    ]
}

fn approval_request_runner<'args>(
    _cfg: &'args crate::config::BotConfig,
    args: &'args Value,
) -> BoxFut<'args, anyhow::Result<Value>> {
    Box::pin(async move {
        let summary = args
            .get("summary")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let kind = args.get("kind").and_then(Value::as_str).unwrap_or("skill");
        let action = match kind {
            "http_call" => DeferredAction::HttpCall {
                method: args
                    .get("method")
                    .and_then(Value::as_str)
                    .unwrap_or("POST")
                    .to_string(),
                url: args
                    .get("url")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                body: args.get("body").cloned(),
            },
            _ => DeferredAction::Skill {
                gate: args
                    .get("gate")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                skill: args
                    .get("skill")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                args: args.get("args").cloned().unwrap_or_else(|| json!({})),
            },
        };
        request(summary, action, 0).await
    })
}

fn approval_status_runner<'args>(
    _cfg: &'args crate::config::BotConfig,
    args: &'args Value,
) -> BoxFut<'args, anyhow::Result<Value>> {
    Box::pin(async move {
        let id = args.get("id").and_then(Value::as_str);
        status(id).await
    })
}

fn approval_cancel_runner<'args>(
    _cfg: &'args crate::config::BotConfig,
    args: &'args Value,
) -> BoxFut<'args, anyhow::Result<Value>> {
    Box::pin(async move {
        let id = args
            .get("id")
            .and_then(Value::as_str)
            .context("approval.cancel: id required")?;
        cancel(id).await
    })
}

// ---------------------------------------------------------------------------
// Tests (pure parts: store, validation, patching, card text, id grammar)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Per-test serialization: the store + config seams are process-global
    /// statics, so every touch holds one mutex for its whole body.
    static TEST_SEQUENCER: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Point the store at a fresh scratch dir (create it) and swap the
    /// config for an allowlisted one. The returned guard releases the
    /// sequencer when the test ends.
    fn with_test_setup(allowlist: &[&str]) -> std::sync::MutexGuard<'static, ()> {
        let guard = TEST_SEQUENCER.lock().unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir()
            .join(format!("jm-approvals-{}", std::process::id()))
            .join(new_id().to_ascii_lowercase());
        std::fs::create_dir_all(&dir).expect("scratch store dir");
        *TEST_STORE.lock().expect("test store lock") = Some(dir);
        let mut cfg =
            crate::config::BotConfig::from_parts("http://127.0.0.1:8900", "http://127.0.0.1:8322");
        // The bot deployment's own owner id (resolved from bot.toml/env).
        cfg.owner_discord_id = Some(
            std::env::var("JERITH_BOT_OWNER_ID")
                .ok()
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(424242),
        );
        cfg.approvals.http_call_allowlist = allowlist.iter().map(|s| s.to_string()).collect();
        *TEST_CONFIG.lock().expect("test config lock") = Some(Arc::new(cfg));
        guard
    }

    /// A non-owner-only skill the approval bridge may legally defer.
    fn embed_action() -> DeferredAction {
        DeferredAction::Skill {
            gate: "memory".into(),
            skill: "memory.embed".into(),
            args: json!({"text": "remember the antarctic meeting notes"}),
        }
    }

    #[tokio::test]
    // Single-threaded tokio test: the std-sequencer guard IS the
    // serialization contract and is deliberately held across the awaits.
    #[allow(clippy::await_holding_lock)]
    async fn request_refuses_owner_only_skills() {
        let _g = with_test_setup(&[]);
        // workflow.run belongs to the owner-only workflows plugin: its
        // name is invisible to a plain-session registry, so the gate
        // refuses the deferred execution instead of escalating later.
        let err = request("replay the purchase flow", skill_action(), 0).await;
        let msg = err.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(msg.contains("not dispatchable"), "got: {msg}");
    }

    #[tokio::test]
    // Single-threaded tokio test: the std-sequencer guard IS the
    // serialization contract and is deliberately held across the awaits.
    #[allow(clippy::await_holding_lock)]
    async fn deny_roundtrip_is_one_shot_and_identity_gated() {
        let _g = with_test_setup(&["api.stripe.com"]);
        let v = request("Remember one note", embed_action(), 0)
            .await
            .unwrap();
        let id = v["id"].as_str().unwrap().to_string();
        // A guest's resolver id is refused outright.
        assert!(
            resolve(ResolverId::Discord(999), &id, Decision::Approve, None)
                .await
                .is_err()
        );
        // Owner denies; nothing executes.
        let v2 = resolve(ResolverId::Discord(cfg_owner()), &id, Decision::Deny, None)
            .await
            .unwrap();
        assert_eq!(v2["state"], json!("cancelled"));
        // Second press on the same card is a stable no-op, not an error,
        // and never re-runs anything.
        let v3 = resolve(
            ResolverId::Discord(cfg_owner()),
            &id,
            Decision::Approve,
            None,
        )
        .await
        .unwrap();
        assert_eq!(v3["state"], json!("cancelled"));
        let v4 = resolve(ResolverId::Discord(cfg_owner()), &id, Decision::Deny, None)
            .await
            .unwrap();
        assert_eq!(v4["state"], json!("cancelled"));
        // The store shows Cancelled with a result.
        let row = status(Some(&id)).await.unwrap();
        assert_eq!(row["status"], json!("cancelled"));
        // Already-resolved requests refuse cancellation.
        assert!(cancel(&id).await.is_err());
        std::fs::remove_dir_all(store_dir()).ok();
    }

    #[tokio::test]
    // Single-threaded tokio test: the std-sequencer guard IS the
    // serialization contract and is deliberately held across the awaits.
    #[allow(clippy::await_holding_lock)]
    async fn correct_path_persists_patched_action() {
        let _g = with_test_setup(&["api.stripe.com"]);
        let v = request("Remember one note", embed_action(), 0)
            .await
            .unwrap();
        let id = v["id"].as_str().unwrap().to_string();
        // Correct without args is an invalid decision.
        assert!(
            resolve(
                ResolverId::Discord(cfg_owner()),
                &id,
                Decision::Correct,
                None
            )
            .await
            .is_err()
        );
        // With args it lands Corrected + the patch merged.
        let v2 = resolve(
            ResolverId::Discord(cfg_owner()),
            &id,
            Decision::Correct,
            Some(json!({"text": "remember the FIXED note"})),
        )
        .await
        .unwrap();
        assert_eq!(v2["state"], json!("approved"));
        let req = load_request(&id).unwrap().unwrap();
        assert_eq!(req.status, Status::Corrected);
        let corrected = req.corrected_action.unwrap();
        match corrected {
            DeferredAction::Skill { args, skill, .. } => {
                assert_eq!(skill, "memory.embed");
                assert_eq!(args["text"], json!("remember the FIXED note"));
            }
            _ => panic!("wrong kind"),
        }
        // Cleanup: the spawned executor (memory.embed → gateway) will
        // fail loudly offline and drain into Failed, never executing
        // anything — and the sequencer guard keeps other tests clear.
        let _ = std::fs::remove_dir_all(store_dir());
    }

    fn cfg_owner() -> u64 {
        config().unwrap().owner_discord_id.unwrap_or(424242)
    }

    #[tokio::test]
    // Single-threaded tokio test: the std-sequencer guard IS the
    // serialization contract and is deliberately held across the awaits.
    #[allow(clippy::await_holding_lock)]
    async fn ttl_expiry_closes_pending() {
        let _g = with_test_setup(&[]);
        let v = request("Remember one note", embed_action(), 0)
            .await
            .unwrap();
        let id = v["id"].as_str().unwrap().to_string();
        let mut req = load_request(&id).unwrap().unwrap();
        req.expires_ms = now_ms() - 1; // force the expiry window open
        save_request(&req).unwrap();
        let v2 = resolve(
            ResolverId::Discord(cfg_owner()),
            &id,
            Decision::Approve,
            None,
        )
        .await
        .unwrap();
        assert_eq!(v2["state"], json!("expired"));
        let row = status(Some(&id)).await.unwrap();
        assert_eq!(row["status"], json!("expired"));
        std::fs::remove_dir_all(store_dir()).ok();
    }

    #[tokio::test]
    // Single-threaded tokio test: the std-sequencer guard IS the
    // serialization contract and is deliberately held across the awaits.
    #[allow(clippy::await_holding_lock)]
    async fn cancel_before_resolution_works() {
        let _g = with_test_setup(&[]);
        let v = request("Remember one note", embed_action(), 0)
            .await
            .unwrap();
        let id = v["id"].as_str().unwrap().to_string();
        let v2 = cancel(&id).await.unwrap();
        assert_eq!(v2["state"], json!("cancelled"));
        // The requester-cancelled request can no longer be approved.
        let v3 = resolve(
            ResolverId::Discord(cfg_owner()),
            &id,
            Decision::Approve,
            None,
        )
        .await
        .unwrap();
        assert_eq!(v3["state"], json!("cancelled"));
        std::fs::remove_dir_all(store_dir()).ok();
    }

    // The card never carries model args or http bodies.
    #[test]
    fn notification_card_format() {
        let req = Request {
            id: "K7M2XQ4P".into(),
            created_ms: now_ms(),
            expires_ms: now_ms() + 30 * 60_000,
            requester_id: 0,
            summary: "Buy the office outfit".into(),
            action: DeferredAction::HttpCall {
                method: "POST".into(),
                url: "https://api.stripe.com/v1/charges".into(),
                body: Some(json!({"card_number": "4242"})),
            },
            status: Status::Pending,
            result: None,
            corrected_action: None,
        };
        let card = notification_card(&req);
        assert!(card.contains("K7M2XQ4P"));
        assert!(card.contains("Buy the office outfit"));
        assert!(card.contains("api.stripe.com"));
        assert!(!card.contains("4242"), "card leaked the http body");
    }

    fn skill_action() -> DeferredAction {
        DeferredAction::Skill {
            gate: "workflows".into(),
            skill: "workflow.run".into(),
            args: json!({"name": "buy Staples gift card", "overwrite": false}),
        }
    }

    #[test]
    fn id_shape_is_unguessable_vocabulary() {
        for _ in 0..64 {
            let id = new_id();
            assert_eq!(id.len(), 8);
            assert!(
                id.bytes()
                    .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
            );
            assert!(
                !id.contains('I') && !id.contains('O') && !id.contains('0') && !id.contains('1')
            );
        }
    }

    #[test]
    fn deferred_action_serde_roundtrip() {
        let a = skill_action();
        let s = serde_json::to_string(&a).unwrap();
        assert_eq!(serde_json::from_str::<DeferredAction>(&s).unwrap(), a);
        let h = DeferredAction::HttpCall {
            method: "POST".into(),
            url: "https://api.stripe.com/v1/charges".into(),
            body: Some(json!({"amount": 2000})),
        };
        let s2 = serde_json::to_string(&h).unwrap();
        assert_eq!(serde_json::from_str::<DeferredAction>(&s2).unwrap(), h);
    }

    #[test]
    fn describe_never_reveals_args() {
        let card = skill_action().describe();
        assert!(card.contains("workflow.run"), "{card}");
        assert!(
            !card.contains("Staples"),
            "args leaked into describe: {card}"
        );
        let h = DeferredAction::HttpCall {
            method: "POST".into(),
            url: "https://api.stripe.com/v1/charges".into(),
            body: Some(json!({"card_number": "4242"})),
        };
        let desc = h.describe();
        assert!(!desc.contains("4242"), "body leaked into describe: {desc}");
    }

    #[test]
    fn patch_merges_skill_args_and_replaces_http_body() {
        let patched = patch_action(&skill_action(), &json!({"name": "buy mug"})).unwrap();
        match patched {
            DeferredAction::Skill { args, .. } => {
                assert_eq!(args["name"], json!("buy mug"));
                // Unmentioned fields survive the patch.
                assert_eq!(args["overwrite"], json!(false));
            }
            _ => panic!("wrong kind"),
        }
        let http = DeferredAction::HttpCall {
            method: "POST".into(),
            url: "https://api.stripe.com/v1/charges".into(),
            body: Some(json!({"amount": 2000})),
        };
        let patched2 = patch_action(&http, &json!({"amount": 999})).unwrap();
        match patched2 {
            DeferredAction::HttpCall { body, url, .. } => {
                assert_eq!(url, "https://api.stripe.com/v1/charges");
                assert_eq!(body.unwrap(), json!({"amount": 999}));
            }
            _ => panic!("wrong kind"),
        }
        // Patching a non-object is a typed refusal.
        assert!(patch_action(&skill_action(), &json!("nope")).is_err());
    }

    #[test]
    fn button_grammar_parses_only_approval_ids() {
        assert_eq!(
            parse_button("ap:K7M2XQ4P:yes"),
            Some(("K7M2XQ4P".to_string(), true))
        );
        assert_eq!(
            parse_button("ap:K7M2XQ4P:no"),
            Some(("K7M2XQ4P".into(), false))
        );
        assert_eq!(parse_button("ap::yes"), None);
        assert_eq!(parse_button("ap:TOOLONGID123456:yes"), None);
        assert_eq!(parse_button(":K7M2XQ4P:yes"), None);
        assert_eq!(parse_button("ap:K7M2XQ4P:maybe"), None);
        // A bot's other buttons (emoji custom ids etc.) never collide.
        assert_eq!(parse_button("role:admin"), None);
    }

    #[test]
    fn allowlist_gates_http_calls() {
        let _g = with_test_setup(&["api.stripe.com"]);
        // https + allowlisted host + body under cap: ok.
        validate_http_call(
            "POST",
            "https://api.stripe.com/v1/charges",
            Some(&json!({"a":1})),
        )
        .expect("allowlisted call");
        // http:// refused.
        assert!(validate_http_call("POST", "http://api.stripe.com/v1/charges", None).is_err());
        // Non-allowlisted host refused.
        assert!(
            validate_http_call("POST", "https://evil.example.net/pay", None).is_err(),
            "unlisted host must refuse"
        );
        // Disallowed method refused.
        assert!(validate_http_call("DELETE", "https://api.stripe.com/v1/charges", None).is_err());
        // Oversized body refused.
        let big = json!({ "blob": "x".repeat(70_000) });
        assert!(
            validate_http_call("POST", "https://api.stripe.com/v1/charges", Some(&big)).is_err()
        );
        // Host with port / path still matches the bare-host entry.
        validate_http_call("PUT", "https://api.stripe.com:443/v1/a?b=c", None).expect("host match");
    }

    /// LIVE Telegram round-trip — maintenance-only, #[ignore] because it
    /// talks to the real deployment: loads ~/.jerith-mesh/bot.toml,
    /// persists one real pending approval into the REAL store dir, and
    /// fans the card out to the real configured Telegram chat. The
    /// RUNNING bot's poller answers the ✔ press (resolution happens in
    /// its own process); this harness only prints the id — watch the
    /// journal and the .json file for the executed side. The store stays
    /// at its default: TEST_STORE must stay None here, a redirected one
    /// would break the hand-off to the running bot.
    #[tokio::test]
    #[ignore = "fires a real approval card into the live Telegram chat"]
    // Single-threaded tokio test: the std-sequencer guard IS the
    // serialization contract and is deliberately held across the awaits.
    #[allow(clippy::await_holding_lock)]
    async fn live_telegram_roundtrip() {
        let _g = TEST_SEQUENCER.lock().unwrap_or_else(|e| e.into_inner());
        // JERITH_DISCORD_TOKEN comes bound from the maintenance command
        // line (BotConfig::load requires it; this harness never connects).
        let cfg = crate::config::BotConfig::load()
            .context("loading the real bot.toml")
            .unwrap();
        assert!(
            cfg.approvals.telegram_bot_token.is_some() && cfg.approvals.telegram_chat_id.is_some(),
            "live roundtrip needs the [approvals] telegram pair in the real bot.toml"
        );
        *TEST_CONFIG.lock().expect("test config lock") = Some(Arc::new(cfg.clone()));
        // Zero-side-effect deferred action (private memory note); the
        // resolution notification is the visible end of the round trip.
        let action = DeferredAction::Skill {
            gate: "memory".into(),
            skill: "memory.embed".into(),
            args: json!({"text": "[bridge self-test] Telegram approve-callback round-trip — fired by the approval.rs live harness, safe to forget."}),
        };
        let out = request(
            "Harness round-trip: press ✔ Approve to prove the Telegram callback lane executes this deferred memory.embed.",
            action,
            cfg.owner_discord_id.unwrap_or(0),
        )
        .await
        .expect("approval.request fired against the live config");
        println!(
            "approval id = {}  →  press ✔ Approve in Telegram now",
            out["id"]
        );
    }

    #[test]
    fn transport_token_ownership_gates_the_standalone_poller() {
        let mut cfg =
            crate::config::BotConfig::from_parts("http://127.0.0.1:1", "http://127.0.0.1:1");
        // No transport configured: the approvals lane owns its token alone.
        cfg.approvals.telegram_bot_token = Some("t".into());
        assert!(!transport_owns_token(&cfg));
        // Transport on a DIFFERENT token: both bots can poll independently.
        cfg.transports.telegram = Some(crate::transports::telegram::TelegramConfig {
            api_url: "https://api.telegram.org".into(),
            token: "other".into(),
            bot_username: "b".into(),
            allowed_chats: vec![],
            respond_to: "mention".into(),
            poll_timeout_secs: 25,
        });
        assert!(!transport_owns_token(&cfg));
        // Same token: the transport is the single getUpdates consumer.
        cfg.transports.telegram.as_mut().unwrap().token = "t".into();
        assert!(transport_owns_token(&cfg));
    }

    #[test]
    fn transport_handoff_consumes_only_approval_shapes() {
        let mut cfg =
            crate::config::BotConfig::from_parts("http://127.0.0.1:1", "http://127.0.0.1:1");
        cfg.approvals.telegram_chat_id = Some(7);

        // Inline-button press on an approval card is consumed.
        assert!(is_approval_event(
            &cfg,
            &json!({"callback_query": {"data": "ap:ABC12345:yes"}})
        ));
        // Unrelated callback data must reach the chat lane.
        assert!(!is_approval_event(
            &cfg,
            &json!({"callback_query": {"data": "something:else"}})
        ));

        // approve/deny text from the configured chat is consumed.
        assert!(is_approval_event(
            &cfg,
            &json!({"message": {"from": {"id": 7}, "text": "approve ABC12345"}})
        ));
        assert!(is_approval_event(
            &cfg,
            &json!({"message": {"from": {"id": 7}, "text": "deny ABC12345"}})
        ));
        // Wrong chat, short id, ordinary chatter — all relay as chat.
        assert!(!is_approval_event(
            &cfg,
            &json!({"message": {"from": {"id": 9}, "text": "approve ABC12345"}})
        ));
        assert!(!is_approval_event(
            &cfg,
            &json!({"message": {"from": {"id": 7}, "text": "approve short"}})
        ));
        assert!(!is_approval_event(
            &cfg,
            &json!({"message": {"from": {"id": 7}, "text": "hello mesh"}})
        ));
    }

    #[test]
    fn config_validation_fails_loud() {
        let mut a = ApprovalConfig::default();
        assert!(a.validate().is_ok());
        // Telegram half-configured.
        a.telegram_bot_token = Some("t".into());
        assert!(a.validate().is_err());
        a.telegram_chat_id = Some(7);
        assert!(a.validate().is_ok());
        // Allowlist with a scheme.
        a.http_call_allowlist = vec!["https://api.stripe.com".into()];
        assert!(a.validate().is_err());
        a.http_call_allowlist = vec!["api.stripe.com".into()];
        assert!(a.validate().is_ok());
        // Notify webhooks must be http(s).
        a.notify_webhooks = vec!["ftp://relay".into()];
        assert!(a.validate().is_err());
    }

    #[test]
    fn store_roundtrip_is_exact_and_0600() {
        let _g = with_test_setup(&[]);
        let req = Request {
            id: new_id(),
            created_ms: now_ms(),
            expires_ms: now_ms() + 60_000,
            requester_id: 0,
            summary: "Buy the warehouse outfit".into(),
            action: skill_action(),
            status: Status::Pending,
            result: None,
            corrected_action: None,
        };
        save_request(&req).unwrap();
        let back = load_request(&req.id).unwrap().unwrap();
        assert_eq!(back.summary, req.summary);
        assert_eq!(back.action, req.action);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(request_path(&req.id))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(
                mode & 0o777,
                0o600,
                "approvals carry secrets, stay owner-only"
            );
        }
    }
}
