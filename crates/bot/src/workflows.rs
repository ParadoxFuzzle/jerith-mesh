//! Workflow recording — show the bot a sequence of clicks ONCE and it
//! saves the routing as a replayable playbook. Plugin id `workflows`,
//! owner-DM-only (OWNER_ONLY_PLUGINS + `[plugins] owner_only`).
//!
//! A playbook is a named list of browser steps over one site:
//!
//! ```json
//! { "name": "acme-invoices", "url": "https://acme.example/login",
//!   "steps": [
//!     {"t":"type","selector":"input#email","text":"ops@acme.example"},
//!     {"t":"click","selector":"#login","label":"Log in"},
//!     {"t":"goto","url":"https://acme.example/invoice/new"} ] }
//! ```
//!
//! **Record** (owner-driven, not screen casting): the bot launches the
//! real Chrome in a window on the configured display, holding a DevTools
//! (CDP) automation session with an injected recorder script. The owner
//! does the task themselves — clicks, filled fields, Enter presses and
//! page changes come back as structured `jerith-rec::` events over the
//! CDP console channel. `workflow.rec_stop` coalesces them into steps
//! and persists the playbook (0600 JSON under ~/.jerith-mesh/workflows/).
//!
//! **Replay** runs the same engine against a HEADLESS chrome with a
//! throwaway profile: fields fill via the native value setter + input
//! event (framework-safe), clicks dispatch real mouse-press/release at
//! the element's center through the CDP Input domain (OS-level
//! activation), recorded `goto`s heal mid-flow page changes. Bounded:
//! 200 steps, 1k-char selectors, per-run wall clock, HTTP(S) nav only,
//! argv-only chrome spawn on an ephemeral loopback port.
//!
//! Config ([workflows] in bot.toml): `chrome_binary` ("google-chrome"),
//! `recorder_display` (":0"), `run_budget_secs` (120).

use crate::config::BotConfig;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use anyhow::Context as _;

// ---------------------------------------------------------------------------
// Model + store
// ---------------------------------------------------------------------------

pub const MAX_STEPS: usize = 200;
pub const SELECTOR_CAP: usize = 1_000;
pub const TEXT_CAP: usize = 4_000;
pub const LABEL_CAP: usize = 200;

/// One playbook step.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "t", rename_all = "snake_case")]
pub enum Step {
    Goto {
        url: String,
    },
    /// Activate an element; `label` is captured text (humans read it; the
    /// model sees it when a selector rots, then re-records).
    Click {
        selector: String,
        #[serde(default)]
        label: String,
    },
    Type {
        selector: String,
        text: String,
    },
    Press {
        key: String,
    },
    Wait {
        ms: u64,
    },
}

impl Step {
    fn validate(&self) -> anyhow::Result<()> {
        match self {
            Step::Goto { url } => ensure_url(url),
            Step::Click { selector, label } => {
                ensure_selector(selector)?;
                anyhow::ensure!(
                    label.chars().count() <= LABEL_CAP,
                    "label exceeds {LABEL_CAP} chars"
                );
                Ok(())
            }
            Step::Type { selector, text } => {
                ensure_selector(selector)?;
                anyhow::ensure!(
                    text.chars().count() <= TEXT_CAP,
                    "typed text exceeds {TEXT_CAP} chars"
                );
                Ok(())
            }
            Step::Press { key } => {
                anyhow::ensure!(
                    matches!(
                        key.as_str(),
                        "Enter" | "Tab" | "Escape" | "ArrowDown" | "ArrowUp"
                    ),
                    "unsupported key {key:?} (recorded set: Enter/Tab/Escape/ArrowDown/ArrowUp)"
                );
                Ok(())
            }
            Step::Wait { ms } => {
                anyhow::ensure!(*ms <= 10_000, "dwell exceeds 10s");
                Ok(())
            }
        }
    }
}

fn ensure_selector(sel: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !sel.trim().is_empty() && sel.chars().count() <= SELECTOR_CAP,
        "selector must be 1..={SELECTOR_CAP} chars"
    );
    anyhow::ensure!(
        !sel.chars().any(|c| c == '\n' || c == '\0'),
        "selector carries control characters"
    );
    Ok(())
}

fn ensure_url(url: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        url.starts_with("http://") || url.starts_with("https://"),
        "url must be http(s)"
    );
    anyhow::ensure!(
        url.chars().count() <= 2_048 && !url.contains(char::is_whitespace),
        "url malformed (whitespace or >2k chars)"
    );
    Ok(())
}

/// One saved playbook (0600 JSON — recorded flows can carry logins).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workflow {
    pub name: String,
    pub url: String,
    pub steps: Vec<Step>,
    pub created_ms: u64,
    #[serde(default)]
    pub last_run_outcome: Option<String>,
}

fn valid_workflow_name(name: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !name.is_empty() && name.chars().count() <= 64,
        "workflow name must be 1..=64 chars"
    );
    anyhow::ensure!(
        name.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')),
        "workflow name must be [A-Za-z0-9._-]"
    );
    Ok(())
}

pub fn workflows_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".jerith-mesh").join("workflows")
}

fn workflow_path(name: &str) -> anyhow::Result<PathBuf> {
    valid_workflow_name(name)?;
    Ok(workflows_dir().join(name).with_extension("json"))
}

fn save(w: &Workflow) -> anyhow::Result<PathBuf> {
    let path = workflow_path(&w.name)?;
    let dir = workflows_dir();
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let raw = serde_json::to_string_pretty(w)?;
    std::fs::write(&path, raw.as_bytes()).with_context(|| format!("writing {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
    Ok(path)
}

fn load(name: &str) -> anyhow::Result<Workflow> {
    let path = workflow_path(name)?;
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("workflow {name:?} not found ({})", path.display()))?;
    let w: Workflow = serde_json::from_str(&raw).with_context(|| format!("parsing {name}"))?;
    validate_workflow(&w)?;
    Ok(w)
}

fn validate_workflow(w: &Workflow) -> anyhow::Result<()> {
    valid_workflow_name(&w.name)?;
    ensure_url(&w.url)?;
    anyhow::ensure!(
        w.steps.len() <= MAX_STEPS,
        "workflow carries {} steps (cap {MAX_STEPS})",
        w.steps.len()
    );
    for (i, s) in w.steps.iter().enumerate() {
        s.validate()
            .map_err(|e| anyhow::anyhow!("step {}: {e}", i + 1))?;
    }
    Ok(())
}

fn remove(name: &str) -> anyhow::Result<()> {
    let path = workflow_path(name).with_context(|| format!("invalid workflow name {name:?}"))?;
    std::fs::remove_file(&path).with_context(|| format!("removing {name}"))
}

fn list_names() -> anyhow::Result<Vec<String>> {
    let dir = workflows_dir();
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&dir).with_context(|| format!("reading {}", dir.display()))? {
        let p = entry?.path();
        if p.extension().is_some_and(|e| e == "json")
            && let Some(stem) = p.file_stem().and_then(|s| s.to_str())
        {
            out.push(stem.to_string());
        }
    }
    out.sort();
    Ok(out)
}

// ---------------------------------------------------------------------------
// Recorder events → steps
// ---------------------------------------------------------------------------

/// One raw recorder event (before coalescing).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Recorded {
    pub kind: String, // click | input | nav | press
    pub selector: String,
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub text: String,
}

/// Coalesce the recorder stream into playbook steps. The recorder fires
/// an input event PER KEYSTROKE — consecutive inputs on the SAME field
/// collapse into one `type` step carrying the LAST value. Navs on the
/// flow's start page are dropped (the start goto is implicit).
pub fn coalesce(events: Vec<Recorded>, start_url: &str) -> Vec<Step> {
    let mut steps: Vec<Step> = Vec::new();
    for ev in events {
        match ev.kind.as_str() {
            "click" => steps.push(Step::Click {
                selector: ev.selector,
                label: ev.label.chars().take(LABEL_CAP).collect(),
            }),
            "input" => {
                if let Some(Step::Type { selector, .. }) = steps.last()
                    && *selector == ev.selector
                {
                    steps.pop();
                }
                steps.push(Step::Type {
                    selector: ev.selector,
                    text: ev.text.chars().take(TEXT_CAP).collect(),
                });
            }
            "press" if ev.text == "Enter" || ev.text.is_empty() => {
                steps.push(Step::Press {
                    key: "Enter".into(),
                });
            }
            "press" => steps.push(Step::Press { key: ev.text }),
            "nav" => {
                if ev.selector == start_url {
                    continue;
                }
                if let Some(Step::Goto { url }) = steps.last()
                    && *url == ev.selector
                {
                    continue;
                }
                steps.push(Step::Goto { url: ev.selector });
            }
            _ => {}
        }
    }
    steps.into_iter().take(MAX_STEPS).collect()
}

// ---------------------------------------------------------------------------
// Browser driver trait — the flow engine against the browser seam
// ---------------------------------------------------------------------------

pub type BoxFut<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// One live browser session the engine drives. Two impls: the real
/// Chrome (CDP, below) and the tests' scripted mock — the engine never
/// reaches for chrome internals.
pub trait BrowserDriver: Send {
    fn goto<'a>(&'a mut self, url: &'a str) -> BoxFut<'a, anyhow::Result<()>>;
    fn click_on<'a>(&'a mut self, selector: &'a str) -> BoxFut<'a, anyhow::Result<()>>;
    fn type_text<'a>(
        &'a mut self,
        selector: &'a str,
        text: &'a str,
    ) -> BoxFut<'a, anyhow::Result<()>>;
    fn press_key<'a>(&'a mut self, key: &'a str) -> BoxFut<'a, anyhow::Result<()>>;
    fn drain_recorded(&mut self) -> BoxFut<'_, anyhow::Result<Vec<Recorded>>>;
}

/// Replay the playbook against a driver, step by step. Typed per-step
/// errors: the model learns WHICH step and WHAT broke without
/// re-deriving anything.
async fn replay(w: &Workflow, driver: &mut dyn BrowserDriver, budget: Duration) -> Vec<String> {
    let started = tokio::time::Instant::now();
    let mut out = Vec::new();
    for (i, step) in w.steps.iter().enumerate() {
        if started.elapsed() > budget {
            out.push(format!(
                "✗ step {}: wall-clock budget hit after {i} steps",
                i + 1
            ));
            break;
        }
        let res = match step {
            Step::Goto { url } => driver.goto(url).await,
            Step::Click { selector, .. } => driver.click_on(selector).await,
            Step::Type { selector, text } => driver.type_text(selector, text).await,
            Step::Press { key } => driver.press_key(key).await,
            Step::Wait { ms } => {
                tokio::time::sleep(Duration::from_millis((*ms).min(10_000))).await;
                Ok(())
            }
        };
        match res {
            Ok(()) => out.push(format!("✓ step {}", i + 1)),
            Err(e) => {
                out.push(format!("✗ step {} ({step:?}): {e:#}", i + 1));
                break;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// The injected recorder script
// ---------------------------------------------------------------------------

/// Stamps clicks / field fills / Enter presses into
/// `console.log("jerith-rec::" + JSON)` events the CDP session relays.
/// Selectors prefer #id, then [name=…], else nth-of-type CSS paths; the
/// label captures visible text/value.
const RECORDER_JS: &str = r#"(() => {
  if (window.__jerithRec) return 'already';
  window.__jerithRec = true;
  const send = (kind, p) => {
    try {
      console.log('jerith-rec::' + JSON.stringify({
        kind: kind,
        selector: p.selector || '',
        label: (p.label || '').slice(0, 200),
        text: p.text == null ? '' : p.text,
      }));
    } catch (e) {}
  };
  const build = (el) => {
    try {
      if (!el || el.nodeType !== 1) return '';
      if (el.id) return '#' + CSS.escape(el.id);
      if (el.name) return el.tagName.toLowerCase() + '[name="' + el.name + '"]';
      let path = '';
      let node = el;
      while (node && node.nodeType === 1 && path.length < 400) {
        if (node.id) return '#' + CSS.escape(node.id) + ' > ' + path;
        let part = node.tagName.toLowerCase();
        const parent = node.parentElement;
        if (parent) {
          const sibs = Array.from(parent.children).filter(c => c.tagName === node.tagName);
          if (sibs.length > 1) part += ':nth-of-type(' + (sibs.indexOf(node) + 1) + ')';
        }
        path = path.length ? (part + ' > ' + path) : part;
        node = parent;
      }
      return path;
    } catch (e) { return ''; }
  };
  document.addEventListener('click', (e) => {
    const el = (e.target.closest && e.target.closest('a,button,input,[role=button],label,textarea,select')) || e.target;
    send('click', {selector: build(el), label: (el.innerText || el.value || '').trim()});
  }, true);
  document.addEventListener('input', (e) => {
    const el = e.target;
    if (!(el instanceof HTMLInputElement) && !(el instanceof HTMLTextAreaElement)) return;
    send('input', {selector: build(el), text: String(el.value ?? '').slice(0, 4000)});
  }, true);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') send('press', {text: 'Enter'});
  }, true);
  return 'ok';
})()"#;

// ---------------------------------------------------------------------------
// Chrome CDP driver
// ---------------------------------------------------------------------------

/// The live chrome child + its per-target CDP websocket. One connection
/// serves record (recorder script console events) and headless replay.
struct CdpSession {
    next_id: u64,
    /// Response id → the oneshot waiting for its reply.
    pending: PendingMap,
    /// Recorder events received so far (record sessions only).
    events: Arc<Mutex<Vec<Recorded>>>,
    /// Request lane into the writer half of the ws task.
    feeds: tokio::sync::mpsc::UnboundedSender<String>,
    task: tokio::task::JoinHandle<()>,
    /// The chrome process itself — owned here so the session's lifetime
    /// IS the browser's (drop = kill_on_drop fires).
    child: tokio::process::Child,
}

type PendingMap = Arc<Mutex<HashMap<u64, tokio::sync::oneshot::Sender<anyhow::Result<Value>>>>>;

impl CdpSession {
    /// Spawn chrome (argv-only, throwaway profile, ephemeral debug port),
    /// attach to the page target's debugger ws, wire the event loop.
    async fn launch(
        chrome_binary: &str,
        target_url: &str,
        headless: bool,
        recorder_display: Option<&str>,
    ) -> anyhow::Result<Self> {
        let profile = std::env::temp_dir().join(format!(
            "jerith-wf-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&profile)?;

        let mut cmd = tokio::process::Command::new(chrome_binary);
        cmd.args([
            "--remote-debugging-port=0",
            "--no-first-run",
            "--no-default-browser-check",
            &format!("--user-data-dir={}", profile.display()),
            target_url,
        ])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
        if headless {
            cmd.arg("--headless=new").env_remove("DISPLAY");
        } else if let Some(d) = recorder_display {
            cmd.arg(format!("--display={d}"));
        }
        let mut child = cmd
            .spawn()
            .with_context(|| format!("spawning {chrome_binary}"))?;
        // Chrome prints `DevTools listening on ws://127.0.0.1:<port>/…` at
        // startup; that port also serves /json/list for the page target.
        let ws_url = wait_devtools_line(&mut child, Duration::from_secs(30)).await?;
        let port = ws_url
            .trim_start_matches("ws://127.0.0.1:")
            .split('/')
            .next()
            .context("devtools url shape")?
            .to_string();

        // The page target's own ws: events arrive WITHOUT session ids —
        // no flatten/attach dance needed for a single-page flow.
        let page_ws: String = loop {
            let http = format!("http://127.0.0.1:{port}/json/list");
            let resp = crate::api::client()
                .get(&http)
                .timeout(Duration::from_secs(10))
                .send()
                .await
                .with_context(|| format!("devtools list {http}"))?;
            let v: Value = resp.json().await.unwrap_or(json!([]));
            let found = v.as_array().and_then(|a| {
                a.iter()
                    .find(|t| {
                        t.get("type").and_then(Value::as_str) == Some("page")
                            && t.get("webSocketDebuggerUrl")
                                .and_then(Value::as_str)
                                .is_some()
                    })
                    .and_then(|t| {
                        t.get("webSocketDebuggerUrl")
                            .and_then(Value::as_str)
                            .map(str::to_string)
                    })
            });
            if let Some(ws) = found {
                break ws;
            }
            anyhow::ensure!(
                tokio::time::Instant::now().elapsed() < Duration::from_secs(30),
                "no chrome page target appeared within 30s"
            );
            tokio::time::sleep(Duration::from_millis(250)).await;
        };

        let connect_req = {
            use tokio_tungstenite::tungstenite::client::IntoClientRequest as _;
            page_ws.into_client_request().context("cdp ws request")?
        };
        let (ws, _handshake) = tokio_tungstenite::connect_async(connect_req)
            .await
            .context("cdp websocket connect")?;

        let pending: PendingMap = Arc::default();
        let events: Arc<Mutex<Vec<Recorded>>> = Arc::default();
        let (feeds_tx, feeds_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        use futures_util::StreamExt as _;
        let (mut ws_sink, mut ws_stream) = ws.split();
        let pw = pending.clone();
        let ev_snap = events.clone();
        let task = tokio::spawn(async move {
            use futures_util::SinkExt as _;
            use futures_util::StreamExt as _;
            use tokio_tungstenite::tungstenite::Message as M;

            let mut feeds = feeds_rx;
            loop {
                tokio::select! {
                    feed = feeds.recv() => {
                        match feed {
                            Some(text) => {
                                let _ = ws_sink.send(M::Text(text.into())).await;
                            }
                            None => return, // session dropped
                        }
                    }
                    frame = ws_stream.next() => {
                        let Some(Ok(M::Text(t))) = frame else {
                            // Pongs/closures: the reader's job is responses;
                            // a closed socket strands pending oneshots,
                            // which fire their typed timeouts on their own.
                            if frame.is_none() {
                                return;
                            }
                            continue;
                        };
                        let Ok(v) = serde_json::from_str::<Value>(&t) else { continue };
                        if let Some(id) = v.get("id").and_then(Value::as_u64) {
                            if let Some(cb) = pw.lock().expect("pending map").remove(&id) {
                                let _ = cb.send(Ok(v));
                            }
                            continue;
                        }
                        let Some(method) = v.get("method").and_then(Value::as_str) else { continue };
                        match method {
                            "Runtime.consoleAPICalled" => {
                                for arg in v
                                    .pointer("/params/args")
                                    .and_then(Value::as_array)
                                    .map(|a| a.as_slice())
                                    .unwrap_or(&[])
                                {
                                    if let Some(rest) = arg
                                        .get("value")
                                        .and_then(Value::as_str)
                                        .and_then(|s| s.strip_prefix("jerith-rec::"))
                                        && let Ok(r) = serde_json::from_str::<Recorded>(rest)
                                    {
                                        ev_snap.lock().expect("events").push(r);
                                    }
                                }
                            }
                            "Page.frameNavigated" => {
                                let main_frame =
                                    v.pointer("/params/frame/parentId").is_none();
                                let url = v
                                    .pointer("/params/frame/url")
                                    .and_then(Value::as_str)
                                    .unwrap_or("");
                                if main_frame && url.starts_with("http") {
                                    ev_snap.lock().expect("events").push(Recorded {
                                        kind: "nav".into(),
                                        selector: url.to_string(),
                                        label: String::new(),
                                        text: String::new(),
                                    });
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        });

        let mut session = Self {
            next_id: 1,
            pending,
            events,
            feeds: feeds_tx,
            task,
            child,
        };
        session.command("Page.enable", json!({})).await?;
        session.command("Runtime.enable", json!({})).await?;
        Ok(session)
    }

    /// One CDP round-trip (bounded, typed).
    async fn command(&mut self, method: &str, params: Value) -> anyhow::Result<Value> {
        let id = self.next_id;
        self.next_id += 1;
        let payload = json!({"id": id, "method": method, "params": params}).to_string();
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.pending.lock().expect("pending map").insert(id, tx);
        self.feeds.send(payload).context("cdp writer task exited")?;
        let v = tokio::time::timeout(Duration::from_secs(20), rx)
            .await
            .with_context(|| format!("cdp {method} timed out (chrome wedged)"))?
            .context("cdp dropped the response")??;
        if !v.get("error").is_none() {
            anyhow::bail!("cdp {method}: {}", v.get("error").unwrap());
        }
        Ok(v)
    }

    /// Runtime.evaluate with a `returnByValue` result.
    async fn evaluate(&mut self, expression: &str) -> anyhow::Result<Value> {
        let v = self
            .command(
                "Runtime.evaluate",
                json!({
                    "expression": expression,
                    "returnByValue": true,
                    "awaitPromise": true,
                }),
            )
            .await?;
        // Runtime.evaluate shape: result.result = the returned object.
        Ok(v.pointer("/result/result").cloned().unwrap_or(Value::Null))
    }
}

impl Drop for CdpSession {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn wait_devtools_line(
    child: &mut tokio::process::Child,
    timeout: Duration,
) -> anyhow::Result<String> {
    use tokio::io::AsyncBufReadExt as _;
    let stderr = child.stderr.take().context("chrome stderr not captured")?;
    let mut lines = tokio::io::BufReader::new(stderr).lines();
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let remaining = deadline
            .checked_duration_since(tokio::time::Instant::now())
            .context("chrome missing a DevTools endpoint (still booting)")?;
        match tokio::time::timeout(remaining, lines.next_line()).await {
            Ok(Ok(Some(line))) if line.contains("DevTools listening") => {
                return line
                    .split_whitespace()
                    .find(|tok| tok.starts_with("ws://"))
                    .map(str::to_string)
                    .with_context(|| format!("devtools line shape: {line}"));
            }
            Ok(Ok(Some(_))) => continue,
            Ok(Ok(None)) => anyhow::bail!("chrome exited before printing a DevTools endpoint"),
            Ok(Err(e)) => return Err(anyhow::anyhow!("reading chrome stderr: {e}")),
            Err(_) => {
                anyhow::bail!(
                    "timed out ({}s) waiting for the DevTools endpoint",
                    timeout.as_secs()
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The CdpSession as a BrowserDriver (replay path)
// ---------------------------------------------------------------------------

impl BrowserDriver for CdpSession {
    fn goto<'a>(&'a mut self, url: &'a str) -> BoxFut<'a, anyhow::Result<()>> {
        Box::pin(async move {
            ensure_url(url)?;
            self.command("Page.navigate", json!({"url": url})).await?;
            Ok(())
        })
    }

    fn click_on<'a>(&'a mut self, selector: &'a str) -> BoxFut<'a, anyhow::Result<()>> {
        Box::pin(async move {
            ensure_selector(selector)?;
            let sel = serde_json::to_string(selector)?;
            // Locate + scroll + measure once in JS, then dispatch REAL
            // mouse press/release at the center — same activation class
            // the owner's physical clicks belong to.
            let v = self
                .evaluate(&format!(
                    "(() => {{ const el = document.querySelector({sel}); \
                     if (!el) return {{err: 'selector did not match'}}; \
                     el.scrollIntoView({{block: 'center'}}); \
                     const r = el.getBoundingClientRect(); \
                     return {{x: r.x + r.width / 2, y: r.y + r.height / 2}}; \
                     }})()"
                ))
                .await?;
            // returnByValue wraps the object at result.result.value — the
            // error/coords live one level deeper than the bare result.
            let inner = v.get("value").cloned().unwrap_or(v.clone());
            if let Some(err) = inner.get("err").and_then(Value::as_str) {
                anyhow::bail!("{err} ({selector})");
            }
            let (Some(x), Some(y)) = (
                inner.get("x").and_then(Value::as_f64),
                inner.get("y").and_then(Value::as_f64),
            ) else {
                anyhow::bail!("element center not measurable ({selector})");
            };
            self.command(
                "Input.dispatchMouseEvent",
                json!({"type": "mousePressed", "x": x, "y": y, "button": "left", "clickCount": 1}),
            )
            .await?;
            self.command(
                "Input.dispatchMouseEvent",
                json!({"type": "mouseReleased", "x": x, "y": y, "button": "left", "clickCount": 1}),
            )
            .await?;
            Ok(())
        })
    }

    fn type_text<'a>(
        &'a mut self,
        selector: &'a str,
        text: &'a str,
    ) -> BoxFut<'a, anyhow::Result<()>> {
        Box::pin(async move {
            ensure_selector(selector)?;
            anyhow::ensure!(text.chars().count() <= TEXT_CAP, "text exceeds cap");
            let sel = serde_json::to_string(selector)?;
            let txt = serde_json::to_string(text)?;
            // Native value setter + input/change events: the same shape
            // framework listeners (React et al.) observe for real typing.
            let v = self
                .evaluate(&format!(
                    "(() => {{ const el = document.querySelector({sel}); \
                     if (!el) return {{err: 'selector did not match'}}; \
                     if (el.value === undefined) return {{err: 'element has no value'}}; \
                     try {{ \
                       const d = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(el), 'value'); \
                       if (!d || !d.set) return {{err: 'no native value setter'}}; \
                       d.set.call(el, {txt}); \
                       el.dispatchEvent(new Event('input', {{bubbles: true}})); \
                       el.dispatchEvent(new Event('change', {{bubbles: true}})); \
                       return {{ok: true}}; \
                     }} catch (e) {{ return {{err: String(e)}}; }} \
                     }})()"
                ))
                .await?;
            let inner = v.get("value").cloned().unwrap_or(v.clone());
            if let Some(err) = inner.get("err").and_then(Value::as_str) {
                anyhow::bail!("{err} ({selector})");
            }
            Ok(())
        })
    }

    fn press_key<'a>(&'a mut self, key: &'a str) -> BoxFut<'a, anyhow::Result<()>> {
        Box::pin(async move {
            let (vk, code) = match key {
                "Enter" => (13, "Enter"),
                "Tab" => (9, "Tab"),
                "Escape" => (27, "Escape"),
                "ArrowDown" => (40, "ArrowDown"),
                "ArrowUp" => (38, "ArrowUp"),
                other => anyhow::bail!("unsupported key {other:?}"),
            };
            self.command(
                "Input.dispatchKeyEvent",
                json!({
                    "type": "keyDown", "key": code, "code": code,
                    "windowsVirtualKeyCode": vk, "nativeVirtualKeyCode": vk,
                    "text": if key == "Enter" { "\r" } else { " " },
                    "unmodifiedText": if key == "Enter" { "\r" } else { " " },
                }),
            )
            .await?;
            self.command(
                "Input.dispatchKeyEvent",
                json!({
                    "type": "keyUp", "key": code, "code": code,
                    "windowsVirtualKeyCode": vk, "nativeVirtualKeyCode": vk,
                }),
            )
            .await?;
            Ok(())
        })
    }

    fn drain_recorded<'s>(&'s mut self) -> BoxFut<'s, anyhow::Result<Vec<Recorded>>> {
        Box::pin(async move { Ok(self.events.lock().expect("events").drain(..).collect()) })
    }
}

impl CdpSession {
    /// Explicit shutdown helper: logs the child's liveness for the
    /// journal (Drop aborts the ws task; kill_on_drop kills the child).
    async fn shutdown(mut self) {
        let chrome_exited = self
            .child
            .try_wait()
            .map(|status| status.is_some())
            .unwrap_or(true);
        tracing::info!(
            target: "bot",
            chrome_exited,
            "workflow browser session shut down"
        );
        drop(self);
    }
}

// ---------------------------------------------------------------------------
// The live record session (owner-driven capture)
// ---------------------------------------------------------------------------

struct RecState {
    name: String,
    start_url: String,
    driver: CdpSession,
}

fn recorder() -> &'static Mutex<Option<RecState>> {
    static R: OnceLock<Mutex<Option<RecState>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(None))
}

async fn record_start(cfg: &BotConfig, name: &str, url: &str) -> anyhow::Result<Value> {
    anyhow::ensure!(
        valid_workflow_name(name).is_ok(),
        "invalid workflow name: {name}"
    );
    ensure_url(url)?;
    {
        let rec = recorder().lock().expect("recorder lock");
        if rec.is_some() {
            let active = rec.as_ref().unwrap().name.clone();
            anyhow::bail!(
                "a recording session is already live ({active}) — run workflow.record.stop first"
            );
        }
    }
    anyhow::ensure!(
        which(cfg.workflows.chrome_binary.as_str()).is_ok(),
        "chrome binary not found: {} — set [workflows] chrome_binary",
        cfg.workflows.chrome_binary
    );
    let mut driver = CdpSession::launch(
        &cfg.workflows.chrome_binary,
        url,
        false, // headed: the owner drives the window
        Some(&cfg.workflows.recorder_display),
    )
    .await?;
    // Seed the recorder into the fresh page.
    let injected = driver.evaluate(RECORDER_JS).await?;
    let seeded = injected
        .get("value")
        .and_then(Value::as_str)
        .unwrap_or_default();
    anyhow::ensure!(
        seeded == "ok" || seeded == "already",
        "recorder did not arm (got {seeded:?})"
    );
    let mut rec = recorder().lock().expect("recorder lock");
    *rec = Some(RecState {
        name: name.to_string(),
        start_url: url.to_string(),
        driver,
    });
    Ok(json!({
        "ok": true,
        "recording": name,
        "started_at": url,
        "note": "do the task in the chrome window now; run workflow.record.stop when done",
    }))
}

async fn record_stop(_cfg: &BotConfig, name: &str) -> anyhow::Result<Value> {
    let mut state = recorder()
        .lock()
        .expect("recorder lock")
        .take()
        .with_context(|| "no live recording session — run workflow.record.start first")?;
    anyhow::ensure!(
        state.name == name,
        "live session is {:?} — stop that one (asked for {name:?})",
        state.name
    );
    let events = state.driver.drain_recorded().await?;
    let raw_vels = events.len();
    let steps = coalesce(events, &state.start_url);
    let w = Workflow {
        name: state.name.clone(),
        url: state.start_url.clone(),
        created_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0),
        steps,
        last_run_outcome: None,
    };
    anyhow::ensure!(
        !w.steps.is_empty(),
        "the recorder captured 0 steps — nothing to save"
    );
    let clicks = w
        .steps
        .iter()
        .filter(|s| matches!(s, Step::Click { .. }))
        .count();
    let fields = w
        .steps
        .iter()
        .filter(|s| matches!(s, Step::Type { .. }))
        .count();
    let path = save(&w)?;
    state.driver.shutdown().await;
    Ok(json!({
        "ok": true,
        "workflow": w.name,
        "steps": w.steps.len(),
        "clicks": clicks,
        "fields": fields,
        "raw_events": raw_vels,
        "saved_to": path.display().to_string(),
        "note": "replay it with workflow.run"
    }))
}

async fn run(cfg: &BotConfig, name: &str) -> anyhow::Result<Value> {
    let mut w = load(name)?;
    anyhow::ensure!(
        which(cfg.workflows.chrome_binary.as_str()).is_ok(),
        "chrome binary not found: {}",
        cfg.workflows.chrome_binary
    );
    let mut driver = CdpSession::launch(
        &cfg.workflows.chrome_binary,
        &w.url,
        true, // replay is headless
        None,
    )
    .await?;
    // Give the start page a moment to settle before the first step.
    tokio::time::sleep(Duration::from_millis(1_500)).await;
    let lines = replay(
        &w,
        &mut driver,
        Duration::from_secs(cfg.workflows.run_budget_secs),
    )
    .await;
    driver.shutdown().await;
    let ok = lines.iter().all(|l| l.starts_with('✓'));
    w.last_run_outcome = Some(format!(
        "{} · {}",
        lines.last().unwrap_or(&"(no steps)".to_string()),
        if ok { "full replay" } else { "partial" }
    ));
    let _ = save(&w);
    Ok(json!({
        "workflow": w.name,
        "ok": ok,
        "detail": lines.join("\n"),
    }))
}

async fn show(name: &str) -> anyhow::Result<Value> {
    let w = load(name)?;
    Ok(serde_json::to_value(&w).unwrap_or(Value::Null))
}

async fn delete(name: &str, confirm: &str) -> anyhow::Result<Value> {
    anyhow::ensure!(
        name == confirm,
        "confirm must restate the workflow name to delete ({name:?})"
    );
    remove(name)?;
    Ok(json!({"ok": true}))
}

/// Cheap binary existence check (PATH or an absolute path).
fn which(bin: &str) -> anyhow::Result<PathBuf> {
    if bin.contains('/') {
        let p = PathBuf::from(bin);
        anyhow::ensure!(p.is_file(), "{bin} missing");
        return Ok(p);
    }
    if let Ok(path) = std::env::var("PATH") {
        for dir in path.split(':') {
            let cand = PathBuf::from(dir).join(bin);
            if cand.is_file() {
                return Ok(cand);
            }
        }
    }
    anyhow::bail!("{bin} not on PATH")
}

// ---------------------------------------------------------------------------
// The model-facing skill surface (plugin id `workflows`, owner-only)
// ---------------------------------------------------------------------------

fn require_field(args: &Value, field: &str) -> anyhow::Result<String> {
    args.get(field)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .with_context(|| format!("field {field:?} is required"))
}

fn record_start_runner<'a>(
    cfg: &'a BotConfig,
    args: &'a Value,
) -> BoxFut<'a, anyhow::Result<Value>> {
    Box::pin(async move {
        let name = require_field(args, "name")?;
        let url = require_field(args, "url")?;
        record_start(cfg, &name, &url).await
    })
}

fn record_stop_runner<'a>(
    cfg: &'a BotConfig,
    args: &'a Value,
) -> BoxFut<'a, anyhow::Result<Value>> {
    Box::pin(async move {
        let name = require_field(args, "name")?;
        record_stop(cfg, &name).await
    })
}

fn run_runner<'a>(cfg: &'a BotConfig, args: &'a Value) -> BoxFut<'a, anyhow::Result<Value>> {
    Box::pin(async move {
        let name = require_field(args, "name")?;
        run(cfg, &name).await
    })
}

fn list_runner<'a>(_cfg: &'a BotConfig, _args: &'a Value) -> BoxFut<'a, anyhow::Result<Value>> {
    Box::pin(async move {
        let names = list_names()?;
        Ok(json!({"workflows": names}))
    })
}

fn show_runner<'a>(_cfg: &'a BotConfig, args: &'a Value) -> BoxFut<'a, anyhow::Result<Value>> {
    Box::pin(async move {
        let name = require_field(args, "name")?;
        show(&name).await
    })
}

fn delete_runner<'a>(_cfg: &'a BotConfig, args: &'a Value) -> BoxFut<'a, anyhow::Result<Value>> {
    Box::pin(async move {
        let name = require_field(args, "name")?;
        let confirm = require_field(args, "confirm")?;
        delete(&name, &confirm).await
    })
}

/// The workflows skills joined into the chat registry (plugin id
/// `workflows`, owner-DM-only via OWNER_ONLY_PLUGINS).
pub fn skills() -> Vec<crate::plugins::SkillSpec> {
    use crate::plugins::SkillSpec;
    vec![
        SkillSpec {
            name: "workflow.list",
            plugin: "workflows",
            description: "List saved workflows (recorded browser routings).",
            parameters: json!({"type": "object"}),
            run: list_runner,
        },
        SkillSpec {
            name: "workflow.record.start",
            plugin: "workflows",
            description: "Start RECORDING a workflow: opens the site in a real chrome window for the OWNER to drive; every click/field/page change is captured. Do the task, then run workflow.record.stop with the same name.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Workflow name [A-Za-z0-9._-]"},
                    "url": {"type": "string", "description": "Where the flow starts"}
                },
                "required": ["name", "url"]
            }),
            run: record_start_runner,
        },
        SkillSpec {
            name: "workflow.record.stop",
            plugin: "workflows",
            description: "Stop the live recording and save the coalesced steps as the named workflow.",
            parameters: json!({
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"]
            }),
            run: record_stop_runner,
        },
        SkillSpec {
            name: "workflow.run",
            plugin: "workflows",
            description: "Replay a saved workflow headlessly (fresh throwaway chrome profile). Returns per-step results — which step and what broke, on failure.",
            parameters: json!({
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"]
            }),
            run: run_runner,
        },
        SkillSpec {
            name: "workflow.show",
            plugin: "workflows",
            description: "Show one saved workflow's steps.",
            parameters: json!({
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"]
            }),
            run: show_runner,
        },
        SkillSpec {
            name: "workflow.delete",
            plugin: "workflows",
            description: "Delete a saved workflow. Two-key gate: confirm must restate the workflow name.",
            parameters: json!({
                "type": "object",
                "properties": {"name": {"type": "string"}, "confirm": {"type": "string"}},
                "required": ["name", "confirm"]
            }),
            run: delete_runner,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- pure model tests (no chrome, no display) -----

    #[test]
    fn coalesce_collapses_typing_and_drops_the_start_nav() {
        let events = vec![
            Recorded {
                kind: "input".into(),
                selector: "#email".into(),
                label: String::new(),
                text: "o".into(),
            },
            Recorded {
                kind: "input".into(),
                selector: "#email".into(),
                label: String::new(),
                text: "ops@x".into(),
            },
            Recorded {
                kind: "click".into(),
                selector: "#login".into(),
                label: "Log in".into(),
                text: String::new(),
            },
            Recorded {
                kind: "nav".into(),
                selector: "https://acme.example/invoice/new".into(),
                label: String::new(),
                text: String::new(),
            },
            Recorded {
                kind: "nav".into(),
                selector: "https://acme.example/login".into(),
                label: String::new(),
                text: String::new(),
            },
        ];
        let steps = coalesce(events, "https://acme.example/login");
        assert_eq!(
            steps,
            vec![
                Step::Type {
                    selector: "#email".into(),
                    text: "ops@x".into()
                },
                Step::Click {
                    selector: "#login".into(),
                    label: "Log in".into()
                },
                Step::Goto {
                    url: "https://acme.example/invoice/new".into()
                },
            ],
            "typing collapses to the LAST value; start-page nav dropped"
        );
    }

    #[test]
    fn workflow_validation_is_typed() {
        let good = Workflow {
            name: "acme-invoices".into(),
            url: "https://acme.example".into(),
            steps: vec![Step::Click {
                selector: "#a".into(),
                label: "x".into(),
            }],
            created_ms: 0,
            last_run_outcome: None,
        };
        assert!(validate_workflow(&good).is_ok());

        let bad_name = Workflow {
            name: "has space".into(),
            ..good.clone()
        };
        assert!(validate_workflow(&bad_name).is_err());
        let bad_url = Workflow {
            url: "ftp://acme".into(),
            ..good.clone()
        };
        assert!(validate_workflow(&bad_url).is_err());
        let bad_selector = Workflow {
            steps: vec![Step::Type {
                selector: "".into(),
                text: "x".into(),
            }],
            ..good.clone()
        };
        assert!(validate_workflow(&bad_selector).is_err(), "empty selector");
        let bad_key = Workflow {
            steps: vec![Step::Press { key: "F13".into() }],
            ..good.clone()
        };
        assert!(
            validate_workflow(&bad_key).is_err(),
            "key outside the recorded set"
        );
        assert!(workflow_path("has space").is_err(), "bad name → no path");
    }

    // ----- replay engine against a scripted mock driver -----

    struct MockDriver {
        fail_on: Option<String>,
        actions: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    }

    impl BrowserDriver for MockDriver {
        fn goto<'a>(&'a mut self, url: &'a str) -> BoxFut<'a, anyhow::Result<()>> {
            self.actions.lock().unwrap().push(format!("goto {url}"));
            Box::pin(async move { Ok(()) })
        }
        fn click_on<'a>(&'a mut self, selector: &'a str) -> BoxFut<'a, anyhow::Result<()>> {
            self.actions
                .lock()
                .unwrap()
                .push(format!("click {selector}"));
            let fail = self.fail_on.as_deref() == Some(selector);
            Box::pin(async move {
                if fail {
                    anyhow::bail!("selector did not match (=failing mock)");
                }
                Ok(())
            })
        }
        fn type_text<'a>(
            &'a mut self,
            selector: &'a str,
            text: &'a str,
        ) -> BoxFut<'a, anyhow::Result<()>> {
            self.actions
                .lock()
                .unwrap()
                .push(format!("type {selector}={text}"));
            Box::pin(async move { Ok(()) })
        }
        fn press_key<'a>(&'a mut self, key: &'a str) -> BoxFut<'a, anyhow::Result<()>> {
            self.actions.lock().unwrap().push(format!("press {key}"));
            Box::pin(async move { Ok(()) })
        }
        fn drain_recorded(&mut self) -> BoxFut<'_, anyhow::Result<Vec<Recorded>>> {
            Box::pin(async move { Ok(Vec::new()) })
        }
    }

    /// LIVE smoke (chrome must exist on the host): launches a headless
    /// chrome with a throwaway profile through the production CDP path —
    /// spawn, devtools-ws attach, Runtime.evaluate round-trip. Run with:
    ///   cargo test -p jerith-mesh-bot workflows -- --ignored
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore]
    async fn chrome_cdp_live_smoke() {
        let chrome = std::env::var("JERITH_CHROME").unwrap_or_else(|_| "google-chrome".into());
        let mut s = CdpSession::launch(&chrome, "data:text/html,<div id='a'>hi</div>", true, None)
            .await
            .expect("chrome spawns + devtools attaches");
        let v = s.evaluate("1 + 1").await.expect("evaluate roundtrip");
        assert_eq!(
            v["value"].as_i64(),
            Some(2),
            "runtime.evaluate returns by value"
        );
        // Click the div through the Input domain (mouse press + release).
        s.click_on("#a").await.expect("real input click dispatched");
        s.type_text("#a", "x")
            .await
            .expect_err("type on a non-field fails typed");
        s.shutdown().await;
    }

    #[tokio::test]
    async fn replays_steps_and_reports_the_failing_one() {
        let actions = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut driver = MockDriver {
            fail_on: None,
            actions: actions.clone(),
        };
        let w = Workflow {
            name: "t".into(),
            url: "https://x.example".into(),
            steps: vec![
                Step::Type {
                    selector: "#a".into(),
                    text: "hello".into(),
                },
                Step::Click {
                    selector: "#go".into(),
                    label: String::new(),
                },
            ],
            created_ms: 0,
            last_run_outcome: None,
        };
        let lines = replay(&w, &mut driver, Duration::from_secs(5)).await;
        assert_eq!(lines.len(), 2, "two ✓ lines");
        assert!(lines.iter().all(|l| l.starts_with('✓')));
        let acts = actions.lock().unwrap().clone();
        assert_eq!(acts, vec!["type #a=hello", "click #go"]);

        // A rotted selector breaks the flow AT that step, typed.
        let mut driver = MockDriver {
            fail_on: Some("#go".into()),
            actions,
        };
        let lines = replay(&w, &mut driver, Duration::from_secs(5)).await;
        assert_eq!(lines.len(), 2, "flow stops at the failure");
        assert!(lines[0].starts_with('✓'));
        let failing = &lines[1];
        assert!(
            failing.starts_with('✗') && failing.contains("step 2") && failing.contains("#go"),
            "{failing}"
        );
    }
}
