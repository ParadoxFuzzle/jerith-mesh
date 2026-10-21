//! Discord voice lane (always-on VC + Gemini Live) — FEATURES §23.
//!
//! Deliberate split of concerns:
//!   * songbird = the Discord transport ONLY: joins the VC, hands us
//!     members' PCM (stereo 48k interleaved i16, decoded by its own
//!     receive pipeline) and pulls playback through its mixer. The
//!     Gemini side rides tokio-tungstenite, already in the tree.
//!   * `gemini-3.8-live` = the dialog model. ONE stateful WSS session at
//!     a time, re-established under the wall-time cap; barge-in is the
//!     model's own VAD (`interrupted` → drain the play ring).
//!   * `gemini-3.5-transcribe-live` = a PARALLEL captions lane (STT)
//!     over a second WSS fed the SAME room audio; captions sink into the
//!     node's private memory (+ optional text channel later).
//!
//! Honesty carried over from every other lane: live room capture starts
//! ONLY by explicit config (`[voice] enabled = true`, default OFF);
//! failures degrade to typed journal lines and supervised restarts —
//! never a crash; the key is never logged; the wall-time cap bounds
//! server session cost.

use crate::config::BotConfig;
use base64::Engine as _;
use futures_util::{SinkExt, StreamExt};
use poise::serenity_prelude as serenall;
use serde_json::{Value, json};
use songbird::events::{CoreEvent, Event, EventContext, EventHandler};
use songbird::input::Input;
use std::io::{Read, Seek};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;
use tokio::sync::broadcast;

// WSS message type shorthands (tokio-tungstenite 0.28).
use tokio_tungstenite::tungstenite::Message as WsMsg;

// ---------------------------------------------------------------------------
// Config — `[voice]` in bot.toml (default OFF)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Deserialize)]
pub struct VoiceConfig {
    /// Master switch (default off — live room capture is opt-in only).
    #[serde(default)]
    pub enabled: bool,
    /// Always-on seat: guild + voice channel.
    pub guild_id: Option<u64>,
    pub channel_id: Option<u64>,
    /// Google AI Studio key (pushed at setup from the operator's
    /// litellm config; never logged).
    pub gemini_api_key: String,
    /// Dialog model (Google's recommended Live default).
    #[serde(default = "default_voice_model")]
    pub model: String,
    /// Prebuilt voice the model speaks with.
    #[serde(default = "default_voice_name")]
    pub voice_name: String,
    /// Optional text channel captions stream into (reserved knob).
    pub captions_channel_id: Option<u64>,
    /// Live session wall-time cap (minutes; clamp 5–120, default 30).
    #[serde(default = "default_session_max_minutes")]
    pub session_max_minutes: u64,
}

impl Default for VoiceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            guild_id: None,
            channel_id: None,
            gemini_api_key: String::new(),
            model: default_voice_model(),
            voice_name: default_voice_name(),
            captions_channel_id: None,
            session_max_minutes: default_session_max_minutes(),
        }
    }
}

pub fn default_voice_model() -> String {
    "gemini-3.8-live".to_string()
}
pub fn default_voice_name() -> String {
    "Charon".to_string()
}
pub fn default_session_max_minutes() -> u64 {
    30
}

impl VoiceConfig {
    /// Clamp operator mistakes (the sandbox pattern).
    pub fn clamp(&mut self) {
        self.session_max_minutes = self.session_max_minutes.clamp(5, 120);
    }
    /// runnable() = the lane is actually startable (full config).
    pub fn runnable(&self) -> bool {
        self.enabled
            && self.guild_id.is_some()
            && self.channel_id.is_some()
            && !self.gemini_api_key.trim().is_empty()
            && !self.model.trim().is_empty()
    }
}

/// Live WSS endpoint (official v1beta GenerativeService BidiGenerate
/// path; JSON wire; key as a query param — the URL is never logged).
/// The MODEL is NOT a query parameter — the server rejects `?model=`
/// with 1007 «Unknown name model» (found live 2026-10-21); it rides
/// inside the setup message instead.
pub fn live_ws_url(cfg: &VoiceConfig, _model: &str) -> String {
    format!(
        "wss://generativelanguage.googleapis.com/ws/{}?key={}",
        "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent",
        cfg.gemini_api_key.trim(),
    )
}

/// The FIRST session message (official docs shape): dialog setup.
pub fn setup_message(cfg: &VoiceConfig, model: &str, system_prompt: Option<&str>) -> Value {
    let mut setup = json!({
        "model": format!("models/{}", model.trim()),
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": cfg.voice_name.trim()}
                }
            }
        },
        "inputAudioTranscription": {},
        "outputAudioTranscription": {},
    });
    if let Some(p) = system_prompt.map(str::trim).filter(|p| !p.is_empty()) {
        setup["systemInstruction"] = json!({"parts": [{"text": p}]});
    }
    json!({"setup": setup})
}

/// Captions lane setup (STT model — TEXT in response, plain input
/// transcriptions; `diarization` is REJECTED by BidiGenerateContent
/// with 1007 «diarization is not supported» — found live 2026-10-21).
pub fn setup_message_stt(_cfg: &VoiceConfig, model: &str) -> Value {
    json!({
        "setup": {
            "model": format!("models/{}", model.trim()),
            "generationConfig": {"responseModalities": ["TEXT"]},
            "inputAudioTranscription": {},
        }
    })
}

// ---------------------------------------------------------------------------
// DSP bridges
// ---------------------------------------------------------------------------

/// Discord capture (stereo 48k interleaved) → mono 16k for the Live
/// API: a 3-fold average — cheap LP, honest about aliasing, speech
/// readable.
pub fn downmix_to_16k(stereo48k: &[i16]) -> Vec<i16> {
    if stereo48k.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(stereo48k.len() / 6 + 2);
    let mut acc = 0i64;
    let mut n = 0i64;
    for &v in stereo48k {
        acc += v as i64;
        n += 1;
        if n == 3 {
            out.push((acc / 3) as i16);
            acc = 0;
            n = 0;
        }
    }
    if n > 0 {
        out.push((acc / n) as i16);
    }
    out
}

/// 24k mono (model out) → 48k stereo interleaved (mixer reference):
/// interpolated ×2, identical channels.
pub fn upsample_to_48k_stereo(mono24k: &[i16]) -> Vec<i16> {
    let mut out = Vec::with_capacity(mono24k.len() * 4);
    if mono24k.is_empty() {
        return out;
    }
    for w in mono24k.windows(2) {
        let (a, b) = (w[0], w[1]);
        let mid = ((a as i32 + b as i32 + 1) / 2) as i16;
        out.push(a);
        out.push(a);
        out.push(mid);
        out.push(mid);
    }
    let l = *mono24k.last().unwrap();
    out.push(l);
    out.push(l);
    out
}

// ---------------------------------------------------------------------------
// Play ring: PCM 24k mono in; the reader serves f32 LE stereo interleaved
// at 48k (mixer reference). Blocking MediaSource semantics: a zero-byte
// Ok answer would END the symphonia track forever, so the reader waits
// (condvar) instead — a barge-in drain just re-arms that wait.
// ---------------------------------------------------------------------------

pub struct PlayRing {
    inner: Mutex<std::collections::VecDeque<i16>>,
    cv: Condvar,
    live: AtomicBool,
    wrote: AtomicU64,
    read: AtomicU64,
}

impl Default for PlayRing {
    fn default() -> Self {
        Self {
            inner: Mutex::new(std::collections::VecDeque::new()),
            cv: Condvar::new(),
            live: AtomicBool::new(true),
            wrote: AtomicU64::new(0),
            read: AtomicU64::new(0),
        }
    }
}

impl PlayRing {
    pub fn push_stereo48k(&self, stereo: &[i16]) {
        if stereo.is_empty() {
            return;
        }
        {
            let mut g = self.inner.lock().unwrap();
            for &v in stereo {
                g.push_back(v);
            }
            while g.len() > 48_000 * 2 * 15 {
                g.pop_front();
            }
        }
        self.cv.notify_all();
        self.wrote
            .fetch_add(stereo.len() as u64, std::sync::atomic::Ordering::Relaxed);
    }
    pub fn drain(&self) {
        self.inner.lock().unwrap().clear();
        self.cv.notify_all();
    }
    fn live(&self) -> bool {
        self.live.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// The songbird-facing source: symphonia MediaSource over the ring —
/// interleaved STEREO f32 LE (8 bytes per L/R pair).
pub struct RingSource(pub Arc<PlayRing>);

impl Read for RingSource {
    fn read(&mut self, dst: &mut [u8]) -> std::io::Result<usize> {
        let max_pairs = dst.len() / 8;
        if max_pairs == 0 {
            return Ok(0);
        }
        let mut g = self.0.inner.lock().unwrap();
        loop {
            let pairs = g.len() / 2;
            if pairs > 0 {
                let n = pairs.min(max_pairs);
                for i in 0..n {
                    // ONE ring pair = L, R (interleaved stored — values
                    // with identical L/R are written by upsample's dup).
                    let l = g.pop_front().unwrap_or(0i16);
                    let r = g.pop_front().unwrap_or(0i16);
                    let fl = (l as f32 / 32768.0).to_le_bytes();
                    let fr = (r as f32 / 32768.0).to_le_bytes();
                    dst[i * 8..i * 8 + 4].copy_from_slice(&fl);
                    dst[i * 8 + 4..i * 8 + 8].copy_from_slice(&fr);
                    self.0.read.fetch_add(2, Ordering::Relaxed);
                }
                return Ok(n * 8);
            }
            if !self.0.live() {
                // Lane shutdown: an honest end-of-stream for the mixer.
                return Ok(0);
            }
            let (guard, _) = self
                .0
                .cv
                .wait_timeout(g, Duration::from_millis(250))
                .unwrap();
            g = guard;
        }
    }
}

impl songbird::input::core::io::MediaSource for RingSource {
    fn is_seekable(&self) -> bool {
        false
    }
    fn byte_len(&self) -> Option<u64> {
        None
    }
}

impl Seek for RingSource {
    fn seek(&mut self, _pos: std::io::SeekFrom) -> std::io::Result<u64> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "live voice source is unseekable",
        ))
    }
}

// ---------------------------------------------------------------------------
// The lane body
// ---------------------------------------------------------------------------

/// Called from main's setup (after ready): joins the VC once and runs
/// supervised session cycles (dialog + captions) forever. Any failure
/// restarts under the caps; the Discord side never crashes because the
/// voice lane wobbles.
pub async fn run_lane(
    manager: Arc<songbird::Songbird>,
    cfg: BotConfig,
    system_prompt: Option<String>,
) {
    if !cfg.voice.runnable() {
        tracing::warn!(
            channel = cfg.voice.channel_id.unwrap_or(0),
            "voice enabled but not runnable (ids/key/model missing) — lane stays off"
        );
        return;
    }
    let vcfg = cfg.voice.clone();
    let guild = serenall::GuildId::new(vcfg.guild_id.unwrap());
    let channel = serenall::ChannelId::new(vcfg.channel_id.unwrap());
    lane_started(&vcfg.model, vcfg.captions_channel_id);
    // The persona seeds the Live systemInstruction only in its head:
    // the model is spoken-to, not written-at; long rules degrade realtime.
    let seed: Option<String> = system_prompt
        .as_deref()
        .map(|p| p.chars().take(1200).collect());
    tracing::info!(
        guild = guild.get(),
        channel = channel.get(),
        model = %vcfg.model,
        session_cap_minutes = vcfg.session_max_minutes,
        "voice lane starting (always-on)"
    );
    let call_lock =
        match tokio::time::timeout(Duration::from_secs(25), manager.join(guild, channel)).await {
            Ok(Ok(call)) => call,
            Ok(Err(e)) => {
                lane_stopped();
                tracing::error!(error = %e, "voice join failed — lane off until next boot");
                return;
            }
            Err(_) => {
                lane_stopped();
                tracing::error!("voice join timed out — lane off until next boot");
                return;
            }
        };
    let mut call = call_lock.lock().await;
    // Receive stays ON (the bot must hear the room); Discord does not
    // echo our own packets back, so there is no echo loop to defend.
    let _ = call.deafen(false).await;
    // ONE playback source for the lane's life: the mixer pulls from this
    // ring forever; sessions only change what lands in the ring.
    // play_input returns the TrackHandle (kept unnamed — the ring is
    // immortal by construction).
    let playback = Arc::new(PlayRing::default());
    call.play_input(Input::from(songbird::input::RawAdapter::new(
        RingSource(playback.clone()),
        48_000,
        2,
    )));
    // Room-audio fan-out: BOTH lanes hear every chunk (broadcast = each
    // consumer gets its own copy; laggards skip — honest for live audio).
    // One captions sink serves both lanes' transcription output.
    let (audio_tx, _audio_rx) = broadcast::channel::<Arc<Vec<i16>>>(256);
    let (cap_tx, cap_rx) = broadcast::channel::<String>(256);
    tokio::spawn(caption_worker(cfg.clone(), cap_rx));
    spawn_capture(&mut call, audio_tx.clone());
    lane_halfway(true);
    loop {
        // The stop switch is watched on a one-second tick so /voiceleave
        // takes effect promptly (the Live sessions die within the
        // second, not at the next cycle boundary — a session under the
        // cap runs up to a half hour). The seat leaves through the SAME
        // manager the lane joined with.
        let res = tokio::select! {
            biased;
            _ = watch_for_op_leave() => {
                honor_leave_if_requested(&manager, guild).await;
                return;
            }
            res = async {
                tokio::join! {
                    supervise_dialog(
                        cfg.clone(),
                        vcfg.clone(),
                        audio_tx.subscribe(),
                        playback.clone(),
                        cap_tx.clone(),
                        seed.as_deref(),
                    ),
                    supervise_captions(cfg.clone(), vcfg.clone(), audio_tx.subscribe(), cap_tx.clone()),
                }
            } => res,
        };
        if honor_leave_if_requested(&manager, guild).await {
            return;
        }
        match res.0 {
            Ok(()) => tracing::warn!("dialog session ended — restarting under caps"),
            Err(e) => tracing::error!(error = %e, "dialog session error — restarting"),
        }
        match res.1 {
            Ok(()) => tracing::debug!("captions session ended"),
            Err(e) => tracing::warn!(error = %e, "captions session error — restarting"),
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

/// The Discord http handle parked by main's setup — captions posting
/// (and any other live surface) rides the same Arc the approval bridge
/// stores. Before it lands, captions degrade to journal + memory only.
static CAPTIONS_HTTP: std::sync::OnceLock<Arc<serenall::http::Http>> = std::sync::OnceLock::new();
pub fn set_captions_http(http: Arc<serenall::http::Http>) {
    let _ = CAPTIONS_HTTP.set(http);
}

fn spawn_capture(
    call: &mut tokio::sync::MutexGuard<'_, songbird::Call>,
    tx: broadcast::Sender<Arc<Vec<i16>>>,
) {
    call.add_global_event(Event::Core(CoreEvent::VoiceTick), capture_handler(tx));
}

/// VoiceTick handler: songbird's own receive pipeline already DECODED
/// the packets when the manager is registered with DecodeMode::Decoding
/// (decoded_voice: interleaved stereo 48k i16 per user, jitter-
/// buffered). We just downmix and fan out — no local opus decoder.
pub struct Capture {
    tx: broadcast::Sender<Arc<Vec<i16>>>,
}

/// The remaining call site in this module is the async-trait impl above.
#[async_trait::async_trait]
impl EventHandler for Capture {
    async fn act(&self, ctx: &EventContext<'_>) -> Option<Event> {
        if let EventContext::VoiceTick(vt) = ctx {
            for v in vt.speaking.values() {
                if let Some(pcm) = &v.decoded_voice
                    && pcm.len() >= 6
                {
                    let arc = Arc::new(downmix_to_16k(pcm));
                    let _ = self.tx.send(arc);
                }
            }
        }
        None
    }
}

fn capture_handler(tx: broadcast::Sender<Arc<Vec<i16>>>) -> Capture {
    Capture { tx }
}

// ---------------------------------------------------------------------------
// Supervised session cycles
// ---------------------------------------------------------------------------

async fn supervise_dialog(
    cfg: BotConfig,
    vcfg: VoiceConfig,
    rx: broadcast::Receiver<Arc<Vec<i16>>>,
    playback: Arc<PlayRing>,
    cap_tx: broadcast::Sender<String>,
    seed: Option<&str>,
) -> anyhow::Result<()> {
    let cap = Duration::from_secs(vcfg.session_max_minutes * 60);
    match tokio::time::timeout(cap, dialog_session(&cfg, &vcfg, rx, playback, cap_tx, seed)).await {
        Ok(Ok(())) => tracing::info!("vc dialog session clean end"),
        Ok(Err(e)) => tracing::error!(error = %e, "vc dialog session failed — restarting"),
        Err(_) => tracing::info!(
            minutes = vcfg.session_max_minutes,
            "vc dialog session hit the wall-time cap"
        ),
    }
    Ok(())
}

async fn supervise_captions(
    _cfg: BotConfig,
    vcfg: VoiceConfig,
    rx: broadcast::Receiver<Arc<Vec<i16>>>,
    cap_tx: broadcast::Sender<String>,
) -> anyhow::Result<()> {
    let cap = Duration::from_secs(vcfg.session_max_minutes * 60);
    match tokio::time::timeout(cap, captions_session(&vcfg, rx, cap_tx)).await {
        Ok(Ok(())) => tracing::debug!("vc captions session ended"),
        Ok(Err(e)) => tracing::warn!(error = %e, "vc captions session failed — restarting"),
        Err(_) => tracing::debug!("vc captions session hit the wall-time cap"),
    }
    Ok(())
}

/// The common session leg: connect, hand the FIRST message to the pump
/// (setup ordering is the pump's first frame, so no ordering race), then
/// run the receiver loop with the cap supervisor watching from outside
/// (the async fn is awaited by the supervisors below).
#[allow(clippy::too_many_arguments)]
async fn run_session<F>(
    vcfg: &VoiceConfig,
    model: &str,
    first: Value,
    mut rx: broadcast::Receiver<Arc<Vec<i16>>>,
    mut on_frame: F,
) -> anyhow::Result<()>
where
    F: FnMut(Value) -> anyhow::Result<bool> + Send,
{
    let url = live_ws_url(vcfg, model);
    let (ws, _) = tokio_tungstenite::connect_async(url).await?;
    let (mut w, mut r) = ws.split();
    tracing::info!(model = %model, "vc ws connected");
    // The pump: setup FIRST, then paced realtimeInput. The send half is
    // owned here alone, so no ordering hazard exists.
    tokio::spawn(async move {
        if let Err(e) = w.send(WsMsg::Text(first.to_string().into())).await {
            tracing::error!(error = %e, "vc setup send failed");
            return;
        }
        tracing::info!("vc setup sent");
        let mut quantum: Vec<i16> = Vec::with_capacity(1280);
        loop {
            // Broadcast lags are honest for live audio: a missed chunk is
            // skipped and the stream keeps flowing.
            let chunk = match tokio::time::timeout(std::time::Duration::from_millis(150), rx.recv())
                .await
            {
                Ok(Ok(c)) => c,
                Ok(Err(broadcast::error::RecvError::Lagged(n))) => {
                    tracing::debug!(skipped = n, "vc audio consumer lagged — samples skipped");
                    continue;
                }
                Ok(Err(broadcast::error::RecvError::Closed)) => return,
                // The room is quiet and songbird emits nothing (capture
                // rides real voice packets). Stream 40 ms of digital
                // silence as a keepalive — with zero media Google's
                // bidi session aborts at the 60 s mark with 1008
                // «The operation was aborted» (observed live 2026-10-21).
                Err(_) => {
                    quantum.extend(std::iter::repeat_n(0i16, 640));
                    Arc::new(Vec::new())
                }
            };
            quantum.extend_from_slice(&chunk);
            while quantum.len() >= 640 {
                let taken: Vec<i16> = quantum.drain(..640).collect();
                let mut bytes = Vec::with_capacity(taken.len() * 2);
                for v in taken {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                let msg = json!({
                    "realtimeInput": {
                        "audio": {
                            "mimeType": "audio/pcm;rate=16000",
                            "data": base64::engine::general_purpose::STANDARD.encode(bytes)
                        }
                    }
                });
                if w.send(WsMsg::Text(msg.to_string().into())).await.is_err() {
                    return;
                }
            }
        }
    });
    // First-frame marker (INFO once, then DEBUG per frame).
    let mut first_frame = true;
    // Receiver: parse each text frame; the handler decides whether
    // to continue (false = clean end, e.g. goAway).
    loop {
        let frame = r
            .next()
            .await
            .ok_or_else(|| anyhow::anyhow!("ws closed"))??;
        // Text AND Binary haul the same JSON — observed live 2026-10-21:
        // the bidi endpoint ships setupComplete and friends as BINARY
        // frames; a Text-only match dropped every server reply and the
        // session idled to the server's 1008 abort at the 60 s mark.
        let txt = match frame {
            WsMsg::Text(t) => String::from_utf8_lossy(t.as_bytes()).into_owned(),
            WsMsg::Binary(b) => String::from_utf8_lossy(&b).into_owned(),
            WsMsg::Close(f) => {
                let (code, reason) = f
                    .as_ref()
                    .map(|c| (c.code.to_string(), c.reason.to_string()))
                    .unwrap_or_else(|| ("?".into(), "".into()));
                tracing::warn!(model = %model, close_code = %code, close_reason = %reason, "server closed the session");
                return Ok(());
            }
            _ => continue,
        };
        let v: Value = serde_json::from_str(&txt)?;
        if v.get("error").is_some() {
            anyhow::bail!("server error frame: {v}");
        }
        // Diagnostic honesty: the FIRST server frame is journaled at
        // INFO (truncated) — the setupComplete-never-arrives failure
        // shape of 2026-10-21 was invisible without a frame trace —
        // and every following frame at DEBUG so a busy room doesn't
        // flood the journal.
        if first_frame {
            first_frame = false;
            let head: String = txt.chars().take(200).collect();
            tracing::info!(model = %model, frame = %head, "vc first server frame");
        } else {
            let head: String = txt.chars().take(100).collect();
            tracing::debug!(model = %model, frame = %head, "vc server frame");
        }
        if !on_frame(v)? {
            return Ok(());
        }
    }
}

/// Dialog session over on_frame: setupComplete receipt, model audio →
/// play ring, barge-in → drain, transcriptions → captions sink, goAway
/// → clean end.
async fn dialog_session(
    _cfg: &BotConfig,
    vcfg: &VoiceConfig,
    rx: broadcast::Receiver<Arc<Vec<i16>>>,
    playback: Arc<PlayRing>,
    cap_tx: broadcast::Sender<String>,
    seed: Option<&str>,
) -> anyhow::Result<()> {
    let setup = setup_message(vcfg, &vcfg.model, seed);
    let mut seen_setup = false;
    run_session(vcfg, &vcfg.model, setup, rx, move |v| {
        if v.get("setupComplete").is_some() {
            seen_setup = true;
            tracing::info!(model = %cfg_vc_model(vcfg), "live dialog session up");
            return Ok(true);
        }
        if let Some(sc) = v.get("serverContent") {
            if sc.get("interrupted").and_then(Value::as_bool) == Some(true) {
                playback.drain();
                let _ = cap_tx.send("🎙 [barge-in] stopped model output".to_string());
            }
            if let Some(parts) = sc.pointer("/modelTurn/parts").and_then(Value::as_array) {
                for part in parts {
                    if let Some((data, mime)) = part
                        .pointer("/inlineData/data")
                        .and_then(Value::as_str)
                        .zip(part.pointer("/inlineData/mimeType").and_then(Value::as_str))
                        && mime.starts_with("audio/pcm")
                        && let Ok(raw) = base64::engine::general_purpose::STANDARD.decode(data)
                    {
                        // Decode 24 kHz mono PCM → ×2 upsample into the
                        // 48 kHz stereo ring (the mixer reference); raw
                        // 24k mono plays double-speed — a real bug this
                        // bridge exists to stop.
                        let mono: Vec<i16> = raw
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .map(|c| i16::from_le_bytes(*c))
                            .collect();
                        let stereo48k = upsample_to_48k_stereo(&mono);
                        playback.push_stereo48k(&stereo48k);
                    }
                }
            }
            if let Some(t) = sc
                .pointer("/outputTranscription/text")
                .and_then(Value::as_str)
            {
                let _ = cap_tx.send(format!("🎙 {t}"));
            }
            if let Some(t) = sc
                .pointer("/inputTranscription/text")
                .and_then(Value::as_str)
            {
                let _ = cap_tx.send(t.to_string());
            }
        }
        if v.get("goAway").is_some() {
            tracing::warn!("server goAway — ending dialog session early");
            let _ = cap_tx.send("🎙 [voice] session restarting under caps".to_string());
            return Ok(false);
        }
        let _ = seen_setup;
        Ok(true)
    })
    .await
}

/// Captions session: STT-only model over the same room audio; the
/// transcription lines ride the same captions sink.
async fn captions_session(
    vcfg: &VoiceConfig,
    rx: broadcast::Receiver<Arc<Vec<i16>>>,
    cap_tx: broadcast::Sender<String>,
) -> anyhow::Result<()> {
    let setup = setup_message_stt(vcfg, "gemini-3.5-transcribe-live");
    run_session(vcfg, "gemini-3.5-transcribe-live", setup, rx, move |v| {
        if v.get("setupComplete").is_some() {
            tracing::info!(model = "gemini-3.5-transcribe-live", "captions session up");
            return Ok(true);
        }
        if let Some(t) = v
            .pointer("/serverContent/inputTranscription/text")
            .and_then(Value::as_str)
        {
            let _ = cap_tx.send(t.to_string());
        }
        if v.get("goAway").is_some() {
            return Ok(false);
        }
        Ok(true)
    })
    .await
}

/// Borrows the configured model name for journal lines (keeps the
/// closure-free shape of the on_frame call).
fn cfg_vc_model(vcfg: &VoiceConfig) -> &str {
    &vcfg.model
}

// ---------------------------------------------------------------------------
// Operator-facing runtime state: one honest /status line and a stop
// switch (/voiceleave). This is the LIVE state; `[voice]` in bot.toml
// stays the intent. Zero-config conversations (no /status yet, no
// lane) still get honest strings off the same process globals.
// ---------------------------------------------------------------------------

pub struct VoiceRuntimeState {
    /// The lane cycle is alive (config-run + post-ready spawn).
    running: AtomicBool,
    /// The seat is actually taken (join succeeded).
    in_vc: AtomicBool,
    /// /voiceleave: the cycle stops and the bot stays out until the
    /// next restart (rejoin without a restart is deliberately absent
    /// — joining back is the restart path, one honest control).
    left_by_op: AtomicBool,
    dialog_model: Mutex<String>,
    captions_channel: Mutex<Option<u64>>,
    session_started: Mutex<Option<std::time::Instant>>,
}

static VOICE_RT: std::sync::OnceLock<VoiceRuntimeState> = std::sync::OnceLock::new();

fn rt() -> &'static VoiceRuntimeState {
    VOICE_RT.get_or_init(|| VoiceRuntimeState {
        running: AtomicBool::new(false),
        in_vc: AtomicBool::new(false),
        left_by_op: AtomicBool::new(false),
        dialog_model: Mutex::new(String::new()),
        captions_channel: Mutex::new(None),
        session_started: Mutex::new(None),
    })
}

fn lane_started(model: &str, captions_channel: Option<u64>) {
    let r = rt();
    r.running.store(true, Ordering::Relaxed);
    r.in_vc.store(false, Ordering::Relaxed);
    *r.dialog_model.lock().unwrap() = model.to_string();
    *r.captions_channel.lock().unwrap() = captions_channel;
    *r.session_started.lock().unwrap() = Some(std::time::Instant::now());
}

fn lane_halfway(state: bool) {
    rt().in_vc.store(state, Ordering::Relaxed);
}

fn lane_stopped() {
    let r = rt();
    r.running.store(false, Ordering::Relaxed);
    r.in_vc.store(false, Ordering::Relaxed);
    *r.session_started.lock().unwrap() = None;
}

/// The one /status line. Honest in every state the process can be in.
pub fn status_line() -> String {
    let r = rt();
    if r.left_by_op.load(Ordering::Relaxed) {
        return "voice: left by operator — restart the bot to rejoin".to_string();
    }
    if !r.running.load(Ordering::Relaxed) {
        return "voice: 💤 not running (config-off, or the seat failed to open)".to_string();
    }
    if !r.in_vc.load(Ordering::Relaxed) {
        return "voice: 🔁 lane up — joining the voice channel…".to_string();
    }
    let model = r.dialog_model.lock().unwrap().clone();
    let captions = match *r.captions_channel.lock().unwrap() {
        Some(chan) => format!("captions on <#{chan}>"),
        None => "captions off".to_string(),
    };
    let age = match *r.session_started.lock().unwrap() {
        Some(t) => {
            let s = t.elapsed().as_secs();
            let min = s / 60;
            let sec = s % 60;
            if min > 0 {
                format!(" · {min}m{sec:02}s this session")
            } else {
                format!(" · {sec}s this session")
            }
        }
        None => String::new(),
    };
    format!("voice: 🔊 in VC · {model}{age} · {captions}")
}

/// /voiceleave: the honest verdict for the invoking message.
pub fn request_leave_by_op() -> String {
    let r = rt();
    if r.left_by_op.load(Ordering::Relaxed) || !r.running.load(Ordering::Relaxed) {
        return "the voice lane is not running (or already left) — restart the bot to rejoin"
            .to_string();
    }
    r.left_by_op.store(true, Ordering::Relaxed);
    "leaving the voice channel now — it stays out until the next bot restart".to_string()
}

/// The lane's cycle tests the stop switch and (when set) leaves the VC
/// through the SAME manager it joined with.
async fn honor_leave_if_requested(
    manager: &Arc<songbird::Songbird>,
    guild: serenall::GuildId,
) -> bool {
    if !rt().left_by_op.load(Ordering::Relaxed) {
        return false;
    }
    match tokio::time::timeout(Duration::from_secs(5), manager.leave(guild)).await {
        Ok(Ok(())) => tracing::info!("voice left on operator request — out until the next restart"),
        Ok(Err(e)) => {
            tracing::warn!(error = %e, "voice leave after operator request errored (seat may already be gone)")
        }
        Err(_) => tracing::warn!(
            "voice leave timed out — Discord will drop the seat on its own; lane stopped"
        ),
    }
    lane_stopped();
    true
}

/// One-second tick until the operator's leave lands (or forever while
/// the seat stays normal — the lane's cycle poll is the primary stop
/// path; this tick only bounds the WAIT).
async fn watch_for_op_leave() {
    loop {
        tokio::time::sleep(Duration::from_secs(1)).await;
        if rt().left_by_op.load(Ordering::Relaxed) {
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// Captions sink: journal now, optional Discord channel knob reserved,
// and the same memory lane the archiver + transcript hook ride.
// ---------------------------------------------------------------------------

async fn caption_worker(cfg: BotConfig, mut rx: broadcast::Receiver<String>) {
    let mut acc: String = String::new();
    while let Ok(line) = rx.recv().await {
        tracing::info!(line = %line, "vc caption");
        acc.push_str(line.trim());
        acc.push('\n');
        // Fire-and-forget into the /recall memory (bounded notes, same
        // shape as the attachment transcript hook).
        if acc.chars().count() >= 400 {
            let note = format!(
                "voice-chat transcript fragment (accrued, capped):\n{}",
                acc.chars().take(1600).collect::<String>()
            );
            acc.clear();
            // Live captions surface: when `[voice] captions_channel_id`
            // is configured AND the http handle is parked, each flush
            // ALSO posts as a Discord message (acessibility view).
            if let Some(chan) = cfg.voice.captions_channel_id
                && let Some(http) = CAPTIONS_HTTP.get()
            {
                let excerpt: String = note.chars().take(1800).collect();
                let _ = serenall::ChannelId::new(chan).say(http, excerpt).await;
            }
            match crate::api::memory_embed(&cfg, &note).await {
                Ok((id, provider)) => {
                    tracing::info!(id = %id, provider = %provider, "vc captions indexed")
                }
                Err(e) => tracing::warn!(error = %e, "vc captions embed skipped"),
            }
        }
    }
    if !acc.is_empty() {
        tracing::info!(chars = acc.chars().count(), "vc caption flush at lane end");
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> VoiceConfig {
        VoiceConfig {
            enabled: true,
            guild_id: Some(1),
            channel_id: Some(2),
            gemini_api_key: "TESTKEY".to_string(),
            ..Default::default()
        }
    }

    /// /status + /voiceleave read process-global runtime state, so the
    /// state-shape tests serialize on one mutex.
    static RT_SEQUENCER: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn reset_rt() {
        rt().running.store(false, Ordering::Relaxed);
        rt().in_vc.store(false, Ordering::Relaxed);
        rt().left_by_op.store(false, Ordering::Relaxed);
        *rt().session_started.lock().unwrap() = None;
    }

    #[test]
    fn status_line_is_honest_in_every_runtime_state() {
        let _g = RT_SEQUENCER.lock().unwrap();
        reset_rt();
        assert!(crate::voice::status_line().contains("not running"));
        lane_started("glm-5.3-live", Some(42));
        assert!(crate::voice::status_line().contains("joining"));
        lane_halfway(true);
        let live = crate::voice::status_line();
        assert!(live.contains("in VC") && live.contains("glm-5.3-live"));
        assert!(live.contains("captions on <#42>"));
        // The leave switch: once set, status flips to `left` and a
        // second request is refused with the restart honesty.
        let first = request_leave_by_op();
        assert!(first.contains("leaving"));
        assert!(crate::voice::status_line().contains("left by operator"));
        assert!(request_leave_by_op().contains("restart"));
        reset_rt();
    }

    #[test]
    fn request_leave_when_lane_never_started_is_the_restart_honesty() {
        let _g = RT_SEQUENCER.lock().unwrap();
        reset_rt();
        assert!(request_leave_by_op().contains("not running"));
    }

    #[test]
    fn defaults_say_off_and_google_default_model() {
        let d = VoiceConfig::default();
        assert!(!d.enabled);
        assert_eq!(d.model, "gemini-3.8-live");
        assert_eq!(d.session_max_minutes, 30);
        assert!(!d.runnable());
    }

    #[test]
    fn runnable_needs_ids_key_and_enabled() {
        let mut c = cfg();
        assert!(c.runnable());
        c.enabled = false;
        assert!(!c.runnable());
        c.enabled = true;
        c.gemini_api_key = " ".to_string();
        assert!(!c.runnable(), "blank key is not configured");
        c.gemini_api_key = "K".into();
        c.channel_id = None;
        assert!(!c.runnable());
    }

    #[test]
    fn clamp_binds_session_minutes() {
        let mut c = VoiceConfig {
            session_max_minutes: 1,
            ..VoiceConfig::default()
        };
        c.clamp();
        assert_eq!(c.session_max_minutes, 5);
        c.session_max_minutes = 9_999;
        c.clamp();
        assert_eq!(c.session_max_minutes, 120);
    }

    #[test]
    fn ws_url_shape_is_honest() {
        let u = live_ws_url(&cfg(), "gemini-3.8-live");
        assert!(u.starts_with("wss://generativelanguage.googleapis.com/ws/"));
        assert!(u.contains("v1beta.GenerativeService.BidiGenerateContent"));
        // The key rides as a query param; the model does NOT — the
        // server rejects `?model=` outright (setup-message bound).
        assert!(u.ends_with("?key=TESTKEY"), "{u}");
        assert!(!u.contains("model="), "{u}");
    }

    #[test]
    fn setup_message_shape_holds() {
        let m = setup_message(&cfg(), "gemini-3.8-live", Some("be terse"));
        let s = &m["setup"];
        assert_eq!(s["model"], "models/gemini-3.8-live");
        assert_eq!(s["generationConfig"]["responseModalities"][0], "AUDIO");
        assert_eq!(
            s["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"],
            "Charon"
        );
        assert_eq!(s["systemInstruction"]["parts"][0]["text"], "be terse");
        assert!(s.get("inputAudioTranscription").is_some());
        assert!(s.get("outputAudioTranscription").is_some());
    }

    #[test]
    fn stt_setup_shape_holds() {
        let m = setup_message_stt(&cfg(), "gemini-3.5-transcribe-live");
        let s = &m["setup"];
        assert_eq!(s["model"], "models/gemini-3.5-transcribe-live");
        assert_eq!(s["generationConfig"]["responseModalities"][0], "TEXT");
        // diarization is rejected by the bidi endpoint (code 1007) —
        // the pin stays on the honest accepted shape.
        assert!(s["inputAudioTranscription"].is_object());
    }

    #[test]
    fn downmix_folds_3_and_bounds_hold() {
        // The fold is PER-3-SAMPLES (a cheap LP), no frame awareness.
        let s: Vec<i16> = vec![0, 1, 2, 3, 4, 5];
        let d = downmix_to_16k(&s);
        assert_eq!(d.len(), 2);
        assert_eq!(d[0], 1, "avg(0,1,2)");
        assert_eq!(d[1], 4, "avg(3,4,5)");
    }

    #[test]
    fn upsample_interpolates_and_dups_channels() {
        let mono: Vec<i16> = vec![100, 200];
        assert_eq!(
            upsample_to_48k_stereo(&mono),
            vec![100, 100, 150, 150, 200, 200]
        );
    }

    #[tokio::test]
    async fn play_ring_serves_a_stereo_pair_per_pair() {
        let ring = Arc::new(PlayRing::default());
        // One stereo PAIR pushed (interleaved L=100, R=-100) → exactly
        // one 8-byte f32 pair back. Interleaved data stays interleaved.
        ring.push_stereo48k(&[100, -100]);
        let mut src = RingSource(ring.clone());
        let mut buf = [0u8; 16]; // space for 2 pairs, only 1 exists
        // [u8; 16] is Copy — the closure must RETURN the written buffer,
        // or the outer asserts see the untouched zero copy (proven live).
        let (n, buf) = tokio::task::spawn_blocking(move || {
            let n = src.read(&mut buf).unwrap();
            (n, buf)
        })
        .await
        .unwrap();
        assert_eq!(n, 8, "one interleaved pair → 8 bytes (L+R f32)");
        // 100/32768 is a power-of-two division — exactly representable.
        let l = f32::from_le_bytes(buf[0..4].try_into().unwrap());
        let r = f32::from_le_bytes(buf[4..8].try_into().unwrap());
        assert_eq!(l, 100.0 / 32768.0, "L must be first (got {l}, {r})");
        assert_eq!(r, -100.0 / 32768.0, "R must be second (got {l}, {r})");
    }

    /// Live harness probe (2026-10-21 debugging): tungstenite itself,
    /// unfiltered, against the real endpoint reading the parked key.
    /// Two legs: (a) setup only, (b) setup then INSTANT audio spam — the
    /// bot's exact shape (python's client receives setupComplete in both
    /// shapes; this tells us whether tungstenite or the streaming order
    /// is the discriminator). Run with `-- --ignored --nocapture`.
    #[tokio::test]
    #[ignore = "host harness: parked bot.toml key + network access required"]
    async fn live_ws_probe_setup_complete_arrives() {
        let toml_txt = std::fs::read_to_string(std::path::PathBuf::from(
            "/home/noodly/.jerith-mesh/bot.toml",
        ))
        .expect("bot.toml");
        let mut key = String::new();
        for line in toml_txt.lines() {
            let t = line.trim();
            if let Some(v) = t.strip_prefix("gemini_api_key") {
                let v = v.trim_start().strip_prefix('=').unwrap().trim();
                key = v.trim_matches('"').to_string();
            }
        }
        assert!(!key.is_empty(), "no parked gemini key found");
        let vcfg = VoiceConfig {
            gemini_api_key: key,
            ..Default::default()
        };

        for (tag, spam) in [("setup-only", false), ("setup+instant-audio", true)] {
            let url = live_ws_url(&vcfg, "gemini-3.8-live");
            let (ws, _) = tokio_tungstenite::connect_async(url).await.unwrap();
            let (mut w, mut r) = ws.split();
            let setup = setup_message(&vcfg, "gemini-3.8-live", None);
            w.send(WsMsg::Text(setup.to_string().into())).await.unwrap();
            if spam {
                // The bot's exact pump shape: a silent 16k PCM quantum
                // streams out immediately after setup, before any
                // setupComplete could arrive.
                let msg = json!({
                    "realtimeInput": {
                        "audio": {
                            "mimeType": "audio/pcm;rate=16000",
                            "data": base64::engine::general_purpose::STANDARD
                                .encode([0u8; 2560])
                        }
                    }
                });
                w.send(WsMsg::Text(msg.to_string().into())).await.unwrap();
            }
            for i in 0..3 {
                match tokio::time::timeout(std::time::Duration::from_secs(10), r.next()).await {
                    Ok(Some(Ok(frame))) => println!("[{tag}] frame {i}: {frame:?}"),
                    Ok(Some(Err(e))) => println!("[{tag}] err: {e}"),
                    Ok(None) => {
                        println!("[{tag}] stream closed");
                        break;
                    }
                    Err(_) => {
                        println!("[{tag}] recv timeout 10s");
                        break;
                    }
                }
            }
        }
    }
}
