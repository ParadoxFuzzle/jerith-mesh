//! Video attachment lane (FEATURES §19): uploaded videos get *understood*,
//! not just relayed, using the same honesty-first shape as the image and
//! sandbox lanes.
//!
//! Pipeline for one video attachment:
//! 1. **Download** — the same capped downloader the sandbox lane uses.
//! 2. **Probe** — `ffprobe` reads codec streams + duration. No video
//!    stream (an audio-only file riding a video container), no duration
//!    (live/fragmented stream), or duration over the cap → an honest
//!    refusal, never a guess. This is also where an mp3 wearing a .mp4
//!    extension gets bounced to an honest line instead of a fake read.
//! 3. **Audio strip** — when the clip has an audio track, `ffmpeg`
//!    extracts 16 kHz mono WAV into the scratch dir.
//! 4. **Transcribe** — `openai-whisper` (the host CLI) turns the WAV
//!    into text, entirely locally; the transcript only ever lives inside
//!    the composed context block. Missing toolchain / timeout → honest
//!    line. A clip with no audio track says so as a story line, not an
//!    error; an empty transcript says so too.
//! 5. **Frame sampling** — timestamps spaced `frame_stride_secs` apart
//!    (default 5 s), capped at `max_frames`; a clip too long for the
//!    cap gets its timestamps evenly *resampled across the whole
//!    duration* (understanding the arc of a long video beats seeing its
//!    first 40 s in detail), and the block says so.
//! 6. **Per-frame vision** — each sampled frame is extracted with
//!    `ffmpeg`, base64'd into a `data:image/jpeg` payload and described
//!    through the gateway's LOCAL vision tier — the same lane images
//!    already ride, so nothing frame-shaped leaves the node.
//! 7. **Compose** — one multi-line Ok(String) for `attachments_context`:
//!    duration, sampling note, transcript (or honest absence), then
//!    per-timestamp frame notes. Cached in the sandbox-extraction cache
//!    (same id/name+size keying), so a re-ask never re-runs whisper.
//!
//! Deliberately: no model watches the video end-to-end (frames +
//! transcript already answer "what's happening"); scratch files are
//! wiped by a Drop guard; every subprocess is timeout-bounded and
//! kill-on-drop; the ffmpeg/ffprobe/whisper argv are pure, grep-able,
//! unit-tested builders (the sandbox's `docker_argv` pattern); and a
//! probed-ok result whose frames all fail refuses to fabricate
//! understanding.

use anyhow::{Context as _, Result};
use std::path::{Path, PathBuf};

/// `[video]` in bot.toml — see the module doc (video.rs).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct VideoConfig {
    /// Master switch: `enabled = false` degrades every video attachment
    /// to an honest "(video lane disabled by config)" line. Default on.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// ffmpeg / ffprobe / whisper binaries (PATH-resolved; the openai-
    /// whisper CLI installs `whisper`).
    #[serde(default = "default_ffmpeg_bin")]
    pub ffmpeg_bin: String,
    #[serde(default = "default_ffprobe_bin")]
    pub ffprobe_bin: String,
    #[serde(default = "default_whisper_bin")]
    pub whisper_bin: String,
    /// Whisper model (tiny/base/small/…). Default `tiny` — the cheapest
    /// tier that still frankly understands speech; `base` costs ~2× the
    /// wall time for noticeably better transcripts.
    #[serde(default = "default_whisper_model")]
    pub whisper_model: String,
    /// Seconds between sampled frames (clamp 1–30, default 5).
    #[serde(default = "default_frame_stride_secs")]
    pub frame_stride_secs: u64,
    /// Max frames analyzed per video (clamp 1–16, default 8).
    #[serde(default = "default_max_frames")]
    pub max_frames: usize,
    /// Videos longer than this (seconds) are refused honestly;
    /// transcription is the rate limiter (clamp 30–3600, default 600).
    #[serde(default = "default_max_duration_secs")]
    pub max_duration_secs: u64,
    /// Download cap for one video in MB (clamp 1–2048, default 512).
    #[serde(default = "default_max_file_mb")]
    pub max_file_mb: usize,
    /// Whisper wall-time budget in seconds (clamp 30–1800, default 600).
    #[serde(default = "default_whisper_timeout_secs")]
    pub whisper_timeout_secs: u64,
}

impl Default for VideoConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            ffmpeg_bin: "ffmpeg".to_string(),
            ffprobe_bin: "ffprobe".to_string(),
            whisper_bin: "whisper".to_string(),
            whisper_model: "tiny".to_string(),
            frame_stride_secs: DEFAULT_FRAME_STRIDE_SECS,
            max_frames: DEFAULT_MAX_FRAMES as usize,
            max_duration_secs: DEFAULT_MAX_DURATION_SECS,
            max_file_mb: DEFAULT_MAX_FILE_MB as usize,
            whisper_timeout_secs: DEFAULT_WHISPER_TIMEOUT_SECS,
        }
    }
}

pub const DEFAULT_FRAME_STRIDE_SECS: u64 = 5;
pub const DEFAULT_MAX_FRAMES: u64 = 8;
pub const DEFAULT_MAX_DURATION_SECS: u64 = 600;
pub const DEFAULT_MAX_FILE_MB: u64 = 512;
pub const DEFAULT_WHISPER_TIMEOUT_SECS: u64 = 600;

fn default_true() -> bool {
    true
}
fn default_ffmpeg_bin() -> String {
    "ffmpeg".to_string()
}
fn default_ffprobe_bin() -> String {
    "ffprobe".to_string()
}
fn default_whisper_bin() -> String {
    "whisper".to_string()
}
fn default_whisper_model() -> String {
    "tiny".to_string()
}
fn default_frame_stride_secs() -> u64 {
    DEFAULT_FRAME_STRIDE_SECS
}
fn default_max_frames() -> usize {
    DEFAULT_MAX_FRAMES as usize
}
fn default_max_duration_secs() -> u64 {
    DEFAULT_MAX_DURATION_SECS
}
fn default_max_file_mb() -> usize {
    DEFAULT_MAX_FILE_MB as usize
}
fn default_whisper_timeout_secs() -> u64 {
    DEFAULT_WHISPER_TIMEOUT_SECS
}

impl VideoConfig {
    /// Clamp operator mistakes at load time (the sandbox pattern).
    pub fn clamp(&mut self) {
        self.frame_stride_secs = self.frame_stride_secs.clamp(1, 30);
        self.max_frames = self.max_frames.clamp(1, 16);
        self.max_duration_secs = self.max_duration_secs.clamp(30, 3600);
        self.max_file_mb = self.max_file_mb.clamp(1, 2048);
        self.whisper_timeout_secs = self.whisper_timeout_secs.clamp(30, 1800);
    }
}

/// Stable face for the pure helpers without the whole bot config.
#[derive(Debug, Clone, Copy)]
pub struct LaneCaps {
    pub stride_secs: u64,
    pub max_frames: usize,
    pub max_duration_secs: u64,
}

fn lane_caps(vcfg: &VideoConfig) -> LaneCaps {
    LaneCaps {
        stride_secs: vcfg.frame_stride_secs,
        max_frames: vcfg.max_frames,
        max_duration_secs: vcfg.max_duration_secs,
    }
}

/// Shared classification for every attachment site (Discord @mention/
/// DM fallback + /read): image / video / other, ONE place. A lying
/// extension is corrected later by the probe (audio-only container →
/// honest refusal — no fake understanding).
pub fn classify_kind(label: &str, content_type: Option<&str>) -> crate::converse::AttKind {
    if content_type.is_some_and(|t| t.trim().starts_with("image/")) {
        crate::converse::AttKind::Image
    } else if is_video(label, content_type) {
        crate::converse::AttKind::Video
    } else {
        crate::converse::AttKind::Other
    }
}

/// Classify an attachment toward the video lane: a video/ MIME wins, an
/// extension guess covers mime-less forwarded files. A lying extension
/// is corrected later by the probe (audio-only container → honest
/// refusal — no fake understanding).
pub fn is_video(label: &str, content_type: Option<&str>) -> bool {
    if content_type.is_some_and(|t| t.trim().starts_with("video/")) {
        return true;
    }
    let ext = label.rsplit_once('.').map(|(_, e)| e).unwrap_or(label);
    [
        "mp4", "m4v", "mkv", "webm", "mov", "avi", "mpg", "mpeg", "wmv", "flv", "ogv", "3gp", "ts",
    ]
    .iter()
    .any(|e| ext.eq_ignore_ascii_case(e))
}

/// Frame sample plan: evenly spaced mid-frame timestamps, capped by
/// `max_frames`. `None`/non-finite/non-positive durations degrade to
/// one frame at 0.0 (the real lane refuses absent durations before
/// this; those shapes feed the unit tests). Pure.
pub fn frame_timestamps(duration: Option<f64>, caps: LaneCaps) -> Vec<f64> {
    let usable = duration.filter(|d| d.is_finite() && *d > 0.0);
    let n = match usable {
        Some(d) => {
            let natural = (d / caps.stride_secs.max(1) as f64).ceil() as usize;
            natural.min(caps.max_frames).max(1)
        }
        None => 1,
    };
    let d = usable.unwrap_or(0.0);
    (0..n).map(|i| d * (i as f64 + 0.5) / n as f64).collect()
}

/// "Every 5 s" only holds for short clips; when duration/stride exceeds
/// the frame cap the block gets a note saying how sampling was
/// stretched (honesty beats silently front-loading a long video).
fn sampling_note(duration: f64, sampled: usize, caps: LaneCaps) -> Option<String> {
    let natural = (duration / caps.stride_secs.max(1) as f64).ceil() as usize;
    let sampled = sampled.max(1);
    if sampled < natural {
        Some(format!(
            "stride stretched: {natural} frames at {} s would exceed the {}-frame cap, so {sampled} frames were sampled evenly across the clip (≈ every {})",
            caps.stride_secs,
            caps.max_frames,
            fmt_ts(duration / sampled as f64)
        ))
    } else {
        None
    }
}

/// `sec → m:ss` for block lines.
fn fmt_ts(sec: f64) -> String {
    let s = sec.max(0.0).round() as u64;
    format!("{}:{:02}", s / 60, s % 60)
}

// ---- argv builders (pure, grep-able, unit-tested — the docker_argv pattern) ----

/// Probe argv: stream codec types + duration only, JSON on stdout.
pub fn probe_argv(bin: &str, input: &Path) -> Vec<String> {
    vec![
        bin.to_string(),
        "-v".into(),
        "error".into(),
        "-show_entries".into(),
        "stream=index,codec_type,codec_name".into(),
        "-show_entries".into(),
        "format=duration".into(),
        "-of".into(),
        "json".into(),
        input.display().to_string(),
    ]
}

/// Frame argv: fast input seek (`-ss` BEFORE `-i`), a single frame,
/// JPEG, scaled to ≤1024 px wide. An extract that fails or leaves a
/// zero-byte file is skipped with an honest frame line.
pub fn frame_argv(bin: &str, input: &Path, out: &Path, ts: f64) -> Vec<String> {
    vec![
        bin.to_string(),
        "-y".into(),
        "-v".into(),
        "error".into(),
        "-ss".into(),
        format!("{ts:.3}"),
        "-i".into(),
        input.display().to_string(),
        "-frames:v".into(),
        "1".into(),
        "-vf".into(),
        "scale=min(1024\\,iw):-2".into(),
        "-f".into(),
        "image2".into(),
        out.display().to_string(),
    ]
}

/// Audio-strip argv: 16 kHz mono WAV (whisper's happy shape).
pub fn audio_argv(bin: &str, input: &Path, out: &Path) -> Vec<String> {
    vec![
        bin.to_string(),
        "-y".into(),
        "-v".into(),
        "error".into(),
        "-i".into(),
        input.display().to_string(),
        "-vn".into(),
        "-ac".into(),
        "1".into(),
        "-ar".into(),
        "16000".into(),
        "-f".into(),
        "wav".into(),
        out.display().to_string(),
    ]
}

/// Whisper argv: transcribe one WAV to `<stem>.txt` inside `out_dir`.
pub fn whisper_argv(bin: &str, wav: &Path, out_dir: &Path, model: &str) -> Vec<String> {
    vec![
        bin.to_string(),
        wav.display().to_string(),
        "--model".into(),
        model.to_string(),
        "--task".into(),
        "transcribe".into(),
        "--output_format".into(),
        "txt".into(),
        "--output_dir".into(),
        out_dir.display().to_string(),
    ]
}

// ---- the lane ----

/// Run one argv with a wall-time cap. `tokio::time::timeout` cancels
/// the wait future on elapse, which drops the child (kill_on_drop) —
/// a hung ffmpeg/whisper is killed, not leaked. `what` names the stage
/// honestly in the error.
async fn run_argv(argv: &[String], timeout: std::time::Duration, what: &str) -> Result<()> {
    let started = std::time::Instant::now();
    let out = tokio::time::timeout(
        timeout,
        tokio::process::Command::new(&argv[0])
            .args(&argv[1..])
            .stderr(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .kill_on_drop(true)
            .output(),
    )
    .await;
    let dt = started.elapsed().as_secs_f32();
    let out = match out {
        Ok(Ok(o)) => o,
        Ok(Err(e)) if e.kind() == std::io::ErrorKind::NotFound => anyhow::bail!(
            "missing tool: no `{}` on PATH — set the [video] bins in bot.toml",
            argv[0]
        ),
        Ok(Err(e)) => return Err(e).context(what.to_string()),
        Err(_) => anyhow::bail!(
            "{} timed out after {:.1}s (budget {}s) — killed",
            what,
            dt,
            timeout.as_secs()
        ),
    };
    let stderr = String::from_utf8_lossy(&out.stderr);
    anyhow::ensure!(
        out.status.success(),
        "{} failed after {dt:.1}s: {}",
        what,
        last_stderr_lines(&stderr, 2)
    );
    Ok(())
}

fn last_stderr_lines(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.is_empty() {
        "(no stderr)".into()
    } else {
        let start = lines.len().saturating_sub(n);
        lines[start..].join(" | ")
    }
}

/// Wipe-on-drop scratch dir: video bytes, wav and frames live far too
/// briefly to be left lying around (privacy hygiene + disk).
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Result<Self> {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "jerith-vid-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).context("video scratch dir create failed")?;
        Ok(Self(dir))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Probe outcome, honestly split.
#[derive(Debug)]
struct Probe {
    has_video: bool,
    has_audio: bool,
    duration: Option<f64>,
}

fn parse_probe(body: &str) -> Probe {
    let v = serde_json::from_str::<serde_json::Value>(body).unwrap_or(serde_json::Value::Null);
    let streams = v
        .get("streams")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    Probe {
        has_video: streams
            .iter()
            .any(|s| s.get("codec_type").and_then(serde_json::Value::as_str) == Some("video")),
        has_audio: streams
            .iter()
            .any(|s| s.get("codec_type").and_then(serde_json::Value::as_str) == Some("audio")),
        duration: v
            .pointer("/format/duration")
            .and_then(serde_json::Value::as_str)
            .and_then(|d| d.parse::<f64>().ok()),
    }
}

/// Transcribe one WAV through whisper; returns the trimmed transcript.
async fn transcribe(vcfg: &VideoConfig, wav: &Path, dir: &Path) -> Result<String> {
    run_argv(
        &whisper_argv(&vcfg.whisper_bin, wav, dir, &vcfg.whisper_model),
        std::time::Duration::from_secs(vcfg.whisper_timeout_secs),
        "whisper",
    )
    .await?;
    let txt = dir.join("audio.txt");
    std::fs::read_to_string(&txt)
        .map(|t| t.trim().to_string())
        .with_context(|| format!("whisper produced no txt at {}", txt.display()))
}

/// One frame through the image lane's own gateway skill: the local JPEG
/// becomes a `data:image/jpeg` payload (the same shape
/// `fetch_vision_image` returns) — no re-fetch, no mesh exposure.
async fn describe_frame(
    cfg: &crate::config::BotConfig,
    jpeg: &Path,
    label: &str,
) -> Result<String> {
    use base64::Engine as _;
    let bytes = tokio::fs::read(jpeg).await.context("frame read")?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    let data_url = format!("data:image/jpeg;base64,{b64}");
    let prompt = format!(
        "Frame from a video titled \"{label}\". Describe concisely what is visible and what it suggests about what is happening here. Mention any readable text."
    );
    crate::api::vision_describe_skill(cfg, &prompt, &data_url).await
}

/// Full lane for one video attachment, end to end (download → probe →
/// audio → transcript → frames → vision → compose). `Ok` = composed
/// multi-line block for `attachments_context`; `Err` = honest failure
/// line. Both land in the walk visibly; neither is ever a silent drop.
pub async fn video_context(
    cfg: &crate::config::BotConfig,
    label: &str,
    url: &str,
) -> Result<String, String> {
    let vcfg = &cfg.video;
    if !vcfg.enabled {
        return Err("video lane disabled by config ([video] enabled = false)".to_string());
    }
    let scratch = Scratch::new().map_err(|e| format!("{e:#}"))?;
    let input = scratch.0.join("input.bin");
    let cap_bytes = vcfg.max_file_mb.saturating_mul(1024 * 1024);
    crate::sandbox::download_to(&scratch.0, "input.bin".into(), url, cap_bytes)
        .await
        .map_err(|e| format!("video download failed: {e}"))?;
    // scratch outlives the await: lane products live until compose.
    run_lane(cfg, label, &input).await
}

/// The pipeline proper over an already-staged input file (also the live
/// test's entry point: the harness generates its own clip locally).
async fn run_lane(
    cfg: &crate::config::BotConfig,
    label: &str,
    input: &Path,
) -> Result<String, String> {
    let vcfg = &cfg.video;
    let caps = lane_caps(vcfg);
    let scratch = Scratch::new().map_err(|e| format!("{e:#}"))?;
    let dir: &PathBuf = &scratch.0;

    // Probe: streams + duration with honest refusals.
    let probe_json = probe(vcfg, input)
        .await
        .map_err(|e| format!("ffprobe: {e:#}"))?;
    let probe = parse_probe(&probe_json);
    if !probe.has_video {
        return Err("no video stream (audio-only file in a video container)".to_string());
    }
    let Some(duration) = probe.duration else {
        return Err(
            "no duration metadata (live/fragmented stream) — the lane cannot bound the work"
                .to_string(),
        );
    };
    if duration > caps.max_duration_secs as f64 {
        return Err(format!(
            "video is {} long, cap {} — raise [video] max_duration_secs if it must be read",
            fmt_ts(duration),
            fmt_ts(caps.max_duration_secs as f64)
        ));
    }

    // Audio → transcript. Absence is a story line, not an error.
    let transcript: Result<String, String> = if probe.has_audio {
        let wav = dir.join("audio.wav");
        match strip_audio(vcfg, input, &wav, &caps).await {
            Ok(()) => match transcribe(vcfg, &wav, dir).await {
                Ok(t) if t.is_empty() => {
                    Err("(audio track present but transcript came back empty)".to_string())
                }
                Ok(t) => Ok(t),
                Err(e) => Err(format!("transcription failed: {e:#}")),
            },
            Err(e) => Err(format!("transcription skipped: {}", format_err(&e))),
        }
    } else {
        Err("(no audio track)".to_string())
    };

    // Frames → vision descriptions, in sampled order.
    let timestamps = frame_timestamps(probe.duration, caps);
    let note = sampling_note(duration, timestamps.len(), caps);
    let mut frame_lines = Vec::with_capacity(timestamps.len());
    let mut good_frames = 0usize;
    let mut extractable = 0usize;
    for (i, ts) in timestamps.iter().enumerate() {
        let out = dir.join(format!("frame_{i}.jpg"));
        let ok = run_argv(
            &frame_argv(&vcfg.ffmpeg_bin, input, &out, *ts),
            std::time::Duration::from_secs(30),
            "frame extract",
        )
        .await
        .is_ok()
            && out.exists()
            && out.metadata().map(|m| m.len() > 0).unwrap_or(false);
        if !ok {
            frame_lines.push(format!("[{}] (frame not extractable)", fmt_ts(*ts)));
            continue;
        }
        extractable += 1;
        match describe_frame(cfg, &out, label).await {
            Ok(desc) if !desc.trim().is_empty() => {
                good_frames += 1;
                frame_lines.push(format!("[{}] {}", fmt_ts(*ts), desc.trim()));
            }
            Ok(_) => frame_lines.push(format!("[{}] (vision described nothing)", fmt_ts(*ts))),
            Err(e) => frame_lines.push(format!("[{}] (frame analysis failed: {e:#})", fmt_ts(*ts))),
        }
    }
    if extractable == 0 {
        return Err(format!(
            "ffprobe ok (duration {}) but NO frame could be extracted — refusing to fabricate understanding",
            fmt_ts(duration)
        ));
    }

    // Compose.
    let mut block = format!(
        "video: duration {}, {} frame(s) analyzed{}:",
        fmt_ts(duration),
        good_frames,
        note.map(|n| format!(" ({n})")).unwrap_or_default()
    );
    match transcript {
        Ok(t) => {
            block.push_str("\ntranscript:\n");
            block.push_str(&t);
        }
        Err(e) => block.push_str(&format!("\ntranscript: {e}")),
    }
    if !frame_lines.is_empty() {
        block.push_str("\nframe notes:");
        for line in &frame_lines {
            block.push('\n');
            block.push_str(line);
        }
    }
    Ok(block)
}

fn format_err(e: &anyhow::Error) -> String {
    format!("{e:#}")
}

/// ffprobe with the JSON on stdout captured.
async fn probe(vcfg: &VideoConfig, input: &Path) -> Result<String> {
    let argv = probe_argv(&vcfg.ffprobe_bin, input);
    let out = tokio::process::Command::new(&argv[0])
        .args(&argv[1..])
        .stderr(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .kill_on_drop(true)
        .output()
        .await
        .context("ffprobe spawn")?;
    anyhow::ensure!(
        out.status.success(),
        "ffprobe failed: could not read the file"
    );
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

/// Audio-strip stage with its own stage name for honest errors.
async fn strip_audio(vcfg: &VideoConfig, input: &Path, wav: &Path, caps: &LaneCaps) -> Result<()> {
    run_argv(
        &audio_argv(&vcfg.ffmpeg_bin, input, wav),
        std::time::Duration::from_secs(caps.max_duration_secs.saturating_add(120)),
        "audio strip",
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps() -> LaneCaps {
        LaneCaps {
            stride_secs: 5,
            max_frames: 8,
            max_duration_secs: 600,
        }
    }

    #[test]
    fn is_video_matches_mime_and_extensions() {
        assert!(is_video("clip.mp4", None));
        assert!(is_video("clip.MKV", Some("")));
        assert!(is_video("no-ext", Some("video/webm")));
        assert!(!is_video("pic.png", Some("image/png")));
        // mp3s ride the sandbox lane, not the video one.
        assert!(!is_video("song.mp3", Some("audio/mpeg")));
        assert!(!is_video("notes.txt", None));
    }

    #[test]
    fn classify_kind_routes_all_three_lanes() {
        use crate::converse::AttKind;
        assert_eq!(classify_kind("pic.png", Some("image/png")), AttKind::Image);
        assert_eq!(classify_kind("clip.mp4", Some("video/mp4")), AttKind::Video);
        // Video by extension even when Discord supplies no mime.
        assert_eq!(classify_kind("clip.mov", None), AttKind::Video);
        assert_eq!(
            classify_kind("doc.pdf", Some("application/pdf")),
            AttKind::Other
        );
    }

    #[test]
    fn timestamps_short_clip_follows_stride() {
        // 23 s / 5 s stride → ceil = 5 frames, evenly mid-frame.
        let ts = frame_timestamps(Some(23.0), caps());
        assert_eq!(ts.len(), 5, "{ts:?}");
        assert!((ts[0] - 2.3).abs() < 1e-9, "{ts:?}");
        assert!((ts[4] - 20.7).abs() < 1e-9, "{ts:?}");
        assert!(ts.iter().all(|t| *t >= 0.0 && *t < 23.0), "{ts:?}");
    }

    #[test]
    fn timestamps_long_clip_resampled_evenly() {
        // 1 hour / 5 s stride would be 720 frames; the cap squeezes to 8
        // spread across the WHOLE clip, not the first 40 s.
        let ts = frame_timestamps(Some(3600.0), caps());
        assert_eq!(ts.len(), 8, "{ts:?}");
        assert!((ts[0] - 225.0).abs() < 1e-6, "{ts:?}");
        assert!((ts[7] - 3375.0).abs() < 1e-6, "{ts:?}");
    }

    #[test]
    fn timestamps_degrade_to_one_frame_for_absent_duration() {
        assert_eq!(frame_timestamps(None, caps()), vec![0.0]);
        assert_eq!(frame_timestamps(Some(0.0), caps()), vec![0.0]);
    }

    #[test]
    fn sampling_note_admits_the_stretch_only_when_resampled() {
        // Short clip: natural = 6 ≤ cap 8 → no note.
        assert!(sampling_note(30.0, 6, caps()).is_none());
        // 1 h: natural 720, sampled 8 → note mentions both numbers.
        let note = sampling_note(3600.0, 8, caps()).unwrap();
        assert!(note.contains("720"), "{note}");
        assert!(note.contains("frame cap"), "{note}");
    }

    #[test]
    fn probe_argv_is_exact() {
        assert_eq!(
            probe_argv("ffprobe", Path::new("/tmp/v.bin")),
            vec![
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=index,codec_type,codec_name",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                "/tmp/v.bin",
            ]
        );
    }

    #[test]
    fn frame_argv_seeks_before_input() {
        let argv = frame_argv("ffmpeg", Path::new("/v.bin"), Path::new("/f.jpg"), 12.345);
        let ss = argv.iter().position(|a| a == "-ss").unwrap();
        let ii = argv.iter().position(|a| a == "-i").unwrap();
        assert!(ss < ii, "fast input seek: -ss must precede -i: {argv:?}");
        assert!(
            argv.contains(&"scale=min(1024\\,iw):-2".to_string()),
            "{argv:?}"
        );
        assert_eq!(argv[argv.len() - 1], "/f.jpg");
    }

    #[test]
    fn audio_argv_strips_video_and_downmixes() {
        let argv = audio_argv("ffmpeg", Path::new("/v.bin"), Path::new("/a.wav"));
        assert!(argv.contains(&"-vn".to_string()), "{argv:?}");
        assert!(argv.contains(&"16000".to_string()), "{argv:?}");
        assert_eq!(argv[argv.len() - 1], "/a.wav");
    }

    #[test]
    fn whisper_argv_points_output_at_scratch_dir() {
        let argv = whisper_argv("whisper", Path::new("/a.wav"), Path::new("/scr"), "tiny");
        assert!(
            argv.contains(&"--model".to_string()) && argv.contains(&"tiny".to_string()),
            "{argv:?}"
        );
        assert!(argv.contains(&"--output_format".to_string()) && argv.contains(&"txt".to_string()));
        assert_eq!(argv[argv.len() - 1], "/scr");
    }

    #[test]
    fn probe_parse_splits_streams_and_duration() {
        let json = r#"{"streams":[{"codec_type":"video","codec_name":"h264"},{"codec_type":"audio","codec_name":"aac"}],"format":{"duration":"93.5"}}"#;
        let p = parse_probe(json);
        assert!(p.has_video && p.has_audio);
        assert_eq!(p.duration, Some(93.5));

        // Audio-only container: honest refusal upstream, parse stays ok.
        let p = parse_probe(r#"{"streams":[{"codec_type":"audio"}],"format":{"duration":"5"}}"#);
        assert!(!p.has_video && p.has_audio);

        // Live/fragmented: no duration → refusal upstream.
        let p = parse_probe(r#"{"streams":[{"codec_type":"video"}]}"#);
        assert!(p.has_video && p.duration.is_none());

        // Garbage degrades to all-absent, never a panic.
        let p = parse_probe("not json at all");
        assert!(!p.has_video && !p.has_audio && p.duration.is_none());
    }

    #[test]
    fn config_defaults_and_clamps() {
        let mut c = VideoConfig::default();
        assert!(c.enabled);
        assert_eq!(c.frame_stride_secs, 5);
        assert_eq!(c.max_frames, 8);
        assert_eq!(c.max_duration_secs, 600);
        assert_eq!(c.max_file_mb, 512);
        assert_eq!(c.whisper_model, "tiny");
        assert_eq!(c.whisper_timeout_secs, 600);
        c.frame_stride_secs = 0;
        c.max_frames = 99;
        c.max_duration_secs = 1;
        c.max_file_mb = 999_999;
        c.whisper_timeout_secs = 0;
        c.clamp();
        assert_eq!(c.frame_stride_secs, 1);
        assert_eq!(c.max_frames, 16);
        assert_eq!(c.max_duration_secs, 30);
        assert_eq!(c.max_file_mb, 2048);
        assert_eq!(c.whisper_timeout_secs, 30);
    }

    /// Live harness: builds a REAL 6-second clip (testsrc + sine audio)
    /// with the host ffmpeg, then runs the whole lane directly on it
    /// (probe → audio strip → whisper → frames → a stub gateway's vision
    /// tier). Run with `cargo test -p <bot> video -- --ignored`.
    #[tokio::test]
    #[ignore = "host toolchain harness: ffmpeg + ffprobe + whisper (and tiny model cached)"]
    async fn live_full_lane_on_a_real_generated_clip() {
        // Vision-tier stub: every frame lands on a fixed, honest line.
        let app = axum::Router::new().route(
            "/v1/skills/vision/describe",
            axum::routing::post(|| async {
                axum::Json(serde_json::json!({
                    "choices": [{"message": {"role": "assistant", "content": "test pattern frame"}}]
                }))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let mut cfg =
            crate::config::BotConfig::from_parts(&format!("http://{addr}"), "http://127.0.0.1:1");
        cfg.video.whisper_model = "tiny".to_string();

        // Generate the clip (fast, silent: short + ultrafast preset).
        let clipdir = Scratch::new().unwrap();
        let clip = clipdir.0.join("gen.mp4");
        let status = tokio::process::Command::new("ffmpeg")
            .args([
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=6:size=320x240:rate=10",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=6",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
            ])
            .arg(&clip)
            .output()
            .await
            .unwrap();
        assert!(
            status.status.success(),
            "clip generation failed: {}",
            last_stderr_lines(&String::from_utf8_lossy(&status.stderr), 3)
        );
        // THE harness assertion: the whole lane runs on a real, local
        // file — frames referenced with real timestamps, transcript and
        // vision lines present, honest shapes throughout.
        let block = run_lane(&cfg, "gen.mp4", &clip)
            .await
            .expect("lane should succeed");
        assert!(block.contains("duration 0:06"), "{block}");
        assert!(block.contains("transcript"), "{block}");
        assert!(block.contains("frame notes"), "{block}");
        assert!(
            block
                .lines()
                .any(|l| l.starts_with("[0:") && l.contains("test pattern frame")),
            "{block}"
        );
    }

    /// Deep live harness: the WHOLE production shape, no stubs — a local
    /// axum file server plays the attachment CDN (real HTTP download
    /// through `video_context`), then the full lane (probe → audio
    /// strip → whisper → frames) with descriptions landing on the NODE'S
    /// REAL vision tier (the live gateway's MiniCPM-V, :8900). Requires
    /// the host toolchain + the node up. Complements the stub harness
    /// above, which pins determinism instead of reality.
    #[tokio::test]
    #[ignore = "live harness: ffmpeg + whisper on PATH, and jerith-node's gateway live on the default :8900 with MiniCPM up"]
    async fn live_full_lane_real_download_and_real_vision_tier() {
        // A real clip (same recipe as the stub harness). A 10-frame
        // clip: stride 5 → 2 sampled frames → 2 real vision calls.
        let clipdir = Scratch::new().unwrap();
        let clip = clipdir.0.join("deep.mp4");
        let status = tokio::process::Command::new("ffmpeg")
            .args([
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=8:size=320x240:rate=5",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=220:duration=8",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
            ])
            .arg(&clip)
            .output()
            .await
            .unwrap();
        assert!(
            status.status.success(),
            "clip generation failed: {}",
            last_stderr_lines(&String::from_utf8_lossy(&status.stderr), 3)
        );

        let bytes = std::fs::read(&clip).unwrap();
        let app = axum::Router::new().route(
            "/deep.mp4",
            axum::routing::get(move || {
                let bytes = bytes.clone();
                async move {
                    axum::response::Response::builder()
                        .header("content-type", "video/mp4")
                        .body(axum::body::Body::from(bytes))
                        .unwrap()
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        // The PRODUCTION gateway (default :8900) — where real chats'
        // vision calls already land.
        let cfg = crate::config::BotConfig::from_parts(
            crate::config::DEFAULT_GATEWAY_URL,
            "http://127.0.0.1:1",
        );
        let block = video_context(
            &cfg,
            "deep.mp4",
            &format!("http://{addr}/deep.mp4"),
        )
        .await
        .expect(
            "full lane against the live node — check jerith-node is up and whisper/ffmpeg are on PATH",
        );

        assert!(block.contains("duration 0:08"), "{block}");
        // A sine tone may transcribe to nothing — either story is
        // honest; the section heading must exist either way.
        assert!(block.contains("transcript"), "{block}");
        // REAL tier output: every frame line carries actual prose (not
        // a failure marker, not a one-word stub).
        let frame_lines: Vec<&str> = block.lines().filter(|l| l.starts_with("[0:")).collect();
        assert!(!frame_lines.is_empty(), "no frame lines: {block}");
        for line in &frame_lines {
            assert!(
                !line.contains("frame analysis failed") && !line.contains("(frame not extractable"),
                "frame failed against the LIVE tier: {line}\n{block}"
            );
            assert!(
                line.len() > 40,
                "frame description too thin to be real prose: {line}"
            );
        }
    }
}
