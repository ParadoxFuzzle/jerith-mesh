//! Audio attachment lane (FEATURES §21): plain audio files (mp3, ogg,
//! m4a, wav, flac, voice notes…) get *transcribed*, not binary-garbled —
//! the same honesty-first shape as the video lane (video.rs), whose
//! machinery this lane deliberately shares (probe, run_argv, transcribe,
//! Scratch) but with its own economics: no frames, no vision tier — one
//! whisper run per file and a compact block.
//!
//! Pipeline for one audio attachment:
//! 1. **Download** — the shared capped downloader.
//! 2. **Probe** — `ffprobe` honesty: no audio stream (a "sound" file
//!    that isn't sound), no duration (live/fragmented), or duration
//!    over the cap → honest refusals. A file whose video stream is a
//!    REAL codec (not cover-art mjpeg/png) is redirected to the video
//!    lane with an honest line — an mp4 movie wearing a .mp3 name
//!    doesn't get a fake transcript.
//! 3. **Transcribe** — whisper directly on the original file (its
//!    internal ffmpeg decodes mp3/ogg/m4a/flac/wav). Missing tool /
//!    timeout → honest lines; an empty transcript says so (a pure tone
//!    transcribes to nothing — "track present but transcript empty").
//! 4. **Compose** — duration + transcript (or honest absence), cached
//!    in the extraction cache so a re-ask never re-runs whisper.
//!
//! Embedded cover art (mjpeg/png video streams inside mp3s/flacs) is
//! expected and ignored — never mistaken for a video misfile.

use std::path::Path;

/// `[audio]` in bot.toml — see the module doc (audio.rs).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct AudioConfig {
    /// Master switch: `enabled = false` degrades every audio attachment
    /// to an honest "(audio lane disabled by config)" line. Default on.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// ffprobe / whisper binaries (PATH-resolved).
    #[serde(default = "default_ffprobe_bin")]
    pub ffprobe_bin: String,
    #[serde(default = "default_whisper_bin")]
    pub whisper_bin: String,
    /// Whisper model (default `tiny` — the video lane's tradeoff).
    #[serde(default = "default_whisper_model")]
    pub whisper_model: String,
    /// Audio longer than this (seconds) refused honestly (clamp
    /// 30–3600, default 600).
    #[serde(default = "default_max_duration_secs")]
    pub max_duration_secs: u64,
    /// Download cap for one file in MB (clamp 1–2048, default 512).
    #[serde(default = "default_max_file_mb")]
    pub max_file_mb: usize,
    /// Whisper wall-time budget in seconds (clamp 30–1800, default 600).
    #[serde(default = "default_whisper_timeout_secs")]
    pub whisper_timeout_secs: u64,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            ffprobe_bin: "ffprobe".to_string(),
            whisper_bin: "whisper".to_string(),
            whisper_model: "tiny".to_string(),
            max_duration_secs: DEFAULT_MAX_DURATION_SECS,
            max_file_mb: DEFAULT_MAX_FILE_MB as usize,
            whisper_timeout_secs: DEFAULT_WHISPER_TIMEOUT_SECS,
        }
    }
}

pub const DEFAULT_MAX_DURATION_SECS: u64 = 600;
pub const DEFAULT_MAX_FILE_MB: u64 = 512;
pub const DEFAULT_WHISPER_TIMEOUT_SECS: u64 = 600;

fn default_true() -> bool {
    true
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
fn default_max_duration_secs() -> u64 {
    DEFAULT_MAX_DURATION_SECS
}
fn default_max_file_mb() -> usize {
    DEFAULT_MAX_FILE_MB as usize
}
fn default_whisper_timeout_secs() -> u64 {
    DEFAULT_WHISPER_TIMEOUT_SECS
}

impl AudioConfig {
    /// Clamp operator mistakes at load time (the sandbox pattern).
    pub fn clamp(&mut self) {
        self.max_duration_secs = self.max_duration_secs.clamp(30, 3600);
        self.max_file_mb = self.max_file_mb.clamp(1, 2048);
        self.whisper_timeout_secs = self.whisper_timeout_secs.clamp(30, 1800);
    }
}

/// Classify an attachment toward the audio lane: an audio/ MIME wins,
/// an extension guess covers mime-less forwards. `.ogg`/`.opus` are
/// voice-note country — they land HERE, not the video lane; a rare
/// actual ogg VIDEO is caught by the probe's video-stream check and
/// redirected honestly. A lying extension is corrected by the probe,
/// not believed.
pub fn is_audio(label: &str, content_type: Option<&str>) -> bool {
    if content_type.is_some_and(|t| t.trim().starts_with("audio/")) {
        return true;
    }
    let ext = label.rsplit_once('.').map(|(_, e)| e).unwrap_or(label);
    [
        "mp3", "wav", "ogg", "oga", "opus", "m4a", "flac", "aac", "wma",
    ]
    .iter()
    .any(|e| ext.eq_ignore_ascii_case(e))
}

/// Cover-art video codec names found INSIDE audio containers (mp3
/// album art is an mjpeg/png stream — expected, never a misfile).
fn is_cover_art_codec(codec: &str) -> bool {
    codec == "mjpeg" || codec == "png" || codec == "gif" || codec == "bmp"
}

/// Does the probe's stream list show video ONLY as cover-art codecs?
fn probe_is_cover_art_video_only(probe_json: &str) -> bool {
    let v =
        serde_json::from_str::<serde_json::Value>(probe_json).unwrap_or(serde_json::Value::Null);
    v.get("streams")
        .and_then(serde_json::Value::as_array)
        .map(|streams| {
            streams
                .iter()
                .filter(|s| {
                    s.get("codec_type").and_then(serde_json::Value::as_str) == Some("video")
                })
                .all(|s| {
                    s.get("codec_name")
                        .and_then(serde_json::Value::as_str)
                        .map(is_cover_art_codec)
                        .unwrap_or(false)
                })
        })
        .unwrap_or(false)
}

/// `$sec → m:ss` for block lines.
fn fmt_ts(sec: f64) -> String {
    let s = sec.max(0.0).round() as u64;
    format!("{}:{:02}", s / 60, s % 60)
}

/// Full lane for one audio attachment, end to end (download → probe →
/// whisper → compose). `Ok` = composed multi-line block for
/// `attachments_context`; `Err` = honest failure line. Neither is ever
/// a silent drop.
pub async fn audio_context(
    cfg: &crate::config::BotConfig,
    label: &str,
    url: &str,
) -> Result<String, String> {
    let acfg = &cfg.audio;
    if !acfg.enabled {
        return Err("audio lane disabled by config ([audio] enabled = false)".to_string());
    }
    let scratch = crate::video::Scratch::new().map_err(|e| format!("{e:#}"))?;
    let input = scratch.0.join("input.bin");

    let cap_bytes = acfg.max_file_mb.saturating_mul(1024 * 1024);
    crate::sandbox::download_to(&scratch.0, "input.bin".into(), url, cap_bytes)
        .await
        .map_err(|e| format!("audio download failed: {e}"))?;

    run_lane(cfg, label, &input).await
}

/// The pipeline proper over an already-staged input file (also the
/// live harness's entry point — the harness generates its own file).
async fn run_lane(
    cfg: &crate::config::BotConfig,
    label: &str,
    input: &Path,
) -> Result<String, String> {
    let _ = label; // block stays terse; the label lives on the walk line
    let acfg = &cfg.audio;

    // Probe: the same streams+duration JSON honesty as the video lane.
    // The probe binary comes from THIS lane's config, not the video
    // lane's, so an operator can point the lanes at different days of
    // the toolchain without one silently dragging the other along.
    let probe_json = crate::video::ffprobe_json(&acfg_probe_vcfg(acfg), input)
        .await
        .map_err(|e| format!("ffprobe: {e:#}"))?;
    let probe = crate::video::parse_probe(&probe_json);
    if !probe.has_audio {
        return Err("no audio stream (this file is not sound)".to_string());
    }
    let Some(duration) = probe.duration else {
        return Err(
            "no duration metadata (live/fragmented stream) — the lane cannot bound the work"
                .to_string(),
        );
    };
    if duration > acfg.max_duration_secs as f64 {
        return Err(format!(
            "audio is {} long, cap {} — raise [audio] max_duration_secs if it must be read",
            fmt_ts(duration),
            fmt_ts(acfg.max_duration_secs as f64)
        ));
    }

    // A REAL video stream (not cover art) means this is a movie wearing
    // an audio extension — the video lane does it justice, say so.
    if probe.has_video && !probe_is_cover_art_video_only(&probe_json) {
        return Err(format!(
            "this file carries a real video stream (duration {}) — re-send as a video so the video lane (frames + transcript) can read it",
            fmt_ts(duration)
        ));
    }

    // Whisper directly on the original file — its internal ffmpeg
    // decodes mp3/ogg/m4a/flac/wav; no pre-strip needed.
    let transcript = crate::video::transcribe(
        &acfg.whisper_bin,
        &acfg.whisper_model,
        acfg.whisper_timeout_secs,
        input,
        &input.parent().map(Path::to_path_buf).unwrap_or_default(),
    )
    .await;

    let mut block = format!("audio: duration {}:", fmt_ts(duration));
    match transcript {
        Ok(t) if t.trim().is_empty() => block.push_str(
            "\ntranscript: (audio track present but transcript came back empty — speech may be absent)",
        ),
        Ok(t) => {
            block.push_str("\ntranscript:\n");
            block.push_str(t.trim());
        }
        Err(e) => block.push_str(&format!("\ntranscript: {e:#}")),
    }
    Ok(block)
}

/// The shared ffprobe runner reads its binary off a VideoConfig; an
/// AsyncConfig-only lane borrows the shape (one AudioConfig→VideoConfig
/// translation for the probe call — the FIELDS the probe uses are the
/// bin path only, so the translation is constant-safe and honest).
fn acfg_probe_vcfg(acfg: &AudioConfig) -> crate::video::VideoConfig {
    crate::video::VideoConfig {
        ffprobe_bin: acfg.ffprobe_bin.clone(),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_audio_matches_mime_and_extensions() {
        assert!(is_audio("song.mp3", None));
        assert!(is_audio("voice.ogg", Some("")));
        assert!(is_audio("no-ext", Some("audio/x-m4a")));
        assert!(is_audio("recording.opus", None));
        // mp4s route to the video lane, never the audio one.
        assert!(!is_audio("clip.mp4", Some("video/mp4")));
        assert!(!is_audio("pic.png", Some("image/png")));
        assert!(!is_audio("notes.txt", None));
    }

    #[test]
    fn cover_art_codec_classification() {
        assert!(is_cover_art_codec("mjpeg"));
        assert!(is_cover_art_codec("png"));
        assert!(is_cover_art_codec("gif"));
        assert!(!is_cover_art_codec("h264"));
        assert!(!is_cover_art_codec("hevc"));
    }

    #[test]
    fn cover_art_stream_is_not_a_video_misfile() {
        // An mp3 with embedded album art: one mjpeg video stream.
        let json = r#"{"streams":[{"codec_type":"audio"},{"codec_type":"video","codec_name":"mjpeg"}],"format":{"duration":"180.5"}}"#;
        assert!(probe_is_cover_art_video_only(json));
        // Real video: h264 inside an "mp3" → redirect to the video lane.
        let json = r#"{"streams":[{"codec_type":"audio"},{"codec_type":"video","codec_name":"h264"}],"format":{"duration":"30"}}"#;
        assert!(!probe_is_cover_art_video_only(json));
        // Garbage degrades to "not cover art" (conservative — refuses
        // rather than fabricating an audio read).
        assert!(!probe_is_cover_art_video_only("not json"));
        // No video streams at all → trivially cover-art-only.
        let json = r#"{"streams":[{"codec_type":"audio"}],"format":{"duration":"5"}}"#;
        assert!(probe_is_cover_art_video_only(json));
    }

    #[test]
    fn config_defaults_and_clamps() {
        let mut c = AudioConfig::default();
        assert!(c.enabled);
        assert_eq!(c.max_duration_secs, 600);
        assert_eq!(c.max_file_mb, 512);
        assert_eq!(c.whisper_model, "tiny");
        assert_eq!(c.whisper_timeout_secs, 600);
        c.max_duration_secs = 1;
        c.max_file_mb = 999_999;
        c.whisper_timeout_secs = 0;
        c.clamp();
        assert_eq!(c.max_duration_secs, 30);
        assert_eq!(c.max_file_mb, 2048);
        assert_eq!(c.whisper_timeout_secs, 30);
    }

    /// Live harness: a REAL 6-second mp3 (sine tone) through the whole
    /// lane with the host toolchain + the REAL whisper. A tone is not
    /// speech, so the honest expectation is the emptiness line — pin
    /// exactly that, plus no failure shapes. Run with
    /// `cargo test -p <bot> audio -- --ignored`.
    #[tokio::test]
    #[ignore = "host toolchain harness: ffmpeg + ffprobe + whisper (tiny cached)"]
    async fn live_audio_lane_on_a_real_generated_tone() {
        let mut cfg =
            crate::config::BotConfig::from_parts("http://127.0.0.1:1", "http://127.0.0.1:1");
        cfg.audio.whisper_model = "tiny".to_string();

        let clipdir = crate::video::Scratch::new().unwrap();
        let tone = clipdir.0.join("tone.mp3");
        let status = tokio::process::Command::new("ffmpeg")
            .args([
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=330:duration=6",
                "-c:a",
                "libmp3lame",
                "-b:a",
                "64k",
            ])
            .arg(&tone)
            .output()
            .await
            .unwrap();
        assert!(
            status.status.success(),
            "tone generation failed: {}",
            crate::video::last_stderr_lines(&String::from_utf8_lossy(&status.stderr), 3)
        );

        let block = run_lane(&cfg, "tone.mp3", &tone).await.expect(
            "the lane should succeed on a healthy mp3 — check ffmpeg/ffprobe/whisper are on PATH",
        );
        assert!(block.contains("audio: duration 0:06"), "{block}");
        assert!(block.contains("transcript"), "{block}");
        // The toolchain works, so NO failure shapes belong in the block;
        // a tone transcribes to the honest emptiness line exactly.
        assert!(!block.contains("failed"), "{block}");
        assert!(
            block.contains("(audio track present but transcript came back empty"),
            "a pure tone should transcribe empty-honest: {block}"
        );
    }
}
