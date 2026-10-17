//! Docker-sandboxed file reading for non-image attachments.
//!
//! The bot never opens an attachment's bytes in its own process: files (and
//! only non-image files — images ride the vision skill) are downloaded into
//! a fresh throwaway directory and handed to a container from `[sandbox]`
//! `image`, which reads them under a full safety contract:
//!
//! - `--network none` (no egress — the sandbox can't phone home)
//! - `--cap-drop ALL` + `--security-opt no-new-privileges`
//! - `--read-only` rootfs + write scratch limited to a 16 MiB noexec tmpfs
//! - uid 65534 (nobody): the staged files are world-readable, nothing more
//! - `--memory 256m --pids-limit 64 --cpus 0.5` (bounding a crafted file)
//! - one fresh volume per request, `--rm`, kill-on-drop + wall-clock timeout
//!
//! The extractor (`sandbox_extract.py`, baked into the bot binary via
//! `include_str!`) treats filenames as data and never executes anything.
//! Per-file outcomes return as honest bracketed lines the chat model can
//! quote; failures say WHY instead of dropping the file silently.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context as _, Result};
use tokio::process::Command;

use crate::config::SandboxConfig;

/// The extractor program, executed inside the container. Baked into the
/// binary (no image rebuild when it changes — the bot owns the staging
/// directory, so it carries its own extractor into every request).
pub const EXTRACT_SCRIPT: &str = include_str!("sandbox_extract.py");

/// Sandbox uid/gid for staged files and the container run: nobody.
const SANDBOX_UID_GID: &str = "65534:65534";

/// One non-image attachment contract from a transport: the URL to fetch
/// and the display label (the attachment's filename).
#[derive(Debug, Clone)]
pub struct FileAttachment {
    pub url: String,
    pub label: String,
}

/// Reduce a Discord filename to a container-safe name: basename only,
/// `A-Za-z0-9._-` kept, everything else `_`, never empty or `.`/`..`.
/// The label (display form) travels separately, so the text stays
/// human-readable even when the file had a hostile name
/// (`../../etc/passwd`, unicode spaces, a 300-char emoji parade).
pub fn safe_name(index: usize, label: &str) -> String {
    let base = label.rsplit('/').next().unwrap_or("");
    let mut out = String::new();
    for ch in base.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() || out == "." || out == ".." {
        out = "file".to_string();
    }
    format!("f{index}_{out}")
}

/// One staged file: its container-side safe name and display label.
#[derive(Debug, Clone)]
struct StagedFile {
    safe: String,
    label: String,
}

/// Read the given attachments inside the sandbox, in order. Per-file
/// results are (label, text) — `Err` carries the WHY of the failure. A
/// failed batch (docker down, staging error, timeout) surfaces one
/// honest failure per file rather than dropping the request silently.
pub async fn read_attachments(
    cfg: &SandboxConfig,
    files: &[FileAttachment],
) -> Vec<(String, Result<String, String>)> {
    if files.is_empty() {
        return Vec::new();
    }
    if !cfg.enabled {
        return files
            .iter()
            .map(|f| (f.label.clone(), Err("sandbox disabled by config".to_string())))
            .collect();
    }
    match stage_and_run(cfg, files).await {
        Ok(results) => results,
        Err(e) => files
            .iter()
            .map(|f| (f.label.clone(), Err(format!("{e:#}"))))
            .collect(),
    }
}

/// Build the exact `docker run` argv (pure — the safety contract is
/// grep-able and unit-tested; the runner only appends the file pairs).
pub fn docker_argv(dir: &Path, cfg: &SandboxConfig, pairs: &[(String, String)]) -> Vec<String> {
    let mut argv = vec![
        "run".to_string(),
        "--rm".to_string(),
        "--network".to_string(),
        "none".to_string(),
        "--cap-drop".to_string(),
        "ALL".to_string(),
        "--security-opt".to_string(),
        "no-new-privileges".to_string(),
        "--read-only".to_string(),
        // the only writable path: bounded, noexec, wiped with the container
        "--tmpfs".to_string(),
        "/tmp:rw,noexec,nosuid,size=16m,mode=1777".to_string(),
        "--user".to_string(),
        SANDBOX_UID_GID.to_string(),
        "--memory".to_string(),
        "256m".to_string(),
        "--pids-limit".to_string(),
        "64".to_string(),
        "--cpus".to_string(),
        "0.5".to_string(),
        "-v".to_string(),
        format!("{}:/sandbox:ro", dir.display()),
        cfg.image.clone(),
        "python3".to_string(),
        "/sandbox/.extract.py".to_string(),
    ];
    for (safe, label) in pairs {
        // `safe=label` — the container keys output blocks on the safe
        // name and echoes the human label without re-deriving it.
        argv.push(format!("{safe}={label}"));
    }
    argv
}

trait ReplaceOne {
    fn replace_one(&mut self, label: &str, val: (String, Result<String, String>));
}

impl ReplaceOne for Vec<(String, Result<String, String>)> {
    fn replace_one(&mut self, label: &str, val: (String, Result<String, String>)) {
        if let Some(slot) = self.iter_mut().find(|(l, _)| l == label) {
            *slot = val;
        }
    }
}

/// Stage the files and run ONE container over the whole batch.
async fn stage_and_run(
    cfg: &SandboxConfig,
    files: &[FileAttachment],
) -> Result<Vec<(String, Result<String, String>)>> {
    let dir = std::env::temp_dir().join(format!(
        "jerith-sbx-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0)
    ));
    tokio::fs::create_dir_all(&dir)
        .await
        .context("sandbox staging dir create failed")?;
    // The container reads as uid 65534: dir + everything in it are
    // world-readable/traversable, and nothing else.
    set_mode(&dir, 0o755).await?;

    let cap_bytes = cfg.max_file_mb.saturating_mul(1024 * 1024);
    // Per-file: Ok(Some(path)) = staged; Ok(None) = honest failure text
    // stored for the result line; Err = batch-level staging breakage.
    let mut staged: Vec<(StagedFile, Result<Option<PathBuf>, String>)> =
        Vec::with_capacity(files.len());
    let mut downloads = Vec::with_capacity(files.len());
    for (i, f) in files.iter().enumerate() {
        let safe = safe_name(i, &f.label);
        staged.push((
            StagedFile {
                safe: safe.clone(),
                label: f.label.clone(),
            },
            Ok(None),
        ));
        downloads.push(download_to(&dir, safe.clone(), &f.url, cap_bytes));
    }
    let bodies = futures_util::future::join_all(downloads).await;
    for ((_, slot), body) in staged.iter_mut().zip(bodies) {
        // Ok(path) → staged for the container; Err(why) → the honest
        // per-file failure line, preserved verbatim.
        *slot = body.map(Some);
    }

    // The extractor rides INTO the volume (world-readable) — no image
    // rebuild loop when it changes; the bot binary is the single source.
    let script_path = dir.join(".extract.py");
    tokio::fs::write(&script_path, EXTRACT_SCRIPT)
        .await
        .context("sandbox extractor staging failed")?;
    set_mode(&script_path, 0o644).await?;

    // One container reads every successfully staged file; pre-seed the
    // per-file honest failures for whatever didn't stage (download
    // errors keep their WHY verbatim).
    let mut results: Vec<(String, Result<String, String>)> = Vec::with_capacity(files.len());
    let mut pairs: Vec<(String, String)> = Vec::new();
    for (st, slot) in &staged {
        match slot {
            Ok(Some(_)) => {
                pairs.push((st.safe.clone(), st.label.clone()));
                results.push((st.label.clone(), Err("no extractor output".to_string())));
            }
            Ok(None) => {
                results.push((st.label.clone(), Err("file did not stage".to_string())));
            }
            Err(e) => {
                results.push((st.label.clone(), Err(e.clone())));
            }
        }
    }

    if !pairs.is_empty() {
        let mut cmd = Command::new("docker");
        cmd.args(docker_argv(&dir, cfg, &pairs))
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);
        enum BatchFailure {
            Timeout,
            Spawn(String),
            Exit(String, std::process::Output),
        }
        let outcome = match
            tokio::time::timeout(Duration::from_secs(cfg.timeout_secs), cmd.output()).await
        {
            Err(_) => Err(BatchFailure::Timeout),
            Ok(Err(e)) => Err(BatchFailure::Spawn(e.to_string())),
            Ok(Ok(o)) if o.status.success() => Ok(o),
            Ok(Ok(o)) => Err(BatchFailure::Exit(String::from_utf8_lossy(&o.stderr).into_owned(), o)),
        };
        match outcome {
            Ok(o) => {
                let stdout = String::from_utf8_lossy(&o.stdout);
                let by_key: HashMap<String, Result<String, String>> =
                    parse_output(&stdout).into_iter().collect();
                for (st, slot) in &staged {
                    if matches!(slot, Ok(Some(_))) {
                        let val = by_key
                            .get(&st.safe)
                            .cloned()
                            .unwrap_or_else(|| Err("no extractor output".to_string()));
                        results.replace_one(&st.label, (st.label.clone(), val));
                    }
                }
            }
            Err(BatchFailure::Timeout) => {
                for (st, slot) in &staged {
                    if matches!(slot, Ok(Some(_))) {
                        results.replace_one(
                            &st.label,
                            (st.label.clone(), Err("sandbox timed out".to_string())),
                        );
                    }
                }
            }
            Err(BatchFailure::Spawn(e)) => {
                for (st, slot) in &staged {
                    if matches!(slot, Ok(Some(_))) {
                        results.replace_one(
                            &st.label,
                            (st.label.clone(), Err(format!("sandbox run failed: {e}"))),
                        );
                    }
                }
            }
            Err(BatchFailure::Exit(stderr, o)) => {
                // The container ran but failed (bad image, python crash):
                // honest per-file lines with the tail of the stderr.
                let err_tail = tail(&stderr, 160);
                let why = if err_tail.is_empty() {
                    format!("sandbox exited with {}", o.status)
                } else {
                    format!("sandbox exited with {}: {}", o.status, err_tail)
                };
                for (st, slot) in &staged {
                    if matches!(slot, Ok(Some(_))) {
                        results.replace_one(&st.label, (st.label.clone(), Err(why.clone())));
                    }
                }
            }
        }
    }

    tokio::fs::remove_dir_all(&dir).await.ok();
    Ok(results)
}

fn tail(s: &str, n: usize) -> String {
    let s = s.trim();
    if s.chars().count() <= n {
        s.to_string()
    } else {
        s.chars()
            .skip(s.chars().count() - n)
            .collect::<String>()
    }
}

/// Parse the extractor's deterministic blocks into keyed results. Tolerant
/// of mixing: unknown keys, truncated content (SIGKILL mid-write), and
/// missing blocks all degrade to honest per-file errors — never a panic.
fn parse_output(stdout: &str) -> Vec<(String, Result<String, String>)> {
    let mut out = Vec::new();
    let mut rest = stdout;
    while let Some(bidx) = rest.find("===BEGIN ") {
        let after = &rest[bidx + "===BEGIN ".len()..];
        let Some(key_end) = after.find('\n') else { break };
        let key = after[..key_end].trim().to_string();
        let after_key = &after[key_end + 1..];
        // status line: `status: <code>[:<why>]`
        let Some(sline_end) = after_key.find('\n') else { break };
        let sline = after_key[..sline_end].trim();
        let Some(marker_end) = after_key.find("---CONTENT---\n") else { break };
        let content_start = marker_end + "---CONTENT---\n".len();
        let end_marker = format!("===END {key}\n");
        let (content, remainder) = match after_key[content_start..].find(&end_marker) {
            Some(c) => (
                &after_key[content_start..content_start + c],
                &after_key[content_start + c + end_marker.len()..],
            ),
            None => {
                // truncated block (batch killed) — take what exists
                (after_key[content_start..].trim_end(), "")
            }
        };
        let status = sline.strip_prefix("status:").unwrap_or(sline).trim();
        let (code, why) = status
            .split_once(':')
            .map(|(c, w)| (c, Some(w)))
            .unwrap_or((status, None));
        let val = match (code, why) {
            ("ok", _) if content.trim().is_empty() => {
                Err("extractor returned empty text".to_string())
            }
            ("ok", _) => Ok(content.trim_end().to_string()),
            ("binary", Some(n)) => Ok(format!(
                "(binary file, {} bytes — no text extracted)",
                n.trim()
            )),
            ("binary", None) => Ok("(binary file — no text extracted)".to_string()),
            ("empty", _) => Ok("(empty file)".to_string()),
            ("unsupported", Some(w)) => Err(format!("unsupported: {}", w.trim())),
            ("error", Some(w)) => Err(format!("sandbox error: {}", w.trim())),
            ("error", None) => Err("sandbox error (no detail)".to_string()),
            _ => Err(format!("extractor status unreadable: {status}")),
        };
        out.push((key, val));
        rest = remainder;
        if rest.is_empty() {
            break;
        }
    }
    out
}

async fn set_mode(path: &Path, mode: u32) -> Result<()> {
    use std::os::unix::fs::PermissionsExt as _;
    tokio::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))
        .await
        .with_context(|| format!("chmod {mode:o} {}", path.display()))
}

async fn download_to(
    dir: &Path,
    safe: String,
    url: &str,
    cap_bytes: usize,
) -> Result<PathBuf, String> {
    if url.trim().is_empty() {
        return Err("empty file url".to_string());
    }
    let resp = crate::api::client()
        .get(url)
        .send()
        .await
        .map_err(|e| format!("file download failed: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        return Err(format!("file download: HTTP {status}"));
    }
    // Cap BEFORE buffering the body (Discord exposes content-length);
    // the post-read check is the second line of defense.
    if let Some(len) = resp.content_length()
        && len as usize > cap_bytes
    {
        return Err(format!("file too large: {len} bytes (cap {cap_bytes})"));
    }
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| format!("file body read failed: {e}"))?;
    if bytes.len() > cap_bytes {
        return Err(format!(
            "file too large: {} bytes (cap {})",
            bytes.len(),
            cap_bytes
        ));
    }
    let path = dir.join(safe);
    tokio::fs::write(&path, &bytes)
        .await
        .map_err(|e| format!("staging write failed: {e}"))?;
    set_mode(&path, 0o644).await.map_err(|e| format!("{e:#}"))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> SandboxConfig {
        SandboxConfig::default()
    }

    #[test]
    fn safe_names_strip_hostility() {
        assert_eq!(safe_name(0, "notes.txt"), "f0_notes.txt");
        // path traversal never survives
        assert_eq!(safe_name(1, "../../etc/passwd"), "f1_passwd");
        // spaces/unicode → _; empty → file
        assert_eq!(safe_name(2, "møøse spread .pdf"), "f2_m__se_spread_.pdf");
        assert_eq!(safe_name(3, ""), "f3_file");
        assert_eq!(safe_name(4, ".."), "f4_file");
        // collision-free by index prefix
        assert_ne!(safe_name(5, "a.txt"), safe_name(6, "a.txt"));
    }

    #[test]
    fn docker_argv_carries_the_full_contract() {
        let argv = docker_argv(
            Path::new("/tmp/sbx-1"),
            &cfg(),
            &[("f0_notes.txt".into(), "notes.txt".into())],
        );
        let j = argv.join(" ");
        for must in [
            "--network none",
            "--cap-drop ALL",
            "no-new-privileges",
            "--read-only",
            "--user 65534:65534",
            "--memory 256m",
            "--pids-limit 64",
            "--rm",
            ":/sandbox:ro",
            "/sandbox/.extract.py",
            "f0_notes.txt=notes.txt",
        ] {
            assert!(j.contains(must), "missing {must:?} in {j}");
        }
        // the image is config-driven, not hardcoded
        assert!(argv.contains(&cfg().image));
        // a hostile label stays within ONE argument (no shell, no new
        // element): the tail is still the single `safe=label` pair.
        let hostile = docker_argv(
            Path::new("/tmp/sbx-2"),
            &cfg(),
            &[("f0_x".into(), "x'; rm -rf /\n--privileged".into())],
        );
        assert!(
            !hostile.iter().any(|a| a == "--privileged"),
            "label became an argv flag: {hostile:?}"
        );
        assert!(
            hostile.last().unwrap().starts_with("f0_x="),
            "pair shape kept: {:?}",
            hostile.last()
        );
    }

    #[test]
    fn parse_output_shapes() {
        let out = parse_output(
            "===BEGIN f0_notes.txt\nstatus: ok\n---CONTENT---\nhello\nworld\n===END f0_notes.txt\n\
             ===BEGIN f1_broken\nstatus: error: pdftotext timed out\n---CONTENT---\n\n===END f1_broken\n\
             ===BEGIN f2_blob\nstatus: binary:7777\n---CONTENT---\n\n===END f2_blob\n\
             ===BEGIN f3_blank\nstatus: empty:\n---CONTENT---\n\n===END f3_blank\n",
        );
        assert_eq!(out.len(), 4);
        assert_eq!(out[0].0, "f0_notes.txt");
        assert_eq!(out[0].1.as_ref().unwrap(), "hello\nworld");
        assert!(out[1].1.as_ref().unwrap_err().contains("pdftotext"));
        assert_eq!(out[2].1.as_ref().unwrap(), "(binary file, 7777 bytes — no text extracted)");
        assert_eq!(out[3].1.as_ref().unwrap(), "(empty file)");
        // truncated block (container killed mid-write): degrades to the
        // partial content, never a panic.
        let truncated = parse_output("===BEGIN f0_x\nstatus: ok\n---CONTENT---\npartial text");
        assert_eq!(truncated[0].1.as_ref().unwrap(), "partial text");
        // garbage in → nothing out (no panic)
        assert!(parse_output("not a block at all").is_empty());
    }

    #[test]
    fn ok_with_empty_content_is_an_error_not_a_silent_pass() {
        let out = parse_output("===BEGIN f0\nstatus: ok\n---CONTENT---\n\n===END f0\n");
        assert!(out[0].1.is_err(), "empty 'ok' must read honestly");
    }

    /// Live-docker integration: a real text file + a real binary file
    /// through the whole staging → container → parse path (host has
    /// docker; skips rather than fails where it doesn't).
    #[tokio::test]
    async fn live_sandbox_reads_text_and_flags_binary() {
        let probe = tokio::process::Command::new("docker")
            .arg("info")
            .output()
            .await;
        if !probe.map(|p| p.status.success()).unwrap_or(false) || cfg().image.is_empty() {
            eprintln!("docker not available — skipping live sandbox test");
            return;
        }
        // a text file and a true binary, both served by a stub
        let app = axum::Router::new()
            .route(
                "/note.txt",
                axum::routing::get(|| async {
                    (
                        [("content-type", "text/plain")],
                        axum::body::Bytes::from_static(b"hello from the sandbox probe"),
                    )
                }),
            )
            .route(
                "/blob.bin",
                axum::routing::get(|| async {
                    (
                        [("content-type", "application/octet-stream")],
                        axum::body::Bytes::from(vec![0xFF, 0x00, 0xFE, 0x01, 0x00, 0x02]),
                    )
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let files = vec![
            FileAttachment {
                url: format!("http://{addr}/note.txt"),
                label: "note.txt".into(),
            },
            FileAttachment {
                url: format!("http://{addr}/blob.bin"),
                label: "blob.bin".into(),
            },
        ];
        let results = read_attachments(&cfg(), &files).await;
        assert_eq!(results.len(), 2);
        assert!(
            results[0].1.as_ref().unwrap().contains("hello from the sandbox probe"),
            "text result: {:?}",
            results[0]
        );
        assert!(
            results[1].1.as_ref().unwrap().contains("binary file"),
            "binary result: {:?}",
            results[1]
        );
    }

    /// A deliberately unreachable URL degrades to an honest per-file
    /// failure (no docker involvement, exercising the staging branch).
    #[tokio::test]
    async fn unreachable_file_reads_honestly() {
        let files = vec![FileAttachment {
            url: "http://127.0.0.1:1/nope.txt".into(),
            label: "nope.txt".into(),
        }];
        let results = read_attachments(&cfg(), &files).await;
        match &results[0].1 {
            Err(e) => assert!(e.contains("download failed: ") || e.contains("HTTP"), "why: {e}"),
            Ok(t) => panic!("unreachable file must not read: {t}"),
        }
    }

}
