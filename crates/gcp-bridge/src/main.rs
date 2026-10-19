//! `jerith-gcp-bridge` — the external plugin behind the bot's `[plugins]
//! owner_only = ["external:gcp"]` drop (Phase 3a, docs/PLUGIN-SDK.md).
//!
//! The bot's chat tool loop dispatches `gcp.*` skills here over the signed
//! HTTP contract (`external.rs`): `POST /rpc` with the `x-jerith-signature`
//! HMAC-SHA256 header over `"{ts_ms}.{body}"`. Each skill maps to a
//! grounded `gcloud` invocation — argv-only, no shell, charset-validated
//! tokens, wall-clock timeout, byte-capped output — and its JSON result
//! (or honest typed error) comes back as the tool result the model reads.
//!
//! Scope notes, mirroring the design conversation:
//! - The bridge enforces NO allowlist beyond token validation (`gcp.exec`
//!   reaches any `gcloud` subcommand), because the operator's chosen
//!   posture is full management under the CLI's existing identity. The
//!   hard boundary is instead WHO can talk to the bridge: the bot joins
//!   the `gcp.*` skills into a session's tool registry only for owner-DM
//!   conversations (plugins.rs `OWNER_ONLY_PLUGINS`), and the wire secret
//!   keeps other local processes from speaking for the tool loop. The
//!   bridge additionally trusts only requests whose signature verifies —
//!   an unsigned or stale-timed request is refused before it spawns
//!   anything.
//! - Every invocation is journaled (`tracing`) with skill + argv + outcome:
//!   an operator can read the machine's paper trail for every gcloud run
//!   the model pulled the trigger on.
//!
//! Runs as a systemd --user unit (`jerith-gcp-bridge.service`) bound to
//! loopback only; the manifest pin is `http://127.0.0.1:7955/rpc`.

use anyhow::{Context as _, Result};
use axum::extract::State;
use axum::http::HeaderMap;
use axum::routing::post;
use hmac::{Hmac, Mac};
use regex::Regex;
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Duration;

type HmacSha256 = Hmac<sha2::Sha256>;

/// One response's stdout cap (bytes): a fleet-wide `instances list` JSON
/// still fits comfortably; a runaway dump dies here with a typed error.
const MAX_OUTPUT_BYTES: usize = 512 * 1024;
/// argv volume caps for the generic `gcp.exec` skill (token count and
/// total bytes) — generous for any real gcloud line, tiny for abuse.
const EXEC_MAX_TOKENS: usize = 96;
const EXEC_MAX_BYTES: usize = 8192;

/// Valid identifier tokens anywhere in a gcloud argv: names, zones,
/// emails, `--filter=` values, URI-ish args. Same shape CLIs already
/// enforce; anything shell-ish (whitespace, `$();|` etc.) is refused —
/// gcloud is spawned with argv directly, so this is defense in depth for
/// the journaled parsing, not injection prevention.
fn valid_token(tok: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        // ASCII alnum + `.` `_` `@` `:` `/` `+` `=` `,` `-`; no control
        // chars, no whitespace, no shell metacharacters; bounded length.
        Regex::new(r"^[A-Za-z0-9._@:/+=,\-]{1,512}$").unwrap()
    });
    re.is_match(tok)
}

/// The bridge's full runtime config (CLI flags).
#[derive(clap::Parser, Debug)]
#[command(name = "jerith-gcp-bridge", about = "gcloud skill bridge for jerith-mesh external plugins")]
struct Args {
    /// Loopback port to listen on (the manifest endpoint must agree).
    #[arg(long, default_value_t = 7955)]
    port: u16,
    /// File holding the shared HMAC secret (bot.toml drop `secret` loads
    /// the same bytes; the file format is one line, trimmed).
    #[arg(long, default_value = "~/.jerith-mesh/keys/gcp-bridge.secret")]
    secret_file: String,
    /// gcloud binary to invoke (absolute path beats $PATH ambiguity).
    #[arg(long, default_value = "~/google-cloud-sdk/bin/gcloud")]
    gcloud: String,
    /// Wall-clock cap for ONE gcloud run (seconds). Compute lifecycle
    /// skills run `--async` so the CLI returns in well under this.
    #[arg(long, default_value_t = 25u64)]
    gcloud_timeout_secs: u64,
}

impl Args {
    fn gcloud_path(&self) -> PathBuf {
        dunce_home_expand(&self.gcloud)
    }
}

/// Expand `~` and `$HOME` (env_of HOME), no full shell expansion on purpose.
fn dunce_home_expand(spec: &str) -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/".into());
    let s = spec.strip_prefix("~/").map(|rest| format!("{home}/{rest}"));
    let s = s.unwrap_or_else(|| spec.to_string());
    let s = s.replace("$HOME", &home);
    PathBuf::from(s)
}

fn load_secret(path: &str) -> Result<String> {
    let path = dunce_home_expand(path);
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("reading secret file {}", path.display()))?;
    let trimmed = raw.trim().to_string();
    anyhow::ensure!(!trimmed.is_empty(), "secret file {} is empty", path.display());
    Ok(trimmed)
}

/// Verify the signed request exactly like hooks.rs/the bot's external.rs
/// recipe: `hex(HMAC-SHA256(secret, "{ts_ms}.{body}"))` with the ts in
/// `x-jerith-timestamp`. A stale (older than 5 min) or malformed request
/// is refused here, before any gcloud is spawned.
fn verify_request(secret: &str, headers: &HeaderMap, body: &str) -> Result<()> {
    let ts_raw = headers
        .get("x-jerith-timestamp")
        .and_then(|v| v.to_str().ok())
        .context("missing x-jerith-timestamp")?;
    let ts: u64 = ts_raw.parse().context("x-jerith-timestamp is not u64")?;
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    anyhow::ensure!(
        now_ms.abs_diff(ts) < 5 * 60 * 1000,
        "stale or future-dated request (ts {ts}, now {now_ms})"
    );
    let sig = headers
        .get("x-jerith-signature")
        .and_then(|v| v.to_str().ok())
        .context("missing x-jerith-signature")?;
    let expected = compute_signature(secret, ts, body);
    anyhow::ensure!(
        constant_time_eq(sig.as_bytes(), expected.as_bytes()),
        "signature mismatch"
    );
    Ok(())
}

fn compute_signature(secret: &str, ts_ms: u64, body: &str) -> String {
    let mut mac =
        HmacSha256::new_from_slice(secret.as_bytes()).expect("HMAC accepts any key length");
    mac.update(format!("{ts_ms}.{body}").as_bytes());
    mac.finalize().into_bytes().iter().map(|b| format!("{b:02x}")).collect()
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Shared handler state.
#[derive(Clone)]
struct Bridge {
    secret: String,
    gcloud: PathBuf,
    gcloud_timeout: Duration,
}

/// Build the `gcloud` argv for one skill + args. Returns Err with an
/// honest message when an argument is refused as malformed. Every skill
/// accepts an optional `project` (validated) threaded as `--project=…`.
fn argv_for(skill: &str, args: &Value) -> Result<Vec<String>> {
    let project = args.get("project").and_then(Value::as_str);
    if let Some(p) = project {
        anyhow::ensure!(valid_token(p), "gcp: malformed project {:?}", p);
    }
    let mut argv: Vec<String> = Vec::new();
    let push_token = |argv: &mut Vec<String>, tok: &str| -> Result<()> {
        anyhow::ensure!(
            valid_token(tok),
            "gcp: malformed argument {:?} (must match [A-Za-z0-9._@:/+=,-] and stay ≤512 chars)",
            tok
        );
        argv.push(tok.to_string());
        Ok(())
    };
    let needs = |key: &str| -> Result<String> {
        args.get(key)
            .and_then(Value::as_str)
            .map(str::to_string)
            .with_context(|| format!("gcp: missing required argument {key:?}"))
    };

    match skill {
        "gcp.status" => {
            argv.push("auth".into());
            argv.push("list".into());
            argv.push("--format=json".into());
        }
        "gcp.compute.list" => {
            argv.push("compute".into());
            argv.push("instances".into());
            argv.push("list".into());
            if let Some(zone) = args.get("zone").and_then(Value::as_str) {
                push_token(&mut argv, &format!("--filter=zone:{zone}"))?;
            }
            argv.push("--format=json".into());
        }
        "gcp.compute.describe" => {
            let name = needs("name")?;
            let zone = needs("zone")?;
            argv.push("compute".into());
            argv.push("instances".into());
            argv.push("describe".into());
            push_token(&mut argv, &format!("--zone={zone}"))?;
            push_token(&mut argv, &name)?;
            argv.push("--format=json".into());
        }
        "gcp.compute.start" | "gcp.compute.stop" => {
            let name = needs("name")?;
            let zone = needs("zone")?;
            argv.push("compute".into());
            argv.push("instances".into());
            argv.push(skill.split('.').next_back().unwrap_or_default().into());
            // Long-running lifecycle ops: the CLI returns the operation
            // reference immediately instead of holding the dispatcher's
            // ≤30 s window hostage to a minute-long VM transition.
            argv.push("--async".into());
            argv.push("--format=json".into());
            push_token(&mut argv, &format!("--zone={zone}"))?;
            push_token(&mut argv, &name)?;
        }
        "gcp.compute.delete" => {
            // Two-step trigger: `confirm` must restate the instance name.
            // The model's skill call cannot confirm accidentally — the
            // name must be typed twice, in different fields.
            let name = needs("name")?;
            let zone = needs("zone")?;
            let confirm = needs("confirm")?;
            anyhow::ensure!(
                name == confirm,
                "gcp: confirm ({confirm:?}) must restate the instance name ({name:?}) — deletion refused"
            );
            argv.push("compute".into());
            argv.push("instances".into());
            argv.push("delete".into());
            argv.push("--async".into());
            argv.push("--format=json".into());
            argv.push("--keep-alive".into()); // delete keeps intent alive; escalate no
            argv.push("--quiet".into());
            push_token(&mut argv, &format!("--zone={zone}"))?;
            push_token(&mut argv, &name)?;
        }
        "gcp.storage.list" => {
            argv.push("storage".into());
            argv.push("buckets".into());
            argv.push("list".into());
            argv.push("--format=json".into());
        }
        "gcp.storage.ls" => {
            let bucket_uri = needs("bucket_uri")?;
            // Must be a full (or prefix) URI gcloud storage understands.
            anyhow::ensure!(
                bucket_uri.starts_with("gs://"),
                "gcp: bucket_uri must start with gs:// (got {bucket_uri:?})"
            );
            argv.push("storage".into());
            argv.push("ls".into());
            push_token(&mut argv, &bucket_uri)?;
            argv.push("--format=json".into());
        }
        "gcp.billing.show" => {
            argv.push("billing".into());
            argv.push("projects".into());
            argv.push("describe".into());
            argv.push(needs("project")?);
            argv.push("--format=json".into());
        }
        "gcp.iam.get" => {
            argv.push("projects".into());
            argv.push("get-iam-policy".into());
            argv.push(needs("project")?);
            argv.push("--format=json".into());
        }
        "gcp.exec" => {
            // Full control surface: the model supplies the FULL gcloud
            // argv (without the `gcloud` token itself). Bounded volume +
            // per-token validation; results and stderr come back as the
            // tool result.
            let tokens = args
                .get("argv")
                .and_then(Value::as_array)
                .context("gcp: exec needs an argv array")?;
            anyhow::ensure!(
                tokens.len() <= EXEC_MAX_TOKENS,
                "gcp: exec argv exceeds {EXEC_MAX_TOKENS} tokens"
            );
            let mut total = 0usize;
            for tok in tokens {
                let tok = tok
                    .as_str()
                    .map(str::to_string)
                    .with_context(|| "gcp: exec argv entries must be strings")?;
                total += tok.len();
                anyhow::ensure!(
                    total <= EXEC_MAX_BYTES,
                    "gcp: exec argv exceeds {EXEC_MAX_BYTES} bytes total"
                );
                push_token(&mut argv, &tok)?;
            }
            anyhow::ensure!(
                !argv.is_empty(),
                "gcp: exec argv must name at least one subcommand"
            );
        }
        other => anyhow::bail!("gcp: unknown skill {other:?} for this bridge"),
    }
    if let Some(p) = project {
        argv.insert(0, format!("--project={p}"));
    }
    Ok(argv)
}

/// Run gcloud for one skill; returns the full dispatch JSON value. The
/// envelope keeps every invocation's wiring honest: `ok` says spawned +
/// parsed (not "operation succeeded" — gcloud's exit + stderr ride
/// alongside), `data` is the parsed `--format=json` stdout, `text` is
/// stdout that wasn't JSON, `stderr` is bounded, `exit` is the CLI code.
async fn run_gcloud(
    bridge: &Bridge,
    skill: &str,
    args: &Value,
) -> Result<Value> {
    let argv0 = bridge.gcloud.clone();
    anyhow::ensure!(
        argv0.exists(),
        "gcp: gcloud binary not found at {}",
        argv0.display()
    );
    let argv = argv_for(skill, args)?;
    tracing::info!(
        target: "gcp-bridge",
        skill,
        argv = %argv.join(" "),
        "gcloud invocation"
    );

    let mut cmd = tokio::process::Command::new(&argv0);
    cmd.args(&argv)
        .env("CLOUDSDK_CORE_DISABLE_PROMPTS", "1")
        .env("CLOUDSDK_CORE_FORMAT", "json");
    // No prompt ceremony, no shell: exact spawn of the validated argv.
    let child_fut = cmd
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output();
    let output = tokio::time::timeout(bridge.gcloud_timeout, child_fut)
        .await
        .map_err(|_| {
            tracing::warn!(
                target: "gcp-bridge",
                skill,
                timeout_secs = bridge.gcloud_timeout.as_secs(),
                "gcloud timed out and was killed"
            );
            anyhow::anyhow!("gcp: gcloud timed out after {}s", bridge.gcloud_timeout.as_secs())
        })?
        .context("gcp: spawning gcloud failed")?;

    let stdout_text = String::from_utf8_lossy(&output.stdout);
    anyhow::ensure!(
        stdout_text.len() <= MAX_OUTPUT_BYTES,
        "gcp: stdout exceeds {MAX_OUTPUT_BYTES} bytes — narrow the query"
    );
    let stderr_text = String::from_utf8_lossy(&output.stderr);
    let stderr_bounded: String = stderr_text
        .chars()
        .take(4000)
        .collect::<String>()
        .trim()
        .to_string();

    let exit = output.status.code().unwrap_or(-1);
    let data: Value = if !stdout_text.trim().is_empty() {
        serde_json::from_str(stdout_text.trim()).unwrap_or(Value::Null)
    } else {
        Value::Null
    };
    let ok = exit == 0;
    if !ok {
        tracing::warn!(
            target: "gcp-bridge",
            skill,
            exit,
            stderr = %stderr_bounded,
            "gcloud exited nonzero"
        );
    }
    tracing::info!(
        target: "gcp-bridge",
        skill,
        exit,
        bytes = stdout_text.len(),
        "gcloud finished"
    );
    let mut envelope = json!({
        "ok": ok,
        "skill": skill,
        "exit": exit,
        "stderr": stderr_bounded,
    });
    let obj = envelope.as_object_mut().unwrap();
    if data.is_null() {
        obj.insert("text".into(), Value::String(stdout_text.trim().to_string()));
    } else {
        obj.insert("data".into(), data);
    }
    Ok(envelope)
}

/// The single POST /rpc handler: verify → dispatch → respond.
async fn rpc(State(bridge): State<Bridge>, headers: HeaderMap, body: String) -> axum::Json<Value> {
    if let Err(e) = verify_request(&bridge.secret, &headers, &body) {
        tracing::warn!(target: "gcp-bridge", error = %e, "request refused");
        return axum::Json(json!({"ok": false, "error": format!("gcp: {e}")}));
    }
    let parsed: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => return axum::Json(json!({"ok": false, "error": format!("gcp: bad body: {e}")})),
    };
    let skill = parsed.get("skill").and_then(Value::as_str).unwrap_or_default();
    let args = parsed.get("args").cloned().unwrap_or(json!({}));
    match run_gcloud(&bridge, skill, &args).await {
        Ok(v) => axum::Json(v),
        Err(e) => axum::Json(json!({"ok": false, "error": format!("{e:#}")})),
    }
}

fn main() -> Result<()> {
    let args = <Args as clap::Parser>::parse();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                "info".into()
            }),
        )
        .init();
    let secret = load_secret(&args.secret_file)?;
    let bridge = Bridge {
        secret,
        gcloud: args.gcloud_path(),
        gcloud_timeout: Duration::from_secs(args.gcloud_timeout_secs),
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("tokio runtime")?;
    runtime.block_on(async move {
        let app = axum::Router::new().route("/rpc", post(rpc)).with_state(bridge);
        let addr = std::net::SocketAddr::from(([127, 0, 0, 1], args.port));
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .with_context(|| format!("binding {addr}"))?;
        tracing::info!(
            target: "gcp-bridge",
            addr = %addr,
            "gcp-bridge listening (signature-gated, argv-only gcloud surface)"
        );
        axum::serve(listener, app).await.context("axum serve")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn token_validation_shape() {
        assert!(valid_token("instance-name-1"));
        assert!(valid_token("us-central1-a"));
        assert!(valid_token("user@google.com"));
        assert!(valid_token("--filter=name:foo"));
        assert!(valid_token("gs://my-bucket/prefix/"));
        assert!(valid_token("--format=json"));
        assert!(valid_token("fdc1:1f2b::/32"));
        // Shell-ish and control shapes are all refused.
        assert!(!valid_token(""));
        assert!(!valid_token("a b"));
        assert!(!valid_token("a;b"));
        assert!(!valid_token("a$b"));
        assert!(!valid_token("$(id)"));
        assert!(!valid_token("a\nb"));
        assert!(!valid_token("a|b"));
        assert!(!valid_token("a'b"));
        assert!(!valid_token("a\\b"));
        assert!(!valid_token(&"x".repeat(513)));
    }

    #[test]
    fn argv_maps_grounded_skills() {
        // No-shell argv for a zone instance stop with --async high up so
        // the CLI returns in time for the dispatcher's bounded window.
        let argv = argv_for("gcp.compute.stop", &json!({"zone": "us-central1-a", "name": "vm-1"}))
            .unwrap();
        assert_eq!(
            argv,
            vec![
                "compute",
                "instances",
                "stop",
                "--async",
                "--format=json",
                "--zone=us-central1-a",
                "vm-1",
            ]
        );

        let list = argv_for("gcp.compute.list", &json!({})).unwrap();
        assert_eq!(
            list,
            vec!["compute", "instances", "list", "--format=json"]
        );

        // Malformed identifiers can never land in argv.
        let bad = argv_for("gcp.compute.describe", &json!({"zone": "us-central1-a", "name": "a b"}));
        assert!(bad.is_err());
        // Missing required args name the missing key.
        let missing = argv_for("gcp.compute.describe", &json!({}))
            .unwrap_err()
            .to_string();
        assert!(missing.contains("missing required argument"), "{missing}");
    }

    #[test]
    fn delete_needs_confirm_and_exec_is_bounded() {
        let no_confirm = argv_for(
            "gcp.compute.delete",
            &json!({"zone": "us-central1-a", "name": "vm-1", "confirm": "other"}),
        );
        assert!(no_confirm.is_err(), "mismatched confirm must refuse");

        let too_big = argv_for("gcp.exec", &json!({"argv": vec!["x"; 97]}));
        assert!(too_big.is_err(), "token-count bound");
        let shell_char = argv_for("gcp.exec", &json!({"argv": ["compute", "instances", "list", "--filter=a b?"]}));
        assert!(shell_char.is_err(), "metachar token must refuse");
        let ok = argv_for("gcp.exec", &json!({"argv": ["compute", "instances", "list", "--format=json"]}))
            .unwrap();
        assert_eq!(
            ok.first().and_then(|a| Some(a.as_str())),
            Some("compute")
        );
        let project_threaded = argv_for("gcp.exec", &json!({
            "argv": ["compute", "instances", "list"],
            "project": "my-proj"
        }))
        .unwrap();
        assert_eq!(project_threaded[0], "--project=my-proj");
    }

    #[test]
    fn signature_matches_the_shared_recipe() {
        // Identical math to the bot's external.rs (the shared "{ts}.{body}"
        // HMAC-SHA256 hex): boundaries hold exactly when the secrets match.
        let ts = 1_700_000_000_000u64;
        let body = "{\"method\":\"invoke\"}";
        let sig = compute_signature("s3cret", ts, body);

        let mut mac = Hmac::<sha2::Sha256>::new_from_slice(b"s3cret").unwrap();
        mac.update(format!("{ts}.{body}").as_bytes());
        let expected: String = mac.finalize().into_bytes().iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(sig, expected);
        assert!(constant_time_eq(sig.as_bytes(), expected.as_bytes()));
        assert!(!constant_time_eq(sig.as_bytes(), b"wrong"));
    }

    #[tokio::test]
    async fn spawn_pipeline_runs_a_stub_gcloud() {
        // /bin/echo as the "gcloud" proves argv threading without network;
        // the envelope marks the non-JSON stdout honestly.
        let bridge = Bridge {
            secret: String::new(),
            gcloud: PathBuf::from("/bin/echo"),
            gcloud_timeout: Duration::from_secs(5),
        };
        let out = run_gcloud(&bridge, "gcp.compute.list", &json!({}))
            .await
            .unwrap();
        assert!(out["skill"] == "gcp.compute.list");
        assert!(out["text"].is_string(), "stub echoes non-JSON");
        let say = out["text"].as_str().unwrap();
        assert!(say.starts_with("compute instances list"), "{say}");
    }

    #[tokio::test]
    async fn missing_gcloud_binary_is_typed() {
        let bridge = Bridge {
            secret: String::new(),
            gcloud: PathBuf::from("/nonexistent/gcloud"),
            gcloud_timeout: Duration::from_secs(1),
        };
        let err = run_gcloud(&bridge, "gcp.compute.list", &json!({}))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not found"), "{err}");
    }
}
