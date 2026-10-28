//! External plugin drops (Phase 3a, `docs/PLUGIN-SDK.md`).
//!
//! Skill manifests live in the config dir as TOML drops — flat files
//! (`plugins.d/weather.toml`) or one-recipe directories
//! (`plugins.d/weather/plugin.toml`). Each drop declares an out-of-process
//! HTTP endpoint plus the skills it serves; the bot's tool loop can then
//! summon those skills like any in-tree one, dispatched here over signed
//! HTTP. Parsing is fail-closed at config load: any malformed drop (bad
//! URL, bad skill name, duplicate skill, unknown gating id) aborts the
//! bot start, never silently disables a skill.
//!
//! Wire contract (one round trip per invoke, bounded):
//!
//! ```text
//! POST {endpoint}
//! x-jerith-signature: hex(HMAC-SHA256(secret, "{ts_ms}.{body}"))
//! x-jerith-timestamp: {ts_ms}
//! {"method":"invoke","skill":"weather.now","args":{…},"ctx":{"ts_ms":N}}
//! → 200 {"…result…"}            (any JSON object; fed to the model verbatim)
//! → anything else / timeout / oversize → Err (tool loop sends {"error": …})
//! ```
//!
//! The signature recipe is deliberately the hooks.rs one — a plugin
//! author learns one HMAC scheme for both directions.

use anyhow::{Context as _, Result};
use hmac::{Hmac, Mac};
use serde_json::{Value, json};
use sha2::Sha256;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Cap on one dispatcher response (a bloated barker can't eat memory).
pub const MAX_RESPONSE_BYTES: usize = 65536;

/// Model-visible skill names are the OpenClaw-style dot-or-bare tokens:
/// ASCII letter/digit start, then letters/digits/`._-` (no whitespace —
/// the name lands in tool declarations and the dispatch whitelist).
fn valid_skill_name(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic())
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

/// One parsed manifest: the plugin id, its endpoint, and its skills.
#[derive(Debug, Clone)]
pub struct ExternalPlugin {
    pub id: String,
    pub endpoint: String,
    /// When set, requests carry `x-jerith-signature` (and the timestamp).
    pub secret: Option<String>,
    /// Per-invoke timeout, seconds (clamped 1–30, hook discipline).
    pub timeout_secs: u64,
    /// `[{ name, description, parameters }]` as parsed from `[[skill]]`.
    pub skills: Vec<ExternalSkill>,
}

/// One `[[skill]]` row of a manifest (pre-runner: pure declaration).
#[derive(Debug, Clone)]
pub struct ExternalSkill {
    /// Model-visible name, plugin-qualified by the parser (`now` →
    /// `weather.now` when the drop id is `weather`; already-qualified
    /// names must match the plugin prefix).
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// The `plugin.toml` file format (serde view; validation happens after).
#[derive(Debug, serde::Deserialize)]
struct ManifestFile {
    plugin: PluginTable,
    #[serde(default)]
    skill: Vec<SkillTable>,
}

#[derive(Debug, serde::Deserialize)]
struct PluginTable {
    id: String,
    endpoint: String,
    #[serde(default)]
    secret: Option<String>,
    #[serde(default)]
    timeout_secs: Option<u64>,
}

#[derive(Debug, serde::Deserialize)]
struct SkillTable {
    name: String,
    description: String,
    #[serde(default = "default_parameters")]
    parameters: Value,
}

fn default_parameters() -> Value {
    json!({"type": "object"})
}

impl ExternalPlugin {
    /// Parse + validate one manifest file. Fail-closed: every rule below
    /// names the drop and the offense — a bad drop kills the start.
    pub fn parse(path: &Path, raw: &str) -> Result<Self> {
        let file: ManifestFile =
            toml::from_str(raw).with_context(|| format!("{}: not valid TOML", path.display()))?;
        let id = file.plugin.id.trim().to_string();
        anyhow::ensure!(
            !id.is_empty() && valid_skill_name(&id),
            "{}: plugin id {id:?} is not a valid identifier",
            path.display()
        );
        let endpoint = file.plugin.endpoint.trim().to_string();
        anyhow::ensure!(
            endpoint.starts_with("http://") || endpoint.starts_with("https://"),
            "{}: endpoint {endpoint:?} is not http(s)",
            path.display()
        );
        anyhow::ensure!(
            !endpoint.contains(char::is_whitespace),
            "{}: endpoint contains whitespace",
            path.display()
        );
        // Loopback by default is the design's trust posture: a drop that
        // points at a non-loopback host is an explicit operator choice —
        // allowed, but only for https (a plain-http plugin over the LAN
        // would carry signed requests a sniffed network could replay
        // verbatim; loopback plain http cannot be sniffed remotely).
        let host = endpoint
            .trim_start_matches("http://")
            .trim_start_matches("https://");
        let host = host.split(['/', ':']).next().unwrap_or(host);
        anyhow::ensure!(
            host == "127.0.0.1"
                || host == "localhost"
                || host == "::1"
                || endpoint.starts_with("https://"),
            "{}: non-loopback endpoint must be https (got {endpoint:?})",
            path.display()
        );
        let timeout_secs = file.plugin.timeout_secs.unwrap_or(10).clamp(1, 30);
        anyhow::ensure!(
            file.plugin
                .secret
                .as_ref()
                .is_none_or(|s| !s.trim().is_empty()),
            "{}: secret must be non-empty when present",
            path.display()
        );

        let mut skills = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        for s in file.skill {
            let name = if s.name.contains('.') {
                let display = path.display();
                anyhow::ensure!(
                    s.name.starts_with(&format!("{id}.")),
                    "{display}: skill name {s:?} must start with the plugin id prefix `{id}.`",
                );
                s.name
            } else {
                format!("{id}.{}", s.name)
            };
            anyhow::ensure!(
                valid_skill_name(&name),
                "{}: skill name {name:?} is not a valid identifier",
                path.display()
            );
            anyhow::ensure!(
                seen.insert(name.clone()),
                "{}: duplicate skill {name:?}",
                path.display()
            );
            anyhow::ensure!(
                !s.description.trim().is_empty(),
                "{}: skill {name:?} needs a non-empty description",
                path.display()
            );
            anyhow::ensure!(
                s.parameters.get("type").and_then(Value::as_str) == Some("object"),
                "{}: skill {name:?} parameters must be a JSON object schema",
                path.display()
            );
            skills.push(ExternalSkill {
                name,
                description: s.description,
                parameters: s.parameters,
            });
        }
        anyhow::ensure!(
            !skills.is_empty(),
            "{}: manifest declares no skills — a lifecycle-only plugin has no out-of-process \
             surface; remove the drop or add a [[skill]]",
            path.display()
        );
        Ok(Self {
            id,
            endpoint,
            secret: file.plugin.secret.map(|s| s.trim().to_string()),
            timeout_secs,
            skills,
        })
    }

    /// Scan a drops directory: `*.toml` files and `*/plugin.toml` dirs.
    /// Missing dir → empty (the feature is opt-in by directory existence);
    /// unreadable dir → loud error (an operator who created it expects it
    /// to load).
    pub fn scan_drops(dir: &Path) -> Result<Vec<Self>> {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e).with_context(|| format!("reading {}", dir.display())),
        };
        let mut paths: Vec<PathBuf> = Vec::new();
        for entry in entries {
            let p = entry
                .with_context(|| format!("reading {}", dir.display()))?
                .path();
            if p.is_dir() {
                let inner = p.join("plugin.toml");
                if inner.exists() {
                    paths.push(inner);
                }
            } else if p.extension().is_some_and(|e| e == "toml") {
                paths.push(p);
            }
        }
        paths.sort();
        let mut plugins = Vec::new();
        let mut ids = std::collections::BTreeSet::new();
        for p in paths {
            let raw = std::fs::read_to_string(&p)
                .with_context(|| format!("reading drop {}", p.display()))?;
            let plugin = Self::parse(&p, &raw)?;
            anyhow::ensure!(
                ids.insert(plugin.id.clone()),
                "{}: duplicate plugin id {:?} across drops",
                p.display(),
                plugin.id
            );
            plugins.push(plugin);
        }
        Ok(plugins)
    }

    /// The gating id operators use in `[plugins] enabled` for this drop.
    pub fn gate_id(&self) -> String {
        format!("external:{}", self.id)
    }
}

/// Sign one request body exactly like hooks.rs does (`"{ts_ms}.{body}"`,
/// hex). Shared recipe = one thing for a plugin author to verify.
pub fn sign_body(secret: &str, ts_ms: u64, body: &str) -> String {
    let mut mac =
        Hmac::<Sha256>::new_from_slice(secret.as_bytes()).expect("HMAC accepts any key length");
    mac.update(format!("{ts_ms}.{body}").as_bytes());
    let out = mac.finalize().into_bytes();
    out.iter().map(|b| format!("{b:02x}")).collect()
}

/// Invoke one external skill over the plugin's endpoint. Errors come
/// back as `Err` with the plugin-named cause — the tool loop translates
/// them to `{"error": …}` so the model can recover (refusal contract).
pub async fn invoke(plugin: &ExternalPlugin, skill: &str, args: &Value) -> Result<Value> {
    let ts_ms = jerith_mesh_core::bootstrap::now_ms();
    let body = json!({
        "method": "invoke",
        "skill": skill,
        "args": args,
        "ctx": {"ts_ms": ts_ms},
    })
    .to_string();
    let client = reqwest::Client::builder()
        .use_rustls_tls()
        .timeout(Duration::from_secs(plugin.timeout_secs.clamp(1, 30)))
        .build()
        .context("external plugin: client build")?;
    let mut req = client
        .post(&plugin.endpoint)
        .header("content-type", "application/json");
    if let Some(secret) = &plugin.secret {
        req = req
            .header("x-jerith-signature", sign_body(secret, ts_ms, &body))
            .header("x-jerith-timestamp", ts_ms.to_string());
    }
    let resp = req
        .body(body)
        .send()
        .await
        .with_context(|| format!("external:{}: request failed", plugin.id))?;
    let status = resp.status();
    if !status.is_success() {
        anyhow::bail!("external:{}: upstream answered {}", plugin.id, status);
    }
    let bytes = resp
        .bytes()
        .await
        .with_context(|| format!("external:{}: response read failed", plugin.id))?;
    anyhow::ensure!(
        bytes.len() <= MAX_RESPONSE_BYTES,
        "external:{}: response exceeds {MAX_RESPONSE_BYTES} bytes",
        plugin.id
    );
    let value: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("external:{}: response is not JSON", plugin.id))?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn write_drop(dir: &Path, name: &str, body: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, body).unwrap();
        p
    }

    const GOOD: &str = r#"
[plugin]
id = "weather"
endpoint = "http://127.0.0.1:7777/rpc"
secret = "s3cret"

[[skill]]
name = "now"
description = "Current conditions."
[[skill]]
name = "weather.forecast"
description = "Multi-day forecast."
"#;

    #[test]
    fn parses_good_manifest_and_qualifies_names() {
        let dir = std::env::temp_dir().join("jerith-ext-test-good");
        std::fs::create_dir_all(&dir).unwrap();
        let p = write_drop(&dir, "weather.toml", GOOD);
        let plugin = ExternalPlugin::parse(&p, GOOD).unwrap();
        assert_eq!(plugin.id, "weather");
        assert_eq!(plugin.gate_id(), "external:weather");
        assert_eq!(plugin.timeout_secs, 10);
        let names: Vec<&str> = plugin.skills.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["weather.now", "weather.forecast"]);
        assert_eq!(plugin.secret.as_deref(), Some("s3cret"));
    }

    #[test]
    fn rejects_bad_manifests_loud() {
        let dir = std::env::temp_dir().join("jerith-ext-test-bad");
        std::fs::create_dir_all(&dir).unwrap();
        let cases: [(&str, &str); 6] = [
            (
                "no-skill",
                "[plugin]\nid = \"w\"\nendpoint = \"http://127.0.0.1:1/x\"\n",
            ),
            (
                "bad-url",
                "[plugin]\nid = \"w\"\nendpoint = \"file:///etc\"\n[[skill]]\nname = \"x\"\ndescription = \"d\"\n",
            ),
            (
                "lan-plain-http",
                "[plugin]\nid = \"w\"\nendpoint = \"http://192.168.1.5:7/x\"\n[[skill]]\nname = \"x\"\ndescription = \"d\"\n",
            ),
            (
                "wrong-prefix",
                "[plugin]\nid = \"w\"\nendpoint = \"http://127.0.0.1:1/x\"\n[[skill]]\nname = \"other.x\"\ndescription = \"d\"\n",
            ),
            (
                "dup-skill",
                "[plugin]\nid = \"w\"\nendpoint = \"http://127.0.0.1:1/x\"\n[[skill]]\nname = \"x\"\ndescription = \"d\"\n[[skill]]\nname = \"x\"\ndescription = \"d2\"\n",
            ),
            (
                "bad-schema",
                "[plugin]\nid = \"w\"\nendpoint = \"http://127.0.0.1:1/x\"\n[[skill]]\nname = \"x\"\ndescription = \"d\"\nparameters = \"nope\"\n",
            ),
        ];
        for (name, body) in cases {
            let p = write_drop(&dir, &format!("{name}.toml"), body);
            assert!(
                ExternalPlugin::parse(&p, body).is_err(),
                "{name} must fail loud"
            );
        }
    }

    #[test]
    fn scan_drops_sorts_and_rejects_duplicate_ids() {
        let dir = std::env::temp_dir().join("jerith-ext-test-scan");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let b = GOOD
            .replace("id = \"weather\"", "id = \"beta\"")
            .replace("weather.forecast", "beta.forecast");
        write_drop(&dir, "weather.toml", GOOD);
        write_drop(&dir, "beta.toml", &b);
        let plugins = ExternalPlugin::scan_drops(&dir).unwrap();
        let ids: Vec<&str> = plugins.iter().map(|p| p.id.as_str()).collect();
        assert_eq!(ids, vec!["beta", "weather"], "sorted by path");

        write_drop(&dir, "again.toml", GOOD); // same id as weather.toml
        assert!(
            ExternalPlugin::scan_drops(&dir).is_err(),
            "dup id fails loud"
        );
    }

    /// Phase 3d pin (docs/PLUGIN-SDK.md): the SHIPPED template manifest
    /// (`crates/plugin-template/plugin.toml`) must parse right here,
    /// today's rules — a template that drifts out of the drop parser's
    /// fail-closed contract breaks the build, never a live bot.
    #[test]
    fn shipped_template_manifest_parses_under_todays_rules() {
        let raw = include_str!("../../plugin-template/plugin.toml");
        let p = ExternalPlugin::parse(Path::new("plugin-template/plugin.toml"), raw)
            .expect("the shipped template manifest must stay valid");
        assert_eq!(p.id, "example");
        assert_eq!(p.gate_id(), "external:example");
        let names: Vec<&str> = p.skills.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["example.echo", "example.roll"]);
        assert!(p.secret.is_some(), "template ships signed by default");
    }

    /// Same pin discipline for the mesh pet (`crates/pet-plugin/`, the
    /// SDK's second real consumer): the pet's shipped manifest must
    /// parse under today's drop rules — five skills, gated as
    /// `external:pet`, signed by default. The pet plugin's OWN test
    /// pins the other direction (its declared rows = the handlers its
    /// bin registers), so neither half can drift the union away.
    #[test]
    fn shipped_pet_manifest_parses_under_todays_rules() {
        let raw = include_str!("../../pet-plugin/plugin.toml");
        let p = ExternalPlugin::parse(Path::new("pet-plugin/plugin.toml"), raw)
            .expect("the shipped pet manifest must stay valid");
        assert_eq!(p.id, "pet");
        assert_eq!(p.gate_id(), "external:pet");
        let mut names: Vec<&str> = p.skills.iter().map(|s| s.name.as_str()).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            vec!["pet.feed", "pet.look", "pet.play", "pet.rename", "pet.talk"]
        );
        assert!(p.secret.is_some(), "pet ships signed by default");
    }

    /// Same pin discipline for the quarantine scanner
    /// (`crates/quarantine-plugin/`, download checker over the same
    /// signed contract): its shipped manifest parses under today's
    /// rules.
    #[test]
    fn shipped_quarantine_manifest_parses_under_todays_rules() {
        let raw = include_str!("../../quarantine-plugin/plugin.toml");
        let p = ExternalPlugin::parse(Path::new("quarantine-plugin/plugin.toml"), raw)
            .expect("the shipped quarantine manifest must stay valid");
        assert_eq!(p.id, "quarantine");
        assert_eq!(p.gate_id(), "external:quarantine");
        assert!(p.secret.is_some(), "quarantine ships signed by default");
    }

    /// Same pin discipline for the house card (`crates/house-plugin/`,
    /// the SDK's third consumer — answeneurs «how are you doing» over
    /// the same signed contract): two skills (`house.status` /
    /// `house.journal`), gated as `external:house`, signed by default.
    #[test]
    fn shipped_house_manifest_parses_under_todays_rules() {
        let raw = include_str!("../../house-plugin/plugin.toml");
        let p = ExternalPlugin::parse(Path::new("house-plugin/plugin.toml"), raw)
            .expect("the shipped house manifest must stay valid");
        assert_eq!(p.id, "house");
        assert_eq!(p.gate_id(), "external:house");
        let mut names: Vec<&str> = p.skills.iter().map(|s| s.name.as_str()).collect();
        names.sort_unstable();
        assert_eq!(names, vec!["house.journal", "house.status"]);
        assert!(p.secret.is_some(), "house ships signed by default");
    }

    /// Same pin discipline for the weather skill
    /// (`crates/weather-plugin/`, open-meteo forecast skills over the
    /// same signed contract; a PUBLIC skill — gated via
    /// `[plugins] enabled`, never `owner_only`, by design). Two skills
    /// (`weather.now` / `weather.forecast`), signed by default.
    #[test]
    fn shipped_weather_manifest_parses_under_todays_rules() {
        let raw = include_str!("../../weather-plugin/plugin.toml");
        let p = ExternalPlugin::parse(Path::new("weather-plugin/plugin.toml"), raw)
            .expect("the shipped weather manifest must stay valid");
        assert_eq!(p.id, "weather");
        assert_eq!(p.gate_id(), "external:weather");
        let mut names: Vec<&str> = p.skills.iter().map(|s| s.name.as_str()).collect();
        names.sort_unstable();
        assert_eq!(names, vec!["weather.forecast", "weather.now"]);
        assert!(p.secret.is_some(), "weather ships signed by default");
    }

    #[test]
    fn missing_drop_dir_is_empty_not_error() {
        let ghost = std::env::temp_dir().join("jerith-ext-nope-does-not-exist");
        let _ = std::fs::remove_dir_all(&ghost);
        assert!(ExternalPlugin::scan_drops(&ghost).unwrap().is_empty());
    }

    #[test]
    fn signature_matches_the_hooks_recipe() {
        let ts = 1_700_000_000_000u64;
        let body = "{\"method\":\"invoke\"}";
        let sig = sign_body("k", ts, body);
        assert!(crate::hooks::verify_signature(ts, body, "k", &sig));
    }

    /// Full dispatch round-trip over a real local axum stub: verifies the
    /// signature header, the body shape, and the bounded-JSON response.
    #[tokio::test]
    async fn invoke_roundtrips_signed_body_and_result() {
        let app = axum::Router::new().route(
            "/rpc",
            axum::routing::post(|headers: axum::http::HeaderMap, body: String| async move {
                let ts: u64 = headers["x-jerith-timestamp"]
                    .to_str()
                    .unwrap()
                    .parse()
                    .unwrap();
                let sig = headers["x-jerith-signature"].to_str().unwrap();
                assert!(crate::hooks::verify_signature(ts, &body, "s3cret", sig));
                let v: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(v["method"], "invoke");
                assert_eq!(v["skill"], "weather.now");
                assert_eq!(v["args"]["place"], "Praha");
                axum::Json(json!({"provider": "open-meteo", "temperature_c": 21.5}))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let plugin = ExternalPlugin {
            id: "weather".into(),
            endpoint: format!("http://{addr}/rpc"),
            secret: Some("s3cret".into()),
            timeout_secs: 5,
            skills: Vec::new(),
        };
        let out = invoke(&plugin, "weather.now", &json!({"place": "Praha"}))
            .await
            .unwrap();
        assert_eq!(out["provider"], "open-meteo");
    }

    /// A dead endpoint errors (typed) instead of hanging — the tool loop
    /// turns it into a model-recoverable error, never a chat failure.
    #[tokio::test]
    async fn invoke_dead_endpoint_is_a_typed_error() {
        let plugin = ExternalPlugin {
            id: "dead".into(),
            endpoint: "http://127.0.0.1:9/rpc".into(),
            secret: None,
            timeout_secs: 1,
            skills: Vec::new(),
        };
        let err = invoke(&plugin, "dead.now", &json!({})).await;
        let msg = err.unwrap_err().to_string();
        assert!(msg.contains("external:dead"), "typed: {msg}");
    }
}
