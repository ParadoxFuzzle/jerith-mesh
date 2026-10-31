//! Dashboard settings surface (`GET`/`POST /v1/config`).
//!
//! The web UI may change the node's *hot-reloadable* knobs — the same ones
//! `jerith config set` writes — and nothing else. Identity, ports, the
//! hosted model and the memory backend/DSN are deliberately read-only
//! here: they either need a restart, a deliberate CLI decision, or carry
//! a secret. Every value is validated with the same rules the CLI applies
//! before `config.toml` is written.
//!
//! The HTTP layer (gateway.rs) owns the request guard; this module is the
//! pure half (labels, extraction, validation) so it tests without a server
//! or a real config file.

use anyhow::Context as _;
use serde_json::{Value, json};

use crate::config::{MIN_CONTRIBUTION_PERCENT, NodeConfig, PrivacyLevel};

/// Settings the dashboard may change. Everything absent here is read-only
/// on purpose (see the module docs).
pub const EDITABLE: [&str; 12] = [
    "contribution_percent",
    "memory_max_rows",
    "bootstrap_url",
    "runtime_url",
    "runtime_primary_model",
    "runtime_deep_model",
    "runtime_vision_model",
    "runtime_tool_model",
    "runtime_num_predict",
    "embedding_model",
    "privacy_level",
    "websearch_enabled",
];

pub fn privacy_label(level: PrivacyLevel) -> &'static str {
    match level {
        PrivacyLevel::FullMesh => "full_mesh",
        PrivacyLevel::TrustedPeers => "trusted_peers",
        PrivacyLevel::LocalOnly => "local_only",
    }
}

fn opt(v: &Option<String>) -> String {
    v.clone().unwrap_or_default()
}

/// The editable settings, already formatted for the form. `value` is a
/// string in every case (the form round-trips strings; `apply` parses).
pub fn settings_json(cfg: &NodeConfig) -> Value {
    json!([
        {
            "key": "contribution_percent",
            "label": "Contribution %",
            "value": cfg.contribution_percent.to_string(),
            "kind": "number",
            "min": MIN_CONTRIBUTION_PERCENT,
            "help": "share of this node's compute offered to the mesh (minimum 25%)"
        },
        {
            "key": "memory_max_rows",
            "label": "Memory retention (rows)",
            "value": cfg.memory_max_rows.to_string(),
            "kind": "number",
            "min": 0,
            "help": "newest rows survive; 0 = uncapped (pruning follows the next insert)"
        },
        {
            "key": "bootstrap_url",
            "label": "Bootstrap URL",
            "value": cfg.bootstrap_url,
            "kind": "text",
            "help": "primary discovery server (loopback is fastest for a co-hosted node)"
        },
        {
            "key": "runtime_url",
            "label": "Runtime URL",
            "value": cfg.runtime_url,
            "kind": "text",
            "help": "local OpenAI-compatible runtime that serves chat"
        },
        {
            "key": "runtime_primary_model",
            "label": "Primary chat model",
            "value": opt(&cfg.runtime_primary_model),
            "kind": "text",
            "help": "blank = the caller/base model (classic behavior)"
        },
        {
            "key": "runtime_deep_model",
            "label": "Deep tier model",
            "value": opt(&cfg.runtime_deep_model),
            "kind": "text",
            "help": "reasoning/long-context lane; blank disables the tier"
        },
        {
            "key": "runtime_vision_model",
            "label": "Vision tier model",
            "value": opt(&cfg.runtime_vision_model),
            "kind": "text",
            "help": "image-content lane; blank disables the tier"
        },
        {
            "key": "runtime_tool_model",
            "label": "Tool tier model",
            "value": opt(&cfg.runtime_tool_model),
            "kind": "text",
            "help": "function-calling lane; blank disables the tier"
        },
        {
            "key": "runtime_num_predict",
            "label": "Max output tokens",
            "value": cfg.runtime_num_predict.map(|n| n.to_string()).unwrap_or_default(),
            "kind": "text",
            "help": "runaway-generation cap; blank or 0 = uncapped"
        },
        {
            "key": "embedding_model",
            "label": "Embedding model",
            "value": cfg.specialists.embedding_model,
            "kind": "text",
            "help": "local Ollama tag behind memory.* skills"
        },
        {
            "key": "privacy_level",
            "label": "Privacy tier",
            "value": privacy_label(cfg.privacy.level),
            "kind": "select",
            "options": ["full_mesh", "trusted_peers", "local_only"],
            "help": "how far prompt-bearing hops may leave this node"
        },
        {
            "key": "websearch_enabled",
            "label": "Web search",
            "value": cfg.websearch.enabled.to_string(),
            "kind": "select",
            "options": ["true", "false"],
            "help": "Brave search + URL fetch skills in the bot tool loop"
        }
    ])
}

/// Read-only facts shown beside the form.
pub fn readonly_json(cfg: &NodeConfig) -> Value {
    let bind = cfg
        .gateway_bind
        .clone()
        .unwrap_or_else(|| "127.0.0.1".to_string());
    let model = cfg
        .model
        .as_ref()
        .map(|m| format!("{} ({})", m.id, m.quant))
        .unwrap_or_else(|| "not set".to_string());
    json!([
        {"label": "node id", "value": cfg.node_id},
        {"label": "model", "value": model},
        {"label": "gateway", "value": format!("{bind}:{}", cfg.gateway_port)},
        {"label": "memory store", "value": cfg.memory_backend.label()},
        {"label": "config file", "value": NodeConfig::config_path().display().to_string()},
    ])
}

/// Coerce one JSON form value to the string the validator expects.
fn as_text(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

fn bool_of(v: &str) -> anyhow::Result<bool> {
    match v.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "on" | "yes" => Ok(true),
        "false" | "0" | "off" | "no" => Ok(false),
        other => anyhow::bail!("expected true/false, got {other:?}"),
    }
}

fn url_of(key: &str, v: &str) -> anyhow::Result<String> {
    let v = v.trim();
    anyhow::ensure!(!v.is_empty(), "{key}: empty URL");
    anyhow::ensure!(
        v.starts_with("http://") || v.starts_with("https://"),
        "{key}: must start with http:// or https://"
    );
    Ok(v.to_string())
}

fn tier_of(key: &str, v: &str) -> anyhow::Result<Option<String>> {
    let v = v.trim();
    if v.is_empty() || v.eq_ignore_ascii_case("none") {
        return Ok(None);
    }
    anyhow::ensure!(
        !v.chars().any(char::is_whitespace),
        "{key}: model ids carry no spaces"
    );
    Ok(Some(v.to_string()))
}

/// Apply a `{key: value}` map to `cfg`, validating exactly like the CLI.
/// Returns the keys that were applied (in request order). An unknown or
/// read-only key is refused before any value is written, so a partial
/// patch never half-applies.
pub fn apply(
    cfg: &mut NodeConfig,
    changes: &serde_json::Map<String, Value>,
) -> anyhow::Result<Vec<String>> {
    // Validate everything against a clone first — all-or-nothing.
    let mut next = cfg.clone();
    let mut applied = Vec::new();
    for (key, raw) in changes {
        let value =
            as_text(raw).with_context(|| format!("{key}: expected a string, number or boolean"))?;
        match key.as_str() {
            "contribution_percent" => {
                let v: u64 = value
                    .parse()
                    .context("contribution_percent must be a number")?;
                anyhow::ensure!(
                    v >= MIN_CONTRIBUTION_PERCENT,
                    "minimum is {MIN_CONTRIBUTION_PERCENT}%"
                );
                next.contribution_percent = v;
            }
            "memory_max_rows" => {
                let n: i64 = value
                    .parse()
                    .context("memory_max_rows must be a whole number")?;
                anyhow::ensure!(n >= 0, "memory_max_rows must be ≥ 0 (0 = uncapped)");
                next.memory_max_rows = n;
            }
            "bootstrap_url" => next.bootstrap_url = url_of(key, &value)?,
            "runtime_url" => next.runtime_url = url_of(key, &value)?,
            "runtime_primary_model" => next.runtime_primary_model = tier_of(key, &value)?,
            "runtime_deep_model" => next.runtime_deep_model = tier_of(key, &value)?,
            "runtime_vision_model" => next.runtime_vision_model = tier_of(key, &value)?,
            "runtime_tool_model" => next.runtime_tool_model = tier_of(key, &value)?,
            "runtime_num_predict" => {
                let v = value.trim();
                next.runtime_num_predict =
                    if v.is_empty() || v.eq_ignore_ascii_case("none") || v == "0" {
                        None
                    } else {
                        Some(v.parse().context("runtime_num_predict must be a number")?)
                    };
            }
            "embedding_model" => {
                let v = value.trim();
                anyhow::ensure!(!v.is_empty(), "embedding_model: empty model tag");
                anyhow::ensure!(
                    !v.chars().any(char::is_whitespace),
                    "embedding_model: model tags carry no spaces"
                );
                next.specialists.embedding_model = v.to_string();
            }
            "privacy_level" => {
                next.privacy.level = match value.trim().to_ascii_lowercase().as_str() {
                    "full_mesh" => PrivacyLevel::FullMesh,
                    "trusted_peers" => PrivacyLevel::TrustedPeers,
                    "local_only" => PrivacyLevel::LocalOnly,
                    other => anyhow::bail!(
                        "privacy_level must be full_mesh, trusted_peers or local_only (got {other:?})"
                    ),
                };
            }
            "websearch_enabled" => next.websearch.enabled = bool_of(&value)?,
            other => anyhow::bail!(
                "setting {other:?} is unknown or read-only — identity, ports, model and the \
                 memory backend/DSN are changed from the CLI"
            ),
        }
        applied.push(key.clone());
    }
    // Final-state validation, same as the CLI: assignments can fix each
    // other up, so validate once after the loop.
    next.validate().context("config validation")?;
    *cfg = next;
    Ok(applied)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> NodeConfig {
        // A setup-shaped config: `apply` ends with the same `validate()`
        // the CLI runs, and a bare default has no identity/model.
        let mut c = NodeConfig::default();
        c.ensure_identity().expect("identity");
        c.set_model(crate::config::default_model("qwen3:8b"));
        c
    }

    #[test]
    fn settings_and_readonly_lists_are_complete() {
        let c = cfg();
        let settings = settings_json(&c);
        let keys: Vec<&str> = settings
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s["key"].as_str().unwrap())
            .collect();
        for k in EDITABLE {
            assert!(keys.contains(&k), "missing editable setting {k}");
        }
        let ro = readonly_json(&c);
        assert!(!ro.as_array().unwrap().is_empty());
        assert!(ro.to_string().contains("node id"));
    }

    #[test]
    fn apply_validates_like_the_cli() {
        let mut c = cfg();
        let mut m = serde_json::Map::new();
        m.insert("contribution_percent".into(), json!("45"));
        m.insert("memory_max_rows".into(), json!("0"));
        m.insert("privacy_level".into(), json!("local_only"));
        m.insert("websearch_enabled".into(), json!("false"));
        m.insert("runtime_num_predict".into(), json!("256"));
        apply(&mut c, &m).expect("valid patch");
        assert_eq!(c.contribution_percent, 45);
        assert_eq!(c.memory_max_rows, 0);
        assert_eq!(c.privacy.level, PrivacyLevel::LocalOnly);
        assert!(!c.websearch.enabled);
        assert_eq!(c.runtime_num_predict, Some(256));

        // Below the 25% floor is refused, nothing changes.
        let mut m = serde_json::Map::new();
        m.insert("contribution_percent".into(), json!("5"));
        assert!(apply(&mut c, &m).is_err());
        assert_eq!(c.contribution_percent, 45);

        // Blank tier / max tokens map back to "off".
        let mut m = serde_json::Map::new();
        m.insert("runtime_deep_model".into(), json!(""));
        m.insert("runtime_num_predict".into(), json!(""));
        apply(&mut c, &m).unwrap();
        assert_eq!(c.runtime_deep_model, None);
        assert_eq!(c.runtime_num_predict, None);
    }

    #[test]
    fn apply_refuses_unknown_and_read_only_keys() {
        let mut c = cfg();
        for key in ["gateway_port", "node_id", "memory_dsn", "model", "role"] {
            let mut m = serde_json::Map::new();
            m.insert(key.into(), json!("1"));
            let err = apply(&mut c, &m).expect_err("read-only key must be refused");
            assert!(
                err.to_string().contains("unknown or read-only"),
                "{key}: {err}"
            );
        }
    }

    #[test]
    fn apply_is_all_or_nothing() {
        let mut c = cfg();
        let before = c.contribution_percent;
        let mut m = serde_json::Map::new();
        m.insert("contribution_percent".into(), json!("50"));
        m.insert("gateway_port".into(), json!("1"));
        assert!(apply(&mut c, &m).is_err());
        assert_eq!(
            c.contribution_percent, before,
            "a refused patch must not half-apply"
        );
    }
}
