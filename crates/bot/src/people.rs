//! Live Postgres lookup of the `jerith_people` registry (per chat turn).
//!
//! Connection contract mirrors OpenClaw's `scripts/shared/pg_pool.py`:
//! `root@127.0.0.1:5432/jerith_people`, the password resolved from
//! `~/.pgpass` (read in-process, never echoed or logged). bot.toml's
//! optional `[postgres]` table overrides every field. The lookup degrades
//! to "no speaker context" when the database is unreachable — a chat must
//! never fail because the people database is down.

use std::time::Duration;

use anyhow::Result;
use serde::Deserialize;

/// Rough per-message context budget for the speaker block (chars).
const SPEAKER_BLOCK_MAX: usize = 2000;

#[derive(Debug, Clone, Deserialize)]
pub struct PgConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default = "default_user")]
    pub user: String,
    #[serde(default = "default_database")]
    pub database: String,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
}

fn default_host() -> String {
    "127.0.0.1".into()
}
fn default_port() -> u16 {
    5432
}
fn default_user() -> String {
    "root".into()
}
fn default_database() -> String {
    "jerith_people".into()
}
fn default_enabled() -> bool {
    true
}
fn default_timeout_secs() -> u64 {
    3
}

impl Default for PgConfig {
    /// The OpenClaw-stack connection contract (pg_pool.py defaults):
    /// root@127.0.0.1:5432/jerith_people, auth via ~/.pgpass.
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            user: default_user(),
            database: default_database(),
            enabled: default_enabled(),
            timeout_secs: default_timeout_secs(),
        }
    }
}

impl PgConfig {
    /// libpq key=value connection string, password from ~/.pgpass.
    fn conn_string(&self) -> Option<String> {
        let password = pgpass_password(&self.host, self.port, &self.database, &self.user);
        Some(format!(
            "host={} port={} user={} dbname={}{}",
            self.host,
            self.port,
            self.user,
            self.database,
            password
                .map(|p| format!(" password={p}"))
                .unwrap_or_default()
        ))
    }
}

/// Extract the tagged speaker's Discord id from our own controlled prompt
/// format: a leading `[Discord user: ... (username ..., id ...)]` line.
pub fn extract_speaker_id(prompt: &str) -> Option<String> {
    let tag = prompt.lines().next()?;
    let start = tag.find("(username ")?;
    let rest = tag[start..].strip_prefix("(username ")?;
    let id_at = rest.rfind(", id ")?;
    let raw = &rest[id_at + ", id ".len()..];
    // The tag always ends "id ...)]" — strip both closing chars (a bare
    // find(']') slice keeps the ')' and fails the digit check).
    let id = raw
        .trim_end()
        .trim_end_matches(|c| c == ')' || c == ']')
        .trim();
    let ok = !id.is_empty() && id.chars().all(|c| c.is_ascii_digit());
    ok.then(|| id.to_string())
}

/// `~/.pgpass` lookup honoring wildcards, exactly like libpq: rows are
/// `host:port:database:user:password`; `*` matches any field. Passwords
/// are only ever returned to the caller into the connection string.
pub fn pgpass_password(host: &str, port: u16, database: &str, user: &str) -> Option<String> {
    let path = pgpass_path();
    let text = std::fs::read_to_string(path).ok()?;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.splitn(5, ':').collect();
        if f.len() != 5 {
            continue;
        }
        let (fh, fp, fd, fu, pw) = (f[0], f[1], f[2], f[3], f[4]);
        if (fh == "*" || fh == host)
            && (fp == "*" || fp == port.to_string())
            && (fd == "*" || fd == database)
            && (fu == "*" || fu == user)
        {
            return Some(pw.to_string());
        }
    }
    None
}

fn pgpass_path() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    std::path::PathBuf::from(home).join(".pgpass")
}

/// One-query lookup: the `people` row + its `user_profiles` facts, aggregated
/// server-side as JSON-as-text (no extra serde features needed).
async fn fetch_row(conn: &str, discord_id: &str) -> Result<Option<String>> {
    use tokio_postgres::NoTls;

    let (client, connection) =
        tokio::time::timeout(Duration::from_secs(2), tokio_postgres::connect(conn, NoTls))
            .await
            .map_err(|_| anyhow::anyhow!("postgres connect timeout"))?
            .map_err(|e| anyhow::anyhow!("postgres connect failed: {e}"))?;

    // Drive the connection in the background; dropped when this task ends.
    let driver = tokio::spawn(async move {
        if let Err(e) = connection.await {
            tracing::debug!(error = %e, "people-db connection closed with error");
        }
    });

    // Single line on purpose: Rust's `\`-newline continuations swallow the
    // next line's indentation, and a missing trailing space then glues SQL
    // keywords together (this bit us once: `factsFROM`).
    let sql = "SELECT to_jsonb(t)::text FROM (SELECT p.discord_id, p.username, coalesce(NULLIF(p.display_name,''), p.username) AS display_name, p.trust, p.timezone, p.bio, f.facts FROM people p LEFT JOIN LATERAL (SELECT jsonb_agg(jsonb_build_object('k', up.profile_key, 'v', up.profile_value) ORDER BY up.profile_key) AS facts FROM user_profiles up WHERE up.user_id = p.discord_id) f ON true WHERE p.discord_id = $1) t";
    let row = client.query_opt(sql, &[&discord_id]).await.map_err(|e| {
        let detail = e
            .as_db_error()
            .map(|d| format!(" | pg: {}", d.message()))
            .unwrap_or_default();
        anyhow::anyhow!("people query failed: {e}{detail}")
    })?;

    let value = row
        .as_ref()
        .and_then(|r| r.try_get::<_, String>(0).ok())
        .map(|text| text.to_string());

    drop(client);
    driver.abort();
    Ok(value)
}

/// Compose a compact "about the speaker" system block from the registry.
/// `Ok(None)` = speaker is not in the database (fine — they're a guest).
pub async fn lookup(cfg: &PgConfig, discord_id: &str) -> Result<Option<String>> {
    let raw = match tokio::time::timeout(
        Duration::from_secs(cfg.timeout_secs),
        fetch_row(
            &cfg.conn_string()
                .ok_or_else(|| anyhow::anyhow!("no connection info"))?,
            discord_id,
        ),
    )
    .await
    .map_err(|_| anyhow::anyhow!("people lookup timed out"))?
    {
        Ok(r) => r,
        Err(e) => return Err(e),
    };
    let Some(text) = raw else { return Ok(None) };
    let v: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| anyhow::anyhow!("people row not JSON: {e}"))?;
    Ok(Some(format_speaker_block(&v)))
}

/// Pure block formatting (tests cover without a database).
pub fn format_speaker_block(v: &serde_json::Value) -> String {
    let s = |opt: Option<&serde_json::Value>| -> String {
        opt.and_then(|x| x.as_str())
            .map(|x| x.trim().to_string())
            .unwrap_or_default()
    };
    let mut out = String::from("## About this speaker (live from the jerith_people database)\n");
    let uname = s(v.get("username"));
    let display = s(v.get("display_name"));
    let name_piece = if uname.is_empty() && display.is_empty() {
        "username: (not set in the registry — add one in jerith_people.people)".to_string()
    } else if display.is_empty() || display == uname {
        format!("username: {uname}")
    } else {
        format!("username: {uname} • display: {display}")
    };
    let mut head: Vec<String> = Vec::new();
    head.push(name_piece);
    for field in ["trust", "timezone"] {
        let val = s(v.get(field));
        if !val.is_empty() {
            head.push(format!("{field}: {val}"));
        }
    }
    let id = s(v.get("discord_id"));
    if !id.is_empty() {
        head.push(format!("discord_id: {id}"));
    }
    out.push_str(&head.join(" • "));
    if !s(v.get("bio")).is_empty() {
        out.push_str(&format!("\nbio: {}", s(v.get("bio"))));
    }
    if let Some(facts) = v.get("facts").and_then(|f| f.as_array()) {
        let lines: Vec<String> = facts
            .iter()
            .filter_map(|f| {
                let k = s(f.get("k"));
                let val = s(f.get("v"));
                (!k.is_empty() && !val.is_empty()).then(|| format!("- {k}: {val}"))
            })
            .take(12)
            .collect();
        if !lines.is_empty() {
            out.push_str("\nprofile facts:\n");
            out.push_str(&lines.join("\n"));
        }
    }
    cap_block(&out, SPEAKER_BLOCK_MAX)
}

/// Keep the head (identity first) — same convention as the persona cap.
pub fn cap_block(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let head: String = text.chars().take(max).collect();
    format!("{head}\n…(speaker context truncated)")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extract_speaker_id_from_our_tag() {
        let p = "[Discord user: NoodlyPanda (username noodly_h, id 285471197825859584)]\nhi";
        eprintln!("DBG first-line: {:?}", p.lines().next());
        eprintln!("DBG extract: {:?}", extract_speaker_id(p));
        assert_eq!(
            extract_speaker_id(p),
            Some("285471197825859584".to_string())
        );
        assert_eq!(extract_speaker_id("no tag here"), None);
        assert_eq!(
            extract_speaker_id("[Discord user: (username x, id abc)]"),
            None
        );
    }

    #[test]
    fn format_speaker_block_renders_registry_row() {
        let row = json!({
            "discord_id": "202580937261711375",
            "username": "lemonaid",
            "display_name": "lemonaid",
            "trust": "Trusted (NP's husband)",
            "timezone": null,
            "bio": null,
            "facts": [
                {"k": "relationship_to_np", "v": "Husband"}
            ]
        });
        let block = format_speaker_block(&row);
        assert!(block.starts_with("## About this speaker"));
        assert!(block.contains("username: lemonaid"));
        assert!(block.contains("trust: Trusted (NP's husband)"));
        assert!(block.contains("- relationship_to_np: Husband"));
        assert!(!block.contains("timezone")); // null fields skipped
        assert!(!block.contains("bio:")); // null bio skipped
    }

    #[test]
    fn format_speaker_block_handles_unset_username_and_id() {
        let row = json!({"discord_id": "285471197825859584", "username": null,
                         "display_name": null, "trust": "Unknown",
                         "facts": []});
        let block = format_speaker_block(&row);
        assert!(block.contains("username: (not set in the registry"));
        assert!(block.contains("discord_id: 285471197825859584"));
        assert!(block.contains("trust: Unknown"));
    }

    /// Live smoke: real jerith_people lookup against the operator PG.
    /// Skipped unless JERITH_HB_PG_SMOKE=1 (CI has no database).
    #[tokio::test]
    async fn live_lookup_smoke() {
        if std::env::var("JERITH_HB_PG_SMOKE").is_err() {
            return;
        }
        let cfg = PgConfig::default();
        let block = lookup(&cfg, "285471197825859584").await.unwrap();
        assert!(
            block.is_some(),
            "owner row should resolve from the live jerith_people database"
        );
        let block = block.unwrap();
        assert!(
            block.contains("discord_id: 285471197825859584"),
            "speaker block must carry the owner's discord id"
        );
    }

    #[test]
    fn cap_block_keeps_head_and_marks() {
        let long = "x".repeat(3000);
        let capped = cap_block(&long, 2000);
        assert_eq!(
            capped.chars().count(),
            2000 + "\n…(speaker context truncated)".chars().count()
        );
        assert!(capped.starts_with("xxx"));
        assert_eq!(cap_block("short", 2000), "short");
    }
}
