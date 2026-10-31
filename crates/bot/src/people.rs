//! Live Postgres lookup of the `jerith_people` registry (per chat turn).
//!
//! Connection contract mirrors the previous stack's `scripts/shared/pg_pool.py`:
//! `root@127.0.0.1:5432/jerith_people`, the password resolved from
//! `~/.pgpass` (read in-process, never echoed or logged). bot.toml's
//! optional `[postgres]` table overrides every field. The lookup degrades
//! to "no speaker context" when the database is unreachable — a chat must
//! never fail because the people database is down.

use std::pin::Pin;
use std::time::Duration;

use anyhow::Result;
use serde::Deserialize;
use serde_json::json;

use crate::plugins::SkillSpec;

/// Rough per-message context budget for the speaker block (chars).
const SPEAKER_BLOCK_MAX: usize = 2000;

// ---------------------------------------------------------------------------
// people.lookup — the registry lookup the bot was missing
// ---------------------------------------------------------------------------
//
// The per-turn speaker block only describes the person TALKING, and the
// persona's known-users export is capped, so a question about *another*
// member ("what name colour does X use?") had no honest answer path — the
// bot reported "I don't have a tool to look them up". This skill is that
// tool: read-only, bounded, and deliberately narrow about what it returns.
//
// The registry holds operator notes, not a public profile: only
// preference-shaped facts (name colour, pronouns, nickname, …) are
// shareable. Medical notes, real names and contact details never leave the
// database through this skill — see `is_shareable_fact`.

/// Fact-key fragments that describe a *preference* — safe to show a chat.
const PREFERENCE_TOKENS: &[&str] = &[
    "color",
    "colour",
    "pronoun",
    "timezone",
    "language",
    "nickname",
    "emoji",
    "preference",
    "favorite",
    "favourite",
    "style",
    "tone",
    "call_me",
    "calls_me",
    "display",
    // Self-selected roles are a preference the picker itself records.
    "role",
];

/// Fact-key fragments that are NEVER shareable, even when the key also
/// matches a preference token (`real_name`, `pain_meds`, `address`, …).
const SENSITIVE_TOKENS: &[&str] = &[
    "med",
    "real_name",
    "address",
    "password",
    "token",
    "secret",
    "phone",
    "email",
];

/// Per-match preference lines; the total response stays small enough for a
/// chat turn even when a partial name matches several people.
const LOOKUP_MAX_FACTS: usize = 8;
const LOOKUP_MAX_MATCHES: i64 = 5;

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
    /// The local connection contract (pg_pool.py defaults):
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
    let id = raw.trim_end().trim_end_matches([')', ']']).trim();
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

/// Is this profile key safe to show in a chat lookup?
///
/// Allowlist-by-meaning: the key must look like a preference AND must not
/// look like a sensitive note. Anything unrecognized stays private (a
/// missing preference is a fine answer; a leaked medical note is not).
pub fn is_shareable_fact(key: &str) -> bool {
    let k = key.to_ascii_lowercase();
    if SENSITIVE_TOKENS.iter().any(|t| k.contains(t)) {
        return false;
    }
    PREFERENCE_TOKENS.iter().any(|t| k.contains(t))
}

/// The model-facing `people.lookup` skill (read-only registry search).
pub fn skills() -> Vec<SkillSpec> {
    vec![SkillSpec {
        name: "people.lookup",
        plugin: "people",
        description: "Look up a Discord member in the jerith_people registry by id, username or display name (partial names allowed). Returns username, display name, trust and preference facts such as name colour, pronouns or nickname. Read-only. Private notes, medical facts, real names and contacts are never returned; an empty result means the registry has no preference recorded.",
        parameters: json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Discord id, username or display name"
                }
            },
            "required": ["query"]
        }),
        run: lookup_runner,
    }]
}

fn lookup_runner<'a>(
    cfg: &'a crate::config::BotConfig,
    args: &'a serde_json::Value,
) -> Pin<Box<dyn std::future::Future<Output = anyhow::Result<serde_json::Value>> + Send + 'a>> {
    Box::pin(async move {
        let query = args
            .get("query")
            .and_then(|q| q.as_str())
            .map(str::trim)
            .unwrap_or_default();
        anyhow::ensure!(!query.is_empty(), "people.lookup: 'query' is required");
        anyhow::ensure!(
            query.chars().count() <= 100,
            "people.lookup: query is too long"
        );
        let pg = cfg.postgres.as_ref().ok_or_else(|| {
            anyhow::anyhow!("people.lookup: no [postgres] lane configured — registry lookup is off")
        })?;
        let matches = lookup_any(pg, query).await?;
        Ok(json!({
            "matches": matches,
            "found": matches.len(),
            "note": if matches.is_empty() {
                "no jerith_people row matches this query, and no preference is recorded — do not invent one"
            } else {
                "preference facts only; the registry's private notes are not shared"
            },
        }))
    })
}

/// Read-only registry search: exact id / username / display name first,
/// then a partial username/display match. One statement, bounded rows.
async fn lookup_any(cfg: &PgConfig, query: &str) -> Result<Vec<serde_json::Value>> {
    anyhow::ensure!(cfg.enabled, "people.lookup: postgres lookup disabled");
    let conn = cfg
        .conn_string()
        .ok_or_else(|| anyhow::anyhow!("people.lookup: no connection info"))?;
    let rows = tokio::time::timeout(
        Duration::from_secs(cfg.timeout_secs),
        fetch_people(&conn, query),
    )
    .await
    .map_err(|_| anyhow::anyhow!("people.lookup timed out"))??;
    Ok(rows)
}

async fn fetch_people(conn: &str, query: &str) -> Result<Vec<serde_json::Value>> {
    use tokio_postgres::NoTls;

    let (client, connection) =
        tokio::time::timeout(Duration::from_secs(2), tokio_postgres::connect(conn, NoTls))
            .await
            .map_err(|_| anyhow::anyhow!("postgres connect timeout"))?
            .map_err(|e| anyhow::anyhow!("postgres connect failed: {e}"))?;
    let driver = tokio::spawn(async move {
        if let Err(e) = connection.await {
            tracing::debug!(error = %e, "people-lookup connection closed with error");
        }
    });

    let sql = "SELECT p.discord_id, coalesce(p.username,'') AS username, coalesce(NULLIF(p.display_name,''), coalesce(p.username,'')) AS display_name, coalesce(p.trust,'') AS trust, coalesce(p.timezone,'') AS timezone, coalesce((SELECT jsonb_agg(jsonb_build_object('k', up.profile_key, 'v', up.profile_value) ORDER BY up.profile_key) FROM user_profiles up WHERE up.user_id = p.discord_id), '[]'::jsonb)::text AS facts FROM people p WHERE p.discord_id = $1 OR lower(coalesce(p.username,'')) = lower($1) OR lower(coalesce(p.display_name,'')) = lower($1) OR coalesce(p.username,'') ILIKE '%' || $1 || '%' OR coalesce(p.display_name,'') ILIKE '%' || $1 || '%' ORDER BY (p.discord_id = $1) DESC, p.username LIMIT $2";
    let rows = client
        .query(sql, &[&query, &LOOKUP_MAX_MATCHES])
        .await
        .map_err(|e| anyhow::anyhow!("people lookup failed: {e}"))?;

    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let facts_text: String = row.try_get("facts").unwrap_or_else(|_| "[]".into());
        let facts: Vec<serde_json::Value> = serde_json::from_str(&facts_text).unwrap_or_default();
        let preferences: Vec<serde_json::Value> = facts
            .into_iter()
            .filter(|f| {
                f.get("k")
                    .and_then(|k| k.as_str())
                    .map(is_shareable_fact)
                    .unwrap_or(false)
            })
            .filter(|f| f.get("v").and_then(|v| v.as_str()).is_some())
            .take(LOOKUP_MAX_FACTS)
            .collect();
        out.push(json!({
            "discord_id": row.try_get::<_, String>("discord_id").unwrap_or_default(),
            "username": row.try_get::<_, String>("username").unwrap_or_default(),
            "display_name": row.try_get::<_, String>("display_name").unwrap_or_default(),
            "trust": row.try_get::<_, String>("trust").unwrap_or_default(),
            "timezone": row.try_get::<_, String>("timezone").unwrap_or_default(),
            "preferences": preferences,
        }));
    }

    drop(client);
    driver.abort();
    Ok(out)
}

/// Persist a preference row (`user_profiles`) for one member. Narrow,
/// single-statement upsert on the registry's own table — the same shape the
/// legacy picker wrote to the people file, now visible to `people.lookup`
/// and the known-users export.
async fn upsert_profile(
    cfg: &PgConfig,
    discord_id: &str,
    key: &'static str,
    value: &str,
) -> Result<()> {
    anyhow::ensure!(cfg.enabled, "registry persistence disabled");
    let conn = cfg
        .conn_string()
        .ok_or_else(|| anyhow::anyhow!("no connection info"))?;
    tokio::time::timeout(Duration::from_secs(cfg.timeout_secs), async move {
        use tokio_postgres::NoTls;
        let (client, connection) =
            tokio::time::timeout(Duration::from_secs(2), tokio_postgres::connect(&conn, NoTls))
                .await
                .map_err(|_| anyhow::anyhow!("postgres connect timeout"))?
                .map_err(|e| anyhow::anyhow!("postgres connect failed: {e}"))?;
        let driver = tokio::spawn(async move {
            if let Err(e) = connection.await {
                tracing::debug!(error = %e, "registry connection closed with error");
            }
        });
        // The registry's unique (user_id, profile_key) makes this an upsert;
        // no created_at/confidence guessing — those columns carry defaults.
        let sql = "INSERT INTO user_profiles (user_id, profile_key, profile_value, confidence, source, updated_at) VALUES ($1, $2, $3, 1.0, 'jerith-bot', now()) ON CONFLICT (user_id, profile_key) DO UPDATE SET profile_value = EXCLUDED.profile_value, updated_at = now(), source = EXCLUDED.source";
        client
            .execute(sql, &[&discord_id, &key, &value])
            .await
            .map_err(|e| anyhow::anyhow!("profile write failed: {e}"))?;
        drop(client);
        driver.abort();
        Ok::<(), anyhow::Error>(())
    })
    .await
    .map_err(|_| anyhow::anyhow!("profile persistence timed out"))??;
    Ok(())
}

/// The name colour the member just picked (`name_color`).
pub async fn remember_name_color(cfg: &PgConfig, discord_id: &str, value: &str) -> Result<()> {
    upsert_profile(cfg, discord_id, "name_color", value).await
}

/// The self-service roles the member picked (`discord_roles`, comma
/// separated names) — written by roles.rs after each add/remove.
pub async fn remember_roles(cfg: &PgConfig, discord_id: &str, value: &str) -> Result<()> {
    upsert_profile(cfg, discord_id, "discord_roles", value).await
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

    /// The lookup returns preference-shaped facts and refuses everything
    /// personal, even when a key contains both signals.
    #[test]
    fn lookup_shares_preferences_but_never_private_notes() {
        for safe in [
            "color_hex_preference",
            "name_color",
            "pronouns",
            "timezone",
            "nickname",
        ] {
            assert!(is_shareable_fact(safe), "{safe} should be shareable");
        }
        for private in [
            "real_name",
            "pain_meds",
            "address",
            "phone",
            "email",
            "favorite_med",
            "sexuality_note",
            "core_belief",
        ] {
            assert!(!is_shareable_fact(private), "{private} must stay private");
        }
    }

    #[test]
    fn lookup_skill_declares_read_only_narrow_contract() {
        let s = skills();
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].name, "people.lookup");
        assert_eq!(s[0].plugin, "people");
        assert!(s[0].description.contains("never returned"));
        assert_eq!(s[0].parameters["required"], json!(["query"]));
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
