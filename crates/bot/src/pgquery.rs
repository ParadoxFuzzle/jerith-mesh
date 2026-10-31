//! `pg.query` — read-only PostgreSQL lookups on demand (owner-DM only).
//!
//! The bot can inspect the operator's own local databases (`jerith`,
//! `community_memory`, `self_improving`, `jerith_people`, `jerith_mesh` by
//! default) without shelling out to `psql`. Four independent gates keep it
//! read-only:
//!
//! 1. exactly one statement, and its first keyword must be `SELECT`/`WITH`;
//!    every write/DDL/control keyword is refused before a connection opens
//!    (string literals and quoted identifiers are stripped first, so a
//!    stray word in data cannot trip — or hide — the scan);
//! 2. the query runs inside `BEGIN TRANSACTION READ ONLY`, so the server
//!    itself refuses any write the scanner might miss;
//! 3. `statement_timeout` bounds wall time, and a hard row cap + response
//!    size cap bound what comes back to the model;
//! 4. the database must be an exact member of `[pgquery] databases`; the
//!    password comes from `~/.pgpass` (the local `pg_pool.py` contract),
//!    never from bot.toml.
//!
//! The skill is owner-only: `pgquery` sits in
//! [`crate::plugins::OWNER_ONLY_PLUGINS`], so a plain guild chat / guest DM
//! never sees the declaration and can never dispatch it.

use std::pin::Pin;
use std::time::Duration;

use anyhow::Context as _;
use serde_json::{Value, json};

use crate::plugins::SkillSpec;

/// Hard row ceiling for one call (the model may ask for less, never more).
const MAX_ROWS_CEILING: usize = 200;

/// Response-size ceiling: rows are dropped from the tail until the JSON
/// under this cap, so one wide row cannot flood the model's context.
const RESULT_CAP_BYTES: usize = 256 * 1024;

/// `[pgquery]` in bot.toml. Absent = defaults below (loopback, `root`,
/// `~/.pgpass`, the local databases + the node's own pgvector store).
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
pub struct PgQueryConfig {
    /// Switches the skill off without removing the plugin family.
    pub enabled: bool,
    pub host: String,
    pub port: u16,
    pub user: String,
    /// Exact database allowlist — a name absent here is unreachable.
    pub databases: Vec<String>,
    /// Default row cap when the model does not pass `max_rows`.
    pub max_rows: usize,
    /// Connect + statement budget.
    pub timeout_secs: u64,
}

impl Default for PgQueryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            host: "127.0.0.1".into(),
            port: 5432,
            user: "root".into(),
            databases: [
                "jerith",
                "community_memory",
                "self_improving",
                "jerith_people",
                "jerith_mesh",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
            max_rows: 50,
            timeout_secs: 5,
        }
    }
}

/// The model-facing skill declaration (joined only by an owner-privileged
/// registry — see `plugins.rs`).
pub fn skills() -> Vec<SkillSpec> {
    vec![SkillSpec {
        name: "pg.query",
        plugin: "pgquery",
        description: "Run ONE read-only SQL statement (SELECT or WITH) against a local Postgres database the operator owns, and return the rows as JSON. Writes, DDL and multi-statement input are refused. Use to look up existing tables/data on demand instead of guessing.",
        parameters: json!({
            "type": "object",
            "properties": {
                "database": {
                    "type": "string",
                    "description": "Database name (default \"jerith\"). Only [pgquery] databases are reachable."
                },
                "sql": {
                    "type": "string",
                    "description": "A single SELECT/WITH statement (no trailing second statement)"
                },
                "max_rows": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 200,
                    "description": "Row cap for this call (default [pgquery] max_rows, 50)"
                }
            },
            "required": ["sql"]
        }),
        run: runner,
    }]
}

fn runner<'a>(
    cfg: &'a crate::config::BotConfig,
    args: &'a Value,
) -> Pin<Box<dyn std::future::Future<Output = anyhow::Result<Value>> + Send + 'a>> {
    Box::pin(async move {
        let sql = args.get("sql").and_then(Value::as_str).unwrap_or_default();
        anyhow::ensure!(!sql.trim().is_empty(), "pg.query: empty sql");
        let database = args
            .get("database")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|d| !d.is_empty())
            .unwrap_or("jerith");
        let max_rows = args
            .get("max_rows")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(cfg.pgquery.max_rows)
            .clamp(1, MAX_ROWS_CEILING);
        run_query(&cfg.pgquery, database, sql, max_rows).await
    })
}

/// Keywords that mean "not a read". Multi-word constructs (`FOR UPDATE`,
/// `SELECT INTO`) are covered by `into`/the read-only transaction; `lock`
/// and `set` are here because they are also valid standalone statements.
const FORBIDDEN: &[&str] = &[
    "insert",
    "update",
    "delete",
    "merge",
    "drop",
    "alter",
    "create",
    "truncate",
    "grant",
    "revoke",
    "copy",
    "vacuum",
    "analyze",
    "reindex",
    "cluster",
    "refresh",
    "comment",
    "call",
    "do",
    "set",
    "reset",
    "lock",
    "begin",
    "commit",
    "rollback",
    "savepoint",
    "release",
    "listen",
    "notify",
    "unlisten",
    "prepare",
    "execute",
    "deallocate",
    "discard",
    "checkpoint",
    "import",
    "into",
    "nextval",
    "setval",
    "pg_sleep",
    "pg_read_file",
    "pg_read_binary_file",
    "pg_ls_dir",
    "lo_import",
    "lo_export",
    "dblink",
    "pg_terminate_backend",
    "pg_cancel_backend",
    "pg_reload_conf",
    "pg_rotate_logfile",
    "pg_switch_wal",
    "pg_promote",
    "pg_create_restore_point",
];

/// Remove string literals, quoted identifiers and comments before keyword
/// scanning — `SELECT 'drop table' AS note` is data, not a write, and a
/// `-- drop` comment must not hide the real statement either.
fn strip_noise(sql: &str) -> String {
    let mut out = String::with_capacity(sql.len());
    let mut chars = sql.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\'' | '"' => {
                let quote = c;
                out.push(' ');
                while let Some(d) = chars.next() {
                    if d == quote {
                        // SQL escapes a quote by doubling it.
                        if quote == '\'' && chars.peek() == Some(&'\'') {
                            chars.next();
                            continue;
                        }
                        break;
                    }
                }
            }
            '-' if chars.peek() == Some(&'-') => {
                for d in chars.by_ref() {
                    if d == '\n' {
                        out.push('\n');
                        break;
                    }
                }
            }
            '/' if chars.peek() == Some(&'*') => {
                chars.next();
                let mut prev = '\0';
                for d in chars.by_ref() {
                    if prev == '*' && d == '/' {
                        break;
                    }
                    prev = d;
                }
                out.push(' ');
            }
            _ => out.push(c),
        }
    }
    out
}

/// Validate + normalize one statement. Returns the statement with a single
/// trailing semicolon removed; anything else is a typed refusal.
pub fn validate_read_only(sql: &str) -> anyhow::Result<String> {
    let trimmed = sql.trim().trim_end_matches(';').trim_end();
    anyhow::ensure!(!trimmed.is_empty(), "pg.query: empty sql");
    let clean = strip_noise(trimmed).to_ascii_lowercase();
    // Comment stripping can leave the separator behind; any remaining
    // semicolon means a second statement was smuggled in.
    anyhow::ensure!(
        !clean.contains(';'),
        "pg.query: exactly one statement is allowed (found ';')"
    );
    let first = clean.split_whitespace().next().unwrap_or("");
    anyhow::ensure!(
        first == "select" || first == "with",
        "pg.query: only SELECT/WITH statements are allowed (got {first:?})"
    );
    let tokens: Vec<&str> = clean
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .filter(|t| !t.is_empty())
        .collect();
    for bad in FORBIDDEN {
        anyhow::ensure!(
            !tokens.iter().any(|t| t == bad),
            "pg.query: {bad:?} is not allowed — this skill is read-only"
        );
    }
    Ok(trimmed.to_string())
}

async fn run_query(
    cfg: &PgQueryConfig,
    database: &str,
    sql: &str,
    max_rows: usize,
) -> anyhow::Result<Value> {
    anyhow::ensure!(
        cfg.enabled,
        "pg.query: disabled ([pgquery] enabled = false)"
    );
    anyhow::ensure!(
        cfg.databases.iter().any(|d| d == database),
        "pg.query: database {database:?} is not allowed — [pgquery] databases = [{}]",
        cfg.databases.join(", ")
    );
    let sql = validate_read_only(sql)?;

    let mut pc = tokio_postgres::Config::new();
    pc.host(&cfg.host)
        .port(cfg.port)
        .user(&cfg.user)
        .dbname(database)
        .application_name("jerith-pgquery");
    if let Some(pw) = crate::people::pgpass_password(&cfg.host, cfg.port, database, &cfg.user) {
        pc.password(pw);
    }
    let timeout = Duration::from_secs(cfg.timeout_secs.max(1));
    let (client, connection) = tokio::time::timeout(timeout, pc.connect(tokio_postgres::NoTls))
        .await
        .map_err(|_| anyhow::anyhow!("pg.query: connect to {database} timed out"))?
        .with_context(|| {
            format!(
                "pg.query: connect to {database} at {}:{} failed",
                cfg.host, cfg.port
            )
        })?;
    let driver = tokio::spawn(async move {
        let _ = connection.await;
    });

    let millis = cfg.timeout_secs.max(1) * 1000;
    let outcome = async {
        client
            .batch_execute(&format!(
                "BEGIN TRANSACTION READ ONLY; SET LOCAL statement_timeout = {millis}"
            ))
            .await
            .context("pg.query: cannot start a read-only transaction")?;
        // Server-side serialization: row_to_json renders ANY column type
        // (jsonb, arrays, numerics, timestamps) without a per-type table.
        let wrapped = format!(
            "SELECT coalesce(json_agg(row_to_json(q))::text, '[]') \
             FROM (SELECT * FROM ({sql}) AS _jerith_q LIMIT {max_rows}) AS q"
        );
        let row = client.query_one(&wrapped, &[]).await.map_err(|e| {
            let detail = e
                .as_db_error()
                .map(|d| d.message().to_string())
                .unwrap_or_else(|| e.to_string());
            anyhow::anyhow!("pg.query: {detail}")
        })?;
        let text = row
            .get::<_, Option<String>>(0)
            .unwrap_or_else(|| "[]".into());
        serde_json::from_str::<Value>(&text).context("pg.query: result was not JSON")
    }
    .await;
    // Always close the read-only transaction before dropping the client.
    let _ = client.batch_execute("ROLLBACK").await;
    driver.abort();

    let parsed = outcome?;
    let mut rows: Vec<Value> = match parsed {
        Value::Array(a) => a,
        other => vec![other],
    };
    let mut truncated = rows.len() >= max_rows;
    while !rows.is_empty()
        && serde_json::to_string(&rows).map(|s| s.len()).unwrap_or(0) > RESULT_CAP_BYTES
    {
        rows.pop();
        truncated = true;
    }
    let count = rows.len();
    Ok(json!({
        "database": database,
        "rows": rows,
        "count": count,
        "truncated": truncated,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_select_and_cte() {
        assert!(validate_read_only("SELECT * FROM memories LIMIT 5").is_ok());
        assert!(validate_read_only("select id, topic from memories;").is_ok());
        assert!(validate_read_only("WITH x AS (SELECT 1 AS n) SELECT * FROM x").is_ok());
        // Word-boundary discipline: `created_at` must not read as `create`.
        assert!(validate_read_only("select created_at from memories").is_ok());
    }

    #[test]
    fn refuses_writes_ddl_and_multi_statement() {
        for sql in [
            "INSERT INTO memories(text) VALUES ('x')",
            "update memories set text='x'",
            "delete from memories",
            "drop table memories",
            "select 1; drop table memories",
            "select 1; select 2",
            "/* hi */ truncate memories",
            "select * into copy_of from memories",
            "select pg_sleep(10)",
        ] {
            assert!(validate_read_only(sql).is_err(), "must refuse: {sql}");
        }
        assert!(validate_read_only("   ").is_err());
    }

    #[test]
    fn cte_cannot_smuggle_a_write() {
        let err = validate_read_only(
            "WITH gone AS (DELETE FROM memories RETURNING *) SELECT * FROM gone",
        )
        .expect_err("write inside a CTE must be refused");
        assert!(err.to_string().contains("read-only"), "{err}");
    }

    #[test]
    fn literals_and_comments_are_not_scanned_as_keywords() {
        // Data says "drop table"; the statement is a plain read.
        assert!(validate_read_only("select 'drop table x' as note").is_ok());
        // A comment mentioning a write must not hide a real read.
        assert!(validate_read_only("select 1 -- delete everything later\n").is_ok());
    }

    #[test]
    fn default_allowlist_covers_the_operator_databases() {
        let cfg = PgQueryConfig::default();
        for db in [
            "jerith",
            "community_memory",
            "self_improving",
            "jerith_people",
            "jerith_mesh",
        ] {
            assert!(cfg.databases.iter().any(|d| d == db), "{db} reachable");
        }
        assert!(!cfg.databases.iter().any(|d| d == "postgres"));
    }

    #[tokio::test]
    async fn refuses_a_database_outside_the_allowlist_before_connecting() {
        let cfg = PgQueryConfig::default();
        let err = run_query(&cfg, "postgres", "select 1", 5)
            .await
            .expect_err("postgres is not on the allowlist");
        assert!(err.to_string().contains("not allowed"), "{err}");
    }

    #[tokio::test]
    async fn refuses_a_disabled_skill_before_connecting() {
        let cfg = PgQueryConfig {
            enabled: false,
            ..PgQueryConfig::default()
        };
        let err = run_query(&cfg, "jerith", "select 1", 5)
            .await
            .expect_err("disabled");
        assert!(err.to_string().contains("disabled"), "{err}");
    }

    /// Live round-trip against the operator's real database (env-gated, like
    /// the core memory PG tests — the reason is printed when it skips):
    /// `JERITH_TEST_PGQUERY=1 cargo test -p jerith-mesh-bot pgquery -- --include-ignored`
    #[tokio::test]
    #[ignore]
    async fn live_query_reads_real_rows_json() {
        if std::env::var("JERITH_TEST_PGQUERY").ok().as_deref() != Some("1") {
            eprintln!("skip live_query_reads_real_rows_json: set JERITH_TEST_PGQUERY=1");
            return;
        }
        let cfg = PgQueryConfig::default();
        let out = run_query(
            &cfg,
            "jerith",
            "select count(*)::bigint as n from memories",
            5,
        )
        .await
        .expect("live read-only query");
        assert_eq!(out["database"], json!("jerith"));
        assert!(out["rows"][0]["n"].as_i64().unwrap_or(0) > 0, "{out}");
        // The same connection, a write: refused before SQL reaches the DB.
        let err = run_query(&cfg, "jerith", "delete from memories", 5)
            .await
            .expect_err("write refused");
        assert!(err.to_string().contains("only SELECT/WITH"), "{err}");
    }
}
