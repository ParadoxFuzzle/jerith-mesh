//! `jerith-reddit` — the reddit lane's wire glue, over the plugin SDK
//! (`crates/plugin-template`, docs/PLUGIN-SDK.md; the SDK's seventh live
//! consumer). The engine (credentials, URL shapes, parsing, outbox, watch
//! state, cards) is in `lib.rs` — pure and tested. This binary:
//!
//! - reads its OWN `plugin.toml` for `[plugin] endpoint` + `secret` and
//!   binds that exact address (manifest and service cannot drift);
//! - holds ONE OAuth token (refreshed a minute before Reddit's expiry)
//!   and ONE outbox of unconfirmed writes;
//! - serves the nine model-visible skills the manifest declares:
//!   `reddit.browse`, `.search`, `.thread` (reads), `.submit`,
//!   `.comment`, `.edit` (two-step preview → confirm), `.vote`, `.watch`
//!   (the lookout cursor the bot's monitor sweeps), `.about`;
//! - never logs a token, a password, or a client secret. A failure names
//!   the operation and Reddit's own error code, nothing sensitive.
//!
//! Why the two-step rail is here and not only in the prompt: a prompt is
//! advice. A proposal id + token that the plugin itself mints and spends
//! once is a gate — no single call can post, no replay can post twice.

use anyhow::{Context as _, Result};
use jerith_plugin_template::{SkillServer, endpoint_addr, skill};
use jerith_reddit_plugin::{
    Credentials, Outbox, PENDING_TTL_MS, Post, clamp_limit, comment_form, edit_form, form_errors,
    fullname, listing_card, listing_url, load_watch, normalize_sub, parse_listing_value,
    parse_thread_value, parse_token, post_id_from, post_json, save_watch, search_url,
    select_matches, sort_or_default, submit_form, thread_card, thread_url, vote_dir, vote_form,
    window_or_default, write_result,
};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn now_secs() -> f64 {
    now_ms() as f64 / 1000.0
}

/// Clip by characters (never mid-codepoint) for card text.
fn clip(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        return s.to_string();
    }
    let mut out: String = s.chars().take(n.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// Reddit's `x-ratelimit-*` headroom on the last call — reported by
/// `reddit.about` so "are we about to be throttled?" is answerable.
#[derive(Debug, Clone, Default)]
struct Rate {
    remaining: Option<f64>,
    used: Option<f64>,
    reset_secs: Option<u64>,
}

impl Rate {
    fn from_headers(h: &reqwest::header::HeaderMap) -> Self {
        let num = |k: &str| {
            h.get(k)
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.trim().parse::<f64>().ok())
        };
        Self {
            remaining: num("x-ratelimit-remaining"),
            used: num("x-ratelimit-used"),
            reset_secs: h
                .get("x-ratelimit-reset")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.trim().parse::<u64>().ok()),
        }
    }
    fn to_json(&self) -> Value {
        json!({
            "remaining": self.remaining,
            "used": self.used,
            "reset_secs": self.reset_secs,
        })
    }
}

/// The service's whole state: one HTTP client, the credential answer
/// (Err = exactly what's missing), the cached token, the write outbox,
/// and the watch file path.
struct Reddit {
    http: reqwest::Client,
    creds: std::result::Result<Credentials, String>,
    token: tokio::sync::Mutex<Option<(String, u64)>>,
    outbox: std::sync::Mutex<Outbox>,
    watch_file: PathBuf,
}

impl Reddit {
    fn new() -> Result<Self> {
        let http = reqwest::Client::builder()
            .use_rustls_tls()
            .timeout(std::time::Duration::from_secs(20))
            .build()
            .context("reddit http client build")?;
        let creds = Credentials::load();
        Ok(Self {
            http,
            creds,
            token: tokio::sync::Mutex::new(None),
            outbox: std::sync::Mutex::new(Outbox::new()),
            watch_file: jerith_reddit_plugin::watch_path(),
        })
    }

    fn creds(&self) -> std::result::Result<&Credentials, String> {
        self.creds.as_ref().map_err(Clone::clone)
    }

    fn outbox(&self) -> std::result::Result<std::sync::MutexGuard<'_, Outbox>, String> {
        self.outbox
            .lock()
            .map_err(|_| "the proposal store is poisoned — restart the reddit service".to_string())
    }

    /// OAuth access token, cached until a minute before it expires.
    async fn access_token(&self) -> std::result::Result<String, String> {
        let creds = self.creds()?;
        creds.validate()?;
        {
            let guard = self.token.lock().await;
            if let Some((t, expires)) = guard.as_ref()
                && now_ms() + 60_000 < *expires
            {
                return Ok(t.clone());
            }
        }
        let form: std::collections::HashMap<String, String> =
            creds.token_form().into_iter().collect();
        let resp = self
            .http
            .post(jerith_reddit_plugin::TOKEN_URL)
            .header("authorization", creds.basic_auth_header())
            .header("user-agent", &creds.user_agent)
            .form(&form)
            .send()
            .await
            .map_err(|e| format!("reddit's token endpoint is unreachable: {e}"))?;
        let status = resp.status();
        let body = resp
            .text()
            .await
            .map_err(|e| format!("reddit's token reply could not be read: {e}"))?;
        if !status.is_success() {
            // Reddit sends {"error": "invalid_grant"} with a 401; that
            // parse gives the operator the actual fix.
            return Err(parse_token(&body)
                .err()
                .unwrap_or_else(|| format!("reddit's token endpoint answered {status}")));
        }
        let reply = parse_token(&body)?;
        let expires = now_ms() + reply.expires_in.max(60) * 1000;
        let token = reply.access_token.clone();
        *self.token.lock().await = Some((token.clone(), expires));
        tracing::info!(
            mode = creds.auth_mode().as_str(),
            expires_in = reply.expires_in,
            "reddit token acquired"
        );
        Ok(token)
    }

    async fn get(&self, url: &str) -> std::result::Result<(Value, Rate), String> {
        let token = self.access_token().await?;
        let creds = self.creds()?;
        let resp = self
            .http
            .get(url)
            .bearer_auth(token)
            .header("user-agent", &creds.user_agent)
            .send()
            .await
            .map_err(|e| format!("reddit unreachable: {e}"))?;
        let status = resp.status();
        let rate = Rate::from_headers(resp.headers());
        let body = resp
            .text()
            .await
            .map_err(|e| format!("reddit's reply could not be read: {e}"))?;
        if status.as_u16() == 429 {
            return Err(format!(
                "reddit rate limit hit — retry in {}s (OAuth allows ~600 calls / 10 min)",
                rate.reset_secs.unwrap_or(60)
            ));
        }
        if !status.is_success() {
            return Err(format!(
                "reddit answered {status}: {}",
                clip(body.trim(), 220)
            ));
        }
        let v: Value = serde_json::from_str(&body).map_err(|e| {
            format!(
                "reddit's reply is not JSON ({e}): {}",
                clip(body.trim(), 160)
            )
        })?;
        Ok((v, rate))
    }

    async fn post_form(
        &self,
        path: &str,
        form: Vec<(String, String)>,
    ) -> std::result::Result<(Value, Rate), String> {
        let token = self.access_token().await?;
        let creds = self.creds()?;
        let url = format!("{}{path}", creds.base());
        let fields: std::collections::HashMap<String, String> = form.into_iter().collect();
        let resp = self
            .http
            .post(url)
            .bearer_auth(token)
            .header("user-agent", &creds.user_agent)
            .form(&fields)
            .send()
            .await
            .map_err(|e| format!("reddit unreachable: {e}"))?;
        let status = resp.status();
        let rate = Rate::from_headers(resp.headers());
        let body = resp
            .text()
            .await
            .map_err(|e| format!("reddit's reply could not be read: {e}"))?;
        if status.as_u16() == 429 {
            return Err(format!(
                "reddit rate limit hit — retry in {}s",
                rate.reset_secs.unwrap_or(60)
            ));
        }
        if !status.is_success() {
            return Err(format!(
                "reddit answered {status}: {}",
                clip(body.trim(), 220)
            ));
        }
        // /api/vote answers an empty 200 — an empty body is success, not
        // a parse failure.
        if body.trim().is_empty() {
            return Ok((Value::Null, rate));
        }
        let v: Value = serde_json::from_str(&body).map_err(|e| {
            format!(
                "reddit's reply is not JSON ({e}): {}",
                clip(body.trim(), 160)
            )
        })?;
        Ok((v, rate))
    }
}

// ---------------------------------------------------------------------------
// Reads
// ---------------------------------------------------------------------------

async fn do_browse(r: Arc<Reddit>, args: Value) -> std::result::Result<Value, String> {
    let sub = normalize_sub(args["subreddit"].as_str().unwrap_or(""));
    if sub.is_empty() {
        return Err("which subreddit? pass a name (\"all\" for the site-wide feed)".into());
    }
    let sort = sort_or_default(args["sort"].as_str().unwrap_or(""));
    let window = window_or_default(args["window"].as_str().unwrap_or(""));
    let limit = clamp_limit(args["limit"].as_u64().unwrap_or(10));
    let after = args["after"]
        .as_str()
        .map(str::trim)
        .filter(|a| !a.is_empty())
        .map(str::to_string);
    let url = listing_url(
        &r.creds()?.base(),
        &sub,
        sort,
        limit,
        window,
        after.as_deref(),
    );
    let (body, rate) = r.get(&url).await?;
    let page = parse_listing_value(&body)?;
    let now = now_secs();
    Ok(json!({
        "subreddit": sub,
        "sort": sort,
        "count": page.posts.len(),
        "after": page.after,
        "posts": page.posts.iter().map(|p| post_json(p, now)).collect::<Vec<_>>(),
        "card": listing_card(&sub, &page.posts, now),
        "rate": rate.to_json(),
    }))
}

async fn do_search(r: Arc<Reddit>, args: Value) -> std::result::Result<Value, String> {
    let query = args["query"].as_str().unwrap_or("").trim();
    if query.is_empty() {
        return Err("a search query is required".into());
    }
    let sub = normalize_sub(args["subreddit"].as_str().unwrap_or(""));
    let sort = sort_or_default(args["sort"].as_str().unwrap_or("relevance"));
    let sort = if sort == "hot" || sort == "rising" || sort == "controversial" {
        "relevance"
    } else {
        sort
    };
    let window = window_or_default(args["window"].as_str().unwrap_or(""));
    let limit = clamp_limit(args["limit"].as_u64().unwrap_or(10));
    let url = search_url(
        &r.creds()?.base(),
        Some(sub.as_str()).filter(|s| !s.is_empty()),
        query,
        sort,
        limit,
        window,
    );
    let (body, rate) = r.get(&url).await?;
    let page = parse_listing_value(&body)?;
    let now = now_secs();
    let label = format!(
        "{} — search “{}”",
        if sub.is_empty() {
            "all".to_string()
        } else {
            sub.clone()
        },
        clip(query, 60)
    );
    Ok(json!({
        "query": query,
        "subreddit": sub,
        "sort": sort,
        "count": page.posts.len(),
        "posts": page.posts.iter().map(|p| post_json(p, now)).collect::<Vec<_>>(),
        "card": listing_card(&label, &page.posts, now),
        "rate": rate.to_json(),
    }))
}

async fn do_thread(r: Arc<Reddit>, args: Value) -> std::result::Result<Value, String> {
    let id = post_id_from(args["id"].as_str().unwrap_or(""));
    if id.is_empty() {
        return Err("a post id, fullname (t3_…) or permalink is required".into());
    }
    let limit = clamp_limit(args["limit"].as_u64().unwrap_or(15));
    let url = thread_url(&r.creds()?.base(), &id, limit);
    let (body, rate) = r.get(&url).await?;
    let view = parse_thread_value(&body)?;
    let now = now_secs();
    Ok(json!({
        "post": post_json(&view.post, now),
        "comments": view.comments.iter().map(|c| json!({
            "fullname": if c.id.is_empty() { Value::Null } else { json!(format!("t1_{}", c.id)) },
            "author": c.author,
            "score": c.score,
            "age": jerith_reddit_plugin::relative_age(c.created_utc, now),
            "body": clip(&c.body, 900),
        })).collect::<Vec<_>>(),
        "card": thread_card(&view, now),
        "rate": rate.to_json(),
    }))
}

// ---------------------------------------------------------------------------
// Writes (two-step: preview → confirm)
// ---------------------------------------------------------------------------

fn no_write_credentials(mode: jerith_reddit_plugin::AuthMode) -> String {
    format!(
        "these credentials are {} — Reddit app-only tokens cannot post. Use a script app with a refresh token (or username + password) in {}",
        mode.as_str(),
        Credentials::path().display()
    )
}

async fn do_submit(r: Arc<Reddit>, args: Value) -> std::result::Result<Value, String> {
    let mode = r.creds()?.auth_mode();
    if !mode.can_write() {
        return Err(no_write_credentials(mode));
    }
    // Confirm step: everything comes from the stored proposal, so the
    // posted text is exactly what was previewed.
    if let Some(token) = args["confirm_token"]
        .as_str()
        .map(str::trim)
        .filter(|t| !t.is_empty())
    {
        let pid = args["proposal_id"]
            .as_u64()
            .ok_or("confirming a write needs both proposal_id and confirm_token")?;
        let p = r.outbox()?.confirm(pid as u32, token, now_ms())?;
        if p.action != "submit" {
            return Err(format!(
                "proposal #{pid} is a {} preview, not a submit",
                p.action
            ));
        }
        let sub = p.payload["subreddit"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        let kind = p.payload["kind"].as_str().unwrap_or("self").to_string();
        let title = p.payload["title"].as_str().unwrap_or_default().to_string();
        let text = p.payload["text"].as_str().unwrap_or_default().to_string();
        let url = p.payload["url"].as_str().unwrap_or_default().to_string();
        let (body, rate) = r
            .post_form("/api/submit", submit_form(&kind, &sub, &title, &text, &url))
            .await?;
        if let Some(err) = form_errors(&body) {
            return Err(format!("reddit refused the post: {err}"));
        }
        let written = write_result(&body);
        let link = written["url"].as_str().unwrap_or("(no url echoed)");
        return Ok(json!({
            "posted": written,
            "card": format!("✅ posted to r/{sub} — {link}"),
            "rate": rate.to_json(),
        }));
    }

    // Preview step.
    let sub = normalize_sub(args["subreddit"].as_str().unwrap_or(""));
    if sub.is_empty() {
        return Err("which subreddit?".into());
    }
    if matches!(sub.as_str(), "all" | "popular" | "frontpage" | "home") {
        return Err(format!(
            "r/{sub} is a feed, not a subreddit you can post to"
        ));
    }
    let title = args["title"].as_str().unwrap_or("").trim().to_string();
    let tlen = title.chars().count();
    if !(3..=300).contains(&tlen) {
        return Err(format!(
            "a reddit title is 3-300 characters — this one is {tlen}"
        ));
    }
    let kind = match args["kind"]
        .as_str()
        .unwrap_or("self")
        .trim()
        .to_lowercase()
        .as_str()
    {
        "link" => "link",
        _ => "self",
    };
    let text = args["text"].as_str().unwrap_or("").to_string();
    let url = args["url"].as_str().unwrap_or("").trim().to_string();
    if kind == "self" && text.trim().is_empty() {
        return Err("a self post needs body text (markdown, can be one line)".into());
    }
    if kind == "link" && !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err("a link post needs an http(s) url".into());
    }
    if kind == "self" && text.chars().count() > 40_000 {
        return Err("reddit's self-text ceiling is 40,000 characters".into());
    }
    let payload = json!({
        "kind": kind,
        "subreddit": sub,
        "title": title,
        "text": if kind == "self" { text.clone() } else { String::new() },
        "url": if kind == "link" { url.clone() } else { String::new() },
    });
    let p = r.outbox()?.propose("submit", payload, now_ms());
    let body_preview = if kind == "link" {
        url.clone()
    } else {
        clip(text.trim(), 400)
    };
    Ok(json!({
        "status": "preview",
        "proposal_id": p.id,
        "confirm_token": p.token,
        "expires_in_seconds": PENDING_TTL_MS / 1000,
        "would_post": p.payload,
        "card": format!(
            "📝 **preview — submit to r/{sub}** ({kind})\n**{title}**\n{body_preview}\n\n_Nothing has been posted._ Show this to the owner; only after an explicit yes, call `reddit.submit` again with proposal_id={} and confirm_token={}.",
            p.id, p.token
        ),
    }))
}

async fn do_comment(r: Arc<Reddit>, args: Value) -> std::result::Result<Value, String> {
    let mode = r.creds()?.auth_mode();
    if !mode.can_write() {
        return Err(no_write_credentials(mode));
    }
    if let Some(token) = args["confirm_token"]
        .as_str()
        .map(str::trim)
        .filter(|t| !t.is_empty())
    {
        let pid = args["proposal_id"]
            .as_u64()
            .ok_or("confirming a write needs both proposal_id and confirm_token")?;
        let p = r.outbox()?.confirm(pid as u32, token, now_ms())?;
        if p.action != "comment" {
            return Err(format!(
                "proposal #{pid} is a {} preview, not a comment",
                p.action
            ));
        }
        let thing = p.payload["thing_id"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        let text = p.payload["text"].as_str().unwrap_or_default().to_string();
        let (body, rate) = r
            .post_form("/api/comment", comment_form(&thing, &text))
            .await?;
        if let Some(err) = form_errors(&body) {
            return Err(format!("reddit refused the comment: {err}"));
        }
        let written = write_result(&body);
        return Ok(json!({
            "commented": {"on": thing, "thing": written},
            "card": format!(
                "✅ comment posted on {thing} — {}",
                written["url"].as_str().unwrap_or("(no url echoed)")
            ),
            "rate": rate.to_json(),
        }));
    }
    let thing = fullname(args["thing_id"].as_str().unwrap_or(""));
    if thing
        .trim_start_matches("t3_")
        .trim_start_matches("t1_")
        .is_empty()
    {
        return Err("a thing_id is required (t3_ post or t1_ comment)".into());
    }
    let text = args["text"].as_str().unwrap_or("").trim().to_string();
    if text.is_empty() {
        return Err("a comment needs text".into());
    }
    if text.chars().count() > 10_000 {
        return Err("reddit's comment ceiling is 10,000 characters".into());
    }
    let payload = json!({"thing_id": thing, "text": text});
    let p = r.outbox()?.propose("comment", payload, now_ms());
    Ok(json!({
        "status": "preview",
        "proposal_id": p.id,
        "confirm_token": p.token,
        "expires_in_seconds": PENDING_TTL_MS / 1000,
        "would_comment": {"on": thing, "text": clip(&text, 400)},
        "card": format!(
            "💬 **preview — comment on {thing}**\n{}\n\n_Nothing has been posted._ Only after an explicit yes, call `reddit.comment` again with proposal_id={} and confirm_token={}.",
            clip(&text, 600), p.id, p.token
        ),
    }))
}

async fn do_edit(r: Arc<Reddit>, args: Value) -> std::result::Result<Value, String> {
    let mode = r.creds()?.auth_mode();
    if !mode.can_write() {
        return Err(no_write_credentials(mode));
    }
    if let Some(token) = args["confirm_token"]
        .as_str()
        .map(str::trim)
        .filter(|t| !t.is_empty())
    {
        let pid = args["proposal_id"]
            .as_u64()
            .ok_or("confirming a write needs both proposal_id and confirm_token")?;
        let p = r.outbox()?.confirm(pid as u32, token, now_ms())?;
        if p.action != "edit" {
            return Err(format!(
                "proposal #{pid} is a {} preview, not an edit",
                p.action
            ));
        }
        let thing = p.payload["thing_id"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        let text = p.payload["text"].as_str().unwrap_or_default().to_string();
        let (body, rate) = r.post_form("/api/edit", edit_form(&thing, &text)).await?;
        if let Some(err) = form_errors(&body) {
            return Err(format!("reddit refused the edit: {err}"));
        }
        let written = write_result(&body);
        return Ok(json!({
            "edited": {"thing": thing, "result": written},
            "card": format!("✏️ edited {thing} — the replacement text is live"),
            "rate": rate.to_json(),
        }));
    }
    let thing = fullname(args["thing_id"].as_str().unwrap_or(""));
    if thing
        .trim_start_matches("t3_")
        .trim_start_matches("t1_")
        .is_empty()
    {
        return Err("a thing_id is required (t3_ post or t1_ comment)".into());
    }
    let text = args["text"].as_str().unwrap_or("").trim().to_string();
    if text.is_empty() {
        return Err("an edit needs the full replacement text".into());
    }
    let payload = json!({"thing_id": thing, "text": text});
    let p = r.outbox()?.propose("edit", payload, now_ms());
    Ok(json!({
        "status": "preview",
        "proposal_id": p.id,
        "confirm_token": p.token,
        "expires_in_seconds": PENDING_TTL_MS / 1000,
        "would_edit": {"on": thing, "text": clip(&text, 400)},
        "card": format!(
            "✏️ **preview — edit {thing}**\n{}\n\n_Nothing has changed._ Only after an explicit yes, call `reddit.edit` again with proposal_id={} and confirm_token={}. Reddit only allows edits on the bot's own content.",
            clip(&text, 600), p.id, p.token
        ),
    }))
}

async fn do_vote(r: Arc<Reddit>, args: Value) -> std::result::Result<Value, String> {
    let mode = r.creds()?.auth_mode();
    if !mode.can_write() {
        return Err(no_write_credentials(mode));
    }
    let thing = fullname(args["thing_id"].as_str().unwrap_or(""));
    if thing
        .trim_start_matches("t1_")
        .trim_start_matches("t3_")
        .is_empty()
    {
        return Err("a thing_id is required (t3_ post or t1_ comment)".into());
    }
    let dir = vote_dir(args["dir"].as_str().unwrap_or(""))?;
    let (_, rate) = r.post_form("/api/vote", vote_form(&thing, dir)).await?;
    let word = match dir {
        1 => "upvoted",
        -1 => "downvoted",
        _ => "vote cleared",
    };
    Ok(json!({
        "voted": thing,
        "dir": dir,
        "card": format!("🗳️ {word}: {thing}"),
        "rate": rate.to_json(),
    }))
}

// ---------------------------------------------------------------------------
// The lookout
// ---------------------------------------------------------------------------

fn watch_card(matches: &[Post], now: f64) -> String {
    if matches.is_empty() {
        return "👀 **reddit watch** — nothing new matched".to_string();
    }
    let mut out = format!("👀 **reddit watch** — {} new match(es)\n", matches.len());
    for p in matches.iter().take(8) {
        out.push_str(&format!(
            "• r/{} — {} (▲{})\n  {}\n",
            p.subreddit,
            clip(&p.title, 110),
            p.score,
            p.permalink_url()
        ));
    }
    let _ = now;
    out.trim_end().to_string()
}

async fn do_check(r: Arc<Reddit>, limit: usize) -> std::result::Result<Value, String> {
    let base = r.creds()?.base();
    let w = load_watch(&r.watch_file);
    if w.subs.is_empty() {
        return Ok(json!({
            "new_count": 0,
            "checked_subs": [],
            "matches": [],
            "note": "no subreddits on the watch list yet — add one with action=add",
        }));
    }
    let mut posts: Vec<Post> = Vec::new();
    let mut checked: Vec<String> = Vec::new();
    let mut errors: Vec<String> = Vec::new();
    for sub in w.subs.iter().take(8) {
        // /new is the only sort that gives a stable "since last time"
        // window without paging back through history.
        let url = listing_url(&base, sub, "new", 50, "day", None);
        match r.get(&url).await {
            Ok((body, _)) => match parse_listing_value(&body) {
                Ok(page) => {
                    checked.push(sub.clone());
                    posts.extend(page.posts);
                }
                Err(e) => errors.push(format!("r/{sub}: {e}")),
            },
            Err(e) => errors.push(format!("r/{sub}: {e}")),
        }
    }
    let now = now_secs();
    let matches = select_matches(&posts, &w, limit);
    // Mark every fetched post, not just the matches: the cursor then
    // advances past posts that didn't match and they can't resurface.
    let mut next = w.clone();
    for p in &posts {
        next.mark(p);
    }
    // Keep the matches for the morning digest. Recording is separate from
    // the cursor, so `watch recent` can re-read last night without
    // consuming what this check just announced.
    next.record(&matches, now_ms());
    if let Err(e) = save_watch(&r.watch_file, &next) {
        return Err(format!(
            "the watch file {} could not be saved: {e}",
            r.watch_file.display()
        ));
    }
    Ok(json!({
        "new_count": matches.len(),
        "checked_subs": checked,
        "errors": errors,
        "matches": matches.iter().map(|p| post_json(p, now)).collect::<Vec<_>>(),
        "card": watch_card(&matches, now),
    }))
}

async fn do_watch(r: Arc<Reddit>, args: Value) -> std::result::Result<Value, String> {
    let action = args["action"]
        .as_str()
        .unwrap_or("list")
        .trim()
        .to_lowercase();
    let path = r.watch_file.clone();
    match action.as_str() {
        "add" => {
            let sub = normalize_sub(args["subreddit"].as_str().unwrap_or(""));
            if sub.is_empty() {
                return Err("action=add needs a subreddit".into());
            }
            let terms = args["terms"].as_str().unwrap_or("");
            let mut w = load_watch(&path);
            let added_sub = w.add_sub(&sub);
            let added_terms = w.add_terms(terms);
            save_watch(&path, &w).map_err(|e| format!("watch file save failed: {e}"))?;
            Ok(json!({
                "added_subreddit": added_sub,
                "added_terms": added_terms,
                "subreddits": w.subs,
                "terms": w.terms,
                "watch_file": path.display().to_string(),
                "card": format!(
                    "👀 watching r/{sub}{} — {} term(s) on {} sub(s)",
                    if added_sub { "" } else { " (already watched)" },
                    w.terms.len(),
                    w.subs.len()
                ),
            }))
        }
        "remove" => {
            let sub = normalize_sub(args["subreddit"].as_str().unwrap_or(""));
            let term = args["term"].as_str().unwrap_or("").trim().to_string();
            if sub.is_empty() && term.is_empty() {
                return Err("action=remove needs a subreddit or a term".into());
            }
            let mut w = load_watch(&path);
            let removed_sub = !sub.is_empty() && w.remove_sub(&sub);
            let removed_term = !term.is_empty() && w.remove_term(&term);
            save_watch(&path, &w).map_err(|e| format!("watch file save failed: {e}"))?;
            Ok(json!({
                "removed_subreddit": removed_sub,
                "removed_term": removed_term,
                "subreddits": w.subs,
                "terms": w.terms,
            }))
        }
        "check" => {
            let limit = args["limit"]
                .as_u64()
                .map(|n| n.clamp(1, jerith_reddit_plugin::MAX_MATCHES as u64) as usize)
                .unwrap_or(10);
            do_check(r, limit).await
        }
        "recent" => {
            // The digest read: matches the checks already caught in the
            // window, WITHOUT advancing anything — the daily briefing calls
            // this at 08:00 and the monitor's next tick must still be able
            // to announce what it finds.
            let hours = args["hours"].as_u64().unwrap_or(24).clamp(1, 168);
            let w = load_watch(&path);
            let cutoff = now_ms().saturating_sub(hours * 3_600_000);
            let caught = w.caught_since(cutoff);
            let now = now_secs();
            let mut card = if caught.is_empty() {
                format!("👀 reddit lookout — nothing caught in the last {hours}h")
            } else {
                format!(
                    "👀 reddit lookout — {} caught in the last {hours}h\n",
                    caught.len()
                )
            };
            for c in caught.iter().take(8) {
                card.push_str(&format!(
                    "• r/{} — {} (▲{})\n  {}\n",
                    c.subreddit,
                    clip(&c.title, 110),
                    c.score,
                    c.permalink
                ));
            }
            Ok(json!({
                "count": caught.len(),
                "hours": hours,
                "matches": caught.iter().map(|c| json!({
                    "id": c.id,
                    "subreddit": c.subreddit,
                    "title": c.title,
                    "score": c.score,
                    "permalink": c.permalink,
                    "age": jerith_reddit_plugin::relative_age(c.created_utc, now),
                })).collect::<Vec<_>>(),
                "card": card.trim_end(),
            }))
        }
        "list" => {
            let w = load_watch(&path);
            let now = now_secs();
            Ok(json!({
                "subreddits": w.subs,
                "terms": w.terms,
                "cursors": w.cursor.iter().map(|(sub, ts)| json!({
                    "subreddit": sub,
                    "last_seen": jerith_reddit_plugin::relative_age(*ts, now),
                })).collect::<Vec<_>>(),
                "watch_file": path.display().to_string(),
            }))
        }
        other => Err(format!(
            "unknown watch action {other:?} — use add, remove, list, check or recent"
        )),
    }
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

async fn do_about(r: Arc<Reddit>, _args: Value) -> std::result::Result<Value, String> {
    let creds = match r.creds() {
        Ok(c) => c,
        Err(e) => return Ok(json!({"connected": false, "problem": e})),
    };
    if let Err(e) = creds.validate() {
        return Ok(json!({"connected": false, "problem": e}));
    }
    let mode = creds.auth_mode();
    let base = creds.base();
    let url = format!("{base}/api/v1/me");
    match r.get(&url).await {
        Ok((me, rate)) => Ok(json!({
            "connected": true,
            "auth_mode": mode.as_str(),
            "can_write": mode.can_write(),
            "username": me.get("name").cloned().unwrap_or(Value::Null),
            "link_karma": me.get("link_karma").cloned().unwrap_or(Value::Null),
            "comment_karma": me.get("comment_karma").cloned().unwrap_or(Value::Null),
            "account_created_utc": me.get("created_utc").cloned().unwrap_or(Value::Null),
            "base": base,
            "rate": rate.to_json(),
            "card": format!(
                "🟢 reddit: u/{} via {} · karma {} link / {} comment",
                me.get("name").and_then(Value::as_str).unwrap_or("?"),
                mode.as_str(),
                me.get("link_karma").and_then(Value::as_i64).unwrap_or(0),
                me.get("comment_karma").and_then(Value::as_i64).unwrap_or(0),
            ),
        })),
        Err(e) => Ok(json!({
            "connected": false,
            "auth_mode": mode.as_str(),
            "base": base,
            "problem": e,
        })),
    }
}

/// `[plugin] endpoint` + `[plugin] secret` (None = unsigned dev mode)
/// straight out of the manifest — one file rules both halves.
fn read_manifest(path: &str) -> Result<(std::net::SocketAddr, Option<String>)> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("reading manifest {path}"))?;
    // Document parse (toml::Table), NOT `raw.parse::<toml::Value>()` — the
    // pet's lesson, pinned by every drop since.
    let file: toml::Table =
        toml::from_str(&raw).with_context(|| format!("{path}: not valid TOML"))?;
    let plugin = file
        .get("plugin")
        .and_then(toml::Value::as_table)
        .context("manifest has no [plugin] table")?;
    let endpoint = plugin
        .get("endpoint")
        .and_then(toml::Value::as_str)
        .context("[plugin] endpoint missing")?;
    let addr = endpoint_addr(endpoint)?;
    let secret = plugin
        .get("secret")
        .and_then(toml::Value::as_str)
        .map(str::to_string);
    Ok((addr, secret))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let manifest = std::env::args().nth(1).unwrap_or("plugin.toml".to_string());
    let (addr, secret) = read_manifest(&manifest)?;
    let reddit = Arc::new(Reddit::new()?);
    match reddit.creds() {
        Ok(c) => tracing::info!(
            auth_mode = c.auth_mode().as_str(),
            can_write = c.auth_mode().can_write(),
            base = %c.base(),
            "reddit credentials loaded"
        ),
        Err(e) => {
            tracing::warn!(error = %e, "reddit credentials missing — reads/writes will refuse until configured")
        }
    }
    let watch_file = reddit.watch_file.clone();
    tracing::info!(
        manifest,
        %addr,
        secret = secret.is_some(),
        watch_file = %watch_file.display(),
        "reddit plugin starting"
    );

    SkillServer::new("reddit", secret)
        .handler(
            "reddit.browse",
            skill({
                let r = reddit.clone();
                move |args: Value| {
                    let r = r.clone();
                    async move { do_browse(r, args).await }
                }
            }),
        )
        .handler(
            "reddit.search",
            skill({
                let r = reddit.clone();
                move |args: Value| {
                    let r = r.clone();
                    async move { do_search(r, args).await }
                }
            }),
        )
        .handler(
            "reddit.thread",
            skill({
                let r = reddit.clone();
                move |args: Value| {
                    let r = r.clone();
                    async move { do_thread(r, args).await }
                }
            }),
        )
        .handler(
            "reddit.submit",
            skill({
                let r = reddit.clone();
                move |args: Value| {
                    let r = r.clone();
                    async move { do_submit(r, args).await }
                }
            }),
        )
        .handler(
            "reddit.comment",
            skill({
                let r = reddit.clone();
                move |args: Value| {
                    let r = r.clone();
                    async move { do_comment(r, args).await }
                }
            }),
        )
        .handler(
            "reddit.edit",
            skill({
                let r = reddit.clone();
                move |args: Value| {
                    let r = r.clone();
                    async move { do_edit(r, args).await }
                }
            }),
        )
        .handler(
            "reddit.vote",
            skill({
                let r = reddit.clone();
                move |args: Value| {
                    let r = r.clone();
                    async move { do_vote(r, args).await }
                }
            }),
        )
        .handler(
            "reddit.watch",
            skill({
                let r = reddit.clone();
                move |args: Value| {
                    let r = r.clone();
                    async move { do_watch(r, args).await }
                }
            }),
        )
        .handler(
            "reddit.about",
            skill({
                let r = reddit.clone();
                move |args: Value| {
                    let r = r.clone();
                    async move { do_about(r, args).await }
                }
            }),
        )
        .serve(addr)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use jerith_reddit_plugin::Credentials;

    /// The SHIPPED manifest must stay servable by THIS binary: document
    /// parses, the endpoint binds loopback, the secret is set, and the
    /// nine declared rows are the nine handlers registered above.
    #[test]
    fn shipped_manifest_fits_this_binary() {
        let raw = include_str!("../plugin.toml");
        let file: toml::Table =
            toml::from_str(raw).expect("shipped manifest is valid TOML (document parse)");
        let plugin = file
            .get("plugin")
            .expect("[plugin] table")
            .as_table()
            .expect("[plugin] is a table");
        assert_eq!(
            plugin.get("id").and_then(toml::Value::as_str),
            Some("reddit")
        );
        let addr = endpoint_addr(
            plugin
                .get("endpoint")
                .and_then(toml::Value::as_str)
                .expect("endpoint line"),
        )
        .expect("endpoint binds");
        assert!(addr.ip().is_loopback(), "loopback by contract");
        assert!(addr.port() != 7793, "weather's port stays weather's");
        assert!(
            plugin.get("secret").and_then(toml::Value::as_str).is_some(),
            "ships signed by default"
        );
        let names: Vec<&str> = file
            .get("skill")
            .and_then(toml::Value::as_array)
            .expect("[[skill]] rows")
            .iter()
            .filter_map(|s| s.get("name").and_then(toml::Value::as_str))
            .collect();
        assert_eq!(
            names,
            vec![
                "browse", "search", "thread", "submit", "comment", "edit", "vote", "watch", "about"
            ],
            "the nine served handlers"
        );
    }

    #[test]
    fn write_instructions_are_in_the_write_descriptions() {
        let raw = include_str!("../plugin.toml");
        let file: toml::Table = toml::from_str(raw).unwrap();
        let rows = file
            .get("skill")
            .and_then(toml::Value::as_array)
            .expect("rows");
        for name in ["submit", "comment", "edit"] {
            let row = rows
                .iter()
                .find(|s| s.get("name").and_then(toml::Value::as_str) == Some(name))
                .expect(name);
            let desc = row
                .get("description")
                .and_then(toml::Value::as_str)
                .expect("description");
            assert!(
                desc.contains("confirm") && desc.contains("preview"),
                "{name} must describe the two-step rail: {desc}"
            );
        }
    }

    #[test]
    fn rates_read_the_headers_reddit_sends() {
        let mut h = reqwest::header::HeaderMap::new();
        h.insert("x-ratelimit-remaining", "83.0".parse().unwrap());
        h.insert("x-ratelimit-used", "17".parse().unwrap());
        h.insert("x-ratelimit-reset", "412".parse().unwrap());
        let r = Rate::from_headers(&h);
        assert_eq!(r.remaining, Some(83.0));
        assert_eq!(r.used, Some(17.0));
        assert_eq!(r.reset_secs, Some(412));
        assert_eq!(r.to_json()["remaining"], 83.0);
        // Missing headers report nulls, not zeros.
        let none = Rate::from_headers(&reqwest::header::HeaderMap::new());
        assert!(none.remaining.is_none() && none.to_json()["remaining"].is_null());
    }

    #[test]
    fn app_only_credentials_refuse_writes_with_the_fix() {
        let msg = no_write_credentials(jerith_reddit_plugin::AuthMode::AppOnly);
        assert!(msg.contains("app_only"), "{msg}");
        assert!(msg.contains("refresh token"), "{msg}");
        assert!(msg.contains("credentials.toml"), "{msg}");
        let _ = Credentials::default();
    }

    #[test]
    fn watch_card_is_quiet_and_bounded() {
        assert!(watch_card(&[], 0.0).contains("nothing new"));
        let posts: Vec<Post> = (0..12)
            .map(|i| Post {
                id: format!("p{i}"),
                title: "t".repeat(300),
                subreddit: "rust".into(),
                permalink: format!("/r/rust/comments/p{i}/x/"),
                score: 1,
                ..Post::default()
            })
            .collect();
        let card = watch_card(&posts, 0.0);
        assert!(card.contains("12 new match"), "{card}");
        assert_eq!(
            card.matches("• r/").count(),
            8,
            "the card caps at eight rows"
        );
        assert!(card.len() < 4000, "cards stay small");
    }
}
