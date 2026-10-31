//! The reddit lane's pure half (crates/reddit-plugin, the SDK's seventh
//! live drop after example/pet/house/quarantine/ledger/weather).
//!
//! Everything here is clockless and networkless: credential parsing, the
//! OAuth token form, URL builders, listing/thread parsing, the vote
//! direction table, the propose-and-confirm outbox for writes, the watch
//! list, and the cards each skill returns. The wire lives in `main.rs`.
//!
//! Design notes that matter:
//!
//! - **Reads are free, writes are two-step.** `submit`/`comment`/`edit`
//!   never touch Reddit on the first call; they mint a `Proposal` in the
//!   [`Outbox`] and only the second call, carrying both the proposal id
//!   and its token (within [`PENDING_TTL_MS`]), posts anything. That is
//!   the same propose-and-confirm rail the bot's approval bridge uses,
//!   enforced in the plugin so no prompt can skip it.
//! - **Votes are one call but not a ring.** [`vote_dir`] maps the three
//!   legal directions; the manifest tells the model that coordinated
//!   voting is out of bounds — Reddit's bot policy, not our preference.
//! - **Watch state is a cursor, not a queue.** Per-sub `created_utc`
//!   watermarks plus a capped seen-id ring mean a restarted service
//!   replays nothing and a busy sub can't grow the file without bound.
//! - **Credentials never live in the manifest.** The drop file carries
//!   only the plugin secret; Reddit's client id/secret/user-agent live in
//!   `~/.jerith-mesh/reddit/credentials.toml` (0600) or env vars. A
//!   missing half is a typed error that names the exact key, which is
//!   what the operator sees instead of a mystery 401.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::{Path, PathBuf};

/// Reddit's OAuth token endpoint (form POST, HTTP basic with
/// client_id:client_secret).
pub const TOKEN_URL: &str = "https://www.reddit.com/api/v1/access_token";
/// The authenticated API base. `oauth.reddit.com` is the only host that
/// accepts a bearer token for writes.
pub const DEFAULT_BASE: &str = "https://oauth.reddit.com";
/// Model-facing listing cap (the dispatcher allows more; a chat answer
/// with 100 posts is noise, not information).
pub const MAX_LISTING: u32 = 50;
/// Ceiling on watch matches returned per check.
pub const MAX_MATCHES: usize = 25;
/// Seen-id ring size — bounded so the watch file cannot grow forever.
pub const SEEN_CAP: usize = 2000;
/// Digest rows kept for the morning-briefing line — bounded like the
/// seen ring, because a week of a busy sub is more history than a
/// morning note can use.
pub const INBOX_CAP: usize = 200;
/// How long a caught match stays in the digest (7 days): long enough for
/// a week of briefings, short enough that a typo'd term doesn't linger.
pub const INBOX_RETENTION_MS: u64 = 7 * 24 * 60 * 60 * 1000;
/// How long a write preview stays confirmable. Long enough to read a
/// draft, short enough that a stale "yes" can't post something else.
pub const PENDING_TTL_MS: u64 = 10 * 60 * 1000;

// ---------------------------------------------------------------------------
// The credential-free read path (public Atom feeds)
// ---------------------------------------------------------------------------

/// The signed-out read host. `www.reddit.com` serves Atom for every
/// listing, search and comment tree without an OAuth token — which is what
/// keeps reading (and the lookout) alive on a box with no Reddit app
/// registered. Writes are never possible here; they are not a grey area,
/// they need a token.
pub const ANON_BASE: &str = "https://www.reddit.com";
/// Reddit refuses requests with no User-Agent. This one names the software
/// instead of claiming an account that does not exist.
pub const ANON_USER_AGENT: &str = "linux:jerith-mesh:0.1.4 (anonymous lookout)";
/// Measured on the operator's box (2026-10-31): the anonymous feed answers
/// roughly one request per thirty seconds and 429s in between, against
/// ~600 calls / 10 min on an OAuth app. The watch tick spends its whole
/// allowance on one sub for exactly that reason.
pub const ANON_MIN_GAP_SECS: u64 = 30;
/// What every signed-out answer says about itself, so nothing downstream
/// mistakes a feed read for a full API read.
pub const ANON_NOTE: &str = "signed-out read from reddit's public Atom feed: title, author, permalink, age and body only — score, comment count and vote state need a script app in ~/.jerith-mesh/reddit/credentials.toml";

// ---------------------------------------------------------------------------
// Credentials
// ---------------------------------------------------------------------------

/// Which OAuth flow the configured credentials support. The order is the
/// preference order: a refresh token never stores the account password.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthMode {
    RefreshToken,
    Password,
    AppOnly,
}

impl AuthMode {
    pub fn as_str(self) -> &'static str {
        match self {
            AuthMode::RefreshToken => "refresh_token",
            AuthMode::Password => "password",
            AuthMode::AppOnly => "app_only",
        }
    }
    /// App-only tokens are read-only in practice (no `identity`/`submit`
    /// scopes), so the write skills say so up front instead of letting
    /// Reddit answer 403.
    pub fn can_write(self) -> bool {
        !matches!(self, AuthMode::AppOnly)
    }
}

/// The credential file's shape. Every field is optional at parse time so
/// a half-filled file reports WHICH key is missing rather than failing
/// with a serde line number.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Credentials {
    #[serde(default)]
    pub client_id: String,
    #[serde(default)]
    pub client_secret: String,
    #[serde(default)]
    pub username: Option<String>,
    #[serde(default)]
    pub password: Option<String>,
    #[serde(default)]
    pub refresh_token: Option<String>,
    #[serde(default)]
    pub user_agent: String,
    /// Override for tests / a proxy (`https://oauth.reddit.com` by
    /// default).
    #[serde(default)]
    pub base: Option<String>,
}

impl Credentials {
    /// `$JERITH_REDDIT_CREDENTIALS`, else `~/.jerith-mesh/reddit/credentials.toml`.
    pub fn path() -> PathBuf {
        if let Ok(p) = std::env::var("JERITH_REDDIT_CREDENTIALS")
            && !p.trim().is_empty()
        {
            return PathBuf::from(p);
        }
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        PathBuf::from(home).join(".jerith-mesh/reddit/credentials.toml")
    }

    /// Parse a credentials document. The outer `Result` is the TOML
    /// error; missing keys are reported by [`Self::validate`].
    pub fn parse(raw: &str) -> Result<Self, String> {
        toml::from_str(raw).map_err(|e| format!("credentials file is not valid TOML: {e}"))
    }

    /// Env overrides win over the file (systemd `Environment=` friendly).
    pub fn apply_env(&mut self) {
        let take = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
        if let Some(v) = take("JERITH_REDDIT_CLIENT_ID") {
            self.client_id = v;
        }
        if let Some(v) = take("JERITH_REDDIT_CLIENT_SECRET") {
            self.client_secret = v;
        }
        if let Some(v) = take("JERITH_REDDIT_USERNAME") {
            self.username = Some(v);
        }
        if let Some(v) = take("JERITH_REDDIT_PASSWORD") {
            self.password = Some(v);
        }
        if let Some(v) = take("JERITH_REDDIT_REFRESH_TOKEN") {
            self.refresh_token = Some(v);
        }
        if let Some(v) = take("JERITH_REDDIT_USER_AGENT") {
            self.user_agent = v;
        }
        if let Some(v) = take("JERITH_REDDIT_BASE") {
            self.base = Some(v);
        }
    }

    /// Load file + env. A missing file is a typed error naming the path
    /// and the keys to put in it — the operator's first-run experience.
    pub fn load() -> Result<Self, String> {
        let path = Self::path();
        let mut creds = match std::fs::read_to_string(&path) {
            Ok(raw) => Self::parse(&raw)?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(format!(
                    "no reddit credentials at {} — create it (chmod 600) with {}",
                    path.display(),
                    Self::required_keys_line()
                ));
            }
            Err(e) => {
                return Err(format!(
                    "could not read reddit credentials at {}: {e}",
                    path.display()
                ));
            }
        };
        creds.apply_env();
        Ok(creds)
    }

    pub fn required_keys_line() -> &'static str {
        "client_id, client_secret, user_agent, and refresh_token (preferred) or username + password"
    }

    /// Validate the half that every call needs, then the auth half.
    /// Errors name the exact missing key.
    pub fn validate(&self) -> Result<(), String> {
        let mut missing = Vec::new();
        if self.client_id.trim().is_empty() {
            missing.push("client_id");
        }
        if self.client_secret.trim().is_empty() {
            missing.push("client_secret");
        }
        if self.user_agent.trim().is_empty() {
            missing.push(
                "user_agent (Reddit requires e.g. \"linux:jerith-mesh:0.1.4 (by /u/yourname)\")",
            );
        }
        if !missing.is_empty() {
            return Err(format!(
                "reddit credentials incomplete — missing {} (set them in {} or via JERITH_REDDIT_*)",
                missing.join(", "),
                Self::path().display()
            ));
        }
        match self.auth_mode() {
            AuthMode::RefreshToken => Ok(()),
            AuthMode::Password => Ok(()),
            AuthMode::AppOnly => Ok(()),
        }
    }

    /// Which flow the filled-in fields describe.
    pub fn auth_mode(&self) -> AuthMode {
        let has = |v: &Option<String>| v.as_deref().is_some_and(|s| !s.trim().is_empty());
        if has(&self.refresh_token) {
            AuthMode::RefreshToken
        } else if has(&self.username) && has(&self.password) {
            AuthMode::Password
        } else {
            AuthMode::AppOnly
        }
    }

    pub fn base(&self) -> String {
        self.base
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(DEFAULT_BASE)
            .trim_end_matches('/')
            .to_string()
    }

    /// The token request form, exactly as Reddit's OAuth docs specify.
    /// Refresh is preferred: the password never lands in a config file
    /// the operator has to keep.
    pub fn token_form(&self) -> Vec<(String, String)> {
        let mut form = Vec::new();
        match self.auth_mode() {
            AuthMode::RefreshToken => {
                form.push(("grant_type".into(), "refresh_token".into()));
                form.push((
                    "refresh_token".into(),
                    self.refresh_token.clone().unwrap_or_default(),
                ));
            }
            AuthMode::Password => {
                form.push(("grant_type".into(), "password".into()));
                form.push(("username".into(), self.username.clone().unwrap_or_default()));
                form.push(("password".into(), self.password.clone().unwrap_or_default()));
            }
            AuthMode::AppOnly => form.push(("grant_type".into(), "client_credentials".into())),
        }
        form
    }

    /// Reddit's token endpoint wants client_id:client_secret as HTTP
    /// basic auth (not form fields). Built here so the wire keeps no
    /// secrets logic of its own.
    pub fn basic_auth_header(&self) -> String {
        use base64::Engine as _;
        let raw = format!("{}:{}", self.client_id, self.client_secret);
        format!(
            "Basic {}",
            base64::engine::general_purpose::STANDARD.encode(raw.as_bytes())
        )
    }
}

/// One parsed token reply.
#[derive(Debug, Clone, Deserialize)]
pub struct TokenReply {
    #[serde(default)]
    pub access_token: String,
    #[serde(default)]
    pub expires_in: u64,
    #[serde(default)]
    pub scope: Option<String>,
    #[serde(default)]
    pub refresh_token: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
}

/// Parse Reddit's token answer. A body with `error` is a refusal, not a
/// token — surfaced with the operator-actionable hint.
pub fn parse_token(body: &str) -> Result<TokenReply, String> {
    let t: TokenReply =
        serde_json::from_str(body).map_err(|e| format!("reddit token reply is not JSON: {e}"))?;
    if let Some(err) = t.error.as_deref().filter(|e| !e.trim().is_empty()) {
        return Err(token_hint(err));
    }
    if t.access_token.trim().is_empty() {
        return Err("reddit token reply carried no access_token".into());
    }
    Ok(t)
}

/// Reddit's three OAuth errors, each with the thing to actually fix.
pub fn token_hint(err: &str) -> String {
    match err {
        "invalid_grant" => "reddit refused the grant — the refresh token was revoked/expired, or the password or script-app credentials are wrong. Regenerate the token at reddit.com/prefs/apps and update the credentials file.".into(),
        "unauthorized_client" | "invalid_client" => "reddit refused the client — check client_id/client_secret belong to a \"script\" app, and that the app is not disabled.".into(),
        "unsupported_grant_type" => "reddit does not support that grant for this app type — create the app as type \"script\".".into(),
        other => format!("reddit auth failed: {other}"),
    }
}

// ---------------------------------------------------------------------------
// URL builders + ids
// ---------------------------------------------------------------------------

/// Percent-encode a query value (reserved chars always) — the same
/// helper shape the weather drop and gifs.rs use.
pub fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            other => out.push_str(&format!("%{other:02X}")),
        }
    }
    out
}

/// "r/rust", "/r/rust/", a full reddit URL, or "Rust" all become `rust`.
pub fn normalize_sub(raw: &str) -> String {
    let mut s = raw.trim();
    for p in ["https://", "http://"] {
        s = s.strip_prefix(p).unwrap_or(s);
    }
    for p in ["www.reddit.com", "old.reddit.com", "reddit.com"] {
        s = s.strip_prefix(p).unwrap_or(s);
    }
    let s = s.trim_start_matches('/');
    let s = s
        .strip_prefix("r/")
        .or_else(|| s.strip_prefix("R/"))
        .unwrap_or(s);
    s.trim_matches('/').replace(' ', "").to_lowercase()
}

/// Listing sorts Reddit actually serves; anything else falls back to hot
/// rather than 404ing on `/r/x/nonsense`.
pub fn sort_or_default(raw: &str) -> &'static str {
    match raw.trim().to_lowercase().as_str() {
        "new" => "new",
        "top" => "top",
        "rising" => "rising",
        "controversial" => "controversial",
        "best" => "best",
        _ => "hot",
    }
}

/// `t=` window for top/controversial. Unknown → day.
pub fn window_or_default(raw: &str) -> &'static str {
    match raw.trim().to_lowercase().as_str() {
        "hour" => "hour",
        "week" => "week",
        "month" => "month",
        "year" => "year",
        "all" => "all",
        _ => "day",
    }
}

/// Model-facing listing cap.
pub fn clamp_limit(n: u64) -> u32 {
    (n as u32).clamp(1, MAX_LISTING)
}

/// `/r/<sub>/<sort>?limit=…&raw_json=1[&t=…][&after=…]`. An empty sub
/// means the signed-in front page, which Reddit serves at `/<sort>`.
pub fn listing_url(
    base: &str,
    sub: &str,
    sort: &str,
    limit: u32,
    window: &str,
    after: Option<&str>,
) -> String {
    let mut url = if sub.is_empty() {
        format!("{base}/{sort}?limit={limit}&raw_json=1")
    } else {
        format!("{base}/r/{sub}/{sort}?limit={limit}&raw_json=1")
    };
    if matches!(sort, "top" | "controversial") {
        url.push_str(&format!("&t={window}"));
    }
    if let Some(after) = after.filter(|a| !a.trim().is_empty()) {
        url.push_str(&format!("&after={}", urlencode(after)));
    }
    url
}

/// Search: restricted to a sub when one is given, site-wide otherwise.
pub fn search_url(
    base: &str,
    sub: Option<&str>,
    query: &str,
    sort: &str,
    limit: u32,
    window: &str,
) -> String {
    let sort = match sort {
        "new" | "top" | "comments" => sort,
        _ => "relevance",
    };
    let q = urlencode(query.trim());
    match sub.filter(|s| !s.is_empty()) {
        Some(sub) => {
            let mut url = format!(
                "{base}/r/{sub}/search?q={q}&restrict_sr=1&sort={sort}&limit={limit}&raw_json=1"
            );
            if sort == "top" {
                url.push_str(&format!("&t={window}"));
            }
            url
        }
        None => format!("{base}/search?q={q}&sort={sort}&limit={limit}&type=link&raw_json=1"),
    }
}

/// One post + its comment tree (top-level only, `depth=1`).
pub fn thread_url(base: &str, id: &str, limit: u32) -> String {
    format!(
        "{base}/comments/{}?limit={limit}&depth=1&sort=top&raw_json=1",
        post_id_from(id)
    )
}

/// Accept a fullname (`t3_abc`), a bare id (`abc`), or any reddit URL
/// containing `/comments/<id>` — return the bare post id.
pub fn post_id_from(raw: &str) -> String {
    let s = raw.trim();
    if let Some(idx) = s.find("/comments/") {
        let rest = &s[idx + "/comments/".len()..];
        return rest
            .split(['/', '?', '#'])
            .next()
            .unwrap_or(rest)
            .trim()
            .to_string();
    }
    let s = s
        .strip_prefix("t3_")
        .or_else(|| s.strip_prefix("t1_"))
        .unwrap_or(s);
    s.trim().to_string()
}

/// Fullname for a vote/comment target: keep a `t1_`/`t3_` prefix, add
/// `t3_` to a bare id (a bare id is a post — comments always arrive
/// prefixed from `thread`).
pub fn fullname(raw: &str) -> String {
    let s = raw.trim();
    let stripped = s.strip_prefix("t1_").or_else(|| s.strip_prefix("t3_"));
    match stripped {
        Some(rest) => format!("{}{}", &s[..3], rest.trim()),
        None => format!("t3_{}", post_id_from(s)),
    }
}

/// The three legal vote directions. `0` clears.
pub fn vote_dir(raw: &str) -> Result<i8, String> {
    match raw.trim().to_lowercase().as_str() {
        "up" | "upvote" | "1" | "+1" => Ok(1),
        "down" | "downvote" | "-1" => Ok(-1),
        "none" | "clear" | "unvote" | "0" | "remove" => Ok(0),
        other => Err(format!("vote dir {other:?} is not one of up, down, none")),
    }
}

// ---------------------------------------------------------------------------
// Listing / thread parsing
// ---------------------------------------------------------------------------

/// One post row (`data` of a `t3` thing). Every field is defaulted: a
/// shape drift answers with blanks instead of killing the whole listing.
#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct Post {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub author: String,
    #[serde(default)]
    pub subreddit: String,
    #[serde(default)]
    pub score: i64,
    #[serde(default)]
    pub num_comments: u32,
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub permalink: String,
    #[serde(default)]
    pub selftext: String,
    #[serde(default)]
    pub created_utc: f64,
    #[serde(default)]
    pub over_18: bool,
    #[serde(default)]
    pub is_self: bool,
    #[serde(default)]
    pub link_flair_text: Option<String>,
    #[serde(default)]
    pub stickied: bool,
    /// False on the anonymous Atom path. The feed carries no score and no
    /// comment count, and a card printing "▲0" would be inventing numbers
    /// Reddit never sent — so the card asks instead.
    #[serde(default = "metrics_known_default")]
    pub metrics_known: bool,
}

/// The API answers carry scores; only the feed parser turns this off.
fn metrics_known_default() -> bool {
    true
}

impl Post {
    pub fn fullname(&self) -> String {
        if self.name.is_empty() {
            format!("t3_{}", self.id)
        } else {
            self.name.clone()
        }
    }
    pub fn permalink_url(&self) -> String {
        if self.permalink.is_empty() {
            self.url.clone()
        } else if self.permalink.starts_with("http") {
            self.permalink.clone()
        } else {
            format!("https://www.reddit.com{}", self.permalink)
        }
    }
}

#[derive(Debug, Deserialize)]
struct Thing {
    #[serde(default)]
    data: Post,
}

#[derive(Debug, Default, Deserialize)]
struct ListingChildren {
    #[serde(default)]
    children: Vec<Thing>,
    #[serde(default)]
    after: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Listing {
    #[serde(default)]
    data: ListingChildren,
}

/// One page of posts + the `after` cursor.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ListingPage {
    pub posts: Vec<Post>,
    pub after: Option<String>,
}

/// Parse a listing body. A JSON body without a `data` object is not a
/// listing (Reddit's error shape) and says so.
pub fn parse_listing(body: &str) -> Result<ListingPage, String> {
    let v: Value =
        serde_json::from_str(body).map_err(|e| format!("reddit listing is not JSON: {e}"))?;
    parse_listing_value(&v)
}

pub fn parse_listing_value(v: &Value) -> Result<ListingPage, String> {
    if let Some(err) = v.get("error").and_then(Value::as_str) {
        return Err(format!("reddit refused the listing: {err}"));
    }
    if v.get("data").is_none() {
        return Err("reddit returned no listing (wrong subreddit name?)".into());
    }
    let listing: Listing =
        serde_json::from_value(v.clone()).map_err(|e| format!("listing shape drifted: {e}"))?;
    Ok(ListingPage {
        posts: listing.data.children.into_iter().map(|t| t.data).collect(),
        after: listing.data.after,
    })
}

/// One comment row.
#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct Comment {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub author: String,
    #[serde(default)]
    pub body: String,
    #[serde(default)]
    pub score: i64,
    #[serde(default)]
    pub created_utc: f64,
    #[serde(default)]
    pub parent_id: String,
    #[serde(default)]
    pub permalink: Option<String>,
    /// Same contract as [`Post::metrics_known`]: the feed has no scores.
    #[serde(default = "metrics_known_default")]
    pub metrics_known: bool,
}

#[derive(Debug, Deserialize)]
struct CommentThing {
    #[serde(default)]
    data: Comment,
}

#[derive(Debug, Default, Deserialize)]
struct CommentChildren {
    #[serde(default)]
    children: Vec<CommentThing>,
}

#[derive(Debug, Deserialize)]
struct CommentListing {
    #[serde(default)]
    data: CommentChildren,
}

/// A post plus its top-level comments.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ThreadView {
    pub post: Post,
    pub comments: Vec<Comment>,
}

/// Parse `/comments/<id>`: a two-listing array (post, comments). Only
/// top-level comments are kept — replies are nested inside `replies`,
/// which this view deliberately does not recurse into.
pub fn parse_thread(body: &str) -> Result<ThreadView, String> {
    let v: Value =
        serde_json::from_str(body).map_err(|e| format!("reddit thread is not JSON: {e}"))?;
    parse_thread_value(&v)
}

pub fn parse_thread_value(v: &Value) -> Result<ThreadView, String> {
    let arr = v
        .as_array()
        .ok_or_else(|| "reddit thread body is not the [post, comments] array".to_string())?;
    let post_listing: Listing = serde_json::from_value(
        arr.first()
            .cloned()
            .ok_or_else(|| "reddit thread body has no post listing".to_string())?,
    )
    .map_err(|e| format!("thread post shape drifted: {e}"))?;
    let post = post_listing
        .data
        .children
        .into_iter()
        .next()
        .map(|t| t.data)
        .ok_or_else(|| "reddit thread has no post (deleted?)".to_string())?;
    let comments: Vec<Comment> = match arr.get(1) {
        Some(second) => match serde_json::from_value::<CommentListing>(second.clone()) {
            Ok(c) => c.data.children.into_iter().map(|t| t.data).collect(),
            Err(e) => return Err(format!("thread comments shape drifted: {e}")),
        },
        None => Vec::new(),
    };
    Ok(ThreadView { post, comments })
}

// ---------------------------------------------------------------------------
// The signed-out read path (Atom → the same shapes as the API)
// ---------------------------------------------------------------------------

/// `/r/<sub>/<sort>/.rss?limit=…` — the listing a signed-out box can read.
/// An empty sub means the site-wide `all` feed.
pub fn rss_listing_url(base: &str, sub: &str, sort: &str, limit: u32, window: &str) -> String {
    let base = base.trim_end_matches('/');
    let sort = match sort {
        "hot" | "new" | "top" | "rising" | "controversial" => sort,
        _ => "new",
    };
    let sub = if sub.trim().is_empty() {
        "all".to_string()
    } else {
        sub.trim().to_string()
    };
    let mut url = format!("{base}/r/{sub}/{sort}/.rss?limit={limit}");
    if matches!(sort, "top" | "controversial") {
        url.push_str(&format!("&t={window}"));
    }
    url
}

/// `search.rss`, restricted to a sub when one is given — the signed-out
/// twin of [`search_url`].
pub fn rss_search_url(
    base: &str,
    sub: Option<&str>,
    query: &str,
    sort: &str,
    limit: u32,
    window: &str,
) -> String {
    let base = base.trim_end_matches('/');
    let q = urlencode(query.trim());
    let sort = match sort {
        "new" | "top" | "comments" => sort,
        _ => "relevance",
    };
    let mut url = match sub.filter(|s| !s.trim().is_empty()) {
        Some(sub) => format!(
            "{base}/r/{}/search.rss?q={q}&restrict_sr=1&sort={sort}&limit={limit}",
            sub.trim()
        ),
        None => format!("{base}/search.rss?q={q}&sort={sort}&limit={limit}"),
    };
    if sort == "top" {
        url.push_str(&format!("&t={window}"));
    }
    url
}

/// A post's comment feed: the only signed-out view of a thread.
pub fn rss_thread_url(base: &str, id: &str, limit: u32) -> String {
    format!(
        "{}/comments/{}/.rss?limit={limit}",
        base.trim_end_matches('/'),
        post_id_from(id)
    )
}

/// Decode the XML entities Atom uses (named + numeric). Unknown entities
/// pass through untouched rather than eating the rest of the string.
fn decode_entities(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        let tail = &rest[i..];
        let semi = match tail.find(';') {
            Some(semi) if semi > 1 && semi <= 10 => semi,
            _ => {
                out.push('&');
                rest = &tail[1..];
                continue;
            }
        };
        let ent = &tail[1..semi];
        let decoded: Option<String> = match ent {
            "amp" => Some("&".into()),
            "lt" => Some("<".into()),
            "gt" => Some(">".into()),
            "quot" => Some("\"".into()),
            "apos" => Some("'".into()),
            _ if ent.starts_with("#x") || ent.starts_with("#X") => {
                u32::from_str_radix(&ent[2..], 16)
                    .ok()
                    .and_then(char::from_u32)
                    .map(String::from)
            }
            _ if ent.starts_with('#') => ent[1..]
                .parse::<u32>()
                .ok()
                .and_then(char::from_u32)
                .map(String::from),
            _ => None,
        };
        match decoded {
            Some(text) => {
                out.push_str(&text);
                rest = &tail[semi + 1..];
            }
            None => {
                out.push('&');
                rest = &tail[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// Atom `<content>` is HTML escaped once more for XML: strip the tags,
/// decode what is left, and flatten the whitespace runs.
fn html_to_text(html: &str) -> String {
    let mut stripped = String::with_capacity(html.len());
    let mut in_tag = false;
    for ch in html.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => {
                in_tag = false;
                stripped.push(' ');
            }
            _ if !in_tag => stripped.push(ch),
            _ => {}
        }
    }
    let text = decode_entities(&stripped);
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    // Every tag boundary became a space, which is right between words and
    // wrong before punctuation: "docker </code>, honestly" must read as
    // "docker, honestly".
    let mut out = String::with_capacity(collapsed.len());
    for (i, ch) in collapsed.char_indices() {
        if ch == ' '
            && matches!(
                collapsed[i + 1..].chars().next(),
                Some(',' | '.' | ';' | ':' | '!' | '?' | ')' | ']')
            )
        {
            continue;
        }
        out.push(ch);
    }
    out
}

/// Text of the first `<tag …>…</tag>` in `entry`.
fn element_text(entry: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}");
    let start = entry.find(&open)?;
    let gt = start + entry[start..].find('>')? + 1;
    let close = format!("</{tag}>");
    let end = gt + entry[gt..].find(&close)?;
    let raw = entry[gt..end].trim();
    if let Some(inner) = raw
        .strip_prefix("<![CDATA[")
        .and_then(|r| r.strip_suffix("]]>"))
    {
        return Some(inner.to_string());
    }
    Some(decode_entities(raw))
}

/// `href` of the entry's self-closing `<link …/>`.
fn link_href(entry: &str) -> Option<String> {
    let start = entry.find("<link")?;
    let tag = &entry[start..];
    let end = tag.find('>')?;
    let tag = &tag[..end];
    let href = tag.find("href=\"")? + "href=\"".len();
    let rest = &tag[href..];
    let close = rest.find('"')?;
    Some(decode_entities(&rest[..close]))
}

/// `2026-10-29T01:49:15+00:00` → seconds since the epoch. Feeds are UTC,
/// so the offset is read and ignored; a missing seconds field is `:00`.
pub fn parse_rfc3339_secs(s: &str) -> Option<f64> {
    let s = s.trim();
    let (date, rest) = s.split_once('T')?;
    let mut parts = date.split('-');
    let year: i64 = parts.next()?.parse().ok()?;
    let month: i64 = parts.next()?.parse().ok()?;
    let day: i64 = parts.next()?.parse().ok()?;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    let clock = rest.split(['+', 'Z', 'z']).next().unwrap_or(rest);
    let clock = clock.split('-').next().unwrap_or(clock);
    let mut hms = clock.split(':');
    let hour: i64 = hms.next()?.parse().ok()?;
    let minute: i64 = hms.next().unwrap_or("0").parse().ok()?;
    let second: i64 = hms.next().unwrap_or("0").split('.').next()?.parse().ok()?;
    let days = days_from_civil(year, month, day);
    Some((days * 86_400 + hour * 3_600 + minute * 60 + second) as f64)
}

/// Howard Hinnant's civil→serial conversion (no chrono dependency).
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// `https://www.reddit.com/r/rust/comments/abc/x/` → `rust`.
fn sub_of_permalink(url: &str) -> String {
    let Some(idx) = url.find("/r/") else {
        return String::new();
    };
    let rest = &url[idx + 3..];
    rest.split('/').next().unwrap_or("").trim().to_string()
}

fn post_from_entry(entry: &str) -> Option<Post> {
    let id = element_text(entry, "id")
        .unwrap_or_default()
        .trim()
        .trim_start_matches("t3_")
        .to_string();
    let permalink = link_href(entry).unwrap_or_default();
    if id.is_empty() && permalink.is_empty() {
        return None;
    }
    let title = element_text(entry, "title")
        .unwrap_or_default()
        .trim()
        .to_string();
    let author = element_text(entry, "name")
        .unwrap_or_default()
        .trim()
        .trim_start_matches("/u/")
        .trim_start_matches("u/")
        .to_string();
    let created_utc = element_text(entry, "published")
        .or_else(|| element_text(entry, "updated"))
        .and_then(|t| parse_rfc3339_secs(&t))
        .unwrap_or(0.0);
    // A self post's body sits in <content>; a link post's is only the
    // "submitted by … [link] [comments]" boilerplate, which drops out at
    // the split and leaves an empty selftext.
    let content = element_text(entry, "content").unwrap_or_default();
    let text = html_to_text(&content);
    let selftext = text
        .split("submitted by")
        .next()
        .unwrap_or("")
        .trim()
        .to_string();
    let subreddit = sub_of_permalink(&permalink);
    Some(Post {
        id: id.clone(),
        name: if id.is_empty() {
            String::new()
        } else {
            format!("t3_{id}")
        },
        title,
        author,
        subreddit,
        url: permalink.clone(),
        permalink,
        is_self: !selftext.is_empty(),
        selftext,
        created_utc,
        metrics_known: false,
        ..Post::default()
    })
}

fn comment_from_entry(entry: &str) -> Option<Comment> {
    let id = element_text(entry, "id")?
        .trim()
        .trim_start_matches("t1_")
        .to_string();
    let author = element_text(entry, "name")
        .unwrap_or_default()
        .trim()
        .trim_start_matches("/u/")
        .trim_start_matches("u/")
        .to_string();
    let body = html_to_text(&element_text(entry, "content").unwrap_or_default());
    if body.trim().is_empty() {
        return None;
    }
    let created_utc = element_text(entry, "published")
        .or_else(|| element_text(entry, "updated"))
        .and_then(|t| parse_rfc3339_secs(&t))
        .unwrap_or(0.0);
    Some(Comment {
        id,
        author,
        body,
        created_utc,
        permalink: link_href(entry),
        metrics_known: false,
        ..Comment::default()
    })
}

fn entries_of(body: &str) -> Vec<&str> {
    body.split("<entry>")
        .skip(1)
        .map(|chunk| chunk.split("</entry>").next().unwrap_or(chunk))
        .collect()
}

/// Parse an Atom listing into the same [`Post`] rows the API path returns.
/// A feed with no entries is an empty page, not an error; anything without
/// Atom structure is an error that quotes what came back.
pub fn parse_atom_posts(body: &str) -> Result<Vec<Post>, String> {
    if !body.contains("<entry") {
        if body.contains("<feed") {
            return Ok(Vec::new());
        }
        return Err(format!(
            "reddit's public feed returned no Atom: {}",
            clip(body.trim(), 160)
        ));
    }
    Ok(entries_of(body)
        .into_iter()
        .filter_map(post_from_entry)
        .collect())
}

/// Parse a post's comment feed.
pub fn parse_atom_comments(body: &str) -> Result<Vec<Comment>, String> {
    if !body.contains("<entry") {
        if body.contains("<feed") {
            return Ok(Vec::new());
        }
        return Err(format!(
            "reddit's comment feed returned no Atom: {}",
            clip(body.trim(), 160)
        ));
    }
    Ok(entries_of(body)
        .into_iter()
        .filter_map(comment_from_entry)
        .collect())
}

/// "▲12", or "▲?" when the source carries no score at all.
pub fn score_label(score: i64, known: bool) -> String {
    if known {
        format!("▲{score}")
    } else {
        "▲?".to_string()
    }
}

/// "14 comment(s)", or the honest placeholder on the feed path.
fn comments_label(count: u32, known: bool) -> String {
    if known {
        format!("{count} comment(s)")
    } else {
        "comments not in feed".to_string()
    }
}

// ---------------------------------------------------------------------------
// Write forms (pure; main.rs sends them)
// ---------------------------------------------------------------------------

pub fn vote_form(thing: &str, dir: i8) -> Vec<(String, String)> {
    vec![
        ("id".into(), fullname(thing)),
        ("dir".into(), dir.to_string()),
    ]
}

pub fn submit_form(
    kind: &str,
    sub: &str,
    title: &str,
    text: &str,
    url: &str,
) -> Vec<(String, String)> {
    let mut form = vec![
        ("api_type".into(), "json".into()),
        ("kind".into(), kind.to_string()),
        ("sr".into(), sub.to_string()),
        ("title".into(), title.to_string()),
        ("resubmit".into(), "true".into()),
        ("sendreplies".into(), "true".into()),
    ];
    if kind == "link" {
        form.push(("url".into(), url.to_string()));
    } else {
        form.push(("text".into(), text.to_string()));
    }
    form
}

pub fn comment_form(thing: &str, text: &str) -> Vec<(String, String)> {
    vec![
        ("api_type".into(), "json".into()),
        ("thing_id".into(), fullname(thing)),
        ("text".into(), text.to_string()),
    ]
}

pub fn edit_form(thing: &str, text: &str) -> Vec<(String, String)> {
    vec![
        ("api_type".into(), "json".into()),
        ("thing_id".into(), fullname(thing)),
        ("text".into(), text.to_string()),
    ]
}

/// Reddit's write replies carry `{"json":{"errors":[[code,msg,field]]}}`.
/// An empty list is success; anything else is the honest refusal, with
/// the field named when Reddit names it.
pub fn form_errors(v: &Value) -> Option<String> {
    let errors = v.pointer("/json/errors")?.as_array()?;
    if errors.is_empty() {
        return None;
    }
    let parts: Vec<String> = errors
        .iter()
        .filter_map(|e| {
            let arr = e.as_array()?;
            let code = arr.first().and_then(Value::as_str).unwrap_or("?");
            let msg = arr.get(1).and_then(Value::as_str).unwrap_or("");
            let field = arr.get(2).and_then(Value::as_str).filter(|f| !f.is_empty());
            Some(match field {
                Some(f) => format!("{code}: {msg} [{f}]"),
                None => format!("{code}: {msg}"),
            })
        })
        .collect();
    Some(if parts.is_empty() {
        "reddit reported an error with no detail".to_string()
    } else {
        parts.join("; ")
    })
}

/// Where a successful submit/comment lands (the URL Reddit echoes), or
/// the thing's fullname when only an id came back.
pub fn write_result(v: &Value) -> Value {
    let data = v.pointer("/json/data").cloned().unwrap_or(Value::Null);
    json!({
        "url": data.get("url").cloned().unwrap_or(Value::Null),
        "id": data.get("id").cloned().unwrap_or(Value::Null),
        "name": data.get("name").cloned().unwrap_or(Value::Null),
    })
}

// ---------------------------------------------------------------------------
// The propose-and-confirm outbox (writes)
// ---------------------------------------------------------------------------

/// One pending write: what it will do, to where, and the single-use token
/// that lets the confirmed call go through.
#[derive(Debug, Clone, PartialEq)]
pub struct Proposal {
    pub id: u32,
    pub token: String,
    pub action: String,
    pub payload: Value,
    pub expires_ms: u64,
}

/// In-memory proposals. Nothing is persisted on purpose: a restart
/// voids every unconfirmed write rather than resurrecting a draft whose
/// context is gone.
#[derive(Debug, Default)]
pub struct Outbox {
    next: u32,
    pending: HashMap<u32, Proposal>,
}

impl Outbox {
    pub fn new() -> Self {
        Self::default()
    }

    /// Mint a proposal (fresh random token, 10-minute TTL).
    pub fn propose(&mut self, action: &str, payload: Value, now_ms: u64) -> Proposal {
        let token = format!("{:08x}", rand::random::<u32>());
        self.propose_with_token(action, payload, now_ms, token)
    }

    /// Deterministic variant for tests.
    pub fn propose_with_token(
        &mut self,
        action: &str,
        payload: Value,
        now_ms: u64,
        token: String,
    ) -> Proposal {
        self.prune(now_ms);
        self.next = self.next.wrapping_add(1).max(1);
        let p = Proposal {
            id: self.next,
            token,
            action: action.to_string(),
            payload,
            expires_ms: now_ms + PENDING_TTL_MS,
        };
        self.pending.insert(p.id, p.clone());
        p
    }

    /// Spend a proposal. Wrong id, wrong token, expired, or already spent
    /// all refuse with the reason — never a silent no-op.
    pub fn confirm(&mut self, id: u32, token: &str, now_ms: u64) -> Result<Proposal, String> {
        let Some(p) = self.pending.get(&id) else {
            return Err(format!(
                "proposal #{id} is not pending — call the skill once with no confirm token to get a fresh preview"
            ));
        };
        if p.token != token.trim() {
            return Err("confirm_token does not match that proposal".into());
        }
        if now_ms > p.expires_ms {
            self.pending.remove(&id);
            return Err(
                "that proposal expired (10-minute window) — re-preview and confirm again".into(),
            );
        }
        let p = self.pending.remove(&id).expect("checked above");
        Ok(p)
    }

    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    fn prune(&mut self, now_ms: u64) {
        self.pending.retain(|_, p| p.expires_ms >= now_ms);
    }
}

// ---------------------------------------------------------------------------
// Watch list (the lookout)
// ---------------------------------------------------------------------------

/// Watch state: subs, match terms, per-sub `created_utc` watermarks, and
/// a bounded seen-id ring so a restart doesn't re-announce Monday.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Watch {
    #[serde(default)]
    pub subs: Vec<String>,
    #[serde(default)]
    pub terms: Vec<String>,
    #[serde(default)]
    pub cursor: BTreeMap<String, f64>,
    #[serde(default)]
    pub seen: VecDeque<String>,
    /// Delivered matches, kept for the morning digest. Recording does NOT
    /// consume the cursor: the monitor's tick and the briefing's `recent`
    /// read can both see the same match, and neither steals it.
    #[serde(default)]
    pub inbox: VecDeque<Caught>,
    /// When each sub was last ATTEMPTED, success or failure. The signed-out
    /// path may only fetch one sub per tick, so rotation has to count a
    /// throttled attempt too — otherwise the same first sub is retried
    /// forever and every sub behind it starves.
    #[serde(default)]
    pub last_try: BTreeMap<String, f64>,
    /// What the last tick actually managed to do. Persisted because the
    /// morning digest reads `recent` (which never advances anything) and a
    /// throttled tick and a quiet tick otherwise look identical: zero
    /// matches. `None` until a check has run.
    #[serde(default)]
    pub last_check: Option<CheckReport>,
}

/// The outcome of one watch tick, as the file remembers it.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct CheckReport {
    #[serde(default)]
    pub at_ms: u64,
    /// Subs that were actually read this tick.
    #[serde(default)]
    pub checked: Vec<String>,
    /// Subs the signed-out budget could not reach this tick.
    #[serde(default)]
    pub deferred: Vec<String>,
    #[serde(default)]
    pub errors: Vec<String>,
}

impl Watch {
    pub fn add_sub(&mut self, raw: &str) -> bool {
        let sub = normalize_sub(raw);
        if sub.is_empty() || self.subs.iter().any(|s| s == &sub) {
            return false;
        }
        self.subs.push(sub);
        true
    }

    pub fn remove_sub(&mut self, raw: &str) -> bool {
        let sub = normalize_sub(raw);
        let before = self.subs.len();
        self.subs.retain(|s| s != &sub);
        self.cursor.remove(&sub);
        self.subs.len() != before
    }

    /// Comma-separated terms, lowercased, deduped. Returns how many were
    /// actually new.
    pub fn add_terms(&mut self, csv: &str) -> usize {
        let mut added = 0;
        for term in csv.split(',') {
            let t = term.trim().to_lowercase();
            if t.is_empty() || self.terms.iter().any(|x| x == &t) {
                continue;
            }
            self.terms.push(t);
            added += 1;
        }
        added
    }

    pub fn remove_term(&mut self, term: &str) -> bool {
        let t = term.trim().to_lowercase();
        let before = self.terms.len();
        self.terms.retain(|x| x != &t);
        self.terms.len() != before
    }

    /// No terms = every new post in the watched subs. Otherwise any term
    /// matching title or body (case-insensitive substring).
    pub fn matches(&self, p: &Post) -> bool {
        if self.terms.is_empty() {
            return true;
        }
        let hay = format!("{} {}", p.title, p.selftext).to_lowercase();
        self.terms.iter().any(|t| hay.contains(t))
    }

    pub fn unseen(&self, p: &Post) -> bool {
        !self.seen.iter().any(|s| s == &p.id)
    }

    /// Record an attempt (the cursor only moves on a successful read).
    pub fn tried(&mut self, sub: &str, at_secs: f64) {
        self.last_try.insert(sub.to_string(), at_secs);
    }

    /// `0.0` = never attempted; oldest attempt sorts first.
    pub fn last_try_of(&self, sub: &str) -> f64 {
        self.last_try.get(sub).copied().unwrap_or(0.0)
    }

    pub fn cursor_of(&self, sub: &str) -> f64 {
        self.cursor.get(sub).copied().unwrap_or(0.0)
    }

    /// Record one post: seen + advance the sub's watermark.
    pub fn mark(&mut self, p: &Post) {
        if !p.id.is_empty() && self.unseen(p) {
            self.seen.push_back(p.id.clone());
            while self.seen.len() > SEEN_CAP {
                self.seen.pop_front();
            }
        }
        let sub = p.subreddit.to_lowercase();
        let cur = self.cursor_of(&sub);
        if p.created_utc > cur {
            self.cursor.insert(sub, p.created_utc);
        }
    }

    /// Keep the delivered matches for the digest. Deduped by post id (a
    /// re-check cannot double-count), pruned by retention, capped. Returns
    /// how many rows were actually new.
    pub fn record(&mut self, matches: &[Post], now_ms: u64) -> usize {
        self.prune_inbox(now_ms);
        let mut added = 0;
        for p in matches {
            if p.id.is_empty() || self.inbox.iter().any(|c| c.id == p.id) {
                continue;
            }
            self.inbox.push_back(Caught::of(p, now_ms));
            added += 1;
        }
        while self.inbox.len() > INBOX_CAP {
            self.inbox.pop_front();
        }
        added
    }

    /// Digest rows caught at/after `cutoff_ms`, newest first.
    pub fn caught_since(&self, cutoff_ms: u64) -> Vec<&Caught> {
        let mut out: Vec<&Caught> = self
            .inbox
            .iter()
            .filter(|c| c.caught_at_ms >= cutoff_ms)
            .collect();
        out.sort_by_key(|c| std::cmp::Reverse(c.caught_at_ms));
        out
    }

    /// Drop digest rows past the retention window.
    pub fn prune_inbox(&mut self, now_ms: u64) {
        let cutoff = now_ms.saturating_sub(INBOX_RETENTION_MS);
        self.inbox.retain(|c| c.caught_at_ms >= cutoff);
    }
}

/// One match as the digest remembers it (what the morning note prints —
/// not the whole post, so a week of rows stays a small file).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Caught {
    pub id: String,
    pub subreddit: String,
    pub title: String,
    pub score: i64,
    pub permalink: String,
    pub created_utc: f64,
    pub caught_at_ms: u64,
    /// Copied from the post so the digest can print `▲?` for a row caught
    /// on the signed-out feed — a stored 0 would read as "nobody upvoted".
    #[serde(default = "metrics_known_default")]
    pub metrics_known: bool,
}

impl Caught {
    pub fn of(p: &Post, caught_at_ms: u64) -> Self {
        Self {
            id: p.id.clone(),
            subreddit: p.subreddit.clone(),
            title: p.title.clone(),
            score: p.score,
            permalink: p.permalink_url(),
            created_utc: p.created_utc,
            caught_at_ms,
            metrics_known: p.metrics_known,
        }
    }
}

/// `$JERITH_REDDIT_WATCH`, else `~/.jerith-mesh/reddit/watch.json`.
pub fn watch_path() -> PathBuf {
    if let Ok(p) = std::env::var("JERITH_REDDIT_WATCH")
        && !p.trim().is_empty()
    {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".jerith-mesh/reddit/watch.json")
}

/// Load the watch file; a missing or unreadable file is an empty watch
/// (the lane is opt-in by configuring terms, never by file existence).
pub fn load_watch(path: &Path) -> Watch {
    match std::fs::read_to_string(path) {
        Ok(raw) => serde_json::from_str(&raw).unwrap_or_else(|e| {
            tracing::warn!(path = %path.display(), error = %e, "reddit watch file unreadable — starting empty");
            Watch::default()
        }),
        Err(_) => Watch::default(),
    }
}

/// Atomic save (temp + rename): a killed process can't halve the file.
pub fn save_watch(path: &Path, w: &Watch) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(w).unwrap_or_default())?;
    std::fs::rename(&tmp, path)
}

/// New posts for this watch: strictly after the sub's watermark, unseen,
/// term-matching, newest first, capped at `limit`.
pub fn select_matches(posts: &[Post], w: &Watch, limit: usize) -> Vec<Post> {
    let mut out: Vec<Post> = posts
        .iter()
        .filter(|p| p.created_utc > w.cursor_of(&p.subreddit.to_lowercase()))
        .filter(|p| w.unseen(p))
        .filter(|p| w.matches(p))
        .cloned()
        .collect();
    out.sort_by(|a, b| {
        b.created_utc
            .partial_cmp(&a.created_utc)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out.truncate(limit.min(MAX_MATCHES));
    out
}

// ---------------------------------------------------------------------------
// Cards
// ---------------------------------------------------------------------------

fn clip(s: &str, n: usize) -> String {
    let mut c = n.min(s.len());
    while !s.is_char_boundary(c) {
        c -= 1;
    }
    if c < s.len() {
        format!("{}…", s[..c].trim_end())
    } else {
        s.to_string()
    }
}

/// "3m ago" / "5h ago" / "2d ago"; clock skew (future) reads "now".
pub fn relative_age(created_utc: f64, now_secs: f64) -> String {
    let secs = (now_secs - created_utc).max(0.0) as u64;
    match secs {
        0..=59 => "now".to_string(),
        60..=3599 => format!("{}m ago", secs / 60),
        3600..=86_399 => format!("{}h ago", secs / 3600),
        _ => format!("{}d ago", secs / 86_400),
    }
}

pub fn post_json(p: &Post, now_secs: f64) -> Value {
    json!({
        "id": p.id,
        "fullname": p.fullname(),
        "subreddit": p.subreddit,
        "title": p.title,
        "author": p.author,
        "score": if p.metrics_known { json!(p.score) } else { Value::Null },
        "comments": if p.metrics_known {
            json!(p.num_comments)
        } else {
            Value::Null
        },
        "metrics_known": p.metrics_known,
        "age": relative_age(p.created_utc, now_secs),
        "created_utc": p.created_utc,
        "permalink": p.permalink_url(),
        "url": p.url,
        "flair": p.link_flair_text,
        "nsfw": p.over_18,
        "self": p.is_self,
    })
}

pub fn post_line(p: &Post, now_secs: f64) -> String {
    format!(
        "{score} · r/{sub} · {age} · u/{author}\n   {title}\n   {link}",
        score = score_label(p.score, p.metrics_known),
        sub = p.subreddit,
        age = relative_age(p.created_utc, now_secs),
        author = if p.author.is_empty() {
            "[deleted]"
        } else {
            &p.author
        },
        title = clip(&p.title, 110),
        link = p.permalink_url(),
    )
}

/// The listing card: a header plus one 3-line block per post.
pub fn listing_card(label: &str, posts: &[Post], now_secs: f64) -> String {
    if posts.is_empty() {
        return format!("r/{label} — nothing there right now");
    }
    let mut out = format!("📰 **r/{label}** · {} post(s)\n", posts.len());
    for p in posts.iter().take(10) {
        out.push_str(&post_line(p, now_secs));
        out.push('\n');
    }
    if posts.len() > 10 {
        out.push_str(&format!("_…and {} more._\n", posts.len() - 10));
    }
    out.trim_end().to_string()
}

/// The thread card: post block plus the top comments, bodies clipped.
pub fn thread_card(view: &ThreadView, now_secs: f64) -> String {
    let mut out = format!(
        "🧵 {}\n   {}\n",
        clip(&view.post.title, 110),
        view.post.permalink_url()
    );
    if view.post.is_self && !view.post.selftext.is_empty() {
        out.push_str(&format!("   {}\n", clip(&view.post.selftext, 300)));
    }
    out.push_str(&format!(
        "{} · u/{} · {} · {}\n",
        score_label(view.post.score, view.post.metrics_known),
        view.post.author,
        comments_label(view.post.num_comments, view.post.metrics_known),
        relative_age(view.post.created_utc, now_secs)
    ));
    for c in view.comments.iter().take(8) {
        out.push_str(&format!(
            "\n  • u/{} ▲{} — {}\n",
            if c.author.is_empty() {
                "[deleted]"
            } else {
                &c.author
            },
            score_label(c.score, c.metrics_known),
            clip(&c.body.replace('\n', " "), 240)
        ));
        if !c.id.is_empty() {
            out.push_str(&format!("    t1_{}\n", c.id));
        }
    }
    out.trim_end().to_string()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn creds(mode: AuthMode) -> Credentials {
        let mut c = Credentials {
            client_id: "cid".into(),
            client_secret: "csecret".into(),
            user_agent: "linux:jerith-mesh:0.1.4 (by /u/tester)".into(),
            ..Credentials::default()
        };
        match mode {
            AuthMode::RefreshToken => c.refresh_token = Some("rt".into()),
            AuthMode::Password => {
                c.username = Some("user".into());
                c.password = Some("pw".into());
            }
            AuthMode::AppOnly => {}
        }
        c
    }

    #[test]
    fn auth_mode_prefers_the_refresh_token() {
        let mut c = creds(AuthMode::RefreshToken);
        assert_eq!(c.auth_mode(), AuthMode::RefreshToken);
        // A password pair sitting alongside still loses to the token.
        c.username = Some("u".into());
        c.password = Some("p".into());
        assert_eq!(c.auth_mode(), AuthMode::RefreshToken);
        assert!(c.auth_mode().can_write());
    }

    #[test]
    fn token_form_matches_reddits_three_flows() {
        let rt = creds(AuthMode::RefreshToken).token_form();
        assert_eq!(
            rt,
            vec![
                ("grant_type".to_string(), "refresh_token".into()),
                ("refresh_token".to_string(), "rt".into())
            ]
        );
        let pw = creds(AuthMode::Password).token_form();
        assert!(pw.contains(&("grant_type".to_string(), "password".to_string())));
        assert!(pw.contains(&("username".to_string(), "user".to_string())));
        assert!(pw.contains(&("password".to_string(), "pw".to_string())));
        let app = creds(AuthMode::AppOnly).token_form();
        assert_eq!(
            app,
            vec![("grant_type".into(), "client_credentials".into())]
        );
        assert!(!AuthMode::AppOnly.can_write());
    }

    #[test]
    fn basic_auth_header_is_reddits_credential_pair() {
        let h = creds(AuthMode::AppOnly).basic_auth_header();
        use base64::Engine as _;
        let b64 = h.strip_prefix("Basic ").expect("Basic prefix");
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(b64)
            .expect("valid base64");
        assert_eq!(String::from_utf8(decoded).unwrap(), "cid:csecret");
    }

    #[test]
    fn validate_names_each_missing_key() {
        let err = Credentials::default().validate().unwrap_err();
        assert!(err.contains("client_id"), "{err}");
        assert!(err.contains("client_secret"), "{err}");
        assert!(err.contains("user_agent"), "{err}");
        let mut c = creds(AuthMode::RefreshToken);
        c.user_agent = String::new();
        let err = c.validate().unwrap_err();
        assert!(
            err.contains("user_agent") && !err.contains("client_id"),
            "{err}"
        );
        assert!(creds(AuthMode::RefreshToken).validate().is_ok());
        assert!(creds(AuthMode::AppOnly).validate().is_ok());
    }

    #[test]
    fn token_errors_are_actionable() {
        let err = parse_token(r#"{"error":"invalid_grant"}"#).unwrap_err();
        assert!(err.contains("refresh token"), "{err}");
        let err = parse_token(r#"{"error":"unsupported_grant_type"}"#).unwrap_err();
        assert!(err.contains("script"), "{err}");
        let ok = parse_token(r#"{"access_token":"abc","expires_in":86400,"scope":"*"}"#).unwrap();
        assert_eq!(ok.access_token, "abc");
        assert_eq!(ok.expires_in, 86400);
        assert!(parse_token("{}").is_err());
        assert!(parse_token("not json").is_err());
    }

    #[test]
    fn subs_normalize_from_every_common_spelling() {
        assert_eq!(normalize_sub("rust"), "rust");
        assert_eq!(normalize_sub(" r/Rust "), "rust");
        assert_eq!(normalize_sub("/r/rust/"), "rust");
        assert_eq!(normalize_sub("https://www.reddit.com/r/rust"), "rust");
        assert_eq!(normalize_sub("https://reddit.com/r/rust/"), "rust");
        assert_eq!(normalize_sub(""), "");
    }

    #[test]
    fn urls_pin_the_shapes_the_api_expects() {
        let l = listing_url(DEFAULT_BASE, "rust", "hot", 10, "day", None);
        assert_eq!(l, "https://oauth.reddit.com/r/rust/hot?limit=10&raw_json=1");
        let t = listing_url(DEFAULT_BASE, "rust", "top", 5, "week", Some("t3_abc"));
        assert!(t.contains("&t=week") && t.contains("&after=t3_abc"), "{t}");
        let s = search_url(DEFAULT_BASE, Some("rust"), "async fn", "top", 10, "month");
        assert!(
            s.starts_with("https://oauth.reddit.com/r/rust/search?q=async%20fn&restrict_sr=1")
                && s.contains("&t=month"),
            "{s}"
        );
        let site = search_url(DEFAULT_BASE, None, "jerith mesh", "relevance", 5, "day");
        assert_eq!(
            site,
            "https://oauth.reddit.com/search?q=jerith%20mesh&sort=relevance&limit=5&type=link&raw_json=1"
        );
        // An empty sub is the signed-in front page.
        let front = listing_url(DEFAULT_BASE, "", "new", 10, "day", None);
        assert_eq!(front, "https://oauth.reddit.com/new?limit=10&raw_json=1");
    }

    #[test]
    fn ids_come_from_fullnames_ids_and_permalinks() {
        assert_eq!(post_id_from("t3_abc"), "abc");
        assert_eq!(post_id_from("abc"), "abc");
        assert_eq!(
            post_id_from("https://www.reddit.com/r/rust/comments/abc/title_slug/?utm=1"),
            "abc"
        );
        assert_eq!(post_id_from("/r/rust/comments/abc/x?sort=new"), "abc");
        // Fullnames: prefixes are kept, bare ids default to a post, a
        // comment's t1_ survives (votes/comments target it).
        assert_eq!(fullname("abc"), "t3_abc");
        assert_eq!(fullname("t3_abc"), "t3_abc");
        assert_eq!(fullname("t1_def"), "t1_def");
        assert_eq!(fullname("https://reddit.com/r/x/comments/abc/y"), "t3_abc");
    }

    #[test]
    fn vote_directions_are_exactly_three() {
        assert_eq!(vote_dir("up").unwrap(), 1);
        assert_eq!(vote_dir("Upvote").unwrap(), 1);
        assert_eq!(vote_dir("down").unwrap(), -1);
        assert_eq!(vote_dir("-1").unwrap(), -1);
        assert_eq!(vote_dir("none").unwrap(), 0);
        assert_eq!(vote_dir("unvote").unwrap(), 0);
        assert!(vote_dir("sideways").is_err());
        assert!(vote_dir("").is_err());
    }

    #[test]
    fn listing_parses_children_and_cursor() {
        let body = r#"{"kind":"Listing","data":{"after":"t3_next","children":[
            {"kind":"t3","data":{"id":"a1","name":"t3_a1","title":"Async Rust","author":"ann",
             "subreddit":"rust","score":42,"num_comments":7,"permalink":"/r/rust/comments/a1/x/",
             "created_utc":1700000000.0,"selftext":"hello","is_self":true}},
            {"kind":"t3","data":{"id":"a2","title":"Second","subreddit":"rust","score":-3}}
        ]}}"#;
        let page = parse_listing(body).unwrap();
        assert_eq!(page.posts.len(), 2);
        assert_eq!(page.after.as_deref(), Some("t3_next"));
        assert_eq!(page.posts[0].fullname(), "t3_a1");
        assert_eq!(
            page.posts[0].permalink_url(),
            "https://www.reddit.com/r/rust/comments/a1/x/"
        );
        // Defaults survive a sparse row.
        assert_eq!(page.posts[1].score, -3);
        assert_eq!(page.posts[1].author, "");
        assert!(parse_listing(r#"{"message":"Forbidden","error":403}"#).is_err());
        assert!(parse_listing(r#"{"nope":1}"#).is_err());
        assert!(parse_listing("not json").is_err());
    }

    #[test]
    fn thread_parses_post_and_top_level_comments() {
        let body = r#"[
          {"kind":"Listing","data":{"children":[{"kind":"t3","data":{
             "id":"p1","title":"Thread","author":"op","subreddit":"rust","score":9,
             "num_comments":2,"permalink":"/r/rust/comments/p1/t/","selftext":"body","is_self":true,
             "created_utc":1700000000.0}}]}},
          {"kind":"Listing","data":{"children":[
             {"kind":"t1","data":{"id":"c1","author":"bob","body":"first\nsecond","score":5,
              "created_utc":1700000100.0,"parent_id":"t3_p1"}},
             {"kind":"more","data":{"count":3}}
          ]}}
        ]"#;
        let view = parse_thread(body).unwrap();
        assert_eq!(view.post.id, "p1");
        assert_eq!(
            view.comments.len(),
            2,
            "the `more` row parses as a blank comment"
        );
        assert_eq!(view.comments[0].body, "first\nsecond");
        assert_eq!(view.comments[0].parent_id, "t3_p1");
        let card = thread_card(&view, 1_700_000_500.0);
        assert!(card.contains("Thread") && card.contains("bob"), "{card}");
        assert!(card.contains("t1_c1"), "{card}");
        assert!(card.len() < 65536, "cards stay inside the dispatcher cap");
        assert!(parse_thread("{}").is_err());
    }

    #[test]
    fn outbox_is_two_step_and_single_use() {
        let mut box_ = Outbox::new();
        let p = box_.propose("submit", json!({"subreddit":"rust","title":"Hi"}), 1_000);
        assert_eq!(p.id, 1);
        assert_eq!(box_.len(), 1);
        // Wrong token refuses and keeps the proposal.
        let err = box_.confirm(1, "deadbeef", 2_000).unwrap_err();
        assert!(err.contains("does not match"), "{err}");
        assert_eq!(box_.len(), 1);
        // Right token spends it.
        let spent = box_.confirm(1, &p.token, 3_000).unwrap();
        assert_eq!(spent.payload["title"], "Hi");
        assert_eq!(box_.len(), 0);
        // Replaying the same confirmation refuses.
        assert!(box_.confirm(1, &p.token, 3_000).is_err());
        // Unknown id has its own message.
        assert!(
            box_.confirm(99, "x", 3_000)
                .unwrap_err()
                .contains("not pending")
        );
    }

    #[test]
    fn outbox_expires_after_the_ttl() {
        let mut box_ = Outbox::new();
        let p = box_.propose_with_token("comment", json!({"text":"x"}), 0, "tok".into());
        assert!(box_.confirm(p.id, "tok", PENDING_TTL_MS - 1).is_ok());
        let p2 = box_.propose_with_token("comment", json!({"text":"y"}), 5_000, "tok2".into());
        let err = box_
            .confirm(p2.id, "tok2", 5_000 + PENDING_TTL_MS + 1)
            .unwrap_err();
        assert!(err.contains("expired"), "{err}");
        assert!(box_.is_empty());
    }

    #[test]
    fn write_forms_carry_the_fields_reddit_documents() {
        let s = submit_form("self", "rust", "Title", "Body", "");
        assert!(s.contains(&("kind".to_string(), "self".to_string())));
        assert!(s.contains(&("text".to_string(), "Body".to_string())));
        assert!(s.contains(&("api_type".to_string(), "json".to_string())));
        assert!(!s.iter().any(|(k, _)| k == "url"), "self posts send no url");
        let l = submit_form("link", "rust", "Title", "", "https://example.org");
        assert!(l.contains(&("url".to_string(), "https://example.org".to_string())));
        assert!(
            !l.iter().any(|(k, _)| k == "text"),
            "link posts send no text"
        );
        let c = comment_form("t1_abc", "hi");
        assert!(c.contains(&("thing_id".to_string(), "t1_abc".to_string())));
        let e = edit_form("abc", "revised");
        assert!(e.contains(&("thing_id".to_string(), "t3_abc".to_string())));
        let v = vote_form("t1_abc", -1);
        assert_eq!(v[1], ("dir".to_string(), "-1".to_string()));
    }

    #[test]
    fn form_errors_are_the_refusal_not_a_success() {
        assert_eq!(
            form_errors(&json!({"json":{"errors":[]}})),
            None,
            "empty errors = success"
        );
        let msg = form_errors(
            &json!({"json":{"errors":[["RATELIMIT","you are doing that too much",""] ]}}),
        )
        .expect("an error");
        assert!(msg.contains("RATELIMIT"), "{msg}");
        let msg = form_errors(&json!({"json":{"errors":[["NO_TEXT","empty body","text"]]}}))
            .expect("an error");
        assert!(msg.contains("[text]"), "{msg}");
        // A body without the json envelope is not "success" by default.
        assert_eq!(form_errors(&json!({"ok":true})), None);
        let r = write_result(&json!({"json":{"data":{"url":"https://x/y","name":"t3_1"}}}));
        assert_eq!(r["url"], "https://x/y");
        assert_eq!(r["name"], "t3_1");
    }

    #[test]
    fn watch_matches_terms_and_ignores_repeats() {
        let mut w = Watch::default();
        assert!(w.add_sub("rust"));
        assert!(!w.add_sub("r/rust"), "duplicate sub");
        assert_eq!(w.add_terms("async,Tokio, async"), 2);
        assert_eq!(w.terms, vec!["async", "tokio"]);

        let hit = Post {
            id: "p1".into(),
            title: "Async runtimes compared".into(),
            subreddit: "rust".into(),
            created_utc: 100.0,
            ..Post::default()
        };
        let miss = Post {
            id: "p2".into(),
            title: "Cooking".into(),
            subreddit: "rust".into(),
            created_utc: 100.0,
            ..Post::default()
        };
        assert!(w.matches(&hit));
        assert!(!w.matches(&miss));
        // No terms at all = everything new in the sub.
        let all = Watch {
            subs: vec!["rust".into()],
            ..Watch::default()
        };
        assert!(all.matches(&miss));
    }

    #[test]
    fn watch_cursor_and_seen_ring_suppress_replays() {
        let mut w = Watch::default();
        w.add_sub("rust");
        let posts = vec![
            Post {
                id: "old".into(),
                title: "old".into(),
                subreddit: "rust".into(),
                created_utc: 10.0,
                ..Post::default()
            },
            Post {
                id: "new".into(),
                title: "new".into(),
                subreddit: "rust".into(),
                created_utc: 20.0,
                ..Post::default()
            },
            Post {
                id: "newest".into(),
                title: "newest".into(),
                subreddit: "rust".into(),
                created_utc: 30.0,
                ..Post::default()
            },
        ];
        let first = select_matches(&posts, &w, 10);
        assert_eq!(first.len(), 3);
        for p in &first {
            w.mark(p);
        }
        assert_eq!(w.cursor_of("rust"), 30.0);
        // Same page again → nothing.
        assert!(select_matches(&posts, &w, 10).is_empty());
        // A newer post → exactly one.
        let mut more = posts.clone();
        more.push(Post {
            id: "later".into(),
            title: "later".into(),
            subreddit: "rust".into(),
            created_utc: 40.0,
            ..Post::default()
        });
        let again = select_matches(&more, &w, 10);
        assert_eq!(again.len(), 1);
        assert_eq!(again[0].id, "later");
    }

    #[test]
    fn watch_limits_and_ring_stay_bounded() {
        let w = Watch::default();
        let mut posts = Vec::new();
        for i in 0..(MAX_MATCHES + 20) {
            posts.push(Post {
                id: format!("p{i}"),
                title: format!("t{i}"),
                subreddit: "all".into(),
                created_utc: i as f64,
                ..Post::default()
            });
        }
        assert_eq!(select_matches(&posts, &w, 10).len(), 10);
        assert_eq!(select_matches(&posts, &w, 1000).len(), MAX_MATCHES);

        let mut w2 = Watch::default();
        for i in 0..(SEEN_CAP + 50) {
            w2.mark(&Post {
                id: format!("x{i}"),
                subreddit: "all".into(),
                created_utc: i as f64,
                ..Post::default()
            });
        }
        assert_eq!(w2.seen.len(), SEEN_CAP, "the ring is capped");
    }

    #[test]
    fn watch_round_trips_through_disk_atomically() {
        let dir = std::env::temp_dir().join(format!("reddit-watch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("watch.json");
        assert!(load_watch(&path).subs.is_empty(), "missing file = empty");
        let mut w = Watch::default();
        w.add_sub("rust");
        w.add_terms("async");
        w.mark(&Post {
            id: "p1".into(),
            subreddit: "rust".into(),
            created_utc: 5.0,
            ..Post::default()
        });
        save_watch(&path, &w).unwrap();
        let back = load_watch(&path);
        assert_eq!(back, w);
        // A corrupt file degrades to empty, never panics.
        std::fs::write(&path, "{not json").unwrap();
        assert!(load_watch(&path).subs.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn digest_records_dedupe_prunes_and_caps() {
        let now = 10_000_000u64;
        let p = Post {
            id: "p1".into(),
            title: "Async Rust on a NUC".into(),
            subreddit: "rust".into(),
            created_utc: 1.0,
            score: 3,
            permalink: "/r/rust/comments/p1/x/".into(),
            ..Post::default()
        };
        let mut w = Watch::default();
        assert_eq!(w.record(std::slice::from_ref(&p), now), 1);
        assert_eq!(
            w.record(std::slice::from_ref(&p), now + 1),
            0,
            "same id can't double-count"
        );
        assert_eq!(w.caught_since(now - 1_000).len(), 1);
        assert_eq!(w.caught_since(now + 1_000).len(), 0, "cutoff is exclusive");
        // Retention: a row older than a week falls out on the next record.
        let late = now + INBOX_RETENTION_MS + 1;
        assert_eq!(
            w.record(
                &[Post {
                    id: "p2".into(),
                    ..p.clone()
                }],
                late
            ),
            1
        );
        assert_eq!(w.inbox.len(), 1, "the week-old row was pruned");
        assert_eq!(w.inbox[0].id, "p2");
        // Cap.
        let mut w2 = Watch::default();
        for i in 0..(INBOX_CAP + 5) {
            w2.record(
                &[Post {
                    id: format!("i{i}"),
                    ..p.clone()
                }],
                now,
            );
        }
        assert_eq!(w2.inbox.len(), INBOX_CAP);
        // Digest rows carry what a morning line prints.
        let c = &w.inbox[0];
        assert_eq!(c.subreddit, "rust");
        assert_eq!(c.permalink, "https://www.reddit.com/r/rust/comments/p1/x/");
        assert_eq!(c.caught_at_ms, late);
    }

    #[test]
    fn removal_reports_whether_it_changed_anything() {
        let mut w = Watch::default();
        w.add_sub("rust");
        w.add_terms("async,tokio");
        assert!(w.remove_term("ASYNC"), "case-insensitive");
        assert!(!w.remove_term("async"), "already gone");
        assert!(w.remove_sub("r/rust"));
        assert!(!w.remove_sub("rust"));
        assert!(w.subs.is_empty() && w.terms == vec!["tokio".to_string()]);
    }

    #[test]
    fn cards_and_ages_stay_readable() {
        let now = 1_700_000_000.0;
        assert_eq!(relative_age(now, now), "now");
        assert_eq!(relative_age(now - 120.0, now), "2m ago");
        assert_eq!(relative_age(now - 7200.0, now), "2h ago");
        assert_eq!(relative_age(now - 172_800.0, now), "2d ago");
        assert_eq!(relative_age(now + 50.0, now), "now", "skew reads as now");
        let p = Post {
            id: "a".into(),
            title: "x".repeat(400),
            subreddit: "rust".into(),
            author: "ann".into(),
            score: 12,
            metrics_known: true,
            permalink: "/r/rust/comments/a/t/".into(),
            created_utc: now - 60.0,
            ..Post::default()
        };
        let line = post_line(&p, now);
        assert!(line.contains("▲12") && line.contains("r/rust") && line.contains("u/ann"));
        // The feed path has no score to print, so it asks instead of
        // claiming the post scored zero.
        let anon = Post {
            metrics_known: false,
            ..p.clone()
        };
        assert!(
            post_line(&anon, now).contains("▲?"),
            "{}",
            post_line(&anon, now)
        );
        assert!(line.chars().count() < 400, "titles are clipped");
        let empty = listing_card("rust", &[], now);
        assert!(empty.contains("nothing there"));
    }
}
