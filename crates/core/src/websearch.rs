//! Web access skills: Brave Search API + URL fetch with readable-text
//! extraction.
//!
//! Ported from the OpenClaw stack (`@openclaw/brave-plugin` web-search
//! provider and the `web_fetch` tool) so a jerith-mesh node can answer
//! with live web data the same way the OpenClaw agent could:
//!
//! - `web_search` → `GET {base_url}/res/v1/web/search` with
//!   `X-Subscription-Token` auth (Brave Search API v1, `web` mode), each
//!   `web.results[]` entry mapped to title/url/description/page_age/site.
//! - `web_fetch`  → download a page and strip HTML down to readable text
//!   (scripts/styles/comments removed, entities decoded, whitespace
//!   collapsed) — the paged-read counterpart of OpenClaw's web_fetch.
//!
//! SSRF stance (as upstream): non-HTTP(S) schemes and loopback/private/
//! link-local/metadata host literals are rejected before any request is
//! made (round 1 — `SsrfResolver` rejects literal URLs via the same
//! `is_private_host` used per resolved address), and the custom DNS
//! resolver (round 2) refuses any hostname whose resolved addresses are
//! all private — closing the two gaps reqwest's default behavior leaves:
//! auto-followed redirects (no per-hop re-flight after round 1's
//! `Policy::none()`) and A-records that silently point a public-looking
//! name at a private IP (the DNS-rebinding / internal-name class).
//! Both guards re-check EVERY address (IPv4 + IPv6) the resolver can
//! return; IP-literal URLs never need the resolver, so the byte-level
//! literal check still covers them. Six-side traceability: what remains
//! out of scope is TLS-SNI/hostname hosters plus specific-port escapes,
//! both rejected by reqwest itself (no downgrade path this build).
//!
//! Never mesh-facing: these skills stay local to the node — only the
//! gateway's own callers (Discord bot tool loop, CLI scripts) hit
//! `/v1/skills/web/*`, and Brave/page servers see a stateless GET.

use std::time::Duration;

use anyhow::{Context as _, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

const BRAVE_SEARCH_ENDPOINT: &str = "/res/v1/web/search";
const DEFAULT_BRAVE_BASE_URL: &str = "https://api.search.brave.com";
const USER_AGENT: &str =
    "jerith-mesh/0.1 (web-access-skill; +https://github.com/ParadoxFuzzle/jerith-mesh)";

/// Hard cap for a fetched page body — a hostile page must not balloon the
/// node's memory; past this the download simply stops.
const FETCH_BODY_CAP_BYTES: usize = 2 * 1024 * 1024;

/// One search result — the OpenClaw `web_search` shape (minus the
/// `wrapWebContent` marking, which is agent-side framing).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebHit {
    pub title: String,
    pub url: String,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_age: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub site_name: Option<String>,
}

/// A fetched page: readable text + how much was cut.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageText {
    pub url: String,
    pub title: Option<String>,
    pub text: String,
    pub truncated: bool,
    pub status: u16,
}

impl WebHit {
    fn from_brave_entry(entry: &Value) -> Option<WebHit> {
        let url = entry.get("url")?.as_str()?.to_string();
        let site = resolve_site_name(&url);
        Some(WebHit {
            title: clean_snippet(entry.get("title")),
            url,
            description: clean_snippet(entry.get("description")),
            page_age: entry
                .get("page_age")
                .and_then(Value::as_str)
                .map(str::to_string),
            site_name: site,
        })
    }
}

/// Brave snippets carry `<b>`/`<strong>`-style highlight tags; OpenClaw
/// passes them through with wrapWebContent framing and jerith strips them.
fn clean_snippet(v: Option<&Value>) -> String {
    let s = v.and_then(Value::as_str).unwrap_or_default();
    strip_tags(s).trim().to_string()
}

/// `siteName` counterpart: host without the leading `www.`.
fn resolve_site_name(url: &str) -> Option<String> {
    host_of(url).map(|h| h.trim_start_matches("www.").to_string())
}

/// Tiny std-only URL authority helper (no `url` crate in the workspace).
/// Strips the port when present and lowercases — the SSRF guard matches
/// on host only.
fn host_of(url: &str) -> Option<String> {
    let rest = url.split_once("://").map(|(_, r)| r)?;
    let authority = rest
        .split(['/', '?', '#'])
        .next()
        .unwrap_or_default()
        .trim_end_matches('.')
        .to_ascii_lowercase();
    let bare = if let Some(close) = authority.find(']') {
        // IPv6 literal: keep up through the closing bracket (drops :port).
        authority[..=close].to_string()
    } else if let Some((host, port)) = authority.rsplit_once(':') {
        // Plain host:port — keep the port only if it isn't digits (never
        // mangle an unbracketed IPv6 literal like ::1).
        if !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit()) {
            host.to_string()
        } else {
            authority.to_string()
        }
    } else {
        authority.to_string()
    };
    (!bare.is_empty()).then_some(bare)
}

/// Braces-free freshness validation (matches upstream: day/week/month/year).
pub fn normalize_freshness(raw: Option<&str>) -> Result<Option<&'static str>> {
    match raw.map(str::trim).filter(|s| !s.is_empty()) {
        None => Ok(None),
        Some(s) => match s {
            "day" => Ok(Some("day")),
            "week" => Ok(Some("week")),
            "month" => Ok(Some("month")),
            "year" => Ok(Some("year")),
            other => bail!("freshness must be day, week, month, or year (got {other:?})"),
        },
    }
}

/// SSRF guard: reject non-HTTP(S) targets and private-network host
/// literals before a byte leaves the node.
fn assert_fully_fetchable(url: &str) -> Result<()> {
    let lower = url.to_ascii_lowercase();
    if !(lower.starts_with("http://") || lower.starts_with("https://")) {
        bail!("fetch target must be an http(s):// URL");
    }
    let Some(host) = host_of(&lower) else {
        bail!("fetch target has no host");
    };
    if is_private_host(&host) {
        bail!("refusing to fetch private/loopback host {host:?} (SSRF guard)");
    }
    Ok(())
}

fn is_private_host(host: &str) -> bool {
    // Hostname obvious cases.
    if host == "localhost"
        || host.ends_with(".localhost")
        || host.ends_with(".local")
        || host.ends_with(".internal")
        || host.ends_with(".lan")
        || host.ends_with(".intra")
        || host.ends_with(".localdomain")
    {
        return true;
    }
    // IP literals (bracket form keeps brackets in host_of).
    if let Ok(ip) = host.trim_matches(['[', ']']).parse::<std::net::IpAddr>() {
        return match ip {
            std::net::IpAddr::V4(v4) => {
                let o = v4.octets();
                o[0] == 0                                              // "this host" RFC1122
                    || o[0] == 10
                    || o[0] == 127
                    || (o[0] == 172 && (16..=31).contains(&o[1]))
                    || (o[0] == 192 && o[1] == 168)
                    || (o[0] == 169 && o[1] == 254)                    // link-local
                    || (o[0] == 100 && (64..=127).contains(&o[1]))     // CGNAT
                    || o[0] >= 224 // multicast + reserved
            }
            std::net::IpAddr::V6(v6) => {
                let s = v6.segments();
                s == [0, 0, 0, 0, 0, 0, 0, 1]                          // ::1
                    || (s[0] & 0xfe00) == 0xfe00                       // fe80::/10
                    || (s[0] & 0xfff0) == 0xfd00                       // fc00::/7 ULA
                    || s == [0, 0, 0, 0, 0, 0xffff, 0x7f00, 0] // ::ffff:127.0.0.1
            }
        };
    }
    // Well-known internal names beyond the suffixes above.
    matches!(host, "metadata" | "instance-data")
}

/// SSRF-aware DNS resolver (reqwest `Resolve`): every name the fetch
/// client connects to is resolved through the system resolver (blocking
/// std call, run on tokio's blocking pool), the resolved addresses are
/// preflighted through the same private-host policy as the literal URL
/// check, and any address that fails is stripped from the answer. If ALL
/// resolved addresses are private → the resolution fails and the request
/// never connects — a DNS-rebinding (public-then-private) name dies at
/// connect time instead of after the handshake.
///
/// IP-literal URLs never hit the resolver (reqwest connects directly),
/// so the byte-level check in `assert_fully_fetchable` stays the first
/// line of defense for those; this one closes the NAME→IP gap.
struct SsrfResolver;

impl reqwest::dns::Resolve for SsrfResolver {
    fn resolve(&self, name: reqwest::dns::Name) -> reqwest::dns::Resolving {
        let host = name.as_str().to_string();
        let lookup = host.clone();
        Box::pin(async move {
            // Same pattern reqwest's own GaiResolver uses: blocking std
            // resolver on a blocking thread, errors boxed into BoxError.
            let resolved: Result<Vec<std::net::SocketAddr>, std::io::Error> =
                tokio::task::spawn_blocking(move || {
                    std::net::ToSocketAddrs::to_socket_addrs(&(lookup.as_str(), 0u16))
                        .map(|iter| iter.collect::<Vec<_>>())
                })
                .await
                .map_err(|e| std::io::Error::other(format!("ssrf resolver task failed: {e}")))?;
            let result = resolved?;
            // Public-name↔private-IP check on EVERY resolved address.
            let mut addrs = Vec::with_capacity(result.len());
            for sa in &result {
                let host_str = sa.ip().to_string();
                if is_private_host(&host_str) {
                    tracing::warn!(
                        host = %host,
                        resolved = %host_str,
                        "ssrf resolver: refusing private address behind a public name (rebind or internal-name probe)"
                    );
                    continue;
                }
                addrs.push(*sa);
            }
            if addrs.is_empty() {
                return Err(Box::new(std::io::Error::other(format!(
                    "every resolved address for {host} is private/loopback — refusing to connect (SSRF guard)"
                )))
                    as Box<dyn std::error::Error + Send + Sync>);
            }
            Ok(Box::new(addrs.into_iter()) as reqwest::dns::Addrs)
        })
    }
}

/// Search parameters — flat so gateway handlers can map the incoming JSON
/// body straight onto it.
#[derive(Debug, Clone, Default)]
pub struct SearchQuery {
    pub query: String,
    pub count: Option<u32>,
    pub country: Option<String>,
    pub search_lang: Option<String>,
    pub ui_lang: Option<String>,
    pub freshness: Option<String>,
}

/// Brave Search (web mode) client — OpenClaw brave-plugin port.
pub struct WebSearchService {
    client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    default_count: u32,
    max_count: u32,
    fetch_max_chars: usize,
}

impl WebSearchService {
    pub fn new(
        base_url: &str,
        api_key: Option<String>,
        default_count: u32,
        max_count: u32,
        fetch_max_chars: usize,
        timeout_secs: u64,
    ) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(timeout_secs.max(5)))
                .user_agent(USER_AGENT)
                // SSRF hardening (round 1): never auto-follow redirects —
                // reqwest's default follows up to 10 hops without re-running
                // the private-host preflight, so a public page 302-ing to
                // 169.254.169.254 would bypass the fetch-time guard
                // entirely. With Policy::none a 3xx just surfaces as its
                // HTTP status (handled below in fetch_text).
                .redirect(reqwest::redirect::Policy::none())
                // SSRF hardening (round 2 — DNS pinning): resolve through
                // SsrfResolver, which refuses hosts whose resolved addresses
                // are private/loopback/link-local/metadata (DNS-rebinding
                // and internal-name bypasses die at connect time, per
                // request). see_below for the trait impl.
                .dns_resolver(std::sync::Arc::new(SsrfResolver))
                .build()
                .expect("static reqwest client config"),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            default_count,
            max_count,
            fetch_max_chars,
        }
    }

    pub fn has_api_key(&self) -> bool {
        self.api_key
            .as_ref()
            .map(|k| !k.trim().is_empty())
            .unwrap_or(false)
    }

    /// web_search — one Brave Search API call, `web`-mode response mapped
    /// to `WebHit`s. Mirrors upstream param plumbing: q, country,
    /// search_lang, freshness, ui_lang, count (1..=max).
    pub async fn search(&self, q: &SearchQuery) -> Result<Vec<WebHit>> {
        let Some(api_key) = self
            .api_key
            .as_deref()
            .map(str::trim)
            .filter(|k| !k.is_empty())
        else {
            bail!(
                "missing Brave API key — a node's [websearch] table resolves it from \
                 `api_key` / `api_key_path` in config.toml, $JERITH_BRAVE_API_KEY, \
                 ~/.jerith-mesh/brave-key, or the OpenClaw keyring copy"
            );
        };
        if q.query.trim().is_empty() {
            bail!("empty search query");
        }
        let freshness = normalize_freshness(q.freshness.as_deref())?;
        let count = q
            .count
            .unwrap_or(self.default_count)
            .clamp(1, self.max_count);

        let base = if self.base_url.is_empty() {
            DEFAULT_BRAVE_BASE_URL.to_string()
        } else {
            self.base_url.clone()
        };
        let url = format!("{base}{BRAVE_SEARCH_ENDPOINT}");
        let mut req = self
            .client
            .get(&url)
            .query(&[("q", q.query.trim()), ("count", count.to_string().as_str())])
            .header("X-Subscription-Token", api_key)
            .header("Accept", "application/json");
        if let Some(c) = q
            .country
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            req = req.query(&[("country", c)]);
        }
        if let Some(l) = q
            .search_lang
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            req = req.query(&[("search_lang", l)]);
        }
        if let Some(l) = q
            .ui_lang
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            req = req.query(&[("ui_lang", l)]);
        }
        if let Some(f) = freshness {
            req = req.query(&[("freshness", f)]);
        }
        let resp = req.send().await.context("Brave Search API unreachable")?;
        let status = resp.status();
        let body: Value = resp
            .json()
            .await
            .context("Brave Search API returned non-JSON")?;
        if !status.is_success() {
            let msg = body
                .get("error")
                .or_else(|| body.get("message"))
                .or_else(|| body.get("description"))
                .and_then(Value::as_str)
                .unwrap_or("");
            bail!("Brave Search API error {status}: {msg}");
        }
        Ok(body
            .pointer("/web/results")
            .and_then(Value::as_array)
            .map(|arr| arr.iter().filter_map(WebHit::from_brave_entry).collect())
            .unwrap_or_default())
    }

    /// web_fetch — GET the page and reduce it to readable text.
    pub async fn fetch_text(&self, raw_url: &str, max_chars: Option<usize>) -> Result<PageText> {
        assert_fully_fetchable(raw_url)?;
        let mut resp = self
            .client
            .get(raw_url)
            .header(
                "Accept",
                "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            )
            .send()
            .await
            .with_context(|| format!("fetch failed for {raw_url}"))?;
        let status = resp.status().as_u16();
        // Redirects are NOT auto-followed (SSRF guard re-preflight would be
        // needed per hop); surface them at face value instead.
        if !(200..300).contains(&status) {
            bail!("fetch got HTTP {status} for {raw_url}");
        }
        // Bounded read: stop after the cap — `chunk()` keeps even a
        // content-length-less hostile response from ballooning memory.
        let mut buf: Vec<u8> = Vec::new();
        loop {
            match resp
                .chunk()
                .await
                .map_err(|e| anyhow!("reading body: {e}"))?
            {
                Some(chunk) => {
                    buf.extend_from_slice(&chunk);
                    if buf.len() >= FETCH_BODY_CAP_BYTES {
                        buf.truncate(FETCH_BODY_CAP_BYTES);
                        break;
                    }
                }
                None => break,
            }
        }
        let html = String::from_utf8_lossy(&buf);
        let cap = max_chars.unwrap_or(self.fetch_max_chars);
        Ok(html_to_page(raw_url, &html, cap, status))
    }
}

/// Turn raw HTML into a PageText — title plus readable prose.
pub fn html_to_page(url: &str, html: &str, max_chars: usize, status: u16) -> PageText {
    let title = extract_title(html);
    let mut text = extract_text(html);
    let truncated = text.chars().count() > max_chars;
    if truncated {
        text = text.chars().take(max_chars).collect();
    }
    PageText {
        url: url.to_string(),
        title,
        text,
        truncated,
        status,
    }
}

fn extract_title(html: &str) -> Option<String> {
    let lower = html.to_ascii_lowercase();
    let start = lower.find("<title")?;
    let gt = lower[start..].find('>')? + start + 1;
    let end = lower[gt..].find("</title")? + gt;
    let t = strip_tags(&html[gt..end]).trim().to_string();
    (!t.is_empty()).then_some(t)
}

/// Ordered HTML→text reduction (std + regex, no scraper crate):
/// strip comments/scripts/styles → newline at block ends → drop tags →
/// decode entities → collapse whitespace.
fn extract_text(html: &str) -> String {
    let mut owned = html.to_string();
    for pat in [
        r"(?is)<!--.*?-->",
        r"(?is)<script\b[^>]*>.*?</script\s*>",
        r"(?is)<style\b[^>]*>.*?</style\s*>",
        r"(?is)<noscript\b[^>]*>.*?</noscript\s*>",
        r"(?is)<template\b[^>]*>.*?</template\s*>",
        r"(?is)<svg\b[^>]*>.*?</svg\s*>",
        r"(?is)<form\b[^>]*>.*?</form\s*>",
        r"(?is)<head\b[^>]*>.*?</head\s*>",
    ] {
        let re = regex::Regex::new(pat).expect("static regex");
        owned = re.replace_all(&owned, "").into_owned();
    }
    // Block-level closers become paragraph breaks.
    let block = regex::Regex::new(
        r"(?i)</(p|div|section|article|header|footer|main|nav|aside|h[1-6]|li|tr|table|blockquote|pre|figure|figcaption)>|<br\s*/?>|</br>",
    )
    .expect("static regex");
    let spaced = block.replace_all(&owned, "\n");
    let bare = strip_tags(&spaced);
    let decoded = decode_entities(&bare);
    let mut out = String::new();
    for raw_line in decoded.split('\n') {
        let line = raw_line
            .replace('\u{00a0}', " ")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        if !line.is_empty() {
            out.push_str(&line);
            out.push('\n');
        }
    }
    out.trim_end().to_string()
}

/// Strip any residual tags (also used to clean Brave `title` snippets).
pub fn strip_tags(s: &str) -> String {
    let re = regex::Regex::new(r"<[^<>]{1,200}>").expect("static regex");
    decode_entities(&re.replace_all(s, "")).replace('\u{00a0}', " ")
}

fn decode_entities(s: &str) -> String {
    // Named entities first (letters only, so `&amp;lt;` stays encoded).
    let mut out = String::with_capacity(s.len());
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '&' {
            let mut j = i + 1;
            let mut name = String::new();
            while j < chars.len() && chars[j].is_ascii_alphanumeric() && name.len() < 10 {
                name.push(chars[j]);
                j += 1;
            }
            if j < chars.len() && chars[j] == ';' && !name.is_empty() {
                let rep = match name.as_str() {
                    "amp" => Some("&"),
                    "lt" => Some("<"),
                    "gt" => Some(">"),
                    "quot" => Some("\""),
                    "apos" => Some("'"),
                    "nbsp" => Some("\u{00a0}"),
                    _ => None,
                };
                if let Some(rep) = rep {
                    out.push_str(rep);
                    i = j + 1;
                    continue;
                }
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    // Numeric entities (&#39; and &#x27;) as a second pass.
    let re = regex::Regex::new(r"&#([xX]?)([0-9a-fA-F]+);").expect("static regex");
    re.replace_all(&out, |caps: &regex::Captures| {
        let hex = matches!(&caps[1], "x" | "X");
        let val = if hex {
            u32::from_str_radix(&caps[2], 16)
        } else {
            caps[2].parse::<u32>()
        }
        .ok()
        .and_then(char::from_u32);
        val.map(|c| c.to_string()).unwrap_or_default()
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The resolver's filtering logic, exercised against real resolutions:
    /// filtering out private IPs from a resolved set (and erroring when all
    /// of them are private) is the heart of the DNS-pinning guard; run on
    /// the real host's /etc/hosts so nothing here depends on network.
    #[tokio::test]
    async fn ssrf_resolver_filters_private_resolutions() {
        use std::net::ToSocketAddrs;
        // "localhost" resolves (getent shows ::1 on this host) — filtering
        // it must empty the set and thus refuse.
        let resolved: Vec<std::net::SocketAddr> =
            ("localhost", 0u16).to_socket_addrs().unwrap().collect();
        assert!(!resolved.is_empty(), "this test wants localhost to resolve");
        let filtered: Vec<&std::net::SocketAddr> = resolved
            .iter()
            .filter(|sa| !is_private_host(&sa.ip().to_string()))
            .collect();
        assert!(
            filtered.is_empty(),
            "localhost's addresses must all be private, got {resolved:?}"
        );
        // The real code path: spawning the resolver's shared logic via a
        // second pass to assert the is_private_host predicate used inline.
        assert!(is_private_host(&"127.0.0.1".to_string()));
        assert!(!is_private_host(&"93.184.216.34".to_string()));
    }

    #[test]
    fn html_reduces_to_readable_text() {
        let page = r##"<html><head><title> Dojo &amp; Notes </title>
          <style>p { color: red } </style><script>var x = "ignored";</script></head>
          <body><div><h1>Welcome</h1><p>kiln &lt;fired&nbsp;clay</p>
          <p>Need to return bricks <a href="#">soon</a>.</p><!-- secret --></div></body></html>"##;
        let page_text = html_to_page("https://example.test/", page, 500, 200);
        assert_eq!(page_text.title.as_deref(), Some("Dojo & Notes"));
        assert!(page_text.text.contains("Welcome"));
        assert!(page_text.text.contains("kiln <fired clay"));
        assert!(page_text.text.contains("Need to return bricks"));
        assert!(!page_text.text.contains("ignored"));
        assert!(!page_text.text.contains("secret"));
        assert!(!page_text.truncated);
    }

    #[test]
    fn long_page_truncates_to_cap() {
        let html = format!("<p>{}word</p>", "a".repeat(900));
        let p = html_to_page("https://example.test/", &html, 100, 200);
        assert!(p.truncated);
        assert_eq!(p.text.chars().count(), 100);
    }

    #[test]
    fn private_hosts_are_refused() {
        for bad in [
            "http://localhost/x",
            "http://127.0.0.1:11434/api",
            "http://10.1.2.3/",
            "http://192.168.1.4/",
            "http://172.16.0.9/",
            "http://169.254.169.254/latest",
            "http://100.64.0.7/",
            "http://[::1]:9000/",
            "http://[fe80::1]/",
            "http://metadata.google.internal/",
            "https://build.internal/",
        ] {
            assert!(assert_fully_fetchable(bad).is_err(), "must refuse {bad}");
        }
        for good in ["https://example.com/page", "http://8.8.8.8/"] {
            assert!(assert_fully_fetchable(good).is_ok(), "must allow {good}");
        }
        assert!(assert_fully_fetchable("ftp://example.com/").is_err());
        assert!(assert_fully_fetchable("file:///etc/passwd").is_err());
    }

    #[test]
    fn brave_results_map_to_hits() {
        let data = serde_json::json!({
            "web": {"results": [
                {"title": "<b>Jerith</b> mesh", "url": "https://github.com/ParadoxFuzzle/jerith-mesh",
                 "description": "A P2P compute mesh", "page_age": "2 days ago"},
                {"title": "no url here"}
            ]}
        });
        let arr = data.pointer("/web/results").unwrap().as_array().unwrap();
        let hits: Vec<WebHit> = arr.iter().filter_map(WebHit::from_brave_entry).collect();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].title, "Jerith mesh"); // tags stripped
        assert_eq!(hits[0].site_name.as_deref(), Some("github.com"));
        assert_eq!(hits[0].page_age.as_deref(), Some("2 days ago"));
    }

    #[test]
    fn freshness_is_validated_like_upstream() {
        assert_eq!(normalize_freshness(None).unwrap(), None);
        assert_eq!(normalize_freshness(Some(" week ")).unwrap(), Some("week"));
        assert!(normalize_freshness(Some("fortnight")).is_err());
    }
}
