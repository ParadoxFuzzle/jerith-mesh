//! Bootstrap discovery: N servers instead of one (Phase 1).
//!
//! The primary bootstrap (p2p.jerithai.com behind a Cloudflare Tunnel) is
//! a single point of failure for *discovery* — the node keeps running and
//! serving when it is unreachable (graceful degradation), but fresh peers
//! stop flowing in and new nodes cannot find us. This module
//!
//! - keeps the operator's server list (`[discovery] servers` in
//!   config.toml) NEXT TO the primary, first among equals;
//! - merges **learned** URLs that arrive inside peers' signed ads
//!   (`PeerAd.discovery_servers`) — gossip-based list maintenance, no
//!   coordinator: peers exchanging their lists keeps every node's list
//!   fresh without an operator;
//! - persists the learned portion to `<data>/discovery-servers.json` so a
//!   restart keeps yesterday's list;
//! - picks per-call order: last-good first (failover stickiness — a
//!   bootstrap that answered a moment ago is worth trying before an
//!   untested one), then primary, then the rest in stable order.
//!
//! Sanitizing is strict on purpose: URLs enter the operator's config, the
//! gossip ad, the persisted file and per-tick HTTP from this list. Only
//! `http(s)://` URLs with a host pass; `http://…` (not https) is allowed
//! for LAN bootstraps but never auto-learned from peers (a hostile ad
//! could otherwise point the fleet at its own capture proxy).

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::Path;
use std::path::PathBuf;

/// How many servers one node keeps in rotation (configured + learned).
/// Bootstraps are cheap and rare; more than this is a config smell.
const ROTATION_CAP: usize = 16;

/// How many server URLs fit inside one ad (gossip payload).
pub const AD_MAX_SERVERS: usize = 8;

/// Is this URL usable as a bootstrap? Hand-rolled (no `url` crate — the
/// repo's dependency discipline: detect_routable_host hand-rolls its UDP
/// trick the same way). Only what the server-join actually permits:
/// `http(s)://origin[/path]` — a host, no userinfo (a `user@host` trick
/// would repoint the DNS in the join), no query or fragment (the same
/// injection the join would splice), no whitespace.
pub fn valid_server_url(url: &str) -> bool {
    let Some((scheme, rest)) = url.split_once("://") else {
        return false;
    };
    if !matches!(scheme, "http" | "https") {
        return false;
    }
    let rest = rest.trim_end_matches('/');
    if rest.is_empty() || rest.contains([' ', '@', '?', '#', '\\']) {
        return false;
    }
    // Origin (optional :port) or origin/path: everything before the first '/'.
    let authority = rest.split('/').next().unwrap_or_default();
    let (host, port) = match authority.rsplit_once(':') {
        Some((h, p)) if !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()) => (h, true),
        Some(_) => return false, // a colon that is not a port
        None => (authority, false),
    };
    let _ = port;
    let bracketed = host.starts_with('[') && host.ends_with(']');
    if bracketed {
        // IPv6 literal: hex + colons between the brackets.
        let inner = &host[1..host.len() - 1];
        !inner.is_empty() && inner.chars().all(|c| c.is_ascii_hexdigit() || c == ':')
    } else {
        !host.is_empty()
            && host
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_'))
    }
}

/// The sanitized server URL set: trim, dedupe, drop empties and anything
/// unusable, respect the rotation cap. Pure — plumbed into config and ads.
pub fn sanitize(input: &[String], own: Option<&str>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for raw in input {
        let s = raw.trim().trim_end_matches('/');
        if s.is_empty()
            || !valid_server_url(s)
            || out.iter().any(|o| o.eq_ignore_ascii_case(s))
            || own.is_some_and(|o| s.eq_ignore_ascii_case(o.trim_end_matches('/')))
        {
            continue;
        }
        out.push(s.to_string());
        if out.len() >= ROTATION_CAP {
            break;
        }
    }
    out
}

/// What a node knows: the operator's list (principal), learned URLs
/// (gossip), the last-good server (failover stickiness) and where the
/// learned part persists. Configured entries keep their config order and
/// are never evicted by rotation caps; learned ones ride behind them.
#[derive(Debug, Default)]
pub struct DiscoveryServers {
    /// `[discovery] servers` — the operator's, in config order (already
    /// sanitized at load).
    pub configured: Vec<String>,
    /// Learned from peers' ads (gossip), sanitize(d) + capped.
    pub learned: Vec<String>,
    /// The last server that answered us this process (persisted too).
    pub last_good: Option<String>,
    path: Option<PathBuf>,
}

impl DiscoveryServers {
    /// Fresh state: the config's server list, an optional restored
    /// last-good, learning into `data/discovery-servers.json`.
    pub fn new(configured: Vec<String>, last_good: Option<String>, data_dir: &Path) -> Self {
        Self {
            configured: sanitize(&configured, None),
            last_good,
            path: Some(data_dir.join("discovery-servers.json")),
            learned: Vec::new(),
        }
    }

    /// Total rotation list: configured first (config order), then learned
    /// (most recently learned first is NOT attempted — stable order keeps
    /// peer lists deterministic), sanitized as a whole.
    pub fn rotation(&self, own: Option<&str>) -> Vec<String> {
        let mut all = self.configured.clone();
        all.extend(self.learned.iter().cloned());
        sanitize(&all, own)
    }

    /// Per-call order: last-good hoisted, rest stable. Pure.
    pub fn order(&self, own: Option<&str>) -> Vec<String> {
        let rotation = self.rotation(own);
        match self.last_good.as_deref() {
            Some(good) if rotation.iter().any(|s| s == good) => {
                let mut out = vec![good.to_string()];
                out.extend(rotation.into_iter().filter(|s| s != good));
                out
            }
            _ => rotation,
        }
    }

    /// Record that `url` answered (last-good bookkeeping). Returns true if
    /// the recorded value changed (persist signal).
    pub fn note_good(&mut self, url: &str) -> bool {
        let url = url.trim_end_matches('/');
        let changed = self.last_good.as_deref() != Some(url);
        self.last_good = Some(url.to_string());
        changed
    }

    /// Clear the last-good pin (no persist at this call site — the next
    /// re-pinning writes it back). Hostile-ad audit (F2) support: an
    /// answered-but-empty peer list does not earn stickiness.
    pub fn clear_last_good(&mut self) {
        self.last_good = None;
    }

    /// Merge a batch of learned URLs (from one ad batch). Returns true if
    /// the learned set grew (persist signal). Gossip adds are NEVER
    /// auto-promoted to plain http — see the module docs. Full learned
    /// slots evict FIFO (oldest first, last-good exempt) instead of
    /// wedging shut under a hostile ad flood (audit F1).
    pub fn merge_learned(&mut self, incoming: &[String]) -> bool {
        let mut grew = false;
        for url in sanitize(incoming, None) {
            if url == "http://" || url.starts_with("http://") {
                continue; // plain-HTTP bootstraps: config-only, never learned
            }
            if self.learned.iter().any(|s| s == &url)
                || self.configured.iter().any(|s| s == &url)
            {
                continue;
            }
            if self.configured.len() + self.learned.len() >= ROTATION_CAP {
                // Hostile-ad audit (F1): without eviction, hostile signed ads
                // could PERMANENTLY fill every learned slot — the node stops
                // learning real bootstraps and keeps dialing the attacker's
                // list forever. Fair FIFO: the oldest learned entry makes
                // room for a fresh one; a pinned last-good is never evicted
                // (stickiness beats pedantry), configured entries are
                // principal and out of reach entirely.
                let evict_idx = self
                    .learned
                    .iter()
                    .position(|s| self.last_good.as_deref() != Some(s.as_str()));
                match evict_idx {
                    Some(i) => {
                        self.learned.remove(i);
                    }
                    None => break, // cap genuinely full of pinned entries
                }
            }
            self.learned.push(url);
            grew = true;
        }
        grew
    }

    /// Load the learned set + last-good from `path` (missing file → empty).
    pub fn load(path: &Path) -> Self {
        let Ok(text) = std::fs::read_to_string(path) else {
            return Self {
                path: Some(path.to_path_buf()),
                ..Default::default()
            };
        };
        let Ok(v) = serde_json::from_str::<Value>(&text) else {
            return Self {
                path: Some(path.to_path_buf()),
                ..Default::default()
            };
        };
        let learned: Vec<String> = v
            .get("learned")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        let last_good = v
            .get("last_good")
            .and_then(Value::as_str)
            .filter(|s| valid_server_url(s))
            .map(str::to_string);
        Self {
            configured: Vec::new(),
            learned: sanitize(&learned, None)
                .into_iter()
                .filter(|s| !s.starts_with("http://"))
                .collect(),
            last_good,
            path: Some(path.to_path_buf()),
        }
    }

    /// Persist learned + last-good (a compact, human-inspectable file).
    pub fn save(&self) -> Result<()> {
        let Some(path) = &self.path else { return Ok(()) };
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p)?;
        }
        let json = json!({
            "learned": self.learned,
            "last_good": self.last_good,
        });
        serde_json::to_writer_pretty(
            &mut std::io::BufWriter::new(
                std::fs::File::create(path)
                    .with_context(|| format!("creating {}", path.display()))?,
            ),
            &json,
        )?;
        Ok(())
    }

    /// The gossip payload for OUR ad: the rotation list, capped, https or
    /// any http(s) we actually use (the ad is signed; a LAN http URL in
    /// it is the operator's own config speaking, not a hostile learn).
    pub fn ad_servers(&self, own: Option<&str>) -> Vec<String> {
        self.order(own)
            .into_iter()
            .take(AD_MAX_SERVERS)
            .collect()
    }
}

/// Per-call failover (Phase 1): walk the rotation order, first success
/// wins. Returns (value, server_that_answered) so the caller can record
/// last-good. Every failure is remembered so the final error names the
/// last server tried — a dead primary reads as a dead primary, not as
/// "degraded mysteriously".
pub async fn try_each<T, Fut, F>(order: &[String], mut call: F) -> anyhow::Result<(T, String)>
where
    F: FnMut(&str) -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<T>>,
{
    let mut last: Option<(anyhow::Error, String)> = None;
    for url in order {
        match call(url).await {
            Ok(value) => return Ok((value, url.clone())),
            Err(e) => last = Some((e, url.clone())),
        }
    }
    Err(match last {
        Some((e, url)) => e.context(format!(
            "all bootstrap servers failed (tried {order:?}), last: {url}"
        )),
        None => anyhow::anyhow!("no bootstrap servers configured"),
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(default)]
pub struct DiscoveryConfig {
    /// Additional independent bootstrap nodes, primary first
    /// (`[discovery] servers = [...]`). Empty → the primary URL alone.
    pub servers: Vec<String>,
}

impl DiscoveryConfig {
    pub fn validate(&self) -> Result<()> {
        for url in &self.servers {
            anyhow::ensure!(
                !url.trim().is_empty(),
                "[discovery] servers: blank entries are not bootstrap URLs"
            );
            anyhow::ensure!(
                valid_server_url(url.trim().trim_end_matches('/')),
                "[discovery] servers: {url:?} is not a usable http(s) origin"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "jerith-discovery-{tag}-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn valid_server_url_accepts_origins_only() {
        assert!(valid_server_url("https://p2p.jerithai.com"));
        assert!(valid_server_url("http://localhost:8901"));
        assert!(valid_server_url("https://bootstrap.example.org:443/"));
        assert!(valid_server_url("http://[::1]:8901"));
        // NOT origins: wrong scheme, missing host, and the injection shapes
        // the `{url}/v1/peers…` join would splice into the path.
        assert!(!valid_server_url("ftp://x"));
        assert!(!valid_server_url("https://"));
        assert!(!valid_server_url("http://host?next=/evil"));
        assert!(!valid_server_url("https://host#frag"));
        assert!(!valid_server_url("https://evil.com@real/"));
        assert!(!valid_server_url("https:// host"));
        assert!(!valid_server_url("p2p.jerithai.com"));
        assert!(!valid_server_url(""));
    }

    #[test]
    fn sanitize_trims_dedupes_caps_and_drops_own() {
        let list = sanitize(
            &[
                " https://a.example.org/ ".into(),
                "".into(),
                "ftp://b".into(),
                "https://a.example.org".into(), // same origin after trim
                "https://c.example.org".into(),
            ],
            Some("https://c.example.org"),
        );
        assert_eq!(list, vec!["https://a.example.org".to_string()]);

        // Rotation cap holds even under a flood.
        let many: Vec<String> = (0..30)
            .map(|i| format!("https://s{i}.example.org"))
            .collect();
        assert_eq!(sanitize(&many, None).len(), ROTATION_CAP);
    }

    /// Audit F1: a full learned list evicts FIFO so hostile ads cannot
    /// wedge learning shut; the pinned last-good is exempt from eviction.
    #[test]
    fn learned_slots_evict_fifo_and_never_evict_last_good() {
        let mut d = DiscoveryServers {
            configured: sanitize(&["https://primary.example.org".into()], None),
            ..DiscoveryServers::default()
        };

        // Fill every non-configured slot (cap 16) flush: the flood is
        // one short of the cap so no eviction happens during this merge.
        let flood: Vec<String> = (0..ROTATION_CAP - d.configured.len())
            .map(|i| format!("https://hostile-{i}.example.org"))
            .collect();
        assert!(d.merge_learned(&flood));
        assert_eq!(
            d.learned.len(),
            ROTATION_CAP - d.configured.len(),
            "slots stop at the rotation cap"
        );
        assert_eq!(d.learned[0], "https://hostile-0.example.org");

        // A fresh legitimate bootstrap still lands (FIFO eviction).
        assert!(d.merge_learned(&["https://legit.example.org".into()]));
        assert_eq!(d.learned.last().unwrap(), "https://legit.example.org");
        assert!(!d.learned.iter().any(|s| s == "https://hostile-0.example.org"));
        assert_eq!(d.learned.len(), ROTATION_CAP - d.configured.len());

        // Pinned last-good survives even as the rotation turns over.
        d.note_good("https://hostile-3.example.org");
        assert!(d.merge_learned(&["https://legit-2.example.org".into()]));
        assert!(
            d.learned.iter().any(|s| s == "https://hostile-3.example.org"),
            "pinned entry is never evicted"
        );
    }

    /// Audit F2 support: clearing the pin re-diversifies the order.
    #[test]
    fn clear_last_good_rediversifies_order() {
        let mut d = DiscoveryServers {
            configured: sanitize(
                &["https://primary.example.org".into(), "https://backup.example.org".into()],
                None,
            ),
            ..DiscoveryServers::default()
        };
        d.note_good("https://backup.example.org");
        assert_eq!(d.order(None)[0], "https://backup.example.org");
        d.clear_last_good();
        assert_eq!(d.order(None)[0], "https://primary.example.org");
    }

    #[test]
    fn merge_learned_grows_and_never_learns_http() {
        let mut d = DiscoveryServers {
            configured: sanitize(&["https://primary.example.org".into()], None),
            ..DiscoveryServers::default()
        };

        // https learns; plain http NEVER learns from gossip (config-only —
        // a hostile ad must not repoint the fleet at a capture proxy);
        // already-configured and duplicate entries leave the set alone.
        assert!(
            d.merge_learned(&[
                "https://learned-1.example.org".into(),
                "http://local-capture.example.org".into(),
                "https://primary.example.org".into(),
                "https://learned-2.example.org".into(),
            ])
        );
        assert_eq!(
            d.learned,
            vec![
                "https://learned-1.example.org".to_string(),
                "https://learned-2.example.org".to_string(),
            ]
        );

        // Re-learning the same URL is a no-op (no persist churn, ever).
        assert!(!d.merge_learned(&["https://learned-1.example.org".into()]));
    }

    #[test]
    fn order_hoists_last_good_keeps_rest_stable() {
        let mut d = DiscoveryServers {
            configured: sanitize(
                &[
                    "https://primary.example.org".into(),
                    "https://backup.example.org".into(),
                ],
                None,
            ),
            ..DiscoveryServers::default()
        };
        // No last-good yet: config order, primary first.
        assert_eq!(
            d.order(None),
            vec![
                "https://primary.example.org".to_string(),
                "https://backup.example.org".to_string(),
            ]
        );
        // The server that just answered goes first next call (failover
        // stickiness), the rest in stable order behind it.
        d.note_good("https://backup.example.org");
        assert_eq!(
            d.order(None),
            vec![
                "https://backup.example.org".to_string(),
                "https://primary.example.org".to_string(),
            ]
        );
    }

    #[test]
    fn persistence_roundtrip_keeps_learned_and_last_good() {
        let dir = temp_dir("persist");
        let path = dir.join("discovery-servers.json");

        let mut d = DiscoveryServers::new(
            vec!["https://primary.example.org".into()],
            None,
            &dir,
        );
        assert!(d.merge_learned(&["https://learned.example.org".into()]));
        d.note_good("https://primary.example.org");
        d.save().unwrap();

        let mut restored = DiscoveryServers::load(&path);
        assert_eq!(
            restored.learned,
            vec!["https://learned.example.org".to_string()]
        );
        assert_eq!(
            restored.last_good.as_deref(),
            Some("https://primary.example.org")
        );
        // http stays unlearnable across the persistence boundary too.
        assert!(!restored.merge_learned(&["http://x.example.org".into()]));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn discovery_config_validation_rejects_malformed_urls() {
        assert!(DiscoveryConfig {
            servers: vec!["https://b.example.org".into()],
        }
        .validate()
        .is_ok());
        assert!(DiscoveryConfig {
            servers: vec!["   ".into()],
        }
        .validate()
        .is_err());
        assert!(DiscoveryConfig {
            servers: vec!["not a url".into()],
        }
        .validate()
        .is_err());
    }

    /// Per-call failover: a dead first server is transparent to the
    /// caller; the SECOND server's value AND its address come back
    /// together so the loop records last-good. All-dead summarizes
    /// honestly rather than inventing an empty success.
    #[tokio::test]
    async fn failover_walks_the_order_and_reports_the_used_server() {
        let order = vec![
            "https://dead.example.org".to_string(),
            "https://alive.example.org".to_string(),
        ];
        let (value, used) = try_each(&order, |url| {
            let outcome: Result<String> = if url.starts_with("https://dead") {
                Err(anyhow::anyhow!("connection refused (test)"))
            } else {
                Ok(format!("peers-from-{url}"))
            };
            async move { outcome }
        })
        .await
        .unwrap();
        assert_eq!(value, "peers-from-https://alive.example.org");
        assert_eq!(used, "https://alive.example.org");

        // Everything dead: the error names the rotation, not "unknown".
        let err = format!(
            "{:#}",
            try_each::<String, _, _>(&order, |_url| async {
                Err::<String, anyhow::Error>(anyhow::anyhow!("down"))
            })
            .await
            .unwrap_err()
        );
        assert!(err.contains("all bootstrap servers failed"), "got: {err}");
    }
}
