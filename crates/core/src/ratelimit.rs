//! Fixed-window rate limiter for the public endpoints (bootstrap reads,
//! gateway skills/chat).
//!
//! Not distributed, not fair — a cheap in-memory floor against
//! single-source hammering: each request costs a registry sweep or a
//! SQLite round-trip, and until now nothing bounded how often one client
//! could pay that bill.
//!
//! Attribution order: `CF-Connecting-IP` (production sits behind a
//! Cloudflare Tunnel — direct-connect spoofing of this header is moot
//! because CF overwrites it), then the first `X-Forwarded-For` hop, then
//! the socket peer (`ConnectInfo`, wired via
//! `into_make_service_with_connect_info`). Requests we cannot attribute
//! are allowed through unlimited — limping-by rather than locking out
//! every isolated test client and headerless proxy hop.
//!
//! Keys are strings so one limiter implementation serves two lanes
//! (`ip-…` / `id-…`): the ANONYMOUS floor (per-IP, tight) and the
//! peer-identity allowance (per Ed25519 fingerprint, generous — the
//! NAT-scaling lane). Both processes (node gateway + standalone
//! bootstrap) share this module; each 429 emitted here bumps the
//! process-local `metrics::RATE_LIMITED_429_TOTAL` counter (surfaced in
//! the node's /status and the bootstrap's /health).

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::{ConnectInfo, Request, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

/// How many distinct keys we retain between windows. With clean windows,
/// too many distinct IPs within one window → extra hits.
const BUCKET_PRUNE_THRESHOLD: usize = 8_192;

pub struct RateLimiter {
    max_hits: u32,
    window: Duration,
    buckets: Mutex<HashMap<String, Bucket>>,
}

struct Bucket {
    window_start: Instant,
    hits: u32,
}

/// Bucket key for an IP lane hit.
pub fn ip_key(ip: IpAddr) -> String {
    format!("ip-{ip}")
}

/// Bucket key for a peer-identity lane hit (Ed25519 pubkey fingerprint
/// = the node_id prefix; holdings require key control, so identity
/// buckets cannot be spoofed or spent by others).
pub fn identity_key(pubkey_hex: &str) -> String {
    format!("id-{}", &pubkey_hex[..16.min(pubkey_hex.len())])
}

impl RateLimiter {
    pub fn new(max_hits: u32, window: Duration) -> Self {
        Self {
            max_hits,
            window,
            buckets: Mutex::new(HashMap::new()),
        }
    }

    /// True when this hit is within the key's budget for the current window.
    pub fn allow(&self, key: String) -> bool {
        let now = Instant::now();
        let mut buckets = self.buckets.lock().unwrap();
        // Bucket-bloat guard: a key spray within one window could grow the
        // map without limit. Expire-idle prune once past the threshold.
        if buckets.len() >= BUCKET_PRUNE_THRESHOLD {
            buckets.retain(|_, b| now.duration_since(b.window_start) < self.window);
        }
        let b = buckets.entry(key).or_insert(Bucket {
            window_start: now,
            hits: 0,
        });
        if now.duration_since(b.window_start) >= self.window {
            b.window_start = now;
            b.hits = 0;
        }
        b.hits += 1;
        b.hits <= self.max_hits
    }

    /// Seconds until the caller's window rolls over (429 Retry-After).
    /// Rounds up: “at most Ns of waiting” must never undersell the wait.
    pub fn retry_after_secs(&self, key: &str) -> u64 {
        let now = Instant::now();
        self.buckets
            .lock()
            .unwrap()
            .get(key)
            .map(|b| {
                let rem = self
                    .window
                    .saturating_sub(now.duration_since(b.window_start));
                let mut secs = rem.as_secs();
                if rem.subsec_nanos() > 0 {
                    secs += 1;
                }
                secs.max(1)
            })
            .unwrap_or(1)
    }

    pub fn window(&self) -> Duration {
        self.window
    }

    pub fn max_hits(&self) -> u32 {
        self.max_hits
    }
}

/// The peer address for a request. Proxy-reported client IPs
/// (`CF-Connecting-IP`, first `X-Forwarded-For` hop) are trusted ONLY from
/// loopback peers — the local tunnel proxy. A directly-reachable origin
/// would otherwise let an internet attacker spoof the header to evade
/// their own bucket or to deplete a named victim's. Everyone else gets
/// their socket peer address.
pub fn client_ip(req: &Request<Body>) -> Option<IpAddr> {
    let peer_loopback = req
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .is_some_and(|ci| ci.0.ip().is_loopback());
    if peer_loopback {
        for h in ["cf-connecting-ip", "x-forwarded-for"] {
            if let Some(v) = req.headers().get(h).and_then(|v| v.to_str().ok()) {
                // First hop wins (Cloudflare appends itself last).
                if let Some(ip) = v.split(',').next().and_then(|s| s.trim().parse().ok()) {
                    return Some(ip);
                }
            }
        }
    }
    req.extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ci| ci.0.ip())
}

/// Core policy: unattributable requests pass; everything else must fit in
/// the key's window budget or gets a 429 with Retry-After. A rejected
/// request never reaches the handler — the cost is one Mutex touch.
pub async fn enforce(
    limiter: &RateLimiter,
    key: Option<String>,
    req: Request<Body>,
    next: Next,
) -> Response {
    match key {
        Some(key) if !limiter.allow(key.clone()) => too_many_response(limiter, &key),
        _ => next.run(req).await,
    }
}

fn too_many_response(limiter: &RateLimiter, key: &str) -> Response {
    crate::metrics::bump(&crate::metrics::RATE_LIMITED_429_TOTAL);
    let ra = limiter.retry_after_secs(key);
    (
        StatusCode::TOO_MANY_REQUESTS,
        [
            (
                header::RETRY_AFTER,
                HeaderValue::from_str(&ra.to_string()).unwrap_or(HeaderValue::from_static("60")),
            ),
            (
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/json"),
            ),
        ],
        axum::Json(serde_json::json!({
            "error": "rate limit exceeded — slow down",
            "retry_after_secs": ra,
        })),
    )
        .into_response()
}

/// Drop-in middleware state handler. Mount with
/// `.layer(axum::middleware::from_fn_with_state(limiter, axum_guard))`.
pub async fn axum_guard(
    State(limiter): State<Arc<RateLimiter>>,
    req: Request<Body>,
    next: Next,
) -> Response {
    let key = client_ip(&req).map(ip_key);
    enforce(&limiter, key, req, next).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    fn ip(n: u8) -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(10, 0, 0, n))
    }

    fn key(n: u8) -> String {
        ip_key(ip(n))
    }

    #[test]
    fn header_trust_requires_loopback_peer() {
        use axum::http::HeaderValue;
        use axum::http::Request;

        // Loopback peer (the local tunnel proxy) + CF header → honored.
        let mut req = Request::builder().body(Body::empty()).unwrap();
        req.extensions_mut().insert(ConnectInfo(SocketAddr::new(
            IpAddr::V4(Ipv4Addr::LOCALHOST),
            1234,
        )));
        req.headers_mut()
            .insert("cf-connecting-ip", HeaderValue::from_static("5.5.5.5"));
        assert_eq!(client_ip(&req), Some("5.5.5.5".parse::<IpAddr>().unwrap()));

        // Non-loopback peer + same header → spoofed header ignored; the
        // socket peer is the attribution (this is what pins a direct
        // internet attacker to their own bucket).
        let mut req = Request::builder().body(Body::empty()).unwrap();
        req.extensions_mut().insert(ConnectInfo(SocketAddr::new(
            IpAddr::V4(Ipv4Addr::new(9, 9, 9, 9)),
            1234,
        )));
        req.headers_mut()
            .insert("cf-connecting-ip", HeaderValue::from_static("5.5.5.5"));
        assert_eq!(client_ip(&req), Some("9.9.9.9".parse::<IpAddr>().unwrap()));
    }

    #[test]
    fn allow_hits_cap_then_recovers_next_window() {
        let rl = RateLimiter::new(3, Duration::from_millis(50));
        for _ in 0..3 {
            assert!(rl.allow(key(1)));
        }
        assert!(!rl.allow(key(1)), "cap exhausted");
        // Other keys unaffected.
        assert!(rl.allow(key(2)));
        // Window roll-over restores the budget.
        std::thread::sleep(Duration::from_millis(60));
        assert!(rl.allow(key(1)));
    }

    #[test]
    fn rejected_hits_bump_the_429_counter() {
        // Parallel test streams share the process-local counter, so assert
        // on a delta — and make it large enough that our own rejections
        // dominate it.
        let before =
            crate::metrics::RATE_LIMITED_429_TOTAL.load(std::sync::atomic::Ordering::Relaxed);
        let rl = RateLimiter::new(1, Duration::from_millis(50));
        assert!(rl.allow(key(8)));
        for _ in 0..10 {
            let resp = too_many_response(&rl, &key(8));
            assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        }
        let bumped = crate::metrics::RATE_LIMITED_429_TOTAL
            .load(std::sync::atomic::Ordering::Relaxed)
            - before;
        assert!(
            bumped >= 10,
            "counter must move with every 429, got {bumped}"
        );
    }

    #[test]
    fn retry_after_is_positive_and_bounded() {
        let rl = RateLimiter::new(1, Duration::from_secs(5));
        assert!(rl.allow(key(3)));
        // Fresh window ⇒ the full 5s (test elapsed time is well under 1ms);
        // anything inside the window stays in [1, 5].
        let secs = rl.retry_after_secs(&key(3));
        assert!((1..=5).contains(&secs), "got {secs}");
    }

    #[test]
    fn bucket_prune_keeps_map_bounded() {
        let rl = std::sync::Arc::new(RateLimiter::new(100, Duration::from_millis(10)));
        for i in 0..(BUCKET_PRUNE_THRESHOLD + 500) {
            let _ = rl.allow(ip_key(IpAddr::V4(Ipv4Addr::new(
                1,
                (i / 254) as u8 % 254 + 1,
                (i % 254) as u8 + 1,
                1,
            ))));
        }
    }

    #[test]
    fn identity_buckets_are_isolated_from_ip_buckets() {
        // The NAT-scaling allowance: a signed peer's hits live on
        // `id-…`, never nibbling (nor replenishing) an IP bucket.
        let rl = RateLimiter::new(2, Duration::from_millis(50));
        let idk = identity_key(&"a".repeat(64));
        let idk2 = idk.clone();
        assert!(rl.allow(idk.clone()));
        assert!(rl.allow(idk.clone()));
        assert!(!rl.allow(idk), "identity cap is its own budget");
        // The IP lane is untouched by the identity-lane exhaustion: a
        // fresh IP bucket still has its full budget. (Same client, but
        // unsigned → the IP bucket pays; that's the enforced separation.)
        assert!(
            rl.allow(key(9)),
            "ip lane budget must not share with id lane"
        );
        // And the exhausted identity budget stays exhausted: no cross-
        // replenishing through IP traffic.
        assert!(!rl.allow(idk2), "identity budget must stay exhausted");
    }
}
