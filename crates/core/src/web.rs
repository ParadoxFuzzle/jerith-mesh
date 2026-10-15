//! Minimal web UI: single-file status page served at the gateway port.
//!
//! Phase 0 scope: live status cards (model, contribution %, peers, meters).
//! Phase 2/3 scope: the dashboard adds earnings (mining + chain view),
//! expert hit-rate (LRU cache + measured activations) and per-model
//! latency views — all fed from the node's own /status and /v1/metrics,
//! same-origin only.
//!
//! Hardening: the page is served with a strict security-header set —
//! `Cache-Control: no-store`, `X-Content-Type-Options: nosniff`,
//! `Referrer-Policy: no-referrer`, and a CSP that pins the inline
//! `<script>`/`<style>` blocks by their sha256 digests (no
//! `'unsafe-inline'`), locks `frame-ancestors` to 'none' (clickjacking),
//! and drops `base-uri`/`form-action`/`object-src` to nothing. The
//! digests are computed from `PAGE` itself at first request, so page
//! edits re-pin automatically instead of roting into a mismatch.

use axum::Router;
use axum::extract::State;
use axum::http::header;
use axum::response::IntoResponse;
use axum::routing::get;
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

use crate::ping::MeshState;

pub fn router(state: std::sync::Arc<MeshState>) -> Router {
    Router::new().route("/", get(index)).with_state(state)
}

/// The inline block of PAGE between `<tag>` and `</tag>` (tag without
/// attributes; must exist or the CSP builder falls back).
fn inline_block(page: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = page.find(&open)? + open.len();
    let end = start + page.get(start..)?.find(&close)?;
    Some(page.get(start..end)?.to_string())
}

/// Standard base64 (RFC 4648, with `=` padding) — what CSP hash-sources
/// expect. Kept local to avoid a `base64` dependency for ~20 lines.
fn b64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(T[(n >> 18 & 63) as usize] as char);
        out.push(T[(n >> 12 & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            T[(n >> 6 & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            T[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

/// `"sha256-<digest>"` for one inline block.
fn hash_source(block: &str) -> String {
    let digest = Sha256::digest(block.as_bytes());
    format!("sha256-{}", b64(&digest))
}

/// The full CSP built from PAGE's own inline blocks. Falls back to
/// `'unsafe-inline'` per source if the markers ever move (defense must
/// survive edits that reshape the HTML).
fn build_csp(page: &str) -> String {
    let script = match inline_block(page, "script") {
        Some(s) => hash_source(&s),
        None => "'unsafe-inline'".to_string(),
    };
    let style = match inline_block(page, "style") {
        Some(s) => hash_source(&s),
        None => "'unsafe-inline'".to_string(),
    };
    format!(
        "default-src 'none'; script-src {script}; style-src {style}; \
         connect-src 'self'; img-src 'self'; object-src 'none'; \
         base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    )
}

static CSP: OnceLock<String> = OnceLock::new();

fn csp_header_value() -> &'static str {
    CSP.get_or_init(|| build_csp(PAGE))
}

async fn index(State(_gw): State<std::sync::Arc<MeshState>>) -> impl IntoResponse {
    (
        [
            (header::CONTENT_TYPE, "text/html; charset=utf-8"),
            // A status page must never sit in any cache (heuristic
            // caching included), must never be MIME-sniffed, and must
            // not leak referrers. The CSP carries no remote content at
            // all: everything the page loads is same-origin static.
            (header::CACHE_CONTROL, "no-store"),
            (header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
            (header::REFERRER_POLICY, "no-referrer"),
            (header::CONTENT_SECURITY_POLICY, csp_header_value()),
        ],
        PAGE,
    )
}

const PAGE: &str = r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>jerith-mesh node</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: ui-monospace, monospace; background: #0d1117; color: #c9d1d9; margin: 0; padding: 2rem; }
  h1 { color: #58a6ff; font-weight: 500; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; max-width: 900px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1rem; }
  .card h2 { margin: 0 0 .5rem; font-size: .8rem; text-transform: uppercase; color: #8b949e; letter-spacing: .08em; }
  .card p { margin: 0; font-size: 1.6rem; }
  .fine { color: #8b949e; font-size: .8rem; margin-top: 6px; }
  footer { margin-top: 2rem; color: #8b949e; font-size: .8rem; }
</style>
</head>
<body>
<h1>🦜 jerith-mesh node</h1>
<div class="grid">
  <div class="card"><h2>Model</h2><p id="model">—</p><div class="fine">quantization shown under the name</div></div>
  <div class="card"><h2>Contribution</h2><p id="contrib">—%</p><div class="fine">minimum 25% required</div></div>
  <div class="card"><h2>Peers alive</h2><p id="peers">—</p><div class="fine">same-model nodes only</div></div>
  <div class="card"><h2>Role</h2><p id="role">—</p><div class="fine">contributor / requester / hybrid</div></div>
  <div class="card"><h2>Tokens served</h2><p id="tokens">—</p><div class="fine">for the mesh, this window</div></div>
  <div class="card"><h2>Expert activations</h2><p id="acts">—</p><div class="fine">loads served to peers</div></div>
  <div class="card"><h2>Phase 4 verification</h2><p id="ccommit">—</p><div class="fine" id="ccommitfine">commitment checks</div></div>
  <div class="card"><h2>Jerith Guard</h2><p id="guard">—</p><div class="fine" id="guardfine">outbound screens</div></div>
  <div class="card"><h2>Earnings (mined)</h2><p id="earn">—</p><div class="fine" id="earnfine">JER mining wins</div></div>
  <div class="card"><h2>Expert hit-rate</h2><p id="hit">—</p><div class="fine" id="hitfine">LRU cache + demand</div></div>
  <div class="card"><h2>Latency</h2><p id="latency">—</p><div class="fine" id="latencyfine">busiest model, this window</div></div>
</div>
<footer>
  jerith-mesh · a P2P compute mesh for MoE models ·
  <a href="/status" style="color:#58a6ff">raw JSON</a> ·
  <a href="/health" style="color:#58a6ff">health</a> ·
  <a href="/v1/models" style="color:#58a6ff">OpenAI API</a>
</footer>
<script>
async function refresh() {
  try {
    const [r, mr] = await Promise.all([
      fetch('/status'),
      fetch('/v1/metrics').catch(() => null),
    ]);
    const j = await r.json();
    const mj = mr && mr.ok ? await mr.json() : null;
    document.getElementById('model').textContent = j.model || 'not set';
    document.getElementById('contrib').textContent = (j.contribution_percent ?? '—') + '%';
    document.getElementById('peers').textContent = j.peers_alive ?? '—';
    document.getElementById('role').textContent = j.role || '—';
    document.getElementById('tokens').textContent = j.meters?.tokens_served ?? '—';
    document.getElementById('acts').textContent = j.meters?.expert_activations ?? '—';
    // Phase 4 verification widget.
    const v = j.verify ?? {};
    document.getElementById('ccommit').textContent = v.commitment_checks ?? '—';
    document.getElementById('ccommitfine').textContent =
      `commitment checks · failures ${v.commitment_failures ?? 0}` +
      ` · replays ${v.replay_matches ?? 0}/${v.replays_dispatched ?? 0}` +
      ` · fraud reports ${v.fraud_reports_flushed ?? 0}`;
    // Jerith Guard widget (null = disabled in config).
    const g = j.guard;
    document.getElementById('guard').textContent = g ? (g.screens ?? '—') : 'off';
    document.getElementById('guardfine').textContent = g
      ? `outbound screens · blocked ${g.requests_blocked ?? 0} · redacted ${g.values_redacted ?? 0} · logged ${g.detections_logged ?? 0}`
      : 'guard disabled in config.toml ([guard] enabled = false)';
    // Phase 2 modernity: expert hit-rate (LRU cache hit ratio + measured
    // demand from /status.experts; "off" when no expert store lives here).
    const ex = j.experts;
    if (ex && (ex.hits ?? 0) + (ex.misses ?? 0) > 0) {
      const hr = Math.round(100 * ex.hits / (ex.hits + ex.misses));
      document.getElementById('hit').textContent = hr + '%';
      document.getElementById('hitfine').textContent =
        `cache hits/misses ${ex.hits}/${ex.misses} · ${ex.activations_total ?? 0} activations on ${ex.activations_distinct ?? 0} experts · hot ${Array.isArray(ex.hot) ? ex.hot.length : '?'} / cold ${ex.cold ?? '?'}`;
    } else if (ex) {
      document.getElementById('hit').textContent = '—';
      document.getElementById('hitfine').textContent =
        `no loads yet · ${ex.activations_total ?? 0} activations on ${ex.activations_distinct ?? 0} experts`;
    } else {
      document.getElementById('hit').textContent = 'off';
      document.getElementById('hitfine').textContent = 'no expert store on this node';
    }
    // Phase 2/3 dashboard view: latency from /v1/metrics (busiest model
    // headline; per-model detail rides the fine line pattern).
    const models = Object.entries((mj && mj.models) || {});
    if (models.length) {
      const busiest = models.reduce((a, b) => ((b[1].requests ?? 0) >= (a[1].requests ?? 0) ? b : a));
      const st = busiest[1];
      document.getElementById('latency').textContent =
        st.latency_ms_mean != null && st.requests ? st.latency_ms_mean + ' ms' : '—';
      document.getElementById('latencyfine').textContent =
        `${busiest[0]} · p-max ${st.latency_ms_max ?? '—'} ms · ok ${st.ok_percent ?? '—'}% · mesh ${st.mesh_served ?? 0}/${st.requests ?? 0}`;
    } else {
      document.getElementById('latency').textContent = '—';
      document.getElementById('latencyfine').textContent =
        'no completions in the window yet';
    }
    // Earnings view: JER mining wins for this node + chain reachability.
    document.getElementById('earn').textContent = String(j.meters?.mines_won ?? 0);
    const ch = j.chain;
    document.getElementById('earnfine').textContent =
      `mining attempts ${j.meters?.mines_attempted ?? 0}` +
      (ch && ch.reachable
        ? ` · chain height ${ch.height ?? '—'} · circulating ${(ch.circulating_jer ?? '—')} JER`
        : ' · chain view unreachable');
  } catch (e) { /* node stopped */ }
}
refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use axum_test::TestServer;

    fn spawn() -> TestServer {
        let mut cfg = crate::config::NodeConfig::default();
        cfg.ensure_identity().expect("test identity");
        let (state, _rx) = MeshState::new(cfg).expect("test mesh state");
        TestServer::new(router(state).into_make_service()).expect("test server")
    }

    #[test]
    fn b64_matches_rfc4648_reference_vectors() {
        let d = |s: &str| b64(&Sha256::digest(s.as_bytes()));
        // sha256("abc") / sha256("hello world") in standard base64,
        // cross-checked against an independent Python implementation.
        assert_eq!(d("abc"), "ungWv48Bz+pBQUDeXa4iI7ADYaOWF3qctBD/YfIAFa0=");
        assert_eq!(
            d("hello world"),
            "uU0nuZNNPgilLlLX2n2r+sSE7+N6U4DukIj3rOLvzek="
        );
    }

    #[test]
    fn csp_pins_the_actual_inline_blocks() {
        let csp = build_csp(PAGE);
        assert!(
            !csp.contains("'unsafe-inline'"),
            "page ships marker blocks — CSP must hash-pin them, got: {csp}"
        );
        let script = inline_block(PAGE, "script").expect("script marker");
        assert!(
            csp.contains(&hash_source(&script)),
            "CSP must pin the served script digest: {csp}"
        );
        let style = inline_block(PAGE, "style").expect("style marker");
        assert!(
            csp.contains(&hash_source(&style)),
            "CSP must pin the served style digest: {csp}"
        );
        assert!(csp.contains("frame-ancestors 'none'"));
        assert!(csp.contains("connect-src 'self'"));
    }

    #[tokio::test]
    async fn page_serves_with_security_headers() {
        let server = spawn();
        let res = server.get("/").await;
        assert_eq!(res.status_code(), axum::http::StatusCode::OK);
        let t = res.headers();
        assert_eq!(
            t.get("x-content-type-options").unwrap(),
            "nosniff",
            "nosniff required"
        );
        assert_eq!(
            t.get("referrer-policy").unwrap(),
            "no-referrer",
            "no-referrer required"
        );
        assert_eq!(
            t.get("cache-control").unwrap(),
            "no-store",
            "status page must not cache"
        );
        let csp = t.get("content-security-policy").expect("CSP required");
        let csp = csp.to_str().unwrap();
        assert!(
            csp.contains("frame-ancestors 'none'") && csp.contains("sha256-"),
            "CSP must hash-pin the inline blocks and bar framing: {csp}"
        );
        // No cookies are ever set: the UI is unauthenticated (loopback
        // bind) on purpose — this pins the absence of credential state.
        assert_eq!(
            t.get("set-cookie"),
            None,
            "the page must not introduce cookie state"
        );
    }
}
