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
  <div class="card"><h2>Sybil fence</h2><p id="sybil">—</p><div class="fine" id="sybilfine">zero-weight slash attempts (§10)</div></div>
  <div class="card"><h2>Wallet (JER)</h2><p id="earn">—</p><div class="fine" id="earnfine">chain balance · mining wins</div></div>
  <div class="card"><h2>Expert hit-rate</h2><p id="hit">—</p><div class="fine" id="hitfine">LRU cache + demand</div></div>
  <div class="card"><h2>Latency</h2><p id="latency">—</p><div class="fine" id="latencyfine">busiest model, this window</div></div>
  <div class="card"><h2>Trend (5-min buckets)</h2><p id="trend">—</p><div class="fine" id="trendfine">per-bucket load, oldest → newest</div></div>
  <div class="card"><h2>Mesh peers served</h2><p id="peerlist" style="font-size:.9rem">—</p><div class="fine" id="peerfine">per-peer latency breakdown</div></div>
  <div class="card"><h2>Pipeline routes</h2><p id="pipeline">—</p><div class="fine" id="pipelinefine">multi-hop walks</div></div>
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
    // Ops observability: 5-minute trend buckets (bounded ring on the
    // node) rendered as a text sparkline, plus the per-peer latency
    // breakdown of who actually served the mesh chat for us.
    if (mj && Array.isArray(mj.history) && mj.history.length) {
      const hist = mj.history;
      const counts = hist.map((b) => b.requests ?? 0);
      const top = Math.max(...counts, 1);
      document.getElementById('trend').innerHTML =
        '<svg width="190" height="34" viewBox="0 0 190 34" preserveAspectRatio="none">' +
        counts.map((c, i) => {
          const h = Math.round(30 * c / top) + 1;
          return '<rect x="' + i * 1.55 + '" y="' + (32 - h) + '" width="1.4" height="' + h +
            " fill='#58a6ff'></rect>";
        }).join('') + '</svg>';
      const lastH = hist.slice(-12); // last hour
      const req = lastH.reduce((s, b) => s + (b.requests ?? 0), 0);
      const lat = lastH.reduce((s, b) => s + (b.latency_ms_total ?? 0), 0);
      const mesh = lastH.reduce((s, b) => s + (b.mesh_served ?? 0), 0);
      document.getElementById('trendfine').textContent =
        `last hour: ${req} requests · mean ${req ? Math.round(lat / req) : '—'} ms` +
        ` · mesh-served ${mesh}/${req} · buckets ${hist.length}, 5 min each`;
    } else {
      document.getElementById('trend').textContent = '—';
      document.getElementById('trendfine').textContent = 'no completions recorded yet';
    }
    if (mj && Array.isArray(mj.peers) && mj.peers.length) {
      const short = (id) => (id || '?').slice(0, 10);
      document.getElementById('peerlist').textContent =
        mj.peers.map((p) =>
          `${short(p.peer)} ${p.requests}× ${p.requests ? p.latency_ms_mean ?? '—' : '—'}ms ${p.ok_percent ?? 0}%`
        ).join(' · ');
      document.getElementById('peerfine').textContent =
        `${mj.peers.length} peer${mj.peers.length === 1 ? '' : 's'} this window · id, count, mean ms, ok%`;
    } else {
      document.getElementById('peerlist').textContent = '—';
      document.getElementById('peerfine').textContent =
        'no mesh-peer completions yet (local runtime only)';
    }
    // M3 pipeline routes: counters exist and are tested; the live
    // producer rides the routing transport (pending the Colibri-side
    // layer split), so until then the card reads "no route data yet".
    const pj = mj ? mj.pipeline || null : null;
    document.getElementById('pipeline').textContent = pj ? (pj.routes_total ?? 0) : '—';
    document.getElementById('pipelinefine').textContent = pj
      ? `routes walked · over-budget hops ${pj.over_budget_hops ?? 0} · last walk ${pj.last_route_ms ?? 0} ms`
      : 'no route data yet';
    // §10 anti-sybil fence: OUR bootstrap's live zero-weight tally
    // (/status.bootstrap mirror). "—" = probe failed; "off" = none.
    const bs = j.bootstrap;
    if (bs && bs.reachable) {
      document.getElementById('sybil').textContent = bs.young_fraud_reports ?? 0;
      document.getElementById('sybilfine').textContent =
        `zero-weight slash attempts · bootstrap ${bs.url ?? '—'}`;
    } else if (bs) {
      document.getElementById('sybil').textContent = '—';
      document.getElementById('sybilfine').textContent =
        `bootstrap probe failed (${bs.error ?? 'status ' + (bs.status ?? '—')}) · zero-weight slashes unseen`;
    } else {
      document.getElementById('sybil').textContent = 'off';
      document.getElementById('sybilfine').textContent = 'no bootstrap URL configured';
    }
    // Earnings view: chain-truthful JER balance when the wallet card is
    // live (Phase 4); the local mines-won count stays the honest fallback
    // whenever the chain RPC is down or this node isn't a JerithChain one.
    const ch = j.chain;
    const chainInfo =
      (ch && ch.reachable
        ? ` · chain height ${ch.height ?? '—'} · circulating ${(ch.circulating_jer ?? '—')} JER`
        : ' · chain view unreachable');
    const w = j.wallet;
    if (w && w.reachable) {
      const bal = Number(w.balance_uj ?? 0) / 1e6;
      document.getElementById('earn').textContent = bal.toFixed(6) + ' JER';
      document.getElementById('earnfine').textContent =
        (w.exists
          ? `chain wallet ${w.identity}`
          : `chain wallet ${w.identity} · no wallet yet (first mined JER creates it)`) +
        (w.available_jer != null ? ` · spendable ${Number(w.available_jer).toFixed(6)}` : '') +
        ` · mined-this-node ${j.meters?.mines_won ?? 0}/${j.meters?.mines_attempted ?? 0}` +
        chainInfo;
    } else {
      document.getElementById('earn').textContent = String(j.meters?.mines_won ?? 0);
      document.getElementById('earnfine').textContent =
        `mining attempts ${j.meters?.mines_attempted ?? 0}` +
        (w && !w.reachable ? ` · wallet probe failed: ${w.error ?? '—'}` : ' · wallet card off (not JerithChain)') +
        chainInfo;
    }
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
