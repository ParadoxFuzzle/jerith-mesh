//! Node dashboard: single-file status + settings page at the gateway port.
//!
//! Live views (all fed same-origin from the node's own endpoints):
//! `/status` (node/mesh/chain/guard/experts), `/v1/metrics` (latency,
//! trends, per-peer + per-model drills), `/v1/skills/memory/stats`
//! (memory store) and `/v1/models` (runtime catalog). The settings panel
//! reads `GET /v1/config` and writes `POST /v1/config`; only the
//! hot-reloadable knobs ride that surface (identity/ports/model/DSN stay
//! CLI-only — see `dashconfig.rs`).
//!
//! Hardening: the page is served with a strict security-header set —
//! `Cache-Control: no-store`, `X-Content-Type-Options: nosniff`,
//! `Referrer-Policy: no-referrer`, and a CSP that pins the inline
//! `<script>`/`<style>` blocks by their sha256 digests (no
//! `'unsafe-inline'`), locks `frame-ancestors` to 'none' (clickjacking),
//! and drops `base-uri`/`form-action`/`object-src` to nothing. Hash
//! sources MUST be single-quoted (`script-src 'sha256-…'`) — unquoted
//! ones are invalid CSP and are silently ignored, which blocks the whole
//! inline script (live 2026-10-31: every card stuck at "—"). The digests
//! are computed from `PAGE` itself at first request, so page edits re-pin
//! automatically instead of rotting into a mismatch.

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
pub(crate) fn inline_block(page: &str, tag: &str) -> Option<String> {
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

/// `sha256-<digest>` for one inline block (unquoted; `build_csp` adds the
/// single quotes the CSP grammar requires around a hash-source).
pub(crate) fn hash_source(block: &str) -> String {
    let digest = Sha256::digest(block.as_bytes());
    format!("sha256-{}", b64(&digest))
}

/// The full CSP built from a page's own inline blocks. Falls back to
/// `'unsafe-inline'` per source if the markers ever move (defense must
/// survive edits that reshape the HTML). Generic over the page so the
/// bootstrap's vitals page (bootstrap_server) reuses it — one CSP
/// discipline, one place.
pub(crate) fn build_csp(page: &str) -> String {
    let script = match inline_block(page, "script") {
        // QUOTES ARE LOAD-BEARING: an unquoted `sha256-…` source is
        // invalid, browsers ignore it, and the inline script never runs.
        Some(s) => format!("'{}'", hash_source(&s)),
        None => "'unsafe-inline'".to_string(),
    };
    let style = match inline_block(page, "style") {
        Some(s) => format!("'{}'", hash_source(&s)),
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

const PAGE: &str = r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>jerith-mesh node</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; background: #0d1117; color: #c9d1d9; margin: 0; padding: 1.6rem 2rem 3rem; }
  header { display: flex; flex-wrap: wrap; align-items: baseline; gap: .75rem 1rem; margin-bottom: 1.2rem; }
  h1 { color: #58a6ff; font-weight: 600; font-size: 1.45rem; margin: 0; letter-spacing: .01em; }
  .meta { display: flex; flex-wrap: wrap; align-items: center; gap: .5rem; }
  .badge { border: 1px solid #30363d; border-radius: 999px; padding: .15rem .55rem; font-size: .72rem; color: #8b949e; background: #161b22; }
  .badge.ok { color: #3fb950; border-color: #238636; }
  .badge.err { color: #f85149; border-color: #da3633; }
  .muted { color: #8b949e; font-size: .78rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 1rem; max-width: 1180px; }
  .section { grid-column: 1 / -1; margin: 1.1rem 0 -.35rem; font-size: .72rem; text-transform: uppercase; letter-spacing: .12em; color: #8b949e; font-weight: 500; border-bottom: 1px solid #21262d; padding-bottom: .4rem; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1rem 1.05rem; min-width: 0; }
  .card h2 { margin: 0 0 .55rem; font-size: .74rem; text-transform: uppercase; color: #8b949e; letter-spacing: .08em; font-weight: 500; }
  .card p { margin: 0; font-size: 1.45rem; line-height: 1.2; overflow-wrap: anywhere; }
  .p-sm { font-size: .82rem !important; line-height: 1.45 !important; }
  .fine { color: #8b949e; font-size: .76rem; margin-top: 6px; line-height: 1.45; overflow-wrap: anywhere; }
  .card details { margin-top: .6rem; }
  .card summary { cursor: pointer; color: #8b949e; font-size: .76rem; }
  .card summary:hover, .card details[open] summary { color: #58a6ff; }
  .card table { width: 100%; border-collapse: collapse; font-size: .74rem; margin-top: .45rem; overflow-x: auto; }
  .card th { text-align: left; color: #8b949e; font-weight: 500; padding: 2px 8px 3px 0; border-bottom: 1px solid #30363d; }
  .card td { padding: 3px 8px 3px 0; border-bottom: 1px dotted #21262d; white-space: nowrap; }
  .wide { grid-column: 1 / -1; }
  .setgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: .7rem 1.1rem; }
  .setrow { display: flex; flex-direction: column; gap: .25rem; font-size: .78rem; color: #c9d1d9; }
  .setrow small { color: #8b949e; font-size: .7rem; line-height: 1.35; }
  .setrow input, .setrow select { background: #0d1117; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px; padding: .38rem .5rem; font: inherit; font-size: .78rem; width: 100%; }
  .setrow input:focus, .setrow select:focus { outline: none; border-color: #58a6ff; }
  .sharelist { display: flex; flex-direction: column; gap: .1rem; margin-top: .7rem; }
  .shrow { display: flex; align-items: center; gap: .6rem; font-size: .78rem; padding: .3rem .1rem; border-bottom: 1px dotted #21262d; cursor: pointer; }
  .shrow:hover { color: #58a6ff; }
  .shrow input { accent-color: #2ea043; }
  .shid { overflow-wrap: anywhere; }
  .chips { display: flex; flex-wrap: wrap; gap: .25rem; margin-left: auto; }
  .chip { border: 1px solid #30363d; border-radius: 999px; padding: .05rem .45rem; font-size: .68rem; color: #8b949e; }
  .chip.vision { color: #58a6ff; border-color: #1f6feb; }
  .chip.video { color: #d2a8ff; border-color: #8957e5; }
  .chip.embedding { color: #3fb950; border-color: #238636; }
  .chip.image_gen { color: #f0883e; border-color: #bd561d; }
  .actions { display: flex; flex-wrap: wrap; align-items: center; gap: .6rem; margin-top: .9rem; }
  .btn { background: #238636; border: 1px solid #2ea043; color: #fff; border-radius: 6px; padding: .4rem .9rem; font: inherit; font-size: .78rem; cursor: pointer; }
  .btn:hover { background: #2ea043; }
  .btn:disabled { opacity: .5; cursor: default; }
  .btn.ghost { background: transparent; border-color: #30363d; color: #c9d1d9; }
  .btn.ghost:hover { border-color: #58a6ff; color: #58a6ff; }
  footer { margin-top: 2rem; color: #8b949e; font-size: .78rem; line-height: 1.6; }
  .lnk { color: #58a6ff; text-decoration: none; }
  .lnk:hover { text-decoration: underline; }
</style>
</head>
<body>
<header>
  <h1>🦜 jerith-mesh</h1>
  <div class="meta">
    <span id="nodebadge" class="badge">node —</span>
    <span id="conn" class="badge">connecting…</span>
    <span id="stamp" class="muted">—</span>
  </div>
</header>
<div class="grid">

  <h2 class="section">Node</h2>
  <div class="card"><h2>Model</h2><p id="model">—</p><div class="fine" id="modelfine">quantization · role</div></div>
  <div class="card"><h2>Contribution</h2><p id="contrib">—%</p><div class="fine" id="contribfine">minimum 25% required</div></div>
  <div class="card"><h2>Uptime</h2><p id="uptime">—</p><div class="fine" id="uptimefine">this process</div></div>
  <div class="card"><h2>Runtime</h2><p id="runtime" class="p-sm">—</p><div class="fine" id="runtimefine">local OpenAI-compatible server</div></div>

  <h2 class="section">Mesh</h2>
  <div class="card"><h2>Peers alive</h2><p id="peers">—</p><div class="fine" id="peersfine">same-model nodes only</div></div>
  <div class="card"><h2>Tokens served</h2><p id="tokens">—</p><div class="fine" id="tokensfine">for the mesh, this process</div></div>
  <div class="card"><h2>Trend (5-min buckets)</h2><p id="trend">—</p><div class="fine" id="trendfine">per-bucket load, oldest → newest</div></div>
  <div class="card"><h2>Latency</h2><p id="latency">—</p><div class="fine" id="latencyfine">busiest model, this window</div></div>
  <div class="card"><h2>Mesh peers served</h2><p id="peerlist" class="p-sm">—</p><div class="fine" id="peerfine">per-peer latency breakdown</div></div>
  <div class="card"><h2>Per-model latency</h2><p id="mcount">—</p><div class="fine" id="mcountfine">models in this metering window</div><details><summary>drill-down: per-model rows</summary><table id="mtable"></table><div class="fine" id="mtablefine">fastest mean first · req, ok%, mean/max ms, mesh %, tokens</div></details></div>

  <h2 class="section">Memory &amp; models</h2>
  <div class="card"><h2>Memory entries</h2><p id="memcount">—</p><div class="fine" id="memfine">node-private store</div></div>
  <div class="card"><h2>Runtime models</h2><p id="models">—</p><div class="fine" id="modelsfine">catalog served by the runtime</div></div>
  <div class="card"><h2>Expert hit-rate</h2><p id="hit">—</p><div class="fine" id="hitfine">LRU cache + demand</div></div>
  <div class="card"><h2>Expert activations</h2><p id="acts">—</p><div class="fine" id="actsfine">loads served to peers</div></div>

  <h2 class="section">Earnings &amp; trust</h2>
  <div class="card"><h2>Wallet (JER)</h2><p id="earn">—</p><div class="fine" id="earnfine">chain balance · mining wins</div><details><summary>drill-down: earnings lines</summary><table id="wtable"></table><div class="fine" id="wtablefine">window, mining, wallet, chain</div></details></div>
  <div class="card"><h2>Stake gate</h2><p id="stake">—</p><div class="fine" id="stakefine">escrow · slash events</div></div>
  <div class="card"><h2>Phase 4 verification</h2><p id="ccommit">—</p><div class="fine" id="ccommitfine">commitment checks</div></div>
  <div class="card"><h2>Jerith Guard</h2><p id="guard">—</p><div class="fine" id="guardfine">outbound screens</div></div>
  <div class="card"><h2>Sybil fence</h2><p id="sybil">—</p><div class="fine" id="sybilfine">zero-weight slash attempts (§10)</div></div>

  <h2 class="section">Model sharing</h2>
  <div class="card wide">
    <h2>Share local models with the mesh</h2>
    <div class="fine" id="sharestatus">loading…</div>
    <div class="sharelist" id="sharelist"></div>
    <div class="actions">
      <button class="btn" id="sharesave" type="button">Save sharing</button>
      <button class="btn ghost" id="sharerefresh" type="button">Refresh</button>
    </div>
    <div class="fine" id="sharereadout">opt-in: every model is opted out until you tick it and save</div>
  </div>
  <div class="card wide">
    <h2>Mesh capability providers</h2>
    <p id="meshcaps" class="p-sm">—</p>
    <div class="fine" id="meshcapfine">alive peers publishing shared models · vision, video, embedding, text, image_gen</div>
  </div>

  <h2 class="section">Settings</h2>
  <div class="card wide">
    <h2>Node settings</h2>
    <div class="fine" id="setstatus">loading…</div>
    <div class="setgrid" id="settings"></div>
    <div class="actions">
      <button class="btn" id="save" type="button">Save settings</button>
      <button class="btn ghost" id="reload" type="button">Reload</button>
    </div>
    <div class="fine" id="readout"></div>
  </div>

</div>
<footer>
  jerith-mesh · a P2P compute mesh for MoE models ·
  <a class="lnk" href="/status">raw JSON</a> ·
  <a class="lnk" href="/v1/metrics">metrics JSON</a> ·
  <a class="lnk" href="/health">health</a> ·
  <a class="lnk" href="/v1/models">OpenAI API</a>
</footer>
<script>
const $ = (id) => document.getElementById(id);
const set = (id, value) => { const el = $(id); if (el) el.textContent = value; };
const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
  ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const fmt = (n) => (n == null ? '—' : Number(n).toLocaleString());
function uptime(sec) {
  if (sec == null) return '—';
  const s = Math.max(0, Math.floor(sec));
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600),
        m = Math.floor((s % 3600) / 60);
  if (d) return d + 'd ' + h + 'h';
  if (h) return h + 'h ' + m + 'm';
  return m + 'm ' + (s % 60) + 's';
}

async function refresh() {
  try {
    const [r, mr, memr, mdl] = await Promise.all([
      fetch('/status'),
      fetch('/v1/metrics').catch(() => null),
      fetch('/v1/skills/memory/stats').catch(() => null),
      fetch('/v1/models').catch(() => null),
    ]);
    if (!r.ok) throw new Error('status ' + r.status);
    const j = await r.json();
    const mj = mr && mr.ok ? await mr.json() : null;
    const mem = memr && memr.ok ? await memr.json() : null;
    const catalog = mdl && mdl.ok ? await mdl.json() : null;

    $('conn').className = 'badge ok';
    $('conn').textContent = 'live';
    set('stamp', 'updated ' + new Date().toLocaleTimeString());
    set('nodebadge', 'node ' + (j.node_id || '?').slice(0, 16));

    set('model', j.model || 'not set');
    set('modelfine', (j.quant || '—') + ' · ' + (j.role || '—'));
    set('contrib', (j.contribution_percent ?? '—') + '%');
    set('contribfine', `measured share this tick · role ${j.role || '—'}`);
    set('uptime', uptime(j.meters?.uptime_seconds));
    set('uptimefine', `${fmt(j.meters?.tokens_served)} tokens served · ${fmt(j.meters?.expert_activations)} expert loads`);
    set('runtime', j.runtime_url || '—');
    set('runtimefine', `max tokens ${j.runtime_num_predict ?? 'uncapped'} · ${j.peers_alive ?? 0} peer(s) alive`);

    set('peers', j.peers_alive ?? '—');
    set('peersfine', `${j.peer_count ?? 0} known · same-model only`);
    set('tokens', fmt(j.meters?.tokens_served));
    set('tokensfine', `${fmt(j.meters?.tokens_consumed)} consumed · ${fmt(j.meters?.bytes_relayed)} bytes relayed`);
    set('acts', fmt(j.meters?.expert_activations));
    set('actsfine', `token bytes consumed ${fmt(j.meters?.tokens_consumed)}`);

    set('memcount', mem ? fmt(mem.count) : (j.memory_store ? '0' : 'off'));
    const dims = (mem && Array.isArray(mem.dims) && mem.dims.length)
      ? ' · dims ' + mem.dims.map((d) => d[0] + '×' + fmt(d[1])).join(', ')
      : '';
    set('memfine', (j.memory_store || 'no memory store') + dims);
    if (mdl === null || !catalog) {
      set('models', '—');
      set('modelsfine', 'runtime catalog unavailable');
    } else {
      const n = (catalog.data || []).length;
      set('models', fmt(n));
      set('modelsfine', `base ${j.model || '—'} · ${j.runtime_url || ''}`);
    }

    set('stake', j.stake?.gate_clear ? 'clear' : 'locked');
    set('stakefine', `${fmt(j.stake?.staked_uj)} / ${fmt(j.stake?.required_uj)} µJER · slash events ${j.stake?.slash_events ?? 0} · refusals ${j.stake?.gate_refusals ?? 0}`);

    const v = j.verify ?? {};
    set('ccommit', fmt(v.commitment_checks));
    set('ccommitfine',
      `failures ${v.commitment_failures ?? 0} · replays ${v.replay_matches ?? 0}/${v.replays_dispatched ?? 0}` +
      ` · fraud reports ${v.fraud_reports_flushed ?? 0}`);

    const g = j.guard;
    set('guard', g ? fmt(g.screens) : 'off');
    set('guardfine', g
      ? `blocked ${g.requests_blocked ?? 0} · redacted ${g.values_redacted ?? 0} · logged ${g.detections_logged ?? 0}`
      : 'guard disabled in config.toml ([guard] enabled = false)');

    const bs = j.bootstrap;
    if (bs && bs.reachable) {
      set('sybil', bs.young_fraud_reports ?? 0);
      set('sybilfine', `zero-weight slash attempts · bootstrap ${bs.url ?? '—'}`);
    } else if (bs) {
      set('sybil', '—');
      set('sybilfine', `bootstrap probe failed (${bs.error ?? 'status ' + (bs.status ?? '—')})`);
    } else {
      set('sybil', 'off');
      set('sybilfine', 'no bootstrap URL configured');
    }

    const ex = j.experts;
    if (ex && (ex.hits ?? 0) + (ex.misses ?? 0) > 0) {
      const hr = Math.round(100 * ex.hits / (ex.hits + ex.misses));
      set('hit', hr + '%');
      set('hitfine', `hits/misses ${ex.hits}/${ex.misses} · ${ex.activations_total ?? 0} activations on ${ex.activations_distinct ?? 0} experts · hot ${Array.isArray(ex.hot) ? ex.hot.length : '?'} / cold ${ex.cold ?? '?'} · prefetch ${ex.prefetches ?? 0}`);
    } else if (ex) {
      set('hit', '—');
      set('hitfine', `no loads yet · ${ex.activations_total ?? 0} activations on ${ex.activations_distinct ?? 0} experts`);
    } else {
      set('hit', 'off');
      set('hitfine', 'no expert store on this node');
    }

    const models = Object.entries((mj && mj.models) || {});
    if (models.length) {
      const busiest = models.reduce((a, b) => ((b[1].requests ?? 0) >= (a[1].requests ?? 0) ? b : a));
      const st = busiest[1];
      set('latency', st.latency_ms_mean != null && st.requests ? st.latency_ms_mean + ' ms' : '—');
      set('latencyfine', `${busiest[0]} · max ${st.latency_ms_max ?? '—'} ms · ok ${st.ok_percent ?? '—'}% · mesh ${st.mesh_served ?? 0}/${st.requests ?? 0}`);
    } else {
      set('latency', '—');
      set('latencyfine', 'no completions in the window yet');
    }

    const mrows = Object.entries((mj && mj.models) || {}).map(([name, s]) => ({
      name,
      req: s.requests ?? 0,
      okp: s.requests ? Math.round(100 * (s.ok ?? 0) / s.requests) : null,
      mean: s.requests ? Math.round((s.latency_ms_total ?? 0) / s.requests) : null,
      max: s.latency_ms_max ?? 0,
      mesh: s.requests ? Math.round(100 * (s.mesh_served ?? 0) / s.requests) : 0,
      tok: s.total_tokens ?? 0,
    }));
    set('mcount', mrows.length || '—');
    set('mcountfine', mrows.length
      ? `requests ${fmt(mj.requests_total)} · window open ${mj.window_started_ms ? new Date(mj.window_started_ms).toLocaleString() : '—'}`
      : 'no completions in the window yet');
    if (mrows.length) {
      mrows.sort((a, b) => (a.mean ?? Infinity) - (b.mean ?? Infinity));
      document.getElementById('mtable').innerHTML =
        '<tr><th>model</th><th>req</th><th>ok%</th><th>mean</th><th>max</th><th>mesh</th><th>tok</th></tr>' +
        mrows.map((r) =>
          `<tr><td>${esc(r.name)}</td><td>${r.req}</td><td>${r.okp ?? '—'}</td><td>${r.mean ?? '—'}</td><td>${r.max}</td><td>${r.mesh}%</td><td>${r.tok}</td></tr>`
        ).join('');
      set('mtablefine', 'fastest mean first · mesh % = completions served by peers');
    } else {
      document.getElementById('mtable').textContent = '';
      set('mtablefine', 'no per-model rows yet');
    }

    if (mj && Array.isArray(mj.history) && mj.history.length) {
      const hist = mj.history;
      const counts = hist.map((b) => b.requests ?? 0);
      const top = Math.max(...counts, 1);
      document.getElementById('trend').innerHTML =
        '<svg width="190" height="34" viewBox="0 0 190 34" preserveAspectRatio="none">' +
        counts.map((c, i) => {
          const h = Math.round(30 * c / top) + 1;
          return '<rect x="' + i * 1.55 + '" y="' + (32 - h) + '" width="1.4" height="' + h +
            ' fill="#58a6ff"></rect>';
        }).join('') + '</svg>';
      const lastH = hist.slice(-12);
      const req = lastH.reduce((s, b) => s + (b.requests ?? 0), 0);
      const lat = lastH.reduce((s, b) => s + (b.latency_ms_total ?? 0), 0);
      const mesh = lastH.reduce((s, b) => s + (b.mesh_served ?? 0), 0);
      set('trendfine', `last hour: ${req} requests · mean ${req ? Math.round(lat / req) : '—'} ms · mesh-served ${mesh}/${req} · buckets ${hist.length}, 5 min each`);
    } else {
      set('trend', '—');
      set('trendfine', 'no completions recorded yet');
    }

    if (mj && Array.isArray(mj.peers) && mj.peers.length) {
      const short = (id) => (id || '?').slice(0, 10);
      set('peerlist', mj.peers.map((p) =>
        `${short(p.peer)} ${p.requests}× ${p.requests ? p.latency_ms_mean ?? '—' : '—'}ms ${p.ok_percent ?? 0}%`
      ).join(' · '));
      set('peerfine', `${mj.peers.length} peer${mj.peers.length === 1 ? '' : 's'} this window · id, count, mean ms, ok%`);
    } else {
      set('peerlist', '—');
      set('peerfine', 'no mesh-peer completions yet (local runtime only)');
    }

    const pj = mj ? mj.pipeline || null : null;
    // (pipeline counters have no producer yet; the card stays honest)
    const ch = j.chain;
    const chainInfo = ch && ch.reachable
      ? ` · chain height ${ch.height ?? '—'} · circulating ${ch.circulating_jer ?? '—'} JER`
      : ' · chain view unreachable';
    const w = j.wallet;
    if (w && w.reachable) {
      const bal = Number(w.balance_uj ?? 0) / 1e6;
      set('earn', bal.toFixed(6) + ' JER');
      set('earnfine',
        (w.exists ? `chain wallet ${w.identity}` : `chain wallet ${w.identity} · no wallet yet (first mined JER creates it)`) +
        (w.available_jer != null ? ` · spendable ${Number(w.available_jer).toFixed(6)}` : '') +
        ` · mined ${j.meters?.mines_won ?? 0}/${j.meters?.mines_attempted ?? 0}` + chainInfo);
    } else {
      set('earn', String(j.meters?.mines_won ?? 0));
      set('earnfine',
        `mining attempts ${j.meters?.mines_attempted ?? 0}` +
        (w && !w.reachable ? ` · wallet probe failed: ${w.error ?? '—'}` : ' · wallet card off (not JerithChain)') + chainInfo);
    }
    const mkrow = (k, v) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`;
    document.getElementById('wtable').innerHTML =
      (mj ? mkrow('window open', new Date(mj.window_started_ms || Date.now()).toLocaleString()) : '') +
      mkrow('tokens served', j.meters?.tokens_served ?? '—') +
      mkrow('expert activations', j.meters?.expert_activations ?? '—') +
      mkrow('mining wins / attempts', `${j.meters?.mines_won ?? 0} / ${j.meters?.mines_attempted ?? 0}`) +
      (w && w.exists ? mkrow('wallet identity', w.identity) : '') +
      (w && w.reachable ? mkrow('balance (JER)', (Number(w.balance_uj ?? 0) / 1e6).toFixed(6)) : '') +
      (w && w.reachable && w.available_jer != null ? mkrow('spendable (JER)', Number(w.available_jer).toFixed(6)) : '') +
      (ch && ch.reachable ? mkrow('chain height', ch.height ?? '—') : '') +
      (ch && ch.reachable ? mkrow('circulating (JER)', ch.circulating_jer ?? '—') : '');
    set('wtablefine',
      (w && w.reachable ? 'wallet live' : 'wallet offline — local tallies only') +
      (ch && ch.reachable ? ' · chain live' : ' · chain unreachable') +
      (pj ? ' · pipeline routes ' + (pj.routes_total ?? 0) : ''));
  } catch (e) {
    $('conn').className = 'badge err';
    $('conn').textContent = 'unreachable';
    set('stamp', String(e));
  }
}

async function loadSettings() {
  try {
    const r = await fetch('/v1/config');
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || ('status ' + r.status));
    const box = $('settings');
    box.textContent = '';
    (j.settings || []).forEach((s) => {
      const row = document.createElement('label');
      row.className = 'setrow';
      const name = document.createElement('span');
      name.textContent = s.label;
      let input;
      if (s.kind === 'select') {
        input = document.createElement('select');
        (s.options || []).forEach((o) => {
          const opt = document.createElement('option');
          opt.value = o;
          opt.textContent = o;
          input.append(opt);
        });
      } else {
        input = document.createElement('input');
        input.type = s.kind === 'number' ? 'number' : 'text';
        if (s.min != null) input.min = String(s.min);
        if (s.kind === 'number') input.step = '1';
      }
      input.value = s.value ?? '';
      input.dataset.key = s.key;
      const help = document.createElement('small');
      help.textContent = s.help || '';
      row.append(name, input, help);
      box.append(row);
    });
    set('readout', (j.readonly || []).map((x) => `${x.label}: ${x.value}`).join('  ·  ') + ' · ' + (j.note || ''));
    set('setstatus', 'editing the hot-reloadable knobs · save writes config.toml');
  } catch (e) {
    set('setstatus', 'settings unavailable: ' + e);
  }
}

async function loadSharing() {
  try {
    const r = await fetch('/v1/sharing');
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || ('status ' + r.status));
    const box = $('sharelist');
    box.textContent = '';
    const models = j.models || [];
    models.forEach((m) => {
      const row = document.createElement('label');
      row.className = 'shrow';
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = !!m.shared;
      cb.dataset.model = m.id;
      const id = document.createElement('span');
      id.className = 'shid';
      id.textContent = m.id + (m.base ? ' (base)' : '') + (m.available === false ? ' (not served)' : '');
      const chips = document.createElement('span');
      chips.className = 'chips';
      (m.capabilities || []).forEach((c) => {
        const chip = document.createElement('span');
        chip.className = 'chip ' + String(c).replace(/[^a-z_]/g, '');
        chip.textContent = c;
        chips.append(chip);
      });
      row.append(cb, id, chips);
      box.append(row);
    });
    const shared = j.sharing || {};
    set('sharestatus', models.length
      ? `${models.length} runtime model(s) · ${(shared.models || []).length} shared · runtime ${j.runtime_reachable ? 'reachable' : 'unreachable'}`
      : 'no runtime models listed' + (j.runtime_error ? ' — ' + j.runtime_error : ''));
    set('sharereadout', 'opt-in: every model is opted out until you tick it and save'
      + ((shared.capabilities || []).length ? ' · advertising: ' + shared.capabilities.join(', ') : ' · advertising nothing'));
    const mc = (j.mesh && j.mesh.providers) || {};
    const parts = Object.entries(mc).map(([k, v]) =>
      k + ': ' + (Array.isArray(v) && v.length ? v.map((x) => String(x).slice(0, 10)).join(', ') : '—'));
    set('meshcaps', parts.length ? parts.join(' · ') : '—');
    set('meshcapfine', `${((j.mesh && j.mesh.peers) || []).length} peer(s) known · providers are opt-in peers, not assumptions`);
  } catch (e) {
    set('sharestatus', 'sharing unavailable: ' + e);
  }
}

$('sharesave').addEventListener('click', async () => {
  const models = [];
  document.querySelectorAll('#sharelist input[data-model]').forEach((el) => {
    if (el.checked) models.push(el.dataset.model);
  });
  set('sharestatus', 'saving…');
  $('sharesave').disabled = true;
  try {
    const r = await fetch('/v1/sharing', {
      method: 'POST',
      headers: {'content-type': 'application/json', 'x-jerith-config': '1'},
      body: JSON.stringify({models}),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || ('status ' + r.status));
    set('sharestatus', 'saved — sharing ' + ((j.shared || []).length ? j.shared.join(', ') : 'nothing')
      + ' · ' + j.note);
    setTimeout(loadSharing, 900);
  } catch (e) {
    set('sharestatus', 'share save failed: ' + e);
  } finally {
    $('sharesave').disabled = false;
  }
});
$('sharerefresh').addEventListener('click', loadSharing);

$('save').addEventListener('click', async () => {
  const changes = {};
  document.querySelectorAll('#settings [data-key]').forEach((el) => { changes[el.dataset.key] = el.value; });
  set('setstatus', 'saving…');
  $('save').disabled = true;
  try {
    const r = await fetch('/v1/config', {
      method: 'POST',
      headers: {'content-type': 'application/json', 'x-jerith-config': '1'},
      body: JSON.stringify({changes}),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || ('status ' + r.status));
    set('setstatus', `saved ${j.applied.join(', ')} · ${j.note}`);
    setTimeout(loadSettings, 1200);
  } catch (e) {
    set('setstatus', 'save failed: ' + e);
  } finally {
    $('save').disabled = false;
  }
});
$('reload').addEventListener('click', loadSettings);

refresh();
loadSettings();
loadSharing();
setInterval(refresh, 3000);
setInterval(loadSharing, 15000);
</script>
</body>
</html>
"##;

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

    /// The 2026-10-31 outage: unquoted hash sources are invalid CSP, so
    /// browsers ignored them and blocked the page's whole inline script
    /// (every card stuck at "—"). Quotes are the fix; pin them.
    #[test]
    fn csp_hash_sources_are_quoted() {
        let csp = build_csp(PAGE);
        assert!(
            csp.contains("script-src 'sha256-"),
            "quoted script hash: {csp}"
        );
        assert!(
            csp.contains("style-src 'sha256-"),
            "quoted style hash: {csp}"
        );
        assert!(
            !csp.contains("script-src sha256-") && !csp.contains("style-src sha256-"),
            "unquoted hash sources are ignored by browsers: {csp}"
        );
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
            csp.contains("frame-ancestors 'none'") && csp.contains("'sha256-"),
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

    #[test]
    fn drilldowns_ship_with_escaped_cells() {
        // Both drill-down tables are in the page.
        assert!(
            PAGE.contains(r#"id="mtable""#),
            "per-model drill-down table"
        );
        assert!(PAGE.contains(r#"id="wtable""#), "earnings drill-down table");
        // Model/action names ride the unauthenticated API surface as
        // free strings: the page must escape them before innerHTML.
        assert!(PAGE.contains("const esc = (s) =>"), "cell escaper present");
        assert!(PAGE.contains("esc(r.name)"), "per-model cells escaped");
        assert!(PAGE.contains("esc(k)"), "earnings label cells escaped");
    }

    /// The settings panel ships in this page and talks to the guarded
    /// same-origin endpoint (the header forces a CORS preflight, which a
    /// cross-site caller cannot pass).
    #[test]
    fn settings_panel_ships_with_the_guarded_endpoint() {
        assert!(PAGE.contains(r#"id="settings""#));
        assert!(PAGE.contains("fetch('/v1/config')"));
        assert!(PAGE.contains("'x-jerith-config': '1'"));
        assert!(PAGE.contains("Save settings"));
        // Model sharing: per-model opt-in toggles over detected
        // capabilities, saved through the same guarded header.
        assert!(PAGE.contains(r#"id="sharelist""#), "sharing list ships");
        assert!(PAGE.contains("fetch('/v1/sharing')"), "sharing panel wired");
        assert!(PAGE.contains("Save sharing"));
        assert!(PAGE.contains(r#"id="meshcaps""#), "provider view ships");
        // No inline style attributes: CSP pins the <style> block only, so
        // `style="…"` attributes would be blocked (and silently degrade).
        assert!(
            !PAGE.contains("style=\""),
            "inline style attributes are CSP-blocked"
        );
    }
}
