//! Minimal web UI: single-file status page served at the gateway port.
//!
//! Phase 0 scope: live status cards (model, contribution %, peers, meters).
//! Later phases turn this into the full observability dashboard.

use axum::extract::State;
use axum::http::header;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;

use crate::ping::MeshState;

pub fn router(state: std::sync::Arc<MeshState>) -> Router {
    Router::new()
        .route("/", get(index))
        .with_state(state)
}

async fn index(State(_gw): State<std::sync::Arc<MeshState>>) -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
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
    const r = await fetch('/status');
    const j = await r.json();
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
  } catch (e) { /* node stopped */ }
}
refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>
"#;
