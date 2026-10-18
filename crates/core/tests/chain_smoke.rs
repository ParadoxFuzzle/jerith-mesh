//! Env-gated live-chain smoke tests (§13 "Chain-side hardening").
//!
//! The testing-posture pattern every backend smoke suite copies is the
//! live-PG one (`JERITH_TEST_PG_URL`): gated by env, SKIPPING LOUDLY with
//! the reason when absent — never silently. This suite is the chain
//! node's adopter: `JERITH_TEST_CHAIN_URL` set to the operator's chain
//! api base (http://127.0.0.1:8322 on the co-hosted deployment) turns
//! these on; the api token comes from the mesh default (`JERITH_TOKEN`
//! or `keys/api.token`).
//!
//! READ-ONLY by construction: everything here GETs public views
//! (`/status`, `/balance`, `/stake?discord_id=`) whose handlers are the
//! same ones the mesh's /status probe, wallet card and stake resolver
//! ride. No POST, no mutation — a smoke run can never move escrow or
//! mint. The set pinns the CONTTRACTS the mesh code assumes:
//!
//! 1. `/status` shape: height, tip_hash, mined totals, supply invariants
//!    (circulating ≤ max), the miner count.
//! 2. Token auth is REAL: the same GET without the bearer header must be
//!    refused (401/403) — the mesh rides this every probe.
//! 3. `/balance` for a never-created identity answers `exists:false`
//!    with zero balances (NOT a transport error).
//! 4. `/stake?discord_id=` view: numeric µJ fields when the identity is
//!    known; honest 4xx when it isn't (the resolver treats any non-2xx
//!    as "no view" — the shape must never be a malformed 200).
//! 5. Bookkeeping is monotonic across two live reads — the drift watch's
//!    whole premise, checked against the real chain.

use jerith_mesh_core::economics;
use jerith_mesh_core::http::mesh_http;
use serde_json::Value;
use std::time::Duration;

const GATE_ENV: &str = "JERITH_TEST_CHAIN_URL";
/// The synthetic probe identity (never created, never funded): exercises
/// the unknown-identity contracts without touching any real wallet.
const PROBE_ID: &str = "codebuff-chain-smoke-probe";

/// Resolve the gate: the chain base from `JERITH_TEST_CHAIN_URL`, the
/// api token from the mesh default. `None` = skip — with the reason
/// printed ONCE per test (loud, searchable, exactly like the PG gates).
/// The resolved target, latched once (parallel test bodies share it).
static TARGET: std::sync::OnceLock<String> = std::sync::OnceLock::new();

async fn gate() -> Option<reqwest::Client> {
    let base = match std::env::var(GATE_ENV) {
        Ok(b) if !b.trim().is_empty() => b.trim().trim_end_matches('/').to_string(),
        _ => {
            eprintln!(
                "skipped: {GATE_ENV} not set — no live chain target (set it, e.g. \
                 JERITH_TEST_CHAIN_URL=http://127.0.0.1:8322 cargo test -p jerith-mesh-core \
                 --test chain_smoke; the skip is loud BY DESIGN, never silent)"
            );
            return None;
        }
    };
    // The token: only existence is this function's business — tests that
    // need to REFUSE wrong tokens probe the anonymous path themselves.
    match economics::read_token() {
        Some(t) if !t.trim().is_empty() => {
            let _ = t; // tests re-read via economics::read_token per call
            TARGET.get_or_init(|| base.clone());
            Some(mesh_http())
        }
        _ => {
            eprintln!(
                "skipped: no chain api token — set JERITH_TOKEN or {} \
                 (the chain 401s everything without it; the SKIP is loud by design)",
                economics::default_token_file().display()
            );
            None
        }
    }
}

fn chain_base() -> String {
    TARGET.get().expect("gate() ran first").clone()
}

fn authed_get(client: &reqwest::Client, path: &str) -> reqwest::RequestBuilder {
    let token = economics::read_token().expect("gate() verified the token");
    client
        .get(format!("{}{}", chain_base(), path))
        .bearer_auth(token.trim())
        .timeout(Duration::from_secs(5))
}

#[tokio::test]
async fn chain_status_shape_checks() {
    let Some(client) = gate().await else { return };
    let resp = authed_get(&client, "/status")
        .send()
        .await
        .expect("chain /status reachable");
    assert!(
        resp.status().is_success(),
        "authed /status must answer 2xx on a healthy chain node, got {}",
        resp.status()
    );
    let body: Value = resp.json().await.expect("/status is JSON");

    let height = body
        .pointer("/chain/height")
        .and_then(|v| v.as_f64())
        .expect("chain.height is a number");
    assert!(
        height >= 1.0,
        "a live chain has a real height, got {height}"
    );

    let tip = body
        .pointer("/chain/tip_hash")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    assert!(!tip.is_empty(), "chain.tip_hash is a non-empty string");

    let total_mined = body
        .pointer("/mining/total_mined_uj")
        .and_then(|v| v.as_f64())
        .expect("mining.total_mined_uj is a number");
    assert!(total_mined >= 0.0, "mined total is non-negative");

    let circulating = body
        .pointer("/chain/circulating_jer")
        .and_then(|v| v.as_f64())
        .expect("chain.circulating_jer is a number");
    let max_supply = body
        .pointer("/chain/max_supply_jer")
        .and_then(|v| v.as_f64())
        .expect("chain.max_supply_jer is a number");
    assert!(
        circulating <= max_supply,
        "supply invariant: circulating {circulating} ≤ max_supply {max_supply}"
    ); // LIVE contract fix from the first real-chain run: miners is a
    // COUNT (int), not a roster — the gateway mirrors it verbatim as a
    // number. Assert the numeric contract, not an imagined list.
    let miners = body
        .pointer("/mining/miners")
        .and_then(|v| v.as_f64())
        .expect("mining.miners is a number (the miner count)");
    assert!(miners >= 0.0, "miner count is non-negative, got {miners}");
}

#[tokio::test]
async fn unauthenticated_status_is_refused() {
    let Some(client) = gate().await else { return };
    let resp = client
        .get(format!("{}/status", chain_base()))
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .expect("chain reachable");
    let status = resp.status().as_u16();
    assert!(
        (401..=403).contains(&status),
        "the chain must refuse token-less reads (the mesh rides bearer auth on \
         every probe), got {status}"
    );
}

#[tokio::test]
async fn balance_contract_for_unknown_identity() {
    let Some(client) = gate().await else { return };
    let resp = authed_get(&client, &format!("/balance?discord_id={PROBE_ID}"))
        .send()
        .await
        .expect("chain /balance reachable");
    assert!(
        resp.status().is_success(),
        "unknown identity must NOT be a transport error, got {}",
        resp.status()
    );
    let body: Value = resp.json().await.expect("/balance is JSON");
    let exists = body.pointer("/exists").and_then(|v| v.as_bool());
    assert_eq!(
        exists,
        Some(false),
        "a never-created identity reads exists:false (the exact chain-node \
         contract the wallet card renders as 0-balance, not unreachable)"
    );
    let balance = body
        .pointer("/balance")
        .or_else(|| body.pointer("/balance_uj"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    assert_eq!(balance, 0.0, "unknown identity holds nothing");
}

#[tokio::test]
async fn stake_view_shape() {
    let Some(client) = gate().await else { return };
    let resp = authed_get(&client, &format!("/stake?discord_id={PROBE_ID}"))
        .send()
        .await
        .expect("chain reachable");
    let status = resp.status().as_u16();
    if !(200..300).contains(&status) {
        // The honest alternatives for an unknown identity: a 4xx (resolver
        // contract: any non-2xx = "no view", quiet debug, no parse) or —
        // equally valid — an all-zero 2xx view.
        assert!(
            (400..500).contains(&status),
            "unexpected chain behavior for the probe identity: {status}"
        );
        eprintln!("stake view: unknown identity → {status} (contract: non-2xx = no view)");
        return;
    }
    let body: Value = resp.json().await.expect("/stake is JSON");
    // Both µJ fields must be numeric when the view exists — the mesh's
    // refresh parses exactly these with as_u64.
    body.pointer("/staked_uj")
        .and_then(|v| v.as_u64())
        .expect("staked_uj is a u64");
    body.pointer("/slashed_total_uj")
        .and_then(|v| v.as_u64())
        .expect("slashed_total_uj is a u64");
}

#[tokio::test]
async fn bookkeeping_is_monotonic_across_two_reads() {
    let Some(client) = gate().await else { return };
    let read = || async {
        let body: Value = authed_get(&client, "/status")
            .send()
            .await
            .expect("chain reachable")
            .json()
            .await
            .expect("/status is JSON");
        (
            body.pointer("/mining/total_mined_uj")
                .and_then(|v| v.as_f64()),
            body.pointer("/chain/height").and_then(|v| v.as_f64()),
        )
    };
    let (t1, h1) = read().await;
    tokio::time::sleep(Duration::from_millis(400)).await;
    let (t2, h2) = read().await;
    let (t1, h1, t2, h2) = (
        t1.expect("total_mined_uj present"),
        h1.expect("height present"),
        t2.expect("total_mined_uj present"),
        h2.expect("height present"),
    );
    assert!(
        t2 >= t1,
        "mined total is monotonic between live reads: {t1} → {t2} (a decrease \
         is the drift watch's regression class)"
    );
    assert!(
        h2 >= h1,
        "height is monotonic between live reads: {h1} → {h2}"
    );
}
