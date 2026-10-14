//! Live cross-node expert dispatch E2E.
//!
//! `tensor_transport_serves_real_expert_weights_over_quic` is
//! self-contained (real QUIC server + store, no environment needed).
//!
//! `dispatches_to_peer_b_and_holds_the_honest_contract` runs only with
//! `--ignored` against a real running node, addressed by its identity:
//!   EXPECT_NODE_B_ID   — the peer's node_id (cert pinning SNI)
//!   EXPECT_RPC_ADDR    — its QUIC rpc address (default 127.0.0.1:8911)
//!   EXPECT_EXPERT_ID   — expert to ask for (default exp.router.test)
//! It asserts the Phase 2 contract: if the node hosts the expert we get
//! real activations plus a verifying Phase 4 commitment; if it does not,
//! we get a typed refusal — never fabricated numbers. Skips gracefully
//! when the environment isn't up, so normal `cargo test` stays green.

use std::net::SocketAddr;
use std::time::Duration;

use jerith_mesh_core::quic::{
    NodeRpcReply, NodeRpcRequest, PinMap, pinned_client_endpoint, sign_request,
};
use jerith_mesh_core::signing::SigningIdentity;

#[tokio::test]
#[ignore = "requires a running mesh node (dev E2E)"]
async fn dispatches_to_peer_b_and_holds_the_honest_contract() {
    let identity = SigningIdentity::generate();

    let addr: SocketAddr = std::env::var("EXPECT_RPC_ADDR")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "127.0.0.1:8911".to_string())
        .parse()
        .expect("EXPECT_RPC_ADDR must be host:port");
    let expert_id = std::env::var("EXPECT_EXPERT_ID")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "exp.router.test".to_string());

    // Cert pinning fails closed: the dial is pinned to node B's identity,
    // so the reply can only come from the node we meant.
    let sni_node = std::env::var("EXPECT_NODE_B_ID")
        .ok()
        .filter(|s| !s.is_empty())
        .expect("dev E2E: EXPECT_NODE_B_ID must carry node B's node_id");
    let mut pins: PinMap = std::collections::HashMap::new();
    pins.insert(sni_node.clone(), addr);
    let client = pinned_client_endpoint(std::sync::Arc::new(pins)).expect("pinned client endpoint");

    let input = vec![1.5f32, -2.0, 0.0, 3.0, 0.25, -0.75, 1.0, 0.5];
    let mut req = NodeRpcRequest::ExpertQuery {
        req_id: "e2e-cross-1".into(),
        expert_id: expert_id.clone(),
        input: input.clone(),
        from_node_id: identity.node_id(),
        stamp_ms: 0,
        sig: String::new(),
    };
    sign_request(&mut req, &identity);

    let reply = tokio::time::timeout(
        Duration::from_secs(10),
        jerith_mesh_core::quic::call(
            &client,
            addr,
            &sni_node,
            &req,
            &identity,
            Duration::from_secs(5),
        ),
    )
    .await
    .expect("dispatch timeout")
    .expect("dispatch failed");

    match reply {
        NodeRpcReply::ExpertResult {
            req_id,
            output,
            node_id,
            commitment,
        } => {
            assert_eq!(req_id, "e2e-cross-1");
            // Attribution: the server is node B (cert-pinned + self-reported).
            assert_eq!(node_id, sni_node);
            // Real activations: an expert forward pass returns a vector the
            // width of its hidden size, not a hash stub of some other length.
            assert!(!output.is_empty(), "expert output must not be empty");
            // Deterministic across calls (same weights, same input).
            let reply2 = jerith_mesh_core::quic::call(
                &client,
                addr,
                &sni_node,
                &req,
                &identity,
                Duration::from_secs(5),
            )
            .await
            .expect("second dispatch");
            match reply2 {
                NodeRpcReply::ExpertResult { output: out2, .. } => {
                    assert_eq!(out2, output, "expert forward must be deterministic");
                }
                other => panic!("unexpected reply: {other:?}"),
            }
            // Phase 4: the commitment binds the exact body it came with.
            let c = commitment
                .as_deref()
                .expect("ExpertResult must carry a Phase 4 commitment");
            assert!(
                jerith_mesh_core::verify::verify_commitment(
                    &req_id,
                    &expert_id,
                    &node_id,
                    &input,
                    &output,
                    c
                ),
                "commitment must bind req_id/expert_id/node_id/input/output"
            );
        }
        NodeRpcReply::Error { message, .. } => {
            // Honest refusal is a legitimate outcome: the node does not
            // host this expert and does not fabricate activations for it.
            assert!(
                message.contains("not hosted") || message.contains("refusing to fabricate"),
                "unexpected peer error (must be a typed refusal): {message}"
            );
            panic!(
                "node does not host {expert_id} — publish one with `jerith experts make` and restart, or point EXPECT_EXPERT_ID at a hosted expert (peer said: {message})"
            );
        }
        other => panic!("unexpected reply: {other:?}"),
    }
}

/// Phase 2 tensor transport, end to end and self-contained: a node serves
/// a real expert from its on-disk weight store over QUIC, a peer's router
/// dispatches an activation vector to it, and the returned vector must
/// equal an independently written SwiGLU computation on the same weights.
/// No runtime, no pre-existing mesh — this is the whole feature in one
/// test (the ignored test above stays for the dev two-node E2E).
#[tokio::test]
async fn tensor_transport_serves_real_expert_weights_over_quic() {
    use jerith_mesh_core::expert_store::{ExpertStore, ExpertWeights};
    use jerith_mesh_core::quic::{ExpertRouter, ExpertServing, RouterOptions, SharedPinMap};
    use std::collections::HashMap;
    use std::sync::Arc;

    // --- Node B: a store holding one real expert -------------------------
    let dir = std::env::temp_dir().join(format!("jerith-tensor-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let store = Arc::new(ExpertStore::new(&dir, 8).expect("store"));
    let weights = ExpertWeights::synthetic("L3.E17", 8, 6);
    store.put(&weights).expect("publish expert");
    assert_eq!(store.hosted(), vec!["L3.E17".to_string()]);

    // Server B serves that store and has NO chat runtime at all.
    let identity = SigningIdentity::generate();
    let server =
        jerith_mesh_core::quic::server_endpoint_for(&identity, "127.0.0.1:0".parse().unwrap())
            .expect("server endpoint");
    let addr = server.local_addr().unwrap();
    let node_id = identity.node_id();
    let serving = Arc::new(ExpertServing {
        store: Some(store.clone()),
        runtime: None,
        tensors_only: true,
    });
    let (_tx, rx) = tokio::sync::watch::channel(false);
    {
        let node_id = node_id.clone();
        tokio::spawn(async move {
            let _ = jerith_mesh_core::quic::serve(server, serving, node_id, rx).await;
        });
    }
    let mut pins: jerith_mesh_core::quic::PinMap = HashMap::new();
    pins.insert(node_id.clone(), addr);
    let pins: SharedPinMap = Arc::new(tokio::sync::RwLock::new(pins));
    let client = jerith_mesh_core::quic::pinned_client_endpoint_dyn(pins).expect("client endpoint");

    // --- Node A: the router asks B for the expert ------------------------
    let router = ExpertRouter::new(RouterOptions {
        nearby_rtt_ms: 50,
        dispatch_timeout: Duration::from_secs(10),
        ..Default::default()
    });
    let requester = SigningIdentity::generate();
    let mut peer_addrs: HashMap<String, SocketAddr> = HashMap::new();
    peer_addrs.insert(node_id.clone(), addr);
    // B advertises holding this expert, is alive, and is unreported (RTT
    // order, no resource sample — the pre-pooling path).
    let peers = vec![(node_id.clone(), 5u64, true, true, None)];
    let input = vec![0.5f32, -0.25, 1.0, 0.0, -1.5, 2.0, 0.75, -0.5];

    // Node A holds no local weights: `local_run` always fails, so the
    // result must come from B over the wire.
    let local_run = |_: &[f32]| -> anyhow::Result<Vec<f32>> {
        anyhow::bail!("node A holds no local expert weights")
    };
    let mut completed = HashMap::new();
    let mut served_by = None;
    let out = router
        .execute_over_quic_speculative_attributed(
            &client,
            &requester,
            &peer_addrs,
            "e2e-tensor-1",
            "L3.E17",
            &input,
            &peers,
            &mut completed,
            local_run,
            &mut served_by,
        )
        .await
        .expect("remote tensor dispatch");

    // Attribution: node B computed it, and the numbers are the expert's.
    assert_eq!(served_by.as_deref(), Some(node_id.as_str()));
    let want = reference_swiglu(&weights, &input);
    assert_eq!(out.len(), want.len());
    for (got, expected) in out.iter().zip(&want) {
        assert!(
            (got - expected).abs() < 1e-6,
            "activation mismatch: got {got}, want {expected}"
        );
    }
    // Second call is served from B's resident cache and stays bit-identical.
    let mut completed2 = HashMap::new();
    let mut served_by2 = None;
    let out2 = router
        .execute_over_quic_speculative_attributed(
            &client,
            &requester,
            &peer_addrs,
            "e2e-tensor-2",
            "L3.E17",
            &input,
            &peers,
            &mut completed2,
            local_run,
            &mut served_by2,
        )
        .await
        .expect("second dispatch");
    assert_eq!(out2, out, "expert forward must be deterministic");
    assert!(store.is_resident("L3.E17"), "hot expert stays resident");

    // --- Honest refusal: an expert B does not host -----------------------
    let mut completed3 = HashMap::new();
    let mut err_out = None;
    let err = router
        .execute_over_quic_speculative_attributed(
            &client,
            &requester,
            &peer_addrs,
            "e2e-tensor-3",
            "L9.E99",
            &input,
            &peers,
            &mut completed3,
            local_run,
            &mut err_out,
        )
        .await
        .expect_err("unhosted expert must fail, not fabricate");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("not hosted") || msg.contains("refusing to fabricate"),
        "unexpected error: {msg}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// Independent SwiGLU reference (loops only, no shared helpers) — the
/// arithmetic this test holds the wire result to.
fn reference_swiglu(w: &jerith_mesh_core::expert_store::ExpertWeights, x: &[f32]) -> Vec<f32> {
    let mut act = vec![0.0f32; w.intermediate];
    for unit in 0..w.intermediate {
        let mut g = 0.0f32;
        let mut u = 0.0f32;
        for k in 0..w.hidden {
            g += w.gate[unit * w.hidden + k] * x[k];
            u += w.up[unit * w.hidden + k] * x[k];
        }
        act[unit] = (g / (1.0 + (-g).exp())) * u;
    }
    let mut y = vec![0.0f32; w.hidden];
    for out in 0..w.hidden {
        let mut acc = 0.0f32;
        for unit in 0..w.intermediate {
            acc += w.down[out * w.intermediate + unit] * act[unit];
        }
        y[out] = acc;
    }
    y
}
