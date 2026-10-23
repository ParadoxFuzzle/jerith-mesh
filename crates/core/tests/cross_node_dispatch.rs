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
                    &req_id, &expert_id, &node_id, &input, &output, c
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
        stats: None,
        isolate: None,
        prefetch: None,
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
    let peers = vec![(node_id.clone(), 5u64, true, true, None, None)];
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
            jerith_mesh_core::pipeline::RequestBudget::default(),
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
            jerith_mesh_core::pipeline::RequestBudget::default(),
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
            jerith_mesh_core::pipeline::RequestBudget::default(),
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

/// Phase 1 resource isolation, on the wire: a node whose kill switch has
/// tripped refuses an expert query HONESTLY — a typed error that names
/// the switch, never invented activations, never a hang. The router must
/// surface that refusal as its walk error (no other candidate exists).
/// The ringer/builtin path is exercised separately (it must stay live
/// even tripped); this covers the real weight-serving gate.
#[tokio::test]
async fn kill_switch_tripped_refuses_serving_honestly_over_quic() {
    use jerith_mesh_core::expert_store::{ExpertStore, ExpertWeights};
    use jerith_mesh_core::isolate::IsolateState;
    use jerith_mesh_core::quic::{ExpertRouter, ExpertServing, RouterOptions, SharedPinMap};
    use std::collections::HashMap;
    use std::sync::Arc;

    // --- Node B: a real hosted expert, but the switch is TRIPPED ---------
    let dir = std::env::temp_dir().join(format!("jerith-killswitch-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let store = Arc::new(ExpertStore::new(&dir, 8).expect("store"));
    let weights = ExpertWeights::synthetic("L3.E17", 8, 6);
    store.put(&weights).expect("publish expert");

    let iso = Arc::new(IsolateState::default());
    assert!(!iso.tripped());
    // Trip it exactly as the governor would: one transition, root cause.
    assert!(iso.record_trip("test: RAM ceiling breached"));
    assert!(iso.tripped());

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
        stats: None,
        isolate: Some(iso.clone()),
        prefetch: None,
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

    // --- Node A: dispatch to B and expect the honest refusal -------------
    let router = ExpertRouter::new(RouterOptions {
        nearby_rtt_ms: 50,
        dispatch_timeout: Duration::from_secs(10),
        ..Default::default()
    });
    let requester = SigningIdentity::generate();
    let mut peer_addrs: HashMap<String, SocketAddr> = HashMap::new();
    peer_addrs.insert(node_id.clone(), addr);
    let peers = vec![(node_id.clone(), 5u64, true, true, None, None)];
    let input = vec![0.5f32, -0.25, 1.0, 0.0, -1.5, 2.0, 0.75, -0.5];
    let local_run = |_: &[f32]| -> anyhow::Result<Vec<f32>> {
        anyhow::bail!("node A holds no local expert weights")
    };

    let mut completed = HashMap::new();
    let mut served_by = None;
    let err = router
        .execute_over_quic_speculative_attributed(
            &client,
            &requester,
            &peer_addrs,
            "killswitch-1",
            "L3.E17",
            &input,
            &peers,
            &mut completed,
            local_run,
            &mut served_by,
            jerith_mesh_core::pipeline::RequestBudget::default(),
        )
        .await
        .expect_err("tripped peer must refuse, not serve");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("kill switch"),
        "refusal must NAME the kill switch (honest, typed): {msg}"
    );
    assert!(
        msg.contains("refused"),
        "refusal must be a refusal, not data: {msg}"
    );
    assert_eq!(served_by, None, "nobody computed anything");

    // The same node, untripped, serves the expert normally — proving the
    // refusal is the switch's doing and recovery is just the flag (the
    // governor's automatic un-trip has no other path to re-arm serving).
    assert!(iso.record_recover());
    let mut completed2 = HashMap::new();
    let mut served_by2: Option<String> = None;
    let out = router
        .execute_over_quic_speculative_attributed(
            &client,
            &requester,
            &peer_addrs,
            "killswitch-2",
            "L3.E17",
            &input,
            &peers,
            &mut completed2,
            |_x: &[f32]| anyhow::bail!("still no local weights"),
            &mut served_by2,
            jerith_mesh_core::pipeline::RequestBudget::default(),
        )
        .await
        .expect("untripped peer must serve again");
    assert_eq!(served_by2.as_deref(), Some(node_id.as_str()));
    let want = reference_swiglu(&weights, &input);
    for (got, expected) in out.iter().zip(&want) {
        assert!((got - expected).abs() < 1e-6, "recovered result wrong");
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// Phase 3 per-peer blocking, on the wire: a node's block list refuses
/// EVERY inbound RPC from the listed peer — honestly (typed error naming
/// the block), while a different requester is served normally by the same
/// server. Blocking is personal, not a node-wide shutdown.
#[tokio::test]
async fn peer_block_refuses_listed_requester_and_serves_others() {
    use jerith_mesh_core::expert_store::{ExpertStore, ExpertWeights};
    use jerith_mesh_core::quic::{ExpertRouter, ExpertServing, RouterOptions, SharedPinMap};
    use jerith_mesh_core::ratelimit::BlockedPeers;
    use std::collections::HashMap;
    use std::sync::Arc;

    // Node B hosts one real expert (so "refused" cannot be confused with
    // "not hosted" — the refusal must name the block).
    let dir = std::env::temp_dir().join(format!("jerith-peerblock-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let store = Arc::new(ExpertStore::new(&dir, 8).expect("store"));
    let weights = ExpertWeights::synthetic("L3.E17", 8, 6);
    store.put(&weights).expect("publish expert");

    // The block list carries the OPERATOR view first: a shorter prefix
    // (what you copy from a log line) must still match the full id.
    let blocked_identity = SigningIdentity::generate();
    let blocked_prefix = &blocked_identity.node_id()[..8];
    let block = BlockedPeers::from_list(&[blocked_prefix.to_string()]);

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
        stats: None,
        isolate: None,
        prefetch: None,
    });
    let (_tx, rx) = tokio::sync::watch::channel(false);
    {
        let node_id = node_id.clone();
        tokio::spawn(async move {
            let _ = jerith_mesh_core::quic::serve_with_peer_block(
                server, serving, node_id, rx, block, None,
            )
            .await;
        });
    }
    let mut pins: jerith_mesh_core::quic::PinMap = HashMap::new();
    pins.insert(node_id.clone(), addr);
    let pins: SharedPinMap = Arc::new(tokio::sync::RwLock::new(pins));
    let client = jerith_mesh_core::quic::pinned_client_endpoint_dyn(pins).expect("client endpoint");

    let router = ExpertRouter::new(RouterOptions {
        nearby_rtt_ms: 50,
        dispatch_timeout: Duration::from_secs(10),
        ..Default::default()
    });
    let mut peer_addrs: HashMap<String, SocketAddr> = HashMap::new();
    peer_addrs.insert(node_id.clone(), addr);
    let peers = vec![(node_id.clone(), 5u64, true, true, None, None)];
    let input = vec![0.5f32, -0.25, 1.0, 0.0, -1.5, 2.0, 0.75, -0.5];

    // --- The BLOCKED requester: prefix-listed, refused by identity -------
    let refused = router
        .execute_over_quic_speculative_attributed(
            &client,
            &blocked_identity,
            &peer_addrs,
            "peerblock-1",
            "L3.E17",
            &input,
            &peers,
            &mut HashMap::new(),
            |_: &[f32]| anyhow::bail!("no local weights here"),
            &mut None,
            jerith_mesh_core::pipeline::RequestBudget::default(),
        )
        .await
        .expect_err("blocked requester must be refused");
    let msg = format!("{refused:#}");
    assert!(
        msg.contains("blocked"),
        "refusal must name the block, not masquerade as 'not hosted': {msg}"
    );
    assert!(
        !msg.contains("not hosted"),
        "a blocked peer is a policy refusal, not a hosting miss: {msg}"
    );

    // --- A DIFFERENT requester: the same server serves it normally -------
    let approved = SigningIdentity::generate();
    let mut served_by: Option<String> = None;
    let out = router
        .execute_over_quic_speculative_attributed(
            &client,
            &approved,
            &peer_addrs,
            "peerblock-2",
            "L3.E17",
            &input,
            &peers,
            &mut HashMap::new(),
            |_: &[f32]| anyhow::bail!("no local weights here"),
            &mut served_by,
            jerith_mesh_core::pipeline::RequestBudget::default(),
        )
        .await
        .expect("non-blocked requester must be served");
    assert_eq!(served_by.as_deref(), Some(node_id.as_str()));
    let want = reference_swiglu(&weights, &input);
    for (got, expected) in out.iter().zip(&want) {
        assert!(
            (got - expected).abs() < 1e-6,
            "wrong result for allowed peer"
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// Independent SwiGLU reference (explicit loops over sliced rows, no
/// shared helpers) — the arithmetic this test holds the wire result to.
fn reference_swiglu(w: &jerith_mesh_core::expert_store::ExpertWeights, x: &[f32]) -> Vec<f32> {
    let act: Vec<f32> = w
        .gate
        .as_ref()
        .unwrap()
        .chunks(w.hidden)
        .zip(w.up.chunks(w.hidden))
        .map(|(gate_row, up_row)| {
            let mut g = 0.0f32;
            let mut u = 0.0f32;
            for (gr, k) in gate_row.iter().zip(x) {
                g += gr * k;
            }
            for (ur, k) in up_row.iter().zip(x) {
                u += ur * k;
            }
            (g / (1.0 + (-g).exp())) * u
        })
        .collect();
    w.down
        .chunks(w.intermediate)
        .map(|down_row| down_row.iter().zip(&act).map(|(d, a)| d * a).sum::<f32>())
        .collect()
}

/// Phase 2 hot-expert replication, end to end over real QUIC: node B holds
/// an expert and advertises it, node A has recorded demand for it (so it is
/// hot) and pulls a replica with one `replication_tick` — after which A
/// serves the same numbers locally, without B in the picture.
#[tokio::test]
async fn hot_expert_replication_pulls_a_replica_over_quic() {
    use jerith_mesh_core::bootstrap::{PeerAd, PeerRegistry};
    use jerith_mesh_core::config::{NodeConfig, NodeRole};
    use jerith_mesh_core::expert_store::{ExpertStore, ExpertWeights};
    use jerith_mesh_core::ping::{MeshFallbackChat, MeshState};
    use jerith_mesh_core::quic::{ExpertHandler, ExpertServing, SharedPinMap};
    use jerith_mesh_core::replication::ReplicationConfig;
    use std::collections::HashMap;
    use std::sync::Arc;

    let dir_b = std::env::temp_dir().join(format!("jerith-repl-src-{}", std::process::id()));
    let dir_a = std::env::temp_dir().join(format!("jerith-repl-dst-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir_b);
    let _ = std::fs::remove_dir_all(&dir_a);

    // --- Node B: holds the expert and advertises exactly that -----------
    let store_b = Arc::new(ExpertStore::new(&dir_b, 8).unwrap());
    let weights = ExpertWeights::synthetic("L3.E17", 8, 6);
    store_b.put(&weights).unwrap();
    let identity_b = SigningIdentity::generate();
    let server =
        jerith_mesh_core::quic::server_endpoint_for(&identity_b, "127.0.0.1:0".parse().unwrap())
            .expect("server endpoint");
    let addr_b = server.local_addr().unwrap();
    let node_b = identity_b.node_id();
    let serving_b: Arc<dyn ExpertHandler> = Arc::new(ExpertServing {
        store: Some(store_b.clone()),
        runtime: None,
        tensors_only: true,
        stats: None,
        isolate: None,
        prefetch: None,
    });
    let (_tx_b, rx_b) = tokio::sync::watch::channel(false);
    {
        let node_b = node_b.clone();
        tokio::spawn(async move {
            let _ = jerith_mesh_core::quic::serve(server, serving_b, node_b, rx_b).await;
        });
    }

    // --- Node A: knows B's ad, has demand for the expert, holds nothing --
    assert!(
        NodeConfig::default().identity().is_err(),
        "default config must not carry an identity until we set one"
    );
    let cfg_a = NodeConfig {
        node_id: "node-a".into(),
        secret_key_hex: Some(SigningIdentity::generate().secret_bytes().to_vec()),
        model: Some(jerith_mesh_core::config::default_model("glm-5.3")),
        role: NodeRole::Hybrid,
        contribution_percent: 40,
        ..NodeConfig::default()
    };
    let (state_a, _rx_a) = MeshState::new(cfg_a).unwrap();
    let store_a = Arc::new(ExpertStore::new(&dir_a, 8).unwrap());
    state_a.set_expert_store(store_a.clone()).await;
    state_a.record_activation("L3.E17");
    assert!(store_a.hosted().is_empty(), "A starts with no weights");

    // B's signed ad, including the expert it serves. The registry is
    // created the way the ping loop creates it (from the configured
    // model), since a registry with no model refuses every ad.
    let (hosted, count) = PeerAd::advertise_hosted_experts(&["L3.E17".to_string()]);
    {
        let mut reg = PeerRegistry::new(Some(jerith_mesh_core::config::default_model("glm-5.3")));
        assert!(
            reg.upsert(PeerAd {
                node_id: node_b.clone(),
                address: format!("{}", addr_b),
                model_id: "glm-5.3".into(),
                quant: "int4".into(),
                contribution_percent: 40,
                last_seen_ms: jerith_mesh_core::bootstrap::now_ms(),
                role: NodeRole::Hybrid,
                rpc_address: addr_b.to_string(),
                resources: None,
                hosted_experts: hosted,
                hosted_expert_count: count,
                discovery_servers: Vec::new(),
                gossip_reputations: None,
            }),
            "a same-model peer ad must pass the model-homogeneous gate"
        );
        reg.record_ping_ok(&node_b, 3);
        *state_a.registry.lock().await = reg;
    }
    // A's chat handle: pins must let the transfer dial B.
    let mut pins: jerith_mesh_core::quic::PinMap = HashMap::new();
    pins.insert(node_b.clone(), addr_b);
    let shared_pins: SharedPinMap = Arc::new(tokio::sync::RwLock::new(pins));
    let fb = Arc::new(MeshFallbackChat {
        endpoint: jerith_mesh_core::quic::pinned_client_endpoint_dyn(shared_pins)
            .expect("client endpoint"),
        router: Arc::new(jerith_mesh_core::quic::ExpertRouter::new(
            jerith_mesh_core::quic::RouterOptions::default(),
        )),
        pins: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
    });

    // --- One replication round ------------------------------------------
    let actions = fb
        .replication_tick(
            &state_a,
            ReplicationConfig {
                hot_percent: 15,
                replication_factor: 3,
                max_actions_per_round: 4,
            },
        )
        .await;
    assert_eq!(actions.len(), 1, "the hot expert must be pulled once");
    assert_eq!(actions[0].expert_id, "L3.E17");
    assert_eq!(actions[0].source_node, node_b);

    // The replica is on A's disk, complete and valid…
    assert_eq!(store_a.hosted(), vec!["L3.E17".to_string()]);
    let replica = store_a.get("L3.E17").expect("replica readable");
    assert_eq!(
        *replica, weights,
        "replicated weights must be bit-identical"
    );

    // …and A now serves the same activations *locally*, with no peer needed.
    let input = vec![0.5f32, -0.25, 1.0, 0.0, -1.5, 2.0, 0.75, -0.5];
    let served_locally = store_a.run("L3.E17", &input).unwrap();
    assert_eq!(served_locally, reference_swiglu(&weights, &input));

    // Demand accounting: A's hot set now names the replicated expert.
    let stats = state_a.expert_stats_snapshot();
    assert!(stats.hot_set(15).contains(&"L3.E17".to_string()));

    // A second round is a no-op: the replica already exists, so the policy
    // does not re-fetch (and the factor still counts B + A).
    let again = fb
        .replication_tick(
            &state_a,
            ReplicationConfig {
                hot_percent: 15,
                replication_factor: 3,
                max_actions_per_round: 4,
            },
        )
        .await;
    assert!(
        again.is_empty(),
        "replication must be idempotent per expert"
    );

    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

/// The cold-shard path: with the expert hot but no advertised holder, the
/// replication policy has no source and does nothing (rather than dialing
/// a peer that would refuse) — and the shard ring still names an owner for
/// placement to try.
#[tokio::test]
async fn cold_shard_ring_names_an_owner_when_nothing_is_advertised() {
    use jerith_mesh_core::bootstrap::{PeerAd, PeerRegistry};
    use jerith_mesh_core::config::NodeRole;
    use jerith_mesh_core::replication::{
        ReplicaPeer, ReplicationConfig, ShardMap, plan_replication,
    };

    let peers = vec![
        ReplicaPeer {
            node_id: "peer-1".into(),
            rtt_ms: 5,
            hosted_experts: Some(vec![]), // advertises holding nothing
            alive: true,
        },
        ReplicaPeer {
            node_id: "peer-2".into(),
            rtt_ms: 9,
            hosted_experts: Some(vec![]),
            alive: true,
        },
    ];
    let actions = plan_replication(
        &["L8.E3".to_string()],
        &[],
        &peers,
        ReplicationConfig {
            hot_percent: 15,
            replication_factor: 3,
            max_actions_per_round: 2,
        },
    );
    assert!(
        actions.is_empty(),
        "no advertised holder ⇒ no fetch source: {actions:?}"
    );

    // The shard ring still answers "who should own it" — deterministically.
    let ads: Vec<PeerAd> = peers
        .iter()
        .map(|p| PeerAd {
            node_id: p.node_id.clone(),
            address: "127.0.0.1:1".into(),
            model_id: "glm-5.3".into(),
            quant: "int4".into(),
            contribution_percent: 40,
            last_seen_ms: 0,
            role: NodeRole::Hybrid,
            rpc_address: "127.0.0.1:9001".into(),
            resources: None,
            hosted_experts: p.hosted_experts.clone(),
            hosted_expert_count: Some(0),
            discovery_servers: Vec::new(),
            gossip_reputations: None,
        })
        .collect();
    let ring_a = ShardMap::from_ads("node-a", &ads);
    let owner = ring_a.primary_owner("L8.E3").expect("ring names an owner");
    assert!(ring_a.members().iter().any(|m| m == &owner));
    assert!(ring_a.owns(&owner, "L8.E3"));
    // Determinism is what lets every node agree without a coordinator:
    // the same membership in a different ad order names the same owner.
    let reversed: Vec<PeerAd> = ads.iter().rev().cloned().collect();
    let ring_again = ShardMap::from_ads("node-a", &reversed);
    assert_eq!(
        ring_again.primary_owner("L8.E3").as_deref(),
        Some(owner.as_str())
    );
    // A different node computes its *own* membership (each node adds
    // itself), so its ring must contain whoever it names — and both rings
    // agree the two peers are in the mesh.
    let ring_b = ShardMap::from_ads("node-b", &ads);
    let owner_b = ring_b.primary_owner("L8.E3").expect("ring names an owner");
    assert!(ring_b.members().iter().any(|m| m == &owner_b));
    assert_eq!(ring_a.members().len(), 3);
    assert_eq!(ring_b.members().len(), 3);

    // Registry sanity: with the model gate satisfied both ads are stored,
    // so the membership the ring uses matches what a node actually knows.
    let mut reg = PeerRegistry::new(Some(jerith_mesh_core::config::default_model("glm-5.3")));
    for ad in &ads {
        assert!(reg.upsert(ad.clone()), "ad must be admitted");
    }
    assert_eq!(reg.all_ads().len(), 2);
    assert_eq!(reg.all_ads().len() + 1, ring_a.members().len());
}

/// The replication *loop*, not one decision round: `ReplicaManager` must
/// stay alive on its interval and pull the hot expert by itself. The node
/// binds the manager for its whole run — a dropped manager ends the loop,
/// which is exactly the bug this test exists for (it is silent from the
/// outside, since a node that never replicates looks like one with nothing
/// to replicate).
#[tokio::test]
async fn replica_manager_pulls_a_hot_expert_on_its_own_interval() {
    use jerith_mesh_core::bootstrap::{PeerAd, PeerRegistry};
    use jerith_mesh_core::config::{NodeConfig, NodeRole};
    use jerith_mesh_core::expert_stats::ExpertStats;
    use jerith_mesh_core::expert_store::{ExpertStore, ExpertWeights};
    use jerith_mesh_core::ping::{MeshFallbackChat, MeshState, ReplicaManager};
    use jerith_mesh_core::quic::{ExpertHandler, ExpertServing};
    use std::sync::Arc;

    let dir_b = std::env::temp_dir().join(format!("jerith-loop-src-{}", std::process::id()));
    let dir_a = std::env::temp_dir().join(format!("jerith-loop-dst-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir_b);
    let _ = std::fs::remove_dir_all(&dir_a);

    // --- Node B: holds the expert and serves it over QUIC ---------------
    let store_b = Arc::new(ExpertStore::new(&dir_b, 8).unwrap());
    let weights = ExpertWeights::synthetic("L3.E17", 8, 6);
    store_b.put(&weights).unwrap();
    let identity_b = SigningIdentity::generate();
    let node_b = identity_b.node_id();
    let server =
        jerith_mesh_core::quic::server_endpoint_for(&identity_b, "127.0.0.1:0".parse().unwrap())
            .expect("server endpoint");
    let addr_b = server.local_addr().unwrap();
    let serving_b: Arc<dyn ExpertHandler> = Arc::new(ExpertServing {
        store: Some(store_b.clone()),
        runtime: None,
        tensors_only: true,
        stats: None,
        isolate: None,
        prefetch: None,
    });
    let (_tx_b, rx_b) = tokio::sync::watch::channel(false);
    {
        let node_b = node_b.clone();
        tokio::spawn(async move {
            let _ = jerith_mesh_core::quic::serve(server, serving_b, node_b, rx_b).await;
        });
    }

    // --- Node A: demand for the expert, B in the registry, no weights ---
    let cfg_a = NodeConfig {
        node_id: "node-a".into(),
        secret_key_hex: Some(SigningIdentity::generate().secret_bytes().to_vec()),
        model: Some(jerith_mesh_core::config::default_model("glm-5.3")),
        role: NodeRole::Hybrid,
        contribution_percent: 40,
        ..NodeConfig::default()
    };
    let (state_a, _rx_a) = MeshState::new(cfg_a).unwrap();
    let store_a = Arc::new(ExpertStore::new(&dir_a, 8).unwrap());
    state_a.set_expert_store(store_a.clone()).await;
    // This run's counters and their file are its own: a test must never
    // write over the live node's `<data>/expert-stats.json`.
    let stats_path = dir_a.join("expert-stats.json");
    let mut seed = ExpertStats::new();
    seed.record("L3.E17");
    state_a.set_expert_stats(seed, stats_path.clone());
    let (hosted, count) = PeerAd::advertise_hosted_experts(&["L3.E17".to_string()]);
    {
        let mut reg = PeerRegistry::new(Some(jerith_mesh_core::config::default_model("glm-5.3")));
        assert!(reg.upsert(PeerAd {
            node_id: node_b.clone(),
            address: format!("{addr_b}"),
            model_id: "glm-5.3".into(),
            quant: "int4".into(),
            contribution_percent: 40,
            last_seen_ms: jerith_mesh_core::bootstrap::now_ms(),
            role: NodeRole::Hybrid,
            rpc_address: addr_b.to_string(),
            resources: None,
            hosted_experts: hosted,
            hosted_expert_count: count,
            discovery_servers: Vec::new(),
            gossip_reputations: None,
        }));
        reg.record_ping_ok(&node_b, 3);
        *state_a.registry.lock().await = reg;
    }

    // The loop, wired exactly as the node wires it (its own pins, refreshed
    // from the registry before dialing).
    let fb = MeshFallbackChat::spawn().unwrap();
    let manager = ReplicaManager::spawn(
        state_a.clone(),
        fb,
        Duration::from_millis(200),
        jerith_mesh_core::replication::ReplicationConfig {
            hot_percent: 15,
            replication_factor: 3,
            max_actions_per_round: 2,
        },
    );

    // Real-QUIC under a loaded CI runner: one round's transfer can sit in
    // a starved handshake far longer than the 200 ms interval (the fetch's
    // own connect/transfer timeouts are 60 s) and the round after it waits
    // on the same single-threaded loop, so the pull may legitimately land
    // well past this test's old 10 s envelope. The envelope widens to sit
    // under that 60 s ceiling with room for one slow round followed by a
    // fast retry — the assertion itself never changes: the loop must
    // still land the complete, bit-identical replica.
    let mut waited = 0u64;
    while store_a.hosted().is_empty() && waited < 30_000 {
        tokio::time::sleep(Duration::from_millis(50)).await;
        waited += 50;
    }
    assert_eq!(
        store_a.hosted(),
        vec!["L3.E17".to_string()],
        "the loop itself must pull the hot expert within {waited} ms"
    );
    let replica = store_a.get("L3.E17").expect("replica readable");
    assert_eq!(
        *replica, weights,
        "replicated weights must be bit-identical"
    );
    manager.shutdown().await;

    // The demand signal is on disk in this run's own file.
    let persisted = std::fs::read_to_string(&stats_path).expect("demand signal persisted");
    assert!(persisted.contains("L3.E17"), "{persisted}");

    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

/// Counters persist even when nothing replicates (a lone node, or a mesh
/// where every observed expert is cold and unhosted): replication decisions
/// that forget yesterday's traffic would oscillate, so the loop flushes
/// changed counts on their own.
#[tokio::test]
async fn replica_manager_persists_demand_without_any_fetch() {
    use jerith_mesh_core::config::{NodeConfig, NodeRole};
    use jerith_mesh_core::expert_stats::ExpertStats;
    use jerith_mesh_core::expert_store::ExpertStore;
    use jerith_mesh_core::ping::{MeshFallbackChat, MeshState, ReplicaManager};
    use std::sync::Arc;

    let dir = std::env::temp_dir().join(format!("jerith-loop-solo-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let cfg = NodeConfig {
        node_id: "node-solo".into(),
        secret_key_hex: Some(SigningIdentity::generate().secret_bytes().to_vec()),
        role: NodeRole::Hybrid,
        contribution_percent: 40,
        ..NodeConfig::default()
    };
    let (state, _rx) = MeshState::new(cfg).unwrap();
    state
        .set_expert_store(Arc::new(ExpertStore::new(&dir, 8).unwrap()))
        .await;
    let stats_path = dir.join("expert-stats.json");
    let mut seed = ExpertStats::new();
    for _ in 0..3 {
        seed.record("nobody.hosts.this");
    }
    state.set_expert_stats(seed, stats_path.clone());

    let fb = MeshFallbackChat::spawn().unwrap();
    let manager = ReplicaManager::spawn(
        state.clone(),
        fb,
        Duration::from_millis(100),
        jerith_mesh_core::replication::ReplicationConfig::default(),
    );
    // Demand arrives while the loop runs (no peers, no holder anywhere).
    state.record_activation("nobody.hosts.this");

    let mut waited = 0u64;
    while !stats_path.exists() && waited < 5_000 {
        tokio::time::sleep(Duration::from_millis(50)).await;
        waited += 50;
    }
    assert!(
        stats_path.exists(),
        "counts must persist even when nothing replicates"
    );
    let reloaded = ExpertStats::load(&stats_path);
    assert_eq!(reloaded.count("nobody.hosts.this"), 4);
    manager.shutdown().await;

    let _ = std::fs::remove_dir_all(&dir);
}

/// Phase 5 escrow join gate, on the wire: with `[crypto.stake]
/// required_uj` set and an unprimed ledger, work-carrying requests
/// (ExpertQuery) get the honest typed refusal naming the stake numbers
/// (counted for /status) while Ping keeps flowing — a live-but-
/// unqualified node, never a vanished one. Once the escrow lands
/// (ledger primed to the line), the same request serves numerically
/// exact. The gate flips on ledger truth only; reputation (§10's
/// placement demotion) stays statistics — this is the escrow half of
/// the interlock, the only refusal path.
#[tokio::test]
async fn stake_gate_refuses_work_until_the_escrow_lands() {
    use jerith_mesh_core::expert_store::{ExpertStore, ExpertWeights};
    use jerith_mesh_core::quic::{
        ExpertRouter, ExpertServing, NodeRpcReply, NodeRpcRequest, RouterOptions, SharedPinMap,
        call, pinned_client_endpoint_dyn, server_endpoint_for,
    };
    use std::collections::HashMap;
    use std::sync::Arc;

    let dir = std::env::temp_dir().join(format!("jerith-stakegate-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let store = Arc::new(ExpertStore::new(&dir, 8).expect("store"));
    let weights = ExpertWeights::synthetic("L3.E17", 8, 6);
    store.put(&weights).expect("publish expert");

    // One gate server per phase: staking ON (1000 µJ line) with an
    // empty (absent-file) or primed ledger, hosting the real expert so
    // "refused" can never be confused with "not hosted".
    async fn gate_server(
        dir: &std::path::Path,
        store: Arc<ExpertStore>,
        primed_uj: Option<u64>,
    ) -> anyhow::Result<(
        std::net::SocketAddr,
        String,
        Arc<jerith_mesh_core::stake::StakeManager>,
        SharedPinMap,
    )> {
        let crypto = jerith_mesh_core::config::CryptoConfig {
            stake: jerith_mesh_core::stake::StakeConfig {
                required_uj: 1_000,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut stake = jerith_mesh_core::stake::StakeManager::new(&crypto, None, None);
        let ledger_path = dir.join(format!(
            "stake-{}.json",
            primed_uj
                .map(|v| v.to_string())
                .unwrap_or_else(|| "empty".into())
        ));
        if let Some(staked) = primed_uj {
            std::fs::write(
                &ledger_path,
                serde_json::json!({
                    "staked_uj": staked, "slashed_total_uj": 0,
                    "escrowed_uj": 0, "slash_events": []
                })
                .to_string(),
            )
            .expect("prime ledger");
        }
        Arc::get_mut(&mut stake)
            .expect("manager unshared at construction time")
            .rearm_ledger(ledger_path);

        let identity = SigningIdentity::generate();
        let server = server_endpoint_for(&identity, "127.0.0.1:0".parse().unwrap())?;
        let addr = server.local_addr().unwrap();
        let node_id = identity.node_id();
        let serving = Arc::new(ExpertServing {
            store: Some(store),
            runtime: None,
            tensors_only: true,
            stats: None,
            isolate: None,
            prefetch: None,
        });
        let (_tx, rx) = tokio::sync::watch::channel(false);
        let block = jerith_mesh_core::ratelimit::BlockedPeers::from_list(&[] as &[String]);
        let st = stake.clone();
        let nid = node_id.clone();
        tokio::spawn(async move {
            let _ = jerith_mesh_core::quic::serve_with_peer_block(
                server,
                serving,
                nid,
                rx,
                block,
                Some(st),
            )
            .await;
        });
        let mut pins: jerith_mesh_core::quic::PinMap = HashMap::new();
        pins.insert(node_id.clone(), addr);
        let pins: SharedPinMap = Arc::new(tokio::sync::RwLock::new(pins));
        Ok((addr, node_id, stake, pins))
    }

    let router = ExpertRouter::new(RouterOptions {
        nearby_rtt_ms: 50,
        dispatch_timeout: Duration::from_secs(10),
        ..Default::default()
    });
    let input = vec![0.5f32, -0.25, 1.0, 0.0, -1.5, 2.0, 0.75, -0.5];

    // ---- Phase 1: empty ledger — work refused, ping answered -----------
    let (addr, node_id, stake, pins) = gate_server(&dir, store.clone(), None).await.unwrap();
    let client = pinned_client_endpoint_dyn(pins).expect("client endpoint");
    let mut peer_addrs: HashMap<String, std::net::SocketAddr> = HashMap::new();
    peer_addrs.insert(node_id.clone(), addr);
    let peers = vec![(node_id.clone(), 5u64, true, true, None, None)];

    let refused = router
        .execute_over_quic_speculative_attributed(
            &client,
            &SigningIdentity::generate(),
            &peer_addrs,
            "stakegate-1",
            "L3.E17",
            &input,
            &peers,
            &mut HashMap::new(),
            |_: &[f32]| anyhow::bail!("no local weights here"),
            &mut None,
            jerith_mesh_core::pipeline::RequestBudget::default(),
        )
        .await
        .expect_err("the gate must refuse with an empty escrow");
    let msg = format!("{refused:#}");
    assert!(
        msg.contains("stake join gate not clear"),
        "refusal must name the gate, not masquerade as 'not hosted': {msg}"
    );
    assert!(
        msg.contains("1000 µJ"),
        "refusal must carry the required line and the staked number: {msg}"
    );
    assert_eq!(
        stake.gate_refusals(),
        1,
        "the refusal is counted for /status"
    );

    // Ping keeps flowing: the mesh watches a LIVE-but-unqualified node.
    let me = SigningIdentity::generate();
    let mut ping = NodeRpcRequest::Ping {
        from_node_id: me.node_id(),
        stamp_ms: 0,
        sig: String::new(),
    };
    jerith_mesh_core::quic::sign_request(&mut ping, &me);
    let reply = call(&client, addr, &node_id, &ping, &me, Duration::from_secs(5))
        .await
        .expect("ping survives the gate");
    assert!(
        matches!(reply, NodeRpcReply::Pong { .. }),
        "pings must answer while the gate is closed: {reply:?}"
    );

    // ---- Phase 2: the escrow lands — the same request serves -----------
    let (addr_b, node_id_b, _stake_b, pins_b) =
        gate_server(&dir, store.clone(), Some(1_000)).await.unwrap();
    let client_b = pinned_client_endpoint_dyn(pins_b).expect("client endpoint");
    let mut peer_addrs_b: HashMap<String, std::net::SocketAddr> = HashMap::new();
    peer_addrs_b.insert(node_id_b.clone(), addr_b);
    let peers_b = vec![(node_id_b.clone(), 5u64, true, true, None, None)];

    let mut served_by: Option<String> = None;
    let out = router
        .execute_over_quic_speculative_attributed(
            &client_b,
            &me,
            &peer_addrs_b,
            "stakegate-2",
            "L3.E17",
            &input,
            &peers_b,
            &mut HashMap::new(),
            |_: &[f32]| anyhow::bail!("no local weights here"),
            &mut served_by,
            jerith_mesh_core::pipeline::RequestBudget::default(),
        )
        .await
        .expect("a primed escrow opens the gate");
    assert_eq!(served_by.as_deref(), Some(node_id_b.as_str()));
    let want = reference_swiglu(&weights, &input);
    for (got, expected) in out.iter().zip(&want) {
        assert!((got - expected).abs() < 1e-6, "served result wrong");
    }

    let _ = client;
    let _ = client_b;
    std::fs::remove_dir_all(&dir).ok();
}

/// Phase 3 speculative prefetch over the real QUIC serving path
/// (FEATURES §13): node B serves `L0.E1` from its store; a background
/// walk must then preload B's hosted, demand-ranked `L1.*` experts into
/// the LRU — with no extra wire request — while the serving path's own
/// counters stay honest (prefetch books its own counters).
#[tokio::test]
async fn prefetch_after_serve_preloads_the_next_layer_over_quic() {
    use jerith_mesh_core::config::{NodeConfig, NodeRole};
    use jerith_mesh_core::expert_store::{ExpertStore, ExpertWeights};
    use jerith_mesh_core::ping::MeshState;
    use jerith_mesh_core::quic::{ExpertHandler, ExpertRouter, ExpertServing, RouterOptions};
    use jerith_mesh_core::signing::SigningIdentity;
    use std::collections::HashMap;
    use std::sync::Arc;

    let dir = std::env::temp_dir().join(format!("jerith-prefetch-quic-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let store = Arc::new(ExpertStore::new(&dir, 8).expect("store"));
    store.put(&ExpertWeights::synthetic("L0.E1", 8, 6)).unwrap();
    store.put(&ExpertWeights::synthetic("L1.E1", 8, 6)).unwrap();
    store.put(&ExpertWeights::synthetic("L1.E2", 8, 6)).unwrap();
    store.put(&ExpertWeights::synthetic("L9.E9", 8, 6)).unwrap();

    // Node B's state: a real MeshState so the serve-side hook is the one
    // production uses. Demand history: L1.E1 asked twice, L1.E2 once —
    // the walk must rank them in that order and never touch L9.E9.
    let cfg = NodeConfig {
        node_id: "prefetch-node".into(),
        secret_key_hex: Some(SigningIdentity::generate().secret_bytes().to_vec()),
        model: Some(jerith_mesh_core::config::default_model("glm-5.3")),
        role: NodeRole::Hybrid,
        contribution_percent: 40,
        ..NodeConfig::default()
    };
    let (state, _rx) = MeshState::new(cfg).unwrap();
    state.set_expert_store(store.clone()).await;
    let stats_path = std::env::temp_dir().join(format!(
        "jerith-prefetch-quic-stats-{}.json",
        std::process::id()
    ));
    let stats = {
        let mut s = jerith_mesh_core::expert_stats::ExpertStats::new();
        s.record("L1.E1");
        s.record("L1.E1");
        s.record("L1.E2");
        s
    };
    state.set_expert_stats(stats, stats_path);

    let identity = SigningIdentity::generate();
    let server =
        jerith_mesh_core::quic::server_endpoint_for(&identity, "127.0.0.1:0".parse().unwrap())
            .expect("server endpoint");
    let addr = server.local_addr().unwrap();
    let serving: Arc<dyn ExpertHandler> = Arc::new(ExpertServing {
        store: Some(store.clone()),
        runtime: None,
        tensors_only: true,
        stats: Some(state.expert_stats.clone()),
        isolate: None,
        prefetch: Some(state.clone()),
    });
    let (_tx, rx) = tokio::sync::watch::channel(false);
    {
        let node_id = identity.node_id();
        tokio::spawn(async move {
            let _ = jerith_mesh_core::quic::serve(server, serving, node_id, rx).await;
        });
    }

    // A peer's router dispatches one real expert query to B — the same
    // helper every other drill uses, so the serve fast-path (store hit →
    // run → stats.record → spawn prefetch) is the production one.
    let mut pins: HashMap<String, SocketAddr> = HashMap::new();
    pins.insert(identity.node_id(), addr);
    let pins: Arc<tokio::sync::RwLock<HashMap<String, SocketAddr>>> =
        Arc::new(tokio::sync::RwLock::new(pins));
    let client = jerith_mesh_core::quic::pinned_client_endpoint_dyn(pins).expect("client endpoint");
    let router = ExpertRouter::new(RouterOptions {
        nearby_rtt_ms: 50,
        dispatch_timeout: Duration::from_secs(10),
        ..Default::default()
    });
    let requester = SigningIdentity::generate();
    let mut peer_addrs: HashMap<String, SocketAddr> = HashMap::new();
    peer_addrs.insert(identity.node_id(), addr);
    // B is a live candidate holding the expert (unreported resource
    // sample and no reputation score — the pre-pooling shape).
    let peers = vec![(identity.node_id(), 5u64, true, true, None, None)];
    let input = vec![0.5f32, -0.25, 1.0, 0.0, -1.5, 2.0, 0.75, -0.5];
    let mut completed = HashMap::new();
    let mut served_by = None;
    let out = router
        .execute_over_quic_speculative_attributed(
            &client,
            &requester,
            &peer_addrs,
            "prefetch-1",
            "L0.E1",
            &input,
            &peers,
            &mut completed,
            move |_: &[f32]| anyhow::bail!("requester holds no local weights"),
            &mut served_by,
            jerith_mesh_core::pipeline::RequestBudget::default(),
        )
        .await
        .expect("expert query served over the wire");
    assert_eq!(served_by.as_deref(), Some(identity.node_id().as_str()));
    assert!(!out.is_empty());

    // The prefetch walk is spawned, not awaited; PROVE the outcome on
    // the store instead of sleeping a fixed beat: a loaded CI runner can
    // starve the spawned walk well past any fixed delay (the same
    // loaded-runner family the other timing envelopes tolerate). Poll
    // until the LAST ranked prefetch (L1.E2) is resident — that proves
    // the cap-firing walk got to its second action — then assert the
    // full bundle.
    let mut waited = 0u64;
    while !store.is_resident("L1.E2") && waited < 10_000 {
        tokio::time::sleep(Duration::from_millis(50)).await;
        waited += 50;
    }
    let s = store.stats();
    assert_eq!(s.prefetches, 2, "L1.E1 then L1.E2 preloaded (demand-rank)");
    assert!(store.is_resident("L1.E1"));
    assert!(store.is_resident("L1.E2"));
    // The observed-but-unhosted expert never existed here; nothing with
    // zero demand was touched.
    assert!(!store.is_resident("L9.E9"));
    // The serving path's own counters stay honest: exactly one miss (the
    // cold L0.E1 the wire query itself paid for), zero load errors.
    assert_eq!(s.misses, 1);
    assert_eq!(s.load_errors, 0);
    // And the walk released its one-at-a-time guard.
    assert!(
        !state.prefetching.load(std::sync::atomic::Ordering::SeqCst),
        "the in-flight guard must clear after the walk"
    );
    let _ = store.hosted();
    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_file(std::env::temp_dir().join(format!(
        "jerith-prefetch-quic-stats-{}.json",
        std::process::id()
    )))
    .ok();
}
