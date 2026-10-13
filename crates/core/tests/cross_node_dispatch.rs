//! Live cross-node expert dispatch E2E.
//!
//! Requires two running nodes (as in the dev E2E):
//!   A: gateway 127.0.0.1:8900, rpc UDP 9000, node B advertised in registry
//!   B: gateway 127.0.0.1:8910, rpc UDP 8911
//! Runs only with `--ignored --nocapture`. Skips gracefully when the
//! environment isn't up, so normal `cargo test` stays green offline.

use std::net::SocketAddr;
use std::time::Duration;

use jerith_mesh_core::quic::{
    pinned_client_endpoint, sign_request, NodeRpcReply, NodeRpcRequest, PinMap,
};
use jerith_mesh_core::signing::SigningIdentity;

#[tokio::test]
#[ignore = "requires two running mesh nodes (dev E2E)"]
async fn dispatches_to_peer_b_runtime_and_verifies_attribution() {
    let identity = SigningIdentity::generate();

    // Discover B's runtime-backed handler directly at its advertised RPC
    // address — the integration under test is the wire round-trip.
    let addr: SocketAddr = "127.0.0.1:8911".parse().unwrap();

    // Cert pinning fails closed: the dial is pinned to node B's identity via
    // EXPECT_NODE_B_ID, so the reply can only come from the node we meant.
    let sni_node = std::env::var("EXPECT_NODE_B_ID")
        .ok()
        .filter(|s| !s.is_empty())
        .expect("dev E2E: EXPECT_NODE_B_ID must carry node B's node_id");
    let mut pins: PinMap = std::collections::HashMap::new();
    pins.insert(sni_node.clone(), addr);
    let client =
        pinned_client_endpoint(std::sync::Arc::new(pins)).expect("pinned client endpoint");

    let mut req = NodeRpcRequest::ExpertQuery {
        req_id: "e2e-cross-1".into(),
        expert_id: "exp.router.test".into(),
        input: vec![1.5, -2.0, 0.0, 3.0],
        from_node_id: identity.node_id(),
        stamp_ms: 0,
        sig: String::new(),
    };
    sign_request(&mut req, &identity);

    let reply = tokio::time::timeout(
        Duration::from_secs(10),
        jerith_mesh_core::quic::call(&client, addr, &sni_node, &req, &identity, Duration::from_secs(5)),
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
            // Attribution: the server is node B; its advertised node id is
            // fixed in the dev E2E (config /tmp/jq-b).
            assert!(
                !node_id.is_empty() && node_id != "unknown",
                "server must attribute the reply, got {node_id}"
            );
            assert_eq!(
                node_id,
                std::env::var("EXPECT_NODE_B_ID").unwrap_or_else(|_| node_id.clone())
            );
            // Output is the runtime's completion hashed to the input dim —
            // non-empty and stable across a repeat call.
            assert!(!output.is_empty());
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
                    assert_eq!(out2, output, "runtime handler must be deterministic");
                }
                other => panic!("unexpected reply: {other:?}"),
            }
            // Phase 4: the reply must carry a commitment that verifies
            // against the exact body it was returned with.
            let c = commitment
                .as_deref()
                .expect("ExpertResult must carry a Phase 4 commitment");
            assert!(
                jerith_mesh_core::verify::verify_commitment(
                    &req_id,
                    "exp.router.test",
                    &node_id,
                    &[1.5, -2.0, 0.0, 3.0],
                    &output,
                    c
                ),
                "commitment must bind req_id/expert_id/node_id/input/output"
            );
        }
        NodeRpcReply::Error { message, .. } => panic!("peer error: {message}"),
        other => panic!("unexpected reply: {other:?}"),
    }

}
