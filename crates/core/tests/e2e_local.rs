//! Self-contained mesh chat E2E: a real QUIC peer server driven through the
//! same `MeshFallbackChat` path the gateway handler uses. `cargo test` runs
//! it normally — no manually-started nodes required.
//!
//! Under test: dead local runtime → QUIC fallback → reply with content +
//! `mesh_fallback` = the peer id that actually answered (true attribution).

use std::sync::Arc;

use serde_json::{Value, json};

use jerith_mesh_core::bootstrap::{PeerAd, PeerRegistry};
use jerith_mesh_core::config::{NodeConfig, NodeRole};
use jerith_mesh_core::ping::{MeshFallbackChat, MeshState};
use jerith_mesh_core::quic::{
    ExpertHandler, ExpertRouter, RouterOptions, serve, server_endpoint_for,
};
use jerith_mesh_core::signing::SigningIdentity;

/// Peer-side handler: deterministic chat answer, real identity + signing.
struct PeerChat {
    content: String,
}

impl ExpertHandler for PeerChat {
    fn run(&self, _expert_id: &str, _input: &[f32]) -> anyhow::Result<Vec<f32>> {
        anyhow::bail!("no experts in this test peer")
    }

    fn chat<'a>(
        &'a self,
        _messages: &'a [Value],
    ) -> futures_core::future::BoxFuture<'a, anyhow::Result<String>> {
        let content = self.content.clone();
        Box::pin(async move { Ok(content) })
    }
}

#[tokio::test]
async fn mesh_chat_falls_back_over_quic_with_real_attribution() {
    // 1. Spawn a real QUIC peer serving a chat-capable handler.
    let peer_identity = SigningIdentity::generate();
    let server = server_endpoint_for(&peer_identity, "127.0.0.1:0".parse().unwrap()).unwrap();
    let peer_addr = server.local_addr().unwrap();
    let (_tx, rx) = tokio::sync::watch::channel(false);
    let peer_node_id = peer_identity.node_id();
    let peer_id_for_registry = peer_node_id.clone();
    tokio::spawn(async move {
        serve(
            server,
            Arc::new(PeerChat {
                content: "mesh answer".into(),
            }),
            peer_node_id.clone(),
            rx,
        )
        .await
    });
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // 2. Build the fallback handle + registry pointing at the peer.
    let mut cfg = NodeConfig::default();
    cfg.runtime_url = "http://127.0.0.1:1".to_string(); // deliberately dead
    cfg.ensure_identity().unwrap();
    let (state, _rx2) = MeshState::new(cfg).unwrap();
    {
        let mut reg = PeerRegistry::new(Some(jerith_mesh_core::models::ModelId {
            id: "m".into(),
            quant: "q".into(),
            source: jerith_mesh_core::models::ModelSource::Curated,
        }));
        reg.upsert(PeerAd {
            node_id: peer_id_for_registry.clone(),
            address: format!("http://127.0.0.1:{}", peer_addr.port() + 1),
            model_id: "m".into(),
            quant: "q".into(),
            contribution_percent: 50,
            last_seen_ms: jerith_mesh_core::bootstrap::now_ms(),
            role: NodeRole::Hybrid,
            rpc_address: peer_addr.to_string(),
        });
        reg.record_ping_ok(&peer_id_for_registry, 5);
        *state.registry.lock().await = reg;
    }
    {
        let router = Arc::new(ExpertRouter::new(RouterOptions {
            nearby_rtt_ms: 50,
            dispatch_timeout: std::time::Duration::from_secs(30),
            ..Default::default()
        }));
        let _ = router; // fb owns its own router internally
    }
    let fb = MeshFallbackChat::spawn().unwrap();

    // 3. Drive the exact gateway code path.
    let messages = vec![json!({"role": "user", "content": "hi"})];
    let (content, served_by) = fb
        .chat_with_fallback(&state, &messages, || Err(anyhow::anyhow!("local down")))
        .await
        .expect("fallback should succeed");

    assert_eq!(content, "mesh answer");
    // True attribution: the exact peer that answered, not an RTT-order guess.
    assert_eq!(served_by.as_deref(), Some(peer_id_for_registry.as_str()));
}
