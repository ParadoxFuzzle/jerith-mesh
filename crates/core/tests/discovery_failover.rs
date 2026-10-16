//! Discovery failover against REAL mock bootstraps (Phase 1).
//!
//! The unit tests prove `discovery::try_each` walks an order; this file
//! proves the live leg — `BootstrapClient::fetch_peers_signed` driven
//! through `try_each` the same way `ping::run_loop` does — transparently
//! fails over to a second, ALIVE mocked bootstrap, returns that server's
//! peer list along with its URL, and that the last-good bookkeeping
//! hoists it into the next call's rotation order.

use std::net::SocketAddr;

use jerith_mesh_core::bootstrap::{BootstrapClient, PeerAd};
use jerith_mesh_core::discovery::{DiscoveryServers, try_each};
use jerith_mesh_core::signing::SigningIdentity;
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

const MODEL: &str = "glm-5.3";

/// A minimal signed-ad-JSON peer list the mock bootstrap serves.
fn peers_json(node_id: &str) -> serde_json::Value {
    let ad: PeerAd = serde_json::from_value(serde_json::json!({
        "node_id": node_id,
        "address": format!("127.0.0.1:{}", 9000 + node_id.len()),
        "model_id": MODEL,
        "quant": "int4",
        "contribution_percent": 30,
        "last_seen_ms": 1,
        "role": "hybrid",
        "rpc_address": "",
    }))
    .expect("mock ad parses");
    serde_json::json!({ "peers": [ad] })
}

/// One-shot mock bootstrap: binds a free port, serves
/// `GET /v1/peers?model=…` with the given JSON on every connection, and
/// keeps the listener alive for the test's duration. Hand-rolled on
/// tokio rather than axum: the response is constant and the repository's
/// dependency discipline favors small surface areas in tests.
async fn spawn_mock(body: serde_json::Value) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let body = serde_json::to_vec(&body).unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                break;
            };
            let body = body.clone();
            tokio::spawn(async move {
                let mut buf = [0u8; 4096];
                let read = sock.read(&mut buf).await.unwrap_or(0);
                if read == 0 {
                    return;
                }
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = sock.write_all(resp.as_bytes()).await;
                let _ = sock.write_all(&body).await;
                let _ = sock.flush().await;
            });
        }
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    addr
}

/// Bind a free port and close the listener → connections to it are
/// refused, standing in for a dead bootstrap without a second process.
async fn dead_port() -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    addr
}

#[tokio::test]
async fn fetch_leg_fails_over_from_dead_to_alive_bootstrap() {
    let live = spawn_mock(peers_json("peer-from-mock")).await;
    let dead = dead_port().await;
    let order = vec![format!("http://{dead}"), format!("http://{live}")];

    let identity = SigningIdentity::generate();
    let outcome = try_each(&order, |url| {
        let client = BootstrapClient::new(url.to_string());
        let identity = &identity;
        async move { client.fetch_peers_signed(MODEL, identity).await }
    })
    .await
    .expect("the second leg answers");

    let (peers, used) = outcome;
    assert_eq!(used, format!("http://{live}"));
    assert_eq!(peers.len(), 1);
    assert_eq!(peers[0].node_id, "peer-from-mock");
}

#[tokio::test]
async fn last_good_stickiness_reorders_after_a_failover() {
    let live = spawn_mock(peers_json("p")).await;
    let dead = dead_port().await;

    let mut d = DiscoveryServers::default();
    d.configured = vec![format!("http://{dead}"), format!("http://{live}")];

    // Tick 1: primary dead → the alive server answers.
    let order_1 = d.order(None);
    assert_eq!(&order_1[0], &format!("http://{dead}"));
    let identity = SigningIdentity::generate();
    let (_, used) = try_each(&order_1, |url| {
        let client = BootstrapClient::new(url.to_string());
        let identity = &identity;
        async move { client.fetch_peers_signed(MODEL, identity).await }
    })
    .await
    .expect("failover succeeds");
    d.note_good(&used);

    // Tick 2: the server that answered now goes FIRST (no repeat visit
    // to the dead primary), so this call succeeds on the first leg.
    let order_2 = d.order(None);
    assert_eq!(&order_2[0], &format!("http://{live}"));
    let (_peers, used_2) = try_each(&order_2, |url| {
        let client = BootstrapClient::new(url.to_string());
        let identity = &identity;
        async move { client.fetch_peers_signed(MODEL, identity).await }
    })
    .await
    .expect("sticky order starts with the good server");
    assert_eq!(used_2, used, "the good server is used again directly");
}
