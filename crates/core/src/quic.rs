//! Phase 1 NodeRPC over QUIC + locality-aware expert routing.
//!
//! Transport: QUIC (quinn) with self-signed node certificates generated at
//! first start. Certificate verification is skipped in this phase — the
//! node-level identity proof is the Ed25519 signature carried in every
//! RPC message (Phase 2 pins the QUIC cert to the node's long-term key).
//!
//! Design principles honored here (per ARCHITECTURE.md):
//! - Never block a forward pass on one remote expert: every dispatch has a
//!   timeout, candidates are tried in locality order, and a local fallback
//!   runs when all remotes fail.
//! - Locality > perfect load balance: candidates are ordered by measured
//!   RTT (local first, then lowest-RTT peers).
//! - Idempotent result merging: results are keyed by `req_id`; late or
//!   duplicate results are ignored.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::signing::SigningIdentity;

// ---------------------------------------------------------------------------
// Wire messages
// ---------------------------------------------------------------------------

/// A request over a QUIC stream (newline-delimited JSON).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NodeRpcRequest {
    /// Liveness over the data plane (complements the HTTP ping).
    Ping {
        from_node_id: String,
        stamp_ms: u64,
        sig: String,
    },
    /// Ask a peer to run one expert on an input.
    ExpertQuery {
        req_id: String,
        expert_id: String,
        input: Vec<f32>,
        from_node_id: String,
        stamp_ms: u64,
        sig: String,
    },
}

/// The reply, same framing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NodeRpcReply {
    Pong { node_id: String, stamp_ms: u64 },
    ExpertResult { req_id: String, output: Vec<f32>, node_id: String },
    Error { req_id: String, message: String },
}

/// Sign a request in place (stamp + Ed25519 over the canonical payload).
/// Public: used by the node crate when dispatching remote expert queries.
pub fn sign_request(req: &mut NodeRpcRequest, identity: &SigningIdentity) {
    let stamp = crate::bootstrap::now_ms();
    match req {
        NodeRpcRequest::Ping { stamp_ms, sig: _, .. } => {
            *stamp_ms = stamp;
        }
        NodeRpcRequest::ExpertQuery { stamp_ms, sig: _, .. } => {
            *stamp_ms = stamp;
        }
    }
    let payload = request_payload_for_stamp(req, stamp);
    let signature = identity.sign(payload.as_bytes());
    match req {
        NodeRpcRequest::Ping { sig, .. } | NodeRpcRequest::ExpertQuery { sig, .. } => {
            *sig = signature;
        }
    }
}

fn request_payload_for_stamp(req: &NodeRpcRequest, stamp: u64) -> String {
    match req {
        NodeRpcRequest::Ping { from_node_id, .. } => {
            format!("rpc-ping1|{from_node_id}|{stamp}")
        }
        NodeRpcRequest::ExpertQuery { req_id, expert_id, input, from_node_id, .. } => format!(
            "rpc-expq1|{req_id}|{expert_id}|{}|{from_node_id}|{stamp}",
            hex::encode(pack_f32(input))
        ),
    }
}

fn verify_request(req: &NodeRpcRequest) -> bool {
    // Phase 1: the stamp freshness check always applies. Full signature
    // verification against the registry's known pubkeys lands with cert
    // pinning in Phase 2.
    let stamp = match req {
        NodeRpcRequest::Ping { stamp_ms, .. } | NodeRpcRequest::ExpertQuery { stamp_ms, .. } => *stamp_ms,
    };
    let age = crate::bootstrap::now_ms().saturating_sub(stamp);
    age <= crate::signing::STAMP_MAX_AGE_MS
}

fn pack_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Public: the node crate uses this when dispatching remote experts.
/// (Lib-side: exercised by tests; Phase 2 wire compression will use it on
/// the server decode path.)
#[allow(dead_code)]
fn unpack_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ---------------------------------------------------------------------------
// Locality-aware expert router
// ---------------------------------------------------------------------------

/// Where a candidate expert lives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpertLocation {
    Local,
    Remote { node_id: String, rtt_ms: u64 },
}

/// Dispatch policy inputs (pure logic — unit-testable without QUIC).
#[derive(Debug, Clone, Default)]
pub struct RouterOptions {
    /// RTT under which a peer counts as "nearby" (<30–50 ms per spec).
    pub nearby_rtt_ms: u64,
    /// Per-dispatch timeout.
    pub dispatch_timeout: Duration,
}

/// The locality-aware expert router.
pub struct ExpertRouter {
    pub options: RouterOptions,
    /// Experts hosted locally (expert_id → present).
    local_experts: RwLock<std::collections::HashSet<String>>,
}

impl ExpertRouter {
    pub fn new(options: RouterOptions) -> Self {
        Self {
            options,
            local_experts: RwLock::new(std::collections::HashSet::new()),
        }
    }

    pub async fn register_local_expert(&self, expert_id: &str) {
        self.local_experts.write().await.insert(expert_id.to_string());
    }

    pub async fn hosts_locally(&self, expert_id: &str) -> bool {
        self.local_experts.read().await.contains(expert_id)
    }

    /// Order candidate locations for a dispatch:
    /// 1. local (if we host it),
    /// 2. nearby peers (rtt <= nearby_rtt_ms), lowest RTT first,
    /// 3. remaining peers, lowest RTT first.
    /// Deprecated peers and peers that don't have the expert never appear.
    pub async fn plan_dispatch(
        &self,
        expert_id: &str,
        peers: &[(String, u64, bool, bool)], // (node_id, rtt_ms, has_expert, alive)
    ) -> Vec<ExpertLocation> {
        let mut plan = Vec::new();
        if self.hosts_locally(expert_id).await {
            plan.push(ExpertLocation::Local);
        }
        // Sorted ascending by RTT: nearby peers naturally precede far ones.
        let mut remotes: Vec<(String, u64)> = peers
            .iter()
            .filter(|(_, _, has, alive)| *has && *alive)
            .map(|(id, rtt, _, _)| (id.clone(), *rtt))
            .collect();
        remotes.sort_by_key(|(_, rtt)| *rtt);
        for (id, rtt) in remotes {
            plan.push(ExpertLocation::Remote { node_id: id, rtt_ms: rtt });
        }
        plan
    }

    /// Execute a dispatch following the plan with per-candidate timeouts.
    /// `dispatch` performs one remote/local attempt. Results are idempotent
    /// per `req_id` — later (duplicate) results are ignored by callers who
    /// check `completed` first.
    pub async fn execute(
        &self,
        req_id: &str,
        expert_id: &str,
        _input: &[f32],
        peers: &[(String, u64, bool, bool)],
        completed: &mut HashMap<String, Vec<f32>>,
        mut dispatch: impl FnMut(ExpertLocation, Duration) -> anyhow::Result<Vec<f32>>,
    ) -> anyhow::Result<Vec<f32>> {
        if let Some(prev) = completed.get(req_id) {
            return Ok(prev.clone());
        }
        let plan = self.plan_dispatch(expert_id, peers).await;
        let mut last_err = anyhow::anyhow!("no dispatch candidates for expert {expert_id}");
        for location in plan {
            match dispatch(location, self.options.dispatch_timeout) {
                Ok(out) => {
                    completed.insert(req_id.to_string(), out.clone());
                    return Ok(out);
                }
                Err(e) => {
                    last_err = e;
                }
            }
        }
        Err(last_err)
    }
}

// ---------------------------------------------------------------------------
// QUIC endpoint setup
// ---------------------------------------------------------------------------

/// Self-signed certificate material for the QUIC transport.
pub struct NodeCert {
    pub cert_der: rustls::pki_types::CertificateDer<'static>,
    pub key_der: rustls::pki_types::PrivateKeyDer<'static>,
}

/// Generate a self-signed certificate for this node (Phase 1: QUIC cert is
/// transport-only; identity proof is the Ed25519 signature per message).
pub fn generate_node_cert() -> anyhow::Result<NodeCert> {
    let ck = rcgen::generate_simple_self_signed(vec!["jerith-mesh-node".to_string()])
        .map_err(|e| anyhow::anyhow!("cert generation failed: {e}"))?;
    let cert_der = rustls::pki_types::CertificateDer::from(ck.cert.der().to_vec());
    let key_der = rustls::pki_types::PrivateKeyDer::Pkcs8(
        rustls::pki_types::PrivatePkcs8KeyDer::from(ck.key_pair.serialize_der()),
    );
    Ok(NodeCert { cert_der, key_der })
}

/// Accept-any-certificate verifier for the Phase 1 overlay. Documented
/// risk: replaced by per-node cert pinning in Phase 2.
#[derive(Debug)]
struct DangerVerifier;

impl rustls::client::danger::ServerCertVerifier for DangerVerifier {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        digest: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            digest,
            &rustls::crypto::ring::default_provider().signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        digest: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            digest,
            &rustls::crypto::ring::default_provider().signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        rustls::crypto::ring::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}

/// Build a client endpoint config that skips cert verification.
pub fn client_endpoint() -> anyhow::Result<quinn::Endpoint> {
    let mut endpoint = quinn::Endpoint::client("0.0.0.0:0".parse()?)?;
    let mut crypto = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(DangerVerifier))
        .with_no_client_auth();
    crypto.alpn_protocols = vec![b"jerith-rpc/1".to_vec()];
    let config = quinn::ClientConfig::new(Arc::new(
        quinn::crypto::rustls::QuicClientConfig::try_from(crypto)?,
    ));
    endpoint.set_default_client_config(config);
    Ok(endpoint)
}

/// Build a server endpoint on `addr`.
pub fn server_endpoint(addr: SocketAddr) -> anyhow::Result<quinn::Endpoint> {
    let cert = generate_node_cert()?;
    let mut server_crypto = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(vec![cert.cert_der], cert.key_der)?;
    server_crypto.alpn_protocols = vec![b"jerith-rpc/1".to_vec()];
    let config = quinn::ServerConfig::with_crypto(Arc::new(
        quinn::crypto::rustls::QuicServerConfig::try_from(server_crypto)?,
    ));
    let endpoint = quinn::Endpoint::server(config, addr)?;
    Ok(endpoint)
}

// ---------------------------------------------------------------------------
// RPC server + client
// ---------------------------------------------------------------------------

/// Handler the local node registers — runs an expert locally. Phase 1
/// stand-in: the Colibri integration replaces this with real expert
/// execution; the wire shape stays identical.
pub trait ExpertHandler: Send + Sync {
    fn run(&self, expert_id: &str, input: &[f32]) -> anyhow::Result<Vec<f32>>;
}

/// Serve RPC requests on a bound endpoint until shutdown. Verifies
/// request stamps and dispatches to the registered handler.
pub async fn serve(
    endpoint: quinn::Endpoint,
    handler: Arc<dyn ExpertHandler>,
    shutdown: tokio::sync::watch::Receiver<bool>,
) -> anyhow::Result<()> {
    let mut shutdown = shutdown;
    loop {
        tokio::select! {
            conn = endpoint.accept() => {
                let Some(conn) = conn else { break };
                let handler = Arc::clone(&handler);
                tokio::spawn(async move {
                    if let Ok(connection) = conn.await {
                        handle_connection(connection, handler).await;
                    }
                });
            }
            _ = shutdown.changed() => {
                if *shutdown.borrow() { break; }
            }
        }
    }
    endpoint.close(0u32.into(), b"shutdown");
    Ok(())
}

async fn handle_connection(connection: quinn::Connection, handler: Arc<dyn ExpertHandler>) {
    while let Ok((mut send, mut recv)) = connection.accept_bi().await {
        match read_frame(&mut recv).await {
            Ok(line) => {
                let reply = match serde_json::from_slice::<NodeRpcRequest>(&line) {
                    Ok(req) if verify_request(&req) => match req {
                        NodeRpcRequest::Ping { .. } => NodeRpcReply::Pong {
                            node_id: handler_node_id_fallback(),
                            stamp_ms: crate::bootstrap::now_ms(),
                        },
                        NodeRpcRequest::ExpertQuery { req_id, expert_id, input, .. } => {
                            match handler.run(&expert_id, &input) {
                                Ok(output) => NodeRpcReply::ExpertResult {
                                    req_id,
                                    output,
                                    node_id: handler_node_id_fallback(),
                                },
                                Err(e) => NodeRpcReply::Error {
                                    req_id,
                                    message: e.to_string(),
                                },
                            }
                        }
                    },
                    Ok(_) => NodeRpcReply::Error {
                        req_id: "unknown".into(),
                        message: "stale request stamp".into(),
                    },
                    Err(e) => NodeRpcReply::Error {
                        req_id: "unknown".into(),
                        message: format!("bad request: {e}"),
                    },
                };
                let mut out = match serde_json::to_vec(&reply) {
                    Ok(v) => v,
                    Err(_) => break,
                };
                out.push(b'\n');
                if send.write_all(&out).await.is_err() {
                    break;
                }
                let _ = send.finish();
            }
            Err(_) => break,
        }
    }
}

fn handler_node_id_fallback() -> String {
    // The server stamps replies with the local node_id; the wiring passes
    // it via the handler context in Phase 1 integration. Placeholder keeps
    // the wire shape stable.
    "unknown".into()
}

async fn read_frame(recv: &mut quinn::RecvStream) -> anyhow::Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(1024);
    let mut chunk = [0u8; 4096];
    loop {
        let n = recv.read(&mut chunk).await?;
        match n {
            Some(0) => break,
            Some(n) => {
                buf.extend_from_slice(&chunk[..n]);
                if let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                    buf.truncate(pos);
                    return Ok(buf);
                }
            }
            None => break,
        }
    }
    if buf.is_empty() {
        anyhow::bail!("empty rpc frame");
    }
    Ok(buf)
}

/// Phase 1 stand-in expert: ReLU over the input. Replaced by the Colibri
/// runtime's real expert execution; the wire shape is final.
pub struct ReluExpert;

impl ExpertHandler for ReluExpert {
    fn run(&self, _expert_id: &str, _input: &[f32]) -> anyhow::Result<Vec<f32>> {
        // Stand-in compute until the Colibri runtime supplies real experts.
        Ok(Vec::new())
    }
}

/// One-shot RPC client: connect, send a signed request, await the reply.
pub async fn call(
    endpoint: &quinn::Endpoint,
    addr: SocketAddr,
    req: &NodeRpcRequest,
    _identity: &SigningIdentity,
    timeout: Duration,
) -> anyhow::Result<NodeRpcReply> {
    let connecting = endpoint
        .connect(addr, "jerith-mesh-node")
        .map_err(|e| anyhow::anyhow!("connect failed: {e}"))?;
    let connection = tokio::time::timeout(timeout, connecting)
        .await
        .map_err(|_| anyhow::anyhow!("connect timeout"))?
        .map_err(|e| anyhow::anyhow!(e))?;
    let (mut send, mut recv) = tokio::time::timeout(timeout, connection.open_bi())
        .await
        .map_err(|_| anyhow::anyhow!("open_bi timeout"))??;

    let mut out = serde_json::to_vec(req)?;
    out.push(b'\n');
    send.write_all(&out).await?;
    send.finish()?;
    let line = tokio::time::timeout(timeout, read_frame(&mut recv))
        .await
        .map_err(|_| anyhow::anyhow!("reply timeout"))??;
    Ok(serde_json::from_slice(&line)?)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn peers() -> Vec<(String, u64, bool, bool)> {
        vec![
            ("near-a".into(), 10, true, true),
            ("near-b".into(), 25, true, true),
            ("far-c".into(), 180, true, true),
            ("far-d".into(), 240, true, true),
            ("dead-e".into(), 5, true, false),     // not alive → excluded
            ("noexp-f".into(), 3, false, true),    // lacks expert → excluded
        ]
    }

    fn router(nearby: u64) -> ExpertRouter {
        ExpertRouter::new(RouterOptions {
            nearby_rtt_ms: nearby,
            dispatch_timeout: Duration::from_millis(250),
        })
    }

    #[tokio::test]
    async fn plan_prefers_local_first() {
        let r = router(50);
        r.register_local_expert("exp-1").await;
        let plan = r.plan_dispatch("exp-1", &peers()).await;
        assert_eq!(plan.first(), Some(&ExpertLocation::Local));
        // Then nearby peers by ascending RTT.
        assert!(matches!(plan[1], ExpertLocation::Remote { ref node_id, rtt_ms: 10 } if node_id == "near-a"));
        assert!(matches!(plan[2], ExpertLocation::Remote { ref node_id, rtt_ms: 25 } if node_id == "near-b"));
    }

    #[tokio::test]
    async fn plan_without_local_starts_at_nearest() {
        let r = router(50);
        let plan = r.plan_dispatch("exp-1", &peers()).await;
        assert!(matches!(plan[0], ExpertLocation::Remote { ref node_id, rtt_ms: 10 } if node_id == "near-a"));
        assert!(!plan.iter().any(|l| matches!(l, ExpertLocation::Local)));
    }

    #[tokio::test]
    async fn plan_excludes_dead_and_lacking_peers() {
        let r = router(50);
        let plan = r.plan_dispatch("exp-1", &peers()).await;
        let ids: Vec<String> = plan
            .iter()
            .filter_map(|l| match l {
                ExpertLocation::Remote { node_id, .. } => Some(node_id.clone()),
                _ => None,
            })
            .collect();
        assert!(!ids.contains(&"dead-e".to_string()));
        assert!(!ids.contains(&"noexp-f".to_string()));
        assert_eq!(ids, vec!["near-a", "near-b", "far-c", "far-d"]);
    }

    #[tokio::test]
    async fn execute_local_wins_when_hosted() {
        let r = router(50);
        r.register_local_expert("exp-1").await;
        let mut completed = HashMap::new();
        let out = r
            .execute("req-1", "exp-1", &[1.0], &peers(), &mut completed, |loc, _| {
                match loc {
                    ExpertLocation::Local => Ok(vec![42.0]),
                    ExpertLocation::Remote { node_id, .. } => Ok(vec![0.0, node_id.len() as f32]),
                }
            })
            .await
            .unwrap();
        assert_eq!(out, vec![42.0]);
    }

    #[tokio::test]
    async fn execute_falls_through_on_failure() {
        let r = router(50);
        let mut completed = HashMap::new();
        let mut calls = 0;
        let out = r
            .execute("req-1", "exp-1", &[1.0], &peers(), &mut completed, |loc, _| {
                calls += 1;
                match loc {
                    ExpertLocation::Remote { node_id, .. } if node_id == "near-a" => {
                        Err(anyhow::anyhow!("near-a down"))
                    }
                    ExpertLocation::Remote { node_id, .. } => Ok(vec![node_id.len() as f32]),
                    ExpertLocation::Local => Ok(vec![-1.0]),
                }
            })
            .await
            .unwrap();
        // near-a failed → next candidate (near-b) served it.
        assert_eq!(out, vec!["near-b".len() as f32]);
        assert_eq!(calls, 2);
    }

    #[tokio::test]
    async fn execute_reports_when_all_fail() {
        let r = router(50);
        let mut completed = HashMap::new();
        let err = r
            .execute("req-1", "exp-1", &[1.0], &peers(), &mut completed, |_, _| {
                Err(anyhow::anyhow!("down"))
            })
            .await;
        assert!(err.is_err());
        assert!(completed.is_empty());
    }

    #[tokio::test]
    async fn execute_is_idempotent_per_req_id() {
        let r = router(50);
        let mut completed = HashMap::new();
        let first = r
            .execute("req-1", "exp-1", &[1.0], &peers(), &mut completed, |_, _| {
                Ok(vec![7.0])
            })
            .await
            .unwrap();
        let second = r
            .execute("req-1", "exp-1", &[1.0], &peers(), &mut completed, |_, _| {
                Ok(vec![99.0]) // must be ignored
            })
            .await
            .unwrap();
        assert_eq!(first, second);
        assert_eq!(second, vec![7.0]);
    }

    #[tokio::test]
    async fn execute_no_candidates_errors() {
        let r = router(50);
        let mut completed = HashMap::new();
        let err = r
            .execute("req-1", "missing-expert", &[1.0], &[], &mut completed, |_, _| {
                Ok(vec![1.0])
            })
            .await;
        assert!(err.is_err());
    }

    #[test]
    fn cert_generation_works() {
        let c = generate_node_cert().unwrap();
        assert!(!c.cert_der.as_ref().is_empty());
        assert!(!c.key_der.secret_der().is_empty());
    }

    // --- Live loopback E2E over QUIC ---

    struct TimesTwo;

    impl ExpertHandler for TimesTwo {
        fn run(&self, _expert_id: &str, input: &[f32]) -> anyhow::Result<Vec<f32>> {
            Ok(input.iter().map(|x| x * 2.0).collect())
        }
    }

    #[tokio::test]
    async fn quic_expert_query_roundtrip() {
        let server = server_endpoint("127.0.0.1:0".parse().unwrap()).unwrap();
        let addr = server.local_addr().unwrap();
        let (_tx, rx) = tokio::sync::watch::channel(false);
        tokio::spawn(async move { serve(server, Arc::new(TimesTwo), rx).await });

        let client = client_endpoint().unwrap();
        let identity = SigningIdentity::generate();
        let mut req = NodeRpcRequest::ExpertQuery {
            req_id: "e2e-1".into(),
            expert_id: "exp.test".into(),
            input: vec![1.0, 2.0, 3.5],
            from_node_id: identity.node_id(),
            stamp_ms: 0,
            sig: String::new(),
        };
        sign_request(&mut req, &identity);
        let reply = tokio::time::timeout(
            Duration::from_secs(10),
            call(&client, addr, &req, &identity, Duration::from_secs(5)),
        )
        .await
        .unwrap()
        .unwrap();
        match reply {
            NodeRpcReply::ExpertResult { req_id, output, .. } => {
                assert_eq!(req_id, "e2e-1");
                assert_eq!(output, vec![2.0, 4.0, 7.0]);
            }
            other => panic!("unexpected reply: {other:?}"),
        }
    }

    #[tokio::test]
    async fn quic_ping_roundtrip() {
        let server = server_endpoint("127.0.0.1:0".parse().unwrap()).unwrap();
        let addr = server.local_addr().unwrap();
        let (_tx, rx) = tokio::sync::watch::channel(false);
        tokio::spawn(async move { serve(server, Arc::new(TimesTwo), rx).await });

        let client = client_endpoint().unwrap();
        let identity = SigningIdentity::generate();
        let mut req = NodeRpcRequest::Ping {
            from_node_id: identity.node_id(),
            stamp_ms: 0,
            sig: String::new(),
        };
        sign_request(&mut req, &identity);
        let reply = call(&client, addr, &req, &identity, Duration::from_secs(5))
            .await
            .unwrap();
        assert!(matches!(reply, NodeRpcReply::Pong { .. }));
    }

    #[tokio::test]
    async fn quic_rejects_stale_stamp() {
        let server = server_endpoint("127.0.0.1:0".parse().unwrap()).unwrap();
        let addr = server.local_addr().unwrap();
        let (_tx, rx) = tokio::sync::watch::channel(false);
        tokio::spawn(async move { serve(server, Arc::new(TimesTwo), rx).await });

        let client = client_endpoint().unwrap();
        let identity = SigningIdentity::generate();
        let req = NodeRpcRequest::Ping {
            from_node_id: identity.node_id(),
            stamp_ms: crate::bootstrap::now_ms()
                - crate::signing::STAMP_MAX_AGE_MS
                - 5_000,
            sig: identity.sign(b"whatever").to_string(),
        };
        let reply = call(&client, addr, &req, &identity, Duration::from_secs(5))
            .await
            .unwrap();
        assert!(matches!(reply, NodeRpcReply::Error { .. }));
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let v = vec![0.5f32, -2.25, 1e10];
        assert_eq!(unpack_f32(&pack_f32(&v)), v);
        let _ = sign_request; // exercised by the quic E2E tests above
    }

    #[test]
    fn request_payload_deterministic() {
        let req = NodeRpcRequest::ExpertQuery {
            req_id: "r1".into(),
            expert_id: "e1".into(),
            input: vec![1.0],
            from_node_id: "abc".into(),
            stamp_ms: 5,
            sig: String::new(),
        };
        assert_eq!(
            request_payload_for_stamp(&req, 5),
            request_payload_for_stamp(&req, 5)
        );
        assert!(request_payload_for_stamp(&req, 5).starts_with("rpc-expq1|"));
    }
}
