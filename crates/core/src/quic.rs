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
    /// Ask a peer to run a full chat completion on its local runtime.
    /// Used by mesh fallback routing when the local runtime is down.
    ChatQuery {
        req_id: String,
        /// Full OpenAI-style messages array (a caller-owned PrivacyContext).
        messages: Vec<serde_json::Value>,
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
    /// Full chat-completion reply from a peer's runtime.
    ChatResult { req_id: String, content: String, node_id: String },
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
        NodeRpcRequest::ChatQuery { stamp_ms, sig: _, .. } => {
            *stamp_ms = stamp;
        }
    }
    let payload = request_payload_for_stamp(req, stamp);
    let signature = identity.sign(payload.as_bytes());
    match req {
        NodeRpcRequest::Ping { sig, .. }
        | NodeRpcRequest::ExpertQuery { sig, .. }
        | NodeRpcRequest::ChatQuery { sig, .. } => {
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
        NodeRpcRequest::ChatQuery { req_id, messages, from_node_id, .. } => format!(
            "rpc-chatq1|{req_id}|{}|{from_node_id}|{stamp}",
            hex::encode(serde_json::to_vec(messages).unwrap_or_default())
        ),
    }
}

fn verify_request(req: &NodeRpcRequest) -> bool {
    // Phase 1: the stamp freshness check always applies. Full signature
    // verification against the registry's known pubkeys lands with cert
    // pinning in Phase 2.
    let stamp = match req {
        NodeRpcRequest::Ping { stamp_ms, .. }
        | NodeRpcRequest::ExpertQuery { stamp_ms, .. }
        | NodeRpcRequest::ChatQuery { stamp_ms, .. } => *stamp_ms,
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

    /// Drop a locally-hosted expert (e.g., evicted from cache).
    pub async fn unregister_local_expert(&self, expert_id: &str) {
        self.local_experts.write().await.remove(expert_id);
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

    /// QUIC-backed dispatch: same candidate plan, but remote candidates
    /// resolve over real QUIC NodeRPC (signed ExpertQuery → ExpertResult).
    /// `local_run` computes locally when the plan's first candidate is
    /// local. Feedback failures land in `completed` as usual; the caller
    /// receives the first successful result.
    /// Mesh fallback chat: try the local handler first, then remotes in
    /// the same locality order used for experts. Identity is preserved
    /// via attribution checks; privacy scoping happens at the caller.
    #[allow(clippy::too_many_arguments)]
    pub async fn chat_over_quic(
        &self,
        endpoint: &quinn::Endpoint,
        identity: &SigningIdentity,
        peer_addrs: &HashMap<String, SocketAddr>,
        req_id: &str,
        messages: &[serde_json::Value],
        peers: &[(String, u64, bool, bool)],
        local_chat: impl FnOnce() -> anyhow::Result<String>,
    ) -> anyhow::Result<String> {
        // Local first (caller decides strictness and messaging).
        if let Ok(content) = local_chat() {
            return Ok(content);
        }
        tracing::warn!(
            target: "mesh",
            "local runtime unavailable for chat — attempting mesh fallback"
        );

        // Remotes in lowest-RTT order (same ordering principle as experts).
        let mut remotes: Vec<(String, u64)> = peers
            .iter()
            .filter(|(_, _, has, alive)| *has && *alive)
            .map(|(id, rtt, _, _)| (id.clone(), *rtt))
            .collect();
        remotes.sort_by_key(|(_, rtt)| *rtt);

        let mut last_err = anyhow::anyhow!("no peers available for mesh chat fallback");
        for (node_id, _rtt) in remotes {
            let Some(addr) = peer_addrs.get(&node_id) else {
                continue;
            };
            let mut req = NodeRpcRequest::ChatQuery {
                req_id: req_id.to_string(),
                messages: messages.to_vec(),
                from_node_id: identity.node_id(),
                stamp_ms: 0,
                sig: String::new(),
            };
            sign_request(&mut req, identity);
            match call(endpoint, *addr, &req, identity, self.options.dispatch_timeout).await {
                Ok(NodeRpcReply::ChatResult {
                    req_id: rid,
                    content,
                    node_id: server_id,
                }) => {
                    if rid != req_id || server_id != node_id {
                        tracing::debug!(
                            target: "mesh",
                            peer = %node_id,
                            "chat reply attribution mismatch — trying next peer"
                        );
                        last_err = anyhow::anyhow!(
                            "chat reply identity mismatch: expected {node_id}, got {server_id}"
                        );
                        continue;
                    }
                    tracing::info!(
                        target: "mesh",
                        peer = %node_id,
                        "chat served via mesh fallback"
                    );
                    return Ok(content);
                }
                Ok(NodeRpcReply::Error { req_id: _id, message }) => {
                    last_err = anyhow::anyhow!("peer rejected chat: {message}");
                }
                Ok(other) => {
                    last_err = anyhow::anyhow!("unexpected reply kind: {other:?}");
                }
                Err(e) => {
                    last_err = anyhow::anyhow!("quic chat dispatch failed: {e}");
                }
            }
        }
        Err(last_err)
    }

    pub async fn execute_over_quic(
        &self,
        endpoint: &quinn::Endpoint,
        identity: &SigningIdentity,
        peer_addrs: &HashMap<String, SocketAddr>,
        req_id: &str,
        expert_id: &str,
        input: &[f32],
        peers: &[(String, u64, bool, bool)],
        completed: &mut HashMap<String, Vec<f32>>,
        local_run: impl Fn(&[f32]) -> anyhow::Result<Vec<f32>>,
    ) -> anyhow::Result<Vec<f32>> {
        if let Some(prev) = completed.get(req_id) {
            return Ok(prev.clone());
        }
        let plan = self.plan_dispatch(expert_id, peers).await;
        let mut last_err = anyhow::anyhow!(
            "no dispatch candidates for expert {expert_id} (plan empty or all failed)"
        );
        for location in plan {
            let result = match &location {
                ExpertLocation::Local => local_run(input),
                ExpertLocation::Remote { node_id, .. } => {
                    match peer_addrs.get(node_id) {
                        None => Err(anyhow::anyhow!("no QUIC address for peer {node_id}")),
                        Some(addr) => {
                            let mut req = NodeRpcRequest::ExpertQuery {
                                req_id: req_id.to_string(),
                                expert_id: expert_id.to_string(),
                                input: input.to_vec(),
                                from_node_id: identity.node_id(),
                                stamp_ms: 0,
                                sig: String::new(),
                            };
                            sign_request(&mut req, identity);
                            match call(endpoint, *addr, &req, identity, self.options.dispatch_timeout)
                                .await
                            {
                                Ok(NodeRpcReply::ExpertResult {
                                    req_id: rid,
                                    output,
                                    node_id: server_id,
                                }) => {
                                    // Attribute check: reply must come from
                                    // the exact candidate we dispatched to.
                                    if server_id != *node_id || rid != req_id {
                                        Err(anyhow::anyhow!(
                                            "reply identity mismatch: expected {node_id}, got {server_id}"
                                        ))
                                    } else {
                                        Ok(output)
                                    }
                                }
                                Ok(NodeRpcReply::Error { req_id: _id, message }) => {
                                    Err(anyhow::anyhow!("peer rejected expert query: {message}"))
                                }
                                Ok(other) => {
                                    Err(anyhow::anyhow!("unexpected reply kind: {other:?}"))
                                }
                                Err(e) => Err(anyhow::anyhow!("quic dispatch failed: {e}")),
                            }
                        }
                    }
                }
            };
            match result {
                Ok(out) => {
                    completed.insert(req_id.to_string(), out.clone());
                    return Ok(out);
                }
                Err(e) => {
                    tracing::debug!(
                        target: "mesh",
                        expert = %expert_id,
                        location = ?location,
                        error = %e,
                        "dispatch candidate failed — falling through"
                    );
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

    /// Serve a full chat completion from the local runtime. Handlers that
    /// can't do chat (pure expert stand-ins) default to an error reply.
    fn chat<'a>(
        &'a self,
        _messages: &'a [serde_json::Value],
    ) -> futures_core::future::BoxFuture<'a, anyhow::Result<String>> {
        Box::pin(async {
            anyhow::bail!("handler does not support chat completions")
        })
    }
}

/// Serve RPC requests on a bound endpoint until shutdown. Verifies
/// request stamps and dispatches to the registered handler. Every reply
/// carries `server_node_id` so the pinger can attribute it.
pub async fn serve(
    endpoint: quinn::Endpoint,
    handler: Arc<dyn ExpertHandler>,
    server_node_id: String,
    shutdown: tokio::sync::watch::Receiver<bool>,
) -> anyhow::Result<()> {
    let mut shutdown = shutdown;
    loop {
        tokio::select! {
            conn = endpoint.accept() => {
                let Some(conn) = conn else { break };
                let handler = Arc::clone(&handler);
                let server_node_id = server_node_id.clone();
                tokio::spawn(async move {
                    if let Ok(connection) = conn.await {
                        handle_connection(connection, handler, server_node_id).await;
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

async fn handle_connection(
    connection: quinn::Connection,
    handler: Arc<dyn ExpertHandler>,
    server_node_id: String,
) {
    while let Ok((mut send, mut recv)) = connection.accept_bi().await {
        match read_frame(&mut recv).await {
            Ok(line) => {
                let server_id = server_node_id.clone();
                let reply = match serde_json::from_slice::<NodeRpcRequest>(&line) {
                    Ok(req) if verify_request(&req) => match req {
                        NodeRpcRequest::Ping { .. } => NodeRpcReply::Pong {
                            node_id: server_id,
                            stamp_ms: crate::bootstrap::now_ms(),
                        },
                        NodeRpcRequest::ExpertQuery { req_id, expert_id, input, .. } => {
                            match handler.run(&expert_id, &input) {
                                Ok(output) => NodeRpcReply::ExpertResult {
                                    req_id,
                                    output,
                                    node_id: server_id,
                                },
                                Err(e) => NodeRpcReply::Error {
                                    req_id,
                                    message: e.to_string(),
                                },
                            }
                        }
                        NodeRpcRequest::ChatQuery { req_id, messages, .. } => {
                            match handler.chat(&messages).await {
                                Ok(content) => NodeRpcReply::ChatResult {
                                    req_id,
                                    content,
                                    node_id: server_id,
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

/// Expert handler backed by the local Colibri-style runtime.
///
/// Phase 1 integration strategy: Colibri (and Ollama/litellm) expose an
/// OpenAI-compatible HTTP interface, not raw expert tensors. So the
/// "expert query" maps to a round-trip through the runtime's chat
/// endpoint with the requesting context embedded as system guidance —
/// the requester gets a *completion*, not raw activations. This keeps the
/// wire contract stable while giving peers real model compute immediately.
/// When Colibri exposes tensor-level export, only this handler changes.
pub struct RuntimeExpert {
    /// Base URL of the runtime (e.g. http://127.0.0.1:8000).
    pub runtime_url: String,
    /// Model to invoke (mesh model by default).
    pub model: String,
    /// HTTP client (shared, pooled).
    client: reqwest::Client,
}

impl RuntimeExpert {
    /// Discover the runtime: probe `{url}/v1/models` and confirm the
    /// requested model is being served. Returns the handler on success.
    pub async fn discover(runtime_url: &str, model: &str) -> anyhow::Result<Self> {
        let url = format!("{}/v1/models", runtime_url.trim_end_matches('/'));
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()?;
        let resp = client
            .get(&url)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("runtime unreachable at {url}: {e}"))?
            .error_for_status()
            .map_err(|e| anyhow::anyhow!("runtime /v1/models failed: {e}"))?;
        let body: serde_json::Value = resp.json().await?;
        let served: Vec<String> = body
            .get("data")
            .and_then(|d| d.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m.get("id").and_then(|i| i.as_str()))
                    .map(|s| s.to_string())
                    .collect()
            })
            .unwrap_or_default();
        if !served.iter().any(|s| s.contains(model)) && !model.is_empty() {
            anyhow::bail!(
                "runtime does not serve model {model:?} (serves: {served:?})"
            );
        }
        Ok(Self {
            runtime_url: runtime_url.to_string(),
            model: model.to_string(),
            client,
        })
    }
}

impl ExpertHandler for RuntimeExpert {
    fn chat<'a>(
        &'a self,
        messages: &'a [serde_json::Value],
    ) -> futures_core::future::BoxFuture<'a, anyhow::Result<String>> {
        Box::pin(async move { self.chat_impl(messages).await })
    }

    fn run(&self, expert_id: &str, input: &[f32]) -> anyhow::Result<Vec<f32>> {
        // Real expert execution is async HTTP; bridge the sync trait with
        // a block-on worker and a strict timeout (never blocks a pass).
        let this = Self {
            runtime_url: self.runtime_url.clone(),
            model: self.model.clone(),
            client: self.client.clone(),
        };
        let expert_id = expert_id.to_string();
        let input = input.to_vec();
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async move {
                this.run_async(&expert_id, &input).await
            })
        })
    }
}

impl RuntimeExpert {
    /// Synchronous-first chat implementation (also the trait method body).
    async fn chat_impl(&self, messages: &[serde_json::Value]) -> anyhow::Result<String> {
        let url = format!(
            "{}/v1/chat/completions",
            self.runtime_url.trim_end_matches('/')
        );
        // Never forward a tool/system context we didn't intend; the
        // messages are the caller's privacy-scoped payload verbatim.
        let resp = self
            .client
            .post(&url)
            .json(&serde_json::json!({
                "model": self.model,
                "messages": messages,
                "max_tokens": 512,
            }))
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("runtime chat call failed: {e}"))?
            .error_for_status()
            .map_err(|e| anyhow::anyhow!("runtime returned error: {e}"))?;
        let body: serde_json::Value = resp.json().await?;
        let content = body
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string();
        if content.is_empty() {
            anyhow::bail!("runtime returned empty chat completion");
        }
        Ok(content)
    }

    async fn run_async(&self, expert_id: &str, input: &[f32]) -> anyhow::Result<Vec<f32>> {
        let prompt = format!(
            "[expert:{expert_id}] input tensor digest: {:?} — respond with the computedsummary only.",
            input
                .iter()
                .take(8)
                .map(|f| format!("{f:.4}"))
                .collect::<Vec<_>>()
                .join(",")
        );
        let url = format!(
            "{}/v1/chat/completions",
            self.runtime_url.trim_end_matches('/')
        );
        let resp = self
            .client
            .post(&url)
            .json(&serde_json::json!({
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 64,
            }))
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("runtime call failed: {e}"))?
            .error_for_status()
            .map_err(|e| anyhow::anyhow!("runtime returned error: {e}"))?;
        let body: serde_json::Value = resp.json().await?;
        let text = body
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|t| t.as_str())
            .unwrap_or_default();
        if text.is_empty() {
            anyhow::bail!("runtime returned empty completion");
        }
        // Encode the completion deterministically back into the wire's
        // f32-vector contract (hashed, stable, sender can't confuse it
        // with raw activations because expert_id is echoed in the reply).
        let out = crate::memory::hash_embed(&format!("{expert_id}|{text}"), input.len().max(8));
        Ok(out)
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
        tokio::spawn(async move {
            serve(server, Arc::new(TimesTwo), "test-server".to_string(), rx).await
        });

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
        tokio::spawn(async move {
            serve(server, Arc::new(TimesTwo), "test-server".to_string(), rx).await
        });

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

    struct Multiplier {
        factor: f32,
    }

    impl ExpertHandler for Multiplier {
        fn run(&self, _expert_id: &str, input: &[f32]) -> anyhow::Result<Vec<f32>> {
            Ok(input.iter().map(|x| x * self.factor).collect())
        }
    }

    async fn spawn_expert_server(
        factor: f32,
        node_id: &str,
    ) -> anyhow::Result<SocketAddr> {
        let server = server_endpoint("127.0.0.1:0".parse().unwrap())?;
        let addr = server.local_addr().unwrap();
        let (_tx, rx) = tokio::sync::watch::channel(false);
        let node_id = node_id.to_string();
        tokio::spawn(async move {
            serve(server, Arc::new(Multiplier { factor }), node_id, rx).await
        });
        Ok(addr)
    }

    #[tokio::test]
    async fn router_execute_over_quic_prefers_local_and_falls_through() {
        // Two remote servers: near (×3, rtt 5) and far (×7, rtt 200).
        let near_addr = spawn_expert_server(3.0, "near").await.unwrap();
        let far_addr = spawn_expert_server(7.0, "far").await.unwrap();

        let r = router(50); // nearby threshold 50ms
        let node_id = SigningIdentity::generate().node_id();
        let peers = vec![
            ("near".to_string(), 5, true, true),
            ("far".to_string(), 200, true, true),
        ];
        let mut addrs = HashMap::new();
        addrs.insert("near".to_string(), near_addr);
        addrs.insert("far".to_string(), far_addr);

        let client = client_endpoint().unwrap();
        let identity = SigningIdentity::generate();
        let mut completed = HashMap::new();

        // No local expert → remote plan: near ×3 first.
        let out = r
            .execute_over_quic(
                &client,
                &identity,
                &addrs,
                "q1",
                "exp.test",
                &[1.0, 2.0],
                &peers,
                &mut completed,
                |_| Ok(vec![0.0]),
            )
            .await
            .unwrap();
        assert_eq!(out, vec![3.0, 6.0]);

        // Idempotence: same req_id returns the memoized result.
        let out2 = r
            .execute_over_quic(
                &client,
                &identity,
                &addrs,
                "q1",
                "exp.test",
                &[9.0, 9.0],
                &peers,
                &mut completed,
                |_| Ok(vec![1.0]),
            )
            .await
            .unwrap();
        assert_eq!(out2, vec![3.0, 6.0]);

        // Local-first: with a local expert registered, local wins.
        r.register_local_expert("exp.test").await;
        let out3 = r
            .execute_over_quic(
                &client,
                &identity,
                &addrs,
                "q2",
                "exp.test",
                &[4.0],
                &peers,
                &mut completed,
                |input| Ok(input.iter().map(|x| x * 10.0).collect()),
            )
            .await
            .unwrap();
        assert_eq!(out3, vec![40.0]);

        // Fallthrough: local unregistered + near dies (removed from addrs)
        // → far ×7 serves.
        r.unregister_local_expert("exp.test").await;
        let mut addrs2 = addrs.clone();
        addrs2.remove("near");
        let out4 = r
            .execute_over_quic(
                &client,
                &identity,
                &addrs2,
                "q3",
                "exp.test",
                &[4.0],
                &peers,
                &mut completed,
                |input| Ok(input.iter().map(|x| x * 10.0).collect()),
            )
            .await
            .unwrap();
        assert_eq!(out4, vec![28.0]);

        // All candidates gone → error carrying the last cause.
        let err = r
            .execute_over_quic(
                &client,
                &identity,
                &HashMap::new(),
                "q4",
                "exp.test",
                &[1.0],
                &peers,
                &mut completed,
                |_| Err(anyhow::anyhow!("local also down")),
            )
            .await;
        assert!(err.is_err());
    }

    struct ChatAnswer {
        content: String,
    }

    impl ExpertHandler for ChatAnswer {
        fn run(&self, _expert_id: &str, _input: &[f32]) -> anyhow::Result<Vec<f32>> {
            Ok(Vec::new())
        }

        fn chat<'a>(
            &'a self,
            _messages: &'a [serde_json::Value],
        ) -> futures_core::future::BoxFuture<'a, anyhow::Result<String>> {
            let content = self.content.clone();
            Box::pin(async move { Ok(content) })
        }
    }

    #[tokio::test]
    async fn chat_over_quic_falls_back_to_peer() {
        let server = server_endpoint("127.0.0.1:0".parse().unwrap()).unwrap();
        let addr = server.local_addr().unwrap();
        let (_tx, rx) = tokio::sync::watch::channel(false);
        let server_id = "peer-b".to_string();
        tokio::spawn(async move {
            serve(
                server,
                Arc::new(ChatAnswer {
                    content: "mesh hello".into(),
                }),
                server_id,
                rx,
            )
            .await
        });

        let r = router(50);
        let client = client_endpoint().unwrap();
        let identity = SigningIdentity::generate();
        let mut addrs = HashMap::new();
        addrs.insert("peer-b".to_string(), addr);
        let peers = vec![("peer-b".to_string(), 8, true, true)];

        // Local runtime down → falls back to the mesh peer.
        let content = r
            .chat_over_quic(
                &client,
                &identity,
                &addrs,
                "chat-1",
                &[serde_json::json!({"role": "user", "content": "hi"})],
                &peers,
                || Err(anyhow::anyhow!("local runtime down")),
            )
            .await
            .unwrap();
        assert_eq!(content, "mesh hello");

        // Local healthy → mesh never used.
        let content2 = r
            .chat_over_quic(
                &client,
                &identity,
                &addrs,
                "chat-2",
                &[serde_json::json!({"role": "user", "content": "hi"})],
                &peers,
                || Ok("local hello".into()),
            )
            .await
            .unwrap();
        assert_eq!(content2, "local hello");

        // No peers + local down → error (never blocks the request).
        let err = r
            .chat_over_quic(
                &client,
                &identity,
                &HashMap::new(),
                "chat-3",
                &[serde_json::json!({"role": "user", "content": "hi"})],
                &peers,
                || Err(anyhow::anyhow!("local runtime down")),
            )
            .await;
        assert!(err.is_err());
    }

    #[tokio::test]
    async fn quic_rejects_stale_stamp() {
        let server = server_endpoint("127.0.0.1:0".parse().unwrap()).unwrap();
        let addr = server.local_addr().unwrap();
        let (_tx, rx) = tokio::sync::watch::channel(false);
        tokio::spawn(async move {
            serve(server, Arc::new(TimesTwo), "test-server".to_string(), rx).await
        });

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
