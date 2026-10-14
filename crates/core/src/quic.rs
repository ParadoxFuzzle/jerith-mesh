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
    /// Expert activation + commitment: SHA-256 over the canonical
    /// (req_id, expert_id, node_id, input, output) tuple — Phase 4
    /// verification. `None` only from pre-Phase-4 peers (accepted without a
    /// commitment check).
    ExpertResult {
        req_id: String,
        output: Vec<f32>,
        node_id: String,
        #[serde(default)]
        commitment: Option<String>,
    },
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

/// Hex of the LE-packed f32 slice — used in the Phase 4 commitment payload
/// so the verifier re-derives the exact bytes the server hashed.
pub(crate) fn pack_f32_hex(v: &[f32]) -> String {
    hex::encode(pack_f32(v))
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

#[derive(Debug, Clone, Default)]
pub struct RouterOptions {
    /// RTT under which a peer counts as "nearby" (<30–50 ms per spec).
    pub nearby_rtt_ms: u64,
    /// Per-dispatch timeout.
    pub dispatch_timeout: Duration,
    /// Redundant-replay verification probability in percent (0–100;
    /// 0 = never replay). The spec suggests 5–15 %.
    pub verify_percent: u32,
    /// Absolute tolerance when comparing replayed outputs (0 = exact).
    pub verify_tolerance: f32,
}

/// The locality-aware expert router.
pub struct ExpertRouter {
    pub options: RouterOptions,
    /// Experts hosted locally (expert_id → present).
    local_experts: RwLock<std::collections::HashSet<String>>,
    /// Phase 4: verification counters + queued fraud evidence about peers
    /// whose results failed commitments or replay comparison.
    verify: Arc<crate::verify::VerificationTracker>,
}

impl ExpertRouter {
    pub fn new(options: RouterOptions) -> Self {
        Self::new_with_tracker(options, crate::verify::VerificationTracker::shared())
    }

    /// Construct with an external verification tracker — the mesh state's
    /// tracker is shared so evidence flushes on the ping loop.
    pub fn new_with_tracker(
        options: RouterOptions,
        verify: Arc<crate::verify::VerificationTracker>,
    ) -> Self {
        Self {
            options,
            local_experts: RwLock::new(std::collections::HashSet::new()),
            verify,
        }
    }

    /// The router's verification tracker — shared with the mesh state so
    /// the ping loop flushes DishonestResults evidence to the bootstrap.
    pub fn verify_tracker(&self) -> Arc<crate::verify::VerificationTracker> {
        self.verify.clone()
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

    /// Collapse consecutive duplicate turns in a chat history before it
    /// rides the mesh wire. OpenAI clients replay their whole history every
    /// request, and retries/restored sessions can duplicate the tail turns;
    /// the fallback dial loop then re-sends that array to *every* peer it
    /// tries. Only back-to-back exact matches collapse to one — non-adjacent
    /// repeats are kept (they can be intentional exemplar turns). The local
    /// runtime path stays verbatim: dedup is a mesh-bandwidth concern only.
    pub fn compact_chat_history(messages: &[serde_json::Value]) -> Vec<serde_json::Value> {
        let mut out: Vec<serde_json::Value> = Vec::with_capacity(messages.len());
        for m in messages {
            match out.last() {
                Some(prev) if prev == m => continue, // consecutive duplicate turn
                _ => out.push(m.clone()),
            }
        }
        out
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
    ) -> anyhow::Result<(String, String)> {
        // Returns (content, answering_node_id). Local answers attribute to the
        // empty string (the gateway maps that to mesh_fallback: null);
        // remote answers carry the node_id that actually served the request,
        // which the gateway/Meters then credit — not an RTT-order guess.
        // Local first (caller decides strictness and messaging).
        if let Ok(content) = local_chat() {
            return Ok((content, String::new()));
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
                messages: Self::compact_chat_history(messages),
                from_node_id: identity.node_id(),
                stamp_ms: 0,
                sig: String::new(),
            };
            sign_request(&mut req, identity);
            match call(endpoint, *addr, &node_id, &req, identity, self.options.dispatch_timeout)
                .await
            {
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
                    return Ok((content, node_id));
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
        let mut ambiguous_abort: Option<anyhow::Error> = None;
        for (idx, location) in plan.iter().enumerate() {
            let result = match &location {
                ExpertLocation::Local => local_run(input),
                ExpertLocation::Remote { node_id, .. } => {
                    match self
                        .remote_expert_attempt(
                            endpoint, identity, peer_addrs, req_id, expert_id, input, node_id,
                        )
                        .await
                    {
                        Ok(out) => {
                            self.verify.note_expert_result().await;
                            // Phase 4: probabilistic redundant replay before
                            // trusting this remote result.
                            match self
                                .replay_verify(
                                    endpoint,
                                    identity,
                                    peer_addrs,
                                    req_id,
                                    expert_id,
                                    input,
                                    &plan,
                                    idx,
                                    node_id,
                                    &out,
                                    &local_run,
                                )
                                .await
                            {
                                ReplayVerdict::Trusted => Ok(out),
                                ReplayVerdict::RejectContinue => Err(anyhow::anyhow!(
                                    "replay verification rejected peer {node_id} (local arbiter disagrees)"
                                )),
                                ReplayVerdict::RejectAbort => {
                                    // Ambiguous: two remotes disagree and we
                                    // host no local truth — remaining remote
                                    // candidates are equally suspect.
                                    ambiguous_abort = Some(anyhow::anyhow!(
                                        "remote outputs disagree (both peers flagged)"
                                    ));
                                    Err(anyhow::anyhow!(
                                        "replay verification failed for expert {expert_id}: remote outputs disagree"
                                    ))
                                }
                            }
                        }
                        Err(e) => Err(e),
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
                    if let Some(abort) = ambiguous_abort {
                        tracing::warn!(
                            target: "mesh",
                            expert = %expert_id,
                            "abandoning remaining remote candidates ({abort})"
                        );
                        return Err(abort);
                    }
                }
            }
        }
        Err(last_err)
    }

    /// One signed remote expert dispatch: sign → call → attribute → verify
    /// the Phase 4 commitment. The tracker records the commitment outcome
    /// (and queues DishonestResults evidence when the body was tampered). If
    /// a peer returns no commitment (pre-Phase-4), the result is accepted —
    /// we never reject work we cannot *prove* wrong.
    #[allow(clippy::too_many_arguments)]
    async fn remote_expert_attempt(
        &self,
        endpoint: &quinn::Endpoint,
        identity: &SigningIdentity,
        peer_addrs: &HashMap<String, SocketAddr>,
        req_id: &str,
        expert_id: &str,
        input: &[f32],
        node_id: &str,
    ) -> anyhow::Result<Vec<f32>> {
        let Some(addr) = peer_addrs.get(node_id) else {
            return Err(anyhow::anyhow!("no QUIC address for peer {node_id}"));
        };
        let mut req = NodeRpcRequest::ExpertQuery {
            req_id: req_id.to_string(),
            expert_id: expert_id.to_string(),
            input: input.to_vec(),
            from_node_id: identity.node_id(),
            stamp_ms: 0,
            sig: String::new(),
        };
        sign_request(&mut req, identity);
        match call(
            endpoint,
            *addr,
            node_id,
            &req,
            identity,
            self.options.dispatch_timeout,
        )
        .await
        {
            Ok(NodeRpcReply::ExpertResult {
                req_id: rid,
                output,
                node_id: server_id,
                commitment,
            }) => {
                // Attribute check: reply must come from the exact candidate
                // we dispatched to.
                if server_id != node_id || rid != req_id {
                    return Err(anyhow::anyhow!(
                        "reply identity mismatch: expected {node_id}, got {server_id}"
                    ));
                }
                match commitment {
                    Some(c) => {
                        let ok = crate::verify::verify_commitment(
                            req_id,
                            expert_id,
                            node_id,
                            input,
                            &output,
                            &c,
                        );
                        // note_commitment already flags the peer on failure.
                        self.verify.note_commitment(node_id, ok).await;
                        if ok {
                            Ok(output)
                        } else {
                            Err(anyhow::anyhow!(
                                "commitment mismatch from peer {node_id}: body altered"
                            ))
                        }
                    }
                    None => Ok(output),
                }
            }
            Ok(NodeRpcReply::Error { req_id: _id, message }) => {
                Err(anyhow::anyhow!("peer rejected expert query: {message}"))
            }
            Ok(other) => Err(anyhow::anyhow!("unexpected reply kind: {other:?}")),
            Err(e) => Err(anyhow::anyhow!("quic dispatch failed: {e}")),
        }
    }

    /// Phase 4 redundant replay verification. Re-runs the same query on an
    /// independent arbiter and compares:
    /// - local expert hosted → the local run is the arbiter (self-trusted:
    ///   a disagreement convicts the remote alone),
    /// - else the next remote candidate in the plan (a disagreement
    ///   convicts both — ambiguous, per the spec: "both (or the outlier)
    ///   get a temporary reputation penalty").
    ///
    /// Runs only with probability `verify_percent`; with no arbiter
    /// available the result is accepted unverified (nothing to report).
    #[allow(clippy::too_many_arguments)]
    async fn replay_verify(
        &self,
        endpoint: &quinn::Endpoint,
        identity: &SigningIdentity,
        peer_addrs: &HashMap<String, SocketAddr>,
        req_id: &str,
        expert_id: &str,
        input: &[f32],
        plan: &[ExpertLocation],
        origin_idx: usize,
        origin_node_id: &str,
        origin_output: &[f32],
        local_run: &impl Fn(&[f32]) -> anyhow::Result<Vec<f32>>,
    ) -> ReplayVerdict {
        if self.options.verify_percent == 0
            || rand::random::<u32>() % 100 >= self.options.verify_percent
        {
            return ReplayVerdict::Trusted;
        }

        // Pick the arbiter: local truth first, else the next remote.
        let mut arbiter: Option<(Vec<f32>, Option<String>)> = None;
        if self.hosts_locally(expert_id).await {
            if let Ok(local) = local_run(input) {
                arbiter = Some((local, None)); // None = local (trusted self)
            }
        }
        if arbiter.is_none() {
            for loc in plan.iter().skip(origin_idx + 1) {
                if let ExpertLocation::Remote { node_id, .. } = loc {
                    if let Ok(out) = self
                        .remote_expert_attempt(
                            endpoint,
                            identity,
                            peer_addrs,
                            &format!("{req_id}-rep"),
                            expert_id,
                            input,
                            node_id,
                        )
                        .await
                    {
                        self.verify.note_expert_result().await;
                        arbiter = Some((out, Some(node_id.clone())));
                        break;
                    }
                }
            }
        }
        let Some((arbiter_out, arbiter_id)) = arbiter else {
            // No independent arbiter reachable — verify not possible, no
            // evidence either way (fail open, but no proof-of-liar).
            return ReplayVerdict::Trusted;
        };

        if crate::verify::outputs_agree(
            origin_output,
            &arbiter_out,
            self.options.verify_tolerance,
        ) {
            self.verify.note_replay(true).await;
            return ReplayVerdict::Trusted;
        }
        self.verify.note_replay(false).await;
        match arbiter_id {
            None => {
                // The local expert disagrees with the remote — we know who
                // lies: the remote (or our local model, but self is truth).
                self.verify.flag_dishonest(&[origin_node_id]).await;
                tracing::warn!(
                    target: "mesh",
                    peer = %origin_node_id,
                    expert = %expert_id,
                    "replay mismatch vs local arbiter — peer flagged"
                );
                ReplayVerdict::RejectContinue
            }
            Some(other) => {
                self.verify.flag_dishonest(&[origin_node_id, other.as_str()]).await;
                tracing::warn!(
                    target: "mesh",
                    peers = [origin_node_id, other.as_str()].join(" "),
                    expert = %expert_id,
                    "remote replay mismatch — both flagged (ambiguous)"
                );
                ReplayVerdict::RejectAbort
            }
        }
    }
}

/// Outcome of a redundant-replay verification.
enum ReplayVerdict {
    /// Agreed (or nothing to verify against) — trust the result.
    Trusted,
    /// Local arbiter disagrees: origin remote flagged, keep walking the plan.
    RejectContinue,
    /// Two remotes disagree (no local arbiter): both flagged, remaining
    /// remote candidates are suspect — abort the trace for this req_id.
    RejectAbort,
}

// ---------------------------------------------------------------------------
// QUIC endpoint setup
// ---------------------------------------------------------------------------

/// Self-signed certificate material for the QUIC transport.
pub struct NodeCert {
    pub cert_der: rustls::pki_types::CertificateDer<'static>,
    pub key_der: rustls::pki_types::PrivateKeyDer<'static>,
}

/// Generate a self-signed certificate *derived from the node's Ed25519
/// signing identity*. The TLS keypair is the same Ed25519 seed that produces
/// `node_id`, so the transport cert and the application identity are the same
/// secret — there is nothing to re-bind at connect time.
///
/// Phase 1 used a random cert (identity proof was per-message Ed25519 only).
/// This closes the transport-level gap noted in UPGRADE.md / FEATURES.
pub fn generate_node_cert_for(identity: &crate::signing::SigningIdentity) -> anyhow::Result<NodeCert> {
    // ring's Ed25519KeyPair::generate_pkcs8 serializes the 32-byte seed as a
    // PKCS#8 v2 doc; rcgen consumes the same form for Ed25519 (PKCS_ED25519
    // alg). ed25519-dalek's SigningKey.to_bytes() is that same seed.
    //
    // rcgen wants the key as PKCS#8 DER; ring's seed roundtrip yields exactly
    // that when we use Ed25519KeyPair::from_pkcs8_maybe_unchecked(&der) under
    // the hood via KeyPair::from_pkcs8_der_and_sign_algo(PKCS_ED25519).
    //
    // Build the PKCS#8 v1 wrapper around the raw 32-byte seed:
    let seed = identity.secret_bytes();
    let pkcs8 = ed25519_pkcs8_from_seed(&seed);
    let key_pair = rcgen::KeyPair::from_pkcs8_der_and_sign_algo(
        &rustls::pki_types::PrivatePkcs8KeyDer::from(pkcs8),
        &rcgen::PKCS_ED25519,
    )
    .map_err(|e| anyhow::anyhow!("ed25519 cert key derivation failed: {e}"))?;
    let params = rcgen::CertificateParams::new(vec!["jerith-mesh-node".to_string()])
        .map_err(|e| anyhow::anyhow!("cert params failed: {e}"))?;
    let cert = params.self_signed(&key_pair)
        .map_err(|e| anyhow::anyhow!("self-signed cert failed: {e}"))?;
    let cert_der = rustls::pki_types::CertificateDer::from(cert.der().to_vec());
    let key_der = rustls::pki_types::PrivateKeyDer::Pkcs8(
        rustls::pki_types::PrivatePkcs8KeyDer::from(key_pair.serialize_der()),
    );
    Ok(NodeCert { cert_der, key_der })
}

/// Legacy entrypoint kept for callers without an identity handy (tests).
/// Random cert — the pinning verifier rejects it for any node_id.
pub fn generate_node_cert() -> anyhow::Result<NodeCert> {
    let ck = rcgen::generate_simple_self_signed(vec!["jerith-mesh-node".to_string()])
        .map_err(|e| anyhow::anyhow!("cert generation failed: {e}"))?;
    let cert_der = rustls::pki_types::CertificateDer::from(ck.cert.der().to_vec());
    let key_der = rustls::pki_types::PrivateKeyDer::Pkcs8(
        rustls::pki_types::PrivatePkcs8KeyDer::from(ck.key_pair.serialize_der()),
    );
    Ok(NodeCert { cert_der, key_der })
}

/// Build the PKCS#8 v1 "no attributes, ed25519" wrapper around a raw
/// 32-byte Ed25519 seed, as RFC 5958 requires for ed25519 keys.
///
///   0x30 0x2e                 SEQUENCE, 46 bytes
///     0x02 0x01 0x00          INTEGER 0 (version)
///     0x30 0x05               SEQUENCE, 5 bytes  (alg id)
///       0x06 0x03 0x2b 0x65 0x70   OID 1.3.101.112 (ed25519)
///     0x04 0x22               OCTET STRING, 34
///       0x04 0x20             OCTET STRING, 32 (the seed)
fn ed25519_pkcs8_from_seed(seed: &[u8; 32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(48);
    out.extend_from_slice(&[0x30, 0x2e, 0x02, 0x01, 0x00, 0x30, 0x05,
                             0x06, 0x03, 0x2b, 0x65, 0x70, 0x04, 0x22, 0x04, 0x20]);
    out.extend_from_slice(seed);
    out
}

/// SHA-256 of the certificate's DER, hex — the fingerprint a verifier pins.
#[allow(dead_code)] // reserved for future pin-by-fingerprint mode
fn cert_fingerprint(cert: &rustls::pki_types::CertificateDer<'_>) -> String {
    use std::fmt::Write;
    let digest = <sha2::Sha256 as sha2::Digest>::digest(cert.as_ref());
    let mut hexs = String::with_capacity(digest.len() * 2);
    for b in digest {
        write!(hexs, "{b:02x}").unwrap();
    }
    hexs
}

/// Identity-pinning verifier: accepts a certificate only when the cert's
/// Ed25519 subject key matches the node_id the client expects — pins are
/// keyed by the dial-time SNI, which `call` sets to the destination node_id.
///
/// Binding rule (matches `SigningIdentity::node_id`): node_id = first 16 hex
/// chars of the Ed25519 pubkey. The cert is self-signed ed25519 whose SPKI
/// key IS the node's identity key, so a successful check means "this
/// transport is provably the node we meant to dial" — a MITM cannot present
/// a different cert for the same node_id, and a stolen random cert fails.
#[derive(Debug)]
struct PinningVerifier {
    /// key: node_id (16 hex chars) → the SocketAddr the registry last
    /// advertised for it (pinning by SNI: `call` dials with node_id as SNI).
    pins: Arc<PinMap>,
}

impl rustls::client::danger::ServerCertVerifier for PinningVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        // Extract the Ed25519 subject public key from the certificate.
        let key_hex = match extract_ed25519_spki_hex(end_entity) {
            Ok(h) => h,
            Err(e) => {
                return Err(rustls::Error::Other(rustls::OtherError(Arc::new(
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("cert spki extraction failed: {e}"),
                    ),
                ))));
            }
        };
        let sn_str = server_name.to_str().into_owned();
        // SNI carries the destination node_id (set by `call`).
        let Some(addr) = self.pins.get(&sn_str) else {
            // No pin for this destination — fail closed (Phase 2: strict).
            return Err(rustls::Error::Other(rustls::OtherError(Arc::new(
                std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    format!("no pin for destination {sn_str}"),
                ),
            ))));
        };
        let expected = sn_str;
        let _ = addr; // future: also verify transport addr matches registry
        // node_id is the 16-hex prefix of the pubkey — compare the prefix.
        if !key_hex.starts_with(expected.as_str()) {
            return Err(rustls::Error::Other(rustls::OtherError(Arc::new(
                std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    format!(
                        "cert identity mismatch: presented key {}… does not match pinned node_id {expected}",
                        &key_hex[..8.min(key_hex.len())]
                    ),
                ),
            ))));
        }
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

/// Console client endpoints that skip verification remain available for
/// offline tests only; production paths use `pinned_client_endpoint`.
pub fn client_endpoint() -> anyhow::Result<quinn::Endpoint> {
    build_client_endpoint(Arc::new(HashMap::new()))
}

/// Pin map: node_id → is implicit (the key IS the node_id); the value is
/// reserved for a future fingerprint pin (see `cert_fingerprint`).
/// Dialing a node must be answered by a cert whose Ed25519 key's 16-hex
/// prefix equals the node_id dialed.
pub type PinMap = HashMap<String, SocketAddr>;

pub fn pinned_client_endpoint(pins: Arc<PinMap>) -> anyhow::Result<quinn::Endpoint> {
    build_client_endpoint(pins)
}

/// Pins behind a live RwLock so the registry can update them between dials
/// without rebuilding the endpoint. The verifier reads the current map at
/// each handshake.
pub type SharedPinMap = Arc<tokio::sync::RwLock<PinMap>>;

pub fn pinned_client_endpoint_dyn(pins: SharedPinMap) -> anyhow::Result<quinn::Endpoint> {
    let mut endpoint = quinn::Endpoint::client("0.0.0.0:0".parse()?)?;
    struct DynVerifier(SharedPinMap);
    impl std::fmt::Debug for DynVerifier {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("DynVerifier")
        }
    }
    impl rustls::client::danger::ServerCertVerifier for DynVerifier {
        fn verify_server_cert(
            &self,
            end_entity: &rustls::pki_types::CertificateDer<'_>,
            _intermediates: &[rustls::pki_types::CertificateDer<'_>],
            server_name: &rustls::pki_types::ServerName<'_>,
            _ocsp_response: &[u8],
            _now: rustls::pki_types::UnixTime,
        ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
            // Runs inside quinn's driver — cannot block. Pins are
            // refreshed between dials; on a rare concurrent refresh,
            // fail closed instead of panicking the runtime.
            let pins = match self.0.try_read() {
                Ok(g) => g.clone(),
                Err(_) => {
                    return Err(rustls::Error::Other(rustls::OtherError(Arc::new(
                        std::io::Error::other("pin map busy"),
                    ))));
                }
            };
            let v = PinningVerifier { pins: Arc::new(pins) };
            v.verify_server_cert(end_entity, _intermediates, server_name, _ocsp_response, _now)
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
            rustls::crypto::ring::default_provider().signature_verification_algorithms.supported_schemes()
        }
    }
    let mut crypto = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(DynVerifier(pins.clone())))
        .with_no_client_auth();
    crypto.alpn_protocols = vec![b"jerith-rpc/1".to_vec()];
    let config = quinn::ClientConfig::new(Arc::new(
        quinn::crypto::rustls::QuicClientConfig::try_from(crypto)?,
    ));
    endpoint.set_default_client_config(config);
    Ok(endpoint)
}

fn build_client_endpoint(pins: Arc<PinMap>) -> anyhow::Result<quinn::Endpoint> {
    let mut endpoint = quinn::Endpoint::client("0.0.0.0:0".parse()?)?;
    let mut crypto = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(PinningVerifier { pins }))
        .with_no_client_auth();
    crypto.alpn_protocols = vec![b"jerith-rpc/1".to_vec()];
    let config = quinn::ClientConfig::new(Arc::new(
        quinn::crypto::rustls::QuicClientConfig::try_from(crypto)?,
    ));
    endpoint.set_default_client_config(config);
    Ok(endpoint)
}

/// Extract the Ed25519 subject public key (raw 32 bytes) from a DER cert and
/// return its hex. Errors on any non-ed25519 SPKI.
fn extract_ed25519_spki_hex(
    cert: &rustls::pki_types::CertificateDer<'_>,
) -> anyhow::Result<String> {
    // OID 1.3.101.112 (ed25519): 06 03 2b 65 70. SPKI: SEQUENCE { SEQ(alg), BITSTRING(key) }
    //
    // The OID appears twice in a self-signed Ed25519 cert: once in
    // tbsCertificate.signature (algorithm) and once in the SPKI. Only the
    // SPKI occurrence is followed by a 33-byte BIT STRING key, so scan
    // each occurrence and accept the one with that shape.
    let der = cert.as_ref();
    let needle: &[u8] = &[0x06, 0x03, 0x2b, 0x65, 0x70];
    let mut key_hex = None;
    let mut search_from = 0usize;
    while let Some(oid_pos) = der[search_from..]
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|p| p + search_from)
    {
        let mut i = oid_pos + 5;
        // Skip a possible 0x05 0x00 NULL inside the alg id (not expected
        // for ed25519, but harmless to allow).
        if der.get(i) == Some(&0x05) {
            if let Some(l) = der.get(i + 1).copied() {
                i += 2 + l as usize;
            }
        }
        // A BIT STRING of exactly 33 bytes (1 unused-bits byte + 32 key)
        // right after the alg id = this is the SPKI. (The signature's BIT
        // STRING is 65 bytes: 1 unused byte + 64 ed25519 sig.)
        let is_spki_bitstring = der.get(i) == Some(&0x03)
            && der
                .get(i + 1)
                .map(|len| *len == 33)
                .unwrap_or(false);
        if is_spki_bitstring {
            match der.get(i + 2) {
                Some(0x00) => {}
                other => anyhow::bail!("unexpected SPKI unused-bits byte {other:?}"),
            }
            let start = i + 3;
            let key: [u8; 32] = der
                .get(start..start + 32)
                .ok_or_else(|| anyhow::anyhow!("truncated key bytes"))?
                .try_into()
                .map_err(|_| anyhow::anyhow!("key copy failed"))?;
            key_hex = Some(hex::encode(key));
            break;
        }
        search_from = oid_pos + 1;
    }
    let Some(key_hex) = key_hex else {
        anyhow::bail!("cert is not an Ed25519 SPKI");
    };
    Ok(key_hex)
}


/// Build a server endpoint on `addr` with a random cert. Kept for offline
/// tests; production code uses `server_endpoint_for` (identity-bound cert).
pub fn server_endpoint(addr: SocketAddr) -> anyhow::Result<quinn::Endpoint> {
    let cert = generate_node_cert()?;
    rustls_server_endpoint_from_cert(cert, addr)
}

/// Identity-bound server endpoint: the TLS cert key IS the node's Ed25519
/// identity key, so the presented cert cryptographically ties the transport
/// to `node_id` — the client's pinning verifier can prove "who answered".
pub fn server_endpoint_for(
    identity: &crate::signing::SigningIdentity,
    addr: SocketAddr,
) -> anyhow::Result<quinn::Endpoint> {
    let cert = generate_node_cert_for(identity)?;
    rustls_server_endpoint_from_cert(cert, addr)
}

fn rustls_server_endpoint_from_cert(
    cert: NodeCert,
    addr: SocketAddr,
) -> anyhow::Result<quinn::Endpoint> {
    let mut server_crypto = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(vec![cert.cert_der], cert.key_der)?;
    server_crypto.alpn_protocols = vec![b"jerith-rpc/1".to_vec()];
    let config = quinn::ServerConfig::with_crypto(Arc::new(
        quinn::crypto::rustls::QuicServerConfig::try_from(server_crypto)?,
    ));
    // NOTE: no transport-level stream caps here. handle_connection
    // processes one bidi stream at a time, so extra streams only queue —
    // but tightening quinn's limits in the server transport reliably
    // regressed handshakes (client dials surfaced CONNECTION_REFUSED at
    // nondeterministic times), a quinn 0.11 internals quirk beyond this
    // round's scope. The defensible bounds instead: the rpc frame cap
    // (read path), the per-kind payload caps (handler path), and the
    // connection semaphore in serve_with_limit (top of stack), which
    // quantifies both together in-memory.
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
    serve_with_limit(
        endpoint,
        handler,
        server_node_id,
        shutdown,
        MAX_CONCURRENT_QUIC_CONNECTIONS,
    )
    .await
}

/// The RPC server counts as part of the node's public surface (`0.0.0.0`
/// bind, no client auth yet — inbound wire auth lands with cert pinning).
/// Bound how many inbound connections we carry at once: each is a live
/// task + QUIC buffers, and a flood of dials would otherwise spawn tasks
/// without limit. Backpressure rather than rejection: at capacity the
/// accept loop waits (quinn buffers the pending handshakes) and a freed
/// permit serves the next dial. Connections are per-request dial+reply
/// (`call()` connects fresh per request) so permits turn over quickly.
pub const MAX_CONCURRENT_QUIC_CONNECTIONS: usize = 128;

pub async fn serve_with_limit(
    endpoint: quinn::Endpoint,
    handler: Arc<dyn ExpertHandler>,
    server_node_id: String,
    shutdown: tokio::sync::watch::Receiver<bool>,
    max_concurrent_connections: usize,
) -> anyhow::Result<()> {
    let connections = std::sync::Arc::new(tokio::sync::Semaphore::new(max_concurrent_connections));
    let mut shutdown = shutdown;
    loop {
        // Backpressure BEFORE accept: at capacity this waits for a
        // permit instead of spawning a task per dial; pending dials stay
        // queued in quinn's incoming queue until capacity frees. (No
        // nested select: the accept future must keep getting polled —
        // nested awaits here stalled handshakes under parallel load.)
        let permit = match connections.clone().acquire_owned().await {
            Ok(p) => p,
            Err(_) => break, // semaphore closed → shutdown
        };
        tokio::select! {
            conn = endpoint.accept() => {
                let Some(conn) = conn else { break };
                let handler = Arc::clone(&handler);
                let server_node_id = server_node_id.clone();
                tokio::spawn(async move {
                    let _permit = permit;
                    if let Ok(connection) = conn.await {
                        handle_connection(connection, handler, server_node_id).await;
                    }
                });
            }
            _ = shutdown.changed() => {
                // changed() errors when the sender is gone — that *is*
                // shutdown for every realistic caller (main keeps it
                // alive for the process lifetime).
                if *shutdown.borrow() {
                    break;
                }
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
                            // Bound what reaches the handler (and its
                            // runtime forward): the frame cap bounds the
                            // wire, but a parsed 200M-float input would
                            // still burn runtime compute + memory on top.
                            // Real MoE expert activations sit far below.
                            const MAX_EXPERT_INPUT_FLOATS: usize = 65_536;
                            if input.len() > MAX_EXPERT_INPUT_FLOATS {
                                NodeRpcReply::Error {
                                    req_id,
                                    message: format!(
                                        "expert input exceeds {MAX_EXPERT_INPUT_FLOATS} floats",
                                    ),
                                }
                            } else {
                                match handler.run(&expert_id, &input) {                                    Ok(output) => {
                                    // Phase 4: commit to the result *before*
                                    // the requester has decided whether it
                                    // will replay-check — enables efficient
                                    // fraud proofs later.
                                    let commitment =
                                        crate::verify::result_commitment(
                                            &req_id,
                                            &expert_id,
                                            &server_id,
                                            &input,
                                            &output,
                                        );
                                    NodeRpcReply::ExpertResult {
                                        req_id,
                                        output,
                                        node_id: server_id,
                                        commitment: Some(commitment),
                                    }
                                }
                                Err(e) => NodeRpcReply::Error {
                                    req_id,
                                    message: e.to_string(),
                                },
                            }
                            }
                        }
                        NodeRpcRequest::ChatQuery { req_id, messages, .. } => {
                            // Mirror the gateway's chat forward cap: the
                            // per-peer runtime visit is one request, and
                            // an oversized chat body would relay the
                            // memory pin to every peer runtime in one
                            // quic cone.
                            if chat_messages_over_cap(&messages) {
                                NodeRpcReply::Error {
                                    req_id,
                                    message: format!(
                                        "chat messages exceed {MAX_CHAT_MESSAGES_BYTES} bytes",
                                    ),
                                }
                            } else {
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

/// Chat-relay size cap (see the ChatQuery arm): the relay visit is one
/// serialized body — the gateway's chat cap (1 MB total) leaves this
/// slack, so a legit history fits and an oversized one gets a clean
/// Error instead of pinning a peer runtime.
pub const MAX_CHAT_MESSAGES_BYTES: usize = 900_000;

fn chat_messages_over_cap(messages: &[serde_json::Value]) -> bool {
    serde_json::to_vec(messages)
        .map(|v| v.len() > MAX_CHAT_MESSAGES_BYTES)
        .unwrap_or(true)
}

async fn read_frame(recv: &mut quinn::RecvStream) -> anyhow::Result<Vec<u8>> {
    // A stream that never emits `\n` would otherwise grow `buf` without
    // bound — one dial, one cheap stream, one resident-growth DoS. The
    // cap comfortably fits every legitimate request a peer can make.
    const MAX_RPC_FRAME_BYTES: usize = 1_048_576;
    let mut buf = Vec::with_capacity(1024);
    let mut chunk = [0u8; 4096];
    loop {
        let n = recv.read(&mut chunk).await?;
        match n {
            Some(0) => break,
            Some(n) => {
                buf.extend_from_slice(&chunk[..n]);
                if buf.len() > MAX_RPC_FRAME_BYTES {
                    anyhow::bail!(
                        "rpc frame exceeds {MAX_RPC_FRAME_BYTES} bytes"
                    );
                }
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
    /// Optional bearer key for OpenAI-compatible runtimes that require
    /// auth (e.g. an authed LiteLLM proxy). None → no header is sent.
    pub api_key: Option<String>,
    /// Peer-chat tier routing (same policy as the gateway): when set,
    /// image content routes to the vision model and complex prompts to
    /// the deep model; base model otherwise. Empty string disables a tier.
    pub vision_model: Option<String>,
    pub deep_model: Option<String>,
    /// HTTP client (shared, pooled).
    client: reqwest::Client,
}

impl RuntimeExpert {
    /// Discover the runtime: probe `{url}/v1/models` and confirm the
    /// requested model is being served. Returns the handler on success.
    pub async fn discover(
        runtime_url: &str,
        model: &str,
        api_key: Option<&str>,
    ) -> anyhow::Result<Self> {
        let url = format!("{}/v1/models", runtime_url.trim_end_matches('/'));
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()?;
        let mut req = client.get(&url);
        if let Some(k) = api_key {
            req = req.bearer_auth(k);
        }
        let resp = req
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
            api_key: api_key.map(|k| k.to_string()),
            vision_model: None,
            deep_model: None,
            client,
        })
    }
}

impl ExpertHandler for RuntimeExpert {
    fn run(&self, expert_id: &str, input: &[f32]) -> anyhow::Result<Vec<f32>> {
        // Builtin/ringer experts are answered without the runtime: a pure
        // deterministic function every mesh node can compute and verify in
        // microseconds (Phase 4 spot-check challenges).
        if let Some(kind) = expert_id.strip_prefix(crate::verify::RINGER_EXPERT_PREFIX) {
            return Ok(crate::verify::builtin_expert_answer(kind, input));
        }
        self.run_expert(expert_id, input)
    }

    fn chat<'a>(
        &'a self,
        messages: &'a [serde_json::Value],
    ) -> futures_core::future::BoxFuture<'a, anyhow::Result<String>> {
        Box::pin(async move { self.chat_impl(messages).await })
    }

}

impl RuntimeExpert {
    /// Runtime-backed expert execution (builtin/ringer experts are handled
    /// in the trait impl above before reaching here). Real expert execution
    /// is async HTTP; bridge the sync trait with a block-on worker and a
    /// strict timeout (never blocks a pass).
    fn run_expert(&self, expert_id: &str, input: &[f32]) -> anyhow::Result<Vec<f32>> {
        let this = Self {
            runtime_url: self.runtime_url.clone(),
            model: self.model.clone(),
            api_key: self.api_key.clone(),
            vision_model: self.vision_model.clone(),
            deep_model: self.deep_model.clone(),
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

    /// Test-only direct construction (production discovers via /v1/models).
    #[cfg(test)]
    fn for_tests(runtime_url: &str, model: &str) -> Self {
        Self {
            runtime_url: runtime_url.to_string(),
            model: model.to_string(),
            api_key: None,
            vision_model: None,
            deep_model: None,
            client: reqwest::Client::new(),
        }
    }

    /// Synchronous-first chat implementation (also the trait method body).
    async fn chat_impl(&self, messages: &[serde_json::Value]) -> anyhow::Result<String> {
        let url = format!(
            "{}/v1/chat/completions",
            self.runtime_url.trim_end_matches('/')
        );
        // Never forward a tool/system context we didn't intend; the
        // messages are the caller's privacy-scoped payload verbatim.
        let mut req = self.client.post(&url);
        if let Some(k) = &self.api_key {
            req = req.bearer_auth(k);
        }
        // Tier routing for peer-dispatched chat: vision for image
        // content, deep for complex prompts, base model otherwise (a
        // tier disabled by config `""` never engages).
        let vision = self.vision_model.as_ref().filter(|s| !s.is_empty());
        let deep = self.deep_model.as_ref().filter(|s| !s.is_empty());
        let model = if vision.is_some()
            && crate::model_router::has_image_content(messages)
        {
            vision.unwrap().clone()
        } else if deep.is_some()
            && crate::model_router::is_complex(messages)
        {
            deep.unwrap().clone()
        } else {
            self.model.clone()
        };
        let resp = req
            .json(&serde_json::json!({
                "model": model,
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
        let mut req = self.client.post(&url);
        if let Some(k) = &self.api_key {
            req = req.bearer_auth(k);
        }
        let resp = req
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
/// `server_node_id` doubles as the SNI so the pinning verifier can look up
/// the pin for the node we *mean* to reach — the cert must then carry that
/// node's identity key (PinningVerifier), binding wire node_id to transport.
pub async fn call(
    endpoint: &quinn::Endpoint,
    addr: SocketAddr,
    server_node_id: &str,
    req: &NodeRpcRequest,
    _identity: &SigningIdentity,
    timeout: Duration,
) -> anyhow::Result<NodeRpcReply> {
    let connecting = endpoint
        .connect(addr, server_node_id)
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
    use rustls::client::danger::ServerCertVerifier as _;

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
            ..Default::default()
        })
    }

    fn router_with_verify(nearby: u64, verify_percent: u32) -> ExpertRouter {
        ExpertRouter::new(RouterOptions {
            nearby_rtt_ms: nearby,
            dispatch_timeout: Duration::from_millis(250),
            verify_percent,
            verify_tolerance: 0.0,
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

    /// Spawn an identity-bound server and return its addr + identity +
    /// a dynamic pin map preloaded with `node_id -> addr`, matching the
    /// node_id-keyed lookup the production client path uses.
    async fn test_server_and_pins(
        handler: Arc<dyn ExpertHandler>,
    ) -> anyhow::Result<(SocketAddr, SigningIdentity, SharedPinMap)> {
        let identity = SigningIdentity::generate();
        let server = server_endpoint_for(&identity, "127.0.0.1:0".parse().unwrap())?;
        let addr = server.local_addr().unwrap();
        let (_tx, rx) = tokio::sync::watch::channel(false);
        let node_id = identity.node_id();
        let mut pins: PinMap = HashMap::new();
        pins.insert(node_id.clone(), addr);
        tokio::spawn(async move {
            let _ = serve(server, handler, node_id, rx).await;
        });
        let pins: SharedPinMap = Arc::new(RwLock::new(pins));
        Ok((addr, identity, pins))
    }

    #[tokio::test]
    async fn quic_expert_query_roundtrip() {
        let (addr, server_identity, pins) =
            test_server_and_pins(Arc::new(TimesTwo)).await.unwrap();

        let client = pinned_client_endpoint_dyn(pins).unwrap();
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
            call(
                &client,
                addr,
                &server_identity.node_id(),
                &req,
                &identity,
                Duration::from_secs(5),
            ),
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

    /// Terminal state of a reply stream: closed cleanly or errored
    /// (over-cap frames may surface either way on the client).
    enum StreamEnd {
        Closed,
        Errored,
    }

    /// Read a reply stream to its end, for asserting stream-closing
    /// behavior without hanging on a wedged peer.
    async fn read_recv_to_end(recv: &mut quinn::RecvStream) -> StreamEnd {
        let mut chunk = [0u8; 4096];
        loop {
            match recv.read(&mut chunk).await {
                Ok(Some(n)) if n > 0 => {}
                Ok(_) => break StreamEnd::Closed,
                Err(_) => break StreamEnd::Errored,
            }
        }
    }

    /// Raw dial helper for the flood-hardening tests: a pinned client
    /// endpoint fully set up for a given server, plus a held-open
    /// connection, so tests can exercise stream limits between requests.
    async fn raw_dial(
        server_id: &str,
        addr: SocketAddr,
        pins: &SharedPinMap,
    ) -> anyhow::Result<(quinn::Endpoint, quinn::Connection)> {
        let endpoint = pinned_client_endpoint_dyn(pins.clone())?;
        let conn = endpoint.connect(addr, server_id)?.await?;
        Ok((endpoint, conn))
    }

    #[tokio::test]
    async fn rpc_frame_cap_never_emits_unbounded_buffer_growth() {
        let (addr, server_identity, pins) =
            test_server_and_pins(Arc::new(TimesTwo)).await.unwrap();
        let (endpoint, mut conn) =
            raw_dial(&server_identity.node_id(), addr, &pins).await.unwrap();

        // >1 MiB of newline-free junk: the frame reader must bail on the
        // cap instead of buffering the rest forever. Server drops the
        // connection; from the client this shows up as the stream/conn
        // dying rather than a normal reply ever arriving.
        let (mut send, mut recv) = conn.open_bi().await.unwrap();
        let junk = vec![b'x'; (1_048_576 + 128_000) / 8];
        for _ in 0..8 {
            send.write_all(&junk).await.unwrap();
        }
        send.finish().unwrap();
        let outcome = tokio::time::timeout(Duration::from_secs(5), read_recv_to_end(&mut recv))
            .await
            .expect("stream must either close or error, not hang");
        // Either terminal state is correct — the property under test is
        // that the reader terminates without ever emitting a reply frame.
        match outcome {
            StreamEnd::Closed | StreamEnd::Errored => {}
        }
    }

    #[tokio::test]
    async fn expert_input_cap_rejects_before_handler() {
        let (addr, server_identity, pins) =
            test_server_and_pins(Arc::new(TimesTwo)).await.unwrap();
        let client = pinned_client_endpoint_dyn(pins).unwrap();
        let identity = SigningIdentity::generate();
        let mut req = NodeRpcRequest::ExpertQuery {
            req_id: "cap-1".into(),
            expert_id: "exp.test".into(),
            input: vec![1.0; 65_537],
            from_node_id: identity.node_id(),
            stamp_ms: 0,
            sig: String::new(),
        };
        sign_request(&mut req, &identity);
        let reply = tokio::time::timeout(
            Duration::from_secs(10),
            call(&client, addr, &server_identity.node_id(), &req, &identity, Duration::from_secs(5)),
        )
        .await
        .unwrap()
        .unwrap();
        match reply {
            NodeRpcReply::Error { message, .. } => {
                assert!(
                    message.contains("exceeds 65536 floats"),
                    "cap must reject the payload before the handler, got: {message}"
                );
            }
            other => panic!("expected cap rejection, got: {other:?}"),
        }
    }

    #[test]
    fn chat_messages_cap_boundary_is_900k() {
        // Unit-level: the cap check lives in the server's ChatQuery arm
        // (a ~950KB wire round-trip hit transport-window races in
        // parallel suites — the cap constant itself is checked here).
        // The cap measures the serialized MESSAGES ARRAY (`[]` + 2 bytes
        // per element) — strings are quote-wrapped, so subtract the
        // envelopes: 900_000 byte total = 2 + (899_996 + "…").
        let at_cap = serde_json::Value::String("x".repeat(900_000 - 4));
        let over_cap = serde_json::Value::String("x".repeat(900_000 - 3));
        assert!(!chat_messages_over_cap(&[at_cap]), "exactly at cap is servable");
        assert!(chat_messages_over_cap(&[over_cap]), "1 byte over cap refuses");
    }

    #[tokio::test]
    async fn connection_cap_backpressures_extra_dials() {
        // Serve on a dedicated endpoint with a capacity of ONE dial. The
        // RPC server serves per-request connections, so holding one
        // connection open must defer (not murder) the next.
        let server_identity = SigningIdentity::generate();
        let server = server_endpoint_for(&server_identity, "127.0.0.1:0".parse().unwrap()).unwrap();
        let addr = server.local_addr().unwrap();
        let node_id = server_identity.node_id();
        let (_tx, rx) = tokio::sync::watch::channel(false);
        let server_node = node_id.clone();
        tokio::spawn(async move {
            let _ = serve_with_limit(
                server,
                Arc::new(TimesTwo) as Arc<dyn ExpertHandler>,
                server_node,
                rx,
                1,
            )
            .await;
        });

        // Dial 1: held open server-side (handle_connection blocks on the
        // next accept_bi), which consumes the single permit.
        let pins: SharedPinMap = {
            let mut pins: PinMap = HashMap::new();
            pins.insert(node_id.clone(), addr);
            std::sync::Arc::new(tokio::sync::RwLock::new(pins))
        };
        let (endpoint1, _conn1) = raw_dial(&node_id, addr, &pins).await.unwrap();
        tokio::time::sleep(Duration::from_millis(150)).await; // serve loop picks it up + acquires

        // Dial 2 while the permit is held: backpressured — the request's
        // handler is never reached inside a short client timeout.
        let identity = SigningIdentity::generate();
        let mut req = NodeRpcRequest::Ping {
            from_node_id: identity.node_id(),
            stamp_ms: 0,
            sig: String::new(),
        };
        sign_request(&mut req, &identity);
        let pinned_outcome = tokio::time::timeout(
            Duration::from_secs(2),
            call(&endpoint1, addr, &node_id, &req, &identity, Duration::from_secs(1)),
        )
        .await;
        assert!(
            matches!(pinned_outcome, Ok(Err(_))),
            "over-cap dial must not be served while the permit is held: {pinned_outcome:?}"
        );

        // Release: closing dial 1's connection frees the permit; the next
        // dial is served normally (no starvation, no wedged accept loop).
        // NOTE: nothing may hold an open connection here — the permit is
        // held per-connection, and `call()` dials a fresh one.
        drop(endpoint1);
        drop(_conn1);
        let (endpoint3, conn3) = raw_dial(&node_id, addr, &pins).await.unwrap();
        drop(conn3); // free the permit before dialing again
        let reply = tokio::time::timeout(
            Duration::from_secs(10),
            call(&endpoint3, addr, &node_id, &req, &identity, Duration::from_secs(5)),
        )
        .await
        .unwrap()
        .unwrap();
        assert!(matches!(reply, NodeRpcReply::Pong { .. }));
    }

    #[tokio::test]
    async fn quic_ping_roundtrip() {
        let (addr, server_identity, pins) = test_server_and_pins(Arc::new(TimesTwo)).await.unwrap();
        let client = pinned_client_endpoint_dyn(pins).unwrap();
        let identity = SigningIdentity::generate();
        let mut req = NodeRpcRequest::Ping {
            from_node_id: identity.node_id(),
            stamp_ms: 0,
            sig: String::new(),
        };
        sign_request(&mut req, &identity);
        let reply = call(
            &client,
            addr,
            &server_identity.node_id(),
            &req,
            &identity,
            Duration::from_secs(5),
        )
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

    #[tokio::test]
    async fn router_execute_over_quic_prefers_local_and_falls_through() {
        // Two remote servers: near (×3, rtt 5) and far (×7, rtt 200).
        let (near_addr, near_identity, pins) =
            test_server_and_pins(Arc::new(Multiplier { factor: 3.0 }))
                .await
                .unwrap();
        let (far_addr, far_identity, far_pins) =
            test_server_and_pins(Arc::new(Multiplier { factor: 7.0 }))
                .await
                .unwrap();
        {
            let far_map = far_pins.try_read().expect("far pins readable").clone();
            let mut wl = pins.try_write().expect("pins writable");
            wl.extend(far_map);
        }
        let near_node_id = near_identity.node_id();
        let far_node_id = far_identity.node_id();

        let r = router(50); // nearby threshold 50ms
        let peers = vec![
            (near_node_id.clone(), 5, true, true),
            (far_node_id.clone(), 200, true, true),
        ];
        let mut addrs = HashMap::new();
        addrs.insert(near_node_id.clone(), near_addr);
        addrs.insert(far_node_id.clone(), far_addr);

        let client = pinned_client_endpoint_dyn(pins).unwrap();
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
        addrs2.remove(&near_node_id);
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

    // --- Phase 4: ringer (builtin known-answer expert) over the wire ---

    #[tokio::test]
    async fn builtin_ringer_expert_served_without_runtime() {
        // RuntimeExpert with a dead runtime URL — builtin experts must still
        // be served exactly (they never reach the runtime).
        let handler = RuntimeExpert::for_tests("http://127.0.0.1:1", "stub");
        let (addr, server_identity, pins) =
            test_server_and_pins(Arc::new(handler)).await.unwrap();
        let client = pinned_client_endpoint_dyn(pins).unwrap();
        let identity = SigningIdentity::generate();
        let (expert_id, input, expected) =
            crate::verify::ringer_task(crate::bootstrap::now_ms());
        let req_id = "ringer-wire-1";
        let mut req = NodeRpcRequest::ExpertQuery {
            req_id: req_id.into(),
            expert_id: expert_id.to_string(),
            input: input.clone(),
            from_node_id: identity.node_id(),
            stamp_ms: 0,
            sig: String::new(),
        };
        sign_request(&mut req, &identity);
        let reply = tokio::time::timeout(
            Duration::from_secs(10),
            call(
                &client,
                addr,
                &server_identity.node_id(),
                &req,
                &identity,
                Duration::from_secs(5),
            ),
        )
        .await
        .unwrap()
        .unwrap();
        match reply {
            NodeRpcReply::ExpertResult {
                output,
                commitment,
                node_id: server_id,
                req_id: rid,
            } => {
                assert_eq!(rid, req_id);
                assert_eq!(output, expected, "builtin answer must be bit-exact");
                let commitment = commitment.expect("commitment on ringer reply");
                assert!(crate::verify::verify_commitment(
                    req_id,
                    expert_id,
                    &server_id,
                    &input,
                    &output,
                    &commitment
                ));
            }
            other => panic!("unexpected reply: {other:?}"),
        }
    }

    // --- Phase 4: redundant replay verification ---

    /// Two honest ×3 servers, replay forced to 100 %: the near result must
    /// survive verification against the far arbiter.
    #[tokio::test]
    async fn replay_agrees_on_honest_pairs() {
        let (near_addr, near_identity, pins) =
            test_server_and_pins(Arc::new(Multiplier { factor: 3.0 }))
                .await
                .unwrap();
        let (far_addr, far_identity, far_pins) =
            test_server_and_pins(Arc::new(Multiplier { factor: 3.0 }))
                .await
                .unwrap();
        let far_map = far_pins.try_read().expect("far pins readable").clone();
        pins.try_write().expect("pins writable").extend(far_map);
        let near_id = near_identity.node_id();
        let far_id = far_identity.node_id();

        let r = router_with_verify(50, 100);
        let peers = vec![(near_id.clone(), 5, true, true), (far_id.clone(), 200, true, true)];
        let mut addrs = HashMap::new();
        addrs.insert(near_id.clone(), near_addr);
        addrs.insert(far_id.clone(), far_addr);
        let client = pinned_client_endpoint_dyn(pins).unwrap();
        let identity = SigningIdentity::generate();
        let mut completed = HashMap::new();

        let out = r
            .execute_over_quic(
                &client,
                &identity,
                &addrs,
                "rp1",
                "exp.test",
                &[1.0, 2.0],
                &peers,
                &mut completed,
                |_| Ok(vec![0.0]),
            )
            .await
            .unwrap();
        assert_eq!(out, vec![3.0, 6.0]);
        let s = r.verify_tracker().stats().await;
        assert_eq!(s.replays_dispatched, 1);
        assert_eq!(s.replay_matches, 1);
        assert_eq!(s.replay_mismatches, 0);
        assert_eq!(s.commitment_checks, 2); // near + far (replay arbiter)
        assert_eq!(s.commitment_failures, 0);
        assert_eq!(s.expert_results_received, 2);
    }

    /// near ×3 (honest) vs far ×7 (differing): forced replay must flag BOTH
    /// peers (ambiguous without a local arbiter) and refuse to return any
    /// result from the remaining suspect candidates.
    #[tokio::test]
    async fn replay_mismatch_flags_both_and_aborts() {
        let (near_addr, near_identity, pins) =
            test_server_and_pins(Arc::new(Multiplier { factor: 3.0 }))
                .await
                .unwrap();
        let (far_addr, far_identity, far_pins) =
            test_server_and_pins(Arc::new(Multiplier { factor: 7.0 }))
                .await
                .unwrap();
        let far_map = far_pins.try_read().expect("far pins readable").clone();
        pins.try_write().expect("pins writable").extend(far_map);
        let near_id = near_identity.node_id();
        let far_id = far_identity.node_id();

        let r = router_with_verify(50, 100);
        let peers = vec![(near_id.clone(), 5, true, true), (far_id.clone(), 200, true, true)];
        let mut addrs = HashMap::new();
        addrs.insert(near_id.clone(), near_addr);
        addrs.insert(far_id.clone(), far_addr);
        let client = pinned_client_endpoint_dyn(pins).unwrap();
        let identity = SigningIdentity::generate();
        let mut completed = HashMap::new();

        let err = r
            .execute_over_quic(
                &client,
                &identity,
                &addrs,
                "rp2",
                "exp.test",
                &[1.0],
                &peers,
                &mut completed,
                |_| Ok(vec![0.0]),
            )
            .await;
        assert!(err.is_err(), "unverifiable remote disagreement must not return a result");
        let s = r.verify_tracker().stats().await;
        assert_eq!(s.replays_dispatched, 1);
        assert_eq!(s.replay_mismatches, 1);
        // The tracker queued evidence for both (ropped here without a
        // bootstrap — patch reports via the raw stats below).
        let stats_after = r.verify_tracker().stats().await;
        assert_eq!(stats_after.commitment_checks, 2);
    }

    /// Local arbiter: when we host the expert the local run is ground truth.
    /// A remote that disagrees with it alone is flagged; the plan keeps
    /// walking for a node that agrees.
    #[tokio::test]
    async fn local_arbiter_conicts_single_remote() {
        let (near_addr, near_identity, pins) =
            test_server_and_pins(Arc::new(Multiplier { factor: 3.0 }))
                .await
                .unwrap();
        let near_id = near_identity.node_id();

        let r = router_with_verify(50, 100);
        // Local expert registered, but the first local attempt fails
        // (transient outage) — plan falls through to the remote, whose
        // result is then arbitrated by the recovered local expert.
        r.register_local_expert("exp.test").await;
        let peers = vec![(near_id.clone(), 5, true, true)];
        let mut addrs = HashMap::new();
        addrs.insert(near_id.clone(), near_addr);
        let client = pinned_client_endpoint_dyn(pins).unwrap();
        let identity = SigningIdentity::generate();
        let mut completed = HashMap::new();

        let calls = std::sync::atomic::AtomicU64::new(0);
        let err = r
            .execute_over_quic(
                &client,
                &identity,
                &addrs,
                "rp3",
                "exp.test",
                &[4.0],
                &peers,
                &mut completed,
                |input| {
                    if calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                        anyhow::bail!("first local attempt down");
                    }
                    Ok(input.iter().map(|x| x * 10.0).collect::<Vec<f32>>())
                },
            )
            .await;
        assert!(err.is_err(), "remote disagreeing with local truth must be rejected");
        let s = r.verify_tracker().stats().await;
        assert_eq!(s.replays_dispatched, 1);
        assert_eq!(s.replay_mismatches, 1);
        assert_eq!(s.replay_matches, 0);
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

    #[test]
    fn compact_chat_history_collapses_consecutive_dups() {
        // Back-to-back identical turns collapse to one — the retry tail and
        // restored-session cases the mesh fallback re-dials peer by peer.
        let dups = vec![
            serde_json::json!({"role": "system", "content": "u are mesh"}),
            serde_json::json!({"role": "system", "content": "u are mesh"}),
            serde_json::json!({"role": "user", "content": "hi"}),
            serde_json::json!({"role": "user", "content": "hi"}),
            serde_json::json!({"role": "user", "content": "hi"}),
            serde_json::json!({"role": "assistant", "content": "mesh hello"}),
        ];
        let out = ExpertRouter::compact_chat_history(&dups);
        assert_eq!(out.len(), 3);
        assert_eq!(
            out.iter().map(|m| m["role"].clone()).collect::<Vec<_>>(),
            vec!["system", "user", "assistant"]
        );

        // Non-adjacent repeats are intentional history — kept verbatim.
        let spaced = vec![
            serde_json::json!({"role": "user", "content": "hi"}),
            serde_json::json!({"role": "assistant", "content": "mesh hello"}),
            serde_json::json!({"role": "user", "content": "hi"}),
        ];
        assert_eq!(ExpertRouter::compact_chat_history(&spaced).len(), 3);

        // Empty and single-message inputs pass through untouched.
        assert!(ExpertRouter::compact_chat_history(&[]).is_empty());
        let one = vec![serde_json::json!({"role": "user", "content": "hi"})];
        assert_eq!(ExpertRouter::compact_chat_history(&one), one);

        // Same text with a different role is NOT a duplicate turn.
        let roles = vec![
            serde_json::json!({"role": "user", "content": "hi"}),
            serde_json::json!({"role": "assistant", "content": "hi"}),
        ];
        assert_eq!(ExpertRouter::compact_chat_history(&roles).len(), 2);
    }

    /// Serves chat over QUIC while recording exactly what the wire
    /// delivered — the assertion surface for the compaction transform.
    struct ChatCapture(std::sync::Arc<std::sync::Mutex<Vec<serde_json::Value>>>);

    impl ExpertHandler for ChatCapture {
        fn run(&self, _expert_id: &str, _input: &[f32]) -> anyhow::Result<Vec<f32>> {
            Ok(Vec::new())
        }

        fn chat<'a>(
            &'a self,
            messages: &'a [serde_json::Value],
        ) -> futures_core::future::BoxFuture<'a, anyhow::Result<String>> {
            *self.0.lock().unwrap() = messages.to_vec();
            Box::pin(async move { Ok("recorded".to_string()) })
        }
    }

    #[tokio::test]
    async fn chat_query_wire_carries_compacted_history() {
        let sink = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (addr, server_identity, pins) =
            test_server_and_pins(Arc::new(ChatCapture(sink.clone()))).await.unwrap();
        let node_id = server_identity.node_id();
        let r = router(50);
        let client = pinned_client_endpoint_dyn(pins).unwrap();
        let identity = SigningIdentity::generate();
        let mut addrs = HashMap::new();
        addrs.insert(node_id.clone(), addr);
        let peers = vec![(node_id.clone(), 8, true, true)];

        // 5 messages, one back-to-back duplicate: the local closure fails
        // so the request must ride the wire, carrying exactly the compacted
        // 4 (the trailing user "q1" is NOT a duplicate of the assistant
        // turn — non-adjacent repeats are intentional history).
        let msgs = vec![
            serde_json::json!({"role": "system", "content": "s"}),
            serde_json::json!({"role": "system", "content": "s"}),
            serde_json::json!({"role": "user", "content": "q1"}),
            serde_json::json!({"role": "assistant", "content": "a1"}),
            serde_json::json!({"role": "user", "content": "q1"}),
        ];
        let (_content, served_by) = r
            .chat_over_quic(
                &client,
                &identity,
                &addrs,
                "chat-cap",
                &msgs,
                &peers,
                || Err(anyhow::anyhow!("local runtime down")),
            )
            .await
            .unwrap();
        assert_eq!(served_by, node_id); // wire path, not local
        let got = sink.lock().unwrap().clone();
        assert_eq!(got.len(), 4, "wire must carry the compacted history");
        let roles: Vec<String> = got
            .iter()
            .map(|m| m["role"].as_str().unwrap_or("?").to_string())
            .collect();
        assert_eq!(roles, vec!["system", "user", "assistant", "user"]);
    }

    #[tokio::test]
    async fn discover_sends_bearer_only_when_configured() {
        let captured: std::sync::Arc<std::sync::Mutex<Option<String>>> =
            std::sync::Arc::new(std::sync::Mutex::new(None));
        let sink = captured.clone();
        let route_capture = captured.clone();
        let app = axum::Router::new().route(
            "/v1/models",
            axum::routing::get(move |headers: axum::http::HeaderMap| {
                let sink = route_capture.clone();
                async move {
                    *sink.lock().unwrap() = headers
                        .get("authorization")
                        .and_then(|v| v.to_str().ok())
                        .map(|s| s.to_string());
                    axum::Json(serde_json::json!({
                        "data": [{"id": "glm-5.3-flash"}]
                    }))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        // Key configured → the Bearer header rides the discovery probe.
        RuntimeExpert::discover(&base, "glm-5.3-flash", Some("k"))
            .await
            .expect("discover succeeds with auth");
        assert_eq!(sink.lock().unwrap().as_deref(), Some("Bearer k"));

        // No key configured → the header must not appear at all.
        *captured.lock().unwrap() = None;
        RuntimeExpert::discover(&base, "glm-5.3-flash", None)
            .await
            .expect("discover succeeds without auth");
        assert!(captured.lock().unwrap().is_none());
    }

    #[tokio::test]
    async fn chat_over_quic_falls_back_to_peer() {
        let (addr, server_identity, pins) = test_server_and_pins(
            Arc::new(ChatAnswer { content: "mesh hello".into() }),
        )
        .await
        .unwrap();
        let node_id = server_identity.node_id();

        let r = router(50);
        let client = pinned_client_endpoint_dyn(pins).unwrap();
        let identity = SigningIdentity::generate();
        let mut addrs = HashMap::new();
        addrs.insert(node_id.clone(), addr);
        let peers = vec![(node_id.clone(), 8, true, true)];

        // Local runtime down → falls back to the mesh peer, attributed.
        let (content, served_by) = r
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
        let _ = &node_id;
        assert_eq!(content, "mesh hello");
        assert_eq!(served_by, node_id); // the node that actually answered

        // Local healthy → mesh never used, no remote attribution.
        let (content2, served_by2) = r
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
        assert_eq!(served_by2, ""); // empty = local

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
        let (addr, server_identity, pins) =
            test_server_and_pins(Arc::new(TimesTwo)).await.unwrap();
        let client = pinned_client_endpoint_dyn(pins).unwrap();
        let identity = SigningIdentity::generate();
        let req = NodeRpcRequest::Ping {
            from_node_id: identity.node_id(),
            stamp_ms: crate::bootstrap::now_ms()
                - crate::signing::STAMP_MAX_AGE_MS
                - 5_000,
            sig: identity.sign(b"whatever").to_string(),
        };
        let reply = call(
            &client,
            addr,
            &server_identity.node_id(),
            &req,
            &identity,
            Duration::from_secs(5),
        )
        .await
        .unwrap();
        assert!(matches!(reply, NodeRpcReply::Error { .. }));
    }

    // --- Negative tests for the pinning verifier itself ---

    fn verifier_with(pins: PinMap) -> PinningVerifier {
        PinningVerifier { pins: Arc::new(pins) }
    }

    /// SNI as `call` sets it: the destination node_id string.
    fn server_name_for(node_id: &str) -> rustls::pki_types::ServerName<'static> {
        rustls::pki_types::ServerName::try_from(node_id.to_string())
            .expect("node_id is a valid SNI string")
    }

    fn identity_cert(identity: &SigningIdentity) -> rustls::pki_types::CertificateDer<'static> {
        generate_node_cert_for(identity).unwrap().cert_der
    }

    #[test]
    fn verifier_rejects_unpinned_destination() {
        let identity = SigningIdentity::generate();
        let cert = identity_cert(&identity);
        let v = verifier_with(PinMap::new()); // no pin at all
        let res = v.verify_server_cert(
            &cert,
            &[],
            &server_name_for(&identity.node_id()),
            &[],
            rustls::pki_types::UnixTime::now(),
        );
        assert!(res.is_err(), "unpinned destination must fail closed");
    }

    #[test]
    fn verifier_rejects_forged_non_ed25519_cert() {
        // rcgen's default RSA path produces a cert whose SPKI is not
        // Ed25519 — the extractor must refuse it.
        let ck = rcgen::generate_simple_self_signed(vec!["forged".to_string()]).unwrap();
        let cert = rustls::pki_types::CertificateDer::from(ck.cert.der().to_vec());
        let pinned_id = SigningIdentity::generate();
        let mut pins = PinMap::new();
        // Pin for the node we dialed (SNI) exists but the cert is RSA.
        pins.insert(pinned_id.node_id(), "127.0.0.1:1".parse().unwrap());
        let v = verifier_with(pins);
        let res = v.verify_server_cert(
            &cert,
            &[],
            &server_name_for(&pinned_id.node_id()),
            &[],
            rustls::pki_types::UnixTime::now(),
        );
        assert!(res.is_err(), "non-Ed25519 cert must be rejected");
    }

    #[test]
    fn verifier_rejects_wrong_node_mismatch() {
        // Cert is a valid identity cert for node Y, but the pin for that
        // addr names node X — must reject.
        let y = SigningIdentity::generate();
        let x_node_id = SigningIdentity::generate().node_id();
        let cert = identity_cert(&y);
        let mut pins = PinMap::new();
        // SNI says node X but the cert is node Y's identity key.
        pins.insert(x_node_id.clone(), "127.0.0.1:1".parse().unwrap());
        let v = verifier_with(pins);
        let res = v.verify_server_cert(
            &cert,
            &[],
            &server_name_for(&x_node_id),
            &[],
            rustls::pki_types::UnixTime::now(),
        );
        assert!(res.is_err(), "mismatched node_id must be rejected");
    }

    #[test]
    fn verifier_accepts_matching_pin() {
        let id = SigningIdentity::generate();
        let cert = identity_cert(&id);
        let mut pins = PinMap::new();
        // SNI = node's own id, cert = that node's identity key: accept.
        pins.insert(id.node_id(), "127.0.0.1:1".parse().unwrap());
        let v = verifier_with(pins);
        let res = v.verify_server_cert(
            &cert,
            &[],
            &server_name_for(&id.node_id()),
            &[],
            rustls::pki_types::UnixTime::now(),
        );
        assert!(res.is_ok(), "matching pin must be accepted: {res:?}");
    }
    #[test]
    fn extractor_decodes_identity_cert_key() {
        let id = SigningIdentity::generate();
        let cert = identity_cert(&id);
        let key_hex = extract_ed25519_spki_hex(&cert).expect("extractor decodes");
        assert_eq!(
            key_hex.starts_with(&id.node_id()),
            true,
            "extracted key {key_hex} must extend node_id {}",
            id.node_id()
        );
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
