//! Ed25519 node identity and canonical payload signing.
//!
//! Every node owns a keypair generated at setup. The `node_id` is the
//! fingerprint of the public key (first 16 hex chars), which makes identity
//! spoofing impossible: a forged advertisement claiming someone else's
//! node_id cannot produce a valid signature without the private key.
//!
//! Wire envelopes (Phase 1 wire auth):
//! - Advertise: `PUT /v1/advertise` with headers
//!   `x-jerith-pubkey` (hex), `x-jerith-stamp` (unix ms),
//!   `x-jerith-sig` (hex, over [`advertise_payload`]).
//! - Ping request: same headers over [`ping_payload`].
//! - Health reply: body fields `pubkey`, `stamp`, `sig` over
//!   [`health_payload`].

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};

use crate::bootstrap::{PeerAd, now_ms};

/// Domain separation prefixes — never reuse across payload kinds.
pub const ADV_PREFIX: &str = "jerith-adv1";
pub const PING_PREFIX: &str = "jerith-ping1";
pub const HEALTH_PREFIX: &str = "jerith-health1";
pub const DEAD_PREFIX: &str = "jerith-dead1";

/// Max acceptable age for a stamp (advertisement TTL is 90 s; the stamp
/// window is tighter to block replays).
pub const STAMP_MAX_AGE_MS: u64 = 120_000;
pub const STAMP_MAX_FUTURE_MS: u64 = 30_000;

/// A node's signing identity.
#[derive(Clone)]
pub struct SigningIdentity {
    signing: SigningKey,
}

impl SigningIdentity {
    /// Derive from the 32-byte secret stored in config.
    pub fn from_secret_bytes(secret: &[u8]) -> anyhow::Result<Self> {
        let arr: [u8; 32] = secret
            .try_into()
            .map_err(|_| anyhow::anyhow!("secret key must be exactly 32 bytes"))?;
        Ok(Self {
            signing: SigningKey::from_bytes(&arr),
        })
    }

    /// Fresh random keypair.
    pub fn generate() -> Self {
        let secret: [u8; 32] = rand::random();
        Self {
            signing: SigningKey::from_bytes(&secret),
        }
    }

    pub fn secret_bytes(&self) -> [u8; 32] {
        self.signing.to_bytes()
    }

    pub fn pubkey_hex(&self) -> String {
        hex::encode(self.signing.verifying_key().to_bytes())
    }

    /// node_id fingerprint: first 16 hex chars of the pubkey.
    pub fn node_id(&self) -> String {
        self.pubkey_hex()[..16].to_string()
    }

    pub fn sign(&self, msg: &[u8]) -> String {
        hex::encode(self.signing.sign(msg).to_bytes())
    }
}

/// Canonical advertisement payload — both sides must build it identically.
pub fn advertise_payload(ad: &PeerAd, stamp_ms: u64) -> String {
    format!(
        "{ADV_PREFIX}|{stamp_ms}|{}|{}|{}|{}|{}|{}|{}|{}",
        ad.node_id,
        ad.address,
        ad.model_id,
        ad.quant,
        ad.contribution_percent,
        ad.last_seen_ms,
        role_str(ad.role),
        ad.rpc_address,
    )
}

/// Canonical ping payload.
pub fn ping_payload(from_node_id: &str, to_node_id: &str, stamp_ms: u64) -> String {
    format!("{PING_PREFIX}|{stamp_ms}|{from_node_id}|{to_node_id}")
}

/// Canonical health-reply payload. Extras outside the signature stay out of
/// it so both sides can reproduce the exact string.
pub fn health_payload(
    node_id: &str,
    stamp_ms: u64,
    uptime_seconds: u64,
    contribution_percent: u64,
) -> String {
    format!("{HEALTH_PREFIX}|{stamp_ms}|{node_id}|{uptime_seconds}|{contribution_percent}")
}

/// Transitional: the canonical payload exactly as it existed before
/// `rpc_address` was appended. Server-side only, for rolling upgrades.
pub fn legacy_ad_excerpt(ad: &PeerAd, stamp_ms: u64) -> String {
    format!(
        "{ADV_PREFIX}|{stamp_ms}|{}|{}|{}|{}|{}|{}|{}",
        ad.node_id,
        ad.address,
        ad.model_id,
        ad.quant,
        ad.contribution_percent,
        ad.last_seen_ms,
        role_str(ad.role),
    )
}

/// Canonical dead-report payload. Reports that can EVICT a peer from the
/// registry must be attributed: reporter signs `prefix|stamp|target`, and
/// the bootstrap verifies before counting a vote.
pub fn dead_payload(reporter_node_id: &str, target_node_id: &str, stamp_ms: u64) -> String {
    format!("{DEAD_PREFIX}|{stamp_ms}|{reporter_node_id}|{target_node_id}")
}

fn role_str(role: crate::config::NodeRole) -> &'static str {
    match role {
        crate::config::NodeRole::Contributor => "contributor",
        crate::config::NodeRole::Requester => "requester",
        crate::config::NodeRole::Hybrid => "hybrid",
    }
}

/// Verify a signature against a hex pubkey.
pub fn verify(pubkey_hex: &str, msg: &[u8], sig_hex: &str) -> bool {
    let Ok(pk_bytes) = hex::decode(pubkey_hex) else {
        return false;
    };
    let Ok(pk_arr) = <[u8; 32]>::try_from(pk_bytes.as_slice()) else {
        return false;
    };
    let Ok(vk) = VerifyingKey::from_bytes(&pk_arr) else {
        return false;
    };
    let Ok(sig_bytes) = hex::decode(sig_hex) else {
        return false;
    };
    let Ok(sig_arr) = <[u8; 64]>::try_from(sig_bytes.as_slice()) else {
        return false;
    };
    vk.verify(msg, &Signature::from_bytes(&sig_arr)).is_ok()
}

/// Canonical payload for a signed bootstrap peers-fetch: the rate-limit
/// lane token. The bootstrap's middleware verifies this payload to
/// attribute the hit to a peer-identity bucket (per pubkey) instead of
/// the per-IP anonymous bucket — the NAT-scaling allowance. The model
/// id deliberately stays outside the signed payload; the lane proves
/// "a real key-holder asks, now", nothing more.
pub fn peers_fetch_payload(stamp_ms: u64) -> String {
    format!("peers-req1|{stamp_ms}")
}

/// Headers carried by signed wire requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireAuth {
    pub pubkey: String,
    pub stamp_ms: u64,
    pub sig: String,
}

impl WireAuth {
    pub fn now(identity: &SigningIdentity, msg: &[u8]) -> Self {
        Self {
            pubkey: identity.pubkey_hex(),
            stamp_ms: now_ms(),
            sig: identity.sign(msg),
        }
    }

    /// Sign with an explicit, single-sourced stamp.
    ///
    /// REQUIRED whenever the message payload itself embeds the stamp (the
    /// canonical advertise/ping/health/fraud payloads all do): the verifying
    /// side rebuilds the payload from `stamp_ms`, so a caller that computes
    /// the payload with one `now_ms()` and then calls `now()` — which stamps
    /// AGAIN — signs a different byte-stream than the header claims when the
    /// two clock reads straddle a millisecond boundary. Under load (parallel
    /// tests, busy runtimes) that race produced intermittent 401s.
    pub fn with_stamp(identity: &SigningIdentity, stamp_ms: u64, msg: &[u8]) -> Self {
        Self {
            pubkey: identity.pubkey_hex(),
            stamp_ms,
            sig: identity.sign(msg),
        }
    }

    /// Validate freshness + signature + node_id binding.
    /// `expected_node_id`: the identity the peer claims (its node_id must
    /// equal the pubkey fingerprint).
    pub fn validate(&self, msg: &[u8], expected_node_id: &str) -> Result<(), AuthError> {
        // node_id must be derived from the presented pubkey.
        let fp = &self.pubkey[..16.min(self.pubkey.len())];
        if self.pubkey.len() != 64 || fp != expected_node_id {
            return Err(AuthError::IdentityMismatch);
        }
        // Stamp freshness blocks replay.
        let age = now_ms().saturating_sub(self.stamp_ms);
        if age > STAMP_MAX_AGE_MS {
            return Err(AuthError::StaleStamp);
        }
        if self.stamp_ms > now_ms() + STAMP_MAX_FUTURE_MS {
            return Err(AuthError::FutureStamp);
        }
        if !verify(&self.pubkey, msg, &self.sig) {
            return Err(AuthError::BadSignature);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthError {
    IdentityMismatch,
    StaleStamp,
    FutureStamp,
    BadSignature,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::NodeRole;

    fn ad() -> PeerAd {
        PeerAd {
            node_id: "aabbccdd11223344".into(),
            address: "127.0.0.1:8900".into(),
            model_id: "glm-5.3".into(),
            quant: "int4".into(),
            contribution_percent: 30,
            rpc_address: String::new(),
            last_seen_ms: 1_000,
            role: NodeRole::Hybrid,
            resources: None,
            hosted_experts: None,
            hosted_expert_count: None,
        }
    }

    #[test]
    fn node_id_is_pubkey_fingerprint() {
        let id = SigningIdentity::generate();
        assert_eq!(id.node_id(), id.pubkey_hex()[..16]);
        assert_eq!(id.pubkey_hex().len(), 64);
    }

    #[test]
    fn sign_verify_roundtrip() {
        let id = SigningIdentity::generate();
        let msg = b"hello mesh";
        let sig = id.sign(msg);
        assert!(verify(&id.pubkey_hex(), msg, &sig));
    }

    #[test]
    fn tampered_message_fails() {
        let id = SigningIdentity::generate();
        let sig = id.sign(b"hello mesh");
        assert!(!verify(&id.pubkey_hex(), b"hello mesf", &sig));
    }

    #[test]
    fn secret_restore_reproduces_identity() {
        let id = SigningIdentity::generate();
        let restored = SigningIdentity::from_secret_bytes(&id.secret_bytes()).unwrap();
        assert_eq!(restored.pubkey_hex(), id.pubkey_hex());
        assert_eq!(restored.node_id(), id.node_id());
    }

    #[test]
    fn wire_auth_validates_good_stamp() {
        let id = SigningIdentity::generate();
        let msg = advertise_payload(&ad(), now_ms());
        let auth = WireAuth::now(&id, msg.as_bytes());
        assert_eq!(auth.validate(msg.as_bytes(), &id.node_id()), Ok(()));
    }

    #[test]
    fn wire_auth_rejects_forged_signature() {
        let id = SigningIdentity::generate();
        let attacker = SigningIdentity::generate();
        let stamp = now_ms();
        let msg = advertise_payload(&ad(), stamp);
        let mut forged = WireAuth {
            pubkey: id.pubkey_hex(),
            stamp_ms: stamp,
            sig: attacker.sign(msg.as_bytes()),
        };
        // Attacker signs with own key but claims victim's pubkey.
        forged.sig = attacker.sign(msg.as_bytes());
        assert_eq!(
            forged.validate(msg.as_bytes(), &id.node_id()),
            Err(AuthError::BadSignature)
        );
    }

    #[test]
    fn wire_auth_rejects_identity_mismatch() {
        let id = SigningIdentity::generate();
        let stamp = now_ms();
        let msg = ping_payload("someone-else", "target", stamp);
        let auth = WireAuth::now(&id, msg.as_bytes());
        assert_eq!(
            auth.validate(msg.as_bytes(), "someone-else"),
            Err(AuthError::IdentityMismatch)
        );
    }

    #[test]
    fn wire_auth_rejects_stale_stamp() {
        let id = SigningIdentity::generate();
        let old_stamp = now_ms() - STAMP_MAX_AGE_MS - 1_000;
        let msg = advertise_payload(&ad(), old_stamp);
        let auth = WireAuth {
            pubkey: id.pubkey_hex(),
            stamp_ms: old_stamp,
            sig: id.sign(msg.as_bytes()),
        };
        assert_eq!(
            auth.validate(msg.as_bytes(), &id.node_id()),
            Err(AuthError::StaleStamp)
        );
    }

    #[test]
    fn wire_auth_rejects_future_stamp() {
        let id = SigningIdentity::generate();
        let future = now_ms() + STAMP_MAX_FUTURE_MS + 60_000;
        let msg = advertise_payload(&ad(), future);
        let auth = WireAuth {
            pubkey: id.pubkey_hex(),
            stamp_ms: future,
            sig: id.sign(msg.as_bytes()),
        };
        assert_eq!(
            auth.validate(msg.as_bytes(), &id.node_id()),
            Err(AuthError::FutureStamp)
        );
    }

    #[test]
    fn canonical_payloads_are_deterministic() {
        let id = SigningIdentity::generate();
        let stamp = 1_234_567;
        let a = advertise_payload(&ad(), stamp);
        let b = advertise_payload(&ad(), stamp);
        assert_eq!(a, b);
        assert!(a.starts_with(ADV_PREFIX));
        let p = ping_payload("x", "y", stamp);
        assert!(p.starts_with(PING_PREFIX));
        let h = health_payload("x", stamp, 10, 25);
        assert!(h.starts_with(HEALTH_PREFIX));
        let _ = id;
    }

    #[test]
    fn different_prefixes_never_collide() {
        let s1 = advertise_payload(&ad(), 1);
        let s2 = ping_payload(&ad().node_id, "t", 1);
        assert_ne!(s1, s2);
    }
}
