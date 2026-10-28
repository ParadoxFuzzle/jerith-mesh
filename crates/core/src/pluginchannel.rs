//! Signed plugin channel index (docs/PLUGIN-CHANNEL.md Phase 4).
//!
//! One publisher's `channel.json` — a signed list of that publisher's
//! published bundle entries — so an operator can LIST what is
//! installable without this repo becoming a registry service. The
//! index is a flat artifact pair, like the bundles it names:
//!
//! ```text
//! channel.json        { publisher, generated_ms, bundles[...] }
//! channel.json.sig    hex(Ed25519 over the canonical payload, see
//!                      signing::plugin_channel_payload)
//! ```
//!
//! Trust posture (deliberately narrower than bundle verify):
//!
//! - **Key pinning: the anchor never rides the wire.** The verifier
//!   takes the publisher pubkey as an argument (`--anchor`, defaulting
//!   to the committed `keys/plugin-releases.pub` for the first-party
//!   channel) and the file's `publisher` field must EQUAL it — a
//!   self-signed index proves nothing, so the field is an identity
//!   claim the anchor confirms, not the key itself.
//! - **Freshness window.** `generated_ms` must sit inside
//!   [now − max_age, now + future tolerance]: a stale index (an
//!   abandoned lane) and a future-stamped one (clock games) both
//!   refuse. Default window 24 h — a publisher stops being trusted as
//!   current the day their channel rots.
//! - **Entries are claims, not verification.** Installing still runs
//!   the full bundle verify chain against the ACTUAL artifact; the
//!   index only shortens the list step. Its digests are checked for
//!   shape here and for reality by `plugin install` itself.
//!
//! Fail-closed everywhere (same house rule as pluginbundle): every
//! refusal names the rule and the offending field.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;

use crate::pluginbundle::{hex_strict, valid_name, version_ok};
use crate::signing::{SigningIdentity, plugin_channel_payload, verify};

/// How old a fetched index may be (the default freshness window).
pub const DEFAULT_MAX_AGE_MS: u64 = 24 * 60 * 60 * 1000;
/// A future stamp beyond this absolute tolerance is a clock game, not skew.
pub const FUTURE_TOLERANCE_MS: u64 = 10 * 60 * 1000;
/// A maintained channel is small; a bloated index is a lure payload.
pub const MAX_ENTRIES: usize = 64;
pub const MAX_URL_BYTES: usize = 512;
pub const MAX_NOTES_BYTES: usize = 256;

pub const CHANNEL_FILE: &str = "channel.json";
pub const CHANNEL_SIG_FILE: &str = "channel.json.sig";

/// One published bundle row.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChannelEntry {
    pub id: String,
    pub version: String,
    pub os: String,
    pub arch: String,
    pub url: String,
    /// The bundle ARTIFACT sha256 (same digest the `.sha256` companion
    /// carries — install recompares it against the fetched bytes).
    pub sha256: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub notes: String,
}

/// The index itself.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChannelIndex {
    /// 64-hex publisher pubkey — must equal the operator's anchor.
    pub publisher: String,
    pub generated_ms: u64,
    pub bundles: Vec<ChannelEntry>,
}

fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Build a deterministic channel.json text (entries sorted by
/// id/version/os/arch/url so a re-publish of unchanged rows is
/// byte-stable) plus its detached sig. Entry shape rules run here too:
/// a publisher should not sign a malformed index any more than an
/// operator should trust one.
pub fn build_channel(
    mut entries: Vec<ChannelEntry>,
    identity: &SigningIdentity,
    generated_ms: u64,
) -> anyhow::Result<(String, String)> {
    anyhow::ensure!(
        !entries.is_empty(),
        "channel: refusing an EMPTY index — a dead channel is a stale channel, publish rows"
    );
    entries.sort_by(|a, b| {
        (&a.id, &a.version, &a.os, &a.arch, &a.url)
            .cmp(&(&b.id, &b.version, &b.os, &b.arch, &b.url))
    });
    let publisher = identity.pubkey_hex();
    validate_publisher(&publisher).map_err(|err| anyhow::anyhow!("channel {err}"))?;
    let mut seen = std::collections::BTreeSet::new();
    for e in &entries {
        validate_entry(e).map_err(|err| anyhow::anyhow!("channel {err}"))?;
        let key = (
            e.id.as_str(),
            e.version.as_str(),
            e.os.as_str(),
            e.arch.as_str(),
        );
        anyhow::ensure!(seen.insert(key), "channel: duplicate entry {key:?}");
    }
    let index = ChannelIndex {
        publisher,
        generated_ms,
        bundles: entries,
    };
    let json = serde_json::to_string_pretty(&index)
        .map_err(|e| anyhow::anyhow!("channel: serialization failed ({e})"))?
        + "\n";
    let sig = sign_channel(&json, identity);
    Ok((json, sig))
}

/// Sign the FINAL bytes of a channel.json (what goes on the wire,
/// trailing newline included) — the signature binds exactly these
/// bytes. The canonical payload's stamp slot carries the file's OWN
/// `generated_ms` and `publisher`, re-read from the JSON, so sign and
/// verify reconstruct the identical string.
pub fn sign_channel(json_bytes: &str, identity: &SigningIdentity) -> String {
    let json_sha = hex::encode(Sha256::digest(json_bytes.as_bytes()));
    let payload = stamped_payload(json_bytes, &json_sha);
    identity.sign(payload.as_bytes())
}

/// The canonical payload a verifier reconstructs from the received
/// bytes: `jerith-chidx1|<publisher>|<generated_ms>|<sha256(bytes)>`.
pub fn stamped_payload(json_bytes: &str, json_sha: &str) -> String {
    let v: ChannelIndex = serde_json::from_str(json_bytes).unwrap_or(ChannelIndex {
        publisher: String::new(),
        generated_ms: 0,
        bundles: Vec::new(),
    });
    plugin_channel_payload(&v.publisher, v.generated_ms, json_sha)
}

/// Verify a fetched/local channel pair. `anchor` is the pinned
/// publisher pubkey (64 hex); `max_age_ms` = 0 means the default
/// window.
pub fn verify_channel(
    json_bytes: &str,
    sig_hex: &str,
    anchor: &str,
    max_age_ms: u64,
) -> anyhow::Result<ChannelIndex> {
    let index: ChannelIndex = serde_json::from_str(json_bytes)
        .map_err(|e| anyhow::anyhow!("channel: not the expected JSON shape ({e})"))?;
    anyhow::ensure!(
        anchor.trim().eq_ignore_ascii_case(&index.publisher),
        "channel: publisher {}… does not equal the pinned anchor {}… — key pinning says refuse (pass the right --anchor or don't trust this source)",
        publisher_thumb(&index.publisher),
        publisher_thumb(anchor)
    );
    validate_publisher(&index.publisher).map_err(|err| anyhow::anyhow!("channel {err}"))?;

    // Freshness window.
    let now = now_ms();
    anyhow::ensure!(
        index.generated_ms <= now + FUTURE_TOLERANCE_MS,
        "channel: generated_ms {} is in the FUTURE beyond the {FUTURE_TOLERANCE_MS} ms tolerance (clock games)",
        index.generated_ms
    );
    let max_age = if max_age_ms == 0 {
        DEFAULT_MAX_AGE_MS
    } else {
        max_age_ms
    };
    anyhow::ensure!(
        now.saturating_sub(index.generated_ms) <= max_age,
        "channel: stale — generated_ms {} is past the {max_age} ms window (maintain or retire the channel)",
        index.generated_ms
    );

    // Signature over the exact bytes received (before any per-row work:
    // identity first, contents second — but sig binds contents, so an
    // edited index can't sneak a valid stale sig through).
    let json_sha = hex::encode(Sha256::digest(json_bytes.as_bytes()));
    let payload = stamped_payload(json_bytes, &json_sha);
    anyhow::ensure!(
        verify(&index.publisher, payload.as_bytes(), sig_hex.trim()),
        "channel: signature does not verify over the received bytes (tamper, wrong sig file, or wrong anchor)"
    );

    anyhow::ensure!(
        !index.bundles.is_empty(),
        "channel: EMPTY bundle list — refusing (a maintained channel carries rows)"
    );
    anyhow::ensure!(
        index.bundles.len() <= MAX_ENTRIES,
        "channel: {} entries exceeds the {MAX_ENTRIES} cap (a bloated index is a lure payload)",
        index.bundles.len()
    );
    let mut seen = std::collections::BTreeSet::new();
    for e in &index.bundles {
        validate_entry(e).map_err(|err| anyhow::anyhow!("channel {err}"))?;
        let key = (
            e.id.as_str(),
            e.version.as_str(),
            e.os.as_str(),
            e.arch.as_str(),
        );
        anyhow::ensure!(seen.insert(key), "channel: duplicate entry {key:?}");
    }
    Ok(index)
}

/// Read a local pair: `<dir>/channel.json` + `<dir>/channel.json.sig`.
pub fn load_channel_pair(dir: &Path) -> anyhow::Result<(String, String)> {
    let json = std::fs::read_to_string(dir.join(CHANNEL_FILE)).map_err(|e| {
        anyhow::anyhow!(
            "channel: {} missing ({e})",
            dir.join(CHANNEL_FILE).display()
        )
    })?;
    let sig = std::fs::read_to_string(dir.join(CHANNEL_SIG_FILE)).map_err(|e| {
        anyhow::anyhow!(
            "channel: {} missing ({e})",
            dir.join(CHANNEL_SIG_FILE).display()
        )
    })?;
    Ok((json, sig))
}

fn validate_publisher(publisher: &str) -> anyhow::Result<()> {
    hex_strict(publisher).map_err(|e| anyhow::anyhow!("publisher {e}"))?;
    Ok(())
}

/// Row-level rules (shape only — `plugin install` re-verifies reality).
fn validate_entry(e: &ChannelEntry) -> anyhow::Result<()> {
    anyhow::ensure!(
        valid_name(&e.id),
        "entry id {:?} is not a valid identifier",
        e.id
    );
    anyhow::ensure!(
        version_ok(&e.version),
        "entry version {:?} is not dec-version shaped",
        e.version
    );
    anyhow::ensure!(
        !e.os.is_empty() && !e.arch.is_empty(),
        "entry {}: os/arch both required",
        e.id
    );
    anyhow::ensure!(
        e.sha256.len() == 64,
        "entry {}: sha256 must be 64 hex chars",
        e.id
    );
    hex_strict(&e.sha256).map_err(|err| anyhow::anyhow!("entry {}: sha256 {err}", e.id))?;
    anyhow::ensure!(
        !e.url.is_empty(),
        "entry {}: url required — an entry without an artifact is a dead row",
        e.id
    );
    anyhow::ensure!(
        !e.url.contains(char::is_whitespace),
        "entry {}: url contains whitespace",
        e.id
    );
    anyhow::ensure!(
        e.url.len() < MAX_URL_BYTES,
        "entry {}: url exceeds the {MAX_URL_BYTES}-byte cap",
        e.id
    );
    anyhow::ensure!(
        e.url.starts_with("https://"),
        "entry {}: url {}… is not https (a channel with plain-http rows is a tamper invitation)",
        e.id,
        e.url.get(..24).unwrap_or(&e.url)
    );
    anyhow::ensure!(
        e.notes.len() <= MAX_NOTES_BYTES,
        "entry {}: notes exceeds the {MAX_NOTES_BYTES}-byte cap",
        e.id
    );
    Ok(())
}

fn publisher_thumb(hex: &str) -> String {
    let head = hex.get(..8).unwrap_or(hex);
    format!("{head}…")
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn publisher() -> SigningIdentity {
        SigningIdentity::from_secret_bytes(&[7u8; 32]).unwrap()
    }

    fn impostor() -> SigningIdentity {
        SigningIdentity::from_secret_bytes(&[9u8; 32]).unwrap()
    }

    fn entry(id: &str, os: &str) -> ChannelEntry {
        ChannelEntry {
            id: id.to_string(),
            version: "0.1.0".into(),
            os: os.into(),
            arch: "x86_64".into(),
            url: format!("https://example.com/{id}-0.1.0-{os}-x86_64.jbundle"),
            sha256: "a".repeat(64),
            notes: "first".into(),
        }
    }

    #[test]
    fn build_then_verify_roundtrip_pins_the_publisher() {
        let id = publisher();
        let (json, sig) = build_channel(vec![entry("weather", "linux")], &id, now_ms()).unwrap();
        assert!(json.contains("\"publisher\""));
        let anchor = id.pubkey_hex();
        let idx = verify_channel(&json, &sig, &anchor, 0).unwrap();
        assert_eq!(idx.bundles.len(), 1);
        assert_eq!(idx.bundles[0].id, "weather");
        // Rebuilding the same rows at the SAME stamp is byte-stable
        // (deterministic sort + fixed serde field order) — only the
        // stamp differs between issues.
        let (json2, _) =
            build_channel(vec![entry("weather", "linux")], &id, idx.generated_ms).unwrap();
        assert_eq!(json, json2);
    }

    #[test]
    fn a_self_signed_index_is_worthless() {
        let real = publisher();
        let fake = impostor();
        // Someone else signs a file that CLAIMS the publisher's identity.
        let (json, _real_sig) =
            build_channel(vec![entry("weather", "linux")], &real, now_ms()).unwrap();
        let json_sha = hex::encode(Sha256::digest(json.as_bytes()));
        let fake_sig = fake.sign(stamped_payload(&json, &json_sha).as_bytes());
        let anchor = real.pubkey_hex();
        let err = verify_channel(&json, &fake_sig, &anchor, 0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("does not verify"), "{err}");
        // And a file whose own publisher differs from the anchor refuses
        // on identity BEFORE the signature even matters.
        let (json2, sig2) =
            build_channel(vec![entry("weather", "linux")], &fake, now_ms()).unwrap();
        let err = verify_channel(&json2, &sig2, &anchor, 0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("does not equal the pinned anchor"), "{err}");
    }

    #[test]
    fn tampered_bytes_break_the_signature() {
        let id = publisher();
        let (json, sig) = build_channel(vec![entry("weather", "linux")], &id, now_ms()).unwrap();
        // The classic mirror tamper: swap the entry's sha256 in the text.
        let rewritten = json.replace(&"a".repeat(64), &"b".repeat(64));
        assert_ne!(json, rewritten);
        let err = verify_channel(&rewritten, &sig, &id.pubkey_hex(), 0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("does not verify"), "{err}");
        // Even a re-signed stale tamper must fail on staleness, not ride
        // through because "the signature is fresh".
        let stamp = now_ms() - DEFAULT_MAX_AGE_MS - 10_000;
        let (json, sig) = build_channel(vec![entry("weather", "linux")], &id, stamp).unwrap();
        let err = verify_channel(&json, &sig, &id.pubkey_hex(), 0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("stale"), "{err}");
    }

    #[test]
    fn freshness_window_bounds_both_directions() {
        let id = publisher();
        let anchor = id.pubkey_hex();
        // One minute old: fine (also covers the 0 → default max_age).
        let (json, sig) =
            build_channel(vec![entry("weather", "linux")], &id, now_ms() - 60_000).unwrap();
        verify_channel(&json, &sig, &anchor, 0).unwrap();
        // Just inside a custom shorter window: fine.
        verify_channel(&json, &sig, &anchor, 120_000).unwrap();
        // Just outside that custom window: stale.
        let err = verify_channel(&json, &sig, &anchor, 30_000)
            .unwrap_err()
            .to_string();
        assert!(err.contains("stale"), "{err}");
        // Future beyond tolerance: clock game.
        let (json, sig) = build_channel(
            vec![entry("weather", "linux")],
            &id,
            now_ms() + FUTURE_TOLERANCE_MS + 10_000,
        )
        .unwrap();
        let err = verify_channel(&json, &sig, &anchor, 0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("in the FUTURE"), "{err}");
    }

    #[test]
    fn entry_rules_fail_closed_with_named_offenses() {
        let id = publisher();

        let mut e = entry("weather", "linux");
        e.url = "http://example.com/weather.jbundle".into();
        let err = build_channel(vec![e], &id, now_ms())
            .unwrap_err()
            .to_string();
        assert!(err.contains("not https"), "{err}");

        let mut e = entry("weather", "linux");
        e.id = "bad/slash".into();
        e.url = "https://example.com/x.jbundle".into();
        let err = build_channel(vec![e], &id, now_ms())
            .unwrap_err()
            .to_string();
        assert!(err.contains("not a valid identifier"), "{err}");

        let mut e = entry("weather", "linux");
        e.sha256 = "xyz".into();
        e.url = "https://example.com/x.jbundle".into();
        let err = build_channel(vec![e], &id, now_ms())
            .unwrap_err()
            .to_string();
        assert!(err.contains("sha256"), "{err}");

        let mut e = entry("weather", "linux");
        e.notes = "n".repeat(MAX_NOTES_BYTES + 1);
        e.url = "https://example.com/x.jbundle".into();
        let err = build_channel(vec![e], &id, now_ms())
            .unwrap_err()
            .to_string();
        assert!(err.contains("notes"), "{err}");

        // Verify caps the row count even when a publisher signs more
        // (a pubkey-signed 65-row index still refuses at the receiver).
        let many: Vec<ChannelEntry> = (0..MAX_ENTRIES + 1)
            .map(|i| entry(&format!("p{i:03}"), "linux"))
            .collect();
        let err = match build_channel(many, &id, now_ms()) {
            Ok((json, sig)) => verify_channel(&json, &sig, &id.pubkey_hex(), 0)
                .unwrap_err()
                .to_string(),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("cap") || err.contains("cap "), "{err}");
    }

    #[test]
    fn duplicates_refuse_on_both_sides() {
        let id = publisher();
        let rows = vec![entry("weather", "linux"), entry("weather", "linux")];
        let err = build_channel(rows, &id, now_ms()).unwrap_err().to_string();
        assert!(err.contains("duplicate entry"), "{err}");
        // Empty index refuses at build; empty-after-verify refuses too.
        let err = build_channel(Vec::new(), &id, now_ms())
            .unwrap_err()
            .to_string();
        assert!(err.contains("EMPTY"), "{err}");
    }
}
