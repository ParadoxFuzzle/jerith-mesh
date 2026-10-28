//! The plugin-lane receipt log (docs/PLUGIN-CHANNEL.md's audit-chain
//! open question, now TAMPER-EVIDENT via a hash chain). Every CLI
//! plugin-lane ACTION appends one JSON line to `<node home>/plugin.json`:
//!
//! - `install` / `remove` — what landed (or left) plugins.d, with the
//!   bundle digest and the PIN that authorized it — including WHERE the
//!   pin came from, so a rotation (pin-rotate receipt) explains any
//!   change in what subsequent installs accept;
//! - `pin-rotate` — the trust-robot's own record: old key → new key
//!   per plugin id, the moment rotation lands;
//! - `channel-verify` — the freshness check itself, stamped with the
//!   index's `generated_ms`, the age OBSERVED at check time and the
//!   window it passed under.
//!
//! TAMPER-EVIDENCE (2026-10-28): every chained line carries
//!
//! - `prev` — the PREVIOUS chained line's digest (`GENESIS_HASH` for
//!   the first: line-to-line linkage, so a spliced or reordered entry
//!   breaks the chain at the splice);
//! - `chain` — SHA-256 over the canonical serialization of the entry
//!   WITHOUT `chain` (declaration-order fields — the digest pins the
//!   exact content), so editing any field after the fact fails
//!   re-verification at that line.
//!
//! A sibling `<home>/plugin.json.head` file carries the last digest:
//! the log truncated while the head stayed gets flagged (covers
//! deletes-the-tail attacks the hash chain alone cannot see).
//!
//! SIGNATURES (2026-10-28, the old open question landed): each chained
//! line can additionally carry `signer` (a per-box RECEIPTS key's
//! pubkey, 64-hex) and `sig` — Ed25519 over
//! `jerith-recpt1|<chain digest>`, so the signature transitively binds
//! the same content+linkage the digest does, but with a KEY: rewrite
//! the log coherently and the digests line up, yet the signatures must
//! come from the receipts key recorded in the lines. The key lives in
//! `<home>/plugin-receipts.key` (32-hex seed, 0600, auto-generated on
//! first receipt) with its public half in `plugin-receipts.pub` — an
//! independently-copyable anchor: a whole-log re-key (attacker deletes
//! the seed and re-signs everything) is invisible to the log itself
//! but detectable against ANY copy of the pubkey the operator kept,//! since the log's first signer thumbprint changes too. Reader rules:
//! a signed entry verifies or the verdict is Broken at that line; a
//! signed log that suddenly changes signer mid-log is Broken (re-key
//! splice); entirely UNSIGNED chains (pre-0.1.3 logs) stay Intact since
//! sig-less lines are simply not covered by the signature lane.
//!
//! Appends only; a torn/missing line skips on read rather than
//! blackholing the file, but malformed LOGGED shapes should not occur —
//! validation happens before record, like everywhere else in the lane.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::config::NodeConfig;
use crate::signing::{self, SigningIdentity};

/// The receipt log lives at the node home root — siblings of
/// config.toml / bot.toml, NOT under `plugins/` (an id may legally be
/// "plugin.json"; the home root cannot collide with the bundle tree).
pub const RECEIPT_FILE: &str = "plugin.json";
/// The chain head: the LAST line's digest, written after each append.
/// Deletion-tamper check (the chain alone cannot see a truncated file).
pub const RECEIPT_HEAD_FILE: &str = "plugin.json.head";
/// The first chained line's `prev` (64 zeros — there is no zero digest).
pub const GENESIS_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";
/// The per-box RECEIPTS signing key (32-hex Ed25519 seed, 0600,
/// auto-generated on first receipt; NOT the publisher key — this one
/// never leaves the box and signs only this log).
pub const RECEIPT_SEED_FILE: &str = "plugin-receipts.key";
/// The receipts key's public half (the knock-off anchor: compare any
/// independent copy against the log's `signer` thumbprints).
pub const RECEIPT_PUB_FILE: &str = "plugin-receipts.pub";

pub fn receipts_path_under(home: &Path) -> PathBuf {
    home.join(RECEIPT_FILE)
}

pub fn receipts_path() -> PathBuf {
    receipts_path_under(&NodeConfig::home())
}

pub fn head_path_under(home: &Path) -> PathBuf {
    home.join(RECEIPT_HEAD_FILE)
}

pub fn seed_path_under(home: &Path) -> PathBuf {
    home.join(RECEIPT_SEED_FILE)
}

pub fn pub_path_under(home: &Path) -> PathBuf {
    home.join(RECEIPT_PUB_FILE)
}

/// Load (or mint) the per-box receipts key: the seed file is the truth;
/// the .pub file is (re)written from it on every load so the anchor can
/// never drift from the log's actual signer. 0600 both — the .pub is
/// not a secret, but nothing else in this home rides world-readable.
fn load_or_create_identity(home: &Path) -> anyhow::Result<SigningIdentity> {
    let seed_path = seed_path_under(home);
    let identity = if let Ok(seed_hex) = std::fs::read_to_string(&seed_path) {
        SigningIdentity::from_secret_hex(&seed_hex)?
    } else {
        if let Some(parent) = seed_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let identity = SigningIdentity::generate();
        std::fs::write(
            &seed_path,
            format!("{}\n", hex::encode(identity.secret_bytes())),
        )?;
        #[cfg(unix)]
        crate::config::NodeConfig::restrict_file_permissions(&seed_path);
        identity
    };
    let pub_hex = identity.pubkey_hex();
    let pub_path = pub_path_under(home);
    if std::fs::read_to_string(&pub_path)
        .map(|t| t.trim() != pub_hex)
        .unwrap_or(true)
    {
        std::fs::write(&pub_path, format!("{pub_hex}\n"))?;
        #[cfg(unix)]
        crate::config::NodeConfig::restrict_file_permissions(&pub_path);
    }
    Ok(identity)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceiptEntry {
    /// Wall-clock append stamp (epoch ms).
    pub ts_ms: u64,
    /// `install` | `remove` | `pin-rotate` | `channel-verify`.
    pub action: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plugin: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Publisher pubkey in play (64-hex; install's signer / rotate's
    /// new pin / channel's publisher). Full value — a receipt should
    /// not make the operator go hunting for the other half.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub publisher: Option<String>,
    /// The bundle artifact digest (the same one `.sha256` + a channel
    /// row carry — the join key across all three surfaces).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bundle_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pin: Option<PinReceipt>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel: Option<ChannelReceipt>,
    /// Free-form specifics (trash path, row count, rotate edges).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// The RECEIPTS key's pubkey (64-hex) that signed this line. Sits
    /// BEFORE `prev` in declaration order — the signer is CONTENT the
    /// chain digest covers, so the digest pins which key signed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signer: Option<String>,
    /// The previous CHAINED line's digest (orphans: `GENESIS_HASH`).
    /// Written by record; included in THIS line's hash input — the
    /// splice order IS the content. Never written by hand. MUST stay
    /// the second-to-last field (canonical hashing is declaration-order
    /// — see `chain_hash`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prev: Option<String>,
    /// SHA-256 over the canonical serialization of this entry minus
    /// `chain` itself. MUST stay the LAST field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chain: Option<String>,
    /// Ed25519 over `jerith-recpt1|<chain digest>` by `signer` — only
    /// meaningful WITH `chain` (the digest is what gets signed). Sits
    /// AFTER `chain` and is EXCLUDED from the digest input (chain_hash
    /// zeroes it), so legacy digest bytes stay byte-equal across this
    /// field's introduction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sig: Option<String>,
}

/// The pin that authorized an action, with its SOURCE — the audit
/// trail's answer to "was this bot.toml's table, the legacy pins file,
/// or a one-shot --pubkey".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PinReceipt {
    /// `flag` | `bot.toml` | `pins-file`.
    pub source: String,
    /// 64-hex pin applied.
    pub pin: String,
}

/// The freshness state a channel check ran under.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelReceipt {
    /// The channel source as the operator addressed it (url or dir).
    pub source: String,
    /// The index's own generated_ms (as published).
    pub generated_ms: u64,
    /// now − generated_ms at the moment of the check.
    pub observed_age_ms: u64,
    /// The window enforced during this check (0 would have meant the
    /// 24 h default — resolved and recorded as seen).
    pub max_age_ms: u64,
}

/// The digest of one entry: canonical serialization (declaration-order
/// fields, `chain` AND `sig` omitted) hashed. The order contract is
/// why `prev` and `chain` sit LAST in the struct (with `sig` after
/// both) — the hash input of line N is exactly the bytes appended for
/// line N minus its own digest and signature.
fn chain_hash(entry: &ReceiptEntry) -> anyhow::Result<String> {
    let mut w = entry.clone();
    w.chain = None;
    w.sig = None;
    let text = serde_json::to_string(&w).map_err(|e| anyhow::anyhow!("receipt {e}"))?;
    Ok(hex::encode(Sha256::digest(text.as_bytes())))
}

/// The last chained line's digest (the append point). A torn tail (a
/// line that does not parse) is skipped — chaining resumes from the
/// last GOOD entry, which is the honest state of the file.
fn last_chain_hash(path: &Path) -> String {
    let Ok(text) = std::fs::read_to_string(path) else {
        return GENESIS_HASH.to_string();
    };
    for line in text.lines().rev() {
        // A parsed legacy line (no chain) does not provide an append
        // point — keep hunting backwards for a chained one.
        if let Some(c) = serde_json::from_str::<ReceiptEntry>(line)
            .ok()
            .and_then(|e| e.chain)
        {
            return c;
        }
    }
    GENESIS_HASH.to_string()
}

impl ReceiptEntry {
    /// Append the entry CHAINED to the log under an explicit home (one
    /// JSON line; the file is created 0600 and NEVER rewritten — an
    /// append-only trail is the whole point). The chain head file
    /// `plugin.json.head` is (re)written with the new tail digest.
    pub fn record_under(&self, home: &Path) -> anyhow::Result<()> {
        let path = receipts_path_under(home);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut chained = self.clone();
        chained.prev = Some(last_chain_hash(&path));
        // SIGN it: the per-box receipts key goes in FIRST (it is content
        // the chain digest covers), then the digest, then the Ed25519
        // signature over that digest — so the sig binds content +
        // linkage + signer with a KEY. Seed file is created on first
        // receipt (pub file kept in sync).
        let identity = load_or_create_identity(home)?;
        chained.signer = Some(identity.pubkey_hex());
        let digest = chain_hash(&chained)?;
        chained.chain = Some(digest.clone());
        chained.sig = Some(identity.sign(signing::receipt_payload(&digest).as_bytes()));
        let line = serde_json::to_string(&chained).map_err(|e| anyhow::anyhow!("receipt {e}"))?;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        // ONE write: line\n together. Two calls could tear between
        // them (a line with no newline) — harmless on read (a torn
        // tail is skipped, chaining resumes at the last good entry),
        // but the single write_all shrinks that crash window.
        let mut buf = line.into_bytes();
        buf.push(b'\n');
        file.write_all(&buf)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            // The CREATE got default umask perms; ensure 0600 in place
            // (idempotent — an operation on an existing file never
            // loosens anything).
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
        }
        // The surviving out-of-band witness: an attacker must now edit
        // BOTH files coherently to hide a truncation.
        let head = head_path_under(home);
        std::fs::write(&head, format!("{digest}\n"))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&head, std::fs::Permissions::from_mode(0o600))?;
        }
        Ok(())
    }

    /// The real-home record (the CLI call).
    pub fn record(&self) -> anyhow::Result<()> {
        self.record_under(&NodeConfig::home())
    }
}

/// What integrity checking found: intact, a specific break, or a
/// truncation (the head witness disagrees with the log's last entry).
#[derive(Debug, Clone)]
pub enum LogIntegrity {
    /// Every chained line verified, and the head witness (when present)
    /// matches the log's last digest.
    Intact,
    /// Line `at` (0-indexed into the parsed entries) fails; `why` names
    /// which check (prev linkage or content digest).
    Broken { at: usize, why: String },
    /// The head witness names a digest the log no longer ends at — the
    /// log's tail was removed (head: what must be, last: what is).
    Truncated { head: String, last: String },
}

/// The log plus what its integrity check found. A `Broken`/`Truncated`
/// DOES NOT drop entries — the log is shown as parsed with the verdict
/// beside it (hiding the entries would be its own kind of tampering).
#[derive(Debug, Clone)]
pub struct Verdict {
    pub entries: Vec<ReceiptEntry>,
    pub integrity: LogIntegrity,
    /// How many parsed entries actually carry chain fields (those are
    /// the ones the chain covers — pre-chain/legacy lines ride along
    /// uncovered, historically labeled).
    pub chained: usize,
    /// How many parsed entries carry a complete signer+sig trio that
    /// VERIFIED (a pre-sigs chained log reads 0 here without a).
    pub signed: usize,
}

/// Read the log under an explicit home and verify the chain + head.
pub fn read_verified_under(home: &Path) -> Verdict {
    let entries = read_under(home);
    let mut chained = 0usize;
    let mut signed = 0usize;
    let mut expected_prev = GENESIS_HASH.to_string();
    let mut chain_started = false;
    // The receipts key: the FIRST signed entry de-senses it; every
    // later signed entry must match — a mid-log key change is an
    // insertion/replacement, not an operator action.
    let mut first_signer: Option<String> = None;
    let mut integrity = LogIntegrity::Intact;
    for (i, e) in entries.iter().enumerate() {
        match (&e.chain, &e.prev) {
            (Some(_), Some(_)) => {
                // Linked line: prev must continue the walk; digest must
                // re-derive from the content itself.
                if e.prev.as_deref() != Some(expected_prev.as_str()) {
                    integrity = LogIntegrity::Broken {
                        at: i,
                        why: format!(
                            "prev {}… does not continue the chain (expected {}…)",
                            &e.prev.as_ref().unwrap()[..8],
                            &expected_prev[..8]
                        ),
                    };
                    break;
                }
                match chain_hash(e) {
                    Ok(d) if d == e.chain.as_deref().unwrap_or("") => {
                        expected_prev = d;
                        chain_started = true;
                        chained += 1;
                        // Signature lane: complete trios verify (key pin
                        // + Ed25519 over the domain-separated digest);
                        // half-filled ones are shaped like forgeries.
                        let trio = (e.signer.as_deref(), e.sig.as_deref(), e.chain.as_deref());
                        match trio {
                            (Some(signer), Some(sig), Some(digest)) => {
                                if let Some(first) = &first_signer
                                    && first != signer
                                {
                                    integrity = LogIntegrity::Broken {
                                        at: i,
                                        why: format!(
                                            "signer key changed mid-log ({s0}… → {sn}…) — re-key splice?",
                                            s0 = &first[..8],
                                            sn = &signer[..8]
                                        ),
                                    };
                                    break;
                                }
                                if !signing::verify(
                                    signer,
                                    signing::receipt_payload(digest).as_bytes(),
                                    sig,
                                ) {
                                    integrity = LogIntegrity::Broken {
                                        at: i,
                                        why: "receipt signature does not verify".into(),
                                    };
                                    break;
                                }
                                if first_signer.is_none() {
                                    first_signer = Some(signer.to_string());
                                }
                                signed += 1;
                            }
                            (Some(_), None, _) | (None, Some(_), _) => {
                                integrity = LogIntegrity::Broken {
                                    at: i,
                                    why: "half-signed line (signer/sig must come as a pair)".into(),
                                };
                                break;
                            }
                            // A chained-but-unsigned line: legal legacy
                            // shape (pre-signature logs) — the digest and
                            // linkage already verified; the signature lane
                            // simply does not cover it.
                            _ => {}
                        }
                    }
                    Ok(_) => {
                        integrity = LogIntegrity::Broken {
                            at: i,
                            why: "content digest does not match the line (edited in place)".into(),
                        };
                        break;
                    }
                    Err(e2) => {
                        integrity = LogIntegrity::Broken {
                            at: i,
                            why: e2.to_string(),
                        };
                        break;
                    }
                }
            }
            // A line with no chain fields is a LEGACY (pre-chain) one.
            // Allowed only BEFORE any chained line — after the chain
            // starts, an unchained line is an insertion.
            (None, None) if !chain_started => {}
            (None, _) if !chain_started => {}
            _ => {
                integrity = LogIntegrity::Broken {
                    at: i,
                    why: "unchained line after the chain started (inserted?)".into(),
                };
                break;
            }
        }
    }
    // Truncation witness: the head file must equal the log's tail
    // digest (or GENESIS while there is nothing chained).
    let last_digest = entries
        .iter()
        .rev()
        .find_map(|e| e.chain.clone())
        .unwrap_or_else(|| GENESIS_HASH.to_string());
    if let Ok(head) = std::fs::read_to_string(head_path_under(home)) {
        let head = head.trim().to_string();
        if !head.is_empty() && head != last_digest {
            // A specific Break outranks the witness: the mismatch is
            // the more precise evidence (and often just a CONSEQUENCE
            // of the same rewrite that broke the chain). Truncation is
            // reported when the chain walked clean but the log's tail
            // was removed underneath the head file.
            if matches!(integrity, LogIntegrity::Intact) {
                integrity = LogIntegrity::Truncated {
                    head,
                    last: last_digest,
                };
            }
        }
    }
    Verdict {
        entries,
        integrity,
        chained,
        signed,
    }
}

pub fn read_verified() -> Verdict {
    read_verified_under(&NodeConfig::home())
}

/// Read the log under an explicit home: entries in append order.
/// Unreadable lines are SKIPPED (a torn tail from a crash must not
/// blackhole an audit trail) — but only whole-line garbage counts; a
/// malformed KNOWN shape would not have been written.
fn read_under(home: &Path) -> Vec<ReceiptEntry> {
    let Ok(text) = std::fs::read_to_string(receipts_path_under(home)) else {
        return Vec::new();
    };
    text.lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("jerith-receipt-test-{}-{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("tmp dir");
        dir
    }

    fn entry(action: &str, plugin: &str, detail: &str) -> ReceiptEntry {
        ReceiptEntry {
            ts_ms: 1730000000000,
            action: action.into(),
            plugin: Some(plugin.into()),
            version: Some("0.1.1".into()),
            publisher: Some("a".repeat(64)),
            bundle_sha256: Some("b".repeat(64)),
            pin: Some(PinReceipt {
                source: "bot.toml".into(),
                pin: "c".repeat(64),
            }),
            channel: Some(ChannelReceipt {
                source: "https://example.com/channel.json".into(),
                generated_ms: 1729999999000,
                observed_age_ms: 10_000,
                max_age_ms: 86_400_000,
            }),
            detail: Some(detail.into()),
            signer: None,
            prev: None,
            chain: None,
            sig: None,
        }
    }

    fn lines(home: &Path) -> Vec<String> {
        std::fs::read_to_string(receipts_path_under(home))
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect()
    }

    fn rewrite_lines(home: &Path, ls: &[String]) {
        std::fs::write(receipts_path_under(home), ls.join("\n") + "\n").unwrap();
    }

    #[test]
    fn append_read_roundtrip_is_ordered() {
        let home = tmp("append");
        entry("install", "weather", "first")
            .record_under(&home)
            .unwrap();
        entry("pin-rotate", "weather", "second")
            .record_under(&home)
            .unwrap();
        entry("remove", "weather", "third")
            .record_under(&home)
            .unwrap();
        let entries = read_under(&home);
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].action, "install");
        assert_eq!(entries[2].action, "remove");
        // Optional fields survive the JSONL round-trip.
        assert_eq!(entries[0].pin.as_ref().unwrap().source, "bot.toml");
        assert_eq!(entries[0].channel.as_ref().unwrap().observed_age_ms, 10_000);
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn receipts_are_signed_and_sigs_verify() {
        let home = tmp("signed-ok");
        entry("install", "weather", "a")
            .record_under(&home)
            .unwrap();
        entry("pin-rotate", "weather", "b")
            .record_under(&home)
            .unwrap();
        let v = read_verified_under(&home);
        assert!(matches!(v.integrity, LogIntegrity::Intact));
        assert_eq!(v.signed, 2);
        // Every appended line carries signer + sig; the per-box key
        // file pair exists and its .pub matches the signer.
        let ls = lines(&home);
        for l in &ls {
            let e: ReceiptEntry = serde_json::from_str(l).unwrap();
            assert!(e.signer.is_some() && e.sig.is_some(), "unsigned line: {l}");
            let signer = e.signer.as_deref().unwrap();
            assert!(signing::verify(
                signer,
                signing::receipt_payload(e.chain.as_deref().unwrap()).as_bytes(),
                e.sig.as_deref().unwrap()
            ));
        }
        let pub_hex = std::fs::read_to_string(pub_path_under(&home)).unwrap();
        let signer0: ReceiptEntry = serde_json::from_str(&ls[0]).unwrap();
        assert_eq!(pub_hex.trim(), signer0.signer.as_deref().unwrap());
        // Key files are 0600.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            for p in [seed_path_under(&home), pub_path_under(&home)] {
                let mode = std::fs::metadata(&p).unwrap().permissions().mode() & 0o777;
                assert_eq!(mode, 0o600, "{} mode {mode:o}", p.display());
            }
        }
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn tampered_signature_breaks_the_verdict_at_that_line() {
        let home = tmp("signed-tamper");
        entry("install", "weather", "a")
            .record_under(&home)
            .unwrap();
        entry("pin-rotate", "weather", "b")
            .record_under(&home)
            .unwrap();
        // Flip sig hex on line 1 in place (content + digests untouched).
        let mut ls = lines(&home);
        assert!(ls[1].contains("\"sig\""), "{}", ls[1]);
        ls[1] = ls[1].replacen("sig\":\"", "sig\":\"ffff", 1);
        rewrite_lines(&home, &ls);
        let v = read_verified_under(&home);
        match v.integrity {
            LogIntegrity::Broken { at, why } => {
                assert_eq!(at, 1, "{}", why);
                assert!(why.contains("signature does not verify"), "{}", why);
            }
            other => panic!("expected a signature break, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn chain_intacts_and_links_line_to_line() {
        let home = tmp("chain-ok");
        entry("install", "weather", "a")
            .record_under(&home)
            .unwrap();
        entry("verify", "weather", "b").record_under(&home).unwrap();
        entry("remove", "weather", "c").record_under(&home).unwrap();
        let v = read_verified_under(&home);
        assert_eq!(v.chained, 3);
        assert!(matches!(v.integrity, LogIntegrity::Intact));
        // Genesis for the first, then a real link per line.
        let ls = lines(&home);
        let e0: ReceiptEntry = serde_json::from_str(&ls[0]).unwrap();
        let e1: ReceiptEntry = serde_json::from_str(&ls[1]).unwrap();
        let e2: ReceiptEntry = serde_json::from_str(&ls[2]).unwrap();
        assert_eq!(e0.prev.as_deref(), Some(GENESIS_HASH));
        assert_eq!(e1.prev.as_deref(), e0.chain.as_deref());
        assert_eq!(e2.prev.as_deref(), e1.chain.as_deref());
        // The head witness names the true tail.
        let head = std::fs::read_to_string(head_path_under(&home)).unwrap();
        assert_eq!(head.trim(), e2.chain.as_deref().unwrap());
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn edited_content_breaks_the_chain_at_that_line() {
        let home = tmp("chain-edit");
        entry("install", "weather", "a")
            .record_under(&home)
            .unwrap();
        entry("pin-rotate", "weather", "b")
            .record_under(&home)
            .unwrap();
        entry("remove", "weather", "c").record_under(&home).unwrap();
        // In-place edit of line 1's detail WITHOUT touching its digest.
        let mut ls = lines(&home);
        assert!(ls[1].contains("\"detail\":\"b\""), "anchor: {}", ls[1]);
        let forged_detail = "\"detail\":\"".to_owned() + &"f".repeat(60) + "\"";
        ls[1] = ls[1].replace("\"detail\":\"b\"", &forged_detail);
        rewrite_lines(&home, &ls);
        let v = read_verified_under(&home);
        match v.integrity {
            LogIntegrity::Broken { at, why } => {
                assert_eq!(at, 1, "{why}");
                assert!(why.contains("digest does not match"), "{why}");
            }
            other => panic!("expected a break, got {other:?}"),
        }
        // The entries are still read (nothing is hidden).
        assert_eq!(v.entries.len(), 3);
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn spliced_line_breaks_the_prev_linkage() {
        let home = tmp("chain-splice");
        entry("install", "weather", "a")
            .record_under(&home)
            .unwrap();
        entry("remove", "weather", "c").record_under(&home).unwrap();
        // Forged "verify" inserted between them, no chain fields.
        let forged = r#"{"ts_ms":1730000000001,"action":"verify","plugin":"weather","version":"0.1.1","detail":"forged"}"#;
        let mut ls = lines(&home);
        ls.insert(1, forged.to_string());
        rewrite_lines(&home, &ls);
        let v = read_verified_under(&home);
        match v.integrity {
            LogIntegrity::Broken { at, why } => {
                assert_eq!(at, 1, "{why}");
                assert!(why.contains("unchained line"), "{why}");
            }
            other => panic!("expected a break, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn truncation_is_caught_by_the_head_witness() {
        let home = tmp("chain-trunc");
        entry("install", "weather", "a")
            .record_under(&home)
            .unwrap();
        entry("pin-rotate", "weather", "b")
            .record_under(&home)
            .unwrap();
        // Delete the last line but leave the head file at its digest.
        let ls = lines(&home);
        assert_eq!(ls.len(), 2);
        rewrite_lines(&home, &ls[..1]);
        let v = read_verified_under(&home);
        match v.integrity {
            LogIntegrity::Truncated { head, last } => {
                assert_ne!(head, last, "head {} vs last {}", &head[..8], &last[..8]);
            }
            other => panic!("expected truncation, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn legacy_pre_chain_lines_verify_intact() {
        let home = tmp("legacy");
        // A pre-chain log from the original shape: no prev/chain fields.
        let legacy = r#"{"ts_ms":1729999990000,"action":"install","plugin":"pet","version":"0.1.0","detail":"drop: legacy"}"#;
        rewrite_lines(&home, &[legacy.to_string()]);
        // The FIRST chained entry chains from genesis after a legacy
        // prefix; nothing breaks, the prefix just rides uncovered.
        entry("remove", "pet", "after upgrade")
            .record_under(&home)
            .unwrap();
        let v = read_verified_under(&home);
        assert_eq!(v.chained, 1);
        assert_eq!(v.entries.len(), 2);
        assert!(matches!(v.integrity, LogIntegrity::Intact));
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn mid_log_signer_change_breaks_the_chain() {
        let home = tmp("rekey-splice");
        entry("install", "weather", "a")
            .record_under(&home)
            .unwrap();
        entry("pin-rotate", "weather", "b")
            .record_under(&home)
            .unwrap();
        // An attacker with filesystem access but no key material:
        // a HALF-job splice — rewrite line 1 coherently (content,
        // digest, new key, NEW signature) without touching line 0. Each
        // line verifies in isolation, but the signer key changes at
        // line 1 and that is exactly the re-key signature.
        let evil = SigningIdentity::generate();
        let mut ls = lines(&home);
        let e0: ReceiptEntry = serde_json::from_str(&ls[0]).unwrap();
        let mut e1: ReceiptEntry = serde_json::from_str(&ls[1]).unwrap();
        e1.signer = Some(evil.pubkey_hex());
        let digest = chain_hash(&e1).unwrap();
        e1.chain = Some(digest.clone());
        e1.sig = Some(evil.sign(signing::receipt_payload(&digest).as_bytes()));
        ls[1] = serde_json::to_string(&e1).unwrap();
        rewrite_lines(&home, &ls);
        rewrite_lines(&home, &ls);
        std::fs::write(
            head_path_under(&home),
            format!("{}\n", e1.chain.as_deref().unwrap()),
        )
        .unwrap();
        assert_ne!(e1.signer.as_deref(), e0.signer.as_deref());
        // Sanity: line 1 alone WOULD verify standalone (evil signed it).
        assert!(signing::verify(
            e1.signer.as_deref().unwrap(),
            signing::receipt_payload(e1.chain.as_deref().unwrap()).as_bytes(),
            e1.sig.as_deref().unwrap()
        ));
        // And its prev doesn't break the walk (e0 untouched, e1.prev
        // still points at e0's original digest).
        let v = read_verified_under(&home);
        match v.integrity {
            LogIntegrity::Broken { at, why } => {
                assert_eq!(at, 1, "{}", why);
                assert!(why.contains("signer key changed"), "{}", why);
            }
            other => panic!("expected a re-key break, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn unsigned_chained_lines_stay_intact_pre_sigs_shape() {
        let home = tmp("unsigned-legacy");
        // Hand-build a PRE-SIGNATURE log (what a 0.1.2 home holds):
        // chained-but-unsigned lines. Note stripping signer+sig from a
        // SIGNED line is NOT this test — the signer is chain-digest
        // content, so that rewrite breaks the digest (covered by the
        // edit-detection test); the legacy shape here was never signed.
        let mut e0 = entry("install", "weather", "a");
        e0.prev = Some(GENESIS_HASH.to_string());
        e0.chain = Some(chain_hash(&e0).unwrap());
        let mut e1 = entry("pin-rotate", "weather", "b");
        e1.prev = e0.chain.clone();
        e1.chain = Some(chain_hash(&e1).unwrap());
        let head_digest = e1.chain.clone().unwrap();
        let ls = vec![
            serde_json::to_string(&e0).unwrap(),
            serde_json::to_string(&e1).unwrap(),
        ];
        rewrite_lines(&home, &ls);
        std::fs::write(head_path_under(&home), format!("{head_digest}\n")).unwrap();
        let v = read_verified_under(&home);
        assert_eq!(v.chained, 2);
        assert_eq!(v.signed, 0);
        assert!(matches!(v.integrity, LogIntegrity::Intact));
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn half_signed_line_breaks_immediately() {
        let home = tmp("half-signed");
        entry("install", "weather", "a")
            .record_under(&home)
            .unwrap();
        // Signer attached but sig stripped — parses fine and is exactly
        // the incomplete shape a hand-crafted line would produce.
        let ls = lines(&home)
            .iter()
            .map(|l| {
                let mut e: ReceiptEntry = serde_json::from_str(l).unwrap();
                e.sig = None;
                serde_json::to_string(&e).unwrap()
            })
            .collect::<Vec<_>>();
        rewrite_lines(&home, &ls);
        let v = read_verified_under(&home);
        match v.integrity {
            LogIntegrity::Broken { at, why } => {
                assert_eq!(at, 0, "{}", why);
                assert!(why.contains("half-signed"), "{}", why);
            }
            other => panic!("expected a half-signature break, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn missing_log_reads_empty_and_torn_lines_skip() {
        let home = tmp("missing");
        assert!(read_under(&home).is_empty());

        // A torn tail (power loss mid-append) must not blackhole the log.
        let home2 = tmp("torn");
        entry("install", "weather", "kept")
            .record_under(&home2)
            .unwrap();
        let path = receipts_path_under(&home2);
        let mut raw = std::fs::read_to_string(&path).unwrap();
        raw.push_str("{\"ts_ms\": 1, \"actio");
        std::fs::write(&path, raw).unwrap();
        let entries = read_under(&home2);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].detail.as_deref(), Some("kept"));
        let _ = std::fs::remove_dir_all(&home);
        let _ = std::fs::remove_dir_all(&home2);
    }
}
