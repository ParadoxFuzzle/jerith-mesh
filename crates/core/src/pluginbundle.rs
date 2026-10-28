//! Signed plugin bundle channel — the `.jbundle` engine (Phase 2 of
//! `docs/PLUGIN-CHANNEL.md`).
//!
//! A bundle is a deterministic tar.gz (fixed member order, zeroed
//! mtimes/uids) of exactly one root directory `<id>-<version>/` holding
//! `bundle.toml` (install metadata), `plugin.toml` (the runtime
//! manifest), `README.md` and `bin/<binary>`. Two detached companions
//! ride beside it on every distribution surface:
//!
//! - `<bundle>.sha256` — hex of the tarball bytes (`<hex>  <name>` form
//!   also accepted). The transport check: fails fast before any crypto
//!   or parse work.
//! - `<bundle>.sig` — hex Ed25519 by the publisher over
//!   [`signing::plugin_bundle_payload`](`crate::signing::plugin_bundle_payload`)
//!   = prefix|id|version|tarball sha256. The identity check that BINDS
//!   the exact archive bytes: changing anything inside (even a README
//!   line) changes the tarball hash and breaks the signature.
//!
//! Verify chain, strictly ordered, every refusal naming the offense:
//! sha256 → structure → `bundle.toml` fields+root naming → signature →
//! publisher pin → platform → min node contract → inner binary hash.
//! Only after ALL of it pass does anything touch disk
//! ([`extract_installed`]), under the node's home convention
//! (`$JERITH_HOME|~/.jerith-mesh`).
//!
//! The HMAC secret never travels: [`inject_fresh_secret`] rewrites the
//! extracted manifest's `secret` with a fresh box-side value
//! (surgically, keeping the author's comments), and [`stage_drop`]
//! copies that patched manifest into the live drop
//! (`<home>/plugins.d/<id>/plugin.toml`). The bot's own fail-closed
//! manifest scan at every boot remains the runtime authority: whatever
//! lands in `plugins.d` must survive the drop scan or the bot refuses to
//! start, naming the drop — this module's structural manifest check is
//! the install-time pre-check, not a second parser authority.
//!
//! CLI surface (`jerith plugin …`) lives in crates/node; this module is
//! the engine (home-root injectable, so tests never touch the real home).

use std::cmp::Ordering;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::Context as _;
use sha2::{Digest, Sha256};

use crate::config::NodeConfig;
use crate::signing::{SigningIdentity, plugin_bundle_payload, verify};

/// Per-file read cap inside an archive (a gzip bomb can't inflate past this).
const MAX_ENTRY_BYTES: u64 = 64 * 1024 * 1024;
/// Total decompressed cap across the whole archive.
const MAX_TOTAL_BYTES: u64 = 256 * 1024 * 1024;
/// Entry count cap (pathological tar).
const MAX_ENTRIES: usize = 4096;

/// The `[bundle]` table of `bundle.toml` (serde view; validation after).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct BundleMeta {
    /// Plugin id — also the drop id and the model-visible skill prefix.
    pub id: String,
    pub version: String,
    /// Binary name inside `bin/` (the unit's ExecStart program).
    pub binary: String,
    /// sha256 of `bin/<binary>` bytes inside the archive.
    pub binary_sha256: String,
    /// Publisher's Ed25519 pubkey hex — the key that made `.sig`.
    pub publisher: String,
    /// Build target (compared against the installing machine).
    pub os: String,
    pub arch: String,
    /// Minimum `jerith` contract version this bundle supports.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_node: Option<String>,
    /// One-line changelog.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct BundleFile {
    bundle: BundleMeta,
}

/// Identifier rule — the same shape as the bot's `valid_skill_name`
/// (ASCII letter start, then letters/digits/`._-`). The bot's rule is
/// authoritative and pinned by its own tests; this mirrors it for the
/// install-time pre-check.
fn valid_name(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic())
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

/// Strict version shape: 1–3 dot-separated NUMERIC components, optional
/// `-pre-release` (alnum + dots) and `+build` metadata. e.g. `0.1.0`,
/// `1.10`, `1.0.0-rc.2`, `1.0.0+build.7`.
pub fn version_ok(v: &str) -> bool {
    let (core, rest) = match v.split_once(['-', '+']) {
        Some((c, r)) => (c, Some(r)),
        None => (v, None),
    };
    let comps: Vec<&str> = core.split('.').collect();
    if comps.is_empty()
        || comps.len() > 3
        || comps
            .iter()
            .any(|c| c.is_empty() || !c.chars().all(|c| c.is_ascii_digit()))
    {
        return false;
    }
    if let Some(rest) = rest {
        // "1.0+build.7" is fine — the non-dashed half is +build only.
        let had_pre = v.contains('-');
        let pre = rest.split_once('+').map_or(rest, |(pre, _)| pre);
        if (!pre.is_empty() || had_pre)
            && !pre.chars().all(|c| c.is_ascii_alphanumeric() || c == '.')
        {
            return false;
        }
        if !had_pre && pre.is_empty() {
            return false;
        }
    }
    true
}

/// Ordering for the downgrade guard. Both versions must be shape-valid
/// ([`version_ok`]) — anything else is an error (fail-closed: refuse
/// rather than guess). Core components compare numerically; the shorter
/// numeric list sorts lower; a `-pre-release` sorts BELOW the same core
/// without one; two pre-releases compare lexically (a slightly wrong
/// lexical order between exotic tags only ever REFUSES more, never
/// accepts a downgrade).
pub fn version_cmp(a: &str, b: &str) -> anyhow::Result<Ordering> {
    // Numeric core + optional pre-release part, owned (no borrow games).
    fn parse(v: &str) -> anyhow::Result<(Vec<u64>, Option<String>)> {
        anyhow::ensure!(version_ok(v), "{v:?}: not a valid version shape");
        let (core, extra) = v.split_once('-').map_or((v, None), |(c, r)| (c, Some(r)));
        let core = core.split_once('+').map_or(core, |(c, _)| c);
        let core: Vec<u64> = core.split('.').map(|c| c.parse().unwrap_or(0)).collect();
        let pre = extra.map(|e| e.split_once('+').map_or(e, |(pre, _)| pre).to_string());
        Ok((core, pre))
    }
    let (acore, apre) = parse(a)?;
    let (bcore, bpre) = parse(b)?;
    for (ac, bc) in acore.iter().zip(bcore.iter()) {
        match ac.cmp(bc) {
            Ordering::Equal => continue,
            other => return Ok(other),
        }
    }
    match acore.len().cmp(&bcore.len()) {
        Ordering::Equal => {}
        other => return Ok(other),
    }
    match (apre.as_deref(), bpre.as_deref()) {
        (Some(x), Some(y)) => Ok(x.cmp(y)),
        (Some(_), None) => Ok(Ordering::Less),
        (None, Some(_)) => Ok(Ordering::Greater),
        (None, None) => Ok(Ordering::Equal),
    }
}

/// A fresh 64-hex HMAC secret (the install-side rotation; never shipped).
pub fn fresh_secret() -> String {
    hex::encode(rand::random::<[u8; 32]>())
}

/// Exactly 64 lowercase hex chars.
fn hex_strict(s: &str) -> Result<(), String> {
    if s.len() != 64
        || !s
            .chars()
            .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c))
    {
        return Err(format!("{s:?} is not exactly 64 lowercase hex chars"));
    }
    Ok(())
}

impl BundleMeta {
    pub fn parse(raw: &str) -> anyhow::Result<Self> {
        let file: BundleFile =
            toml::from_str(raw).context("not valid TOML (need exactly one [bundle] table)")?;
        Ok(file.bundle)
    }

    /// Field-level fail-closed validation — every rule names its field.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            valid_name(&self.id),
            "bundle.id {:?} is not a valid identifier",
            self.id
        );
        anyhow::ensure!(
            version_ok(&self.version),
            "bundle.version {:?} is not dec-version shaped",
            self.version
        );
        anyhow::ensure!(
            !self.binary.is_empty()
                && !self.binary.contains('/')
                && self
                    .binary
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.')),
            "bundle.binary {:?} is not a plain binary name",
            self.binary
        );
        hex_strict(&self.binary_sha256).map_err(|e| anyhow::anyhow!("bundle.binary_sha256 {e}"))?;
        hex_strict(&self.publisher).map_err(|e| anyhow::anyhow!("bundle.publisher {e}"))?;
        anyhow::ensure!(
            !self.os.is_empty() && !self.arch.is_empty(),
            "bundle.os/arch must both be set (target parts)"
        );
        if let Some(m) = &self.min_node {
            anyhow::ensure!(
                version_ok(m),
                "bundle.min_node {m:?} is not dec-version shaped"
            );
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct BuildReport {
    pub bundle_path: PathBuf,
    pub sha_path: PathBuf,
    pub sig_path: PathBuf,
    pub tarball_sha256: String,
    pub meta: BundleMeta,
}

/// Build a bundle from a prepared source directory (the publisher's
/// staging shape: `bundle.toml`, `plugin.toml`, `README.md`,
/// `bin/<binary>`). Deterministic: fixed member order, mtime/uid/gid 0,
/// modes 0644 (0755 under `bin/`), so the same source bytes rebuild to
/// the same sha256 the `.sha256` companion publishes.
pub fn build(
    source_dir: &Path,
    out_path: &Path,
    publisher: &SigningIdentity,
) -> anyhow::Result<BuildReport> {
    anyhow::ensure!(
        source_dir.is_dir(),
        "source dir {} missing",
        source_dir.display()
    );
    let raw_meta = std::fs::read_to_string(source_dir.join("bundle.toml"))
        .context("source dir: bundle.toml missing")?;
    let meta = BundleMeta::parse(&raw_meta).context("source dir: bundle.toml invalid")?;
    meta.validate()?;
    let root = format!("{}-{}", meta.id, meta.version);
    // Build-time coherence: the key making the signature must be the key
    // the manifest CLAIMS as publisher — refuse before shipping a bundle
    // whose `.sig` could never verify.
    anyhow::ensure!(
        meta.publisher == publisher.pubkey_hex(),
        "bundle.publisher {}… does not match the signing key {}… — fix bundle.toml or use the right key",
        &meta.publisher[..8],
        &publisher.pubkey_hex()[..8]
    );
    anyhow::ensure!(
        std::fs::exists(source_dir.join("plugin.toml"))?,
        "source dir: plugin.toml missing (the runtime manifest is not optional)"
    );
    anyhow::ensure!(
        std::fs::exists(source_dir.join("README.md"))?,
        "source dir: README.md missing"
    );
    let bin_path = source_dir.join("bin").join(&meta.binary);
    anyhow::ensure!(
        std::fs::exists(&bin_path)?,
        "source dir: bin/{} missing (declared in bundle.toml)",
        meta.binary
    );
    let binary_bytes = std::fs::read(&bin_path)?;
    let actual = hex::encode(Sha256::digest(&binary_bytes));
    anyhow::ensure!(
        actual == meta.binary_sha256,
        "bin/{} hash {actual} ≠ bundle.toml binary_sha256 — fix the manifest or rehash",
        meta.binary
    );

    // The archive: one root dir named <id>-<version>, members sorted.
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    let mut tar = tar::Builder::new(gz);
    push_dir(&mut tar, &root)?;
    let plugin_bytes = std::fs::read(source_dir.join("plugin.toml"))?;
    let readme_bytes = std::fs::read(source_dir.join("README.md"))?;
    push_file(
        &mut tar,
        &format!("{root}/bundle.toml"),
        raw_meta.as_bytes(),
        0o644,
    )?;
    push_file(
        &mut tar,
        &format!("{root}/plugin.toml"),
        &plugin_bytes,
        0o644,
    )?;
    push_file(&mut tar, &format!("{root}/README.md"), &readme_bytes, 0o644)?;
    push_dir(&mut tar, &format!("{root}/bin"))?;
    push_file(
        &mut tar,
        &format!("{root}/bin/{}", meta.binary),
        &binary_bytes,
        0o755,
    )?;
    let gz = tar.into_inner()?;
    let tarball = gz.finish()?;

    // Companions: sha256 (transport) + detached sig over the payload that
    // binds id|version|tarball-sha256 (identity).
    let sha = hex::encode(Sha256::digest(&tarball));
    let sha_path = companion(out_path, ".sha256");
    std::fs::write(&sha_path, format!("{sha}  {}\n", out_path.display()))?;
    let payload = plugin_bundle_payload(&meta.id, &meta.version, &sha);
    let sig_path = companion(out_path, ".sig");
    std::fs::write(
        &sig_path,
        format!("{}\n", publisher.sign(payload.as_bytes())),
    )?;
    std::fs::write(out_path, &tarball)?;

    Ok(BuildReport {
        bundle_path: out_path.to_path_buf(),
        sha_path,
        sig_path,
        tarball_sha256: sha,
        meta,
    })
}

fn companion(path: &Path, suffix: &str) -> PathBuf {
    let mut s = path.as_os_str().to_os_string();
    s.push(suffix);
    s.into()
}

fn push_dir(
    tar: &mut tar::Builder<flate2::write::GzEncoder<Vec<u8>>>,
    path: &str,
) -> anyhow::Result<()> {
    let mut header = tar::Header::new_gnu();
    header.set_size(0);
    header.set_entry_type(tar::EntryType::Directory);
    header.set_mode(0o755);
    header.set_mtime(0);
    header.set_uid(0);
    header.set_gid(0);
    tar.append_data(&mut header, path, std::io::empty())?;
    Ok(())
}

fn push_file(
    tar: &mut tar::Builder<flate2::write::GzEncoder<Vec<u8>>>,
    path: &str,
    bytes: &[u8],
    mode: u32,
) -> anyhow::Result<()> {
    let mut header = tar::Header::new_gnu();
    header.set_size(bytes.len() as u64);
    header.set_entry_type(tar::EntryType::Regular);
    header.set_mode(mode);
    header.set_mtime(0);
    header.set_uid(0);
    header.set_gid(0);
    tar.append_data(&mut header, path, bytes)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Verify chain
// ---------------------------------------------------------------------------

/// Everything needed for a strict verify, borrowed.
#[derive(Debug, Clone, Copy)]
pub struct VerifyOptions<'a> {
    /// Expected tarball sha256 (from the `.sha256` companion). When Some,
    /// it is enforced — never bypassed silently; a caller MAY pass None
    /// (structure+signature only) but the CLI always refuses missing
    /// companions, keeping the shipped chain strict.
    pub expected_sha256: Option<&'a str>,
    /// Hex of the detached publisher signature (the `.sig` companion).
    pub sig_hex: &'a str,
    /// Pinned expected publisher pubkey — when Some, the bundle's own
    /// `publisher` must match EXACTLY (no trust-on-first-use).
    pub pinned_pubkey: Option<&'a str>,
    /// Installing platform (compare os/arch).
    pub platform_os: &'a str,
    pub platform_arch: &'a str,
    /// This CLI's version — the `min_node` contract is checked against it.
    pub node_contract_version: Option<&'a str>,
}

/// Everything verified, nothing yet on disk (extraction is a separate,
/// explicit step so callers can audit between verify and install).
#[derive(Debug, Clone)]
pub struct VerifiedBundle {
    pub meta: BundleMeta,
    pub plugin_toml: String,
    pub readme: String,
    /// The bundle.toml raw text (audit copy, unchanged).
    pub bundle_toml_raw: String,
    pub binary_name: String,
    pub binary: Vec<u8>,
    pub tarball_sha256: String,
}

/// The strict chain. Refusal messages all start `bundle:` and name the
/// offending rule/field, in order.
pub fn verify_bytes(raw: &[u8], opts: &VerifyOptions<'_>) -> anyhow::Result<VerifiedBundle> {
    // 1. Transport integrity (fast, before any crypto or parse work).
    let actual = hex::encode(Sha256::digest(raw));
    if let Some(expected) = opts.expected_sha256 {
        anyhow::ensure!(
            actual == expected,
            "bundle: sha256 mismatch — transport or mirror tamper (computed {actual}, expected {expected})"
        );
    }

    // 2. Structure: valid gzip+tar, bounded sizes, plain files/dirs only,
    //    no path escapes, exactly one top-level root dir.
    let gz = flate2::read::GzDecoder::new(std::io::Cursor::new(raw));
    let mut archive = tar::Archive::new(gz);
    let mut entries: Vec<(String, tar::EntryType, Vec<u8>)> = Vec::new();
    let mut total: u64 = 0;
    for entry in archive.entries()? {
        let entry = entry.context("bundle: unreadable tar entry")?;
        // Directory members carry a trailing `/` in some writers — trim
        // it so name rules and the root filter are reader-agnostic.
        let path = entry
            .path()?
            .to_string_lossy()
            .trim_end_matches('/')
            .to_owned();
        anyhow::ensure!(
            !path.is_empty() && !path.starts_with('/') && !path.contains(".."),
            "bundle: path escape in archive member {path:?}"
        );
        let kind = entry.header().entry_type();
        anyhow::ensure!(
            matches!(kind, tar::EntryType::Regular | tar::EntryType::Directory),
            "bundle: member {path:?} is a symlink/hardlink/device — never allowed in a bundle"
        );
        let mut bytes = Vec::new();
        let mut limited = entry.take(MAX_ENTRY_BYTES.saturating_add(1));
        limited
            .read_to_end(&mut bytes)
            .context("bundle: member read failed")?;
        anyhow::ensure!(
            bytes.len() as u64 <= MAX_ENTRY_BYTES,
            "bundle: member {path:?} exceeds the {MAX_ENTRY_BYTES}-byte per-member cap"
        );
        total += bytes.len() as u64;
        anyhow::ensure!(
            total <= MAX_TOTAL_BYTES,
            "bundle: archive inflates past the {MAX_TOTAL_BYTES}-byte total cap"
        );
        entries.push((path, kind, bytes));
        anyhow::ensure!(
            entries.len() <= MAX_ENTRIES,
            "bundle: more than {MAX_ENTRIES} members"
        );
    }
    anyhow::ensure!(!entries.is_empty(), "bundle: archive is empty");
    // Exactly one top-level directory is the root; everything else lives
    // under it.
    let top_dirs: Vec<String> = entries
        .iter()
        .filter(|(p, k, _)| *k == tar::EntryType::Directory && !p.contains('/'))
        .map(|(p, _, _)| p.clone())
        .collect();
    anyhow::ensure!(
        top_dirs.len() == 1,
        "bundle: archive must have exactly one top-level directory (found {})",
        top_dirs.len()
    );
    let root = top_dirs.into_iter().next().unwrap();

    // 3. bundle.toml: parse + field validation, and the root dir name
    //    must be exactly <id>-<version> (a mismatch names both).
    //    The root dir member itself is accounted by `root` (dropped from
    //    the set so the remaining-member check only judges bin/).
    entries.retain(|(p, k, _)| !(*k == tar::EntryType::Directory && p == &root));
    let bundle_raw_bytes =
        take_member(&mut entries, &format!("{root}/bundle.toml"), "bundle.toml")?;
    let bundle_raw = String::from_utf8(bundle_raw_bytes)
        .map_err(|e| anyhow::anyhow!("bundle: bundle.toml is not valid utf-8 ({e})"))?;
    let meta =
        BundleMeta::parse(&bundle_raw).map_err(|e| anyhow::anyhow!("bundle: bundle.toml: {e}"))?;
    meta.validate()?;
    anyhow::ensure!(
        root == format!("{}-{}", meta.id, meta.version),
        "bundle: root dir {root:?} does not match bundle.toml id-version {}-{}",
        meta.id,
        meta.version
    );

    // 4. Signature over prefix|id|version|tarball-sha256 — the identity
    //    check that binds the exact bytes (any inside tamper refails it).
    let payload = plugin_bundle_payload(&meta.id, &meta.version, &actual);
    anyhow::ensure!(
        verify(&meta.publisher, payload.as_bytes(), opts.sig_hex),
        "bundle: signature does not verify with publisher key {}… over version {}",
        &meta.publisher[..8.min(meta.publisher.len())],
        meta.version
    );
    if let Some(pin) = opts.pinned_pubkey {
        anyhow::ensure!(
            meta.publisher.eq_ignore_ascii_case(pin),
            "bundle: publisher {}… is not the pinned key — paste the presented key into the \
             pins file YOURSELF first (human in the loop, no trust-on-first-use)",
            &meta.publisher[..8]
        );
    }

    // 5. Platform.
    anyhow::ensure!(
        meta.os == opts.platform_os && meta.arch == opts.platform_arch,
        "bundle: wrong platform — bundle targets {}/{} but this machine is {}/{}",
        meta.os,
        meta.arch,
        opts.platform_os,
        opts.platform_arch
    );

    // 6. Node contract floor.
    if let (Some(min), Some(cur)) = (&meta.min_node, opts.node_contract_version) {
        anyhow::ensure!(
            version_cmp(min, cur)? != Ordering::Greater,
            "bundle: needs node contract {min} but this CLI is {cur}"
        );
    }

    // 7. Required members: plugin.toml, README.md, bin/<binary>; every
    //    other remaining member must be the bin/ dir or a bin/ file.
    let plugin_toml = take_member(
        &mut entries,
        &format!("{root}/plugin.toml"),
        "plugin.toml (the runtime manifest)",
    )?;
    let readme = take_member(&mut entries, &format!("{root}/README.md"), "README.md")?;
    let binary = take_member(
        &mut entries,
        &format!("{root}/bin/{}", meta.binary),
        &format!("bin/{}", meta.binary),
    )?;
    anyhow::ensure!(
        entries.iter().all(|(p, k, _)| {
            (*k == tar::EntryType::Directory && p == &format!("{root}/bin"))
                || (*k == tar::EntryType::Regular && p.starts_with(&format!("{root}/bin/")))
        }),
        "bundle: unexpected members beyond the declared set: {}",
        entries
            .iter()
            .map(|(p, _, _)| p.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );

    // 8. Inner binary hash — what bundle.toml promises is what bin/ has.
    let inner = hex::encode(Sha256::digest(&binary));
    anyhow::ensure!(
        inner == meta.binary_sha256,
        "bundle: bin/{} hash {inner} ≠ bundle.toml binary_sha256",
        meta.binary
    );

    let binary_name = meta.binary.clone();
    Ok(VerifiedBundle {
        meta,
        plugin_toml: String::from_utf8(plugin_toml)
            .map_err(|e| anyhow::anyhow!("bundle: plugin.toml is not valid utf-8 ({e})"))?,
        readme: String::from_utf8(readme)
            .map_err(|e| anyhow::anyhow!("bundle: README.md is not valid utf-8 ({e})"))?,
        bundle_toml_raw: bundle_raw,
        binary_name,
        binary,
        tarball_sha256: actual,
    })
}

/// Pull the required member `want` out of the collected list, leaving
/// the rest; refusal names the member.
fn take_member(
    entries: &mut Vec<(String, tar::EntryType, Vec<u8>)>,
    want: &str,
    label: &str,
) -> anyhow::Result<Vec<u8>> {
    let idx = entries
        .iter()
        .position(|(p, _, _)| p == want)
        .ok_or_else(|| anyhow::anyhow!("bundle: required member {label:?} missing ({want})"))?;
    let (_, _, bytes) = entries.remove(idx);
    Ok(bytes)
}

/// Pre-verify probe used by the CLI to look up a per-id publisher pin
/// BEFORE the full chain runs: gunzip → find `*/bundle.toml` → parse +
/// validate fields (including max entry size and member count). Garbage
/// bundles fail here with the same named rules; the returned
/// `(id, version, publisher)` is not trusted on its own — the full
/// chain re-derives everything.
pub fn peek_identity(raw: &[u8]) -> anyhow::Result<(String, String, String)> {
    let gz = flate2::read::GzDecoder::new(std::io::Cursor::new(raw));
    let mut archive = tar::Archive::new(gz);
    let mut bundle_toml: Option<String> = None;
    for entry in archive.entries()? {
        let mut entry = entry.context("bundle: unreadable tar entry")?;
        let path = entry.path()?.to_string_lossy().into_owned();
        anyhow::ensure!(
            !path.starts_with('/') && !path.contains(".."),
            "bundle: path escape in archive member {path:?}"
        );
        if entry.header().entry_type() != tar::EntryType::Regular || !path.ends_with("/bundle.toml")
        {
            continue;
        }
        anyhow::ensure!(
            bundle_toml.is_none(),
            "bundle: more than one bundle.toml member ({path:?} and a previous one)"
        );
        let mut bytes = Vec::new();
        read_capped(&mut entry, &mut bytes)?;
        bundle_toml = Some(
            String::from_utf8(bytes)
                .map_err(|e| anyhow::anyhow!("bundle: bundle.toml is not valid utf-8 ({e})"))?,
        );
    }
    let raw_meta =
        bundle_toml.ok_or_else(|| anyhow::anyhow!("bundle: no bundle.toml member found"))?;
    let meta =
        BundleMeta::parse(&raw_meta).map_err(|e| anyhow::anyhow!("bundle: bundle.toml: {e}"))?;
    meta.validate()?;
    Ok((meta.id, meta.version, meta.publisher))
}

fn read_capped<R: Read>(entry: &mut R, bytes: &mut Vec<u8>) -> anyhow::Result<()> {
    let mut limited = entry.take(MAX_ENTRY_BYTES.saturating_add(1));
    limited
        .read_to_end(bytes)
        .context("bundle: member read failed")?;
    anyhow::ensure!(
        bytes.len() as u64 <= MAX_ENTRY_BYTES,
        "bundle: member exceeds the {MAX_ENTRY_BYTES}-byte cap"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Install: extract → fresh secret → stage the drop
// ---------------------------------------------------------------------------

/// Paths an install uses for one id+version under an explicit home root
/// (`home/plugins/<id>/<version>/`, `home/plugins.d/<id>/[plugin.toml]`).
pub fn install_paths_under(home: &Path, id: &str, version: &str) -> (PathBuf, PathBuf, PathBuf) {
    (
        home.join("plugins").join(id).join(version),
        home.join("plugins.d").join(id),
        home.join("plugins.d").join(id).join("plugin.toml"),
    )
}

/// The real-home variant (the CLI).
pub fn install_paths(id: &str, version: &str) -> (PathBuf, PathBuf, PathBuf) {
    install_paths_under(&NodeConfig::home(), id, version)
}

/// Version directories already installed for one id under an explicit
/// home (validation-filtered).
pub fn installed_versions_under(home: &Path, id: &str) -> Vec<String> {
    let dir = home.join("plugins").join(id);
    let Ok(rd) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut out: Vec<String> = rd
        .flatten()
        .filter(|p| p.path().is_dir())
        .filter_map(|p| p.file_name().into_string().ok())
        .filter(|n| version_ok(n))
        .collect();
    out.sort();
    out
}

/// Highest installed version under an explicit home (None when nothing
/// valid there).
pub fn latest_installed_under(home: &Path, id: &str) -> anyhow::Result<Option<String>> {
    let mut best: Option<String> = None;
    for v in installed_versions_under(home, id) {
        best = Some(match best {
            None => v,
            Some(b) => match version_cmp(&b, &v)? {
                Ordering::Less | Ordering::Equal => v,
                Ordering::Greater => b,
            },
        });
    }
    Ok(best)
}

/// The real-home variant (the CLI's downgrade guard).
pub fn latest_installed(id: &str) -> anyhow::Result<Option<String>> {
    latest_installed_under(&NodeConfig::home(), id)
}

/// Write a verified bundle into a version dir: 0700 dirs, 0600 files,
/// 0700 binary. The manifest written is the freshly-secreted one; the
/// original bundle.toml rides along 0600 for the audit trail. Returns
/// the fresh secret.
pub fn extract_installed(v: &VerifiedBundle, version_dir: &Path) -> anyhow::Result<String> {
    make_private_dir(version_dir)?;
    // Fresh secret FIRST (the bundle's shipped value is never kept).
    let (patched, secret) = inject_fresh_secret(&v.plugin_toml)?;
    write_private(&version_dir.join("plugin.toml"), patched.as_bytes())?;
    write_private(
        &version_dir.join("bundle.toml"),
        v.bundle_toml_raw.as_bytes(),
    )?;
    write_private(&version_dir.join("README.md"), v.readme.as_bytes())?;
    let bin_dir = version_dir.join("bin");
    make_private_dir(&bin_dir)?;
    let bin_path = bin_dir.join(&v.binary_name);
    std::fs::write(&bin_path, &v.binary)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&bin_path, std::fs::Permissions::from_mode(0o755))?;
    }
    Ok(secret)
}

fn make_private_dir(dir: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

fn write_private(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    std::fs::write(path, bytes)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

/// Replace or insert `secret = "<fresh 64-hex>"` inside the manifest's
/// `[plugin]` table — textually, so the author's comments survive.
/// Fail-closed: no (or several) `[plugin]` tables, or a duplicate secret
/// key, is an error naming the manifest.
pub fn inject_fresh_secret(plugin_toml: &str) -> anyhow::Result<(String, String)> {
    let secret = fresh_secret();
    let secret_line = format!("secret = {secret:?}");
    let mut has_plugin_table = false;
    for line in plugin_toml.lines() {
        let tbl = line
            .trim()
            .strip_prefix('[')
            .map(|t| t.strip_suffix(']').unwrap_or(t).trim());
        if tbl == Some("plugin") {
            anyhow::ensure!(
                !has_plugin_table,
                "manifest: more than one [plugin] table — refusing to touch the secret"
            );
            has_plugin_table = true;
        }
    }
    anyhow::ensure!(
        has_plugin_table,
        "manifest: no [plugin] table — not a valid drop schema, can't inject a secret"
    );

    let mut out: Vec<String> = Vec::new();
    let mut in_plugin = false;
    let mut replaced = false;
    for line in plugin_toml.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            let tbl = trimmed
                .strip_prefix('[')
                .and_then(|t| t.strip_suffix(']'))
                .unwrap_or(trimmed);
            in_plugin = tbl.trim() == "plugin";
        } else if in_plugin && trimmed.split('=').next().map(str::trim) == Some("secret") {
            anyhow::ensure!(
                !replaced,
                "manifest: duplicate secret key inside [plugin] — refusing"
            );
            out.push(secret_line.clone());
            replaced = true;
            continue;
        }
        out.push(line.to_string());
    }
    if !replaced {
        // Insert right after the `[plugin]` header line.
        let mut with_secret: Vec<String> = Vec::new();
        let mut inserted = false;
        for line in &out {
            with_secret.push(line.clone());
            if !inserted && line.trim() == "[plugin]" {
                with_secret.push(secret_line.clone());
                inserted = true;
            }
        }
        anyhow::ensure!(
            inserted,
            "manifest: [plugin] table exists but no header line found — refusing to inject a secret"
        );
        out = with_secret;
    }
    Ok((out.join("\n"), secret))
}

/// Stage the freshly-secreted manifest into the live drop dir
/// (`plugins.d/<id>/plugin.toml` — the only file the bot scans). The
/// binary NEVER rides in the drop: the systemd unit points at the
/// version dir's `bin/<binary>`.
pub fn stage_drop(version_dir: &Path, drop_dir: &Path) -> anyhow::Result<PathBuf> {
    std::fs::create_dir_all(drop_dir)?;
    let staged = drop_dir.join("plugin.toml");
    let content = std::fs::read_to_string(version_dir.join("plugin.toml"))
        .context("staging: extracted plugin.toml unreadable")?;
    std::fs::write(&staged, content)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&staged, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(staged)
}

/// `jerith plugin remove <id>`'s trash move: the drop is MOVED out
/// (never deleted) into `<home>/plugins.d/.trash/<id>-<stamp>`; the
/// downloaded bundle stays wherever it was fetched (the audit trail
/// holds). Returns the trash path, or None when no drop existed.
pub fn trash_drop(id: &str) -> anyhow::Result<Option<PathBuf>> {
    let (_, drop_dir, _) = install_paths(id, "_");
    if !drop_dir.is_dir() {
        return Ok(None);
    }
    let home = NodeConfig::home();
    let trash_root = home.join("plugins.d").join(".trash");
    std::fs::create_dir_all(&trash_root)?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let trash_path = trash_root.join(format!("{id}-{stamp}"));
    std::fs::rename(&drop_dir, &trash_path).with_context(|| {
        format!(
            "trash move: {} → {}",
            drop_dir.display(),
            trash_path.display()
        )
    })?;
    Ok(Some(trash_path))
}

/// Read one `.sha256` companion file: bare hex or `<hex>  <name>`.
/// Uppercase hex is accepted (normalized lowercase); anything else fails.
pub fn parse_sha256_file(raw: &str) -> anyhow::Result<String> {
    let token = raw.split_whitespace().next().unwrap_or("");
    anyhow::ensure!(
        token.len() == 64 && token.chars().all(|c| c.is_ascii_hexdigit()),
        "sha256 companion: not a sha256 line ({token:?})"
    );
    Ok(token.to_ascii_lowercase())
}

// ---------------------------------------------------------------------------
// Tests (unit pins for the whole chain)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const BIN_BYTES: &[u8] = b"#!/bin/sh\necho weather-stub\n";
    const MANIFEST: &str = "# a comment the operator should keep seeing\n[plugin]\nid = \"weather\"\nendpoint = \"http://127.0.0.1:9041\"\nsecret = \"PLACEHOLDER_ROTATE_ME\"\ntimeout_secs = 10\n\n[[skill]]\nname = \"now\"\ndescription = \"current weather\"\nparameters = { type = \"object\" }\n";

    fn tmp(tag: &str) -> PathBuf {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let dir = std::env::temp_dir().join(format!(
            "jerith-bundle-test-{}-{}-{tag}",
            std::process::id(),
            N.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&dir).expect("tmp dir");
        dir
    }

    fn meta(id: &str, version: &str, publisher: &str) -> BundleMeta {
        BundleMeta {
            id: id.into(),
            version: version.into(),
            binary: "jerith-weather".into(),
            binary_sha256: hex::encode(Sha256::digest(BIN_BYTES)),
            publisher: publisher.into(),
            os: std::env::consts::OS.into(),
            arch: std::env::consts::ARCH.into(),
            min_node: Some("0.1.0".into()),
            notes: Some("first".into()),
        }
    }

    fn render_meta(m: &BundleMeta) -> String {
        let min = m
            .min_node
            .as_ref()
            .map(|v| format!("min_node = {v:?}\n"))
            .unwrap_or_default();
        let notes = m
            .notes
            .as_ref()
            .map(|v| format!("notes = {v:?}\n"))
            .unwrap_or_default();
        format!(
            "[bundle]\nid = {:?}\nversion = {:?}\nbinary = {:?}\nbinary_sha256 = {:?}\npublisher = {:?}\nos = {:?}\narch = {:?}\n{min}{notes}",
            m.id, m.version, m.binary, m.binary_sha256, m.publisher, m.os, m.arch
        )
    }

    /// A publisher's staging dir whose publisher field names `signer`.
    fn ready_source(tag: &str, m: BundleMeta, _signer: &SigningIdentity) -> PathBuf {
        let dir = tmp(tag);
        std::fs::create_dir_all(dir.join("bin")).unwrap();
        std::fs::write(dir.join("bin/jerith-weather"), BIN_BYTES).unwrap();
        std::fs::write(dir.join("plugin.toml"), MANIFEST).unwrap();
        std::fs::write(dir.join("README.md"), "# weather plugin\n").unwrap();
        std::fs::write(dir.join("bundle.toml"), render_meta(&m)).unwrap();
        dir
    }

    fn build_bundle(tag: &str) -> (PathBuf, SigningIdentity, BundleMeta) {
        let signer = SigningIdentity::generate();
        let src = ready_source(
            &format!("{tag}-src"),
            meta("weather", "0.1.3", &signer.pubkey_hex()),
            &signer,
        );
        let out = tmp(&format!("{tag}-out")).join("weather-0.1.3.jbundle");
        let report = build(&src, &out, &signer).expect("build");
        (report.bundle_path, signer, report.meta)
    }

    /// Companion files beside a bundle: (lowercased sha, trimmed sig).
    fn companions(path: &Path) -> (String, String) {
        let sha = parse_sha256_file(
            &std::fs::read_to_string(format!("{}.sha256", path.display())).unwrap(),
        )
        .unwrap();
        let sig = std::fs::read_to_string(format!("{}.sig", path.display()))
            .unwrap()
            .trim()
            .to_string();
        (sha, sig)
    }

    fn opts<'a>(
        sha: &'a str,
        sig: &'a str,
        pin: Option<&'a str>,
        contract: Option<&'a str>,
    ) -> VerifyOptions<'a> {
        VerifyOptions {
            expected_sha256: if sha.is_empty() { None } else { Some(sha) },
            sig_hex: sig,
            pinned_pubkey: Some(pin.unwrap_or("")),
            platform_os: std::env::consts::OS,
            platform_arch: std::env::consts::ARCH,
            node_contract_version: contract,
        }
    }

    /// The happy path: build → verify → extract → stage, under a TEST
    /// home (never the real ~/.jerith-mesh), and what the secret story
    /// guarantees (fresh value, comments kept, perms).
    #[test]
    fn roundtrip_happy_path() {
        let (out, signer, _meta) = build_bundle("happy");
        let (sha, sig) = companions(&out);
        let v = verify_bytes(
            &std::fs::read(&out).unwrap(),
            &opts(&sha, &sig, Some(&signer.pubkey_hex()), Some("0.1.0")),
        )
        .expect("verify");
        assert_eq!(v.meta.id, "weather");
        assert_eq!(v.binary_name, "jerith-weather");
        // The sig the build wrote really is over the bound payload.
        let payload = plugin_bundle_payload("weather", "0.1.3", &sha);
        assert!(verify(&signer.pubkey_hex(), payload.as_bytes(), &sig));
        // The shipped placeholder must NOT be present in the first place...
        assert!(v.plugin_toml.contains("PLACEHOLDER_ROTATE_ME"));
        assert_eq!(v.tarball_sha256, sha);

        let home = tmp("install");
        let (vdir, _drop_dir, _drop_manifest) = install_paths_under(&home, "weather", "0.1.3");
        let secret = extract_installed(&v, &vdir).expect("extract");
        let drop_dir = install_paths_under(&home, "weather", "0.1.3").1;
        let staged = stage_drop(&vdir, &drop_dir).expect("stage");
        let drop_manifest = std::fs::read_to_string(&staged).unwrap();
        assert!(drop_manifest.contains(&secret), "{drop_manifest}");
        assert!(!drop_manifest.contains("PLACEHOLDER_ROTATE_ME"));
        // Author's comments survive the injection.
        assert!(drop_manifest.contains("a comment the operator should keep"));
        // Re-running the injection produces a DIFFERENT secret (rotation).
        let (_, secret2) = inject_fresh_secret(&v.plugin_toml).unwrap();
        assert_ne!(secret, secret2);
        // Permissions: private dirs/files, executable binary.
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            assert_eq!(
                std::fs::metadata(vdir.clone()).unwrap().mode() & 0o777,
                0o700
            );
            assert_eq!(std::fs::metadata(&staged).unwrap().mode() & 0o777, 0o600);
            assert_eq!(
                std::fs::metadata(vdir.join("bin/jerith-weather"))
                    .unwrap()
                    .mode()
                    & 0o777,
                0o755
            );
        }
    }

    /// peek_identity sees the id/version/publisher before any pin work.
    #[test]
    fn peek_identity_roundtrip() {
        let (out, signer, _meta) = build_bundle("peek");
        let (id, version, publisher) = peek_identity(&std::fs::read(&out).unwrap()).expect("peek");
        assert_eq!(id, "weather");
        assert_eq!(version, "0.1.3");
        assert_eq!(publisher, signer.pubkey_hex());
    }

    /// Flipping an archive byte fails the transport sha FIRST (named).
    #[test]
    fn tampered_tarball_fails_sha() {
        let (out, _signer, _meta) = build_bundle("tamper");
        let (sha, sig) = companions(&out);
        let mut raw = std::fs::read(&out).unwrap();
        let mid = raw.len() / 2;
        raw[mid] ^= 0xff;
        let err = verify_bytes(&raw, &opts(&sha, &sig, None, None)).expect_err("tamper refuses");
        assert!(err.to_string().contains("sha256 mismatch"), "{err}");
    }

    /// A DIFFERENT archive verified with the FIRST archive's sig: the
    /// sha layer is skipped (expected=None) and the signature still
    /// refuses — the sig binds the exact tarball bytes, not just the
    /// bundle.toml metadata.
    #[test]
    fn different_tarball_fails_signature_binding() {
        let signer = SigningIdentity::generate();
        let key = signer.pubkey_hex();
        let (out_a, _s, _m) = build_bundle("diffsig");
        let (_, sig_a) = companions(&out_a);
        // Build B: same meta, a plugin.toml comment changed.
        let src_b = ready_source("diffsig-b-src", meta("weather", "0.1.3", &key), &signer);
        std::fs::write(
            src_b.join("plugin.toml"),
            MANIFEST.replace("keep seeing", "keep SEENING"),
        )
        .unwrap();
        let out_b = tmp("diffsig-out-b").join("weather-0.1.3.jbundle");
        build(&src_b, &out_b, &signer).expect("build b");
        let err = verify_bytes(
            &std::fs::read(&out_b).unwrap(),
            &VerifyOptions {
                expected_sha256: None,
                sig_hex: &sig_a,
                pinned_pubkey: Some(&key),
                platform_os: std::env::consts::OS,
                platform_arch: std::env::consts::ARCH,
                node_contract_version: None,
            },
        )
        .expect_err("archive A's sig must refuse archive B");
        assert!(
            err.to_string().contains("signature does not verify"),
            "{err}"
        );
    }

    /// The forge-shape: an attacker's signature over the bound payload
    /// doesn't verify against the CLAIMED publisher key.
    #[test]
    fn wrong_publisher_key_refused() {
        let (out, _signer, _meta) = build_bundle("wrongkey");
        let (sha, _sig) = companions(&out);
        let attacker = SigningIdentity::generate();
        let payload = plugin_bundle_payload("weather", "0.1.3", &sha);
        let forged = attacker.sign(payload.as_bytes());
        let err = verify_bytes(
            &std::fs::read(&out).unwrap(),
            &opts(&sha, &forged, None, None),
        )
        .expect_err("forged sig refuses");
        assert!(
            err.to_string().contains("signature does not verify"),
            "{err}"
        );
    }

    /// No trust-on-first-use: a good signature by an UNPINNED key still
    /// refuses, naming the human step.
    #[test]
    fn unpinned_publisher_refused() {
        let (out, _signer, _meta) = build_bundle("pinless");
        let (sha, sig) = companions(&out);
        let bogus = hex::encode([7u8; 32]);
        let err = verify_bytes(
            &std::fs::read(&out).unwrap(),
            &opts(&sha, &sig, Some(&bogus), None),
        )
        .expect_err("pin mismatch refuses");
        assert!(err.to_string().contains("not the pinned key"), "{err}");
    }

    /// min_node above the running CLI = hard stop.
    #[test]
    fn min_node_contract_refused() {
        let (out, signer, _meta) = build_bundle("minnode");
        let (sha, sig) = companions(&out);
        let err = verify_bytes(
            &std::fs::read(&out).unwrap(),
            &opts(&sha, &sig, Some(&signer.pubkey_hex()), Some("0.0.1")),
        )
        .expect_err("min_node above CLI refuses");
        assert!(err.to_string().contains("node contract"), "{err}");
    }

    /// Wrong platform archive → named refusal.
    #[test]
    fn mismatched_platform_refused() {
        let (out, signer, _meta) = build_bundle("platform");
        let (sha, sig) = companions(&out);
        let other_os = if std::env::consts::OS == "linux" {
            "windows"
        } else {
            "linux"
        };
        let err = verify_bytes(
            &std::fs::read(&out).unwrap(),
            &VerifyOptions {
                platform_os: other_os,
                ..opts(&sha, &sig, Some(&signer.pubkey_hex()), None)
            },
        )
        .expect_err("platform mismatch refuses");
        assert!(err.to_string().contains("wrong platform"), "{err}");
    }

    /// Bad sha256 companion = the FIRST refusal, named.
    #[test]
    fn bad_sha256_companion_refused() {
        let (out, _signer, _meta) = build_bundle("badsha");
        let (_sha, sig) = companions(&out);
        let bogus = hex::encode([0u8; 32]);
        let err = verify_bytes(
            &std::fs::read(&out).unwrap(),
            &opts(&bogus, &sig, None, None),
        )
        .expect_err("sha mismatch refuses");
        assert!(err.to_string().contains("sha256 mismatch"), "{err}");
    }

    /// A required member missing (README stripped) is refused BY NAME.
    /// The archive is rebuilt (test-only) and signed afresh with the
    /// SAME key so the sig gate passes and the member rule is what
    /// fires (the structural refusal, not the signature one).
    #[test]
    fn missing_required_member_is_refused() {
        let (out, signer, _meta) = build_bundle("missing");
        let members = extract_members(&std::fs::read(&out).unwrap());
        let members: Vec<(String, Vec<u8>)> = members
            .into_iter()
            .filter(|(p, _)| !p.ends_with("README.md"))
            .collect();
        let raw = gzip_tar(&["weather-0.1.3", "weather-0.1.3/bin"], &members);
        // fresh sig over the rebuilt tarball (same publisher key)
        let sha = hex::encode(Sha256::digest(&raw));
        let payload = plugin_bundle_payload("weather", "0.1.3", &sha);
        let sig = signer.sign(payload.as_bytes());
        let err = verify_bytes(
            &raw,
            &VerifyOptions {
                expected_sha256: None, // intentional rebuild; sig already bound to THESE bytes
                sig_hex: &sig,
                pinned_pubkey: Some(&signer.pubkey_hex()),
                platform_os: std::env::consts::OS,
                platform_arch: std::env::consts::ARCH,
                node_contract_version: None,
            },
        )
        .expect_err("missing README refuses");
        assert!(err.to_string().contains("README.md"), "{err}");
    }

    /// Hand-built refusals: a root dir NOT named <id>-<version>, and a
    /// path escape — each refusing with its named rule.
    #[test]
    fn refused_structures_named() {
        // wrong root name: valid parse, but root ≠ id-version.
        let files = vec![
            (
                "nope-1.0.0/bundle.toml".to_string(),
                render_meta(&meta("weather", "0.1.3", &hex::encode([1u8; 32]))).into_bytes(),
            ),
            (
                "nope-1.0.0/bin/jerith-weather".to_string(),
                BIN_BYTES.to_vec(),
            ),
        ];
        let raw = gzip_tar(&["nope-1.0.0", "nope-1.0.0/bin"], &files);
        let err = verify_bytes(&raw, &opts("", "", None, None)).expect_err("wrong root refuses");
        assert!(err.to_string().contains("does not match"), "{err}");
        // path escape: the tar BUILDER refuses `..` paths, so rebuild a
        // valid-looking archive and patch the member's raw header name
        // field (with its checksum corrected, like a real attacker). The
        // verify chain must call it out by name.
        let mut unc = tar_bytes(&["esc"], &[("esc/jerith".to_string(), b"x".to_vec())]);
        // the file member's 512-byte header block, name field [0..100]
        let h = &mut unc[512..1024];
        h[..16].copy_from_slice(b"../escape/jerith");
        fix_cksum(h);
        let esc = gzip_bytes(&unc);
        let err = verify_bytes(&esc, &opts("", "", None, None)).expect_err("escape refuses");
        assert!(err.to_string().contains("path escape"), "{err}");
    }

    /// version ordering + shape pins (the downgrade guard's grammar).
    #[test]
    fn version_ordering_pins() {
        assert_eq!(version_cmp("1.2.3", "1.10.0").unwrap(), Ordering::Less);
        assert_eq!(version_cmp("1.2.3", "1.2.3").unwrap(), Ordering::Equal);
        assert_eq!(version_cmp("1.0", "1.0.0").unwrap(), Ordering::Less);
        assert_eq!(version_cmp("2.0", "1.999").unwrap(), Ordering::Greater);
        assert_eq!(version_cmp("1.0.0-rc.1", "1.0.0").unwrap(), Ordering::Less);
        assert_eq!(
            version_cmp("1.0.0-b", "1.0.0-a").unwrap(),
            Ordering::Greater
        );
        assert!(version_cmp("bogus", "1.0").is_err());
        assert!(version_ok("0.1.0"));
        assert!(version_ok("1.0.0-rc.2"));
        assert!(version_ok("1.0.0+build.7"));
        assert!(!version_ok("v1.0.0"));
        assert!(!version_ok("1.x"));
        assert!(!version_ok("1..0"));
        assert!(!version_ok(""));
    }

    /// Secret injection rules: replace in place, insert when absent,
    /// refuse without (or with several) [plugin] tables, and never touch
    /// other tables.
    #[test]
    fn secret_injection_rules() {
        // replace, keep comments, fresh value differs from the old.
        let (patched, secret) =
            inject_fresh_secret("# keep\n[plugin]\nsecret = \"OLD\"\nid = \"x\"\n").unwrap();
        assert!(patched.contains(&secret), "{patched}");
        assert!(!patched.contains("OLD"));
        assert!(patched.contains("# keep"));
        // insert when absent
        let (patched, secret) = inject_fresh_secret("# hi\n[plugin]\nid = \"x\"\n").unwrap();
        assert!(patched.contains(&format!("secret = {secret:?}")));
        // more than one [plugin] table → refuse
        assert!(inject_fresh_secret("[plugin]\n[other]\n[plugin]\n").is_err());
        // no [plugin] at all → refuse
        assert!(inject_fresh_secret("[other]\nsecret = \"x\"\n").is_err());
        // secret in an unrelated table stays untouched; the plugin table
        // gets its own.
        let (patched, _secret) =
            inject_fresh_secret("[other]\nsecret = \"keep\"\n[plugin]\nid = \"x\"\n").unwrap();
        assert!(patched.contains("\"keep\""));
        assert_eq!(patched.matches("secret =").count(), 2);
        // sub-tables named like plugin.* are NOT the plugin table.
        let (patched, _secret) =
            inject_fresh_secret("[plugin.robust]\nsecret = \"keep\"\n[plugin]\nid = \"x\"\n")
                .unwrap();
        assert!(patched.contains("\"keep\""));
    }

    /// Downgrade guard graunch: latest_installed_under picks the top
    /// valid version; an incomparable/invalid dir is skipped.
    #[test]
    fn latest_installed_under_pins() {
        let home = tmp("installs");
        let dir = home.join("plugins").join("weather");
        for v in ["0.1.0", "0.1.2", "0.1.10"] {
            std::fs::create_dir_all(dir.join(v)).unwrap();
        }
        std::fs::create_dir_all(dir.join("not-a-version")).unwrap();
        assert_eq!(
            latest_installed_under(&home, "weather").unwrap().as_deref(),
            Some("0.1.10")
        );
        // An empty install tree → None.
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(latest_installed_under(&home, "weather").unwrap(), None);
    }

    /// extract ALL members (dirs + files) from an archive (tests only).
    fn extract_members(raw: &[u8]) -> Vec<(String, Vec<u8>)> {
        let gz = flate2::read::GzDecoder::new(std::io::Cursor::new(raw));
        let mut ar = tar::Archive::new(gz);
        let mut files = Vec::new();
        for e in ar.entries().unwrap() {
            let mut e = e.unwrap();
            if e.header().entry_type() != tar::EntryType::Regular {
                continue;
            }
            let path = e.path().unwrap().to_string_lossy().into_owned();
            let mut bytes = Vec::new();
            e.read_to_end(&mut bytes).unwrap();
            files.push((path, bytes));
        }
        files
    }

    /// Uncompressed tar bytes with explicit dir members up front (tests only).
    fn tar_bytes(dirs: &[&str], files: &[(String, Vec<u8>)]) -> Vec<u8> {
        let mut b = tar::Builder::new(Vec::new());
        for d in dirs {
            let mut h = tar::Header::new_gnu();
            h.set_size(0);
            h.set_entry_type(tar::EntryType::Directory);
            h.set_mode(0o755);
            h.set_mtime(0);
            b.append_data(&mut h, *d, std::io::empty()).unwrap();
        }
        for (p, bytes) in files {
            let mut h = tar::Header::new_gnu();
            h.set_size(bytes.len() as u64);
            h.set_entry_type(tar::EntryType::Regular);
            h.set_mode(0o644);
            h.set_mtime(0);
            b.append_data(&mut h, p, bytes.as_slice()).unwrap();
        }
        b.into_inner().unwrap()
    }

    /// gzip raw tar bytes (tests only).
    fn gzip_bytes(unc: &[u8]) -> Vec<u8> {
        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        std::io::Write::write_all(&mut gz, unc).unwrap();
        gz.finish().unwrap()
    }

    /// tests-only gzip_tar over the shared helpers.
    fn gzip_tar(dirs: &[&str], files: &[(String, Vec<u8>)]) -> Vec<u8> {
        gzip_bytes(&tar_bytes(dirs, files))
    }

    /// Recompute a tar header's checksum field (offset 148..156, octal
    /// sum with the field itself spaces) — tests only.
    fn fix_cksum(h: &mut [u8]) {
        h[148..156].fill(b' ');
        let sum: u32 = h.iter().map(|b| *b as u32).sum();
        let field = format!("{sum:06o}\0 ");
        h[148..156].copy_from_slice(field.as_bytes());
    }
}
