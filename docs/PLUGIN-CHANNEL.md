# Plugin bundle channel — the install-side design

Status: **Phase 2 ✅ implemented** (2026-10-28) — the `.jbundle`
engine lives in `crates/core/src/pluginbundle.rs` (build / the strict
verify chain / extract / stage, home-injectable and fully test-pinned)
and the human surface is `jerith plugin install|verify|remove` in
crates/node. The keys/trust flow of Phase 3 (repo-committed
`keys/plugin-releases.pub`, `trusted_publishers` in bot.toml, rotation
commands) and the release.yml signing job stay **design, not built**.
Phase 4 (signed channel index) likewise. Authored 2026-10-28,
immediately after the release-artifact scope decision in
`.github/workflows/release.yml` settled that plugin binaries
deliberately DON'T ride releases — this channel is the revisit
trigger's real answer.

Phase 2 CLI reality notes (differs from the early draft, deliberately):

- Pins live in `$JERITH_HOME|~/.jerith-mesh/keys/plugin-pins.json`
  (`{"<id>": "<64-hex pub>"}`) — the Phase 3 `trusted_publishers`
  bot.toml table becomes the migration target.
- `--pubkey` overrides the pins file per run; an un-pinned install
  refuses and PRINTS the presented key for the human to paste after
  trusting the source (no trust-on-first-use).
- The extracted manifest's structural check is an install-time
  pre-check; the RUNTIME authority stays the bot's fail-closed
  `plugins.d` scan at every boot (`ExternalPlugin::scan_drops`) — no
  second parser authority ever ships.

---

## What exists today (the contracts this builds on)

- **A plugin drop is a manifest + a running server.**
  `~/.jerith-mesh/plugins.d/<id>/plugin.toml` declares an out-of-process
  HTTP endpoint + skills ([`crates/bot/src/external.rs`]); the bot
  scans drops at start (fail-closed) and gates dispatch through
  `[plugins] enabled = [... "external:<id>" ...]` in bot.toml.
- **The server reads the SAME manifest** for its endpoint + secret
  (`crates/weather-plugin/src/main.rs`), so one file is the shared
  contract between the bot's dispatcher and the plugin's listener.
- **One HMAC recipe both directions** (`hooks.rs` ingest,
  `external.rs` dispatch): `hex(HMAC-SHA256(secret, "{ts_ms}.{body}"))`.
- **Signing primitives already exist** (`crates/core/src/signing.rs`):
  Ed25519 `SigningIdentity` (generate/node_id/sign/verify), the
  canonical-payload habit, and the wire-auth header discipline that
  `/v1/peers` and the hooks lane both already carry.
- **The operator trust root is bot.toml.** Installing a binary grants
  nothing: skills only dispatch when `external:<id>` appears in
  `[plugins] enabled` (fail-closed everywhere else).
- **First-party plugin servers run under systemd user units** with
  per-plugin `ExecStart` (e.g. `jerith-ledger
  ~/.jerith-mesh/plugins.d/ledger/plugin.toml`).

## Goals

1. A third-party operator can install a plugin **without a source
   checkout**: one fetch, one verify, one command — matching how the
   mesh node itself is already consumable (tag-released `jerith` +
   sha256).
2. Verify before disk: signature and checksum checks happen BEFORE any
   file lands in the bot's live `plugins.d/`.
3. The HMAC secret never travels in any artifact. Everyone learns the
   channel's rule the same way as the in-tree rules: secrets are
   generated box-side at install time, and signatures cover the
   publisher's bytes, not the operator-local mutations.
4. Explicit human opt-in stays the last gate, exactly like the approval
   bridge: install never auto-enables — activate prints the precise
   bot.toml row and the operator adds it.

## Non-goals

- No in-bot plugin daemon / dynamic scan(IS: the bot's node stays
  long-running; installs are a CLI concern only.
- No marketplace / registry service; a signed directory index (Phase
  migration below) is the most this ever becomes.

## Artifact format

    <id>-<version>-<os>-<arch>.jbundle          # tar.gz, one dir at root
      bundle.toml                               # install metadata (see below)
      plugin.toml                               # the manifest (may carry a placeholder secret)
      README.md                                 # what it does, what it needs
      bin/<binary>                              # executable plugin server (one or more)

Companion files ride beside the bundle (release assets or any URL):

    <bundle>.sha256                              # sha256 of the tarball
    <bundle>.sig                                 # detached Ed25519 hex

`bundle.toml` (the channel-unique part — the manifest stays schema-exact
as the SDK defines it):

    [bundle]
    id = "ledger"
    version = "0.1.3"          # developer semver; never a downgrade target
    min_node = "0.1.0"         # minimum `jerith` CLI/SDK contract version
    os = "linux"; arch = "x86_64"
    binary_sha256 = "…"        # hash of bin/<binary> INSIDE the archive
    publisher = "<64-hex ed25519 pubkey>"
    notes = "<one-line changelog>"

Signature payload: `crate::signing::plugin_bundle_payload(id, version,
binary_sha256)` — the same canonical-string habit as
`advertise_payload`/`measurement_payload`. The `.sha256` file is the
transport integrity check (fast fail before crypto); the `.sig` is the
identity check; both are required.

## Key model (publisher ⇒ operator)

- **First-party publishing** signs with a project holding in the repo:
  `keys/plugin-releases.hex` (0600, gitignored) — the PUBLIC half
  (`plugin-releases.pub`, fingerprint printed by
  `jerith plugin trust show`) is committed, so the trust anchor ships
  with the source like the bootstrap URL does. Release
  workflow (`release.yml`) gains a bundle-signing step at tag cut; the
  jersey-identity manifest `bundle.toml` carries the current keyid.
- **Third-party publishers** are CREATOR-controlled, and pin the
  key per-plugin-id:

      [plugins.trusted_publishers]
      "weather" = "8b1f…"     # exact hex this plugin id must be signed by

  Install refuses an untrusted publisher, prints the presented
  fingerprint, and requires the operator to paste it into
  `trusted_publishers` first — the human-in-the-loop shape of the
  approval bridge, applied at the supply-chain layer.
- **Key rotation**: `jerith plugin trust rotate <id> --new <hex>` re-pins
  the key and REQUIRES the next install to re-verify; a signature by
  the old key is refused the moment rotation lands (no trust-lag).

## Install / activate flow

```
jerith plugin install <url|local-path>
  1 fetch bundle + .sha256 + .sig (+key pin when third-party)
  2 sha256 field vs. byte hash of the tarball        (fast fail)
  3 Ed25519 verify against the publisher key         (identity trip)
  4 bundle.toml sanity (id/version/os/arch/binary_sha256) — fail-closed,
    all rules named (the same soup as external::parse)
  5 extract to ~/.jerith-mesh/plugins/<id>/<version>/  (0700 dirs, 0600 secret)
    • generate a FRESH 32-hex secret for this box, patch the local
      plugin.toml with it (the shipped file may carry a placeholder —
      the SECRET NEVER TRAVELS; see goals)
    • the extracted manifest must survive ExternalPlugin::parse
      (the full runtime fail-closed validation)
  6 stage into ~/.jerith-mesh/plugins.d/<id>/ plugin.toml (the *live*
      drop dir) only after print-confirm:
        "next: add external:<id> to [plugins].enabled in bot.toml
         restart jerith-bot"
  7 print the exact systemd ExecStart for the bundled binary
```

`jerith plugin verify <path>` re-runs 2–5 without touching disk (for
picking apart third-party bundles). `jerith plugin remove <id>` moves
the drop out and disables its unit; it never deletes the downloaded
artifact (audit trail).

Update: re-run install; the CLI refuses a descending `version` unless
`--force` (downgrade protection, the model-router lesson on accidental
downgrades), reuses the existing secret in place (the pair stays valid
across versions unless `--rotate` also passed).

## Threat model (what each layer holds back)

| Threat                                | Held back by            |
|---------------------------------------|-------------------------|
| Shipping HTTP tamper                  | sha256 before sig work  |
| Wrong-publisher lure                  | Ed25519 over digest + per-id key pin |
| Secrets in the supply chain           | secret generated at install; bundle sig covers publisher bytes only, never the mutation  |
| Malformed/ransom manifest             | extracted plugin.toml must pass ExternalPlugin::parse (fail-closed) |
| installing = silently live            | plugins.d staging + explicit bot.toml enable row (the bot's `[plugins]` gate is untouched) |
| loopback contract dodged by a bundle  | parser's non-loopback-needshttps rule runs at install time on the local copy too |
| version downgrade attack              | version monotonicity vs. install dir (bundle refusals need --force) |
| bundle key minted to typosquat an id  | id+gate collision check (`check_external_collisions`) plus the per-id pin above |

## Where it plugs in (code anchors for the build)

- **CLI**: `crates/node/src/main.rs` `Cmd` enum — add
  `Plugin install|verify|activate|remove|trust show`.
- **Crypto**: `crates/core/src/signing.rs` (already has what both `node`
  and `bot` link). New: `plugin_bundle_payload(id, version,
  sha256_hex)`, one canonical string, signed with the SAME Ed25519.
- **Manifest**: everything revalidates through
  `external::ExternalPlugin::parse` — no second manifest parser ever
  exists (the bolts stay in ONE place).
- **Config**: new optional bot config keys
  `[plugins.trusted_publishers]` (map plugin-id → hex pubkey); default
  empty. The bot doesn't read this map (it never fetches bundles or
  secret), only `jerith` does — bot-side changes stay zero.
- **Release publishing**: `release.yml` gains a job that builds the
  four first-party plugins (pet/house/quarantine/ledger/weather? —
  only when the house: **weather is the flagship public first-target**),
  signs and attaches `jbundle`/`sha256`/`sig` triples to the same
  GitHub release. The revision of the release.yml artifact-scope note
  MUST ride the same commit that flips this on.

## Rollout phases

1. **Now (this doc)**: ~~decisions + shape; no binary code changes~~ ✅.
2. **First-party bundles**: ✅ engine + CLI + test pins (happy path,
   tampered tarball, wrong key, unpinned publisher, platform/contract
   refusal, downgrade ordering, trash-removal — all in
   `pluginbundle.rs`'s test mod; end-to-end smoke: openssl-signed
   bundle → verify → install → downgrade guard → remove). REMAINING:
   the `keys/plugin-releases.pub` first-party key + release.yml
   signing job (needs the maintainer's private half in a GitHub secret
   when CI signs), and the shipped weather bundle as the first public
   sample.
3. **Third-party publishers**: `trusted_publishers` config, per-id pins,
   key rotation flow, plugin removal/re-download journey documented in
   RUNBOOK under "Author a plugin".
4. **Signed channel index**: one `channel.json` per publisher (signed
   with the same identity, contains a list of bundle URLs + digests),
   enabling `jerith plugin search` WITHOUT becoming a registry service.

## Open questions (deliberately unresolved)

- Does Phase 2 sign the audit chain? (A `plugin.json` CLI receipt log
  seems reasonable and needs no crypto now.)
- Does the CLI auto-download from a pinned URL (the mesh bootstrap CDN
  URL?) or fetch ONLY from operator-supplied URLs? Phase 2's default
  is: operator-supplied URLs only; no automatic remote trust.
- macOS signing/notarization as a bundle requirement? (Only matters if
  macOS paddles ever become first-class; the release matrix already
  builds for them, but plugins run on the bot box, which is
  Linux-first.)

That's the shape. The house's rule stands here too: every layer that
exists, fail-closed, everything human-in-the-loop, fail-closed message
fileNames the drop and the offense.
