# Plugin bundle channel — the install-side design

Status: **Phase 2 ✅ implemented** (2026-10-28) — the `.jbundle`
engine lives in `crates/core/src/pluginbundle.rs` (build / the strict
verify chain / extract / stage, home-injectable and fully test-pinned)
and the human surface is `jerith plugin install|verify|remove|build`.
**Phase 4 ✅ implemented same day** — the signed channel index lives in
`crates/core/src/pluginchannel.rs` (build/sign/verify with key pinning
+ a freshness window) behind `jerith plugin channel build|verify`, and
`release.yml` carries the full tag pipeline: `plugin-bundles` (both
platforms sign the weather triple) and `channel-index` (one signed
index over the gathered rows, self-verified against the committed
anchor before upload). The FIRST-PARTY KEY IS LIVE: the public half is
the committed trust anchor `keys/plugin-releases.pub`
(1e715b6671b3c2de4d3f64849cdc07cafcba30c11a29d5738fe8c7f25cc3538f),
the private hex seed rides `keys/plugin-releases.hex` (0600,
gitignored — see `keys/README.md`) and the GitHub secret
`PLUGIN_RELEASE_SIGNING_KEY`; the generator is
`scripts/first-party-bundle.sh` (build + sign + self-verify against a
throwaway home pinned to the committed pub — CI runs the same script,
so what ships is what was verified). The weather sample was produced
and locally verified on 2026-10-28 (dist/, gitignored — artifacts are
release-side, not repo-side).
**Phase 3 + search ✅ implemented same day** — the operator trust
lane lives in `jerith plugin trust show|rotate` over bot.toml's
`[plugins.trusted_publishers]` (via toml_edit — comments and other
tables survive rotation verbatim), pins-file-vs-table priority is
bot.toml-first with the patches file as a LEGACY backstop, and
`jerith plugin search` rides `channel verify`'s full gate (verify
first, filter second; rows name artifacts, install re-verifies) —
verify-first means an unverified or stale index is never searched.
The only thing left unbuilt is the actual release cut that publishes
the sample triple.

Phase 2 CLI reality notes (differs from the early draft, deliberately):

- Pins live FIRST in bot.toml `[plugins.trusted_publishers]`
  (`"<id>" = "<64-hex pub>"` — Phase 3's primary table, listed by
  `jerith plugin trust show`, written by `rotate`); the
  `$JERITH_HOME|~/.jerith-mesh/keys/plugin-pins.json` file still
  resolves but is legacy — a bot.toml row OVERRANKS a file row, and
  `trust rotate` prints a naming note when the two disagree.
- `--pubkey` overrides both pin sources per run; an un-pinned install
  refuses and PRINTS the presented key as a ready-to-paste
  `[plugins.trusted_publishers]` row for the human to add after
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

- **CLI**: `crates/node/src/main.rs` `Cmd` enum —
  `Plugin install|verify|activate|remove|trust show|trust rotate|search|channel build|channel verify`.
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
2. **First-party bundles**: ✅ engine + CLI + key + release.yml
   signing job + the weather sample built and verified (despite the
   doc's early draft — the sample ships on the next tag cut, not from
   this repo branch).
3. **Third-party publishers**: ✅ `trusted_publishers` config,
   per-id pins, key rotation flow (`jerith plugin trust show|rotate` —
   the table is bot.toml's `[plugins.trusted_publishers]`, rotation
   goes through toml_edit so operator comments survive verbatim,
   and a disagreeing legacy pins-file row gets named, not silently
   outranked forever), plugin removal/re-download journey documented in
   RUNBOOK under "Author a plugin".
4. **Signed channel index**: ✅ `pluginchannel.rs` +`jerith plugin channel build|verify` + the release.yml `channel-index` gather job.
   `channel.json` + `channel.json.sig` per publisher: the canonical
   payload is `jerith-chidx1|<publisher>|<generated_ms>|<sha256(bytes)>`
   — the sig binds the EXACT received bytes; the freshness window (24 h
   default, `--max-age-ms`; future-stamps beyond a 10 min tolerance
   refuse) keeps an abandoned lane from posing as a maintained one; and
   the anchor NEVER rides the wire (`--anchor` or the committed
   `keys/plugin-releases.pub` — the file's own `publisher` field must
   equal it; a self-signed index is worthless by construction). The
   index is a CLAIM SHEET: installing a row still runs the full bundle
   verify chain against the actual artifact.   `jerith plugin search` on
   top of it shipped the same day: `--channel <url|dir>` (or
   $JERITH_CHANNEL_URL) — verify FIRST (pinning + freshness, the full
   channel checks), filter SECOND (case-insensitive substring over
   id/version/os/arch/url, with an «installed version» marker from the
   local bundle store). An unverified or stale index is never searched.

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
