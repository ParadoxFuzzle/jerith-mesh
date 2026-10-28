# keys/

- `plugin-releases.hex` — **the first-party plugin-release signing
  seed** (32-byte hex). 0600, GITIGNORED, never leaves the maintainer
  box except as the GitHub secret `PLUGIN_RELEASE_SIGNING_KEY` that the
  release.yml `plugin-bundles` job injects to sign tag builds.
- `plugin-releases.pub` — the matching Ed25519 public key, hex. This is
  the COMMITTED trust anchor: it ships with the source, operators pin
  it (bot.toml's `[plugins.trusted_publishers]` — the primary table
  since 2026-10-28; the legacy `keys/plugin-pins.json` still resolves
  under it), and replacing it is a trust event, not a code change.

Fingerprint: `sed 's/\(..\)/\1:/g;s/:$//' keys/plugin-releases.pub`
i.e. first 8 hex chars printed by `jerith plugin verify`'s output line.
Regeneration procedure lives in docs/PLUGIN-CHANNEL.md (key rotation
is a deliberate, documented reset of both files + the secret).
