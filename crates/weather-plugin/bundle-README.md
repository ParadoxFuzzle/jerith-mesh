# The weather plugin (as an installed `.jbundle`)

The weather skill, operator view: one out-of-process plugin server
(`bin/jerith-weather`) serving `weather.now` and `weather.forecast`
from open-meteo.com — no API key, no key to steal, no rate meter. The
runtime contract is the same manifest the bot's dispatcher parses;
see `crates/weather-plugin/plugin.toml` and `docs/PLUGIN-SDK.md`.

## What an install does

`jerith plugin install <bundle-url-or-path>` — after the pinned
publisher chain (sha256 → Ed25519 over
`jerith-plbun-1|weather|<version>|<tarball-sha256>` → platform +
min-node checks) it extracts to
`~/.jerith-mesh/plugins/weather/<version>/`, generates a FRESH HMAC
secret box-side (the shipped `secret` line is a placeholder; the real
one never travels), validates the extracted manifest fail-closed
through the bot's parser, and stages the live drop at
`~/.jerith-mesh/plugins.d/weather/plugin.toml`.

## Turning it on (never automatic)

1. `bot.toml` → `[plugins] enabled = [..., "external:weather"]`
2. Point the plugin unit's `ExecStart` at the version dir's binary (the
   install output prints the exact line) and restart the unit.
3. Restart `jerith-bot` and chat: ask "what's the weather in Prague?"

## Rebuilding it (publishers)

`scripts/first-party-bundle.sh weather <version> "<release notes>"` —
builds `target/release/jerith-weather`, stages `bundle.toml` with the
inner-binary sha256, signs with `keys/plugin-releases.hex`, and
self-verifies the result against a throwaway home pinned to the
committed `keys/plugin-releases.pub`. The three output files (`jbundle`
+ `.sha256` + `.sig`) land under `dist/` and attach to the release.
