#!/usr/bin/env python3
"""The Python plugin-template example — same shape as
`crates/plugin-template/src/main.rs`, same `plugin.toml`:

    mkdir -p ~/.jerith-mesh/plugins.d/example
    cp crates/plugin-template/plugin.toml ~/.jerith-mesh/plugins.d/example/
    python3 sdk/python/example_plugin.py ~/.jerith-mesh/plugins.d/example/plugin.toml
    # add "external:example" to [plugins] enabled in bot.toml, restart the bot

Authoring story: copy this file, replace `echo`/`roll` with real
handlers, keep the manifest's `[[skill]]` rows in sync — the runner
fails loud at boot on any mismatch (a typo never becomes a silent dead
lane). Everything wire-shaped below the handlers is the SDK's job.
"""

import sys
from time import time_ns

from jerith_plugin import Plugin


def main() -> None:
    # ONE source of truth for the endpoint line: the drop's manifest.
    manifest = sys.argv[1] if len(sys.argv) > 1 else "plugin.toml"
    p = Plugin.from_manifest(manifest)

    @p.skill("echo")
    def echo(args: dict) -> dict:
        text = str(args.get("text", "")).strip()
        if not text:
            raise ValueError("args.text must be a non-empty string")
        return {"echo": text}

    @p.skill("roll")
    def roll(args: dict) -> dict:
        sides = int(args.get("sides", 6))
        if not 0 < sides <= 1000:
            raise ValueError("args.sides must be 1..=1000")
        return {"sides": sides, "roll": (time_ns() % sides) + 1}

    p.run()


if __name__ == "__main__":
    main()
