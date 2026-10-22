#!/usr/bin/env python3
"""The Python half of the plugin SDK (Phase 3d, docs/PLUGIN-SDK.md).

One file, standard library only (Python 3.11+ for `tomllib`; earlier
versions pass the manifest fields by hand). A plugin author writes only
`main(args) -> dict` handlers — the runner owns everything else:

- the exact wire contract the bot's dispatcher speaks (`external.rs`):
  one POST round trip, `{"method":"invoke","skill":…,"args":…,"ctx":…}` in,
  a bounded JSON object out;
- the ONE signing recipe (`hooks.rs` ⇄ `external.rs` ⇄ the Rust SDK):
  `x-jerith-signature` = hex(HMAC-SHA256(secret, "{ts_ms}.{body}")) +
  `x-jerith-timestamp`;
- the `gcp-bridge` freshness gate (±5 minutes) so a sniffed request
  cannot replay forever;
- fail-loud boot checks: every `[[skill]]` manifest row must have a
  registered handler and vice versa — a typo never becomes a silent
  dead lane;
- honest typed errors: wrong signature → 403 before ANY handler runs;
  unknown skill → 200 `{"error": …}` the model reads and recovers from;
  oversize answers refused with the same 65536-byte cap the dispatcher
  enforces.

Usage (see `example_plugin.py` next to this file):

    from jerith_plugin import Plugin

    p = Plugin("example", secret="per-plugin-rotate-me")   # or Plugin.from_manifest("plugin.toml")

    @p.skill("echo")
    def echo(args):                                        # returns a JSON-able dict
        return {"echo": args["text"]}

    p.run()                                                # binds the manifest endpoint

Wait — the manifest declares `[[skill]] name = "echo"` UNQUALIFIED and
the bot qualifies it to `example.echo`; the runner does the same
qualification, so decorators use the short name.
"""

from __future__ import annotations

import hashlib
import hmac as hmac_mod
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

# Same caps both Rust halves enforce — shrink the answer, not the cap.
MAX_RESPONSE_BYTES = 65536
MAX_CLOCK_SKEW_MS = 5 * 60 * 1000


def _now_ms() -> int:
    import time

    return int(time.time() * 1000)


def _sign(secret: str, ts_ms: int, body: str) -> str:
    """`hex(HMAC-SHA256(secret, "{ts}.{body}"))` — the shared recipe."""
    mac = hmac_mod.new(secret.encode(), f"{ts_ms}.{body}".encode(), hashlib.sha256)
    return mac.hexdigest()


class Plugin:
    """One plugin service: handlers in, bot-dispatch traffic served."""

    def __init__(self, plugin_id: str, *, endpoint: str | None = None,
                 secret: str | None = None, timeout_secs: int = 10):
        self.id = plugin_id
        self.endpoint = endpoint or f"http://127.0.0.1:7788/{plugin_id}/rpc"
        self.secret = secret  # None = unsigned dev mode (loopback only)
        self.timeout_secs = max(1, min(30, timeout_secs))
        self._handlers: dict[str, Callable[[dict], Any]] = {}
        self._manifest: dict[str, Any] | None = None

    # ---- builders ------------------------------------------------------

    @classmethod
    def from_manifest(cls, path: str | Path) -> "Plugin":
        """Read `[plugin]` straight out of the drop's plugin.toml (the
        ONE source of truth both halves honor): endpoint line binds here
        and is declared there — the pair cannot drift."""
        try:
            import tomllib
        except ModuleNotFoundError as e:  # pragma: no cover
            raise SystemExit(
                "jerith_plugin: Python 3.11+ needed for tomllib — "
                "report plugin fields by hand instead."
            ) from e
        with open(path, "rb") as fh:
            manifest = tomllib.load(fh)
        plugin = manifest.get("plugin")
        if not plugin or "id" not in plugin or "endpoint" not in plugin:
            raise SystemExit(f"jerith_plugin: {path} lacks [plugin] id/endpoint")
        self = cls(
            plugin["id"],
            endpoint=plugin["endpoint"],
            secret=plugin.get("secret"),
            timeout_secs=plugin.get("timeout_secs", 10),
        )
        self._manifest = manifest
        return self

    def skill(self, short_name: str) -> Callable:
        """Register a handler under the manifest's short name — the
        model-visible name is qualified to `<id>.<name>` exactly as the
        drop parser does."""

        def deco(fn: Callable[[dict], Any]) -> Callable[[dict], Any]:
            full = self._qualify(short_name)
            if full in self._handlers:
                raise SystemExit(f"jerith_plugin: duplicate handler {full}")
            self._handlers[full] = fn
            return fn

        return deco

    def _qualify(self, name: str) -> str:
        return name if "." in name else f"{self.id}.{name}"

    # ---- boot fail-loudness --------------------------------------------

    def _crosscheck_manifest(self) -> list[str]:
        """Manifest rows ⇄ registered handlers must match exactly."""
        rows: list[str] = []
        if self._manifest:
            for row in self._manifest.get("skill", []):
                rows.append(self._qualify(row["name"]))
        missing_manifest = sorted(set(self._handlers) - set(rows)) if rows else []
        missing_handler = sorted(set(rows) - set(self._handlers))
        if rows:
            if missing_handler:
                raise SystemExit(
                    "jerith_plugin: manifest declares " + ", ".join(missing_handler)
                    + " with no @skill handler"
                )
            if missing_manifest:
                raise SystemExit(
                    "jerith_plugin: handlers registered but not in the manifest: "
                    + ", ".join(missing_manifest)
                )
        return [self._qualify(r["name"]) for r in self._manifest.get("skill", [])] if self._manifest else []

    # ---- serving -------------------------------------------------------

    def run(self) -> None:
        names = self._crosscheck_manifest()
        parsed = urlparse(self.endpoint)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        path = parsed.path or "/rpc"
        plugin, handlers, secret = self.id, self._handlers, self.secret

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):  # journaled via stderr, one line
                sys.stderr.write(f"plugin {plugin}: {fmt % args}\n")

            def do_POST(self):  # noqa: N802 - http.server name
                if self.path != path:
                    self._json(404, {"error": f"no route {self.path} — POST {path}"})
                    return
                blob = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                body = blob.decode("utf-8", "replace")
                if secret is not None:
                    why = _verify(secret, self.headers, body)
                    if why:
                        self._json(403, {"error": why})
                        return
                try:
                    req = json.loads(body)
                except json.JSONDecodeError:
                    self._json(400, {"error": "body is not JSON"})
                    return
                if req.get("method") != "invoke":
                    self._json(400, {"error": f"unknown method {req.get('method')!r}"})
                    return
                name = req.get("skill", "")
                fn = handlers.get(name)
                if fn is None:
                    self._json(200, {"error": f"no skill {name!r} on plugin {plugin!r}"})
                    return
                try:
                    out = fn(req.get("args") or {})
                except Exception as e:  # typed refusal, model-recoverable
                    self._json(200, {"error": f"{type(e).__name__}: {e}"})
                    return
                blob_out = json.dumps(out).encode()
                if len(blob_out) > MAX_RESPONSE_BYTES:
                    self._json(502, {"error": f"response {len(blob_out)}B > {MAX_RESPONSE_BYTES}B cap"})
                    return
                self._json(200, out)

            def _json(self, status: int, obj: dict) -> None:
                blob = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(blob)))
                self.end_headers()
                self.wfile.write(blob)

        server = ThreadingHTTPServer((parsed.hostname or "127.0.0.1", port), Handler)
        print(
            f"plugin {plugin}: serving POST {path} on "
            f"{parsed.hostname or '127.0.0.1'}:{port} "
            f"(skills: {', '.join(sorted(handlers)) or 'none registered'}, "
            f"{'signed' if secret is not None else 'UNSIGNED dev mode'})",
            file=sys.stderr,
        )
        server.serve_forever()


def _verify(secret: str, headers: Any, body: str) -> str | None:
    """Refusal reasons for a bad request — None means verified."""
    ts_raw = headers.get("x-jerith-timestamp")
    if not ts_raw:
        return "missing x-jerith-timestamp"
    try:
        ts = int(str(ts_raw).strip())
    except ValueError:
        return f"x-jerith-timestamp {ts_raw!r} is not a number"
    now = _now_ms()
    if abs(now - ts) > MAX_CLOCK_SKEW_MS:
        return f"stale or future-dated request (ts {ts}, now {now})"
    sig = headers.get("x-jerith-signature")
    if not sig:
        return "missing x-jerith-signature"
    if hmac_mod.compare_digest(_sign(secret, ts, body), str(sig).strip()):
        return None
    return "signature mismatch"
