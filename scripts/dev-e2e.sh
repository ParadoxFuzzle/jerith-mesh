#!/bin/bash
# dev-e2e.sh — one-command live two-node verification stack.
#
# What it does (fully local, tears itself down on exit):
#   1. builds the workspace,
#   2. starts a local bootstrap (:8999) + a deterministic OpenAI-compatible
#      stub runtime (:8090),
#   3. `jerith setup`s two nodes (A: gateway 8900/rpc 9000, B: gateway 8910/
#      rpc 8911) with verify_percent & ringer_percent swept to 100,
#   4. runs crates/core/tests/cross_node_dispatch.rs (--ignored) — the test
#      that dials node B through the fail-closed cert-pinning verifier and
#      asserts the Phase 4 commitment binds the wire round-trip,
#   5. prints each node's live verification counters from /status,
#   6. kills every process it started (trap EXIT).
#
# Env overrides:
#   E2E_MODEL      curated model id (default glm-5.3-flash)
#   E2E_DIR        scratch dir (default /tmp/jerith-e2e — wiped each run)
#   E2E_KEEP=1     leave the stack running after the test (for poking /status)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/target/debug/jerith"
E2E_DIR="${E2E_DIR:-/tmp/jerith-e2e}"
MODEL="${E2E_MODEL:-glm-5.3-flash}"
KEEP="${E2E_KEEP:-0}"
BOOT_PORT=8999
RUNTIME_PORT=8090
PIDS=()

cleanup() {
  if [ "$KEEP" = "1" ]; then
    echo "==> E2E_KEEP=1: leaving the stack up (PIDs: ${PIDS[*]})"
    return
  fi
  echo "==> teardown"
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  # Scoped pkill as belt-and-braces: only THIS e2e's processes, never the
  # user's live services (release bootstrap, cloudflared, jerith_node.py).
  pkill -f "$E2E_DIR/runtime.py" 2>/dev/null || true
  pkill -f "JERITH_HOME=$E2E_DIR" 2>/dev/null || true
  for pid in "${PIDS[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

wait_url() { # url, name, timeout_s
  local url="$1" name="$2" waited=0
  until curl -sf "$url" >/dev/null 2>&1; do
    if [ "$waited" -ge "$3" ]; then
      echo "timeout waiting for $name ($url)" >&2; exit 1
    fi
    sleep 0.5; waited=$((waited + 1))
  done
  echo "   ready: $name"
}

echo "==> 1/6 build"
( cd "$ROOT" && cargo build --quiet )

echo "==> 2/6 fresh scratch + stub runtime + local bootstrap"
rm -rf "$E2E_DIR"
mkdir -p "$E2E_DIR/a" "$E2E_DIR/b"

# Stub runtime: deterministic OpenAI-shaped answers (RuntimeExpert hashes
# "{expert_id}|{content}" — identical replies keep the E2E stable).
cat > "$E2E_DIR/runtime.py" <<'PY'
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

CONTENT = "jerith-stub deterministic completion"
MODEL_ID = "glm-5.3-flash-stub"

class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"data": [{"id": MODEL_ID}]} if self.path.startswith("/v1/models") else {"error": "nf"})

    def do_POST(self):
        _ = self.rfile.read(int(self.headers.get("content-length", 0)))
        self._send(200, {"choices": [{"message": {"role": "assistant", "content": CONTENT}}]})

    def log_message(self, fmt, *args):
        return

HTTPServer(("127.0.0.1", 8090), Handler).serve_forever()
PY

python3 "$E2E_DIR/runtime.py" & PIDS+=($!)
"$BIN" bootstrap --port "$BOOT_PORT" & PIDS+=($!)
wait_url "http://127.0.0.1:$RUNTIME_PORT/v1/models" "stub runtime" 20
wait_url "http://127.0.0.1:$BOOT_PORT/health" "bootstrap" 20

echo "==> 3/6 setup nodes A (gw 8900) and B (gw 8910, rpc 8911)"
setup_node() { # home gateway_port extra-config...
  local home="$1"
  JERITH_HOME="$home" "$BIN" setup --model "$MODEL" --memory sqlite \
    --contribution 25 --role hybrid --crypto jerithchain \
    --bootstrap "http://127.0.0.1:$BOOT_PORT" --no-wait >/dev/null
}
setup_node "$E2E_DIR/a"
setup_node "$E2E_DIR/b"

JERITH_HOME="$E2E_DIR/a" "$BIN" config \
  "runtime_url=http://127.0.0.1:$RUNTIME_PORT" >/dev/null
JERITH_HOME="$E2E_DIR/b" "$BIN" config \
  "runtime_url=http://127.0.0.1:$RUNTIME_PORT" "gateway_port=8910" \
  "rpc_port=8911" >/dev/null

# Sweep the Phase 4/5 verification dials to 100 % for the run.
sed -i 's/^verify_percent = .*/verify_percent = 100/; s/^ringer_percent = .*/ringer_percent = 100/' \
  "$E2E_DIR/a/config.toml" "$E2E_DIR/b/config.toml"

echo "==> 4/6 start nodes"
JERITH_HOME="$E2E_DIR/b" "$BIN" run & PIDS+=($!)
JERITH_HOME="$E2E_DIR/a" "$BIN" run & PIDS+=($!)

# Wait for both nodes to see the full two-node mesh.
waited=0
until [ "$(curl -sf http://127.0.0.1:8910/status | python3 -c 'import json,sys; print(json.load(sys.stdin)["peers_alive"])' 2>/dev/null || echo 0)" -ge 2 ] \
   && [ "$(curl -sf http://127.0.0.1:8900/status | python3 -c 'import json,sys; print(json.load(sys.stdin)["peers_alive"])' 2>/dev/null || echo 0)" -ge 2 ]; do
  if [ "$waited" -ge 30 ]; then echo "timeout waiting for mesh connectivity" >&2; exit 1; fi
  sleep 1; waited=$((waited + 1))
done
echo "   ready: two-node mesh (rtt pings signed, pins live)"

echo "==> 5/6 run the pinned cross-node E2E"
EXPECT_NODE_B_ID="$(curl -sf http://127.0.0.1:8910/status | python3 -c 'import json,sys; print(json.load(sys.stdin)["node_id"])')"
echo "   EXPECT_NODE_B_ID=$EXPECT_NODE_B_ID"
( cd "$ROOT" && EXPECT_NODE_B_ID="$EXPECT_NODE_B_ID" \
    cargo test -p jerith-mesh-core --test cross_node_dispatch -- --ignored | tail -3 )

echo "==> 6/6 live verification counters (from the loop's ping/ringer ticks)"
for port in 8900 8910; do
  echo "--- node :$port"
  curl -sf "http://127.0.0.1:$port/status" | python3 -c '
import json, sys
d = json.load(sys.stdin)
v = d["verify"]
print("  node_id", d["node_id"], "| peers_alive", d["peers_alive"])
print("  commitments", v["commitment_checks"], "failures", v["commitment_failures"])
print("  ringer/replay", v["replays_dispatched"], "matched", v["replay_matches"], "mismatched", v["replay_mismatches"])
print("  fraud reports flushed", v["fraud_reports_flushed"])
print("  guard screens", d["guard"]["screens"], "| blocked", d["guard"]["requests_blocked"])
print("  stake required", d["stake"]["required_uj"], "| slashed", d["stake"]["slashed_total_uj"])
'
done

echo
echo "✅ dev E2E complete."
