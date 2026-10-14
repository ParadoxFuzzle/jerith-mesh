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
#   5. proves Phase 2 replication live: node A pulls node B's expert as a
#      replica (chunked transfer, checksum-validated publish) and both nodes
#      report the same cold-shard owners from /status,
#   6. prints each node's live verification counters from /status,
#   7. kills every process it started (trap EXIT).
#
# Env overrides:
#   E2E_MODEL      curated model id (default glm-5.3-flash)
#   E2E_DIR        scratch dir (default /tmp/jerith-e2e — wiped each run)
#   E2E_KEEP=1     leave the stack running after the test (for poking /status)
#   E2E_*_PORT     ports, for a box already running something on the
#                  defaults (a live node holds 8900/9000, so the defaults
#                  are what CI uses and a dev box overrides)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/target/debug/jerith"
E2E_DIR="${E2E_DIR:-/tmp/jerith-e2e}"
MODEL="${E2E_MODEL:-glm-5.3-flash}"
KEEP="${E2E_KEEP:-0}"
BOOT_PORT="${E2E_BOOT_PORT:-8999}"
RUNTIME_PORT="${E2E_RUNTIME_PORT:-8090}"
A_GW="${E2E_A_GATEWAY_PORT:-8900}"
A_RPC="${E2E_A_RPC_PORT:-9000}"
B_GW="${E2E_B_GATEWAY_PORT:-8910}"
B_RPC="${E2E_B_RPC_PORT:-8911}"
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
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

CONTENT = "jerith-stub deterministic completion"
MODEL_ID = "glm-5.3-flash-stub"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8090

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

HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
PY

python3 "$E2E_DIR/runtime.py" "$RUNTIME_PORT" & PIDS+=($!)
"$BIN" bootstrap --port "$BOOT_PORT" & PIDS+=($!)
wait_url "http://127.0.0.1:$RUNTIME_PORT/v1/models" "stub runtime" 20
wait_url "http://127.0.0.1:$BOOT_PORT/health" "bootstrap" 20

echo "==> 3/6 setup nodes A (gw $A_GW, rpc $A_RPC) and B (gw $B_GW, rpc $B_RPC)"
setup_node() { # home gateway_port extra-config...
  local home="$1"
  JERITH_HOME="$home" "$BIN" setup --model "$MODEL" --memory sqlite \
    --contribution 25 --role hybrid --crypto jerithchain \
    --bootstrap "http://127.0.0.1:$BOOT_PORT" --no-wait >/dev/null
}
setup_node "$E2E_DIR/a"
setup_node "$E2E_DIR/b"

JERITH_HOME="$E2E_DIR/a" "$BIN" config \
  "runtime_url=http://127.0.0.1:$RUNTIME_PORT" "gateway_port=$A_GW" \
  "rpc_port=$A_RPC" >/dev/null
JERITH_HOME="$E2E_DIR/b" "$BIN" config \
  "runtime_url=http://127.0.0.1:$RUNTIME_PORT" "gateway_port=$B_GW" \
  "rpc_port=$B_RPC" >/dev/null

# Sweep the Phase 4/5 verification dials to 100 % for the run.
sed -i 's/^verify_percent = .*/verify_percent = 100/; s/^ringer_percent = .*/ringer_percent = 100/' \
  "$E2E_DIR/a/config.toml" "$E2E_DIR/b/config.toml"

# Both E2E nodes share one box, so they advertise loopback: each node's
# signed health ping must reach the address it advertised. (The gateway
# binds loopback while the advertised host defaults to the routable
# interface address, so on a single box every ping misses — the peers then
# deprecate each other three ticks in and no mesh work happens at all.)
sed -i '1i advertise_host = "127.0.0.1"' "$E2E_DIR/a/config.toml" "$E2E_DIR/b/config.toml"

# Phase 2 replication leg:
#   - B holds (and therefore advertises) one expert, L3.E17;
#   - both nodes boot with the same persisted demand record — 8 distinct
#     experts seen, L3.E17 dominant — which is exactly what a node that had
#     served/dispatched them would have written;
#   - the replication round is 30 s, so the pull is watchable in a test run.
JERITH_HOME="$E2E_DIR/b" "$BIN" experts make L3.E17 --hidden 8 --intermediate 6 >/dev/null
for home in "$E2E_DIR/a" "$E2E_DIR/b"; do
  mkdir -p "$home/data"
  cat > "$home/data/expert-stats.json" <<'JSON'
{ "total": 107, "counts": { "L3.E17": 100, "L3.E40": 1, "L3.E41": 1, "L3.E42": 1, "L3.E43": 1, "L3.E44": 1, "L3.E45": 1, "L3.E46": 1 } }
JSON
  if grep -q '^\[experts\]' "$home/config.toml"; then
    sed -i 's/^replication_interval_secs = .*/replication_interval_secs = 30/' "$home/config.toml"
  else
    printf '\n[experts]\nreplication_interval_secs = 30\n' >> "$home/config.toml"
  fi
done

echo "==> 4/6 start nodes"
JERITH_HOME="$E2E_DIR/b" "$BIN" run & PIDS+=($!)
JERITH_HOME="$E2E_DIR/a" "$BIN" run & PIDS+=($!)

# Wait for both nodes to see the full two-node mesh.
waited=0
until [ "$(curl -sf http://127.0.0.1:$B_GW/status | python3 -c 'import json,sys; print(json.load(sys.stdin)["peers_alive"])' 2>/dev/null || echo 0)" -ge 2 ] \
   && [ "$(curl -sf http://127.0.0.1:$A_GW/status | python3 -c 'import json,sys; print(json.load(sys.stdin)["peers_alive"])' 2>/dev/null || echo 0)" -ge 2 ]; do
  if [ "$waited" -ge 30 ]; then echo "timeout waiting for mesh connectivity" >&2; exit 1; fi
  sleep 1; waited=$((waited + 1))
done
echo "   ready: two-node mesh (rtt pings signed, pins live)"

echo "==> 5/6 run the pinned cross-node E2E"
EXPECT_NODE_B_ID="$(curl -sf http://127.0.0.1:$B_GW/status | python3 -c 'import json,sys; print(json.load(sys.stdin)["node_id"])')"
echo "   EXPECT_NODE_B_ID=$EXPECT_NODE_B_ID (rpc 127.0.0.1:$B_RPC, expert L3.E17)"
# The test refuses to call a typed refusal a pass: it needs B's hosted
# expert, which this run published above (L3.E17).
( cd "$ROOT" && EXPECT_NODE_B_ID="$EXPECT_NODE_B_ID" \
    EXPECT_RPC_ADDR="127.0.0.1:$B_RPC" EXPECT_EXPERT_ID="L3.E17" \
    cargo test -p jerith-mesh-core --test cross_node_dispatch -- --ignored | tail -3 )

echo "==> 5b/6 hot-expert replication + cold-shard agreement (live)"
# B advertises the expert it holds.
curl -sf http://127.0.0.1:$B_GW/status | python3 -c '
import json, sys
hosted = json.load(sys.stdin)["experts"]["hosted_ids"]
assert "L3.E17" in hosted, hosted
print("   B advertises:", hosted)
'

# A has the demand but holds nothing: its replication loop must pull the
# one expert it can — the hot one, from the node that advertises it.
waited=0
until [ "$(curl -sf http://127.0.0.1:$A_GW/status \
     | python3 -c 'import json,sys; e=json.load(sys.stdin).get("experts") or {}; print(1 if "L3.E17" in e.get("hosted_ids", []) else 0)' 2>/dev/null || echo 0)" = "1" ]; do
  if [ "$waited" -ge 120 ]; then
    echo "timeout: node A never replicated L3.E17" >&2
    curl -sf http://127.0.0.1:$A_GW/status >&2 || true
    exit 1
  fi
  sleep 2; waited=$((waited + 2))
done
echo "   A replicated L3.E17 from B in ${waited}s (chunked + checksum-validated)"

# The replica is the real weights: A's own forward pass matches B's.
IN="0.5,-0.25,1.0,0.0,-1.5,2.0,0.75,-0.5"
a_out="$(JERITH_HOME="$E2E_DIR/a" "$BIN" experts run L3.E17 --input "$IN" | grep '→')"
b_out="$(JERITH_HOME="$E2E_DIR/b" "$BIN" experts run L3.E17 --input "$IN" | grep '→')"
if [ -z "$a_out" ] || [ "$a_out" != "$b_out" ]; then
  echo "replica forward mismatch: A='$a_out' B='$b_out'" >&2
  exit 1
fi
echo "   A's replica computes identically:$a_out"

# Demand split + shard agreement straight from /status: identical demand ⇒
# identical hot set ⇒ identical cold-shard owner per expert, each node
# deriving it from its own ads (no coordinator).
fetch_experts_view() { # port
  curl -sf "http://127.0.0.1:$1/status" | python3 -c '
import json, sys
d = json.load(sys.stdin)
e = d["experts"]
print(json.dumps({
    "node_id": d["node_id"],
    "hot": e["hot"],
    "cold": e["cold"],
    "ring_nodes": e["shard_ring_nodes"],
    "owners": {p["expert_id"]: p["cold_shard_owner"] for p in e["cold_placements"]},
    "policy": e["replication"],
}))
'
}
python3 - "$(fetch_experts_view "$A_GW")" "$(fetch_experts_view "$B_GW")" <<'PY'
import json, sys
a, b = (json.loads(x) for x in sys.argv[1:3])
print("   A: hot", a["hot"], "| cold", a["cold"], "| ring nodes", a["ring_nodes"])
print("   A: cold shard owners", a["owners"])
print("   A: replication policy", a["policy"])
assert a["node_id"] != b["node_id"]
assert a["hot"] == ["L3.E17", "L3.E40"], a["hot"]
assert a["cold"] == 6 and b["cold"] == 6, (a["cold"], b["cold"])
assert a["owners"] == b["owners"], (a["owners"], b["owners"])
assert len(a["owners"]) == 6, a["owners"]
assert set(a["owners"].values()) <= {a["node_id"], b["node_id"]}, a["owners"]
assert a["ring_nodes"] == 2 == b["ring_nodes"], (a["ring_nodes"], b["ring_nodes"])
PY
echo "   both nodes derive the same cold-shard owners independently"

echo "==> 6/6 live verification counters (from the loop's ping/ringer ticks)"
for port in "$A_GW" "$B_GW"; do
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
