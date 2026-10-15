#!/usr/bin/env bash
# setup-lan-bootstrap.sh — turn THIS machine into the fleet's second
# (LAN) bootstrap server, per the Phase 1 discovery redundancy wire.
#
# Run ON the second node (the Jetson: noodlypanda@192.168.1.150). It:
#   1. finds the jerith binary,
#   2. installs a systemd USER unit serving `jerith bootstrap` on 8901,
#   3. appends the [discovery] rotation to ~/.jerith-mesh/config.toml
#      (idempotent; a timestamped .bak is kept),
#   4. verifies the endpoint answers locally and prints the LAN URL to
#      paste into the OTHER nodes' [discovery] servers.
#
# Why 8901 speaks plain http on the LAN: by Phase 1 design (see
# discovery.rs + RUNBOOK) plain-http bootstraps are config-only — peers
# never AUTO-learn them from ads, so a LAN capture proxy cannot
# proliferate. The primary (p2p.jerithai.com) stays https.
set -euo pipefail

PORT=8901
CONF_DIR="$HOME/.jerith-mesh"
CONF="$CONF_DIR/config.toml"

# 1. Locate the mesh node binary. The repo build path comes FIRST on
# purpose: `jerith` on PATH may be the unrelated JerithChain WALLET cli
# (balance/mine/transfer…), which would silently fail later.
BIN=""
for c in "$HOME/Documents/Projects/jerith-mesh/target/release/jerith" jerith; do
  if [ -x "$HOME/Documents/Projects/jerith-mesh/target/release/jerith" ] || command -v "$c" >/dev/null 2>&1; then
    candidate="$(command -v "$c" 2>/dev/null || echo "$HOME/Documents/Projects/jerith-mesh/target/release/jerith")"
    if "$candidate" bootstrap --port 1 --help >/dev/null 2>&1 || "$candidate" bootstrap --help >/dev/null 2>&1; then
      BIN="$candidate"
      break
    fi
  fi
done
if [ -z "$BIN" ]; then
  echo "error: mesh node binary with a `bootstrap` subcommand not found." >&2
  echo "  (a PATH `jerith` may be the JerithChain wallet cli — not this tool)" >&2
  echo "  cargo build --release   (repo root), then re-run this script" >&2
  exit 1
fi
echo "binary: $BIN"

# 2. systemd user unit for the bootstrap server.
UNIT_DIR="$HOME/.config/systemd/user"
UNIT="$UNIT_DIR/jerith-bootstrap.service"
mkdir -p "$UNIT_DIR"
if [ -f "$UNIT" ]; then
  echo "unit exists: $UNIT (leaving untouched)"
else
  cat > "$UNIT" <<UNIT
[Unit]
Description=jerith-mesh LAN bootstrap node (second trust anchor, :$PORT)
After=network-online.target

[Service]
Type=simple
ExecStart=$BIN bootstrap --port $PORT
Restart=always
RestartSec=5
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=default.target
UNIT
  systemctl --user daemon-reload
  systemctl --user enable --now jerith-bootstrap
  echo "installed + started: jerith-bootstrap"
fi

# 3. Append the [discovery] rotation (idempotent).
if [ -f "$CONF" ] && grep -q '^\[discovery\]' "$CONF"; then
  echo "config already has [discovery] — leaving untouched"
else
  mkdir -p "$CONF_DIR"
  [ -f "$CONF" ] && cp "$CONF" "$CONF/config.toml.bak-lan-bootstrap"
  printf '\n# Discovery failover (Phase 1): primary first, then the LAN\n# second bootstrap (this machine, plain http — config-only).\n[discovery]\nservers = ["https://p2p.jerithai.com", "http://%s:%d"]\n' "$(hostname -I | awk '{print $1}')" "$PORT" >> "$CONF"
  echo "appended [discovery] to $CONF (backup: $CONF.bak-lan-bootstrap)"
fi

# 4. Local verify + tell-tale output.
sleep 1
if curl -fsS -m 5 "http://127.0.0.1:$PORT/v1/models/counts" >/dev/null; then
  echo "local verify OK: http://127.0.0.1:$PORT/v1/models/counts answers"
else
  echo "NOT YET: local endpoint refused — journal:" >&2
  journalctl --user -u jerith-bootstrap -n 20 --no-pager || true
  exit 1
fi

ip="$(hostname -I | awk '{print $1}')"
echo
echo "✔ second bootstrap serving. On the OTHER nodes append/verify:"
echo "  [discovery]"
echo "  servers = [\"https://p2p.jerithai.com\", \"http://$ip:$PORT\"]"
echo "Then confirm with:  curl -s http://127.0.0.1:8900/status | jq .discovery"
