#!/bin/bash
# migrate-from-openclaw.sh — one-shot: move from the OpenClaw stack to a
# running jerith-mesh node + bot on this machine, reusing what's already
# there (model, memory DB backbone, Discord bot token) instead of
# duplicating config. Non-destructive: OpenClaw services are stopped and
# NOT disabled, and nothing is deleted.
set -euo pipefail

CLIENT_ROOT="${MIGRATE_OPENCLAW_ROOT:-$HOME/.openclaw}"
MESH_ROOT="${MIGRATE_MESH_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
BOT_TOML="${MIGRATE_BOT_TOML:-$HOME/.jerith-mesh/bot.toml}"
NODE_CONFIG="${MIGRATE_NODE_CONFIG:-$HOME/.jerith-mesh/config.toml}"
# MIGRATE_SKIP_SYSTEMD=1 → dry validation: no real unit installs/stops; commands are printed.
SKIP_SYSTEMD="${MIGRATE_SKIP_SYSTEMD:-0}"
sysctl() {
  if [ "$SKIP_SYSTEMD" = "1" ]; then
    echo "   [dry-run] systemctl --user $*"
  else
    systemctl --user "$@"
  fi
}

echo "==> 1/5 stopping OpenClaw services (reversible — restart list at the end)"
STOPPED=""
for svc in openclaw-gateway openclaw-model-router litellm-proxy openclaw-sidecar \
           openclaw-watchdog.service openclaw-discord-listener.service \
           openclaw-watchdog.timer openclaw-discord-listener.timer \
           openclaw-backup.timer openclaw-reflect.timer openclaw-pg-memory-sync.timer; do
  if [ "$SKIP_SYSTEMD" = "1" ]; then
    STOPPED="$STOPPED $svc"
  elif systemctl --user is-active --quiet "$svc" 2>/dev/null; then
    systemctl --user stop "$svc" && STOPPED="$STOPPED $svc"
  fi
done
echo "   stopped:$STOPPED"

echo "==> 2/5 building jerith-mesh (release)"
cd "$MESH_ROOT"
# Keep the toolchain resolvable even when HOME was redirected (validation sandboxes).
export RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
if ! command -v rustc >/dev/null 2>&1; then
  export RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
fi
cargo build --release --workspace

echo "==> 3/5 node setup: reuse the model you ran in OpenClaw (ollama)"
# Default pick: first ollama model that looks qwen/nemotron/glm; override with JERITH_MODEL=ollama:<name>
MODEL="${JERITH_MODEL:-}"
if [ -z "$MODEL" ]; then
  for cand in qwen3:8b nemotron-3-nano:4b qwen3:4b; do
    if ollama list 2>/dev/null | grep -q "$cand"; then MODEL="ollama:$cand"; break; fi
  done
fi
MODEL="${MODEL:-ollama:qwen3:8b}"
echo "   model: $MODEL"

if [ ! -f "$NODE_CONFIG" ]; then
  ./target/release/jerith setup --model "$MODEL" --memory sqlite \
    --contribution 40 --role hybrid --crypto jerithchain --no-wait </dev/null
fi
# Runtime = local ollama in both stacks
sed -i 's|^runtime_url = .*|runtime_url = "http://127.0.0.1:11434"|' "$NODE_CONFIG"

echo "==> 4/5 services: node gateway + bot (user-level systemd)"
SYSTEMD_DIR="$HOME/.config/systemd/user"
MESH_DIR="$(dirname "$BOT_TOML")"
mkdir -p "$SYSTEMD_DIR" "$MESH_DIR"
cp deploy/systemd/jerith-node.service "$SYSTEMD_DIR/"
# Bot unit only if a bot token is present (env JERITH_DISCORD_TOKEN or an existing file)
if [ -n "${JERITH_DISCORD_TOKEN:-}" ]; then
  printf 'discord_token = "%s"\n' "$JERITH_DISCORD_TOKEN" > "$BOT_TOML"
  chmod 600 "$BOT_TOML"
  cp deploy/systemd/jerith-bot.service "$SYSTEMD_DIR/"
  BOT=1
elif python3 - "$CLIENT_ROOT" <<'PY' 2>/dev/null
import json, os, sys
cfg = json.load(open(os.path.join(sys.argv[1], "openclaw.json")))
p = cfg["secrets"]["providers"]["discord-token"]["path"]
sys.exit(0 if os.path.exists(p) else 1)
PY
then
  # Borrow the OpenClaw bot token (same bot, same Discord side) — remove for a clean production token later.
  python3 - "$BOT_TOML" "$CLIENT_ROOT" <<'PY'
import json, os, sys
out, client_root = sys.argv[1], sys.argv[2]
cfg = json.load(open(os.path.join(client_root, "openclaw.json")))
tok = open(cfg["secrets"]["providers"]["discord-token"]["path"]).read().strip()
with open(out, "w") as f:
    f.write(f'discord_token = "{tok}"\n')
os.chmod(out, 0o600)
print("   bot token borrowed from OpenClaw secrets (remove for real production)")
PY
  cp deploy/systemd/jerith-bot.service "$SYSTEMD_DIR/"
  BOT=1
else
  echo "   (no bot token found — bot skipped; set JERITH_DISCORD_TOKEN to enable)"
  BOT=0
fi
sysctl daemon-reload
sysctl enable --now jerith-node
[ "$BOT" = "1" ] && sysctl enable --now jerith-bot || true

echo "==> 5/5 verify"
sleep 6
sysctl is-active jerith-node || true
[ "$BOT" = "1" ] && (sysctl is-active jerith-bot || true)
curl -fsS -m5 http://127.0.0.1:8900/status >/dev/null && echo "   gateway :8900 OK" || echo "   gateway not ready yet"
echo
echo "Done. To undo entirely:"
echo "  systemctl --user disable --now jerith-bot jerith-node"
[ -n "$STOPPED" ] && echo "  systemctl --user start$STOPPED  # restore OpenClaw"
