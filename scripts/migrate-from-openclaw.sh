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
for svc in openclaw-gateway openclaw-model-router openclaw-sidecar \
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
  # Fresh setup default: runtime = local Ollama (explicit, since the binary's
  # NodeConfig::default hardcodes :8000 — never rely on ambient defaults).
  if [ -z "${JERITH_RUNTIME_URL:-}" ]; then
    sed -i 's|^runtime_url = .*|runtime_url = "http://127.0.0.1:11434"|' "$NODE_CONFIG"
  fi
fi
# Runtime choice: JERITH_RUNTIME_URL wins; a brand-new config defaults to
# local Ollama (11434); an EXISTING config's runtime_url is left untouched
# so a deliberate coupling (e.g. the authed LiteLLM proxy at :8000) survives
# re-runs instead of being silently reverted here.
if [ -n "${JERITH_RUNTIME_URL:-}" ]; then
  sed -i "s|^runtime_url = .*|runtime_url = \"${JERITH_RUNTIME_URL}\"|" "$NODE_CONFIG"
  echo "   runtime: $JERITH_RUNTIME_URL (JERITH_RUNTIME_URL)"
fi
# LiteLLM-proxy lifecycle follows the runtime choice: when the proxy IS the
# runtime (coupling at :8000) it must stay up; otherwise it is exactly like
# the rest of the OpenClaw stack — stopped reversibly.
if grep -q 'runtime_url = "http://127.0.0.1:8000"' "$NODE_CONFIG"; then
  echo "   runtime: LiteLLM proxy coupled — litellm-proxy stays running"
else
  if systemctl --user is-active --quiet litellm-proxy; then
    systemctl --user stop litellm-proxy
    STOPPED="$STOPPED litellm-proxy"
    echo "   runtime: local Ollama — stopped unused litellm-proxy (reversible)"
  fi
fi

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

echo "==> 4b/5 assets: scripts, plugin-skills, extensions, workspace docs/config"
# A jerith-mesh install can't run OpenClaw's plugin system directly; instead we
# stage everything script/project-shaped into a browsable migrated tree preserved
# full-fidelity, plus machine-readable manifests that list what each service
# used to do (so nothing silently disappears). Modify nothing in ~/.openclaw.
ASSETS_ROOT="$HOME/.jerith-mesh/openclaw-migrated"
MW="$ASSETS_ROOT/workspace"     # workspace scripts/docs/config
MS="$ASSETS_ROOT/plugin-skills"
ME="$ASSETS_ROOT/extensions"
mkdir -p "$MW" "$MS" "$ME"
if [ -d "$CLIENT_ROOT/workspace/scripts" ]; then
  rsync -a "$CLIENT_ROOT/workspace/scripts/" "$MW/scripts/"
  echo "   scripts: $(find "$MW/scripts" -maxdepth 1 -type f | wc -l) files → $MW/scripts"
fi
# Top-level workspace markdown: every persona/agent doc present (SOUL,
# IDENTITY, USER, USER_MODEL, AGENTS, DREAMS, and anything else in *.md)
# — a fixed-name list silently dropped docs like USER.md whenever the
# agent stack renamed or grew them.
for doc in "$CLIENT_ROOT"/workspace/*.md; do
  [ -f "$doc" ] || continue
  cp "$doc" "$MW/"
  echo "   doc: $(basename "$doc") copied"
done
# Persona-doc variants kept per directory-scope (osdev/, main/, skills/*/…):
# same canon names, one copy per scope, relative path preserved.
for doc in $(find "$CLIENT_ROOT/workspace" -mindepth 2 \( -iname 'SOUL.md' \
  -o -iname 'IDENTITY.md' -o -iname 'USER.md' -o -iname 'USER_MODEL.md' \
  -o -iname 'AGENTS.md' -o -iname 'MEMORY.md' -o -iname 'DREAMS.md' \
  -o -iname 'HEARTBEAT.md' -o -iname 'TOOLS.md' -o -iname 'grounding-protocol.md' \) 2>/dev/null); do
  rel="${doc#$CLIENT_ROOT/workspace/}"
  mkdir -p "$MW/$(dirname "$rel")"
  cp "$doc" "$MW/$rel"
  echo "   doc: $rel (scope variant) copied"
done
if [ -d "$CLIENT_ROOT/workspace/config" ]; then
  rsync -a "$CLIENT_ROOT/workspace/config/" "$MW/config/"
  echo "   workspace config: $(find "$MW/config" -type f | wc -l) files"
fi
if [ -d "$CLIENT_ROOT/plugin-skills" ]; then
  # Copy skill code but skip logs/caches — anything regenerable stays behind.
  for skill in "$CLIENT_ROOT"/plugin-skills/*; do
    name="$(basename "$skill")"
    [ -d "$skill" ] || continue
    rsync -a --exclude '*.log' --exclude '__pycache__' --exclude '.venv' "$skill/" "$MS/$name/"
  done
  echo "   plugin-skills: $(ls "$MS" | wc -l) skills → $MS"
fi
if [ -d "$CLIENT_ROOT/extensions" ]; then
  for ext in "$CLIENT_ROOT"/extensions/*; do
    name="$(basename "$ext")"
    [ -d "$ext" ] || continue
    rsync -a --exclude node_modules --exclude dist --exclude target "$ext/" "$ME/$name/"
    echo "   extension: $name (src copied; node_modules/dist excluded — rebuild later)"
  done
fi
# Cron/periodic jobs → manifest (not auto-installed: they ran agent logic; wired
# back up deliberately, per job).
python3 - "$ASSETS_ROOT" "$CLIENT_ROOT" <<'PY'
import json, os, sys, datetime
assets, client_root = sys.argv[1], sys.argv[2]
rows = []
for cand in ("cron/jobs.json", "cron/jobs.json.migrated"):
    p = os.path.join(client_root, cand)
    if os.path.exists(p):
        try:
            data = json.load(open(p))
        except Exception:
            continue
        for j in data.get("jobs", []):
            sched = j.get("schedule") or {}
            kind = sched.get("kind")
            if kind == "every":
                ms = sched.get("everyMs", 0)
                every_s = ms // 1000
                desc = f"every {every_s // 3600}h{(every_s % 3600) // 60}m" if every_s >= 60 else f"every {every_s}s"
            elif kind == "at":
                desc = f"one-shot at {sched.get('at')}"
            else:
                desc = f"unknown: {sched}"
            rows.append({"name": j.get("name", "?"), "id": j.get("id", "?"), "schedule": desc})
out = os.path.join(assets, "cron-jobs-manifest.json")
json.dump({"generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "note": "OpenClaw cron jobs. Re-create deliberately (systemd timers or /mine-style bot loops) — not auto-installed.",
           "jobs": rows}, open(out, "w"), indent=2)
print(f"   cron: {len(rows)} jobs → cron-jobs-manifest.json")
PY
# Workspace-driven systemd services → manifest of what ran outside the gateway
{
  echo "# OpenClaw workspace services found active/in unit files on this machine."
  echo "# Each needs a jerith replacement or explicit retirement decision."
  echo "# unit | purpose | script"
  for f in "$HOME/.config/systemd/user"/openclaw-*.service; do
    [ -f "$f" ] || continue
    name="$(basename "$f" .service)"
    desc="$(grep -m1 '^Description=' "$f" | cut -d= -f2)"
    execs="$(grep -E '^ExecStart=' "$f" | tail -1 | cut '-d=' -f2 | cut -d' ' -f2-)"
    echo "$name | $desc | $execs"
  done
} > "$ASSETS_ROOT/workspace-services-manifest.tsv" 2>/dev/null || true
echo "   workspace services → workspace-services-manifest.tsv"

# Reuse the OpenClaw nightly-dump DB list if you hit a name you want:
# community_memory jerith_people self_improving jerith mysqlclaw

echo "==> 4c/5 PostgreSQL: detect stack usage, dump all its databases"
# The OpenClaw memory stack runs PostgreSQL (scripts/shared/pg_pool.py:
# root@127.0.0.1:5432, auth via ~/.pgpass). Capture the usage + a fresh
# dump per database so nothing agent-memory-shaped is left behind.
PG_DIR="$ASSETS_ROOT/postgres"
mkdir -p "$PG_DIR"
PGD_HOST="${PGD_HOST:-127.0.0.1}"
PGD_PORT="${PGD_PORT:-5432}"
PGD_USER="${PGD_USER:-root}"
# DBs the migration must not fold into itself: system db + other services'
# data (nmonit_identity belongs to the nmonit monitoring service).
PGD_EXCLUDE="${PGD_EXCLUDE:-postgres nmonit_identity}"
if psql -h "$PGD_HOST" -p "$PGD_PORT" -U "$PGD_USER" -d postgres \
    -tAc "SELECT 1" >/dev/null 2>&1; then
  LIVE_DBS=$(psql -h "$PGD_HOST" -p "$PGD_PORT" -U "$PGD_USER" -d postgres \
      -tAc "SELECT datname FROM pg_database WHERE NOT datistemplate AND datname <> 'postgres'")
  dumped=""
  for db in $LIVE_DBS; do
    skip=0
    for ex in $PGD_EXCLUDE; do
      if [ "$db" = "$ex" ]; then skip=1; fi
    done
    if [ "$skip" = "1" ]; then
      echo "   pg: $db — excluded (separate service/system)"
      continue
    fi
    dump="$PG_DIR/$db.dump"
    if pg_dump -h "$PGD_HOST" -p "$PGD_PORT" -U "$PGD_USER" -Fc -f "$dump" "$db"; then
      dumped="$dumped $db"
      echo "   pg: dumped $db ($(du -h "$dump" | cut -f1))"
    else
      echo "   pg: FAILED dumping $db (dump left out — check ~/.pgpass auth)"
    fi
  done
  # Capture the connection details + auth: pgpass carries service:user
  # passwords and must be 0600 for libpq to accept it.
  if [ -f "$HOME/.pgpass" ]; then
    cp "$HOME/.pgpass" "$PG_DIR/pgpass"
    chmod 600 "$PG_DIR/pgpass"
    echo "   pg: ~/.pgpass copied (auth for pg_restore)"
  else
    echo "   pg: no ~/.pgpass found — restore auth must be provided manually"
  fi
  PGDB_INFO=$(python3 - "$PG_DIR" "$PGD_HOST" "$PGD_PORT" "$PGD_USER" $dumped <<'PY'
import json, os, sys
a = (list(sys.argv) + [""] * 5)[1:6]
pg_dir, host, port, user = a[:4]
dumped = " ".join(sys.argv[5:])
dbs = []
for db in dumped.split():
    p = os.path.join(pg_dir, f"{db}.dump")
    if os.path.exists(p):
        dbs.append({"name": db, "dump": os.path.basename(p), "size_bytes": os.path.getsize(p),
                    "restore": f"pg_restore -h {host} -p {port} -U {user} --clean --if-exists --dbname={db} {p}"})
json.dump({"source": {"host": host, "port": int(port), "user": user},
           "auth": "~/.pgpass equivalent staged as pgpass (chmod 600; place at ~/.pgpass and restore works)",
           "databases": dbs,
           "note": "nmonit_identity excluded (belongs to the separate nmonit service). "
                   "These DBs fed the OpenClaw memory stack (community_memory, self_improving, jerith_people…); "
                   "jerith-mesh itself uses sqlite — restore these only to keep/reharvest agent memory."},
          open(os.path.join(pg_dir, "databases-manifest.json"), "w"), indent=2)
print(f"pg manifest: {len(dbs)} databases → databases-manifest.json")
PY
)
  echo "   $PGDB_INFO"
else
  echo "   pg: no reachable PostgreSQL at $PGD_HOST:$PGD_PORT as $PGD_USER — skipped (not in use?)"
fi

echo "==> 4d/5 brave key: stage the OpenClaw web-search key for jerith web skills"
# The node's [websearch] resolves the Brave Search API key (in order):
# config api_key/path → $JERITH_BRAVE_API_KEY → ~/.jerith-mesh/brave-key →
# the OpenClaw keyring file (read-only borrow). Staging a local copy at
# 0600 makes the node self-sufficient once OpenClaw is retired.
BRAVE_SRC=~/".openclaw/keys/secrets/brave-key"
BRAVE_DST=~/".jerith-mesh/brave-key"
if [ -s "$BRAVE_SRC" ]; then
  if [ ! -s "$BRAVE_DST" ]; then
    cp "$BRAVE_SRC" "$BRAVE_DST" && chmod 600 "$BRAVE_DST"
    echo "   brave key: staged $BRAVE_SRC → $BRAVE_DST (web_search ready)"
  else
    echo "   brave key: $BRAVE_DST already exists — left untouched"
  fi
else
  echo "   brave key: no OpenClaw brave-key found — set JERITH_BRAVE_API_KEY or ~/.jerith-mesh/brave-key for web_search"
fi

echo "==> 4d2/5 chain token: stage the JerichChain api token for the mesh stack"
# The JerithChain node's api token is the shared auth for the mesh's miner,
# the bot's /mine, the gateway's chain probe and `jerith chain …`. The
# chain is co-hosted (jerithchain repo on this machine); stage its token
# into the mesh keys dir when the mesh side lacks one.
CHAIN_TOK_SRC="${JERITHCHAIN_ROOT:-$HOME/Documents/Projects/jerithchain/keys/api.token}"
CHAIN_TOK_DST="$HOME/.jerith-mesh/keys/api.token"
if [ -s "$CHAIN_TOK_DST" ]; then
  echo "   chain token: $CHAIN_TOK_DST already exists — left untouched"
elif [ -s "$CHAIN_TOK_SRC" ]; then
  mkdir -p "$HOME/.jerith-mesh/keys" && chmod 700 "$HOME/.jerith-mesh/keys"
  cp "$CHAIN_TOK_SRC" "$CHAIN_TOK_DST" && chmod 600 "$CHAIN_TOK_DST"
  echo "   chain token: staged $CHAIN_TOK_SRC → $CHAIN_TOK_DST (600) — miner/bot/CLI share the chain's api auth"
else
  echo "   chain token: no chain api.token at $CHAIN_TOK_SRC — set JERITH_TOKEN or stage the file manually"
fi

echo "==> 4e/5 owner gate: stage the bot owner id for system-level command ACL"
# The owner's Discord id gates anything beyond user-level commands
# (owner.rs default-deny classification). Borrowed read-only from
# OpenClaw's commands.ownerAllowFrom when it isn't already in bot.toml.
python3 - "$BOT_TOML" "$CLIENT_ROOT" <<'PY' 2>/dev/null || true
import json, os, re, sys
out, client_root = sys.argv[1], sys.argv[2]
cfg = json.load(open(os.path.join(client_root, "openclaw.json")))
entries = cfg.get("commands", {}).get("ownerAllowFrom", []) or []
owner = None
for raw in entries or []:
    raw = str(raw)
    id_s = raw.split("discord:", 1)[1] if "discord:" in raw else raw
    if id_s.strip().isdigit():
        owner = id_s.strip()
        break
if owner:
    try:
        text = open(out).read()
    except FileNotFoundError:
        text = ""
    if not re.search(r"^\[owner\]", text, re.M) and "owner_discord_id" not in text and "discord_id" not in text:
        with open(out, "a") as f:
            if text and not text.endswith("\n"):
                f.write("\n")
            f.write(f'[owner]\ndiscord_id = {owner}\n')
        os.chmod(out, 0o600)
        print(f"   owner gate: [owner] discord_id = {owner} staged into {out} (default-deny open to the owner)")
    else:
        print("   owner gate: bot.toml already carries an owner id — left untouched")
else:
    print("   owner gate: no OpenClaw ownerAllowFrom id found — system-level commands stay default-deny (set [owner] discord_id)")
PY

echo "==> 5/5 verify"
sleep 6
sysctl is-active jerith-node || true
[ "$BOT" = "1" ] && (sysctl is-active jerith-bot || true)
curl -fsS -m5 http://127.0.0.1:8900/status >/dev/null && echo "   gateway :8900 OK" || echo "   gateway not ready yet"
echo
echo "Done. To undo entirely:"
echo "  systemctl --user disable --now jerith-bot jerith-node"
if [ -n "$STOPPED" ]; then
  echo "  systemctl --user start$STOPPED  # restore OpenClaw"
fi
