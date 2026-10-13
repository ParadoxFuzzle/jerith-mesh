# Post-deploy runbook — bootstrap & mesh nodes

Short, command-driven guide for operators: verify a node right after a
deploy, then how to roll back cleanly. Companion docs:
`DEPLOY-BOOTSTRAP.md` (tunnel setup), `../FEATURES.md` (what is live).

## Services involved

| Service | Unit | What it is |
|---|---|---|
| Bootstrap | `jerith-bootstrap.service` (systemd --user) | `target/release/jerith bootstrap --port 8901` |
| Tunnel | `cloudflared-jerith.service` (systemd --user) | p2p.jerithai.com → localhost:8901 |
| Chain node | `jerith-node-dev.service` (transient) | launcher `jerithchain/deploy/start_jerith_node.sh` |
| Chain snapshot | `jerith-snapshot.timer` (systemd --user) | publishes a chain snapshot every 5 min |
| Node | `deploy/systemd/jerith-node.service` (template) | `jerith run` (gateway :8900 + QUIC rpc) |

## JerithChain node (co-hosted, integrated into the mesh install)

The chain node is part of the mesh stack, not a sidecar — the mesh shares
its auth and introspects it:

- **RPC**: FastAPI on `127.0.0.1:8322` (token-gated; the `~/.jerith-mesh/config.toml`
  `chain_url` field / `$JERITH_CHAIN_URL` override points here).
- **Shared auth**: `~/.jerith-mesh/keys/api.token` must carry the same value
  as the chain's `~/Documents/Projects/jerithchain/keys/api.token` (the
  migration script stages it; `economics::read_token` — $JERITH_TOKEN env
  wins — is the single resolver for the miner, the bot's `/mine`, the
  gateway's chain probe and the CLI).
- **Ledger**: SQLite at `<jerithchain repo>/data/jerith.db` (config
  `chain_data_dir` / `$JERITH_CHAIN_DATA` override). Reads are read-only.
- **CLI** (read-only):
  ```
  jerith chain status              # height, supply, wallet counts, mining stats
  jerith chain wallets             # balances table (+ discord ids + mined totals)
  jerith chain balance <id|addr>   # one wallet (Discord id or address)
  jerith chain miners --limit 25   # /mystate leaderboard
  jerith chain block <height> / tx <txid>
  ```
- **Gateway passthrough**: `curl :8900/status` carries a `chain` section
  (height/tip/circulating/wallet counts; `reachable:false` with the error
  when the chain node is down — never a failed status page).

## Post-deploy verification (in order)

```bash
# 1. Build is clean, binaries fresh.
cargo build --release --workspace

# 2. Units came back up after `systemctl --user restart ...`:
systemctl --user is-active jerith-bootstrap jerith-node-dev cloudflared-jerith
#    → three "active" lines; anything else = stop and read logs:
journalctl --user -u jerith-bootstrap -n 50

# 3. Health: local + public through the tunnel.
curl -s http://127.0.0.1:8901/health        # {"ok":true,"service":"jerith-bootstrap"}
curl -s https://p2p.jerithai.com/health      # same body through the tunnel

# 4. Chain node contract surface (should be 401 token-gated, never 404):
curl -s -o /dev/null -w "%{http_code}\n" "http://127.0.0.1:8322/stake?discord_id=probe"
#    → 401 {"detail":"invalid or missing API token"}

# 5. Advertise probe: a scratch node must appear in the public peer list
#    within one ping cycle (~10 s), then fall out of the list via the
#    90 s advertisement TTL after it is stopped.
export JERITH_HOME=/tmp/probe-$$
./target/release/jerith setup --model glm-5.3-flash --memory sqlite \
  --contribution 25 --role contributor --crypto none \
  --bootstrap https://p2p.jerithai.com --no-wait >/dev/null
./target/release/jerith config gateway_port=8940 >/dev/null
nohup ./target/release/jerith run >/tmp/probe.log 2>&1 &
sleep 15 && curl -s https://p2p.jerithai.com/v1/peers | grep node_id   # expect you
ps -eo pid,cmd | grep -E 'target/release/jerith run' | grep -v grep | awk '{print $1}' \
  | xargs -r kill
sleep 95 && curl -s https://p2p.jerithai.com/v1/peers      # expect {"peers":[]}
rm -rf /tmp/probe-$$ /tmp/probe.log
#    NOTE: do NOT use a bare `pkill -f <pattern>` here — the pattern text
#    also matches the invoking shell's own cmdline and it kills itself.

# 6. Design checks that must always hold (fail-closed, not fail-open):
#    - unsigned/replayed advertisements and pings get 401/403, never accepted
#    - peers list only returns same-model nodes
#    - `jerith stake status` shows chain truth (staked/slashed from GET /stake)
#    - guards: guard screens / blocked counters present in /status JSON
```

If every step above passes, the deploy is live and consistent with the
pushed code. Record the deployed commit (`git log -1`) for rollback.

## Rollback

```bash
# 1. Freeze the bad deploy (nodes just re-advertise; nothing is lost —
#    the bootstrap registry is in-memory and peers refresh within a ping
#    cycle once a good binary is back).
systemctl --user stop jerith-bootstrap

# 2. Check out the last known-good release/commit and rebuild.
git fetch --tags
git checkout v0.2.0                      # or the previous verified commit
cargo build --release --workspace

# 3. Restart, then re-run the post-deploy verification above.
systemctl --user start jerith-bootstrap
```

The chain node rolls back the same way (`git checkout <commit>` in
`~/Documents/Projects/jerithchain`, then `systemctl --user restart
jerith-node-dev`). Its state lives in `data/` + `keys/` and survives
restarts; schema migrations are additive (`CREATE TABLE IF NOT EXISTS`),
so older binaries reading a newer DB must be reviewed before a downgrade.

## Secrets handling (Discord token)

The bot token lives **not** in bot.toml but in a 0600 env file the user
service imports:

- File: `~/.config/systemd/user/jerith-bot.service.d/token.env`
  (`JERITH_DISCORD_TOKEN=...`, mode 0600, directory 0700).
- Drop-in: `.../override.conf` sets `EnvironmentFile=` to that path;
  the bot's config resolves `$JERITH_DISCORD_TOKEN` first and bot.toml
  is a fallback — bot.toml carries no token anymore.

```bash
# Rotate: edit token.env, then:
systemctl --user daemon-reload && systemctl --user restart jerith-bot
# Verify:
systemctl --user show jerith-bot -p EnvironmentFiles
journalctl --user -u jerith-bot | grep "slash commands"
```

The web skill's fetch path also DNS-pins: a custom reqwest resolver
refuses any hostname whose resolved addresses are private/loopback
(SSRF round 2, closes DNS-rebinding); see websearch.rs.

## Bot-facing ops (lore index, monitor scaffold, guild-admin)

### Lore index (OpenClaw void-lore vault port)

A per-request index over the lore vault (`$JERITH_LORE_VAULT`, else the
`[lore] vault_dir` in config.toml, else the migrated OpenClaw vault),
served by the node and surfaced in Discord as `/lore` (excerpts only —
no model call, no cost):

```bash
# Live check (index builds on demand; expect entries + scored hits):
curl -s -X POST http://127.0.0.1:8900/v1/skills/lore/search \
  -H 'content-type: application/json' -d '{"query":"membrane","top_k":2}'

# CLI equivalent (node crate):
./target/release/jerith lore "membrane" --top 3
```

No vault configured → the endpoint answers 503 with a hint, never a
wrong empty result.

### Monitor scaffold (OpenClaw ops-monitor family port)

`jerith-bot` runs five checks every `[monitor] interval_minutes`
(default 30): disk (warn 5 points before the threshold), SSL expiry of
p2p.jerithai.com, hwmon temperatures, oversized local logs, and backup
freshness (newest file in each `[monitor] backup_dirs` must be ≤26h).
Anything warn/fail → **one DM to the bot owner**; passing checks stay
quiet (no periodic "all good" spam).

```bash
# Started / healthy:
journalctl --user -u jerith-bot | grep "monitor scaffold"
# → "monitor scaffold started ... owner_dm=\"<id>\" interval_minutes=30"
```

Configured in `~/.jerith-mesh/bot.toml`:

```toml
[monitor]
enabled = true
interval_minutes = 30
backup_dirs = ["/home/noodly/.openclaw/jerith-coin/live", "/home/noodly/.jerith-mesh/data"]
```

The first `backup_dirs` entry is the chain snapshot the 5-min
`jerith-snapshot.timer` republishes — if that timer dies, the owner
gets a DM within half an hour. The monitor is **reports only**: it
never restarts services or repairs anything (system-level actions stay
behind the owner gate).

### Guild-admin `/timeout` (owner-DM only)

`/timeout <guild_id> <member_id> [minutes] [reason] [clear]` sets or
clears a member timeout (Discord caps at 28 days = 40320 min). It is
deliberately **not** classified user-level in `owner.rs`, so the
default-deny gate refuses it everywhere except a DM from the bot
owner's account — including guild channels even when the owner types
it. Because invocation is from a DM, guild + member ids are explicit
arguments.

```bash
# Verify registration (read-only, token stays in bot.toml):
APP=$(grep '^discord_token' ~/.jerith-mesh/bot.toml | sed 's/.*= *"\(.*\)"/\1/' \
      | base64 -d | head -c 20); APP=${APP%%.*}
curl -s -H "Authorization: Bot $(grep '^discord_token' ~/.jerith-mesh/bot.toml \
  | sed 's/.*= *"\(.*\)"/\1/')" "https://discord.com/api/v10/applications/$APP/commands" \
  | grep -o '"name":"timeout"'
```

## Escalation pointers

- Tunnel misroute (CNAME pointed at the wrong tunnel UUID): see the
  gotcha in `DEPLOY-BOOTSTRAP.md` — route by UUID explicitly.
- A node demoted/degraded: check `/status` `gate_clear` (stake escrow
  short?), bootstrap reachability, and `journalctl --user -u jerith-node`.
- Fraud-correctness: `verify_percent`/ringer counters live in `/status`
  (`meters`, `verify` fields); a spike in mismatches precedes scam/round
  slashing notices — check `GET /stake` on the chain node for who burned what.
