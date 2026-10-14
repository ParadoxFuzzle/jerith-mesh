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

## Nodes

Jetson node (added 2026-10-14): Orin Nano `noodlypanda@192.168.1.150`, aarch64 Ubuntu 22.04 — binary built natively from a fresh clone, `~/.jerith-mesh/config.toml` with fresh Ed25519 identity (`node_id 4353ea12d8100f45`), runtime at local ollama (`127.0.0.1:11434`, JetPack6 NVIDIA build) carrying the same model digests as the main node (`qwen3:8b 500a1f06…`, `qwen3:4b 359d7dd4…`, `qwen3-embedding:0.6b ac6da0df…`). Runs as a systemd **user** unit (`~/.config/systemd/user/jerith-node.service`) with linger enabled, so it survives reboot without sudo. Verified: `/status` green, `POST /v1/chat/completions` round-trip (139 tokens served),`GET /v1/peers?model=qwen3:8b` lists both nodes (Jetson + main).

Known quirk (both nodes): advertised `address`/`rpc_address` are `127.0.0.1` — cross-host reachability rides the bootstrap CHAT lane until an advertise override is added to the config.

## Security hardening log

Round 3 (`9493165`): wire-auth verification for `/v1/report-dead` and
`/v1/report-fraud`; guard screens multimodal content arrays;
`/v1/skills/memory/embed` text cap (64k chars → 413).

Round 4 (this round) — unbounded-growth + flood tightening:

- **Bootstrap registry bounds**: `PeerAd` string fields capped
  (node_id 64, model_id 64, quant 32, address/rpc_address 128; rejected
  400 BEFORE signature work), `contribution_percent` 0–100 sanity, and a
  20 000-entry registry cap (NEW node ids beyond the cap → 503;
  refreshes always pass).
- **Background TTL sweeper**: registry expiry no longer depends on
  someone polling `/v1/peers` — `serve()` sweeps at half the advertise
  TTL (floor 15 s), so an idle bootstrap cannot accumulate stale ads.
- **Chat forward cap**: `/v1/chat/completions` refuses serialized
  payloads > 1 000 000 bytes with 413 before touching the runtime or
  mesh fallback.
- **Memory retention**: the local SQLite memories table prunes the
  oldest rows at insert once past `MAX_MEMORY_ROWS` (10 000) — the
  append-only table can no longer grow without bound.
- **Per-IP rate limiting** (`core/ratelimit.rs`, fixed 60 s windows):
  bootstrap anonymous reads 240/min/IP, gateway all-routes 600/min/IP.
  429s carry `Retry-After`. Attribution: `CF-Connecting-IP` / first
  XFF hop trusted ONLY from loopback peers (the local tunnel proxy) —
  a directly-reachable origin cannot spoof client IPs to hide behind
  or drain other buckets; everyone else is pinned to their socket peer.
  - **Peer-identity allowance (NAT scaling)**: a `/v1/peers` fetch
    carrying wire-auth headers that verify against
    `peers-req1|{stamp}` pays an Ed25519-fingerprint bucket
    (`id-…`, 1440/min) instead of the IP's anonymous floor — the ping
    loop signs every fetch, so legitimate nodes scale past the ~20
    nodes-per-NAT-IP ceiling while unsigned floods stay locked to
    240/min/IP. A forged signature degrades to the anonymous floor
    (never the allowance). Node compatibility: unsigned fetches still
    work, just within the floor.
- **Edge layer (LIVE 2026-10-14, ruleset ac289166024247efa78c1e1f979fcca0)**:
  a Cloudflare WAF **rate limiting rule** on the zone complements the
  origin limiter —
  it absorbs floods before they cost origin CPU/runtimes and stays
  alive across origin restarts (which reset the local windows). It is
  NOT a replacement: it only covers the tunnelled zone path (the
  direct-reachable bootstrap + gateway surfaces and QUIC NodeRPC see
  only the origin limiter), its counters are approximate and
  per-data-center (no global counter, so N PoPs ≈ N× budget), and its
  enforcement delay makes the exact-origin budget (240/min, verified)
  an origin-side property. Free-plan rule recipe (dashboard →
  Security → WAF → Rate limiting rules):
  - expression:
    `not cf.client.bot and (starts_with(http.request.uri.path, "/v1/peers") or http.request.uri.path eq "/v1/advertise" or http.request.uri.path eq "/health")`
  - counting: same IP; period **10 s** (free-plan max); threshold
    **100** requests — ~half the origin budget's sustained rate
    (240/min ≈ 40–240 bursts per 60 s), far above a legit TTL
    heartbeat (~4 req/min); action **Block**, mitigation **10 s**.
  - Pro upgrade: period 60 s, threshold 200, mitigation 1 h (2 rules
    — add a gateway-lane rule if a public gateway hostname ever
    exists).
  - As code: the API-shaped ruleset + apply/verify commands live in
    `deploy/cloudflare/` (`rate-limit-rule.json` + README).
  - **Rollout verification (2026-10-14)**: live burst through the
    tunnel tripped at 102×200 + 28×429 — threshold lands at ~100/10s
    as configured; the 429s were CF's `error code: 1015` edge page
    (text/plain, retry-after: 10) and BOTH origin 429 counters
    (node /status + bootstrap /health) stayed untouched through the
    burst — the edge absorbs it before a single origin budget hit.
    Recovery to 200s confirmed right after the 10 s mitigation. Plan
    caveat: Free zone → the expression can only scope paths (no Host
    field); those paths are bootstrap-only in practice.
  - Stress-tested against legitimate traffic (2026-10-14): a node's
    ping loop ticks every 10 s with 1× GET `/v1/peers` + 1× PUT
    `/v1/advertise` = **12 req/min/node** (journal stamps confirm the
    live 10 s cadence; peer pings are direct peer-to-peer, not via the
    tunnel). Replays through the tunnel: steady 2/10s × 7 ticks, a
    30-request reconnect burst, and a worst-plausible-scale 60-in-one-
    window burst — **all 200s, zero 429s**. Headroom vs the planned
    edge threshold (100/10s): ≥1.67× burst, ~33× steady per node.
    Binding constraint note: the ORIGIN 240/min holds 20 nodes on one
    shared IP; the edge rule (600/min sustained equivalent) is the
    more generous of the two — NAT deployments beyond ~20 nodes/IP
    hit the origin budget first and need the allowance pattern
    (separate identities otherwise). This test validated 
    *legit volumes* vs the planned thresholds on the real edge path —
    the live rule isn't configured yet, so edge-side 429 behavior
    itself is verified only after rollout.

```bash
# QUIC dispatch flood hardening: the RPC server now bounds —
#   1. rpc frame read (1 MiB cap — a newline-less stream used to grow
#      memory without bound; over-cap frames close the connection);
#   2. handler-facing payloads (ExpertQuery input <= 65 536 floats,
#      ChatQuery messages <= 900 000 bytes — refused with an Error
#      reply before the runtime is touched);
#   3. concurrent inbound connections (128 via serve_with_limit;
#      backpressure — pending dials queue in quinn's incoming queue
#      instead of spawning unbounded tasks; per-request connections
#      turn permits over quickly). The known outbound side was
#      audited serial (chat/expert dispatch goes peer-by-peer, no
#      broadcast storm) and router `completed` maps are per-call.
#   Not done this round: transport-level stream caps — tightening
#   quinn's max_concurrent_* in the server transport regressed
#   handshakes nondeterministically (client dials got
#   CONNECTION_REFUSED; a quinn 0.11 internals quirk) — revisit only
#   after a quinn upgrade. Remaining accepted exposure: inbound
#   requests are stamp-freshness-checked but not signature-verified
#   yet (inbound wire auth = cert-pinning phase); flood cost per
#   request is modest (conn cap + serial per-conn + payload caps).
# Web UI hardening: the node dashboard (/) ships hash-pinned CSP (the
# inline script/style digests are recomputed from the page itself at
# first request — page edits re-pin automatically, no 'unsafe-inline'),
# plus nosniff / no-referrer / no-store; /status answers no-store too.
# The whole gateway + UI bind 127.0.0.1:<gateway_port> — loopback only;
# the only internet-exposed mesh surface remains p2p.jerithai.com:8901.
# Security-reject metrics: the node /status carries a `security` section
# (rate_limited_429_total, oversize_reject_total, oversize_chat_rejects,
# oversize_memory_rejects, oversize_bootstrap_rejects); the standalone
# bootstrap /health carries the same section for its own process. The
# bootstrap's counters never appear in the node's /status — they are
# separate processes.
# Verify round 4 (should show ~240 OK then 429s; window = 60 s):
for i in $(seq 1 250); do curl -s -o /dev/null -w "%{http_code}\n" \
  http://127.0.0.1:8901/v1/models/counts; done | sort | uniq -c
# Oversize probes:
#   413 node:  any chat body > 1 MB serialized.
#   400 bootstrap: PUT /v1/advertise with model_id 2 000 chars (no
#   wire-auth headers needed — validation precedes crypto).
# Production CF-path verification (2026-10-14, all green):
#   - Burst https://p2p.jerithai.com/v1/peers from the probe host:
#     trips at exactly 240 (240 = 239 burst + 1 health), 429s carry
#     Retry-After; while the public-path bucket is dry, direct
#     loopback requests still 200 — buckets are distinct, so public
#     traffic is attributed via the CF header (real client IP), not
#     the tunnel's socket.
#   - Isolation: with the probe host's bucket dry (429s), requests
#     from two other/external IPs (Serper fetcher post-window,
#     r.jina.ai mid-window) still served normal 200 peer lists —
#     one client's flood cannot lock out others. (Serper's one-off
#     mid-window 429 looked like CF-edge throttling of that fetcher,
#     not origin behavior — origin is per-IP by construction.)
```

Known remaining limitations (accepted, bounded):

- Signed-but-malicious identity holders can still file dead/fraud
  reports; per-(reporter, kind) dedupe + reputation floor bound the
  damage by design.
- Header-attribution trusts the loopback tunnel proxy; a compromised
  local process could impersonate other buckets' clients (host-level
  trust, same as the tunnel itself).
- Rate limits are per-process in-memory: a bootstrap restart resets
  every window (a client's burst budget re-arms briefly).
- memory.search clamps `top_k` to 1–25 and answers 413 above a
  10 000-char query (round 4 follow-up, same live node session):
  silent clamp, not a rejection — oversized top_k returns as many hits
  as the store has, capped at 25.
- Stake tests used to read/write the *global* ledger
  (`~/.jerith-mesh/data/stake.json`) — shared-state pollution once
  flipped a lock test's verdict order-dependently and journaled fake
  slash events into the production file. Fixed: every test manager
  rearms to a private temp ledger and `StakeManager::new` no longer
  persists eagerly; only real mutations write. Production datum was
  all-zeros throughout (live node's own status confirmed), so nothing
  was lost.
- ~~QUIC e2e test `router_execute_over_quic_prefers_local_and_falls_through`
  is load-sensitive…~~ FIXED (`d502627`): the flake was the test
  routers' 250 ms dispatch timeout — load noise could time out one
  attempt and flip candidate selection. Test timeouts are now 5 s and
  the suite verifies green 3/3 under 2× CPU overcommit; no production
  behaviour involved.



## Escalation pointers

- Tunnel misroute (CNAME pointed at the wrong tunnel UUID): see the
  gotcha in `DEPLOY-BOOTSTRAP.md` — route by UUID explicitly.
- A node demoted/degraded: check `/status` `gate_clear` (stake escrow
  short?), bootstrap reachability, and `journalctl --user -u jerith-node`.
- Fraud-correctness: `verify_percent`/ringer counters live in `/status`
  (`meters`, `verify` fields); a spike in mismatches precedes scam/round
  slashing notices — check `GET /stake` on the chain node for who burned what.
