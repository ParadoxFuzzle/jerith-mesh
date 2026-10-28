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

### Control-plane locality (co-hosted convention)

When the mesh node and its bootstrap share a host (this deployment),
the node's control plane must speak LOOPBACK, not the public tunnel:
`config.toml` sets `bootstrap_url = "http://127.0.0.1:8901"` and puts
`http://127.0.0.1:8901` FIRST in `[discovery] servers` (the tunnel URL
and the LAN address stay behind it as failover legs). Every ping tick,
census fetch, fraud flush, `/status` mirror probe and stake
chain-probe then rides loopback instead of a Cloudflare round trip.
Measured on the live node 2026-10-18: `/status` 140-190 ms → ~2.4 ms
end-to-end (probe parallelization + pooled clients + this), and the
ping tick's peer refresh reads `bootstrap=http://127.0.0.1:8901` in
the journal. Remote nodes keep the tunnel URL as their primary — the
ordering is per-host config, not a protocol change. If the learned
server set (`data/discovery-servers.json`) still pins a remote server
as last-good after the swap, clear the file once; the node re-pins
from the config-ordered rotation within one tick.

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
- **Wallet-card endpoint** (verified 2026-10-18): the startup /balance
  probe resolves `[crypto] wallet_endpoint` → the mesh's chain base
  (config `chain_url` / `$JERITH_CHAIN_URL`, i.e. the :8322 co-host the
  miner credits) → the legacy miner-facing :8300 default. The :8300 and
  :8322 instances token-gate independently — a 401 at startup means the
  resolved endpoint and `keys/api.token` disagree, not that the wallet
  is empty.
- **LLM runtime**: LiteLLM proxy on `127.0.0.1:8000` (`litellm-proxy.service`,
  config `~/.openclaw/litellm-config.json`, master key in
  `general_settings.master_key`). The node's `[chat] runtime_*` model pins
  resolve through it. Every fallback ladder ends at the LOCAL Ollama
  `qwen3:8b` last resort — when the OpenRouter free tier exhausts its
  daily cap (`free-models-per-day`, HTTP 429 with `type: throttling_error`),
  chats degrade to local instead of failing (2026-10-17 outage: the
  ladders were OpenRouter-only, so a 50/day-cap 429 killed every rung and
  the bot showed the failure banner ~35 s later — the whole ladder has to
  walk its 429s first, so expect ~30–40 s before the local answer lands).
  Config edits need `systemctl --user restart litellm-proxy`; boot takes
  ~45 s before the port opens (model registration).
- **Bot attachment sandbox**: non-image Discord attachments are read
  inside a disposable container from `jerith-sandbox:latest` — built by
  `docker build -f deploy/docker/sandbox.Dockerfile -t jerith-sandbox:latest .`
  (extends the migrated OpenClaw sandbox base with poppler-utils; rebuild
  it on the host after a fresh clone or the files line reads
  "could not read: sandbox …"). Safety contract lives in
  `crates/bot/src/sandbox.rs` (`--network none`, caps dropped, read-only
  rootfs, uid 65534, fresh staging dir per message); knobs in bot.toml
  `[sandbox]`. Successful extractions cache per Discord attachment id
  (+ filename/size fallback; 64 entries, 2 h TTL) — re-forwarded files
  don't re-run the container, restart clears it. Archives read as a
  listing plus the nested text members (bounded). `/read <attachment>`
  shows the extraction on demand; missing docker only degrades files to
  honest "could not read" lines — images (vision tier) are unaffected.
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

### Chain API token rotation (drill-proven procedure)

The chain node token-gates `/stake*`, `/balance`, `/transfer` with its own
`keys/api.token`; the mesh's chain probes derive theirs from
`$JERITH_TOKEN` → `~/.jerith-mesh/keys/api.token`. When they drift, probes
read `401 {"detail":"invalid or missing API token"}` and the damage is
silent — both directions, drill-proven (2026-10-17 discovery drill, again
2026-10-18 during the §10 drill): the anti-sybil stake half degrades to
the age rule (a young but staked reporter loses its weight), stake
lock/unlock fails, the wallet card empties.

Rotation procedure:

1. Write the new token into the chain repo:
   `~/Documents/Projects/jerithchain/keys/api.token` — the chain node
   loads it once at boot (`API_TOKEN = sec.load_or_create_api_token`),
   so it must restart to adopt it.
2. Mirror it into the mesh home, restricted:
   `cp <chain repo>/keys/api.token ~/.jerith-mesh/keys/api.token && chmod 600`
   (repeat for any `JERITH_TOKEN_FILE` location a co-hosted process uses).
3. Restart the token-cachers: `systemctl --user restart jerith-chain-node`
   (loads the new token) and `systemctl --user restart jerith-node` —
   the StakeManager caches the token at boot, so the node's stake flows
   will keep the old secret until restarted. The standalone bootstrap
   probes the token fresh per citizenship check, but restart
   `jerith-bootstrap` too to keep the fleet symmetric.
4. Verify from the mesh side (the drill's exact check):
   ```
   curl -s -H "Authorization: Bearer $(cat ~/.jerith-mesh/keys/api.token)" \
        "http://127.0.0.1:8322/stake?discord_id=<node_id>"
   ```
   → 200 JSON (`staked_uj`, escrow address) is the sync proof; 401 means
   a copy is stale. Then `/status` (`wallet.reachable`) and the
   bootstrap's `/health` (still 200).

### Chain smoke + bookkeeping drift watch (2026-10-18)

The rotation section above was written because token drift was SILENT.
Two surfaces now catch it — and catch the other chain-side failure class
(a ledger that shrinks between live reads):

```bash
# Read-only live-chain smoke suite: run it after any chain upgrade,
# token rotation, or restore. READ-ONLY by construction (GETs only).
JERITH_TEST_CHAIN_URL=http://127.0.0.1:8322 \
  cargo test -p jerith-mesh-core --test chain_smoke
# → 5/5 pass on a healthy chain: /status shape + supply invariant,
# token-less reads REFUSED, exists:false balance for an unknown identity,
# /stake view contract, and monotonic bookkeeping across two reads.
# WITHOUT the env var the suite SKIPS LOUDLY (the reason is printed —
# never silent).
```

Runtime drift surfaces (live in every `/status` since 2026-10-18):

- `/status` → chain section → `drift`: the monotonicity watch's counters
  (`regressions` MUST stay 0; a nonzero value means the chain's
  `total_mined_uj` or height SHRANK between live reads — reorg, rollback,
  or a manual restore rewound the ledger). Every regression also lands a
  journal error: grep `bookkeeping regressed`.
- `/status.crypto` → `mine_failure_streak` / `mine_failures_total`: the
  miner-half alarm. A streak ≥ 5 journal-warns on its own
  (`miner failing repeatedly — token drift?`); in practice a rising
  streak + frozen `mine_wins_total` is exactly the silent-401 class the
  rotation procedure above fixes.

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

# 5b. Discovery failover drill (Phase 1 rotation, verified live 2026-10-15):
#     briefly DARK the primary bootstrap and watch the node degrade
#     honestly tick-by-tick, then recover within one tick. Requires the
#     node under test to run the discovery-aware binary.
#
#     a. Stop the primary (on the bootstrap host):
systemctl --user stop jerith-bootstrap
#
#     b. For ~2 ping cycles watch the node's journal: every tick must
#        WARN and NAME THE ROTATION — never crash, never fake success:
journalctl --user -u jerith-node --since '1 minute ago' --no-pager | \
  grep -E 'bootstrap list refresh failed|advertise rejected'
#        expect: WARN ... error=all bootstrap servers failed
#        (tried ["https://p2p.jerithai.com", "http://<lan>:8901"]), last: ...
#
#     c. Recover (restart the primary), wait one cycle, expect INFO:
systemctl --user start jerith-bootstrap
sleep 12 && journalctl --user -u jerith-node --since '30 sec ago' --no-pager | \
  grep 'peer list refreshed'
#        expect: INFO mesh: peer list refreshed from bootstrap fetched=N
#        bootstrap=https://p2p.jerithai.com
#
#     d. Truth surface: /status discovery must carry the rotation and the
#        re-pinned last-good:
curl -s http://127.0.0.1:8900/status | python3 -c \
  'import json,sys; d=json.load(sys.stdin)["discovery"];
   print(len(d["configured"]), d["last_good"], len(d["ad_servers"]))'
#        expect: 2 https://p2p.jerithai.com 2
#
#     NOTE: with BOTH servers dark the node must keep serving local
#     traffic (degraded ≠ down); with the second (LAN) bootstrap live,
#     steps (b) become 'refreshed from bootstrap=<lan-url>' instead —
#     that is the two-server leg working. Run scripts/
#     setup-lan-bootstrap.sh ON the second node to arm it.
#
#     OBSERVED 2026-10-15 (both variants verified on main node):
#     - all-dark: every tick WARNed with the full rotation named; node
#       kept serving locally; recovery INFO landed on the first tick
#       after `systemctl --user start jerith-bootstrap`.
#     - two-server: with a working second bootstrap in the rotation,
#       the FIRST tick after the primary went dark already refreshed
#       `fetched=N bootstrap=http://127.0.0.1:8902` — zero degrades,
#       zero warns; `last_good` followed the server that answered.#     - ops detail learned: raw config.toml edits do NOT hot-apply —
#       a restart (or `jerith config set`) loads them; /status's
#       discovery section is the truth of what the loop is using.
#
# 5c. Memory backend switch drill (PostgreSQL+pgvector, Phase 0.5,
#     verified live 2026-10-15 against pgvector/pgvector:pg16 in Docker):
#
#     a. Choose it at setup — `--memory postgres --memory-dsn
#        postgres://user:pass@127.0.0.1:5432/<db>?sslmode=disable` —
#        or set `memory_backend = "postgres_pgvector"` + `memory_dsn`
#        in config.toml (env override JERITH_MEMORY_DSN wins). Memory
#        stays strictly node-private: the DB is the operator's own.
#     b. Boot banner is honest about the store that actually opened:
#        "memory PostgreSQL + pgvector (dim 256), local-only" when the
#        extension is installed, "PostgreSQL (bytea polyfill — no
#        pgvector extension)" when the connection lacks it (e.g.
#        non-superuser on a bare install) — cosine in Rust, same
#        semantics, mode always reported.
#     c. Round-trip: `jerith memory add`, memory.search embeds nomic and
#        ranks via `<=>` (or the polyfill scan); skills say so:
#        POST /v1/skills/memory/{embed,search} return
#        {"id":N,"provider":"nomic"} / scored hits; verify the scored
#        ordering matches reality before trusting the backend switch.
#     d. Fail-loud checks: postgres chosen WITHOUT memory_dsn → the
#        process refuses to start (typed error naming the missing DSN —
#        never a silent swap to SQLite); unreachable DSN → node runs
#        degraded with "memory unavailable" and memory.* skills return
#        the honest "memory backend not configured" error.
#     e. Layout guardrail: a memories table written by the OTHER mode
#        (bytea vs vector) is refused at connect with a migrate
#        manually note — the node never reshapes the user's table.
#        Env-gated live tests prove both modes:
#        JERITH_TEST_PG_URL (superuser, extension present) and
#        JERITH_TEST_PG_URL_NOEXT (extension-less DB) with
#        `cargo test -p jerith-mesh-core memory::tests::pg`.
#
#     f. Retention cap (`memory_max_rows` in config.toml, default
#        10 000; `0` = UNCAPPED): newest rows survive, prune fires at
#        insert time; `jerith config set memory_max_rows=…` rides the
#        memory hot-swap triple (applies within one ping tick, nothing
#        deleted at swap time; pruning follows from the next insert).
#        Semantics per lane: pgvector keeps/creates its ANN index
#        (HNSW; IVFFlat fallback covers pgvector < 0.5 servers with
#        lists sized rows/1000, √rows past 1M); SQLite stays a FULL
#        EXACT SCAN (no native vector index) — raising/uncapping it
#        warns loud at `config set` and config validate.
#
#     g. Legacy pgvector tables (made before `vector(N)` pinning):
#        they now migrate IN-NODE at connect — a table that is
#        uniform-dim AND matches the configured `embedding_dim` is
#        rebuilt pinned transactionally (rows verbatim: ids, text,
#        provider, created_ms, vectors; sequence re-linked; no
#        re-embedding) and the ANN index becomes buildable. Mixed-dim
#        or mismatched-dim tables stay untouched (the scan-path shapes).
#        Live-verified 2026-10-20 against pgvector/pgvector:pg16:
#        post-migrate column probed `vector(64)` server-side, ids and
#        timestamps verbatim, indexed search green; the mismatched-dim
#        table stayed dimension-free. No operator action needed — the
#        journal line "legacy dimension-free memories table MIGRATED"
#        is the receipt.
#
# 5d. Memory-backend hot-swap + migration drill (Phase 0.5 completion,
#     verified live 2026-10-15 against pgvector/pgvector:pg16 in Docker):
#
#     a. With the node RUNNING on sqlite: `jerith config set
#        memory_backend=postgres memory_dsn=...` — prints the migration
#        guidance (which store rows live in, the copy command, the
#        verify step); config saves, nothing restarts.
#     b. Within the next ping tick the mesh loop's config file re-read
#        live-swaps the store: journal shows
#        "memory store live-swapped (config hot-reload) from=…SQLite…
#        to=PostgreSQL + pgvector (dim 256)" and /status memory_store
#        tracks the store actually serving — verified in BOTH
#        directions (sqlite→pg→sqlite→pg, two swap events, no restart).
#        The pinned `vector(256)` column and the HNSW cosine index
#        (idx_memories_embedding_hnsw) build at connect.
#     c. Rows are NOT auto-moved. `jerith memory migrate` copies them
#        into the now-configured store verbatim (no re-embedding),
#        skipping duplicates by text: rerun reports copied: 0 · skipped
#        N, so it is idempotent; `--drop-source` empties the SOURCE
#        only after a verified copy (verified: copied 2 → rerun
#        skipped 2 → --drop-source emptied 2 from the sqlite side,
#        target still 2).
#     d. Fail path: point memory_dsn at a dead postgres — one WARN
#        "memory store swap failed — the previous store keeps serving"
#        (60 s backoff, exactly one WARN across three ticks), /status
#        stays honest on the working store, and the failed config is
#        never adopted. Restore the DSN and the node heals quietly.
#     e. Identity guardrail: config.toml node_id/secret changes are
#        refused at the tick with an ERROR (hot identity swap would
#        forge every signed wire surface); cold fields (model, ports,
#        runtime…) WARN restart-required instead of hot-applying.
#
# 5e. SQLite file-backed stores open with `PRAGMA synchronous = NORMAL`
#     (verified live 2026-10-18). WAL alone still fsyncs every
#     transaction: 10k inserts at the MAX_MEMORY_ROWS retention cap took
#     60.4 s (~6 ms each); with NORMAL the same corpus builds in 2.5-3.2 s
#     steady state (~20x). Durability contract: an app crash cannot
#     corrupt or roll back committed writes; ONLY an OS power-loss can
#     drop the most recent (un-checkpointed) transactions - the accepted
#     stance for a retention-capped, fully-rebuildable chat-memory cache.
#     If an operator needs power-loss safety, set synchronous=FULL in
#     memory.rs's open path, then re-run the release bench
#     (`cargo test --release -p jerith-mesh-core -- --ignored
#     bench_sqlite_search_at_retention_cap --nocapture`) to see the new
#     per-insert cost before committing to it.
# 6. Design checks that must always hold (fail-closed, not fail-open):
#    - unsigned/replayed advertisements and pings get 401/403, never accepted
#    - peers list only returns same-model nodes
#    - `jerith stake status` shows chain truth (staked/slashed from GET /stake)
#    - guards: guard screens / blocked counters present in /status JSON
```

If every step above passes, the deploy is live and consistent with the
pushed code. Record the deployed commit (`git log -1`) for rollback.

## Optional-chain wallet adapters (§9, per-chain examples)

The default chain is JerithChain. Setting `[crypto] chain` to a PoW chain
switches the node to the **wallet-card half only**: `/status` probes the
endpoint you supply for a live, chain-truthful balance (`crypto.
optional_wallet`); mining stays refused by the honest note — PoW chains
don't mesh-mine. Config keys (config.toml):

```toml
[crypto]
chain = "bitcoin"            # bitcoin | ethereum | dogecoin | monero
wallet_endpoint = "..."      # ALWAYS user-supplied; see per-chain below
wallet_address = "..."       # required for ethereum; optional (recommended) for Core chains; unused by monero
```

A wrong endpoint never becomes a fake zero — the card degrades to
`{ok:false, error:"..."}` in /status and `wallet(oc) … (<error>)` in the
banner. Verify each shape after edit (no restart needed for a
config-reload drill; the card probes per /status request):

```bash
curl -s :8900/status | python3 -c 'import json,sys; print(json.load(sys.stdin)["crypto"]["optional_wallet"])'
```

```toml
# — Bitcoin Core (bitcoind-defaults node; Dogecoin Core is the same shape) —
# RPC auth rides the URL: http://<rpcuser>:<rpcpass>@127.0.0.1:8332
[crypto]
chain = "bitcoin"
wallet_endpoint = "http://__rpcuser__:__rpcpass__@127.0.0.1:8332"
# watch exactly one address (RPC getreceivedbyaddress, 1-confirmation):  
wallet_address = "bc1qyy74txkqpa6y4mpanh5y5phqqktghwx30nyn8y"
# omit wallet_address to show the node wallet's own getbalance instead

# — Dogecoin Core — same as Bitcoin, default port 22555 —
[crypto]
chain = "dogecoin"
wallet_endpoint = "http://__rpcuser__:__rpcpass__@127.0.0.1:22555"
wallet_address = "D8aX7z1KL4bYQKQ1znyEbEhhhkYXPf1Qps"

# — Ethereum (any node with the standard JSON-RPC; plain HTTP okay for LAN) —
[crypto]
chain = "ethereum"
wallet_endpoint = "http://127.0.0.1:8545"
# REQUIRED + exactly 0x + 40 hex (eth_getBalance is address-keyed):
wallet_address = "0x95222290dd7278aa3ddd389cc1e1d165cc4bafe5"

# — Monero (monero-wallet-rpc serving ONE wallet file; no address needed) —
# Point it at wallet-rpc's json_rpc surface; the doc-default port is 18083:
[crypto]
chain = "monero"
wallet_endpoint = "http://127.0.0.1:18083/json_rpc"
# wallet_address unused (the probe is whichever wallet file you started with)
```

Pre-wire check per chain, outside the mesh — run the exact call
the adapter makes (these must succeed before blaming the node):

```bash
# bitcoin/dogecoin Core (the probe asks getreceivedbyaddress when an
# address is set, else getbalance):
curl -su u:p -H 'content-type: application/json' \
  -d '{"jsonrpc":"1.0","id":"probe","method":"getreceivedbyaddress","params":["<addr>",1]}' \
  http://127.0.0.1:8332
# ethereum:
curl -s -H 'content-type: application/json' -d '{"jsonrpc":"2.0","id":1,"method":"eth_getBalance","params":["0x…","latest"]}' http://127.0.0.1:8545
# monero-wallet-rpc:
curl -s -H 'content-type: application/json' \
  -d '{"jsonrpc":"2.0","id":"0","method":"get_balance","params":{}}' \
  http://127.0.0.1:18083/json_rpc
```

Rolling back a chain switch is a config edit away (chain back to
`jerithchain`); the JER wallet card and the miner come straight back.

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

### Model router tiers (tool / vision / deep) — pinning

The gateway router re-classifies each *auto* chat request (placeholder /
advertised base model callers only — explicit model ids always win):
tool-bearing requests → the **tool** tier (`runtime_tool_model`, parse
reliability first), image content → **vision** (`runtime_vision_model`),
complex prompts → **deep** (`runtime_deep_model`), else the caller's
model untouched. All three are cold-reload keys (restart required); set
them with `jerith config set <key>=<model>` (or `""` / `none` to disable
the tier). The tool tier dates from 2026-10-16 (before that, agent tool
loops ran on the base model — qwen3:8b returns `{}` on tool payloads
while thinking; see the LangChain integration README for the byte-level
finding).

The tool tier's forwarded payloads also carry **`think: false`**
(2026-10-16): nemotron put its whole response inside the reasoning
channel and ollama answered `content: ""` at exactly the cap
(`finish_reason: stop`, zero content) — the "⚠️ the runtime returned an
empty answer" family the bot showed its users. Merged at the gateway on
tool-tier reroutes ONLY (deep-tier/pinned requests keep their thinking;
LiteLLM drops the unknown field, so the merge is inert elsewhere).
Live-proven: the previously-empty probe now lands `finish_reason:
tool_calls` in-budget.

**Single-model trap:** every tier key serde-defaults *enabled* (a fresh
config without the key gets the tool pick `nemotron-3-nano:4b` and the
vision pick `minicpm-v4.6:1b`). On a node that deliberately serves one
model, a missing key means the *next* binary update silently reroutes
tier traffic at models not pulled there. The Jetson node is pinned
ahead of time (`runtime_deep_model` / `runtime_tool_model` /
`runtime_vision_model` all `""`, set 2026-10-16 with the update-pending
old binary — unknown keys parse as no-ops, so pre-pinning is safe);
pin the same way before updating any other single-model node.

## Nodes

Jetson node (added 2026-10-14; binary updated to current main 2026-10-16): Orin Nano `noodlypanda@192.168.1.150`, aarch64 Ubuntu 22.04 — release binary built natively from the clone at `~/jerith-mesh` (2026-10-16: pulled `a84d7d7`, `cargo build --release --bin jerith`, 2m27s incremental, unit restart), `~/.jerith-mesh/config.toml` with fresh Ed25519 identity (`node_id 4353ea12d8100f45`), runtime at local ollama (`127.0.0.1:11434`, JetPack6 NVIDIA build). **Single-model node: `qwen3:8b 500a1f06…` only** (same digest as the main node) — 8b+4b in 7.4 GB RAM thrashes with thinking models, so the 4b and the embedding model were removed (2026-10-14); `runtime_deep_model`/`runtime_tool_model`/`runtime_vision_model` are all `""` (tiers disabled; deep pinned 2026-10-14, tool+vision preemptively pinned 2026-10-16 — see "Model router tiers" above, whose serde defaults would otherwise have auto-rerouted at unpulled models on this very binary update) and `embeddings_enabled = false`. Deep-tier, tool-tier and embedding lanes stay on the main node (`qwen3:8b` + tool pick `nemotron-3-nano:4b` pinned 2026-10-16 + `qwen3:4b` + `qwen3-embedding:0.6b` there). Update-day verification (new binary + pins live): boots clean (gateway :8900, peer fetch 2 peers, `/health` 200); tool-bearing auto request → 200 served by `qwen3:8b` with **no `model router: rerouted` journal line** (the direct proof the pin survived the update); image-bearing auto request → no reroute, runtime 400s "Multimodal data provided, but model does not support multimodal requests" (correct single-model behavior — image traffic belongs on the main node); basic chat round-trips. Runs as a systemd **user** unit (`~/.config/systemd/user/jerith-node.service`) with linger enabled, so it survives reboot without sudo. Verified (2026-10-14): `/status` green, `POST /v1/chat/completions` round-trip (139 tokens served), `GET /v1/peers?model=qwen3:8b` lists both nodes (Jetson + main).

Direct cross-host meshing (same day, commits `7924408` + `c79eca3`):

- `advertise_host` (config, default: auto-detect via the std UDP connect
  trick) replaces the hardcoded `127.0.0.1` advertise — the registry now
  shows both nodes at their LAN addresses (`192.168.1.6` / `192.168.1.150`).
- `gateway_bind` (config, default `127.0.0.1`) — the HTTP gateway used to
  bind loopback only, so remote peers couldn't complete the signed /health
  ping or cross-host chat even with routable ads. Both nodes run with
  `gateway_bind = "0.0.0.0"`. Verified live: cross-host `/health` 200 both
  directions, cross-host `POST /v1/chat/completions` 200.

Discovery failover — `[discovery] servers` in `config.toml` (Phase 1):

The primary bootstrap (`p2p.jerithai.com`, Cloudflare Tunnel) is now first
among equals, not a single point of failure. Additional bootstraps go in
`[discovery]`:

```toml
[discovery]
servers = ["https://backup-1.example.org", "https://backup-2.example.org"]
```

- **Per-call failover, not per-process.** Every discovery HTTP leg (peer
  fetch) tries last-good first, then config order, then learned: a dead
  primary drops one tick, not the node. `/status` → `discovery` shows
  `configured` / `learned` / `last_good` / `ad_servers` — check `last_good`
  when suspicious (it names the server that answered the last tick).
- **Gossip + persistence.** Servers learned from peers' signed ads
  (https-only, max 8 per ad, 16 total in rotation) persist to
  `<data>/discovery-servers.json` — a restart keeps the list; delete the
  file to reset learned entries (configured ones always survive).
- **Hot reload.** Editing `servers` (or `jerith config set`) takes effect
  within one ping cycle — no restart needed.
- **Validation is fail-closed.** Malformed URLs (`config check`) refuse to
  boot: only `http(s)://host[:port][/path]` origins pass — no userinfo,
  query or fragment (the `{url}/v1/peers…` join would splice those in).
- **Plain http stays config-only.** LAN bootstraps (`http://192.168.1.6…`)
  rotate fine, but are never learned from gossip and never gossiped in
  ads from peers — a hostile ad can't repoint the fleet at a capture
  proxy. Cap: the first 16 valid https entries win; the rest are dropped.

Hostile-ads audit (2026-10-15) — what a malicious peer CAN and CANNOT do
through its signed ad's `discovery_servers`:

- **URL injection: no.** Only `http(s)://host[:port][/path]` origins pass
  validation (no userinfo `@`, no `?`/`#`, no whitespace) — the
  `{url}/v1/peers…` join cannot be spliced (unit-tested).
- **Plain-http multiplication: no.** `http://` never auto-learns; a
  hostile ad can't flood the fleet with its capture proxy. And learned
  https hosts go through normal cert validation — no bypass.
- **Slot exhaustion: contained.** Learned slots are capped at 16
  (configured entries protected first). Since the audit, a full learned
  list evicts FIFO (oldest first, pinned last-good exempt) instead of
  wedging shut — a flood can slow learning, never stop it permanently.
- **Stickiness capture: contained.** A hostile learned server that
  answers with a valid-but-EMPTY peer list does not keep the hoisted
  last-good slot (`clear_last_good` on empty) — a lying bootstrap can't
  privatize discovery's view of the mesh.
- **Signed-header forwarding: accepted.** Learned hosts receive the same
  signed fetch headers the primary does (signature covers only the fetch
  payload); no new capability, just exposure by design.
- **Availability (not security):** per-call failover is serial — each
  dead server costs up to 10 s per tick; survives, but a large hostile
  list in the rotation can stretch a tick. Delete
  `<data>/discovery-servers.json` to flush learned entries.

## Main-node runtime (LiteLLM) notes

The main node's runtime is the local LiteLLM proxy (`127.0.0.1:8000`,
unit `litellm-proxy.service`, config `~/.openclaw/litellm-config.json`),
fronting local ollama plus OpenRouter models. Two properties worth
remembering:

- **Auth: master key + virtual keys.** `general_settings.master_key` is
  set; since 2026-10-14 the proxy also runs against the co-hosted
  Postgres (`DATABASE_URL` in systemd drop-in
  `~/.config/systemd/user/litellm-proxy.service.d/override.conf`, role/db
  `litellm` on `127.0.0.1:5432`, prisma client generated against the
  bundled `litellm_proxy_extras/schema.prisma` and pushed once with
  `prisma db push`). Sub-clients now mint keys through the master key:
  `POST /key/generate` (alias/duration/max_budget) → use the returned
  `sk-…` as a normal bearer; `POST /key/delete` retires them. A request
  with NO bearer still gets 401. The master key itself only leaves the
  machine inside `~/.jerith-mesh/config.toml` (`runtime_api_key`, file
  chmod 0600, enforced at ensure/save since commit `8f0011b`). Losing it
  = re-read the master key from the config json; never guess.
  Gotchas, both hit live during setup: the drop-in rejected a comment
  with a non-ASCII dash (whole file silently ignored ⇒ env never
  arrived), and the boot-time "prisma package not found" message actually
  meant the bare `prisma` shellout couldn't find `~/.local/bin/prisma` —
  the drop-in now also exports a PATH that includes it. Schema drift
  after a litellm upgrade: rerun `prisma generate` + `prisma db push`
  with the new bundled schema before trusting /key endpoints.
- **num_ctx is explicit.** qwen3 is a hybrid thinking model; under
  ollama's default slot its prompt + thinking can outrun the window and
  produce HTTP-200-with-empty-content (observed live 2026-10-14 04:18:
  `n_ctx_slot = 4096, task.n_tokens = 2050` on every failing call, bot
  showed "runtime returned an empty answer"). The qwen3:8b entry now
  carries `num_ctx: 8192` in its `litellm_params`; raise it (RAM is 31 GB
  — 16 k fits comfortably) rather than debugging ghosts.
- **Runaway thinking + daemon guardrails (2026-10-14).** Two stall modes
  hit live the same morning:
  (a) `deepseek-r1:8b` loops forever on certain prompts — one short math
  riddle generated past a 6-minute streaming window (and a 120 s
  non-stream probe) without terminating; qwen3:8b answers the same
  prompt in <90 s (`done_reason: stop`). It was never a VRAM deadlock
  and not even a queue deadlock: the generation is simply unbounded
  without `options.num_predict`, and the point of failure is any client
  with a finite timeout whose next chat then waits behind the runaway.
  (b) With `OLLAMA_KEEP_ALIVE=-1` (this box's default), any second model
  a probe or benchmark loads *stays resident* — an 8b test model then
  fights qwen3 for VRAM on every next load, visible as 17/83 CPU–GPU
  splits and 30–80 s reload latencies on chats.
  Guardrails now in `/etc/systemd/system/ollama.service.d/maxload-queue.conf`:
  `OLLAMA_MAX_QUEUE=16` (further chats fail fast with server-busy
  instead of silently queueing behind a runaway) and
  `OLLAMA_MAX_LOADED_MODELS=1` (explicit serialization; a foreign model
  evicts qwen3 and must reload after). Restart ollama (models evict —
  one ~6 s cold reload) and re-verify with `ollama ps` (runtime env:
  `systemctl show ollama -p Environment`). Keep benchmarks on per-request
  `keep_alive` ≤ 5 m so a test model cannot camp in VRAM.
- **Bot chat timeout is a knob.** The old hard 90 s reqwest timeout died
  in live use twice (legit deep-thinking answers, 20–40 s reloads).
  `[chat] timeout_secs` in `~/.jerith-mesh/bot.toml` now sets it
  (default 240, clamped 60–600). Coherence note: the gateway bounds a
  non-stream local runtime at 180 s header-wait and then hands to mesh
  fallback, so >240 s buys you a slower fallback lane.
- **Bot-side streaming is live** (2026-10-14, default `[chat] stream =
  true`). The bot now sends `stream: true` and live-types the reply into
  Discord (both the @mention/DM fallback path AND `/ask`, same live sink:
  placeholder in the command's channel → throttled edits → one final
  formatted replacement) while the gateway passes SSE through CHUNK-BY-CHUNK — the old passthrough did
  `bytes().await`, silently buffering whole generations (fixed same day;
  verified live: first chunk 7.2 s into a 62 s generation, 876 chunks).
  Mechanics: OpenAI deltas fold per round into the same (status, body)
  shape the buffered path produces (tool-call fragments, finish_reason,
  usage chunk, mesh_fallback wrap included), so the tool/degenerate/retry
  logic stays single-source. Reality check: qwen3:8b thinking arrives as
  `delta.reasoning_content`, so live-typing shows the wait, then the
  answer tails in fast; a broken stream with real content already typed
  degrades to a cut-marked partial, never a lost  chat. `stream = false` restores the buffered one-shot.
  Edit cadence + thinking marker (2026-10-20, live-verified on the
  captured 369-chunk probe stream): the OLD ≥120-new-chars edit floor
  meant chat-length answers (content arrives as 1–2-char fragments —
  48 content chars total in the probe) NEVER live-edited — the
  placeholder sat frozen at `…` until finish() replaced it wholesale,
  exactly the "shows it all only at the end" report. The sink now
  edits every 2 s tick whenever the SHOWN text changed (first reveal
  ≤ 2 s after content starts; ~2.5 edits/5 s, safely under Discord's
  edit limiter), and the think phase shows an honest `🤔 …` marker
  (reasoning deltas outnumber content ~25:1 before an answer starts —
  the api layer taps `Thinking` on reasoning-only data rows and stops
  the moment content flows; tool-bearing rounds still type nothing).
  The mesh-fallback lane stays single-chunk by wire reality (a peer
  serves one non-streaming reply, wrapped — gateway-side), so mesh-
  served answers appear at once; this is the local-runtime lane's
  streaming, which is the lane in normal use. Tool rounds
  STREAM but never type — NARROWED 2026-10-20: only rounds AFTER an
  executed tool call fire no live events (the 2026-10-16 recall-slop
  incident's echo risk lives exactly there: web rows and PRIVATE
  embedded-memory hits exist in context only once a tool has run and
  fed results back). ROUND 0 — tools attached by the heuristic but
  nothing executed — has no tool results to echo, so it types live like
  the tools-never shape: the common casual chat (model answers
directly) streams instead of freezing at `…`. Surfaced tradeoffs: a
  web-informed answer round (results in context) lands via finish() in
  place within a second of readiness — never typed token-by-token; a
  round that streams prose AND a tool call can leave pre-call prose
  visible until finish() replaces it (the same shape the toolless path
  has always accepted). Companion same-day: the `<200 chars = web
  request` catch-all in the tool-attachment heuristic is GONE — it
  forced web tools onto every casual chat, which combined with the
  blanket suppression meant ZERO live typing on nearly every bot chat
  (the same-day "nothing changed" live report). Information  requests attach tools via factual openers + a trailing '?'
  (`search` rejoined the opener list); casual text does not.
  Tool-declaration WIRE names (2026-10-20, live-verified in the
  litellm/OpenRouter journal): tool names are validated by
  OpenAI-compatible providers against `[a-zA-Z0-9_-]+` and a dotted
  name (`memory.embed`) 400'd EVERY tool-attached chat through the
  router's remote legs — repeatedly, across its whole fallback list,
  before the chat reached a working leg at all (tools-attached answers
  sat at the placeholder every time, the third "not streaming" report
  shape). `wire_tool_name` now folds dots/colons to underscores at the
  declaration (`memory.embed` → `memory_embed`), with `dispatch`
  resolving the wire form back to the real skill — the model can only
  ever name the wire form, and every emitted def is charset-pinned by
  test. Toolless
  rounds type, but their fragments are JSON-gated: while the round's
  content still opens `{ "…` (the 193-char intent-slop landing round),
  fragments are HELD and never surface — the moment the content proves
  to be prose, the held burst fires and live-typing resumes (the
  "double response" incident: brace-slop typed live, RoundStart-cleared,
  and the retry's answer read as a second reply). Nothing a round yields
  becomes chat-visible before it is either real prose or the delivered
  final. The
  Discord edit wire path is verified live (say + two partial edits +
  final edit, all 200, content readback ok). Chat-guardrail state is
  observable: the node's `/status` carries `runtime_num_predict`, the
  site stats generator (site/generate_site_stats.sh, top-of-file TOML
  peek only — appending a key after the last `[section]` would land it
  section-scoped, not top-level) mirrors it plus the bot's `[chat]`
  stream flag into `data/stats.json`, and status.php renders both on the
  "Chat Guardrails" card.
- **Runaway-generation cap option** (`runtime_num_predict` in config.toml,
  default unset = off): when set, the gateway stamps `max_tokens = N` on
  every chat it forwards to the local runtime lane — but only when the
  caller set no cap of their own (max_tokens / max_completion_tokens /
  num_predict always win). NOTE (2026-10-20, live-verified at 17:16:
  the "didn't finish posting" report — the bot's answer stopped
  mid-sentence): the stamp is applied in the chat ENTRY handler, so it
  rides EVERY leg the gateway serves without a caller cap — the model
  router's rerouted remote tiers (litellm/OpenRouter) included — and a
  generation cut at the cap used to post silently truncated. The bot
  now surfaces `finish_reason: length` with an honest marker + a
  continuation affordance ("… say **continue** to have me carry on";
  the history carries the truncated turn, so a plain continue resumes).
  Truncation diagnostics: the bot journals `answer hit the token cap`
  at the cut; a stream MId-cut already ships with its own cut marker.
  This is the missing runtime-lane bound for
  thinking models that loop unbounded (deepseek-r1:8b, see the
  guardrails entry above): `MAX_QUEUE` fails fast, `MAX_LOADED_MODELS`
  serializes residency, `runtime_num_predict` bounds the generation
  itself. Set e.g. `runtime_num_predict = 1200` (~27 s of generation at
  ~45 t/s) and restart the node. Mesh-fallback lanes are the serving
  peers' own configs, not this node's. **Armed on the main node
  2026-10-14: `runtime_num_predict = 2000`** (~45 s of generation headroom
  at ~45 t/s; leaves room for qwen3's bash/code answers that the
  benchmark showed truncating at 1200), config backup at
  `~/.jerith-mesh/config.toml.bak-pre-captop`.

### Remote model endpoints + API keys (OpenRouter-free lane, 2026-10-16)

A mesh bot's node can already pull chat lanes from ANY remote
OpenAI-compatible endpoint — `runtime_url` + `runtime_api_key` in the
node's config.toml (`bearer_auth` is wired; Jerith Guard screens the
forward when the endpoint is not loopback). Nothing else is required to
run against a relay/hosted provider; what was missing is the recipe.

OpenRouter-free through the LiteLLM proxy (the recipe + today's gotchas):

- **Model entries**: `model_list` rows with
  `"model": "openrouter/<vendor>/<name>:free"` under an alias (the
  proxy's list already carries ten of these). The node routes a lane at
  the ALIAS — e.g. `runtime_deep_model = "ling-3.1-flash"` (armed
  2026-10-16, config backup `config.toml.bak-openrouter-deep`).
- **Provider key delivery — use an EnvironmentFile**: this LiteLLM build
  does NOT resolve provider auth from the config file's
  `environment_variables` section despite it holding the key (observed:
  provider 401 until the key reached the process). Working wiring: the
  key line in `~/.openclaw/litellm.env` (chmod 600) + a user-unit drop-in
  `~/.config/systemd/user/litellm-proxy.service.d/openrouter-env.conf`
  with `EnvironmentFile=` + `daemon-reload` + restart. Inside
  `litellm_params`, either a literal api_key or LiteLLM's documented
  "os.environ/OPENROUTER_API_KEY" reference now both resolve.
- **Proxy key gate — send the Bearer**: when the proxy runs with key
  management armed (its Postgres/user_api_key_auth), chat requests
  without `Authorization: Bearer <proxy key>` die at the PROXY with
  401 "No api key passed in" before any model call (masked as an
  OpenRouter auth error — same message shape, wrong layer). The node's
  `runtime_api_key` is the proxy key and the gateway sends it
  automatically; probe scripts must include it too.
- **Verified free lanes (PONG probes, 2026-10-16)**: `ling-3.1-flash` ✅,
  `laguna-m` ✅ — both answer text through the proxy; `inkling` 403
  (agentic-harness-only); `or-free` / `north-mini-code` /
  `nemotron-3-super` return 200 with empty content at small
  `max_tokens` caps (think-swallow — only usable with 300+ tokens);
  local `qwen3:8b` stays the tool tier because free lanes are
  rate-limited and a multi-round tool loop would hammer the cap.
- **End-to-end proof**: a deep-tier-triggering prompt through the node
  gateway rerouted `to=ling-3.1-flash` (node journal) and returned a
  full prose answer (684 usage tokens) from OpenRouter.

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
- ~~macOS CI flake family (test-side timing, 2026-10-20/21)~~ FIXED:
  three test-side races, each cause-repaired, no assertions weakened:
  `quic` fuzz reply bound 3→8 s (a 150-frame storm legitimately keeps
  the server busy; same loaded-runner math as its own redial bound,
  which went 5→10 s), `bootstrap_server` TTL test fresh-read window
  50→300 ms (the read raced axum serve startup inside the TTL), and
  `lore` vault dir was shared per-pid so parallel test threads rewrote
  each other's files mid-read (unique temp dir per call now).

Round 5 (2026-10-23 security sweep) — webinbox + parse-layer hardening
(`db4dd3a` + `241a70b`):

- **GGUF import poisoner** (`db4dd3a`): a crafted tensor-directory
  offset wrapped `data_start + offset` past 2⁶⁴ in RELEASE builds, the
  EOF gate then passed, and the file's header bytes decoded as floats
  into a "valid" expert written to the live store AS A SUCCESS
  (verified live; debug builds panicked instead). Every unchecked u64
  on that attack surface (offset add, `at+len`, block/byte sizes, dims
  product) is now a `checked_*` → typed refusal, with a regression
  test holding both the wrap-offset and dims-overflow shapes.
- **WhatsApp webhook replay** (`241a70b`): Meta's `X-Hub-Signature-256`
  covers the body ONLY — no timestamp, unlike Slack — so a captured
  legitimate POST re-verified forever and each replay spawned a
  session spending the bot token on a real channel. A bounded
  8192-entry seen-id cache now silently acks repeats before any
  session spawns; messages without an id are passed through with a
  loud warn (the id is the dedupe key, and Meta always sends one).
- **Two `Box::leak` warts** (`241a70b`): the Slack `url_verification`
  challenge and the message chunker's per-boundary remainder are
  returned/owned now — both used to free nothing for the life of the
  long-lived bot process.

### Trust-on-first-use: learned bootstrap keys (accepted, understood)

The §10 reputation chain pins a bootstrap's signing key TOFU-style:
`pin_key(served_by, &key.pubkey)` on the first answering contact per
configured/learned URL (`ping.rs::pin_and_ingest`), a rotation
replacing the pin with a journal note. The first contact with a
bootstrap we did NOT explicitly configure is therefore an unverified
distance-1 event: if an attacker both advertises a hostile
`discovery_servers` URL inside a peer ad AND intercepts our first
dial to it (DNS/HTTP MITM on that URL), the MITM's key becomes our
"pin" for that URL.

The Exploit CEILING is deliberately small and does not grow with
time:

- Gossip stamps arriving by peer ad refuse any origin we have not
  pinned, refuse the ad owner naming itself, and are freshness-
  capped — so the poison stays ONE bootstrap's reputation view
  (subjective scores), never code, wallet, or auth state.
- A lying bootstrap cannot sticky-hold discovery either: an empty
  answer clears the last-good pin (ping.rs "answered alone does not
  earn stickiness"), so the next tick rotates to the config order.
- Everything announced is auditable: every first sight or rotation
  journals `mesh: bootstrap signing key pinned (reputation stamps
  verifiable from here)` naming the URL and key.

Operator rule (how to close it): run every bootstrap you plan to
trust in `[bootstraps]` config so the first contact is your
infrastructure, not a learned URL. Learned servers are fine for
failover rotation once their PIN entry is in the journal and looks
right — check it with
`journalctl --user -u jerith-node | grep 'signing key pinned'`.
Re-pin (rotation) after a compromised key is one restart of the
node against the rotated bootstrap; the old stamp lineage dies with
the freshness cap.



## Escalation pointers

- Tunnel misroute (CNAME pointed at the wrong tunnel UUID): see the
  gotcha in `DEPLOY-BOOTSTRAP.md` — route by UUID explicitly.
- A node demoted/degraded: check `/status` `gate_clear` (stake escrow
  short?), bootstrap reachability, and `journalctl --user -u jerith-node`.
- Fraud-correctness: `verify_percent`/ringer counters live in `/status`
  (`meters`, `verify` fields); a spike in mismatches precedes scam/round
  slashing notices — check `GET /stake` on the chain node for who burned what.

## CI runner image policy (ubuntu-24.04 pin)

Effective `ebe542d` (2026-10-16): every Linux leg in `ci.yml` and
`release.yml` pins `ubuntu-24.04` explicitly; macOS stays `macos-latest`.
Rationale: GitHub migrates the `ubuntu-latest` label to Ubuntu 26.04
over **2026-10-19 → 2026-11-19** (actions/runner-images#14748), and a
silent environment flip mid-push is exactly the failure mode the gates
exist to catch — so the environment changes only when we change it.

Evaluation of moving to `ubuntu-26.04` (2026-10-16): *no dependency on
the newer image exists* — Rust arrives via `dtolnay/rust-toolchain`
(1.98.1 on both images per the announcement), `rusqlite` is bundled,
`cargo-audit` builds from source, and the E2E uses stock `python3`/shell.
The real deltas are base-image only (kernel 6.17 → 7.0, systemd
255 → 259) — zero expected benefit for these suites, concrete risk in
the first weeks of a fresh image (our three live-infra gates are the
most environment-sensitive jobs in the fleet).

Plan when the migration window closes (after 2026-11-19): open a branch
that flips both pins to `ubuntu-26.04`, soak the full matrix on it for
a week, then land explicitly. Until then `ubuntu-24.04` is deliberate,
not inertia.

## Dependency security posture (cargo audit, 2026-10-15)

### Feature-unification audit (TLS/HTTP stack, 2026-10-15)

A sweep for the hazard class the serenity stopgap evaluation measured
(one stray `default-features`/backend feature flips the shared stack's
default silently). Verdict: **no unknown leaks**. Facts, all verified
with `cargo tree -e features`:

- **TLS backends present:** rustls only. `native-tls`/`openssl`/
  `aws-lc-rs` are absent from the tree; the vulnerable webpki 0.102.8
  sits solely in the documented serenity websocket subgraph (the six
  accepted advisories below).
- **reqwest feature set:** `json`, `stream` (our manifest pin, shared
  by all three crates) + `multipart` — enabled by *our own* crates for
  mesh file uploads — plus the internal `__rustls`/`rustls-tls`
  webpki-roots aliases. No `default-tls` carrier anywhere.
- **rustls feature set (0.23.46):** `ring` + `std` + `tls12` from our
  pin, plus `webpki-roots`/`webpki-tokio`/`http1` linked by
  quinn-proto and reqwest. No second crypto provider: trimming our rustl
  `default-features` could invite `aws_lc_rs` from an uncareful future
  dep, so the pinned `features = ["ring", "std", "tls12"]` stays.
- **quinn defaults (by design):** `default`, `ring`,
  `platform-verifier`, `runtime-tokio` — the platform-verifier feature
  is what pulls `rustls-platform-verifier`; that is the QUIC mesh's
  cert-verification path, not an accidental leak.
- **Manifests:** no crate-local dependency spec leaves TLS-relevant
  `default-features` on beyond the workspace-level pins documented
  above; serenity-as-direct-dep does not exist on `main` (the
  evaluation branch carries it, unmerged).

Repeat recipe for the next sweep (the commands that produced the
facts): `cargo tree -i native-tls`, `cargo tree -i openssl`,
`cargo tree -i aws-lc-rs`, `cargo tree -i rustls-webpki@0.102.8`,
`cargo tree -e features -i reqwest`, `cargo tree -e features -i
rustls@<current>`, `cargo tree -e features -i quinn`, plus a
manifest grep for spec lines without explicit `default-features`
intent. The hazard to re-check whenever a new dependency arrives: does
it (or its default features) enable `reqwest/native-tls`,
`tokio-tungstenite/native-tls`, or a second rustls crypto provider?

`cargo audit` runs itself now — no manual pre-release run to remember.
It is wired into CI twice: on every push/PR
(`ci.yml` → `dependency-audit`, advisory tripwire so a new RUSTSEC
advisory surfaces the moment it lands) and as a blocking gate on the
release workflow (`release.yml` → `dependency-audit`, which must pass
before any tag builds or ships binaries — evaluated at the moment of
the release cut, not at the moment of the last push that may be weeks
older). The accepted serenity-websocket advisories remain the only
exceptions, via `.cargo/audit.toml`. Current state after this pass:

- **Pipeline live drill (Phase 3 gate, 2026-10-15):**
  `cargo test -p jerith-mesh-core --test pipeline_remote_drill` spins a
  mesh in-process — real stores, real QUIC — and drills the M0–M3
  ladder end-to-end: hop owners boot with empty stores and harvest
  their expert containers over the wire (chunked `ExpertFetch` +
  checksummed ingest, byte-identity proven, and a deliberately
  corrupted replica rejected at the ingest checksum), then route over
  the transport, wire commitments with pinned attribution, a hung
  owner dying at its deadline and ring-rule re-hosted, honest
  partials, priced budgets — including a simulated wide-area hop
  (180 ms `rtt_of`) whose budget is floored on `base + 3×RTT`. Pass
  runs in under a second; repeated green 5/5. The drill AND the full
  clippy `--all-targets -D warnings` gate both run in CI on every push
  (`.github/workflows/ci.yml`, jobs `pipeline-drill` and
  `clippy-all-targets`), so branch work lands on them pre-merged; the
  manual run still matters locally before touching the pipeline,
  transport, stores, or budget pricing.
- **Fixed:** `rustls 0.23.45 → 0.23.46` (targeted `cargo update
  --precise`). Clears four advisories on the core QUIC/reqwest serving
  path — most importantly RUSTSEC-2026-0346 (medium), where `Debug`
  impls can print TLS session secrets into logs. The workspace compiles
  and the full suite is green on 0.23.46.
- **Accepted for now (7 advisories, all one subgraph):** `rustls
  0.22.4` + `rustls-webpki 0.102.8`, pulled *only* by `serenity 0.12.5`
  → `tokio-tungstenite 0.21` (the bot's Discord websocket). The fixes
  live in the 0.23/0.103 lines, so clearing them means bumping
  serenity/poise/tokio-tungstenite majors — a deliberate migration task,
  not a drive-by. Exposure is limited: those advisories concern
  websocket handshake downgrade/cert-verification details on the bot's
  TLS connection to Discord (a single well-known endpoint over an
  upstream-verified channel), not the mesh's own QUIC mesh traffic,
  which rides rustls 0.23.46. Revisit when serenity 0.13/poise lands.
  The 2026-10-20 additions landed the SAME day they were published,
  in two halves each:
  - RUSTSEC-2026-0382 (webpki cert verification accepts distrusted
    intermediates): the main rustls line's webpki pin was `cargo
    update`d `0.103.15 → 0.103.16` (patched, done — no gate
    relaxation); the acceptance covers ONLY serenity's webpki
    `0.102.8` pin, which has no patched release in its own line.
  - RUSTSEC-2026-0381 (bundled roots trust distrusted emSign
    intermediates): patched EVERYWHERE the tree controls — the 1.0
    pins `cargo update`d to `1.0.10` AND the bot crate's own direct
    `webpki-roots 0.26 → 1.0` dependency bump (the IRC/Slack TLS leg
    compiles unchanged on 1.x — `TLS_SERVER_ROOTS` kept its shape);
    the acceptance covers ONLY serenity's tokio-tungstenite-pinned
    `0.26.11`, no patched release in that line. Same exposure logic
    as the rest of the subgraph: Discord websocket TLS only.

### Voice-lane block (songbird 0.6.1 subtree, 2026-10-21)

The Discord voice lane (§ «Discord voice lane») brought songbird 0.6.1
and with it nine advisories, accepted the day they tripped the CI gate:

- **Eight libcrux-* (aesgcm/chacha20poly1305/kem/secrets/sha3, 0209
  0211 0124 0330 0331 0212 0207 0208):** all inside `davey`, the
  Discord DAVE end-to-end-encryption MLS stack songbird links
  UNCONDITIONALLY (verified: `cargo tree -e features -i davey` — no
  feature removes it). The shipped voice config never turns MLS/DAVE
  on — plain transport encryption only — so davey is dormant linked
  code, and none of those code paths (HPKE encapsulation, MLS
  handshake, SHAKE squeeze) execute. The fixes are 0.0.x renames
  (`libcrux-aesgcm → libcrux-aes`) and ≥0.0.10 jumps blocked by
  openmls_rust_crypto's pinned `hpke-rs 0.6` — unreachable without an
  upstream songbird release; revisit then.
- **ringbuf 0.4.8 (RUSTSEC-2026-0293):** double-free requires an
  element whose `Drop` panics while `Consumer::skip`/`clear` runs; the
  bot's rings hold pods (`[u8; 16]`, `[f32; 2]`, no Drop), so the
  precondition is unreachable in our code. Fix blocked: songbird pins
  the 0.4 API.

Both facts were checked against the resolved tree, not assumed. The
`.cargo/audit.toml` voice-lane block holds the entries with the same
revisit trigger (next songbird release): delete it then.

Verified upstream facts (2026-10-15): **serenity 0.13 does not exist
yet** — 0.12.5 (2025-12-20) is the newest release and it pins
`tokio-tungstenite 0.21` (hence rustls 0.22 + webpki 0.102.8).
tokio-tungstenite itself moved on (0.27+ by mid-2025), but serenity's
semver demand pins the old line, so a solo bump is impossible.

Three paths, in preference order:

1. **Wait for serenity 0.13 + poise 0.8** (default). Prepare now so the
   bump is a day's work when it lands: keep the bot's serenity/poise
   API surface concentrated in `crates/bot/src/{main,commands,api}.rs`
   (already true); do NOT reach into serenity internals from other
   files. Upstream snapshot (checked 2026-10-15, no committed date):
   serenity has NO 0.13 milestone and no milestone carries a due date,
   and the 0.12 cadence shows majors shipping whenever ready (0.12.0
   2023-11 → 0.12.4 2024-11 → 0.12.5 2025-12). The concrete thing to
   watch instead is **serenity PR #3599** (open, 2026-08-24):
   tokio-tungstenite 0.21 → 0.23.1 — if merged into a serenity
   release, the websocket stack lands on rustls 0.23 even without a
   0.13, and this milestone's exit criterion below can be met on a
   serenity patch-line bump. When the day comes: bump `serenity` +
   `poise` + `tokio-tungstenite` together, fix CompileError-level API
   breaks (framework builder, context/gateway plumbing, `EditMessage`
   builder changes), then `cargo tree -i rustls` MUST report only the
   0.23 line and `cargo audit` must come back clean — that is the exit
   criterion, after which the six `ignore` entries in
   `.cargo/audit.toml` are deleted (their comments say exactly that).
2. **Stopgap: flip serenity to `native_tls_backend`** (only if the
   risk calculus changes — not the current default). Serenity 0.12.5
   exposes `default_native_tls` / `native_tls_backend`, which moves
   tokio-tungstenite + reqwest onto native-tls (OpenSSL) and drops
   rustls 0.22.4 + webpki 0.102.8 from the tree entirely — clearing
   today's six advisories WITHOUT waiting for 0.13. It is NOT adopted
   by default because (a) it trades rustls for OpenSSL (a new linking
   and trust-store surface on the bot), and (b) feature unification:
   serenity enabling `reqwest/native-tls` would ALSO enable it for the
   shared reqwest in core/node — unless our manifests pin
   `reqwest = { default-features = false, features = ["rustls-tls"] }`
   first. If adopted, do the manifest pinning BEFORE the backend flip
   and verify the mesh's own HTTP clients still ride rustls via
   `cargo tree -i reqwest -e features` and a smoke `/status` probe.
3. **Rejected: git-dependency on serenity/poise master.** Pins an
   upstream moving target into the supply chain to chase one advisory
   set; not worth it while exposure is limited (see above).

Measured on the evaluation branch `stopgap-native-tls` (never merged):
the native_tls_backend stopgap DOES work — audit goes to zero — but the
flip exposed two durable facts worth knowing on migration day. (a)
poise 0.7's `default` feature is what bundles serenity's
`rustls_backend`; leaving `poise` default-features on makes ANY serenity
backend flip a no-op. (b) serenity's native_tls_backend also enables
`reqwest/native-tls` on the workspace's shared reqwest — and with
default-tls + rustls-tls both compiled in, reqwest's default backend
becomes OpenSSL, and 33 unpinned client-construction sites across the
crates would silently switch the mesh's HTTP path. Adopting that
stopgap therefore requires `.use_rustls_tls()` at every site (recipe
and trade-offs in the branch commit) — the reason it stays an emergency
path and the primary answer remains serenity 0.13.

Interim guardrail (done this pass): `cargo audit` now gates CI
(`.github/workflows/ci.yml`, `dependency-audit` job) against every
advisory EXCEPT the six documented here — `.cargo/audit.toml` ignores
exactly those IDs, each with a pointer to this runbook section. A NEW
advisory anywhere in any crate's dependency graph breaks the build;
the accepted six cannot silently grow. Maintenance rule: whenever a new
advisory is accepted-with-reason, it gets a runbook paragraph and an
audit.toml entry in the same commit — never one without the other.

## Telegram approver channel (approval bridge, live 2026-10-19)

The `[approvals]` table in `~/.jerith-mesh/bot.toml` (file is 0600) now
carries a live Telegram pair — pending approval cards fan out to the
owner's chat with an inline ✔ Approve / ✕ Deny keyboard, and **only that
chat id may resolve**. Verified that day: `getMe` ok (@Jerithbot), a real
card delivered via `sendMessage`, poller long-polling clean in the
journal (it is silent while armed — it only logs on dispatch failures,
so absence of `telegram approval notice failed` lines is the healthy
signature).

```bash
# Health: config pair present + service active + no dispatch errors
grep -A3 '\[approvals\]' ~/.jerith-mesh/bot.toml
systemctl --user is-active jerith-bot
journalctl --user -u jerith-bot --since "1h ago" | grep -ic 'telegram approval notice failed'

# Token rotation (creds are in bot.toml, keep the file 0600):
edit ~/.jerith-mesh/bot.toml   # replace telegram_bot_token
systemctl --user restart jerith-bot

# Store checkpoint state (one 0600 JSON per request):
ls ~/.jerith-mesh/bot/approvals/        # <id>.json, status per file
# Sweeper expires pending cards after [approvals] ttl_minutes (default 60).
```

Round-trip drill (maintenance only): `FEATURES.md §15` documents the
`#[ignore]`d `approval::tests::live_telegram_roundtrip` harness — fires
one REAL card (persisted into this real store, Telegram chat gets it,
the running bot resolves the ✔ press and executes a harmless deferred
`memory.embed`), then print the id. Never run it from a temp-config
suite: the card must persist in this store and resolve in the live bot.

## External chat transports (12 platforms; Telegram live-fire 2026-10-22)

`crates/bot/src/transports/` ships adapters for IRC, Slack (Socket
Mode), Twitch, Mattermost, Signal (signal-cli daemon), Nextcloud Talk,
MS Teams, Google Chat, Feishu/Lark, WhatsApp Cloud, Telegram, and GG.
ALL OTHERS REMAIN
DORMANT: a platform runs only if its `[transports.<platform>]` table
exists in bot.toml — no table, no task, zero cost (so the bot that
ships today behaves exactly as before). See FEATURES §18 for the
per-platform inbound/outbound surface and `transports/<platform>.rs`
module doc for the setup contract (tokens, endpoints, caps).

Single-consumer rule (Telegram, learned live 2026-10-22): getUpdates
admits exactly one poller per bot token. The approvals lane's
`[approvals] telegram_bot_token` and the transport's
`[transports.telegram].token` MUST NOT be two different bots polling no
one — when they hold the SAME token, `approval::transport_owns_token`
gates the standalone approval poller off and approval buttons/commands
ride the transport's stream (`handle_transport_update`); on a different
approvals token both lanes poll their own bots independently. A 409
from getUpdates is a typed config-fatal ("another consumer holds this
token") — it means exactly one thing: two pollers, usually the bot
against itself.

```bash
# Arm a platform: add its table + restart (fail-loud errors explain gaps)
edit ~/.jerith-mesh/bot.toml   # e.g. [transports.irc] server/channel/nick
systemctl --user restart jerith-bot

# Health: task is running + which platforms came up
journalctl --user -u jerith-bot | grep -E 'transports starting|transport '

# The webhook in-box (Teams / Google Chat / WhatsApp / Slack Events):
# [transports.webinbox] bind = "127.0.0.1:5543" (+ slack_signing_secret,
# allow_unverified flags; front the public side with your TLS proxy).
# WhatsApp verification needs [transports.whatsapp] app_secret (fail-closed).
```

Token hygiene: platform tokens live under `[transports.*]` in bot.toml
(0600); rotate by edit + restart, same as the Discord lane. All
platforms write their logs under the shared tracing names
(`transport = "<platform>"`), and a task that dies repeatedly logs
`transport failed permanently` only on a config-level fatal — service
blips fan back internally with backoff.

## Author a plugin (Phase 3d SDK, shipped 2026-10-22)

The community path needs no bot recompile: a plugins.d drop + a running
skill service. Two SDK shapes, ONE wire contract (signed dispatch —
`x-jerith-signature` = hex(HMAC-SHA256(secret, "{ts_ms}.{body}"))):

- **Rust**: `crates/plugin-template/` — a library (`SkillServer`) that
  IS the service (signature gate, ±5-min freshness, bounded responses,
  typed errors), plus the `jerith-plugin-example` binary with two demo
  skills. Copy `main.rs`, swap the handlers, keep `plugin.toml` in sync.
- **Python (3.11+)**: `sdk/python/jerith_plugin.py` — one stdlib-only
  file; `Plugin.from_manifest(...)` + `@p.skill("name")` decorators is
  the whole surface. `example_plugin.py` is the same two demo skills.

```bash
# The five-minute lane (example plugin, both SDKs verified live):
mkdir -p ~/.jerith-mesh/plugins.d/example
cp crates/plugin-template/plugin.toml ~/.jerith-mesh/plugins.d/example/
cargo run -p jerith-plugin-template --bin jerith-plugin-example &
#    (or: python3 sdk/python/example_plugin.py \
#          ~/.jerith-mesh/plugins.d/example/plugin.toml &)
edit ~/.jerith-mesh/bot.toml        # add "external:example" to [plugins] enabled
systemctl --user restart jerith-bot
```

Health: a dropped manifest with NO running endpoint costs nothing until
invoked (then the model reads the typed timeout error); a malformed
drop fails bot startup loud — the drop parser is fail-closed. The
shipped template manifest is pinned by a bot test
(`shipped_template_manifest_parses_under_todays_rules`), so template
and parser cannot drift silently. Reference plugin in production:
`crates/gcp-bridge` (the same contract at fleet scale).

A lived-in second consumer to crib from beyond the demos — **the mesh
pet**, `crates/pet-plugin/` (2026-10-23): a persistent void-companion
over the same signed `/rpc` contract. Five skills (`pet.look / talk /
feed / play / rename`), one small JSON state file whose experience
decays in real time between calls and survives restarts. The manifest
discipline is the same as above — and its own tests pin the other
half (declared rows = registered handlers). The five-min
lane in its copyable manifest comment is identical to the example's:

```bash
mkdir -p ~/.jerith-mesh/plugins.d/pet
cp crates/pet-plugin/plugin.toml ~/.jerith-mesh/plugins.d/pet/
cargo run -p jerith-pet-plugin --release --bin jerith-pet &
edit ~/.jerith-mesh/bot.toml        # add "external:pet" to [plugins] enabled — or to
                                    # [plugins] owner_only (like gcp: owner-DM skills only)
systemctl --user restart jerith-bot # then just talk to it in chat
```

Live-fire (2026-10-23, this house): drop installed with a rotated
secret, `external:pet` joined `owner_only` (the owner-DM posture — a
companion gcp already has), service on `127.0.0.1:7789`, unit
restarted clean — a malformed drop aborts the start, so the clean boot
IS the registry proof. Wire-pin: unsigned probe → 403; HMAC-signed
probe with the manifest's secret → 200 + a full card (`stage:
droplet, mood: orange-cap`, born same hour, one line in the pet's own
voice). The daily briefing greets the owner with that card when the
companion is opted in — see FEATURES §24.

History note for SDK authors: the pet build caught a latent boot bug
in the example's `read_manifest` — `"…".parse::<toml::Value>()` only
accepts ONE value, so a whole manifest document failed "unexpected
content, expected nothing". Document parsing is
`toml::from_str::<toml::Table>`; both crates' main.rs now do that and
pin their shipped manifests by test (2026-10-23).

## The house card (third live consumer, 2026-10-23)

`crates/house-plugin/` answers the owner's "how are you doing" over
the same signed contract as pet: `house.status` = one card (bot unit
state, gateway health, mesh contribution %, recent warns, one word);
`house.journal` = the bot's recent journal lines (5–1440 min window).
Gateway/unreachability degrade to card fields — a speaking card is
worth more than a refused one. Eponymous manifest rules apply, and
its shipped-manifest bot-side pin test is third in the row:

```bash
mkdir -p ~/.jerith-mesh/plugins.d/house
cp crates/house-plugin/plugin.toml ~/.jerith-mesh/plugins.d/house/
cargo build -p jerith-house-plugin --release
systemd-run --user --unit=jerith-house-plugin --working-directory=~/.jerith-mesh/plugins.d/house \
  ./target/release/jerith-house ~/.jerith-mesh/plugins.d/house/plugin.toml   # rotate the secret first
edit ~/.jerith-mesh/bot.toml        # "external:house" → [plugins] owner_only (owner-DM posture)
systemctl --user restart jerith-bot
```

Live-fire (2026-10-23, this house): drop installed with a rotated
secret, `external:house` joined `owner_only` with both prior
consumers, service on `127.0.0.1:7790` (loopback, secret on) via unit
`jerith-house-plugin`, bot restarted clean. Unsigned probe → 403
refusal; HMAC-signed probe (`house.status`) → 200 + a full card:
`word: steady`, bot active, 1 peer alive, contribution 40%, zero
recent warns. Restarting the bot had the drop picked up without
house-side changes.

### The pet's diary lane (into the [ai] mailbox, 2026-10-23)

The pet archives its own life now: a **stage landing** (`droplet →
glowcap → …`) or a **long silent drift** (two or more integrated
days between sightings) is queued into the state file and delivered
as ONE .eml per entry into `~/.jerith-mail/inbox-archive/ai/`
(`$JERITH_MAIL_ARCHIVE` overrides the base) — the same box the
mailhook writes, readable with `/mail ai`:

```
From: Droplet <pet@jerith.mesh>
To: ai <ai@jerith.mesh>
Subject: [pet diary] stage: glowcap
Date: Fri, 23 Oct 2026 08:11:22 +0000
X-Pet-Kind: stage_up
X-Pet-Stage: glowcap

The droplet holds its color today — orange, holding orange, like sunrise does.
```

Filename stamps carry the EVENT time (`YYYYMMDD-HHMMSS-pet-<kind>-
<rand>.eml`, the mailhook's shape), so the archive sort reads the
pet's life in order even when delivery lagged the event. Delivery is
in the handler's serving path: mail written BEFORE the save, fails
requeued into the state for the next sighting — a mail is sent once
or not lost. Live-fire (2026-10-23): a probe pet (temp state, temp
archive, unsigned dev manifest on :7799) crossed a stage by talk →
card said `stage: glowcap, stage_up: true`, and BOTH the stage mail
and an eight-day-drift mail landed in the [ai] box with clean
headers; then the real unit restarted on the diary binary and its
signed `pet.look` card came back unchanged. The real pet's own
diary starts writing with its next landing or long silence.

### The handoff note lane (agent sign-offs, 2026-10-23)

Before an agent signs off (a working session, an upgrade, a repair),
it writes ONE structured mail into the `[ai]` box and the daily
briefing surfaces the newest one while fresh (today/yesterday by
archive stamp). The writer is stdlib-only:

```bash
scripts/write-handoff.py --by "<agent name>" --line "<one-line what>" <<'BODY'
What shipped
: one line per landed thing, each with the WHY
Left live / wounds
: services started, pins changed, what the next session must know
Next steps
: one sentence per follow-up
BODY
```

The mail lands as `YYYYMMDD-HHMMSS-handoff-<rand>.eml` (mailhook
stamp shape, local wall; `Date:` is UTC RFC-5322), subject tagged
`[handoff]`, and sections say exactly:

- **What shipped** — what landed this session;
- **Left live / wounds** — what is running or raw right now;
- **Next steps** — the follow-ups, one line each.

Briefing render (`briefing.rs handoff_section_in`): fresh →
`🤝 handoff (<stamp>) <subject>` + the first three body lines;
stale → one quiet pointer line (`no fresh handoff — last signed off
…`); empty box → the section simply isn't there (same empty-pew rule
as the pet's greeting). Live-fire (2026-10-23): the session that BUILT
the lane signed itself off through the script into the real [ai] box
(`20261023-032353-handoff-1c28.eml`) — the first mail the surface
ever had to read.

## The download-quarantine lane (fourth live consumer, 2026-10-23)

`crates/quarantine-plugin/` watches the droppable files lane: new
files in the watch dir (`$JERITH_QUARANTINE_DIR`, default
`~/Downloads`) get the BetterCraft treatment — real unzip (stored +
deflate), marker extraction from decoded class pools, typed
benign/suspicious/malicious verdicts, SHA-256 on every scan — with
verdicts appended to ONE JSONL log
(`~/.jerith-mesh/quarantine/verdicts.jsonl`; `$JERITH_QUARANTINE_STATE`
override — keep bot.toml-adjacent overrides in lockstep on BOTH
sides). The bot reads that log two ways: the monitor's
`check_quarantine` DMs the owner when a MALICIOUS verdict's file is
STILL on disk (deleted threats stay quiet — old news is not news), and
the daily briefing carries a quarantined `🧯` line when
`external:quarantine` is opted in per `[plugins]` (owner_only row,
like gcp/pet/house).

```bash
mkdir -p ~/.jerith-mesh/plugins.d/quarantine
cp crates/quarantine-plugin/plugin.toml ~/.jerith-mesh/plugins.d/quarantine/
cargo build -p jerith-quarantine-plugin --release
systemctl --user enable --now jerith-quarantine-plugin   # deploy/systemd/ unit; rotate the manifest secret first
edit ~/.jerith-mesh/bot.toml        # "external:quarantine" → [plugins] owner_only (owner-DM posture)
systemctl --user restart jerith-bot
```

Live-fire (2026-10-23, this house): drop installed with a rotated
secret, `external:quarantine` joined `owner_only` beside all prior
consumers, service on `127.0.0.1:7791` (loopback, secret on), unit
restarted clean — a malformed drop aborts the start, so the clean boot
IS the registry proof. Wire-pin: unsigned probe → 403; HMAC-signed
`quarantine.recent` → 200 + card (the opening tick had already judged
`image.jpg` benign). End-to-end: a synthetic dropper-shaped jar (ZKM +
ProcessBuilder + HttpURLConnection + setExecutable classes, deflated,
real zip writer) landed in ~/Downloads and was judged `malicious`
within one poll, the row naming all four marker kinds with its sha;
deleting the file kept the monitor quiet per the standing rule, and
the log keeps the verdict for the record either way.

Operator rules of the lane:

- a MALICIOUS verdict never asks ownership first — the monitor DMs,
  the owner deletes; the scanner never deletes or moves anything
  (reporter only, the monitor's privilege boundary);
- suspicious is suspicion, not a verdict death sentence — the model
  can read the findings and elaborate; the rules table is deliberate
  about network-only jars staying benign (update checkers are the
  norm; the lane never cries wolf);
- the plugin's skills (`quarantine.scan <path?>`,
  `quarantine.recent`) are the chat's re-run: any file, any time,
  verdict card back — the same signed contract as every other drop.

**Reporting a new sample to other recipients** (owner decides when —
this lane never publishes anything on its own):

1. Pull the conviction row straight from the log, not from memory:
   `python3 -c "import json,sys; [print(json.dumps(json.loads(l))) for l in open('/home/noodly/.jerith-mesh/quarantine/verdicts.jsonl') if '<SHA256-PREFIX>' in l or '<filename>' in l]"`
   — one JSON object per hit: path (note the exact filename),
   sha256, verdict, the marks list, size, both timestamps.
2. Attach the sample file itself only to a channel that accepts
   malware for analysis; anything else gets the verdict row + the
   filename + where it was seen, never a live copy.
3. Sender-friendly one-liner others can act on: "BetterCraft
   BetterSurvival-1.21.1.zip from CurseForge reached us; sha256
   <hash>; verdict malicious, marks <list>."
4. If the owner is closing the loop on a live compromise too: keep the
   file, grep downloads for the same hash on other machines, keep the
   row, and check back after the next briefing for a repeat opinion.

## Task ledger / intentions lane (LIVE 2026-10-23, crates/ledger-plugin)

The fifth plugin drop (port 7792, signed contract, one JSON state
file at `$JERITH_LEDGER_STATE` default
`~/.jerith-mesh/ledger/state.json`). Goals the owner voices in chat
become `ledger.add` entries; the daily briefing stands them back up
(`📋` line) so nothing said in scrollback evaporates overnight.
Skills: `ledger.add` / `ledger.list` / `ledger.done <id>` /
`ledger.drop <id>` — ids only come from `list`.

Install (standing unit, permanent this time):

    cp deploy/systemd/jerith-ledger-plugin.service ~/.config/systemd/user/
    cp crates/ledger-plugin/plugin.toml ~/.jerith-mesh/plugins.d/ledger/
    # rotate the manifest's secret first (openssl rand -hex 16)
    systemctl --user daemon-reload && systemctl --user enable --now jerith-ledger-plugin
    # then add "external:ledger" to [plugins] owner_only in bot.toml + restart jerith-bot

Survival notes from install day (same shape as the house/quarantine
drops, so the old gremlin rules hold verbatim):

- **the service carries no heartbeat** — `is-active` and one signed
  `ledger.list` probe (the 403-goes-200 recipe in this section's
  quarantine sibling) are the health pair; the lane only moves when
  asked;
- **entry ids are stable across restarts but renumber nothing** —
  `drop` on an unknown id is a typed error naming the miss, and there
  is no wildcard; if a goal must be amended, `done` or `drop` the
  old and `add` the new wording;
- **closed history is the record** — a `done` entry keeps its closed
  timestamp; the archive cap (500) trims the oldest CLOSED rows only;
  never edit `state.json` by hand while the service is up (atomic
  saves mean the plugin wins);
- **full restart, this time on the permanent unit**: enabled and
  started explicitly (no systemd-run temporary this time), the bot
  restarted clean with the gate row, and the live verified round
  (unsigned 403, signed add→#1, done→#2, drop→#1-unknown-#3-error,
  list card matching the briefing's pinned keys) holds the boilerplate
  for relaucher installs.

## MongoDB Atlas lane (native driver, LIVE 2026-10-21)

`crates/bot/src/mongo.rs` + `[mongo]` in bot.toml (0600). The wiring is
complete and tested (driver 3.x: rustls + dns-resolver + zlib-compression
+ bson-2/compat-3-3-0): boot validates config, connects, pings the
cluster's admin db, and installs a shared handle for future features —
and a failure there is NO fatality (typed log, lane off, bot continues:
the 2026-10-19 crash-loop from a block_on-inside-runtime calling hole
was fixed to an awaited async init the same day).

**Live 2026-10-21** (was OFF pending credential fix — 2026-10-19).
The runbook's fix path below worked exactly as written; two separate
traps surfaced, both diagnosable from the journal + harness *before*
any Atlas-guessing:

1. **Username typo** — the config had `glenemlee_db_user` (one n);
   the Atlas principal is `glennemlee_db_user`. `SCRAM failure: bad
   auth (AtlasError 8000)` can hide either a wrong username OR password.
2. **Wrong cluster hostname suffix** — the URI said
   `cluster0.xf7oa2t.mongodb.net`, under which NOTHING is published:
   the SRV seedlist, the TXT options record, and the shard aliases
   are all absent on three independent resolvers while the
   `mongodb.net` zone itself answers authoritatively (AWS DNS).
   A DNS-first diagnosis (`dig -t SRV _mongodb._tcp.<host>`) catches
   this in seconds; the driver error is a flattened "DNS error" chain.
   Correct host per the operator's credentials file:
   `cluster0.1mzqrfo.mongodb.net`.
Cache the diagnostics before touching config, then verify the write
path (not just reachability) with the live smoke:

```bash
# 1. In the Atlas console (cluster0): check the DB user name + reset its
#    password (or create one) with project access from this host.
# 2. Update bot.toml:
edit ~/.jerith-mesh/bot.toml   # [mongo] uri = "mongodb+srv://user:pass@…"
                               # enabled = false → true
# 3. Verify BEFORE restarting the service (maintenance harness):
JERITH_DISCORD_TOKEN=x cargo test -p jerith-mesh-bot \
  mongo::tests::atlas_ping -- --ignored --exact --nocapture
# 3b. Verify the WRITE PATH (write → read → delete on jerith.memory_mirror,
#     leaves the cluster spotless) — new 2026-10-21:
JERITH_DISCORD_TOKEN=x cargo test -p jerith-mesh-bot \
  mongo::tests::mirror_live_roundtrip -- --ignored --exact --nocapture
# 4. systemctl --user restart jerith-bot   # boot log: "mongodb pinged OK"
```

Never log or paste the uri with credentials; the journal carries the
database name and the ping verdict only.

## Workflow record/replay targets (shadow + driver-VM lanes, 2026-10-20)

The workflow engine (owner-DM-only skills: record → playbook → replay,
`FEATURES §14/§15`) can run its browser three ways. Both container
targets are OFF by default — with no config, the legacy host-chrome
path behaves exactly as day one.

**Lane rules** — precedence is deliberate, documented, and tested:

- `[workflows.shadow] enabled` → replay rides a disposable HEADLESS
  containerized chrome (FEATURES §17.4). Wins over the VM when both are
  enabled: headless-over-headful is the cheaper contract for the same job.
- `[workflows.driver_vm] enabled` → the VM owns RECORD (the owner drives
  windows INSIDE the disposable desktop and watches over its loopback
  VNC map instead of the host display), and replay uses it when the
  shadow is off.
- Neither → host chrome (`[workflows] chrome_binary`), the day-one path.
- Pairing rule (unit-pinned): every workflow verb is owner-DM-only via
  `OWNER_ONLY_PLUGINS`; the §15 approval bridge is the approval surface
  and executes in an owner-privileged registry, never a plain session's.

### §17.4 shadow — headless replay isolation

```toml
[workflows.shadow]
enabled = true
image = "zenika/alpine-chrome:latest"  # the operator vouches; empty+enabled refuses
browser = "chromium-browser"           # browser binary PATH inside the image
mounts = ["/srv/share:/data"]          # host:container[:ro|rw]; default ro
```

Operator facts (all unit-pinned in the argv contract + two live fires):
- fresh tmpfs HOME (`/home/shadow`, 64 MiB, `mode=1777` — images run
  their browser as a non-root user) + manifest-only mounts: replay
  state is born disposable and dies with the container;
- CDP over a loopback-ONLY port map — the endpoint never leaves the
  host (`-p 127.0.0.1:<fresh>:9222`);
- `--entrypoint=<browser>` argv form: works for every image layout,
  including ones whose ENTRYPOINT is already the browser;
- `--headless=old` on purpose: chromium IGNORES the DevTools bind
  address under the new headless runtime (binds container-loopback
  only — debugged live with in-container + host probes 2026-10-20).
  Honest constraint: chromium ≥132 removes that runtime; when images
  migrate, the fix is an in-image socat relay (the driver-VM image
  already ships one), flagged by the live harness;
- docker networking stays default-ON on purpose: replays must reach
  the pages they drive (`--network none` remains the
  sandbox.rs/chat-attachment contract, not this tool's);
- containers are uniquely named `jerith-wf-shadow-*`, removed BY NAME
  at shutdown, and swept by prefix at the next launch (a killed docker
  CLI does not stop a container when the bot dies);
- a plain replay persists only the playbook's `last_run_outcome`;
  the §17.3 escalation packages (AES-256-GCM under
  `~/.jerith-mesh/bot/escalations/`) belong to the cloud-escalation
  macro-tool, not to this lane.

Live fire (one real container replay):
```bash
cargo test -p jerith-mesh-bot \\
  workflows::tests::live_workflow_replay_inside_shadow \\
  -- --ignored --exact --nocapture
```

### §17.1 driver VM — the disposable desktop

Build the shipped image once:

```bash
docker build -t jerith-driver-vm:latest images/driver-vm/
```

```toml
[workflows.driver_vm]
enabled = true
image = "jerith-driver-vm:latest"
browser = "chromium"
size = "1280x800"
```

- The image's ENTRYPOINT is the window-manager bootstrap
  (`images/driver-vm/entrypoint.sh`): Xvfb :1 at
  `$JERITH_VM_SIZE`x24 → openbox (a WM-less X is the typed fallback) →
  x11vnc :5900 → socat CDP relay 9223 → chrome 127.0.0.1:9222 (headful
  chromium binds DevTools loopback-only, so the relay is what makes
  the map reachable) → exec the browser argv, HEADFUL inside Xvfb.
- While a session runs, point a VNC client at the mapped loopback port
  (the journal logs `cdp_port` and `vnc_port` per session).
- Long-tail software: the exercised recipe ships as
  `images/driver-vm-longtail/` (sqlite-web — a real self-hosted app —
  started on the container loopback by the image's own wrapper,
  `longtail-entrypoint.sh`, before the layout bootstrap).
  Operator contract, verified live 2026-10-20:
  1. bake: `docker build -t jerith-longtail-vm:latest \
     images/driver-vm-longtail/` (based on jerith-driver-vm — install
     the app root-side, then `USER chrome`; the install layers need
     root, the runtime does not);
  2. point the lane at the wrapper: `[workflows.driver_vm] boot =
     "/longtail-entrypoint.sh"` — without it the engine chains the
     layout's plain `/entrypoint.sh` and the app never starts (the
     debugged hole: the browser showed `chrome-error://chromewebdata`,
     entity-named by the typed failure's `page: <href> :: <text>`
     triage payload);
  3. persist state through a manifest mount — the demo db rides
     `rw` (`/path/demo.sqlite:/home/chrome/demo:rw`; the file must be
     writable by the container's uid — mode 666 works). The desktop
     dies with its container; ONLY manifest-listed files carry over.
  Full E2E (author → save → run through the REAL app over CDP →
  per-step ✓ → row-count + value verified host-side through the mount):
  `cargo test -p jerith-mesh-bot \\
    workflows::tests::live_workflow_replay_inside_longtail \\
    -- --ignored --exact --nocapture` (9.3 s live).
  Replay-engine robustness the fire earned: click/type wait up to 5 s
  for the selector to exist (just-started apps answer mid-load), and
  CDP attaches only to a real http(s) page target (chrome's own
  omnibox targets are type-page too — order-dependent attach meant
  querying the wrong page forever).

Live fire (the replay-side E2E — author, save, run per-step): 
```bash
cargo test -p jerith-mesh-bot \\
  workflows::tests::live_workflow_replay_inside_driver_vm \\
  -- --ignored --exact --nocapture
```

### Shared accounting

- Playbooks persist 0600 JSON under `~/.jerith-mesh/workflows/` in all
  lanes (the container targets change WHERE the browser runs, never
  where the playbook lives).
- Both container lanes may run while a real owner session goes on —
  they sweep only their OWN prefixed containers
  (`jerith-wf-shadow-*` / `jerith-wf-vm-*`), nothing else.

## Video attachment lane (live 2026-10-20)

FEATURES §20 / `crates/bot/src/video.rs` — uploaded videos are probed
(ffprobe), audio-stripped (ffmpeg → 16 kHz mono WAV), transcribed
(LOCAL openai-whisper, default `tiny` — cached at `~/.cache/whisper`),
and evenly-sampled frames (default every 5 s, cap 8) described through
the gateway's LOCAL vision tier. One video per message; results cache
in the sandbox-extraction cache so a re-ask never re-runs whisper.

- **Ops knobs** (`[video]` in bot.toml): `enabled` (default on;
  `false` degrades every video to an honest config line), bin paths,
  `whisper_model` (`tiny` → `base` doubles wall time for better
  transcripts), `frame_stride_secs`, `max_frames`, `max_duration_secs`
  (default 600), `max_file_mb` (default 512), `whisper_timeout_secs`
  (default 600).
- **Downgrade shapes are by design**: no tool on PATH / download /
  ffprobe / no video stream / over-cap duration / frame-extract /
  vision failures all degrade to honest per-attachment lines in the
  context block — the journal + the chat both say exactly why, nothing
  is silent.
- **Timing reality**: whisper-tiny on CPU ≈ near-realtime (a 6 s clip
  ran the whole lane in ~15 s in the live harness; a 10-minute clip
  will spend most of its time in whisper). The `[chat]` timeout (240 s)
  bounds the wait; the 600 s default duration cap keeps worst-case
  runs inside it.
- **Verify**: two `#[ignore]` live harnesses, run with `cargo test -p
  jerith-mesh-bot --bins video -- --ignored --nocapture`:
  `live_full_lane_on_a_real_generated_clip` (stub vision — pins
  determinism of the toolchain path), and
  `live_full_lane_real_download_and_real_vision_tier` (the production
  shape with zero stubs: local axum "CDN" download → toolchain → the
  LIVE gateway's MiniCPM-V on :8900; verified live 2026-10-21, 9.9 s —
  frame prose was real, no failure markers). Both need ffmpeg +
  ffprobe + whisper with a cached model; the second also needs
  jerith-node up.

## Audio attachment lane (live 2026-10-21)

FEATURES §21 / `crates/bot/src/audio.rs` — plain audio attachments are
ffprobe'd (caps + cover-art honesty) and transcribed DIRECTLY by local
openai-whisper (no strip step; whisper's internal ffmpeg decodes
mp3/ogg/m4a/flac). `.ogg`/`.opus` classify to this lane FIRST, so
Discord voice notes always land here; a real video stream wearing an
audio extension is honestly redirected to the video lane. Two per
message; transcripts warm the extraction cache.

- **Ops knobs** (`[audio]` in bot.toml): `enabled` (default on),
  bin paths, `whisper_model` (tiny → base doubles wall time),
  `max_duration_secs` (default 600), `max_file_mb` (512),
  `whisper_timeout_secs` (600).
- **Downgrade shapes**: tool-missing / download / no-audio-stream /
  no-duration / over-cap / transcription failure all degrade to
  honest per-attachment lines. An EMPTY transcript is by-design
  honest ("track present but transcript came back empty") — the most
  common case is a tone or silent clip, not a toolchain break; check
  the journal if a REAL voice file comes back empty.
- **Verify**: `cargo test -p jerith-mesh-bot --bins live_audio_lane
  -- --ignored --nocapture` — generates a real sine-tone mp3 and runs
  the whole lane on the host toolchain + real whisper (pinned to the
  honest empty-transcript shape).
- **Transcripts are /recall-able (FEATURES §22)**: every successful
  transcription fire-and-forgets one `memory.embed` note
  ("transcript of audio attachment …"), so a later `/recall <words>`
  finds what a voice note said without re-uploading it. Knob:
  `[memory] index_transcripts` (default on). Once per extraction —
  cache-warm re-asks never re-run the embed; embed failures journal
  only and stay re-runnable on the next ask.
- **Model bump, live 2026-10-21**: `[audio]`/`[video]` in the live
  bot.toml now run whisper `base` (was the code default `tiny`) with
  `whisper_timeout_secs = 900` (was default 600) so a 600-s clip at
  base's ~2× rate always fits. Measured on-box: base.pt warmed by the
  lane's exact argv shape; a 6-s tone 23.7 s cold / 5.3 s warm; truth
  probe on real speech (openai-whisper's jfk.flac, 11 s, ~5 s run):
  base = word-perfect WITH punctuation, tiny = near-perfect without —
  so the honest claim is "not worse on clean speech, better on noisy
  notes". The espeak synthetic probe misheard with BOTH models —
  robot voice, not a whisper tier issue. Revert: delete the two
  sections (defaults fall back to tiny/600). **small NOT worth it on
  this box (measured 2026-10-21)**: 46.0 s vs base 5.0 s on the same
  11-s real-speech clip (~9× slower, identical words) and still
  partly garbled on the synthetic probe — a 3-min voice note would
  knife past the 900 s timeout; base stands.
- **Deployment trap (found live 2026-10-21)**: both lanes resolve bare
  bin names from the PROCESS PATH, and a systemd --user service runs a
  NARROWER PATH than a login shell — `~/.local/bin` (where
  openai-whisper lived here) is NOT in it. The login-shell live harness
  passes while the deployed bot returns «no `whisper` on PATH» in the
  model's context; refusals never journal, so nothing looked wrong.
  Fix: `jerith-bot.service.d/10-path.conf` drop-in (and the template's
  `Environment=PATH=%h/.cargo/bin:%h/.local/bin:…`) — reproduce by
  running the harness with the service's exact PATH. Also 2026-10-21:
  the shared `missing tool` refusal no longer points at "[video] bins"
  regardless of lane — it names the real knobs (`[video]/[audio]
  ffprobe_bin / whisper_bin`, absolute paths) and the PATH cause.
- **Refusals are journal-visible (2026-10-21)**: every per-attachment
  failure line now also emits `attachment failure — honest line went to
  the model context` in the journal (converse.rs), so a degraded lane
  is diagnosable from `journalctl` alone — the PATH trap above had zero
  log trace all day because refusal lines lived only in the model's
  context. Over-cap sandbox files also get honest `skipped: N files per
  message…` lines instead of a "pending" that never pends.

## Daily owner self-briefing (FEATURES §24, live 2026-10-22)

`[briefing]` in bot.toml — one owner DM per local day (default hour 08
local; `enabled = true` is the code default) that stitches the past
24h into a single honest note: WARN/ERROR counts + the newest journal
lines (in-process `JournalRing` tracing layer — nothing spawns
journalctl), monitor checks fresh at brief time, mail archive arrivals
(stamps, not receipts), voice seat state + loss/re-establish
counters, outstanding approvals, and mesh peer truth. No owner id
configured → the report is journaled each day instead of DM'd
(fail-quiet, same as the monitor).

- **Knobs**: `hour` (local 0–23, default 8 — the day's brief fires on
  the first hourly tick at/after that hour; a boot after the hour
  catches up immediately, before it waits), `enabled`.
- **Fire state** lives at `$JERITH_HOME/bot/briefing-state.json`
  (`{last_day}`) — exactly one brief per local day, restart-surviving;
  a FAILED DM send is not persisted, so the next tick retries.
- **The drain is the reset**: the journal section's counters mean
  "since the last briefing" — a manual `/status` interlude does not
  reset them; only the daily drain does.
- **Verify live**: set `hour` to the current hour, delete the state
  file, restart — the owner DM lands within the first tick (≤ a
  minute), and the INFO line `daily self-briefing DM delivered`
  confirms. Reset the hour to taste afterwards.

## Discord voice lane (config-opt-in, code live 2026-10-21)

FEATURES §23 / `crates/bot/src/voice.rs` — an always-on `songbird` seat
in one voice channel running a Gemini Live dialog WSS (downmixed room
16 kHz in, model TTS into the mixer reference) plus a parallel
transcribe-live WSS feeding /recall notes and optional live captions.
NOT for voice NOTES — those stay on the audio attachment lane above.

- **It is OFF until configured**: `[voice] enabled = false` is the code
  default. Turn it on in bot.toml with `enabled = true`,
  `guild_id`/`channel_id` (the join targets),
  `gemini_api_key` (parked 0600, never logged), and — if wanted —
  `captions_channel_id` (live caption posts). Without the key the lane
  mistrusts startup with an honest journal line.
- **Knobs**: `model` (gemini-3.8-live dialog / gemini-3.5-transcribe-live
  transcript), `session_max_minutes` clamp 5–120 default 30 — ONE
  wall-time cap re-establishes BOTH sessions together (never one leg
  alone), `max_caption_len` (default 400), `memory.turn_note_max`
  (bounded /recall notes).
- **Watch in the journal** (INFO level): `voice lane starting (always-on)`,
  `vc ws connected` + `vc setup sent` per leg, `vc first server frame`,
  then `live dialog session up` and `captions session up` — that is
  the healthy startup sequence. WARN `server closed the session` carries
  the model + close code + reason verbatim. `session cap reached`
  (every 30 min default) re-establishes both legs together.
- **Seat-hold watcher (added 2026-10-21)**: the live seat can be lost
  silently — observed live that day as the bot dropping out of the VC
  roster while the journal carried no line at all (voice sessions never
  noticed). The lane now spawns a watcher (15 s cadence) that polls the
  parked call handle's `current_connection`; None = seat lost → WARN
  `voice seat lost` + automatic re-join of the configured channel
  through the same manager (ring + capture events ride through the
  rejoin, no rewiring). Lines to know: WARN `voice seat lost` first,
  INFO `voice seat re-join requested`, then the verdict — INFO `voice
  seat back (driver reconnected)` or another WARN while it retries
  (~30 s cadence while out). For instant truth between polls the
  driver's own CoreEvents also journal: `voice driver connected`,
  `voice driver reconnected after a network blip`, WARN `voice driver
  disconnected` (payloads deliberately not logged — connect/disconnect
  data carries voice-server endpoints/keys). /voiceleave disables the
  watcher: an
  operator stop stays stopped.
- **Operator leaves + status (2026-10-21)**: `/status` ends with one
  honest line about the LIVE seat (in-VC · model · session age ·
  captions target) — or `not running` / `joining…` / `left by operator`
  as the process state actually is. `/voiceleave` (owner-DM only, same
  default-deny gate as /timeout) flips the stop switch: the Live
  sessions close within a second (one-second watch tick, not at the
  next cycle boundary), the seat leaves via the same songbird manager,
  and the lane stays out until the next bot restart (`rejoin` =
  restart, deliberately). Journal lines to expect: `voice left on
  operator request` (or a honest WARN if the seat was already gone).
- **Live-fire is the operator's hop**: the automated suite (11 voice
  units: config honesty + clamps, WS/URL shapes, setup JSON both
  lanes, DSP bridges, play-ring pair decode, runtime-state honesty
  for /status + /voiceleave) can't speak; a human
  voice in the VC is what proves the round trip end-to-end. The
  wire-level lessons already learned live (2026-10-21, FEATURES §23
  has the detail): `?model=` is rejected (setup carries the model),
  `diarization` is rejected, server→client frames are BINARY carrying
  JSON, and a silent room needs the pump's 40 ms silence keepalive —
  zero media earns a 1008 abort at exactly 60 s.
- **TLS trap (fixed 2026-10-21, affects other ws users of the stack)**:
  the workspace `tokio-tungstenite` pin had feature `rustls` — the
  IMPLICIT optional-dependency feature, which pulls the rustls crate
  but never tokio-rustls; the handshake compiled, ran, and honestly
  failed at connect time with «URL error: TLS support not compiled
  in» (the supervised restart loop surfaced it immediately). The pin
  is now `rustls-tls-webpki-roots`, which pulls the full TLS path
  (webpki 0.26 rides the same accepted serenity-subgraph pin). Slack/
  twitch/mattermost `connect_async` callers were silently in the same
  boat. Lesson: in tokio-tungstenite ≥0.26, `rustls` alone is not
  TLS — `cargo tree -f "{p} {f}" -p tokio-tungstenite` shows the
  difference (`tokio-rustls,webpki-roots` present = real). A tungstenite
  live probe test (`voice::tests::live_ws_probe_setup_complete_arrives`,
  `--ignored`) reproduces the correct handshake both ways.
- **Revert**: `[voice] enabled = false` (or delete the section) leaves
  the bot exactly as before — no voice code starts unless the section
  says so.

## Cloudflare Tunnel hardening + the two-week nmonit 530 (2026-10-27)

Trigger: Cloudflare's Oct 20 advisory — eight emSign/eMudhra intermediates with weak keys; `cloudflared` < 2026.10.2 still trusts them.

- **State found**: both connectors (`cloudflared-jerith.service`,
  `nmonit-cloudflared.service` — both systemd *user* units under
  `~/.config/systemd/user/`, easy to miss with plain `systemctl`) ran
  the dpkg-managed binary at `2026.10.0`. `/usr/local/bin/cloudflared`
  is a symlink to `/usr/bin/cloudflared`, so one package covers both.
- **Upgrade**: official `cloudflared-linux-amd64.deb` 2026.10.3 →
  `dpkg -i`, then restart per unit. Newer 2026.10.5 exists; cloudflared
  will warn whether we're current — check before assuming an upgrade.
- **Origin sweep (advisory asks: no `noTLSVerify`, no origin chaining
  to the affected CAs)**: the bootstrap tunnel's origins are plain
  `http://localhost:8126`/`:8901` — no TLS surface at all. nmonit
  however had two HTTPS loopback origins pinned with
  `noTLSVerify: true`; see below for the fix.
- **Root cause of the weeks-old `nmonit.jerithai.com` 530** (NOT the
  advisory, and predates it): on Oct 16 the mesh rewrite turned
  `~/.cloudflared/config.yml` into the *bootstrap* tunnel's config
  (`tunnel: fa81a9d9…`). The nmonit unit is
  `cloudflared tunnel run nmonit` with **no `--config`**, and without it
  cloudflared reads that default file — its `tunnel:` key wins, so the
  process silently joined the WRONG tunnel. `cloudflared tunnel list`
  told the truth: `nmonit` (c4139072…) had zero connections while
  jerith-bootstrap carried 8 (2 processes × 4). The dashboard-side
  hostname URL was fine; the tunnel behind it was empty. The user fixed
  the DNS/CNAME record, which moved the error 1016 → 1033 (edge
  propagation), which is how it looked "half-fixed" for a few minutes.
- **Fix**: dedicated `~/.cloudflared/nmonit.yml` (tunnel id + creds +
  ingress for `nmonit.jerithai.com`; stale `grpc-nmonit →
  https://127.0.0.1:9000` route dropped — :9000 now binds
  `192.168.1.6:9000` and no longer speaks TLS on loopback).
  `grpc-nmonit.jerithai.com`'s DNS record was itself deleted from the
  zone 2026-10-27 evening — authoritative NS answers NXDOMAIN for it,
  and a reference sweep of both hosts found zero clients resolving it
  before deletion. Unit ExecStart now pins
  `--config ~/.cloudflared/nmonit.yml`. **Rule going forward: one
  config file per connector; never let `tunnel run <name>` inherit a
  default config that belongs to a different tunnel.**
- **TLS verification restored, not disabled**: origin is the nmonit
  control-plane on `https://127.0.0.1:8080` — cert CN
  `control-plane`, issued by the private root
  `~/Documents/Projects/nmonit/deploy/certs/ca.crt` (SANs:
  `control-plane`, `localhost`, host, `127.0.0.1`, `192.168.1.6`).
  `nmonit.yml` sets `originRequest.caPool` to that root and
  `originServerName: control-plane` — verification passes with
  nothing  disabled, which is what the advisory wants instead of
  `noTLSVerify`. Affected-CAs check: the local root is unrelated to
  the emSign/eMudhra intermediates.
- **Verified**: `cloudflared tunnel list` → c4139072 nmonit 4×http2 conns,
  jerith-bootstrap back to its own 4; `curl https://nmonit.jerithai.com`
  → 200; `p2p.jerithai.com` still 404-from-origin at `/` (normal).
  Remaining noise only: ICMP proxy warning (`ping_group_range`),
  pre-existing.
- **Oddity resolved** (was "to watch"): the dead `jerith-wordpress`
  tunnel (401207aa…) reports live connectors not accounted for on this
  box — **it's the Jetson Orin Nano** (`noodlypanda@192.168.1.150`,
  aarch64; same LAN WAN IP 66.168.170.18). SSH-verified 2026-10-27:
  `cloudflared 2026.5.2` from `/usr/local/bin`, process
  `tunnel --config /home/noodlypanda/.cloudflared/config.yml run` —
  own config, own tunnel, healthy. **Jetson action from the advisory
  closed 2026-10-27**: `cloudflared-linux-arm64.deb` 2026.10.3
  installed (dpkg-switched `/usr/local/bin/cloudflared` to a symlink;
  the old 2026.5.2 binary kept at `…/cloudflared-2026.5.2`). The
  unsupervised orphan process was replaced by a supervised user unit
  `~/.config/systemd/user/cloudflared-wordpress.service` (linger was
  already on) — `tunnel info` now shows only 2026.10.3 connectors,
  8 connections, unit active after the orphan handoff.
  The guard is installed there too as a user unit
  (`jerith-cloudflared-guard.service`; script at
  `~/.local/bin/check-cloudflared-units.sh` — deliberately outside the
  Jetson's `~/jerith-mesh` clone to keep its tree clean), runs at boot
  and currently checks the new pinned unit: PASS.
- **Regression guard (this host)**: `scripts/check-cloudflared-units.sh`
  + oneshot user unit `deploy/systemd/jerith-cloudflared-guard.service`
  (install: cp to `~/.config/systemd/user/`, `enable --now`). Runs at
  every boot and fails hard if any `*cloudflared*.service` unit runs a
  tunnel without its own `--config`, points `--config`/
  `credentials-file` at a missing file, or two units claim the same
  `tunnel:` id. CI can't see host systemd units, so this is the
  enforcement point.
- **Main site restored 2026-10-27 evening — the DNS was NEVER wrong**
  (correction to the earlier read of the same day): the dashboard rows
  for `jerithai.com`/`www.jerithai.com` say `Tunnel: nmonit`, and that
  was always intentional — the desktop's site (`/var/www/jerithai`,
  nginx `:80`, title "Jerith Mesh + Jerith Coin (JER) — JerithAI") got
  to the internet through the nmonit tunnel, per
  `config.yml.bak-prejerithai`/`bak-prenmonit2`. The morning nmonit.yml
  restoration carried only the `nmonit.jerithai.com` rule, dropping the
  two apex/www rules — hence the body-less 404s. Fix was local only:
  re-added `jerithai.com` + `www.jerithai.com` → `http://localhost:80`
  to nmonit.yml; all three nmonit-tunnel hostnames now 200 from the
  edge. Meanwhile the marker-probe tooling (200-line in the candidate
  tunnel's ingress + restart + curl) remains the correct discriminator
  for "which tunnel does this hostname really hit" — the dashboard rows
  are evidence but the edge is truth. The Jetson-side
  `cloudflared-wordpress` tunnel still declares apex/www in its local
  ingress that DNS never routes to it — harmless duplication, on the
  Jetson as `~/.config/systemd/user/cloudflared-wordpress.service` if
  anyone ever repoints the rows again.
- **Revert**: `nmonit.yml` back to
  `config.yml.bak-prenmonit2` ingress rules, and the unit's ExecStart
  back to bare `tunnel run nmonit` — recreates the wrong-tunnel bug if
  done without ALSO restoring the old config.yml, which is the trap
  that caused this. Do it atomically.

## The weather skill (sixth live consumer, 2026-10-27)

`crates/weather-plugin/` gives the room a REAL weather lane over the
same signed contract — not an echo demo like `crates/plugin-template` —
public by design (open-meteo upstream, no api key), gated via
`[plugins] enabled` and NEVER `owner_only`:
`weather.now` = current conditions at any place on Earth;
`weather.forecast` = 1-7 day series. Pure parsing lives in its lib.rs
(9 tests), wire glue in main.rs (manifest row/handler pin), and a
bot-side manifest pin is the fifth in external.rs's row.
```bash
mkdir -p ~/.jerith-mesh/plugins.d/weather
cp crates/weather-plugin/plugin.toml ~/.jerith-mesh/plugins.d/weather/   # rotate the secret first
cp deploy/systemd/jerith-weather-plugin.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now jerith-weather-plugin
edit ~/.jerith-mesh/bot.toml   # 'external:weather' → [plugins] enabled (public surface)
systemctl --user restart jerith-bot
```

Live-fire (2026-10-27, this house): unit `jerith-weather-plugin`
serving `127.0.0.1:7793` (signed), bot restarted clean with the drop
gating on. HMAC-signed probe (`weather.forecast`, place=Prague,
days=2) → 200: `open-meteo` card with real values (14.2 °C / 3.9 °C,
overcast, next 2 days); unknown place (`Zzzzxcvbnope`) → the typed
refusal `no place matched — check the spelling` the model relays
verbatim.

Same session, the guild-watcher retired: the welcome-DM lane moved
into `crates/bot/src/welcome.rs` (`[welcome]` in bot.toml — the
OpenClaw guild-watcher port: same three embeds, the red-panda GIF pool
migrated to `~/.jerith-mesh/workspace/gifworkshop/redpandas/`, members
intent unioned in main.rs). The OpenClaw-era Python unit
(`guild-admin-watcher.service`) is stopped + disabled 2026-10-27
(reversible with `systemctl --user start`); both lanes firing at once
would double-DM — keep only one up.
