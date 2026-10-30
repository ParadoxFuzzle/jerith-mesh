# jerith-mesh

> A model-homogeneous P2P compute mesh for MoE LLM inference.
> Nodes running the *same model* pool RAM, VRAM, GPU and CPU — in the spirit of
> [nmonit](https://github.com/ParadoxFuzzle/nmonit) — using a
> [Colibri](https://github.com/JustVugg/colibri)-style expert-loading runtime.

---

## What is this?

Classic expert parallelism assumes microsecond interconnects. The public
internet gives you 30–300 ms per hop. **jerith-mesh** restructures the problem:

1. **Only connect to nodes running your exact model** (GLM-5.3 talks to GLM-5.3,
   never to Qwen3.8-30B).
2. **Keep experts local-first** — use Colibri's on-demand expert loading as the
   local primitive, and only cross the network for hot experts that a nearby
   peer owns.
3. **Hide latency, never assume low latency.** Pipeline fallback
   (Petals-style), speculative prefetch, timeout-tolerant async execution.
4. **Contribution is verifiable.** Health pings, redundant replay-checks,
   cryptographic commitments, and statistical freerider detection.
5. **Earnings are real.** Metered inference/expert-activation/bandwidth/uptime
   settles into [JerithChain](https://github.com/ParadoxFuzzle/jerithchain)
   (**JER**, default) — or Bitcoin, Ethereum, Dogecoin, Monero with user-supplied
   wallets/miners.

### The hard rule

> A node running **GLM-5.3** discovers and connects only to other **GLM-5.3**
> nodes. Changing models disconnects from all peers, then joins the mesh for
> the new model. There are no mixed-model edges in the mesh.

---

## User surfaces

| Surface | Status | Notes |
|---|---|---|
| CLI (`jerith`) | ✅ MVP | setup, run, status, peers, model switch, stake, memory — plus `jerith chain` (JerithChain node introspection: status / wallets / balance / miners / block / tx over the co-hosted chain's :8322 API) |
| Web UI | ✅ MVP status page | full dashboard in later phases |
| Discord bot | ✅ MVP | user slash: `/ask` `/reset` `/remember` `/recall` `/briefing` `/lore` `/status` `/peers` `/read` `/mine` `/leaderboard` `/myjercraft` `/help` · owner-DM-only levers: `/mail` `/gif` `/timeout` `/voiceleave` (`crates/bot`) — chats can web_search/web_fetch via the node's web skills; always-on voice lane (FEATURES §23) rides `[voice]` opt-in; OpenClaw trust model: system-level commands are owner-DM only (default-deny for unclassified commands) |
| OpenClaw migration | ✅ | `scripts/migrate-from-openclaw.sh` — stops the stack, reuses model + bot token + bot owner id, stages the Brave Search API key to `~/.jerith-mesh/brave-key` (node `[websearch]` skills on by default: web search + URL fetch), and stages workspace scripts/plugin-skills/extensions/docs into `~/.jerith-mesh/openclaw-migrated/` with cron + service manifests |
| Thread archiver (OpenClaw skill port) | ✅ | Discord threads → decision/note extraction → node memory (`/recall`) + optional webhook + markdown archive; opt-in via bot.toml `[archiver] channel_ids = [...]`, per-guild marker overrides via `[archiver.guild_markers.<id>]` |
| Bot workspace (OpenClaw workspace pattern) | ✅ | `~/.jerith-mesh/workspace/` canon docs (IDENTITY/SOUL/USER/USER_MODEL/AGENTS/MEMORY) staged from the OpenClaw workspace on first boot, composed into the bot persona on every restart — edit the docs, restart the bot |
| `/briefing` (what-did-i-miss port) | ✅ | Recent decision/note lines across the watched threads + a 3-line mesh digest, on demand |
| Owner-DM trust model | ✅ | OpenClaw `ownerAllowFrom` port: commands outside the user-level list are refused for everyone; system-level actions need a DM from the bot owner (default-deny for unclassified commands) |
| JerithChain co-hosted node | ✅ | The local JerithChain node (FastAPI :8322, SQLite ledger, api token shared at `~/.jerith-mesh/keys/api.token`) is part of the mesh install: shared auth, `jerith chain …` CLI, chain section on the node's `/status` |
| OpenAI-compatible API | ✅ MVP | `/v1/chat/completions`, `/v1/models` |
| Telegram / WhatsApp etc. | 📋 Plugin SDK | user-contributed modules |

---

## Quick start

> ✅ **The bootstrap network is LIVE:** `https://p2p.jerithai.com` answers
> `/health`, `/v1/peers?model=…` and `/v1/models/counts` on the public
> internet. New nodes find each other with zero configuration.

Prerequisites: **Rust ≥ 1.85** (edition 2024 — `rustup update stable`) and
git; Linux hardware probing uses `inxi` (`sudo apt install inxi`) if present,
else a fallback scan. Linux-only GPU flags need your vendor's driver stack.

```bash
git clone https://github.com/ParadoxFuzzle/jerith-mesh.git
cd jerith-mesh
cargo build --release

# First setup: hardware scan → model pick (with live mesh counts) →
# memory backend → contribution %
./target/release/jerith setup

# Run the node: signed discovery via p2p.jerithai.com, mutual signed
# pings, expert RPC, local gateway + web UI
./target/release/jerith run

# Check the mesh (peers = same-model nodes only)
./target/release/jerith status
./target/release/jerith peers

# Local memory skills (SQLite or PostgreSQL+pgvector — your setup choice —
# + nomic embeddings, never leaves the node)
./target/release/jerith memory add "my note"
./target/release/jerith memory search "note"

# One-command live two-node E2E: local bootstrap + stub runtime + nodes
# A/B with verify_percent/ringer_percent = 100 (pinned dial, commitment
# checks, ringer spot-counters) — builds, runs, reports counters, and
# tears itself down:
scripts/dev-e2e.sh
```

### What's live today

| Capability | Status |
|---|---|
| Public bootstrap at `p2p.jerithai.com` (Cloudflare Tunnel) | ✅ live |
| Ed25519-signed advertise/pings (forged & replayed requests rejected) | ✅ live |
| Model-homogeneous meshing (GLM↔GLM, never GLM↔Qwen) | ✅ live |
| 3-failed-ping deprecation + failed-worker fraud reports & scoring | ✅ live |
| Local memory + nomic embeddings (`memory.*` skills; SQLite or the chosen PostgreSQL+pgvector store) | ✅ live |
| Web access skills (Brave Search + URL fetch — OpenClaw migration, bot tool loop) | ✅ live |
| QUIC NodeRPC expert dispatch (runtime-backed) | ✅ Phase 1 |
| JER mining on served inference | ✅ live |
| Expert dispatch: locality-aware placement + speculative remote pre-flight (hybrid local + remote, replay verification preserved on the speculative path) | ✅ Phase 2 |
| Tensor transport: on-disk expert weights served over QUIC (on-demand LRU hot set, real SwiGLU forward, honest refusal for unhosted experts, peers advertise what they hold) | ✅ Phase 2 |
| Expert replication + sharding: measured demand (`expert_stats`) drives hot-expert replica fetches (chunked, checksum-validated weight transfer) and a rendezvous ring gives cold experts exactly one owner | ✅ Phase 2 |
| Discord bot + plugin/skill SDK (OpenClaw-style registry, bot.toml `[plugins]` gating; first-class `memory.*`/`lore.*` + web tools; six external-process drops live: template example, pet, house, quarantine, ledger, weather — the last one PUBLIC) | ✅ Phase 3 core |
| Signed plugin channel: `.jbundle` install chain, per-box receipt audit log, fail-closed receipt gates (CLI install + bot restart), hourly systemd verdict check | ✅ 0.1.4 |

### Setup flow

1. **Hardware scan** — Linux: `inxi -Fxz`; Windows: `systeminfo`; macOS:
   `system_profiler SPHardwareDataType` (all wrapped with a sysinfo fallback in
   Rust, so it works even if those tools aren't installed).
2. **Model choice** — waits for you to confirm; can show live peer counts from
   the bootstrap node first so you see which meshes are actually alive.
3. **Memory backend** — PostgreSQL+pgvector or SQLite (strictly local, never
   replicated; no distributed databases in the mesh; the chosen backend is the
   one used — postgres needs `memory_dsn`, memory skills refuse to run
   against a different store than the configured one).
4. **Contribution %** — minimum **25 %** required to be eligible; configurable
   upward.
5. **Small-model tiers** — pick from the curated small-model catalog per tier:
   *thinking/tools* (`qwen3:8b`, `qwen3:4b`, `deepseek-r1:8b`, `nemotron-3-nano:4b`),
   *vision* (`minicpm-v4.6:1b`, `qwen3-vl:4b`, `moondream2`), *embedding*
   (`nomic-embed-text-v2-moe`, …) — or any `ollama:<tag>`.
6. **Optional AI tool stack** — install any of the companions below (each
   confirmed step by step; `jerith stack install <tool>` works standalone too).
   Tier models are `ollama pull`-ed as part of setup.
7. **Optional crypto wallets** — JerithChain by default; you provide BTC/ETH/
   DOGE/XMR endpoints yourself.

---

## AI tool stack & integrations

```bash
./target/release/jerith stack list            # tools + the small-model catalog
./target/release/jerith stack install ollama  # …also: litellm, langchain,
                                              # langgraph, langsmith, postgres
./target/release/jerith stack pull qwen3:8b   # pull small models into ollama
```

| Tool | What it gives the mesh |
|---|---|
| **Ollama** | hosts the tier models locally (setup's default runtime URL: `http://127.0.0.1:11434`) |
| **LiteLLM** | OpenAI-compatible proxy over any backend — `runtime_url=http://127.0.0.1:4000` |
| **LangChain / LangGraph** | agents that consume the node's OpenAI-compatible gateway (`integrations/langchain/` — pinned `JerithChat` + skill tools) |
| **LangSmith** | optional tracing for LangChain runs (env vars only, nothing phones home otherwise) |
| **PostgreSQL + pgvector** | the node's scalable memory backend (`memory_backend=postgres`) |

The LangChain integration lives at
[`integrations/langchain/`](integrations/langchain/README.md) — the node's
memory/lore/web-search skills are exposed as LangChain tools, everything stays
on the node except the model calls you route through the mesh.

---

## Documentation

| Doc | Contents |
|---|---|
| [FEATURES.md](FEATURES.md) | Full feature list distilled from the spec |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Phases, wire protocol, MoE routing, anti-fraud, economics |
| [docs/PLUGINS.md](docs/PLUGINS.md) | Plugin/skill system: registry, gating, authoring a skill, hooks |
| [docs/PLUGIN-SDK.md](docs/PLUGIN-SDK.md) | Design (Phase 3+): external-process plugins, Telegram/WhatsApp transports |
| [docs/PLUGIN-CHANNEL.md](docs/PLUGIN-CHANNEL.md) | Signed `.jbundle` distribution: install chain, signed channel index, receipt audit log, fail-closed gates |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | Operator runbook: post-deploy verification, rollback, plugin-channel ops |
| [docs/UPGRADE.md](docs/UPGRADE.md) | Upgrading a node + bootstrap (ordering rule, wire-contract checks, bump protocol) |
| docs/SETUP.md | Setup walkthrough (planned) |

## Repo layout

```
crates/
  node/       the `jerith` CLI binary (setup, run, status, peers, stack,
              + the plugin lane: `plugin install/verify/remove/trust/
              search/channel/receipts` — docs/PLUGIN-CHANNEL.md)
  bot/        the `jerith-bot` Discord bot (slash commands → gateway + JerithChain)
  core/       hardware probe, config, peer registry, ping loop, model gateway
  weather-plugin/  the public weather plugin (weather.now/forecast; the
              signed `.jbundle` channel's first-party row)
  …           pet, house, quarantine and ledger plugins, the
              `plugin-template` SDK scaffold, `gcp-bridge`
integrations/
  langchain/  LangChain/LangGraph integration package (uses the local gateway)
proto/        wire protocol (planned Phase 1)
docs/         design documents
```

## What "Contribution" means

The mesh tracks four meters per node:

| Meter | What counts | Who can see it |
|---|---|---|
| Inference tokens | tokens generated for peers | all peers |
| Expert activations | expert loads served for peers | coordinator |
| Bandwidth | bytes relayed for experts | all peers |
| Uptime | time healthy & reachable | bootstrap + peers |

Below-25 % nodes are flagged; persistent free-riders get peer-reputation
penalties and eventually exclusion. All meters are visible to every peer
in the mesh; individual users' data stays private.

---

## Anti-fraud (Phases 4+)

Redundant replay on a fraction of requests → cryptographic commitments →
ringer challenges → statistical freerider scoring → stake-and-slash (JER) →
optional TEE / ZK tiers. See ARCHITECTURE.md §8.

## License

MIT OR Apache-2.0
