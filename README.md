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
| CLI (`jerith`) | ✅ MVP | setup, run, status, peers, model switch |
| Web UI | ✅ MVP status page | full dashboard in later phases |
| Discord bot | ✅ MVP | `/ask` `/reset` `/remember` `/recall` `/status` `/peers` `/mine` (`crates/bot`) |
| OpenClaw migration | ✅ | `scripts/migrate-from-openclaw.sh` (stop stack, reuse model + bot token, start node/bot) |
| OpenAI-compatible API | ✅ MVP | `/v1/chat/completions`, `/v1/models` |
| Telegram / WhatsApp etc. | 📋 Plugin SDK | user-contributed modules |

---

## Quick start

> ✅ **The bootstrap network is LIVE:** `https://p2p.jerithai.com` answers
> `/health`, `/v1/peers?model=…` and `/v1/models/counts` on the public
> internet. New nodes find each other with zero configuration.

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

# Local memory skills (SQLite + nomic embeddings, never leaves the node)
./target/release/jerith memory add "my note"
./target/release/jerith memory search "note"
```

### What's live today

| Capability | Status |
|---|---|
| Public bootstrap at `p2p.jerithai.com` (Cloudflare Tunnel) | ✅ live |
| Ed25519-signed advertise/pings (forged & replayed requests rejected) | ✅ live |
| Model-homogeneous meshing (GLM↔GLM, never GLM↔Qwen) | ✅ live |
| 3-failed-ping deprecation + failed-worker fraud reports & scoring | ✅ live |
| Local SQLite memory + nomic embeddings (`memory.*` skills) | ✅ live |
| QUIC NodeRPC expert dispatch (runtime-backed) | ✅ Phase 1 |
| JER mining on served inference | ✅ live |
| Remote expert tensor execution with placement/speculation | 📋 Phase 2–3 |
| Discord bot / plugin SDK | 📋 Phase 3 |

### Setup flow

1. **Hardware scan** — Linux: `inxi -Fxz`; Windows: `systeminfo`; macOS:
   `system_profiler SPHardwareDataType` (all wrapped with a sysinfo fallback in
   Rust, so it works even if those tools aren't installed).
2. **Model choice** — waits for you to confirm; can show live peer counts from
   the bootstrap node first so you see which meshes are actually alive.
3. **Memory backend** — PostgreSQL+pgvector or SQLite (strictly local, never
   replicated; no distributed databases in the mesh).
4. **Contribution %** — minimum **25 %** required to be eligible; configurable
   upward.
5. **Optional crypto wallets** — JerithChain by default; you provide BTC/ETH/
   DOGE/XMR endpoints yourself.

---

## Documentation

| Doc | Contents |
|---|---|
| [FEATURES.md](FEATURES.md) | Full feature list distilled from the spec |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Phases, wire protocol, MoE routing, anti-fraud, economics |
| docs/SETUP.md | Setup walkthrough (planned) |

## Repo layout

```
crates/
  node/       the `jerith` CLI binary (setup, run, status, peers)
  bot/        the `jerith-bot` Discord bot (slash commands → gateway + JerithChain)
  core/       hardware probe, config, peer registry, ping loop, model gateway
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
