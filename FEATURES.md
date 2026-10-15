# jerith-mesh — Detailed Feature List

*(P2P model-homogeneous distributed MoE compute mesh)*

Status legend: ✅ in MVP · 🔄 in progress · 📋 planned (phase noted)

## 1. Core Distributed Compute
- ✅ Model-homogeneous peer meshes: nodes connect only to others running the *same* model & quantization
- ✅ Colibri-style MoE runtime integration (Phase 0: single-node expert streaming; local primitive)
- ✅ Resource pooling of RAM, VRAM, GPU, CPU across the mesh (Phase 1)
  - Peers advertise live `ResourceSample` snapshots (free/total RAM, swap,
    logical cores, per-GPU VRAM) inside their signed ads at every ping cycle
  - Placement orders candidates by the pool: reported-capacity peers first
    (most free VRAM → RAM → RTT tiebreak), then unreported peers by pure RTT
  - Wire-compatible: pre-pooling ads deserialize fine (missing field → None)
- ✅ Hybrid local + remote expert execution with locality-aware placement + speculative pre-flight (Phase 2)
  - Planning is resource-aware (Phase 1 pool order) and local-first: Local,
    then remotes with the expert by placement key; fallback walks the plan
  - Speculative pre-flight: when the plan opens with Local and the best remote
    also holds the expert, the remote attempt is built up front and churns
    while the local run executes — consumed immediately if the local try
    fails, dropped if local wins; the pre-flighted peer is never worked twice
  - Redundant-replay verification + verification-tracker semantics
    (trusted / reject-continue / reject-abort) are preserved on the
    speculative path; production MeshLoop router opts in
  - ✅ **Tensor transport (Phase 2)**: experts are real weights on disk
    (`expert_store.rs`) — one `<expert_id>.expert` container per expert
    (magic/version/dims/id/gate-up-down tensors/checksum), loaded
    on demand into a byte-budgeted LRU so the hot set stays resident and
    cold experts are re-read (Colibri's shape, plain files for now)
  - Serving runs the actual expert math: `down(silu(gate·x) * (up·x))`
    over the activation vector that arrived on the wire, with typed
    refusals for a corrupt/misnamed/missing expert (a node never
    fabricates activations; `[experts] tensors_only` keeps the legacy
    Phase-1 text stand-in available but off by default)
  - Peers **advertise the experts they hold** (`hosted_experts` +
    `hosted_expert_count`, capped + wire-compatible with pre-Phase-2
    ads), and expert placement only counts peers that advertise holding
    *that* expert — resource-aware ordering now runs over real holders
  - A node serves expert tensors with **no chat runtime at all**: the
    QUIC expert server starts when the node hosts experts, not only when
    an LLM runtime is reachable
  - Operator surfaces: `jerith experts list|make|run`, `/status.experts`
    (hosted ids, resident bytes, hit/miss/load/eviction counters), and a
    MeshLoop `dispatch_expert` entry that reports which node answered
  - Quantized shards (Q4_K/Q8_0 dequant, mmap'd GGUF export) are the next
    step; the store boundary (`run(expert_id, input) -> Vec<f32>`) holds
- ✅ Hot-expert replication (top 10–20 %) + cold-expert sharding (Phase 2)
  - Demand is *measured*, not guessed: every expert this node serves or
    dispatches is counted (`expert_stats`, persisted to
    `data/expert-stats.json`); the hot set is the top slice of the
    *distinct* experts observed, at `[experts] hot_percent` (clamped to
    the spec's 10–20 % band, deterministic ties)
  - Under-replicated hot experts are pulled in as replicas from the
    nearest peer that **advertises** holding them: `ReplicaManager` runs
    one background round every `replication_interval_secs` (off with
    `replication_enabled = false`), starting at most
    `replication_max_actions_per_round` fetches and aiming at
    `replication_factor` holders mesh-wide; a peer whose hosting is only
    unknown (legacy ad) counts toward the factor but is never a source
  - The transfer is chunked and checksummed end to end: one signed
    `ExpertFetch` per 256 KiB slice (1 MiB server-side chunk cap, 512 MiB
    container cap) on a single QUIC stream, and the replica is only
    published after the container parses — magic/dims/id/checksum — so a
    truncated or tampered transfer is refused, never served or advertised
  - Cold experts (the long tail) get exactly one owner from a rendezvous
    (highest-random-weight) ring over the live node ids: every node
    derives the same owner from the ads it already has — no coordinator —
    and a departing node reassigns only its own experts; `dispatch_expert`
    asks that owner when nobody advertises the expert
  - Operator surfaces: `/status.experts` reports the activation counts,
    the hot set, the cold size, each sampled cold expert's shard owner,
    the ring size and the live replication policy
- ✅ Pipeline / layer-group fallback (Petals-style) for high-latency links (Phase 2–3)
- 🔄 Speculative decoding + expert budgeting + prefetching (Phase 3)
- 🔄 Asynchronous, timeout-tolerant forward passes; graceful degradation (Phase 3)
- ✅ Dedicated local specialist models:
  - Vision: MiniCPM-V 4.6:1b (never leaves the node unless the user opts in)
  - Embeddings: nomic-embed-text-v2-moe (Matryoshka dims 768/512/256)
  - Both excluded from the 25 % contribution accounting — they are personal infrastructure

## 2. Node Roles & Participation
- ✅ Three roles: **Contributor**, **Requester**, **Hybrid**
- ✅ Minimum **25 %** resource contribution required to be eligible to connect; user-configurable upward
- ✅ Contribution changeable at any time (hot reload via `jerith config set contribution_percent N`)
- ✅ Model switching cleanly disconnects from all current peers, then joins the new mesh
- 📋 Real-time contribution measurement and cryptographic verification (Phase 4)

## 3. Connectivity & Discovery
- ✅ Bootstrap + DHT hybrid discovery (Phase 0: single bootstrap file + HTTP peer list)
- ✅ Primary bootstrap at `p2p.jerithai.com` via Cloudflare Tunnel; overridable
- ✅ Continuous health pings (minimal-resource requests); offline nodes deprecate and return on re-up
- ✅ Fraudulent / non-contributing nodes deprecated and refused further work
- ✅ Additional independent bootstrap nodes (`[discovery] servers`): per-call failover rotation — last-good server first, then primary, then the rest; one dead bootstrap degrades a single tick instead of the process lifetime
- ✅ Gossip-based list maintenance: peers carry their rotation inside their signed ads (`PeerAd.discovery_servers`, capped at 8); recipients merge https-only (plain-HTTP entries stay config-only), capped at 16 total, and persist the learned set to `<data>/discovery-servers.json` so a restart keeps it
- ✅ Discovery state visible in `/status` (`discovery` section) and in the bot's `/status` line

## 4. Resource Measurement & Capability Gating
- ✅ OS detection with per-OS inventory tools:
  - Linux → `inxi -Fxz`
  - Windows → `systeminfo`
  - macOS → `system_profiler SPHardwareDataType`
  - Every tool has a Rust/sysinfo fallback so scanning never fails on a missing binary
- ✅ Model-fit detection: which curated models this machine can realistically run alone
- ✅ Leech prevention: warning/block on models far beyond local capacity
- ✅ Clear per-node report of available vs contributed resources

## 5. Models
- ✅ Curated defaults: GLM-5.2, GLM-5.3, GLM-5.3-Flash, DeepSeek-V4-Flash, DeepSeek-V4.1-Flash, Kimi-K3, Qwen3.8-Flash-Next, Qwen-Image-2.1
- ✅ Any GGUF / safetensors the Colibri-style runtime loads (the model mesh identity includes quantization and context length)
- ✅ HuggingFace and Ollama model selection
- ✅ Local specialists (MiniCPM-V vision + nomic embeddings) routed locally, never meshed

## 6. Memory & State
- ✅ Backend choice at setup: **PostgreSQL+pgvector** or **SQLite**
- ✅ Memory is strictly local — never replicated into the mesh, never distributed in the mesh
- ✅ Local embedding pipeline integrated with the chosen vector store (Phase 0.5)

## 7. User Interfaces & Access Surfaces
- ✅ CLI (setup, run, status, peers, config);
- ✅ Web UI status/dashboard page
- ✅ OpenAI-compatible HTTP API (`/v1/chat/completions`, `/v1/models`, streaming)
  Chat replies carry `mesh_fallback` (node id when a mesh peer served it,
  explicit `null` otherwise — including 502s when neither the local runtime
  nor mesh peers are reachable: never a bare 500).
- ✅ Discord bot (`crates/bot`, poise/serenity): `/ask` (mesh-peer fallback included,
  screened outbound by Jerith Guard before the gateway call — optional
  `[guard]` in bot.toml),
  `/reset`, `/remember`, `/recall`, `/status`, `/peers`, `/mine`, `/help`;
  per-user chat history (in-memory, capped) + per-user cooldowns; token via
  `JERITH_DISCORD_TOKEN` or `~/.jerith-mesh/bot.toml`; gateway/chain URLs via
  `JERITH_GATEWAY_URL` / `JERITH_CHAIN_URL`;
  `/myjercraft` — your own JER balance from the chain's `GET /balance`
  (balance + available-after-mempool amounts, short address, opted-out
  warning; no-wallet state points at `/mine` for auto-creation)
  `/leaderboard` — JER mining leaderboard; prefers chain-authoritative
  lifetime totals via jerith_node.py `GET /mystate` (labelled "chain
  totals"), falls back to the bot's persisted tally (labelled
  "bot-observed wins") when the chain is unreachable; tally saved to
  `$JERITH_HOME/bot/mine-tally.json` (or `~/.jerith-mesh/bot/`) after
  each win and restored on startup
  `/mine` wins optionally post to a channel via
  `JERITH_ANNOUNCE_CHANNEL_ID` (or `announce_channel_id` in bot.toml)
- 📋 Plugin SDK surfaces for Telegram, WhatsApp, others (Phase 3+)

## 8. Plugin / Skill System
- 🔄 OpenClaw-style modular plugin and skill architecture (Phase 3 core ✅;
  see plugins.rs):
  - Capabilities are **plugins** owning dot-named **skills** (`web_search`,
    `memory.embed`, …): one `SkillSpec` table entry per skill derives the
    OpenAI tool declaration, the dispatch whitelist and the gating — the
    chat tool loop reads only from the registry (`crates/bot/src/plugins.rs`)
  - Owners gate per plugin via bot.toml `[plugins] enabled = [...]`
    (websearch/memory/lore/squeeze; default all on, unknown ids fail loud)
  - Model-tool registry dispatch is whitelist-only: a disabled or unknown
    skill is refused to the model (fed back as an error it can recover
    from), never a chat failure — unit-tested (registry, gating, dispatch,
    config validation)
- ✅ Jerith Guard (pre-packaged content filter) — outbound screening gate on
  every `/v1/chat/completions` (local forward AND mesh fallback):
  - Per-category PII actions (off/log/redact/block) with spec defaults:
    SSN + payment cards (Luhn-validated) + credentials hard-
    block; emails + phones redacted with stable placeholders
    (`[REDACTED_EMAIL]`…); name/address/medical heuristics opt-in
  - Prompt-injection / jailbreak blocklist + malware-shape request
    blocklist ("this is not a malware creation system")
  - Owner controls: custom regex blocklist, do-not-exfiltrate substrings,
    allowlist of never-PII values
  - Detections logged locally without raw values; stats on `/status`
  - Configured via `[guard]` in config.toml (`guard.enabled = false` to
    switch the whole gate off)
- ✅ Default pre-packaged plugins (registry-managed; OpenClaw lineage kept
  in the names): Juice Compact = web tools (`web_search`, `web_fetch` via
  `[websearch]`), Context Squeeze = bot-side compaction (registers no
  model tools — a lifecycle plugin), plus `memory` and `lore` plugins over
  the node's skill endpoints
- ✅ First-class skills `memory.embed`, `memory.search` (model-facing over
  the gateway's `/v1/skills/memory/*`); `lore.search` likewise
- ✅ First-class skills `vision.describe` / `vision.analyze` — served by
  the gateway's `/v1/skills/vision/*` routes (multimodal forward with the
  model router's vision tier) and mirrored as bot plugins
- 📋 Webhook / event hooks for plugin authors

## 9. Incentives & Economics
- 📋 Native JerithChain integration (default) — import/reuse `jerith_core` mining + RPC flow (Phase 4)
- 📋 Optional Bitcoin, Ethereum, Dogecoin, Monero (user supplies own miner/wallet endpoints) (Phase 4)
- ✅ Interval metering hooks (tokens, expert activations, bandwidth, uptime) wired through the ping loop
- 📋 Automatic JER mining attempt on each served inference (Phase 4)
- ✅ Optional staking + slashing tier (Phase 5, off by default) — live
  against the JerithChain escrow contract (`/stake/lock`, `/stake/unlock`,
  `GET /stake`, `/stake/slash`):
  `[crypto.stake] required_uj` (µJER escrow line for Contributor/Hybrid);
  each mesh process escrows the shortfall exactly once (chain wallet
  auto-created on first 404, persisted `escrowed_uj` marker keeps restarts
  idempotent) and returns it on shutdown; the local ledger trues up to the
  chain's `GET /stake` view every tick, so `/status` shows chain truth
  (the old fake "assume staked" gate seed is gone);
  `slash_percent_per_offense` (default 25 %) penalized per
  bootstrap-verified offense — recorded in the persisted slash ledger
  (`data/stake.json`, audit ring + totals in `/status`) and POSTed as an
  Ed25519-signed `stake/slash` notice; `jerith stake status|lock|unlock`
  drives the same escrow by hand

## 10. Contribution Proof & Anti-Fraud
Phased — see ARCHITECTURE.md §8 for the full layered plan:
- ✅ Health-ping + contribution accounting now (rolling 24 h windows)
- ✅ Cryptographic commitments on expert results: every `ExpertResult` carries
  a SHA-256 commitment over (req_id, expert_id, node_id, input, output);
  the requester recomputes and rejects tampered bodies (accepted body-only
  from pre-Phase-4 peers — no unprovable rejection)
- ✅ Redundant replay verification on a configurable % of remote expert work
  (`verify_percent`, default 10 %, spec range 5–15 %): re-run on an
  independent arbiter (local expert preferred as self-trusted ground truth,
  else the next remote in the plan). Mismatch → both nodes (or the outlier
  alone when the local arbiter decides) flagged as `DishonestResults`,
  evidence signed and submitted to the bootstrap, ambiguous remote-vs-remote
  disputes abort the remaining remote candidates for that req_id
- ✅ Spot-check “ringer” tasks with known answers (`ringer_percent`, default
  10 % of ping ticks): a signed builtin expert query
  (`jerith-builtin/reverse2` — bit-exact f32 reverse-and-scale, seed-varying
  inputs that no runtime/model touches) is sent to one random alive peer; the
  exact answer + tamper-proof commitment are graded locally. Mismatches queue
  `DishonestResults` evidence to the bootstrap; timeouts and infra failures
  are never treated as dishonesty
- 📋 Statistical freerider detection (latency-vs-claim, activation patterns, work ratios) (Phase 4)
- 📋 Reputation system with gossiped signed scores (Phase 4–5)
- 📋 TEE attestation & selective ZK proofs (Phase 5+, research-grade)

## 11. Setup & Onboarding
- ✅ Guided flow: hardware scan → model (with live peer counts, overridable) → memory backend → contribution % (≥25) → optional wallets → enable/disable local vision & embeddings
- ✅ Capability check with clear warnings before selecting a too-large model (`--ok-beyond-capacity` to override)
- 📋 “Recommended meshes for your hardware” suggestion (Phase 1)

## 12. Quality-of-life
- ✅ Graceful offline/degradation: the node starts and runs locally even if the bootstrap is unreachable
- ✅ Local OpenAI-compatible endpoint so stock agent frameworks (Continue.dev, Open WebUI, LangChain…) work day one
- ✅ Resource isolation + emergency kill switch (Phase 1) — `[resources]` in config.toml, enforced by the governor (`isolate.rs`):
  - RAM ceiling: a background governor samples the node's own RSS every 10 s;
    cross a ceiling's soft line (90 %) → the expert hot-set cache is shed
    (LRU, re-read from disk on demand — a disk hit, not a correctness loss);
    cross the hard line with `kill_switch` on → the switch trips: peer-facing
    expert serving, expert dispatch, mesh chat fallback and replication are
    refused (honest refusals — routers walk on) until usage recedes under
    the soft line again (hysteresis, no boundary flapping), then recovery is
    automatic. Local chat, the gateway and the WebUI are never killed
  - VRAM ceiling: same soft/hard semantics over total GPU VRAM used
    (nvidia-smi); unknown VRAM is skipped, never treated as zero
  - CPU pinning: `pin_cores = N` pins the process to the first N logical
    cores (Linux sched_setaffinity, best-effort; other OSes log and continue)
  - Mis-set ceilings (below the 256 MiB sanity floor) fail the node at boot,
    not at breach time; `/status.resources` carries ceilings + live switch
    state (tripped/reason/trips/recovers/cache-sheds) + node RSS
- ✅ Dashboard verification + guard widgets: live Phase 4 counters
  (commitment checks/failures, replay match rate, fraud reports flushed)
  and Jerith Guard counters (screens, blocked, redacted, logged) on the
  node status page, fed from `/status`
- 📋 Dashboard with earnings/expert-hit-rate/latency views (Phase 2–3)
- 📋 Bandwidth caps (soft/hard) and cost estimation (Phase 2)
- ✅ Backup/restore/migrate full node state (Phase 2) — `jerith backup
  create|verify|restore` over one self-describing checksummed container
  (`core::backup`, magic+version / manifest frame / per-entry sha256+payload
  frames, exact-EOF check):
  - create captures config.toml (the node identity key — backups are
    private), bot.toml, persona files and the whole data/ tree (memory
    sqlite, expert demand stats, stake + slash ledgers); expert weights are
    omitted by default (a moved node re-pulls replicas) and ride along with
    --include-experts; per-file 1 GiB / total 2 GiB loud caps; streaming
    two-pass write (memory held to one file's size, not the archive's)
  - verify checks magic/version + manifest count + every entry checksum +
    trailing EOF; a flipped byte or truncated tail is a typed error naming
    the entry, never a silent shorter restore
  - restore verifies first, then writes atomically (tmp+rename); existing
    files survive unless --force; traversal paths (`..`, absolute, empty
    components) refused before a byte is written; copying the backup to a
    new machine (JERITH_HOME) and restoring is the migrate story
  - CLI prints entries/bytes/skips and the privacy note; unit tests cover
    the roundtrip, weight opt-in, checksum/truncation failures, overwrite
    semantics, path-traversal refusal and the no-config refusal
- 📋 Privacy levels: local-only / trusted-peers / full-mesh (Phase 3)
- 📋 Compatibility check for quantization, context length, tokenizer version (Phase 2)
- ✅ Rate limiting, per-peer blocking, content filters (Phase 3):
  - Rate limiting: fixed-window limiter on the public HTTP surfaces (node
    gateway + bootstrap), per-IP anonymous floor + per-Ed25519-identity
    generous lane (NAT scaling); every 429 bumps the process-local metric
    surfaced on /status
  - Per-peer blocking: `[peers] blocked = [...]` node ids (or the shorter
    fingerprint prefixes copied from logs/`/status`). Enforced at BOTH ends:
    inbound, the QUIC RPC server refuses every work-carrying request from a
    listed identity with an honest typed error ("peer … is blocked") and
    counts refusals for /status; outbound, placement never builds a plan
    around them and replication never pulls weights from them. Matching is
    exact-or-prefix both directions; hot-reloadable within one ping tick
    (unit + wire tests: prefix listed, full-id requester refused by identity,
    a different requester served normally on the same server)
  - Content filters: Jerith Guard (§8) screens every outbound chat payload
