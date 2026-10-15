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
- ✅ Layer-group fallback (Petals-style placement) for high-latency links (Phase 2)
  — expert dispatch plans route through the deterministic layer-ownership ring when
  expert hosting falls short: every layer's owner is derived from the rendezvous ring
  (`ShardMap::layer_ownership`, reassigning only the departed node's layers, just like
  cold-expert sharding), and the QUIC dispatch planner prefers the owning peer for
  each layer group. Placement-only today: the multi-hop residual-pipeline runtime is
  the remaining part and stays on the Phase 3 list (§13)
- 🔄 Speculative decoding + expert budgeting + prefetching (Phase 3)
- 🔄 Asynchronous, timeout-tolerant forward passes; graceful degradation (Phase 3)
- ✅ Dedicated local specialist models:
  - Vision: MiniCPM-V 4.6:1b (never leaves the node unless the user opts in)
  - Embeddings: nomic-embed-text-v2-moe (Matryoshka dims 768/512/256)
  - Both excluded from the 25 % contribution accounting — they are personal infrastructure

## 2. Node Roles & Participation
- ✅ Three roles: **Contributor**, **Requester**, **Hybrid**
- ✅ Minimum **25 %** resource contribution required to be eligible to connect; user-configurable upward
- ✅ Contribution changeable at any time (hot reload via `jerith config set contribution_percent N` —
  the config file is re-read every ping tick: hot fields land within one tick, cold
  ones WARN restart-required, identity fields are refused outright)
- ✅ Model switching cleanly disconnects from all current peers, then joins the new mesh
- ✅ Real-time contribution measurement and cryptographic verification
  (Phase 4): every ping tick diffs the cumulative serve/consume meters
  into an interval share (idle = absence, not 0 %), EWMA-smooths it
  (α=0.5), and drives the contribution-health gate from the *measured*
  ratio against the *declared* one (Requesters never gate; two dry
  intervals in a row drop below a 25 % floor). The last completed
  interval is signed as a `measurement` block inside the health reply
  (`crates/core/src/contrib.rs` + `signing::measurement_payload`) —
  same identity, pinger recomputes it field-for-field, so a node owns
  every number it publishes. Pingers tally verified verdicts per peer:
  two tampered blocks or three consecutive verified shortfalls (measured
  < half the declared, ≥25 % floor) queue signed `Leeching` evidence —
  statistics never slash; `/status.contribution` shows declared vs
  measured side by side. 10 dedicated tests incl. the crypto roundtrip.
  (MVP scope: this measurement is the node's own statement, signed and
  locally verified — a coordinator-side quorum over multiple pingers'
  tallies stays with the Phase 4–5 reputation work.)

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
- ✅ Backend choice at setup: **PostgreSQL+pgvector** or **SQLite** — and the *chosen* store is
  the one actually used (`MemoryHandle`): a postgres choice without a usable store fails
  honest and loud (misconfiguration at boot, refused = never a silent swap to a different
  store); the DSN lives in `config.toml` `memory_dsn` (or `JERITH_MEMORY_DSN`),
  described with credentials redacted everywhere
- ✅ PostgreSQL+pgvector backend (`core::memory::PgMemoryStore`): long-lived tokio-postgres
  connection, `vector(N)` columns pinned to the configured `embedding_dim` (server-side
  uniformity; a dimension-mismatched or layout-mismatched table is refused loudly, never
  reshaped or re-dimmed under the user's data) with an HNSW cosine index built best-effort
  at connect (`vector_cosine_ops`; legacy dimension-free tables keep scrolling scan) and
  index-eligible `<=>` cosine ordering when the extension is present; an honest bytea
  polyfill (packed-f32 blobs, cosine in Rust — the exact SQLite semantics) when the
  extension is unavailable to the connection, always reported in the banner/describe;
  retention cap (10 000 rows, newest survive) inherited from the SQLite store
- ✅ Backend hot-swap at runtime (Phase 0.5 completion): the mesh loop re-reads
  `config.toml` every tick, so `jerith config set memory_backend/memory_dsn` live-swaps
  the store without a restart (the previous store keeps serving until the new one opens;
  a failed swap keeps serving the old one, warns once per 60 s backoff, and never adopts
  the failed config — `/status.memory_store` always names the store actually serving);
  `jerith memory migrate` copies rows across the two stores verbatim (no re-embedding,
  idempotent by text dedupe, `--drop-source` empties the source only after a verified
  copy), and `config set` prints the migration guidance at switch time
- ✅ Local embedding pipeline integrated with the chosen vector store (Phase 0.5): local
  nomic nomic-embed-text-v2-moe via Ollama with Matryoshka truncation (768/512/256), offline
  hashing fallback always labeled in results (`provider` on every hit/skill response — raw
  text goes only to the node's embedder and its own store); identical behavior on both
  backends; env-gated live tests (`JERITH_TEST_PG_URL` / `JERITH_TEST_PG_URL_NOEXT`)
  verified against a real pgvector server in both modes
- ✅ Memory is strictly local — never replicated into the mesh, never distributed in the mesh

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
- ✅ Webhook / event hooks for plugin authors (FEATURES §8 port): the bot
  POSTs signed JSON events fire-and-forget to operator URLs (`[hooks]` in
  bot.toml: `urls`, `secret`, `timeout_secs`, `max_retries`) —
  `skill_invoked` (skill name + ok/error, raw arguments never leave the
  bot), `guard_violation` (public block reason only), `mine_won`
  (`discord_id`, `amount_uj`). HMAC-SHA256 over `"{ts_ms}.{body}"` in
  `x-jerith-signature` (+ `x-jerith-timestamp` for replay windows), linear
  backoff, 1–5 attempts, global in-flight cap; empty `urls` = the whole
  subsystem is a no-op. See hooks.rs (unit + live local-receiver tests:
  signature recomputation, payload schema, retry bound, disabled no-op).

## 9. Incentives & Economics
- ✅ Native JerithChain integration (default chain) — a native Rust flow against the
  local chain node (`economics.rs`, no external runtime): bearer-token RPC mining
  attempts with wallet auto-creation on first use, the phase-5 escrow tier
  (`/stake/lock|unlock`, `GET /stake`, `/stake/slash` with Ed25519-signed slash
  notices) trueing up the chain view every tick, and the `/status` chain probe
  (height, tip, circulating, miners, total mined) that the bot and the dashboard read
- 📋 Optional Bitcoin, Ethereum, Dogecoin, Monero (user supplies own miner/wallet endpoints) (Phase 4)
- ✅ Interval metering hooks (tokens, expert activations, bandwidth, uptime) wired through the ping loop
- ✅ Automatic JER mining attempt on each served inference — the chain's passive-mining
  mechanic reimplemented mesh-side: every served completion rolls `mine_chance_permille`
  (default 120/1000; `0` disables; never blocks the inference path — the chain POST
  runs in a spawned task), and wins/attempts land in the interval meters and on the
  dashboard's earnings card
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
- ✅ Statistical freerider detection (Phase 4) — `freerider.rs`, evidence-not-verdict:
  - Coordinator-side ledger feeds on every COMPLETED peer-served dispatch
    (candidates walked + who answered + latency); failed dispatches and
    local-served runs feed nothing — honest blackouts are never outliers
  - Two conservative signals over a recency window: **latency-vs-pool**
    (peer median > 4× the median of ≥ 3 other peers serving the SAME
    expert, ≥ 3 samples) and **selected-but-absent** (walk-order asks: a
    peer answering ≤ 25 % of its last ≥ 10 asks while ≥ 2 other peers
    answer ≥ 40 % — ranks BEHIND the server never count as asks)
  - Suspicion decays every evaluation (0.93/round on a 0–100 scale); the
    Leeching evidence flag raises once per crossing and re-arms only
    under half the line (hysteresis) — behavior, not one clean round,
    restores trust; the whole ledger is in-memory per process
  - Crossings queue signed `Leeching` evidence through the SAME
    per-kind-cooldown channel as every other fraud report (`verify.rs`
    carries (kind, target) pairs now); the bootstrap weighs it (+20
    reputation) — statistics never slash
  - `freerider_enabled` config gate (default on, hot per tick); running
    suspicion visible on `/status.freerider` (top-5 scores + counters)
- 📋 Reputation system with gossiped signed scores (Phase 4–5)
- 📋 TEE attestation & selective ZK proofs (Phase 5+, research-grade)

## 11. Setup & Onboarding
- ✅ Guided flow: hardware scan → model (with live peer counts, overridable) → memory
  backend (with the DSN prompt when postgres is chosen) → local specialists (interactive
  y/n prompts for vision + embeddings; `--no-vision` / `--no-embeddings` for scripted
  setups — defaults both-on, identical to config.toml's `[specialists]` defaults) →
  contribution % (≥25) → chain choice
- ✅ Capability check with clear warnings before selecting a too-large model (`--ok-beyond-capacity` to override)
- ✅ “Recommended meshes for your hardware” suggestion (Phase 1) — `suggest.rs`:
  the setup flow joins the hardware-capable model list (`detect_capable_models`)
  with the bootstrap's live peer counts (`/v1/models/counts`) into a
  deterministic ranking — joinable-today beats empty (a mesh exists to find
  peers), bigger live meshes first, curated order as the tiebreak so
  suggestions never flip-flop; the top pick is the setup default (falling
  back to the hardware recommendation when the bootstrap is unreachable or
  empty), with the honest per-mesh rationale printed ("N nodes live" vs
  "you would be seeding this mesh"); 5 unit tests

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
- ✅ Dashboard with earnings/expert-hit-rate/latency views (Phase 2–3) — same-origin only
  (CSP-pinned, no remote content): earnings = JER mining wins + attempts for this node
  with the chain probe's height/circulating line (honest "chain view unreachable"
  when the co-hosted chain node is down), expert hit-rate = the expert store's LRU
  cache hit/miss ratio plus the measured activation demand (total/distinct, hot/cold
  split; "off" with no expert store), latency = per-model mean/p-max/ok%/mesh share
  from `/v1/metrics` (busiest model as the headline; honest empty-window state)
- ✅ Bandwidth caps (soft/hard) and cost estimation (Phase 2) — `[resources]
  bandwidth_ceiling_bytes_per_min` in the isolation governor:
  - measures `meters.bytes_relayed` over a rolling window (rate published to
    `/status` only once the window spans ≥ 45 s — unknown ≠ zero, the VRAM
    rule); soft line (90 %) sheds + warns with mesh work running, the hard
    line trips the kill switch exactly like RAM/VRAM and recedes the same
    (hysteresis: recovery only under the soft line, a too-young window never
    holds a trip)
  - failed-loud validation at boot (256 MiB/min sanity floor, `jerith config
    check` gates on it); `/status.resources` shows ceiling + live rate + a
    monthly projection (`bandwidth_est_gib_month`)
  - bandwidth-only ceilings arm the governor (a previously no-op config is
    now a live one); 5 dedicated isolate tests (trip/shed/skip/hold/recover
    + state publication)
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

## 13. Roadmap ahead — everything left, phase by phase

Everything here is a plan, not a promise. Items graduate into the numbered
sections above as they land (statuses there are the ground truth); what this
section adds is *why*, *what already exists to build on*, and the known
gotchas surfaced while building the finished parts.

### Phase 3 — finish the core mesh runtime

- **Multi-hop pipeline forward passes** (completing §1's layer-group item).
  M0 landed (2026-10-15): the ownership ring now reports the *full* map (it
  previously handed planners a self-only map, leaving their remote path dead
  code), and `pipeline::plan_route` builds deterministic multi-hop routes —
  ordered by ring position, consecutive same-owner layers merged into one
  hop, all-local meshes planning as a single hop. The layer-aware dispatch
  planner delegates to it. M1 landed (2026-10-15): `pipeline::execute_route`
  walks those hops with the residual stream chaining hop-to-hop,  per-hop RTT-derived deadlines, and per-hop commitments recomputed against a
  per-hop expert id so a tampering hop is caught downstream (typed
  `RouteError` aborts carry the hop, owner, and completed-hop count).
  M2 landed (2026-10-15): `pipeline::execute_route_resilient` keeps walking
  past owner death — dead hops get their layers re-hosted by the ring's own
  rule (re-pin only the departed owner's layers, `REASSIGN_BUDGET`-bounded),
  unrecoverable losses are honest hops-short partial answers typed in
  `RouteResult::degraded` (the router walks on, mirroring reject-continue),
  and `RouteStability`'s two-line hysteresis keeps a flapping owner from
  chattering the route plan. What
  remains is live transport integration (the hop RPC rides
  `remote_expert_attempt` once the Colibri-side layer split provides real
  runtime weights) and latency budgeting per hop (M3).
  Gotcha learned: everything must survive the existing commitment layer —
  a pipeline hop is several `ExpertResult`-shaped exchanges, so the
  anti-tamper plumbing composes rather than replaces.
- **Speculative decoding + expert budgeting + prefetching (§1)**. The
  measured-demand store (`expert_stats`, activation counts, hot/cold split)
  is the prefetch oracle: predicted next experts preload into the LRU while
  the current layer runs. Open design point: a per-request expert budget
  so one huge speculative run cannot starve other askers.
- **Asynchronous, timeout-tolerant forward passes + graceful degradation
  (§1)**. Today's dispatch is plan-ordered with fallbacks but synchronous
  per attempt; mesh hops need per-hop timeouts, retry walks that respect
  non-idempotency (commitments prevent a tampered node from re-selling the
  same answer, but a dead hop must not wedge the request) and flapping-peer
  hysteresis at the router.
- **Privacy levels: local-only / trusted-peers / full-mesh (§12)**. Memory
  is strictly local already; what is left is a routing-tier policy for
  *prompt* and metadata forwarding (logs, metrics, usage), a config
  surface, and honest refusals at the gateway when a tier forbids a hop.
- **Compatibility check for quantization / context length / tokenizer
  version (§12)**. Model identity already hashes quantization and context
  length; the remaining piece is a pre-join validation against live peers
  with typed, readable refusals ("same model, wrong quant, this many nodes
  could still take you").
- **Plugin SDK surfaces (§7): Telegram / WhatsApp / others.** The shape to
  copy already exists: the bot crate's SkillSpec registry + signed webhook
  hooks are the SDK; an adapter only plugs a different transport in front
  of the same loop.
- **Model-switch hot path (§2's model switching)**. Config hot-reload
  deliberately leaves `[model]` cold (echo: model switching is mesh
  re-identity work — disconnect then rejoin); the remaining piece is the
  graceful-disconnect/rejoin machinery, then the cold-field WARN turns
  into a live apply.
- **Cold-list watch for the config hot-reload.** Every new hot field must
  genuinely read from the running config at its use site — the reload
  plan classifier (`ping.rs`) keeps the cold list short and deliberate.

### Phase 4 — economics & reputation close the loop

- **Coordinator-side quorum over contribution measurements** (§2's marked
  MVP gap). The cryptographic substrate is done: per-interval signed
  `measurement` blocks a pinger re-verifies field-for-field. What remains:
  several peers talling the same interval at the bootstrap, aggregate
  publishing, and protection against a single-biased-pinger world.
- **Gossiped signed reputation (§10)**. Bootstrap-verified offenses today
  score; the gossip layer carries signed scores node-to-node inside ads
  (capped like the discovery rotation), with the no-statistics-slash
  boundary kept: thresholds act through the Phase 5 stake tier, not
  through reputation alone. Anti-sybil: identity age + stake weighting
  before scores count.
- **Per-node wallet surfaces on-chain**. The dashboard shows mining wins
  and the chain-at-large stats; the natural next step is the node's own
  JER balance from the chain RPC (payout address in config), keeping the
  earnings card chain-truthful rather than locally-counted.
- **Optional chains (§9): Bitcoin / Ethereum / Dogecoin / Monero.** The
  config accepts the labels today and nothing else exists — each chain
  needs its miner/wallet adapter against user-supplied endpoints, with
  the *honest* answer being typed refusals until then (no fake mining
  stats for an unconfigured chain).
- **Chain-side hardening**: chain-RPC token rotation, mine-rate behavior
  vs. the chain's bookkeeping (`total_mined_uj` drift watch), payout and
  escrow edge cases on node exit (escrow returned on graceful shutdown —
  what happens when the process is killed mid-cycle deserves its own
  drill).

### Phase 5+ — trust, attestation, scale

- **Reputation-weighted placement + slashing interlock (§10's Phase 5
  work)**: the stake tier exists and is live against the escrow contract;
  wiring verified-offense thresholds into placement preference (and the
  refusals) is the interlock that makes economics bite.
- **TEE attestation & selective ZK proofs (research-grade §10)**. Honest
  ground truth today is the commitment layer + spot-check ringer + signed
  measurement blocks — hardware attestation arrives later, gated on a
  chip-support survey across the fleet rather than a protocol built in a
  vacuum.
- **Vector-store scale beyond the retention cap**: HNSW is approximate —
  exact scans stay correct under the 10 000-row cap; beyond that, an
  IVFFlat alternative (or an optional uncap with index maintenance) plus
  an in-node migration for legacy dimension-free tables (today they run
  hnsw-less with exact scans — documented, silent-skip on the index
  build). Postgres connection pooling only when query concurrency
  actually grows — one client is the right default at this size.
- **Ops observability**: metrics history beyond the single window (ring
  retention on the node, trend lines on the dashboard), per-peer latency
  breakdowns, and expert-budget telemetry once budgeting lands
  (Phase 3).
- **Testing posture**: the env-gated live-PG tests (`JERITH_TEST_PG_URL`,
  `JERITH_TEST_PG_URL_NOEXT`) are the pattern to copy for any future
  backend (chain node smoke tests are the likely next adopter) — skip
  loudly with the reason, never silently.

### Final note

The honest status bar: stay suspicious of any ✅ that cannot show its code
path and its test — this file's markers are audited against the tree
(2026-10-15: pipeline/layer-group, auto-mining, native chain integration
and the Phase 0.5 memory backend were all re-verified with code grep +
live drills), and the roadmap section is where future claims land first —
with the gotchas written down *before* the feature, not after.
