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
- ✅ Speculative **prefetching** over measured demand (Phase 3, landed
  2026-10-18; shape below):
  - After this node serves (or dispatches) expert `L<k>.E<n>`, a
    background walk preloads the most-demanded **locally-hosted**
    experts of `L<k+1>` (then `L<k+2>`…, `[experts].prefetch_horizon`,
    default 1) into the byte-budgeted LRU while the requester still
    consumes the current layer's output — the activation pipeline walks
    layer by layer, so the next layer is the natural prediction target.
  - Demand-ranked via `expert_stats.prefetch_candidates` (count desc,
    ties by id — deterministic everywhere); only experts already on
    disk are touched (`prefetch` warms the cache, it never pulls mesh
    weights); `prefetch_per_layer` (default 2) caps candidates.
  - Politeness is structural: one background walk per node at a time
    (`prefetching` guard; a skipped activation costs nothing — the next
    restates demand), the walk is spawned so serve paths never block on
    a disk read, and disabled configs cost one atomic swap.
  - Honest counters: `prefetches` / `prefetch_errors` in the store's
    own stats (hits/misses stay serving-only — a warmed expert's first
    real query shows as an honest cache HIT, the prefetch paid the disk
    cost), surfaced in `/status.experts` with the knobs and on the
    dashboard's expert hit-rate fine line.
  - Knobs are hot-reloadable: `[experts].prefetch_enabled /
    prefetch_per_layer / prefetch_horizon` classify hot in the reload
    plan (`experts.prefetch`) — the hook re-reads live config each walk.
  - Tests: layer parsing + candidate ranking (pure), prefetch counters
    never disturbing the serving counters, and a MeshState-level test
    proving a serve of `L0.*` makes the hosted `L1.*` candidate
    resident with the guard reset afterwards.
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
- ✅ Model switching cleanly disconnects from all current peers, then joins the
  new mesh (landed for real 2026-10-17: a cold `[model]` change in the reload
  plan fires ONE signed `/v1/leave` — the old-model ad is retracted instantly
  instead of riding out the 90 s TTL, the advertise latch closes, and
  `/status` `mesh_exit` shows `left_mesh` + `pending_model`; the restart is the
  join — see §13's model-switch item for the live drill evidence)
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
  (MVP scope note, updated 2026-10-17: the coordinator-side quorum over
  multiple pingers' tallies has since landed — every pinger forwards its
  verified block to the bootstrap and `/v1/peers` publishes the
  corroborated aggregate; see §13 Phase 4. Statistics, not verdicts:
  the reputation-weighted use of that tally stays with Phase 4–5.)

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
- ✅ Model router tiers on the gateway (`core::model_router` + the forward path),
  every *auto* chat classified — an explicit caller model id always wins, the
  auto gate covers the placeholder `jerith-mesh`, an absent model, and the
  advertised base resolved from `/status`:
  - tool-bearing requests → the **tool** tier (`runtime_tool_model`, parse
    reliability first); its forwarded body carries `think: false` so a
    thinking tool model cannot burn the generation budget inside its
    reasoning channel and land empty (`content: ""` at the cap — the
    empty-answer family). The merge rides the tier's *engagement*, not the
    model rewrite: the pinned same-as-base shape (tool pick == base model)
    suppresses the rewrite as a no-op but still gets the merge
    (`auto_tool_tier_merge`).
  - image content → the **vision** tier; complex prompts → the **deep** tier;
    plain chat rides the **PRIMARY chat lane** (`runtime_primary_model`) —
    pin a served model (e.g. a relay-hosted alias) as the everyday chatting
    model without repointing the locally hosted `[model]`, which must stay
    servable for peer dispatch
  - tier keys are cold-reload config (`""` disables a tier); a tier pick
    that equals the caller's model suppresses the rewrite (and the journal
    line) but not the tool tier's think:false merge
- ✅ Discord bot (`crates/bot`, poise/serenity): `/ask` (mesh-peer fallback included,
  screened outbound by Jerith Guard before the gateway call — optional
  `[guard]` in bot.toml),
  - ✅ **Attachment reading** (landed 2026-10-17): the @mention/DM fallback
    turns Discord attachments into one bracketed context block ahead of
    the prompt — the model-visible text stays a String, no multimodal
    plumbing in the pipeline. Images ride the gateway's vision skill
    (data URL → `[image]` description lines; capped 3/message); every
    OTHER file rides the **Docker sandbox** (`crates/bot/src/sandbox.rs`,
    `[sandbox]` in bot.toml: enabled/image/timeout_secs/max_file_mb/
    max_files/max_output_kb — on by default over the repo-baked
    `jerith-sandbox` image built by `deploy/docker/sandbox.Dockerfile`
    from the migrated OpenClaw sandbox base + poppler-utils): one
    disposable container per batch, `--network none`, `--cap-drop ALL`,
    no-new-privileges, read-only rootfs, 16 MiB noexec tmpfs as the only
    writable path, uid 65534, memory/pids/cpu caps, fresh staging volume
    per request, wall-clock kill. The extractor (`sandbox_extract.py`,
    baked into the bot binary via include_str! — filenames are data,
    never executed) reads text/code raw (binary-sniffed), .pdf through
    pdftotext (first 10 pages), the OOXML/ODF family through stdlib
    zipfile+untagging, and archives (.zip/.tar/.tar.gz/tgz/tar.bz2/
    tar.xz) as a bounded listing PLUS the nested text-like members'
    bodies (256 KiB and 6 members per archive; nested archives and
    oversized members honestly skipped, not dropped). Extractions cache
    per Discord attachment id (+ filename/size fallback for re-uploads)
    — a re-forwarded file doesn't re-run the container (BotState cache:
    64 entries, 2 h TTL, successes only; failures stay re-runnable).
    `/read` shows a file's sandbox-extracted content on demand (attach
    with the command; images get their vision description), Discord-
    bounded with an honest truncation tail. Every per-file
    outcome — extracted text, `(binary file, N bytes)`, or the honest
    `(could not read: …)` line — reaches the prompt; nothing drops
    silently. Tests cover argv construction/label-injection, output
    parsing, hostile names, cache hits/eviction, and a LIVE container
    round-trip over an axum stub (text + binary) plus a second-pass
    cache hit against dead URLs; verified live with a real whitepaper PDF
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
  - ✅ Streamed delivery with an honesty contract (`crates/bot/src/api.rs` +
    `chat_fallback.rs`; `[chat] stream`, default on; `[chat] timeout_secs`
    bounds the whole chat): the answer live-types into Discord through the
    gateway's SSE passthrough — and nothing a round yields becomes visible
    before it is real. Tool-bearing rounds fire **no** live events at all:
    streamed tool-intent JSON and echoed tool results (web rows, private
    embedded-memory recall hits) can no longer surface in the chat before a
    reply exists (the recall-slop incident). Toolless rounds are JSON-gated:
    fragments are held while the streamed content still opens `{ "…` (a
    degenerate brace-led intent stub — 193 chars live-typed, then
    RoundStart-cleared when the retry types — read as a double response),
    released the moment the content proves to be prose, and a held round
    that ends degenerate never surfaces; the final formatted reply always
    replaces the streamed surface in one message. Tests:
    `chat_tool_round_fragments_never_reach_the_live_surface`,
    `chat_json_shaped_stream_rounds_never_type_live` (bot),
    `tool_tier_no_op_rewrite_still_merges_think_false` (gateway),
    `auto_tool_tier_merge_follows_the_same_gate_as_the_rewrite` — plus the
    live gateway smoke (plain chat, web tool-loop, follow-up; zero empty
    refusals)
- 🔄 Manifest-registered **external skills** (Phase 3a ✅;
  `crates/bot/src/external.rs`): TOML drops in `~/.jerith-mesh/plugins.d/`
  parse fail-closed at bot start, join the skill registry, gate via
  `external:<id>` in `[plugins] enabled`, and dispatch over signed HTTP
  (HMAC recipe shared with the hooks events; bounded responses; typed
  errors keep the refusal contract). [docs/PLUGIN-SDK.md](docs/PLUGIN-SDK.md)
  carries the full design (transports for Telegram/WhatsApp remain 📋 Phase 3+)

## 8. Plugin / Skill System
- 🔄 OpenClaw-style modular plugin and skill architecture (Phase 3 core ✅;
  see plugins.rs — full reference now in [docs/PLUGINS.md](docs/PLUGINS.md)):
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
- 🔄 Optional Bitcoin, Ethereum, Dogecoin, Monero (user supplies own miner/wallet endpoints) (Phase 4)
  - ✅ Typed refusal landed (2026-10-17): a configured-but-unimplemented optional
    chain surfaces an honest note instead of unlabeled silence —
    `economics::optional_chain_note` feeds the startup banner
    (`economics off — <note>`) and the `/status` `crypto` section
    (`chain` / `mining_enabled` / `wallet_card` / `note` — no fake mining stats
    for an unconfigured chain). Mining stays refused for these chains.
  - ✅ Wallet-card half of the adapters landed (2026-10-17):
    `economics::OptionalChainWallet` probes the operator's OWN chain node
    for a chain-truthful balance — Bitcoin/Dogecoin Core JSON-RPC
    (`getreceivedbyaddress` when `[crypto] wallet_address` is set,
    else the node wallet's `getbalance`; credentials ride the
    `wallet_endpoint` as `user:pass@`), Ethereum `eth_getBalance`
    (hex wei → human ETH + the full raw-wei decimal string), Monero
    wallet-RPC `get_balance` (piconero → XMR). `/status` carries it as
    `crypto.optional_wallet` (`{ok, chain, unit, amount, raw, address}`
    or `{ok:false, error}` — a transport/typed-config failure is an
    honest error, never a fake zero), the banner prints
    `wallet(oc)` at boot, and the note stays accurate in both states
    (mining disabled, wallet card live when configured). Tests: per-chain
    parses, basic-auth from URL, missing/malformed-address typed
    refusals, unreachable-node error, and the live /status integration.
    The mining half remains 📋 (PoW chains don't mesh-mine; ERC-20-style
    balance-only adapters may extend the card later).
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
  drives the same escrow by hand;
  ✅ **the join gate is behavioral** (landed 2026-10-18): with
  `required_uj` > 0 and the gate not clear, the node's QUIC RPC server
  refuses work-carrying requests (expert/chat) with an honest typed
  message naming the numbers (`stake join gate not clear: …`) and
  counts them (`/status.crypto.stake.gate_refusals`) — while Ping keeps
  flowing so the mesh watches a live-but-unqualified node rather than a
  vanished one. This is the escrow half of §10's interlock (the only
  refusal path — reputation demotion stays statistics); wire test:
  gate closed refuses ExpertQuery + answers Ping, escrow-primed ledger
  serves the same request numerically exact

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
  (CSP-pinned, no remote content): earnings = the node's own ~JER wallet card
  (chain-truthful balance from the token-authed chain RPC /balance —
  `[crypto] wallet_address` when set, the miner's credit identity
  (node id) otherwise; "no wallet yet" vs "unreachable" kept honest and
  separate) + mining wins/attempts for this node
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
- ✅ Ops observability: metrics history + per-peer latency view (Phase 5+
  item, landed 2026-10-18) — the metering layer (`core::metrics`) keeps
  two bounded structures beyond the (shorter) recent-record ring:
  5-minute trend buckets (`BUCKET_MS` epoch-aligned, 120 retained = 10 h,
  oldest dropped; a clock-rewound stamp skips the bucket update rather
  than inserting mid-ring) and a per-peer served breakdown (bounded at
  128 peers; an unseen peer beyond the cap folds into one honest
  `_other` row instead of unbounded map growth). Both ride `/v1/metrics`
  (`history`, `peers` — additive keys), and the dashboard renders them:
  a sparkline card (`trend` — the bucket bars, fine line carrying
  last-hour requests / mean latency / mesh share) and a "mesh peers
  served" card (per-peer id, count, mean ms, ok %). 6 new metrics tests
  pin the aggregation, the bounds, the overflow fold and the rewound-
  clock skip; the page's CSP re-pins to the edited script's sha256
  automatically (digests computed from PAGE at first request).
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
- ✅ Privacy levels: local-only / trusted-peers / full-mesh (Phase 3, landed
  2026-10-17): `[privacy] level` in config.toml — `full_mesh` (default,
  today's shape), `trusted_peers` (hops land only on the `[privacy]
  trusted_nodes` allowlist), `local_only` (no prompt ever leaves the node
  via mesh dispatch; the local runtime still serves). Enforcement is a
  policy read at the chat-dispatch choke point (`MeshFallbackChat::
  chat_with_fallback`, re-read every attempt — HOT in the config reload
  classifier, binding within one tick): tier-forbidden hops fail with a
  message that names the policy ("no mesh hop attempted — privacy tier
  …"), so a downed runtime under a strict tier reads as a refusal, not a
  generic outage. `/status` gained a `privacy` section (level,
  trusted_nodes, allowlist_size — null means the full-mesh default).
  Honest boundary: the gate covers prompt-bearing chat hops; expert
  activations and the signed node-level meter blocks (serve/consume
  counts, no prompt content) still ride the mesh at every tier — the
  pipeline-hop gate lands with live pipeline transport. Config values
  are snake_case (`local_only`, `trusted_peers`, `full_mesh`).
- ✅ Compatibility check for model id + quantization (Phase 2, landed
  2026-10-17): the bootstrap's `/v1/models/counts` census now splits per
  quant, and every node runs a ONE-TIME pre-join compat probe on its first
  successful bootstrap leg (when nothing was admitted yet): the typed
  verdict lands in /status `compat` and logs warn —
  - `wrong_quant`: "same model, wrong quant — the mesh runs <model> on
    quant(s) [...]; you run '<quant>', which no peer admits (peers match
    id+quant exactly) `jerith config set model.quant <theirs>` is the fix"
  - `no_peers_on_model`: "you ARE the mesh" for a brand-new network, or
    the models live right now when the fleet runs something else
  - `healthy` when nodes on the exact id+quant are live
  Context length + tokenizer: not carried on ModelId yet (the mesh peering
  gate is id+quant), so that half stays 📋 until the identity gains them
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
  chattering the route plan. M3 landed (2026-10-15):
  `pipeline::plan_route_budget` prices each hop from the measured
  activation ordering (`ExpertStats` via a caller-wired demand hook —
  RTT-floored deadline first, demand-proportional share of the
  remainder), the walk times every attempt and types overruns in
  `RouteResult::over_budget` (reports, never clamps), and the metrics
  snapshot gained a `pipeline` section (routes, over-budget hops, last walk)
  rendered on the dashboard — producer-pending, like the transport.
  The live gate is passed (2026-10-15):
  `tests/pipeline_remote_drill.rs` drives the whole ladder through the
  real QUIC transport — a weight-source node plus three serving nodes
  that boot with empty stores, wire commitments with pinned
  attribution, a hung (mute) owner dying at its deadline and ring-rule
  re-hosted, the honest partial variant, and demand-priced budgets
  against measured latencies. The hop owners' weights travel: each
  container is harvested over the wire (chunked `ExpertFetch` +
  checksummed `put_bytes`) and proven byte-identical, an empty-store
  node refuses to fabricate rather than invent activations, and a
  forged replica with one flipped byte is rejected at the ingest
  checksum. A third hop owner is priced as a simulated wide-area hop
  (180 ms `rtt_of`): its budget is floored on `base + 3×RTT` —
  strictly above the near hops' — and the three-node walk still
  completes with honest empty over-budget. 5/5 repeat runs green.
  M4 landed (2026-10-18): whole-walk budgets are now ENFORCED, not just
  reported — `execute_route_resilient` takes the priced `walk_budget_ms`
  and, once the walk outruns its envelope, skips issuing NEW dials (the
  in-flight hop always lands — enforcement never abandons running work)
  and degrades the remaining hops by name into `RouteResult::degraded`
  with the typed `BUDGET_EXHAUSTED_REASON` plus the overrun ms — hops-
  short graceful degradation instead of an unbounded wall-clock tail.
  A walk that fits its planner-shaped envelope (the `BudgetPlan::walk`
  pricing) completes with `runs_within_budget = true` and untouched
  hops; a walk that fits only its remaining-budget headroom degrades
  exactly the outrunning attempts with a per-attempt typed overrun.
  Unit-tested with real mutations (dials pinned to the walk clock) and
  timing-flake-safe against `started_at` invariants; the honest RTT-
  floored drill route still completes untouched (its M3 leg asserts an
  empty over-budget through real QUIC, which now also proves the
  enforcement clause never misfires on a fitting walk).
  The per-request expert budget (§1's remaining half) landed ON the
  walk 2026-10-18: `pipeline::RequestBudget` caps what ONE request may
  spend on the mesh in work units — `max_remote_attempts` (reassignment
  dials included; default 64, sized to fit a full REASSIGN_BUDGET storm
  plus a route with room to spare) and `max_wire_bytes` (counted as the
  walk ships them, 4 B per f32 in and out; default 256 MiB, a flood
  guard not a squeeze). Enforced inline like the wall: in-flight work
  always lands, past a cap the walk refuses to START new REMOTE dials
  and degrades the remaining REMOTE hops with the typed
  `REQUEST_BUDGET_EXHAUSTED_REASON` — LOCAL hops stay free (the caps
  bound mesh spend, not local compute), so a walk past a tripped cap
  still finishes its local work. This is the exact path speculative
  replays will reuse, so the budget exists before real speculative runs
  do; config knobs are deliberately DEFERRED to transport integration
  (no unread config fields — the cold-field gotcha). Drill-verified
  through real QUIC (criterion 7 bind: a two-remote-attempt cap walks
  A + B live and refuses the far D hop by name, byte-equal to the two-
  hop reference; default caps end the same A → B → D walk untouched).
  Still pending: real model shard contents (the Colibri-side layer
  split — the served containers stay synthetic inside) and real
  inter-region links (the wide-area RTT is simulated through pricing;
  the latencies themselves stay loopback-real). What remains is live
  transport integration (the hop RPC rides `remote_expert_attempt` once
  the Colibri-side layer split provides real runtime weights);
  reordering/preemption is downstream.
  Gotcha learned: everything must survive the existing commitment layer —
  a pipeline hop is several `ExpertResult`-shaped exchanges, so the
  anti-tamper plumbing composes rather than replaces.
- **Speculative decoding + expert budgeting + prefetching (§1)**. The
  prefetching half LANDED 2026-10-18 (see §1 for the full shape), and as
  of the same day the per-request expert budget's WORK-CAP half is in
  (see "Route pricing + enforcement" for `RequestBudget`: per-request
  remote-attempt and wire-byte caps on the resilient walk, typed hops-
  short degradation, local hops never collared, real-QUIC drill-verified).
  Still open: speculative decoding itself (the budget's consumer), and
  the per-request budget's CONFIG surface (deferred until real
  speculative runs ride the transport — the caps exist and are enforced
  at generous defaults meanwhile).
- ✅ **Asynchronous, timeout-tolerant forward passes + graceful degradation
  (§1)** — the mesh CHAT walk landed async 2026-10-18 (see §2's gateway
  tier notes and §13's config surface for the knobs; the expert walk keeps
  its existing plan + replay semantics, its async shape is the speculative
  pre-flight). Shape:
  - `chat_over_quic`'s remote phase keeps up to `chat_hedged_peers`
    (default 3; 0/1 = the legacy serial walk) dials in flight — plan
    order, staggered `chat_hedge_delay_ms` (default 1500 ms) so the best
    candidate is tried first, and dials beyond the cap refill as permits
    free — the plan is still walked, just never one peer at a time.
  - The first attribution-checked ChatResult wins; everything else
    cancels in flight (a peer that was mid-dial served speculatively —
    the same trade the expert pre-flight makes, bounded by cap + budget).
  - Whole-walk budget `chat_walk_budget_ms` (default 45 s): past it the
    walk degrades with a typed error naming the budget and the best
    failure — a mesh of flapping peers can never wedge the gateway.
  - Flapping-peer hysteresis at the router: two consecutive chat
    failures inside a 60 s window sink a peer BEHIND clean peers in the
    walk's order (statistics, never refusal — it stays dialable and one
    clean answer clears it outright). Cancels don't book; real
    answers/failures do.
  - Knobs read at router boot (`chat_hedged_peers` /
    `chat_hedge_delay_ms` / `chat_walk_budget_ms` — classified cold
    `chat_hedge` in the reload plan, so drift warns restart-required),
    verified by the wire drills: a slow-plan-lead peer (8 s sleep) loses
    to a staggered contender at ~300 ms, and the streak/clear semantics
    are unit-pinned.
- ✅ **Privacy levels: local-only / trusted-peers / full-mesh (§12)** —
  landed 2026-10-17 (see §12 for the full shape): `[privacy] level`
  config surface, hot-reloadable policy read at the chat-dispatch choke
  point, typed policy-named refusals, and the `/status` `privacy`
  section. Remaining next step: gating expert-activation hops (the
  pipeline side) when real pipeline transport goes live, and a
  metadata-tier verdict if the meter blocks ever grow prompt-derived
  payloads.
- ✅ **Compatibility check for quantization / context length / tokenizer
  version (§12)** — LANDED (the pre-join half): the mesh loop's first
  successful signed peer fetch takes a live model census from the
  bootstrap and verdicts OUR identity against it once per process
  (`ping::compat_verdict`). Typed, readable shapes: `healthy` (exact
  id+quant peers live), `wrong_quant` — the spec's refusal text lives
  here: "same model, wrong quant — the mesh runs <model> on quant(s)
  [int8 (2 nodes), …]; you run 'x', which no peer admits (peers match
  id+quant exactly)" — and `no_peers_on_model` ("you ARE the mesh" or
  the live alternatives with counts). The verdict journals (WARN on the
  two drift shapes) and renders live on `/status.compat`. The 4 unit
  tests pin each shape.
- **Plugin SDK surfaces (§7): Telegram / WhatsApp / others.** The shape to
  copy already exists: the bot crate's SkillSpec registry + signed webhook
  hooks are the SDK; an adapter only plugs a different transport in front
  of the same loop.
- **Model-switch hot path (§2's model switching)**. Config hot-reload
  deliberately leaves `[model]` cold (echo: model switching is mesh
  re-identity work — disconnect then rejoin). Landed 2026-10-17, the
  graceful half is LIVE (verified end-to-end against the real bootstrap,
  2026-10-17): a cold model change fires ONE signed `/v1/leave` on the
  ping tick — the old-model ad is retracted instantly (no 90 s TTL of a
  peer that no longer serves), the advertise latch closes, and `/status`
  `mesh_exit` carries `left_mesh` + `pending_model`. Restart `jerith run`
  to rejoin under the new model (Rest: in-process rejoin is deliberately
  NOT attempted — routers, tier bindings and every boot-time identity
  path read the config once; the cold-field WARN stays for the restart
  half). Live drill evidence: real node switched models, the journal
  warned `model changed in config — LEFT the old mesh (signed /v1/leave,
  not re-advertising)`, our node id dropped off the public peers list
  within one tick (entries 3→2, no TTL wait), and after restoring the
  config + restart the node rejoined (`left_mesh: false`, peers alive).
  What remains: the rejoin half (a supervisor/watchdog applying the new
  identity in-process) stays 📋 — the honest boundary is documenting
  that the restart is the intended join mechanic.
- **Cold-list watch for the config hot-reload.** Every new hot field must
  genuinely read from the running config at its use site — the reload
  plan classifier (`ping.rs`) keeps the cold list short and deliberate.

### Phase 4 — economics & reputation close the loop

- ✅ **Coordinator-side quorum over contribution measurements** (§2's marked
  MVP gap) — landed 2026-10-17. The cryptographic substrate (per-interval
  signed `measurement` blocks a pinger verifies field-for-field) now has
  its tally: every pinger forwards its verified block to the bootstrap
  (`POST /v1/report-measure`) with BOTH signatures — the subject's own sig
  anchors the figures (a biased pinger can attest delivery, never alter
  numbers), the reporter's attestation binds who corroborated it. The
  bootstrap tallies per subject (deduped per (reporter, block), bounded
  ring) and publishes the aggregate additively in `/v1/peers`
  (`measurements`: distinct_reporters, median, spread, consistent).
  Protection against the single-biased-pinger world: one report is
  explicitly UNQUORUMED (`consistent: null`) — statistics, not verdicts,
  and the no-statistics-slash boundary kept (thresholds act through the
  Phase 5 stake tier). Divergence beyond 10 pp between distinct reporters'
  recent blocks flags `consistent:false` (per-asker lying evidence).
  Wire tests cover the double-signature accept, dedupe, unquorumed,
  consistent, tampered-figure refusal (the subject's own sig catches a
  colluding falsifier) and forged-reporter refusal; live-deployed 2026-
  10-17 with the node's uploads tallied on the real bootstrap
- ✅ **Anti-sybil fraud-report weighting (§10)** (landed 2026-10-18): a
  fraud report only carries score weight once its REPORTER is
  established — identity age in the bootstrap registry
  (`first_seen_ms` ≥ 24h; a never-advertised identity reads as age zero)
  OR visible locked stake on the chain (> 0 µJ via the token-authed
  `GET /stake`, the skin-in-the-game half beating the calendar for
  freshly-minted but  staked operators; chain unreachable = age rule only, no fake grace).
  Zero-weight reports are visible: /health's `sybil.young_fraud_reports`
  counter  (Registry-owned) plus an operator log line name the mass-mint
  shape the weighting exists to expose — and the node's `/status`
  mirrors OUR bootstrap's tally live (`bootstrap` section:
  `reachable` + `young_fraud_reports`; a downed bootstrap reads
  `reachable:false` and never takes /status down with it, chain_probe
  contract). The stake resolver speaks the
  operator's `chain_url` (the :8322 chain-node api) on every path now
  \u2014 run loop, stake CLI, bootstrap live half \u2014 not just the CLI.. Young/unstaked reports are still RECORDED (they
  count toward the corroboration shape and the per-reporter flood
  dedup) but subtract nothing — minting identities to slash a peer's
  score is the shape removed. The report response now carries
  `reporter_established` + `penalty_applied` so operators can see the
  weighting acting live. Wire-tested three ways: young-then-aged
  (zero weight until established, penalty only via established
  corroboration), flood-dedup still holding for a re-reporting young
  reporter, and a stub-chain leg where a fresh reporter with visible
  stake lands the full penalty that its zero-stake neighbor cannot
  publishes its own signing identity + per-subject signed score stamps in
  `/v1/peers`; nodes pin the origin key trust-on-first-use from the
  operator-configured URL (rotation journaled) and re-verify every stamp —
  direct or carried inside ads against OUR pins. Self-stamps, unknown
  origins, forged sigs, and stale stamps drop at the pin wall; the honest
  path lands in a bounded newest-per-subject cache riding ads fleet-wide
  WITHOUT trusting the advertiser. The no-statistics-slash boundary kept:
  thresholds act through the Phase 5 stake tier, not reputation alone.
  Anti-sybil weighting landed the same day (its own entry below).
- ✅ **Reputation-weighted placement (§10, the statistics half)** (landed
  2026-10-17): pin-wall-verified negative scores now feed the dispatch
  planner. Semantics: a verified score ≤ 0 (0 = clean, negatives =
  offenses, floor −100) demotes a peer to `reputation_tier` 1, and
  placement sorts clean peers before penalized ones — penalized peers
  land at the BACK of the plan instead of the front-of-key ordering they
  used to take, so they're only exercised when no clean peer holds the
  expert. That's statistics, never refusal: without the Phase 5 stake
  tier the verified-offense line can only reorder the plan, and the
  typed-refusal boundary (no refusals without stake) is unit-tested.
  Implementation carries it through the `MeshCandidate` type every
  builder + planner site now shares (the fat `(id, rtt, has, alive,
  res, rep)` tuple is gone), and `/status.reputation` renders the live
  verification cache — per-subject score, tier, and the penalized flag
  the planner is acting on right now
- ✅ **Per-node wallet surfaces on-chain** (landed 2026-10-17):
  `economics::WalletCard` probes the token-authed chain RPC /balance live
  in /status (`wallet` section: reachable / exists / balance_uj+jer /
  available_jer / by_address), the dashboard's card reads it with the
  local mined-wins counters as the honest fallback, and the startup
  banner prints the wallet line. Discovery drill bonus: the mesh default
  token file (`keys/api.token`) can drift stale from the co-hosted
  chain node's own key — the miner was silently 401ing until the token
  was resynced; wallet_balance surfaces that as an explicit error
- 🔄 **Optional chains (§9): Bitcoin / Ethereum / Dogecoin / Monero.** The
  config accepts the labels and (since 2026-10-17) supplies the WALLET half:
  `OptionalChainWallet` probes the user's own chain node live
  (`crypto.optional_wallet` in /status; see §9 for the per-chain shapes).
  What remains 📋 for this line: the mining half (PoW chains don't
  mesh-mine — mining stays refused by the typed note) and any
  dotenv-free chain-specific subtleties an operator hits running real
  Core/wallet-rpc endpoints.
- **Chain-side hardening**: chain-RPC token rotation, mine-rate behavior
  vs. the chain's bookkeeping (`total_mined_uj` drift watch), payout and
  escrow edge cases on node exit (the mid-cycle kill drill LANDED
  2026-10-18 — the §2/§13 escrow training notes and “live drills” record
  it: zero re-escrow after SIGKILL, chain truth intact).
  The drift watch + chain smoke half LANDED 2026-10-18:
  `economics::ChainDriftWatch` (process-global, fed by every `/status`
  chain probe) enforces MONOTONICITY on the live bookkeeping —
  `mining.total_mined_uj` and `chain.height` may never shrink between
  live reads; a violation is counted exactly once (the new lower value
  becomes the accepted baseline — no peak-worship flag loop), surfaced
  in the `/status.chain_value.drift` subsection AND journal-errored
  (`chain bookkeeping regressed — monotonicity broken`) next to every
  dashboard reading. The miner's silent-failure half is covered too:
  `MineStats` gained failure_streak/failures totals — a stale api token
  (the wallet-drift incident class) now journal-WARNs every 5
  consecutive failures (`miner failing repeatedly — token drift?`) and
  `/status.crypto` carries attempts/wins/streak/failure counters. The
  env-gated live-chain smoke suite (`JERITH_TEST_CHAIN_URL` → the
  skip-loud test-posture pattern) pins the real chain's READ contracts:
  /status shape + supply invariant, token-less reads REFUSED (401–403),
  `exists:false` /balance for a never-created identity, the /stake view
  contract (numeric µJ fields or honest 4xx), and bookkeeping monotonic
  across two live reads — READ-ONLY by construction (no POST anywhere,
  a smoke run can never move escrow or mint). Live-verified against the
  co-hosted chain node 2026-10-18: 5/5 pass, and the very first run
  caught a contract drift (miners is a COUNT, not a roster — the smoke
  assertion now pins the real shape). Token rotation stays
  runbook-procedural (RUNBOOK §rotation).

### Phase 5+ — trust, attestation, scale

- **Reputation-weighted placement + slashing interlock (§10's Phase 5
  work)**: LANDED, both halves — the placement half (§10 above:
  verified-offense peers drop to the back of the dispatch plan,
  statistics only) and the escrow half (§2's staking entry, landed
  2026-10-18: the join gate is behavioral — with `required_uj` set and
  a short escrow, the node's RPC server typed-refuses work-carrying
  requests while pings keep flowing). The anti-sybil weighting itself
  landed 2026-10-18 (see the §10 entry above).
- ✅ **Live drill: gate + demotion + slash loop on a real staked second
  node** (landed 2026-10-18): two scratch identities (drill-victim node
  + drill-reporter, drill_fraud example driver) against the real
  bootstrap and the real chain. Gate half: with `required_uj` above the
  wallet the boot escrow was chain-typed-rejected ("insufficient
  funds: 500000 < 5100000") and `/status.stake` read `gate_clear:
  false`; at a coverable 0.2 JER the boot lock landed on-chain and the
  gate opened (`staked 200000 µJ \u2265 required`). Demotion half: two
  fresh-mint reports recorded with ZERO weight while the stake-
  established reporter's claim subtracted \u221230 \u2192 halted \u2192 the victim
  stood WITHHELD from `/v1/peers` (and `/health.sybil
  .young_fraud_reports` moved 0 \u2192 2 live). Slash half: the reporter's
  wire-signed `/stake/slash` moved 0.05 JER escrow \u2192 treasury, the
  chain view read back `staked 150000 | slashed 50000`, and the
  victim's gate honestly CLOSED again (150k < 200k required) \u2014 the
  full interlock: slash below the requirement shuts the same gate that
  opened it. Cleanup: both drill escrows unlocked back to the wallets
  (0.05 JER slash + audit trail is the drill's permanent residue),
  drill node stopped. Drill residue: `sybil.young_fraud_reports = 2`
  counts the two zero-weight drill claims, by design. Wiring found by
  the drill and fixed: the stake resolver now uses the operator's
  `chain_url` (:8322, where /stake* lives) everywhere \u2014 run loop,
  stake CLI, and the bootstrap's live half \u2014 and the co-hosted token
  had drifted from the chain node's key again (resynced).
- ✅ **Join gate ENFORCED on the production node** (landed 2026-10-18):
  the production node's chain wallet was created + funded (1.5 JER from
  the owner wallet) and `[crypto.stake] required_uj` set to 1 JER — the
  next boot auto-escrowed exactly that (`stake escrowed on chain
  escrowed_uj=1000000`), `/status.stake` reads `gate_clear: true,
  staked 1000000 µJ, gate_refusals 0`, and the node serves as `hybrid`
  with real stake behind it now. Two fixes landed with it: the boot
  pre-flight was REWRITTEN — the legacy version probed the legacy
  miner-facing chain RPC's /balance as its only evidence and DEMOTED
  the freshly-gated hybrid to `requester` on the very first enforced
  boot despite a live 1 JER escrow — now it reads the same chain truth
  the gate does (the /stake escrow view via the config `chain_url`
  resolver first; the wallet balance only as fallback evidence when
  the escrow view can't clear the gate; demotion only when neither
  clears the requirement), and the dashboard gained a **Sybil fence**
  card rendering the /status.bootstrap sybil mirror (live counter,
  probe-failed and unconfigured states; the page's CSP re-pins to the
  new digests automatically). The runbook (docs/RUNBOOK.md) gained the
  drill-proven chain API token rotation procedure.
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
- ✅ **Ops observability** (Phase 5+ item, landed 2026-10-18 — see §12 for
  the full shape): 5-minute trend buckets (bounded 10 h ring) + a
  per-peer latency breakdown ride `/v1/metrics`, and the dashboard
  renders both — a sparkline card (last-hour requests/mean/mesh share
  over the bucket bars) and a "mesh peers served" card (id, count, mean
  ms, ok % per answering peer).
- **Expert-budget telemetry once budgeting lands (Phase 3)**:
- **Testing posture**: the env-gated live-PG tests (`JERITH_TEST_PG_URL`,
  `JERITH_TEST_PG_URL_NOEXT`) are the pattern to copy for any future
  backend — and the chain node smoke tests ADOPTED it 2026-10-18
  (`JERITH_TEST_CHAIN_URL`, `crates/core/tests/chain_smoke.rs`, read-only
  contracts above) — skip loudly with the reason, never silently.

### Final note

The honest status bar: stay suspicious of any ✅ that cannot show its code
path and its test — this file's markers are audited against the tree
(2026-10-15: pipeline/layer-group, auto-mining, native chain integration
and the Phase 0.5 memory backend were all re-verified with code grep +
live drills), and the roadmap section is where future claims land first —
with the gotchas written down *before* the feature, not after.
