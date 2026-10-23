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
  - ✅ **Real model import — the GGUF layer split** (gguf_import.rs,
    landed 2026-10-23; the store boundary stays
    `run(expert_id, input) -> Vec<f32>`): `jerith experts import <gguf>`
    slices a real GGUF checkpoint's FFN stacks into store containers —
    MoE 3-D stacks become experts `L<n>.E<k>` (SwiGlu kind for the
    stacked llama.cpp mixtral/deepseek family, plain-MLP kind with a
    tagged activation for the nomic-bert family — `--activation`
    overrides), `ffn_gate_inp` becomes the REAL routers `L<n>.router`,
    and dense 2-D FFN layers land as one `L<n>.E0` each — F32/F16/BF16/
    Q4_0/Q8_0 dequantized to f32, one atomic put per container, and an
    `ImportReport` that names every tensor the file carried that the
    importer did NOT convert, with reason (an import that silently
    skipped a tensor would be a lie on the mesh wire). Proven on the
    real nomic-embed-text-v2-moe Q8_0: imported live into a node's own
    store (`~/.jerith-mesh/experts` — 57 hosted) and SERVING real
    forwards; the unit pilot imports the same file and cross-checks EVERY
    imported expert's served math against an F32-dequant oracle at a
    banded relative tolerance (honest 8-bit noise, measured — the pilot
    also caught + fixed a scalar-KV-array seek bug in the reader that
    left the cursor mid-tensor-data)
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
  at connect (`vector_cosine_ops`) and an **IVFFlat fallback** when the server lacks HNSW
  (pgvector < 0.5 — the ANN shape an old server would hit exactly when a store is scaled
  beyond the cap; `lists` sized by pgvector's own guidance: rows/1000, √rows past 1M,
  floor 1) and index-eligible `<=>` cosine ordering when the extension is present; an honest
  bytea polyfill (packed-f32 blobs, cosine in Rust — the exact SQLite semantics) when the
  extension is unavailable to the connection, always reported in the banner/describe;
  retention cap (10 000 rows default, newest survive) inherited from the SQLite store and
  now CONFIGURABLE (next bullet). The in-node **legacy-layout migration** (landed
  2026-10-20, live-verified against real pgvector/pgvector:pg16): a dimension-free legacy
  table — uniform dim, matching config — is rebuilt PINNED inside one transaction at
  connect (rows preserved verbatim: ids, providers, timestamps, vectors; no re-embedding)
  and the ANN index becomes buildable; non-uniform or mismatched-dim tables stay exactly
  where they were (the scan-path shapes), with an honest journal line either way
- ✅ Vector-store scale beyond the retention cap (Phase 5, landed 2026-10-20):
  `memory_max_rows` in config (default 10 000, `0` = uncapped) — the deliberate scaling
  decision the store honors with index maintenance: the pgvector lane keeps/creates its
  HNSW (or IVFFlat-fallback) index so `<=>` stays index-eligible at any size, while the
  SQLite lane honestly stays a full exact scan (no native vector index — the uncapped
  SQLite choice is journaled as a loud WARN at config-validate and printed at `config set`);
  the cap rides the memory hot-swap triple (a change live-swaps within one ping tick,
  pruning follows from the NEXT insert — nothing is deleted at swap time), and negative
  caps are refused at every door
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
  carries the full design; packaging shipped too — Phase 3d ✅
  (2026-10-22): `crates/plugin-template/` (`SkillServer` lib + example
  binary + copyable `plugin.toml`) and `sdk/python/jerith_plugin.py`
  (stdlib-only runner), both serving the ONE signed-dispatch contract
  the parser does. Telegram's transport is live (§18); WhatsApp remains
  push-only via `webinbox` (public webhook URL required — 📋 open lane)

## 8. Plugin / Skill System
- 🔄 OpenClaw-style modular plugin and skill architecture (Phase 3 core ✅;
  see plugins.rs — full reference now in [docs/PLUGINS.md](docs/PLUGINS.md)):
  - Capabilities are **plugins** owning dot-named **skills** (`web_search`,
    `memory.embed`, …): one `SkillSpec` table entry per skill derives the
    OpenAI tool declaration, the dispatch whitelist and the gating — the
    chat tool loop reads only from the registry (`crates/bot/src/plugins.rs`)
  - Owners gate per plugin via bot.toml `[plugins] enabled = [...]`
    (websearch/memory/lore/squeeze; default all on, unknown ids fail loud)
  - **Packaging (Phase 3d ✅, 2026-10-22):** the out-of-process author
    path ships as two SDK flavors over ONE signed-dispatch contract —
    Rust: `crates/plugin-template/` (the `SkillServer` service builder
    + `jerith-plugin-example` demo + copyable `plugin.toml`); Python:
    `sdk/python/jerith_plugin.py` (stdlib-only single file). CI pins
    template ⇄ parser agreement; RUNBOOK has the five-minute lane.
    Second live consumer ✅ **the mesh pet** (`crates/pet-plugin/`,
    2026-10-23): a persistent void-companion over the same contract
    (`pet.look/talk/feed/play/rename`, state decayed in real time and
    persisted across restarts; `external:pet` gate row). Its build
    also fixed the example's latent manifest-parse boot bug
    (document TOML needs `toml::from_str::<toml::Table>`, now pinned
    on both crates). The pet also keeps a DIARY (2026-10-23): stage
    landings and long silent drifts (≥ 2 days integrated between
    sightings) are queued into the state file and delivered by the
    wire half as one .eml each into the house's `[ai]` mailbox —
    mailhook-stamped filenames, RFC-5322 Date (std-only civil math),
    `X-Pet-Kind/Stage` headers, written before the save and requeued
    on delivery failure so a mail is sent once or not lost. Third live consumer ✅ **the house**
    (`crates/house-plugin/`, 2026-10-23): `house.status` answers the
    how-are-you question over the signed contract — the bot unit's
    state, the gateway's health + mesh contribution numbers, and one
    recent-warn scan, degraded fields marked in the card rather than
    failing the call; `house.journal` reads the bot's recent journal
    lines. `JERITH_GATEWAY`/`JERITH_BOT_UNIT` bindable; the bot pin
    test (manifest ⇄ drop parser) holds for all three drops.
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
  do; the CONFIG surface LANDED 2026-10-23 (before transport
  integration, and with no unread fields — the live dispatch walk
  enforces it, see below). Still NOT carried by the ChatPath/chat lane,
  whose budget shape stays ChatResult-flavored. Drill-verified
  through real QUIC (criterion 7 bind: a two-remote-attempt cap walks
  A + B live and refuses the far D hop by name, byte-equal to the two-
  hop reference; default caps end the same A → B → D walk untouched).
  Still pending: mesh-side live transport wiring (the shard
  contents themselves are REAL now — the gguf import, §1: the real
  nomic Q8_0 runs in this node's own store; what remains is the hop RPC
  riding `remote_expert_attempt` over the mesh inter-region links, whose
  wide-area RTT is simulated through pricing while the latencies stay
  loopback-real); reordering/preemption is downstream.
  The budget's LIVE enforcement landed on the real dispatch walk
  2026-10-23: `execute_over_quic_speculative_attributed` keeps a
  per-request `RequestLedger` (every request-cause remote dial counts an
  attempt — pre-flight counted once at creation, the replay-verify
  arbiter dial included — and wire bytes count as the request ships
  them: 4 B per f32 in at dial start, out when the answer lands).
  Enforcement is inline like the wall: in-flight work always lands, past
  a cap the walk refuses to START new remote dials and ends with the
  typed `REQUEST_BUDGET_EXHAUSTED_REASON` naming the refused hops + the
  best accumulated failure; local hops stay free. The pre-flight is
  speculative extra work — when the ledger has no room for it the
  headroom rule drops it (honest no-op) while the REAL walk keeps its
  dial; the replay arbiter past a cap just fails open (no arbiter ⇒
  Trust, we never reject work we cannot prove wrong).
  Config: `[experts].max_remote_attempts` / `max_wire_bytes` (0 =
  uncapped, the chat_walk convention) → `request_budget()` typed
  conversion; read LIVE at every dispatch (ping.rs) so caps are hot —
  reload plan classifies them `experts.request_budget` hot. Defaults
  equal the pipeline's enforced defaults (64 attempts / 256 MiB wire),
  so a config without the block behaves exactly as before. Tests:
  knob defaults + 0→uncapped mapping, reload classification, and three
  real-QUIC walks (attempt cap refuses the second hop by name with the
  typed reason, wire cap trips after bytes moved, preflight-only drop
  still COMPLETES the walk).
  Gotcha learned: everything must survive the existing commitment layer —
  a pipeline hop is several `ExpertResult`-shaped exchanges, so the
  anti-tamper plumbing composes rather than replaces.
- **Speculative decoding + expert budgeting + prefetching (§1)**. The
  prefetching half LANDED 2026-10-18 (see §1 for the full shape), and as
  of the same day the per-request expert budget's WORK-CAP half is in
  (see "Route pricing + enforcement" for `RequestBudget`: per-request
  remote-attempt and wire-byte caps on the resilient walk, typed hops-
  short degradation, local hops never collared, real-QUIC drill-verified).
  The SPECULATIVE half LANDED 2026-10-18 (transport-side warm pre-pay):
  `pipeline::WalkLimits` carries an optional `prefetch_next` hook —
  while each remote hop dials, the NEXT hop's containers travel into the
  walker's local cache over the real transport (a genuinely concurrent
  `tokio` task whose bytes join the SAME per-request ledger), and the
  caller's reassign resolution can now fall back to a LOCAL hop — the
  self-host reassignment — so a dead owner's layers upgrade from
  "degraded" to "served-from-warm-cache" (the completion the prefetch
  exists to enable). The headroom rule is a hard invariant: prefetches
  fire only when the remaining byte budget covers ≥2× the estimate, and
  a tripped or absent-caps state disables them — speculation never
  starves the real walk. Telemetry: `RouteResult.wire_bytes_moved` (the
  single ledger: transport bytes + warm pulls) and `RouteResult.
  prefetches` (landed pulls; failures are honest no-ops). Unit-tested
  (fire/accrue exactness, headroom-off, self-host complements the
  degraded path) and drill-proven through real QUIC (criterion 8 bound:
  two warm pulls + one mute-owner deadline death + a self-hosted local
  run = a byte-equal-to-reference COMPLETION with the ledger exactly
  5×transport + the warm bytes). The speculative DECODING half LANDED
  2026-10-19 as the chat-path lane (`gateway.rs`, config
  `[runtime] runtime_spec_decode = { num_speculative_tokens }`): the
  DECODING itself stays engine-side by design — draft-and-verify needs
  the logits, which the engine (Colibri's MTP header, llama.cpp `-md`,
  vLLM `speculative_config`) holds — while the mesh owns the honest
  wherever-it-must-live part: the forward carries a `spec_decode` hint
  when (and only when) a ONE-TIME synthetic admission probe proved the
  runtime accepts the field (a clean probe round-trip guarantees an
  infra hiccup is never misread as a refusal; a refusal locks the hint
  inert for the process lifetime so user traffic never faces an
  unproven engine field), and /status `spec_decode` measures BOTH lanes
  (`hinted`/`plain` tokens/sec) publishing a `speedup` ratio only past
  5 measured records each — statistics, never a one-shot verdict.
  Wired cold (reload plan `runtime_spec_decode` restarts like the other
  runtime keys). Tests: hint-merge gating (refused/zero/admitted),
  probe classifications, stats math with the ≥5 gate, plus the live
  mock-runtime drill (probe admits → hint rides forward 2+, stats
  fill; strict runtime 400s the field → refused → user traffic stays
  clean). The per-request budget's config surface LANDED 2026-10-23
  (above), and the budget now rides the MESH CHAT lane on the same live
  read: `chat_over_quic` takes the budget the fallback already built and
  refuses NEW chat dials past a cap by name with
  `RouteResult::REQUEST_BUDGET_EXHAUSTED_REASON` — in-flight work always
  lands, local runtime hops stay free, and a budget refusal never books
  the refused peer's flapping streak (no fault of theirs). Drill-
  verified over real QUIC on both edges: a honestly-failing lead books
  its real streak while the budget-refused spare books nothing, and a
  sub-payload wire cap refuses the dial before it starts.
  Draft-expert scheduling (the gguf store has real routers + experts to
  draft from — the engine still owns draft-and-verify) started landing
  phase 1 of 3 on 2026-10-23: `[experts].draft_priority` (default off;
  hot — the dispatch loop re-reads it with the budget every dispatch,
  and /status.experts publishes it). When on, the FIRST remote hop leads
  by pure RTT among reputation-gated qualified peers — the draft must
  not wait behind a slower, resource-richer peer; resources keep
  ordering the TAIL (retries land resource-first), and a
  verified-offense peer never heads the hop. Named rule (draft_head),
  planner unit tests (knob off = byte-for-byte the Phase-1 order; on =
  head-by-RTT, tail intact; offender gated) plus the real-QUIC drill:
  with local weights down, a resource-reported far peer serves knob-off
  (attribution ×7) and the NEAREST named peer serves knob-on (×3) — the
  fall-through stays intact.
  Phase 2 — the DRAFT HOT SET — LANDED 2026-10-23: `[experts].
  draft_hot_percent` (0 = off; hot — the replication round re-derives
  pins and plan membership from the live knob every interval, and
  /status.experts publishes `draft_hot_percent` + `draft_pinned`). The
  tier is % of hosted experts drawn deterministically LOWEST-LAYER-
  FIRST (the draft pass re-runs the front of the model every decode
  step; dense E0 before numbered experts, non-layer synthetic ids last,
  numeric layer order, >50 clamps down). Its members are (a) PINNED
  always-resident: `apply_draft_pins` eagerly loads them when the pin
  lands and `pinned` ids are never eviction victims — the governor's
  FULL shed and the LRU's own budget pressure both stop at them (pins
  may exceed the budget; that is the operator's stated ask) — and lock
  order is cache→pinned everywhere (the inverse order would deadlock
  load against shed); (b) replicated LIKE HOT EXPERTS: the round plans
  draft members FIRST (their fetches survive the per-round cap) with
  the demand hot set deduped behind them. An off/knob=0 round derives
  an empty tier, which fades stale pins within one interval. Tests:
  draft_set ordering/clamping, pin survival through full shed +
  pressure + clear, knob 0/default/TOML roundtrip, reload classifies
  `experts.draft_hot` hot; live drill: dev-e2e.sh E2E_DRAFT_HOT=1 —
  after node A replicates node B's expert the NEXT round pins it
  resident (draft_pinned / resident_bytes proven from /status).  Phase 3 — BUDGETED SPECULATION — LANDED 2026-10-23: `[experts].
  spec_max_wire_bytes` (0/absent = invite-only: byte-identical parities
  with the honest ledger, the default), together with the honest wire
  cap, derives `request_spec_budget()` — typed like `request_budget()`
  — which the dispatch loop installs on the router (`set_spec_budget`)
  and re-reads LIVE on the same read as the honest budget (reload plan
  `experts.spec_budget` hot; 0 maps to uncapped like the honest knob).
  Enforced at the TWO speculative surfaces, always BY ITS OWN ledger so
  a tightened spec knob can never starve the real walk: the pipeline
  `WalkLimits` prefetch gate refuses speculative prefetches the honest
  ledger would still allow, and `execute_over_quic_speculative_
  attributed` pre-checks its pre-flight dial at walk start and drops it
  (honest no-op) when the spec ledger has no room — the honest ledger's
  room and caps are untouched, no typed refusal leaks into the answer,
  and the walk still dials and completes. A spec entanglement past the
  cap refuses before it invites any peer's work. Tests: knob
  defaults + 0→uncapped mapping, reload classification, pipeline spec-
  gate unit test, and the real-QUIC drill (tightened spec sub-budget
  drops ONLY the pre-flight — both legs complete byte-equal answers,
  honest knobs unplugged).
  Also landed 2026-10-23: dev-e2e.sh PORT PREFLIGHT — any of the six
  run ports (bootstrap/runtime TCP, gateways TCP, QUIC rpc UDP) held by
  another process aborts BEFORE the build with the exact
  E2E_*_PORT override to set (E2E_CHECK_PORTS=0 restores the old
  silent-garbage behavior); a live 2026-10-23 run reproduced the old
  5b-replication-timeout failure mode (live node on 8900/9000).
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
- **Model-switch hot path (§2's model switching)** — LANDED, both halves.
  The graceful leave half landed 2026-10-17 (verified end-to-end against
  the real bootstrap): a model change in config fires a signed `/v1/leave`
  on the ping tick — the old-model ad is retracted instantly (no 90 s TTL
  of a peer that no longer serves), the advertise latch closes, and
  `/status` `mesh_exit` carries the pending rejoin target. The rejoin
  half LANDED 2026-10-22 as the SUPERVISOR (`ping.rs`, no restart): the
  leave must be ACCEPTED by a bootstrap before re-identifying (a downed
  bootstrap reads honestly as "old ad expires by TTL — rejoin waits";
  leaves retry every tick until one accepts, so no dual-mesh presence is
  ever risked between retraction and rejoin), then the new model is
  adopted IN-PROCESS — the registry gate rebuilds next tick (it already
  re-reads the model each iteration), the compat census re-checks once
  (explicit reset), and the next advertise tick joins the new model's
  mesh. `/status.mesh_exit` gained `model_switches` (completed in-process
  switches, so the round-trips are auditable). The WARN log line is
  honestly reworded: `config model changed — the §2 supervisor handles
  it`. One live drill evidence entry from 2026-10-17 still stands (real
  node switched models, our node id dropped off the public peers list
  within one tick — entries 3→2, no TTL wait — and the node rejoined
  with `left_mesh: false`, peers alive). Scope note, still honest: the
  supervisor's rejoin applies the MESH REGISTRY identity (ads, census,
  peers, compat). The runtime engine's serving-side models (`[runtime]`)
  remain boot-bound in the reload plan, untouched — a serving-language
  model change stays a restart.
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
  vacuum.- ✅ **Vector-store scale beyond the retention cap** (landed 2026-10-20,
  see §6 for the fielded shape): the HNSW-is-approximate concern is
  resolved by keeping exact scans as the DEFAULT retention contract —
  `memory_max_rows` (config, default 10 000; `0` = uncapped) makes the
  beyond-cap world an operator DELIBERATE choice, and the choice comes
  with its machinery: the pgvector lane maintains its ANN index end to
  end (HNSW, with the IVFFlat fallback sized by pgvector's guidance for
  old servers that never had 0.5, so `<=>` stays index-eligible), the
  SQLite lane honestly says stay-small (WARN + printed note; a full scan
  is the whole engine), and legacy dimension-free pgvector tables — the
  silent skip the old index build left behind — now migrate in-node: at
  connect, a uniform-dim legacy table matching the config dim is rebuilt
  pinned transactionally (rows verbatim, sequence re-linked) and the ANN
  index becomes buildable. Live-verified 2026-10-20 against real
  `pgvector/pgvector:pg16`: the rebuild pinned the server-side column
  type (`vector(64)` probed via pg_attribute), ids/providers/timestamps
  survived verbatim, indexed search served the migrated store, and a
  mismatched-dim legacy table stayed untouched on the scan path. The
  cap rides the memory hot-swap triple (pruning follows the NEXT
  insert; swap-time deletes nothing). Postgres connection pooling stays
  deliberately NOT done — one client is the right default at this size;
  pooling only when query concurrency actually grows.
- ✅ **Ops observability** (Phase 5+ item, landed 2026-10-18 — see §12 for
  the full shape): 5-minute trend buckets (bounded 10 h ring) + a
  per-peer latency breakdown ride `/v1/metrics`, and the dashboard
  renders both — a sparkline card (last-hour requests/mean/mesh share
  over the bucket bars) and a "mesh peers served" card (id, count, mean
  ms, ok % per answering peer).
- **Expert-budget telemetry once budgeting lands (Phase 3)**: the walk
  budgets (chat/remote caps, §13's enforced inline spend guards) exist
  today but surface nothing about how they constrict dispatch — the
  telemetry half (refusals-by-cap counters in `/v1/metrics`, per-peer
  spend vs. disallowed-by-budget ratios) waits until real expert
  transport rides the caps, the same gate as the §17.3 escalation
  package; data collected before a real workload is noise.
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

## 14. Multi-bot coordination + workflow recording (2026-10-19)

Three surfaces, one shared store (`crates/core/src/coordinator.rs`, SQLite at
`<data>/coordinator.db`, served by every node gateway at
`/v1/coordinator/*` — merge into the gateway router when
`[coordinator] enabled`, default on):

- **The mesh coordinator** (the abstraction layer): a named-worker
  registry, a DEDICATED HAND-OFF CHANNEL (FIFO work queue with claim
  leases — a worker that dies mid-job loses the lease and the job
  requeues; removal expires a worker's backlog), and a SHARED MEMORY POOL
  (namespaced context blobs; delegators stage context under a key and
  hand only the key name to the worker). Tests: `coordinator` store +
  HTTP wire surface (`CoordinatorStore` + `/v1/coordinator/*`).
- **Multi-bot coordination** (bots-side, `crates/bot/src/coordinator.rs`):
  users shape distinct named bots in owner DMs — `bot.worker.spawn` pins
  each a model (`qwen3:8b`, any gateway model id — local weights or
  remote tiers), `bot.delegate` posts onto the hand-off channel, a
  per-worker claim loop solves through the gateway with THAT worker's
  model, and a worker's answer may end with one
  `HANDOFF> <worker> | <subject>` line that spawns a depth-1 chained
  task (`[chained]` marker) — workers hand each other work with no human
  pasting context between turns. The registry is the source of truth: a
  rejoin updates the running loop the next tick; removal exits it.
- **Workflow recording** (`crates/bot/src/workflows.rs`, plugin
  `workflows`, owner-DM-only like `external:gcp` and `coordinator`):
  `workflow.record.start` opens the REAL chrome (headed, on the
  configured display) holding a CDP session with an injected recorder —
  the OWNER does the task once and every click/field/Enter/page change
  lands as a structured step; `workflow.record.stop` coalesces
  (keystrokes → one `type` with the final value) and saves the playbook
  (0600 JSON). `workflow.run` replays HEADLESSLY through the same
  engine: native value setter + input events for fields (framework-safe
  typing), Input-domain mouse press/release at the element center (real
  activation), recorded `goto`s healing mid-flow page changes. Bounded:
  ≤200 steps, 1k-char selectors, per-run wall clock, typed per-step
  errors. Live proof path: `cargo test -p jerith-mesh-bot
  chrome_cdp_live_smoke -- --ignored` (spawn → devtools-ws →
  Runtime.evaluate → Input click → typed failures).

Trust boundary: `OWNER_ONLY_PLUGINS` now covers `gcp` (external drop),
`coordinator` and `workflows` (in-tree) — none of these ride
`[plugins] enabled` (refused at load); their skills join a session's
registry ONLY in owner-DM conversations, enforced twice (config load +
registry owner-mode gate at dispatch `enabled`). Workflows carry
recorded credentials → JSON files are 0600; the coordinator carries
mesh identities and host compute → owner-DM only.


---

## 15. Human-in-the-Loop approval bridge (2026-10-19)

`crates/bot/src/approval.rs` — the agent carries a task right up to the
critical side effect (an API payment, an email send, a purchase flow, a
privileged external skill) and parks **exactly that step**:

```json
{ "skill": "approval.request",
  "args": { "summary": "Charge the $24 provider invoice to Shopify",
            "kind": "http_call", "method": "POST",
            "url": "https://api.shopified.example/v1/pay",
            "body": {"amount": 2400} } }
```

The request is persisted (`~/.jerith-mesh/bot/approvals/<id>.json`,
0600, atomically rewritten on every transition — the on-disk checkpoint
of the paused state) and the model is answered **immediately** with a
pending id: the tool loop never blocks on a human. State machine:
`Pending → (Approve|Correct|Deny→Cancelled|TTL→Expired) → Done/Failed`,
with a sweeper repairing Approved-but-unspawned orphans after a restart
(the deterministic, typed local analogue of a LangGraph
`interrupt_before` node).

 Certified live 2026-10-19: the Telegram pair is configured in bot.toml
(token + chat id), `getMe` validated the token, a real card was
confirmed delivered to the owner's chat via the API (send + inline
keyboard), the poller runs clean in the journal, and the FULL CALLBACK
ROUND TRIP is verified live: harness card `T56VZVSK` → owner ✔ press →
one-shot resolve → deferred `memory.embed` executed through the live
registry → checkpoint flipped `done` (result recorded in the 0600
store). The shipped harness is the `#[ignore]`d
`approval::tests::live_telegram_roundtrip` (maintenance runs only)
that fired the very card that proved it.

Notifications fan out to wherever the operator reads:

| Channel | Shape | Resolution |
|---|---|---|
| Discord owner DM | card + **Approve ✓ / Deny ✕** buttons | button press, identity-checked (owner id only) |
| Telegram | inline keyboard via `@BotFather` token | button callback or `approve <id> [json]` / `deny <id>` (chat-id-checked), 25 s long-poll `getUpdates` — zero open ports; when `§18`'s transport shares the token, the transport is the only poller and resolves these shapes through its stream |
| Webhooks | signed `approval_requested` / `approval_resolved` events (hooks.rs HMAC) + optional raw `[approvals] notify_webhooks` card bodies | Slack bridge / ntfy / WhatsApp relays see the request; resolution stays on the approved channels |

Correct-resolution semantics: skill args are **merged** (patch the one
field you spotted, keep the recorded call), `http_call` bodies are
**replaced**, and an `http_call`'s method/url are frozen — retargeting
is a NEW approval card, never a silent redirect.

Trust model (fail-closed):
- **Only the configured approver can resolve** — the owner's Discord id
  or the configured Telegram chat id; everyone else's presses get a
  typed refusal, and `resolve()` re-checks identity as defence in
  depth.
- **Deferred skills are down-scoped by construction**: at REQUEST time
  the named skill must be dispatchable in the NON-owner-privileged
  registry. Owner-only actions (gcp, coordinator, workflows-as-gate)
  execute live in owner-DMs instead — the bridge never escalates a
  plain session's privileges. EXECUTION at APPROVE time does run in an
  owner-privileged registry (the approver IS the owner).
- **`http_call` is a double-validated allowlist lane**: `https://`
  only, GET/POST/PUT/PATCH only, exact-host `[approvals]
  http_call_allowlist`, body ≤ 64 KiB, response snippet capped at 200
  chars — validated at deferral time AND again when the trigger is
  pulled.
- Request ids: 8-glyph unguessable codes (I/l/O/0/1 excluded, typed
  safely on a phone); resolution is one-shot and race-safe (Discord +
  Telegram racing the same card: first wins, everything after is a
  stable no-op — **never** a second execution).
- Event payloads never carry model args verbatim (channels are relays;
  webhook receivers are third parties) — the summary + a coarse action
  description only.

Config: `[approvals]` in bot.toml (enabled, discord_owner_dm, telegram
pair, ttl_minutes 1–2880, http_call_allowlist, notify_webhooks,
prune_resolved_hours). New plugin `approvals` joins
`PREPACKAGED_PLUGINS` with `approval.request / approval.status /
approval.cancel`; hooks gained `approval_requested` / `approval_resolved`
event types.

Tests: 14 in-module (id grammar, effect-arg-blind describe, patch
merge/replace, down-scope refusal, deny roundtrip one-shot +
identity-gated, correct-path persistence, TTL expiry, cancel-before-
resolution, 0600 store roundtrip, card format, config fail-loud —
14 green, plus the #[ignore]d live Telegram round-trip harness above.

## 16. Rust-native multi-model task router (2026-10-19)

`crates/bot/src/router.rs` — the fresh local lineup (qwen3:0.6b,
bonsai-2:2b, tev1:0.8b, nemotron-3-nano:4b, qwen3:4b — new downloads
2026-10-19) turns one-model-fits-all into a waste: intent sorting is a
sub-second 0.6B job, code generation wants the 4B. The router maps a
prompt onto a plain Rust enum and swaps the LANE for that chat:

- classification = one direct Ollama `/api/chat` call (router model
  `qwen3:0.6b`, 10 s cap) with **the answer's JSON schema passed as
  `format`** — sampler-level grammar, so malformed JSON is physically
  impossible; the answer then parses through serde into
  `TaskKind { CodeGeneration, DataExtraction, CloudEscalation,
  IntentRouting }`. Unknown classes are a compile-time impossibility,
  and an off-vocabulary answer is refused at the enum wall even if a
  runtime loosens its own format support.
- lane map → bare runtime ids only (`qwen3:4b` for code, `qwen3:8b`
  for extraction/escalation, intent keeps the runtime default). The
  live-learned rule from 2026-10-17 is encoded: bare ids only —
  provider-prefixed (`ollama:…`, `openrouter/…`) lanes are refused,
  because the runtime's `model_name` vocabulary is bare.
- OPTIMIZER, never a dependency: `[router] enabled` defaults false
  (opt-in), and any failure (ollama down, timeout, schema break)
  degrades to no-swap — the runtime's normal tiering continues
  untouched. The router never picks a remote/cloud model; escalation
  stays explicit.

Wired into the chat path at `api.rs` (`maybe_lane_model`, before the
payload model is chosen) with a `tracing::debug` carrying class +
constrained reason. Config: `[router]` + `[router.lanes]` in bot.toml.

## 17. Staged roadmap — universal driving, macro-tools, escalation, shadowing

What this release establishes, and what stages next on the same rails:

1. **Universal interface driving (docker VM + front-end control) —
   LANDED (2026-10-20, `images/driver-vm/` + `workflows.rs`
   `[workflows.driver_vm]`)**: the dockerized desktop exists — Xvfb +
   a REAL window-manager bootstrap (openbox; WM-less X is the typed
   fallback) + an x11vnc preview the owner watches through a
   loopback-mapped port + chromium + a socat CDP relay (9223 → chrome's
   loopback 9222, because headful chromium binds DevTools
   loopback-only and IGNORES the bind-address flag — debugged live).
   Record AND replay can target it (`launch_vm_session`; shadow wins
   for replay when both lanes are enabled — headless-over-headful is
   the cheaper contract; the VM always owns record when enabled, so
   the owner drives windows INSIDE the VM instead of on the host
   display). Same ownership discipline as the §17.4 shadow: unique
   `--rm` container, prefixed sweep, shutdown removes BY NAME.
   E2E verified live: a real playbook saved through the real store,
   run end-to-end through `run()` with the VM config (playbook →
   container → WM bootstrap → relay → CDP → per-step ✓ → named
   removal), 7.9 s — and again against a REAL LONG-TAIL APP (the
   recipe exercised): `images/driver-vm-longtail/` bakes sqlite-web
   onto the layout, the replay typed through the app's real form and
   the row PERSISTED host-side through the manifest-rw mount (9.3 s;
   `live_workflow_replay_inside_longtail`). Contracts the fires
   earned: `[workflows.driver_vm] boot` (the engine's bootstrap chain
   is configurable — a long-tail image bakes its own wrapper or its
   app never runs; the debugged hole, entity-named by the typed
   failure's new `page: <href> :: <text>` triage payload), VM lane
   manifest mounts (shared §17.4 parse rules), a bounded 5 s selector
   wait in replay click/type (just-started apps answer mid-load), and
   CDP attaching only to real http(s) page targets (chrome's own
   omnibox targets are type-page too — order-dependent attach meant
   querying the wrong page forever). Approval-gated pairing rule:
   unchanged by
   construction and unit-pinned (`workflow_targets_never_outgrow_the
   _owner_gate` — the workflows plugin is owner-only, the §15 bridge
   is the approval surface). Long-tail software as ephemeral UI
   targets now reduces to: bake the software into a driver-VM-layout
   image and point `image` at it.
2. **Rust-native macro-tools** (the "heavy lifting out of the model"
   point): the approval bridge is the first composition pattern — a
   single model verb (`approval.request`) that bundles fetch/validate/
   persist/notify/execute for it. **`scrape_summarize` landed
   (2026-10-19, `pipeline.rs`, plugin `pipeline`)**: one call = fetch
   (≤5 urls, 2-wide, ≤800 KiB each, the SSRF predicate shared with
   web_fetch) → strip → word-boundary chunks → embed into private
   memory → ONE gateway summarize (or question-answer), full
   per-source status honest on partials. Live-fire verified against
   example.com + the production gateway (fetch/strip/embed/summary all
   green, 25.8 s end-to-end). Next: `refactor_module` (parse → patch →
   build-check → report, `syn`-shaped when a code crate lands).
   **`refactor_module` LANDED (2026-10-19, `pipeline.rs`, plugin
   `pipeline`)** — the second macro-tool, same one-call rails:
   structural balance scan (language-aware string/comment/template
   states for rust/python/ts-js; `Other` reports unchecked, never
   false) refuses half-modules BEFORE the model leg, ONE gateway
   completion applies the instruction, the patch is re-scanned, and   the build-check runs the language's cheapest real tool —
   `rustc --emit=metadata` (typecheck, no codegen) / `python3 -m
   py_compile` (bytecode compile, no execution) — never executing the
   module; absent toolchains report `checked: false` with the reason,
   never a folded pass. Result carries per-stage status + the patched
   source; truncation (>64 k chars) is refused, not shipped. Registry
   gate unchanged (`[plugins] pipeline`); 13 model-facing skills now.
   Tests: 9 new (scanner per-language positive/negative, fence
   extraction, cap refusals, no-tool honesty) + an `#[ignore]`d
   rustc live harness.
3. **Tiered cloud escalation with state serialization**: the approval
   checkpoint files are the serialization shape. Stage: on a structured
   local failure (schema break ×3, build-verify fail), serialize the
   step into an encrypted context package, hand it to ONE cloud call
   over the existing gateway lanes, pull the corrected structure back,
   drop the connection. The `MeshTask::CloudEscalation` lane (§16) is
   the router's eye on it; the state package rides the same
   **`cloud_escalate` LANDED (2026-10-20, `pipeline.rs`, plugin
   `pipeline`)** — the third macro-tool, same one-call rails: validate
   + pre-scan (a structurally broken artifact is refused locally —
   nothing sealed, nothing spent) → the step is sealed into a versioned
   `EscalationPackage` (v1) under AES-256-GCM (ring — the workspace
   rustls pin's own instance, no new provider ambiguity) at
   `$JERITH_HOME/bot/escalations/`: 0700 dir, 0600 packages, a stable
   0600 key file created once, atomic writes, and a 32-package
   retention window pruned best-effort → ONE bounded
   `/v1/chat/completions` round trip over the EXISTING gateway lanes
   (no other connection, none held) → the corrected artifact is
   re-scanned and reported with an honest `suspect_truncation` flag. A
   failed cloud leg RETAINS the sealed checkpoint and types the result
   (`ok: false` + package id + note) — an escalation is never lost
   mid-call. The write self-verifies: the persisted blob must open back
   to the exact package before the report claims it sealed. Registry
   gate unchanged (`[plugins] pipeline`); 13 model-facing skills now.
   Tests: 6 new (seal/open round-trip + nonce freshness, tamper /
   wrong-key / truncation / header honesty, cap refusals before any
   sealing, pre-scan refusal with an unreachable gateway, store seal +
   retention prune + key persistence, unbalanced-context never sealed).
   Live fire: VERIFIED (2026-10-20, owner-approved, via the ignored
   `pipeline::tests::live_cloud_escalate_real_failure` harness): a REAL
   structured failure (`2 * r` — integer×f64 multiply that fails the
   local rustc metadata check every retry) escalated through the
   production gateway in ONE round trip (8.4 s), the corrected artifact
   (`2.0 * r`) came back re-scanned (`suspect_truncation: false`) and
   passed the REAL local build-check inside the harness; the sealed
   package landed in the real store (0700 dir, 0600 package, opaque:
   `esc-…-5f3a.esc`, 556 bytes) beside the freshly-created 0600 key —
   the full at-rest chain audited live, not asserted.
4. **Local workspace shadowing — LANDED (2026-10-20, `workflows.rs`,
   `[workflows.shadow]` in bot.toml)**: `[workflows.shadow] enabled =
   true` + an `image` the operator vouched for (the bot NEVER picks an
   image silently — enabled-without-image is a typed refusal at run
   time) and replays ride a disposable containerized chrome instead of
   a host profile: unique-name `--rm` container per session,
   loopback-only CDP port map, `--cap-drop ALL` + no-new-privileges
   (chrome's own sandbox off — the container IS the sandbox), a fresh
   nosuid/noexec tmpfs HOME (`/home/shadow`, 64 MiB) carrying the
   profile, and ONLY manifest-listed mounts (`host:container[:ro|rw]`,
   default ro — rw must be visible in the manifest). The exact argv is
   a pure, grep-able, unit-pinned function. Shutdown removes the
   container BY NAME (docker-daemon-owned children outlive a killed
   CLI), and an arrival sweeper removes leftover prefix-owned containers
   before a new session. DEFAULT docker networking stays ON (replays
   must reach the pages they drive — `--network none` remains the
   chat-attachment sandbox contract, not this tool's). Record stays on
   the host display; only replay shadows. Live-fire: VERIFIED
   (2026-10-20 via `workflows::tests::live_workflow_replay_inside_shadow`,
   run alone): `zenika/alpine-chrome:latest` pulled, one real replay
   INSIDE the container in 4.7 s — goto example.com + a REAL CDP mouse
   click on its live link, per-step ✓ through run()'s full path
   (launch_shadow → mapped CDP → replay engine → typed report). Two
   debugged-and-landed contracts make it work across image layouts:
   (a) the argv uses `--entrypoint=<browser>` so images whose ENTRYPOINT
   is already the browser don't fight it, and the fresh HOME tmpfs is
   `mode=1777` because the image runs its browser as uid 1000; (b) the
   builder requests `--headless=old` — chromium IGNORES
   `--remote-debugging-address=0.0.0.0` under the new headless runtime
   (binds container-loopback only, port map unreachable — debugged live
   with in-container + host probes); honest constraint: chromium ≥132
   removes the old runtime, and when images migrate the fix is an
   in-image relay, flagged by this harness. The boot-probe loop also
   gained a REAL 30 s deadline + transient-refused tolerance (the old
   check called `Instant::now().elapsed()` on a fresh instant — ~0 — so
   the host path's timeout never actually fired).

## 18. External chat transports: 11 platforms (2026-10-19)

`crates/bot/src/transports/` — Phase 3d of docs/PLUGIN-SDK.md Role 2,
shipped BEFORE any operator asks for it (the module doc states the
dormancy discipline: no `[transports.<platform>]` table in bot.toml =
no task, zero cost; a table = a supervised task on the SAME dialogue
core as Discord — `handle_incoming` → gate once →
`converse::run_session` → delivery — with per-user history keys hashed
`platform:user`-style so lanes never alias).

| Platform | Inbound | Outbound | Dependencies + notes |
|---|---|---|---|
| IRC | native RFC 2812 TCP client (NICK/USER/PING, CTCP ACTION normalize), TLS via tokio-rustls + webpki roots | PRIVMSG chunks @400 chars, flood-paced | no vendor SDK; `respond_to: mention \| all` gate |
| Slack | **Socket Mode** (wss `apps.connections.open`, envelope acks, zero ports) | `chat.postMessage` @3900 | self-echo via `auth.test`; subtypes skipped |
| Twitch | IRC-over-WSS (`wss://irc-ws.chat.twitch.tv`), tags parsed (display-name) | PRIVMSG @480 via single writer task (frame pacing) | read-only without `oauth_token` — the deliverer errs loudly per send |
| Mattermost | personal-access-token websocket `/api/v4/websocket` + authentication_challenge | REST `/api/v4/posts` @4000 | `channels` scoping + mention gate |
| Signal | signal-cli REST daemon long-poll (`/v1/receive`, loopback default) | `/v2/send` (DM recipients / groupIds) | daemon owns the crypto plumbing; `respond_groups` gate |
| Nextcloud Talk | OCS long-poll `…/room/{token}/messages?lookIntoFuture=1&timeout=10` + `sinceMessageId` cursor | POST `…/messages` @3200 | app-password (per-device revocable) Base auth |
| MS Teams | PUSH-only → `webinbox /in/teams` (Bot Framework; JWT validation not this build — loopback `allow_unverified` only) | client-credentials token cache + `{serviceUrl}/v3/conversations/{id}/activities` @3500 | `service_url_origins` allowlist gates every send |
| Google Chat | (push-only; RS256 service-account ingest not shipped — typed limitation logged) | space webhook URL POST `{text}` @3500 | outbound-only relay |
| Feishu / Lark | (callback needs public URL + AES; WS long-connection wire is SDK-private — typed limitation) | `tenant_access_token` cache + `im/v1/messages` (text as JSON string), feishu/lark domain switch | boot-time token round-trip = loud credential check |
| WhatsApp | PUSH-only → `webinbox /in/whatsapp` (X-Hub-Signature-256 HMAC w/ app_secret + hub.verify_token GET handshake) | Graph `/{phone_number_id}/messages` @3500 | `app_secret` REQUIRED — unsigned callbacks refused |
| GG | — | — | typed dormant stub: NO public bot API exists (researched 2026-10); `validate()` refuses at LOAD with the reason, citing the completion file |
| Telegram | Bot API long-poll `getUpdates` (offset = max update_id + 1, restart-safe) | `sendMessage` @4000, token rides the URL path — error contexts are REDACTED, never journaling the token | `respond_to: mention \| all`, private chats always relay, groups mention-gated; 409 = a second getUpdates consumer = typed config-fatal. The §15 approvals lane shares the token here by design: `transport_owns_token` gates the standalone poller off and `handle_transport_update` resolves button presses + `approve/deny` text inline — ONE getUpdates consumer per token, always |

`webinbox` (the one optional listener): loopback-bound axum server
(`127.0.0.1:5543` default) when `[transports.webinbox]` exists; each
route fail-closed-verifies its platform's signature (Slack v0 HMAC w/
±5-min replay window; WhatsApp Hub signature + verify handshake;
Teams/Google Chat secured flags). `spawn_all` wraps every platform in
crash-supervision: internal reconnect loops per adapter, config-level
fatals log loud and stop (never a silent half-transport).

Tests: 30+ new in-module tests — IRC config/parse/addressing, Slack
token shapes + event router (bot-echo/subtype/reaction → zero extra
routes), Twitch tag parse + config, Mattermost posted-parse/relay
gate/ws-url, Signal envelope parse + daemon loopback rule, Nextcloud
config/basic-auth, Teams activity parse + serviceUrl gate, WhatsApp
Meta-nest parse + app-secret requirement, Feishu domain map, GG refusal
lock, webinbox RFC-corpus HMAC + loopback default — suite 197 green.

## 19. MongoDB Atlas lane + recall mirror (2026-10-19)

`crates/bot/src/mongo.rs` + `[mongo]` in bot.toml — the first real
feature on the shared handle (`memory.embed` / `memory.search` gain a
cluster-backed mirror):

- **Write side**: every successful `memory.embed` mirrors the note into
  the cluster (`jerith.memory_mirror`, deduped by the gateway's memory
  id via upsert) — fire-and-forget from the tool loop, a mirror failure
  is a typed journal line, never a chat failure.
- **Read side**: `memory.search` keeps local-first semantics (the
  node's own nomic embeddings); ZERO local hits fall through to a
  `$text` search over the mirror, and every mirrored hit carries an
  explicit `source: "mongo-mirror"` + the payload keys
  `fallback: "mongo-mirror"` — provenance is never silent. The index
  is created best-effort at init; a bad index degrades search loudly,
  never boot.
- **Boot contract**: validate → connect → ping `admin` → install the
  handle; any failure = typed log, lane off, bot continues (the
  crash-loop from a block_on-inside-runtime was fixed same-day to an
  awaited async init). Live fire is BLOCKED on the Atlas credential
  fix (`bad auth` 8000 — see RUNBOOK `MongoDB Atlas lane`): everything
  upstream (SRV DNS, TLS, zlib, reach) is verified.

Tests: config URI scheme rules, absent-table = disabled default,
mirror doc shape (exactly `_id`/`text`/`ts_ms`), fresh-process lane-off
refusals — suite 200 green.

## 20. Video attachment lane: understand uploaded videos (2026-10-20)

`crates/bot/src/video.rs` + `[video]` in bot.toml — videos get
*understood* on the honesty-first shape the image and sandbox lanes
already use (no new wire protocol: the composed block is the same
bracketed attachment context):

- **Pipeline**: capped download (the sandbox lane's downloader) →
  `ffprobe` caps + stream honesty (audio-only containers, live/
  fragmented streams, over-cap durations → honest refusal lines, never
  guesses) → `ffmpeg` audio strip (16 kHz mono WAV) → local
  `openai-whisper` transcription (default `tiny`, wall-time budgeted) →
  evenly-strided frame sampling (default **every 5 s**, capped 8
  frames; long clips resample evenly across the WHOLE duration and the
  block says so) → per-frame vision through the gateway's LOCAL tier
  (same skill images ride; frames ride as data URLs, nothing leaves the
  node) → composed `Ok` block with duration, sampling note, transcript
  (or honest absence — a no-audio clip says so, an empty transcript
  says so) and per-timestamp frame notes.
- **Caps + knobs** (`[video]`): `enabled` (default on), bin paths
  (ffmpeg/ffprobe/whisper), `whisper_model`, `frame_stride_secs`
  (clamp 1–30), `max_frames` (clamp 1–16), `max_duration_secs` (clamp
  30–3600), `max_file_mb` (clamp 1–2048), `whisper_timeout_secs`
  (clamp 30–1800). Missing tooling degrades to honest lines, never
  boot failures nor silent drops.
- **Budget honesty**: ONE video per message (`MAX_VIDEOS_PER_MESSAGE`)
  keeps a lane run inside the chat timeout; extras get an honest
  "send 1 at a time" line. Results cache in the sandbox-extraction
  cache (same id/name+size keying) — a re-ask never re-runs whisper.
  Scratch files wipe on drop; every subprocess is timeout-bounded and
  kill-on-drop; argv builders are pure and unit-tested (the
  `docker_argv` pattern). A probed-ok run whose frames ALL fail
  refuses to fabricate understanding.
- **Routing**: shared `video::classify_kind` routes image / video /
  other at both attachment sites (the @mention/DM fallback and `/read`);
  a lying extension is caught by the probe, not believed.

Tests: 12 new units (classify, timestamps incl. long-clip resample,
argv shapes, probe-parse honesty, clamps) + a `#[ignore]` live harness
that generates a REAL 6-second clip and runs the entire lane through
the host toolchain (ffmpeg → ffprobe → whisper → frames → vision stub)
— suite 237 green, live harness pass.

## 21. Audio attachment lane: transcribe uploads locally (2026-10-21)

`crates/bot/src/audio.rs` + `[audio]` in bot.toml — plain audio (mp3,
ogg voice notes, m4a, wav, flac…) stops being sandbox binary-garbage
and gets *transcribed*, on the video lane's honesty-first shape, with
the same shared machinery (probe/runner/scratch/transcribe) but its
own economics: no frames, no vision tier, one whisper run per file.

- **Pipeline**: capped download → `ffprobe` honesty (no audio stream,
  live/fragmented no-duration, over-cap duration → honest refusal
  lines) → a REAL video stream (not cover-art mjpeg/png inside mp3s,
  which is expected and ignored) redirects honestly to the video lane
  instead of a fake transcript → whisper DIRECTLY on the original
  file (its internal ffmpeg decodes mp3/ogg/m4a/flac) → composed
  block: duration + transcript, or the honest emptiness line (a pure
  tone transcribes to nothing — the block says speech may be absent).
- **Caps + knobs** (`[audio]`): `enabled` (default on), bin paths,
  `whisper_model` (default `tiny`), `max_duration_secs` (clamp
  30–3600, default 600), `max_file_mb` (clamp 1–2048), 
  `whisper_timeout_secs` (clamp 30–1800). TWO audio attachments per
  message (whisper-only is cheaper than video's frames+transcribe,
  but still budget-honest with a skip line for the rest).
- **Routing**: the shared classifier routes audio BEFORE video, so
  Discord `.ogg` voice notes can never ride the video lane; a lying
  extension is corrected by the probe, not believed. Transcripts
  cache in the extraction cache (same id/name+size keying) — a
  re-ask never re-runs whisper.

Tests: 4 new units (classify, cover-art codec honesty, clamps) + a
`#[ignore]` live harness that generates a REAL sine-tone mp3 and runs
the entire lane through the host toolchain + real whisper — pinning
the honest emptiness shape. Suite 241 green.

## 22. Transcript indexing into /recall (2026-10-21)

`crates/bot/src/converse.rs` + `[memory]` in bot.toml — the hook the
daily memory-incident follow-up asked for: what a voice note SAID is
now recallable without re-uploading it:

- **Hook shape**: after a successful audio/video lane run (the same
  site that warms the extraction cache), the speech text is extracted
  from the composed block — a real transcript is one line exactly
  `transcript:` with the words underneath; the INLINE `transcript: …`
  honest-absence/failure shapes are never indexed, and (video) the
  `frame notes:` section stays out of the note.
- **Fire-and-forget**: one `memory.embed` note per transcribed
  attachment (archiver.rs convention — the node's private
  `/recall`-searchable store; mongo-config permitting the mirror rides
  the same id). Never slows a reply; a failure is a typed journal line
  and the transcript stays chat-cached either way. The note is capped
  at 4000 chars (`transcript of audio attachment voice.ogg: …`) with
  the ellipsis inside the cap.
- **Knob**: `[memory] index_transcripts` (default on) — when absent,
  the merge default carries indexing ON, consistent with the archiver.
- **Anti-double-index**: cache-warm re-asks reuse the cached block and
  never re-fire the embed (the hook sits on the lane-success path, so
  it fires exactly once per extraction).

Tests: extraction honesty (speech-only, absence/failure skipped, frame
notes excluded), note cap totals, config default + knob. Suite 244
green, clippy `-D warnings` 0.

## 23. Discord voice lane — always-on VC + Gemini Live (2026-10-21)

`crates/bot/src/voice.rs` + `[voice]` in bot.toml — the bot takes a
ONLY-by-config seat in a Discord voice channel (guild + channel ids)
and holds a real-time conversation:

- **Transport split**: songbird 0.6 (default-features OFF, explicit
  `driver/gateway/rustls/serenity/tungstenite/receive` — the workspace
  rustls=ring pin respected) handles ONLY the Discord voice transport;
  the Gemini side rides the tokio-tungstenite stack already in the
  tree. songbird's own receive pipeline delivers DECODED per-user PCM
  (`VoiceData.decoded_voice`, jitter-buffered stereo 48k) — no local
  opus decoder to maintain.
- **Dialog lane** (`gemini-3.8-live`, the v1beta BidiGenerateContent
  WSS): room audio downmixed 48k→16k (honest 3-fold average, cheap LP
  for speech) and pushed as `realtimeInput` PCM in 40 ms quanta; model
  audio returns 24 kHz mono → ×2 upsample → 48 kHz stereo f32 ring →
  the mixer reference (RawAdapter as a blocking symphonia MediaSource —
  get-no-bytes-because-mixer-zero-fill lessons pinned in unit tests).
  Barge-in = the model's own VAD: `interrupted` drains the ring.
- **Captions lane** (parallel `gemini-3.5-transcribe-live` WSS on the
  SAME broadcast room stream): diarized input transcriptions sink into
  the /recall memory lane (bounded `memory.embed` notes, archiver
  convention) and — when `[voice] captions_channel_id` is set — post
  live to a text channel via the parked http handle.
- **Caps + honesty**: one wall-time cap (`session_max_minutes`,
  clamp 5–120, default 30) re-establishes BOTH sessions together; a
  room studio (goAway/error) restarts supervised, never a crash; the
  key is never logged and never committed (parked in bot.toml 0600
  from the operator's litellm config at setup); `[voice] enabled`
  defaults false — live room capture is opt-in always.
- **Persona into voice**: the first 1200 characters of the system
  prompt seed the Live systemInstruction (spoken-to, not written-at;
  long rules degrade realtime).

- **Operator surfaces** (2026-10-21): `/status` always carries one
  honest voice line about the LIVE seat (in-VC · dialogue model ·
  session age · `captions on <#chan>`/off; config-off, seat-failed, or
  left states name themselves), and `/voiceleave` (owner-DM only via
  the default-deny gate, like /timeout) stops the lane until the next
  bot restart — the Leave switch is watched on a one-second tick, so
  the Live sessions close within the second instead of riding out the
  rest of a session that can run up to 30 min; a second call (or one
  before the lane ever ran) answers with the restart honesty instead
  of pretending. Rejoining without a restart is deliberately absent:
  rejoin-ability is the restart path, one honest control.

Tests: 11 units (config honesty + clamps, WS/URL shape, setup JSON
shapes for both lanes, DSP bridges both directions, play-ring pair
decode with an exact power-of-two f32 comparison, runtime-state
honesty for /status + /voiceleave in every process state) plus one
ignored live `ws` probe harness. Suite 256 green (bot bin), clippy
`-D warnings` 0.

Live-fire against the real endpoint happened the same day — four
wire-level lessons, each now pinned where it bit:

1. **The model is NOT a query parameter.** `?model=` on the bidi URL
   earns close 1007 «Unknown name model» — the model rides inside the
   setup message (`setup.model = "models/<name>"`), where it already
   was; the URL is just service + key.
2. **`diarization` is rejected**, close 1007 «not supported in
   BidiGenerateContent» — the captions setup sends a plain
   `inputAudioTranscription: {}`.
3. **Server-to-client frames are BINARY carrying JSON** (verified byte
   for byte: identical url + setup; a Text-only read loop receives
   NOTHING and the session idles to a server 1008 abort at exactly
   60 s). Fixed by unifying Text/Binary in the session read loop; the
   bot→server direction stays Text and is accepted.
4. **A silent room needs a heartbeat.** songbird only emits packets
   when someone speaks, so an empty channel = zero media and Google
   kills both legs at the 60 s mark (1008 «The operation was
   aborted»). The pump now streams 40 ms of digital silence whenever
   no room packet lands for 150 ms — sessions are stable for minutes
   in an empty room (measured live 2026-10-21).

Also discovered-then-fixed: the workspace `tokio-tungstenite` pin's
bare `rustls` feature is the implicit optional-dep feature (pulls the
rustls crate but NOT tokio-rustls) — `connect_async` ran until runtime
and honestly errored «TLS support not compiled in». The pin is now
`rustls-tls-webpki-roots` (RUNBOOK's voice block has the story; the
slack/twitch/mattermost transports ride the same fixed stack).

A human voice in the VC is the last hop: the automated suite cannot
speak. Manual try: join the configured VC, talk, listen for the model,
check /recall notes and the (optional) captions channel.

## 24. Daily owner self-briefing (2026-10-22)

`crates/bot/src/briefing.rs` + `[briefing]` in bot.toml — one owner DM
per local day that stitches the past 24h into a single honest note, so
the operator doesn't need `journalctl` to know how the bot is doing.
Composition, not invention — every section reads a surface the bot
already owns:

- **journal**: an in-process tracing layer (`JournalRing`, installed on
  the composed subscriber alongside the fmt layer) records every
  WARN/ERROR the process emits into a capped ring (24 lines, msg-first
  rendering, `field=value` context kept) + counters; the briefing
  drains it (the drain IS the reset — the numbers mean "since the last
  briefing"). Spawning `journalctl` is deliberately NOT done: the
  process journal is what the bot can always see about itself, unit-run
  or dev-run alike; and the ring only keeps the operator's own log
  lines, so nothing user-content leaks.
- **monitor**: `run_checks` fresh at brief time — warn/fail lines lead,
  the pass count is named honestly. Same threshold family as the
  monitor's own loop; the two messages are not redundant — the monitor
  DMs only when something is wrong; the briefing also reports when
  everything is.
- **mail**: archive listing via `mailbox::list_messages_in` — today's
  count (per archive filename stamp, labelled as such — these are
  archive stamps, not delivery receipts) + the two newest subjects.
- **voice**: the live seat line (`/status`'s proven wording) plus the
  seat-loss / session-re-establishment counters voice.rs now exposes
  (driving the next drop's diagnosis instead of folklore).
- **approvals + mesh**: outstanding human-in-the-loop card count, and
  peer aliveness/contribution from the gateway status (unreachable →
  named, with the error, not hidden).
- **the pet's morning greeting** (2026-10-23, `crates/pet-plugin`): when
  `external:pet` is opted in per `[plugins]` (either `enabled` or
  `owner_only` — both are the operator turning the companion on), the
  briefing opens with one line from `pet.look` over the SAME signed
  service contract the chat loop uses: name, stage and mood in the
  house lore's own vocabulary (`droplet → glowcap → drifter →
  sentinel`; `loop-de-loop / orange-cap / …`), food/energy + one line
  of the pet's own voice. Honesty rules: no mood in the card → no line
  (the pet's silence isn't ours to invent); the service down → no line
  and a DEBUG note, never a fake greeting. No direct state-file reads —
  the briefing asks, like any chat caller would. The card's own
  `age_days` counts the days: a quiet `· day N` suffix on any day, a
  small `🎂` festival on the milestones (1, 7, 30, 100, 365) — the
  count comes from the card, never from the briefing's own clock. And
  each brief asks for `noteback` — one kept word brought back out of
  the vault of self (`pet.look`'s optional pick, deterministic per
  minute, always captioned with the pet's own phrasing of keeping, no
  recency claims) rendered as a second indented line when the pet kept
  anything at all; silence stays silence.

Schedule semantics: hourly cadence; fires once per local day once the
local hour reaches `hour` (default 08, clamp 0–23). Fire state persists
to `$JERITH_HOME/bot/briefing-state.json` (same `$JERITH_HOME` rebase
convention as the mine-tally ledger), so a restart AFTER the hour
briefs immediately (catch-up) and a restart BEFORE it waits — exactly
once per local day. A failed DM is NOT persisted: the next hourly tick
retries today's briefing. Without a resolved owner id the report is
logged at INFO instead of DM'd — the monitor's fail-quiet rule.
Delivery rides the same owner resolution the monitor uses
(`cfg.owner_discord_id`); body is clipped to 1900 chars with the
honest tail (Discord's 2000-char bound).

Tests: 6 units — journal visitor + ring-drain semantics, the
once-per-day / catch-up/before-the-hour gate algebra, the pet greeting
renderer + its `[plugins]` opt-in gate, and the golden live-DM fixture
(format conventions pinned against the real 2026-10-22 briefing, so a
format drift breaks the build, not the owner's DM).

**Handoff note lane** (2026-10-23): an agent signs a working session
off with ONE structured mail (`scripts/write-handoff.py`, stdlib
only: `What shipped` / `Left live / wounds` / `Next steps`) into the
`[ai]` mailbox, and the briefing surfaces the newest `[handoff]`
subject while fresh (today/yesterday by archive stamp — Hinnant civil
math, no chrono): `🤝 handoff` + the first three body lines; stale →
one quiet pointer line; empty → the section isn't there. The first
mail the lane read was written by the session that built the lane
(RUNBOOK has the live-fire shape).

**Seat-hold watcher** (2026-10-21, after an observed silent drop): a
live seat can be lost between boot's join and the WSS lanes' view —
the day's evidence was the bot disappearing from the VC roster while
the journal carried NOTHING (no error, no leave line; the voice
sessions themselves never noticed). The lane therefore spawns a
seat-hold watcher: every 15 s it asks the parked call handle for its
live connection state (songbird `current_connection`, None = no voice
connection), journals the loss honestly, and re-joins the configured
channel through the SAME manager — the `Call` keeps its playback ring
and capture events across a rejoin, so the mixer and room fan-out ride
straight through. Between polls, the driver's own CoreEvents journal
instant truth (connect / reconnect / drop, payloads never logged —
they carry voice-server endpoints/keys). Retries while out run on a
~30 s cadence; `re-join
requested` is logged but the NEXT cadence's poll is the verdict (the
truth line is `voice seat back (driver reconnected)`). /voiceleave
exits the watcher — an operator stop is a full stop; the watcher only
repairs incidental drops.

## 25. The download-quarantine lane (2026-10-23)

`crates/quarantine-plugin/` + `external:quarantine` — the fourth live
consumer of the plugin SDK, born of the same day's BetterCraft hand-
run: a CurseForge-style modpack arrived by DM carrying an obfuscated
jar-class dropper inside `overrides/mods`; the inspection routine
(unzip → strings → verdict → hash → refuse, never executed) is now a
standing lane, and the bot does it BY DEFAULT before the owner ever
double-clicks.

Four pieces:

- **The scanner** (`crates/quarantine-plugin`): a minimal ZIP
  central-directory reader (stored + raw-deflate entries, per-entry
  inflate cap `MAX_ENTRY_INFLATED` = 32 MiB — `miniz_oxide`, pure Rust,
  no provider ambiguity for the workspace's rustls/ring pin), then the
  conviction marks — obfuscator fingerprints (`ZKM`/`Allatori`/…),
  process spawn (`java/lang/Runtime`/`ProcessBuilder`), network
  bootstrap (`Socket`/`HttpURLConnection`), file plant
  (`setExecutable`/`FileOutputStream`) — read from the DECODED
  entries' printable runs: a class's constant pool carries those
  identifiers whole, so obfuscators cannot frizzle them away. Typed
  verdicts, honest rules in `Marks::verdict` (one place, tests pin it):
  `malicious` = a native executable (PE/ELF/Mach-O magic, with the
  class-file CAFEBABE carve-out) smuggled into an archive, or a
  dropper COMBINATION (obf+spawn, obf+plant, spawn+plant, net+plant);
  `suspicious` = a lone strong mark that didn't combo; `benign` =
  nothing strong — a network-only jar (update checkers) stays quiet,
  the lane never cries wolf. Every scan is hashed (SHA-256 — the
  handle for comparing one sample across recipients), every failure is
  a typed `note` on a completed report, and the file edge respects a
  512 MiB read cap.
- **The verdict log**: append-only JSONL at
  `~/.jerith-mesh/quarantine/verdicts.jsonl` (`$JERITH_QUARANTINE_STATE`
  override) — the ONE cross-file surface the bot's lanes read from
  outside the service contract. Tolerant reader, 500-row cap, counts.
- **The watcher** (30 s poll, `$JERITH_QUARANTINE_WATCH_SECS`): scans
  the quarantine dir (`$JERITH_QUARANTINE_DIR`, default `~/Downloads`),
  settles files younger than 10 s (never scans a half-written file),
  keys scans on path+size+mtime so a re-downloaded name re-scans,
  forgets deletions, and never double-judges across restarts (the
  known-set is primed FROM the log). Strong verdicts journal at WARN —
  the event trail lands in the process journal on its own.
- **The bot's readings**, both sides of a lane that surfaces its
  verdicts without any new trust surface:
  - **monitor `check_quarantine`** (the sender): reads the log std-only;
    only verdicts whose FILE IS STILL ON DISK rise to warn/fail (a
    deleted threat is an old threat, never DM news again) — malicious
    standing FAILS, suspicious standing WARNs, and the DM names path +
    sha-prefix before the owner could ever open the file;
  - **briefing `quarantine_section`** (2026-10-23 join): when
    `external:quarantine` is opted in per `[plugins]` (either
    `enabled` or `owner_only`), the daily note carries a `🧯` line —
    standing verdicts lead ("1 malicious · 2 suspicious standing —
    top: `<path>` (sha …)"), a judged-quiet day shows the watch + the
    count, an empty day says so. Same signed-contract discipline as
    the pet's greeting (`external::invoke`, no direct state reads from
    the briefing; the monitor is the state-file reader).

The two model-visible skills (`quarantine.scan` — deep-inspect a path
on demand, `~` understood; `quarantine.recent` — counts, standing
rows, the 8-card tail) let the CHAT re-run the routine live: "scan this
presentation the stranger sent me" works, bounded at 200 files for a
directory sweep. Verified live 2026-10-23 in this house: unsigned
probe → 403; HMAC-signed probe → 200 + card; the deployment's own
opening tick scanned `~/Downloads` on first poll (1 benign); a
synthetic dropper-shaped jar (ZKM + ProcessBuilder + HttpURLConnection
+ setExecutable, deflated, dropped by a REAL zip writer) walked in,
was judged `malicious` inside one poll naming all four marker
kinds, and left an exactly-shaped JSONL row; deletion kept it quiet on
the bot side per the standing rule. Bot restarted clean on the
`external:quarantine` owner_only row, the drop parser accepting it —
a malformed drop aborts the start, so the clean boot IS the registry
proof. Full workspace suite green (453 bot + 16 lib tests in the new
crate among them).

## 26. The task ledger / intentions lane (2026-10-23)

`crates/ledger-plugin/` + `external:ledger` — the fifth live consumer
of the plugin SDK, and the lane that closes the loop the other
surfaces can't: goals the owner (or the model on their behalf) VOICES
in chat are promises today and gone tomorrow in scrollback. This lane
keeps them as typed entries on disk and stands them back up in the
morning note — said in chat, briefing tomorrow.

Four pieces, exactly the drop anatomy the pet/house/quarantine proved:

- **The ledger** (`ledger-plugin/lib.rs`, pure + tested): one
  `state.json` (`$JERITH_LEDGER_STATE` override, else
  `~/.jerith-mesh/ledger/state.json`), atomic save (temp + rename — a
  killed process can't halve the file), entries with a monotone id,
  text (≤ 200 chars), optional one-line note, `added_at_ms`, and
  `closed_at_ms` (None = open). Honest rules, typed errors: a
  duplicate of an already-OPEN text refuses with its existing id,
  blank text is not a goal, the open tail caps at 200 (one more add
  says close a few first — a ledger of 200 unfinished thoughts is a
  mood board), and closed entries are KEPT as history capped at 500 —
  the trim drops only the oldest CLOSED rows, never an open one
  (tested: a week-old intention does not become archive debris).
- **Four skills**, one contract: `ledger.add` (manifest tells the
  model to OFFER it whenever the owner voices an intention),
  `ledger.list` (open tail: count, oldest id + honest age "2 d" /
  "90 min" / "just now", closed-today count, the 8 oldest open cards
  with id + text + note + age), `ledger.done` (close by id — kept,
  timestamped; closing twice says so in a typed error), `ledger.drop`
  (delete outright — the "that was a typo, not a goal" lane; no
  archive row). Ids only ever come from `ledger.list`.
- **No heartbeat**: unlike quarantine's watcher and the pet's
  time-decayed pethood, this lane moves only when asked — intentions
  don't decay on a clock; they decay when the owner closes them. The
  daily briefing is the clock the owner asked for.
- **The briefing's `ledger_section`**: when `external:ledger` is
  opted in per `[plugins]` (either `enabled` or `owner_only` — the
  lane lives in `owner_only` alongside house/quarantine), the note
  gains a `📋` line — "4 open (oldest 2 d) — `ship the ledger lane` ·
  `migrate bot.toml` · `smoke test` · 1 closed today" (texts capped at
  three; a finished-but-quiet day says "nothing open · 2 closed
  today"; a clean day says "nothing open"). Same signed-contract
  discipline as the pet greeting and the quarantine line:
  `external::invoke` on `ledger.list`, no direct state-file reads —
  the renderer is pure and pinned by tests against the exact card the
  live service emits (verified 2026-10-23: unsigned probe 403, signed
  add/done/drop/list round-trip incl. the typed unknown-id error,
  smoke rows then closed; bot restarted clean on the gate row).
