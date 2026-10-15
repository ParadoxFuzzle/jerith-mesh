# jerith-mesh — Architecture

Status legend: ✅ built (MVP) · 🔄 partial · 📋 planned (phase)

## 0. The core problem

Classic Expert Parallelism dispatches every token's chosen experts all-to-all
across the mesh. That assumes microsecond interconnects. Over the public
internet an all-to-all hop costs **tens to hundreds of milliseconds**, which
destroys interactive latency.

So every cross-node expert call must be **locally skippable, prefetchable,
and timeout-tolerant**, and the mesh must be structured so most traffic
never leaves the local machine at all.

## 1. Layered solution

| # | Layer | Technique | Goals |
|---|---|---|---|
| 1 | Locality-first placement | Activation-stat-aware expert placement (Prism/DanceMoE-style) | Maximize % of tokens whose top-k experts are local or <30–50 ms RTT neighbors |
| 2 | Hot-expert replication | Top 10–20 % most-activated experts replicated across multiple peers; router prefers local/nearby replica | Highest-leverage internet technique |
| 3 | Pipeline fallback | Petals-style layer-group ownership when expert-parallelism is too chatty | One activation per layer boundary instead of per-expert all-to-all |
| 4 | Latency hiding | Prefetch last-token/last-layer experts; speculative decoding with expert budgets; early-exit when local top-1 is strong | Hide remaining RTT |
| 5 | Async fault tolerance | Per-request timeouts, fallback nominal experts, micro-batch queues, idempotent result merging | Never block a whole forward pass on one remote expert |

## 2. Design principles

- Never assume low latency — every remote call is optional and timeout-able.
- Locality > perfect load balance.
- Measure real activation patterns on the target models.
- Start with pipeline + hot-expert replication; add fine-grained expert
  parallelism only between close peers.

## 3. Node lifecycle

```
                 install
                    │
              hardware scan ─────┐
                    │            │ (sysinfo fallback)
                 setup flow      │
        model → memory → contrib% → wallets
                    │            │
                 node start  ────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   bootstrap    local model    web/API
   discovery     gateway      surfaces
        │
   connect to peers running the SAME model
        │
   health-ping loop + contribution metering
```

### Node roles

- **Contributor** — only serves peers (no outbound chat requests of its own)
- **Requester** — only consumes mesh compute
- **Hybrid** — both (default)

### Model-homogeneous rule

The mesh identity is scoped by **exact model ID + quantization + context
length + tokenizer version**. GLM-5.3-q4-k_m never peers with GLM-5.3-fp16.
Model change = leave all meshes, then join the new one.

## 4. Discovery

Phase 0 ✅: HTTP bootstrap from `p2p.jerithai.com` (Cloudflare Tunnel) —
returns a signed list of `{node_id, address, model_id, quant, contribution%,
last_seen_ms, reputation}`.

Nodes ping each other continuously; alive/dead status propagates back to the
bootstrap, with TTLs so stale entries age out and nodes recover their
listing on re-up. Later phases add a real DHT/gossip overlay.

## 5. Contribution model

Four meters, keyed to a rolling window (24 h default):

| Meter | Who measures it | Units |
|---|---|---|
| Tokens inferred | peers | tokens |
| Expert activations | coordinator | activations |
| Bandwidth relayed | receiving peer | bytes |
| Uptime | bootstrap + peers | seconds |

Minimum contribution is **25 %**. Below it, the node is flagged; sustained
shortfall → priority demotion → exclusion from work offers.

### Anti-fraud

| Layer | What it does | When |
|---|---|---|
| Health pings + accounting | Verify node is alive and contributing | Phase 0 ✅ |
| Redundant replay | Same work to a second node on a configurable % (`verify_percent`, default 10, local expert arbitrates when hosted); mismatch flags both — or the outlier vs a local arbiter — and aborts ambiguous remote disputes | Phase 4 ✅ |
| Commitments | Every `ExpertResult` carries SHA-256 over (req_id, expert_id, node_id, input, output); requester recomputes and rejects tampered bodies; committed before it can know whether the work will be replay-checked | Phase 4 ✅ |
| Ringer challenges | Known-answer tasks injected at random — builtin `jerith-builtin/reverse2` expert (bit-exact f32, no model involvement) sent with `ringer_percent` probability per ping tick to a random alive peer; wrong outputs flag the peer | Phase 4 ✅ |
| Statistical scoring | Latency-vs-claim, work ratios, activation-pattern fit — ✅ Phase 4 (`freerider.rs`: coordinator-side latency-vs-pool + walk-order selected-but-absent signals over recency windows; decay + hysteresis; queues signed `Leeching` evidence — the bootstrap weighs it, statistics never slash) | Phase 4 ✅ |
| Gossiped reputation | Signed interaction records with decay | Phase 5 |
| Stake + slashing (optional tier) | Stake join gate — Contributor/Hybrid must show a chain balance ≥ `required_uj` or run demoted to Requester; verified offenses slash `slash_percent_per_offense` (25 %) with a persisted ledger and signed `/stake/slash` notices to the chain (contract pending chain-side) | Phase 5 ✅ |
| TEE / ZK | Optional high-trust tiers | Phase 5+ |

## 6. Wire protocol

Phase 0 uses HTTP/JSON for bootstrap and pings. Phase 1+ adds QUIC-based
NodeRPC (planned):

| Message | Direction | Purpose |
|---|---|---|
| `Ping` | peer → peer | liveness + RTT probe |
| `PingReply` | peer → originator | carries contribution stats |
| `DisclaimWork` | peer → peer | "I can't take this right now" |
| `ExpertLoad` | coordinator → peer | load specific expert(s) |
| `ExpertQuery` | coordinator → peer | run expert; return activations |
| `ExpertResult` | peer → coordinator | output + commitment hash |
| `ModelAdvertise` | any → any | version, quant, capacity, contribution% |

Transport sketch: quinn/QUIC with 0-RTT resumption, FP8-compressed
activations, batched tokens, Merkle-chunked results.

## 7. Node internal architecture

```
┌──────────────────────────────────────────────┐
│ jerith CLI  ──  setup │ run │ status │ peers │
├──────────────────────────────────────────────┤
│ core/                                        │
│   hw.rs        hardware probe (OS-specific)  │
│   config.rs    config.toml + identity        │
│   bootstrap.rs discovery client              │
│   peers.rs     peer registry + filtering     │
│   ping.rs      health-ping + metering loop   │
│   gateway.rs   OpenAI-compatible HTTP API    │
│   web.rs       status page / dashboard       │
│   economics.rs JER reward hooks (Phase 4)    │
├──────────────────────────────────────────────┤
│ Colibri runtime (external, wrapped via HTTP) │
└──────────────────────────────────────────────┘
```

## 8. Local runtime = Colibri

jerith-mesh wraps whatever Colibri can load — any GGUF / safetensors,
fetchable from HuggingFace or Ollama. Curated model list:

| Model | Notes |
|---|---|
| glm-5.2 / glm-5.3 / glm-5.3-flash | Colibri v2.0.0 engines |
| deepseek-v4-flash / deepseek-v4.1-flash | Colibri v2.0.0 engines |
| kimi-k3 | Colibri v2.0.0 engine |
| qwen3.8-flash-next | Colibri v2.0.0 engine |
| qwen-image-2.1 | Qwen-Image engine (image generation) |
| (user-installed) | any GGUF / safetensors path |

Local specialists (not meshed):

- Vision: MiniCPM-V 4.6:1b — image → text description / OCR (never leaves the
  node unless opted in)
- Embeddings: nomic-embed-text-v2-moe — vector search for the memory backend,
  Matryoshka-truncated to 768/512/256 dims

## 9. Memory & state

- **Enabled backends:** PostgreSQL + pgvector, or SQLite
- **Strictly local**: memory never replicates into the mesh; no distributed
  database anywhere in the system
- Embeddings created with the local nomic model only — raw text never leaves
  the node just to be embedded

## 10. Implementation phases

### Phase 3 in progress — multi-hop pipeline runtime (milestones)

Started 2026-10-15 (M0). The ladder, each milestone verifiable on its own:

- **M0 ✅ — route planning on the honest ownership map.** The layer-ownership
  ring previously handed the planners a SELF-ONLY map, so their remote path
  was dead code. The full map (every layer's owner, self or remote) now feeds
  `pipeline::plan_route` — deterministic, ordered, consecutive same-owner
  layers merged into hops (`None` = local run); the layer-aware dispatch
  planner delegates to it, drops the wrong `has_expert` gate (a layer owner
  is not an expert host), and an all-local mesh plans as a single hop.
- **M1 ✅ — hop execution over the existing activation contract.**
  `pipeline::execute_route` walks `plan_route`'s hops carrying the
  residual stream: each hop's output becomes the next hop's input, local
  slice = local executor, remote slice = the same signed expert RPC with
  a per-hop deadline (dispatch base + 3× the owner's measured RTT). On
  the anti-tamper side, each hop's commitment is recomputed locally
  against a *per-hop* expert id (`{model}#{first}-{last}`), so a
  tampering hop is caught downstream exactly like a single-expert reply
  (PR-composed, not replaced). Failures are typed `RouteError` (hop
  index, owner, layers, reason, hops-completed) — no silent layer skips;
  M2 turns the abort into ring-rule reassignment. The hop executor is
  weights-agnostic: tests use the synthetic expert weights; real runtime
  layer splits arrive with the Colibri-side work, and live QUIC wiring
  (`remote_expert_attempt` as the hop transport) rides along with it.

  **Live gate, passed (2026-10-15):** the wire-level semantics are no
  longer unit claims — `tests/pipeline_remote_drill.rs` runs the whole
  M-ladder through the real QUIC transport: a two-node mesh with
  on-disk synthetic stores, plan → execute → commit-verify over signed
  `NodeRPC ExpertQuery`, the per-hop deadline enforced at the wire, a
  mute (hung) owner tripping its deadline and getting ring-rule
  re-hosted onto the healthy node, the honest partial variant, and
  demand-priced budgets checked against measured attempt latencies.
  5/5 repeat runs green. What the Colibri split still brings: real
  runtime weights per layer run and wide-area RTT realism.
- **M2 ✅ — degradation + hysteresis.** `pipeline::execute_route_resilient`
  keeps routes alive through owner death: a failed hop's layers go to the
  caller-wired reassignment (shaped to
  `ShardMap::layer_ownership` over the alive set minus the dead owner —
  dropping a node re-pins *only its layers*, bounded by `REASSIGN_BUDGET`
  so the walk can never spin), replacement sub-hops re-enter the walk in
  order. When nobody can host the layers, the hop is skipped and typed
  into `RouteResult::degraded` and the walk continues — honest hops-short
  answers, mirroring `reject-continue`; `RouteResult` carries the residual,
  the successful-hop count, and the typed losses. `RouteStability` is the
  freerider-style two-line hysteresis (exclude at `HOP_EXCLUDE_SCORE`,
  clear only after decay below `HOP_CLEAR_SCORE`), fed by the walk itself;
  `filter_alive` drops excluded owners from the alive set but never
  empties it — a flaky hop beats an unroutable mesh.
- **M3 ✅ — latency budgeting.** `pipeline::plan_route_budget` prices a
  planned route against the measured-activation ordering: the caller
  wires a demand hook over `ExpertStats` (the same oracle prefetching
  uses — no second source of truth), every hop gets its RTT-floored
  deadline (`hop_timeout`, the M1 deadline IS the floor), and the
  remaining budget spreads in proportion to measured demand (hot hops
  win headroom, cold hops keep their floor; stable demand-descending
  order, deterministic u128 arithmetic). The resilient walk times every
  attempt (successes AND failures — a failed attempt's dead wait is as
  real to the route), and hops that overrun their priced budget are
  typed into `RouteResult::over_budget` with both durations; budgeting
  reports, never clamps (enforcement/reordering is later roadmap).
  Metrics surfacing is live-shaped but producer-pending: `Metrics`
  carries `record_pipeline_route` + a `pipeline` section (routes, over-
  budget hops, last-walk ms) rendered on the dashboard — zeroes until a
  live producer reports, i.e. when routing transport lands with the
  Colibri-side layer split (same pending status as M1/M2 transport).

| Phase | Goal | What gets built | Latency expectation |
|---|---|---|---|
| 0 ✅ | Local-only foundation | hardware probe, setup flow, bootstrap discovery, health pings, local gateway + OpenAI endpoint, web status | Baseline |
| 1 | Same-LAN mesh | NodeRPC + expert placement + local-first routing | Near single-node speed |
| 2 | Geo-aware mesh | Latency-aware placement + hot-expert replication + pipeline fallback | Interactive for near peers |
| 3 | Full internet + speculation | Prefetch, speculative decoding + expert budgets, async fallbacks | Acceptable chat latency |
| 4 | Verification & incentives | Cryptographic/statistical proofs + JER mining (≥ 5 % chance per served inference) | Economic layer live |
| 5 | Hardening | Stake+slashing, TEE tiers, ZK optional | Trust maturity |

## 11. Surfaces

| Surface | MVP | Later |
|---|---|---|
| CLI | ✅ `jerith setup/run/status/peers` | plugins, model switch UX |
| Web UI | ✅ status page | full dashboard |
| OpenAI API | ✅ | tool calling, embeddings |
| Discord | ✅ `/ask /reset /remember /recall /status /peers /mine` | expanded beyond OpenClaw |
| Telegram / WhatsApp | 📋 Phase 3 | via plugin SDK |

## 12. Non-goals

- **No distributed memory/storage.** Databases are per-node only.
- **No GPU/RAM pooling from laptops to desktops.** Compute follows the mesh,
  not the other way around — a node that can't host its slice of a model
  can't join that model's mesh.
- **No cross-model routing** in early phases. A request stays within its
  model's mesh; later multi-model routing is scoped as an optional skill.
- **No global ledger in the mesh.** Blockchain settlement is peer-to-JerithChain
  only, never mesh-internal.
