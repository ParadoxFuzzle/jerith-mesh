# jerith-mesh — Detailed Feature List

*(P2P model-homogeneous distributed MoE compute mesh)*

Status legend: ✅ in MVP · 🔄 in progress · 📋 planned (phase noted)

## 1. Core Distributed Compute
- ✅ Model-homogeneous peer meshes: nodes connect only to others running the *same* model & quantization
- ✅ Colibri-style MoE runtime integration (Phase 0: single-node expert streaming; local primitive)
- 📋 Resource pooling of RAM, VRAM, GPU, CPU across the mesh (Phase 1–2)
- 📋 Hybrid local + remote expert execution with locality-aware placement (Phase 2)
- 📋 Hot-expert replication (top 10–20 %) + cold-expert sharding (Phase 2)
- 📋 Pipeline / layer-group fallback (Petals-style) for high-latency links (Phase 2–3)
- 📋 Speculative decoding + expert budgeting + prefetching (Phase 3)
- 📋 Asynchronous, timeout-tolerant forward passes; graceful degradation (Phase 3)
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
- 📋 Additional independent bootstrap nodes + gossip-based list maintenance (Phase 1)

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
- 📋 Local embedding pipeline integrated with the chosen vector store (Phase 0.5)

## 7. User Interfaces & Access Surfaces
- ✅ CLI (setup, run, status, peers, config);
- ✅ Web UI status/dashboard page
- ✅ OpenAI-compatible HTTP API (`/v1/chat/completions`, `/v1/models`, streaming)
  Chat replies carry `mesh_fallback` (node id when a mesh peer served it,
  explicit `null` otherwise — including 502s when neither the local runtime
  nor mesh peers are reachable: never a bare 500).
- ✅ Discord bot (`crates/bot`, poise/serenity): `/ask` (mesh-peer fallback included),
  `/reset`, `/remember`, `/recall`, `/status`, `/peers`, `/mine`, `/help`;
  per-user chat history (in-memory, capped) + per-user cooldowns; token via
  `JERITH_DISCORD_TOKEN` or `~/.jerith-mesh/bot.toml`; gateway/chain URLs via
  `JERITH_GATEWAY_URL` / `JERITH_CHAIN_URL`;
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
- 📋 OpenClaw-style modular plugin and skill architecture (Phase 3)
- 📋 Default pre-packaged plugins: Context Squeeze, Juice Compact, Jerith Guard, Brave Search
- 📋 First-class skills `vision.*`, `memory.embed`, `memory.search`
- 📋 Webhook / event hooks for plugin authors

## 9. Incentives & Economics
- 📋 Native JerithChain integration (default) — import/reuse `jerith_core` mining + RPC flow (Phase 4)
- 📋 Optional Bitcoin, Ethereum, Dogecoin, Monero (user supplies own miner/wallet endpoints) (Phase 4)
- ✅ Interval metering hooks (tokens, expert activations, bandwidth, uptime) wired through the ping loop
- 📋 Automatic JER mining attempt on each served inference (Phase 4)
- 📋 Optional staking + slashing for higher-trust tiers (Phase 4–5)

## 10. Contribution Proof & Anti-Fraud
Phased — see ARCHITECTURE.md §8 for the full layered plan:
- ✅ Health-ping + contribution accounting now (rolling 24 h windows)
- 📋 Redundant / challenge-based verification on a configurable % of work (Phase 4)
- 📋 Cryptographic commitments on expert results (Phase 4)
- 📋 Spot-check “ringer” tasks with known answers (Phase 4)
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
- 📋 Resource isolation (GPU/CPU pinning, RAM ceiling), emergency kill switch (Phase 1)
- 📋 Dashboard with earnings/expert-hit-rate/latency views (Phase 2–3)
- 📋 Bandwidth caps (soft/hard) and cost estimation (Phase 2)
- 📋 Backup/restore/migrate full node state (Phase 2)
- 📋 Privacy levels: local-only / trusted-peers / full-mesh (Phase 3)
- 📋 Compatibility check for quantization, context length, tokenizer version (Phase 2)
- 📋 Rate limiting, per-peer blocking, optional local content filters (Phase 3)
