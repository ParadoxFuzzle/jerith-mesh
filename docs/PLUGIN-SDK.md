# Plugin SDK (Phase 3+)

**Status: Phase 3a + 3b + transports are implemented** — external
skills ship (`crates/bot/src/external.rs`, manifests in
`~/.jerith-mesh/plugins.d/`), the dialogue core ships
(`crates/bot/src/converse.rs`, the `ConversationSink` + `run_session`
seam Role 2 proposed), and the **transports layer now exists**:
`crates/bot/src/transports/` (FEATURES §18, 2026-10-19) — IRC, Slack,
Twitch, Mattermost, Signal, Nextcloud Talk, MS Teams, Google Chat,
Feishu/Lark, WhatsApp, GG as dormant `[transports.*]` adapters over
the shared core, plus the `webinbox` loopback listener for push-only
platforms (Telegram joined LIVE 2026-10-22). **Packaging ships too**
(Phase 3d, 2026-10-22): the template crate (`crates/plugin-template/`,
the `SkillServer` SDK lib + running example) and the Python single-file
runner (`sdk/python/jerith_plugin.py`) — two SDK flavors, one wire
contract. All dormant until their bot.toml table exists, per the module
doc.

This document pre-registers how plugins leave the compiled-in model
(`crates/bot/src/plugins.rs`, in-tree Rust) so the skills/channels
cosystem can extend without forking the bot. If an entry here conflicts
with `features.md §8`, the code wins and this doc gets corrected — that
ordering is the point of shipping the design before the implementation.

## Why in-tree plugins hit a wall

The shipped model is honest, small, and closed: one `SkillSpec` table
derives everything (tool schema, whitelist, gating), the refusal
contract is a few lines, everything is compile-checked. But a
community skill that way must (a) patch the bot's source, (b) rust-toolchain
enforce their build expectations on the operator's CI, and (c) ship on
*their* cadence. External surfaces (Telegram, WhatsApp) also cannot be
"just more SkillSpec entries" — they are transports, not skills. The SDK
design therefore splits in two, aligned with what the code already has:

1. **External skills** — processes the dispatcher can call (:8888-style
   local HTTP or whichever major-versioned surface decisions get made),
   registered by a manifest instead of a recompile. The registry gains a
   second constructor, not a new gate concept.
2. **External channels** — a *transport* layer beneath the dialogue
   loop: Telegram and WhatsApp adapters (Phase 3+ approved surfaces per
   `features.md §7`) stand alongside the current poise/serenity Discord
   implementation without replacing it.

## Role 1: manifest-registered external skills

A skill drop directory (`~/.jerith-mesh/plugins.d/<id>/plugin.toml`)
declares what a compiled-in table row can do today:

```toml
# ~/.jerith-mesh/plugins.d/weather/plugin.toml
[plugin]
id = "weather"
endpoint = "http://127.0.0.1:7777/rpc"   # http(s); loopback by default
secret = "per-plugin-rotate-me"          # enables x-jerith-signature over both directions
timeout_secs = 10                        # clamped like hooks (1–30)
# (response caps are dispatcher-side: 65536 bytes hard — no manifest knob;
#  the known plugin table is id/endpoint/secret/timeout_secs — anything
#  else in [plugin] is silently ignored, so don't ship knobs that don't exist)

[[skill]]
name = "now"                       # final model-visible name: "weather.now"
description = "Current temperature and conditions for a place."
parameters = { type = "object", properties = { place = { type = "string" } }, required = ["place"] }
```

Implemented as: `SkillRegistry::with_external(drops)` joins the manifest
skills into a **second constructor** over the same core — persisted shape
(`SkillSpec` pairs pre-parsed `run` = an HTTP adapter that speaks three
methods). Clients keep reading one registry list (tool definitions →
whitelist → dispatch) regardless; issuance of a manifest is exactly the
third party's weather-skill example made to skip the three-part edit
routine of `docs/PLUGINS.md`'s worked example.

Gating: `[plugins] enabled = [...]` grows a namespaced id form
(`weather` for an in-tree plugin, `external:weather` for a directory
drop); unknown names keep failing loud at startup for both kinds.
In-tree skills stay reachable by their bare ids exactly as today —
nothing about the shipped registry or its refusal contract changes.

Dispatch contract (what every endpoint must honor):

- `POST {endpoint}` — one JSON object: `{method, skill, args, ctx}` —
  matches what the current runner types give the loop:
  `method = "invoke"`, `skill = "weather.now"`, `args` = the model's
  parsed arguments, `ctx = {ts_ms}`.
- Bounded response, JSON, one object — mapped to the dispatch result;
  error strings feed the same `{"error": …}` the tool loop already
  translates, so the refusal contract holds unchanged.
- Both directions carry the same HMAC-SHA256 that hooks already use
  (`"{ts_ms}.{body}"` in `x-jerith-signature`), so a plugin author
  learns **one** signing recipe for both listen and emit.
- A wedged plugin can never wedge the chat: its timeout default is the
  hook discipline's 10 s cap.

Trust model — the operator ships the lateral trust that their machine
today spends on bot-side runtime config (websearch Brav, discord token,
and so on): a plugin is *local process code* written by a third party.
The manifest declares, but can't prove, the runner's internal promise;
what enforces behavior today is:

- `/<id>/plugin.toml` contents are validated like `[hooks]` (fail-closed
  startup, http(s)-only, reject file:// or 0.0.0.0 hosts cause node
  removal);
- the guard still screens whatever the bot sends upstream — the mesh
  edge does not move, it only gains subjects;
- `skill_invoked` hooking (plugin id → skill name + ok/error) means a
  misbehaving plugin is observable, noisy, and removable via `enabled`
  gating — the same fail-loud lens already unit-tested per skill.

## Role 2: transports (Telegram, WhatsApp, ...)

What the in-tree Discord bot has, and an adapter must substitute:

| concern        | in-tree Discord today | adapter contract (proposed) |
|----------------|----------------------|------------------------------|
| inbound text   | poise (`/ask`, threads) | `EVENT message → {channel_id, author_id, body}` posted to the bot core loop |
| reply          | serenity sends | bot core returns `{channel_id, response}` → the adapter sends it |
| threads/state  | per-channel in-memory | channel keyed by `{transport}-{channel_id}` |
| parsing skills | commands live in `commands.rs` as poise/serenity surfaces | transport-specific rendering of common `/ask|/status|/peers` — no new behavior |

The correct seam is not "rewrite the bot"; it is a **dialogue-core
extraction**: today `api.rs` is transport-agnostic except for where
commands and message texts call out to serenity. That boundary — a bench
`ConversationSink` trait with the Discord implementation as the first
impl, plus an in-memory test impl — is what the API already has at the
tool loop (transport-agnostic, proven by the mock-gateway test). We
extend it a step upstream onto the reply surface, not sideways into a
rewrite. Telegram then is one thin adapter: `poll_updates → event lane`,
`reply lane → terse sendText`, no new model contact surfaces.

Rate limiting, cooldowns and per-user history already live in the loop
(`api.rs`), not in serenity — so they port unchanged; the per-transport
adaptation needed is Discord's 2000-char split and backoff (pin
reactions and message chunking differ per transport).

## Phasing (what lands in what order)

1. **Phase 3a — registry joins manifests. ✅ implemented** (`c7d6f3d`,
   2026-10-16): `with_external(drops)` + manifest validation + HTTP
   dispatcher + the `[plugins]` namespaced ids + tests. External skills
   ship in `crates/bot/src/external.rs`; see the status header above.
2. **Phase 3b — dialogue-core extraction. ✅ implemented** (2026-10-16):
   `crates/bot/src/converse.rs` — the `ConversationSink` seam (sync
   best-effort `on_event`, Arc-consuming `finish`), `run_session` as
   the single definition of the squeeze→tag→chat→bookkeeping→format
   pipeline, and the shipped sinks: Discord's `LiveSink` impl plus the
   in-memory test transport (streaming + buffered end-to-end proofs
   against a real gateway stub, no serenity). `/ask` and the
   @mention/DM path now consume the core; the cooldown gate and error
   display stay transport-side by design (different wordings, and the
   gate must fire exactly once per request).
3. **Phase 3c — Telegram adapter.** The first external-channel
   consumer, shipping as a separate binary (`jerith-bot-telegram`)
   configured via `bot.toml [telegram]`, reusing everything else.
4. **Phase 3d — SDK packaging. ✅ implemented** (2026-10-22):
   `crates/plugin-template/` packages the contract as code — a
   `SkillServer` builder (the signed `/rpc` service: verification,
   ±5-min freshness like `gcp-bridge`, the 65536-byte response cap,
   typed errors) + the `jerith-plugin-example` binary (two demo skills)
   + the copyable `plugin.toml`. The Python appliance for the
   no-toolchain author is `sdk/python/jerith_plugin.py` (stdlib only):
   `Plugin.from_manifest` reads the SAME `[plugin]` endpoint line both
   halves bind/declare, `@p.skill("name")` registers handlers, boot
   fails loud on a manifest/handler mismatch. CI pins the union:
   the template crate's own tests round-trip the wire shapes, and a bot
   test (`shipped_template_manifest_parses_under_todays_rules`) parses
   the shipped manifest with the drop parser — template and parser
   cannot drift silently. The community README is RUNBOOK's "Author a
   plugin" section. Second live consumer: the mesh pet
   (`crates/pet-plugin/`, 2026-10-23 — `pet.look/talk/feed/play/rename`
   on the same signed contract with persisted, time-decayed state).
   Its build caught the example's latent `parse::<toml::Value>()`
   manifest-parse boot bug (full-document manifests must be
   `toml::from_str::<toml::Table>`); fixed and pinned by tests in
   both crates' binaries the same day. Third live consumer: the
   house (`crates/house-plugin/`, 2026-10-23 — `house.status`: the
   how-are-you card answering over the signed contract — gateway
   health, bot unit state, mesh contribution; `house.journal`: the
   bot's recent journal lines; `JERITH_GATEWAY` bindable, refusal
   to fetch is a card field, never a service error; `external:house`
   gate row, owner-DM posture like pet). Fourth live consumer: the
   download-quarantine lane (`crates/quarantine-plugin/`, 2026-10-23 —
   `quarantine.scan`/`.recent`: the byte-truth inspection of dropped
   files, verdicts into the JSONL log the bot's monitor + briefing
   lanes read; `external:quarantine` gate row, owner-DM posture).
   Fifth live consumer: the task ledger (`crates/ledger-plugin/`,
   2026-10-23 — `ledger.add`/`.list`/`.done`/`.drop`: the owner's
   open goals as typed entries on ONE atomic-save state file, the
   daily briefing's `📋` line rendering the open tail over the same
   signed contract; `external:ledger` gate row, owner-DM posture;
   no watcher — the lane moves only when asked). Sixth live consumer: the
weather lane (`crates/weather-plugin/`, 2026-10-27 — `weather.now` /
`weather.forecast`: open-meteo geocoding + current/forecast cards over
the same signed contract; `external:weather` gate row, deliberately
PUBLIC — a room-wide chat skill, the first non-owner_only drop).

## Distribution (separate design, not built)

Getting an external plugin onto an operator's box without a source
checkout — signed `.jbundle` artifacts, publisher keys, the fail-closed
install chain — is its own doc: [PLUGIN-CHANNEL.md](PLUGIN-CHANNEL.md).

## What would falsify this design

- If a real external skill needs streaming (model-driven, long-running
  outputs) instead of a bounded JSON response, the dispatcher contract
  above is insufficient — SSE/streaming responses would become the
  second wire shape, to be designed at that point rather than up front.
- If a channel's transport is *stateful* in a way that applies to model
  turns rather than replies (WhatsApp threads vs. Discord threads),
  `ConversationSink` needs upward API growth, not a rework — the loop
  itself never changes.

Both tests are cheap: they are CI-drivable (mock dispatcher, mock
transport) before any community surface exists.
