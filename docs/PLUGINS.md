# Plugin / skill system (Phase 3)

The bot's model-facing capabilities follow the OpenClaw lineage: a
**plugin** owns capabilities, a **skill** is one callable the model can
summon mid-answer. The whole architecture lives in one table
(`crates/bot/src/plugins.rs`): every skill is a single `SkillSpec` entry
and three independent consumers derive from it.

```text
bot.toml [plugins] enabled = [...]
        │ (default: all pre-packaged plugins on)
        ▼
SkillRegistry::standard(enabled)          ← one construction, bot boot
        │
        ├─► registry.tool_defs()          → the chat payload's `tools` array
        │                                    (OpenAI function declarations)
        ├─► registry.enabled_skill_names()→ the dispatch whitelist
        │                                    (incl. the content-intent parser's
        │                                     allowlist for non-strict tool output)
        └─► registry.dispatch(name,args)  → the executor behind the loop's
                                             tool round (errors → {"error": …},
                                             the model recovers; chat never fails)
```

One table entry defines name + owning plugin + description + JSON-schema
parameters + runner. Touch `plugins.rs`, not three call sites.

## The pre-packaged set

| plugin id   | skills (model-facing)                     | what the runner does |
|-------------|-------------------------------------------|----------------------|
| `websearch` | `web_search`, `web_fetch`                 | `POST {gateway}/v1/skills/web/{search,fetch}` — Brave Search API + readable-text fetch (HTML stripped, length-capped, private targets refused) |
| `memory`    | `memory.embed`, `memory.search`           | `POST {gateway}/v1/skills/memory/{embed,search}` — node-private SQLite + embeddings; data never leaves the node |
| `lore`      | `lore.search`                             | `POST {gateway}/v1/skills/lore/search` — community-canon index (`503` with a hint when no vault is configured, never an empty fake result) |
| `vision`    | `vision.describe`, `vision.analyze`       | `POST {gateway}/v1/skills/vision/{describe,analyze}` — multimodal forward through the model router's vision tier (base64 image or `data:` URL in, text out) |
| `squeeze`   | *(none — lifecycle plugin)*               | bot-side history compaction (`crates/bot/src/squeeze.rs`): passive transcript trim + deep pass; registers no model tools deliberately |

`vision.analyze` shares the describe endpoint's transport; the schema
makes `prompt` required so the model signals intent; both ride the
node's `[vision]` tier routing (image-content chats), never the base
model.

## Gating — `[plugins]` in bot.toml

```toml
# ~/.jerith-mesh/bot.toml — omit the section for the default set
[plugins]
enabled = ["websearch", "memory", "lore", "squeeze"]
```

- **Omitting the section** (the default) switches on every pre-packaged
  plugin — `default_plugins_enabled()` is literally `PREPACKAGED_PLUGINS`.
- Unknown ids **fail loud** at bot start: `unknown plugin {e}` in
  `[plugins] enabled — known: …` (a mistyped flag must never sit
  silently off; unit-tested).
- One line describes the *whole* bot: lifecycle-only plugins (squeeze)
  share the same surface, so what's off in `enabled = [...]` is off —
  including for the non-model surfaces.
- Omitting a plugin id from `enabled` while listing others disables all
  its skills model-facingly; the `enabled_skill_names()` list is the
  dispatch whitelist, so a disabled or unknown skill **cannot be
  summoned** by the model at all — including via the lenient
  content-intent path (its allowlist is the same registry list).

## The refusal contract

A model request for an unknown or disabled skill never fails the chat:

- `registry.dispatch()` refuses with `skill {name} unknown or disabled`;
- the tool loop feeds back `{"error": …}` as the tool-round content, so
  the model can correct course mid-answer;
- geometry wired for hooks too: the refusal still emits a `skill_invoked`
  event with `ok: false` — observability triggers on both sides of the
  gate (raw arguments still never leave the node; see Hooks below).

Unit-tested: `standard_registry_lists_all_prepackaged_skills`,
`gating_removes_disabled_plugin_skills`, `empty_gate_is_all_off`,
`dispatch_routes_by_name_and_refuses_unknown_or_disabled`,
`unknown_plugin_id_in_config_fails_loud`, `tool_defs_are_openai_shaped`
(plus the `content_tool_intent` allowlist tests in `api.rs`).

## Writing a skill (the author path)

A skill is one `SkillSpec` with a runner of exact signature:

```rust
/// Gateway config + parsed tool arguments in → JSON out.
type SkillRunner =
    for<'args> fn(&'args BotConfig, &'args serde_json::Value)
        -> Pin<Box<dyn Future<Output = anyhow::Result<Value>> + Send + 'args>>;
```

Rules that make a good citizen:

1. **Validate arguments with `anyhow::ensure!`** — empty required
   strings (`memory.embed: empty text`), clamped ranges (`top_k` 1–20,
   `max_chars` 200–16000)…
2. **Return a JSON value** (not text): the materialized result goes back
   to the model as the tool-round content verbatim.
3. **Errors are recoverable, not fatal** — a refused helper call must
   never `panic!` or bubble into a chat failure; the runner's
   `Result` is translated to `{"error": …}` for the model.
4. **Requests ride the gateway**, not the model: skill runners are thin
   argument adapters over `POST {gateway_url}/v1/skills/*` — they never
   call an LLM directly.

The extension point is `SkillRegistry::from_skills(skills, enabled)` —
the pre-packaged registry is just `standard(enabled)` over the same
constructor, which is exactly how the tests inject synthetic skills:

```rust
let registry = SkillRegistry::from_skills(
    vec![SkillSpec {
        name: "debug.echo",
        plugin: "debug",
        description: "echo args",
        parameters: json!({"type": "object"}),
        run: |_cfg, args| Box::pin(async move { Ok(args.clone()) }),
    }],
    ["debug"],
);
// registry.tool_defs() → OpenAI declaration for the tools array
// registry.dispatch(&cfg, "debug.echo", &args).await? → args
```

Third-party integrations (e.g. the LangChain integration's runtime
tools loop under `integrations/langchain/`) consume the same registry
shape, so a skill added here is usable from any client on the mesh.

## Node-side skill endpoints (HTTP, gateway :8900)

The bot's runners are thin adapters over these routes; any client (curl,
LangChain, a future UI) can call them directly:

| route | method | body | notes |
|-------|--------|------|-------|
| `/v1/skills/memory/embed`  | POST | `{"text": …}` | node-private; embeds via configured embedding model |
| `/v1/skills/memory/search` | POST | `{"query": …, "top_k": …}` | vector search over local notes |
| `/v1/skills/memory/similar`| POST | `{"id": <note id>, "top_k": …}` | nearest neighbours of an existing note (needs the memory backend; `503` when unconfigured) |
| `/v1/skills/memory/stats`  | GET  | —  | note/graph counters for dashboards |
| `/v1/skills/lore/search`   | POST | `{"query": …, "top_k": …}` | lore index; **`503` + hint** when no vault configured |
| `/v1/skills/web/search`    | POST | `{"query", "count", "freshness"}` | Brave Search API (`[websearch]` key required) |
| `/v1/skills/web/fetch`     | POST | `{"url", "max_chars"}` | readable text only; private targets refused |
| `/v1/skills/vision/describe` POST | `{"image", "prompt"}` | routes via the model router's vision tier |
| `/v1/skills/vision/analyze`  POST | `{"image", "prompt"}` | same transport, analytic prompt required |

Responses self-label with the producing lane (`provider` — e.g.
`brave` on web results, the embedding/hash-fallback provider on memory
hits) where it matters for attribution; the vault-less lore path answers
`503` with a *set-it* hint ("no lore vault — set $JERITH_LORE_VAULT
or the [lore] vault_dir"), never a fake empty result.

## Hooks — observability for plugin authors

Webhooks (`[hooks]` in bot.toml) let operators *observe* what the model
summons, signed and fire-and-forget (delivery never slows a chat):

```toml
[hooks]
urls = ["https://example.com/jerith-hook"]
secret = "rotate-me"        # omits → unsigned delivery
timeout_secs = 5            # clamped 1–30
max_retries = 3             # clamped 1–5 (linear backoff 0.5 s / 1 s / 2 s)
```

Events (type on the wire is snake_case):

- `skill_invoked` — `{"skill": "<name>", "ok": true|false, "error"}`
  (raw arguments **never** included; receivers are third parties and
  arguments can carry user text)
- `guard_violation` — `{"reason": "<public reason>"}` (never the
  blocked text)
- `mine_won` — `{"discord_id": …, "amount_uj": …}`

Delivery discipline: HMAC-SHA256 over `"{ts_ms}.{body}"` in
`x-jerith-signature` (hex; `x-jerith-timestamp` rides along so
receivers can bound the replay window), up to `max_retries` attempts
per URL, and a global in-flight cap of 8 so a slow receiver can never
wedge the bot. **Empty `urls` (the default) disables the subsystem
entirely** — nothing is constructed, nothing is sent; URLs must be
`http(s)` only.

Receiver-side verification recipe (pseudo):

```
expected = hex( HMAC-SHA256(secret, "{x-jerith-timestamp}.{raw_body}") )
assert constant_time_eq(expected, x-jerith-signature)
assert now_ms - x-jerith-timestamp < your_replay_window
```

## Running the plugin tests

```bash
cargo test -p jerith-mesh-bot plugins::       # registry, gating, dispatch, config
cargo test -p jerith-mesh-bot hooks::         # signing, retries, minimality
cargo test -p jerith-mesh-core skill_         # node-side skill routes
```

## Cross-references

- `FEATURES.md §8` — the feature ledger this doc details
- `docs/RUNBOOK.md` — operator side: `[websearch]` key setup, guard
  config, `[monitor]`, lore vault wiring
- `crates/bot/src/plugins.rs` — the single-table architecture
- `crates/bot/src/hooks.rs` — delivery mechanics + minimality policy
- `crates/bot/src/api.rs` (`run_tool`, `content_tool_intent`) — the
  tool loop's consumption of the registry
- `crates/core/src/gateway.rs` — the `/v1/skills/*` route table
