# LLM router + fallback: lessons learned

Scope: the chat-lane waves of 2026-10-14 → 2026-10-16 — the bot-side
streaming/live sink (43fee95, b103c18, 929aa47, 2cbfb76, 9ee4c8a), the
tiered model router over the local runtime (20ac7d0, 683eabf), and the
three live-incident fixes in the same lane (132f8ed, f1281fe, e652066).
Runtime side documented in `docs/RUNBOOK.md` ("Main-node runtime
(LiteLLM) notes"); each incident below names its commit and its live
journal evidence, and carries a test pin where one exists.

The one-line summary: **every failure in this lane presented as
something it wasn't.** Empty answers looked like infra failures, tool
name errors looked like auth errors, a doubled message looked like a
duplicate send, silent chats looked like broken streaming. The router
and fallback work made the lane better; the durable lesson is that the
lane needed cheap, honest observability before it needed more routing
logic.

---

## What shipped, in order

1. **Streaming + live sink (2026-10-14).** The bot sends `stream: true`
   and live-types into Discord (placeholder → throttled edits → final
   edit); the gateway passes SSE chunk-for-chunk; streamed deltas fold
   per round into the same `(status, body)` shape the buffered path
   produces, so tool/degenerate/retry logic stays single-source.
2. **Guardrail knobs (2026-10-14→15).** `runtime_num_predict` gateway
   stamp, `[chat] timeout_secs` (default 240, clamped 60–600), ollama
   `OLLAMA_MAX_QUEUE=16` + `OLLAMA_MAX_LOADED_MODELS=1` in the systemd
   unit.
3. **Tier router (2026-10-16, `model_router.rs`).** `route()` classifies
   auto requests — image content → vision, tool shape → tool,
   complexity → deep, plain → primary (when pinned) — and
   `reroute_for_gateway()` only rewrites unpinned requests; explicit
   caller models always pass through.
4. **Three live-incident fixes (2026-10-16).** Tool-round slop never
   live-types (132f8ed); delivery steps log at INFO/WARN (f1281fe); the
   final edit retries in place before any resend (e652066).

## The incidents and their real causes

| Presented as | Armature | Actual cause | Fix + pin |
|---|---|---|---|
| "Runtime returns empty answer" (many times) | Chat ends 200 with `content: ""` | *Three* distinct causes over four days: (a) `num_ctx` overflow — prompt + thinking outran the 4 096 default slot (journal: `n_ctx_slot = 4096, task.n_tokens = 2050` on every failing call); (b) think-swallow — free lanes return 200 + empty content below ~300 `max_tokens`; (c) thinking-model budget burn — qwen3:8b with a tool payload reasoned past the caller's cap and emitted empty `{}` | (a) pin `num_ctx: 8192` on the runtime entry; (b) only use free lanes with generous caps; (c) `auto_tool_tier_merge` merges `think: false` for *any* tool-bearing auto round whose tool tier is engaged — no-op rewrite included (`auto_tool_tier_merge_follows_the_same_gate_as_the_rewrite`, `tool_tier_no_op_rewrite_still_merges_think_false`) |
| "Can't finish posting" (2026-10-20) | Answer stops mid-sentence | The `runtime_num_predict` stamp rides every served leg — including the rerouted remote tiers — and `finish_reason: length` was dropped silently | Bot surfaces `finish_reason: length` with a continuation affordance; journals `answer hit the token cap`; Runbook entry 2026-10-20 17:16 |
| "Bot isn't streaming" (third report of a different cause each time) | Chats sit frozen at the placeholder | (i) tool-declaration wire names: dotted names (`memory.embed`) 400'd **every remote leg of the fallback list in order** — the chat only worked when some leg tolerated it; (ii) legacy ≥120-new-char edit floor meant nothing but huge buffers ever edited live; (iii) the `<200 chars = web request` catch-all forced tools onto every casual chat, and the then-blanket tool-round suppression froze them all | `wire_tool_name` folds dots/colons to underscores at declaration (`dispatch` resolves the wire form back); edit on any shown change per 2 s tick; catch-all removed — factual openers + trailing `?` attach tools instead |
| "Two identical answers" (2026-10-16 19:08) | Duplicate send? | The final-edit path: a transient Discord error on the last edit blind**resent** the same full text as a *new* message after the ticker had already typed it | Three in-place edit retries (1.5 s apart) before any resend (e652066); every delivery step INFO/WARN so the next doubled message names its path from the journal (f1281fe) |
| "Chat shows garbage JSON then rewrites" (same morning) | Live-typing races? | Tool-bearing rounds stream tool-intent JSON (`{ "…`, 193-char stubs) and echoed tool results — including **private embedded-memory recall** — before the reply | Tool rounds after an executed call fire no live events at all (suppression, not clear-later); toolless rounds hold brace-led fragments until content proves to be prose (`chat_json_shaped_stream_rounds_never_type_live`, `chat_tool_round_fragments_never_reach_the_live_surface`) |

## Lessons

1. **An empty HTTP-200 is a symptom class, not a cause.** Three
   unrelated faults surfaced as identical empty answers inside a week.
   Diagnose against runtime-side numbers (`n_ctx_slot`, `n_tokens`,
   token cap) before touching the router — retrying an empty answer
   through a fallback lane made cause (b) look like cause (c).
2. **Runaway generation needs three bounds at three layers.**
   `OLLAMA_MAX_QUEUE` makes chats fail fast instead of queueing behind a
   runaway; `OLLAMA_MAX_LOADED_MODELS=1` stops a probe model from
   camping VRAM; `runtime_num_predict` bounds the generation itself.
   Full-lane coverage, full-restart only once.
3. **A fallback list turns one wire error into total latency.** A dotted
   tool name was rejected by every remote leg in order, with errors that
   never said "charset". Declaration-side munging + a charset-pinned
   test is cheaper than per-leg debugging; validate against the
   narrowest provider charset in the ladder.
4. **Auth errors get layer-attributed wrong.** The proxy's bearer gate
   and the provider's 401 present the same "No api key" shape — a
   missing proxy key reads as an OpenRouter problem. Fix: one secret
   owner per layer, and probe scripts carry the bearer too.
5. **Buffering hides in innocent code.** The old passthrough's
   `bytes().await` silently buffered whole generations and only showed
   downstream as "not streaming". If a lane promises streaming, the wire
   contract is chunk-for-chunk and a probe names the first-chunk latency.
6. **Live surfaces gate on realness, not cleanliness.** Tool-round JSON
   and echoed tool results (private recall included) must not merely be
   *cleared* — they must never surface; RoundStart-clearing still showed
   the slop for a round's length. Suppress tool rounds outright; hold
   brace-led toolless fragments until prose is proven; the held burst
   fires on the fragment that proves it.
7. **Policy belongs to the tier decision, not the rewrite expressing
   it.** A same-as-base tier suppressed its own rewrite *and* the
   `think:false` merge riding it. Keep the tier as the
   policy-taking point (`auto_tool_tier_merge`) so a no-op can't silently
   drop a contract.
8. **Observability precedes redundancy.** The doubled delivery was
   invisible because its every step logged at debug. The fix order that
   worked: make structural paths announce themselves (INFO on
   placeholders/edits, explicit "a second message appears" WARN before
   any resend) *before* adding retry policy on top.
9. **Retries collapse in place or they duplicate.** State matters
   mid-delivery: the placeholder already holds visible text at
   final-edit time. In-place retry loops (bounded, spaced) precede any
   new-message path — the client-side twin of the idempotency rule.
10. **Heuristic catch-alls are corrosive in both directions.** The
    forced-web catch-all made every casual chat carry tools; the
    blanket tool suppression then made those same chats stream
    nothing. A heuristic that creates work + a guard that hides work =
    the silent-chat incident. Attach tools by shape (factual openers +
    trailing `?`), not by length.
11. **Keep rate-limited lanes out of loops.** The local tool tier stays
    local on purpose: a multi-round tool loop hammers a free-lane
    cap where single-shot chat does not. Coupling tier→quota is a
    product decision, not a fallback detail — record it.
12. **Never reroute a deliberate choice.** `reroute_for_gateway` gates
    on auto-ness (empty / placeholder / advertised base) and every tier
    probe indexes through the caller's anchor. Changes to pins land with
    a config backup, a PONG probe against the target alias, and the
    live gateway smoke — pin changes are config, and config is
    reversible.

## Held beliefs before this work, retired by it

- "The runtime failing to answer must be the runtime's fault." — A
  well-formed request can *make* the runtime fail (num_ctx, tool
  charset, caller cap).
- "Fallback chains make the lane resilient." — They make it *slow first*
  when the failure is request-shaped; the same bad request fails on
  every leg. Retry is for availability faults, not contract faults.
- "Duplicate messages mean the send path is broken." — Mid-delivery
  state re-entry is the usual culprit; the send path was fine.

## What we'd do again

- Incident-shaped test names (`chat_json_shaped_stream_rounds_never_type_live`)
  — the test suite now reads as the incident record and the scalar
  contract check at once.
- One folded `(status, body)` shape for streamed and buffered paths —
  tool/degenerate/retry decisions never forked by transport.
- Runbook-first changes: every knob, backup path, and journal line
  recorded same-day, dated, with the journal quote that proved it.
