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
- **Anti-double-index**: cache-warm re-asks reuse the queued block and
  never re-fire the embed (the hook sits on the lane-success path, so
  it fires exactly once per extraction).

Tests: extraction honesty (speech-only, absence/failure skipped, frame
notes excluded), note cap totals, config default + knob. Suite 244
green, clippy `-D warnings` 0.
