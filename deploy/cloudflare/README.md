# Cloudflare edge rate-limit rule (as code)

The edge rule that complements the origin per-IP limiter
(`core/ratelimit.rs`). Rationale and the free/Pro parameter table live
in `docs/RUNBOOK.md` ("Edge layer"); this directory holds the rule in
Cloudflare API shape plus the exact apply/verify commands.

Current status: **recommended, not yet configured** — the zone rule is
applied manually via the dashboard or the commands below.

## Files

- `rate-limit-rule.json` — the ruleset body (phase `http_ratelimit`,
  one rule, free-plan parameters: 100 requests / 10 s per IP per data
  center, block 10 s). Edit thresholds here, not in the dashboard, then
  re-apply.

## Apply

Requires the zone id (`CF_ZONE_ID`) for `p2p.jerithai.com`'s zone and a
scoped API token (`CLOUDFLARE_API_TOKEN`): **Zone → WAF → Edit** and
**Zone → Zone → Read** — nothing broader.

```bash
# Read the current entrypoint ruleset (also gives the ruleset id):
curl -s -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
  "https://api.cloudflare.com/client/v4/zones/$CF_ZONE_ID/rulesets/phases/http_ratelimit/entrypoint"

# Create or overwrite the phase entrypoint with our rule set:
curl -s -X PUT -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
  -H "Content-Type: application/json" \
  --data @rate-limit-rule.json \
  "https://api.cloudflare.com/client/v4/zones/$CF_ZONE_ID/rulesets/phases/http_ratelimit/entrypoint"
```

Integrating an entrypoint that already has other rules: merge our rule
into the existing `rules` array instead of PUT-ing this file verbatim.

## Verify

```bash
# Listing (rule present + enabled):
curl -s -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
  "https://api.cloudflare.com/client/v4/zones/$CF_ZONE_ID/rulesets/phases/http_ratelimit/entrypoint" \
  | python3 -m json.tool

# Live behavior (from ONE host, inside 10 s): ~100 OKs then edge-level
# 429s — ideally BEFORE the origin's own 240-cap trip:
seq 1 130 | xargs -P 12 -I{} \
  curl -s -o /dev/null -w "%{http_code}\n" \
  "https://p2p.jerithai.com/v1/peers?model=edge-verify" \
  | sort | uniq -c
```

Caveats worth remembering (full list in the RUNBOOK):

- Counters are per data center (`cf.colo.id` is implicit) — a spray
  across PoPs gets N× the edge budget; the origin limiter remains the
  correctness layer for that case.
- Legit mesh peers poll on TTL heartbeats (~4 req/min) — nowhere near
  the threshold; verified bots are excluded by the expression.
- Free plan ceiling: periods only up to 10 s. Pro unlocks 60 s / 1 h —
  then bump `period: 60`, `requests_per_period: 200`,
  `mitigation_timeout: 3600` (see the JSON's rule description).
