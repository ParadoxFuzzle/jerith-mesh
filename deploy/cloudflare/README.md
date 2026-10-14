# Cloudflare edge rate-limit rule (as code)

The edge rule that complements the origin per-IP limiter
(`core/ratelimit.rs`). Rationale and the free/Pro parameter table live
in `docs/RUNBOOK.md` ("Edge layer"); this directory holds the rule in
Cloudflare API shape plus the exact apply/verify commands.

Current status: **LIVE** (2026-10-14) — ruleset
`ac289166024247efa78c1e1f979fcca0` on the jerithai.com zone (Free plan:
path-scoped expression only, no Host field). Verified rollout: a burst
from one client tripped at 102×200 + 28×429 — the threshold lands at
~100/10s as configured; the 429s are CF's `error code: 1015` edge page
(no origin involvement: both origin 429 counters stayed at 0 through the
burst), and service resumed 200s right after the 10 s mitigation.

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

# Create or overwrite the phase entrypoint with our rule set. NOTE: the
# entrypoint PUT accepts {"rules": ...} ONLY — strip the file's
# metadata fields (name/kind/phase — rejected as "unknown field"):
python3 -c "import json; d=json.load(open('rate-limit-rule.json')); \
  json.dump({'rules': d['rules']}, open('/tmp/put-body.json','w'))"
curl -s -X PUT -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
  -H "Content-Type: application/json" \
  --data @/tmp/put-body.json \
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

# Discriminate: an EDGE 429 is CF's `error code: 1015` text/plain page
# (retry-after: 10); the ORIGIN 429 is JSON with retry_after_secs.
# Confirm the origin never fired: node /status and bootstrap /health
# `security.rate_limited_429_total` unchanged across the burst.
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
