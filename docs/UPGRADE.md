# Upgrading jerith-mesh (node + bootstrap)

## The rule: **bootstrap first, nodes second**

Nodes sign against a canonical payload whose exact shape is part of the wire
contract. The bootstrap verifies against its own idea of that shape. When a
release changes the canonical payload (review the diff of
`advertise_payload` in `crates/core/src/signing.rs` and
`request_payload_for_stamp` in `crates/core/src/quic.rs` before deploying),
a version mix between nodes and bootstrap produces `401 Unauthorized` on
advertise until both sides agree.

Observed live in the 2026-10-12 deploy when `rpc_address` was appended to
the advertise payload: upgraded nodes 401'd against the old bootstrap until
the service was restarted on the new binary. That outage window is what
this procedure (and the transitional compatibility below) exists to avoid.

## Transitional compatibility (current release)

The bootstrap now **accepts both canonical shapes** during verification:

1. Try the current payload (with `rpc_address`).
2. On failure, try the legacy payload (the pre-`rpc_address` shape).
3. Only if both fail is the request rejected with 401.

Legacy acceptance logs at INFO with an upgrade pointer; the special case is
forgery-safe — the legacy fallback still enforces the same Ed25519 identity
and stamp-freshness rules, so it cannot be used to bypass wire auth.

This compatibility is **temporary**. Deprecation policy:

- Ships as usual starting 2026-10-12.
- Removed after **two minor releases** (or 90 days, whichever is later).
- Removal is itself a breaking change: nodes older than the removal
  release must upgrade first at that point.

## Standard upgrade procedure

### 0. Check whether this release moves the wire format

```bash
git log -p $(git describe --tags --abbrev=0)..HEAD -- \
  crates/core/src/signing.rs crates/core/src/quic.rs | grep -E "^\+.*(format!|\|)" | head
```

If `advertise_payload` / `ping_payload` / `request_payload_for_stamp`
changed shape → this is a **contract release**: follow §A. Otherwise §B.

### A. Contract release (canonical payload or reply shapes changed)

Order matters. Downtime is one ping cycle (10 s) maximum.

1. **Backup registry state** (bootstrap keeps scores/history in memory, so
   the only loss on restart is in-flight TTL data):

   ```bash
   systemctl --user restart jerith-bootstrap       # downtime begins
   journalctl --user -u jerith-bootstrap -n 20      # confirm clean start
   curl -fsS https://p2p.jerithai.com/health        # {"ok":true,...}
   ```

2. **Confirm compat window active** (if this release *adds* a shape):
   the new binary accepts legacy payloads — old nodes keep advertising.

3. **Upgrade nodes**, oldest mesh first:

   ```bash
   cargo build --release
   for p in $(pgrep -f "jerith run"); do kill "$p"; done
   ./target/release/jerith run     # per node
   ```

4. **Verify no 401 storms**:

   ```bash
   journalctl --user -u jerith-bootstrap -n 50 | grep -cE "wire auth failed"
   journalctl --user -u nodes -n 50 | grep -cE "advertise rejected"
   # both should be 0 once nodes are upgraded
   ```

5. **Schedule the legacy-acceptance removal** in the release notes.

### B. Non-contract release

No ordering constraint. Restart the bootstrap service, then nodes at
convenience.

## Rollback

Both binaries are backwards-verifying (during the compat window), so
downgrading the bootstrap software mid-incident is safe as long as nodes
have an equally old or rung-up payload. If a rollback is required outside
that window, downgrade **all** binaries at once and expect one ping cycle
of 401s on restart.

## Cross-component version settling checklist

- [ ] `git log -p` on `signing.rs`, `quic.rs` — payload shape diffed
- [ ] Bootstrap restarted first; `/health` and one test advertise healthy
- [ ] Nodes restarted on the new binary
- [ ] 401 counters at 0 on both bootstrap and node logs
- [ ] Legacy-compat deprecation date added to release notes
- [ ] `systemctl --user` units updated if ExecStart moved (`rpc_port`,
      `gateway_port` renames)
