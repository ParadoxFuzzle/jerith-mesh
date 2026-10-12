# Deploying the bootstrap node at p2p.jerithai.com

The bootstrap node binds `0.0.0.0:8901` (TCP) and must be reachable from the
open internet. A Cloudflare Tunnel means **no open ports** on the host: an
outbound-only `cloudflared` process bridges the public hostname to localhost.

## ✅ LIVE — deployed 2026-10-12

The tunnel is live: https://p2p.jerithai.com/health →
`{"ok":true,"service":"jerith-bootstrap"}`.

This host runs it as **user-level systemd services** (lingering enabled, so
they start at boot without a login):

- `~/.config/systemd/user/jerith-bootstrap.service` — runs
  `target/release/jerith bootstrap --port 8901` from the repo checkout
- `~/.config/systemd/user/cloudflared-jerith.service` — runs
  `cloudflared tunnel --config ~/.cloudflared/config.yml run jerith-bootstrap`
- Tunnel id: `fa81a9d9-eddc-4f4c-8442-3fe3375a7c1f`; DNS CNAME
  `p2p.jerithai.com` → tunnel (overwritten from a stale A record with
  `--overwrite-dns`)
- Manage: `systemctl --user {status|restart|stop} jerith-bootstrap cloudflared-jerith`

⚠️ **cloudflared CLI gotcha found during setup:**
`cloudflared tunnel route dns <name> <host>` resolved the tunnel by the
wrong id (pointed the CNAME at the `nmonit` tunnel instead of
`jerith-bootstrap`) despite the correct name. Fix: pass the **tunnel UUID**
explicitly and verify with the tunnel id printed in the "Added CNAME" line:
`cloudflared tunnel route dns --overwrite-dns fa81a9d9-… p2p.jerithai.com`.

---

## Reproducing the setup on a fresh host

### 1. Install cloudflared

```bash
# Debian/Ubuntu
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg \
  | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main" \
  | sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt update && sudo apt install -y cloudflared
```

### 2. Build the bootstrap binary

```bash
cargo build --release
```

### 3. Create the tunnel (requires a Cloudflare account on the zone)

```bash
cloudflared tunnel login
cloudflared tunnel create jerith-bootstrap
TUNNEL_ID=$(cloudflared tunnel list | awk '/jerith-bootstrap/{print $1}')

# Route DNS — use the UUID (see gotcha above) and verify the id in the output:
cloudflared tunnel route dns --overwrite-dns "$TUNNEL_ID" p2p.jerithai.com
dig +short p2p.jerithai.com CNAME   # should be <TUNNEL_ID>.cfargotunnel.com
```

### 4. Configure ingress

`~/.cloudflared/config.yml` (see `deploy/cloudflared/config.yml` in the
repo for the template):

```yaml
tunnel: <TUNNEL_ID>
credentials-file: /home/<user>/.cloudflared/<TUNNEL_ID>.json
ingress:
  - hostname: p2p.jerithai.com
    service: http://localhost:8901
    originRequest:
      connectTimeout: 10s
  - service: http_status:404
```

### 5. Services

Install the two user-level units shown above (adapting `ExecStart` paths),
enable lingering, then:

```bash
systemctl --user daemon-reload
systemctl --user enable --now jerith-bootstrap cloudflared-jerith
sudo loginctl enable-linger "$USER"
```

(Looking to move to a system-level hardened unit + dedicated `jerith` user
when sudo access is available — see `deploy/systemd/`.)

### 6. Verify

```bash
curl https://p2p.jerithai.com/health
# {"ok":true,"service":"jerith-bootstrap"}

curl "https://p2p.jerithai.com/v1/peers?model=glm-5.3-flash"
# {"peers":[ ... live nodes ... ]}
curl https://p2p.jerithai.com/v1/models/counts
# [{"count":N,"model_id":"..."}]
```

**Verified end-to-end on deploy day:** a local mesh node with
`bootstrap_url = "https://p2p.jerithai.com"` advertised itself over the
tunnel and appeared in the public peer list within one ping cycle (10 s),
and dropped off the list after shutdown when its TTL expired.

## Notes

- **Security posture:** the tunnel only exposes `/v1/*` + `/health` of the
  bootstrap server — no file access, no admin surface. Advertisements and
  pings are Ed25519-signed and stamp-freshness-checked; unsigned or replayed
  ads are rejected with 401.
- **No TLS at the host** — Cloudflare terminates TLS; the ingress rule points
  at plain `http://localhost:8901`.
- **Scaling:** run additional independent bootstrap nodes with the same
  binary and list them as fallbacks in each node's config (`bootstrap_url`).
  Phase 1 gossip will keep their registries consistent.
- The advertise TTL is 90 s; a node that stops refreshing disappears from
  `/v1/peers` automatically, and `/v1/report-dead` (2 votes) evicts a
  misbehaving node immediately.

## Discord bot deployment (`jerith-bot`)

The bot (`crates/bot`) talks to a **local node gateway** (default
`http://127.0.0.1:8900`) and the local JerithChain node
(`http://127.0.0.1:8322`). It does not need to be reachable from the
internet — Discord's gateway is an outbound websocket, so no tunnel
entry is required.

Setup on the same box as a running node:

1. Token: create a Discord application + bot, copy the token, then either
   - write `discord_token = "..."` into `~/.jerith-mesh/bot.toml`
     (recommended), or
   - set `JERITH_DISCORD_TOKEN` in the service unit.
2. Install the unit (from the repo root):
   `cp deploy/systemd/jerith-bot.service ~/.config/systemd/user/ &&
    systemctl --user daemon-reload && systemctl --user enable --now jerith-bot`
3. Verify: `systemctl --user status jerith-bot` then in Discord use
   `/status` and `/peers` — they should show the local node snapshot.
   Global slash-command registration can take a few minutes to
   propagate the first time.
4. Logs: `journalctl --user -u jerith-bot -f`

Optional: `/mine` wins announce to a Discord channel when
`JERITH_ANNOUNCE_CHANNEL_ID=<decimal channel id>` (or `announce_channel_id`
in bot.toml) is set; failures to post are logged, never surfaced to the
user's command.

Config precedence (bot): env var > `~/.jerith-mesh/bot.toml` > defaults.
`JERITH_GATEWAY_URL` / `JERITH_CHAIN_URL` override the URLs;
`JERITH_TOKEN` (or `~/.jerith-mesh/keys/api.token` via the core default)
authenticates `/mine` against the local JerithChain node.
