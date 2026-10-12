# Deploying the bootstrap node at p2p.jerithai.com

The bootstrap node binds `0.0.0.0:8901` and must be reachable from the open
internet. A Cloudflare Tunnel means **no open ports** on the host: an
outbound-only `cloudflared` process bridges the public hostname to localhost.

## 1. Install cloudflared on the bootstrap host

```bash
# Debian/Ubuntu
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg \
  | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main" \
  | sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt update && sudo apt install -y cloudflared
```

## 2. Build & install the bootstrap binary

```bash
cargo build --release
sudo install -m 0755 target/release/jerith /usr/local/bin/jerith
```

## 3. Create the tunnel (requires a Cloudflare account with the zone jerithai.com)

```bash
sudo cloudflared tunnel login                      # opens browser auth
sudo cloudflared tunnel create jerith-bootstrap    # writes credentials JSON
# Copy the credentials file to /etc/cloudflared/jerith-bootstrap.json
sudo cp ~/.cloudflared/<tunnel-id>.json /etc/cloudflared/jerith-bootstrap.json
sudo cp deploy/cloudflared/config.yml /etc/cloudflared/config.yml

# Route DNS (creates the CNAME → tunnel automatically)
sudo cloudflared tunnel route dns jerith-bootstrap p2p.jerithai.com
```

## 4. Run the bootstrap node as a service

```bash
sudo useradd -r -s /usr/sbin/nologin jerith || true
sudo cp deploy/systemd/jerith-bootstrap.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now jerith-bootstrap
sudo systemctl enable --now cloudflared
```

## 5. Verify

```bash
curl https://p2p.jerithai.com/health
# {"ok":true,"service":"jerith-bootstrap"}

curl "https://p2p.jerithai.com/v1/peers?model=glm-5.3"
# {"peers":[ ... live nodes ... ]}
```

## Notes

- **Security:** the tunnel only exposes `/v1/*` + `/health` via the bootstrap
  server (no file access, no auth secrets in Phase 0). Real Ed25519 signature
  enforcement on `advertise` lands in Phase 1.
- **No TLS at the host** — Cloudflare terminates TLS; the ingress rule points
  at plain `http://localhost:8901`.
- **Scaling:** run additional bootstrap nodes with the same binary and list
  them as fallbacks in each node's config (`bootstrap_url`). Phase 1 gossip
  will keep their registries consistent.
- The default advertise TTL is 90 s; a node that stops refreshing disappears
  from `/v1/peers` automatically, and `/v1/report-dead` (2 votes) evicts a
  misbehaving node immediately.
