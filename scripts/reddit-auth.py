#!/usr/bin/env python3
"""Mint and verify the reddit drop's OAuth credentials.

The reddit lane (crates/reddit-plugin) reads/writes via OAuth when
`~/.jerith-mesh/reddit/credentials.toml` exists, and falls back to the
public Atom feeds when it does not. This script is the missing half of
that upgrade: it turns a Reddit *script* app (client id + secret) into a
validated refresh token, and refuses to leave a half-written file behind
when anything fails.

Three subcommands:

    authorize   browser flow (no password stored anywhere): prints an
                authorize URL, catches the redirect on 127.0.0.1, exchanges
                the code, writes the file — 0600, atomically, on success only.
    password    script-app password grant for accounts WITHOUT 2FA (the
                old classic bot path). Same validation, same write rule.
    check       validate whatever is on disk right now: token exchange +
                /api/v1/me, prints the account, karma and whether writes
                are allowed. Read-only — never rewrites the file.

Nothing here prints a secret, an access token or a refresh token. The
redirect URI you register on Reddit must match `--redirect` byte for byte
(script apps accept localhost).

Register the app at https://www.reddit.com/prefs/apps (type: script).
See docs/RUNBOOK.md § "Getting credentials" for the click path and the
known 2026 registration failure modes.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import http.server
import json
import os
import secrets
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request

TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
AUTHORIZE_URL = "https://www.reddit.com/api/v1/authorize"
ME_URL = "https://oauth.reddit.com/api/v1/me"
# Everything the drop's nine skills need, and nothing more: identity for
# `about`, read for browse/search/thread/watch, submit/edit for posts and
# comments, vote for the single reversible vote, history for its own
# activity, mysubreddits for where it is allowed to post.
SCOPES = "identity,read,submit,edit,vote,history,mysubreddits"
DEFAULT_REDIRECT = "http://localhost:8899/callback"
DEFAULT_UA = "linux:jerith-mesh:0.1.4 (by /u/CHANGE-ME)"


def creds_path() -> str:
    override = os.environ.get("JERITH_REDDIT_CREDENTIALS", "").strip()
    if override:
        return override
    return os.path.join(os.path.expanduser("~"), ".jerith-mesh/reddit/credentials.toml")


def die(msg: str, code: int = 1) -> "None":
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def post_form(fields: dict[str, str], client_id: str, client_secret: str, ua: str) -> dict:
    """Token-endpoint POST. Reddit answers JSON even for 401s."""
    body = urllib.parse.urlencode(fields).encode()
    basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    req = urllib.request.Request(
        TOKEN_URL,
        data=body,
        headers={
            "authorization": f"Basic {basic}",
            "user-agent": ua,
            "content-type": "application/x-www-form-urlencoded",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            return json.loads(raw or "{}")
        except json.JSONDecodeError:
            return {"http_error": e.code, "body": raw[:200]}
    except urllib.error.URLError as e:
        die(f"reddit's token endpoint is unreachable ({e.reason}) — network, not credentials")


def me(access_token: str, ua: str) -> dict:
    req = urllib.request.Request(
        ME_URL, headers={"authorization": f"bearer {access_token}", "user-agent": ua}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        die(
            f"/api/v1/me answered {e.code} — the token was issued but is not usable for "
            "this app's scopes (check identity is in the scope list)"
        )
    except urllib.error.URLError as e:
        die(f"reddit is unreachable ({e.reason})")


def explain_token_error(reply: dict) -> "None":
    err = reply.get("error")
    if isinstance(err, int):
        # Reddit's 401 body is {"message": "Unauthorized", "error": 401} —
        # no error code, so the fix has to be spelled out.
        die(
            f"reddit rejected the credential pair ({reply.get('message', 'Unauthorized')}, "
            f"HTTP {err}). Check that client_id is the string under the app name and "
            "client_secret is the field labelled `secret` (not the app name or the redirect "
            "URI); an app created as `web app`/`installed app` cannot use these flows either — "
            "recreate it as type **script**."
        )
    if err == "invalid_grant":
        die(
            "invalid_grant — the code or refresh token was already used, expired, or was "
            "minted for a different client_id. Re-run `authorize` from the start (codes are "
            "single-use and die after ~10 minutes)."
        )
    if err == "unsupported_grant_type":
        die(
            "unsupported_grant_type — this app is not a `script` app. Recreate it at "
            "https://www.reddit.com/prefs/apps with type **script**."
        )
    if err:
        die(f"reddit refused the token request: {json.dumps(reply)[:300]}")
    if reply.get("error_description"):
        die(f"reddit refused the token request: {reply['error_description'][:300]}")
    die(
        "reddit refused the token request without an error code — usually a wrong "
        "client_id/client_secret pair (the secret is the value labelled `secret`, not the "
        "app name) or an IP-level block page. Try the same request in a browser."
    )


def write_creds(path: str, fields: dict[str, str]) -> None:
    """Atomically, 0600, and only after validation has passed."""
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    lines = [f'{k} = "{v}"' for k, v in fields.items()]
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, path)
    os.chmod(path, 0o600)
    print(f"wrote {path} (0600)")


def report_identity(account: dict, mode: str, can_write: bool) -> None:
    name = account.get("name") or "?"
    print(f"connected as u/{name} via {mode}")
    print(f"  karma: {account.get('link_karma', 0)} link / {account.get('comment_karma', 0)} comment")
    created = account.get("created_utc")
    if isinstance(created, (int, float)) and created > 0:
        when = datetime.datetime.fromtimestamp(created, datetime.timezone.utc)
        age_days = (datetime.datetime.now(datetime.timezone.utc) - when).days
        note = " — young account, Reddit throttles these harder" if age_days < 30 else ""
        print(f"  account created: {when:%Y-%m-%d} ({age_days}d ago){note}")
    print(f"  writes: {'allowed (two-step)' if can_write else 'REFUSED (app-only token)'}")


def wait_for_code(redirect: str, state: str, timeout: int) -> str:
    parsed = urllib.parse.urlparse(redirect)
    port = parsed.port or 80
    captured: dict[str, str] = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            got_state = qs.get("state", [""])[0]
            if got_state != state:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"state mismatch - close this tab and re-run")
                return
            captured["code"] = qs.get("code", [""])[0]
            captured["error"] = qs.get("error", [""])[0]
            msg = b"code received - you can close this tab"
            if captured["error"]:
                msg = f"reddit said: {captured['error']}".encode()
            self.send_response(200)
            self.send_header("content-type", "text/plain")
            self.end_headers()
            self.wfile.write(msg)
            threading.Thread(target=self.server.shutdown, daemon=True).start()

        def log_message(self, *args) -> None:
            pass

    try:
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as e:
        die(
            f"cannot listen on 127.0.0.1:{port} for the redirect ({e}). Free the port, or pass "
            "--paste and copy the code from the URL bar instead."
        )
    srv.timeout = timeout
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        srv.shutdown()
        die(f"no redirect within {timeout}s — re-run and approve in the browser")
    if captured.get("error"):
        die(f"reddit answered the authorize step with: {captured['error']}")
    if not captured.get("code"):
        die("the redirect carried no code")
    return captured["code"]


def cmd_authorize(args: argparse.Namespace) -> None:
    state = secrets.token_urlsafe(16)
    query = urllib.parse.urlencode(
        {
            "client_id": args.client_id,
            "response_type": "code",
            "state": state,
            "redirect_uri": args.redirect,
            "duration": "permanent",
            "scope": SCOPES,
        }
    )
    url = f"{AUTHORIZE_URL}?{query}"
    print("1. Open this URL, sign in as the bot account and press Allow:\n")
    print(f"   {url}\n")
    print(f"   (the redirect URI {args.redirect} must be exactly what the app lists)")
    if args.print_url_only:
        return
    if args.paste:
        pasted = input("\n2. Paste the `code` value from the redirect URL: ").strip()
        code = pasted
    else:
        print("\n2. Waiting for the redirect on that local port (Ctrl-C to abort)...")
        code = wait_for_code(args.redirect, state, args.timeout)
    if not code:
        die("empty code")

    reply = post_form(
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": args.redirect,
        },
        args.client_id,
        args.client_secret,
        args.user_agent,
    )
    token = reply.get("access_token")
    refresh = reply.get("refresh_token")
    if not token:
        explain_token_error(reply)
    if not refresh:
        die(
            "reddit returned an access token but no refresh token — the app is not a script "
            "app, or `duration=permanent` did not stick. Recreate the app as type script."
        )
    account = me(token, args.user_agent)
    report_identity(account, "refresh_token", True)
    if args.dry_run:
        print("dry run — credentials validated but NOT written")
        return
    write_creds(
        args.path,
        {
            "client_id": args.client_id,
            "client_secret": args.client_secret,
            "user_agent": args.user_agent,
            "refresh_token": refresh,
        },
    )
    print("next: systemctl --user restart jerith-reddit-plugin, then reddit.about")


def cmd_password(args: argparse.Namespace) -> None:
    reply = post_form(
        {
            "grant_type": "password",
            "username": args.username,
            "password": args.password,
        },
        args.client_id,
        args.client_secret,
        args.user_agent,
    )
    token = reply.get("access_token")
    refresh = reply.get("refresh_token")
    if not token:
        # 2FA is the usual reason: the endpoint answers invalid_grant.
        explain_token_error(reply)
    account = me(token, args.user_agent)
    report_identity(account, "password (refresh token minted)" if refresh else "password", True)
    if args.dry_run:
        print("dry run — credentials validated but NOT written")
        return
    if refresh:
        write_creds(
            args.path,
            {
                "client_id": args.client_id,
                "client_secret": args.client_secret,
                "user_agent": args.user_agent,
                "refresh_token": refresh,
            },
        )
    else:
        write_creds(
            args.path,
            {
                "client_id": args.client_id,
                "client_secret": args.client_secret,
                "user_agent": args.user_agent,
                "username": args.username,
                "password": args.password,
            },
        )


def read_creds(path: str) -> dict:
    if not os.path.exists(path):
        die(f"no credentials at {path} — the drop is in signed-out read mode", 2)
    try:
        import tomllib
    except ImportError:  # pragma: no cover - python < 3.11
        def parse(raw: str) -> dict:
            out = {}
            for line in raw.splitlines():
                if "=" not in line or line.strip().startswith("#"):
                    continue
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"')
            return out

        return parse(open(path).read())
    with open(path, "rb") as f:
        return tomllib.load(f)


def cmd_check(args: argparse.Namespace) -> None:
    path = args.path
    creds = read_creds(path)
    missing = [k for k in ("client_id", "client_secret", "user_agent") if not creds.get(k)]
    if missing:
        die(f"{path} is missing {', '.join(missing)}", 2)
    ua = creds["user_agent"]
    if creds.get("refresh_token"):
        fields = {"grant_type": "refresh_token", "refresh_token": creds["refresh_token"]}
        mode = "refresh_token"
    elif creds.get("username") and creds.get("password"):
        fields = {
            "grant_type": "password",
            "username": creds["username"],
            "password": creds["password"],
        }
        mode = "password"
    else:
        fields = {"grant_type": "client_credentials"}
        mode = "app_only"
    reply = post_form(fields, creds["client_id"], creds["client_secret"], ua)
    token = reply.get("access_token")
    if not token:
        explain_token_error(reply)
    account = me(token, ua)
    report_identity(account, mode, mode != "app_only")
    print(f"  credentials: {path} (untouched by this check)")
    if args.scopes and reply.get("scope"):
        print(f"  scopes granted: {reply['scope']}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Mint/verify the reddit drop's OAuth credentials (no secrets are ever printed)."
    )
    p.add_argument("--path", default=creds_path(), help="credentials.toml location")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, fn, needs_app in (
        ("authorize", cmd_authorize, True),
        ("password", cmd_password, True),
        ("check", cmd_check, False),
    ):
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
        if needs_app:
            s.add_argument("--client-id", required=True)
            s.add_argument("--client-secret", required=True)
            s.add_argument("--user-agent", default=DEFAULT_UA)
            s.add_argument("--dry-run", action="store_true", help="validate but do not write")
        if name == "authorize":
            s.add_argument("--redirect", default=DEFAULT_REDIRECT)
            s.add_argument("--paste", action="store_true", help="paste the code instead of catching the redirect")
            s.add_argument("--timeout", type=int, default=300)
            s.add_argument("--print-url-only", action="store_true")
        if name == "password":
            s.add_argument("--username", required=True)
            s.add_argument("--password", required=True, help="stdin-safe: pass via env instead when possible")
        if name == "check":
            s.add_argument("--scopes", action="store_true", help="print the scopes the token actually has")
    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
