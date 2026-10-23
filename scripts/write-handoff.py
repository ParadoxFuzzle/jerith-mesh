#!/usr/bin/env python3
"""write-handoff — the agent's structured sign-off into the [ai] mailbox.

One mail per working session, written by the agent BEFORE it signs off,
surfaced by the bot's daily briefing while fresh (today/yesterday).
Companion to the pet's diary lane (same box, different subjects) and
the mailhook's own archive shape (`YYYYMMDD-HHMMSS-<rand>.eml`).

Structure (convention, enforced by the template — keep it):

    What shipped
    : one line per landed thing, with the one-line WHY
    Left live / wounds
    : services started, pins changed, anything the next session must know
    Next steps
    : the one sentence per follow-up, no basura

Usage:
    scripts/write-handoff.py --by "Codebuff (Freebuff)" \
        --line "house card + pet diary live" <<'BODY'
    What shipped
    : house.status answers "how are you doing" on the signed contract
    Left live
    : jerith-house-plugin unit on :7790, external:house in owner_only
    Next steps
    : tell the owner the house card exists
    BODY

Flags: --box (default `ai`), --base (default `$JERITH_MAIL_ARCHIVE` else
`$HOME/.jerith-mail/inbox-archive`; same override the pet diary reads).
stdlib only — no toolchain needed beyond Python.
"""

import argparse
import email.utils
import os
import re
import secrets
import sys
import time
from pathlib import Path


def stamp() -> str:
    """The mailhook's local filename stamp: YYYYMMDD-HHMMSS."""
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:24] or "agent"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--by", required=True, help="the agent's name (From display)")
    ap.add_argument("--line", required=True,
                    help="one-line what: becomes `Subject: [handoff] <line>`")
    ap.add_argument("--box", default="ai",
                    help=f"mailbox to file into (default: ai)")
    ap.add_argument("--base", default=None,
                    help="archive base (default: $JERITH_MAIL_ARCHIVE else "
                         "$HOME/.jerith-mail/inbox-archive)")
    args = ap.parse_args()

    # The body is the sections text — piped or heredoc (see usage),
    # EOF-terminated. An empty one is refused: a handoff without a
    # body is a rumor.
    body = sys.stdin.read()
    if not body.strip():
        print("write-handoff: a handoff without a body is a rumor — "
              "write at least `What shipped\\n: ...`", file=sys.stderr)
        return 2

    base = Path(args.base or os.environ.get("JERITH_MAIL_ARCHIVE")
                or Path.home() / ".jerith-mail" / "inbox-archive")
    box = base / args.box
    box.mkdir(parents=True, exist_ok=True)

    fname = f"{stamp()}-handoff-{secrets.token_hex(2)}.eml"
    text = (
        f"From: {args.by} <{slug(args.by)}@jerith.mesh>\n"
        f"To: ai <ai@jerith.mesh>\n"
        f"Subject: [handoff] {args.line}\n"
        f"Date: {email.utils.formatdate(localtime=False, usegmt=True)}\n"
        f"X-Handoff-By: {args.by}\n"
        "\n"
        f"{body.rstrip()}\n"
    )
    path = box / fname
    path.write_text(text)
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
