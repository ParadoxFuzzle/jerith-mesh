#!/usr/bin/env python3
"""Jerith sandbox extractor — runs INSIDE the disposable container.

The container holds the safety contract (no network, caps dropped,
read-only rootfs, uid 65534, memory/pids/cpu caps); this script holds the
format policy: text/code reads raw, pdf via pdftotext, the OOXML/ODF
family via stdlib zipfile+regex untagging, zip archives get an entry
listing. Every path below is an argv STRING we were handed by the bot —
the script treats them as names, never executes them, and every story is
capped so a crafted file cannot waste memory or flood the chat prompt.

argv: pairs `<safe-name>=<display-label>`; files live under /sandbox.
Output per file (deterministic, parsed by the bot):

  ===BEGIN <safe-name>
  status: ok            (content follows)
  status: binary:<n>    (no text; n = file size in bytes)
  status: empty:
  status: unsupported:<why>
  status: error:<why>
  ---CONTENT---
  ...capped text...
  ===END
"""

import os
import re
import subprocess
import sys
import zipfile

# Per-file extracted-text cap (chars). The bot passes its own budget via
# env (EXTRACT_CAP) and caps again client-side; twice-bounded on purpose.
TEXT_CAP = int(os.environ.get("EXTRACT_CAP", "24000"))

# OOXML/ODF member paths (zipfile-relative) per extension.
OOXML = {
    "docx": ["word/document.xml"],
    "odt": ["content.xml"],
    "ods": ["content.xml"],
    "pptx": ["ppt/slides/slide1.xml", "ppt/slides/slide2.xml"],
    "xlsx": ["xl/sharedStrings.xml"],
}

MAX_ZIP_XML_BYTES = 32 * 1024 * 1024  # refuse absurd members before decompress


def emit(key, status, content=""):
    """One deterministic per-file block; the bot's Rust parser keys the
    END marker on the same safe name."""
    sys.stdout.write(
        "===BEGIN %s\nstatus: %s\n---CONTENT---\n%s\n===END %s\n" % (key, status, content, key)
    )


def cap_text(s):
    return s[:TEXT_CAP] + ("…[truncated]" if len(s) > TEXT_CAP else "")


def strip_tags(xml):
    body = re.sub(r"<[^>]*>", " ", xml)
    body = body.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    body = body.replace("&quot;", '"').replace("&apos;", "'")
    return re.sub(r"\s+", " ", body).strip()


def read_text(path):
    """Raw read for text-like files; binary sniffed on the first 4 KiB."""
    with open(path, "rb") as f:
        raw = f.read(TEXT_CAP + 4096)
    if b"\x00" in raw[:4096]:
        return ("binary", str(len(raw)))
    text = raw.decode("utf-8", errors="replace")
    if not text.strip():
        return ("empty", "")
    return ("ok", cap_text(text))


def read_pdf(path):
    try:
        p = subprocess.run(
            ["pdftotext", "-l", "10", "-q", path, "-"],
            capture_output=True, timeout=45,
        )
    except FileNotFoundError:
        return ("unsupported", "pdftotext missing from sandbox image")
    except subprocess.TimeoutExpired:
        return ("error", "pdftotext timed out")
    out = p.stdout.decode("utf-8", errors="replace")
    if not out.strip():
        return ("empty", "")
    return ("ok", cap_text(out))


def read_ooxml(path, ext):
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            if ext == "pptx":
                members = sorted(n for n in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", n))
                members = members[:20]
            else:
                members = OOXML[ext]
            parts = []
            for m in members:
                if m not in names:
                    continue
                if zf.getinfo(m).file_size > MAX_ZIP_XML_BYTES:
                    return ("error", "member too large: %s" % m)
                parts.append(strip_tags(zf.read(m).decode("utf-8", errors="replace")))
            text = "\n\n".join(p for p in parts if p)
    except zipfile.BadZipFile:
        return ("binary", "not a valid archive")
    except Exception as e:  # noqa: BLE001 — report honestly, never crash the batch
        return ("error", str(e)[:120])
    if not text.strip():
        return ("empty", "")
    return ("ok", cap_text(text))


def read_zip(path):
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
    except zipfile.BadZipFile:
        return ("binary", "not a valid archive")
    except Exception as e:  # noqa: BLE001
        return ("error", str(e)[:120])
    listing = ", ".join(names[:50])
    return ("ok", "archive with %d entries: %s%s" % (len(names), listing, "…" if len(names) > 50 else ""))


def main():
    for pair in sys.argv[1:]:
        key, _, label = pair.partition("=")
        path = os.path.join("/sandbox", key)
        ext = label.rsplit(".", 1)[-1].lower() if "." in label else ""
        try:
            if not os.path.isfile(path):
                emit(key, "error:", "file did not stage")
            elif ext == "pdf":
                emit(key, *read_pdf(path))
            elif ext in ("zip",):
                emit(key, *read_zip(path))
            elif ext in OOXML or ext == "pptx":
                emit(key, *read_ooxml(path, ext))
            else:
                emit(key, *read_text(path))
        except Exception as e:  # noqa: BLE001 — report honestly, keep the batch alive
            emit(key, "error:", str(e)[:120])


if __name__ == "__main__":
    main()
