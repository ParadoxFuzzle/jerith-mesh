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

# Archive members: per-member byte cap (a crafted member can't flood the
# story) and the text-like extensions we dearchivate.
MAX_MEMBER_BYTES = 256 * 1024
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".gz", ".bz2", ".xz", ".7z", ".rar")
ZIP_TEXT_EXTS = {
    "txt", "md", "rst", "csv", "tsv", "json", "yaml", "yml", "toml", "ini",
    "log", "xml", "html", "htm", "svg", "py", "js", "ts", "rs", "go", "c",
    "cpp", "h", "hpp", "sh", "bat", "sql", "tex", "srt", "vtt",
}


def is_text_like(name):
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return ext in ZIP_TEXT_EXTS


def cap_listing(names, n=20):
    return ", ".join(names[:n]) + ("…" if len(names) > n else "")


def if_(cond, a, b):
    return a if cond else b


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
    """Listing first, then the text-like members — nested files are the
    interest, not just the manifest. One level deep, honestly bounded."""
    try:
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            story = ["archive (%d files): %s" % (len(names), cap_listing(names))]
            text_members = [n for n in names if is_text_like(n)]
            picked = [m for m in text_members if not m.lower().endswith(ARCHIVE_SUFFIXES)][:6]
            for m in picked:
                info = zf.getinfo(m)
                if info.file_size > MAX_MEMBER_BYTES:
                    story.append("[%s] (member too large: %d bytes — not extracted)" % (m, info.file_size))
                    continue
                body = zf.read(m).decode("utf-8", errors="replace").strip()
                story.append("[%s] %s" % (m, if_(body == "", "(empty)", cap_text(body))))
            nested = [n for n in text_members if n.lower().endswith(ARCHIVE_SUFFIXES)]
            if nested:
                story.append("(nested archives skipped: %d — %s)" % (len(nested), cap_listing(nested)))
            skipped = len(text_members) - len(picked) - len(nested)
            if skipped > 0:
                story.append("(+%d text members skipped — member cap)" % skipped)
    except zipfile.BadZipFile:
        return ("binary", "not a valid archive")
    except Exception as e:  # noqa: BLE001
        return ("error", str(e)[:120])
    return ("ok", "\n".join(story))


def read_tar(path):
    """tar/tar.gz/tgz/tar.bz2/tar.xz through stdlib tarfile (no unpack to
    disk — member bodies are read in-memory and bounded)."""
    import tarfile

    try:
        with tarfile.open(path, "r:*") as tf:
            files = [m for m in tf.getmembers() if m.isreg()]
            story = ["archive (%d files): %s" % (len(files), cap_listing([m.name for m in files]))]
            text_members = [m for m in files if is_text_like(m.name)]
            picked = [m for m in text_members if not m.name.lower().endswith(ARCHIVE_SUFFIXES)][:6]
            for m in picked:
                if m.size > MAX_MEMBER_BYTES:
                    story.append("[%s] (member too large: %d bytes — not extracted)" % (m.name, m.size))
                    continue
                f = tf.extractfile(m)
                body = (f.read(1024 * 1024).decode("utf-8", errors="replace").strip() if f else "")
                story.append("[%s] %s" % (m.name, if_(body == "", "(empty)", cap_text(body))))
            skipped = len(text_members) - len(picked)
            if skipped > 0:
                story.append("(+%d text members skipped — member cap)" % skipped)
    except tarfile.TarError:
        return ("binary", "not a valid tar archive")
    except Exception as e:  # noqa: BLE001
        return ("error", str(e)[:120])
    return ("ok", "\n".join(story))


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
            elif label.lower().endswith(".zip"):
                emit(key, *read_zip(path))
            elif label.lower().endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
                emit(key, *read_tar(path))
            elif ext in OOXML or ext == "pptx":
                emit(key, *read_ooxml(path, ext))
            else:
                emit(key, *read_text(path))
        except Exception as e:  # noqa: BLE001 — report honestly, keep the batch alive
            emit(key, "error:", str(e)[:120])


if __name__ == "__main__":
    main()
