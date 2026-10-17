# Jerith sandbox image for attachment reading (bot feature: files the
# Discord transports hand over are read INSIDE a disposable container —
# see crates/bot/src/sandbox.rs for the runtime safety flags).
#
# Base: the migrated OpenClaw stack sandbox (python3 present, already on
# the host). We add only the text-extraction tools the extractor script
# uses: poppler-utils (pdftotext for .pdf). No network packages beyond
# that; the extractor itself is stdlib-only and rides the bot binary.
FROM openclaw-sandbox:bookworm-slim

# The base bakes in a non-root USER — flip back for the package install
# (the runtime container doesn't use apt at all; uid 65534 comes from the
# `docker run --user` flag, not the image).
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends poppler-utils ca-certificates \
    && rm -rf /var/lib/apt/lists/*
