# jerith-mesh long-tail example (FEATURES §17.1 recipe, exercised) — a
# REAL self-hosted app baked onto the driver-VM layout. The app here is
# sqlite-web (charlesleifer/sqlite-web — a self-hosted SQLite web
# front-end): rename/extend this image for any long-tail software that
# serves a web UI on localhost.
#
# Build:  docker build -t jerith-longtail-vm:latest images/driver-vm-longtail/
FROM jerith-driver-vm:latest

ENV DEBIAN_FRONTEND=noninteractive

# The base image drops to its non-root `chrome` user; the install layers
# need root (apt lists + pip), then the runtime user is restored so the
# seeded db and the app run as chrome, not root.
USER root

# python3 + pip bring sqlite-web (pulled from PyPI at build time).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --break-system-packages --no-cache-dir sqlite-web

# The baked app's boot glue: start sqlite-web on THIS container's
# loopback (the browser the engine drives reaches it at
# http://127.0.0.1:8080), then hand over to the driver-VM layout's
# window-manager bootstrap — Xvfb → openbox → VNC → CDP relay → browser.
COPY longtail-entrypoint.sh /longtail-entrypoint.sh
RUN chmod 755 /longtail-entrypoint.sh

# A seeded demo database: one table, a couple of rows — enough surface
# for record/replay. Replace at runtime with a manifest mount
# (host.sqlite:/home/chrome/demo:rw) for owner data.
COPY seed_db.py /tmp/seed_db.py
RUN chmod 644 /tmp/seed_db.py
USER chrome

RUN python3 /tmp/seed_db.py
USER root
RUN rm /tmp/seed_db.py
USER chrome

ENV JERITH_LONGTAIL_DB=/home/chrome/demo

ENTRYPOINT ["/longtail-entrypoint.sh"]
