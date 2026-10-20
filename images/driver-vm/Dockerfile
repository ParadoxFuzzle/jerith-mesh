# jerith-mesh driver VM (FEATURES §17.1) — a disposable desktop for the
# workflow record/replay engine: Xvfb + a real window manager (openbox) +
# a VNC preview + chromium, driven over CDP the same as any other target.
# Build:  docker build -t jerith-driver-vm:latest images/driver-vm/
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        chromium xvfb openbox x11vnc socat && \
    rm -rf /var/lib/apt/lists/*

# Non-root browser user (the entrypoint's Xvfb and WM run as this user).
RUN useradd -m chrome

COPY entrypoint.sh /entrypoint.sh
RUN chmod 755 /entrypoint.sh

USER chrome
ENV HOME=/home/chrome

# The §17.1 window-manager bootstrap runs first, then execs the browser
# argv the engine passes (see entrypoint.sh for the honest fallbacks).
ENTRYPOINT ["/entrypoint.sh"]
