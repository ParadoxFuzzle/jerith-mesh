#!/bin/sh
# jerith-mesh driver-VM entrypoint (FEATURES §17.1) — the window-manager
# bootstrap for a disposable desktop the workflow engine can target
# instead of the host display.
#
# Boot order (each step degrades with a note on stderr, never a crash):
#   1. Xvfb on :1, sized by JERITH_VM_SIZE (default 1280x800x24).
#   2. openbox as the window manager when installed — real desktop so
#      menus/dialogs land INSIDE the VM (a WM-less X is the honest
#      fallback: fullscreen chrome still records/replays fine).
#   3. x11vnc on :5900 when installed — the owner watches the desktop
#      through the loopback-mapped 5900 while recording.
#   4. exec the browser argv the engine passes (chromium + CDP flags).
#
# Env knobs: JERITH_VM_SIZE (geometry), no other surface on purpose.
set -u

DISPLAY_NUM=1
export DISPLAY=":${DISPLAY_NUM}"

echo "[jerith-vm] booting Xvfb :${DISPLAY_NUM} ${JERITH_VM_SIZE:-1280x800}x24" >&2
Xvfb ":${DISPLAY_NUM}" -screen 0 "${JERITH_VM_SIZE:-1280x800}x24" -nolisten tcp &
sleep 1

if command -v openbox >/dev/null 2>&1; then
    openbox >/dev/null 2>&1 &
    echo "[jerith-vm] window manager: openbox" >&2
else
    echo "[jerith-vm] no window manager available — bare X (fullscreen browser)" >&2
fi
sleep 1

if command -v x11vnc >/dev/null 2>&1; then
    echo "[jerith-vm] VNC preview on :5900 (monitor your credentials)" >&2
    x11vnc -display ":${DISPLAY_NUM}" -forever -shared -rfbport 5900 -nopw -quiet \
        >/dev/null 2>&1 &
fi

# CDP relay on :9223 -> 127.0.0.1:9222 (socat). Needed because headful
# chromium IGNORES --remote-debugging-address and binds the DevTools
# endpoint to container-loopback only — docker's port map targets the
# container interface, not its loopback, so without the relay the engine
# could never reach the browser. socat forwards host -> container IP
# 9223 -> chrome's 127.0.0.1:9222 inside the same namespace.
if command -v socat >/dev/null 2>&1; then
    socat TCP-LISTEN:9223,fork,reuseaddr TCP:127.0.0.1:9222 >/dev/null 2>&1 &
    echo "[jerith-vm] CDP relay on :9223 (headful chrome binds loopback-only)" >&2
fi

# handover happens here: the engine's chromium argv becomes PID 1 of the
# desktop, so the session's lifetime IS the browser's (kill_on_drop).
exec "$@"
