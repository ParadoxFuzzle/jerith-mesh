#!/bin/sh
# jerith-mesh long-tail entrypoint (driver-VM-layout recipe) — starts the
# baked app on this container's loopback, then hands over to the layout's
# window-manager bootstrap (which execs the browser argv). The app dies
# with the container; anything an experiment must keep lives on a
# manifest-listed mount (the laid-out contract, exercised by the demo db).
set -u

: "${JERITH_LONGTAIL_DB:=/home/chrome/demo}"
: "${JERITH_LONGTAIL_PORT:=8080}"

if command -v sqlite_web >/dev/null 2>&1; then
    echo "[jerith-longtail] sqlite_web on 127.0.0.1:${JERITH_LONGTAIL_PORT} db=${JERITH_LONGTAIL_DB}" >&2
    # Bare argv only: flags like --no-password / --url-prefix are version-
    # dependent and died silently on this sqlite-web until the app's own
    # stderr was heard — the app logs go to the container stderr now.
    sqlite_web \
        --host 127.0.0.1 \
        --port "${JERITH_LONGTAIL_PORT}" \
        "${JERITH_LONGTAIL_DB}" \
        >/home/chrome/.sqlite_web.log 2>&1 &
    echo "[jerith-longtail] app log: /home/chrome/.sqlite_web.log" >&2
    sleep 1
else
    echo "[jerith-longtail] sqlite_web missing — baked app did not install?" >&2
fi

# Handover: the driver-VM layout bootstrap takes the browser argv from here.
exec /entrypoint.sh "$@"
