#!/usr/bin/env bash
# Guard for the 2026-10-27 wrong-tunnel incident (RUNBOOK §"Cloudflare
# Tunnel hardening + the two-week nmonit 530"): `cloudflared tunnel run
# NAME` without --config silently reads the default config file, whose
# `tunnel:` key makes the connector join a DIFFERENT tunnel. nmonit sat
# broken for two weeks behind exactly this bug.
#
# Checks every *cloudflared*.service unit in a systemd unit directory
# (default: this user's ~/.config/systemd/user):
#   1. any ExecStart that runs a tunnel must declare --config
#   2. that config file must exist and parse enough to name a tunnel id
#   3. each pinned config targets a DISTINCT tunnel id, and its
#      credentials-file exists
# Fails hard (exit 1) on any violation. Wire it into the oneshot user
# unit deploy/systemd/jerith-cloudflared-guard.service so a bad unit
# edit turns red at boot instead of quietly hijacking another tunnel.
# CI cannot see host systemd units, so this guard is the enforcement
# point — install it on every host running cloudflared (Jetson included).
set -euo pipefail

units_dir="${1:-$HOME/.config/systemd/user}"
shopt -s nullglob
units=("$units_dir"/*cloudflared*.service)

if [ ${#units[@]} -eq 0 ]; then
	echo "check-cloudflared-units: no *cloudflared*.service units in $units_dir — nothing to check"
	exit 0
fi

declare -A tunnel_owner=()
fail=0

for unit in "${units[@]}"; do
	name=$(basename "$unit")

	# First ExecStart line; systemd spec escapes literal spaces as \x20.
	exec_line=$(awk '/^ExecStart=/{sub(/^ExecStart=/,""); gsub(/\\x20/, " "); print; exit}' "$unit")
	if [ -z "$exec_line" ]; then
		echo "FAIL $name: no ExecStart line found"
		fail=1
		continue
	fi

	read -r -a argv <<<"$exec_line"
	is_tunnel_run=false
	for a in "${argv[@]}"; do
		case "$a" in
		tunnel) is_tunnel_run=true ;;
		esac
	done
	if ! $is_tunnel_run; then
		echo "PASS $name: ExecStart is not a tunnel runner (skipped)"
		continue
	fi

	# Accept --config X / --config=X / -c X / -c=X (arg order may place it
	# anywhere around `tunnel`, e.g. `cloudflared tunnel --config X run N`).
	cfg=""
	i=0
	while [ $i -lt ${#argv[@]} ]; do
		case "${argv[$i]}" in
		--config=*) cfg="${argv[$i]#--config=}" ;;
		--config)
			[ $((i + 1)) -lt ${#argv[@]} ] && cfg="${argv[$i + 1]}"
			;;
		-c=*) cfg="${argv[$i]#-c=}" ;;
		-c)
			[ $((i + 1)) -lt ${#argv[@]} ] && cfg="${argv[$i + 1]}"
			;;
		esac
		[ -n "$cfg" ] && break
		i=$((i + 1))
	done

	if [ -z "$cfg" ]; then
		echo "FAIL $name: runs 'tunnel' without --config - it will read the default config, which may belong to a different tunnel (the nmonit 530 bug)"
		fail=1
		continue
	fi

	# Units may use systemd specifiers; expand the common %h (home).
	cfg=${cfg//%h/$HOME}

	if [ ! -f "$cfg" ]; then
		echo "FAIL $name: --config $cfg does not exist"
		fail=1
		continue
	fi

	tid=$(awk '/^tunnel:/{print $2; exit}' "$cfg")
	if [ -z "$tid" ]; then
		echo "FAIL $name: config $cfg has no 'tunnel:' id at top level"
		fail=1
		continue
	fi

	if [ -n "${tunnel_owner[$tid]:-}" ]; then
		echo "FAIL $name: config $cfg targets tunnel $tid, already claimed by ${tunnel_owner[$tid]} - one config file per connector"
		fail=1
	else
		tunnel_owner[$tid]="$name"
	fi

	creds=$(awk '/^credentials-file:/{print $2; exit}' "$cfg")
	creds=${creds//%h/$HOME}
	if [ -n "$creds" ] && [ ! -f "$creds" ]; then
		echo "FAIL $name: credentials-file $creds does not exist"
		fail=1
	fi
done

if [ "$fail" -ne 0 ]; then
	echo "check-cloudflared-units: FAILED — fix the units above; refusing to call this healthy"
	exit 1
fi

echo "check-cloudflared-units: PASS — ${#units[@]} unit(s) checked, all pinned to distinct tunnels"
