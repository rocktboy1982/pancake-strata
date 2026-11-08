#!/bin/bash
# capped.sh - run a strata command in its own memory-capped systemd scope.
#
# On 2026-11-08 a strata run inside the terminal's cgroup drove the 57 GB PC into memory pressure and systemd-oomd killed
# the whole terminal (and everything running in it).  In its own scope, oomd kills only the engine.  The engine also
# keeps STRATA_RAM_KEEP_GIB free for the desktop, and the scope's MemoryMax is a hard backstop.  The start waits until
# the RAM of a previous run is given back (a run started on top of one still exiting was killed during its load).
#     tools/pancake/capped.sh CMD [ARGS...]      env: STRATA_RAM_KEEP_GIB (12), CAP (54G), WAIT_GIB (48)
avail() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
for _ in $(seq 120); do [ "$(avail)" -ge "${WAIT_GIB:-48}" ] && break; sleep 5; done
[ "$(avail)" -ge "${WAIT_GIB:-48}" ] || echo "capped.sh: only $(avail) GiB available after 10 min, starting anyway" >&2
exec systemd-run --user --scope -q -p MemoryMax="${CAP:-54G}" -p MemorySwapMax=0 -- \
    env STRATA_RAM_KEEP_GIB="${STRATA_RAM_KEEP_GIB:-12}" "$@"
