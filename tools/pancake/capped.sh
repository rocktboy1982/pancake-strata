#!/bin/bash
# capped.sh - run a strata command in its own memory-capped systemd scope.
#
# On 2026-11-08 a strata run inside the terminal's cgroup drove the 57 GB PC into memory pressure and systemd-oomd killed
# the whole terminal (and everything running in it).  In its own scope, oomd kills only the engine.  The engine also
# keeps STRATA_RAM_KEEP_GIB free for the desktop, and the scope's MemoryMax is a hard backstop.  The start waits until
# the RAM of a previous run is given back (a run started on top of one still exiting was killed during its load).
#     tools/pancake/capped.sh [-n] CMD [ARGS...]      env: STRATA_RAM_KEEP_GIB (12), CAP (54G), WAIT_GIB (48)
#     -n: print the command it would run; wait for nothing, start nothing.
# Before the start it prints one line to stderr, "capped: tier=arena|mmap keep_gib=N big_free_gib=M", and exports
# STRATA_TIER_GUESS=arena|mmap for tools that record the tier.  The arena tier needs keep <= 12 and at least 33 GiB free
# in blocks of 2 MiB or more (/proc/buddyinfo, order >= 9); otherwise mmap (~8% slower decode, but it loads without a
# defragmented RAM).  It is a guess from the free blocks at the start, not the engine's own decision.
avail() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
big_free_gib() {
  awk -v pg="$(getconf PAGESIZE)" '/^Node/{for (i = 5; i <= NF; i++) if (i - 5 >= 9) b += $i * 2^(i - 5) * pg} END{print int(b / 2^30)}' /proc/buddyinfo
}
DRY=0; [ "$1" = "-n" ] && { DRY=1; shift; }
if [ $DRY = 0 ]; then
  for _ in $(seq 120); do [ "$(avail)" -ge "${WAIT_GIB:-48}" ] && break; sleep 5; done
  [ "$(avail)" -ge "${WAIT_GIB:-48}" ] || echo "capped.sh: only $(avail) GiB available after 10 min, starting anyway" >&2
fi
keep=${STRATA_RAM_KEEP_GIB:-12}
big=$(big_free_gib)
tier=mmap
awk -v k="$keep" -v b="$big" 'BEGIN { exit !(k + 0 <= 12 && b + 0 >= 33) }' && tier=arena
echo "capped: tier=$tier keep_gib=$keep big_free_gib=$big" >&2
export STRATA_TIER_GUESS=$tier
cmd=(systemd-run --user --scope -q -p MemoryMax="${CAP:-54G}" -p MemorySwapMax=0 --
     env STRATA_RAM_KEEP_GIB="$keep" STRATA_TIER_GUESS="$tier" "$@")
if [ $DRY = 1 ]; then printf 'would run:'; printf ' %q' "${cmd[@]}"; echo; exit 0; fi
exec "${cmd[@]}"
