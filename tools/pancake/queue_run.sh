#!/bin/bash
# queue_run.sh - run abtest specs back to back, unattended (no idle GPU between specs).
#
#   tools/pancake/queue_run.sh [-n] [--night-only HH-HH] [--drop-cache] [--reclaim-max X] OUTDIR SPEC[:abtest args...] ...
#   e.g. tools/pancake/queue_run.sh --night-only 23-07 --drop-cache bench/results/2026-11-11-gpu \
#            bench/specs/19-turbo-guards.json "bench/specs/25-a8-confirm.json:--no-early-stop" bench/specs/20-verify-ecap.json:--tier=screen
#
# One job at a time: it refuses to start while another abtest/strata process runs, waits for MemAvailable >= MIN_GIB
# (default 30) before each spec, skips a spec that already has a summary in OUTDIR (resumable), and stops the whole queue
# if the machine is short of RAM after the wait (a run started on top of a RAM shortage gets killed by systemd-oomd).
# -n                  print what it would run.
# --night-only HH-HH  run only while the local hour is in the window: 23-07 wraps midnight, equal hours mean all day.
#                     Outside it the queue exits 5 at once (no waiting), so a cron can call it at any hour.  The window is
#                     checked again before each spec, so a queue does not run into the day.
# --drop-cache        run drop_cache.py once, before the first spec (non-root; more free 2 MiB blocks for the arena).
# --reclaim-max X     abtest's --reclaim-max for every spec that does not set its own (default 3.0).
# STRATA_RAM_KEEP_GIB defaults to 20 (the mmap tier: loads without a defragmented RAM).
# Env: MIN_GIB (30), STRATA_RAM_KEEP_GIB (20), PYTHON (.venv/bin/python), ABTEST (tools/pancake/abtest.py),
#      DROP_CACHE (tools/pancake/drop_cache.py), BUSY_PATTERN (the pgrep -f pattern that means "a test or engine runs").
# Exit: 0 finished, 2 usage, 3 another test or engine running, 4 RAM short after the wait, 5 outside the night window.
# Results: OUTDIR/<spec>.jsonl, <spec>.log, <spec>.summary.txt; the queue log is OUTDIR/queue.log; OUTDIR/queue.summary.txt
# has one line per spec that ran: exit code, run lines and minutes.
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/../.." || exit 1
usage() { sed -n '2,/^[^#]/p' "$HERE/queue_run.sh" | sed '$d'; exit 2; }
DRY=0; NIGHT=""; DROP=0; RECLAIM=3.0
while [ $# -gt 0 ]; do
  case $1 in
    -n) DRY=1 ;;
    --night-only) [ $# -ge 2 ] && [[ $2 =~ ^([01][0-9]|2[0-3])-([01][0-9]|2[0-3])$ ]] || usage; NIGHT=$2; shift ;;
    --drop-cache) DROP=1 ;;
    --reclaim-max) [ $# -ge 2 ] && [[ $2 =~ ^[0-9]+(\.[0-9]+)?$ ]] || usage; RECLAIM=$2; shift ;;
    -*) usage ;;
    *) break ;;
  esac
  shift
done
[ $# -ge 2 ] || usage
OUT=$1; shift
mkdir -p "$OUT"
MIN_GIB=${MIN_GIB:-30}
export STRATA_RAM_KEEP_GIB=${STRATA_RAM_KEEP_GIB:-20}
PYTHON=${PYTHON:-.venv/bin/python}
ABTEST=${ABTEST:-tools/pancake/abtest.py}
DROP_CACHE=${DROP_CACHE:-tools/pancake/drop_cache.py}
BUSY=${BUSY_PATTERN:-"tools/pancake/abtest.py|restore_check.py|bin/strata "}   # a process matching this stops the queue
avail() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
say() { echo "[$(date +%H:%M)] $*" | tee -a "$OUT/queue.log"; }
in_window() {   # true when the local hour is in the window HH-HH
  local s=$((10#${1%-*})) e=$((10#${1#*-})) h=$((10#$(date +%H)))
  if [ "$s" -lt "$e" ]; then [ "$h" -ge "$s" ] && [ "$h" -lt "$e" ]; else [ "$h" -ge "$s" ] || [ "$h" -lt "$e" ]; fi
}
need_window() { [ -z "$NIGHT" ] || in_window "$NIGHT" || { say "outside the night window $NIGHT: queue stopped"; exit 5; }; }
dropped=0
for item in "$@"; do
  spec=${item%%:*}; args=""; [ "$item" != "$spec" ] && args=${item#*:}
  args=${args//=/ }                       # tier=screen -> --tier screen style: "--tier=screen" becomes "--tier screen"
  case " $args " in *" --reclaim-max "*) ;; *) args="${args:+$args }--reclaim-max $RECLAIM" ;; esac
  n=$(basename "$spec" .json)
  need_window
  [ -f "$spec" ] || { say "$n: no such spec, skipped"; continue; }
  if [ -s "$OUT/$n.summary.txt" ]; then say "$n: done before, skipped"; continue; fi
  if [ $DRY = 1 ]; then
    if [ $DROP = 1 ] && [ $dropped = 0 ]; then say "would run: $PYTHON $DROP_CACHE"; dropped=1; fi
    say "would run: $PYTHON $ABTEST $spec --out $OUT/$n.jsonl $args"
    continue
  fi
  if pgrep -f "$BUSY" > /dev/null; then say "another test or engine is running: queue stopped"; exit 3; fi
  if [ $DROP = 1 ] && [ $dropped = 0 ]; then
    dropped=1
    say "drop_cache: start"
    "$PYTHON" "$DROP_CACHE" 2>&1 | tee -a "$OUT/queue.log"
    [ "${PIPESTATUS[0]}" = 0 ] || say "drop_cache failed: going on without it"
  fi
  for _ in $(seq 60); do [ "$(avail)" -ge "$MIN_GIB" ] && break; sleep 10; done
  [ "$(avail)" -ge "$MIN_GIB" ] || { say "$n: only $(avail) GiB available after 10 min: queue stopped"; exit 4; }
  need_window
  say "$n: start ($args)"
  start=$(date +%s)
  # shellcheck disable=SC2086
  "$PYTHON" "$ABTEST" "$spec" --out "$OUT/$n.jsonl" $args > "$OUT/$n.log" 2>&1
  rc=$?
  mins=$(awk -v s="$start" -v e="$(date +%s)" 'BEGIN { printf "%.1f", (e - s) / 60 }')
  sed -n '/^summary/,$p' "$OUT/$n.log" | grep -v "drafts accepted" > "$OUT/$n.summary.txt"
  runs=$(grep -c '^rep ' "$OUT/$n.log")
  echo "$n exit=$rc runs=$runs minutes=$mins" >> "$OUT/queue.summary.txt"
  say "$n: exit $rc, $runs run lines"
done
say "queue finished"
