#!/bin/bash
# queue_run.sh - run abtest specs back to back, unattended (no idle GPU between specs).
#
#   tools/pancake/queue_run.sh [-n] OUTDIR SPEC[:abtest args...] ...
#   e.g. tools/pancake/queue_run.sh bench/results/2026-11-11-gpu bench/specs/19-turbo-guards.json \
#            "bench/specs/25-a8-confirm.json:--no-early-stop" bench/specs/20-verify-ecap.json:--tier=screen
#
# One job at a time: it refuses to start while another abtest/strata process runs, waits for MemAvailable >= MIN_GIB
# (default 30) before each spec, skips a spec that already has a summary in OUTDIR (resumable), and stops the whole queue
# if the machine is short of RAM after the wait (a run started on top of a RAM shortage gets killed by systemd-oomd).
# -n prints what it would run.  STRATA_RAM_KEEP_GIB defaults to 20 (the mmap tier: loads without a defragmented RAM).
# Results: OUTDIR/<spec>.jsonl, <spec>.log, <spec>.summary.txt;  the queue log is OUTDIR/queue.log.
cd "$(dirname "$0")/../.." || exit 1
DRY=0; [ "$1" = "-n" ] && { DRY=1; shift; }
OUT=$1; shift
[ -n "$OUT" ] && [ $# -gt 0 ] || { sed -n '2,12p' "$0"; exit 2; }
mkdir -p "$OUT"
MIN_GIB=${MIN_GIB:-30}
export STRATA_RAM_KEEP_GIB=${STRATA_RAM_KEEP_GIB:-20}
avail() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
say() { echo "[$(date +%H:%M)] $*" | tee -a "$OUT/queue.log"; }
for item in "$@"; do
  spec=${item%%:*}; args=""; [ "$item" != "$spec" ] && args=${item#*:}
  args=${args//=/ }                       # tier=screen -> --tier screen style: "--tier=screen" becomes "--tier screen"
  n=$(basename "$spec" .json)
  [ -f "$spec" ] || { say "$n: no such spec, skipped"; continue; }
  if [ -s "$OUT/$n.summary.txt" ]; then say "$n: done before, skipped"; continue; fi
  if [ $DRY = 1 ]; then say "would run: abtest.py $spec --out $OUT/$n.jsonl $args"; continue; fi
  if pgrep -f "tools/pancake/abtest.py|restore_check.py|bin/strata " > /dev/null; then say "another test or engine is running: queue stopped"; exit 3; fi
  for _ in $(seq 60); do [ "$(avail)" -ge "$MIN_GIB" ] && break; sleep 10; done
  [ "$(avail)" -ge "$MIN_GIB" ] || { say "$n: only $(avail) GiB available after 10 min: queue stopped"; exit 4; }
  say "$n: start ($args)"
  # shellcheck disable=SC2086
  .venv/bin/python tools/pancake/abtest.py "$spec" --out "$OUT/$n.jsonl" $args > "$OUT/$n.log" 2>&1
  rc=$?
  sed -n '/^summary/,$p' "$OUT/$n.log" | grep -v "drafts accepted" > "$OUT/$n.summary.txt"
  say "$n: exit $rc, $(grep -c '^rep ' "$OUT/$n.log") run lines"
done
say "queue finished"
