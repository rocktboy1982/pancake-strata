#!/bin/sh
# Flag queue 4 (2026-11-20, TEST_LEDGER.md T-1004): continues run-queue-sweep3.sh from slice 2/24 with the warm-up start in every block and the
# drift judged on the last request of each start (harness pinned to the T-1004 commit).  Slice 1 (retest rdC-1) was started by run-queue-sweep3.sh
# and runs to its end; this script waits for its scheduler (bench-sched.v8.sh) and chain (bench-retest.v1.sh) to be gone, logs its end, then runs
# slices 2..24 exactly as run-queue-sweep3.sh would have: retest mixC-3.. (rdC-2 first), blocked sweeps, carry slices.
# Each run under bench-sched.v8.sh (BENCH_ONLY=EXTRA, BENCH_NO_RESUME=1: the AA shell stays frozen), pancake.service stays down; a failed
# slice is logged and the next one runs.  Log run-queue-sweep4.log, last line QUEUE_SWEEP4_DONE.
cd /home/dan/clone-of-d/src || exit 1
L=/home/dan/clone-of-d/src/run-queue-sweep4.log
say() { echo "$*" >> $L; }
export SWEEP_HARNESS_SHA=5ae76fbc8702d7dfdeb6c5fc48029a2073db0be4
M=/home/dan/auto-runs/benchsweep/work/rd-p1-1of3.json.meta.json
say "QUEUE_SWEEP4 START $(date +%F_%T) harness $SWEEP_HARNESS_SHA; waiting for slice 1/24 (retest rdC-1, started by run-queue-sweep3.sh)"
n=1; N=24
check() {  # check LOG MARKER NAME rc
  last=$(tail -n 1 $1 2>/dev/null)
  if [ $4 -ne 0 ] || [ "$last" != "$2" ] || tail -n 4 $1 | grep -q "${2%_DONE}_ABORTED"; then
    say "ERROR $n/$N $3 rc=$4 last-line=$last (see $1 / bench-sched.log): next slice"; fi
}
while pgrep -fx "sh ./bench-sched.v8.sh" >/dev/null 2>&1 || pgrep -fx "sh ./bench-retest.v1.sh" >/dev/null 2>&1; do sleep 20; done
n=1; check bench-retest.v1.log WINBENCHRETEST_DONE "retest rdC-1 (slice 1, run-queue-sweep3.sh)" 0
say "END 1/$N retest rdC-1 (run-queue-sweep3.sh) $(date +%F_%T)"
retest() {  # retest SLICE
  n=$((n + 1)); say "START $n/$N retest $1 $(date +%F_%T)"
  env RETEST_SLICE=$1 BENCH_ONLY=EXTRA BENCH_EXTRA=./bench-retest.v1.sh BENCH_NO_RESUME=1 sh ./bench-sched.v8.sh; rc=$?
  check bench-retest.v1.log WINBENCHRETEST_DONE "retest $1" $rc
  say "END $n/$N retest $1 rc=$rc $(date +%F_%T)"
}
sweep() {  # sweep BRANCH K/N TAG [VAR=VALUE ...]
  b=$1; k=$2; tag=$3; shift 3
  n=$((n + 1)); say "START $n/$N sweep $b $tag ($k) $(date +%F_%T)"
  env "$@" SWEEP_BRANCH=$b SWEEP_SLICE=$k SWEEP_SPLIT=even SWEEP_TAG=$tag BENCH_ONLY=EXTRA BENCH_EXTRA=./bench-sweep.v3.sh BENCH_NO_RESUME=1 sh ./bench-sched.v8.sh; rc=$?
  check bench-sweep.log WINBENCHSWEEP_DONE "sweep $b $tag" $rc
  say "END $n/$N sweep $b $tag rc=$rc $(date +%F_%T)"
}
for s in rdC-2 mixC-3 usC-4 mixR-5 mixR-6 usR-7; do retest $s; done
[ -f $M ] || say "ERROR $M is missing: the rdB / rdR slices cannot cut their lists"
for k in 1 2 3 4; do sweep rd $k/4 rdB-${k}of4 SWEEP_EXCLUDE_META=$M; done
for k in 1 2 3 4 5 6 7 8 9; do sweep us $k/9 usB-${k}of9; done
for k in 1 2; do sweep rd $k/2 rdR-${k}of2 SWEEP_ONLY_META=$M; done
retest carry-rd
retest carry-us
say "QUEUE_SWEEP4_DONE $(date +%F_%T)"
