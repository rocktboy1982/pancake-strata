#!/bin/sh
# US-ONLY queue, REORDERED (orchestrator decision 2026-11-20 under owner autonomy: long queues are unacceptable, headline first; TEST_LEDGER.md T-1005).
# Replaces the sequencer run-queue-us.sh (killed; ONLY that shell: its running slice usT-1 and the scheduler / chain of it were NOT touched).  This script first
# waits until the scheduler and chain of usT-1 are gone, then runs:
#   1 PRELIMINARY headline comparison (bench-final-us.v1.sh with FINAL_PREFIX=prelim FINAL_REPS=3 FINAL_NO_TRIAGE=1): stock upstream v0.2.16 against updated-strata
#     with the arms of specs 220 / 222 as shipped NOW (no triage applied), lossy + exact, short / edit + long 10K / 32K / 64K / 100K, 3 reps, + quality gate (<= 1 h)
#   2-4 retest usT-2, usT-3, usT-4
#   5 FINAL comparison (5 reps, the shipped flags after apply_triage on the US triage, + quality gate)  <- right after the retest stage
#   6-14 usB-1of9 .. usB-9of9, 15 carry-us (the long tail, after the final)
# Same rules as run-queue-us.sh (BENCH_NO_RESUME=1: the AA shell stays frozen, pancake.service stays down, a failed slice is logged and the next one runs).
# Log run-queue-us2.log, last line QUEUE_US2_DONE.
cd /home/dan/clone-of-d/src || exit 1
L=/home/dan/clone-of-d/src/run-queue-us2.log
say() { echo "$*" >> $L; }
: ${SWEEP_HARNESS_SHA:?SWEEP_HARNESS_SHA must be set (the pinned harness commit)}
export SWEEP_HARNESS_SHA RETEST_NO_RD=1 RETEST_MAX_S=${RETEST_MAX_S:-10800}
n=0; N=15
say "QUEUE_US2 START $(date +%F_%T) harness $SWEEP_HARNESS_SHA; waiting for the scheduler of usT-1 (slice 1, run-queue-us.sh); then $N slices, US only"
check() {  # check LOG MARKER NAME rc
  last=$(tail -n 1 $1 2>/dev/null)
  if [ $4 -ne 0 ] || [ "$last" != "$2" ] || tail -n 4 $1 | grep -q "${2%_DONE}_ABORTED"; then
    say "ERROR $n/$N $3 rc=$4 last-line=$last (see $1 / bench-sched.log): next slice"; fi
}
while pgrep -fx "sh ./bench-sched.v8.sh" >/dev/null 2>&1 || pgrep -fx "sh ./bench-retest.v2.sh" >/dev/null 2>&1; do sleep 20; done
say "END usT-1 (slice 1 of run-queue-us.sh) $(date +%F_%T): $(tail -n 1 bench-retest.v2.log)"
retest() {  # retest SLICE
  n=$((n + 1)); say "START $n/$N retest $1 $(date +%F_%T)"
  env RETEST_SLICE=$1 BENCH_ONLY=EXTRA BENCH_EXTRA=./bench-retest.v2.sh BENCH_NO_RESUME=1 sh ./bench-sched.v8.sh; rc=$?
  check bench-retest.v2.log WINBENCHRETEST_DONE "retest $1" $rc
  say "END $n/$N retest $1 rc=$rc $(date +%F_%T)"
}
sweep() {  # sweep BRANCH K/N TAG [VAR=VALUE ...]
  b=$1; k=$2; tag=$3; shift 3
  n=$((n + 1)); say "START $n/$N sweep $b $tag ($k) $(date +%F_%T)"
  env "$@" SWEEP_BRANCH=$b SWEEP_SLICE=$k SWEEP_SPLIT=even SWEEP_TAG=$tag BENCH_ONLY=EXTRA BENCH_EXTRA=./bench-sweep.v3.sh BENCH_NO_RESUME=1 sh ./bench-sched.v8.sh; rc=$?
  check bench-sweep.log WINBENCHSWEEP_DONE "sweep $b $tag" $rc
  say "END $n/$N sweep $b $tag rc=$rc $(date +%F_%T)"
}
final() {  # final NAME [VAR=VALUE ...]
  nm=$1; shift
  n=$((n + 1)); say "START $n/$N $nm comparison (US only) $(date +%F_%T)"
  env "$@" BENCH_ONLY=EXTRA BENCH_EXTRA=./bench-final-us.v1.sh BENCH_NO_RESUME=1 sh ./bench-sched.v8.sh; rc=$?
  check bench-final-us.v1.log WINBENCHFINALUS_DONE "$nm comparison" $rc
  say "END $n/$N $nm comparison rc=$rc $(date +%F_%T)"
}
final preliminary FINAL_PREFIX=prelim FINAL_REPS=3 FINAL_NO_TRIAGE=1 FINAL_GATE=1
for s in usT-2 usT-3 usT-4; do retest $s; done
final final FINAL_PREFIX=final FINAL_REPS=5
for k in 1 2 3 4 5 6 7 8 9; do sweep us $k/9 usB-${k}of9; done
retest carry-us
say "QUEUE_US2_DONE $(date +%F_%T)"
