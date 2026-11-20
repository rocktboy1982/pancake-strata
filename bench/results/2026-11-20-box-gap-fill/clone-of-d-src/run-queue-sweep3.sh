#!/bin/sh
# Flag queue 3 (owner orders 2026-11-20): replaces run-queue-sweep2.sh (stopped with the old slice rd 2/3 at 08:00).
#   1. RETEST slices, every flag confirmed ONE BY ONE in its own paired block (base, flag x 3 interleaved, base; bench-retest.v1.sh,
#      tools/pancake/box_chains/retest_groups.json): rdC-1, rdC-2, mixC-3 (RD priority list, then US), usC-4, mixR-5 (US core flags, the RD
#      spec 250 bundle reference, POOL_FUSE_PHASES, PF_F16=1 vs 2, hipBLASLt 100401), mixR-6 (the other spec 251 arms, the US 222 updated-exact
#      bundle, the --lookup-max-t quality gate), usR-7 (spec 223 completed);
#   2. blocked pass 1 + per-flag paired pass 2 (bench-sweep.v3.sh): rdB-1of4 .. rdB-4of4, usB-1of9 .. usB-9of9, rdR-1of2 .. rdR-2of2;
#   3. the carried flags (pass 2 over the budget, drifted blocks): carry-rd, carry-us (bench-retest.v1.sh).
# Each run under bench-sched.v8.sh (BENCH_ONLY=EXTRA, BENCH_NO_RESUME=1: the AA shell stays frozen), pancake.service stays down; a failed
# slice is logged and the next one runs.  Harness pinned to one commit.  Log run-queue-sweep3.log, last line QUEUE_SWEEP3_DONE.
cd /home/dan/clone-of-d/src || exit 1
L=/home/dan/clone-of-d/src/run-queue-sweep3.log
say() { echo "$*" >> $L; }
export SWEEP_HARNESS_SHA=a58318c60f278bbaa366c8de54ab224a19664ce9
M=/home/dan/auto-runs/benchsweep/work/rd-p1-1of3.json.meta.json
say "QUEUE_SWEEP3 START $(date +%F_%T) harness $SWEEP_HARNESS_SHA"
n=0; N=24
check() {  # check LOG MARKER NAME rc
  last=$(tail -n 1 $1 2>/dev/null)
  if [ $4 -ne 0 ] || [ "$last" != "$2" ] || tail -n 4 $1 | grep -q "${2%_DONE}_ABORTED"; then
    say "ERROR $n/$N $3 rc=$4 last-line=$last (see $1 / bench-sched.log): next slice"; fi
}
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
for s in rdC-1 rdC-2 mixC-3 usC-4 mixR-5 mixR-6 usR-7; do retest $s; done
[ -f $M ] || say "ERROR $M is missing: the rdB / rdR slices cannot cut their lists"
for k in 1 2 3 4; do sweep rd $k/4 rdB-${k}of4 SWEEP_EXCLUDE_META=$M; done
for k in 1 2 3 4 5 6 7 8 9; do sweep us $k/9 usB-${k}of9; done
for k in 1 2; do sweep rd $k/2 rdR-${k}of2 SWEEP_ONLY_META=$M; done
retest carry-rd
retest carry-us
say "QUEUE_SWEEP3_DONE $(date +%F_%T)"
