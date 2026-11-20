#!/bin/sh
# Flag-sweep queue, BLOCKED DESIGN (2026-11-20, TEST_LEDGER.md T-1002): replaces run-queue-sweep.sh (one-run design, its results void by base drift).
# Waits until the running old-design slice (sh ./bench-sweep.v1.sh under sh ./bench-sched.v8.sh) has ended, then runs bench-sweep.v2.sh under
# bench-sched.v8.sh, slice by slice (16-17 arms each = 3 blocks of <= 6, each block: base, arms in random order, base; <= 2.5 h per slice):
#   rd rest   rdB-1of4 .. rdB-4of4  (the 65 RD arms not in old slice 1: SWEEP_EXCLUDE_META = work/rd-p1-1of3.json.meta.json; includes old slice 2/3's arms)
#   us all    usB-1of9 .. usB-9of9  (all 153 US arms; old us 1/5 is void)
#   rd redo   rdR-1of2 .. rdR-2of2  (the 33 arms of old RD slice 1: its 8 RETEST, the KEEP and the 24 DROP; SWEEP_ONLY_META = the same meta)
# AA stays frozen (BENCH_NO_RESUME=1), pancake.service stays down.  A slice that fails is logged and the next one runs.  Harness pinned to one commit.
# Log /home/dan/clone-of-d/src/run-queue-sweep2.log, last line QUEUE_SWEEP2_DONE.
cd /home/dan/clone-of-d/src || exit 1
L=/home/dan/clone-of-d/src/run-queue-sweep2.log
say() { echo "$*" >> $L; }
export SWEEP_HARNESS_SHA=ae314fd45546f209afca09a2c34c4a63349cab15
M=/home/dan/auto-runs/benchsweep/work/rd-p1-1of3.json.meta.json
say "QUEUE_SWEEP2 START $(date +%F_%T) harness $SWEEP_HARNESS_SHA"
while pgrep -fx 'sh ./bench-sweep.v1.sh' >/dev/null 2>&1 || pgrep -fx 'sh ./bench-sched.v8.sh' >/dev/null 2>&1; do sleep 30; done
say "the old-design slice has ended $(date +%F_%T) (bench-sweep.v1.sh and its scheduler gone)"
[ -f $M ] || { say "ERROR $M is missing: nothing run"; say "QUEUE_SWEEP2_DONE $(date +%F_%T)"; exit 1; }
n=0; N=15
run() {  # run BRANCH K/N TAG [VAR=VALUE ...]
  b=$1; k=$2; tag=$3; shift 3
  n=$((n + 1))
  say "START $n/$N $b $tag ($k) $(date +%F_%T)"
  env "$@" SWEEP_BRANCH=$b SWEEP_SLICE=$k SWEEP_SPLIT=even SWEEP_TAG=$tag BENCH_ONLY=EXTRA BENCH_EXTRA=./bench-sweep.v2.sh BENCH_NO_RESUME=1 sh ./bench-sched.v8.sh; rc=$?
  last=$(tail -n 1 bench-sweep.log 2>/dev/null)
  if [ $rc -ne 0 ] || [ "$last" != WINBENCHSWEEP_DONE ] || tail -n 4 bench-sweep.log | grep -q WINBENCHSWEEP_ABORTED; then
    say "ERROR $n/$N $b $tag rc=$rc last-sweep-line=$last (see bench-sweep.log / bench-sched.log): next slice"
  fi
  say "END $n/$N $b $tag rc=$rc $(date +%F_%T)"
}
for k in 1 2 3 4; do run rd $k/4 rdB-${k}of4 SWEEP_EXCLUDE_META=$M; done
for k in 1 2 3 4 5 6 7 8 9; do run us $k/9 usB-${k}of9; done
for k in 1 2; do run rd $k/2 rdR-${k}of2 SWEEP_ONLY_META=$M; done
say "QUEUE_SWEEP2_DONE $(date +%F_%T)"
