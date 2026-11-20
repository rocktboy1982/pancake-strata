#!/bin/sh
# Flag-sweep queue (owner order 2026-11-20, autonomy): the 8 slices of bench-sweep.v1.sh one after the other, priority order first
# (rd 1/3, us 1/5 = all [arm] + [named] switches; then rd 2/3, rd 3/3, us 2/5 .. us 5/5).  AA stays frozen (BENCH_NO_RESUME=1),
# pancake.service stays down.  A slice that fails is logged and the next one runs.  Harness pinned to one commit (same registry for all slices).
# Log /home/dan/clone-of-d/src/run-queue-sweep.log, last line QUEUE_SWEEP_DONE.
cd /home/dan/clone-of-d/src || exit 1
L=/home/dan/clone-of-d/src/run-queue-sweep.log
say() { echo "$*" >> $L; }
export SWEEP_HARNESS_SHA=e56f9dc124b11497ed385ea16c694ca2a1e1ce87
say "QUEUE_SWEEP START $(date +%F_%T) harness $SWEEP_HARNESS_SHA"
while pgrep -f "[b]ench-sched" >/dev/null 2>&1; do sleep 20; done
n=0
for s in rd:1/3 us:1/5 rd:2/3 rd:3/3 us:2/5 us:3/5 us:4/5 us:5/5; do
  n=$((n + 1)); b=${s%%:*}; k=${s#*:}
  say "START $n/8 $b $k $(date +%F_%T)"
  SWEEP_BRANCH=$b SWEEP_SLICE=$k BENCH_ONLY=EXTRA BENCH_EXTRA=./bench-sweep.v1.sh BENCH_NO_RESUME=1 sh ./bench-sched.v8.sh; rc=$?
  last=$(tail -n 1 bench-sweep.log 2>/dev/null)
  if [ $rc -ne 0 ] || [ "$last" != WINBENCHSWEEP_DONE ] || tail -n 4 bench-sweep.log | grep -q WINBENCHSWEEP_ABORTED; then
    say "ERROR $n/8 $b $k rc=$rc last-sweep-line=$last (see bench-sweep.log / bench-sched.log): next slice"
  fi
  say "END $n/8 $b $k rc=$rc $(date +%F_%T)"
done
say "QUEUE_SWEEP_DONE $(date +%F_%T)"
