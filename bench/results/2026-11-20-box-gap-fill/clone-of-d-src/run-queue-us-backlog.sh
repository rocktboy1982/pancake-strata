#!/bin/sh
# US BACKLOG queue (STAGE 4; NOT STARTED: owner 2026-11-20 deferred these switches to the week of 2026-11-23; TEST_LEDGER.md T-1005).  The 31 switches of
# sweep_candidates.json "backlog.us" (queue file us-backlog.json: KVA research x6, Q2_0 / Q3_K / Q5_0 / K-quant / IQ3_S / TQ / PLE other-quant-or-model paths, Q4_0-KV prompt
# attention, conversation checkpoint x5, admission, remote contexts, sampling, diagnostic) with the SAME design as run-queue-us3.sh: one arm each (minus-X for a preset flag, plus-X
# for an off-by-default one), 1 rep, blocks of <= 8 arms, ONE warm-up start per slice, base before + base-late after every block, 5 % drift guard (one re-run, then INCONCLUSIVE),
# cases short + edit + longdec64k; then the PROMISING flags ranked by their best case: >= 2.0 % (HIGH) get the compact paired 3-rep confirm first, 0 < gain < 2 % (PROMISING-LOW)
# after them; <= 0 % in a clean block is dropped.  Run it by hand when the box is free (no other benchmark queue): on the box
#   cd /home/dan/clone-of-d/src && SWEEP_HARNESS_SHA=<pinned commit> setsid -f sh ./run-queue-us-backlog.sh > /dev/null 2>&1 < /dev/null
# (needs bench-sweep.v4.sh, bench-retest.v3.sh, bench-sched.v8.sh next to it).  Log run-queue-us-backlog.log, last line QUEUE_US_BACKLOG_DONE.  Estimate: 2 screen slices ~2.3 h
# (31 arms + 2 x 4 blocks of overhead, 190 s a start), the confirms 16 min per flag.
cd /home/dan/clone-of-d/src || exit 1
L=/home/dan/clone-of-d/src/run-queue-us-backlog.log
say() { echo "$*" >> $L; }
: ${SWEEP_HARNESS_SHA:?SWEEP_HARNESS_SHA must be set (the pinned harness commit)}
O=/home/dan/auto-runs/benchsweep; HN=/mnt/pancake/benchsweep/harness; HPY=/home/dan/clone-of-d/src/Strata-win/.venv/bin/python
export SWEEP_HARNESS_SHA RETEST_NO_RD=1 SWEEP_MAX_S=${SWEEP_MAX_S:-9000} RETEST_MAX_S=${RETEST_MAX_S:-9000} BENCH_PREIDLED=1
export SWEEP_CASES=short,edit,longdec64k SWEEP_BLOCK=8 SWEEP_WARM=first RETEST_LAYOUT=compact RETEST_WARM=first SWEEP_SOURCE=backlog
CARRY_GLOB="$O/us-carry-usBk-*.json"
n=0
say "QUEUE_US_BACKLOG START $(date +%F_%T) harness $SWEEP_HARNESS_SHA; 2 screen slices (31 backlog arms), ranked confirms"
check() {  # check LOG MARKER NAME rc
  last=$(tail -n 1 $1 2>/dev/null)
  if [ $4 -ne 0 ] || [ "$last" != "$2" ] || tail -n 4 $1 | grep -q "${2%_DONE}_ABORTED"; then
    say "ERROR $n $3 rc=$4 last-line=$last (see $1 / bench-sched.log): next slice"; fi
}
screen() {  # screen K/N TAG
  n=$((n + 1)); say "START $n screen us $2 ($1) $(date +%F_%T)"
  env SWEEP_BRANCH=us SWEEP_SLICE=$1 SWEEP_SPLIT=even SWEEP_TAG=$2 SWEEP_SCREEN_ONLY=1 BENCH_ONLY=EXTRA BENCH_EXTRA=./bench-sweep.v4.sh BENCH_NO_RESUME=1 sh ./bench-sched.v8.sh; rc=$?
  check bench-sweep.log WINBENCHSWEEP_DONE "screen $2" $rc
  say "END $n screen $2 rc=$rc $(date +%F_%T)"
}
retest() {  # retest SLICE [VAR=VALUE ...]
  sl=$1; shift
  n=$((n + 1)); say "START $n retest $sl $(date +%F_%T)"
  env "$@" RETEST_SLICE=$sl BENCH_ONLY=EXTRA BENCH_EXTRA=./bench-retest.v3.sh BENCH_NO_RESUME=1 sh ./bench-sched.v8.sh; rc=$?
  check bench-retest.v3.log WINBENCHRETEST_DONE "retest $sl" $rc
  say "END $n retest $sl rc=$rc $(date +%F_%T)"
}
sw() { "$HPY" $HN/tools/pancake/box_chains/sweep.py "$@"; }
confirm() {  # confirm FIRST COUNT: the ranked flags [FIRST, FIRST+COUNT) in slices of 6
  first=$1; cnt=$2; i=0
  while [ $((i * 6)) -lt $cnt ]; do
    i=$((i + 1)); left=$((cnt - (i - 1) * 6)); [ $left -gt 6 ] && left=6
    retest carry-us-usBk-$((first + 1))-$i RETEST_CARRY_GLOB="$CARRY_GLOB" RETEST_CARRY_OFFSET=$((first + (i - 1) * 6)) RETEST_CARRY_MAX=$left
  done
}
for k in 1 2; do screen $k/2 usBk-${k}of2; done
rk=$(sw rank --carry $(ls $CARRY_GLOB 2>/dev/null) --high 2.0 --out $O/us-backlog-ranked.json 2>> $L | tee -a $L | sed -n 's/^RANKCUT //p')
cut=${rk% *}; tot=${rk#* }; cut=${cut:-0}; tot=${tot:-0}
say "BACKLOG RANK: $cut HIGH-gain flag(s) of $tot promising (ranked file $O/us-backlog-ranked.json)"
[ "$cut" -gt 0 ] && confirm 0 $cut
[ "$tot" -gt "$cut" ] && confirm $cut $((tot - cut))
say "QUEUE_US_BACKLOG_DONE $(date +%F_%T)"
