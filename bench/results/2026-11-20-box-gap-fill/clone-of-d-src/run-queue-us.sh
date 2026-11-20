#!/bin/sh
# US-ONLY flag queue (owner order 2026-11-20 09:19: "stop doing the Strata-rdna3 tests, compare only to stock Strata"; TEST_LEDGER.md T-1005).
# Replaces run-queue-sweep4.sh (stopped with its slice rdC-1 before it ran slice 2; the RDNA3 branch is paused: no RD arm, no RD block, no RD file anywhere).
# Slices, each run under bench-sched.v8.sh (BENCH_ONLY=EXTRA, BENCH_NO_RESUME=1: the AA shell stays frozen), pancake.service stays down, a failed slice
# is logged and the next one runs; harness pinned by SWEEP_HARNESS_SHA (the commit that carries retest_groups.json usT-1..4, sweep.py final-specs, bench-retest.sh
# RETEST_NO_RD, bench-final-us.sh); the scripts on the box are bench-retest.v2.sh, bench-sweep.v3.sh, bench-final-us.v1.sh.
#   a. 4 US retest slices, every flag its own paired block (base, flag, base, flag, base, flag, base; the warm-up start rule of 5ae76fbc8 / T-1004):
#      1 usT-1  ENGINE_NICE, SELECT_WMMA, ATTN_DAILY, KV_PREFETCH, STAGER_RING
#      2 usT-2  PF_GROUP_SCRATCH, PROMPT_ATTN_IMMA, --lookup-max-t 12, GR_RDNA_Q8, GR_RDNA_A8
#      3 usT-3  GR_INJ_UP, GR_FUSED, GR_RDNA, RDNA_MMVQ_WAVE, RDNA_Q6K
#      4 usT-4  STRATA_HIPBLASLT_TUNING (100401), BUNDLE spec 222 updated-exact vs plain, quality gate updated vs updated minus --lookup-max-t, spec 223 completion
#   b. 5..13 the blocked pass 1 of the remaining untested US flags (153 arms): usB-1of9 .. usB-9of9 (per-flag paired confirm blocks inside each slice)
#   c. 14 carry-us (flags carried by the sweeps: pass 2 over budget or a drifted block)
#   d. 15 FINAL COMPARISON bench-final-us.v1.sh: stock upstream v0.2.16 against updated-strata with the final shipped flags (apply_triage on the US triage),
#      exact and lossy variant, short / edit (code edit) + long 10K / 32K / 64K / 100K, confirm 5 reps (--block 3 --warmup 1 --outlier robust --reclaim-max 3
#      --reps 5), plus the quality gate of the final arms against stock.  The headline result.
# Rules kept: flags judged individually, OPTIONAL for mixed results, DROP only from a clean pass 2, INCONCLUSIVE when the base drift is above 5 %.
# Log run-queue-us.log, last line QUEUE_US_DONE.
cd /home/dan/clone-of-d/src || exit 1
L=/home/dan/clone-of-d/src/run-queue-us.log
say() { echo "$*" >> $L; }
: ${SWEEP_HARNESS_SHA:?SWEEP_HARNESS_SHA must be set (the pinned harness commit)}
export SWEEP_HARNESS_SHA RETEST_NO_RD=1 RETEST_MAX_S=${RETEST_MAX_S:-10800}
n=0; N=15
say "QUEUE_US START $(date +%F_%T) harness $SWEEP_HARNESS_SHA; $N slices, US only (RETEST_NO_RD=1, RETEST_MAX_S=$RETEST_MAX_S)"
check() {  # check LOG MARKER NAME rc
  last=$(tail -n 1 $1 2>/dev/null)
  if [ $4 -ne 0 ] || [ "$last" != "$2" ] || tail -n 4 $1 | grep -q "${2%_DONE}_ABORTED"; then
    say "ERROR $n/$N $3 rc=$4 last-line=$last (see $1 / bench-sched.log): next slice"; fi
}
while pgrep -fx "sh ./bench-sched.v8.sh" >/dev/null 2>&1; do sleep 20; done
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
for s in usT-1 usT-2 usT-3 usT-4; do retest $s; done
for k in 1 2 3 4 5 6 7 8 9; do sweep us $k/9 usB-${k}of9; done
retest carry-us
n=$((n + 1)); say "START $n/$N final comparison (US only) $(date +%F_%T)"
env BENCH_ONLY=EXTRA BENCH_EXTRA=./bench-final-us.v1.sh BENCH_NO_RESUME=1 sh ./bench-sched.v8.sh; rc=$?
check bench-final-us.v1.log WINBENCHFINALUS_DONE "final comparison" $rc
say "END $n/$N final comparison rc=$rc $(date +%F_%T)"
say "QUEUE_US_DONE $(date +%F_%T)"
