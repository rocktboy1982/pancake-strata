#!/bin/sh
# US-ONLY queue, THIRD version: the owner's restated protocol + "push through faster" (2026-11-20; TEST_LEDGER.md T-1005): ONE screen of all flags (1 rep), then a 3-rep
# confirm of the flags that showed a benefit, highest gain first.  Replaces run-queue-us2.sh (stopped at 09:46 with its usT-1 slice: nothing of it ran).
# Every slice runs under bench-sched.v8.sh (BENCH_ONLY=EXTRA, BENCH_NO_RESUME=1: the AA shell stays frozen), pancake.service stays down, a failed slice is logged
# and the next one runs, slices run back to back (BENCH_PREIDLED=1: the 30 s quick idle between steps); harness pinned by SWEEP_HARNESS_SHA.  Scripts on the box:
# bench-sweep.v4.sh, bench-retest.v3.sh, bench-final-us.v2.sh.
#   STAGE 1  SCREEN (5 slices, bench-sweep.v4.sh, SWEEP_SCREEN_ONLY=1): every US candidate flag of this round ONCE, 124 arms = the 119 untested switches left in sweep_candidates.json
#            "untested" (34 were deferred to the BACKLOG, 3 are pruned, see below) + the 5 named flags the registry lists as measured (--lookup-max-t 12, GR_RDNA, RDNA_MMVQ_WAVE, RDNA_Q6K,
#            HIPBLASLT_TUNING); one arm each (minus-X for a preset flag, plus-X for an off-by-default one: a switch whose preset value equals the engine default still gets
#            the arm that flips it), 1 rep, blocks of <= 8 arms, ONE warm-up start per slice (rows of rdC-1 k01r / k02: a block that follows another one starts steady, only the
#            first start after an idle gap is high), base before and base-late after every block, last-request drift > 5 % => one re-run, then INCONCLUSIVE; cases
#            short + edit + longdec64k (prompt and decode tok/s of a cold 64K prompt).  Per flag: PROMISING (any case > 0 %, clean block) / NO-GAIN-P1 (<= 0 %) /
#            INCONCLUSIVE.  ~8.9 h (169 starts of 190 s + one re-run per drifted block), 5 slices of <= 25 arms, each <= 2.3 h with one re-run.
#   PRELIMINARY comparison (bench-final-us.v2.sh, prelim, 3 reps, arms of specs 220 / 222 as shipped, short + edit + long 64K + 100K, gate included, ~1.9 h).
#   STAGE 2  (a) the INCONCLUSIVE / UNTESTED flags are re-screened once (tag usS2r-*); (b) the PROMISING flags are RANKED by their best screen case (sweep.py rank ->
#            us-promising-ranked.json): HIGH = best case >= 2.0 % (the owner's line between high and low gain) are confirmed this round, highest gain first, each in its own
#            COMPACT paired 3-rep block (base, flag, flag, flag, base-late = 5 starts ~16 min, the median of the 3 paired gains; warm-up once per slice), 6 per slice
#            (carry-us-usS2-N), verdict KEEP / OPTIONAL / DROP / INCONCLUSIVE; (c) quality gates (gates-us-N, 3 per slice, ~45 min each): --lookup-max-t always and every flag
#            whose clean confirm is lossy >= 3 %.
#   STAGE 3  FINAL comparison (bench-final-us.v2.sh, 5 reps, confirm standard --block 3 --warmup 1 --outlier robust --reclaim-max 3, exact + lossy, short / edit +
#            long 10K / 32K / 64K / 100K, gate, ~3.3 h): the arms of specs 220 / 222 after apply_triage's rules on the merged US triage (sweep.py final-specs).
#   STAGE 2b (after the final comparison, the long tail): the PROMISING-LOW flags (0 < best case < 2.0 %): skipped for now, NOT dropped and NOT shipped; the same compact
#            paired 3-rep blocks in the same descending order (carry-us-usM-N, 6 per slice).  Flags with a best case <= 0 % in a clean block (NO-GAIN-P1) are the only ones dropped.
# Order: stage 1 -> preliminary comparison -> stage 2 (re-screen, high-gain confirms, gates) -> final comparison -> stage 2b low-gain confirms.  Log run-queue-us3.log, last
# line QUEUE_US3_DONE.
# BACKLOG (owner 2026-11-20: deferred to the week of 2026-11-23, NOT pruned): 31 switches in sweep_candidates.json "backlog.us" (KVA research x6, Q2_0 / Q3_K / Q5_0 / K-quant / IQ3_S /
# TQ / PLE other-quant-or-model paths, Q4_0-KV prompt attention, conversation checkpoint x5, admission, remote contexts, sampling, diagnostic) with the reason each was deferred; the
# ready-to-run queue file is us-backlog.json, the launcher run-queue-us-backlog.sh (same screen design); NOT started.  PRUNED = never tested: PA_V2T (read only on a Turing card:
# `turing && !v2t`), MMVQ_IL_120 (the table of big gfx12.0 cards), NO_NORM_ROPE (alias of NORM_ROPE).  A switch whose arm would equal the base because the preset already sets it is NOT
# pruned: it is tested as the minus arm.
cd /home/dan/clone-of-d/src || exit 1
L=/home/dan/clone-of-d/src/run-queue-us3.log
say() { echo "$*" >> $L; }
: ${SWEEP_HARNESS_SHA:?SWEEP_HARNESS_SHA must be set (the pinned harness commit)}
O=/home/dan/auto-runs/benchsweep; HN=/mnt/pancake/benchsweep/harness; HPY=/home/dan/clone-of-d/src/Strata-win/.venv/bin/python
export SWEEP_HARNESS_SHA RETEST_NO_RD=1 SWEEP_MAX_S=${SWEEP_MAX_S:-9000} RETEST_MAX_S=${RETEST_MAX_S:-9000} BENCH_PREIDLED=1
export SWEEP_CASES=short,edit,longdec64k SWEEP_BLOCK=8 SWEEP_WARM=first RETEST_LAYOUT=compact RETEST_WARM=first
export SWEEP_EXTRA="us:--lookup-max-t us:STRATA_GR_RDNA us:STRATA_RDNA_MMVQ_WAVE us:STRATA_RDNA_Q6K us:STRATA_HIPBLASLT_TUNING"
CARRY_GLOB="$O/us-carry-usS1-*.json $O/us-carry-usS2r-*.json"
n=0
say "QUEUE_US3 START $(date +%F_%T) harness $SWEEP_HARNESS_SHA; stage 1 = 5 screen slices (124 arms, cases short/edit/long64k, ~8.9 h), preliminary comparison (~1.9 h), stage 2 (re-screen, HIGH-gain confirms >= 2.0 %, gates), final comparison (~3.3 h), stage 2b (low-gain confirms)"
check() {  # check LOG MARKER NAME rc
  last=$(tail -n 1 $1 2>/dev/null)
  if [ $4 -ne 0 ] || [ "$last" != "$2" ] || tail -n 4 $1 | grep -q "${2%_DONE}_ABORTED"; then
    say "ERROR $n $3 rc=$4 last-line=$last (see $1 / bench-sched.log): next slice"; fi
}
screen() {  # screen K/N TAG [VAR=VALUE ...]: a blocked pass-1 screen, no pass 2
  k=$1; tag=$2; shift 2
  n=$((n + 1)); say "START $n screen us $tag ($k) $(date +%F_%T)"
  env "$@" SWEEP_BRANCH=us SWEEP_SLICE=$k SWEEP_SPLIT=even SWEEP_TAG=$tag SWEEP_SCREEN_ONLY=1 BENCH_ONLY=EXTRA BENCH_EXTRA=./bench-sweep.v4.sh BENCH_NO_RESUME=1 sh ./bench-sched.v8.sh; rc=$?
  check bench-sweep.log WINBENCHSWEEP_DONE "screen $tag" $rc
  say "END $n screen $tag rc=$rc $(date +%F_%T)"
}
retest() {  # retest SLICE [VAR=VALUE ...]
  sl=$1; shift
  n=$((n + 1)); say "START $n retest $sl $(date +%F_%T)"
  env "$@" RETEST_SLICE=$sl BENCH_ONLY=EXTRA BENCH_EXTRA=./bench-retest.v3.sh BENCH_NO_RESUME=1 sh ./bench-sched.v8.sh; rc=$?
  check bench-retest.v3.log WINBENCHRETEST_DONE "retest $sl" $rc
  say "END $n retest $sl rc=$rc $(date +%F_%T)"
}
final() {  # final NAME [VAR=VALUE ...]
  nm=$1; shift
  n=$((n + 1)); say "START $n $nm comparison (US only) $(date +%F_%T)"
  env "$@" BENCH_ONLY=EXTRA BENCH_EXTRA=./bench-final-us.v2.sh BENCH_NO_RESUME=1 sh ./bench-sched.v8.sh; rc=$?
  check bench-final-us.v2.log WINBENCHFINALUS_DONE "$nm comparison" $rc
  say "END $n $nm comparison rc=$rc $(date +%F_%T)"
}
sw() { "$HPY" $HN/tools/pancake/box_chains/sweep.py "$@"; }
ceil() { echo $(( ($1 + $2 - 1) / $2 )); }
confirm() {  # confirm TAGPREFIX FIRST COUNT: the ranked flags [FIRST, FIRST+COUNT) in slices of 6 (the tail spills into the next slice)
  pre=$1; first=$2; cnt=$3; i=0
  while [ $((i * 6)) -lt $cnt ]; do
    i=$((i + 1)); left=$((cnt - (i - 1) * 6)); [ $left -gt 6 ] && left=6
    retest carry-us-$pre-$i RETEST_CARRY_GLOB="$CARRY_GLOB" RETEST_CARRY_OFFSET=$((first + (i - 1) * 6)) RETEST_CARRY_MAX=$left
  done
}

# ---- STAGE 1: screen every flag once
for k in 1 2 3 4 5; do screen $k/5 usS1-${k}of5; done

# ---- PRELIMINARY comparison: stock v0.2.16 against the arms of specs 220 / 222 as shipped now (3 reps, short + edit + long 64K / 100K, gate included)
final preliminary FINAL_PREFIX=prelim FINAL_REPS=3 FINAL_NO_TRIAGE=1 FINAL_GATE=1 FINAL_LONG=longdec64k,longdec100k

# ---- STAGE 2a: re-screen once what stayed INCONCLUSIVE / UNTESTED in stage 1
r=$(sw select --triage $O/us-triage.v2.json --slice-prefix usS1 --out $O/work/us-rescreen.meta.json 2>> $L | tee -a $L | sed -n 's/^SELECT \([0-9][0-9]*\) .*/\1/p')
r=${r:-0}
if [ "$r" -gt 0 ]; then
  nr=$(ceil $r 25)
  for k in $(seq 1 $nr); do screen $k/$nr usS2r-${k}of$nr SWEEP_ONLY_META=$O/work/us-rescreen.meta.json; done
else say "STAGE 2a: nothing to re-screen"; fi

# ---- STAGE 2b-rank: the PROMISING flags by gain: HIGH (best case >= 2.0 %) this round, the rest PROMISING-LOW (stage 2b after the final comparison)
rk=$(sw rank --carry $(ls $CARRY_GLOB 2>/dev/null) --high 2.0 --out $O/us-promising-ranked.json 2>> $L | tee -a $L | sed -n 's/^RANKCUT //p')
cut=${rk% *}; tot=${rk#* }; cut=${cut:-0}; tot=${tot:-0}
say "STAGE 2 RANK: $cut HIGH-gain flag(s) of $tot promising (ranked file $O/us-promising-ranked.json)"
[ "$cut" -gt 0 ] && confirm usS2 0 $cut

# ---- STAGE 2c: quality gates (--lookup-max-t always; the flags whose clean confirm is lossy >= 3 %), 3 per slice
g=$(sw carry-count --carry $(ls $CARRY_GLOB 2>/dev/null) --gate-from $(ls $O/us-triage-rt-carry-us-usS2*.v2.json 2>/dev/null) --gate-prefix carry-us-usS2 \
      --gate-base-spec $HN/bench/specs/220-updated-vs-upstream-short.json 2>> $L | sed -n 's/^GATECOUNT //p')
g=${g:-1}
say "STAGE 2c: $g quality gate(s)"
for i in $(seq 1 $(ceil $g 3)); do retest gates-us-$i RETEST_CARRY_OFFSET=$(( (i - 1) * 3 )) RETEST_CARRY_MAX=3; done

# ---- STAGE 3: the FINAL comparison on the merged US triage
final final FINAL_PREFIX=final FINAL_REPS=5

# ---- STAGE 2b: the PROMISING-LOW flags, same order, after the headline
if [ "$tot" -gt "$cut" ]; then say "STAGE 2b: $((tot - cut)) PROMISING-LOW flag(s) (best case < 2.0 %), confirmed in descending order"; confirm usM $cut $((tot - cut)); fi
say "QUEUE_US3_DONE $(date +%F_%T)"
