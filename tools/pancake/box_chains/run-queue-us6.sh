#!/bin/sh
# US-ONLY queue, SIXTH version (owner correction 2026-11-20 ~11:10, supersedes run-queue-us5.sh): triage of run 1 by the PER-CASE effects (short, edit, long64k prompt, long64k decode):
#   HIGH = ANY case >= +2 % (even when another case is negative: a big gain on one case can end OPTIONAL) -> rescreen; LOW = best case > 0 % and < 2 % -> NOT rescreened this round, BACKLOG
#   (us-backlog-low.json with the run-1 numbers, next week with the 31 deferred switches); NEGATIVE = every case <= 0 %: a plus arm (off-by-default flag) is NO-GAIN, dropped; a minus arm
#   (preset flag: having it ON hurts) is HURTS and is rescreened once (no shipped removal from one run).  The RESCREEN (run 2) is only for HIGH + HURTS flags; the two-run rule decides.
#   NO low-gain rescreen stage any more.  Order: screen 1 -> rescreen(HIGH + HURTS) -> screen 2 -> rescreen -> ... screen 5 -> rescreen -> gates -> FINAL comparison.
# Earlier text (fifth version):
# US-ONLY queue, FIFTH version (owner protocol clarified 2026-11-20 ~10:55; TEST_LEDGER.md T-1005; supersedes run-queue-us4.sh, whose sequencer shell is stopped):
#   (1) ALL flags get ONE run (stage 1 screen: slice 1 = usS1-1of5, running since 10:31, started by run-queue-us3.sh);
#   (2) a RESCREEN = a SECOND, independent 1-run test of the flags that showed a benefit (any case above 0 %), highest gain first: the HIGH flags (best case >= 2.0 %) of the slice are
#       rescreened right after the slice, the LOW ones (0 < gain < 2.0 %) after the final comparison; blocked like the screen (blocks of 8, base before + base-late after, the slice's
#       first start is the only warm-up, last-request drift > 5 % => one re-run, then INCONCLUSIVE), fresh block neighbours and order (another seed: its own tag usS2r-K);
#   (3) NO separate 3-rep confirm stage (owner 2026-11-20 ~11:05): the verdict comes from run 1 + run 2 together (sweep.py two_run): a flag STANDS (KEEP) when the benefit repeats in the
#       same direction (both runs > 0 on the case(s) that carried it; effect = mean of the two runs, spread = their difference); OPTIONAL when a case >= +10 % and a case <= -3 % both
#       repeat in both runs; NO-GAIN (DROP) when run 2 does not repeat the benefit or both are <= 0; MIXED (nothing ships) when the gain repeats but a case lost >= 3 % in both runs;
#       INCONCLUSIVE only when a block drifted > 5 % after its one re-run (such a flag is rescreened once more); quality gates only for --lookup-max-t and lossy flags with a two-run
#       mean >= 3 % (RETEST-GATE); the FINAL 5-rep comparison (stock vs updated with the shipped set) validates the whole set;
#   (4) NO stock-Strata arm anywhere in the flag work (the base is the updated arm); the PRELIMINARY stock comparison is REMOVED (stock vs updated exists since 2026-11-19, T-991..T-995);
#       stock appears only in the FINAL comparison (5 reps, exact + lossy, with the gate).
# Order: screen 1 (running) -> rescreen of its HIGH (>= 2.0 %) flags + HURTS -> screen 2 -> rescreen -> ... -> screen 5 -> rescreen -> quality gates -> FINAL comparison (apply_triage on the
# two-run verdicts) -> the LOW-gain flags (0 < gain < 2.0 %: rescreen only, the same two-run rule).  A step with nothing to do is skipped cleanly.  ETA assumption: ~6 HIGH flags per
# slice = ONE rescreen block (~0.5 h) per slice.  Kill only the sequencer shell run-queue-us4.sh (it has no child but the idle wait) to start this one.
# Same cases (short, edit, longdec64k), blocks, drift rule as run-queue-us3.sh; scripts bench-sweep.v5.sh (SWEEP_PRIOR, the rescreen; slice 1 still runs under bench-sweep.v4.sh, which is waited for
# by name), bench-retest.v4.sh (gates only), bench-final-us.v2.sh; a NEW harness pin (sweep.py two_run / select --min-gain).
# BACKLOG / PRUNED: see run-queue-us3.sh (31 deferred switches, us-backlog.json, run-queue-us-backlog.sh not started; PA_V2T, MMVQ_IL_120, NO_NORM_ROPE never tested).
cd /home/dan/clone-of-d/src || exit 1
L=/home/dan/clone-of-d/src/run-queue-us6.log
say() { echo "$*" >> $L; }
: ${SWEEP_HARNESS_SHA:?SWEEP_HARNESS_SHA must be set (the pinned harness commit)}
O=/home/dan/auto-runs/benchsweep; HN=/mnt/pancake/benchsweep/harness; HPY=/home/dan/clone-of-d/src/Strata-win/.venv/bin/python
export SWEEP_HARNESS_SHA RETEST_NO_RD=1 SWEEP_MAX_S=${SWEEP_MAX_S:-9000} RETEST_MAX_S=${RETEST_MAX_S:-9000} BENCH_PREIDLED=1
export SWEEP_CASES=short,edit,longdec64k SWEEP_BLOCK=8 SWEEP_WARM=first RETEST_LAYOUT=compact RETEST_WARM=first
export SWEEP_EXTRA="us:--lookup-max-t us:STRATA_GR_RDNA us:STRATA_RDNA_MMVQ_WAVE us:STRATA_RDNA_Q6K us:STRATA_HIPBLASLT_TUNING"
CARRY_GLOB="$O/us-carry-usS1-*.json $O/us-carry-usS2r-*.json"
n=0
say "QUEUE_US6 START $(date +%F_%T) harness $SWEEP_HARNESS_SHA; waiting for screen slice 1 (usS1-1of5, run-queue-us3.sh); then per slice: rescreen of the HIGH flags -> 3-rep confirms of those that stood -> next screen slice; final comparison; low-gain flags"
check() {  # check LOG MARKER NAME rc
  last=$(tail -n 1 $1 2>/dev/null)
  if [ $4 -ne 0 ] || [ "$last" != "$2" ] || tail -n 4 $1 | grep -q "${2%_DONE}_ABORTED"; then
    say "ERROR $n $3 rc=$4 last-line=$last (see $1 / bench-sched.log): next slice"; fi
}
screen() {  # screen K/N TAG [VAR=VALUE ...]: a blocked pass-1 screen, no pass 2
  k=$1; tag=$2; shift 2
  n=$((n + 1)); say "START $n screen us $tag ($k) $(date +%F_%T)"
  env "$@" SWEEP_BRANCH=us SWEEP_SLICE=$k SWEEP_SPLIT=even SWEEP_TAG=$tag SWEEP_SCREEN_ONLY=1 BENCH_ONLY=EXTRA BENCH_EXTRA=./bench-sweep.v5.sh BENCH_NO_RESUME=1 sh ./bench-sched.v8.sh; rc=$?
  check bench-sweep.log WINBENCHSWEEP_DONE "screen $tag" $rc
  say "END $n screen $tag rc=$rc $(date +%F_%T)"
}
retest() {  # retest SLICE [VAR=VALUE ...]
  sl=$1; shift
  n=$((n + 1)); say "START $n retest $sl $(date +%F_%T)"
  env "$@" RETEST_SLICE=$sl BENCH_ONLY=EXTRA BENCH_EXTRA=./bench-retest.v4.sh BENCH_NO_RESUME=1 sh ./bench-sched.v8.sh; rc=$?
  check bench-retest.v4.log WINBENCHRETEST_DONE "retest $sl" $rc
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

# ---- the interleaved steps
PRIOR=$O/us-triage.v2.json
rescreen() {  # rescreen TAG "select args": the second 1-run test (the RESCREEN) of the selected flags, blocks of 8, <= 25 flags per slice; the two-run rule gives the verdicts
  tag=$1; shift
  r=$(sw select --out $O/work/us-$tag.meta.json "$@" 2>> $L | tee -a $L | sed -n 's/^SELECT \([0-9][0-9]*\) .*//p')
  r=${r:-0}
  if [ "$r" -gt 0 ]; then
    nr=$(ceil $r 25)
    for q in $(seq 1 $nr); do screen $q/$nr $tag-${q}of$nr SWEEP_ONLY_META=$O/work/us-$tag.meta.json SWEEP_PRIOR=$PRIOR; done
  else say "RESCREEN $tag: no flag selected"; fi
}
rerun1() {  # rerun1 K: run 1 once more for the flags of screen slice K that stayed INCONCLUSIVE / UNTESTED (a drifted block); no prior: it is a run 1 (tag usS1-KofNx)
  r=$(sw select --triage $PRIOR --verdict INCONCLUSIVE,UNTESTED --slice-prefix usS1-${1}of5 --out $O/work/us-rerun1-$1.meta.json 2>> $L | tee -a $L | sed -n 's/^SELECT \([0-9][0-9]*\) .*//p')
  r=${r:-0}
  if [ "$r" -gt 0 ]; then screen 1/1 usS1-${1}of5x SWEEP_ONLY_META=$O/work/us-rerun1-$1.meta.json; else say "RERUN1 slice $1: nothing INCONCLUSIVE"; fi
}
cycle() {  # cycle K: screen slice K has just ended (its triage is merged): re-run its drifted blocks' flags, rescreen its HIGH flags, then rescreen once more what drifted in run 2
  rerun1 $1
  rescreen usS2r-$1 --triage $PRIOR --verdict PROMISING,HURTS --slice-prefix usS1-${1}of5
  rescreen usS2rr-$1 --triage $PRIOR --verdict INCONCLUSIVE --slice-prefix usS2r-${1}-
}
gates() {  # quality gates, 3 per slice, ~45 min each: --lookup-max-t always, and every flag with a two-run mean >= 3 % that is lossy (verdict RETEST-GATE)
  g=$(sw carry-count --gate-from $(ls $O/us-triage-usS2r*.v2.json 2>/dev/null) --gate-prefix usS2r --gate-base-spec $HN/bench/specs/220-updated-vs-upstream-short.json 2>> $L | sed -n 's/^GATECOUNT //p')
  g=${g:-1}
  say "GATES: $g quality gate(s)"
  j=0
  while [ $((j * 3)) -lt $g ]; do
    j=$((j + 1)); lg=$((g - (j - 1) * 3)); [ $lg -gt 3 ] && lg=3
    retest gates-us-$j RETEST_GATE_SLICES=usS2r RETEST_CARRY_OFFSET=$(( (j - 1) * 3 )) RETEST_CARRY_MAX=$lg
  done
}

# ---- slice 1 is already running (started by run-queue-us3.sh): wait for its scheduler + chain, then cycle 1; slices 2..5 follow, each with its cycle
while pgrep -fx "sh ./bench-sched.v8.sh" >/dev/null 2>&1 || pgrep -fx "sh ./bench-sweep.v4.sh" >/dev/null 2>&1; do sleep 20; done
check bench-sweep.log WINBENCHSWEEP_DONE "screen usS1-1of5" 0
say "END screen usS1-1of5 (slice 1, started by run-queue-us3.sh) $(date +%F_%T): $(tail -n 1 bench-sweep.log)"
cycle 1
for k in 2 3 4 5; do screen $k/5 usS1-${k}of5; cycle $k; done
gates
sw select --triage $PRIOR --verdict PROMISING-LOW --slice-prefix usS1- --out $O/us-backlog-low.json >> $L 2>&1

# ---- the FINAL comparison on the merged US triage (apply_triage's rules on the two-run verdicts): the only place where stock Strata appears
final final FINAL_PREFIX=final FINAL_REPS=5

# ---- the LOW-gain flags (best case 0 < gain < 2.0 % in run 1) go to the BACKLOG with their run-1 numbers (next week, together with the 31 deferred switches)
sw select --triage $PRIOR --verdict PROMISING-LOW --slice-prefix usS1- --out $O/us-backlog-low.json >> $L 2>&1
say "QUEUE_US6_DONE $(date +%F_%T)"
