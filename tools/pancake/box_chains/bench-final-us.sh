#!/bin/sh
# bench-final-us.sh - the FINAL COMPARISON of the US-only queue (owner order 2026-11-20 09:19: "stop doing the Strata-rdna3 tests, compare only to stock
# Strata"; TEST_LEDGER.md T-1005): stock upstream Strata v0.2.16 (plain-upstream: /mnt/pancake/benchus/build-up/strata with the plain-upstream config) against
# updated-strata (/mnt/pancake/benchus/build-us/strata) with the FINAL SHIPPED FLAGS, i.e. spec 220's `updated` arm and spec 222's `updated-exact` arm after
# apply_triage's rules on the merged US triage ($O/us-triage.v2.json: the retest slices usT-1..4, the blocked sweeps usB-1..9 and carry-us; a clean pass 2
# DROP / OPTIONAL leaves the arm, a clean KEEP is added, HOLD / INCONCLUSIVE / RETEST-GATE / bundles change nothing).  This is the headline result.
# Marker WINBENCHFINALUS_DONE (the line before it is WINBENCHFINALUS_ABORTED after an abort or a signal).  Log: <script name without .sh>.log in
# /home/dan/clone-of-d/src (bench-final-us.v1.sh -> bench-final-us.v1.log).  The file name has no "win-chain", "redo-chain", "tail-chain" or bench-(exact|...):
# no busy() pattern of another chain sees it.
# FINAL_PREFIX=prelim FINAL_REPS=3 FINAL_NO_TRIAGE=1 FINAL_GATE=1 runs the PRELIMINARY headline comparison (T-1005: arms of specs 220 / 222 as shipped now, 3 reps).
# Steps (estimate, sweep.py final-specs prints it):
#   1 harness sync (pinned by SWEEP_HARNESS_SHA), service down, idle, prerequisites
#   2 sweep.py final-specs: final-short.json, final-long.json, final-gate.json + the list of flag changes (FINALARM / FINALSHIPPED / FINALVERDICTS lines in the log)
#   3 final-short: abtest.py --tier confirm --block 3 --warmup 1 --outlier robust --reclaim-max 3 --reps 5 --no-early-stop, cases short + edit (edit = the code-edit
#     case), arms plain-upstream / updated (lossy variant) / updated-exact (exact variant)                                   ~ 35 min
#   4 final-long: the same flags, cases longdec10k / 32k / 64k / 100k (prompt and decode tok/s)                               ~ 2 h
#   5 final-gate: quality_gate.py (set-v2, --near-tie: the T-828 settings) of the final shipped arms against stock (plain-upstream scored twice)  ~ 45 min
# Rules (as bench-retest.sh): box jobs never overlap; pancake.service stays DOWN; the AA shell stays frozen (no signal); nothing is deleted; every abtest /
# quality_gate run in its own session under a watchdog (run_group) and reaped; results in /home/dan/auto-runs/benchsweep/ (final-short.jsonl, final-long.jsonl,
# final-gate/), nothing is overwritten (the chain refuses to run when they exist).  Started by run-queue-us.sh through bench-sched.v8.sh
# (BENCH_ONLY=EXTRA BENCH_EXTRA=./bench-final-us.v1.sh BENCH_NO_RESUME=1).  No RD arm, no RD file is read.
FP=${FINAL_PREFIX:-final}; FR=${FINAL_REPS:-5}   # FINAL_PREFIX=prelim FINAL_REPS=3 FINAL_NO_TRIAGE=1: the PRELIMINARY comparison (arms of specs 220 / 222 as shipped now, no triage); FINAL_GATE=0 skips the gate
S=/home/dan/clone-of-d/src; R=$S/Strata-win; L=${FINAL_LOG:-$S/$(basename "$0" .sh).log}; O=/home/dan/auto-runs/benchsweep; T=$O/work; PM=/mnt/pancake/benchsweep; HN=$PM/harness
USB=/mnt/pancake/benchus; 
DATA=/home/dan/clone-of-d/Strata-data; ROCM=/opt/rocm/core-10.0; REPO_URL=https://github.com/rocktboy1982/pancake-strata.git
T0=$(date +%s); POLL=5; MAXS=${FINAL_MAX_S:-172800}; BUDGET=${BENCH_BUDGET_S:-$MAXS}
DMAX=${SWEEP_DRIFT_MAX:-5}
export PATH=$ROCM/bin:$R/.venv/bin:$PATH LD_LIBRARY_PATH=$ROCM/lib HIP_VISIBLE_DEVICES=0
HPY=${BENCH_HARNESS_PY:-$R/.venv/bin/python}
export HPY ABTEST_PYTHON=$HPY
mkdir -p $O $T $PM
echo "START $(date +%F_%T) final comparison (US only)" >> $L
say() { echo "$*" >> $L; }
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[c]onv_trace.py|[a]uto-chain|[a]uto-post|[i]q_avx2_parity|[t]tft_bench.py|[w]in-chain[A-Za-z0-9]*\.sh|[x-z]tail-chain[A-Z]\.sh|[a]fter7|[b]ench-(updated(AA|RD)|diagUS|exact|rdgate)[.a-z0-9]*\.sh" >/dev/null 2>&1; }
aa_live() { for p in $(pgrep -f "[r]edo-chainAA\.sh" 2>/dev/null); do case "$(ps -o stat= -p $p 2>/dev/null)" in T*) ;; *) return 0;; esac; done; return 1; }
bz() { busy || aa_live; }
QI=${BENCH_PREIDLED:-0}
wait_idle() { if [ "$QI" = 1 ]; then q=0; while [ $q -lt 2 ]; do if bz; then q=0; sleep 15; else q=$((q+1)); sleep 15; fi; done; else q=0; while [ $q -lt 3 ]; do if bz; then q=0; sleep 60; else q=$((q+1)); sleep 60; fi; done; QI=1; fi; }
elapsed() { echo $(( $(date +%s) - T0 )); }
left() { echo $(( BUDGET - $(elapsed) )); }
left_ok() { [ "$(left)" -ge "$1" ]; }
clients() { ss -Htn state established '( sport = :11437 )' 2>/dev/null | wc -l; }
freekb() { df -Pk "$1" | awk 'NR==2{print $4}'; }
finish() {
  say "service stays down (owner order 2026-11-19: only the benchmarks run; this chain never starts pancake.service) $(date +%T)"
  say "/ free $(freekb /) KB, /mnt/pancake free $(freekb /mnt/pancake) KB, box time $(elapsed) s"
  echo WINBENCHFINALUS_DONE >> $L
}
abort() { say "ERROR $*"; say "WINBENCHFINALUS_ABORTED: $*"; finish; exit 1; }
errline() {
  pat='error|fatal|abort|died|failed|needs|cannot|exception|terminate|no such|not found|unrecognized|unknown'
  a=$(grep -i -m1 -E "$pat" "$1" 2>/dev/null | cut -c1-240); [ -n "$a" ] || a=$(grep -m1 . "$1" 2>/dev/null | cut -c1-240)
  e=; for f in "$2"/*.engine.log "$2"/*.out; do [ -f "$f" ] || continue; e=$(grep -i -E "$pat" "$f" 2>/dev/null | tail -n 1 | cut -c1-240); [ -n "$e" ] && break; done
  echo "first error line of the stdout: [$a]; engine log: [$e]"
}

# ------------------------------------------------------------------ process groups: a watchdog per step, no orphan afterwards (the same functions as bench-exact.sh)
# ptree SID: SID, every descendant (by parent pid) and every process of session SID.  stamps PIDS: "pid start-time" lines of the live ones (pid reuse safe).
ptree() { ps -eo pid=,ppid=,sid= 2>/dev/null | awk -v r="$1" '{ c[$2] = c[$2] " " $1; if ($3 == r) s[$1] = 1 } END { q = r; while (q != "") { n = split(q, a, " "); q = ""; for (i = 1; i <= n; i++) if (!(a[i] in seen)) { seen[a[i]] = 1; q = q c[a[i]] } } for (p in s) seen[p] = 1; for (p in seen) print p }'; }
stamps() { [ -n "$1" ] && ps -o pid=,lstart= -p "$(echo $1 | tr ' ' ',')" 2>/dev/null | sed 's/^ *//; s/  */ /g'; }
alive() { kill -0 "$1" 2>/dev/null || return 1; case "$(ps -o stat= -p "$1" 2>/dev/null)" in Z*) return 1;; esac; return 0; }
sendsig() { ${REAPCMD:-kill} "$@" 2>/dev/null; }
CUR_SID=; CUR_SEEN=
run_group() {  # run_group NAME LIMIT_S LOG CMD...: CMD in its own session; at min(LIMIT, budget left) TERM to its tree, KILL 100 s later; then REAP
  nm=$1; lim=$2; log=$3; shift 3
  lb=$(left); [ "$lb" -lt 60 ] && lb=60; [ "$lb" -lt "$lim" ] && lim=$lb
  seen=$T/$nm.pids; : > "$seen"
  setsid "$@" > "$log" 2>&1 &   # a background job of a non-interactive sh is no group leader: setsid does not fork, $! is the session id
  sid=$!; CUR_SID=$sid; CUR_SEEN=$seen; t=0; killed=0
  while alive $sid; do
    stamps "$(ptree $sid)" >> "$seen"
    if [ $killed -eq 0 ] && [ $t -ge $lim ]; then
      tp=$(ptree $sid); say "TIMEOUT $nm after $t s (limit $lim s, budget left $(left) s): TERM to its process tree: $(echo $tp)"
      for p in $tp; do sendsig -TERM $p; done; killed=1; tk=$t
    elif [ $killed -eq 1 ] && [ $t -ge $((tk + 100)) ]; then
      tp=$(ptree $sid); say "TIMEOUT $nm: still running 100 s after TERM: KILL: $(echo $tp)"
      for p in $tp; do sendsig -KILL $p; done; killed=2
    elif [ $killed -eq 2 ] && [ $t -ge $((tk + 130)) ]; then say "ERROR $nm: the session leader $sid survives SIGKILL"; break; fi
    sleep $POLL; t=$((t + POLL))
  done
  wait $sid 2>/dev/null; rc=$?; [ $killed -gt 0 ] && rc=124
  reap $nm "$seen"; CUR_SID=; CUR_SEEN=
  return $rc
}
reap() {  # reap NAME SEEN_FILE: every process the step had that still runs (same pid AND start time) gets TERM, then KILL after 30 s; each is logged
  pids=$(awk '{print $1}' "$2" 2>/dev/null | sort -u | tr '\n' ' ')
  [ -n "$pids" ] || return 0
  left_over=$(stamps "$pids" | while read -r p st; do grep -qxF "$p $st" "$2" && echo "$p"; done | tr '\n' ' ')
  [ -n "$(echo $left_over)" ] || return 0
  say "REAP $1: still running after the step ended, TERM: $(for p in $left_over; do printf '%s [%s] ' $p "$(ps -o args= -p $p 2>/dev/null | cut -c1-100)"; done)"
  for p in $left_over; do sendsig -TERM $p; done
  still=$left_over; n=0; while [ $n -lt 6 ]; do sleep 5; n=$((n + 1)); still=$(stamps "$left_over" | while read -r p st; do grep -qxF "$p $st" "$2" && echo "$p"; done | tr '\n' ' '); [ -n "$(echo $still)" ] || break; done
  if [ -n "$(echo $still)" ]; then say "REAP $1: KILL $still"; for p in $still; do sendsig -KILL $p; done; sleep 3
    gone=$(stamps "$still" | while read -r p st; do grep -qxF "$p $st" "$2" && echo "$p"; done | tr '\n' ' ')
    [ -z "$(echo $gone)" ] || say "ERROR REAP $1: processes survive SIGKILL: $gone"
  fi
  say "REAP $1 done"
}
on_signal() {
  say "signal received $(date +%T): the running step's tree gets TERM, then the chain ends"
  if [ -n "$CUR_SID" ]; then for p in $(ptree $CUR_SID); do sendsig -TERM $p; done; sleep 5; reap signal "$CUR_SEEN"; fi
  say "WINBENCHFINALUS_ABORTED: signal"; finish; exit 1
}
trap on_signal HUP INT TERM


# ------------------------------------------------------------------ steps
ab() {  # ab NAME LIMIT SPEC flags...: one abtest run from the synced harness, in its own session
  nm=$1; lim=$2; sp=$3; shift 3
  [ "$(freekb /)" -ge 307200 ] || { say "ERROR $nm: / has < 300 MB free: not run"; return 1; }
  say "START $nm $(date +%F_%T) (limit $lim s, budget left $(left) s)"
  run_group $nm $lim $O/$nm.stdout sh -c 'cd "$1" && shift && exec "$HPY" tools/pancake/abtest.py "$@"' ab $HN $sp "$@" --out $O/$nm.jsonl; rc=$?
  say "END $nm exit=$rc $(date +%T)"
  [ $rc -eq 0 ] || say "ERROR $nm exit=$rc: $(errline $O/$nm.stdout $O/$nm) (the rows that were measured stand)"
  grep -E "ORDER EFFECT|DRIFT" $O/$nm.stdout | cut -c1-220 >> $L
  return $rc
}
sw() { "$HPY" $HN/tools/pancake/box_chains/sweep.py "$@"; }

# ------------------------------------------------------------------ 0. service down, idle, prerequisites
for f in $O/$FP-short.jsonl $O/$FP-long.jsonl $O/$FP-short.stdout $O/$FP-long.stdout $O/$FP-gate $O/$FP-gate.stdout; do
  [ -e "$f" ] && abort "the comparison $FP was run before ($f exists): nothing is overwritten"
done
[ -n "$FINAL_NO_TRIAGE" ] || [ -f $O/us-triage.v2.json ] || abort "no $O/us-triage.v2.json: the US retest / sweep slices have not written a triage"
if [ "${BENCH_SCHED:-0}" != 1 ]; then say "standalone start $(date +%T)"; else say "started by the scheduler: the AA shell is frozen, the box was idle $(date +%T)"; fi
if [ "${BENCH_SCHED:-0}" != 1 ]; then say "standalone start $(date +%T)"; else say "started by the scheduler: the AA shell is frozen, the box was idle $(date +%T)"; fi
if systemctl --user is-active --quiet pancake.service; then
  q=0; while [ $q -lt 3 ]; do if [ "$(clients)" -gt 0 ]; then q=0; say "WAIT service has clients $(date +%T)"; else q=$((q+1)); fi; sleep 60; done
  systemctl --user stop pancake.service; say "service stopped (idle) $(date +%T)"
else say "service already stopped"; fi
wait_idle
[ "$(freekb /mnt/pancake)" -ge 2097152 ] || abort "START: /mnt/pancake has $(freekb /mnt/pancake) KB free (< 2 GB); nothing was deleted"
[ "$(freekb /)" -ge 307200 ] || abort "START: / has $(freekb /) KB free (< 300 MB)"
miss=; for f in $DATA/packs/iq2_xs/tokenizer $DATA/mtp/rt $HPY; do [ -e "$f" ] || miss="$miss $f"; done
[ -z "$miss" ] || abort "PREREQ missing:$miss"
"$HPY" -c 'import sys' >/dev/null 2>&1 || abort "PREREQ the harness python $HPY does not run"
say "PREREQ ok"


# ------------------------------------------------------------------ 1. harness (pinned by SWEEP_HARNESS_SHA), mtimes reset
SHA=${SWEEP_HARNESS_SHA:-$(timeout 60 git ls-remote $REPO_URL refs/heads/port/rdna3-decode 2>/dev/null | cut -f1)}
HARN=$PM/harness.tgz; rm -f $HARN.part
if [ -n "$SHA" ] && timeout 1200 curl -sfL -o $HARN.part https://codeload.github.com/rocktboy1982/pancake-strata/tar.gz/$SHA && tar tzf $HARN.part >/dev/null 2>&1; then
  mv $HARN.part $HARN; say "SYNC harness: origin port/rdna3-decode $SHA${SWEEP_HARNESS_SHA:+ (pinned by SWEEP_HARNESS_SHA)} -> $HN"
else abort "SYNC harness download failed"; fi
mkdir -p $HN && tar xzmf $HARN --no-same-owner --strip-components=1 -C $HN || abort "SYNC harness tar failed"
for need in tools/pancake/abtest.py tools/pancake/quality_gate.py bench/quality/set-v2.json tools/pancake/box_chains/sweep.py \
            tools/pancake/box_chains/sweep_candidates.json tools/pancake/apply_triage.py bench/specs/220-updated-vs-upstream-short.json \
            bench/specs/222-updated-exact-vs-upstream-short.json configs/pancake/flashnext-iq2_xs-plain-upstream.json; do
  [ -f "$HN/$need" ] || abort "SYNC: $need is missing in the synced harness"
done
grep -q ABTEST_PYTHON $HN/tools/pancake/abtest.py || abort "PREREQ harness: $HN/tools/pancake/abtest.py does not read ABTEST_PYTHON"
e=$(cd $HN && PYTHONDONTWRITEBYTECODE=1 "$HPY" -B -c 'import serve.server' 2>&1) || abort "PREREQ harness: $HPY cannot import serve.server from $HN: $(echo "$e" | tail -n 1 | cut -c1-240)"

# ------------------------------------------------------------------ 2. the final specs from the shipped flags (apply_triage's rules on the merged US triage)
sw final-specs --triage ${FINAL_NO_TRIAGE:-$O/us-triage.v2.json} --us-arm-spec $HN/bench/specs/220-updated-vs-upstream-short.json --us-exact-spec $HN/bench/specs/222-updated-exact-vs-upstream-short.json    --shipped $USB/src-us/terminal/data/tuned-flags.json --out-dir $T --reps $FR --prefix $FP ${FINAL_NO_TRIAGE:+--no-triage} > $T/final-specs.txt 2>&1 || abort "SPEC: sweep.py final-specs failed: $(tail -n 1 $T/final-specs.txt)"
for f in $FP-short $FP-long $FP-gate; do [ -s $T/$f.json ] || abort "SPEC: $T/$f.json was not written"; done
grep -E '^(FINAL|ESTIMATE)' $T/final-specs.txt | cut -c1-700 >> $L
cp $T/$FP-short.json $T/$FP-long.json $T/$FP-gate.json $O/ 2>/dev/null; say "SPECS copied to $O/$FP-*.json (the exact arms of this run)"

# ------------------------------------------------------------------ 3-5. short, long, gate
CONF="--tier confirm --block 3 --warmup 1 --outlier robust --reclaim-max 3 --reps $FR --no-early-stop"
rcs=0; rcl=0; rcg=0
summ() { grep -E "vs .*(better|worse|no difference)|verdict|PROMOTE|DRIFT|ORDER EFFECT" $O/$1.stdout | cut -c1-240 >> $L; grep -A8 "generated text vs base" $O/$1.stdout | cut -c1-200 >> $L; }
wait_idle; ab $FP-short 14400 $O/$FP-short.json $CONF; rcs=$?; summ $FP-short
wait_idle; ab $FP-long 36000 $O/$FP-long.json $CONF; rcl=$?; summ $FP-long
rcg=-
if [ "${FINAL_GATE:-1}" = 1 ]; then
wait_idle
nm=$FP-gate
say "START $nm $(date +%F_%T) (quality_gate.py, set bench/quality/set-v2.json, --near-tie, plain-upstream scored twice, limit 7200 s)"
run_group $nm 7200 $O/$nm.stdout sh -c 'cd "$1" && shift && exec "$HPY" tools/pancake/quality_gate.py "$@"' qg $HN $O/$FP-gate.json --out $O/$nm --set bench/quality/set-v2.json --near-tie; rcg=$?
say "END $nm exit=$rcg $(date +%T)"
[ $rcg -eq 0 ] || say "ERROR $nm exit=$rcg: $(errline $O/$nm.stdout $O/$nm)"
grep -E "PASS|REJECT|INCONCLUSIVE|FAILED|not scored|near-tie|mismatch|events" $O/$nm.stdout | cut -c1-220 >> $L
fi
say "SUMMARY final comparison (US only): final-short exit=$rcs, final-long exit=$rcl, final-gate exit=$rcg; box time $(elapsed) s; results in $O (final-*.jsonl, final-*.stdout, final-gate/)"
finish
