#!/bin/sh
# bench-retest.sh - RETEST slices: every flag confirmed ONE BY ONE in its own paired block (owner orders 2026-11-20: "confirm the flags one by one";
# the two-pass rule restated; flags judged individually, a bundle never changes a flag; OPTIONAL for a big gain on one case and a loss on another).
# Marker WINBENCHRETEST_DONE (the line before it is WINBENCHRETEST_ABORTED after an abort or a signal).  Log: <script name without .sh>.log in
# /home/dan/clone-of-d/src (bench-retest.v1.sh -> bench-retest.v1.log; RETEST_LOG overrides).  The file name has no "win-chain", "redo-chain" or
# "tail-chain": no busy() pattern of another chain sees it.
# RETEST_NO_RD=1: the RD configs are not read at all (the US-only queue run-queue-us.sh); the slices usT-1..usT-4 hold US entries only.
# RETEST_SLICE=<name> picks a slice of tools/pancake/box_chains/retest_groups.json (rdC-1, rdC-2, mixC-3, usC-4, mixR-5, mixR-6, usR-7) or
#   carry-rd / carry-us / carry (the flags bench-sweep.sh carried: $O/<branch>-carry-*.json, pass 2 over its budget or a drifted block).
# STAGE 2 of the US queue run-queue-us3.sh (owner protocol restated 2026-11-20, T-1005): RETEST_SLICE=carry-us-<tag> confirms the flags the stage-1 screen marked PROMISING
#   (carry files $O/us-carry-usS1-*.json, RETEST_CARRY_GLOB overrides the glob), the strongest first: RETEST_CARRY_OFFSET / RETEST_CARRY_MAX take the flags [offset, offset+max)
#   (6 per slice: 6 paired blocks + one re-run fit 2.5 h).  RETEST_SLICE=gates-us runs the quality gates (updated against updated minus / plus the flag): the static
#   T-1020: the gate list also reads $O/us-triage-rigor.json (the RIGOR verdicts; written by sweep.py rigor-verdict, never merged into us-triage.v2.json).
#   --lookup-max-t gate (attributes the earlier whole-preset REJECT) and one per flag whose clean confirm (source retest slice carry-us-usS2*, RETEST_GATE_SLICES) is lossy >= 3 %.
# A PAIRED BLOCK (sweep.py paired_spec) = abtest --tier screen --drift-check on a spec whose arms are warmup (the base, an extra first start: not counted, not judged), base, flag, base~2, flag~2, base~3, flag~3, and
#   abtest's base-late at the end: 8 engine starts (~19 min with short + edit + longdec32k), each flag rep between two base runs of the same block.
#   The drift (counted base runs against the first counted one) is judged on the last request of each case per start (sweep.py, T-1004).
#   The base drift (every later base run against the first) above SWEEP_DRIFT_MAX (5 %) on a case: the block runs once more when the budget holds
#   it; still above: INCONCLUSIVE (carried).  Verdicts (sweep.py retest-triage, per flag, its own block): KEEP >= 2 % (exact) / RETEST-GATE >= 3 %
#   (lossy) with no case below -3 %; OPTIONAL a case >= +10 % and a case <= -3 % (opt-in, not default); else DROP.  Bundles (a whole arm) and the
#   spec-223 reference are recorded with "bundle": true and never change a flag.  A gate entry runs quality_gate.py (set-v2, --near-tie: T-828).
# Bases: rd = /home/dan/auto-runs/benchRD/config-plain.json + what config-updated.json adds MINUS STRATA_POOL_FUSE_PHASES (RETEST_RD_DROP) with
#   ECAP as shipped (RETEST_RD_SET), the sweep's ARMFIX; us = the harness's plain-upstream config + spec 220's updated arm (222's updated-exact).
# Results: /home/dan/auto-runs/benchsweep/ (retest-<slice>-kNN.jsonl, -kNNr.jsonl re-run, <branch>-triage-rt-<slice>.v2.json, merged into
#   <branch>-triage.v2.json: what tools/pancake/apply_triage.py reads).  Budget 2.5 h (RETEST_MAX_S 9000 s): a block starts only while the budget
#   holds it; a block left out is logged as CARRY (it stays for the next carry slice / queue).
# Rules (as bench-sweep.sh): box jobs never overlap; pancake.service stays DOWN; the AA shell stays frozen (no signal); nothing is deleted; every
#   abtest / quality_gate run in its own session under a watchdog (run_group) and reaped.  Started by run-queue-sweep3.sh through bench-sched.v8.sh
#   (BENCH_ONLY=EXTRA BENCH_EXTRA=./bench-retest.v1.sh BENCH_NO_RESUME=1).
S=/home/dan/clone-of-d/src; R=$S/Strata-win; L=${RETEST_LOG:-$S/$(basename "$0" .sh).log}; O=/home/dan/auto-runs/benchsweep; T=$O/work; PM=/mnt/pancake/benchsweep; HN=$PM/harness
USB=/mnt/pancake/benchus; RDB=/mnt/pancake/benchRD; RDC=/home/dan/auto-runs/benchRD/config-plain.json; RDU=/home/dan/auto-runs/benchRD/config-updated.json
DATA=/home/dan/clone-of-d/Strata-data; ROCM=/opt/rocm/core-10.0; REPO_URL=https://github.com/rocktboy1982/pancake-strata.git
T0=$(date +%s); POLL=5; MAXS=${RETEST_MAX_S:-9000}; BUDGET=${BENCH_BUDGET_S:-$MAXS}
DMAX=${SWEEP_DRIFT_MAX:-5}
LAYOUT=${RETEST_LAYOUT:-interleaved}; WARMM=${RETEST_WARM:-each}   # compact: base, flag x 3, base-late (5 starts); RETEST_WARM=first: one warm-up start per slice
RS=${RETEST_SLICE:-}
export PATH=$ROCM/bin:$R/.venv/bin:$PATH LD_LIBRARY_PATH=$ROCM/lib HIP_VISIBLE_DEVICES=0
HPY=${BENCH_HARNESS_PY:-$R/.venv/bin/python}
export HPY ABTEST_PYTHON=$HPY
mkdir -p $O $T $PM
echo "START $(date +%F_%T) retest slice ${RS:-?}" >> $L
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
  echo WINBENCHRETEST_DONE >> $L
}
abort() { say "ERROR $*"; say "WINBENCHRETEST_ABORTED: $*"; finish; exit 1; }
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
  say "WINBENCHRETEST_ABORTED: signal"; finish; exit 1
}
trap on_signal HUP INT TERM

# ------------------------------------------------------------------ steps
ab() {  # ab NAME LIMIT SPEC flags...: one abtest run from the synced harness, in its own session
  nm=$1; lim=$2; sp=$3; shift 3
  [ "$(freekb /)" -ge 307200 ] || { say "ERROR $nm: / has < 300 MB free: not run"; return 1; }
  say "START $nm $(date +%F_%T) (limit $lim s, budget left $(left) s)"
  run_group $nm $lim $O/$nm.stdout sh -c 'cd "$1" && shift && exec "$HPY" tools/pancake/abtest.py "$@"' ab $HN $sp "$@" --out $O/$nm.jsonl; rc=$?
  say "END $nm exit=$rc $(date +%T)"
  [ $rc -eq 0 ] || say "ERROR $nm exit=$rc: $(errline $O/$nm.stdout $O/$nm) (the triage uses the rows that were measured)"
  grep -E "ORDER EFFECT|DRIFT" $O/$nm.stdout | cut -c1-220 >> $L
  return $rc
}
sw() { "$HPY" $HN/tools/pancake/box_chains/sweep.py "$@"; }

# ------------------------------------------------------------------ 0. slice, service down, idle, prerequisites
case "$RS" in ''|*[!A-Za-z0-9_-]*) abort "RETEST_SLICE must be a slice name of retest_groups.json or carry[-us|-rd] (got '$RS')";; esac
ls $O/retest-$RS-k[0-9][0-9].jsonl >/dev/null 2>&1 && abort "the retest slice $RS was run before ($O/retest-$RS-k*.jsonl exist): nothing is overwritten"
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
            tools/pancake/box_chains/retest_groups.json tools/pancake/box_chains/sweep_candidates.json bench/specs/220-updated-vs-upstream-short.json \
            bench/specs/222-updated-exact-vs-upstream-short.json configs/pancake/flashnext-iq2_xs-plain-upstream.json; do
  [ -f "$HN/$need" ] || abort "SYNC: $need is missing in the synced harness"
done
grep -q ABTEST_PYTHON $HN/tools/pancake/abtest.py || abort "PREREQ harness: $HN/tools/pancake/abtest.py does not read ABTEST_PYTHON"
e=$(cd $HN && PYTHONDONTWRITEBYTECODE=1 "$HPY" -B -c 'import serve.server' 2>&1) || abort "PREREQ harness: $HPY cannot import serve.server from $HN: $(echo "$e" | tail -n 1 | cut -c1-240)"
say "PREREQ harness ok: $HPY imports serve.server from $HN $(date +%T)"

# ------------------------------------------------------------------ 2. the blocks of the slice (one paired spec per flag)
fx=; for k in ${RETEST_RD_DROP-STRATA_POOL_FUSE_PHASES}; do fx="$fx --drop-env $k"; done
for kv in ${RETEST_RD_SET-STRATA_VERIFY_ECAP=0 STRATA_VERIFY_ECAP_MIN_ROWS=9}; do fx="$fx --set-env $kv"; done
cf=; case "$RS" in
  carry*) cf="--carry $(ls ${RETEST_CARRY_GLOB:-$O/*-carry-*.json} 2>/dev/null | tr '\n' ' ') --carry-offset ${RETEST_CARRY_OFFSET:-0}${RETEST_CARRY_MAX:+ --carry-max $RETEST_CARRY_MAX}";;
  gates*) cf="--gate-from $(ls $O/us-triage-rt-${RETEST_GATE_SLICES:-usS2r}*.v2.json $O/us-triage-${RETEST_GATE_SLICES:-usS2r}*.v2.json $O/us-triage-rigor.json 2>/dev/null | tr '\n' ' ') --gate-prefix ${RETEST_GATE_SLICES:-usS2r} --carry-offset ${RETEST_CARRY_OFFSET:-0}${RETEST_CARRY_MAX:+ --carry-max $RETEST_CARRY_MAX}";;
esac
rdargs="--rd-plain $RDC --rd-updated $RDU --rd-src $RDB/src-updated $fx"
[ "${RETEST_NO_RD:-0}" = 1 ] && rdargs=   # US-only queue (owner order 2026-11-20 09:19): no RD config is read, no RD arm is built
sw retest --slice $RS --out-dir $T --layout $LAYOUT --warm $WARMM $rdargs \
   --us-plain $HN/configs/pancake/flashnext-iq2_xs-plain-upstream.json --us-arm-spec $HN/bench/specs/220-updated-vs-upstream-short.json \
   --us-exact-spec $HN/bench/specs/222-updated-exact-vs-upstream-short.json --us-src $USB/src-us $cf > $T/retest-$RS.txt 2>&1 \
   || abort "SPEC: sweep.py retest failed: $(tail -n 1 $T/retest-$RS.txt)"
grep -E '^(ARMFIX|BLOCK|RETEST)' $T/retest-$RS.txt | cut -c1-400 >> $L
set -- $(grep '^ESTIMATE' $T/retest-$RS.txt); EST=${2:-0}
say "BUDGET $BUDGET s (estimate of the slice $EST s; a block starts only while the budget holds it, one re-run of a drifted block when it still fits)"

# ------------------------------------------------------------------ 3. block by block
nb=0; nclean=0; nrr=0; nbad=0; ncarry=0; rcs=0; runs=
tr -d '\r' < $T/retest-$RS.txt > $T/retest-$RS.lst
while read -r tag bid typ est br path bname aname cases flag; do
  [ "$tag" = BLOCK ] || continue
  nb=$((nb + 1))
  if ! left_ok "$est"; then ncarry=$((ncarry + 1)); say "CARRY $bid ($flag, $br): budget: $(left) s left (< $est s): not run in this slice"; continue; fi
  wait_idle
  case "$typ" in
    paired)
      ab retest-$RS-$bid $((est + 600)) $path --tier screen --drift-check --retries 0 || rcs=$?
      runs="$runs --run $bid=$O/retest-$RS-$bid.jsonl"
      d=$(sw paired-drift --rows $O/retest-$RS-$bid.jsonl --base $bname --arm $aname --cases $cases --max $DMAX --layout $LAYOUT); drc=$?
      say "DRIFT $bid ($flag, $br; paired block, base runs vs the first): $d"
      if [ $drc -ne 0 ]; then
        if left_ok $((est + 60)); then
          say "RERUN $bid: base drift above $DMAX % (or no base rows): the paired block runs once more"; wait_idle; nrr=$((nrr + 1))
          ab retest-$RS-${bid}r $((est + 600)) $path --tier screen --drift-check --retries 0 || rcs=$?
          runs="$runs --run $bid=$O/retest-$RS-${bid}r.jsonl"
          d=$(sw paired-drift --rows $O/retest-$RS-${bid}r.jsonl --base $bname --arm $aname --cases $cases --max $DMAX --layout $LAYOUT); drc=$?
          say "DRIFT ${bid}r: $d"
          if [ $drc -eq 0 ]; then nclean=$((nclean + 1)); else nbad=$((nbad + 1)); say "INCONCLUSIVE $bid ($flag): still above $DMAX % after its re-run"; fi
        else nbad=$((nbad + 1)); say "SKIP rerun $bid: budget: $(left) s left: $flag is INCONCLUSIVE (carried)"; fi
      else nclean=$((nclean + 1)); fi ;;
    gate)
      nm=retest-$RS-$bid-gate
      say "START $nm $(date +%F_%T) (quality_gate.py $flag, set bench/quality/set-v2.json, --near-tie, limit $((est + 300)) s, budget left $(left) s)"
      run_group $nm $((est + 300)) $O/$nm.stdout sh -c 'cd "$1" && shift && exec "$HPY" tools/pancake/quality_gate.py "$@"' qg $HN $path \
        --out $O/$nm --set bench/quality/set-v2.json --near-tie; rc=$?
      say "END $nm exit=$rc $(date +%T)"
      [ $rc -eq 0 ] || { rcs=$rc; say "ERROR $nm exit=$rc: $(errline $O/$nm.stdout $O/$nm)"; }
      grep -E "PASS|REJECT|INCONCLUSIVE|FAILED|not scored|near-tie|mismatch|events" $O/$nm.stdout | cut -c1-220 >> $L ;;
    spec)
      ab retest-$RS-$bid $((est + 600)) $HN/$path --tier screen --drift-check --retries 0 || rcs=$?
      runs="$runs --run $bid=$O/retest-$RS-$bid.jsonl" ;;
  esac
done < $T/retest-$RS.lst
sw retest-triage --blocks $T/retest-$RS.blocks.json $runs --drift-max $DMAX --out-dir $O > $T/retest-tri-$RS.txt 2>&1 || say "ERROR retest triage failed: $(tail -n 1 $T/retest-tri-$RS.txt)"
grep -v '^TRIAGEFILE' $T/retest-tri-$RS.txt | sed 's/^/TRIAGE /' | cut -c1-300 >> $L
for br in $(tr -d '\r' < $T/retest-tri-$RS.txt | awk '/^TRIAGEFILE/{print $2}'); do
  parts="$(ls -tr $O/$br-triage-[1-9]of[1-9].v2.json 2>/dev/null) $(ls -tr $O/$br-triage-*.v2.json 2>/dev/null | grep -v -e '-triage-[1-9]of[1-9]\.v2\.json$')"
  sw merge --out $O/$br-triage.v2.json $parts 2>> $L && say "TRIAGE file $O/$br-triage.v2.json (with $O/$br-triage-rt-$RS.v2.json; INCONCLUSIVE / NO-GAIN-P1 never replace a decided verdict)"
done
say "SUMMARY retest slice $RS: $nb blocks ($nclean clean, $nrr re-run, $nbad INCONCLUSIVE, $ncarry not run: budget), exit=$rcs; box time $(elapsed) s of $BUDGET s; results in $O"
finish
