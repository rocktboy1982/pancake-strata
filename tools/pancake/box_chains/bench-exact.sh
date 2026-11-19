#!/bin/sh
# bench-exact.sh - EXACT arm + QUALITY GATE follow-up of the plain-vs-updated benchmarks (owner order 2026-11-19).  The file name has no "win-chain",
# "redo-chain" or "tail-chain": no busy() pattern of another chain sees it (T-944).  Log /home/dan/clone-of-d/src/bench-exact.log, last line WINBENCHEXACT_DONE
# (also after an abort or a signal; the line before it is then WINBENCHEXACT_ABORTED).  Results: /home/dan/auto-runs/benchexact/.
#
# What it runs (no build, no download of engine sources: it reuses the builds bench-updatedAA.sh and bench-updatedRD.sh left on /mnt/pancake):
#   US part (needs /mnt/pancake/benchus/build-up/strata = plain v0.2.16, build-us/strata = updated-strata, the upstream expert profile of src-up):
#     a. spec 222 SCREEN (arms plain-upstream, updated-exact, updated + base-late = --drift-check), 2 requests per case (short, edit)
#     b. QUALITY GATE: tools/pancake/quality_gate.py on spec 222's three arms (plain-upstream scored twice = the noise; updated-exact and updated once each;
#        the standard set bench/quality/set-v2.json, 48 requests / 9,410 forced tokens, --near-tie: the settings of TEST_LEDGER.md T-828)
#     c. spec 223 long SCREEN (10K / 32K / 64K / 100K, 1 request per case and arm, no drift arm)
#   RD part (needs /mnt/pancake/benchRD/build-plain/strata, build-updated/strata and /home/dan/auto-runs/benchRD/config-plain.json from bench-updatedRD.sh):
#     d. spec 252 SCREEN (plain, updated-exact = the full-preset updated arm minus VERIFY_ECAP(+MIN_ROWS), updated = spec 250 updated; short, edit, long32k; T-984)
#     e. RD quality gate (quality_gate.py on spec 252) ONLY when both RD trees read STRATA_FOLLOW (token forcing, force_check.py --path decode); Strata-rdna3
#        does not (2026-11-19), so it is logged as SKIP with that reason instead of an engine start that cannot score
#   then, only for an arm whose screen showed a benefit (chainT-decide.py, >= 1.0 % lossless / >= 1.5 % lossy) and only while the budget lasts:
#     f. spec 222 CONFIRM, 3 reps (updated excluded: spec 220's confirm measured it on the same builds)    g. spec 252 CONFIRM, 3 reps (short, edit)
# A part whose builds / files are missing logs ERROR naming them and is skipped; the other part still runs.  EXACT_PARTS="US RD" (default) picks the parts.
#
# Rules (standing, as bench-updatedAA.sh): box jobs never overlap (busy() sees every other job and the other benchmark chains; a FROZEN AA shell is not a job);
# ~30 s quick idle (two clean polls of 15 s) before each part's steps; pancake.service stays DOWN (owner order 2026-11-19: only the benchmarks run; this chain
# never starts it; it stops it only after 3 minutes without a client, when it finds it running); AA stays frozen (no signal is ever sent to it); nothing is deleted;
# every long step runs in its OWN SESSION under a watchdog (run_group): at its limit the whole process tree gets TERM, 100 s later KILL; every process the step
# ever had (polled every 5 s, pid + start time, so the engines force_check.py starts in their own sessions are covered) that is still alive afterwards is killed
# and logged (REAP): no orphan engine survives a step.  A step's limit is capped at the budget left.
# BUDGET (owner: <= 75 min): BENCH_BUDGET_S = 4500 s.  Estimate (minutes, from the 2026-11-19 box run of bench-updatedAA.sh: a short/edit screen start ~4.2 min,
# a confirm start ~1.7 min; a gate scoring pass ~7 min, TEST_LEDGER.md T-828 / T-960): harness sync 2, idle polls 2.5, (a) 4 starts 17, (b) 4 passes 28,
# (c) 3 starts ~15 (unmeasured: one start per arm reads 206K prompt tokens), (d) 3 starts ~15 => ~80 min worst case for a-d: the steps run in that order and
# (d) / (f) / (g) start only when their estimate still fits the budget (logged as SKIP ... budget otherwise; EXACT_PARTS=RD runs the RD part alone later).
# Start: by the scheduler, after bench-updatedRD.sh has run:
#   cd /home/dan/clone-of-d/src && BENCH_ONLY=EXTRA BENCH_EXTRA=./bench-exact.sh BENCH_NO_RESUME=1 setsid nohup sh ./bench-sched.sh > /dev/null 2>&1 &
# (BENCH_SCHED=1, BENCH_PREIDLED=1, BENCH_KEEP_SERVICE_DOWN=1 come from the scheduler).  Standalone it first waits for WINBENCHUS_DONE and WINBENCHRD_DONE.
S=/home/dan/clone-of-d/src; R=$S/Strata-win; L=$S/bench-exact.log; O=/home/dan/auto-runs/benchexact; T=$O/work; PM=/mnt/pancake/benchexact; HN=$PM/harness
USB=/mnt/pancake/benchus; RDB=/mnt/pancake/benchRD; RDC=/home/dan/auto-runs/benchRD/config-plain.json; DEC=$S/chainT-decide.py
DATA=/home/dan/clone-of-d/Strata-data; ROCM=/opt/rocm/core-10.0; REPO_URL=https://github.com/rocktboy1982/pancake-strata.git
T0=$(date +%s); BUDGET=${BENCH_BUDGET_S:-4500}; PARTS=${EXACT_PARTS:-US RD}; POLL=5
LIM_SCREEN=${EXACT_LIM_SCREEN:-1500}; LIM_GATE=${EXACT_LIM_GATE:-2700}; LIM_LONG=${EXACT_LIM_LONG:-2100}; LIM_CONFIRM=${EXACT_LIM_CONFIRM:-1200}
# minimum budget left (s) for an optional step to start = its estimate above
NEED_GATE=${EXACT_NEED_GATE:-1200}; NEED_LONG=${EXACT_NEED_LONG:-600}; NEED_RD=${EXACT_NEED_RD:-900}; NEED_CONFIRM=${EXACT_NEED_CONFIRM:-700}; NEED_RDGATE=${EXACT_NEED_RDGATE:-1500}
export PATH=$ROCM/bin:$R/.venv/bin:$PATH LD_LIBRARY_PATH=$ROCM/lib HIP_VISIBLE_DEVICES=0
# 2026-11-19 16:42 (first run, TEST_LEDGER.md T-982): 222-screen, gate-us and 223-screen all failed in seconds: "env: '/mnt/pancake/benchexact/harness/.venv/bin/python':
# No such file or directory": abtest.py (and quality_gate.py through abtest.PY) starts serve/server.py with <its tree>/.venv/bin/python and the downloaded harness $HN has
# no .venv.  Now the harness runs with the box's existing venv interpreter HPY and ABTEST_PYTHON hands it to the server (as bench-updatedRD.sh, T-981); nothing is created
# under Strata-win or in $HN.  PREREQ checks HPY; right after the harness sync HPY must import serve.server from $HN and $HN's abtest.py must read ABTEST_PYTHON.
HPY=${BENCH_HARNESS_PY:-$R/.venv/bin/python}
export HPY ABTEST_PYTHON=$HPY
mkdir -p $O $T $PM
echo "START $(date +%F_%T) parts: $PARTS, budget $BUDGET s" >> $L
say() { echo "$*" >> $L; }
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[c]onv_trace.py|[a]uto-chain|[a]uto-post|[i]q_avx2_parity|[t]tft_bench.py|[w]in-chain[A-Za-z0-9]*\.sh|[x-z]tail-chain[A-Z]\.sh|[a]fter7|[b]ench-(updated(AA|RD)|diagUS)[.a-z0-9]*\.sh" >/dev/null 2>&1; }
# a FROZEN (state T) AA shell is not a running job; an AA shell that is not stopped is
aa_live() { for p in $(pgrep -f "[r]edo-chainAA\.sh" 2>/dev/null); do case "$(ps -o stat= -p $p 2>/dev/null)" in T*) ;; *) return 0;; esac; done; return 1; }
bz() { busy || aa_live; }
QI=${BENCH_PREIDLED:-0}   # 1: the scheduler already waited for an idle box; every wait here is the quick one (two clean polls of 15 s)
wait_idle() { if [ "$QI" = 1 ]; then q=0; while [ $q -lt 2 ]; do if bz; then q=0; sleep 15; else q=$((q+1)); sleep 15; fi; done; else q=0; while [ $q -lt 3 ]; do if bz; then q=0; sleep 60; else q=$((q+1)); sleep 60; fi; done; QI=1; fi; }
elapsed() { echo $(( $(date +%s) - T0 )); }
left() { echo $(( BUDGET - $(elapsed) )); }
left_ok() { [ "$(left)" -ge "$1" ]; }
clients() { ss -Htn state established '( sport = :11437 )' 2>/dev/null | wc -l; }
freekb() { df -Pk "$1" | awk 'NR==2{print $4}'; }
has_part() { case " $PARTS " in *" $1 "*) return 0;; esac; return 1; }
finish() {  # the end on every path: the service stays down, the marker is the last line
  say "service stays down (owner order 2026-11-19: only the benchmarks run; this chain never starts pancake.service) $(date +%T)"
  say "/ free $(freekb /) KB, /mnt/pancake free $(freekb /mnt/pancake) KB, box time $(elapsed) s"
  echo WINBENCHEXACT_DONE >> $L
}
abort() { say "ERROR $*"; say "WINBENCHEXACT_ABORTED: $*"; finish; exit 1; }
errline() {  # errline STDOUT_FILE WORKDIR: the first error-looking line of the stdout and the last one of the engine logs of the failed starts
  pat='error|fatal|abort|died|failed|needs|cannot|exception|terminate|no such|not found|unrecognized|unknown'
  a=$(grep -i -m1 -E "$pat" "$1" 2>/dev/null | cut -c1-240); [ -n "$a" ] || a=$(grep -m1 . "$1" 2>/dev/null | cut -c1-240)
  e=; for f in "$2"/*.engine.log "$2"/*.out; do [ -f "$f" ] || continue; e=$(grep -i -E "$pat" "$f" 2>/dev/null | tail -n 1 | cut -c1-240); [ -n "$e" ] && break; done
  echo "first error line of the stdout: [$a]; engine log: [$e]"
}

# ------------------------------------------------------------------ process groups: a watchdog per step, no orphan afterwards
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
  say "WINBENCHEXACT_ABORTED: signal"; finish; exit 1
}
trap on_signal HUP INT TERM

# ------------------------------------------------------------------ steps
ab() {  # ab NAME LIMIT SPEC flags...: one abtest run from the synced harness, in its own session
  nm=$1; lim=$2; sp=$3; shift 3
  [ "$(freekb /)" -ge 307200 ] || { say "ERROR $nm: / has < 300 MB free: not run"; return 1; }
  say "START $nm $(date +%F_%T) (limit $lim s, budget left $(left) s)"
  run_group $nm $lim $O/benchexact-$nm.stdout sh -c 'cd "$1" && shift && exec "$HPY" tools/pancake/abtest.py "$@"' ab $HN $sp "$@" --out $O/benchexact-$nm.jsonl; rc=$?
  say "END $nm exit=$rc $(date +%T)"
  [ $rc -eq 0 ] || say "ERROR $nm exit=$rc: $(errline $O/benchexact-$nm.stdout $O/benchexact-$nm)"
  grep -E "vs .*(better|worse|no difference)|verdict|PROMOTE|drop|DRIFT|ORDER EFFECT" $O/benchexact-$nm.stdout | cut -c1-220 >> $L
  grep -A6 "generated text vs base" $O/benchexact-$nm.stdout | cut -c1-200 >> $L
  return $rc
}
gate() {  # gate NAME SPEC: token-forcing quality gate of every arm against the first (scored twice), the T-828 settings
  nm=$1; sp=$2
  say "START $nm $(date +%F_%T) (quality_gate.py, set bench/quality/set-v2.json, --near-tie, limit $LIM_GATE s, budget left $(left) s)"
  run_group $nm $LIM_GATE $O/benchexact-$nm.stdout sh -c 'cd "$1" && shift && exec "$HPY" tools/pancake/quality_gate.py "$@"' qg $HN $sp \
    --out $O/$nm --set bench/quality/set-v2.json --near-tie; rc=$?
  say "END $nm exit=$rc $(date +%T)"
  [ $rc -eq 0 ] || say "ERROR $nm exit=$rc: $(errline $O/benchexact-$nm.stdout $O/$nm)"
  grep -E "PASS|REJECT|INCONCLUSIVE|FAILED|not scored|near-tie|mismatch|events" $O/benchexact-$nm.stdout | cut -c1-220 >> $L
  return $rc
}
decide() {  # decide NAME SPEC CASES_JSON [EXCLUDE]: 0 and $T/NAME-confirm.json (cases cut, no drift arm) when an arm earned the confirm
  rm -f "${T:?}/$1-confirm.json"
  python3 $DEC $O/benchexact-$1-screen.stdout $2 $T/$1-confirm.json ${4:+--exclude $4} > $T/$1-decide.txt 2>&1
  grep -E "WIN|CONFIRM|NONE|dropped" $T/$1-decide.txt | cut -c1-230 >> $L
  [ -s $T/$1-confirm.json ] || return 1
  python3 -c 'import json,sys; p=sys.argv[1]; d=json.load(open(p)); d["cases"]=json.loads(sys.argv[2]); d["drift_check"]=False; json.dump(d,open(p,"w"),indent=1)' $T/$1-confirm.json "$3"
}
confirm() {  # confirm NAME: the 3-rep confirm of a decided spec, when the budget still holds it
  if left_ok $NEED_CONFIRM; then
    wait_idle; ab $1-confirm $LIM_CONFIRM $T/$1-confirm.json --tier confirm --block 3 --warmup 1 --outlier robust --reclaim-max 3 --reps 3 --no-early-stop
  else say "SKIP $1 confirm: budget: $(left) s left (< $NEED_CONFIRM s); the screen numbers stand"; return 0; fi
}

# ------------------------------------------------------------------ 0. (standalone: wait for both benchmark chains), service down, idle, prerequisites
if [ "${BENCH_SCHED:-0}" != 1 ]; then
  until grep -q WINBENCHUS_DONE $S/bench-updatedAA.log 2>/dev/null && grep -q WINBENCHRD_DONE $S/bench-updatedRD.log 2>/dev/null; do sleep 300; done
  say "both benchmark chains are done $(date +%T)"
else say "started by the scheduler: the AA shell is frozen, the box was idle $(date +%T)"; fi
if systemctl --user is-active --quiet pancake.service; then
  q=0; while [ $q -lt 3 ]; do if [ "$(clients)" -gt 0 ]; then q=0; say "WAIT service has clients $(date +%T)"; else q=$((q+1)); fi; sleep 60; done
  systemctl --user stop pancake.service; say "service stopped (idle) $(date +%T)"
else say "service already stopped"; fi
wait_idle
[ "$(freekb /mnt/pancake)" -ge 2097152 ] || abort "START: /mnt/pancake has $(freekb /mnt/pancake) KB free (< 2 GB); nothing was deleted"
[ "$(freekb /)" -ge 307200 ] || abort "START: / has $(freekb /) KB free (< 300 MB)"
miss=; for f in $DATA/packs/iq2_xs/tokenizer $DATA/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
               $DATA/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf $DATA/mtp/rt $DEC $HPY; do [ -e "$f" ] || miss="$miss $f"; done
[ -z "$miss" ] || abort "PREREQ missing files (both parts need them):$miss"
"$HPY" -c 'import sys' >/dev/null 2>&1 || abort "PREREQ the harness python $HPY does not run"
US_OK=0; RD_OK=0
if has_part US; then
  miss=; for f in $USB/build-up/strata $USB/build-us/strata $USB/src-up/terminal/data/expert-profile.bin; do [ -e "$f" ] || miss="$miss $f"; done
  if [ -z "$miss" ]; then US_OK=1; say "PREREQ US ok: $USB/build-up/strata (plain v0.2.16), $USB/build-us/strata (updated-strata), upstream expert profile"
  else say "ERROR PREREQ US part SKIPPED: missing$miss (bench-updatedAA.sh builds them; nothing is built here)"; fi
fi
if has_part RD; then
  miss=; for f in $RDB/build-plain/strata $RDB/build-updated/strata $RDC; do [ -e "$f" ] || miss="$miss $f"; done
  if [ -z "$miss" ]; then RD_OK=1; say "PREREQ RD ok: $RDB/build-plain/strata, $RDB/build-updated/strata, $RDC"
  else say "ERROR PREREQ RD part SKIPPED: missing$miss (bench-updatedRD.sh builds / writes them; nothing is built here)"; fi
fi
[ $US_OK = 1 ] || [ $RD_OK = 1 ] || abort "no part can run (see the PREREQ lines)"

# ------------------------------------------------------------------ 1. harness: origin port/rdna3-decode (abtest, quality_gate, specs 222 / 223 / 252), mtimes reset
SHA=$(timeout 60 git ls-remote $REPO_URL refs/heads/port/rdna3-decode 2>/dev/null | cut -f1)
HARN=$PM/harness.tgz; rm -f $HARN.part
if [ -n "$SHA" ] && timeout 1200 curl -sfL -o $HARN.part https://codeload.github.com/rocktboy1982/pancake-strata/tar.gz/$SHA && tar tzf $HARN.part >/dev/null 2>&1; then
  mv $HARN.part $HARN; say "SYNC harness: origin port/rdna3-decode $SHA -> $HN"
else abort "SYNC harness download failed"; fi
mkdir -p $HN && tar xzmf $HARN --no-same-owner --strip-components=1 -C $HN || abort "SYNC harness tar failed"
for need in tools/pancake/abtest.py tools/pancake/quality_gate.py tools/force_check.py bench/quality/set-v2.json configs/pancake/flashnext-iq2_xs-plain-upstream.json \
            bench/specs/222-updated-exact-vs-upstream-short.json bench/specs/223-updated-exact-vs-upstream-long.json bench/specs/252-rdna3-updated-exact.json; do
  [ -f "$HN/$need" ] || abort "SYNC: $need is missing in the synced harness"
done
grep -q ABTEST_PYTHON $HN/tools/pancake/abtest.py || abort "PREREQ harness: $HN/tools/pancake/abtest.py does not read ABTEST_PYTHON: its server would start with $HN/.venv/bin/python, which does not exist"
e=$(cd $HN && PYTHONDONTWRITEBYTECODE=1 "$HPY" -B -c 'import serve.server' 2>&1) || abort "PREREQ harness: $HPY cannot import serve.server from $HN: $(echo "$e" | tail -n 1 | cut -c1-240)"
say "PREREQ harness ok: $HPY imports serve.server from $HN; abtest / quality_gate start the server with ABTEST_PYTHON=$HPY $(date +%T)"
S222=bench/specs/222-updated-exact-vs-upstream-short.json; S223=bench/specs/223-updated-exact-vs-upstream-long.json
if [ $RD_OK = 1 ]; then
  sed "s#@PLAIN@#$RDB/build-plain#; s#@UPDATED@#$RDB/build-updated#; s#@CONFIG_PLAIN@#$RDC#" $HN/bench/specs/252-rdna3-updated-exact.json > $T/252.json
fi

# ------------------------------------------------------------------ 2. US: spec 222 screen, the quality gate, spec 223 long screen
rc222=-; rcg=-; rc223=-; rc252=-; rcrg=-
if [ $US_OK = 1 ]; then
  wait_idle; ab 222-screen $LIM_SCREEN $S222 --tier screen --drift-check; rc222=$?
  if left_ok $NEED_GATE; then wait_idle; gate gate-us $S222; rcg=$?
  else say "SKIP gate-us: budget: $(left) s left (< $NEED_GATE s)"; fi
  if left_ok $NEED_LONG; then wait_idle; ab 223-screen $LIM_LONG $S223 --tier screen; rc223=$?
  else say "SKIP 223-screen: budget: $(left) s left (< $NEED_LONG s)"; fi
fi

# ------------------------------------------------------------------ 3. RD: spec 252 screen, its gate when the trees can be scored
if [ $RD_OK = 1 ]; then
  if left_ok $NEED_RD; then
    wait_idle; ab 252-screen $LIM_SCREEN $T/252.json --tier screen; rc252=$?
    if grep -q '"STRATA_FOLLOW"' $RDB/src-plain/src/program/generate.cpp 2>/dev/null && grep -q '"STRATA_FOLLOW"' $RDB/src-updated/src/program/generate.cpp 2>/dev/null; then
      if left_ok $NEED_RDGATE; then wait_idle; gate gate-rd $T/252.json; rcrg=$?
      else say "SKIP gate-rd: budget: $(left) s left (< $NEED_RDGATE s)"; fi
    else say "SKIP gate-rd: token forcing needs STRATA_FOLLOW (force_check.py score --path decode) and $RDB/src-plain or src-updated src/program/generate.cpp does not read it; the prompt path (STRATA_LOGPOS --short-read) bypasses the prompt kernels PF_F16 changes, so it would not test the switch"; fi
  else say "SKIP RD part: budget: $(left) s left (< $NEED_RD s); run it alone with EXACT_PARTS=RD"; fi
fi

# ------------------------------------------------------------------ 4. confirms (3 reps) only for arms with a screen benefit, while the budget lasts
if [ "$rc222" = 0 ]; then
  if decide 222 $HN/$S222 '["short","edit"]' updated; then confirm 222
  else say "222: no arm with a benefit in the screen (updated excluded: spec 220 confirmed it): no confirm"; fi
fi
if [ "$rc252" = 0 ]; then
  if decide 252 $T/252.json '["short","edit"]'; then confirm 252
  else say "252: no arm with a benefit in the screen: no confirm"; fi
fi

say "SUMMARY spec222 screen exit=$rc222, gate-us exit=$rcg, spec223 screen exit=$rc223, spec252 screen exit=$rc252, gate-rd exit=$rcrg (- = not run); box time $(elapsed) s of $BUDGET s; results in $O"
finish
