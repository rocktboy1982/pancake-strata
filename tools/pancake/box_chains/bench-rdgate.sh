#!/bin/sh
# bench-rdgate.sh - TASK GATE of the RDNA3 benchmark (owner approved 2026-11-19): updated-strata-rdna3 `updated` (the shipped recommended flags: POOL_FUSE_PHASES=1
# exact + PF_F16=1 lossy, exactly the env config-updated.json adds over config-plain.json) against the PLAIN Strata-rdna3, judged by tools/pancake/task_gate.py.
# Log /home/dan/clone-of-d/src/bench-rdgate.log, last line WINBENCHRDGATE_DONE (also after an abort or a signal; the line before it is then WINBENCHRDGATE_ABORTED).
# Results: /home/dan/auto-runs/benchrdgate/ (gate/REPORT.txt, gate/runs-*.jsonl, benchrdgate-gate.stdout, work/rdgate.json).  The file name has no "win-chain",
# "redo-chain" or "tail-chain": no busy() pattern of another chain sees it (T-944).
#
# Why the task gate and not the token-forcing gate: quality_gate.py needs STRATA_FOLLOW (force_check.py score --path decode) and Strata-rdna3 does not read it
# (bench-exact.sh logs gate-rd as SKIP for that reason); task_gate.py needs no engine hook, only the server API, so it scores both builds as they are.
# Settings (the documented standard, AGENTS.md "for close variants use task_gate.py"; TEST_LEDGER.md T-819 / taskgate-lossy.json / spec 111): task set
# bench/tasks/default.json (18 tasks: code_edit, tool_call, qa), 5 reps, temperature 0.2, max_tokens 1024, thinking off, margin 5 points (validity margin 3),
# seed 1000 + rep paired across arms, the base arm (plain) run twice (first and last: run-to-run noise and drift), time-AUC horizon 1800 s.  Shorter documented
# modes exist (--no-base-repeat: base once, drops the noise / drift check; --limit N: a task prefix) but neither gives a stronger verdict: T-481 already puts
# 18 tasks x 5 reps at INCONCLUSIVE in ~90 % of trials for a true loss of 0, so they are not used.  Known limit (T-819): default.json sat at 100 % on the pancake
# base, so a PASS here can be a ceiling; the report says so when every run passes.
#
# What it runs (no build, no engine download: it reuses /mnt/pancake/benchRD/build-plain/strata, build-updated/strata and the two server configs
# /home/dan/auto-runs/benchRD/config-plain.json, config-updated.json that bench-updatedRD.sh left):
#   0. (standalone only: wait for WINBENCHRD_DONE in bench-updatedRD.log)  service check, ~30 s quick idle, PREREQ (builds, configs, model files, harness python)
#   1. harness: origin port/rdna3-decode as a codeload tarball into its own dir /mnt/pancake/benchrdgate/harness (mtimes reset); it must have task_gate.py,
#      abtest.py reading ABTEST_PYTHON (the downloaded tree has no .venv: the server abtest starts runs with the box venv python), bench/tasks/default.json,
#      and the venv python must import serve.server from it
#   2. the gate spec work/rdgate.json: base = config-plain.json, arm plain = build-plain/strata, arm updated = build-updated/strata + every env key / top-level key
#      config-updated.json changes against config-plain.json
#   3. RD PARITY CTESTS (the RD rerun's ctest step did not run: "ctest: not found", exit 127: ctest is not on PATH; it is /home/linuxbrew/.linuxbrew/bin/ctest
#      next to cmake): verify_ecap_test|sampler_parity|elementwise_parity, as bench-updatedRD.sh intends, on build-updated AND build-plain, by full path, each in
#      its own session (<= 600 s).  The plain build was built for the strata target only, so its parity binaries may be missing: ctest then reports them as
#      not run (logged, never a stop)
#   4. RD DIAG-ONLY (the RD rerun's diag-updated ended at once with all 5 flags UNTESTED: its base arm got abtest exit=3 "another test or engine is running",
#      a leftover of the smoke step before it): flag_diag.py of the updated-strata-rdna3 tree /mnt/pancake/benchRD/src-updated (per-flag process groups,
#      diag_group.py) on config-updated.json, flags POOL_FUSE_PHASES,VERIFY_ECAP,PF_F16,RDNA_MMVQ_WAVE,KV_HOST_DMA, parity ctest of the exact flags on build-updated
#      (ctest on PATH now), the harness abtest.py, ABTEST_PYTHON = HPY; before it the reap of the step before and a "no engine running" check (strata --serve,
#      serve/server.py, abtest.py, task_gate.py: wait <= 120 s, then the diag is skipped with an ERROR naming them; nothing of another job is killed).
#      Verdicts /home/dan/auto-runs/benchrdgate/diag/verdicts.json (rewritten after every arm, so a cut run keeps what it measured); limit RDGATE_LIM_DIAG = 1500 s
#   5. task_gate.py --dry-run (spec + task set + every checker, no server, its own session, <= 600 s)
#   6. task_gate.py (its own session, watchdog, limit RDGATE_LIM_GATE = 4800 s capped at the budget left, then the process-tree reap; same "no engine" check first)
# Every python of the harness is the box venv interpreter HPY = /home/dan/clone-of-d/src/Strata-win/.venv/bin/python (as bench-updatedAA.sh and the RD fix,
# T-980), also exported as ABTEST_PYTHON: the code_edit checkers run with the same interpreter (task_gate.py uses sys.executable).
#
# Rules (standing, as bench-exact.sh): box jobs never overlap (busy() sees every other job and the other benchmark chains; a FROZEN AA shell is not a job);
# ~30 s quick idle (two clean polls of 15 s) before the gate; pancake.service stays DOWN (this chain never starts it; it stops it only after 3 minutes without a
# client, when it finds it running); AA stays frozen (no signal is ever sent to it); nothing is deleted; every long step runs in its OWN SESSION under a watchdog
# (run_group): at its limit the whole process tree gets TERM, 100 s later KILL; every process the step ever had (polled every 5 s, pid + start time, so the
# engines the gate starts are covered) that is still alive afterwards is killed and logged (REAP): no orphan engine survives a step.
# BUDGET: BENCH_BUDGET_S = 5400 s (90 min).  Estimate (minutes): harness sync 2, idle polls 1-2, ctests 2 x 1-3, diag 6 arms x ~1.5-2 (the RD smoke of 2026-11-19:
# engine start + one short request 63 s, decode 108.9 tok/s) + parity ctests ~2 => ~12-15 (limit 25), dry run 1, gate = 3 server runs (plain, updated, plain again)
# x (engine start ~1 + 90 requests: T-819 took ~3 min an arm on the pancake base, 1.2 s median a success) => ~4-8 an arm, ~15-25 => ~35-50 in all; worst case the
# budget: every step's limit is capped at the budget left, and the gate starts only with >= RDGATE_NEED_GATE = 1800 s left (else SKIP, logged).
# Start: by the scheduler, after bench-updatedRD.sh has built both trees (a launcher script's name must not contain bench-exact / bench-updated / bench-rdgate:
# the scheduler's idle wait matches those names):
#   cd /home/dan/clone-of-d/src && BENCH_NO_RESUME=1 BENCH_ONLY=EXTRA BENCH_EXTRA=./bench-rdgate.sh setsid nohup sh ./bench-sched.sh > /dev/null 2>&1 &
# (BENCH_SCHED=1, BENCH_PREIDLED=1, BENCH_KEEP_SERVICE_DOWN=1 come from the scheduler).
S=/home/dan/clone-of-d/src; R=$S/Strata-win; L=$S/bench-rdgate.log; O=/home/dan/auto-runs/benchrdgate; T=$O/work; PM=/mnt/pancake/benchrdgate; HN=$PM/harness
RDB=/mnt/pancake/benchRD; RDP=/home/dan/auto-runs/benchRD/config-plain.json; RDU=/home/dan/auto-runs/benchRD/config-updated.json
DATA=/home/dan/clone-of-d/Strata-data; ROCM=/opt/rocm/core-10.0; REPO_URL=https://github.com/rocktboy1982/pancake-strata.git
UPD_SRC=$RDB/src-updated; FD=$UPD_SRC/tools/pancake/flag_diag.py; BREW=/home/linuxbrew/.linuxbrew/bin; CT=$BREW/ctest
DIAG_FLAGS=POOL_FUSE_PHASES,VERIFY_ECAP,PF_F16,RDNA_MMVQ_WAVE,KV_HOST_DMA; CT_RX="verify_ecap_test|sampler_parity|elementwise_parity"
# the harness python: the box venv (the downloaded harness has no .venv; TEST_LEDGER.md T-981 / T-982); ABTEST_PYTHON hands it to the servers abtest starts
HPY=${BENCH_HARNESS_PY:-$R/.venv/bin/python}
T0=$(date +%s); BUDGET=${BENCH_BUDGET_S:-5400}; POLL=5
LIM_CT=${RDGATE_LIM_CT:-600}; LIM_DIAG=${RDGATE_LIM_DIAG:-1500}; FLAG_TIMEOUT=${FLAG_TIMEOUT:-240}; CTEST_TIMEOUT=300
LIM_DRY=${RDGATE_LIM_DRY:-600}; LIM_GATE=${RDGATE_LIM_GATE:-4800}; NEED_GATE=${RDGATE_NEED_GATE:-1800}; NEED_DIAG=${RDGATE_NEED_DIAG:-3300}
TASKS=bench/tasks/default.json; REPS=${RDGATE_REPS:-5}
export HPY ABTEST_PYTHON=$HPY
# $BREW: ctest (flag_diag.py calls `ctest` through PATH; the RD rerun failed with "ctest: not found")
export PATH=$ROCM/bin:$R/.venv/bin:$BREW:$PATH LD_LIBRARY_PATH=$ROCM/lib HIP_VISIBLE_DEVICES=0
mkdir -p $O $T $PM
echo "START $(date +%F_%T) budget $BUDGET s, harness python $HPY" >> $L
say() { echo "$*" >> $L; }
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[c]onv_trace.py|[a]uto-chain|[a]uto-post|[i]q_avx2_parity|[t]tft_bench.py|[w]in-chain[A-Za-z0-9]*\.sh|[x-z]tail-chain[A-Z]\.sh|[a]fter7|[b]ench-(updated(AA|RD)|diagUS|exact)[.a-z0-9]*\.sh" >/dev/null 2>&1; }
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
finish() {  # the end on every path: the service stays down, the marker is the last line
  say "service stays down (owner order 2026-11-19: only the benchmarks run; this chain never starts pancake.service) $(date +%T)"
  say "/ free $(freekb /) KB, /mnt/pancake free $(freekb /mnt/pancake) KB, box time $(elapsed) s"
  echo WINBENCHRDGATE_DONE >> $L
}
abort() { say "ERROR $*"; say "WINBENCHRDGATE_ABORTED: $*"; finish; exit 1; }
errline() {  # errline STDOUT_FILE WORKDIR: the first error-looking line of the stdout and the last one of the engine logs of the failed starts
  pat='error|fatal|abort|died|failed|needs|cannot|exception|terminate|no such|not found|unrecognized|unknown'
  a=$(grep -i -m1 -E "$pat" "$1" 2>/dev/null | cut -c1-240); [ -n "$a" ] || a=$(grep -m1 . "$1" 2>/dev/null | cut -c1-240)
  e=; for f in "$2"/*.engine.log "$2"/*.out "$2"/*/*.engine.log "$2"/*/*.out; do [ -f "$f" ] || continue; e=$(grep -i -E "$pat" "$f" 2>/dev/null | tail -n 1 | cut -c1-240); [ -n "$e" ] && break; done
  echo "first error line of the stdout: [$a]; engine log: [$e]"
}

# ------------------------------------------------------------------ process groups: a watchdog per step, no orphan afterwards (as bench-exact.sh)
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
  say "WINBENCHRDGATE_ABORTED: signal"; finish; exit 1
}
trap on_signal HUP INT TERM

# ------------------------------------------------------------------ 0. (standalone: wait for the RD benchmark), service down, idle, prerequisites
if [ "${BENCH_SCHED:-0}" != 1 ]; then
  until grep -q WINBENCHRD_DONE $S/bench-updatedRD.log 2>/dev/null; do sleep 300; done
  say "the RD benchmark chain is done $(date +%T)"
else say "started by the scheduler: the AA shell is frozen, the box was idle $(date +%T)"; fi
if systemctl --user is-active --quiet pancake.service; then
  q=0; while [ $q -lt 3 ]; do if [ "$(clients)" -gt 0 ]; then q=0; say "WAIT service has clients $(date +%T)"; else q=$((q+1)); fi; sleep 60; done
  systemctl --user stop pancake.service; say "service stopped (idle) $(date +%T)"
else say "service already stopped"; fi
wait_idle
[ "$(freekb /mnt/pancake)" -ge 2097152 ] || abort "START: /mnt/pancake has $(freekb /mnt/pancake) KB free (< 2 GB); nothing was deleted"
[ "$(freekb /)" -ge 307200 ] || abort "START: / has $(freekb /) KB free (< 300 MB)"
miss=; for f in $RDB/build-plain/strata $RDB/build-updated/strata $RDP $RDU $HPY $DATA/packs/iq2_xs/tokenizer \
               $DATA/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf $DATA/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf \
               $DATA/mtp/rt; do [ -e "$f" ] || miss="$miss $f"; done
[ -z "$miss" ] || abort "PREREQ missing:$miss (bench-updatedRD.sh builds the RD trees and writes the configs; nothing is built here)"
"$HPY" -c 'import sys' >/dev/null 2>&1 || abort "PREREQ the harness python $HPY does not run"
say "PREREQ ok: $RDB/build-plain/strata, $RDB/build-updated/strata, $RDP, $RDU, model + pack + mtp files, harness python $HPY"

# ------------------------------------------------------------------ 1. harness: origin port/rdna3-decode (task_gate, abtest, the task set) in its own dir, mtimes reset
SHA=$(timeout 60 git ls-remote $REPO_URL refs/heads/port/rdna3-decode 2>/dev/null | cut -f1)
HARN=$PM/harness.tgz; rm -f $HARN.part
if [ -n "$SHA" ] && timeout 1200 curl -sfL -o $HARN.part https://codeload.github.com/rocktboy1982/pancake-strata/tar.gz/$SHA && tar tzf $HARN.part >/dev/null 2>&1; then
  mv $HARN.part $HARN; say "SYNC harness: origin port/rdna3-decode $SHA -> $HN"
else abort "SYNC harness download failed"; fi
mkdir -p $HN && tar xzmf $HARN --no-same-owner --strip-components=1 -C $HN || abort "SYNC harness tar failed"
for need in tools/pancake/task_gate.py tools/pancake/abtest.py tools/pancake/capped.sh serve/server.py $TASKS; do
  [ -f "$HN/$need" ] || abort "SYNC: $need is missing in the synced harness"
done
grep -q ABTEST_PYTHON $HN/tools/pancake/abtest.py || abort "PREREQ harness: $HN/tools/pancake/abtest.py does not read ABTEST_PYTHON: its server would start with $HN/.venv/bin/python, which does not exist"
e=$(cd $HN && PYTHONDONTWRITEBYTECODE=1 "$HPY" -B -c 'import serve.server' 2>&1) || abort "PREREQ harness: $HPY cannot import serve.server from $HN: $(echo "$e" | tail -n 1 | cut -c1-240)"
say "PREREQ harness ok: $HPY imports serve.server from $HN; the gate's servers start with ABTEST_PYTHON=$HPY"

# ------------------------------------------------------------------ 2. the gate spec: plain vs updated exactly as config-plain.json / config-updated.json differ
"$HPY" -c '
import json, sys
bp, bu, cp, cu, tasks, reps, out = sys.argv[1:8]
p, u = json.load(open(cp)), json.load(open(cu))
pe, ue = p.get("env") or {}, u.get("env") or {}
env = {k: v for k, v in ue.items() if pe.get(k) != v}
env.update({k: None for k in pe if k not in ue})
cfg = {k: u.get(k) for k in sorted(set(p) | set(u)) if k not in ("exe", "env") and p.get(k) != u.get(k)}
upd = {"exe": bu + "/strata", "env": env}
if cfg:
    upd["config"] = cfg
if not env and not cfg:
    sys.exit("config-updated.json does not differ from config-plain.json: nothing to gate")
spec = {"base": cp, "tasks": tasks, "reps": int(reps), "temperature": 0.2, "margin": 0.05, "auc_horizon_s": 1800,
        "arms": {"plain": {"exe": bp + "/strata"}, "updated": upd},
        "note": "bench-rdgate.sh: task gate of updated-strata-rdna3 (config-updated.json = config-plain.json + the shipped recommended flags) vs plain Strata-rdna3"}
json.dump(spec, open(out, "w"), indent=1)
print("updated arm env:", json.dumps(env, sort_keys=True), "config keys:", sorted(cfg))
' $RDB/build-plain $RDB/build-updated $RDP $RDU $TASKS $REPS $T/rdgate.json > $T/spec.txt 2>&1 || abort "SPEC: the gate spec could not be made: $(tail -n 1 $T/spec.txt | cut -c1-240)"
say "SPEC $T/rdgate.json: base $RDP, plain = $RDB/build-plain/strata, $(cut -c1-400 $T/spec.txt)"
grep -q STRATA_PF_F16 $T/rdgate.json || say "NOTE the updated arm does not change STRATA_PF_F16 (the shipped file changed?): the gate judges what config-updated.json holds"

# ------------------------------------------------------------------ 3. dry run of the gate (no server, no GPU: first, so a broken spec or checker is known before the engines run)
GO=$O/gate; DO=$O/diag
say "START dryrun $(date +%F_%T) (task_gate.py --dry-run, limit $LIM_DRY s)"
run_group dryrun $LIM_DRY $O/benchrdgate-dryrun.stdout sh -c 'cd "$1" && shift && exec "$@"' dr $HN "$HPY" tools/pancake/task_gate.py $T/rdgate.json \
  --out $GO --tasks $TASKS --reps $REPS --dry-run; rcd=$?
say "END dryrun exit=$rcd $(date +%T)"
grep -E "dry run|FAIL|problem|error" $O/benchrdgate-dryrun.stdout | cut -c1-220 >> $L
[ $rcd -eq 0 ] || say "ERROR dryrun exit=$rcd: $(errline $O/benchrdgate-dryrun.stdout $GO); the gate will not run (the ctests and the diag still do)"

# "no engine running": the reap of the step before already ran (run_group); a server / engine / harness of ANOTHER job is waited for (<= 120 s), never killed
engines() { pgrep -f "[s]trata --serve|[s]erve/server.py|[a]btest.py|[t]ask_gate.py|[f]lag_diag.py" 2>/dev/null; }
no_engine() {  # no_engine STEP: 0 when no engine / server / harness runs (polled every 15 s, <= 8 polls); else ERROR naming them
  n=0; while engines >/dev/null; do
    if [ $n -ge 8 ]; then say "ERROR $1 not started: an engine / server / harness is still running after 120 s: $(for p in $(engines); do printf '%s [%s] ' $p "$(ps -o args= -p $p 2>/dev/null | cut -c1-100)"; done)"; return 1; fi
    n=$((n + 1)); sleep 15
  done
  say "NO-ENGINE check ok before $1 $(date +%T)"; return 0
}

# ------------------------------------------------------------------ 4. RD parity ctests, by full path, on both builds (a failure is logged, it stops nothing)
rcct=-
if [ -x "$CT" ]; then
  wait_idle; rcct=
  for b in updated plain; do
    say "START ctest-$b $(date +%F_%T) ($CT --test-dir $RDB/build-$b -R '$CT_RX', limit $LIM_CT s)"
    run_group ctest-$b $LIM_CT $O/benchrdgate-ctest-$b.out "$CT" --test-dir $RDB/build-$b -R "$CT_RX" --output-on-failure --timeout $CTEST_TIMEOUT; rc=$?
    say "END ctest-$b exit=$rc $(date +%T)"
    grep -E "tests passed|tests failed|Passed|Failed|Not Run|Could not find executable|No tests were found" $O/benchrdgate-ctest-$b.out | sort -u | cut -c1-200 >> $L
    [ $rc -eq 0 ] || say "ERROR ctest-$b exit=$rc (log $O/benchrdgate-ctest-$b.out; the plain build was built for the strata target only: missing parity binaries show as Not Run); the chain goes on"
    rcct="$rcct$b=$rc "
  done
else say "ERROR ctest SKIPPED: $CT is missing (the parity ctests need it); the chain goes on"; fi

# ------------------------------------------------------------------ 5. RD diag-only of the updated build (flag_diag.py of updated-strata-rdna3, per-flag process groups)
rcdg=-
miss=; for f in $FD $UPD_SRC/tools/pancake/diag_group.py; do [ -e "$f" ] || miss="$miss $f"; done
if [ -n "$miss" ]; then say "ERROR diag SKIPPED: missing$miss (bench-updatedRD.sh syncs the updated-strata-rdna3 tree); the chain goes on"
elif ! left_ok $NEED_DIAG; then say "SKIP diag: budget: $(left) s left (< $NEED_DIAG s: the gate keeps its $NEED_GATE s)"
else
  wait_idle
  if no_engine diag; then
    mkdir -p $DO
    say "START diag $(date +%F_%T) (flag_diag.py run, config $RDU, flags $DIAG_FLAGS, ctest $RDB/build-updated, $FLAG_TIMEOUT s per arm, limit $LIM_DIAG s)"
    run_group diag $LIM_DIAG $O/benchrdgate-diag.stdout sh -c 'cd "$1" && shift && exec "$@"' dg $HN "$HPY" -u $FD run --config $RDU --flags $DIAG_FLAGS \
      --abtest $HN/tools/pancake/abtest.py --out $DO --flag-timeout $FLAG_TIMEOUT --ctest-timeout $CTEST_TIMEOUT --ctest $RDB/build-updated; rcdg=$?
    say "END diag exit=$rcdg $(date +%T)"
    cut -c1-200 $O/benchrdgate-diag.stdout | grep -E "UNTESTED|GAIN|LOSS|NEUTRAL|FAIL|EXACT|LOSSY|UNSUPPORTED|exit=|verdict" >> $L
    [ -f $DO/verdicts.json ] || say "ERROR diag wrote no verdicts.json: $(grep -m1 . $O/benchrdgate-diag.stdout 2>/dev/null | cut -c1-240) (log $O/benchrdgate-diag.stdout); the chain goes on"
    grep -q "another test or engine is running" $O/benchrdgate-diag.stdout && say "ERROR diag: abtest refused to start (another test or engine is running) although the no-engine check passed"
  else rcdg=skip; fi
fi

# ------------------------------------------------------------------ 6. the gate
rcg=-
if [ $rcd -ne 0 ]; then say "SKIP gate: the dry run failed (exit $rcd)"
elif ! left_ok $NEED_GATE; then say "SKIP gate: budget: $(left) s left (< $NEED_GATE s)"
elif [ "$(freekb /)" -lt 307200 ]; then say "ERROR gate: / has < 300 MB free: not run"
else
  wait_idle
  if no_engine gate; then
    say "START gate $(date +%F_%T) (task_gate.py, $TASKS, $REPS reps, temperature 0.2, margin 5 points, base plain run twice; limit $LIM_GATE s, budget left $(left) s)"
    run_group gate $LIM_GATE $O/benchrdgate-gate.stdout sh -c 'cd "$1" && shift && exec "$@"' tg $HN "$HPY" tools/pancake/task_gate.py $T/rdgate.json \
      --out $GO --tasks $TASKS --reps $REPS; rcg=$?
    say "END gate exit=$rcg $(date +%T)"
    [ $rcg -eq 0 ] || say "ERROR gate exit=$rcg: $(errline $O/benchrdgate-gate.stdout $GO)"
    grep -E "PASS|REJECT|INCONCLUSIVE|BETTER|success|validity|run 2 vs run 1|ceiling|died|FAILED" $O/benchrdgate-gate.stdout | cut -c1-220 >> $L
  else rcg=skip; fi
fi

say "SUMMARY dryrun exit=$rcd, ctest ${rcct:-ok}, diag exit=$rcdg, gate exit=$rcg (- = not run); box time $(elapsed) s of $BUDGET s; gate report $GO/REPORT.txt, diag $DO/verdicts.json"
finish
