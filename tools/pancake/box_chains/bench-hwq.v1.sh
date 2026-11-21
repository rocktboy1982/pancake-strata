#!/bin/sh
# bench-hwq.v1.sh - LEADS.md Part S S1: GPU_MAX_HW_QUEUES = 1 and 2 against the preset's 6 (arms base / hwq1 / hwq2 of specs 253 decode and 254 long64k), exact, 5 reps.
# Completion marker (the last line of /home/dan/clone-of-d/src/bench-hwq.v1.log): WINBENCHHWQ_DONE   (also after an abort or a signal; the line before is then WINBENCHHWQ_ABORTED)
# The file name has no "win-chain", "redo-chain" or "tail-chain": it runs ONLY through the scheduler (BENCH_EXTRA slot), which waits for an idle box first;
# its abtest and server children are seen by every other chain's busy() (abtest.py, serve/server.py).
# NOT RUN: written 2026-11-21 on lead/S1-S2 while the flag screen holds the box.  DEPLOY (never edit a running script; tmp file + mv):
#   scp bench-hwq.v1.sh dan@192.168.100.82:/home/dan/clone-of-d/src/.bench-hwq.v1.sh.tmp && ssh dan@192.168.100.82 'mv /home/dan/clone-of-d/src/.bench-hwq.v1.sh.tmp /home/dan/clone-of-d/src/bench-hwq.v1.sh'
# START (only when the screen queue has ended or the owner says so):
#   ssh -n dan@192.168.100.82 'cd /home/dan/clone-of-d/src && BENCH_ONLY=EXTRA BENCH_EXTRA=./bench-hwq.v1.sh BENCH_NO_RESUME=1 setsid -f sh ./bench-sched.sh >/dev/null 2>&1 < /dev/null'
# (BENCH_SCHED=1, BENCH_PREIDLED=1, BENCH_KEEP_SERVICE_DOWN=1 come from the scheduler; the scheduler reads this header's first marker word and logs <script name>.log.)
# What it does (no build, nothing deleted, no sudo): a. waits for idle (quick), b. fetches ONLY specs 253 / 254 from origin (tar xmf of two files; the tree $R is otherwise
#   left as it is), c. spec_lint, d. spec 253 screen then spec 254 screen (abtest --tier screen --drift-check), e. a confirm (3 blocks, 5 reps) only for an arm >= 2% on a case
#   of its screen (chainT-decide.py), f. with ABTEST_QUEUEDIAG=1 the S2 queue-layout lines (STARTQUEUE / STOPQUEUE) land in the stdout files; read them with
#   python3 tools/pancake/box_chains/startdiag_report.py /home/dan/auto-runs/benchhwq/*.stdout
# Needs: $R/build-hip/strata (the preset exe; the chain stops with ERROR naming it when missing) and the model files of the preset.
S=/home/dan/clone-of-d/src; R=$S/Strata-win; L=$S/bench-hwq.v1.log; O=/home/dan/auto-runs/benchhwq; T=$O/work; PM=/mnt/pancake/benchhwq
DEC=$S/chainT-decide.py; REPO_URL=https://github.com/rocktboy1982/pancake-strata.git; HWQ_REF=${HWQ_REF:-port/rdna3-decode}
DATA=/home/dan/clone-of-d/Strata-data; ROCM=/opt/rocm/core-10.0; T0=$(date +%s); BUDGET=${BENCH_BUDGET_S:-9000}
LIM_SCREEN=${HWQ_LIM_SCREEN:-3600}; LIM_CONFIRM=${HWQ_LIM_CONFIRM:-3600}
HPY=${BENCH_HARNESS_PY:-$R/.venv/bin/python}
export PATH=$ROCM/bin:$R/.venv/bin:$PATH LD_LIBRARY_PATH=$ROCM/lib HIP_VISIBLE_DEVICES=0 ABTEST_PYTHON=$HPY
export ABTEST_SETTLE_AFTER_LONG_S=${ABTEST_SETTLE_AFTER_LONG_S:-120} ABTEST_READY_MAX_S=${ABTEST_READY_MAX_S:-900} ABTEST_PREREAD=${ABTEST_PREREAD:-2}
mkdir -p $O $T $PM
echo "START $(date +%F_%T) budget $BUDGET s, ref $HWQ_REF" >> $L
say() { echo "$*" >> $L; }
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[c]onv_trace.py|[a]uto-chain|[a]uto-post|[i]q_avx2_parity|[t]tft_bench.py|[w]in-chain[A-Za-z0-9]*\.sh|[x-z]tail-chain[A-Z]\.sh|[a]fter7|[b]ench-(updated(AA|RD)|diagUS|exact|rdgate|sweep)[.a-z0-9]*\.sh" >/dev/null 2>&1; }
aa_live() { for p in $(pgrep -f "[r]edo-chainAA\.sh" 2>/dev/null); do case "$(ps -o stat= -p $p 2>/dev/null)" in T*) ;; *) return 0;; esac; done; return 1; }   # a FROZEN AA shell is not a job
bz() { busy || aa_live; }
wait_idle() { q=0; while [ $q -lt 2 ]; do if bz; then q=0; sleep 15; else q=$((q+1)); sleep 15; fi; done; }   # quick idle: the scheduler waited already
elapsed() { echo $(( $(date +%s) - T0 )); }
left() { echo $(( BUDGET - $(elapsed) )); }
freekb() { df -Pk "$1" | awk 'NR==2{print $4}'; }
finish() { say "service stays down (owner order: only the benchmarks run) / free $(freekb /) KB, box time $(elapsed) s"; echo WINBENCHHWQ_DONE >> $L; }
abort() { say "ERROR $*"; say "WINBENCHHWQ_ABORTED: $*"; finish; exit 1; }
CUR=
on_signal() { say "signal received $(date +%T)"; [ -n "$CUR" ] && kill -TERM $CUR 2>/dev/null; say "WINBENCHHWQ_ABORTED: signal"; finish; exit 1; }
trap on_signal HUP INT TERM

# ------------------------------------------------------------------ 0. prerequisites (the service is stopped only after 3 minutes without a client)
clients() { ss -Htn state established '( sport = :11437 )' 2>/dev/null | wc -l; }
if systemctl --user is-active --quiet pancake.service; then
  q=0; while [ $q -lt 3 ]; do if [ "$(clients)" -gt 0 ]; then q=0; say "WAIT service has clients $(date +%T)"; else q=$((q+1)); fi; sleep 60; done
  systemctl --user stop pancake.service; say "service stopped (idle) $(date +%T)"
fi
wait_idle
[ "$(freekb /)" -ge 307200 ] || abort "START: / has $(freekb /) KB free (< 300 MB)"
miss=; for f in $R/build-hip/strata $R/data/expert-profile-pancake.bin $R/tools/pancake/abtest.py $HPY $DEC \
   $DATA/packs/iq2_xs/tokenizer $DATA/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf $DATA/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf; do
  [ -e "$f" ] || miss="$miss $f"; done
[ -z "$miss" ] || abort "PREREQ missing files:$miss"

# ------------------------------------------------------------------ 1. the two specs only (tar xmf: mtimes reset); the tree is otherwise left as it is
SHA=$(timeout 60 git ls-remote $REPO_URL refs/heads/$HWQ_REF 2>/dev/null | cut -f1)
TGZ=$PM/src.tgz; rm -f $TGZ.part
if [ -n "$SHA" ] && timeout 1200 curl -sfL -o $TGZ.part https://codeload.github.com/rocktboy1982/pancake-strata/tar.gz/$SHA && tar tzf $TGZ.part >/dev/null 2>&1; then
  mv $TGZ.part $TGZ; say "SYNC $HWQ_REF $SHA"
else abort "SYNC download of $HWQ_REF failed"; fi
tar xzmf $TGZ --no-same-owner --strip-components=1 -C $R --wildcards '*/bench/specs/253-hwq-decode.json' '*/bench/specs/254-hwq-long64k.json' || abort "SYNC tar failed"
cd $R || abort "no $R"
for sp in 253-hwq-decode 254-hwq-long64k; do [ -f bench/specs/$sp.json ] || abort "SYNC: bench/specs/$sp.json missing in $HWQ_REF"; done
python3 tools/pancake/spec_lint.py --dir bench/specs --quiet > $T/spec_lint.txt 2>&1; say "HEADER spec_lint exit=$?"
grep -Eq "^[A-Za-z-]+ +25[34]-" $T/spec_lint.txt && abort "spec_lint flags spec 253 / 254: see $T/spec_lint.txt"
if grep -q ABTEST_QUEUEDIAG tools/pancake/abtest.py; then export ABTEST_QUEUEDIAG=1; say "HEADER ABTEST_QUEUEDIAG=1 (S2 queue layout lines; hqds needs root: shared=NA without it)"
else say "HEADER the tree's abtest.py has no ABTEST_QUEUEDIAG: the S2 lines are absent"; fi

# ------------------------------------------------------------------ 2. screens, then a confirm for an arm that earned it
ab() {  # ab NAME LIMIT SPEC flags...
  nm=$1; lim=$2; sp=$3; shift 3
  [ "$(freekb /)" -ge 307200 ] || { say "ERROR $nm: / has < 300 MB free: not run"; return 9; }
  [ "$(left)" -ge 600 ] || { say "SKIP $nm: budget $(left) s left"; return 9; }
  wait_idle; say "START $nm $(date +%F_%T) (limit $lim s, budget left $(left) s)"
  timeout -k 100 $lim "$HPY" tools/pancake/abtest.py $sp "$@" --out $O/benchhwq-$nm.jsonl > $O/benchhwq-$nm.stdout 2>&1 & CUR=$!
  wait $CUR; rc=$?; CUR=
  say "END $nm exit=$rc $(date +%T)"
  [ $rc -eq 0 ] || say "ERROR $nm exit=$rc: $(grep -i -m1 -E 'error|fatal|died|failed|cannot' $O/benchhwq-$nm.stdout | cut -c1-240)"
  grep -E "vs .*(better|worse|no difference)|verdict|DRIFT|ORDER EFFECT|STARTQUEUE|STOPQUEUE" $O/benchhwq-$nm.stdout | cut -c1-220 >> $L
  return $rc
}
for part in 253-hwq-decode:dec 254-hwq-long64k:l64; do
  sp=bench/specs/${part%%:*}.json; pf=${part##*:}
  ab $pf-screen $LIM_SCREEN $sp --tier screen --drift-check; rc=$?
  if [ $rc -eq 0 ]; then
    rm -f $T/$pf-confirm.json
    python3 $DEC $O/benchhwq-$pf-screen.stdout $sp $T/$pf-confirm.json > $T/$pf-decide.txt 2>&1
    grep -E "WIN|CONFIRM|NONE|dropped" $T/$pf-decide.txt | cut -c1-230 >> $L
    if [ -s $T/$pf-confirm.json ]; then
      ab $pf-confirm $LIM_CONFIRM $T/$pf-confirm.json --tier confirm --block 3 --warmup 1 --outlier robust --reclaim-max 3 --reps 5
    else say "$pf: no arm with a benefit in the screen: no confirm (a loss on the prefill side is the screen's result)"; fi
  fi
done
say "SUMMARY box time $(elapsed) s of $BUDGET s; results in $O; queue layout: python3 tools/pancake/box_chains/startdiag_report.py $O/*.stdout"
finish
