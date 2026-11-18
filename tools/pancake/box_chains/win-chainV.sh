#!/bin/sh
# Windows-session chain V (owner approval 2026-11-18, autonomous): a MULTI-CONVERSATION routing dump WITH real router weights, and the
# offline replay of STRATA_ADAPT_TRANSITION's conversation-switch term and of the real STRATA_ADAPT_PROB_MASS on it
# (MOD_PLAN.md section 5 chain V; TEST_LEDGER.md T-929..T-932 saw only one single-stream trace with a zero weight column).
# FAST VERSION (owner order 2026-11-18; the old script is win-chainV.sh.bak): only the quick idle between its own steps changed.
# Runs after chain U (the queue: N -> G -> R -> W2 -> H2 -> S -> T -> U -> V).  Chains N..U are not touched.
# Rules (standing): box jobs never overlap (this script waits for WINCHAINU_DONE in win-chainU.log, then 3 idle minutes, and every step waits
# 3 idle minutes); nothing is killed or deleted (only this script's own conv_trace has a timeout); pancake.service is stopped only after 3 minutes
# without a client on port 11437 and started again at the end; the sudoers file and permissions are never touched; code is synced with
# `tar xmf` semantics (mtimes reset) so ninja rebuilds; the root disk is nearly full, so the build checks `df` first and is SKIPPED (logged,
# nothing deleted) when less than 2 GB is free.
# Weights: the served --dump-routing weight column is unit or zero (T-460, T-776, T-839); the router weights come from the window trace
# with STRATA_WINDOW_TRACE_WEIGHTS=1 (MOD_PLAN.md section 6 item 20 route (b), chain W2 / T-841), recorded in the same run as the dump,
# and tools/pancake/replay_fill_transition.py --stwt matches the two by ids.  Chain W2's own result decides nothing here: if the
# weights come out all zero again, the replay says 'pmw NOT available' and the switch-term arms (no weights needed) still run.
# Order: 0 sync + header | A build (incremental, strata) | B record real4 (4 conversations interleaved A B A C B D A C D: 8 switches, A continues
#        3 turns) and real5 (one after the other A A A B B C C D D: 3 switches, a continuing 3-turn conversation) | C replay (CPU only):
#        real4, real5, and the existing chain-16a dump real1 (9 turns, 4 conversations, unit weights) | D copy the small outputs
#        (no trace / stwt) into bench/results/2026-11-18-chainV/ of the box tree.  Log: win-chainV.log (START / END lines with exit codes);
#        output: /home/dan/auto-runs/chainV/ and winV-*.
# Completion marker: the last line of win-chainV.log is WINCHAINV_DONE.
S=/home/dan/clone-of-d/src; R=$S/Strata-win; L=$S/win-chainV.log
O=/home/dan/auto-runs; T=$O/chainV; PM=/mnt/pancake/chainV; C=$O/winV-conv
CM=/home/linuxbrew/.linuxbrew/bin/cmake; B=$S/build-win
REPO_URL=https://github.com/rocktboy1982/pancake-strata.git
PROF=data/expert-profile-pancake.bin
export PATH=$R/.venv/bin:$PATH LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib HIP_VISIBLE_DEVICES=0
mkdir -p $T $PM $C
echo "START $(date +%F_%T)" >> $L
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[c]onv_trace.py|[a]uto-chain|[a]uto-post|[i]q_avx2_parity|[t]tft_bench.py|[w]in-chain[A-UW-Za-z0-9]*\.sh" >/dev/null 2>&1; }
wait_idle() { if [ "${QI:-0}" = 1 ]; then q=0; while [ $q -lt 2 ]; do if busy; then q=0; sleep 15; else q=$((q+1)); sleep 15; fi; done; else q=0; while [ $q -lt 3 ]; do if busy; then q=0; sleep 60; else q=$((q+1)); sleep 60; fi; done; QI=1; fi; }   # quick idle (owner order 2026-11-18, box hours): the full 3 minutes only the first time, 30 s between this chain's own later steps; busy() still sees every other job
clients() { ss -Htn state established '( sport = :11437 )' 2>/dev/null | wc -l; }
freekb() { df -Pk "$1" | awk 'NR==2{print $4}'; }
say() { echo "$*" >> $L; }
lowdisk() { [ "$(freekb $S)" -lt 307200 ]; }   # < 300 MB on / : no engine step starts

until grep -q WINCHAINU_DONE $S/win-chainU.log 2>/dev/null; do sleep 300; done
say "chain U done $(date +%T)"
q=0; while [ $q -lt 3 ]; do if [ "$(clients)" -gt 0 ]; then q=0; say "WAIT service has clients $(date +%T)"; else q=$((q+1)); fi; sleep 60; done
systemctl --user stop pancake.service; say "service stopped (idle) $(date +%T)"
cd $R; echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null

# ------------------------------------------------------------------ 0. sync the tree to origin HEAD (tar xmf: mtimes reset)
wait_idle
say "SYNC start $(date +%T) df / = $(freekb /) KB, /mnt/pancake = $(freekb /mnt/pancake) KB"
SHA=$(timeout 60 git ls-remote $REPO_URL refs/heads/port/rdna3-decode 2>/dev/null | cut -f1)
SRC=$PM/src.tgz; rm -f $SRC.part
if [ -n "$SHA" ] && timeout 1200 curl -sfL -o $SRC.part https://codeload.github.com/rocktboy1982/pancake-strata/tar.gz/$SHA && tar tzf $SRC.part >/dev/null 2>&1; then
  mv $SRC.part $SRC; say "SYNC source: origin port/rdna3-decode $SHA (codeload)"
else
  SRC=; say "SYNC source: download failed (no fallback: an old tree lacks the replay options)"
fi
if [ -n "$SRC" ] && [ -s "$SRC" ] && [ "$(freekb /)" -gt 1000000 ] && tar tzf $SRC >/dev/null 2>&1; then
  tar xzmf $SRC --no-same-owner --strip-components=1 -C $R; rc=$?; say "SYNC exit=$rc ($(tar tzf $SRC | wc -l) entries) $(date +%T)"
else
  say "SYNC SKIPPED: no usable source or / has < 1 GB free ($(freekb /) KB)"
fi
HAVE_TOOL=0
grep -q -- "--requests" tools/pancake/replay_fill_transition.py 2>/dev/null && grep -q -- "--order" tools/pancake/conv_trace.py 2>/dev/null && HAVE_TOOL=1
say "HEADER replay tool with --requests/--stwt and conv_trace --order present: $HAVE_TOOL"
python3 tools/pancake/test_disk_env_table.py > $T/disk_env_table.txt 2>&1; say "HEADER disk_env_table exit=$?"
python3 -m pytest -q tools/pancake/test_replay_fill_transition.py > $T/pytest.txt 2>&1; say "HEADER pytest replay_fill_transition exit=$? $(tail -1 $T/pytest.txt | cut -c1-120)"

# ------------------------------------------------------------------ A. build build-win (incremental)
ENGINE=0
if [ "$(freekb /)" -lt 2097152 ]; then
  say "BUILD SKIPPED: / has $(freekb /) KB free (< 2 GB); nothing was deleted; the recordings below are skipped"
else
  wait_idle; say "BUILD build-win strata $(date +%T) (/ free $(freekb /) KB)"
  nice -n 10 $CM --build $B -j8 --target strata > $S/build-win.chainV.log 2>&1; rc=$?
  say "BUILD_END build-win exit=$rc $(date +%T) (/ free $(freekb /) KB)"
  [ $rc -ne 0 ] && grep -m8 -E "error:|undefined|CMake Error" $S/build-win.chainV.log >> $L
  [ $rc -eq 0 ] && ENGINE=1
fi
say "HEADER STRATA_WINDOW_TRACE_WEIGHTS strings in build-win/strata: $(grep -c STRATA_WINDOW_TRACE_WEIGHTS $B/strata)"

# ------------------------------------------------------------------ B. record: interleaved (real4) and one-after-the-other (real5)
rec() {  # rec TAG [conv_trace flags]: dump + window trace with weights + sidecar
  tag=$1; shift
  if lowdisk; then say "SKIP record $tag: / has < 300 MB free"; return 9; fi
  wait_idle; say "START record $tag $(date +%F_%T)"
  timeout 5400 python3 tools/pancake/conv_trace.py $C/config.json $C --max-new 384 --tag $tag \
      --window-trace $C/$tag.stwt --engine-env STRATA_WINDOW_TRACE_WEIGHTS=1 "$@" > $C/$tag.stdout 2>&1; rc=$?
  say "END record $tag exit=$rc $(date +%T) ($(wc -l < $C/$tag.requests.json 2>/dev/null) lines of sidecar, trace $(stat -c %s $C/$tag.trace 2>/dev/null) bytes, stwt $(stat -c %s $C/$tag.stwt 2>/dev/null) bytes)"
  grep "STRATA_WINDOW_TRACE" $C/$tag.engine.log 2>/dev/null | head -3 | cut -c1-200 >> $L
  return $rc
}
if [ $ENGINE -eq 1 ] && [ $HAVE_TOOL -eq 1 ]; then
  sed 's#/Strata/build-hip/strata#/build-win/strata#' $S/Strata/pancake-flashnext-iq2_xs-fast-hip.json > $C/config.json
  rec real4
  rec real5 --order AAABBCCDD
else
  say "SKIP record: no engine build or no new tools in the tree"
fi

# ------------------------------------------------------------------ C. replay (CPU only; the dump has the conversations, the window trace the weights)
rep() {  # rep NAME TRACE REQUESTS STWT-or-none SLOTS [more flags]
  nm=$1; tr=$2; rq=$3; st=$4; sl=$5; shift 5
  [ -s "$tr" ] && [ -s "$rq" ] || { say "SKIP replay $nm: no trace or sidecar"; return 9; }
  stw=""; [ "$st" != none ] && [ -s "$st" ] && stw="--stwt $st"
  wait_idle; say "START replay $nm $(date +%T)"
  python3 tools/pancake/replay_fill_transition.py $tr $PROF $sl --requests $rq $stw --json $O/winV-replay-$nm.json "$@" > $O/winV-replay-$nm.txt 2>&1; rc=$?
  say "END replay $nm exit=$rc $(date +%T)"
  head -1 $O/winV-replay-$nm.txt | cut -c1-330 >> $L
  grep -E "^(base|pm|pmw|ff:0.7|tr:2.0|trb:2.0|ff:0.7\+tr:2.0|ff:0.7\+pmw|tr:2.0\+pmw|trb:2.0\+pmw) " $O/winV-replay-$nm.txt | cut -c1-90 >> $L
  return $rc
}
if [ $HAVE_TOOL -eq 1 ]; then
  for t in real4 real5; do
    rep $t-main $C/$t.trace $C/$t.requests.json $C/$t.stwt 10909 --sweep
    rep $t-every1 $C/$t.trace $C/$t.requests.json $C/$t.stwt 10909 --every 1
    rep $t-slots5000 $C/$t.trace $C/$t.requests.json $C/$t.stwt 5000
  done
  W16=$O/win16-conv
  rep real1-chain16a $W16/real1.trace $W16/real1.requests.json none 10909 --sweep
fi

# ------------------------------------------------------------------ D. keep the small outputs in the box tree (no trace / stwt: too big)
RD=$R/bench/results/2026-11-18-chainV; mkdir -p $RD
for f in $O/winV-replay-*.txt $O/winV-replay-*.json $C/real4.requests.json $C/real5.requests.json $C/real4.stdout $C/real5.stdout \
         $C/real4.engine.log $C/real5.engine.log $T/pytest.txt $T/disk_env_table.txt; do
  [ -f "$f" ] && cp -n "$f" $RD/ 2>/dev/null
done
say "COPY small outputs to $RD ($(ls $RD | wc -l) files); big files stay: $C/real4.trace real4.stwt real5.trace real5.stwt"

systemctl --user start pancake.service; say "service started $(date +%T); disk / free $(freekb /) KB"
echo WINCHAINV_DONE >> $L
