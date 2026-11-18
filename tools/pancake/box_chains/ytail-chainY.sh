#!/bin/sh
# Tail chain Y (owner order 2026-11-18): the unused-victim / direct-swap idea for the adaptive VRAM tier (LEADS.md B145 / B146, MOD_PLAN.md section 6 item 59, TEST_LEDGER.md T-951..T-954):
# STRATA_ADAPT_VICTIM_UNUSED (the victim must have had no use in the last K tier rounds, never the previous round's) and STRATA_ADAPT_DIRECT_SWAP (the swap rewrites the victim's slot with one copy and
# skips the victim's host-page readback).  Runs after EVERY other queued job: the queue is T -> U -> V -> after7.sh -> X (after7b.sh is a read-only disk probe that follows after7.sh), so Y waits for the markers
# WINCHAINV_DONE (win-chainV.log), AFTER7_DONE (after7.log) and WINCHAINX_DONE (xtail-chainX.log) and must not overlap X.  The file name has no "win-chain": chain V's busy() ([A-UW-Z]) and chain X's
# busy() ([A-Za-z0-9]*) both match any win-chain<letter>.sh, so a waiter named win-chainY.sh would deadlock V and X (TEST_LEDGER.md T-944); after7.sh matches [w]in-chain as well.
# TWO-PASS RULE (owner order 2026-11-18): pass 1 = the spec-214 SCREEN, every arm once (3 reps, drift check); pass 2 = the spec-215 CONFIRM (5 reps) only for the arms >= 1 % faster on a case and not below -3 % on
# any (chainT-decide.py PASS2_BAR); every other arm is dropped after pass 1 and logged so.  Offline first (CPU only, cheap): the routing replay of the victim rules on every recording the box has (T-950 family).
# Order: 0 wait + stop service + sync origin (tar xmf: mtimes reset) + header | A build strata + tier_victim_test | B ctest tier_victim_test | C replays (CPU) | D screen 214 -> decide -> confirm 215.
# Completion marker: the last line of /home/dan/clone-of-d/src/ytail-chainY.log is WINCHAINY_DONE.
S=/home/dan/clone-of-d/src; R=$S/Strata-win; L=$S/ytail-chainY.log; O=/home/dan/auto-runs; T=$O/chainY; PM=/mnt/pancake/chainY
CM=/home/linuxbrew/.linuxbrew/bin/cmake; CT=/home/linuxbrew/.linuxbrew/bin/ctest
DEC=$S/chainT-decide.py; B=$S/build-win; PROF=data/expert-profile-pancake.bin
REPO_URL=https://github.com/rocktboy1982/pancake-strata.git
export PATH=$R/.venv/bin:$PATH LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib HIP_VISIBLE_DEVICES=0
mkdir -p $T $PM
echo "START $(date +%F_%T)" >> $L
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[c]onv_trace.py|[a]uto-chain|[a]uto-post|[i]q_avx2_parity|[t]tft_bench.py|[w]in-chain[A-X0-9][A-Za-z0-9]*\.sh" >/dev/null; }
wait_idle() { if [ "${QI:-0}" = 1 ]; then q=0; while [ $q -lt 2 ]; do if busy; then q=0; sleep 15; else q=$((q+1)); sleep 15; fi; done; else q=0; while [ $q -lt 3 ]; do if busy; then q=0; sleep 60; else q=$((q+1)); sleep 60; fi; done; QI=1; fi; }   # quick idle (owner order 2026-11-18, box hours): the full 3 minutes only the first time, 30 s between this chain's own later steps; busy() still sees every other job
clients() { ss -Htn state established '( sport = :11437 )' 2>/dev/null | wc -l; }
freekb() { df -Pk "$1" | awk 'NR==2{print $4}'; }
say() { echo "$*" >> $L; }
lowdisk() { [ "$(freekb $S)" -lt 307200 ]; }   # < 300 MB on / : no engine step starts

until grep -q WINCHAINV_DONE $S/win-chainV.log 2>/dev/null && grep -q AFTER7_DONE $O/after7.log 2>/dev/null && grep -q WINCHAINX_DONE $S/xtail-chainX.log 2>/dev/null; do sleep 300; done
say "chains V, after7 and X done $(date +%T)"
q=0; while [ $q -lt 3 ]; do if [ "$(clients)" -gt 0 ]; then q=0; say "WAIT service has clients $(date +%T)"; else q=$((q+1)); fi; sleep 60; done
systemctl --user stop pancake.service; say "service stopped (idle) $(date +%T)"
cd $R; echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null

# ------------------------------------------------------------------ 0. sync the tree to origin HEAD (tar xmf: mtimes reset, else ninja keeps stale objects)
wait_idle
say "SYNC start $(date +%T) df / = $(freekb /) KB, /mnt/pancake = $(freekb /mnt/pancake) KB"
SHA=$(timeout 60 git ls-remote $REPO_URL refs/heads/port/rdna3-decode 2>/dev/null | cut -f1)
SRC=$PM/src.tgz; rm -f $SRC.part
if [ -n "$SHA" ] && timeout 1200 curl -sfL -o $SRC.part https://codeload.github.com/rocktboy1982/pancake-strata/tar.gz/$SHA && tar tzf $SRC.part >/dev/null 2>&1; then
  mv $SRC.part $SRC; say "SYNC source: origin port/rdna3-decode $SHA (codeload)"
else
  SRC=; say "SYNC source: download failed (no fallback: an old tree lacks the victim switches)"
fi
if [ -n "$SRC" ] && [ -s "$SRC" ] && [ "$(freekb /)" -gt 1000000 ] && tar tzf $SRC >/dev/null 2>&1; then
  tar xzmf $SRC --no-same-owner --strip-components=1 -C $R; rc=$?; say "SYNC exit=$rc ($(tar tzf $SRC | wc -l) entries) $(date +%T)"
else
  say "SYNC SKIPPED: no usable source or / has < 1 GB free ($(freekb /) KB)"
fi
HAVE_CODE=0
[ -f include/strata/program/tier_victim.hpp ] && grep -q STRATA_ADAPT_VICTIM_UNUSED src/core/conversation_disk_env.def 2>/dev/null && HAVE_CODE=1
HAVE_TOOL=0
grep -q -- "--victim" tools/pancake/replay_fill_transition.py 2>/dev/null && HAVE_TOOL=1
say "HEADER tier_victim.hpp + env table entries present: $HAVE_CODE; replay tool with --victim: $HAVE_TOOL"
ls $R/bench/specs/214-*.json $R/bench/specs/215-*.json >/dev/null 2>&1 || say "HEADER specs 214 / 215 missing"
python3 tools/pancake/spec_lint.py --dir bench/specs --quiet > $T/spec_lint.txt 2>&1; say "HEADER spec_lint exit=$?"
lint_bad() { grep -Eq "^[A-Za-z-]+ +$1-" $T/spec_lint.txt; }
spec_ok() { ls $R/bench/specs/$1-*.json >/dev/null 2>&1 && ! lint_bad $1; }
python3 tools/pancake/test_disk_env_table.py > $T/disk_env_table.txt 2>&1; say "HEADER disk_env_table exit=$?"
python3 -m pytest -q tools/pancake/test_replay_fill_transition.py > $T/pytest.txt 2>&1; say "HEADER pytest replay_fill_transition exit=$? $(tail -1 $T/pytest.txt | cut -c1-120)"

# ------------------------------------------------------------------ helpers
ab() {  # ab SPECFILE PREFIX flags... : abtest with the chain-N retry on 'already running'
  sp=$1; pfx=$2; shift 2; tries=0
  while :; do
    if lowdisk; then say "SKIP $pfx: / has < 300 MB free"; return 9; fi
    wait_idle; say "START $pfx $(date +%F_%T)"
    python3 tools/pancake/abtest.py $sp "$@" --out $O/winY-$pfx.jsonl > $O/winY-$pfx.stdout 2>&1; rc=$?
    say "END $pfx exit=$rc $(date +%T)"
    if [ $rc -eq 3 ] && grep -q "already running" $O/winY-$pfx.stdout && [ $tries -lt 12 ]; then tries=$((tries+1)); sleep 300; continue; fi
    return $rc
  done
}
proof() {  # proof DIR ARMPREFIX GOODPATTERN: 0 if every engine log of the arm has the good line
  d=$1; a=$2; g=$3; ok=0; n=0
  for f in $d/$a*-r*-a*.engine.log; do [ -f "$f" ] || continue; n=$((n+1))
    grep -q -- "$g" $f || { say "PROOF $a: '$g' missing in $(basename $f)"; ok=1; }
  done
  [ $n -eq 0 ] && { say "PROOF $a: no engine log under $d"; return 1; }
  [ $ok -eq 0 ] && say "PROOF $a: ok ($n logs, '$g')"; return $ok
}

# ------------------------------------------------------------------ A. build build-win (incremental): the engine and the new unit test
ENGINE=0
if [ $HAVE_CODE -ne 1 ]; then
  say "BUILD SKIPPED: the synced tree has no tier_victim.hpp (origin lacks the commit); the engine steps below are skipped"
elif [ "$(freekb /)" -lt 2097152 ]; then
  say "BUILD SKIPPED: / has $(freekb /) KB free (< 2 GB); nothing was deleted; every engine step below is skipped (offline steps still run)"
else
  wait_idle; say "BUILD build-win strata tier_victim_test $(date +%T) (/ free $(freekb /) KB)"
  nice -n 10 $CM --build $B -j8 --target strata tier_victim_test > $S/build-win.chainY.log 2>&1; rc=$?
  say "BUILD_END build-win exit=$rc $(date +%T) (/ free $(freekb /) KB)"
  [ $rc -ne 0 ] && grep -m8 -E "error:|undefined|CMake Error" $S/build-win.chainY.log >> $L
  [ $rc -eq 0 ] && ENGINE=1
fi

# ------------------------------------------------------------------ B. ctest of the new unit test (a failure is logged, the chain continues)
if [ $ENGINE -eq 1 ]; then
  cd $B
  wait_idle; say "START ctest tier_victim $(date +%T)"
  $CT -R 'tier_victim_test|tier_stat_policy_test|tier_protect_test' --timeout 600 --output-on-failure > $T/ctest-victim.log 2>&1
  say "END ctest tier_victim exit=$? $(grep -E 'tests passed|tests failed' $T/ctest-victim.log)"
  grep -E "tier_victim_test: victim-unused" $T/ctest-victim.log | head -2 | cut -c1-200 >> $L
  cd $R
fi

# ------------------------------------------------------------------ C. offline replays (CPU only): the victim rules on every routing recording the box has
rep() {  # rep NAME TRACE SLOTS [more flags]
  nm=$1; tr=$2; sl=$3; shift 3
  [ -s "$tr" ] || { say "SKIP replay $nm: no trace $tr"; return 9; }
  [ $HAVE_TOOL -eq 1 ] || { say "SKIP replay $nm: no --victim in the replay tool"; return 9; }
  wait_idle; say "START replay $nm $(date +%T)"
  python3 tools/pancake/replay_fill_transition.py $tr $PROF $sl --victim --probe --json $O/winY-replay-$nm.json "$@" > $O/winY-replay-$nm.txt 2>&1; rc=$?
  say "END replay $nm exit=$rc $(date +%T)"
  head -1 $O/winY-replay-$nm.txt | cut -c1-200 >> $L
  grep -E "^(base|vu:1|vu:2|vu:4|vu:8|vus:1|vus:4|ff:0.7\+vu:4|ff:0.7\+tr:2.0\+vu:4) " $O/winY-replay-$nm.txt | head -12 | cut -c1-110 >> $L
  return $rc
}
if [ $HAVE_TOOL -eq 1 ]; then
  rep routing-main bench/prefetch/routing.bin 10909
  rep routing-slots5000 bench/prefetch/routing.bin 5000
  rep routing-slots3000 bench/prefetch/routing.bin 3000
  W16=$O/win16-conv; WV=$O/winV-conv
  for t in real4 real5; do
    if [ -s $WV/$t.trace ] && [ -s $WV/$t.requests.json ]; then
      stw=""; [ -s $WV/$t.stwt ] && stw="--stwt $WV/$t.stwt"
      rep $t-main $WV/$t.trace 10909 --requests $WV/$t.requests.json $stw
      rep $t-slots5000 $WV/$t.trace 5000 --requests $WV/$t.requests.json $stw
      rep $t-slots3000 $WV/$t.trace 3000 --requests $WV/$t.requests.json $stw
    else say "SKIP replays of $t: chain V left no trace / sidecar in $WV"; fi
  done
  rep real1-chain16a-main $W16/real1.trace 10909 --requests $W16/real1.requests.json
  rep real1-chain16a-slots3000 $W16/real1.trace 3000 --requests $W16/real1.requests.json
fi

# ------------------------------------------------------------------ D. spec 214 screen (pass 1: every arm once), decision rule, spec 215 confirm (pass 2: only the arms that earned it)
if [ $ENGINE -eq 1 ] && spec_ok 214; then
  cd $R
  SPEC=$(ls $R/bench/specs/214-*.json | head -1); SIB=$(ls $R/bench/specs/215-*.json 2>/dev/null | head -1)
  ab $SPEC 214-screen --tier screen --drift-check
  PROOF_FAIL=0
  proof $O/winY-214-screen unused "STRATA_ADAPT_VICTIM_UNUSED=1:" || PROOF_FAIL=1
  proof $O/winY-214-screen direct "STRATA_ADAPT_DIRECT_SWAP=1:" || PROOF_FAIL=1
  grep -h "victim unused:" $O/winY-214-screen/*.engine.log 2>/dev/null | sort | uniq -c | sort -rn | head -6 | cut -c1-230 >> $L
  rm -f $T/214-confirm.json
  extra=""; [ -n "$SIB" ] && extra="--confirm-spec $SIB"
  python3 $DEC $O/winY-214-screen.stdout $SPEC $T/214-confirm.json $extra > $T/214-decide.txt 2>&1
  grep -E "WIN|CONFIRM|NONE|dropped" $T/214-decide.txt | cut -c1-230 >> $L
  if [ -s $T/214-confirm.json ] && [ $PROOF_FAIL -eq 0 ]; then
    ab $T/214-confirm.json 214-confirm --tier confirm --block 3 --warmup 1 --outlier robust --reclaim-max 3 --reps 5
    grep -E "vs .*(better|worse|no difference)" $O/winY-214-confirm.stdout | cut -c1-200 >> $L
  elif [ -s $T/214-confirm.json ]; then say "confirm 214 skipped: a proof line was missing (an arm is the base)"; fi
else
  say "SKIP spec 214: no engine build, or the spec is missing or flagged by spec_lint"
fi
systemctl --user start pancake.service; say "service started $(date +%T); disk / free $(freekb /) KB"
echo WINCHAINY_DONE >> $L
