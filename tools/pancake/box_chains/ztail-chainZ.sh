#!/bin/sh
# Tail chain Z (owner order 2026-11-18): the speculative early copy of the adaptive tier's PREDICTED incoming experts (LEADS.md B147, MOD_PLAN.md section 6 item 60, TEST_LEDGER.md T-955..T-960):
# STRATA_ADAPT_SPEC_COPY (+ _SLOTS, _MAX, _CONF, _AGE): at the end of a round the CPU predicts the next round's incoming experts and copies them into a bounded spare VRAM staging area on a second stream; a swap of a staged expert is a
# device copy, the others take the normal host copy.  Runs after EVERY other queued job: the queue is T -> U -> V -> after7.sh -> X -> Y, so Z waits for the markers WINCHAINV_DONE (win-chainV.log), AFTER7_DONE (after7.log),
# WINCHAINX_DONE (xtail-chainX.log) and WINCHAINY_DONE (ytail-chainY.log) and must not overlap any of them.  The file name has no "win-chain" (chain V's busy() ([A-UW-Z]) and chain X's busy() ([A-Za-z0-9]*) match any
# win-chain<letter>.sh, so a waiter named win-chainZ.sh would deadlock V and X, TEST_LEDGER.md T-944); it is named like chain Y's.
# TWO-PASS RULE (owner order 2026-11-18): pass 1 = the spec-216 SCREEN, every arm once (3 reps, drift check); pass 2 = the spec-217 CONFIRM (5 reps) only for the arms >= 1 % faster on a case and not below -3 % on any
# (chainT-decide.py PASS2_BAR); every other arm is dropped after pass 1 and logged so.  Offline first (CPU only, cheap): the routing replay of the staging predictors on every recording the box has (T-955 family) and the unit test.
# Order: 0 wait + stop service + sync origin (tar xmf: mtimes reset) + header | A build strata + tier_spec_copy_test | B ctest | C replays (CPU) | D screen 216 -> decide -> confirm 217.
# Completion marker: the last line of /home/dan/clone-of-d/src/ztail-chainZ.log is WINCHAINZ_DONE.
S=/home/dan/clone-of-d/src; R=$S/Strata-win; L=$S/ztail-chainZ.log; O=/home/dan/auto-runs; T=$O/chainZ; PM=/mnt/pancake/chainZ
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

until grep -q WINCHAINV_DONE $S/win-chainV.log 2>/dev/null && grep -q AFTER7_DONE $O/after7.log 2>/dev/null && grep -q WINCHAINX_DONE $S/xtail-chainX.log 2>/dev/null && grep -q WINCHAINY_DONE $S/ytail-chainY.log 2>/dev/null; do sleep 300; done
say "chains V, after7, X and Y done $(date +%T)"
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
  SRC=; say "SYNC source: download failed (no fallback: an old tree lacks the spec-copy switch)"
fi
if [ -n "$SRC" ] && [ -s "$SRC" ] && [ "$(freekb /)" -gt 1000000 ] && tar tzf $SRC >/dev/null 2>&1; then
  tar xzmf $SRC --no-same-owner --strip-components=1 -C $R; rc=$?; say "SYNC exit=$rc ($(tar tzf $SRC | wc -l) entries) $(date +%T)"
else
  say "SYNC SKIPPED: no usable source or / has < 1 GB free ($(freekb /) KB)"
fi
HAVE_CODE=0
[ -f include/strata/program/tier_spec_copy.hpp ] && grep -q STRATA_ADAPT_SPEC_COPY src/core/conversation_disk_env.def 2>/dev/null && HAVE_CODE=1
HAVE_TOOL=0
[ -f tools/pancake/replay_spec_copy.py ] && HAVE_TOOL=1
say "HEADER tier_spec_copy.hpp + env table entries present: $HAVE_CODE; replay_spec_copy.py present: $HAVE_TOOL"
ls $R/bench/specs/216-*.json $R/bench/specs/217-*.json >/dev/null 2>&1 || say "HEADER specs 216 / 217 missing"
python3 tools/pancake/spec_lint.py --dir bench/specs --quiet > $T/spec_lint.txt 2>&1; say "HEADER spec_lint exit=$?"
lint_bad() { grep -Eq "^[A-Za-z-]+ +$1-" $T/spec_lint.txt; }
spec_ok() { ls $R/bench/specs/$1-*.json >/dev/null 2>&1 && ! lint_bad $1; }
python3 tools/pancake/test_disk_env_table.py > $T/disk_env_table.txt 2>&1; say "HEADER disk_env_table exit=$?"
python3 -m pytest -q tools/pancake/test_replay_spec_copy.py > $T/pytest.txt 2>&1; say "HEADER pytest replay_spec_copy exit=$? $(tail -1 $T/pytest.txt | cut -c1-120)"

# ------------------------------------------------------------------ helpers
ab() {  # ab SPECFILE PREFIX flags... : abtest with the chain-N retry on 'already running'
  sp=$1; pfx=$2; shift 2; tries=0
  while :; do
    if lowdisk; then say "SKIP $pfx: / has < 300 MB free"; return 9; fi
    wait_idle; say "START $pfx $(date +%F_%T)"
    python3 tools/pancake/abtest.py $sp "$@" --out $O/winZ-$pfx.jsonl > $O/winZ-$pfx.stdout 2>&1; rc=$?
    say "END $pfx exit=$rc $(date +%T)"
    if [ $rc -eq 3 ] && grep -q "already running" $O/winZ-$pfx.stdout && [ $tries -lt 12 ]; then tries=$((tries+1)); sleep 300; continue; fi
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
  say "BUILD SKIPPED: the synced tree has no tier_spec_copy.hpp (origin lacks the commit); the engine steps below are skipped"
elif [ "$(freekb /)" -lt 2097152 ]; then
  say "BUILD SKIPPED: / has $(freekb /) KB free (< 2 GB); nothing was deleted; every engine step below is skipped (offline steps still run)"
else
  wait_idle; say "BUILD build-win strata tier_spec_copy_test $(date +%T) (/ free $(freekb /) KB)"
  nice -n 10 $CM --build $B -j8 --target strata tier_spec_copy_test > $S/build-win.chainZ.log 2>&1; rc=$?
  say "BUILD_END build-win exit=$rc $(date +%T) (/ free $(freekb /) KB)"
  [ $rc -ne 0 ] && grep -m8 -E "error:|undefined|CMake Error" $S/build-win.chainZ.log >> $L
  [ $rc -eq 0 ] && ENGINE=1
fi

# ------------------------------------------------------------------ B. ctest of the new unit test (a failure is logged, the chain continues)
if [ $ENGINE -eq 1 ]; then
  cd $B
  wait_idle; say "START ctest tier_spec_copy $(date +%T)"
  $CT -R 'tier_spec_copy_test' --timeout 600 --output-on-failure > $T/ctest-spec.log 2>&1
  say "END ctest tier_spec_copy exit=$? $(grep -E 'tests passed|tests failed' $T/ctest-spec.log)"
  grep -E "tier_spec_copy_test: (mock device|[0-9.]+ us per round)" $T/ctest-spec.log | head -2 | cut -c1-200 >> $L
  cd $R
fi

# ------------------------------------------------------------------ C. offline replays (CPU only): the staging predictors on every routing recording the box has
rep() {  # rep NAME TRACE SLOTS
  nm=$1; tr=$2; sl=$3
  [ -s "$tr" ] || { say "SKIP replay $nm: no trace $tr"; return 9; }
  [ $HAVE_TOOL -eq 1 ] || { say "SKIP replay $nm: no replay_spec_copy.py"; return 9; }
  wait_idle; say "START replay $nm $(date +%T)"
  python3 tools/pancake/replay_spec_copy.py $tr $PROF $sl --json $O/winZ-replay-$nm.json > $O/winZ-replay-$nm.txt 2>&1; rc=$?
  say "END replay $nm exit=$rc $(date +%T)"
  head -1 $O/winZ-replay-$nm.txt | cut -c1-200 >> $L
  grep -E "^(bl|blt|ps|pst|la:0.6|or) " $O/winZ-replay-$nm.txt | head -8 | cut -c1-170 >> $L
  return $rc
}
if [ $HAVE_TOOL -eq 1 ]; then
  rep routing-main bench/prefetch/routing.bin 10909
  W16=$O/win16-conv; WV=$O/winV-conv
  for t in real4 real5; do
    if [ -s $WV/$t.trace ]; then rep $t-main $WV/$t.trace 10909
    else say "SKIP replays of $t: chain V left no trace in $WV"; fi
  done
  rep real1-chain16a-main $W16/real1.trace 10909
fi

# ------------------------------------------------------------------ D. spec 216 screen (pass 1: every arm once), decision rule, spec 217 confirm (pass 2: only the arms that earned it)
if [ $ENGINE -eq 1 ] && spec_ok 216; then
  cd $R
  SPEC=$(ls $R/bench/specs/216-*.json | head -1); SIB=$(ls $R/bench/specs/217-*.json 2>/dev/null | head -1)
  ab $SPEC 216-screen --tier screen --drift-check
  PROOF_FAIL=0
  proof $O/winZ-216-screen spec "STRATA_ADAPT_SPEC_COPY=1:" || PROOF_FAIL=1
  grep -h "spec copy:" $O/winZ-216-screen/*.engine.log 2>/dev/null | sort | uniq -c | sort -rn | head -6 | cut -c1-230 >> $L
  grep -h "staging area" $O/winZ-216-screen/*.engine.log 2>/dev/null | sort -u | head -3 | cut -c1-160 >> $L
  rm -f $T/216-confirm.json
  extra=""; [ -n "$SIB" ] && extra="--confirm-spec $SIB"
  python3 $DEC $O/winZ-216-screen.stdout $SPEC $T/216-confirm.json $extra > $T/216-decide.txt 2>&1
  grep -E "WIN|CONFIRM|NONE|dropped" $T/216-decide.txt | cut -c1-230 >> $L
  if [ -s $T/216-confirm.json ] && [ $PROOF_FAIL -eq 0 ]; then
    ab $T/216-confirm.json 216-confirm --tier confirm --block 3 --warmup 1 --outlier robust --reclaim-max 3 --reps 5
    grep -E "vs .*(better|worse|no difference)" $O/winZ-216-confirm.stdout | cut -c1-200 >> $L
  elif [ -s $T/216-confirm.json ]; then say "confirm 216 skipped: a proof line was missing (an arm is the base)"; fi
else
  say "SKIP spec 216: no engine build, or the spec is missing or flagged by spec_lint"
fi
systemctl --user start pancake.service; say "service started $(date +%T); disk / free $(freekb /) KB"
echo WINCHAINZ_DONE >> $L
