#!/bin/sh
# Windows-session chain T (owner order 2026-11-18, autonomous): ALL newly coded, never-run box work, in value / cost order.
# MOD_PLAN.md section 5 items 54-57, LEADS.md Parts L, N, O, P.  Runs after chain S (the last queued chain: N -> G -> R -> W2 -> H2 -> S -> T).
# Rules (standing): box jobs never overlap (this script waits for WINCHAINS_DONE, and every step waits 3 idle minutes); nothing is killed or
# deleted; pancake.service is stopped only after 3 minutes without a client on port 11437 and started again at the end; the sudoers file and
# permissions are never touched; code is synced with `tar xmf` semantics (mtimes reset) so ninja rebuilds; the root disk is 98% full, so every
# build checks `df` first and is SKIPPED (logged, nothing deleted) when less than 2 GB is free; the variant builds go to /mnt/pancake.
# FAST ORDER (owner order 2026-11-18, box hours; the old script is win-chainT.sh.bak): 0 sync + header | A build | B ctest | D quality gates of the promotion candidates
#        (DQ_FAST, floor, route-budget family) | E D4 replay | F env-only screens by expected gain; the long-shot rest (C probes + ablation, 194 / 195, 196, 167) is in tail chain X.
# Old order: 0 sync + header | A build | B ctest of the new tests | C probes (cheap first) | D quality gates (DQ_FAST, floor, route-budget family)
#        | E D4 suffix-index replay | F abtest specs screen -> confirm only for screen winners (>= 2%, lossy >= 3%; chainT-decide.py)
#        | G the long ones (196 twice, 167 race soak).  Log: win-chainT.log (START / END lines with exit codes); output: /home/dan/auto-runs/chainT/ and winT-*.
# Completion marker: the last line of win-chainT.log is WINCHAINT_DONE.
S=/home/dan/clone-of-d/src; R=$S/Strata-win; L=$S/win-chainT.log; O=/home/dan/auto-runs; T=$O/chainT; PM=/mnt/pancake/chainT
CM=/home/linuxbrew/.linuxbrew/bin/cmake; CT=/home/linuxbrew/.linuxbrew/bin/ctest; HIPCC=/opt/rocm/core-10.0/bin/hipcc
DEC=$S/chainT-decide.py; B=$S/build-win
REPO_URL=https://github.com/rocktboy1982/pancake-strata.git
export PATH=$R/.venv/bin:$PATH LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib HIP_VISIBLE_DEVICES=0
mkdir -p $T $PM
echo "START $(date +%F_%T)" >> $L
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[c]onv_trace.py|[a]uto-chain|[a]uto-post|[i]q_avx2_parity|[t]tft_bench.py|[w]in-chain[A-Sa-z0-9][A-Za-z0-9]*\.sh" >/dev/null; }
wait_idle() { if [ "${QI:-0}" = 1 ]; then q=0; while [ $q -lt 2 ]; do if busy; then q=0; sleep 15; else q=$((q+1)); sleep 15; fi; done; else q=0; while [ $q -lt 3 ]; do if busy; then q=0; sleep 60; else q=$((q+1)); sleep 60; fi; done; QI=1; fi; }   # quick idle (owner order 2026-11-18, box hours): the full 3 minutes only the first time, 30 s between this chain's own later steps; busy() still sees every other job
quick_idle() { while busy; do sleep 20; done; }
clients() { ss -Htn state established '( sport = :11437 )' 2>/dev/null | wc -l; }
freekb() { df -Pk "$1" | awk 'NR==2{print $4}'; }
say() { echo "$*" >> $L; }
lowdisk() { [ "$(freekb $S)" -lt 307200 ]; }   # < 300 MB on / : no engine step starts

until grep -q WINCHAINS_DONE $S/win-chainS.log 2>/dev/null; do sleep 300; done
say "chain S done $(date +%T)"
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
  SRC=$PM/fallback-src.tgz; SHA="$(cat $PM/fallback-src.sha 2>/dev/null) (FALLBACK staged on the Windows PC; the download failed)"
  say "SYNC source: $SHA"
fi
if [ -s $SRC ] && [ "$(freekb /)" -gt 1000000 ] && tar tzf $SRC >/dev/null 2>&1; then
  tar xzmf $SRC --no-same-owner --strip-components=1 -C $R; rc=$?; say "SYNC exit=$rc ($(tar tzf $SRC | wc -l) entries) $(date +%T)"
else
  say "SYNC SKIPPED: no usable source or / has < 1 GB free ($(freekb /) KB)"
fi
[ -f $S/chainT-extra.tar ] && tar xmkf $S/chainT-extra.tar -C $R && say "extra stage applied (new specs kept if the repo already has them): $(tar tf $S/chainT-extra.tar | wc -l) files"
PG=; PA=; PB=; PROOF_FAIL=0
SPECS="164 165 166 167 168 169 170 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211"
MISSING=""; for n in $SPECS; do ls $R/bench/specs/$n-*.json >/dev/null 2>&1 || MISSING="$MISSING $n"; done
say "HEADER missing specs:${MISSING:- none}"
python3 tools/pancake/spec_lint.py --dir bench/specs --quiet > $T/spec_lint.txt 2>&1; say "HEADER spec_lint exit=$? ($(wc -l < $T/spec_lint.txt) lines, $T/spec_lint.txt)"
lint_bad() { grep -Eq "^[A-Za-z-]+ +$1-" $T/spec_lint.txt; }
for n in $SPECS; do lint_bad $n && say "HEADER spec_lint flags spec $n: it is SKIPPED"; done
for g in 209-dq-fast-quality 210-route-budget-quality 168-quality-floor 193-wmma-rows-quality 195-prompt-lazy-quality; do
  python3 tools/pancake/quality_gate.py bench/specs/$g.json --out $T/dry-$g --near-tie --dry-run > $T/dry-$g.txt 2>&1; say "HEADER gate dry-run $g exit=$?"; done
python3 tools/pancake/replay_turns.py bench/specs/211-d4-suffix-index-replay.json --out $T/dry-211 --reps 1 --dry-run > $T/dry-211.txt 2>&1; say "HEADER replay dry-run 211 exit=$?"
spec_ok() { ls $R/bench/specs/$1-*.json >/dev/null 2>&1 && ! lint_bad $1; }

# ------------------------------------------------------------------ helpers
step() {  # step NAME OUTFILE cmd...   (3 idle minutes first; exit code in the log)
  sn=$1; so=$2; shift 2
  if lowdisk; then say "SKIP $sn: / has < 300 MB free"; return 9; fi
  wait_idle; say "START $sn $(date +%F_%T)"
  "$@" > $so 2>&1; rc=$?; say "END $sn exit=$rc $(date +%T)"; return $rc
}
pstep() {  # probe step inside a group (no 3-minute wait, only until nothing else runs); timeout 15 min
  sn=$1; so=$2; shift 2
  quick_idle; say "START $sn $(date +%T)"
  timeout 900 "$@" > $so 2>&1; rc=$?; say "END $sn exit=$rc $(date +%T)"; return $rc
}
ab() {  # ab SPECFILE PREFIX flags... : abtest with the chain-N retry on 'already running'
  sp=$1; pfx=$2; shift 2; tries=0
  while :; do
    if lowdisk; then say "SKIP $pfx: / has < 300 MB free"; return 9; fi
    wait_idle; say "START $pfx $(date +%F_%T)"
    python3 tools/pancake/abtest.py $sp "$@" --out $O/winT-$pfx.jsonl > $O/winT-$pfx.stdout 2>&1; rc=$?
    say "END $pfx exit=$rc $(date +%T)"
    if [ $rc -eq 3 ] && grep -q "already running" $O/winT-$pfx.stdout && [ $tries -lt 12 ]; then tries=$((tries+1)); sleep 300; continue; fi
    return $rc
  done
}
gate() {  # gate NAME SPEC [extra]
  gn=$1; gs=$2; shift 2
  step "gate $gn" $O/winT-$gn.stdout python3 tools/pancake/quality_gate.py bench/specs/$gs.json --out $O/winT-$gn/gate --near-tie "$@"
  grep -E "PASS|REJECT|INCONCLUSIVE|near-tie|top-1" $O/winT-$gn.stdout | tail -12 | cut -c1-200 >> $L
}
proof() {  # proof DIR ARMPREFIX GOODPATTERN [BADPATTERN]: 0 if every engine log of the arm has the good line and none the bad one
  d=$1; a=$2; g=$3; bd=$4; ok=0; n=0
  for f in $d/$a*-r*-a*.engine.log; do [ -f "$f" ] || continue; n=$((n+1))
    grep -q -- "$g" $f || { say "PROOF $a: '$g' missing in $(basename $f)"; ok=1; }
    [ -n "$bd" ] && grep -q -- "$bd" $f && { say "PROOF $a: '$bd' present in $(basename $f)"; ok=1; }
  done
  [ $n -eq 0 ] && { say "PROOF $a: no engine log under $d"; return 1; }
  [ $ok -eq 0 ] && say "PROOF $a: ok ($n logs, '$g')"; return $ok
}
auto() {  # auto NAME SCREENSPEC SIBLING|- REF|- [screen flags...]: screen, decision rule, confirm only for winners
  nm=$1; sp=$2; sib=$3; rf=$4; shift 4
  spec_ok $sp || { say "SKIP $nm: spec $sp missing or flagged by spec_lint"; return 1; }
  file=$(ls $R/bench/specs/$sp-*.json | head -1)
  ab $file $nm-screen "$@" || { say "screen $nm failed or skipped: no confirm"; return 1; }
  extra=""; [ "$sib" != "-" ] && extra="--confirm-spec $(ls $R/bench/specs/$sib-*.json | head -1)"
  [ "$rf" != "-" ] && extra="$extra --ref $rf"
  rm -f $T/$nm-confirm.json
  python3 $DEC $O/winT-$nm-screen.stdout $file $T/$nm-confirm.json $extra > $T/$nm-decide.txt 2>&1
  cat $T/$nm-decide.txt | grep -E "WIN|CONFIRM|NONE" | cut -c1-230 >> $L
  [ -s $T/$nm-confirm.json ] || return 0
  [ -n "$PG" ] && { proof $O/winT-$nm-screen "$PA" "$PG" "$PB" || say "PROOF $nm: informational only, the confirm still runs"; }
  [ "$PROOF_FAIL" = "1" ] && { say "confirm $nm skipped: the screen's proof line was missing (the arm is the base)"; return 0; }
  ab $T/$nm-confirm.json $nm-confirm --tier confirm --block 3 --warmup 1 --outlier robust --reclaim-max 3 --reps 5
  grep -E "vs .*(better|worse|no difference)" $O/winT-$nm-confirm.stdout | cut -c1-200 >> $L
  return 0
}
ctest_pass() { grep -Eq "Test +#[0-9]+: $1 +\.+ +Passed" $T/ctest-new.log $T/ctest-gpu2.log 2>/dev/null; }

# ------------------------------------------------------------------ A. build build-win (incremental; strata + every test exe + the new probes)
ENGINE=0
if [ "$(freekb /)" -lt 2097152 ]; then
  say "BUILD SKIPPED: / has $(freekb /) KB free (< 2 GB); nothing was deleted; every engine step below is skipped (offline steps still run)"
else
  wait_idle; say "BUILD build-win strata strata_tests $(date +%T) (/ free $(freekb /) KB)"
  nice -n 10 $CM --build $B -j8 --target strata strata_tests > $S/build-win.chainT.log 2>&1; rc=$?
  say "BUILD_END build-win exit=$rc $(date +%T) (/ free $(freekb /) KB)"
  [ $rc -ne 0 ] && grep -m8 -E "error:|undefined|CMake Error" $S/build-win.chainT.log >> $L
  if [ $rc -eq 0 ]; then
    ENGINE=1
    for t in act_f16_probe link_touch_probe wmma_rate_probe int4_wmma_probe int4_moe_probe attn_fold_probe raw_buffer_probe \
             persist_tape_probe persist_layer_parity retained_replay_probe expdec_bench; do
      nice -n 10 $CM --build $B -j8 --target $t >> $S/build-win.chainT.log 2>&1; say "BUILD_END target $t exit=$?"
    done
    $HIPCC -O2 -std=c++17 --offload-arch=gfx1100 $R/tools/pancake/vmm_probe.cpp -o $PM/vmm_probe >> $S/build-win.chainT.log 2>&1; say "BUILD_END vmm_probe exit=$?"
    say "disk after the builds: / free $(freekb /) KB"
  fi
fi

# ------------------------------------------------------------------ B. ctest of the new tests (failures are logged, the chain continues)
if [ $ENGINE -eq 1 ]; then
  cd $B
  wait_idle; say "START ctest new $(date +%T)"
  $CT -R 'int4_iq2|causal_order|fold_merge|raw_buffer|kv_ring|expert_cache_segmented|vmm_test|qsa_split|gdn_snap|persist_launch|act_f16|copy_lane|free_settle|hip_decode_attn_wmma_parity|qsa_merge_daily_parity|gdn_snap_parity|persist_layer_parity|int4_rot|persist_tape_test' --timeout 600 --output-on-failure > $T/ctest-new.log 2>&1
  say "END ctest new exit=$? $(grep -E 'tests passed|tests failed' $T/ctest-new.log)"
  grep -E "Failed|\*\*\*|Not Run|SKIP" $T/ctest-new.log | head -12 | cut -c1-160 >> $L
  # on HIP the qsa_merge_daily_parity sweep is registered as qsa_qmajor_parity (chunk kernels 0..5 incl. split-grid 4 and vwin 3 / 5)
  wait_idle; say "START ctest qsa_qmajor_parity $(date +%T)"; $CT -R 'qsa_qmajor_parity|hip_qsa_wave32' --timeout 600 --output-on-failure > $T/ctest-gpu2.log 2>&1
  say "END ctest qsa_qmajor_parity exit=$? $(grep -E 'tests passed|tests failed' $T/ctest-gpu2.log)"
  cd $R
  PARITY_QSA=0; ctest_pass qsa_qmajor_parity && PARITY_QSA=1
  PARITY_WMMA=0; ctest_pass hip_decode_attn_wmma_parity && ! grep -q "NOT bit-identical" $T/ctest-new.log && PARITY_WMMA=1
  say "parity gates: qsa_qmajor_parity=$PARITY_QSA hip_decode_attn_wmma_parity=$PARITY_WMMA"
fi

# ------------------------------------------------------------------ D. quality gates (token forcing, near-tie column on)
if [ $ENGINE -eq 1 ]; then
  cd $R; echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null
  spec_ok 209 && gate 209 209-dq-fast-quality
  spec_ok 168 && gate 168 168-quality-floor
  spec_ok 210 && gate 210 210-route-budget-quality
fi


# ------------------------------------------------------------------ E. D4 suffix index on the real-session replay (a fresh index file per rep: no repeated-prompt contamination)
if [ $ENGINE -eq 1 ] && spec_ok 211; then
  for r in 1 2 3; do
    case $r in 1) ord=base,sidx-4mib,sidx-16mib;; 2) ord=sidx-16mib,sidx-4mib,base;; 3) ord=sidx-4mib,base,sidx-16mib;; esac
    sed "s#d4-replay-\([0-9a-z]*\)\.sxi#d4-replay-\1-r$r.sxi#g" $R/bench/specs/211-d4-suffix-index-replay.json > $T/211-r$r.json
    step "replay 211 rep $r ($ord)" $O/winT-211-r$r.stdout python3 tools/pancake/replay_turns.py $T/211-r$r.json --out $O/winT-211-r$r --reps 1 --arms $ord
    grep -E "verdict|total prompt|sidx|base" $O/winT-211-r$r.stdout | tail -8 | cut -c1-200 >> $L
  done
fi


# ------------------------------------------------------------------ F. abtest specs: screen, decision rule, confirm for the winners
if [ $ENGINE -eq 1 ]; then
  cd $R
  # ORDER (owner priority order 2026-11-18): promotion candidates first (gates D and replay 211 above), then the env-only screens by expected gain: decode attention
  # 197 / 199 / 192 / 164, then the expert-cache and cache-start levers 201 / 203, then 169 / 165 / 207 / 191 / 170 / 205.  194 / 195 (five variant builds), 196, 167, the
  # probes with the expdec ablation (C) and specs 212 / 213 run in the tail chain X (xtail-chainX.sh) after every other queued job.
  if [ "$PARITY_QSA" = 1 ]; then
    auto 199 199 200 - --tier screen
    auto 197 197 198 - --tier screen
  else say "SKIP 197-200: qsa_qmajor_parity (chunk kernels 3 / 4 / 5) did not pass"; fi
  # 192 / 193: WMMA rows against `wmma` (needs hip_decode_attn_wmma_parity without 'NOT bit-identical')
  if [ "$PARITY_WMMA" = 1 ]; then
    PA=wmma-rows; PG="STRATA_ATTN_WMMA_ROWS="; PB=; auto 192 192 - wmma --tier screen; PG=
    grep -q "wmma-rows" $T/192-confirm.json 2>/dev/null && spec_ok 193 && gate 193 193-wmma-rows-quality
  else say "SKIP 192 / 193: hip_decode_attn_wmma_parity did not pass"; fi
  auto 164 164 - lanecell --tier screen
  PA=chain3-copylane; PG="STRATA_LOOKUP_COPYLANE: copy lane"; PB=; auto 169 169 - - --tier screen
  PG=
  PA=warm; PG="STRATA_IC_WARM_MB acted"; PB=; auto 201 201 202 - --tier screen
  PA=defer-warm; PG="STRATA_IC_WARM_DEFER acted"; auto 203 203 204 defer --tier screen; PG=
  # 165 / 166: the elastic K/V on the whole K/V (proof: 'elastic K/V: N of 131072 cells' in the stderr log, no '--kv-grow is off because')
  spec_ok 165 && {
    ab $(ls $R/bench/specs/165-*.json) 165-screen --tier screen
    PROOF_FAIL=0
    proof $O/winT-165-screen kvgrow-whole "elastic K/V:" "kv-grow is off because" || PROOF_FAIL=1
    grep -h "elastic K/V" $O/winT-165-screen/*.engine.log 2>/dev/null | sort | uniq -c | head -5 | cut -c1-200 >> $L
    rm -f $T/165-confirm.json
    python3 $DEC $O/winT-165-screen.stdout $(ls $R/bench/specs/165-*.json) $T/165-confirm.json --confirm-spec $(ls $R/bench/specs/166-*.json) > $T/165-decide.txt 2>&1
    grep -E "WIN|CONFIRM|NONE" $T/165-decide.txt | cut -c1-230 >> $L
    if [ -s $T/165-confirm.json ] && [ $PROOF_FAIL -eq 0 ]; then
      ab $T/165-confirm.json 165-confirm --tier confirm --block 3 --warmup 1 --outlier robust --reclaim-max 3 --reps 5
      grep -E "vs .*(better|worse|no difference)" $O/winT-165-confirm.stdout | cut -c1-200 >> $L
      proof $O/winT-165-confirm kvgrow-whole "elastic K/V:" "kv-grow is off because"
    elif [ -s $T/165-confirm.json ]; then say "confirm 165 skipped: proof line missing (the arm is the base)"; fi
  }
  PA=vram-elastic; PG="STRATA_HIP_VRAM_ELASTIC=1: --vram-elastic on HIP"; PB="--vram-elastic is off"; auto 207 207 208 - --tier screen; PG=; PB=
  auto 191 191 - - --tier screen
  auto 170 170 - - --tier screen
  if [ -f /home/dan/clone-of-d/models/qwen36-UD-Q4_K_XL/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf ]; then PA=heavy-first; PG="STRATA_Q36_FA_HEAVY_FIRST=1: the flash attention"; PB=; auto 205 205 206 - --tier screen; PG=; else say "SKIP 205 / 206: qwen36 model file missing"; fi
fi

systemctl --user start pancake.service; say "service started $(date +%T); disk / free $(freekb /) KB"
echo WINCHAINT_DONE >> $L