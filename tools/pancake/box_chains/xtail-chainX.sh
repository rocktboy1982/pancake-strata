#!/bin/sh
# Tail chain X (owner order 2026-11-18): the long-shot / low-prior box work moved out of chains T and U, run only after every queued job (V, then after7.sh):
# C probes + expdec ablation, 194 / 195 (five variant builds, prompt-kernel macros), 196 (A/A adaptive), 167 (race soak), and the specs 212 / 213 (statistical expert cache: the offline replay
# T-929..T-931 predicts no gain).  Uses the build-win and the tree chain T left; builds only the variants.  The file name has no "win-chain" so that after7.sh busy test never sees this waiter.
# Completion marker: the last line of /home/dan/clone-of-d/src/xtail-chainX.log is WINCHAINX_DONE.
S=/home/dan/clone-of-d/src; R=$S/Strata-win; L=$S/xtail-chainX.log; O=/home/dan/auto-runs; T=$O/chainX; PM=/mnt/pancake/chainX
CM=/home/linuxbrew/.linuxbrew/bin/cmake; CT=/home/linuxbrew/.linuxbrew/bin/ctest; HIPCC=/opt/rocm/core-10.0/bin/hipcc
DEC=$S/chainT-decide.py; B=$S/build-win
export PATH=$R/.venv/bin:$PATH LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib HIP_VISIBLE_DEVICES=0
mkdir -p $T $PM
echo "START $(date +%F_%T)" >> $L
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[c]onv_trace.py|[a]uto-chain|[a]uto-post|[i]q_avx2_parity|[t]tft_bench.py|[w]in-chain[A-Za-z0-9]*\.sh" >/dev/null; }
wait_idle() { if [ "${QI:-0}" = 1 ]; then q=0; while [ $q -lt 2 ]; do if busy; then q=0; sleep 15; else q=$((q+1)); sleep 15; fi; done; else q=0; while [ $q -lt 3 ]; do if busy; then q=0; sleep 60; else q=$((q+1)); sleep 60; fi; done; QI=1; fi; }   # quick idle (owner order 2026-11-18, box hours): the full 3 minutes only the first time, 30 s between this chain's own later steps; busy() still sees every other job
quick_idle() { while busy; do sleep 20; done; }
clients() { ss -Htn state established '( sport = :11437 )' 2>/dev/null | wc -l; }
freekb() { df -Pk "$1" | awk 'NR==2{print $4}'; }
say() { echo "$*" >> $L; }
lowdisk() { [ "$(freekb $S)" -lt 307200 ]; }   # < 300 MB on / : no engine step starts
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
    python3 tools/pancake/abtest.py $sp "$@" --out $O/winX-$pfx.jsonl > $O/winX-$pfx.stdout 2>&1; rc=$?
    say "END $pfx exit=$rc $(date +%T)"
    if [ $rc -eq 3 ] && grep -q "already running" $O/winX-$pfx.stdout && [ $tries -lt 12 ]; then tries=$((tries+1)); sleep 300; continue; fi
    return $rc
  done
}
gate() {  # gate NAME SPEC [extra]
  gn=$1; gs=$2; shift 2
  step "gate $gn" $O/winX-$gn.stdout python3 tools/pancake/quality_gate.py bench/specs/$gs.json --out $O/winX-$gn/gate --near-tie "$@"
  grep -E "PASS|REJECT|INCONCLUSIVE|near-tie|top-1" $O/winX-$gn.stdout | tail -12 | cut -c1-200 >> $L
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
  python3 $DEC $O/winX-$nm-screen.stdout $file $T/$nm-confirm.json $extra > $T/$nm-decide.txt 2>&1
  cat $T/$nm-decide.txt | grep -E "WIN|CONFIRM|NONE|dropped" | cut -c1-230 >> $L
  [ -s $T/$nm-confirm.json ] || return 0
  [ -n "$PG" ] && { proof $O/winX-$nm-screen "$PA" "$PG" "$PB" || say "PROOF $nm: informational only, the confirm still runs"; }
  [ "$PROOF_FAIL" = "1" ] && { say "confirm $nm skipped: the screen's proof line was missing (the arm is the base)"; return 0; }
  ab $T/$nm-confirm.json $nm-confirm --tier confirm --block 3 --warmup 1 --outlier robust --reclaim-max 3 --reps 5
  grep -E "vs .*(better|worse|no difference)" $O/winX-$nm-confirm.stdout | cut -c1-200 >> $L
  return 0
}
ctest_pass() { grep -Eq "Test +#[0-9]+: $1 +\.+ +Passed" $T/ctest-new.log $T/ctest-gpu2.log 2>/dev/null; }

PG=; PA=; PB=; PROOF_FAIL=0
until grep -q AFTER7_DONE /home/dan/auto-runs/after7.log 2>/dev/null; do sleep 300; done
say "after7 done $(date +%T)"
q=0; while [ $q -lt 3 ]; do if [ "$(clients)" -gt 0 ]; then q=0; say "WAIT service has clients $(date +%T)"; else q=$((q+1)); fi; sleep 60; done
systemctl --user stop pancake.service; say "service stopped (idle) $(date +%T)"
cd $R; echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null
python3 tools/pancake/spec_lint.py --dir bench/specs --quiet > $T/spec_lint.txt 2>&1; say "HEADER spec_lint exit=$?"
lint_bad() { grep -Eq "^[A-Za-z-]+ +$1-" $T/spec_lint.txt; }
spec_ok() { ls $R/bench/specs/$1-*.json >/dev/null 2>&1 && ! lint_bad $1; }
ENGINE=0; [ -x $B/strata ] && ENGINE=1
say "ENGINE=$ENGINE (build-win/strata present)"
# ------------------------------------------------------------------ C. probes, cheap first (engine off, GPU alone)
if [ $ENGINE -eq 1 ]; then
  wait_idle
  pstep act_f16 $T/act_f16.out $B/act_f16_probe --reps 200
  pstep link_touch $T/link_touch.out $B/link_touch_probe --us 4000 --reps 400 --touch-us 5
  pstep pcie_link $T/pcie_link.out python3 tools/pancake/pcie_link.py
  pstep pcie_link_json $T/pcie_link.json python3 tools/pancake/pcie_link.py --json
  pstep wmma_rate $T/wmma_rate.out $B/wmma_rate_probe --iters 20000 --reps 7
  pstep int4_wmma_parity $T/int4_wmma_parity.out $B/int4_wmma_probe --parity-only; rcp=$?
  if [ $rcp -eq 0 ]; then
    pstep int4_wmma_gateup_a8 $T/int4_wmma_gateup_a8.out $B/int4_wmma_probe --m 640 --k 2560 --n 512 --block 256 --a4 0 --reps 20
    pstep int4_wmma_gateup_a4 $T/int4_wmma_gateup_a4.out $B/int4_wmma_probe --m 640 --k 2560 --n 512 --block 256 --a4 1 --reps 20
    pstep int4_wmma_down_a8 $T/int4_wmma_down_a8.out $B/int4_wmma_probe --m 2560 --k 640 --n 512 --block 128 --a4 0 --reps 20
    pstep int4_wmma_down_a4 $T/int4_wmma_down_a4.out $B/int4_wmma_probe --m 2560 --k 640 --n 512 --block 128 --a4 1 --reps 20
  else say "int4_wmma bench skipped: parity exit=$rcp"; fi
  pstep int4_moe_parity $T/int4_moe_parity.out $B/int4_moe_probe --parity-only; rcp=$?
  if [ $rcp -eq 0 ]; then
    pstep int4_moe_gateup $T/int4_moe_gateup.out $B/int4_moe_probe --m 512 --k 2560 --tokens 2048 --experts 64 --block 256 --reps 20
    pstep int4_moe_down $T/int4_moe_down.out $B/int4_moe_probe --m 512 --k 640 --tokens 2048 --experts 64 --block 128 --reps 20
  else say "int4_moe bench skipped: parity exit=$rcp"; fi
  pstep attn_fold $T/attn_fold.out $B/attn_fold_probe --cap 8192 --ctx 8192 --layers 40 --reps 20 --launches 200
  pstep raw_buffer $T/raw_buffer.out $B/raw_buffer_probe --rows 65536 --rowbytes 640 --reps 30; rcp=$?
  [ $rcp -ne 0 ] && pstep raw_buffer_flags2 $T/raw_buffer_flags2.out $B/raw_buffer_probe --rows 65536 --rowbytes 640 --reps 30 --flags 0x31014000
  # vmm_test alone with the per-step ZERO_BACK diagnostics (T-950; chain T's ctest failed vmm_test on gfx1100): rebuild only that target, run it
  # directly (not ctest), output in $T/vmm_test_zero.out; the answer is the "zero-back step" lines.  First run of this chain after the tree on the box has the diagnostics.
  grep -q "ZERO_BACK diag" $R/tests/core/vmm_test.cpp 2>/dev/null || say "vmm_test: the tree on the box has no ZERO_BACK diagnostics (not synced); the old test runs"
  pstep vmm_test_build $T/vmm_test_build.log nice -n 10 $CM --build $B --target vmm_test; rcv=$?
  if [ $rcv -eq 0 ] && [ -x $B/vmm_test ]; then pstep vmm_test_zero $T/vmm_test_zero.out $B/vmm_test; grep -E "zero-back|vmm_test:|FAIL|strata:" $T/vmm_test_zero.out | head -80 | cut -c1-220 >> $L
  else say "vmm_test_zero skipped: build exit=$rcv"; fi
  # the probe is rebuilt from the tree (probe 5b, the zero-back cycle, is new since chain T built its copy); chain T's binary is the fallback
  VP=/mnt/pancake/chainT/vmm_probe
  if [ -f $R/tools/pancake/vmm_probe.cpp ] && $HIPCC -O2 -std=c++17 --offload-arch=gfx1100 $R/tools/pancake/vmm_probe.cpp -o $PM/vmm_probe > $T/vmm_probe_build.log 2>&1 && [ -x $PM/vmm_probe ]; then VP=$PM/vmm_probe; say "vmm_probe rebuilt from the tree"; else say "vmm_probe: rebuild failed or missing, using chain T's binary"; fi
  if [ -x $VP ]; then pstep vmm_probe $T/vmm_probe.out $VP 64 512; else say "vmm_probe skipped: $VP is missing"; fi
  pstep persist_tape $T/persist_tape.out $B/persist_tape_probe --segments 432 --tasks 192 --wbytes 16384 --grids 96,192,288,384 --reps 20
  pstep persist_layer_parity $T/persist_layer_parity.out $B/persist_layer_parity
  pstep retained_replay $T/retained_replay.out $B/retained_replay_probe --n 2248,2648 --blocks 96 --threads 256 --reps 20 --window-ms 24.8
  # expdec ablation (LEADS O14): default = build-win's expdec_bench; the two ablation builds go to /mnt/pancake
  AB=$PM/ablate; mkdir -p $AB
  vbuild() {  # vbuild DIR TARGET cmake-extra... (configure like chain 11's variants, build one target)
    vd=$1; vt=$2; shift 2
    if [ "$(freekb /mnt/pancake)" -lt 8000000 ]; then say "VARIANT $vd skipped: /mnt/pancake has < 8 GB free"; return 1; fi
    wait_idle; say "CONFIGURE+BUILD $vd $(date +%T)"
    $CM -S $R -B $vd -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
      -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.0/lib/llvm/bin/clang++ \
      -DCMAKE_MAKE_PROGRAM=$S/Strata/.venv/bin/ninja -DCMAKE_PREFIX_PATH=/opt/rocm/core-10.0 \
      -DSTRATA_GGML_DIR=/home/dan/clone-of-d/build-int/_deps/strata_llamacpp-src -DSTRATA_AMD_COMMUNITY_TUNING=ON \
      -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_MMQ_TQ_OVERLAY=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PREFILL_MMQ=ON \
      -DSTRATA_WINDOW_MAX_T=auto -DSTRATA_PLE_FIXTURE_DIR=$R/bench/micro "$@" > $vd.log 2>&1 && \
      nice -n 10 $CM --build $vd -j8 --target $vt >> $vd.log 2>&1
    rc=$?; say "BUILD_END $vd exit=$rc $(date +%T)"; [ $rc -ne 0 ] && grep -m6 -E "error:|undefined|CMake Error" $vd.log >> $L
    return $rc
  }
  PAIRS=16/42,17/42,22/42
  pstep ablate_default $T/ablate-default.txt $B/expdec_bench 300 0 20 $PAIRS def,dec
  vbuild $AB/nomac expdec_bench "-DCMAKE_HIP_FLAGS=-DSTRATA_KERNEL_ABLATE=1" && pstep ablate_nomac $T/ablate-nomac.txt $AB/nomac/expdec_bench 300 0 20 $PAIRS def,dec
  vbuild $AB/samerow expdec_bench "-DCMAKE_HIP_FLAGS=-DSTRATA_KERNEL_ABLATE=2" && pstep ablate_samerow $T/ablate-samerow.txt $AB/samerow/expdec_bench 300 0 20 $PAIRS def,dec
  say "START ablate_expdec report $(date +%T)"
  python3 tools/pancake/ablate_expdec.py report $T/ablate-default.txt $T/ablate-nomac.txt $T/ablate-samerow.txt > $T/ablate-report.txt 2>&1; say "END ablate_expdec report exit=$?"
fi

if [ $ENGINE -eq 1 ]; then
  cd $R
  # 194 / 195: build macros, one binary per arm (variant builds on /mnt/pancake, linked as $S/build-<name>)
  if spec_ok 194; then
    OKV=1
    for v in "ldsbar -DCMAKE_HIP_FLAGS=-DSTRATA_LDS_ONLY_BARRIER" "lazy2 -DCMAKE_HIP_FLAGS=-DSTRATA_ATTN_LAZY_RESCALE=2" \
             "lazy8 -DCMAKE_HIP_FLAGS=-DSTRATA_ATTN_LAZY_RESCALE=8" "lazy2ns -DCMAKE_HIP_FLAGS=-DSTRATA_ATTN_LAZY_RESCALE=2+-DSTRATA_ATTN_LAZY_NOSKIP" \
             "xlds -DCMAKE_HIP_FLAGS=-DSTRATA_PF_MOE_XLDS"; do
      nm=${v%% *}; fl=${v#* }; fl=$(echo "$fl" | tr '+' ' ')
      [ -e $S/build-$nm ] && [ ! -L $S/build-$nm ] && { say "VARIANT build-$nm: a real directory already exists at $S/build-$nm, left alone"; OKV=0; continue; }
      if [ -x $PM/builds/build-$nm/strata ]; then say "VARIANT build-$nm: already built"; else mkdir -p $PM/builds; vbuild $PM/builds/build-$nm strata "$fl" || OKV=0; fi
      [ -x $PM/builds/build-$nm/strata ] && ln -sfn $PM/builds/build-$nm $S/build-$nm
    done
    if [ $OKV -eq 1 ]; then
      auto 194 194 - - --tier screen --drift-check
      grep -q "lazy" $T/194-confirm.json 2>/dev/null && spec_ok 195 && gate 195 195-prompt-lazy-quality
    else say "SKIP 194 / 195: a variant build failed (arms need one binary each)"; fi
  fi
fi
if [ $ENGINE -eq 1 ]; then
  cd $R
  # 196: the adaptive machinery on a repeated vs a rotated prompt (A/A), two confirm runs
  if spec_ok 196; then
    ab $(ls $R/bench/specs/196-*.json) 196-plain --tier confirm --block 3 --warmup 1 --outlier robust --reclaim-max 3 --reps 5
    grep -E "vs .*(better|worse|no difference)" $O/winX-196-plain.stdout | cut -c1-200 >> $L
    ab $(ls $R/bench/specs/196-*.json) 196-rotated --tier confirm --block 3 --warmup 1 --outlier robust --reclaim-max 3 --reps 5 --rotate-prompts
    grep -E "vs .*(better|worse|no difference)" $O/winX-196-rotated.stdout | cut -c1-200 >> $L
  fi
  # 167: race soak, 60 starts of the edit case per arm, same binary twice (the longest step: last)
  if spec_ok 167; then
    ab $(ls $R/bench/specs/167-*.json) 167-soak --tier confirm --outlier robust --no-early-stop
    grep -A4 "generated text vs base" $O/winX-167-soak.stdout | cut -c1-200 >> $L
  fi
fi
proof() {  # proof DIR ARMPREFIX GOODPATTERN: 0 if every engine log of the arm has the good line
  d=$1; a=$2; g=$3; ok=0; n=0
  for f in $d/$a*-r*-a*.engine.log; do [ -f "$f" ] || continue; n=$((n+1))
    grep -q -- "$g" $f || { say "PROOF $a: '$g' missing in $(basename $f)"; ok=1; }
  done
  [ $n -eq 0 ] && { say "PROOF $a: no engine log under $d"; return 1; }
  [ $ok -eq 0 ] && say "PROOF $a: ok ($n logs, '$g')"; return $ok
}


if [ $ENGINE -eq 1 ] && spec_ok 212 && spec_ok 213; then
  cd $R
  SPEC=$(ls $R/bench/specs/212-*.json | head -1); SIB=$(ls $R/bench/specs/213-*.json | head -1)
  ab $SPEC 212-screen --tier screen --drift-check
  PROOF_FAIL=0
  proof $O/winX-212-screen ff70 "STRATA_ADAPT_FILL_FIRST=0.7:" || PROOF_FAIL=1
  proof $O/winX-212-screen tr2 "STRATA_ADAPT_TRANSITION=2:" || PROOF_FAIL=1
  grep -h "tier stat:" $O/winX-212-screen/*.engine.log 2>/dev/null | sort | uniq -c | sort -rn | head -6 | cut -c1-200 >> $L
  rm -f $T/212-confirm.json
  python3 $DEC $O/winX-212-screen.stdout $SPEC $T/212-confirm.json --confirm-spec $SIB > $T/212-decide.txt 2>&1
  grep -E "WIN|CONFIRM|NONE|dropped" $T/212-decide.txt | cut -c1-230 >> $L
  if [ -s $T/212-confirm.json ] && [ $PROOF_FAIL -eq 0 ]; then
    ab $T/212-confirm.json 212-confirm --tier confirm --block 3 --warmup 1 --outlier robust --reclaim-max 3 --reps 5
    grep -E "vs .*(better|worse|no difference)" $O/winX-212-confirm.stdout | cut -c1-200 >> $L
  elif [ -s $T/212-confirm.json ]; then say "confirm 212 skipped: a proof line was missing (an arm is the base)"; fi
fi
systemctl --user start pancake.service; say "service started $(date +%T); disk / free $(freekb /) KB"
echo WINCHAINX_DONE >> $L