#!/bin/sh
# Windows-session chain 10 (MOD_PLAN.md section 6 item 17 part 18): after chain 9 (WINCHAIN9_DONE): configure and build
# build-win-hf (= build-win's options + -DSTRATA_HIP_HEAD_FMT=ON), run head_argmax_parity on the card, then spec 108
# (STRATA_HEAD_ARGMAX_FUSED) at screen only if the parity passes.  Same rules: 3 idle minutes before every step;
# pancake.service stopped only after 3 minutes without a client; restarted at the end.
S=/home/dan/clone-of-d/src; R=$S/Strata-win; B=$S/build-win-hf; L=$S/win-chain10.log; O=/home/dan/auto-runs
CM=/home/linuxbrew/.linuxbrew/bin/cmake
export PATH=$R/.venv/bin:$PATH LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib
echo "START $(date +%F_%T)" >> $L
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[a]uto-chain|[a]uto-post|[i]q_avx2_parity|[t]tft_bench.py|[w]in-chain9.sh" >/dev/null; }
wait_idle() { q=0; while [ $q -lt 3 ]; do if busy; then q=0; sleep 60; else q=$((q+1)); sleep 60; fi; done; }
clients() { ss -Htn state established '( sport = :11437 )' 2>/dev/null | wc -l; }

until grep -q WINCHAIN9_DONE $S/win-chain9.log 2>/dev/null; do sleep 300; done
echo "chain 9 done $(date +%T)" >> $L
q=0; while [ $q -lt 3 ]; do if [ "$(clients)" -gt 0 ]; then q=0; echo "WAIT service has clients $(date +%T)" >> $L; else q=$((q+1)); fi; sleep 60; done
systemctl --user stop pancake.service; echo "service stopped (idle) $(date +%T)" >> $L

wait_idle
echo "CONFIGURE+BUILD build-win-hf $(date +%T)" >> $L
$CM -S $R -B $B -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.0/lib/llvm/bin/clang++ \
  -DCMAKE_MAKE_PROGRAM=$S/Strata/.venv/bin/ninja -DCMAKE_PREFIX_PATH=/opt/rocm/core-10.0 \
  -DSTRATA_GGML_DIR=/home/dan/clone-of-d/build-int/_deps/strata_llamacpp-src -DSTRATA_AMD_COMMUNITY_TUNING=ON \
  -DSTRATA_BUILD_TESTS=ON -DSTRATA_BUILD_CONVERSATION_TESTS=ON -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_MMQ_TQ_OVERLAY=ON \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PREFILL_MMQ=ON -DSTRATA_WINDOW_MAX_T=auto \
  -DSTRATA_PLE_FIXTURE_DIR=$R/bench/micro -DSTRATA_HIP_HEAD_FMT=ON > $B.chain10.log 2>&1 && \
  nice -n 10 $CM --build $B -j8 --target strata head_argmax_parity >> $B.chain10.log 2>&1
rc=$?; echo "BUILD_END build-win-hf exit=$rc $(date +%T)" >> $L
[ $rc -ne 0 ] && grep -m8 -E "error:|undefined|CMake Error" $B.chain10.log >> $L
if [ $rc -eq 0 ]; then
  wait_idle
  HIP_VISIBLE_DEVICES=0 $B/head_argmax_parity > $O/win10-head_argmax_parity.txt 2>&1; prc=$?
  echo "head_argmax_parity exit=$prc $(tail -1 $O/win10-head_argmax_parity.txt)" >> $L
  if [ $prc -eq 0 ]; then
    cd $R; echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null
    wait_idle
    echo "START 108-head-argmax-fused $(date +%F_%T)" >> $L
    python3 tools/pancake/abtest.py bench/specs/108-head-argmax-fused.json --tier screen --out $O/win10-108-head-argmax-fused.jsonl \
      > $O/win10-108-head-argmax-fused.stdout 2>&1
    echo "END 108-head-argmax-fused exit=$? $(date +%T)" >> $L
  else
    echo "parity failed: spec 108 skipped" >> $L
  fi
fi
systemctl --user start pancake.service; echo "service started $(date +%T)" >> $L
echo WINCHAIN10_DONE >> $L
