#!/bin/sh
# Windows-session chain 11 (MOD_PLAN.md section 5 item 2, 2026-11-16 sweep of coded switches with no spec): after chain 10
# (WINCHAIN10_DONE): build the three compile-time variants (build-win-fm / -lds / -psf from build-win's options), then
# spec 109 (engine switches, screen), 112 (build macros, screen), the replay 110 and the task gate 111.
# Same rules: 3 idle minutes before every step; pancake.service stopped only after 3 minutes without a client; restarted.
S=/home/dan/clone-of-d/src; R=$S/Strata-win; L=$S/win-chain11.log; O=/home/dan/auto-runs
CM=/home/linuxbrew/.linuxbrew/bin/cmake
export PATH=$R/.venv/bin:$PATH LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib
echo "START $(date +%F_%T)" >> $L
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[a]uto-chain|[a]uto-post|[i]q_avx2_parity|[t]tft_bench.py|[w]in-chain10.sh" >/dev/null; }
wait_idle() { q=0; while [ $q -lt 3 ]; do if busy; then q=0; sleep 60; else q=$((q+1)); sleep 60; fi; done; }
clients() { ss -Htn state established '( sport = :11437 )' 2>/dev/null | wc -l; }

until grep -q WINCHAIN10_DONE $S/win-chain10.log 2>/dev/null; do sleep 300; done
echo "chain 10 done $(date +%T)" >> $L
q=0; while [ $q -lt 3 ]; do if [ "$(clients)" -gt 0 ]; then q=0; echo "WAIT service has clients $(date +%T)" >> $L; else q=$((q+1)); fi; sleep 60; done
systemctl --user stop pancake.service; echo "service stopped (idle) $(date +%T)" >> $L

variant() {  # $1 dir suffix, rest = extra cmake args
  B=$S/build-win-$1; shift
  wait_idle
  echo "CONFIGURE+BUILD $B $(date +%T)" >> $L
  $CM -S $R -B $B -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
    -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.0/lib/llvm/bin/clang++ \
    -DCMAKE_MAKE_PROGRAM=$S/Strata/.venv/bin/ninja -DCMAKE_PREFIX_PATH=/opt/rocm/core-10.0 \
    -DSTRATA_GGML_DIR=/home/dan/clone-of-d/build-int/_deps/strata_llamacpp-src -DSTRATA_AMD_COMMUNITY_TUNING=ON \
    -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_MMQ_TQ_OVERLAY=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PREFILL_MMQ=ON \
    -DSTRATA_WINDOW_MAX_T=auto -DSTRATA_PLE_FIXTURE_DIR=$R/bench/micro "$@" > $B.chain11.log 2>&1 && \
    nice -n 10 $CM --build $B -j8 --target strata >> $B.chain11.log 2>&1
  rc=$?; echo "BUILD_END $B exit=$rc $(date +%T)" >> $L
  [ $rc -ne 0 ] && grep -m6 -E "error:|undefined|CMake Error" $B.chain11.log >> $L
}
variant fm -DSTRATA_HIP_FAST_MATH=ON
variant lds -DCMAKE_HIP_FLAGS=-DSTRATA_HIP_LDS_SYNC=1
variant psf -DCMAKE_HIP_FLAGS=-DSTRATA_PA_SCHED_FENCE=1

cd $R; echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null
run() {  # $1 spec, rest = abtest flags
  sp=$1; shift; tries=0
  while :; do
    wait_idle
    echo "START $sp $(date +%F_%T)" >> $L
    python3 tools/pancake/abtest.py bench/specs/$sp.json "$@" --out $O/win11-$sp.jsonl > $O/win11-$sp.stdout 2>&1; rc=$?
    echo "END $sp exit=$rc $(date +%T)" >> $L
    if [ $rc -eq 3 ] && grep -q "already running" $O/win11-$sp.stdout && [ $tries -lt 12 ]; then tries=$((tries+1)); sleep 300; continue; fi
    break
  done
}
run 109-coded-nospec-engine --tier screen
run 112-hip-build-macros --tier screen
wait_idle
echo "START replay 110 $(date +%F_%T)" >> $L
python3 tools/pancake/replay_turns.py bench/specs/110-coded-nospec-serve-replay.json --out $O/win11-110 > $O/win11-110.stdout 2>&1
echo "END replay 110 exit=$? $(date +%T)" >> $L
wait_idle
echo "START taskgate 111 $(date +%F_%T)" >> $L
python3 tools/pancake/task_gate.py bench/specs/111-coded-nospec-taskgate.json --out $O/win11-111.jsonl > $O/win11-111.stdout 2>&1
echo "END taskgate 111 exit=$? $(date +%T)" >> $L

systemctl --user start pancake.service; echo "service started $(date +%T)" >> $L
echo WINCHAIN11_DONE >> $L
