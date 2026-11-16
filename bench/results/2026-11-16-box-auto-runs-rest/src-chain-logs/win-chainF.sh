#!/bin/sh
# Windows-session chain F (owner 2026-11-16): after chain 15 (WINCHAIN15_DONE) and the stage tar chainF-stage.tar.ready:
# apply chainF-stage.tar (the seven fork bug fixes STRATA_CONV_DISK_WANTS / CACHE_ROOT_PAST / WINDOW_EVICT_LONG / CAPTURE_BEAT /
# DISK_NO_DRAFT, KEEPALIVE_DATA, half-tile MoE, miss-budget and tier code of the R&D agent), rebuild build-win, run the ctests
# (TEST_LEDGER.md T-651..T-661), then the SHORT specs: 119 (PF_COMBINE_WRITE alone, T-630 fault, engine logs kept), 118
# (half-tile), 120 (D3, 2 runs), 122 (D5).  The long ones (121/121b D4, 123 cache root) are chain F2 after chain 17.
# Same rules: 3 idle minutes before every step; pancake.service stopped only after 3 minutes without a client; restarted.
S=/home/dan/clone-of-d/src; R=$S/Strata-win; L=$S/win-chainF.log; O=/home/dan/auto-runs
CM=/home/linuxbrew/.linuxbrew/bin/cmake; CT=/home/linuxbrew/.linuxbrew/bin/ctest
export PATH=$R/.venv/bin:$PATH LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib
echo "START $(date +%F_%T)" >> $L
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[c]onv_trace.py|[a]uto-chain|[a]uto-post|[i]q_avx2_parity|[t]tft_bench.py|[w]in-chain15.sh" >/dev/null; }
wait_idle() { q=0; while [ $q -lt 3 ]; do if busy; then q=0; sleep 60; else q=$((q+1)); sleep 60; fi; done; }
clients() { ss -Htn state established '( sport = :11437 )' 2>/dev/null | wc -l; }
until grep -q WINCHAIN15_DONE $S/win-chain15.log 2>/dev/null; do sleep 300; done
until [ -f $S/chainF-stage.tar.ready ]; do sleep 300; done
echo "chain 15 done and stage ready $(date +%T)" >> $L
q=0; while [ $q -lt 3 ]; do if [ "$(clients)" -gt 0 ]; then q=0; echo "WAIT service has clients $(date +%T)" >> $L; else q=$((q+1)); fi; sleep 60; done
systemctl --user stop pancake.service; echo "service stopped (idle) $(date +%T)" >> $L
cd $R && tar xmf $S/chainF-stage.tar && echo "stage applied: $(tar tf $S/chainF-stage.tar | wc -l) files" >> $L
wait_idle
echo "BUILD build-win $(date +%T)" >> $L
nice -n 10 $CM --build $S/build-win -j8 --target strata strata_tests hip_combine_write_parity > $S/build-win.chainF.log 2>&1; rc=$?
echo "BUILD_END exit=$rc $(date +%T)" >> $L
[ $rc -ne 0 ] && grep -m10 -E "error:|undefined" $S/build-win.chainF.log >> $L
if [ $rc -eq 0 ]; then
  cd $S/build-win
  wait_idle; $CT -R "fork_bugfix|conversation_snapshot|moe_tile|hip_combine_write|miss_budget|tier_|disk_env" --output-on-failure > $S/ctest-fix.chainF.log 2>&1; echo "CTEST fixes exit=$? $(grep -E 'tests passed|tests failed' $S/ctest-fix.chainF.log)" >> $L
  wait_idle; $CT -L cpu -j4 > $S/ctest-cpu.chainF.log 2>&1; echo "CTEST cpu exit=$? $(grep -E 'tests passed|tests failed' $S/ctest-cpu.chainF.log)" >> $L
  cd $R; echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null
  run() {  # $1 spec, rest = abtest flags
    sp=$1; shift; tries=0
    while :; do
      wait_idle
      echo "START $sp $(date +%F_%T)" >> $L
      python3 tools/pancake/abtest.py bench/specs/$sp.json "$@" --out $O/winF-$sp.jsonl > $O/winF-$sp.stdout 2>&1; rc2=$?
      echo "END $sp exit=$rc2 $(date +%T)" >> $L
      if [ $rc2 -eq 3 ] && grep -q "already running" $O/winF-$sp.stdout && [ $tries -lt 12 ]; then tries=$((tries+1)); sleep 300; continue; fi
      break
    done
  }
  run 119-pf-combine-write-alone --tier screen
  run 118-f16-var-bm --tier screen
  run 120-d3-defer-fast --tier confirm --reps 2 --no-early-stop
  run 122-d5-pf-fused-f16 --tier screen
else
  echo "build failed: specs skipped" >> $L
fi
# owner 2026-11-16: move the unused Qwen3.8-27B GGUF (native engine cannot load arch qwen35; nothing references it) to the USB drive
# PANCAKE (/mnt/pancake): copy, compare sha256, delete the original only when equal.
M=/home/dan/clone-of-d/models/ista-qwen38-27b-iq2xs; U=/mnt/pancake/models-moved-2026-11-16/ista-qwen38-27b-iq2xs; G=Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf
if mountpoint -q /mnt/pancake && [ -f $M/$G ]; then
  wait_idle; echo "USB move START $(date +%T)" >> $L
  mkdir -p $U && nice -n 19 ionice -c3 cp -p $M/* $U/
  a=$(nice -n 19 sha256sum $M/$G | cut -d' ' -f1); b=$(nice -n 19 sha256sum $U/$G | cut -d' ' -f1)
  if [ -n "$a" ] && [ "$a" = "$b" ]; then rm -f $M/$G $M/download.log; rmdir $M 2>/dev/null; echo "USB move OK sha256 $a; original removed $(date +%T)" >> $L
  else echo "USB move FAILED or hash differs (a=$a b=$b): original kept $(date +%T)" >> $L; fi
else echo "USB move skipped: /mnt/pancake not mounted or file gone" >> $L; fi
systemctl --user start pancake.service; echo "service started $(date +%T)" >> $L
echo WINCHAINF_DONE >> $L
