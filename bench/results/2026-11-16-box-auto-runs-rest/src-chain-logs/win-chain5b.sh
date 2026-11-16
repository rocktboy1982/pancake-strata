#!/bin/sh
# 5b: copy of win-chain5.sh by the Linux session 2026-11-16 10:08 - builds only the targets it needs (strata, strata_tests): `all` now holds strata-poly, which does not compile on HIP (cuda.h), so every build exited 1 and the gpu ctest + specs were skipped
# Windows-session chain 5 (MOD_PLAN.md section 5 item 2): chain4 died in the 2026-11-16 08:26 reboot.
# Build the synced tree (build-win, build-win-il, build-win-j32), ctest, then specs 80-98 and the never-run specs.
# owner 2026-11-16: never overlap another job; never stop pancake.service while it serves (busy() matches it).
S=/home/dan/clone-of-d/src; L=$S/win-chain5.log; CT=/home/linuxbrew/.linuxbrew/bin/ctest
CM=/home/linuxbrew/.linuxbrew/bin/cmake
export PATH=/home/dan/clone-of-d/src/Strata/.venv/bin:$PATH LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib
echo "START $(date +%F_%T)" >> $L
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[a]uto-chain|[a]uto-post|[i]q_avx2_parity" >/dev/null; }
wait_idle() { q=0; while [ $q -lt 3 ]; do if busy; then q=0; echo "WAIT busy $(date +%T)" >> $L; sleep 60; else q=$((q+1)); sleep 60; fi; done; }

for b in build-win build-win-il build-win-j32; do
  wait_idle
  echo "BUILD $b $(date +%T)" >> $L
  case $b in build-win) T="strata strata_tests";; *) T="strata";; esac; nice -n 10 $CM --build $S/$b -j8 --target $T > $S/$b.chain5.log 2>&1; rc=$?
  echo "BUILD_END $b exit=$rc $(date +%T)" >> $L
  [ $rc -ne 0 ] && grep -m5 -E "error:" $S/$b.chain5.log >> $L
done
wait_idle; cd $S/build-win && $CT -L cpu -j4 > $S/ctest-cpu.chain5.log 2>&1; echo "CTEST cpu exit=$? $(tail -3 $S/ctest-cpu.chain5.log | head -1)" >> $L
if grep -q "BUILD_END build-win exit=0" $L; then
  wait_idle; cd $S/build-win && $CT -L gpu > $S/ctest-gpu.chain5.log 2>&1; echo "CTEST gpu exit=$? $(tail -3 $S/ctest-gpu.chain5.log | head -1)" >> $L
  cd $S/Strata-win
  echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null
  for sp in 81-new-exact-switches 82-pcie-share-exact-switches 83-prompt-exact-switches 84-pf-gu-exact 85-mmvq-il-plain 86-adapt-prompt 80-mmq-rdna3-j32 87-coded-exact-backlog 88-coded-lossy-backlog 89-rdna3-prefill 90-q36 91-qsa-long-select 92-leads-exact 93-leads-lossy 94-pass2-exact 95-pass2-lossy 96-pass2-long 97-pass2-prompt 98-pass2-q36-xattn 26-stack-small-exact 12-spec-sampling a21-spec-lowtemp 24-a15-park-ttl 22-a7-promo-chunk 24-a45-calibrate 16-a19-hidden-dump 25-f-env-rerun a33-a35-fusion-retest 15-kmiss 19-turbo-guards; do
    tries=0
    while :; do
      wait_idle
      echo "START $sp $(date +%F_%T)" >> $L
      python3 tools/pancake/abtest.py bench/specs/$sp.json --tier screen --out /home/dan/auto-runs/win5-$sp.jsonl > /home/dan/auto-runs/win5-$sp.stdout 2>&1; rc=$?
      echo "END $sp exit=$rc $(date +%T)" >> $L
      if [ $rc -eq 3 ] && grep -q "already running" /home/dan/auto-runs/win5-$sp.stdout && [ $tries -lt 12 ]; then tries=$((tries+1)); echo "BUSY retry $tries in 300 s" >> $L; sleep 300; continue; fi
      break
    done
  done
fi
systemctl --user start pancake.service; echo "service started $(date +%T)" >> $L
echo WINCHAIN5_DONE >> $L
