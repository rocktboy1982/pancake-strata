#!/bin/sh
# Windows-session chain 6 (MOD_PLAN.md section 5 item 2): chain 5b stopped (TEST_LEDGER.md T-601, stale objects after tar
# syncs that kept the Windows mtimes).  The tree was re-synced with fresh mtimes; rebuild build-win (strata, strata_tests,
# strata-poly tried apart), build-win-il and build-win-j32 (strata), ctest, then specs 80-101 and the never-run specs.
# owner 2026-11-16: never overlap another job; wait for 3 idle minutes before every step (busy() matches pancake.service).
S=/home/dan/clone-of-d/src; L=$S/win-chain6.log; CT=/home/linuxbrew/.linuxbrew/bin/ctest
CM=/home/linuxbrew/.linuxbrew/bin/cmake
export PATH=/home/dan/clone-of-d/src/Strata/.venv/bin:$PATH LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib
echo "START $(date +%F_%T)" >> $L
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[a]uto-chain|[a]uto-post|[i]q_avx2_parity" >/dev/null; }
wait_idle() { q=0; while [ $q -lt 3 ]; do if busy; then q=0; echo "WAIT busy $(date +%T)" >> $L; sleep 60; else q=$((q+1)); sleep 60; fi; done; }

for b in build-win build-win-il build-win-j32; do
  wait_idle
  case $b in build-win) T="strata strata_tests";; *) T="strata";; esac
  echo "BUILD $b ($T) $(date +%T)" >> $L
  nice -n 10 $CM --build $S/$b -j8 --target $T > $S/$b.chain6.log 2>&1; rc=$?
  echo "BUILD_END $b exit=$rc $(date +%T)" >> $L
  [ $rc -ne 0 ] && grep -m5 -E "error:|undefined symbol" $S/$b.chain6.log >> $L
done
wait_idle
echo "BUILD build-win strata-poly (not blocking) $(date +%T)" >> $L
nice -n 10 $CM --build $S/build-win -j8 --target strata-poly > $S/build-win-poly.chain6.log 2>&1
echo "BUILD_END strata-poly exit=$? $(date +%T)" >> $L; grep -m5 -E "error:" $S/build-win-poly.chain6.log >> $L
wait_idle; cd $S/build-win && $CT -L cpu -j4 > $S/ctest-cpu.chain6.log 2>&1; echo "CTEST cpu exit=$? $(grep -E 'tests passed|tests failed' $S/ctest-cpu.chain6.log)" >> $L
if grep -q "BUILD_END build-win exit=0" $L; then
  wait_idle; cd $S/build-win && $CT -L gpu > $S/ctest-gpu.chain6.log 2>&1; echo "CTEST gpu exit=$? $(grep -E 'tests passed|tests failed' $S/ctest-gpu.chain6.log)" >> $L
  grep -E "\(Failed\)|\(SEGFAULT\)|\(Timeout\)" $S/ctest-gpu.chain6.log >> $L
  cd $S/Strata-win
  echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null
  for sp in 81-new-exact-switches 82-pcie-share-exact-switches 83-prompt-exact-switches 84-pf-gu-exact 85-mmvq-il-plain 86-adapt-prompt 80-mmq-rdna3-j32 87-coded-exact-backlog 88-coded-lossy-backlog 89-rdna3-prefill 90-q36 91-qsa-long-select 92-leads-exact 93-leads-lossy 94-pass2-exact 95-pass2-lossy 96-pass2-long 97-pass2-prompt 98-pass2-q36-xattn 99-hotrod-subfit 100-gputier-skim-serpool 101-gputier-q36-swaps 26-stack-small-exact 12-spec-sampling a21-spec-lowtemp 24-a15-park-ttl 22-a7-promo-chunk 24-a45-calibrate 16-a19-hidden-dump 25-f-env-rerun a33-a35-fusion-retest 15-kmiss 19-turbo-guards; do
    tries=0
    while :; do
      wait_idle
      # spec 99: the subfit learner keeps its state beside the expert profile; a cold start for the run (its note)
      [ "$sp" = 99-hotrod-subfit ] && rm -f $S/Strata-win/data/subfit-learn-*.bin
      echo "START $sp $(date +%F_%T)" >> $L
      python3 tools/pancake/abtest.py bench/specs/$sp.json --tier screen --out /home/dan/auto-runs/win6-$sp.jsonl > /home/dan/auto-runs/win6-$sp.stdout 2>&1; rc=$?
      echo "END $sp exit=$rc $(date +%T)" >> $L
      if [ $rc -eq 3 ] && grep -q "already running" /home/dan/auto-runs/win6-$sp.stdout && [ $tries -lt 12 ]; then tries=$((tries+1)); echo "BUSY retry $tries in 300 s" >> $L; sleep 300; continue; fi
      break
    done
  done
fi
systemctl --user start pancake.service; echo "service started $(date +%T)" >> $L
echo WINCHAIN6_DONE >> $L
