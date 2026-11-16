#!/bin/sh
# Windows-session chain 15 (MOD_PLAN.md section 6 items 19 and 26): after chain 14 (WINCHAIN14_DONE) apply chain15-stage.tar
# (STRATA_ADAPT_PROTECT, tier_protect_test, spec 116), rebuild build-win (strata strata_tests), ctest tier_protect_test,
# then spec 116 at screen. The generate.cpp edit was never compiled whole on Windows: a build failure skips the spec.
# Same rules: 3 idle minutes before every step; pancake.service stopped only after 3 minutes without a client; restarted.
S=/home/dan/clone-of-d/src; R=$S/Strata-win; L=$S/win-chain15.log; O=/home/dan/auto-runs
CM=/home/linuxbrew/.linuxbrew/bin/cmake; CT=/home/linuxbrew/.linuxbrew/bin/ctest
export PATH=$R/.venv/bin:$PATH LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib
echo "START $(date +%F_%T)" >> $L
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[a]uto-chain|[a]uto-post|[i]q_avx2_parity|[t]tft_bench.py|[w]in-chain8a.sh" >/dev/null; }
wait_idle() { q=0; while [ $q -lt 3 ]; do if busy; then q=0; sleep 60; else q=$((q+1)); sleep 60; fi; done; }
clients() { ss -Htn state established '( sport = :11437 )' 2>/dev/null | wc -l; }

until grep -q WINCHAIN8A_DONE $S/win-chain8a.log 2>/dev/null; do sleep 300; done
echo "chain 14 done $(date +%T)" >> $L
q=0; while [ $q -lt 3 ]; do if [ "$(clients)" -gt 0 ]; then q=0; echo "WAIT service has clients $(date +%T)" >> $L; else q=$((q+1)); fi; sleep 60; done
systemctl --user stop pancake.service; echo "service stopped (idle) $(date +%T)" >> $L

cd $R && tar xmf $S/chain15-stage.tar && echo "stage applied: $(tar tf $S/chain15-stage.tar | wc -l) files" >> $L
wait_idle
echo "BUILD build-win (strata strata_tests) $(date +%T)" >> $L
nice -n 10 $CM --build $S/build-win -j8 --target strata strata_tests tier_protect_test > $S/build-win.chain15.log 2>&1; rc=$?
echo "BUILD_END build-win exit=$rc $(date +%T)" >> $L
[ $rc -ne 0 ] && grep -m8 -E "error:|undefined" $S/build-win.chain15.log >> $L
if [ $rc -eq 0 ]; then
  wait_idle; cd $S/build-win && $CT -R tier_protect --output-on-failure > $S/ctest-protect.chain15.log 2>&1; echo "CTEST tier_protect exit=$? $(grep -E 'tests passed|tests failed' $S/ctest-protect.chain15.log)" >> $L
  wait_idle; $CT -L cpu -j4 > $S/ctest-cpu.chain15.log 2>&1; echo "CTEST cpu exit=$? $(grep -E 'tests passed|tests failed' $S/ctest-cpu.chain15.log)" >> $L
  cd $R; echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null
  tries=0
  while :; do
    wait_idle
    echo "START 116 $(date +%F_%T)" >> $L
    python3 tools/pancake/abtest.py bench/specs/116-tier-protect.json --tier screen --out $O/win15-116-tier-protect.jsonl > $O/win15-116-tier-protect.stdout 2>&1; rc2=$?
    echo "END 116 exit=$rc2 $(date +%T)" >> $L
    if [ $rc2 -eq 3 ] && grep -q "already running" $O/win15-116-tier-protect.stdout && [ $tries -lt 12 ]; then tries=$((tries+1)); sleep 300; continue; fi
    break
  done
else
  echo "build failed: spec 116 skipped" >> $L
fi
systemctl --user start pancake.service; echo "service started $(date +%T)" >> $L
echo WINCHAIN15_DONE >> $L
