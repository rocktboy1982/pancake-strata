#!/bin/sh
# After the night chain N (was: after chain 7; the Windows session re-ordered the queue 2026-11-16 evening, MOD_PLAN section 5) (Linux session, 2026-11-16; MOD_PLAN section 5 "Queue changes"): no sudo anywhere.
#  A rebuild build-hip (Strata, the pulled tree)   B ctest -L cpu   C mmvq_il_parity with STRATA_MMVQ_IL_PLAIN=1 (T-602, kernel alone)
#  D spec 85 il-plain with serialized kernels + verify trace (T-602)   E spec 88's six unrun arms, screen (T-613)
#  F spec 88 adapt-idle-copy alone, serialized + trace (T-613)   G restart pancake.service
# Not here: the quality gate and the confirm specs of 87-101 (they need chain 7's confirm results first).
# Waits for WINCHAINN_DONE, then for 3 minutes with no client on 11437 before it stops the service (as chain 7 does).
S=/home/dan/clone-of-d/src; R=$S/Strata-win; T=$S/Strata; L=/home/dan/auto-runs/after7.log; O=/home/dan/auto-runs/after7
export PATH=$T/.venv/bin:$PATH LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib
CM=/home/linuxbrew/.linuxbrew/bin/cmake; CT=/home/linuxbrew/.linuxbrew/bin/ctest
echo "QUEUED $(date +%F_%T)" >> $L
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[a]uto-chain|[a]uto-post|[t]tft_bench.py|[w]in-chain" >/dev/null; }
wait_idle() { q=0; while [ $q -lt 3 ]; do if busy; then q=0; echo "WAIT busy $(date +%T)" >> $L; sleep 60; else q=$((q+1)); sleep 60; fi; done; }
clients() { ss -Htn state established '( sport = :11437 )' 2>/dev/null | wc -l; }
until grep -q WINCHAINN_DONE $S/win-chainN.log 2>/dev/null; do sleep 300; done
echo "night chain N done $(date +%T)" >> $L
q=0; while [ $q -lt 3 ]; do if [ "$(clients)" -gt 0 ]; then q=0; echo "WAIT service has clients $(date +%T)" >> $L; else q=$((q+1)); fi; sleep 60; done
systemctl --user stop pancake.service; echo "service stopped $(date +%T)" >> $L
wait_idle
echo "A build-hip $(date +%T)" >> $L
cd $T && nice -n 10 $CM --build build-hip -j8 -- -k 0 > $O/build-hip.log 2>&1; echo "A_END exit=$? $(date +%T)" >> $L; grep -m6 -E "^FAILED|error:" $O/build-hip.log >> $L
wait_idle
echo "B ctest cpu $(date +%T)" >> $L
cd $T/build-hip && $CT -L cpu -j4 > $O/ctest-cpu.log 2>&1; echo "B_END exit=$? $(grep -E 'tests passed' $O/ctest-cpu.log)" >> $L
wait_idle
echo "C mmvq_il_parity IL_PLAIN $(date +%T)" >> $L
cd $S/build-win-il && STRATA_MMVQ_IL_PLAIN=1 HIP_LAUNCH_BLOCKING=1 ./mmvq_il_parity --selftest > $O/mmvq_il_parity-plain.log 2>&1; echo "C_END exit=$? $(tail -n 2 $O/mmvq_il_parity-plain.log | tr '\n' ' ' | cut -c1-160)" >> $L
cd $R
run() { sp=$1; shift; tries=0
  while :; do wait_idle; echo "START $sp $(date +%F_%T)" >> $L
    env "$@" python3 tools/pancake/abtest.py $O/$sp.json --tier screen --out $O/$sp.jsonl > $O/$sp.stdout 2>&1; rc=$?
    echo "END $sp exit=$rc $(date +%T)" >> $L
    if [ $rc -eq 3 ] && grep -q "already running" $O/$sp.stdout && [ $tries -lt 12 ]; then tries=$((tries+1)); sleep 300; continue; fi
    break; done; }
run 85-il-plain-diag HIP_LAUNCH_BLOCKING=1 AMD_SERIALIZE_KERNEL=3 STRATA_VERIFY_TRACE=1
run 88-idle-copy-diag HIP_LAUNCH_BLOCKING=1 AMD_SERIALIZE_KERNEL=3 STRATA_VERIFY_TRACE=1
run 88-rerun X=1
systemctl --user start pancake.service; echo "service started $(date +%T)" >> $L
echo AFTER7_DONE >> $L
