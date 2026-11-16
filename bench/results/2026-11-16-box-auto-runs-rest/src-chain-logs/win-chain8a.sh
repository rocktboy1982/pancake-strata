#!/bin/sh
# Windows-session chain 8a (daytime part of chain 8: the 107-auto confirm specs moved to the night chain) (MOD_PLAN.md section 5 item 2): after chain 7 (WINCHAIN7_DONE) apply the staged 2026-11-16
# afternoon code (PARK_ASYNC on HIP, status-page HIP label / AMD name, promote_confirm.py, specs 104-106), rebuild build-win,
# ctest, run spec 104 (D6) at confirm, 18-f22 at screen, the replays 105 / 106, then the confirm specs that
# tools/pancake/promote_confirm.py writes from chain 6's screen PROMOTE arms (specs 87-101 and the old specs).
# owner 2026-11-16: never overlap another job (3 idle minutes before every step); never stop pancake.service while it serves.
S=/home/dan/clone-of-d/src; R=$S/Strata-win; L=$S/win-chain8a.log; O=/home/dan/auto-runs
CM=/home/linuxbrew/.linuxbrew/bin/cmake; CT=/home/linuxbrew/.linuxbrew/bin/ctest
export PATH=$R/.venv/bin:$PATH LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib
echo "START $(date +%F_%T)" >> $L
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[a]uto-chain|[a]uto-post|[i]q_avx2_parity|[t]tft_bench.py|[w]in-chain7a.sh" >/dev/null; }
wait_idle() { q=0; while [ $q -lt 3 ]; do if busy; then q=0; sleep 60; else q=$((q+1)); sleep 60; fi; done; }
clients() { ss -Htn state established '( sport = :11437 )' 2>/dev/null | wc -l; }

until grep -q WINCHAIN7A_DONE $S/win-chain7a.log 2>/dev/null; do sleep 300; done
echo "chain 7 done $(date +%T)" >> $L
q=0; while [ $q -lt 3 ]; do if [ "$(clients)" -gt 0 ]; then q=0; echo "WAIT service has clients $(date +%T)" >> $L; else q=$((q+1)); fi; sleep 60; done
systemctl --user stop pancake.service; echo "service stopped (idle) $(date +%T)" >> $L

cd $R && tar xmf $S/chain8-stage.tar && echo "stage applied: $(tar tf $S/chain8-stage.tar | wc -l) files" >> $L

wait_idle
echo "BUILD build-win (strata strata_tests) $(date +%T)" >> $L
nice -n 10 $CM --build $S/build-win -j8 --target strata strata_tests > $S/build-win.chain8.log 2>&1; rc=$?
echo "BUILD_END build-win exit=$rc $(date +%T)" >> $L
[ $rc -ne 0 ] && grep -m8 -E "error:|undefined" $S/build-win.chain8.log >> $L
wait_idle; cd $S/build-win && $CT -L cpu -j4 > $S/ctest-cpu.chain8.log 2>&1; echo "CTEST cpu exit=$? $(grep -E 'tests passed|tests failed' $S/ctest-cpu.chain8.log)" >> $L
[ $rc -eq 0 ] || { systemctl --user start pancake.service; echo "build failed: specs skipped; service started" >> $L; echo WINCHAIN8A_DONE >> $L; exit 1; }
wait_idle; cd $S/build-win && $CT -L gpu > $S/ctest-gpu.chain8.log 2>&1; echo "CTEST gpu exit=$? $(grep -E 'tests passed|tests failed' $S/ctest-gpu.chain8.log)" >> $L
grep -E "\(Failed\)|\(SEGFAULT\)|\(Timeout\)" $S/ctest-gpu.chain8.log >> $L

cd $R

echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null
run() {  # $1 spec, rest = abtest flags
  sp=$1; shift; tries=0
  while :; do
    wait_idle
    echo "START $sp $(date +%F_%T)" >> $L
    python3 tools/pancake/abtest.py bench/specs/$sp.json "$@" --out $O/win8-$sp.jsonl > $O/win8-$sp.stdout 2>&1; rc=$?
    echo "END $sp exit=$rc $(date +%T)" >> $L
    if [ $rc -eq 3 ] && grep -q "already running" $O/win8-$sp.stdout && [ $tries -lt 12 ]; then tries=$((tries+1)); sleep 300; continue; fi
    break
  done
}
replay() {
  wait_idle
  echo "START replay $1 $(date +%F_%T)" >> $L
  python3 tools/pancake/replay_turns.py bench/specs/$1.json --out $O/win8-$1 > $O/win8-$1.stdout 2>&1
  echo "END replay $1 exit=$? $(date +%T)" >> $L
}
C="--tier confirm --block 3 --warmup 1 --outlier robust --reclaim-max 3 --reps 9"
run 104-rdna-kq-cols-d6 $C
run 18-f22-wmma-q4-gfx11 --tier screen
replay 105-park-async-hip-replay
replay 106-serve-levers-replay

systemctl --user start pancake.service; echo "service started $(date +%T)" >> $L
echo WINCHAIN8A_DONE >> $L
