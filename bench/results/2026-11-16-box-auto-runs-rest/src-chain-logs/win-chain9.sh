#!/bin/sh
# Windows-session chain 9 (MOD_PLAN.md section 5 items 8, 9): after chain 8 (WINCHAIN8_DONE): the never-run probes 4.2.1
# (empty_grid_probe) and 4.2.4 (q2_rows_bench), then the TTFT measurements M1 (single long prompts with
# STRATA_PREFILL_TIMING=1), M2 (scenario C: side requests + a sub-agent switch, 3 sessions) and M5 (scenario E: warm deltas
# at 32K) on the daily preset (build-win).  Same rules: 3 idle minutes before every step; pancake.service stopped only
# after 3 minutes without a client; restarted at the end.
S=/home/dan/clone-of-d/src; R=$S/Strata-win; L=$S/win-chain9.log; O=/home/dan/auto-runs
CM=/home/linuxbrew/.linuxbrew/bin/cmake
export PATH=$R/.venv/bin:$PATH LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib
echo "START $(date +%F_%T)" >> $L
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[a]uto-chain|[a]uto-post|[i]q_avx2_parity|[t]tft_bench.py|[w]in-chain15.sh" >/dev/null; }
wait_idle() { q=0; while [ $q -lt 3 ]; do if busy; then q=0; sleep 60; else q=$((q+1)); sleep 60; fi; done; }
clients() { ss -Htn state established '( sport = :11437 )' 2>/dev/null | wc -l; }

until grep -q WINCHAINF_DONE $S/win-chainF.log 2>/dev/null; do sleep 300; done
echo "chain 8 done $(date +%T)" >> $L
q=0; while [ $q -lt 3 ]; do if [ "$(clients)" -gt 0 ]; then q=0; echo "WAIT service has clients $(date +%T)" >> $L; else q=$((q+1)); fi; sleep 60; done
systemctl --user stop pancake.service; echo "service stopped (idle) $(date +%T)" >> $L

# probes 4.2.1 / 4.2.4 (targets are EXCLUDE_FROM_ALL: turning the option on adds nothing to `strata`)
wait_idle
echo "BUILD probes $(date +%T)" >> $L
$CM -DSTRATA_BUILD_PROBES=ON $S/build-win > $S/build-win-probes.chain9.log 2>&1 && \
  nice -n 10 $CM --build $S/build-win -j8 --target q2_rows_bench empty_grid_probe >> $S/build-win-probes.chain9.log 2>&1
echo "BUILD_END probes exit=$? $(date +%T)" >> $L
wait_idle
echo "START probe 4.2.4 q2_rows_bench $(date +%T)" >> $L
taskset -c 3 $S/build-win/q2_rows_bench > $O/win9-q2_rows_bench.txt 2>&1; echo "END q2_rows_bench exit=$? $(date +%T)" >> $L
wait_idle
echo "START probe 4.2.1 empty_grid $(date +%T)" >> $L
HIP_VISIBLE_DEVICES=0 $S/build-win/empty_grid_probe --csv $O/win9-empty-grid.csv > $O/win9-empty-grid.txt 2>&1; rc=$?
python3 $R/tools/pancake/probes/launch_shapes.py --static --sweep $O/win9-empty-grid.csv >> $O/win9-empty-grid.txt 2>&1
echo "END empty_grid exit=$rc $(date +%T)" >> $L

# TTFT M1 / M2 / M5 on one served daily preset
cd $R
wait_idle
echo "START serve for M1/M2/M5 $(date +%F_%T)" >> $L
EL=$O/win9-ttft.engine.log
PORT=11502 STRATA_PREFILL_TIMING=1 setsid ./run-pancake.sh iq2_xs fast > $EL 2>&1 &
sp=$!; n=0
until curl -sf http://127.0.0.1:11502/v1/models >/dev/null 2>&1 || [ $n -ge 60 ]; do sleep 10; n=$((n+1)); done
T="python3 tools/pancake/ttft_bench.py --port 11502 --prefill-timing-log $EL"
$T --scenario a --sizes 8k,32k,64k,100k --runs 3 --label m1 --out $O/win9-ttft-m1.json > $O/win9-ttft-m1.stdout 2>&1
echo "END M1 exit=$? $(date +%T)" >> $L
$T --scenario c --sessions 3 --side-after 3,6 --switch-after 5 --label m2 --out $O/win9-ttft-m2.json > $O/win9-ttft-m2.stdout 2>&1
echo "END M2 exit=$? $(date +%T)" >> $L
$T --scenario e --delta-base 32k --label m5-32k --out $O/win9-ttft-m5-32k.json > $O/win9-ttft-m5-32k.stdout 2>&1
echo "END M5 32k exit=$? $(date +%T)" >> $L
$T --scenario e --delta-base 64k --label m5-64k --out $O/win9-ttft-m5-64k.json > $O/win9-ttft-m5-64k.stdout 2>&1
echo "END M5 64k exit=$? $(date +%T)" >> $L
kill -TERM -$sp 2>/dev/null; sleep 20; pkill -f "serve/server.py.*--port 11502" 2>/dev/null; pkill -f "[s]trata --serve" 2>/dev/null; sleep 10

systemctl --user start pancake.service; echo "service started $(date +%T)" >> $L
echo WINCHAIN9_DONE >> $L
