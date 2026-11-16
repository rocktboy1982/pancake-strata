#!/bin/sh
# Windows-session chain F2 (owner 2026-11-16): after chain 15 (WINCHAIN17_DONE):
# the LONG specs of chain F: 121 (D4 suffix-index file, confirm 5 reps) + 121b (its quality gate), 123 (STRATA_CACHE_ROOT_PAST replay, T-665).
# Runs before the night confirm round (chain N waits for WINCHAINF2_DONE).
# Same rules: 3 idle minutes before every step; pancake.service stopped only after 3 minutes without a client; restarted.
S=/home/dan/clone-of-d/src; R=$S/Strata-win; L=$S/win-chainF2.log; O=/home/dan/auto-runs
CM=/home/linuxbrew/.linuxbrew/bin/cmake; CT=/home/linuxbrew/.linuxbrew/bin/ctest
export PATH=$R/.venv/bin:$PATH LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib
echo "START $(date +%F_%T)" >> $L
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[c]onv_trace.py|[a]uto-chain|[a]uto-post|[i]q_avx2_parity|[t]tft_bench.py|[w]in-chain17.sh" >/dev/null; }
wait_idle() { q=0; while [ $q -lt 3 ]; do if busy; then q=0; sleep 60; else q=$((q+1)); sleep 60; fi; done; }
clients() { ss -Htn state established '( sport = :11437 )' 2>/dev/null | wc -l; }

until grep -q WINCHAIN17_DONE $S/win-chain17.log 2>/dev/null; do sleep 300; done
echo "chain 17 done $(date +%T)" >> $L
q=0; while [ $q -lt 3 ]; do if [ "$(clients)" -gt 0 ]; then q=0; echo "WAIT service has clients $(date +%T)" >> $L; else q=$((q+1)); fi; sleep 60; done
systemctl --user stop pancake.service; echo "service stopped (idle) $(date +%T)" >> $L
cd $R; echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null
rm -f $S/d4-*.sxi
wait_idle
echo "START 121 $(date +%F_%T)" >> $L
python3 tools/pancake/abtest.py bench/specs/121-d4-suffix-index-file.json --tier confirm --block 3 --warmup 1 --outlier robust --reclaim-max 3 --reps 5 --no-early-stop --out $O/winF2-121.jsonl > $O/winF2-121.stdout 2>&1
echo "END 121 exit=$? $(date +%T)" >> $L
wait_idle
echo "START 121b quality $(date +%F_%T)" >> $L
mkdir -p $O/winF2-121b
python3 tools/pancake/quality_gate.py bench/specs/121b-d4-suffix-index-file-quality.json --out $O/winF2-121b/gate > $O/winF2-121b.stdout 2>&1
echo "END 121b exit=$? $(date +%T)" >> $L
wait_idle
echo "START 123 replay $(date +%F_%T)" >> $L
python3 tools/pancake/replay_turns.py bench/specs/123-cache-root-past.json --out $O/winF2-123 > $O/winF2-123.stdout 2>&1
echo "END 123 exit=$? $(date +%T)" >> $L
systemctl --user start pancake.service; echo "service started $(date +%T)" >> $L
echo WINCHAINF2_DONE >> $L
