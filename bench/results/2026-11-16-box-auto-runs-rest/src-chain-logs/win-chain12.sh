#!/bin/sh
# Windows-session chain 12 (MOD_PLAN.md section 5 item 2): after chain 11 (WINCHAIN11_DONE), the server-side MCP switches
# on read-only repository look-ups: python3 tools/pancake/task_gate.py bench/specs/113-mcp-taskgate.json (task set
# bench/tasks/mcp-repo.json, MCP server tools/pancake/mcp_bench_server.py).  Same service and idle rules.
S=/home/dan/clone-of-d/src; R=$S/Strata-win; L=$S/win-chain12.log; O=/home/dan/auto-runs
export PATH=$R/.venv/bin:$PATH LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib
echo "START $(date +%F_%T)" >> $L
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[a]uto-chain|[a]uto-post|[i]q_avx2_parity|[t]tft_bench.py|[w]in-chain11.sh" >/dev/null; }
wait_idle() { q=0; while [ $q -lt 3 ]; do if busy; then q=0; sleep 60; else q=$((q+1)); sleep 60; fi; done; }
clients() { ss -Htn state established '( sport = :11437 )' 2>/dev/null | wc -l; }
until grep -q WINCHAIN11_DONE $S/win-chain11.log 2>/dev/null; do sleep 300; done
echo "chain 11 done $(date +%T)" >> $L
q=0; while [ $q -lt 3 ]; do if [ "$(clients)" -gt 0 ]; then q=0; echo "WAIT service has clients $(date +%T)" >> $L; else q=$((q+1)); fi; sleep 60; done
systemctl --user stop pancake.service; echo "service stopped (idle) $(date +%T)" >> $L
cd $R; wait_idle
echo "START taskgate 113 $(date +%F_%T)" >> $L
python3 tools/pancake/task_gate.py bench/specs/113-mcp-taskgate.json --out $O/win12-113 > $O/win12-113.stdout 2>&1
echo "END taskgate 113 exit=$? $(date +%T)" >> $L
systemctl --user start pancake.service; echo "service started $(date +%T)" >> $L
echo WINCHAIN12_DONE >> $L
