#!/bin/sh
# Windows-session chain S (owner 2026-11-16): state-saving cost (spec 128, T-690, MOD_PLAN item 37) after the night confirm round
# (WINCHAINN_DONE): the decode/prompt screen of the six arms, then the returning-conversation replay (6 arms x 3 reps, ~4 h).
# FAST VERSION (owner order 2026-11-18, box hours; the old script is win-chainS.sh.bak): replay 1 rep of every arm, then 2 more reps of the decisive pair
# (base, no-state) and of the single arms only when no-state resolves (the replay run-to-run spread is ~10%, TEST_LEDGER.md: 3 reps of six arms cannot resolve 2% anyway); quick idle between steps.
# Same rules: 3 idle minutes before the first step; pancake.service stopped only after 3 minutes without a client; restarted.
S=/home/dan/clone-of-d/src; R=$S/Strata-win; L=$S/win-chainS.log; O=/home/dan/auto-runs
export PATH=$R/.venv/bin:$PATH LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib
echo "START $(date +%F_%T)" >> $L
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[c]onv_trace.py|[a]uto-chain|[a]uto-post|[i]q_avx2_parity|[t]tft_bench.py|[w]in-chainH2.sh" >/dev/null; }
wait_idle() { if [ "${QI:-0}" = 1 ]; then q=0; while [ $q -lt 2 ]; do if busy; then q=0; sleep 15; else q=$((q+1)); sleep 15; fi; done; else q=0; while [ $q -lt 3 ]; do if busy; then q=0; sleep 60; else q=$((q+1)); sleep 60; fi; done; QI=1; fi; }   # quick idle (owner order 2026-11-18, box hours): the full 3 minutes only the first time, 30 s between this chain's own later steps; busy() still sees every other job
clients() { ss -Htn state established '( sport = :11437 )' 2>/dev/null | wc -l; }
until grep -q WINCHAINH2_DONE $S/win-chainH2.log 2>/dev/null; do sleep 300; done
echo "chain N done $(date +%T)" >> $L
q=0; while [ $q -lt 3 ]; do if [ "$(clients)" -gt 0 ]; then q=0; echo "WAIT service has clients $(date +%T)" >> $L; else q=$((q+1)); fi; sleep 60; done
systemctl --user stop pancake.service; echo "service stopped (idle) $(date +%T)" >> $L
cd $R; echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null
wait_idle
echo "START 128 screen $(date +%F_%T)" >> $L
python3 tools/pancake/abtest.py bench/specs/128-state-saving-cost.json --tier screen --out $O/winS-128.jsonl > $O/winS-128.stdout 2>&1
echo "END 128 screen exit=$? $(date +%T)" >> $L
wait_idle
echo "START 128 replay $(date +%F_%T)" >> $L
python3 tools/pancake/replay_turns.py bench/specs/128-state-saving-cost.json --reps 1 --out $O/winS-128-replay > $O/winS-128-replay.stdout 2>&1
echo "END 128 replay (6 arms x 1 rep) exit=$? $(date +%T)" >> $L
wait_idle
echo "START 128 replay2 base,no-state +2 reps $(date +%F_%T)" >> $L
python3 tools/pancake/replay_turns.py bench/specs/128-state-saving-cost.json --reps 2 --arms base,no-state --out $O/winS-128-replay2 > $O/winS-128-replay2.stdout 2>&1
echo "END 128 replay2 exit=$? $(date +%T)" >> $L
grep -E "no-state .*(better|worse|no difference)" $O/winS-128-replay2.stdout | cut -c1-200 >> $L
if grep -Eq "no-state .*-> (better|worse)" $O/winS-128-replay2.stdout; then
  wait_idle
  echo "START 128 replay3 single arms +2 reps (no-state resolved) $(date +%F_%T)" >> $L
  python3 tools/pancake/replay_turns.py bench/specs/128-state-saving-cost.json --reps 2 --arms base,no-conv-cache,no-ckpt-periodic,no-prompt-cache,disk-on --out $O/winS-128-replay3 > $O/winS-128-replay3.stdout 2>&1
  echo "END 128 replay3 exit=$? $(date +%T)" >> $L
else echo "128 replay3 skipped: no-state does not resolve against base at 3 reps; the single arms cannot either" >> $L; fi
python3 tools/pancake/state_cost.py --replay-stdout $O/winS-128-replay.stdout > $O/winS-128-statecost.txt 2>&1
[ -s $O/winS-128-replay2.stdout ] && python3 tools/pancake/state_cost.py --replay-stdout $O/winS-128-replay2.stdout > $O/winS-128-statecost2.txt 2>&1
systemctl --user start pancake.service; echo "service started $(date +%T)" >> $L
echo WINCHAINS_DONE >> $L
