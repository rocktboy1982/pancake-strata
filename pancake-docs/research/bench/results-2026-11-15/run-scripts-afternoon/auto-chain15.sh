#!/bin/sh
cd /home/dan/clone-of-d/src/Strata
export PATH=/home/dan/clone-of-d/src/Strata/.venv/bin:$PATH LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib
L=/home/dan/auto-chain.log
ab() { n=$1; echo "START $n $(date +%T)" >> $L; python3 tools/pancake/abtest.py bench/specs/$n.json --tier confirm --out /home/dan/auto-runs/$n.jsonl > /home/dan/auto-runs/$n.stdout 2>&1; echo "END $n exit=$? $(date +%T)" >> $L; }
ab 58-v5-upper-bound
ab 59-ecap1-lookup16-confirm
ab 60-ecap-gated-speed
echo "START 61-qgate $(date +%T)" >> $L
python3 tools/pancake/quality_gate.py bench/specs/61-ecap-gated-quality.json --out bench/quality-hip/2026-11-15-ecap-gated --set bench/quality-hip/2026-11-15-ecap/set.filled.json > /home/dan/auto-runs/61-qgate.stdout 2>&1
echo "END 61-qgate exit=$? $(date +%T)" >> $L
echo "START 62-cache-combo $(date +%T)" >> $L
python3 tools/pancake/replay_turns.py bench/specs/62-cache-combo-replay.json --out /home/dan/replay-live/run3 > /home/dan/auto-runs/62-cache-combo.stdout 2>&1
echo "END 62-cache-combo exit=$? $(date +%T)" >> $L
echo CHAIN15_DONE >> $L
