#!/bin/sh
L=/home/dan/auto-chain.log
until grep -q 'END qgate-ecap' "$L" 2>/dev/null; do sleep 30; done
cd /home/dan/clone-of-d/src/Strata
export PATH=/home/dan/clone-of-d/src/Strata/.venv/bin:$PATH LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib
echo "START 57-ecap-stack $(date +%T)" >> $L
python3 tools/pancake/abtest.py bench/specs/57-ecap-stack.json --tier confirm --out /home/dan/auto-runs/57-ecap-stack.jsonl > /home/dan/auto-runs/57-ecap-stack.stdout 2>&1
echo "END 57-ecap-stack exit=$? $(date +%T)" >> $L
