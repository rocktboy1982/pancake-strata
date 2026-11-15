#!/bin/sh
cd /home/dan/clone-of-d/src/Strata
export PATH=/home/dan/clone-of-d/src/Strata/.venv/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib
echo "START 55-verify-ecap-screen $(date +%T)" >> /home/dan/auto-chain.log
python3 tools/pancake/abtest.py bench/specs/55-verify-ecap-screen.json --tier confirm --out /home/dan/auto-runs/55-verify-ecap-screen.jsonl > /home/dan/auto-runs/55-verify-ecap-screen.stdout 2>&1
echo "END 55-verify-ecap-screen exit=$? $(date +%T)" >> /home/dan/auto-chain.log
