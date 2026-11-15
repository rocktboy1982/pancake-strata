#!/bin/sh
cd /home/dan/clone-of-d/src/Strata
export PATH=/home/dan/clone-of-d/src/Strata/.venv/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib
echo "START 53-wmma-ksplit $(date +%T)" >> /home/dan/auto-chain.log
python3 tools/pancake/abtest.py bench/specs/53-wmma-ksplit.json --tier confirm --out /home/dan/auto-runs/53-wmma-ksplit.jsonl > /home/dan/auto-runs/53-wmma-ksplit.stdout 2>&1
echo "END 53-wmma-ksplit exit=$? $(date +%T)" >> /home/dan/auto-chain.log
