#!/bin/sh
cd /home/dan/clone-of-d/src/Strata
export PATH=/home/dan/clone-of-d/src/Strata/.venv/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib
echo "START 50-rdna-16col $(date +%T)" >> /home/dan/auto-chain.log
python3 tools/pancake/abtest.py bench/specs/50-rdna-16col.json --tier confirm --out /home/dan/auto-runs/50-rdna-16col.jsonl > /home/dan/auto-runs/50-rdna-16col.stdout 2>&1
echo "END 50-rdna-16col exit=$? $(date +%T)" >> /home/dan/auto-chain.log
