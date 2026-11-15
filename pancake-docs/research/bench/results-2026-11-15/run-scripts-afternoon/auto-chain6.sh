#!/bin/sh
cd /home/dan/clone-of-d/src/Strata
export PATH=/home/dan/clone-of-d/src/Strata/.venv/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib
echo "START 51-rdna-16col-confirm $(date +%T)" >> /home/dan/auto-chain.log
python3 tools/pancake/abtest.py bench/specs/51-rdna-16col-confirm.json --tier confirm --out /home/dan/auto-runs/51-rdna-16col-confirm.jsonl > /home/dan/auto-runs/51-rdna-16col-confirm.stdout 2>&1
echo "END 51-rdna-16col-confirm exit=$? $(date +%T)" >> /home/dan/auto-chain.log
