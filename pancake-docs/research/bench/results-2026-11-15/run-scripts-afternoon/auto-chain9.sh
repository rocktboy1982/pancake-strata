#!/bin/sh
cd /home/dan/clone-of-d/src/Strata
export PATH=/home/dan/clone-of-d/src/Strata/.venv/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib
echo "START 54-pcie-frac-decode $(date +%T)" >> /home/dan/auto-chain.log
python3 tools/pancake/abtest.py bench/specs/54-pcie-frac-decode.json --tier confirm --out /home/dan/auto-runs/54-pcie-frac-decode.jsonl > /home/dan/auto-runs/54-pcie-frac-decode.stdout 2>&1
echo "END 54-pcie-frac-decode exit=$? $(date +%T)" >> /home/dan/auto-chain.log
