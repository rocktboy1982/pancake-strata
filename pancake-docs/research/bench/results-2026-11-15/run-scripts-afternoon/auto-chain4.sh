#!/bin/sh
cd /home/dan/clone-of-d/src/Strata
export PATH=/home/dan/clone-of-d/src/Strata/.venv/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib
echo "START 49-cache-replay-confirm $(date +%T)" >> /home/dan/auto-chain.log
python3 tools/pancake/replay_turns.py bench/specs/49-cache-replay-confirm.json --out /home/dan/replay-live/run2 > /home/dan/auto-runs/49-cache-replay-confirm.stdout 2>&1
echo "END 49-cache-replay-confirm exit=$? $(date +%T)" >> /home/dan/auto-chain.log
echo CHAIN4_DONE >> /home/dan/auto-chain.log
