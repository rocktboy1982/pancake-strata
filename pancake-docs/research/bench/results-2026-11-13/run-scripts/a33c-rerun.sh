#!/bin/sh
cd /home/dan/clone-of-d/src/Strata
export PATH=/home/dan/clone-of-d/src/Strata/.venv/bin:$PATH
rm -f /home/dan/plan-runs/a33c-confirm-router-fuse.jsonl
echo "START a33c-rerun $(date +%T)" >> /home/dan/plan-tests.log
python3 tools/pancake/abtest.py bench/specs/a33c-confirm-router-fuse.json --tier confirm --out /home/dan/plan-runs/a33c-confirm-router-fuse.jsonl > /home/dan/plan-runs/a33c-confirm-router-fuse.stdout 2>&1
echo "END a33c-rerun exit=$? $(date +%T)" >> /home/dan/plan-tests.log
echo A33C_RERUN_DONE >> /home/dan/plan-tests.log
