#!/bin/sh
cd /home/dan/clone-of-d/src/Strata
export PATH=/home/dan/clone-of-d/src/Strata/.venv/bin:$PATH
echo "START a49-confirm $(date +%T)" >> /home/dan/plan-tests.log
python3 tools/pancake/abtest.py bench/specs/a49-decode-context.json --tier confirm --out /home/dan/plan-runs/a49c.jsonl > /home/dan/plan-runs/a49c.stdout 2>&1
echo "END a49-confirm exit=$? $(date +%T)" >> /home/dan/plan-tests.log
echo A49C_DONE >> /home/dan/plan-tests.log
