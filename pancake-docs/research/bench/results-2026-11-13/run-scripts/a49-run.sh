#!/bin/sh
cd /home/dan/clone-of-d/src/Strata
export PATH=/home/dan/clone-of-d/src/Strata/.venv/bin:$PATH
echo "START a49 $(date +%T)" >> /home/dan/plan-tests.log
python3 tools/pancake/abtest.py bench/specs/a49-decode-context.json --tier screen --out /home/dan/plan-runs/a49.jsonl > /home/dan/plan-runs/a49.stdout 2>&1
echo "END a49 exit=$? $(date +%T)" >> /home/dan/plan-tests.log
echo A49_DONE >> /home/dan/plan-tests.log
