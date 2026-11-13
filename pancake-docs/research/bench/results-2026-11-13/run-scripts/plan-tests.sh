#!/bin/sh
cd /home/dan/clone-of-d/src/Strata
export PATH=/home/dan/clone-of-d/src/Strata/.venv/bin:$PATH
for s in 17-f9-route-fast a33-router-fuse-nodefer 24-a44-bandit; do
  echo "START $s $(date +%T)" >> /home/dan/plan-tests.log
  python3 tools/pancake/abtest.py bench/specs/$s.json --tier screen --out /home/dan/plan-runs/$s.jsonl > /home/dan/plan-runs/$s.stdout 2>&1
  echo "END $s exit=$? $(date +%T)" >> /home/dan/plan-tests.log
done
echo ALL_DONE >> /home/dan/plan-tests.log
