#!/bin/sh
cd /home/dan/clone-of-d/src/Strata
export PATH=/home/dan/clone-of-d/src/Strata/.venv/bin:$PATH
for s in 31-env-both-gpu 32-env-prompt-gpu; do
  echo "START $s $(date +%T)" >> /home/dan/fk-tests.log
  python3 tools/pancake/abtest.py bench/specs/$s.json --tier screen --out /home/dan/fk-runs/$s.jsonl > /home/dan/fk-runs/$s.stdout 2>&1
  echo "END $s exit=$? $(date +%T)" >> /home/dan/fk-tests.log
done
echo ALL_DONE >> /home/dan/fk-tests.log
