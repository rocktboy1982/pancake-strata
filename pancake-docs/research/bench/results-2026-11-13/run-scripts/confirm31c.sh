#!/bin/sh
while [ -z "$(grep ALL_DONE /home/dan/fk-tests.log 2>/dev/null)" ]; do sleep 20; done
cd /home/dan/clone-of-d/src/Strata
export PATH=/home/dan/clone-of-d/src/Strata/.venv/bin:$PATH
echo "START 31c-confirm $(date +%T)" >> /home/dan/fk-tests.log
python3 tools/pancake/abtest.py bench/specs/31c-confirm-aux-kvgrow.json --tier confirm --out /home/dan/fk-runs/31c-confirm.jsonl > /home/dan/fk-runs/31c-confirm.stdout 2>&1
echo "END 31c-confirm exit=$? $(date +%T)" >> /home/dan/fk-tests.log
echo CONFIRM31C_DONE >> /home/dan/fk-tests.log
