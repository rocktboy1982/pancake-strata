#!/bin/sh
cd /home/dan/clone-of-d/src/Strata/.claude/worktrees/agent-ad25cdf1dc7ea704a
export PATH=/home/dan/clone-of-d/src/Strata/.claude/worktrees/agent-ad25cdf1dc7ea704a/.venv/bin:$PATH
for s in 28-phase18-draft 29-phase18-device-plan; do
  echo "START $s $(date +%T)" >> /home/dan/remaining-tests.log
  python3 tools/pancake/abtest.py bench/specs/$s.json --tier screen --out /home/dan/p18-runs/$s.jsonl > /home/dan/p18-runs/$s.stdout 2>&1
  echo "END $s exit=$? $(date +%T)" >> /home/dan/remaining-tests.log
done
cd /home/dan/clone-of-d/build-int
echo "START copy-streams $(date +%T)" >> /home/dan/remaining-tests.log
ctest -R '^conversation_copy_streams_test$' --output-on-failure > /home/dan/copy-streams.log 2>&1; echo "END copy-streams exit=$?" >> /home/dan/remaining-tests.log
echo "START flag-order $(date +%T)" >> /home/dan/remaining-tests.log
ctest -R '^hip_flag_order_test$' --output-on-failure > /home/dan/flag-order.log 2>&1; echo "END flag-order exit=$?" >> /home/dan/remaining-tests.log
echo "START ctest-gpu $(date +%T)" >> /home/dan/remaining-tests.log
ctest -L gpu -j1 --timeout 240 > /home/dan/ctest-gpu-final.log 2>&1; echo "END ctest-gpu exit=$?" >> /home/dan/remaining-tests.log
echo ALL_DONE >> /home/dan/remaining-tests.log
