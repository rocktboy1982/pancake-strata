#!/usr/bin/env bash
until grep -q 'END exit' /home/dan/night-queue-fullset.log 2>/dev/null; do sleep 120; done
REPO=/home/dan/clone-of-d/src/Strata/.claude/worktrees/agent-ad25cdf1dc7ea704a
LOG=/home/dan/p18-runs.log
for n in p18-item2-poll1 p18-item4-fusewait; do
  echo "[$(date +%F' '%T)] START $n" >> $LOG
  cd $REPO && PATH=$REPO/.venv/bin:$PATH python3 tools/pancake/abtest.py /home/dan/abtest-specs/$n.json --out /home/dan/p18-runs/$n.jsonl --max-load 1.0 --foreign-cpu 5 --max-wait 1800 >> /home/dan/p18-runs/$n.stdout 2>&1
  echo "[$(date +%F' '%T)] END $n exit=$?" >> $LOG
done
echo "[$(date +%F' '%T)] START 17.3 ablation" >> $LOG
cd $REPO && python3 tools/pancake/ablate_greedy.py ~/abtest-specs/ablation-17-3-plan.json --out-dir /home/dan/ablation-17-3 >> /home/dan/ablation-17-3.log 2>&1
echo "[$(date +%F' '%T)] END ablation exit=$?" >> $LOG
