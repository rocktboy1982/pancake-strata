#!/usr/bin/env bash
until grep -q 'END exit' /home/dan/night-queue-fullset.log 2>/dev/null; do sleep 120; done
REPO=/home/dan/clone-of-d/src/Strata/.claude/worktrees/agent-ad25cdf1dc7ea704a
echo "[$(date +%F' '%T)] START 17.3 ablation" >> /home/dan/ablation-17-3.log
cd $REPO && python3 tools/pancake/ablate_greedy.py ~/abtest-specs/ablation-17-3-plan.json --out-dir /home/dan/ablation-17-3 >> /home/dan/ablation-17-3.log 2>&1
echo "[$(date +%F' '%T)] END exit=$?" >> /home/dan/ablation-17-3.log
