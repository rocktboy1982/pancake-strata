#!/usr/bin/env bash
until grep -q 'END exit' /home/dan/abtest-2026-11-12/tonight/ttft.log 2>/dev/null; do sleep 120; done
REPO=/home/dan/clone-of-d/src/Strata/.claude/worktrees/agent-ad25cdf1dc7ea704a
echo "[$(date +%F' '%T)] START full prose set" >> /home/dan/night-queue-fullset.log
cd $REPO && python3 tools/pancake/night_queue.py --plan /home/dan/night-queue-fullset-plan.json --run-dir /home/dan/night-queue-fullset >> /home/dan/night-queue-fullset.log 2>&1
echo "[$(date +%F' '%T)] END exit=$?" >> /home/dan/night-queue-fullset.log
