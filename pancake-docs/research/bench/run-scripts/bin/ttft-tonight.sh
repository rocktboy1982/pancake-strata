#!/usr/bin/env bash
until grep -q 'END exit' /home/dan/abtest-2026-11-12/tonight/rank-guard.log 2>/dev/null; do sleep 60; done
REPO=/home/dan/clone-of-d/src/Strata/.claude/worktrees/agent-ad25cdf1dc7ea704a
echo "[$(date +%F' '%T)] START ttft baseline rerun" >> /home/dan/abtest-2026-11-12/tonight/ttft.log
cd $REPO && python3 tools/pancake/night_queue.py --run-dir /home/dan/night-queue-rerun-ttft --only ttft-pancake-iq2xs-a >> /home/dan/abtest-2026-11-12/tonight/ttft.log 2>&1
echo "[$(date +%F' '%T)] END exit=$?" >> /home/dan/abtest-2026-11-12/tonight/ttft.log
