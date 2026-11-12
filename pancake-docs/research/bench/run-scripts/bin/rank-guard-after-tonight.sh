#!/usr/bin/env bash
REPO=/home/dan/clone-of-d/src/Strata/.claude/worktrees/agent-ad25cdf1dc7ea704a
OUT=/home/dan/abtest-2026-11-12/tonight; LOG=$OUT/rank-guard.log
until grep -q 'tonight chain done' $OUT/tonight.log 2>/dev/null; do sleep 60; done
echo "[$(date +%F' '%T)] START rank-guard (screen tier)" >> $LOG
cd $REPO && PATH=$REPO/.venv/bin:$PATH python3 tools/pancake/abtest.py /home/dan/abtest-specs/rank-guard-hip.json --tier screen \
  --out $OUT/rank-guard.jsonl --max-load 1.0 --foreign-cpu 5 --max-wait 1800 >> $OUT/rank-guard.stdout 2>&1
echo "[$(date +%F' '%T)] END exit=$?" >> $LOG
