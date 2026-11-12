#!/usr/bin/env bash
REPO=/home/dan/clone-of-d/src/Strata/.claude/worktrees/agent-ad25cdf1dc7ea704a
OUT=/home/dan/abtest-2026-11-12; LOG=$OUT/kern-redo.log
while ! grep -q 'done' $OUT/run2.log 2>/dev/null; do sleep 20; done
echo "[$(date +%F' '%T)] START 25-kern-hip redo (quiet gates: max-load 1.0, foreign-cpu 5)" >> $LOG
cd $REPO && PATH=$REPO/.venv/bin:$PATH python3 tools/pancake/abtest.py /home/dan/abtest-specs/25-kern-hip.json \
  --out $OUT/25-kern-hip-redo.jsonl --max-load 1.0 --foreign-cpu 5 --max-wait 1800 >> $OUT/25-kern-hip-redo.stdout 2>&1
echo "[$(date +%F' '%T)] END exit=$?" >> $LOG
