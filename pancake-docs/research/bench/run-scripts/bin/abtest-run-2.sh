#!/usr/bin/env bash
REPO=/home/dan/clone-of-d/src/Strata/.claude/worktrees/agent-ad25cdf1dc7ea704a
OUT=/home/dan/abtest-2026-11-12; LOG=$OUT/run2.log
rm -rf $OUT/25-kern-hip $OUT/25-a8-swaps-auto-blocking $OUT/*.jsonl $OUT/*.stdout
cd $REPO
for name in 25-kern-hip 25-a8-swaps-auto-blocking; do
  echo "[$(date +%F' '%T)] START $name" >> $LOG
  PATH=$REPO/.venv/bin:$PATH python3 tools/pancake/abtest.py /home/dan/abtest-specs/$name.json --out $OUT/$name.jsonl >> $OUT/$name.stdout 2>&1
  echo "[$(date +%F' '%T)] END $name exit=$?" >> $LOG
done
echo "[$(date +%F' '%T)] done" >> $LOG
