#!/usr/bin/env bash
REPO=/home/dan/clone-of-d/src/Strata/.claude/worktrees/agent-ad25cdf1dc7ea704a
OUT=/home/dan/abtest-2026-11-12; LOG=$OUT/kern-odd.log
echo "[$(date +%F' '%T)] START odd-arms rerun" >> $LOG
cd $REPO && PATH=$REPO/.venv/bin:$PATH python3 tools/pancake/abtest.py /home/dan/abtest-specs/25-kern-hip-odd.json \
  --out $OUT/25-kern-hip-odd.jsonl --max-load 1.0 --foreign-cpu 5 --max-wait 1800 >> $OUT/25-kern-hip-odd.stdout 2>&1
echo "[$(date +%F' '%T)] END exit=$?" >> $LOG
