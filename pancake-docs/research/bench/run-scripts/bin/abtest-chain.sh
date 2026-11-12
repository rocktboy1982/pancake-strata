#!/usr/bin/env bash
# Runs the untested / incomplete 2026-11-11 A/B specs AFTER the night queue exits (never in parallel).
# Output on the ROOT disk (/home/dan/abtest-2026-11-12), not on D (ntfs3).
REPO=/home/dan/clone-of-d/src/Strata/.claude/worktrees/agent-ad25cdf1dc7ea704a
OUT=/home/dan/abtest-2026-11-12; LOG=$OUT/chain.log
mkdir -p $OUT; say() { echo "[$(date +%F' '%T)] $*" >> $LOG; }
true
say "night queue exited; starting A/B chain"
cd $REPO
for spec in "10-rank-guard-fold --tier screen" "25-kern-hip" "25-a8-swaps-auto-blocking"; do
  set -- $spec; name=$1; shift
  say "START $name $*"
  PATH=$REPO/.venv/bin:$PATH python3 tools/pancake/abtest.py /home/dan/abtest-specs/$name.json --out $OUT/$name.jsonl "$@" >> $OUT/$name.stdout 2>&1
  say "END $name exit=$?"
done
say "chain finished"
