#!/usr/bin/env bash
# Owner confirmed memory compaction. Runs the A/B chain, then the Avalon TTFT rerun, strictly one after the other.
LOG=/home/dan/rerun-2026-11-12.log
say() { echo "[$(date +%F' '%T)] $*" >> $LOG; }
rm -rf /home/dan/abtest-2026-11-12/*.jsonl /home/dan/abtest-2026-11-12/*.stdout /home/dan/abtest-2026-11-12/chain.log
say "START A/B chain"; ~/bin/abtest-chain.sh; say "END A/B chain"
REPO=/home/dan/clone-of-d/src/Strata/.claude/worktrees/agent-ad25cdf1dc7ea704a
cd $REPO && say "START avalon rerun"
python3 tools/pancake/night_queue.py --run-dir /home/dan/night-queue-rerun-avalon --only ttft-avalon-iq2xs >> $LOG 2>&1
say "END avalon rerun exit=$?"
