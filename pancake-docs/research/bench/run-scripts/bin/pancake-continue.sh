#!/usr/bin/env bash
# Continue the night watch on a 9router pool when the main session is out of budget.
# Usage: pancake-continue.sh [pool-or-model]   (default coding-high; any 9r pool/model, e.g. coding-mid, kr/claude-sonnet-4.5)
# Needs 9router (started here if it is down). Headless, narrow tool allowlist, no commit/push/build.
MODEL=${1:-coding-high}
if ! curl -sf -m 3 -o /dev/null http://127.0.0.1:20128/v1/models; then
  nohup 9router -n -H 127.0.0.1 --skip-update -l > /home/dan/.9router/server.log 2>&1 &
  for i in $(seq 1 30); do curl -sf -m 2 -o /dev/null http://127.0.0.1:20128/v1/models && break; sleep 2; done
fi
echo "[$(date +%F' '%T)] continuation agent started on $MODEL" >> /home/dan/pancake-night-log-2026-11-12.md
cd /home/dan && exec /home/dan/bin/claude-9r -m "$MODEL" -p "Read /home/dan/pancake-handoff-2026-11-12.md and follow it exactly. Monitor the night queue until it finishes, handle only what the handoff allows, append timestamped facts to the night log, then stop." \
  --allowedTools "Read" "Bash(tail:*)" "Bash(cat:*)" "Bash(ls:*)" "Bash(ps:*)" "Bash(date:*)" "Bash(grep:*)" "Bash(sleep:*)" "Bash(/home/dan/bin/pancake-backup.sh:*)" "Bash(echo:*)" "Bash(cd /home/dan/clone-of-d/src/Strata/.claude/worktrees/agent-ad25cdf1dc7ea704a && setsid nohup python3 tools/pancake/night_queue.py:*)" \
  < /dev/null
