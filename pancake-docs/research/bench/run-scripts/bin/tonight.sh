#!/usr/bin/env bash
# Tonight's A/B run: waits until 23:00, then runs three specs one after another (quiet gates, junction guard 85C).
REPO=/home/dan/clone-of-d/src/Strata/.claude/worktrees/agent-ad25cdf1dc7ea704a
OUT=/home/dan/abtest-2026-11-12/tonight; LOG=$OUT/tonight.log; TEMP=/sys/class/drm/card1/device/hwmon/hwmon3/temp2_input
mkdir -p $OUT
say() { echo "[$(date +%F' '%T)] $*" >> $LOG; }
until [ "$(date +%H%M)" -ge 2300 ] && [ "$(date +%H%M)" -lt 2400 ]; do sleep 30; done
say "start 23:00, perf=$(cat /sys/class/drm/card1/device/power_dpm_force_performance_level)"
for name in recheck-kern confirm-kern confirm-a8; do
  say "START $name"
  cd $REPO && PATH=$REPO/.venv/bin:$PATH python3 tools/pancake/abtest.py /home/dan/abtest-specs/$name.json \
    --out $OUT/$name.jsonl --max-load 1.0 --foreign-cpu 5 --max-wait 1800 >> $OUT/$name.stdout 2>&1 &
  PID=$!
  while kill -0 $PID 2>/dev/null; do
    T=$(( $(cat $TEMP) / 1000 )); say "junction ${T}C"
    if [ "$T" -ge 85 ]; then say "JUNCTION ${T}C: STOPPING $name"; kill $PID; break; fi
    sleep 60
  done
  wait $PID 2>/dev/null; say "END $name exit=$?"
done
say "tonight chain done"
