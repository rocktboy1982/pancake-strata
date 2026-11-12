#!/usr/bin/env bash
REPO=/home/dan/clone-of-d/src/Strata/.claude/worktrees/agent-ad25cdf1dc7ea704a
OUT=/home/dan/abtest-2026-11-12/kern-pinned; LOG=/home/dan/abtest-2026-11-12/kern-pinned.log
mkdir -p $OUT
TEMP=/sys/class/drm/card1/device/hwmon/hwmon3/temp2_input
echo "[$(date +%F' '%T)] START full 25-kern-hip, perf=$(cat /sys/class/drm/card1/device/power_dpm_force_performance_level), junction guard 85C" >> $LOG
cd $REPO && PATH=$REPO/.venv/bin:$PATH python3 tools/pancake/abtest.py /home/dan/abtest-specs/25-kern-hip.json \
  --out $OUT/25-kern-hip.jsonl --max-load 1.0 --foreign-cpu 5 --max-wait 1800 >> $OUT/25-kern-hip.stdout 2>&1 &
PID=$!
while kill -0 $PID 2>/dev/null; do
  T=$(( $(cat $TEMP) / 1000 ))
  echo "[$(date +%T)] junction ${T}C" >> $LOG
  if [ "$T" -ge 85 ]; then echo "[$(date +%T)] JUNCTION ${T}C >= 85C: STOPPING run" >> $LOG; kill $PID; break; fi
  sleep 30
done
wait $PID 2>/dev/null
echo "[$(date +%F' '%T)] END exit=$?" >> $LOG
