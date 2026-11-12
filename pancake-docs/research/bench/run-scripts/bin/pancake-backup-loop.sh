#!/usr/bin/env bash
# every 30 min while the night queue runs, then one final backup
PIDF=/home/dan/night-queue-run/night_queue.pid
sleep 60
while [ -f $PIDF ] && kill -0 "$(cat $PIDF 2>/dev/null)" 2>/dev/null; do
  /home/dan/bin/pancake-backup.sh periodic >/dev/null 2>&1
  for i in $(seq 1 60); do [ -f $PIDF ] && kill -0 "$(cat $PIDF 2>/dev/null)" 2>/dev/null || break; sleep 30; done
done
/home/dan/bin/pancake-backup.sh after-night >/dev/null 2>&1
