#!/usr/bin/env bash
# Pancake backup: all-refs bundle (root disk, verified) + copy of bundle/docs/status/notes/night-queue results on the USB.
# Usage: pancake-backup.sh [label]      Log: ~/pancake-backup.log.  Never writes to the D drive.
STAMP=$(date +%Y%m%d-%H%M%S); LABEL=${1:-auto}
REPO=/home/dan/clone-of-d/src/Strata
INT=$REPO/.claude/worktrees/agent-ad25cdf1dc7ea704a
ROOTDIR=/home/dan/backup-pancake-2026-11-11/auto-bundles
USBMNT=/run/media/dan/PANCAKE; USB=$USBMNT/pancake-archive-2026-11-12
LOG=/home/dan/pancake-backup.log
say() { echo "[$(date +%F' '%T)] $*" | tee -a $LOG; }
mkdir -p $ROOTDIR
B=$ROOTDIR/refs-$STAMP-$LABEL.bundle
if timeout 600 nice -n 19 ionice -c3 git -C $REPO bundle create $B --all >/dev/null 2>&1 && git -C $REPO bundle verify $B >/dev/null 2>&1; then
  say "bundle ok $(du -h $B | cut -f1) $B"; ls -t $ROOTDIR/*.bundle | tail -n +6 | xargs -r rm -f
else say "BUNDLE FAILED"; rm -f $B; fi
if mountpoint -q $USBMNT && mkdir -p $USB && touch $USB/.w 2>/dev/null; then
  rm -f $USB/.w; mkdir -p $USB/bundles $USB/docs-port $USB/docs-integration $USB/memory $USB/notes
  [ -f $B ] && cp -f $B $USB/bundles/ && ls -t $USB/bundles/*.bundle | tail -n +6 | xargs -r rm -f
  nice -n 19 ionice -c3 rsync -a $REPO/pancake-docs/ $USB/docs-port/ 2>>$LOG; cp -f $REPO/PANCAKE_STATUS.md $REPO/MOD_PLAN.md $USB/docs-port/ 2>>$LOG
  nice -n 19 ionice -c3 rsync -a $INT/pancake-docs/ $USB/docs-integration/ 2>>$LOG
  nice -n 19 ionice -c3 rsync -a /home/dan/.claude/projects/-home-dan/memory/ $USB/memory/ 2>>$LOG
  nice -n 19 ionice -c3 rsync -a /home/dan/.9router/pool-rationale-2026-11-11.md /home/dan/pancake-night-log-2026-11-12.md $USB/notes/ 2>>$LOG
  [ -d /home/dan/night-queue-run ] && nice -n 19 ionice -c3 rsync -a --exclude='*.gguf' /home/dan/night-queue-run/ $USB/night-queue-run/ 2>>$LOG
  sync; say "usb ok $(du -sh $USB | cut -f1) free $(df -h $USBMNT | awk 'NR==2{print $4}')"
else say "USB NOT WRITABLE - only the root-disk bundle was made"; fi
