#!/bin/sh
# Windows-session chain U, FAST VERSION (owner order 2026-11-18, priority order; the old script is win-chainU.sh.bak): the specs 212 / 213 (statistical expert cache, LEADS.md B143 / B144,
# MOD_PLAN.md section 6 item 58) are a low-prior lead (the offline replay T-929..T-931 predicts no gain) and moved to the tail chain X (xtail-chainX.sh), which runs after V and after7.sh
# with the same screen -> confirm-for-winners protocol.  This script only keeps the queue order H2 -> S -> T -> U -> V and its completion marker WINCHAINU_DONE.
S=/home/dan/clone-of-d/src; L=$S/win-chainU.log
echo "START $(date +%F_%T)" >> $L
until grep -q WINCHAINT_DONE $S/win-chainT.log 2>/dev/null; do sleep 300; done
echo "chain T done $(date +%T)" >> $L
echo "specs 212 / 213 moved to tail chain X (xtail-chainX.log, WINCHAINX_DONE): nothing runs here $(date +%T)" >> $L
echo WINCHAINU_DONE >> $L
