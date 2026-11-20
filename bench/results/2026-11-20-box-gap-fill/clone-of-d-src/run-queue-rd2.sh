#!/bin/sh
# RD queue 2 (owner order 2026-11-19 late: the 20:28 RD run was aborted, its configs were the pancake preset instead of Strata-rdna3's own setup).
# Runs one scheduler after the other: (1) the fixed RD chain (config-plain = Strata-rdna3's iq3-1gpu.json exactly, config-updated = that + our remaining
# flags, no value of theirs overridden), (2) the exact chain's RD part, (3) the RD task gate.  AA stays frozen (BENCH_NO_RESUME=1); nothing is killed.
cd /home/dan/clone-of-d/src || exit 1
L=/home/dan/clone-of-d/src/run-queue-rd2.log
say() { echo "$*" >> $L; }
say "QUEUED $(date +%F_%T) pid $$: waits until no other scheduler or benchmark chain runs"
while pgrep -f "[b]ench-sched" >/dev/null 2>&1 || pgrep -f "[b]ench-(updated(AA|RD)|exact|rdgate|diagUS)[.a-z0-9]*\.sh" >/dev/null 2>&1; do sleep 20; done
say "STEP 1 RD benchmark (bench-updatedRD.v7.sh) $(date +%T)"
BENCH_NO_RESUME=1 BENCH_ONLY=RD BENCH_US=/bin/true BENCH_RD=./bench-updatedRD.v7.sh sh ./bench-sched.v7.sh; say "END 1 rc=$? $(date +%T)"
say "STEP 2 exact RD part (bench-exact.v3.sh) $(date +%T)"
EXACT_PARTS=RD BENCH_BUDGET_S=3000 BENCH_NO_RESUME=1 BENCH_ONLY=EXTRA BENCH_EXTRA=./bench-exact.v3.sh sh ./bench-sched.v7.sh; say "END 2 rc=$? $(date +%T)"
say "STEP 3 RD task gate (bench-rdgate.v2.sh) $(date +%T)"
BENCH_NO_RESUME=1 BENCH_ONLY=EXTRA BENCH_EXTRA=./bench-rdgate.v2.sh sh ./bench-sched.v7.sh; say "END 3 rc=$? $(date +%T)"
say "QUEUE_DONE $(date +%T)"
