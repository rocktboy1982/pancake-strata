#!/bin/sh
# Final benchmark queue (owner 2026-11-19: only benchmarks; AA stays frozen). Runs one scheduler after the other.
cd /home/dan/clone-of-d/src
L=/home/dan/clone-of-d/src/run-queue-final.log
say() { echo "$*" >> $L; }
say "QUEUED $(date +%T): waits until the running RD chain and its scheduler are gone"
while pgrep -fx "sh ./[b]ench-updatedRD.v5.sh" >/dev/null 2>&1 || pgrep -f "[b]ench-sched" >/dev/null 2>&1; do sleep 20; done
say "STEP 1 exact US part $(date +%T)"
EXACT_PARTS=US BENCH_BUDGET_S=6000 BENCH_NO_RESUME=1 BENCH_ONLY=EXTRA BENCH_EXTRA=./bench-exact.v3.sh sh ./bench-sched.v7.sh; say "END 1 rc=$? $(date +%T)"
say "STEP 2 RD full-preset rerun $(date +%T)"
BENCH_NO_RESUME=1 BENCH_ONLY=RD BENCH_US=/bin/true BENCH_RD=./bench-updatedRD.v6.sh sh ./bench-sched.v7.sh; say "END 2 rc=$? $(date +%T)"
say "STEP 3 exact RD part $(date +%T)"
EXACT_PARTS=RD BENCH_BUDGET_S=3000 BENCH_NO_RESUME=1 BENCH_ONLY=EXTRA BENCH_EXTRA=./bench-exact.v3.sh sh ./bench-sched.v7.sh; say "END 3 rc=$? $(date +%T)"
say "STEP 4 RD task gate $(date +%T)"
BENCH_NO_RESUME=1 BENCH_ONLY=EXTRA BENCH_EXTRA=./bench-rdgate.v1.sh sh ./bench-sched.v7.sh; say "END 4 rc=$? $(date +%T)"
say "QUEUE_DONE $(date +%T)"
