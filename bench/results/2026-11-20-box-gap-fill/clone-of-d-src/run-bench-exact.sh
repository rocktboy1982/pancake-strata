#!/bin/sh
cd /home/dan/clone-of-d/src
BENCH_BUDGET_S=6000 BENCH_NO_RESUME=1 BENCH_ONLY=EXTRA BENCH_EXTRA=./bench-exact.v1.sh exec sh ./bench-sched.v6.sh
