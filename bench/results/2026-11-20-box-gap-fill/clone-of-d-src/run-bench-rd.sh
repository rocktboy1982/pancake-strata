#!/bin/sh
cd /home/dan/clone-of-d/src
BENCH_NO_RESUME=1 BENCH_ONLY=RD BENCH_US=/bin/true BENCH_RD=./bench-updatedRD.v4.sh exec sh ./bench-sched.v5.sh
