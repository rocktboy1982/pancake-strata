#!/bin/sh
cd /home/dan/clone-of-d/src
BENCH_NO_RESUME=1 BENCH_ONLY=US BENCH_US=./bench-updatedAA.v3.sh BENCH_RD=./bench-updatedRD.v3.sh exec sh ./bench-sched.v4.sh
