#!/bin/sh
cd /home/dan/clone-of-d/src
echo "QUEUED $(date +%T): waits for the RD scheduler to exit" >> run-diag-us.log
while pgrep -fx "sh ./bench-sched.v5.sh" >/dev/null 2>&1; do sleep 20; done
echo "START $(date +%T)" >> run-diag-us.log
BENCH_SCHED=1 BENCH_PREIDLED=1 BENCH_KEEP_SERVICE_DOWN=1 sh ./bench-diagUS.v1.sh; echo "END rc=$? $(date +%T)" >> run-diag-us.log
