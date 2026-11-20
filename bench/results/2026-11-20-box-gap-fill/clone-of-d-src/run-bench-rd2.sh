#!/bin/sh
# run-bench-rd2.sh: waits until the diag-only US rerun and the exact chain (bench-exact, run-bench-exact.sh) and every running scheduler (bench-sched*) are gone, then the v6 scheduler runs ONLY the RD chain (v5: harness with the Strata-win venv python, AA stays frozen).
cd /home/dan/clone-of-d/src || exit 1
echo "WAIT run-diag-us / exact chain $(date +%F_%T)" >> run-bench-rd2.log
while [ -n "$(pgrep -fx 'sh ./run-diag-us.sh')$(pgrep -f '[b]ench-exact')$(pgrep -fx 'sh ./run-bench-exact.sh')$(pgrep -f '[b]ench-sched')" ]; do sleep 30; done
echo "START scheduler v6 RD-only $(date +%F_%T)" >> run-bench-rd2.log
BENCH_NO_RESUME=1 BENCH_ONLY=RD BENCH_US=/bin/true BENCH_RD=./bench-updatedRD.v5.sh exec sh ./bench-sched.v6.sh
