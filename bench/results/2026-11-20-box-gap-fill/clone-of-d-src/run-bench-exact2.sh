#!/bin/sh
# run-bench-exact2.sh: waits until the RD re-run (bench-updatedRD.v5.sh, run-bench-rd2.sh) and every running scheduler (bench-sched*) are gone, then the v6 scheduler runs ONLY the exact chain v2 (harness with the Strata-win venv python, AA stays frozen).
# The patterns use [x] brackets so that no pgrep command line of this loop contains a chain name another busy() check would match.
cd /home/dan/clone-of-d/src || exit 1
echo "WAIT RD re-run / schedulers $(date +%F_%T)" >> run-bench-exact2.log
while [ -n "$(pgrep -fx 'sh ./[b]ench-updatedRD.v5.sh')$(pgrep -fx 'sh ./[r]un-bench-rd2.sh')$(pgrep -f '[b]ench-sched')" ]; do sleep 30; done
echo "START scheduler v6 EXTRA (exact v2) $(date +%F_%T)" >> run-bench-exact2.log
BENCH_BUDGET_S=6000 BENCH_NO_RESUME=1 BENCH_ONLY=EXTRA BENCH_EXTRA=./bench-exact.v2.sh exec sh ./bench-sched.v6.sh
