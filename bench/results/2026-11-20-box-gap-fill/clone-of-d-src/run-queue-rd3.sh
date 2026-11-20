#!/bin/sh
# RD queue 3 (2026-11-19 late; TEST_LEDGER.md T-987): the 21:23 queue (run-queue-rd2.sh) measured nothing: the plain smoke died
# ("strata serve: mtp: writing the draft head": their --mtp-q4 all cannot rewrite the IQ2_XS model's IQ4_XS head in place as Q4_0),
# and steps 2 and 3 still ran on the failed RD run.  bench_updatedRD_conf.py (updated-strata-rdna3, synced by the chain) now
# substitutes --mtp-q4 all -> proj for a head below Q4_0, logged as a CONFIG SUBST line.
# Same three steps as run-queue-rd2.sh: (1) RD chain bench-updatedRD.v8.sh, (2) exact chain RD part, (3) RD task gate.
# NEW: the queue STOPS after step 1 unless that RD run ended with WINBENCHRD_DONE and no WINBENCHRD_ABORTED (the chain exits 0 either way).
# AA stays frozen (BENCH_NO_RESUME=1); nothing is killed.
cd /home/dan/clone-of-d/src || exit 1
L=/home/dan/clone-of-d/src/run-queue-rd3.log
RL=/home/dan/clone-of-d/src/bench-updatedRD.log
say() { echo "$*" >> $L; }
say "QUEUED $(date +%F_%T) pid $$: waits until no other scheduler or benchmark chain runs"
while pgrep -f "[b]ench-sched" >/dev/null 2>&1 || pgrep -f "[b]ench-(updated(AA|RD)|exact|rdgate|diagUS)[.a-z0-9]*\.sh" >/dev/null 2>&1; do sleep 20; done
n0=$(wc -l < $RL 2>/dev/null || echo 0)
say "STEP 1 RD benchmark (bench-updatedRD.v8.sh) $(date +%T) (bench-updatedRD.log from line $((n0 + 1)))"
BENCH_NO_RESUME=1 BENCH_ONLY=RD BENCH_US=/bin/true BENCH_RD=./bench-updatedRD.v8.sh sh ./bench-sched.v7.sh; say "END 1 rc=$? $(date +%T)"
new=$(tail -n +$((n0 + 1)) $RL 2>/dev/null)
if echo "$new" | grep -q "WINBENCHRD_ABORTED" || ! echo "$new" | grep -q "WINBENCHRD_DONE"; then
  say "STOP after step 1: the RD run aborted or did not finish ($(echo "$new" | grep -m1 "WINBENCHRD_ABORTED" | cut -c1-300)); steps 2 and 3 NOT run $(date +%T)"
  say "QUEUE_STOPPED $(date +%T)"
  exit 1
fi
say "STEP 2 exact RD part (bench-exact.v3.sh) $(date +%T)"
EXACT_PARTS=RD BENCH_BUDGET_S=3000 BENCH_NO_RESUME=1 BENCH_ONLY=EXTRA BENCH_EXTRA=./bench-exact.v3.sh sh ./bench-sched.v7.sh; say "END 2 rc=$? $(date +%T)"
say "STEP 3 RD task gate (bench-rdgate.v2.sh) $(date +%T)"
BENCH_NO_RESUME=1 BENCH_ONLY=EXTRA BENCH_EXTRA=./bench-rdgate.v2.sh sh ./bench-sched.v7.sh; say "END 3 rc=$? $(date +%T)"
say "QUEUE_DONE $(date +%T)"
