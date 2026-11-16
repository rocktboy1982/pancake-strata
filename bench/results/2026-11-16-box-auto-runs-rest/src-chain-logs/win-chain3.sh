#!/bin/sh
# Windows-session chain 3 (MOD_PLAN.md section 5 items 5-6): after win-chain2; no spec starts after 10:00 local.
S=/home/dan/clone-of-d/src; L=$S/win-chain3.log
export PATH=/home/dan/clone-of-d/src/Strata/.venv/bin:$PATH LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib
echo "WAIT win-chain2 $(date +%T)" >> $L
until grep -q -E "WINCHAIN2_DONE|ABORT" $S/win-chain2.log 2>/dev/null; do sleep 60; done
grep -q ABORT $S/win-chain2.log && { echo "SKIP: win-chain2 aborted" >> $L; exit 1; }
cd $S/Strata-win
for sp in 71-coded-exact-backlog 72-coded-lossy-backlog 73-rdna3-prefill 74-q36 75-qsa-long-select 76-leads-exact 77-leads-lossy 26-stack-small-exact 12-spec-sampling a21-spec-lowtemp 24-a15-park-ttl 22-a7-promo-chunk 24-a45-calibrate 16-a19-hidden-dump 25-f-env-rerun a33-a35-fusion-retest 15-kmiss 19-turbo-guards; do
  while pgrep -f "[a]uto-chain1[7-9]|[r]eplay_turns.py|[q]uality_gate.py" >/dev/null; do sleep 60; done
  echo "START $sp $(date +%F_%T)" >> $L
  python3 tools/pancake/abtest.py bench/specs/$sp.json --tier screen --out /home/dan/auto-runs/win-$sp.jsonl > /home/dan/auto-runs/win-$sp.stdout 2>&1
  echo "END $sp exit=$? $(date +%T)" >> $L
done
echo WINCHAIN3_DONE >> $L
