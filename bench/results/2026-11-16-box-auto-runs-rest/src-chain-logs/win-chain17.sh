#!/bin/sh
# Windows-session chain 17 (MOD_PLAN.md section 6 item 20, miss budget, tier-0 step): after chain 16 (WINCHAIN16_DONE) run
# tools/pancake/gate_importance.py on the real routing trace chain 16 records (conv_trace.py, --dump-routing). CPU only, a minute
# or two; bench/prefetch/routing.bin has no gate weights (all zero), so this is the first real-trace skip analysis (T-432 was synthetic).
S=/home/dan/clone-of-d/src; R=$S/Strata-win; L=$S/win-chain17.log; O=/home/dan/auto-runs
export PATH=$R/.venv/bin:$PATH
echo "START $(date +%F_%T)" >> $L
until grep -q WINCHAIN16A_DONE $S/win-chain16a.log 2>/dev/null; do sleep 300; done
echo "chain 16 done $(date +%T)" >> $L
cd $R
if [ -f $O/win16-conv/real1.trace ]; then
  python3 tools/pancake/gate_importance.py --dump-routing $O/win16-conv/real1.trace --budgets 0.01,0.03,0.05,0.1 --us-per-miss 48.7 > $O/win17-gate-importance.txt 2>&1
  echo "END gate_importance exit=$? $(date +%T)" >> $L
else
  echo "no real1.trace: skipped" >> $L
fi
echo WINCHAIN17_DONE >> $L
