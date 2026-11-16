#!/bin/sh
# Windows-session night chain (MOD_PLAN.md section 5 item 2). Starts only after the box chain16 (CHAIN16_DONE).
C=/home/linuxbrew/.linuxbrew/bin/cmake; CT=/home/linuxbrew/.linuxbrew/bin/ctest
S=/home/dan/clone-of-d/src; L=$S/win-chain2.log
export PATH=/home/dan/clone-of-d/src/Strata/.venv/bin:$PATH LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib
step() { n=$1; shift; echo "START $n $(date +%F_%T)" >> $L; "$@" > /home/dan/auto-runs/win-$n.stdout 2>&1; rc=$?; echo "END $n exit=$rc $(date +%T)" >> $L; return $rc; }
echo "WAIT chain16 $(date +%T)" >> $L
until grep -q CHAIN16_DONE /home/dan/auto-chain16.log 2>/dev/null; do sleep 60; done
while pgrep -f "[a]btest.py|[r]eplay_turns.py|[q]uality_gate.py|[a]uto-chain1[7-9]" >/dev/null; do sleep 60; done
cd $S
step build-win $C --build build-win --target strata strata_tests -j6 || { echo "ABORT after $s" >> $L; exit 1; }
step build-il $C --build build-win-il --target strata strata_tests mmvq_il_parity -j6 || { echo "ABORT after $s" >> $L; exit 1; }
step build-j32 $C --build build-win-j32 --target strata -j6 || { echo "ABORT after $s" >> $L; exit 1; }
step ctest-cpu $CT --test-dir build-win -L cpu -j4 --output-on-failure || { echo "ABORT after $s" >> $L; exit 1; }
step ctest-gpu $CT --test-dir build-win -L gpu --output-on-failure --timeout 900
step il-parity sh -c "cd build-win-il && ./mmvq_il_parity"
cd $S/Strata-win
for sp in 65-new-exact-switches 66-pcie-share-exact-switches 67-prompt-exact-switches 68-pf-gu-exact 69-mmvq-il-plain 70-adapt-prompt 64-mmq-rdna3-j32; do
  step $sp python3 tools/pancake/abtest.py bench/specs/$sp.json --tier screen --out /home/dan/auto-runs/win-$sp.jsonl
done
echo WINCHAIN2_DONE >> $L
