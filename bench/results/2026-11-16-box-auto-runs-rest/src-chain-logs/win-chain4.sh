#!/bin/sh
# Windows-session chain 4 (MOD_PLAN.md section 5 item 2): the re-run of win-chain2/3 that the box post17 job blocked.
S=/home/dan/clone-of-d/src; L=$S/win-chain4.log; CT=/home/linuxbrew/.linuxbrew/bin/ctest
export PATH=/home/dan/clone-of-d/src/Strata/.venv/bin:$PATH LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib
echo "START $(date +%F_%T)" >> $L

# owner 2026-11-16: never overlap another job. Wait until no build, test, benchmark or engine runs on the box (ours or
# not) and the box agent's queue scripts are idle, for 3 minutes in a row.
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[a]uto-chain|[a]uto-post|[i]q_avx2_parity" >/dev/null; }
wait_idle() { q=0; while [ $q -lt 3 ]; do if busy; then q=0; echo "WAIT busy $(date +%T)" >> $L; sleep 60; else q=$((q+1)); sleep 60; fi; done; }
wait_idle
systemctl --user stop pancake.service; echo "service stopped $(date +%T)" >> $L; sleep 20
cd $S
cd $S/Strata-win
echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null
for sp in 81-new-exact-switches 82-pcie-share-exact-switches 83-prompt-exact-switches 84-pf-gu-exact 85-mmvq-il-plain 86-adapt-prompt 80-mmq-rdna3-j32 87-coded-exact-backlog 88-coded-lossy-backlog 89-rdna3-prefill 90-q36 91-qsa-long-select 92-leads-exact 93-leads-lossy 26-stack-small-exact 12-spec-sampling a21-spec-lowtemp 24-a15-park-ttl 22-a7-promo-chunk 24-a45-calibrate 16-a19-hidden-dump 25-f-env-rerun a33-a35-fusion-retest 15-kmiss 19-turbo-guards; do
  tries=0
  while :; do
    wait_idle
    echo "START $sp $(date +%F_%T)" >> $L
    python3 tools/pancake/abtest.py bench/specs/$sp.json --tier screen --out /home/dan/auto-runs/win4-$sp.jsonl > /home/dan/auto-runs/win4-$sp.stdout 2>&1; rc=$?
    echo "END $sp exit=$rc $(date +%T)" >> $L
    if [ $rc -eq 3 ] && grep -q "already running" /home/dan/auto-runs/win4-$sp.stdout && [ $tries -lt 12 ]; then tries=$((tries+1)); echo "BUSY retry $tries in 300 s" >> $L; sleep 300; continue; fi
    break
  done
done
systemctl --user start pancake.service; echo "service started $(date +%T)" >> $L
echo WINCHAIN4_DONE >> $L
