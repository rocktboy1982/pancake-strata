#!/bin/sh
# Windows-session chain 13 (MOD_PLAN.md section 7, defects T-602 and T-613): after chain 12 (WINCHAIN12_DONE) the six
# one-arm diagnostic specs 114a-f at screen, each its own abtest run so one crash ends only that spec.
# Same rules: 3 idle minutes before every step; pancake.service stopped only after 3 minutes without a client; restarted.
S=/home/dan/clone-of-d/src; R=$S/Strata-win; L=$S/win-chain13.log; O=/home/dan/auto-runs
export PATH=$R/.venv/bin:$PATH LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib
echo "START $(date +%F_%T)" >> $L
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[a]uto-chain|[a]uto-post|[i]q_avx2_parity|[t]tft_bench.py|[w]in-chain6.sh" >/dev/null; }
wait_idle() { q=0; while [ $q -lt 3 ]; do if busy; then q=0; sleep 60; else q=$((q+1)); sleep 60; fi; done; }
clients() { ss -Htn state established '( sport = :11437 )' 2>/dev/null | wc -l; }

until grep -q WINCHAIN6_DONE $S/win-chain6.log 2>/dev/null; do sleep 300; done
echo "chain 12 done $(date +%T)" >> $L
q=0; while [ $q -lt 3 ]; do if [ "$(clients)" -gt 0 ]; then q=0; echo "WAIT service has clients $(date +%T)" >> $L; else q=$((q+1)); fi; sleep 60; done
systemctl --user stop pancake.service; echo "service stopped (idle) $(date +%T)" >> $L

cd $R; echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null
run() {  # $1 spec, rest = abtest flags
  sp=$1; shift; tries=0
  while :; do
    wait_idle
    echo "START $sp $(date +%F_%T)" >> $L
    python3 tools/pancake/abtest.py bench/specs/$sp.json "$@" --out $O/win13-$sp.jsonl > $O/win13-$sp.stdout 2>&1; rc=$?
    echo "END $sp exit=$rc $(date +%T)" >> $L
    if [ $rc -eq 3 ] && grep -q "already running" $O/win13-$sp.stdout && [ $tries -lt 12 ]; then tries=$((tries+1)); sleep 300; continue; fi
    break
  done
}
for sp in 114a-idle-copy-signal-host 114b-idle-copy-one-slice 114c-idle-copy-trace 114f-gr-rdna0-alone 114e-il-plain-no-rdna-read 114d-il-plain-check; do
  run $sp --tier screen
done

systemctl --user start pancake.service; echo "service started $(date +%T)" >> $L
echo WINCHAIN13_DONE >> $L
