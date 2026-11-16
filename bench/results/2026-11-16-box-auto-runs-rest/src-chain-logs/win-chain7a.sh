#!/bin/sh
# Windows-session chain 7a (daytime part of chain 7: the confirm specs 102 / 103 moved to the night chain) (MOD_PLAN.md section 5 items 1-2, 4): after chain 6 (WINCHAIN6_DONE), build strata-poly with the
# 5b867bf3a compat fix, run the confirm specs 102 / 103 (screen PROMOTE arms of TEST_LEDGER.md T-604, T-606, T-608), the
# EQUAL_TAIL TTFT A/B (item 4), then specs 30 (D7, env A/B), 33, 34 at screen.
# owner 2026-11-16: never overlap another job (3 idle minutes before every step); never stop pancake.service while it serves
# (it is stopped only after 3 minutes with no client connection on 11437).
S=/home/dan/clone-of-d/src; R=$S/Strata-win; L=$S/win-chain7a.log; O=/home/dan/auto-runs
CM=/home/linuxbrew/.linuxbrew/bin/cmake
export PATH=$R/.venv/bin:$PATH LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib
echo "START $(date +%F_%T)" >> $L
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[a]uto-chain|[a]uto-post|[i]q_avx2_parity|[t]tft_bench.py|[w]in-chain13.sh" >/dev/null; }
wait_idle() { q=0; while [ $q -lt 3 ]; do if busy; then q=0; sleep 60; else q=$((q+1)); sleep 60; fi; done; }
clients() { ss -Htn state established '( sport = :11437 )' 2>/dev/null | wc -l; }

until grep -q WINCHAIN13_DONE $S/win-chain13.log; do sleep 300; done
echo "chain 6 done $(date +%T)" >> $L
q=0; while [ $q -lt 3 ]; do if [ "$(clients)" -gt 0 ]; then q=0; echo "WAIT service has clients $(date +%T)" >> $L; else q=$((q+1)); fi; sleep 60; done
systemctl --user stop pancake.service; echo "service stopped (idle) $(date +%T)" >> $L

wait_idle
echo "BUILD build-win strata-poly $(date +%T)" >> $L
nice -n 10 $CM --build $S/build-win -j8 --target strata-poly > $S/build-win-poly.chain7.log 2>&1
echo "BUILD_END strata-poly exit=$? $(date +%T)" >> $L; grep -m8 -E "error:" $S/build-win-poly.chain7.log >> $L

cd $R
echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null
run() {  # $1 spec, rest = abtest tier flags
  sp=$1; shift; tries=0
  while :; do
    wait_idle
    echo "START $sp $(date +%F_%T)" >> $L
    python3 tools/pancake/abtest.py bench/specs/$sp.json "$@" --out $O/win7-$sp.jsonl > $O/win7-$sp.stdout 2>&1; rc=$?
    echo "END $sp exit=$rc $(date +%T)" >> $L
    if [ $rc -eq 3 ] && grep -q "already running" $O/win7-$sp.stdout && [ $tries -lt 12 ]; then tries=$((tries+1)); sleep 300; continue; fi
    break
  done
}

# MOD_PLAN.md section 5 item 4: EQUAL_TAIL TTFT, sizes off the 32768-token chunk boundaries plus the 99,997-token #5539 case
for arm in off on; do
  wait_idle
  echo "START equal-tail-$arm $(date +%F_%T)" >> $L
  if [ $arm = on ]; then export STRATA_PREFILL_EQUAL_TAIL=1; else unset STRATA_PREFILL_EQUAL_TAIL; fi
  PORT=11501 STRATA_TRACE=1 setsid ./run-pancake.sh iq2_xs fast > $O/win7-equal-tail-$arm.engine.log 2>&1 &
  sp=$!; n=0
  until curl -sf http://127.0.0.1:11501/v1/models >/dev/null 2>&1 || [ $n -ge 60 ]; do sleep 10; n=$((n+1)); done
  python3 tools/pancake/ttft_bench.py --port 11501 --scenario a --sizes 30000,34000,60000,66500,70000,99997 --runs 3 \
    --label equal-tail-$arm --out $O/win7-equal-tail-$arm.json > $O/win7-equal-tail-$arm.stdout 2>&1
  echo "END equal-tail-$arm exit=$? $(date +%T)" >> $L
  kill -TERM -$sp 2>/dev/null; sleep 20; pkill -f "serve/server.py.*--port 11501" 2>/dev/null; pkill -f "[s]trata --serve" 2>/dev/null; sleep 10
done
unset STRATA_PREFILL_EQUAL_TAIL
python3 tools/pancake/ttft_bench.py --compare $O/win7-equal-tail-off.json $O/win7-equal-tail-on.json > $O/win7-equal-tail-compare.txt 2>&1

run 30-phase18-merge --tier screen
run 33-gfx1100-mtp-bundle --tier screen
run 34-rdna3-env-resonance --tier screen

systemctl --user start pancake.service; echo "service started $(date +%T)" >> $L
echo WINCHAIN7A_DONE >> $L
