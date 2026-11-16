#!/bin/sh
# Windows-session NIGHT chain (owner 2026-11-16: the short work first until night, the big confirm round tonight): after chain 17
# (WINCHAIN17_DONE) and not before 22:00 local: the confirm specs 102 / 103 (screen PROMOTE arms of T-604, T-606, T-608), spec 117
# (tier rule x swap cap), then the confirm specs that tools/pancake/promote_confirm.py writes from every chain-6 screen result
# (specs 87-101 and the old specs, T-612 ... : written at the start of this chain so it sees all of them), bench/specs/107-auto-confirm-*.json.
# Same rules: 3 idle minutes before every step; pancake.service stopped only after 3 minutes without a client; restarted.
S=/home/dan/clone-of-d/src; R=$S/Strata-win; L=$S/win-chainN.log; O=/home/dan/auto-runs
export PATH=$R/.venv/bin:$PATH LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib
echo "START $(date +%F_%T)" >> $L
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[c]onv_trace.py|[a]uto-chain|[a]uto-post|[i]q_avx2_parity|[t]tft_bench.py|[w]in-chain17.sh" >/dev/null; }
wait_idle() { q=0; while [ $q -lt 3 ]; do if busy; then q=0; sleep 60; else q=$((q+1)); sleep 60; fi; done; }
clients() { ss -Htn state established '( sport = :11437 )' 2>/dev/null | wc -l; }

until grep -q WINCHAINF2_DONE $S/win-chainF2.log 2>/dev/null; do sleep 300; done
echo "chain 17 done $(date +%T)" >> $L
until [ "$(date +%H)" -ge 22 ] || [ "$(date +%H)" -lt 6 ]; do sleep 300; done
echo "night window $(date +%T)" >> $L
q=0; while [ $q -lt 3 ]; do if [ "$(clients)" -gt 0 ]; then q=0; echo "WAIT service has clients $(date +%T)" >> $L; else q=$((q+1)); fi; sleep 60; done
systemctl --user stop pancake.service; echo "service stopped (idle) $(date +%T)" >> $L

cd $R; echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null
run() {  # $1 spec, $2 output prefix, rest = abtest flags
  sp=$1; pfx=$2; shift 2; tries=0
  while :; do
    wait_idle
    echo "START $sp $(date +%F_%T)" >> $L
    python3 tools/pancake/abtest.py bench/specs/$sp.json "$@" --out $O/$pfx-$sp.jsonl > $O/$pfx-$sp.stdout 2>&1; rc=$?
    echo "END $sp exit=$rc $(date +%T)" >> $L
    if [ $rc -eq 3 ] && grep -q "already running" $O/$pfx-$sp.stdout && [ $tries -lt 12 ]; then tries=$((tries+1)); sleep 300; continue; fi
    break
  done
}
C="--tier confirm --block 3 --warmup 1 --outlier robust --reclaim-max 3 --reps 5"
run 102-chain6-decode-confirm win7 $C
run 103-chain6-prompt-confirm win7 $C --no-early-stop
run 117-tier-swap-cap-kmiss win16 --tier confirm --block 3 --warmup 1 --outlier robust --reclaim-max 3 --reps 5 --no-early-stop

# the confirm specs of every chain-6 screen's PROMOTE arms (81 / 83 / 86 are already specs 102 / 103)
P=""
for sp in 87-coded-exact-backlog 88-coded-lossy-backlog 89-rdna3-prefill 90-q36 91-qsa-long-select 92-leads-exact 93-leads-lossy 94-pass2-exact 95-pass2-lossy 96-pass2-long 97-pass2-prompt 98-pass2-q36-xattn 99-hotrod-subfit 100-gputier-skim-serpool 101-gputier-q36-swaps 26-stack-small-exact 12-spec-sampling a21-spec-lowtemp 24-a15-park-ttl 22-a7-promo-chunk 24-a45-calibrate 16-a19-hidden-dump 25-f-env-rerun a33-a35-fusion-retest 15-kmiss 19-turbo-guards; do
  [ -f $O/win6-$sp.stdout ] && P="$P --pair $O/win6-$sp.stdout bench/specs/$sp.json"
done
python3 tools/pancake/promote_confirm.py --out-dir bench/specs --prefix 107-auto-confirm $P >> $L 2>&1
for f in bench/specs/107-auto-confirm-*.json; do
  [ -f "$f" ] || continue
  run "$(basename $f .json)" win8 $C --no-early-stop
done

systemctl --user start pancake.service; echo "service started $(date +%T)" >> $L
echo WINCHAINN_DONE >> $L
