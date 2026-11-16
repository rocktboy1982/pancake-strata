#!/bin/sh
# Windows-session chain 16a (daytime part of chain 16: the real conversation trace and its V1-V4; spec 117 moved to the night chain) (MOD_PLAN.md section 6 items 19 and 26): after chain 15 (WINCHAIN15_DONE) run
#  A. spec 117 (tier rule vs k-th-miss at swap caps 32 / 64 / 96) at confirm, --no-early-stop;
#  B. a REAL routing trace of interleaved multi-turn conversations (tools/pancake/conv_trace.py, ~10 min, build-win engine with
#     --dump-routing), then tier_laws.py --validate-conv on it with its requests.json (V1-V4 on real turns, not the synthetic
#     interleave of T-618).
# Same rules: 3 idle minutes before every step; pancake.service stopped only after 3 minutes without a client; restarted.
S=/home/dan/clone-of-d/src; R=$S/Strata-win; L=$S/win-chain16a.log; O=/home/dan/auto-runs
export PATH=$R/.venv/bin:$PATH LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib
echo "START $(date +%F_%T)" >> $L
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[c]onv_trace.py|[a]uto-chain|[a]uto-post|[i]q_avx2_parity|[t]tft_bench.py|[w]in-chain14.sh" >/dev/null; }
wait_idle() { q=0; while [ $q -lt 3 ]; do if busy; then q=0; sleep 60; else q=$((q+1)); sleep 60; fi; done; }
clients() { ss -Htn state established '( sport = :11437 )' 2>/dev/null | wc -l; }

until grep -q WINCHAIN14_DONE $S/win-chain14.log 2>/dev/null; do sleep 300; done
echo "chain 15 done $(date +%T)" >> $L
q=0; while [ $q -lt 3 ]; do if [ "$(clients)" -gt 0 ]; then q=0; echo "WAIT service has clients $(date +%T)" >> $L; else q=$((q+1)); fi; sleep 60; done
systemctl --user stop pancake.service; echo "service stopped (idle) $(date +%T)" >> $L

cd $R; echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null

# B. the resolved preset of the checkout the box agent serves, pointed at the build-win engine
mkdir -p $O/win16-conv
sed 's#/Strata/build-hip/strata#/build-win/strata#' $S/Strata/pancake-flashnext-iq2_xs-fast-hip.json > $O/win16-conv/config.json
wait_idle
echo "START conv_trace $(date +%F_%T)" >> $L
python3 tools/pancake/conv_trace.py $O/win16-conv/config.json $O/win16-conv --max-new 384 --tag real1 > $O/win16-conv/conv_trace.stdout 2>&1
echo "END conv_trace exit=$? $(date +%T)" >> $L
if [ -f $O/win16-conv/real1.trace ] && [ -f $O/win16-conv/real1.requests.json ]; then
  wait_idle
  python3 tools/pancake/tier_laws.py data/expert-profile-pancake.bin 10909 $O/win16-conv/real1.trace --turns $O/win16-conv/real1.requests.json --validate-conv --policies tier kmiss:2,4 kmiss:2,8 phase:4,4 conv:1.0 --cap 32 --no-holdout --json $O/win16-conv/validate.json > $O/win16-conv/validate.txt 2>&1
  echo "END tier_laws validate exit=$? $(date +%T)" >> $L
else
  echo "conv_trace wrote no trace: validation skipped" >> $L
fi

systemctl --user start pancake.service; echo "service started $(date +%T)" >> $L
echo WINCHAIN16A_DONE >> $L
