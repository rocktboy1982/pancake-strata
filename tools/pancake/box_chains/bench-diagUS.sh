#!/bin/sh
# bench-diagUS.sh - DIAG-ONLY rerun for the US benchmark (plain upstream v0.2.16 build and updated-strata build of bench-updatedAA.sh), prepared 2026-11-19
# after the first run's diagnostics failed: diag plain hit the chain's timeout after 18 min (2 flags) and left its abtest / server / engine running, so diag
# updated's abtest refused to start (exit 3: all 5 flags UNTESTED).  This script builds NOTHING and downloads only the updated-strata tree (for the fixed
# tools/pancake/flag_diag.py + diag_group.py), into its own folder, so the existing src-us / build-us stay untouched.
# It runs only the fixed flag_diag for both existing builds:
#   plain   /mnt/pancake/benchus/build-up/strata  STRATA_HIPBLASLT_TUNING,STRATA_PF_F16
#   updated /mnt/pancake/benchus/build-us/strata  STRATA_RDNA_Q6K,STRATA_GR_RDNA,STRATA_RDNA_MMVQ_WAVE,STRATA_COMBINE_CPU,STRATA_VERIFY_ECAP
# Each diagnostic: own session (setsid) under `timeout --signal=TERM --kill-after=100`, limit (flags + 1) x (FLAG_TIMEOUT + 90) + 180 s; afterwards the chain waits
# (<= 60 s) until no process of that session is left and SIGKILLs only that session's leftovers (DIAG-REAP line).  flag_diag.py: one abtest start per arm (the
# default once), case short, no retries, FLAG_TIMEOUT s per arm (UNTESTED-TIMEOUT past it), verdicts.json rewritten after every arm.
# Worst case about (3 + 6) x 330 s + 2 x 180 s = 56 min; normal (engine start ~70-100 s + short case) about 9 x 2.5 min = 23 min.
# Rules (standing, as bench-updatedAA.sh): box jobs never overlap (busy() sees every other job; a FROZEN AA shell is not a job and is never resumed or touched here:
# AA stays frozen); pancake.service is stopped only after 3 minutes without a client on port 11437 and comes back at the end unless BENCH_KEEP_SERVICE_DOWN=1;
# nothing outside the diagnostic's own session is killed, nothing is deleted outside this script's own folders; sources are unpacked with `tar xmf`; no sudo.
# NOT launched by default (scheduler: BENCH_SCHED=1 BENCH_PREIDLED=1 BENCH_KEEP_SERVICE_DOWN=1).  US only: Strata-rdna3 (RD) is paused since 2026-11-20, so
# standalone it no longer waits for the RD marker; it waits for 3 idle minutes (busy() still sees a running bench-updatedRD).
# Results: /home/dan/auto-runs/winbenchus/diag-rerun/ (diag-plain/, diag-updated/ with verdicts.json, *.stdout, tuned-flags-recommended.json).
# Log: /home/dan/clone-of-d/src/bench-diagUS.log; the last line is WINDIAGUS_DONE (also after an abort: the line before it is then WINDIAGUS_ABORTED).
# Start (box):  cd /home/dan/clone-of-d/src && setsid nohup sh ./bench-diagUS.sh > /dev/null 2>&1 &
S=/home/dan/clone-of-d/src; R=$S/Strata-win; L=$S/bench-diagUS.log; O=/home/dan/auto-runs/winbenchus/diag-rerun; PM=/mnt/pancake/benchus
ROCM=/opt/rocm/core-10.0
BUP=$PM/build-up; BUS=$PM/build-us; USD=$PM/src-us-diag
US_URL=https://codeload.github.com/rocktboy1982/pancake-strata/tar.gz/refs/heads/updated-strata
FLAG_TIMEOUT=${FLAG_TIMEOUT:-240}
export PATH=$ROCM/bin:$R/.venv/bin:$PATH LD_LIBRARY_PATH=$ROCM/lib HIP_VISIBLE_DEVICES=0
mkdir -p $O $USD
echo "START $(date +%F_%T)" >> $L
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[c]onv_trace.py|[a]uto-chain|[a]uto-post|[i]q_avx2_parity|[t]tft_bench.py|[w]in-chain[A-Za-z0-9]*\.sh|[x-z]tail-chain[A-Z]\.sh|[a]fter7|[b]ench-updated(AA|RD)[.a-z0-9]*\.sh" >/dev/null 2>&1; }
# a FROZEN (state T) AA shell is not a running job; an AA shell that is not stopped is
aa_live() { for p in $(pgrep -f "[r]edo-chainAA\.sh" 2>/dev/null); do case "$(ps -o stat= -p $p 2>/dev/null)" in T*) ;; *) return 0;; esac; done; return 1; }
bz() { busy || aa_live; }
QI=${BENCH_PREIDLED:-0}
wait_idle() { if [ "$QI" = 1 ]; then q=0; while [ $q -lt 2 ]; do if bz; then q=0; sleep 15; else q=$((q+1)); sleep 15; fi; done; else q=0; while [ $q -lt 3 ]; do if bz; then q=0; sleep 60; else q=$((q+1)); sleep 60; fi; done; QI=1; fi; }
service_back() { if [ "${BENCH_KEEP_SERVICE_DOWN:-0}" = 1 ]; then say "service stays down (BENCH_KEEP_SERVICE_DOWN=1) $(date +%T)"; else systemctl --user start pancake.service; say "service started $(date +%T)"; fi; }
clients() { ss -Htn state established '( sport = :11437 )' 2>/dev/null | wc -l; }
freekb() { df -Pk "$1" | awk 'NR==2{print $4}'; }
say() { echo "$*" >> $L; }
abort() { say "ERROR $*"; say "WINDIAGUS_ABORTED: $*"; service_back; echo WINDIAGUS_DONE >> $L; exit 1; }
sess_members() { ps -eo pid=,sid=,pgid=,args= 2>/dev/null | awk -v s="$1" '$2 == s || $3 == s { pid = $1; $1 = $2 = $3 = ""; sub(/^ +/, ""); print pid " " $0 }'; }
run_diag_group() {  # run_diag_group SECONDS LOG CMD...: CMD in its own session; TERM to its group at SECONDS, KILL 100 s later; then reap that session only
  secs=$1; log=$2; shift 2
  setsid timeout --signal=TERM --kill-after=100 "$secs" "$@" > "$log" 2>&1 &   # (a background job of a non-interactive sh is no group leader: setsid does not fork, $! is the session id)
  sid=$!; wait $sid; rc=$?
  n=0; while [ -n "$(sess_members $sid)" ] && [ $n -lt 30 ]; do sleep 2; n=$((n + 1)); done
  left=$(sess_members $sid)
  if [ -n "$left" ]; then
    say "DIAG-REAP session $sid still running 60 s after the diagnostic ended, killed: $(echo "$left" | cut -c1-160 | tr '\n' ';')"
    echo "$left" | while read -r p _; do kill -KILL "$p" 2>/dev/null; done; sleep 3
    [ -z "$(sess_members $sid)" ] || say "ERROR DIAG-REAP session $sid: processes survive SIGKILL: $(sess_members $sid | cut -c1-160 | tr '\n' ';')"
  fi
  return $rc
}

# ------------------------------------------------------------------ 0. service, prerequisites
if [ "${BENCH_SCHED:-0}" = 1 ]; then say "started by the scheduler: the AA shell is frozen and stays frozen here $(date +%T)"; fi
if systemctl --user is-active --quiet pancake.service; then
  q=0; while [ $q -lt 3 ]; do if [ "$(clients)" -gt 0 ]; then q=0; say "WAIT service has clients $(date +%T)"; else q=$((q+1)); fi; sleep 60; done
  systemctl --user stop pancake.service; say "service stopped (idle) $(date +%T)"
else say "service already stopped"; fi
wait_idle
[ "$(freekb /)" -ge 307200 ] || abort "/ has $(freekb /) KB free (< 300 MB)"
[ "$(freekb /mnt/pancake)" -ge 1048576 ] || abort "/mnt/pancake has $(freekb /mnt/pancake) KB free (< 1 GB); nothing was deleted"
miss=
for f in $BUP/strata $BUS/strata $R/tools/pancake/abtest.py $R/configs/pancake/flashnext-iq2_xs-plain-upstream.json $ROCM/bin/rocminfo; do [ -e "$f" ] || miss="$miss $f"; done
[ -z "$miss" ] || abort "PREREQ missing:$miss (this script builds nothing: run bench-updatedAA.sh first)"
rocminfo 2>/dev/null | grep -q -E "Name: +gfx" || abort "PREREQ rocminfo lists no gfx device"
say "PREREQ ok: both builds, harness, config, rocminfo $(date +%T)"

# ------------------------------------------------------------------ 1. the fixed flag_diag (updated-strata head) into its own folder
f=$PM/updated-strata-diag.tgz; rm -f $f.part
timeout 1800 curl -sfL -o $f.part "$US_URL" && tar tzf $f.part >/dev/null 2>&1 || abort "download of updated-strata failed ($US_URL)"
mv $f.part $f; tar xzmf $f --no-same-owner --strip-components=1 -C $USD || abort "tar of updated-strata failed"
for need in tools/pancake/flag_diag.sh tools/pancake/flag_diag.py tools/pancake/diag_group.py; do [ -f $USD/$need ] || abort "SYNC: $need is missing (the fixed flag_diag is not on updated-strata)"; done
grep -q -- "--flag-timeout" $USD/tools/pancake/flag_diag.py || abort "SYNC: flag_diag.py has no --flag-timeout (not the fixed version)"
say "SYNC updated-strata (flag_diag only) -> $USD ($(tar tzf $f | wc -l) entries)"

# ------------------------------------------------------------------ 2. the two diagnostics, one after the other
diag() {  # diag NAME EXE FLAGS
  nm=$1; exe=$2; only=$3
  nf=$(echo "$only" | tr ',' '\n' | grep -c .); budget=$(( (nf + 1) * FLAG_TIMEOUT / 60 + 1 )); lim=$(( (nf + 1) * (FLAG_TIMEOUT + 90) + 180 ))
  sed "s#\"exe\": \"[^\"]*\"#\"exe\": \"$exe\"#; s#<REPO>#$R#g" $R/configs/pancake/flashnext-iq2_xs-plain-upstream.json > $O/diag-$nm-config.json
  say "START diag $nm $(date +%F_%T) (limit $lim s: $nf flags x $FLAG_TIMEOUT s per arm + base + margin)"
  export FLAG_DIAG_ABTEST=$R/tools/pancake/abtest.py; run_diag_group $lim $O/diag-$nm.stdout sh $USD/tools/pancake/flag_diag.sh run $O/diag-$nm-config.json $O/diag-$nm \
    --budget-min $budget --flag-timeout $FLAG_TIMEOUT --only $only; rc=$?
  say "END diag $nm exit=$rc $(date +%T)"; cut -c1-200 $O/diag-$nm.stdout | tail -8 >> $L
  [ -f $O/diag-$nm/verdicts.json ] || say "ERROR diag $nm wrote no verdicts.json: $(grep -m1 . $O/diag-$nm.stdout 2>/dev/null | cut -c1-240) (log $O/diag-$nm.stdout)"
}
cd $R || abort "no $R"
diag plain $BUP/strata STRATA_HIPBLASLT_TUNING,STRATA_PF_F16
wait_idle
diag updated $BUS/strata STRATA_RDNA_Q6K,STRATA_GR_RDNA,STRATA_RDNA_MMVQ_WAVE,STRATA_COMBINE_CPU,STRATA_VERIFY_ECAP

# ------------------------------------------------------------------ 3. recommended-flags file (only exact or gate-passing GAIN flags enter it; not installed)
[ -f $O/diag-updated/verdicts.json ] && sh $USD/tools/pancake/flag_diag.sh --write-recommended $O/diag-updated/verdicts.json $O/tuned-flags-recommended.json >> $L 2>&1
say "SUMMARY verdicts in $O/diag-plain/verdicts.json and $O/diag-updated/verdicts.json; recommended (not installed): $O/tuned-flags-recommended.json $(date +%T)"
service_back
echo WINDIAGUS_DONE >> $L
