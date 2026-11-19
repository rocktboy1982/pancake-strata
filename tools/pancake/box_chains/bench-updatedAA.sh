#!/bin/sh
# bench-updatedAA.sh - benchmark on the box: the plain upstream Strata release (built with ITS OWN default flags) against the
# updated-strata branch (optimised flags of tools/pancake/build_updated.sh, adopted preset), same workloads, serialized.
# Owner order 2026-11-19 (autonomy).  The file name has no "win-chain" so that no other busy() pattern ever sees this script (T-944).
#
# Rules (standing): box jobs never overlap (busy() sees every other job; 3 idle minutes before the first step, 30 s between this
# chain's own later steps); nothing is killed or deleted (previous binaries are copied to bak/ first, a failed build stops the chain
# and leaves them in place); pancake.service is stopped only after 3 minutes without a client on port 11437 and started again at the end
# (also on abort); the sudoers file and permissions are never touched; code is synced with `tar xmf` semantics (mtimes reset, else ninja
# keeps stale objects); the disk / is 93% full, so every source tree and build lives under /mnt/pancake/benchus, and a step stops with ERROR
# when /mnt/pancake has < 2 GB free (or / < 300 MB for the logs).  Every failed step logs ERROR and stops; nothing is skipped silently.
# Start: normally launched by bench-sched.sh (BENCH_SCHED=1, BENCH_PREIDLED=1, BENCH_KEEP_SERVICE_DOWN=1): the scheduler has frozen the AA shell (SIGSTOP), waited
# for the box to be idle (3 minutes) and resumes AA after the benchmarks.  Standalone (no BENCH_SCHED) it still waits for WINCHAINAA_DONE as before.
# SHORT PLAN (owner order 2026-11-19: not multiple hours; target <= 2 h of box time, hard budget BENCH_BUDGET_S = 7200 s, optional passes skipped when it is over).
# Two-pass protocol: spec 220 (short, edit; decode) is a SCREEN first (1 start per arm + the base again, drift check); only when `updated` shows a benefit
# (>= 1.0 % lossless / >= 1.5 % lossy, chainT-decide.py) the CONFIRM follows, 3 reps (paired SD short 3.2 %, edit 2.7 %: 3 reps give +-1.8 %).  Spec 221 (long set
# 10000 / 32000 / 64000 / 100000 tokens) is ONE screen rep per arm without the drift arm (paired SD of a long prompt 1.3 %: one rep is decisive for the gap
# between a plain and a tuned build); its 2-rep confirm runs only when 70 minutes of the budget are still left (normally it does not).
# Builds: plain and updated tree one after the other, ninja -j8 (the rest of the 16 threads stay for the OS), on /mnt/pancake, no idle wait between them.
# Idle waits: the scheduler's 3 minutes once; here ~30 s (two clean polls) before each PART (tune-up, spec 220, spec 221), none between steps of a part.
# Steps and the estimate (minutes):                                                                           estimate
#   1 sync harness + specs, download upstream v0.2.16 and updated-strata (tar xmf)                              5
#   2 build plain upstream (its documented HIP command) and updated-strata (build_updated.sh), -j8              2 x 5-10 (a full strata build on this box took 3.6 min)
#   3 TUNE-UP (<= 30 min): flag_diag.sh hw, pcie_link.py, host_tune.sh --check, host_tune_ab.sh plan, os_tune.sh (check),
#     flag_diag.sh run plain (STRATA_HIPBLASLT_TUNING, STRATA_PF_F16: 10 min) and updated (STRATA_RDNA_Q6K, STRATA_GR_RDNA, STRATA_RDNA_MMVQ_WAVE,
#     STRATA_COMBINE_CPU, STRATA_VERIFY_ECAP: 18 min; STRATA_HIPBLASLT_TUNING is not repeated, the plain run judged the same table), the exact
#     flags get the parity smoke + the quick short-prompt screen that flag_diag does for every flag                                    29
#   4 spec 220 screen (3 starts) -> decide -> confirm 3 reps for `updated` only when it earned it                    15 + 25
#   5 spec 221 long screen, 1 rep per arm, no drift arm (8 starts)                                                  35
#   6 summary, recommended-flags file from the verdicts of the updated build (only exact or gate-passing GAIN)       1
#   total about 2 h (builds 15 + sync 5 + tune-up 29 + 220 40 + 221 35 = 124 min worst case; 3 idle polls 2 min).
# Verdict files: /home/dan/auto-runs/winbenchus/ (diag-plain/, diag-updated/, tuneup/, winbenchus-*.jsonl / .stdout, tuned-flags-recommended.json).
# Completion marker: the last line of /home/dan/clone-of-d/src/bench-updatedAA.log is WINBENCHUS_DONE (also after an abort; the line
# before it is then WINBENCHUS_ABORTED).  Results: /home/dan/auto-runs/winbenchus/ (abtest jsonl + stdout per spec, diag verdicts).
S=/home/dan/clone-of-d/src; R=$S/Strata-win; L=$S/bench-updatedAA.log; O=/home/dan/auto-runs/winbenchus; PM=/mnt/pancake/benchus
AA_LOG=$S/redo-chainAA.log; DEC=$S/chainT-decide.py; T=$O/work; T0=$(date +%s); BUDGET=${BENCH_BUDGET_S:-7200}
CM=/home/linuxbrew/.linuxbrew/bin/cmake; ROCM=/opt/rocm/core-10.0; NINJA=$S/Strata/.venv/bin/ninja
UPSTREAM_TAG=${UPSTREAM_TAG:-v0.2.16}
UP_URL=https://codeload.github.com/Niko1221/Strata/tar.gz/refs/tags/$UPSTREAM_TAG
US_URL=https://codeload.github.com/rocktboy1982/pancake-strata/tar.gz/refs/heads/updated-strata
REPO_URL=https://github.com/rocktboy1982/pancake-strata.git
UP=$PM/src-up; US=$PM/src-us; BUP=$PM/build-up; BUS=$PM/build-us
export PATH=$R/.venv/bin:$PATH LD_LIBRARY_PATH=$ROCM/lib HIP_VISIBLE_DEVICES=0
mkdir -p $O $T $PM $PM/bak $UP $US
echo "START $(date +%F_%T)" >> $L
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[c]onv_trace.py|[a]uto-chain|[a]uto-post|[i]q_avx2_parity|[t]tft_bench.py|[w]in-chain[A-Za-z0-9]*\.sh|[x-z]tail-chain[A-Z]\.sh|[a]fter7" >/dev/null 2>&1; }
# a FROZEN (state T) AA shell is not a running job; an AA shell that is not stopped is (standalone start without the scheduler)
aa_live() { for p in $(pgrep -f "[r]edo-chainAA\.sh" 2>/dev/null); do case "$(ps -o stat= -p $p 2>/dev/null)" in T*) ;; *) return 0;; esac; done; return 1; }
bz() { busy || aa_live; }
QI=${BENCH_PREIDLED:-0}   # 1: the scheduler already waited 3 idle minutes; every wait here is the quick one (two clean polls of 15 s)
wait_idle() { if [ "$QI" = 1 ]; then q=0; while [ $q -lt 2 ]; do if bz; then q=0; sleep 15; else q=$((q+1)); sleep 15; fi; done; else q=0; while [ $q -lt 3 ]; do if bz; then q=0; sleep 60; else q=$((q+1)); sleep 60; fi; done; QI=1; fi; }
elapsed() { echo $(( $(date +%s) - T0 )); }
left_ok() { [ $(( BUDGET - $(elapsed) )) -ge "$1" ]; }   # left_ok SECONDS: at least that much of the budget is left
service_back() { if [ "${BENCH_KEEP_SERVICE_DOWN:-0}" = 1 ]; then say "service stays down (the AA chain is resumed next and needs the box without the service) $(date +%T)"; else systemctl --user start pancake.service; say "service started $(date +%T)"; fi; }
clients() { ss -Htn state established '( sport = :11437 )' 2>/dev/null | wc -l; }
freekb() { df -Pk "$1" | awk 'NR==2{print $4}'; }
say() { echo "$*" >> $L; }
abort() {  # abort MESSAGE: ERROR line + marker, service back, STOP
  say "ERROR $*"; say "WINBENCHUS_ABORTED: $*"
  service_back; say "/ free $(freekb /) KB, /mnt/pancake free $(freekb /mnt/pancake) KB"
  echo WINBENCHUS_DONE >> $L; exit 1
}
need_disk() { [ "$(freekb /mnt/pancake)" -ge 2097152 ] || abort "$1: /mnt/pancake has $(freekb /mnt/pancake) KB free (< 2 GB); nothing was deleted"
              [ "$(freekb /)" -ge 307200 ] || abort "$1: / has $(freekb /) KB free (< 300 MB)"; }
step_ok() { rc=$1; nm=$2; [ "$rc" -eq 0 ] || abort "$nm failed with exit $rc"; }

# ------------------------------------------------------------------ 0. (standalone: wait for chain AA), stop the service
if [ "${BENCH_SCHED:-0}" != 1 ]; then
  until [ "$(tail -n 1 $AA_LOG 2>/dev/null)" = WINCHAINAA_DONE ]; do sleep 300; done
  say "chain AA is done $(date +%T)"
else say "started by the scheduler: the AA shell is frozen, the box was idle for 3 minutes $(date +%T)"; fi
if systemctl --user is-active --quiet pancake.service; then
  q=0; while [ $q -lt 3 ]; do if [ "$(clients)" -gt 0 ]; then q=0; say "WAIT service has clients $(date +%T)"; else q=$((q+1)); fi; sleep 60; done
  systemctl --user stop pancake.service; say "service stopped (idle) $(date +%T)"
else say "service already stopped"; fi
wait_idle
need_disk START

# ------------------------------------------------------------------ 1. sync harness + specs, download both source trees
cd $R || abort "no $R"
SHA=$(timeout 60 git ls-remote $REPO_URL refs/heads/port/rdna3-decode 2>/dev/null | cut -f1)
HARN=$PM/harness.tgz; rm -f $HARN.part
if [ -n "$SHA" ] && timeout 1200 curl -sfL -o $HARN.part https://codeload.github.com/rocktboy1982/pancake-strata/tar.gz/$SHA && tar tzf $HARN.part >/dev/null 2>&1; then
  mv $HARN.part $HARN; say "SYNC harness: origin port/rdna3-decode $SHA"
else abort "SYNC harness download failed"; fi
tar xzmf $HARN --no-same-owner --strip-components=1 -C $R; step_ok $? "SYNC harness tar"
for need in tools/pancake/abtest.py bench/specs/220-updated-vs-upstream-short.json bench/specs/221-updated-vs-upstream-long.json \
            configs/pancake/flashnext-iq2_xs-plain-upstream.json configs/pancake/flashnext-iq2_xs-updated-hip.json; do
  [ -f "$R/$need" ] || abort "SYNC: $need is missing in the synced tree"
done
fetch() {  # fetch URL DIR NAME
  f=$PM/$3.tgz; rm -f $f.part
  timeout 1800 curl -sfL -o $f.part "$1" && tar tzf $f.part >/dev/null 2>&1 || abort "download $3 failed ($1)"
  mv $f.part $f; tar xzmf $f --no-same-owner --strip-components=1 -C "$2"; step_ok $? "tar $3"
  say "SYNC $3: $1 ($(tar tzf $f | wc -l) entries)"
}
need_disk SYNC; fetch $UP_URL $UP upstream-$UPSTREAM_TAG
fetch $US_URL $US updated-strata
[ -f $UP/engine/CMakeLists.txt ] || abort "upstream tree has no engine/CMakeLists.txt"
[ -f $US/tools/pancake/flag_diag.sh ] || abort "updated-strata tree has no tools/pancake/flag_diag.sh"
[ -f $US/tools/pancake/build_updated.sh ] || abort "updated-strata tree has no tools/pancake/build_updated.sh"
say "VERSIONS upstream tag $UPSTREAM_TAG; updated-strata branch head as downloaded (see the tarball $PM/updated-strata.tgz)"

# ------------------------------------------------------------------ 2. builds (old binaries copied to bak/ first; a failure stops the chain)
[ -x $BUP/strata ] && cp -p $BUP/strata $PM/bak/strata-up.prev && say "BUILD backup strata-up.prev"
[ -x $BUS/strata ] && cp -p $BUS/strata $PM/bak/strata-us.prev && say "BUILD backup strata-us.prev"
need_disk BUILD; wait_idle; say "BUILD plain upstream $(date +%T) (its documented HIP configure; defaults for everything else)"
mkdir -p $BUP
# the documented upstream build (docs/AMD_HIP.md "Build") plus only what this box needs to find ROCm and ninja
$CM -S $UP/engine -B $BUP -G Ninja -DCMAKE_MAKE_PROGRAM=$NINJA -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=$ROCM > $BUP/configure.log 2>&1; rc=$?
[ $rc -eq 0 ] || { grep -m8 -E "error|CMake Error" $BUP/configure.log >> $L; abort "plain upstream configure exit=$rc (log $BUP/configure.log)"; }
nice -n 10 $CM --build $BUP -j8 --target strata > $BUP/build.log 2>&1; rc=$?
say "BUILD_END plain upstream exit=$rc $(date +%T) (/mnt/pancake free $(freekb /mnt/pancake) KB)"
[ $rc -eq 0 ] || { say "BUILD ERROR LINES (plain upstream, first 40 of $BUP/build.log):"; grep -m40 -E "error|undefined|CMake Error|FAILED" $BUP/build.log >> $L
  [ -f $PM/bak/strata-up.prev ] && { cp -p $PM/bak/strata-up.prev $BUP/strata; say "BUILD FAILED: previous strata-up restored"; }
  abort "plain upstream build exit=$rc (log $BUP/build.log)"; }
[ -x $BUP/strata ] || abort "plain upstream build exit=0 but $BUP/strata is missing"
need_disk BUILD2; say "BUILD updated-strata $(date +%T) (tools/pancake/build_updated.sh)"
JOBS=8 nice -n 10 sh $US/tools/pancake/build_updated.sh $US $BUS > $PM/build-us.stdout 2>&1; rc=$?
say "BUILD_END updated-strata exit=$rc $(date +%T) (/mnt/pancake free $(freekb /mnt/pancake) KB)"
[ $rc -eq 0 ] || { say "BUILD ERROR LINES (updated-strata: the HIP code of this branch was never compiled by the ROCm compiler before this step; first 60 of $BUS/build.log):"
  grep -m60 -E "error|undefined|CMake Error|FAILED" $BUS/build.log >> $L 2>/dev/null; tail -5 $PM/build-us.stdout >> $L
  [ -f $PM/bak/strata-us.prev ] && { cp -p $PM/bak/strata-us.prev $BUS/strata; say "BUILD FAILED: previous strata-us restored"; }
  abort "updated-strata build exit=$rc (log $BUS/build.log)"; }
[ -x $BUS/strata ] || abort "updated-strata build exit=0 but $BUS/strata is missing"
say "BUILD sizes: plain $(stat -c %s $BUP/strata) bytes, updated $(stat -c %s $BUS/strata) bytes"

# ------------------------------------------------------------------ 3. TUNE-UP (read-only, <= 30 min): hardware, PCIe link, host tuning checks, per-flag diagnostics of both builds
tuneup_checks() {  # read-only: sysfs / proc reads only (no sudo, no sysctl write, nothing applied): `os_tune.sh` without argument is its check, `host_tune_ab.sh plan` prints the order
  td=$US/tools/pancake; od=$O/tuneup; mkdir -p $od
  sh $US/tools/pancake/flag_diag.sh hw > $O/hardware.json 2>&1; say "HARDWARE $(tr -d '\n ' < $O/hardware.json | cut -c1-300)"
  timeout 120 python3 $td/pcie_link.py > $od/pcie_link.txt 2>&1; say "TUNEUP pcie_link exit=$? $(tail -n 3 $od/pcie_link.txt | tr '\n' ' ' | cut -c1-240)"
  timeout 120 sh $td/host_tune.sh --check > $od/host_tune_check.txt 2>&1; say "TUNEUP host_tune --check exit=$? $(tail -n 4 $od/host_tune_check.txt | tr '\n' ' ' | cut -c1-240)"
  timeout 120 sh $td/host_tune_ab.sh plan > $od/host_tune_ab_plan.txt 2>&1; say "TUNEUP host_tune_ab plan exit=$? (the A/B itself changes sysctl and is the owner's)"
  timeout 120 sh $td/os_tune.sh > $od/os_tune_check.txt 2>&1; say "TUNEUP os_tune check exit=$? $(tail -n 3 $od/os_tune_check.txt | tr '\n' ' ' | cut -c1-240)"
}
diag() {  # diag NAME EXE BUDGET_MIN FLAGS
  nm=$1; exe=$2; budget=$3; only=$4
  sed "s#\"exe\": \"[^\"]*\"#\"exe\": \"$exe\"#; s#<REPO>#$R#g" $R/configs/pancake/flashnext-iq2_xs-plain-upstream.json > $O/diag-$nm-config.json
  say "START diag $nm $(date +%F_%T)"
  FLAG_DIAG_ABTEST=$R/tools/pancake/abtest.py timeout $(( (budget + 8) * 60 )) sh $US/tools/pancake/flag_diag.sh run $O/diag-$nm-config.json $O/diag-$nm --budget-min $budget \
    --only $only > $O/diag-$nm.stdout 2>&1; rc=$?
  say "END diag $nm exit=$rc $(date +%T)"; cut -c1-200 $O/diag-$nm.stdout | tail -8 >> $L
  [ -f $O/diag-$nm/verdicts.json ] || say "ERROR diag $nm wrote no verdicts.json (the benchmark goes on: the diagnostics are a first, cheap step)"
}
# plain release: only the two switches that exist in v0.2.16 (10 min).  updated build: its five hardware-dependent switches (18 min); STRATA_HIPBLASLT_TUNING
# is NOT repeated (the plain run judged the same table).  flag_diag starts the engine once per flag + twice for the default; exact flags must give the
# same text (parity smoke) and are screened on 2 short prompts.
need_disk DIAG; wait_idle; tuneup_checks
diag plain $BUP/strata 10 STRATA_HIPBLASLT_TUNING,STRATA_PF_F16
diag updated $BUS/strata 18 STRATA_RDNA_Q6K,STRATA_GR_RDNA,STRATA_RDNA_MMVQ_WAVE,STRATA_COMBINE_CPU,STRATA_VERIFY_ECAP
say "TUNEUP done after $(elapsed) s"

# ------------------------------------------------------------------ 4 / 5. the two specs (two-pass: screen, confirm only for an arm with a benefit)
ab() {  # ab SPEC NAME flags... : one abtest run
  sp=$1; nm=$2; shift 2
  need_disk "ab $nm"; say "START $nm $(date +%F_%T)"
  python3 tools/pancake/abtest.py $sp "$@" --out $O/winbenchus-$nm.jsonl > $O/winbenchus-$nm.stdout 2>&1; rc=$?
  say "END $nm exit=$rc $(date +%T)"
  grep -E "vs .*(better|worse|no difference)|verdict|promote|drop" $O/winbenchus-$nm.stdout | cut -c1-220 >> $L
  grep -A6 "generated text vs base" $O/winbenchus-$nm.stdout | cut -c1-200 >> $L
  return $rc
}
decide() {  # decide NAME SPEC: 0 and $T/NAME-confirm.json when an arm earned the confirm, 1 when not
  rm -f "${T:?}/$1-confirm.json"
  python3 $DEC $O/winbenchus-$1-screen.stdout $2 $T/$1-confirm.json > $T/$1-decide.txt 2>&1
  grep -E "WIN|CONFIRM|NONE|dropped" $T/$1-decide.txt | cut -c1-230 >> $L
  [ -s $T/$1-confirm.json ]
}
cd $R
wait_idle
ab bench/specs/220-updated-vs-upstream-short.json 220-screen --tier screen --drift-check; rc1=$?
if [ $rc1 -ne 0 ]; then say "ERROR spec 220 screen exit=$rc1 (log $O/winbenchus-220-screen.stdout); no confirm; spec 221 still runs"
elif decide 220 bench/specs/220-updated-vs-upstream-short.json; then
  if left_ok 1800; then ab $T/220-confirm.json 220-confirm --tier confirm --block 3 --warmup 1 --outlier robust --reclaim-max 3 --reps 3 --no-early-stop; rc1=$?
    [ $rc1 -eq 0 ] || say "ERROR spec 220 confirm exit=$rc1 (log $O/winbenchus-220-confirm.stdout)"
  else say "SKIP 220 confirm: only $(( BUDGET - $(elapsed) )) s of the budget left (the screen numbers stand)"; fi
else say "220: no arm with a benefit in the screen: no confirm (dropped after pass 1: no benefit observed, n=1 pass, not a refutation)"; fi
# spec 221: long prompts; one screen rep per arm, no drift arm (a copy of the spec; the repo spec is untouched)
sed 's/"drift_check": true/"drift_check": false/' bench/specs/221-updated-vs-upstream-long.json > $T/221-screen.json
wait_idle
ab $T/221-screen.json 221-screen --tier screen; rc2=$?
if [ $rc2 -ne 0 ]; then say "ERROR spec 221 screen exit=$rc2 (log $O/winbenchus-221-screen.stdout); a plain-upstream arm that cannot hold 100K tokens is a result, not a skip"
elif decide 221 $T/221-screen.json; then
  if left_ok 4200; then ab $T/221-confirm.json 221-confirm --tier confirm --reps 2 --no-early-stop; rc2=$?
  else say "221 confirm not run: only $(( BUDGET - $(elapsed) )) s of the budget left (< 4200 s); the one screen rep per arm stands (long-prompt paired SD 1.3 %)"; fi
else say "221: no arm with a benefit in the screen: no confirm"; fi

# ------------------------------------------------------------------ 6. recommended-flags file from the measured verdicts (only exact or gate-passing GAIN flags enter it), service
for nm in updated; do
  [ -f $O/diag-$nm/verdicts.json ] && sh $US/tools/pancake/flag_diag.sh --write-recommended $O/diag-$nm/verdicts.json $O/tuned-flags-recommended.json >> $L 2>&1
done
say "SUMMARY spec220 exit=$rc1, spec221 exit=$rc2, box time $(elapsed) s; results in $O; the recommended-flags file (not installed): $O/tuned-flags-recommended.json"
service_back; say "/ free $(freekb /) KB, /mnt/pancake free $(freekb /mnt/pancake) KB"
echo WINBENCHUS_DONE >> $L
