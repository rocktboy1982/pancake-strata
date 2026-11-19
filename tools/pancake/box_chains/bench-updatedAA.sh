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
# Starts only after WINCHAINAA_DONE is the last line of /home/dan/clone-of-d/src/redo-chainAA.log (chain AA, the only job running now).
#
# Steps and the estimate (hours, target <= 8):                                                               estimate
#   0 wait for chain AA, 3 idle minutes, stop the service                                                      0-1 (wait)
#   1 sync the harness + specs (origin port/rdna3-decode, tar xmf), download upstream v0.2.16 and updated-strata   0.1
#   2 build plain upstream (its documented HIP command, its own defaults) and updated-strata (build_updated.sh)  1.5-2.5 (two cold HIP builds)
#   3 flag_diag.sh on BOTH builds (hardware, per-flag smoke / parity / quick tok/s; budget 15 minutes each)       0.5
#   4 spec 220: short + edit (code-edit), decode tok/s, 5 reps confirm, text identity vs plain                    0.7
#   5 spec 221: long 10K / 32K / 64K / 100K (cold read + 256-token reply: prompt and decode tok/s), 5 reps       2.5-3
#   6 summary lines, the recommended-flags file from the measured verdicts (written beside the results, not installed), service back   0.1
#   total about 5.5-7.5 h of box time (plus the wait for chain AA).
# Completion marker: the last line of /home/dan/clone-of-d/src/bench-updatedAA.log is WINBENCHUS_DONE (also after an abort; the line
# before it is then WINBENCHUS_ABORTED).  Results: /home/dan/auto-runs/winbenchus/ (abtest jsonl + stdout per spec, diag verdicts).
S=/home/dan/clone-of-d/src; R=$S/Strata-win; L=$S/bench-updatedAA.log; O=/home/dan/auto-runs/winbenchus; PM=/mnt/pancake/benchus
AA_LOG=$S/redo-chainAA.log
CM=/home/linuxbrew/.linuxbrew/bin/cmake; ROCM=/opt/rocm/core-10.0; NINJA=$S/Strata/.venv/bin/ninja
UPSTREAM_TAG=${UPSTREAM_TAG:-v0.2.16}
UP_URL=https://codeload.github.com/Niko1221/Strata/tar.gz/refs/tags/$UPSTREAM_TAG
US_URL=https://codeload.github.com/rocktboy1982/pancake-strata/tar.gz/refs/heads/updated-strata
REPO_URL=https://github.com/rocktboy1982/pancake-strata.git
UP=$PM/src-up; US=$PM/src-us; BUP=$PM/build-up; BUS=$PM/build-us
export PATH=$R/.venv/bin:$PATH LD_LIBRARY_PATH=$ROCM/lib HIP_VISIBLE_DEVICES=0
mkdir -p $O $PM $PM/bak $UP $US
echo "START $(date +%F_%T)" >> $L
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[c]onv_trace.py|[a]uto-chain|[a]uto-post|[i]q_avx2_parity|[t]tft_bench.py|[w]in-chain[A-Za-z0-9]*\.sh|[x-z]tail-chain[A-Z]\.sh|[r]edo-chainAA\.sh|[a]fter7" >/dev/null 2>&1; }
wait_idle() { if [ "${QI:-0}" = 1 ]; then q=0; while [ $q -lt 2 ]; do if busy; then q=0; sleep 15; else q=$((q+1)); sleep 15; fi; done; else q=0; while [ $q -lt 3 ]; do if busy; then q=0; sleep 60; else q=$((q+1)); sleep 60; fi; done; QI=1; fi; }
clients() { ss -Htn state established '( sport = :11437 )' 2>/dev/null | wc -l; }
freekb() { df -Pk "$1" | awk 'NR==2{print $4}'; }
say() { echo "$*" >> $L; }
abort() {  # abort MESSAGE: ERROR line + marker, service back, STOP
  say "ERROR $*"; say "WINBENCHUS_ABORTED: $*"
  systemctl --user start pancake.service; say "service started $(date +%T); / free $(freekb /) KB, /mnt/pancake free $(freekb /mnt/pancake) KB"
  echo WINBENCHUS_DONE >> $L; exit 1
}
need_disk() { [ "$(freekb /mnt/pancake)" -ge 2097152 ] || abort "$1: /mnt/pancake has $(freekb /mnt/pancake) KB free (< 2 GB); nothing was deleted"
              [ "$(freekb /)" -ge 307200 ] || abort "$1: / has $(freekb /) KB free (< 300 MB)"; }
step_ok() { rc=$1; nm=$2; [ "$rc" -eq 0 ] || abort "$nm failed with exit $rc"; }

# ------------------------------------------------------------------ 0. wait for chain AA, idle, stop the service
until [ "$(tail -n 1 $AA_LOG 2>/dev/null)" = WINCHAINAA_DONE ]; do sleep 300; done
say "chain AA is done $(date +%T)"
q=0; while [ $q -lt 3 ]; do if [ "$(clients)" -gt 0 ]; then q=0; say "WAIT service has clients $(date +%T)"; else q=$((q+1)); fi; sleep 60; done
systemctl --user stop pancake.service; say "service stopped (idle) $(date +%T)"
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
for need in tools/pancake/abtest.py tools/pancake/flag_diag.py bench/specs/220-updated-vs-upstream-short.json bench/specs/221-updated-vs-upstream-long.json \
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
[ $rc -eq 0 ] || { grep -m8 -E "error:|undefined|CMake Error" $BUP/build.log >> $L
  [ -f $PM/bak/strata-up.prev ] && { cp -p $PM/bak/strata-up.prev $BUP/strata; say "BUILD FAILED: previous strata-up restored"; }
  abort "plain upstream build exit=$rc (log $BUP/build.log)"; }
[ -x $BUP/strata ] || abort "plain upstream build exit=0 but $BUP/strata is missing"
need_disk BUILD2; wait_idle; say "BUILD updated-strata $(date +%T) (tools/pancake/build_updated.sh)"
nice -n 10 sh $US/tools/pancake/build_updated.sh $US $BUS > $PM/build-us.stdout 2>&1; rc=$?
say "BUILD_END updated-strata exit=$rc $(date +%T) (/mnt/pancake free $(freekb /mnt/pancake) KB)"
[ $rc -eq 0 ] || { tail -8 $PM/build-us.stdout >> $L
  [ -f $PM/bak/strata-us.prev ] && { cp -p $PM/bak/strata-us.prev $BUS/strata; say "BUILD FAILED: previous strata-us restored"; }
  abort "updated-strata build exit=$rc (log $BUS/build.log)"; }
[ -x $BUS/strata ] || abort "updated-strata build exit=0 but $BUS/strata is missing"
say "BUILD sizes: plain $(stat -c %s $BUP/strata) bytes, updated $(stat -c %s $BUS/strata) bytes"

# ------------------------------------------------------------------ 3. diagnostics on both builds (read-only, budget 15 minutes each)
diag() {  # diag NAME EXE
  nm=$1; exe=$2
  sed "s#\"exe\": \"[^\"]*\"#\"exe\": \"$exe\"#; s#<REPO>#$R#g" $R/configs/pancake/flashnext-iq2_xs-plain-upstream.json > $O/diag-$nm-config.json
  wait_idle; say "START diag $nm $(date +%F_%T)"
  timeout 2700 sh $R/tools/pancake/flag_diag.sh run $O/diag-$nm-config.json $O/diag-$nm --budget-min 15 \
    --only STRATA_RDNA_Q6K,STRATA_HIPBLASLT_TUNING,STRATA_PF_F16,STRATA_HIP_WMMA > $O/diag-$nm.stdout 2>&1; rc=$?
  say "END diag $nm exit=$rc $(date +%T)"; cut -c1-200 $O/diag-$nm.stdout | tail -8 >> $L
  [ -f $O/diag-$nm/verdicts.json ] || say "ERROR diag $nm wrote no verdicts.json (the benchmark goes on: the diagnostics are a first, cheap step)"
}
need_disk DIAG; diag plain $BUP/strata; diag updated $BUS/strata
sh $R/tools/pancake/flag_diag.sh hw > $O/hardware.json 2>&1; say "HARDWARE $(tr -d '\n ' < $O/hardware.json | cut -c1-300)"

# ------------------------------------------------------------------ 4 / 5. the two specs
ab() {  # ab SPEC NAME
  sp=$1; nm=$2
  need_disk "ab $nm"; wait_idle; say "START $nm $(date +%F_%T)"
  python3 tools/pancake/abtest.py $sp --tier confirm --block 3 --warmup 1 --outlier robust --reclaim-max 3 --reps 5 --no-early-stop \
    --out $O/winbenchus-$nm.jsonl > $O/winbenchus-$nm.stdout 2>&1; rc=$?
  say "END $nm exit=$rc $(date +%T)"
  grep -E "vs .*(better|worse|no difference)" $O/winbenchus-$nm.stdout | cut -c1-220 >> $L
  grep -A6 "generated text vs base" $O/winbenchus-$nm.stdout | cut -c1-200 >> $L
  return $rc
}
cd $R
ab bench/specs/220-updated-vs-upstream-short.json short-edit; rc1=$?
[ $rc1 -eq 0 ] || say "ERROR spec 220 exit=$rc1 (log $O/winbenchus-short-edit.stdout); spec 221 still runs, the failure is recorded"
ab bench/specs/221-updated-vs-upstream-long.json long; rc2=$?
[ $rc2 -eq 0 ] || say "ERROR spec 221 exit=$rc2 (log $O/winbenchus-long.stdout); a plain-upstream arm that cannot hold 100K tokens is a result, not a skip"

# ------------------------------------------------------------------ 6. recommended-flags file from the measured verdicts, service back
for nm in updated; do
  [ -f $O/diag-$nm/verdicts.json ] && sh $R/tools/pancake/flag_diag.sh --write-recommended $O/diag-$nm/verdicts.json $O/tuned-flags-recommended.json >> $L 2>&1
done
say "SUMMARY spec220 exit=$rc1, spec221 exit=$rc2; results in $O; the recommended-flags file (not installed): $O/tuned-flags-recommended.json"
systemctl --user start pancake.service; say "service started $(date +%T); / free $(freekb /) KB, /mnt/pancake free $(freekb /mnt/pancake) KB"
echo WINBENCHUS_DONE >> $L
