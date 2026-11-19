#!/bin/sh
# bench-updatedRD.sh (owner order 2026-11-19): benchmark the PLAIN Strata-rdna3 against updated-strata-rdna3 on the box.  The file name has no "win-chain"
# so that no other busy() pattern ever sees this script (T-944); its log is /home/dan/clone-of-d/src/bench-updatedRD.log.
#   (a) plain   = github.com/nicejunjie/Strata-rdna3 main at $RD_SHA (default 8b01b0c85, 2026-11-15), its own documented flags (README "Manual build")
#   (b) updated = origin updated-strata-rdna3 (the selective port of the adopted pancake enhancements) built by tools/pancake/build_updated_rdna3.sh
# Workloads of the upstream benchmark: short, edit (code edit), long 10000 / 32000 / 64000 / 100000 tokens; decode tok/s and prompt tok/s, generated text vs
# base (identical = exact); the A/B harness is tools/pancake/abtest.py of origin port/rdna3-decode (spec 250), a screen first, then the confirm: --block 3,
# 5 reps, arms serialized (one engine at a time, A B B A).
# Rules (standing): waits for WINCHAINAA_DONE in redo-chainAA.log AND WINBENCHUS_DONE in bench-updatedAA.log; then 3 minutes without a client on port 11437;
# pancake.service is stopped only after that and started again at the end; box jobs never overlap (3 idle minutes before the first step, busy() sees every
# other job); nothing is killed or deleted; sources are synced with `tar xmf` semantics (mtimes reset, else ninja keeps stale objects); builds go OUTSIDE / to
# /mnt/pancake/benchRD and stop when it has < 2 GB free; a failed configure / build / missing binary logs ERROR, puts the previous binary back (cp over, nothing
# removed), starts the service and STOPS the chain (never a silent skip); the DONE marker is the last line: WINBENCHRD_DONE (also after an abort).
# Order: 0 wait | 1 stop service, sync three trees | 2 build plain, build updated | 3 parity ctests of the updated build | 4 flag diagnostics (both builds, <= 30 min)
#        | 5 spec 250 screen | 6 spec 250 confirm | 7 service back.
# Estimate (hours, box idle): sync 0.2, builds 2 x (1.0 - 1.5), ctests 0.1, diagnostics 0.5, screen 0.5, confirm 2.2 (three arms) => about 6 h, worst case 7.3 h (target <= 8 h).
S=/home/dan/clone-of-d/src; R=$S/Strata-win; L=$S/bench-updatedRD.log; O=/home/dan/auto-runs; T=$O/benchRD; PM=/mnt/pancake/benchRD
CM=/home/linuxbrew/.linuxbrew/bin/cmake
PLAIN_SRC=$PM/src-plain; UPD_SRC=$PM/src-updated; TOOLS=$PM/src-pancake
BP=$PM/build-plain; BU=$PM/build-updated; ROCM=/opt/rocm/core-10.0
RD_SHA=${RD_SHA:-8b01b0c8512d388552e8601b3aff6753d05286e0}
export PATH=$R/.venv/bin:$PATH LD_LIBRARY_PATH=$ROCM/lib HIP_VISIBLE_DEVICES=0
mkdir -p $T $PM $PM/bak
echo "START $(date +%F_%T)" >> $L
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[c]onv_trace.py|[a]uto-chain|[a]uto-post|[i]q_avx2_parity|[t]tft_bench.py|[w]in-chain[A-Za-z0-9]*\.sh|[x-z]tail-chain[A-Z]\.sh|[r]edo-chain[A-Z]*\.sh|[a]fter7" >/dev/null 2>&1; }
wait_idle() { q=0; while [ $q -lt 3 ]; do if busy; then q=0; sleep 60; else q=$((q+1)); sleep 60; fi; done; }
clients() { ss -Htn state established '( sport = :11437 )' 2>/dev/null | wc -l; }
freekb() { df -Pk "$1" | awk 'NR==2{print $4}'; }
say() { echo "$*" >> $L; }
abort() {  # abort MESSAGE: ERROR line + marker, service back, STOP
  say "ERROR $*"; say "WINBENCHRD_ABORTED: $*"
  systemctl --user start pancake.service; say "service started $(date +%T); / free $(freekb /) KB, /mnt/pancake free $(freekb /mnt/pancake) KB"
  echo WINBENCHRD_DONE >> $L; exit 1
}
fetch() {  # fetch REPO REF DEST: codeload tarball of a GitHub repo at a ref (a branch name is resolved to its commit first), unpacked with mtimes reset
  repo=$1; ref=$2; dest=$3
  case $ref in
    ????????????????????????????????????????) sha=$ref ;;
    *) sha=$(timeout 60 git ls-remote https://github.com/$repo.git refs/heads/$ref 2>/dev/null | cut -f1) ;;
  esac
  [ -n "$sha" ] || abort "SYNC: cannot resolve $repo $ref"
  tgz=$PM/$(echo "$repo" | tr / _).tgz; rm -f $tgz.part
  timeout 1200 curl -sfL -o $tgz.part https://codeload.github.com/$repo/tar.gz/$sha && tar tzf $tgz.part >/dev/null 2>&1 || abort "SYNC: download of $repo $sha failed"
  mv $tgz.part $tgz; mkdir -p $dest
  tar xzmf $tgz --no-same-owner --strip-components=1 -C $dest || abort "SYNC: tar of $repo $sha failed"
  say "SYNC $repo $ref -> $sha in $dest"
}
build() {  # build NAME BUILDDIR cmd...: guarded; the previous binary is copied first and put back when the build fails
  nm=$1; bd=$2; shift 2
  [ "$(freekb /mnt/pancake)" -ge 2097152 ] || abort "BUILD $nm: /mnt/pancake has $(freekb /mnt/pancake) KB free (< 2 GB); nothing was deleted"
  [ -x $bd/strata ] && cp -p $bd/strata $PM/bak/strata.$nm.prev && say "BUILD backup of the previous $bd/strata ($nm)"
  wait_idle; say "BUILD $nm $(date +%T) (/mnt/pancake free $(freekb /mnt/pancake) KB)"
  "$@" > $T/build-$nm.log 2>&1; rc=$?
  say "BUILD_END $nm exit=$rc $(date +%T)"
  if [ $rc -ne 0 ]; then
    grep -m8 -E "error:|undefined|CMake Error|ERROR" $T/build-$nm.log >> $L
    [ -f $PM/bak/strata.$nm.prev ] && cp -p $PM/bak/strata.$nm.prev $bd/strata && say "BUILD FAILED: the previous $bd/strata was restored"
    abort "BUILD $nm FAILED exit=$rc (log $T/build-$nm.log); the chain stops, nothing after it ran"
  fi
  [ -x $bd/strata ] || abort "BUILD $nm exit=0 but $bd/strata is missing"
  say "ENGINE $nm: $bd/strata present"
}
ab() {  # ab SPEC PREFIX flags...: abtest from the pancake tools tree, one engine at a time
  sp=$1; pfx=$2; shift 2
  [ "$(freekb /)" -ge 307200 ] || abort "AB $pfx: / has < 300 MB free"
  wait_idle; say "START $pfx $(date +%F_%T)"
  ( cd $TOOLS && python3 tools/pancake/abtest.py $sp "$@" --out $O/benchRD-$pfx.jsonl ) > $O/benchRD-$pfx.stdout 2>&1; rc=$?
  say "END $pfx exit=$rc $(date +%T)"
  return $rc
}

# ------------------------------------------------------------------ 0. wait for chain AA and the upstream benchmark, then 3 idle minutes without a client
until grep -q WINCHAINAA_DONE $S/redo-chainAA.log 2>/dev/null && grep -q WINBENCHUS_DONE $S/bench-updatedAA.log 2>/dev/null; do sleep 300; done
say "chain AA and the upstream benchmark are done $(date +%T)"
q=0; while [ $q -lt 3 ]; do if [ "$(clients)" -gt 0 ]; then q=0; say "WAIT service has clients $(date +%T)"; else q=$((q+1)); fi; sleep 60; done
systemctl --user stop pancake.service; say "service stopped (idle) $(date +%T)"
echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null
wait_idle
say "SYNC start $(date +%T) df / = $(freekb /) KB, /mnt/pancake = $(freekb /mnt/pancake) KB"
[ "$(freekb /mnt/pancake)" -ge 2097152 ] || abort "SYNC: /mnt/pancake has < 2 GB free"
fetch nicejunjie/Strata-rdna3 $RD_SHA $PLAIN_SRC
fetch rocktboy1982/pancake-strata updated-strata-rdna3 $UPD_SRC
fetch rocktboy1982/pancake-strata port/rdna3-decode $TOOLS
[ -f $UPD_SRC/tools/pancake/build_updated_rdna3.sh ] || abort "SYNC: the updated tree has no tools/pancake/build_updated_rdna3.sh"
[ -f $TOOLS/tools/pancake/abtest.py ] || abort "SYNC: the pancake tools tree has no tools/pancake/abtest.py"
SPEC=$TOOLS/bench/specs/250-rdna3-updated-vs-plain.json
[ -f $SPEC ] || abort "SYNC: $SPEC is missing"

# ------------------------------------------------------------------ 1. builds
NINJA=$S/Strata/.venv/bin/ninja
build plain $BP sh -c "HIP_PLATFORM=amd ROCM_PATH=$ROCM HIP_PATH=$ROCM $CM -S $PLAIN_SRC -B $BP -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_PREFILL_MMQ=ON -DSTRATA_MMQ_KQUANTS=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=$ROCM -DSTRATA_GGML_DIR=/home/dan/clone-of-d/build-int/_deps/strata_llamacpp-src -DCMAKE_MAKE_PROGRAM=$NINJA && \
  nice -n 10 $CM --build $BP -j8 --target strata"
build updated $BU sh $UPD_SRC/tools/pancake/build_updated_rdna3.sh $UPD_SRC $BU --target strata --target verify_ecap_test --target sampler_parity --target elementwise_parity

# ------------------------------------------------------------------ 2. parity ctests of the updated build (a failure is reported, it does not hide the benchmark)
wait_idle; say "START ctest $(date +%T)"
ctest --test-dir $BU -R "verify_ecap_test|sampler_parity|elementwise_parity" --output-on-failure > $T/ctest.out 2>&1; rc=$?
say "END ctest exit=$rc $(date +%T)"
if [ $rc -ne 0 ]; then say "ERROR ctest of the updated build exit=$rc (see $T/ctest.out); the benchmark goes on"; tail -n 15 $T/ctest.out >> $L; fi

# ------------------------------------------------------------------ 3. the two server configs (the box's filled-in preset, args and env made to fit Strata-rdna3)
python3 $UPD_SRC/tools/pancake/box_chains/bench_updatedRD_conf.py $S/Strata/pancake-flashnext-iq2_xs-fast-hip.json $PLAIN_SRC $UPD_SRC $BP $BU \
    $T/config-plain.json $T/config-updated.json >> $L 2>&1 || abort "CONFIG: bench_updatedRD_conf.py failed"
say "CONFIG written: $T/config-plain.json $T/config-updated.json"

# ------------------------------------------------------------------ 4. flag diagnostics on both builds (budget 30 min: updated 4 flags + base, plain 2 flags + base)
diag() {  # diag NAME CONFIG FLAGS [CTEST_DIR]
  nm=$1; cf=$2; fl=$3; ct=${4:-}
  wait_idle; say "START diag-$nm $(date +%F_%T)"
  ( cd $TOOLS && timeout 1800 python3 $UPD_SRC/tools/pancake/flag_diag.py run --config $cf --flags $fl --abtest $TOOLS/tools/pancake/abtest.py --out $T/diag-$nm \
      ${ct:+--ctest $ct} ) > $T/diag-$nm.stdout 2>&1; rc=$?
  say "END diag-$nm exit=$rc $(date +%T)"
  cut -c1-200 $T/diag-$nm.stdout >> $L
  return $rc
}
diag updated $T/config-updated.json POOL_FUSE_PHASES,VERIFY_ECAP,PF_F16,HIPBLASLT_TUNING $BU
diag plain $T/config-plain.json PF_F16,HIPBLASLT_TUNING

# ------------------------------------------------------------------ 5. spec 250: screen, then the confirm (5 reps, blocks of 3, drift check; the arms differ by binary and by the shipped recommended flags)
sed "s#@PLAIN@#$BP#; s#@UPDATED@#$BU#; s#@CONFIG_PLAIN@#$T/config-plain.json#" $SPEC > $T/250.json
ab $T/250.json 250-screen --tier screen --drift-check
grep -E "vs base|identical|DIFFERENT" $O/benchRD-250-screen.stdout | cut -c1-200 >> $L
ab $T/250.json 250-confirm --tier confirm --block 3 --warmup 1 --outlier robust --reclaim-max 3 --reps 5 --drift-check
grep -E "vs base|identical|DIFFERENT" $O/benchRD-250-confirm.stdout | cut -c1-200 >> $L

# ------------------------------------------------------------------ 6. back
systemctl --user start pancake.service; say "service started $(date +%T); / free $(freekb /) KB, /mnt/pancake free $(freekb /mnt/pancake) KB"
echo WINBENCHRD_DONE >> $L
