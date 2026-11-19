#!/bin/sh
# bench-updatedRD.sh (owner order 2026-11-19): benchmark the PLAIN Strata-rdna3 against updated-strata-rdna3 on the box.  The file name has no "win-chain"
# so that no other busy() pattern ever sees this script (T-944); its log is /home/dan/clone-of-d/src/bench-updatedRD.log.
#   (a) plain   = github.com/nicejunjie/Strata-rdna3 main at $RD_SHA (default 8b01b0c85, 2026-11-15), its own documented flags (README "Manual build")
#   (b) updated = origin updated-strata-rdna3 (the selective port of the adopted pancake enhancements) built by tools/pancake/build_updated_rdna3.sh
# Workloads of the upstream benchmark: short, edit (code edit), long 10000 / 32000 / 64000 / 100000 tokens; decode tok/s and prompt tok/s, generated text vs
# base (identical = exact); the A/B harness is tools/pancake/abtest.py of origin port/rdna3-decode (spec 250).
# Start: normally launched by bench-sched.sh (BENCH_SCHED=1, BENCH_PREIDLED=1, BENCH_KEEP_SERVICE_DOWN=1) right after bench-updatedAA.sh: the scheduler froze the AA
# shell (SIGSTOP), waited for an idle box and resumes AA after this chain.  Standalone (no BENCH_SCHED) it still waits for WINCHAINAA_DONE and WINBENCHUS_DONE.
# SHORT PLAN (owner order 2026-11-19: not multiple hours; target <= 2 h of box time, hard budget BENCH_BUDGET_S = 7200 s, optional passes skipped when it is over).
# Two-pass protocol: spec 250 (3 arms x short, edit, long 10K..100K) is a SCREEN first (1 start per arm + the base again, drift check; the long prompts need no more:
# paired SD 1.3 %); the CONFIRM (3 reps of short + edit, the decode cases: paired SD 3.2 % / 2.7 %) runs only for arms that earned it in the screen (>= 1.0 %
# lossless / >= 1.5 % lossy, chainT-decide.py); arms without a screen benefit get no confirm.  Spec 251 (table + learned profile) is a short screen (short + long32k,
# no drift arm), its confirm only when the budget has 40 minutes left.
# Builds: plain and updated tree one after the other, ninja -j8, on /mnt/pancake, no idle wait between them.  Idle waits: the scheduler's 3 minutes once;
# here ~30 s (two clean polls) before each PART (ctest + tune-up + diagnostics, spec 250, spec 251), none between steps of a part.
# Rules (standing): pancake.service is stopped only after 3 minutes without a client on port 11437 (already stopped under the scheduler: AA keeps it down); box jobs
# never overlap (busy() sees every other job; a FROZEN AA shell is not a job, an AA shell that is not stopped is); nothing is killed or deleted; sources are synced
# with `tar xmf` semantics (mtimes reset, else ninja keeps stale objects); builds go OUTSIDE / to /mnt/pancake/benchRD and stop when it has < 2 GB free; a failed
# configure / build / missing binary logs ERROR, puts the previous binary back (cp over, nothing removed), and STOPS the chain (never a silent skip); the DONE
# marker is the last line: WINBENCHRD_DONE (also after an abort).
# Order: 0 wait | 1 stop service, PREREQ, sync three trees | 2 configs + spec, build plain + smoke, build updated + smoke | 3 parity ctests | 4 TUNE-UP (<= 30 min): flag_diag.py hardware, pcie_link.py, host_tune.sh
#        --check, host_tune_ab.sh plan, os_tune.sh (check; no sudo, nothing applied), flag diagnostics of the updated build (POOL_FUSE_PHASES exact + ctest parity, VERIFY_ECAP
#        lossy + ctest, PF_F16, RDNA_MMVQ_WAVE exact, KV_HOST_DMA exact; 20 min; the plain build's diagnostics are dropped: its two flags are the updated build's own and are
#        judged by spec 250; HIPBLASLT_TUNING and EXPERT_PROFILE are dropped from the diagnostics: spec 251 measures exactly those two) | 5 spec 250 screen, decide, confirm |
#        6 spec 251 screen | 7 recommended-flags file from the verdicts | 8 summary (service stays down under the scheduler).
# 2026-11-19 PM (first scheduled run aborted at the plain build after 3 minutes, nothing measured): src/core/peer_experts.cpp of Strata-rdna3 (and of updated-strata-rdna3) has
# `#include <rccl/rccl.h>` under STRATA_USE_HIP on Linux and believed unused; this box's ROCm 10.0 has librccl.so and NO rccl headers.  Then: PREREQ made an EMPTY stub
# $PM/rccl-stub/rccl/rccl.h when the ROCm tree has none and the two builds see it through CPLUS_INCLUDE_PATH (no source and no CMake flag of the plain tree changed; the updated
# tree also guards the include with __has_include); ninja runs with -k 0 so ONE build shows every error; a PREBUILD SMOKE (engine start + one short request of the arm exactly
# as spec 250 runs it, <= 15 min) follows each build, so a broken build or launch ends the chain with the engine's error line instead of after hours.
# 2026-11-19 late (second RD attempt, plain build): the EMPTY stub was wrong: src/core/peer_transport.inc (included by peer_experts.cpp) declares ncclComm_t and
# decltype(&ncclSend) etc. (39 errors: 'ncclGroupEnd' was not declared in this scope ...).  Now PREREQ writes a stub with the exact RCCL/NCCL C declarations that file
# uses (rccl_stub below); the functions are taken with dlopen("librccl.so.1") + dlsym only in the rccl peer transport, so no -lrccl and no linker flag is needed.
# Verdicts: /home/dan/auto-runs/benchRD-diag-updated/ (verdicts.json, abtest.stdout), benchRD-tuneup/, benchRD-recommended/, benchRD-250-*.jsonl / .stdout.
# Estimate (minutes, box idle): sync 5, builds 2 x 5-10, ctests 3, tune-up 5 + diagnostics 20, 250 screen 30, 250 confirm 25, 251 screen 15 => about 115 min, worst case 125 (target <= 120).
S=/home/dan/clone-of-d/src; R=$S/Strata-win; L=$S/bench-updatedRD.log; O=/home/dan/auto-runs; T=$O/benchRD; PM=/mnt/pancake/benchRD
CM=/home/linuxbrew/.linuxbrew/bin/cmake
PLAIN_SRC=$PM/src-plain; UPD_SRC=$PM/src-updated; TOOLS=$PM/src-pancake
BP=$PM/build-plain; BU=$PM/build-updated; ROCM=/opt/rocm/core-10.0
DEC=$S/chainT-decide.py; T0=$(date +%s); BUDGET=${BENCH_BUDGET_S:-7200}
RD_SHA=${RD_SHA:-8b01b0c8512d388552e8601b3aff6753d05286e0}
NINJA=$S/Strata/.venv/bin/ninja; DATA=/home/dan/clone-of-d/Strata-data; GGML_DIR=/home/dan/clone-of-d/build-int/_deps/strata_llamacpp-src; RCCL_INC=
# 2026-11-19 (third RD attempt, TEST_LEDGER.md T-981: plain build ok, then the plain smoke: "server died before ready; engine log: env: '/mnt/pancake/benchRD/src-pancake/.venv/bin/python':
# No such file or directory"): abtest.py starts serve/server.py with <its own tree>/.venv/bin/python, and the downloaded harness tree $TOOLS has no .venv (the US chain
# runs the harness from Strata-win, which has one).  Now the harness (abtest, flag_diag and through ABTEST_PYTHON the server abtest starts) runs with the box's existing
# venv interpreter HPY (nothing is created under Strata-win or in the downloaded trees); PREREQ checks HPY, and right after the sync HPY must import serve.server from
# $TOOLS and $TOOLS' abtest.py must honour ABTEST_PYTHON, so this fails in seconds instead of after the plain build.
HPY=${BENCH_HARNESS_PY:-$R/.venv/bin/python}
export ABTEST_PYTHON=$HPY
# $ROCM/bin first: flag_diag.py reads `rocminfo` through PATH (2026-11-19 13:33: "no AMD graphics card could be read" because it was not on PATH)
export PATH=$ROCM/bin:$R/.venv/bin:$PATH LD_LIBRARY_PATH=$ROCM/lib HIP_VISIBLE_DEVICES=0
mkdir -p $T $PM $PM/bak
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
  say "ERROR $*"; say "WINBENCHRD_ABORTED: $*"
  service_back; say "/ free $(freekb /) KB, /mnt/pancake free $(freekb /mnt/pancake) KB"
  echo WINBENCHRD_DONE >> $L; exit 1
}
errline() {  # errline STDOUT_FILE WORKDIR: the first error-looking line of the stdout and the last one of the engine logs / narration of the failed starts (the cause is there)
  pat='error|fatal|abort|died|failed|needs|cannot|exception|terminate|no such|not found|unrecognized|unknown'
  a=$(grep -i -m1 -E "$pat" "$1" 2>/dev/null | cut -c1-240); [ -n "$a" ] || a=$(grep -m1 . "$1" 2>/dev/null | cut -c1-240)
  e=; for f in "$2"/*.engine.log "$2"/*.out; do [ -f "$f" ] || continue; e=$(grep -i -E "$pat" "$f" 2>/dev/null | tail -n 1 | cut -c1-240); [ -n "$e" ] && break; done
  echo "first error line of the stdout: [$a]; engine log: [$e]"
}
rccl_stub() {  # stdout: the minimal rccl/rccl.h for this box (ROCm 10.0 ships librccl.so.1 and no rccl headers). Strata-rdna3's src/core/peer_transport.inc (inside
  # peer_experts.cpp, STRATA_USE_HIP + Linux) uses exactly: ncclComm_t, ncclResult_t/ncclSuccess, ncclUint8, and decltype(&f) of ncclCommInitAll, ncclCommDestroy,
  # ncclGroupStart, ncclGroupEnd, ncclSend, ncclRecv, ncclGetErrorString; the functions are taken with dlopen("librccl.so.1") + dlsym only when STRATA_PEER_TRANSPORT=rccl
  # (never in a single-GPU run), so the build needs the declarations and no link flag.  Signatures and enum values as in the public nccl.h / rccl.h (BSD-licensed API).
  cat <<'EOF'
/* rccl/rccl.h stub written by bench-updatedRD.sh PREREQ: ROCm 10.0 on this box ships librccl.so.1 without headers.
 * Only the subset of the public NCCL/RCCL C API (nccl.h / rccl.h, BSD license) that Strata-rdna3 src/core/peer_transport.inc uses.
 * The functions are resolved with dlsym at run time (librccl.so.1 exports them with C linkage); nothing here is linked. */
#ifndef NCCL_H_
#define NCCL_H_
#define RCCL_H_
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ihipStream_t* hipStream_t;   /* identical to HIP's own typedef */
typedef struct ncclComm* ncclComm_t;
typedef enum { ncclSuccess = 0, ncclUnhandledCudaError = 1, ncclSystemError = 2, ncclInternalError = 3, ncclInvalidArgument = 4,
               ncclInvalidUsage = 5, ncclRemoteError = 6, ncclInProgress = 7, ncclNumResults = 8 } ncclResult_t;
typedef enum { ncclInt8 = 0, ncclChar = 0, ncclUint8 = 1, ncclInt32 = 2, ncclInt = 2, ncclUint32 = 3, ncclInt64 = 4, ncclUint64 = 5,
               ncclFloat16 = 6, ncclHalf = 6, ncclFloat32 = 7, ncclFloat = 7, ncclFloat64 = 8, ncclDouble = 8, ncclBfloat16 = 9 } ncclDataType_t;
ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist);
ncclResult_t ncclCommDestroy(ncclComm_t comm);
ncclResult_t ncclGroupStart(void);
ncclResult_t ncclGroupEnd(void);
ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, hipStream_t stream);
ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, hipStream_t stream);
const char* ncclGetErrorString(ncclResult_t result);
#ifdef __cplusplus
}
#endif
#endif /* NCCL_H_ */
EOF
}
prereq() {  # build and launch prerequisites: one ERROR naming every missing file, before anything is downloaded or built; writes the rccl stub header when ROCm has no rccl header
  miss=
  for f in $ROCM/bin/rocminfo $ROCM/bin/hipcc $ROCM/lib/llvm/bin/clang++ $CM $NINJA $ROCM/include/hip/hip_runtime.h $GGML_DIR $S/Strata/pancake-flashnext-iq2_xs-fast-hip.json \
           $DATA/packs/iq2_xs/tokenizer $DATA/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
           $DATA/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf $DATA/mtp/rt $HPY; do
    [ -e "$f" ] || miss="$miss $f"
  done
  [ -z "$miss" ] || abort "PREREQ missing files:$miss"
  "$HPY" -c 'import sys' >/dev/null 2>&1 || abort "PREREQ the harness python $HPY does not run"
  command -v rocminfo >/dev/null 2>&1 || abort "PREREQ rocminfo is not on PATH ($PATH): flag_diag could not read the card"
  rocminfo 2>/dev/null | grep -q -E "Name: +gfx" || abort "PREREQ rocminfo lists no gfx device (HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES)"
  if [ -f $ROCM/include/rccl/rccl.h ]; then say "PREREQ rccl/rccl.h found in ROCm: no stub"
  else
    { mkdir -p $PM/rccl-stub/rccl && rccl_stub > $PM/rccl-stub/rccl/rccl.h; } || abort "PREREQ cannot write the rccl stub in $PM/rccl-stub"
    RCCL_INC=$PM/rccl-stub; say "PREREQ rccl/rccl.h is not in $ROCM/include: a stub header with the RCCL declarations src/core/peer_transport.inc uses (resolved with dlopen/dlsym at run time, no -lrccl) in $RCCL_INC is put on CPLUS_INCLUDE_PATH for both builds (nothing else of the plain build changed)"
  fi
  say "PREREQ ok: ROCm tools, compiler, cmake, ninja, llama.cpp checkout, model + pack + mtp files, harness python $HPY, rocminfo sees a gfx device $(date +%T)"
}
prereq_harness() {  # right after the sync, before any config or build: the harness python imports the server of the downloaded tree (no .pyc written) and that tree's abtest.py takes ABTEST_PYTHON
  grep -q ABTEST_PYTHON $TOOLS/tools/pancake/abtest.py || abort "PREREQ harness: $TOOLS/tools/pancake/abtest.py does not read ABTEST_PYTHON: its server would start with $TOOLS/.venv/bin/python, which does not exist"
  e=$(cd $TOOLS && PYTHONDONTWRITEBYTECODE=1 "$HPY" -B -c 'import serve.server' 2>&1) || abort "PREREQ harness: $HPY cannot import serve.server from $TOOLS: $(echo "$e" | tail -n 1 | cut -c1-240)"
  say "PREREQ harness ok: $HPY imports serve.server from $TOOLS; abtest starts the server with ABTEST_PYTHON=$HPY $(date +%T)"
}
smoke() {  # smoke NAME SPEC ARM: PREBUILD SMOKE of one arm exactly as the spec runs it: one engine start + one short request (abtest, confirm tier, 1 rep), limit 900 s
  nm=$1; sp=$2; arm=$3; sd=$O/benchRD-smoke-$nm; mkdir -p $sd
  python3 -c 'import json,sys; sp=json.load(open(sys.argv[1])); json.dump({"base": sp["base"], "arms": {"smoke": sp["arms"][sys.argv[2]]}, "cases": ["short"], "reps": 1, "lossy": True, "prefill": False}, open(sys.argv[3], "w"), indent=1)' \
    $sp $arm $sd/spec.json || abort "smoke $nm: the smoke spec could not be made from $sp (arm $arm)"
  rm -f $sd/smoke.jsonl
  say "START smoke $nm $(date +%F_%T) (engine start + one short request, limit 900 s)"
  ( cd $TOOLS && timeout 900 "$HPY" tools/pancake/abtest.py $sd/spec.json --tier confirm --reps 1 --retries 0 --no-early-stop --out $sd/smoke.jsonl ) > $sd/smoke.stdout 2>&1; rc=$?
  say "END smoke $nm exit=$rc $(date +%T)"
  grep -q '"decode_tps"' $sd/smoke.jsonl 2>/dev/null || abort "smoke $nm FAILED exit=$rc (log $sd/smoke.stdout): $(errline $sd/smoke.stdout $sd/smoke)"
  say "SMOKE $nm ok: $(grep -o -m1 '"decode_tps": [0-9.]*' $sd/smoke.jsonl | head -n 1) tok/s on one short request"
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
  say "BUILD $nm $(date +%T) (/mnt/pancake free $(freekb /mnt/pancake) KB)"
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
  say "START $pfx $(date +%F_%T)"
  ( cd $TOOLS && "$HPY" tools/pancake/abtest.py $sp "$@" --out $O/benchRD-$pfx.jsonl ) > $O/benchRD-$pfx.stdout 2>&1; rc=$?
  say "END $pfx exit=$rc $(date +%T)"
  [ $rc -eq 0 ] || say "ERROR $pfx exit=$rc: $(errline $O/benchRD-$pfx.stdout $O/benchRD-$pfx)"
  return $rc
}

# ------------------------------------------------------------------ 0. (standalone: wait for chain AA and the upstream benchmark), service stopped
if [ "${BENCH_SCHED:-0}" != 1 ]; then
  until grep -q WINCHAINAA_DONE $S/redo-chainAA.log 2>/dev/null && grep -q WINBENCHUS_DONE $S/bench-updatedAA.log 2>/dev/null; do sleep 300; done
  say "chain AA and the upstream benchmark are done $(date +%T)"
else say "started by the scheduler: the AA shell is frozen, the upstream benchmark is done $(date +%T)"; fi
if systemctl --user is-active --quiet pancake.service; then
  q=0; while [ $q -lt 3 ]; do if [ "$(clients)" -gt 0 ]; then q=0; say "WAIT service has clients $(date +%T)"; else q=$((q+1)); fi; sleep 60; done
  systemctl --user stop pancake.service; say "service stopped (idle) $(date +%T)"
else say "service already stopped"; fi
echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null
wait_idle
say "SYNC start $(date +%T) df / = $(freekb /) KB, /mnt/pancake = $(freekb /mnt/pancake) KB"
[ "$(freekb /mnt/pancake)" -ge 2097152 ] || abort "SYNC: /mnt/pancake has < 2 GB free"
prereq
fetch nicejunjie/Strata-rdna3 $RD_SHA $PLAIN_SRC
fetch rocktboy1982/pancake-strata updated-strata-rdna3 $UPD_SRC
fetch rocktboy1982/pancake-strata port/rdna3-decode $TOOLS
[ -f $UPD_SRC/tools/pancake/build_updated_rdna3.sh ] || abort "SYNC: the updated tree has no tools/pancake/build_updated_rdna3.sh"
[ -f $TOOLS/tools/pancake/abtest.py ] || abort "SYNC: the pancake tools tree has no tools/pancake/abtest.py"
SPEC=$TOOLS/bench/specs/250-rdna3-updated-vs-plain.json
[ -f $SPEC ] || abort "SYNC: $SPEC is missing"
prereq_harness

# ------------------------------------------------------------------ 1. the two server configs and the spec (made first: the prebuild smokes use them), then the builds, a smoke after each
python3 $UPD_SRC/tools/pancake/box_chains/bench_updatedRD_conf.py $S/Strata/pancake-flashnext-iq2_xs-fast-hip.json $PLAIN_SRC $UPD_SRC $BP $BU \
    $T/config-plain.json $T/config-updated.json >> $L 2>&1 || abort "CONFIG: bench_updatedRD_conf.py failed"
say "CONFIG written: $T/config-plain.json $T/config-updated.json"

sed "s#@PLAIN@#$BP#; s#@UPDATED@#$BU#; s#@CONFIG_PLAIN@#$T/config-plain.json#" $SPEC > $T/250.json
wait_idle
# the plain tree is built as its README says (Manual build) plus what this box needs: ROCm / ninja / the offline llama.cpp checkout, and the rccl stub header dir on CPLUS_INCLUDE_PATH
# (RCCL_INC is empty when ROCm has the header); ninja -k 0 keeps going after a failure so that the log shows every error
build plain $BP sh -c "${RCCL_INC:+export CPLUS_INCLUDE_PATH=$RCCL_INC; }HIP_PLATFORM=amd ROCM_PATH=$ROCM HIP_PATH=$ROCM $CM -S $PLAIN_SRC -B $BP -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_PREFILL_MMQ=ON -DSTRATA_MMQ_KQUANTS=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=$ROCM -DSTRATA_GGML_DIR=$GGML_DIR -DCMAKE_MAKE_PROGRAM=$NINJA && \
  nice -n 10 $CM --build $BP -j8 --target strata -- -k 0"
smoke plain $T/250.json plain
build updated $BU env ${RCCL_INC:+CPLUS_INCLUDE_PATH=$RCCL_INC} sh $UPD_SRC/tools/pancake/build_updated_rdna3.sh $UPD_SRC $BU --target strata --target verify_ecap_test --target sampler_parity --target elementwise_parity
smoke updated $T/250.json updated

# ------------------------------------------------------------------ 2. parity ctests of the updated build (a failure is reported, it does not hide the benchmark)
wait_idle; say "START ctest $(date +%T)"
ctest --test-dir $BU -R "verify_ecap_test|sampler_parity|elementwise_parity" --output-on-failure > $T/ctest.out 2>&1; rc=$?
say "END ctest exit=$rc $(date +%T)"
if [ $rc -ne 0 ]; then say "ERROR ctest of the updated build exit=$rc (see $T/ctest.out); the benchmark goes on"; tail -n 15 $T/ctest.out >> $L; fi

# ------------------------------------------------------------------ 4. TUNE-UP (<= 30 min, read-only): hardware, PCIe link, host tuning checks, flag diagnostics of the updated build
tuneup_checks() {  # sysfs / proc reads only: no sudo, no sysctl write, nothing applied (`os_tune.sh` without argument is its check; `host_tune_ab.sh plan` prints the order)
  td=$UPD_SRC/tools/pancake; od=$O/benchRD-tuneup; mkdir -p $od
  python3 $td/flag_diag.py --hardware > $od/hardware.json 2>&1; say "HARDWARE $(tr -d '\n ' < $od/hardware.json | cut -c1-300)"
  timeout 120 python3 $td/pcie_link.py > $od/pcie_link.txt 2>&1; say "TUNEUP pcie_link exit=$? $(tail -n 3 $od/pcie_link.txt | tr '\n' ' ' | cut -c1-240)"
  timeout 120 sh $td/host_tune.sh --check > $od/host_tune_check.txt 2>&1; say "TUNEUP host_tune --check exit=$? $(tail -n 4 $od/host_tune_check.txt | tr '\n' ' ' | cut -c1-240)"
  timeout 120 sh $td/host_tune_ab.sh plan > $od/host_tune_ab_plan.txt 2>&1; say "TUNEUP host_tune_ab plan exit=$? (the A/B itself changes sysctl and is the owner's)"
  timeout 120 sh $td/os_tune.sh > $od/os_tune_check.txt 2>&1; say "TUNEUP os_tune check exit=$? $(tail -n 3 $od/os_tune_check.txt | tr '\n' ' ' | cut -c1-240)"
}
# 2026-11-19 (US chain, diag plain): `timeout 1080 sh flag_diag.sh` ended flag_diag after 18 min (2 flags) but its abtest.py / serve/server.py / strata engine
# kept running, and the next diagnostic's abtest refused to start (exit 3: all 5 flags UNTESTED).  Now: the diagnostic runs in its OWN SESSION (setsid), under
# `timeout --signal=TERM --kill-after=100` (flag_diag.py stops its running arm's process group on TERM: up to 30 + 60 s); afterwards the chain waits (<= 60 s) until
# no process of that session is left and SIGKILLs only that session's leftovers, each one logged (DIAG-REAP).  Nothing outside the session is touched.
# flag_diag.py itself: one abtest start per arm (base once), case short only, no retries, a hard limit per arm (FLAG_TIMEOUT s: UNTESTED-TIMEOUT), verdicts.json
# rewritten after every arm, parity ctest only for flags claimed exact.  The chain limit is sized from that: (flags + 1) x (FLAG_TIMEOUT + 90) + ctest + 180 s.
FLAG_TIMEOUT=${FLAG_TIMEOUT:-240}; CTEST_TIMEOUT=300
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
diag() {  # diag NAME CONFIG FLAGS [CTEST_DIR]: one abtest screen per arm (base once, then each flag; the harness of flag_diag.py), ctest parity for exact flags
  nm=$1; cf=$2; fl=$3; ct=${4:-}
  nf=$(echo "$fl" | tr ',' '\n' | grep -c .); lim=$(( (nf + 1) * (FLAG_TIMEOUT + 90) + ${ct:+$CTEST_TIMEOUT} + 180 ))
  say "START diag-$nm $(date +%F_%T) (limit $lim s: $nf flags x $FLAG_TIMEOUT s per arm + base + margin)"
  ( cd $TOOLS && run_diag_group $lim $O/benchRD-diag-$nm.stdout "$HPY" -u $UPD_SRC/tools/pancake/flag_diag.py run --config $cf --flags $fl \
      --abtest $TOOLS/tools/pancake/abtest.py --out $O/benchRD-diag-$nm --flag-timeout $FLAG_TIMEOUT --ctest-timeout $CTEST_TIMEOUT ${ct:+--ctest $ct} ); rc=$?
  say "END diag-$nm exit=$rc $(date +%T)"
  cut -c1-200 $O/benchRD-diag-$nm.stdout >> $L
  [ -f $O/benchRD-diag-$nm/verdicts.json ] || say "ERROR diag-$nm wrote no verdicts.json: $(grep -m1 . $O/benchRD-diag-$nm.stdout 2>/dev/null | cut -c1-240) (log $O/benchRD-diag-$nm.stdout; the benchmark goes on)"
  return $rc
}
tuneup_checks
diag updated $T/config-updated.json POOL_FUSE_PHASES,VERIFY_ECAP,PF_F16,RDNA_MMVQ_WAVE,KV_HOST_DMA $BU
say "TUNEUP done after $(elapsed) s"

# ------------------------------------------------------------------ 5. spec 250: screen, decide, confirm (3 reps of the decode cases) for the arms that earned it
decide() {  # decide NAME SPEC [CASES_JSON]: 0 and ${T}/NAME-confirm.json when an arm earned the confirm (cases cut to the decode cases), 1 when not
  rm -f "${T:?}/$1-confirm.json"
  python3 $DEC $O/benchRD-$1-screen.stdout $2 $T/$1-confirm.json > $T/$1-decide.txt 2>&1
  grep -E "WIN|CONFIRM|NONE|dropped" $T/$1-decide.txt | cut -c1-230 >> $L
  [ -s $T/$1-confirm.json ] || return 1
  python3 -c 'import json,sys; p=sys.argv[1]; d=json.load(open(p)); d["cases"]=json.loads(sys.argv[2]); d["drift_check"]=False; json.dump(d,open(p,"w"),indent=1)' $T/$1-confirm.json "${3:-[\"short\",\"edit\"]}"
  return 0
}
wait_idle
ab $T/250.json 250-screen --tier screen --drift-check; rc1=$?
grep -E "vs base|identical|DIFFERENT" $O/benchRD-250-screen.stdout | cut -c1-200 >> $L
if [ $rc1 -ne 0 ]; then say "ERROR spec 250 screen exit=$rc1 (log $O/benchRD-250-screen.stdout); no confirm"
elif decide 250 $T/250.json; then
  if left_ok 1800; then
    ab $T/250-confirm.json 250-confirm --tier confirm --block 3 --warmup 1 --outlier robust --reclaim-max 3 --reps 3 --no-early-stop; rc1=$?
    grep -E "vs base|identical|DIFFERENT" $O/benchRD-250-confirm.stdout | cut -c1-200 >> $L
  else say "SKIP 250 confirm: only $(( BUDGET - $(elapsed) )) s of the budget left (the screen numbers stand)"; fi
else say "250: no arm with a benefit in the screen: no confirm (dropped after pass 1: no benefit observed, n=1 pass, not a refutation)"; fi

# ------------------------------------------------------------------ 6. spec 251: the 100401 table and the learned profile against the updated build's own table and profile (screen of short + long32k, no drift arm)
SPEC2=$TOOLS/bench/specs/251-rdna3-hipblaslt-and-profile.json
rc2=0
if [ -f $SPEC2 ]; then
  sed "s#@UPDATED_SRC@#$UPD_SRC#g; s#@UPDATED@#$BU#g; s#@CONFIG_PLAIN@#$T/config-plain.json#" $SPEC2 > $T/251.json
  python3 -c 'import json,sys; p=sys.argv[1]; d=json.load(open(p)); d["cases"]=["short","long32k"]; d["drift_check"]=False; json.dump(d,open(p,"w"),indent=1)' $T/251.json
  wait_idle
  ab $T/251.json 251-screen --tier screen; rc2=$?
  grep -E "vs base|identical|DIFFERENT" $O/benchRD-251-screen.stdout | cut -c1-200 >> $L
  if [ $rc2 -eq 0 ] && decide 251 $T/251.json '["short","long32k"]'; then
    if left_ok 2400; then ab $T/251-confirm.json 251-confirm --tier confirm --reps 3 --no-early-stop; rc2=$?
    else say "251 confirm not run: only $(( BUDGET - $(elapsed) )) s of the budget left (< 2400 s); a GAIN here needs a confirm later"; fi
  fi
else say "ERROR spec 251 is missing in the synced pancake tree ($SPEC2); the table / profile arms did not run"; rc2=1; fi

# ------------------------------------------------------------------ 7. recommended-flags file from the verdicts (only exact or gate-passing GAIN flags enter it), service
if [ -f $O/benchRD-diag-updated/verdicts.json ]; then
  python3 -c 'import sys,json; sys.path.insert(0,sys.argv[1]); import tuned_flags as tf; hw=tf.detect_hardware(); v=json.load(open(sys.argv[2])); print("wrote", tf.write_recommended(hw,v,sys.argv[3]))' \
    $UPD_SRC/rdna3 $O/benchRD-diag-updated/verdicts.json $O/benchRD-recommended >> $L 2>&1 || say "ERROR the recommended-flags file was not written"
fi
say "SUMMARY spec250 exit=$rc1, spec251 exit=$rc2, box time $(elapsed) s; results in $O (benchRD-*); the recommended-flags file (not installed): $O/benchRD-recommended/"
service_back; say "/ free $(freekb /) KB, /mnt/pancake free $(freekb /mnt/pancake) KB"
echo WINBENCHRD_DONE >> $L
