#!/bin/sh
# bench-batch-leads.v1.sh - THE BOX BATCH TEST of the 15 lead branches merged on integration/leads-2026-11-21 (owner order 2026-11-21: "develop all the agents work and
# we will test after the batch is done").  NOT RUN: written on the Windows PC while the box held the flag screen.  Every lead switch is default OFF; this chain builds
# the integration branch, runs the unit / parity tests, the compile-only tools, the read-only probes, the A/B specs of the leads (flag testing protocol) and the
# profiling chains, in that order, in NEW directories only.  The ordered runbook with owner actions is TEST_LEDGER.md T-1104 and MOD_PLAN.md section 6 item 66.
#
# RULES (all enforced below): IDLE ONLY - it REFUSES (exit 3, one log line) while any busy-pattern process runs, and it does not wait for one (start it by hand when the
#   flag screen / queue is done); it writes only into NEW directories ($ROOT under /mnt/pancake, $OUT under /home/dan/auto-runs; an existing one stops it); it NEVER deletes,
#   moves or edits anything of the queue, the frozen queue files (sweep_candidates.json, us-triage*, retest_groups.json) or an older build; it never kills a foreign process;
#   a failed BUILD stops the chain with ERROR (nothing after it can run); every other step logs START / END rc and the chain CONTINUES after a failed step; the last
#   line of $OUT/batch.log is BATCHLEADS_DONE (also after an abort: the line before is BATCHLEADS_ABORTED <why>).  No sudo anywhere.
#
#   sh tools/pancake/box_chains/bench-batch-leads.v1.sh                      all phases
#   BL_PHASES="1 2" sh ...                                                   only phases 1 and 2 (1 build+tests, 2 compile-only, 3 probes, 4 A/B specs, 5 profiling)
#   BL_DRY=1 sh ...                                                          print the plan (phases, steps, spec order) and exit: nothing is built, no pgrep, no service touched
#   BL_ONLY_SPECS="258 282" sh ...                                           phase 4 restricted to these spec numbers (the order below is kept)
#   BL_REF=integration/leads-2026-11-21  BL_STOP_SERVICE=1 (stop pancake.service after 3 min without a client; restart it at the end only if this script stopped it)
#   BL_CMAKE_REF=/home/dan/clone-of-d/src/build-win   the build whose CMakeCache.txt supplies the compiler and STRATA_* options (default as shown)
#
# ALL RESULTS OF THIS RUN ARE PCIe GEN3 (the owner upgrades the link after this work): expert-streaming / swap / copy / prefetch verdicts must be re-measured on Gen4.
# gpu_snapshot.py (phase 3) records the link state before and after; read it with every number.
#
# OWNER ACTIONS (the script never does these; each is marked in the log with a NEEDS-OWNER line when its step cannot run without it):
#   root: ROCm 10.2 install (/opt/rocm/core-10.2; spec 270 arms and the 10.2 hipBLASLt table step), power profile COMPUTE (pp_power_profile_mode index 5, LEADS.md T14),
#   debugfs hqds (S2 pipe layout needs root: `cat /sys/kernel/debug/kfd/hqds` as root; without it the STARTQUEUE lines say shared=NA), stopping pancake.service (or BL_STOP_SERVICE=1).
#   Not root: the verdicts (decisions go to MOD_PLAN.md sections 3-5, numbers to TEST_LEDGER.md rows, never a new document).
S=${BL_S:-/home/dan/clone-of-d/src}
TS=$(date +%Y%m%d_%H%M%S)
ROOT=${BL_ROOT:-/mnt/pancake/batchleads-$TS}
OUT=${BL_OUT:-/home/dan/auto-runs/batchleads-$TS}
L=$OUT/batch.log
REF=${BL_REF:-integration/leads-2026-11-21}
REPO_URL=${BL_REPO_URL:-https://github.com/rocktboy1982/pancake-strata.git}
ROCM=${BL_ROCM:-/opt/rocm/core-10.0}
CM=${BL_CMAKE:-/home/linuxbrew/.linuxbrew/bin/cmake}
NINJA=${BL_NINJA:-$S/Strata/.venv/bin/ninja}
CMAKE_REF=${BL_CMAKE_REF:-/home/dan/clone-of-d/src/build-win}
DATA=/home/dan/clone-of-d/Strata-data
PHASES=${BL_PHASES:-"1 2 3 4 5"}
SRC=$ROOT/src                     # the synced tree of $REF (tar xmf: mtimes reset)
BUILD=$SRC/build-hip              # default flags: the conventional name tools/pancake/test_phase.sh and the presets expect
KVPF=$ROOT/build-kvpf             # -DSTRATA_ATTN_KV_PREFETCH=ON (spec 275)
HF_ROOT=$ROOT/hipflags            # hipflags-build.v1.sh builds (specs 260 / 261)
ABL=$ROOT/ablate-f16              # bench-ablate-f16.sh (spec 276)
LIM_SCREEN=${BL_LIM_SCREEN:-5400}; LIM_RESCREEN=${BL_LIM_RESCREEN:-5400}
HIGH_BAR=${BL_HIGH_BAR:-2.0}      # flag testing protocol (owner, 2026-11-20): a case >= 2 % in run 1 (any case) earns run 2; the verdict needs two runs
T0=$(date +%s)

# ------------------------------------------------------------------ the plan (one place; BL_DRY=1 prints it)
# spec order: cheap first (arms x cases x reps ascending, host-rule switches before kernel switches, build-vs-build and ablation last).  GATE = the step whose rc must
# be 0 before the spec runs ("cpu" = ctest -L cpu of phase 1; a parity name = that GPU test of phase 1; none = no gate).  Numbers are the FINAL renumbered spec numbers.
SPECS="253:hwq-decode:none 254:hwq-long64k:none 273:decode-order-small:cpu 271:pcopy-rest-usage:cpu 272:swap-budget-residency:cpu 259:q7-tune-paired-screen:cpu
256:pf-no-tail-chunk:cpu 262:q4-swap-grace-count-kept-screen:cpu 224:q2-prompt-tail-screen:cpu 225:q5-mtp-kv-fp16-screen:cpu 257:snap-state-f16:cpu
258:mtp-block-verify:spec_block_verify_parity 282:q3-mtp-slots-screen:s2_expert_grouped_parity 283:q3-mtp-slots-installs-screen:s2_expert_grouped_parity
274:mtp-head-host:cpu 277:reduce-f32:cpu 280:gdn-wmma-prefill:cpu 255:hipblaslt-heuristic-arms:cpu 263:census-mapped-switches:none 270:copy-batch-rocm102:cpu
264:verify-gemv-chains:rdna_chains_parity 275:attn-kv-prefetch:qsa_kv_prefetch_parity 260:hip-vopd-build-ab:hipflags_build 261:hip-extra-flags-build-ab:hipflags_build
276:f16-ablation-ladder:none"
PARITY="rdna_mmvq_parity rdna_gr_parity verify_ecap_test s2_expert_grouped_parity qsa_kv_prefetch_parity hip_gdn_chunked_ck rdna_chains_parity spec_block_verify_parity"
if [ -n "${BL_DRY:-}" ]; then
  echo "PLAN root=$ROOT out=$OUT ref=$REF phases='$PHASES'"
  echo "PHASE 1 sync $REF; configure + build (default flags) $BUILD; ctest -L cpu; GPU parity: $PARITY (qsa_kv_prefetch_parity in build-kvpf)"
  echo "PHASE 2 compile-only: kres-t9.sh (kernel resources + ISA audit), hipflags-probe.v1.sh, hipflags-build.v1.sh x3 + hipflags-repeat.v1.sh, isa_default_check.py (when a git clone)"
  echo "PHASE 3 read-only probes: gpu_snapshot.py idle, u4-valu-rate-chain.v1.sh (valu_rate + wmma_rate), probe-chained.v1.sh (retained replay), hipblaslt table tune is NOT run"
  echo "PHASE 4 A/B specs (screen run 1, HIGH >= ${HIGH_BAR} % any case -> rescreen run 2):"
  for e in $SPECS; do echo "  spec ${e%%:*} ${e#*:}" | sed 's/:/ gate=/'; done
  echo "PHASE 5 profiling: prof-t11.v1.sh, trace-t12.v1.sh, att-t13.v1.sh (each needs PROF_CMD; see its header), gpu_snapshot.py load/after"
  echo BATCHLEADS_DONE
  exit 0
fi

# ------------------------------------------------------------------ helpers
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[b]ench-sweep|[b]ench-sched|[b]ench-retest|[b]ench-final|[r]un-queue-us|[w]in-chain[A-Za-z0-9]*\.sh|[v]alu_rate|[w]mma_rate|[r]etained_replay|[r]ocprofv3|[h]ipflags-build|[k]res-t9|[b]ench-hwq|[b]ench-ablate" >/dev/null 2>&1; }
aa_live() { for p in $(pgrep -f "[r]edo-chainAA\.sh" 2>/dev/null); do case "$(ps -o stat= -p $p 2>/dev/null)" in T*) ;; *) return 0;; esac; done; return 1; }   # a FROZEN AA shell is not a job
bz() { busy || aa_live; }
if bz; then
  echo "REFUSED $(date +%F_%T): a busy-pattern process is running (flag screen / queue / build / test / probe). Nothing was done." >&2
  exit 3
fi
[ -e "$OUT" ] && { echo "REFUSED: $OUT exists (every run uses new directories)" >&2; exit 2; }
[ -e "$ROOT" ] && { echo "REFUSED: $ROOT exists (every run uses new directories)" >&2; exit 2; }
mkdir -p "$OUT" "$ROOT" "$OUT/specs" || { echo "cannot create $OUT / $ROOT" >&2; exit 2; }
say() { echo "$(date +%T) $*" >> "$L"; echo "$*"; }
elapsed() { echo $(( $(date +%s) - T0 )); }
freekb() { df -Pk "$1" | awk 'NR==2{print $4}'; }
STOPPED=
finish() {
  [ -n "$STOPPED" ] && { systemctl --user start pancake.service 2>/dev/null && say "pancake.service started again (this script had stopped it)"; }
  say "free / $(freekb /) KB, $ROOT $(freekb "$ROOT") KB, box time $(elapsed) s, results $OUT"
  say "BATCHLEADS_DONE"
}
abort() { say "ERROR $*"; say "BATCHLEADS_ABORTED $*"; finish; exit 1; }
CUR=
on_signal() { say "signal received"; [ -n "$CUR" ] && kill -TERM $CUR 2>/dev/null; say "BATCHLEADS_ABORTED signal"; finish; exit 1; }
trap on_signal HUP INT TERM
say "START bench-batch-leads.v1 ref=$REF root=$ROOT out=$OUT phases='$PHASES' (PCIe Gen3 run: re-measure the link-bound verdicts on Gen4)"
[ "$(freekb "$ROOT")" -ge 20971520 ] || abort "START: $ROOT has less than 20 GB free ($(freekb "$ROOT") KB): nothing built"
[ "$(freekb /)" -ge 307200 ] || abort "START: / has less than 300 MB free"
has_phase() { case " $PHASES " in *" $1 "*) return 0;; esac; return 1; }

# the service: stopped only with BL_STOP_SERVICE=1 and only after 3 minutes without a client; started again at the end only if this script stopped it
if systemctl --user is-active --quiet pancake.service 2>/dev/null; then
  if [ -n "${BL_STOP_SERVICE:-}" ]; then
    q=0; while [ $q -lt 3 ]; do
      if [ "$(ss -Htn state established '( sport = :11437 )' 2>/dev/null | wc -l)" -gt 0 ]; then q=0; say "WAIT pancake.service has clients"; else q=$((q+1)); fi; sleep 60; done
    systemctl --user stop pancake.service && STOPPED=1 && say "pancake.service stopped (idle)"
  else
    abort "pancake.service is running: stop it (or BL_STOP_SERVICE=1); an A/B beside a running engine would be void"
  fi
fi

# step NAME LIMIT_SECONDS CMD...   run CMD (a sh -c string), log START / END rc, never abort; rc kept in $OUT/rc.NAME
step() {
  nm=$1; lim=$2; shift 2
  [ "$(freekb /)" -ge 307200 ] || { say "SKIP $nm: / has less than 300 MB free"; echo 9 > "$OUT/rc.$nm"; return 9; }
  say "START $nm $(date +%F_%T) (limit $lim s)"
  timeout -k 100 "$lim" sh -c "$*" > "$OUT/$nm.log" 2>&1 & CUR=$!
  wait $CUR; rc=$?; CUR=
  echo $rc > "$OUT/rc.$nm"
  say "END $nm exit=$rc $(date +%T)"
  [ $rc -eq 0 ] || say "ERROR $nm exit=$rc: $(grep -i -m1 -E 'error|fatal|died|failed|cannot|FAIL' "$OUT/$nm.log" | cut -c1-240)"
  return $rc
}
ok() { [ "$(cat "$OUT/rc.$1" 2>/dev/null)" = "0" ]; }
export PATH=$ROCM/bin:$PATH LD_LIBRARY_PATH=$ROCM/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
export ABTEST_SETTLE_AFTER_LONG_S=${ABTEST_SETTLE_AFTER_LONG_S:-120} ABTEST_READY_MAX_S=${ABTEST_READY_MAX_S:-900} ABTEST_PREREAD=${ABTEST_PREREAD:-2}

# cmake options: the reference build's cache supplies the compilers and every STRATA_* option; the batch adds only what it needs
cmake_opts() {
  CMX=
  if [ -f "$CMAKE_REF/CMakeCache.txt" ]; then
    for k in CMAKE_HIP_COMPILER CMAKE_CXX_COMPILER CMAKE_C_COMPILER CMAKE_PREFIX_PATH CMAKE_HIP_FLAGS; do
      v=$(grep -m1 "^$k:" "$CMAKE_REF/CMakeCache.txt" | sed 's/^[^=]*=//'); [ -n "$v" ] && CMX="$CMX -D$k=$v"; done
    for kv in $(grep -E '^STRATA_[A-Za-z0-9_]+:(BOOL|STRING|PATH)=' "$CMAKE_REF/CMakeCache.txt" | grep -v -E '^STRATA_(HIP_VOPD|HIP_EXTRA_FLAGS|ATTN_KV_PREFETCH)' | sed 's/:[A-Z]*=/=/'); do CMX="$CMX -D$kv"; done
    say "CMAKE options copied from $CMAKE_REF/CMakeCache.txt"
  else say "NOTE no $CMAKE_REF/CMakeCache.txt: default cmake line (set BL_CMAKE_EXTRA for the compilers / STRATA_GGML_DIR)"; fi
  CMX="$CMX ${BL_CMAKE_EXTRA:-}"
}

# ------------------------------------------------------------------ PHASE 1: sync, build (default flags), cpu tests, GPU parity
if has_phase 1; then
  say "PHASE 1 sync + build + tests"
  SHA=$(timeout 60 git ls-remote "$REPO_URL" "refs/heads/$REF" 2>/dev/null | cut -f1)
  TGZ=$ROOT/src.tgz
  if [ -n "$SHA" ] && timeout 1200 curl -sfL -o "$TGZ.part" "https://codeload.github.com/rocktboy1982/pancake-strata/tar.gz/$SHA" && tar tzf "$TGZ.part" >/dev/null 2>&1; then
    mv "$TGZ.part" "$TGZ"; mkdir -p "$SRC" && tar xzmf "$TGZ" --no-same-owner --strip-components=1 -C "$SRC" || abort "SYNC: tar failed"   # xmf: mtimes reset, else ninja keeps stale objects
    say "SYNC $REF $SHA -> $SRC"
  else abort "SYNC: download of $REF failed (sha '$SHA')"; fi
  [ -f "$SRC/CMakeLists.txt" ] && [ -f "$SRC/bench/specs/258-mtp-block-verify.json" ] || abort "SYNC: $SRC is not the integration tree"
  cmake_opts
  CMAKE_LINE="$CM -S $SRC -B $BUILD -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=$NINJA -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DSTRATA_BUILD_TESTS=ON $CMX"
  say "CMAKE $CMAKE_LINE"
  step configure 1800 "$CMAKE_LINE" || abort "BUILD: cmake configure failed (see $OUT/configure.log)"
  JOBS=${JOBS:-8}
  step build 14400 "$CM --build $BUILD --target strata strata_tests -j$JOBS" || abort "BUILD: strata / strata_tests failed (see $OUT/build.log)"
  # the GPU parity targets that exist in this configuration (a missing one is logged, not an error); hip_gdn_chunked_ck is a test of the target hip_gdn_chunked
  PT=
  for t in rdna_mmvq_parity rdna_gr_parity verify_ecap_test s2_expert_grouped_parity hip_gdn_chunked rdna_chains_parity spec_block_verify_parity; do
    if "$NINJA" -C "$BUILD" -t targets all 2>/dev/null | grep -q "^$t:"; then PT="$PT $t"; else say "NOTE target $t does not exist in this configuration"; fi
  done
  step build_parity 7200 "$CM --build $BUILD --target $PT -j${JOBS:-8}"
  step ctest_cpu 3600 "cd $BUILD && ctest -L cpu --output-on-failure --timeout 120"
  [ -s "$OUT/rc.ctest_cpu" ] && cp "$OUT/rc.ctest_cpu" "$OUT/rc.cpu"
  for t in rdna_mmvq_parity rdna_gr_parity verify_ecap_test s2_expert_grouped_parity hip_gdn_chunked_ck rdna_chains_parity spec_block_verify_parity; do
    if ok build_parity; then step "$t" 1800 "cd $BUILD && ctest -R '^$t\$' --output-on-failure --timeout 900"; else say "SKIP parity $t: build_parity failed"; echo 9 > "$OUT/rc.$t"; fi
  done
  # U8: the prefetch kernel is a build macro: its own build directory, the default build stays as it is
  step configure_kvpf 1800 "$(echo "$CMAKE_LINE" | sed "s# -B $BUILD # -B $KVPF #") -DSTRATA_ATTN_KV_PREFETCH=ON"
  ok configure_kvpf && step build_kvpf 7200 "$CM --build $KVPF --target strata qsa_kv_prefetch_parity -j${JOBS:-8}" && \
    step qsa_kv_prefetch_parity 900 "cd $KVPF && ctest -R '^qsa_kv_prefetch_parity\$' --output-on-failure --timeout 600"
  # the default-path control of U7 / U8 and the sweep of the existing GPU tests (bitwise, switch off): a failure here is a regression of the default
  step ctest_gpu 7200 "cd $BUILD && ctest -L gpu --output-on-failure --timeout 900 -E '^(ple_parity|platform_memory_test)\$'"
fi

# (kres-t9.sh and prof-*.sh match the substring 'bench-' in ANY command line, including this script's own: they are started with KRES_SKIP_BUSY=1 / PROF_SKIP_BUSY=1; this script has
#  already refused a busy box and runs one step at a time)
# ------------------------------------------------------------------ PHASE 2: compile-only tools (no GPU run; they build in their own NEW directories)
if has_phase 2; then
  say "PHASE 2 compile-only tools"
  [ -d "$SRC" ] || abort "phase 2 needs the synced tree $SRC (run phase 1 first)"
  BC=$SRC/tools/pancake/box_chains
  cmake_opts; export HF_CMAKE_EXTRA="$CMX"
  # T9 + T10: a gfx1100 build with the kernel-resource remarks, the code object's notes and the ISA audit (its own build dir under $OUT)
  step kres_t9 14400 "cd $SRC && SRC=$SRC OUT=$OUT/kres STRATA_T9_RUN=1 KRES_SKIP_BUSY=1 sh $BC/kres-t9.sh"
  # S4 / T8: which flags this compiler accepts; three builds (default, default-b = same flags again, vopd-off) = the binaries of spec 260, then the repeatability check
  step hipflags_probe 900 "HF_ROOT=$HF_ROOT sh $BC/hipflags-probe.v1.sh"
  for tag in default default-b; do step "hipflags_build_$tag" 14400 "HF_ROOT=$HF_ROOT HF_SRC=$SRC HF_TAG=$tag sh $BC/hipflags-build.v1.sh"; done
  step hipflags_build_vopd-off 14400 "HF_ROOT=$HF_ROOT HF_SRC=$SRC HF_TAG=vopd-off HF_VOPD=off sh $BC/hipflags-build.v1.sh"
  step hipflags_build_extra-ilp 14400 "HF_ROOT=$HF_ROOT HF_SRC=$SRC HF_TAG=extra-ilp HF_EXTRA='-mllvm -amdgpu-sched-strategy=max-ilp' HF_EXTRA_SCOPE=all sh $BC/hipflags-build.v1.sh"
  step hipflags_build_extra-cumode 14400 "HF_ROOT=$HF_ROOT HF_SRC=$SRC HF_TAG=extra-cumode HF_EXTRA='-mcumode' HF_EXTRA_SCOPE=all sh $BC/hipflags-build.v1.sh"
  if ok hipflags_build_default && ok hipflags_build_default-b && ok hipflags_build_vopd-off; then echo 0 > "$OUT/rc.hipflags_build"; else echo 1 > "$OUT/rc.hipflags_build"; fi
  step hipflags_repeat 7200 "HF_ROOT=$HF_ROOT HF_BUILD=$HF_ROOT/build-default sh $BC/hipflags-repeat.v1.sh"
  # the default path of every touched kernel is unchanged: needs a git clone for `git archive <rev>`; the PC did this with SDK 6.2 (T-1104), the box repeats it with LLVM 24
  if [ -d "$SRC/.git" ]; then step isa_default 7200 "cd $SRC && python3 tools/pancake/isa_default_check.py --rev ${BL_ISA_REV:-origin/port/rdna3-decode} --clang $ROCM/lib/llvm/bin/clang++ --files src/kernels/cuda/elementwise.cu src/kernels/cuda/gdn.cu src/kernels/cuda/ple.cu src/kernels/cuda/qsa.cu src/kernels/cuda/sampler.cu src/kernels/cuda/rdna_mmvq.cu src/kernels/cuda/rdna_gr.cu src/kernels/cuda/qsa_decode_attn.cu"
  else say "SKIP isa_default: $SRC is a tarball sync, not a git clone (do it on a clone: python3 tools/pancake/isa_default_check.py --rev <port ref>)"; fi
fi

# ------------------------------------------------------------------ PHASE 3: read-only probes (idle box, seconds to minutes of GPU time)
if has_phase 3; then
  say "PHASE 3 probes"
  [ -d "$SRC" ] || abort "phase 3 needs the synced tree $SRC"
  BC=$SRC/tools/pancake/box_chains
  step snap_idle 300 "cd $SRC && python3 tools/pancake/gpu_snapshot.py --phase idle --out $OUT/snap"
  step valu_wmma_rate 1800 "cd $SRC && sh $BC/u4-valu-rate-chain.v1.sh $SRC"
  step retained_replay 3600 "cd $SRC && OUT=$OUT/probe-chained sh $BC/probe-chained.v1.sh"
fi

# ------------------------------------------------------------------ PHASE 4: the A/B specs of the leads, flag testing protocol (owner 2026-11-20): screen run 1 (--tier screen --drift-check), an arm with a
# case >= HIGH_BAR % earns RESCREEN run 2 (the same screen again, arms = base + the HIGH arms); a verdict needs both runs; never drop on one run, one bundle or a text difference
# (a text difference is flagged, the quality gate decides); stock is compared only in the final comparison.  Arms live in the specs; the registry for the sweep machinery
# is tools/pancake/box_chains/sweep_candidates_leads.json (the frozen sweep_candidates.json is not touched).
fill_spec() {   # fill_spec FILE -> $OUT/specs/FILE (placeholders and the author's build paths replaced by this run's builds; the repo file is untouched)
  f=$1
  sed "s#@UPDATED_SRC@#$SRC#g; s#@UPDATED@#$BUILD#g; s#@CONFIG_PLAIN@#$SRC/configs/pancake/flashnext-iq2_xs-fast-hip.json#g; \
       s#/home/dan/clone-of-d/src/build-win/strata#$BUILD/strata#g; s#/home/dan/clone-of-d/src/build-kvpf/strata#$KVPF/strata#g; \
       s#/mnt/pancake/build-lead-t7s13/strata#$BUILD/strata#g; s#/mnt/pancake/hipflags/#$HF_ROOT/#g; s#/mnt/pancake/ablate-f16/#$ABL/#g" \
      "$SRC/bench/specs/$f.json" > "$OUT/specs/$f.json"
}
ab() {   # ab NAME LIMIT SPECFILE abtest-flags...
  nm=$1; lim=$2; sp=$3; shift 3
  [ "$(freekb /)" -ge 307200 ] || { say "ERROR $nm: / has less than 300 MB free: not run"; return 9; }
  if bz; then say "SKIP $nm: the box became busy"; return 9; fi
  step "ab_$nm" "$lim" "cd $SRC && ABTEST_PYTHON=python3 python3 tools/pancake/abtest.py $sp $* --out $OUT/ab-$nm.jsonl"
  rc=$?
  grep -E "vs .*(better|worse|no difference)|verdict|DRIFT|ORDER EFFECT|STARTQUEUE|STOPQUEUE|INERT|ACTIVE|DECLINED" "$OUT/ab_$nm.log" | cut -c1-220 >> "$L"
  return $rc
}
if has_phase 4; then
  say "PHASE 4 A/B specs (screen, then rescreen of the HIGH arms)"
  [ -d "$SRC" ] || abort "phase 4 needs the synced tree $SRC"
  [ -x "$BUILD/strata" ] || abort "phase 4 needs $BUILD/strata (phase 1)"
  cd "$SRC" || abort "no $SRC"
  python3 tools/pancake/spec_lint.py --dir bench/specs --quiet > "$OUT/spec_lint.txt" 2>&1; say "HEADER spec_lint exit=$?"
  grep -q ABTEST_QUEUEDIAG tools/pancake/abtest.py && { export ABTEST_QUEUEDIAG=1; say "HEADER ABTEST_QUEUEDIAG=1 (S2 queue layout lines; hqds needs root: shared=NA without it)"; }
  export ABTEST_DISPATCH_REPORT=1; say "HEADER ABTEST_DISPATCH_REPORT=1 (U3 served-path census lines in every spec)"
  step snap_before_ab 300 "cd $SRC && python3 tools/pancake/gpu_snapshot.py --phase before-ab --out $OUT/snap"
  for e in $SPECS; do
    num=${e%%:*}; rest=${e#*:}; name=${rest%%:*}; gate=${rest#*:}; f=$num-$name
    if [ -n "${BL_ONLY_SPECS:-}" ]; then case " $BL_ONLY_SPECS " in *" $num "*) ;; *) continue;; esac; fi
    case $gate in
      none) ;;
      cpu) ok cpu || { say "SKIP spec $num: the cpu tests did not pass (gate cpu)"; continue; } ;;
      hipflags_build) ok hipflags_build || { say "SKIP spec $num: the hipflags builds did not all succeed (gate)"; continue; } ;;
      *) ok "$gate" || { say "SKIP spec $num: parity $gate did not pass (gate)"; continue; } ;;
    esac
    [ -f "bench/specs/$f.json" ] || { say "SKIP spec $num: bench/specs/$f.json is missing in $REF"; continue; }
    case $num in
      270) [ -d /opt/rocm/core-10.2 ] || { say "NEEDS-OWNER spec 270: ROCm 10.2 is not installed (/opt/rocm/core-10.2; root): the arms need it"; continue; } ;;
      276) step ablate_build 28800 "cd $SRC && ABL_BRANCH=$REF ABL_PM=$ABL ABL_OUTROOT=$OUT/ablate ABL_PHASE=build sh tools/pancake/box_chains/bench-ablate-f16.sh"; ok ablate_build || { say "SKIP spec 276: the ablation builds failed"; continue; } ;;
    esac
    fill_spec "$f"
    sp=$OUT/specs/$f.json
    ab "$num-s1" "$LIM_SCREEN" "$sp" --tier screen --drift-check || { say "spec $num screen run 1 failed: no verdict (the next spec still runs)"; continue; }
    rm -f "$OUT/specs/$num-high.json"
    python3 tools/pancake/box_chains/chainT-decide.py "$OUT/ab_$num-s1.log" "$sp" "$OUT/specs/$num-high.json" --bar "$HIGH_BAR" --lossy-bar "$HIGH_BAR" > "$OUT/decide-$num.txt" 2>&1
    grep -E "CONFIRM|NONE|WIN|dropped" "$OUT/decide-$num.txt" | cut -c1-230 >> "$L"
    if [ -s "$OUT/specs/$num-high.json" ]; then
      ab "$num-s2" "$LIM_RESCREEN" "$OUT/specs/$num-high.json" --tier screen --drift-check
      say "spec $num: run 2 done; verdict needs run 1 + run 2 (two-run rule): record both in TEST_LEDGER.md"
    else say "spec $num: no arm reached ${HIGH_BAR} % on a case in run 1: no rescreen (n=1: not a refutation, record as 'no benefit observed in run 1')"; fi
  done
  step snap_after_ab 300 "cd $SRC && python3 tools/pancake/gpu_snapshot.py --phase after-ab --out $OUT/snap"
fi

# ------------------------------------------------------------------ PHASE 5: profiling chains T11 / T12 / T13 (last: the profiler slows the engine); each needs PROF_CMD (one short request to the built engine)
if has_phase 5; then
  say "PHASE 5 profiling chains"
  [ -d "$SRC" ] || abort "phase 5 needs the synced tree $SRC"
  BC=$SRC/tools/pancake/box_chains
  if [ -z "${PROF_CMD:-}" ]; then
    say "NEEDS-OWNER phase 5: set PROF_CMD to the engine command that answers one short request (header of $BC/prof-t11.v1.sh); T11 / T12 / T13 not run"
  else
    step prof_t11 3600 "cd $SRC && PROF_ROOT=$OUT/prof PROF_SKIP_BUSY=1 STRATA_T11_RUN=1 sh $BC/prof-t11.v1.sh"
    step trace_t12 3600 "cd $SRC && PROF_ROOT=$OUT/prof PROF_SKIP_BUSY=1 STRATA_T12_RUN=1 sh $BC/trace-t12.v1.sh"
    step att_t13 3600 "cd $SRC && PROF_ROOT=$OUT/prof PROF_SKIP_BUSY=1 STRATA_T13_RUN=1 sh $BC/att-t13.v1.sh"
  fi
  step snap_end 300 "cd $SRC && python3 tools/pancake/gpu_snapshot.py --phase end --out $OUT/snap"
fi
finish
