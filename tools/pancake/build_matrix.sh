#!/usr/bin/env bash
# tools/pancake/build_matrix.sh - build-matrix check (MOD_PLAN item 92).
#
# Configures (and by default links `strata` in) several option sets, each in its own build dir under
# build-matrix/<row>, so a change that breaks one combination is caught before push.  It was written after a
# STRATA_PREFILL_MMQ=OFF HIP build stopped linking for weeks unnoticed (no build dir had that combination).
#
#   tools/pancake/build_matrix.sh                 # every row: configure + build `strata`
#   tools/pancake/build_matrix.sh --configure-only
#   tools/pancake/build_matrix.sh hip-nommq cuda-86
#   tools/pancake/build_matrix.sh --tests hip-tests   # every test target, then `ctest -L cpu` (item 93)
#   JOBS=6 tools/pancake/build_matrix.sh          # parallel jobs per build (default 12)
#   tools/pancake/build_matrix.sh --head          # build the committed HEAD from a detached worktree
#                                                 # (build-matrix/src), not the shared working tree: other sessions'
#                                                 # half-finished edits cannot break the check (seen 2026-11-09)
#
# Rows:
#   hip-mmq      HIP gfx1100, STRATA_PREFILL_MMQ=ON  (the daily build)
#   hip-nommq    HIP gfx1100, STRATA_PREFILL_MMQ=OFF (the stub path)
#   hip-tests    HIP gfx1100 + STRATA_BUILD_TESTS=ON (builds `strata` only here; `--tests` also builds `strata_tests`,
#                every executable a registered test runs - not `all`: strata-poly does not compile on HIP - and
#                runs `ctest -L cpu --output-on-failure`: the tests that need no GPU and no model files, with the GPUs
#                hidden; the `gpu` and `model` labels are run by hand.  Pass/fail/skip counts go in the summary, the
#                ctest output in build-matrix/<row>.ctest.log.  CTEST_JOBS=N runs N tests at once, default 1)
#   cuda-86      CUDA sm_86 (RTX 3060 line), skipped when nvcc is missing
#
# Uses ccache when it is installed.  Exit status: 0 when every selected row passed, 1 otherwise.  Logs:
# build-matrix/<row>.log.  Do not run while a benchmark is measuring the machine (tools/pancake/abtest.py marks such
# runs NOISY and repeats them).
set -u
cd "$(dirname "$0")/../.." || exit 2
REPO=$(pwd)
OUT=${BUILD_MATRIX_DIR:-$REPO/build-matrix}
JOBS=${JOBS:-12}
CTEST_JOBS=${CTEST_JOBS:-1}
ROCM=${ROCM_PATH:-/opt/rocm/core-10.0}
NINJA=${NINJA:-$REPO/.venv/bin/ninja}
CONFIGURE_ONLY=0
HEAD_ONLY=0
TESTS=0
ROWS=()
for a in "$@"; do
  case "$a" in
    --configure-only) CONFIGURE_ONLY=1 ;;
    --tests) TESTS=1 ;;
    --head) HEAD_ONLY=1 ;;
    hip-mmq|hip-nommq|hip-tests|cuda-86) ROWS+=("$a") ;;
    -h|--help) sed -n '2,29p' "$0"; exit 0 ;;
    *) echo "unknown argument: $a (rows: hip-mmq hip-nommq hip-tests cuda-86)" >&2; exit 2 ;;
  esac
done
[ ${#ROWS[@]} -eq 0 ] && ROWS=(hip-mmq hip-nommq hip-tests cuda-86)
[ -x "$NINJA" ] || NINJA=$(command -v ninja || true)
GEN=(); [ -n "$NINJA" ] && GEN=(-G Ninja -DCMAKE_MAKE_PROGRAM="$NINJA")
LAUNCH=(); command -v ccache >/dev/null 2>&1 && LAUNCH=(-DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_HIP_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache)
HIP=(-DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_PREFIX_PATH="$ROCM"
     -DCMAKE_HIP_COMPILER="$ROCM/lib/llvm/bin/clang++" -DSTRATA_NATIVE_EXPERTS=ON)
mkdir -p "$OUT"
SRC=$REPO
if [ $HEAD_ONLY -eq 1 ]; then
  SRC=$OUT/src
  if [ -d "$SRC/.git" ] || [ -f "$SRC/.git" ]; then git -C "$SRC" checkout -q --detach "$(git rev-parse HEAD)" || exit 2
  else git worktree add -q --detach "$SRC" HEAD || exit 2; fi
  echo "building HEAD $(git rev-parse --short HEAD) from $SRC"
fi
fail=0
summary=()
for row in "${ROWS[@]}"; do
  tag=$row; [ $HEAD_ONLY -eq 1 ] && tag=head-$row
  dir=$OUT/$tag; log=$OUT/$tag.log; env=(ROCM_PATH="$ROCM"); targets=(--target strata)
  case "$row" in
    hip-mmq)   opts=("${HIP[@]}" -DSTRATA_PREFILL_MMQ=ON) ;;
    hip-nommq) opts=("${HIP[@]}" -DSTRATA_PREFILL_MMQ=OFF) ;;
    hip-tests) opts=("${HIP[@]}" -DSTRATA_PREFILL_MMQ=ON -DSTRATA_BUILD_TESTS=ON); [ $TESTS -eq 1 ] && targets=(--target strata strata_tests) ;;
    cuda-86)
      if [ ! -x /usr/local/cuda/bin/nvcc ] && ! command -v nvcc >/dev/null 2>&1; then
        summary+=("$row: SKIP (no nvcc)"); continue
      fi
      env+=(PATH="/usr/local/cuda/bin:$PATH")
      opts=(-DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF -DCMAKE_CUDA_ARCHITECTURES=86-real -DSTRATA_NATIVE_EXPERTS=ON) ;;
  esac
  start=$(date +%s)
  if ! env "${env[@]}" cmake -S "$SRC" -B "$dir" "${GEN[@]}" -DCMAKE_BUILD_TYPE=Release "${LAUNCH[@]}" "${opts[@]}" >"$log" 2>&1; then
    summary+=("$row: CONFIGURE FAILED ($log)"); fail=1; continue
  fi
  if [ $CONFIGURE_ONLY -eq 0 ]; then
    if ! env "${env[@]}" cmake --build "$dir" "${targets[@]}" -j "$JOBS" >>"$log" 2>&1; then
      first=$(grep -m1 -E "error:|undefined reference" "$log" || true)
      summary+=("$row: BUILD FAILED - ${first:-see $log}"); fail=1; continue
    fi
  fi
  if [ "$row" = hip-tests ] && [ $TESTS -eq 1 ] && [ $CONFIGURE_ONLY -eq 0 ]; then
    clog=$OUT/$tag.ctest.log
    (cd "$dir" && ctest -L cpu --output-on-failure --parallel "$CTEST_JOBS") >"$clog" 2>&1
    crc=$?
    # ctest's per-test lines: "  12/150 Test  #34: name ......   Passed    0.01 sec" / "***Skipped" / "***Failed" ...
    npass=$(grep -cE '^ *[0-9]+/[0-9]+ Test +#[0-9]+: .* Passed ' "$clog")
    nskip=$(grep -cE '^ *[0-9]+/[0-9]+ Test +#[0-9]+: .*\*\*\*Skipped|^ *[0-9]+/[0-9]+ Test +#[0-9]+: .* Not Run ' "$clog")
    nfail=$(grep -cE '^ *[0-9]+/[0-9]+ Test +#[0-9]+: .*(\*\*\*Failed|\*\*\*Timeout|\*\*\*Exception|Subprocess aborted|SegFault|Child aborted)' "$clog")
    ntot=$(grep -oE 'out of [0-9]+' "$clog" | tail -1 | grep -oE '[0-9]+' || true)
    if [ $crc -ne 0 ]; then
      bad=$(grep -E '^\s+[0-9]+ - ' "$clog" | grep -vE '\((Skipped|Disabled)\)' | sed -E 's/^\s+[0-9]+ - //; s/ \(.*//' | tr '\n' ' ')
      summary+=("$row: ctest -L cpu FAILED - ${npass} passed, ${nfail} failed, ${nskip} skipped of ${ntot:-?}: ${bad}($clog)")
      fail=1; continue
    fi
    summary+=("$row: ok ($(( $(date +%s) - start )) s; ctest -L cpu: ${npass} passed, ${nfail} failed, ${nskip} skipped of ${ntot:-?})")
    continue
  fi
  summary+=("$row: ok ($(( $(date +%s) - start )) s)")
done
printf '%s\n' "${summary[@]}"
exit $fail
