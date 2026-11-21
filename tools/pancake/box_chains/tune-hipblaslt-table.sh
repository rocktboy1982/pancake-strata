#!/bin/sh
# tune-hipblaslt-table.sh - regenerate the gfx1100 hipBLASLt prompt-GEMM table for ANOTHER hipBLASLt version on the RX 7900 XT (LEADS.md Part T, lead T2).
# NEVER RUN BY THE AUTHOR: written on the Windows PC (no ROCm, no GPU), recorded in TEST_LEDGER.md as "coded, not run".  Run it by hand ON THE BOX, only when
# the box is idle: it refuses to start while any benchmark, build, test or engine process is running, and it never touches pancake.service or a queue.
# The file name has no busy() pattern string on purpose (no "bench-", "win-chain", "abtest"); while it tunes, the GPU is busy: nothing else may start.
#
# Why: the engine looks for tuning/<arch>-hipblaslt-<hipblasLtGetVersion()>.txt (src/prefill/gemm.cu create_hipblaslt_state).  A table's solution ids are valid
# for one hipBLASLt build only (the file header says so).  ROCm 10.0 = hipBLASLt 1.4.x = key 100401 (tools/hip/gfx1100-hipblaslt-100401.txt, box-tuned, T-096:
# 1,345 -> 1,850-1,920 tok/s at 11.7K).  ROCm 10.2.0 = hipBLASLt 1.5.0 = key 100500.  tools/hip/gfx1100-hipblaslt-100500.txt in the repo was tuned on an
# RX 7900 XTX with the 10.2.0a NIGHTLY (commit d74e110f0, 34 rows); 10.2.0 retuned gfx1100 with Origami, so its ids may name other kernels than the GA build.
#
# What it does (each step logs; any failure stops with the step's name):
#   0. idle check (refuses when busy), ROCm tree and hipBLASLt version check (the version must give the key you ask for)
#   1. configure + build ONLY the tuner target tune_hipblaslt against that ROCm tree, in its own build directory (no engine build, no change to any existing build)
#   2. tune: tools/hip/tune_hipblaslt --shapes-file <the 100401 table: its (dtype, N, K, ldy) shapes> --tokens 4096,8192 --tuning-out <out>/tuned.txt
#        (the tuner times every hipBLASLt solution per shape on the card, checks accuracy against a reference, and writes the fastest; header = arch + version of the
#         library it ran against, so the file is stamped with the right key by itself)
#   3. check the header line says "STRATA_HIPBLASLT_TUNING_V1 gfx1100 <version>" and that the row count is close to the 100401 table's
#   4. optional (EXTEND=1): tools/pancake/hipblaslt_extend.py plan, which prints the tuner command for the T buckets the prompt chunks need beyond 4096 / 8192
#   5. print the files and the commands for the owner's next steps; NOTHING is copied into the repository by this script
#
# Usage on the box:
#   cd <pancake checkout of lead/T2-T4 or port/rdna3-decode>          # the tree that has tools/hip/tune_hipblaslt.cpp and CMakeLists.txt
#   ROCM=/opt/rocm/core-10.2 WANT_KEY=100500 sh tools/pancake/box_chains/tune-hipblaslt-table.sh
# Variables: ROCM (default /opt/rocm/core-10.2), WANT_KEY (default 100500), SRC (default: this checkout), OUT (default $HOME/auto-runs/tune-hipblaslt-$WANT_KEY),
#   SHAPES (default tools/hip/gfx1100-hipblaslt-100401.txt), TOKENS (default 4096,8192), GPU (default 0: HIP_VISIBLE_DEVICES), EXTEND (default 0).
# Afterwards, to use the table (as files, still the owner's call): copy $OUT/gfx1100-hipblaslt-$WANT_KEY.txt to tools/hip/ (it replaces the nightly-tuned 100500 file:
# keep that one's git history), then prompt A/B it with spec 255's `table` arm pointed at it (bench/specs/255-hipblaslt-heuristic-arms.json), 10K / 32K / 64K.
# Add STRATA_HIPBLASLT_TABLE_STRICT=1 to the engine environment so an engine that cannot load the table refuses to start instead of falling back.
set -u
ROCM=${ROCM:-/opt/rocm/core-10.2}
WANT_KEY=${WANT_KEY:-100500}
SRC=${SRC:-$(cd "$(dirname "$0")/../../.." && pwd)}
OUT=${OUT:-$HOME/auto-runs/tune-hipblaslt-$WANT_KEY}
SHAPES=${SHAPES:-$SRC/tools/hip/gfx1100-hipblaslt-100401.txt}
TOKENS=${TOKENS:-4096,8192}
GPU=${GPU:-0}
EXTEND=${EXTEND:-0}
CM=${CM:-$(command -v cmake || echo /home/linuxbrew/.linuxbrew/bin/cmake)}
NINJA=${NINJA:-$(command -v ninja || echo "$SRC/Strata/.venv/bin/ninja")}
BUILD=$OUT/build-tuner
LOG=$OUT/tune.log

say() { echo "[tune-hipblaslt-table] $*" | tee -a "$LOG"; }
die() { say "ERROR step $1: $2"; exit 1; }

mkdir -p "$OUT" || { echo "cannot create $OUT"; exit 1; }
: > "$LOG"

# 0. idle check: the same process patterns the benchmark chains use (nothing of ours matches them: see the file name note above)
if pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[r]un-queue-us|[b]ench-sched|[b]ench-sweep" > /dev/null 2>&1; then
  die 0 "the box is busy (a benchmark, build, test or engine process runs); nothing was started"
fi
for f in "$ROCM/bin/hipcc" "$ROCM/include/hipblaslt/hipblaslt.h" "$ROCM/include/hipblaslt/hipblaslt-version.h" "$CM" "$NINJA" "$SRC/tools/hip/tune_hipblaslt.cpp" "$SHAPES"; do
  [ -e "$f" ] || die 0 "missing: $f"
done
maj=$(sed -n 's/^#define HIPBLASLT_VERSION_MAJOR \([0-9]*\).*/\1/p' "$ROCM/include/hipblaslt/hipblaslt-version.h")
min=$(sed -n 's/^#define HIPBLASLT_VERSION_MINOR \([0-9]*\).*/\1/p' "$ROCM/include/hipblaslt/hipblaslt-version.h")
pat=$(sed -n 's/^#define HIPBLASLT_VERSION_PATCH \([0-9]*\).*/\1/p' "$ROCM/include/hipblaslt/hipblaslt-version.h")
[ -n "$maj" ] && [ -n "$min" ] && [ -n "$pat" ] || die 0 "cannot read the hipBLASLt version from $ROCM/include/hipblaslt/hipblaslt-version.h"
KEY=$((maj * 100000 + min * 100 + pat))
say "ROCm tree $ROCM: hipBLASLt $maj.$min.$pat = key $KEY (asked for $WANT_KEY)"
[ "$KEY" = "$WANT_KEY" ] || die 0 "the tree's key is $KEY, not $WANT_KEY: wrong ROCM, or set WANT_KEY=$KEY"
if [ -e "$SRC/tools/hip/gfx1100-hipblaslt-$KEY.txt" ]; then
  say "note: tools/hip/gfx1100-hipblaslt-$KEY.txt exists in the repo already (nightly-tuned or older); this run writes its own file in $OUT and leaves the repo alone"
fi

# 1. build the tuner only
say "1. configure + build tune_hipblaslt in $BUILD"
HIP_PLATFORM=amd ROCM_PATH=$ROCM HIP_PATH=$ROCM "$CM" -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_PREFIX_PATH="$ROCM" -DCMAKE_MAKE_PROGRAM="$NINJA" >> "$LOG" 2>&1 || die 1 "cmake configure (see $LOG)"
nice -n 10 "$CM" --build "$BUILD" -j4 --target tune_hipblaslt >> "$LOG" 2>&1 || die 1 "build of tune_hipblaslt (see $LOG)"
TUNER=$BUILD/tune_hipblaslt
[ -x "$TUNER" ] || TUNER=$(find "$BUILD" -name tune_hipblaslt -type f | head -1)
[ -x "$TUNER" ] || die 1 "tune_hipblaslt binary not found under $BUILD"

# 2. tune on the card (the long step: every solution of every shape is timed; budget about 20-40 minutes for 45 rows at two buckets, not measured)
say "2. tuning on GPU $GPU: shapes from $SHAPES, tokens $TOKENS"
HIP_VISIBLE_DEVICES=$GPU LD_LIBRARY_PATH=$ROCM/lib "$TUNER" --shapes-file "$SHAPES" --tokens "$TOKENS" --tuning-out "$OUT/tuned.txt" \
  > "$OUT/tuner.stdout" 2>&1 || die 2 "the tuner failed (see $OUT/tuner.stdout)"

# 3. check the stamp and the size
HEAD=$(grep -v '^#' "$OUT/tuned.txt" | head -1)
say "3. header: $HEAD"
[ "$HEAD" = "STRATA_HIPBLASLT_TUNING_V1 gfx1100 $KEY" ] || die 3 "the header is not 'STRATA_HIPBLASLT_TUNING_V1 gfx1100 $KEY'"
rows_new=$(grep -v '^#' "$OUT/tuned.txt" | tail -n +2 | wc -l)
rows_old=$(grep -v '^#' "$SHAPES" | tail -n +2 | wc -l)
say "rows: $rows_new tuned, $rows_old in the shapes file ($SHAPES); a tuned count far below the shapes file means shapes the tuner skipped (read $OUT/tuner.stdout)"
cp "$OUT/tuned.txt" "$OUT/gfx1100-hipblaslt-$KEY.txt" || die 3 "copy"

# 4. optional extra T buckets
if [ "$EXTEND" = "1" ]; then
  say "4. planned extra buckets (hipblaslt_extend.py plan; the printed tuner command is NOT run here):"
  python3 "$SRC/tools/pancake/hipblaslt_extend.py" plan --table "$OUT/gfx1100-hipblaslt-$KEY.txt" --script "$OUT/extend-tuner.sh" >> "$LOG" 2>&1 \
    || say "hipblaslt_extend.py plan failed (see $LOG); the base table above is unaffected"
  say "extend script written to $OUT/extend-tuner.sh for the owner to read and run"
fi

# 5. next steps
say "5. done. Table: $OUT/gfx1100-hipblaslt-$KEY.txt (header and rows checked, not yet A/B-ed). Next, by the owner:"
say "   a. engine smoke with the table and STRICT: STRATA_HIPBLASLT_TUNING=$OUT/gfx1100-hipblaslt-$KEY.txt STRATA_HIPBLASLT_TABLE_STRICT=1 (the log must say 'hipBLASLt table status=exact key=gfx1100-hipblaslt-$KEY')"
say "   b. spec 255 arm table pointed at it, prompt cases 10K / 32K / 64K; the quality gate before adopting (a retuned table can change the last bits)"
say "   c. only then: copy it to tools/hip/ in the repo and the engine package (packaging/stage_tuning.py picks tools/hip/<arch>-hipblaslt-<version>.txt)"
echo "TUNE_HIPBLASLT_TABLE_DONE" >> "$LOG"
