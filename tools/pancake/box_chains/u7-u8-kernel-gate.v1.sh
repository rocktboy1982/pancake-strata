#!/bin/sh
# LEADS.md Part U leads U7 (STRATA_VERIFY_GEMV_CHAINS) and U8 (STRATA_ATTN_KV_PREFETCH), box kernel gate v1.  NEW file; never edits or kills anything,
# never deletes a file.  IDLE ONLY: refuses (exit 3) while a busy-pattern process runs, does not wait.  No engine start: only the new test binaries and a
# kernel timing, as LEADS.md U.11 asks ("kernel microbench and parity binaries only; engine screens not justified").
#   sh tools/pancake/box_chains/u7-u8-kernel-gate.v1.sh [SRC_TREE] [BUILD_DIR_FOR_U7] [BUILD_DIR_FOR_U8]
#   SRC_TREE default /home/dan/clone-of-d/src/Strata-win (synced with tar xmf from branch lead/U4-U7-U8 or port/rdna3-decode after the merge)
#   BUILD_DIR_FOR_U7 default /home/dan/clone-of-d/src/build-win       (the normal build: the chain kernels are always compiled, switch off by default)
#   BUILD_DIR_FOR_U8 default /mnt/pancake/build-kvpf                  (NEW directory: configured with -DSTRATA_ATTN_KV_PREFETCH=ON; created here if absent)
# CMAKE_ARGS (env) = the cmake line of build-win (see $BUILD/CMakeCache.txt: STRATA_ENABLE_HIP, CMAKE_HIP_ARCHITECTURES=gfx1100, STRATA_BUILD_TESTS=ON,
# the compiler and generator); the default below is a guess for a first configure, check it against the cache before the U8 build.
# Steps:  A  U7 build of rdna_chains_parity + rdna_mmvq_parity + b6_mmvq_bench (ninja in the existing build dir: the only ninja this script starts)
#         B  U7 ctest -R "rdna_chain|rdna_mmvq_parity" (bitwise chains; the existing 384-case tolerance sweep with the switch OFF = the default-path control)
#         C  U7 kernel timing: b6_mmvq_bench per shape with STRATA_VERIFY_GEMV_CHAINS unset / 2 / 4 (A B B A order, output files only)
#         D  U8 configure + build qsa_kv_prefetch_parity in $B8, ctest it (bitwise, 288 cases), and the default-build control (qsa_merge_daily_parity)
#         E  U8 kernel timing: qsa_kv_prefetch_parity is a correctness test; the timing is rocprof on hip_decode_attn_wmma_parity --bench-style runs or the
#            engine log; this script only prints the exact rocprof line to run by hand (no engine start here).
# Output: /home/dan/auto-runs/u7u8-gate-<timestamp>/ (chain.log, ctest-*.txt, b6-*.txt).  Record the numbers in TEST_LEDGER.md rows T-1060.. only.
SRC=${1:-/home/dan/clone-of-d/src/Strata-win}
B7=${2:-/home/dan/clone-of-d/src/build-win}
B8=${3:-/mnt/pancake/build-kvpf}
TS=$(date +%Y%m%d_%H%M%S)
OUT=/home/dan/auto-runs/u7u8-gate-$TS
CM=${CM:-/home/linuxbrew/.linuxbrew/bin/cmake}; CT=${CT:-/home/linuxbrew/.linuxbrew/bin/ctest}
export LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[b]ench-sweep|[b]ench-sched|[r]un-queue-us|[w]in-chain[A-Za-z0-9]*\.sh|[u]4-valu-rate|[r]dna_chains_parity|[q]sa_kv_prefetch_parity" >/dev/null; }
if busy; then
  echo "REFUSED $(date +%F_%T): a busy-pattern process is running; the gate needs an idle box. Nothing was done." >&2
  exit 3
fi
mkdir -p "$OUT" || exit 2
L=$OUT/chain.log
say() { echo "$(date +%T) $*" >> "$L"; echo "$*"; }
freekb() { df -Pk "$1" | awk 'NR==2{print $4}'; }
say "START u7-u8-kernel-gate.v1 src=$SRC b7=$B7 b8=$B8 out=$OUT"
[ -f "$SRC/include/strata/kernels/rdna_chain_split.hpp" ] || { say "FATAL the tree $SRC has no U7 code: sync lead/U4-U7-U8 first (tar xmf, mtimes reset)"; exit 2; }
[ "$(freekb /)" -gt 2097152 ] || { say "SKIPPED: / has less than 2 GB free ($(freekb /) KB); nothing deleted"; exit 2; }
# ---- U7
say "A: build U7 targets in $B7"
nice -n 10 "$CM" --build "$B7" -j8 --target rdna_chains_parity rdna_mmvq_parity rdna_chain_split_test >> "$L" 2>&1; say "A exit=$?"
nice -n 10 "$CM" --build "$B7" -j8 --target b6_mmvq_bench >> "$L" 2>&1; say "A(b6_mmvq_bench) exit=$?"
say "B: ctest"
( cd "$B7" && "$CT" -R "rdna_chain|rdna_mmvq_parity" --output-on-failure ) > "$OUT/ctest-u7.txt" 2>&1; say "B exit=$? (ctest-u7.txt)"
B6="$B7/b6_mmvq_bench"; [ -x "$B6" ] || B6=$(find "$B7" -name b6_mmvq_bench -type f 2>/dev/null | head -1)
if [ -x "$B6" ] && ! busy; then
  say "C: b6_mmvq_bench A B B A (read src/kernels/b6_mmvq_bench.cpp for its arguments; the switch is the environment)"
  for arm in unset 2 4 4 2 unset; do
    if [ "$arm" = unset ]; then env -u STRATA_VERIFY_GEMV_CHAINS STRATA_RDNA_Q6K=1 "$B6" > "$OUT/b6-$arm.$(date +%s).txt" 2>&1
    else STRATA_VERIFY_GEMV_CHAINS=$arm STRATA_RDNA_Q6K=1 "$B6" > "$OUT/b6-chains$arm.$(date +%s).txt" 2>&1; fi
    say "C arm=$arm exit=$?"
  done
else say "C skipped (no b6_mmvq_bench binary or the box became busy)"; fi
# ---- U8
if [ ! -f "$B8/CMakeCache.txt" ]; then
  say "D: configure $B8 (CMAKE_ARGS=${CMAKE_ARGS:-<default guess>})"
  mkdir -p "$B8" && "$CM" -S "$SRC" -B "$B8" ${CMAKE_ARGS:--G Ninja -DSTRATA_ENABLE_HIP=ON -DSTRATA_BUILD_TESTS=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release} -DSTRATA_ATTN_KV_PREFETCH=ON >> "$L" 2>&1; say "D configure exit=$?"
else "$CM" -S "$SRC" -B "$B8" -DSTRATA_ATTN_KV_PREFETCH=ON >> "$L" 2>&1; fi
say "D: build qsa_kv_prefetch_parity qsa_merge_daily_parity in $B8"
nice -n 10 "$CM" --build "$B8" -j8 --target qsa_kv_prefetch_parity qsa_merge_daily_parity >> "$L" 2>&1; say "D build exit=$?"
( cd "$B8" && "$CT" -R "qsa_kv_prefetch_parity|qsa_merge_daily_parity" --output-on-failure ) > "$OUT/ctest-u8.txt" 2>&1; say "D ctest exit=$? (ctest-u8.txt)"
say "E: timing by hand (no engine here): STRATA_ATTN_KV_PREFETCH=0 and =1 on the same $B8 binary, rocprofv3 --kernel-trace --stats on hip_decode_attn_wmma_parity --bench, compare attn_chunk_kernel against attn_chunk_pf_kernel"
say "END u7-u8-kernel-gate.v1 (U7U8_GATE_DONE)"
