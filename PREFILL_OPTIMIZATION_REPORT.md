# Prefill Path Optimization Analysis: pancake-strata

## File: `src/prefill/prefill.cpp` (550 KB, ~8000 lines)

## Current State

| Pattern | Count | Impact |
|---------|-------|--------|
| `cudaStreamSynchronize` | 53 | High - 53 sync points in prefill path |
| `cudaMemcpy` | 86 | Medium - H2D/D2H transfers |
| `cudaMalloc` | 30 | Low - mostly init allocations |
| `printf` | 71 | Low - debug/trace gated |
| `push_back` | 44 | Low - vector growth |
| `reserve` | 5 | Low - few reserve() calls |
| `static std` | 9 | Low - global state |

## Analysis: What's Causing the Problem

### 1. 53 `cudaStreamSynchronize` in Prefill (High Impact)
The prefill path has more stream syncs than the decode path (generate.cpp has 27). Each sync stalls the CPU until the GPU stream completes. In prefill, this is expected for data dependencies, but 53 is high for a hot path.

**Key sync locations:**
- Lines 996-1012: 5 syncs in cleanup/shutdown (acceptable)
- Lines 1394-1397: 4 syncs in destructor (acceptable)
- Line 1783: 1 sync in hot path (check context)
- Line 2207: 1 sync in hot path (check context)
- **~46 others**: Need to profile which are in the hot prefill loop vs. init/teardown

### 2. 86 `cudaMemcpy` (Medium Impact)
- Line 1053-1056: Pinned buffer H2D upload (acceptable)
- Line 1984: `cudaMemcpy` to ident table (init, acceptable)
- Lines 2393, 3233, 3298: D2D and H2D in hot prefill loop
- **Action**: Convert sync `cudaMemcpy` to `cudaMemcpyAsync` + stream sync where possible

### 3. FK-28 ADAPT_PROMPT Hook Missing
**Critical finding**: `drive.d.usage` is populated in `generate.cpp` (line 13692) but NOT in `prefill.cpp`. The FK-28 `STRATA_ADAPT_PROMPT` feature requires expert usage tracking during the prompt phase.

**Evidence:**
- `generate.cpp:13692`: `drive.d.usage.assign((size_t) (g.n_layers * g.n_expert), 0.0f);`
- `prefill.cpp` has **0** references to `usage`, `adapt`, or `drive.d.`
- The `m.cnt` counter (line 1926, 5157) tracks per-expert counts but does NOT feed into `drive.d.usage`

**Required change** (FK-28 port):
```cpp
// In prefill.cpp, after m.cnt is populated (line ~5157):
if (!drive.d.usage.empty()) {
    for (size_t e = 0; e < m.cnt.size(); ++e)
        drive.d.usage[e] += (float) m.cnt[e] * o.adapt_scale;
    for (float& v : drive.d.usage) v *= o.adapt_decay;
}
```

## Proposed Changes

### Priority 1: Add FK-28 ADAPT_PROMPT Hook (High Effort, High Impact)
**File**: `src/prefill/prefill.cpp`
- After the MMQ expert routing at line ~5157, feed `m.cnt` into `drive.d.usage`
- Enable adaptive expert selection during prefill, not just decode
- **Expected**: Better expert residency decisions on the first tokens (improves decode start time)
- **Risk**: Need to verify the `adapt_scale` and `adapt_decay` parameters match the generate path

### Priority 2: Convert Sync cudaMemcpy to Async (Low Effort, Medium Impact)
**File**: `src/prefill/prefill.cpp`
- Lines 2393, 3233, 3298: `cudaMemcpy` → `cudaMemcpyAsync` with stream
- **Expected**: 5-15% improvement in prefill throughput by overlapping H2D with compute
- **Risk**: Low if streams are used correctly

### Priority 3: Reduce Stream Synchronization (Medium Effort, Medium Impact)
**File**: `src/prefill/prefill.cpp`
- Profile with `rocprof` to identify which of the 53 `cudaStreamSynchronize` are in the hot prefill loop
- Replace unnecessary syncs with `cudaEventRecord`/`cudaEventSynchronize`
- **Expected**: 10-30% prefill latency reduction
- **Risk**: May break data dependencies. Must verify with parity tests.

## Verification Plan

```bash
# Build on Linux ROCm box
cmake -S . -B build-hip -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build build-hip --target strata strata_tests -j2

# Parity tests (CPU)
ctest --test-dir build-hip -L cpu --output-on-failure

# GPU tests
ctest --test-dir build-hip -L gpu --output-on-failure

# Measure FK-28 improvement
export STRATA_ADAPT_PROMPT=1
python tools/pancake/abtest.py bench/specs/32.json --out bench/results/adapt-prompt.jsonl
```

## Notes
- The 53 `cudaStreamSynchronize` calls include legitimate syncs (init, teardown, data dependencies). A profiler would identify which are truly on the hot prefill loop.
- FK-28 port is the highest-priority item from MOD_PLAN. It requires a Linux build + parity tests.
- The `m.cnt` counter already exists in prefill.cpp — the FK-28 hook is a small addition (~10 lines) that feeds it into `drive.d.usage`.
