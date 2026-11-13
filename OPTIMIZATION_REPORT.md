# Optimization Report: pancake-strata (Strata LLM Inference Engine)

## Current State: Baseline Metrics

### Codebase Overview
- **Total source**: ~55 MB C++/CUDA/HIP across 450+ files
- **Hot path files** (by size):
  - `src/program/generate.cpp`: 1.6 MB (main inference loop)
  - `src/core/expert_source.cpp`: 390 KB
  - `src/core/verify.cpp`: 378 KB
  - `src/kernels/iq_kernels.cu`: 426 KB (largest kernel)
  - `src/kernels/native_mmvq.cu`: 222 KB
  - `src/kernels/qsa_select.cu`: 216 KB

### Performance Anti-Patterns Found

| File | Pattern | Count | Impact |
|------|---------|-------|--------|
| `generate.cpp` | `cudaDeviceSynchronize()` | 39 | High - full device sync in hot path |
| `generate.cpp` | `cudaStreamSynchronize()` | 27 | Medium - stream-level sync |
| `generate.cpp` | `cudaMemcpy` / `hipMemcpy` | 74 | Medium - H2D/D2H transfers |
| `generate.cpp` | `cudaMalloc` / `hipMalloc` | 52 | Low-Med - init allocations |
| `generate.cpp` | `printf` | 1188 | High - I/O in hot path |
| `generate.cpp` | `push_back` (no reserve) | ~150 | Low - reallocations |
| `generate.cpp` | `static std::vector` in hot path | 2 | Medium - thread safety |

## Analysis: What's Causing the Problem

### 1. Excessive Synchronization (High Impact)
- **39 `cudaDeviceSynchronize()` calls** in the inference hot path. Each one stalls the CPU until all GPU work completes. In a decode loop with 1000+ tokens, this multiplies overhead.
- **27 `cudaStreamSynchronize()` calls** add more latency, especially in the expert swap path.
- Comment at line 12454-12455 notes: *"cudaMemcpyAsync + cudaLaunchHostFunc inside a verify window while the GPU spins on the flag — issue #31's thread dumps show the host stuck in that cudaMemcpyAsync on a driver lock"*

### 2. printf I/O in Hot Path (High Impact)
- **1188 `printf` calls** in `generate.cpp` alone. Many are likely debug/trace, but even if gated by `if (verbose)`, the string formatting + I/O syscalls can stall.
- For a 1000-token decode, if any of these fire per token, that's 1000+ I/O syscalls.

### 3. Static Variables in Hot Paths (Medium Impact)
- Lines 245, 270: `static std::vector<float> row;` and `static std::vector<int32_t> ord;` — these persist across calls. Not a perf issue per se, but:
  - **Thread safety risk** if multiple sessions share the same code path
  - **Memory leak risk** — these vectors grow and are never freed

### 4. cudaMalloc in Hot Paths (Low-Medium Impact)
- Most `cudaMalloc` calls are in initialization (weight loading, arena setup), which is fine.
- The 7 calls in `route_resident_parity.cu` are test-only, not production.
- However, some calls at lines 5577, 6203 look like they may be in the expert swap hot path — needs verification.

### 5. cudaMemcpy (Medium Impact)
- 74 `cudaMemcpy` calls in `generate.cpp` — many are in the expert swap / cache eviction path.
- Some can be replaced with `cudaMemcpyAsync` + stream synchronization to overlap H2D/D2H transfers with compute.

## Proposed Changes

### Priority 1: Reduce Synchronization Overhead
**File**: `generate.cpp`
- Replace `cudaDeviceSynchronize()` with `cudaStreamSynchronize()` where possible (39 → ~15)
- Use `cudaEventRecord()` + `cudaEventSynchronize()` to avoid full device sync when only one stream matters
- Expected impact: 20-40% reduction in decode latency

### Priority 2: Gate printf Calls
**File**: `generate.cpp`
- Add a global `static bool s_verbose = false;` flag
- Wrap hot-path `printf` calls: `if (s_verbose) { printf(...); }`
- Move trace logging to a ring buffer that flushes on demand
- Expected impact: 10-20% reduction in decode latency (removes I/O stalls)

### Priority 3: Eliminate Static Vectors
**Files**: `generate.cpp` lines 245, 270
- Replace `static std::vector<float> row` with a `thread_local` or pass-by-reference version
- Expected impact: Thread safety improvement, slight perf gain from avoiding global symbol

### Priority 4: Async Copies
**Files**: `generate.cpp` (expert swap path)
- Replace synchronous `cudaMemcpy` with `cudaMemcpyAsync` + stream
- Expected impact: 5-15% improvement in expert swap throughput

### Priority 5: Kernel Optimization (Lower Impact, Higher Effort)
**Files**: `kernels/iq_kernels.cu`, `kernels/native_mmvq.cu`, `kernels/qsa_select.cu`
- Review for shared memory tiling opportunities
- Check tensor core usage in MMQ kernels
- Expected impact: 10-30% improvement in kernel throughput (requires profiling with Nsight/uProf)

## Expected Impact (Combined)

If all priorities 1-4 are applied:
- **Decode latency**: 40-70% reduction in wall-clock time
- **Prefill throughput**: 20-40% improvement
- **Memory efficiency**: Minor improvement (static vector cleanup)

## Verification

To verify improvements:
1. **Run abtest**: `python bench/abtest.py` on the Linux ROCm box before/after changes
2. **Profile with Nsight**: `nsys profile ./strata generate --model <path>` to confirm sync reduction
3. **Check token throughput**: Compare tokens/sec in abtest results

Note: **All changes require the Linux ROCm box for build/test** — Windows box has no GPU compiler.

## Notes
- This is a **static analysis** of the codebase. Actual impact depends on which code paths are hot.
- The 39 `cudaDeviceSynchronize` calls may include legitimate ones (error paths, initialization). A profiler would confirm which are truly on the hot path.
- The `printf` count is high but most are likely gated behind verbose/debug flags — needs confirmation.
