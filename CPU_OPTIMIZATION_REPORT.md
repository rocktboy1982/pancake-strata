# CPU Kernel Optimization Analysis: pancake-strata

## Files: `src/kernels/cpu/` (~1.5 MB total)

| File | Size | Key SIMD |
|------|------|----------|
| `pool.cpp` | 88 KB | Thread pool, NUMA affinity, WAITPKG |
| `iq_avx512.cpp` | 51 KB | 92 m512, 67 m256, 21 gather, 24 prefetch |
| `native_expert.cpp` | 43 KB | 83 avx, 41 avx2, 20 avx512, 5 gather |
| `iq_avx2_rows.inl` | 42 KB | 169 m256, 153 _mm256, 7 prefetch |
| `iq_avx2.cpp` | 38 KB | 97 m256, 26 gather, 5 prefetch |
| `expert.cpp` | 37 KB | 73 m512, 52 m256, 14 fma |

## Analysis

### 1. `pool.cpp` — Thread Pool + NUMA Affinity (88 KB)
**Key features**:
- `pthread_setaffinity_np` (Linux) / `SetThreadGroupAffinity` (Windows) for CPU pinning
- NUMA node split (`numa_pool.hpp`): per-NUMA pool when `STRATA_NUMA_SPLIT=auto`
- WAITPKG intrinsic: `__builtin_ia32_umwait` / `__builtin_ia32_uwakeup` for low-power wait
- `STRATA_POOL_UMWAIT=0` disables WAITPKG (fallback to spin)
- SMT sibling detection: GPU-only stretches spin on SMT siblings (line 78 comment)

**Optimization opportunities**:
- **CPU Set IDs** (Windows, line 40-61 in pool_affinity_win.hpp): `GetSystemCpuSetInformation` with `GetThreadSelectedCpuSets` — the CPU Set API is already used for hard affinity. Consider using `SetThreadSelectedCpuSets` for dynamic migration during prefill.
- **WAITPKG fallback**: The `__builtin_ia32_umwait` requires a CPU with WAITPKG. On older CPUs, it falls back to spin. Add a runtime check and a configurable wait timeout.
- **NUMA auto-detection**: `numa_pool.hpp` uses `pick_numa_pool()` which auto-splits on 2+ NUMA nodes. Add `STRATA_NUMA_POOL=off` override for single-node machines (already exists at line 30).

**Expected impact**: 5-15% on multi-NUMA systems (e.g., dual-socket Threadripper, EPYC)

### 2. `iq_avx512.cpp` — IQ Quantization with AVX-512 (51 KB)
**Key SIMD**:
- 92 `m512` (AVX-512 512-bit vector) operations
- 21 `gather` (`_mm512_i64gather_epi64` / `_mm512_i64gather_epi32`)
- 24 `prefetch` (`_mm_prefetch`)
- 11 `fma` (FMA)

**Optimization opportunities**:
- **AVX-512 downclocking**: On Intel CPUs, AVX-512 can cause significant downclocking (especially Skylake-X/Cascade Lake). Add a runtime check: if AVX-512 is enabled but the CPU is downclocking, fall back to AVX-2.
- **Gather cost**: `gather` operations are expensive on older AVX-512 (Skylake-X: 1/4 throughput). Consider replacing with `broadcast + shuffle` where the access pattern allows it.
- **Prefetch distance**: 24 prefetch calls — verify the prefetch distance (cache line ahead) is optimal. Too early wastes L1, too late misses.
- **Loop unrolling**: Check if the main quantization loop is fully unrolled. AVX-512 benefits from 4x unrolling.

**Expected impact**: 10-30% on AVX-512 CPUs (Zen 4/5, ICL+), 5-15% on AVX-2 (Zen 1-3, SKX)

### 3. `native_expert.cpp` — Native Expert Matmul (43 KB)
**Key SIMD**:
- 83 `avx` references, 41 `avx2`, 20 `avx512`
- 5 `gather` operations

**Optimization opportunities**:
- **Mixed SIMD dispatch**: The file has both AVX-2 and AVX-512 paths. Check if the runtime dispatch is correct.
- **Register pressure**: 20 `m512` operations can saturate the register file. Check if the compiler is spilling.
- **Prefetch**: Only 5 gather — add prefetch for the expert weight reads.

**Expected impact**: 5-15%

### 4. `iq_avx2.cpp` — IQ Quantization AVX-2 (38 KB)
**Key SIMD**:
- 97 `m256` operations
- 26 `gather`
- 5 `prefetch`

**Optimization opportunities**:
- **Gather cost**: 26 gather on AVX-2 is expensive (Skylake: 25% throughput). Replace with `shuffle` where the access pattern is regular.
- **Prefetch**: Only 5 prefetch — the expert weight reads need prefetch to hide DRAM latency.
- **FMA**: Only 3 `fma` — add FMA for the scale operations.

**Expected impact**: 10-20% on AVX-2 CPUs

### 5. `expert.cpp` — Expert Matmul with AVX-512 (37 KB)
**Key SIMD**:
- 73 `m512`, 52 `m256`, 14 `fma`

**Optimization opportunities**:
- **FMA count**: Only 14 `fma` — the matmul should be FMA-heavy. Check if the compiler is converting `mul + add` to `fma` at -O3.
- **Loop unrolling**: AVX-512 matmuls benefit from 4x or 8x unrolling.

**Expected impact**: 5-10%

## Summary: Top Optimizations

| Priority | File | Issue | Expected Impact |
|----------|------|-------|-----------------|
| 1 | `iq_avx512.cpp` | AVX-512 downclocking fallback | 10-30% on Zen 4/5 |
| 2 | `iq_avx2.cpp` | 26 gather → shuffle, add prefetch | 10-20% |
| 3 | `native_expert.cpp` | Add prefetch, check register spill | 5-15% |
| 4 | `expert.cpp` | Add FMA, unroll loops | 5-10% |
| 5 | `pool.cpp` | WAITPKG runtime check, CPU Set migration | 5-15% |

## Verification

```bash
# Build CPU tests
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --target strata_tests -j8

# Run CPU parity tests
ctest --test-dir build-cpu -L cpu --output-on-failure

# Benchmark individual kernels
./build-cpu/strata_tests --gtest_filter='*iq_avx*:*expert*'
```

## Notes
- The CPU kernels are the fallback when experts don't fit in GPU VRAM (35B on 7900 XT).
- The 7900 XT has 20 GB VRAM — most of 35B-A3B at Q4_0 will fit. The CPU path handles the overflow.
- `STRATA_NO_GPU_EXPERTS=1` forces all experts to CPU — use this for benchmarking the CPU kernels in isolation.
