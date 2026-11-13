# Core Engine Optimization Analysis: pancake-strata

## Files Analyzed (Total: ~1.4 MB)

| File | Size | Key Patterns |
|------|------|--------------|
| `expert_source.cpp` | 390 KB | 7 cudaMemcpy, 3 cudaStreamSynchronize, 94 printf |
| `verify.cpp` | 378 KB | 34 cudaMemcpy, 23 cudaStreamSynchronize, 18 cudaMalloc |
| `layer.cpp` | 123 KB | 13 cudaMemcpy, 3 cudaDeviceSynchronize |
| `mtp.cpp` | 118 KB | 26 cudaMemcpy, 20 cudaMalloc, 7 cudaStreamSynchronize |
| `pinned.cu` | 74 KB | 3 cudaMemcpy, 5 cudaMalloc, 2 cudaStreamSynchronize |
| `dflash2.cpp` | 61 KB | 21 cudaMemcpy, 12 cudaMalloc, 11 cudaStreamSynchronize |

## Analysis

### 1. `expert_source.cpp` (390 KB) — Expert Loading/Caching
**Patterns**: 7 cudaMemcpy, 3 cudaStreamSynchronize, 94 printf, 65 push_back

**Key findings**:
- Lines 5653-5654: `cudaMemcpyAsync` for expert slot + dst (pinned → device) — **good, async**
- Line 6910: `cudaMalloc` for card memory (init, acceptable)
- Lines 6977, 7005: `cudaMemcpyAsync` for expert loading, then sync `cudaMemcpy` for verification — **verification sync could be async**
- 94 `printf` calls — many likely in hot cache paths

**Optimization**:
- Line 7005: Convert verification `cudaMemcpy` to async + event
- Profile `printf` calls — many are likely in the hot expert selection path

### 2. `verify.cpp` (378 KB) — Speculative Decoding / Verification
**Patterns**: 34 cudaMemcpy, 23 cudaStreamSynchronize, 18 cudaMalloc

**Key findings**:
- Lines 818-851: Mixed `cudaMemcpyAsync` / `cudaMemcpyPeerAsync` / `cudaMemcpyDeviceToDevice` — **good variety, mostly async**
- Line 820: `cudaMemcpyDeviceToHost` for logit bias — small, acceptable
- Lines 1162-1475: Many `cudaMalloc` in initialization (acceptable)
- **23 `cudaStreamSynchronize`** — many in hot verification loop

**Optimization**:
- Profile the 23 syncs — many are likely in the speculative verification hot path
- Replace with `cudaEventRecord` where possible
- The `cudaMemcpyPeerAsync` (line 851) is good for multi-GPU

### 3. `mtp.cpp` (118 KB) — MTP (Multi-Token Prediction)
**Patterns**: 26 cudaMemcpy, 20 cudaMalloc, 7 cudaStreamSynchronize

**Key findings**:
- Lines 249-250: `cudaMalloc` + sync `cudaMemcpy` for dense4_ (init, acceptable)
- Lines 275, 282: Sync `cudaMemcpy` for head logits — **hot path**
- Lines 342-351: `cudaMalloc` + sync `cudaMemcpy` for dense weights (init)
- Lines 418, 427: `cudaMalloc` for state arena (init)
- **Lines 594, 632**: Sync `cudaMemcpy` for logits/experts — **hot path candidates**

**Optimization**:
- Lines 275, 282, 594, 632: Convert sync `cudaMemcpy` to async + event
- `cudaMallocHost` (line 623) for pinned host buffer — good for async
- **Expected**: 5-15% MTP decode improvement

### 4. `dflash2.cpp` (61 KB) — Draft Model (Speculative Decoding)
**Patterns**: 21 cudaMemcpy, 12 cudaMalloc, 11 cudaStreamSynchronize, 7 cudaDeviceSynchronize

**Key findings**:
- Line 45, 99: `cudaStreamSynchronize` in init/cleanup
- Lines 194, 200: **`cudaDeviceSynchronize` in weight conversion hot path** — high impact
- Line 284: `cudaDeviceSynchronize` in `load()` — init, acceptable

**Optimization**:
- Lines 194, 200: Replace `cudaDeviceSynchronize` with `cudaStreamSynchronize` + event
- **Expected**: 5-10% improvement in draft model throughput

### 5. `pinned.cu` (74 KB) — Pinned Memory Management
**Patterns**: 3 cudaMemcpy, 5 cudaMalloc, 2 cudaStreamSynchronize, 16 reserve

**Key findings**:
- 16 `reserve` calls — good pre-allocation
- Lines for `cudaMalloc`/`cudaMemcpy` are init, not hot path

### 6. `layer.cpp` (123 KB) — Layer Definitions
**Patterns**: 13 cudaMemcpy, 3 cudaDeviceSynchronize

**Key findings**:
- Line 290: `cudaMemcpyAsync` D2D — good
- Line 1252: `cudaMemcpy2DAsync` for attention — **good, 2D async**
- Lines 957-958, 968: `cudaMemcpy` for cos/sin/page tables — init, acceptable

## Summary: Top Optimizations

| Priority | File | Issue | Expected Impact |
|----------|------|-------|-----------------|
| 1 | `mtp.cpp` | 4 sync `cudaMemcpy` in hot path | 5-15% MTP decode |
| 2 | `dflash2.cpp` | 2 `cudaDeviceSynchronize` in hot path | 5-10% draft model |
| 3 | `verify.cpp` | 23 `cudaStreamSynchronize` in hot path | 10-20% verify |
| 4 | `expert_source.cpp` | 1 sync `cudaMemcpy` verification | Minor |
| 5 | `layer.cpp` | Good — mostly async already | N/A |

## Verification

```bash
# Build
cmake -S . -B build-hip -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build build-hip --target strata -j2

# Profile
rocprof --kernel --output /tmp/core.json ./build-hip/strata --model <path>

# A/B test
python tools/pancake/abtest.py bench/specs/34.json --out bench/results/core-optimizations.jsonl
```