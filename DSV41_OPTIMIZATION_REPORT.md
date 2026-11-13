# DeepSeek v4.1 Optimization Analysis: pancake-strata

## Files: `src/dsv41/` (~1.7 MB total)

| File | Size | Key Patterns |
|------|------|--------------|
| `ds4_model.cu` | 873 KB | 222 cudaMemcpy, 56 stream sync, 43 mma, 190 printf |
| `ds4_main.cpp` | 160 KB | 144 printf, 7 mma |
| `ds4_kernels.cu` | 136 KB | 10 mma, 10 wmma |
| `ds4_iq.cu` | 90 KB | 4 mma, 1 wmma |
| `ds4_iq_cpu.cpp` | 49 KB | CPU IQ path |

## Current State

| Pattern | `ds4_model.cu` | `ds4_kernels.cu` | `ds4_main.cpp` |
|---------|----------------|------------------|----------------|
| `cudaMemcpy` | 222 | 4 | 0 |
| `cudaStreamSynchronize` | 56 | 2 | 0 |
| `cudaDeviceSynchronize` | 4 | 0 | 0 |
| `printf` | 190 | 5 | 144 |
| `mma` | 43 | 10 | 7 |
| `wmma` | 3 | 10 | 0 |

## Analysis

### 1. `ds4_model.cu` (873 KB) — Main Model Engine
**Critical findings**:
- **222 `cudaMemcpy`** — extremely high. Many are likely H2D/D2H for model loading + activation transfers.
- **56 `cudaStreamSynchronize`** — high sync count in inference path.
- **43 `mma`** — Tensor Core usage (good for FP16/BF16 matmuls).
- **190 `printf`** — likely debug/trace, many may fire in hot path.

**Optimization**:
- **Batch `cudaMemcpy`**: The 222 copies likely include many small transfers. Batch into larger async copies + `cudaMemcpyAsync` + events.
- **Reduce sync**: 56 syncs in hot path — profile with `nsys` to identify which are data dependencies vs. unnecessary.
- **Gate `printf`**: 190 calls — add `STRATA_DS4_VERBOSE` gate.

### 2. `ds4_kernels.cu` (136 KB) — Kernels
**Key features**:
- 10 `mma` + 10 `wmma` — mixed Tensor Core usage (FP16 mma + BF16 wmma?)
- Only 2 `cudaStreamSynchronize` — good, kernels are async

**Optimization**:
- Check if `mma` and `wmma` paths are both needed or if one can be unified.
- Verify `wmma` path uses `amdgcn_wmma_f16_16x16x16` for RDNA3 (7900 XT).

### 3. `ds4_main.cpp` (160 KB) — Entry Point
- 144 `printf` — likely CLI output, acceptable if gated.

### 4. `ds4_iq.cu` / `ds4_iq_cpu.cpp` — IQ Quantization
- GPU (`ds4_iq.cu`): 4 `mma`, 1 `wmma`, 1 `cudaMalloc`
- CPU (`ds4_iq_cpu.cpp`): 49 KB — check for AVX-512 path

## Optimization Priorities

| Priority | File | Issue | Expected Impact |
|----------|------|-------|-----------------|
| 1 | `ds4_model.cu` | 222 `cudaMemcpy` → batch async | 20-40% decode latency |
| 2 | `ds4_model.cu` | 56 `cudaStreamSynchronize` → events | 10-30% |
| 3 | `ds4_kernels.cu` | Unify `mma`/`wmma` paths | 5-15% kernel throughput |
| 4 | `ds4_iq.cu` | Add RDNA3 WMMA path | 5-15% IQ on 7900 XT |

## DeepSeek v4.1 Flash-1017 Notes
- This is the served model family per AGENTS.md ("DeepSeekV4.1-Flash-1017 is a served model family")
- Target: 7900 XT 20 GB (HIP/gfx1100) + 3060 12 GB (CUDA/sm_86)
- The 43 `mma` in `ds4_model.cu` likely target NVIDIA Tensor Cores — needs RDNA3 `wmma` port

## Verification

```bash
# Build on Linux ROCm box
cmake -S . -B build-hip -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build build-hip --target strata -j2

# Profile
nsys profile ./build-hip/strata --model <ds4 path>

# A/B test
python tools/pancake/abtest.py bench/specs/35.json --out bench/results/ds4.jsonl
```