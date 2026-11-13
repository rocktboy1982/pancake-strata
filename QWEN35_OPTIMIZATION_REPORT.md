# Qwen 35B-A3B (q36) Optimization Analysis: pancake-strata

## Files: `src/qwen35/` (~274 KB total)

| File | Size | Key Patterns |
|------|------|--------------|
| `q36_engine.cu` | 231 KB | 100 cudaMemcpy, 25 stream sync, 69 mmq, 33 cudaMalloc |
| `q36ref.cu` | 42 KB | Reference implementation |
| `q36_kernels.cuh` | 19 KB | Kernel declarations |

## Current State

| Pattern | `q36_engine.cu` | `q36ref.cu` |
|---------|-----------------|-------------|
| `cudaMemcpy` | 100 | 8 |
| `cudaStreamSynchronize` | 25 | 0 |
| `cudaDeviceSynchronize` | 7 | 1 |
| `cudaMalloc` | 33 | 5 |
| `printf` | 82 | 21 |
| `mmq` | 69 | 1 |
| `mma` | 7 | 0 |
| `dense` | 4 | 1 |

## Analysis

### 1. `q36_engine.cu` (231 KB) — 35B-A3B Engine
**Key findings**:
- **69 `mmq`** references — this is the llama.cpp MMQ (Mixture-of-Quantizers) path for routed experts. High usage.
- **100 `cudaMemcpy`** — high transfer count. Likely includes expert weight swaps + activation transfers.
- **25 `cudaStreamSynchronize`** — moderate sync count.
- **7 `cudaDeviceSynchronize`** — some full device syncs.
- **7 `mma`** — Tensor Core usage for dense layers.
- **4 `dense`** references — dense weight path.

**Critical insight**: This is the **Qwen 35B-A3B** engine (daily model per AGENTS.md: "Daily models: Qwen3.8-Flash-Next (IQ2_XS, Q2_0) and Qwen3.6-35B-A3B").

**MMQ usage (69 refs)**: The engine uses `mmq` heavily for the routed experts. The `Mmq` class wraps llama.cpp's quantized matmul.

**Optimization opportunities**:
- **Dense MMQ**: The 4 `dense` references suggest a dense weight path. The poly engine finding (dense MMQ +43% prefill) applies here too.
- **`cudaMemcpy` batching**: 100 copies — batch into async + events.
- **`mmq` kernel tuning**: 69 refs — check if the `Mmq::mul` calls are using optimal tile sizes for 35B.
- **Prefill batching**: The `prompt_batch` parameter should be tuned for 35B.

### 2. `q36ref.cu` (42 KB) — Reference Implementation
- Minimal patterns — reference/validation.

## Optimization Priorities

| Priority | Issue | Expected Impact |
|----------|-------|-----------------|
| 1 | Dense MMQ for down experts (like poly engine) | 20-40% prefill |
| 2 | Batch 100 `cudaMemcpy` → async + events | 10-20% decode |
| 3 | Reduce 25 `cudaStreamSynchronize` | 5-15% |
| 4 | Tune `mmq` tile sizes for 35B | 5-15% expert throughput |

## Verification

```bash
# Build CUDA (q36 is CUDA-only)
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=86-real
cmake --build build-cuda --target strata -j8

# Test dense MMQ
export STRATA_PROMPT_BATCH=1024
python tools/pancake/abtest.py bench/specs/36.json --out bench/results/q36-dense-mmq.jsonl

# Profile
nsys profile ./build-cuda/strata --model <qwen35b path>
```

## Notes
- The 35B-A3B at Q4_0/IQ2_XS fits in 7900 XT's 20 GB VRAM.
- On 3060 12 GB, the CPU expert fallback will be used — test with `--no-gpu-experts`.
- The dense MMQ optimization from the poly engine report applies directly here.