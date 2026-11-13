# Poly Engine Optimization Analysis: 35B-A3B Dense MMQ Opportunity

## File: `src/poly/engine.cpp` (212 KB) + `src/poly/kernels.cu` (174 KB) + `src/poly/fattn_mma.cuh` (101 KB)

## Context

The `src/poly/` engine (`strata-poly`) is a second engine for long-context work. It's **CUDA-only** (not built on ROCm/7900 XT). It has:
- `Mmq` class wrapping llama.cpp's quantized matrix products (ggml-cuda mmq.cuh, MIT)
- `fattn_mma.cuh` with 311 `mma` references (Tensor Core attention)
- 41 `cudaStreamSynchronize` in engine.cpp
- 37 `cudaMemcpy` in engine.cpp
- `opt_.gpu_experts` flag (line 1207, 1332, 1915)

## The 35B-A3B Dense MMQ Finding

From the burzumishi scan: dense MMQ gave **+43% prefill** on the 7900 XT.

### How it works

**Current path** (`engine.cpp` lines 1907-1917):
```cpp
L.mmq = opt_.gpu_experts && rows_ >= kStageRows 
       && Mmq::fits((int) L.gu_type, 2 * n_ff_exp_, n_embd_)   // gu = up+gate experts
       && Mmq::fits((int) L.d_type, n_embd_, n_ff_exp_);       // d = down experts
```

The MMQ is only enabled when **both** the up+gate AND down experts have compatible GGUF types. For 35B-A3B, the down experts might not fit the MMQ tile constraint (line 1916), which forces the engine to fall back to the CPU expert path — the bottleneck.

**Dense MMQ optimization**: Route the **down experts** through MMQ even when they're not "routed" experts but dense layers. This is what `--no-gpu-experts` partially achieves — but the dense MMQ path needs explicit investigation in `src/poly/engine.cpp`.

### Why it matters

The poly engine is used for:
- Long-context (64K+) conversations
- Qwen3.6-35B-A3B (the daily 35B model)
- The 7900 XT has 20 GB VRAM — enough for the 35B at Q4_0/IQ2_XS

## Optimization Opportunities

### Priority 1: Dense MMQ for Down Experts (High Effort, High Impact)
**File**: `src/poly/engine.cpp` line 1907-1917

**Current**: `L.mmq` requires `Mmq::fits(L.d_type, n_embd_, n_ff_exp_)` — the down experts must be a compatible GGUF type AND have the right dimensions.

**Proposed change**: When `L.d_type` is `GGML_TYPE_Q4_0` or `GGML_TYPE_IQ4_XS` (common in 35B), force the down experts through MMQ regardless of tile constraint, by using the `dense_` arena path (line 1134, 1163).

**Key insight**: The `dense_` arena (line 1772) already allocates `dense_bytes_` for dense weights. The `dense_at_` cursor (line 1135) tracks the offset. If the down expert type doesn't fit MMQ, the engine falls back to CPU — that's the 43% loss.

**Expected**: 20-40% prefill improvement on 35B-A3B (CUDA-only, 3060 or above).
**Verification**: `abtest.py` spec 32 with `--prompt-batch=1024`.

### Priority 2: `cudaStreamSynchronize` Reduction (Low Effort, Medium Impact)
**File**: `src/poly/engine.cpp` (41 syncs)

**Action**: Profile with `nsys` to identify which syncs are in the hot prefill loop vs. init/teardown. Replace unnecessary ones with `cudaEventRecord`.
**Expected**: 10-20% prefill latency reduction.
**Risk**: Poly is CUDA-only. No ROCm impact.

### Priority 3: Tensor Core Attention (`fattn_mma.cuh`) (High Effort, High Impact)
**File**: `src/poly/fattn_mma.cuh` (101 KB, 311 `mma` refs, 38 `wmma`)

The `fattn_mma.cuh` file is the attention path using Tensor Cores. It's used by `fattn.cu` (39 KB, 20 `mma` refs).

**Action**: Check if the attention kernel is running at full SM occupancy. The `__syncthreads` count is 5 in `fattn.cu` + 17 in `fattn_mma.cuh`.
**Expected**: 5-15% attention throughput improvement.
**Risk**: Low if the kernel is already well-tuned.

## Verification Plan

```bash
# Poly is CUDA-only (3060 or above)
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=86-real
cmake --build build-cuda --target strata-poly -j8

# Test dense MMQ on 35B-A3B
export STRATA_PROMPT_BATCH=1024
python tools/pancake/abtest.py bench/specs/33.json --out bench/results/poly-dense-mmq.jsonl

# Profile
nsys profile ./build-cuda/strata-poly --model <35B-A3B path> --prompt-batch=1024
```

## Notes
- The poly engine is **CUDA-only** — no ROCm build. All verification requires a CUDA card (3060 12 GB or above).
- The `--no-gpu-experts` flag partially addresses the down expert issue but is a coarse control. Dense MMQ is the targeted fix.
- The `fattn_mma.cuh` file is inherited from llama.cpp and may have its own optimization opportunities (e.g., for sm_86 vs. sm_90 tensor core differences).
