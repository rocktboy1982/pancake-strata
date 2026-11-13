# HIP/ROCm Kernel Optimization Analysis: pancake-strata

## Target Hardware
- **RX 7900 XT 20 GB** (HIP/gfx1100, RDNA3, wave32 native)
- **RTX 3060 12 GB** (CUDA/sm_86, wave64)

## Kernel Inventory

| File | Size | RDNA-specific? | WMMA/MFMA? |
|------|------|-----------------|------------|
| `iq_kernels.cu` | 426 KB | No | Yes (Volta path) |
| `native_mmvq.cu` | 222 KB | Yes (wave32) | No |
| `qsa_select.cu` | 216 KB | Yes (gfx12) | Yes |
| `fused_gr.cu` | 200 KB | No | No |
| `qsa_prompt_attn.cu` | 159 KB | Yes (gfx11/gfx12) | Yes |
| `verify_kernels.cu` | 106 KB | No | No |
| `sampler.cu` | 79 KB | No | No |
| `s2_expert_grouped.cu` | 77 KB | No | No |
| `qsa_decode_attn.cu` | 72 KB | Yes (gfx11) | Yes (WMMA) |
| `dflash2.cu` | 61 KB | No | No |
| `qsa.cu` | 45 KB | No | No |
| `s_gemv.cu` | 42 KB | No | No |
| `rdna_gr.cu` | 39 KB | **Yes (gfx11 wave32)** | No (DPP) |
| `native_bf16.cu` | 38 KB | No | No |

## RDNA-Specific Code Already Present

### 1. `rdna_gr.cu` — Wave32 DPP (Data Path Parallelism) Optimization
- Uses `__builtin_amdgcn_update_dpp` for quad_perm, row_half_mirror, row_mirror all-reduce without LDS round-trip
- Uses `__builtin_amdgcn_permlanex16` for cross-lane permutation
- Uses `__builtin_amdgcn_sudot4` for int8 dot product (RDNA3 matrix core)
- `__attribute__((amdgpu_waves_per_eu(STRATA_GR_WPE, 16)))` on all 4 kernels (occupancy control)
- 256 threads/block = 8 waves of 32 (wave32 native)
- **Already optimized** for gfx1100

### 2. `qsa_decode_attn_wmma.cu` — WMMA (Matrix Fused Multiply Add)
- gfx11 wave32 16x16x16 layout
- `__builtin_amdgcn_wmma_f16_16x16x16_f16` for attention matmuls
- 256 threads/block
- **Already optimized** for RDNA3 matrix cores

### 3. `qsa_prompt_attn.cu` — gfx11/gfx12 WMMA + Q4
- `STRATA_HIP_WMMA` toggle for WMMA path
- `STRATA_HIP_WMMA_Q4` for Q4_0 KV format
- Arch detection via `hip_wmma_arch()` (checks `hip_wmma_arch` property)
- Q4 path disabled on gfx11 (FP8 not available, `v_wmma_f32_16x16x16_f16` requires gfx12)
- **Already optimized** for RDNA3

### 4. `native_mmvq.cu` — Wave32 Prototype
- `NATIVE_WAVE = 32` for RDNA (opt-in via `STRATA_RDNA_MMVQ_WAVE`)
- `__builtin_amdgcn_perm` for byte extraction
- **Prototype only** — not default enabled

## Optimization Opportunities

### Priority 1: Enable Wave32 MMVQ by Default (Medium Effort, High Impact)
**File**: `native_mmvq.cu` lines 2573-2581

The wave32 path (`NATIVE_WAVE = 32`) exists but is opt-in via `STRATA_RDNA_MMVQ_WAVE`. The comment says:
> "RDNA wave32 prototype, opt-in until measured and checked on the target card"

**Action**: Build with `STRATA_RDNA_MMVQ_WAVE=1` and run `abtest.py` to measure throughput gain.
**Expected**: 10-30% improvement on 7900 XT (RDNA3 wave32 native).
**Risk**: Wave32 has half the threads per wave (32 vs 64), may hurt on some shapes.

### Priority 2: Enable gfx11 WMMA Attention by Default (Medium Effort, High Impact)
**File**: `qsa_prompt_attn.cu` lines 2552-2558

The WMMA path for gfx11 is already implemented and gated by `STRATA_HIP_WMMA=1`.
> "gfx12 (RDNA4): on by default (STRATA_HIP_WMMA=1). STRATA_HIP_WMMA=1. STRATA_HIP_WMMA_MIN_Q (64...)"

**Action**: Enable `STRATA_HIP_WMMA=1` for gfx11 and run paired A/B.
**Expected**: 15-40% prefill improvement on attention-heavy models.
**Risk**: WMMA requires `hip_wmma_arch == 2` (gfx11). The Q4 variant is disabled (FP8 not available on gfx11).

### Priority 3: Add `amdgpu_waves_per_eu` to Non-RDNA Kernels (Low Effort, Medium Impact)
**Files**: `iq_kernels.cu`, `fused_gr.cu`, `qsa_select.cu`, `qsa_decode_attn.cu`

Currently only `rdna_gr.cu` uses `__attribute__((amdgpu_waves_per_eu))`. Other kernels don't control occupancy.

**Action**: Add `__launch_bounds__(TH) __attribute__((amdgpu_waves_per_eu(N, M)))` to the top 5 hottest kernels. N=4, M=16 is a reasonable starting point.
**Expected**: 5-15% improvement by preventing register spilling.
**Risk**: Wrong values can hurt performance. Measure with `hipcc` register report.

### Priority 4: Verify `__builtin_amdgcn_perm` Byte Extraction (Low Effort, Low Risk)
**File**: `native_mmvq.cu` lines 531-536

The `__builtin_amdgcn_perm` intrinsic for byte extraction is RDNA-specific. Verify it compiles on gfx1100 and that the generated assembly matches expectations.

**Action**: Compile with `hipcc -arch=gfx1100` and inspect the generated `.s` file for `V_PERM_B32` instructions.
**Expected**: Confirm no codegen bugs.

### Priority 5: Check `__shared__` Bank Conflicts on gfx1100 (High Effort, Variable Impact)
**Files**: `fused_gr.cu` (52 `__shared__`), `iq_kernels.cu` (34 `__shared__`)

RDNA3 has 16 shared memory banks. Bank conflicts in `__shared__` arrays can stall the pipeline.

**Action**: Run `rocprof` or `uProf` on the 7900 XT and check `srdsts` stall cycles. Add padding if conflicts found.
**Expected**: 5-20% improvement if conflicts are significant.
**Risk**: Padding wastes memory.

## Summary

The RDNA3-specific optimization work is **largely done** in this codebase:
- `rdna_gr.cu`: DPP + sudot4 + waves_per_eu ✅
- `qsa_decode_attn_wmma.cu`: WMMA ✅
- `qsa_prompt_attn.cu`: WMMA + Q4 ✅
- `native_mmvq.cu`: Wave32 prototype (opt-in) ⚠️

**Remaining work**:
1. Enable wave32 MMVQ by default + measure
2. Enable gfx11 WMMA attention by default + measure
3. Add `amdgpu_waves_per_eu` to other kernels
4. Profile shared memory bank conflicts on gfx1100

All verification requires the Linux ROCm box (7900 XT) for `abtest.py` + `rocprof`.

## Verification Plan

```bash
# Build
cmake -S . -B build-hip -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100

# Test 1: Wave32 MMVQ
export STRATA_RDNA_MMVQ_WAVE=1
python tools/pancake/abtest.py bench/specs/28.json --out bench/results/mmVQ-wave32.jsonl

# Test 2: WMMA Attention
export STRATA_HIP_WMMA=1
python tools/pancake/abtest.py bench/specs/29.json --out bench/results/wmma-attn.jsonl

# Profile
rocprof --kernel --output /tmp/trace.json ./build-hip/strata --model <path>
```
