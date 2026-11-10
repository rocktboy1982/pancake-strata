# Strata (fork) — experimental sm_75 prefill support

<p align="center">
  <b>High-Performance Local Inference for Qwen3.8-Flash-Next on NVIDIA Turing GPUs</b><br>
  <sub>Fork of upstream <a href="https://github.com/Niko1221/Strata">Strata</a> by Niko1221 · Licensed under MIT</sub>
</p>

> [!NOTE]
> **Experimental workbench fork — read this first.**
>
> This is a personal experimental fork of upstream [Strata](https://github.com/Niko1221/Strata) by Niko1221.
> I run models on an RTX 2060 12GB (Turing / `sm_75`), and I was hitting prefill bottlenecks on this generation of card.
> I worked on tracing and testing these adjustments on my machine with an AI coding agent as a second pair of eyes.
>
> **What is modified in this fork:**
>
> | File | What changed | Upstream PR status |
> |---|---|:---:|
> | `src/prefill/moe_fused_iq.cu` | Removed the check that blocked sm_75, since IQ3_S/IQ2_S blocks fit in 64 KB shared memory with `ASTAGES_DEV=2` | ✅ **[1-File PR branch](https://github.com/Niko1221/Strata/compare/main...dd777mine:Strata:fix/moe-iq-prefill-sm75-gate?expand=1)** |
> | `src/prefill/gemm.cu` | Enabled Tensor Cores for BF16 on Turing (`cc <= 75`) via FP16 fast path | ❌ Personal fork only |
> | `src/prefill/kernels.cu` | Enabled hardware GDN keyhead recurrence on sm_75 | ❌ Personal fork only |
> | `src/kernels/cuda/iq_kernels.cu` | Staged 2 KB `iq3s_grid` LUT in shared memory and unpacked signs via single-cycle IMAD | ❌ Personal fork only |
> | `serve/server.py` | Safe memlock limits | ❌ Personal fork only |
> | `tools/bench_turing_prefill.py` | Standalone script to test prefill latency | ❌ Personal fork only |
>
> Some of these I understand well, some I got to work by trying and testing things on my hardware. Treat this fork as an experimental workbench, not an official release. If you want only the clean, isolated fix, use the single-file PR branch.
>
> All numbers below are from my own machine and my own card (RTX 2060 12GB), not a formal benchmark rig.

---

## What this fork changes (and why)

In upstream Strata (including v0.1.45), Fused MoE prefill was restricted to `Q2_0` on NVIDIA Turing (`sm_75`) GPUs due to an assumed 64 KB Shared Memory (SMEM) limitation. As a result, formats like **`IQ3_S`** fell back to unfused scalar dequantization and host synchronization, causing noticeable latency on Turing cards.

This repository collects the changes made to get Turing cards running efficiently:

### Part 1: Unblocking Fused MoE Gate (`src/prefill/moe_fused_iq.cu`)
* **Shared Memory Adjustment:** Under `ASTAGES_DEV=2`, `IQ3_S` at 4 warps/row requires **58.75 KB** (fitting inside Turing's 64 KB limit with 5.25 KB headroom). For `IQ2_S` and `IQ2_XS`, using 2 warps/row reduces SMEM to **43.50 KB**.
* **Zero Host Transfers:** Removes the fallback to CPU dequantization, keeping the routing on the GPU.
* *Note:* This is the only change submitted upstream as an isolated 1-file PR: **[PR branch: fix/moe-iq-prefill-sm75-gate](https://github.com/Niko1221/Strata/compare/main...dd777mine:Strata:fix/moe-iq-prefill-sm75-gate?expand=1)**.

### Part 2: BF16 Tensor Cores on Turing (`src/prefill/gemm.cu`)
* Upstream excluded Turing from the tensor core path for BF16 (`cc < 75`), falling back to FP32 CUDA cores.
* Changing the check to `cc <= 75` enables the fast FP16 conversion path, reducing latency by another ~1.2 s on long prompts.

### Part 3: GDN Recurrence & Sign Unpacking (`src/prefill/kernels.cu`, `src/kernels/cuda/iq_kernels.cu`)
* **GDN Keyhead:** Unlocks Gated DeltaNet linear attention recurrence for compute capability 7.5.
* **Shared Memory Staging:** Stages the 2 KB `iq3s_grid` LUT in shared memory and accelerates sign bit unpacking via single-cycle `IMAD` instructions (`0x00204081u`).

---

## Verified Performance (RTX 2060 12GB @ 200W VBIOS)

Tested on Ubuntu 24.04 (Linux 7.0), CUDA 12.4, Intel Xeon E5-2650 v3, **Qwen3.8-Flash-Next IQ3_S** (3.5-bit, 262K native context limit):

| Workload / Prompt Length | Stock Strata (`v0.1.45`) | Pillar 1 Only (Upstream PR) | Full Turing Edition (Fork) | Full Speedup |
| :--- | :---: | :---: | :---: | :---: |
| **Warmup (169 tokens)** | 2.43 s (69.6 tok/s) | 2.73 s (61.9 tok/s) | **2.23 s (75.9 tok/s)** | **-0.20 s (-8.2%)** |
| **Short Prompt (343 tokens)** | 3.12 s (110.0 tok/s) | 3.26 s (105.2 tok/s) | **2.86 s (120.1 tok/s)** | **-0.26 s (-8.3%)** |
| **Medium Prompt (1 213 tokens)** | 7.69 s (157.8 tok/s) | 6.98 s (173.8 tok/s) | **6.85 s (177.1 tok/s)** | **-0.84 s (-10.9%)** |
| **Bulk Prompt (2 953 tokens)** | 9.81 s (301.1 tok/s) | 8.50 s (347.3 tok/s) | **7.69 s (383.9 tok/s)** | **-2.12 s (-21.6%)** |
| **Heavy Prompt (5 853 tokens)** | **14.11 s** (414.8 tok/s) | **12.34 s** (474.2 tok/s) | **10.11 s** (578.8 tok/s) | **-4.00 s (-28.3% latency!)** |
| **OpenCode Context (17 453 tokens)**| **43.51 s** (401.1 tok/s) | ~37.20 s (469.1 tok/s) | **32.68 s** (534.1 tok/s) | **-10.83 s (-24.9% latency!)** |
| **Clean Delta Prefill (~5.5K tokens)**| **501.3 tok/s** | **606.6 tok/s (+21.0%)** | **759.4 tok/s (+51.5%)** | **+258.1 tok/s (+51.5% speedup!)** |
| **Dequantization Overhead** | 2 059 ms (in DRAM) | **0 ms (resident on GPU)** | **0 ms (resident on GPU)** | **Eliminated in full** |
| **Host Grouping Stalls** | 48 ms (CPU stall) | **0 ms (resident on GPU)** | **0 ms (resident on GPU)** | **Eliminated in full** |
| **Pure Speculative Decode** | 31.7 tok/s | 31.8 tok/s | **32.0 tok/s** | Parity (Zero regression) |
| **MTP Draft Acceptance** | 82.2% | 81.9% | **73.9% – 82.2%** | High accuracy |

> **Hardware note & Transparency:** GPU running with VRAM at 8000 MHz (384 GB/s GDDR6 bandwidth vs 336 GB/s stock 7000 MHz), 200W VBIOS. All configurations were measured back-to-back on the exact same card at identical clocks, thermals, and machine configuration.

---

## Quickstart

### 1. Build from Source
```bash
git clone https://github.com/dd777mine/Strata.git
cd Strata
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build -j$(nproc)
```

### 2. Recommended Runtime Settings
```bash
export STRATA_PF_FUSED=1
export STRATA_BF16_TC=1
export STRATA_PREFILL_RING=248
export STRATA_MMVQ_IL_T1=0
export STRATA_GDN_KEYHEAD=1

# Start the server
python serve/server.py --config strata-iq3_s.json
```

---

## Reproduction Benchmark

To reproduce the exact numbers reported above, run the included standalone benchmark script (zero external dependencies, pure standard library Python):

```bash
# Standard sweep (Warmup -> 343 -> 1.2K -> 3.0K -> 5.8K -> Decode):
python tools/bench_turing_prefill.py --url http://127.0.0.1:8080/v1/chat/completions

# Full suite including the 17.5K token OpenCode prompt test:
python tools/bench_turing_prefill.py --url http://127.0.0.1:8080/v1/chat/completions --heavy
```

---

## Documentation

* 📊 **Official Community Benchmark Report:** [`bench/results/2026-10-18-community-turing-sm75/README.md`](bench/results/2026-10-18-community-turing-sm75/README.md)
* 📖 **Older GPUs Architecture Guide:** [`docs/OLDER_GPUS.md`](docs/OLDER_GPUS.md)
* 📋 **Community Benchmarks Index:** [`docs/COMMUNITY_BENCHMARKS.md`](docs/COMMUNITY_BENCHMARKS.md)
* 📜 **License:** [MIT License](LICENSE) (Niko1221 and Strata contributors).
