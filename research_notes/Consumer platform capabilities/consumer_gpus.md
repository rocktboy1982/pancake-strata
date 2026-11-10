# Consumer GPU capability matrix for pancake (local MoE inference: hot experts in VRAM, PCIe promotion, low-bit expert kernels)

Scope: AMD RDNA2/RDNA3/RDNA4 (+ Radeon PRO W7900 / AI PRO R9700), NVIDIA Ampere/Ada/Blackwell consumer, Intel Arc Alchemist/Battlemage.
Repo evidence is from a read-only grep of `/home/dan/clone-of-d/src/Strata` (pancake fork) on 2026-11-09; repo citations are given as file paths.
Current card: RX 7900 XT 20 GB, gfx1100, PCIe 4.0 x16 card at PCIe 3.0 on a Ryzen 7 5700G.

## Q1. Hardware capability matrix (VRAM, bandwidth, PCIe, matrix ops, wave size, LDS, caches, copy engines, ReBAR, software support)

### Takeaway
Within each vendor, the cards split into three tiers for pancake: (a) wave32 AMD with WMMA (RDNA3 gfx110x, RDNA4 gfx120x), which runs pancake's existing HIP kernels; (b) NVIDIA sm_86/89/120 with the CUDA path (sm_80+ floor); (c) Intel Arc on the separate SYCL port. The deciding hardware variables are VRAM (how many expert pairs fit) and PCIe link width x generation. Several mid-range cards are x8 (RX 7600/7600 XT, RTX 4060 Ti, RTX 5060 Ti, Arc B580), which halves promotion bandwidth on a PCIe 3.0/4.0 host.

### Cited Findings

**AMD architecture-level facts (ROCm GPU spec table)**
- ROCm's table lists the per-card LLVM target, VRAM, wavefront 32/64, LDS 128 KiB (WGP-level figure), L2 and Infinity Cache: RX 7900 XTX gfx1100 24 GiB, L2 6 MiB, IC 96 MiB, 96 CU; RX 7900 XT gfx1100 20 GiB, L2 6 MiB, IC 80 MiB, 84 CU; RX 7900 GRE gfx1100 16 GiB, L2 6 MiB, IC 64 MiB, 80 CU; RX 7800 XT gfx1101 16 GiB, L2 4 MiB, IC 64 MiB, 60 CU; RX 7700 XT gfx1101 12 GiB, IC 48 MiB; RX 7600 gfx1102 8 GiB, L2 2 MiB, IC 32 MiB, 32 CU — [ROCm GPU arch specs](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)
- Same table, RDNA4: RX 9070 XT gfx1201 16 GiB, L2 8 MiB, IC 64 MiB, 64 CU; RX 9070 gfx1201 16 GiB, 56 CU; RX 9070 GRE gfx1201 16 GiB, L2 6 MiB, IC 48 MiB; RX 9060 XT gfx1200 16 GiB, L2 4 MiB, IC 32 MiB, 32 CU; RX 9060 gfx1200 8 GiB; Radeon AI PRO R9700 gfx1201 32 GiB, L2 8 MiB, IC 64 MiB, 64 CU; AI PRO R9600 gfx1201 32 GiB, 48 CU — [ROCm GPU arch specs](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)
- Radeon PRO W7900 gfx1100 48 GiB, L2 6 MiB, IC 96 MiB, 96 CU; W7800 gfx1100 32 GiB (and a 48 GB variant) — [ROCm GPU arch specs](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)
- RDNA2 (RX 6000) is absent from that current table — [ROCm GPU arch specs](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)
- Official Linux ROCm (7.2.1 matrix) supports RX 9060/9060 XT/9070/9070 XT/9070 GRE, RX 7900 XTX/XT/GRE, 7800 XT, 7700/7700 XT, PRO W7900/W7800/W7700, AI PRO R9600/R9700; RX 7600 and all RX 6000 are not listed — [AMD Radeon ROCm Linux compatibility](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html)

**AMD matrix instructions**
- RDNA3 WMMA: only 16x16x16 tiles; A/B inputs FP16, BF16, IU8, IU4; accumulators FP32/FP16/BF16 (float inputs) or I32 (integer inputs); wave32 and wave64 variants; in wave32 the A/B data is replicated across lanes 0-15 and 16-31 — [GPUOpen, How to accelerate AI applications on RDNA 3 using WMMA](https://gpuopen.com/learn/wmma_on_rdna3/)
- Per-CU per-clock theoretical: 7900 XTX vs 6950 XT — FP16 512 vs 256, BF16 512 vs N/A, IU8 512 vs 512, IU4 1024 vs 1024 (i.e. RDNA3 doubled FP16 and added BF16; INT8/INT4 rate equals RDNA2's dot-product rate) — [GPUOpen WMMA on RDNA3](https://gpuopen.com/learn/wmma_on_rdna3/)
- RDNA4 WMMA intrinsics carry a `_gfx12` suffix; the fragment layout is unified (8 elements per thread, K-major A/B allowing 128-bit loads; no RDNA3-style lane replication described); FP16 native 16x16x16; FP8 and INT8 use 64-bit per-thread vectors and a "16x16x32" form built by fusing two WMMA instructions — [GPUOpen RDNA4 WMMA guide part 1](https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-1/), [part 2](https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-2/)
- RDNA4 (GFX12) adds FP8 and BF8 WMMA, including mixed FP8xBF8/BF8xFP8 forms, with FP32 accumulation — [Chips and Cheese, Examining AMD's RDNA 4 changes in LLVM](https://chipsandcheese.com/2024/01/28/examining-amds-rdna-4-changes-in-llvm/)
- RDNA4 adds hardware 4:2 structured sparsity (SWMMAC) that roughly doubles dense WMMA throughput when used — [GPUOpen, Accelerating generative AI on AMD Radeon GPUs](https://gpuopen.com/learn/accelerating_generative_ai_on_amd_radeon_gpus/)
- RDNA2 has no WMMA and no `v_dot4_i32_iu8`; dp4a uses the signed `v_dot4_i32_i8` (`__builtin_amdgcn_sdot4`) — repo `docs/AMD_HIP.md` (RDNA2 section)
- Repo statement of shared per-workgroup resources: RDNA2, RDNA3 and RDNA4 all run "wave32, 64 KiB of LDS per workgroup" with the same kernels — repo `docs/AMD_HIP.md` (RDNA4 and RDNA2 sections)

**AMD bandwidth / PCIe**
- RX 9070 XT: 256-bit, 640 GB/s, 64 MB Infinity Cache, PCIe 5.0 x16 — [StorageReview](https://www.storagereview.com/?p=142830); [Guru3D](https://www.guru3d.com/review/review-radeon-rx-9070-xt-reference-edition/)
- RX 9060 XT: full PCIe 5.0 x16 (unlike RX 7600 XT, restricted to PCIe 4.0 x8); TechSpot's PCIe test found the 9060 XT benefits from its 16 lanes — [TechSpot RX 9060 XT review](https://www.techspot.com/review/2996-amd-radeon-rx-9060-xt/); [Club386](https://www.club386.com/amd-radeon-rx-9060-xt-16gb-review/)
- RX 7600 class: PCIe 4.0 x8 — [TechSpot RX 9060 XT review spec table](https://www.techspot.com/review/2996-amd-radeon-rx-9060-xt/)
- RX 7900 XT: 320-bit, 20 Gbps, 800 GB/s — [technical.city comparison](https://technical.city/en/gpu/Radeon-RX-7900-XT-vs-Radeon-RX-9070-XT) (aggregator; consistent with 320-bit x 20 Gbps arithmetic)
- Measured host-to-device on an RX 7900 GRE (PCIe link not read): 28.0-28.1 GB/s by the engine's startup probe — repo `bench/results/2026-10-04-community-rx-7900-gre/README.md`
- Measured on pancake's own host (PCIe 3.0 x16, 5700G): PCIe probe 13.4 GB/s — repo `MOD_PLAN.md` (~line 327)

**NVIDIA (Blackwell RTX whitepaper tables)**
- RTX 3090 / 4090 / 5090: 24 / 24 / 32 GB; 384 / 384 / 512-bit; 936 / 1008 / 1792 GB/s; L2 6,144 / 73,728 / 98,304 KB; total L1/shared 10,496 / 16,384 / 21,760 KB; PCIe Gen 4 / Gen 4 / Gen 5; peak INT8 tensor 284.7/569.4, 660.6/1321.2, 838/1676 TOPS (dense/sparse) — [NVIDIA RTX Blackwell GPU Architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
- RTX 3080 / 4080 / 5080: 10 / 16 / 16 GB; 760 / 716.8 / 960 GB/s; L2 5,120 / 65,536 / 65,536 KB — same whitepaper
- RTX 3070 Ti / 4070 Ti / 5070 Ti: 608 / 504 / 896 GB/s; 5070 Ti has 49,152 KB L2 — same whitepaper
- RTX 3070 / 4070 / 5070: 448 / 504 / 672 GB/s; PCIe Gen 4 / Gen 4 / Gen 5 — same whitepaper
- Each Blackwell SM: 4 5th-gen tensor cores, 256 KB register file, 128 KB L1/shared — same whitepaper
- Blackwell 5th-gen tensor cores add FP4 and FP6; FP4 doubles FP8 throughput — same whitepaper
- Ada's 4th-gen tensor cores add FP8 relative to Ampere — [Wikipedia GeForce 40 series](https://www.Wikipedia.org/wiki/RTX_4080) (secondary; the NVIDIA Ada whitepaper was not retrieved)
- RTX 4060 / 4060 Ti: PCIe 4.0 x8; other RTX 40 cards x16 — [Wikipedia GeForce 40 series](https://www.Wikipedia.org/wiki/RTX_4080); [TechPowerUp RTX 4060 Ti review](https://www.techpowerup.com/review/asus-geforce-rtx-4060-ti-tuf-oc/single-page.html)
- RTX 5060 Ti 16 GB: PCIe 5.0 x8; 8 GB: x8; x8 limit hurts most on PCIe 4.0 hosts — [TechSpot RTX 5060 Ti PCIe benchmark](https://www.techspot.com/review/3004-nvidia-rtx-5060-ti-pcie-benchmark/)

**Intel Arc**
- Arc B580: 12 GB GDDR6, 192-bit, 456 GB/s, PCIe 4.0 x8, 20 Xe cores, 160 XMX engines — [Intel ARK B580](https://www.intel.com/content/www/us/en/products/sku/241598/intel-arc-b580-graphics/specifications.html)
- Battlemage: 18 MB shared L2 (B580); XMX carried forward from Alchemist, improved INT8 handling — [Chips and Cheese, Intel's Battlemage architecture](https://chipsandcheese.com/p/intels-battlemage-architecture)
- XMX data types INT2, INT4, INT8, FP16, BF16, TF32 (no FP8 listed) — [TechPowerUp B580 preview](https://techpowerup.com/review/intel-arc-b580-battlemage-unboxing-preview/2.html); Xe2 XMX 2048 FP16 / 4096 INT8 ops per Xe-core per clock — [3DTested](https://www.3dtested.com/pc-components/gpus/intel-announces-the-arc-b580-and-arc-b570-gpus)
- Resizable BAR: Intel's support article only says it "can result in a performance improvement" and gives BIOS steps (Above 4G Decoding, Re-Size BAR) — [Intel support 000090831](https://www.intel.com/content/www/us/en/support/articles/000090831/graphics.html)

**Copy / DMA engines (repo evidence; vendor counts not found)**
- HIP: adaptive-tier expert swaps run on the SDMA engine by default; `STRATA_HIP_ADAPT_KERNEL_COPY=1` copies with a kernel instead (workaround for a gfx1030 hang) — repo `docs/AMD_HIP.md` (line ~273)
- HIP blocking expert uploads use one reusable pinned staging buffer — repo `docs/AMD_HIP_PERFORMANCE.md`
- `--pcie-mode dma` + `STRATA_DMA_BATCH=1` batches a verifier group's copies — repo `docs/BATCHED_DMA.md`
- SYCL: prompt expert copies run on the copy engine by default on Battlemage (`STRATA_COPY_ENGINE`) — repo `docs/INTEL.md` (lines ~204, 388)

### Inferences
- Condensed matrix (VRAM / BW / link / matrix types / backend). Values are from the cited rows above unless marked (u) = commonly published vendor spec not re-fetched this session (see Gaps):

| Card | Target | VRAM | BW GB/s | Card link | L2 / IC | Matrix ops | pancake path |
|---|---|---|---|---|---|---|---|
| RX 6800/6900 XT | gfx1030 | 16 GB (u) | 512 (u) | 4.0 x16 (u) | 4 MB / 128 MB (u) | none (sdot4 only) | HIP plain kernels, unvalidated package |
| RX 7600 / 7600 XT | gfx1102 | 8 / 16 GB | 288 (u) | **4.0 x8** | 2 / 32 MB | WMMA FP16/BF16/IU8/IU4 | HIP gfx11, "reported working" |
| RX 7700 XT / 7800 XT | gfx1101 | 12 / 16 GB | 432 / 624 (u) | 4.0 x16 (u) | 4 / 48-64 MB | WMMA (RDNA3) | HIP gfx11, owner-validated |
| RX 7900 GRE / XT / XTX | gfx1100 | 16 / 20 / 24 GB | 576 (u) / 800 / 960 (u) | 4.0 x16 (u) | 6 / 64-96 MB | WMMA (RDNA3) | HIP gfx11, maintainer-tested; hipBLASLt table |
| PRO W7900 | gfx1100 | 48 GB | 864 (u) | 4.0 x16 (u) | 6 / 96 MB | WMMA (RDNA3) | gfx1100 binary applies (untested in repo) |
| RX 9060 XT | gfx1200 | 8 / 16 GB | 320 (u) | **5.0 x16** | 4 / 32 MB | WMMA FP16/BF16/FP8/BF8/IU8/IU4 + 4:2 sparsity | HIP gfx12, owner-validated |
| RX 9070 / 9070 XT | gfx1201 | 16 GB | 640 | **5.0 x16** | 8 / 64 MB | as above | HIP gfx12, maintainer-tested |
| AI PRO R9700 | gfx1201 | 32 GB | 640 (u) | 5.0 x16 (u) | 8 / 64 MB | as above | HIP gfx12, maintainer-tested (main RDNA4 dev box) |
| RTX 3060 / 3090 | sm_86 | 12 / 24 GB | 360 / 936 | 4.0 x16 | 3 (u) / 6 MB | TC FP16/BF16/TF32/INT8/INT4, 2:4 sparsity | CUDA, pancake's previous card (3060) |
| RTX 4060 Ti / 4080 / 4090 | sm_89 | 8-16 / 16 / 24 GB | 288 (u) / 717 / 1008 | **4.0 x8** / x16 / x16 | 32 (u) / 64 / 72 MB | + FP8 | CUDA sm_89 in release build |
| RTX 5060 Ti / 5070 / 5070 Ti / 5080 / 5090 | sm_120 | 8-16 / 12 / 16 / 16 / 32 GB | 448 (u) / 672 / 896 / 960 / 1792 | **5.0 x8** / 5.0 x16 | up to 96 MB | + FP4/FP6 | CUDA sm_120; NVFP4 w4a4 (sm_120a) |
| Arc A750 / A770 | Xe-HPG | 8 / 16 GB (u) | 512 / 560 (u) | 4.0 x16 (u) | 16 MB (u) | XMX FP16/BF16/INT8/INT4/INT2 | SYCL, JIT, FP16 XMX path only |
| Arc B580 / Pro B60 / Pro B70 | Xe2 | 12 / 24 / 32 GB | 456 / ? / ? | **4.0 x8** (B580) | 18 MB (B580) | XMX + int8 DPAS (ESIMD) | SYCL, AOT bmg-g21/g31, fused int8 DPAS experts |

- The 128 KiB LDS in the ROCm table is per WGP (two CUs); kernels see 64 KiB per workgroup on RDNA2-4 (repo statement), so LDS budgets are identical across all three AMD generations and need no retuning by generation.
- NVIDIA consumer parts all report a 128 KB L1/shared per SM; usable shared memory per block (~99-100 KB on sm_86/89/120) was not verified this session (see Gaps).

### Gaps
- Values marked (u) are standard vendor-published specs that I did not re-fetch this session (AMD product pages timed out; NVIDIA Ada whitepaper and CUDA compute-capability appendix were not retrieved): RX 6000 VRAM/bandwidth/link/IC; RX 7600 288 GB/s; RX 7700 XT/7800 XT/7900 GRE/XTX/W7900/R9700 bandwidth and link; RX 9060 XT 320 GB/s; RTX 3060 L2; RTX 4060 Ti/5060 Ti bandwidth; Arc A750/A770 specs; Arc Pro B60/B70 bandwidth and link width. Verify before relying on them.
- Number of SDMA engines per AMD consumer die, and NVIDIA copy engine counts per GeForce SKU: no vendor source found.
- Resizable BAR: no vendor statement found that ReBAR is *required* for compute (Intel only says "performance improvement"; AMD/NVIDIA consumer ReBAR is a gaming feature). Whether pancake's pinned-staging path depends on a large BAR was not found in the repo docs (grep for ReBAR/Resizable in docs returned nothing relevant).
- RDNA4 throughput per CU per clock for FP16/FP8/INT8 WMMA and the exact SWMMAC operand/metadata format were not found in the fetched GPUOpen pages.
- ROCm support for RX 6000 / RX 7600 on recent ROCm is unofficial; repo ships TheRock wheels for gfx1030/gfx1102 (see Q2), but AMD's matrix does not list them.

## Q2. Known pancake/Strata support per architecture and which kernel paths apply

### Takeaway
gfx1100 is pancake's tuned path today (ported Strata-rdna3 kernels, hipBLASLt table, *-hip presets). RDNA4 shares wave32/LDS/dot4 so the generic HIP kernels run, but it has its own `_gfx12` WMMA kernels, its own default table and a different PCIe split default; several gfx11-only prompt kernels (PF_GEMM/PF_PAD, chunked GDN on matrix cores) do not run on gfx12 and vice versa (PF_FUSED int8, SELECT_WMMA, gdn_wmma12). No code uses RDNA4 FP8 WMMA or SWMMAC sparsity. CUDA builds for 86/89/120; FP4 MMA is only used by the opt-in NVFP4 w4a4 path on sm_120a. Intel runs a separate SYCL port with XMX/ESIMD DPAS kernels.

### Cited Findings
- Support table: RX 7900 XT/XTX gfx1100 maintainer-tested; RX 9070/9070 XT and R9700 gfx1201 maintainer-tested; RX 9060 XT gfx1200 owner-validated; RX 7800 XT/7700 XT gfx1101 owner-validated; RX 7600 XT gfx1102 reported working, not validated; RX 6800/6900/6700 XT gfx1030/1031 packaged but not tested on a real card — repo `docs/AMD_HIP.md` lines 32-37
- Packages: Linux `rx7900` (gfx1100/1101/1102, ROCm 7.10.0a20251120 gfx110X-dgpu wheels), `rx9070` (gfx1200/1201, gfx120X-all), `rx6000` (gfx1030/1031, 7.13.0a); hipBLASLt tuning table shipped only for gfx1100 (1.2.0) on Linux; gfx1201 tables exist for 1.2.2/1.4.1/1.5.0 in `tools/hip` — repo `docs/AMD_HIP.md` lines 57-83, 397-399
- Binaries are arch-locked: startup compares GPU arch with compiled archs and requires wave32; a gfx1100-only build stops on gfx1201 — repo `docs/AMD_HIP.md` lines 193-195, 396
- gfx11-only switches: `STRATA_PF_GEMM`, `STRATA_PF_PAD` ("RDNA4 runs neither"); `STRATA_GDN_CHUNKED=1` runs prompt GDN in 16-token chunks on gfx11 matrix cores (accepts gfx1100/1101/1102/1150/1151); `STRATA_SCORES_THREAD` opt-in on gfx1100 (+10-13% at ~131K, +22% at 240K on a 7900 XTX) — repo `docs/AMD_HIP.md` lines 302-365
- gfx12-default switches: `STRATA_HIP_WMMA` (QSA prompt attention on matrix cores, default on gfx12 Linux, opt-in on gfx11), `STRATA_PF_FUSED` (prompt experts on int8 matrix-core kernels for IQ3_S/IQ3_XXS/Q2_0), `STRATA_SELECT_WMMA`, `STRATA_HC_Q8`, length-gated chunked GDN scan from 30K tokens — repo `docs/AMD_HIP.md` lines 302-312; `src/core/arch_defaults.cpp`
- gfx12 default env table (GDN_HEAD, GDN_PP=2, GDN_CONVL2, GDN_NOY, CVEC_FUSE, ATTN_LANECELL, ATTN_MERGE_V2, TSUM, EXPDEC, SMALL_UNR, GR_Q32_G, GR_Q8_PFS), measured +3.5% decode on an R9700 — repo `src/core/arch_defaults.cpp` lines 29-62
- AMD matrix intrinsics actually used in src: `wmma_f32_16x16x16_f16_w32` (15 sites), `_bf16_w32` (14), `_iu8_w32` (2) for gfx11; `_f16_w32_gfx12` (11), `_bf16_w32_gfx12` (1), `_iu8_w32_gfx12` (1) for gfx12 (files incl. `src/prefill/gdn_wmma12.inc`, `src/kernels/cuda/qsa_prompt_attn.cu`, `qsa_select.cu`, `src/split/src/hip/mma.h`); `__builtin_amdgcn_sudot4` (6). Zero FP8/BF8 WMMA or SWMMAC intrinsics — grep of repo `src/`
- gfx11-specific files: `src/kernels/cuda/rdna_mmvq.cu`, `fused_gr.cu`, `native_mmvq.cu`, `src/prefill/moe_fused.cu`, `src/split/src/gdn_tc_prompt.cuh`, parity test `src/kernels/qsa_scores_gfx1100_parity.cpp` — grep of repo `src/`
- RDNA2: no WMMA, QSA scorer takes ordered FP32 fallback; hipBLASLt has no gfx1030 kernels; rocBLAS FP16-in/FP32-out runs ~6.6x slower, hence opt-in `STRATA_HIP_PROMPT_F16`, `STRATA_RDNA2_SGEMM`, rocBLAS tuning table, `STRATA_HC_PERSIST` (not offered on gfx11/12) — repo `docs/AMD_HIP.md` lines 501-515
- Community results: RX 7800 XT decode 38-44 tok/s, prompt 898-953; RX 9060 XT decode 27.1 tok/s (llama.cpp HIP 20 tok/s on same card); RX 7600 XT ~43 tok/s Swift IQ3_XXS — repo `docs/AMD_HIP.md` lines 489-491; RX 7900 GRE 16 GB: IQ2_XS 65.3 tok/s decode, prompt 1,140-1,216 tok/s — repo `bench/results/2026-10-04-community-rx-7900-gre/README.md`
- RDNA4 measured: R9700 32 GB 12,288 slots vs RX 9070 XT 16 GB 4,931 slots (Coder IQ1_M): 16K decode 48.3 vs 35.4 tok/s — repo `docs/AMD_HIP.md` lines 432-439
- pancake 7900 XT presets: `configs/pancake/flashnext-iq2_xs-fast-hip.json` uses `--expert-cache auto --pcie-frac 0 --adapt-every 1 --adapt-swaps 32 --kv int8 --kv-resident 32768 --spec 4` + MTP + draft vocab; `run-pancake.sh` notes ~126 tok/s iq2_xs fast, ~168 turbo (lossy), q2_0 ~135 tok/s; strata-poly is CUDA-only — repo `run-pancake.sh`, `configs/pancake/`
- gfx1100 kernels for pancake were hand-ported from nicejunjie/Strata-rdna3 (rdna_gr.cu, rdna_mmvq.cu, fused DeltaNet megakernel, fused IQ MoE prompt kernels, SDMA copies, 2-GPU TP) — repo `docs/PANCAKE_FINDINGS.md` line 159
- CUDA: kernels need sm_80+ (tf32 mma, bf16); default release arch list "86-real;89-real;120"; sm_60/70 only via `STRATA_EXPERIMENTAL_SM60` — repo `CMakeLists.txt` lines 7, 229-248
- pancake's CUDA build for RTX 30 uses `-DCMAKE_CUDA_ARCHITECTURES=86-real` — repo `docs/PANCAKE_FINDINGS.md`
- NVFP4 experts: default `w4a8` uses int8 tensor-core MMQ (sm_75+) and software E2M1/UE4M3 decode; only `w4a4`/`w4a4x2` use Blackwell FP4 MMA (sm_120a only) — repo `docs/NVFP4.md` lines 88-93, 173-176
- `STRATA_GDN_CONVL2` is AMD-only because it is not bitwise on CUDA 13.3 sm_120 — repo `src/core/arch_defaults.cpp`
- Intel: SYCL port, official since 0.1.40.2; tested on Arc Pro B70 32 GB, A750 8 GB, two Pro B65; AOT `bmg-g31` (B70), `bmg-g21` (B580/B570/Pro B60); Alchemist is JIT-only; Linux only — repo `docs/INTEL_ARC.md` lines 3-37, 111-112, 185
- Intel kernels: XMX (`qsa_prompt_attn_xmx.dp.cpp`, `xmx_gemm_bench.cpp`) and fused int8 ESIMD DPAS expert products on Battlemage decoding IQ3_S/IQ3_XXS/IQ2_S in registers into the s8 DPAS B operand; without ESIMD (Alchemist AOT) the FP16 XMX path runs — repo `docs/INTEL_INT8_EXPERTS.md`, `docs/INTEL.md` line 88, `sycl/src/`
- Intel community: 2x Arc Pro B60 24 GB at PCIe 3.0 x8 per card, layer split, all experts in VRAM: Flash-Next IQ2_XS 58.6-61.3 tok/s decode; one B60 with 4,042 of 12,288 experts in RAM: 11.9 tok/s; B580 12 GB (WSL2) IQ2_XS ~21 tok/s — repo `docs/INTEL_ARC.md` lines 169-173

### Inferences
- Moving from the 7900 XT to any gfx110x card (7900 XTX/GRE, W7900) needs zero kernel porting; gfx1101/gfx1102 need only the arch added to `CMAKE_HIP_ARCHITECTURES` plus a hipBLASLt table (none shipped).
- Moving to RDNA4 (9070 XT / 9060 XT / R9700): the generic and `_gfx12` paths already exist upstream and are tuned by upstream maintainers on an R9700; pancake-specific ports from Strata-rdna3 (rdna_gr/rdna_mmvq, fused DeltaNet megakernel) would need re-validation on gfx12, and the `*-hip` presets (pcie-frac 0, adapt every window) were measured on gfx1100 only.
- Unused hardware: RDNA4 FP8 WMMA and 4:2 SWMMAC, Ada/Blackwell FP8, Blackwell FP4 outside NVFP4 packs. pancake's IQ2/IQ3/Q2_0 expert formats are integer-codebook formats decoded to int8, so the int8 WMMA/DPAS/IMMA paths matter more than FP8. Sparsity is unlikely to help: pancake measured Flash-Next experts as dense (gate-mask cos 0.844 at 70% sparsity) — repo `MOD_PLAN.md` item 13 result.
- RDNA2 and Alchemist lack the int8 matrix-core expert kernels pancake relies on for prompts; they are prompt-speed regressions relative to gfx1100 even at equal VRAM.

### Gaps
- No repo measurement of a W7900 or 7900 XTX running pancake's ported presets; no repo evidence for RX 9070 GRE or RX 9060 (8 GB).
- Whether pancake's fork retains every upstream gfx12 path after the hand-port (merge state) was not verified file-by-file.

## Q3. What each capability means for pancake: experts in VRAM, PCIe promotion vs CPU compute

### Takeaway
On Qwen3.8-Flash-Next IQ2_XS (24,576 expert pairs, ~1.4 MB each), VRAM sets the hit rate: 20 GB holds 10,909 pairs (44%); 24 GB ~56%, 32 GB ~79%, 48 GB all. Promotion costs ~115 us per pair on PCIe 3.0 x16 vs ~42 us of CPU compute, so the break-even link is ~33 GB/s effective; PCIe 4.0 x16 (~28 GB/s measured) is just below it, PCIe 5.0 x16 (~2x) is clearly above it — but a PCIe 5.0 host also has faster DDR5 and CPU cores, and upstream measured on an R9700 that the CPU pool, not PCIe, was the cheaper side (pcie-frac default lowered to 0.15). x8 cards on PCIe 3.0 hosts make promotion essentially unusable.

### Cited Findings
- 7900 XT: "Replay ceilings at 10,909 cached experts (44%): engine tier 79.6%, per-layer LRU 89.5%, Belady 94.7%" — repo `MOD_PLAN.md` line 411
- Promotion cost on pancake's host: "~115 us / 1.4 MB vs ~42 us CPU compute" (PCIe 3.0); PCIe 3.0 x16 ~12 GB/s, probe 13.4 GB/s; at 13.4 GB/s a 1.4 ms layer moves ~18 MB, ~12-15 experts — repo `MOD_PLAN.md` lines 321-336, 496
- Reuse-aware admission priced at PCIe 3.0 cost gave no gain; 7900 XT best setting `--pcie-frac 0` (Round A: 111.7 -> 118.4 tok/s) — repo `MOD_PLAN.md` lines ~405, 418
- Decode on the 3060 era: CPU experts ~half of decode (4.1 s of 8.75 s at 33.4 GB/s DDR4); prompt reading is GPU-compute bound, not PCIe bound — repo `docs/PANCAKE_FINDINGS.md` lines 50-59
- R9700 (Flash-Next IQ3_S, 24,576 pairs, 13,207 resident, i.e. 54% on 32 GB with larger IQ3_S pairs): pcie-frac 0.55 -> 0.15 gave +9.9% at 8K, +25.6% at 32K decode; "on an R9700 with idle CPU cores it is the pool" that is cheap; gfx12 native packs now start at 0.15 while CUDA and gfx11 keep their own measured values; the same class of finding as pcie-frac being wrong on the 3060 and P100 — repo `docs/AMD_HIP.md` lines 414-429 (host: Ryzen 9 3900X, PCIe 4.0 platform)
- With all experts on a card, splitting is pointless: R9700 alone holds all 12,288 pairs of Coder IQ1_M — repo `docs/AMD_HIP.md` lines 402-405
- 16 GB RDNA2: keep default pcie-frac/adapt-every; the 7900 XTX values cost 8.6 and 13.6 tok/s on an RX 6900 XT — repo `docs/AMD_HIP.md` RDNA2 section
- Narrow links: 2x RX 6900 XT at PCIe 4.0 x8 each, helper PCIe share cost the primary 43 ms per verify window before a fix (40 -> 66 tok/s); `--pcie-frac 0` avoided it — repo `docs/SECOND_GPU.md` line 129; multi-GPU probes each card's link separately, x4/x1 slots work — repo `docs/MULTI_GPU.md` lines 102-103
- x8 penalty is real on PCIe 4.0 hosts for RTX 5060 Ti; 9060 XT x16 benefits from 4.0/5.0 — [TechSpot RTX 5060 Ti PCIe benchmark](https://www.techspot.com/review/3004-nvidia-rtx-5060-ti-pcie-benchmark/); [TechSpot RX 9060 XT](https://www.techspot.com/review/2996-amd-radeon-rx-9060-xt/)

### Inferences
- Expert-fit model (IQ2_XS Flash-Next). From the measured point (20 GB -> 10,909 pairs x 1.4 MB = ~15.3 GB of experts), non-expert overhead with the pancake preset (int8 KV 32K resident, dense weights, MTP, workspace, reserve) is ~4.7 GB. Then pairs ~ (VRAM - 4.7 GB) / 1.4 MB:
  - 8 GB: ~2,400 pairs (~10%) — RX 7600, RX 9060, RTX 4060 Ti 8 GB, Arc A750
  - 12 GB: ~5,200 (~21%) — RTX 3060, RTX 5070, RX 7700 XT, Arc B580
  - 16 GB: ~8,100 (~33%) — RX 7900 GRE/7800 XT/9070 XT/9060 XT 16 GB, RTX 4080/5080/5070 Ti/5060 Ti 16 GB, Arc A770
  - 20 GB: 10,909 (44%, measured) — RX 7900 XT
  - 24 GB: ~13,800 (~56%) — RX 7900 XTX, RTX 3090/4090, Arc Pro B60
  - 32 GB: ~19,500 (~79%) — RTX 5090, AI PRO R9700, Arc Pro B70
  - 48 GB: all 24,576 (~34.4 GB experts + overhead) — PRO W7900 (and W7800 48 GB)
  These are linear estimates; KV/context settings move the overhead, and IQ3 packs have larger pairs (R9700 held 54% of IQ3_S pairs). The marginal value of VRAM is high below ~80% coverage because the hot-set hit curve is concave (Belady 94.7% at 44%).
- Promote vs compute balance: break-even effective link = 1.4 MB / 42 us ~ 33 GB/s for one use. Per pair: PCIe 3.0 x8 (~6.5 GB/s) ~215 us; 3.0 x16 ~105-115 us (measured); 4.0 x8 ~same as 3.0 x16; 4.0 x16 (~28 GB/s measured probe) ~50 us; 5.0 x8 ~ 4.0 x16; 5.0 x16 (~50-55 GB/s plausible effective, not measured in repo) ~26-28 us. So on paper only a PCIe 5.0 x16 card on a PCIe 5.0 host makes a single-use promotion cheaper than CPU compute on the *5700G*. But the comparison moves on such a host: AM5/Intel PCIe 5.0 platforms have DDR5 (roughly 2x the 33 GB/s DDR4 the CPU experts are bound by) and AVX-512 on Zen 4/5, cutting the CPU-side 42 us too. The upstream R9700 result (PCIe 4.0 x16, 12-core Zen 2) already favored CPU compute (0.15). Net: PCIe 5.0 favors promotion mainly for experts reused over several tokens (adaptive swaps / cache admission with reuse), not for one-shot misses; it should be re-measured with the `--pcie-frac` sweep rather than assumed.
- On the current 5700G (PCIe 3.0 only), any upgrade card runs at PCIe 3.0; x8 cards (RX 7600/7600 XT, RTX 4060 Ti, RTX 5060 Ti, Arc B580/B570) would get ~6.5 GB/s, so `--pcie-frac 0` and a larger VRAM hot set are the only levers. For this host, VRAM capacity dominates link speed: a 24 GB 7900 XTX (same gfx1100 binary/presets, +~2,900 pairs) is the lowest-porting-cost upgrade; a 32 GB R9700 (+~8,600 pairs) needs the gfx12 build and re-measured presets.
- Bandwidth for the resident part: decode on hits is GPU-memory bound; 7900 XT 800 GB/s vs 9070 XT 640 GB/s means RDNA4 16 GB is both smaller and slower for resident experts — its advantages are PCIe 5.0, FP8/sparsity (unused) and upstream gfx12 tuning. RTX 5090 (1,792 GB/s, 32 GB) is the strongest consumer card but runs on the CUDA path (pancake's strata-poly engine is CUDA-only, so it would also enable that engine).
- Intel Arc Pro B60/B70 (24/32 GB) give high VRAM per dollar but on the SYCL port only, with x8 links on at least the B580 die family; community 2x B60 results at PCIe 3.0 x8 show that with all experts resident, the link does not matter.

### Gaps
- No repo measurement of PCIe 5.0 x16 host-to-device bandwidth for the expert upload path, nor a pcie-frac sweep on a PCIe 5.0 host; the 50-55 GB/s figure above is an assumption.
- The ~4.7 GB overhead is derived from one data point; per-card fit numbers other than 20 GB are estimates, not measurements.
- No data on whether promotion uses SDMA concurrently with compute equally well on RDNA4 vs RDNA3, or the number of copy queues per card.
