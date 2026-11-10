# AMD / older-GPU ports of Strata: techniques for RX 7900 XT (gfx1100) + Ryzen 7 5700G

State as of 2026-11-10. All repos are MIT-licensed (checked via `gh api repos/...`). Every tok/s number below was measured
by the repo author on the author's own hardware, not on ours, unless it says "ours". Our comparison point is
`/home/dan/clone-of-d/src/Strata` (branch `port/rdna3-decode`, HEAD `f15ad2b51`) and its daily preset
`configs/pancake/flashnext-iq2_xs-fast-hip.json` (IQ2_XS experts, `--prefill auto` with `STRATA_PREFILL_AUTO_MAX=32768`,
`STRATA_GDN_PP=1`, no `STRATA_DENSE_MMQ`). The earlier survey (`reports/Proiecte derivate din Strata.md`, 2026-11-09)
covers only nicejunjie/Strata-rdna3 from this list; the other seven repos were not in it.

Repo snapshot (gh api, 2026-11-10):

| Repo | Last push | Base | Target HW (author) | License |
|---|---|---|---|---|
| nicejunjie/Strata-rdna3 | 2026-11-08 (no commits since 11-08) | upstream v0.2.4.1+ | 2x 7900 XTX, 9950X3D | MIT |
| jstamagal/Strata-gfx1030 | 2026-11-09 (last commit 11-06) | upstream ~v0.2.11 | 2x Radeon PRO V620 (gfx1030), 9950X, 92 GB | MIT |
| vectorweft/strata-rdna4 | 2026-10-06 | upstream 0.1.40.1 | 2x R9700 (gfx1201), 9950X, DDR5 | MIT |
| QPhuGit/Strata-gfx906-Argon | 2026-11-09 | v0.2.10 (engine 0.1.56) | 1x MI50, 2x Xeon E5-2673 v4 (AVX2), DDR4-2133 | MIT |
| milpster/strata-dflash2-gfx906-mixed-rocm-cuda | 2026-11-09 | v0.2.11 | 2x Radeon VII + RTX 3080 Laptop, Ryzen 9 5900HX, 62 GB | MIT |
| kuwacom/Strata-legacy-gpu | 2026-11-09 | engine 0.1.57 | Tesla P40 (sm_61), 98 GB | MIT |
| DocPang/strata-k100ai | 2026-11-09 | v0.2.9 (engines 0.1.47 / 0.1.55) | Hygon K100AI (gfx928) | MIT |
| xenodeve/Strata-xeno | 2026-11-05 (last commit 10-15) | v0.1.42 | RTX 5060 Ti + 4070 SUPER, Windows | MIT |

## Which kernel changes are architecture-generic HIP improvements that would also help gfx1100?

### Takeaway
The biggest unported, gfx1100-relevant items are on the prompt (prefill) side: Strata-rdna3's late-October prefill
kernels (never ported; prefill expert GEMM -6..-9%, HC half-layer -7..-9%), Strata-gfx1030's "copy only the routed
experts" prompt path (+40% on Q4_0 prompts with experts outside VRAM) and int8-KV prompt attention on `v_dot4`, plus
cheap env A/Bs (`STRATA_GDN_PP=2`, `STRATA_DENSE_MMQ=1`). Decode-side candidates are smaller (+1..+3% each: gfx1030's
MMVQ_ROW/MMVQ_SET/NORM_ROPE=2, HEAD_WIDE/ROUTE_PLAN/HEAD_SAMPLE; Argon's AVX2 CPU expert kernels for our 5700G).

### Cited Findings

**nicejunjie/Strata-rdna3: commits that are NOT in our fork (verified: env names absent from `src/`, commit `3dbb40516` not an ancestor of HEAD)**
- `3dbb40516` (2026-11-01): prompt hyper-connection kernels. `gr_upmix` at 512 threads per block (VGPRs held to 127, 12 waves per SIMD instead of 6), `gr_write_norm_rs4` with 16-byte accesses; same bits; author's `hip_prefill_hc_bench` half-layer -7.4% (16K) .. -9.1% (4K), upmix -14..-19%. Toggles `STRATA_HC_UPMIX_NT=256`, `STRATA_HC_WN4=0` restore the old path. gfx1100-native (written on a 7900 XTX). — [commit list](https://github.com/nicejunjie/Strata-rdna3/commits/main)
- `1f4b77d72` (2026-11-01): FP16 pair expert kernel skips the matrix work of a wave's empty 16-token columns (`STRATA_NH_COLSKIP`, default on, bitwise). Author's bench: gate/up -23% / down -20% at 20 rows per expert, -2.5/-4% at 320; engine 2-GPU prefill +0.5..+2%, "1-GPU unchanged (copy-bound)". — [commit 1f4b77d72](https://github.com/nicejunjie/Strata-rdna3/commit/1f4b77d725025ddc93005fa1cdfdd0e31900b779)
- `1799c0a3a` (2026-11-01): same kernel, 3 blocks per WGP where LDS/VGPRs allow (`STRATA_PF_F16_OCC`), IQ2/IQ3 codebooks as FP16 with sign-bit masks (`STRATA_NH_HGRID`), Q2_0 codes straight to FP16 (`STRATA_NH_Q2FAST`, down instance capped 250 -> 154 VGPRs, no scratch); bitwise; "Engine prefill expert GEMM time -6..-9%" (IQ3/IQ4, 1 and 2 GPUs, 8K/32K). Our preset runs `STRATA_PF_F16=2` and IQ2_XS experts, so this kernel is on our prompt path. — [commit 1799c0a3a](https://github.com/nicejunjie/Strata-rdna3/commit/1799c0a3ac09525fd4af11bfb67e90c766e6165b)
- `c80244aab` (2026-10-24): three opt-in node cuts for the MTP draft-step graph (`STRATA_MTP_QFUSE`, `STRATA_MTP_MULTICOPY` (HIP), `STRATA_MTP_GR_R8`). No speed number in the message. Absent from our src. — [commit list](https://github.com/nicejunjie/Strata-rdna3/commits/main)
- `98d8c36f2` (2026-10-25): `STRATA_ARENA_SHM` keeps the expert arena in shared memory across engine starts (startup, not tok/s). `6d1baae13`: late refill on the SDMA engine through HSA (`STRATA_REFILL_SDMA=1`), measured only on 2-GPU TP. Both absent from our src. — [commit list](https://github.com/nicejunjie/Strata-rdna3/commits/main)
- Already closed in MOD_PLAN (do not re-report): long-context top-k (`STRATA_TOPK_SPLIT_MIN`) = upstream SELECT_LONG; `PF_GROUP_RES`; `PA_KLDS`; TP/peer/kfdq/megakernel parked for a second GPU. Newest rdna3 commits (`6677f0398` kfdq per-card accounting, `7df6538d6` slim split-GGUF tool, 11-07/08) are 2-GPU idle-power and tooling. — local `MOD_PLAN.md` lines 1137-1143, 1285, 1457-1464

**jstamagal/Strata-gfx1030 (RDNA2, wave32 like RDNA3, but no WMMA)**
- Author result on 2x PRO V620: prompts 1.7x-2.2x vs upstream with its RDNA2 switches (IQ3_S 16K 1,472 -> 2,487 tok/s), decode ~1.5x (IQ3_S 78 -> 121 tok/s), copy-heavy answers 2.6x (123 -> 323). — [docs/RDNA2.md](https://github.com/jstamagal/Strata-gfx1030/blob/main/docs/RDNA2.md)
- `STRATA_PF_STREAM_ROUTED=1`, `STRATA_PF_DIRECT=1` (+ `PF_NL`, `PF_Q51`, `MMQ_RESIDENT_SORT_NE`): "A prompt chunk copies only the experts its routing picked instead of every expert outside VRAM, and reads each expert where it lies"; Q4_0 prompts +40%/+43%, IQ3_S +10%/+9% (author); bitwise. Commit `043ba58c5` (2026-10-13). None of these env names exist in our src. Directly relevant: our experts mostly live in RAM, and prefill is PCIe-bound. — [docs/RDNA2.md](https://github.com/jstamagal/Strata-gfx1030/blob/main/docs/RDNA2.md); [commit search](https://github.com/jstamagal/Strata-gfx1030/commit/043ba58c5)
- `STRATA_PA_DOT4=1`: int8-KV prompt attention on `v_dot4_i32_i8`, one block per (query, KV head), online softmax; prompts +4%; FP32-level error (2.8e-6 vs FP64), needle checked to 128K; experimental. Commit `13ccf501a`. gfx1100 has the same dot4 instruction family; our preset already uses int8 KV and `STRATA_PROMPT_ATTN_IMMA=1` (WMMA int8), so the gain on gfx1100 is unknown and may be zero. — [docs/RDNA2.md](https://github.com/jstamagal/Strata-gfx1030/blob/main/docs/RDNA2.md)
- `STRATA_DQ8=1` (dense dequant, 4 threads per group, 16-byte stores, same bits) + `STRATA_HC_GATED16=1` (HC mix reads FP16 halves): IQ3_S prompts +6%. Absent from our src. — [docs/RDNA2.md](https://github.com/jstamagal/Strata-gfx1030/blob/main/docs/RDNA2.md)
- `STRATA_GDN_HEAD=1, STRATA_GDN_PP=2, STRATA_GDN_NOY=1`: barrier-free GDN recurrence, "recurrence 1.76x; prompts +2%". Our preset sets `GDN_PP=1`. Our own `src/core/arch_defaults.cpp` sets `GDN_PP=2` by default on gfx12 and gfx1151 (Strix Halo, RDNA3.5) with the note "two columns per lane: 64K prompt +2.9%". Never A/B'd on our card (no `GDN_PP` entry in MOD_PLAN). — [docs/RDNA2.md](https://github.com/jstamagal/Strata-gfx1030/blob/main/docs/RDNA2.md); local `src/core/arch_defaults.cpp:44,70`
- Decode switches: `STRATA_HC_WIDE=3, GR_NORM_FAST=1, TSUM=1` decode +7%; `STRATA_EXP_MODE=8` (routed experts on LDS layouts) +3%; `STRATA_HEAD_WIDE, ROUTE_PLAN, HEAD_SAMPLE` (head, router, sampler as fewer, wider launches) +1%; experimental `STRATA_MMVQ_ROW=1, MMVQ_SET=1, NORM_ROPE=2` (one wavefront per row for dense projections, all projections of one input in one launch, RMSNorm+RoPE fused) decode 119.2 -> 122.1 greedy (one A/B run). Of these, `HC_WIDE`, `HEAD_WIDE`, `ROUTE_PLAN`, `HEAD_SAMPLE`, `MMVQ_ROW`, `MMVQ_SET` are absent from our src. Our rdna3 ports (`GR_RDNA*`, `RDNA_MMVQ_WAVE=3`) cover the same HC/GEMV hot spots, so overlap is likely. — [docs/RDNA2.md](https://github.com/jstamagal/Strata-gfx1030/blob/main/docs/RDNA2.md)
- `STRATA_EXPERT_DOWN_L=1`: a 4-row down kernel behind any gate/up format, Q2_0 included; bitwise on synthetic weights (commit `fd9d70b3a`). `STRATA_TOPK_AUTO=1`: device picks the QSA top-k kernel by the query's block count (`5b4e3e7b3`). `STRATA_KV_PF_RESIDENT=1`: prompt path reads a streamed layer from its VRAM slots while the prefix fits them (`a83a9f11b`); prompts +7..+9% 16K-128K together with KV_PREFETCH. — [commit search results, gh api search/commits](https://github.com/jstamagal/Strata-gfx1030/commits/main)
- 2026-11-04/05 "basalt38" round (authored John Stamagal + GPT-6.1): int8 HC v2 with fp32 scales per 32 (`6c20f1599`), measured agent median +7.5% on 2 cards; "T1 diet" (`f4c65824b`, fewer launches in T=1 windows): "~-0.12 ms/T=1 window, tok/s flat; ~1.6 us per removed launch ... bytes are the lever". We already ship an int8 HC read (`STRATA_GR_RDNA_Q8=1`), so int8 HC v2 is probably a re-implementation of what we have. — [commit 6c20f1599](https://github.com/jstamagal/Strata-gfx1030/commit/6c20f1599); [commit b37918613 message](https://github.com/jstamagal/Strata-gfx1030/commits/main)
- `STRATA_PL_LOOKUP_STREAK=12` (copy-heavy answers 174 -> 302 tok/s) and `--pipeline-windows 2` (+12% decode) both need a 2-GPU layer split; our `generate.cpp:981` documents `--pipeline-windows` as "a layer split on exactly two GPUs". Not applicable on one card. — [docs/RDNA2.md](https://github.com/jstamagal/Strata-gfx1030/blob/main/docs/RDNA2.md)
- Negative results worth knowing: "No effect found on this PC: `--pool-workers` (0 to 12), `--host-core`, `STRATA_IO_THREADS`, `--pcie-frac`"; `STRATA_SPEC_COUPLED` makes sampled requests decode serially (-7..-11%); MTP depth 3 best (110.6/114.6/109.5/... tok/s for depth 2..8). — [docs/RDNA2.md](https://github.com/jstamagal/Strata-gfx1030/blob/main/docs/RDNA2.md)

**vectorweft/strata-rdna4 (stale since 2026-10-06, base 0.1.40.1)**
- Author result (2x R9700, 1 GPU): prompt 4K 454 -> 955, 32K 515 -> 1,422 tok/s; output at 4K 52.9 -> 60.2 tok/s vs upstream 0.1.40.1. — [README](https://github.com/vectorweft/strata-rdna4)
- Generic ideas: Q8_0 MMVQ with 8 warps per row for long rows (n_in >= 4096: `ssm_out`, `attn_output`) (`710b7f24d`); prefill HC injection folded into the FP32 xn pass because "the N = 4 rocBLAS GEMM ran at 0.5 TFLOPS" (`a4d0af68d`); prefill MMQ gate/up read in place through a per-expert pointer table, down gathered in one launch (`1ef49d5d2`); dense Q8_0 prompt projections through llama.cpp MMQ (int8) instead of FP16 dequant + rocBLAS (`8eb2e1976`); MTP draft head requantized to Q4_0 on device plus a smaller second draft head for later chain steps (`558aecbea`, `51c94ea2c`). The FP16-WMMA Q8_0 GEMM and the WMMA prompt attention are gfx12-specific. Both checked commits are absent from our object store; most of the rest was probably absorbed upstream by 0.1.5x (upstream gained gfx12 support at 0.1.30). — [commit list](https://github.com/vectorweft/strata-rdna4/commits/main)

**QPhuGit/Strata-gfx906-Argon (MI50 + AVX2 Xeons; CPU part is relevant to the 5700G)**
- Author result (IQ3_S, 1x MI50 at 130 W): chat decode 27.8 tok/s median, 16K prefill 527.7 tok/s (532.6 in the last commit). No per-change A/B attribution. — [README](https://github.com/QPhuGit/Strata-gfx906-Argon)
- `57aef6189` (2026-11-05) AVX2 CPU expert changes (generic x86 AVX2, so they apply to the Zen 3 5700G): (a) `row_dot_pair`: gate and up rows share each activation load and prefetch both weight streams together, "Bit-identical to two row_dot calls", used for Fmt32 types when NT <= 2 and not for types 17 (IQ2_XS) or 29 (IQ1_M); (b) the non-EX path's independent integer accumulators raised from 1 to 2 (`constexpr int A = EX ? ... : 2`); ours is still `: 1` (`src/kernels/cpu/iq_avx2_rows.inl:91`); (c) IQ2_S / IQ3_S grid indices assembled in SIMD (`_mm_unpacklo_epi8`) instead of a store/reload through `grid_indices`; scale lookup tables instead of `sc32(2*s+1)`; (d) IQ4_NL down rows two at a time with a precomputed |w| table; (e) `ExpertPool` row chunks aligned to 32/16-row grains and `phase_tasks` capped at rows/32; (f) `STRATA_CPU_MT1=1` uses the multi-token kernels from 1 token on AVX2-without-AVX-512 hosts. None of these exist in our tree (`row_dot_pair`, `STRATA_CPU_MT1` absent). — [commit 57aef6189](https://github.com/QPhuGit/Strata-gfx906-Argon/commit/57aef6189)
- GPU side: `gfx906_dlops.hpp` (DLOPS/DPP intrinsics), JTHREADS=64 in the GDN step kernel, 2-tier LDS: gfx906/wave64-specific. One generic piece: the GDN step kernel no longer writes the decayed state back between its two passes (decay recomputed in registers, one global store); the l2-norm kernel does 8 rows per block with wave-level reductions. — [commit 57aef6189](https://github.com/QPhuGit/Strata-gfx906-Argon/commit/57aef6189)
- Its env set includes `STRATA_DENSE_MMQ=1`, `STRATA_HIP_PROMPT_F16=1`, `STRATA_EXP_MODE=8`, `STRATA_SELECT_SIMT=1`, `STRATA_PREFILL_CPU_SHARE=auto`. — [README](https://github.com/QPhuGit/Strata-gfx906-Argon)

**kuwacom/Strata-legacy-gpu (Tesla P40, CUDA; ideas only)**
- Author result vs its 0.2.7 image: prefill 128K 434 -> 705 tok/s (+62%), 480K ~303 -> 561.6 (+85%), decode unchanged. — [docs/p40/README.md](https://github.com/kuwacom/Strata-legacy-gpu/blob/main/docs/p40/README.md)
- `STRATA_DENSE_MMQ=1` + `STRATA_MMQ_KQ=1` (dense MMQ for GDN projections): prefill +11..+13%, not bitwise, KL within noise. On our fork `STRATA_DENSE_MMQ` is a HIP path in `src/prefill/gemm.cu:1376` and is not set in the daily preset; no MOD_PLAN entry. On gfx1100 it competes with hipBLASLt FP16 WMMA, so the sign of the result is unknown. — [docs/p40/README.md](https://github.com/kuwacom/Strata-legacy-gpu/blob/main/docs/p40/README.md)
- Prompt chunk 8192 -> 32768: +6.6% at 24K/128K, "weights are re-read and re-dequantized once per chunk". Our preset already has `STRATA_PREFILL_AUTO_MAX=32768`. — [docs/prefill/chunk-profile.md](https://github.com/kuwacom/Strata-legacy-gpu/blob/main/docs/prefill/chunk-profile.md)
- `__launch_bounds__(64, 8)` on the QSA value kernel (`ps_value_d4_kernel`): 151 -> 128 registers, 6 -> 8 CTAs per SM, bitwise, `qsa attn` -10.9%, 128K prefill +1.45%. The author lists other kernels with no minBlocks in `__launch_bounds__`: `native_gu_multi`, `native_down_multi`, `native_mmvq_il`, `gr_down_staged`, `s2_expert_grouped`. On HIP the same method works: check VGPR count (e.g. `--save-temps` / `-Rpass-analysis=kernel-resource-usage`) and raise waves per SIMD. rdna3's `3dbb40516` used the same method (512-thread bound held VGPRs to 127). — [docs/prefill/qsa-value-occupancy.md](https://github.com/kuwacom/Strata-legacy-gpu/blob/main/docs/prefill/qsa-value-occupancy.md)
- HC down/inject read as BF16 (`STRATA_HC_DOWN_BF16=1`, a narrow 64x64x32 tile kernel) plus one shared bf16->fp32 conversion for down+inject: HC pair phase -18%, 128K prefill +1.36%; Pascal-only path (`bf16_path()==2`), but the profile shows the HC inject (N=4, K=10240) is bandwidth-bound on any GPU, the same thing rdna4 found ("N=4 rocBLAS GEMM at 0.5 TFLOPS"). — [docs/prefill/hc-down-bf16.md](https://github.com/kuwacom/Strata-legacy-gpu/blob/main/docs/prefill/hc-down-bf16.md)
- Rejected there: `STRATA_KV_PREFETCH` (+0.8%), CPU expert pool look-ahead ("waitCPU scales with CPU expert count (memory-bandwidth bound)", +0.8% at best), `STRATA_QFUSE`, `--draft-conf 0.5`, `spec` 3/5. — [docs/p40/README.md](https://github.com/kuwacom/Strata-legacy-gpu/blob/main/docs/p40/README.md)

**DocPang/strata-k100ai (Hygon DCU gfx928)**
- Author result: the 0.1.47 engine reads 8,190 tokens at 1,846 tok/s vs 1,034 for 0.1.55 (40K: 2,086 vs 1,161); 0.1.47 decode 79.37 tok/s with MTP acceptance 56/57. The cause is that "gfx928 prompt attention" never got back the 0.1.47 native MMAC path, so this is a port regression, not an upstream one. — [README](https://github.com/DocPang/strata-k100ai/tree/release/v0.3.0); [docs/k100ai/OPTIMIZATION.md](https://github.com/DocPang/strata-k100ai/blob/release/v0.3.0/docs/k100ai/OPTIMIZATION.md)
- Generic ideas listed (no per-item numbers): each unique token row quantized once, then gathered by expert index; fused SwiGLU + quantize + shared workspace; disabling FP32 write-backs that have no consumer; eight-lane GDN recurrence; rocBLAS stride padding and per-shape Tensile solutions; MTP `spec=6`, accept threshold 0.7, draft-graph warm-up. MMAC kernels are gfx928-only. Repo has only 2 commits on main (release dumps), so diffs are not inspectable. — [docs/k100ai/OPTIMIZATION.md](https://github.com/DocPang/strata-k100ai/blob/release/v0.3.0/docs/k100ai/OPTIMIZATION.md)

**xenodeve/Strata-xeno (NVIDIA, Windows; stale since 2026-10-15, base v0.1.42)**
- Author result: decode 69.2-72.7 tok/s vs 40-53 for upstream v0.1.38 on 2 cards; short prompts read 33-47% faster; RAM working set 23.7 vs 36.9-39.8 GiB. — [README](https://github.com/xenodeve/Strata-xeno)
- Transferable ideas: "An expert a GPU owns has no copy in RAM" (RAM saving); short prompt parts read through the decode path; checkpoints written off the critical path; a CPU pool that rests between rounds; a separate cache slot for side requests so they do not evict the main conversation. AVX-VNNI CPU kernels do not apply (Zen 3 has no AVX-VNNI). "Waking the CPU pool early" was measured and not shipped. — [README](https://github.com/xenodeve/Strata-xeno)

### Inferences
- Cheapest first (env only, our binary already has them): `STRATA_GDN_PP=2` (default on the two archs closest to ours) and `STRATA_DENSE_MMQ=1`. Each is a single A/B on the 41.8K prompt with `STRATA_PREFILL_TIMING=1`.
- Highest expected prefill gain for our RAM-heavy setup: gfx1030's routed-only expert streaming (`PF_STREAM_ROUTED`/`PF_DIRECT`). Our preset uses `STRATA_PIN_BY_NEED=1` and a 0.6 CPU prefill share, so check first whether upstream's current path already copies only routed experts. The +40% was on Q4_0 with more experts outside VRAM, so on IQ2_XS (more fits in 20 GB) expect less. Port difficulty: medium (prefill.cpp plus MMQ glue, HIP-only, gfx103x-gated: change the gate to include gfx11).
- Most direct gfx1100 port: rdna3's `3dbb40516` / `1799c0a3a` / `1f4b77d72` (same arch, same codebase family, bitwise, parity benches included). Easy to medium. Expect a few % of prompt time, not decode.
- Argon's AVX2 changes target the CPU expert path, which gates our decode window (MOD_PLAN: the GPU's own work is 15.9 of a 24.8 ms window). But `row_dot_pair` excludes IQ2_XS (type 17), our daily expert format, so the transferable parts for IQ2_XS are the extra accumulator (A=2), the pool grain alignment / `phase_tasks` cap and `CPU_MT1`. All need a bitwise check and a decode A/B. Easy to port (a few dozen lines), uncertain gain.
- Launch-bounds/occupancy audit (P40 and rdna3 both found 10-19% kernel wins this way) is a generic method worth one profiling pass over our HIP prefill/decode kernels.

### Gaps
- No repo publishes gfx1100 numbers except rdna3. Every gfx1030/gfx906/gfx1201/P40 percentage may not carry over (wave32 without WMMA vs RDNA3 with WMMA; HBM2 vs GDDR6).
- Whether upstream v0.2.13 already contains equivalents of gfx1030's `PF_STREAM_ROUTED`/`PF_DIRECT`: not verified (the env names are absent from our tree, but a renamed equivalent may exist).
- Argon publishes no isolated A/B for its AVX2 changes; their effect on Zen 3 is unknown.
- gfx1030's experimental decode switches were measured in one A/B run only.

## Any DFlash / speculative-draft variants (strata-dflash2) and their acceptance / speed numbers?

### Takeaway
milpster's fork runs DFlash2 on HIP (stock DFlash2 is CUDA-only) and gets +49% decode over stock on 2x Radeon VII, but
acceptance is capped at ~60-65% by the DFlash2 draft model itself. Fresh 16K prompt reads cost -12% for tap collection,
the same long-prompt penalty that made us close DFlash2 (115.9 tok/s short vs 84.3 after a long prompt).

### Cited Findings
- HIP-side DFlash2 drafter, "q4_0 draft measured optimal, `--spec 5` optimum"; additions over stock: "exact-match verification for sampled requests, bounded SWA draft ring with deep rewinds, draft-image parking, batched KV append, confidence gating". With the drafter off decode = stock speed ("the whole win is DFlash2, byte-identical output"). — [README](https://github.com/milpster/strata-dflash2-gfx906-mixed-rocm-cuda)
- Measured vs stock v0.2.10 (author, 2x Radeon VII + RTX 3080 Laptop, Swift-1.5 IQ3_XXS/IQ3_S): TG 494-tok pooled median 34.7 vs 23.3 tok/s (+49%); 1024-token completion 1.3x; fresh 16K prefill 299 vs 338 tok/s (-12%); 88K infill 472 vs 549 (-14%). — [README](https://github.com/milpster/strata-dflash2-gfx906-mixed-rocm-cuda)
- Sweep (JOURNAL 2026-11-04/05): q8_0 draft TG 37.3 vs 37.2, acceptance ~61.6% vs ~64.6% ("the ~60% acceptance ceiling is the dflash2 draft model itself (block 5), not its quantization"); `--spec 7` TG 36.1 (-3%), acceptance ~59.3%; `--dflash-block 3` flat (37.6, ~63.7%). Conclusion: spec 5 / q4_0 / block 5; "the only remaining lever is a different draft model". — [JOURNAL.md](https://github.com/milpster/strata-dflash2-gfx906-mixed-rocm-cuda/blob/main/JOURNAL.md)
- Other speculative numbers in this set: Strata-legacy-gpu runs DFlash2 with `spec=4`, `draft-conf=0.7`; `--draft-conf 0.5` "+7.7% did not reproduce ... effective gain was about +2%". k100ai (MTP, not DFlash) uses `spec=6`, threshold 0.7, MTP acceptance 56/57 on one 8K run. gfx1030: MTP depth 3 best. Xeno: draft acceptance 65-69% (MTP). — [P40 README](https://github.com/kuwacom/Strata-legacy-gpu/blob/main/docs/p40/README.md); [k100ai README](https://github.com/DocPang/strata-k100ai/tree/release/v0.3.0); [gfx1030 RDNA2.md](https://github.com/jstamagal/Strata-gfx1030/blob/main/docs/RDNA2.md); [xeno README](https://github.com/xenodeve/Strata-xeno)
- Draft-policy robustness (gfx1030, `c1af7ea34`, 2026-11-05, opt-in `STRATA_PL_POLICY_CLIP=2.10`): DraftPolicy's wall-clock cost EMA was bimodal ("one 31 ms stall at window 263 flips window 269 MTP T3 -> lookup T8, persists"); the fix winsorizes the upper tail of cost samples against each width's EMA after bootstrap. CPU tests only ("No GPU execution"), and it is gated on `pipeline_windows >= 2` in the code. — [commit c1af7ea34](https://github.com/jstamagal/Strata-gfx1030/commit/c1af7ea34); [commit 378cd8f4d message](https://github.com/jstamagal/Strata-gfx1030/commits/main)
- gfx1030 `f18d65cda`: MTP head proxy enabled only for Q8_0/2560 on gfx1030; replay passed 8403 top-K checks; no speed number. — [commit f18d65cda](https://github.com/jstamagal/Strata-gfx1030/commit/f18d65cda)

### Inferences
- Nothing here reopens DFlash2 for us: the -12/-14% long-read penalty matches our own closed measurement, and acceptance is model-bound.
- The policy-clip idea transfers to our single-GPU serve loop (MOD_PLAN item 43, DraftPolicy cost seeding): our CPU-expert windows also have stall outliers (adapt swaps, disk reads), and one outlier can flip the drafter choice. It would need to be wired outside the `pipeline_windows >= 2` gate. Easy, low expected gain, mostly stability.

### Gaps
- milpster's exact acceptance definition (drafts accepted / proposed per window) is not stated in the README. "drafts accepted 18/20" appears for one probe only.
- No author measured DFlash2 on RDNA3.

## Any mixed-GPU (ROCm+CUDA) pipeline idea relevant to a single GPU + CPU?

### Takeaway
The CUDA-helper design itself needs a second card, but three by-products apply to one GPU + CPU: host-RAM relief
(drop arena copies of experts held elsewhere), the streaming disk conversation cache with park-gain gating, and the
RAM+disk dual-park restores. These help agent-turn TTFT and RAM headroom, not kernel tok/s.

### Cited Findings
- CUDA helper process owns the NVIDIA card, fills ~3.6-4.6k expert slots (~6.5 GiB), and serves decode/verify and prompt rows (`STRATA_PREFILL_HELPER=1`). "Decode tier alone ~neutral on this pack (96–97 % cache hit) — its value is the prompt tier, expert capacity, and relief." — [README](https://github.com/milpster/strata-dflash2-gfx906-mixed-rocm-cuda)
- Host RAM relief (`--expert-helper-ram-relief`): 4116/4116 pairs dropped, 6.24 GiB back; deferred `cudaHostRegister`; opt-in `STRATA_ARENA_COLLAPSE`. Xeno has the same idea for GPU-owned experts ("An expert a GPU owns has no copy in RAM"). — [README](https://github.com/milpster/strata-dflash2-gfx906-mixed-rocm-cuda); [xeno README](https://github.com/xenodeve/Strata-xeno)
- Streaming disk conversation cache (2026-11-07/09): split-aware session format v2, O_DIRECT, bounded host RAM; park 1.1-1.3 s, restore 6.1-6.7 s vs 42-45 s cold reread (~12K tokens); with the RAM tier parked beside it ("dual-park") restores of 12,175 tokens take 0.32-0.51 s and 32,175 tokens ~0.64 s, token-exact. Park-gain gating (`5d6e3c4e4`) plus "idle publishes keep the conversation mounted" (`e155f3391`): 0 restores vs 11 of 12 turns before; sequential turns 16-19 s vs 80-96 s at 130-160K context. — [README](https://github.com/milpster/strata-dflash2-gfx906-mixed-rocm-cuda); [commit list](https://github.com/milpster/strata-dflash2-gfx906-mixed-rocm-cuda/commits/main)
- Our fork already has `--conversation-cache-park-gain` (`src/program/generate.cpp:949,1134`), but not milpster's streaming disk tier or `publish_worth` (absent from src). — local grep
- Short-prompt latency: `STRATA_PREFILL_CPU_SHARE=auto` (214 tokens 2.4x, 620 tokens 1.8x); `STRATA_PROMPT_STAGE_MEMO` +14-20% on short legs. `PROMPT_STAGE_MEMO` is absent from our src. Our preset uses `STRATA_PREFILL_CPU_SHARE=0.6`, `_MAX=4096`. — [README](https://github.com/milpster/strata-dflash2-gfx906-mixed-rocm-cuda)
- Runtime env on gfx906: `HIP_FORCE_P2P=1 HSA_ENABLE_SDMA=1`; a custom rocBLAS 4.1.0 because stock 6.1/6.4 segfaults on gfx906 (gfx906-only). — [README](https://github.com/milpster/strata-dflash2-gfx906-mixed-rocm-cuda)
- rdna4's second-GPU helper computes the experts it caches during prompt reading ("activations cross PCIe instead of weights") plus a share of the pinned-RAM experts: needs a second GPU. The single-GPU analogue is the CPU prefill share we already run. — [README](https://github.com/vectorweft/strata-rdna4)

### Inferences
- For a single 7900 XT, a second device could only be the 5700G's Vega iGPU (gfx90c, Vulkan-only under our ROCm 10), so the helper architecture is not applicable.
- `STRATA_PROMPT_STAGE_MEMO` and the streaming disk tier with dual-park are the transferable pieces. They overlap MOD_PLAN items 39/83 (conversation restore), where our restore target is ~0.5-1 s. Medium port (session format, O_DIRECT reader).

### Gaps
- What `STRATA_PROMPT_STAGE_MEMO` memoizes was not read from source; only the README claim (+14-20% short legs) is available.
- Restore numbers come from an NVMe + 62 GB host with fp16 KV; our models sit on ntfs3 (D drive) with int8 KV, so disk-tier timings will differ.
