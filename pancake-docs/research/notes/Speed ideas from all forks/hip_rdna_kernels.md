# HIP / ROCm / RDNA kernel speed ideas from the commits of all Strata forks, checked against pancake-strata (RX 7900 XT, gfx1100)

Scope and method (2026-11-10). Source: `data/fork-kb/forks.sqlite`, `commits` with `upstream_history = 0` (43,685 rows, 2026-09-24 .. 2026-11-10). I regex-searched messages for AMD terms (hip, rocm, rdna, gfx*, wave32, wmma, dpp, lds, vgpr, occupancy, hipblaslt, ...) together with kernel or speed terms. I also ran a second pass on decode-kernel words (mmvq, gemv, router, doorbell, launch, swiglu, ...) that carry a number. Subjects that already exist in pancake's own `git log --all` (upstream or ours) were dropped. That left 1,172 AMD-tagged subjects plus 20 numeric non-AMD ones, all read by subject line. About 45 got their full message read and 17 got a diff or doc read through `gh api` (read-only). READMEs/docs read: android-imdad/Strata-7900xtx (`docs/RX7900XTX.md`), rxr2/Strata (`docs/RX7900XTX.md`), unimatrix099/Strata (`FORK.md`), Ben1078/Strata (`RDNA3_MOE_OFFLOAD.md`). Pancake checks are greps of `src/`, `include/`, `configs/pancake/flashnext-iq2_xs-fast-hip.json`, MOD_PLAN.md, README.md, docs/PANCAKE_FINDINGS.md and pancake-docs/RESEARCH_AND_FINDINGS.md.

- Not repeated here: items already in `research_notes/Forkuri Strata câștig viteză/amd_rdna_ports.md` (nicejunjie/Strata-rdna3, jstamagal/Strata-gfx1030, vectorweft/strata-rdna4, QPhuGit Argon, kuwacom P40, DocPang k100ai) and `research_notes/Forkuri Strata active noiembrie/batch1-6.md` (this-cute-world STRATA_EXPDEC_V2 / FUSE_* / DPP_XOR / Q6K_V2 / MMVF_ROWS / DENSE_LDS, bsorensen110 COPY_WAIT_DEDUPE / ROUTED_STREAM / FUSE_DECODE, rpcribari, phoenixclyde, Lamina global-address hint). When one of those shows up again, it is only marked "(covered: ...)". Those standalone repos (rdna3, gfx1030, rdna4, ...) are not GitHub forks, so they are not in the KB anyway.
- Our reference point: the 7900 XT profile `bench/results/2026-11-09-7900xt-gpuprof/run_kernel_stats.csv` (local rocprofv3 run). Shares of GPU kernel time: combine_cpu_k10_vec4 38.3% (the GPU waiting for CPU expert rows); rdna_kq_kernel<23,...> IQ4_XS dense GEMVs ~11%; expdec_gu_kernel<22> IQ2_S gate/up 5.83% (53.4 us avg); native_down_multi_kernel<42> Q2_0 down 5.79%; rdna_gr_fused 5.6%; bf16 router GEMV 5.07%; wait_flag_copy_i32 4.07%; route_publish_res 2.68% (17.8 us avg); expdec_gu<16> IQ2_XXS 1.6%.
- MOD_PLAN says the window is about 24.8 ms, of which the GPU's own work is 15.9 ms, and CPU experts gate it. So a pure GPU-kernel saving shows up in tok/s only where the GPU path is on the critical path. Changes that shorten the path from a layer's router to the CPU pool's doorbell, or from the CPU publish back to the combine, act directly.
- Every speed number below is the fork author's own, from the commit message, diff comment or fork doc named. None is verified on our box.

## 1. Decode GEMV / MMVQ for the IQ2_S / IQ2_XXS / IQ2_XS gate/up and Q2_0 down experts (and dense IQ4_XS)

### Takeaway
The one new, clearly transferable idea is the cheaper IQ2/IQ3 grid-sign expansion: a multiply/add trick in place of the generic byte compare + byte subtract. It was measured on an RX 7900 XT itself: IQ2_XXS gate/up -26..-29%, IQ2_S down -35% per kernel in a GLM engine. A drop-in form exists for exactly our `iq_kernels.cu` vec_dot functions (RX5950XT, not measured on AMD). Pancake already has v_perm `__byte_perm`, SWAR `__vsub4/__vcmpne4` and DPP xor shuffles. The rest of this area is covered by earlier reviews (EXPDEC_V2, Q6K_V2) or is gfx906/gfx12-specific.

### Cited Findings
- **Packed sign expansion, RX 7900 XT measured (yqYo1/Strata `6f0bdf665b`, 2026-10-08; author boxwrench, GLM engine `glm_fast.cu`).** A multiply spreads four sign bits into the high bit of four bytes, then masks and shifts give the XOR mask and the two's-complement correction. Codebook bytes are positive and nonzero, so there is no carry between bytes. Down uses 2 output rows per workgroup instead of 4 (fewer VGPRs). Synthetic decode bench on an RX 7900 XT, ROCm 7.2.1, weights rotated through 5 pools to exceed the 80 MiB cache (author, `docs/AMD_EXPERT_KERNELS.md`):
  - IQ2_XXS gate/up 92.6 -> 65.6 us (-29.1%, 374 -> 527 GB/s)
  - IQ2_S down 57.8 -> 37.6 us (-35.0%)
  - IQ2_XXS gate/up (IQ3_XXS fixture) -26.4%
  - IQ3_XXS down -9.8%
  - 620 parity comparisons byte-identical (exact)
  - Env: `STRATA_HIP_EXPERTS_LEGACY=1` restores the old kernels. Shape-gated to n_embd 4096 / n_ff 2048 (GLM), gfx1100 / gfx1151.
  - Sources: [commit](https://github.com/yqYo1/Strata/commit/6f0bdf665b4c01a24d086210a2506beae3f87b56), doc in the same diff.
- **Same idea as a drop-in for our file (RX5950XT/Strata `b9a2111d3f`, 2026-10-19, `-DSTRATA_HIP_SWAR_SIGNS=1`, compile-time).** The 32 `__vcmpne4/__vsub4` sign pairs in `iq_kernels.cu` (vec_dot_iq2_xxs / iq2_xs / iq2_s / iq3_xxs / iq3_s) go through `sign_mask(t) = (t + 0x7F7F7F7F) & 0x80808080` and `apply_sign_mask(g, m) = (g ^ (m | (m - (m >> 7)))) + (m >> 7)` (llama.cpp's ggml_cuda_apply_signs without its multiply). Exact: no grid byte is 0, 48,020 cases checked.
  - Author static analysis: "~25% fewer weighted VALU in the 112 changed kernels, but 17-19 of them lose waves (native_gu_multi 16 -> 10-12)", expdec_gu by 1-2 waves (ROCm 7.2.4, gfx1100/gfx1201).
  - Not measured on an AMD card. The NVIDIA A/B was within noise, as identical SASS predicts.
  - Source: [commit](https://github.com/RX5950XT/Strata/commit/b9a2111d3f4ece53487ce054e1b23aa3a8939214)
- **Pancake status.** `src/kernels/cuda/iq_kernels.cu` still has 34 `__vcmpne4/__vsub4` sites, for example vec_dot_iq2_xs lines 148-153 and iq2_s lines 182-185. On the HIP build these map to the generic SWAR helpers in `include/strata/hip_compat/intrinsics.hpp:76-100`: vsub4 is about 6 ops, vcmpne4 about 6 ops plus a multiply. The specialized sign forms need about 2 + 5. Our profile's matching kernels are expdec_gu<22> (IQ2_S, 5.83%) and expdec_gu<16> (IQ2_XXS, 1.6%). Q2_0 down (5.79%) has no signs, so it is not affected. Not mentioned anywhere in MOD_PLAN or the findings docs.
- **Already in pancake, do not re-port:**
  - StevenChenSE/Strata `7e7e6e15d5` (2026-10-01), v_perm_b32 `__byte_perm` + SWAR byte ops. Author: R9700, engine 0.1.30, decode 55.0 -> 67.5 tok/s. Pancake has both in `include/strata/hip_compat/intrinsics.hpp:60-100`, upstream-absorbed. — [commit](https://github.com/StevenChenSE/Strata/commit/7e7e6e15d548728c42544ba01d8064f2e3935e2a)
  - DPP row_xmask / v_permlanex16 xor shuffles. Pancake: `intrinsics.hpp:119-150`, our commit 89da64b93 (covered: batch3 F5 / batch6, this-cute-world and rpcribari).
- **Warp-per-2-rows slots kernel (RandoTeam/4gbVramStrata `7b6e93d1e4` + `5c0bd2c991`, 2026-10-25, `STRATA_SLOTS_WARP`, default on for HIP).** When a row's blocks fit one warp (`blocks_per_row * T <= 32`, e.g. a 512-wide IQ4_XS down row), native_mmvq_slots_warp_kernel gives each warp its own rows. No LDS, a quarter of the blocks, 91 VGPRs, 16 waves/SIMD. Bit-for-bit. Author on RX 7800 XT (gfx1101), Qwen3.6-35B-A3B UD-Q3_K_M: generation 98.5 -> 111.4 t/s short, 83.5 -> 92.7 after 20K. Pancake has `native_mmvq_slots_kernel` (`src/kernels/cuda/native_mmvq.cu:1692`, used by `src/poly/engine.cpp:2372`) but no warp variant. It applies to the strata-poly / Qwen3.6-35B path, not to the Flash-Next expdec path. — [7b6e93d](https://github.com/RandoTeam/4gbVramStrata/commit/7b6e93d1e4dad5fb9d7f95afd49258f33fd8e856), [5c0bd2c](https://github.com/RandoTeam/4gbVramStrata/commit/5c0bd2c991de7964b2cd4c9495e1be3346976f2c)
- **gfx12-specific, not for us as is.** mandoof1/Strata `418e117033` (2026-10-17): `STRATA_HC_TSUM=1` (TSUM is closed in our MOD_PLAN as neutral on 7900-class cards) and a measured gfx1200 rows table for the interleaved GEMV (IQ4_XS -17..-20%, Q4_K -20..-35% at 2-4 columns, RX 9060 XT). Our dense GEMVs run `rdna_kq_kernel` (`STRATA_RDNA_MMVQ_WAVE=3`), not native_mmvq_il. `e4b4253994`: `STRATA_HC_Q8_LDS=1` (exact, LDS-staged HC Q8 down/up), for the HC_Q8 path that our preset has off (`STRATA_HC_Q8=0`). — [418e117](https://github.com/mandoof1/Strata/commit/418e1170333559f379a792a7202c516df5153c4f), [e4b4253](https://github.com/mandoof1/Strata/commit/e4b425399402cd96276ef5a5b4f1b7adb0664952)
- **Static audit from a 7900 XTX tuner (rxr2/Strata, `docs/RX7900XTX.md`, commit `cc21cbf4a9`, 2026-10-30).** "The decode projections (mmvq) already run at 70-100% of VRAM bandwidth (static audit). Remaining kernel ideas are worth 0.3-2.4% each." Decode window 28-30 ms: GPU ~16 ms (55%), CPU experts after VRAM misses 6.3-6.9 ms (22%), MTP draft ~3 ms. — [commit](https://github.com/rxr2/Strata/commit/cc21cbf4a9bf56c70177fdccd3fc6c5d6ad57d97)
- **Negative on gfx1100, from two independent 7900 XTX forks.**
  - Ben1078 RDNA3_MOE_OFFLOAD.md: "gemv retuning for the dense side 48.5 vs 48.2 tok/s (+0.6%, noise)".
  - android-imdad docs/RX7900XTX.md: upstream's `STRATA_DEQUANT_WIDE=1` "bit-exact here but 8-14% slower than the kernel it replaces, which already writes ~1.6 TB/s".
  - Sources: [Ben1078 doc](https://github.com/Ben1078/Strata/blob/rdna3-moe-offload-perf/RDNA3_MOE_OFFLOAD.md), [android-imdad doc](https://github.com/android-imdad/Strata-7900xtx/blob/rx7900xtx/docs/RX7900XTX.md)

### Inferences
- **Sign trick: estimated gain.** If expdec_gu<22>/<16> get 15-30% faster on our card (boxwrench's numbers are on the same GPU model, but for a GLM kernel with a different launch shape), that saves ~1.1-2.2% of GPU kernel time. Decode gain: 0-1.5%. It is higher only in windows where the VRAM experts, not the CPU share, finish last. The risk is the wave loss RX5950XT reports, so the A/B has to be per kernel. Exact. Port effort S: macro switch in one file, plus a parity run with `native_grouped_parity` / `expdec` bench.
- Q2_0 down (5.79%) is unaffected. Our IQ2_XS-file layers that use IQ2_XS gate/up (type 17) would be, if any of them are in VRAM: vec_dot_iq2_xs is among the changed functions.

### Gaps
- No fork measured the RX5950XT form on any AMD card. boxwrench's measurement is in a different kernel (glm_fast.cu, n_embd 4096) and a different engine. Neither number transfers directly to `expdec_gu_kernel<22,1,4>` at n_embd 2560 / n_ff 640.
- Whether the compiler already reduces `vcmpne4(x & mask, 0)` + `vsub4` to fewer ops on gfx1100: not checked (needs `--save-temps` ISA, i.e. a compile, which was not allowed in this session).

## 2. Attention decode (QSA select / top-k / decode attention)

### Takeaway
Nothing big and new for one-user decode. android-imdad's top-k "keys in registers, block scans" (`STRATA_TOPK_REGKEYS/TOPK_SCAN`, exact, gfx1100) is absent from pancake but has no isolated number. The Lamina and xyzzing QSA scorer work looks upstream-absorbed. Split-K decode attention (`STRATA_ATTN_SPLITS`) saves VRAM for dense-attention models, not speed for Flash-Next.

### Cited Findings
- android-imdad/Strata-7900xtx `caa9b00115` (2026-10-13), part of the withdrawn upstream PR #2231 bundle, on by default in that fork:
  - `STRATA_TOPK_REGKEYS` / `STRATA_TOPK_SCAN`: QSA top-k keys in registers, block scans.
  - `STRATA_MTP_SELECT_WIDE`: drafter row select over 40 blocks.
  - `STRATA_ATTN_MERGE_BOUND`.
  - `STRATA_GDN_NORM128`: GDN output norm with only the 128-thread row group that works, DPP exchanges.
  - Bundle result vs upstream main a0116b27 on an RX 7900 XTX, IQ3_S, strict exact config (author): short chat 86.8 -> 95.8 tok/s (+10%), agent decode +11%, teacher-forced KL 0.0000. Per-switch numbers are not given.
  - None of these env names exists in pancake's `src/` (grep).
  - Sources: [commit](https://github.com/android-imdad/Strata-7900xtx/commit/caa9b00115e7f134292cc5be971e35095934f587), [docs/RX7900XTX.md](https://github.com/android-imdad/Strata-7900xtx/blob/rx7900xtx/docs/RX7900XTX.md)
- eddoursul/Lamina `a28de1f6d4` (2026-10-18): QSA decode selection for 2-8 queries with the queries in registers, plus a multi-block top-k for up to 8 queries in one launch. Bitwise. Author on an RTX 3090: scores 4 queries 131K 29.4 -> 24.7 us; top-k 4 queries 131K 35.3 -> 18.6 us; end-to-end decode after a 131K prompt -2.1..-2.8% round time. Pancake's `qsa_select.cu` already has a recursive-halving scorer and the radix block_topk machinery (lines 66, 1330-1740, 2021), so this is probably upstream already. — [commit](https://github.com/eddoursul/Lamina/commit/a28de1f6d46f470fac24cca2a2267a5d74f8dc23)
- Ben1078/Strata `32cb881703` (2026-10-23): a gfx1100 port of upstream PR1825's long-prompt QSA block scorer, `STRATA_SCORES_THREAD=1`, exact. Prompt-only (>= 16,385 blocks, >= 64 queries). Pancake already has `STRATA_SCORES_THREAD` (in `src_envs`, 2 files). — [commit](https://github.com/Ben1078/Strata/commit/32cb8817035de8f7c36a42024449c54d3ccdbc9d)
- xyzzing/Strata `6d5ca79964` (2026-10-18): on gfx1100 the fp16 (rocWMMA) select arm is 1.64x the bf16 arm. 128K prefill select phase 1,934 ms vs 3,176 ms for upstream's bf16 arm. Pancake runs `STRATA_SELECT_WMMA=1`; whether that is the fp16 or the bf16 arm was not checked. — [commit](https://github.com/xyzzing/Strata/commit/6d5ca79964f97d2f5bc1ef1f68c24f4bb91bbc50)
- Asofwar/Strata-Turbo `f3cfbf04f2` (2026-10-20): `STRATA_ATTN_SPLITS=P`, split-K decode attention over a fixed partition count. It caps the per-chunk partials (67.6 MB per query at 262K for dense attention). Bitwise while a selection fits P x 64 cells, otherwise max rel diff 1.3e-6. CUDA only, speed not measured. — [commit](https://github.com/Asofwar/Strata-Turbo/commit/f3cfbf04f2aaf403ea69b12208d224a4e25fa781)
- UltimateBoomer/Strata `fe2c357174` (2026-10-28): on gfx1101 (RX 7800 XT class), Q2_0, the "gfx12 decode switches" `attn_lanecell, attn_merge_v2, tsum, expdec` plus the RDNA3 prompt set and pcie_frac 0.15 gave decode +9% (code) to +24% (28K context) and prompts +33..+44% (author). rxr2 on a 7900 XTX: "`STRATA_EXPERT_V2`, `STRATA_TSUM`, `STRATA_ATTN_LANECELL`+`MERGE_V2` ... single runs looked +2-7%, combinations did not hold above the decode noise". Pancake has `STRATA_ATTN_LANECELL` / `STRATA_ATTN_MERGE_V2` in src (4 files each) but not in the fast-hip preset. — [UltimateBoomer commit](https://github.com/UltimateBoomer/Strata/commit/fe2c357174e60675ed57867d59325b1d40e082c7), [rxr2 doc](https://github.com/rxr2/Strata/blob/rx7900/docs/RX7900XTX.md)

### Inferences
- Attention is a small share of our decode window. QSA kernels do not appear in the top of our profile: attn_chunk_kernel and attn_merge_kernel are each under 1%. Expected gain for this whole area: under 1% at short context, maybe 1-3% at 64K+ context.
- Cheapest check: `STRATA_ATTN_LANECELL=1 STRATA_ATTN_MERGE_V2=1` as an env A/B at 28K+ context, where UltimateBoomer saw the large gain. Exactness: unknown, so check the token hash.

### Gaps
- No per-switch numbers for TOPK_REGKEYS / TOPK_SCAN / GDN_NORM128. They would need the PR #2231 diff ported and measured.
- Whether pancake's SELECT_WMMA is the fp16 arm (xyzzing's 1.64x): not verified in code.

## 3. Router, shared expert and the MoE plan/combine path (fusions on the critical path to the CPU pool)

### Takeaway
The highest-value new items are two small, exact changes on the path between a layer's router and the CPU pool:
1. A faster router publish. android-imdad `STRATA_ROUTE_FAST`: DPP exchanges; the MoE input rows copied to the host by the router GEMV; the step number stored instead of read-modify-written over PCIe. Ben1078 has the same store-not-RMW doorbell.
2. A wider plan copy. android-imdad `STRATA_WAIT_COPY256`: 256 threads and 16-byte reads for the plan copy; host write -> plan on the GPU 4.47 -> 3.57 us.

Pancake already has the bigger upstream-absorbed fusions: `STRATA_ROUTE_PUBLISH` (top-10 + doorbell in one launch), `STRATA_WAIT_FUSE`, `STRATA_COMBINE_CPU` and `STRATA_QFUSE`.

### Cited Findings
- **android-imdad `caa9b00115` `STRATA_ROUTE_FAST` (default on in that fork, exact).** From the diff comments:
  - "the router's lane exchanges as DPP moves on wave32 RDNA (row_xmask for the masks 1-8, v_permlanex16 for 16) instead of ds_bpermute ... ~1 us less a launch."
  - "The MoE input rows go to the host in 16-byte stores: for 5-8 rows each token's warp copies its row right after its routing."
  - "The step number is stored (`seq_value`, the host's next one) rather than read over PCIe and incremented."
  - "x_out null: the router GEMV launch before this one already copied every row (bf16_gemv_fp32_mmvf_multi's x_pub)."
  - Source: [commit](https://github.com/android-imdad/Strata-7900xtx/commit/caa9b00115e7f134292cc5be971e35095934f587)
- **Ben1078/Strata `23d2fc0450` (2026-10-23, gfx10+ wave32, exact, measured on an RX 7900 XTX inside a combined round).**
  - Top-10 + doorbell in one launch (`STRATA_ROUTE_DOORBELL=0` to disable): "the ring stored as the step's number ... without the read across PCIe (STRATA_DOORBELL_RMW=1: the increment)".
  - CPU rows from mapped memory + hit add + combine in one launch (`STRATA_POST_FUSE=0`).
  - GDN qkv + z + a/b input projections in one launch (`gdn_in_proj_fused`, Q8_0 weights only).
  - QSA k, v, q rows in one grouped Q8_0 mmvq launch.
  - Round-1 result (Ben1078 doc): "38.5 vs ~35.7 tok/s together with --kv-resident and upstream env flags".
  - Sources: [commit](https://github.com/Ben1078/Strata/commit/23d2fc0450a642d984e64724b4925cabc202836c), [RDNA3_MOE_OFFLOAD.md](https://github.com/Ben1078/Strata/blob/rdna3-moe-offload-perf/RDNA3_MOE_OFFLOAD.md)
- **Pancake status, router publish.** `route_publish_res` (`src/kernels/cuda/native_router.cu:216`, body `rr_route_publish` in `route_row.cuh:69-92`) is one 256-thread block. It copies every MoE input row to mapped host memory when any expert misses, then does `*(volatile uint32_t*) seq = *(volatile uint32_t*) seq + 1u`, a PCIe read + write of host memory. It averages 17.8 us x 4,128 calls (2.68%) in our profile, and it sits right before the CPU pool can start a layer. No DPP in it, no stored step number, no copy in the router GEMV. (`STRATA_ROUTE_PUBLISH=1` is set in the fast-hip preset.)
- **android-imdad `caa9b00115` `STRATA_WAIT_COPY256` (default on there, exact).** "256 threads, so a verify plan (568 words, 142 16-byte reads) crosses PCIe in one round of reads instead of two - the same words (ping-pong bench: host write -> plan on the GPU 4.47 -> 3.57 us)". The fused wait+copy kernels also use 16-byte reads ("about one PCIe round trip instead of four or five"). Pancake: `wait_flag_copy_i32_kernel<<<1, 128>>>` with 4-byte reads (`src/kernels/cuda/verify_kernels.cu:1395-1404, 1777`). It is 4.07% of GPU time, though most of that is the wait itself. — [commit](https://github.com/android-imdad/Strata-7900xtx/commit/caa9b00115e7f134292cc5be971e35095934f587)
- **unimatrix099/Strata `3c8980f6a4` (2026-10-10), `STRATA_HIP_FRONT_FUSE=1`.** Top-10 + doorbell in one launch; the shared expert on QFUSE's q8_1 bytes without its own quantize and gate; gate + scale + post()'s head in one launch. FFN-half front 15 -> 11 kernels/layer. "decode +2.7% over six rounds, same text" on 2x RX 7900 XTX, IQ3_XXS (author, `FORK.md`). Pancake has the top-10 + doorbell part (ROUTE_PUBLISH) and QFUSE. The "shared expert without its own gate, applied with post()'s head" part matches this-cute-world's `STRATA_FUSE_SHGATE` (covered: batch3 F2). — [commit](https://github.com/unimatrix099/Strata/commit/3c8980f6a4396b1263a6f5e01e487942fae0d186), [FORK.md](https://github.com/unimatrix099/Strata/blob/main/FORK.md)
- **unimatrix099/Strata `c45a551c57` (2026-10-08), `STRATA_COMBINE_GATHER`.** CPU rows + GPU rows + combine in one kernel, 97 fewer kernels per window, 33.68-34.10 vs 34.00-34.53 ms per window (~1%) on 2x RX 7900 XTX. Its bench notes: "what a kernel costs in a graph here (2.86 us, graph_gap.hip), the kernel inventory of a window (2,435, two thirds under 8 us) ... estimated ceiling (~5%)". Pancake's `STRATA_COMBINE_CPU=1` + `STRATA_WAIT_FUSE=1` (combine reads the mapped rows and waits in-kernel, `native_moe.cu:175`, `verify.cpp:2705`) covers the same fusion. — [commit](https://github.com/unimatrix099/Strata/commit/c45a551c57a68a1f46d17fb6631dbeea3a5bc811)
- **unimatrix099 `5255b00e1a`.** "the router's top-10 timed alone (3.5 us of work per layer; left as is)". So most of our 17.8 us route_publish_res average is the copy and the PCIe round trip, not the top-10. — [commit](https://github.com/unimatrix099/Strata/commit/5255b00e1acf1d74079558b4954fa050a2023de6)
- **Jackwwg83/Strata-DS (DeepSeek engine, RTX 5090 Laptop), method only.**
  - `c0baf733fe` (2026-10-09): doorbell publish with one route use per lane instead of one thread walking chains of dependent loads, 8.8 -> 4.7 us per layer, outputs unchanged.
  - `780e043b08`: one-block rmsnorm holding x and w in registers (all loads issued at once), 4.1 -> 2.0 us at n = 5120, bitwise.
  - Sources: [c0baf73](https://github.com/Jackwwg83/Strata-DS/commit/c0baf733fe1c10ef7a6dce21e66b0cad7445a1e6), [780e043](https://github.com/Jackwwg83/Strata-DS/commit/780e043b08559d3eef14e8acf07e6ce91139b967)

### Inferences
- **Router publish.** Doing the store-not-RMW step number plus moving the row copy into the many-block router GEMV (and DPP in the top-10) could take route_publish_res from ~17.8 us to ~5-8 us. That is ~10 us x 48 MoE layers ≈ 0.4-0.5 ms per ~25 ms window, and it is on the CPU critical path, so a decode gain of +1..+2% is plausible. Exact. Effort S-M (route_row.cuh, native_bf16.cu x_pub, verify.cpp seq values). Check first that the host side tolerates a stored sequence value (Ben1078's comment: stage_inputs resets the ring to 0 before every window).
- **WAIT_COPY256 + 16-byte plan reads.** ~1 us per plan copy x 48-96 copies ≈ 0.05-0.1 ms per window: +0.2..+0.4%. Exact, effort XS.
- The GDN / QSA Q8_0 projection groups from Ben1078 do not apply directly. Our IQ2_XS file's dense projections are IQ4_XS (`rdna_kq_kernel<23>`), so a same-input group launch would need an IQ4_XS version of `rdna_kq_kernel`. this-cute-world measured FUSE_PROJ as small (~60 us per window, covered: batch3 F4).

### Gaps
- Ben1078 and unimatrix did not isolate the router/doorbell part of their fusions. android-imdad gives "~1 us less a launch" for DPP only. The 17.8 us -> 5-8 us estimate is mine, from unimatrix's 3.5 us top-10 timing. It is not measured.

## 4. Launch and graph overhead on gfx1100

### Takeaway
On gfx1100 a dependent kernel in a HIP graph costs ~2.8 us, and graph branches (fork/join) cost far more than they hide. A launch whose block count lands exactly on a "whole card" multiple (unimatrix: 384 x 256 = 48 WGPs x 8 blocks on the XTX) costs 14-16 us even when empty. Upstream fixed this only for gfx12 (`grid_pad`, PR #2492, not merged and absent from pancake). We already avoid the one known gfx1100 case (`--pcie-frac 0` skips fetch_blobs). A microbench for this "fill stall" on the 7900 XT (42 WGPs) is the cheap next step.

### Cited Findings
- **unimatrix099/Strata `d0c8d6c3e2` (2026-10-10), exact.** "a 384 x 256 launch - exactly 48 CUs x 8 blocks - costs 14-16 us even when *n is 0 ..., 352 blocks or fewer 2.7-3.0 us; a real fetch reads mapped host memory at 6.9 GB/s from 16 to 384 blocks alike". Fix: fetch_blobs at 96 blocks on HIP. "decode +6% over eight prompts, same text" (2x RX 7900 XTX, author). Microbench in the diff: `autoresearch/explore-261009-ceilings/mega/empty_grid.cpp`.
  - Pancake: `fetch_blobs_kernel<<<48 * 8, 256>>>` is unchanged (`src/kernels/cuda/verify_kernels.cu:762`). But our preset runs `--pcie-frac 0`, and `verify.cpp:2671` (`if (!pcie_off_)`) then skips the flag-B wait, fetch_blobs, rebase and the PCIe grouped launch. Already avoided for the daily preset; still live with `--pcie-frac > 0`.
  - Source: [commit](https://github.com/unimatrix099/Strata/commit/d0c8d6c3e2c1607447ff342b2aa1fd59c385b1b1)
- **mandoof1/Strata `2b46e4b13d` (2026-10-16), upstream PR #2492 `grid_pad`.** gfx12 only, `STRATA_GRID_PAD=0/1`; "=1 applies it on any HIP card". The diff text: "a launch whose waves land at, or just below, a whole multiple of the card's wave slots (WGPs x 64 ...) runs ~8-14 us longer than one a few waves bigger (ROCm legacy-rocm-build#6689 ...: 125-128 blocks of 512 threads 16.4 us, 124 or 129 blocks 2.2 us)".
  - Decode matvecs on R9700: attn_gate 2560->6144 at 2-4 columns 41-44 -> 33 us; ssm_out/attn_output 41 -> 31 us; the 512-row projections 16.6 -> 5.7 us. Bitwise (empty blocks return at once).
  - Pancake: no `grid_pad.hpp`, no `STRATA_GRID_PAD`, and the PR is not in upstream/main (git log).
  - unimatrix's 384 x 256 = 3,072 waves = 48 WGPs x 64 on the XTX is the same rule seen on gfx1100.
  - Source: [commit](https://github.com/mandoof1/Strata/commit/2b46e4b13d80fa4dc27893a36a63d81e67eedb86)
- **unimatrix099/Strata `e320b78a73` (2026-10-10), `STRATA_HIP_POST_SPLIT=1`, exact.** post()'s small kernels run as two persistent launches around the three expert kernels, with the kernel bodies as virtual blocks. "decode +3.5% over eight prompts". "The expert bodies run 2x slower looped inside one persistent launch, so they stay kernels." The path needs `sink_.pcie_mode == 2` (the PCIe share). `a332ef8a2d` clamps the grid to resident blocks, which avoids a hang on small GPUs. — [e320b78](https://github.com/unimatrix099/Strata/commit/e320b78a73cd98d1f95e7172f3f0144a75c786d7), [a332ef8](https://github.com/unimatrix099/Strata/commit/a332ef8a2d7b5c120e7fcc7430b50c0bb9ea8001)
- **unimatrix099 research commits.**
  - `29fa5095a9`: "launch gap 4.2 us vs grid barrier 0.4 us, 2-MB kernels 2x faster as phases, resident-grid limit 384; graph branches cost 26-30% in the engine (the launch blocks the host 1.5 ms)".
  - `28573a030f`: "the floor of a persistent FFN half on real weights - 67 us per layer against 150 for today's 23 kernels (reads at 830-870 GB/s, serial steps ~24 us); estimate ... decode +10-14%".
  - Sources: [29fa509](https://github.com/unimatrix099/Strata/commit/29fa5095a93c3ffe3c93d2998d58e16094921346), [28573a0](https://github.com/unimatrix099/Strata/commit/28573a030fee66ae827a8ac93cc64361eee4d43b)
- **android-imdad `docs/RX7900XTX.md` "Tried on this card and not kept".** "A dependent kernel in a HIP graph costs 2.8 us here, and raw AQL packets with the cache fence the kernel needs cost the same ... Without the queue's acquire fence a kernel costs 2.2 us and stays correct only if it invalidates its own caches first. HIP has no switch for that fence." Also: prefetching the next layer's experts was 3-4% slower, and "every 1.9 MB copy during a window adds about 14 us to the GPU's path". — [doc](https://github.com/android-imdad/Strata-7900xtx/blob/rx7900xtx/docs/RX7900XTX.md)
- **Ben1078 RDNA3_MOE_OFFLOAD.md.**
  - "Device-side PCIe 'gap fetch' of predicted experts (graph branch) 47.9 -> 37.0 tok/s (-23%): a per-layer cross-stream fork/join inside a HIP graph costs more than it hides".
  - "GPU hyper-connection kernel fusion (one launch per read, in-kernel role sync) worked after fixing a gfx1100 GL1 stale-read hang, but per read slower".
  - Source: [doc](https://github.com/Ben1078/Strata/blob/rdna3-moe-offload-perf/RDNA3_MOE_OFFLOAD.md)
- **xyzzing/Strata `320e4324a8` (2026-10-18).** On gfx1100, `STRATA_SH_STREAM` (shared expert on its own stream) gave "+4.0%, upstream ships it on gfx1151 and never checked gfx1100". With `--spec 5` the composed gain was +11.3% (author, IQ3_S, 1K context).
  - Pancake's fast-hip preset sets `STRATA_SH_STREAM=0`, and I found no recorded A/B in MOD_PLAN / findings.
  - Upstream `docs/AMD_HIP.md:270-291` says it "goes both ways by PC" (it needs CPU-expert time to hide behind).
  - unimatrix's graph-branch result above is a caution: SH_STREAM is also a fork/join.
  - Source: [commit](https://github.com/xyzzing/Strata/commit/320e4324a86e1c2e3bbe7fe4f5b586ac878ea446)
- **MULTI_BLOCK_ARGMAX on gfx1100 (android-imdad `caa9b00115`).** "STRATA_MULTI_BLOCK_ARGMAX ... On HIP (wave32, gfx1100) too: 12 -> 5.6 us a pick, 12.3 -> 9.4 us a top probability". This needed loads issued 4 / 16 at a time ("one at a time the 1,024 sums' 243 loads each are a chain of memory round trips (31 us a call ...)"). Pancake's `multi_block_head_ops()` (`verify_kernels.cu:1025-1046`) returns true on HIP only for gfx12, and `STRATA_MULTI_BLOCK_ARGMAX=1` cannot force it on gfx1100. — [commit](https://github.com/android-imdad/Strata-7900xtx/commit/caa9b00115e7f134292cc5be971e35095934f587)

### Inferences
- **Fill-stall audit, effort S, potentially the biggest kernel win here.** Run unimatrix's `empty_grid.cpp` (blocks 1..400 x 256) and a 512-thread sweep on the 7900 XT when the GPU is free. Then compare the block counts of our decode launches (`STRATA_VERIFY_NODES` lists the graph nodes) with the stall points. On our card that is 42 WGPs x 64 = 2,688 wave slots if the gfx12 rule carries over, e.g. 336 blocks of 256 threads.
  - If any hot launch sits on a stall point, padding is exact and worth 8-14 us per launch per layer.
  - The high spread in our profile (`rdna_kq_kernel<23,4,3>`: avg 18 us, min 3 us, max 805 us) is consistent with a shape-dependent stall but does not prove one.
  - Gain: 0 to several %, unknown until measured.
- **POST_SPLIT / persistent FFN half.** Most of its gain in unimatrix's setup came from the PCIe tail, which pancake skips (`pcie_off_`). What is left for us is the head (wait A + plan copy) and the tail (wait C + combine), which pancake already fuses partly (WAIT_FUSE, COMBINE_CPU). MOD_PLAN parks the "persistent post-bell kernel" because CPU experts gate the window. Keep it parked.
- **SH_STREAM=1** is a free env A/B, exact. Expected -2..+4%: it needs CPU-expert time to hide behind, and our windows have plenty.
- **MULTI_BLOCK_ARGMAX on gfx1100** needs a one-line arch-gate change plus android-imdad's batched-load variant. About 10-20 us per window, +0.05..+0.1%. Low priority.

### Gaps
- Whether the gfx12 wave-slot stall rule holds on gfx1100 in general: unimatrix saw it for one launch shape, and nobody swept other shapes on gfx11.
- HIP's `multiProcessorCount` on gfx1100 (CUs 84 vs WGPs 42) decides the slot formula; not checked on our card.

## 5. Prefill GEMM and prompt-path kernels on gfx1100

### Takeaway
Pancake already runs the lossy fast prompt path (PF_FUSED, PF_F16=2, HIP_WMMA, PF_GEMM, GDN_CHUNKED, hipBLASLt table). Two 7900 XTX forks show this is where the big prompt gains are (+59..+82% PF_FUSED + HIP_WMMA). The new gfx1100-native items (android-imdad's Tensile GU_EXACT, PF_DOWN_STREAMS, PF_PIECE, PF_DQ_OVERLAP) are for an exact, rocBLAS-order prompt path, and pancake's prompt path is not exact anyway. Only PF_PIECE and PF_DQ_OVERLAP carry over in spirit.

### Cited Findings
- **android-imdad/Strata-7900xtx `9caad8f2bb` (2026-10-13), `STRATA_PF_GU_EXACT=1` (+ `STRATA_PF_GU_TENSILE=0` to disable the Tensile kernel), exact.**
  - A Tensile-generated gfx1100 kernel (`gu_tensile_gfx1100.s`, hipModuleLoadData) with rocBLAS's order parameters pinned: 26.5 TFLOPS at 512 tokens, 28.7 at 1024, vs rocBLAS's fallback 12.9 / 13.1. One launch per expert, round-robin over 8 streams: 24 TFLOPS on a 16-32 expert group vs 9.5.
  - Prompt reads (author, RX 7900 XTX): 16K 13.05 -> 10.05-10.61 s, 64K 40.08 -> 31.52 s. KL 0.0000.
  - "That is why `STRATA_PF_GEMM` moves perplexity from 4.447 to 4.508" (android-imdad doc, on the lossy alternative pancake uses).
  - Sources: [commit](https://github.com/android-imdad/Strata-7900xtx/commit/9caad8f2bba29bf5cfb9a247bc76a8f51a2e2815), [doc part 2](https://github.com/android-imdad/Strata-7900xtx/blob/rx7900xtx/docs/RX7900XTX.md)
- **android-imdad `876c3af7e6` / doc part 7, `STRATA_PF_DOWN_STREAMS=4`.** The down products (and SwiGLU) of a 16-expert group spread over 4 streams: "A down product of ~100 tokens fills only ~27 of the card's workgroup slots." Fresh prompts 2K -3..-4%, 16K / 32K -2% read time. "Do not raise GPU_MAX_HW_QUEUES to 8 on this card: every prompt got 10-20% slower." (Pancake runs `GPU_MAX_HW_QUEUES=6`.) — [commit](https://github.com/android-imdad/Strata-7900xtx/commit/876c3af7e68cc1b491096224da2f07f5fd9acbbc), [doc](https://github.com/android-imdad/Strata-7900xtx/blob/rx7900xtx/docs/RX7900XTX.md)
- **android-imdad doc part 5, `STRATA_PF_PIECE=8192` + `--prefill auto:32768`.** Read in 32K chunks but computed with the bits of 8K chunks: "a fresh 32K prompt +13-16%, long documents +10-12%" (exact vs 8K chunks).
  - Pancake already uses 32K chunks (`STRATA_PREFILL_AUTO_MAX=32768`) and does not need 8K-chunk bit-identity.
  - Contrary datum: rxr2 on its 7900 XTX: `--prefill auto:24576` instead of `auto:32768` gave "prefill +15% at 32K, +7% at 64K", because "a 32K chunk needs such large buffers that the engine lends too many expert-cache slots; the experts evicted from them have no RAM copy and are read from the drive again (~10.5 GB per 32K chunk)".
  - Sources: [android-imdad doc](https://github.com/android-imdad/Strata-7900xtx/blob/rx7900xtx/docs/RX7900XTX.md), [rxr2 doc](https://github.com/rxr2/Strata/blob/rx7900/docs/RX7900XTX.md)
- **android-imdad doc part 9, `STRATA_PF_DQ_OVERLAP=1` (default on there).** A group's experts are dequantized on their own stream into one of two buffer sets while the compute stream runs the previous group. "fresh IQ3_S prompts read 7-11% faster (4K 1,174 -> 1,302, 16K 1,599 -> 1,712 tokens/s)" with 8K chunks and no DOWN_STREAMS; no gain with 32K chunks + DOWN_STREAMS. Exact. Absent from pancake. — [doc](https://github.com/android-imdad/Strata-7900xtx/blob/rx7900xtx/docs/RX7900XTX.md)
- **rxr2 `docs/RX7900XTX.md`, 7900 XTX measured env table (author).**
  - PF_FUSED + HIP_WMMA: prefill +59-82% (experts 3.5x).
  - GDN_CHUNKED: +6-7%.
  - PF_F16 + PA_FAST + HC_UPMIX: +7-10%.
  - PF_GEMM + PF_PAD: +2.5-3.5%.
  - `EXP_MODE=8 HC_Q8=1`: decode +9-10% (HC_Q8 is rounding; our own A/B found upstream HC_Q8 worse, 107.2 vs 111.7, MOD_PLAN).
  - `STRATA_PF_HCDOWN=1`: +1.3% prefill, but "first-token KL at 16K doubled": rxr2 did not use it. Pancake's preset sets PF_HCDOWN=1.
  - "Prefill is GPU-compute bound (GPU busy ~89%) ... expert GEMMs 25%, GDN 11%, QSA select 11%, attention 10%, hyper-connections ~14%, waiting for expert copies 5%".
  - Source: [doc](https://github.com/rxr2/Strata/blob/rx7900/docs/RX7900XTX.md)
- **troytse/Strata `c4ddf6eba0` (2026-10-14), `STRATA_PF_KERNEL_INFO`, method.** Prints VGPR / LDS / occupancy of the prompt expert kernels. On gfx11: "Q4_K gate/up 256 VGPRs (the RDNA3 maximum), LDS 41472 B ... pinned to one block per WGP". An ALIAS variant fitting two blocks was slower (2572 vs 2481 ms) because "the extra barrier costs more than the occupancy returns". — [commit](https://github.com/troytse/Strata/commit/c4ddf6eba0800a3d04c00e15e64f7aaf16c02252)
- **zsariboga/Strata `21107e9d05` (2026-10-08), RTX 5090, CUDA-only fusions, same bits.**
  - dequant_h16x8 (16-byte stores) 141.1 -> 11.2 ms per 32K prompt.
  - swiglu + q8_1 quantize in one pass 43.3 -> 19.6 ms (pancake already has `mmq::swiglu_quant`, `prefill.cpp:6145`).
  - moe_combine + gr_write_norm_rs fused 234.4 -> 203.0 ms (`STRATA_PREFILL_COMBINE_WRITE`, absent from pancake).
  - 32K prompt -4.0% overall.
  - android-imdad found the wide dense dequant slower on gfx1100 (above), so only the combine+write_norm fusion is a candidate.
  - Source: [commit](https://github.com/zsariboga/Strata/commit/21107e9d05364385527485fa6294dc5db4903976)
- **eddoursul/Lamina `bedc0c6c8a` (2026-09-28), bitwise.** QSA block scores for prompt batches in a per-thread GEMM form: 256-query batch at 131K 1,734 -> 892 us; prompt 131K -4.6% (2x RTX 3090). Probably upstream already (pancake's qsa_select has `SELECT_TWO_PASS`, `SELECT_SIMT`, `SELECT_WMMA`). — [commit](https://github.com/eddoursul/Lamina/commit/bedc0c6c8a853b77a7a0309974283e2777eda01e)
- **yqYo1/Strata `404d07e710` (2026-10-08).** GLM prompt projections through a tuned hipBLASLt table on the RX 7900 XT: ~198 -> ~413 tok/s. Pancake already ships and uses `tools/hip/gfx1100-hipblaslt-100401.txt` (`STRATA_HIPBLASLT_TUNING` in the preset). Note the tables are hipBLASLt-version-specific (htbmixbox `b45fc5d405`: "It only matches ROCm 7.14.1"). Our box runs ROCm 10.0.0. — [404d07e](https://github.com/yqYo1/Strata/commit/404d07e71081cf662efded7b0f7fba58a95a628a)

### Inferences
- For pancake's lossy prompt path the remaining exact candidates are:
  - (a) `PF_DQ_OVERLAP`-style double-buffered dequant on a side stream, only if our PF_F16=2 path still dequantizes into a buffer before the GEMM (not checked).
  - (b) rxr2's chunk-size finding. An env A/B of `STRATA_PREFILL_AUTO_MAX=24576` vs 32768 on the 41.8K prompt is cheap. Our RAM is 57 GB vs rxr2's 64 GB, so the lend/evict/disk-reread effect could be larger here.
- Recheck `STRATA_PF_HCDOWN=1` in our preset against rxr2's KL warning (first-token KL doubled at 16K). It is a quality question, not speed.

### Gaps
- Whether pancake's hipBLASLt table matches the hipBLASLt shipped with ROCm 10.0.0 (the engine logs a mismatch if not). Not checked from the logs in this session.

## 6. Memory copies (H2D experts, mapped-memory reads, DMA engines, adaptive swaps)

### Takeaway
Most copy-path wins in the AMD forks come from the PCIe share (`--pcie-frac > 0`), which our preset turns off. The transferable pieces:
- Two-DMA-engine adaptive swap copies (`STRATA_ADAPT_COPY2`, +8% copy bandwidth).
- 16-byte reads of mapped host memory (section 3).
- mlock of the ROCm-mapped arena to avoid HMM userptr invalidation stalls.
- O_DIRECT file-tier reads against page-cache reclaim hangs.

### Cited Findings
- **android-imdad `caa9b00115` `STRATA_ADAPT_COPY2` (on there, exact).** "a round's copies alternate over two streams - the card's two DMA engines ...: 48 copies of 2.2 MB in 3.70 instead of 4.00 ms (28.6 vs 26.4 GB/s; one or two copies are no faster)". Their `STRATA_ADAPT_LAYERWAIT=1` (copies under the window's first layers) measured "20.37 vs 20.37 ms a window: off". Pancake runs `--adapt-every 1 --adapt-swaps 32` with `STRATA_ADAPT_OVERLAP=1` (upstream's per-layer wait). ADAPT_COPY2 is absent. — [commit](https://github.com/android-imdad/Strata-7900xtx/commit/caa9b00115e7f134292cc5be971e35095934f587)
- **android-imdad doc part 7 / `570854aded` (2026-10-24), `STRATA_PCIE_USAGE=1` + `STRATA_PCIE_KEEP=96`.** The experts the adaptive tier is about to swap in are sent over PCIe in the window before, then kept in a 96-blob VRAM ring (~130 MB). "About 31 of ~50 swaps per window now come from the ring, and the wait for the copies fell from 1.2-1.7 ms to 0.2-0.5 ms a window ... 106.15 -> 110.48 tokens/s (+4.1%)" (RX 7900 XTX, drafts held fixed). It needs the PCIe share (`--pcie-frac 0.35` there); our preset has 0. — [commit](https://github.com/android-imdad/Strata-7900xtx/commit/570854aded26742f7ddcc676357e5a19356d4c85), [doc](https://github.com/android-imdad/Strata-7900xtx/blob/rx7900xtx/docs/RX7900XTX.md)
- **android-imdad `caa9b00115` `STRATA_DMA_FLAG_WRITE` (HIP).** "PCIe share's flag written by the copy stream (hipStreamWriteValue32) instead of a host function". This is again only on the PCIe-share path. — [commit](https://github.com/android-imdad/Strata-7900xtx/commit/caa9b00115e7f134292cc5be971e35095934f587)
- **Ben1078/Strata `bd0e37bb4b` (2026-10-23).** "On ROCm, hipHostMalloc/hipHostRegister map host memory for the GPU through HMM (a KFD userptr) without pinning the pages. Under memory pressure (zram with swappiness > 100 in particular) the kernel swaps parts of the resident expert arena out, and every invalidation stops the GPU queues while amdgpu's restore_userptr worker re-walks the whole range." Fix: mlock the arena after mapping. Pancake has an mlock facility (`src/platform/memory.cpp:167-178`). Whether it covers the hipHostRegister'd expert arena on our box was not checked. — [commit](https://github.com/Ben1078/Strata/commit/bd0e37bb4bbfdaed547d9fd9045fa15a7b228335)
- **rxr2 `docs/RX7900XTX.md`.** `STRATA_UNBUFFERED_LOAD=1` (O_DIRECT file tier) "removes the hangs: with 64 GB RAM, page-cache reads triggered memory reclaim and `svm_range_restore_work` stalls of the GPU queues, 5 of 11 long prompts hung, 0 of 16 after". Pancake has `STRATA_UNBUFFERED_LOAD` in src (`expert_source.cpp`, `generate.cpp`). `STRATA_PCIE_BALANCE=1`: "+1-2% (noise); ... a PCIe expert costs ~85 us serially, the 7-thread pool does ~3 experts in ~0.13 ms". — [doc](https://github.com/rxr2/Strata/blob/rx7900/docs/RX7900XTX.md)
- **Ben1078 RDNA3_MOE_OFFLOAD.md (7900 XTX), CPU-side but it feeds the GPU's wait.** "Helper threads off CPU 0 (`STRATA_HELPER_CPUS=auto`) 47.5 -> 49.3 tok/s (+3.8%), same binary; GPU wait for the host's plan 1.52 -> 0.51 ms/window" (bitwise). Pancake has no `STRATA_HELPER_CPUS`; it has `--pool-affinity measure` and `STRATA_POOL_CORES`. "CPU writes its expert rows straight into VRAM through the BAR: estimated <= +0.5-0.8%; test run failed; dropped." — [doc](https://github.com/Ben1078/Strata/blob/rdna3-moe-offload-perf/RDNA3_MOE_OFFLOAD.md)
- **maoyufeng1985/Strata `e8ac049e87` (2026-10-29), P100.** Dequant with one 8-byte store per group instead of narrow stores: Q4_K gate/up dequant per expert 70.1 -> 34.7 us, i-quants 1.8-2.3x, byte-identical. HIP not compiled. On gfx1100 the dense dequant is already ~1.6 TB/s (android-imdad), so the expert FP16 dequant is the only possible target. — [commit](https://github.com/maoyufeng1985/Strata/commit/e8ac049e87f15fe43adb8ede25d6fc70c9595436)

### Inferences
- **ADAPT_COPY2:** if our swap copies are waited for at all (ADAPT_OVERLAP hides most of them), +8% copy bandwidth on ~32 swaps gives at most ~0.1-0.2 ms per window: 0..+0.5% decode. Exact, effort S.
- **Helper threads off CPU 0:** the 1.52 -> 0.51 ms plan wait shows how much the doorbell -> plan latency can matter on this platform. A quick check of where pancake's publishing/helper threads run, compared with `--pool-affinity measure`, is worthwhile even though it is CPU-side.

### Gaps
- Pancake's swap-copy wait per window (`STRATA_DECODE_TIMING`) and the mlock state of the expert arena on our box: not measured or read in this session.

## 7. Ranking: what to try first on pancake (gfx1100), with expected gain

### Takeaway
Ranked by expected decode gain per hour of work:
1. Router publish fast path (+1..+2%).
2. Grid-size "fill stall" microbench and audit (0..several %, unknown).
3. IQ2 sign-expansion trick in expdec gate/up (0..+1.5%).
4. Free env A/Bs: `STRATA_SH_STREAM=1`, `STRATA_ATTN_LANECELL=1 STRATA_ATTN_MERGE_V2=1`, `STRATA_PREFILL_AUTO_MAX=24576` (prefill).
5. WAIT_COPY256 + 16-byte plan reads (+0.2..+0.4%).
6. ADAPT_COPY2 (0..+0.5%).

All are exact except possibly the LANECELL/MERGE_V2 pair. None needs a second GPU.

### Cited Findings
- Router publish: android-imdad `caa9b00115` `STRATA_ROUTE_FAST` and Ben1078 `23d2fc0450` stored doorbell value. Our route_publish_res is 17.8 us avg (2.68%) vs unimatrix's 3.5 us top-10 work (`5255b00e1a`). — [caa9b00](https://github.com/android-imdad/Strata-7900xtx/commit/caa9b00115e7f134292cc5be971e35095934f587), [23d2fc0](https://github.com/Ben1078/Strata/commit/23d2fc0450a642d984e64724b4925cabc202836c), [5255b00](https://github.com/unimatrix099/Strata/commit/5255b00e1acf1d74079558b4954fa050a2023de6)
- Fill stall: unimatrix `d0c8d6c3e2` (gfx1100, 14-16 us for an empty 384 x 256 launch, +6% decode); upstream PR #2492 `grid_pad` via mandoof1 `2b46e4b13d` (gfx12, 8-14 us). — [d0c8d6c](https://github.com/unimatrix099/Strata/commit/d0c8d6c3e2c1607447ff342b2aa1fd59c385b1b1), [2b46e4b](https://github.com/mandoof1/Strata/commit/2b46e4b13d80fa4dc27893a36a63d81e67eedb86)
- Sign trick: yqYo1 `6f0bdf665b` (7900 XT, -26..-35% per kernel) and RX5950XT `b9a2111d3f` (drop-in for iq_kernels.cu). — [6f0bdf6](https://github.com/yqYo1/Strata/commit/6f0bdf665b4c01a24d086210a2506beae3f87b56), [b9a2111](https://github.com/RX5950XT/Strata/commit/b9a2111d3f4ece53487ce054e1b23aa3a8939214)
- SH_STREAM on gfx1100 +4.0%: xyzzing `320e4324a8`. — [commit](https://github.com/xyzzing/Strata/commit/320e4324a86e1c2e3bbe7fe4f5b586ac878ea446)
- Chunk 24576: rxr2 doc. — [doc](https://github.com/rxr2/Strata/blob/rx7900/docs/RX7900XTX.md)

### Inferences
- Ceiling for this whole area: the kernel-side ideas together probably add +2..+5% decode on our box, not +10%. The reasons:
  - The GPU is idle in combine_cpu_k10_vec4 for 38% of its kernel time.
  - rxr2's static audit puts mmvq at 70-100% of VRAM bandwidth.
  - unimatrix estimated the fusion ceiling at ~5%.
- The two 7900 XTX forks with +18% / +21% decode vs upstream got most of it from things pancake already has or does differently: drafting (SPEC_CUM, spec 6), CPU parity, placement, helper threads, kv-resident.
- Already-covered high-value items from earlier reviews that this survey re-confirms: this-cute-world STRATA_EXPDEC_V2 (batch3 F1), which targets the same expdec_gu<22> / Q2_0 down kernels as the sign trick. Port and measure the two together, since both change the same kernels.

### Gaps
- Every number is author-measured on other cards (mostly 7900 XTX, 96 CUs / 24 GB, IQ3_S or IQ3_XXS) or in other engines (GLM, DeepSeek, poly). None of it is measured on our 7900 XT with IQ2_XS experts, 84 CUs and 57 GB DDR4. No build or benchmark was run in this session: the GPU was busy, as instructed.
