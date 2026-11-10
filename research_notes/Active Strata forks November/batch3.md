# Active Strata forks, batch 3 (state 2026-11-10)

Scope: 22 forks with their own commits since 2026-11-01 (batch file batch3.md). Target: pancake-strata on RX 7900 XT (gfx1100) + Ryzen 7 5700G, Flash-Next IQ2_XS, preset fast-hip, --pcie-frac 0, MTP.
Three forks touch the engine in ways that matter: this-cute-world (a lot of HIP decode work, measured on RDNA4 R9700), samnolak (CPU / multi-GPU / SSD), atmJava (prefill fix, concurrent prompt reads). Zutfen-LLC is a large Vulkan "native" port that does not apply to us. All the others are docs, packaging, GUI or personal config.

Our own reference point, from bench/results/2026-11-09-7900xt-gpuprof/run_kernel_stats.csv (local file, our rocprofv3 run). Share of GPU kernel time:
- combine_cpu_k10_vec4: 38.3%. This is the GPU waiting for CPU expert rows.
- rdna_kq_kernel<23,...> (IQ4_XS dense): ~11%
- expdec_gu_kernel<22,1,4> (IQ2_S gate/up): 5.83%, 53.4 us avg
- native_down_multi_kernel<42> (Q2_0 down): 5.79%, 37.6 us avg
- bf16_f32_mmvf_multi_kernel<256,4,true> (router + aux): 5.07%
- wait_flag_copy_i32_kernel: 4.07%
- expdec_gu_kernel<16> (IQ2_XXS): 1.60%
- swiglu_q8_1_entries_kernel: 0.29%, 4224 calls

So decode on this box is mostly CPU-bound. GPU-kernel savings shorten a window only where the GPU path is on the critical path (see Inferences).

## Q1. Findings that could raise decode / prompt speed or serve as A/B options for pancake

### Takeaway
The most useful item is this-cute-world's **STRATA_EXPDEC_V2**. It is bitwise exact, AMD-only, and covers exactly our VRAM expert types: IQ2_S (22) and IQ3_S (21) gate/up, plus Q2_0 (42) and IQ4_NL down. Those kernels are ~13% of our GPU kernel time, and the author measured a 20-25% per-kernel cut on RDNA4. The second is a set of cheap, exact launch-count fusions (FUSE_FETCH / COMBINE / SWIGLU_DOWN / SHGATE / PLAN / ROUTEBELL / QGEMV) plus STRATA_MMVF_ROWS for a single token, and turning on the STRATA_MMVF_ROWS that already sits in our tree but is unused on gfx1100. samnolak's AVX2 multi-token CPU rows do not apply to IQ2_XS (only Q6_K / Q8_0).

### Cited Findings

**F1. STRATA_EXPDEC_V2: faster grouped expert kernels on AMD (this-cute-world) - NEW, exact, port M**
- Commits: a67b3790d5 (IQ2_S / IQ3_XXS gate/up + IQ4_NL down), 522f76739a (IQ3_S gate/up), 92983a7405 (Q2_0 down, +153 lines in iq_kernels.cu). Opt-in, AMD only, separate kernels. The V1 kernels are not touched. — [a67b379](https://github.com/this-cute-world/Strata/commit/a67b3790d5), [522f767](https://github.com/this-cute-world/Strata/commit/522f76739a), [92983a7](https://github.com/this-cute-world/Strata/commit/92983a7405)
- How it works:
  - Weights are read through global pointers. Read through grp_ptr they compiled to flat loads, and flat loads count in the LDS counter (dscnt), so every LDS wait also waited for the DRAM reads in flight.
  - All raw weight words are issued first, then the L2 staging loads, then one wait.
  - IQ2_S / IQ3_XXS signs are expanded in registers. The emulation this replaces cost ~170 VALU per call.
  - Scale products use 24-bit multiplies (exact).
  - gate/up uses rb 16. — [a67b379](https://github.com/this-cute-world/Strata/commit/a67b3790d5)
- Author-measured, R9700 (gfx1201), cold, rocprofv3, 25 groups:
  - T=4: expdec_gu<22> 69.1 -> 52.5 us (floor 43.0); expdec_gu<18> 62.7 -> 58.6 us; native_down_multi<20> 48.2 -> 38.5 us
  - T=2: expdec_gu<22> 51.3 -> 39.5 us; down 38.0 -> 29.5 us
  - native_grouped_parity: 672 calls each, bitwise equal. — [a67b379](https://github.com/this-cute-world/Strata/commit/a67b3790d5)
- Applicability: our profile's main VRAM expert kernels are exactly expdec_gu<22> (5.83%) and native_down_multi<42> (5.79%), and V2 has a Q2_0 (42) down. RDNA3 also counts flat loads against both vmcnt and lgkmcnt, so the flat-to-global fix should carry over to gfx1100 (inference, not measured on gfx11). The author did not measure IQ2_XXS (type 16, our 1.6% kernel); it is not listed among the V2 types.
- Suggested A/B: port the three commits behind STRATA_EXPDEC_V2=1, run expdec_bench and native_grouped_parity on gfx1100, then short chat / 2K code edit / 11.7K prompt decode tok/s against fast-hip. Token forcing is not needed if the parity check is bitwise.

**F2. Fewer launches around the MoE layer (this-cute-world) - NEW, exact, port S-M each**
- **STRATA_FUSE_FETCH / STRATA_FUSE_COMBINE** (ed87ad46af): the PCIe share's wait + fetch + rebase becomes one launch, and so does the CPU share's wait + copy + combine (combine_k10_vec4_hits_mapped reads the pool's mapped rows itself). Author-measured on R9700: 2504 -> 2361 kernels per window (-143), parity bitwise apart from NaN payloads. — [ed87ad4](https://github.com/this-cute-world/Strata/commit/ed87ad46af)
- **STRATA_FUSE_SWIGLU_DOWN** (7d1a446927, d33b540ddb): the down kernel (native_down_multi<20> / expdec_down) stages SwiGLU + q8_1 itself, so swiglu_q8_1_entries is gone. Author-measured: 2506 -> 2266 kernels per window together with FETCH+COMBINE. — [7d1a446](https://github.com/this-cute-world/Strata/commit/7d1a446927), [d33b540](https://github.com/this-cute-world/Strata/commit/d33b540ddb)
- **STRATA_FUSE_SWIGLU_GU** (76142ece0d): the gate/up epilogue makes h's q8_1 blocks through per-chunk counters. Author-measured: expdec_gu kernel time +0.21 ms per window against swiglu -0.137 ms, i.e. **net slower** in their trace, and --bench is flat (138.11 / 137.84 us). — [76142ec](https://github.com/this-cute-world/Strata/commit/76142ece0d). Our tree already has the similar STRATA_EXPDEC_SWIGLU (MOD_PLAN item 48, iq_kernels.cu:2639) = ALREADY-HAVE.
- **STRATA_FUSE_SHGATE** (93c97cc230): the shared expert's gate row is computed in the router GEMV (aux row), and its sigmoid is applied in the fused combine. Author-measured: -289 kernels per window with the set above. — [93c97cc](https://github.com/this-cute-world/Strata/commit/93c97cc230)
- **STRATA_FUSE_PLAN** (bdc2369a28): with the device plan, the wait for the host plan and the plan copy run in one kernel. Author-measured: -48 kernels per window; the full set gives 2648 -> 2248 (-400). — [bdc2369](https://github.com/this-cute-world/Strata/commit/bdc2369a28)
- **STRATA_FUSE_ROUTEBELL** (2693040f8d): top-10 + doorbell publish in one kernel, saving one launch per layer, bitwise (24 cases). — [2693040](https://github.com/this-cute-world/Strata/commit/2693040f8d). **STRATA_FUSE_SHBEL** (1c0772a666) also puts the shared gate/up pair into that launch. — [1c0772a](https://github.com/this-cute-world/Strata/commit/1c0772a666)
- **STRATA_FUSE_QGEMV** (818a1eaa57): router bf16 GEMV + MoE-input q8_1 quantize in one launch, bitwise. It requires STRATA_MMVF_ROWS=1 and refuses when STRATA_TSUM is on. — [818a1ea](https://github.com/this-cute-world/Strata/commit/818a1eaa57)
- What our tree has: none of these env names exist in src/ or include/. We do have STRATA_WAIT_FUSE, STRATA_COMBINE_CPU, STRATA_ROUTE_PUBLISH, STRATA_LFUSE and native_mmvq_pair (shared expert gate/up), which cover parts of the same ground.
- Caveat: the author reports kernel counts, not end-to-end tok/s, for most of these. Per-window ms is given only for DRAFTHEAD (batch mode). On our box, at ~2-6 us of launch overhead per kernel, -300 launches would be about 1 ms per window, but only where the GPU stream is the critical path.

**F3. STRATA_MMVF_ROWS for one token + STRATA_MMVF_ROWS / DENSE_LDS defaults (this-cute-world) - partly NEW**
- aa2509dc60: a one-token router GEMV also takes the 4-rows-per-block kernel. Author-measured on R9700: the 2560x512 router went from 25.8 us (~100 GB/s) to 10.2 us, bitwise. — [aa2509d](https://github.com/this-cute-world/Strata/commit/aa2509dc60)
- e3ab06a7bd makes STRATA_DENSE_LDS and STRATA_MMVF_ROWS the defaults on gfx12. Author-measured on 2x R9700, Flash-Next IQ3_S:
  - these defaults alone: +3.9% combined
  - with --mtp-q4 and adapt 32/32: 163.9 -> 178.7 tok/s (+9%)
  - --batch 4: -2 to -4%. — [e3ab06a](https://github.com/this-cute-world/Strata/commit/e3ab06a7bd)
- Our tree has STRATA_MMVF_ROWS (native_bf16.cu:265, 2-8 rows), but it is a default only in the gfx1151 table (arch_defaults.cpp:80). It is not in the fast-hip preset env (pancake-flashnext-iq2_xs-fast-hip.json), and the one-token path still calls the one-row kernel (native_bf16.cu:540, `if (n_tok == 1 ...) bf16_gemv_fp32_mmvf`). Our profile shows bf16_f32_mmvf_multi_kernel<256,4,true> at 5.07% of GPU time.
- Suggested A/B:
  - (a) STRATA_MMVF_ROWS=1 on fast-hip right now, with no code change (ALREADY-HAVE switch, untested on gfx1100 according to MOD_PLAN). Caveat: with ROWS on, the aux-row multi kernel is skipped (native_bf16.cu:492), so check that the shared-gate aux path still fuses.
  - (b) Port aa2509dc60 (S).
- STRATA_DENSE_LDS / b7_mmvq_lds_kernel (c783397845) targets native_mmvq B6 rows on gfx12. Our dense GEMVs run on rdna_kq_kernel / rdna_q6k_kernel (STRATA_RDNA_MMVQ_WAVE=3), which is a different code path, so it is NOT-APPLICABLE as is. — [c783397](https://github.com/this-cute-world/Strata/commit/c783397845)

**F4. STRATA_Q6K_V2 and STRATA_FUSE_PROJ (this-cute-world) - NOT-APPLICABLE as is; technique worth noting**
- q6k_v2_kernel issues all weight words, then the x words, then computes. Author-measured on R9700, 2-4 columns: gdn qkv 4c 53.8 -> 44.8 us, qsa q 4c 70.4 -> 55.9 us; 784,512 outputs bitwise. — [76ed34d](https://github.com/this-cute-world/Strata/commit/76ed34d950)
- FUSE_PROJ puts same-input Q6_K projections in one launch. The author notes the window effect is small (~60 us) and that k+v alone is 8-11 us slower fused. — [3228871](https://github.com/this-cute-world/Strata/commit/322887157a)
- Ours: Q6_K dense runs on rdna_q6k_kernel (0.59%) under STRATA_RDNA_Q6K=1. The kernels differ, but the "all loads first, one wait" pattern could be checked in rdna_kq_kernel<23> (~11% of GPU time) with a quick ISA look for flat loads.

**F5. STRATA_DPP_XOR (this-cute-world) - NEW, exact, CMake option, port S**
- Constant-mask __shfl_xor_sync becomes DPP row_xmask / v_permlanex16 instead of ds_bpermute. Parity: 3968 + 186 cases, 0 differ. Author-measured on R9700 (shexp_bench, 4 cols): down IQ4_NL 6.23 -> 5.86 us, gu 9.13 -> 8.78 us. b6_mmvq_bench: small dense -0.5..-6.7%, but the head Q5_K/Q6_K at 4 cols got +1.6% slower. — [6883a8b](https://github.com/this-cute-world/Strata/commit/6883a8b039)
- Applicability: DPP row_xmask and v_permlanex16 exist on gfx10/gfx11 (RDNA ISA), so it should build for gfx1100. Whether it is a gain depends on the kernel. Suggested A/B: a build-hip-dpp tree, short chat + 2K edit.

**F6. STRATA_GR_Q8_V2 (this-cute-world) - NOT-APPLICABLE to our preset**
- Bitwise q8 HC down/up kernels in fused_gr. — [5b17f47](https://github.com/this-cute-world/Strata/commit/5b17f47d1b)
- Ours runs STRATA_GR_RDNA / GR_RDNA_Q8 / GR_FUSED (rdna_gr_fused) with STRATA_HC_Q8=0, so this targets the upstream fused_gr path that we do not use.

**F7. Prefill barrier (this-cute-world 28a938e4ff) - check, port S**
- A single LDS-only barrier per stage in native_w11_kernel (upstream PR #3222), measured by its author as part of a ten-PR set on the R9700. — [28a938e](https://github.com/this-cute-world/Strata/commit/28a938e4ff)
- Not grepped in detail. Whether upstream merged #3222 into the base we forked from is unknown (gap).

**F8. MTP pair rounds, FUSE_DRAFTHEAD/DRAFTEARLY/ACOMMIT, BCOMMIT/IDXAPP/GDNCONV/QSASEL, STRATA_BATCH_PIECE_TOKENS, replica routing (this-cute-world) - NOT-APPLICABLE (single user)**
- All of these need --batch with 2+ slots, or replicas. Author-measured: DRAFTHEAD 23.40 -> 23.11 ms per window in batch mode. — [2b9ba86](https://github.com/this-cute-world/Strata/commit/2b9ba86185), [f0472fe](https://github.com/this-cute-world/Strata/commit/f0472fe741), [e452e0f](https://github.com/this-cute-world/Strata/commit/e452e0fb4c)
- GDNCONV=2 (one-dimensional grid) is also batch-window only: 18.4 -> 6.3 us at 4 rows. Our single-stream path already fuses conv+a/b via STRATA_FUSE_CONVAB (gdn_conv_ab_kernel<4,true>, 0.61%). — [aac7190](https://github.com/this-cute-world/Strata/commit/aac7190ccf)

**F9. CPU multi-token AVX2 rows for Q6_K / Q8_0 (samnolak e9dac29987) - NOT-APPLICABLE**
- Bit-exact multi-column AVX2 port of ik_llama.cpp's structure. Author-measured on one Zen 3 thread (EPYC 7Y43): Q6_K gate/up 1.47x / 1.92x at 2 / 4 tokens; kq_cpu_bench at nt 8: 31.8 -> 50.3 GB/s. — [e9dac29](https://github.com/samnolak/Strata/commit/e9dac29987)
- Our IQ2_XS CPU experts are i-quants, which already have multi-token AVX2 kernels (native_expert.cpp:436-498, #152, STRATA_IQ_MT_MIN). Only relevant for the qwen36 UD-Q4_K_XL / Q6 presets, and those run strata-poly on CUDA.

**F10. Other samnolak items - NOT-APPLICABLE**
- STRATA_SELECT_TWO_PASS in the layer-major walk: ALREADY-HAVE as a switch (6 files in src/).
- STRATA_DENSE_MMQ is CUDA only.
- Per-device VRAM reserve, SSD refill, swarm STRATA_BATCH_CPU_SHARE, IDLE_PARK and STAGE_PREFETCH are multi-GPU / SSD / batch features.
- STRATA_ROPE_SHARE / KV_HOST_LAZY are multi-session memory savers. — [samnolak fork](https://github.com/samnolak/Strata)

**F11. Concurrent cold prompt reads beside decoding (atmJava 540df69c04) - NOT-APPLICABLE**
- Adds FileExpertSource::prompt_copies_independent(), which allows the concurrent batch prompt reader for the io_pf (experts.bin page-fill) source. — [540df69](https://github.com/atmJava/Strata/commit/540df69c04)
- This only matters with --batch slots (a prompt read while another slot decodes).

### Inferences
- Our decode window is mostly CPU-bound (combine waits for CPU = 38% of GPU kernel time). GPU-side savings (F1-F3) therefore translate into tok/s mainly on windows where VRAM-resident hits dominate, and in prompt-heavy turns. A realistic expectation for F1 is a fraction of the ~13% kernel share times ~20%: low single-digit percent, at most.
- STRATA_MMVF_ROWS=1 (F3a) is the cheapest test: no code, exact. Run it first.
- The launch fusions (F2) mainly cut fixed per-launch cost and host/GPU handoffs. On HIP, launch overhead is larger than on CUDA, so they may be worth more on gfx1100 than on the R9700 traces suggest. That has to be measured.

### Gaps
- None of the this-cute-world numbers are from gfx11. Every one is R9700 gfx1201 except the e3ab06a7bd tok/s, which is 2x R9700. No end-to-end single-stream tok/s is reported for EXPDEC_V2 or the FUSE_* set.
- I did not verify whether the fork's V2 Q2_0 down (92983a7405) has its own parity numbers in the commit; the message is a one-liner.
- I did not check whether upstream PR #3222 (F7) is already in our base.

## Q2. Correctness fixes and whether the bug exists in our tree

### Takeaway
One fix (prefill refusing on a null blob_stable) still exists in our tree but is Windows-specific according to its author. The adaptive-tier "lost experts" fix and the KFD pin-ceiling back-off are, respectively, already in our tree and of low risk for us.

### Cited Findings
- **atmJava 0a03f3b858**: in the layer-major prefill, `blob_stable` returns null when the mapping was dropped for unbuffered reads (Windows), and the old code refused the whole prefill. The fix passes a null blob with source/layer/expert so the stager copies it (as in lm_prefetch). — [0a03f3b](https://github.com/atmJava/Strata/commit/0a03f3b858)
  - Our tree still has the refusal: src/prefill/prefill.cpp:3880-3881 `b = m.src->blob_stable(l, e); if (!b) { err = "prefill: expert source has no blob"; return false; }`. On Linux it only triggers if the source drops its mapping (direct/unbuffered I/O). Low priority, port S.
- **samnolak 026e0d8b1a** (upstream PR #4844): the adaptive round's pipelined CopyBack left a stale [a_lo, a_hi), which lost experts. — [026e0d8](https://github.com/samnolak/Strata/commit/026e0d8b1a)
  - ALREADY-HAVE: our generate.cpp:13886-13889 calls `a_rng.whole_round(aswaps.size())` with the same rationale in its comment.
- **samnolak c3c64346ac**: stage_pool lost wakeup on stop (the join could hang). — [c3c6434](https://github.com/samnolak/Strata/commit/c3c64346ac)
  - NOT-APPLICABLE: our tree has no stage_pool.hpp (it is part of STRATA_STAGE_COPY_PARALLEL, PR #4002, which we do not have).
- **this-cute-world d139f4dc1d**: the amdgpu KFD system-pin ceiling (~RAM - RAM/64 - 1.5 GiB, shared by every process) made the first lazily loaded code object fail when the arena filled it. The fix backs off 1 GiB and logs the ceiling. — [d139f4d](https://github.com/this-cute-world/Strata/commit/d139f4dc1d)
  - Our pinned.cu has no KFD handling. With 57 GB RAM the ceiling is ~54.6 GiB (author's formula), against our ~33 GiB arena + KV host DMA. Not hit today. NEW but only defensive.
- **Zutfen aaf23aed12**: the HIP `__shared__ alignas(16) int4` declarations become `__attribute__((aligned(16)))`. — [aaf23ae](https://github.com/Zutfen-LLC/Strata/commit/aaf23aed12)
  - Our iq_kernels.cu:2592/2664/2859 still use alignas. This is a host-compiler (MSVC) syntax issue; Linux hipcc builds it. No action.

### Inferences
- None of these fixes changes decode speed. Only the prefill null-blob fix is worth porting, as a robustness measure, if we ever use direct-I/O expert sources for prefill.

### Gaps
- I did not test whether our Linux FileExpertSource can return a null blob_stable under STRATA_IO_PREFETCH / direct modes.

## Q3. One-line verdict for every fork in the batch

### Takeaway
Only this-cute-world has something to take (F1-F3, F5). samnolak and atmJava have one fix each that we either already have or that is Windows-only. Every other fork has nothing to take.

### Cited Findings
- **Zutfen-LLC/Strata** (158): a Vulkan/"native T1" port of GDN/QSA/HC/PLE plus a gfx906 W4A4 HIP adapter, mostly docs records. Nothing to take for HIP gfx1100. The int4 alignment change is a build-syntax fix only. — [fork](https://github.com/Zutfen-LLC/Strata)
- **this-cute-world/Strata** (78): RDNA4 HIP decode/prefill kernels and launch fusions. Take EXPDEC_V2, FUSE_FETCH/COMBINE/SWIGLU_DOWN/SHGATE/PLAN/ROUTEBELL/QGEMV, one-token MMVF_ROWS and DPP_XOR as A/B candidates. Batch/replica work does not apply. — [fork](https://github.com/this-cute-world/Strata)
- **atmJava/Strata** (47): mostly tidy-up, i18n and docs. The prefill null-blob fix (Windows) and the concurrent prompt reader (batch only) are the only engine changes. — [fork](https://github.com/atmJava/Strata)
- **samnolak/Strata** (33): 4x RTX 4080 multi-GPU / SSD / swarm tuning. The AVX2 multi-token rows are for Q6_K/Q8_0 only, and the lost-experts fix we already have. Nothing new to take. — [fork](https://github.com/samnolak/Strata)
- **bassmicrobe/Strata-Managed-Models** (19): Windows desktop GUI, installer, Tailscale. Nothing to take.
- **crhy/Strata** (14): personal docs/notes. Nothing to take.
- **dongnh311/Strata** (10): drafting auto-pick (MTP over DFlash2 on sm_86), local notes, MCP sidecar description bound. Nothing to take (MTP is already our choice).
- **xelprep/Strata** (9): Maxwell sm_52 build packaging. Nothing to take.
- **LakoMoor/Strata** (6): V100 deployment configs and dashboard. Nothing to take.
- **UniqueDroid/Strata** (5): Windows AVX (non-AVX2) build. Nothing to take.
- **webzone/strata-gfx906** (5): MI50 gfx906 merge records. Nothing to take.
- **kingdonb/Strata** (4): Mac, built-in MCP tools in the agent harness. Nothing to take.
- **Florious95/Strata-5090** (3): README token-count docs. Nothing to take.
- **luiny0901-hue/Strata** (3): venv rebuild, docs. Nothing to take.
- **Master-MD/NALA-Strata-RS** (2): Rust planner/gateway with marketing claims ("52.9x faster than Python Strata"). It compares against the Python wrapper, not the engine. Nothing to take.
- **landian01/Strata** (2): llama-swap serving config. Nothing to take.
- **AmeowCAT/Strata** (1): AGENTS.md proxy note. Nothing to take.
- **ProTekk/Strata** (1): sm_86 toolchain gate allow-list. Nothing to take.
- **elricdog/Strata** (1): GUI monitor insights. Nothing to take.
- **jobjobjob2024/Strata** (1): "v0", no description. Nothing to take.
- **schmittcn/Strata** (1): CI skips macOS in forks. Nothing to take.

### Inferences
- Priority order for pancake:
  1. STRATA_MMVF_ROWS=1 A/B (no code)
  2. EXPDEC_V2 port (M)
  3. FUSE_FETCH + FUSE_COMBINE + FUSE_SWIGLU_DOWN (S-M, exact)
  4. one-token MMVF_ROWS (S)
  5. DPP_XOR build A/B (S)

### Gaps
- Commits of the docs/packaging forks were not opened; the verdicts come from commit subjects only, as instructed.
