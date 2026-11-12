# Active Strata forks, batch 2 (own commits since 2026-11-01, state 2026-11-10)

Scope: the 22 forks in batch2.md (bsorensen110, serboox, riceharvest, VlDubr, xrip, Thymos2027, mgendre, sky-is-green, haasanen, TwoAnts, realno, dys0re, Expired-Pasta, larsnygard, Dylan997S, jibanez-staticduo, teragridllm, MoonbirdLu, dingyiyuan, jjsuperpower, satakecode). Target: pancake-strata on RX 7900 XT (gfx1100) + Ryzen 7 5700G (AVX2 only), Flash-Next 125B IQ2_XS, fast-hip preset with `--pcie-frac 0` and MTP. Our tree is checked at `39f7a3da3`.

All speed numbers below are **author-measured on the author's hardware** unless they say "pancake". Status labels: NEW / ALREADY-HAVE / REJECTED-BEFORE / NOT-APPLICABLE.

## Q1. Which commits in this batch could raise decode tok/s, prompt speed or start time on gfx1100 + 5700G?

### Takeaway
Only two forks have engine work that carries over: **bsorensen110** (168 commits; R9700 gfx1201 + CUDA RTX 5080, a large DFlash2/batch campaign with a few exact CPU-side and prefill items that apply to solo decode) and **serboox** (75 commits; RTX 3050 laptop + AVX-512 Intel, Qwen3.6-35B on the `qwen35` path). Most of their kernel work is AVX-512, gfx12, DFlash2 or `--batch` work and does not apply. The cheapest wins for us are two env A/Bs that need no code, because the switches already exist in our tree: `STRATA_PREFILL_COPY_WAIT_DEDUPE=1` and `STRATA_PREFILL_ROUTED_STREAM=1`. After those come three small ports: `STRATA_FUSE_DECODE` opened on HIP, the RouterLookahead submit moved after the plan publish, and parallel or fused CPU activation quantization. serboox measured two useful **negatives**: experts on an iGPU (a 30.5 -> 10.9 tok/s loss) and THP for the expert reads (no gain). They also measured one lossy lever we have not tried: a uniform reduction of experts per token.

### Cited Findings

**F1. Prefill copy-wait dedupe on by default on HIP: bsorensen110 `8dbbe8f24d` (port of their `28c907aa1`)**
- What it does: without dedupe, the prefill issuer calls `hipStreamWaitEvent(m.copy, m.used[rel])` for all 8 slots of every group. That is 14,720 calls per chunk and 44,160 for a 4,712-token prompt, although one wait per group would do. The commit makes `STRATA_PREFILL_COPY_WAIT_DEDUPE` default on under HIP when `ROUTED_STREAM` is off. — [commit](https://github.com/bsorensen110/Strata/commit/8dbbe8f24d), [A/B doc 17de4f2e4e](https://github.com/bsorensen110/Strata/commit/17de4f2e4e)
- Author-measured (R9700 gfx1201, ROCm 10.2 nightly, IQ4_XS, Wave 4 stack, so not isolated): the warm 4,712-token prompt read went 5,928 -> 5,340 ms, "+11.0% warm prefill tok/s vs control" (882 tok/s). Token hash bit-identical. — [17de4f2e4e](https://github.com/bsorensen110/Strata/commit/17de4f2e4e)
- Exact. Status: **ALREADY-HAVE the switch, default off.** Our `src/prefill/prefill.cpp:3944` reads the env and defaults to false. Effort: none (env), or S to change the default.
- A/B: `STRATA_PREFILL_COPY_WAIT_DEDUPE=0` vs `=1`, with `STRATA_PREFILL_TIMING=1`, on 2K edit TTFT and the 11.7K prompt read. Gate: token identity.

**F2. Routed prompt streaming on HIP plus its GatherGroup and ring fixes: bsorensen110 `b13ffdabd0`; feature origin mgendre `4b6d0e7a1f` / `d8ce2d980b`**
- What it does: with `STRATA_PREFILL_ROUTED_STREAM=1`, the prefill issuer keeps at most `_DEPTH` (16) copies in flight. It drops a not-yet-issued expert copy when no token of the chunk routes to that expert. The trigger is a chunk of 1,024 tokens or more that streams every non-resident expert. bsorensen's commit turns it on by default on HIP and fixes a premature GatherGroup flush: `release_to` used to record one event per slot, and now it records one per span of up to `MMQ_GROUP` slots. It also pauses the issuer on a layer yield and caps the co-prefill ring at 96. — [mgendre 4b6d0e7a1f](https://github.com/mgendre/Strata/commit/4b6d0e7a1f), [bsorensen b13ffdabd0](https://github.com/bsorensen110/Strata/commit/b13ffdabd0)
- Author-measured: neither commit gives an isolated number. The earlier report (item 3.2) quotes the gfx1030 fork's `PF_STREAM_ROUTED`: IQ3_S +10%/+9%, Q4_0 +40%, bit-identical (2x PRO V620).
- Exact (it only skips copies whose expert has no token). Status: **ALREADY-HAVE the switch** (`prefill.cpp:3956`, `include/strata/prefill/routed_stream.hpp`, default off). The GatherGroup span fix from `b13ffdabd0` is **NEW** (S, about 30 lines in `release_to`) and only matters with the dedupe or the routed walk. This turns the earlier report's item 3.2 from "M port" into "env A/B".
- Applicability: at 11.7K, every chunk streams about 33 GiB of experts over PCIe 4.0 x16, so the gain is largest when many experts get no token. That holds for short chunks and is rarer for 4K+ chunks with top-10 of 512.
- A/B: `ROUTED_STREAM=0/1` × `COPY_WAIT_DEDUPE=0/1` (4 arms), on 2K edit and 11.7K TTFT. Note that bsorensen makes dedupe default on only when routed stream is off, so test the cross.

**F3. `STRATA_FUSE_DECODE` (PR #1097 decode fusions) opened on the HIP build: bsorensen110 `20a0f1ae8a`**
- What it does: `quantize_act_images` builds every activation image in one pass instead of 2-3 kernels, and `project_bf16_pair` does two BF16 projections of the same x in one kernel. The kernels already compile under HIP, but the 7 call sites in `src/core/layer.cpp` are gated `#if defined(STRATA_EXPERIMENTAL_SM60)`, so the HIP engine always runs the unfused sequence. The commit adds `|| defined(STRATA_USE_HIP)` to the gates. The runtime switch stays opt-in. — [commit](https://github.com/bsorensen110/Strata/commit/20a0f1ae8a)
- Author-measured: no number in the commit. Our own `src/core/nv_defaults.cpp:50` enables it on an NV class with the comment "-1.3".
- Exact (the same arithmetic, fewer launches). Status: **NEW.** We have the same 7 gates at `layer.cpp:110,263,325,373,482,1159,1191`. Effort: S (7 preprocessor lines plus `mmvf/bf16` parity tests).
- Possible overlap with our `STRATA_MMVF_AUX` and `STRATA_ROUTER_FUSE`; check that the two do not fight.
- A/B: `STRATA_FUSE_DECODE=0/1` on short chat and 2K edit decode; also check the hipGraph node count.

**F4. CPU-side exact overlap stack for decode: bsorensen110 `6b09a8335e`, `e06154af0e`, `a3aee7b03b`, `0ba00b12a5`, `e18b5c8f5c`, `cd3a3ded7b`, `b8b65f5348`**
- What they do:
  - (a) Activation Q8_1 quantization of the window's tokens is deferred to the pool workers on first touch (`defer_act_quant` + an atomic `q_state_` table). `kMaxActQuantToks` goes from 32 to 1024, and the `cold_mode == 0` restriction is removed.
  - (b) The worker that finishes an expert's last gate/up slice quantizes `ff -> hq/a2` while the data is still hot in L1/L2.
  - (c) `RouterLookahead::submit` moves after `gpu_plan.publish()`. Lookahead and `claim_todo` are skipped when every routed cold expert of the layer is already resident in RAM. The O(N²) dedupe is replaced by 512-entry tables.
  - (d) CPU jobs are built straight from the deduplicated chains.
  - (e) Gate/up and down run as one barrier-free pool phase (`STRATA_POOL_SINGLE_PHASE`).
  - (f) `sample_rows` is enqueued before the final stream sync, and a post-layer-47 hook starts the adaptive tier early.
  - (g) The PLE gather waits on an event instead of a stream sync before Layer 0.
  - — [355f7fb988 A/B doc](https://github.com/bsorensen110/Strata/commit/355f7fb988), [899d6e1ea3 A/B doc](https://github.com/bsorensen110/Strata/commit/899d6e1ea3)
- Author-measured (R9700 + unknown CPU, IQ4_XS, `--resident-experts 34 GiB`, solo greedy 2x192): wave 1, (a)-(c) and others, warm engine decode 136.0 -> 144.8 tok/s (+6.5%), hash bitwise equal. Wave 2, (d)-(f): 137.1 -> 142.4 warm (+3.9%) against the same control. In co-prefill, "CPU experts 13.00 -> 10.71 ms/window" (-17.6%). — same sources
- Status, item by item, against our tree:
  - (b) and (e): **ALREADY-HAVE** as `STRATA_POOL_FUSE_PHASES` mode 7 (`pool.cpp:1309`: "a down task waits only for its own expert, whose last gate/up task quantized it"). Pancake measured it at +2.5% on code edit.
  - (a): **NEW.** Our `expert_source.cpp:5232-5241` quantizes all window tokens serially on the caller thread before dispatch. That is about 5-8 tokens × H=2560 per layer, probably only a few µs per layer × 48 layers, so expect ≤1%. Effort: M (touches `pool.hpp` and `expert_source.cpp`).
  - (c): **NEW** in part. Our lookahead submit is at the top of `expert_pool_dispatch_multi` (`expert_source.cpp:4923`), before the publish at `:5183`. Our miss dedupe already uses a per-expert stamp, so it is O(n). With `STRATA_LOOKAHEAD_GPU` on by default, the CPU-side submit cost is small, so the gain is likely small. Effort: S (move one call, plus the "all resident" skip).
  - (f) and (g): sample-async is **NEW**. Our ADAPT_OVERLAP exists but is opt-in. The early adapt start on the post-layer hook is **NEW**. Effort: M.
- Exact (bitwise hash equal in the author's runs).
- A/B: one env toggle per port, short chat and 2K edit. Read the per-window `CPU pool` / `emit` / `adapt wait` stamps from the window trace (item 64).

**F5. Host RAM backup for lent VRAM cache slots during prefill (`STRATA_LEND_RAM_BACKUP`): bsorensen110 `b38a58c090`, `345c2b505e`**
- What it does: when prefill `lend()` borrows expert-cache VRAM slots, the experts in them become non-resident. Unless the RAM budget covers them, they are re-read from file afterwards (7,380-12,459 `pread` blobs). The commit D2H-copies each lent slot once into host RAM and marks it resident until `refill`. — [d5e98ecbb2 A/B doc](https://github.com/bsorensen110/Strata/commit/d5e98ecbb2)
- Author-measured (R9700): solo engine decode 131.2 -> 146.6 warm (+11.8%). Measured with `--resident-budget-gib 34`, where the lent experts were not in RAM.
- Exact. Status: **NOT-APPLICABLE / probably ALREADY-HAVE in effect.** Our setup holds all ~33 GiB of experts in the pinned RAM arena (`PIN_BY_NEED`, mlock on 33,838 MiB per the earlier report), so a lent expert already has a RAM copy, and the file-tier reads that this fixes should not happen. Check: look for file-tier reads after a long prompt in our log. If there are none, skip.

**F6. Fewer experts per token, uniform reduction (lossy): serboox `c985a4b222`, `f8fb28ca1f`, `0f040ef5e8`, KLD gate `35fed154cf`, measured in [TUNING.md](https://github.com/serboox/strata/blob/main/docs/TUNING.md)**
- What it does: `STRATA_Q36_EXPERTS_USED=K` lowers routed experts per token below the file's count in every pass (decode, verify, prompt, MTP). Weights are renormalized over the picked K, as llama.cpp's `expert_used_count` override does. — [f8fb28ca1f](https://github.com/serboox/strata/commit/f8fb28ca1f)
- Author-measured (RTX 3050 laptop, Qwen3.6-35B-A3B IQ4_XS with MTP, 8 experts in the file): 7/8 gave +14.4%, 6/8 +18.7%, 5/8 +30.6% (35.1 -> 40.2 / 41.9 / 45.9 tok/s). KLD: K=7 0.0254 (top-1 agreement 93.1%, PPL +1.7%), K=6 0.0602 (+5.4% PPL). HumanEval pass@1 164 problems: K=8 92.1%, K=7 92.7%, K=6 92.1%, K=5 91.5%, K=4 86.6%. — [TUNING.md](https://github.com/serboox/strata/blob/main/docs/TUNING.md)
- Lossy. Status: **NEW as a uniform-K lever.** Our register has only the weight-threshold skip idea (`SKIP_MINW`, earlier report item 2.8) and route-prior/stand-ins. Flash-Next is top-10 of 512, so K=9 or 8 removes 10-20% of CPU expert rows. CPU expert time is the dominant decode term for us, so the gain could be similar or larger than on a 3050.
- Effort: S-M (the router top-K kernel for K<10, plus the combine renormalization; check the MTP head path).
- A/B: K=10/9/8 on 2K edit and short chat tok/s. Gate: KLD on the replay set, plus token-forcing agreement, plus a small HumanEval/tool-call subset. Rank it next to `VERIFY_DEFER_CPU` as a lossy item.

**F7. Negative evidence that closes ideas for the 5700G**
- **iGPU expert worker** (serboox `797b4a3bf1`, `6163c4df9c`, `5c8c68584a`, `84434e4d90`): a ggml-vulkan worker computes a share of routed experts on the Iris Xe iGPU while the CPU pool runs the rest. Author-measured on 4 November: "it does not gain, it loses". 30.5 -> 10.9 tok/s at share 0.25, and 16.1 vs 31.1 even at 1 entry per layer. One iGPU call costs about 2 ms, as long as the whole pool needs for a layer. The iGPU shares package power and DRAM bandwidth with the cores. — [TUNING.md §Experts on the Iris Xe iGPU](https://github.com/serboox/strata/blob/main/docs/TUNING.md). Status for us: **REJECTED by evidence** for the 5700G Vega iGPU. The case is even weaker for us, because our iGPU also drives the desktop and shares the same DDR4.
- **Transparent huge pages for the expert reads**: microbenchmark with 6 threads and random 1.9 MB blocks: 52.4 GB/s from mapped file pages, 52.4 from 4 KB anonymous pages, 53.2 from 2 MB pages, "no gain". — [TUNING.md](https://github.com/serboox/strata/blob/main/docs/TUNING.md). This weakens (but does not close) the earlier report's hypothesis that per-start hugepage luck explains our bimodal runs. Their access pattern is sequential within 1.9 MB blobs, as ours is.
- **IQ prefetch distance** 0/2048/4096/8192 B: 35.28/35.24/35.15/35.23 tok/s, no difference. MTP depth 3 is worse than 2 (-6.4%). — same source. This agrees with our neutral `STRATA_Q2_PREFETCH`.
- **A lower CPU frequency cap (2.1 GHz) gives steadier long-session speed** on a thermally limited laptop (serboox `c9258a5867`, `e7858734db`). NOT-APPLICABLE on a desktop 5700G, except as a hint to log clocks (already in spec 10).

**F8. Router-agnostic fixes and items with no gain for us**
- bsorensen `3360d42d1e` removes a duplicated `bf16_proj(wu, m.lo16, m.gated, ...)` in the `!upmixed` prefill fallback (96 redundant GEMMs per chunk). **Our tree does not have the duplicate** (`prefill.cpp:4319` has a single call). No action.
- bsorensen `2437d6ddab` / `6198f83f45` skip the dead PCIe-share nodes when `pcie_mode < 0`. **ALREADY-HAVE**: our `verify.cpp` wraps the flag-B wait, `fetch_blobs`, `rebase_ptrs` and the PCIe grouped launch in `if (!pcie_off_)`. The int4 vectorization of `wait_copy_i32_from_mapped` (`6198f83f45`) is NEW but tiny (S, plan copy of a few hundred bytes).
- bsorensen `c057974cb0` folds the PCIe-share wait into its staging copy. NOT-APPLICABLE with `--pcie-frac 0`; the author turned it off by default anyway (`bb7332caca`).
- bsorensen `9d9837ff32` interleaves per-layer slots rank-major so `lend()` borrows the coldest slots evenly. It only acts with `--expert-cache-per-layer`, which we do not use, so NOT-APPLICABLE. It would matter if we ever retry per-layer admission; our `generate.cpp:2024` notes it decoded 21% slower.
- bsorensen `7a5d4584ff` (Q8_1 activation fragments kept in registers across ROWS>1 in `native_mmvq_multi_kernel`): this is our opt-in lab `STRATA_B6_MMVQ_ROWS` idea (`native_mmvq.cu:1464`). **ALREADY-HAVE (lab).** The fork's commit is not in upstream main (compare: diverged). On gfx12 it is default (`612b45c7f5`). Cheap A/B: `STRATA_B6_MMVQ_ROWS=1` on gfx1100 short chat, if not already in a spec.
- bsorensen `998ae6dee9`, `a99f650161`, `b7770864aa`, `f87da90440` (shared-expert gate+up MMVQ pair with SwiGLU in the epilogue; shared gate deferred into `combine_k10_vec4_hits`; resident plan bucketed by M_e; skip zero-fill of hit rows): these are fork-own commits, not in upstream main. They are **NEW** for us, but we already have the S-family `s_gemv_pair_silu` for the shared expert (`shared_expert.cu:338`) and `STRATA_EXPDEC_SWIGLU`. Expect ≤1%. Effort: M, because they touch combine and verify. Low priority.
- bsorensen `72bfea3d06` / `4b2030acae` (multi-warp stage-2 reduction and exact NT=1..8 dispatch in `bf16_f32_mmvf_rows_kernel`; router and MTP projections through MMVF rows): we have `EXACT_T` templates and `STRATA_MMVF_ROWS` on by default (`arch_defaults.cpp:80`), so mostly **ALREADY-HAVE**. The multi-warp reduction part is NEW but small. Effort: S-M.
- bsorensen `aa2284e9e0` / `d11b03c10f` (Q2_0 expert rows, full 8-block groups, straight-line body, scale via one `vpermps`; bit-exact): AVX-512 only, so **NOT-APPLICABLE** to the 5700G. The idea (no masked tail path for full groups, scales loaded once into registers) could be checked against our `q2_avx2_rows.inl` if our CPU experts are Q2_0 (MOD_PLAN notes "Q2_0 is the daily file"). Port effort for an AVX2 variant: M. Author number: "bit-exact", microbench in `d11b03c10f`; no end-to-end figure read.
- bsorensen `084c122106` `STRATA_GGML_PDEP`: BMI2 pdep is microcoded and slow on Zen 1/2 (IQ1_S/IQ1_M dots 16-19x faster without it). The 5700G is Zen 3 with fast pdep, so **NOT-APPLICABLE**.
- serboox `c37e0a5f88`, `fe057bd143`, `01166ba207` (Q4_K/Q5_K 1.2-1.9x, IQ3_S 1.2-1.4x, IQ4_XS down 1.3-1.7x, bit-identical): all AVX-512 (`kq_avx512.cpp`, `iq3s_vbmi.cpp`, `iq4xs_vnni.cpp`), and not our IQ2_XS/Q2_0 types, so **NOT-APPLICABLE**. Two transferable tricks for AVX2 rows are the token count as a template parameter (accumulators in registers) and the prefetch spread over the block loop. Our IQ2/Q2 AVX2 kernels likely already template the token count.
- serboox `8836b6cc84` (`--protect-ram`: a systemd MemoryLow for the page-cache model, plus `--prewarm`): NOT-APPLICABLE, because our experts are pinned and mlocked, not page-cache streamed.
- dys0re `abc9047d77`: `STRATA_CPU_POOL_COST=1` runs the longest expert first, with row partitions weighted by tokens served (requires their `STRATA_CPU_EXPERT_PIPELINE=1`). Author-measured (gfx936 DCU + Hygon host): adaptive code decode 62.63 -> 66.72 TPS (+6.54%), but the geometric mean is +1.95% adaptive and -0.51% fixed-window. Fusing routing with early CPU payload publication gives +2.73% geomean. Exact. Status: **NEW**, related to earlier report item 2.2 (pool "guided"). Effort: M. A/B on 2K edit, where per-expert token counts are uneven in MTP windows. Measurements are "descriptive single-host runs" per the author. — [commit](https://github.com/dys0re/strata-dcu/commit/abc9047d77)

**Correctness fixes that also exist in our tree**
- bsorensen `3a2b7db4f6`: with DFlash2 plus the MTP fallback and `STRATA_SPEC_PROB=1`, CPU misses were forced onto 1-token WindowInvariant kernels. Our tree has the same code shape in `generate.cpp` (`decode_pool_fn` / `gated_proposal`), but we do not run DFlash2, so it is NOT-APPLICABLE today. Note it if DFlash2 is ever tried.
- riceharvest `462fab63fb` / `4e6841265`: repeat, frequency and presence penalties counted the prompt's last tokens instead of the reply's (upstream #4590). **ALREADY-HAVE**: `STRATA_PENALTY_PROMPT` exists in our `generate.cpp` and `sampling_test.cpp`.
- bsorensen `7291c9847e` (`STRATA_SELECT_GRID` lambda and duplicate grid) and `67462de104` (`launch_multi_n` passes `row_ids=nullptr`) fix the fork's own merges. Not in our tree.
- I found no other correctness bug in this batch that is present in our tree.

### Inferences
- The best value per hour is the 4-arm prefill env test in F1 and F2. It needs no code, and the author reports +11% warm prompt read on RDNA4 HIP from the dedupe alone (in a stack). Our hipStreamWaitEvent count per chunk should be the same order (8 slots per group).
- F3 (`FUSE_DECODE` on HIP) is a 7-line port. Its value depends on launch cost on ROCm, which spec 10's `DUMMY_NODES` slope will tell.
- F6 (uniform K reduction) is the only lever in this batch with a double-digit author gain. It is lossy and belongs next to `VERIFY_DEFER_CPU` in the guard work. serboox's HumanEval curve suggests that dropping 20% of experts costs little on code, but Flash-Next top-10/512 has not been measured.
- bsorensen's solo-decode +4-6.5% came mostly from CPU/GPU overlap items. Our tree already has the largest of them (`POOL_FUSE_PHASES`, `ADAPT_OVERLAP`, lookahead on GPU), so the remaining headroom for us is likely 1-3%.

### Gaps
- No isolated A/B numbers for `ROUTED_STREAM` on HIP or for `FUSE_DECODE` on HIP. The bsorensen A/Bs are stacks of 5-30 commits on gfx1201 with a 34 GiB resident budget, so they cannot be split per commit.
- bsorensen's CPU model is not stated in the A/B docs read, which limits how well its CPU-side gains transfer to an 8-core Zen 3.
- I did not open the `d11b03c10f` Q2_0 microbench numbers. I also did not verify whether our pack's CPU experts are Q2_0 or IQ2_XS rows on the AVX2 path, which decides whether an AVX2 port of `aa2284e9e0`'s idea is worth doing.

## Q2. One-line verdict for every fork in the batch

### Takeaway
2 forks with transferable engine work (bsorensen110, serboox), 1 with a small transferable CPU-scheduling idea (dys0re), 1 already covered by our tree (mgendre routed stream). The rest are other platforms, serving or GUI changes, docs or packaging.

### Cited Findings
- **bsorensen110/Strata**: see F1-F5 and F8. The rest is DFlash2 (dozens of commits), `--batch` co-prefill and slot handoff, gfx12/RDNA4 WMMA, QSA and hipBLASLt tables (gfx1200/gfx1101 tables: gfx1101 is RDNA3 7800 XT, but our gfx1100 table is already active), AVX-512 VNNI/VBMI and Zen 5, ARM NEON, Codex `/v1/responses` serving, and two-tier disk conversation cache (`f102e3b6e1`, `4383fcec3b`; we have item 39 disk parking). Not applicable or not needed. — [fork](https://github.com/bsorensen110/Strata)
- **serboox/strata**: F6 and F7. Its AVX-512 IQ3_S/IQ4_XS/Q4_K kernels, Q36 overlap (`STRATA_Q36_OVERLAP`, qwen35 engine only), thermal/frequency kit, MTP-graft tool, Q8_0->Q6_K dense re-quantizer (lossy, smaller VRAM; our dense is already small) and sandbox tools give nothing to take. — [fork](https://github.com/serboox/strata)
- **riceharvest/strata-b70-only**: SYCL-only (Arc B70) kernels. The penalty fix is ALREADY-HAVE. The `STRATA_SPEC_PROB` default on B70 (+5.5% on sampled turns) uses an engine switch we have (`mtp.cpp`, `verify.cpp`). It could be an A/B for sampled (non-greedy) agent turns, but we run greedy-ish opencode. "Overlap expert copies with batched read" and FP16 small-row prompt experts are SYCL-specific. — [fork](https://github.com/riceharvest/strata-b70-only)
- **VlDubr/strata-mi50**: GLM-5.3-Flash on 4x MI50 (gfx906 wave64), RAM striping across NUMA regions, 4-GPU split. Nothing to take (single GPU, single NUMA node). — [fork](https://github.com/VlDubr/strata-mi50)
- **xrip/Strata**: sm_75 (Turing / CMP 50HX) MMVQ planes and IQ4 W4A8 MMA, upstreamed as PRs #5137 and #5498. Nothing to take. — [fork](https://github.com/xrip/Strata)
- **Thymos2027/Strata**: docs and test archives only. Nothing to take.
- **mgendre/Strata**: `STRATA_PREFILL_ROUTED_STREAM` (F2, already in our tree). Peer-GPU slow link, prelude cache and conversation disk level (we have item 39 / pinning). Nothing new.
- **sky-is-green/Strata**: Scion 176B port onto v0.2.10. Nothing to take.
- **haasanen/Strata**: Windows CI/test fixes for v0.2.13. Nothing to take.
- **TwoAnts/Strata**: serving only (`json_repair`, `STRATA_PARK_TRUNCATE` parks the prefix that fits, traffic class profiles). No speed. `PARK_TRUNCATE` is a possible cache-hit improvement for long agent sessions, but low priority.
- **realno/Strata**: GUI tool-approval popup. Nothing to take.
- **dys0re/strata-dcu**: gfx936 DCU. `STRATA_CPU_POOL_COST` (F8, NEW, M, mixed geomean); a shared Q8 MMAC allocation (DCU-specific). Little to take.
- **Expired-Pasta/Strata**: Korean README. Nothing to take.
- **larsnygard/Strata**: setup.py host gcc selection for nvcc. Nothing to take.
- **Dylan997S/benchmarks-strata**: 4x R9700 benchmark graphic. Nothing to take.
- **jibanez-staticduo/multistrata**: NVFP4 TP3 over RoCE on DGX Spark. Nothing to take.
- **teragridllm/Strata**: benchmark docs (RTX 5060 Ti, no comparison done). Nothing to take.
- **MoonbirdLu/Strata**: RTX 4090 benchmark entry. Nothing to take.
- **dingyiyuan/Strata**: README link. Nothing to take.
- **jjsuperpower/Strata**: a Nix flake. Nothing to take.
- **satakecode/Strata**: independent weight and context mlock in Core and DS4. We already mlock the expert arena. Nothing to take.

### Inferences
- The batch confirms the earlier report's conclusion: no exact large jump. The prefill env cross (F1+F2) and one lossy lever (F6) are the only items worth a spec entry. F3 and F4(c) are small ports for spec 13.

### Gaps
- I did not open riceharvest's SYCL commits beyond the subjects. Given the SYCL-only tree, that is unlikely to matter.
