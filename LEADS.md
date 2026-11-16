# Pancake leads catalogue

Every unported idea collected by the research rounds (forks of Niko1221/Strata, upstream PRs, llama.cpp, papers, hardware notes, Avalon comparison), checked against our tree on 2026-11-15. This file is the only record of unported ideas; add new leads here, never in a separate note.

What is ranked and scheduled lives in MOD_PLAN.md (sections 5 and 6). This file is the full list to pick from; a lead moves into MOD_PLAN.md when it is chosen, and its test results go to TEST_LEDGER.md.

Tags: author numbers are on the author's hardware, not ours; "exact" = bit-identical output intended, "lossy" = needs token forcing + the task gate. Two-GPU and RTX 3060 / sm_86 items are parked (single RX 7900 XT box since 2026-11-08).

## 0. Leads added on 2026-11-15 (from the Speedy audit, the side docs and the forgotten-findings pass)

| lead | number + hw | exact/lossy | in our src |
|---|---|---|---|
| 0xluk `STRATA_PF_AHEAD` + `STRATA_REFILL_ASYNC`: prefetch the next chunk's experts into an N-slot pool from the loan while attention runs (the copy engine idles through attention on a warm turn); split must ignore the pool; log `refilled ... ms` before porting REFILL_ASYNC | agent prefill +35%, TTFT 4.97 -> 3.67 s (bundle); refill ~143 ms off TTFT; 2x RTX 3060 PCIe 3 | exact | 2026-11-16: CODED STRATA_PF_AHEAD + STRATA_REFILL_ASYNC |
| llama.cpp #31751: one pragma fixes RDNA3 MMQ J=32 register spills | +36% at 17-32-token batches, gfx1100 | exact | CODED 2026-11-15 as opt-in build switch (MOD_PLAN.md section 6 item 3, T-579) |
| launch fusions `FUSE_SHGATE/COMBINE/SWIGLU_DOWN/PLAN/ROUTEBELL/QGEMV` | 2,648 -> 2,248 kernels/window, R9700 | exact | 2026-11-16: CODED STRATA_FUSE_* (this-cute-world) |
| uniform K experts (`STRATA_Q36_EXPERTS_USED`) | +14.4% K=7/8, +18.7% K=6/8, KLD 0.0254 at K=7; RTX 3050 laptop, Qwen3.6 IQ4_XS | lossy | 2026-11-16: CODED STRATA_Q36_EXPERTS_USED |
| `EXPDEC_V2` remainder (issue-first loads, 24-bit multiplies) | expdec_gu<22> 69.1 -> 52.5 us, down 48.2 -> 38.5 us; R9700 | exact | 2026-11-16: CODED STRATA_EXPDEC_V2 |
| phoenixclyde `STRATA_MMVQ_WAVE_ROWS` per-shape table | +4.0% decode, 46.8 -> 48.6 tok/s; MI50 | bitwise | 2026-11-16: CODED STRATA_MMVQ_WAVE_ROWS |
| gputier/StrataGP 16 packaged patches | +5-15% decode, +42-49% prompt; RTX 5090 | mixed | 2026-11-16: CODED STRATA_SER, STRATA_SER_LAYERS, ROUTE_PRIOR_DELTA; then STRATA_Q36_PROMPT_SWAPS (swap cap, 5f7c616102 / 819b1dba79), STRATA_SWIGLU_SKIM / STRATA_Q36_SWIGLU_SKIM (803f98afdc, CPU pool's ggml-cpu dot path, both engines), STRATA_SER_POOL (--ser-pool); cores list = ALREADY-HAVE (--pool-cores / STRATA_POOL_CORES); specs 100, 101 |
| Maya `MAYA_GLM_MISS_SKIP_W` (skip a missed expert by gate-weight share) | +7-13% decode at KL 0.037-0.052; GLM | lossy | 2026-11-16: CODED STRATA_MISS_SKIP_W |
| small exact bundle: `copy_rows_from_mapped` loads before stores (elementwise.cu:255-259); sampler/verify 8 logit loads in flight; single-token `MMVF_ROWS`; RouterLookahead submit after plan publish | 4.7-5.7 -> 2.4-3.9 us/launch (5090); ~110 us/argmax round (gfx906); 25.8 -> 10.2 us (R9700); part of +6.5% (R9700) | exact | 2026-11-16: CODED ROWS_LOADS_FIRST, SAMPLER_X8, MMVF_ROWS1, LOOKAHEAD_AFTER_PUB |
| two-pass session restore; `CKPT_MOVE_PARK` port | 198,830-token session restored in 3.4-3.5 s; ~92 ms/park (MI50) | exact | 2026-11-15: CODED STRATA_CKPT_MOVE_PARK (two-pass restore left) |
| Speedy `STRATA_KV_STAGE_REQ`: size the streamed K/V staging pool by the request, not `--max-context` (moves the startup loan to the auto prompt chunk) | 1M ctx: auto chunk 512 -> 4608; 300K cold prompt 466 -> 1527 t/s (2x RTX 3080) | exact || CODED 2026-11-15 (MOD_PLAN.md section 6 item 11) |
| Speedy `STRATA_PF_SEL_BATCH` (QSA selection batch 16-256) | VRAM 256 -> 64 MiB per card at 1M cells | exact || CODED 2026-11-15 (MOD_PLAN.md section 6 item 11) |
| Speedy `STRATA_TOOL_STEP_SPLIT` + `"canonical_tools"` (re-split client-merged tool steps, stable tool render order; serve/ only) | turn-2 cache reuse; no tok/s given | exact || CODED 2026-11-15 (MOD_PLAN.md section 6 item 11) |
| upstream #5273 short-chunk ring release (`give_back(consumed)` only when `stream_all`) | correctness | exact | 2026-11-16: already fixed (T-575) |
| routed-id out-of-range message with layer/entry/id | diagnostics | exact | 2026-11-15: CODED routed-id error message (no switch) |
| rdna3 `791cca4b` + `5906629d`: `PREFILL_AUTO_MAX` up to 49152 + `STRATA_PREFILL_EQUAL_TAIL_MAX` | fixes 49-57K chunk reads; 49,152 grid est. 2-4%, +0.6 GiB loan, needs refill head `2b19d510` (`REFILL_HEAD_BIG`), 2 GiB R-buffer limit | exact | AUTO_MAX yes, EQUAL_TAIL_MAX no; our equal-tail rule differs (their 49K trap cannot happen at our 32K ceiling) |
| rdna3 `f557b544`: prefill CPU share retries the pool at every layer | - | exact | 2026-11-16: CODED STRATA_PREFILL_POOL_RETRY |
| rdna3 `88686782`/`877bf971` `STRATA_PREFILL_TIMING` CPU-share thread time + host wait; `7df6538d` slim prompt-only GGUF | diagnostics | exact | 2026-11-16: CODED STRATA_GR_UPPRE, PREFILL_TIMING CPU-share line, nvfp4 slim |
| Juud_engine CPU tweaks: shorter worker spin, no activation quant for GPU-resident tokens; env-only test: `STRATA_POOL_SPIN_US` sweep (#4625: 0 means zero, pool.cpp:794) | +9.6-15.6% (4090); 15.1 -> 12.6 ms/token | exact | 2026-11-15: CODED STRATA_ACTQ_SKIP_NOCPU |
| #4005 Vop3p sparse copy events + slot-wait dedup in prefill (PR closed unmerged) | +36% HIP prompt at 4K (RDNA2) | exact | 2026-11-15: already prefill.cpp COPY_EVENT_EVERY / COPY_WAIT_DEDUPE |
| `SELECT_TWO_PASS` on HIP (never checked) | <3% at 100K | exact | yes (untested on HIP) |
| prefill ring 96 vs 192 | ~1% (R9700: 192 cost 1.3%) | exact | yes (`PREFILL_RING=192`) |
| `PREFILL_ROUTED_STREAM` matters only just above 4,096 tokens; CPU share 0.6/0.7/auto only for <=400-token deltas (-0.05..-0.15 s) | - | exact | yes |
| upstream PRs not yet triaged: #3925, #5566, #4876, #5593, #4972, #5637/#5437, #5588, #2392, #5643, #5470/#5596/#5453/#5472, #5545, #5517, #5471 (Q4_K/Q5_K decode GEMVs, DeepSeek path), #5623 (shared immutable host KV prefixes), #5609 (verify q36 MTP block loads but never drafts is fixed in our base), server #5516/#5512/#3971/#5628/#4777/#5466/#5518, bahooo22 q36 #5595/#5467/#5468, #5475 `HC_PERSIST_Q8`, #5535 int8-WMMA prompt attention on gfx1100, #3509 `ARENA_REUSE` + 0xluk `e2ffeef8da`, alesha-pro trellis "S" quant #3384 (lossy) | - | - | 2026-11-16: method: triage list (each PR its own row) |
| Maya `MAYA_EXPERT_DIRECT`, Maya prompt memo, `--calibrate`, drafter auto-keep, Private_Context_COW | +26%; ~6% TTFT | - | 2026-11-16: already: O_DIRECT miss reads + PromptEncoder (Private_Context_COW: no source) |
| havok2-htwo/StrataQ38FN agent-switch re-read patch | 25 s -> 1.5 s (author) | unknown | 2026-11-16: park: 2+ GPUs (layer-split borrow) |
| storm-ace q36 GQA decode, sampled MTP, `Q36_FA` WMMA prompt attention | pp32768 360 -> 517 -> 938 tok/s | FP16-level lossy | 2026-11-16: CODED STRATA_Q36_ATTN_GQA, Q36_ATT_GROUPED, Q36_SPEC_SAMPLE, Q36_FA (storm-ace) |
| svgop hugetlb pool for the expert copy | +4.9% IQ2_XS (CUDA) | exact | 2026-11-16: CODED STRATA_COMPLEMENT_HUGETLB |
| deferred activation quantization in the pool + sample-async | +6.5% / +3.9% bundles (R9700) | exact | 2026-11-16: CODED STRATA_POOL_DEFER_ACTQ + STRATA_SAMPLE_ASYNC |
| GatherGroup fix `b13ffdabd0` for `ROUTED_STREAM`; dardeaw exact wave32 QSA decode kernels; `STRATA_FUSE_SHGATE` (this-cute-world) | - | exact | 2026-11-16: CODED PF_GG_DEFER, FUSE_SHGATE, STRATA_QSA_WAVE32 (dardeaw) |
| watch list: Infernix (+23% at 262K, 5090); Qwen3.6 issues #5069 #2464 #5551 #1402 | - | - | 2026-11-15: method |

Closed by the 2026-11-15 pass (do not re-queue without new evidence): AMD MEXT (closed commercial NVMe-as-DRAM product for >64 GB servers; expert lookahead prefetch already measured 0.97x, T-190); new compression ISAs (AVX-512/GFNI/VPCLMULQDQ are Zen 4+, MXFP4/MXFP8 matrix cores are RDNA4, BMI2 pdep/pext microcoded on Zen 3; entropy-coded weights bound 8-12% < 20% bar, T-282); thread-count / SMT / CCD / futex / nice variants (measured worse; the pool is bandwidth-bound); BFLA on Flash-Next (paper BFLA 1.273x vs QSA 1.319x at 128K, selector 2.6% share); dedup (T-165); Avalon prompt path (fit 0.13 s + 0.61 ms/token, nothing to port); `--short-read` (crossover already ~64); DSpine (no drafter for our models); basalt draft head / K/V append / .basalt format; qwen38-megakernel (AGPL, ideas only); Aviary, jitllm, colibri, vLLM-class engines; SpecPrefill (lossy, breaks GDN); Speedy `speedy_defaults.hpp` (silent default flips), `RP_SNAPSHOT` (measured decode loss), its AMD block and its non-commercial licence; root `*_OPTIMIZATION_REPORT.md` recommendations (grep counts; the syncs/printf/memcpy they name are init, trace or error paths); BFLA on Flash-Next QSA layers: no speed ceiling (author numbers, H200 vs dense FA3: QSA 1.490x vs BFLA 1.435x TTFT on LongBench-v2, 1.319x vs 1.273x on RULER 128K; BFLA wins only on accuracy, RULER 94.36 vs QSA 92.39, dense 94.65), so it was ported to the qwen35 engine instead (MOD_PLAN.md section 10; tile-shared QSA selection T-098); strata-forge (gemovicnemanja, Apache-2.0, static reading only): an async Python LLM toolkit over LiteLLM/vLLM/SGLang/TGI with no engine of its own, nothing for decode; its own risks: `pickle.loads` (llm/cache.py:215), `eval` (calculator.py:87); the 2026-11-15 LLM report fact-check (102-agent workflow): Strata-rdna3 "2x 7900 XTX ~3,000 tok/s prefill" no support; Qwen 27B 180-200 tok/s on 2x 3090 contradicted (55-89 single-stream; the high figure is aggregate); Jetson Orin 32-46 tok/s not supported (1.11-1.64x in NVIDIA tables); 27K cold prefill in 18.8 s one source, low confidence; MorphServe verified but datacenter-only; MiniMax H3 80 -> 31.7 s, QEngine 411 tok/s, Elastic KV 1 s resize unverified. A15 note: the built rule evicts the shortest tool-pending entry by tokens, not the design's "smallest entry whose bytes cover the shortfall" (owner question, MOD_PLAN.md D11); android-imdad caa9b00115 and the PF_GU_EXACT series: all ported 2026-11-15 as opt-ins (MOD_PLAN.md section 6 item 4)



## Part A: fork-research leads vs our tree and MOD_PLAN.md

Branch `docs/consolidate-plan`. MOD_PLAN.md line numbers are from the working copy (3984 lines, uncommitted edits present).
"in our src": `git grep` hit in `src/` or `include/` (file:line). A hit only in `conversation_disk_env.def` is not counted; the reading site is given instead.
"no" = absent. Numbers are fork-author claims on the author's hardware unless marked "ours".

SI = speed-ideas-all-forks/report.md; HN = speed-ideas-all-forks/notes/hip_rdna_kernels.md; MO = .../notes/moe_offload_cpu.md; SM = .../notes/speculative_mtp.md;
SG = strata-forks-speed-gains/report.md; AR = strata-forks-speed-gains/notes/amd_rdna_ports.md; PU = .../notes/platforms_upstream.md;
AF = active-forks-nov/report.md; B2 = active-forks-nov/notes/batch2.md; KB = fork-knowledge-base/report.md; SC = source-commits/notes/commits_2026-11-09.md.

### A1. speed-ideas-all-forks (FK-01..FK-59 plus extra leads from the notes)

| lead | number + hw | exact/lossy | in our src |
|---|---|---|---|
| FK-01 exact spec sampling `SPEC_PROB=1 COUPLED=1 GATE=top DT=0.85` | +5.5% agent T=1.0 (riceharvest, Intel B70); +5.8%, +13.4% with DT 0.85 (svgop, RTX 5090); 0 on greedy | exact | coupled_draft.hpp:69; spec_prob.hpp:53; sampler.cu:950 |
| FK-02 `--prompt-cache-spread` in preset | edited 82K resume 40.0 to 10.2 s (xrip, 2x Turing); re-read cells 130K to 86K (Lamina) | exact | generate.cpp:994 |
| FK-03 `--lookup-max-t 16` x `--lookup-chain 3` | 1,477/1,487 lookup drafts accepted on rename edit (zsariboga, RTX 2060S); est. 0-4% edits | exact | generate.cpp:1024 (chain); lookup_default.hpp:1 |
| FK-04 `--aux-cpus auto` (= Ben1078 `HELPER_CPUS`) | +3.8% decode, host-plan wait 1.52 to 0.51 ms/window (Ben1078, 7800X3D + RX 7900 XTX) | exact | session.cpp:570; aux_cpus.hpp:9 |
| FK-12 `--batch 2` for parallel sub-agents | 70/105/150 tok/s aggregate at 2/4/8 requests (unimatrix099, 2x XTX) | exact per stream | generate.cpp:3963 |
| FK-07 `STRATA_SH_STREAM=1` | +4.0% decode on gfx1100 (xyzzing, XTX); noise on RTX 3060 | exact | arch_defaults.cpp:88 |
| FK-08 `PREFILL_AUTO_MAX=24576` | prefill +15% at 32K, +7% at 64K (rxr2, XTX 64 GB) | exact | generate.cpp:847 |
| FK-09 `STRATA_WMMA_GEMM=1` | prompt +5.6-9.1% (troytse, gfx1151) | rounding | gemm.cu:1162 |
| FK-10 `ATTN_LANECELL=1 ATTN_MERGE_V2=1` at 28K+ | decode +9..+24% in a bundle (UltimateBoomer, gfx1101); rxr2 XTX: did not hold above noise | unknown (hash) | arch_defaults.cpp:51-52 |
| FK-13 `STRATA_PF_HCDOWN` KL recheck (preset sets 1) | +1.3% prefill but first-token KL doubled at 16K (rxr2, XTX) | lossy check | prefill.cpp:580 |
| FK-20 `STRATA_PARK_TRUNCATE` | 160K chain re-read 225 s avoided (TwoAnts); ours: 7 cold 54-61K turns, ~162 s | exact | generate.cpp:974 |
| FK-21 `STRATA_SPEC_CUM` cumulative draft stop | +6.4% short, +2.1% agent with chain 3 (android-imdad, XTX IQ3_S) | exact | src/spec/disco.cpp:106 |
| FK-21b `SPEC_LOOKUP_Q` lookup-agreement arm | +5.8% unseen prompts; "a wash" with suffix drafts on (android-imdad, XTX) | exact | 2026-11-15: CODED STRATA_SPEC_LOOKUP_Q, STRATA_SPEC_DEPTH_DECAY |
| FK-21c SPEC_CUM + `--spec 6` two-arm | `--spec 6` +5-20% vs 4 (android-imdad, XTX); ours: flat alone | exact | 2026-11-15: method |
| FK-22 router publish fast path `STRATA_ROUTE_FAST` | ours: route_publish_res 17.8 us x 4,128 = 2.68% GPU; DPP ~1 us/launch (android-imdad, XTX); +2.5% code (Lamina, RTX 3090) | exact | verify.cpp:2529; native_router.cu |
| FK-23 bit-exact Q2_0 AVX2 fast unpack | 1.11-1.45x on 1 thread; all-Q2_0 layer -17% in a 7-worker pool (imanu86) | exact | q2_avx2.cpp:60 (`unpack64_fast`, opt-in `STRATA_Q2_FAST`, :247) |
| FK-24 Q2_0 down 4-lane fold `EXP9_DOWN` | down 0.47-0.96x of mode 8, +1.6-2.0% e2e (KhaledElzarw, RTX 3090) | exact | iq_kernels.cu:6500 |
| FK-26 fill-stall grid padding | empty 384x256 launch 14-16 us; trim to 96 blocks +6% decode (unimatrix099, 2x XTX) | exact | 2026-11-15: CODED STRATA_FETCH_BLOCKS |
| FK-27 cache bundle: `PARK_REWIND_MAX` / skip-tail / fork-point / borrow / trailing reminder | 12 turns 10.4 to 6.3 s (Xiaomi2008, RTX 5090); 6 tasks 412 to 369 s, switch 3.00 to 0.67 s (Lamina, RTX 3060) | exact | 2026-11-16: CODED PARK_REWIND_MAX, STRATA_CACHE_FORK_POINT, STRATA_TRAILING_REMINDER (serve); borrow already conversation_cache.hpp:359 |
| FK-28 `STRATA_ADAPT_PROMPT` | est. +2-5% first decode tokens; RandoTeam counter -3% prompt (RTX 5090) | exact | 2026-11-15: CODED STRATA_ADAPT_PROMPT (FK-28) |
| FK-30 IQ2 grid-sign expansion | IQ2_XXS gate/up -29%, IQ2_S down -35% per kernel (yqYo1, RX 7900 XT, GLM engine) | exact | iq_kernels.cu:91 (`STRATA_IQ_SIGN_FAST`) |
| FK-31 sub-warp grouped expert kernels | +5.7-8.3% e2e (Lamina, RTX 3090 + 5070 Ti); est. 0-1% on XT | exact | 2026-11-16: already iq_kernels.cu sub-warp kernels (default, GU_SWIGLU, EXP_MODE=10) |
| FK-32 `WAIT_COPY256` + 16 B plan reads | host-write-to-plan 4.47 to 3.57 us (android-imdad, XTX) | exact | switch_report.hpp:31 |
| FK-32b `STRATA_ADAPT_COPY2` (2 DMA engines) | 48 x 2.2 MB copies 4.00 to 3.70 ms (android-imdad, XTX) | exact | 2026-11-16: CODED STRATA_ADAPT_COPY2 |
| FK-32c MULTI_BLOCK_ARGMAX on gfx1100 | 12 to 5.6 us a pick (android-imdad, XTX) | exact | verify_kernels.cu:1026 (gfx12 only) |
| FK-33 `DRAFT_EARLY` + `DRAFT_SUFFIX_SKIP` | no number; draft step 2.2-2.5 ms caps it ~1% | exact | 2026-11-16: CODED STRATA_DRAFT_EARLY + STRATA_DRAFT_SUFFIX_SKIP |
| FK-34 `--light-expert-cpu` byte-aware placement | est. 0-2% (IQ2_XS file only) | exact | 2026-11-16: CODED STRATA_LIGHT_EXPERT_CPU |
| FK-35 shared n-gram pool `NGRAM_MOD` | 234/234 drafts on a copy fixture, no tok/s (tctien342) | exact | 2026-11-16: CODED STRATA_NGRAM_MOD |
| FK-36 prefill combine + write_norm fusion | 234 to 203 ms; 32K -4.0% (zsariboga, RTX 5090) | exact | 2026-11-16: CODED STRATA_PF_COMBINE_WRITE + STRATA_PF_SWIGLU_Q8 / PREFILL_SWIGLU_QUANT |
| FK-36b `PF_DQ_OVERLAP` side-stream dequant | prompts +7-11% at 8K chunks, 0 at 32K + DOWN_STREAMS (android-imdad, XTX) | exact | 2026-11-16: CODED STRATA_PF_DQ_OVERLAP |
| FK-40 post-READY warm-up replay | first turn after restart only (svgop) | exact | 2026-11-16: CODED STRATA_WARMUP_REPLAY (serve) |
| FK-41 arena mlock vs HMM userptr stalls | queue stalls under memory pressure (Ben1078, XTX) | exact | 2026-11-15: method |
| FK-50 `STRATA_VERIFY_ECAP` tail expert cap | +2.8% pooled, +11.5% drift-paired; tight cap broke 12/20 tool turns (jmnargi, 2x V100) | lossy | expert_source.cpp:4989; verify_ecap.hpp; 2026-11-17: ECAP v2 CODED (`STRATA_VERIFY_ECAP_ROWS`, `_MINP`, `_MAX_ROWS`, MOD_PLAN section 6 item 41, T-723, spec 139 queued T-724) |
| FK-51 route-prior guards `PROTECT/KEEP_P/SCOPE` | 0 alone; lets turbo run lambda >0.5 (ours turbo 0.5: +25-34%) | lossy | route_prior.cu:10-13 |
| logit-range-scaled prior bias (`--cache-prior`), SER | no number (gputier) | lossy | 2026-11-16: CODED STRATA_ROUTE_PRIOR_DELTA |
| turbo vs uniform tail skip at equal KL | uniform top-k beat miss-targeted dropping (lighttransport, MiniMax) | lossy method | ROUTE_TAIL_SKIP expert_source.hpp:670 |
| FK-52 sub-2-bit shadow store for CPU misses | +2.9% solo, +20.8% concurrent; tmpfs/hugetlb +5.5% (svgop, RTX 5090) | lossy | 2026-11-16: already STRATA_SHADOW_MISS |
| FK-53 ISTA Coder pack 256/512 | +7.8/+10.0/+11.1% masked-router analogue (svgop, RTX 5090) | lossy | 2026-11-15: method |
| FK-54 window decode attention | 200K decode 48.6 to 68.2 exact, 88.9 fp16 (Asofwar, RTX 5080); Flash-Next capped ~1% | rounding | q36_engine.cu:363 (`STRATA_ATTN_WINDOW`, q36 only) |
| FK-55 Qwen3.6 on HIP: storm-ace WMMA flash prompt + GQA decode | pp32768 360 to 938 tok/s (storm-ace, gfx1151) | fp16 | 2026-11-16: CODED STRATA_Q36_FA (needs Q36_ATT_GROUPED) |
| FK-57 requant study UD-IQ2_XS vs our GSQ IQ2_XS | est. up to -20% expert bytes | lossy | 2026-11-15: method |
| FK-58 typical acceptance in thinking only | unknown; thinking ~6.6% of turn time | lossy | 2026-11-16: already STRATA_TYPICAL_THINK |
| FK-59 fp16 rocWMMA QSA select arm / `SELECT_SPLIT2` | fp16 arm 1.64x select at 128K (xyzzing, gfx1100) | exact / lossy | 2026-11-16: CODED STRATA_SELECT_SPLIT2 (gfx12 scorer gate) |
| android-imdad exact decode bundle `TOPK_REGKEYS/TOPK_SCAN/MTP_SELECT_WIDE/GDN_NORM128/ATTN_MERGE_BOUND` | bundle (with ROUTE_FAST etc.) short chat 86.8 to 95.8 (+10%), agent +11%, KL 0 (XTX IQ3_S); no per-switch split | exact | 2026-11-16: CODED android-imdad caa9b00115 bundle |
| `PF_GU_EXACT` Tensile gfx1100 prompt GEMM | 26.5 vs 12.9 TFLOPS; 16K 13.05 to 10.05 s, 64K 40.1 to 31.5 s, KL 0 (android-imdad, XTX) | exact | 2026-11-16: CODED STRATA_PF_GU_EXACT |
| `PF_DOWN_STREAMS=4` (down over 4 streams) | prompts 2K -3..-4%, 16/32K -2% (android-imdad, XTX) | exact | 2026-11-16: CODED STRATA_PF_DOWN_STREAMS |
| `HIP_POST_SPLIT` / persistent FFN half | +3.5% (needs PCIe share); FFN-half floor 67 vs 150 us/layer, est. +10-14% (unimatrix099, 2x XTX) | exact | 2026-11-15: park: 2 GPUs |
| `HIP_FRONT_FUSE` / `FUSE_SHGATE` (shared gate into router GEMV) | +2.7% decode, 15 to 11 kernels/layer (unimatrix099, 2x XTX) | exact | 2026-11-16: CODED STRATA_FUSE_SHGATE |
| `PCIE_USAGE` + `PCIE_KEEP=96` swap-ahead ring | +4.1%, copy wait 1.2-1.7 to 0.2-0.5 ms (android-imdad, XTX, pcie-frac 0.35) | exact | 2026-11-15: already ADAPT_FETCH |
| IQ2_S/IQ3_S AVX2 decode without shifts (HiSpread, imanu86 862c18aaea) | IQ2_S layers pool -20..-33%/window | exact | yes: iq_avx2.cpp HiSpread / Fmt32<22>, <21>, <121> are the default AVX2 row path (checked 2026-11-15; iq_avx2_parity covers 21, 22) |
| pool spin ~10 ms before nap | CPU experts 15.1 to 12.6 ms/token (Jackwwg83, Zen 5 laptop) | exact | `STRATA_POOL_SPIN_US` pool.hpp:291 (value sweep unverified) |
| one-token exact AVX2 rows (`CPU_SINGLE_EXACT`) | IQ3_S gate/up 2.0x on P-core after fix (Strata-GLM) | exact | `STRATA_CPU_IQ_ALL_EXACT1` (def:348; reader unverified) |
| `SLOTS_WARP` warp-per-row slots kernel | 98.5 to 111.4 t/s short (RandoTeam, RX 7800 XT, Qwen3.6 poly) | exact | 2026-11-15: CODED STRATA_SLOTS_WARP |
| `MTP_HC_Q8` draft-layer Q8 copies | draft step -0.03 ms | rounding | 2026-11-15: CODED STRATA_MTP_HC_Q8 |

### A2. strata-forks-speed-gains

| lead | number + hw | exact/lossy | in our src |
|---|---|---|---|
| `STRATA_GDN_PP=2` | recurrence 1.76x, prompts +2% (2x PRO V620); 13.6K recurrence 999 to 643 ms (Linsche, RX 6800) | exact | arch_defaults.cpp:44 (default only gfx12/gfx1151) |
| `STRATA_DENSE_MMQ=1` (+`MMQ_KQ`) | prefill +11-13%, KL noise (P40); q36 "+43%" (burzumishi) | rounding | gemm.cu:1377 |
| `STRATA_PREFILL_LAYER_MAJOR=1` | 4,400 tok 59.8 to 7.4 s (SSD-bound, RTX 5090); +1.5% (asahi) | exact | prefill.cpp:1283 |
| `--kv-grow` | +6% (94.5 to 100.0, RTX 3090); 69.6 to 73.5 (RTX 4070) | exact | layer.cpp:621 |
| `--adapt-decay 0.97` | 0.97 best; 0.95: 120.2, 0.99: 113.8 tok/s (RTX 3090) | exact | generate.cpp:923 |
| `--vram-reserve-mib` 444 / 1000 | 444: prefill +10-17% (RTX 3080 10 GB); 1000 stopped evictions (XTX) | exact | expert_source.cpp:6914 |
| `STRATA_DUMMY_NODES` per-node cost slope | 0.86 us/node x 2,158 = 1.9 ms/window (RTX 3090) | measurement | verify.cpp:1935 |
| `--spec-min-p 0.7` (+ `--adapt-every 8`) | 0.70 best (RTX 3090); +2.9% story, +9.6% code, -2.7% 16-topic (Spark-NV, RTX 5090) | exact | mtp.cpp:1859 |
| `PREFILL_CPU_SHARE` auto / MAX | auto: 214 tok 2.4x, 620 tok 1.8x (2x Radeon VII) | small-lossy | prefill.cpp:271 |
| no PCIe staging slots at `--pcie-frac 0` (strata-slim 0003) | ~61 MiB/GPU back to cache (CUDA) | exact | 2026-11-15: CODED STRATA_PCIE0_NO_STAGING |
| `STRATA_POOL_GUIDED` shrinking tail ranges | 29.6 to 31.3 tok/s with a CPU fix (Core Ultra 285K) | exact | 2026-11-15: CODED STRATA_POOL_GUIDED |
| Argon AVX2: 2 accumulators, 32/16-row granularity, `row_dot_pair` | no isolated A/B (MI50 + AVX2 Xeon, 27.8 tok/s) | exact | 2026-11-15: CODED STRATA_ROWS_ACC2 |
| `STRATA_IQ_ZEN3` gather-free IQ decode | +0.8% (119.5 to 120.5, RTX 3090 + Zen 3) | ~exact | 2026-11-16: already iq_avx2.cpp:521 iq256_variant (Zen 3 decodes gather-free) |
| DraftPolicy cost-outlier clip (`PL_POLICY_CLIP`) | one 31 ms stall flips MTP T3 to lookup T8, persists (gfx1030, CPU tests) | exact | 2026-11-15: already STRATA_DRAFT_CLIP |
| HotRod weight x logit-gap budget (`ROUTE_BUDGET`) | ranks 3-9: 44.7 vs 46.4 tok/s, 12/12 JSON (RTX 3060) | lossy | 2026-11-16: CODED STRATA_ROUTE_BUDGET / _TAIL, STRATA_STANDIN |
| skip non-resident experts < 10% weight (`SKIP_MINW`) | 20 to 25 tok/s, KL 0.016-0.022 vs floor 0.013 (RTX 5090, GLM) | lossy | 2026-11-15: already STRATA_MISS_SKIP_W |
| balanced chunks (+1/32 rows) | small, not isolated (asahi) | exact | 2026-11-16: already STRATA_PREFILL_EQUAL / _EQUAL_TAIL |
| HotRod fragile-token re-verify + auto budget + brake | 1-3% windows cut; 27/27 clean at 37-48 tok/s (RTX 3060, NVMe) | lossy guard | 2026-11-16: CODED STRATA_FRAGILE_RECHECK (+ BRAKE, K0), STRATA_ROUTE_BUDGET_AUTO, STRATA_STANDIN_LEARN / _APPLY; 2026-11-16: CODED the substitute fit (ridge / low-rank part): STRATA_SUBFIT_LOG, STRATA_SUBFIT_LEARN, STRATA_SUBFIT_APPLY=dynamic\|all\|flip, STRATA_SUBFIT_LADDER, STRATA_SUBFIT_PAIR_MIN_SEEN, STRATA_SUBFIT_GAIN_FIT / _GAIN_REVOKE, STRATA_SUBFIT_SHADOW / _SHADOW_CPU_YE, STRATA_SUBFIT_PROBE_EVERY, STRATA_SUBFIT_RANK / _PAIRS, tools/subfit_fit; spec 99-hotrod-subfit, not run |
| rdna3 unported prompt kernels `1799c0a3a`/`3dbb40516`/`1f4b77d72` | expert GEMM -6..-9%; HC half-layer -7.4..-9.1%; colskip 1-GPU unchanged (XTX) | exact | 2026-11-16: CODED STRATA_HC_WN4, STRATA_NW_COLSKIP |
| `STRATA_VRAM_CALIB` (PR #5691) | +314/+166 slots, free 884 to 300 MiB (2x RX 6900 XT, HIP) | exact | 2026-11-16: CODED STRATA_VRAM_CALIB |
| `STRATA_PROMPT_STAGE_MEMO` | short turns +14-20% (gfx906) | ~exact | 2026-11-15: left |
| disk conversation cache + RAM/disk dual-park | 12,175 tok restore 0.32-0.51 s vs 42-45 s cold (gfx906 box) | exact | 2026-11-16: CODED STRATA_CONV_DUAL_PARK |
| `DQ8` + `HC_GATED16`, `KV_PF_RESIDENT` | IQ3_S prompt +6%; +7-9% at 16-128K (2x PRO V620) | exact | 2026-11-16: CODED STRATA_HC_GATED16, STRATA_KV_PF_RESIDENT; DQ8 already |
| gfx1030 decode: `MMVQ_ROW/SET`, `NORM_ROPE=2`, `HEAD_WIDE/ROUTE_PLAN/HEAD_SAMPLE` | 119.2 to 122.1, one run (+1%) | exact | 2026-11-16: CODED STRATA_NORM_ROPE, STRATA_MMVQ_SET, STRATA_HEAD_WIDE, STRATA_ROUTE_PLAN, STRATA_HEAD_SAMPLE; MMVQ_ROW = RDNA_MMVQ_WAVE |
| `__launch_bounds__` / occupancy audit | QSA value -10.9%, 128K prefill +1.45% (P40) | exact | 2026-11-15: method |
| HotRod prompt stand-ins (`PREFILL_STANDIN`) | 76-118 tok prompts 35-45% faster; 8.5K 15.8 to 15.1 s (RTX 3060) | lossy | 2026-11-16: CODED STRATA_PREFILL_STANDIN |
| `--adapt-async 1` (swaps on own thread) | short chat 24.9-26.0 to 28.7-30.1 (Strata-GLM, RTX 5090) | exact | adapt_async_default.hpp:1 |
| `STRATA_NT_COPY` non-temporal blob copies | DMA 46.6 GB/s alone vs 17.0 beside 4 memcpy threads (Strata-GLM) | exact | 2026-11-16: CODED STRATA_NT_COPY |
| hipBLASLt table auto only on gfx1151 after v0.2.13 (`343b8eae6`) | 131K prefill 926 to 1,687 with right table (XTX community); ours table +38% prompt | rebase hazard | gemm.cu:375 (`STRATA_HIPBLASLT_TABLE`) |
| bimodal: per-run AnonHugePages, `compact_memory` vs `NO_ARENA_THP` | ours: 38 starts "THP NOT requested" (6 GiB free 2 MiB blocks of 33) | method | 2026-11-15: method |
| bimodal: DPM clock ramp | first request -26% (59.3 vs 80.4, XTX) | method | 2026-11-15: method |
| method: median [min-max], `strata_bench.py --runs 3 --long`, sweep calibration, n >= 6 for prompt A/B | cold TTFT CV 71-96% (phoenixclyde) | method | 2026-11-15: method |
| momoe quality protocol (KL + top-token + scored tool tasks) | tool use 95.0 to 52.5% at lambda 1.5 with KL only 0.251 | method | 2026-11-15: method |

### A3. active-forks-nov

| lead | number + hw | exact/lossy | in our src |
|---|---|---|---|
| `STRATA_LOOKAHEAD=0` when RAM complement is whole | window 17.4 to 15.8 ms, 162.2 to 182.9 tok/s (Spark-NV, RTX 5090 + 7800X3D) | exact | generate.cpp:9516 |
| `PREFILL_COPY_WAIT_DEDUPE=1` | warm 4,712 tok 5,928 to 5,340 ms, +11% in a bundle (R9700) | exact | prefill.cpp:3958 |
| `PREFILL_ROUTED_STREAM=1` + GatherGroup fix `b13ffdabd0` | no isolated figure; +10% IQ3_S / +40% Q4_0 (gfx1030, `PF_STREAM_ROUTED`) | exact | 2026-11-15: CODED STRATA_PF_GG_DEFER |
| `STRATA_MMVF_ROWS=1` (+ single-token router) | +3.9% with DENSE_LDS (2x R9700); router 25.8 to 10.2 us (R9700) | exact | arch_defaults.cpp:80 (gfx1151 only); 1-token: no |
| `STRATA_MMVQ_R=2/4` / per-shape `MMVQ_WAVE_ROWS` table | +4.0% decode, 46.8 to 48.6 (MI50, wave64) | rounding vs preset | native_mmvq.cu:2662 |
| `PARK_ASYNC=1` check + `CKPT_MOVE_PARK` port | ~92 ms saved per park (MI50) | exact | 2026-11-15: CODED STRATA_CKPT_MOVE_PARK |
| `STRATA_B6_MMVQ_ROWS` on gfx11 | default on gfx12, no gfx11 data | exact | nv_defaults.cpp:39; native_mmvq.hpp:34 |
| global (not flat) loads for expert weights | 0-9% kernel time, 0/460 diffs (Lamina, RTX 5070 Ti / 5060 Ti); ours flat_load 13,119 to 108 | exact | iq_kernels.cu:30 (build macro `STRATA_HIP_GLOBAL_WEIGHTS`) |
| LDS-only barrier on gfx11 | no number | exact | 2026-11-15: CODED build macro STRATA_HIP_LDS_SYNC |
| DPP / ds_swizzle for constant-mask `__shfl_xor` | ~4 us/token HC (gfx906); shexp down 6.23 to 5.86 us but head +1.6% slower (R9700) | exact | 2026-11-15: already intrinsics.hpp:119-152 (shfl_down variants left) |
| sampler / verify 8 logit loads in flight | argmax ~110 us/round, latency-bound (gfx906) | exact | 2026-11-16: CODED STRATA_SAMPLER_X8 |
| `STRATA_ACTQ_SKIP` | tier-5 bundle +4.0% (RTX 3090), no isolated | exact | 2026-11-15: CODED STRATA_ACTQ_SKIP |
| `copy_rows_from_mapped` loads before stores | 4.7-5.7 to 2.4-3.9 us/launch (RTX 5090) | exact | 2026-11-16: CODED STRATA_ROWS_LOADS_FIRST |
| RouterLookahead submit after publish + skip when resident | part of +6.5% bundle (R9700) | exact | 2026-11-15: CODED STRATA_LOOKAHEAD_AFTER_PUB, STRATA_LOOKAHEAD_SKIP_RES |
| `STRATA_PF_AHEAD` + `STRATA_REFILL_ASYNC` (staged path, <4,096-tok deltas) | agent prefill 512 to 692 tok/s, TTFT 4.97 to 3.67 s in bundle; refill -143 ms (0xluk, 2x RTX 3060, PCIe 3.0) | exact | 2026-11-16: CODED STRATA_PF_AHEAD + STRATA_REFILL_ASYNC |
| `STRATA_EXPDEC_V2` grouped expert kernels | expdec_gu<22> 69.1 to 52.5 us, down 48.2 to 38.5 us, 672 bit-identical (R9700) | exact | 2026-11-16: CODED STRATA_EXPDEC_V2 |
| `STRATA_POOL_DAG=2` | pool 5.197 to 4.788 ms/window; engine neutral (9950X DDR5) | exact | 2026-11-15: CODED STRATA_POOL_DAG=2 |
| launch fusions `FUSE_COMBINE/SWIGLU_DOWN/SHGATE/PLAN/ROUTEBELL/QGEMV` | 2,648 to 2,248 kernels/window, no tok/s (R9700) | exact | 2026-11-16: CODED STRATA_FUSE_* launch fusions |
| deferred / parallel act quant in pool + sample-async | +6.5% / +3.9% bundles (R9700) | exact | 2026-11-16: CODED STRATA_POOL_DEFER_ACTQ |
| hugetlb CPU expert copy + RAM budget fix | IQ2_XS solo +4.9%, lane +7.2% (svgop, CUDA) | exact | 2026-11-15: already pinned.cu MAP_HUGETLB (svgop file-source complement left) |
| two-pass session restore | 198,830 tok (3.05 GiB) restored 3.4-3.5 s (ElXreno) | exact | 2026-11-16: CODED STRATA_SESSION_STREAMED, STRATA_SESSION_TWO_PASS, STRATA_CONV_DISK_ONLY (a3eeb917c4 = our disk tier) |
| `STRATA_CPU_POOL_COST` longest-first | code +6.54%, geomean +1.95% (gfx936 DCU) | exact | 2026-11-15: CODED STRATA_CPU_POOL_COST |
| uniform fewer experts K=9/8 (`EXPERTS_USED`) | K=7/8 +14.4%, 6/8 +18.7%; KLD 0.0254 at K=7; HumanEval flat (RTX 3050 laptop, Qwen3.6 IQ4_XS) | lossy | 2026-11-16: CODED STRATA_Q36_EXPERTS_USED |
| `gdn_gate_kernel` computes only first token | latent (callers pass n_tokens=1) | fix | elementwise.cu:43 (bug present) |
| temperature < 1e-30 / NaN treated as greedy | gap | fix | sampler.cu:869 `<= 0.0f` only |
| bugs 5.5-5.10 (spec-split B flag, pipeline-windows rows, stats-flush crash, iq_embed grid.y, conv-cache publish, JSON watcher) | latent | fix | 2026-11-16: FIXED 5.6, 5.7, 5.9, 5.10 (tests); 5.5 already (verify.cpp:4912) |
| layer-major group scratch `G = m.T*K` ignores group_cap (precondition for LAYER_MAJOR test) | latent | fix | 2026-11-15: already prefill.cpp:7596 |
| `STRATA_ARENA_REUSE` (#3509) + 0xluk GGUF arenas `e2ffeef8da` | startup only | exact | 2026-11-15: already expert_source.cpp shared-arena state word |

### A4. source-commits (2026-11-09) and fork-knowledge-base

| lead | number + hw | exact/lossy | in our src |
|---|---|---|---|
| tool-arg typing (Maya `_declared_type`, StrataFast bool/null) | robustness | exact | ported (2f0cee057, serve/) |
| PleReader::close() resets `max_inflight` | hang fix | exact | ple_reader.cpp:113 |
| `response_format` top-level schema (llama.cpp #28697) | compat | exact | 2026-11-15: CODED STRATA_RF_TOPLEVEL_SCHEMA (serve) |
| quarter-size last pool tasks (`part_row`, basalt 553e9290d) | est. 0.5-1.5% CPU-expert time | exact | 2026-11-15: already STRATA_POOL_TAIL4 |
| down-row prefetch during fready wait (basalt 4e6841265) | basalt down phase -6% | exact | 2026-11-15: already STRATA_POOL_TAIL4 |
| SpinWait (pause 50 us then yield) in MTP draft wait | est. 0-1% | exact | 2026-11-15: CODED STRATA_MTP_SPIN_US |
| DraftPolicy cost = shared scale x per-size base | 0-2% | exact | 2026-11-15: CODED STRATA_DRAFT_SCALE |
| commit queued behind MTP draft launch (basalt 1fd53a447) | idea | exact | 2026-11-16: CODED STRATA_DRAFT_EARLY |
| file-cache floor rule (>= 1 chunk + 2 layers, >= 4 GiB) | design for RAM_KEEP/mmap tier | n/a | 2026-11-15: method |
| key saved sessions by lossy routing switches (llamAmpere) | correctness | n/a | conversation_disk_env.def (KeyIfSet scheme) |
| Qwen official sampling presets (thinking T 1.0 / top_p 0.95 / top_k 20; non-thinking T 0.7, presence 1.5) | fact; makes FK-01 relevant if opencode keeps T 1.0 | n/a | n/a |
| ggml vendoring + STRATA_GEOMETRY rebase hazard (upstream 067b4dda5 etc.) | hand-port only | n/a | 2026-11-15: method |
| this-cute-world `FUSE_SHGATE`; dardeaw exact wave32 QSA decode kernels | no number | exact | 2026-11-16: CODED STRATA_FUSE_SHGATE, STRATA_QSA_WAVE32 (dardeaw) |
| fork-KB refresh `tools/forks/fork_kb.py` weekly | method | n/a | n/a |

### Closed questions (conclusions that close a lead; do not re-queue without a new reason)

- `--pcie-frac` > 0 on the 7900 XT: closed. Ours: 0 is faster, +6% chat / +13% edit. This parks every PCIe-share lead on the XT: `PCIE_USAGE`/`PCIE_KEEP`, `FETCH_ADMIT`, `HIP_POST_SPLIT`, `DMA_FLAG_WRITE`, fill-stall in `fetch_blobs` (skipped under `pcie_off_`, verify.hpp:459). [SI:166; HN:124,218]
- The strata-manager warning "never `pcie_frac 0` on AMD" is about gfx1201 hanging at boot. It does not apply to gfx1100. [SG:72]
- PCIe link generation is contested. The SG report reads 16 GT/s x16 in sysfs for `0000:12:00.0`. MOD_PLAN 1118 and 1574 say the 5700G forces PCIe 3.0. Cezanne APUs are Gen3, so the 16 GT/s reading is probably the card's internal switch link [derived, unverified]. Check the root-port link with `lspci -vv` before relying on either. [SG:3,9; MOD_PLAN 1118, 1574]
- Expert look-ahead / next-layer prefetch / L3 arena warm on RDNA3 + DDR4: closed. Ours: recall 19-21%, RouterLookahead -13%. android-imdad: -3-4% (XTX). Ben1078 graph-branch gap fetch: -23%. q4-direct-io: "moves the wait, does not remove it". [SI:167-168; SG:64; PU:24]
- Graph branches (fork/join) inside a HIP graph: closed. They cost 26-30% in the engine (unimatrix099, XTX). This also argues against `SH_STREAM` [inference]. [SI:169; HN:133,144]
- Window-sizing controllers (`SPEC_COST`, cost policy, adaptive depth, `ADAPTIVE_SPEC`, `SPEC_KEEP`, `--spec-policy cost`): closed as flat or noisy. `--spec 3/5/6` alone is flat on ours. [SI:170-171; SM:85]
- DFlash2, draft trees, deeper chains, eager/ungated draft launch, n-gram replacing MTP: closed. A 4-bit MTP head is slower too (69.60 to 67.88, RTX 4070). MTP depth 3 loses -6.4% vs 2 (serboox). [SI:172-175; SG:64; AF:81]
- `HC_Q8` on HIP: closed (ours 107.2 vs 111.7). Upstream `TSUM`, int8 WMMA attention, expert layouts, `PF_WMMA`, `IQ2S_PAIR`: neutral or worse. `STRATA_DEQUANT_WIDE`: 8-14% slower on gfx1100 (dense dequant already ~1.6 TB/s). Dense GEMV retune: +0.6% noise. [SI:184-185; HN:35-37]
- `GPU_MAX_HW_QUEUES=8`: every prompt 10-20% slower on XTX. Ours 4 vs 6: flat. [SI:186; HN:172]
- mmap expert tier default, FS_SLOTS, VRAM lending, per-layer budgets, ClassLRU: closed (ours). The zh60k draft vocab is worse; full vs en is flat. [SI:187-188]
- Whole-model IQ2_XXS experts: KL 0.47 vs 0.042, closed. CPU 4-bit repack: no faster on a byte-bound pool. Activation sparsity: FF-640 experts are dense. [SI:179-181]
- Pool parking/placement (futex, SMT lanes, CCD split, settings sweep): closed (ours: defaults best). Window-final-sync polling and device spin flags on AMD: closed. [SI:182-183]
- VRAM reserve: do not go below ~700 MiB on a 20 GB card (svgop "hard stall"). Speedy's 444 MiB was tuned on 10 GB. [SI:193; MOD_PLAN 3961]
- `STRATA_COMMIT_BATCH` has no effect. A forced 4 KB arena on gfx1100 adds 5-8 s to start-up. [SI:194]
- int8 paths on gfx1100 save bytes, not math: `v_wmma_i32_16x16x16_iu8` runs at the FP16 WMMA rate. So int8 verify projections and int8-MMA attention ports give ~0 compute gain on RDNA3. [SI:68,95]
- A HIP graph node costs ~2.8 us on gfx1100 (2.2 us without the acquire fence; HIP has no switch for it). So fusions and grid trims pay more on the XT than on sm_86. [SI:68; HN:136]
- The kernel ceiling on the XT is about +2-5% decode for all HIP kernel work combined. `combine_cpu_k10_vec4` is 38.3% of GPU kernel time (waiting on CPU rows). rxr2 puts mmvq at 70-100% of VRAM BW. The fusion ceiling is ~5% (unimatrix099). [SI:56; HN:253-256; AF:73]
- Rule of thumb: cutting CPU expert bytes by X% buys ~0.46?X% on edit windows and ~0.21?X% on chat. A perfect draft stop is worth at most ~5-6% of window time. [SI:7,9]
- CPU kernels are at ~80% of DDR4 BW (5700G pool 32.4 GB/s, strateu). 24 threads halve decode vs 8. AVX-512 / VNNI / VBMI / NUMA kernels do not apply (no AVX-512 on Zen 3). Zen 3 has fast pdep, so `GGML_PDEP` is moot. There is no WAITPKG, so `POOL_UMWAIT` is moot. IQ prefetch distance is flat (35.28/35.24/35.15/35.23). `STRATA_Q2_PREFETCH` is neutral on ours (so gputier's 2 KB Q2_0 prefetch is closed). [SG:86; AF:81; MO:131; MOD_PLAN 219, 3950]
- Experts on the iGPU (Vega in the 5700G) lose. On an Iris Xe, ggml-vulkan dropped 30.5 to 10.9 tok/s, because an iGPU call costs ~2 ms and the iGPU shares DRAM BW and package power. [AF:81]
- `STRATA_POOL_DAG=1` is not new: it equals pool.cpp:1309 mode 7 (`POOL_FUSE_PHASES=1`, ours +2.5% edit). Only DAG=2 is new, and it is engine-neutral for its author. [AF:9,41]
- `MMVF_ROWS` is not on by default on HIP. arch_defaults enables it only on gfx1151, so it stays a valid gfx1100 env test. [AF:9]
- `FUSE_DECODE` on HIP is inert with `--native`. [MOD_PLAN 1574]
- The prompt cache root not being set below 2,048 tokens (active-forks bug 5.1) is intended behaviour, not a bug. [MOD_PLAN 3936; AF:50]
- The KV-stream unbounded `miss_block` write (kllmagn) is already fixed in our tree (kv_stream.cu:110). [AF:51]
- #5655 route-prior NaN guard: present (route_prior.cu:55-61, 84-127). The SG report's "no equivalent in src/" is stale. [MOD_PLAN 3016; SG:68]
- MMQ rows not a multiple of 256 ("!!!!" answers): the row-margin fixes are in HEAD (14dd746f, d7f7d967). [MOD_PLAN 3929, 3034; SG:72]
- FK-28 `ADAPT_PROMPT` is dead code. `set_usage` has no callers and no env read, so there is nothing to A/B until it is wired. [MOD_PLAN 3928]
- The 2026-11-10 "05" screen did not measure `LOOKAHEAD=0`, routed-dedupe, `MMVF_ROWS`, `GDN_PP=2` or `spec-min-p 0.7`. It had no clean base (n=2, intervals +-100-450 tok/s), so they stay open. [MOD_PLAN 1574, 1751]
- `STRATA_RDNA_MMVQ_WAVE=3` red flag (rdna3 issue #4): answered. Ours: WAVE=1 vs 3 gives short -4.5%, edit -2.5%. Keep 3. [MOD_PLAN 3933]
- `STRATA_LEND_RAM_BACKUP` (+11.8%, R9700) does not apply. Our arena holds all experts in pinned RAM, so lent slots already have a RAM copy. [B2:53-56]
- SSD/page-cache-tier leads do not apply: arena fully in RAM. This covers `COLD_SKIP*`, `ALT_EXPERTS`, `DEFER_COLD`, `PREFILL_UNMAP`/`DECODE_UNMAP`, `MIRROR_*`, HotRod learned look-ahead, and q4-direct-io. [MO:104,112; SI:28; SG:64]
- `PF_PIECE=8192` (+13-16% vs 8K chunks) is moot. We already read and compute with 32K chunks and do not need 8K bit-identity. [HN:173-174]
- `PA_DOT4` (RDNA2 int8 prompt attention) is probably 0 on gfx1100, which has WMMA. `PF_GU_EXACT` targets an exact rocBLAS-order prompt path. Ours is already lossy (PF_GEMM, hipBLASLt), so only its speed, not its exactness, matters. [SG:60; HN:164]
- 2-GPU-only leads are parked: `--pipeline-windows 2`, `PL_LOOKUP_STREAK`, `PW_RES_PATCH`, layer split, kfdq, peer, TP, `REFILL_SDMA`, CUDA helper. 3060/CUDA-only leads are parked: FK-05, FK-06, FK-11, FK-25, FK-29, FK-56, sm86 HC read set, `MMVQ_IL` tables, poly `FATTN`/`FDEC` (poly is CUDA-only). [SG:64; AR:51; MOD_PLAN ~3605]
- Miss-targeted expert dropping and per-layer SER tables are dominated by uniform top-k at equal quality (lighttransport, MiniMax). Masked pruning lost MMLU 92.9 to 78.6% (svgop). Lossy routing gates need scored tasks, not KLD alone. [SI:177; AF:81]
- The hugepage hypothesis for the ~25% bimodal runs is not proven. serboox: 4 KB vs 2 MB pages give 52.4 vs 53.2 GB/s on a similar random-block pattern. It may explain part of the gap, not 25%. Reserving `nr_hugepages` without svgop's budget fix caps the complement (19.80 of 37.97 GiB). [AF:67,83]
- Upstream since base 146df3c46 has nothing that speeds Flash-Next on gfx1100. Avalon's 11-09 prompt-path batch is flat on the 7900 XT (1824.8 to 1833.8). Basalt Q6_K staging falls outside Flash-Next shapes. Upstream retired DF2_GATE (close ledger). [SC:25,129-137,98]
- Upstream #5327 arena file-cache load does not apply. It is disabled under a cgroup limit and adds memory pressure. [SC:86]

---

## Part B: leads from engine-speed/{grouped-technologies, projects-derived, pain-points, code-deep-research, moe-speedup}

PP = pain-points/report.md; PPe = pain-points/notes/engine.md; CD = code-deep-research/report.md; MS = moe-speedup/report.md;
MSr/MSt/MSc/MSx/MSe = moe-speedup/notes/{rdna3_gpu_decode, transfer_and_kv, cpu_expert_compute, expert_caching_prediction, engines_labs_companies}.md.
"in our src" = `git grep` over `src include` (serve/ checked for server items, marked `serve:`); "-" = no greppable name, checked by concept only.
"in plan" = first relevant MOD_PLAN.md line(s); "no" = no hit by ID, switch or concept. Hardware: 7900XT = our RX 7900 XT gfx1100 + 5700G DDR4-3200 + PCIe 3.0.
Dropped as irrelevant (2-GPU / multi-user / non-HIP hardware): Strata-DualGPU + #4522/#4534/#4549/#859, rdna3 tensor-parallel/RCCL/kfdq, StrataFast `--prompt-split`, StrataQ38FN multi-user scaling, XeStrata/#3607/#5540 SYCL, Gaudi2, #2392 (gfx1101 table), #4876/#3277 (sm_75), #4854 (AVX1), #4886/#5647/#5644 (Metal), #5592 (Jetson), #5453/#5472 (DGX Spark UMA), Infernix NVFP4, `.basalt` pack (sm_120), `--attn-split` (CUDA tensor-core), DeepEP/EPLB/AFD/MegaScale (data-center), poly fused FA, #5467 (mma.m16n8k16, NVIDIA PTX).

### 1. grouped-technologies (F1-F10 family map; overlaps pain-points)

| # | lead | number + hw | exact/lossy | in our src |
|---|---|---|---|---|
| B1 | CPU-expert activation sparsity masks (Berkeley gate / TEAL / CATS / CHESS / SpAx), item 13 | 84.5-87.4% sparsity Qwen3.5 MoE (paper); ceiling 1.25x, est. +10-20% | lossy | 2026-11-16: CODED STRATA_CPU_ACT_SPARSE |
| B2 | Union sparsity over verify window w=1/2/4/8 (BlockFFN) column mandatory | 80% -> 71% at w=8 (Asia) | lossy | 2026-11-16: CODED STRATA_CPU_ACT_SPARSE_UNION |
| B3 | turbo-hip `STRATA_ROUTE_PRIOR 0.5` (Cache-Conditional Experts) | 149-169 vs 128.7 tok/s, 7900XT; 2.89% mismatch 3060 | lossy | src/core/conversation_disk_env.def:36 |
| B4 | AcceptMoE depth-graded prior, pos 0 exact (item 23) | 2.06x vs SD under offload, RTX 5090, -0.27 pp; est. 5-15% over fast-hip | lossy | include/strata/kernels/route_prior.hpp:12 |
| B5 | OEA sibling-aware bonus in verify window | idea, low ceiling | lossy | 2026-11-16: CODED STRATA_OEA |
| B6 | Tail skip `STRATA_ROUTE_TAIL_SKIP 7` / tail-row drop w/ MTP confidences (item 5) | 129.7 tok/s 7900XT (skip) | lossy (skip) / exact (row drop, per plan) | include/strata/core/expert_source.hpp:670 |
| B7 | ISTA Coder variant 256/512 experts (item 25) | 29.6 GB resident; one-evening A/B | different model | - |
| B8 | Admission on k-th miss `STRATA_ADAPT_KMISS` (item 14) | sim +5..16% edit; replay hit 76.7->86.5% edits | exact | include/strata/core/epoch_bandit.hpp:9 |
| B9 | `--adapt-every 1 --adapt-swaps 64/96/160` tier at every window | replay 79.6% -> 85.0% VRAM hits; sim +4..9% | exact | include/strata/core/adapt_async_default.hpp:24 |
| B10 | Keep decode cache through long prompts / lend sweep (item 2) | 9,412 slots 12.6 GiB borrowed; 125->105 tok/s; +20% after-long turns | exact | 2026-11-15: already PREFILL_LEND_PCT + TIER_REWARM |
| B11 | SIEVE / S3-FIFO, OpenEvolve/ADRS, Hawkeye, Harmonia, MoE-Infinity as cache policy | Belady-bounded; only if sim >=5% | exact | 2026-11-16: CODED STRATA_ADAPT_SIEVE |
| B12 | ProMoE planner / reordered execution (cache -> in flight -> CPU) | 1.09x over LRU, PCIe 4.0 | exact | - |
| B13 | Next-expert predictors (Fate/HOBBIT/SmallThinker/SP-MoE) recall test | local recall 19-29%; ProMoE 1.09x | exact | - |
| B14 | Seeds `STRATA_DOMAIN_PROFILES` / `STRATA_CONV_WORKSET` (items 3-4) | +0.6-1.5 hit pts | exact | include/strata/program/expert_seed.hpp:7,13 |
| B15 | Expert pre-staging during tool runs (KVFlow, item 17) | sim first | exact | - |
| B16 | `--pcie-frac 0` default on gfx1100 | mean 111.7->118.4 (+6%), after-long 92.5->104.3 (+13%), 7900XT | exact | include/strata/core/calibrate_ratio.hpp:14 |
| B17 | Fiddler token-count threshold for `STRATA_PREFILL_CPU_SHARE` | code reading | exact | include/strata/core/arch_defaults.hpp:30 |
| B18 | Expert deferral `STRATA_VERIFY_DEFER_CPU` (KTransformers) + HIP non-finite bisect (item 8) | +33% / up to 1.45x DeepSeek-V3 Xeon; est. +10-25% edit; HIP gives NaN logits | lossy (small) | src/core/conversation_disk_env.def:42 |
| B19 | Rank guard `STRATA_VERIFY_DEFER_CPU_TOP` (item 28) | keeps +3-5% edits (plan A4) | lossy (small) | include/strata/core/defer_auto.hpp:91; env.def:44 |
| B20 | Per-expert pool barriers + fused SiLU*mul `STRATA_POOL_FUSE_PHASES` (item 29) | 1-4% | exact | include/strata/kernels/cpu/pool.hpp:382 |
| B21 | `STRATA_POOL_PREREAD` (tandem.cpp) | A/B queued | exact | include/strata/core/epoch_bandit.hpp:10 |
| B22 | Athena epoch bandit for pre-read / mixed opt-ins | - | exact | include/strata/core/epoch_bandit.hpp |
| B23 | DISCO learned chain stop (item 15/36, `feat/disco`) | 0-10% combined | exact | include/strata/core/row_miss.hpp:2 |
| B24 | Drafter entropy (MiMo, Entropy Sentinel) as DISCO feature | - | exact | 2026-11-16: CODED STRATA_DISCO_ENTROPY |
| B25 | SpecFit / DraftExpert / EcoSpec miss-aware window (new experts per position) | DraftExpert verify-cost 2.35x->1.48x, RTX 4090; est. +3-8% over SPEC_COST | exact | 2026-11-16: CODED STRATA_SPEC_MISS_WIN (+ VERIFY_ECAP) |
| B26 | Cross-session suffix index (SuffixDecoding) | `--simulate` first; A24 spec 45 | exact | include/strata/spec/suffix_index.hpp:114 |
| B27 | `--lookup-max-t 12/16` | +1% first test | exact | include/strata/core/lookup_default.hpp:1 |
| B28 | MTP draft inside captured verify graph (item 10) | 1-3% est. | exact | 2026-11-16: CODED STRATA_MTP_IN_VERIFY (+ MTP_ONE_GRAPH, DEVICE_ACCEPT) |
| B29 | `--mtp-q4` | queued | exact? | include/strata/core/mtp.hpp:64 |
| B30 | ReplaySSM / SpecLA audit of GDN replay (item 20) | 1-3% decode | exact | - |
| B31 | SSD-lite (Saguaro): draft while GPU verifies | CPU already busy | exact | - |
| B32 | Restore-equivalence test (UNSW, item 35/83) | gate for 3/11/16/31/39 | test | src/core/conversation_disk_env.def:333 (`STRATA_CKPT_REREAD`) |
| B33 | `preserve_thinking` end to end (item 31) + PR #5643 restore dropped reasoning | ~8% turn time is same-session re-reads | exact | serve:chat_template.jinja:119 |
| B34 | Message-boundary checkpoint `STRATA_CACHE_MESSAGE_BOUNDARY` + `--prompt-cache-tail` (item 11) | first edit 17.24->11.53 s (-33%), author hw | exact | src/core/conversation_disk_env.def:321 |
| B35 | Tool-boundary checkpoints + Marconi FLOP/byte eviction (item 16) | Marconi TTFT -71.1% SWE-bench traces | exact | - |
| B36 | RadixAttention multi-state store / Continuum TTL (item 3/71) | 5.4-6.1 GiB for 4 parked convs | exact | - |
| B37 | Disk tier `--conversation-disk` on main engine (item 39) | 70K resume 7.2 s vs 83 s (FastStrata, 3060) | exact | include/strata/core/conversation_cache.hpp:483 |
| B38 | `env_to_user`, effort at end, "tools first" prompt reordering (item 88) | ~14K tok / 8-11 s per day/folder (comment) | exact | serve:frontend.py:405; serve:runconfig.py:32 |
| B39 | Thinking caps `reasoning_budget_by_effort` + grace 512 + `reasoning_close_retry` (item 12) | 15 turns cut cost 2,726 s; p99 ~4-5K tok | lossy | serve:server.py:5130; include/strata/program/think_probe.hpp:21 |
| B40 | Learned thinking stop probe (Thought Calibration, item 21/A19) | up to -60% thinking tok (paper) | lossy | serve:test_think_probe.py |
| B41 | Rumination 5-gram signal (item 34) | - | log-only then lossy | serve:rumination.py |
| B42 | GDN prefill chunk 64 + 16x16 block inverse (item 18) | 5-11% prefill; V100 chunk-64 attempt 4-6x slower | exact (bits change) | include/strata/prefill/gdn_chunk.hpp:52; 2026-11-17: premise read, NOT coded (MOD_PLAN.md item 49, T-746: sm_70-only file, host model already T-103) |
| B43 | BFLA as one shared QSA selection per query tile; `STRATA_SEL_OVERLAP` first (item 19) | 128K: 386.9->1,219.8 tok/s Qwen3.6 (q36 engine); Flash-Next QSA 2,051 cells/query | lossy | src/core/conversation_disk_env.def:1002 |
| B44 | Q2_0 expdec down kernel at n_ff 640 (`native_down_multi_kernel<42>` = 5.79% GPU) | est. 0.3-0.6 ms short window (~1-2%) | exact | src/kernels/cuda/iq_kernels.cu:2295 |
| B45 | Gated Residual small-M Mix/Combine kernels M=1-8 (item 32) | 1-5% short | exact | - |
| B46 | UFMG coordinate-descent launch/tile sweep (item 37) | 1-3% | exact | - |
| B47 | Group A arch switches bisect one at a time | all on: 101.2 mean, 80.3 after-long, 7900XT (worse) | check each | include/strata/core/sm120_series.hpp:7 |
| B48 | BitDecoding WMMA decode attention `STRATA_ATTN_WMMA` | ~2-3%, attention ~3% of decode | exact | include/strata/kernels/qsa_decode_attn_wmma.hpp:2 |
| B49 | Revised quality gate (token forcing filter + fixed task set); KLD/top-1 rho=+0.00 with task score | Displacement Is Not Direction, Qwen3.6-A3B | method | tools/ (task_gate.py) |
| B50 | `reasoning_tokens` in chat usage + per-turn JSONL log (items 87/73) | gates 11/12/16/31 | n/a | serve:server.py:8399, 5086 |
| B51 | `pancake.service` with `Conflicts=`, port 11437, own MemoryMax, NOCAP (item 91) | restart 28-65 s | n/a | - |
| B52 | `MADV_SEQUENTIAL`/`WILLNEED` at load / #5561 `STRATA_EXPERTS_SEQ_READAHEAD` | later turns 65-76 s -> 23-28 s, RX 6900 XT 32 GB | exact | 2026-11-15: CODED STRATA_EXPERTS_SEQ_READAHEAD |

### 2. projects-derived (forks and upstream PRs)

| # | lead | number + hw | exact/lossy | in our src |
|---|---|---|---|---|
| B53 | PR #5655 `route_prior` sentinel id leak / OOB on degenerate logits | correctness | exact | src/kernels/cuda/route_prior.cu (guard present) |
| B54 | #5609 q36 MTP block loads but never drafts; verify in base | correctness | exact | - |
| B55 | llama.cpp #31751 one-pragma MMQ J=32 register-spill fix on RDNA3 | +36% at 17-32-token batches, gfx1100 | exact | 2026-11-15: CODED STRATA_MMQ_RDNA3_J32_ROLLED |
| B56 | #4625 `STRATA_POOL_SPIN_US=0` really means zero | trivial | exact | include/strata/kernels/cpu/pool.hpp:291 |
| B57 | #5466 need/free MiB on expert-buffer/staging OOM | diagnostics | n/a | - |
| B58 | #5518 HIP `alignas(16)` int4 staging compile fix | build | n/a | - |
| B59 | #4005 (Vop3p) prefill sparse copy events + slot-wait dedup | Windows HIP prompt +36% @4K, RX 6900 XT | exact | 2026-11-15: already prefill.cpp COPY_EVENT_EVERY / COPY_WAIT_DEDUPE |
| B60 | #5468 layer-major super-chunk prefill (stage a layer's experts once per N chunks) | q36 only, author | exact | - |
| B61 | #5595 grouped Q4_K down-expert kernel (q36) | q36 only | exact | - |
| B62 | StrataGP 16 packaged patches | +5-15% decode, +42-49% prompt, RTX 5090 | mixed | - |
| B63 | Juud_engine: shorter worker spin after expert batch + skip activation quant for GPU-resident tokens | +9.6-15.6% decode, RTX 4090 | exact | 2026-11-15: CODED STRATA_ACTQ_SKIP_NOCPU (+ POOL_SPIN_US) |
| B64 | #5475 `STRATA_HC_PERSIST_Q8` int8 persistent HC read | - | exact? | 2026-11-16: CODED STRATA_HC_PERSIST_Q8 |
| B65 | #5535 int8-WMMA prompt attention on gfx1100 with recipe | attention 6.7% of 42K prompt | exact? | - |
| B66 | #5471 faster Q4_K/Q5_K decode GEMVs (DS4 path) | - | exact | - |
| B67 | #3925 q36 load of K-quant/IQ4_XS `ssm_alpha`/`ssm_beta` | pack coverage | exact | - |
| B68 | #3509 `STRATA_ARENA_REUSE=1` shared arena across restarts | restart 28-65 s on 33 GiB | exact | 2026-11-15: already expert_source.cpp shared arena |
| B69 | Maya parallel direct expert reads `MAYA_EXPERT_DIRECT`/`STAGERS` (2-16 O_DIRECT threads) | +26% decode RAM-poor PC | exact | 2026-11-16: CODED STRATA_STAGER_PROBE (+ open_direct, STAGER_THREADS) |
| B70 | #5516 in-place TUNE + conversation-cache routing by traffic shape | - | exact | - |
| B71 | Maya weight-based miss skip `MAYA_GLM_MISS_SKIP_W` | +7-13% decode at KL 0.037-0.052, GLM | lossy | 2026-11-15: CODED STRATA_MISS_SKIP_W |
| B72 | #5566 opt-in cold store for expert arena | memory elasticity | exact | - |
| B73 | #5623 immutable host KV prefixes shared between sessions | 5.4-6.1 GiB parked | exact | - |
| B74 | basalt sleep-in-place `--idle-sleep` (64167ba73, ~815 lines) | - | exact | 2026-11-16: CODED --idle-sleep (acts on CUDA only; ROCm has no process checkpoint: prints unavailable) |
| B75 | Trellis "S" 2-bit quant (#3384, alesha-pro) evaluate offline | - | lossy | - |
| B76 | Maya leftovers: `--calibrate` startup sweep, drafter auto-keep | - | exact | - |
| B77 | Maya prompt memo | ~6% of follow-up TTFT | exact | - |
| B78 | #5593 `STRATA_PROMPT_DIRECT=1` without a RAM budget | - | exact | src/core/conversation_disk_env.def:639 |
| B79 | #4777 per-request `route_prior`; #3971 keep thinking past cut; #5628 stream tool args; #5512 thinking skip | - | mixed | - |
| B80 | #5637/#5437 inhibit sleep during inference | - | n/a | serve: keepawake.py |
| B81 | Private_Context_COW copy-on-write context (#3865) | parked RAM 5.4-6.1 GiB | exact | - |
| B82 | StrataQ38FN agent-switch re-read patch | 25 s -> 1.5 s re-read, author (4x3090/5090) | exact | - |
| B83 | Upstream v0.2.13 / engine 0.1.59 (shape-engine for Qwen4, signed geometry manifest) affects rebase | base 146df3c4 one release behind | n/a | - |
| B84 | #5545 GLM-5.3-Flash engine path (`strata-glm`) | - | n/a | - |

### 3. pain-points (engine + server/ops), not already above

| # | lead | number + hw | exact/lossy | in our src |
|---|---|---|---|---|
| B85 | Edit-window budget: 37.4 ms = 15.8 GPU + 17.4 CPU experts + 2.5 draft; 357 CPU experts/window at ~27 of 33 GB/s | 7900XT fast-hip | fact | - |
| B86 | `combine_cpu_k10_vec4` spin = 38.3% GPU kernel time; `wait_flag_copy_i32` 4.1%; 395 kernels/token | rocprofv3, 7900XT | fact | src (combine_cpu) |
| B87 | Profile on edit window + after long prompt (item 1 extension) | - | diag | - |
| B88 | Fixed per-turn overhead ~0.5-0.9 s breakdown (`t_parse/t_render/t_encode`) | Python est. 60-300 ms | diag | - |
| B89 | Hugepages 16,920 x 2 MiB (owner) | unmeasured | exact | src/core/pinned.cu:110 (THP) |
| B90 | DDR4-3600 / FCLK 1800 + STREAM read test | up to 1.1-1.19x CPU part (est.) | exact | - |
| B91 | Deterministic test mode (fixed profile, adapt off) + load-gated A/B distributions | ~20% run-to-run variance | method | tools/pancake/abtest.py |
| B92 | `strata-coder` sub-agent model/context mismatch (163,840 vs 131,072) | - | config | - |

### 4. code-deep-research (items 74-95b; all in plan lines 389-500)

| # | lead | number + hw | exact/lossy | in our src |
|---|---|---|---|---|
| B93 | 74 MTP draft eager launch (`STRATA_MTP_EAGER`), min_p gate on device | 0.3-0.5% est. | exact | src/core/conversation_disk_env.def:494 |
| B94 | 75 persistent adapt worker (`STRATA_ADAPT_PERSIST`) | 0.1-0.3% est. | exact | env.def:292 |
| B95 | 76 `hipDeviceScheduleSpin` A/B (`STRATA_DEVICE_SPIN`) | 0.2-0.6% est.; ~3% polling cost on CUDA | exact | env.def:354 |
| B96 | Deferral turns off `ROUTE_PUBLISH` (verify.cpp:2355): +48 launches/window bias in 8/28 A/B | - | note | include/strata/core/defer_auto.hpp:16 |
| B97 | 78 staged prefill path for <4096-token agent deltas (CPU_SHARE 0.6 + MAX 4096; per-layer sync) | 4K 1,757 vs 32K 2,580 tok/s, 7900XT; est. 0-15% of re-reads | exact | 2026-11-15: CODED STRATA_PF_AHEAD + STRATA_REFILL_ASYNC |
| B98 | 0xluk `STRATA_PF_AHEAD` + `STRATA_REFILL_ASYNC` on the staged path | agent prefill +35%, TTFT 4.97->3.67 s, 2x3060 PCIe3 | exact | 2026-11-16: CODED STRATA_PF_AHEAD + STRATA_REFILL_ASYNC |
| B99 | 79 PLE gather at request admission, multi-threaded decode | 3060: 10.6% GPU timeline at 42K; est. 3-10% cold 16-32K | exact | - |
| B100 | 80 hipBLASLt descriptor cache + tuning table beyond T=8192 | est. 7-20 ms host/chunk | exact | - |
| B101 | 81 GDN scan 48 WG on 84 CUs: split value columns | - | bits change | src/prefill/kernels.cu:1809 |
| B102 | 82 `STRATA_PREFILL_TIMING=1` + `STRATA_TRACE=1` run on 7900XT | never run | diag | include/strata/prefill/timer_marks.hpp:1 |
| B103 | 38 Linux O_DIRECT in `load_experts_gguf` (16 readers, FADV_DONTNEED stop-gap) | 31-41 s -> est. 12-15 s; likely oomd cause | exact | 2026-11-15: already STRATA_UNBUFFERED_LOAD |
| B104 | 86 deterministic conversation-cache RAM budget + PSI-paused readers | cache 4,096-6,144 MiB today | exact | env.def:964 (`STRATA_PSI_GUARD`); CODED 2026-11-16: `STRATA_PSI_SHED` (Speedy C4, basalt #113), opt-in |
| B105 | 85 THP collapse in background, pipelined registration | Windows/5070 -10% decode | exact | src/core/pinned.cu:110 |
| B106 | 89 grammar on Anthropic/Responses paths | real bug | exact | serve:server.py:10057, 10122 (fixed) |
| B107 | 90 `pin_sibling` threshold (`STRATA_PIN_PARK_TOKENS`) before Qwen auto-pin | full re-read of 19-41K parent | exact | src/program/generate.cpp:17642; env.def:582 |
| B108 | 92/93/94/95/95b build matrix, tests dir, flags.def registry, main() split, stubs | 872 STRATA_* names, 18,361-line main | n/a | tools/pancake/build_matrix.sh; include/strata/prefill/moe_fused_iq.hpp:47 |
| B109 | dan-99 list: drop trailing `wait_parked` (pool.cpp:1373) | est. 0.2-0.5 ms/window | exact | include/strata/kernels/cpu/pool.hpp:308 |
| B110 | dan-99 list: `row_multi<1>` FMA chain split, per-window getenv, serial activation quant, scalar exp SwiGLU | 0-3% each | exact / not bit-identical (FMA) | 2026-11-17: CODED default off, host-tested, NOT RUN on the box: `STRATA_POOL_MICRO_Q2FMA2` (two accumulator chains in the one-token Q2_0 row, rounding class) and `STRATA_POOL_MICRO_SWIGLU` (8-lane exp, one-token IQ512 gather site only), T-716, specs 137 / 138 (MOD_PLAN.md item 40); serial activation quant = `STRATA_POOL_DEFER_ACTQ` (already coded); per-window getenv: not found in the one-token path (reads are once-static) |
| B111 | dan-99 list: MWAITX instead of pause-spin | 0-3% est. | exact | src/core/conversation_disk_env.def:615 |
| B112 | Fusion targets: swiglu+quant as `expdec_gu` epilogue (~49 launches/window), router GEMV + `route_publish_res` | - | exact | - |
| B113 | 84 seeds inert under KMISS: virtual miss bits | placement | exact | - |
| B114 | 53 ForesightSwap stuck slot on failed copy: generation stamp | bounded | exact | - |
| B115 | Loop-guard `collapse` rewrites history -> deep re-read each turn while active | perf risk | exact | serve:loop_guard.py |
| B116 | Session files keep only deepest checkpoint (root lost after restore) | matters for 39 | exact | include/strata/core/conversation_file.hpp:189; CODED 2026-11-17 as `STRATA_SESSION_KEEP_ROOT=1` for SAVE / QUIT (the disk tier already kept root + deepest): T-734, spec 145 queued T-735, MOD_PLAN.md section 6 item 44, nothing measured |

### 5. moe-speedup (window model + literature)

| # | lead | number + hw | exact/lossy | in our src |
|---|---|---|---|---|
| B117 | `STRATA_SPEC_COST=1` for gfx1100 (EVICT rule) | mean flat 112.0, after-long 92.5->98.3 (+6%), 7900XT | exact | include/strata/core/mtp.hpp:146 |
| B118 | Re-warm / protect decode tier after prefill (prefill-routing seeded bulk swap) | after-long 92.5 vs 111.7; est. +1-4 hit pts | exact | 2026-11-15: CODED STRATA_TIER_REWARM |
| B119 | One pinned pool worker per physical core, spin waits (`--pool-affinity`) | <5%; 12 workers worse (99 vs 112) | exact | src/dsv41/ds4_os.hpp:581 (DS4 only) |
| B120 | Reuse-aware admission with H2D DMA scheduled into CPU-idle phases (attention/GDN) | est. +10-18% over Tier 1; SeqMoE +8.5 pts vs LRU | exact | 2026-11-15: CODED STRATA_ADAPT_IDLE_COPY |
| B121 | Exact tail-row truncation mid-window (novel) | 23% of misses only for last row, 47% last two (local) | exact | - |
| B122 | CPU pool bandwidth push: 2 fused phases/layer across all missed experts, row-chunk work split, touch next expert early | 33 -> 38-42 GB/s est., +5-12% | exact | 2026-11-15: CODED STRATA_POOL_ROW_SPLIT / TOUCH_NEXT |
| B123 | Row-interleaved R4/R8 repack of CPU arena (ik_llama `_R4`) | 5-15% CPU part (qualitative) | exact | 2026-11-16: CODED STRATA_CPU_R4 |
| B124 | Fused T-row GDN verify kernel with lazy commit (FlashInfer gated_delta_rule_mtp / ReplaySSM) | write only final state + per-token k/v/g; small unless T full-state writes today | exact (sequential form) | - |
| B125 | Prefill: bigger chunks / decode tier resident during prefill (SLP, MoE-Gen) | est. 1.3-1.8x prefill at 16-32K; ~75 ms saved per resident GB per pass | exact | - |
| B126 | SeqMoE learned multi-step forecaster + probabilistic-Belady eviction | 96.97% vs 88.50% LRU at 45% residency, RTX 4090/5090 | exact | 2026-11-16: CODED STRATA_ADAPT_FORECAST |
| B127 | Prompt-family expert profiles (FineMoE / MoE-Sieve) | FineMoE -47% latency, +39% hit (6-GPU) | exact | expert_seed.hpp:7 |
| B128 | DFlash2 workload-gated drafter only | +4% short, -9% after-long, 7900XT | exact | src/core/verify.cpp:77 |
| B129 | In-graph CPU flag polling with two HW queues | KT CUDA graph up to 1.23x | exact | src/core/mtp.cpp:514 (priority queues) |
| B130 | Per-window dynamic CPU/PCIe split (HybriMoE) | 1.70x vs KT (author); ~0-5% on PCIe 3.0 | exact | - |
| B131 | Deterministic cache-aware routing (Prefetch-Steered Routing) | PSR 2.2-2.3x vs exact LRU, OLMoE RTX 4090 | lossy | 2026-11-15: already STRATA_ROUTE_PRIOR |
| B132 | Lower-bit CPU copies of cold experts (HOBBIT; F4 shadow pack) | ~25-30% fewer CPU bytes; F4: -18.2% bytes, +17.6 GB RAM | lossy | - |
| B133 | Layer-dependent expert skip (HookMoE, ik `-ser`) | HookMoE 1.42x prefill, -2.5% quality | lossy | 2026-11-15: CODED STRATA_ROUTE_TAIL_SKIP_LAYERS |
| B134 | Router-probability-mass EWMA / LFRU (freq-age, hub experts) scoring | +1-3 hit pts (est.) | exact | 2026-11-16: CODED STRATA_ADAPT_PROB_MASS (+ STRATA_ADAPT_CLASSLRU) |
| B135 | Per-layer marginal-gain budget allocation | +0.5-2 pts | exact | 2026-11-15: CODED STRATA_ADAPT_PROB_MASS |
| B136 | Quest page-level / partial top-k / fused select+attn for QSA after long prompt | gap 2.18 ms/token at 12K; est. recover most of 20% | exact (b,c,d) / lossy (a) | - |
| B137 | Streamed KV (>32K) at 4-bit, int8 resident | halves PCIe bytes at 64-131K (est.) | lossy | include/strata/kernels/kv_q4.hpp:9 (`--kv-q4`) |
| B138 | Per-layer persistent super-kernels + "productive spin" MALL/L2 prefetch of next layer dense weights | est. 1-2.4 ms (4-10%) only after CPU wait hidden; Alpindale 905->1000 tok/s 5090 | exact | - |
| B139 | Non-temporal (slc/nt) GPU loads for streaming weights to keep MALL for GDN state/KV | low single-digit; evict-first -8% on 5090 | exact | 2026-11-16: CODED STRATA_EXPDEC_NT (+ EXPERT_V2 / native_mmvq nt loads) |
| B140 | Remove graph-internal memset/memcpy nodes; linear-chain graph to avoid barrier-AND packets | unsourced; ceiling ~2.4 ms/window (1000 nodes x 2.4 us) | exact | - |
| B141 | Batch per-layer contiguous expert copies on SDMA for prefill | low single digits | exact | verify.cpp:1294 (logs HSA_ENABLE_SDMA); 2026-11-17: CODED `STRATA_PF_COPY_BATCH` on the prefill.cpp side only (PF_AHEAD pool copies as 2D copies; MOD_PLAN.md item 48, T-744, spec 149, never built) |
| B142 | Software prefetch in Q2_0 `row_multi` (item 40) | IQ prefetch -1.3% ms/round Zen 3 | exact | - |

### Counts

142 leads (B1-B142) after dedup. Not in plan: 16 strict, plus 12 only partly covered (28 total). Parked or closed entries count as in the plan.
B98 comes from MOD_PLAN section E. It is listed because it targets the same staged path as CD:15.
Strict "no" (16): B56, B59, B63, B67, B72, B77, B78, B81, B83, B115, B123, B131, B134, B139, B140, B141.
Partial or "~" (12): B69, B80, B118, B120, B122, B124, B125, B126, B133, B136, B137, B138.

### Closed questions

- Multi-row "decode once" IQ2_XS CPU kernel: already implemented with software prefetch, so the item is closed (MS:23, MS:85).
- The CPU pool is bandwidth-bound at ~27-33 of 33 GB/s DDR4. More threads are worse (12 workers: 99 vs 112), and nice +5 costs 4 tok/s. Only fewer or hidden bytes help (PP:9, MS:23, MS:86).
- Upload-on-miss caches (llama.cpp #29887, ik #2644) do not fit PCIe 3.0. A copy takes ~107 us vs ~44 us of CPU compute, and DMA competes for DDR4. Copies should be driven only by reuse (MS:13, MSt:22, MS:89).
- Zero-copy host reads, ReBAR and SDMA vs blit for decode cannot beat the 13.4 GB/s link (85% of raw) or 44 us per CPU miss (MSt:34-44, MS:92).
- Lossless compression of IQ2_XS/GGUF experts: essentially incompressible (ZipNN), so the bound is ~8-12% (plan A29) (MSt:52, MS:91).
- LUT kernels (T-MAC/Vec-LUT) need a format change (lossy) and hit the same DRAM wall. They give 0 on Q2_0 unless the kernel proves compute-bound (MS:90, GT:342).
- DFlash2 as default: -9% after long prompts on 7900XT (84.3 vs 92.5), plus an upstream crash on QFN. Closed (MS:27, GT:111).
- Longer windows `--spec 5/6` (111.0 / 109.2) and full or zh60k draft vocab (111.9 / 92.5) are flat or worse. The expert union grows per extra row (MS:45).
- Draft trees, TreeWY, Lookahead/Jacobi and Token Recycling are excluded: GDN state forking plus a wider expert union (GT:109, MS:94).
- GDN megakernel / full megakernel: 0.0 ms on one GPU while CPU-bound. Revisit only after the CPU wait is gone (PD:30, MS:93). HIP graphs and global unroll were tested with no gain (GT:181).
- GPU kernel speedups give ~0 on edit windows while the GPU spins in `combine_cpu`. A faster GPU expert phase shrinks the overlap window (MS:11, GT:183, CD:90).
- Item 22 (FreeToken q* split) is effectively `--pcie-frac`. HybriMoE/Fiddler dynamic split is covered by `--pcie-frac`/PcieAuto (GT:68, plan 1299).
- Sparsity masks (item 13) were rejected offline: Flash-Next FF-640 experts are dense (gate cos 0.844 at 70%) (PD:107).
- Predictor recall item closed: 19-21% coverage, and promotions do not pay on PCIe 3.0 (PD:107, GT:337).
- ContextPilot: 1.6% savings on 40 sessions, and non-prefix reuse breaks GDN state. It stays off (PD:103, GT:129).
- BFLA on Flash-Next: the QSA prompt path already runs 2,051 cells per query, and a BFLA budget of up to 32K cells would be slower. Only a shared-selection-per-tile variant remains (PP:64).
- Item 26 (`oldString` constraint) has almost nothing to fix: 1 failure in 180 edits (PP:44).
- `STRATA_PREFILL_MMQ=OFF` link break: fixed by stubs (95b). An OFF build silently loses the fused int8 prompt experts (PP:62, plan 500).
- #5655 route_prior OOB: the guard is present in our tree (`route_prior.cu:55-61`), so do not re-port it (plan 3016).
- Decode slowdown after 12K is not raw KV reads. At 600 GB/s the 2.18 ms/token would need ~1.3 GB of reads. GDN state is constant-size. Suspects are QSA selection cost or tier pollution from prefill (MSt:127-135).
- KV compression to free VRAM for experts: ~0.2 GB is <1% of experts, ~1-2% decode at most (MSt:97).
- Skipping unrouted experts in prefill gives no gain at 4K chunks with 512/top-10: the unused fraction is ~0 (MSt:79).
- `s_prefetch_data` is gfx12-only and not available on gfx1100. Prefetch on RDNA3 means early loads or dummy loads (MSr:75; unsourced).
- WMMA 16x16 wastes at least half its tile at M<=8 verify rows. Multi-row MMVQ with dot4 is the right shape for decode (done by the rdna3 port). WMMA pays only in prefill and GDN state math (MSr:59, MSr:76).
- Chunked WY/UT GDN forms are not bit-exact vs the sequential recurrence. Replay/commit and per-token checkpoints are exact (MSr:98). The chunked WY GDN scan measured 0.73x on the 3060 (plan 2483).
- int8-IMMA/WMMA attention as a main-engine default was rejected: attention is 6.7% of the window (~3% of decode) (PD:52, PD:83).
- Strata gfx1100 switches (int8 WMMA attention, TSUM, expert layouts) are neutral or negative on the 7900. Turning on all of group A together regresses (PD:83, MS:29).
- PLE-in-RAM: 57 GB of RAM cannot hold the model, the 28.8 GB table and the reserve (PD:83).
- Upstream router residency bias as a default: +37% decode but agent score 5.6 -> 4.5 and loops. Any routing bias stays opt-in behind task gates (GT:26, PP:15).
- 4,096-token staged threshold: the PREFILL_LEVERS 6.4 prediction was refuted by spec 46 (plan 3932). The env-only CPU_SHARE A/B is the measurement that settles it (CD:15).
- No paper evaluates effort levels on agentic coding turns. A "medium" default stays a hypothesis, and only a tail-above-p99 cap is low-risk (GT:151, PP:40).
- `answer_room` already puts an implicit 24,000-token thinking cap on opencode turns (max_tokens 32000) (CD:31).
- Restore off-by-one (UNSW) is already guarded: the prefix is used only at <= n-1 tokens, and the last output never enters the saved list (CD:27).
- `strata-poly` is CUDA-only on HIP even though CMake declares it. `STRATA_SELECT_LONG` and `STRATA_PF_GROUP_SCRATCH` do nothing on the HIP preset (PP:62).
- Physical limits with no software lever: DDR4 bandwidth (owner BIOS: 3600/FCLK 1800), PCIe 3.0 at 13.4 GB/s, and 20 GB VRAM = 44% of expert pairs resident (only a 2nd GPU changes that, parked). No paper covers PCIe 3.0 + AMD + 512 experts + GDN hybrid (PP:70).
- Exact ceilings: ~140-146 tok/s from cache and bandwidth alone, ~155-165 with the GPU path and window policy. The theoretical edit ceiling for F1-F3 combined is ~1.9x (floor ~20 ms) (MS:121, GT:3).

---

## Part C: unported-academic leads (pancake-docs/research/engine-speed/unported-academic/**)


### 1. Expert offload, cache, routing

| lead | number + hw | exact/lossy | in our src |
|---|---|---|---|
| A1 cache-conditional routing (bonus on resident-expert logits) | miss 35->16% Qwen1.5, 2x vs LRU (Snapdragon, CPU llama.cpp); ours +30-50% before deferral | lossy (ppl +0.1-3%; "2.9%" is Mixtral ppl, not token diff) | src/core/verify.cpp:296 |
| Cache-Prior top-J guarantee (top experts load regardless of cache) as A1 guard | paper Sec 3.3; lambda 0.8 over-uses cache | lossy guard | 2026-11-15: already STRATA_ROUTE_PRIOR_PROTECT |
| AcceptMoE residency-aware utility set (root anchor, effective-rank size, reroute non-resident) | 2.06x vs standard SD, H2D -73.6-77.1%, RTX 5090 48 slots; acc -0.27 pt mean | lossy | 2026-11-16: CODED STRATA_ROUTE_PRIOR_ERANK |
| A2 depth-graded prior (our geometric bonus) | "+5-15%" unsourced | lossy | include/strata/kernels/route_prior.hpp:12 |
| HOBBIT cumulative gate-score rule (T1 0.6 / T2 0.9) as skip/defer rule, no low-bit copy | Mixtral 67/30/3% split; dyn loading 1.19-1.57x (4090, Orin) | lossy | 2026-11-15: CODED STRATA_MISS_SKIP_CUM |
| A4 rank guard `_TOP/_BELOW/_FOLD` | vs deferral: edit -2.6 to -4.2%, short ~0 (n=2) -> HOLD | lossy-small | src/core/verify.cpp:985; include/strata/core/defer_rank.hpp:48 |
| A27 sub-2-bit copy on CPU miss | ~25% bytes/miss (IQ2_XS->IQ1_M, from memory); IQ1_M expert error 0.55-0.91 | lossy | 2026-11-16: CODED STRATA_SHADOW_MISS |
| SliceMoE Matryoshka slices (low-bit = shifted high-bit, no duplicate bytes) | 2-bit ppl 10.85 vs 4-bit 8.09 (Qwen1.5); up to 1.81x on simulated mobile SoC | lossy | 2026-11-16: CODED STRATA_SLICE_MISS |
| MoBiLE big/little (half experts, confidence gate 0.7) | 1.72x/1.60x RTX 4080; OLMoE GSM8K -6.7 pt | lossy | 2026-11-16: CODED STRATA_MOBILE |
| Routing dump with gate weights (unblocks A4/A27/A32/A46 replays) | existing 219-window trace has 0 weights | n/a | src/program/generate.cpp:1150 (`--dump-routing` writes ids+weights) |
| A7 chunked promotions + drop queued speculative + reordered exec | ProMoE 1.09x (max 1.17x) llama.cpp decode vs LRU, 4090 PCIe 4.0; our sim +4.6% | exact | include/strata/program/promo_chunk.hpp:2 |
| ProMoE per-layer MLP predictor + stride prefetch | 84.7% acc; stride +1 layer at ~5% acc cost | exact | router lookahead exists: include/strata/core/expert_source.hpp:544 |
| HOBBIT next-1-3-layer prefetch from gating input | 96% top-1 next layer, ~90% at 2-3 | exact | yes: include/strata/core/expert_source.hpp:341,544 |
| HOBBIT multi-precision cache score (LRU+LFU+LHU+layer distance) | no isolated number | exact | 2026-11-15: CODED STRATA_TIER_LRU |
| A8 online swap budget, blocking tier | async -9.6%; est 0-5% over best fixed | exact | include/strata/program/swap_budget.hpp:2 |
| A10 working-set prefetch (MoE-Infinity) | -19 to -32% on our 1 replay trace | rounding | include/strata/core/expert_source.hpp:813 (working set) |
| A44 epoch bandit | Athena +5.0% vs MAB (ChampSim CPU, 100 traces); stateless Athena < MAB | policy (via lossy arms) | include/strata/core/epoch_bandit.hpp:103 |
| Athena reward correction (subtract workload-driven term) + state features for bandit | state +1.4/+1.7/+0.8/+0.1, reward +1.0 pt (CPU sim) | policy | 2026-11-15: CODED STRATA_BANDIT_CORRECT |
| A45 start-up calibration probe | expected share 0 on this box (raw ~0.28 < 0.35) | exact | src/program/generate.cpp:8366 |
| DALI per-layer per-step greedy CPU/GPU assignment (vs static share) | assignment alone +23% vs HybriMoE; decode 1.32-3.97x (EPYC 7532 + 3090, PCIe 4.0) | exact | 2026-11-15: CODED STRATA_DALI |
| A46 learned cache policy | Belady 94.7 vs LRU 89.5 vs engine 79.6%; best online +3.2 pt but 2-3x promotions | exact | 2026-11-16: CODED STRATA_ADAPT_LEARNED |
| A29 expert-byte entropy coding | bound 11.7-12.0% Q2_0, zstd 8.0%; IQ ~0% | exact | 2026-11-16: CODED STRATA_EXPERT_ENTROPY_TIER (lossless; coder + tier) |
| A28 ISTA-DASLab GSQ-RCO Coder | SWE-bench 95.7% retained (card); 58.4 GB Qwen / 111 GB DeepSeek | lossy (50% expert pruning + quant) | 2026-11-15: left (owner 2026-11-13: do not test GSQ) |
| ExpertCache task-specific expert preload (RL) | Qwen3-235B, BigCodeBench (snippet) | exact (placement) | 2026-11-15: already STRATA_DOMAIN_PROFILES |

### 2. Sessions, prefix and KV cache

| lead | number + hw | exact/lossy | in our src |
|---|---|---|---|
| A9 pre-stage experts during tool gaps | bound 2-4% of wall | exact | src/program/generate.cpp:14928 (`STRATA_IDLE_ADAPT`) |
| TokenCake: offload idle KV during function calls, predictive upload | up to 47.06% e2e latency, memory-constrained (abstract) | exact | 2026-11-16: CODED STRATA_TOOL_IDLE_PREWARM (serve) |
| A11 tool-boundary checkpoints + FLOP-aware eviction (Marconi) | eviction-only +19.0-219.7% hit vs LRU (7B hybrid, 4xA100); ours 0.5-0.8% | exact | include/strata/program/conv_cache.hpp:99 (`--prompt-cache-spread`) |
| A12 pinned system+tools prefix | -1.3% modelled on Qwen | exact | include/strata/core/conversation_cache.hpp:533 |
| Pinned-entry eviction fix (prereq for A12) | pinned parked entries never evicted | exact | 2026-11-15: already STRATA_AUTO_PIN |
| A15 park TTL for sub-agents | Continuum gain is queueing term, ~0 single user | exact | include/strata/core/conversation_cache.hpp:494 |
| Rank-3 eviction bug (evicts whole held set on byte shortfall; take smallest covering entry) | 6 of 7 cold re-reads were over-budget skips | exact | 2026-11-15: already fixed conversation_cache.hpp:783-789 |
| MORI: idleness from tool durations, cross-tier KV migration | 20-71% thpt, 18-43% TTFT (search summary, unverified) | exact | 2026-11-16: CODED STRATA_TOOL_IDLE (serve) |
| A14 restore check: teacher-forced / second recompute control arm | paper 34/36 -> 36/36; CPU reload TTFT -46-64% | exact | tools (restore_check.py --control) |

### 3. Speculative decoding

| lead | number + hw | exact/lossy | in our src |
|---|---|---|---|
| A21 exact sampled speculation + `STRATA_SPEC_MIN_TEMP=0.7` | +5.5% B70 at T1.0; +13.4% 5090 DT 0.85; loses below T~0.7 | exact (distribution) | include/strata/core/spec_prob.hpp:64 |
| A22 entropy chain stop (AdaEDL) | vs top-1 stop +0.2-9.3% (mean 3.2%); vs best static +4.4-10.5% (Llama2-7B, A100, T0.7) | exact | 2026-11-16: CODED STRATA_SPEC_ENTROPY=adaedl |
| Top-1 confidence stop (Max-Confidence baseline) | AdaEDL's main baseline; MTP returns top-1 already | exact | include/strata/spec/window_rule.hpp:25 (`--spec-min-p`); src/spec/disco.cpp:106 (`SPEC_CUM`) |
| DISCO learned stop | 10.3% vs best static, 31.4% vs heuristic (4-dataset avg, A100, greedy) | exact | src/spec/disco.cpp:107 |
| SVIP entropy-bound draft length | +17% MT-Bench 8K, +22% QwQ (snippet) | exact | 2026-11-16: CODED STRATA_SPEC_ENTROPY=svip |
| Drafter top-k + entropy on device (extend record 4) | build only if entropy beats top-1 on replay | exact | 2026-11-16: CODED STRATA_DRAFT_ENTROPY (+ STRATA_DISCO_ENTROPY) |
| Record 6 target p(d) / Leviathan EAR min(1,p/q) (A56) | metric | exact | include/strata/core/window_trace.hpp:29,125 |
| A24 cross-session suffix index | replay +2.0% accepted, -2.4% rounds (sub-agents +6.0% / -7.5%); paper 5.3x SQL, 2.5x SWE | exact | src/spec/suffix_index.cpp:279 |
| A25 position-decay check (draft_report.py) | 0-5% est. | exact | tools only |
| A26 draft-source selector | none beats always-lookup; per-window longest match +2.0% | exact | DraftPolicy (draft_policy.hpp) |

### 4. Prefill, kernels, attention

| lead | number + hw | exact/lossy | in our src |
|---|---|---|---|
| A30 CPU share in prefill: Fiddler latency gate, env A/B on gfx11 | Fiddler 1.26x/1.30x (Mixtral FP16, RTX 6000); share off on gfx11; spec 46 -5 to -30% | exact (KL ~0.004) | src/prefill/prefill.cpp:288; include/strata/prefill/share_rules.hpp:22 |
| A31 GDN chunk 64 / blockwise inverse | no paper speedup; GDN 11.9% of prefill (RTX 3060) | lossy (FP16-class) | src/prefill/gdn_chunk.cu:29; 2026-11-17: see B42 (MOD_PLAN.md item 49) |
| A32 stream experts for large prefill ubatch | OSDI'26 1,200 / 1,800 tok/s (2x 5090, PCIe 5.0); 64-tok ubatch touches ~87% experts (uniform bound) | scheduling; "exact" unsupported | 2026-11-15: already prefill layer-major walk |
| SeqMoE: chunked prefill re-transfers experts per chunk (size chunks to avoid it) | snippet, no number | exact | 2026-11-15: already prefill layer-major walk |
| A33 small-M fusion | +0.6% [-1.6, +2.8] HOLD | exact if q8_1 order matches | include/strata/core/switch_report.hpp:35 |
| A34 coordinate-descent launch/tile sweep | 5.5-8.5% (A100 thread blocks); Kernel Tuner: tuning ~10x on AMD vs 2x NV (snippet) | lossy unless bit-exact | 2026-11-15: method |
| A35 Infinity Cache warm during combine wait | PRESERVE 1.61x on Ascend 910B L2 (192 MB) | exact | include/strata/core/switch_report.hpp:34 (`STRATA_IC_WARM_MB`) |
| A36 hit-path ceiling | +1.99% if all VRAM-hit work vanished | n/a (paper FP8 lossy) | 2026-11-15: method |
| A48 Q2_0 vs 2-bit GEMV bench | 1.23x 1 thread, 1.03x 8 threads (DDR4-bound) | exact | tools (q2_rows_bench) |
| A49 attention share vs context | decode flat with context within noise | measurement | n/a |
| A50 verify-window attention, prefix K/V read once | F11 200K 48.6 -> 88.9 tok/s (5080); ~0 short | exact | 2026-11-16: CODED STRATA_ATTN_VWIN (Flash-Next), q36 STRATA_ATTN_WINDOW |
| A51 Quest/Twilight sparse decode attention | Quest 7.03x attn, 2.23x e2e (4-bit); Twilight 3.9x vs FlashInfer | lossy | 2026-11-15: CODED STRATA_QSA_PAGE_SEL |

### 5. Agent loop and thinking (client/server side)

| lead | number + hw | exact/lossy | in our src |
|---|---|---|---|
| A19 ThinkBrake margin stop: log(p_top/p_`</think>`) <= 0.25 at sentence end | -25.1% thinking tokens BFCL (Qwen3-4B-Thinking, v5); <= ~5.5% turn time | lossy | 2026-11-15: CODED STRATA_THINK_BRAKE |
| Per-token top-2 margin in turn log (unblocks A19 replay) | turn log has no logprobs | exact (log) | 2026-11-15: CODED STRATA_TURN_MARGINS (serve) |
| A19 Thought Calibration probe | 60% ID / 20% OOD (math/science) | lossy | src/program/generate.cpp:7715 |
| A17 per-effort caps / graceful close | up to 60% drop on late updates; s1 cap is a quality tool | lossy | serve (presets) |
| A18 effort per task (time-to-success) | "-20-40%" unsourced | lossy | serve (turn log effort) |
| A20 rumination + entropy close | r=0.91 one model; host logprobs 1.84 ms/token | lossy | serve/server.py:7940 |
| A40 decode during tools (AsyncFC futures) | 1.40x SWE-bench Lite vs sequential (hosted); our bound 5.4% of span | lossy | 2026-11-16: CODED STRATA_MCP_ASYNC (serve) |
| Speculative read-only tool execution (Atomix, PASTE, speculative actions) | PASTE -48.5% task time (unverified) | lossy/side effects | 2026-11-16: CODED STRATA_MCP_SPEC (serve) |
| Agent-run early termination (EarlyEval, EET) | EarlyEval stops 35% runs, -26% steps, -30% tokens; EET -19 to -55% cost, <=0.2 pt (unverified) | lossy | 2026-11-16: CODED STRATA_MCP_EARLY_STOP (serve) |
| A41 ECLoop hard hold (read-before-edit, test-before-done) | +4.8 to +11.8 pp SWE-bench Verified (hosted); 33/339 edits had no prior read | lossy | hook registered warn-only on the box 2026-11-15 (MOD_PLAN.md D10, T-562) |
| A38 behaviour counters | pilot 11/21 edit sessions no test after last edit | exact (log) | serve/agent_loop.py:11 |
| A39 async compaction (Slipstream) | -11.3 to -39.7% vs sync (H100, 9B-36B) | lossy | 2026-11-16: CODED STRATA_ASYNC_COMPACT (serve) |
| A42 externalize big tool output | RLM vs CC+offload 0/+8.7/+16.7% (cross-model); read+bash = 37% of output chars | lossy | serve/agent_loop.py:20 |
| A43 oldString check | 0 of 339 edits failed | lossy (constraint) / exact (log) | serve/agent_loop.py:24 |

### 6. Measurement and gates

| lead | number + hw | exact/lossy | in our src |
|---|---|---|---|
| A6 overlap time model | 3.4 pt on fitted arms, 8.1 pt held-out (rank guard) | n/a | tools (time_ceiling.py) |
| A54 KLD/top-1 rule narrowed | silent-zone rho 0.00 Qwen / -0.24 Devstral, CI ~+-0.5, zone chosen post hoc | n/a | tools |
| A55 task-gate power | 5-pt margin needs ~492 runs/arm, 10-pt ~123; current 18 tasks x 5 reps | n/a | tools (task_gate.py) |
| BFCL multi-turn 30-entry subset adapter | not built | n/a | 2026-11-15: method |
| A57 time-to-success AUC, pinned horizon | horizon 1800 s pinned | n/a | tools |

Counts: 74 leads; not in plan (no row or only a note/attribution): 15 = top-J guard, AcceptMoE utility set, HOBBIT cumulative rule (wording only), SliceMoE, HOBBIT LHU cache, Athena reward correction, DALI per-step assignment, ExpertCache, TokenCake, SVIP, drafter on-device top-k/entropy (partial), SeqMoE re-transfer, speculative read-only tools, agent-run early termination, A55 power figure (partial).

### Closed questions

- A37 LDS LUT duplication: closed NO-GO. Kernel is 1.7% of GPU time, ceiling 0.15-0.55%, x16 halves occupancy (UA/round3/notes/batch6.md:22,31; UA/STATUS.md:41).
- A43 constrained decoding for oldString: closed. 0 of 339 edits failed; the paper (type-constrained whole programs) gives no edit evidence (UA/round3/notes/batch6.md:156,163).
- A48 2-bit CPU kernel: closed. 1.03x at 8 threads, about 3% ceiling. x86 LUT GEMV needs AVX-VNNI, which Zen 3 lacks (UA/STATUS.md:51; UA/round3/notes/batch7.md:69).
- A29 expert-byte entropy coding: bound 8-12%, below the 20% bar. IQ types are ~random (UA/round3/notes/batch4.md:156). Plan reopened it only as an owner decision (2887).
- A49 / A51: decode is flat with context within noise, so the A51 gate (attention >= 15%) is not met. Drop A51 unless a profile says otherwise (UA/STATUS.md:71-72).
- A36 MonoMoE: not applicable. "Weight-major" is CTA tiling, FP8 on H200 and lossy (cosine >= 0.998). The ceiling is +1.99%, under the 2% bar (UA/round3/notes/batch5.md:123,136; UA/round4/notes/V4.md:132).
- A53 IOMMU: F&S speeds up translated mode only. This box's GPU group is `identity` (passthrough), so nothing applies. Start-up note done (UA/round3/notes/batch8.md:8,20; include/strata/platform/iommu_note.hpp:2).
- A4 rank guard: no gain over deferral (edit -2.6 to -4.2%). `_TOP=1` is a no-op at `_BELOW 0.1` (UA/round3/notes/batch1.md:85).
- A26 draft-source selector: no per-request selector beats always-lookup (UA/STATUS.md:66).
- A15 Continuum TTL: its gain comes from a queueing term that is about 0 with one user. Sub-agents are untested. The "MORI" attribution is unsupported (UA/round3/notes/batch3.md:46,67).
- A27 unconditional sub-2-bit copy: drop. 2-bit AMAT is 34-44% worse in ppl than 4-bit; IQ1_M saves only ~25% of IQ2_XS bytes; shadow error is 0.55-0.91. Only a gate-score-gated form stays open (UA/round2/notes/offload_round2.md:96-98).
- A30: the "4096 token cut" is not in Fiddler. Fiddler uses a latency gate, which `auto` already implements. The share is off on gfx11 by arch rule (include/strata/core/arch_defaults.hpp:30). Spec 46 measured -5 to -30% and was dropped (MOD_PLAN 3840; UA/round3/notes/batch5.md:8,15).
- A31: chunk 64 is already in code (src/prefill/gdn_chunk.cu:29). The "5-11%" is unsourced, and the paper has no prefill speedup (UA/round4/notes/V3.md:15).
- A35 PRESERVE: the 1.6x is Ascend 910B L2 in a TP setting. No AMD, no Infinity Cache, no 80 MiB; the mapping is ours (UA/round4/notes/V4.md:112).
- A32 "PCIe 3.0 caps it" and "exact": not from the OSDI paper (PCIe 5.0 x16, no bit-equivalence claim) (UA/round4/notes/V4.md:50,55).
- A44 framing: Athena is SARSA on CPU prefetch/OCP, not a bandit over window types. A stateless Athena is slightly worse than MAB, so "+5%" is not a prediction for our stateless MAB (UA/round4/notes/V1.md:34). A time-only reward cannot learn the quality cost of lossy arms (UA/round2/notes/offload_round2.md:47).
- A2: the AcceptMoE mechanism is a utility set with rerouting, not a depth prior. "+5-15%" has no source (UA/round3/notes/batch1.md:49).
- A7: ProMoE does not cancel in-flight copies ("cannot be preempted midway"). Only queued speculative work is dropped (UA/round4/notes/V1.md:30).
- A1: "2.9% tokens differ" is a Mixtral perplexity increase, not a token-diff rate (UA/round4/notes/V1.md:21).
- A56: under greedy verify, EAR equals top-token agreement. Leviathan alpha needs p(d) and q(d) and differs only under sampled verify (UA/ROUND2_REPORT.md:42).
- A22: SpecFit is a tree method (trees already rejected); only its cost term transfers, and that is A23 (done). MiMo has no stop rule. DISCO "7.9% ceiling" is a plan figure, not a paper figure (UA/notes/agent_thinking_papers.md:60; UA/round4/notes/V1.md:18).
- A24: 5.3x is AgenticSQL only (SWE-Bench 2.5x). "~10% on edits" is not in the paper (UA/round4/notes/V4.md:73,76).
- A19: OS-Pruner is math-only and fine-tuned, with AIME -15 pt at lambda 1e-4. The ICLR 2410.04707 paper is per-query compute routing, not a thinking stop (UA/ROUND2_REPORT.md:29,31).
- A41: a natural-language nudge does worse than no guidance (58 vs 63 solved). Only the hard hold is the tested mechanism (UA/notes/agent_thinking_papers.md:99,106).
- A42: the RLM gain over "Claude Code + offloading" is cross-model (GPT-5 vs Opus 4.1). The offload baseline captures most of the gain on 3 of 4 tasks (UA/round4/notes/V3.md:39; UA/round3/notes/batch6.md:130).
- A39: delta compaction is ACE's (a playbook), not Slipstream's. Most of Slipstream's latency gain is async-only; its accuracy gain comes from the judge (UA/round3/notes/batch6.md:91,101).
- A28: GSQ/RCO papers contain neither the 95.7% nor "LUT types". 256/512 is 50% expert pruning. The DeepSeek variant is 111 GB. Owner decision 2026-11-13: the 58 GB Coder GGUF is on D (`models/ista-gsq-rco-coder-gguf`), kept, NOT tested (Avalon-only, ~29.6 GB resident) (UA/triage/check_T2.md:11-12,17; UA/ROUND2_REPORT.md:55).
- A45: DALI has no "6-13%". A static share on this box is expected to be 0 by construction (UA/triage/check_T2.md:10; UA/round3/notes/batch7.md:16).
- A46: Harmonia splits placement vs migration, not admission (HPC storage, not LLM) (UA/triage/check_T1.md:12).
- A33: the Qwen3 report has no "gated residual" and no "greedy discouraged" claim (UA/triage/check_T1.md:13).
- A6: "validate to 5-9%" is Vidur's headline for another system. Rejections of A7/A8/A46 rest on a model with 8.1-pt held-out error (UA/round3/notes/batch1.md:113-114).
- A55: 18 tasks x 5 reps cannot support a 5-pt non-inferiority verdict. Read the first live run as a cost and failure-rate measurement (UA/round3/notes/batch8.md:74,122).
- A18: the 43% compute cut is candidate selection on SWE-bench, not low-vs-high effort (UA/round3/notes/batch3.md:111).
- HOBBIT next-layer prefetch: already covered by the router lookahead (include/strata/core/expert_source.hpp:544).
- A12: do not enable `STRATA_AUTO_PIN` for Qwen until pinned-entry eviction is fixed (-1.3% modelled) (UA/round3/notes/batch2.md:119).

---

## Part D: hardware-platform research leads (llm-runtimes-2026, data-path, consumer-platforms)

Short names: `LR/` = `llm-runtimes-2026/`, `DP/` = `data-path/`, `CP/` = `consumer-platforms/`, `rep` = `report.md`, `n/` = `notes/`.
"In plan" means a line in `MOD_PLAN.md`. "In src" means a hit from `git grep` over `src include`. Hits elsewhere (serve, tools) are marked.
Box: RX 7900 XT (gfx1100, 20 GB) + Ryzen 7 5700G (Zen 3, AVX2, no VNNI) + DDR4 + PCIe 3.0.

### A. Runtime techniques (llm-runtimes-2026)

| lead | number + hw | exact/lossy | in our src |
|---|---|---|---|
| Expert Deferral, tuned: defer only cache-missed, lowest-weight experts to layer k+2 | KTransformers up to 1.45x decode, avg acc drop <=0.5% (DS-V3 class, 8 routed experts; A100/RTX 4080 + Xeon). Local light variant +8% short / -7% edit (CUDA); HIP gives non-finite logits | lossy | yes: `STRATA_VERIFY_DEFER_CPU` include/strata/core/defer_auto.hpp:2; rank guard defer_rank.hpp:2 |
| One graph per token with CPU work inside (host callbacks) | KT up to 1.23x | exact | banked: verify window HIP graph + `STRATA_WAIT_FUSE` include/strata/kernels/verify_kernels.hpp:102 |
| MTP draft captured with verify; no recapture when the window shape changes | llama.cpp CUDA-graph re-enable +5% bf16 MoE bs1 (RTX 5090); draft ~9% of window | exact | draft already graph-chained src/core/mtp.cpp chain (mtp.hpp:92); `STRATA_MTP_HIP_PRIO` mtp.cpp:513 (default on gfx12 only) |
| Anticipatory checkpoint at each message boundary (DeltaNet state) | first edit 17.2 -> 11.5 s (-33%) with 32K chunks (Strata) | exact | `STRATA_CACHE_MESSAGE_BOUNDARY` src/core/conversation_disk_env.def:321 |
| Cross-session shared prefix (system prompt + tool schema) | none | exact | pinned shared prefix include/strata/core/conversation_cache.hpp:194 |
| Thinking budget, `</think>` + transition phrase at cap | Qwen3 tolerates truncation (thinking-mode fusion); no tok/s gain, less time per reply | lossy (text) | 2026-11-15: already serve reasoning_budget_tokens |
| Dynasor-CoT certainty probe (early exit) | up to 29% fewer tokens; Certaindex up to 50% | lossy | include/strata/program/think_probe.hpp:4 |
| Stream experts to GPU for large prefill ubatches | 1,200 tok/s DS-V3 (arXiv 2606.10493); PCIe 3.0 ~12 GB/s caps it | exact | prefill streamed experts src/prefill/prefill.cpp:2445 |
| KV prefetch during prefill (next layer's KV on own stream) | -5.3% prefill at 32K, -4.1% at 64K, 0 at 8K (2x RTX 3080, PCIe 3.0) | exact | `STRATA_KV_PREFETCH` src/prefill/prefill.cpp:3359 |
| Server never blocks engine: async token streaming over the pipe | est. a few %, not measured | exact | 2026-11-15: CODED STRATA_SERVE_ASYNC_STREAM (serve) |
| Single-node KV/state tier RAM + disk (LMCache-style) | 2.2-32.8x lower hit TTFT vs recompute (vLLM, Llama 3.2 3B, NVMe) | exact | item 39 include/strata/core/conversation_disk_tier.hpp:1 |
| Expert-cache policy headroom | replay at 44%: engine tier 79.6%, every-window x160 85.0%, per-layer LRU 89.5%, Belady 94.7% | exact | adaptive tier (n/a) |
| ClassLRU decay (llama.cpp CPU share) | within 0.5 pt hit rate in replay | exact | 2026-11-15: CODED STRATA_ADAPT_CLASSLRU |
| HybriMoE dynamic intra-layer CPU/GPU split | 1.70x decode / 1.33x prefill over prior hybrid | exact | PcieAuto include/strata/core/pcie_share_rule.hpp:2 |
| SpecReason step-level speculation | 1.4-3.0x (paper) | lossy | 2026-11-16: CODED STRATA_SPEC_REASON |
| Reasoning KV compression (R-KV, ThinKV, TIP) | eviction can lengthen traces; TIP needs a trained model | lossy | 2026-11-16: CODED STRATA_RKV (TIP part park: trained model) |

### B. Data path (CPU/RAM, PCIe, NVMe, GPU sync)

| lead | number + hw | exact/lossy | in our src |
|---|---|---|---|
| 38 Direct O_DIRECT arena loader on Linux (DirectFile, 16 threads) | 28-65 s at 0.49-1.18 GiB/s -> est. 10-14 s; measured floor 16.9 s for 35.45 GB (DRAM-less NV2) | exact | `load_experts_direct` src/core/pinned.cu:1028; `experts_unbuffered` include/strata/core/pinned.hpp:201 |
| `POSIX_FADV_DONTNEED` after buffered chunks (interim) | memory pressure, not speed | exact | src/core/expert_source.cpp:928 |
| `POSIX_FADV_SEQUENTIAL` on buffered fallback | none | exact | yes, other path: src/core/expert_source.cpp:5954 |
| `MADV_POPULATE_WRITE` arena prefault | load-time only, unmeasured | exact | 2026-11-15: CODED STRATA_ARENA_POPULATE |
| Huge pages (hugetlb / THP) | log: "MAP_HUGETLB unavailable (needed 16920 2 MiB pages)"; hugetlb +5-7% does not fit box | exact | `MAP_HUGETLB` include/strata/core/pinned.hpp:8 |
| 39 Auto disk tier + Bidaw (tool-call stays in RAM, final answer to disk first) | restore 1.13 GB est. 0.5-1 s; CachedAttention -87% TTFT; HCache 5.7x | exact | `--conversation-disk-dir` include/strata/core/conversation_disk_tier.hpp:1 |
| 40 `_mm_prefetch` 1-2 KB ahead in Q2_0 `row_multi` | 0-5% of 17.4 ms CPU | exact | `STRATA_Q2_PREFETCH` src/kernels/cpu/q2_avx2.cpp:82 |
| 41 Smallest saturating thread count (FUSE_PHASES) + `performance` governor | 0-3% each; 6-12 threads equal = saturation | exact | 2026-11-15: method |
| 42 Max-plus per-layer simulator; measure alpha (DDR contention pool vs DMA) | validate to 5-9% error (Vidur, LLMServingSim) | n/a | 2026-11-15: method |
| 43 `DraftPolicy::seed_costs` from traces (replaces RTX 5070 `kShape`) | 0-5% | exact | include/strata/spec/draft_policy.hpp:76 |
| 44 Critical-path pass (slack per layer, virtual speedup) | diagnostic | n/a | 2026-11-15: method |
| 45 `STRATA_IO_THREADS` sweep, filefrag, PLE frequency order | diagnostic | exact | src/platform/direct_file.cpp:62 |
| 46 Single poller in `combine_cpu`, VRAM flag republish | 0-3%; ~50 pollers per window today | exact | `STRATA_COMBINE_POLL1` src/kernels/cuda/native_moe.cu:147 |
| 47 Warm Infinity Cache with layer l+1 pre-doorbell weights during spin | 0-2 ms/window if warm/cold < 0.85 | exact | `STRATA_IC_WARM_MB` include/strata/core/verify.hpp:686 |
| 48 Fuse pre-doorbell chains (HC norm/down/up, router+plan+bell) | 1-4% per chain, both window types | exact | `STRATA_ROUTER_FUSE`, `STRATA_EXPDEC_SWIGLU` src/core/conversation_disk_env.def:23,35 |
| 49 Timer stamps inside `combine_cpu` | instrument | n/a | src/core/verify.cpp:2894 |
| 50 Spin vs row pull: combine with flag pre-set | decides 51/52 (pull < 3 us/layer -> ~0) | n/a | 2026-11-15: method |
| 51 CPU pre-sums missed rows per token | 2-4x fewer pull bytes, 0.5-1.5% edit | not bit-identical | `STRATA_CPU_PRESUM` include/strata/core/expert_source.hpp:428 |
| 52 CPU pushes rows + flag into VRAM via ReBAR (WC stores) | 1-2% est.; GPU bound <=0.27 ms; datapath <=2.1% | exact (with 51: no) | `STRATA_PRESUM_VRAM` include/strata/core/verify.hpp:782 |
| 53 Online swap budget; 256 KB chunked promotions; stale-drop in `fill_loop` | 0-5% over best fixed `--adapt-swaps` | exact | 2026-11-15: CODED STRATA_FS_STALE_DROP |
| 54 IOMMU passthrough guard at start | protects status quo; strict 4 KiB = IOTLB misses (~344 pages per 1.4 MB) | exact | include/strata/platform/iommu_note.hpp:17 |
| GPU-busy decomposition: total - calls x (mean - min) for spin kernels | combine_cpu min 5.6 us vs mean 254 us: >=97.8% of its 38.3% is waiting | n/a | 2026-11-15: method |
| HIP micro-bench set: zero-copy read BW, one PCIe read RTT, CPU WC-write BW into BAR, pool + H2D contention | gates 52 and alpha | n/a | 2026-11-15: method |
| `GPU_MAX_HW_QUEUES` 4 vs 6; CU-masked stream for spin/warm kernels | 4 vs 6 flat today; ROCm recommends <=4 | exact | 2026-11-15: CODED STRATA_WARM_CU_MASK |
| `STRATA_COPY_HIP_PRIO` + coalesce adjacent promotion copies | 0-1%, likely flat at pcie-frac 0 | exact | 2026-11-15: CODED STRATA_DMA_COALESCE |
| io_uring in DirectFile | ~1.06x drop-in; SQPOLL burns a core | exact | 2026-11-15: CODED STRATA_IO_URING |
| Q2_0 code entropy check | Q2_0 down <=12% compressible, IQ <=3.4% | n/a | tools/pancake/expert_entropy.py |

### C. Platform, BIOS and hard facts for this box

| lead | number + hw | exact/lossy | in our src |
|---|---|---|---|
| 55 Startup calibration on window time (2-3 `--pcie-frac` values, hysteresis, per-machine cache) | PcieAuto picked 0.20-0.25, `--pcie-frac 0` won 111.7 -> 118.4 tok/s; R9700 0.55 -> 0.15 gave +9.9% / +25.6% (8K/32K) | exact | include/strata/core/calibrate_ratio.hpp:4 |
| 61 Defaults by VRAM size and link probe; emulate small cards by capping slots | 20 GB -> 10,909 of 24,576 pairs (44%), ~4.7 GB non-expert | exact | 2026-11-15: CODED STRATA_LINK_DEFAULTS (+ EXPERT_SLOTS_MAX) |
| 57 cheap: platform log line + `STRATA_FORCE_ISA=avx512\|avx-vnni` | XS | exact | 2026-11-15: park: no AVX-512 on the 5700G |
| 59 code read: do Q2_0/IQ AVX-512 rows use the multi-token window | DS path 1.05-1.3x at 1 tok, 2.4-3.5x at 4, 2.8-4.8x at 16 (Zen 4/5) | exact | `row_dot<TY,NT>` in iq_avx512.cpp |
| BIOS: iGPU carve-out 4 -> 2 GB; DDR4-3600 at FCLK 1800 | unmeasured; no paper on desktop FCLK vs decode | exact | 2026-11-15: method |
| BIOS: force GPU slot to Gen2 to emulate an x8 card on Gen3 | ~6.5 GB/s, ~215 us per promotion | n/a | 2026-11-15: method |
| Check board chipset (`dmidecode -t baseboard`) and card `LnkCap` | B550/X570 -> Gen4 with non-APU Zen 3; B450/X470 stays Gen3 | n/a | 2026-11-15: method |
| Fact: link = root port Gen3 x16 (card Gen4 x16), ReBAR 32 GB BAR0, IOMMU identity, probe 13.4 GB/s, promotion 115 us vs 107 us theoretical (~93%) | measured | n/a | n/a |
| Fact: CPU 42 us/expert vs promote 115 us (1:2.7); Q2_0 ~80% of 33 GB/s DDR4 (Intel's best x86 2-bit GEMV: 80-90% band) | measured | n/a | n/a |
| Fact: Zen 3 has no AVX-VNNI; `maddubs`+`madd` fallback costs 2 extra uops per 32 B | from cpuinfo | n/a | VNNI probe src/kernels/cpu/expert.hpp:192 |
| Fact: edit window 37.4 ms = GPU-reach 15.8 + CPU 17.4 + draft 2.5 + ~1.7 sync; ideal overlap ~20 ms; pre-doorbell kernels critical in both windows | measured (profile-fasthip) | n/a | n/a |
| Fact: fillable GPU slack per layer = CPU - hits: ~40 us short, ~275 us edit (bounds 47 and any idle fill) | derived | n/a | n/a |
| Fact: NVMe Kingston NV2 PCIe 3.0 x4 (3.5 GB/s spec); disk tier wear ~2%/yr of 640 TBW at 30 parks/day; D 91% full | spec + est. | n/a | n/a |
| Fact: hit curve concave; VRAM beats PCIe gen on this host; 7900 XTX = same binary, +~2,900 pairs (56%) | est. | n/a | n/a |
| Fact: Gen4 x16 promotion ~50 us (28 GB/s measured on 7900 GRE); break-even link ~33 GB/s | measured (other box) + arithmetic | n/a | n/a |

### D. Other-hardware items (in plan, not actionable on this box)

| lead | number + hw | exact/lossy | in our src |
|---|---|---|---|
| 56 Hybrid P/E pool affinity default | 8P+12E: 12-16 workers beat 19 | exact | `--pool-affinity` include/strata/kernels/cpu/pool.hpp:73 |
| 58 CCD-aware pool on dual-CCD Zen 4/5 | ~64 GB/s per CCD cap | exact | 2026-11-15: park: hardware |
| 57 rest: AVX-512 without VBMI default; old Q2_0 folders on AVX2 | Skylake-X/Cascade Lake | rounding differs | `STRATA_VNNI_TIER` src/kernels/cpu/expert_layout.cpp:196 |
| 60 Strix Halo unified preset | 19-21 tok/s code (vs ~126 here) | exact | 2026-11-15: park: hardware |
| 62 gfx12 re-measure | R9700 pcie-frac 0.15 | exact | gfx12 defaults |

### Closed questions

- Physical P/D disaggregation does not apply: one GPU, one user. Strata's 2-GPU split serving lost on all 6 workloads (2x RTX A6000: 1.16x slower at 8K, 1.48x at 32K, 1.04-1.30x on chats). vLLM docs say disaggregation "DOES NOT improve throughput". (LR/rep:9,13,58,86; plan ledger (d) 1235)
- Chunked-prefill interleaving only protects other requests' TPOT. Chunk size stays a TTFT knob: 32K beat 16K with boundary checkpoints. (LR/rep:87)
- An SGLang/TRT-LLM overlap scheduler is already present in spirit (HIP graph + C++ scheduler). TRT-LLM's extra decode step conflicts with speculative verify. (LR/rep:60,88)
- A TokenSpeed-style typed FSM control plane, xLLM/vLLM-plugin service decoupling, a distributed KV store (Mooncake, Redis), PASCAL reasoning-aware scheduling and Exo: no effect at one request on one gfx1100. (LR/rep:89-95)
- Batch slots: each slot takes VRAM from the expert cache (11-24% on a 12 GB card). They help only with concurrent agents. (LR/rep:92)
- KTransformers' host-callback single-graph 1.23x is already banked by the HIP-graph verify window and WAIT_FUSE. (LR/rep:68)
- MTP-in-graph (item 10) is closed: the draft is already graph-chained, so its 2.2-2.5 ms is GPU work, not launches. The remaining lever is `STRATA_MTP_HIP_PRIO`. (plan 221-229)
- Bandwidth is not the bottleneck; overlap is. DDR4 is at ~80%, PCIe at ~93% on large copies, NVMe is idle in decode. No bandwidth technique moves the ~20 ms floor. (DP/rep:3,13,98)
- The DALI/HybriMoE rule "GPU if t_gpu < t_cpu" almost never fires at batch 1: CPU ~45 us vs PCIe ~115 us per expert. It belongs in the simulator only. (DP/rep:13,88)
- `hipMemcpyBatchAsync` is absent in ROCm 10 (`STRATA_DMA_BATCH` is CUDA-only). 1.4 MB promotions already lose only ~7%. (DP/rep:76)
- Q2_0 wire or entropy compression is closed: down tensors are at most 12% compressible, IQ at most 3.4% (A29). (DP/rep:77-78; plan 1789)
- Zero-copy GEMV on missed experts over Gen3 costs >=107 us vs 42 us on the CPU. (DP/rep:79)
- GPU-initiated I/O is out: `kfd_ais_rw_file` is missing from the kernel, the hipFile fastpath needs ext4/xfs, and BaM is NVIDIA-only and broken on 6.x+ kernels. VRAM fills from RAM anyway. (DP/rep:80)
- A filesystem move does not help: ntfs3 already has iomap reads, and the 7.1 "ntfs" driver declares identical reads. ntfs-3g (FUSE) loses badly on random PLE reads. (DP/rep:37,81)
- LUT or bit-serial Q2_0 rewrites (T-MAC, Vec-LUT) are out: the kernel is at the roofline, and `row_multi<NT>` already amortizes over tokens. (DP/rep:25,82)
- Non-temporal loads/stores and extra cache blocking: nothing to gain. (DP/rep:83)
- Megakernels (MPK, FlashDMoE, MonoMoE) have no RDNA3 backend. The GDN decode megakernel gave 0.0 ms (measured). (DP/rep:84)
- `hipStreamWaitValue32` or host callbacks instead of the spin: beta API, and it brings back launch and OS-wake latency. Spin backoff and `s_setprio` are not needed: detection is bound by the PCIe round trip. (DP/rep:85-86)
- Klotski, PowerInfer-2, LIA and a MoE-Lens scheduler all assume batch > 1, phone NVMe or AMX/CXL. (DP/rep:89)
- An online bandit or BO on scheduler knobs cannot work with ~20% run-to-run noise. Tune offline. Only DraftPolicy stays online. (DP/rep:90)
- Experts on SSD by design: the arena fits in RAM, and NVMe is ~8x slower than DDR4. RAM_KEEP=20 decoded ~8% slower (measured). (DP/rep:91)
- cache_ext (eBPF) needs a modified kernel. A UMWAIT port to Linux gives nothing: a desktop dGPU has no shared CPU/GPU power budget. (DP/rep:92,94; plan 1228)
- L3 prefetch (`STRATA_ARENA_WARM`) and X3D for weights are out: experts are read once per window. 96 MB of L3 holds only ~65 experts. (CP/rep:87,173)
- Vega gfx90c (5700G iGPU) and 780M gfx1103 are not in the ROCm matrices. Vega is Vulkan-only here. (CP/rep:13,174)
- RDNA4 FP8/BF8 WMMA and SWMMAC 4:2 do not apply: experts are integer codes, and Flash-Next experts are dense. RDNA2 is unsupported. (CP/rep:175-176)
- An 8000G APU host cuts the GPU slot to 4.0 x8/x4. AMX and Intel iGPU/NPU have no path in pancake. (CP/rep:170-172)
- A static per-expert ratio (PcieAuto) is wrong on this box. Calibration must measure window time, because DMA and the pool contend for DDR. (CP/rep:83)
- A 6-core AM5 part is not automatically faster per expert: Zen 4 7600 at 6 threads did 19.7-24.2 GB/s vs ~27 GB/s on the 5700G. (CP/rep:65)

---

## Part E: avalon/literature leads (7 reports; notes used for cross-check only)


Dropped as off-target: cluster/batch systems (DeepEP, EPLB, MegaScale, Comet, Mooncake, MoE-Gen, NEO, Pie-batch, Hydragen, Tokasaurus...), retraining-only (NSA, MoBA, Kimi Linear, ProSparse, EAGLE-3, Pre-gated, Read-ME, PASTA, SEAL...), pure agent/NLP with no engine or gate link (AutoCodeRover, AutoCompact, ECLoop, STAIR, Aegis, AsyncFC, Slipstream, RLM, MEM1, Self-repair, CodeSteer, ACE, DSPy/MIPRO/GEPA, AWM, SWE-agent, LLMCompiler, KNUST tools).

### Expert path, CPU pool, cache (decode-critical)

| lead | number + hw | exact/lossy | in our src |
|---|---|---|---|
| KTransformers expert deferral (low-weight misses finish under next layer) | DS-V3 -26%/layer, +33% decode, up to +45%; tasks <=2 pts (NVIDIA) | L | include/strata/core/defer_rank.hpp:3 |
| Rank guard "top-2 or weight>=w" | est +10-25% decode after HIP fix | L | include/strata/core/defer_auto.hpp:91 (`_TOP`), defer_rank.hpp:48 (`_FOLD`) |
| Per-expert barriers + fused combine in CPU pool | est 1-4% exact; plan measured +2.5% edit | E | include/strata/kernels/cpu/pool.hpp:382 (`STRATA_POOL_FUSE_PHASES`) |
| Cache-conditional routing (= `turbo`/`STRATA_ROUTE_PRIOR`) | up to 2x phones; ours 111.7->149.3 tok/s 7900 XT, 2.9% diff tokens (3060) | L | include/strata/kernels/route_prior.hpp:12 |
| AcceptMoE depth-graded verify prior | 1.29x resident / 2.06x offload, traffic -74-77%, -0.27 pts (SGLang) | L | route_prior.hpp:12 (`STRATA_ROUTE_PRIOR_DEPTH`) |
| OEA sibling-aware routing in verify window | -39% latency at batch 16 | L | 2026-11-16: CODED STRATA_OEA |
| HybriMoE / AdapMoE / MoBiLE / BuddyMoE (turbo-family variants) | 1.70x; 1.60-1.72x; BuddyMoE -5% ARC | L | 2026-11-15: CODED STRATA_BUDDY (+ MISS_SKIP_W, DALI) |
| Cost-aware Belady ceiling (Baleen method) | promo ~115 us vs ~42 us CPU per 1.4 MB expert | - | tools only (time_ceiling.py) |
| k-th-miss admission (SIEVE/S3-FIFO probation) | SIEVE miss -1.5% avg (up to 63%) | E (rounding) | include/strata/core/epoch_bandit.hpp:9 (`STRATA_ADAPT_KMISS`) |
| ADRS/OpenEvolve policy search; Harmonia split | ADRS 3.7 ms vs 540 ms EPLB (inconsistent); Harmonia +49.5% | E | 2026-11-15: method |
| MoE-Infinity early-layer bias + nearest-conversation seeds | 17% hit @10% cache (MoE-Beyond replication) | E | 2026-11-15: already STRATA_CONV_WORKSET |
| fMoE embedding-matched expert maps (warm-up profile) | -47% latency, +36% hit (v1) | E | 2026-11-15: CODED STRATA_CONV_WORKSET_MATCH |
| ProMoE chunked/cancellable promotions + reordered exec | 1.09x over LRU avg (max 1.17x), llama.cpp PCIe4 | E | include/strata/program/promo_chunk.hpp:1 |
| Next-expert predictor recall test (Fate/HOBBIT/SP-MoE/SmallThinker pre-attn) | papers 84.7-97% on 8-64 experts; ours 19-21% recall | - | include/strata/core/expert_source.hpp:345 (`STRATA_LOOKAHEAD_STATS`) |
| ExFlow P(l+1\|l) affinity table | affinity from ~5k tokens | E | 2026-11-15: CODED STRATA_FS_AFFINITY |
| KVFlow-style expert pre-staging during tool runs | KVFlow 1.83x vs SGLang HiCache | E (rounding) | conversation_disk_env.def:439 (`STRATA_IDLE_ADAPT`) |
| FreeToken q* CPU/PCIe split vs `--pcie-frac` | q*~0.36m; ours pcie-frac 0 best (+6/+13%) | E | include/strata/core/calibrate_ratio.hpp:14 |
| Fiddler token-count threshold for CPU share (prefill/verify) | - | E (rounding) | include/strata/core/arch_defaults.hpp:30 (`PREFILL_CPU_SHARE`) |
| Athena epoch bandit for mixed opt-ins / POOL_PREREAD | 0-3% | E | include/strata/core/epoch_bandit.hpp:10 |
| Activation sparsity masks (Berkeley gate / TEAL / CATS / SpAx / CHESS / per-family) | Berkeley 84.5-87.4% Qwen3.5; TEAL 40% PPL 6.21 vs 5.87; SpAx 2.06x@4bit | L | 2026-11-16: CODED STRATA_TEAL |
| BlockFFN union-over-window sparsity column | 80% -> 71% at 8-token union | - | 2026-11-16: CODED STRATA_TEAL_UNION |
| CETT-PPL-1% per-layer threshold (Sparsing Law) | - | - | 2026-11-15: method |
| Transposed n_ff-major down layout (PowerInfer) | prereq for any column skip | E | 2026-11-16: CODED STRATA_DOWN_T |
| T-MAC LUT GEMV / Q2_0 kernel micro-bench | T-MAC up to 6.6x vs 2024 llama.cpp | E | 2026-11-15: method |
| HOBBIT/MC-MoE low-precision copy for misses | MC-MoE 2.54 bit, <0.6% loss; HOBBIT <=1% GSM8K | L | include/strata/artifact/shadow_pack.hpp:1 (A27) |
| IQ3-hot / IQ2-cold per-tensor mix from two GGUFs (Precision scaling law + LQ-LoRA ILP) | PTQ loss grows with pretrain tokens (<=1.7B fit) | L | 2026-11-15: method |
| MatGPTQ matryoshka file (4-bit slice VRAM, upper bits RAM) | - | E/L | 2026-11-15: method |
| Any-Precision bitplanes for draft | n/a: MTP drafter's 512 experts in VRAM (835 MiB) | - | 2026-11-15: method |

### Drafting, verify, recurrent state

| lead | number + hw | exact/lossy | in our src |
|---|---|---|---|
| MTP depth 4 (Qwen tech report) | 4.07 accepted @4 steps; SGLang 3.3 incl bonus, 540 tok/s B200 | E | preset `--spec 4` (configs/pancake) |
| DISCO drafter-entropy feature (MiMo) | MiMo acc <=3.6, 2.6x with 3 MTP layers | E | include/strata/spec/disco.hpp:30 (entropy "not exposed") |
| SpecFit per-position new-miss cost | +21.6% / +7.0% over EAGLE-3 same cache (Qwen3-30B) | E | include/strata/core/row_miss.hpp:2 |
| Top-k entropy (Entropy Sentinel) in DISCO + loop guard | r=0.91 vs judge (some models) | E/- | serve/rumination.py:19 (entropy opt-in) |
| Draft source as learned action (bandit) | 0-5% est | E | include/strata/spec/draft_source.hpp:1 (source, not learned) |
| SuffixDecoding cross-session index + frequency scoring | up to 3.9x SWE-Bench, ~2.2x vs EAGLE-2/3 | E | include/strata/spec/suffix_index.hpp:1 (A24) |
| MTP per-position decay check (SD Illusion) | 0.92/0.68/0.38; verify 42-95% of time | E | 2026-11-15: already spec depth + accepted_hist |
| Typical acceptance on thinking only (Medusa) | est 5-15% on thinking | L | 2026-11-16: CODED STRATA_TYPICAL_THINK |
| Cactus KL-budget acceptance | acc 4.27->5.44, 1.88x vs 1.81x (T>0) | L | 2026-11-15: park: MOD_PLAN 8 scope |
| AdaSD adaptive length rule (as DISCO check) | - | L (relaxed accept) | 2026-11-16: CODED STRATA_SPEC_ENTROPY=adasd (length rule) |
| Token Recycling drafter | ~2x SpecBench, <2 MB matrix | E | 2026-11-16: CODED STRATA_TOKEN_RECYCLE |
| SSD-lite (Saguaro): precompute ngram drafts on CPU during verify | Saguaro ~30% over best SD (dedicated H100 drafter) | E | 2026-11-16: CODED STRATA_SSD_LITE |
| Staged SD / Hydra top-2 tree | Hydra 1.31x vs Medusa; staged 3.16x GPT-2-L | E | 2026-11-16: CODED STRATA_HYDRA_TOP2 |
| ReplaySSM audit of GDN replay (u ring buffer) | 1.87-1.96x batched vLLM; ours est 1-3% decode, 0-8% spec | E | 2026-11-15: method (superseded by GDN_SPECLA) |
| SpecLA: recover accepted GDN states from verify factors (no replay) | - | E | 2026-11-15: CODED STRATA_GDN_SPECLA |
| LongSpec/Hybrid Tree: verify rows read prefix KV once | ~75% less attn latency vs HF | E | 2026-11-16: CODED STRATA_ATTN_QMAJOR |
| Exact speculative sampling for T>0 requests | - | E | include/strata/core/coupled_draft.hpp:65 (`STRATA_SPEC_PROB`) |

### GPU kernels, attention, prefill

| lead | number + hw | exact/lossy | in our src |
|---|---|---|---|
| Gated Residual small-M fusion (SGLang M=4) | Mix 2.05x, Combine 1.96x; +7.6% / +5.49% e2e (B200) | E | include/strata/core/switch_report.hpp:35 (`STRATA_EXPDEC_SWIGLU`); GR_FUSED in conversation_disk_env.def:418 |
| GDN prefill chunk 64 + 16x16 block inverse + fused prep (gfx11 path is chunk 16) | est 5-11% prefill (GDN 22% of prefill) | E (FP16 parity) | include/strata/prefill/gdn_chunk.hpp:1 (sm_70 only, chunk 64) |
| BFLA sparse prefill on QSA layers (measured on Flash-Next) | 1.342-1.435x TTFT @128-256K, LongBench 57.32=57.32 (H200) | L | include/strata/kernels/bfla.hpp:1 |
| XAttention block-sparse prefill >64K | up to 13.5x @256K vs FA | L | 2026-11-16: CODED STRATA_XATTN (q36) |
| Quest / Twilight sparse decode attention | Quest 2.23x self-attn | L | 2026-11-15: CODED STRATA_QSA_PAGE_SEL |
| FLUTE duplicated LUT in LDS (bank conflicts) | 0-10% GPU expert time if conflicts | E | include/strata/kernels/iq_kernels.hpp:136 (`STRATA_IQ_LUT_DUP`) |
| UFMG coordinate-descent launch/tile sweep | 5.5-8.5% untuned (A100); est 1-3% ours | E | 2026-11-15: method |
| SageAttention/SpargeAttn | CUDA only | L | 2026-11-15: park: CUDA only |
| XGrammar-2 TagDispatch (grammar masks only after tag) | - | E | 2026-11-15: already grammar/guard.hpp:36 |
| QuaRot folded Hadamard (if rotated GGUF appears) | Llama2-70B 4-bit <=0.47 PPL | E (runtime) | include/strata/kernels/cpu/expert_layout.hpp:109 (hadamard path) |
| LQER/QERA low-rank error correction | - | L | 2026-11-16: CODED STRATA_LQER |
| QTIP / IQ*_KT | ~0.2 bpw gain (ik) | - | src/kernels/cpu/tq_computed.hpp:7 |
| YAQA KL-based imatrix | -30% KL, needs backward | - | 2026-11-15: method |

### Model files, quality gates, server

| lead | number + hw | exact/lossy | in our src |
|---|---|---|---|
| Q2_0 over IQ2_XS (DASLab card) | 93.8 vs 70.3 tok/s decode, 89.07 vs 89.16 avg (llama.cpp) | - | n/a |
| IQ1_M gate/up in 3 layers of IQ2_XS file | est 1-4% decode on IQ2_XS, 0 on Q2_0 | E | src/kernels/cpu/iq_avx2_rows.inl:226 (IQ1_M rows now native; iq_avx2.cpp:12 comment stale) |
| Coder variant A/B (256/512 experts, 29.6 GB) | SWE-V 79.20 vs 82.80 BF16 (95.7%), LCB v6 87.43 | L | include/strata/artifact/model_family.hpp:17 (name only) |
| Request "Coder Q2_0" build from DASLab | - | - | n/a |
| EAR metric (SLQ) beside same-top-token | lossless = EAR>=0.99, KL<~0.01 | - | tools (quality_gate --ear planned) |
| KLD/top-1 blind in quiet zone -> task gate decides | rho 0.00 (Qwen3.6-A3B), -0.24 (Devstral) vs -0.72/-0.86 cohort | - | tools/pancake/task_gate.py |
| BFCL v4 + LCB + HELMET gate ladder | - | - | tools (task_gate) |
| SWT-Bench Lite as lossy gate | - | - | 2026-11-15: method |
| SWE-Effi AUC time-to-success | failures ~8.8M tok/658 s vs 1.8M/167 s | - | 2026-11-15: method |
| Repeated runs as distributions (Price Reversal) | thinking tokens vary up to 9.7x | - | tools/pancake/abtest.py |
| Restore-equivalence test (UNSW hybrid-state bug) | 34/36 -> 36/36 after fix (vLLM) | - | tools restore_check.py |
| preserve_thinking end to end (opencode resend) | re-prefill from first dropped think block | E | serve/chat_template.jinja:119 |
| Checkpoints at tool boundaries + Marconi FLOP/byte eviction | Marconi 34.4x token hit, TTFT -71.1% | E | include/strata/program/conv_cache.hpp:121 |
| Multi-entry prefix store (RadixAttention) | up to 6.4x throughput | E | include/strata/core/conversation_cache.hpp:533 (`STRATA_AUTO_PIN`) |
| Continuum/MORI TTL for parked sub-agents | MORI +20-71% thru (Claude Code traces) | E | 2026-11-15: already STRATA_PARK_TTL_S / _SUB_RATIO |
| Graceful close at thinking cap (Interruptible LRMs) | up to -60% acc under abrupt cuts | L | serve (grace 512 + close_retry) |
| Learned thinking stop probe (Thought Calibration / LHHT) | -60% thinking ID, -20% OOD, no loss | L | include/strata/core/verify.hpp:130 (`STRATA_THINK_PROBE` dump) |
| Rumination 5-gram rate -> soft close | no threshold in paper | L (opt-in close) | serve/rumination.py:1 |
| Loop-guard behaviour counters | failed runs +39-343% tokens; Qwen Code 25.81% | - | serve/agent_loop.py:9 (A38) |
| oldString substring constraint | ~94% compile errors are type errors; halves them | L | serve/agent_loop.py:22 (A43 check) |
| presence_penalty = 0 on tool turns | SWE-bench 74.7 -> 70.6 with it on | - | serve/server.py:1829 |
| Effort per task: xhigh vs medium | Qwen: low effort can lengthen tasks; Overthinking -43% compute | L | n/a |
| PLE path: keep `--ple-io direct`, measure prefetch time + row-cache hit | 1M-row x 90 B cache | E | src/program/generate.cpp:695 |
| Official sampling T=1.0/top_p .95/top_k 20; no static YaRN | - | - | n/a |
| Tool-argument hidden-state probe (LLMs Know More) | probes do not generalize across datasets | - | 2026-11-15: method |
| ContextSniper tool-output filtering before context | -51.5% tokens SWE-bench Lite | L | 2026-11-15: CODED STRATA_TOOL_SNIP (serve) |
| Calibration on opencode traces (only at requant) | PPL -3.52 (4-bit dense 7-8B) | - | 2026-11-15: method |
| Mind the Gap: official source + SHA-256 vs HF LFS | attack delta 88.7% insecure code | - | 2026-11-15: method |
| AgentDojo injection run vs opencode+pancake | ~97 tasks, ~629 security cases | - | 2026-11-15: method |
| Demons in the Detail: global-batch LBL -> session-stable expert sets | - | - | 2026-11-15: method |

Counts: 88 leads; not in plan (no item, or index-row only): 20 (MatGPTQ, IQ3/IQ2 mix, CETT, AdaSD, SpecLA, XGrammar-2, YAQA, Coder-Q2_0 request, SWT-Bench, PLE measurement, official sampling note, tool-arg probe, ContextSniper, calibration, Mind-the-Gap hash, AgentDojo, Demons, fMoE (partial via A10), ExFlow (partial via item 30), MTP-depth note).

### Closed questions

- Pancake MTP depth 4 matches Qwen's trained depth (4.07 accepted); `--spec 3/5/6` measured flat, so only the stop rule can gain, not the ceiling (AS:30).
- SGLang draft shortcuts already done: drafter has no PLE block, dense KV-ring window and no QSA indexer (same choice as DeepSeek-V4 SWA-only MTP head) (AS:21-22, AS:32).
- PLE default `--ple-io direct` is right; do not switch to mmap. EU note "mmap stays" was wrong; `STRATA_PLE_AUTO_RAM` correctly refuses the 28.8 GB shard (AS:32 corrects EU:25).
- FP8 Gated Residual stream is irrelevant at batch 1 (~4 MB/token) and RDNA3 has no native FP8; kernel count matters (AS:34).
- Any-Precision for draft steps does not apply: MTP drafter holds all 512 experts in VRAM (835 MiB) (AS:64).
- Per-expert mixed precision (MC-MoE/HOBBIT/MxMoE) has no room on Q2_0; only on a >2-bit file (AS:64, AS:180) - partly reopened as A27 miss-only copy (plan 1936).
- Expert prediction on PCIe 3.0: gains over LRU small (ProMoE 1.09x); measured next-layer recall 19-21% here, predicted promotions do not pay (AS:56; plan 976-983).
- Layer-skip self-speculation (SWIFT, CAS-Spec) breaks GDN state; alpha=0.038 at k=2 on Qwen3.5 hybrids (AS:70).
- KV compression (KVzip, TurboQuant, OSCAR, ZipCache, MiniCache, QServe KV4) gives nothing: KV only on 1 of 4 layers; extra VRAM does not buy decode (AS:72, MH:34, PA:79).
- Cactus does nothing at greedy and ~0 at T=0.1 (CA:35).
- FlashMoE learned cache only +7% over LRU; Belady gap already measured (CA:39).
- Q2_0 beats IQ2_XS (93.8 vs 70.3 tok/s, 0.09 pt) - item 6 closed; base is 176.9B (354 GB BF16), requant at home impossible (GSQ needs 8-16x H200) (EU:15-17).
- `maddubs` Q2_0 offset trick already in `q2_avx2.cpp` (EU:19).
- 2:4/4:8 weight sparsity cannot pay on AVX2 (metadata +0.5-0.75 bpw) and GGUF has no 2:4 (EU:47; plan 1198).
- Hermes-style miss prediction redundant with router lookahead (EU:53).
- DISCO/TurboSpec/Sequoia/TLT already equal `DraftPolicy` for chains (EU:39, US:37, MH:72).
- CATS collapses at 40% on dense (PPL ~7.6e4); later the offline pass showed Flash-Next FF-640 experts are dense (gate cos 0.844 @70%), closing item 13 (MH:13; plan 919-927).
- GDN megakernel at decode gives 0.0 ms: window waits on CPU rows (MH:7).
- HIP graphs not to be retested without a profile showing inter-kernel gaps (ST:49).
- FreeToken split is the same mechanism as `--pcie-frac` (0 best, +6/+13%), likely closed by code read (US:33; plan 952).
- Hit rate is not time: a PCIe 3.0 promotion costs ~3 CPU computes, so the 79.6/89.5/94.7 gap overstates the prize (US:27).
- Draft trees (Staged SD, Hydra, SpecInfer, Lookahead/Jacobi) are bad here: GDN state fork per branch and wider expert union (US:39-41, ST:47).
- Non-prefix reuse (Prompt Cache, ContextPilot) breaks GDN state; ContextPilot measured 1.6% on real sessions, stays off; Entropy Gate rejected (US:49, PA:36).
- Grammar must stay deferred until `</think>` (Let Me Speak Freely) - already true in guard.hpp (AS:72).
- No paper anywhere covers CPU-expert offload on AMD/ROCm, PCIe 3.0, 512 experts or a GDN hybrid - every author number is an upper bound (PA:3, AS:194, US:192).

---

## Part F: avalon/upgrades-from-upstream + avalon/prose-gap leads

Tree: `C:\Users\iancu\repos\pancake-strata`, branch `docs/consolidate-plan`. "in our src" = `git grep` over `src include` (serve/ named where the lead is server-side). "in plan" = MOD_PLAN.md line.
Dropped as irrelevant to a single 7900 XT box: NVIDIA-only kernels (#31802, #31533 as written, #27140, #28714/#28195/#26301 RDNA3.5), multi-GPU (#29963, Avalon pipeline stages / expert split, Maya `--tp`, `STRATA_DISJOINT_ADAPT`), Vulkan (#31139), DSpark/EXL3/DFlash2-GLM, Responses/realtime/router API items, GLM-5.3 on Maya.

### Kernels / GPU (upstream llama.cpp, Strata)

| lead | number + hw | exact/lossy | in our src |
|---|---|---|---|
| llama.cpp #31751 `#pragma unroll 1` stops J=32 MMQ register spills (Q8_0-layout types) | PP17 +36.5%, PP32 +36.1%; W7800 gfx1100, ROCm 10.1, Q8_0 | exact | 2026-11-15: CODED STRATA_MMQ_RDNA3_J32_ROLLED |
| llama.cpp #31746 D256 MMA FA tune for gfx1100 | attn op 0.70-0.84x of master; W7800 ROCm 10.1 | exact (rounding) | 2026-11-16: CODED build macro STRATA_PA_SCHED_FENCE |
| llama.cpp #31234 / #30792 native q4/q8 KV in FA tile kernel | q4_0 tg +13% @64k, +24% @131k; R9700 RDNA4 | exact | 2026-11-16: already qsa_decode_attn.cu int8 / q4_0 KV |
| llama.cpp #29948/#31188 (+#30489) gate/up+GLU (+Q8_1 quant) fused in one launch | pp16384 +6-17% NVIDIA; gfx906 -15..-50% (#31599) | exact | yes, own: `src/kernels/cuda/s_gemv_pair.cu:1` |
| llama.cpp #31047 linearized MoE layout / tile tuning | 2-4%; NVIDIA | exact | 2026-11-16: CODED STRATA_MMQ_LINEAR |
| llama.cpp #31443 / #31660 qwen4exp conv-state copies (tail w/o cont; snapshots in one set_rows) | "free 2%" QFN gen; NVIDIA | exact | 2026-11-15: already gdn_conv_commit |
| llama.cpp #31625 HIP fast-math | none given | lossy-risk | 2026-11-15: CODED cmake STRATA_HIP_FAST_MATH |
| llama.cpp #31838 fixed rocBLAS workspace (= U4 432433637) | untested | exact | N/A (rocBLAS opt-in only; hipBLASLt workspace once) |
| ROCm 10.1 compiler: ~126 kernels regressed vs 7.2.4 | gfx906 Discussion #30622 | n/a (A/B rule) | n/a |
| Strata hipBLASLt tuning table matching runtime (100401, not 100200) | prompt +32/+78/+82% @3K/6.5K/131K; 7900 XTX | exact-ish (GEMM solution choice) | yes `src/core/conversation_disk_env.def:197` STRATA_HIPBLASLT_TABLE |
| Strata `PA_FAST` + `PF_FUSED` + `PREFILL_RING=512` | +13.8% @94K deep prefill; 7900 XTX | lossy (not bitwise) | yes `include/strata/core/arch_defaults.hpp:7`, `include/strata/prefill/prefill.hpp:129` |
| Strata `STRATA_TSUM=1` | -0.91% decode time; 7900 XTX | exact (bitwise) | yes `src/core/arch_defaults.cpp:53` (arch table) |
| Strata exact long-prompt QSA scorer `STRATA_SCORES_THREAD=1` | +10-13% @131K, +22% @240K; 7900 XTX | exact | yes `src/core/conversation_disk_env.def:679` |
| Strata int8-WMMA prompt attention (#5535), expert layouts, EXPERT_V2, EXPDEC (#5458) on gfx1100 | 0.995x e2e; layouts 2-3x slower; 7900 XTX | mixed | 2026-11-15: already (measured, MOD_PLAN 9) |

### MoE tier, I/O, memory

| lead | number + hw | exact/lossy | in our src |
|---|---|---|---|
| Strata PR #5561 `madvise(MADV_SEQUENTIAL)` on expert mmap (`STRATA_EXPERTS_SEQ_READAHEAD`) | follow-ups 65-76 s -> 23-28 s, decode 33-44 -> 42-52 t/s; RX 6900 XT, 32 GB RAM | exact (byte-identical) | 2026-11-15: CODED STRATA_EXPERTS_SEQ_READAHEAD |
| Maya direct parallel expert reads (`MAYA_EXPERT_DIRECT`, `MAYA_EXPERT_STAGERS` 2-16, probed per drive) | decode +26% RAM-short (V100, 30 GB RAM); probed readers +5% | exact | 2026-11-16: CODED STRATA_STAGER_PROBE (+ open_direct, STAGER_THREADS) |
| Out-of-core O_DIRECT streaming #31469 / #25294 | TG 1.5 -> 16-22 t/s, 68 GB on 16 GB RAM; NVIDIA | exact | 2026-11-15: already UNBUFFERED_LOAD + SEQ_READAHEAD |
| Registered-RAM expert arena (Avalon 03c749e/213b569) / hot-expert mlock #26414 | none measured | exact | yes (arena; mlock note `conversation_memory.hpp:25`) |
| Router-predicted next-layer `pread` prefetch (`STRATA_IO_PREFETCH`) | not isolated | exact | yes `include/strata/core/expert_source.hpp:341` |
| llama.cpp #29599 PLE row madvise prefetch (qwen4exp) | pp512 177 -> 383; Strix Halo ROCm, 104 GiB | exact | 2026-11-15: already ngram WILLNEED, --ple-io direct |
| Strata PLE table resident in RAM (`STRATA_PLE_AUTO_RAM`) | 8K prompt +22.5%; R9700 | exact | yes `include/strata/core/ple_residency.hpp:32` |
| ClassLRU/CRF eviction (#30637 -> #31233, #31853) | +6-7% decode QFN vs SLRU; 4090/5090 | exact | 2026-11-15: method (replayed, T-395 / T-452); engine port CODED STRATA_ADAPT_CLASSLRU |
| CPU-miss share + tuner #30966 | QFN 24.2 -> 47.6 t/s @6.4 GB cache | exact | yes (`--pcie-frac`, PCIe share) |
| #31888 CPU part does not wait for cached experts | +7-9% TG Gemma 4; 0 on QFN | exact | yes-ish (`STRATA_ROUTE_PUBLISH`) |
| Avalon prefill knobs: `--moe-cache-cpu-overlap 2`, `GGML_SCHED_PREFETCH_EXPERTS=6`, `-ub 19072` | prefill 2,113-2,130 t/s; 7900 XTX PCIe 4 | exact | Avalon-only; analog `PREFILL_CPU_SHARE`, ring, chunk |
| Strata `STRATA_ADAPT_FETCH` (fetch admission) | AdrianBM96 3060 +5-7% (plan) | exact | yes `include/strata/core/expert_source.hpp:406` |
| Strata `STRATA_ADAPT_OVERLAP=1` | +4-8% A4000/P100; loses 5070 | exact | yes `include/strata/core/adapt_async_default.hpp:31` |
| Strata `STRATA_PCIE_BALANCE=1` | +5.6% A4000 | exact | yes `include/strata/core/nv_defaults.hpp:7` |
| Strata `STRATA_KV_PREFETCH=1` (next layer KV upload in prefill) | not stated | exact | yes `src/core/conversation_disk_env.def:463` |
| Maya `--calibrate`-style sweep (CPU threads, pcie share, read size; keep > 3%) | few % to ~10% (unknown) | exact | 2026-11-15: already calibrate_ratio.hpp |
| Maya miss skip by gate-weight share (`MAYA_GLM_MISS_SKIP_W`/`_FROM`) | +7-13% decode, KL 0.037-0.052; RTX 3090 GLM | lossy | 2026-11-15: CODED STRATA_MISS_SKIP_W |
| Strata `STRATA_ROUTE_TAIL_SKIP=7` on gfx1100 | pancake "tail skip 7 129.7" (829); KL 0.003-0.006 | lossy | yes `include/strata/core/expert_source.hpp:670` |
| Strata router residency bias `ROUTE_PRIOR` / `CACHE_ROUTE` (PR #5596) | code +37% but agent 5.6 -> 4.5/10; DS +56% top-1 93-96% | lossy | 2026-11-15: already ROUTE_PRIOR |
| Strata KVA approximate late-layer prefill | prefill +28%, decode -11%; R9700 | lossy | yes `include/strata/core/defaults_026.hpp:7` |
| llama.cpp #31886 ROCm staged <=64 MiB pinned uploads from mmap | load-hang fix | exact | N/A (own loader) |

### Speculative / drafting

| lead | number + hw | exact/lossy | in our src |
|---|---|---|---|
| Longer prompt-lookup window `--lookup-max-t 16` | +11-14% on edits; RTX A6000 IQ3_S | exact | yes `include/strata/core/lookup_default.hpp:1` (HIP kVerifyLongT 16) |
| llama.cpp #29548 ngram-simple keeps scanning older matches on periodic text | 0 -> nonzero drafts (198 calls) | exact | 2026-11-15: CODED STRATA_LOOKUP_OLDER |
| `--spec 3` (cap 2 drafts, keep `--spec-min-p 0.5`) | prose 0..+2% est.; code may lose | exact | yes (`--spec`, generate.cpp:4352-4356 per note) |
| MTP head quality match (Avalon IQ3_K/IQ4_NL 1.48 GB, FR 65k) | Avalon +5-9 pts prose acceptance -> up to ~5% | exact | 2026-11-15: method |
| Draft vocab 42.7k proxy list vs 40.5k | misses 4.27% -> 3.68% offline; <1% | exact | yes `include/strata/core/mtp.hpp:68` |
| Strata `STRATA_DF2_GATE` calibrated verify-window gate | none (Mac) | exact | yes `include/strata/spec/window_rule.hpp:73` (DFlash2 only) |
| DFlash2 drafter | +14% M5 Max, +7%/+1% gfx1201, slower Spark; HIP reverted to MTP | exact | yes `include/strata/core/dflash2.hpp:1` |
| Adaptive / deeper MTP depth #27210, #27173 | up to 151 t/s recall @depth 12; +10% @depth 5, 2x 5090 | exact | 2026-11-15: already DraftPolicy + DRAFT_CLIP |
| Few-row MMQ for verify T 2-24 (#31533 = U1) | Ampere only | exact | 2026-11-15: park: Ampere only |
| Suffix decoding / tree spec #26283 / #30931 | unmeasured | exact | suffix drafter yes (`--suffix-draft`) |

### Server / agent (serve/ is Python)

| lead | number + hw | exact/lossy | in our src |
|---|---|---|---|
| Checkpoint at last-user-end / tool-request invalidation (#30745, #24891) | reuse 91 -> 94%, save 3.6 -> 0.8 GB; V100 | exact | yes analog: `STRATA_CACHE_MESSAGE_BOUNDARY` (env.def:321), `--prompt-cache-tail` (`include/strata/program/conv_cache.hpp:43`) |
| #30137 ref-counted checkpoints | grows with context; 3x Arc B60 | exact | N/A (own conversation cache / parking) |
| Disk prompt cache #28092 | avoids ~70 s cold prefill @131k | exact | yes (conversation disk tier) |
| #30188 `cache_prompt:false` title requests do not evict agent prompt | none | exact | unchecked (serve has `strata_checkpoint` per request) |
| Multi-slot #30992 / #30970 / #29738 | latency only | exact | n/a |
| Strata thinking skip for short answers (PR #5512) | 0/6 -> 5/6 answered @max_tokens 24; median 0.72 -> 0.50 s | lossy (behaviour) | 2026-11-16: CODED STRATA_THINKING_POLICY (serve) |
| Strata `POST /v1/tune` + `--conv-route` (PR #5516) | single-shot TTFT -42%; prefill +10.1% code w/ cache off | exact | 2026-11-16: CODED STRATA_TUNE, STRATA_CONV_ROUTE (serve) |
| Maya prompt memo (incremental re-tokenize of re-sent history) | follow-up -6% @115K; 3x V100 GLM | exact | yes `serve/loop_guard.py:365` (STRATA_PROMPT_MEMO) |
| Keep-alive after streamed response #30492 | 1 TCP reconnect/turn saved | exact | unchecked |
| Tolerant tool-call recovery (98.3% vs 92.9% of 1,462 turns) + Maya unclosed `<parameter=` | Strata corpus, Qwen3.8 | exact | yes `serve/frontend.py:1406` |
| `env_to_user`, `effort_position: end` | stable prefix (~14K tok / 8-11 s per new day/folder) | exact | yes `serve/frontend.py:405`, `serve/runconfig.py:32` |
| `reasoning_close_retry`, `answer_room`, loop guard, soft budget grace | unmeasured | exact | yes `serve/server.py:237,5034`, `serve/loop_guard.py` |
| Watchdog, `fit_max_tokens`, any-model-name | reliability | exact | yes `include/strata/core/progress.hpp:5`, `serve/runconfig.py:29` |

### Prose gap: tests and baseline hygiene

| lead | number + hw | exact/lossy | in our src |
|---|---|---|---|
| Control arm: Avalon with `GGML_CUDA_MOE_TAIL_SKIP=0` (its default 7 is lossy, bench left it on) | gap 3.6-8.4% may be lossy path | control | n/a (Avalon `moe-cache.cu:9677`) |
| Control: Avalon head on matched IQ3_KS main (bench paired it with IQ2_XS) | unmeasured | control | n/a |
| A-A `base_rep` arm + any-base-hash text check (abtest compares only base rep 1) | 6 distinct hashes in 6 runs (story); between-start CV ~5% | method | 2026-11-15: method |
| Add the 5 prose prompts to `abtest.prompts()` | none | method | 2026-11-15: method |
| Profile fixed ~14 ms/window (1 ms ~ 5% decode; ~2.2 ms per extra draft) | derived, 2-point fit, night queue 7900 XT | profile | tools exist (gpu_profile.py, time_ceiling.py) |
| Greedy exactness check: `--spec-min-p 0.5` vs `0` vs `--spec 2` give identical text | none | exact claim | n/a |
| Lossy flags still in daily `flashnext-iq2_xs-fast-hip.json`: `STRATA_GR_RDNA_A8=1` (:98), `STRATA_PF_F16=2` (:88) | baseline not output-exact | lossy | yes (preset) |
| Rep budget: 3% needs ~12 starts/arm @CV 2.5%, ~45 @5%; state 5-6% bar otherwise | power formula | method | n/a |

Lead count: 66 rows. Not in plan: 14 rows with no plan line, plus 2 partly missing (#31443 half of the conv-copy row; the any-base-hash text check).

### Closed questions

- Prose gap mechanism: Avalon commits 2.27 tok/window at 20.1 ms vs pancake 1.96 at 18.5 ms (+16% tokens, +9% time, net ~+6%). Lever is fixed window cost and head quality, not Avalon's 2-draft rule (pancake's 3rd draft nets ~-4%). PR:9-13.
- Avalon has no prose-specific code path (grep of speculative/ngram/server/context: none); its edge is fixed policy (2 drafts, p_min 0, greedy) plus head. `-b/-ub` cannot explain it (3-row window either way). ngram-simple rarely fires on prose (drafted/window 1.98-2.00). AV:64,110,149.
- Gap is not established: 3.6-8.4% sits inside ~5% start-to-start noise (01A vs 04A same config differ -4.5..-19.7%). AD:62-73,95.
- Old MOD_PLAN 17.2 "same acceptance / steadier" is wrong: acceptance differs 2-9 pts, Avalon sd 2.7-4.7 vs 1.1-2.7 tok/s. 17.2 now rewritten (2608). PR:15.
- `--spec-min-p` is a window-width rule, not an acceptance filter: output-exact by `include/strata/core/verify.hpp:6-8`. PV:51-64.
- Pancake preset MTP window is 4 rows (max 3 drafts), lookup up to 12 rows on HIP; PROSE_GAP doc's "4 drafts" was wrong. PV:19-42,114.
- Measured losers, do not rerun: `--spec 2` (-5..-24%), `--spec-min-p 0` (flat..-6% prose, acceptance 52.8 -> 33.7%), route prior (-15..-25% prose), `STRATA_SPEC_COST=1` (flat prose, -7..-10% code). PR:39; AD:107-112.
- `--spec-draft-n-max` and any "verify batch" flag do not exist in pancake; do not use as arms. AD:9,115.
- abtest's text hash cannot certify exactness (same config gives different greedy text across and within server starts). AD:14-28.
- `STRATA_VERIFY_DEFER_CPU`: decided OFF in both daily presets (MOD_PLAN 2727); grep confirms it is absent from this branch's `flashnext-iq2_xs-fast-hip.json` (prose-gap note's ":104" is stale).
- Project Maya does not beat Avalon/pancake: its Qwen engine is a Strata 0.1.27 snapshot, 1.4-1.7x slower decode (#604), prefill 126 t/s @57K (#898). Only server items worth taking, already ported. PM:5-9,78-80; MOD_PLAN 861.
- Strata kernels on gfx1100 are near a local optimum (expert layouts 2-3x slower, int8 WMMA attention 0.995x); Strata on 7900 XTX (80.3 t/s, 1,381 t/s @32K) does not beat Avalon (~100, 1,805). UR:159.
- Upstream expert-cache PRs are ideas only; ClassLRU replay on our tier gave no gain (MOD_PLAN 859); pancake PCIe share = #30966 split.
- Freeing VRAM (q4_0 K, 131k ctx) gave nothing on this box, so "frees VRAM" items (#31669, #27489) are skipped. UR:237,244; AB:120.
- #28118 spec-checkpoint collapse and #27140 q4_0 prefill collapse do not occur here (spec 168-181 t/s; HIP prefill 1.8k t/s). UR:114,143.
- PLE conflict resolved: QFN does use a PLE/n-gram table (28.8 GB), but PLE-in-RAM needs model + table + 16 GB and cannot fit 57 GB. UR:105,242.
- Avalon issue: model folder missing and avalon.service disabled as of 2026-11-08; D needs ~20 GB freed before any Avalon comparison. MOD_PLAN 858 records source fast-forwarded to 88313b6. UR:16-22.

---

### Window overhead (Phase 18) + README/TASKS

Profile basis (short case, fast-hip preset minus deferral, `--spec 4 --spec-min-p 0.5`): decode 3553 ms wall, kernels 2712 ms (76%), 470 kernels/token, launch gaps <50 us 699 ms, host stalls >1 ms 93 ms. combine_cpu_k10_vec4<true> 670.6 ms (18.9% wall), wait_flag_ge 150.1 ms (288 calls), wait_flag_copy_i32 143.1 ms (4320 calls) (window-overhead/report.md:7-9). Ceilings below are % of wall.

| lead | number + hw | exact/lossy | in our src |
|---|---|---|---|
| Item 1: dump effective `STRATA_*` env + re-profile with per-stream gaps (map 48 mtp_select->copy and 83 copyBuffer->copy_i32_from_mapped pairs) | enabler; tool writes run_environment.json | n/a | gpu_profile.py (tools) DONE 2026-11-13 (T-565) |
| Item 2 COMBINE_POLL1 (one elected poller per combine) | ceiling 18.9%; measured short -4.2%, text differs | measured lossy(!) | yes native_moe.cu |
| Item 3 WAIT_FUSE 1 vs 0 | 0-3% est.; 12-rep: -2.0%/-0.2%, text differs | exact by class, observed text diff | yes verify.cpp |
| Item 4 FUSE_WAIT (with WAIT_FUSE off) | ceiling 0.3% (4320 nodes x 2.4 us); 12-rep -9.9%/-10.2% | exact by class | yes verify.cpp:441-462 |
| Item 5 `--spec 3/4/5` sweep | spec3 -3.2%, spec5 +1.2% (lossy, text differs) | greedy exact in theory; observed text diff | CLI |
| Item 6 ROUTE_FAST | ceiling 2.2% wall (78.1 ms), NOT 2.9% (that is % of kernel time); earlier +8.0% n=4 = noise | exact by design | yes native_router.cu:490 |
| Item 7 WAIT_COPY256 inert at `--spec 4`: wide copy needs plan n>512 words; plan=80*T+42 words, so acts only for T>=6; earlier +6.8% cannot be its effect | ceiling 0.3-0.4% | exact | yes verify_kernels.cu:1797 |
| Item 8 MTP_EAGER | ceiling 0.5% minus wasted steps; measured -3.6%/-3.9% | exact (code) vs def says "tokens change" | yes mtp.cpp:1863 |
| Item 9 merge 3 window-input copies via existing `copy_from_mapped_multi` | 0.04% (2 nodes x 288 windows) | exact | helper yes (elementwise.cu) |
| Item 10 pre-stage host records: premise stale (mtp.cpp:1821-1835 already stages); 16.9 ms pair likely `stage_source_R` D2D copy (mtp.cpp:2010) or sampled spec copies | ceiling 1.0% | exact | stage_source_R yes |
| Item 11 pool pinning: existing `--host-core first/last/sibling`, `--pool-workers 6/7/8/14`, `STRATA_POOL_MWAITX` arm; `--pool-affinity measure` is a NO-OP on non-hybrid 5700G (returns All) | ceiling 8.3% (293 ms waits) | exact (scheduling) | yes generate.cpp, pool.cpp |
| Item 12 event + device wait for window-end sync | ceiling 2.6% (93 ms stalls), realistic ~0 (argmax read is true dependency) | exact | 2026-11-15: CODED STRATA_EVENT_WAIT |
| Item 13 VERIFY_DEVICE_PLAN | <0.5%; measured -5.9%/-2.2%, text differs | exact by class, observed diff | yes |
| Item 14 ROUTER_FUSE w/o deferral | ~0.6-1%; TASKS: lossy, needs quality gate | check output | yes |
| Item 15 PIPED_MTP: inert without `--batch-mtp` + layer-split stages (generate.cpp:7194/7245/7406) | AMUSD 29% (paper excerpt) | claimed exact | yes |
| Item 16 `hipStreamWaitValue32` replacing spin kernels; `hipDeviceAttributeCanUseStreamWaitValue`=1 on 7900 XT (ROCm core-10.0) | ceiling ~0.3% node cost, latency unchanged; must honour `release_gpu_waits` UINT32_MAX watchdog (verify.cpp:612-625) | exact | symbol referenced (3 files) |
| Item 17 commit into window graph, device-side accept | <=2.6% + one launch | exact iff accept matches | 2026-11-16: CODED STRATA_DEVICE_ACCEPT |
| Item 18 fused argmax+accept (SonicSampler 2607.20475) | ~0 on greedy; host head sync verify.cpp:4228/4234 only for sampled | exact if tie-breaks match | 2026-11-16: CODED STRATA_HEAD_ARGMAX_FUSED (HIP build path fixed, -DSTRATA_HIP_HEAD_FMT=ON) |
| Item 19 VERIFY_DEFER_CPU | +15.8% +-3.9 edit, short n.s.; DEFER_AUTO -1.2..-12.5% | LOSSY | yes |
| Item 20 Mirror-SD (2.8-5.8x, +30% vs EAGLE3) / PipeSpec (2.25 vs 1.5) / AMUSD (29%) / Speculative Aggregation | papers, datacenter GPUs, need 2nd draft model | claimed exact | 2026-11-15: park: second accelerator |
| Consumer-HW spec-decode caution (Apple silicon: best 1.61x, 3 of 5 configs slower) arXiv:2607.17283 | Apple M-series | exact (chi2 p=0.976) | n/a |
| llama.cpp #23126: cross-model waits on shared Radeon iGPU drop draft 33-35 -> 3-5 tok/s (Vulkan) | Radeon iGPU | n/a | n/a |
| Combine kernel is >=96% wait (5.6 us floor/call x 4032 = 22.6 ms = 0.6% wall): 18.9% is a pool-latency ceiling, not an arithmetic one | derived from gpu_sync note | n/a | n/a |
| Untested-switch inventory: 153 window-keyword `STRATA_ENV` (of 1017) - Speed 84, KeyIfSet 22, Key 12, Unsure 17, Inert 17, InConfig 1; ~20 have an A/B | - | - | conversation_disk_env.def |
| Still-coded-only, no A/B: ROUTE_PRIOR_AUTO, ROUTE_PRIOR_DEPTH, ROUTE_PRIOR_PROTECT/KEEP/SCOPE/MARKS, ROUTE_RESIDENT/_MTP/_RANKS, COPY_HIP_PRIO, GRAPH_PRIO, FUSE_HITS, GR_FUSED (non-finite suspect), FUSE_CONVAB, PREFILL_ROUTED_STREAM, SPEC_COST_TABLE, SPEC_PROB/_COUPLED (exact sampled spec), SPEC_CUM, KV_RING_OFF, ADAPT_NOWAIT/ASYNC_WAIT, MTP_* batch variants | - | mixed | all yes |
| Engine does not print its switch state; add startup dump of effective STRATA_* | - | n/a | 2026-11-15: CODED STRATA_ENV_DUMP |
| Label mismatch: PIPED_MTP and OLD_SAMPLER are class Speed but change tokens; fixing needs conversation-disk key version bump (conversation_disk_tier.cpp:242-251) - owner decision | - | n/a | yes |
| Config conflict: notes said DEFER_CPU in fast-hip preset; file has none | - | - | - |
| hip_flag_order_test failing (fix proposed, not applied) | GPU suite 133/134 | - | test yes |
| Stale MOD_PLAN status lines found by inventory: 233 (EXPDEC_DOWN measured +1.7% +-0.5), 1753 (DEFER_CPU_FOLD measured flat), 1759-1760 (WAIT_COPY256/ROUTE_FAST measured), 2623 (DEFER_AUTO measured worse) | - | - | - |

### Closed questions (window overhead)
- Per-window overhead on gfx1100 is host-pool latency the GPU waits on, not launch cost; fusion/copy/device-wait items each remove at most a few tenths of a percent (report.md:60).
- COMBINE_POLL1, WAIT_FUSE-off, FUSE_WAIT, MTP_EAGER, spec3, VERIFY_DEVICE_PLAN, ROUTE_FAST: all measured and dropped (MOD_PLAN 3284, 3291-3294, 3683-3693; TASKS:133). WAIT_FUSE/FUSE_WAIT arms also show a placement-dependent slow mode (78-91 tok/s).
- WAIT_COPY256 cannot act at `--spec 4` (needs T>=6); any measured gain there was noise.
- `--pool-affinity measure`/`auto` do nothing on the 5700G (non-hybrid).
- PIPED_MTP is off on the single-GPU decode path (warning at generate.cpp:7406).
- Deferral (VERIFY_DEFER_CPU/AUTO, rank-guard TOP/FOLD) is lossy and flat-to-negative; OFF in daily presets, do not re-run.
- STRATA_POOL_UMWAIT on Linux desktop: closed (MOD_PLAN 1228). MTP_HIP_PRIO: dropped for good.
- hipStreamWaitValue32 is supported on the 7900 XT (attribute = 1); only its benefit is unmeasured (ceiling 0.3%).
- No paper measures this overhead on a single RDNA card; all literature numbers are abstract/excerpt only.
- Fused sampler (SonicSampler) is near-zero on the greedy path.

## Part G: placement-statistics leads (web research 2026-11-16, for MOD_PLAN.md section 6 items 19 and 26)

Read as search-result summaries only (abstract / page level, not the papers); every number is the authors' hardware, label `[paper]`, details `[unverified]`. Simulator policies named here are in `tools/pancake/tier_laws.py` (MOD_PLAN.md item 19) when marked so.

| lead | number + hw | exact/lossy | in our src |
|---|---|---|---|
| LeCaR (Vietri et al., HotStorage 2018, usenix.org/conference/hotstorage18/presentation/vietri): cache replacement as online learning over two rules (LRU, LFU); weights follow regret, a miss on a recently evicted page is the regret signal; beats ARC with two base rules `[paper]` | no MoE number | exact | not in the engine; simulator policy `lecar` (the owner's upgrade commit, MOD_PLAN.md item 19) |
| OLeCaR (arXiv 2009.11330): LeCaR with the learning rate set from a regret bound, vanishing regret over time `[paper]` | no MoE number | exact | simulator: the learning rate of `lecar` is a parameter of the evolutionary search |
| CACHEUS (FAST 2021): LeCaR-style adaptive mix of scan-resistant LRU (SR-LRU) and churn-resistant LFU (CR-LFU), no static hyper-parameters, best on 30 workload x size cells `[paper]` | no MoE number | exact | TESTED OFFLINE 2026-11-16 (TEST_LEDGER.md T-669): `cacheus` (SR-LRU + CR-LFU, regret weights, adaptive rate) at cap 32 is about 1 hit point and 0.9 ms worse than kmiss 2/8 on the offline trace: DROPPED |
| ARC (Megiddo and Modha, FAST 2003): recency and frequency lists with ghost lists; a ghost hit moves the split `[paper]` (background) | no MoE number | exact | simulator policy `ghost` (adaptive protect fraction) |
| MoE-CORE (arXiv 2610.01950): domain-informed cache initialization, routing-history-aware replacement, cross-layer prefetch, NONUNIFORM cache capacity per layer; optional score-based substitution for low-score misses; prefill stages whole layers in alternating buffers `[paper]` | TPOT 38.0-44.8 ms vs 1268.9 ms (vLLM prefetch), DeepSeek-V4-Flash W4A8, an 84 GB NPU appliance; no hit rate on the page | exact main path; substitution lossy | `STRATA_CONV_WORKSET`, `DOMAIN_PROFILES`, `STRATA_SUBFIT_*` cover the pieces; per-layer capacity: ours comes from the profile split, untested against a fitted one (simulator: layer budgets as a search parameter) |
| DAOP (arXiv 2501.10375): data-aware offloading with predictive pre-calculation `[paper]` | not read | exact | not compared; read before porting `[unverified]` |
| Two-Stage Expert Offloading for Domain-Aware MoE Inference (ResearchGate 400888358): domain-aware prefill, locality-aware prediction and preemptive load of upcoming experts during decode `[paper]` | not read | exact | `STRATA_CONV_WORKSET_MATCH`, item 26 V4 (lead time) tests the same claim on our traces; its two-stage domain policy was SKIPPED offline 2026-11-16 (TEST_LEDGER.md T-669): the repo trace has no domain labels, wait for the chain 16a trace |
| vLLM RFC #38256 "Incremental MoE expert offloading": fixed-size GPU expert cache, pluggable eviction policy, conversation-level domain shifts `[paper]` (issue text) | reports 97-100% hit rate with temporal prediction, conditions not given | exact | none; compare only as a design |
| Cache-Aware Joint Router Adaptation (arXiv 2609.04895): adapts the model with light auxiliary cache routers, native top-K at inference `[paper]` | not read | model change | not applicable without training: parked (model-side) |

## Part H: fork scan of 2026-11-16 (tools/forks/fork_kb.py run 7: 478 forks, 959 commits dated 2026-11-12 or later, 359 triaged, every diff read through the GitHub API and compared with our tree)

Labels: `[code]` = the diff was read and our tree grepped (file:line given where it matters); author numbers `[measured]` on the AUTHOR's hardware, never ours. Counts by sha, best class: batch A (180): 21 new-applies, 16 have, 2 listed, 55 parked, 86 noise; batch B (179): 34 new-applies, 6 have, 26 listed, 55 parked, 58 noise (several new-applies entries are bundles of up to 8 shas). The raw per-commit results are not archived (scratchpad JSON); the commit sha lists below are the evidence.

| lead | number + hw | exact/lossy | in our src |
|---|---|---|---|
| adambenhassen `ef156accf1` `61f3e6176f`: `STRATA_STOP_LAYERS` / `STRATA_STOP_STAGES`, a Stop is checked between layers inside a prompt chunk `[code]` | next request after a Stop 17.9 s -> 1.1 s, 2x P100 `[measured]` | exact | 2026-11-16: CODED `STRATA_STOP_LAYERS` (default off, MOD_PLAN.md item 27, T-695; `STRATA_STOP_STAGES` needs two GPUs, parked). Was: the stop was polled only at the chunk start (src/prefill/prefill.cpp:3937 and :8563) and our chunks reach 32768 tokens; the layer loop is ~4818 |
| ElXreno `2c1219eb` `6e4126cd` `e8c82e8f`: a TRAINED per-layer look-ahead predictor replaces the router in `RouterLookahead` (+ `STRATA_PRED_DUMP` / `FEAT_DUMP` for the fit) `[code]` | no number in the commits | exact | CODED 2026-11-16, default off, engine NOT built or run (MOD_PLAN.md section 6 items 29 and 39; `STRATA_LOOKAHEAD_WEIGHTS` / `_B` / `_TAU`, `STRATA_PRED_DUMP`, `tools/pancake/fit_lookahead.py`; T-704; the data run and the fit are queued, T-710: NOT MEASURED). The fork's diffs were read through the GitHub API as untrusted data; the code is written here. Original: LEADS B13 and MOD_PLAN closed the predictor on 19-21% recall of the router-based one; a trained one was never tried: src/core/expert_source.cpp RouterLookahead::start/run |
| ElXreno `4373596a`: Claude Code side queries (no tools, one user message) go out with `strata_checkpoint=false`: they neither park nor save a root `[code]` | no number | exact | 2026-11-16: CODED `STRATA_SIDE_QUERY_ONESHOT` (default off, serve/server.py `mark_side_query`, T-697) |
| Indigoshores `12d7cde5` `0c398d60`: small-T MoE prefill tiles (BM 16 / 32 / 64 by the average rows per expert, `STRATA_MOE_MAX_BM`, `STRATA_PF_F16_VAR_BM`, `STRATA_PF_BK64`) `[code]` | 38.0 -> 41.3 TFLOPS bit-identical; 1490 t/s prefill bundle, gfx1151 APU `[measured]` | exact | 2026-11-16: CODED `STRATA_PF_F16_VAR_BM` / `STRATA_MOE_MAX_BM` (half tiles of 32 rows in the F16 expert kernels, default off, T-659, spec 118; BM 16: 2026-11-17 CODED too, `STRATA_MOE_MAX_BM=16` / `STRATA_PF_F16_VAR_BM16` (the activation fetch of BN 16 is a partial-thread slot, a small change), GPU code never built, host rule T-737, spec 146, MOD_PLAN.md item 45); `STRATA_PF_BK64` is already ours (T-661). Was: NOT in ours (no moe_tile_buckets.cuh): src/prefill/moe_fused_iq.cu, wmma_gemm.cu:401, prefill.cpp; the target is the 100-1000 token agent delta |
| bsorensen110 `3d4a130dfa` `c55e3de9db` `c6442c50d4`: interleaved Q8_1_IL / MMVQ_IL extended to 9..16 columns (CP = 16, `STRATA_IL_WIDE`) `[code]` | width bench +11.4% solo, -7.6% burst (8 -> 16), R9700 gfx1201 with DFlash2 `[measured]` | exact | 2026-11-16: SKIPPED (T-660): the interleaved MMVQ is CUDA-only in our tree, HIP returns `native_mmvq_il_supported` false, so there is no gfx1100 path to widen. Was: ours stops at 8 columns (include/strata/kernels/native_mmvq.hpp:194 `native_q8_1_il_cp`) and the preset runs `--lookup-max-t 12` |
| dardeaw `b1212e15` `9a2b81da` `ffb6110a` `a845d57f`: fp16-KV WMMA prompt attention for gfx11 (`STRATA_PROMPT_ATTN_F16`) `[code]` | prefill attention 1584 -> 485 ms, 1010 -> 1237 tok/s, gfx1151 APU `[measured]` | near-exact | fires on an fp16 KV pool only: our daily preset is `--kv int8` (src/kernels/cuda/qsa_prompt_attn.cu), so it needs an fp16-KV arm; 2026-11-16 checked: the env does not exist in our tree and the kernel is not small, 2026-11-17: premise re-read, NOT coded (MOD_PLAN.md item 49, T-747): our kernel is NVIDIA mma.sync, the fork's is a new gfx11 kernel; spec 150 (`--kv fp16` vs int8, no kernel needed) queued |
| xszwow `e0f97dbf7e`: `STRATA_PF_DM16` `[code]` | none | exact | HAVE (arch_defaults.cpp:189, moe_fused_iq.cu:2125); spec 129 written 2026-11-16 (T-698 NOT MEASURED, queued) |
| adambenhassen `3faf5f04a0` (upstream #5952 `tool_addition` blocks) and `54ef684d05` (an undeclared tool call counts as a call when ToolSearch is declared) `[code]` | none | exact | 2026-11-16: CODED `STRATA_TOOL_ADDITION` and `STRATA_TOOLSEARCH_PASSTHROUGH` (default off, serve/frontend.py, T-697); our `defer_loading` / `tool_reference` handling (#4663) already existed |
| dongnh311 `b745d6b14f` = upstream v0.2.14.1 / engine 0.1.60: prompt experts on Strata's own CUDA / HIP kernels instead of llama.cpp MMQ, equal-piece long prompts `[code]` | 244K prompt 825.6 -> 906.7 tok/s (+9.8%), decode flat, a Windows box, GPU not named `[measured]` | exact | a REBASE decision: ours is engine 0.1.58 (CMakeLists.txt:11) and LEADS has no row for 0.2.14.1; our hipBLASLt table prompt path (1980 tok/s at 64K) may not gain |
| bsorensen110 `76d3e83199` `0dd018de2c` `db68d534ec` `eb7eda0a4c` `fe5de2afc7` `71dfb869f8` `d5823b2b13` `0b8b0d91f3`: verify-window launch-fusion bundle (`STRATA_MMVQ_SEGS`, non-finite check fused into argmax_rows) `[code]` | +3.7% solo / +3.1% burst / +1.6% prefill, inside the author's own A/A floor of +-4.0%, R9700 `[measured]` | exact | ours screened launch fusions as noise (T-604, T-612, T-614, T-616) |
| bsorensen110 `90926ab939` (`STRATA_TOPK_SAMPLE` on HIP), `931ede52b5` (`STRATA_ATTN_V4`) `[code]` | the author's A/B: not separable from noise | exact | screen only |
| serboox `d1eea56e6b`: two-stage verify head with a norm bound (`STRATA_HEAD_BOUND_PRUNE`) `[code]` | the author never built or ran it | exact in principle | speculative: measure offline how often the bound holds first |
| Indigoshores `7a02e537`: gfx1151-tuned decode kernel variants for the same 512 / 10 model (`STRATA_DOWN_SUB`, `EXP8_SUB16`, `S27_DOWN`, `GR_DOWN_TUNED`, `HC_NT`, `HC_DEC_QUANT` lossy) `[code]` | "80+ t/s prose" with their drafter, no per-switch number | exact except HC_DEC_QUANT | low confidence: a 13k-line dump; ours screened kernel switches as noise (T-604, T-612) |
| pszeder `b2d00855`: bit-exact one-token IQ2_XS / XXS / IQ1_M AVX2 rows (`STRATA_CPU_IQ_OWN1`) `[code]` | 31.2 -> 21.2 ns/block on one thread, about +2% when DRAM-bound, Broadwell `[measured]` | exact | we have the same idea as `STRATA_CPU_IQ_ALL_EXACT1` (native_expert.cpp:610, iq_avx2_rows.inl:152): 2026-11-17: CODED `STRATA_CPU_IQ_OWN1` (own code; IQ2_XXS / IQ2_XS gate/up + down at one token; IQ1_M not coded), parity T-730, host timing slower than the exact rows, spec 143 (MOD_PLAN.md item 43) |
| Spark-NV `1b814f92`: PCIe probe repeated while the link reads under 20 GB/s (`STRATA_PCIE_PROBE_ROUNDS`) `[code]` | 123 vs 134 tok/s on one third of starts, RTX 5090 PCIe 5 `[measured]` | exact | low: our preset runs `--pcie-frac 0` and never probes |
| BUG FIXES, exact, no speed claim `[code]`: ElXreno `3da39ce1` (disk-only save skips short conversations: ours generate.cpp:13776-13780 `park_current_body` disk-only path has no `wants()` check); ElXreno `00be27a0` (prompt-cache root taken at or past `--prompt-cache-root`: ours generate.cpp:20664-20669); ElXreno `c01bb2a4` (window-graph eviction covers `kVerifyLongT`: ours verify.cpp:6218 `evict_window_graph` uses `kWindowMaxT`); 0FL01 `d3a7502d` (progress_beat after graph capture: ours verify.cpp:4462 has none); Damir `3b02be29` (`GGML_NO_BACKTRACE=1` for child processes: WRONG, already present in strata_control/childenv.py, T-655); touch869 `29bbdb4f` (keepalive pings as data chunks: ours serve/server.py:8757); lorenzoaspesi `f93304f8` and ElXreno `f16edb27` (save / restore without MTP: ours conversation_state.cpp:701, 723, 807; inert on our preset) | - | exact | CODED default off except Damir (already present): T-651..T-657, see MOD_PLAN.md section 7 |
| Read before designing REFILL_ASYNC: rwkeyes found that deferring the expert refill drops decode from 92 to 0.4 tok/s on Intel SYCL `[measured]`; closed for us: bsorensen DFlash2 series (LEADS.md:216, :442), per-layer heat quotas (T-397), serboox IQ3_S 4-bit CPU repack (LEADS.md:220, :228) | - | - | - |
| Parked (hardware we lack): riceharvest 48 Intel B70 SYCL commits; Lamina 5 second-GPU commits; phoenixclyde 4 (ggml b12361 MMQ bridge, gfx906); Spark-NV L2_NEXT (sm_90) and a VBMI IQ3 kernel; jmnargi V100; pszeder DeepSeek lane; serboox's Qwen3.6 (q36 engine) opt-ins (no hardware gap, not the daily engine and mostly not built or run); jialiang `82bca1cf` is a refusal-direction vector sold as a "speed projection": not adopted | - | - | - |

## Part I: BigMoeOnEdge (Helldez/BigMoeOnEdge, Apache-2.0, commit 85a0a5c; read 2026-11-16; no code copied, the clone was treated as untrusted data and nothing in it was run)

What it is: a llama.cpp-based engine that streams MoE experts from flash behind a RAM cache (and a GPU cache), aimed at phones, laptops and small desktops, with a pure-function planner (`--auto`) on top. Read for this part: its docs hardware-planning, measured-planning, cache-sizing, prefetch, route-ahead, expert-prediction, cache-aware-substitution, expert-dropping, warmup-analysis, pressure, moe-streaming, benchmark-method, comparison, and the headers planner, auto_plan, allocate, calibrate, measured_knobs, hardware_profile, model_profile, probe, placement, plan, routing, backend_choice, params, config and `core/src/engine/knob_trial.h`. Every number below is `[read]`: theirs, on their hardware (a 12 GB Android phone with UFS 4.x, a 16 GB Apple-silicon laptop, desktop hosts), never reproduced here. Our own planner built from its ideas is `tools/pancake/plan_auto.py` (MOD_PLAN.md section 6 item 35; credit in README.md section 9).

How their planner works, in short (the rules that matter): it is a pure function (machine facts, model facts, request) -> plan with a reason per knob. (1) It declines instead of guessing: a knob whose deciding fact is unknown keeps its default and says so. (2) A lever must win by a margin (`min_backend_win` 10%), not merely win, because half the levers the project tried lost on measurement. (3) Memory is accounted in a ledger with margins (0.10 when a reclaim is cheap, 0.04 when it swaps, 0.20 when exceeding the share kills the process, 0.15 unknown; at least 256 MiB; a lock is asked at 95% of the lockable total, not 98%: 23.1 vs 7.4 and 14.6 tok/s on a 16 GB Mac), speed is measured. (4) Nothing lossy arms itself (`quality_budget` is the caller's). (5) The cache floor is the model's own token cycle (layers x top-k x expert bytes): under it an LRU cache returns exactly 0% hits (gpt-oss-120b top-4: a 1500 MiB budget under a 1815 MiB cycle, 0.0% hit, the same 1817 MiB read per token as no cache). (6) Candidates are priced in one unit, seconds per byte, so flash, RAM and device memory compete in one ranking; a missing measurement makes a candidate ineligible, never cheap. (7) The lossless knobs that can change with the session open (threads, overlap) are settled by an interleaved A/B on the user's first answer (`KnobTrial`: windows of 8 tokens, incumbent-candidate-incumbent so a linear drift cancels, 3 rounds, median of the per-round ratios, a win needs 10%, a candidate is dropped once more than half its rounds lost or one window ran apart from the incumbent's whole spread, the trial waits until two incumbent windows agree on hit rate within 0.02) and the verdict is kept per hardware fingerprint, engine version and model. (8) Decisions that cannot be tried inside one load (device cache or the capacity fitter's layers, graph width) are timed across loads (`calibrate.h`). (9) No runtime loop resizes memory against the kernel: their cache governor was retired after it turned a 0.517 tok/s cache-off run into 0.156 (a reclaim war the loop fed).

Where their prediction and prefetch designs differ from ours, plainly:
- Same predictor, different yardstick. Both apply layer l+1's router matrix to layer l's input (FATE, arXiv 2502.12224; ours `RouterLookahead`, src/core/expert_source.cpp:3260-3430, with a GPU-side table by default, generate.cpp:12722). They report 88.6% (Qwen3-30B, top-8 of 128) and 80.7% (Qwen3.6-35B, top-8 of 256) slot overlap, with a control (layer l's own matrix on its own input) that reads 100.0%. Ours measured 19-29% recall (B13, MOD_PLAN.md item 29) against the experts the next layer routes OUTSIDE the GPU cache, on 512 experts with top-10: another denominator and another model, and we never ran a control. The 3-4x gap is therefore unexplained, not a finding: I2 tests it first.
- Different use, because the tier is different. A miss in their engine is a flash read that competes for flash bandwidth, so speculative reads lost every matched pair (previous-token prefetch 2x slower on gpt-oss; stale-gate top-2 read-ahead -21%: hit rate up 4.3 points, 79% of reads useful, +37% bytes, wall clock behind) and their one win is the lossy route-ahead (+17% / +22% on the host under `--overlap`, +21% on the phone, the stall 0.060 -> 0.010 s/token). A miss in ours is a CPU computation from the RAM arena (80.6 us per expert) or a PCIe copy (114.6 us; both from the engine's own probe, T-640), so prediction only warms pages (file/mmap tier) or feeds the Foresight swap copy. Our measured results: Foresight swap with 4 slots 0.97x (T-190, 3060), GPU lookahead on the mmap tier +10% (T-192, 3060), and STRATA_PF_AHEAD for staged prompt chunks is a different design again (src/prefill/prefill.cpp:483-502). Their two findings agree with ours: acting on a better prediction improves its own metrics and not the wall clock, and popularity beats recency (their static hottest list 37.6 / 39.6 / 26.7% vs previous-token 38.0 / 35.7 / 17.9%; ours static 71.87% vs tier 81.17% hit, T-625).
- Different regime for the tier laws. Their eviction finding (decayed frequency with a half-life of 8 tokens beats LRU below one cycle by 25-36% of bytes, every admission filter on decode loses +2-15%, random and layer-order eviction lose) is the family of our `tier` rule (decay 0.7 per window) and of what tier_laws.py replays (T-618..T-625, T-669: kmiss and LeCaR/CACHEUS add nothing at the engine's cap of 32 swaps). They price the cache on recorded routing with the engine's own arithmetic (their gate G2r holds replay and engine to the same hit rate); ours does the same through cache_learn.py / tier_laws.py (the hit rate reproduces to the row, T-684).
- What they have that we do not: the control, the in-answer interleaved trial with a kept verdict (ours is calibrate.py at setup and the offline abtest), and the margin rule and the memory ledger as code (now `plan_auto.py`, offline only).

| lead | number + hw | exact/lossy | in our src, and the default-off switch we would add (rule 8) |
|---|---|---|---|
| I1 Temporal prefetch `--prefetch K`: speculative reads of the experts the PREVIOUS token routed at layers l+1..l+K on idle lanes, cancelled when a real read appears, identical bytes `[read]` | previous-token top-k predicts the next token's 38.0% Qwen / 35.7% Gemma / 17.9% gpt-oss; a static hottest list 37.6 / 39.6 / 26.7%; gpt-oss `--prefetch 1` 828-895 ms/token vs 414 baseline (2x slower, no hit gain); phone route traces | exact | CODED 2026-11-16, default off, engine NOT built or run: `STRATA_PREFETCH_PREV=K` (MOD_PLAN.md section 6 item 39, `include/strata/program/prefetch_prev.hpp`, ctest-style host test T-703, spec 133 queued T-707). The look-ahead thread exists only with a warming source (the arena warms under `STRATA_POOL_PREREAD=1`), so the arm is judged against `preread`. Original: NOT in ours as a read-ahead; the adaptive tier already carries the popularity signal (decayed counts, tier_laws `tier`). Add `STRATA_PREFETCH_PREV=K` (default off): at layer l's submit (expert_source.cpp:3300) warm what the previous window routed at l+1..l+K on the CPU lane. By their numbers and T-190 an expected loss: test anyway (rule 8) |
| I2 Predictor CONTROL and per-layer table (`--predict-log`): stale-gate vs previous-token vs fresh-gate control (must read 100%; below that the probe is wrong) `[read]` | stale-gate 88.6% / 80.7%, control 100.0% on every layer of both models, previous-token 43.3 / 35.4%, first layers worst; 12 GB phone | exact (measurement only) | CODED 2026-11-16, measurement only, engine NOT built or run: `STRATA_LOOKAHEAD_CTRL=1` (MOD_PLAN.md section 6 item 39, `include/strata/program/lookahead_ctrl.hpp`, `tools/pancake/lookahead_ctrl.py`; T-700 host tests, the control run itself T-711 queued: NOT MEASURED). Prints per layer the all-slots overlap of the GPU table and of the host router, the control and the previous token. Original: NOT in ours: `STRATA_LOOKAHEAD_STATS=1` (generate.cpp:9949) scores top k / k+4 / k+10 against next-layer misses, with no control and no per-layer table. Add `STRATA_LOOKAHEAD_CTRL=1` (default off, measurement only): also score layer l's own router on its own input and print per layer. If the control is under 100% our 19-29% recall is a bug, not a property. First test of this part. The GPU lookahead (+10%, T-192) was adopted on a 3060 and has no isolated row on the 7900 XT (LOOKAHEAD=0 appears only inside the T-474 stack) |
| I3 Route-ahead `--route-ahead N`: LOSSY, the expert selection of layer L is replaced by the ranking layer L's gate gives on the hidden state N layers earlier, so the routing is known N layers of compute early and a read or copy for it can never be wrong; the gate still runs so the true renormalised weights apply; deterministic; a watchdog disarms it; layers 0..N-1, prefill and the first decode token pass through `[read]` | committed selection agrees with the router on 84.7% of slots at N=1 (128 experts); host +17% / +22% (second model) / +40% with headroom, N=2 recommended under overlap (stall 0.060 -> 0.010 s/token, 97% of early reads useful); phone +21% (5.29 vs 4.36 tok/s, interleaved cells); quality 4/4 on a four-prompt battery, an essay and a second model: no loss detected, no task benchmark | lossy | NOT in ours (no `ROUTE_AHEAD` in src, include or serve; `route_ahead_kernel` in src/dsv41/ds4_kernels.cu:1677 is DeepSeek's own routing). Closest: `STRATA_ROUTE_PRIOR` biases toward residents, not toward a prediction (verify.cpp:646). Add `STRATA_ROUTE_AHEAD=N` (default off, lossy). Hypothesis for our tier, not tested: knowing layer L+N's routing early lets the CPU pool start the missed rows of L+N while the GPU works on L, which is the spin (combine_cpu 25.8% + wait_flag 14.8% of decode kernel time, MOD_PLAN.md item 22). Needs token forcing and the task gate |
| I4 Cache-aware substitution `--expert-substitute L`: before the top-k is committed every RESIDENT expert's score is raised by L x (this token's score range), the top-k is taken again, the router's own weights are kept; the margin is relative to the token's own range (max - min of the scores the graph sorted), so one L means the same on any router `[read]` | L=0.15: decode 2.37 -> 3.84 tok/s (+62%), 258 -> 119 MiB/token, hit 48.8 -> 69.4%, 27% of slots substituted, perplexity +1-4%, tinyMMLU 88 -> 84 of 100, HumanEval run; L=0.30 +25% perplexity, L=0.60 perplexity 31; Qwen3.6-35B-A3B on a desktop host, cache 2000 MiB, not a thermally matched on-device A/B | lossy | CODED 2026-11-16, lossy, default off, kernel mode UNBUILT: `STRATA_ROUTE_RANGE=L` (MOD_PLAN.md section 6 item 39, `include/strata/program/route_range.hpp` + a mode of `route_prior_k`; needs `STRATA_ROUTE_PRIOR` set; T-702 host reference only, spec 134 queued T-708). Note `STRATA_ROUTE_PRIOR_DELTA=1` already scaled by the LAYER'S running mean range; this one is the token's own range. Original: HAVE the family: `STRATA_ROUTE_PRIOR` (verify.cpp:646; lambda 0.5 +25% decode on the 3060, token mismatch +1.66 pts, rate-rejected, T-294; lambda 2 dropped, T-295; retired turbo preset), `STRATA_ROUTE_RESIDENT` margin in logits over ranks 6-9 (verify.cpp:144), `STRATA_ROUTE_BUDGET` (verify.cpp:212; spec 95 screen on a noisy base, T-628), `STRATA_BUDDY` (expert_source.cpp:5757; -18.3% short, T-622), `STRATA_STANDIN` (verify.cpp:199). The one idea we lack is the range-relative margin: add `STRATA_ROUTE_RANGE=L` (default off, lossy), L x (max - min) of the token's scores instead of an absolute logit margin |
| I5 Cache-aware dropping `--drop-cold-experts F`: skip a routed expert that is NOT resident AND weighted under F x (1/top-k); a routing is never emptied (the top expert is pinned, F <= 1), needs the cache, prefill excluded `[read]` | replay F=1.0: 66% of flash reads avoided, 9.5% of the router weight discarded (`--n-expert-used 5`: 23% / 10.6%); phone Qwen3.6-35B-A3B: F 0.75 2.549 -> 3.938 tok/s, F 1.0 4.702, 28.4% of routings dropped; GSM8K 12/15 off vs 13/15 at every F (15 questions: rules out a collapse only) | lossy | HAVE the family: `STRATA_MISS_SKIP_W` / `_CUM` (expert_source.cpp:5206), `STRATA_ROUTE_TAIL_SKIP` (:5160; 7 dropped, T-298), `STRATA_SLICE_MISS` (:5339), `STRATA_SER` (:5230). Not measured on the box: T-613 aborted before its four miss-skip arms ran, so no MISS_SKIP number exists (MOD_PLAN.md item 15). Their form is relative to the uniform share (F / top-k), ours a share of the router mass: the same at top-10 when F x 0.1 = share. Nothing new to port; run spec 88 again |
| I6 Speculation must know the drop policy: an early read that makes a to-be-dropped expert resident un-drops it, so committed experts under the drop threshold are left cold `[read]` | phone, depth 4, drop 0.75: 78 -> 244 MB/token and half the tok/s without it | lossy interaction | CODED 2026-11-16, default off, engine NOT built or run: `STRATA_LOOKAHEAD_SKIP_DROPPED=1` (MOD_PLAN.md section 6 item 39, `include/strata/program/lookahead_skip.hpp`, T-701, spec 135 queued T-709). Premise checked in code: our skip rule reads GPU residency, so a file / arena warm does not un-drop (only a Foresight swap copy does); the switch saves wasted warms. Original: possible in ours: the lookahead warms what MISS_SKIP would skip. Add `STRATA_LOOKAHEAD_SKIP_DROPPED=1` (default off): leave predicted experts under the miss-skip threshold unwarmed. Only matters once I5's arms are on |
| I7 Warm-up: sequential dense sweep at load (`--dense-weights warm`; `anon` past RAM) so the non-expert pages do not fault in one 4 KiB page at a time; per-token CSV with wall, compute, stall, io, majflt, cpu_ms `[read]` | gpt-oss-120b (5.2x RAM, phone): first-5-token wall 16,641 -> 829 ms (20x), first token 18 s -> 1 s; models near RAM neutral | exact | N/A for the dense set (ours is in VRAM, loaded unbuffered at start). The only SSD-read part is the PLE shard (28.8 GB; `--ple-io direct\|mmap\|ram`, generate.cpp:1148; mmap dropped, T-274). Add `STRATA_PLE_WARM=1` (default off): one sequential read of the PLE shard into the page cache at load; risk: 28.8 GB of page cache against `MemoryMax=54G`. Judged on the first request after start (ttft_bench), not on steady decode |
| I8 Prompt misses enter the cache COLD: from 32 tokens wide a graph's misses go to the cold end, so the prompt does not flush what decode needs `[read]` | 4 documents of 3.6-4.3k tokens, 8000 MiB cache, Mac: 423 -> 211 GiB read in the prefills, prefill -17%, decode unchanged | exact | N/A as built: our prefill streams through the loan and never touches the decode tier. The same loss is the decode tier after a long prompt (92.5 vs 111.7 tok/s): `STRATA_TIER_REWARM=1` (MOD_PLAN.md item 8; spec 86 inside noise, T-608; confirm spec 103) |
| I9 Pressure: no governor (retired: 0.156 vs 0.517 tok/s), the cache pinned with `mlock` and `--cache-pin-fit` (it steps down to what the lock grants), or OFFERED to the kernel with `MADV_FREE` so reclaim drops instead of compressing or swapping, a lock asked at 95% of the lockable total, `fits_air_ratio` (a model that just fits is reclaimed under you: 0.1 tok/s resident vs 5.0 streamed) `[read]` | 16 GB Mac: pinned 18.2 vs unpinned 7.4 tok/s (the compressor held 2 GiB/s); 95% vs 98% of the lockable total 23.1 vs 7.4 / 14.6 tok/s; swap 3.3 GB in 7 min while pinned, 0.13 GB idle | exact | HAVE: no governor in ours, `STRATA_PSI_SHED` (src/platform/mem_pressure.cpp:148), `STRATA_PSI_GUARD` default on, `--ram-keep-gib`, `STRATA_PIN_RESERVE_GIB`, `pancake.service MemoryMax=54G` of 57 GB (94.7% of the box). New piece: the margin rule. `plan_auto.py` keeps their kill-class margin (20%) under a hard cap and holds 42.3 of 43.2 GiB; whether 54G is the right cap has no row |
| I10 Measured knob trial on the first answer (`KnobTrial`, `RunConfig::trial`): the lossless knobs that change live (threads, overlap) alternate A B A B in windows of the user's own answer; the verdict is kept per hardware fingerprint + engine + model; a load-time calibration times the decisions that cannot switch live `[read]` | window 8 tokens, 3 rounds, margin 10%, early drop of losers; one phone: 8 threads vs 4 ran 20x slower (the trial cut it after one window); threads from core classes was shipped and withdrawn in a day (4.24 -> 2.68 tok/s); a day of A/B lost to throttling (6.54 -> 3.93 tok/s with byte-identical I/O) | exact | RULE CODED 2026-11-16, NOT WIRED: `include/strata/program/knob_trial.hpp` (decision rule + log line, T-705; MOD_PLAN.md section 6 item 39); no `STRATA_KNOB_TRIAL` variable exists because generate.cpp has no per-window live-knob switch and the engine cannot be built here. Original: HAVE in part: `tools/calibrate.py` (setup; per-request `strata_tune` keys `pcie_frac`, `spec_min_p`, `aux_cpus`, `prefill_pipe`, serve/server.py:1956; MIN_GAIN 3%, interleaved re-measurement; `--pool-workers`, the tier and `STRATA_SH_STREAM` need a restart), `STRATA_CALIBRATE` / `calibrate_ratio.hpp` (T-640: it picked `--pcie-frac` 0.40 and ran 17-30% below the daily 0, exactly the error a trial on the real window avoids), `--pool-affinity measure` (generate.cpp:8712), `STRATA_CALIBRATE_CACHE` (:8758, keeps probe figures per GPU, pool and pack). NOT in ours: the in-answer interleave and a kept knob verdict. CODED 2026-11-17: `STRATA_KNOB_TRIAL=1` (default off, exact) over the per-request keys, run between requests in `serve/server.py` (`serve/knob_trial.py`, T-742), spec 148 queued (T-743); the box-side form is the abtest spec of MOD_PLAN.md section 6 item 36; nothing measured |
| I11 `--auto` plan as a value: a rationale per knob citing the fact, its source (measured / probed / default) and the number; a prediction printed against the run (0.037 predicted vs 0.107 s/token measured on a phone) and the next plan scaled by the ratio `[read]` | nothing to quote | exact | CODED offline 2026-11-16: `tools/pancake/plan_auto.py` (MOD_PLAN.md section 6 item 35, T-685). In-engine: `--expert-cache auto` (auto_cache.hpp:33, generate.cpp:8825), `--prefill auto`, `STRATA_VRAM_CALIB` (generate.cpp:3742) are the narrow versions. No engine switch added: the tool is the form. Not coded: the predicted-vs-measured printout and the next-plan scaling (we have no plan prediction of speed) |
| I12 Cost in seconds per byte, ONE ranking for residency groups (dense set, cache, device layers); a non-host device must win by a margin; the plan that streams nothing stays reachable `[read]` | no measured gain quoted | exact | planner logic, not an engine switch; used as rules P1-P5 in `plan_auto.py`. Our groups do not compete (dense and KV in VRAM, experts split by the tier, the RAM arena holds the rest), so the ranking collapses to the VRAM tier size, the loan and the RAM ledger |
| I13 Token-cycle floor: a cache under layers x top-k x expert bytes has no use (LRU: exactly 0% hit; decayed frequency: a smooth curve), priced from the model's shape alone and warned at load `[read]` | LRU 0.0% at 0.5-0.9 cycles, decayed frequency -25..-36% bytes vs LRU there | exact | TESTED OFFLINE 2026-11-16 (T-684): our `tier` rule has no cliff, hit 81.17 / 43.27 / 31.72 / 23.80 / 16.75 / 10.40% at 10909 / 2400 / 960 / 480 (= one token cycle: 48 x 10 x 1.37 MiB) / 240 / 120 slots. `plan_auto.py` rule R1 still refuses a tier under one cycle. No engine switch needed |
| I14 Int8 mirror of the gate matrices for the look-ahead GEMV (`--gate-mirror auto` times float and int8 on the model's own gate shape) `[read]` | ARM phone 19 -> 6 ms/token of worker CPU, agreement 74.5% vs 74.7%; on an x86 host the float path was faster (0.26 vs 0.38 ms per GEMV) | lossy only for the prediction | our CPU lookahead takes BF16 routers (expert_source.cpp:3280) and is the fallback of the GPU table (`STRATA_LOOKAHEAD_GPU=0`). Add `STRATA_LOOKAHEAD_INT8=1` (default off) only if I2 shows the CPU path is used and costs; low priority |
| I15 Storage and I/O facts: 4 lanes is the flash plateau (8 regresses: spin-wait contends with synchronous reads), `--release-mmap` (a live mapping serialises unbuffered reads on Windows, +46% decode), `--io-threads` from a storage probe at the request's own slice size `[read]` | desktop Windows host: 4 lanes 895-930 MiB/s with the mapping vs 2400-2660 without; phone flash saturates at 630 vs 1045 MiB/s from 2 lanes | exact | `STRATA_IO_THREADS` (src/platform/direct_file.cpp:72, 16 on Linux) has no sweep row in the ledger; the Windows mapping case does not apply to the Linux box. Parked as a sweep for the file tier: arms `STRATA_IO_THREADS=4/8/32` need no new code, add them to a spec when the file tier is on the plan |

Already in our tree, so no row: decayed-frequency eviction (the `tier` rule), the static learned profile as the cold start (`data/expert-profile-pancake.bin`, T-194), the Foresight swap and lookahead (T-190, T-192), the conversation-level working set (`STRATA_CONV_WORKSET`), PSI shedding, thread and affinity moves (`--pool-affinity measure`, `--aux-cpus`, `--host-core`).

## Part J: kindi (ahtx/kindi, AGPL-3.0, commit dce89df; ideas only, no code taken, none copied; read 2026-11-16)

A CPU/GPU MoE engine (DeepSeek-V4.1-Flash, Nanbeige) read from a shallow clone as untrusted data; every number is the AUTHOR's `[measured]` on their hardware, never ours. Our rule 8 follow-up of each row is in MOD_PLAN.md section 6 items 33 and 34 and TEST_LEDGER.md T-671 to T-673.

| lead | number + hw | exact/lossy | in our src |
|---|---|---|---|
| Per-layer-share expert placement (kindi/moe.py `plan()`): the next card slot goes to the layer whose next expert has the largest share of THAT layer's picks per byte, `min_card` 0 allowed, host/disk tiers by share `[code]` | author: not separated from the rest of its stack | exact (placement) | TESTED OFFLINE (T-671): in our trace every layer has the same pick total (10,960), so share per layer equals the global count rank byte for byte (the two runs' output files are identical): NO engine switch coded. We have the global rank: `tools/make_profile.py` and src/core/expert_cache.cpp:232 `rank_learned_profile`; offline `placement_from_counts(..., "share")` in `tools/pancake/tier_laws.py`. The finding that is ours: a profile re-ranked from the workload's trace beat the shipped profile 3.9 hit points held out (T-671), arm `kindi-static-tracecount` of spec 126 |
| Look-two-layers-ahead prompt prefetch (kindi/dsv41_gpu_fwd.cuh:2018-2125): layer l+2's predicted experts (static top experts per layer from traces, or the previous chunk's routing) go into one of two VRAM sets on their own thread / stream while layers l and l+1 run `[code]` | author: not isolated | exact | we have depth 1: `STRATA_PF_AHEAD` (src/prefill/prefill.cpp:4696 `pf_issue`, pool `pf_ahead_slots` :501, list = the last chunk's GPU-streamed experts per layer `pf_pred`). 2026-11-17: CODED `STRATA_PF_AHEAD_DEPTH=2` (MOD_PLAN.md item 47: the same pool in two sets, per-set slot map and tag, absolute slots; host rule T-740, spec 147; prefill.cpp never built) |
| Shortlist output head: cluster the vocabulary, score the cluster centres, exact-score the best ~13% of clusters `[code]` | author: about 15% faster decode, 99.4% token agreement on 310 steps | LOSSY | offline half CODED: `tools/pancake/head_cluster.py` (k-means++ on the head rows, SHL1 table, `eval` agreement), 4 tests on synthetic data (T-672). ENGINE SIDE CODED 2026-11-16, unbuilt (MOD_PLAN.md section 6 item 38, T-691..T-694): `STRATA_HEAD_SHORTLIST` + `_CLUSTERS_SCORED` (default off, lossy), host rule `include/strata/program/head_shortlist.hpp` + ctest, REFERENCE head in verify.cpp (mask, no speed), `src/kernels/cuda/head_shortlist.cu`, spec 131 quality gate, `quality_gate.py --near-tie`; ceiling by arithmetic 1.9-4.2% of a window (kindi's 256 of 1024: under 3%); the gathered speed kernel is NOT written. Was: engine side NOT built: our head is `lm_head` (src/core/layer.cpp:1410) with the fused greedy pick `STRATA_HEAD_ARGMAX_FUSED` (src/core/verify.cpp:670) and `STRATA_HEAD_WIDE`; a shortlist needs a kernel that gathers a cluster's rows (head_fmt.cu), `STRATA_HEAD_SHORTLIST` (not coded) and the quality gate. Related exact lead: serboox `STRATA_HEAD_BOUND_PRUNE` (Part H) |
| Expert-union draft-length cost (kindi/dsv41.py:572-600): cost(k) = ms0 + ms1 k + ms2 ufac(k+1), ufac = the measured mean number of distinct CPU experts per forward of m positions `[code]` | author: +1.9%, exact output | exact | HAVE the measured version: `DraftPolicy::cost_ms` (src/spec/draft_policy.cpp:68) prices a window by its measured round time per size and context bucket, which contains the union term; `STRATA_SPEC_COST`, `STRATA_SPEC_COST_TABLE` seeded by `tools/pancake/trace/draft_costs.py:27` `new_misses_per_row` (distinct new CPU misses per extra row). Not coded: a parametric ufac fit; no switch |
| Dense thread pool on the hyper-thread siblings of the expert pool's cores (kindi/dsv41.py:7-8, `KINDI_DSV_PLACE=siblings`; the default shares the expert cores with the idle expert workers at SCHED_IDLE) `[code]` | none given | exact (scheduling) | CLOSED for us: SMT siblings, CCD split and futex parking were measured worse (MOD_PLAN.md section 9 do-not-repeat list; LEADS.md Part B pool parking/placement closed; spec arms `--host-core sibling`, T-162 never run). Not re-queued |
| 98.5% top-1 near-tie quality gate (kindi/dsv41.py:851 `verify`: PASS = at least 98.5% top-1 agreement and every disagreement a near-tie, reference top-2 margin at most 0.75) `[code]` | author: 20 prompts x 64 tokens | gate rule | ours: `tools/pancake/quality_gate.py` (token forcing, 20 requests x 300 tokens, two stock arms give the noise floor) and `tools/force_check.py` (rate rule + large-margin event rule). The idea we lacked, classifying each disagreement by the reference's top-2 margin, is CODED as a default-off report column: `quality_gate.py --near-tie` (`near_tie_columns`, T-693), no switch |
| Prompt lookup on by default (kindi/config.py: `lookup` 5 tokens per check; 2x on copy and edit) `[code]` | author: 91.6 vs 45.1 tok/s on a copy-with-changes file | exact | HAVE and on in the daily preset: `--lookup-max-t 12` (configs/pancake/flashnext-iq2_xs-fast-hip.json:40), `DraftPolicy::choose` (src/spec/draft_policy.cpp) |
| Recurrent-state checkpoints every 256 tokens, a request that shares a start resumes at the last one (kindi/config.py `checkpoint_every`) `[code]` | none given | exact | ours checkpoints at the prompt-cache root and fork points, not on a fixed stride (`STRATA_CACHE_ROOT_PAST` and `STRATA_CACHE_FORK_POINT`, src/core/conversation_disk_env.def:347, :419) plus the conversation cache; a fixed stride is untested: `STRATA_CKPT_STRIDE` CODED 2026-11-17 (`include/strata/program/ckpt_stride.hpp`, T-727; tool `mid_edit_bench.py` T-728; spec 142 queued T-729; MOD_PLAN.md section 6 item 42; nothing measured) |
| One shared MoE placement module for the CPU, GPU and disk tiers (kindi/moe.py `plan()` used by every engine) `[code]` | none | exact | design only: ours splits it between expert_cache.cpp (profile rank), expert_source.cpp (tiers) and `tools/pancake/tier_laws.py` (offline), no single module; hygiene, no speed, no switch |

## Part K: Tensward (LocalLLM post, u/tensward, read second-hand 2026-11-16: the owner's paste, not fetched by us; every number is the AUTHOR's, on their hardware)

| lead | number + hw | exact/lossy | in our src |
|---|---|---|---|
| llama.cpp copies GPU state to the host per request for prompt-cache / context checkpoints of a hybrid (recurrent-state) model; `--cache-ram 0 --ctx-checkpoints 0` removes it (Tensward tool found it; commenters: agentic multi-turn needs the cache) `[post]` | author: Qwen3.8-27B, H100, 16 concurrent clients, 643 MB GPU->host per request; 58 -> 110 tok/s, GPU util 35% -> 88%; on an 8B / L4 a returning-conversation test about 20% worse | exact (state placement) | HAVE the same copy in kind: one 112.2 MiB GDN-state checkpoint per request, synchronous on HIP (`src/core/conversation_state.cpp:285-292, 361`, taken at `src/program/generate.cpp:13883` / `:20700-20719`), plus the park of the outgoing conversation at the next request (`generate.cpp:19039-19062`, `src/core/conversation_snapshot.cpp:303`) and its restore. Not the contention case: `serve/server.py:7` is one request at a time and the preset has no `--batch`. Flags that switch each off: `--prompt-cache 0`, `--prompt-cache-every 0`, `--conversation-cache-mib 0`, `--conversation-cache-min-tokens` (preset 8192). TESTED OFFLINE ONLY (T-688, T-689): upper bound of the park time in the 11-09 replay 3.3-5.0% of prompt time for the preset; checkpoint ms on HIP NOT MEASURED. Spec 128 queued (T-690, MOD_PLAN.md section 6 item 37). CODED 2026-11-17: `STRATA_CKPT_ASYNC=1` now works on HIP (gfx1100; gfx906 and Metal stay off), rule `include/strata/program/ckpt_async_plan.hpp` (T-721), spec 140 queued (T-722), nothing measured |
