# Pancake leads catalogue

Every unported idea collected by the research rounds (forks of Niko1221/Strata, upstream PRs, llama.cpp, papers, hardware notes, Avalon comparison), checked against our tree on 2026-11-15. This file replaces about 140 research reports and notes and the side docs; those are deleted and stay in git history (`git show 5ab3d2db:<path>`; the "source" columns cite those paths, relative to `pancake-docs/research/` unless a full path is given). Column "in MOD_PLAN.md (line)" refers to the OLD plan at 5ab3d2db.

What is ranked and scheduled lives in MOD_PLAN.md (sections 5 and 6). This file is the full list to pick from; a lead moves into MOD_PLAN.md when it is chosen, and its test results go to TEST_LEDGER.md.

Tags: author numbers are on the author's hardware, not ours; "exact" = bit-identical output intended, "lossy" = needs token forcing + the task gate. Two-GPU and RTX 3060 / sm_86 items are parked (single RX 7900 XT box since 2026-11-08).

## 0. Leads added on 2026-11-15 (from the Speedy audit, the side docs and the forgotten-findings pass)

| lead | number + hw | exact/lossy | in our src | source |
|---|---|---|---|---|
| 0xluk `STRATA_PF_AHEAD` + `STRATA_REFILL_ASYNC`: prefetch the next chunk's experts into an N-slot pool from the loan while attention runs (the copy engine idles through attention on a warm turn); split must ignore the pool; log `refilled ... ms` before porting REFILL_ASYNC | agent prefill +35%, TTFT 4.97 -> 3.67 s (bundle); refill ~143 ms off TTFT; 2x RTX 3060 PCIe 3 | exact | no (only the unrelated `STRATA_IO_PF_AHEAD`); code in 0xluk `flashnext-pcie3` `896481c943`, `f9b14a4965` | active-forks-nov/report.md:38-39; PREFILL_LEVERS.md:263-282 |
| llama.cpp #31751: one pragma fixes RDNA3 MMQ J=32 register spills | +36% at 17-32-token batches, gfx1100 | exact | check the pinned ggml first | STRATA_DERIVATIVES_DEEP_RESEARCH.md:287 |
| launch fusions `FUSE_SHGATE/COMBINE/SWIGLU_DOWN/PLAN/ROUTEBELL/QGEMV` | 2,648 -> 2,248 kernels/window, R9700 | exact | no | active-forks-nov/report.md:42 |
| uniform K experts (`STRATA_Q36_EXPERTS_USED`) | +14.4% K=7/8, +18.7% K=6/8, KLD 0.0254 at K=7; RTX 3050 laptop, Qwen3.6 IQ4_XS | lossy | no (rank tail skip only) | active-forks-nov/report.md:48 |
| `EXPDEC_V2` remainder (issue-first loads, 24-bit multiplies) | expdec_gu<22> 69.1 -> 52.5 us, down 48.2 -> 38.5 us; R9700 | exact | partial (flat-load cast, F12 signs exist) | active-forks-nov/report.md:40 |
| phoenixclyde `STRATA_MMVQ_WAVE_ROWS` per-shape table | +4.0% decode, 46.8 -> 48.6 tok/s; MI50 | bitwise | no | active-forks-nov/notes/batch4.md:36 |
| gputier/StrataGP 16 packaged patches | +5-15% decode, +42-49% prompt; RTX 5090 | mixed | no | STRATA_DERIVATIVES_DEEP_RESEARCH.md:120 |
| Maya `MAYA_GLM_MISS_SKIP_W` (skip a missed expert by gate-weight share) | +7-13% decode at KL 0.037-0.052; GLM | lossy | no | STRATA_DERIVATIVES_DEEP_RESEARCH.md:49 |
| small exact bundle: `copy_rows_from_mapped` loads before stores (elementwise.cu:255-259); sampler/verify 8 logit loads in flight; single-token `MMVF_ROWS`; RouterLookahead submit after plan publish | 4.7-5.7 -> 2.4-3.9 us/launch (5090); ~110 us/argmax round (gfx906); 25.8 -> 10.2 us (R9700); part of +6.5% (R9700) | exact | no | active-forks-nov/report.md:30-36 |
| two-pass session restore; `CKPT_MOVE_PARK` port | 198,830-token session restored in 3.4-3.5 s; ~92 ms/park (MI50) | exact | no. NOTE: `STRATA_PARK_ASYNC` cannot act on HIP (`park_async_on()` is false, conversation_snapshot.cpp:231) | active-forks-nov/report.md:24, :45 |
| Speedy `STRATA_KV_STAGE_REQ`: size the streamed K/V staging pool by the request, not `--max-context` (moves the startup loan to the auto prompt chunk) | 1M ctx: auto chunk 512 -> 4608; 300K cold prompt 466 -> 1527 t/s (2x RTX 3080) | exact | no | Speedy `8f0d2e38` (+`0ccf9811`) |
| Speedy `STRATA_PF_SEL_BATCH` (QSA selection batch 16-256) | VRAM 256 -> 64 MiB per card at 1M cells | exact | no | Speedy `8f0d2e38` |
| Speedy `STRATA_TOOL_STEP_SPLIT` + `"canonical_tools"` (re-split client-merged tool steps, stable tool render order; serve/ only) | turn-2 cache reuse; no tok/s given | exact | no | Speedy `2bc7a7e0` (bundled with RP_SNAPSHOT, which we do not take) |
| upstream #5273 short-chunk ring release (`give_back(consumed)` only when `stream_all`) | correctness | exact | partial: one site guarded (prefill.cpp:6215), not 4252/4396/4527/5029 | Speedy `1c7a0b62` |
| routed-id out-of-range message with layer/entry/id | diagnostics | exact | message without detail (expert_source.cpp:4852, 5370) | Speedy `18de2117` |
| rdna3 `791cca4b` + `5906629d`: `PREFILL_AUTO_MAX` up to 49152 + `STRATA_PREFILL_EQUAL_TAIL_MAX` | fixes 49-57K chunk reads; 49,152 grid est. 2-4%, +0.6 GiB loan, needs refill head `2b19d510` (`REFILL_HEAD_BIG`), 2 GiB R-buffer limit | exact | AUTO_MAX yes, EQUAL_TAIL_MAX no; our equal-tail rule differs (their 49K trap cannot happen at our 32K ceiling) | batch2/rdna3-commits-of-interest.md:21 |
| rdna3 `f557b544`: prefill CPU share retries the pool at every layer | - | exact | no | batch2/rdna3-commits-of-interest.md:22 |
| rdna3 `88686782`/`877bf971` `STRATA_PREFILL_TIMING` CPU-share thread time + host wait; `7df6538d` slim prompt-only GGUF | diagnostics | exact | `STRATA_MTP_PROFILE` exists; the rest no | batch2/rdna3-commits-of-interest.md:23-24 |
| Juud_engine CPU tweaks: shorter worker spin, no activation quant for GPU-resident tokens; env-only test: `STRATA_POOL_SPIN_US` sweep (#4625: 0 means zero, pool.cpp:794) | +9.6-15.6% (4090); 15.1 -> 12.6 ms/token | exact | spin knob yes | STRATA_DERIVATIVES_DEEP_RESEARCH.md:131-136 |
| #4005 Vop3p sparse copy events + slot-wait dedup in prefill (PR closed unmerged) | +36% HIP prompt at 4K (RDNA2) | exact | no | STRATA_DERIVATIVES_DEEP_RESEARCH.md |
| `SELECT_TWO_PASS` on HIP (never checked) | <3% at 100K | exact | yes (untested on HIP) | TTFT_LEVERS.md:303 |
| prefill ring 96 vs 192 | ~1% (R9700: 192 cost 1.3%) | exact | yes (`PREFILL_RING=192`) | PREFILL_LEVERS.md |
| `PREFILL_ROUTED_STREAM` matters only just above 4,096 tokens; CPU share 0.6/0.7/auto only for <=400-token deltas (-0.05..-0.15 s) | - | exact | yes | PREFILL_LEVERS.md:213-236 |
| upstream PRs not yet triaged: #3925, #5566, #4876, #5593, #4972, #5637/#5437, #5588, #2392, #5643, #5470/#5596/#5453/#5472, #5545, #5517, #5471 (Q4_K/Q5_K decode GEMVs, DeepSeek path), #5623 (shared immutable host KV prefixes), #5609 (verify q36 MTP block loads but never drafts is fixed in our base), server #5516/#5512/#3971/#5628/#4777/#5466/#5518, bahooo22 q36 #5595/#5467/#5468, #5475 `HC_PERSIST_Q8`, #5535 int8-WMMA prompt attention on gfx1100, #3509 `ARENA_REUSE` + 0xluk `e2ffeef8da`, alesha-pro trellis "S" quant #3384 (lossy) | - | - | no | STRATA_DERIVATIVES_DEEP_RESEARCH.md:212-339 |
| Maya `MAYA_EXPERT_DIRECT`, Maya prompt memo, `--calibrate`, drafter auto-keep, Private_Context_COW | +26%; ~6% TTFT | - | no | STRATA_DERIVATIVES_DEEP_RESEARCH.md |
| havok2-htwo/StrataQ38FN agent-switch re-read patch | 25 s -> 1.5 s (author) | unknown | no | STRATA_DERIVATIVES_DEEP_RESEARCH.md:127 |
| storm-ace q36 GQA decode, sampled MTP, `Q36_FA` WMMA prompt attention | pp32768 360 -> 517 -> 938 tok/s | FP16-level lossy | no (overlaps F19) | active-forks-nov/notes/batch5.md:46 |
| svgop hugetlb pool for the expert copy | +4.9% IQ2_XS (CUDA) | exact | budget half yes (ple_residency.hpp:126); `HUGEPAGE_ARENAS` no | active-forks-nov/report.md:44 |
| deferred activation quantization in the pool + sample-async | +6.5% / +3.9% bundles (R9700) | exact | no | active-forks-nov/report.md:43 |
| GatherGroup fix `b13ffdabd0` for `ROUTED_STREAM`; dardeaw exact wave32 QSA decode kernels; `STRATA_FUSE_SHGATE` (this-cute-world) | - | exact | no | active-forks-nov/report.md:20; fork-knowledge-base/report.md:83-84 |
| watch list: Infernix (+23% at 262K, 5090); Qwen3.6 issues #5069 #2464 #5551 #1402 | - | - | - | STRATA_DERIVATIVES_DEEP_RESEARCH.md |

Closed by the 2026-11-15 pass (do not re-queue without new evidence): AMD MEXT (closed commercial NVMe-as-DRAM product for >64 GB servers; expert lookahead prefetch already measured 0.97x, T-190); new compression ISAs (AVX-512/GFNI/VPCLMULQDQ are Zen 4+, MXFP4/MXFP8 matrix cores are RDNA4, BMI2 pdep/pext microcoded on Zen 3; entropy-coded weights bound 8-12% < 20% bar, T-282); thread-count / SMT / CCD / futex / nice variants (measured worse; the pool is bandwidth-bound); BFLA on Flash-Next (paper BFLA 1.273x vs QSA 1.319x at 128K, selector 2.6% share); dedup (T-165); Avalon prompt path (fit 0.13 s + 0.61 ms/token, nothing to port); `--short-read` (crossover already ~64); DSpine (no drafter for our models); basalt draft head / K/V append / .basalt format; qwen38-megakernel (AGPL, ideas only); Aviary, jitllm, colibri, vLLM-class engines; SpecPrefill (lossy, breaks GDN); Speedy `speedy_defaults.hpp` (silent default flips), `RP_SNAPSHOT` (measured decode loss), its AMD block and its non-commercial licence; root `*_OPTIMIZATION_REPORT.md` recommendations (grep counts; the syncs/printf/memcpy they name are init, trace or error paths).



## Part A: fork-research leads vs our tree and MOD_PLAN.md

Scope: `pancake-docs/research/engine-speed/{speed-ideas-all-forks, strata-forks-speed-gains, active-forks-nov, fork-knowledge-base, source-commits}/**`.
Branch `docs/consolidate-plan`. MOD_PLAN.md line numbers are from the working copy (3984 lines, uncommitted edits present).
"in our src": `git grep` hit in `src/` or `include/` (file:line). A hit only in `conversation_disk_env.def` is not counted; the reading site is given instead.
"no" = absent. Numbers are fork-author claims on the author's hardware unless marked "ours".

Source path abbreviations (all under `pancake-docs/research/engine-speed/`):
SI = speed-ideas-all-forks/report.md; HN = speed-ideas-all-forks/notes/hip_rdna_kernels.md; MO = .../notes/moe_offload_cpu.md; SM = .../notes/speculative_mtp.md;
SG = strata-forks-speed-gains/report.md; AR = strata-forks-speed-gains/notes/amd_rdna_ports.md; PU = .../notes/platforms_upstream.md;
AF = active-forks-nov/report.md; B2 = active-forks-nov/notes/batch2.md; KB = fork-knowledge-base/report.md; SC = source-commits/notes/commits_2026-11-09.md.

### A1. speed-ideas-all-forks (FK-01..FK-59 plus extra leads from the notes)

| lead | number + hw | exact/lossy | in our src | in plan (line) | source file:line |
|---|---|---|---|---|---|
| FK-01 exact spec sampling `SPEC_PROB=1 COUPLED=1 GATE=top DT=0.85` | +5.5% agent T=1.0 (riceharvest, Intel B70); +5.8%, +13.4% with DT 0.85 (svgop, RTX 5090); 0 on greedy | exact | coupled_draft.hpp:69; spec_prob.hpp:53; sampler.cu:950 | 2086 (F6), 2992 | SI:36,206; SM:168 |
| FK-02 `--prompt-cache-spread` in preset | edited 82K resume 40.0 to 10.2 s (xrip, 2x Turing); re-read cells 130K to 86K (Lamina) | exact | generate.cpp:994 | 2082 (F2), 2991, 3005; parked row ~3603 | SI:48,207 |
| FK-03 `--lookup-max-t 16` x `--lookup-chain 3` | 1,477/1,487 lookup drafts accepted on rename edit (zsariboga, RTX 2060S); est. 0-4% edits | exact | generate.cpp:1024 (chain); lookup_default.hpp:1 | 2989, 2991 | SI:42,208 |
| FK-04 `--aux-cpus auto` (= Ben1078 `HELPER_CPUS`) | +3.8% decode, host-plan wait 1.52 to 0.51 ms/window (Ben1078, 7800X3D + RX 7900 XTX) | exact | session.cpp:570; aux_cpus.hpp:9 | 2989 (spec 31) | SI:64,209; HN:222 |
| FK-12 `--batch 2` for parallel sub-agents | 70/105/150 tok/s aggregate at 2/4/8 requests (unimatrix099, 2x XTX) | exact per stream | generate.cpp:3963 | 2991, 3043 | SI:48,210 |
| FK-07 `STRATA_SH_STREAM=1` | +4.0% decode on gfx1100 (xyzzing, XTX); noise on RTX 3060 | exact | arch_defaults.cpp:88 | 2989 (spec 31) | SI:56,213; HN:141 |
| FK-08 `PREFILL_AUTO_MAX=24576` | prefill +15% at 32K, +7% at 64K (rxr2, XTX 64 GB) | exact | generate.cpp:847 | 2990 (spec 32), 3954 | SI:64,214; HN:175 |
| FK-09 `STRATA_WMMA_GEMM=1` | prompt +5.6-9.1% (troytse, gfx1151) | rounding | gemm.cu:1162 | 2990 (spec 32) | SI:64,215 |
| FK-10 `ATTN_LANECELL=1 ATTN_MERGE_V2=1` at 28K+ | decode +9..+24% in a bundle (UltimateBoomer, gfx1101); rxr2 XTX: did not hold above noise | unknown (hash) | arch_defaults.cpp:51-52 | 3043 (near do-not-repeat) | SI:216; HN:66 |
| FK-13 `STRATA_PF_HCDOWN` KL recheck (preset sets 1) | +1.3% prefill but first-token KL doubled at 16K (rxr2, XTX) | lossy check | prefill.cpp:580 | no | SI:218; HN:184 |
| FK-20 `STRATA_PARK_TRUNCATE` | 160K chain re-read 225 s avoided (TwoAnts); ours: 7 cold 54-61K turns, ~162 s | exact | generate.cpp:974 | 2081 (F1), 1757; parked ~3603 (night test <54K, no gain) | SI:46,219 |
| FK-21 `STRATA_SPEC_CUM` cumulative draft stop | +6.4% short, +2.1% agent with chain 3 (android-imdad, XTX IQ3_S) | exact | src/spec/disco.cpp:106 | 2088 (F8), 1567 | SI:38,220; SM:49 |
| FK-21b `SPEC_LOOKUP_Q` lookup-agreement arm | +5.8% unseen prompts; "a wash" with suffix drafts on (android-imdad, XTX) | exact | no | 3000 | SI:38; SM:57 |
| FK-21c SPEC_CUM + `--spec 6` two-arm | `--spec 6` +5-20% vs 4 (android-imdad, XTX); ours: flat alone | exact | flag only | no (only do-not-repeat 3036 for spec 6 alone) | SI:38; SM:62 |
| FK-22 router publish fast path `STRATA_ROUTE_FAST` | ours: route_publish_res 17.8 us x 4,128 = 2.68% GPU; DPP ~1 us/launch (android-imdad, XTX); +2.5% code (Lamina, RTX 3090) | exact | verify.cpp:2529; native_router.cu | 2089 (F9, coded on branch) | SI:52,221; HN:86 |
| FK-23 bit-exact Q2_0 AVX2 fast unpack | 1.11-1.45x on 1 thread; all-Q2_0 layer -17% in a 7-worker pool (imanu86) | exact | q2_avx2.cpp:60 (`unpack64_fast`, opt-in `STRATA_Q2_FAST`, :247) | 2087 (F7), 2999 | SI:32,222; MO:128 |
| FK-24 Q2_0 down 4-lane fold `EXP9_DOWN` | down 0.47-0.96x of mode 8, +1.6-2.0% e2e (KhaledElzarw, RTX 3090) | exact | iq_kernels.cu:6500 | 1675, 2996 | SI:54,223 |
| FK-26 fill-stall grid padding | empty 384x256 launch 14-16 us; trim to 96 blocks +6% decode (unimatrix099, 2x XTX) | exact | no | 2407 (F20), 2996 | SI:56,227; HN:123 |
| FK-27 cache bundle: `PARK_REWIND_MAX` / skip-tail / fork-point / borrow / trailing reminder | 12 turns 10.4 to 6.3 s (Xiaomi2008, RTX 5090); 6 tasks 412 to 369 s, switch 3.00 to 0.67 s (Lamina, RTX 3060) | exact | generate.cpp:393 (REWIND_MAX); skip-tail, fork-point, reminder: no | 2627, 2083 (F3), 2995, 3005 | SI:48,224; AF:33 |
| FK-28 `STRATA_ADAPT_PROMPT` | est. +2-5% first decode tokens; RandoTeam counter -3% prompt (RTX 5090) | exact | no (`set_usage` dead code, prefill.hpp:299) | 2090 (F10), 3928 (correction: dead code) | SI:32,225 |
| FK-30 IQ2 grid-sign expansion | IQ2_XXS gate/up -29%, IQ2_S down -35% per kernel (yqYo1, RX 7900 XT, GLM engine) | exact | iq_kernels.cu:91 (`STRATA_IQ_SIGN_FAST`) | 2092 (F12) | SI:56,230; HN:16 |
| FK-31 sub-warp grouped expert kernels | +5.7-8.3% e2e (Lamina, RTX 3090 + 5070 Ti); est. 0-1% on XT | exact | no | 2996 | SI:54,229 |
| FK-32 `WAIT_COPY256` + 16 B plan reads | host-write-to-plan 4.47 to 3.57 us (android-imdad, XTX) | exact | switch_report.hpp:31 | 2408 (F21), 1759 | SI:56,231; HN:100 |
| FK-32b `STRATA_ADAPT_COPY2` (2 DMA engines) | 48 x 2.2 MB copies 4.00 to 3.70 ms (android-imdad, XTX) | exact | no | 2408 (F21) | HN:217 |
| FK-32c MULTI_BLOCK_ARGMAX on gfx1100 | 12 to 5.6 us a pick (android-imdad, XTX) | exact | verify_kernels.cu:1026 (gfx12 only) | 2408 (F21) | HN:146 |
| FK-33 `DRAFT_EARLY` + `DRAFT_SUFFIX_SKIP` | no number; draft step 2.2-2.5 ms caps it ~1% | exact | no | 3044 | SI:42,232 |
| FK-34 `--light-expert-cpu` byte-aware placement | est. 0-2% (IQ2_XS file only) | exact | no | 3044 | SI:32,233 |
| FK-35 shared n-gram pool `NGRAM_MOD` | 234/234 drafts on a copy fixture, no tok/s (tctien342) | exact | no | 2995 | SI:42,234 |
| FK-36 prefill combine + write_norm fusion | 234 to 203 ms; 32K -4.0% (zsariboga, RTX 5090) | exact | no | 3044 | SI:62,235; HN:191 |
| FK-36b `PF_DQ_OVERLAP` side-stream dequant | prompts +7-11% at 8K chunks, 0 at 32K + DOWN_STREAMS (android-imdad, XTX) | exact | no | 3011 | HN:177 |
| FK-40 post-READY warm-up replay | first turn after restart only (svgop) | exact | no | 3044 | SI:236 |
| FK-41 arena mlock vs HMM userptr stalls | queue stalls under memory pressure (Ben1078, XTX) | exact | mlock facility (platform/memory.cpp); VmLck unverified | 2995, 3044 | SI:64,237; HN:220 |
| FK-50 `STRATA_VERIFY_ECAP` tail expert cap | +2.8% pooled, +11.5% drift-paired; tight cap broke 12/20 tool turns (jmnargi, 2x V100) | lossy | expert_source.cpp:4989; verify_ecap.hpp | 2085 (F5), 1763 | SI:40,238; SM:98 |
| FK-51 route-prior guards `PROTECT/KEEP_P/SCOPE` | 0 alone; lets turbo run lambda >0.5 (ours turbo 0.5: +25-34%) | lossy | route_prior.cu:10-13 | 2093 (F13), 1762, 2622 (17.1) | SI:239; MO:98 |
| logit-range-scaled prior bias (`--cache-prior`), SER | no number (gputier) | lossy | no | no | MO:101,111 |
| turbo vs uniform tail skip at equal KL | uniform top-k beat miss-targeted dropping (lighttransport, MiniMax) | lossy method | ROUTE_TAIL_SKIP expert_source.hpp:670 | 2093/2095 | MO:102,113 |
| FK-52 sub-2-bit shadow store for CPU misses | +2.9% solo, +20.8% concurrent; tmpfs/hugetlb +5.5% (svgop, RTX 5090) | lossy | no | 2084 (F4) | SI:28,240 |
| FK-53 ISTA Coder pack 256/512 | +7.8/+10.0/+11.1% masked-router analogue (svgop, RTX 5090) | lossy | n/a (model) | 3014; item 25 / A28 | SI:30,241 |
| FK-54 window decode attention | 200K decode 48.6 to 68.2 exact, 88.9 fp16 (Asofwar, RTX 5080); Flash-Next capped ~1% | rounding | q36_engine.cu:363 (`STRATA_ATTN_WINDOW`, q36 only) | 2091 (F11) | SI:60,242 |
| FK-55 Qwen3.6 on HIP: storm-ace WMMA flash prompt + GQA decode | pp32768 360 to 938 tok/s (storm-ace, gfx1151) | fp16 | no (`Q36_FA`, `Q36_ATTN_GQA` absent) | 3046, 3978 | SI:60,243 |
| FK-57 requant study UD-IQ2_XS vs our GSQ IQ2_XS | est. up to -20% expert bytes | lossy | n/a | 3045 | SI:245 |
| FK-58 typical acceptance in thinking only | unknown; thinking ~6.6% of turn time | lossy | no | 3045 | SI:246 |
| FK-59 fp16 rocWMMA QSA select arm / `SELECT_SPLIT2` | fp16 arm 1.64x select at 128K (xyzzing, gfx1100) | exact / lossy | SELECT_WMMA in src (which arm: unverified); SPLIT2 no | 2070 (id only) | SI:247; HN:64 |
| android-imdad exact decode bundle `TOPK_REGKEYS/TOPK_SCAN/MTP_SELECT_WIDE/GDN_NORM128/ATTN_MERGE_BOUND` | bundle (with ROUTE_FAST etc.) short chat 86.8 to 95.8 (+10%), agent +11%, KL 0 (XTX IQ3_S); no per-switch split | exact | no | no | HN:54-60 |
| `PF_GU_EXACT` Tensile gfx1100 prompt GEMM | 26.5 vs 12.9 TFLOPS; 16K 13.05 to 10.05 s, 64K 40.1 to 31.5 s, KL 0 (android-imdad, XTX) | exact | no | no | HN:167 |
| `PF_DOWN_STREAMS=4` (down over 4 streams) | prompts 2K -3..-4%, 16/32K -2% (android-imdad, XTX) | exact | no | no | HN:172 |
| `HIP_POST_SPLIT` / persistent FFN half | +3.5% (needs PCIe share); FFN-half floor 67 vs 150 us/layer, est. +10-14% (unimatrix099, 2x XTX) | exact | no | parked 1213 (persistent post-bell) | HN:131,134 |
| `HIP_FRONT_FUSE` / `FUSE_SHGATE` (shared gate into router GEMV) | +2.7% decode, 15 to 11 kernels/layer (unimatrix099, 2x XTX) | exact | no | 3969 | HN:101; KB:83 |
| `PCIE_USAGE` + `PCIE_KEEP=96` swap-ahead ring | +4.1%, copy wait 1.2-1.7 to 0.2-0.5 ms (android-imdad, XTX, pcie-frac 0.35) | exact | no | no | HN:218 |
| IQ2_S/IQ3_S AVX2 decode without shifts (HiSpread, imanu86 862c18aaea) | IQ2_S layers pool -20..-33%/window | exact | iq_avx1.cpp:72,102 (AVX1 file; AVX2 row path unverified) | no | MO:129 |
| pool spin ~10 ms before nap | CPU experts 15.1 to 12.6 ms/token (Jackwwg83, Zen 5 laptop) | exact | `STRATA_POOL_SPIN_US` pool.hpp:291 (value sweep unverified) | no (3950: threading variants measured worse) | MO:134 |
| one-token exact AVX2 rows (`CPU_SINGLE_EXACT`) | IQ3_S gate/up 2.0x on P-core after fix (Strata-GLM) | exact | `STRATA_CPU_IQ_ALL_EXACT1` (def:348; reader unverified) | no | MO:130; PU:16 |
| `SLOTS_WARP` warp-per-row slots kernel | 98.5 to 111.4 t/s short (RandoTeam, RX 7800 XT, Qwen3.6 poly) | exact | no (only `native_mmvq_slots_kernel`) | no; poly is CUDA-only: parked | HN:32 |
| `MTP_HC_Q8` draft-layer Q8 copies | draft step -0.03 ms | rounding | no | no | SM:108 |

### A2. strata-forks-speed-gains

| lead | number + hw | exact/lossy | in our src | in plan (line) | source file:line |
|---|---|---|---|---|---|
| `STRATA_GDN_PP=2` | recurrence 1.76x, prompts +2% (2x PRO V620); 13.6K recurrence 999 to 643 ms (Linsche, RX 6800) | exact | arch_defaults.cpp:44 (default only gfx12/gfx1151) | 2989 (spec 31); 1751 screen void, 1574 rerun | SG:21; AF:67 |
| `STRATA_DENSE_MMQ=1` (+`MMQ_KQ`) | prefill +11-13%, KL noise (P40); q36 "+43%" (burzumishi) | rounding | gemm.cu:1377 | 3060, 3929 | SG:22; AR:66 |
| `STRATA_PREFILL_LAYER_MAJOR=1` | 4,400 tok 59.8 to 7.4 s (SSD-bound, RTX 5090); +1.5% (asahi) | exact | prefill.cpp:1283 | 2622 (lever 7, est. 2-4% at 64-100K) | SG:23; PU:22 |
| `--kv-grow` | +6% (94.5 to 100.0, RTX 3090); 69.6 to 73.5 (RTX 4070) | exact | layer.cpp:621 | 2989 (spec 31) | SG:24 |
| `--adapt-decay 0.97` | 0.97 best; 0.95: 120.2, 0.99: 113.8 tok/s (RTX 3090) | exact | generate.cpp:923 | 2991 | SG:25 |
| `--vram-reserve-mib` 444 / 1000 | 444: prefill +10-17% (RTX 3080 10 GB); 1000 stopped evictions (XTX) | exact | expert_source.cpp:6914 | 2991, 3961 | SG:26 |
| `STRATA_DUMMY_NODES` per-node cost slope | 0.86 us/node x 2,158 = 1.9 ms/window (RTX 3090) | measurement | verify.cpp:1935 | no | SG:27 |
| `--spec-min-p 0.7` (+ `--adapt-every 8`) | 0.70 best (RTX 3090); +2.9% story, +9.6% code, -2.7% 16-topic (Spark-NV, RTX 5090) | exact | mtp.cpp:1859 | 2989; 1751 screen void | SG:28; AF:23 |
| `PREFILL_CPU_SHARE` auto / MAX | auto: 214 tok 2.4x, 620 tok 1.8x (2x Radeon VII) | small-lossy | prefill.cpp:271 | 411, 3932 (ours spec 46: MAX 1024 -25.5%) | SG:29 |
| no PCIe staging slots at `--pcie-frac 0` (strata-slim 0003) | ~61 MiB/GPU back to cache (CUDA) | exact | no (verify.cpp:1253 allocates always) | 3002 | SG:36 |
| `STRATA_POOL_GUIDED` shrinking tail ranges | 29.6 to 31.3 tok/s with a CPU fix (Core Ultra 285K) | exact | no | 3004 | SG:37; PU:17 |
| Argon AVX2: 2 accumulators, 32/16-row granularity, `row_dot_pair` | no isolated A/B (MI50 + AVX2 Xeon, 27.8 tok/s) | exact | no (iq_avx2_rows.inl:91 still `: 1`) | no | SG:38; AR:60 |
| `STRATA_IQ_ZEN3` gather-free IQ decode | +0.8% (119.5 to 120.5, RTX 3090 + Zen 3) | ~exact | no | no | SG:39 |
| DraftPolicy cost-outlier clip (`PL_POLICY_CLIP`) | one 31 ms stall flips MTP T3 to lookup T8, persists (gfx1030, CPU tests) | exact | no | 3003, 3022 | SG:41; AR:105 |
| HotRod weight x logit-gap budget (`ROUTE_BUDGET`) | ranks 3-9: 44.7 vs 46.4 tok/s, 12/12 JSON (RTX 3060) | lossy | no | no | SG:42 |
| skip non-resident experts < 10% weight (`SKIP_MINW`) | 20 to 25 tok/s, KL 0.016-0.022 vs floor 0.013 (RTX 5090, GLM) | lossy | no (rank-based ROUTE_TAIL_SKIP only) | no | SG:43; PU:23 |
| balanced chunks (+1/32 rows) | small, not isolated (asahi) | exact | no (related `PREFILL_EQUAL_TAIL` rule) | 3947 | SG:44 |
| HotRod fragile-token re-verify + auto budget + brake | 1-3% windows cut; 27/27 clean at 37-48 tok/s (RTX 3060, NVMe) | lossy guard | no | 3014 | SG:50 |
| rdna3 unported prompt kernels `1799c0a3a`/`3dbb40516`/`1f4b77d72` | expert GEMM -6..-9%; HC half-layer -7.4..-9.1%; colskip 1-GPU unchanged (XTX) | exact | no (`NH_*`, `PF_F16_OCC` absent) | 3010 | SG:52; AR:35-37 |
| `STRATA_VRAM_CALIB` (PR #5691) | +314/+166 slots, free 884 to 300 MiB (2x RX 6900 XT, HIP) | exact | no | no | SG:53 |
| `STRATA_PROMPT_STAGE_MEMO` | short turns +14-20% (gfx906) | ~exact | no | no | SG:54; AR:128 |
| disk conversation cache + RAM/disk dual-park | 12,175 tok restore 0.32-0.51 s vs 42-45 s cold (gfx906 box) | exact | park-gain only (generate.cpp:1166) | 3005; items 39/83 | SG:55; AR:126 |
| `DQ8` + `HC_GATED16`, `KV_PF_RESIDENT` | IQ3_S prompt +6%; +7-9% at 16-128K (2x PRO V620) | exact | no | 3011 | SG:56; AR:46 |
| gfx1030 decode: `MMVQ_ROW/SET`, `NORM_ROPE=2`, `HEAD_WIDE/ROUTE_PLAN/HEAD_SAMPLE` | 119.2 to 122.1, one run (+1%) | exact | no (only `STRATA_NO_NORM_ROPE`) | no | SG:57; AR:48 |
| `__launch_bounds__` / occupancy audit | QSA value -10.9%, 128K prefill +1.45% (P40) | exact | method (405 uses exist) | no | SG:58; AR:68 |
| HotRod prompt stand-ins (`PREFILL_STANDIN`) | 76-118 tok prompts 35-45% faster; 8.5K 15.8 to 15.1 s (RTX 3060) | lossy | no | no | SG:59 |
| `--adapt-async 1` (swaps on own thread) | short chat 24.9-26.0 to 28.7-30.1 (Strata-GLM, RTX 5090) | exact | adapt_async_default.hpp:1 | 1800 (as A7 prerequisite) | PU:18 |
| `STRATA_NT_COPY` non-temporal blob copies | DMA 46.6 GB/s alone vs 17.0 beside 4 memcpy threads (Strata-GLM) | exact | no | no | PU:19 |
| hipBLASLt table auto only on gfx1151 after v0.2.13 (`343b8eae6`) | 131K prefill 926 to 1,687 with right table (XTX community); ours table +38% prompt | rebase hazard | gemm.cu:375 (`STRATA_HIPBLASLT_TABLE`) | no (hipBLASLt at 420, 507) | SG:70; SC:81 |
| bimodal: per-run AnonHugePages, `compact_memory` vs `NO_ARENA_THP` | ours: 38 starts "THP NOT requested" (6 GiB free 2 MiB blocks of 33) | method | pinned.cu:474 | 3019 | SG:78; AF:83 |
| bimodal: DPM clock ramp | first request -26% (59.3 vs 80.4, XTX) | method | n/a | 3020 | SG:80 |
| method: median [min-max], `strata_bench.py --runs 3 --long`, sweep calibration, n >= 6 for prompt A/B | cold TTFT CV 71-96% (phoenixclyde) | method | n/a | 836, 1791, 2567 | SG:82; AF:83 |
| momoe quality protocol (KL + top-token + scored tool tasks) | tool use 95.0 to 52.5% at lambda 1.5 with KL only 0.251 | method | n/a | 3014 | SG:82 |

### A3. active-forks-nov

| lead | number + hw | exact/lossy | in our src | in plan (line) | source file:line |
|---|---|---|---|---|---|
| `STRATA_LOOKAHEAD=0` when RAM complement is whole | window 17.4 to 15.8 ms, 162.2 to 182.9 tok/s (Spark-NV, RTX 5090 + 7800X3D) | exact | generate.cpp:9516 | 1574 (rerun first), 1751 (screen void) | AF:18,75 |
| `PREFILL_COPY_WAIT_DEDUPE=1` | warm 4,712 tok 5,928 to 5,340 ms, +11% in a bundle (R9700) | exact | prefill.cpp:3958 | 1751 ("routed-dedupe", void) | AF:19 |
| `PREFILL_ROUTED_STREAM=1` + GatherGroup fix `b13ffdabd0` | no isolated figure; +10% IQ3_S / +40% Q4_0 (gfx1030, `PF_STREAM_ROUTED`) | exact | prefill.cpp:3961; GatherGroup fix: no | 620, 3034, 3978 | AF:20; SG:51 |
| `STRATA_MMVF_ROWS=1` (+ single-token router) | +3.9% with DENSE_LDS (2x R9700); router 25.8 to 10.2 us (R9700) | exact | arch_defaults.cpp:80 (gfx1151 only); 1-token: no | 1751, 3975 | AF:21,36 |
| `STRATA_MMVQ_R=2/4` / per-shape `MMVQ_WAVE_ROWS` table | +4.0% decode, 46.8 to 48.6 (MI50, wave64) | rounding vs preset | native_mmvq.cu:2662 | 1574 (skip), 3972 | AF:22 |
| `PARK_ASYNC=1` check + `CKPT_MOVE_PARK` port | ~92 ms saved per park (MI50) | exact | conversation_copy.hpp:124; CKPT_MOVE: no | 3976 | AF:24 |
| `STRATA_B6_MMVQ_ROWS` on gfx11 | default on gfx12, no gfx11 data | exact | nv_defaults.cpp:39; native_mmvq.hpp:34 | 3978 | AF:25 |
| global (not flat) loads for expert weights | 0-9% kernel time, 0/460 diffs (Lamina, RTX 5070 Ti / 5060 Ti); ours flat_load 13,119 to 108 | exact | iq_kernels.cu:30 (build macro `STRATA_HIP_GLOBAL_WEIGHTS`) | 1574 (3 kernels gain scratch: speed risk) | AF:27 |
| LDS-only barrier on gfx11 | no number | exact | no | 1574 (low) | AF:28 |
| DPP / ds_swizzle for constant-mask `__shfl_xor` | ~4 us/token HC (gfx906); shexp down 6.23 to 5.86 us but head +1.6% slower (R9700) | exact | partial: intrinsics.hpp:119-150 (our 89da64b93) | 1574 (DPP_XOR skip) | AF:29; HN:31 |
| sampler / verify 8 logit loads in flight | argmax ~110 us/round, latency-bound (gfx906) | exact | no | 3975 | AF:30 |
| `STRATA_ACTQ_SKIP` | tier-5 bundle +4.0% (RTX 3090), no isolated | exact | no | 1574 (skip, 0.1-0.3%), 2283 | AF:31 |
| `copy_rows_from_mapped` loads before stores | 4.7-5.7 to 2.4-3.9 us/launch (RTX 5090) | exact | no | 3975 | AF:34 |
| RouterLookahead submit after publish + skip when resident | part of +6.5% bundle (R9700) | exact | no | 3975 | AF:35 |
| `STRATA_PF_AHEAD` + `STRATA_REFILL_ASYNC` (staged path, <4,096-tok deltas) | agent prefill 512 to 692 tok/s, TTFT 4.97 to 3.67 s in bundle; refill -143 ms (0xluk, 2x RTX 3060, PCIe 3.0) | exact | no | 3967, 2622, 3932 | AF:38-39,77 |
| `STRATA_EXPDEC_V2` grouped expert kernels | expdec_gu<22> 69.1 to 52.5 us, down 48.2 to 38.5 us, 672 bit-identical (R9700) | exact | no | 2092, 3971 | AF:40 |
| `STRATA_POOL_DAG=2` | pool 5.197 to 4.788 ms/window; engine neutral (9950X DDR5) | exact | DAG=1 = pool.cpp:1309 mode 7; DAG=2 no | 1574 (skip), 2285 | AF:41 |
| launch fusions `FUSE_COMBINE/SWIGLU_DOWN/SHGATE/PLAN/ROUTEBELL/QGEMV` | 2,648 to 2,248 kernels/window, no tok/s (R9700) | exact | no | 3969 | AF:42 |
| deferred / parallel act quant in pool + sample-async | +6.5% / +3.9% bundles (R9700) | exact | no | 3978 | AF:43 |
| hugetlb CPU expert copy + RAM budget fix | IQ2_XS solo +4.9%, lane +7.2% (svgop, CUDA) | exact | no (budget via available_memory_bytes blind to pool) | 3978 | AF:44 |
| two-pass session restore | 198,830 tok (3.05 GiB) restored 3.4-3.5 s (ElXreno) | exact | partial | 3976 | AF:45 |
| `STRATA_CPU_POOL_COST` longest-first | code +6.54%, geomean +1.95% (gfx936 DCU) | exact | no | 1574 (skip) | AF:46 |
| uniform fewer experts K=9/8 (`EXPERTS_USED`) | K=7/8 +14.4%, 6/8 +18.7%; KLD 0.0254 at K=7; HumanEval flat (RTX 3050 laptop, Qwen3.6 IQ4_XS) | lossy | no | 3970 | AF:48; B2:58 |
| `gdn_gate_kernel` computes only first token | latent (callers pass n_tokens=1) | fix | elementwise.cu:43 (bug present) | 3980 | AF:54 |
| temperature < 1e-30 / NaN treated as greedy | gap | fix | sampler.cu:869 `<= 0.0f` only | 3980 | AF:61 |
| bugs 5.5-5.10 (spec-split B flag, pipeline-windows rows, stats-flush crash, iq_embed grid.y, conv-cache publish, JSON watcher) | latent | fix | present per AF | 3980 (not re-checked) | AF:55-60 |
| layer-major group scratch `G = m.T*K` ignores group_cap (precondition for LAYER_MAJOR test) | latent | fix | prefill.cpp group_cap at 1183/2190; fix at 6798/7148 unverified | no | AF:53 |
| `STRATA_ARENA_REUSE` (#3509) + 0xluk GGUF arenas `e2ffeef8da` | startup only | exact | no | 3978 | AF:69 |

### A4. source-commits (2026-11-09) and fork-knowledge-base

| lead | number + hw | exact/lossy | in our src | in plan (line) | source file:line |
|---|---|---|---|---|---|
| tool-arg typing (Maya `_declared_type`, StrataFast bool/null) | robustness | exact | ported (2f0cee057, serve/) | 874 | SC:16 |
| PleReader::close() resets `max_inflight` | hang fix | exact | ple_reader.cpp:113 | 874, 3624 | SC:17 |
| `response_format` top-level schema (llama.cpp #28697) | compat | exact | no | 874 (parked, low) | SC:18 |
| quarter-size last pool tasks (`part_row`, basalt 553e9290d) | est. 0.5-1.5% CPU-expert time | exact | no | 874 (to dan-99) | SC:19,54 |
| down-row prefetch during fready wait (basalt 4e6841265) | basalt down phase -6% | exact | no | 874 (bundle with item 29) | SC:20,61 |
| SpinWait (pause 50 us then yield) in MTP draft wait | est. 0-1% | exact | no | 874 (parked, fold into 74) | SC:21 |
| DraftPolicy cost = shared scale x per-size base | 0-2% | exact | no | 874 (parked) | SC:22 |
| commit queued behind MTP draft launch (basalt 1fd53a447) | idea | exact | no | 874 (item 10) | SC:112 |
| file-cache floor rule (>= 1 chunk + 2 layers, >= 4 GiB) | design for RAM_KEEP/mmap tier | n/a | no | no | SC:23,194 |
| key saved sessions by lossy routing switches (llamAmpere) | correctness | n/a | conversation_disk_env.def (KeyIfSet scheme) | 874 | SC:173 |
| Qwen official sampling presets (thinking T 1.0 / top_p 0.95 / top_k 20; non-thinking T 0.7, presence 1.5) | fact; makes FK-01 relevant if opencode keeps T 1.0 | n/a | n/a | 874 | SC:183 |
| ggml vendoring + STRATA_GEOMETRY rebase hazard (upstream 067b4dda5 etc.) | hand-port only | n/a | no | no | SC:94 |
| this-cute-world `FUSE_SHGATE`; dardeaw exact wave32 QSA decode kernels | no number | exact | no | 3969, 3978 | KB:83-85 |
| fork-KB refresh `tools/forks/fork_kb.py` weekly | method | n/a | n/a | 3984 | KB:96 |

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

Repo `C:\Users\iancu\repos\pancake-strata`, branch `docs/consolidate-plan` (HEAD `5ab3d2db2`). Read-only check.
Source keys (all under `pancake-docs/research/engine-speed/`): GT = grouped-technologies/report.md; PD = projects-derived/report.md;
PP = pain-points/report.md; PPe = pain-points/notes/engine.md; CD = code-deep-research/report.md; MS = moe-speedup/report.md;
MSr/MSt/MSc/MSx/MSe = moe-speedup/notes/{rdna3_gpu_decode, transfer_and_kv, cpu_expert_compute, expert_caching_prediction, engines_labs_companies}.md.
"in our src" = `git grep` over `src include` (serve/ checked for server items, marked `serve:`); "-" = no greppable name, checked by concept only.
"in plan" = first relevant MOD_PLAN.md line(s); "no" = no hit by ID, switch or concept. Hardware: 7900XT = our RX 7900 XT gfx1100 + 5700G DDR4-3200 + PCIe 3.0.
Dropped as irrelevant (2-GPU / multi-user / non-HIP hardware): Strata-DualGPU + #4522/#4534/#4549/#859, rdna3 tensor-parallel/RCCL/kfdq, StrataFast `--prompt-split`, StrataQ38FN multi-user scaling, XeStrata/#3607/#5540 SYCL, Gaudi2, #2392 (gfx1101 table), #4876/#3277 (sm_75), #4854 (AVX1), #4886/#5647/#5644 (Metal), #5592 (Jetson), #5453/#5472 (DGX Spark UMA), Infernix NVFP4, `.basalt` pack (sm_120), `--attn-split` (CUDA tensor-core), DeepEP/EPLB/AFD/MegaScale (data-center), poly fused FA, #5467 (mma.m16n8k16, NVIDIA PTX).

### 1. grouped-technologies (F1-F10 family map; overlaps pain-points)

| # | lead | number + hw | exact/lossy | in our src | in plan (line) | source file:line |
|---|---|---|---|---|---|---|
| B1 | CPU-expert activation sparsity masks (Berkeley gate / TEAL / CATS / CHESS / SpAx), item 13 | 84.5-87.4% sparsity Qwen3.5 MoE (paper); ceiling 1.25x, est. +10-20% | lossy | no | 919 (item 13), 1176 | GT:30-35, GT:47 |
| B2 | Union sparsity over verify window w=1/2/4/8 (BlockFFN) column mandatory | 80% -> 71% at w=8 (Asia) | lossy | no | 919 (item 13) | GT:35, GT:47 |
| B3 | turbo-hip `STRATA_ROUTE_PRIOR 0.5` (Cache-Conditional Experts) | 149-169 vs 128.7 tok/s, 7900XT; 2.89% mismatch 3060 | lossy | src/core/conversation_disk_env.def:36 | 159, 908 (item 7) | GT:36, PP:15 |
| B4 | AcceptMoE depth-graded prior, pos 0 exact (item 23) | 2.06x vs SD under offload, RTX 5090, -0.27 pp; est. 5-15% over fast-hip | lossy | include/strata/kernels/route_prior.hpp:12 | 955 (item 23), 191 | GT:37, MS:77 |
| B5 | OEA sibling-aware bonus in verify window | idea, low ceiling | lossy | no | 132 (pick one of 7/23/OEA) | GT:38 |
| B6 | Tail skip `STRATA_ROUTE_TAIL_SKIP 7` / tail-row drop w/ MTP confidences (item 5) | 129.7 tok/s 7900XT (skip) | lossy (skip) / exact (row drop, per plan) | include/strata/core/expert_source.hpp:670 | 906 (item 5), 230, 1038 | GT:40 |
| B7 | ISTA Coder variant 256/512 experts (item 25) | 29.6 GB resident; one-evening A/B | different model | - | 960 (item 25) | GT:41 |
| B8 | Admission on k-th miss `STRATA_ADAPT_KMISS` (item 14) | sim +5..16% edit; replay hit 76.7->86.5% edits | exact | include/strata/core/epoch_bandit.hpp:9 | 929 (item 14), 1816 | GT:56, PP:11 |
| B9 | `--adapt-every 1 --adapt-swaps 64/96/160` tier at every window | replay 79.6% -> 85.0% VRAM hits; sim +4..9% | exact | include/strata/core/adapt_async_default.hpp:24 | 216, 445 | GT:57, MS:43 |
| B10 | Keep decode cache through long prompts / lend sweep (item 2) | 9,412 slots 12.6 GiB borrowed; 125->105 tok/s; +20% after-long turns | exact | no (`lend-sweep`) | 897 (item 2), 1494 | GT:58, PP:17 |
| B11 | SIEVE / S3-FIFO, OpenEvolve/ADRS, Hawkeye, Harmonia, MoE-Infinity as cache policy | Belady-bounded; only if sim >=5% | exact | no | 1332 (parked) | GT:59-61 |
| B12 | ProMoE planner / reordered execution (cache -> in flight -> CPU) | 1.09x over LRU, PCIe 4.0 | exact | - | 975 (item 30), 1331 | GT:62, GT:87 |
| B13 | Next-expert predictors (Fate/HOBBIT/SmallThinker/SP-MoE) recall test | local recall 19-29%; ProMoE 1.09x | exact | - | 975 (item 30) | GT:63, GT:337 |
| B14 | Seeds `STRATA_DOMAIN_PROFILES` / `STRATA_CONV_WORKSET` (items 3-4) | +0.6-1.5 hit pts | exact | include/strata/program/expert_seed.hpp:7,13 | 902-905, 1424 | GT:64 |
| B15 | Expert pre-staging during tool runs (KVFlow, item 17) | sim first | exact | - | 945 (item 17), 1508 | GT:66 |
| B16 | `--pcie-frac 0` default on gfx1100 | mean 111.7->118.4 (+6%), after-long 92.5->104.3 (+13%), 7900XT | exact | include/strata/core/calibrate_ratio.hpp:14 | 286, 827 | GT:68, MS:41 |
| B17 | Fiddler token-count threshold for `STRATA_PREFILL_CPU_SHARE` | code reading | exact | include/strata/core/arch_defaults.hpp:30 | 411, 3932 | GT:69 |
| B18 | Expert deferral `STRATA_VERIFY_DEFER_CPU` (KTransformers) + HIP non-finite bisect (item 8) | +33% / up to 1.45x DeepSeek-V3 Xeon; est. +10-25% edit; HIP gives NaN logits | lossy (small) | src/core/conversation_disk_env.def:42 | 909 (item 8), 108, 621 | GT:80, PP:13 |
| B19 | Rank guard `STRATA_VERIFY_DEFER_CPU_TOP` (item 28) | keeps +3-5% edits (plan A4) | lossy (small) | include/strata/core/defer_auto.hpp:91; env.def:44 | 969 (item 28), 2040 | GT:81, PD:53 |
| B20 | Per-expert pool barriers + fused SiLU*mul `STRATA_POOL_FUSE_PHASES` (item 29) | 1-4% | exact | include/strata/kernels/cpu/pool.hpp:382 | 973 (item 29) | GT:82, PP:26 |
| B21 | `STRATA_POOL_PREREAD` (tandem.cpp) | A/B queued | exact | include/strata/core/epoch_bandit.hpp:10 | 842, 853 | GT:83, PD:89 |
| B22 | Athena epoch bandit for pre-read / mixed opt-ins | - | exact | include/strata/core/epoch_bandit.hpp | 965, 2053 (A44) | GT:84 |
| B23 | DISCO learned chain stop (item 15/36, `feat/disco`) | 0-10% combined | exact | include/strata/core/row_miss.hpp:2 | 1010 (item 36), 124 | GT:98 |
| B24 | Drafter entropy (MiMo, Entropy Sentinel) as DISCO feature | - | exact | no | 2054 (A22), 150 | GT:99 |
| B25 | SpecFit / DraftExpert / EcoSpec miss-aware window (new experts per position) | DraftExpert verify-cost 2.35x->1.48x, RTX 4090; est. +3-8% over SPEC_COST | exact | no (SpecFit name) | 1010 (item 36), 150, 3934 | GT:100, MS:55 |
| B26 | Cross-session suffix index (SuffixDecoding) | `--simulate` first; A24 spec 45 | exact | include/strata/spec/suffix_index.hpp:114 | 940 (item 15), 3694 | GT:102 |
| B27 | `--lookup-max-t 12/16` | +1% first test | exact | include/strata/core/lookup_default.hpp:1 | 108, 133 | GT:103, PD:84 |
| B28 | MTP draft inside captured verify graph (item 10) | 1-3% est. | exact | no | 911 (item 10), 221 | GT:104, PP:21 |
| B29 | `--mtp-q4` | queued | exact? | include/strata/core/mtp.hpp:64 | 887, 2625 | GT:105 |
| B30 | ReplaySSM / SpecLA audit of GDN replay (item 20) | 1-3% decode | exact | - | 949 (item 20) | GT:106 |
| B31 | SSD-lite (Saguaro): draft while GPU verifies | CPU already busy | exact | - | 1336 (parked) | GT:107 |
| B32 | Restore-equivalence test (UNSW, item 35/83) | gate for 3/11/16/31/39 | test | src/core/conversation_disk_env.def:333 (`STRATA_CKPT_REREAD`) | 1007 (item 35), 444 (83) | GT:121, CD:27 |
| B33 | `preserve_thinking` end to end (item 31) + PR #5643 restore dropped reasoning | ~8% turn time is same-session re-reads | exact | serve:chat_template.jinja:119 | 984 (item 31), 175 | GT:122, PD:244 |
| B34 | Message-boundary checkpoint `STRATA_CACHE_MESSAGE_BOUNDARY` + `--prompt-cache-tail` (item 11) | first edit 17.24->11.53 s (-33%), author hw | exact | src/core/conversation_disk_env.def:321 | 912 (item 11), 879 | GT:123, PP:38 |
| B35 | Tool-boundary checkpoints + Marconi FLOP/byte eviction (item 16) | Marconi TTFT -71.1% SWE-bench traces | exact | - | 943 (item 16), 179 | GT:124 |
| B36 | RadixAttention multi-state store / Continuum TTL (item 3/71) | 5.4-6.1 GiB for 4 parked convs | exact | - | 368 (71), 602 | GT:125 |
| B37 | Disk tier `--conversation-disk` on main engine (item 39) | 70K resume 7.2 s vs 83 s (FastStrata, 3060) | exact | include/strata/core/conversation_cache.hpp:483 | 1027 (item 39), 601 | GT:126, PD:319 |
| B38 | `env_to_user`, effort at end, "tools first" prompt reordering (item 88) | ~14K tok / 8-11 s per day/folder (comment) | exact | serve:frontend.py:405; serve:runconfig.py:32 | 469 (88 DONE); reordering: 858 | GT:127, CD:31 |
| B39 | Thinking caps `reasoning_budget_by_effort` + grace 512 + `reasoning_close_retry` (item 12) | 15 turns cut cost 2,726 s; p99 ~4-5K tok | lossy | serve:server.py:5130; include/strata/program/think_probe.hpp:21 | 914 (item 12), 1904, 1826 | GT:140-141, PP:40 |
| B40 | Learned thinking stop probe (Thought Calibration, item 21/A19) | up to -60% thinking tok (paper) | lossy | serve:test_think_probe.py | 950 (item 21), 2058 | GT:143 |
| B41 | Rumination 5-gram signal (item 34) | - | log-only then lossy | serve:rumination.py | 993 (item 34) | GT:145 |
| B42 | GDN prefill chunk 64 + 16x16 block inverse (item 18) | 5-11% prefill; V100 chunk-64 attempt 4-6x slower | exact (bits change) | include/strata/prefill/gdn_chunk.hpp:52 | 947 (item 18), 423 (81) | GT:159, CD:17 |
| B43 | BFLA as one shared QSA selection per query tile; `STRATA_SEL_OVERLAP` first (item 19) | 128K: 386.9->1,219.8 tok/s Qwen3.6 (q36 engine); Flash-Next QSA 2,051 cells/query | lossy | src/core/conversation_disk_env.def:1002 | 249, 948 (item 19) | GT:160, PP:64, PD:104 |
| B44 | Q2_0 expdec down kernel at n_ff 640 (`native_down_multi_kernel<42>` = 5.79% GPU) | est. 0.3-0.6 ms short window (~1-2%) | exact | src/kernels/cuda/iq_kernels.cu:2295 | 232, 1574 | GT:174, PP:19 |
| B45 | Gated Residual small-M Mix/Combine kernels M=1-8 (item 32) | 1-5% short | exact | - | 986 (item 32) | GT:175 |
| B46 | UFMG coordinate-descent launch/tile sweep (item 37) | 1-3% | exact | - | 1012 (item 37) | GT:176 |
| B47 | Group A arch switches bisect one at a time | all on: 101.2 mean, 80.3 after-long, 7900XT (worse) | check each | include/strata/core/sm120_series.hpp:7 | 199 | GT:177, MS:44 |
| B48 | BitDecoding WMMA decode attention `STRATA_ATTN_WMMA` | ~2-3%, attention ~3% of decode | exact | include/strata/kernels/qsa_decode_attn_wmma.hpp:2 | 2485, 1726 | GT:179, PD:105 |
| B49 | Revised quality gate (token forcing filter + fixed task set); KLD/top-1 rho=+0.00 with task score | Displacement Is Not Direction, Qwen3.6-A3B | method | tools/ (task_gate.py) | 1672, 2024 | GT:187, GT:191 |
| B50 | `reasoning_tokens` in chat usage + per-turn JSONL log (items 87/73) | gates 11/12/16/31 | n/a | serve:server.py:8399, 5086 | 467 (87 DONE), 142 | GT:195, CD:33 |
| B51 | `pancake.service` with `Conflicts=`, port 11437, own MemoryMax, NOCAP (item 91) | restart 28-65 s | n/a | - | 479 (91), 141 | GT:194, CD:33 |
| B52 | `MADV_SEQUENTIAL`/`WILLNEED` at load / #5561 `STRATA_EXPERTS_SEQ_READAHEAD` | later turns 65-76 s -> 23-28 s, RX 6900 XT 32 GB | exact | no | 1349 (parked) | GT:200, PP:60 |

### 2. projects-derived (forks and upstream PRs)

| # | lead | number + hw | exact/lossy | in our src | in plan (line) | source file:line |
|---|---|---|---|---|---|---|
| B53 | PR #5655 `route_prior` sentinel id leak / OOB on degenerate logits | correctness | exact | src/kernels/cuda/route_prior.cu (guard present) | 3016 (present, do not re-port) | PD:11, PD:291 |
| B54 | #5609 q36 MTP block loads but never drafts; verify in base | correctness | exact | - | 3978 | PD:273, PD:292 |
| B55 | llama.cpp #31751 one-pragma MMQ J=32 register-spill fix on RDNA3 | +36% at 17-32-token batches, gfx1100 | exact | no | 3968 (E#2); 3356 (no spill on hot path) | PD:283, PD:293 |
| B56 | #4625 `STRATA_POOL_SPIN_US=0` really means zero | trivial | exact | include/strata/kernels/cpu/pool.hpp:291 | no | PD:248, PD:294 |
| B57 | #5466 need/free MiB on expert-buffer/staging OOM | diagnostics | n/a | - | 3978 | PD:215, PD:295 |
| B58 | #5518 HIP `alignas(16)` int4 staging compile fix | build | n/a | - | 3978 | PD:223, PD:296 |
| B59 | #4005 (Vop3p) prefill sparse copy events + slot-wait dedup | Windows HIP prompt +36% @4K, RX 6900 XT | exact | no | no | PD:217, PD:300 |
| B60 | #5468 layer-major super-chunk prefill (stage a layer's experts once per N chunks) | q36 only, author | exact | - | 3978 | PD:214, PD:301 |
| B61 | #5595 grouped Q4_K down-expert kernel (q36) | q36 only | exact | - | 3978 | PD:212 |
| B62 | StrataGP 16 packaged patches | +5-15% decode, +42-49% prompt, RTX 5090 | mixed | - | 3973 (E#7) | PD:116, PD:302 |
| B63 | Juud_engine: shorter worker spin after expert batch + skip activation quant for GPU-resident tokens | +9.6-15.6% decode, RTX 4090 | exact | no | no | PD:130, PD:303 |
| B64 | #5475 `STRATA_HC_PERSIST_Q8` int8 persistent HC read | - | exact? | env.def:429 has `STRATA_HC_PERSIST` (not _Q8) | 3978 | PD:218, PD:304 |
| B65 | #5535 int8-WMMA prompt attention on gfx1100 with recipe | attention 6.7% of 42K prompt | exact? | - | 3978 | PD:216, PD:305 |
| B66 | #5471 faster Q4_K/Q5_K decode GEMVs (DS4 path) | - | exact | - | 3978 | PD:235, PD:307 |
| B67 | #3925 q36 load of K-quant/IQ4_XS `ssm_alpha`/`ssm_beta` | pack coverage | exact | - | no | PD:227, PD:308 |
| B68 | #3509 `STRATA_ARENA_REUSE=1` shared arena across restarts | restart 28-65 s on 33 GiB | exact | no | 3978 | PD:247, PD:312 |
| B69 | Maya parallel direct expert reads `MAYA_EXPERT_DIRECT`/`STAGERS` (2-16 O_DIRECT threads) | +26% decode RAM-poor PC | exact | no | ~1021 (item 38 concept) | PD:45, PD:313 |
| B70 | #5516 in-place TUNE + conversation-cache routing by traffic shape | - | exact | - | 3978 | PD:244, PD:315 |
| B71 | Maya weight-based miss skip `MAYA_GLM_MISS_SKIP_W` | +7-13% decode at KL 0.037-0.052, GLM | lossy | no | 3974 (E#8) | PD:45, PD:316 |
| B72 | #5566 opt-in cold store for expert arena | memory elasticity | exact | - | no | PD:239, PD:317 |
| B73 | #5623 immutable host KV prefixes shared between sessions | 5.4-6.1 GiB parked | exact | - | 3978 | PD:246, PD:318 |
| B74 | basalt sleep-in-place `--idle-sleep` (64167ba73, ~815 lines) | - | exact | no | 114, 92, 603 | PD:38, PD:320 |
| B75 | Trellis "S" 2-bit quant (#3384, alesha-pro) evaluate offline | - | lossy | - | 3978 | PD:173, PD:329 |
| B76 | Maya leftovers: `--calibrate` startup sweep, drafter auto-keep | - | exact | - | 148, 304 (calibrate) | PD:45, PD:334 |
| B77 | Maya prompt memo | ~6% of follow-up TTFT | exact | - | no | PD:45 |
| B78 | #5593 `STRATA_PROMPT_DIRECT=1` without a RAM budget | - | exact | src/core/conversation_disk_env.def:639 | no | PD:254, PD:322 |
| B79 | #4777 per-request `route_prior`; #3971 keep thinking past cut; #5628 stream tool args; #5512 thinking skip | - | mixed | - | 3978 | PD:249-252, PD:322 |
| B80 | #5637/#5437 inhibit sleep during inference | - | n/a | serve: keepawake.py | 260 (keepawake test), not the PR | PD:256 |
| B81 | Private_Context_COW copy-on-write context (#3865) | parked RAM 5.4-6.1 GiB | exact | - | no | PD:166 |
| B82 | StrataQ38FN agent-switch re-read patch | 25 s -> 1.5 s re-read, author (4x3090/5090) | exact | - | 3978 | PD:123, PD:125 |
| B83 | Upstream v0.2.13 / engine 0.1.59 (shape-engine for Qwen4, signed geometry manifest) affects rebase | base 146df3c4 one release behind | n/a | - | no (3144 is unrelated EXL3 text) | PD:199-204 |
| B84 | #5545 GLM-5.3-Flash engine path (`strata-glm`) | - | n/a | - | 3194 (GLM future goal) | PD:152, PD:262 |

### 3. pain-points (engine + server/ops), not already above

| # | lead | number + hw | exact/lossy | in our src | in plan (line) | source file:line |
|---|---|---|---|---|---|---|
| B85 | Edit-window budget: 37.4 ms = 15.8 GPU + 17.4 CPU experts + 2.5 draft; 357 CPU experts/window at ~27 of 33 GB/s | 7900XT fast-hip | fact | - | 869 | PP:9 |
| B86 | `combine_cpu_k10_vec4` spin = 38.3% GPU kernel time; `wait_flag_copy_i32` 4.1%; 395 kernels/token | rocprofv3, 7900XT | fact | src (combine_cpu) | 199, 2674 (Phase 18) | PP:9, CD:7 |
| B87 | Profile on edit window + after long prompt (item 1 extension) | - | diag | - | 199 | PP:83, GT:173 |
| B88 | Fixed per-turn overhead ~0.5-0.9 s breakdown (`t_parse/t_render/t_encode`) | Python est. 60-300 ms | diag | - | 142 (turn log) partial | PP:72, PP:82, CD:33 |
| B89 | Hugepages 16,920 x 2 MiB (owner) | unmeasured | exact | src/core/pinned.cu:110 (THP) | 451 (85), 1081 | PP:60 |
| B90 | DDR4-3600 / FCLK 1800 + STREAM read test | up to 1.1-1.19x CPU part (est.) | exact | - | 1143 | PP:70, MS:47 |
| B91 | Deterministic test mode (fixed profile, adapt off) + load-gated A/B distributions | ~20% run-to-run variance | method | tools/pancake/abtest.py | 1667, 1679 | PP:90 |
| B92 | `strata-coder` sub-agent model/context mismatch (163,840 vs 131,072) | - | config | - | 479 (91) | PP:58 |

### 4. code-deep-research (items 74-95b; all in plan lines 389-500)

| # | lead | number + hw | exact/lossy | in our src | in plan (line) | source file:line |
|---|---|---|---|---|---|---|
| B93 | 74 MTP draft eager launch (`STRATA_MTP_EAGER`), min_p gate on device | 0.3-0.5% est. | exact | src/core/conversation_disk_env.def:494 | 389 (74 CODED) | CD:9 |
| B94 | 75 persistent adapt worker (`STRATA_ADAPT_PERSIST`) | 0.1-0.3% est. | exact | env.def:292 | 394 (75 CODED) | CD:11 |
| B95 | 76 `hipDeviceScheduleSpin` A/B (`STRATA_DEVICE_SPIN`) | 0.2-0.6% est.; ~3% polling cost on CUDA | exact | env.def:354 | 397 (76 CODED) | CD:11 |
| B96 | Deferral turns off `ROUTE_PUBLISH` (verify.cpp:2355): +48 launches/window bias in 8/28 A/B | - | note | include/strata/core/defer_auto.hpp:16 | 402 | CD:11, CD:86 |
| B97 | 78 staged prefill path for <4096-token agent deltas (CPU_SHARE 0.6 + MAX 4096; per-layer sync) | 4K 1,757 vs 32K 2,580 tok/s, 7900XT; est. 0-15% of re-reads | exact | include/strata/prefill/prefill.hpp:290 | 415 (78), 3707 (spec 46), 3932 | CD:15 |
| B98 | 0xluk `STRATA_PF_AHEAD` + `STRATA_REFILL_ASYNC` on the staged path | agent prefill +35%, TTFT 4.97->3.67 s, 2x3060 PCIe3 | exact | no | 3966 (E#1) | (plan-only cross-ref to CD:15) |
| B99 | 79 PLE gather at request admission, multi-threaded decode | 3060: 10.6% GPU timeline at 42K; est. 3-10% cold 16-32K | exact | - | 416 (79) | CD:17 |
| B100 | 80 hipBLASLt descriptor cache + tuning table beyond T=8192 | est. 7-20 ms host/chunk | exact | - | 420 (80) | CD:17 |
| B101 | 81 GDN scan 48 WG on 84 CUs: split value columns | - | bits change | src/prefill/kernels.cu:1809 | 422 (81) | CD:17 |
| B102 | 82 `STRATA_PREFILL_TIMING=1` + `STRATA_TRACE=1` run on 7900XT | never run | diag | include/strata/prefill/timer_marks.hpp:1 | 425 (82) | CD:17, PP:84 |
| B103 | 38 Linux O_DIRECT in `load_experts_gguf` (16 readers, FADV_DONTNEED stop-gap) | 31-41 s -> est. 12-15 s; likely oomd cause | exact | no: src/core/expert_source.cpp:6155 still `(void) unbuffered;` | 1021 (38), 433 (correction) | CD:21-25 |
| B104 | 86 deterministic conversation-cache RAM budget + PSI-paused readers | cache 4,096-6,144 MiB today | exact | env.def:964 (`STRATA_PSI_GUARD`) | 455 (86) | CD:27 |
| B105 | 85 THP collapse in background, pipelined registration | Windows/5070 -10% decode | exact | src/core/pinned.cu:110 | 451 (85) | CD:27 |
| B106 | 89 grammar on Anthropic/Responses paths | real bug | exact | serve:server.py:10057, 10122 (fixed) | 474 (89 FIXED) | CD:37 |
| B107 | 90 `pin_sibling` threshold (`STRATA_PIN_PARK_TOKENS`) before Qwen auto-pin | full re-read of 19-41K parent | exact | src/program/generate.cpp:17642; env.def:582 | 476 (90 DONE), 1673 | CD:33 |
| B108 | 92/93/94/95/95b build matrix, tests dir, flags.def registry, main() split, stubs | 872 STRATA_* names, 18,361-line main | n/a | tools/pancake/build_matrix.sh; include/strata/prefill/moe_fused_iq.hpp:47 | 488-500 | CD:59-72 |
| B109 | dan-99 list: drop trailing `wait_parked` (pool.cpp:1373) | est. 0.2-0.5 ms/window | exact | include/strata/kernels/cpu/pool.hpp:308 | 402 | CD:86 |
| B110 | dan-99 list: `row_multi<1>` FMA chain split, per-window getenv, serial activation quant, scalar exp SwiGLU | 0-3% each | exact / not bit-identical (FMA) | - | 399-404 (77) | CD:86 |
| B111 | dan-99 list: MWAITX instead of pause-spin | 0-3% est. | exact | src/core/conversation_disk_env.def:615 | 404 | CD:86 |
| B112 | Fusion targets: swiglu+quant as `expdec_gu` epilogue (~49 launches/window), router GEMV + `route_publish_res` | - | exact | - | 405 | CD:86 |
| B113 | 84 seeds inert under KMISS: virtual miss bits | placement | exact | - | 450 (84) | CD:42 |
| B114 | 53 ForesightSwap stuck slot on failed copy: generation stamp | bounded | exact | - | 463 | CD:43 |
| B115 | Loop-guard `collapse` rewrites history -> deep re-read each turn while active | perf risk | exact | serve:loop_guard.py | no | CD:49 |
| B116 | Session files keep only deepest checkpoint (root lost after restore) | matters for 39 | exact | include/strata/core/conversation_file.hpp:189 | 464, 455 (39 sketch keeps root) | CD:45 |

### 5. moe-speedup (window model + literature)

| # | lead | number + hw | exact/lossy | in our src | in plan (line) | source file:line |
|---|---|---|---|---|---|---|
| B117 | `STRATA_SPEC_COST=1` for gfx1100 (EVICT rule) | mean flat 112.0, after-long 92.5->98.3 (+6%), 7900XT | exact | include/strata/core/mtp.hpp:146 | 220, 828, 2483 (flat) | MS:42 |
| B118 | Re-warm / protect decode tier after prefill (prefill-routing seeded bulk swap) | after-long 92.5 vs 111.7; est. +1-4 hit pts | exact | no | ~897 (item 2, different mechanism) | MS:46, MSt:139, MSx:168 |
| B119 | One pinned pool worker per physical core, spin waits (`--pool-affinity`) | <5%; 12 workers worse (99 vs 112) | exact | src/dsv41/ds4_os.hpp:581 (DS4 only) | 2692, 262 | MS:47, MSc:158 |
| B120 | Reuse-aware admission with H2D DMA scheduled into CPU-idle phases (attention/GDN) | est. +10-18% over Tier 1; SeqMoE +8.5 pts vs LRU | exact | no | 1162 (closed variant); DMA scheduling no | MS:54 |
| B121 | Exact tail-row truncation mid-window (novel) | 23% of misses only for last row, 47% last two (local) | exact | - | 906 (item 5), 230 | MS:56 |
| B122 | CPU pool bandwidth push: 2 fused phases/layer across all missed experts, row-chunk work split, touch next expert early | 33 -> 38-42 GB/s est., +5-12% | exact | no (row-chunk) | partial 973 (item 29) | MS:57, MSc:185 |
| B123 | Row-interleaved R4/R8 repack of CPU arena (ik_llama `_R4`) | 5-15% CPU part (qualitative) | exact | no | no | MSc:184 |
| B124 | Fused T-row GDN verify kernel with lazy commit (FlashInfer gated_delta_rule_mtp / ReplaySSM) | write only final state + per-token k/v/g; small unless T full-state writes today | exact (sequential form) | - | partial 949 (item 20) | MS:58, MSr:99 |
| B125 | Prefill: bigger chunks / decode tier resident during prefill (SLP, MoE-Gen) | est. 1.3-1.8x prefill at 16-32K; ~75 ms saved per resident GB per pass | exact | - | partial 428 (ring 192 vs 512-1024) | MS:59, MSt:78-81 |
| B126 | SeqMoE learned multi-step forecaster + probabilistic-Belady eviction | 96.97% vs 88.50% LRU at 45% residency, RTX 4090/5090 | exact | no | ~1332 (parked family) | MS:65, MSx:170 |
| B127 | Prompt-family expert profiles (FineMoE / MoE-Sieve) | FineMoE -47% latency, +39% hit (6-GPU) | exact | expert_seed.hpp:7 | 854 (items 3-4) | MS:66 |
| B128 | DFlash2 workload-gated drafter only | +4% short, -9% after-long, 7900XT | exact | src/core/verify.cpp:77 | 1172 (closed) | MS:67 |
| B129 | In-graph CPU flag polling with two HW queues | KT CUDA graph up to 1.23x | exact | src/core/mtp.cpp:514 (priority queues) | 1171 (parked) | MS:68, MSr:119 |
| B130 | Per-window dynamic CPU/PCIe split (HybriMoE) | 1.70x vs KT (author); ~0-5% on PCIe 3.0 | exact | - | 1299 (closed) | MS:69 |
| B131 | Deterministic cache-aware routing (Prefetch-Steered Routing) | PSR 2.2-2.3x vs exact LRU, OLMoE RTX 4090 | lossy | no | no | MS:75, MSx:173 |
| B132 | Lower-bit CPU copies of cold experts (HOBBIT; F4 shadow pack) | ~25-30% fewer CPU bytes; F4: -18.2% bytes, +17.6 GB RAM | lossy | - | 1274, 1822 (F4) | MS:78 |
| B133 | Layer-dependent expert skip (HookMoE, ik `-ser`) | HookMoE 1.42x prefill, -2.5% quality | lossy | expert_source.hpp:670 (rank tail skip only) | ~1038 (tail skip); layer-dependent no | MS:79 |
| B134 | Router-probability-mass EWMA / LFRU (freq-age, hub experts) scoring | +1-3 hit pts (est.) | exact | no | no | MSx:169, MSe:157 |
| B135 | Per-layer marginal-gain budget allocation | +0.5-2 pts | exact | - | 722 (item 7, tested no gain) | MSx:170 |
| B136 | Quest page-level / partial top-k / fused select+attn for QSA after long prompt | gap 2.18 ms/token at 12K; est. recover most of 20% | exact (b,c,d) / lossy (a) | - | partial 249, 823 (SELECT_LONG) | MSt:135-141 |
| B137 | Streamed KV (>32K) at 4-bit, int8 resident | halves PCIe bytes at 64-131K (est.) | lossy | include/strata/kernels/kv_q4.hpp:9 (`--kv-q4`) | 2315 (poly kv-q4); main-engine streamed no | MSt:98 |
| B138 | Per-layer persistent super-kernels + "productive spin" MALL/L2 prefetch of next layer dense weights | est. 1-2.4 ms (4-10%) only after CPU wait hidden; Alpindale 905->1000 tok/s 5090 | exact | - | ~1171 (MALL warm, parked) | MSr:48-49 |
| B139 | Non-temporal (slc/nt) GPU loads for streaming weights to keep MALL for GDN state/KV | low single-digit; evict-first -8% on 5090 | exact | no | no (1298 is CPU nt, closed) | MSr:74 |
| B140 | Remove graph-internal memset/memcpy nodes; linear-chain graph to avoid barrier-AND packets | unsourced; ceiling ~2.4 ms/window (1000 nodes x 2.4 us) | exact | - | no | MSr:24 |
| B141 | Batch per-layer contiguous expert copies on SDMA for prefill | low single digits | exact | verify.cpp:1294 (logs HSA_ENABLE_SDMA) | no | MSt:42 |
| B142 | Software prefetch in Q2_0 `row_multi` (item 40) | IQ prefetch -1.3% ms/round Zen 3 | exact | - | 1044 (item 40) | MSc:186, CD:86 |

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

Prefix `UA/` = `pancake-docs/research/engine-speed/unported-academic/`. Plan = `MOD_PLAN.md` on docs/consolidate-plan (3984 lines). "in our src" = `git grep` over `src include` (serve/tools noted where the code lives there). Dropped as off-target for a 1-GPU consumer box: Harvest (NVLink peer GPUs), Pie (GH200 pooling), Continuum queueing term (multi-user), CAR-MoE (router retraining), OS-Pruner and ALP (fine-tuning), Danger-of-Overthinking candidate selection and short-m@k (multi-sample), MonoMoE FP8/H200 kernel, q-sqs (64-GPU model), KVFlow multi-agent scheduling, Princeton option-critic (snippet only).

### 1. Expert offload, cache, routing

| lead | number + hw | exact/lossy | in our src | in plan (line) | source file:line |
|---|---|---|---|---|---|
| A1 cache-conditional routing (bonus on resident-expert logits) | miss 35->16% Qwen1.5, 2x vs LRU (Snapdragon, CPU llama.cpp); ours +30-50% before deferral | lossy (ppl +0.1-3%; "2.9%" is Mixtral ppl, not token diff) | src/core/verify.cpp:296 | 1859, 2759 | UA/round3/notes/batch1.md:15; UA/round4/notes/V1.md:20-21 |
| Cache-Prior top-J guarantee (top experts load regardless of cache) as A1 guard | paper Sec 3.3; lambda 0.8 over-uses cache | lossy guard | no ("top-J"); F13 guards on branch only | no (F13 guard rows only) | UA/round3/notes/batch1.md:21,26 |
| AcceptMoE residency-aware utility set (root anchor, effective-rank size, reroute non-resident) | 2.06x vs standard SD, H2D -73.6-77.1%, RTX 5090 48 slots; acc -0.27 pt mean | lossy | no (only depth prior, include/strata/kernels/route_prior.hpp:12) | partial: 1601, 1635 note mechanism; no row | UA/round3/notes/batch1.md:52-55,60,71 |
| A2 depth-graded prior (our geometric bonus) | "+5-15%" unsourced | lossy | include/strata/kernels/route_prior.hpp:12 | 1860, 1601, 2760 | UA/round3/notes/batch1.md:46 |
| HOBBIT cumulative gate-score rule (T1 0.6 / T2 0.9) as skip/defer rule, no low-bit copy | Mixtral 67/30/3% split; dyn loading 1.19-1.57x (4090, Orin) | lossy | no (defer uses per-weight `_BELOW`, src/core/verify.cpp:985) | partial: 1696, 2821 (wording); no test row | UA/round3/notes/batch1.md:89,98; UA/round2/notes/offload_round2.md:99 |
| A4 rank guard `_TOP/_BELOW/_FOLD` | vs deferral: edit -2.6 to -4.2%, short ~0 (n=2) -> HOLD | lossy-small | src/core/verify.cpp:985; include/strata/core/defer_rank.hpp:48 | 1752-1753, 1862 | UA/round3/notes/batch1.md:74-86 |
| A27 sub-2-bit copy on CPU miss | ~25% bytes/miss (IQ2_XS->IQ1_M, from memory); IQ1_M expert error 0.55-0.91 | lossy | no (`STRATA_SHADOW_EXPERTS` gone) | 1936, 2742, 2870 | UA/notes/expert_offload_papers.md:13,39; UA/round2/notes/offload_round2.md:66 |
| SliceMoE Matryoshka slices (low-bit = shifted high-bit, no duplicate bytes) | 2-bit ppl 10.85 vs 4-bit 8.09 (Qwen1.5); up to 1.81x on simulated mobile SoC | lossy | no | no | UA/round2/notes/offload_round2.md:79-89 |
| MoBiLE big/little (half experts, confidence gate 0.7) | 1.72x/1.60x RTX 4080; OLMoE GSM8K -6.7 pt | lossy | no | 1274 (parked re-selection) | UA/notes/expert_offload_papers.md:29-35 |
| Routing dump with gate weights (unblocks A4/A27/A32/A46 replays) | existing 219-window trace has 0 weights | n/a | src/program/generate.cpp:1150 (`--dump-routing` writes ids+weights) | 2870 | UA/round3/notes/batch1.md:93,159 |
| A7 chunked promotions + drop queued speculative + reordered exec | ProMoE 1.09x (max 1.17x) llama.cpp decode vs LRU, 4090 PCIe 4.0; our sim +4.6% | exact | include/strata/program/promo_chunk.hpp:2 | 1800, 1865, 1695, 2761 | UA/round3/notes/batch1.md:124-151; UA/round4/notes/V1.md:28-31 |
| ProMoE per-layer MLP predictor + stride prefetch | 84.7% acc; stride +1 layer at ~5% acc cost | exact | router lookahead exists: include/strata/core/expert_source.hpp:544 | generic only (705-796) | UA/round3/notes/batch1.md:131 |
| HOBBIT next-1-3-layer prefetch from gating input | 96% top-1 next layer, ~90% at 2-3 | exact | yes: include/strata/core/expert_source.hpp:341,544 | generic (709) | UA/notes/expert_offload_papers.md:23 |
| HOBBIT multi-precision cache score (LRU+LFU+LHU+layer distance) | no isolated number | exact | no ("LHU") | no | UA/notes/expert_offload_papers.md:23 |
| A8 online swap budget, blocking tier | async -9.6%; est 0-5% over best fixed | exact | include/strata/program/swap_budget.hpp:2 | 1866, 2762, 2874 | UA/round3/notes/batch2.md:7-30 |
| A10 working-set prefetch (MoE-Infinity) | -19 to -32% on our 1 replay trace | rounding | include/strata/core/expert_source.hpp:813 (working set) | 1868, 2764, 2876 | UA/round3/notes/batch2.md:58-78 |
| A44 epoch bandit | Athena +5.0% vs MAB (ChampSim CPU, 100 traces); stateless Athena < MAB | policy (via lossy arms) | include/strata/core/epoch_bandit.hpp:103 | 1990, 2754, 2866 | UA/round4/notes/V1.md:16,34 |
| Athena reward correction (subtract workload-driven term) + state features for bandit | state +1.4/+1.7/+0.8/+0.1, reward +1.0 pt (CPU sim) | policy | no | no (2754 has only +5%) | UA/round4/notes/V1.md:34; UA/round2/notes/offload_round2.md:29,47 |
| A45 start-up calibration probe | expected share 0 on this box (raw ~0.28 < 0.35) | exact | src/program/generate.cpp:8366 | 1991, 2900 | UA/round3/notes/batch7.md:7-32 |
| DALI per-layer per-step greedy CPU/GPU assignment (vs static share) | assignment alone +23% vs HybriMoE; decode 1.32-3.97x (EPYC 7532 + 3090, PCIe 4.0) | exact | no | partial: 1697 (correction note, no row) | UA/round3/notes/batch7.md:14,20 |
| A46 learned cache policy | Belady 94.7 vs LRU 89.5 vs engine 79.6%; best online +3.2 pt but 2-3x promotions | exact | no (tools/pancake/cache_learn.py) | 1992, 2776, 2901 | UA/round3/notes/batch7.md:34-58 |
| A29 expert-byte entropy coding | bound 11.7-12.0% Q2_0, zstd 8.0%; IQ ~0% | exact | no (tools/pancake/expert_entropy.py) | 1938, 1789, 2770, 2887 | UA/round3/notes/batch4.md:153-187 |
| A28 ISTA-DASLab GSQ-RCO Coder | SWE-bench 95.7% retained (card); 58.4 GB Qwen / 111 GB DeepSeek | lossy (50% expert pruning + quant) | no (no GSQ/LUT loader) | 1937, 2873 | UA/ROUND2_REPORT.md:14,35-36; UA/triage/check_T2.md:11-12 |
| ExpertCache task-specific expert preload (RL) | Qwen3-235B, BigCodeBench (snippet) | exact (placement) | no | no | UA/round2/notes/offload_round2.md:38 |

### 2. Sessions, prefix and KV cache

| lead | number + hw | exact/lossy | in our src | in plan (line) | source file:line |
|---|---|---|---|---|---|
| A9 pre-stage experts during tool gaps | bound 2-4% of wall | exact | src/program/generate.cpp:14928 (`STRATA_IDLE_ADAPT`) | 1867, 2763, 2875 | UA/round3/notes/batch2.md:34-54; UA/STATUS.md:67 |
| TokenCake: offload idle KV during function calls, predictive upload | up to 47.06% e2e latency, memory-constrained (abstract) | exact | no | no | UA/round3/notes/batch3.md:54 |
| A11 tool-boundary checkpoints + FLOP-aware eviction (Marconi) | eviction-only +19.0-219.7% hit vs LRU (7B hybrid, 4xA100); ours 0.5-0.8% | exact | include/strata/program/conv_cache.hpp:99 (`--prompt-cache-spread`) | 1886, 1700, 2765, 2877 | UA/round3/notes/batch2.md:82-105; UA/round4/notes/V4.md:173-178 |
| A12 pinned system+tools prefix | -1.3% modelled on Qwen | exact | include/strata/core/conversation_cache.hpp:533 | 1887, 2909 | UA/round3/notes/batch2.md:109-131 |
| Pinned-entry eviction fix (prereq for A12) | pinned parked entries never evicted | exact | no fix | 2909 | UA/round3/notes/batch2.md:119,127 |
| A15 park TTL for sub-agents | Continuum gain is queueing term, ~0 single user | exact | include/strata/core/conversation_cache.hpp:494 | 1890, 1699, 2879 | UA/round3/notes/batch3.md:43-64 |
| Rank-3 eviction bug (evicts whole held set on byte shortfall; take smallest covering entry) | 6 of 7 cold re-reads were over-budget skips | exact | bug at conversation_cache.hpp:754 (per note) | 1568, 2879, 2932 | UA/round3/notes/batch3.md:59,64 |
| MORI: idleness from tool durations, cross-tier KV migration | 20-71% thpt, 18-43% TTFT (search summary, unverified) | exact | no | attribution only (1699, 1890) | UA/round3/notes/batch3.md:55 |
| A14 restore check: teacher-forced / second recompute control arm | paper 34/36 -> 36/36; CPU reload TTFT -46-64% | exact | tools (restore_check.py --control) | 1889, 1716, 2766, 2878 | UA/round3/notes/batch3.md:18-41 |

### 3. Speculative decoding

| lead | number + hw | exact/lossy | in our src | in plan (line) | source file:line |
|---|---|---|---|---|---|
| A21 exact sampled speculation + `STRATA_SPEC_MIN_TEMP=0.7` | +5.5% B70 at T1.0; +13.4% 5090 DT 0.85; loses below T~0.7 | exact (distribution) | include/strata/core/spec_prob.hpp:64 | 1919, 1786, 2086, 2884 | UA/round3/notes/batch4.md:17-53 |
| A22 entropy chain stop (AdaEDL) | vs top-1 stop +0.2-9.3% (mean 3.2%); vs best static +4.4-10.5% (Llama2-7B, A100, T0.7) | exact | no (include/strata/spec/disco.hpp:30: entropy not available) | 1920, 2755, 2819 | UA/round2/notes/decoding_round2.md:20-56; UA/round4/notes/V1.md:19 |
| Top-1 confidence stop (Max-Confidence baseline) | AdaEDL's main baseline; MTP returns top-1 already | exact | include/strata/spec/window_rule.hpp:25 (`--spec-min-p`); src/spec/disco.cpp:106 (`SPEC_CUM`) | 2819, 1651 | UA/round2/notes/decoding_round2.md:43 |
| DISCO learned stop | 10.3% vs best static, 31.4% vs heuristic (4-dataset avg, A100, greedy) | exact | src/spec/disco.cpp:107 | 1920, 1651 | UA/notes/agent_thinking_papers.md:47; UA/round4/notes/V1.md:17 |
| SVIP entropy-bound draft length | +17% MT-Bench 8K, +22% QwQ (snippet) | exact | no | no | UA/round2/notes/decoding_round2.md:38 |
| Drafter top-k + entropy on device (extend record 4) | build only if entropy beats top-1 on replay | exact | no | partial (2755 asks for trace) | UA/triage/TRACE_EXTENSION_DESIGN.md:11,21 |
| Record 6 target p(d) / Leviathan EAR min(1,p/q) (A56) | metric | exact | include/strata/core/window_trace.hpp:29,125 | 2030, 2867, 2914 | UA/triage/TRACE_EXTENSION_DESIGN.md:18-26; UA/round2/notes/decoding_round2.md:72-79 |
| A24 cross-session suffix index | replay +2.0% accepted, -2.4% rounds (sub-agents +6.0% / -7.5%); paper 5.3x SQL, 2.5x SWE | exact | src/spec/suffix_index.cpp:279 | 1922, 1788, 2768, 2885 | UA/round3/notes/batch4.md:58-87; UA/STATUS.md:65 |
| A25 position-decay check (draft_report.py) | 0-5% est. | exact | tools only | 1923, 2769 | UA/round3/notes/batch4.md:91-117 |
| A26 draft-source selector | none beats always-lookup; per-window longest match +2.0% | exact | DraftPolicy (draft_policy.hpp) | 1924, 2886 | UA/STATUS.md:66; UA/round3/notes/batch4.md:121 |

### 4. Prefill, kernels, attention

| lead | number + hw | exact/lossy | in our src | in plan (line) | source file:line |
|---|---|---|---|---|---|
| A30 CPU share in prefill: Fiddler latency gate, env A/B on gfx11 | Fiddler 1.26x/1.30x (Mixtral FP16, RTX 6000); share off on gfx11; spec 46 -5 to -30% | exact (KL ~0.004) | src/prefill/prefill.cpp:288; include/strata/prefill/share_rules.hpp:22 | 1950, 2771, 2888, 3840 | UA/round3/notes/batch5.md:5-33 |
| A31 GDN chunk 64 / blockwise inverse | no paper speedup; GDN 11.9% of prefill (RTX 3060) | lossy (FP16-class) | src/prefill/gdn_chunk.cu:29 | 1951, 2772, 2889 | UA/round3/notes/batch5.md:37-64; UA/round4/notes/V3.md:15 |
| A32 stream experts for large prefill ubatch | OSDI'26 1,200 / 1,800 tok/s (2x 5090, PCIe 5.0); 64-tok ubatch touches ~87% experts (uniform bound) | scheduling; "exact" unsupported | no | 1952, 2871 | UA/round2/notes/decoding_round2.md:88-115; UA/round4/notes/V4.md:50-55 |
| SeqMoE: chunked prefill re-transfers experts per chunk (size chunks to avoid it) | snippet, no number | exact | no | no | UA/notes/decoding_measurement_papers.md:42 |
| A33 small-M fusion | +0.6% [-1.6, +2.8] HOLD | exact if q8_1 order matches | include/strata/core/switch_report.hpp:35 | 1964, 2773, 2890 | UA/round3/notes/batch5.md:68-91 |
| A34 coordinate-descent launch/tile sweep | 5.5-8.5% (A100 thread blocks); Kernel Tuner: tuning ~10x on AMD vs 2x NV (snippet) | lossy unless bit-exact | no (tools sweep.py) | 1965, 2872 | UA/round2/notes/decoding_round2.md:120-143; UA/triage/check_T2.md:13 |
| A35 Infinity Cache warm during combine wait | PRESERVE 1.61x on Ascend 910B L2 (192 MB) | exact | include/strata/core/switch_report.hpp:34 (`STRATA_IC_WARM_MB`) | 1966, 2774, 2891 | UA/round3/notes/batch5.md:95-116; UA/round4/notes/V4.md:112 |
| A36 hit-path ceiling | +1.99% if all VRAM-hit work vanished | n/a (paper FP8 lossy) | no | 1967, 2743, 2775, 2892 | UA/round3/notes/batch5.md:120-146 |
| A48 Q2_0 vs 2-bit GEMV bench | 1.23x 1 thread, 1.03x 8 threads (DDR4-bound) | exact | tools (q2_rows_bench) | 2001, 2902 | UA/STATUS.md:51,70 |
| A49 attention share vs context | decode flat with context within noise | measurement | n/a | 2010, 2779 | UA/STATUS.md:71 |
| A50 verify-window attention, prefix K/V read once | F11 200K 48.6 -> 88.9 tok/s (5080); ~0 short | exact | no (branch work/f11-attn-window) | 2011, 2777 | UA/round3/notes/batch7.md:87-111 |
| A51 Quest/Twilight sparse decode attention | Quest 7.03x attn, 2.23x e2e (4-bit); Twilight 3.9x vs FlashInfer | lossy | no | 2012, 2778 | UA/round4/notes/V3.md:12-13 |

### 5. Agent loop and thinking (client/server side)

| lead | number + hw | exact/lossy | in our src | in plan (line) | source file:line |
|---|---|---|---|---|---|
| A19 ThinkBrake margin stop: log(p_top/p_`</think>`) <= 0.25 at sentence end | -25.1% thinking tokens BFCL (Qwen3-4B-Thinking, v5); <= ~5.5% turn time | lossy | no (think_probe is the probe method, src/program/generate.cpp:7715) | 2756, 2807, 2919 | UA/round2/notes/agent_round2.md:13-29; UA/round4/notes/V2.md:11,35 |
| Per-token top-2 margin in turn log (unblocks A19 replay) | turn log has no logprobs | exact (log) | partial: serve/server.py:7940 `thinking_entropy` summary only | 2919, 2931 | UA/STATUS.md:23 |
| A19 Thought Calibration probe | 60% ID / 20% OOD (math/science) | lossy | src/program/generate.cpp:7715 | 1906, 2744 | UA/notes/agent_thinking_papers.md:14-38 |
| A17 per-effort caps / graceful close | up to 60% drop on late updates; s1 cap is a quality tool | lossy | serve (presets) | 1904, 2848, 2880 | UA/round3/notes/batch3.md:71-91 |
| A18 effort per task (time-to-success) | "-20-40%" unsourced | lossy | serve (turn log effort) | 1905, 2846, 2881 | UA/round3/notes/batch3.md:98-117 |
| A20 rumination + entropy close | r=0.91 one model; host logprobs 1.84 ms/token | lossy | serve/server.py:7940 | 1907, 2849, 2883 | UA/round3/notes/batch3.md:124-140 |
| A40 decode during tools (AsyncFC futures) | 1.40x SWE-bench Lite vs sequential (hosted); our bound 5.4% of span | lossy | no (client) | 1981, 2758, 2896 | UA/notes/agent_thinking_papers.md:68-88; UA/STATUS.md:69 |
| Speculative read-only tool execution (Atomix, PASTE, speculative actions) | PASTE -48.5% task time (unverified) | lossy/side effects | no | no | UA/round2/notes/agent_round2.md:56,61 |
| Agent-run early termination (EarlyEval, EET) | EarlyEval stops 35% runs, -26% steps, -30% tokens; EET -19 to -55% cost, <=0.2 pt (unverified) | lossy | no | no | UA/round2/notes/agent_round2.md:22 |
| A41 ECLoop hard hold (read-before-edit, test-before-done) | +4.8 to +11.8 pp SWE-bench Verified (hosted); 33/339 edits had no prior read | lossy | hook written, not registered (serve) | 1982, 2757, 2897 | UA/notes/agent_thinking_papers.md:90-111; UA/STATUS.md:45,73 |
| A38 behaviour counters | pilot 11/21 edit sessions no test after last edit | exact (log) | serve/agent_loop.py:11 | 1979, 2790, 2894 | UA/round3/notes/batch6.md:51-77 |
| A39 async compaction (Slipstream) | -11.3 to -39.7% vs sync (H100, 9B-36B) | lossy | no | 1980, 2791, 2895 | UA/round3/notes/batch6.md:81-111 |
| A42 externalize big tool output | RLM vs CC+offload 0/+8.7/+16.7% (cross-model); read+bash = 37% of output chars | lossy | serve/agent_loop.py:20 | 1983, 2792, 2898 | UA/round3/notes/batch6.md:115-149 |
| A43 oldString check | 0 of 339 edits failed | lossy (constraint) / exact (log) | serve/agent_loop.py:24 | 1984, 2793, 2899 | UA/round3/notes/batch6.md:153-178 |

### 6. Measurement and gates

| lead | number + hw | exact/lossy | in our src | in plan (line) | source file:line |
|---|---|---|---|---|---|
| A6 overlap time model | 3.4 pt on fitted arms, 8.1 pt held-out (rank guard) | n/a | tools (time_ceiling.py) | 1864, 1772, 2908 | UA/round3/notes/batch1.md:101-122; UA/STATUS.md:68 |
| A54 KLD/top-1 rule narrowed | silent-zone rho 0.00 Qwen / -0.24 Devstral, CI ~+-0.5, zone chosen post hoc | n/a | tools | 2028, 2904 | UA/round3/notes/batch8.md:33-61 |
| A55 task-gate power | 5-pt margin needs ~492 runs/arm, 10-pt ~123; current 18 tasks x 5 reps | n/a | tools (task_gate.py) | 2029, 2905 (no power figure) | UA/round3/notes/batch8.md:74 |
| BFCL multi-turn 30-entry subset adapter | not built | n/a | no | 335, 2029 | UA/round3/notes/batch8.md:78,83 |
| A57 time-to-success AUC, pinned horizon | horizon 1800 s pinned | n/a | tools | 2031, 2921 | UA/round3/notes/batch8.md:91-109 |

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
- A28: GSQ/RCO papers contain neither the 95.7% nor "LUT types". 256/512 is 50% expert pruning. The DeepSeek variant is 111 GB. It is blocked on disk (D has 26 GB free) (UA/triage/check_T2.md:11-12,17; UA/ROUND2_REPORT.md:55).
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

Repo: `pancake-strata` @ `docs/consolidate-plan` (read-only). Source paths are relative to `pancake-docs/research/hardware-platform/`.
Short names: `LR/` = `llm-runtimes-2026/`, `DP/` = `data-path/`, `CP/` = `consumer-platforms/`, `rep` = `report.md`, `n/` = `notes/`.
"In plan" means a line in `MOD_PLAN.md`. "In src" means a hit from `git grep` over `src include`. Hits elsewhere (serve, tools) are marked.
Box: RX 7900 XT (gfx1100, 20 GB) + Ryzen 7 5700G (Zen 3, AVX2, no VNNI) + DDR4 + PCIe 3.0.

### A. Runtime techniques (llm-runtimes-2026)

| lead | number + hw | exact/lossy | in our src | in plan (line) | source file:line |
|---|---|---|---|---|---|
| Expert Deferral, tuned: defer only cache-missed, lowest-weight experts to layer k+2 | KTransformers up to 1.45x decode, avg acc drop <=0.5% (DS-V3 class, 8 routed experts; A100/RTX 4080 + Xeon). Local light variant +8% short / -7% edit (CUDA); HIP gives non-finite logits | lossy | yes: `STRATA_VERIFY_DEFER_CPU` include/strata/core/defer_auto.hpp:2; rank guard defer_rank.hpp:2 | item 8/28: 884, 1140 | LR/rep:60,72; LR/n/single_node_applicability.md:113 |
| One graph per token with CPU work inside (host callbacks) | KT up to 1.23x | exact | banked: verify window HIP graph + `STRATA_WAIT_FUSE` include/strata/kernels/verify_kernels.hpp:102 | closed 1212, 1307 | LR/rep:60,68 |
| MTP draft captured with verify; no recapture when the window shape changes | llama.cpp CUDA-graph re-enable +5% bf16 MoE bs1 (RTX 5090); draft ~9% of window | exact | draft already graph-chained src/core/mtp.cpp chain (mtp.hpp:92); `STRATA_MTP_HIP_PRIO` mtp.cpp:513 (default on gfx12 only) | item 10 closed 221-229 (MTP_HIP_PRIO arm) | LR/rep:74 |
| Anticipatory checkpoint at each message boundary (DeltaNet state) | first edit 17.2 -> 11.5 s (-33%) with 32K chunks (Strata) | exact | `STRATA_CACHE_MESSAGE_BOUNDARY` src/core/conversation_disk_env.def:321 | item 11: 879 | LR/rep:68,76 |
| Cross-session shared prefix (system prompt + tool schema) | none | exact | pinned shared prefix include/strata/core/conversation_cache.hpp:194 | 368, 2938 (A12) | LR/rep:76 |
| Thinking budget, `</think>` + transition phrase at cap | Qwen3 tolerates truncation (thinking-mode fusion); no tok/s gain, less time per reply | lossy (text) | no in src; serve/frontend.py:853 `reasoning_budget_tokens` | item 12: 881, 914 | LR/rep:78; LR/n/runtime_intelligence.md:215 |
| Dynasor-CoT certainty probe (early exit) | up to 29% fewer tokens; Certaindex up to 50% | lossy | include/strata/program/think_probe.hpp:4 | 881 (deferred), 265 | LR/rep:29,78 |
| Stream experts to GPU for large prefill ubatches | 1,200 tok/s DS-V3 (arXiv 2606.10493); PCIe 3.0 ~12 GB/s caps it | exact | prefill streamed experts src/prefill/prefill.cpp:2445 | A32: 1952; "already implemented" 1580 | LR/rep:80 |
| KV prefetch during prefill (next layer's KV on own stream) | -5.3% prefill at 32K, -4.1% at 64K, 0 at 8K (2x RTX 3080, PCIe 3.0) | exact | `STRATA_KV_PREFETCH` src/prefill/prefill.cpp:3359 | 620 | LR/rep:68 |
| Server never blocks engine: async token streaming over the pipe | est. a few %, not measured | exact | no (serve side) | **no** | LR/rep:88; LR/n/single_node_applicability.md:55,112 |
| Single-node KV/state tier RAM + disk (LMCache-style) | 2.2-32.8x lower hit TTFT vs recompute (vLLM, Llama 3.2 3B, NVMe) | exact | item 39 include/strata/core/conversation_disk_tier.hpp:1 | 1027 | LR/rep:60 |
| Expert-cache policy headroom | replay at 44%: engine tier 79.6%, every-window x160 85.0%, per-layer LRU 89.5%, Belady 94.7% | exact | adaptive tier (n/a) | 851, 930 | LR/n/single_node_applicability.md:35,114 |
| ClassLRU decay (llama.cpp CPU share) | within 0.5 pt hit rate in replay | exact | no | closed 859 | LR/n/single_node_applicability.md:34 |
| HybriMoE dynamic intra-layer CPU/GPU split | 1.70x decode / 1.33x prefill over prior hybrid | exact | PcieAuto include/strata/core/pcie_share_rule.hpp:2 | closed 1301 | LR/n/single_node_applicability.md:115 |
| SpecReason step-level speculation | 1.4-3.0x (paper) | lossy | no | **no** | LR/rep:29 |
| Reasoning KV compression (R-KV, ThinKV, TIP) | eviction can lengthen traces; TIP needs a trained model | lossy | no | 2132 | LR/rep:29,94 |

### B. Data path (CPU/RAM, PCIe, NVMe, GPU sync)

| lead | number + hw | exact/lossy | in our src | in plan (line) | source file:line |
|---|---|---|---|---|---|
| 38 Direct O_DIRECT arena loader on Linux (DirectFile, 16 threads) | 28-65 s at 0.49-1.18 GiB/s -> est. 10-14 s; measured floor 16.9 s for 35.45 GB (DRAM-less NV2) | exact | `load_experts_direct` src/core/pinned.cu:1028; `experts_unbuffered` include/strata/core/pinned.hpp:201 | 1021; A52 2016; default-on 1580 | DP/rep:33,35 |
| `POSIX_FADV_DONTNEED` after buffered chunks (interim) | memory pressure, not speed | exact | src/core/expert_source.cpp:928 | 1025 | DP/rep:35 |
| `POSIX_FADV_SEQUENTIAL` on buffered fallback | none | exact | yes, other path: src/core/expert_source.cpp:5954 | **no** | DP/n/nvme_storage.md:203 |
| `MADV_POPULATE_WRITE` arena prefault | load-time only, unmeasured | exact | no | parked 1345 | DP/n/cpu_ram.md:71 |
| Huge pages (hugetlb / THP) | log: "MAP_HUGETLB unavailable (needed 16920 2 MiB pages)"; hugetlb +5-7% does not fit box | exact | `MAP_HUGETLB` include/strata/core/pinned.hpp:8 | 1580; item 85: 451 | DP/rep:29 |
| 39 Auto disk tier + Bidaw (tool-call stays in RAM, final answer to disk first) | restore 1.13 GB est. 0.5-1 s; CachedAttention -87% TTFT; HCache 5.7x | exact | `--conversation-disk-dir` include/strata/core/conversation_disk_tier.hpp:1 | 1027 | DP/rep:37 |
| 40 `_mm_prefetch` 1-2 KB ahead in Q2_0 `row_multi` | 0-5% of 17.4 ms CPU | exact | `STRATA_Q2_PREFETCH` src/kernels/cpu/q2_avx2.cpp:82 | 219, 1044 | DP/rep:27 |
| 41 Smallest saturating thread count (FUSE_PHASES) + `performance` governor | 0-3% each; 6-12 threads equal = saturation | exact | FUSE_PHASES include/strata/kernels/cpu/pool.hpp:382; governor no (os_tune.sh sets only iGPU dpm) | 236, 1046 | DP/rep:27 |
| 42 Max-plus per-layer simulator; measure alpha (DDR contention pool vs DMA) | validate to 5-9% error (Vidur, LLMServingSim) | n/a | tools only: tools/pancake/chain_model.py, time_ceiling.py | 152, 1048 | DP/rep:41 |
| 43 `DraftPolicy::seed_costs` from traces (replaces RTX 5070 `kShape`) | 0-5% | exact | include/strata/spec/draft_policy.hpp:76 | 220, 1053 | DP/rep:41 |
| 44 Critical-path pass (slack per layer, virtual speedup) | diagnostic | n/a | partial, tools only: tools/pancake/probes/hit_critical.py | 304, 1056 | DP/rep:41 |
| 45 `STRATA_IO_THREADS` sweep, filefrag, PLE frequency order | diagnostic | exact | src/platform/direct_file.cpp:62 | 256, 1057 | DP/rep:37 |
| 46 Single poller in `combine_cpu`, VRAM flag republish | 0-3%; ~50 pollers per window today | exact | `STRATA_COMBINE_POLL1` src/kernels/cuda/native_moe.cu:147 | 216, 1064 | DP/rep:17,21 |
| 47 Warm Infinity Cache with layer l+1 pre-doorbell weights during spin | 0-2 ms/window if warm/cold < 0.85 | exact | `STRATA_IC_WARM_MB` include/strata/core/verify.hpp:686 | 215, 1067 | DP/rep:61 |
| 48 Fuse pre-doorbell chains (HC norm/down/up, router+plan+bell) | 1-4% per chain, both window types | exact | `STRATA_ROUTER_FUSE`, `STRATA_EXPDEC_SWIGLU` src/core/conversation_disk_env.def:23,35 | 204, 1069 | DP/rep:9,59 |
| 49 Timer stamps inside `combine_cpu` | instrument | n/a | src/core/verify.cpp:2894 | 1072 | DP/rep:53 |
| 50 Spin vs row pull: combine with flag pre-set | decides 51/52 (pull < 3 us/layer -> ~0) | n/a | no (bench) | 1082 | DP/rep:19 |
| 51 CPU pre-sums missed rows per token | 2-4x fewer pull bytes, 0.5-1.5% edit | not bit-identical | `STRATA_CPU_PRESUM` include/strata/core/expert_source.hpp:428 | 234, 1084 | DP/rep:21 |
| 52 CPU pushes rows + flag into VRAM via ReBAR (WC stores) | 1-2% est.; GPU bound <=0.27 ms; datapath <=2.1% | exact (with 51: no) | `STRATA_PRESUM_VRAM` include/strata/core/verify.hpp:782 | 254, 1086, 1580 | DP/rep:21 |
| 53 Online swap budget; 256 KB chunked promotions; stale-drop in `fill_loop` | 0-5% over best fixed `--adapt-swaps` | exact | `STRATA_ADAPT_SWAPS_AUTO` def:295; `fill_loop` has no staleness check (expert_source.cpp:6960) | 217, 1088; stale-drop rejected 2023 | DP/rep:21 |
| 54 IOMMU passthrough guard at start | protects status quo; strict 4 KiB = IOTLB misses (~344 pages per 1.4 MB) | exact | include/strata/platform/iommu_note.hpp:17 | 1091; A53 2019 | DP/rep:21 |
| GPU-busy decomposition: total - calls x (mean - min) for spin kernels | combine_cpu min 5.6 us vs mean 254 us: >=97.8% of its 38.3% is waiting | n/a | no | **no** (1063 states the ratio only) | DP/rep:11,70; DP/n/gpu_sync.md:229 |
| HIP micro-bench set: zero-copy read BW, one PCIe read RTT, CPU WC-write BW into BAR, pool + H2D contention | gates 52 and alpha | n/a | no | partial (50: 1082, alpha: 1051); **no** explicit bench | DP/rep:70 |
| `GPU_MAX_HW_QUEUES` 4 vs 6; CU-masked stream for spin/warm kernels | 4 vs 6 flat today; ROCm recommends <=4 | exact | queues logged src/core/verify.cpp:1294; CU mask no | 1075, 1171 | DP/n/gpu_sync.md:193-194 |
| `STRATA_COPY_HIP_PRIO` + coalesce adjacent promotion copies | 0-1%, likely flat at pcie-frac 0 | exact | src/core/verify.cpp:1377; coalesce no | parked 1347 | DP/n/pcie_transfers.md:190 |
| io_uring in DirectFile | ~1.06x drop-in; SQPOLL burns a core | exact | no | closed 1342 | DP/rep:93 |
| Q2_0 code entropy check | Q2_0 down <=12% compressible, IQ <=3.4% | n/a | tools/pancake/expert_entropy.py | A29 closed 1789 | DP/rep:70,77 |

### C. Platform, BIOS and hard facts for this box

| lead | number + hw | exact/lossy | in our src | in plan (line) | source file:line |
|---|---|---|---|---|---|
| 55 Startup calibration on window time (2-3 `--pcie-frac` values, hysteresis, per-machine cache) | PcieAuto picked 0.20-0.25, `--pcie-frac 0` won 111.7 -> 118.4 tok/s; R9700 0.55 -> 0.15 gave +9.9% / +25.6% (8K/32K) | exact | include/strata/core/calibrate_ratio.hpp:4 | 1098, 203 (pkg 63), 1565 | CP/rep:83,103 |
| 61 Defaults by VRAM size and link probe; emulate small cards by capping slots | 20 GB -> 10,909 of 24,576 pairs (44%), ~4.7 GB non-expert | exact | no dedicated code | 1124 | CP/rep:57,105 |
| 57 cheap: platform log line + `STRATA_FORCE_ISA=avx512\|avx-vnni` | XS | exact | `STRATA_FORCE_ISA` accepts only avx2/avx/sse (include/strata/kernels/cpu/expert_layout.hpp:67); `STRATA_PROBE` no | 257, 287 | CP/rep:113 |
| 59 code read: do Q2_0/IQ AVX-512 rows use the multi-token window | DS path 1.05-1.3x at 1 tok, 2.4-3.5x at 4, 2.8-4.8x at 16 (Zen 4/5) | exact | `row_dot<TY,NT>` in iq_avx512.cpp | 1113 | CP/rep:93,121 |
| BIOS: iGPU carve-out 4 -> 2 GB; DDR4-3600 at FCLK 1800 | unmeasured; no paper on desktop FCLK vs decode | exact | no | owner call 1143 | CP/rep:160; DP/n/cpu_ram.md:121 |
| BIOS: force GPU slot to Gen2 to emulate an x8 card on Gen3 | ~6.5 GB/s, ~215 us per promotion | n/a | no | **no** | CP/rep:129,145 |
| Check board chipset (`dmidecode -t baseboard`) and card `LnkCap` | B550/X570 -> Gen4 with non-APU Zen 3; B450/X470 stays Gen3 | n/a | n/a | dmidecode 1119; LnkCap **no** | CP/rep:158 |
| Fact: link = root port Gen3 x16 (card Gen4 x16), ReBAR 32 GB BAR0, IOMMU identity, probe 13.4 GB/s, promotion 115 us vs 107 us theoretical (~93%) | measured | n/a | n/a | 749, 1080-1081 | DP/rep:3,17 |
| Fact: CPU 42 us/expert vs promote 115 us (1:2.7); Q2_0 ~80% of 33 GB/s DDR4 (Intel's best x86 2-bit GEMV: 80-90% band) | measured | n/a | n/a | 1099-1100, 1994 | CP/rep:63; DP/rep:25 |
| Fact: Zen 3 has no AVX-VNNI; `maddubs`+`madd` fallback costs 2 extra uops per 32 B | from cpuinfo | n/a | VNNI probe src/kernels/cpu/expert.hpp:192 | **no** | DP/rep:27 |
| Fact: edit window 37.4 ms = GPU-reach 15.8 + CPU 17.4 + draft 2.5 + ~1.7 sync; ideal overlap ~20 ms; pre-doorbell kernels critical in both windows | measured (profile-fasthip) | n/a | n/a | 1052, 1061 | DP/rep:3,9 |
| Fact: fillable GPU slack per layer = CPU - hits: ~40 us short, ~275 us edit (bounds 47 and any idle fill) | derived | n/a | n/a | **no** | DP/rep:11; DP/n/gpu_sync.md:38 |
| Fact: NVMe Kingston NV2 PCIe 3.0 x4 (3.5 GB/s spec); disk tier wear ~2%/yr of 640 TBW at 30 parks/day; D 91% full | spec + est. | n/a | n/a | 1580 (floor); wear **no** | DP/rep:35,37 |
| Fact: hit curve concave; VRAM beats PCIe gen on this host; 7900 XTX = same binary, +~2,900 pairs (56%) | est. | n/a | n/a | 1130 | CP/rep:57,164 |
| Fact: Gen4 x16 promotion ~50 us (28 GB/s measured on 7900 GRE); break-even link ~33 GB/s | measured (other box) + arithmetic | n/a | n/a | 1120, 1132 | CP/rep:63 |

### D. Other-hardware items (in plan, not actionable on this box)

| lead | number + hw | exact/lossy | in our src | in plan (line) | source file:line |
|---|---|---|---|---|---|
| 56 Hybrid P/E pool affinity default | 8P+12E: 12-16 workers beat 19 | exact | `--pool-affinity` include/strata/kernels/cpu/pool.hpp:73 | 1101 | CP/rep:97,117 |
| 58 CCD-aware pool on dual-CCD Zen 4/5 | ~64 GB/s per CCD cap | exact | no | 1111 | CP/rep:87,119 |
| 57 rest: AVX-512 without VBMI default; old Q2_0 folders on AVX2 | Skylake-X/Cascade Lake | rounding differs | `STRATA_VNNI_TIER` src/kernels/cpu/expert_layout.cpp:196 | 1104 | CP/rep:91,123 |
| 60 Strix Halo unified preset | 19-21 tok/s code (vs ~126 here) | exact | unified path (docs) | 1116 | CP/rep:99,125 |
| 62 gfx12 re-measure | R9700 pcie-frac 0.15 | exact | gfx12 defaults | 1128 | CP/rep:127 |

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

Base path: `pancake-docs/research/avalon/literature/`. Abbreviations: AS=asia, CA=canada, EU=europe, MH=mit-harvard, PA=pacific-africa-latin-america, ST=stanford, US=us-universities (all `<dir>/report.md`). E = exact, L = lossy, - = no engine output effect (measurement/agent). Plan = MOD_PLAN.md line (first relevant; "idx" = only the Phase 11 research-index summary row 858-870, no item). Src = first `git grep` hit in `src include` (serve/tools noted when the feature lives there). Numbers are the authors', on NVIDIA/other HW unless marked.

Dropped as off-target: cluster/batch systems (DeepEP, EPLB, MegaScale, Comet, Mooncake, MoE-Gen, NEO, Pie-batch, Hydragen, Tokasaurus...), retraining-only (NSA, MoBA, Kimi Linear, ProSparse, EAGLE-3, Pre-gated, Read-ME, PASTA, SEAL...), pure agent/NLP with no engine or gate link (AutoCodeRover, AutoCompact, ECLoop, STAIR, Aegis, AsyncFC, Slipstream, RLM, MEM1, Self-repair, CodeSteer, ACE, DSPy/MIPRO/GEPA, AWM, SWE-agent, LLMCompiler, KNUST tools).

### Expert path, CPU pool, cache (decode-critical)

| lead | number + hw | exact/lossy | in our src | in plan (line) | source file:line |
|---|---|---|---|---|---|
| KTransformers expert deferral (low-weight misses finish under next layer) | DS-V3 -26%/layer, +33% decode, up to +45%; tasks <=2 pts (NVIDIA) | L | include/strata/core/defer_rank.hpp:3 | 1861 (A3 done, in fast-hip); 14 sim +33% | AS:44-48, AS:169 |
| Rank guard "top-2 or weight>=w" | est +10-25% decode after HIP fix | L | include/strata/core/defer_auto.hpp:91 (`_TOP`), defer_rank.hpp:48 (`_FOLD`) | 989, 1862 (A4) | AS:46, AS:169 |
| Per-expert barriers + fused combine in CPU pool | est 1-4% exact; plan measured +2.5% edit | E | include/strata/kernels/cpu/pool.hpp:382 (`STRATA_POOL_FUSE_PHASES`) | 973, 2000 (A47 done) | AS:50, AS:172 |
| Cache-conditional routing (= `turbo`/`STRATA_ROUTE_PRIOR`) | up to 2x phones; ours 111.7->149.3 tok/s 7900 XT, 2.9% diff tokens (3060) | L | include/strata/kernels/route_prior.hpp:12 | 1859 (A1) | EU:33, EU:143 |
| AcceptMoE depth-graded verify prior | 1.29x resident / 2.06x offload, traffic -74-77%, -0.27 pts (SGLang) | L | route_prior.hpp:12 (`STRATA_ROUTE_PRIOR_DEPTH`) | 955, 1860 (A2) | EU:35-37, EU:144 |
| OEA sibling-aware routing in verify window | -39% latency at batch 16 | L | no | 1247 (parked), 1876 | MH:38, MH:124 |
| HybriMoE / AdapMoE / MoBiLE / BuddyMoE (turbo-family variants) | 1.70x; 1.60-1.72x; BuddyMoE -5% ARC | L | no | 1274, 1299 (closed/parked) | AS:50, AS:58 |
| Cost-aware Belady ceiling (Baleen method) | promo ~115 us vs ~42 us CPU per 1.4 MB expert | - | tools only (time_ceiling.py) | 926-937 (item 14 result), 1863 | US:25-29, US:166 |
| k-th-miss admission (SIEVE/S3-FIFO probation) | SIEVE miss -1.5% avg (up to 63%) | E (rounding) | include/strata/core/epoch_bandit.hpp:9 (`STRATA_ADAPT_KMISS`) | 931, 1863 (A5) | US:25-29, US:167 |
| ADRS/OpenEvolve policy search; Harmonia split | ADRS 3.7 ms vs 540 ms EPLB (inconsistent); Harmonia +49.5% | E | no | 932, 1992 (A46 parked) | US:29, EU:53, US:167 |
| MoE-Infinity early-layer bias + nearest-conversation seeds | 17% hit @10% cache (MoE-Beyond replication) | E | no (A10 opt-in under other name) | 965, 1868 (A10) | EU:51, EU:147 |
| fMoE embedding-matched expert maps (warm-up profile) | -47% latency, +36% hit (v1) | E | no | ~1868 (A10 covers match) | US:31 |
| ProMoE chunked/cancellable promotions + reordered exec | 1.09x over LRU avg (max 1.17x), llama.cpp PCIe4 | E | include/strata/program/promo_chunk.hpp:1 | 977, 1865 (A7) | AS:58, AS:174 |
| Next-expert predictor recall test (Fate/HOBBIT/SP-MoE/SmallThinker pre-attn) | papers 84.7-97% on 8-64 experts; ours 19-21% recall | - | include/strata/core/expert_source.hpp:345 (`STRATA_LOOKAHEAD_STATS`) | 975-983 (item 30 closed) | AS:54-56, AS:185 |
| ExFlow P(l+1\|l) affinity table | affinity from ~5k tokens | E | no | ~976 (item 30 closed) | US:31 |
| KVFlow-style expert pre-staging during tool runs | KVFlow 1.83x vs SGLang HiCache | E (rounding) | conversation_disk_env.def:439 (`STRATA_IDLE_ADAPT`) | 945, 1867 (A9) | US:51, US:170 |
| FreeToken q* CPU/PCIe split vs `--pcie-frac` | q*~0.36m; ours pcie-frac 0 best (+6/+13%) | E | include/strata/core/calibrate_ratio.hpp:14 | 952 (item 22), 1299 | US:33, US:168 |
| Fiddler token-count threshold for CPU share (prefill/verify) | - | E (rounding) | include/strata/core/arch_defaults.hpp:30 (`PREFILL_CPU_SHARE`) | 1950 (A30), 3707 (spec 46) | US:33, US:174 |
| Athena epoch bandit for mixed opt-ins / POOL_PREREAD | 0-3% | E | include/strata/core/epoch_bandit.hpp:10 | 1990 (A44) | EU:53, EU:150 |
| Activation sparsity masks (Berkeley gate / TEAL / CATS / SpAx / CHESS / per-family) | Berkeley 84.5-87.4% Qwen3.5; TEAL 40% PPL 6.21 vs 5.87; SpAx 2.06x@4bit | L | no kernel (STRATA_ACT_DUMP_RAW diag only) | 919-927 (item 13 CLOSED: experts dense, gate cos 0.844@70%) | US:15-21, MH:13-17, CA:33, AS:62, EU:43 |
| BlockFFN union-over-window sparsity column | 80% -> 71% at 8-token union | - | no | 987, 1188 | AS:62, AS:186 |
| CETT-PPL-1% per-layer threshold (Sparsing Law) | - | - | no | no (moot: item 13 closed) | AS:62 |
| Transposed n_ff-major down layout (PowerInfer) | prereq for any column skip | E | no | 987, 921 | AS:62 |
| T-MAC LUT GEMV / Q2_0 kernel micro-bench | T-MAC up to 6.6x vs 2024 llama.cpp | E | no | 1047 (Q2_0 near roofline), 2001 (A48) | AS:64, AS:190 |
| HOBBIT/MC-MoE low-precision copy for misses | MC-MoE 2.54 bit, <0.6% loss; HOBBIT <=1% GSM8K | L | include/strata/artifact/shadow_pack.hpp:1 (A27) | 1294, 1936 (A27 reopened) | AS:64, AS:180 |
| IQ3-hot / IQ2-cold per-tensor mix from two GGUFs (Precision scaling law + LQ-LoRA ILP) | PTQ loss grows with pretrain tokens (<=1.7B fit) | L | no | idx 863 only | MH:36, MH:129 |
| MatGPTQ matryoshka file (4-bit slice VRAM, upper bits RAM) | - | E/L | no | no | EU:23 |
| Any-Precision bitplanes for draft | n/a: MTP drafter's 512 experts in VRAM (835 MiB) | - | no | 989 (does not apply) | AS:64 |

### Drafting, verify, recurrent state

| lead | number + hw | exact/lossy | in our src | in plan (line) | source file:line |
|---|---|---|---|---|---|
| MTP depth 4 (Qwen tech report) | 4.07 accepted @4 steps; SGLang 3.3 incl bonus, 540 tok/s B200 | E | preset `--spec 4` (configs/pancake) | idx 866; 3/5/6 measured flat | AS:19, AS:30 |
| DISCO drafter-entropy feature (MiMo) | MiMo acc <=3.6, 2.6x with 3 MTP layers | E | include/strata/spec/disco.hpp:30 (entropy "not exposed") | 1920 (A22) | AS:20, AS:30, AS:171 |
| SpecFit per-position new-miss cost | +21.6% / +7.0% over EAGLE-3 same cache (Qwen3-30B) | E | include/strata/core/row_miss.hpp:2 | 1010, 1921 (A23 coded) | PA:11, PA:73 |
| Top-k entropy (Entropy Sentinel) in DISCO + loop guard | r=0.91 vs judge (some models) | E/- | serve/rumination.py:19 (entropy opt-in) | 1011, 1907 (A20) | PA:53, PA:74 |
| Draft source as learned action (bandit) | 0-5% est | E | include/strata/spec/draft_source.hpp:1 (source, not learned) | 1924 (A26 parked) | CA:37, CA:93 |
| SuffixDecoding cross-session index + frequency scoring | up to 3.9x SWE-Bench, ~2.2x vs EAGLE-2/3 | E | include/strata/spec/suffix_index.hpp:1 (A24) | 940, 1922, 3694 (spec 45) | US:39, US:171 |
| MTP per-position decay check (SD Illusion) | 0.92/0.68/0.38; verify 42-95% of time | E | no | 942, 1923 (A25) | US:37, US:173 |
| Typical acceptance on thinking only (Medusa) | est 5-15% on thinking | L | no | 1337 (parked) | US:41, US:173 |
| Cactus KL-budget acceptance | acc 4.27->5.44, 1.88x vs 1.81x (T>0) | L | no | 1284 (parked: ~0 at T=0.1) | CA:35, CA:95 |
| AdaSD adaptive length rule (as DISCO check) | - | L (relaxed accept) | no | no | AS:70, AS:148 |
| Token Recycling drafter | ~2x SpecBench, <2 MB matrix | E | no | 1263 (closed) | AS:70 |
| SSD-lite (Saguaro): precompute ngram drafts on CPU during verify | Saguaro ~30% over best SD (dedicated H100 drafter) | E | no | 1336 (parked: host-gap profile) | ST:45, ST:161 |
| Staged SD / Hydra top-2 tree | Hydra 1.31x vs Medusa; staged 3.16x GPT-2-L | E | no | 1237, 1261 (parked/closed) | ST:47, MH:72 |
| ReplaySSM audit of GDN replay (u ring buffer) | 1.87-1.96x batched vLLM; ours est 1-3% decode, 0-8% spec | E | no | 949 (item 20), 1928 (closed: replay stays) | US:43, US:172 |
| SpecLA: recover accepted GDN states from verify factors (no replay) | - | E | no | no | AS:70 |
| LongSpec/Hybrid Tree: verify rows read prefix KV once | ~75% less attn latency vs HF | E | no (Flash-Next not built) | 2011 (A50: QSA caps ~1%) | AS:68, AS:176 |
| Exact speculative sampling for T>0 requests | - | E | include/strata/core/coupled_draft.hpp:65 (`STRATA_SPEC_PROB`) | 1919 (A21) | CA:35 (context) |

### GPU kernels, attention, prefill

| lead | number + hw | exact/lossy | in our src | in plan (line) | source file:line |
|---|---|---|---|---|---|
| Gated Residual small-M fusion (SGLang M=4) | Mix 2.05x, Combine 1.96x; +7.6% / +5.49% e2e (B200) | E | include/strata/core/switch_report.hpp:35 (`STRATA_EXPDEC_SWIGLU`); GR_FUSED in conversation_disk_env.def:418 | 986 (item 32), 1964 (A33 retest) | AS:24, AS:34, AS:173 |
| GDN prefill chunk 64 + 16x16 block inverse + fused prep (gfx11 path is chunk 16) | est 5-11% prefill (GDN 22% of prefill) | E (FP16 parity) | include/strata/prefill/gdn_chunk.hpp:1 (sm_70 only, chunk 64) | 947 (item 18), 1951 (A31 parked) | MH:44, MH:121 |
| BFLA sparse prefill on QSA layers (measured on Flash-Next) | 1.342-1.435x TTFT @128-256K, LongBench 57.32=57.32 (H200) | L | include/strata/kernels/bfla.hpp:1 | 1008-1009 (item 19), 124 | AS:66, AS:177 |
| XAttention block-sparse prefill >64K | up to 13.5x @256K vs FA | L | no | 948 (item 19, superseded by BFLA) | MH:52, MH:122 |
| Quest / Twilight sparse decode attention | Quest 2.23x self-attn | L | no | 2010 (A49 measure), 2012 (A51 parked) | MH:52, US:188 |
| FLUTE duplicated LUT in LDS (bank conflicts) | 0-10% GPU expert time if conflicts | E | include/strata/kernels/iq_kernels.hpp:136 (`STRATA_IQ_LUT_DUP`) | 1968 (A37) | MH:29, MH:130 |
| UFMG coordinate-descent launch/tile sweep | 5.5-8.5% untuned (A100); est 1-3% ours | E | no (tools sweep.py) | 1012 (item 37), 1965 (A34) | PA:51, PA:75 |
| SageAttention/SpargeAttn | CUDA only | L | no | 1632 | AS:66 |
| XGrammar-2 TagDispatch (grammar masks only after tag) | - | E | no | no | US:59, US:175 |
| QuaRot folded Hadamard (if rotated GGUF appears) | Llama2-70B 4-bit <=0.47 PPL | E (runtime) | include/strata/kernels/cpu/expert_layout.hpp:109 (hadamard path) | 1201 (parked: 354 GB BF16) | EU:45, EU:154 |
| LQER/QERA low-rank error correction | - | L | no | 1202 (parked) | EU:45 |
| QTIP / IQ*_KT | ~0.2 bpw gain (ik) | - | src/kernels/cpu/tq_computed.hpp:7 | 1295 | US:124 |
| YAQA KL-based imatrix | -30% KL, needs backward | - | no | no | US:125 |

### Model files, quality gates, server

| lead | number + hw | exact/lossy | in our src | in plan (line) | source file:line |
|---|---|---|---|---|---|
| Q2_0 over IQ2_XS (DASLab card) | 93.8 vs 70.3 tok/s decode, 89.07 vs 89.16 avg (llama.cpp) | - | n/a | idx 865; item 6 closed | EU:17 |
| IQ1_M gate/up in 3 layers of IQ2_XS file | est 1-4% decode on IQ2_XS, 0 on Q2_0 | E | src/kernels/cpu/iq_avx2_rows.inl:226 (IQ1_M rows now native; iq_avx2.cpp:12 comment stale) | 958 (item 24) | EU:19, EU:145 |
| Coder variant A/B (256/512 experts, 29.6 GB) | SWE-V 79.20 vs 82.80 BF16 (95.7%), LCB v6 87.43 | L | include/strata/artifact/model_family.hpp:17 (name only) | 960 (item 25), 1937 (A28) | EU:21, EU:151 |
| Request "Coder Q2_0" build from DASLab | - | - | n/a | no | EU:152 |
| EAR metric (SLQ) beside same-top-token | lossless = EAR>=0.99, KL<~0.01 | - | tools (quality_gate --ear planned) | 1799, 2030 (A56) | EU:23, EU:158 |
| KLD/top-1 blind in quiet zone -> task gate decides | rho 0.00 (Qwen3.6-A3B), -0.24 (Devstral) vs -0.72/-0.86 cohort | - | tools/pancake/task_gate.py | 2028 (A54 done), 2029 (A55) | CA:13-15, CA:87 |
| BFCL v4 + LCB + HELMET gate ladder | - | - | tools (task_gate) | 1139, 2029 | US:63, US:185 |
| SWT-Bench Lite as lossy gate | - | - | no | no | EU:59, EU:159 |
| SWE-Effi AUC time-to-success | failures ~8.8M tok/658 s vs 1.8M/167 s | - | no | 2031 (A57) | CA:17, CA:96 |
| Repeated runs as distributions (Price Reversal) | thinking tokens vary up to 9.7x | - | tools/pancake/abtest.py | 2032 (A58 done) | US:63, US:187 |
| Restore-equivalence test (UNSW hybrid-state bug) | 34/36 -> 36/36 after fix (vLLM) | - | tools restore_check.py | 1007 (item 35), 1889 (A14) | PA:13, PA:72 |
| preserve_thinking end to end (opencode resend) | re-prefill from first dropped think block | E | serve/chat_template.jinja:119 | 984 (item 31) | AS:25, AS:36, AS:170 |
| Checkpoints at tool boundaries + Marconi FLOP/byte eviction | Marconi 34.4x token hit, TTFT -71.1% | E | include/strata/program/conv_cache.hpp:121 | 943 (item 16), 1886 (A11 done) | US:47-49, US:169 |
| Multi-entry prefix store (RadixAttention) | up to 6.4x throughput | E | include/strata/core/conversation_cache.hpp:533 (`STRATA_AUTO_PIN`) | 1887 (A12 done) | ST:29, ST:156 |
| Continuum/MORI TTL for parked sub-agents | MORI +20-71% thru (Claude Code traces) | E | no | 1890 (A15 new) | US:47, US:183 |
| Graceful close at thinking cap (Interruptible LRMs) | up to -60% acc under abrupt cuts | L | serve (grace 512 + close_retry) | 1904 (A17) | US:57, US:176 |
| Learned thinking stop probe (Thought Calibration / LHHT) | -60% thinking ID, -20% OOD, no loss | L | include/strata/core/verify.hpp:130 (`STRATA_THINK_PROBE` dump) | 950, 1906 (A19) | MH:68-70, MH:120 |
| Rumination 5-gram rate -> soft close | no threshold in paper | L (opt-in close) | serve/rumination.py:1 | 993, 1907 (A20) | CA:25, CA:89 |
| Loop-guard behaviour counters | failed runs +39-343% tokens; Qwen Code 25.81% | - | serve/agent_loop.py:9 (A38) | 992, 1979 (A38) | CA:21-23, CA:88 |
| oldString substring constraint | ~94% compile errors are type errors; halves them | L | serve/agent_loop.py:22 (A43 check) | 963, 1984 (A43 parked: 1/180) | EU:57, EU:148 |
| presence_penalty = 0 on tool turns | SWE-bench 74.7 -> 70.6 with it on | - | serve/server.py:1829 | 967, 261 | EU:23, EU:149 |
| Effort per task: xhigh vs medium | Qwen: low effort can lengthen tasks; Overthinking -43% compute | L | n/a | 1905 (A18) | AS:38, AS:189, US:57 |
| PLE path: keep `--ple-io direct`, measure prefetch time + row-cache hit | 1M-row x 90 B cache | E | src/program/generate.cpp:695 | idx 866 only | AS:23, AS:32, AS:175 |
| Official sampling T=1.0/top_p .95/top_k 20; no static YaRN | - | - | n/a | idx 866; 1286 (YaRN parked) | AS:26-28, AS:179 |
| Tool-argument hidden-state probe (LLMs Know More) | probes do not generalize across datasets | - | no (would reuse A19 dump) | no | EU:61, EU:156 |
| ContextSniper tool-output filtering before context | -51.5% tokens SWE-bench Lite | L | no | no | AS:72, AS:181 |
| Calibration on opencode traces (only at requant) | PPL -3.52 (4-bit dense 7-8B) | - | no | no | PA:34, PA:76 |
| Mind the Gap: official source + SHA-256 vs HF LFS | attack delta 88.7% insecure code | - | no | idx 865 only | EU:29, EU:160 |
| AgentDojo injection run vs opencode+pancake | ~97 tasks, ~629 security cases | - | no | no | EU:59, EU:161 |
| Demons in the Detail: global-batch LBL -> session-stable expert sets | - | - | n/a | no (supports items 3-4) | AS:40, AS:188 |

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
Source keys (all under `pancake-docs/research/avalon/`): UR = upgrades-from-upstream/report.md; LP = .../notes/llamacpp_performance.md; LS = .../notes/llamacpp_server_features.md; PM = .../notes/project_maya.md; ST = .../notes/strata.md; AB = .../notes/avalon_baseline.md; PR = prose-gap/report.md; AV = prose-gap/notes/avalon_verify_path.md; PV = prose-gap/notes/pancake_verify_path.md; AD = prose-gap/notes/prose_ab_test_design.md.
Dropped as irrelevant to a single 7900 XT box: NVIDIA-only kernels (#31802, #31533 as written, #27140, #28714/#28195/#26301 RDNA3.5), multi-GPU (#29963, Avalon pipeline stages / expert split, Maya `--tp`, `STRATA_DISJOINT_ADAPT`), Vulkan (#31139), DSpark/EXL3/DFlash2-GLM, Responses/realtime/router API items, GLM-5.3 on Maya.

### Kernels / GPU (upstream llama.cpp, Strata)

| lead | number + hw | exact/lossy | in our src | in plan (line) | source file:line |
|---|---|---|---|---|---|
| llama.cpp #31751 `#pragma unroll 1` stops J=32 MMQ register spills (Q8_0-layout types) | PP17 +36.5%, PP32 +36.1%; W7800 gfx1100, ROCm 10.1, Q8_0 | exact | no (no 31751 hit; our verify uses own kernels, ggml MMQ only on prompt path) | 3968 (E#2, "check pinned ggml first") | UR:5,59,194; LP:14 |
| llama.cpp #31746 D256 MMA FA tune for gfx1100 | attn op 0.70-0.84x of master; W7800 ROCm 10.1 | exact (rounding) | no | no (only 626 = 8.8 own D256 FA prompt kernel) | UR:60,206; LP:15 |
| llama.cpp #31234 / #30792 native q4/q8 KV in FA tile kernel | q4_0 tg +13% @64k, +24% @131k; R9700 RDNA4 | exact | no (own attention/QSA path) | no (2409 F22 = Q4 KV prompt attn, related) | UR:61-62,205; LP:16-17 |
| llama.cpp #29948/#31188 (+#30489) gate/up+GLU (+Q8_1 quant) fused in one launch | pp16384 +6-17% NVIDIA; gfx906 -15..-50% (#31599) | exact | yes, own: `src/kernels/cuda/s_gemv_pair.cu:1` | 204 (S7 expdec gate/up+SwiGLU+q8_1) | UR:69,74; LP:27,33 |
| llama.cpp #31047 linearized MoE layout / tile tuning | 2-4%; NVIDIA | exact | no | 3437, 3513 (V4 OPEN) | UR:73; LP:32 |
| llama.cpp #31443 / #31660 qwen4exp conv-state copies (tail w/o cont; snapshots in one set_rows) | "free 2%" QFN gen; NVIDIA | exact | #31660 N/A (no snapshot slots, gdn_conv_commit) ; #31443 not checked | 3439, 3490 (V6 checked); #31443 no | UR:110-111,207; AB:32-33 |
| llama.cpp #31625 HIP fast-math | none given | lossy-risk | no | 1352 (parked) | UR:66; LP:21 |
| llama.cpp #31838 fixed rocBLAS workspace (= U4 432433637) | untested | exact | N/A (rocBLAS opt-in only; hipBLASLt workspace once) | 3419, U4 status table | UR:71; LP:25 |
| ROCm 10.1 compiler: ~126 kernels regressed vs 7.2.4 | gfx906 Discussion #30622 | n/a (A/B rule) | n/a | no | UR:80; LP:38 |
| Strata hipBLASLt tuning table matching runtime (100401, not 100200) | prompt +32/+78/+82% @3K/6.5K/131K; 7900 XTX | exact-ish (GEMM solution choice) | yes `src/core/conversation_disk_env.def:197` STRATA_HIPBLASLT_TABLE | 858, 878, 887 (A/B queued) | UR:165,203; ST:70-78 |
| Strata `PA_FAST` + `PF_FUSED` + `PREFILL_RING=512` | +13.8% @94K deep prefill; 7900 XTX | lossy (not bitwise) | yes `include/strata/core/arch_defaults.hpp:7`, `include/strata/prefill/prefill.hpp:129` | 650 (PF_FUSED measured worse); bundle no | UR:173; ST:164-167 |
| Strata `STRATA_TSUM=1` | -0.91% decode time; 7900 XTX | exact (bitwise) | yes `src/core/arch_defaults.cpp:53` (arch table) | 494 (parsing note only) | UR:174; ST:166 |
| Strata exact long-prompt QSA scorer `STRATA_SCORES_THREAD=1` | +10-13% @131K, +22% @240K; 7900 XTX | exact | yes `src/core/conversation_disk_env.def:679` | 1329 (parked >131K) | UR:172; ST:155 |
| Strata int8-WMMA prompt attention (#5535), expert layouts, EXPERT_V2, EXPDEC (#5458) on gfx1100 | 0.995x e2e; layouts 2-3x slower; 7900 XTX | mixed | partial | 3978 (#5535 listed) | UR:159,243; ST:167-169 |

### MoE tier, I/O, memory

| lead | number + hw | exact/lossy | in our src | in plan (line) | source file:line |
|---|---|---|---|---|---|
| Strata PR #5561 `madvise(MADV_SEQUENTIAL)` on expert mmap (`STRATA_EXPERTS_SEQ_READAHEAD`) | follow-ups 65-76 s -> 23-28 s, decode 33-44 -> 42-52 t/s; RX 6900 XT, 32 GB RAM | exact (byte-identical) | no (only WILLNEED `src/core/expert_source.cpp:3080`, DONTNEED :4165) | 1349 (parked: major faults > 0 on mmap tier) | UR:11,163,202; ST:59-63 |
| Maya direct parallel expert reads (`MAYA_EXPERT_DIRECT`, `MAYA_EXPERT_STAGERS` 2-16, probed per drive) | decode +26% RAM-short (V100, 30 GB RAM); probed readers +5% | exact | partial: O_DIRECT arena loader `include/strata/core/conversation_memory.hpp:82`; no per-drive stager probe | 861 (test idea), 256 (45 IO_THREADS sweep, unowned) | PM:37-38,172 |
| Out-of-core O_DIRECT streaming #31469 / #25294 | TG 1.5 -> 16-22 t/s, 68 GB on 16 GB RAM; NVIDIA | exact | partial (arena) | 1267 (closed) | UR:91; LP:69-70 |
| Registered-RAM expert arena (Avalon 03c749e/213b569) / hot-expert mlock #26414 | none measured | exact | yes (arena; mlock note `conversation_memory.hpp:25`) | 169 (38 arena loader) | UR:98,192; AB:84 |
| Router-predicted next-layer `pread` prefetch (`STRATA_IO_PREFETCH`) | not isolated | exact | yes `include/strata/core/expert_source.hpp:341` | 709 (RouterLookahead) | UR:164; ST:56 |
| llama.cpp #29599 PLE row madvise prefetch (qwen4exp) | pp512 177 -> 383; Strix Halo ROCm, 104 GiB | exact | partial (own PLE reader, `--ple-io direct`) | 416 (79 PLE gather off crit path), 256 (45) | UR:102,105; LP:64 |
| Strata PLE table resident in RAM (`STRATA_PLE_AUTO_RAM`) | 8K prompt +22.5%; R9700 | exact | yes `include/strata/core/ple_residency.hpp:32` | no (does not fit 57 GB) | UR:170,242; ST:49-50 |
| ClassLRU/CRF eviction (#30637 -> #31233, #31853) | +6-7% decode QFN vs SLRU; 4090/5090 | exact | no (adaptive tier, other algorithm) | 859, 3494 (replayed: no gain, not ported) | UR:86,211; LP:61-63 |
| CPU-miss share + tuner #30966 | QFN 24.2 -> 47.6 t/s @6.4 GB cache | exact | yes (`--pcie-frac`, PCIe share) | 859, 3510 | UR:87; LP:62 |
| #31888 CPU part does not wait for cached experts | +7-9% TG Gemma 4; 0 on QFN | exact | yes-ish (`STRATA_ROUTE_PUBLISH`) | 3493 (V2 checked) | UR:88; LP:63 |
| Avalon prefill knobs: `--moe-cache-cpu-overlap 2`, `GGML_SCHED_PREFETCH_EXPERTS=6`, `-ub 19072` | prefill 2,113-2,130 t/s; 7900 XTX PCIe 4 | exact | Avalon-only; analog `PREFILL_CPU_SHARE`, ring, chunk | 2606 (17.0 c/d analog) | UR:193; AB:85,123 |
| Strata `STRATA_ADAPT_FETCH` (fetch admission) | AdrianBM96 3060 +5-7% (plan) | exact | yes `include/strata/core/expert_source.hpp:406` | 2207 (F15, only if pcie share > 0) | UR:168; ST:180 |
| Strata `STRATA_ADAPT_OVERLAP=1` | +4-8% A4000/P100; loses 5070 | exact | yes `include/strata/core/adapt_async_default.hpp:31` | 155 (non-finite bisect only) | ST:241 |
| Strata `STRATA_PCIE_BALANCE=1` | +5.6% A4000 | exact | yes `include/strata/core/nv_defaults.hpp:7` | no | ST:242 |
| Strata `STRATA_KV_PREFETCH=1` (next layer KV upload in prefill) | not stated | exact | yes `src/core/conversation_disk_env.def:463` | 620 (8.2) | ST:185 |
| Maya `--calibrate`-style sweep (CPU threads, pcie share, read size; keep > 3%) | few % to ~10% (unknown) | exact | partial `include/strata/core/calibrate_ratio.hpp:17` | 205 (63 probe + calibration) | PM:41,174 |
| Maya miss skip by gate-weight share (`MAYA_GLM_MISS_SKIP_W`/`_FROM`) | +7-13% decode, KL 0.037-0.052; RTX 3090 GLM | lossy | no | 3974 (E#8), 861 | PM:49,173 |
| Strata `STRATA_ROUTE_TAIL_SKIP=7` on gfx1100 | pancake "tail skip 7 129.7" (829); KL 0.003-0.006 | lossy | yes `include/strata/core/expert_source.hpp:670` | 2608 (17.2 step 5), 829 | PR:19,35; PV:93 |
| Strata router residency bias `ROUTE_PRIOR` / `CACHE_ROUTE` (PR #5596) | code +37% but agent 5.6 -> 4.5/10; DS +56% top-1 93-96% | lossy | yes (`STRATA_ROUTE_PRIOR`); CACHE_ROUTE no | 2607 (17.1), 627 (8.9) | UR:169,240; ST:42-46 |
| Strata KVA approximate late-layer prefill | prefill +28%, decode -11%; R9700 | lossy | yes `include/strata/core/defaults_026.hpp:7` | 1278 (closed) | UR:171,241 |
| llama.cpp #31886 ROCm staged <=64 MiB pinned uploads from mmap | load-hang fix | exact | N/A (own loader) | no | UR:100,210; LP:78 |

### Speculative / drafting

| lead | number + hw | exact/lossy | in our src | in plan (line) | source file:line |
|---|---|---|---|---|---|
| Longer prompt-lookup window `--lookup-max-t 16` | +11-14% on edits; RTX A6000 IQ3_S | exact | yes `include/strata/core/lookup_default.hpp:1` (HIP kVerifyLongT 16) | 858, 887 (A/B queued) | UR:176,204; ST:149 |
| llama.cpp #29548 ngram-simple keeps scanning older matches on periodic text | 0 -> nonzero drafts (198 calls) | exact | no (our suffix/lookup drafter differs; same give-up unchecked) | no | UR:123,195; LP:109 |
| `--spec 3` (cap 2 drafts, keep `--spec-min-p 0.5`) | prose 0..+2% est.; code may lose | exact | yes (`--spec`, generate.cpp:4352-4356 per note) | 2608 (17.2 step 4) | PR:34 |
| MTP head quality match (Avalon IQ3_K/IQ4_NL 1.48 GB, FR 65k) | Avalon +5-9 pts prose acceptance -> up to ~5% | exact | partial (own Q2_0 head; ref-head compare `src/core/mtp.cpp` ~575) | 2608 (step 3), 2614 (17.8) | PR:33; AV:122-142 |
| Draft vocab 42.7k proxy list vs 40.5k | misses 4.27% -> 3.68% offline; <1% | exact | yes `include/strata/core/mtp.hpp:68` | 2614 (17.8) | PR:36 |
| Strata `STRATA_DF2_GATE` calibrated verify-window gate | none (Mac) | exact | yes `include/strata/spec/window_rule.hpp:73` (DFlash2 only) | 874 (listed) | UR:177; ST:145 |
| DFlash2 drafter | +14% M5 Max, +7%/+1% gfx1201, slower Spark; HIP reverted to MTP | exact | yes `include/strata/core/dflash2.hpp:1` | 828 (measured worse), 2612 (17.7 reject) | UR:175; ST:139-146 |
| Adaptive / deeper MTP depth #27210, #27173 | up to 151 t/s recall @depth 12; +10% @depth 5, 2x 5090 | exact | partial (DraftPolicy, `--spec`) | 1352 (parked), 830 (`--spec 5/6` flat) | UR:117-118; LP:103-104 |
| Few-row MMQ for verify T 2-24 (#31533 = U1) | Ampere only | exact | no (HIP aborts) | 3416, 3517 (U1 OPEN) | UR:122; LP:108 |
| Suffix decoding / tree spec #26283 / #30931 | unmeasured | exact | suffix drafter yes (`--suffix-draft`) | 1261 (parked: watch #30931) | UR:125; LP:110 |

### Server / agent (serve/ is Python)

| lead | number + hw | exact/lossy | in our src | in plan (line) | source file:line |
|---|---|---|---|---|---|
| Checkpoint at last-user-end / tool-request invalidation (#30745, #24891) | reuse 91 -> 94%, save 3.6 -> 0.8 GB; V100 | exact | yes analog: `STRATA_CACHE_MESSAGE_BOUNDARY` (env.def:321), `--prompt-cache-tail` (`include/strata/program/conv_cache.hpp:43`) | 177 (item 11), 879, 2606 (17.0a) | UR:133-134,201; ST:321-322 |
| #30137 ref-counted checkpoints | grows with context; 3x Arc B60 | exact | N/A (own conversation cache / parking) | no | UR:132,196; LP:132 |
| Disk prompt cache #28092 | avoids ~70 s cold prefill @131k | exact | yes (conversation disk tier) | 1350 (superseded by item 39) | UR:137,209 |
| #30188 `cache_prompt:false` title requests do not evict agent prompt | none | exact | unchecked (serve has `strata_checkpoint` per request) | no | UR:140,221; LS:77 |
| Multi-slot #30992 / #30970 / #29738 | latency only | exact | n/a | 1255 (parked) | UR:136-138; LS:65-70 |
| Strata thinking skip for short answers (PR #5512) | 0/6 -> 5/6 answered @max_tokens 24; median 0.72 -> 0.50 s | lossy (behaviour) | no | 3978 (server PRs listed) | UR:179; ST:172 |
| Strata `POST /v1/tune` + `--conv-route` (PR #5516) | single-shot TTFT -42%; prefill +10.1% code w/ cache off | exact | no | 3978 (#5516 listed) | UR:178; ST:173-175 |
| Maya prompt memo (incremental re-tokenize of re-sent history) | follow-up -6% @115K; 3x V100 GLM | exact | yes `serve/loop_guard.py:365` (STRATA_PROMPT_MEMO) | no (not in plan text) | PM:145,177 |
| Keep-alive after streamed response #30492 | 1 TCP reconnect/turn saved | exact | unchecked | no | UR:145; LS:102 |
| Tolerant tool-call recovery (98.3% vs 92.9% of 1,462 turns) + Maya unclosed `<parameter=` | Strata corpus, Qwen3.8 | exact | yes `serve/frontend.py:1406` | 858, 1381 (done) | UR:180,208; ST:276-298; PM:136 |
| `env_to_user`, `effort_position: end` | stable prefix (~14K tok / 8-11 s per new day/folder) | exact | yes `serve/frontend.py:405`, `serve/runconfig.py:32` | 469 (done), 472 | UR:182; ST:310,320 |
| `reasoning_close_retry`, `answer_room`, loop guard, soft budget grace | unmeasured | exact | yes `serve/server.py:237,5034`, `serve/loop_guard.py` | 163, 472, 1362-1381 (done) | UR:181; PM:120-137 |
| Watchdog, `fit_max_tokens`, any-model-name | reliability | exact | yes `include/strata/core/progress.hpp:5`, `serve/runconfig.py:29` | no (inherited) | UR:184; ST:313-316 |

### Prose gap: tests and baseline hygiene

| lead | number + hw | exact/lossy | in our src | in plan (line) | source file:line |
|---|---|---|---|---|---|
| Control arm: Avalon with `GGML_CUDA_MOE_TAIL_SKIP=0` (its default 7 is lossy, bench left it on) | gap 3.6-8.4% may be lossy path | control | n/a (Avalon `moe-cache.cu:9677`) | 2608 (17.2 step 1) | PR:19,31; AV:153 |
| Control: Avalon head on matched IQ3_KS main (bench paired it with IQ2_XS) | unmeasured | control | n/a | 2608 (step 1) | PR:21; AV:20 |
| A-A `base_rep` arm + any-base-hash text check (abtest compares only base rep 1) | 6 distinct hashes in 6 runs (story); between-start CV ~5% | method | no (`tools/pancake/abtest.py:866` text_check; no base_rep) | 2608 (A-A named); any-hash check no | PR:45; AD:17-30 |
| Add the 5 prose prompts to `abtest.prompts()` | none | method | no (`tools/pancake/prose_gap.py` exists, abtest has no prose case) | 2608 ("no prose case yet") | PR:39; AD:49 |
| Profile fixed ~14 ms/window (1 ms ~ 5% decode; ~2.2 ms per extra draft) | derived, 2-point fit, night queue 7900 XT | profile | tools exist (gpu_profile.py, time_ceiling.py) | 2608 (step 2) | PR:13,32 |
| Greedy exactness check: `--spec-min-p 0.5` vs `0` vs `--spec 2` give identical text | none | exact claim | n/a | no | PV:63,133 |
| Lossy flags still in daily `flashnext-iq2_xs-fast-hip.json`: `STRATA_GR_RDNA_A8=1` (:98), `STRATA_PF_F16=2` (:88) | baseline not output-exact | lossy | yes (preset) | no (GR_RDNA_A8 has no plan hit) | PR:23; PV:103 |
| Rep budget: 3% needs ~12 starts/arm @CV 2.5%, ~45 @5%; state 5-6% bar otherwise | power formula | method | n/a | 2608 ("~12 starts per arm for 3%, ~45 at 5%") | PR:43; AD:59-91 |

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

| lead | number + hw | exact/lossy | in our src | in plan (line) | source file:line |
|---|---|---|---|---|---|
| Item 1: dump effective `STRATA_*` env + re-profile with per-stream gaps (map 48 mtp_select->copy and 83 copyBuffer->copy_i32_from_mapped pairs) | enabler; tool writes run_environment.json | n/a | gpu_profile.py (tools) | yes 2681, 2944 (stream re-profile still OPEN, 2734) | report.md:35; profile-environment.md:35 |
| Item 2 COMBINE_POLL1 (one elected poller per combine) | ceiling 18.9%; measured short -4.2%, text differs | measured lossy(!) | yes native_moe.cu | yes 2683; verdict DROP 3284 | report.md:36; design-worker-a.md:23-52 |
| Item 3 WAIT_FUSE 1 vs 0 | 0-3% est.; 12-rep: -2.0%/-0.2%, text differs | exact by class, observed text diff | yes verify.cpp | yes 2684; DROP 3683-3693 | report.md:37 |
| Item 4 FUSE_WAIT (with WAIT_FUSE off) | ceiling 0.3% (4320 nodes x 2.4 us); 12-rep -9.9%/-10.2% | exact by class | yes verify.cpp:441-462 | yes 2685; DROP 3690 | design-worker-a.md:80-101 |
| Item 5 `--spec 3/4/5` sweep | spec3 -3.2%, spec5 +1.2% (lossy, text differs) | greedy exact in theory; observed text diff | CLI | yes 2686; 3292-3293 (spec5 needs quality gate) | design-worker-a.md:105-124 |
| Item 6 ROUTE_FAST | ceiling 2.2% wall (78.1 ms), NOT 2.9% (that is % of kernel time); earlier +8.0% n=4 = noise | exact by design | yes native_router.cu:490 | yes 2687 (says ~2.9%: stale); dropped at confirm (TASKS:133) | design-worker-a.md:15,128-147 |
| Item 7 WAIT_COPY256 inert at `--spec 4`: wide copy needs plan n>512 words; plan=80*T+42 words, so acts only for T>=6; earlier +6.8% cannot be its effect | ceiling 0.3-0.4% | exact | yes verify_kernels.cu:1797 | partial 2688 (inert-at-T<6 finding: no) | design-worker-a.md:14,151-170 |
| Item 8 MTP_EAGER | ceiling 0.5% minus wasted steps; measured -3.6%/-3.9% | exact (code) vs def says "tokens change" | yes mtp.cpp:1863 | yes 2689; DROP 3291 | design-worker-a.md:174-195 |
| Item 9 merge 3 window-input copies via existing `copy_from_mapped_multi` | 0.04% (2 nodes x 288 windows) | exact | helper yes (elementwise.cu) | yes 2690, 2951 (not run, TASKS:149) | design-worker-a.md:199-219 |
| Item 10 pre-stage host records: premise stale (mtp.cpp:1821-1835 already stages); 16.9 ms pair likely `stage_source_R` D2D copy (mtp.cpp:2010) or sampled spec copies | ceiling 1.0% | exact | stage_source_R yes | item yes 2691; stage_source_R/stale-premise: no | design-worker-a.md:16,223-241 |
| Item 11 pool pinning: existing `--host-core first/last/sibling`, `--pool-workers 6/7/8/14`, `STRATA_POOL_MWAITX` arm; `--pool-affinity measure` is a NO-OP on non-hybrid 5700G (returns All) | ceiling 8.3% (293 ms waits) | exact (scheduling) | yes generate.cpp, pool.cpp | item yes 2692 (arm "measure vs alt" is wrong); host-core sibling 3036; pool-workers/MWAITX arms: no | design-worker-b.md:16,28,46-80 |
| Item 12 event + device wait for window-end sync | ceiling 2.6% (93 ms stalls), realistic ~0 (argmax read is true dependency) | exact | no | yes 2693 | design-worker-b.md:17,110 |
| Item 13 VERIFY_DEVICE_PLAN | <0.5%; measured -5.9%/-2.2%, text differs | exact by class, observed diff | yes | yes 2694; DROP 3294 | design-worker-b.md:18 |
| Item 14 ROUTER_FUSE w/o deferral | ~0.6-1%; TASKS: lossy, needs quality gate | check output | yes | yes 2695; TASKS:133 | design-worker-b.md:19,162 |
| Item 15 PIPED_MTP: inert without `--batch-mtp` + layer-split stages (generate.cpp:7194/7245/7406) | AMUSD 29% (paper excerpt) | claimed exact | yes | yes 2696, 2954 | design-worker-b.md:20,29 |
| Item 16 `hipStreamWaitValue32` replacing spin kernels; `hipDeviceAttributeCanUseStreamWaitValue`=1 on 7900 XT (ROCm core-10.0) | ceiling ~0.3% node cost, latency unchanged; must honour `release_gpu_waits` UINT32_MAX watchdog (verify.cpp:612-625) | exact | symbol referenced (3 files) | yes 2697, 2737 | untested_switches.md:188; design-worker-b.md:41,230 |
| Item 17 commit into window graph, device-side accept | <=2.6% + one launch | exact iff accept matches | no | yes 2698 | design-worker-b.md:22,257 |
| Item 18 fused argmax+accept (SonicSampler 2607.20475) | ~0 on greedy; host head sync verify.cpp:4228/4234 only for sampled | exact if tie-breaks match | no | yes 2699 | report.md:25 |
| Item 19 VERIFY_DEFER_CPU | +15.8% +-3.9 edit, short n.s.; DEFER_AUTO -1.2..-12.5% | LOSSY | yes | yes 2700; deferral OFF 2727 | report.md:17,53 |
| Item 20 Mirror-SD (2.8-5.8x, +30% vs EAGLE3) / PipeSpec (2.25 vs 1.5) / AMUSD (29%) / Speculative Aggregation | papers, datacenter GPUs, need 2nd draft model | claimed exact | no | yes 2701 | report.md:23 |
| Consumer-HW spec-decode caution (Apple silicon: best 1.61x, 3 of 5 configs slower) arXiv:2607.17283 | Apple M-series | exact (chi2 p=0.976) | n/a | no | report.md:25 |
| llama.cpp #23126: cross-model waits on shared Radeon iGPU drop draft 33-35 -> 3-5 tok/s (Vulkan) | Radeon iGPU | n/a | n/a | no | report.md:27 |
| Combine kernel is >=96% wait (5.6 us floor/call x 4032 = 22.6 ms = 0.6% wall): 18.9% is a pool-latency ceiling, not an arithmetic one | derived from gpu_sync note | n/a | n/a | no | design-worker-a.md:19 |
| Untested-switch inventory: 153 window-keyword `STRATA_ENV` (of 1017) - Speed 84, KeyIfSet 22, Key 12, Unsure 17, Inert 17, InConfig 1; ~20 have an A/B | - | - | conversation_disk_env.def | no (inventory not in plan) | untested_switches.md:78-133 |
| Still-coded-only, no A/B: ROUTE_PRIOR_AUTO, ROUTE_PRIOR_DEPTH, ROUTE_PRIOR_PROTECT/KEEP/SCOPE/MARKS, ROUTE_RESIDENT/_MTP/_RANKS, COPY_HIP_PRIO, GRAPH_PRIO, FUSE_HITS, GR_FUSED (non-finite suspect), FUSE_CONVAB, PREFILL_ROUTED_STREAM, SPEC_COST_TABLE, SPEC_PROB/_COUPLED (exact sampled spec), SPEC_CUM, KV_RING_OFF, ADAPT_NOWAIT/ASYNC_WAIT, MTP_* batch variants | - | mixed | all yes | partly: ROUTE_PRIOR_AUTO 2607; ROUTE_RESIDENT 1038; COPY_HIP_PRIO 1347; GRAPH_PRIO 2287; GR_FUSED 155; others no | untested_switches.md:94-133 |
| Engine does not print its switch state; add startup dump of effective STRATA_* | - | n/a | no | partly 2945 (tool-side json) | profile-environment.md:35 |
| Label mismatch: PIPED_MTP and OLD_SAMPLER are class Speed but change tokens; fixing needs conversation-disk key version bump (conversation_disk_tier.cpp:242-251) - owner decision | - | n/a | yes | no | untested_switches.md:184 |
| Config conflict: notes said DEFER_CPU in fast-hip preset; file has none | - | - | - | resolved by 2727 | design-worker-b.md:40 |
| hip_flag_order_test failing (fix proposed, not applied) | GPU suite 133/134 | - | test yes | yes 3295 | TASKS:130 |
| Stale MOD_PLAN status lines found by inventory: 233 (EXPDEC_DOWN measured +1.7% +-0.5), 1753 (DEFER_CPU_FOLD measured flat), 1759-1760 (WAIT_COPY256/ROUTE_FAST measured), 2623 (DEFER_AUTO measured worse) | - | - | - | stale lines | untested_switches.md:154 |

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
