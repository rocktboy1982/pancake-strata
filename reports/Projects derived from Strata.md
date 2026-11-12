# Projects derived from Strata: what pancake has already ported and what it still has to take

In November 2026 the Strata ecosystem reached **42,317 stars, 4,627 forks and 1,216 open issues**, with a release pace few projects have: v0.2.9 (31.10), v0.2.10 (02.11), v0.2.11 (05.11), v0.2.12 (07.11) and **v0.2.13 (09.11, engine 0.1.59)**. The base of our fork `pancake-strata` is `146df3c4` (v0.2.12, engine 0.1.58), so upstream moved one release ahead of us on the very day we wrote this report. We checked about **30 derived projects with real work in them**: 14 studied in depth and already used by pancake (tier 1), plus about 20 forks and derivatives found or reconfirmed in the live pass on 09.11, from which we have ported nothing yet (tier 2). On top of these come the ecosystem of quantization packs on HuggingFace, a closed competing engine (Infernix) and the coverage in blogs and Reddit, including a backlash against the hype and the author's debunking of a crypto scam. On its own ground, **pancake compares well**: on a single RX 7900 XT we measure ~112 tok/s decode after the Strata-rdna3 kernel ports, faster than Strata-rdna3 itself (109-119), than Avalon (95-100) and than stock Strata (75-77) in the same suites. What we lack is breadth: dual-GPU pipelines, multi-user scaling, backends outside CUDA/HIP, and a series of upstream kernel and startup patches that landed or matured after our base.

A few conventions for the whole report. "Fork" means any project derived from `Niko1221/Strata`, even if its history was rewritten. "Ported" points to the ledger in [MOD_PLAN.md][mp] (phases 2-14, with the "Not adopted" register at lines 1073-1286). Branch state was checked with git on 09.11: `port/faststrata-poly`, `exp/expert-prefetch`, `feat/grammar` and `feat/pattern-cache` are merged into the `port/rdna3-decode` work line (which is not yet merged into `main`); still unmerged are `feat/disco` (+1), `feat/bfla-hip` (+7), `feat/bitdecoding-hip` (+2) and `feat/contextpilot` (+2). Numbers measured by authors on other hardware are marked as such; nothing described here was measured on our machine unless stated explicitly.

## 1. Executive summary: the ecosystem and the three big gaps

Of the 4,627 forks, the vast majority are passive clones. Those with real work fall into three tiers, and the comparison with pancake comes down to three gaps:

1. **A correctness bug that touches our shipped code.** Upstream PR [#5655][pr5655] shows that `route_prior` leaks a sentinel expert ID and can write out of bounds on degenerate logits. We ship `STRATA_ROUTE_PRIOR` (the `turbo`/`turbo-hip` presets) and `STRATA_ROUTE_PRIOR_DEPTH`, so it is our bug too until proven otherwise. It is the only item in the report with a "stop and check first" flag.
2. **Startup and memory elasticity.** Our restarts pay **28-65 s just to read the 33 GiB expert arena** from ntfs3. Three independent solutions exist in the ecosystem and we have ported none: [#3509][pr3509] (`STRATA_ARENA_REUSE=1`, attaching a shared arena instead of re-copying), the parallel direct expert reads from Project Maya (O_DIRECT, 2-16 reader threads, **+26% decode on a RAM-poor PC**) and basalt's sleep-in-place (commit `64167ba73`). From the same family is the weight-based miss skip (`MAYA_GLM_MISS_SKIP_W`, **+7-13% decode at KL 0.037-0.052** on GLM).
3. **Dual-GPU and multi-user serving.** Hardin22/Strata-DualGPU showed **29 → 161 tok/s** on two GPUs with 32 GB RAM, and upstream issue [#1352][i1352] shows that the upstream merge is incomplete (upstream is still ~2x slower than the fork on a 4070 Ti Super + 5060 Ti). cobra91's three open PRs (#4522, #4534, #4549) are the rest. havok2-htwo/StrataQ38FN shows 43 patches that take a 4x3090 box to **~578 tok/s with 16 users (6.6x)** and cut re-reads on agent switches from **25 s to 1.5 s**. All of this sits behind phase 12 (the second GPU), but the PRs do not wait for us.

## 2. Tier 1: the big forks already studied and ported into pancake

For each: URL, license, what we took and what stayed unported.

### 2.1 Jarson1408/FastStrata and simplexai-labs/PolyStrata (MIT)

- URLs: [Jarson1408/FastStrata][faststrata] (commit `df91582`) and [simplexai-labs/PolyStrata][polystrata] (commit `a82999fd`). MIT license. PolyStrata is FastStrata's parent; both carry the secondary engine `src/poly`.
- **Ported and merged**: the whole `src/poly` engine, ported as `strata-poly` on the `port/faststrata-poly` branch (now merged), with `--attn-split`, `--graph-experts`, `--elastic-prefill`, `--lend-staging`, `--stage-next`, `--suffix-drafts`, `--cpu-multi`, `--kv-q8` and `--draft-vocab` from poly. Measured on the 3060: short chat **100.7 tok/s**, code edit **151.1 tok/s**, 41.8K prompt in **35.8 s (71.7 tok/s)**, against stock strata at 64.6 / 40.1 / 42.1 s.
- **Rejected**: adaptive MTP depth (poly's `Guesses` cost rule already covers it); the fused FA kernel for the prompt in the main engine (attention is 6.7% of a 42K prompt; parked until the prefill profile on the 7900 XT).
- **Not ported**: `--conversation-disk` (automatic disk sessions) exists only in strata-poly; the main-engine equivalent (MOD_PLAN item 39, restore in ~0.5-1 s) is not built. Tensor-core decode attention (`--attn-split`) is CUDA-only; there is no WMMA path for HIP/gfx1100.

### 2.2 nicejunjie/Strata-rdna3 (MIT)

- URL: [nicejunjie/Strata-rdna3][rdna3] (local remote `rdna3`).
- **Ported and merged** (branch `port/rdna3-decode`): the hyper-connection read family for RDNA (+Q8, A8, INJ_UP, FUSED), the wave32 multi-row GEMVs (`RDNA_Q6K`, `MMVQ_WAVE`), `COMBINE_CPU`, `WAIT_FUSE`, `ROUTE_PUBLISH`. Decode went 87-101 → 104-108 (kernels) → **~112 tok/s** (fusions).
- **Rejected**: long-context top-k (= upstream `SELECT_LONG`), `PF_GROUP_RES` (= upstream `group_resident`), `PA_KLDS` (upstream rewrote it as `STRATA_PA_RPF`), `RDNA_PAIR` (neutral), rdna3's expert profile and adapt settings (ours measured better: 121.2 vs 116.4), the zh60k draft vocabulary (92.5, worse) and the GDN decode megakernel (0.0 ms on a single GPU; the window waits for the CPU rows).
- **Not ported, all parked on phase 12 (second GPU)**: tensor-parallel decode (`src/core/tp.cpp`, `--tensor-parallel 2`; rdna3 measured **129-170 tok/s on 2x 7900 XTX**), expert transfer between GPUs / RCCL (`peer_experts.cpp`, `peer_transport.inc`), kfdq parking at idle (power 98+91 W → 5+13 W), the GDN megakernel revised under TP, and the gfx12 (RDNA4) re-measurement of the gfx11 tune ports (item 62).

### 2.3 jesdga95/basalt (MIT)

- URL: [jesdga95/basalt][basalt] (commit `a9468e5c`; a fork with history rewritten from Strata `822a5c783`; local remote `basalt`).
- **Ported and merged**: (a) the grammar engine, done on 08.11 on `feat/grammar`: the GBNF / JSON-schema / tool-call builders (`d64768699`), the per-request guard (`02e1006fe`), `Verifier::resample_row` (`c7813a6c3`), `--grammar on` in the engine plus the per-request `grammar=` key, with masked redraw in the verify window so MTP keeps drafting, and `serve/grammar.py`. Cost: 0.1-0.2% per window; shipped in the IQ2_XS presets. (b) The head kernels (owner request, done): `q5k_head.cu` ported for Q5_K heads (Q2_0/IQ3_XXS) and the same designs written for IQ4_XS: exact head GEMV plus a two-step 2-bit draft pick.
- **Rejected**: the 2-bit draft-head screen and K/V-append-beside-draft-query (the draft step is 2.5-3.2 ms of a 46-130 ms window, at most 1-2%); the `.basalt` pack format (needs Blackwell sm_120).
- **Not ported**: `--idle-sleep` / sleep-in-place (commit `64167ba73`, ~815 lines: drops and redoes the registration of the RAM copy, refills on wake). Parked, but it stays the only item that closes the `--vram-elastic`-refuses-with-pinned-complement gap.

### 2.4 mw00/project-maya (MIT)

- URL: [mw00/project-maya][maya]. MIT license. A Strata fork frozen at engine 0.1.27, specialized for Qwen.
- **Ported and merged** (the server side, the "Agent robustness from Project Maya" section in MOD_PLAN): (a) **the loop guard**, `serve/loop_guard.py`, on by default: detection of token loops with drop and a 1.15 repetition penalty, closing the thinking on the 3rd loop, stopping on the answer, notes and blocking for tools repeated after 3 notes, prompt collapse, keep-going nudges, `/metrics` plus Prometheus counters, tests in `serve/test_loop_guard.py`. (b) **The soft thinking budget**, `reasoning_budget_grace` (closing at the end of a sentence or line; Maya #613). (c) **Tolerant tool calls**: an unclosed `<parameter=` tag is malformed but recoverable, and a malformed call never raises an exception (#613/#706; tests in `serve/test_tolerant_tool_calls.py`).
- **Rejected**: running Maya itself (its Qwen engine is a stale 0.1.27 snapshot, 1.4-1.7x slower than current Strata, which already loses to pancake); resuming thinking after the end-of-turn token (both Qwen end-of-turn tokens are EOS for the engine; the empty-answer case is covered by #1053); DFlash2/DSpark/EXL3 (license or architecture); multi-GPU tensor-parallel; vision lending; the dashboard and the savings counter.
- **Not ported**: (a) **parallel direct expert reads** for the experts that do not fit in RAM: O_DIRECT with 2-16 reader threads probed per drive (`MAYA_EXPERT_DIRECT` / `MAYA_EXPERT_STAGERS`); Maya measured **+26% decode** on a RAM-poor PC. Our related item 38 (direct arena loader on Linux, estimated 28-65 s → 10-14 s at startup) is planned but not built. (b) **The weight-based miss skip** (`MAYA_GLM_MISS_SKIP_W`, +7-13% decode at KL 0.037-0.052 on GLM); we only have a rank skip. (c) Smaller items: a `--calibrate`-style startup sweep, drafter auto-keep, the Turbo preset as a launcher profile, a prompt memo (~6% of follow-up TTFT) and multi-request batching on a single GPU.

### 2.5 DevDock-AI/Speedy (MIT)

- URL: [DevDock-AI/Speedy][speedy]. MIT license.
- **Ported and merged**: Speedy's sm_86 presets as a starting config (item 8.1); the upstream opt-ins that Speedy measured: `STRATA_PF_GROUP_SCRATCH`/`_HC_REGION` (#4039), `STRATA_PREFILL_ROUTED_STREAM` (#4283), `STRATA_PROMPT_ATTN_IMMA`, `STRATA_KV_PREFETCH` (item 8.2, done or measured).
- **In progress**: `STRATA_VERIFY_DEFER_CPU` (#2689/#2690, item 8.3) is coded and is the biggest lossy lever we know (up to **1.45x** per KTransformers), but it produces **non-finite logits on HIP** with our ported kernels. A 6-run bisect is in the queue (PANCAKE_STATUS: Dev 100%, Test 10% after the ROUTE_PUBLISH bisect fix). The `STRATA_PSI_SHED` baseline is in the stage 1 group queue.
- **Rejected**: Speedy's INT8-IMMA attention variant (attention is 6.7% of the main engine's window).
- **Not ported**: the rank guard for deferral, `STRATA_VERIFY_DEFER_CPU_TOP` (item 28), coded (Dev 100%) but only 5% tested; the HIP non-finite fix gates both.

### 2.6 JustVugg/colibri (Apache-2.0)

- URL: [JustVugg/colibri][colibri]. Apache-2.0 license.
- **Ported (opt-in)**: cache-aware routing, `CACHE_ROUTE` → our `STRATA_ROUTE_PRIOR` (item 8.9). Measured: lambda 0.5 gives **+25% decode at 2.89% mismatch**; shipped as the opt-in `turbo`/`turbo-hip` preset. colibri's PILOT idea became the routing lookahead, now on by default on the GPU (`STRATA_LOOKAHEAD_GPU`, **+10% decode** on mmap'd experts, same choices).
- **Rejected**: lambda 2.0 (9.90% mismatch, 8 large-margin events); the lookahead predictor on CPU threads (-13% decode, replaced by the GPU variant).
- **Not ported**: nothing at the code level. The quality-gate decision for turbo-hip (P2a) is open: the preset exists, token forcing was INCONCLUSIVE (+0.4 points), Test 30%.

### 2.7 eddoursul/StrataFast (MIT)

- URL: [eddoursul/StrataFast][stratafast]. MIT license.
- **Not ported**: `--prompt-split` (two-stage prompts; item 8.7; **1.24-1.55x on long prompts** there). Blocked: it needs a second GPU, so it is parked in phase 12. Nothing else from this fork appears in our notes.

### 2.8 SensAI-PT/aviary-hy3 (Apache-2.0)

- URL: [SensAI-PT/aviary-hy3][aviary]. Apache-2.0 license.
- **Ported**: nothing attributed to this source appears in the ledger as a separate item; its relevant ideas arrived through the upstream opt-ins already covered under Speedy (the `STRATA_PF_GROUP_*` and routed-stream families).
- **Not ported**: no distinctive feature left unexamined. If a new release appears, check the kernel directory against the "Not adopted" register before assuming overlap.

### 2.9 JakeATX/llamAmpere (MIT)

- URL: [JakeATX/llamAmpere][llamampere]. MIT license.
- **Rejected / closed**: adaptive MTP depth (poly's cost rule covers it); the SJ-KVaRN-style 8-bit KV body and mb-MTP (VRAM does not buy decode here; the draft is 2-5% of a window).
- **Not ported**: nothing else attributed to this source.

### 2.10 freriadoc/Avalon

- URL: [freriadoc/Avalon][avalon]. A llama.cpp-based engine (not a Strata fork) that tracks upstream llama.cpp and competes with Strata on the same models; we studied it as a derivative of the same ecosystem and as a benchmark target.
- **Ported / adopted**: the gfx1100 hipBLASLt tuning table `100401` in our 7900 XT presets (done; **+38% prompt** provisional, Test 80%); hosted-MoE "finish together" (item 10.4) turned out to be already present in Strata as PcieAuto; the `--lookup-max-t 12` flag exists (A/B queued, +1% noise so far). We fast-forwarded the Avalon source from `c24c03d` to `88313b6` (48 commits: registered RAM arena, the hosted-MoE limit from link rate, IQ3_KS MMA MMQ, the tile fix for sparse-FA) and built it, with a binary rollback at `Avalon/bin-c24c03d-rollback`; the rebuild and bench are waiting because the model is missing (~20 GB short on D).
- **Rejected** (highlights; the full list is in "Avalon upgrades from upstream projects.md"): the upstream expert-cache line #29887/#30637/#30966/#31888/#31286/#30594/#27861 (our tier plus `--pcie-frac` already do this; #30594 breaks MTP verify); #31497 (contradicts the shared MTP cache); KVA approximate prefill (decode -11%); PLE-in-RAM (57 GB of RAM cannot hold the model + the 28.8 GB table + the reserve); the router's residency bias as a default (loops in agent sessions); Strata's gfx1100 switches (int8 WMMA attention, TSUM, expert layouts: neutral or negative on the 7900); long-memory ClassLRU scoring (under 0.5 points in replay); HIP graphs, Vulkan, turbo4/TurboQuant KV and the rest, tested with no gain.
- **Not ported**: the `--lookup-max-t 16` A/B and the hipBLASLt `100401` confirmation A/B (both queued, both need a quiet machine). And one item the Avalon report ranked "Add now" and for which no pancake counterpart is recorded anywhere: **llama.cpp #31751**, the single-pragma fix for MMQ J=32 register spills on RDNA3, **+36% on 17-32 token batches on gfx1100**, exactly the size of our verify windows. See section 4.4.

### 2.11 ewhacc/tandem.cpp and itlab-cl/tandem.cpp-amd

- URLs: [ewhacc/tandem.cpp][tandem] and [itlab-cl/tandem.cpp-amd][tandemamd] (the AMD variant).
- **Ported**: `STRATA_POOL_PREREAD` (idle pool workers pre-read the CPU experts of the next layer). The A/B is still pending. `MTP_DEFER` was already present.
- **Rejected**: the Q4_K requant of the MTP head (our head is ~2 bits, 787 MB; the requant buys nothing).
- **Not ported**: nothing else identified.

### 2.12 cutlerbenjamin1-cmd/qwen38-megakernel (AGPL)

- URL: [cutlerbenjamin1-cmd/qwen38-megakernel][q38mk]. AGPL license. Ideas only; we copy no code because of the license.
- **Taken as ideas**: the draft vocabulary experiments (full/en/zh60k) all measured flat or worse; the GDN replay idea was already present.
- **Not ported**: nothing. The AGPL license makes porting code a non-goal.

### 2.13 Smaller sources audited alongside tier 1

- **gpu-unlock**: measured, not adopted. MMA with FP32 sums reached 49.7 TFLOPS, but decode was unchanged; a 5.3K prompt dropped 7.6 → 6.0 s. Closed for decode purposes.
- **SGLang**: speculative sampling ported into the engine (`STRATA_SPEC_PROB`, measured; the default path is as fast or faster); the GPU token mask for grammar is done (phase 6); draft trees rejected (GDN state copies plus expert wakeups; we follow upstream #30931).
- **ContextPilot** (`feat/contextpilot`, unmerged, +2): coded (lossy block dedup, opt-in, exact cache-aware admission), then rejected: **1.6% savings** on 40 real opencode sessions, and non-prefix reuse breaks the GDN state. It stays off.
- **BFLA (the CityU paper)** (`feat/bfla-hip`, unmerged, +7, HIP only): block-sparse prompt attention, measured end to end on Qwen3.6 UD-Q4_K_XL: a 128K prompt went 386.9 → **1,219.8 tok/s sparse (3.15x)** → 771.2 tok/s dense; recall 48/48, KL 0.0014/0.0045. The merge decision is pending (Test 10%).
- **BitDecoding** (`feat/bitdecoding-hip`, unmerged, +2): `STRATA_ATTN_WMMA`, decode attention on HIP gfx11 matrix cores, measured with the model plus token forcing; parked as non-default (attention is ~3% of decode; ~2-3% confirmed).
- **DISCO** (`feat/disco`, unmerged, +1): a learned stop rule for the MTP draft chain (opt-in). The SpecFit missed-experts feature (item 36) is assigned to this branch, not started.
- **The university paper series** (the Stanford/MIT/Harvard/USA/Europe/Asia/Canada/Pacific reports): the per-level thinking cap is done (`5dd620fb9`); the sparsity masks (Berkeley gate/TEAL/CATS/CHESS/SpAx) rejected offline (Flash-Next's FF-640 experts are dense: gate cos 0.844 at 70%); XAttention replaced by BFLA; the predictor-recall item closed (19-21% coverage; promotions do not pay). Pending: learned thinking stop (item 21), effort defaults (12), loop-guard counters (33), the rumination signal (34), the restore-equivalence test (35), GDN chunk 64 (18), the AVX2 IQ1_M kernel (24, 0 on Q2_0), the Coder variant A/B (25) and the semantic oldString constraint (26).

## 3. Tier 2: new and active forks, not yet ported into pancake

Found in the live pass on 09.11 through cross-references in issues and PRs, probes of author profiles, code searches for Strata's distinctive symbols (`STRATA_`, `native_mmvq`, `qsa_scores`, `combine_cpu`) and web searches. Star counts are from that day.

### 3.1 gputier/StrataGP

- URL: [gputier/StrataGP][stratagp] (9 stars).
- **What it has**: 16 performance/precision patches packaged under `docs/perf/`, measured against upstream `f1b1d96` on an RTX 5090: **+5-15% decode and +42-49% prompt**.
- **Why it matters**: the largest single pack of ready-made kernel patches we have not examined. The prompt-side gains are the interesting part, because our prompt path is the one that degrades after long contexts.
- **Action**: diff the 16 patches against our base; expect overlap with the bahooo22 cluster (#5595/#5467/#5468) and with our rejected list (int8 WMMA attention). Port what survives.

### 3.2 havok2-htwo/StrataQ38FN

- URL: [havok2-htwo/StrataQ38FN][strataq38fn] (1 star).
- **What it has**: a series of **43 upstream-ready patches** on v0.1.43, plus a vLLM-compatible facade and a dashboard. Measured by the author: 4x3090 with 16 users at **~578 tok/s (6.6x)**, RTX 5090 **+16% decode** single-user, and re-reads on agent switches cut from **25 s to 1.5 s**.
- **Why it matters**: the re-read number on agent switches is exactly our pain (our opencode data shows ~18% of turn time in re-reads). The multi-user part is irrelevant to our single-slot server today, but it is the best-documented serving-scaling reference in the ecosystem.
- **Action**: read the re-read patch first; it can be combined with our message-boundary checkpoints (item 11) and with #5643 below.

### 3.3 bnkaiteam/Juud_engine

- URL: [bnkaiteam/Juud_engine][juud] (2 stars).
- **What it has**: two CPU decode tweaks: shortening the worker spin after the expert batch, and skipping activation quantization for GPU-resident tokens. Measured by the author **+9.6-15.6% decode** on an RTX 4090 (HF Forums post).
- **Why it matters**: our pool already runs at ~80% of the measured DDR4 bandwidth, so shortening the spin is exactly the kind of overhead cut that can still show up on edit windows, and the activation skip for resident tokens is free whenever our tier keeps experts resident.
- **Action**: port both as opt-in flags; A/B against the defaults from our pool sweep.

### 3.4 Hardin22/Strata-DualGPU and cobra91/Strata-DualGPU

- URLs: [Hardin22/Strata-DualGPU][dualgpu] (84 stars) and cobra91's follow-up fork.
- **What they have**: the two-GPU pipeline for 32 GB RAM machines: Swift IQ2_XS went from 29 to **161 tok/s** on code. Partly merged upstream: issue [#1352][i1352] shows upstream is still ~2x slower than the fork on a 4070 Ti Super + 5060 Ti, so the merge is incomplete. The pieces that reached upstream: #848 (resident RAM mode on a layer split plus `--vram-reserve-later-mib`, shipped in 0.1.40), #859 (pipelined verify windows, **+16% average, +30% copy-heavy**), #876/#905 (async tier). Combined effect on a 4060 Ti + 5080 with 32 GB RAM: 29 → ~84+ tok/s.
- **The unmerged rest** (cobra91's open PRs to upstream): [#4522][pr4522] (thinning dense weights at `--layer-split auto`), [#4534][pr4534] (pricing a pipelined split by its slowest stage), [#4549][pr4549] (draft-head space kept out of the last card's cache).
- **Why it matters**: this is the phase 12 manual, already written and measured. Our 20 GB of VRAM keeps only 44% of the expert pairs resident; a second GPU is the only lever that changes that.
- **Action**: when the second GPU arrives, start from #4522/#4534/#4549 and from the pipelined verify windows in #859, not from a new design.

### 3.5 MistVVK/XeStrata and NeneNyann/Strata-intel-sycl-windows

- URLs: [MistVVK/XeStrata][xestrata] (334 stars), [NeneNyann/Strata-intel-sycl-windows][nenenyann] (10 stars).
- **What they have**: XeStrata is the multi-vendor build without CUDA: SYCL/Level Zero with XMX paths; `xe0.2.2-1` beats upstream sycl 0.1.45.3 on an Arc Pro B50 (issue [#4506][i4506]). NeneNyann packages Windows SYCL builds with bundled oneAPI runtimes and GPU vision.
- **Why it matters**: low priority. We are HIP/CUDA. Relevant only if we ever target Intel GPUs, or as a reference for SYCL kernel ideas that also exist in HIP form.
- **Action**: watch only.

### 3.6 lighttransport/Strata

- URL: [lighttransport/Strata][lighttransport] (20 stars), branch `glm53f`.
- **What it has**: a native GGUF backend for GLM-5.3-Flash inside Strata, ~29 tok/s on 2x V100 and ~7.5 tok/s on a Threadripper (CPU). It motivated feature request #2067 and now upstream PR [#5545][pr5545] (the `strata-glm` engine path, quantization pipeline, multimodal bring-up).
- **Why it matters**: we run GLM-5.3-Flash ourselves (the sessions for this report run on it). A native engine path would replace the fallback used today.
- **Action**: we follow #5545; when it merges, we rebase the fork on it and re-measure our GLM serving path.

### 3.7 JChen-318/gaudi2-single-card-strata

- URL: [JChen-318/gaudi2-single-card-strata][gaudi2] (10 stars).
- **What it has**: a native Intel Gaudi2 backend (`hpu/`) with its own TPC kernels; Qwen3.8-Flash-Next IQ4_XS fully resident; measured by the author **126.8 tok/s chat decode at 262K** context (issue [#4322][i4322]).
- **Why it matters**: directly, not at all (we have no Gaudi hardware). For the ecosystem it matters as proof that the Strata engine ports beyond GPU vendors.
- **Action**: watch only.

### 3.8 CC-David-CC/Private_Context_COW

- URL: [CC-David-CC/Private_Context_COW][cow] (1 star).
- **What it has**: copy-on-write private context, discussed in upstream issue [#3865][i3865], self-described as "stable enough for testing".
- **Why it matters**: our parked conversations live only in RAM and a restart loses them; COW context is one of the ideas in the memory family (next to the shared immutable host KV prefixes in #5623) that could cut the 5.4-6.1 GiB spent on 4 parked conversations.
- **Action**: read the thread; compare with our item 3 (multi-state store) before building anything.

### 3.9 alesha-pro/Strata

- URL: [alesha-pro/Strata][alesha] (3 stars).
- **What it has**: the port of Mirai's 2-bit trellis "S" quantization patch to Strata 0.2.0 (upstream PR [#3384][pr3384]), plus abliterated packs on HuggingFace.
- **Why it matters**: our head is already ~2 bits (787 MB), but trellis quantization of the body is unexamined. If it shrinks the CPU-resident experts at the same KL, it multiplies everything the DDR4-bound pool does.
- **Action**: evaluate the quantization offline (perplexity/KL on our model) before any kernel work.

### 3.10 Tools, tuning logs and the competitor

- **Vop3p/dual-6900xt-flash-next-tuning**: [repo][vop3p] (3 stars). A standalone tuning log for llama.cpp → Strata on 2x RX 6900 XT. Same author as PRs #4005 and #5475 below, so the repo is a useful field notebook for HIP behaviour on RDNA2/3. Watch.
- **xyzzing/kernelbench-hip** and **xyzzing/cupola**: [KernelBench ported to HIP/ROCm][kernelbench] and [an auditable inference supervisor for Radeon][cupola]. The same author opened #5535 (int8-WMMA prompt attention on gfx1100). The supervisor idea overlaps with our loop-guard metrics; KernelBench-HIP is a handy harness if we start writing our own kernels. Watch.
- **lwYeo/llamaPeek**: [Windows tray monitor][lamapeek] for llama.cpp-compatible servers. We run Linux; cosmetic. Watch.
- **Xolotl** (formerly Strata-DSV4.1): named in the stringers.io ecosystem summary; not checked individually. Probably a DeepSeek V4.1 specialization. Watch.
- **Infernix** (2026.10.09.2): **not a fork, a competing engine**. An NVFP4 recipe-C artifact plus an mmap'd n-gram volume: measured by the author **+23% decode at 262K and +9% at 512K** against Strata 0.1.41 on a single RTX 5090 (149.0 vs 120.6 tok/s), working vision, cold start ~28 s (prismix.dev, r/LocalLLaMA). The number to beat if we claim the lead on long-context decode. We follow the release notes.

### 3.11 Named in community coverage, not checked individually

PolyStrata (863 stars per the summary; adds a second engine for GLM/DeepSeek/Gemma), strata-nvfp4, strata-mlx (Mac), Strata-V100, eddoursul/Lamina, momoe, talus-public (Strix Halo), FastStrata. Also in the coverage: the SEO squatter `stratallm.org` and the author's debunking of the crypto scam on X (19.10). None has a verified feature we lack; check before citing.

### 3.12 External derivatives (outside GitHub)

- **HuggingFace quantization packs**: [Maximilian228/Qwen3.8-Flash-Next-NVFP4-Q8-Mix-Strata][hfmax] (issue [#4063][i4063], RTX PRO 5000 48GB); the DavidAU Turbo-Brilliance GSQ-RCO packs (issue [#2841][i2841]); alesha-pro's abliterated/trellis packs; ISTA-DASLab GSQ-RCO (Q2_0 → IQ3_S plus a Coder variant with 256/512 experts, a 0.91 GB BF16 mmproj); Unsloth Dynamic v3.1 (re-uploaded on 04.11; issue [#5297][i5297] asks for the UD-IQ2_XS variant, ~9 GB smaller; UD-IQ1_S does not load because of mixed expert formats within a layer); agentionai's Gyro-S/Gyro-M (issues [#5176][i5176], [#5584][i5584]; Gyro-M TQ2_0 on an RX 7900 XTX; [#3784][i3784] Gyro-S OOM on Mac).
- **Community/blog**: a dev.to quickstart for 64 GB (rosgluk, 07.11); a hype-reaction piece on traictory.com; an explainx.ai guide; the HF Forums post about Juud_engine.
- **Reddit**: the original r/LocalLLaMA post by KnownAd4832 (65 tok/s on 12 GB VRAM), the "stop the Strata spam" backlash thread and the dual-5090 WDDM field data in the comments on issue #642.

## 4. Tier 3: upstream Niko1221/Strata (v0.2.12 → v0.2.13) and the valuable PRs

### 4.1 What v0.2.13 changed (and what it means for our base)

v0.2.13 came out on 09.11 (engine 0.1.59), two days after our v0.2.12 base (`146df3c4`). The commits on `main` in the last 48 hours run from `f33c8c305` to `8a9feaedd`:

- **The shape-engine architecture for Qwen4** (`f-0213-shape`): a new-shape Qwen4 model ships as its own engine component `strata-q4-<shape>`, next to the Strata engine, selected by the model. `strata --geometry` prints the compiled kernel geometry (`flash_next` is the default), and the geometry check at install runs against a signed manifest. It is an architecture change: future models may stop being drop-in GGUFs for old engines, which affects how we rebase.
- **Per-device DFlash2 defaults** (`e3bd5cfde`): Strix Halo and Blackwell cc 12.0 with 48 GB or more get a draft window of 6 → 8, and conf 0.55 for IQ3_S on that Blackwell class. Measured by the author: Strix UD-Q4_K_XL x1.052, l620 IQ3_S x1.028. We measured DFlash2 and closed it (115.9 tok/s on short chat but 84.3 after a long prompt; an upstream crash on QFN), so it matters only if we reopen it.
- **The Strix Halo G16 fix** (`46ea3947e`): no 2560 MiB Windows VRAM reserve floor on integrated HIP devices (UMA); setup writes `--vram-reserve-mib 700` on single-card gfx1151. Start with the old config 77 s, 20.5-24.3 tok/s. Not our hardware, but the UMA logic may touch shared setup code.
- Smaller: unified fit labels ("fits" / "may run slower"), a fix for the data folder on an external drive, NVFP4-QAD sized by its 76.8 GB pack, and a feed-model install gate (`tools/feed_model_install_gate.py`).

### 4.2 The register of open PRs, by cluster (all unmerged as of 09.11)

**Kernels and MoE:**

| PR | Author | Mechanism |
|---|---|---|
| [#5595][pr5595] | bahooo22 | q36: grouped down-expert kernel for Q4_K (fmt 12) |
| [#5467][pr5467] | bahooo22 | q36: flash attention on mma.m16n8k16 for the prompt (K/V fp16) |
| [#5468][pr5468] | bahooo22 | q36: layer-major prefill that stages a layer's experts once per N chunks ("super-chunk") |
| [#5466][pr5466] | bahooo22 | q36: need/free MiB reporting on expert-buffer/staging OOM |
| [#5535][pr5535] | xyzzing | int8-WMMA prompt attention on RDNA3/gfx1100 (RX 7900 XTX), with recipe, validation and results |
| [#4005][pr4005] | Vop3p | prefill: opt-in sparse copy events + slot-wait dedup; Windows HIP prompt **+36% @4K** (closed, unmerged) |
| [#5475][pr5475] | Vop3p | `STRATA_HC_PERSIST_Q8`: int8 copies for the persistent hyper-connection read |
| [#4876][pr4876] | xrip | `STRATA_GDN_TC=2`: tensor-core GDN recurrence on sm_75, 2.2x faster |
| [#3277][pr3277] | c3c | extends the native-IQ prefill dispatch to SM75 (Turing) |
| [#4854][pr4854] | Keoian | native Q2_0 rows on AVX1 (pre-AVX2 CPUs) |
| [#4886][pr4886] | Spinoza0 | Metal: IQ1_M expert gate/up in GSQ packs (missing type-29 instantiations, #2537) |
| [#5518][pr5518] | rwkeyes | HIP: the `alignas(16)` int4 staging arrays do not compile (two sites from #778) |
| [#2392][pr2392] | skuridin | hipBLASLt tuning table for gfx1101 (RX 7800 XT, hipBLASLt 1.2.0); open since 14.10 |
| [#5540][pr5540] | Captain-Tripps | sycl: counts helper/peer caches before the adaptive tier; helper-mode decode 32 → 61 tok/s |
| [#5562][pr5562] / [#5473][pr5473] | lwYeo / YamineRL | shrinking the fused layout asks each layer's path; link stubs for `fused::want_dm_f16` |
| [#3925][pr3925] | schmittcn | q36: loading GGUFs with K-quant or IQ4_XS `ssm_alpha`/`ssm_beta` |

**The correctness bug (see P0 below):**

- [#5655][pr5655] (WilliamStone): `route_prior` leaks a sentinel expert ID and can write out of bounds on degenerate logits.

**The DeepSeek V4.1-Flash cluster (R-Okauchi, all open):**

- [#5471][pr5471]: faster Q4_K/Q5_K decode GEMVs, plus light experts at the same bits.
- [#5470][pr5470]: `DS4_PROF_WIN`, a decode profile of the drafter's verify windows.
- [#5472][pr5472]: DGX Spark `STRATA_DS4_DISK_SKIP_RANK`/`PROMPT_SKIP_RANK`.
- [#5453][pr5453]: DGX Spark `DS4_UMA_TIER=1`, a single expert tier that the GPU reads in place.
- [#5566][pr5566] (yeyeyeason): opt-in cold store for the expert arena (the source-read half of #5391).
- [#5596][pr5596] (AlexGabbia): `STRATA_DS4_CACHE_ROUTE`, a selection bonus for experts already in VRAM/RAM.

**Server and serving behaviour:**

- [#5516][pr5516] (rwkeyes): in-place retune (TUNE) plus routing the conversation cache by traffic shape.
- [#5643][pr5643] (kelvincht): restoring the reasoning the client dropped, so the next turn reuses the cached prompt prefix.
- [#5623][pr5623] (shoucandanghehe): sharing immutable host KV prefixes between streaming sessions.
- [#3509][pr3509] (Jerry-Lee661): `STRATA_ARENA_REUSE=1`, attaching a shared arena instead of re-copying.
- [#4625][pr4625] (GoodSlopBadSlop): `STRATA_POOL_SPIN_US=0` really means zero.
- [#5512][pr5512] (rwkeyes): skipping the thinking block on short direct-answer prompts (opt-in).
- [#3971][pr3971] (macintog): keeping the thinking tokens generated past a budget cut.
- [#5628][pr5628] (gener-vscode): streaming tool arguments at JSON token boundaries.
- [#4777][pr4777] (code-martin): dynamic `route_prior` via the API/per request.
- [#5561][pr5561] (njy1991): sequential read-ahead on the Linux file tier (`STRATA_EXPERTS_SEQ_READAHEAD`).
- [#5593][pr5593] (krispgece): `STRATA_PROMPT_DIRECT=1` without a RAM budget.
- [#4972][pr4972] (01xjw): Linux file-tier defaults off when the GPU caches hold all experts.
- [#5637][pr5637] / [#5437][pr5437]: inhibiting sleep during inference (Linux / cross-platform).
- [#5588][pr5588] (noahark): `--ple-gguf`, a side-loaded PLE table wins name resolution.

**New model families and platforms:**

- [#5517][pr5517] (archadrian): the qwen3next family, a geometry-derived trunk plus grouped-head GDN support (the Groot lane bit-identical).
- [#5545][pr5545] (zhuxm-ai): the GLM-5.3-Flash engine path (`strata-glm`), quantization pipeline, multimodal bring-up.
- [#5638][pr5638] (xiezhipeng-git): a Qwen3.8-Flash-Next fine-tune for +0.40 GiB VRAM (zeroth-order training).
- [#5592][pr5592] (DobricLilujun): Jetson AGX Thor (sm_110), aarch64 CUDA machine code.
- [#3607][pr3607] (demetree): native Windows Intel Arc SYCL support (open since 21.10).
- [#5647][pr5647] / [#5644][pr5644] (kelvincht): elastic expert arena on Metal (tracks free memory) plus reserved K/V pools instead of allocating at start.
- [#4522][pr4522] / [#4534][pr4534] / [#4549][pr4549] (cobra91): the unmerged Strata-DualGPU pieces (see 3.4).

**Qwen3.6 issues still open upstream** (relevant because Qwen3.6-35B-A3B is one of our daily models):

- [#5069][i5069]: Qwen3.6-35B-A3B has no `--kv int8`/`--kv-resident` and no long-context decode path (Flash-Next has them).
- [#2464][i2464]: Qwen3.6 decode drops with context (V100: 109 → 14.6 tok/s at 114K).
- [#5609][i5609] (closed on 08.11): the q36 MTP block loads but never drafts. Fixed upstream; check whether our base has the fix.
- [#5551][i5551]: q36 on the qwen35moe 19B-A3B shape, three localized walls.
- [#1402][i1402]: Qwen3.6 plus Ornith-1.5-35B-A3B as a small-model line for 8-12 GB cards.

### 4.3 What upstream has already merged (the late-September wave, for orientation)

#43 multi-token i-quant row kernels on AVX2; #109 batched verify-window kernels (+~10% decode, bit-identical); #21 Q4_0 KV cache with Hadamard FWHT-256 rotation; #54 GSQ-RCO-Coder pruned experts (256/512); #8 conversation cache; #62 hybrid-CPU pool workers (P-cores plus half of the E-cores); #67 OrcaRouter BF16 packing; #89 bulk-read expert arena (the MSVC 4095-byte ifstream split); #81 runtime accept of sm_80+; #122 llama.cpp discovery endpoints; #118 routing traces; #107 reporting of cached prompt tokens; #126 Task Scheduler 24x slower at expert loading; #42 the MEM_LARGE_PAGES privilege; #44 a real H2D bandwidth probe for `pcie_frac`; #64 hard-fail for the parity tool. Most predate our base or are already in it; the list matters as a map of what the project merges.

### 4.4 The only llama.cpp item

**llama.cpp #31751** ([PR][pr31751]): a single-pragma fix for MMQ register spills on RDNA3, **+36% on 17-32 token verify batches on gfx1100**. Our verify windows are exactly this size. The Avalon report ranked it "Add now" and no pancake counterpart is recorded anywhere. Nobody has checked whether Strata's MMQ kernels or ours have the spill pattern. It is the cheapest unexamined kernel item in the whole report.

## 5. Prioritized gap analysis: what pancake lacks and what we take next

Ranked by (impact x evidence) within each priority class. Items marked "ours" point to MOD_PLAN item numbers.

### P0: Bug fixes and risk-free gains (do these first, in this order)

1. **Check #5655 in our tree** (`route_prior`: sentinel expert ID leak + OOB write on degenerate logits). We ship `STRATA_ROUTE_PRIOR` (turbo/turbo-hip) and `STRATA_ROUTE_PRIOR_DEPTH`. If our base `146df3c4` predates the fix, we have the bug. Read the PR, reproduce on degenerate logits, port the fix regardless of upstream state.
2. **Check that the #5609 fix for q36 MTP is in our base** (the MTP block loads but does not draft; fixed upstream on 08.11). One diff, five minutes.
3. **The spill-pattern check from llama.cpp #31751**: look in our MMQ kernels for the J=32 register spill pattern; if present, one pragma, **+36% at our verify batch sizes on gfx1100**. Never evaluated.
4. **#4625** (`STRATA_POOL_SPIN_US=0` really zero): trivial, and we tune the pool spin.
5. **#5466** (need/free MiB reporting on expert-buffer/staging OOM): our arena builds trigger systemd-oomd; better OOM messages would have saved hours.
6. **#5518** (the HIP compile fix for `alignas(16)` int4 staging): two sites; keeps future HIP builds honest.

### P1: Single-GPU decode and prefill gains, RDNA3/CUDA

1. **#4005** (Vop3p): sparse copy events for prefill + slot-wait dedup, **+36% Windows HIP prompt @4K**. Closed unmerged, so the code has to be revived. Our prompt path is the one that degrades after long contexts.
2. **The bahooo22 cluster #5595/#5467/#5468**: grouped Q4_K down-expert kernel, prompt FA on mma.m16n8k16, layer-major super-chunk prefill. All q36, all on our daily model.
3. **The 16 StrataGP patches** (gputier): +5-15% decode, +42-49% prompt on an RTX 5090. Diff against our base; expect overlap with the bahooo22 cluster and with the int8-WMMA attention we rejected.
4. **The two Juud_engine CPU tweaks**: shortened spin after the expert batch, skipping activation quantization for GPU-resident tokens, +9.6-15.6% decode on a 4090. Our pool is bandwidth-bound, so overhead cuts have the right shape.
5. **#5475** (`STRATA_HC_PERSIST_Q8`): int8 copies for the persistent hyper-connection read. Check overlap first: our rdna3 port already has the hyper-connection read with Q8; the persistent variant may be new.
6. **#5535** (xyzzing): int8-WMMA prompt attention on gfx1100, with recipe and validation. We rejected int8-IMMA attention for the main engine (attention: 6.7% of the window), but this is on the prompt side, on our exact GPU, with a validation recipe; cheap to measure, low expectations.
7. **#2392** (skuridin): the gfx1101 hipBLASLt tuning table. We already have the gfx1100 table `100401`; this extends coverage to 7800 XT-class cards.
8. **#5471** (R-Okauchi): faster Q4_K/Q5_K decode GEMVs. Our DeepSeek V4.1 path benefits directly if the kernel shapes match.
9. **#3925** (schmittcn): loading q36 GGUFs with K-quant or IQ4_XS `ssm_alpha`/`ssm_beta`. Removes a class of packs that do not load today.

### P2: Startup, memory and agent-server gains

1. **#3509** (`STRATA_ARENA_REUSE=1`): attaching an expert arena shared across restarts instead of re-copying. Our restarts pay 28-65 s to read the 33 GiB arena on ntfs3; this attacks exactly that, and it combines with item 38 (direct arena loader on Linux, est. 10-14 s) as an either/or.
2. **Maya's parallel direct expert reads** (`MAYA_EXPERT_DIRECT`/`MAYA_EXPERT_STAGERS`, +26% decode on a RAM-poor PC) plus our item 38: the startup decision from 08.11 (oomd vs `STRATA_RAM_KEEP_GIB=20`) must be made together with this.
3. **#5643** (kelvincht): restoring the reasoning dropped by the client, so the next turn reuses the cached prompt prefix. Our opencode data shows ~18% of turn time in re-reads; it is one of the few server patches that attacks them directly. Gate it on our restore-equivalence test (item 35).
4. **#5516** (rwkeyes): in-place TUNE plus routing the conversation cache by traffic shape. We closed upstream `/v1/tune` as low value, but the cache-routing half fits our `STRATA_ROUTE_PRIOR` work.
5. **Maya's weight-based miss skip** (`MAYA_GLM_MISS_SKIP_W`, +7-13% decode at KL 0.037-0.052 on GLM): we only have a rank skip. Quality-gate it like turbo-hip.
6. **#5566** (yeyeyeason): opt-in cold store for the expert arena. Memory elasticity for the arena; it fits the elastic-arena idea in #5647/#5644 (Metal, but the design ports).
7. **#5623** (shoucandanghehe): immutable host KV prefixes shared between sessions. The 4 parked conversations cost us 5.4-6.1 GiB; shared prefixes are the cheapest compression available.
8. **FastStrata/poly's `--conversation-disk` on the main engine** (item 39): restore in ~0.5-1 s after a restart; FastStrata measured a 70K conversation resumed in 7.2 s instead of 83 s.
9. **basalt's sleep-in-place** (`64167ba73`): parked, but it stays the only item that closes the `--vram-elastic`-refuses-with-pinned-complement gap.
10. **#5561** (`STRATA_EXPERTS_SEQ_READAHEAD`): parked in our ledger (only if experts fault from NVMe; the arena removes the faults). Revisit only if item 38 slips.
11. **Smaller server items**: #5593 (`STRATA_PROMPT_DIRECT=1` without a RAM budget), #4972 (file tier off by default when the GPU caches hold all experts), #5637/#5437 (inhibiting sleep during inference), #5588 (`--ple-gguf`), #4777 (dynamic per-request `route_prior`), #3971 (thinking tokens kept past the budget cut), #5628 (streaming tool arguments at JSON boundaries), #5512 (thinking skip on short direct-answer prompts; we called it heuristic, upstream it is still opt-in).

### P3: Unported features from the known forks (quality-gated or parked)

1. **`STRATA_VERIFY_DEFER_CPU` on HIP** (the Speedy port, item 8.3): coded, up to 1.45x per KTransformers, blocked by non-finite logits; the 6-run bisect is in the queue. Plus the rank guard `STRATA_VERIFY_DEFER_CPU_TOP` (item 28, 5% tested). The biggest lossy lever we hold.
2. **The turbo-hip quality-gate decision** (the colibri port, item 7): the preset exists, token forcing INCONCLUSIVE, Test 30%. Owner decision.
3. **The branch merge decisions**: `feat/bfla-hip` (3.15x on a 128K prompt, recall 48/48), `feat/bitdecoding-hip` (~2-3%, parked as non-default), `feat/disco` (+SpecFit item 36, not started) and the `port/rdna3-decode` → `main` merge. All owner decisions, all pending.
4. **Mirai's trellis "S" quantization, ported by alesha-pro** (#3384): evaluate offline first (section 3.9).
5. **#4876** (`STRATA_GDN_TC=2`, tensor-core GDN recurrence on sm_75, 2.2x): CUDA only; relevant for our 3060 path.
6. **#3277** (native-IQ prefill on SM75) and **#4854** (Q2_0 rows on AVX1): wider hardware coverage; low priority for us.
7. **#5545 / #5517**: the GLM-5.3-Flash engine path and the geometry-derived qwen3next trunk. We follow both; they change how we serve GLM and future Qwen families.
8. **The rest of the DeepSeek V4.1 cluster**: #5470 (the `DS4_PROF_WIN` profiler), #5596 (`DS4_CACHE_ROUTE`), #5453/#5472 (UMA tier and skip-rank for DGX Spark; the skip-rank idea can generalize to our file tier).
9. **Maya leftovers**: `--calibrate`-style startup sweep, drafter auto-keep, prompt memo (~6% of follow-up TTFT), multi-request batching on a single GPU.
10. **Pending items from the university series** (ours): learned thinking stop (21), effort defaults (12), loop-guard counters (33), the rumination signal (34), the restore-equivalence test (35), GDN chunk 64 (18), the AVX2 IQ1_M kernel (24), the Coder variant A/B (25), the semantic oldString constraint (26).

### P4: Dual-GPU and the phase 12 gains (blocked by hardware, not by research)

1. **Strata-rdna3's tensor-parallel decode** (`src/core/tp.cpp`, `--tensor-parallel 2`; 129-170 tok/s on 2x 7900 XTX), expert transfer between GPUs / RCCL, kfdq parking at idle (98+91 W → 5+13 W) and the GDN megakernel revised under TP.
2. **The remaining Strata-DualGPU pieces**: cobra91's #4522/#4534/#4549, plus the pipelined verify windows from #859 (+16% average, +30% copy-heavy) if our base lacks them. Issue #1352 is the warning: partial merges leave upstream ~2x slower than the fork.
3. **StrataFast's `--prompt-split`** (1.24-1.55x on long prompts).
4. **The gfx12 (RDNA4) re-measurement** of the gfx11 tune ports (item 62) and DeepSeek V4.1 on gfx1100.
5. **Watch**: Infernix's long-context numbers (+23% at 262K on a 5090) as a reference for any future claim; XeStrata and #3607 if Intel GPUs ever matter; strata-mlx if a Mac shows up.

### What we do NOT take (already rejected, recorded so we do not re-judge)

The full rejected register is in MOD_PLAN.md, lines 1073-1286. Highlights: the upstream expert-cache PR line (#29887 and its relatives), KVA approximate prefill, PLE-in-RAM, DFlash2 for our hardware, int8-WMMA decode attention as a default, ContextPilot (1.6% savings, breaks the GDN state), draft trees, the `.basalt` pack format, NVFP4 (Blackwell only), dense-model offload, and all multi-user/batching items until the server gets a second slot. Re-checking them duplicates work; the register is the canonical record.

## Appendix: sources and verification notes

- **The local audit**: MOD_PLAN.md (phases 2-14, the "Not adopted" register at lines 1073-1286, the Maya robustness section at 1287-1311, the 2-GPU table at 1409-1424), PANCAKE_STATUS.md, the five research notes in `pancake-docs/research/avalon/upgrades-from-upstream/notes/` and the report `pancake-docs/research/avalon/upgrades-from-upstream/report.md`. Branch merge state checked with git on 09.11.
- **The live GitHub pass**: 09.11, through the repo API (forks sorted by stars and by recency), PR and issue searches, probes of author profiles (cobra91, Vop3p, xyzzing, lwYeo, kelvincht, rwkeyes, bahooo22, R-Okauchi), code searches for Strata's distinctive symbols and web searches (the stringers.io summary, prismix.dev, r/LocalLLaMA, HF Forums, dev.to).
- **Verification caveats**: (a) star counts and PR states are from 09.11 and will drift; (b) the numbers measured by authors (StrataGP, StrataQ38FN, Juud_engine, DualGPU, Infernix, Gaudi2, XeStrata) are claims from hardware we do not own, taken from READMEs, issues and posts, not reproduced by us; (c) the fork list cannot be exhaustive at 4,627 forks; symbol searches turned up 10+ derivatives beyond the fork tree, and community coverage names repos we did not check individually (section 3.11); (d) the fork inspection session ended midway without a final summary; its findings are covered by the live pass summarized here, and the confirmed intermediate results (4,516 forks at 13:27, the downstream target `rocktboy1982/pancake-strata`, 10+ finds through symbol searches) are consistent with the numbers above.
- **Companion document**: the English version of this report is at `pancake-docs/STRATA_DERIVATIVES_DEEP_RESEARCH.md`. The port ledger and status live in MOD_PLAN.md and PANCAKE_STATUS.md.

[mp]: file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md
[faststrata]: https://github.com/Jarson1408/FastStrata
[polystrata]: https://github.com/simplexai-labs/PolyStrata
[rdna3]: https://github.com/nicejunjie/Strata-rdna3
[basalt]: https://github.com/jesdga95/basalt
[maya]: https://github.com/mw00/project-maya
[speedy]: https://github.com/DevDock-AI/Speedy
[colibri]: https://github.com/JustVugg/colibri
[stratafast]: https://github.com/eddoursul/StrataFast
[aviary]: https://github.com/SensAI-PT/aviary-hy3
[llamampere]: https://github.com/JakeATX/llamAmpere
[avalon]: https://github.com/freriadoc/Avalon
[tandem]: https://github.com/ewhacc/tandem.cpp
[tandemamd]: https://github.com/itlab-cl/tandem.cpp-amd
[q38mk]: https://github.com/cutlerbenjamin1-cmd/qwen38-megakernel
[stratagp]: https://github.com/gputier/StrataGP
[strataq38fn]: https://github.com/havok2-htwo/StrataQ38FN
[juud]: https://github.com/bnkaiteam/Juud_engine
[dualgpu]: https://github.com/Hardin22/Strata-DualGPU
[xestrata]: https://github.com/MistVVK/XeStrata
[nenenyann]: https://github.com/NeneNyann/Strata-intel-sycl-windows
[lighttransport]: https://github.com/lighttransport/Strata
[gaudi2]: https://github.com/JChen-318/gaudi2-single-card-strata
[cow]: https://github.com/CC-David-CC/Private_Context_COW
[alesha]: https://github.com/alesha-pro/Strata
[vop3p]: https://github.com/Vop3p/dual-6900xt-flash-next-tuning
[kernelbench]: https://github.com/xyzzing/kernelbench-hip
[cupola]: https://github.com/xyzzing/cupola
[lamapeek]: https://github.com/lwYeo/llamaPeek
[hfmax]: https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-Q8-Mix-Strata
[i1352]: https://github.com/Niko1221/Strata/issues/1352
[i4506]: https://github.com/Niko1221/Strata/issues/4506
[i4322]: https://github.com/Niko1221/Strata/issues/4322
[i3865]: https://github.com/Niko1221/Strata/issues/3865
[i4063]: https://github.com/Niko1221/Strata/issues/4063
[i2841]: https://github.com/Niko1221/Strata/issues/2841
[i5297]: https://github.com/Niko1221/Strata/issues/5297
[i5176]: https://github.com/Niko1221/Strata/issues/5176
[i5584]: https://github.com/Niko1221/Strata/issues/5584
[i3784]: https://github.com/Niko1221/Strata/issues/3784
[i5069]: https://github.com/Niko1221/Strata/issues/5069
[i2464]: https://github.com/Niko1221/Strata/issues/2464
[i5609]: https://github.com/Niko1221/Strata/issues/5609
[i5551]: https://github.com/Niko1221/Strata/issues/5551
[i1402]: https://github.com/Niko1221/Strata/issues/1402
[pr5655]: https://github.com/Niko1221/Strata/pull/5655
[pr5535]: https://github.com/Niko1221/Strata/pull/5535
[pr4005]: https://github.com/Niko1221/Strata/pull/4005
[pr5475]: https://github.com/Niko1221/Strata/pull/5475
[pr5595]: https://github.com/Niko1221/Strata/pull/5595
[pr5467]: https://github.com/Niko1221/Strata/pull/5467
[pr5468]: https://github.com/Niko1221/Strata/pull/5468
[pr5466]: https://github.com/Niko1221/Strata/pull/5466
[pr3509]: https://github.com/Niko1221/Strata/pull/3509
[pr5643]: https://github.com/Niko1221/Strata/pull/5643
[pr5516]: https://github.com/Niko1221/Strata/pull/5516
[pr4625]: https://github.com/Niko1221/Strata/pull/4625
[pr5471]: https://github.com/Niko1221/Strata/pull/5471
[pr5470]: https://github.com/Niko1221/Strata/pull/5470
[pr5472]: https://github.com/Niko1221/Strata/pull/5472
[pr5453]: https://github.com/Niko1221/Strata/pull/5453
[pr5566]: https://github.com/Niko1221/Strata/pull/5566
[pr5596]: https://github.com/Niko1221/Strata/pull/5596
[pr5623]: https://github.com/Niko1221/Strata/pull/5623
[pr5512]: https://github.com/Niko1221/Strata/pull/5512
[pr3971]: https://github.com/Niko1221/Strata/pull/3971
[pr5628]: https://github.com/Niko1221/Strata/pull/5628
[pr4777]: https://github.com/Niko1221/Strata/pull/4777
[pr5561]: https://github.com/Niko1221/Strata/pull/5561
[pr5593]: https://github.com/Niko1221/Strata/pull/5593
[pr4972]: https://github.com/Niko1221/Strata/pull/4972
[pr5637]: https://github.com/Niko1221/Strata/pull/5637
[pr5437]: https://github.com/Niko1221/Strata/pull/5437
[pr5588]: https://github.com/Niko1221/Strata/pull/5588
[pr5517]: https://github.com/Niko1221/Strata/pull/5517
[pr5545]: https://github.com/Niko1221/Strata/pull/5545
[pr5638]: https://github.com/Niko1221/Strata/pull/5638
[pr5592]: https://github.com/Niko1221/Strata/pull/5592
[pr3607]: https://github.com/Niko1221/Strata/pull/3607
[pr5647]: https://github.com/Niko1221/Strata/pull/5647
[pr5644]: https://github.com/Niko1221/Strata/pull/5644
[pr4522]: https://github.com/Niko1221/Strata/pull/4522
[pr4534]: https://github.com/Niko1221/Strata/pull/4534
[pr4549]: https://github.com/Niko1221/Strata/pull/4549
[pr4876]: https://github.com/Niko1221/Strata/pull/4876
[pr3277]: https://github.com/Niko1221/Strata/pull/3277
[pr4854]: https://github.com/Niko1221/Strata/pull/4854
[pr4886]: https://github.com/Niko1221/Strata/pull/4886
[pr5518]: https://github.com/Niko1221/Strata/pull/5518
[pr2392]: https://github.com/Niko1221/Strata/pull/2392
[pr5540]: https://github.com/Niko1221/Strata/pull/5540
[pr5562]: https://github.com/Niko1221/Strata/pull/5562
[pr5473]: https://github.com/Niko1221/Strata/pull/5473
[pr3925]: https://github.com/Niko1221/Strata/pull/3925
[pr3384]: https://github.com/Niko1221/Strata/pull/3384
[pr31751]: https://github.com/ggml-org/llama.cpp/pull/31751
