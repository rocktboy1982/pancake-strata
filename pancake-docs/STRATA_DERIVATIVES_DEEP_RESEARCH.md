# Strata Derivatives: Deep Research and Gap Analysis for pancake-strata

Date: 2026-11-09. Scope: every project derived from `Niko1221/Strata` that we could verify on GitHub or in community coverage, and a ranked list of what `pancake-strata` (our fork, `rocktboy1982/pancake-strata`) is still missing from them. Method: a local audit of our own port ledger (MOD_PLAN.md phases 2-14, the "Not adopted" ledger, PANCAKE_STATUS.md), plus a live GitHub pass on 2026-11-09 (repo API, PR/issue search, author-profile sweeps, fork list, code search on Strata's distinctive symbols like `STRATA_`, `native_mmvq`, `qsa_scores`, `combine_cpu`). Every number below carries its source. Claims from other people's hardware are marked as such; nothing in this report was measured on our box unless it says so.

## 1. Executive Summary

The Strata ecosystem as of November 2026:

- **Upstream `Niko1221/Strata`**: 42,317 stars, 4,627 forks, 1,216 open issues, `main` pushed the same day this report was written. Release cadence is extreme: v0.2.9 (10-31), v0.2.10 (11-02), v0.2.11 (11-05), v0.2.12 (11-07), v0.2.13 (11-09, engine 0.1.59). Our fork's base is `146df3c4` (v0.2.12, engine 0.1.58), so upstream is one release ahead of us and moving fast.
- **The fork tree is large and mostly quiet, with a loud minority.** Of the 4,627 forks, the overwhelming majority are passive clones. We verified roughly 30 derivatives with real work in them: 14 studied in depth and already mined by `pancake-strata` (Tier 1), and about 20 discovered or re-confirmed in the live pass that we have not ported from (Tier 2). Beyond GitHub there is a HuggingFace quant-pack ecosystem, a competing closed engine (Infernix), and blog/Reddit coverage including a hype backlash and a crypto-scam denial by the author.
- **`pancake-strata` compares well on its own turf.** On a single RX 7900 XT we measure ~112 tok/s decode after the Strata-rdna3 kernel ports, which beats Strata-rdna3 itself (109-119), Avalon (95-100) and stock Strata (75-77) in the same suites. Our server-side agent robustness (loop guard, tolerant tool calls, thinking budgets, grammar engine) is ahead of upstream and of every fork we inspected. What we lack is breadth: dual-GPU pipelines, multi-user serving scale, non-CUDA/HIP backends, and a series of upstream kernel and startup patches that landed or matured after our base.

The three biggest gaps, in one paragraph each:

1. **A correctness bug that touches code we ship.** Upstream PR #5655 found that `route_prior` leaks a sentinel expert ID and can write out of bounds on degenerate logits. We ship `STRATA_ROUTE_PRIOR` (the `turbo`/`turbo-hip` presets) and `STRATA_ROUTE_PRIOR_DEPTH`, so this is our bug too until proven otherwise. It is the only item in this report with a "stop and check first" flag on it.
2. **Startup and memory elasticity.** Our restarts pay 28-65 s just to read the 33 GiB expert arena from ntfs3. Three independent solutions exist in the ecosystem and we have ported none: upstream PR #3509 (`STRATA_ARENA_REUSE=1`, attach a shared arena instead of re-copying), Project Maya's direct parallel SSD expert reads (O_DIRECT, 2-16 reader threads, +26% decode on a RAM-short PC), and basalt's sleep-in-place (commit `64167ba73`). Maya's weight-share miss skip (`MAYA_GLM_MISS_SKIP_W`, +7-13% decode at KL 0.037-0.052) is in the same family.
3. **Dual-GPU and multi-user serving.** Hardin22/Strata-DualGPU demonstrated 29 to 161 tok/s on two GPUs with 32 GB RAM, and upstream issue #1352 shows the upstream merge is still incomplete (upstream ~2x slower than the fork on a 4070 Ti Super + 5060 Ti). cobra91's three open PRs (#4522, #4534, #4549) are the remainder. havok2-htwo/StrataQ38FN shows 43 patches taking a 4x3090 box to ~578 tok/s with 16 users (6.6x) and cutting agent-switch re-reads from 25 s to 1.5 s. All of this is parked behind our Phase 12 (second GPU), but the PRs will not wait for us.

## 2. Tier 1: Major Engine Forks Already Studied and Ported into pancake-strata

These 14 projects were audited against our port ledger. For each: URL, license, what we took, and what remains unported. Branch state was verified with git: `port/faststrata-poly`, `exp/expert-prefetch`, `feat/grammar` and `feat/pattern-cache` are merged into our working line `port/rdna3-decode` (which is not yet merged into `main`). Unmerged branches: `feat/disco` (+1), `feat/bfla-hip` (+7), `feat/bitdecoding-hip` (+2), `feat/contextpilot` (+2).

### 2.1 Jarson1408/FastStrata and simplexai-labs/PolyStrata (MIT)

- URLs: https://github.com/Jarson1408/FastStrata (commit `df91582`), https://github.com/simplexai-labs/PolyStrata (commit `a82999fd`). License: MIT. PolyStrata is FastStrata's parent; both ship the `src/poly` second engine.
- **Ported and merged**: the whole `src/poly` engine, ported as `strata-poly` on branch `port/faststrata-poly` (now merged), with `--attn-split`, `--graph-experts`, `--elastic-prefill`, `--lend-staging`, `--stage-next`, `--suffix-drafts`, `--cpu-multi`, `--kv-q8`, and poly's `--draft-vocab`. Measured on the RTX 3060: short chat 100.7 tok/s, code edit 151.1 tok/s, 41.8K prompt in 35.8 s (71.7 tok/s), against stock strata at 64.6 / 40.1 / 42.1 s.
- **Rejected**: adaptive MTP depth (poly's cost-based `Guesses` rule already covers it); fused FA prompt kernel for the main engine (attention is 6.7% of a 42K prompt there; parked pending a 7900 XT prefill profile).
- **Still unported**: `--conversation-disk` automatic disk sessions exist only in strata-poly. The main-engine equivalent (MOD_PLAN item 39, restore in ~0.5-1 s) is not built. FastStrata's tensor-core decode attention (`--attn-split`) is CUDA-only; a HIP/gfx1100 WMMA path does not exist.

### 2.2 nicejunjie/Strata-rdna3 (MIT)

- URL: https://github.com/nicejunjie/Strata-rdna3 (local remote `rdna3`).
- **Ported and merged** (branch `port/rdna3-decode`): the RDNA hyper-connection read family (+Q8, A8, INJ_UP, FUSED), wave32 multi-row GEMVs (`RDNA_Q6K`, `MMVQ_WAVE`), `COMBINE_CPU`, `WAIT_FUSE`, `ROUTE_PUBLISH`. Decode went 87-101 to 104-108 (kernels) to ~112 tok/s (fusions).
- **Rejected**: long-context top-k (= upstream `SELECT_LONG`), `PF_GROUP_RES` (= upstream `group_resident`), `PA_KLDS` (upstream rewrote it as `STRATA_PA_RPF`), `RDNA_PAIR` (neutral), rdna3's own expert profile and adapt settings (ours measured better: 121.2 vs 116.4), the zh60k draft vocab (92.5, worse), and the GDN decode megakernel (0.0 ms on one GPU; the window waits on CPU rows).
- **Still unported, all parked on Phase 12 (second GPU)**: tensor-parallel decode (`src/core/tp.cpp`, `--tensor-parallel 2`; rdna3 measured 129-170 tok/s on 2x 7900 XTX), peer expert transfer / RCCL (`peer_experts.cpp`, `peer_transport.inc`), kfdq idle-queue parking (idle power 98+91 W down to 5+13 W), the GDN megakernel revisited under TP, and a gfx12 (RDNA4) re-measure of the gfx11-tuned ports (item 62).

### 2.3 jesdga95/basalt (MIT)

- URL: https://github.com/jesdga95/basalt (commit `a9468e5c`; a history-rewritten fork of Strata `822a5c783`; local remote `basalt`).
- **Ported and merged**: (a) the grammar engine, done 2026-11-08 on `feat/grammar`: GBNF / JSON-schema / tool-call builders (`d64768699`), per-request guard (`02e1006fe`), `Verifier::resample_row` (`c7813a6c3`), engine `--grammar on` plus a request-level `grammar=` key with verify-window masked redraw so MTP keeps drafting, and `serve/grammar.py`. Cost: 0.1-0.2% per window; shipped in the IQ2_XS presets. (b) Head kernels (owner request, done): `q5k_head.cu` ported for Q5_K heads (Q2_0/IQ3_XXS) and the same designs written for IQ4_XS: exact head GEMV plus 2-bit two-stage draft pick.
- **Rejected**: the 2-bit draft-head screen and K/V-append-beside-draft-query (the draft step is 2.5-3.2 ms of a 46-130 ms window, 1-2% max); the `.basalt` pack format (needs Blackwell sm_120).
- **Still unported**: `--idle-sleep` / sleep-in-place (commit `64167ba73`, ~815 lines: drop and redo the RAM copy's registration, refill on wake). Parked unless we want the GPU for other work while idle. It would also close the `--vram-elastic`-refuses-with-pinned-complement gap.

### 2.4 mw00/project-maya (MIT)

- URL: https://github.com/mw00/project-maya. License: MIT. A Strata fork pinned at engine 0.1.27, specialized for Qwen.
- **Ported and merged** (server side, MOD_PLAN "Agent robustness from Project Maya"): (a) the loop guard, `serve/loop_guard.py`, on by default: token-loop detection with drop and repetition penalty 1.15, thinking close at the 3rd loop, answer stop, tool-repeat notes and blocking after 3 notes, prompt collapse, keep-going nudges, `/metrics` plus Prometheus counters, tests in `serve/test_loop_guard.py`. (b) The soft thinking budget, `reasoning_budget_grace` (close at sentence or line end; Maya #613). (c) Tolerant tool calls: an unclosed `<parameter=` tag is malformed but recoverable, and a malformed call never raises (#613/#706; tests in `serve/test_tolerant_tool_calls.py`).
- **Rejected**: running Maya itself (its Qwen engine is a stale 0.1.27 snapshot, 1.4-1.7x slower than current Strata, which already loses to pancake); Maya's resume-thinking after an end-of-turn token (both Qwen turn-end tokens are the engine's EOS; the empty-answer case is handled by #1053); DFlash2/DSpark/EXL3 (license or architecture); tensor-parallel multi-GPU; vision lending; the dashboard and savings counter.
- **Still unported**: (a) direct parallel expert reads for experts RAM cannot hold: O_DIRECT with drive-probed 2-16 reader threads (`MAYA_EXPERT_DIRECT` / `MAYA_EXPERT_STAGERS`); Maya measured +26% decode on a RAM-short PC. Our related item 38 (Linux direct expert-arena loader, estimated 28-65 s down to 10-14 s startup) is planned but not built. (b) Miss skip by weight share (`MAYA_GLM_MISS_SKIP_W`, +7-13% decode at KL 0.037-0.052 on GLM); we only have rank-based tail skip. (c) Smaller items: a `--calibrate`-style startup sweep, drafter auto-keep, the Turbo preset as a launcher profile, a prompt memo (~6% of follow-up TTFT), and one-GPU multi-request batching.

### 2.5 DevDock-AI/Speedy (MIT)

- URL: https://github.com/DevDock-AI/Speedy. License: MIT.
- **Ported and merged**: Speedy's sm_86 presets as our starting config (item 8.1); the upstream opt-ins Speedy measured: `STRATA_PF_GROUP_SCRATCH`/`_HC_REGION` (#4039), `STRATA_PREFILL_ROUTED_STREAM` (#4283), `STRATA_PROMPT_ATTN_IMMA`, `STRATA_KV_PREFETCH` (item 8.2, done or measured).
- **In progress**: `STRATA_VERIFY_DEFER_CPU` (#2689/#2690, item 8.3) is coded, the highest-ceiling lossy item we know (up to 1.45x per KTransformers), but it produces non-finite logits on HIP with our ported kernels. A 6-run bisect is queued (PANCAKE_STATUS: Dev 100%, Test 10% after the ROUTE_PUBLISH bisect fix). The `STRATA_PSI_SHED` baseline is queued in the stage-1 group.
- **Rejected**: Speedy's INT8-IMMA attention variant (attention is 6.7% of the main-engine window).
- **Still unported**: the deferral rank guard `STRATA_VERIFY_DEFER_CPU_TOP` (item 28) is coded (Dev 100%) but only 5% tested; the HIP non-finite fix gates both.

### 2.6 JustVugg/colibri (Apache-2.0)

- URL: https://github.com/JustVugg/colibri. License: Apache-2.0.
- **Ported (opt-in)**: cache-aware routing, `CACHE_ROUTE` → our `STRATA_ROUTE_PRIOR` (item 8.9). Measured: lambda 0.5 gives +25% decode at 2.89% mismatch; shipped as the opt-in `turbo`/`turbo-hip` preset. colibri's PILOT idea became our router lookahead, now on by default on the GPU (`STRATA_LOOKAHEAD_GPU`, +10% decode on mmap'd experts, same picks).
- **Rejected**: lambda 2.0 (9.90% mismatch, 8 large-margin events); the CPU-thread lookahead predictor (-13% decode, replaced by the GPU version).
- **Still unported**: nothing code-wise. The `turbo-hip` quality-gate decision (P2a) is open: the preset exists, token forcing was INCONCLUSIVE (+0.4 pts), Test 30%.

### 2.7 eddoursul/StrataFast (MIT)

- URL: https://github.com/eddoursul/StrataFast. License: MIT.
- **Still unported**: `--prompt-split` two-stage prompts (item 8.7; 1.24-1.55x on long prompts there). Blocked: it needs a second GPU, so it is parked in Phase 12. Nothing else from this fork appears in our notes.

### 2.8 SensAI-PT/aviary-hy3 (Apache-2.0)

- URL: https://github.com/SensAI-PT/aviary-hy3. License: Apache-2.0.
- **Ported**: nothing attributed to this source appears in the ledger as a separate item; its relevant ideas arrived through the upstream opt-ins already covered under Speedy (the `STRATA_PF_GROUP_*` and routed-stream families).
- **Still unported**: no distinctive feature remains unexamined. If a new release appears, re-check its kernel directory against our "Not adopted" ledger before assuming overlap.

### 2.9 JakeATX/llamAmpere (MIT)

- URL: https://github.com/JakeATX/llamAmpere. License: MIT.
- **Rejected / closed**: adaptive MTP depth (poly's cost rule covers it); the SJ-KVaRN-style 8-bit KV body and mb-MTP (VRAM does not buy decode here; drafting is 2-5% of a window).
- **Still unported**: nothing else is attributed to this source.

### 2.10 freriadoc/Avalon

- URL: https://github.com/freriadoc/Avalon. A llama.cpp-based engine (not a Strata fork) that tracks upstream llama.cpp and competes with Strata on the same models; we studied it as a derivative of the same ecosystem and benchmark target.
- **Ported / adopted**: the gfx1100 hipBLASLt tuning table `100401` in our 7900 XT presets (done; +38% prompt provisional, Test 80%); "finish together" hosted-MoE (item 10.4) turned out to already exist in Strata as PcieAuto; the `--lookup-max-t 12` flag exists (A/B queued, +1% noise so far). We fast-forwarded the Avalon source from `c24c03d` to `88313b6` (48 commits: registered-RAM arena, link-rate hosted-MoE limit, IQ3_KS MMA MMQ, sparse-FA tile fix) and built it, with a binary rollback at `Avalon/bin-c24c03d-rollback`; the rebuild/bench is pending because the model is missing (~20 GB short on D).
- **Rejected** (highlights; the full list is in "Avalon upgrades from upstream projects.md"): the upstream expert-cache line #29887/#30637/#30966/#31888/#31286/#30594/#27861 (our tier plus `--pcie-frac` already do this; #30594 hurts MTP verify); #31497 (contradicts the shared MTP cache); KVA approximate prefill (decode -11%); PLE-in-RAM (57 GB RAM cannot hold model + 28.8 GB table + headroom); router residency bias as default (loops in agent sessions); Strata's gfx1100 kernel switches (int8 WMMA attention, TSUM, expert layouts: neutral or negative on 7900-class); ClassLRU long-memory scoring (within 0.5 points on replay); HIP graphs, Vulkan, turbo4/TurboQuant KV, and the rest, all tested with no gain.
- **Still unported**: the `--lookup-max-t 16` A/B and the hipBLASLt `100401` confirmation A/B (both queued, both need a quiet machine). And one item the Avalon report ranked "Add now" that has no pancake counterpart recorded anywhere: **llama.cpp #31751**, the one-pragma RDNA3 J=32 MMQ register-spill fix, +36% at 17-32-token batches on gfx1100, which is exactly our verify-window size. See section 4.3.

### 2.11 ewhacc/tandem.cpp and itlab-cl/tandem.cpp-amd

- URLs: https://github.com/ewhacc/tandem.cpp and https://github.com/itlab-cl/tandem.cpp-amd (the AMD variant).
- **Ported**: `STRATA_POOL_PREREAD` (idle pool workers pre-read the next layer's CPU experts). The A/B is still pending. `MTP_DEFER` was already present.
- **Rejected**: the MTP-head Q4_K requant (our head is ~2-bit, 787 MB; requantizing buys nothing).
- **Still unported**: nothing else identified.

### 2.12 cutlerbenjamin1-cmd/qwen38-megakernel (AGPL)

- URL: https://github.com/cutlerbenjamin1-cmd/qwen38-megakernel. License: AGPL. Ideas only; we do not copy code from it because of the license.
- **Taken as ideas**: draft-vocab experiments (full/en/zh60k) all measured flat or worse; the GDN replay idea was already present.
- **Still unported**: nothing. The AGPL license makes code porting a non-goal.

### 2.13 Smaller sources audited alongside Tier 1

- **gpu-unlock**: measured, no adoption. FP32-sum MMA reached 49.7 TFLOPS but decode was unchanged; a 5.3K prompt went 7.6 → 6.0 s. Closed for decode purposes.
- **SGLang**: speculative sampling ported into the engine (`STRATA_SPEC_PROB`, measured; the default path is as fast or faster); the GPU token mask for grammar is done (phase 6); draft trees rejected (GDN state copies plus expert waking; watch upstream #30931).
- **ContextPilot** (`feat/contextpilot`, unmerged, +2): coded (lossy block dedup, opt-in, exact cache-aware admission), then rejected: 1.6% dedup savings on 40 real opencode sessions, and non-prefix reuse breaks GDN state. Stays off.
- **BFLA (CityU paper)** (`feat/bfla-hip`, unmerged, +7, HIP only): block-sparse prompt attention, measured end to end on Qwen3.6 UD-Q4_K_XL: a 128K prompt went 386.9 → 1,219.8 tok/s sparse (3.15x) → 771.2 tok/s dense; recall 48/48, KL 0.0014/0.0045. Merge decision pending (Test 10%).
- **BitDecoding** (`feat/bitdecoding-hip`, unmerged, +2): `STRATA_ATTN_WMMA`, HIP gfx11 decode attention on matrix cores, measured with the model plus token forcing; parked as a non-default (attention is ~3% of decode; ~2-3% confirmed).
- **DISCO** (`feat/disco`, unmerged, +1): learned stop rule for the MTP draft chain (opt-in). The SpecFit missed-expert feature (item 36) is assigned to this branch, not started.
- **University paper series** (Stanford/MIT/Harvard/US/EU/Asia/Canada/Pacific reports): the per-level thinking cap is done (`5dd620fb9`); sparsity masks (Berkeley gate/TEAL/CATS/CHESS/SpAx) rejected offline (Flash-Next's FF-640 experts are dense: gate cos 0.844 at 70%); XAttention superseded by BFLA; the predictor-recall item closed (19-21% coverage; promotions do not pay). Pending: learned thinking stop (item 21), effort defaults (12), loop-guard counters (33), the rumination signal (34), the restore-equivalence test (35), GDN chunk 64 (18), the IQ1_M AVX2 kernel (24, 0 on Q2_0), the Coder-variant A/B (25), and the semantic oldString constraint (26).

## 3. Tier 2: Newly Discovered and Active Forks and Derivatives Not Yet in pancake-strata

Found in the live pass of 2026-11-09 through issue/PR cross-references, author-profile sweeps, code search on Strata's distinctive symbols, and web search. Star counts are from that day.

### 3.1 gputier/StrataGP

- URL: https://github.com/gputier/StrataGP (9 stars).
- **What it has**: 16 packaged performance/precision patches under `docs/perf/`, measured against upstream `f1b1d96` on an RTX 5090: +5-15% decode and +42-49% prompt.
- **Why it matters to us**: the largest single bundle of ready-made, documented kernel patches we have not examined. The prompt-side gains are the interesting part, since our prompt path is the one that loses after long contexts.
- **Action**: diff the 16 patches against our base; expect overlap with the bahooo22 PR cluster (#5595/#5467/#5468) and with our rejected list (int8 WMMA attention). Port what survives.

### 3.2 havok2-htwo/StrataQ38FN

- URL: https://github.com/havok2-htwo/StrataQ38FN (1 star).
- **What it has**: a 43-patch, upstream-ready series on v0.1.43, plus a vLLM-compatible facade and a dashboard. Measured by the author: 4x3090 with 16 users at ~578 tok/s (6.6x over base), RTX 5090 +16% single-user decode, and agent-switch re-reads cut from 25 s to 1.5 s.
- **Why it matters to us**: the agent-switch re-read number is our exact pain point (our opencode data shows ~18% of turn time in re-reads). The multi-user work is irrelevant to our single-slot server today but is the best-documented serving-scale reference in the ecosystem.
- **Action**: read the re-read patch first; it may compose with our message-boundary checkpoints (item 11) and #5643 below.

### 3.3 bnkaiteam/Juud_engine

- URL: https://github.com/bnkaiteam/Juud_engine (2 stars).
- **What it has**: two CPU-decode tweaks: a shortened worker spin after the expert batch, and skipping activation quantization for GPU-resident tokens. Author-measured +9.6-15.6% decode on an RTX 4090 (HF Forums post).
- **Why it matters to us**: our pool already runs at ~80% of measured DDR4 bandwidth, so spin shortening is exactly the kind of overhead cut that can still show up on edit windows, and the GPU-resident activation skip is free whenever our tier keeps experts resident.
- **Action**: port both as opt-in flags; A/B against our pool sweep defaults.

### 3.4 Hardin22/Strata-DualGPU and cobra91/Strata-DualGPU

- URLs: https://github.com/Hardin22/Strata-DualGPU (84 stars) and the cobra91 continuation fork.
- **What they have**: the two-GPU pipeline for machines with 32 GB RAM: Swift IQ2_XS went from 29 to 161 tok/s on code. Partially merged upstream: issue #1352 shows upstream is still ~2x slower than the fork on a 4070 Ti Super + 5060 Ti, so the merge is incomplete. The pieces that reached upstream: #848 (resident RAM mode on a layer split plus `--vram-reserve-later-mib`, shipped 0.1.40), #859 (pipelined verify windows, +16% mean, +30% copy-heavy), #876/#905 (async tier). Combined effect on a 4060 Ti + 5080 with 32 GB RAM: 29 → ~84+ tok/s.
- **The unmerged remainder** (cobra91's open PRs against upstream): #4522 (trim dense weights on `--layer-split auto`), #4534 (price a pipelined split by its slower stage), #4549 (keep draft-head room out of the last card's cache).
- **Why it matters to us**: this is the Phase 12 playbook, already written and measured. Our 20 GB VRAM holds only 44% of expert pairs resident; a second GPU is the only lever that changes that.
- **Action**: when the second GPU lands, start from #4522/#4534/#4549 and the #859 pipelined verify windows, not from a fresh design.

### 3.5 MistVVK/XeStrata and NeneNyann/Strata-intel-sycl-windows

- URLs: https://github.com/MistVVK/XeStrata (334 stars), https://github.com/NeneNyann/Strata-intel-sycl-windows (10 stars).
- **What they have**: XeStrata is the multi-vendor, no-CUDA build: SYCL/Level Zero with XMX paths; its `xe0.2.2-1` beats upstream's sycl 0.1.45.3 on an Arc Pro B50 (issue #4506). NeneNyann packages Windows SYCL builds with bundled oneAPI runtimes and GPU vision.
- **Why it matters to us**: low priority. We are HIP/CUDA. Relevant only if we ever target Intel GPUs, or as a reference for SYCL-side kernel ideas that also exist in HIP form.
- **Action**: watch only.

### 3.6 lighttransport/Strata

- URL: https://github.com/lighttransport/Strata (20 stars), branch `glm53f`.
- **What it has**: a native GLM-5.3-Flash GGUF backend inside Strata, ~29 tok/s on 2x V100 and ~7.5 tok/s on a Threadripper (CPU). It motivated feature request #2067 and now upstream PR #5545 (`strata-glm` engine path, quantization pipeline, multimodal bring-up).
- **Why it matters to us**: we run GLM-5.3-Flash ourselves (this report's own sessions run on it). A native engine path would replace whatever fallback we use today.
- **Action**: track #5545; when it merges, rebase our fork onto it and re-measure our GLM serving path.

### 3.7 JChen-318/gaudi2-single-card-strata

- URL: https://github.com/JChen-318/gaudi2-single-card-strata (10 stars).
- **What it has**: a native Intel Gaudi2 (`hpu/`) backend with custom TPC kernels; Qwen3.8-Flash-Next IQ4_XS fully resident; author-measured 126.8 tok/s chat decode at 262K context (issue #4322).
- **Why it matters to us**: none directly (we have no Gaudi hardware). It matters to the ecosystem as proof that Strata's engine ports beyond GPU vendors.
- **Action**: watch only.

### 3.8 CC-David-CC/Private_Context_COW

- URL: https://github.com/CC-David-CC/Private_Context_COW (1 star).
- **What it has**: copy-on-write private context, discussed in upstream issue #3865, self-described as "stable enough for testing".
- **Why it matters to us**: our parked conversations live only in RAM and a restart loses them; COW context is one of the memory-family ideas (alongside #5623's shared immutable host KV prefixes) that could cut the 5.4-6.1 GiB we spend on 4 parked conversations.
- **Action**: read the issue thread; compare against our item 3 (multi-state store) before building anything.

### 3.9 alesha-pro/Strata

- URL: https://github.com/alesha-pro/Strata (3 stars).
- **What it has**: a port of Mirai's 2-bit trellis "S" quant patch to Strata 0.2.0 (upstream PR #3384), plus abliterated model packs on HuggingFace.
- **Why it matters to us**: our head is already ~2-bit (787 MB), but the body-side trellis quant is unexamined. If it shrinks CPU-resident experts at the same KL, it multiplies everything the DDR4-bound pool does.
- **Action**: evaluate the quant offline (perplexity/KL on our model) before any kernel work.

### 3.10 Tooling, tuning logs, and the competitor

- **Vop3p/dual-6900xt-flash-next-tuning**: https://github.com/Vop3p/dual-6900xt-flash-next-tuning (3 stars). A standalone tuning log for llama.cpp → Strata on 2x RX 6900 XT. Same author as PRs #4005 and #5475 below, so the repo is a useful field notebook for RDNA2/3 HIP behavior. Watch.
- **xyzzing/kernelbench-hip** and **xyzzing/cupola**: KernelBench ported to HIP/ROCm, and an auditable inference supervisor for Radeon. The same author opened #5535 (int8-WMMA prompt attention on gfx1100). The supervisor idea overlaps our loop-guard metrics; KernelBench-HIP is a handy harness if we start writing our own kernels. Watch.
- **lwYeo/llamaPeek**: a standalone Windows tray monitor for llama.cpp-compatible servers. We run Linux; cosmetic. Watch.
- **Xolotl** (formerly Strata-DSV4.1): named in the stringers.io ecosystem roundup; not individually verified. Presumably a DeepSeek V4.1 specialization. Watch.
- **Infernix** (2026.10.09.2): **not a fork, a competing engine**. NVFP4 recipe-C artifact plus an mmap'd n-gram volume: author-measured +23% decode at 262K and +9% at 512K versus Strata 0.1.41 on one RTX 5090 (149.0 vs 120.6 tok/s), vision working, ~28 s cold start (prismix.dev, r/LocalLLaMA). This is the number to beat if we claim long-context decode leadership. Watch its release notes.

### 3.11 Named in community coverage, not yet verified individually

PolyStrata (863 stars per the roundup; adds a second engine for GLM/DeepSeek/Gemma), strata-nvfp4, strata-mlx (Mac), Strata-V100, eddoursul/Lamina, momoe, talus-public (Strix Halo), FastStrata. Also in the coverage: the SEO squatter `stratallm.org` and the author's crypto-scam denial on X (10-19). None of these has a verified feature we lack; verify before citing.

### 3.12 External (non-GitHub) derivatives

- **HuggingFace quant packs**: Maximilian228/Qwen3.8-Flash-Next-NVFP4-Q8-Mix-Strata (issue #4063, RTX PRO 5000 48GB); DavidAU Turbo-Brilliance GSQ-RCO packs (issue #2841); alesha-pro's abliterated/trellis packs; ISTA-DASLab GSQ-RCO (Q2_0 → IQ3_S plus a Coder 256/512-expert variant, 0.91 GB BF16 mmproj); Unsloth Dynamic v3.1 (re-uploaded 11-04; issue #5297 asks for the ~9 GB-smaller UD-IQ2_XS; UD-IQ1_S does not load because of mixed expert formats in one layer); Gyro-S/Gyro-M from agentionai (issues #5176, #5584; Gyro-M TQ2_0 on an RX 7900 XTX; #3784 Gyro-S Mac OOM).
- **Community/blog**: a dev.to 64 GB quickstart (rosgluk, 11-07); a traictory.com hype-backlash piece; an explainx.ai guide; the HF Forums Juud_engine post.
- **Reddit**: the r/LocalLLaMA origin post by KnownAd4832 (65 tok/s on 12 GB VRAM), the "stop the Strata spam" backlash thread, and dual-5090 WDDM field data in the issue #642 comments.

## 4. Tier 3: Upstream Niko1221/Strata (v0.2.12 to v0.2.13) and High-Value Open PRs

### 4.1 What v0.2.13 changed (and what it means for our base)

v0.2.13 shipped 11-09 (engine 0.1.59), two days after our base v0.2.12 (`146df3c4`). The last-48-hours `main` commits run `f33c8c305` through `8a9feaedd`:

- **Qwen4 shape-engine architecture** (`f-0213-shape`): a new-shape Qwen4 model ships as its own engine component `strata-q4-<shape>` beside the Strata engine, selected by the model. `strata --geometry` prints the compiled kernel geometry (`flash_next` is the default), and install-time geometry checks run against a signed manifest. This is an architecture change: future models may stop being drop-in GGUFs for older engines, which affects how we rebase.
- **DFlash2 per-device defaults** (`e3bd5cfde`): Strix Halo and Blackwell cc 12.0 with 48 GB or more get draft window 6 → 8, and conf 0.55 for IQ3_S on that Blackwell class. Author-measured: Strix UD-Q4_K_XL x1.052, l620 IQ3_S x1.028. We measured DFlash2 ourselves and closed it (115.9 tok/s short but 84.3 after a long prompt; an upstream crash on QFN), so this only matters if we reopen it.
- **Strix Halo G16 fix** (`46ea3947e`): no 2560 MiB Windows VRAM-reserve floor on integrated (UMA) HIP devices; setup writes `--vram-reserve-mib 700` on single-card gfx1151. Stale-config start 77 s, 20.5-24.3 tok/s. Not our hardware, but the UMA logic may touch shared setup code.
- Smaller: fit-label unification ("fits" / "may run slower"), an external-drive data folder fix, NVFP4-QAD sized by its 76.8 GB pack, and a feed-model install gate (`tools/feed_model_install_gate.py`).

### 4.2 The open PR backlog, by cluster (all unmerged as of 2026-11-09)

**Kernels and MoE:**

| PR | Author | Mechanism |
|---|---|---|
| [#5595](https://github.com/Niko1221/Strata/pull/5595) | bahooo22 | q36: grouped down-expert kernel for Q4_K (fmt 12) |
| [#5467](https://github.com/Niko1221/Strata/pull/5467) | bahooo22 | q36: flash attention on mma.m16n8k16 for prompt (fp16 K/V) |
| [#5468](https://github.com/Niko1221/Strata/pull/5468) | bahooo22 | q36: layer-major prefill stages a layer's experts once per N chunks ("super-chunk") |
| [#5466](https://github.com/Niko1221/Strata/pull/5466) | bahooo22 | q36: report need/free MiB on expert-buffer/staging OOM |
| [#5535](https://github.com/Niko1221/Strata/pull/5535) | xyzzing | int8-WMMA prompt attention on RDNA3/gfx1100 (RX 7900 XTX), with recipe, validation and results |
| [#4005](https://github.com/Niko1221/Strata/pull/4005) | Vop3p | prefill: opt-in sparse copy events + dedup slot waits; Windows HIP prompt +36% @4K (closed, unmerged) |
| [#5475](https://github.com/Niko1221/Strata/pull/5475) | Vop3p | `STRATA_HC_PERSIST_Q8`: int8 copies for the persistent hyper-connection read |
| [#4876](https://github.com/Niko1221/Strata/pull/4876) | xrip | `STRATA_GDN_TC=2`: tensor-core GDN recurrence on sm_75, 2.2x faster |
| [#3277](https://github.com/Niko1221/Strata/pull/3277) | c3c | extend native-IQ prefill dispatch to SM75 (Turing) |
| [#4854](https://github.com/Niko1221/Strata/pull/4854) | Keoian | AVX1 Q2_0 native rows (pre-AVX2 CPUs) |
| [#4886](https://github.com/Niko1221/Strata/pull/4886) | Spinoza0 | Metal: IQ1_M expert gate/up in GSQ packs (missing type-29 instantiations, #2537) |
| [#5518](https://github.com/Niko1221/Strata/pull/5518) | rwkeyes | HIP: `alignas(16)` int4 staging arrays do not compile (two sites from #778) |
| [#2392](https://github.com/Niko1221/Strata/pull/2392) | skuridin | hipBLASLt tuning table for gfx1101 (RX 7800 XT, hipBLASLt 1.2.0); open since 10-14 |
| [#5540](https://github.com/Niko1221/Strata/pull/5540) | Captain-Tripps | sycl: count helper/peer caches before the adaptive tier; helper-mode decode 32 → 61 tok/s |
| [#5562](https://github.com/Niko1221/Strata/pull/5562) / [#5473](https://github.com/Niko1221/Strata/pull/5473) | lwYeo / YamineRL | fused-layout shrink asks every layer's path; link stubs for `fused::want_dm_f16` |
| [#3925](https://github.com/Niko1221/Strata/pull/3925) | schmittcn | q36: load GGUFs whose `ssm_alpha`/`ssm_beta` are K-quants or IQ4_XS |

**The correctness bug (see P0 below):**

- [#5655](https://github.com/Niko1221/Strata/pull/5655) (WilliamStone): `route_prior` leaks a sentinel expert ID and can write out of bounds on degenerate logits.

**DeepSeek V4.1-Flash cluster (R-Okauchi, all open):**

- [#5471](https://github.com/Niko1221/Strata/pull/5471): faster Q4_K/Q5_K decode GEMVs, plus light experts at the same bits.
- [#5470](https://github.com/Niko1221/Strata/pull/5470): `DS4_PROF_WIN`, a decode profile of drafter verify windows.
- [#5472](https://github.com/Niko1221/Strata/pull/5472): DGX Spark `STRATA_DS4_DISK_SKIP_RANK`/`PROMPT_SKIP_RANK`.
- [#5453](https://github.com/Niko1221/Strata/pull/5453): DGX Spark `DS4_UMA_TIER=1`, one expert tier the GPU reads in place.
- [#5566](https://github.com/Niko1221/Strata/pull/5566) (yeyeyeason): opt-in cold store for the expert arena (the read-source half of #5391).
- [#5596](https://github.com/Niko1221/Strata/pull/5596) (AlexGabbia): `STRATA_DS4_CACHE_ROUTE`, a selection bonus for experts already in VRAM/RAM.

**Server and serving behavior:**

- [#5516](https://github.com/Niko1221/Strata/pull/5516) (rwkeyes): in-place retune (TUNE) plus conversation-cache routing by traffic shape.
- [#5643](https://github.com/Niko1221/Strata/pull/5643) (kelvincht): restore reasoning a client dropped, so the next turn reuses the cached prompt prefix.
- [#5623](https://github.com/Niko1221/Strata/pull/5623) (shoucandanghehe): share immutable host KV prefixes across streaming sessions.
- [#3509](https://github.com/Niko1221/Strata/pull/3509) (Jerry-Lee661): `STRATA_ARENA_REUSE=1`, attach a shared arena instead of re-copying it.
- [#4625](https://github.com/Niko1221/Strata/pull/4625) (GoodSlopBadSlop): `STRATA_POOL_SPIN_US=0` actually means zero.
- [#5512](https://github.com/Niko1221/Strata/pull/5512) (rwkeyes): skip the thinking block on short-direct-answer prompts (opt-in).
- [#3971](https://github.com/Niko1221/Strata/pull/3971) (macintog): keep thinking tokens generated past a budget cut.
- [#5628](https://github.com/Niko1221/Strata/pull/5628) (gener-vscode): stream tool args at JSON token boundaries.
- [#4777](https://github.com/Niko1221/Strata/pull/4777) (code-martin): dynamic `route_prior` via API/per-request.
- [#5561](https://github.com/Niko1221/Strata/pull/5561) (njy1991): Linux file-tier sequential read-ahead (`STRATA_EXPERTS_SEQ_READAHEAD`).
- [#5593](https://github.com/Niko1221/Strata/pull/5593) (krispgece): `STRATA_PROMPT_DIRECT=1` without a RAM budget.
- [#4972](https://github.com/Niko1221/Strata/pull/4972) (01xjw): file-tier Linux defaults off when GPU caches hold every expert.
- [#5637](https://github.com/Niko1221/Strata/pull/5637) / [#5437](https://github.com/Niko1221/Strata/pull/5437): inhibit sleep during inference (Linux / cross-platform).
- [#5588](https://github.com/Niko1221/Strata/pull/5588) (noahark): `--ple-gguf`, a side-loaded PLE table wins by-name resolution.

**New model families and platforms:**

- [#5517](https://github.com/Niko1221/Strata/pull/5517) (archadrian): qwen3next family, geometry-derived trunk plus GDN grouped-head support (Groot lane byte-identical).
- [#5545](https://github.com/Niko1221/Strata/pull/5545) (zhuxm-ai): GLM-5.3-Flash engine path (`strata-glm`), quantization pipeline, multimodal bring-up.
- [#5638](https://github.com/Niko1221/Strata/pull/5638) (xiezhipeng-git): fine-tune Qwen3.8-Flash-Next for +0.40 GiB VRAM (zero-order training).
- [#5592](https://github.com/Niko1221/Strata/pull/5592) (DobricLilujun): Jetson AGX Thor (sm_110) aarch64 CUDA machine code.
- [#3607](https://github.com/Niko1221/Strata/pull/3607) (demetree): native Windows Intel Arc SYCL support (open since 10-21).
- [#5647](https://github.com/Niko1221/Strata/pull/5647) / [#5644](https://github.com/Niko1221/Strata/pull/5644) (kelvincht): Metal elastic expert arena (follows free memory) plus reserved K/V pools instead of up-front allocation.
- [#4522](https://github.com/Niko1221/Strata/pull/4522) / [#4534](https://github.com/Niko1221/Strata/pull/4534) / [#4549](https://github.com/Niko1221/Strata/pull/4549) (cobra91): the un-ported Strata-DualGPU pieces (see 3.4).

**Qwen3.6 issues still open upstream** (relevant because Qwen3.6-35B-A3B is one of our daily models):

- [#5069](https://github.com/Niko1221/Strata/issues/5069): Qwen3.6-35B-A3B lacks `--kv int8`/`--kv-resident` and a long-context decode path (Flash-Next has them).
- [#2464](https://github.com/Niko1221/Strata/issues/2464): Qwen3.6 decode falls with context (V100: 109 → 14.6 tok/s at 114K).
- [#5609](https://github.com/Niko1221/Strata/issues/5609) (closed 11-08): q36 MTP block loads but never drafts. Fixed upstream; verify our base has the fix.
- [#5551](https://github.com/Niko1221/Strata/issues/5551): q36 on the qwen35moe 19B-A3B shape, three localized walls.
- [#1402](https://github.com/Niko1221/Strata/issues/1402): Qwen3.6 plus Ornith-1.5-35B-A3B as the small-model line for 8-12 GB cards.

### 4.3 What upstream already merged (late September wave, for orientation)

#43 AVX2 multi-token i-quant row kernels; #109 batched verify-window kernels (+~10% decode, bit-identical); #21 Q4_0 KV cache with FWHT-256 Hadamard rotation; #54 GSQ-RCO-Coder pruned experts (256/512); #8 conversation cache; #62 hybrid-CPU pool workers (P-cores plus half the E-cores); #67 OrcaRouter BF16 packing; #89 bulk-read expert arena (MSVC 4095-byte ifstream split); #81 sm_80+ runtime accept; #122 llama.cpp discovery endpoints; #118 routing traces; #107 cached-prompt token reporting; #126 Task Scheduler 24x slower expert load; #42 MEM_LARGE_PAGES privilege; #44 a real H2D bandwidth probe for `pcie_frac`; #64 parity tool hard-fail. Most of these predate or are already inside our base; the list matters as a map of what the project merges.

### 4.4 The one llama.cpp item

**llama.cpp #31751** ([PR](https://github.com/ggml-org/llama.cpp/pull/31751)): a one-pragma fix for RDNA3 MMQ register spills, +36% at 17-32-token verify batches on gfx1100. Our verify windows are exactly that size. The Avalon report ranked it "Add now" and no pancake counterpart is recorded anywhere. Nobody has checked whether Strata's or our MMQ kernels share the spill pattern. This is the cheapest unexamined kernel item in the entire report.

## 5. Prioritized Gap Analysis: What pancake-strata Is Missing and Should Take Next

Ranked by (impact x evidence) within each priority class. Items marked "ours" reference MOD_PLAN item numbers.

### P0: Bugfixes and zero-risk wins (do first, in this order)

1. **Check #5655 against our tree** (`route_prior` sentinel expert ID leak + OOB write on degenerate logits). We ship `STRATA_ROUTE_PRIOR` (turbo/turbo-hip) and `STRATA_ROUTE_PRIOR_DEPTH`. If our base `146df3c4` predates the fix, we have the bug. Read the PR, reproduce on degenerate logits, port the fix regardless of upstream merge state.
2. **Verify the #5609 q36 MTP fix is in our base** (MTP block loads but never drafts; fixed upstream 11-08). One diff, five minutes.
3. **llama.cpp #31751 spill-pattern check**: grep our MMQ kernels for the J=32 register-spill pattern; if present, one pragma, +36% at our verify batch sizes on gfx1100. Never evaluated.
4. **#4625** (`STRATA_POOL_SPIN_US=0` actually means zero): trivial, and we tune pool spin.
5. **#5466** (report need/free MiB on expert-buffer/staging OOM): our arena builds trigger systemd-oomd; better OOM messages would have saved hours.
6. **#5518** (HIP `alignas(16)` int4 staging compile fix): two sites; keeps future HIP builds honest.

### P1: Single-GPU RDNA3/CUDA decode and prefill wins

1. **#4005** (Vop3p): prefill sparse copy events + dedup slot waits, +36% Windows HIP prompt @4K. Closed unmerged, so the code needs resurrection. Our prompt path is the one that degrades after long contexts.
2. **bahooo22 cluster #5595/#5467/#5468**: grouped Q4_K down-expert kernel, mma.m16n8k16 prompt FA, super-chunk layer-major prefill. All q36, all matching our daily model.
3. **StrataGP's 16 patches** (gputier): +5-15% decode, +42-49% prompt on RTX 5090. Diff against our base; expect overlap with the bahooo22 cluster and with our rejected int8-WMMA attention.
4. **Juud_engine's two CPU tweaks**: shortened worker spin after the expert batch, skip activation quant for GPU-resident tokens, +9.6-15.6% decode on a 4090. Our pool is bandwidth-bound, so overhead cuts are the right shape.
5. **#5475** (`STRATA_HC_PERSIST_Q8`): int8 copies for the persistent hyper-connection read. Check overlap first: our rdna3 port already has the hyper-connection read with Q8; the persistent variant may be new.
6. **#5535** (xyzzing): int8-WMMA prompt attention on gfx1100, with recipe and validation. We rejected int8-IMMA attention for the main engine (attention 6.7% of the window), but this is prompt-side on our exact GPU with a validation recipe; cheap to measure, keep expectations low.
7. **#2392** (skuridin): gfx1101 hipBLASLt tuning table. We already carry the gfx1100 `100401` table; this extends coverage to 7800 XT-class cards.
8. **#5471** (R-Okauchi): faster Q4_K/Q5_K decode GEMVs. Our DeepSeek V4.1 path benefits directly if the kernel shapes match.
9. **#3925** (schmittcn): load q36 GGUFs whose `ssm_alpha`/`ssm_beta` are K-quants or IQ4_XS. Removes a class of packs that fail to load today.

### P2: Startup, memory, and agent-server wins

1. **#3509** (`STRATA_ARENA_REUSE=1`): attach a shared expert arena across restarts instead of re-copying. Our restarts pay 28-65 s for the 33 GiB arena read on ntfs3; this attacks exactly that, and it composes with item 38 (Linux direct arena loader, est. 10-14 s) as either/or.
2. **Maya's direct parallel expert reads** (`MAYA_EXPERT_DIRECT`/`MAYA_EXPERT_STAGERS`, +26% decode on a RAM-short PC) plus our item 38: the startup decision from 2026-11-08 (oomd vs `STRATA_RAM_KEEP_GIB=20`) should be taken together with this.
3. **#5643** (kelvincht): restore reasoning a client dropped so the next turn reuses the cached prompt prefix. Our opencode data shows ~18% of turn time in re-reads; this is one of the few server-side patches that attacks them directly. Gate on our restore-equivalence test (item 35).
4. **#5516** (rwkeyes): in-place TUNE plus conversation-cache routing by traffic shape. We closed upstream's `/v1/tune` as low value, but the cache-routing half pairs with our `STRATA_ROUTE_PRIOR` work.
5. **Maya's weight-share miss skip** (`MAYA_GLM_MISS_SKIP_W`, +7-13% decode at KL 0.037-0.052 on GLM): we only have rank-based tail skip. Quality-gate it like turbo-hip.
6. **#5566** (yeyeyeason): opt-in cold store for the expert arena. Memory elasticity for the arena; pairs with #5647/#5644's elastic-arena idea (Metal, but the design ports).
7. **#5623** (shoucandanghehe): shared immutable host KV prefixes across streaming sessions. Our 4 parked conversations cost 5.4-6.1 GiB; shared prefixes are the cheapest compression available.
8. **FastStrata/poly `--conversation-disk` on the main engine** (item 39): restore in ~0.5-1 s after restart; FastStrata measured a 70K conversation resumed in 7.2 s instead of 83 s.
9. **basalt sleep-in-place** (`64167ba73`): parked, but it remains the only item that closes the `--vram-elastic`-refuses-with-pinned-complement gap.
10. **#5561** (`STRATA_EXPERTS_SEQ_READAHEAD`): parked in our ledger (only if experts fault from NVMe; the arena removes faults). Revisit only if item 38 slips.
11. **Smaller server items**: #5593 (`STRATA_PROMPT_DIRECT=1` without a RAM budget), #4972 (file-tier defaults off when GPU caches hold every expert), #5637/#5437 (inhibit sleep during inference), #5588 (`--ple-gguf`), #4777 (dynamic `route_prior` per request), #3971 (keep thinking tokens past a budget cut), #5628 (stream tool args at JSON token boundaries), #5512 (thinking skip on short-direct prompts; we called it a heuristic, it is still opt-in upstream).

### P3: Unported features from known forks (quality-gated or parked)

1. **`STRATA_VERIFY_DEFER_CPU` on HIP** (Speedy port, item 8.3): coded, up to 1.45x per KTransformers, blocked on non-finite logits; the 6-run bisect is queued. Plus the rank guard `STRATA_VERIFY_DEFER_CPU_TOP` (item 28, 5% tested). The single biggest lossy lever we own.
2. **turbo-hip quality-gate decision** (colibri port, item 7): preset exists, token forcing INCONCLUSIVE, Test 30%. Owner decision.
3. **Branch merge decisions**: `feat/bfla-hip` (3.15x at 128K prompt, recall 48/48), `feat/bitdecoding-hip` (~2-3%, parked as non-default), `feat/disco` (+SpecFit item 36, not started), and merging `port/rdna3-decode` → `main`. All owner calls, all pending.
4. **alesha-pro's Mirai trellis "S" quant** (#3384): evaluate offline first (section 3.9).
5. **#4876** (`STRATA_GDN_TC=2`, tensor-core GDN recurrence on sm_75, 2.2x): CUDA-only; relevant to our 3060 path.
6. **#3277** (native-IQ prefill on SM75) and **#4854** (AVX1 Q2_0 rows): widen hardware coverage; low priority for us.
7. **#5545 / #5517**: GLM-5.3-Flash engine path and the qwen3next geometry-derived trunk. Track both; they change how we serve GLM and future Qwen families.
8. **DeepSeek V4.1 cluster remainder**: #5470 (`DS4_PROF_WIN` profiler), #5596 (`DS4_CACHE_ROUTE`), #5453/#5472 (DGX Spark UMA tier and skip-rank; the skip-rank idea may generalize to our file tier).
9. **Maya leftovers**: `--calibrate`-style startup sweep, drafter auto-keep, prompt memo (~6% of follow-up TTFT), one-GPU multi-request batching.
10. **University-series pending items** (ours): learned thinking stop (21), effort defaults (12), loop-guard counters (33), rumination signal (34), restore-equivalence test (35), GDN chunk 64 (18), IQ1_M AVX2 kernel (24), Coder-variant A/B (25), semantic oldString constraint (26).

### P4: Dual-GPU and Phase 12 wins (blocked on hardware, not on research)

1. **Strata-rdna3's tensor-parallel decode** (`src/core/tp.cpp`, `--tensor-parallel 2`; 129-170 tok/s on 2x 7900 XTX), peer expert transfer / RCCL, kfdq idle-queue parking (98+91 W → 5+13 W idle), and the GDN megakernel revisited under TP.
2. **Strata-DualGPU's remaining pieces**: cobra91's #4522/#4534/#4549, plus #859's pipelined verify windows (+16% mean, +30% copy-heavy) if our base lacks them. Issue #1352 is the warning: partial merges leave upstream ~2x slower than the fork.
3. **StrataFast `--prompt-split`** (1.24-1.55x on long prompts).
4. **gfx12 (RDNA4) re-measure** of the gfx11-tuned ports (item 62), and DeepSeek V4.1 on gfx1100.
5. **Watch**: Infernix's long-context numbers (+23% at 262K on a 5090) as the bar for any future claim; XeStrata and #3607 if Intel GPUs ever matter to us; strata-mlx if a Mac enters the picture.

### What we are NOT taking (already rejected, recorded to prevent re-litigation)

The full rejection ledger lives in MOD_PLAN.md lines 1073-1286. Highlights: the upstream expert-cache PR line (#29887 and kin), KVA approximate prefill, PLE-in-RAM, DFlash2 for our hardware, int8-WMMA decode attention as a default, ContextPilot (1.6% savings, breaks GDN state), draft trees, the `.basalt` pack format, NVFP4 (Blackwell-only), dense-model offload, and all multi-user/batch items until the server grows a second slot. Re-checking these duplicates work; the ledger is the canonical record.

## Appendix: Sources and verification notes

- **Local audit**: MOD_PLAN.md (phases 2-14, "Not adopted" ledger at lines 1073-1286, Maya robustness section at 1287-1311, 2-GPU table at 1409-1424), PANCAKE_STATUS.md, the five research notes under `research_notes/Avalon upgrades from upstream projects/`, and the report `reports/Avalon upgrades from upstream projects.md`. Branch merge state verified with git on 2026-11-09.
- **Live GitHub pass**: 2026-11-09, via the repo API (forks sorted by stargazers and by newest), PR/issue search, author-profile sweeps (cobra91, Vop3p, xyzzing, lwYeo, kelvincht, rwkeyes, bahooo22, R-Okauchi), code search on Strata's distinctive symbols, and web search (stringers.io roundup, prismix.dev, r/LocalLLaMA, HF Forums, dev.to).
- **Verification caveats**: (a) star counts and PR states are from 2026-11-09 and will drift; (b) author-measured numbers (StrataGP, StrataQ38FN, Juud_engine, DualGPU, Infernix, Gaudi2, XeStrata) are claims from hardware we do not own, taken from READMEs, issues, and posts, not reproduced by us; (c) the fork list cannot be exhaustive at 4,627 forks; the symbol searches surfaced 10+ derivatives beyond the fork tree, and community coverage names several repos we did not verify individually (section 3.11); (d) session `ses_e3f832da0ffeKUiutIsc46iNxq` (fork inspection) ended mid-run without a final summary; its findings are covered by the live pass summarized here, and its confirmed intermediate results (4,516 forks at 13:27, the downstream target `rocktboy1982/pancake-strata`, 10+ symbol-search discoveries) are consistent with the numbers above.
- **Companion documents**: the Romanian version of this report lives at `reports/Projects derived from Strata.md` (and a copy at `/home/dan/reports/`). The port ledger and status live in MOD_PLAN.md and PANCAKE_STATUS.md.
