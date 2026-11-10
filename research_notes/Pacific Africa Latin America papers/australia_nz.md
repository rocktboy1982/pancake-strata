# Australian and New Zealand AI papers (Nov 2022 - Nov 2026) relevant to pancake / Avalon

Method: affiliation-filtered OpenAlex queries (`institutions.country_code:AU|NZ`, from 2022-11, about 40 topic keywords), then arXiv API metadata and abstracts for every candidate, then OpenAlex author-affiliation checks. None of the entries below appear in /home/dan/reports/ (checked with grep for each title and key term; SnapKV and generic "EDIT" were the only hits, and both were unrelated). Lead = first author at an AU/NZ institution. Engine context: Qwen3.8-Flash-Next 125B MoE (GDN + sparse attention, 512 experts, MTP), one 7900 XT, CPU computes missed experts (about half of decode), batch 1, opencode agent.

Overall: AU/NZ output in this area is thin. There are many vision, VLA, multi-agent and edge-computing papers, but few systems papers on single-GPU MoE decode. Only **SpecFit** (Monash co-author) is directly on pancake's critical path. **DuoServe-MoE** (Sydney), **counter-causal KV eviction** (ANU) and **OSCAR** (Sydney) are secondary ideas. Most of the rest is evaluation or SE-agent material for the opencode side.

## Q1. Inference-efficiency papers (MoE offload, speculative decoding, KV, quantization, kernels)

### Takeaway
SpecFit is the one paper aimed at pancake's actual bottleneck: draft tokens that pull extra offloaded experts. Its throughput-aware choice of draft size refines what pancake's DraftPolicy already does per window width. Expected gain is a few percent of decode, not tens of percent. Everything else targets batch serving, long-context KV or dense models, and transfers poorly to a GDN-hybrid at batch 1.

### Cited Findings

**1. SpecFit: Not All Draft Tokens Cost the Same in Offloaded MoE Inference.** Hongliang Cao, Xiuyu Li, Siqi Wang, Ziming Wang, Jingmin Zhu, Hailong Yang. Beihang is lead; Xiuyu Li and Jingmin Zhu are at Monash (co-authors). arXiv 2610.15417, 2026-10-11. Code: github.com/Chlience/SpecFit — [arXiv](https://arxiv.org/abs/2610.15417); affiliations per [OpenAlex](https://api.openalex.org/works/doi:10.48550/arxiv.2610.15417)
- Core idea: with offloaded experts, each verified draft token brings its own experts, so a larger draft tree means more H2D transfers. Speculative decoding "is sometimes even slower than autoregressive decoding" in this setting. SpecFit picks the verified subtree that maximizes predicted throughput instead of acceptance length. Committed tokens are estimated from draft path probabilities. Cycle time is predicted from node count plus the per-layer *union* of experts the subtree activates, which is a monotone submodular coverage function, so each node pays only for experts it adds. An exact search runs over ancestor-closed subtrees after each draft level and also decides when to stop drafting. The target model is unmodified — [arXiv](https://arxiv.org/abs/2610.15417)
- Numbers: Qwen3-30B-A3B on CNN/DM and a GSM8K subset. It is 21.6% and 7.0% faster than EAGLE-3 with the same expert cache, and 1.97x and 1.72x as fast as vanilla EAGLE-3 — [arXiv](https://arxiv.org/abs/2610.15417)
- Verdict for pancake: **applicable, medium priority**. `src/spec/draft_policy.cpp` already prices each window width with measured round time (EMA per width and bucket, `mtp_tokens(t)/cost_ms(t)`). Its own comment notes "+10 ms per token with every missed expert on the CPU". So pancake has the coarse version. What SpecFit adds is a *per-round* cost conditioned on the expert union. If the router outputs (or MTP-head routing guesses) for draft positions show that token k+1 adds no new missed experts, a longer window is nearly free; if it adds several CPU-side experts, stop early. Pancake drafts chains, not trees, so the "subtree" search reduces to choosing T with a submodular cost. Expected gain: low single digits to perhaps 5-8% decode on prompts where window length currently varies. The paper's +7-22% over cache-aware EAGLE-3 is an upper bound, since pancake already measures cost per width. Avalon: same idea applies to its MTP draft-length logic if CPU-offloaded experts are used.

**2. DuoServe-MoE: Dual-Phase Expert Prefetch and Caching for LLM Inference QoS Assurance.** Yuning Zhang, Grant Pinkert, Nan Yang, Yanli Li, Dong Yuan. All University of Sydney (lead). arXiv 2509.07379, 2025-09 — [arXiv](https://arxiv.org/abs/2509.07379); [OpenAlex](https://api.openalex.org/works/doi:10.48550/arxiv.2509.07379)
- Core idea: prefill activates experts densely and decode sparsely, so one caching policy either blows peak memory (prefill) or inflates tail latency (decode). For prefill, a two-stream pipeline overlaps expert prefetch with non-MoE compute. For decode, a lightweight *layer-level predictor trained offline from activation traces* prefetches only likely experts. Up to 5.34x TTFT and 7.55x end-to-end latency over baselines (unnamed in abstract) — [arXiv](https://arxiv.org/abs/2509.07379)
- Verdict: **low**. Pancake already separates prefill and decode, and the dGPU report series already covers trace-trained expert predictors (fMoE etc.). Use it only as a comparison point for an offline-trained per-layer predictor test. No code link was found in the abstract. Expected gain beyond existing prefetch: about 0-3%, unproven on 512-expert routing.

**3. Dynamic Depth Decoding (DDD): Faster Speculative Decoding for LLMs.** Oscar Brown (ANU, lead), Zhengjie Wang, Andrea Do, Nikhil Mathew, Cheng Yu. arXiv 2409.00142, 2024-08 — [arXiv](https://arxiv.org/abs/2409.00142); [OpenAlex](https://api.openalex.org/works/doi:10.48550/arxiv.2409.00142)
- Core idea: EAGLE-2's draft tree gets a dynamic depth, so drafting stops when the tree's confidence drops. It extends EAGLE-2's average speedup over EAGLE by 44%, to 3.16x average — [arXiv](https://arxiv.org/abs/2409.00142)
- Verdict: **low**. Pancake's DraftPolicy already stops by confidence and cost. It is a confidence-only rule, which SpecFit supersedes for offloaded MoE. No code is stated.

**4. GPU-Resident Speculative Decoding for In-Browser LLM Inference with WebGPU.** Wei Song (Griffith, lead), Yuxin Cao (NUS), Ziqi Ding (UNSW), Xiao Cheng (Macquarie). arXiv 2610.39761, 2026-10-29 — [arXiv](https://arxiv.org/abs/2610.39761); [OpenAlex](https://api.openalex.org/works/doi:10.48550/arxiv.2610.39761)
- Core idea: a GPU-resident loop does drafting, verification, acceptance and KV maintenance with no per-token CPU decision. A *draft-free context-mixing drafter* reuses page content, conversation history and session memory. A device-aware controller adapts speculation width to measured costs. Speedups: 3.5-5.4x over WebLLM on a laptop GPU, 4.8-6.2x on an iGPU, 4.8-8.6x on a workstation GPU. The GPU-resident loop plus speculation adds 1.8-3.2x over optimized kernels. Output is exact versus greedy. Optional draft heads reach 10.2x peak — [arXiv](https://arxiv.org/abs/2610.39761)
- Verdict: **low-to-medium, idea-level only**. Pancake already has prompt-lookup, SuffixDrafter and a conversation cache. The "session memory" drafter (n-gram sources mixed across page, history and session) is the only possibly new ingredient; check whether SuffixDrafter already mixes the conversation and session corpus with weights. The GPU-resident loop is not viable because missed experts force CPU work every token. Expected gain: about 0-2% from drafter mixing.

**5. Back from the Future: KV Cache Management by Counter-Causal Surprise.** Stephen Gould (ANU, lead), Anton van den Hengel (affiliation blank in OpenAlex; historically Adelaide). arXiv 2607.27600, 2026-07-30. Code: github.com/metacognitionai/counter_causal — [arXiv](https://arxiv.org/abs/2607.27600); [OpenAlex](https://api.openalex.org/works/doi:10.48550/arxiv.2607.27600)
- Core idea: past tokens that are well predicted from *more recent* tokens are redundant. Eviction scores come from rerunning the model over cached K/V with a counter-causal mask, where each position attends only to its future. It needs no training. A single-layer (last layer) approximation makes refresh cheap. The authors report "competitive or improved" accuracy versus SOTA, with no headline number in the abstract — [arXiv](https://arxiv.org/abs/2607.27600)
- Verdict: **low**. Only the sparse-attention layers of Qwen3.8-Flash-Next hold KV, and GDN layers have fixed-size state, so KV is not the decode limiter at opencode context lengths. It could matter only if very long sessions push attention KV out of VRAM. Not worth implementing now.

**6. OSCAR: Offline Spectral Covariance-Aware Rotation for 2-bit KV Cache Quantization.** Zhongzhu Zhou (University of Sydney, lead), Donglin Zhuang (Sydney), Jisen Li (UIUC), Ziyan Chen (Sydney), Shuaiwen Leon Song, Ben Athiwaratkun, Xiaoxia Wu (the last three unaffiliated in OpenAlex; Athiwaratkun and Wu are industry). arXiv 2605.17757, 2026-05-18 — [arXiv](https://arxiv.org/abs/2605.17757); [OpenAlex](https://api.openalex.org/works/doi:10.48550/arxiv.2605.17757)
- Core idea: offline attention-aware covariance estimation gives fixed rotations and clipping for INT2 KV, with a custom INT2 attention kernel compatible with paged KV (SGLang/vLLM). On Qwen3-4B-Thinking and Qwen3-8B the BF16 gap is 3.78 and 1.42 points, while naive rotation INT2 collapses to about 0. It is near BF16 on Qwen3-32B and GLM-4.7 358B, robust on RULER-NIAH to 128K, gives about 8x KV memory reduction, and up to 3x batch-1 decode over BF16 — [arXiv](https://arxiv.org/abs/2605.17757)
- Verdict: **low, and closely related to an already-failed test**. TurboQuant KV gave no gain in pancake, for the same reason: KV bandwidth is not the bottleneck in a GDN hybrid with expert misses. Revisit only for 100K+ contexts. Avalon (llama.cpp) users with dense attention models might benefit more.

**7. Flash-LLM (VLDB 2024) and FP6-LLM / TC-FPx.** Haojun Xia (University of Sydney, lead in both), with Donglin Zhuang and Zhongzhu Zhou (Sydney), Shuaiwen Leon Song, plus Alibaba (Flash-LLM) and Microsoft DeepSpeed (FP6-LLM). arXiv 2309.10285 (2023-09) and 2401.14112 (2024-01). Code: github.com/usyd-fsalab/fp6_llm — [Flash-LLM](https://arxiv.org/abs/2309.10285); [FP6-LLM](https://arxiv.org/abs/2401.14112); [OpenAlex](https://api.openalex.org/works/doi:10.48550/arxiv.2401.14112)
- Flash-LLM: "Load-as-Sparse, Compute-as-Dense" for unstructured-sparse skinny GEMMs on Tensor Cores. SpMM is 2.9x faster than Sputnik and 1.5x faster than SparTA, and end-to-end is up to 3.8x and 3.6x better than DeepSpeed and FasterTransformer (OPT-30B to 175B) — [arXiv](https://arxiv.org/abs/2309.10285)
- FP6-LLM: a unified Tensor Core kernel for irregular-bit-width float weights (FP5/FP6), with bit-level pre-packing and fast dequant. LLaMA-70B runs on one GPU at 1.69-2.65x normalized throughput over FP16 — [arXiv](https://arxiv.org/abs/2401.14112)
- Verdict: **low**. These are CUDA Tensor Core (A100) designs for batch serving. Pancake's Q2_0 GSQ-RCO kernels already solve irregular-bit dequant for gfx1100, and the CPU half of decode is untouched. The one transferable idea is the "load compressed, decompress in registers, compute dense" principle, which pancake already uses.

**8. Monash ZIP Lab (Bohan Zhuang, Jing Liu, Yefei He, Akide Liu; Gholamreza Haffari): KV and quantization line.**
- MiniCache (NeurIPS 2024): Akide Liu (Monash, lead) et al. Merges KV across adjacent middle-to-deep layers by interpolating direction while keeping magnitude, and keeps distinct tokens unmerged. Up to 5.02x compression, about 5x throughput, 41% memory reduction versus FP16, near-lossless. Code at minicache.vmv.re — [arXiv 2405.14366](https://arxiv.org/abs/2405.14366)
- ZipCache (NeurIPS 2024): Yefei He (lead, Zhejiang) with Jing Liu and Bohan Zhuang (Monash co). Channel-separable token-wise KV quantization plus normalized-attention saliency, compatible with FlashAttention. 4.98x KV compression with a 0.38% drop on Mistral-7B GSM8k, and on LLaMA3-8B at 4K: -37.3% prefill latency, -56.9% decode latency, -19.8% memory. Code: github.com/ThisisBillhe/ZipCache — [arXiv 2405.14256](https://arxiv.org/abs/2405.14256)
- QLLM (ICLR 2024): Jing Liu (Monash, lead) et al. Adaptive channel disassembly/assembly moves activation-outlier magnitude across channels for low-bit PTQ. Code: github.com/ziplab/QLLM — [arXiv 2310.08041](https://arxiv.org/abs/2310.08041)
- ME-Switch (2024 tech report): Jing Liu (lead). Salient-input-channel-aware mixed-precision delta compression for switching between fine-tuned model "experts" (whole models, not MoE experts) — [arXiv 2406.09041](https://arxiv.org/abs/2406.09041)
- Verdict: **none for decode**. Pancake's KV is small (GDN hybrid), TurboQuant KV already failed, and weight quantization is fixed by GSQ-RCO Q2_0. QLLM-style activation outlier handling targets W4A4/W6A6 activation quantization, which pancake does not use. ME-Switch is about multi-model serving.

**9. Towards Efficient Mixture of Experts: A Holistic Study of Compression Techniques (TMLR).** Shwai He (lead, U Maryland), Daize Dong, Liang Ding and Ang Li (Liang Ding at University of Sydney, co-author). arXiv 2406.02500. Code: github.com/CASE-Lab-UMD/Unified-MoE-Compression — [arXiv](https://arxiv.org/abs/2406.02500); Sydney affiliation per [OpenAlex](https://api.openalex.org/works?filter=title.search:holistic%20study%20of%20compression%20techniques)
- Layer Drop (remove whole MoE layers), Block Drop, Expert Slimming plus Trimming. On Mixtral-8x7B: 6.05x speedup, 77.1% less memory, over 92% performance retained — [arXiv](https://arxiv.org/abs/2406.02500)
- Verdict: **low**. Losing 8% quality is unacceptable for a coding agent. A Layer Drop ablation over Qwen3.8's MoE layers could still be cheap to *measure* (attention/MoE layer redundancy). Not recommended for production.

**10. BudgetDraft (sparse-KV drafter training).** Liang He (lead, Shanghai), with Jingbo Wen (Sydney) as co-author. arXiv 2606.00144. Up to 6.55x, 4.46x and 2.10x over AR at 4K, 8K and 16K — [arXiv](https://arxiv.org/abs/2606.00144). Verdict: **none**. It requires training a separate drafter, while pancake uses MTP plus lookup.

### Inferences
- SpecFit's main message (draft-token cost is not uniform under offloading) supports pancake's current direction. The concrete next step is to add a "new missed experts per extra draft position" term to DraftPolicy's per-width cost, using routing logits already computed for verified positions or MTP-head routing guesses. That makes the cost per round instead of per width. Whether draft-position expert sets can be predicted cheaply *before* verification is not stated in the abstract and must be checked in the paper or code (github.com/Chlience/SpecFit).
- KV-compression papers (ZIP Lab, OSCAR, counter-causal) are well done but miss pancake's bottleneck. This matches the earlier TurboQuant null result.

### Gaps
- No AU/NZ-led paper found on CPU-side expert compute (AVX2/DDR4), NVMe expert streaming, GDN/linear-attention kernels, MTP-head drafting, or HIP/ROCm kernels.
- SpecFit's method for predicting the expert union of not-yet-verified draft tokens is not in the abstract; not verified.
- DuoServe-MoE's baselines and code availability are not in the abstract.

## Q2. Caching, serving and scheduling (prefix/KV reuse, hybrid-state cache recovery)

### Takeaway
Melbourne (Buyya's CLOUDS lab), UNSW and Griffith produced several 2026 prefix-cache and serving papers. They are multi-tenant or multi-replica and mostly irrelevant to single-user batch 1. The exception is two UNSW correctness reports on hybrid-model cache recovery, which are a useful test checklist for pancake's conversation cache with GDN boundary checkpoints.

### Cited Findings
- **Validating Hybrid-State Cache Recovery for GLM-5.3-Flash with vLLM and LMCache.** Frank Li (UNSW, sole author). arXiv 2609.15030, 2026-09. Transfers can "succeed while a hybrid language model resumes from an inconsistent state". In a complete-hit case, state was restored for the full prompt but the scheduler credited one fewer token. The fix was strict-prefix lookup plus matched checkpoint scheduling. Agreement with recomputation went from 34/36 to 36/36 generations. CPU reload cut TTFT by 46-64% versus cold recompute — [arXiv](https://arxiv.org/abs/2609.15030); [OpenAlex](https://api.openalex.org/works/doi:10.48550/arxiv.2609.15030)
- **Shared KV Caching for Replicated 27B Inference.** Frank Li (UNSW). arXiv 2609.15021. Found a CUDA-stream-dependency bug in a raw-pointer fallback. Cross-replica TTFT dropped from 31.7 s to 0.6 s at 128K — [arXiv](https://arxiv.org/abs/2609.15021)
- **Preserving Provenance in Shared KV Caches.** Wei Song (Griffith, lead), with Macquarie co-authors. arXiv 2609.38706. Shared-tier keys that use only token content let incompatible contexts collide. Cross-adapter collisions cut accuracy from 0.94 to 0.64, and incompatible KV representations drop reasoning accuracy to 0 — [arXiv](https://arxiv.org/abs/2609.38706)
- **PrefixShield** (Zhiyu Wang and Rajkumar Buyya, Melbourne; arXiv 2608.01657): multi-tenant admission responsibility, +9.39 pp victim hit ratio versus LRU — [arXiv](https://arxiv.org/abs/2608.01657). **PrefixPlace** (same authors; arXiv 2608.01655): prefix placement across workers, +40.3% materialization saving versus vLLM APC — [arXiv](https://arxiv.org/abs/2608.01655)
- **Multi-Layer Scheduling for MoE-Based LLM Reasoning** (Yifan Sun, Melbourne lead; Haffari, Monash; Buyya; Toosi; arXiv 2602.21626): up to -17.8% TTFT and -13.3% TPOT versus vLLM — [arXiv](https://arxiv.org/abs/2602.21626). **Gimbal: Coordinated Scheduling for MoE LLM Serving** (same group; arXiv 2606.15177): -42.9% TTFT and -33.3% TPOT versus vLLM, +3.0% throughput — [arXiv](https://arxiv.org/abs/2606.15177)
- **How Query Visibility Changes KV-Cache Compression Rankings** (Daming Luo, Christy Liang, Junyu Xuan, all UTS; arXiv 2607.11942): under query-agnostic compression (compress once, reuse), only KeyDiff consistently beats trivial baselines (31 of 36 cells). SnapKV loses to "keep start + recent window" (-0.066) — [arXiv](https://arxiv.org/abs/2607.11942)
- **StepCache** (Nouri, Lincoln University NZ; Research Square preprint 2026-03): step-level reuse with lightweight verification. Abstract not verified — [DOI](https://doi.org/10.21203/rs.3.rs-9077245/v1)

### Inferences
- Verdict for the UNSW hybrid-state reports: **useful as a test checklist, zero speed gain**. Pancake's conversation cache restores GDN state at boundary checkpoints, which is the same class of bug (state restored for N tokens while the scheduler credits N-1). A regression test is cheap: a full-hit restore must match cold recompute token-for-token over many lengths. It also helps with the MTP and GDN-replay interaction.
- The UTS audit means that if a KV-eviction mode is ever added for the conversation cache, where compression happens before the next user turn is known, plain "sink + recent window" is a strong baseline that published methods often fail to beat.
- The Melbourne scheduling and prefix-placement work is multi-tenant or multi-GPU and does not apply to batch 1.

### Gaps
- The StepCache full text was not checked.

## Q3. Reasoning efficiency, agents and SE evaluation (opencode side)

### Takeaway
The Monash and Melbourne SE groups (Tantithamthavorn, Thongtanunam, Bach Le) and CSIRO Data61 produce issue-resolution agents and benchmarks. They are useful for evaluating pancake under opencode, not for speed. Reasoning-efficiency entries from AU/NZ are mostly co-authored and offer no ready mechanism beyond existing thinking budgets.

### Cited Findings
- **XAgent: eXecution-guided Agentic AI for GitHub Issues.** Hieu Huynh, Patanamon Thongtanunam, Michael Fu, Bach Le (University of Melbourne, lead), Kla Tantithamthavorn (Monash). arXiv 2609.09769, 2026-09. Uses dynamic execution behaviour and extra program context for localization and validation. SWE-bench Lite: 62.0% resolved, 72.8% function-localization accuracy, "cost efficient", 7 issues solved that top baselines miss — [arXiv](https://arxiv.org/abs/2609.09769). Verdict: an opencode workflow idea (run a reproduction test before patching). No engine gain, and no code stated.
- **BigCodeBench (ICLR 2025 Oral).** Terry Yue Zhuo (lead; Monash/CSIRO Data61 per OpenAlex co-listing) and the BigCode community. 1,140 tasks with function calls across 139 libraries, 5.6 tests per task, 99% branch coverage — [arXiv 2406.15877](https://arxiv.org/abs/2406.15877). Verdict: **useful as a quick quality regression set** for quantization or drafting changes (e.g., checking that Q2_0 or DISCO changes do not hurt code quality). Its tool-call style suits the tool-call-recovery path.
- **SWE-EVO** (Tue Le et al.; Dung Nguyen Manh at Melbourne, co-author; arXiv 2512.18470): 48 long-horizon tasks averaging 21 files and 874 tests. GPT-5.4 plus OpenHands scores 25% versus 72.8% for GPT-5.2 on SWE-bench Verified. Adds a "Fix Rate" partial-progress metric — [arXiv](https://arxiv.org/abs/2512.18470). Verdict: an evaluation option for long sessions, where the conversation cache, loop guard and thinking budgets matter most.
- **Beyond Localization: Repository-Level RAG-APR** (Pengtao Zhao, Bach Le, Feng Liu, Melbourne lead; arXiv 2603.29067): even Oracle localization leaves success below 50% on SWE-bench Lite. Best-of-K headroom saturates quickly — [arXiv](https://arxiv.org/abs/2603.29067)
- **Not All Errors Are Equal: Consequence-Aware Reasoning Compute Allocation** (Liang He, lead Shanghai; Jingbo Wen and Ziqi He at Sydney; arXiv 2606.04402): routes premium compute by predicted failure consequence on SWE-bench Lite — [arXiv](https://arxiv.org/abs/2606.04402). Verdict: a conceptual fit for thinking budgets (spend more thinking on high-consequence tool calls such as edits, less on reads). Not a drop-in.
- **Think More or Read More? A Token-Budget Theory** (Manh Nguyen, Sunil Gupta, Hung Le, Deakin lead; arXiv 2610.23246): in a fixed window, every token read is a token not spent reasoning. A closed-form model predicts the optimal debate group size — [arXiv](https://arxiv.org/abs/2610.23246). Verdict: theory only.
- **EDIT / From Long to Short** (Wei Han lead; Geng Zhan and Chenyu Wang at Sydney; Bryan Hooi, NUS; arXiv 2509.06174): constraint-guided test-time search for the shortest correct reasoning path — [arXiv](https://arxiv.org/abs/2509.06174). Verdict: needs multiple samples, so it is unfit for batch 1.
- Other agent and SE items seen but not applicable: ChatDev (Sydney co-listing in OpenAlex, mainly Tsinghua) — [arXiv 2307.07924](https://arxiv.org/abs/2307.07924); A3-CodGen (ANU lead, CSIRO, repo-level code reuse, TSE) — [DOI](https://doi.org/10.1109/tse.2024.3486195); Confidence vs Competence in agentic SE (CSIRO lead) — [DOI](https://doi.org/10.1145/3822603); HyperAgent tool-schema hypergraphs (UNSW lead) — [arXiv 2608.02650](https://arxiv.org/abs/2608.02650).

### Inferences
- For the opencode side, the actionable AU/NZ items are evaluation assets: BigCodeBench-Hard as a quality gate and SWE-EVO or SWE-bench Lite subsets for long sessions. XAgent's "execute and reproduce before patching" is a prompt or skill change.
- Nothing here improves tool-call recovery or the loop guard directly.

### Gaps
- No AU/NZ paper found on tool-call grammar or recovery, agent loop detection, or thinking-budget control for local models.
- Terry Yue Zhuo's exact primary affiliation on BigCodeBench (Monash vs CSIRO Data61) was taken from OpenAlex co-listing and not checked against the PDF.
- The University of Auckland, VUW, Otago and Canterbury produced no relevant inference or agent-systems papers in these searches. Auckland items were CS education (Copilot novices) and a speech edge-cloud speculative decoding paper (Interspeech 2026), neither relevant.

## Q4. Bottom line and priority list

### Takeaway
Little from AU/NZ fits. One item is worth implementing (SpecFit-style expert-union-aware draft length), one is worth a test (hybrid-state cache recovery checklist), and two are evaluation sets. The rest are batch-serving, KV or dense-model techniques already covered or already disproven for pancake (TurboQuant KV, L3 prefetch, reuse-aware admission).

### Cited Findings
- Priority 1: SpecFit, expert-union cost in DraftPolicy. Expected +3-8% decode, unproven on Qwen3.8 — [arXiv 2610.15417](https://arxiv.org/abs/2610.15417)
- Priority 2: hybrid-state restore equivalence test for the conversation cache (off-by-one credited tokens). Correctness only — [arXiv 2609.15030](https://arxiv.org/abs/2609.15030)
- Priority 3: BigCodeBench(-Hard) quality gate for quantization and drafting changes — [arXiv 2406.15877](https://arxiv.org/abs/2406.15877); SWE-EVO for long sessions — [arXiv 2512.18470](https://arxiv.org/abs/2512.18470)
- Optional: check whether SuffixDrafter mixes history and session sources the way the Griffith WebGPU drafter does — [arXiv 2610.39761](https://arxiv.org/abs/2610.39761)

### Inferences
- Gain estimates above are my own extrapolations from the abstracts and pancake's DraftPolicy code (`/home/dan/clone-of-d/src/Strata/src/spec/draft_policy.cpp`), not measured.

### Gaps
- Several 2026 arXiv IDs (26xx) are beyond my training data. They were verified only through the arXiv API and OpenAlex metadata, not by reading full PDFs.
