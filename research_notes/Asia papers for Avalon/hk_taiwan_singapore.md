# AI papers from Hong Kong, Taiwan and Singapore universities (Nov 2022 - Nov 2026) relevant to pancake / Avalon

Scope notes: institutions covered are CUHK, HKUST (incl. Guangzhou), HKU, PolyU, CityU, NTU Taiwan, NTHU, NYCU, Academia Sinica, NUS, NTU Singapore, SMU and A*STAR. "Lead" means the first author or the corresponding/last author is at one of these institutions. Affiliations were checked from arXiv HTML front matter. Entries where the in-scope institution is only a co-author are marked "co-author" and kept short. Papers already in /home/dan/reports/ (MEM1, EAGLE, Lookahead, Medusa, Hydra, Fiddler, fMoE, eMoE, SuffixDecoding, DuoAttention, Quest, etc.) are not repeated. All speedups are the authors' own claims on NVIDIA hardware unless stated; none were measured on RDNA3/ROCm.

Engine context assumed: Qwen3.8-Flash-Next (GDN + attention hybrid MoE, MTP head) on RX 7900 XT 20 GB; experts tiered VRAM / pinned RAM / NVMe; the CPU (AVX2) computes missed experts (about half of decode time); MTP + prompt-lookup drafting with GDN replay; ~126 t/s decode. Already tried with no gain: HIP graphs, unroll flag, Vulkan, TurboQuant KV, L3 prefetch, ClassLRU scoring.

## Q1: Which ideas could speed up batch-1 MoE decode with offloaded experts, or long prefill, on a consumer AMD GPU?

### Takeaway
Two pieces of in-scope work are worth trying. MC-MoE (HKU lead) gives experts different bit-widths by importance, so more of them fit in VRAM. DAOP (NUS lead) predicts the next layer's experts and starts computing them on the CPU early. BFLA (CityU lead) is the only paper found that benchmarks Qwen3.8-Flash-Next itself: training-free sparse prefill gives 1.34-1.44x TTFT at 128-256K context with no accuracy loss, but on H200 with FA3-derived kernels. The Taiwan CPU-GPU paper (NTU/NTHU) and ExpertFlow (A*STAR) mostly confirm what pancake already does.

### Cited Findings

**MC-MoE: "Mixture Compressor for Mixture-of-Experts LLMs Gains More"** (HKU lead: Wei Huang, Xiaojuan Qi; co-authors incl. CUHK's Hongsheng Li and Beihang; ICLR 2025; arXiv 2410.06270)
- Core idea: a training-free "Pre-Loading Mixed-Precision Quantization" that gives each expert its own bit-width based on activation frequency, routing scores and reconstruction error, plus "Online Dynamic Pruning" that keeps only the top expert for unimportant tokens — [arXiv abs](https://arxiv.org/abs/2410.06270)
- Key numbers: average 2.54 bits, 76.6% compression, 3.8% average accuracy loss; dynamic pruning cuts activated parameters by a further 15% with less than 0.6% loss — [arXiv abs](https://arxiv.org/abs/2410.06270)
- Affiliations (HKU first and last author; Beihang and CUHK co-authors) — [arXiv HTML](https://arxiv.org/html/2410.06270)
- Verdict: **medium, worth a test.** Pancake already tiers experts by location. MC-MoE adds a second axis: precision by importance. Storing cold or low-score experts at about 2-3 bits in VRAM, and hot ones at higher bits, raises the VRAM hit rate, which shrinks the CPU-miss half of decode. ODP's "only top-1 expert for unimportant tokens" maps onto pancake's miss path: when the second expert is a miss and its gate weight is small, skip it. That is a quality-vs-speed knob and needs an eval gate. Expected gain: proportional to the miss-rate drop. If misses fall by a third, roughly 10-20% decode. This is an estimate, not a measurement.

**DAOP: Data-Aware Offloading and Predictive Pre-Calculation for Efficient MoE Inference** (NUS lead: Yujie Zhang, Shivam Aggarwal, Tulika Mitra; DATE 2025; arXiv 2501.10375)
- Core idea: assigns experts to the CPU or GPU from each sequence's activation pattern; predicts the next layer's experts one layer ahead and pre-computes them on the CPU with approximate inputs; "graceful degradation" keeps accuracy — [arXiv](https://arxiv.org/pdf/2501.10375)
- Key numbers: up to 8.20x faster than expert caching/prefetching baselines and 1.35x faster than other offloading techniques; Mixtral and Phi-MoE on an A6000 — [summary of the arXiv abstract](https://www.emergentmind.com/papers/2501.10375)
- Code: none found.
- Verdict: **medium.** The useful piece for pancake is starting CPU work for layer L+1's predicted miss experts while the GPU is still on layer L, using layer L's input (or the residual) as the approximate input. That overlaps the CPU-miss half of decode with GPU work instead of serialising it. Risk: wrong predictions waste AVX2 cycles and memory bandwidth on a 5700G that is already the bottleneck, and approximate inputs change outputs, so the result is not lossless. Note that a prefetch of a different kind (L3 prefetch) already showed no gain. This is compute-ahead, not data prefetch.

**Efficient CPU-GPU Collaborative Inference for MoE-based LLMs on Memory-Limited Systems** (NTU Taiwan lead + NTHU: En-Ming Huang, Li-Shang Lin, Chun-Yi Lee; ASP-DAC 2026; arXiv 2512.16473)
- Core idea: a set-associative, per-layer GPU expert cache with LRU eviction (LRU slightly beat FIFO, and both clearly beat random). On a miss, the activation goes to the CPU for expert compute while the weights copy to the GPU asynchronously. Two streams overlap copies and transfers — [arXiv HTML](https://arxiv.org/html/2512.16473)
- Key numbers: 4.8 t/s on Mixtral-8x7B and 10.4 t/s on Phi-3.5-MoE (RTX 4090 + Threadripper 7960X); up to 4.4x faster than Pre-gated MoE; about 1.6x faster than Fiddler — [arXiv HTML](https://arxiv.org/html/2512.16473)
- Code: [github.com/elsa-lab/MoE-CPU-GPU-Collaborative-Inference](https://github.com/elsa-lab/MoE-CPU-GPU-Collaborative-Inference)
- Verdict: **already done.** Pancake already computes misses on the CPU and tiers its cache. The one difference is "copy the missed expert to the GPU at the same time as computing it on the CPU." If pancake does not already fill VRAM on a miss, that is a cheap check. Given that ClassLRU scoring showed no gain, the eviction policy is unlikely to matter.

**ExpertFlow: Efficient MoE Inference via Predictive Expert Caching and Token Scheduling** (A*STAR CFAR lead: Xin He, Ivor Tsang, Ong Yew Soon; HKUST, HKBU, NTU Singapore co-authors incl. Xiaowen Chu; DAC 2026; arXiv 2410.17954, v2 Apr 2026)
- Core idea: a routing-path predictor, a predictive expert cache, and a token scheduler that batches tokens with similar routing paths to reduce expert loads — [arXiv abs](https://arxiv.org/abs/2410.17954); affiliations — [arXiv HTML](https://arxiv.org/html/2410.17954v1)
- Key numbers: up to 93.72% GPU memory saved; up to 10x throughput over offloading baselines; expert-cache hit ratio up to 91.96% (v1) — [arXiv abs](https://arxiv.org/abs/2410.17954)
- Verdict: **low for decode.** Token scheduling needs many tokens per step. At batch 1 it only applies to prefill chunks and to verify batches (MTP plus prompt-lookup drafts). The routing-path predictor is the same family as fMoE/eMoE, which are already in the reports.

**HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference** (co-author only: SJTU lead; CUHK co-authors Jiacheng Liu and Pheng-Ann Heng; arXiv 2411.01433) — affiliations — [arXiv HTML](https://arxiv.org/html/2411.01433v2)
- Core idea, as known from the title and the authors' framing: on a cache miss, load a low-precision copy of the expert instead of the full one. Relevance: for pancake this means keeping a 2-3-bit copy of every expert in VRAM or pinned RAM as a fallback to the CPU path. It pairs with MC-MoE above. Out of scope as lead.

**SpecMoEOff: Accelerating MoE Inference by Hiding Offloading Latency with Speculative Decoding** (co-author only: Nanjing University lead; HKUST(GZ) co-author Weilin Cai; arXiv 2508.21706)
- Uses EAGLE drafts to put more tokens through each offloaded expert; has a CPU chunked-attention verify kernel; up to 2.5x decode throughput over MoE-Lightning-class offloaders (A30 and RTX 4090D; Mixtral) — [arXiv HTML](https://arxiv.org/html/2508.21706v1)
- Verdict: supports pancake's existing drafting choice. The argument is that verify batches amortize an expert load or CPU expert compute over k tokens. It does not need to be repeated, but it is a reason to give drafting priority on steps with many misses (see Q2).

**BFLA: Block-Filtered Long-Context Attention Mechanism** (CityU lead: Chong Wu, corresponding author and project leader; NUS, Rutgers, ByteDance co-authors; arXiv 2605.12193, May 2026)
- Core idea: training-free sparse prefill. A deterministic "Lightning-style" block indexer on post-RoPE Q/K with FP8 block keys (block size 64). One shared block list per 128-row query tile. Budget is 1/3 of context, bounded to 2048-32768 KV blocks. Sink and local blocks always kept; the last 8 query groups use full attention. Changes only full-attention prefill layers and leaves recurrent layers and decoding unchanged — [arXiv HTML](https://arxiv.org/html/2605.12193)
- Key numbers on **Qwen3.8-Flash-Next**: operator-level prefill 1.50-2.66x; end-to-end TTFT 1.435x (LongBench-v2) and 1.342x (InfiniteBench); accuracy unchanged or nearly so: LongBench-v2 57.32 vs 57.32 dense, RULER 94.36 vs 94.65, Terminal Bench 2.1 86.89 vs 86.52. Hardware: H200; Flash-Next in FP8 with TP=2 — [arXiv HTML](https://arxiv.org/html/2605.12193)
- Code: [github.com/Alicewithrabbit/BFLA](https://github.com/Alicewithrabbit/BFLA) (listed on the paper page)
- Verdict: **medium for cold long prefill, low otherwise.** This is the only in-scope paper measured on the user's exact model, and the Terminal Bench result suggests agent quality holds. But the gain on the hybrid is much smaller than on dense models (1.34-1.44x TTFT vs about 4x on Gemma), because only the attention layers are sparsified. The kernel is built on FA3/FlashPrefill V2 (Hopper-only), so a HIP port to RDNA3 is real work. Pancake's conversation cache with boundary checkpoints already removes most repeated prefill in opencode sessions. The gain applies only to first-turn or post-compaction prefill above about 32K tokens, which is BFLA's K_min floor.

**Palu** (NYCU lead: Chi-Chih Chang, Kai-Chiang Wu; Cornell co-authors; arXiv 2407.21118) and **xKV** (NYCU/Cornell; ICML 2026; arXiv 2503.18893)
- Palu: low-rank projection of the KV hidden dimension with on-the-fly reconstruction. xKV: a shared low-rank KV subspace across groups of layers via aligned singular vectors — [Palu abs](https://arxiv.org/abs/2407.21118); [xKV abs](https://arxiv.org/abs/2503.18893)
- Verdict: **low.** In a GDN hybrid only a minority of layers keep KV, so KV memory and bandwidth are a small part of batch-1 decode. TurboQuant KV already showed no gain.

**ChunkKV** (HKUST(GZ) lead: Xiang Liu, Zhenheng Tang, Xiaowen Chu; NeurIPS 2025; arXiv 2502.00299) and **KVFundaBench / "Semantic Integrity Matters"** (same group; ICML 2026; arXiv 2502.01941)
- ChunkKV compresses KV by semantic chunks instead of single tokens — [arXiv abs](https://arxiv.org/abs/2502.00299). KVFundaBench finds retrieval tasks hold up under aggressive KV compression but reasoning tasks degrade sharply because chain-of-thought links break — [arXiv abs](https://arxiv.org/abs/2502.01941)
- Verdict: **use as a warning, do not implement.** For a coding agent with thinking, KV eviction is risky. This matches the TurboQuant null result.

### Inferences
- The largest lever left in pancake is the CPU-miss half of decode. In-scope work offers two ways at it: lower-precision experts in VRAM (MC-MoE, and HOBBIT as co-author work) to cut misses, and compute-ahead on the CPU for predicted misses (DAOP) to hide them. Both are lossy and need an eval gate (for example a pass-rate check on the opencode task set).
- On RDNA3 a 2-3-bit expert format needs a fast HIP dequant GEMV. Without one, the hit-rate gain could be eaten by slower GPU expert kernels.
- For prefill, the conversation cache matters more than sparse attention for this workload. BFLA is the fallback for cold contexts above 64K.

### Gaps
- No in-scope paper reports results on AMD GPUs or ROCm. All numbers are on NVIDIA hardware.
- No speed or memory figures for MC-MoE beyond compression were found on the abstract page, and no code link was verified.
- I found no HK/TW/SG-led paper on NVMe-tier expert streaming specifically.
- Kernel or compiler work (Triton/HIP autotuning) from these institutions did not come up in searches.

## Q2: Speculative decoding work — anything beating MTP + prompt-lookup for code editing with a hybrid recurrent model?

### Takeaway
No in-scope paper beats MTP + prompt-lookup for this setup. The relevant in-scope work either needs a trained drafter (GRIFFIN, SpecBlock, LongSpec), does self-speculation by skipping layers (SWIFT, CAS-Spec), which conflicts with GDN recurrent state, or tunes draft length adaptively (AdaSD). The cheap, transferable ideas are adaptive draft-length control (AdaSD) and LongSpec's split-attention trick for verify kernels. The strongest evidence on code editing (EfficientEdit, model-free PLD on code) is from out-of-scope institutions and backs pancake's current design.

### Cited Findings

**SWIFT: On-the-Fly Self-Speculative Decoding** (PolyU lead: Heming Xia, Wenjie Li; ICLR 2025; arXiv 2410.06916)
- Drafts by skipping layers of the target model, with the skip set picked at inference time (random search plus Bayesian optimization); no extra model or training; 1.3-1.6x speedup; the best skip set varies by task (a set tuned for one task drops to about 1.01x on another) — [ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d74d002a9154b4cc433a234feb27c5f4-Abstract-Conference.html); [arXiv HTML](https://arxiv.org/html/2410.06916v2); code [github.com/hemingkx/SWIFT](https://github.com/hemingkx/swift)
- Verdict: **no.** MTP already gives a better draft at lower cost. Skipping GDN layers also changes the recurrent state and adds replay complexity. Out-of-scope evidence agrees: a self-speculative method on sequential hybrids (Qwen3.5) got acceptance α=0.038 at k=2, versus 0.68 on parallel hybrids — [arXiv 2605.01106](https://arxiv.org/abs/2605.01106) (Spanish authors, out of scope, cited as a supporting data point).

**CAS-Spec: Cascade Adaptive Self-Speculative Decoding** (HKUST co-author with TeleAI / SJTU; lead unclear; NeurIPS 2025; arXiv 2510.26843)
- A cascade of self-speculative draft models built from the target, with dynamic tree cascade (DyTC) routing — [arXiv abs](https://arxiv.org/abs/2510.26843)
- Verdict: **no**, for the same layer-skip and state reasons as SWIFT.

**Spec-Bench survey: "Unlocking Efficiency in LLM Inference: A Comprehensive Survey of Speculative Decoding"** (PolyU/PKU, first author Heming Xia; ACL Findings 2024; arXiv 2401.07851) — [arXiv abs](https://arxiv.org/abs/2401.07851)
- Useful as the standard taxonomy and benchmark. Not an engine change.

**LongSpec: Long-Context Lossless Speculative Decoding** (NTU Singapore lead: Penghui Yang, Bo An; Sea AI Lab and NUS co-authors; ACL 2025 Main; arXiv 2502.17421)
- Draft model with a constant-size KV cache (sliding-window self-attention, window 512) that cross-attends to the target's KV. Anchor-Offset position indices for long-context training. **Hybrid Tree Attention:** the cached prefix runs through FlashAttention, the speculative tokens through a small masked Triton kernel, and the two are merged with log-sum-exp — [arXiv HTML](https://arxiv.org/html/2502.17421)
- Key numbers: up to 3.26x on LongBench tasks (vs vanilla HF); about 2.34x on QwQ-32B math reasoning at 32K output; Hybrid Tree Attention cuts attention latency by about 75% vs the HF implementation — [arXiv HTML](https://arxiv.org/html/2502.17421)
- Verdict: **low-medium, kernel idea only.** The drafter is unnecessary given MTP. Hybrid Tree Attention is worth checking in the verify path. If pancake's attention layers run verify with a full masked attention over the whole context, splitting it into "prefix: plain decode-style attention" plus "draft block: tiny masked attention", merged by LSE, makes verify cost scale with draft size rather than context. That matters at 50-200K agent contexts. Expected gain: only in verify-step attention time at long context; small in total because most layers are GDN.

**GRIFFIN: Effective Token Alignment for Faster Speculative Decoding** (co-author: Fudan lead; NUS's Kim-Chuan Toh and SMU's Pan Zhou co-authors; arXiv 2502.11018)
- An EAGLE-style drafter with token-alignable training (loss masking on misaligned tokens) and a draft model that takes input tokens to correct misalignment — [arXiv abs](https://arxiv.org/abs/2502.11018)
- Verdict: **low.** Applies only if the user trains a custom drafter. The idea of masking misaligned positions could inform fine-tuning the MTP head.

**SpecBlock: Block-Iterative Speculative Decoding with Dynamic Tree Drafting** (HKUST first author Weijie Shi and last author Xiaofang Zhou; MetaX co-authors; arXiv 2605.07243, May 2026)
- Each drafter forward emits K=4 dependent positions; a co-trained rank head sets the branching width per position; a cost-aware bandit decides online whether to update the drafter — [arXiv HTML](https://arxiv.org/html/2605.07243)
- Key numbers: mean speedup at T=0 of 3.21x vs EAGLE-3's 2.92x (Llama-3.1-8B), 2.42x vs 2.24x (Qwen3-8B), 2.37x vs 2.15x (Qwen3-32B); drafting share of iteration time drops from about 31% to 16% — [arXiv HTML](https://arxiv.org/html/2605.07243)
- Verdict: **low.** Needs a trained drafter and dense models. The transferable idea is per-position branching width from a confidence head, which is similar to what DraftPolicy should already do with MTP confidence.

**AdaSD: Adaptive Speculative Decoding** (Academia Sinica + NTU Taiwan: Kuan-Wei Lu, Ding-Yong Hong, Pangfeng Liu, Jan-Jan Wu; arXiv 2512.11280)
- Hyperparameter-free: adjusts draft length and acceptance criteria during inference; up to 1.46x over vanilla speculative decoding with under 1.8% accuracy loss — [arXiv abs](https://arxiv.org/abs/2512.11280); [search snippet of the PDF](https://arxiv.org/pdf/2512.11280)
- Verdict: **low-medium.** The adaptive length part overlaps with DraftPolicy. The relaxed acceptance part is lossy. An out-of-scope practical study warns that relaxed acceptance needs real capability evaluation — [arXiv 2607.08690](https://arxiv.org/abs/2607.08690). Use only the length controller as a cross-check for DraftPolicy.

**SpecBranch** (co-author only: USTC/Zhejiang-led with an NUS affiliation present; ICLR 2026; arXiv 2506.01979)
- Branch-prediction-inspired parallel drafting with rollback awareness to remove draft/target waiting bubbles — [arXiv abs](https://arxiv.org/abs/2506.01979)
- Verdict: **no** for a single GPU with an MTP head. Which author is at NUS was not established.

Out-of-scope but directly relevant evidence (for the writer's context, not Asia-led):
- EfficientEdit (Beihang/Huawei, ASE 2025) uses the original code as the draft for code editing — [arXiv 2506.02780](https://arxiv.org/abs/2506.02780). This is the same principle as pancake's prompt-lookup and SuffixDrafter.
- SpecLA (Nanjing/Alibaba, Jul 2026) treats recurrent-state rollback as the core problem for linear-attention speculative decoding. It compares full-state snapshots with token replay and stores compact factors produced during verification to recover accepted states — [arXiv 2607.16673](https://arxiv.org/abs/2607.16673). This is the closest design reference for pancake's GDN replay.

### Inferences
- For a hybrid GDN model with an MTP head, the in-scope literature gives no drafter that should replace MTP + prompt-lookup. The remaining wins are in verify-cost engineering: LongSpec's prefix/draft attention split, and SpecLA-style "recover accepted GDN state from verify-time factors instead of replaying". SpecLA is out of scope but should be passed to whoever covers mainland work.
- Combined with SpecMoEOff (Q1), there is a case for making DraftPolicy aware of misses: draft longer on steps where many experts miss, because verify amortizes CPU expert compute over more tokens.

### Gaps
- I found no HK/TW/SG-led paper evaluating speculative decoding on GDN/Qwen3.x-Next hybrids specifically.
- No in-scope work on code-editing-specific drafting was found.
- No HK/TW/SG co-authors on the EAGLE or Lookahead papers were found (EAGLE: Peking/MSR/Waterloo; Lookahead: UCSD), so the "EAGLE co-authors" angle came up empty for this region.

## Q3: Agent, memory and reasoning-efficiency work for opencode

### Takeaway
NUS/NTU Singapore/SMU and HKUST/PolyU have produced most of the in-scope agent and efficient-reasoning work. Most of it needs RL or fine-tuning (Thinkless, VeriThinker, LASER, TokenSkip, AutoCompact), so it does not plug into an inference engine. The directly usable lessons are three. Don't constrain the model's format during reasoning (NTU Taiwan). Use AST-aware code search tools (NUS AutoCodeRover/SpecRover). Treat context compaction as a "when and what to keep" decision, not a fixed threshold (NTU/SMU AutoCompact).

### Cited Findings

**Let Me Speak Freely? Format restrictions and LLM performance** (NTU Taiwan co-authors Hung-yi Lee and Yun-Nung Chen; first authors at Appier, so company-led with NTU Taiwan participation; arXiv 2408.02442)
- Finds a significant decline in reasoning when output is constrained to JSON/XML formats compared with free-form answers — [arXiv abs](https://arxiv.org/abs/2408.02442)
- Verdict: **applicable at no cost.** For tool-call recovery, don't apply grammar or JSON constraints during the thinking segment. Constrain only the tool-call span after thinking ends, or keep grammar repair post-hoc. Check that the loop guard and tool-call recovery do not force the format early.

**AutoCodeRover** (NUS lead: Yuntong Zhang, Haifeng Ruan, Zhiyu Fan, Abhik Roychoudhury; ISSTA 2024; arXiv 2404.05427) and **SpecRover** (NUS; ICSE 2025; arXiv 2408.02232)
- AutoCodeRover combines LLMs with program-structure-aware code search (classes and methods via AST) for GitHub-issue repair. SpecRover adds iterative specification inference during code search, aimed at efficient, low-cost workflows — [AutoCodeRover abs](https://arxiv.org/abs/2404.05427); [SpecRover abs](https://arxiv.org/abs/2408.02232)
- Verdict: **medium for opencode tooling, none for the engine.** AST-level search tools (search_class, search_method_in_file) return compact code instead of whole files. That cuts prefill tokens per turn, which matters more for a local 126 t/s engine than for an API.

**AutoCompact: Learning When to Compact Context in Long-Horizon Coding Agents** (NTU Singapore + SMU: Xuan Zhang, Longtao Zheng, Cunxiao Du, Bo An, Xin Dong; arXiv 2610.02163, Oct 2026)
- Trains the coding agent to decide when to compact, what working state to keep, and how to continue. Judge-corrected trajectories are used for SFT, then RL jointly optimizes coding and compaction — [arXiv abs](https://arxiv.org/abs/2610.02163)
- Verdict: **low-medium.** The model training is out of reach. The design lesson applies: compaction should keep the working state (files touched, failing test, plan) rather than fire at a token threshold. Interaction with pancake: every compaction invalidates the conversation cache, so a cold prefill follows (see BFLA). Out-of-scope studies agree compaction is not free. One found policies using about a third of the tokens could take 20-80% longer on Terminal-Bench with Qwen — [arXiv 2609.32961](https://arxiv.org/abs/2609.32961) (Microsoft/UT, out of scope).

**ContextSniper** (co-author: Huawei AntTrail-led with an NTU Singapore affiliation present; arXiv 2607.01916)
- Three-level code and action memory, a hybrid ranker, and an intention-aware gate on long tool output; 51.5% fewer tokens and 36.4% lower logged cost on SWE-bench Lite — [arXiv abs](https://arxiv.org/abs/2607.01916)
- Verdict: the idea of filtering long tool output (test logs, grep output) before it enters context is directly usable in opencode. It is company-led, so treat it as supporting evidence only.

**Thinkless: LLM Learns When to Think** (NUS lead: Gongfan Fang, Xinyin Ma, Xinchao Wang; NeurIPS 2025; arXiv 2505.13379)
- RL with two control tokens (short vs long reasoning) and Decoupled GRPO; cuts long-chain reasoning use by about 50-90% on MATH-500/GSM8K/Minerva Algebra — [arXiv](https://arxiv.org/pdf/2505.13379); code [github.com/VainF/Thinkless](https://github.com/VainF/Thinkless)
- Verdict: **concept only.** Pancake's thinking budgets are the inference-time equivalent. Thinkless supports routing "easy" turns (tool-result acknowledgements, small edits) to no-think or a very small budget.

**VeriThinker** (NUS lead: Zigeng Chen, Xinyin Ma, Gongfan Fang, Xinchao Wang; arXiv 2505.17941)
- Compresses CoT by fine-tuning only on an auxiliary verification task; the model learns when self-reflection is unnecessary — [arXiv abs](https://arxiv.org/abs/2505.17941)
- Verdict: **requires fine-tuning; low.**

**LASER: Learn to Reason Efficiently with Adaptive Length-based Reward Shaping** (HKUST lead: Junxian He's group; arXiv 2505.15612)
- Unifies efficient-reasoning methods as length-based reward shaping; proposes a step-function reward controlled by a target length, with adaptive variants — [arXiv abs](https://arxiv.org/abs/2505.15612)
- Verdict: **requires RL; low.** Gives a framework for choosing thinking budgets by difficulty.

**TokenSkip: Controllable Chain-of-Thought Compression** (PolyU lead: Heming Xia, Wenjie Li; EMNLP 2025; arXiv 2502.12067)
- Prunes low-importance CoT tokens at a set ratio and fine-tunes with the ratio as a control input; on GSM8K with Qwen2.5-14B-Instruct, reasoning tokens fell from 313 to 181 (about 40%) with under 0.4% loss; up to 1.4x speedup — [arXiv HTML](https://arxiv.org/html/2502.12067v3); [ACL Anthology](https://preview.aclanthology.org/setup/2025.emnlp-main.165); code [github.com/hemingkx/TokenSkip](https://github.com/hemingkx/TokenSkip)
- Verdict: **requires LoRA fine-tuning; low.**

### Inferences
- The inference-time changes available today are: no format constraints during thinking; compact AST-aware search tools; filtering tool output before it enters context; and compaction that keeps state. These matter more locally than for API agents because every prompt token costs local prefill time and every compaction loses the conversation cache.
- The efficient-reasoning papers (Thinkless, VeriThinker, LASER, TokenSkip) all show that 30-90% of reasoning tokens can be removed with RL or fine-tuning. Without fine-tuning Qwen3.8-Flash-Next, the closest substitute is difficulty-dependent thinking budgets in pancake.

### Gaps
- AutoCodeRover/SpecRover SWE-bench numbers and per-issue costs were not fetched from primary sources, so none are given.
- LASER and VeriThinker quantitative results were not checked (abstracts were truncated in my fetches).
- I found no in-scope work on tool-call recovery or loop detection in agents, and no HK/TW/SG evaluation of local, single-GPU coding agents.
- The OpenHands paper (ICLR 2025) shows no HK/TW/SG affiliation in the author-list check I ran ([arXiv abs](https://arxiv.org/abs/2407.16741)). The OpenHands-co-author angle did not yield anything in scope.
