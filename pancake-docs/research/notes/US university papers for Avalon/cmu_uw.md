# CMU and UW AI/systems papers (Nov 2022 - Nov 2026) for pancake / Avalon / the opencode agent

Scope notes for the report writer:
- "LEAD" means the first or corresponding author's group is at CMU or UW. "CO" means CMU/UW people are co-authors only.
- Already covered elsewhere and skipped here: Quest (MIT Han Lab lead, UW's Kasikci co-author; see the MIT notes), StreamingLLM (MIT lead, Beidi Chen co-author), Hydra (MIT), Gated DeltaNet (MIT/NVIDIA), Hydragen (Stanford).
- Excluded as not CMU/UW-led: MoE-Lightning (UC Berkeley lead, arXiv 2411.11217), SWE-bench / SWE-agent (Princeton), MoE-Gen (Edinburgh).
- Numbers are the authors' own claims unless marked otherwise. Items marked "(abstract, not re-fetched)" come from arXiv abstracts I know well but did not re-open this session.
- Setup facts behind the verdicts:
  - Decode is bound by reads of missed experts (pinned RAM or NVMe) and by CPU compute of those experts. It is not bound by GPU attention.
  - Single user, batch 1. Only a minority of layers are full attention; the rest are Gated DeltaNet (recurrent).
  - Already tested here with no gain: HIP graphs, ClassLRU-style scoring, L3 prefetch, TurboQuant KV, Vulkan.

## Q1. CPU+GPU hybrid MoE execution and expert offload/prefetch: what beats pancake's tiering plus CPU-computes-misses design?

### Takeaway
pancake's design already contains the UW Fiddler idea: compute missed experts on the CPU and ship activations, not weights. I found no CMU- or UW-led successor that beats it for batch-1 decode. Two pieces are still worth porting:
- (a) Fiddler's per-expert choice between CPU compute and weight transfer, keyed on how many tokens hit that expert. This applies to prefill and to speculative verification batches.
- (b) CMU's cache-policy work (S3-FIFO/SIEVE admission, and Baleen's method of bounding the best possible policy offline) for the VRAM and pinned-RAM tiers.

### Cited Findings
- **Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models.** UW LEAD (Keisuke Kamahori, Tian Tang, Yile Gu, Kan Zhu, Baris Kasikci; SyFI lab). arXiv 2402.07033 (Feb 2024), ICLR 2025.
  - Core idea: run expert layers on the CPU when their weights are not on the GPU, copy only activations, and keep frequently used experts on the GPU. — [ICLR 2025 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/8cd1ce03ea58b3d7dfd809e4d42f08ea-Paper-Conference.pdf); [SyFI lab page](https://syfi.cs.washington.edu/publications/fiddler)
  - It picks the execution plan per expert from the number of input tokens routed to that expert: CPU compute when few tokens arrive, weight transfer plus GPU compute when many arrive. — [arXiv 2402.07033](https://arxiv.org/abs/2402.07033?context=cs)
  - Numbers: unquantized Mixtral-8x7B (>90 GB) at >3 tok/s on one 24 GB GPU. v3 abstract: 1.26x single-batch, 1.30x long prefill, 11.57x beam search over baselines. The lab page claims 8.2-10.1x over standard offloading. The two figures differ because they use different baselines. — [arXiv](https://arxiv.org/pdf/2402.07033.pdf); [SyFI](https://syfi.cs.washington.edu/publications/fiddler)
  - Code: [github.com/efeslab/fiddler](https://arxiv.org/pdf/2402.07033.pdf).
- A search for a 2025-2026 UW follow-up on MoE offloading from the Kasikci group found none. The results only returned Fiddler and non-UW work (MoE-Gen, adaptive expert split, CPU-GPU collaborative inference 2512.16473). — [search results incl. MoE-Gen](https://arxiv.org/html/2503.09716v1); [2512.16473](https://arxiv.org/html/2512.16473)
- **S3-FIFO: "FIFO queues are all you need for cache eviction".** CMU LEAD (Juncheng Yang, CMU PhD at the time; now Harvard). SOSP'23.
  - A small probationary FIFO filters out most objects before they reach the main cache, which gives "guaranteed demotion speed and high demotion precision".
  - It had lower miss ratios than state-of-the-art policies on 6594 traces from 14 datasets.
  - — [Alluxio talk summary](https://www.alluxio.io/videos/data-infra-meetup-fifo-queues-are-all-you-need-for-cache-eviction); [Yang homepage](https://juncheng.seas.harvard.edu)
- **SIEVE is Simpler than LRU.** Emory LEAD (Yazhuo Zhang), CMU CO (Juncheng Yang). NSDI'24.
  - FIFO plus a "visited" bit and a moving hand; the change is under 20 LOC in production libraries.
  - Up to 63.2% lower miss ratio than ARC on 1559 traces.
  - — [USENIX NSDI'24](https://www.usenix.org/conference/nsdi24/presentation/zhang-yazhuo); [paper](https://junchengyang.com/publication/nsdi24-SIEVE.pdf)
- **Baleen: ML Admission & Prefetching for Flash Caches.** CMU LEAD (Daniel Wong, Nathan Beckmann, Gregory Ganger; with Meta/Microsoft). FAST'24.
  - It trains admission and prefetch policies with an "episodes" residency model, meaning admission-to-eviction spans.
  - Results: 12% lower peak disk-head time than prior policies under a fixed flash write budget, and an estimated 17% lower total cost of ownership (TCO).
  - Code, traces and a simulator artifact are public.
  - — [USENIX FAST'24](https://www.usenix.org/conference/fast24/presentation/wong); [Chameleon artifact blog](https://chameleoncloud.org/blog/2024/03/26/baleen-ml-admission-prefetching-for-flash-caches)
- Context (affiliations not verified as CMU/UW, cited only for the cost model): MoE speculative-decoding papers from 2025-2026 argue that verification cost under offload is driven by the *union of distinct experts* the draft tokens touch, together with which experts are already cached.
  - Cascade (2025): utility-based throttling of speculation.
  - MoE-Spec (2026): caps experts per layer during verification.
  - AcceptMoE (2026): chooses verifier experts by cache residency; reports 2.06x on one RTX 5090 with offload.
  - — [MoE-Spec](https://www.alphaxiv.org/abs/2602.16052.md); [AcceptMoE](https://www.alphaxiv.org/abs/2608.02989); [Cascade 2506.20675](https://www.alphaxiv.org/abs/2506.20675.md)

### Inferences
- **Fiddler: decode gain ≈ 0; prefill/verify gain possible.**
  - At batch 1 decode each expert sees 1 token, so Fiddler reduces to "compute misses on the CPU", which pancake already does.
  - The transferable piece is the token-count threshold. In a verification pass of k = 3-6 draft tokens, or in prefill chunks, an expert that receives ≥ N tokens may be cheaper to DMA over PCIe 4.0 x16 (~25 GB/s) and run on the GPU than to compute on 8 Zen3 cores. Our RAM bandwidth is 28-42 GB/s, which CPU compute must also stream.
  - Action: measure cost_cpu(expert, t tokens) against cost_dma + cost_gpu for t = 1..8 and set a per-tier threshold.
  - Expected gain: modest. It is largest on 131-262k prefill. It is small on decode unless verification batches are large.
- **Expert cache policy: try S3-FIFO-style admission once; it differs from what failed.**
  - ClassLRU-style scoring gave no gain. S3-FIFO and SIEVE are a different mechanism: an admission filter in which a VRAM slot is earned only by a second hit while the expert sits in the probationary queue. This targets one-off experts polluting VRAM.
  - Baleen's method is the cheaper first step. Log expert-access traces from real opencode sessions, then compute the Belady/OPT miss ratio offline. If OPT is only a few points better than the current policy, stop investing in eviction policies.
  - Expected gain: a few percent of hit rate at most (inference, unmeasured).
- **Speculation must be priced in expert-union cost.** On pancake, every extra draft token can pull new missed experts through the CPU path. Draft length or tree width should therefore be chosen against measured expert-union cost, not token count. See Q2 (Sequoia).

### Gaps
- No CMU/UW paper evaluates expert offload on AMD consumer GPUs, on DDR4-class bandwidth, or on hybrid recurrent MoE models.
- I did not find a UW successor to Fiddler from 2025-2026. A 2026 SyFI paper may exist but was not surfaced by search.
- The affiliations of Cascade, MoE-Spec, EcoSpec, DraftExpert and AcceptMoE were not checked.

## Q2. Speculative decoding for long contexts / offloaded models (SpecInfer, Sequoia, TriForce, MagicDec, SuffixDecoding): gains on one GPU with offloaded experts and a hybrid recurrent model?

### Takeaway
SuffixDecoding (CMU lead, NeurIPS 2025 spotlight) is the most directly useful paper. It is model-free and built for agentic and SWE workloads, and it can upgrade pancake's prompt-lookup drafter. Sequoia's hardware-aware tree-size optimizer is the second most useful. The long-context methods (TriForce, MagicDec, and their KV-sparsity relatives) attack a KV/attention bottleneck that our hybrid model largely does not have.

### Cited Findings
- **SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications.** CMU LEAD (Gabriele Oliaro, Zhihao Jia) with Snowflake (Daniel Campos, Aurick Qiao). arXiv 2411.04975; NeurIPS 2025 Spotlight.
  - Core idea: suffix trees cache token sequences from prompts and *earlier outputs*. Speculation length adapts to the acceptance likelihood.
  - Numbers: up to 3.9x on SWE-Bench and Text-to-SQL agentic workloads, about 2.2x faster than EAGLE-2/3 and 1.6x faster than Token Recycling. The earlier v2 reported smaller figures. Snowflake's blog says "5.3x"; I could not find the source of that figure.
  - Code: snowflakedb/ArcticInference, integrated in vLLM.
  - — [NeurIPS poster](https://neurips.cc/virtual/2025/poster/115453); [arXiv v3](https://arxiv.org/pdf/2411.04975v3); [Snowflake blog](https://www.snowflake.com/en/engineering-blog/suffixdecoding-arctic-inference-vllm/)
- **Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding.** CMU LEAD (Zhuoming Chen, Avner May, ..., Beidi Chen). arXiv 2402.12374 (Feb 2024).
  - Core idea: dynamic programming to choose the optimal token-tree shape, a sampling/verification scheme that is robust across temperatures, and a hardware-aware optimizer for tree size and depth.
  - Numbers:
    - Llama2-70B offloaded on an L40: 0.56 s/token, against 5.6 s/token for the authors' optimized offloading baseline (9.96x) and 5.5 s/token for DeepSpeed-Zero-Inference.
    - On an A100: up to 4.04x (7B), 3.73x (13B), 2.27x (33B).
  - Code: Infini-AI-Lab/Sequoia.
  - — [arXiv 2402.12374](https://arxiv.org/abs/2402.12374v2); [PDF](https://arxiv.org/pdf/2402.12374.pdf)
- **TriForce: Lossless Acceleration of Long Sequence Generation with Hierarchical Speculative Decoding.** CMU LEAD (Hanshi Sun, Zhuoming Chen, ..., Beidi Chen; with Meta). arXiv 2404.11912; COLM 2024.
  - Core idea: a hierarchy of drafters. A tiny model with a StreamingLLM cache drafts for the target model running on a retrieval-selected partial KV cache, which in turn drafts for the full model.
  - Numbers:
    - 2.31x for Llama2-7B-128K on an A100.
    - Offloaded on 2x RTX 4090: 7.78x and 0.108 s/token.
    - Llama2-13B at 128K: 0.226 s/token (7.94x).
    - 4.86x over DeepSpeed-Zero-Inference on one RTX 4090.
  - — [arXiv 2404.11912](https://arxiv.org/abs/2404.11912); [HTML v3](https://arxiv.org/html/2404.11912v3)
- **MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding.** CMU LEAD (Beidi Chen's lab; with Moffett AI and Together AI). arXiv 2408.11049; ICLR 2025.
  - Core idea: at long context and large batch the KV cache dominates, so a draft with a sparse StreamingLLM KV cache gives speedups *even at batch 32-256*.
  - Numbers: up to 2.51x for LLaMA-3.1-8B (final version); v1 reported 1.84x.
  - Code: Infini-AI-Lab.
  - — [arXiv 2408.11049 v3](https://arxiv.org/html/2408.11049v3); [ICLR 2025](https://iclr.cc/virtual/2025/poster/30513)
- **SpecInfer: Accelerating LLM Serving with Tree-based Speculative Inference and Verification.** CMU LEAD (Xupeng Miao, Gabriele Oliaro, ..., Zhihao Jia). arXiv 2305.09781; ASPLOS 2024.
  - Core idea: multiple small speculative models build a token tree, which is verified in one pass with a tree-attention mask.
  - Numbers: abstract reports 1.5-2.8x for distributed inference and 2.6-3.5x for offloading-based inference (abstract, not re-fetched).
  - Code: in FlexFlow Serve (flexflow/FlexFlow). — [arXiv 2305.09781](https://arxiv.org/abs/2305.09781)
- **Related CMU work on long-context KV sparsity. All of these are approximate and attack attention/KV cost.**
  - **MagicPIG** (CMU LEAD, Beidi Chen). arXiv 2410.16179; ICLR 2025. LSH sampling of attention, computed on the CPU. The abstract reports 54 ms decode for Llama-3.1-8B at 96k on one RTX 4090 (abstract, not re-fetched). — [arXiv 2410.16179](https://arxiv.org/abs/2410.16179)
  - **TidalDecode** (CMU LEAD, Zhihao Jia). arXiv 2410.05076; ICLR 2025. Position-persistent sparse attention: token selection is done in a few layers and reused by the others (abstract, not re-fetched). — [arXiv 2410.05076](https://arxiv.org/abs/2410.05076)
  - **ShadowKV** (ByteDance LEAD, CMU CO Beidi Chen). arXiv 2410.21465. Low-rank pre-RoPE keys on the GPU, values offloaded. — [arXiv 2410.21465](https://arxiv.org/abs/2410.21465)
- **Kinetics: Rethinking Test-Time Scaling Laws.** CMU LEAD (Ranajoy Sadhukhan, Zhuoming Chen, Haizhong Zheng, Yang Zhou, Emma Strubell, Beidi Chen). arXiv 2506.05333; NeurIPS 2025.
  - Argument: compute-optimal test-time scaling laws ignore memory access, and attention, not parameter count, dominates the cost of long generations.
  - Block top-k sparse attention lets the budget go to longer generations or more trials.
  - Numbers: up to 3x lower resource use at equal accuracy (v3). The AIME gain figures differ between versions.
  - Code: Infini-AI-Lab/Kinetics.
  - — [NeurIPS poster](https://neurips.cc/virtual/2025/poster/115931); [arXiv v3](https://arxiv.org/pdf/2506.05333v3)

### Inferences
- **SuffixDecoding: HIGH applicability, expected meaningful gain on agent turns.**
  - pancake's prompt-lookup drafter matches n-grams against the prompt. SuffixDecoding extends this in two ways: (1) the tree is indexed over *all prior outputs and prompts*, which in a single-user agent means every previous turn, file read and tool output in the parked conversation cache; (2) speculation length adapts to frequency-based acceptance scores.
  - Coding-agent outputs repeat heavily: re-emitting edited code, diffs and tool-call JSON. This is exactly where the paper reports its largest numbers (SWE-Bench).
  - Integration: keep MTP as the drafter for novel text, and use the suffix tree when its match score is high. In effect this is a better prompt-lookup with cross-turn memory.
  - The suffix tree is CPU-side and cheap, but it competes for CPU cores with expert compute. Keep its update incremental.
  - Expected gain: a guess of +10-30% decode t/s on edit-heavy turns, ~0 on novel reasoning. It must be measured against current prompt-lookup acceptance.
- **Sequoia: MEDIUM.**
  - Its value for us is the *hardware-aware optimizer*. Given (a) the measured acceptance vector for MTP and n-gram drafts and (b) the measured verify cost c(t) for t tokens, it picks the best draft length and tree shape.
  - On pancake c(t) is not flat. Extra tokens add new missed experts computed on the CPU, which is the expert-union effect from Q1. Fit c(t) from real traces rather than assuming GPU-flat cost.
  - Expected gain: a guess of 5-15% if current draft lengths are hand-tuned.
- **Trees vs chains on a Gated DeltaNet hybrid.** SpecInfer/Sequoia token trees need branching recurrent state in the DeltaNet layers: one state copy per branch, or a recompute after acceptance. For a chain, only one rollback checkpoint is needed. This is an inference from how recurrent layers work; I found no CMU/UW paper on tree speculation for hybrid SSM models. Prefer chains, or narrow trees of width ≤ 2, unless profiling shows the state copies are cheap.
- **TriForce / MagicDec / MagicPIG / TidalDecode / ShadowKV: LOW.**
  - Their premise is that KV reads or attention dominate decode at long context. On our hybrid model only the minority full-attention layers carry KV, and decode is bound by expert reads.
  - MagicDec targets large batches, while we serve a single user.
  - The hierarchy idea in TriForce (draft the drafter) is already partly present: n-gram plus MTP.
  - They become worth revisiting only if a profile at 200k+ shows attention-layer time above ~15% of token time. This matches the threshold in the MIT notes.
- **Kinetics: an insight for thinking budgets, not a kernel.** On a pure transformer, long reasoning traces become attention-bound. On our hybrid model the per-token cost of long thinking grows more slowly, which favors longer thinking budgets over best-of-n in the opencode agent (inference).

### Gaps
- No CMU/UW speculative-decoding paper evaluates MoE expert offload combined with a recurrent/hybrid target model.
- SuffixDecoding's acceptance on *our* opencode traces is unknown, as is its overlap with the existing prompt-lookup drafter.

## Q3. Structured generation (XGrammar) and kernel libraries (FlashInfer, MLC/TVM, Mirage): anything usable on RDNA3/HIP?

### Takeaway
XGrammar and XGrammar-2 are hardware-agnostic because token-mask generation runs on the CPU. XGrammar-2's dynamic "TagDispatch" design fits opencode tool calling directly, so it is the one item here worth adopting or copying. FlashInfer's ROCm port targets only Instinct (CDNA3/4), not gfx1100. Mirage and MPK generate CUDA code only. MLC-LLM/TVM supports ROCm but has no expert offload.

### Cited Findings
- **XGrammar: Flexible and Efficient Structured Generation Engine for LLMs.** CMU LEAD (Yixin Dong, ..., Tianqi Chen; with NVIDIA, SJTU). arXiv 2411.15100 (Nov 2024).
  - Core idea: a byte-level pushdown automaton for CFGs. Context-independent tokens are precomputed into an adaptive token-mask cache, and grammar work overlaps with GPU execution.
  - Benchmarks ran on H100. The abstract claims up to 100x over prior engines (abstract, not re-fetched).
  - — [arXiv 2411.15100](https://arxiv.org/abs/2411.15100); [MLC blog](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar)
  - The PyPI page lists AMD GPUs and CPU among supported hardware. It is the default structured-generation backend in vLLM, SGLang, TensorRT-LLM and MLC-LLM. — [PyPI xgrammar](https://pypi.org/project/xgrammar)
  - vLLM v0.14.0 on ROCm added upstream XGrammar integration. — [AMD ROCm blog](https://rocm.blogs.amd.com/software-tools-optimization/vllm-omni/README.html)
- **XGrammar-2: Dynamic and Efficient Structured Generation Engine for Agentic LLMs.** CMU LEAD (MLC/Tianqi Chen group). arXiv 2601.04426; ACM CAIS '26.
  - Core ideas:
    - TagDispatch: free text until a trigger tag, then dispatch to that tool's grammar, with tool sets that can change within a request.
    - JIT grammar compilation, cross-grammar caching, Earley-based mask generation, and repetition compression.
  - Numbers: more than 6x faster compilation and near-zero end-to-end overhead.
  - It supports major tool-call formats and OpenAI Harmony, and adds a JSON "Structural Tag" DSL. Released May 2026.
  - — [arXiv 2601.04426](https://arxiv.org/html/2601.04426v2); [MLC blog May 2026](https://blog.mlc.ai/2026/05/04/xgrammar-2-fast-customizable-structured-generation)
- **FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving.** UW LEAD (Zihao Ye, Luis Ceze, Arvind Krishnamurthy), CMU CO (Ruihang Lai, Tianqi Chen). arXiv 2501.01005; MLSys 2025 Best Paper.
  - Core ideas: block-sparse and composable KV formats, JIT-compiled attention variants, and a load-balanced scheduler that is compatible with CUDA graphs.
  - — [arXiv 2501.01005](https://arxiv.org/abs/2501.01005); [UW news](https://news.cs.washington.edu/2025/07/01/allen-school-researchers-receive-best-paper-award-for-speeding-up-llm-performance-with-flashinfer)
  - The ROCm port (AMD-maintained, ROCm/flashinfer) targets MI300X, MI325X and MI355X (CDNA3/CDNA4). Prefill kernels arrived via AITER in v0.5.3, and some features are still missing. — [ROCm docs](https://rocm.docs.amd.com/projects/flashinfer/en/latest/index.html); [AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/flashinfer-release2/README.html)
- **Mirage Persistent Kernel (MPK): A Compiler and Runtime for Mega-Kernelizing Tensor Programs.** CMU LEAD (Xinhao Cheng, Zhihao Zhang, ..., Tianqi Chen, Zhihao Jia). arXiv 2512.22219; OSDI '26.
  - Core idea: compile the whole model into one persistent kernel made of SM-level task graphs, which removes launch overhead and overlaps compute with communication.
  - It generates CUDA. The demo is Qwen3-8B. — [arXiv 2512.22219](https://arxiv.org/abs/2512.22219v1); [OSDI'26](https://www.usenix.org/conference/osdi26/presentation/cheng)
- Other CMU-led items in the TVM/MLC lineage (abstracts, not re-fetched):
  - **Mirage superoptimizer:** multi-level search over GPU kernels, OSDI'25 ([arXiv 2405.05751](https://arxiv.org/abs/2405.05751)).
  - **Relax:** composable abstractions for dynamic-shape ML in TVM ([arXiv 2311.02103](https://arxiv.org/abs/2311.02103)).
  - **WebLLM:** in-browser engine on WebGPU, claimed to retain up to 80% of native performance ([arXiv 2412.15803](https://arxiv.org/abs/2412.15803)).
- **UW serving-systems papers aimed at multi-request throughput:**
  - **NanoFlow** (UW LEAD, Kan Zhu; OSDI'25): nano-batching and intra-device overlap of operations, up to 1.91x throughput on 8xA100. — [USENIX OSDI'25](https://www.usenix.org/conference/osdi25/presentation/zhu-kan)
  - **Punica** (UW LEAD; multi-tenant LoRA serving, SGMV kernel; [arXiv 2310.18547](https://arxiv.org/abs/2310.18547)).
  - **Atom** (UW LEAD, Kasikci; W4A4 serving with mixed precision and outliers; [arXiv 2310.19102](https://arxiv.org/abs/2310.19102)). Both abstracts were not re-fetched.

### Inferences
- **XGrammar-2: HIGH relevance to the "grammar inside speculative decoding" and tool-call recovery work.**
  - Mask computation runs on the CPU, so the GPU architecture does not matter. On our box the risk is CPU contention with expert compute.
  - XGrammar's design answers that risk. Its adaptive mask cache leaves only "context-dependent" tokens to be checked at runtime, and its rollback/jump-forward APIs fit speculative verification.
  - TagDispatch matches opencode's pattern: free reasoning text, then a `<tool_call>` that must match one of N JSON schemas.
  - Options: (a) link libxgrammar (C++ core, Apache-2.0) into pancake behind the existing grammar interface; (b) copy only the TagDispatch and Structural-Tag semantics.
  - Expected gain: better tool-call validity, and lower per-token CPU cost of grammar masks if pancake's current matcher is measurable in profiles. No direct t/s gain unless grammar shows up in the decode timeline.
  - Avalon (llama.cpp fork) uses GBNF. llama.cpp's grammar sampler is known to be slow on large schemas (general knowledge, not sourced here). An XGrammar backend would be the larger relative win there.
- **FlashInfer: SKIP for kernels.** There is no RDNA3 target. Its batch-oriented load balancing and cascade/shared-prefix tricks do not help a single user. The JIT-variant idea is not needed for one model.
- **MPK / Mirage: SKIP.** It is CUDA-only. HIP graphs already gave no gain here, which suggests launch overhead is not our bottleneck, so a megakernel would have little to remove.
- **MLC-LLM / TVM: LOW.** It runs on ROCm and Vulkan for dense models but has no expert tiering or CPU expert compute. At most it is a reference for TVM-autotuned gfx1100 GEMV schedules.
- **NanoFlow / Punica / Atom: SKIP.** These are multi-request throughput designs. Atom's W4A4 needs integer tensor-core GEMM at batch, while our decode is memory-bound GEMV.

### Gaps
- I did not check whether XGrammar ships a build path without CUDA/Torch for embedding in C++. The core is C++, but the build flags were not verified.
- No benchmark of XGrammar inside speculative decoding on AMD or CPU-contended systems exists.

## Q4. Quantization, MoE and architecture papers (Dettmers / UW, Gu / CMU, Ai2)

### Takeaway
Dettmers' k-bit scaling result (4-bit is optimal per bit, and the trend reverses at 3 bits) says our 2-3-bit experts sit below the accuracy-per-bit optimum. That is acceptable only because RAM bandwidth, not VRAM, is the constraint. It argues for *mixed* precision: shared, attention, DeltaNet and MTP weights at ≥4 bits with small blocks and float data types, and the bulk experts at 2-3 bits. Mamba/Mamba-2 are background for the DeltaNet layers.

### Cited Findings
- **The case for 4-bit precision: k-bit Inference Scaling Laws.** UW LEAD (Tim Dettmers, Luke Zettlemoyer). arXiv 2212.09720; ICML 2023.
  - Experiments: over 35,000 zero-shot runs at 3-8 bits on models from 19M to 66B parameters.
  - Findings: 4-bit is generally optimal for total model bits. At 4 bits, small block sizes and float data types improve scaling, and the trend reverses at 3 bits.
  - Recommendation: always use 4-bit with a small block size and a float data type.
  - — [PMLR](https://proceedings.mlr.press/v202/dettmers23a.html); [arXiv](https://arxiv.org/abs/2212.09720v2)
- **QLoRA: Efficient Finetuning of Quantized LLMs.** UW LEAD (Dettmers, Pagnoni, Holtzman, Zettlemoyer). arXiv 2305.14314; NeurIPS 2023. NF4 data type, double quantization and paged optimizers. Fine-tunes a 65B model on one 48 GB GPU (abstract, not re-fetched). — [arXiv 2305.14314](https://arxiv.org/abs/2305.14314)
- **SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression.** UW LEAD (Dettmers first author; with Yandex, IST Austria, ETH). arXiv 2306.03078. Keeps outlier weights in higher precision as a sparse set, the rest at 3-4 bits (abstract, not re-fetched). — [arXiv 2306.03078](https://arxiv.org/abs/2306.03078)
- **Mamba: Linear-Time Sequence Modeling with Selective State Spaces.** CMU and Princeton co-LEAD (Albert Gu, Tri Dao). arXiv 2312.00752; COLM 2024. **Mamba-2 / SSD** ("Transformers are SSMs"; Dao, Gu): arXiv 2405.21060, ICML 2024. Introduces the chunked state-space-duality algorithm (abstracts, not re-fetched). — [Mamba](https://arxiv.org/abs/2312.00752); [Mamba-2](https://arxiv.org/abs/2405.21060)
- **OLMoE: Open Mixture-of-Experts Language Models.** Ai2/UW LEAD. arXiv 2409.02060. A fully open MoE whose routing analysis reports strong expert specialization (abstract, not re-fetched). — [arXiv 2409.02060](https://arxiv.org/abs/2409.02060)

### Inferences
- **k-bit scaling.** The paper studies dense models and memory size. For us the binding cost is *bytes streamed per missed expert*.
  - Experts that live in VRAM could carry more bits at no decode-bandwidth cost. Experts read from RAM should stay at 2-3 bits.
  - Suggestion: a tier-aware quantization in which hot VRAM-resident experts use a 4-bit variant and cold experts use 2-3 bits. This costs a second copy of the hot experts on disk.
  - This is a quality gain, not a speed gain, and is unmeasured.
- **QLoRA.** It is not practical on the full 125B model with 20 GB of VRAM. The one plausible use is fine-tuning *only the MTP head* on hidden states captured from our own opencode traces, to raise draft acceptance. This is speculative; no paper tests it.
- **SpQR.** Its outlier idea is already approximated by importance-matrix GGUF quants. Not worth porting.
- **Mamba/SSD.** The chunked SSD algorithm is the ancestor of the chunked Gated DeltaNet prefill kernels, which the MIT notes already discuss. Nothing new for decode.

### Gaps
- Mamba-3 (Gu group, reported as an ICLR 2026 submission) was not verified this session.
- I found no CMU/UW paper quantifying 2-bit vs 3-bit MoE expert quality on coding agents.

## Q5. Agent evaluation and reliability work usable for our opencode harness

### Takeaway
CMU leads the realistic agent benchmarks (WebArena, TheAgentCompany) and co-leads OpenHands. For a single-user local harness, the most reusable items are:
- (a) TheAgentCompany's checkpoint-based partial-credit grading, packaged for the Harbor framework, as a local regression suite.
- (b) Agent Workflow Memory, which induces reusable workflows from past successful trajectories as cheap prompt memory.

### Cited Findings
- **WebArena: A Realistic Web Environment for Building Autonomous Agents.** CMU LEAD (Shuyan Zhou, Frank F. Xu, ..., Graham Neubig). arXiv 2307.13854; ICLR 2024. Self-hosted web apps with functional correctness checks. GPT-4 agent success was 14.41% against 78.24% for humans (abstract, not re-fetched). — [arXiv 2307.13854](https://arxiv.org/abs/2307.13854)
- **TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks.** CMU LEAD (Frank F. Xu, Neubig et al.). arXiv 2412.14161; latest v3 is 10 Sep 2025.
  - A simulated software company covering software development, data science, HR, project management, finance and admin.
  - The Harbor adapter packages 174 tasks over GitLab, Plane, OwnCloud and RocketChat, and runs OpenHands as the agent.
  - — [arXiv 2412.14161](https://arxiv.org/abs/2412.14161v2); [Harbor](https://hub.harborframework.com/datasets/theagentcompany/theagentcompany/latest); [Inspect Evals](https://ukgovernmentbeis.github.io/inspect_evals/evals/theagentcompany/index.html)
- **OpenHands: An Open Platform for AI Software Developers as Generalist Agents.** UIUC/CMU co-LEAD (Xingyao Wang first author; Graham Neubig CMU senior). arXiv 2407.16741; ICLR 2025. Event-stream architecture, CodeAct agent, sandboxed runtime, evaluation harness (abstract, not re-fetched). The OpenHands-Versa generalist agent was evaluated on TheAgentCompany. — [arXiv 2407.16741](https://arxiv.org/abs/2407.16741); [OpenHands blog](https://www.openhands.dev/blog/building-a-provably-versatile-agent)
- **Agent Workflow Memory (AWM).** CMU LEAD (Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, Graham Neubig). arXiv 2409.07429. Induces reusable workflows from past trajectories and adds them to agent memory. Reported relative success gains of 24.6% on Mind2Web and 51.1% on WebArena (abstract, not re-fetched). — [arXiv 2409.07429](https://arxiv.org/abs/2409.07429)

### Inferences
- **Harness regression suite: MEDIUM.**
  - The full TheAgentCompany suite is heavy: several self-hosted services and long runs at our ~126 t/s.
  - A cheaper path for an opencode agent is to borrow the *grader design* rather than the environment: checkpoint-based partial credit with deterministic checkers per subtask.
  - Apply it to a small set of our own repo tasks. Use it to A/B engine changes such as grammar-in-spec, thinking budgets, loop guard and tool-call recovery on *task outcome*, not only t/s.
- **AWM: MEDIUM, cheap.** Implement it as an opencode skill or memory file that stores summarized successful workflows. Examples: "build and run Strata tests", "rebuild GGUF". Workflows are retrieved into the prompt.
  - It interacts well with the parked conversation cache, because a stable workflow prefix gets a cache hit.
  - Expected gain: fewer exploratory tool calls on repeated chores (inference).
- **WebArena: LOW** for a coding agent, apart from its functional-correctness grading philosophy.

### Gaps
- I found no CMU/UW study of agent reliability under local 2-3-bit quantized models, and none on whether grammar-constrained tool calling changes agent task success.
- I did not retrieve the TheAgentCompany or OpenHands-Versa score tables.

## Summary table (for the report writer)

| Paper | Lead/Co | Venue | Code | Verdict for us | Expected gain |
|---|---|---|---|---|---|
| SuffixDecoding ([2411.04975](https://arxiv.org/pdf/2411.04975v3)) | CMU lead + Snowflake | NeurIPS'25 spotlight | ArcticInference | Upgrade prompt-lookup to a cross-turn suffix tree with adaptive length | Guess +10-30% t/s on edit-heavy turns |
| XGrammar-2 ([2601.04426](https://arxiv.org/html/2601.04426v2)) | CMU lead | CAIS'26 | mlc-ai/xgrammar | TagDispatch tool-call grammars; CPU-side masks; Avalon GBNF replacement | Reliability; CPU time if grammar shows in profile |
| XGrammar ([2411.15100](https://arxiv.org/abs/2411.15100)) | CMU lead | arXiv'24 | same | Mask cache and rollback for grammar-in-spec | Same as above |
| Sequoia ([2402.12374](https://arxiv.org/abs/2402.12374v2)) | CMU lead | arXiv'24 | Infini-AI-Lab | Hardware-aware draft length/tree with measured expert-union cost | Guess 5-15% |
| Fiddler ([2402.07033](https://arxiv.org/abs/2402.07033?context=cs)) | UW lead | ICLR'25 | efeslab/fiddler | Already implemented; add token-count CPU-vs-DMA threshold | Prefill/verify only |
| S3-FIFO / SIEVE | CMU lead / CMU co | SOSP'23 / NSDI'24 | many libs | Probationary admission for VRAM expert tier | Few % hit rate (unmeasured) |
| Baleen | CMU lead | FAST'24 | artifact | Offline OPT bound on expert traces before more policy work | Decision aid |
| k-bit scaling ([2212.09720](https://arxiv.org/abs/2212.09720v2)) | UW lead | ICML'23 | bitsandbytes | Tier-aware mixed precision (≥4-bit non-expert or hot experts) | Quality |
| TriForce / MagicDec / MagicPIG / TidalDecode / Kinetics | CMU lead | COLM'24 / ICLR'25 / NeurIPS'25 | Infini-AI-Lab etc. | KV-bound premise does not hold on our hybrid model | ~0 unless attention >15% at 200k |
| SpecInfer ([2305.09781](https://arxiv.org/abs/2305.09781)) | CMU lead | ASPLOS'24 | FlexFlow | Trees need DeltaNet state branching; prefer chains | Low |
| FlashInfer ([2501.01005](https://arxiv.org/abs/2501.01005)) | UW lead, CMU co | MLSys'25 best | ROCm fork for CDNA only | No gfx1100 | None |
| MPK / Mirage | CMU lead | OSDI'26 / OSDI'25 | mirage-project | CUDA-only; launch overhead not our bottleneck | None |
| NanoFlow / Punica / Atom | UW lead | OSDI'25 / MLSys'24 | yes | Batch throughput designs | None |
| QLoRA / SpQR | UW lead | NeurIPS'23 / ICLR'24 | bitsandbytes | Possible MTP-head-only fine-tune (speculative) | Unknown |
| Mamba / Mamba-2 | CMU + Princeton | COLM'24 / ICML'24 | state-spaces | Background for DeltaNet kernels | None new |
| TheAgentCompany / WebArena / OpenHands / AWM | CMU lead / co | 2024-2025 | yes | Checkpoint grader for harness A/B; AWM workflow memory | Agent quality |
