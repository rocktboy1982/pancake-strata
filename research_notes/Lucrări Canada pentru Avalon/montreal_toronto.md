# Montreal and Toronto papers (Nov 2022 - Nov 2026) for pancake / Avalon

Scope: Mila, Université de Montréal, McGill, Polytechnique Montréal, Concordia, University of Toronto, Vector Institute. Company-led papers are excluded or flagged. Before writing, I checked `/home/dan/reports/` (Stanford, MIT/Harvard, US universities, Europe, upstream). None of the papers below appears there. The only overlap is the EAGLE family (Peking lead, co-authored at Waterloo/Vector), which earlier reports already cover, so it is not repeated here. "Lead" means a Montreal/Toronto first author, or a last author running the work there. "Co" means the institution appears only among the co-authors.

Method: I listed every arXiv paper since Nov 2022 by about 25 key Montreal/Toronto faculty using the arXiv API: Pekhimenko, Vijaykumar, Mehri Dehnavi, Moshovos, Xujie Si, Jimmy Ba and Ashvin Goel (Toronto); Courville, Chandar, Reddy, Sordoni, Rish, Berseth and Lajoie (Mila/UdeM/McGill/Poly); W. Gross (McGill); Khomh and Leduc-Primeau (Polytechnique); T.-H. Chen (Concordia). I then pulled abstracts and full text for the shortlist. Faculty I did not list may have relevant papers that are missing here (see Gaps).

Ratings for "verdict / expected gain" are my own inferences from the cited facts. They apply to batch-1 decode at about 126 t/s, Qwen3.8-Flash-Next GSQ-RCO Q2_0, about 44% of experts in VRAM, and CPU computing the missed experts.

---

## Q1. Which Montreal/Toronto ideas could speed up batch-1 MoE decode with offloaded experts, long-prompt prefill, or drafting on a hybrid recurrent model?

### Takeaway
SpAx (Toronto lead, Oct 2026) is the only paper from these groups aimed squarely at batch-1 decode with offloaded weights. Its gain shrinks as the base precision drops: 3.86x at BF16 but 2.06x at 4-bit. Q2_0 experts computed on the CPU are close to its worst case, so only its "skip" tier and a repacked down_proj are worth testing. I found nothing from these groups on MoE expert caching or prefetch, on drafting for hybrid or recurrent models, or on GDN kernels. The drafting work is limited to adaptive speculation length and a learned choice of draft model. Pancake's DraftPolicy already covers most of both.

### Cited Findings

**1. SpAx: Activation Sparsity with Weight Approximation for Faster LLM Decoding on Offloaded Weights**. JuneHyung Kim (work done at UofT, now LG Electronics), Sankeerth Durvasula, Nandita Vijaykumar (UofT). **Toronto lead.** arXiv 2610.02598, 1 Oct 2026, preprint.
- Core idea: each weight column gets one of three options instead of read or skip. Columns for the largest activations are read at full precision. The middle band is read as a 2-bit approximation. Columns for near-zero activations are skipped — [arXiv abs](https://arxiv.org/abs/2610.02598)
- Speedups with CPU-RAM offload: 3.86x average (up to 5.57x) for BF16 weights, and 2.06x (up to 2.74x) for 4-bit, at no more than 10% WikiText-2 perplexity increase. With flash/NVMe offload: 3.31x for BF16 and 1.54x (up to 2.03x) for 4-bit — [arXiv abs](https://arxiv.org/abs/2610.02598)
- Implemented in llama.cpp at batch size 1. Hardware: RTX 4090, PCIe 4.0 x8, 128 GB RAM, NVMe. The selected codes are moved to the GPU and dequantized and multiplied there. The 4-bit deployment format is Q4_K_M. Models are dense only (Llama-3.1-8B, Gemma-4-31B, Qwen3.6-27B), with no MoE results — [arXiv HTML](https://arxiv.org/html/2610.02598v1)
- The middle tier uses 2-bit Lloyd-Max scalar codes, or trellis codes with a shared 2^16 Gaussian codebook (QTIP-style, Viterbi-encoded). Per-token top-k ranking replaces fixed thresholds. Per-layer read budgets (QKV, O, gate/up, down) are calibrated offline on WikiText-2 — [arXiv HTML](https://arxiv.org/html/2610.02598v1)
- On Llama-3.1-8B BF16, SpAx at a 27% read budget matches TEAL at 57%: PPL 6.87 vs 6.89 against 6.24 dense. GSM8K is 43.9 vs 42.2, against 50.2 dense. Both lose about 6 GSM8K points versus dense — [arXiv HTML](https://arxiv.org/html/2610.02598v1)
- 2-bit QTIP alone raises PPL by 44.8%, versus 13.5% for SpAx-trellis. No code repository is linked — [arXiv HTML](https://arxiv.org/html/2610.02598v1)

**2. The Synergy of Speculative Decoding and Batching in Serving LLMs**. Qidong Su, Christina Giannoula, Gennady Pekhimenko (UofT). **Toronto lead.** arXiv 2310.18813, Oct 2023.
- Finds that the optimal speculation length depends on batch size, builds a quantitative model of why, and proposes adaptive speculation length that matches or beats fixed-length schemes — [arXiv](https://arxiv.org/abs/2310.18813)

**3. Context-Aware Assistant Selection for Improved Inference Acceleration with LLMs**. Jerry Huang, Prasanna Parthasarathi, Mehdi Rezagholizadeh, Sarath Chandar (Mila/Polytechnique; two co-authors are from Huawei Noah's Ark). **Montreal lead.** EMNLP 2024 main.
- Frames the choice among several draft models as a contextual bandit. A policy trained offline on how well each drafter's outputs align with the target picks the drafter per context, which speeds up decoding across domains — [arXiv](https://arxiv.org/abs/2408.08470)

**4. Forgetting Transformer (FoX)**. Zhixuan Lin, Evgenii Nikishin, Xu Owen He, Aaron Courville (Mila). ICLR 2025. **Adaptive Computation Pruning (ACP) for FoX**: Lin, Obando-Ceron, He, Courville. COLM 2025. **Mila lead.**
- FoX adds a data-dependent forget gate to softmax attention. It needs no positional embeddings, is FlashAttention-compatible, and keeps better needle-in-a-haystack recall than Mamba-2, HGRN2 and DeltaNet — [arXiv 2503.02130](https://arxiv.org/abs/2503.02130)
- ACP skips attention entries that the forget gate has decayed below a provably safe threshold. That removes about 70% of attention FLOPs and memory accesses, makes attention 2-3x faster, and raises training throughput 10-40% with no quality loss. Code: github.com/zhixuan-lin/forgetting-transformer — [arXiv 2504.06949](https://arxiv.org/abs/2504.06949)

**5. Hexcute** (Xiao Zhang, Yaoyao Ding, …, Pekhimenko; UofT lead, arXiv 2504.16214) and **Tilus** (Yaoyao Ding, …, Pekhimenko; UofT lead with NVIDIA/CMU, arXiv 2504.12984)
- Hexcute treats GPU layout synthesis as constraint programming. On mixed-type MoE operators it averages 6.46x over Triton. In vLLM it gives up to 2.60x on DeepSeek-R1-AWQ and 2.04x on a Mamba-based model — [arXiv](https://arxiv.org/abs/2504.16214)
- Tilus is a tile-level language for arbitrary 1-8-bit types. The authors report 1.75x over Triton, 2.61x over Ladder, 1.29x over QuantLLM and 1.03x over Marlin. Code: github.com/NVIDIA/tilus, NVIDIA GPUs only — [arXiv](https://arxiv.org/abs/2504.12984)

**6. Spectra 1.1: Scaling Laws and Efficient Inference for Ternary LMs**. Tejas Vaidhya, …, Irina Rish (Mila/UdeM with Nolano). **Montreal lead.** arXiv 2506.23025, Jun 2025.
- Proposes 2-bit and 1.6-bit packings of ternary weights that speed up CPU inference on several architectures, plus a TriRun GPU kernel with up to 5x end-to-end speedup over floating point — [arXiv](https://arxiv.org/abs/2506.23025)

**7. What Does Layer-Importance Reveal About Transformers and State-Space Models?** Abbes, Islah, Rish, Chandar (Mila). **Montreal lead.** arXiv 2609.16537, Sep 2026.
- Splits layer importance into Necessity (loss increase when a layer is bypassed) and Plasticity (size of fine-tuning updates). In transformers up to 14B the two anti-align across depth. In Mamba-style SSMs they overlap — [arXiv](https://arxiv.org/abs/2609.16537)

**8. InnerQ: Hardware-Aware Tuning-Free Quantization of KV Cache**. Tayaranian Hosseini, Ardakani, Warren J. Gross (McGill). **Montreal lead.** arXiv 2602.23200, Feb 2026.
- Quantizes the KV cache in groups along the inner dimension so dequantization lines up with the vector-matrix product. Adds hybrid sym/asym groups, high-precision windows for recent and sink tokens, and folded per-channel key normalization. Reports 1.3x over earlier KV-quant methods and 2.7x over unquantized — [arXiv](https://arxiv.org/abs/2602.23200)

**9. Proteus: Incremental Memory Activation for Long-Context Sequence Modeling**. Bayat, Behrouz, Mirrokni, Courville (Mila with Google). arXiv 2608.16844, Aug 2026.
- Gradually unlocks recurrent-memory capacity as context grows. Gains on SWLA, Comba, Titans and Hope-Attention grow with length. It is a training-time architecture change — [arXiv](https://arxiv.org/abs/2608.16844)

**10. Lower-relevance systems papers, for completeness**
- Seesaw (UofT lead, arXiv 2503.06433): re-shards the model between prefill and decode for multi-GPU throughput — [arXiv](https://arxiv.org/abs/2503.06433)
- CRAFT (UofT lead with AWS): cost-aware expert replication for multi-GPU expert parallelism, 1.14x average throughput — [arXiv](https://arxiv.org/abs/2603.28768)
- ScatterMoE (Shawn Tan, …, Courville; Mila lead with IBM, arXiv 2403.08245): padding-free MoE kernels in Triton, aimed at training and throughput — [arXiv](https://arxiv.org/abs/2403.08245)
- Stick-breaking attention (Mila lead with IBM, arXiv 2410.17980): replaces softmax+RoPE, FlashAttention-adapted — [arXiv](https://arxiv.org/abs/2410.17980)
- minGRU/minLSTM "Were RNNs All We Needed?" (Borealis AI lead, Bengio co-author): architecture only — [arXiv](https://arxiv.org/abs/2410.01201)

### Inferences
- **SpAx, verdict medium as an experiment, low as a port.** Most of its gain comes from sending fewer bytes over PCIe at high base precision. In pancake, missed experts are already Q2_0 (about 2 bits) and are computed on the CPU, not transferred. A 2-bit middle tier therefore saves almost nothing, and the drop from 3.86x (BF16) to 2.06x (4-bit) suggests a 2-bit base would gain much less. Two parts may still be worth testing.
  - (a) A per-token top-k skip tier on the **intermediate activation (SwiGLU output) feeding down_proj** of CPU-computed experts. That cuts DDR4 bytes read per missed expert. GGUF stores down_proj with blocks along the intermediate (K) dimension, so skipping columns means skipping whole 256-wide blocks unless the weights are repacked by intermediate channel.
  - (b) For NVMe-tier experts, reading only the top channels. This is where the paper's flash results apply.
  - Expected gain: unknown. At best a fraction of the CPU-expert time (about half of decode) if 30-50% of down_proj can be skipped at acceptable quality. TEAL-style degradation (about 6 GSM8K points at the matched-PPL setting) is a real risk at Q2. Measure first on code tasks, not perplexity (see item 1 in Q3).
- **Adaptive speculation length (Su et al.)**: no new gain. At batch 1 it reduces to the per-step adaptive length that DraftPolicy already uses.
- **Assistant selection (Chandar)**: low to medium gain. The bandit framing could make DraftPolicy's choice among MTP, prompt-lookup and SuffixDrafter context-driven, using features such as code vs prose, tool-output echo or recent acceptance. Expect a small change in tokens per step (a few %) only if the current heuristic often picks badly. Check DraftPolicy logs before building anything.
- **FoX/ACP**: not applicable at inference. The trick depends on a model trained with forget-gated attention. The QSA layer is already sparse, and the GDN layers carry their own gates.
- **Hexcute/Tilus**: no direct gain. Both are NVIDIA/CUDA, and gfx1100 HIP is not supported. The useful lesson is Tilus's arbitrary-bit (3/5/6-bit) layout algebra if pancake ever adds odd-bit expert formats.
- **Spectra 1.1 CPU packings**: low gain. The 2-bit and 1.6-bit ternary packing tricks could be compared against pancake's AVX2 Q2_0 expert kernel. They target ternary weights, not Q2_0's 4-level grid.
- **Layer-importance paper**: low gain, an idea only. For a hybrid model it argues for checking Necessity separately in GDN layers and in QSA layers before giving them different bit budgets.
- **InnerQ**: no gain expected. TurboQuant KV already brought nothing, and only 1 in 4 layers has a KV cache.
- No Montreal/Toronto paper addresses expert caching, admission, prefetch, router prediction, NVMe expert I/O or MTP/GDN replay. Those topics are owned by the US, Europe and Asia reports.

### Gaps
- No Montreal/Toronto paper on MoE expert offloading, caching or prefetch was found. A search tying the topic to Toronto affiliations returned nothing, and the faculty arXiv lists contain none.
- No Montreal/Toronto paper on speculative decoding for SSM, hybrid or Gated-DeltaNet models was found.
- SpAx: no code link, no MoE results, and no results below a 4-bit base. Its behaviour on Q2 experts is untested.

---

## Q2. Reasoning-efficiency and agent work usable for opencode

### Takeaway
Montreal (Mila/McGill) is strong on reasoning-length analysis. DeepSeek-R1 Thoughtology finds a problem-specific sweet spot and measures rumination with 5-gram repetition. The Markovian Thinker / Delethink makes reasoning chunked with constant memory. Toronto (Pekhimenko group) and Concordia have recent agent papers. The most usable for opencode are ECLoop (+4.8-11.8 pp on SWE-bench Verified with up to 12.1% fewer tokens), STAIR (reusing past repair trajectories), and Aegis/SpecCache (environment-side fixes and speculative tool actions). Test-time-scaling papers (RSA, Scorpio, RL^V) need parallel sampling and are a poor fit for one GPU at about 126 t/s.

### Cited Findings

**1. DeepSeek-R1 Thoughtology: Let's Think About LLM Reasoning**. Sara Vera Marjanović, Arkil Patel, …, Siva Reddy (Mila / McGill / Copenhagen). **Montreal lead (last author and correspondence at Mila).** TMLR, arXiv 2504.07128, Apr 2025 (v3 Jan 2026).
- Finds a problem-specific optimal reasoning length: past it, performance falls. On AIME-24, MATH500 and GSM8k, correct thoughts are on average much shorter than incorrect ones — [arXiv HTML](https://arxiv.org/html/2504.07128v3)
- Unconstrained R1 averages about 1,388 tokens per GSM8k thought. With budgets of 64-1,024 tokens, output tokens can be cut nearly in half with little loss — [arXiv HTML](https://arxiv.org/html/2504.07128v3)
- Rumination, the repeated revisiting of approaches already tried, is measured by 5-gram repetition rate and normalized lexical entropy. It rises with difficulty and may be negatively associated with accuracy. Code: github.com/McGill-NLP/thoughtology — [arXiv HTML](https://arxiv.org/html/2504.07128v3)

**2. The Markovian Thinker: Architecture-Agnostic Linear Scaling of Reasoning (Delethink)**. Milad Aghajohari, Kamran Chitsaz, Amirhossein Kazemnejad, Sarath Chandar, Alessandro Sordoni, Aaron Courville, Siva Reddy (Mila/McGill/Polytechnique; Sordoni is at Microsoft Research). **Montreal lead.** arXiv 2510.06557, Oct 2025, ICLR 2026 poster.
- Reasoning runs in fixed-size chunks. At each boundary the context is reset to the prompt plus a short carryover, giving linear compute with constant memory. An R1-Distill-1.5B model thinks in 8K chunks up to 24K tokens and matches or beats LongCoT-RL at 24K — [arXiv](https://arxiv.org/abs/2510.06557)
- At a 96K average thinking length, the estimated cost is 27 H100-months for LongCoT-RL versus 7 for Delethink. Off-the-shelf reasoning models from 1.5B to 120B often produce Markovian traces zero-shot — [arXiv](https://arxiv.org/abs/2510.06557); ICLR acceptance: [ICLR 2026 poster](https://www.iclr.cc/virtual/2026/poster/10011666)

**3. Robust Reasoning Benchmark**. Golikov, Opryshko, Pekhimenko, Mark C. Jeffrey (UofT). **Toronto lead.** arXiv 2604.08571, Mar 2026.
- Open-weights reasoning models lose up to 54% average accuracy under 13 deterministic text perturbations of AIME. Solving several independent problems in one context shows "Intra-Query Attention Dilution": accuracy decays on later problems for 7B-120B models. The authors argue for explicit context resets inside the chain of thought — [arXiv](https://arxiv.org/abs/2604.08571)

**4. ArbiGraph: Verifiable Task Graphs for Evaluating Context Management**. Same UofT group. **Toronto lead.** arXiv 2607.20764, Jul 2026.
- A generator for chained tool-assisted tasks with exact verification. A Qwen3.5-27B tool agent loses up to 33.3% accuracy on branching chains of dependent tasks. Code: github.com/pavelgolikov/ArbiGraph — [arXiv](https://arxiv.org/abs/2607.20764)

**5. Aegis: Taxonomy and Optimizations for Overcoming Agent-Environment Failures**. Kevin Song, …, Gennady Pekhimenko (UofT). **Toronto lead.** arXiv 2508.19504, Aug 2025.
- From 142 traces (3,656 turns) across 5 agent benchmarks, the authors derive 6 agent-environment failure modes. Three fixes target the environment: better observability, offloading common computations, and speculative agentic actions. Together they raise success rates 6.7-12.5% without changing the model or agent — [arXiv](https://arxiv.org/abs/2508.19504)

**6. What Limits Agentic Systems Efficiency? (SpecCache)**. Song Bian, Minghao Yan (Wisconsin), Anand Jayarajan, Gennady Pekhimenko (UofT), Shivaram Venkataraman (Wisconsin). **Toronto co-author.** arXiv 2510.16276, Oct 2025.
- Web-environment latency can reach 53.7% of end-to-end agent latency. SpecCache, a cache plus speculative execution, raises cache hit rate up to 58x over random caching and cuts environment overhead up to 3.2x — [arXiv](https://arxiv.org/abs/2510.16276)

**7. ECLoop: Preventing Premature Commitment in Coding Agents with an Evidence-Conditioned Execution Layer**. Yisen Xu, …, Jinqiu Yang, Tse-Hsun (Peter) Chen (Concordia). **Montreal lead.** arXiv 2607.28815, Jul 2026.
- From the issue text and repository structure, it compiles the evidence an agent must observe before each kind of edit or submission. It then delays actions until that evidence has been seen. Pass@1 on all 500 SWE-bench Verified tasks rises 4.8-11.8 pp across two models and two scaffolds, and token use falls by up to 12.1% — [arXiv](https://arxiv.org/abs/2607.28815)

**8. STAIR: Reusing Past Repairs Through Hierarchical Trajectory Abstraction for Coding Agents**. Yisen Xu, Jiayuan Zhou, Ruiqi Pan, Tse-Hsun Chen (Concordia). **Montreal lead.** arXiv 2607.29658, Jul 2026.
- Turns past repair trajectories into multi-level plan trees, from diagnostic steps up to strategies, and injects the relevant nodes into the prompt. Results: 81.2% Pass@1 on SWE-bench Verified (Lingxi + MiniMax M2.5). mini-SWE-agent v2 goes from 75.8% to 81.0% with no code change. Raw, unabstracted trajectories transfer much worse — [arXiv](https://arxiv.org/abs/2607.29658)

**9. CodePromptZip**. Pengfei He, Shaowei Wang, Tse-Hsun Chen (Concordia with Manitoba). **Montreal co-lead.** Findings of ACL 2026.
- A small LM with a copy mechanism compresses retrieved code by dropping token types in a priority order learned through program analysis. It beats the best compression baseline by 23.4%, 28.7% and 8.7% on three code tasks — [arXiv](https://arxiv.org/abs/2502.14925)

**10. KV-streams for Efficient Compaction in Agentic RL**. Penaloza, …, Glen Berseth, …, Alessandro Sordoni, …, Laurent Charlin, Guillaume Lajoie (Mila). **Montreal lead.** arXiv 2609.35750, Sep 2026.
- Streams the KV cache forward through context compaction instead of flushing it and re-prefilling. Training is 2.6-5x faster in wall-clock time, and the streamed KV acts as a recurrent state — [arXiv](https://arxiv.org/abs/2609.35750)

**11. Test-time scaling (Mila lead)**
- RSA, Recursive Self-Aggregation (Venkatraman, …, Bengio, …, Moksh Jain): repeatedly aggregates subsets of a population of reasoning chains. Qwen3-4B-Instruct becomes competitive with R1 and o3-mini (high) on AIME-25, LiveCodeBench-v6 and others — [arXiv 2509.26626](https://arxiv.org/abs/2509.26626)
- Scorpio (Buvanesh, …, Courville): scores randomly grouped candidates jointly. Verification improves up to 7 pp, peak accuracy is matched with 17-42% less inference, and aggregation on code improves 11 pp. Code: github.com/anirudhb11/scorpio — [arXiv 2611.00350](https://arxiv.org/abs/2611.00350)
- RL^V (Sareen, …, Hosseini): trains a joint reasoner and verifier, improving MATH by over 20% with parallel sampling and making test-time compute 8-32x more efficient — [arXiv 2505.04842](https://arxiv.org/abs/2505.04842)

**12. Agent safety and tooling (Toronto/Polytechnique lead)**
- ToolEmu (Ruan, …, Jimmy Ba, Chris Maddison, Hashimoto; UofT lead, arXiv 2309.15817): an LM emulates tool execution and an LM evaluator scores risk. 68.8% of the failures it finds are valid real-world failures, and even the safest agent fails 23.9% of the time on 36 tools / 144 cases — [arXiv](https://arxiv.org/abs/2309.15817)
- MCP descriptor attacks (Jamshidi, …, Foutse Khomh; Polytechnique lead, arXiv 2512.06556): Tool Poisoning, Shadowing and Rug Pull through tool descriptors cause up to 36% unsafe invocations. Layered defenses cut that to 15% — [arXiv](https://arxiv.org/abs/2512.06556)
- APE, "LLMs Are Human-Level Prompt Engineers" (Yongchao Zhou, …, Jimmy Ba; UofT lead, arXiv 2211.01910, Nov 2022): LLM-proposed instructions match or beat human ones on 19/24 tasks — [arXiv](https://arxiv.org/abs/2211.01910)
- LLMs Can't Play Hangman (Baldelli, …, Chandar; Mila/Polytechnique, CoLLAs 2026): agents limited to the public transcript cannot hold hidden state consistently, and an explicit private working memory fixes it — [arXiv](https://arxiv.org/abs/2601.06973)

**13. Co-authored only**
- "Reasoning on a Budget" survey (Huawei Noah's Ark lead, Mark Coates of McGill/Mila co-author). It notes that thinking length tracks the allocated budget, with a long tail of problems that overshoot it — [arXiv 2507.02076](https://arxiv.org/pdf/2507.02076)
- DARS coding-agent re-sampling (Zhijing Jin co-author; affiliation at the time not verified): 47% pass@1 and 55% pass@k on SWE-bench Lite with Claude 3.5 Sonnet — [arXiv 2503.14269](https://arxiv.org/abs/2503.14269)

### Inferences
- **Thoughtology rumination metric → loop guard / soft close. Verdict: high value, cheap to try.** A rolling 5-gram repetition rate over the current thinking span is an O(1)-per-token signal. Pancake's loop guard and thinking-budget soft close could trigger earlier on rumination rather than only on exact loops. Expected gain is fewer wasted thinking tokens on hard turns. The size is unknown and should be measured on opencode traces (tokens per turn, task success).
- **Thoughtology budgets support pancake's per-level budgets.** The sweet spot depends on the problem, so a static global budget is suboptimal, and budgets tied to difficulty or effort level (which pancake already has) match the evidence.
- **Markovian Thinker / Delethink: low speed gain, medium as an experiment.** Its compute argument (quadratic attention) matters less here: 3 of 4 layers are GDN with constant state, and only QSA keeps a KV cache. The useful parts are behavioural:
  - (a) Off-the-shelf models already produce Markovian traces zero-shot, so a "thinking reset" (drop old thinking, keep a carryover summary) after N tokens is plausible without retraining.
  - (b) RRB's attention-dilution result points the same way.
  - Cheapest version: make sure earlier turns' thinking is stripped from the conversation cache (Qwen templates usually do this), and test in-turn resets only for very long thinking. With GDN, a reset means restoring a boundary checkpoint, which pancake's conversation cache already supports.
- **ECLoop: high for agent quality, medium effort.** It sits between the agent and the repository, so it can run as an opencode plugin or tool wrapper. A rule such as "must have read file X / run test Y before an edit or submit" also cuts tokens by up to 12.1%. That is directly useful for a local model where every token costs about 8 ms.
- **STAIR: medium.** A single user builds a growing library of opencode sessions on the same repositories. Abstracted plans from past fixes, injected as prompt hints, fit well. It costs prompt tokens, but pancake's prompt/conversation caching softens that.
- **Aegis / SpecCache: medium.** Tool-side latency (file reads, greps, test runs) during decode can be hidden by speculatively running likely read-only tool calls and caching their results. Aegis's observability fixes (richer tool error messages) complement pancake's tool-call recovery.
- **RSA / Scorpio / RL^V / DARS: low.** They need N parallel samples. At batch 1 and about 126 t/s on one GPU, that multiplies latency. Use them only offline for hard tasks.
- **MCP descriptor attacks / ToolEmu: low for speed, relevant for safety** if opencode loads third-party MCP servers.
- **CodePromptZip / KV-streams: low.** The first needs a trained compressor, the second is training-side. The KV-streams idea (compaction without re-prefill) maps onto pancake's existing GDN-state checkpoints rather than adding anything new.

### Gaps
- Delethink's carryover size and the zero-shot Markovian rates per model (for example GPT-OSS-120B) were not visible in the abstract. Check the PDF before designing a reset policy.
- Thoughtology's per-budget accuracy numbers are only in figures, which I could not read.
- No Montreal/Toronto paper evaluates thinking budgets or effort levels on agentic coding turns specifically.

---

## Q3. Quantization, sparsity, kernels and evaluation usable with GGUF on CPU+GPU

### Takeaway
The most useful Toronto result for pancake is a methodology warning, not a kernel. "Displacement Is Not Direction" (Moshovos lab, Jun 2026) finds that per-token KLD against a high-precision reference stops predicting benchmark quality among near-baseline quants. It tested this on a 28-quant cohort of Qwen3.6-35B-A3B and a 41-quant cohort of Devstral-24B. The Mehri Dehnavi group's compression papers (SLiM, PATCH, OPTIMA) need re-pruning or re-quantizing a 125B model and depend on 2:4 sparsity support. That makes them impractical here.

### Cited Findings

**1. Displacement Is Not Direction: Evaluating Fidelity Metrics for Quantized LLM Deployment**. Miloš Nikolić, Ali Hadi Zadeh, Enrique Torres Sanchez, Andreas Moshovos (UofT). **Toronto lead.** arXiv 2606.19558, Jun 2026.
- Over the full cohorts, KLD correlates strongly with benchmark score: ρ = -0.72 on Qwen3.6-35B-A3B (28 quants) and -0.86 on Devstral-Small-2-24B (41 quants). In the near-baseline "silent zone" the correlation collapses: ρ = +0.00 on Qwen, and -0.24 (p = 0.36) on Devstral — [arXiv](https://arxiv.org/abs/2606.19558)
- The collapse holds across 14 variants: KLD aggregations, perplexity forms, top-1 agreement, calibration corpora and context lengths. On LiveCodeBench, per-prompt KLD barely predicts failure (failed/passed ratio 1.08-1.22), and as a cross-model router it reaches only 42.3-49.4% — [arXiv](https://arxiv.org/abs/2606.19558)
- KLD mainly measures how much a quant disagrees with the reference, not which direction those disagreements go — [arXiv](https://arxiv.org/abs/2606.19558)

**2. SLiM: One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression**. Mohammad Mozaffari, Amir Yazdanbakhsh (Google DeepMind), Maryam Mehri Dehnavi (UofT). **Toronto lead.** ICML 2025 (PMLR 267).
- Pipeline: quantize (probabilistic SLiM-Quant), prune to 2:4 with Wanda, then add low-rank adapters with a saliency function to absorb the combined error. Accuracy improves up to 5.66% on LLaMA-2-7B (2:4 + 4-bit). The ICML abstract reports up to 4.3x speedup on an RTX 3060 and 3.8x on an A100; arXiv v2 says 3.78x/3.75x layer-wise — [PMLR](https://proceedings.mlr.press/v267/mozaffari25a.html); [arXiv](https://arxiv.org/abs/2410.09615)

**3. PATCH: Learnable Tile-level Hybrid Sparsity for LLMs**. Hourri, Mozaffari, Mehri Dehnavi (UofT). **Toronto lead.** NeurIPS 2026.
- A learnable mask makes each tile either dense or 2:4, giving any sparsity from 0 to 50%. On LLaMA-2-7B / A6000: 1.18-1.38x end-to-end, and +0.37 to +2.96 accuracy over MaskLLM — [arXiv](https://arxiv.org/abs/2509.23410)

**4. OPTIMA: Optimal One-shot Pruning via Quadratic Programming Reconstruction**. Mozaffari, Kushnir, Mehri Dehnavi, Yazdanbakhsh. **Toronto lead.** ACML 2026.
- Reconstructs weights after pruning by solving column-wise QPs that share one Hessian. Zero-shot accuracy averages +0.61, +0.56 and +1.01 at 50% unstructured, 2:4 and 60% unstructured, and up to +3.97. Pruning an 8B model takes 40 h on one H100 — [arXiv](https://arxiv.org/abs/2512.13886)

**5. CuTeGen**. Tara Saba, …, Xujie Si (UofT), Fan Long (UofT). **Toronto lead.** arXiv 2604.01489, Apr 2026.
- Agentic generate-test-refine kernel synthesis over CuTe, with profiling feedback delayed until the kernel's structure stabilizes. Averages 1.71x over PyTorch on KernelBench L1+L2, against 0.89x for CudaForge. Code: github.com/taratt/cutegen — [arXiv](https://arxiv.org/abs/2604.01489)

**6. Exploring Quantization for Efficient Pre-Training of Transformer LMs** (Chitsaz, Fournier, Mordido, Chandar; Mila/Polytechnique, arXiv 2407.11722) and **LORD** low-rank compression of code LLMs (Kaushal, Vaidhya, Rish; Mila/Nolano, arXiv 2309.14021). Both are training or compression-recipe papers, listed for completeness — [arXiv 2407.11722](https://arxiv.org/abs/2407.11722); [arXiv 2309.14021](https://arxiv.org/abs/2309.14021)

### Inferences
- **"Displacement Is Not Direction": high value for methodology, zero runtime gain.** Variant choices such as GSQ-RCO Q2_0 vs IQ2_XS, or which shard to requantize, should not rest on KLD or PPL once candidates are near baseline. Use a fixed opencode-style code or agent task set (pass rate) instead. It also argues against using KLD to tune SpAx/TEAL-style sparsity (Q1) and to validate it. Its cohorts are exactly the model families the user runs (Qwen3.x-A3B MoE, Devstral).
- **SLiM / PATCH / OPTIMA: low.** All need a calibration pass over the model, which for 125B is not realistic at home (earlier reports already judged re-quantization at home unrealistic). Their speedups depend on 2:4 sparse tensor-core support. I do not believe RDNA3/gfx1100 WMMA exposes structured sparsity, and llama.cpp/GGUF has no 2:4 format. That is unverified, so treat it as an assumption.
- **CuTeGen: low to medium as process.** It is CUDA/CuTe-specific, but its method fits how the user already tunes HIP kernels with agents: delay low-level profiling feedback until the structure stabilizes.
- **Tilus/Hexcute and Spectra packings**: see Q1. Ideas only, no port to gfx1100.

### Gaps
- Not verified: whether gfx1100 WMMA supports structured sparsity. My belief that it does not is untested.
- No Montreal/Toronto paper offers GGUF/llama.cpp-native low-bit kernels for AMD or AVX2 CPUs. SpAx is the only one built on llama.cpp, and it ran on an RTX 4090.
- Faculty I did not list (Vector's Wenhu Chen and Hongyang Zhang are at Waterloo, outside scope; also Chris Maddison, Roger Grosse, Rahul Krishnan, Concordia's Emad Shihab and Jinqiu Yang beyond the co-authored papers, McGill's Xue Liu) may have relevant work this sweep missed. Company-affiliated Montreal work (Microsoft Research Montréal, e.g. FrogNano 4B coding agent arXiv 2609.07925 and BugPilot arXiv 2510.19898; ServiceNow BrowserGym/WorkArena) was excluded by rule.
