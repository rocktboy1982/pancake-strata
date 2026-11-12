# Papers from other US universities (UCSD, Rice, Duke, Ohio State, Stevens, UT Austin, UIUC, Yale, UVA, Emory, UW-Madison...) for pancake / Avalon / opencode

Scope: Nov 2022 to Nov 2026. Stanford, MIT, Harvard, Berkeley, CMU, UW, Princeton and Cornell are excluded, so a paper led from one of those appears only as a co-author note. Each paper is flagged LEAD (a target university is first or primary) or CO (a target university is a co-author only).

Target setup: Qwen3.8-Flash-Next, a ~125B MoE hybrid (Gated DeltaNet + attention, MTP head), at 2-3 bits on an RX 7900 XT 20 GB with a Ryzen 7 5700G and NVMe. Batch size is 1 and the workload is one opencode coding agent. Pancake's replay ceilings are tier 79.6%, LRU 89.5% and Belady 94.7%.

Method note: this session ran 21 tool calls, one usage-limit interruption included, and verified every entry below through a search or fetch. Items I know from prior knowledge but did not verify this session are listed only under Gaps, as candidates.

---

## Q1. Which expert caching, prefetching or router-prediction methods could close pancake's gap between the tier hit rate and the LRU/Belady ceilings at batch 1?

### Takeaway
The US-university MoE-offload literature splits into three families:
- (a) Learned or statistical predictors of upcoming experts: SiDA (Duke), MoE-Beyond, eMoE (UVA), fMoE/FineMoE (Stevens).
- (b) Cross-layer expert affinity: ExFlow (Ohio State).
- (c) Router decoupling, where experts are chosen ahead of time: Read-ME (UT Austin).

Pancake's adaptive tier scores *below plain LRU* (79.6% vs 89.5%). That points to a policy problem that recency fixes, not to a missing predictor. The cheapest win is probably a recency-managed dynamic region, for example SIEVE (Emory), next to the frequency tier. Predictors (a, b) matter only for prefetching into PCIe slack, and that slack is small on this machine.

### Cited Findings
- **fMoE / FineMoE: "Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading"**. Hanfei Yu et al., **Stevens Institute of Technology (LEAD)**. arXiv 2502.05370, Feb 2025; listed as EuroSys '26.
  - Core idea: "expert maps" record the gate's per-iteration probability distributions across layers (the "trajectory"). The prompt's semantic embedding is matched against historical expert maps to guide prefetching, caching and offloading.
  - Results: −47% latency and +36% expert hit rate vs prior work (v1); the v2 text says +39% hit rate. Prototype on HF Transformers, six-GPU testbed. — [arXiv html v1](https://arxiv.org/html/2502.05370v1), [alphaxiv v2 title](https://alphaxiv.org/abs/2502.05370)
- **SiDA-MoE: Sparsity-Inspired Data-Aware Serving**. Zhixu Du, Hai Li, Yiran Chen et al., **Duke (LEAD)**. MLSys 2024, arXiv 2310.18859.
  - Core idea: an offline-trained LSTM "hash function" (L classification heads) predicts the experts each input will activate. A separate thread builds the hash table ahead of inference, so non-predicted experts can be offloaded.
  - Results: up to 99% prediction accuracy, up to 80% GPU memory saving at <1% quality drop, up to 3.93× throughput.
  - A third-party summary notes mispredictions fall back to the CPU and hurt tail latency.
  - Code: github.com/timlee0212/SiDA-MoE. — [MLSys proceedings](https://proceedings.mlsys.org/paper_files/paper/2024/hash/698cfaf72a208aef2e78bcac55b74328-Abstract-Conference.html), [slides](https://mlsys.org/media/mlsys-2024/Slides/2638.pdf), [Emergent Mind summary](https://www.emergentmind.com/topics/sida-moe)
- **eMoE: Task-aware Memory Efficient MoE Inference**. Tairin, Mahmud, Shen, Iyer. arXiv 2503.06823, Mar 2025. I believe the group is at the University of Virginia, but the snippets I saw do not confirm the affiliation.
  - Core idea: a correlation matrix of expert selections in consecutive layers, built from earlier requests, drives on-demand loading of "popular" experts. The predictor runs only every few prompts, because reusing experts across subsequent prompts barely changes perplexity. Prediction is skipped for tasks that are insensitive to routing accuracy.
  - Results: up to 80% memory reduction and up to 17% lower latency (author claims). — [arXiv](https://arxiv.org/html/2503.06823v1), [aitopics abstract](https://aitopics.org/doc/arxivorg:E20E93D2)
- **ExFlow: Exploiting Inter-Layer Expert Affinity for Accelerating MoE Inference**. Jinghan Yao, Quentin Anthony, Aamir Shafi, Hari Subramoni, D.K. Panda, **Ohio State (LEAD)**. IPDPS 2024, arXiv 2401.08383.
  - Core idea: the experts chosen in consecutive layers are strongly correlated ("expert affinity"). The affinity can be estimated from about 5,000 sampled tokens.
  - Results: up to 2.2× throughput and up to 40% less all-to-all traffic. The setting is multi-GPU expert placement, not offload. — [arXiv html](https://arxiv.org/html/2401.08383v2), [NOW lab](https://nowlab.cse.ohio-state.edu/publication/1062/plain), [poster](https://mug.mvapich.cse.ohio-state.edu/static/media/mug/presentations/25/Poster-Jinghan_MUG25.pdf)
- **Read-ME: Refactorizing LLMs as Router-Decoupled MoE with System Co-Design**. Ruisi Cai, Yeonju Ro et al., **UT Austin VITA group (LEAD)**. NeurIPS 2024, arXiv 2410.19123.
  - Core idea: a pre-gating router, decoupled from the backbone, computes expert selection ahead of time. That enables lookahead scheduling and expert-aware batching and caching.
  - Requires: converting or training the model. It is built from a dense LLM via activation sparsity.
  - Code: github.com/VITA-Group/READ-ME. — [NeurIPS](https://neurips.cc/virtual/2024/poster/94027), [alphaxiv](https://alphaxiv.org/abs/2410.19123)
- **MoE-Beyond: Learning-Based Expert Activation Prediction on Edge Devices**. arXiv 2508.17137, Aug 2025. Affiliation not verified; a catalyzex author page lists Rohit Chawla, and I could not confirm a university.
  - Core idea: a lightweight transformer is trained on 66M expert-activation traces (DeepSeek-V2-Lite) to do multi-label prediction.
  - Results (simulation only): at a 10% expert budget, hit rate >70% vs 17% for MoE-Infinity. Predictor accuracy is 97.5% (F1 86.6%) on unseen WebGLM-QA. — [arXiv](https://arxiv.org/html/2508.17137v1)
- **SIEVE is Simpler than LRU** (eviction algorithm). Yazhuo Zhang, Ymir Vigfusson (**Emory, LEAD**) with Juncheng Yang (CMU, CO), Yao Yue and K.V. Rashmi. NSDI 2024.
  - Core idea: FIFO plus a "visited" bit and a moving hand. Hits take no lock, and the change is fewer than 20 lines in five production libraries.
  - Results: across 1,559 traces, up to 63.2% lower miss ratio than ARC, but only about 1.5% mean improvement over ARC. 2× throughput vs an optimized 16-thread LRU. — [USENIX](https://www.usenix.org/conference/nsdi24/presentation/zhang-yazhuo), [paper](https://junchengyang.com/publication/nsdi24-SIEVE.pdf), [code](https://github.com/cacheMon/NSDI24-SIEVE)
- Non-US papers the searches surfaced, so they are out of scope here but useful context:
  - Huang/Lin/Lee, "Efficient CPU-GPU Collaborative Inference for MoE-based LLMs on Memory-Limited Systems" (ASP-DAC 2026). GPU expert cache, misses computed on a multithreaded CPU, targets single-request consumer hardware. This is essentially pancake's design. Code: github.com/elsa-lab/MoE-CPU-GPU-Collaborative-Inference. — [arXiv](https://arxiv.org/abs/2512.16473)
  - PreScope/LayerScope (NUDT et al., ICS 2026): layer-aware learnable predictor plus prefetch-aware cross-layer scheduling plus async I/O. +141% throughput, −74.6% latency, but it is multi-batch. — [arXiv](https://arxiv.org/abs/2509.23638)
  - A search summary of these works says prefetching alone struggles at batch 1 because expert transfers take longer than compute. — [search context: MoBiLE](https://arxiv.org/html/2510.12357v1)

### Inferences
- **Diagnosis.** LRU (89.5%) beats pancake's windowed tier (79.6%) by about 10 points. The working set therefore drifts faster than the tier window adapts, so recency carries signal that frequency-over-window misses. The user already tested ClassLRU-style long-memory scoring and reuse-aware admission with no gain, which fits the view that the gain lies in *short-horizon recency*, not in long memory.
- **Recommendation 1: split the VRAM pool.** Keep a frequency-pinned "tier" segment and add a SIEVE- or LRU-managed "dynamic" segment, ARC-style, with an adaptive split. SIEVE fits a GPU-resident expert table because a hit only sets a bit, with no list relinking on the hot path.
- **The caveat.** A higher hit rate means more uploads. An LRU replay hit rate counts each miss as a load, and every miss costs a PCIe upload *on the critical path* unless the CPU computes it anyway. The replay must therefore model "promote on miss" bandwidth. My assumption: the 5700G is a Cezanne APU, whose GPU link is PCIe 3.0 x16 (~12-13 GB/s effective, not verified in this session). If true, that bound is decisive.
- **Prefetch budget, back of the envelope.** Assumptions, not verified for this model: about 48 layers, about 500 experts per layer, a 3-bit expert of roughly 2 MB. At 126 t/s each layer gets about 165 µs, enough to move about 2 MB over PCIe 3.0. So about one expert per layer per token can be prefetched in slack. Predictor precision matters more than recall.
- **Recommendation 2: cross-layer prediction (ExFlow/eMoE-style).** A per-domain table P(expert_{l+1} | expert_l) costs almost nothing. Use it to pick the single most likely missing expert per layer to upload while layer l computes. Applying layer l+1's router to layer l's hidden state is another option (a known trick from Pre-gated MoE / HOBBIT, which are non-US and not verified here).
- **Expected gain is modest.** I estimate +1 to 3 points of effective hit rate, unverified. The DraftExpert-style miss-aware windows the user tested also failed, a warning that prediction-driven uploads may not pay off here.
- **fMoE's semantic expert maps** confirm pancake's per-domain and per-conversation working sets. The new piece worth stealing: at conversation start, pick the stored expert map by prompt-embedding nearest neighbour rather than by domain label.
- **Expected gain from fMoE-style selection.** Mainly a faster warm-up in the first windows of a new conversation. Low to medium.
- **SiDA, MoE-Beyond and Read-ME** need training a predictor (SiDA, MoE-Beyond) or the model itself (Read-ME). MoE-Beyond's 17→70% headline is against a weak baseline at a 10% budget, while pancake caches about 44%. Low priority. Read-ME is not applicable to a fixed pretrained Qwen.

### Gaps
- Not verified this session, so treat these as candidates: "Towards MoE Deployment: Mitigating Inefficiencies in MoE Inference" (Meta with university co-authors), and any UIUC or Georgia Tech MoE expert-caching paper from 2025-26. None surfaced in my searches.
- No paper I found reports *batch-1 hit rate vs Belady* for a ~500-expert top-10 model. All headline numbers come from Mixtral, DeepSeek-V2-Lite or GPT-MoE with far fewer experts.

---

## Q2. Which CPU-side MoE execution or NVMe offload papers fit an AVX2 desktop CPU whose expert compute is RAM-bandwidth bound?

### Takeaway
Few US-university papers target an AVX2 desktop. The relevant ones are NoMAD-Attention (Rice; SIMD in-register lookups for attention, not experts) and DFloat11 (Rice; lossless BF16 compression, which does not apply to 2-3-bit GGUF). The core claim of the CPU-miss-compute design comes from a non-US paper (ASP-DAC 2026, listed in Q1): computing missed experts on a multithreaded CPU beats uploading them at batch 1. I found no US-university NVMe-offload paper for MoE at batch 1.

### Cited Findings
- **NoMAD-Attention: Efficient LLM Inference on CPUs Through Multiply-add-free Attention**. Tianyi Zhang, Jonah Wonkyu Yi, Bowen Yao, Zhaozhuo Xu, Anshumali Shrivastava, **Rice (LEAD; Zhaozhuo Xu is now at Stevens)**. NeurIPS 2024, arXiv 2403.01273.
  - Core idea: replaces attention multiply-adds with batched lookups held in SIMD registers. Needs no finetuning.
  - Results: up to 2× on 4-bit LLaMA-7B at 16k context.
  - Code: tonyzhang617/nomad-dist. — [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ccda3c632cc8590ee60ca5ba226a4c30-Abstract.html), [arXiv](https://arxiv.org/abs/2403.01273v1)
- **DFloat11: "70% Size, 100% Accuracy: Lossless LLM Compression for Efficient GPU Inference via Dynamic-Length Float"**. Shrivastava et al. (Rice is my belief; the snippets do not confirm the affiliation). NeurIPS 2025, arXiv 2504.11651.
  - Core idea: Huffman-codes the BF16 exponents, cutting size by about 30% with bit-identical outputs. A GPU kernel decompresses per transformer block.
  - Results: 2.3 to 46.2× vs CPU offload.
  - Code: github.com/LeanModels/DFloat11. — [arXiv](https://arxiv.org/abs/2504.11651), [NeurIPS](https://neurips.cc/virtual/2025/poster/115225)

### Inferences
- **NoMAD-Attention.** It relies on byte-shuffle lookups (pshufb), which AVX2 has. But pancake runs attention on the GPU, and only about a quarter of the layers are attention. Not applicable unless attention or KV moves to the CPU. Gain for pancake: about 0.
- **DFloat11.** It targets BF16 weights. 2-3-bit GGUF blocks have no exponent field and are already near-entropy, so the idea does not transfer. Gain: 0.
- **CPU expert compute is RAM-bandwidth bound** at roughly 28-42 GB/s, so the main levers are bytes moved per missed expert:
  1. Keep CPU-side copies of the hottest RAM-tier experts in the same 2-3-bit format. Never dequantize to a wider format in RAM.
  2. NUMA is not an issue on the 5700G.
  3. Within-expert neuron sparsity (the Deja Vu-style idea, Rice-led, ICML 2023, not re-verified here) could cut down-projection reads for CPU-computed experts. It needs a predictor, and quality risk at 2-3 bits is unknown. Speculative; I estimate up to 20-30% fewer bytes on CPU-computed experts, but that is unverified.
- **NVMe.** No US-university batch-1 MoE NVMe paper was found. With 57 GB of RAM and a ~125B model at 2-3 bits (~35-45 GB), NVMe should only hold a cold tail. Its main value is the cold start of a new domain working set.

### Gaps
- No US-university paper found on AVX2-specific low-bit MoE GEMV kernels. llama.cpp / ik_llama.cpp community work is the practical state of the art (outside academic scope).
- The SSD-offload literature I know of (Apple "LLM in a flash", PowerInfer-2) is not from the target universities.

---

## Q3. How does Lookahead/Jacobi-style decoding compare with MTP + n-gram drafting for code editing?

### Takeaway
UCSD's Hao AI Lab family covers three methods:
- Lookahead Decoding: training-free, uses an n-gram pool from Jacobi trajectories.
- CLLMs: fine-tune for Jacobi consistency.
- Jacobi Forcing: progressive distillation, up to 3.8 to 4.0× on code.

The two strong methods need *fine-tuning the target model*, which is infeasible for a 125B 2-3-bit GGUF. Training-free Lookahead's own gains are about 1.8× on chat, and up to 4× on code only with multi-GPU Lookahead Parallelism. Pancake's existing MTP head plus prompt-lookup already covers the same ground cheaply. Keep MTP + n-gram drafting.

### Cited Findings
- **Lookahead Decoding: "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"**. Yichao Fu, Peter Bailis, Ion Stoica, Hao Zhang. **UCSD Hao AI Lab (LEAD), Berkeley CO**. arXiv 2402.02057, Feb 2024. My search did not confirm the ICML 2024 venue.
  - Core idea: draft-free decoding. Generation is treated as a Jacobi fixed-point iteration, with a lookahead branch that generates n-grams and a verify branch, both in one forward pass, plus an n-gram pool.
  - Results: about 1.8× on MT-Bench; up to 4× on code completion with Lookahead Parallelism across 8 GPUs. — [arXiv](https://arxiv.org/abs/2402.02057)
- **CLLMs: Consistency Large Language Models**. Kou, Hu, ... Hao Zhang, Deng. **SJTU and UCSD co-led (UCSD = LEAD/co-first)**. ICML 2024, arXiv 2403.00835.
  - Core idea: fine-tune so that Jacobi iteration converges in fewer steps.
  - Results: 2.4 to 3.4× (3.4× on Spider; 2.4× on ShareGPT with MT-bench 6.4), from about 1M training tokens for LLaMA-7B. Vanilla Jacobi gets only about 1.05×. — [PMLR](https://proceedings.mlr.press/v235/kou24a.html), [arXiv](https://arxiv.org/html/2403.00835v4)
- **Jacobi Forcing: "Fast and Accurate Causal Parallel Decoding using Jacobi Forcing"**. **UCSD Hao AI Lab (LEAD)** with SJTU and Snowflake. arXiv 2512.14681, Dec 2025.
  - Core idea: the model is trained on its own parallel-decoding trajectories.
  - Results: 3.8× wall-clock on coding benchmarks. Multi-block decoding with rejection recycling gives up to 4.5× more accepted tokens per iteration and about 4.0× wall-clock. Benchmarks: HumanEval, MBPP, GSM8K, MATH.
  - Code: github.com/hao-ai-lab/JacobiForcing. — [arXiv](https://arxiv.org/html/2512.14681), [blog](https://hao-ai-lab.github.io/blogs/jacobi-forcing)

### Inferences
- **Why offloaded MoE changes the math.** Verifying k candidate tokens or n-grams per step activates the *union* of their experts: up to k×10 routed experts per layer instead of 10. Misses then multiply, and so does CPU expert compute. Wide Lookahead (W×N guesses per step) is therefore much worse for pancake than for dense GPU-resident models. Narrow, high-acceptance drafting wins: MTP (one or two tokens) plus prompt-lookup for verbatim copy spans in code edits.
- **CLLM / Jacobi Forcing.** They need fine-tuning the 125B MoE (and re-quantizing). Out of reach on this hardware. Gain is 0 unless someone publishes a Jacobi-Forcing Qwen3.x checkpoint.
- **The one cheap idea to borrow from Lookahead:** an n-gram pool built from the model's own past outputs in the conversation and earlier sessions (suffix-tree style). Use it as an extra draft source when prompt-lookup finds nothing, gated by expected acceptance. Code-agent sessions repeat boilerplate heavily.
- **Expected gain.** Small, from a few % up to about 10% decode throughput on repetitive edit turns. That is my inference, not measured.
- **Tree drafting.** If drafting is widened at all, prefer drafts whose tokens route to *already cached* experts. "Expert-aware draft acceptance" is my idea and not from a paper.

### Gaps
- My searches did not reach Hao lab's "Lookahead Reasoning" (2025) or Dynasor / Certaindex (reasoning-budget control, relevant to thinking budgets). Both are candidates; not verified this session.
- No paper measures speculative or parallel decoding on *offloaded* MoE at batch 1 with a union-of-experts cost model.

---

## Q4. Which KV compression/reuse and agent-efficiency papers are worth adopting? (plus the broadened scope: learned caching and systems)

### Takeaway
On the hybrid Gated-DeltaNet model, KV compression has little payoff: only the attention layers keep a KV cache, and the DeltaNet state is constant-size. The valuable ideas are *state/prefix reuse and scheduling around tool calls*:
- InferCept (UCSD): what to do with a paused request's state during a tool call.
- KVFlow (UCSD): predictive prefetch of cached prefixes for the next agent step, beating LRU.
- Prompt Cache (Yale): modular reuse. It does *not* carry over to DeltaNet layers, because recurrent state cannot be spliced out of order.

### Cited Findings
- **InferCept: Efficient Intercept Support for Augmented LLM Inference**. Yiying Zhang group, **UCSD (LEAD)**. ICML 2024, arXiv 2402.01869.
  - Core idea: when a request pauses for a tool call, the scheduler chooses between Discard (recompute later, the vLLM default), Preserve (keep the KV on the GPU) and Swap (move it to CPU memory), using GPU-memory-waste equations.
  - Results: more than 2× completed requests per second on A100s.
  - Later critique: Continuum (arXiv 2511.02230, affiliation not verified) argues InferCept assumes predictable tool calls and proposes TTL-based KV pinning. — [arXiv](https://arxiv.org/pdf/2402.01869), [Continuum](https://arxiv.org/html/2511.02230v2)
- **KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows**. Zaifeng Pan, ... Yufei Ding, **UCSD (LEAD)**, with AWS co-authors. NeurIPS 2025, arXiv 2507.07400.
  - Core idea: an Agent Step Graph gives each agent a steps-to-execution value. Eviction is by that value instead of LRU, and the next step's KV is prefetched from CPU to GPU in background threads.
  - Results: up to 1.83× speedup on single workflows and 2.19× on concurrent workflows vs SGLang's hierarchical radix cache. — [arXiv](https://arxiv.org/html/2507.07400v1), [NeurIPS](https://papers.neurips.cc/paper_files/paper/2025/hash/b7971d31a7d5eb0f1eed2f8f6f368195-Abstract-Conference.html)
- **Prompt Cache: Modular Attention Reuse for Low-Latency Inference**. In Gim, Lin Zhong et al., **Yale (LEAD)** with Google. MLSys 2024, arXiv 2311.04934.
  - Core idea: precomputes attention states for reusable "prompt modules" (system prompt, templates, documents), declared by a schema that keeps positions consistent.
  - Results: about 8× TTFT reduction on GPU and about 60× on CPU.
  - Code: yale-sys/prompt-cache. — [arXiv](https://arxiv.org/abs/2311.04934), [Google Research](https://research.google/pubs/prompt-cache-modular-attention-reuse-for-low-latency-inference/)
- **PyramidKV**: per-layer KV budget that is large in lower layers and small in upper layers. Matches full-cache LongBench while keeping 12% of the KV. The search did not confirm that Zefan Cai and Junjie Hu are at UW-Madison. — [arXiv](https://arxiv.org/pdf/2406.02069)
- **SnapKV**: an observation window at the end of the prompt picks important KV positions per head. My search did not confirm the UIUC affiliation. — [context in PyramidKV/KV survey results](https://arxiv.org/pdf/2406.02069)
- **Broadened scope: learned or simple caching.** SIEVE (Emory, Q1) is the one cache-policy paper I verified that falls in the window.

### Inferences
- **InferCept for single-user opencode.** Always "Preserve" across tool calls: there is no contention. The useful piece is a pancake-specific variant: at the tool-call boundary, snapshot the DeltaNet recurrent state plus the attention KV (cheap, constant plus small). When the tool result returns, only the tool output needs prefill.
- **Continuum's TTL idea** maps to "keep the conversation-cache entry hot for N seconds after a tool call, then demote to RAM/NVMe".
- **Expected gain:** fewer re-prefills on long agent loops. Pancake's conversation cache may already do most of this, so check the snapshot granularity.
- **KVFlow's idea transfers to *expert* caching.** In an agent loop the next step is predictable: a tool result is coming, and its content type is known (file read, grep, test output). So *pre-stage the expert working set for the expected prefill domain* (code or logs) while the tool runs. The GPU is idle during tool execution, which leaves free PCIe time to reshape the VRAM tier. I rate this the most promising agent-side idea, my inference: medium gain on prefill of tool outputs, plus better hit rates in the first decode tokens after a tool result.
- **Prompt Cache.** Non-prefix modular reuse is impossible for Gated DeltaNet layers, whose state depends on the entire prefix. Only exact-prefix reuse (already in the conversation cache) works. Keep the system prompt and tool schema byte-stable at the front so the prefix always hits.
- **SnapKV / PyramidKV / KV quantization.** With only about a quarter of the layers holding KV, and TurboQuant KV already tested with no gain, expect little benefit unless contexts exceed about 100k tokens. Low priority.

### Gaps
- Candidates I know but did not verify this session:
  - UIUC CodeAct and Agentless (coding-agent efficiency and reliability).
  - UChicago CacheGen / CacheBlend (KV compression and non-prefix RAG reuse; not applicable to DeltaNet, for the same reason as Prompt Cache).
  - Yale "Asynchronous LLM Function Calling" (AsyncLM).
  - UT Austin H2O / R-Sparse.
  - Rice KIVI / Coupled Quantization.
  - Georgia Tech GEAR.
  - UMich LTE (learned activation sparsity).
- On the broadened scope: I did not find in-window US-university learned cache-replacement or hardware-prefetcher papers (UT Austin's Hawkeye, Glider and Mockingjay predate Nov 2022). Their Belady-imitation principle is still relevant: train a per-expert "cache-friendly vs averse" classifier offline on pancake's own Belady replay traces. That is a direct way to target the gap from 89.5% to 94.7%. This last idea is my inference, not a cited result.
- GPU compiler and autotuning papers from the target universities were not surveyed for lack of budget.
