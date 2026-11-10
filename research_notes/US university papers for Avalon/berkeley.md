# UC Berkeley AI papers (Nov 2022 - Nov 2026) useful for pancake / Avalon / opencode

Scope and method. The main source list was the Sky Computing Lab publication page (DBLP export, 2023-2026) ([sky.cs.berkeley.edu/publications](https://sky.cs.berkeley.edu/publications/)), plus the SqueezeAILab (Keutzer/Gholami) repo list (via `gh api orgs/SqueezeAILab/repos`). Every arXiv entry below was checked against its arXiv abstract (export API, 2026-11-09). Two papers, FreeToken and the MoE intra-expert sparsity paper, were read in full HTML, as were the ADRS paper ("Barbarians at the Gate") and the speculative-decoding study. The scope was broadened during the task, so systems, eval and agent work is included as well as LLM papers.

**Lead flag.** "Lead" means Berkeley holds the first author and the group. "Co" means Berkeley authors are present but another institution leads. Affiliations come from the authors' known groups (Sky Lab, SqueezeAILab, BAIR) and are marked "inferred" where the paper text was not read.

**Exclusions.** These were skipped because earlier notes already cover them:
- In the Stanford report: SGLang/RadixAttention (Stanford-led per that report), Hydragen, FlexGen, s1, DSPy/MIPRO.
- In the MIT/Harvard notes: Quest, DuoAttention, LServe, Hydra, Medusa context, Lookahead (mentioned only).
- In the local pancake notes (`/home/dan/clone-of-d/src/Strata/research_notes/Pancake MoE engine speedup/`): FreeToken appears only as a SeqMoE baseline there. Fiddler, MoE-Infinity and HybriMoE are not Berkeley.

**Box facts used for the verdicts** (from `docs/PANCAKE_FINDINGS.md` and the earlier notes):
- Decode is roughly half CPU-computed missed experts, at about 33.4 GB/s, near the dual-channel DDR4 limit. "They can only go faster by reading fewer bytes."
- The 5700G gives PCIe 3.0.
- An IQ2_XS expert is about 1.4 MB (1,000 experts ≈ 1.4 GB).
- KV is already q8. HIP graphs, unroll, Vulkan, TurboQuant KV, L3 prefetch and ClassLRU scoring were already tested with no gain.

## Q1. Which Berkeley ideas could make pancake/Avalon faster for one user on one consumer AMD GPU with offloaded experts?

### Takeaway
Two recent Berkeley-led papers hit our real bottleneck, the CPU expert bytes. The first is **intra-expert activation sparsity** (Park et al., 2026): mask SwiGLU neurons by |gate activation| and skip their up/down rows, at 85-87% sparsity on Qwen3.5 MoEs. The second is **FreeToken** (Shuo Yang et al., 2026). It splits each step's cache misses between the CPU and concurrent PCIe fills by a closed-form bandwidth ratio, and keeps recurrent-state checkpoints at tool-call and thinking boundaries. Neither has been tried on this box. The sparsity idea is the biggest candidate: a rough upper bound of 1.2-1.4x decode, my estimate. Everything else from Berkeley (vLLM/Jenga/Pie/NEO/MoE-Lightning, sparse attention, self-speculation) either targets batched datacenter serving or a bottleneck we do not have.

### Cited Findings
**MoE / expert offload (the hot path)**
- **Uncovering Intra-expert Activation Sparsity for Efficient MoE Model Execution.** Jongseok Park, Sunga Kim, Zhenyu Gu, Ion Stoica, Alvin Cheung. Berkeley Sky lead. arXiv 2605.08575, 9 May 2026.
  - Pre-trained MoEs show up to 90% sparsity inside each expert "without any modification to the activation function or model parameters". In vLLM this gives up to 2.5x on the MoE layer and 1.2x end to end. — [arXiv 2605.08575](https://arxiv.org/abs/2605.08575)
  - Mechanism: the gate projection is computed densely. Neurons with |activation| under a threshold are masked, and that threshold comes from a sparsity-to-threshold lookup table built at engine start (no runtime top-k). Up and down are computed only for the gathered active neurons, using a fused Triton kernel over 64-neuron tiles. Shared-expert neurons must stay fully active. — [arXiv HTML](https://arxiv.org/html/2605.08575)
  - Accuracy cutoffs (sparsity that keeps 95% of the benchmark average): Qwen3.5-35B-A3B 84.5%, Qwen3.5-122B-A10B 87.4%, Llama-4-Maverick 90.8%, Granite-1B 26.5%. On Qwen3.5-35B-A3B the average was 70.9 dense and 67.5 at 85%.
  - Speedups: MoE layer 2.5x on an RTX 4090 (Qwen3.5-122B), 2.0x on H200, 1.8x on MI355X. The win is smallest at small batch, where fixed overheads dominate; the paper studies GPU execution only, not CPU. No code URL; the source is in the appendix. — [arXiv HTML](https://arxiv.org/html/2605.08575)
- **FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution.** Shuo Yang, Xiaoze Fan, Melissa Pan, Haocheng Xi, Zhe Wang, Shanlin Sun, Kurt Keutzer, Song Han, Matei Zaharia, Chenfeng Xu, +1. Berkeley lead (Sky list; MIT co-author). arXiv 2608.16157, 17 Aug 2026. Code: [github.com/FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken). — [arXiv 2608.16157](https://arxiv.org/abs/2608.16157)
  - The full expert pool lives in host RAM. Leftover VRAM is one global LRU expert cache across all layers.
  - For the m missed experts in a layer, q* ≈ m·B_P/B_H of them are copied over PCIe and run on the GPU, while the rest run on the CPU concurrently. B_P and B_H are bandwidths profiled on the machine, and at least one fill always happens. The CPU branch is launched first, and partial outputs are merged exactly.
  - Agent state reuse: a radix prefix tree for KV, plus a "semantic-aware" recurrent-state cache. Its checkpoints sit at special-token boundaries (thinking segments, tool calls and outputs, turn boundaries); after a context edit the engine restores the deepest surviving checkpoint.
  - Prefill double-buffers a full layer of experts: while layer l computes, layer l+1 streams in. Turning this off costs 19-26%.
  - Results: Qwen3.6-35B-A3B (GDN hybrid) 77-83 t/s on an RTX 5090. That is 1.3-2.1x the best baseline (llama.cpp, KTransformers, Ollama, MoE-Infinity) on five consumer systems. Miss rates at 5090 capacity on Qwen3.6: FreeToken 16%, KTransformers 41%, llama.cpp 62%. Worst-turn TTFT under 44 s, against llama.cpp at 232 s. No AMD GPU and no GGUF were tested. — [arXiv HTML](https://arxiv.org/html/2608.16157)
- **MoE-Lightning.** Shiyi Cao, Shu Liu, Tyler Griggs, Peter Schafhalter, Xiaoxuan Liu, Ying Sheng, Joseph Gonzalez, Matei Zaharia, Ion Stoica. Berkeley lead, ASPLOS 2025, arXiv 2411.11217. CPU-GPU-I/O pipelining (CGOPipe) with paged weights, plus a Hierarchical Roofline Model (HRM) used to choose policies. Up to 10.3x throughput over offloading systems for Mixtral 8x7B on a T4. This is batch throughput. — [arXiv 2411.11217](https://arxiv.org/abs/2411.11217)
- **NEO.** Xuanlin Jiang (PKU), Yang Zhou, Shiyi Cao, Ion Stoica, Minlan Yu. Berkeley co-author, MLSys 2025, arXiv 2411.01142. Offloads part of attention plus KV to the CPU to grow the GPU batch. Throughput gains at the same latency on T4/A10G/H100, up to 79.3% on an A10G with a stronger CPU. — [arXiv 2411.01142](https://arxiv.org/abs/2411.01142)
- **Pie: Pooling CPU Memory for LLM Inference.** Yi Xu, Ziming Mao, Xiangxi Mo, Shu Liu, Ion Stoica. Berkeley lead, arXiv 2411.09317. "Performance-transparent swapping" built on predictable access patterns, on GH200. Up to 1.9x throughput over vLLM and 9.4x over FlexGen. — [arXiv 2411.09317](https://arxiv.org/abs/2411.09317)

**Speculative decoding**
- **Speculative Decoding: Performance or Illusion?** Xiaoxuan Liu, Jiaxiang Yu, Jongseok Park, Ion Stoica, Alvin Cheung. Berkeley lead, arXiv 2601.11580, Dec 2025. — [arXiv 2601.11580](https://arxiv.org/abs/2601.11580); details from the [HTML](https://arxiv.org/html/2601.11580)
  - First production-engine study of SD (vLLM). Verification takes 42-95% of execution time.
  - A single MTP head reused autoregressively (GLM-4.5-Air) has per-position acceptance falling 0.92 → 0.68 → 0.38.
  - n-gram acceptance is heavy-tailed and bursty, and n-gram beats EAGLE when prompt/output overlap is high (BLEU-4 > ~0.6, i.e. code editing).
  - An oracle that picks between n-gram and EAGLE per position would give up to 2.2x over the best fixed strategy. That is an upper bound that assumes a perfect predictor.
  - An oracle proposal length gives ~2.75x versus 2.1x for the best fixed k on InstructCoder.
- **TurboSpec (earlier title: Optimizing SD for serving using goodput).** Xiaoxuan Liu, Jongseok Park, et al. Berkeley lead, arXiv 2406.14066. Profiles the machine and uses feedback control to pick the speculation amount that maximizes goodput online. — [arXiv 2406.14066](https://arxiv.org/abs/2406.14066)
- **Online Speculative Decoding.** Xiaoxuan Liu et al. Berkeley lead, ICML 2024, arXiv 2310.07177. Keeps distilling the draft on the live query distribution. Acceptance +0.1 to +0.65, latency 1.42-2.17x lower. Code: LiuXiaoxuanPKU/OSD. — [arXiv 2310.07177](https://arxiv.org/abs/2310.07177)
- **SparseSpec.** Yilong Zhao et al. Berkeley lead (inferred; Tsinghua and MIT co-authors), arXiv 2512.01278. Self-speculation in which the draft is the same model running sparse attention (PillarAttn); also delayed verification for CPU/GPU overlap. Up to 2.13x throughput on long reasoning. — [arXiv 2512.01278](https://arxiv.org/abs/2512.01278)
- **QuantSpec.** Rishabh Tiwari et al. (SqueezeAILab with Apple). Berkeley lead, arXiv 2502.10424. The draft is the same model with a hierarchical 4-bit KV and 4-bit weights. Acceptance above 90%, up to ~2.5x end to end on long context. — [arXiv 2502.10424](https://arxiv.org/abs/2502.10424)
- **Lookahead Decoding.** Yichao Fu, Peter Bailis, Ion Stoica, Hao Zhang. UCSD/LMSYS lead, Berkeley co-author, ICML 2024. Draft-free Jacobi/n-gram decoding, up to 1.8x on MT-bench. — [arXiv 2402.02057](https://arxiv.org/abs/2402.02057)

**Sparse attention and long-context KV** (all Berkeley lead unless noted)
- **HashAttention** (Desai et al., ICML 2025). Learned Hamming-space keys at 32 bits per token. 16x fewer tokens with minimal loss, attention latency up to 4.3x lower in GPT-FAST. — [arXiv 2412.14468](https://arxiv.org/abs/2412.14468)
- **vAttention: Verified Sparse Attention** (Desai et al., ICLR 2026). Combines top-k with sampling to give a user-set (ε, δ) guarantee. Matches full quality at up to 20x sparsity. — [arXiv 2510.05688](https://arxiv.org/abs/2510.05688)
- **Twilight** (Chaofan Lin, Tsinghua lead; Stoica and Song Han co-authors). Top-p adaptive budget on top of any sparse-attention method. Prunes up to 98% of tokens, 15.4x on self-attention and 3.9x end-to-end per-token latency at long context. — [arXiv 2502.02770](https://arxiv.org/abs/2502.02770)
- **Double Sparsity** (Shuo Yang, Ying Sheng, Gonzalez, Stoica, Lianmin Zheng). Static channel sparsity chosen offline to find important tokens. 14.1x attention speedup, 1.9x end to end, and 16.3x decode with offloading at 256K. Code: andy-yang-1/DoubleSparse. — [arXiv 2408.07092](https://arxiv.org/abs/2408.07092)
- **Squeezed Attention** (Hooper et al., ACL 2025). Offline k-means over the keys of a fixed context, so that only relevant keys are attended. 3.1x KV-budget cut with no visible loss on LongBench, and more than 4x speedups in prefill and decode. — [arXiv 2411.09688](https://arxiv.org/abs/2411.09688)
- **Multipole Attention** (Hooper et al., NeurIPS 2025). Centroids both select important keys and approximate the rest, with fast re-clustering of generated tokens. Up to 4.5x attention speedup for long reasoning (Qwen3-8B). — [arXiv 2506.13059](https://arxiv.org/abs/2506.13059)
- **Inference Time Context Sparsity: Illusion or Opportunity?** (Joshi et al.; Gonzalez, Stoica, Desai co-authors; Rice/Berkeley). A position paper: 20 models across five families are "remarkably robust" to context sparsity without having been trained for it. — [arXiv 2605.24168](https://arxiv.org/abs/2605.24168)
- **KVQuant** (Hooper et al., NeurIPS 2024). Per-channel keys quantized pre-RoPE, non-uniform datatypes and per-vector outliers. Under 0.1 PPL loss at 3 bit, with kernels ~1.7x faster than fp16 matvec. — [arXiv 2401.18079](https://arxiv.org/abs/2401.18079)
- **XQuant** (Tomar et al., 2025). Caches quantized layer inputs X and rematerializes K and V, for 7.7x memory savings at under 0.1 PPL. XQuant-CL reaches 10-12.5x using cross-layer similarity. — [arXiv 2508.10395](https://arxiv.org/abs/2508.10395)
- **Jenga** (Chen Zhang et al., vLLM team, SOSP 2025). A two-level allocator (LCM of embedding sizes) with per-layer-type caching and eviction APIs for heterogeneous and hybrid models. Up to 79.6% better memory use and 4.92x throughput. — [arXiv 2503.18292](https://arxiv.org/abs/2503.18292)
- **PagedAttention / vLLM** (Kwon et al., SOSP 2023). 2-4x throughput through paged KV. — [arXiv 2309.06180](https://arxiv.org/abs/2309.06180)
- **S-LoRA** (Sheng et al., MLSys 2024). Unified paging for adapters and KV. — [arXiv 2311.03285](https://arxiv.org/abs/2311.03285)

**Weight quantization**
- **SqueezeLLM** (Kim, Hooper, Gholami et al., ICML 2024). Sensitivity-based non-uniform quantization plus dense-and-sparse outliers. At 3 bit, the perplexity gap is up to 2.1x smaller than other methods; up to 2.3x speedup on an A6000. The paper also states that single-batch inference is memory-bandwidth bound. — [arXiv 2306.07629](https://arxiv.org/abs/2306.07629)

**Kernels, compilers and runtime**
- **Qrita** (Park, Kim, Cheung, Stoica, 2026). A pivot-based top-k/top-p sampler and now vLLM's default Triton sampler. Up to 1.4x serving throughput at large batch. — [arXiv 2602.01518](https://arxiv.org/abs/2602.01518)
- **K-Search** (Shiyi Cao, Ziming Mao, Gonzalez, Stoica, 2026). LLM kernel generation guided by a co-evolving world model. — [arXiv 2602.19128](https://arxiv.org/abs/2602.19128)
- **Foundry** (Michigan lead, Stoica co-author). Persists CUDA-graph context to cut cold start. — [arXiv 2604.06664](https://arxiv.org/abs/2604.06664)

### Inferences
- **Intra-expert sparsity on CPU experts. Highest-value new idea. Not tried here, and not in the local pancake research notes.**
  - Estimated gain: the gate projection stays dense (about 1/3 of an expert's bytes). At 85% sparsity, up and down drop to about 15%. Bytes per CPU expert fall to about 0.33 + 0.67×0.15 ≈ 0.43 of today's.
  - With CPU experts at about 50% of decode, the upper bound is 1/(0.5 + 0.5×0.43) ≈ 1.4x decode. A realistic figure is 1.15-1.3x after thresholding, gather overhead and accuracy margins. All of this is my estimate.
  - Caveat 1, layout: GGUF IQ blocks run along the input dimension. Skipping up rows (one row per neuron) is natural. Skipping down-projection columns is not, because down's input dimension is the neuron dimension. pancake would have to repack down weights neuron-major at load time (requantize the blocks along the hidden dimension) to skip at neuron granularity.
  - Caveat 2, accuracy: it is unmeasured at 2-3-bit and on Qwen3.8-Flash-Next. The paper's closest models are Qwen3.5-122B-A10B (87.4% cutoff) and Qwen3.5-35B-A3B.
  - Caveat 3, scope: the GPU-side small-batch gain was weak in the paper, so apply this to CPU experts only, where bytes rule.
  - Same family as CATS (Stanford report, item estimated at +20-30% of CPU expert time). This paper is the first evidence on real MoEs, including Qwen.
- **FreeToken's miss split.** It is worth checking whether pancake already overlaps "PCIe-fill and run on GPU" with "CPU compute" inside the same step. Strata's findings mention "hybrid CPU/GPU experts, swaps", but this is not confirmed to be a per-step q* split.
  - With B_P ≈ 12 GB/s (PCIe 3.0) and B_H ≈ 33 GB/s, q* ≈ 0.36m. In the ideal case CPU-miss time falls by about B_P/(B_P+B_H) ≈ 27%, or roughly 10-13% of decode. My estimate.
  - DMA reads come from the same DDR4 the CPU is saturating, so the real gain is smaller. Measure it before building.
  - Its "checkpoint at tool-call/thinking boundaries" point directly extends pancake's message-boundary checkpoints. Low effort.
  - It is also a ready, open-source reference implementation of a GDN-hybrid agent cache.
- **Speculative decoding.**
  - The MTP decay curve (0.92/0.68/0.38) says draft depth beyond 2-3 rarely pays.
  - The paper's strongest lever for us is a per-step choice between prompt lookup and MTP, plus TurboSpec-style online control of draft length. On code edits with high prompt overlap, n-gram drafts should be allowed to run long (5+), while MTP drafts stay short. Expected gain 0-10%; the oracle numbers are upper bounds.
  - Online SD supports the self-trained MTP or draft-layer idea already noted from StrataFast.
  - Not applicable: SparseSpec, QuantSpec and Lookahead. Sparse-attention or 4-bit drafts do not reduce expert reads, and our weights are already 2-3 bit.
- **Sparse attention.** This only matters if full-attention KV reads grow large at 131-262k.
  - Rough check, assuming Qwen3-Next-like shapes (12 full-attention layers, 2 KV heads × 256, q8 KV): 12 × 2 × 2 × 256 × 1 B ≈ 12 KB per token, so about 2.5 GB read per step at 200k, or ~3 ms per token on the 7900 XT (~800 GB/s). That is a noticeable share of the ~8 ms/token at 126 t/s. My estimate; measure decode t/s at 10k against 200k.
  - If the drop is real, Twilight's top-p budget (Tsinghua lead) or vAttention's verified top-k plus sampling are the safest add-ons.
  - Squeezed Attention fits the opencode case well: the system prompt and tool schemas form a long fixed prefix. With only about 12 attention layers, though, the gain is bounded.
- **KV quantization.** KVQuant/XQuant are below priority: KV is already q8 and TurboQuant KV showed no speed gain. The only indirect lever is to free VRAM for expert slots. 4-bit KV at 262k frees about 1.5 GB, roughly 1,000 more IQ2_XS expert slots (my estimate), which raises the hit rate rather than attention speed.
- **Not useful for one user:** Jenga, Pie, NEO, MoE-Lightning, S-LoRA, PagedAttention and Qrita. They are batch or memory-capacity techniques; sampling is under 1.7% of time per the SD study. MoE-Lightning's Hierarchical Roofline Model is still a handy analytic tool for choosing VRAM/RAM/thread splits.

### Gaps
- Neither FreeToken nor the intra-expert sparsity paper tests AMD consumer GPUs, GGUF or CPU-side sparse kernels. The sparsity paper's code was not located (it is "in appendix B").
- I could not confirm from pancake's source whether the per-step concurrent fill-plus-CPU split already exists. Grep found swap and expert-cache logic but no explicit q* rule.
- Exact attention-layer and KV-head counts of Qwen3.8-Flash-Next were not verified, so the KV-read estimate is shape-assumed.

## Q2. Which KV/prefix-cache, quantization or decoding ideas are not already in pancake?

### Takeaway
New and portable: neuron-level sparsity in CPU experts, recurrent-state checkpoints at tool-call and thinking boundaries, adaptive per-step drafter and draft-length control, and idleness-ranked tiering of parked conversations (MORI, Continuum). Already present or not worth it: paged KV, prefix caching, MTP/n-gram drafting, KV quantization for speed, and SqueezeLLM-style non-uniform weight quantization (GGUF IQ quants with imatrix already sit in that space).

### Cited Findings
- **MORI.** Tian Xia, Hanchen Li, Zhifei Li, Xiaokun Chen, Hao Kang, Yifan Qiao, Yi Xu, Ion Stoica. Berkeley lead, arXiv 2606.00866. Agent sessions alternate busy phases (short tool calls) with idle phases (long tool calls), and LRU misses this. MORI ranks programs by continuous "idleness", keeping the busiest in HBM and the most idle in DRAM. On Claude Code traces: 20-71% higher throughput and 18-43% lower TTFT than the best offloading baseline. — [arXiv 2606.00866](https://arxiv.org/abs/2606.00866)
- **Continuum.** Hanchen Li et al. Berkeley lead, arXiv 2511.02230. Pins a request's KV in GPU memory with a time-to-live after it emits a tool call. The TTL comes from the reload cost and the queueing delay. Average job completion time is more than 8x better on SWE-Bench, BFCL and OpenHands traces. — [arXiv 2511.02230](https://arxiv.org/abs/2511.02230)
- **Jenga.** Exposes per-layer-type cache and eviction logic, the layer-specific prefix rules that hybrid (recurrent plus attention) models need. — [arXiv 2503.18292](https://arxiv.org/abs/2503.18292)
- **FreeToken.** Places recurrent-state checkpoints at thinking, tool-call, tool-output and turn boundaries, evicts them by LRU separately from KV, and restores the deepest surviving one after a context edit. — [arXiv HTML 2608.16157](https://arxiv.org/html/2608.16157)
- **PrefillOnly** (Kuntai Du et al., SOSP 2025; Berkeley/UChicago). For single-output-token jobs, keeps only the last layer's KV. — [arXiv 2505.07203](https://arxiv.org/abs/2505.07203)
- **vCache** (Schroeder et al.; Berkeley/TUM). A semantic response cache with per-entry learned thresholds and error guarantees: up to 12.5x more hits and 26x lower error than static thresholds. — [arXiv 2502.03771](https://arxiv.org/abs/2502.03771)
- **Decoding control.** TurboSpec (closed-loop draft amount) — [arXiv 2406.14066](https://arxiv.org/abs/2406.14066). Per-position proposer choice is an open opportunity worth up to 2.2x at the oracle — [arXiv HTML 2601.11580](https://arxiv.org/html/2601.11580).
- **Quantization.** KVQuant (pre-RoPE per-channel keys) — [arXiv 2401.18079](https://arxiv.org/abs/2401.18079). SqueezeLLM (sensitivity-weighted non-uniform plus sparse outliers) — [arXiv 2306.07629](https://arxiv.org/abs/2306.07629). The SqueezeAILab repos for both are still maintained, with updates in Oct 2026 per `gh api orgs/SqueezeAILab/repos`.

### Inferences
- **New for pancake, in priority order:**
  - (1) Intra-expert CPU sparsity, from Q1.
  - (2) Extra checkpoint triggers at `</think>`, tool_call end and tool-result end. opencode edits context exactly there, through compaction and tool-output pruning, so a restore after an edit re-prefills less. Low effort; the gain is seconds per affected turn.
  - (3) Adaptive drafter: per-step choice between prompt lookup and MTP, with draft length set from a running goodput estimate. Low-medium effort, 0-10%.
  - (4) MORI/Continuum-style policy for parked conversations. For one user with opencode sub-agents, keep the parked state of whichever session is "least idle" in VRAM or pinned RAM, and push idle ones to NVMe on a TTL. This helps sub-agent switches, not steady-state decode.
- **Already covered or not worth it:**
  - PagedAttention and prefix caching: pancake already has a parked cache with checkpoints.
  - vCache: semantic response reuse is unsafe for coding turns.
  - PrefillOnly: our workload generates long outputs.
  - KVQuant/XQuant: KV is q8 and TurboQuant showed no gain. XQuant's rematerialization trades compute we do not have spare during prefill.
  - SqueezeLLM: GGUF IQ/imatrix already do non-uniform, importance-weighted quantization. The one transferable idea is dense-and-sparse outlier storage for the most sensitive experts or tensors; that is speculative and high effort.

### Gaps
- Continuum, MORI and Jenga are evaluated multi-tenant. No single-user numbers exist, so the benefit for pancake is inferred.
- No Berkeley paper studies prefix caching for Gated DeltaNet specifically, beyond FreeToken's checkpoints and Jenga's APIs.

## Q3. Which agent/tool-use/prompt-optimization work would make opencode turns faster or more reliable?

### Takeaway
The best levers are behavioral, not kernel-level:
- Cut overthinking and long thinking in agent loops (NoThinking, Danger of Overthinking).
- Make thinking-budget cutoffs graceful (Interruptible LRMs).
- Overlap decoding with tool execution (AsyncFC, 1.4x end to end on coding).
- Use GEPA offline to tune opencode's prompts with Avalon or pancake as the reflection LM.
- Use BFCL v4 multi-turn plus LiveCodeBench as regression gates for quantization levels and tool-call recovery.

### Cited Findings
- **The Danger of Overthinking** (Cuadron et al., Berkeley lead, arXiv 2502.08235). 4,018 SWE-Bench Verified trajectories show three patterns: analysis paralysis, rogue actions and premature disengagement. Higher overthinking correlates with lower success. Picking the lower-overthinking solution improves performance by almost 30% and cuts compute by 43%. — [arXiv 2502.08235](https://arxiv.org/abs/2502.08235)
- **Reasoning Models Can Be Effective Without Thinking** (Wenjie Ma, Jingxuan He, Charlie Snell, Tyler Griggs, Sewon Min, Matei Zaharia; Berkeley lead). Skipping thinking by prompt beats Thinking at equal token budgets (51.3 vs 28.9 on ACM23 at 700 tokens), and parallel NoThinking matches Thinking at up to 9x lower latency. — [arXiv 2504.09858](https://arxiv.org/abs/2504.09858)
- **Are Large Reasoning Models Interruptible?** (Tsung-Han Wu, ..., Trevor Darrell, Joseph Gonzalez; Berkeley lead). Under budget cutoffs, accuracy drops up to 60% when updates arrive late. The failure modes are "reasoning leakage" (reasoning moved into the answer), "panic" (reasoning abandoned) and "self-doubt". — [arXiv 2510.11713](https://arxiv.org/abs/2510.11713)
- **AsyncFC** (Guangyu Feng, Huanzhi Mao, Prabal Dutta, Joseph Gonzalez; Berkeley lead, arXiv 2605.15077). An execution-layer framework: decoding continues while functions run, using symbolic futures, and independent calls run in parallel, with no fine-tuning and the standard FC protocol. 1.4x end to end on coding tasks at the same accuracy. — [arXiv 2605.15077](https://arxiv.org/abs/2605.15077)
- **LLMCompiler** (Sehoon Kim et al., SqueezeAILab, ICML 2024). A planner, task-fetching unit and executor run function calls in parallel. Up to 3.7x latency reduction, 6.7x lower cost and ~9% higher accuracy than ReAct. Code: SqueezeAILab/LLMCompiler (1.9k stars, active Nov 2026). — [arXiv 2312.04511](https://arxiv.org/abs/2312.04511)
- **GEPA** (Lakshya A Agrawal et al., Berkeley lead with Stanford/Databricks co-authors, arXiv 2507.19457). A reflective Genetic-Pareto prompt optimizer that reads trajectories, including tool calls and outputs. It beats GRPO by 6% on average (up to 20%) with up to 35x fewer rollouts, and beats MIPROv2 by more than 10%. Code: [gepa-ai/gepa](https://github.com/gepa-ai/gepa). — [arXiv 2507.19457](https://arxiv.org/abs/2507.19457)
- **BFCL** (Patil et al., ICML 2025). The current version is V4 ("holistic agentic evaluation", with web search). V3 added multi-turn. It scores native FC and prompt-mode models separately and tests format sensitivity only in prompt mode. Last updated 2026-04-12. — [Gorilla leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html); [Sky publications](https://sky.cs.berkeley.edu/publications/)
- **Gorilla** (Patil, Zhang, Wang, Gonzalez, 2023). Retrieval-aware API fine-tuning that reduces argument hallucination. — [arXiv 2305.15334](https://arxiv.org/abs/2305.15334)
- **RAFT** (domain-specific RAG fine-tuning). — [arXiv 2403.10131](https://arxiv.org/abs/2403.10131)
- **TinyAgent** (tool retrieval to shorten prompts, plus 1.1B/7B FC models). — [arXiv 2409.00608](https://arxiv.org/abs/2409.00608)
- **Tool2Vec.** — [arXiv 2409.02141](https://arxiv.org/abs/2409.02141)
- **Plan-and-Act** (Erdogan et al., ICML 2025). A separate planner and executor, trained on synthetic plans. — [arXiv 2503.09572](https://arxiv.org/abs/2503.09572)
- **MemGPT** (Packer et al., 2023; now Letta). Virtual context management across memory tiers. — [arXiv 2310.08560](https://arxiv.org/abs/2310.08560)
- **Sleep-time Compute** (Kevin Lin, Charlie Snell, Yu Wang, Charles Packer, Sarah Wooders, Stoica, Gonzalez). Precompute over the context before the query arrives. About 5x less test-time compute at equal accuracy and 2.5x lower cost per query when amortized; includes an agentic SWE case study. — [arXiv 2504.13171](https://arxiv.org/abs/2504.13171)
- **Autellix** (Michael Luo et al., NSDI 2026). Program-level scheduling against head-of-line blocking. — [arXiv 2502.13965](https://arxiv.org/abs/2502.13965)
- **Why Do Multi-Agent LLM Systems Fail?** MAST taxonomy: 14 failure modes, kappa 0.88, 1,600+ traces. — [arXiv 2503.13657](https://arxiv.org/abs/2503.13657)
- **Measuring Agents in Production.** 68% of agents run at most 10 steps before a human steps in, 70% prompt off-the-shelf models, and reliability is the top challenge. — [arXiv 2512.04123](https://arxiv.org/abs/2512.04123)
- **S\*** (test-time scaling for code: execution-grounded pairwise selection). — [arXiv 2502.14382](https://arxiv.org/abs/2502.14382)
- **R2E-Gym** (hybrid execution-based and execution-free verifiers, 51% SWE-Bench Verified for open weights). — [arXiv 2504.07164](https://arxiv.org/abs/2504.07164)
- **SkyRL-Agent** (SA-SWE-32B at 39.4% SWE-Bench Verified, RL only; an AST-based search tool raises pass@K). — [arXiv 2511.16108](https://arxiv.org/abs/2511.16108)
- **The Price Reversal Phenomenon** (Lingjiao Chen; Stanford lead, Stoica and Zaharia co-authors). Repeated runs of the same query vary up to 9.7x in thinking tokens, and models differ by up to 10x in turns. — [arXiv 2603.23971](https://arxiv.org/abs/2603.23971)

### Inferences
- **Thinking policy** (no engine code). Overthinking and NoThinking both support the Stanford-report recommendation: a low or medium default effort with escalation on failure.
- **pancake's thinking-budget cutoff should avoid "panic" and "leakage".** When the budget hits, inject a short closing cue ("Time is up; finish your reasoning in one sentence, then act") instead of a hard `</think>`. Track whether answers start containing reasoning. Low effort; reliability, not speed.
- **AsyncFC belongs in the opencode/client layer, not the engine.** opencode would need to keep decoding after emitting a call, with results injected later. With one slot that is feasible: pancake is idle during tool runs anyway. A 1.4x end-to-end gain on coding is the best agent-speed number found here. Effort: medium-high (a client fork).
- **LLMCompiler.** opencode already allows parallel tool calls in one message, so the cheap lever is prompting for batched reads and greps (LLMCompiler-style planning) and checking that pancake's tool-call recovery handles several calls per turn.
- **GEPA.** Run it offline on opencode sub-prompts that have a metric:
  - Compaction prompt: can a fresh agent continue?
  - Tool-call format adherence: BFCL-style AST checks.
  - Sub-agent task briefs.
  - pancake would be both the task and the reflection LM, overnight. Low code effort, compute-heavy.
- **Evaluation.** Use the BFCL V4 multi-turn subset and LiveCodeBench (Berkeley lead, contamination-free: [arXiv 2403.07974](https://arxiv.org/abs/2403.07974)) as gates before changing the quantization level (2-bit vs 3-bit), turning on intra-expert sparsity or changing tool-call recovery. Per Price Reversal, run each measurement several times and report distributions, because thinking-token counts vary up to ~10x between runs.
- **Sleep-time compute and MORI's idle windows.** opencode tool runs leave the GPU idle. pancake could use those windows to:
  - (a) bulk-swap experts seeded from the prefill's routing counts (already proposed in `expert_caching_prediction.md`);
  - (b) persist checkpoints to NVMe;
  - (c) optionally pre-summarize large files for later compaction.
  - Item (c) is speculative.
- **Low value here:**
  - Gorilla/RAFT/TinyAgent/Tool2Vec tool retrieval: static tool schemas are prefix-cached, so shortening them saves little after the first turn.
  - Autellix: multi-program scheduling.
  - SkyRL/R2E-Gym: training infrastructure.
  - MemGPT/Letta: opencode has its own compaction. MemGPT tiers are a design reference only.

### Gaps
- AsyncFC's code location and the coding benchmark it used were not checked beyond the abstract.
- BFCL V4 per-model rows, including Qwen3.x, could not be extracted from the leaderboard page.
- Whether BFCL's runner accepts an arbitrary OpenAI-compatible local endpoint was not verified.
- No Berkeley paper measures overthinking on Qwen3.x Next-style models specifically.

## Q4 (scope extension). AI-adjacent Berkeley systems work: learned/evolved policies, caching, kernels, evaluation

### Takeaway
The most transferable method is **ADRS (AI-Driven Research for Systems)**: evolve a policy against a replayable simulator with OpenEvolve, at about 1-5 h and under $20 per problem. It fits pancake's expert-cache eviction and prefetch policy well, now that hand-designed ClassLRU scoring has failed: replay routing traces and score the miss rate. K-Search covers LLM-guided kernel search, but only on CUDA.

### Cited Findings
- **Barbarians at the Gate: How AI is Upending Systems Research** (Audrey Cheng, Shu Liu, Melissa Pan, et al.; Berkeley lead, arXiv 2510.06189). OpenEvolve on 11 systems tasks.
  - MoE expert-parallel load balancer: rebalance in 3.7 ms against 19.6 ms for a non-public frontier-lab reference and ~540 ms for DeepSeek EPLB, at the same imbalance. 5 h, 300 iterations, under $10, Gemini 2.5 Flash/Flash-Lite.
  - LLM-SQL prefix-hit reordering: 3-3.9x faster at comparable hit rate.
  - Spot scheduling: +7% average (up to 16.7%) savings.
  - Sparse-attention design: 7% better error/density.
  - The paper's own numbers are internally inconsistent (EPLB 2x in Table 1 vs 5.0x in the text; transaction scheduling 20% vs 34%), and no code URL is given. — [arXiv 2510.06189](https://arxiv.org/abs/2510.06189); [HTML](https://arxiv.org/html/2510.06189)
- **Rethinking the Cost of Distributed Caches for Datacenter Services** (Mao et al., HotNets 2025), and **"Optimizing the cloud? Don't train models. Build oracles!"** (Bang et al., 2023): learned-systems positions from Sky. — [Sky publications](https://sky.cs.berkeley.edu/publications/)
- **Efficient LLM Scheduling by Learning to Rank** (Yichao Fu et al.; UCSD lead, Stoica co-author). Predicts the relative rank of output lengths to approximate shortest-job-first. — [arXiv 2408.15792](https://arxiv.org/abs/2408.15792)
- **K-Search** (LLM kernel generation, co-evolving world model). — [arXiv 2602.19128](https://arxiv.org/abs/2602.19128)
- **AI and Memory Wall** (Gholami et al.). Peak FLOPS grew 3.0x per 2 years against 1.6x for DRAM bandwidth and 1.4x for interconnect, so serving is memory-bound. — [arXiv 2403.14123](https://arxiv.org/abs/2403.14123)
- **LLM Inference Unveiled** (roofline survey; Berkeley co-authors). — [arXiv 2402.16363](https://arxiv.org/abs/2402.16363)
- **HERALD** (SNU lead, Berkeley co-authors): CPU-GPU cooperative KV retrieval for block-diffusion LLMs — [arXiv 2606.21633](https://arxiv.org/abs/2606.21633). **CDLM** (SqueezeAILab: consistency diffusion LMs, repo only).
- **ETS** (KV-sharing-aware tree search, 1.8x smaller KV and 1.4x throughput) — [arXiv 2502.13575](https://arxiv.org/abs/2502.13575). **SPECS** (latency-aware test-time scaling with a small drafter plus a reward model) — [arXiv 2506.15733](https://arxiv.org/abs/2506.15733).

### Inferences
- **ADRS for pancake's expert-cache policy.** Medium effort, potentially the best ROI among the "method" ideas.
  - Dump per-step routing traces from real opencode sessions; pancake already has an expert profile and telemetry hooks.
  - Write a small simulator of the VRAM, pinned RAM and NVMe tiers with measured costs (CPU 33 GB/s, PCIe 3.0 about 12 GB/s, and NVMe).
  - Let OpenEvolve (or GEPA's code-optimization mode) evolve eviction, prefetch and q*-split code against simulated decode time. Validate the winners on the box.
  - SeqMoE's published hit-rate headroom over LRU (local notes) suggests there is something to find. That is my inference; no Berkeley paper did this for a single-node MoE offload cache.
- **K-Search for gfx1100 kernels.** Speculative: evaluated on CUDA, and HIP graphs and unroll already showed no gain. Low priority.
- **Not applicable:** Learning-to-rank scheduling, ETS and SPECS (single request, no small drafter loaded), HERALD/CDLM (diffusion LMs) and Foundry (cold start; HIP graphs gave nothing). The Memory Wall paper and the roofline survey are background confirming the strategy of "read fewer bytes".

### Ranked shortlist for pancake / Avalon (estimates are mine unless quoted)
| # | Idea (paper, lead) | Where | Expected gain | Effort | Notes |
|---|---|---|---|---|---|
| 1 | Intra-expert neuron sparsity on CPU experts (Park et al. 2026, Berkeley lead) | pancake CPU MoE kernel | 1.15-1.3x decode (upper bound ~1.4x) | Medium-high (repack down weights neuron-major, threshold LUT) | Gate with BFCL/LiveCodeBench; paper cutoffs 84.5-87.4% on Qwen3.5 MoEs |
| 2 | Concurrent PCIe-fill vs CPU split q*=m·B_P/B_H (FreeToken 2026, Berkeley lead) | pancake miss path | 0-13% decode (DDR contention) | Medium | Check first whether it already exists |
| 3 | Checkpoints at `</think>`/tool-call/tool-result boundaries (FreeToken) | pancake parked cache | Seconds per edited turn | Low | Extends message-boundary checkpoints |
| 4 | Graceful thinking-budget stop (Interruptible LRMs) plus low default effort (Overthinking, NoThinking) | pancake server / opencode config | Fewer think tokens (about 10 s per 1k tokens avoided, Stanford report) | Low | Reliability plus speed |
| 5 | Per-step n-gram vs MTP choice and closed-loop draft length (SD Illusion, TurboSpec) | pancake / Avalon spec loop | 0-10% | Low-medium | MTP depth > 3 rarely pays (0.92/0.68/0.38) |
| 6 | ADRS/OpenEvolve-evolved expert-cache policy on replayed traces (Barbarians, Berkeley lead) | offline tooling | Unknown; hit-rate headroom exists | Medium | Replaces hand scoring that failed |
| 7 | AsyncFC (decode during tool runs) | opencode client | 1.4x end to end on coding (paper) | Medium-high | Client fork |
| 8 | GEPA on compaction and tool-format prompts | offline | Quality / reliability | Low code, high compute | Avalon or pancake as reflection LM |
| 9 | Sparse attention at 200k+ (Twilight [Tsinghua lead] / vAttention) | pancake attention | 0-30% at very long context only | High | Only if decode drops measurably at 200k |
| 10 | Idleness/TTL tiering of parked sessions (MORI, Continuum) | pancake server | Faster sub-agent switches | Medium | Multi-tenant papers; single-user benefit inferred |

### Gaps
- ADRS reports no single-node MoE offload cache experiment and gives no code URL. OpenEvolve is a third-party open-source project, and its URL was not verified here.
- Older Sky entries (2023) past the page offset I read may hold more items. The DBLP page was read up to about character 300k of 390k.
