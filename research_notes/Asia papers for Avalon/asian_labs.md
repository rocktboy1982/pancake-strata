# Asian AI company labs (Nov 2022 - Nov 2026): papers and reports that bear on pancake / Avalon running Qwen3.8-Flash-Next

Scope note: the earlier reports in /home/dan/reports/ (Stanford, MIT/Harvard, US universities, upstream projects) only touch Asian-lab work in passing: Kimi Linear/KDA, MoBA and DeepSeek EPLB as co-authored items, plus GDN kernels already ported. Nothing below repeats their verdicts. Already tested in pancake with no gain: HIP graphs, unroll flag, Vulkan, TurboQuant KV, L3 prefetch, ClassLRU scoring.

Verification legend: **[V]** means fetched or confirmed this session. **[K]** means an arXiv ID and figures from training knowledge that were NOT re-fetched this session. The writer should treat [K] numbers as needing a spot check before quoting.

## Q1. What do the Qwen reports say about Flash-Next architecture, MTP, expert routing and recommended agent settings?

### Takeaway
Qwen3.8-Flash-Next is **not just a scaled Qwen3-Next**. It adds three components that matter for our engines. (1) **QSA**, a DeepSeek-DSA-style indexed sparse attention in the 12 attention layers. (2) A **51B hash-addressed N-gram embedding table** (Engram-style) at layer 2. (3) A **4-branch Gated Residual stream**. Its single MTP layer is trained for multi-step drafting and reaches a mean accepted length of about 4.07 with 4-step drafting. Official agent sampling is T=1.0 / top_p 0.95 / top_k 20 with `preserve_thinking` on by default. The things most likely to be wrong or unexploited in our engines are the MTP draft depth, MTP-specific shortcuts (skip PLE, reuse QSA top-k), deterministic N-gram prefetch, and keeping prior thinking in the conversation cache.

### Cited Findings

**Qwen3.8-Flash-Next model card (Qwen, Aug 2026)** [V], https://huggingface.co/Qwen/Qwen3.8-Flash-Next
- Parameters: 125B total and 6B activated, **plus 51B n-gram embedding and 4B MTP**. The Hub widget shows 180B, which explains the "~177B" on the ISTA cards. — [HF card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- Shape: 48 layers, hidden size 2560. Layout is `12 x (3 x (Gated DeltaNet -> MoE) -> 1 x (Qwen Sparse Attention -> MoE))`. Vocabulary is 248,320 (padded). — [HF card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- GDN: 48 V heads, 16 QK heads, head dim 128. QSA: 24 Q heads, 2 KV heads, head dim 256, RoPE dim 64. The indexer is MQA with 4 Q heads and 1 shared K head, dim 128. Budget is 512 blocks or 2048 tokens. — [HF card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- MoE: **512 experts, 10 routed + 1 shared** per token, expert intermediate size **640**. Gated Residual: 4 branches, bottleneck rank 320. N-gram embedding: bigrams/trigrams at layer 2. MTP: **1 layer, "trained with multi-steps"**. Context is 262,144 native, up to 1M with static YaRN (factor 4.0). Qwen warns that static YaRN can hurt short texts. — [HF card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- Sampling: **Thinking: T=1.0, top_p=0.95, top_k=20, min_p=0, presence_penalty=0, repetition_penalty=1.0.** Instruct: T=0.7, top_p=0.80, top_k=20, presence_penalty=1.5. The agentic coding benchmarks (DeepSWE 1.1, SWE-bench Pro/Multilingual) ran at T=1.0, top_p=0.95. Presence penalty in 0-2 reduces loops, but higher values cause language mixing. — [HF card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- Reasoning: `reasoning_effort` defaults to **"xhigh"** (levels xhigh/medium/low). The card suggests budgets up to 262,144 reasoning tokens and 131,072 answer tokens. Qwen states that **lower effort does not always cut total task time in multi-turn agentic work, because it causes more failures and retries**. `enable_thinking` and **`preserve_thinking` (default on)** control thinking. — [HF card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- The model card gives **no quantization guide** and no speculative-decoding config. Official checkpoints are BF16 and FP8. — [HF card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)

**"On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability" (Qwen Team, Aug 26/31 2026)**, arXiv 2608.30320 [V via alphaXiv text], https://www.alphaxiv.org/abs/2608.30320 ; PDF at https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf
- QSA: token budget K=2048, compression ratio r=4 (4-token micro-blocks), partial RoPE on 64 of 128 indexer dims. Indexer cost falls from O(n^2) to O(n^2/r). Kernel speedups at 1M context versus GQA are 7.6x attention prefill and 4.9x decode. Indexer gains are 3.8x prefill and 4.4x decode (r=4 versus r=1). **Gains start at 64K.** QSA replaced full attention during continued pretraining, not original pretraining. — [alphaXiv 2608.30320](https://www.alphaxiv.org/abs/2608.30320.md)
- MTP: **mean accepted length with 4-step speculative decoding is 4.07 with QSA versus 4.06 with full attention.** MT-Bench gives 3.47 and HumanEval 4.26. The kernel benchmark used next_n=4, which the report says means **three MTP prediction steps**; the step count is stated inconsistently in the text. MTP reuses QSA top-k indices across draft steps. — [alphaXiv 2608.30320](https://www.alphaxiv.org/abs/2608.30320.md)
- N-gram embedding: multi-head hashing over short n-grams ending at each token, injected through contextual gating. **Deterministic addressing enables host-memory offload.** The table sits at layer 2 **so that host prefetch overlaps layer-1 compute.** Off-accelerator parameters total 51B. — [alphaXiv 2608.30320](https://www.alphaxiv.org/abs/2608.30320.md)
- Gated Residual: n_r=4, gate rank d/8. Reads and writes are fused, so the stream is traversed once per block in each direction. **Storing the residual in FP8 halves bytes moved "with almost no loss in quality."** Sparse 2-branch writes degraded after post-training. Memory traffic is named as the dominant inference cost of GR. — [alphaXiv 2608.30320](https://www.alphaxiv.org/abs/2608.30320.md)
- The report does **not** state the MoE load-balancing method, expert activation statistics, lookups per token, or weight quantization. — [alphaXiv 2608.30320](https://www.alphaxiv.org/abs/2608.30320.md)
- Secondary summary (aiwiki): Muon optimizer with 8 Newton-Schulz steps; AdamW for router and embeddings. Base model beats Qwen3.7-Plus-Base on 8/14 benchmarks with ~1/9 the training FLOPs. QSA ablation: 76.8 versus 75.9 for full attention on short-context benchmarks. — [aiwiki](https://aiwiki.ai/wiki/qwen3_8_flash_next/raw)

**SGLang day-0 blog for Qwen3.8-Flash-Next (LMSYS, Aug 26 2026)** [V], https://www.lmsys.org/blog/2026-08-26-qwen-flash-next
- **N-gram embedding ("PLE") per token: 16 hashed row IDs (eight 2-gram heads + eight 3-gram heads) x 160 values.** The 51.2B table is ~95.4 GiB in BF16. The table lives in pinned host memory, and a UVA kernel gathers only the 16 rows on a dedicated stream overlapped with block 1. Lookup positions are computed in advance. Result: -23.46 GiB of weights per GPU, with throughput change of -0.07% (geometric mean). — [LMSYS](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next)
- **The one-layer MTP draft disables PLE.** The target keeps PLE in prefill, decode and verify. — [LMSYS](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next)
- **IndexShare MTP:** the QSA top-k computed at draft-extend is reused for all draft steps (plus newly drafted positions). Indexer calls per MTP iteration drop from N to 1, **with accept length unchanged.** — [LMSYS](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next)
- MTP numbers: NVFP4, TP4 B200, batch size 1: **540 tok/s at accept length 3.3 (including the bonus token)**. H200 config "MTP-213" means 2 draft steps, top-k 1, 3 draft tokens per verify. — [LMSYS](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next)
- Index cache: one BF16 key per 4 tokens plus a 4-slot raw-key ring per request, cutting index-cache overhead by 80%. Compatible with Radix prefix cache (page-aligned `full_slot/4`). — [LMSYS](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next)
- Gated Residual small-M kernels: Mix 2.05x (12.36 to 6.03 us at M=4) for **+7.6% end-to-end**; Combine 1.96x for **+5.49% end-to-end** (B300). — [LMSYS](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next)
- Unofficial MTP acceptance: an MLX port reports "MTP-v2 acceptance 66-86%, content-dependent". A llama.cpp user reports 50-58% on reasoning text and **94% on code emission**. — [search summary of community posts](https://huggingface.co/RockTalk/Qwen3.8-Flash-Next-MLX-4bit) (not primary, not comparable)
- A community GGUF for 32 GB GPUs needs a modified llama.cpp and **offloads the N-gram table**. — [aiwiki, citing AgentionAI Gyro GGUF](https://aiwiki.ai/wiki/qwen3_8_flash_next/raw)

**Ancestry: Qwen3-Next-80B-A3B (Sep 2025)** [V partly]
- Layout: 48 layers = 12 x (3 GDN + 1 gated attention), each followed by MoE. MTP layers are bundled in the main checkpoint (SGLang NEXTN / EAGLE-style). — [SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3-Next), [Transformers docs](https://huggingface.co/docs/transformers/model_doc/qwen3_next)
- vLLM's official example uses `{"method":"qwen3_next_mtp","num_speculative_tokens":2}`. — [HF card mirror](https://huggingface.co/Qwen/qwen3-next-80b-a3b-instruct)
- Qwen3.5-397B-A17B (Feb 2026) uses the same 3:1 GDN/gated-attention pattern and 512 experts with 10 routed + 1 shared. **No formal Qwen3.5 tech report was found.** — [mlabonne HF blog](https://huggingface.co/blog/mlabonne/qwen35)

**Qwen routing papers that explain the expert statistics**
- **"Demons in the Detail: On Implementing Load Balancing Loss for Training Specialized MoE"** (Qwen, Jan 2025), arXiv 2501.11873 [V]. Micro-batch load-balance loss routes even domain-specific sequences uniformly across experts. **Global-batch LBL "greatly improves the domain specialization of MoE experts."** Micro-batch balance gets worse, so a small micro-batch term is added. — [arXiv 2501.11873](https://arxiv.org/abs/2501.11873v2), [Qwen blog](https://qwenlm.github.io/blog/global-load-balance)
- **Qwen3 Technical Report** (May 2025), arXiv 2505.09388 [K]. Uses global-batch LBL and drops shared experts in Qwen3 MoE; Flash-Next restores one shared expert. Introduces thinking mode fusion and the **thinking-budget mechanism**: when the budget is hit, the system inserts a stop-thinking instruction plus `</think>` and the model answers from its partial reasoning. Accuracy scales smoothly with the budget. Recommended thinking sampling was T=0.6/top_p 0.95/top_k 20, with greedy decoding discouraged. — [arXiv 2505.09388](https://arxiv.org/abs/2505.09388)
- **Gated Attention for LLMs** (Qwen, May 2025), arXiv 2505.06708 [K]. A sigmoid output gate after SDPA removes attention sinks and massive activations and improves long-context extrapolation. This is the gated-attention block of Qwen3-Next. It matters for quantization because there are fewer activation outliers. — [arXiv 2505.06708](https://arxiv.org/abs/2505.06708)

### Inferences
- **The MTP draft depth is probably too shallow if pancake/Avalon draft 1-2 tokens.** The official evaluation drafts 3-4 steps with one multi-step-trained layer and gets ~4.07 tokens per verify. Code emission accepts at ~94% (community data). Our verify costs the expert union of k+1 tokens, so the optimum is lower than on B200. A sweep of draft depth 2/3/4, ideally adaptive by entropy (see MiMo, Q2), is the cheapest experiment to run. Expected gain: 10-30% decode on code-heavy turns. This is an estimate, not measured.
- **The MTP draft step should skip the N-gram lookup (PLE) and reuse the QSA top-k from the last verify**, as SGLang does with no accept-length loss. If pancake runs PLE or the QSA indexer inside each draft step, that is pure waste. The indexer gain is small below 64K; PLE is a host gather.
- **N-gram table memory:** the BF16 table is 95.4 GiB and cannot fit in 57 GB RAM. It must be quantized or NVMe-mmapped; GSQ/GGUF presumably quantize it. Per-token traffic is only 16 rows x 160 values (~5 KB in BF16). **Row IDs depend only on token IDs**, so for prefill and for the accepted and draft tokens the rows can be prefetched before layer 1 runs. If the table is mmapped from NVMe, misses there are a hidden decode stall. That should be measured; a page-fault counter during decode will show it.
- **Gated Residual:** our engines carry a 4 x 2560 residual. Fusing the read/combine/write (one pass each way) and FP8 storage are officially "almost lossless". SGLang got +5-8% end-to-end from small-M GR kernels on B300. With 6B active on a 20 GB card this is likely a few percent at most. Check whether pancake's GR is one fused kernel per block or several.
- **Do not lower the default effort to save time on agent tasks.** Qwen explicitly says low effort can increase total task time. Thinking-mode sampling should be T=1.0 (not the Qwen3 T=0.6 heritage) with presence_penalty 0.
- **`preserve_thinking` default on** means the official template keeps earlier `<think>` blocks in history. If opencode or our template strips them, two problems follow: (a) behaviour deviates from training, and (b) **the conversation-cache prefix diverges on every turn**, forcing re-prefill from the first stripped block. Verify that the template and opencode both preserve reasoning, so that conversation-cache hits extend through prior turns.
- Global-batch LBL makes experts **domain-specialized**. A single-user coding session should therefore have a concentrated, slowly drifting expert working set, which favours session-level frequency pinning over per-token LRU. ClassLRU scoring gave no gain, so the useful lever is probably **pinning by long-window frequency per layer** rather than a smarter recency score.

### Gaps
- The full PDF of the tech report was not readable (GitHub viewer). The last ~9.6k characters of the alphaXiv text were not read. MTP loss weighting, per-layer expert load statistics and the balancing method could not be confirmed.
- No official Qwen statement gives MTP acceptance per draft position or at a low-bit quant. Community figures conflict (50-94%).
- No official quantization guidance (which tensors are sensitive: router, shared expert, PLE table, GR gates) was found for Flash-Next.

## Q2. Which DeepSeek / Moonshot / ByteDance / MiniMax / Xiaomi / other inference techniques transfer to one consumer GPU with offloaded experts?

### Takeaway
Most hyperscale Asian-lab systems work targets multi-node expert parallelism: DeepEP, MegaScale-Infer, Comet, Mooncake, 3FS, EPLB. That work does not transfer. What does transfer: (a) MTP drafting recipes (DeepSeek-V3, MiMo-V2-Flash, GLM-4.5, LongCat) and entropy-adaptive draft length; (b) DSA-style indexer reuse, since QSA is DSA-derived; (c) Engram-style host-memory tables with deterministic prefetch, which is exactly Flash-Next's PLE; (d) the KV/prefix-cache ideas from Mooncake at single-machine scale (RAM/NVMe tier for the conversation cache).

### Cited Findings

**DeepSeek**
- **DeepSeek-V3 Technical Report** (Dec 2024), arXiv 2412.19437 [K]. Auxiliary-loss-free balancing uses a per-expert **bias added only for top-k selection, not for gating weights**. Layout is 256 routed + 1 shared experts, top-8, node-limited routing. MTP depth 1. **2nd-token acceptance 85-90%, ~1.8x TPS** when MTP is used for speculative decoding. FP8 training with fine-grained 1x128/128x128 scaling. — [arXiv 2412.19437](https://arxiv.org/abs/2412.19437)
- **DeepSeek-V3.2(-Exp): DeepSeek Sparse Attention (DSA)** (Sep/Dec 2025), arXiv 2512.02556 [K]. A lightning indexer (few heads, FP8) scores tokens and the top-2048 KV entries are attended. This is the template for Qwen's QSA, which uses the same K=2048 and adds 4-token compression. — [arXiv 2512.02556](https://arxiv.org/abs/2512.02556)
- **DeepSeek-V4** (Apr 2026). Single-layer MTP head that is **SWA-only (no compressor, no indexer)** and uses the previous hidden state plus the next-token embedding. SGLang adds a "HiSparse CPU-extended KV" and a ShadowRadix prefix cache. **No expert-offload material was found.** — [LMSYS DeepSeek-V4 blog](https://www.lmsys.org/blog/2026-04-25-deepseek-v4), [HF blog](https://huggingface.co/blog/deepseekv4)
- **Engram: "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for LLMs"** (DeepSeek, Jan 2026) [V repo]. Static N-gram embedding tables complement MoE. A U-shaped scaling law sets the MoE-versus-memory split. **Deterministic addressing allows host-memory offload with minimal overhead.** Secondary summary: a 100B-parameter Engram layer offloaded to CPU costs a **2.8% throughput drop** on the 8B model. — [github.com/deepseek-ai/Engram](https://github.com/deepseek-ai/Engram), [hyper.ai summary](https://hyper.ai/en/papers/Engram)
- **Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware** (May 2025), arXiv 2505.09343 [K]. Shows MTP gives 1.8x and argues decode is memory-bound. — [arXiv 2505.09343](https://arxiv.org/abs/2505.09343)
- DeepEP, FlashMLA, DeepGEMM, EPLB, 3FS (Open Infra week, Feb 2025) [K]. These are NVIDIA multi-node or MLA-specific. — [github.com/deepseek-ai/open-infra-index](https://github.com/deepseek-ai/open-infra-index)

**Xiaomi**
- **MiMo-V2-Flash Technical Report** (Jan 2026), arXiv 2601.02780 [V]. Uses **3 MTP layers** (dense FFN + SWA, ~0.33B each). Reports **up to 3.6 acceptance length and 2.6x decode speedup.** Next-token entropy strongly determines acceptance: low-entropy contexts accept more. — [arXiv 2601.02780](https://arxiv.org/html/2601.02780v1), [review](https://www.themoonlight.io/review/mimo-v2-flash-technical-report)
- **MiMo: Unlocking the Reasoning Potential** (May 2025), arXiv 2505.07608 [K]. MTP used for speculative decoding of RL rollouts, with high acceptance. — [arXiv 2505.07608](https://arxiv.org/abs/2505.07608)

**Moonshot**
- **Kimi Linear (KDA)** (Oct 2025), arXiv 2510.26692 [K]. 3:1 KDA/MLA hybrid, up to 75% less KV cache and up to 6x decode throughput at 1M. Requires retraining, and was already covered as co-authored in the MIT report. Not applicable to Qwen weights. — [arXiv 2510.26692](https://arxiv.org/abs/2510.26692)
- **MoBA** (Feb 2025), arXiv 2502.13189 [K]. Block-sparse attention with top-k gating. Superseded for us by QSA, which is baked into the weights. — [arXiv 2502.13189](https://arxiv.org/abs/2502.13189)
- **Mooncake** (Jul 2024), arXiv 2407.00079 [K]. KV-cache-centric disaggregation with a KV pool spanning DRAM/SSD across nodes. The transferable idea is a **tiered RAM/NVMe store for conversation-cache states**. For a hybrid model the per-checkpoint state is small: GDN recurrent state plus the QSA KV of 12 layers. — [arXiv 2407.00079](https://arxiv.org/abs/2407.00079)
- **Kimi K2** (Jul 2025), arXiv 2507.20534 [K]. 384 experts, 8 active + 1 shared, MuonClip, agentic tool-use data synthesis. Relevant only as evidence that sparser MoE (more experts, fewer active) is the trend. — [arXiv 2507.20534](https://arxiv.org/abs/2507.20534)

**ByteDance Seed**
- **MegaScale-Infer** (Apr 2025), arXiv 2504.02263 [K]. Disaggregates attention and FFN/expert nodes with a ping-pong micro-batch pipeline; up to 1.90x per-GPU throughput. The conceptual analogue for us is running GPU attention/GDN concurrently with CPU experts, but that needs multiple micro-batches, which a single-user decode with batch 1 lacks. MTP verify batches are the exception. — [arXiv 2504.02263](https://arxiv.org/abs/2504.02263)
- **Comet** (Feb 2025), arXiv 2502.19811 [K]. Fine-grained compute/communication overlap for MoE layers (1.96x per layer, 1.71x end-to-end in training). Not applicable: training and multi-GPU. — [arXiv 2502.19811](https://arxiv.org/abs/2502.19811)
- **UltraMem** (Nov 2024), arXiv 2411.12364 [K]. Large sparse memory layers instead of MoE, with claimed up to ~6x faster inference than MoE at equal parameters. It is a conceptual sibling of Engram/PLE. Architecture only. — [arXiv 2411.12364](https://arxiv.org/abs/2411.12364)

**MiniMax**
- **MiniMax-01** (Jan 2025), arXiv 2501.08313 [K]. Lightning attention (linear) with 1 softmax layer per 7 linear layers; 456B total / 45.9B active. **MiniMax-M1** (Jun 2025), arXiv 2506.13585 [K]. CISPO RL, 40k/80k thinking budgets, lightning attention at ~25% of DeepSeek-R1 FLOPs at 100k generation length. Kernel ideas overlap with the GDN chunked kernels already ported. — [arXiv 2501.08313](https://arxiv.org/abs/2501.08313), [arXiv 2506.13585](https://arxiv.org/abs/2506.13585)

**Meituan, Zhipu, Huawei, Baidu, Tencent**
- **LongCat-Flash** (Meituan, Sep 2025), arXiv 2509.01322 [K]. **Zero-computation experts** make active parameters vary per token (~18.6-31.3B, ~27B average). Shortcut-connected MoE overlaps compute and communication. Reports MTP acceptance >90%. Architecture-specific. — [arXiv 2509.01322](https://arxiv.org/abs/2509.01322)
- **GLM-4.5** (Zhipu, Aug 2025), arXiv 2508.06471 [K]. 355B/32B, includes an MTP layer for speculative decoding; hybrid thinking/non-thinking modes. Same recipe family. — [arXiv 2508.06471](https://arxiv.org/abs/2508.06471)
- **Pangu Pro MoE / MoGE** (Huawei, May 2025), arXiv 2505.21411 [K]. Grouped experts give equal per-group load by construction. **Pangu Ultra MoE**: real-time expert-load prediction for placement (training). — [arXiv 2505.21411](https://arxiv.org/abs/2505.21411), [Pangu Ultra MoE review](https://liner.com/review/pangu-pro-moe-mixture-grouped-experts-for-efficient-sparsity). Not applicable to Qwen weights.
- **ERNIE 4.5 Technical Report** (Baidu, Jun 2025) [K]. Reports near-lossless 4-bit and 2-bit weight quantization via "convolutional code quantization". Relevant only as evidence that ~2-bit MoE serving is viable, which our GSQ Q2_0 already does. — [ERNIE 4.5 report](https://yiyan.baidu.com/blog/publication/ERNIE_Technical_Report.pdf)
- **Hunyuan-Large** (Tencent, Nov 2024), arXiv 2411.02265 [K]. Cross-layer attention cuts the KV cache. Architecture-specific. — [arXiv 2411.02265](https://arxiv.org/abs/2411.02265)

### Inferences
Ranked by expected value for pancake/Avalon:
1. **Entropy-adaptive MTP draft length** (MiMo evidence plus the Qwen 3-4 step recipe). Draft more when the main-model entropy is low (code emission, tool-call JSON) and stop early on reasoning prose. Expected gain: the largest available, likely 10-30% on agent turns that are mostly code. Low implementation cost.
2. **MTP draft shortcuts**: skip PLE in draft (SGLang), reuse the QSA top-k (IndexShare), and use an SWA-only draft at long context (DeepSeek-V4 shows a cheap-attention draft head is fine). A few percent per draft step; more at long context.
3. **Deterministic N-gram (PLE) prefetch** from RAM/NVMe one step ahead, including draft tokens (Engram, Qwen, SGLang). Avoids stalls. The gain is zero if PLE is already resident; it can be large if the table is mmapped and faulting.
4. **Mooncake-style tiered conversation cache.** Spill GDN state plus 12-layer QSA KV checkpoints to RAM/NVMe for older opencode sessions. Helps the turn-level latency of resumed sessions, not tokens/s.
5. Not transferable: DeepEP, EPLB, MegaScale-Infer, Comet, 3FS, FlashMLA (MLA only), MoGE, Kimi Linear/MoBA/UltraMem/LongCat (all need retraining).

### Gaps
- DeepSeek-V3 / V3.2 / V4, Kimi, MiniMax, ByteDance, LongCat, GLM and ERNIE figures above are [K] and were not re-fetched within this session's tool budget.
- No Asian-lab paper on **single-consumer-GPU expert offloading** was found. Huawei and the others publish datacenter work. Consumer offloading papers (MoBiLE, ProMoE, DALI, HybriMoE) are university-led and belong to other reports.

## Q3. Are there published expert-activation statistics that inform caching or prefetch policy?

### Takeaway
No Asian company lab has published per-expert activation or temporal-locality statistics for Qwen3-Next or Flash-Next. The only first-party signal is qualitative: Qwen's global-batch LBL produces **domain-specialized experts**. Third-party measurements on temporal locality conflict.

### Cited Findings
- Global-batch LBL (used by Qwen) increases domain specialization of experts, and micro-batch balance is weaker as a result. — [arXiv 2501.11873](https://arxiv.org/abs/2501.11873v2)
- The Flash-Next tech report does not give expert load or routing statistics. — [alphaXiv 2608.30320](https://www.alphaxiv.org/abs/2608.30320.md), [aiwiki](https://aiwiki.ai/wiki/qwen3_8_flash_next/raw)
- Third-party evidence conflicts. One study finds SoTA MoEs lack strong temporal locality, so LRU caches thrash, and proposes cache-aware routing. — [arXiv 2412.00099](https://arxiv.org/html/2412.00099v1). DALI finds strong token-to-token workload correlation and reports that HybriMoE reaches only a 25.3% hit rate on Mixtral-8x7B. — [arXiv 2602.03495](https://arxiv.org/html/2602.03495v1). FlashMoE (Qwen3-30B example) claims up to +51% hit rate over LRU/LFU. — [search result](https://www.arxiv.org/pdf/2601.17063)
- DeepSeek-V3's selection-only bias means the routing distribution at inference includes a learned balancing bias. Engines must add it to scores for top-k selection but not to the gate weights. [K] — [arXiv 2412.19437](https://arxiv.org/abs/2412.19437)

### Inferences
- With 10 of 512 experts per layer plus a shared expert, and domain-specialized experts, the right policy for a single-user coding agent is probably **a pinned per-layer "hot set" chosen from long-window counts within the session**, refreshed slowly, plus a small LRU for the rest. ClassLRU (no gain) suggests the remaining misses are close to uniform. In that case the levers are reducing miss cost (CPU bytes, see the Berkeley intra-expert sparsity in the US report) and MTP verify batching, which amortizes each expert miss over k+1 tokens.
- The shared expert (1 per layer) should always be VRAM-resident. Confirm it is never counted in the routed-cache budget.
- Cheapest first-party data source: **log pancake's own router top-10 per layer over real opencode sessions**. No lab has published this for Flash-Next.

### Gaps
- No company-lab expert-activation histogram, reuse-distance curve or cross-layer predictability data exists for Qwen3-Next or Flash-Next as of 2026-11-09.
