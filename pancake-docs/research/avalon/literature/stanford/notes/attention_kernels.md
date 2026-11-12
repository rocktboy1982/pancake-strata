# Stanford papers on efficient attention, GPU kernels and efficient architectures (Nov 2022 - Nov 2026), and how they apply to Avalon

Scope: Stanford-led or Stanford-central work (mostly Hazy Research / Chris Re, CRFM, Scaling Intelligence / Azalia Mirhoseini). Speculative decoding, serving, quantization and MoE-offload papers are left out on purpose.
Target setup: llama.cpp fork "Avalon"; Qwen3.8-Flash-Next (~125B MoE hybrid: gated delta net (GDN) linear-attention layers plus full-attention layers, MTP head); RX 7900 XT (gfx1100, RDNA3, wave32, WMMA, no MFMA), ROCm 10; experts in CPU RAM/NVMe; single user, prompts of 131-262k, q8_0/q4_0 KV.

Verification note: abstracts were fetched this session for ThunderKittens, Based, LoLCATs, Hydragen, FlashAttention-2, the HipKittens blog, the Flash-Decoding blog and Cartridges. Items marked "(not re-fetched)" are cited from their arXiv IDs without opening the page this session. The report writer should treat those numbers as lower confidence.

## Q1. FlashAttention-2/3 and successors, ThunderKittens / HipKittens and kernel DSLs: what is already in llama.cpp, and what could help RDNA3?

### Takeaway
The core FlashAttention ideas (tiled online softmax, no materialized S matrix, split-KV "Flash-Decoding" for long-context decode) are already in llama.cpp's own ggml-cuda/HIP fattn kernels (tile, vec, rocWMMA and MMA paths). FA3 and ThunderKittens are tied to NVIDIA Hopper/Blackwell hardware features, so they offer ideas but no code we can port. HipKittens is the only Stanford kernel work on AMD, and it targets CDNA3/CDNA4 (MI300/MI355X, MFMA), not RDNA3. Its main transferable lessons are "don't use producer/consumer wave specialization on AMD" and "schedule registers and tiles explicitly", which could guide a hand-tuned gfx1100 attention or GDN kernel. Expect at most modest prefill gains: only the full-attention layers benefit, and on this model's ~89 GiB footprint the expert/CPU path probably dominates.

### Cited Findings
**FlashAttention-2** (Tri Dao, Stanford/Hazy alumnus; arXiv 2307.08691, Jul 17 2023; ICLR 2024)
- Better work partitioning (fewer non-matmul FLOPs, parallelism over sequence length, split work across warps). About 2x over FlashAttention-1 and 50-73% of A100 peak FLOPs; up to 225 TFLOPs/s per A100 in end-to-end GPT training — [arXiv 2307.08691](https://arxiv.org/abs/2307.08691)
- Code: Dao-AILab/flash-attention (CUDA; a separate AMD ROCm/CK port exists for CDNA, not RDNA3).
- Applicability: the algorithm is already in llama.cpp (ggml fattn kernels are an independent implementation of the same tiling and online-softmax idea). Nothing to port.

**Flash-Decoding** (Tri Dao, Daniel Haziza, Francisco Massa, Grigory Sizov; Stanford CRFM blog, Oct 2023)
- Splits K/V along the sequence, runs attention on the chunks in parallel, then merges them with a log-sum-exp reduction. Reported up to 8x faster end-to-end generation for very long sequences (CodeLlama-34B, 512 to 64k tokens) and attention up to 50x faster than FlashAttention. Attention time stays roughly flat up to 32k — [CRFM blog](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
- Code: flash-attention >= 2.2, xFormers >= 0.0.22 — [CRFM blog](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
- Applicability: **already present in spirit.** llama.cpp's fattn "vec" kernel splits the KV length across parallel blocks and combines the results, which is the same idea. It matters for 131-262k decode on the full-attention layers. Gain: none new; at most retune the parallel-blocks heuristic for 84 CUs on gfx1100 at very long contexts.

**FlashAttention-3** (arXiv 2407.08608, Jul 2024; Colfax/Meta/NVIDIA/Princeton/Together; Tri Dao is now at Princeton, so Stanford is not the lead) (not re-fetched)
- Hopper-specific: TMA, wgmma, warp-specialized async pipelines, FP8 — [arXiv 2407.08608](https://arxiv.org/abs/2407.08608)
- Applicability: **not portable to RDNA3**, which has none of these hardware features. Listed only because the brief asked.

**ThunderKittens (TK)** (Benjamin Spector, Simran Arora, Aaryan Singhal, Daniel Y. Fu, Christopher Re; Hazy Research; arXiv 2410.20399, Oct 27 2024; ICLR 2025)
- An embedded C++ DSL built around 16x16 register tiles at the warp level, async overlap at the block level and launch-cost hiding at the grid level — [arXiv 2410.20399](https://arxiv.org/abs/2410.20399)
- Matches cuBLAS and FA3 on GEMM and attention inference; attention backward 10-40% faster than the best baselines; SSM kernels up to 8x faster; linear-attention kernels up to 14x faster — [arXiv 2410.20399](https://arxiv.org/abs/2410.20399)
- A secondary Latent Space profile says ThunderKittens 2.0 (Re with Stuart Sul) was released, and that in Aug 2026 the group used agents to generate hardware-specific code for an MoE megakernel instead of the TK abstraction layer — [Latent Space wiki: Chris Re](https://latent.space/wiki/chris-re/) (secondary source, no primary post found)
- Code: github.com/HazyResearch/ThunderKittens (NVIDIA only).
- Applicability: **no direct port.** The linear-attention kernels (up to 14x) are relevant in concept to GDN prefill, but they are written for Hopper.

**HipKittens (HK): "Fast and Furious AMD Kernels"** (William Hu, Drew Wadsworth, Sean Siddens, Stanley Winata, Daniel Y. Fu, Ryan Swann, Muhammad Osama, Christopher Re, Simran Arora; Stanford Hazy Research with AMD co-authors; arXiv 2511.08083, Nov 11 2025)
- Brings TK-style tile abstractions to AMD. Finding: tiles carry over to AMD, but the algorithms on top must change. Wave specialization, the dominant NVIDIA pattern, underperforms on CDNA3/CDNA4, so HK uses other scheduling patterns — [Hazy blog](https://hazyresearch.stanford.edu/blog/2025-11-09-hk); [alphaXiv overview](https://www.alphaxiv.org/overview/2511.08083)
- Main platform is the MI355X (CDNA4), plus CDNA3. **The blog never mentions consumer RDNA GPUs** — [Hazy blog](https://hazyresearch.stanford.edu/blog/2025-11-09-hk)
- Reported results: ~500-line attention-forward kernels beat all AMD baselines on average, including AITER assembly. Attention backward beats the best available baselines; AITER and PyTorch SDPA GQA backward reach only 30% and 24% of SoTA on MI355X. BF16/FP8 GEMMs compete with AITER and are ~1.3-3.0x faster than Triton. Memory-bound ops (RoPE, fused dropout-residual-layernorm) are 1.1-2.2x faster — [Hazy blog](https://hazyresearch.stanford.edu/blog/2025-11-09-hk); [alphaXiv](https://www.alphaxiv.org/overview/2511.08083)
- AMD's ROCm Triton and Composable Kernel (CK) often underperform, even on plain BF16 GEMM — [Hazy blog](https://hazyresearch.stanford.edu/blog/2025-11-09-hk)
- Code: [github.com/HazyResearch/HipKittens](https://github.com/HazyResearch/HipKittens)
- Applicability to gfx1100: **cannot be used as-is.** HK is built on CDNA MFMA instructions and wave64. RDNA3 has WMMA (16x16x16, wave32), a different LDS/register budget and no matrix cores of the CDNA kind. Porting would mean rewriting the tile primitives for WMMA. What transfers: (a) avoid producer/consumer wave specialization on AMD; (b) hand-schedule at tile granularity instead of relying on the compiler or Triton; (c) evidence that Triton/CK baselines on AMD are weak, which argues for hand-written HIP. Possible gain: only the full-attention-layer share of prefill, plus any custom GDN kernel. At most a few tens of percent on attention time, and less end-to-end because the expert/CPU path dominates (inference, not measured).

**Current llama.cpp state on RDNA3 (context, not Stanford)**
- The rocWMMA FA path (`-DGGML_HIP_ROCWMMA_FATTN=ON`) is recommended for RDNA3+/CDNA — [search summary of llama.cpp docs/build](https://git.peisongxiao.com/CS348Project/llama.cpp/commit/3ffbbd5ce130859be91909e9b77d4c1962a6be2c?files=docs)
- A secondary news item says PR #22880 (b9158, May 2026) added RDNA3 support to the MMA FA path, and that RDNA3/RDNA4 did not beat the tile kernel for head sizes above 128 — [startupfortune](https://startupfortune.com/?p=10713) (secondary; check the PR)

**ThunderMLA / megakernels** ("Look Ma, No Bubbles!", Hazy Research blog, May 27 2025)
- Fuses the whole Llama-1B forward pass into one persistent "megakernel" with an on-GPU instruction interpreter (ThunderMLA-derived). Reaches 78% of H100 memory bandwidth and >1.5x over existing systems; a secondary source reports ~680 us per forward on B200, 3.5x faster than vLLM — [Hazy blog](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles); [36kr](https://www.36kr.com/p/3314363207214852) (secondary)
- Applicability: the idea (removing kernel-launch bubbles at batch-1 decode) is sound. llama.cpp already uses CUDA/HIP graphs, which recover part of the gain. With experts on CPU, GPU launch bubbles are a small share of token time, so expected gain is low (inference).

**KernelBench** (Anne Ouyang, Simon Guo, ..., Azalia Mirhoseini; Stanford Scaling Intelligence; arXiv 2502.10517, Feb 2025) (not re-fetched)
- A benchmark for LLM-generated GPU kernels — [arXiv 2502.10517](https://arxiv.org/abs/2502.10517)
- Applicability: a method, not a kernel. Combined with the Aug 2026 "agents generate hardware-specific code" direction above, it suggests using an agent loop to generate and tune a gfx1100 GDN chunk kernel against the ggml reference op.

### Inferences
- The biggest kernel-side opportunity on RDNA3 is probably not full-attention FA (already handled by the tile/rocWMMA/MMA paths) but the **GDN layers**: a single user report says the HIP build of the GDN kernel runs no faster than the CPU path on gfx1151 (see Q2). The TK/HK lessons point to a hand-written WMMA chunked GDN kernel.
- Wave specialization being bad on CDNA likely carries over to RDNA3, which also lacks Hopper-style async TMA/warpgroup features. This is an extrapolation; the HK authors did not test RDNA.

### Gaps
- No Stanford paper or blog benchmarks RDNA3 (gfx1100) or any consumer AMD GPU.
- Exact HipKittens attention TFLOPs appear only in plots (not in text); not extracted.
- The ThunderKittens 2.0 primary source and the 2026 Hazy MoE megakernel post were not found; only a secondary profile.

## Q2. Linear attention, state-space and hybrid architectures (Based, Hyena / Monarch Mixer, linearization such as LoLCATs): relevance to running GDN hybrids fast

### Takeaway
Stanford's linear-attention work (Based, Hedgehog, LoLCATs, Zoology/JRT) is about designing or distilling models, not about kernels for the gated delta rule. The chunked GDN kernel comes from MIT/NVIDIA (Songlin Yang et al., flash-linear-attention), not Stanford. The directly useful Stanford pieces are the TK linear-attention kernel design (up to 14x faster) and Based's IO-aware recurrent kernels as a reference for how to make linear-attention prefill and decode fast. llama.cpp already has a fused GATED_DELTA_NET op with a chunked path, but its HIP performance is reportedly poor. That is the most concrete opportunity for Avalon.

### Cited Findings
**Based** (Simran Arora, Sabri Eyuboglu, Michael Zhang, Aman Timalsina, Silas Alberti, Dylan Zinsley, James Zou, Atri Rudra, Christopher Re; arXiv 2402.18668, Feb 28 2024; ICML 2024)
- Linear attention (Taylor feature map) plus sliding-window attention, chosen to balance recall against state size. At 1.3B it beats Mamba by 6.22 points on recall-heavy tasks, matches Mamba's perplexity, and reaches up to 24x the generation throughput of FlashAttention-2 (1,024 generated tokens) — [arXiv 2402.18668](https://arxiv.org/abs/2402.18668)
- Code: HazyResearch/based (IO-aware kernels later rewritten in TK).
- Applicability: architecture only, and our model is fixed. The insight that recall is the weak spot of recurrent layers explains why Qwen keeps some full-attention layers; those layers set the KV cost at 262k. No direct speedup.

**LoLCATs** (Michael Zhang, Simran Arora, Rahul Chalamala, Alan Wu, Benjamin Spector, Aaryan Singhal, Krithik Ramesh, Christopher Re; arXiv 2410.10254, Oct 2024; ICLR 2025)
- Linearizes a pretrained Transformer: attention transfer, then low-rank (LoRA) fine-tuning, using 0.2% of the parameters and 0.4% of the tokens of earlier methods. Over 20 points better 5-shot MMLU than prior linearizations; first linearized 70B and 405B models, closing 77.8% / 78.1% of the MMLU gap — [arXiv 2410.10254](https://arxiv.org/abs/2410.10254)
- Code: HazyResearch/lolcats.
- Applicability: in theory we could linearize the remaining full-attention layers of the hybrid to kill KV growth at 262k. In practice, distilling a 125B MoE needs multi-GPU training plus quality evaluation, so it is **not feasible on our hardware**. Low.

**Hedgehog** (Michael Zhang, Kush Bhatia, Hermann Kumbong, Christopher Re; arXiv 2402.04347, Feb 2024; ICLR 2024) (not re-fetched)
- Learnable MLP feature maps that imitate softmax attention's spiky/monotonic behaviour, for linear-attention distillation — [arXiv 2402.04347](https://arxiv.org/abs/2402.04347)
- Applicability: a precursor to LoLCATs. Not applicable to inference of an existing GDN model.

**Zoology** (Arora, Eyuboglu, Timalsina, Johnson, Poli, Zou, Rudra, Re; arXiv 2312.04927, Dec 2023) and **Just Read Twice (JRT)** (Arora et al.; arXiv 2407.05483, Jul 2024) (not re-fetched)
- Zoology: measures the associative-recall gap between gated-convolution/recurrent models and attention — [arXiv 2312.04927](https://arxiv.org/abs/2312.04927)
- JRT: repeating the context in the prompt improves recall for recurrent models — [arXiv 2407.05483](https://arxiv.org/abs/2407.05483)
- Applicability: JRT is a prompt-level trick. For a hybrid with full-attention layers it mostly doubles prefill cost, so it is **not recommended**. It is mainly useful to explain recall failures in GDN layers.

**Hyena** (Poli, Massaroli, Nguyen, Fu, Dao, Baccus, Bengio, Ermon, Re; arXiv 2302.10866, Feb 2023; ICML 2023), **Monarch Mixer** (Daniel Y. Fu, Simran Arora et al.; arXiv 2310.12109, Oct 2023; NeurIPS 2023), **FlashFFTConv** (Daniel Fu, Hermann Kumbong, Eric Nguyen, Christopher Re; arXiv 2311.05908, Nov 2023; ICLR 2024)
- FlashFFTConv: a Monarch decomposition of the FFT lets long convolutions run on tensor cores. Up to 7.93x over PyTorch (one setting: sequence length 1K on H100) and 4.4x end-to-end; scales from 256 to 4M sequence length — [Hazy blog](https://hazyresearch.stanford.edu/blog/2023-11-13-flashfftconv); [HF papers 2311.05908](https://huggingface.co/papers/2311.05908)
- Hyena and M2 abstracts were not re-fetched — [arXiv 2302.10866](https://arxiv.org/abs/2302.10866); [arXiv 2310.12109](https://arxiv.org/abs/2310.12109)
- Applicability: **none for our model.** GDN layers use only a short causal conv1d (kernel ~4), not long FFT convolutions.

**H3** (Dao, Fu, Saab, Thomas, Rudra, Re; arXiv 2212.14052, Dec 2022) and **Laughing Hyena Distillery** (Massaroli et al., arXiv 2310.18780, Oct 2023) (not re-fetched)
- Early SSM-LM and conv-to-recurrence distillation work — [arXiv 2212.14052](https://arxiv.org/abs/2212.14052); [arXiv 2310.18780](https://arxiv.org/abs/2310.18780)
- Applicability: historical; none.

**Where the GDN kernel actually comes from (not Stanford)**
- Gated DeltaNet (Songlin Yang, Jan Kautz, Ali Hatamizadeh; MIT/NVIDIA; arXiv 2412.06464) defines the chunkwise-parallel WY/UT-transform algorithm — [arXiv 2412.06464](https://arxiv.org/abs/2412.06464)

**llama.cpp GDN status (context)**
- The fused `GGML_OP_GATED_DELTA_NET` (PR #19504) has CPU and CUDA backends; a chunked fused GDN path was enabled (#20340); CUDA autoregressive improvements shard columns across warps to avoid spills at S_v=128 (#20391); PR #23940 (open) removes redundant copies by writing state snapshots straight into the recurrent cache — [search summary incl. llama.cpp issue #20354](https://2.zoppoz.workers.dev/https/github.com/ggml-org/llama.cpp/issues/20354); [PR #23940 mirror](https://git.huazai-net.workers.dev/ggml-org/llama.cpp/pull/23940)
- The HIP build reuses the CUDA kernel. One user report on Strix Halo (gfx1151) found it runs on the GPU but at the same ~12 t/s as the CPU fallback — [llama.cpp issue #20354 (mirror)](https://2.zoppoz.workers.dev/https/github.com/ggml-org/llama.cpp/issues/20354) (single report; not RDNA3)

### Inferences
- For Avalon, the actionable "linear attention" work is kernel engineering. Benchmark Avalon's GDN op on gfx1100 for prefill (chunked) and decode (recurrent) against the CPU path. If it is slow, write a WMMA-based chunk kernel (intra-chunk WY solve plus inter-chunk state passing, chunk 64) following the TK/HK tile design and the flash-linear-attention reference. Prefill at 131k+ is where gains would show; the decode recurrent step is memory-bound on the state (S_v x S_v per head) and probably already fine.
- Merging PR #23940 (no redundant state copies) into Avalon would help speculative decoding, since state snapshots are needed for MTP/ngram rollback (inference).

### Gaps
- No Stanford paper on gated-delta-rule kernels was found.
- No gfx1100 benchmark of llama.cpp's GDN op was found; only one gfx1151 report.

## Q3. Long-context efficiency from Stanford (KV compression, Cartridges-style offline KV, sparse attention, shared prefixes)

### Takeaway
Stanford's main long-context efficiency work is Cartridges (train a small KV cache offline per corpus) and Hydragen (shared-prefix attention for batches). Both are poor fits for a single-user agent on one 20 GB GPU. Cartridges needs gradient training through the full 125B model. Hydragen's gains come from large batches sharing a prefix, while llama.cpp prompt-cache reuse already covers the single-user case. I found no Stanford KV-quantization or sparse-attention kernel paper inside scope that applies better.

### Cited Findings
**Cartridges** (Sabri Eyuboglu, Ryan Ehrlich, Simran Arora, Neel Guha, Dylan Zinsley, Emily Liu, Will Tennien, Atri Rudra, James Zou, Azalia Mirhoseini, Christopher Re; Stanford Hazy + Scaling Intelligence, with Univ. at Buffalo; arXiv 2506.06266, Jun 2025)
- Freezes the LLM and trains a parameterized KV cache (a "Cartridge") per corpus through "self-study": the model generates synthetic conversations about the corpus, then a context-distillation objective trains the cache. Matches in-context learning quality with 38.6x less memory and 26.4x higher throughput. Extends effective context from 128k to 484k on MTOB. Cartridges load into existing inference servers and can be composed at inference time without retraining — [arXiv 2506.06266](https://arxiv.org/abs/2506.06266v3); [Hazy blog](https://hazyresearch.stanford.edu/blog/2025-06-08-cartridges); [Stanford SI page](https://scalingintelligence.stanford.edu/pubs/cartridges)
- Code: HazyResearch/cartridges (per the project page).
- Applicability: conceptually attractive for a fixed repo or doc corpus that opencode re-sends every session. In practice, self-study needs backprop through a 125B MoE whose weights don't fit on the GPU. A hybrid model also has GDN recurrent state, not only KV, so the method would need extending. **Not feasible locally**; the existing prompt-cache save/restore (slot save) gets most of the practical benefit for repeated prefixes.

**Hydragen** (Jordan Juravsky, Bradley Brown, Ryan Ehrlich, Daniel Y. Fu, Christopher Re, Azalia Mirhoseini; arXiv 2402.05099, Feb 2024)
- Splits attention into prefix and suffix parts and batches the shared-prefix attention across sequences as matrix-matrix products. Up to 32x end-to-end throughput for CodeLlama-13B; <15% throughput loss going from 1K to 16K shared prefix at large batch (baselines lose >90%); 55% less time with tree-based sharing — [arXiv 2402.05099](https://arxiv.org/abs/2402.05099)
- Applicability: we serve batch=1. The only analogue is verifying several draft branches over the same 131k prefix (tree speculation), where prefix attention could be computed once and shared, turning many GEMV-like reads of the KV cache into one GEMM. llama.cpp already processes draft tokens as a small batch over a shared KV, which gets most of this. Low incremental gain.

### Inferences
- For 131-262k contexts, the KV cost that matters comes only from the full-attention layers. q8_0/q4_0 KV plus Flash-Decoding-style split-KV is already the practical frontier for llama.cpp. Stanford work adds no cheaper drop-in technique.

### Gaps
- I did not find a Stanford sparse-attention (top-k / block-sparse inference) paper in the 2022-2026 window to evaluate. The search budget ran out before a dedicated query; coverage here is not exhaustive.

## Q4. Stanford work on AMD GPUs specifically

### Takeaway
HipKittens (Nov 2025) is the only Stanford work I found on AMD GPUs, and it targets data-center CDNA3/CDNA4 (MI300X/MI355X) only. No Stanford work targets RDNA3 or consumer Radeon.

### Cited Findings
- HipKittens targets MI355X (CDNA4) and CDNA3; consumer RDNA is not mentioned; code is at HazyResearch/HipKittens — [Hazy blog](https://hazyresearch.stanford.edu/blog/2025-11-09-hk)
- Authors include AMD engineers (Ryan Swann, Muhammad Osama, Stanley Winata), so the work is a Stanford-AMD collaboration — [Hazy blog](https://hazyresearch.stanford.edu/blog/2025-11-09-hk)
- Results were validated on CDNA3 and CDNA4 — [catalyzex: Daniel Y. Fu](https://www.catalyzex.com/author/Daniel%20Y.%20Fu)

### Inferences
- Applicability summary for Avalon, ranked:
  1. **Highest:** a custom gfx1100 GDN chunk kernel following TK/HK tile-programming lessons. This is the likely bottleneck if HIP GDN runs at CPU speed, as reported on gfx1151.
  2. **Medium:** HK-style hand-scheduled (non-wave-specialized) attention for the full-attention layers in long prefill, compared against the tile/rocWMMA/MMA paths. Gain is bounded by those layers' share of prefill time.
  3. **Low:** megakernel/launch-bubble ideas (decode dominated by CPU experts); Hydragen (batch=1).
  4. **Not applicable:** FA3/TK code (NVIDIA-only), FlashFFTConv/Hyena/M2, LoLCATs/Hedgehog/Based (training or architecture), Cartridges (needs training a 125B model).

### Gaps
- No public benchmark exists of HipKittens or any Stanford kernel on RDNA3; porting effort and gains are estimates.
