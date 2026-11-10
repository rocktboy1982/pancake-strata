# MIT/Harvard papers on efficient attention, linear-attention/SSM architectures and GPU kernels (Nov 2022 - Nov 2026), and whether pancake/Avalon can use them

Scope note: this covers papers on attention, linear attention/SSMs and kernels only. Quantization, MoE offload, decoding/serving and agent papers belong to other researchers. Stanford-led items (FlashAttention-2, ThunderKittens, Based/Zoology/JRT, LoLCATs, Hyena family, CATS) are already in `/home/dan/reports/Stanford papers useful for Avalon.md` and are not repeated.

Local context checked for the verdicts. The Strata tree at `/home/dan/clone-of-d/src/Strata/src/prefill/` already has a chunked GDN prefill:
- `gdn_chunk.cu` has three kernels (cumsum / kkt / fwd). Their header comments say they follow the same three stages as llama.cpp-v100's gdn-chunk-sm70.
- There is a gfx11 WMMA chunked path, `STRATA_GDN_CHUNKED=1`. Per the `gdn_wmma12.inc` comment it uses chunks of 16, FP16 operands, FP32 accumulate and state, a fused norm, and T and P from a prep kernel.
- There is also a gfx12 port of the same algorithm (`gdn_wmma12.inc`) and an sm_80 tensor-core scan (`gdn_tc.cuh`).

So the basic FLA-style chunked algorithm is already ported. The remaining question is how to tune it.

Lead/co flag: **LEAD** means the first or senior author is at MIT or Harvard. **CO** means MIT/Harvard people are co-authors and another institution leads.

## Q1: Gated DeltaNet / DeltaNet / flash-linear-attention (Songlin Yang, Yoon Kim et al., MIT): algorithms, kernels, and what a fast gfx1100 GDN kernel should look like

### Takeaway
The MIT Yang/Kim line (GLA, then parallel DeltaNet, then Gated DeltaNet, plus the FLA library) defines the algorithm pancake already runs for the model's GDN layers: a chunkwise WY/UT-transform delta rule with a gated cumulative decay. Pancake already has a gfx11 WMMA chunked kernel, so these papers bring no new algorithm. What remains is tuning, mainly a larger chunk with blocked triangular inversion and more kernel fusion. GDN is about 22% of prefill, so even a 1.5x faster GDN kernel saves only about 7% of total prefill time (inference). The follow-ups (PaTH, Log-Linear, stick-breaking, KL-guided hybrid distillation) all need a retrained model and do not apply.

### Cited Findings

**Gated Linear Attention (GLA) and FlashLinearAttention: LEAD (MIT)**
- Paper: "Gated Linear Attention Transformers with Hardware-Efficient Training". Authors: Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, Yoon Kim. Venue: ICML 2024. Link: [arXiv 2312.06635](https://arxiv.org/abs/2312.06635).
- Core idea: a chunkwise form of linear attention. Within a chunk it uses a dense, matmul-friendly parallel form. Across chunks it passes a recurrent state.
- The paper introduces FLASHLINEARATTENTION, an I/O-aware kernel. The authors report it is faster than FlashAttention-2 as a standalone layer even at short lengths such as 1K tokens ([arXiv 2312.06635](https://arxiv.org/abs/2312.06635)).
- Code: fla-org/flash-linear-attention.
- Verdict: no new algorithm for us. It is the conceptual basis of pancake's chunked kernels, which already exist.

**Parallelizing Linear Transformers with the Delta Rule over Sequence Length: LEAD (MIT)**
- Authors: Yang, Wang, Zhang, Shen, Kim. Venue: NeurIPS 2024. Link: [arXiv 2406.06484](https://arxiv.org/html/2406.06484v2).
- Core idea: it makes the DeltaNet recurrence S_t = S_{t-1}(I - β k kᵀ) + β v kᵀ parallel within a chunk. It does this with a memory-efficient WY representation of the product of Householder-like matrices, so the per-token states never have to be materialized.
- A chunk then becomes one triangular solve plus dense matmuls ([arXiv 2406.06484](https://arxiv.org/html/2406.06484v4)).
- This is exactly the scheme pancake's `gdn_chunk.cu` describes: "a 64-token chunk becomes one triangular solve and a batch of matmuls instead of 64 sequential rank-1 steps" (local source comment).
- Verdict: already ported.

**Gated Delta Networks: Improving Mamba2 with Delta Rule: LEAD (first author Songlin Yang, MIT; co-authors Jan Kautz and Ali Hatamizadeh, NVIDIA)**
- Venue: ICLR 2025. v1 posted 9 Dec 2024, v3 on 6 Mar 2025. Link: [arXiv 2412.06464](https://arxiv.org/abs/2412.06464).
- Core idea: it combines Mamba2's scalar gating (fast forgetting) with the delta rule (precise overwrite). The abstract claims Gated DeltaNet "consistently surpasses" Mamba2 and DeltaNet. The abstract gives no numbers, and the arXiv page shows no code link ([arXiv 2412.06464](https://arxiv.org/abs/2412.06464)).
- This is the GDN layer type in Qwen3-Next-style hybrids, so it is our model's layer.

**flash-linear-attention (FLA) library and portability to AMD**
- The FLA README says AMD GPUs need the Triton ROCm backend ([PyPI flash-linear-attention](https://pypi.org/project/flash-linear-attention/)).
- AMD's Primus guide validated a 300M pure-GDN model on 8x MI300X against FLA. Per-iteration time was within about 0.2%, throughput was 182,729 tok/s/GPU for FLA versus 183,213 for Primus, and Primus reached 642 TFLOP/s/GPU ([ROCm Primus GDN guide](https://rocm.docs.amd.com/projects/primus/en/latest/04-technical-guides/hybrid-models/gdn-guide.html)).
- AMD's companion KDA guide reports that AMD Triton 3.6 cannot compile the num_stages = 4 variants of FLA's KDA intra-chunk kernels. Primus keeps only the autotune configs that compile ("kept 4/12") ([ROCm Primus KDA guide](https://rocm.docs.amd.com/projects/primus/en/latest/04-technical-guides/hybrid-models/kda-guide.html)).
- A third-party Hugging Face repo, pat883/gdn-hip, offers native HIP + rocWMMA GDN kernels built for RDNA4 (gfx1201) ([HF gdn-hip](https://huggingface.co/pat883/gdn-hip)). It is a direct peer to pancake's gfx12 path and could be a benchmark reference. I did not check whether it supports gfx1100.

**PaTH Attention: LEAD (MIT; Yang and Kim, with IBM co-authors)**
- Venue/date: arXiv May 2025. Link: [arXiv 2505.16381](https://arxiv.org/abs/2505.16381v1).
- Core idea: a data-dependent position encoding built from accumulated identity-plus-rank-one (Householder-like) transforms. The paper gives a FlashAttention-style parallel algorithm using a compact representation of Householder products.
- At 760M parameters it beats RoPE and the Forgetting Transformer (FoX). Code is in FLA ([arXiv 2505.16381](https://arxiv.org/pdf/2505.16381); [HF](https://huggingface.co/papers/2505.16381)).
- Verdict: architecture change requiring training. Not applicable.

**Log-Linear Attention: LEAD (Han Guo, Songlin Yang, ..., Tri Dao, Yoon Kim; MIT)**
- Venue/date: ICLR 2026; arXiv June 2025, revised March 2026. Link: [arXiv 2506.04761](https://arxiv.org/abs/2506.04761v2).
- Core idea: it replaces the fixed-size state with a logarithmically growing set of states (Fenwick-tree style). It is applied on top of Mamba-2 and Gated DeltaNet ([mlanthology ICLR 2026](https://mlanthology.org/iclr/2026/guo2026iclr-loglinear/)).
- Verdict: needs a retrained model. Not applicable to the shipped Qwen weights.

**Distilling to Hybrid Attention Models via KL-Guided Layer Selection: LEAD (Yanhong Li, Songlin Yang, ..., Yoon Kim)**
- Venue/date: ICLR 2026; arXiv Dec 2025. Link: [arXiv 2512.20569](https://arxiv.org/abs/2512.20569v1).
- Core idea: it chooses which softmax layers to convert to linear attention, using importance scores from a small amount of training on generic text ([mlanthology](https://mlanthology.org/iclr/2026/li2026iclr-distilling/)).
- Verdict: needs training of a 125B model. Not applicable.

**Kimi Linear (KDA): CO (Moonshot-led; Yang is a co-author, Kim is not)**
- Link: [arXiv 2510.26692](https://arxiv.org/abs/2510.26692).
- Core idea: it extends Gated DeltaNet with finer-grained, channel-wise gating ([alphaxiv](https://alphaxiv.org/abs/2510.26692)).
- Verdict: relevant only if a future model uses KDA. Our model uses GDN.

### Inferences
- **What a fast gfx1100 GDN prefill kernel should look like** (derived from the GLA/DeltaNet chunkwise formulation and pancake's current design):
  1. **Use a larger chunk with blocked inversion.** Pancake's gfx11 path uses chunk 16. The WY/UT method lets you use C = 64. Form the 64x64 strictly-lower A = β·exp(Δg)·KKᵀ, then invert (I + A) block-wise from 16x16 diagonal blocks, with the off-diagonal blocks done by WMMA matmuls. This cuts the serial inter-chunk scan steps by 4x. Per step the cost is a 128x128 state times a 64-wide matmul, which suits 16x16x16 WMMA. Trade-off: more LDS (64x128 FP16 K/Q tiles plus 64x64 A/P) and FP16 range issues in the exp(Δg) factors. The decay-from-later-to-earlier form already in `gdn_chunk.cu` keeps the factors at or below 1, so that is safe.
  2. **Fuse the prep kernels.** cumsum, KKᵀ, the solve and W/U recompute can run as one kernel per (chunk, head). They are embarrassingly parallel across chunks. Only the state pass is sequential.
  3. **Keep the state in VGPRs.** gfx11 has a large register file. Each wave should own a 16- or 32-column slice of S, with state, accumulate and decay in FP32 and the matmul operands in FP16. That is already the gfx11 design.
  4. **Overlap the next chunk's global loads with the current chunk's WMMA work.** Pancake's comments show this is already partly done via the LDS-only barrier.
  - Expected gain: unmeasured. If GDN is about 22% of prefill and a C=64 kernel gives 1.3-2x on GDN, total prefill improves about 5-11%.
- **Decode:** GDN decode is a memory-bound rank-1 update of a 128x128 FP32 state per head. The papers offer nothing beyond keeping the state resident and fusing conv, norm, gates and update in one kernel. This is likely a small part of the 15.9 ms GPU decode window.
- For a hybrid with a 3:1 GDN:full ratio (assumed, as in Qwen3-Next), the GDN state does not grow with context. Attention, not GDN, is what grows at 131-262k, so attention kernels (Q2) matter more at the longest prompts.

### Gaps
- I did not verify FLA's exact kernel breakdown or chunk sizes for `chunk_gated_delta_rule` in the current repo (expected: chunk 64, a solve_tril step, recompute_w_u, then fwd_h/fwd_o). This should be read from the code in fla-org/flash-linear-attention before redesigning.
- I found no published GDN kernel benchmark on gfx1100/RDNA3. AMD's validation is MI300X training; gdn-hip targets RDNA4.
- I found no 2026 Yang/Kim paper specifically on GDN inference kernels.

## Q2: MIT Han Lab efficient attention for long context (StreamingLLM, DuoAttention, LServe, XAttention, Quest, FlashMoBA): usable for exact or near-exact agent coding?

### Takeaway
All Han Lab long-context methods are approximate: they evict KV, sparsify KV, or skip blocks. None is exact. For a coding agent that must recall exact strings anywhere in a 131-262k prompt, the only candidates worth trying are training-free, query-aware block selection that keeps all KV: **XAttention** for prefill and **Quest** for decode. They should be opt-in modes for prompts above roughly 64k, gated by a needle and code-retrieval regression test. StreamingLLM and the eviction-style parts of DuoAttention drop tokens, so they are unsuitable. DuoAttention's calibration and FlashMoBA both need training or optimization passes that are impossible on a 125B model locally.

### Cited Findings

**StreamingLLM ("Efficient Streaming Language Models with Attention Sinks"): LEAD (Guangxuan Xiao, MIT; with Meta)**
- Venue: ICLR 2024. Link: [arXiv 2309.17453](https://arxiv.org/abs/2309.17453).
- Core idea: keep a few initial "sink" tokens plus a recent window, so a model can stream indefinitely.
- Xiao has since finished his MIT PhD under Song Han and is now at Thinking Machines Lab ([homepage](https://dr.molodetz.nl/_proxy/https/guangxuanx.com/)).
- Verdict: **No.** It discards the middle of the context, which is exactly what a coding agent needs.

**DuoAttention: LEAD (Xiao, Tang, Zuo, Guo, Yang, Tang, Fu, Han)**
- Date: arXiv 14 Oct 2024; the arXiv page lists no venue. Link: [arXiv 2410.10819](https://arxiv.org/abs/2410.10819).
- Core idea: only some heads are "retrieval heads" that need full KV. The rest are "streaming heads" that get sink + recent KV only.
- Reported numbers ([arXiv 2410.10819](https://arxiv.org/abs/2410.10819)):

| Metric | MHA models | GQA models |
|---|---|---|
| Memory reduction | up to 2.55x | up to 1.67x |
| Decode speedup | up to 2.18x | up to 1.50x |
| Prefill speedup | up to 1.73x | up to 1.63x |

- With quantization, Llama-3-8B runs at 3.3M context on one A100.
- Code: [mit-han-lab/duo-attention](https://github.com/mit-han-lab/duo-attention).
- Verdict: **Low.** The GQA gains are the smaller ones, and our hybrid has few full-attention layers. Head identification needs an optimization pass over the model.

**Quest (Query-Aware Sparsity): LEAD (Jiaming Tang, ..., Guangxuan Xiao, Baris Kasikci, Song Han)**
- Venue: ICML 2024. Link: [arXiv 2406.10774](https://arxiv.org/abs/2406.10774).
- Core idea: store per-page min/max of keys, use them to upper-bound each page's attention score against the current query, and load only the top-K pages during decode.
- Reported numbers: "up to 2.23x self-attention speedup" and a 7.03x reduction in inference latency ([arXiv 2406.10774](https://arxiv.org/abs/2406.10774)).
- Code: [mit-han-lab/Quest](http://github.com/mit-han-lab/Quest).
- Verdict: **Possible but low gain.** It is training-free and keeps all KV, so it can be dialed toward exact with a large K. But decode attention covers only the full-attention layers and is not the decode bottleneck (CPU experts are). Expected end-to-end decode gain at 131k+ is a few percent (inference).

**LServe: LEAD (Shang Yang, Junxian Guo, ..., Song Han; MLSys 2025)**
- Link: [arXiv 2502.14866](https://arxiv.org/abs/2502.14866).
- Core idea: unified block-sparse serving that combines static sparsity (streaming heads) with dynamic page selection.
- Reported numbers: average 2.4x faster prefill and up to 3.3x faster decode than state-of-the-art serving systems on Llama-3-8B. The paper finds that a constant number of KV pages preserves long-context ability regardless of length ([hanlab projects](https://hanlab.mit.edu/projects)).
- Verdict: an ideas source (hierarchical page selection, block-sparse kernel), not a drop-in. It is approximate.

**XAttention: LEAD (Ruyi Xu, Guangxuan Xiao, Haofeng Huang, Junxian Guo, Song Han; ICML 2025)**
- Link: [arXiv 2503.16428](https://arxiv.org/html/2503.16428v1).
- Core idea: score each (query block, key block) pair cheaply by summing strided antidiagonal elements, then run block-sparse attention on the selected blocks. Training-free.
- Reported numbers: up to 13.5x faster attention at 256K versus FlashAttention ([mlanthology](https://mlanthology.org/icml/2025/xu2025icml-xattention); [NVIDIA EAI](https://research.nvidia.com/labs/eai/publication/xattention)).
- Verdict: **the most relevant Han Lab item for us**, because prefill at 131-262k is where full attention becomes the top cost. It is still approximate.

**FlashMoBA ("Optimizing Mixture of Block Attention"): LEAD (Xiao, Junxian Guo, Kasra Mazaheri, Song Han; MIT and NVIDIA)**
- Date: Nov 2025. Link: [arXiv 2511.11571](https://arxiv.org/abs/2511.11571v1).
- Core idea: a CUDA kernel that makes small-block MoBA practical. It reports up to 14.7x speedup over FlashAttention-2 ([arXiv 2511.11571](https://arxiv.org/pdf/2511.11571)).
- It needs models trained with MoBA (the paper also uses a key convolution). Code: mit-han-lab/flash-moba.
- Verdict: **No.** It needs a model trained with MoBA.

**Han Lab projects page (checked 2026-11-09)**
- The current first page shows no new 2026 LLM-attention project. It lists Quant VideoGen (ICML 2026, video KV quantization), SMEPO and Lightning OPD/Weave ([hanlab.mit.edu/projects](https://hanlab.mit.edu/projects)).
- Radial Attention is for video generation ([hanlab projects](https://hanlab.mit.edu/projects)). Not applicable.

### Inferences
- **XAttention port sketch (pancake/Avalon fattn on gfx1100):**
  1. Compute antidiagonal block scores with a cheap strided pass over Q and K at block size 64-128.
  2. Keep the blocks whose cumulative softmax mass exceeds a threshold τ. Always keep the diagonal and sink blocks.
  3. Pass a block mask into the existing flash-attention tile loop, which skips the masked K tiles.
  - Effort: moderate. No new math kernel is needed, only scoring plus tile skipping.
  - Gain: attention is 16% of prefill at 4-32K and grows roughly linearly with context per token. At 131-262k it could be the largest prefill component, so a 3-5x reduction in attention FLOPs there might cut 20-40% of prefill time at the longest prompts. This is an unmeasured estimate. Make it opt-in with a conservative τ.
  - Risk: missing exact-match tokens in code. Validate with needle and code-retrieval tests on real opencode transcripts.
- Combining with prefix caching (Stanford report) removes most repeated prefill anyway. Block-sparse prefill helps mainly cold or long new segments.
- DuoAttention-style head pruning would interact badly with a hybrid whose remaining full-attention layers exist specifically for recall (the Stanford Based/Zoology rationale).

### Gaps
- I did not verify the Qwen3.8-Flash-Next layer count, full-attention ratio or KV head count. The gain estimates assume a Qwen3-Next-like 3:1 layout.
- I found no evaluation of XAttention or Quest on coding-agent or repository-level exact-recall tasks.
- I did not verify a ROCm port of mit-han-lab/Block-Sparse-Attention.

## Q3: Other MIT/Harvard kernel or architecture work (TEAL, Hydra/Mamba-related, Jet-Nemotron, Harvard Kempner, DSLs)

### Takeaway
**TEAL (MIT-led) is the one directly portable idea in this question.** It is training-free magnitude thresholding of activations, giving 40-50% sparsity with 1.53-1.8x decode speedup on dense models. It overlaps with CATS in the Stanford report, but it applies to every projection, not just the MLP. For us it only pays off on the CPU-expert path, and only if the weights are re-laid out so that whole input-column blocks can be skipped. Harvard has little in this area: "Repeat After Me" explains why hybrids keep attention, and Flash Inference covers long-convolution models only. Hydra and Mamba are CMU/Princeton work, not MIT/Harvard.

### Cited Findings

**TEAL ("Training-Free Activation Sparsity in Large Language Models"): LEAD (James Liu, Han Guo, Yoon Kim at MIT; Ponnusamy and Athiwaratkun at Together AI; also Tianle Cai)**
- Venue: ICLR 2025. Link: [arXiv 2408.14690](https://arxiv.org/html/2408.14690v3).
- Core idea: zero out low-magnitude entries of the hidden states across the whole model, so the matching weight channels never need to be loaded.
- Reported numbers: 40-50% model-wide sparsity with minimal degradation on Llama-2, Llama-3 and Mistral at 7B-70B. Decode speedups are about 1.53x at 40% and 1.8x at 50% sparsity. It is compatible with weight quantization ([Together blog](https://together.ai/blog/teal-training-free-activation-sparsity-in-large-language-models); [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f3f2ff9579ba6deeb89caa2fe1f0b99c-Abstract-Conference.html)).
- Code: github.com/FasterDecoding/TEAL.

**Jet-Nemotron / PostNAS / JetBlock: CO (NVIDIA-led; Song Han is a co-author and it appears on the Han Lab site)**
- Venue: NeurIPS 2025. Link: [arXiv 2508.15884](https://arxiv.org/html/2508.15884v3).
- Core idea: start from a pretrained model, freeze the MLPs, and search over where full attention goes and which linear-attention block replaces the rest. JetBlock uses dynamic causal convolution kernels on V.
- Reported numbers: the 2B model runs about 47x faster in generation at 64K on an H100. At 256K it is 6.14x faster in prefill and 53.6x in decoding versus Qwen3-1.7B-Base ([hanlab project](https://hanlab.mit.edu/projects/jet-nemotron)).
- Verdict: **No.** It requires a newly trained model.

**Stick-breaking attention ("Scaling Stick-Breaking Attention"): CO (lead Shawn Tan of the MIT-IBM Watson AI Lab / IBM; Yang co-author)**
- Venue: ICLR 2025. Link: [arXiv 2410.17980](https://arxiv.org/abs/2410.17980).
- Verdict: architecture change requiring training. Not applicable.

**Harvard: "Repeat After Me: Transformers are Better than State Space Models at Copying": LEAD (Jelassi, Brandfonbrener, Kakade, Malach; Harvard)**
- Venue: ICML 2024. Link: [arXiv 2402.01032](https://arxiv.org/abs/2402.01032).
- Core idea: fixed-state models fail at copying and retrieval from long context. This is the theoretical reason the hybrid keeps full-attention layers, and an argument against sparsifying those layers aggressively.
- Verdict: guidance only.

**Harvard: Flash Inference (Oncescu, Purandare, Idreos, Kakade): LEAD**
- Link: [arXiv 2410.12982](https://arxiv.org/html/2410.12982v2).
- Core idea: quasilinear exact inference for long-convolution sequence models only ([search summary of arXiv 2410.12982](https://arxiv.org/html/2410.12982v2)).
- Verdict: **No.** GDN uses only a short conv1d.

**Harvard Kempner**
- The only related item found is a Tri Dao seminar on hardware-aware architectures (April 2025) ([Kempner event](https://kempnerinstitute.harvard.edu/events/tri-dao/)).
- I found no Kempner-led linear-attention kernel paper.

### Inferences
- **TEAL on our engine:**
  - On the GPU, expert GEMVs at IQ2_XS/IQ3_KS are already small. TEAL's speedup comes from skipping weight columns, but GGUF k-quant blocks pack 256 consecutive input channels. Scattered 50% sparsity inside a block saves almost no bytes unless the activations are permuted or blocks are skipped only when the whole block is near zero.
  - On the CPU experts (AVX2), a re-pack at load time to an input-channel-major layout with small sub-blocks would let skipped channels save bandwidth.
  - Gain: unmeasured. If CPU experts are about 35% of the 24.8 ms decode window and 40% sparsity yields 25-30% fewer bytes there, decode might improve 8-10%.
  - This is the same engineering as the Stanford report's CATS item. Do them as one project, and measure the |x| distributions on Qwen experts first. Fine-grained MoE experts may be less sparse than the dense 7B-70B models TEAL tested (untested).
- MIT kernel DSLs, Triton-like: nothing verified in this session (see Gaps). ThunderKittens (Stanford) is already covered.

### Gaps
- MIT Exo 2 (Ragan-Kelley group, a user-schedulable kernel DSL) may be relevant as a kernel DSL, but I did not verify it in this session. No source is cited, so it is excluded from the findings.
- Hydra (bidirectional SSM) and Mamba/Mamba-2 are CMU/Princeton-led. I found no MIT/Harvard authorship, so they are excluded.
- I found no MIT/Harvard FlashInfer-style serving-kernel paper (FlashInfer is UW/CMU-led) and no MIT/Harvard RDNA3-specific kernel work.
- I found no results on TEAL for MoE models or for k-quant block formats.
