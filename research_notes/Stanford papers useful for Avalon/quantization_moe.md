# Stanford papers (Nov 2022 - Nov 2026) on quantization, compression, MoE efficiency and memory offloading, judged against Avalon

Setup we are judging against: Avalon (llama.cpp fork) serving Qwen3.8-Flash-Next (~125B MoE hybrid) as IQ3_KS GGUF (~89 GiB). Hardware is one RX 7900 XT (20 GB, ROCm 10), a Ryzen 7 5700G (AVX2, 16 threads), 57 GB RAM and NVMe. Avalon keeps hot experts in a GPU expert cache, computes missed experts on the CPU, and mmaps the rest from NVMe. KV cache is q8_0/q4_0. The workload is a single-user coding agent, so the target is batch size 1 decode latency.

Method and limits: about 18 searches and fetches (arXiv abstract pages, the Scaling Intelligence lab site, Hazy Research, Stanford EE news). arXiv abstract pages do not show affiliations. Stanford affiliation below comes from known lab membership (Ré/Hazy, Liang/CRFM, Mirhoseini/Scaling Intelligence, Pilanci/EE, Olukotun, Zaharia's group at the time) and is flagged where it is inferred rather than shown on a page.

## Q1. Quantization: what Stanford work could beat IQ3_KS on quality per bit or speed?

### Takeaway
Stanford's own quantization output is thin. The famous incoherence and lattice lineage (QuIP, QuIP#, QTIP) is Cornell (De Sa / Kuleshov), and I found no Stanford author on it. The only Stanford-led weight-quantization work is Pilanci's low-rank plus low-precision line (LPLR, CALDERA), which helps only below about 2.5 bpw. The most useful Stanford quantization result for us is a scaling-law insight, not a format.

### Cited Findings
- **QuIP# (NOT Stanford).** Tseng, Chee, Sun, Kuleshov, De Sa. ICML 2024, arXiv 2402.04396 (Feb 2024). Hadamard incoherence plus E8-lattice codebooks for 4 bits per weight or fewer. Code: github.com/Cornell-RelaxML/quip-sharp. The repo sits under the Cornell RelaxML org, and the page lists no Stanford affiliation. — [arXiv 2402.04396](https://arxiv.org/abs/2402.04396)
- **LPLR: "Matrix Compression via Randomized Low Rank and Low Precision Factorization" (Stanford-led).** Rajarshi Saha, Varun Srivastava, Mert Pilanci (Stanford EE). NeurIPS 2023, arXiv 2310.11028 (17 Oct 2023).
  - Idea: factor A ≈ L·R with both factors quantized. It sketches the column space randomly, quantizes that basis, then projects A onto the quantized basis.
  - Reported: compression as aggressive as "one bit per matrix coordinate", matching or beating traditional methods. It was tested on LLaMA-7B layers and embeddings.
  - The abstract page links no code. — [arXiv 2310.11028](https://arxiv.org/abs/2310.11028)
- **CALDERA: "Compressing Large Language Models using Low Rank and Low Precision Decomposition" (Stanford-led, with Princeton).** Saha, Sagan, Srivastava, Goldsmith (Princeton), Pilanci. NeurIPS 2024, arXiv 2405.18886 (May 2024).
  - Idea: W ≈ Q + L·R, where Q is a low-precision backbone (QuIP#), L and R are low-rank low-precision factors, and everything is fit on calibration data.
  - Reported: beats other post-training compression methods below 2.5 bits per parameter on Llama-2 7B/13B/70B and Llama-3 8B. The low-rank factors can be LoRA-fine-tuned afterwards for better zero-shot accuracy.
  - Code: github.com/pilancilab/caldera. — [arXiv 2405.18886](https://arxiv.org/html/2405.18886v2); [Stony Brook record](https://researchconnect.stonybrook.edu/en/publications/compressing-large-language-models-using-low-rank-and-low-precisio/); [Stanford EE news](https://ee.stanford.edu/leaner-large-language-models-could-enable-efficient-local-use-phones-and-laptops)
- **Scaling Laws for Precision (Stanford involvement substantial, Harvard-led).** Kumar, Ankner, Spector, Bordelon, Muennighoff, Paul, Pehlevan, Ré, Raghunathan. arXiv 2411.04330 (Nov 2024); the abstract page names no venue. Spector, Muennighoff and Ré are Stanford (inferred from lab membership).
  - Finding: post-training quantization degradation grows with the number of pretraining tokens, so more data can make a model worse once quantized.
  - Finding: the laws predict loss when different parts of a model use different precisions.
  - Basis: fit on more than 465 pretraining runs, validated up to 1.7B parameters and 26B tokens. — [arXiv 2411.04330](https://arxiv.org/abs/2411.04330)
- **FlexGen's 4-bit compression.** FlexGen compresses weights and the attention (KV) cache to 4 bits with "negligible accuracy loss", using simple group-wise quantization. — [arXiv 2303.06865](https://arxiv.org/abs/2303.06865)

### Applicability verdicts
- **QuIP#, QTIP (Cornell, out of scope for "Stanford").** For porting value: as far as I know, ik_llama.cpp's IQ*_KT "trellis" quants are already a QTIP-inspired port, and IQ3_KS is also an ik_llama type. That is from my background knowledge, not verified in this session. Since Avalon already uses IQ3_KS, the realistic QTIP-lineage upgrade is trying IQ3_KT/IQ2_KT for the routed experts, if Avalon carries those types and their ROCm/AVX2 kernels. Expected gain: lower perplexity at the same bpw, or about 0.5 bpw saved at the same quality. Trellis decode is compute-heavier, which may hurt the CPU path on the 5700G.
- **CALDERA / LPLR.** It is only competitive below about 2.5 bpw, needs QuIP# E8 decode kernels (none exist in llama.cpp/GGUF), and adds a low-rank GEMM per matrix. Porting it to GGUF would mean a new tensor type (Q backbone plus two small low-rank tensors) and ROCm kernels: weeks of work. It was only shown on dense Llama, never on MoE or hybrid models.
  - Payoff: dropping the routed experts to about 2.2-2.4 bpw would shrink the model by roughly 30% and raise the RAM/VRAM hit rate. That is not proven quality-safe for a coding agent.
  - Verdict: research-grade, low priority.
  - Cheap variant worth testing: treat the low-rank correction as a separate optional tensor applied only on the GPU for cached experts.
- **Scaling Laws for Precision.** Directly relevant guidance. Qwen-family models are trained on very many tokens, so they should be more fragile under 3-bit PTQ than older models.
  - Implication: spend bits non-uniformly. Keep attention, shared experts, router/gate, embeddings and output at higher precision. Push only the cold routed experts down.
  - Sanity check: 89 GiB for about 125B parameters is about 6.1 bpw on average, far above IQ3_KS's nominal ~3.25 bpw. Either the parameter count is closer to 230B, or a large share of tensors are already at high precision. It is worth checking the tensor-type breakdown with gguf-dump to see where the bytes go. This is arithmetic on figures from the task statement, not a sourced fact.
- **FlexGen 4-bit group-wise.** Weaker than IQ3_KS or q4_0 KV. No gain.

### Gaps
- I found no Stanford-authored KV-cache quantization paper in 2022-2026. Searches returned only Berkeley (KVQuant) and other groups.
- I found no Stanford vector/lattice quantization or low-bit training paper beyond the scaling-law work. NestQuant-type lattice work appears to be MIT/HUJI (not verified this session).
- "Decoding Compressed Trust" (ICML 2024, reportedly with Sanmi Koyejo of Stanford, comparing quantization and pruning) was not verified this session.

## Q2. MoE efficiency: offloading, caching, prefetching, pruning/merging, router prediction, serving

### Takeaway
I found no Stanford-led paper on MoE expert offloading, caching, prefetching, expert pruning or merging, or router prediction. That literature is dominated by other groups: Yandex (Eliseev & Mazur), Edinburgh (MoE-Infinity), UW (Fiddler), Berkeley (MoE-Lightning) and many Chinese and Korean groups. The Stanford-involved MoE work is MegaBlocks (kernels, training-focused) and SambaNova SN40L / Composition of Experts (a three-tier memory hierarchy, conceptually close to Avalon).

### Cited Findings
- **MegaBlocks: Efficient Sparse Training with Mixture-of-Experts (Stanford-led).** Trevor Gale, Deepak Narayanan, Cliff Young (Google), Matei Zaharia. arXiv 2211.15841 (29 Nov 2022); published at MLSys 2023 (my knowledge).
  - Idea: reformulate MoE as block-sparse GEMM, so there is no token dropping or padding to a fixed capacity.
  - Reported: up to 40% end-to-end training speedup over Tutel, and 2.4x over dense DNNs trained with Megatron-LM.
  - Code: the abstract page links none. The project is known as github.com/databricks/megablocks (formerly stanford-futuredata); not verified this session. — [arXiv 2211.15841](https://arxiv.org/abs/2211.15841)
- **SambaNova SN40L: Scaling the AI Memory Wall with Dataflow and Composition of Experts (Stanford involvement through Kunle Olukotun, last author and SambaNova co-founder; industry-led).** Prabhakar et al. (30 authors). MICRO 2024, arXiv 2405.07518 (May 2024).
  - Idea: three memory tiers (on-chip SRAM, on-package HBM, off-package DDR). Samba-CoE holds 150 experts and about 1T total parameters in DDR, and switches them into HBM on demand.
  - Reported: 15-31x faster model switching, up to 19x smaller machine footprint, and 3.7x over DGX H100 (6.6x over DGX A100) for CoE inference. — [arXiv 2405.07518](https://arxiv.org/abs/2405.07518)
- **Non-Stanford reference points.** These came up in searches, for context only:
  - Eliseev & Mazur (arXiv 2312.17238): the LRU expert cache plus speculative prefetching baseline. — [arXiv 2312.17238](https://arxiv.org/abs/2312.17238)
  - A Nov 2025 caching and prefetching analysis built on that baseline. — [arXiv 2511.05814](https://arxiv.org/pdf/2511.05814)
  - fMoE fine-grained expert maps. — [arXiv 2502.05370](https://arxiv.org/html/2502.05370v1)
  - SSD-LLaMA: SSD-native trillion-parameter MoE on a consumer PC, with SSD holding the full expert pool and VRAM/RAM as bounded caches. — [arXiv 2609.18110](https://arxiv.org/html/2609.18110v1)
  - An energy study: SSD offload of DeepSeek-R1 experts costs about 4.9x the energy per token of HBM-resident weights, and 3.1x that of CPU-RAM offload. — [alphaXiv 2508.06978](https://www.alphaxiv.org/abs/2508.06978)
  - REAM (2026): expert merging beats pruning, building on REAP. — [arXiv 2604.04356](https://arxiv.org/abs/2604.04356v1)

### Applicability verdicts
- **MegaBlocks.** CUDA/Triton and training-oriented. At batch size 1 decode, llama.cpp's mul_mat_id already does the useful part, so there is no block-sparse padding problem to solve. Possible relevance is only to prompt prefill for long agent contexts, where grouped expert GEMM matters, and ggml already groups by expert. Port gain: about 0. Skip.
- **SambaNova CoE.** It validates Avalon's architecture (DDR to HBM expert switching is the same idea as NVMe/RAM to VRAM), but it is custom hardware with no portable algorithm. No direct gain. Its main lesson matches Avalon's: keep the switching unit large and contiguous to amortize transfer cost.
- **For expert caching, prefetching and router prediction, the useful ideas are non-Stanford.** Examples: next-layer gate prediction from the current hidden state (Eliseev & Mazur, Pre-gated MoE), fMoE expert maps, MoBiLE big-little experts, and SSD-LLaMA tiering. They are the right papers for Avalon, but out of scope for this Stanford list. Flag to the report writer: there is a Stanford gap here.

### Gaps
- I found no Stanford paper on expert pruning or merging, router prediction, or MoE quantization (for example QMoE is ISTA, MoE-Pruner is not Stanford). Searches with Stanford in the query returned only non-Stanford work. A Semantic Scholar affiliation filter was not available, so a small Stanford co-authored paper could have been missed.

## Q3. Memory offload for models bigger than RAM/VRAM (FlexGen and successors, SSD offload)

### Takeaway
FlexGen (Stanford-led, 2023) is the canonical Stanford offload paper. It is designed for throughput with huge effective batches, not single-user latency, so its scheduling policy does not help Avalon. H2O and Cartridges (Stanford-involved) compress the KV cache, but our hybrid model's KV is already small and a coding agent needs exact recall.

### Cited Findings
- **FlexGen: High-Throughput Generative Inference of LLMs with a Single GPU (Stanford-led).** Ying Sheng (Stanford) first author, plus Liang, Ré, Barrett and others, with Berkeley, CMU, ETH, Yandex and Meta. arXiv 2303.06865 (Mar 2023); ICML 2023 (my knowledge; the abstract page names no venue).
  - Idea: an LP-optimized offloading policy over GPU, CPU and disk, a zig-zag block schedule that reuses each loaded weight across many batches, and 4-bit weight/KV compression.
  - Reported: OPT-175B on a single 16 GB GPU at 1 token/s with an effective batch of 144, and a 30B model over 7 HELM sub-scenarios in 21 h.
  - Code: github.com/FMInference/FlexGen. — [arXiv 2303.06865](https://arxiv.org/abs/2303.06865)
- **H2O: Heavy-Hitter Oracle (Stanford co-authored; UT Austin led; Sheng, Ré and Barrett are Stanford).** Zhang et al. arXiv 2306.14048 (Jun 2023); NeurIPS 2023 (my knowledge).
  - Idea: KV eviction that keeps about 20% heavy-hitter tokens plus recent tokens.
  - Reported: up to 29x throughput over DeepSpeed Zero-Inference and HF Accelerate, up to 3x over FlexGen, and up to 1.9x lower latency at the same batch size.
  - Code: github.com/FMInference/H2O. — [arXiv 2306.14048](https://arxiv.org/abs/2306.14048)
- **Cartridges (Stanford-led: Eyuboglu, Ré, Mirhoseini, Zou).** arXiv 2506.06266 (Jun 2025).
  - Idea: train a small KV cache offline per corpus ("self-study") instead of putting the corpus in context.
  - Reported: matches ICL quality with 38.6x less memory and 26.4x higher throughput, and extends effective context from 128k to 484k tokens on MTOB. — [arXiv 2506.06266](https://arxiv.org/abs/2506.06266)
- **Intelligence Per Watt (Hazy Research, Nov 2025).** A local-inference efficiency study. It touches quantization only as one factor. It reports that local LMs on local accelerators accurately answer 88.7% of single-turn chat and reasoning queries. — [Hazy blog](https://hazyresearch.stanford.edu/blog/2025-11-11-ipw)

### Applicability verdicts
- **FlexGen.** Its gains come from amortizing one weight load over a batch of about 144. At batch size 1 every token still has to stream every missed expert, so the zig-zag schedule gives about 0 gain.
  - Transferable idea 1: the cost model (an LP over placement fractions per tier, given PCIe, NVMe and RAM bandwidth and compute) could size Avalon's GPU expert cache and pinned-RAM tier more systematically.
  - Transferable idea 2: overlap disk to RAM to GPU transfer with compute.
  - Speculative idea: for an agent that runs parallel tool calls or speculative branches, a small batch (2-8) would let one expert load serve several sequences. That is FlexGen's insight at small scale.
- **H2O.** Qwen3.x-Next-style hybrids have few full-attention layers (the rest are linear/DeltaNet-style), so the KV cache is already small at q8_0/q4_0. Eviction risks dropping exact code tokens the agent needs later. Low gain, real quality risk; not recommended.
- **Cartridges.** Needs per-corpus training on the serving model, so it is infeasible locally for a ~125B model. A possible long-term idea for a fixed repo is out of scope. No gain.

### Gaps
- I found no Stanford SSD-offload or NVMe-streaming paper after FlexGen. SSD-native MoE work in 2025-2026 (SSD-LLaMA and others) did not show Stanford authors.

## Q4. Sparsity and low-rank methods for faster CPU+GPU hybrid inference

### Takeaway
Contextual activation sparsity is the most portable Stanford-involved idea for Avalon: CATS (Stanford-led, 2024) and Deja Vu (Stanford co-authored, 2023). On CPU-computed experts, skipping inactive SwiGLU neurons cuts the down-projection work. If the down-projection weights were stored neuron-major, it could also cut the bytes read from RAM or NVMe.

### Cited Findings
- **CATS: Contextually-Aware Thresholding for Sparsity in LLMs (Stanford-led, Scaling Intelligence Lab / Azalia Mirhoseini).** Lee et al. COLM 2024, arXiv 2404.08763 (Apr 2024, v4 Nov 2024).
  - Idea: a per-block threshold on |SiLU(gate)| in gated MLPs, calibrated on about 500 samples to a target sparsity. Only the active neurons' up/down rows are computed.
  - Reported: 50% MLP activation sparsity with negligible downstream drop on Mistral-7B and Llama-2-7B. It beats ReLUfication, and a custom GPU kernel gives about 15% faster token-generation latency.
  - Code: Scaling Intelligence GitHub (ScalingIntelligence/CATS, not verified this session). — [arXiv 2404.08763](https://arxiv.org/html/2404.08763v4); [Stanford blog](https://scalingintelligence.stanford.edu/blogs/cats)
- **Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time (Stanford co-authored; Rice/CMU led; Tri Dao and Ré were Stanford then).** Liu et al. ICML 2023, arXiv 2310.17157.
  - Idea: small lookahead predictors choose which attention heads and MLP neurons to compute per token.
  - Reported: more than 2x lower latency than FasterTransformer and more than 6x lower than HF on OPT-175B, without quality loss.
  - Code: github.com/FMInference/DejaVu. — [arXiv 2310.17157](https://arxiv.org/abs/2310.17157)
- **LPLR / CALDERA (low-rank, see Q1).** These are low-rank plus quantization for compression, not for speed.

### Applicability verdicts
- **CATS applied to CPU-computed (missed) experts. Best Stanford candidate; moderate gain, moderate effort.**
  - How it maps: Qwen experts are SwiGLU. Compute gate first, threshold, then compute up and down only for surviving neurons.
  - Up-proj saving: in GGUF, up and gate are row-major in n_ff, so skipping rows of up is easy and block-aligned. That saves about 50% of up-proj work.
  - Down-proj saving: needs the transposed layout (n_ff-major) to skip whole quant blocks. That is a re-pack of the down tensors at load time, which also lets skipped neurons' weights never be paged in from NVMe.
  - Estimate at 50% sparsity: about 1/3 fewer bytes touched and FLOPs per CPU-computed expert (gate 100%, up 50%, down 50%), so roughly 20-30% faster CPU expert time and fewer NVMe page-ins.
  - Risks: fine-grained MoE experts (small n_ff) may be less sparse than dense 7B MLPs, which is untested. Coding-agent quality impact is unknown; validate with perplexity and a coding eval.
  - GPU-side gain: smaller. The paper reports only about 15% on dense models, and hipBLAS/MMQ kernels would need sparse variants.
  - Effort: on the ggml-cpu AVX2 path, a custom gather-matvec for IQ3_KS blocks is about 1-2 weeks.
- **Deja Vu predictors.** Needs trained per-layer predictors, and attention-head sparsity is out of scope (attention). MoE routing already gives coarse sparsity. The CATS threshold is simpler and training-free, so prefer CATS. Predictor-based neuron prefetch could in principle guide which expert rows to prefetch from NVMe, but that is speculative.

### Gaps
- I found no Stanford paper evaluating activation sparsity on MoE experts or on hybrid models. TEAL (training-free activation sparsity, 40-50%) is MIT/Together/Princeton, not Stanford, and was not verified this session.
- No Stanford low-rank-for-speed paper (as opposed to for compression) was found in range.

## Summary table (for the report writer)

| Paper | Stanford role | Date / venue | Port to llama.cpp/ROCm? | Expected gain for Avalon |
|---|---|---|---|---|
| CATS | Led (Mirhoseini lab) | Apr 2024, COLM 2024 | Yes, CPU AVX2 gather kernel plus down-proj repack | ~20-30% faster CPU expert compute and fewer NVMe reads, if MoE experts are sparse enough (unverified) |
| Scaling Laws for Precision | Major co-authors (Ré, Spector, Muennighoff) | Nov 2024, arXiv | N/A (guidance) | Better bit allocation: high bits for shared/attention/router, low bits for cold experts |
| CALDERA / LPLR | Led (Pilanci) | NeurIPS 2024 / NeurIPS 2023 | Hard (new tensor type, needs E8/QuIP# kernels) | Only below 2.5 bpw; possible ~30% size cut at quality risk; low priority |
| FlexGen | Led (Sheng, Liang, Ré) | Mar 2023, ICML 2023 | Ideas only | ~0 at batch 1; cost-model and overlap ideas; small batching of parallel agent branches |
| Deja Vu | Co-authors (Dao, Ré) | ICML 2023 | Needs trained predictors | Superseded by CATS for our case |
| H2O | Co-authors (Sheng, Ré, Barrett) | Jun 2023 (NeurIPS 2023) | Easy-ish | Low; hybrid KV is already small; recall risk for code |
| Cartridges | Led (Ré, Mirhoseini) | Jun 2025, arXiv | No (needs training) | None locally |
| MegaBlocks | Led (Gale, Zaharia) | Nov 2022, MLSys 2023 | Not needed | ~0 at batch 1 decode |
| SambaNova SN40L CoE | Olukotun (industry-led) | MICRO 2024 | No (hardware) | Conceptual validation of tiered expert memory |
| QuIP# / QTIP | NONE (Cornell) | ICML 2024 / NeurIPS 2024 | IQ*_KT in ik_llama (unverified) | Try IQ3_KT/IQ2_KT on routed experts if Avalon carries them |
