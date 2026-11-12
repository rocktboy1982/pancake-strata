# IST Austria (DASLab) and Central/Western European AI work relevant to pancake / Avalon (Nov 2022 - Nov 2026)

Scope notes: "LEAD" = first/corresponding author group at the named institution; "CO" = co-author only. ETH/EPFL/UK excluded (GPTQ-2D has an ETH co-author, but ISTA leads it). Prior notes in /home/dan/reports and /home/dan/pancake-docs/research/avalon/literature/us-universities/notes already cover TEAL, CATS, Deja Vu, Quest, DuoAttention, MoE-Lightning, Fiddler, and the Eliseev/Mazur Mixtral offloading paper, so those are not repeated here. Numbers are the authors' own claims. Items marked "(from memory, not re-fetched)" are well-known abstracts whose arXiv IDs and headline claims I did not re-open in this session.

## Q1. How ISTA-DASLab made the Qwen3.8-Flash-Next IQ2_XS / Q2_0 files, and whether better 2-bit MoE quants or faster decode exist for them

### Takeaway
The daily files are **GSQ-RCO** builds from ISTA-DASLab, not classic GPTQ/AQLM/HIGGS. GSQ learns a scalar grid (Gumbel-Softmax). RCO picks one GGUF type per tensor under an exact bpw budget. DASLab's own card measures the trade-off: Q2_0 runs at 93.8 decode t/s and 367 prompt t/s versus 70.3 and 108 for IQ2_XS, for a 0.09-point task-average loss. So staying on Q2_0 is right. The most actionable new item is the **Coder variant** (Oct 2026): it prunes half the routed experts, chosen jointly by RCO on code, agentic, and vision data, so only **29.6 GB** must be resident. But it uses lookup-table (IQ) formats at ~3.5 bpw, so it needs an A/B bench on pancake before switching. Neither file can realistically be re-quantized at home: the GSQ pipeline needs 8x or more H200-class GPUs.

### Cited Findings
**The released files (the user's daily models)**
- Repo: ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF (updated 2026-09-29, ~6.8M downloads). It is a "non-uniform GGUF" of a 512-expert MoE: 48 layers, 10 active experts plus a shared expert, hybrid attention. Each tensor gets its own quant type from a gradient-based search under a size budget, and 95% of the searchable weights are in the routed experts. — [HF card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF); [HF API listing](https://huggingface.co/api/models?author=ISTA-DASLab)
- GGUF cannot express per-expert types, so the search assigns types **per layer** (one fused tensor per layer). It covers 352 tensors: 304 dense ones plus 48 fused routed-expert matrices. — [HF card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
- File sizes and quality. Shard 1 holds the weights; shard 2 is a 28.8 GB IQ4_NL per-layer n-gram embedding table (51.2B params), the same in every variant.
  - Q2_0: 2.40 bpw, 37.6 GB shard 1. ZS 78.00, AIME25 96.67, GPQA-D 89.39, LCB v6 81.14, task avg 89.07.
  - IQ2_XS: 2.50 bpw, 39.2 GB. ZS 77.16, AIME25 96.67, GPQA-D 87.37, LCB v6 83.43, task avg 89.16.
  - IQ3_XXS: 3.0 bpw, 47.0 GB. Task avg 92.57, which is 99.4% of BF16's 93.12.
  - IQ3_S: 3.5 bpw, 54.8 GB. Task avg 93.26. — [HF card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
- Speed in llama.cpp over 55 prompts in 11 categories:
  - Q2_0: 367.49 prompt t/s, 93.79 decode t/s, 6.70 s average latency.
  - IQ2_XS: 108.19 prompt t/s, 70.30 decode t/s, 12.68 s average latency.
  - Q2_0 decode stays within 92.5-94.5 t/s across categories. IQ2_XS swings from 49.4 to 94.5, "because the lookup cost varies with how much of each layer a request touches".
  - The prefill advantage reaches 9.6x on RAG and 6.2x on coding. — [HF card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
- The shard-2 n-gram table is read one row per token. DASLab recommends `-lm mmap --lazy-mode on` so it stays on SSD, and says keeping it in RAM "removes the paging cost entirely". — [HF card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
- The `ffn_down_exps` rows are 640 wide, which is not divisible by 256. That rules out every K/I-quant, so only Q2_0 (block 64) and IQ4_NL (block 32) are possible. Every layer's down_exps is Q2_0 in both 2-bit builds; IQ3_XXS lifts 18 of the 48 layers to IQ4_NL. — [HF card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
- Allocation audit files I downloaded, which ship with each GGUF:
  - **Q2_0 build:** all 144 routed-expert tensors (gate/up/down x 48) are Q2_0. Dense and shared-expert tensors mix Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS, Q5_0 and similar types. Full histogram: BF16=483, F32=292, Q2_0=202, Q3_K=91, IQ4_XS=56, Q4_K=38, Q4_0=16, Q5_K=16, Q6_K=12, IQ4_NL=7, Q5_0=7, Q8_0=2.
  - **"IQ2_XS" build:** it contains **no IQ2_XS tensors** at all. gate/up_exps are IQ2_S in 34 layers, IQ2_XXS in 11 and IQ1_M in 3. down_exps are all Q2_0. — [Q2_0 allocation](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/raw/main/tensor-allocation/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.rco-allocation.txt); [IQ2_XS allocation](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/raw/main/tensor-allocation/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.rco-allocation.txt)
- The base model is listed as **176.9B params, 354 GB in BF16** (the user's notes say ~125B). — [Coder card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)

**GSQ (the per-tensor quantizer)**
- *GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling.* Dadgarnia, Tabesh, Nikdan, Helcig, Kurtic, Kleinegger, Alistarh. ISTA (LEAD). arXiv 2604.18556, Apr 2026 (v2 May 2026). — [arXiv](https://arxiv.org/abs/2604.18556)
  - Idea: jointly learns per-coordinate grid assignments and per-group scales through a Gumbel-Softmax relaxation over the few grid levels available. It starts from GPTQ and uses the Lion optimizer.
  - Result: closes most of the gap to vector/trellis methods (QTIP, AQLM, PV-Tuning) while staying in symmetric group-wise scalar formats. It can refine public GGUF K-quants and write them back in the same format, and it scales to Kimi-K2.5. — [arXiv](https://arxiv.org/abs/2604.18556); [code](https://github.com/IST-DASLab/GSQ)
  - Llama-3.1 at 2.13 bpp, zero-shot average (8B / 70B): GSQ 68.55 / 75.57, QTIP 69.88 / 77.25, PV-Tuning 69.83 / 76.27, GPTQ 37.53 / 57.38. — [GSQ README](https://github.com/IST-DASLab/GSQ)
  - Qwen3-8B Q2_K refined from Unsloth: average 50.03 → 56.28.
  - The 2-bit codebook is `{-2,-1,0,1} × scale`.
  - Requirements: tested only on H100. Qwen3.5-122B needs 8x H200 and 397B needs 16x H200. — [GSQ README](https://github.com/IST-DASLab/GSQ)

**RCO (the type allocator, also used to pick which experts to prune)**
- *Model Compression with Exact Budget Constraints via Riemannian Manifolds.* Helcig, Alistarh. ISTA (LEAD). arXiv 2605.00649, May 2026 (v3 Sep 2026). — [arXiv](https://arxiv.org/abs/2605.00649); [code](https://github.com/IST-DASLab/RCO)
  - Idea: under a softmax relaxation, a budget constraint is a smooth manifold in logit space. RCO wraps tangent projection, binary-search retraction and momentum transport around Adam.
  - Properties: optimizes the true (non-decomposable) loss, enforces the budget exactly at every step, and handles multiple budgets with no extra hyperparameters. — [arXiv](https://arxiv.org/abs/2605.00649)
  - Its predecessors for GGUF were EvoPress (Sieberling, Kuznedelev, Kurtic, Alistarh; arXiv 2410.14649), an evolutionary per-layer search, and gptq-gguf-toolkit (GPTQ plus EvoPress non-uniform GGUF). — [EvoPress arXiv](https://arxiv.org/abs/2410.14649); [Llama-3.1-8B GGUF card mirror](https://hg.istephan-squireso.workers.dev/ISTA-DASLab/Llama-3.1-8B-Instruct-GGUF/blob/main/README.md?code=true); [IST-DASLab repos](https://github.com/IST-DASLab)

**Coder variant (new and most relevant to an opencode agent)**
- Repo: ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF, released 2026-10-19 and updated 2026-10-17 per the card. — [HF card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)
  - Pruning: keeps **256 of the 512 routed experts per layer**. RCO chose them by minimizing full-model KL on code, agentic tool-use and vision calibration data, with one budget per layer because GGUF stores one expert count.
  - Size: retained weights are ~3.5 bpw (1.89 bpw effective over the original parameters). Shard 1 is **29.6 GB**, plus the same 28.8 GB n-gram shard.
  - Quality: SWE-bench Verified 79.20 vs 82.80 for BF16 (95.7%), and LCB v6 87.43, equal to BF16, both at xhigh reasoning effort. The Oct-17 update raised SWE-bench from 75.60 and "reduced looping".
  - Caveat: a first search without vision data destroyed vision, so capabilities missing from the calibration mix degrade. — [HF card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)
- The Coder allocation file, which I downloaded, sets a target of 1.89 bpw.
  - Expert gate/up tensors are mostly **IQ2_S, IQ3_XXS and IQ3_S** (lookup-table formats), with some IQ4_XS. down_exps are mostly **IQ4_NL**, with some Q2_0.
  - The file lists experts per expert (48 x 256 entries per projection) rather than as fused per-layer tensors. Its directory is named "IQ1_M" even though the retained weights average ~3.5 bpw. — [Coder allocation](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/raw/main/tensor-allocation/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.rco-allocation.txt)

**Other DASLab quantization work, 2022-2026, and fit for our files**
- *SLQ: Statistically-Lossless Quantization of LLMs.* Helcig, Kurtic, Alistarh. ISTA (LEAD). COLM 2026. — [code/README](https://github.com/IST-DASLab/SLQ); [OpenReview](https://openreview.net/forum?id=Xq1atP3a6S)
  - Finds the lowest-bit non-uniform config that meets a fidelity target. "Task-lossless" (≥99% zero-shot recovery) needs 3.3-4.7 bpp. "Distribution-lossless", defined as **Expected Acceptance Rate** ≥0.99 against BF16, needs 5.0-6.6 bpp.
  - Zero-shot accuracy stays flat while KL to the original model is below ~0.01.
- *MatGPTQ: Post-Training Matryoshka Quantization.* Kleinegger, Crnčević, Alistarh. ISTA (LEAD). arXiv 2602.03537, Feb 2026. — [arXiv](https://arxiv.org/abs/2602.03537v1); [code](https://github.com/IST-DASLab/MatGPTQ)
  - One parent model that can be sliced to {2,3,4,6,8} bits with single-pass PTQ.
  - Within 0.7% of baseline at 4-8 bits, and +1.34% average at 3 bits over uniform quantization.
  - Ships GPU kernels and vLLM integration. HF has Qwen3-8B and Llama-3.1-8B MatGPTQ checkpoints (Oct 2026).
- *Disaggregated Quantization: Specializing LLM Prefill and Decode.* ISTA. arXiv 2609.26333. — [README](https://github.com/IST-DASLab/disaggregated-quantization); [disaggregated-llama.cpp fork](https://github.com/IST-DASLab/disaggregated-llama.cpp)
  - Idea: a separate NVFP4 "prefiller" checkpoint is trained around a frozen GGUF decoder.
  - "Offloaded disaggregated prefill" (ODP) streams the prefill weights from SSD into buffers borrowed from decode weights.
  - Qwen3.8-27B with an Unsloth IQ1_S decoder: MMLU-Pro 29.04 → 61.54%.
  - TTFT at 8K: 12.27 → 6.90 s in their llama.cpp fork on DGX Spark. Prompts shorter than 4K are slower because of SSD loading.
- *MoESQ: Hardware-Native Joint Sparse-Quantization for Trillion-Scale MoE.* Kwanhee Lee, Namhoon Lee, Alistarh. ISTA (LEAD/CO); the Lees' affiliation was not verified. arXiv 2610.02241, Sep 2026. — [arXiv](https://arxiv.org/abs/2610.02241); [HF card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-P48NVFP4-MoESQ)
  - Format: NVFP4 W4A4 plus paired-4:8 semi-structured sparsity on the routed experts.
  - Flash-Next experts: 225.0 GiB → 38.7 GiB, which is 2.75 bpw including mask and scales.
  - Quality: preserves 96.09% of performance, and is up to 4.35 points better than prior joint sparse-quant methods.
  - Speed: kernel 1.65x over the vendor kernel on B200. Blackwell SM100/SM120 only.
  - Their card also notes that for agentic tool calling, `presence_penalty` must stay at 0: with it, SWE-bench drops from 74.7% to 70.6% and missing tool arguments nearly triple.
- *GPTQ-2D: Cubic-Time Two-Sided Adaptive Rounding.* Jiale Chen, Hoefler, Alistarh. ISTA (LEAD), with ETH and Red Hat. arXiv 2607.27042.
  - Same accuracy as YAQA, with Llama-3.1-70B quantization time cut from 8.6 days to 1.6 hours on one A6000. This only changes quantization time. — [README](https://github.com/IST-DASLab/GPTQ-2D)
- *WUSH: Near-Optimal Adaptive Transforms for LLM Quantization.* ISTA (LEAD). ICML 2026, arXiv 2512.00956.
  - Closed-form, data-aware non-orthogonal blockwise transforms for joint weight-activation (MXFP4) quantization.
  - +2.8 points over Hadamard baselines with RTN, +0.7 with GPTQ. — [arXiv](https://arxiv.org/pdf/2512.00956); [ICML](https://icml.cc/virtual/2026/poster/63124)
- *QMoE: Sub-1-bit compression of trillion-parameter MoEs.* Frantar, Alistarh. ISTA (LEAD). MLSys 2024, arXiv 2310.16795.
  - SwitchTransformer-c2048: 3.2 TB → under 160 GB with custom dictionary decoding on GPU. — [arXiv](https://arxiv.org/pdf/2310.16795.pdf); [MLSys](https://proceedings.mlsys.org/paper_files/paper/2024/file/c74b624843218d9b6713fcf299d6d5e4-Paper-Conference.pdf)
- Classic DASLab papers (from memory, not re-fetched):
  - GPTQ (Frantar et al., ICLR 2023, arXiv 2210.17323)
  - SparseGPT (Frantar & Alistarh, ICML 2023, arXiv 2301.00774): 50-60% unstructured sparsity in one shot on OPT/BLOOM-175B
  - SpQR (arXiv 2306.03078)
  - QUIK (arXiv 2310.09259; W4A4 with outliers)
  - AQLM (Egiazarian et al., ICML 2024, arXiv 2401.06118; additive vector quantization with large codebooks)
  - PV-Tuning (Malinovskii et al., NeurIPS 2024, arXiv 2405.14852)
  - Marlin (Frantar et al., PPoPP 2025, arXiv 2408.11743; FP16xINT4 NVIDIA tensor-core kernel)
  - HIGGS (Malinovskii et al., NAACL 2025, arXiv 2411.17525; data-free Hadamard plus MSE-optimal grids)
  - QuEST (arXiv 2502.05003) and Quartet (arXiv 2505.14669): low-precision training
  - HF hosts HIGGS checkpoints (2024-12 / 2025-02) and AQLM 1-bit checkpoints (2025-03). — [HF API listing](https://huggingface.co/api/models?author=ISTA-DASLab)
- Other active DASLab repos in 2026: po2-kernels ("Kernels for 2-grid gemv and gemm", README empty), grouped-sparse-GEMM, qutlass (CUTLASS-based, NVIDIA), MoE-Quant ("data-aware compression of DeepSeek models", Dec 2025). — [GitHub org repo list](https://github.com/IST-DASLab)

### Inferences
**Keep Q2_0 as the daily file.** DASLab's own numbers back the user's +5-9% decode measurement. The 0.09 task-average gap is noise-level, and the IQ2_XS file is actually IQ2_S/IQ2_XXS/IQ1_M lookup-table experts. On pancake the gain is smaller than DASLab's 1.33x decode. A likely reason is that pancake's CPU expert path is RAM-bandwidth bound, so dequant cost matters less there than in their mostly-GPU llama.cpp run. The 2.0 t/s spread versus 45 t/s spread finding suggests IQ formats hurt most on long, varied coding prompts, which is exactly opencode prefill.

**The Q2_0 kernel is a cheap target:**
- Q2_0 is a symmetric 4-level integer grid, `{-2,-1,0,1} × fp16 scale` per 64 weights (about 2.25 bpw raw).
- On AVX2 with Q8 activations, the dot product can be written as `Σ(q_u·a) − 2·Σa`, with q_u in {0..3} unpacked by shift and mask and fed to `_mm256_maddubs_epi16`. Σa per 64-block can be precomputed once per token and shared across all 10 experts and all three projections.
- pancake already has `src/kernels/cpu/q2_bitplane_parity.cpp` and a native expert path. Check that it uses this offset trick and reuses per-block Σa, and that it does no table lookups.
- Expected gain: 0-10% of CPU expert time. It only helps if that kernel is compute-bound rather than DRAM-bound. Measure GB/s achieved against the ~45-50 GB/s DDR4 roofline first.

**Coder variant: A/B it, and do not assume it is faster.**
- In favour:
  - Shard 1 drops from 37.6 GB to 29.6 GB, so expert bytes that fit in 20 GB VRAM rise from roughly 50% to roughly 65% (my estimate, assuming ~18 GB of VRAM goes to experts).
  - The whole file fits easily in VRAM plus pinned RAM, so NVMe leaves the hot path.
  - It is calibrated on code and agentic data, which matches the workload.
- Against:
  - The retained experts are ~3.5 bpw in lookup-table formats (IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL), which DASLab itself shows are slower to decode.
  - Each CPU-missed expert reads about 1.4-1.5x more bytes than in Q2_0.
  - SWE-bench is 95.7% of BF16, and capabilities outside code/agentic/vision degrade.
  - Engine compatibility needs checking: expert count 256 in metadata, per-expert tensors in the allocation listing, and IQ kernels in pancake's CPU and HIP paths.
- Net effect is uncertain, roughly −10% to +20% decode by my estimate. It is worth a one-evening bench with the same opencode trace and top-1 agreement check the user already uses.
- Suggest asking DASLab on the HF discussion for a "Coder Q2_0" build. Its ffn_down_exps are already Q2_0/IQ4_NL-only, so a Q2_0-ladder Coder build would combine fast decode with the smaller file.

**Shard 2 (n-gram table, 28.8 GB):** pancake should keep it mmap'd, or put hot rows in RAM if the remaining 57 GB allows. DASLab explicitly says RAM residency removes the paging cost. This may be an easy latency win if pancake currently pages it from NVMe per token. Verify how Strata handles `per_layer_token_embd`; there is a `ple_residency_test.cpp`, so it may already be handled.

**Re-quantizing for our own workload (GSQ/RCO on opencode traces):**
- The tooling is open, but the compute is 8-16x H200.
- RCO alone over an existing per-tensor database is cheaper, but building that database still requires GSQ at every type.
- It is not feasible on the 7900 XT. The realistic route is to request it, or rent a node for a "coder-calibrated Q2_0" run.

**Other DASLab work, ranked by fit:**
- **SLQ's EAR metric:** adopt it as the fidelity metric alongside "80.6% same top token". It directly predicts MTP/speculative acceptance on our engines. Free to adopt.
- **MatGPTQ:** conceptually very attractive for tiering. A nested file could keep the 4-bit slice in VRAM for hot experts and read only the 2-bit most-significant-bit slice from RAM for CPU experts, giving bandwidth-adaptive precision per tier. But it needs re-quantization plus new CPU and HIP kernels, and no MoE GGUF exists. Long-term, MEDIUM-LOW.
- **Disaggregated prefill:** the accuracy gain needs NVFP4 hardware and a trained prefiller. The ODP idea (stream a separate prefill weight copy from SSD during long prefills) is not useful on our hardware: pancake prefill already dequantizes on the GPU, and our bottleneck is decode. LOW.
- **No runtime gain for us:**
  - MoESQ (Blackwell sparse tensor cores, NVFP4)
  - Marlin and qutlass (NVIDIA)
  - QMoE (Switch-only, GPU dictionary decoding)
  - AQLM and PV-Tuning (large-codebook vector quantization; costly lookups on AVX2, and GSQ reaches near-parity in scalar GGUF)
  - HIGGS (superseded for GGUF by GSQ)
  - GPTQ-2D and GPTQ-Babai (quantization-time only)
  - WUSH (W+A FP4)

### Gaps
- I did not open the GSQ or RCO PDFs for MoE-specific ablations, for example whether GSQ applies to Flash-Next experts at block 64 for Q2_0, or the calibration set used.
- The Q2_0 block layout is my inference from GSQ's 2-bit codebook. The local ggml-common.h has `block_q2_0` = fp16 scale + QK2_0/4 bytes, but I did not confirm whether a zero-point exists.
- The Coder GGUF's per-expert tensor layout and the "IQ1_M" directory name are unexplained. Its decode speed is not published.
- I found no DASLab paper on CPU MoE kernels or expert offloading. The po2-kernels README is empty.
- The MoESQ authors' affiliations (Kwanhee Lee, Namhoon Lee) were not verified.

## Q2. Sparsity (SparseGPT successors, 2:4, activation sparsity) for CPU expert compute

### Takeaway
The only usable sparsity idea for the CPU expert path is **intra-expert activation sparsity** with a transposed down-projection layout. It reduces the RAM bytes actually read, which is the bottleneck. Weight sparsity (SparseGPT, 2:4, DASLab's paired-4:8 MoESQ) only pays off on sparse tensor cores, not on AVX2. ISTA has no recent CPU sparsity work beyond the 2023 DeepSparse/Neural Magic line.

### Cited Findings
- SparseGPT (ISTA LEAD, ICML 2023) prunes in one shot to 50-60% unstructured sparsity (from memory, not re-fetched; arXiv 2301.00774). Its CPU speedups relied on the Neural Magic DeepSparse runtime: "IST Austria & Neural Magic" sparse fine-tuned models "execute fast on CPUs by taking advantage of sparsity". — [search summary of HF blog](https://huggingface.co/blog/mwitiderrick/llm-infrerence-on-cpu)
- MoESQ (ISTA, Sep 2026) is DASLab's current MoE sparsity work.
  - Uses paired-4:8 semi-structured sparsity plus NVFP4. Values are 2 bits per weight kept, 2.75 bpw including mask.
  - Requires a custom grouped sparse GEMM on Blackwell sparse tensor cores.
  - Quality: 96.09% preserved. — [arXiv](https://arxiv.org/abs/2610.02241); [HF card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-P48NVFP4-MoESQ)
- DarwinLM (ISTA, evolutionary structured pruning; repo Feb 2025) exists, but it is dense-model structured pruning. — [GitHub org](https://github.com/IST-DASLab)
- "Uncovering Intra-expert Activation Sparsity for Efficient MoE Execution" (arXiv 2605.08575; affiliation not verified, not shown to be European):
  - Reports up to 90% sparsity within each expert in pretrained MoEs without significant accuracy loss.
  - Implemented as skipped computation in vLLM's MoE path, on GPU. — [arXiv](https://arxiv.org/pdf/2605.08575)
- TEAL and CATS (US-led) are already covered in the US notes and not repeated.

### Inferences
**Weight 2:4 / 4:8 on CPU: SKIP.**
- AVX2 has no sparse instructions.
- At 2 bits per value, 4:8 metadata adds about 0.5-0.75 bpw, so bytes per expert do not fall versus the current ~2.25-2.4 bpw Q2_0.
- Quality loss on top of 2-bit quantization is likely large; MoESQ already loses ~4% at 4-bit values.

**Activation sparsity in the CPU down projection: MEDIUM, a real candidate.**
- Flash-Next expert intermediate size is 640, per the down_exps row width. After SwiGLU, many of the 640 intermediate activations per expert are near zero.
- With a GGUF row-major layout (output rows × 640 input columns in 64-blocks), skipping a zero intermediate neuron saves nothing unless a whole 64-column block is inactive.
- To turn sparsity into fewer DRAM bytes, pancake would need to store a **transposed down_exps copy for the RAM tier**: 640 rows of hidden_size, each as Q2_0 blocks along hidden. Then only rows for active neurons are read.
- At 50-70% activation sparsity with a threshold, down-projection bytes fall by that fraction. Down is about one third of expert bytes, so CPU expert time falls by roughly 15-25%. Decode is about half CPU, so the end-to-end gain is roughly 7-12% (my estimate). The quality cost has to be validated with the top-1 / EAR check.
- gate/up cannot be skipped this way because their inputs are dense. Predictor-based skipping in the Deja Vu style is covered in the US notes.

### Gaps
- I found no ISTA or Central European paper (2023-2026) on activation sparsity for CPU MoE decode.
- The activation-sparsity level of Flash-Next experts at 2-bit is unknown and needs measuring in pancake.

## Q3. Other Central/Western European work on MoE offload, caching, kernels, decoding or agents (TUM, Darmstadt, Saarland/MPI, Tübingen, LMU, KIT, Heidelberg, UvA/Qualcomm, Delft, Leuven, TU Wien, Charles)

### Takeaway
Outside ISTA, the one directly applicable item is **Qualcomm AI Research Amsterdam's cache-conditional expert routing**. It changes routing, not eviction, so it is untested in pancake, and its authors report up to 2x on-device. The rest of the Qualcomm/UvA efficiency output is NPU/W+A quantization. Searches turned up no MoE-offload, caching or kernel paper led by TUM, Darmstadt, KIT, Delft, Leuven, TU Wien, Charles University, Tübingen, LMU, Heidelberg or Saarland/MPI that fits pancake.

### Cited Findings
- **Mixture of Cache-Conditional Experts for Efficient Mobile Device Inference.** Skliar et al. Qualcomm AI Research, Amsterdam (LEAD). arXiv 2412.00099, Nov/Dec 2024; NeurIPS 2024 Expo demo. — [arXiv](https://arxiv.org/html/2412.00099v2); [NeurIPS expo](https://neurips.cc/virtual/2024/expo-demonstration/100334)
  - Problem: current MoEs show little temporal locality in expert choice, so LRU caches thrash.
  - Method: training-free cache-aware routing that adds a locality prior so the router prefers experts already in cache.
  - Result: up to 2x speedup over regular routing with LRU on Qwen-MoE (4-bit and 8-bit) on 10 GB and 16 GB phones. Evaluated on LM perplexity, MMLU and GSM8K. Results are for one device only.
- **GPTVQ** (Van Baalen et al., Qualcomm Amsterdam, 2024, arXiv 2402.15319; from memory, not re-fetched): vector quantization for LLMs, from the same lab.
- **FPTQuant: function-preserving transforms for LLM quantization.** Van Breugel, Bondarenko, Whatmough, Nagel. Qualcomm Amsterdam. arXiv 2506.04985, Jun 2025. Six transform types, aimed at edge W+A quantization. — [arXiv](https://arxiv.org/html/2506.04985v1)
- **Dissecting Quantization Error: a concentration-alignment framework.** Federici, van Breugel, Whatmough, Nagel. Qualcomm Amsterdam. arXiv 2603.04359, Mar 2026. Uses a block-diagonal transform. — [arXiv](https://arxiv.org/pdf/2603.04359)
- **LR-QAT** (Bondarenko, Del Chiaro, Nagel; arXiv 2406.06385): QAT of a 7B model in under 21 GB of memory. — [arXiv](https://arxiv.org/html/2406.06385v1)
- **Efficient Reasoning on the Edge** (Qualcomm, arXiv 2603.16867): reasoning LoRA adapters plus budget forcing on device. — [arXiv](https://arxiv.org/pdf/2603.16867)
- **Scaling Laws for Fine-Grained MoE.** Krajewski et al. University of Warsaw / IDEAS NCBR (LEAD; Poland, Central Europe). arXiv 2402.07871, Feb 2024. FFN-sized experts are suboptimal at almost any compute budget; this is the basis of today's 512-expert fine-grained designs. — [arXiv](https://arxiv.org/html/2402.07871v1)
- A broad search for consumer-GPU MoE offloading papers found MoBiLE, Specter, SpecMoEOff, FineMoE (EuroSys'26), OD-MoE and KTransformers (SOSP'25), none showing TUM, KIT, Darmstadt or Delft affiliation. A non-European CPU-GPU collaborative paper (arXiv 2512.16473) is already in the US notes. — [search results incl. Specter](https://arxiv.org/abs/2511.14102); [KTransformers](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)

### Inferences
**Cache-conditional routing: HIGH, the top non-ISTA pick.**
- It attacks pancake's dominant cost directly: CPU-computed misses are about half of decode.
- It differs from the tested-and-failed ClassLRU scoring, which only changed *which* experts are evicted. This changes *which* experts are selected.
- Implementation is a few lines in the router:
  - For each token, after top-k, swap a non-cached expert for a cached one whenever its logit is within a margin δ of the k-th score (or add +λ to cached-expert logits).
  - Renormalize the gates over the final set.
- Use VRAM residency as the "cache". Optionally use a second, smaller bias for experts in pinned RAM versus NVMe.
- With 10 active out of 512 experts, the scores near the cutoff are usually flat, so swaps should cost little quality. Tune δ against the user's top-1 / EAR agreement test, for example accepting ≤1-2 points of top-1 loss.
- Expected gain, by my estimate: if the VRAM hit rate rises from ~50% to ~70%, CPU-expert work falls by ~40%. With CPU work at about half of decode, that is roughly +20-35% decode t/s, less any quality guardrails.
- It composes with the Coder variant (fewer experts) and with MTP. For MTP, apply the same bias in the draft head so drafts and verification touch the same experts.

**Qualcomm quantization papers:** FPTQuant, Dissecting Quantization Error and GPTVQ are NPU/W+A or vector-quantization work. They add nothing over GSQ for weight-only GGUF on our engines. SKIP.

**Warsaw fine-grained MoE scaling:** training-time insight only. SKIP.

**Agents and evaluation:** nothing from these groups surfaced for single-user coding-agent serving. The MoESQ card's "presence_penalty=0 for tool calling" note (Q1) is the only agent-serving tip, and it applies to opencode sampler settings.

### Gaps
- My searches did not cover each named university separately: TUM, TU Darmstadt (Kersting group), Saarland/MPI-SWS/MPI-INF, Tübingen/MPI-IS, LMU, KIT, Heidelberg, TU Delft, KU Leuven, TU Wien and Charles University. I found no relevant LLM-inference-systems paper from them, which is weak evidence that none exist, not proof.
- I found no Qualcomm/UvA speculative-decoding paper in this session.
- Cache-conditional routing has not been evaluated on 512-expert or hybrid DeltaNet models, and its quality cost on coding benchmarks is unknown.
