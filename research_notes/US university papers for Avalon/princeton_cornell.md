# Princeton and Cornell AI papers (Nov 2022 - Nov 2026) for pancake / Avalon / opencode

Setup reminder: Qwen3.8-Flash-Next (~125B MoE hybrid: Gated DeltaNet (GDN) + attention, MTP head), 2-3-bit GGUF (IQ2_XS / IQ3_KS), one RX 7900 XT 20 GB (gfx1100, ROCm), Ryzen 7 5700G, 57 GB RAM, NVMe expert tier. Single user, opencode agent. Already tried with no gain: HIP graphs, the unroll flag, Vulkan, TurboQuant KV.

Lead/co flags: **Lead** means the first author or senior lab is at Princeton or Cornell. **Co** means a Princeton or Cornell co-author on a paper led elsewhere. Where I write "(background knowledge, not fetched this session)", the arXiv ID and content are from my training data and were not re-verified here. The report writer should treat those as lower-confidence.

Scope note (avoiding duplication):
- Earlier notes already cover Hydra (MIT), TEAL (MIT-led, Tianle Cai co-author), OEA (Harvard-led, Dao co-author) and CALDERA (Stanford-led, Goldsmith/Princeton co-author). They also only pointed to ReplaySSM, QuIP#/QTIP and "Critical Thinking" (Cornell-led) as other institutions' work.
- This file covers those Princeton/Cornell items in depth and does not re-describe the MIT/Stanford ones.

---

## Q1. QTIP / QuIP# / QuIP / YAQA (Cornell RelaxML) vs our IQ2_XS / IQ3_KS. Better quality per bit? Speed? Available as GGUF IQ*_KT? Worth re-quantizing experts?

### Takeaway
QTIP is the best published PTQ family at 2-4 bpw. ik_llama.cpp's IQ*_KT types implement its "3INST" trellis, and the author measured only a "small improvement", about 0.2 bpw-equivalent over i-/k-/iqk-quants.

The trellis types do not fit our setup today, for two reasons:
- **No GPU kernel for our card.** The listed backends are CUDA, Metal, NEON and CPU, not HIP, and ik_llama's ROCm support is "limited / not actively maintained".
- **Slow on the CPU.** Our CPU computes missed experts, and that is the path where trellis decode is weakest.

Verdict: do not re-quantize the whole model. At most, port IQ2_KT/IQ3_KT GPU kernels to HIP and apply them only to VRAM-resident experts, using the roughly 0.2 bpw saving to fit more experts in VRAM. Expected gain: small, around 0-5% decode, mostly through VRAM hit rate, plus a marginal quality gain. YAQA (Cornell, 2025) is the more interesting idea to borrow: a KL-targeted rounding/Hessian method that does not depend on the quantizer.

### Cited Findings
- **QTIP: Quantization with Trellises and Incoherence Processing.** Lead: Cornell. Albert Tseng, Qingyao Sun, Christopher De Sa (Cornell) and David Hou (no affiliation listed). NeurIPS 2024 Spotlight, arXiv 2406.11235 (Jun 2024, v4 Jun 2025). — [arXiv abs](https://arxiv.org/abs/2406.11235); [HTML v4](https://arxiv.org/html/2406.11235v4)
  - Idea: trellis-coded quantization. The paper says a "stateful decoder ... separates the codebook size from the bitrate and effective dimension", so it gets high-dimensional VQ without exponentially large codebooks. Combined with Hadamard incoherence processing. — [arXiv abs](https://arxiv.org/abs/2406.11235)
  - Codes: "1MAD" (LCG, then sum of 4 bytes ≈ Gaussian), "3INST" (LCG, XOR with an FP16 magic constant, add halves; ~3 ALU ops/weight), "HYB" (hash plus a 2 KiB LUT, ~2 instructions/weight, fine-tunable). — [HTML v4](https://arxiv.org/html/2406.11235v4)
  - Wikitext2 perplexity, ctx 4096, no fine-tuning (Llama-2):

    | Model, bits | FP16 | QTIP 3INST | QuIP# no-FT | QuIP# FT | AQLM FT |
    |---|---|---|---|---|---|
    | 7B, ≈2 bit | 5.12 | 6.82 | 8.22 | 6.19 | 6.14 |
    | 7B, ≈3 bit | 5.12 | 5.40 | 5.60 | 5.41 | 5.38 |
    | 70B, ≈2 bit | 3.12 | 3.90 | 4.16 | 3.91 | 3.83 |
    | 70B, ≈3 bit | 3.12 | 3.27 | 3.41 | 3.35 | 3.36 |

    With fine-tuning, the HYB code beats QuIP# and AQLM at every bitrate. — [HTML v4](https://arxiv.org/html/2406.11235v4)
  - Batch-1 decode on RTX 6000 Ada (960 GB/s), Llama-2-70B:

    | Method | tok/s |
    |---|---|
    | QTIP 2-bit | 23.5 |
    | QTIP 3-bit | 19.1 |
    | QTIP 4-bit | 16.3 |
    | QuIP# 2-bit | 22.2 |
    | AQLM 2-bit | 8.78 |

    In other words, QTIP is memory-bandwidth-bound at roughly the speed of QuIP#. — [HTML v4](https://arxiv.org/html/2406.11235v4)
  - CPU: the paper gives only an unmeasured ARM NEON sketch, a 1D HYB code using the `vqtbl4q_u8` table lookup (6.89 PPL on 7B at 2-bit). There is no x86 measurement. — [HTML v4](https://arxiv.org/html/2406.11235v4)
  - Code: github.com/Cornell-RelaxML/qtip. — [HTML v4](https://arxiv.org/html/2406.11235v4)
- **QuIP#** (Lead: Cornell; Tseng, Chee, Sun, Kuleshov, De Sa; ICML 2024, arXiv 2402.04396): Hadamard incoherence plus an E8-lattice codebook. Code: Cornell-RelaxML/quip-sharp. — [arXiv 2402.04396](https://arxiv.org/abs/2402.04396)
- **QuIP: 2-Bit Quantization of LLMs with Guarantees** (Lead: Cornell; Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, Christopher De Sa; NeurIPS 2023, arXiv 2307.13304): introduced incoherence processing with random orthogonal matrices plus LDLQ adaptive rounding. Background knowledge, not fetched this session. — [arXiv 2307.13304](https://arxiv.org/abs/2307.13304)
- **YAQA: Model-Preserving Adaptive Rounding.** Lead: Cornell. Albert Tseng, Zhaofeng Sun, Christopher De Sa. arXiv 2505.22988, May 2025 (v2 Sep 2025). Code: github.com/cornell-relaxml/yaqa. — [arXiv 2505.22988](https://arxiv.org/abs/2505.22988v1); [code listing](https://sotaverified.org/papers/model-preserving-adaptive-rounding)
  - Idea: minimize KL to the original model's output rather than each layer's own activation error. It uses Kronecker-factored sketches of the full-model (Fisher) Hessian, which "can be tractably computed for hundred-billion parameter LLMs", and a rounding algorithm that works with any quantizer. — [arXiv 2505.22988](https://arxiv.org/html/2505.22988v1)
  - Result: "reduces the KL divergence to the original model by 30%" over prior rounding, which the authors call provably better than GPTQ/LDLQ. — [arXiv 2505.22988](https://arxiv.org/abs/2505.22988v1)
- **ik_llama.cpp trellis types (QTIP-derived):**
  - Types and bpw: IQ1_KT, IQ2_KT (2.125 bpw), IQ3_KT (3.125 bpw) and IQ4_KT (4.0 bpw). They use QTIP's 3INST generator with block/super-block scales and a clustered seed search, not QTIP's Hadamard transform or Viterbi search. — [PR #113](https://github.com/ikawrakow/ik_llama.cpp/pull/113)
  - Quality: Llama-2-7B at 2 bpw, IQ2_KT PPL 6.36 (28.7% quantization error) vs QTIP 6.82 (33.2%). On Llama-3.1-8B the author reports about 0.2 bpw less for the same error versus i-/iqk-quants, and calls it "a small improvement ... nowhere near the hype". — [PR #113](https://github.com/ikawrakow/ik_llama.cpp/pull/113)
  - GPU speed: CUDA on RTX 4080, Llama-3.1-8B: IQ2_KT 146-194 t/s (the PR body and commit messages disagree) vs IQ2_XXS 208-216 t/s. So **trellis was slower than an IQ2 type on GPU** in that PR. — [PR #113](https://github.com/ikawrakow/ik_llama.cpp/pull/113)
  - CPU speed: CPU decode was initially very slow. Later PRs raised PP512 from ~57 to 135 t/s (IQ2_KT) and from ~44 to 131 t/s (IQ3_KT) via dequantizing GEMM (PR 482). A bf16 path (PR 484) reached ~234 t/s PP on a Ryzen 7950X, but token generation stayed at ~12 t/s (8B model). PR 453 sped up IQ3_KT and IQ4_KT. — [ik_llama.cpp repo/PR summaries](https://github.com/ikawrakow/ik_llama.cpp); [PR 484](https://github.kxpsado.workers.dev/ikawrakow/ik_llama.cpp/pull/484)
  - Backends: the listed trellis backends are CUDA (PR 113), CPU (PR 441), NEON (PR 471) and Metal (PR 475). **HIP/ROCm is not among them**, and ik_llama's docs label ROCm "Limited / Not actively maintained". — [ik_llama.cpp docs index](https://mintlify.wiki/ikawrakow/ik_llama.cpp/introduction)
  - Community use: some large-MoE GGUF quants mix IQ2_KS with IQ2_KT, but publish no ROCm numbers. On GLM-4.5, an IQ4_KT mix gave PPL 3.3351, similar to IQ4_KSS at the same 4.0 bpw. — [ik_llama discussion #562](https://github.com/ikawrakow/ik_llama.cpp/discussions/562); [Kimi quant page](https://huggingface.co/gghfexp/k27)

### Inferences
- Per bit, QTIP-family quality beats IQ2_XS at about 2 bpw. ik's own measure, 0.2 bpw at equal error, suggests IQ2_KT (2.125 bpw) ≈ IQ2_S-class quality at IQ2_XS-class size (2.31 bpw). For a 125B MoE, 0.2 bpw over ~120B expert parameters is ~3 GB less. That is roughly 15% of VRAM, which could move a meaningful slice of hot experts from RAM to VRAM.
- Decode cost is the catch:
  - **CPU-miss path:** trellis decoding is compute-heavy (an LCG per weight group). On the 5700G (AVX2, 8 cores) this path would likely get slower, so keep RAM/NVMe-tier experts in IQ2_XS/IQ3_KS.
  - **GPU path:** the PR showed trellis slower than IQ2_XXS on CUDA, so a HIP port for gfx1100 would need careful tuning (LDS/wave32) just to break even.
- Net verdict: a mixed format (KT for VRAM-resident experts, IQ_K/IQ_XS for offloaded ones) is technically possible but means maintaining a HIP port. Expected gain 0-5% decode, plus a small quality gain. Low-medium priority, behind engine-level items such as the Marconi-style cache and ReplaySSM.
- **Pancake:** it is a Strata fork with its own kernels, so KT support would need a new dequant kernel. Priority: low.
- **YAQA:** applies to any quantizer, including IQ*_K and IQ*_XS. Porting its KL/Fisher-Hessian sketch into llama-quantize's importance matrix (instead of the diagonal activation imatrix) could reduce KL at the same size. Collecting the sketch needs backward passes through the model, which is hard on a 20 GB GPU for 125B parameters. Research-grade; the payoff would be quality, not speed.

### Gaps
- I found no published IQ2_KT/IQ3_KT vs IQ2_XS/IQ3_KS perplexity on any MoE hybrid model, and no ROCm/gfx1100 speed numbers for KT types.
- I did not verify whether mainline llama.cpp or Avalon already carries KT types with HIP kernels. Check `ggml/src/ggml-cuda` in Avalon for `iq2_kt`.
- I did not verify current x86 token-generation speed for KT types on recent ik_llama builds.

---

## Q2. Hybrid-model (SSM / linear-attention) inference from Dao's lab and Cornell: state rollback for speculative decoding, kernels, prefix caching. What applies to GDN on gfx1100?

### Takeaway
Three Princeton/Cornell items apply directly:
- **ReplaySSM (Dao blog plus a public repo, 2026)** covers Gated DeltaNet with MTP drafters. It caches u = β(v − S·k) in a ring buffer, commits accepted drafts by moving a pointer, and halves state traffic.
- **Marconi (Princeton-led, MLSys 2025)** is a prefix-cache policy designed for hybrid models. It reports 34.4x token hit rate on SWE-bench traces, and maps almost one-to-one onto pancake's conversation cache.
- **The Mamba in the Llama (Cornell-led, NeurIPS 2024)** is the earlier hardware-aware hybrid speculative-decoding kernel. ReplaySSM supersedes it.

FlashAttention-3 is Hopper-only and does not apply. Mamba-2/SSD and Mamba-3 matter only as kernel lineage and future-model notes.

### Cited Findings
- **ReplaySSM** (Dao AI Lab with NVIDIA; Tri Dao's blog, 2026; earlier notes describe it as NVIDIA plus Princeton). The blog lists no paper or arXiv ID. Code: github.com/Johnny-Liou/ReplaySSM. — [Tri Dao blog](https://tridao.me/blog/2026/replayssm/); [NVIDIA blog](https://developer.nvidia.com/blog/replayssm-rethinking-state-storage-for-faster-hybrid-model-decoding/)
  - Mechanism: instead of loading, updating and storing the state every step, it caches recent inputs in a buffer. Outputs are computed from a checkpoint state plus the buffer, and the full state is written back only at periodic flushes. The blog calls this "mathematically equivalent ... up to floating-point error". Flushes trigger when h + 2T > L, so at least T free slots remain. — [Tri Dao blog](https://tridao.me/blog/2026/replayssm/)
  - Rollback: rejected drafts are handled by a buffer-pointer move, instead of keeping one state snapshot per draft token. A device-side commit kernel advances the pointers by the accepted count without host sync. A ring buffer avoids data-dependent copies and helps tree speculation. — [Tri Dao blog](https://tridao.me/blog/2026/replayssm/)
  - GDN specifics: because the delta-rule correction depends on the state, the buffer stores u = β(v − S·k) rather than v. Verification uses a chunk-wise form with one T×T triangular solve, with no serial loop over drafts. States are FP32 and buffers BF16. — [Tri Dao blog](https://tridao.me/blog/2026/replayssm/)
  - Numbers (vLLM, CUDA graphs, Nemotron-3 and Qwen3.5 with MTP drafters):
    - Standard decode: up to 1.48x end-to-end, or 1.43x on large MoE.
    - Speculative decoding: 1.87-1.96x over standard decode and up to 2.14x over vLLM's speculative baseline.
    - SSM kernel speedup: 1.43-1.64x on Qwen3.5.
    - Dominant state traffic per head: 8dn down to 4dn bytes. — [Tri Dao blog](https://tridao.me/blog/2026/replayssm/)
- **Marconi: Prefix Caching for the Era of Hybrid LLMs.** Lead: Princeton. Rui Pan (Princeton) et al. with AWS co-authors, senior author Ravi Netravali (Princeton). MLSys 2025, arXiv 2411.19379. Code: github.com/ruipeterpan/marconi. — [MLSys proceedings](https://proceedings.mlsys.org/paper_files/paper/2025/hash/7c180af017258d239bac6248d1eb26ac-Abstract-Conference.html); [arXiv](https://arxiv.org/abs/2411.19379); [PDF with affiliations](https://cdn.amazon.science/96/d4/ee6df8f84a34b49a71f9c39212f2/marconi-prefix-caching-for-the-era-of-hybrid-llms.pdf)
  - Problem: in-place SSM updates prevent rolling back to a partial-prefix match, so naive hybrid caches get only exact hits and end up with many large, rarely reused entries. — [arXiv](https://arxiv.org/abs/2411.19379)
  - Admission:
    - **Branch points:** speculatively insert the input into a radix tree. If a new branch node appears, checkpoint the SSM state at that branch during prefill.
    - **Last decoded token:** always checkpoint the state at the last decoded token, for conversation and agent continuation.
    - **Chunked kernels:** cache the second-to-last chunk boundary state, for example the state at token 64 to serve a hit at token 80 with chunk size 32.
    - **Limits:** at most 2 SSM states per sequence, and a pure-input prefix is reused from its 3rd occurrence. — [HTML v3](https://arxiv.org/html/2411.19379v3)
  - Eviction score S(n) = recency(n) + α·flop_efficiency(n), where flop_efficiency = FLOPs saved / bytes of state. α is tuned by replaying the bootstrap trace; α = 0 is plain LRU. — [HTML v3](https://arxiv.org/html/2411.19379v3)
  - Results:
    - Token hit rate vs vLLM+: 4.5x (LMSys), 7.3x (ShareGPT), 34.4x (SWE-bench).
    - On the SWE-bench trace: 16.4% to 32.7% hit rate vs SGLang+.
    - Up to 71.1% (617 ms) lower TTFT.
    - Gains grow with a higher SSM:attention ratio and with larger state dimension. — [HTML v3](https://arxiv.org/html/2411.19379v3); [arXiv](https://arxiv.org/abs/2411.19379)
- **The Mamba in the Llama: Distilling and Accelerating Hybrid Models.** Lead: Cornell. Junxiong Wang (Cornell), Daniele Paliotta (Geneva), Avner May (Together), Alexander M. Rush (Cornell), Tri Dao (Princeton/Together). NeurIPS 2024, arXiv 2408.15237. — [arXiv HTML](https://arxiv.org/html/2408.15237v4)
  - It distills Llama-3-8B and Zephyr into Mamba/Mamba-2 hybrids that keep ~¼ of the attention layers, using 20B tokens.
  - Its "hardware-aware speculative sampling algorithm and a fast kernel" target Mamba/hybrids, with over 300 t/s for Mamba 7B.
  - Code: jxiw/MambaInLlama and itsdaniele/speculative_mamba. — [arXiv HTML](https://arxiv.org/html/2408.15237v4); [NeurIPS listing](https://mlanthology.org/neurips/2024/wang2024neurips-mamba)
- **Mamba-3: Improved Sequence Modeling using State Space Principles.** Co-lead: CMU (Gu) and Princeton (Dao), with Together. ICLR 2026, arXiv 2603.15569. — [arXiv](https://arxiv.org/abs/2603.15569); [ICLR](https://iclr.cc/virtual/2026/oral/10010353)
  - Changes: a new discretization, a complex-valued state, and a MIMO variant.
  - Matches Mamba-2 perplexity at half the state size. MIMO uses up to 4x the decode FLOPs at similar wall-clock latency, because decode is memory-bound.
  - +0.6 points over Gated DeltaNet at 1.5B, plus another +1.2 points with MIMO. — [search summary of arXiv 2603.15569](https://arxiv.org/abs/2603.15569)
- **Transformers are SSMs / Mamba-2 (SSD).** Co-lead: Tri Dao (Princeton) and Albert Gu (CMU). ICML 2024, arXiv 2405.21060. SSD chunked matmul formulation. Background knowledge, not fetched this session. — [arXiv 2405.21060](https://arxiv.org/abs/2405.21060)
- **FlashAttention-3.** Co: Princeton/Together (Dao), with Colfax, Meta and NVIDIA; first author Jay Shah (Colfax). NeurIPS 2024, arXiv 2407.08608. Uses Hopper features (WGMMA, TMA, warp specialization, FP8). Background knowledge, not fetched this session. — [arXiv 2407.08608](https://arxiv.org/abs/2407.08608)

### Inferences
- **ReplaySSM vs pancake's existing "GDN state replay".** Pancake already replays GDN state for MTP and prompt-lookup drafts. The questions are whether it (a) snapshots and copies state per draft token or replays from a checkpoint, and (b) writes the full state back every step.
  - Size estimate (assumed shape, not verified): if Qwen3.8-Flash-Next has ~32 heads × 128 × 128 FP32 state per GDN layer (~2 MB) and ~36 GDN layers, that is ~75 MB read plus ~75 MB written per token. At ~126 t/s that is ~19 GB/s, about 2-3% of the 7900 XT's ~800 GB/s.
  - So for us the decode-side gain from halving state traffic is **about 1-3%, not the 1.48x** reported for batched vLLM. Our time goes to expert weights.
  - The spec-decode-side gain is larger only if pancake currently copies state snapshots per draft or re-runs the recurrence serially to roll back. Porting the u-caching plus pointer commit and the chunk-wise verification (one T×T triangular solve) would remove that cost. Estimate 0-8% on accepted-draft-heavy code edits.
  - ReplaySSM's design (ring buffer, device-side commit, no host sync) is HIP-portable. It has no CUDA-only dependencies beyond CUDA graphs, which we tested without gain.
- **Marconi is the most directly useful idea here for pancake's conversation cache.**
  - In single-user opencode traffic the dominant pattern is "input and output" (each turn extends the previous transcript), plus shared system prompts across sessions and subagents.
  - Adopting it means: always checkpoint the GDN plus attention state at the last decoded token, which pancake likely already does. Also checkpoint at radix branch points, such as the end of the system prompt and tool schema, and at the point before compaction. Then use FLOP-per-byte-weighted eviction across VRAM/RAM/NVMe cache tiers.
  - Expected gain: TTFT on subagent spawns, session restarts and edit-and-resubmit, plausibly seconds per occurrence for long prompts on a 125B model. Little decode change.
- **The Mamba in the Llama:** superseded by ReplaySSM for kernels. Its distillation recipe is irrelevant to us.
- **Mamba-2/SSD:** relevant only as the origin of chunked prefill. Check that pancake/Avalon's GDN prefill on gfx1100 uses the chunked WY/UT form, not a token-serial recurrence. If prefill is serial, that is a large prefill win; if already chunked, nothing to gain.
- **Mamba-3:** relevant only if future Qwen/others adopt it. Its MIMO "more FLOPs, same latency" property suits a memory-bound single-user decode like ours.
- **FlashAttention-3:** not applicable on gfx1100 (no WGMMA/TMA). Zero gain.

### Gaps
- I could not confirm ReplaySSM author affiliations beyond the Dao blog and NVIDIA blog, or whether a paper exists.
- I have not verified Qwen3.8-Flash-Next's actual GDN head count or state size. The traffic estimate above is an assumption.
- I did not inspect pancake's replay implementation, so I cannot tell whether it already matches ReplaySSM.

---

## Q3. Medusa/Hydra-style heads vs our MTP. Anything new from Princeton 2025-2026 on decoding?

### Takeaway
Medusa (Princeton-led) gives 2.2x lossless (Medusa-1) to 2.3-3.6x (Medusa-2, which fine-tunes the backbone), but its heads are independent. Qwen's native MTP head is already the stronger, sequentially dependent drafter (Hydra principle, covered in the MIT notes). Two ideas remain transferable:
- **Typical acceptance:** a lossy, entropy-adaptive acceptance rule, offered as an opt-in for thinking tokens.
- **Medusa's tree verification:** a small one.

Princeton's 2025 decoding paper (GTA/GLA, Dao lab) is an architecture change that requires training. It does not apply.

### Cited Findings
- **Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.** Lead: Princeton. Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, Tri Dao. ICML 2024 (PMLR 235), arXiv 2401.10774. Code: FasterDecoding/Medusa (background knowledge). — [arXiv HTML v3](https://arxiv.org/html/2401.10774v3); [Princeton record](https://collaborate.princeton.edu/en/publications/medusa-simple-llm-inference-acceleration-framework-with-multiple-/); [ICML](https://icml.cc/virtual/2024/poster/34133)
  - Method: extra heads predict several future tokens in parallel. Tree attention verifies multiple candidates in one step. "Typical acceptance" accepts plausible candidates rather than requiring exact match to the target distribution. Self-distillation is available when no training data exists. — [arXiv HTML v3](https://arxiv.org/html/2401.10774v3)
  - Speedups: Medusa-1 (frozen backbone, lossless) >2.2x. Medusa-2 (joint fine-tune) 2.3-3.6x. An earlier version reported 2.3-2.8x. — [arXiv HTML v3](https://arxiv.org/html/2401.10774v3); [HF papers](https://huggingface.co/papers/2401.10774)
- **Hardware-Efficient Attention for Fast Decoding (GTA / GLA).** Lead: Princeton. Ted Zadouri, Hubert Strauss, Tri Dao. arXiv 2505.21487, May 2025; ICML 2025 listing. Code: Dao-AILab/grouped-latent-attention. — [arXiv](https://arxiv.org/html/2505.21487v1); [ICML](https://icml.cc/virtual/2025/51861); [Princeton RSE blog](https://blog.rse.princeton.edu/?p=403)
  - GTA (grouped-tied attention): roughly matches GQA with about half the KV cache. At 1.47B, PPL 10.12 vs 10.20 for GQA.
  - GLA (grouped latent attention): matches MLA (60.0% vs 59.1% average accuracy at 1.47B). Its kernel is up to 2x faster than FlashMLA during speculative decoding, when query length > 1.
  - All results are from H100 runs on models of 183M-1.47B trained from scratch. — [arXiv](https://arxiv.org/html/2505.21487v1); [RSE blog](https://blog.rse.princeton.edu/?p=403)
- Context from the earlier MIT notes: Hydra (MIT) shows sequentially dependent heads beat Medusa's independent heads by up to 1.31x in throughput. Qwen's MTP already conditions on the previous token. — [Hydra arXiv 2402.05109](https://arxiv.org/abs/2402.05109)

### Inferences
- MTP (sequential, trained with the base model) dominates Medusa-1 heads. Adding Medusa heads would mean training heads for a 125B model, which is infeasible locally, for little expected gain. Verdict: no.
- **Typical acceptance:** add an opt-in mode only inside `reasoning_content` (thinking). There, exact distribution matching matters less, and acceptance of MTP or prompt-lookup drafts could rise. Code and tool-call tokens must stay strict.
  - Expected gain: maybe 5-15% on thinking-token throughput. This is an inference, not measured.
  - Risk: subtle quality drift in reasoning. Gate it behind a flag and A/B test on opencode tasks.
- **Tree verify:** with expert offload, each extra verified token can pull extra experts. Keep trees tiny (top-2 at the first position only), consistent with the MIT notes.
- **GTA/GLA:** architecture-level and needs pretraining. Not applicable to an existing Qwen checkpoint. The general lesson stands: attention kernels for query length > 1 (the verify step) deserve their own tuning on gfx1100. Check that pancake's attention verify path for 2-6 tokens is not falling back to a single-query decode loop.

### Gaps
- I found no Princeton or Cornell 2025-2026 paper specifically on speculative decoding for MoE with offloaded experts, or on MTP-head improvements.
- AOSpec (arXiv 2608.00881) and SPORK (arXiv 2607.03333) are 2026 agent co-speculation papers. Search results did not show their affiliations, so I cannot attribute them to Princeton or Cornell. — [AOSpec](https://arxiv.org/pdf/2608.00881); [SPORK](https://arxiv.org/pdf/2607.03333)

---

## Q4. Princeton NLP agent and long-context work for opencode reliability and speed

### Takeaway
The useful pieces are:
- **SWE-agent's ACI lessons:** compact, guarded tools, especially edits that a linter checks.
- **mini-SWE-agent's finding:** strong models do fine with bash only, which argues for keeping opencode's tool surface small for a local 2-3-bit model.
- **HELMET:** an evaluation method for checking quantization and cache changes at long context. Use RAG/recall and re-rank tasks, not needle-in-a-haystack (NIAH).
- **Slipstream (Princeton, May 2026):** asynchronous, validated context compaction, a direct fit for opencode's compaction step.

ReAct and Tree of Thoughts are foundational. ToT's multi-sample search does not fit a single slot.

### Cited Findings
- **SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering.** Lead: Princeton. John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, Ofir Press. arXiv 2405.15793 (v3 Nov 2024); NeurIPS 2024 (background knowledge).
  - 12.5% pass@1 on SWE-bench and 87.7% on HumanEvalFix. The ACI "significantly enhances an agent's ability to create and edit code files". — [arXiv](https://arxiv.org/abs/2405.15793)
  - From the paper body (background knowledge, not fetched this session): the ACI design choices are a windowed file viewer (~100 lines), search commands that return summarized results, an edit command that runs a linter and rejects syntactically invalid edits, and collapsing of old observations to save context. The ablations showed that removing linter-guarded editing and using worse search/view tools lowered the resolve rate. The exact ablation numbers are not verified here.
- **mini-SWE-agent** (Princeton and Stanford SWE-bench/SWE-agent team):
  - About 100 lines of Python, bash only, "no custom tools or special interfaces", with a linear history in a ReAct loop. It scored >74% on SWE-bench Verified in an early version.
  - It is the harness for the "SWE-bench Bash Only" leaderboard. Vals.ai notes the benchmark is saturated, with frontier models around 96-97% in a 2026 snapshot. — [SWE-bench Bash Only](https://www.swebench.com/bash-only.html); [mini-swe-agent summary](https://api.emergentmind.com/topics/mini-swe-agent-27d26942-1f63-4337-bee8-576ebb1468c3); [Vals.ai archive](https://www.vals.ai/benchmarks/swebench-2025-09-24)
- **SWE-bench** (Lead: Princeton; Jimenez, Yang, Wettig, Yao, Pei, Press, Narasimhan; ICLR 2024, arXiv 2310.06770). Background knowledge. — [arXiv 2310.06770](https://arxiv.org/abs/2310.06770)
- **ReAct** (Lead: Princeton with Google; Shunyu Yao et al.; ICLR 2023, arXiv 2210.03629) and **Tree of Thoughts** (Lead: Princeton with Google DeepMind; Yao et al.; NeurIPS 2023, arXiv 2305.10601). Background knowledge. — [ReAct](https://arxiv.org/abs/2210.03629); [ToT](https://arxiv.org/abs/2305.10601)
- **HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly.** Lead: Princeton (PLI) with Intel; Howard Yen, Tianyu Gao, Danqi Chen et al. ICLR 2025, arXiv 2410.02694. — [arXiv](https://arxiv.org/abs/2410.02694); [project page](https://princeton-nlp.github.io/HELMET); [HF blog](https://huggingface.co/blog/helmet)
  - 59 models tested. Synthetic NIAH-style tasks "are not good predictors of downstream performance".
  - It has 7 application categories with low correlation to each other, lengths up to 128K, and model-based metrics.
  - It recommends RAG tasks for fast development iteration, because they better predict other downstream performance.
  - Open models lag closed ones on full-context reasoning and instruction following as length grows. Generation-with-citations and re-ranking are the hardest. — [project page](https://princeton-nlp.github.io/HELMET); [ICLR](https://iclr.cc/virtual/2025/poster/31157)
- **Slipstream: Trajectory-Grounded Compaction Validation for Long-Horizon Agents.** Lead: Princeton (Netravali group, the same group as Marconi; affiliation inferred from the group, as the abstract page lists none). Zhuofu Chen, Rui Pan, Yinwei Dai, Ravi Netravali. arXiv 2605.08580, 9 May 2026. — [arXiv](https://arxiv.org/abs/2605.08580)
  - Method: the compactor runs asynchronously while the agent keeps working on the full context. A judge then checks the candidate summary against the agent's next reasoning steps (forward intent, key facts and constraints) before swapping it in.
  - Results: up to +8.8 pp accuracy and −39.7% end-to-end latency on SWE-bench Verified and BrowseComp, with Qwen3.5-9B and Seed-OSS-36B. No code link found. — [arXiv](https://arxiv.org/abs/2605.08580); [search summary](https://codex.danielvaughan.com/2026/07/28/next-generation-context-compaction-slipstream-selfcompact-compactionrl-codex-cli-long-horizon-sessions/)
- **Critical Thinking** (Lead: Cornell; Celine Lee, Alexander Rush, with Keyon Vafa of Harvard; IJCNLP-AACL 2025, arXiv 2504.01935): optimal reasoning length depends on task structure. Already summarized in the MIT notes. Here it is conceptual support for per-task thinking budgets. — [arXiv 2504.01935](https://arxiv.org/pdf/2504.01935v1.pdf)

### Inferences
- **opencode ACI for a 2-3-bit local model:**
  - Keep the tool set small, and make edit tools validate before applying (syntax/lint check, then reject with a short error). This complements pancake's tool-call recovery, and cuts wasted turns that each cost a full long-prompt decode.
  - Keep tool outputs short and windowed (file views of about 100 lines, summarized search). Shorter outputs mean less prefill, and fewer cache-unfriendly mid-context changes.
  - Expected gain: agent reliability (fewer loops and bad edits) rather than t/s. Each avoided turn saves tens of seconds.
- **Slipstream on a single GPU:** true parallel compaction competes with the agent for the same GPU. But opencode has natural GPU-idle windows while tools run (tests, builds, greps).
  - Schedule compaction speculatively during tool execution, then validate it against the next steps before swapping. That captures Slipstream's latency win without extra hardware. Combine with a Marconi-style checkpoint at the compaction boundary.
  - Expected gain: removes a synchronous compaction stall, likely tens of seconds per compaction on a 125B model at long context.
- **HELMET as a test harness:** when testing re-quants (KT types, YAQA), KV changes or cache policies, run HELMET's RAG/recall and re-rank subsets at 32K-128K instead of NIAH, which nearly every model passes. Cheap insurance against quality regressions hidden by perplexity.
- **ToT:** not applicable to single-slot decoding. **ReAct:** already the opencode loop.

### Gaps
- The SWE-agent ablation numbers (linter on edit, viewer window size) were not verified this session.
- Slipstream's affiliation is inferred, and no code release was found.
- I found no Princeton evaluation of agents run on 2-3-bit quantized models.

---

## Q5. Broadened scope: other Princeton/Cornell AI or systems work that fits (caching, prefetching, KV compression, compilers, CPU/NVMe), plus a summary table

### Takeaway
Beyond Marconi, the Princeton/Cornell systems output that fits our box is mostly Cornell Tech's Abdelfattah lab KV work:
- **TokenButler:** learned token-importance prediction, including CPU-offloaded KV.
- **xKV:** cross-layer SVD KV compression, 3x on coding tasks.

Both matter only if long contexts squeeze VRAM away from experts. GDN hybrids have KV in only about ¼ of their layers, and TurboQuant KV already showed no gain here, so priority is low.

I found no Princeton- or Cornell-led MoE expert offload or prefetch paper. The 2025 offload literature (fMoE, ExpertFlow, PreScope, HybriMoE, DAOP) showed no Princeton or Cornell affiliation in the results.

### Cited Findings
- **TokenButler: Token Importance is Predictable.** Lead: Cornell (Abdelfattah lab; the code lives under the abdelfattah-lab org, but the Cornell affiliation was not confirmed in the search snippet). arXiv 2503.07518, Mar 2025.
  - A query-aware light predictor (<1.2% parameter overhead) identifies critical KV tokens. It reports >8% perplexity and downstream improvement over prior importance estimators, up to ~1.6x GPU speedup, and up to 7.6x lower latency vs dense attention with CPU-offloaded KV.
  - Code: github.com/abdelfattah-lab/TokenButler. — [arXiv](https://arxiv.org/abs/2503.07518v1); [HTML](https://arxiv.org/html/2503.07518v1)
- **xKV: cross-layer SVD for KV-cache compression.** Abdelfattah lab, arXiv 2503.18893, 2025; the author list was not confirmed in search. Plug-and-play shared low-rank subspace across grouped layers, 3x compression on coding tasks without RULER degradation. Code: github.com/abdelfattah-lab/xKV. — [arXiv PDF](https://export.arxiv.org/pdf/2503.18893)
- MoE offload/prefetch papers found (fMoE, ExpertFlow, PreScope, DuoServe-MoE, HybriMoE, DAOP, and a Nov 2025 caching analysis) showed no Princeton or Cornell affiliation in the results. — [fMoE](https://arxiv.org/abs/2502.05370v1); [PreScope](https://arxiv.org/html/2509.23638v1); [HybriMoE](https://www.arxiv.org/pdf/2504.05897); [DAOP](https://arxiv.org/html/2501.10375v2); [MoE caching analysis](https://arxiv.org/abs/2511.05814v1)
- Marconi's FLOP-per-byte eviction (Q2) is the Princeton learned-caching item. Its α is tuned by replaying the request trace, which is a simple learned policy that fits a single-user workload. — [HTML v3](https://arxiv.org/html/2411.19379v3)

### Inferences
- **KV compression (xKV, TokenButler):** this hybrid's attention KV at 128K context could still reach a few GB. Freeing it would let more experts stay in VRAM, which helps decode more than the KV saving itself.
  - TurboQuant KV already showed no gain here, which suggests KV is not the bottleneck at our context lengths.
  - Verdict: revisit only if opencode sessions regularly exceed ~64K and pancake's VRAM expert hit rate visibly drops with context length. Expected gain: 0-5%.
- **Marconi's α-replay tuning** carries over to pancake's expert tier as well. It is not Princeton's claim, but the same "reuse forecast × bytes saved" utility could replace LRU for expert residency across VRAM/RAM/NVMe. Pancake's expert tiering already exists. Treat this as an idea to A/B test, not an evidence-backed gain.
- I found no Princeton/Cornell compiler, autotuning or CPU-SIMD inference paper in range that targets AMD RDNA3 or AVX2 MoE expert compute. Kai Li's group produced nothing relevant in these searches.

### Gaps
- I did not check Cornell's Zhiru Zhang lab (FPGA/compilers) or Princeton's Kai Li / Margaret Martonosi groups. The available search budget went to the higher-value LLM items. Their output is likely hardware-accelerator-focused and not applicable to a consumer GPU/CPU.
- I did not confirm the affiliations of TokenButler/xKV authors from the PDFs.

### Summary table (for the report writer)

| Paper | Inst. role | Venue/date | Core idea | Key number | Code | Verdict for us | Expected gain |
|---|---|---|---|---|---|---|---|
| Marconi ([2411.19379](https://arxiv.org/abs/2411.19379)) | Princeton lead | MLSys 2025 | Hybrid prefix cache: branch-point and last-token state checkpoints, FLOP/byte eviction | 34.4x hit rate (SWE-bench), −617 ms TTFT | ruipeterpan/marconi | **Yes, pancake conversation cache** | Large TTFT cuts on re-prefill; ~0 decode |
| ReplaySSM ([blog](https://tridao.me/blog/2026/replayssm/)) | Dao lab (Princeton) + NVIDIA | 2026 blog + repo | Input ring buffer (u for GDN), pointer-move rollback, chunk verify | 1.87-1.96x SD (batched vLLM) | Johnny-Liou/ReplaySSM | **Yes, compare with pancake's GDN replay** | 1-3% decode; 0-8% SD if snapshots are copied today |
| Slipstream ([2605.08580](https://arxiv.org/abs/2605.08580)) | Princeton lead (inferred) | arXiv May 2026 | Async compaction validated against the next steps | +8.8 pp, −39.7% latency | none found | **Yes, opencode compaction during tool-idle time** | Removes compaction stalls |
| QTIP ([2406.11235](https://arxiv.org/abs/2406.11235)) / ik IQ*_KT | Cornell lead | NeurIPS 2024 | Trellis-coded quantization (3INST/HYB) | 2-bit 70B PPL 3.90 vs QuIP# 4.16 (no FT) | Cornell-RelaxML/qtip; ik_llama | Only for VRAM-resident experts; needs a HIP port; skip for the CPU tier | 0-5% via VRAM fit; small quality |
| YAQA ([2505.22988](https://arxiv.org/abs/2505.22988)) | Cornell lead | arXiv 2025 | KL/Fisher Kronecker Hessian rounding, quantizer-agnostic | −30% KL vs prior | cornell-relaxml/yaqa | Research: better imatrix | Quality only |
| QuIP# ([2402.04396](https://arxiv.org/abs/2402.04396)) / QuIP ([2307.13304](https://arxiv.org/abs/2307.13304)) | Cornell lead | ICML 2024 / NeurIPS 2023 | Incoherence + E8 lattice / LDLQ | superseded by QTIP | quip-sharp | No (no GGUF kernels) | 0 |
| Mamba in the Llama ([2408.15237](https://arxiv.org/abs/2408.15237)) | Cornell lead (Dao co) | NeurIPS 2024 | Hybrid distillation + hybrid SD kernel | >300 t/s Mamba-7B | jxiw/MambaInLlama, speculative_mamba | Superseded by ReplaySSM | 0 |
| Medusa ([2401.10774](https://arxiv.org/abs/2401.10774)) | Princeton lead | ICML 2024 | Parallel heads + tree + typical acceptance | 2.2x / 2.3-3.6x | FasterDecoding/Medusa | MTP is better; borrow typical acceptance for thinking tokens only | 5-15% on thinking (est.) |
| GTA/GLA ([2505.21487](https://arxiv.org/abs/2505.21487)) | Princeton lead | ICML 2025 | Attention variants for decode | up to 2x vs FlashMLA (H100) | Dao-AILab/grouped-latent-attention | No (needs pretraining) | 0 |
| Mamba-3 ([2603.15569](https://arxiv.org/abs/2603.15569)) | CMU/Princeton co-lead | ICLR 2026 | Complex state, MIMO, half state size | +0.6 points vs GDN at 1.5B | released kernels | Future models only | 0 now |
| Mamba-2/SSD ([2405.21060](https://arxiv.org/abs/2405.21060)) | Princeton/CMU co-lead | ICML 2024 | Chunked SSD matmul form | n/a | state-spaces/mamba | Check GDN prefill is chunked | Prefill only if currently serial |
| FlashAttention-3 ([2407.08608](https://arxiv.org/abs/2407.08608)) | Princeton co | NeurIPS 2024 | Hopper async + FP8 attention | n/a | Dao-AILab | No (Hopper-only) | 0 |
| SWE-agent ([2405.15793](https://arxiv.org/abs/2405.15793)) / mini-SWE-agent | Princeton lead | NeurIPS 2024 / 2025 | ACI design; bash-only minimal agent | 12.5% then; >74% Verified (mini) | SWE-agent org | Linter-guarded edits, windowed views, small tool set | Fewer wasted turns |
| HELMET ([2410.02694](https://arxiv.org/abs/2410.02694)) | Princeton lead | ICLR 2025 | Long-context eval; NIAH not predictive | 59 models, 7 categories | princeton-nlp/HELMET | Use RAG/re-rank subsets to validate quant/cache changes | Eval quality |
| ReAct / ToT ([2210.03629](https://arxiv.org/abs/2210.03629), [2305.10601](https://arxiv.org/abs/2305.10601)) | Princeton lead | ICLR 2023 / NeurIPS 2023 | Reason+act loop; tree search | n/a | n/a | ReAct = opencode already; ToT no | 0 |
| TokenButler ([2503.07518](https://arxiv.org/abs/2503.07518)) / xKV ([2503.18893](https://arxiv.org/abs/2503.18893)) | Cornell (Abdelfattah) lead, unconfirmed | arXiv 2025 | Learned token importance / cross-layer SVD KV | 7.6x offload latency / 3x KV | abdelfattah-lab | Only if long context evicts experts from VRAM | 0-5% |
| Critical Thinking ([2504.01935](https://arxiv.org/abs/2504.01935)) | Cornell lead | IJCNLP-AACL 2025 | Optimal reasoning length tied to task structure | n/a | n/a | Supports per-task thinking budgets | Conceptual |
