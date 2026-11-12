# Tsinghua University and Peking University papers for pancake / Avalon (Nov 2022 - Nov 2026)

Scope note. "Lead" means the first author or the senior/corresponding group is at Tsinghua (THU) or Peking University (PKU). "Co" means they appear only among the co-authors. "Unverified affiliation" means the affiliation comes from the lab or GitHub org and was not read in the PDF. Every number is the authors' own figure on their own hardware (mostly Xeon + AMX/AVX-512 + NVIDIA). Nothing below has been measured on the RX 7900 XT / 5700G box.

What prior reports already cover. Twilight (THU lead, Berkeley co) already appears in the US-universities report as the "safest add-on" for sparse attention. M5 there is the 10k-vs-200k decode measurement that gates any sparse-attention work. TEAL, CATS and the Berkeley in-expert gate mask cover activation sparsity. Quest, DuoAttention and FlashMoBA are in the MIT report. The repeats below are only there because THU or PKU led the work, or because they change an earlier verdict.

Machine facts behind the verdicts, taken from the US report: the GPU is busy for 15.9 ms of a 24.8 ms decode window, and the rest is waiting on CPU experts at ~33 GB/s. Experts are IQ2_XS at ~1.4 MB each. The link is PCIe 3.0, and `--pcie-frac` measured best at 0.

---

## Q1. KTransformers (THU MADSys + Approaching.AI): what has pancake not got yet, especially on AVX2 without AMX?

### Takeaway
Only one KTransformers idea transfers to the 5700G, and pancake already tried it as VERIFY_DEFER_CPU. That idea is Expert Deferral: KTransformers credits it with +33% decode on its main showcase and up to +45% across models, with HumanEval/MBPP/GSM8K/StrategyQA moving by at most ~2 points. KTransformers picks the deferred experts by **lowest routing score**, never on the final layer, and always keeps at least 2 experts immediate. Pancake's version defers by **residency** (CPU misses), which is a different and riskier rule. The rest of KTransformers' speed comes from AMX/AVX-512 kernels, NUMA tensor parallelism and CUDA-graph capture of the CPU hand-off, and none of that applies here. One small CPU-side idea is worth copying: per-expert barriers with the SiLU·mul combine fused into the gate/up stage, described by a later THU OSDI'26 paper on the KTransformers backend.

### Cited Findings
**KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models.** Hongtao Chen, Weiyu Xie, ... Mingxing Zhang (18 authors). THU MADSys **lead**, Approaching.AI co. SOSP '25, Seoul, Oct 2025, DOI 10.1145/3731569.3764843. No arXiv version. — [MADSys PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf), [code](https://github.com/kvcache-ai/ktransformers)

Headline results:
- 4.62-19.74x prefill and 1.25-4.09x decode over existing systems. — [PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- Decode without deferral: 2.42-4.09x over Fiddler and only **1.25-1.76x over llama.cpp**. The paper credits much of this to CUDA Graph capture, which cuts GPU invocation overhead "from over 20% to almost zero". — [PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)

How Expert Deferral works:
- In each MoE layer, the top-scoring routed experts ("immediate experts", e.g. the top-2 by routing score in the figure) feed layer k+1 directly. The rest ("deferred experts") run on the CPU while the GPU computes layer k+1 attention, and their output is added to the residual afterwards. Deferral is "not applied at the final layer to preserve critical information." — [PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- Profiling and tuning on DeepSeek-V3 (8 routed experts):
  - Without deferral, CPU utilization was 74%.
  - Deferring 3 experts cut single-layer time by 26% and raised end-to-end decode throughput by 33%.
  - Deferring 4 experts gave nothing more, because the CPU was saturated.
  - Their heuristic: defer the minimum number of experts that saturates the CPU, and keep at least 2 immediate experts. — [PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- Settings used and results:
  - Deferred experts: DS-V3 3 (BF16) / 6 (quantized), DS-V2 4, Qwen2-57B 2 (BF16) / 4 (quantized).
  - Gain: "up to 45%" more throughput, giving 1.66-2.56x over llama.cpp.
  - Accuracy: task scores move by no more than 2 points; the abstract says the average drop is at most 0.5%.
  - Deferral is **decode-only**: in prefill nearly every expert gets activated, so deferral "nearly doubles memory access footprints". — [PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)

CPU/GPU synchronization and kernels:
- The CPU-GPU hand-off sits inside the CUDA graph. Both submit and sync go through `cudaLaunchHostFunc`, and a lock-free queue feeds the CPU workers, so the host is never interrupted. — [PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- CPU kernels: AMX tiles for high arithmetic intensity, plus a lightweight AVX-512 kernel for decode, chosen by tokens per expert. NUMA-aware TP splits every expert's weights across sockets. — [PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)

Repo timeline:
- Added in 2025: an **AVX2-only CPU backend for kt-kernel** (dated 2026-03-26 in the repo), ROCm GPU support (2025-03-15), Qwen3-Next support (2025-09-11), and a 2026-01-22 "CPU-GPU expert scheduling" with hot experts on GPU and cold experts on CPU. — [GitHub README](https://github.com/kvcache-ai/ktransformers)
- kt-kernel is now exposed to SGLang, where expert deferral overlaps CPU expert work with GPU attention. — [LMSYS blog](https://lmsys.org/blog/2025-10-22-KTransformers)

**Achieving Cloud-Grade SLOs for Local Mixture-of-Experts Inference through CPU-GPU Hybrid Design.** Wenxin Wang, Peng Qu, Youhui Zhang (THU **lead**), with Yule Hou and Yu Ji (Xingyun IC). OSDI '26, July 2026. — [USENIX PDF](https://www.usenix.org/system/files/osdi26-wang-wenxin.pdf)
- Stream-loading prefill (SLP) pipelines expert-weight transfer with GPU kernels: 1,200 tok/s prefill, and 32K prompts within 30 s. Two RTX 5090s reach 1,800 tok/s. — [PDF](https://www.usenix.org/system/files/osdi26-wang-wenxin.pdf)
- AVX-512 FP8 GEMV with post-scaling, sustaining up to 947 GB/s on dual-socket DDR5. Decode reaches 28 tok/s on INT4 DeepSeek, versus 22 for KTransformers and 14 for ik_llama.cpp. — [PDF](https://www.usenix.org/system/files/osdi26-wang-wenxin.pdf)
- Fine-grained CPU parallelism: each gate/up GEMV is sliced into m output segments per expert, giving m×k independent tasks spread across cores. **Per-expert barriers** replace the global barrier, and the combine op is fused into stage one, which "cuts synchronization overhead". — [PDF](https://www.usenix.org/system/files/osdi26-wang-wenxin.pdf)

**HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management for Efficient MoE Inference.** Shuzhang Zhong, Yanfan Sun, Ling Liang, Runsheng Wang, Ru Huang, Meng Li. PKU **lead** (Institute for AI and School of Integrated Circuits). DAC 2025. [arXiv 2504.05897](https://arxiv.org/abs/2504.05897)
- Combines a dynamic intra-layer CPU/GPU split, impact-driven inter-layer prefetch and a score-based expert cache, built on kTransformers. Average speedup over the state-of-the-art hybrid framework is 1.33x prefill and 1.70x decode. Code: [PKU-SEC-Lab/HybriMoE](https://github.com/PKU-SEC-Lab/HybriMoE). — [arXiv](https://arxiv.org/abs/2504.05897), [HF papers](https://huggingface.co/papers/2504.05897)

**AdapMoE: Adaptive Sensitivity-based Expert Gating and Management for Efficient MoE Inference.** Same PKU-SEC-Lab group (Zhong, Liang, Wang, Wang, Huang, Li). ICCAD 2024 venue is unverified; arXiv Aug 2024. [arXiv 2408.10284](https://arxiv.org/abs/2408.10284)
- Uses a sensitivity-based adaptive number of experts per token (about 25% fewer activated experts than top-2), prefetch from later-layer gating, and DP-allocated cache. 1.35x over prior expert-management methods on Mixtral with no reported accuracy loss. — [arXiv](https://arxiv.org/html/2408.10284v1)

**MoBiLE: Efficient MoE Inference on Consumer GPU with Mixture of Big Little Experts.** Yushu Zhao, ..., Shaojun Wei, Yang Hu, Shouyi Yin. THU **lead** per search snippet; affiliation not on the abstract page. arXiv Oct 2025. [arXiv 2510.12357](https://arxiv.org/abs/2510.12357)
- Unimportant tokens use half the experts; important tokens keep the full set, with a fallback/prefetch mechanism. 1.60-1.72x on a consumer GPU with "negligible" accuracy loss. — [arXiv](https://arxiv.org/abs/2510.12357)

### Inferences
**Expert Deferral is the only KTransformers mechanism with a plausible gain on this machine.** The ceiling is the GPU idle slice of the window (~9 ms of 24.8 ms) that CPU work could overlap.
- **Selection rule.** KTransformers defers the **lowest-score** experts. Pancake's VERIFY_DEFER_CPU defers by **miss status**, which can defer a top-1 expert. The KTransformers accuracy evidence (≤2 points) covers only score-ordered deferral.
- **Proposed fix when VERIFY_DEFER_CPU is repaired:** defer a CPU miss only when it is not in the top-2 by routing weight, and never on the last layer. Misses that are top-2 run immediately on the CPU, as today.
- **Gain estimate.** With ~half of decode on the CPU, a KTransformers-style 26% per-layer cut would be worth roughly +15-30% decode. That is an inference from their DS-V3 profile, not a measurement.
- **Verify windows.** Pancake verifies multi-token windows (MTP / prompt lookup). KTransformers found deferral counter-productive in prefill because the deferred set covers almost all experts. Expect the benefit to shrink as the verify window grows, so measure at window sizes 1, 2, 4 and 8.

**On the HIP breakage.** KTransformers puts both submit and sync inside the graph with `cudaLaunchHostFunc`. HIP has `hipLaunchHostFunc`, but pancake measured no gain from HIP graphs, so the useful lesson is the protocol: a lock-free queue plus a device-side wait at layer k+1 rather than at layer k. Graphs themselves are not the point.

**The CPU kernels give nothing on the 5700G.** AMX is absent, AVX-512 is absent and NUMA is single-socket. The KTransformers AVX2 backend (2026-03) targets INT4/INT8 formats, while pancake already reads 2-bit IQ2_XS rows, so it is unlikely to beat pancake's kernels on bytes per token.
- Cheap audit: diff kt-kernel's AVX2 thread partitioning against pancake's.
- Worth trying from OSDI'26: per-expert barriers with SiLU·mul fused into the gate/up pass. This matters only if pancake's CPU path currently uses a global barrier between gate/up and down across all missed experts. Expected gain: small, a few percent of CPU time, and only on the synchronization share.

**HybriMoE and AdapMoE mostly overlap with things pancake has or has tested.**
- The score-based cache resembles ClassLRU scoring (tested, no gain).
- The dynamic CPU/GPU split resembles `--pcie-frac` (best at 0 on PCIe 3.0).
- Their 1.70x is measured against a kTransformers baseline without adaptive caching; pancake already has an adaptive tier.

**AdapMoE and MoBiLE as a lossy opt-in.** "Fewer experts for insensitive tokens" is a principled version of STRATA_ROUTE_PRIOR. Concretely: drop a missed expert whose renormalized routing weight is below a threshold, instead of computing it on the CPU. This cuts CPU bytes directly, unlike the cache-hit work. It must be gated by the token-forcing KL harness.

**Mooncake** (Moonshot lead, THU MADSys co) is a multi-node KVCache-disaggregated serving architecture and has no single-user decode relevance. Its only transferable idea is an NVMe tier for parked-conversation KV/GDN state, which pancake can design without the paper.

### Gaps
- The KTransformers paper does not state how the deferred experts' output is scaled; it is presumably added unchanged one layer later. I could not extract equations 1-3 cleanly from the PDF.
- There are no KTransformers results on hybrid GDN/linear-attention models with deferral. Qwen3-Next support exists in the repo, but no deferral accuracy data for it was found.
- I did not read the AVX2 kt-kernel code, so its kernel design is unknown.
- I did not read HybriMoE's full text, so the exact baseline behind 1.70x and its cache-score formula are unconfirmed.
- MoBiLE's construction of the "little" experts and its affiliation details were not on the abstract page.

---

## Q2. Activation sparsity in MoE experts: ProSparse and other THU work, compared with the planned Berkeley / TEAL / CATS study

### Takeaway
THUNLP's activation-sparsity line (ProSparse, Sparsing Law, BlockFFN) requires **ReLU-ification or pre-training**, so none of it can be applied post-hoc to Qwen3.8-Flash-Next's SwiGLU experts. The planned training-free study (Berkeley in-expert gate mask first, TEAL second, CATS dropped) stays the right plan. THU work adds two useful things to it:
- Sparsing Law's **CETT-PPL-1%** metric gives a principled per-layer threshold for that study.
- BlockFFN's **chunk-level sparsity** result warns that sparsity collapses when it is unioned over the tokens of a verify window.

### Cited Findings
**ProSparse: Introducing and Enhancing Intrinsic Activation Sparsity within LLMs.** Chenyang Song, Xu Han, Zhengyan Zhang, Shengding Hu, ..., Zhiyuan Liu, Maosong Sun. THUNLP **lead**; affiliations not shown on the abstract page, lab inferred. arXiv Feb 2024, v7 Jan 2025. [arXiv 2402.13516](https://arxiv.org/abs/2402.13516)
- Method: replace SiLU with ReLU, then continue training with progressive sparsity regularization whose factor rises along multi-stage sine curves.
- Sparsity: 89.32% (LLaMA2-7B), 88.80% (LLaMA2-13B), 87.89% (MiniCPM-1B), at performance "comparable" to the Swish originals. ReluLLaMA-7B reaches 66.98% for comparison.
- Up to 4.52x inference speedup. — [arXiv](https://arxiv.org/abs/2402.13516)

**Sparsing Law: Towards Large Language Models with Greater Activation Sparsity.** Yuqi Luo et al. THUNLP **lead**. ICML 2025 (PMLR 267). [arXiv 2411.02335](https://arxiv.org/abs/2411.02335)
- Proposes the CETT-PPL-1% sparsity metric.
- Findings: ReLU is more effective than SiLU for sparsity; sparsity follows a convergent power law in training data; a small width/depth ratio helps; sparsity is largely insensitive to parameter scale.
- Code: [thunlp/SparsingLaw](https://github.com/thunlp/SparsingLaw). — [ICML](https://proceedings.mlr.press/v267/luo25i.html), [arXiv](https://arxiv.org/abs/2411.02335)

**BlockFFN: Towards End-Side Acceleration-Friendly MoE with Chunk-Level Activation Sparsity.** Chenyang Song, Weilin Zhao, Xu Han, Chaojun Xiao, Yingfa Chen, Yuxuan Li, Zhiyuan Liu, Maosong Sun. All authors at THU (**lead**). COLM 2025 per search summary; the arXiv page lists no venue. [arXiv 2507.08771](https://arxiv.org/abs/2507.08771)
- Design: a ReLU + RMSNorm router; non-gated Swish experts (a gated variant "severely reduces sparsity"); an activation-locality loss; a chunk-sparsification loss.
- Sparsity: token-level ~80-85% (80.54% at Small scale); 8-token chunk-level ~71-76% (71.38%). — [arXiv HTML](https://arxiv.org/html/2507.08771v2)
- Speed: CUTLASS kernels combining chunk sparsity with speculative decoding (32 draft tokens) on a Jetson Orin NX. 47.17 tok/s versus 12.84 for the AR baseline, which is **3.67x**. The token-level kernel alone gives 3.14x and EAGLE-2 gives 1.74x.
- Requirements: all models are pre-trained by the authors, and post-hoc conversion is not described.
- Code: [thunlp/BlockFFN](https://github.com/thunlp/BlockFFN). — [arXiv HTML](https://arxiv.org/html/2507.08771v2)

### Inferences
**ProSparse and BlockFFN cannot be used as-is.** Both need training, in ProSparse's case continued training at a scale that is impossible on one 20 GB card. Qwen3.8-Flash-Next's experts are gated SwiGLU, which is exactly the setup Sparsing Law and BlockFFN say is less naturally sparse. This supports the earlier finding that the Berkeley in-expert gate mask, which works on gated experts, is the variant to test first.

**Concrete add-on to the planned offline study: union sparsity.** Measure **union sparsity over the verify window**, meaning the fraction of FFN channels that are inactive for all w tokens, for w = 1, 2, 4, 8.
- BlockFFN, even after training *for* locality, only keeps ~71% at w = 8, against ~80% at w = 1.
- An untrained Qwen will degrade faster.
- Pancake's CPU decode mostly runs inside MTP/lookup verify windows, so the per-token sparsity numbers from TEAL or Berkeley overstate the bytes saved.
- This measurement is free: it only needs another column in the same offline pass.

**Using CETT-PPL-1% in the study.** The metric picks, per layer, the threshold that keeps the cumulative error of the tail truncation within a PPL tolerance. It can replace or cross-check TEAL's greedy per-layer allocation.

**Expected gain.** No THU work changes the earlier estimate. If 50-85% sparsity holds at w ≈ 1-2, the in-expert mask still has the largest potential CPU-byte reduction of any candidate. THU work only adds the w > 1 caveat.

### Gaps
- I found no THU/PKU paper showing training-free activation sparsity on a gated, fine-grained MoE (512 experts) at 2-bit; the Berkeley 2026 work remains the only one.
- The ProSparse venue was not confirmed; no venue appears on arXiv.

---

## Q3. Sparse and long-context attention and KV-cache work for a hybrid GDN + attention model at 131-262k

### Takeaway
The THU/PKU sparse-attention methods with the best results all require **training**: NSA (PKU lead with DeepSeek), MoBA (Moonshot with THU co-authors) and InfLLM-v2 (THU/OpenBMB). They cannot be bolted onto Qwen3.8-Flash-Next.

Two training-free families remain:
- **Twilight** (THU lead), already in the US report and decode-oriented.
- **SageAttention / SpargeAttn** (THU-ML), INT8/INT4 quantized and sparse attention kernels aimed at prefill. They are written for NVIDIA tensor cores; a gfx1100 port would need WMMA iu8 work.

In a hybrid model only the full-attention layers (a minority) scale with context. So both families need the M5 measurement (decode and prefill at 10k vs 200k) before any work starts.

### Cited Findings
**Native Sparse Attention (NSA): Hardware-Aligned and Natively Trainable Sparse Attention.** Jingyang Yuan et al., with Wenfeng Liang. PKU **lead** first author; DeepSeek-AI and UW co. ACL 2025; the best-paper status rests on an aggregator, while Paper Digest lists it as "most influential". [arXiv 2502.11089](https://arxiv.org/abs/2502.11089)
- Design: compressed coarse tokens, selected fine tokens and a sliding window, combined by learned gates.
- Requires pre-training; the paper uses a 27B model. — [hyper.ai](https://hyper.ai/en/papers/acl__2025__2025.acl-long.1126), [Paper Digest](https://www.paperdigest.org/certificate/87d36fdd18aa81cd)
- The claimed "9x faster than FlashAttention at 64k decode" comes from a blog and is unverified. — [blog](https://learn-traeai.pages.dev/t/ai-engineering/phases/10-llms-from-scratch/17-native-sparse-attention)

**InfLLM-V2: Dense-Sparse Switchable Attention for Seamless Short-to-Long Adaptation.** Weilin Zhao et al. THU/OpenBMB **lead**. ICLR 2026. [arXiv 2509.24663](https://arxiv.org/abs/2509.24663)
- Dense attention for short inputs and trainable sparse attention for long ones, with no new parameters.
- Retains 98.1% (long-context) and 99.7% (CoT) of baseline performance at 4x faster inference. It underlies MiniCPM4/4.1.
- Still needs short-to-long fine-tuning. — [ICLR](https://proceedings.iclr.cc/paper_files/paper/2026/hash/716e60181e1782f5bd0a629033f85310-Abstract-Conference.html)

**SageAttention / SageAttention2 / SpargeAttn.** Jintao Zhang, Jianfei Chen et al. THU-ML **lead**. ICLR 2025 / ICML 2025 / ICML 2025. [arXiv 2410.02367](https://arxiv.org/abs/2410.02367), [arXiv 2411.10958](https://arxiv.org/abs/2411.10958)
- SageAttention2 quantizes Q and K to INT4 and P and V to FP8, at ~3x FlashAttention2 OPS on an RTX 4090. Layers where INT4 fails fall back to 8-bit.
- The kernel figures exclude quantization and smoothing overhead.
- SpargeAttn is a training-free sparse + quantized attention built on SageAttention2 for "any model".
- Code: [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention), CUDA only. — [arXiv 2411.10958](https://arxiv.org/html/2411.10958v3)

**Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning.** THU **lead**, Berkeley co. [arXiv 2502.02770](https://arxiv.org/abs/2502.02770)
- Already assessed in the US report as the safest training-free add-on, gated on M5. — [US report row](file:///home/dan/reports/US%20university%20papers%20for%20Avalon.md)

### Inferences
**Decode.** At 131-262k, decode cost on the attention layers is the KV read. Twilight's top-p budget (training-free) is the only THU decode method that applies. Its value depends on whether attention-layer KV reads are a significant part of the 24.8 ms window at 200k. That is the M5 measurement; there is no new reason to skip it.

**Prefill.** For 100k+ prompts after a conversation-cache miss, SageAttention-style INT8 QK^T could cut attention-layer prefill time. Only the attention layers benefit; GDN layers are linear-time.
- Porting requires a gfx1100 WMMA i8 kernel inside pancake's flash-attention path.
- Expected gain: modest. Total TTFT gain is roughly the attention share of long-prompt prefill times ~1.5-2x.
- The conversation cache already avoids most re-prefill in agent turns.
- Verdict: low priority, below Twilight.

**Ruled out:**
- NSA, MoBA and InfLLM-v2 need training; ignore them unless a future Qwen ships with them.
- PKU's PyramidKV-style KV eviction (cited from knowledge, not re-verified here) is lossy on mid-context code and tool outputs. The MIT report already rejects eviction methods for the same reason.

### Gaps
- I did not verify NSA's exact decode speedups or MoBA's THU co-author list in primary sources; the MoBA arXiv ID 2502.13189 comes from prior knowledge and was not re-fetched.
- I found no THU/PKU work specific to sparse attention in GDN/linear-attention hybrids.
- I found no ROCm or gfx1100 port of SageAttention.

---

## Q4. Agent and reasoning efficiency, plus speculative decoding

### Takeaway
**AdaptThink** (THU-KEG) shows that a reasoning model can learn to emit an empty `<think></think>` on easy problems, cutting cost while matching or improving accuracy. Pancake cannot run the RL, but its thinking-budget policy can copy the inference-time half: a zero-think budget on turn types that are trivially easy in an agent loop. In speculative decoding, **EAGLE-3** (with PKU-affiliated authors) and BlockFFN's chunk-sparse plus spec-decoding kernel are the relevant THU/PKU work. Pancake already has an MTP head and drafting policies, so a trained EAGLE-3 head is not worth the cost on a 512-expert MoE whose drafts trigger extra CPU expert work.

### Cited Findings
**AdaptThink: Reasoning Models Can Learn When to Think.** Jiajie Zhang, Nianyi Lin, Lei Hou, Ling Feng, Juanzi Li. THU **lead**. arXiv May 2025; EMNLP 2025 per one listing, unverified. [arXiv 2505.13417](https://arxiv.org/abs/2505.13417)
- An RL algorithm chooses between thinking and no-thinking modes based on difficulty; no-thinking means an empty think segment.
- Evaluated on math with R1-Distill-Qwen-1.5B.
- Code: [THU-KEG/AdaptThink](https://github.com/THU-KEG/AdaptThink). — [arXiv](https://arxiv.org/abs/2505.13417)

**EAGLE-3: Scaling up Inference Acceleration of LLMs via Training-Time Test.** Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang. NeurIPS 2025. [arXiv 2503.01840](https://arxiv.org/abs/2503.01840)
- Predicts tokens directly with multi-layer feature fusion: up to 6.5x over AR and ~1.4x over EAGLE-2, with code the best case. — [NeurIPS](https://neurips.cc/virtual/2025/poster/119930)
- The PKU affiliation of Li and C. Zhang comes from the EAGLE project's history and was **not confirmed** in this search.

**BlockFFN.** Its spec-decoding plus chunk-sparsity kernel gives 3.67x versus 1.74x for EAGLE-2 on an edge device (see Q2). — [arXiv HTML](https://arxiv.org/html/2507.08771v2)

### Inferences
**AdaptThink, inference-side only.** In an opencode loop, many turns are mechanical: acknowledging a tool result, issuing the next obvious read or grep, or formatting a final summary.
- A heuristic classifier on the previous turn's type (tool result short, no error, plan already stated) could set `reasoning_budget = 0` by pre-filling `<think></think>`.
- Measure the effect on task success with the existing agent harness; the paper's evidence is math-only on a 1.5B model.
- This complements the learned-budget idea in the MIT report.
- Expected gain: the fraction of thinking tokens on easy turns, which could be large in agent loops but is unmeasured.

**EAGLE-3: skip.** Pancake's bottleneck is CPU bytes for missed experts. Longer or tree drafts widen the verify window, and the expert union grows with it. That is the same effect that limits KTransformers' deferral in prefill and BlockFFN's chunk sparsity. Pancake's DraftPolicy, which already prices each window size, is the right control, and nothing from THU/PKU beats it.

### Gaps
- I did not research Ouroboros (THUNLP), Lookahead variants or Spec-Bench (PolyU/PKU, Heming Xia) in primary sources this round.
- Agent benchmarks from THU (AgentBench, ToolLLM) are known, but I did not re-verify their relevance to measuring local coding agents.
- I found no THU/PKU work on tool-call recovery or loop guards.
