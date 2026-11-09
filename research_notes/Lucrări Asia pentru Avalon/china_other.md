# Mainland Chinese universities (excluding Tsinghua/PKU) — AI-systems papers relevant to pancake / Avalon

Scope note: about 20 tool calls. I verified primary pages (arXiv HTML) for ProMoE, HOBBIT, SP-MoE and BuddyMoE. The other entries come from arXiv abstracts or search snippets and are marked as such. Already covered in /home/dan/reports, so not repeated in depth: TEAL, CATS, Deja Vu, fMoE/FineMoE, ExFlow, SiDA, eMoE, MoE-Infinity, Fiddler, MoE-Lightning. BuddyMoE was only name-checked before and is detailed here. Things pancake has already tested without gain (HIP graphs, Vulkan, TurboQuant KV, L3 prefetch, ClassLRU, reuse-aware admission, DraftExpert windows) are not recommended again.

Excluded under the rules but encountered: AdapMoE (PKU-SEC-Lab, code github.com/PKU-SEC-Lab/AdapMoE — [arXiv 2408.10284](https://arxiv.org/abs/2408.10284v1)); HybriMoE and kTransformers (PKU and Tsinghua, from my own knowledge, not re-verified); D²MoE (Song Guo's group, MobiCom 2025, probably led from Hong Kong, so excluded — [arXiv 2504.15299](https://arxiv.org/abs/2504.15299)). D²MoE's "Matryoshka weight quantization" (nested bit-widths stored once) and per-token bit-width routing are still worth knowing as an idea.

Pancake reference numbers used below (from the assignment): about 126 t/s decode, which is about 7.9 ms per token. About 44% of experts are VRAM-resident. Hit rates are replay tier 79.6%, LRU 89.5%, Belady 94.7%. The CPU computes misses, which is about half of decode time. The link is PCIe 3.0.

---

## Q1. PowerInfer / PowerInfer-2 hot–cold split and predictors: do they apply to MoE experts on AVX2, and how do they relate to the planned activation-sparsity study (Berkeley gate mask / TEAL / CATS)?

### Takeaway
PowerInfer's hot/cold split already exists in pancake at expert granularity: hot experts sit in VRAM and the CPU computes the cold ones. The new part would be neuron-level sparsity inside the CPU-computed cold experts. That needs either ReLU-fied models (IPADS's TurboSparse/dReLU route, which means retraining) or threshold sparsity on SwiGLU (the TEAL/CATS route, plus Wuhan's CHESS refinement with per-channel thresholds). The useful IPADS ideas for pancake are SmallThinker's pre-attention router (as a predictor signal) and PowerInfer-2's I/O-compute pipelining for the NVMe tier. Neither the neuron predictors nor the ReLU dependence carries over directly.

### Cited Findings
- **PowerInfer** (Yixin Song, Zeyu Mi, Haotong Xie, Haibo Chen; SJTU IPADS, **lead**; SOSP 2024; arXiv 2312.12456). A small set of "hot" neurons fire consistently across inputs and are preloaded on the GPU. Input-dependent "cold" neurons are computed on the CPU. Online predictors decide which neurons activate per input — [SOSP paper PDF](https://ipads.se.sjtu.edu.cn/_media/publications/song-sosp24.pdf). On OPT-30B, a single RTX 4090 reaches 82% of A100 generation speed — [arXiv](https://arxiv.org/abs/2312.12456). Per the authors' slides, about 26% of neurons account for about 80% of computation — [SOSP slides](https://ipads.se.sjtu.edu.cn/_media/talk/sosp2024_powerinfer_slides.pdf). A secondary summary says the predictors are small per-layer MLPs in the Deja Vu style, with size varying by layer — [notes (secondary, Chinese)](https://higiraffe.github.io/notes/Paper%20Reading%20Notes/SOSP%202024/PowerInfer.html). Code: github.com/SJTU-IPADS/PowerInfer (from knowledge; also mirrored at [gitee](https://gitee.com/xhal/PowerInfer)).
- **PowerInfer-2** (Zhenliang Xue et al.; SJTU IPADS, **lead**; arXiv 2406.06282, 2024). The unit is the "neuron cluster". Dense clusters run on the NPU and sparse clusters on the CPU. Flash reads are overlapped with compute, and a segmented neuron cache reduces I/O. Most models had 50% of FFN weights offloaded to flash, and the 47B model needed at least 75% offloaded. TurboSparse-Mixtral-47B (a sparsified variant, not stock Mixtral) runs at 11.68 t/s on a phone, 21.2× llama.cpp. The headline speedups vary by version (29.2× in v1/v2, 27.8× in v3) — [arXiv v1](https://arxiv.org/pdf/2406.06282v1), [HF page](https://huggingface.co/papers/2406.06282). A secondary review gives I/O as 13.7% of the critical path versus 76.7% for LLM-in-a-Flash (not verified against the paper) — [review](https://liner.com/review/powerinfer2-fast-large-language-model-inference-on-smartphone).
- **SmallThinker** (SJTU IPADS + SJTU School of AI + Zenergize AI; IPADS **lead/co-lead with a company**; arXiv 2507.20984, Jul 2025). These are natively trained on-device MoE models (4B-A0.6B, 21B-A3B). The router is placed **before attention**, so expert weights can be fetched from SSD while attention runs. A global load-balance loss trained in data-parallel makes experts specialise, so per-task working sets stay small and cacheable. The authors self-report up to 86× the decode speed of Qwen3-30B-A3B under offload. Code: github.com/SJTU-IPADS/SmallThinker — [arXiv](https://arxiv.org/html/2507.20984v1), [HF README](https://huggingface.co/Tiiny/SmallThinker-4BA0.6B-Instruct/blob/main/README.md).
- **CHESS** (Junhui He, Shangyu Wu, Weidong Wen, Chun Jason Xue, Qingan Li; Wuhan University **lead** with CityU HK / MBZUAI; arXiv 2409.01366; venue reported as EMNLP 2024, not confirmed). It uses per-channel activation thresholds in the FFN instead of the single threshold in CATS, and sparsifies only selected attention submodules. It loses less quality than CATS while activating fewer parameters, with up to 1.27× end-to-end speedup from custom CPU sparse kernels (spvmm/vmmsp). Code is public on GitHub — [arXiv v2](https://arxiv.org/html/2409.01366v2), [review](https://www.themoonlight.io/review/chess-optimizing-llm-inference-via-channel-wise-thresholding-and-selective-sparsification).

### Inferences
- **Expert-level hot/cold:** pancake already does this. PowerInfer offers nothing new at this level except its offline "impact"-based placement solver, which the per-domain working sets already approximate.
- **Neuron-level sparsity inside CPU-computed experts** is where the PowerInfer idea could pay off. The CPU computes about half of decode, so a 30–40% FLOP/byte reduction in the miss path could give roughly 15–20% end-to-end. That is a speculative upper bound and assumes the CPU path is bandwidth- or compute-bound on the expert weights. Caveats specific to Q2_0 on AVX2:
  - **gate/up output rows:** skipping these needs a predictor before computing them, as in PowerInfer and Deja Vu. With SwiGLU you cannot know which rows matter until gate is computed, unless you use a CATS-style "compute gate, threshold, then compute up and down only for survivors". That saves about 2/3 of the up+down bytes for the surviving fraction.
  - **down_proj columns:** skipping these needs a column-major (transposed) layout, because Q2 blocks run along the input dimension. Plan this layout change before running the Berkeley/TEAL/CATS study, or the measured sparsity will not turn into bandwidth savings.
  - **Thresholds:** CHESS per-channel thresholds are a cheap upgrade over CATS's single threshold for that study. They are calibration-only, with no training.
- **TurboSparse/dReLU** (IPADS) needs continued pretraining, which is not feasible for Qwen3.8-Flash-Next. Rule it out.
- **Pre-attention router:** pancake cannot retrain the model, but it can cheaply test the *signal*. Run layer *l*'s router on the attention-input hidden state (pre-attention, available about one attention block earlier) and measure recall against the true top-k. A good result gives a free prefetch lead time of one attention plus one GDN block.
- **PowerInfer-2's flash pipeline** (cluster-granular reads overlapped with compute, plus a segmented cache) maps onto pancake's NVMe → pinned-RAM tier. It only matters if NVMe misses appear on the decode critical path in coding sessions. Measure NVMe-tier hit frequency first.

### Gaps
- I could not verify PowerInfer's RTX 4090 speedup versus llama.cpp. I recall "up to 11.69×", but the search did not confirm it.
- PowerInfer's predictor accuracy and recall figures were not extracted.
- None of these papers evaluates AVX2 desktop CPUs computing quantized (2-bit) experts with neuron sparsity.

---

## Q2. Which expert prefetch / caching / router-prediction papers report batch-1 gains that could close the tier-vs-LRU gap, given PCIe 3.0 promotion cost?

### Takeaway
The strongest batch-1 evidence comes from SJTU's ProMoE and HOBBIT and from Sun Yat-sen's Fate and SP-MoE. All use **cross-layer gate prediction**: feed layer *l*'s gate input, or a draft model's hidden state, into the router of layer *l+1…l+3*. They report 84.7–97% next-layer top-k accuracy on small-expert-count models. Measured over plain LRU, the decode gains are modest: ProMoE in llama.cpp is 1.09× average and 1.17× max over LRU on PCIe 4.0. On PCIe 3.0 with CPU fallback the gain will be smaller still. The biggest structural lesson is ProMoE's: when the CPU computes misses, prefetch only helps when it is precise. Prefetching too much hurts, as SP-MoE also shows. For pancake the actionable items are (a) replace the replay tier with a cross-layer-gate/MTP-hidden-state predictor driving *priority* promotions with chunking and early preemption, and (b) reorder per-layer execution so cached experts run first.

### Cited Findings
- **ProMoE** (Xiaoniu Song, Zihang Zhong [ZJU intern], Rong Chen, Haibo Chen; SJTU IPADS, **lead**; arXiv 2410.22134 v3, Sep 2025; venue not stated).
  - Setup: RTX 4090 on PCIe 4.0, about 23.9 GB/s measured. Models: DeepSeek-MoE-16B, DS-V2-Lite, Qwen1.5-MoE, Qwen2-57B (INT4), Mixtral (INT4).
  - Predictor: a learned 2-layer MLP per layer (about 2M parameters) on layer inputs, run on the CPU at about 200 µs per prediction. Average accuracy is 84.7%, versus 58.3% for token-based and 66.9% for skip-based predictors. "Stride" prefetch (one extra layer ahead) costs about 5% accuracy.
  - Coordination: *chunked prefetch* (one chunk per linear layer, so an urgent fetch waits at most one chunk); *early preemption* (a hook after the gate cancels pending speculative fetches for the layer and turns misses into precise fetches); *reordered inference* (run cached experts first, then in-flight ones, then unstarted ones; this is legal because the expert sum is order-independent).
  - Batch-1 decode: llama.cpp gains 1.49× over a static cache but only **1.09× average (up to 1.17×) over LRU**; transformers gains 1.31× over LRU.
  - ProMoE does not compute misses on the CPU. On Mixtral, llama.cpp's CPU-offload baseline beats ProMoE in decode because CPU compute was cheaper than PCIe transfer.
  - Code: [github.com/promoe-opensource/promoe](https://github.com/promoe-opensource/promoe). Sources: [arXiv v3](https://arxiv.org/html/2410.22134v3).
- **HOBBIT** (Peng Tang, Xiaofeng Hou, Yifei Pu, Jing Wang, Chao Li, Minyi Guo [SJTU, EPCC/Minyi Guo group] + Jiacheng Liu, Pheng-Ann Heng [CUHK]; SJTU **lead**; arXiv 2411.01433, Nov 2024).
  - Built on llama.cpp (about 8k lines of C/C++).
  - Prefetcher: the current layer's gate input is pushed through the next *p* gates at once ("stacking"). Top-1 accuracy is about 96% for the next layer and about 90% for 2–3 layers ahead (Mixtral). Recommended *p* is 1–3.
  - Cache: a weighted mix of LRU, sequence-LFU, "high-precision-use frequency" and farthest-layer-distance. It cuts miss penalty by only 4.69–8.68% versus LRU, and sequence-level LFU adds about 4.5% hit ratio.
  - RTX 4090 + CPU versus Fiddler: 0.99× (Mixtral) and 1.46× (Phi-MoE). Versus llama.cpp: 1.31–1.42×. No code URL — [arXiv v2](https://arxiv.org/html/2411.01433v2).
- **Fate** (Zhiyuan Fang et al.; Sun Yat-sen University [Wuhui Chen, Zibin Zheng] + Peng Cheng Lab + Huawei; SYSU **lead**; arXiv 2502.12224, Feb 2025; v1 title "Accurate Expert Predictions in MoE Inference via Cross-Layer Gate"). Adjacent-layer gate inputs drive prefetch at 97.15% accuracy with no extra GPU cost. A "shallow-favouring" expert cache reaches about 99% hit rate, and quantization keeps frequent experts at higher precision. Decode is up to 4.1× faster than load-on-demand and 2.2× faster than EAP. These figures come from the abstract and secondary summaries; I did not read the full text — [arXiv v2](https://arxiv.org/html/2502.12224v2), [review](https://www.themoonlight.io/review/accurate-expert-predictions-in-moe-inference-via-cross-layer-gate).
- **SP-MoE** (Liangkun Chen, Zijian Wen, Tian Wu, Xiaoxi Zhang [Sun Yat-sen] + Chuan Wu [HKU]; SYSU **lead**; arXiv 2510.10302, Oct/Nov 2025; no venue or code found).
  - Batch 1, speculative decoding plus expert prefetch: each **draft-model layer's attention output is fed into the target layer's gate** to predict the experts verification will need. Top-1 prediction is about 88–89%.
  - A "cutoff layer" limits prefetch depth under memory and overlap constraints, and too much prefetch adds I/O contention (U-shaped curve, optimum around layer 20 for Mixtral/Phi).
  - TPOT is 1.07–3.5× better than prior systems. Averages: −34% versus Mixtral-Offloading, −19% versus MoE-Infinity, −12% versus AdapMoE. On RTX 3090/4090, PCIe 4.0.
  - Table 3 hit rates look inconsistent with the claims and need checking — [arXiv v2](https://arxiv.org/html/2510.10302v2).
- **MoE-SpeQ** (arXiv 2511.14102, Nov 2025; affiliation not verified). A small quantized copy of the MoE serves as the drafter. It shares non-expert weights and the KV cache with the target and predicts future tokens' experts for prefetch. Up to 2.34× over state-of-the-art offloading on Phi-MoE — [arXiv](https://arxiv.org/html/2511.14102v1).
- **BuddyMoE** (all authors SJTU, **lead**; PPoPP 2026; arXiv 2511.10054). On a prefetch miss it substitutes a GPU-resident "buddy" expert chosen offline by conditional co-activation share with a cumulative-frequency threshold. On DS-V2-Lite (A100) throughput rises 7.4% at a 75% cache rate and 10.3% at 37.5%. ARC accuracy drops 0.735 → 0.695 (about 5%), so the method is lossy, unlike what the abstract claims — [arXiv](https://arxiv.org/html/2511.10054v1).
- **Related 2026 items surfaced but not examined**: DraftExpert (alphaxiv 2607.24434; already tested in pancake as "DraftExpert windows", no gain) and SPICE (2608.21240, speculative prefetch with an aligned draft) — [search listing](https://www.alphaxiv.org/abs/2607.24434v1), [emergentmind](https://www.emergentmind.com/papers/2608.21240).

### Inferences
- **The cost of a promotion on PCIe 3.0.** This is my arithmetic; the expert shape is an assumption to check against the GGUF. If a Qwen3-Next-style expert is 3 × 2048 × 512 parameters at about 2.1 bits/weight, it is about 0.8 MB, or about 65–70 µs at about 12 GB/s on PCIe 3.0 x16. At top-10 × 48 layers, a 10-point hit-rate gap is about 48 extra promotions per token, or about 3.3 ms. That is about 40% of a 7.9 ms token if serialised. So treating promotions as free in the 79.6/89.5/94.7 replay numbers overstates achievable gains. Only promotions that overlap with GDN/attention compute, and that beat the CPU-compute alternative, are worth issuing. ProMoE's own data supports this: LRU in llama.cpp is already close (1.09×), and CPU compute beat PCIe transfer on Mixtral.
- **Best concrete experiment (low cost, high information).** Offline on pancake's routing traces, measure top-10 recall@k (k = 10, 16, 24) of:
  - (a) layer *l*'s pre-MoE hidden state pushed through router *l+1* and *l+2* (Fate/HOBBIT "stacking");
  - (b) the pre-attention hidden state through router *l* (SmallThinker signal);
  - (c) the MTP head's hidden state through all routers for token *t+1* (the SP-MoE analog, using pancake's existing MTP head as the "draft").

  With 512 experts and top-10, expect recall well below the 90–97% reported for 8–64-expert models (inference). The residual-stream similarity across layers in a hybrid GDN model is unknown. If recall@16 ≥ 80%, use it to (i) drive *priority* promotions and (ii) reorder execution so GPU-resident experts run while CPU misses are computed in parallel.
- **Scheduler upgrades worth porting regardless of predictor:** ProMoE's chunked transfer with early preemption, and reordered execution (cached → in-flight → CPU). These are deterministic, cost no accuracy and fit pancake's tiering. Expected gain is small but real, perhaps 3–8% decode (inference, consistent with ProMoE's 1.09× over LRU and HOBBIT's 4–9% miss-penalty reduction).
- **The replay tier underperforms LRU (79.6% vs 89.5%).** Every cache-policy paper here (HOBBIT LRU+LFU+layer-distance, Fate shallow-favouring) builds *on* recency rather than replacing it. Suggestion: make LRU the base and let the replay or predictor signal only *protect* predicted experts from eviction (HOBBIT's rule) rather than order the whole cache. Note that ClassLRU scoring and reuse-aware admission already failed in pancake, so expect diminishing returns. Fate's "shallow-favouring" (bias the cache toward early layers, where prediction lead time is shortest) is an untested variant worth one run.
- **BuddyMoE** is lossy (about 5% ARC drop) and overlaps with STRATA_ROUTE_PRIOR. It is mainly useful as a principled source for the substitution table (conditional co-activation share), if route-prior is kept.
- **SP-MoE / MoE-SpeQ caution:** verifying k draft tokens at batch 1 touches the union of experts across k+1 positions. The more draft tokens, the more misses, so the MTP/PLD draft length should be tuned jointly with cache capacity. SP-MoE's cutoff-layer U-curve is the empirical evidence for this.

### Gaps
- No paper here evaluates PCIe 3.0, AMD GPUs, 512-expert layers, or CPU-computes-misses combined with prefetch. All gains are on PCIe 4.0 NVIDIA and mostly ≤64 experts.
- ProMoE's LRU hit rates appear only in figures (not extracted). Fate's full text, including its cache-size settings behind "99% hit", was not read.
- MoE-SpeQ affiliation is unknown. ExpertFlow (2410.17954), which I believe is HKUST-GZ and so excluded, was not verified.

---

## Q3. Mixed-precision expert quantization (hot experts at higher bit-width) and NVMe offload work

### Takeaway
SJTU's HOBBIT is the closest match: low-precision replicas are used for *less critical* missed experts, based on gate weights. MxMoE (ICML 2025) allocates bit-width per linear block and per expert-activation frequency. FloE (ICML 2025, Zhejiang-funded) compresses experts and adds sparsity predictors to cut transfer bytes. For pancake, the transferable idea is a **gate-weight-driven precision/skip policy for misses**: compute low-weight missed experts on the CPU at even lower precision, or skip them. NVMe work (PowerInfer-2, SmallThinker) is relevant only if NVMe ever appears on the decode path.

### Cited Findings
- **HOBBIT (precision policy).** Experts are ranked by gate weight. An "unimportance" score is the cumulative gate weight of higher-ranked experts. With T1 = 0.6 an expert is loaded in low precision, with T2 = 0.9 it is skipped, and the top-1 expert is always high precision. On Mixtral the split is 67% high, 30% low, 3% skipped. Pairs are FP16/INT4 on the 4090 and INT8/INT2 on Jetson. Accuracy drops ≤1% on GSM8K/TruthfulQA (GSM8K 0.52 → 0.51 with INT4) — [arXiv v2](https://arxiv.org/html/2411.01433v2).
- **MxMoE** (Haojie Duanmu et al.; ICML 2025, PMLR 267:14793–14806; arXiv 2505.05799; affiliation not confirmed in the snippets, believed Shanghai AI Lab / SJTU). Quantization sensitivity varies across linear blocks, and expert activation frequency varies. Bit-widths are chosen jointly over sensitivity, activation frequency and hardware cost, with auto-generated mixed-precision GroupGEMM kernels. At 2.25-bit, Wikitext-2 perplexity is 2.4 lower than GPTQ. Code: [github.com/cat538/MxMoE](https://github.com/cat538/MxMoE) — [arXiv](https://arxiv.org/html/2505.05799v1), [PMLR](https://proceedings.mlr.press/v267/duanmu25a.html).
- **FloE** (Yuxin Zhou, Zheng Li et al.; ICML 2025; arXiv 2505.05950; the acknowledgments cite Zhejiang provincial funding, but the exact affiliation is not verified). It compresses experts' internal matrices and uses low-cost sparsity predictors to cut PCIe traffic. It reports 9.3× per-expert compression, Mixtral-8×7B on an 11 GB GPU, and 48.7× over DeepSpeed-MII on an RTX 3090, with 4.4–7.6% average quality loss. arXiv v1 reported 2.6×, so check versions — [arXiv](https://arxiv.org/abs/2505.05950), [PMLR](https://proceedings.mlr.press/v267/zhou25j.html).
- **Fate** also keeps frequently used experts at higher precision and others lower, according to a summary — [review](https://www.themoonlight.io/review/accurate-expert-predictions-in-moe-inference-via-cross-layer-gate).
- **T-MAC** (Jianyu Wei [USTC, first author, with MSRA] et al.; arXiv 2407.00088; 2025, EuroSys per ACM reference format, not confirmed; *note: co-led with a company lab*). Table-lookup low-bit GEMV/GEMM on CPU with no dequantization. Kernels are up to 6.6× (average 3.6×) faster than llama.cpp, and 2-bit Llama-2-7B runs 2.8× faster end to end, with 60–70% energy reduction. Code: [github.com/microsoft/T-MAC](https://github.com/microsoft/T-MAC) — [arXiv](https://arxiv.org/html/2407.00088v2).

### Inferences
- **HOBBIT's rule translated to pancake** (CPU computes misses, so the cost is CPU time rather than PCIe time):
  - For each token and layer, rank missed experts by gate weight. Compute the low-weight tail (cumulative weight > about 0.9) with fewer rows, or skip it.
  - This is lossy, in the same family as STRATA_ROUTE_PRIOR, so it should share the same quality gate (opencode task pass-rate).
  - With top-10 routing over 512 experts, the gate-weight tail is probably longer than Mixtral's top-2, so the skip fraction could exceed 3%. Measure the gate-weight CDF on traces first; this is a free offline check.
- **Mixed precision by hotness** (MxMoE/Fate): pancake's daily GSQ-RCO Q2_0 file could store VRAM-hot experts (per-domain working set) at Q3/Q4 and cold experts at Q2. This is a quality gain at the same VRAM, not a speed gain. Promoting a cold expert would then need a second (higher-bit) copy or would stay at Q2, the D²MoE nested-bit idea. It is only worth doing if Q2 quality is a known pain point.
- **T-MAC for the CPU miss path:** a 2-bit LUT kernel on AVX2 could speed up CPU-computed experts if that path is compute- or dequant-bound rather than DDR4-bandwidth-bound. Micro-benchmark first by comparing pancake's current Q2_0 AVX2 kernel throughput (GB/s of weights) against DDR4 peak (about 40–50 GB/s dual-channel; estimate, not sourced). If already at 70–80% of bandwidth, T-MAC cannot help much. Its 3.6× average is against 2024 llama.cpp kernels.
- **FloE:** its contextual-sparsity predictor plus compression is the closest published analog of "sparse CPU expert compute" for MoE. Its 4–8% quality loss is a warning that combining 2-bit with sparsity compounds error.
- **NVMe:** PowerInfer-2's pipelined, cluster-granular reads with a segmented cache, and SmallThinker's router-ahead SSD prefetch, are the reference designs if the NVMe tier ever becomes critical-path. No action is suggested until tracing shows NVMe reads during decode.

### Gaps
- MxMoE and FloE affiliations were not confirmed from primary pages.
- EdgeMoE (BUPT, expert-wise bit-width adaptation + preloading) and Klotski (Sun Yat-sen, multi-batch expert pipeline) were not verified this round. Klotski is batch-oriented and probably not relevant.
- No paper measures mixed-precision experts with 2-bit as the *high* tier.

---

## Supplementary: speculative decoding / drafting (mainland non-THU/PKU)

### Cited Findings
- **Token Recycling** (Xianzhen Luo et al.; Harbin Institute of Technology **lead**; ACL 2025; arXiv 2408.08696). Candidate tokens rejected during decoding are stored in an adjacency matrix (<2 MB). A BFS-like pass builds a draft tree that is verified with tree attention, and the matrix is updated online. Training-free. About 2× speedup, about 30% over prior train-free methods (Lookahead, PLD, REST) and about 25% over Medusa. Evaluated on SpecBench/MBPP with Vicuna and Code Llama — [arXiv](https://arxiv.org/abs/2408.08696), [ACL listing](https://hyper.ai/en/papers/acl__2025__2025.acl-long.338).

### Inferences
- Token Recycling could fill the gap when prompt-lookup finds no match (novel code), as a third drafter next to MTP and PLD. Expected gain is small (a few % on acceptance-weighted throughput), because MTP already covers the no-match case and every extra draft token widens the expert union per verification step (see the SP-MoE caveat). Low implementation cost: a vocab×k table.

### Gaps
- DuoDecoding (Fudan, CPU draft + GPU target), SAM-Decoding (suffix automaton) and Draft&Verify self-speculation (Zhejiang) were not verified this round. In pancake the CPU is already busy computing misses, so CPU-side drafting is likely counterproductive (inference).
- KV-cache, agent and compiler papers from these groups were not surveyed. With mostly GDN layers and batch-1 single-user use, KV work has low priority for pancake.
