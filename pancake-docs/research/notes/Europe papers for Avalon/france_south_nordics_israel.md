# AI papers from France, Southern Europe, the Nordics, Poland, Romania, Greece and Israel (Nov 2022 - Nov 2026) for pancake / Avalon

Scope: university or public-institute work, read against our setup. The model is Qwen3.8-Flash-Next (~125B MoE, Gated DeltaNet + attention hybrid, MTP head), run as a 2-3-bit GGUF on an RX 7900 XT 20 GB with a Ryzen 5700G (AVX2, DDR4) and 57 GB RAM. Experts are tiered across VRAM, RAM and NVMe, and the CPU computes the missed experts (about half of decode time, bound by RAM bandwidth). The workload is a single-user opencode agent.

I checked for repeats against /home/dan/reports/* (Stanford, MIT/Harvard, upstream) and /home/dan/research_notes/US university papers for Avalon/*. None of the arXiv IDs below appear there. Excluded as instructed: ETH/EPFL, ISTA, Germany, Netherlands, Belgium, Austria, Czechia, UK, and company-led labs.

Method note: the web search tool does not filter reliably by affiliation. Queries for Inria/CNRS kernels, Nordic MoE offload, Greek/Romanian work and Polish expert skipping returned mostly papers from other countries. Affiliations below come from the cited listings. Where a listing did not show an affiliation, the entry says so.

## Q1. Which of these groups' work could speed up batch-1 MoE decode with offloaded experts, or long-prompt prefill, on a consumer AMD GPU?

### Takeaway
No paper from these regions attacks our exact bottleneck: batch-1 decode where the CPU computes missed experts over DDR4. The nearest work is on speculative-decoding control (Weizmann/Intel DISCO, DSI) and on activation sparsity inside experts (Warsaw/Sapienza D2DMoE and its 2025 follow-up). Of those, the activation-sparsity results are the only ones that touch the CPU-bound half of decode, and they mostly add evidence for the TEAL-style plan already in the MIT report. KV-compression work (Inria Q-Filters, Hebrew U TOVA) matters little, because the hybrid model has few attention layers.

### Cited Findings

**1. DISCO: Dynamic Speculation Lookahead Accelerates Speculative Decoding of LLMs** (arXiv 2405.04304, v1 May 2024, revised to v5)
- Who: Intel Labs Israel leads; Weizmann Institute and Hebrew University of Jerusalem are co-authors. Company-led with university co-authors, so it counts as borderline. — [arXiv](https://arxiv.org/html/2405.04304v5)
- Idea: a small classifier picks the draft length (speculation lookahead, SL) at every iteration instead of using a fixed SL. — [arXiv v1](https://arxiv.org/html/2405.04304v1)
- Numbers: about 10% average speedup over a tuned static SL, and about 31% over earlier dynamic heuristics. — [arXiv v5](https://arxiv.org/pdf/2405.04304v5)
- Verdict for us: medium-low. It is the published basis for an adaptive MTP depth, which upstream is already trying (#27210 is on our "watch" list). Deeper drafts touch more experts under offload, so the classifier would need expert-miss count as a feature. Expected gain is at most ~5-10% on spec-decode t/s, and only if static depth is currently mistuned.

**2. DSI: Distributed Speculative Inference** (ICLR 2025; arXiv 2405.14105)
- Who: Nadav Timor (Weizmann) is lead and corresponding author; Intel Labs and others are co-authors. — [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b36554b97da741b1c48c9de05c73993e-Abstract-Conference.html)
- Idea: "speculation parallelism" runs the target and drafter instances concurrently and overlaps verification with drafting. The authors prove it is faster than both standard speculative inference and plain decoding for any drafter. — [arXiv](https://arxiv.org/html/2405.14105v5)
- Numbers: 1.29-1.92x over standard speculative inference, single-node, simulated. Code is open source. — [arXiv](https://arxiv.org/html/2405.14105v5)
- Verdict for us: low. DSI needs several target-model instances working in parallel. We have one GPU that cannot even hold one copy of the experts. The idea that does carry over is overlapping draft generation with the CPU expert computation of the verify pass, but that is our own inference, not a result from the paper.

**3. Accelerating LLM Inference with Lossless Speculative Decoding Algorithms for Heterogeneous Vocabularies** (ICML 2025 oral; arXiv 2502.05202)
- Who: Timor et al. (Weizmann + Intel Labs). — [ICML](https://icml.cc/virtual/2025/oral/47176); [mlanthology](https://mlanthology.org/icml/2025/timor2025icml-accelerating)
- Idea: three lossless methods let drafter and target use different vocabularies. One is SLEM (string-level exact matching, with text as the shared representation). Hugging Face Transformers made SLEM and TLI its defaults for heterogeneous speculative decoding. — [arXiv](https://arxiv.org/abs/2502.05202v3)
- Numbers: up to 2.8x over autoregressive decoding on summarization, code and long-context tasks. — [Intel newsroom](https://www.intel.com/content/www/us/en/newsroom/news/artificial-intelligence/intel-weizmann-institute-speed-ai-with-speculative-decoding-advance.html)
- Verdict for us: low. We already draft with MTP and prompt lookup, and no separate drafter is needed. A string-level matching idea could, in principle, make prompt-lookup drafting robust to retokenization boundaries in edited code (inference only).

**4. Accelerating Transformer Inference for Translation via Parallel Decoding** (ACL 2023; arXiv 2305.10427)
- Who: Santilli, Severino, ..., Rodolà (Sapienza, GLADIA lab). Sapienza leads. — [GLADIA](https://gladia.di.uniroma1.it/publication/ipi)
- Idea: recasts greedy decoding as Jacobi / Gauss-Seidel fixed-point iterations, training-free. — [HF papers](https://huggingface.co/papers/2305.10427)
- Numbers: up to 38% faster, and about 2x with more parallel hardware. — [ACL virtual](https://virtual2023.aclweb.org/paper_P505.html)
- Verdict for us: low. Jacobi verification of a block touches the union of experts across all positions, which hurts under offload. Prompt lookup plus MTP already has a higher acceptance rate for code edits.

**5. Exploiting Activation Sparsity with Dense to Dynamic-k MoE Conversion (D2DMoE)** (NeurIPS 2024; arXiv 2310.04361)
- Who: Szatkowski, Wójcik, Piórczyński, Scardapane. Polish lead (Warsaw / IDEAS NCBR per the author list; the snippet did not confirm affiliations); Scardapane (Sapienza) is co-author. — [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4c2092ec0b1370cce3fb5965ab255fae-Abstract.html)
- Idea: a Hoyer penalty raises activation sparsity, the MLP is split into experts, and a dynamic-k router chooses how many experts each token runs. — [arXiv](https://arxiv.org/html/2310.04361v4)
- Numbers: up to 60% lower inference cost, and recovery needs under 1% of pretraining tokens (1B tokens for GPT2-base / Gemma-2B). — [arXiv](https://arxiv.org/pdf/2310.04361)
- Verdict for us: low as written, because it needs fine-tuning and our model is already MoE. Its dynamic-k principle (vary active-expert count per token) is the same lever as skipping low-gate-weight experts on CPU misses.

**6. Universal Properties of Activation Sparsity in Modern LLMs** (ICLR 2026; arXiv 2509.00454, Aug 2025)
- Who: Szatkowski + 7 co-authors (same Polish/Sapienza group). — [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ec3c79dc0c2b85532cfd1012a4aaa923-Abstract-Conference.html); [arXiv](https://arxiv.org/abs/2509.00454v1)
- Idea: a general framework for how robust non-ReLU LLM activations (FFN) are to sparsification. It finds universal patterns across families and reports that the potential for sparsity grows with model size. It also includes the first study of activation sparsity in diffusion LLMs. — [ICLR virtual](https://iclr.cc/virtual/2026/poster/10010332)
- Verdict for us: medium as evidence, low as new code. It backs the TEAL-style magnitude sparsification of activations into the down-projection of experts computed on the CPU, which the MIT report already proposed. On the CPU path, fewer weight rows read means proportionally less DDR4 traffic. Expected gain: if 40-50% of down-proj input activations can be zeroed at acceptable quality, CPU expert time falls by roughly 15-25%, because gate and up still read all rows. That is roughly 7-12% end-to-end decode. This is an inference from the paper's claims plus our profile, not a measured result. Check the paper's per-family thresholds for Qwen-style SwiGLU experts before implementing.

**7. Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression** (arXiv 2503.02812, Mar 2025; ICLR 2025 workshop)
- Who: Nathan Godey (lead), Devoto, Zhao, Scardapane, Minervini, de la Clergerie, Sagot. De la Clergerie and Sagot are Inria ALMAnaCH; the listing I saw did not show affiliations. Co-authors are from Sapienza and Edinburgh. — [arXiv](https://arxiv.org/pdf/2503.02812); [mlanthology](https://mlanthology.org/iclrw/2025/godey2025iclrw-qfilters)
- Idea: training-free KV eviction that projects keys onto one SVD-derived query direction per head, computed once offline. It needs no attention scores, so it works with FlashAttention; for GQA the filters are averaged within each group. — [arXiv](https://arxiv.org/html/2503.02812v1)
- Numbers: 99% needle-in-a-haystack at 32x compression, and up to 65% less perplexity degradation than StreamingLLM. — [HF papers](https://huggingface.co/papers/2503.02812)
- Code: only a third-party mirror was found (github.com/standardgalactic/qfilters).
- Verdict for us: low. Only the full-attention layers of the hybrid have a KV cache, and TurboQuant KV already gave no gain in pancake. Consider it only for very long (>128K) sessions where KV memory competes with VRAM expert slots. FA-compatible scoring is its advantage over TOVA/H2O.

**8. Transformers are Multi-State RNNs / TOVA** (EMNLP 2024; arXiv 2401.06104)
- Who: Oren, Hassid, Adi, Schwartz. Hebrew University leads; Meta FAIR is co-author. — [arXiv](https://www.arxiv.org/pdf/2401.06104)
- Idea: bounded-size KV, evicting by attention weight.
- Numbers: near full-cache quality at 1/8 cache in some settings, and up to 4.8x throughput. — [Meta](https://ai.meta.com/research/publications/transformers-are-multi-state-rnns/)
- Code: [schwartz-lab-NLP/TOVA](https://github.com/schwartz-lab-NLP/TOVA).
- Verdict for us: low, for the same reason as Q-Filters. It also needs attention weights, which fused FA kernels do not expose.

**9. DeciMamba: Exploring the Length Extrapolation Potential of Mamba** (ICLR 2025; arXiv 2406.14528)
- Who: Ben-Kish, Zimerman, Abu-Hussein, Cohen, Globerson, Wolf, Giryes. Tel Aviv University leads; Google Research is co-author. — [arXiv](https://arxiv.org/pdf/2406.14528); [mlanthology](https://mlanthology.org/iclr/2025/benkish2025iclr-decimamba)
- Idea: data-dependent token decimation before the S6 layer, training-free, so the model extrapolates to 25x the training length with faster inference. — [arXiv v1](https://arxiv.org/html/2406.14528v1)
- Code: assafbk/DeciMamba.
- Verdict for us: low. Qwen's GDN layers already handle the context lengths we use. Token dropping in recurrent layers during prefill would be an approximate-prefill experiment with quality risk for code.

**10. Priority-Aware Preemptive Scheduling for Mixed-Priority Workloads in MoE Inference** (arXiv 2503.09304)
- Who: KTH. — [arXiv](https://arxiv.org/html/2503.09304v1)
- Verdict for us: low. It targets multi-request serving. The only use would be letting opencode's title or summary side-requests preempt or yield to the main agent turn, which the upstream report already handles via the #30188-style cache fixes.

**11. Mind the Memory Gap: Unveiling GPU Bottlenecks in Large-Batch LLM Inference**
- Who: BSC + UPC + IBM Research. — referenced via [semiengineering](https://semiengineering.com/tag/dram-bandwidth)
- Verdict for us: not applicable. It studies large-batch GPU serving.

**12. Efficient Server Consolidation through a balanced mix of Transformer-based and Conventional Applications** (ICS '25, June 2025, DOI 10.1145/3721145.3725751)
- Who: Abad, Prieto, Puente, Gregorio (Universidad de Cantabria, Spain).
- Idea: profiles llama.cpp and neuralspeed on general-purpose CPUs and focuses on the microarchitecture backend. It reports low contention when co-located with SPEC CPU17. — [ICS paper PDF](https://www.ce.unican.es/pdf/LLM-ICS-2025.pdf)
- Numbers: one function takes 98% of time for fp32 and nearly 90% for quantized models. — [summary via search of the same PDF](https://www.ce.unican.es/pdf/LLM-ICS-2025.pdf)
- Verdict for us: low. It confirms that the CPU expert path is one hot kernel, with no new technique.

### Inferences
- The only regional line with a plausible decode gain is activation sparsity on CPU-computed experts (#6 as evidence, #5 as the dynamic-k concept). Treat it as validation of the existing TEAL item, not a new task. A realistic expected gain is high single digits to ~12% end-to-end, provided quality holds at 2-3-bit.
- Adaptive MTP depth (DISCO) is a cheap policy experiment worth a few percent at most. Fold it into the upstream #27210 test with expert-miss count as a feature.
- Nothing found here helps prefill on RDNA3 beyond what upstream llama.cpp kernels already provide.

### Gaps
- No Inria/CNRS, Polimi/PoliTo, BSC, Chalmers/Aalto/DTU/NTNU/Copenhagen, Greek or Romanian paper on MoE expert offload, expert prefetching, learned caching, NVMe expert I/O or GPU kernel autotuning surfaced. The searches returned other countries' work (MoBiLE, FloE, DAOP, Fiddler, MoE-Infinity from Edinburgh, which is excluded). This may reflect search-engine affiliation blindness, not absence. A direct HAL (hal.inria.fr) and DBLP affiliation crawl would close it.
- D2DMoE author affiliations (Warsaw University of Technology vs IDEAS NCBR vs Jagiellonian) were not confirmed in the snippets.
- Q-Filters' official code repository was not confirmed.
- I did not find the Cantabria paper's proposed speedups, if it has any; it is mainly a consolidation study.

## Q2. Quantization, sparsity, compression or CPU SIMD work usable with GGUF?

### Takeaway
I found no regionally led, GGUF-compatible weight-quantization or CPU-SIMD method that beats the IQ2/IQ3 formats we use. The relevant regional output is sparsity analysis (Poland/Sapienza) and sparse attention that requires training (Lisbon), plus MoE scaling-law theory (Warsaw). None of it can be dropped into GGUF kernels without retraining, except the activation-sparsity thresholds discussed in Q1.

### Cited Findings
- **Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient** (ICML 2025; arXiv 2502.05172). University of Warsaw / IDEAS NCBR lead. It is a joint dense/MoE scaling law in active parameters, data and expert count, and it shows MoE can be more memory-efficient than dense at equal loss and compute. — [alphaxiv](https://alphaxiv.org/abs/2502.05172). Earlier from the same group: **Scaling Laws for Fine-Grained MoE** (arXiv 2402.07871, Krajewski, Ludziejewski et al.). — [HF papers](https://huggingface.co/papers/2402.07871.md). Verdict: no runtime use; they are training-design results.
- **AdaSplash: Adaptive Sparse Flash Attention** (ICML 2025 oral/spotlight; arXiv 2502.12082). Gonçalves, Treviso, Martins (Lisbon: IST / Instituto de Telecomunicações; affiliation not shown in the snippet). It speeds up α-entmax attention: a hybrid Halley-bisection solver needs 7x fewer iterations, and Triton kernels exploit the resulting sparsity, often matching or beating FlashAttention-2. — [PMLR](https://proceedings.mlr.press/v267/goncalves25a.html); [ICML](https://icml.cc/virtual/2025/poster/45440). Code: github.com/deep-spin/adasplash. Verdict: not applicable. It needs models trained with entmax, and the kernels are Triton/CUDA.
- **Long-Context Generalization with Sparse Attention (ASEntmax)** (ICLR 2026; arXiv 2506.16640). Vasylenko, Pitorro, Martins, Treviso. α-entmax with a learnable temperature avoids attention dispersion and greatly beats softmax baselines on length generalization. — [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/19c9708f31ec44b5b1cbd67f91d05d95-Abstract-Conference.html); [arXiv](https://arxiv.org/abs/2506.16640). Verdict: not applicable, since it requires training.
- **Universal Properties of Activation Sparsity** (Warsaw/Sapienza, ICLR 2026) and **D2DMoE** (NeurIPS 2024): see Q1 #5 and #6. These are the only GGUF-usable items, as runtime activation thresholding inside the CPU and GPU expert matvec. — [arXiv 2509.00454](https://arxiv.org/abs/2509.00454v1); [arXiv 2310.04361](https://arxiv.org/pdf/2310.04361)
- Technion low-bit PTQ: the targeted search returned only Intel (SignRoundV2, arXiv 2512.04746), ISTA (GSQ, excluded) and Qualcomm work. — [search results incl. SignRoundV2](https://huggingface.co/papers/2512.04746)

### Inferences
- For GGUF at 2-3 bits, these regions add no new quantizer. The MIT/Harvard report's conclusion stands: IQ2_XS-class formats are already competitive.
- Activation sparsity is orthogonal to weight quantization and compounds with it on the bandwidth-bound CPU path. It is the only lever here worth a prototype: a threshold per layer, applied before the down-proj in the AVX2 expert kernel.

### Gaps
- I could not find Technion/Soudry-group LLM PTQ papers from 2025-2026 via search. Soudry's FP8-training work is training-side and out of scope. Kernels from Greece, Romania or Poland for CPU SIMD (AVX2) inference did not surface.
- No regional work on lossless weight compression (Huffman/entropy coding of expert weights to cut NVMe/RAM traffic) was found.

## Q3. Agent, reasoning and evaluation work for opencode

### Takeaway
Israeli groups lead here. Hebrew U/Meta show that shorter thinking chains are more accurate per question, which supports tighter thinking budgets. Technion shows that answer-token hidden states encode errors, which supports cheap probes. TAU's AssistantBench is an agent benchmark, but for web tasks, not coding.

### Cited Findings
- **Don't Overthink it: Preferring Shorter Thinking Chains for Improved LLM Reasoning** (arXiv 2505.17813, May 2025, revised Feb 2026). Hassid, Synnaeve, Adi, Schwartz; Hebrew University with Meta FAIR, and Meta's role may be leading. — [arXiv](https://arxiv.org/abs/2505.17813v2)
  - Within a question, the shortest sampled chain is up to 34.5% more accurate than the longest. — [arXiv html](https://arxiv.org/html/2505.17813v2)
  - short-m@k (stop at the first m finished chains, then majority vote) matches or beats majority voting with up to 33% less wall time. Fine-tuning on short chains also helped. The authors caution that harder questions still benefit from length. — [HF papers](https://huggingface.co/papers/2505.17813)
  - Verdict for us: medium (policy, no engine work). It supports lower default thinking budgets and early-stop-on-answer heuristics for routine opencode turns. short-m@k itself needs k parallel chains, which costs a lot on batch-1 offloaded MoE because the expert union grows. Skip it, except possibly k=2 on hard turns.
- **LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations** (ICLR 2025; arXiv 2410.02707). Orgad, ..., Reichart, Belinkov; Technion leads, with Google Research and Apple co-authors. — [arXiv](https://arxiv.org/html/2410.02707v4); [mlanthology](https://mlanthology.org/iclr/2025/orgad2025iclr-llms)
  - Truthfulness signal concentrates in the exact answer tokens, where probes detect errors better. The probes do not generalize across datasets, and models can encode the correct answer while generating a wrong one.
  - Code: github.com/technion-cs-nlp/LLMsKnow.
  - Verdict for us: low-medium. It complements the MIT report's hidden-state probes for the thinking cap. A task-specific probe on tool-call argument tokens could flag likely-bad tool calls before execution and trigger the existing tool-call recovery. Given the poor cross-dataset generalization, the probe would have to be trained on our own opencode logs.
- **AssistantBench: Can Web Agents Solve Realistic and Time-Consuming Tasks?** (EMNLP 2024; arXiv 2407.15711). Yoran, Amouyal, Malaviya, Bogin, Press, Berant; Tel Aviv University leads. — [ACL](https://aclanthology.org/2024.emnlp-main.505); [arXiv](https://arxiv.org/abs/2407.15711)
  - 214 tasks requiring browsing more than 525 pages. The best model scores about 25%. It also proposes the SPA agent (planning + memory). — [project](https://assistantbench.github.io)
  - Verdict for us: low. It covers web research, not coding. Only useful if opencode's web-fetch tool use is evaluated.

### Inferences
- The actionable change is a thinking-budget policy, backed by Hebrew U's results plus the earlier Stanford and MIT findings: shorter default budgets for routine edits, and escalation only on failure or retry.
- Probing for tool-call errors is a research-grade item. It is worth it only if tool-call failures remain frequent after the existing recovery and loop guard.

### Gaps
- No regional coding-agent benchmark or repository-level coding evaluation (a SWE-bench analogue) from these institutions was found.
- No regional work on agent context compaction or tool-call grammar enforcement surfaced.
- Lisbon's evaluation work (xCOMET etc.) is for machine-translation metrics and was not pursued.
