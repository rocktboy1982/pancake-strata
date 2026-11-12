# Round 2 deep research: decoding and prefill speed (A22, A56, A32, A34)

Date: 2026-11-12. Read-only pass: no code edits, no builds, no GPU. Round 1 notes: `unported-academic/notes/decoding_measurement_papers.md` and `agent_thinking_papers.md`.

Read levels used below:
- **full text read**: body text fetched and read (in this pass, via WebFetch, which returns a model-written summary of the page, not verbatim text; numbers should be spot-checked against the PDF before quoting in a paper).
- **full text read (round 1)**: read in round 1 with the same caveat.
- **abstract only**: arXiv abstract page or abstract text only.
- **snippet**: search snippet or secondary aggregator only.
- **unverified**: claim I could not trace to a primary page.

Tool budget: about 17 calls (4 searches, 9 fetches, plus setup and this write). The skill guidance (about 15) was exceeded by about two calls to finish the four questions.

Two corrections to round 1:
- MoE-Lightning (2411.11217) is now read at full-text level via summary. The "up to 10.3x" headline is confirmed as "without padding" on one T4; "3.5x with padding" is the other headline.
- Round 1 listed the OSDI '26 stream-loading prefill paper as "unverified". It is now confirmed as an accepted OSDI '26 paper (arXiv 2606.10493), and its Section 4.2 text is read at summary level.

---

## Q1. A22: entropy-based or top-k stop rule for speculative decoding (beyond DISCO)

### Takeaway
Two papers besides DISCO fit a drafter-entropy stop rule: AdaEDL (2410.18351, full text read via summary) and SVIP (2411.18462, abstract only). AdaEDL measured against static draft lengths (3, 7, 16) and against a top-1-probability stop, on a Llama2-7B target at temperature 0.7 on one A100. On the best static length per dataset, AdaEDL is about 4 to 10 percent faster, and the largest gains are against long fixed lengths. The minimal exposure is the top-1 draft probability (already returned by Strata's MTP selection kernel). Full entropy needs one extra pass over the vocabulary per draft step; no paper measures that cost on a GPU.

### Cited Findings
- AdaEDL stop rule: drafting stops when 1 - sqrt(gamma * H_DM) < lambda, where H_DM is the draft model's entropy at that step, gamma = 0.2 in all experiments, and lambda is updated online toward a target acceptance of alpha = 0.9 (beta1 0.5, beta2 0.9, step 0.01). The paper argues this approximates a lower bound on token acceptance probability, under a "stable entropy" assumption that is approximate. [AdaEDL, https://arxiv.org/html/2410.18351v1, full text read via summary; abstract https://arxiv.org/pdf/2410.18351]
- AdaEDL entropy input: full-vocabulary entropy, cost O(N) in vocabulary size N. No top-k approximation is discussed. Overhead is a theoretical bound only (Appendix B.1); no measured overhead is reported. Experiments run on one A100 80GB in FP32. [AdaEDL, full text read via summary]
- AdaEDL baselines: autoregressive; Base-SPD (static draft length 3, 7, 16); Max-Confidence-SPD (stop when top-1 probability is below a threshold, with the same dynamic threshold update). [AdaEDL, full text read via summary]
- AdaEDL main result (Table 1, Llama2-7B target, Llama2-Drafter-115M draft, temperature 0.7, tokens/s on CNN-DM / Dolly-15k / WMT-19):
  - Autoregressive: 25.74 / 29.02 / 29.80.
  - Max DL 16: Base-SPD 36.30 / 32.10 / 22.30; Max-Confidence 49.50 / 55.80 / 43.70; AdaEDL 54.10 / 56.10 / 43.90.
  - Max DL 7: Base-SPD 51.50 / 47.60 / 32.70; Max-Confidence 53.50 / 56.60 / 45.10; AdaEDL 56.90 / 57.10 / 45.20.
  - Max DL 3: Base-SPD 54.10 / 54.70 / 40.90; Max-Confidence 53.10 / 55.70 / 42.50; AdaEDL 55.70 / 55.80 / 45.00.
  - Derived (my arithmetic on the table): AdaEDL vs the best static length per dataset is 56.90 / 54.10 = +5.2% (CNN-DM), 57.10 / 54.70 = +4.4% (Dolly-15k), 45.20 / 40.90 = +10.5% (WMT-19). [AdaEDL, full text read via summary]
- AdaEDL, other pairs (summary): Pythia-6.9B target with Pythia-1B draft on CNN-DM: AdaEDL 46.4 tok/s vs 29.7 autoregressive (+56%); best Base-SPD 32.2 (+8%). TinyLlama-1B draft with Llama2-7B target on CNN-DM: AdaEDL and Max-Confidence about +43% over autoregressive; Base-SPD 16% slower. [AdaEDL, full text read via summary]
- AdaEDL temperature: tested at 0.7, 1.0, 1.3, 1.7 on CNN-DM. At 1.7 Base-SPD (length 3) falls 57% and is below autoregressive, while AdaEDL is about 8% above autoregressive. Dolly and WMT temperature results are charts only. [AdaEDL, full text read via summary]
- AdaEDL abstract claim, 10% to 57% over static draft length: this is the authors' headline and was not re-derived beyond the table above. [AdaEDL, https://arxiv.org/pdf/2410.18351]
- SVIP (Draft Model Knows When to Stop, EMNLP 2025): sets draft length from draft entropy via a lower bound on acceptance; training-free. Reported speedups: up to 17% on MT-Bench at 8K context vs fixed draft length, and 22% for QwQ long-form reasoning (EMNLP listing); a v2 preprint also reports +13% with EAGLE-2 (snippet only). The abstract compares against fixed-length policies; it does not name a static-lookahead baseline and gives no stop threshold. [SVIP, https://arxiv.org/abs/2411.18462, abstract only; speedup numbers from search snippet]
- DISCO (round 1, full text read): 2-layer FFN on draft top-k (k=10) probabilities, entropy and position; stop when below threshold; reported 10.3% over the best static lookahead; one A100 80GB; Starcoder-15B and Vicuna-13B. [DISCO, https://arxiv.org/abs/2405.04304, full text read (round 1)]
- Strata-side fact (round 1 notes): the MTP selection kernel returns only the top-1 probability, so full entropy is not available cheaply today. [repo notes, round 1]

### Inferences
- Minimal top-k exposure: AdaEDL needs full entropy, which needs the whole softmax (or a second pass over the logits with log-probabilities). Max-Confidence needs only the top-1 probability, which the MTP kernel already returns. DISCO needs top-10 probabilities plus entropy. So the cheapest test is the top-1 rule (already exposed); the entropy rule needs one vocabulary-wide reduction per draft step. Cost of that reduction on this card is not measured anywhere in the papers. Inference: for a vocabulary of about 150k, one extra reduction per draft step is small next to one expert GEMV, but this is unverified on the engine.
- The AdaEDL advantage over Max-Confidence in the table is small (about 0 to 4 points on the best setting), and it is largest against static long draft lengths. Strata's own ceiling (round 1: at most about 8% on edits, about 1% on short turns) suggests a stop rule can only buy a few percent.
- DISCO's exactness argument still applies: a stop rule only changes when to verify. AdaEDL and SVIP have the same property under greedy verification. Their own tables use temperature sampling (AdaEDL), so the exactness claim for sampling is the standard rejection-sampling argument, not a measured check.
- Note that AdaEDL's baselines are a static chain and a top-1 threshold, which are the two rules Strata would test first. A comparison that adds entropy only helps if it beats Max-Confidence on the same trace.

### Settling measurement
On the MTP drafter, record for each draft position the top-1 probability and the full entropy (log both in the window trace, one GPU run in the night queue). Replay offline on the same accepted sequences: static chain, top-1 threshold, AdaEDL-style entropy threshold, and DISCO-style rule where top-10 is available. Compare tokens per target call and estimated tokens/s. No new decode path is needed; the replay is CPU-only once the trace exists.

### Gaps
- SVIP full text not read; its lower-bound derivation and per-dataset numbers are unverified.
- AdaEDL is read via a summarizing fetch, not verbatim. Table numbers should be checked against the PDF before quoting.
- No paper measures entropy overhead on RDNA or any GPU at vocabulary size 150k.
- No paper here uses an MoE target; AdaEDL and DISCO use dense targets (Llama2, Starcoder, Vicuna).

---

## Q2. A56: expected acceptance rate (EAR) definition and computability from greedy traces

### Takeaway
In Leviathan et al. (round 1, full text read), alpha is the expected per-token acceptance probability, and under the acceptance rule min(1, p/q) the expected number of tokens per target call is (1 - alpha^(gamma+1)) / (1 - alpha). Per-position EAR is reported as a conditional rate in PosS (2506.03566, full text read via summary), which is a different convention from the unconditional per-position rate. EAR as Leviathan defines it is not what greedy verification measures: greedy verification measures top-token agreement (the draft token equals the target argmax). The Leviathan quantity needs the target probability of each drafted token and the drafter probability of that token, not the full distributions, but it does not reduce to argmax agreement.

### Cited Findings
- Leviathan et al., acceptance rule: a draft token x ~ q is kept with probability min(1, p(x)/q(x)); on rejection a replacement is drawn from an adjusted distribution. alpha = E(beta); under i.i.d. per-token acceptance, expected tokens per target call = (1 - alpha^(gamma+1)) / (1 - alpha). [Leviathan, https://arxiv.org/html/2211.17192v2, full text read (round 1)]
- SDSL (round 1, partial full text): beta is the acceptance probability of a draft token given a prefix, and alpha is its average over prefixes. [https://arxiv.org/html/2603.11053v1, full text read in part (round 1)]
- PosS (2506.03566): pos-acc at position i is defined as the probability that position i is accepted given that position i-1 was accepted (a conditional rate). The paper does not say whether pos-acc is measured under greedy or sampling verification; its Tables 1 and 2 use temperature 0 and 1, but Figure 4 does not state the temperature. Per-position numbers for EAGLE-2 (Llama-3-8B-Instruct, MT-Bench, depth 8): below 65 percent from the 5th position on. HASS keeps adequate pos-acc through about 4 positions. No MTP per-position numbers are given. [PosS, https://arxiv.org/html/2506.03566v1, full text read via summary]
- AMD ATOM docs (snippet): converting a mean acceptance length into unconditional per-position rates via a helper (acceptance_length_to_rates). Shows that conditional and unconditional per-position rates are both used in practice. [https://rocm.docs.amd.com/projects/atom/en/latest/_sources/forced_acceptance_length.md, snippet only]
- vLLM logs a per-position acceptance vector and exports per-position accepted counts (Prometheus metric names given in a search snippet). [https://github.com/vllm-project/vllm/pull/15151, snippet only]
- Naming: "acceptance rate", "acceptance length" (tau, average accepted tokens per round), and "EAR" are used for different quantities across sources (round 1, NVIDIA AIPerf docs, snippet).

### Inferences
- Under greedy draft and greedy verification, a draft token is accepted iff it equals the target argmax. The per-position indicator is therefore computable from argmax-only traces, and its mean is the top-token agreement rate. This is a metric of the greedy path, not Leviathan's alpha.
- Leviathan's alpha with a greedy drafter: the drafter distribution is a point mass on the drafted token, so the rule reduces to min(1, p(d)/q(d)) with q(d) being the drafter's own probability of its token (= its top-1 probability when greedy). That needs the target probability p(d) at the drafted id, which is a logit read at verification, not argmax-only. So: not computable from argmax-only traces; computable from target p(d) plus drafter q(d), without the full distributions. (Inference from the acceptance rule; no paper checks this exact estimator on a greedy engine.)
- Conditional per-position rates (PosS) and unconditional rates differ by a ratio of adjacent unconditional values. Any table must state which one it reports.
- Metric only: adding these logs does not change output.

### Settling measurement
At each MTP draft position, dump argmax-match, target p(d) and drafter q(d) for the drafted id (no full distributions). From one dump, compute (a) greedy agreement per position, (b) the Leviathan estimator mean of min(1, p(d)/q(d)) per position, and (c) the observed tokens per round vs the i.i.d. formula. The GPU run is one logging run (night queue); the metric is CPU.

### Gaps
- No primary source found that gives the EAR name as a single definition; the concept is Leviathan's alpha, and the name EAR appears in later papers (round 1 gap, still open).
- No source checks whether the Leviathan estimator on a greedy drafter matches argmax agreement; the reduction above is my inference.
- PosS did not state the verification mode for pos-acc.

---

## Q3. A32: streaming experts for large prefill; batch size where streaming beats resident

### Takeaway
No primary paper gives a batch size at which streaming beats keeping experts resident. MoE-Lightning (full text read via summary) pipelines paged weights and reports up to 10.3x over offloading baselines on one 16 GB T4, but it says nothing about expert-level skipping, and it says streaming gives no benefit below a first turning point. The OSDI '26 paper (2606.10493, full text read via summary) reports that weight transfer dominates at 4K prompts and switches its prefill policy at about 4K tokens; it places a ring-buffer shrink point near 50K tokens. For a 20 GB card the relevant question is not streaming vs resident but which experts stay resident; the answer depends on routing skew, which no paper measures on RDNA.

### Cited Findings
- MoE-Lightning (2411.11217), Table 4 (tokens/s, summary): Synthetic Reasoning S1 (1x T4 16 GB): FlexGen(c) 16.903, FlexGen 22.691, DeepSpeed 11.832, MoE-Lightning(p) 26.349. S2 (1x L4 24 GB): 20.015, 50.138, 18.589, 105.29. Summarization S1: 2.614, 3.868, 0.965, 4.52; S2: 4.307, 7.14, 1.447, 12.393. [MoE-Lightning, https://arxiv.org/html/2411.11217, full text read via summary]
- MoE-Lightning headline: up to 10.3x over the best baseline without padding, 3.5x with padding, on one T4 (Mixtral 8x7B). [MoE-Lightning, abstract https://arxiv.org/abs/2411.11217, abstract only, and full text read via summary]
- MoE-Lightning granularity: weights chunked into n pages where n = number of micro-batches (Sec 4.1). Each expert in the MoE FFN kernel uses two pages, looked up via a page table (Appendix A.1). Layer-level buffer of 2 x sizeof(W_L) enables prefetch overlap. Loading only routed experts is not described; whether unrouted experts are skipped is not stated. [MoE-Lightning, full text read via summary]
- MoE-Lightning, small-batch case (Sec 3.3): below the first turning point, there is no benefit in swapping data to GPU for computation; a static placement is suggested. Table 5 uses N = 504 and 1116 and gives no crossover. [MoE-Lightning, full text read via summary]
- MoE-Lightning, numerics: the paper does not claim numerical identity with baselines; CPU attention is computed in float32. [MoE-Lightning, full text read via summary]
- OSDI '26, "Achieving Cloud-Grade SLOs for Local Mixture-of-Experts Inference through CPU-GPU Hybrid Design" (Wang Wenxin et al.), accepted, arXiv 2606.10493. Section 4.2: "For short contexts (4K), weight transfer dominates SLP and DSLP latency, thus no dominant advantage over KT AMX." The system uses an AVX FP8 kernel for requests up to 4K tokens and switches to SLP/DSLP for longer prompts (a policy switch, not a measured crossover). Section 3.1: beyond a prompt-length turning point (e.g. 50K) compute becomes the bottleneck, so the ring buffer shrinks to 2. Hardware (Sec 4.1): 2x RTX 5090 (32 GB each, 419 FP8 TFLOPS), 2x EPYC 9355, PCIe 5.0 x16 link, no GB/s figure given for the link. Model class: DeepSeek-V3 and Kimi-K2 (FP8 experts, 58 MoE layers, 256 experts, 3 modules each). Full per-layer ring buffer: 11.3 to 16.9 GB for these models. Quality: within about 1 to 1.2 points of official scores; not bit-exact. [arXiv 2606.10493, https://arxiv.org/html/2606.10493, full text read via summary, first 100K of 122K characters]
- OSDI '26 headline ~1,200 tokens/s and 32K prompts in 30 s on one node; ~1,800 tokens/s distributed on two RTX 5090s (snippet, from the earlier round and search). [https://www.usenix.org/conference/osdi26/presentation/wang-wenxin, snippet]
- Related, not read in full: FreeToken (arXiv 2608.16157, snippet: 8,192-token chunk in about 1.2 s on RTX 5090); MoE-Prefill (arXiv 2605.02960, snippet: datacenter multi-GPU prefill streaming). [snippet only]

### Inferences
- Routing coverage (my arithmetic, uniform-routing assumption, so an upper bound on distinct experts because real routing is skewed): for a 256-expert top-8 layout, the expected number of distinct experts per layer for a ubatch of T tokens is 256 x (1 - (1 - 8/256)^T). T = 8: about 57 experts (22%). T = 64: about 223 (87%). Under this model a 64-token ubatch already touches almost every expert in a layer, so streamed bytes per token fall roughly as 1/T up to about 64 tokens and then approach full-layer bytes per ubatch. Routing skew would move the knee to larger T. Measure, do not assume.
- For a 20 GB card: streaming only applies to the part of the experts that does not stay resident. If the resident set covers a fraction f of expert bytes, streamed bytes per ubatch are about (1 - f) x (bytes of experts touched). The break-even is where the transfer of those bytes over the real PCIe link (not assumed; read the link state on this box) hides under the layer compute at that T. No paper gives this crossover for consumer RDNA.
- The OSDI paper's 4K switch and the 50K turning point are both policy thresholds for a different hardware and model; they are not the crossover for Pancake.
- Exactness: scheduling change only (same kernels, same reduction order) would keep output bit-identical; a different kernel path (for example a float32 CPU attention) changes bits. Verify per path.

### Settling measurement
On this box: (1) read the PCIe link speed and measure host-to-device bandwidth; (2) from a routing dump, count distinct experts per layer for ubatch sizes 64, 256, 1024 (CPU); (3) measure per-layer expert compute at each T (GPU). The crossover is the T where (2) x bytes / bandwidth falls below (3). One GPU run plus a CPU analysis.

### Gaps
- No primary source gives the batch or ubatch size where expert streaming beats resident experts.
- The OSDI full text beyond 100K characters was not read; the Section 4.2 quote is from a summary and should be checked against the PDF.
- No paper on RDNA3 or ROCm.
- The 1,200 tokens/s and 1,800 tokens/s figures are from snippets, not the body.

---

## Q4. A34: kernel parameter autotuning on AMD GPUs; coordinate descent or model-based tuning; numerics

### Takeaway
The AMD autotuning evidence is mostly qualitative. Kernel Tuner's HIP extension (Lurati et al., Euro-Par 2024) reports that tuning matters more on AMD (about 10x) than on Nvidia (about 2x), and that AMD-tuned kernels run well on Nvidia but not the reverse. The coordinate-descent evidence is either a compiler paper with no speedup in its abstract (Ansor "Explore as a Storm", round 1) or an AMD engineering blog (hipBLASLt, snippet only) that describes coordinate descent over parameter groups without a guaranteed optimum. No paper I read quantifies numerical change from tile or launch changes on AMD. Treat any tile or split change as lossy until logits are compared.

### Cited Findings
- Kernel Tuner HIP (Lurati et al., Euro-Par 2024, arXiv 2407.11488): adds HIP support via PyHIP; language must be set by hand; auto-tuning impact on AMD about 10x vs about 2x on Nvidia; tuned AMD kernels perform well on Nvidia, but not the reverse. [https://arxiv.org/abs/2407.11488, snippet and search summary only; full text not read]
- Ansor "Explore as a Storm, Exploit as a Raindrop" (arXiv 2406.20037): exploration capped at the first 300 kernels, then coordinate descent (from AutoTVM Droplet Search) refines the best candidate; evaluated on 20 DL models across AMD Ryzen 7, NVIDIA A100, RTX 3080, ARM A64FX. Abstract gives no speedup number. [https://arxiv.org/abs/2406.20037, abstract only (round 1)]
- AMD ROCm blog on iteratively tuning hipBLASLt TensileLite kernels (snippet): coordinate descent over parameter groups; each round searches one group with the rest pinned at the running best; the authors note this is not guaranteed to find the global optimum because parameters interact. [https://rocm.blogs.amd.com/software-tools-optimization/iteratively-tuning-hipblaslt/README.html, snippet only; the search summary gave no speedup figure]
- IBM (Ringlein et al., 2025), autotuning portable LLM attention kernels across A100 and MI250, up to 15x more configurations explored than prior approaches (snippet; no URL verified). [unverified]
- CLTune (arXiv 1703.06503) and Lim, Norris, Malony (arXiv 1701.08547): round 1, abstract or snippet only.

### Inferences
- Coordinate descent costs roughly the sum of the grid sizes per pass, not the product (round 1 reasoning). The AMD blog's caveat that parameters interact is the main risk, which is the same warning as a single-pass sweep.
- Numerics (inference, not read in any source): a tile or split-K change alters float accumulation order, so output can change in the last bits; atomics in reductions are order-dependent. A bit-exact check before any timing is the only way to call a change exact.
- Overfitting: a sweep tuned at one context or batch size may not transfer (round 1 link to A49).

### Settling measurement
One coordinate-descent pass over a single decode kernel (the GEMV for the expert path, fixed shape) with the bit-exact logit check first, then the timing A B B A with the discard-first-run rule. Needs the GPU; benchmarks go to the night queue.

### Gaps
- No primary paper with coordinate-descent speedups on RDNA3 or on ROCm LLM kernels.
- Kernel Tuner HIP full text and numbers not read.
- No source states whether tile or launch changes alter float results on AMD. The numerics point above is an inference.
- The 15x and 10x numbers are from snippets, not verified tables.

---

## Source list (primary where possible)
- Leviathan et al., https://arxiv.org/html/2211.17192v2 (full text read, round 1)
- DISCO, https://arxiv.org/abs/2405.04304 (full text read, round 1)
- AdaEDL, https://arxiv.org/html/2410.18351v1 (full text read via summary); abstract https://arxiv.org/pdf/2410.18351
- SVIP, https://arxiv.org/abs/2411.18462 (abstract only)
- PosS, https://arxiv.org/html/2506.03566v1 (full text read via summary)
- MoE-Lightning, https://arxiv.org/html/2411.11217 (full text read via summary); abstract https://arxiv.org/abs/2411.11217
- OSDI '26 (arXiv), https://arxiv.org/html/2606.10493 (full text read via summary, partial)
- Kernel Tuner HIP, https://arxiv.org/abs/2407.11488 (snippet)
- Ansor storm, https://arxiv.org/abs/2406.20037 (abstract, round 1)
- hipBLASLt blog, https://rocm.blogs.amd.com/software-tools-optimization/iteratively-tuning-hipblaslt/README.html (snippet)
- ATOM docs, https://rocm.docs.amd.com/projects/atom/en/latest/_sources/forced_acceptance_length.md (snippet)
