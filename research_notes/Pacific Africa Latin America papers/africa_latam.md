# Africa and Latin America university papers (Nov 2022 - Nov 2026) for pancake / Avalon

Scope: African universities (UCT, Wits, Stellenbosch, Pretoria, Makerere, Nairobi, KNUST, AIMS, Cairo, UM6P, plus Masakhane / Indaba-linked work) and Latin American universities (USP, Unicamp, UFMG, UFRJ, PUC-Rio, UBA, CONICET, U. de Chile, PUC Chile, CENIA, UNAM, Tec de Monterrey, CINVESTAV, Uniandes). Company-led work is excluded (InstaDeep, Lelapa AI's InkubaLM, Maritaca's Sabiá, Microsoft-era SkipDecode).

Method:
1. OpenAlex works search filtered by institution ID and date (>= 2022-11-01) over about 20 topic queries (quantization, MoE inference, speculative decoding, KV cache, LLM serving, agents/tools, tensor compilers, pruning, reasoning efficiency, tokenization cost, early exit, edge deployment). The Latin American second pass (serving, edge, early exit, structured output, test-time compute) hit the OpenAlex daily rate limit (HTTP 429, about 15 h reset), so those queries ran for Africa only.
2. arXiv API listings by key faculty: Quintão Pereira (UFMG), Guido Araujo and Sandro Rigo (Unicamp), Rodrigo Nogueira (Unicamp/Maritaca), Artur Jordão (USP), Luciano Del Corro (UBA), Jan Buys (UCT), Benjamin Rosman (Wits), Herman Kamper (Stellenbosch), Bruce Bassett (UCT/AIMS), Vukosi Marivate (Pretoria) and Pablo Barceló (PUC Chile). After that came abstract retrieval for the shortlist, plus targeted web searches for Mexico, Chile, Colombia, PUC-Rio and UM6P.
3. Repeat check: none of the arXiv IDs below appears in `/home/dan/reports/*.md` or the other research_notes folders (218 previously cited IDs checked).

Lead/co: "Lead" means the first author or the senior (last) author is at the named institution, per OpenAlex affiliation metadata or arXiv author lists. I did not re-read the PDF affiliation blocks. Verdicts and expected gains are my inferences. Unless stated otherwise, all numbers are the authors' own, on their hardware (none used AMD/ROCm, PCIe 3.0, a 512-expert MoE or a GDN hybrid).

Headline: **Nothing from either region targets the core bottleneck: CPU-computed missed experts, expert tiering and MTP depth.** No paper found addresses MoE offload, speculative decoding, KV/expert caching or quantized-kernel work at batch 1. What does transfer is small and peripheral: (a) calibration-data guidance for requantizing (AIMS/UCT), (b) cheap coordinate-descent autotuning of kernel launch parameters (UFMG), (c) agent-side token and reliability tooling for opencode (KNUST, arXiv-only, author-run benchmarks), and (d) entropy and hidden-state probes as cheap quality/loop signals (UBA). Latin America has slightly more systems/compiler content (UFMG, Unicamp) than Africa. Africa's LLM output is dominated by low-resource-language data, benchmarks and models, which do not apply here.

---

## Q1. Which African university papers fit pancake / Avalon, and what would they gain us?

### Takeaway
Only one African paper touches the inference stack directly: multilingual/non-English calibration data improves GPTQ/AWQ quantization (AIMS/UCT, EACL 2026). It matters only if Qwen3.8-Flash-Next is requantized, and it brings zero decode-speed gain. A KNUST group (Agyemang, Kponyo et al.) published a burst of 2026 arXiv tools for coding agents: token reduction, MCP tool discovery, durable writes and an agent scheduler. These map to the opencode side (prefill tokens, tool-call recovery), not to decode t/s, and their benchmarks are author-constructed and not peer-reviewed.

### Cited Findings

**1. Calibrating Beyond English: Language Diversity for Better Quantized Multilingual LLMs.** Everlyn Asiko Chimoto, Mostafa Elhoushi, Bruce A. Bassett. Lead: AIMS/UCT (Chimoto first author, Bassett senior; OpenAlex matched the paper to the African institution set; Elhoushi is a co-author outside the region). EACL 2026 main conference; arXiv Jan 2026. [arXiv 2601.18306](https://arxiv.org/abs/2601.18306)
- Core idea: compare 8 calibration settings (5 single-language, 3 multilingual mixes) for GPTQ and AWQ on 10 languages. — [arXiv 2601.18306](https://arxiv.org/abs/2601.18306)
- Key numbers: non-English and multilingual calibration consistently beats English-only. Multilingual mixes cut perplexity by up to 3.52 points on Llama-3.1-8B and Qwen2.5-7B. Matching the calibration language to the evaluation language gives the largest per-language gains. Some language-quantizer pairs fail, which the authors trace to different activation-range distributions across languages. — [arXiv 2601.18306](https://arxiv.org/abs/2601.18306)
- Code: none stated in the abstract.
- Verdict: **applies only when requantizing** (GSQ-RCO Q2_0 or an Avalon GGUF imatrix). The domain-alignment finding supports building the calibration/imatrix set from the actual workload: opencode traces with code, tool-call JSON, thinking text and Romanian/English chat, rather than wiki text. Expected gain: quality only (lower perplexity/KLD at Q2), with **0 t/s**. Activation-range mismatch is the same failure mode that matters most at 2 bits, so the effect plausibly grows at Q2. This is an inference: the paper tested 4-bit-class GPTQ/AWQ on dense 7-8B models.

**2. Cartograph: Federated Tool Discovery with Operator-Attested Retrieval for AI Agents.** Justice Owusu Agyemang, Michael Agyare, K. O.-B. O. Agyekum, K. A.-P. Agyekum, F. A. Acheampong, Jerry John Kponyo. Lead: KNUST (OpenAlex lists first and last author at KNUST). arXiv Sept 2026, not peer-reviewed. [arXiv 2609.30293](https://arxiv.org/abs/2609.30293)
- Core idea: an MCP proxy that exposes 3 proxy tools instead of the full catalog. It uses two-stage retrieval (servers, then tools) over signed capability cards, turning O(n) tool definitions in context into O(k) progressive disclosure. — [arXiv 2609.30293](https://arxiv.org/abs/2609.30293)
- Key numbers: on 22 servers / 374 tools, a top-5 discovery exchange uses 475 tokens versus 42,450 under full-catalog accounting. R@5 is 0.816 versus 0.592 for a Jaccard baseline, on a 49-query author-built benchmark. The gateway adds 5 ms mean latency (0.8%). — [arXiv 2609.30293](https://arxiv.org/abs/2609.30293)
- Code: not stated in the abstract.
- Verdict: **relevant to prefill cost if opencode loads many MCP tools.** Every tool definition sits in the system prompt, which the conversation cache only helps with after the first turn and after tool-set changes. With a small tool set (rea MCP is disabled per memory), the gain is near zero. With a large catalog, it removes tens of thousands of prefill tokens per new session. Expected gain: session-start prefill only; decode is unchanged.

**3. Resilient Write: A Six-Layer Durable Write Surface for LLM Coding Agents.** Agyemang, Kponyo, Amponsah, Boakye, Agyekum. Lead: KNUST. arXiv Apr 2026 (v3), not peer-reviewed. [arXiv 2604.10842](https://arxiv.org/abs/2604.10842)
- Core idea: an MCP server that sits between the agent and the filesystem, with six layers: risk pre-flight, atomic transactional writes, resume-safe chunking, typed structured errors, an out-of-band scratchpad, and task-continuity handoff. — [arXiv 2604.10842](https://arxiv.org/abs/2604.10842)
- Key numbers: 186-test suite; 5x faster recovery and a 13x higher agent self-correction rate versus naive/defensive baselines (author comparison). MIT license. — [arXiv 2604.10842](https://arxiv.org/abs/2604.10842)
- Verdict: **complements pancake's tool-call recovery** on the agent side. Truncated or failed large file writes are a known waste of decode tokens for a 126 t/s local model, so resumable chunked writes plus typed errors could cut retry turns. Expected gain: fewer wasted turns on large edits, which is unquantified for our setup and has 0 kernel impact. It is cheap to trial as an MCP server in opencode.

**4. Local-Splitter: A Measurement Study of Seven Tactics for Reducing Cloud LLM Token Usage on Coding-Agent Workloads.** Agyemang et al. Lead: KNUST. arXiv Apr 2026. [arXiv 2604.12301](https://arxiv.org/abs/2604.12301)
- Core idea: a small local model triages for a cloud model. The seven tactics are local routing, prompt compression, semantic caching, local draft + cloud review, minimal-diff edits, structured intent extraction, and batching with vendor prompt caching. — [arXiv 2604.12301](https://arxiv.org/abs/2604.12301)
- Key numbers: routing + compression saves 45-79% of cloud tokens on edit- and explanation-heavy workloads. The full set incl. draft-review saves 51% on RAG-heavy workloads. The best subset depends on the workload. — [arXiv 2604.12301](https://arxiv.org/abs/2604.12301)
- Verdict: **mostly not applicable**, since pancake is already the local model. The one transferable tactic is "minimal-diff edits": forcing edit-tool output instead of whole-file rewrites cuts decode tokens directly. opencode's edit tool already does this, so the expected gain is about 0 unless the model falls back to full rewrites.

**5. Entropy Gate: Entropy Quenching for Near-Lossless Token Compression in LLM Pipelines.** Agyemang et al. Lead: KNUST (last-author match). arXiv June 2026. [arXiv 2606.03739](https://arxiv.org/abs/2606.03739)
- Core idea: a heuristic per-token "information energy" (statistical + structural + positional) with an annealed removal schedule and a fidelity gate. It runs as an OpenAI-compatible proxy. — [arXiv 2606.03739](https://arxiv.org/abs/2606.03739)
- Key numbers: 40-60% prompt compression at energy-similarity > 0.80. Context dedup saves 50-70% on repeated blocks. The paper claims 88-96% combined with external memory for agentic workloads. The fidelity metric is the authors' own, not task accuracy. — [arXiv 2606.03739](https://arxiv.org/abs/2606.03739)
- Verdict: **redundant / risky.** Repeated-block dedup overlaps the `feat/contextpilot` branch already in progress (per the Asia report). Lossy prompt-token deletion also breaks exact-prefix conversation-cache hits, which is a net loss for pancake's cache. No adoption.

**6. HiveMind: OS-Inspired Scheduling for Concurrent LLM Agent Workloads.** Agyemang, Kponyo, Somuah, Amponsah, Boakye, Agyekum. KNUST authors (OpenAlex affiliation blank on this record; same group). arXiv Apr 2026. [arXiv 2604.17111](https://arxiv.org/abs/2604.17111)
- Key numbers: uncoordinated agents fail 72-100% under contention (5-50 agents). The proxy cuts failures to 0-18% and removes 48-100% of wasted compute, with < 3 ms overhead against Ollama. Transparent retry is the most important primitive. — [arXiv 2604.17111](https://arxiv.org/abs/2604.17111)
- Verdict: **low.** It helps only if opencode fires parallel subagents at one batch-1 pancake server. There, admission control (queue rather than reject or time out) plus transparent retry would avoid dropped subagent requests. Not a speed item.

**7. ACEM: A Cost Estimation Model for Agentic Software Engineering.** Mohammad El-Ramly. Lead: Cairo University. arXiv Aug 2026 (v2), "under journal publication". [arXiv 2608.02582](https://arxiv.org/abs/2608.02582)
- Core idea: total cost = LLM + human-in-the-loop + infrastructure, with a Revision Factor (retry token overhead) and a Context Factor (token growth as context accumulates). Constants are left symbolic, uncalibrated. — [arXiv 2608.02582](https://arxiv.org/abs/2608.02582)
- Verdict: **no direct use.** Its RF/CF decomposition is a reasonable way to report agent-level efficiency (tokens per accepted task, including retries) in pancake A/Bs, but the paper has no data.

**8. AMESA: Adaptive Mixture-of-Execution Service Architecture for efficient LLM serving.** Doaa Shawky. Lead: Cairo University (OpenAlex). Future Generation Computer Systems, 2026 ([DOI 10.1016/j.future.2026.108889](https://doi.org/10.1016/j.future.2026.108889)); SSRN preprint ([DOI 10.2139/ssrn.6901711](https://doi.org/10.2139/ssrn.6901711))
- Core idea: rule-based routing among "execution experts": exact/semantic cache, in-flight request coalescing, batching, full inference. It targets template-driven backends with high request repetition. — [SSRN abstract via Crossref](https://doi.org/10.2139/ssrn.6901711)
- Verdict: **not applicable.** The name says "mixture of experts", but it is service-level routing with no model MoE. Single-user coding traffic has no duplicate concurrent requests.

**Checked and rejected (Africa):**
- UCT (Buys): MzansiLM ([arXiv 2603.20732](https://arxiv.org/abs/2603.20732)) and subword-segmentation dynamics ([arXiv 2511.09197](https://arxiv.org/abs/2511.09197)) are low-resource modeling, not inference.
- Pretoria (Marivate): Pula, Setswana LLMs ([NAACL 2025](https://doi.org/10.18653/v1/2025.naacl-long.338)), is data/training work.
- Stellenbosch (Kamper): speech and spoken-LM tokenization ([arXiv 2602.15537](https://arxiv.org/abs/2602.15537)) is out of scope.
- Wits (Rosman/James): RL, moral benchmarks. InkubaLM ([arXiv 2408.17024](https://arxiv.org/abs/2408.17024)) is Lelapa AI-led, so it is excluded.
- UM6P: early-exit work for edge CNNs and accelerators (e.g., "Hardware-Algorithm Co-Optimization of Early-Exit Neural Networks for Multi-Core Edge Accelerators", IEEE OJ-CS 2026, International University of Rabat lead with UM6P co-authors, per the OpenAlex listing) is not LLM work. GemMaroc ([arXiv 2505.17082](https://arxiv.org/abs/2505.17082)) is Darija fine-tuning.
- Wits: "Hierarchical Load Balancing for Multi-Cloud LLM Serving" (2026, OpenAlex listing) is multi-cloud serving, irrelevant at batch 1.
- Tokenization-cost papers ("The Token Tax", [arXiv 2509.05486](https://arxiv.org/abs/2509.05486); "The African Language Tax", [arXiv 2606.24460](https://arxiv.org/abs/2606.24460), which reports a median 1.88x token premium on o200k_base) were not confirmed as led by a listed university and only matter for African-language traffic.

### Inferences
- For pancake's daily goal (decode t/s on Flash-Next), Africa contributes **nothing to the kernel or offload path.** The best item, calibration-data alignment, is a requantization guideline: build the GSQ/imatrix calibration set from real opencode traces.
- The KNUST agent tools are the only African work aimed at coding agents. Cartograph (tool-definition prefill) and Resilient Write (write-failure recovery) are the two worth a quick opencode trial. All their numbers are self-reported on author-built setups, so treat them as hypotheses.

### Gaps
- PDF affiliation blocks were not read for the Chimoto/Bassett paper (AIMS vs UCT vs other) or for the KNUST arXiv papers whose OpenAlex affiliation was blank (HiveMind).
- Code availability for Calibrating Beyond English and Cartograph is unconfirmed.
- Makerere, Nairobi and AIMS-Rwanda/Senegal produced no matches on inference topics. Either nothing exists or OpenAlex affiliation coverage is incomplete. Weak evidence of absence.
- Masakhane / Deep Learning Indaba work found is data, benchmarks and models (IrokoBench etc.), with no inference-efficiency papers.

---

## Q2. Which Latin American university papers fit pancake / Avalon, and what would they gain us?

### Takeaway
Latin America has a few real systems papers: UFMG's coordinate-descent kernel autotuning (merged into TVM Ansor) and Unicamp's LP-GEMM (layout propagation across sequential GEMMs on CPU). They also have USP structured-pruning work and UBA work on cheap entropy and hidden-state probes. None addresses MoE offload, speculative decoding or caching. The most actionable item is the cheap coordinate-descent tuning recipe, applied offline to pancake's HIP launch parameters. Expected gain is low single digits on the tuned kernels, if any.

### Cited Findings

**1. Explore as a Storm, Exploit as a Raindrop: On the Benefit of Fine-Tuning Kernel Schedulers with Coordinate Descent.** Michael Canesche, Gaurav Verma, Fernando Magno Quintão Pereira. Lead: UFMG. arXiv June 2024. The companion "Droplet Search Algorithm for Kernel Scheduling" appeared in ACM TACO 2024 ([DOI 10.1145/3650109](https://doi.org/10.1145/3650109)). [arXiv 2406.20037](https://arxiv.org/abs/2406.20037)
- Core idea: cap Ansor's exploration at about 300 sampled kernels, then exploit the best one with coordinate descent (Droplet Search). — [arXiv 2406.20037](https://arxiv.org/abs/2406.20037)
- Key numbers: tuning the first 300 kernels "usually" gives better kernels in less time than Ansor's 10,000-sample search. This was replicated on 20 models on an AMD Ryzen 7 (x86), A100, RTX 3080 and A64FX. The patch was merged into Ansor in Feb 2024, and a similar patch was submitted to TVM MetaSchedule. — [arXiv 2406.20037](https://arxiv.org/abs/2406.20037)
- Code: upstream in Apache TVM (Ansor).
- Verdict: **method transfers, framework does not.** pancake's kernels are hand-written HIP/C++, not TVM. The transferable recipe is an offline coordinate-descent sweep over each hot kernel's launch/tile parameters (threads per block, rows per warp, unroll, LDS tile) on the RX 7900 XT, plus the AVX2 CPU expert kernel's block sizes on the 5700G. MOD_PLAN already records hand-tuned and "unroll: no gain" results, so the remaining headroom is probably small. Expected gain: 0-5% on the tuned kernels (inference from the A100 thread-block figure below), which is a smaller share of end-to-end time because CPU experts dominate the window.

**2. Enhancing the Power of Polyhedral-Based Optimizations with Coordinate-Based Hill Climbing.** Gaurav Verma, Michael Canesche, Fernando Magno Quintão Pereira. Lead: UFMG. arXiv Sept 2026. [arXiv 2609.03114](https://arxiv.org/abs/2609.03114)
- Key numbers: hill climbing over tile sizes and thread-block dimensions after Pluto picks the loop structure gives 1.06-1.28x geomean over Pluto defaults on x86/ARM (11 benchmarks), competitive with AutoTVM at lower search cost. Thread-block tuning on an A100 gave 5.5-8.5% over defaults. — [arXiv 2609.03114](https://arxiv.org/abs/2609.03114)
- Verdict: **same as item 1.** The A100 5.5-8.5% from thread-block tuning alone is the most relevant number for an untuned GPU kernel. For already hand-tuned kernels, expect less.

**3. LP-GEMM: Integrating Layout Propagation into GEMM Operations.** César Guedes Carneiro, Lucas Alvarenga, Guido Araujo, Sandro Rigo. Lead: Unicamp. arXiv Apr 2026. [arXiv 2604.04599](https://arxiv.org/abs/2604.04599)
- Core idea: decompose GEMM so that the packed layout propagates between dependent GEMMs, avoiding repacking and unpacking at every BLAS boundary. — [arXiv 2604.04599](https://arxiv.org/abs/2604.04599)
- Key numbers: 2.25x average over OpenBLAS on Intel x86 (AVX-512) for sequential GEMMs, competitive with MKL. Also RISC-V RVV. Shown on a standalone C++ Llama-3.2 inference path. — [arXiv 2604.04599](https://arxiv.org/abs/2604.04599)
- Verdict: **not applicable at batch 1.** pancake's CPU expert path runs custom quantized GEMV on pre-packed weights. The activation vector per expert is tiny (k+1 rows with speculation), so repacking costs are negligible compared to streaming weight bytes from DDR4. The paper also targets AVX-512, not AVX2. It might matter for CPU prefill of missed experts, where M is large, but pancake prefill runs on the GPU. Expected gain: about 0.

**4. Compressing LLMs with MoP: Mixture of Pruners.** Bruno Lopes Yamamoto, ..., Anna Helena Reali Costa, Artur Jordão. Lead: USP (c2d-usp). arXiv Feb 2026. [arXiv 2602.06127](https://arxiv.org/abs/2602.06127); code: [github.com/c2d-usp/Efficient-LLMs-with-MoP](https://github.com/c2d-usp/Efficient-LLMs-with-MoP)
- Key numbers: an iterative depth-vs-width structured-pruning search beats depth-only and width-only pruning on LLaMA-2/3, with 39% lower end-to-end latency at 40% compression. — [arXiv 2602.06127](https://arxiv.org/abs/2602.06127)
- Related USP work, same lab: **AMP**, head + MLP pruning, 30% pruning with minimal zero-shot loss, IJCNN 2025 ([arXiv 2504.21174](https://arxiv.org/abs/2504.21174)); **Layer Pruning with Consensus**, up to 78.8% FLOPs reduction on vision CNNs ([arXiv 2411.14345](https://arxiv.org/abs/2411.14345)); **Effective Layer Pruning Through Similarity Metric Perspective** ([arXiv 2405.17081](https://arxiv.org/abs/2405.17081)).
- Verdict: **not applicable.** All of it is dense-model structured pruning with recovery fine-tuning. For a 125B MoE already at 2 bits, the analog would be expert pruning or merging, which these papers do not study. Recovery fine-tuning of Flash-Next is infeasible on this hardware.

**5. The Virtues of Brevity: Avoid Overthinking in Parallel Test-Time Reasoning.** Raul Cavalcante Dinardi, Bruno Yamamoto, Anna Helena Reali Costa, Artur Jordão. Lead: USP. NeurIPS 2025 Workshop on Efficient Reasoning. [arXiv 2510.21067](https://arxiv.org/abs/2510.21067)
- Core idea: in best-of-N, picking the shortest solution matches self-consistency at much lower cost. The authors posit two regimes, a concise confident one and a verbose uncertain "overthinking" one, with a critical length where overthinking starts to dominate. — [arXiv 2510.21067](https://arxiv.org/abs/2510.21067)
- Verdict: **indirect support for thinking budgets; no direct use.** pancake samples one trajectory at batch 1, so best-of-N is unaffordable. The finding that long traces correlate with uncertainty and lower accuracy supports the existing per-level thinking budgets, and suggests combining a length cap with an uncertainty signal (see UBA item 6). Expected gain: none beyond what thinking budgets already deliver.

**6. Entropy Sentinel: Probing Entropy Traces for LLM Monitoring.** Pedro Memoli Buffa, Luciano Del Corro. Lead: UBA (Del Corro's group; affiliation from UBA listings of his other 2025-26 papers, not re-read in this PDF). arXiv Jan 2026 (v6). [arXiv 2601.09001](https://arxiv.org/abs/2601.09001)
- Core idea: summarize each response's top-k logprob entropy profile into a compact vector, then train a light classifier to predict response quality. — [arXiv 2601.09001](https://arxiv.org/abs/2601.09001)
- Key numbers: on 10 STEM benchmarks × 9 LLMs, estimates often track held-out accuracy. On subjective chat slices, correlation with LLM-judge scores reaches r = 0.91 for some models, though not for all. — [arXiv 2601.09001](https://arxiv.org/abs/2601.09001)
- Verdict: **low-cost diagnostic for the loop guard and thinking budgets.** pancake already computes logits, so a rolling entropy profile is essentially free. It could serve as (a) an extra loop-guard trigger, since degenerate repetition tends to show collapsing entropy, and (b) a DISCO feature (the MTP stop rule on `feat/disco` already uses top-1 probability; the Asia report recommended adding entropy). Expected gain: fewer wasted tokens on bad trajectories, unquantified, with 0 kernel speed impact.

**7. A BERTology View of LLM Orchestrations: Token- and Layer-Selective Probes for Efficient Single-Pass Classification.** Gonzalo Ariel Meyoyan, Luciano Del Corro. Lead: UBA (OpenAlex). ACL 2026 main conference. [arXiv 2601.13288](https://arxiv.org/abs/2601.13288)
- Core idea: train small probes (100K-param scoring gate, up to 35M-param downcast MHA) on the serving model's token×layer hidden states, so they classify in the same forward pass used for generation. This replaces a separate guard model. — [arXiv 2601.13288](https://arxiv.org/abs/2601.13288)
- Key numbers: beats logit-only reuse (MULI) and is competitive with much larger task-specific baselines on safety and sentiment, at near-serving latency. Validated on MoE backbones (Qwen3-30B-A3B, GPT-OSS-20B) and Llama-3.2-3B. — [arXiv 2601.13288](https://arxiv.org/abs/2601.13288)
- Verdict: **speculative but interesting.** The same "free probe on hidden states" trick could predict an impending malformed tool call or a loop before it is decoded, feeding tool-call recovery or the loop guard at no extra forward pass. It needs labeled traces from opencode sessions and a training step. Expected gain: unknown, an R&D item, not a quick win.

**8. Are Optimal Algorithms Still Optimal? Rethinking Sorting in LLM-Based Pairwise Ranking with Batching and Caching.** Wisznia, ..., Luciano Del Corro. Lead: UBA. ACL 2025 (short). [arXiv 2505.24643](https://arxiv.org/abs/2505.24643)
- Verdict: **not applicable.** It covers reranking cost models, where LLM calls dominate the cost.

**9. PRISM: Transfer and Limits of Activation Interventions in Tool-Calling Language Models.** Giovana Kerche Bonás, Thales Sales Almeida, Marcelo da Silva Reis, Rodrigo Nogueira, Marcos M. Raimundo. Lead: Unicamp (OpenAlex). arXiv Oct 2026. [arXiv 2610.20808](https://arxiv.org/abs/2610.20808); code: [github.com/GiKerche/prism](https://github.com/GiKerche/prism)
- Key numbers: steering directions estimated from an NF4-quantized copy transfer to full precision and across languages. Unsafe tool-call violations in Llama-3.1-8B fall by 21.95 points on held-out Arabic. — [arXiv 2610.20808](https://arxiv.org/abs/2610.20808)
- Verdict: **low.** It is a safety intervention for tool calls, not efficiency or validity. The one notable point is that behavior estimated on a low-bit copy transferred to the full model, which is mildly reassuring for Q2 deployments, but it was not tested at 2 bits.

**Checked and rejected (Latin America):**
- Unicamp/Maritaca: Sabiá-2/3/4, Juru, Curió-Edu and the Portuguese benchmarks are company-led or about data and evaluation, e.g., the Sabiá-4 report ([arXiv 2603.10213](https://arxiv.org/abs/2603.10213)).
- Unicamp (Araujo): ConvBench ([arXiv 2407.10730](https://arxiv.org/abs/2407.10730)) and NPU tensor slicing ([arXiv 2304.03013](https://arxiv.org/abs/2304.03013)) target convolution and NPUs.
- UFMG side-channel mitigation for LLM serving (SBSeg 2026, OpenAlex listing) covers security.
- PUC Chile (Barceló): logic and automata theory of transformers ([arXiv 2509.24024](https://arxiv.org/abs/2509.24024)) is theory only.
- SkipDecode ([arXiv 2307.02628](https://arxiv.org/abs/2307.02628), Del Corro first author) is Microsoft-led, so it is excluded. It is also a batched early-exit method.
- Nothing relevant was found for UFRJ, PUC-Rio, CONICET, U. de Chile, CENIA, UNAM, Tec de Monterrey, CINVESTAV or Uniandes on inference efficiency. Targeted web searches for these institutions plus quantization, FPGA, speculative decoding or KV cache returned only non-affiliated papers ([search example: LUT-LLM, UCLA](https://arxiv.org/abs/2511.06174)).

### Inferences
- The only Latin American work that could move a pancake number is the UFMG coordinate-descent tuning methodology: cheap, offline, generic. Since MOD_PLAN shows GPU time at about 15.9 of 24.8 ms per token with CPU experts setting the rest, a few % on GPU kernels translates to roughly 1-3% end-to-end at best (my estimate).
- UBA's entropy and hidden-state probe line is the most relevant research direction for the agent features: loop guard, thinking budgets, DISCO's entropy feature and tool-call recovery. It costs little because the signals are already computed.
- For a single-user 2-bit MoE on consumer AMD hardware, Latin American output does not replace the Asian, US or European offload and speculative-decoding literature.

### Gaps
- The Latin American OpenAlex second pass (serving, edge, early exit, structured output, test-time compute, sparse attention) failed on the daily rate limit. Some USP, Unicamp or UFRJ items in those topics may be missing.
- Search engines do not filter reliably by affiliation, so "nothing found" for Mexico, Chile, Argentina (beyond UBA) and Colombia is weak evidence of absence.
- Affiliations for Entropy Sentinel and PRISM come from OpenAlex/author listings, not the PDF.
- No paper in either region reports results on AMD GPUs, ROCm, PCIe 3.0 or MoE expert offload, so every applicability verdict above is an inference.
