# Canadian university AI papers (excluding Montreal and Toronto) for pancake / Avalon: Waterloo, Alberta/Amii, UBC, SFU, Queen's, others

Scope: Nov 2022 - Nov 2026. Target: batch-1 decode of Qwen3.8-Flash-Next 125B MoE hybrid (Gated DeltaNet + sparse attention, 512 experts, MTP head) on RX 7900 XT 20 GB + Ryzen 5700G + 57 GB RAM + NVMe, with experts tiered across VRAM, pinned RAM and NVMe, the CPU computing missed experts, MTP plus prompt-lookup drafting, and an opencode agent.
Repeat check against /home/dan/reports/: FineMoE/fMoE (Waterloo co-author), Medusa, EAGLE (named only), SWE-bench, LiveCodeBench, MoE-Infinity, Fiddler and HOBBIT already appear in earlier reports. Cactus, SWE-Effi, the multi-hunk/Maple study, AIDev, MMLU-Pro, AceCoder and VerlTool do not appear there (grep, 2026-11-09).
Method note: about 20 web searches/fetches. The search engine handles affiliation-filtered queries poorly, so coverage of UBC, SFU, McMaster, Ottawa, Calgary and Western is thin (see Gaps).

## Q1. Which ideas from these groups fit batch-1 MoE decode with offloaded experts, drafting on a hybrid recurrent model, or long prefill on consumer AMD?

### Takeaway
The best new candidate is **Cactus** (Alberta/Amii, ICLR 2026). It is training-free and changes only the speculative acceptance rule using element-wise operations, so it can be dropped into pancake's MTP and prompt-lookup verifier when sampling with temperature > 0. In dense Qwen3 tests it raised acceptance length about 27% (4.27 to 5.44). Waterloo's EAGLE line is the main drafting work from these groups. Pancake already has a native MTP head, though, so only EAGLE-2's confidence-driven dynamic draft length can be borrowed cheaply. Waterloo's MoE-offload contribution (FineMoE) is already covered in earlier reports. No relevant work from these universities on long prefill on consumer AMD turned up.

### Cited Findings

**Cactus: Accelerating Auto-Regressive Decoding with Constrained Acceptance Speculative Sampling**. Yongchang Hao, Lili Mou (Univ. of Alberta Computing Science + Amii; **Alberta leads, sole affiliation**). ICLR 2026, arXiv 2604.04987 (Apr 2026). Code: https://github.com/MANGA-UOFA/Cactus
- Core idea: speculative sampling is recast as constrained optimization. Cactus picks an adjusted target distribution that maximizes the chance of accepting the drafted token, subject to a KL budget δ from the verifier. A second-order Taylor approximation gives a closed-form boost to the drafted token's probability, and the remaining mass is redistributed proportionally. It is training-free, uses only element-wise operations, and needs only the verifier's probability for the drafted token. — [alphaXiv 2604.04987](https://www.alphaxiv.org/abs/2604.04987.md)
- Bound: total divergence from the verifier is at most min{Γ(δ), D_f(p‖q)}, with Γ(0) = 0. The paper notes that typical acceptance sampling (TAS, the Medusa-style rule) has no explicit divergence control. — [alphaXiv 2604.04987](https://www.alphaxiv.org/abs/2604.04987.md)
- Numbers: Qwen3-14B verifier with a 0.6B drafter on GSM8K, m = 10: acceptance length 4.27 (standard SpS) → 5.44 (Cactus, δ = 1), 32% fewer rejections, about 1.9× wall-clock over the verifier alone. Spec-Bench overall: 1.88× vs 1.81× for SpS with no δ tuning. GPQA: 44.95 (δ = 0.75) vs 40.07 for the verifier alone, while TAS dropped to 38.89. With a 4B drafter: acceptance length 7.76, 76% fewer rejections. — [alphaXiv 2604.04987](https://www.alphaxiv.org/abs/2604.04987.md)
- Models tested: Qwen3 0.6B-32B, Gemma 2B/9B, DeepSeek-R1 distills, LLaMA 1B/8B. All are dense; no MoE or hybrid model was tested. — [alphaXiv 2604.04987](https://www.alphaxiv.org/abs/2604.04987.md)

**EAGLE-3: Scaling up Inference Acceleration of LLMs via Training-Time Test**. Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang (Zhang is at **Univ. of Waterloo + Vector Institute; co-author / senior**, other authors at Peking U. and Microsoft Research). NeurIPS 2025, arXiv 2503.01840. Code: https://github.com/SafeAILab/EAGLE
- Core idea: drop feature prediction and predict tokens directly, fusing features from several target layers (low, mid, high) instead of only the top layer. "Training-time test" trains the draft head on its own multi-step outputs. — [arXiv 2503.01840](https://arxiv.org/html/2503.01840v3); [NeurIPS 2025 proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c7b5a35ea98b62512a869c19ea7b03cb-Abstract-Conference.html)
- Numbers: up to 6.5× speedup, about 1.4× over EAGLE-2. 1.38× throughput at batch 64 in SGLang v0.4.4 (author-reported). — [arXiv 2503.01840](https://arxiv.org/pdf/2503.01840)

**EAGLE (ICML 2024, arXiv 2401.15077)** and **EAGLE-2 (EMNLP 2024, arXiv 2406.16858)** come from the same SafeAILab group with Hongyang Zhang (Waterloo/Vector) as co-author. EAGLE autoregresses on second-to-top-layer features. EAGLE-2 adds a context-dependent dynamic draft tree that grows and prunes using the draft model's confidence scores as a proxy for acceptance probability. — [arXiv 2401.15077](https://arxiv.org/abs/2401.15077); [arXiv 2406.16858](https://arxiv.org/abs/2406.16858). (Exact speedup figures for these two were not re-fetched in this session; see the EAGLE-3 paper for comparisons.)

**FineMoE (fMoE): fine-grained expert offloading for MoE serving**. Stevens (lead), Rice, **Univ. of Waterloo (Hong Zhang, co-author)**, Rutgers. arXiv 2502.05370, EuroSys '26. 47% lower latency and 39% higher expert hit rate vs prior systems (author-reported). — [arXiv 2502.05370](https://arxiv.org/html/2502.05370v2); [alphaXiv overview](https://www.alphaxiv.org/abs/2502.05370.md). **Already in earlier reports (US and MIT/Harvard); not repeated here.**

### Inferences
- **Cactus → pancake verifier (high fit, low effort).** The change is one closed-form adjustment per drafted token in the accept/reject step. It needs only p_verifier(drafted token) and works with any drafter, including deterministic prompt-lookup drafts (treat q as a point mass). It does nothing under pure greedy decoding. If opencode runs Qwen at the recommended temperature (about 0.6-0.7), it applies. Expected gain: in pancake every extra accepted token saves a full decode step dominated by expert misses. A +10-25% acceptance-length gain on MTP chains might give roughly +5-15% decode t/s; this is my estimate and needs measurement. Caveats: (a) it is lossy within a KL budget, so gate it behind a flag and check on a coding eval (MMLU-Pro or LiveCodeBench subset); (b) longer accepted chains under MoE offload mean more distinct experts per verification step, which is the AcceptMoE concern from the Europe report. Pairing Cactus with a cap on extra expert misses per verification would be the safe design.
- **EAGLE-3 → mostly not applicable.** Training an EAGLE-3 head for a 125B MoE hybrid on a single 20 GB card is impractical (it needs target hidden states over a large corpus), and the native MTP head already plays that role. The borrowable part is **EAGLE-2's confidence-gated dynamic draft length/tree**. If the MTP chain does not already stop when draft confidence falls below a threshold, adding that is cheap. Prefer a chain over a wide tree because tree width multiplies the experts touched during verification. Expected gain is small (a few percent) and only where pancake currently drafts a fixed length.
- **Hybrid recurrent drafting:** none of these papers addresses Gated DeltaNet state rollback on rejection. Cactus's higher acceptance does reduce how often recurrent state must be rolled back or recomputed, which is a second-order benefit specific to the hybrid model.

### Gaps
- I did not find a Waterloo, UBC, SFU, Queen's, McMaster, Ottawa, Calgary or Western paper on MoE expert offload/caching for single-GPU batch-1 decode beyond FineMoE. Targeted searches ("UBC Prashant Nair MoE", "Simon Fraser mixture-of-experts expert caching", "Alameldeen KV cache compression") returned nothing affiliated.
- I found no Canadian (non-Montreal/Toronto) work on long prefill or kernels for consumer AMD/ROCm.
- I found no evidence that Cactus has been tested on MoE or with MTP heads.

## Q2. Learned systems / RL-for-caching work applicable to the expert cache

### Takeaway
I found no University of Alberta / Amii paper (or one from any other in-scope Canadian university) on RL- or ML-learned cache replacement or prefetching for MoE experts or LLM serving in 2022-2026. The closest learned expert-cache papers are from non-Canadian groups and are listed only as pointers. Given that ClassLRU scoring and reuse-aware admission already gave no gain in pancake, the evidence base here is weak.

### Cited Findings
- A search for "University of Alberta reinforcement learning cache replacement" returned only non-Alberta work: an Illinois MS thesis (2018), HUST's end-to-end DRL policy (ICAPS 2022), an IEEE Access 2024 RL replacement policy, and Parrot (imitation learning of Belady). — [ICAPS 2022](https://ojs.aaai.org/index.php/ICAPS/article/view/19840); [Parrot arXiv 2006.16239](https://ar5iv.arxiv.org/html/2006.16239)
- Pointer, not Canadian: **FlashMoE** (Kim, Lee, Han, Yoo, Kim; affiliations not shown on the arXiv page, likely KAIST given Hoi-Jun Yoo, unverified). arXiv 2601.17063, Jan 2026. Experts live on SSD. A small FFN (about 113 KB per layer) takes inverse-recency and normalized-frequency features, is trained to imitate Belady eviction, and runs asynchronously. Hit rate is up to 21% above LRU and 51% above LFU (OLMoE). Speed: +22% vs LRU for OLMoE, but only **+7% for Qwen3-30B-A3B**. Tested on RTX 5070 Ti + PCIe 5 NVMe. — [arXiv 2601.17063](https://arxiv.org/html/2601.17063v1)
- Pointer, not Canadian: **MoE-Beyond** (Univ. of Pennsylvania), arXiv 2508.17137. A lightweight transformer predicts batch-1 expert activations for prefetch. 97.5% accuracy / 86.6% F1. Simulated GPU hit rate rises from about 17% to over 70% vs MoE-Infinity when 10% of experts fit. Trained on about 66M traces from DeepSeek-V2-Lite. Code: github.com/ngavhane/moe-beyond. — [arXiv 2508.17137](https://arxiv.org/html/2508.17137v1)

### Inferences
- The only learned-replacement result on a model close in shape to the target (Qwen3-30B-A3B) gives a single-digit gain (+7%). That is consistent with pancake's null results for ClassLRU and reuse-aware admission. A Belady-imitation FFN would likely bring at most a few percent over a well-tuned frequency/recency policy at 512 experts. Low priority.
- A cheaper experiment that fits pancake: log real opencode expert traces and compute Belady's hit rate offline against the current policy. If the gap is under about 10%, close this line for good.

### Gaps
- I may have missed Alberta RL-for-systems work (e.g., from the RLAI group) because affiliation-filtered search is unreliable. A DBLP/Scholar pass over Alberta CS faculty (e.g., Martha White, Adam White, J. N. Amaral) was not done.

## Q3. Agent, reasoning and evaluation work usable for opencode

### Takeaway
Queen's (Ahmed E. Hassan's group) and UBC (Ali Mesbah) give the most directly usable agent findings. Failed agent runs cost 4× or more in tokens and time (they "do not fail fast"). Input tokens snowball linearly with each call. A repository-skeleton MCP server raised a weaker agent's repair accuracy by 30% relative. These support pancake's loop guard and conversation cache and suggest one cheap opencode add-on. Waterloo's TIGER-Lab supplies MMLU-Pro as a quantization-sensitive regression eval. Its training-oriented work (AceCoder, VerlTool) and Waterloo's implicit-CoT work are not usable without training.

### Cited Findings

**Beyond Accuracy: Behavioral Dynamics of Agentic Multi-Hunk Repair**. Noor Nashid, Daniel Ding, Ali Mesbah (**UBC, lead**), Keheliya Gallaba, Ahmed E. Hassan (Queen's, co-authors). arXiv 2511.11012, Nov 2025, TOSEM format.
- Setup: 4 agents (Claude Code, Codex, Gemini-cli, Qwen Code), 372 multi-hunk Java bugs (Hunk4J), 1,488 trajectories. — [arXiv 2511.11012](https://arxiv.org/html/2511.11012v1)
- Repair accuracy: Claude Code 93.28%, Codex 87.10%, Gemini-cli 41.67%, **Qwen Code 25.81%**. Localization: Codex 75.0%, Claude Code 64.0%, Gemini-cli 46.5%, Qwen Code 38.2%. — [arXiv 2511.11012](https://arxiv.org/html/2511.11012v1)
- Failed runs used 39%-343% more tokens and 43%-427% longer runtimes ("do not fail fast"). Failure modes: over-modification without validation (Gemini-cli), over-exploration without action (Claude Code), navigation overhead (Qwen Code). — [arXiv 2511.11012](https://arxiv.org/html/2511.11012v1)
- **Maple**, an MCP server that provides class/method lookup and class skeletons: Gemini-cli accuracy 40% → 52% and localization 32% → 42%, but compilation fell 98% → 90%. Qwen Code accuracy rose only 22% → 24%. — [arXiv 2511.11012](https://arxiv.org/html/2511.11012v1)

**SWE-Effi: Re-Evaluating Software AI Agent System Effectiveness Under Resource Constraints**. Huawei (co-first authors Zhiyu Fan, Kirill Vasilevski; **Huawei/Huawei Canada leads**, 5 of 9 authors), with CUHK-SZ, KCL and Ahmed E. Hassan (Queen's, co-author). arXiv 2509.09853, Sep 2025. Code/leaderboard: https://centre-for-software-excellence.github.io/SWE-Effi/ (borderline in scope because a company leads).
- Metrics: AUC of resolve rate under token, cost, CPU-time and inference-time budgets, on a 50-issue SWE-bench-Verified subset × 5 scaffolds × 3 LLMs. — [alphaXiv 2509.09853](https://www.alphaxiv.org/abs/2509.09853.md)
- "Token snowball": input tokens grow about linearly per call. SWE-Agent with GPT-4o-mini averaged 181 calls and about 8.1M input tokens, vs 35.5 calls and 440k for Qwen3-32B. — [alphaXiv 2509.09853](https://www.alphaxiv.org/abs/2509.09853.md)
- "Expensive failures": unresolved SWE-Agent runs averaged about 8.8M tokens / 658 s vs about 1.8M / 167 s for resolved runs (more than 4×). AutoCodeRover + Qwen3-32B reached 38% resolved with 14.7 calls and 55.5K input tokens. Agentless + Qwen3-32B reached 48% but used 83.1 calls. — [alphaXiv 2509.09853](https://www.alphaxiv.org/abs/2509.09853.md)

**AIDev: Studying AI Coding Agents on GitHub**. Hao Li, Haoxiang Zhang, Ahmed E. Hassan (**Queen's, lead**). arXiv 2602.09185 (earlier 2507.15003). Dataset of 932,791 agent-authored PRs from 5 agents, cutoff 2025-08-01. — [arXiv 2602.09185](https://arxiv.org/html/2602.09185v1)

**MMLU-Pro** (TIGER-Lab, Wenhu Chen, **Waterloo leads**). NeurIPS 2024 Datasets & Benchmarks, arXiv 2406.01574. A harder 10-option, reasoning-heavy version of MMLU; the repo was still maintained in Mar 2026. — [TIGER-Lab HF/GitHub listing](https://backiee.wasmer.app/http_github_com/TIGER-AI-Lab); [arXiv 2406.01574](https://arxiv.org/abs/2406.01574)

**AceCoder: Acing Coder RL via Automated Test-Case Synthesis** (TIGER-Lab, **Waterloo leads**). arXiv 2502.01718, Feb 2025. Synthesized tests serve as RL rewards. With best-of-32 reranking, a 7B model matches DeepSeek-V2.5 236B. — [arXiv 2502.01718](https://arxiv.org/html/2502.01718v4); [project page](https://tiger-ai-lab.github.io/AceCoder)

**VerlTool** (Dongfu Jiang, Wenhu Chen et al., **Waterloo leads**). arXiv 2509.01055. An agentic RL-with-tools framework on VeRL covering 6 task families including SWE-bench; asynchronous rollouts give more than 2× rollout speedup. Code: TIGER-AI-Lab/verl-tool. — [arXiv 2509.01055](https://arxiv.org/html/2509.01055v2)

**Implicit CoT** (Yuntian Deng, now **Waterloo** faculty; the original work was done before he joined, with Harvard/Microsoft). arXiv 2311.01460 (2023) and "From Explicit CoT to Implicit CoT" (2024). Reasoning moves into hidden states at no-CoT speed, but it needs k-1 training stages. — [arXiv 2311.01460](https://www.arxiv.org/abs/2311.01460); [Cambridge talk, Oct 2025](https://talks.cam.ac.uk/talk/index/237769)

**Length-based reward shaping for efficient reasoning**. arXiv 2505.15612 (affiliations include Waterloo, HKUST, CityU HK, Apple; lead unclear, title not confirmed). It unifies efficient-reasoning methods as length reward shaping and proposes a truncation step-function reward around a target length. — [arXiv 2505.15612](https://arxiv.org/html/2505.15612v1)

### Inferences
- **Loop guard / fail-fast (applicable now, no training).** Both the UBC/Queen's and SWE-Effi papers say failed trajectories are the main cost (4× or more tokens and time). For a local 126 t/s engine this matters even more, because one runaway session costs tens of minutes. Concrete opencode/pancake hooks:
  - detect "N consecutive edits without a build/test" (the over-modification pattern);
  - detect "N consecutive search/read calls without an edit" (the over-exploration pattern);
  - then inject a nudge or end the session.
  These extend the existing loop guard, which presumably catches repeated text rather than behavioral patterns.
- **Conversation cache is validated by the token snowball.** Input grows linearly per call, so prefix reuse is where agent prefill time is won. Pancake's conversation cache addresses this directly. Measure its hit rate on long opencode sessions. With Gated DeltaNet, checkpointing recurrent state at tool-call boundaries is what makes reuse possible.
- **Maple-style repo-skeleton MCP**: it gave +30% relative accuracy for a weaker agent but little for Qwen Code. Expected gain for opencode + Qwen3.8 is uncertain and worth a quick A/B test on a handful of tasks. Watch the compilation regression the paper saw.
- **Use SWE-Effi's AUC-under-time-budget** as the metric for pancake changes that affect agent quality (thinking budgets, tool-call recovery, Cactus δ). It captures speed and success together, unlike t/s alone.
- **MMLU-Pro as a regression check for expert quantization or tiering changes.** It is reasoning-heavy and 10-option, so it is more sensitive than MMLU. Run a fixed few-hundred-question subset before/after each change.
- Not applicable: AceCoder, VerlTool, implicit CoT and length-reward RL all require training a 125B model. AceCoder's best-of-N reranking costs N× decode at batch 1. The truncation-reward work suggests that budget-based thinking caps (already in pancake) mirror what RL length shaping learns, so no change is needed there.

### Gaps
- I did not verify the title, lead institution or numbers for 2505.15612.
- I found no Queen's/UBC study on local open-weight models in opencode specifically. Qwen Code is the closest proxy (Qwen3-Coder via its CLI).
- I did not find TIGER-Lab's newer coding/agent benchmarks (2025-2026) or Waterloo work on thinking budgets.
- McMaster, Ottawa, Calgary and Western produced no hits in this pass; this is a search-coverage gap, not evidence that no such work exists.
