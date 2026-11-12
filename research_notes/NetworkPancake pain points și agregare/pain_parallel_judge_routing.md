# Pain points: parallel thinking + voting, judges/verifiers, routing, speculative decoding (as of 2026-11-10)

Tags: [M] measured/reported in primary source (abstract or page fetched), [C] claimed in secondary summary / snippet only, [U] unverified or not found. Only abstracts/pages listed were read; full-text numbers were NOT checked unless stated. Existing note `Plan NetworkPancake inferență distribuită/academic_papers.md` already covers spec-decoding-over-WAN (2602.16760, PicoSpec, FlowSpec, SLED, GoodSpeed); not repeated here.

## Q1. Accuracy gain from N small-model chains, saturation, small ensembles vs one bigger model

### Takeaway
Coverage (pass@N) grows roughly log-linearly over ~4 orders of magnitude of samples, but real accuracy only follows where an automatic verifier exists (code with tests, formal proofs); for free-form math majority voting/reward models plateau after a few hundred samples. Small-model sampling beats a much larger model only when the small model has non-trivial base success and a good verifier/search; at larger budgets the larger model wins again.

### Cited Findings
- Large Language Monkeys: coverage scales over four orders of magnitude, often log-linear / exponentiated power law; SWE-bench Lite with DeepSeek-Coder-V2-Instruct 15.9% (1 sample) to 56% (250 samples), above prior single-sample SOTA 43% [M] — [arXiv 2407.21787](https://arxiv.org/abs/2407.21787)
- Same paper: with no automatic verifier, majority voting and reward models plateau beyond "several hundred samples" [M] — [arXiv 2407.21787](https://arxiv.org/abs/2407.21787); search summary adds: GSM8K/MATH coverage with Llama-3 models exceeds 95% at 10,000 samples, and a cheaper model with 5 samples beats one GPT-4o/Claude 3.5 Sonnet sample on cost and issues solved [C] — [Stanford page](https://scalingintelligence.stanford.edu/pubs/large_language_monkeys)
- Snell et al.: compute-optimal per-prompt allocation improves efficiency >4x over best-of-N; test-time compute can outperform a 14x larger model in FLOPs-matched comparison, but only on problems where the small model has "somewhat non-trivial" success rates [M] — [arXiv 2408.03314](https://arxiv.org/abs/2408.03314)
- Liu et al.: a 1B model can beat a 405B model on MATH-500 under compute-optimal TTS; best strategy depends on policy model, PRM and difficulty; search-based methods better for policy <7B, best-of-N better for larger; claims up to 256x more efficient than majority voting [C: search snippet, full text not read] — [arXiv 2502.06703](https://arxiv.org/pdf/2502.06703)
- Inference scaling laws (arXiv 2408.00724): smaller models win at low compute budgets, but once small-model accuracy saturates, larger models are favorable; Llemma-7B + REBASE beat Llemma-34B with majority voting at all budgets [C] — [arXiv 2408.00724](https://arxiv.org/html/2408.00724v3)
- Adaptive stopping: Adaptive-Consistency reduces sample budget up to 7.9x with <0.1% average accuracy loss across 17 datasets, 3 LLMs [M] — [arXiv 2305.11860](https://arxiv.org/abs/2305.11860); a figure caption reportedly says 3.3x average with 0.04% drop [C] — [search result](https://arxiv.org/pdf/2305.11860v2.pdf)
- Difficulty-adaptive SC (arXiv 2408.13457) notes ASC and ESC add substantial extra input cost on problems needing many samples, sometimes outweighing output savings [C] — [arXiv 2408.13457](https://arxiv.org/html/2408.13457v2). ESC (Li et al. 2024, windowed early stop) exists per that paper's description; ESC primary paper not fetched [U]. ConSol (SPRT-based stopping) [C] — [arXiv 2503.17587](https://arxiv.org/html/2503.17587v1)

### Inferences
- Plan phase 3 (10-200 chains + quorum + strong verifier) is supported for verifiable tasks (code with tests, math with checkable answers) and weak for free-form tasks: the plateau at several hundred samples is above the 10-200 range, so saturation is not the binding limit; verifier quality is.
- Gain conditions: small model must already solve the problem sometimes; for hard problems at zero base success N chains do nothing.
- Quorum stopping is cheap to add and measured to save 3-8x samples at near-zero accuracy loss, which matters most for a network of slow nodes.

### Gaps
- No clean paper found comparing N-chain small-model ensemble vs one larger model at equal wall-clock on consumer hardware (the 14x / 405B results are FLOPs-matched, GPU-server). Full-text numbers of 2502.06703 not verified.

## Q2. Error correlation: same model vs different models; does diversity help?

### Takeaway
Errors are strongly correlated across models, even different architectures and providers, and correlation rises with accuracy. Diversity helps less than independence would predict. Plan phase 2b experiment (two different big models, joint-wrong rate vs independence) is well motivated.

### Cited Findings
- Kim et al., ICML 2025: >350 LLMs, two leaderboards plus resume screening; on one leaderboard models agree on wrong answers in about 60% of cases when both err; shared architecture/provider raises correlation, but larger and more accurate models have highly correlated errors even with distinct architectures/providers; implications stated for LLM-as-judge, hiring, multi-agent systems [M for abstract; 60% figure from search summary C] — [arXiv 2506.07962](https://arxiv.org/pdf/2506.07962), [PMLR](https://proceedings.mlr.press/v267/kim25e.html)
- Overthinking paper: later solutions mostly re-verify earlier ones and lack diverse strategies [M per ICML slides summary C] — [arXiv 2412.21187](https://arxiv.org/abs/2412.21187)

### Inferences
- Same-model chains at temperature share systematic errors; majority vote cannot fix a consistent wrong answer. Cross-model mixing reduces but does not remove correlation.

### Gaps
- No measurement found for voting gain specifically from heterogeneous small-model ensembles vs homogeneous (e.g., "mixture of agents" style studies not searched). The 60% figure is multiple-choice, so it is inflated by chance agreement among few options; the paper's metric is claimed to correct for accuracy, not for option count [C].

## Q3. Judges and verifiers: failures; what works for code/math/extraction

### Takeaway
LLM judges show position, verbosity and self-enhancement bias; PRMs generalize poorly when trained on Monte Carlo labels and best-of-N evaluation overstates them; executable verification (tests, schema validation) is the most reliable where available. Judge errors correlate with generator errors.

### Cited Findings
- MT-Bench paper: GPT-4 judge >80% agreement with humans (same as human-human); documented position bias (first position favored for GPT-3.5/Claude-v1), verbosity bias, self-enhancement bias; some biases minor or mitigable [M] — [arXiv 2306.05685](https://arxiv.org/pdf/2306.05685)
- Qwen PRM lessons: Monte Carlo-labeled PRM data generalizes worse than LLM-judge or human labels; ProcessBench step localization much lower for MC-trained PRMs; best-of-N evaluation is biased (policy answers right with flawed process; PRMs drift to outcome-style scoring); recommends ProcessBench alongside BoN [M] — [arXiv 2501.07301](https://arxiv.org/pdf/2501.07301). Qwen2.5-Math-PRM-7B/72B 73.5/78.3 ProcessBench F1 [C: via HF discussion, not verified] — [HF discussion](https://huggingface.co/datasets/rl-llm-wiki/knowledge-base/discussions/670)
- PRMBench (arXiv 2501.03124): reports PRMs and critic models still show failure modes [C, not examined] — [arXiv 2501.03124](https://arxiv.org/pdf/2501.03124v5)
- Large Language Monkeys: code/formal proofs benefit fully from verifier (tests); unverifiable domains plateau [M] — [arXiv 2407.21787](https://arxiv.org/abs/2407.21787)
- Kim et al. (above): correlated errors specifically affect LLM-as-judge [M] — [arXiv 2506.07962](https://arxiv.org/pdf/2506.07962)

### Inferences
- Best-of-breed verification for plan tasks: unit tests (code), JSON-schema/regex/source-span check (extraction), exact/symbolic answer check (math); big-model judge only as last resort, with position swap and length control. Using the strong node as verifier is exposed to correlated errors with the workers if same family.
- Verification cost vs generation: verifying a chain by a big model means a prefill-heavy pass of the whole chain (cheaper per token than decode, but N chains each need it). No sourced numbers found.

### Gaps
- No quantified numbers for bigger-model verification cost vs generation, no code-specific judge-vs-tests comparison, no small-judge (<8B) bias numbers retrieved. Self-preference quantitative studies (e.g., later 2024-26 work) not searched.

## Q4. Routing and cascades: accuracy, latency, maturity, tradeoffs

### Takeaway
Papers report 2x to 98% cost cuts at near-equal quality, but on benchmark mixes; independent evaluations find routers lean on category heuristics and generalize poorly out of distribution. RouteLLM is popular but looks lightly maintained; vLLM Semantic Router is far more active.

### Cited Findings
- RouteLLM: routers trained on human preference data with augmentation; >2x cost reduction in some cases with no quality loss; routers transfer when strong/weak models are swapped [M] — [arXiv 2406.18665](https://arxiv.org/abs/2406.18665)
- FrugalGPT: matches best individual LLM (GPT-4) with up to 98% cost reduction, or +4% accuracy at equal cost [M] — [arXiv 2305.05176](https://arxiv.org/abs/2305.05176)
- Hybrid LLM: up to 40% fewer calls to the large model with "no drop in response quality" (abstract, unquantified), quality target tunable at test time [M] — [arXiv 2404.14618](https://arxiv.org/abs/2404.14618)
- "How Robust Are Router-LLMs?" (DSC benchmark): tested RouteLLM and other preference-based routers; they rely on category heuristics rather than query complexity (e.g., a BERT router sent all coding queries to one model); existing benchmarks ill-suited; adversarial/jailbreak inputs probed [C: snippet only] — [arXiv 2504.07113](https://arxiv.org/pdf/2504.07113)
- RouterXBench (Feb 2026): argues routers are mostly evaluated in-distribution; OOD robustness is one of three evaluation axes [C] — [arXiv 2602.11877](https://arxiv.org/abs/2602.11877)
- Shnitzer et al.: generalization to new tasks is a central open question for benchmark-trained routers [C] — [arXiv 2309.15789](https://arxiv.org/pdf/2309.15789v1.pdf)
- Practitioner blog (opinion, not evidence): benchmark-trained routers fail in segments like bulk processing, programmatic clients, thin-language traffic — [tianpan.co](https://tianpan.co/blog/2026-04-27-model-router-eval-traffic-mismatch-open-loop)
- RouteLLM repo: Apache-2.0, ~5.7k stars, 452 forks, 29 open issues, 16 open PRs, 175 commits, Releases empty; routers `mf` (recommended), `sw_ranking`, `bert`, `causal_llm`, `random`; README inconsistently says 4 routers; last-commit date not visible [M for page; maintenance status U] — [GitHub](https://github.com/lm-sys/RouteLLM)
- vLLM Semantic Router: Apache-2.0, ~6.4k stars, 1.1k forks, 3,234 commits, 359 open issues, 255 PRs; v0.4 "Hermes" announced 2026-09-24, newest news item 2026-10-14; pitched as policy-driven decision layer across models (guardrails, Kubernetes, mixture-of-models); no latency overhead figures on page [M] — [GitHub](https://github.com/vllm-project/semantic-router)

### Inferences
- A router adds a classifier call per request; no sourced latency number found for either project. For a LAN with prompts of ~1-10k tokens, a small embedding classifier is likely milliseconds [E, unsourced].
- Router wrongness rate is not reported in a single comparable figure; treat router quality as must-measure on own traffic. RouteLLM routes between two models (strong/weak) only; Semantic Router is closer to a gateway and aimed at vLLM/Kubernetes fleets, so heavier than the plan needs.

### Gaps
- No published per-request routing latency; no RouteLLM last-commit date (gh CLI unavailable, page lacked dates); no mis-route rate. Semantic Router license seen as Apache-2.0 on repo page only.

## Q5. Speculative decoding limits (acceptance, MoE, network)

### Takeaway
On MoE, verifying a draft tree touches many experts, eroding gains and sometimes slowing serving; fixes exist (expert budgeting, adaptive K) but are preprints. Over-network cost is already covered in academic_papers.md.

### Cited Findings
- Draft-tree expert union: on OLMoE-1B-7B a 127-token tree activates 54 of 64 experts per layer; Mixtral 2-of-8 per token so 3 draft tokens can touch up to 6 experts (~3x data movement) [C: search snippets] — [alphaxiv 2607.12696](https://www.alphaxiv.org/abs/2607.12696), [arXiv 2506.20675](https://arxiv.org/abs/2506.20675)
- Cascade: speculation can slow MoE serving by up to 1.5x; adaptive scheme caps slowdown at 5%, +7-14% throughput over static K [C] — [arXiv 2506.20675](https://arxiv.org/abs/2506.20675)
- MoE-Spec (expert budgeting): 10-30% higher throughput than EAGLE-3 at comparable quality [C] — [arXiv 2602.16052](https://arxiv.org/abs/2602.16052)
- Cost-aware drafting up to 1.62x (July 2026 paper) [C] — [alphaxiv 2607.12696](https://www.alphaxiv.org/abs/2607.12696); EVICT up to 2.35x vs autoregressive, 1.21x avg over EAGLE-3 [C] (source in same search, arXiv ID not captured [U])
- Counterpoint: at moderate batch sizes SD can be more effective for sparse MoE than dense [C] — [arXiv 2505.19645](https://arxiv.org/html/2505.19645v2)
- Network: break-even ~1.15-1.20 tokens accepted/step, measured 1.2-1.3 on WAN — see existing `academic_papers.md` ([arXiv 2602.16760](https://arxiv.org/abs/2602.16760)).

### Inferences
- Plan phase 5 caveat ("MoE verification touches more experts") is confirmed by literature; single-stream, batch-1 consumer use is the worst regime.

### Gaps
- Typical acceptance-rate numbers for small-draft/big-target pairs on consumer hardware not collected.

## Q6. Reasoning/thinking models: overhead, overthinking, parallel-thinking papers, KV/link needs

### Takeaway
All four named papers exist. Overthinking wastes many tokens, which multiplies cost when you run N chains. Parallel-reasoning methods report modest speedups (up to ~1.5-2x) and need either training, engine changes, or (Hogwild!) a shared KV cache, which implies fast memory-level links, not a LAN.

### Cited Findings
- Overthinking: o1-like models used ~1,953% more tokens than conventional models on "2+3"; >85% of cases the first answer is already correct; ICML 2025 [M/C] — [arXiv 2412.21187](https://arxiv.org/abs/2412.21187)
- Parallel-R1 exists: arXiv 2509.07980, Tong Zheng et al., RL with curriculum (SFT on easy problems then RL), math (MATH, AMC23, AIME); +8.4% accuracy over sequential RL baseline; ICLR 2026 poster; code promised at github.com/zhengkid/Parallel-R1 (release status [U]) [M/C] — [arXiv 2509.07980](https://arxiv.org/pdf/2509.07980), [ICLR](https://iclr.cc/virtual/2026/poster/10006733)
- ThreadWeaver exists: arXiv 2512.07843; Qwen3-8B; AIME24 79.9%, 71.9% average over six math benchmarks; up to 1.53x token-latency speedup; trie-based rollout runs on any off-the-shelf autoregressive engine [M] — [arXiv 2512.07843](https://arxiv.org/abs/2512.07843). Secondary comparison: Multiverse-32B 53.8%, Parallel-R1-4B 19.4% on AIME24 [C] — [arxiviq](https://arxiviq.substack.com/p/threadweaver-adaptive-threading-for)
- Hogwild! Inference exists: arXiv 2504.06261 (Rodionov et al., Yandex); several instances of the same LLM share one KV cache using RoPE tricks, no fine-tuning; abstract gives no benchmarks [M] — [arXiv 2504.06261](https://arxiv.org/abs/2504.06261)
- Multiverse exists: arXiv 2506.09991 (Yang, An, Liu, Chen, Chen), NeurIPS 2025; Map/Process/Reduce; 1K-example 3-hour fine-tune; Multiverse-32B AIME24 54%, AIME25 46%; ~2x speedup per secondary summary [M/C] — [arXiv 2506.09991](https://arxiv.org/pdf/2506.09991)
- Parason (arXiv 2608.24658) studies subtask/trial parallelism and builds on ThreadWeaver traces [C: snippet] — [arXiv 2608.24658](https://arxiv.org/pdf/2608.24658)

### Inferences
- ThreadWeaver, Multiverse, Parallel-R1 need trained models and (for speedup) concurrent branches on one GPU's batch; Hogwild! needs shared KV, so it does not map onto LAN-connected nodes. Independent-chain voting (plan phase 3) is the only variant that tolerates a slow network because chains share nothing.
- Token overhead of thinking models multiplies N-chain cost; cap thinking budget on workers.

### Gaps
- No measurement of any of these over a network or without shared KV; Hogwild! quality/speed numbers not read (abstract only). Parallel-R1 repo release status not checked.
