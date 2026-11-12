# Expert caching, activation prediction and routing statistics for offloaded MoE at batch size 1

Target used for applicability judgments: pancake-strata (C++/HIP), Qwen3.8-Flash-Next 125B MoE, 48 layers x 512 experts, top-10 (+1 shared, per secondary sources), IQ2_XS ~1.37 MiB/expert, RX 7900 XT holding ~10,900 of 24,576 experts (44%), misses run on a Ryzen 7 5700G (DDR4) or copied over PCIe 3.0 (~13.4 GB/s). Measured hit rate is 80-93% of routed entries. Existing mechanisms: a static frequency profile, an adaptive tier with decayed counts (<=80 swaps), GPU router lookahead for the verify window, and an opt-in lossy cache-aware bonus (+25% decode, 2.9% of tokens differ).

Research ran on 2026-11-08 with about 25 search/fetch calls. Several numbers come from abstracts or secondary summaries and are marked as such.

Reference arithmetic used below, by the author, not from a source: 480 routed entries per token (48 x 10). Misses per token are 72 at 85%, 48 at 90%, 24 at 95% and 14 at 97%. Going from 85% to 90% removes 33% of CPU expert work, and going to 95% removes 67%. A CPU miss streams about 1.37 MiB from DDR4. At an assumed effective ~30-40 GB/s, 72 misses cost roughly 2.5-3.5 ms of pure bandwidth per token, before counting the verify-window rows. A PCIe 3.0 copy costs ~0.1 ms per expert, so 72 copies would take ~7.7 ms. That is why misses are computed, not copied.

## Q1. Cache replacement policies, Belady bounds, budgets and reuse distance

### Takeaway
The 2026 consensus is that recency-based policies (LRU/LFU) fit MoE poorly at batch 1. Policies that look ahead, using predicted future activations as a "probabilistic Belady", are the best published approach. At 45% residency (matching the target's 44%), SeqMoE reports 96.97% hit vs 88.50% for LRU. A target already at ~85-93% with a frequency profile plus decay sits roughly at the "good classical policy" level. The remaining headroom to ~95-97% comes from forecast-driven eviction and prefetch, not from tweaking LRU/LFU.

### Cited Findings
- **SeqMoE** (arXiv 2609.12978, Sep 2026). Evicts with a "probabilistic Belady" score: an exponentially weighted mean (lambda=0.5) of predicted activation probability over the next c=8 decode steps. Empty slots fill first, and the lowest-scoring used slot is evicted. Hit rate vs the best baseline (LRU), averaged over models: 15% residency 84.56 vs 60.02; 25% 91.72 vs 74.24; 35% 95.09 vs 84.49; 40% 96.11 vs 86.37; 45% 96.97 vs 88.50. A Belady-8 oracle and the forecast-based Belady-F8 are said to track each other closely, but no separate numbers are given. Other baselines are LFU and PrefillHot (hot set from prefill). Average miss rate at 45% residency is 3.25%. Models: Qwen3-30B-A3B (128 experts, top-8), Qwen3.6-35B-A3B (256, top-8), GPT-OSS-120B (128, top-4), DeepSeek-V4-Flash (256, top-6). GPUs: RTX 4090/5090/PRO 6000. Decode reaches 80.22% of fully loaded speed on average at 40-45% residency, vs 51.5% for the next-best system (FreeToken). — [SeqMoE arXiv 2609.12978](https://arxiv.org/html/2609.12978)
- **SpecMD** (arXiv 2602.03921, Feb 2026). Argues that MoE expert access "is not consistent with temporal locality assumptions (e.g. LRU, LFU)" because each pass walks the layers front to back. It proposes **Least-Stale** eviction: experts used in the current forward pass are "current" and are evicted only to avoid OOM. Previously used experts are "stale" and go first, FIFO by layer position. At 5% capacity on OLMoE, collision misses are 1.6-1.9% vs 4.5-12.6% for LRU. Claimed hit rates are "over 88%" at 5% capacity (abstract) and "88-92%" at 1% (intro); these conflict. TTFT falls 10.7-34.7% (~17% on average). Setup: A100 with an artificially limited cache, ~5 GB/s link, OLMoE/Mixtral/Qwen1.5-MoE/Phi-3.5-MoE. — [SpecMD arXiv 2602.03921](https://arxiv.org/html/2602.03921v1)
- **mainline llama.cpp `--moe-cache-mib`** (PR #29887 by am17an, merged Oct 7 2026; multi-GPU in #30112). Uses plain **LRU** over host-resident experts. Only batches of 32 tokens or fewer use the cache, and only misses are uploaded over PCIe. Author benchmarks on Qwen3.8-Flash-Next Q4_0 (65.4 GiB of experts), EPYC 7742, PCIe 4.0: RTX 4090 with an 11,000 MiB cache gives 77% hit, 40.7 vs 25.0 t/s; RTX 5090 with 19,000 MiB gives **89% hit**, 67.8 vs 30.8 t/s (2.2x). Independent results: an AMD R9700 32 GB on ROCm went from 11.0 t/s (no cache) to 14.5 (85% hit at 8 GB) and 27.6 (**94.6% hit at 16 GB**). A multi-GPU GLM 5.3 Flash run got 43.5% hit and a slowdown (33 to 16 t/s). Testers put break-even at roughly 75% or more hit. — [developersdigest.tech summary](https://www.developersdigest.tech/blog/llama-cpp-moe-offload-n-cpu-moe-vs-moe-cache-2026) (secondary; I could not open the PR directly). A release date of Oct 15 / v0.7.0 is claimed by [byteiota](https://byteiota.com/llama-cpp-0-7-0-moe-cache-hits-2-2x-server-port-has-changed/).
- **Fate** (arXiv 2502.12224, 2025, edge). Uses a "shallow-favoring" cache and claims a 99% expert hit rate. That figure comes with cross-layer prefetch and quantized experts on small models, so it is not comparable to a 44%-of-125B cache. Fate also claims up to 4.1x decode vs load-on-demand. — [Fate abstract via aitopics](https://aitopics.org/doc/arxivorg:6494E93C); [arXiv 2502.12224](https://arxiv.org/abs/2502.12224v1)
- **Local routing consistency** (arXiv 2505.16056, 2025). Defines Segment Cache Best Hit Rate (SCH): the oracle hit rate when the next segment's routing is known. Across 20 MoE LLMs it finds that shared experts shrink the combination space and lower local routing consistency. Domain-specialized experts help consistency more than vocabulary-specialized ones. A cache of ~2x the active experts per layer is usually the effective/efficient knee. — [arXiv 2505.16056](https://arxiv.org/abs/2505.16056)
- **Per-layer vs global budget evidence**: MoE-Sieve reports that routing is "globally balanced, locally imbalanced", with skew much stronger per layer than globally and substantially higher from mid-layers on than in early layers. — [MoE-Sieve arXiv 2603.24044](https://arxiv.org/html/2603.24044). Fate's "shallow-favoring" design also allocates by depth. — [Fate](https://aitopics.org/doc/arxivorg:6494E93C)
- **AdapMoE** (arXiv 2408.10284, 2024, older) allocates cache per layer by sensitivity. It also states that "expert similarity between successive tokens is not significant." — [AdapMoE](https://arxiv.org/pdf/2408.10284)
- **HOBBIT** (arXiv 2411.01433, 2024, older). Relies on temporal locality for replacement. On a miss it loads a lower-precision copy of the expert instead (lossy), with up to 9.93x decode speedup on edge hardware. — [HOBBIT](https://arxiv.org/html/2411.01433v2)
- **Bevy** (arXiv 2611.04794, Nov 2026). Finds that prefetching can hurt when a GPU expert cache exists: the larger the cache's coverage, the less precise prediction of the remaining (uncached) experts becomes. It overlaps concurrent requests instead and reports up to 3.46x llama.cpp decode throughput (multi-request, so it does not apply at batch 1). — [Bevy abstract](https://arxiv.org/abs/2611.04794)

### Inferences
- The target already uses a frequency profile plus decayed adaptive swaps, so it is essentially an LFU-with-decay policy. At 44% residency, published LFU/LRU-class numbers (~85-89%) match the measured 80-93%. The documented step up to ~95-97% at the same residency (SeqMoE) comes from forecasting several tokens ahead and evicting by predicted future use.
- Caveat: SeqMoE's models have 128-256 experts and top-4 to 8. Qwen3.8-Flash-Next has 512 experts, top-10 and a shared expert, which per 2505.16056 tends to lower local consistency. Expect less headroom than SeqMoE shows; a realistic hope is ~90-94% (author's estimate, not measured).
- Per-layer budgets are probably worth trying, since skew differs by depth. The target's global rank-by-frequency profile implicitly gives more slots to skewed layers, which is roughly right. An explicit marginal-hit-per-slot allocation per layer (greedy on profile counts) is cheap to compute offline and exact. Expected effect is small to moderate, perhaps +0.5-2 points (estimate).
- Least-Stale's insight, never evict an expert needed in the current pass, matters mainly for small caches. With 44% residency and a swap budget of 80, its gain on the target is probably negligible.
- The mainline llama.cpp cache (LRU, demand upload over PCIe) is strictly simpler than what the engine already has. Its 89-94.6% hit numbers on this same model family with 16-19 GB caches set a useful bar: the target's 85% average is at or slightly below plain LRU with a similar-sized cache. Caveat: those caches held Q4_0 experts, roughly 2x the bytes of IQ2_XS, so expert counts are not equal.

### Gaps
- No primary-source Belady-optimal hit-rate number for a 512-expert/top-10 model at 44% residency. The engine can compute it offline from its own routing traces (cheap, exact), which would give the true ceiling.
- I could not open PR #29887 itself (the fetch timed out). Details come from a secondary article.

## Q2. Expert prediction: cross-layer, temporal, prompt-level, learned vs training-free

### Takeaway
Predicting the next layer from the current hidden state, by running layer l+1's router on layer l's residual, is training-free and reaches ~85-90% recall on modern MoEs. The target already does this ("router lookahead"). Accuracy for one layer ahead is already high, so the value of prediction for caching lies in multi-token horizons (SeqMoE: 8 steps, with only -3.6 points of recall). Short-range temporal reuse between consecutive tokens is weak or disputed.

### Cited Findings
- **SeqMoE predictor**: a Mamba2 seq2seq model over per-step routing vectors (layers x experts merged), trained with ranking + KL loss, teacher forcing then scheduled sampling. Two predictors fire, at layer L/2 and L. One-step top-(k+3) recall: Qwen3-30B-A3B 90.90%, Qwen3.6-35B-A3B 88.39%, GPT-OSS-120B 90.76%, DeepSeek-V4-Flash 85.81%. 8-step recursive rollout loses only 3.58 points on average. Overhead is ~3.46% of runtime and 112-228 MB of predictor memory. Prefill prefetch uses an offline popularity profile. — [SeqMoE](https://arxiv.org/html/2609.12978)
- **Speculating Experts** (arXiv 2603.19289, Mar 2026). Uses a "quasi-hidden state" q_l = LN_{l+1}(d_l + r_l), the post-attention residual plus a gate-weighted sum of the selected experts' default vectors, to predict layer l+1 without running layer l's experts. On Qwen3-30B-A3B, recall@k averages ~90% beyond the first two layers. A small trained estimator reaches ~90% and improves early layers by up to ~25%. GPT-OSS-120B 83%, GPT-OSS-20B 88%. If the speculated experts are executed instead of the true ones (lossy), Qwen3-30B-A3B average accuracy drops from 0.817 to 0.641 (Router-PF) / 0.766 (Hybrid), while GPT-OSS is nearly unaffected. TPOT gain from prefetching is 5-14%. — [arXiv 2603.19289](https://arxiv.org/html/2603.19289v1)
- **Fate** (2025). Feeds the adjacent layer's gate input to the next gate and reports 97.15% prefetch accuracy (edge models). — [Fate review](https://www.themoonlight.io/review/accurate-expert-predictions-in-moe-inference-via-cross-layer-gate)
- **MoE-Beyond** (arXiv 2508.17137, Aug 2025). A small transformer trained on 66M activation traces from DeepSeek-V2-Lite. In simulation with a cache holding 10% of experts it raises hit rate from 17% to 72%, which is +55 points vs the MoE-Infinity heuristic. On unseen WebGLM-QA prompts it reports 97.5% accuracy and 86.6% F1. Limits: batch 1 only, retrain per model, one-layer lookahead. Code: github.com/ngavhane/moe-beyond. — [MoE-Beyond](https://arxiv.org/html/2508.17137v1)
- **ExpertFlow** (arXiv 2410.17954 v2, DAC'26). A transformer routing-path predictor estimates expert use for all layers in one pass. It is combined with token scheduling and a predictive cache that corrects mispredictions at runtime. Claims up to 93.72% GPU memory reduction and up to 10x throughput vs offloading baselines. A separate summary cites up to 95% prediction accuracy. — [ExpertFlow abstract](https://arxiv.org/abs/2410.17954v2); [search summary of v2](https://arxiv.org/html/2410.17954v2)
- **FineMoE / fMoE** (arXiv 2502.05370, EuroSys'26). Keeps an "expert map" of iteration-level gate probability distributions plus prompt semantic embeddings, and searches for similar past requests to guide prefetch/offload. Reports -47% latency and +39% hit rate vs prior systems on a 6-GPU testbed. — [FineMoE arXiv 2502.05370](https://arxiv.org/html/2502.05370v2)
- **Pre-Attention Expert Prediction** (arXiv 2511.10676, Nov 2025) predicts and prefetches before attention. — [arXiv 2511.10676](https://arxiv.org/html/2511.10676v1)
- **LayerScope/PreScope** (arXiv 2509.23638) uses a learned layer-aware predictor (LLaPor) and a scheduler (PreSched) that resolves prefetch vs on-demand bandwidth conflicts across all layers. It targets multi-batch inference on legacy servers. The header reads "Conference'26"; I could not confirm ICS as the venue. — [LayerScope](https://arxiv.org/pdf/2509.23638)
- **ProMoE** (arXiv 2410.22134, 2024). Proactive caching from intermediate results: 2.20x prefill and 2.07x decode on average (up to 5.02x). — [ProMoE](https://arxiv.org/html/2410.22134v2) (speedups via secondary summary)
- **Temporal locality between consecutive tokens is disputed**. AdapMoE says successive-token similarity is "not significant" ([AdapMoE](https://arxiv.org/pdf/2408.10284)), and SpecMD says LRU/LFU assumptions do not hold ([SpecMD](https://arxiv.org/html/2602.03921v1)). HOBBIT assumes temporal locality ([HOBBIT](https://arxiv.org/html/2411.01433v2)). ReMoE shows reuse can be trained in, with a +26-27% overlap gain, but that requires training. — [search summary](https://arxiv.org/pdf/2605.27081)
- **vLLM RFC #38256** (Mar 2026) proposes a GPU expert cache with LFRU eviction and cross-layer prediction and lists 97-100% hit for temporal prediction. This is a proposal and the figure is unverified. — [vLLM issue 38256](https://github.com/vllm-project/vllm/issues/38256)

### Inferences
- The target's router lookahead covers the one-layer cross-layer predictor. Published numbers suggest little left to gain there for latency hiding. The untapped part is using a multi-token forecast for **eviction/admission decisions** (SeqMoE-style). Without training, a cheap exact proxy is to score experts by an EWMA of router probability mass, not just top-10 hit counts. Router softmax mass over all 512 experts carries "near-miss" information about what the next tokens will want, much like the Temporal Router in 2609.04895, which sums native and temporal distributions.
- Training a Mamba2/transformer predictor per model (SeqMoE, MoE-Beyond) is medium-high effort: collect traces, train, then add a HIP inference path. It is exact, since it only affects placement. The published residency-matched gain is about +8.5 points over LRU at 45%; against the target's stronger LFU-decay baseline, perhaps +3-7 points (estimate).
- FineMoE-style prompt-semantic lookup (match a new request to similar past requests and preload their expert map) fits an agent workload with recurring prompt families such as code-edit vs chat. It is exact and medium effort. It also directly addresses the measured weak spot: lower hit rates on code-edit prompts.

### Gaps
- No published prediction accuracy numbers for 512-expert/top-10 models (Qwen3-Next class). All numbers come from 128-256-expert models.
- I found no head-to-head numbers between "next layer's router applied to the current hidden state" and the quasi-hidden-state method.

## Q3. Cache-aware / prefetch-steered routing and quality impact

### Takeaway
Altering routing toward cached experts is the biggest published speed lever (~2x on small models). It is inherently lossy, though measured benchmark deltas are within noise for some models. Prefetch-Steered Routing (PSR) makes it deterministic, independent of transfer timing. Training-based variants keep top-K semantics at inference but change weights, giving -0.4 to -3.3 points on Qwen3.

### Cited Findings
- **Prefetch-Steered Routing** (arXiv 2610.34721, Oct 26 2026, Siming Huang). Routing is conditioned on a deterministic *logical* cache: per layer and token, up to n experts predicted by the next layer's router are inserted. Physical transfers then affect only latency, not outputs, so results are reproducible. Results:
  - OLMoE-1B-7B on an RTX 4090 at 50%/25% residency: 2.2-2.3x faster than exact routing with LRU, and 25-30% faster than the "Cache-Prior" operating point. Mean benchmark deltas are +0.1/+0.5 points (95% lower bounds -0.8/-0.3). The advantage over Cache-Prior reverses on HumanEval at 50% residency in greedy generation.
  - Moonlight-16B-A3B at 37.5% residency: 1.9x vs exact LRU.
  - Qwen3.6-35B-A3B: quality advantage over Cache-Prior persists but the speed advantage does not, because copies are cheap relative to compute there.
  - Distillation-trained gates did not help.
  - With a block drafter on Qwen3.6, 2.1-2.9x faster than autoregressive configs in the same memory.
  — [arXiv 2610.34721](https://arxiv.org/abs/2610.34721)
- **Cache-Aware Joint Router Adaptation** (arXiv 2609.04895, Sep 2026). Native top-K still selects which experts execute; auxiliary Temporal and Spatio-Temporal routers decide residency. However, the backbone is post-trained jointly with a cache-coverage loss, so outputs differ from the base model.
  - Qwen3-30B-A3B, Temporal Router: hit rate +10.5 to +33.3 points over the strongest classical baseline, and load drops from 1353 to 974 MB/token on GSM8K. Accuracy: GSM8K unchanged, MATH -0.40, CSQA -1.23.
  - Spatio-Temporal Router: -2.0/-0.6/-3.3 points.
  - No wall-clock numbers. Training cost is 4 epochs of full post-training.
  - The paper warns that raw hit rate can mislead: "Temporally Extended MoE" hits 100% but is worse overall.
  — [arXiv 2609.04895](https://arxiv.org/html/2609.04895v1)
- **Speculating Experts** executing predicted experts instead of true ones: large loss on Qwen3-30B-A3B (0.817 to 0.641 average), small on GPT-OSS. — [arXiv 2603.19289](https://arxiv.org/html/2603.19289v1)
- **HOBBIT** low-precision substitutes for missed experts are a lossy alternative to rerouting. — [HOBBIT](https://arxiv.org/html/2411.01433v2)
- A Research Square preprint (rs-11268552) studies expert substitution on Qwen3-Next-80B-A3B with a quarter of experts in GPU memory, including whether the gate weight of a substituted expert can guide cache policy. Only the title/snippet was accessible, with no numbers. — [Research Square rs-11268552](https://www.researchsquare.com/article/rs-11268552)

### Inferences
- The target's opt-in bonus (+25% decode, 2.9% of tokens differ) is the same family as Cache-Prior. PSR's contribution is determinism plus a small extra speed margin (25-30% over Cache-Prior on OLMoE). Because the engine computes misses on the CPU synchronously and its adaptive tier is asynchronous, the bonus's output may already depend on swap timing. Making it condition on a logical cache state updated at deterministic points (e.g. verify-window boundaries) would give reproducibility at low effort.
- Quality is model-dependent. Qwen3-family models were the most sensitive in 2603.19289, so a per-workload eval on the target (code-edit especially) is needed before making the bonus a default.
- A safer middle ground is suggested by the substitution study's framing (inference): reroute only when the dropped expert's gate weight is small, e.g. the 9th-10th ranked experts with low softmax mass. This limits divergence while still removing the long-tail misses.

### Gaps
- No quality numbers for cache-aware routing on any 512-expert model.
- PSR's numbers are at batch 1 on an RTX 4090, but the target's misses are computed on the CPU rather than copied, which changes the cost model. The speedup may differ.

## Q4. Expert popularity statistics: skew, stability, static vs adaptive

### Takeaway
Modern fine-grained MoEs are trained toward global balance, so popularity is only mildly skewed globally. It is markedly skewed per layer and per domain, increasingly so from mid-depth. A static global profile therefore gets you most of the way at high residency (44%), but domain shifts such as code-edit vs chat leave several points on the table. That argues for per-domain profiles or prompt-conditioned preloading on top of the adaptive tier.

### Cited Findings
- MoE-Sieve (arXiv 2603.24044): "globally balanced, locally imbalanced". A 25% expert budget chosen per layer captures 37-53% of per-layer activations, vs 25% under uniform routing. Skew rises from mid-layers on. The same domain-driven structure holds for DeepSeek and Qwen, with slightly weaker within-domain clustering in Qwen. — [MoE-Sieve](https://arxiv.org/html/2603.24044)
- DBES (arXiv 2605.18498): Qwen3-235B has a markedly lower effective-routing metric R_eff (0.24-0.31) than Qwen3-30B, DeepSeek and GLM (0.45-0.55). — [DBES](https://arxiv.org/pdf/2605.18498) (definition of R_eff not verified)
- A 2025 trace study (arXiv 2510.05497) profiles DeepSeek-V3 and Qwen3-235B over 24,000 requests across tasks, topics and languages. The skew numbers were not visible in the snippet. — [arXiv 2510.05497](https://arxiv.org/pdf/2510.05497)
- IPDPS'26 SMoE notes that "expert activation patterns tend to be increasingly uniform" in modern MoE LLMs, which reduces the value of hot-only caching. — [IPDPS26 SMoE](https://zeyiwen.github.io/papers/ipdps26-smoe.pdf)
- llama.cpp issue #20757 (a proposal) asserts that ~15-20% of experts handle ~80% of tokens. — [llama.cpp #20757](https://github.com/ggml-org/llama.cpp/issues/20757) (anecdotal)
- Shared experts reduce local routing consistency; domain-specialized experts increase it. — [arXiv 2505.16056](https://arxiv.org/abs/2505.16056)
- The Cache-Aware Router paper shows hit-rate gains vary hugely by task: +10 points on GSM8K/MATH vs +33 on CSQA. That illustrates the task dependence of the "right" resident set. — [arXiv 2609.04895](https://arxiv.org/html/2609.04895v1)
- SeqMoE finds that prefill activations reflect "statistical popularity" and therefore seeds prefill from an offline frequency profile, while decode uses forecasts. — [SeqMoE](https://arxiv.org/html/2609.12978)

### Inferences
- The target's measured spread (80-93%, worst on code-edit) is consistent with domain-dependent per-layer skew. Concretely: keep two or three static profiles (chat, code-edit, tool-use) and pick one from the system prompt or first prefill chunk. This is exact and low effort, with an estimated gain of +2-5 points on the weak domain (estimate). The adaptive tier's 80-swap budget (~110 MB) then has less to correct.
- The prefill of each request is itself a high-quality domain sample: 480 x (prompt tokens) routed entries. Using prefill routing counts to seed a large one-time swap before decode starts (SeqMoE's PrefillHot idea, extended) is exact and cheap. PCIe 3.0 moves ~1,000 experts (~1.4 GB) in ~0.1 s, which is negligible against agent prompt prefill times.

### Gaps
- No published per-expert popularity distribution (e.g. Gini or top-x% coverage) for a 512-expert Qwen3-Next/3.8-class model. The engine's own profile data is the best source and should be reported as coverage curves per layer.
- No cross-domain stability measurement (overlap of top sets between code and chat) for Qwen3/DeepSeek-V3 in the sources found.

## Q5. llama.cpp expert caches, FATE, ongunm/llama-moe-cache

### Takeaway
Mainline llama.cpp now has a simple LRU GPU cache for host-resident experts (`--moe-cache-mib`, PR #29887, plus multi-GPU in #30112). It reaches 77-95% hit with 8-19 GB caches on this same model family and up to 2.2x decode. It has no prediction and no frequency profile, and it uploads misses over PCIe instead of computing on the CPU. It offers nothing algorithmically new for the target. "FATE" is the Fate paper (cross-layer gate prefetch with a shallow-favoring cache), not a llama.cpp extension. I found no trace of "ongunm/llama-moe-cache" or its 99.5% claim.

### Cited Findings
- PR #29887 ports the expert cache from the qvac-fabric fork. Policy is LRU, the cache applies only to batches of 32 tokens or fewer, misses are uploaded, and `--fit` counts the cache. Benchmarks are in Q1; a ROCm R9700 reached 94.6% hit at 16 GB. A 2x 4090 config lost prompt-processing speed (0.73x). One unconfirmed report of a degenerate reasoning trace exists. — [developersdigest](https://www.developersdigest.tech/blog/llama-cpp-moe-offload-n-cpu-moe-vs-moe-cache-2026)
- Forks:
  - ParmesanParty/llama.cpp pins the top-K most-activated experts per layer (qwen35moe only). — [ParmesanParty](https://github.com/ParmesanParty/llama.cpp)
  - The neurall fork builds on #27861 with a live-usage cache. — [neurall releases](https://github.com/neurall/llama.cpp/releases)
  - The GenerelSchwerz fork has N slots per expert tensor. — [GenerelSchwerz wiki](https://github.com/GenerelSchwerz/llama.cpp/wiki/CUDA-MoE-Expert-Cache)
  - The adrianhoehne fork has a "hot cache" with profiling (`--moe-layer-perf-out`, a `/moe-layer-perf` endpoint) and a Qwen3-Coder-Next doc. — [adrianhoehne hot cache](https://adrianhoehne.github.io/llama.cpp/docs/moe-hot-cache/qwen3-coder-next-hot-cache-implementation.html)
  - Discussion #29429 is titled "moe hot cache 2-4 speedup"; content not retrieved (timeout). — [discussion 29429](https://github.com/ggml-org/llama.cpp/discussions/29429)
- Fate (arXiv 2502.12224): cross-layer gate prefetch at 97.15% accuracy, shallow-favoring cache at 99% hit, quantized cache/IO, up to 4.1x decode vs load-on-demand on edge hardware. Code is reportedly available. No llama.cpp integration was found. — [Fate](https://arxiv.org/abs/2502.12224v1); [review](https://www.themoonlight.io/review/accurate-expert-predictions-in-moe-inference-via-cross-layer-gate)
- "ongunm/llama-moe-cache": the GitHub fetch timed out, and a web search found no matching repository. The developersdigest article does not mention it. — [search result context](https://www.developersdigest.tech/blog/llama-cpp-moe-offload-n-cpu-moe-vs-moe-cache-2026)

### Inferences
- Any 99%+ hit-rate claim, whether from Fate or the unverified ongunm repo, should be checked for: (a) the share of experts resident (often >50% or small models), (b) per-token vs per-unique-expert counting, and (c) whether prefetched-then-used experts count as hits. The target's 85% is measured per routed entry, which is the strict definition.
- Applicability: the llama.cpp cache supplies a ready reference implementation for an AMD/HIP backend (ggml), and a validation point that ~90-95% is reachable on Qwen3.8-Flash-Next with 16-19 GB caches.

### Gaps
- ongunm/llama-moe-cache could not be verified. Its 99.5% claim remains unsupported.
- The PR's actual code and review thread were not read (fetch issues). Whether it has any admission filter or frequency component is unconfirmed beyond "LRU".

## Q6. Lossless compression to fit more experts in VRAM

### Takeaway
Lossless entropy coding works well on BF16: DFloat11 compresses exponents with Huffman coding, ~30% smaller with bit-identical output. For 2-bit codebook formats like IQ2_XS the published evidence is thin, and the expected gain is small. The bits are already lattice/codebook indices designed to be dense. GPU decode of variable-length codes costs bandwidth-bound kernels real throughput.

### Cited Findings
- **DFloat11** (arXiv 2504.11651): Huffman coding of BF16 exponents, ~70% of original size, bit-identical outputs, custom GPU decode kernel. — [DFloat11](https://arxiv.org/html/2504.11651v2)
- **Exponent-concentration study** (arXiv 2510.02676): explores the lossless frontier for GenAI weights. — [arXiv 2510.02676](https://arxiv.org/html/2510.02676v1)
- **ZipServ** (2026, aggregator) argues fixed-length codes matter more than maximal entropy efficiency for GPU decode. It puts the BF16 lossless ceiling at ~1.51x. — [emergentmind ZipServ](https://www.emergentmind.com/topics/zipserv) (secondary)
- **EntroLLM** (arXiv 2505.02380): Huffman on quantized integer weights for edge storage, with parallel decode. — [EntroLLM](https://arxiv.org/html/2505.02380v4)
- **MLX issue #3043** (proposal) claims quantized weights have entropy well below their bit width (~2.17 bits for 4-bit), suggesting rANS with per-row streams. Not a measured result for 2-bit codebooks. — [MLX issue 3043](https://upd.dev/ml-explore/mlx/issues/3043)
- **Statistically-lossless quantization** (arXiv 2605.02404) exists as an adjacent direction. — [arXiv 2605.02404](https://arxiv.org/pdf/2605.02404)

### Inferences
- IQ2_XS stores 2.31 bpw as E8-lattice grid indices plus scales. Those indices are chosen to use the code space, so their entropy is likely close to the stored width. Lossless gains are probably ~0-10% (author's estimate, unmeasured). At 10%, that is ~1,000 more experts resident (44% to ~48.5% residency), roughly +1.5-3 hit points on SeqMoE's residency curve. The cost is a decode step inside every matvec on a bandwidth-bound decode path. Measure the empirical byte entropy of the expert blobs (a cheap offline `zstd -19` or order-0 entropy count) before investing.
- A higher-yield "compression" for the same goal is lossy: keep tail experts in an even smaller format (HOBBIT-style mixed precision, e.g. IQ1 for rarely used experts). That changes outputs and belongs in a separate quality evaluation.

### Gaps
- No measured entropy or compression ratio for IQ2_XS/IQ-family GGUF tensors found. No GPU rANS/Huffman decode throughput numbers for 2-bit weights on RDNA3.

## Summary catalogue (applicability to the target)

All "expected gain" entries are author estimates unless they are quoted figures.

| Technique | Source | Exact? | Effort | Expected hit gain on target | Notes |
|---|---|---|---|---|---|
| Offline Belady replay on own traces (ceiling) | SeqMoE/2505.16056 concept | n/a | Low | Diagnostic | Tells you if 95% is reachable at 44% |
| Per-domain static profiles + selection from prompt | MoE-Sieve, FineMoE idea | Yes | Low | +2-5 pts on code-edit | Addresses the weakest domain |
| Prefill-routing-seeded bulk swap before decode | SeqMoE PrefillHot ext. | Yes | Low | +1-4 pts | ~0.1 s per 1k experts over PCIe 3 |
| Router-probability-mass EWMA scoring (not top-k counts) | 2609.04895 Temporal Router idea (training-free variant) | Yes | Low | +1-3 pts | Uses near-miss info |
| Per-layer marginal-gain budget allocation | MoE-Sieve, Fate, AdapMoE | Yes | Low | +0.5-2 pts | Offline greedy |
| Learned multi-step forecaster + probabilistic-Belady eviction | SeqMoE (+8.5 pts vs LRU at 45%) | Yes | High | +3-7 pts | Mamba2, 3.5% overhead |
| Prompt-similarity expert-map lookup | FineMoE (EuroSys'26) | Yes | Medium | +1-4 pts | Good fit for agent workloads |
| Deterministic PSR for the cache bonus | arXiv 2610.34721 | No (small deltas) | Low-Med | Speed +25-30% over Cache-Prior (OLMoE) | Reproducibility win |
| Joint router post-training | arXiv 2609.04895 | No (-0.4 to -3.3 on Qwen3) | Very high | +10-33 pts (small models) | Impractical for 125B locally |
| Lossless entropy coding of IQ2_XS | DFloat11/EntroLLM concept | Yes | High | +0-3 pts | Measure entropy first |

Value of hit-rate gains (author arithmetic, with the CPU-miss path as the bottleneck): 85% to 90% removes one third of CPU expert work per token, and 85% to 95% removes two thirds. If CPU expert compute is ~60-70% of decode time at 85%, then 90% would give roughly 1.25-1.3x and 95% roughly 1.7-1.9x decode. This rough Amdahl estimate should be checked against engine profiles.
