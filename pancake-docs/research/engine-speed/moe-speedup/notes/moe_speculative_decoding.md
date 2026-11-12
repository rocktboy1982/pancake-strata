# Speculative decoding for offloaded MoE (batch 1): techniques, mechanisms, gains, applicability to pancake-strata

Research date: 2026-11-08. Web research only, about 26 tool calls. All arXiv IDs the brief named were checked and exist. Most numbers come from the HTML full text where it was fetched (DraftExpert, EcoSpec, EVICT, AcceptMoE, Limits of Speculation); the rest come from abstracts only, as noted.

Target used to judge applicability: pancake-strata (C++/HIP), Qwen3.8-Flash-Next 125B MoE (512 experts top-10, 48 layers, hybrid GDN + sparse attention), native MTP chain drafter (verify windows of 1-8 rows, about 4 typical, acceptance about 70-90%), suffix/n-gram and DFlash2 drafters, and a cost-aware window policy. The bottleneck is CPU compute of experts that miss the 20 GB GPU cache (80-93% hit). Offline replay of 4-row windows: 23% of CPU-missed experts are needed only by the last row, 47% only by the last two rows.

## Q1. Expert-union / expansion-aware drafting: mechanisms, stopping rules, predictors, numbers

### Takeaway
A cluster of 2025-2026 papers agrees that for MoE targets, verification cost scales with the number of unique experts in the window, not the number of tokens. All the lossless winners do the same thing: before verification, they cut or reshape the draft by maximizing expected accepted tokens divided by expected cost, with cost measured as expert-union growth or profiled latency. Measured gains over a plain EAGLE-3/MTP baseline are modest, about 1.05-1.36x, when experts are GPU-resident. They are larger, about 1.3-2x, under CPU/Flash offload, which is the regime that matches the target.

### Cited Findings
**DraftExpert (arXiv 2607.24434, Jul 2026, single author D. Han). Lossless, self-speculative, offload-specific.**
- Trains one small accelerator-resident "draft expert" per layer by self-distilling from the frozen target, using residual, logit and router-agreement (KL) losses. The drafter is the shared expert + top-1 routed expert + the draft expert, so its footprint is fixed. Target verification is exact. — [arXiv abs](https://arxiv.org/abs/2607.24434)
- Stopping rule ("confidence–expansion truncation"): the predicted block expert set is Û_i = ∪_{j≤i} ∪_l Ê_{l,j}, where Ê comes from the draft router. The marginal expansion is Δ_i = |Û_i| − |Û_{i−1}|. Drafting stops before token i when **q_i < τ AND Δ_i > B_i**, where q_i is draft confidence and B_i is a fixed per-layer expert budget or one derived from the current cache. No numeric τ/B is given. — [arXiv HTML](https://arxiv.org/html/2607.24434v1)
- Prefetch: once a token is kept, non-blocking prefetches start for its predicted experts that are not cached or in flight. Verification waits only on unfinished transfers, and misses load on demand. Prefetch hit/waste is 88/12% (DeepSeek-V2-Lite) and 86/14% (Moonlight). — [arXiv HTML](https://arxiv.org/html/2607.24434v1)
- Hardware: RTX 4090 24 GB, BF16 PyTorch, experts in host RAM (CPU→GPU); and a Snapdragon 8 Elite Hexagon NPU, Q4_0 llama.cpp (Flash→NPU). Decode tok/s, AR → DraftExpert: DS-V2-Lite CG 2.19→2.99 (1.36x), MN 10.18→15.47 (1.52x); Moonlight CG 1.94→2.50 (1.29x), MN 8.50→13.69 (1.61x); average 1.45x. A naive "top-r experts" self-draft is slower than AR (1.19 / 1.01 tok/s on CG). — [arXiv HTML](https://arxiv.org/html/2607.24434v1)
- Ablation, cost normalized to one AR step: fixed K=4 costs 2.35x (DS, CG); confidence+expansion truncation cuts this to 1.48x, and adding prefetch to 1.25x. On Moonlight the same steps go 2.55x → 1.52x → 1.28x. Acceptance at K=4 is 85% / 83%, slightly below the abstract's 84–87%. — [arXiv HTML](https://arxiv.org/html/2607.24434v1)

**EcoSpec, "Less Experts, Faster Decoding" (arXiv 2607.12696, Jul 2026). Lossless, tree-selection reshaping.**
- Score: S(t_i) = P(t_i) / (ΔCost(t_i|B) + ε). P is the cumulative path draft probability. ΔCost = |E_traj(t_i) \ B| is the number of predicted experts not already in the dynamic buffer B. Selected nodes add their path's experts to B, so nodes that share experts get cheaper (it rewards "expert reuse", the opposite of "expert scattering"). There is no λ; the score is a ratio. — [arXiv HTML](https://arxiv.org/html/2607.12696v1)
- Predictor: a small decoder LM (DeepSeek-R1-Distill-Qwen-1.5B or Qwen3-0.6B) maps target token embeddings to per-layer routing logits. Top-K accuracy is 80% (DeepSeek-V3.1), 82% (Qwen3-235B) and 93% (GPT-OSS-120B). Tree: 3 draft steps, top-2 per step. Verification rule unchanged, so output is lossless. — [arXiv HTML](https://arxiv.org/html/2607.12696v1)
- Results on 8xH200, HF Transformers prototype, greedy, average of 7 benchmarks. Qwen3-235B-A22B: EAGLE-3 1.22x → 1.36x, active experts 23.7 → 20.5. GPT-OSS-120B: 1.14x → 1.31x, experts 11.6 → 10.6. DeepSeek-V3.1 with MTP: 1.10x → 1.15x, experts 31.4 → 31.2. Best single benchmark is 1.62x (Qwen3-235B, MT-Bench). Experts are GPU-resident (no offload). — [arXiv HTML](https://arxiv.org/html/2607.12696v1)

**EVICT (arXiv 2605.00342, May 2026, rev. Sep 2026). Lossless, training-free, hyperparameter-free tree truncation in SGLang.**
- Node benefit is the product of draft probabilities along the root path, and the expected accepted length is the sum of the kept nodes' scores. The cost C(k) is the end-to-end step latency for verifying the top-k nodes, profiled offline into a lookup table. The rule is **k* = argmax_k Ê[A(T_k)] / C(k)**, computed online from prefix sums. Structurally this is the same as pancake-strata's existing cost-aware window policy. — [arXiv HTML](https://arxiv.org/html/2605.00342v2)
- A100 80 GB, SGLang. Approximate speedups over EAGLE-3 from the tables: Qwen3-30B-A3B about 1.15x (the text claims 1.21x), Ling-flash-2.0 about 1.22–1.24x, Qwen3-235B-A22B about 1.25x, dense Hunyuan-7B about 1.08x. On Qwen3-30B-A3B, unique activated experts fall 32.5% and verify latency 26.6%. The "up to 2.35x over AR" headline is not visible in the tables (max seen about 1.85x). The appendix models unique experts as N_uniq(t) = M[1 − (1 − K/M)^t]. — [arXiv HTML](https://arxiv.org/html/2605.00342v2)

**The Limits of Speculation (arXiv 2609.22156, Aug 2026). Analysis/oracle only, no deployable system.**
- Verify cost fits T ≈ α·Ū + β, where Ū is the layer-averaged count of unique experts. Qwen3-Coder-30B-A3B (128 experts, top-8) on an A100 at B=1 in a vLLM fork: α ≈ 0.08 ms/expert for m ≤ 4 rows, jumping to about 0.265 ms/expert for m ≥ 5 (a kernel/shape regime change). Draft cost is 0.459K+0.073 ms. In-sample R² is 0.967–0.999. — [arXiv HTML](https://arxiv.org/html/2609.22156v1)
- An offline SSP oracle speedup plateaus at about 2.34x for L_max ≥ 6. Accepted length saturates near 2.1, and the oracle caps draft length near 2.8 even at L_max = 12. In "Delta space", rejected budget choices lie on one side of a line through the origin. The necessary condition is that ΔE[Cost]/ΔE[accepted] stays below a constant. The suggested future online heuristic: stop drafting when this marginal ratio crosses the threshold. — [arXiv HTML](https://arxiv.org/html/2609.22156v1)
- Mentions "MoE-Spec", which caps the per-layer expert budget, as related work (not fetched). — [arXiv HTML](https://arxiv.org/html/2609.22156v1)

**AcceptMoE (arXiv 2608.02989, Aug 2026). LOSSY verifier-side expert-set restriction. Closest published relative of the "skip tail-row CPU experts" idea.**
- Commitment probability per draft position d is p̂_d = c_d/m_d, the fraction of proposed nodes at depth d that end up on the accepted prefix, estimated offline. Token weight is α_t = p̂_d^β · N_d^(−1/2) with β = 0.5. Expert utility is u_e = Σ_t α_t r_{t,e} 1[e ∈ topk_t]. The root token's natural top-k (the "anchor") is always kept. The extra set size is n = ⌈exp(H(q))⌉, the effective rank of normalized residual utility. Eligibility is enforced with a −∞ mask on router logits before top-k, so excluded tokens are **rerouted** to the best eligible experts (this is a distribution change, not row dropping). — [arXiv HTML](https://arxiv.org/html/2608.02989)
- Offload mode: a 48-slot per-layer pool on an RTX 5090. Residency-aware pruning drops non-resident members of S in increasing-utility order, up to a rerouting budget. Results at SGLang B=1: 1.290x over standard SD with full residency (RTX PRO 6000) and 2.06x under offload. H2D traffic falls 73.6–77.1%. Mean accuracy is −0.27 pp vs standard SD; the worst pair is −1.22 pp (HumanEval, Qwen3-30B-A3B-Instruct). Models: Qwen3-30B-A3B-Instruct, Qwen3-Coder-30B-A3B, GPT-OSS-120B, each with EAGLE-3. — [arXiv HTML](https://arxiv.org/html/2608.02989)
- It explicitly calls EVICT a complementary draft-side method; there is no head-to-head comparison. — [arXiv HTML](https://arxiv.org/html/2608.02989)

**Offload-system papers that use the draft as a lookahead for prefetch and scheduling (abstract-level details only):**
- Specter (arXiv 2511.14102; originally listed as "MoE-SpeQ", v3 Oct 2026, ASPLOS 2027): a quantized on-device draft gives multi-step lookahead of expert use to schedule PCIe transfers. Single-request TPOT is 1.50x average and 1.88x max over the prior SOTA offload system. Exactness is not stated in the abstract. — [arXiv abs](https://arxiv.org/abs/2511.14102)
- SpecMoEOff (arXiv 2508.21706, 2025): SD enlarges per-expert work. It uses roofline-guided CPU/GPU orchestration, a CPU chunked-attention verification kernel, and an SD hyperparameter auto-tuner. Up to 2.5x decode throughput over SOTA offloading. — [arXiv abs](https://arxiv.org/abs/2508.21706)
- SP-MoE (arXiv 2510.10302, 2025): speculative expert prefetching from draft/target correspondence, a "cutoff-layer" policy that limits per-layer prefetch depth using a latency model, and async batched I/O. 1.07–3.5x TPOT over SOTA. — [arXiv abs](https://arxiv.org/abs/2510.10302)
- MoE-SpAc (arXiv 2603.09983, rev. Oct 2026): a Speculative Utility Estimator for expert demand, online integer-optimization CPU/GPU workload balancing, and joint prefetch/eviction in one utility space. +42% TPS over a SOTA SD baseline and 4.04x average over standard baselines, on edge hardware. — [arXiv abs](https://arxiv.org/abs/2603.09983)
- SpecMoE (arXiv 2604.10152, DAC 2026 extended): self-assisted SD for CPU-offloaded MoE, training-free, up to 4.30x throughput. It targets larger batches. — [arXiv abs](https://arxiv.org/abs/2604.10152)
- S2-MoE (arXiv 2608.15018, Aug 2026): self-speculative for edge, implemented in llama.cpp. It combines "routing-aware adaptive speculative expansion" (utility = acceptance likelihood vs verify cost), "reuse-aware expert gating", and a shared draft/target context. Up to 5.3x and about 2.0x average over AR. Exactness is not stated in the abstract. — [arXiv abs](https://arxiv.org/abs/2608.15018); [search snippet on utility score](https://arxiv.org/pdf/2608.15018)
- MoESD (arXiv 2505.19645, NeurIPS 2025 spotlight): at medium batch sizes MoE benefits more from SD than dense models, and the useful batch range widens with sparsity. It introduces a "target efficiency" metric. Up to 2.29x on Qwen2-57B-A14B. Relevant as theory; it is not offload-specific. — [arXiv abs](https://arxiv.org/abs/2505.19645)
- "Speculating Experts" (arXiv 2603.19289, Mar 2026): uses current hidden states to predict future-layer experts and prefetch them. It **executes the speculated experts** instead of re-fetching the router's choice, which is lossy ("generally maintains" accuracy). Up to −14% TPOT vs on-demand loading. This is not SD-based. — [arXiv abs](https://arxiv.org/abs/2603.19289)

### Inferences
- pancake-strata's cost-aware window policy (expected committed tokens per measured ms per window size) is already the EVICT objective, k* = argmax E[A]/C(k), for chains. The published gains over a fixed-window EAGLE-3/MTP baseline (about 1.15-1.25x on GPU) therefore mostly do **not** stack on top of it. The remaining levers are (a) making C(k) depend on the *actual predicted expert misses* of this window rather than a per-k average, as DraftExpert and EcoSpec do, and (b) making the benefit term per-token-confidence-aware, as EVICT does with path-probability products.
- The CPU path's marginal cost per missed expert is the analogue of α in Limits of Speculation, and is probably much larger than 0.08 ms. It is roughly linear in the number of unique *CPU-missed* experts, so a cost model C(window) ≈ β_gpu(m) + α_cpu · |misses(window)| should fit well (same form as the paper's R² ≈ 0.97-0.999 fit). Missed experts can be predicted before verification: the MTP head's hidden states or router logits, or simply the previous step's routing per layer (expert temporal locality), give a predicted miss set. DraftExpert's rule (q_i < τ AND Δmiss_i > B) translates directly to *miss count* instead of union count.
- Expected gain on the target: DraftExpert's offload ablation shows that expansion-aware truncation cuts the verify-cost multiple from 2.35x to 1.48x relative to fixed K=4. That cost reduction is measured against a *fixed-K* baseline. Against an existing cost-aware window choice, realistic additional gain is probably 5-15% decode throughput. This is a guess, not measured; it depends on how much per-window miss variance the policy currently ignores.
- Prefetch-from-draft (SP-MoE, Specter, MoE-SpAc) matters only if pancake-strata *transfers* missed experts. If missed experts are always computed on the CPU in place, the analogue is "start CPU expert compute for predicted-miss experts early, overlapped with GPU work". That requires predicting the token's hidden state input, which is not available before the layer runs, so prefetch-style overlap mostly applies to cache *admission/eviction* decisions (MoE-SpAc style utility), not to CPU compute.

### Gaps
- Full-text mechanism details were not fetched for Specter, SpecMoE, SpecMoEOff, SP-MoE, S2-MoE or MoE-SpAc (abstract only). Exactness claims for those are unverified.
- "MoE-Spec" (per-layer expert budget cap) is cited only via the Limits paper; its arXiv ID and numbers were not found.
- No code links were found for DraftExpert, EcoSpec, EVICT or AcceptMoE.

## Q2. Does anyone drop or prune verify rows mid-window by expert cost (layer-wise early row exit)? Partial / early-exit / cascade verification

### Takeaway
No exact (lossless) method was found that drops verify rows **inside** the target forward pass, at a given layer, based on that layer's realized expert misses. All lossless methods (EVICT, DraftExpert, EcoSpec, S2-MoE, Limits-of-Speculation's suggested heuristic) truncate the draft *before* verification. The only verifier-side, per-layer, expert-cost-driven method is AcceptMoE, and it is *lossy*: it reroutes rather than drops. The proposed "skip CPU experts needed only by low-confidence tail rows, making those rows unverifiable" appears to be novel as of Nov 2026.

### Cited Findings
- A targeted search for layer-wise early exit during verification found only draft-side early exit (stop drafting when the cost/acceptance ratio crosses a threshold) and verifier-side expert-set restriction (AcceptMoE). Nothing skips layers or rows for some draft tokens inside verification. — [search results incl. Limits of Speculation](https://arxiv.org/html/2609.22156v1), [AcceptMoE](https://arxiv.org/pdf/2608.02989)
- AcceptMoE weights each row's expert demand by an offline per-depth commitment probability p̂_d^β, and keeps the root row's natural experts unconditionally. Tail rows have low p̂_d, so their exclusive experts are the first to go. Under offload it prunes non-resident experts first. This is exactly the "tail rows' CPU-only experts" set, but it reroutes those rows to resident experts (lossy, −0.27 pp mean accuracy) instead of invalidating them. Gain under offload: 2.06x over standard SD and −74-77% H2D traffic. — [arXiv HTML](https://arxiv.org/html/2608.02989)
- DraftExpert's truncation acts on predicted marginal expansion across all layers before verification. It explicitly does not drop individual verify rows or skip per-layer experts. — [arXiv HTML](https://arxiv.org/html/2607.24434v1)
- EVICT reports a mismatch between token count and expert count: 74.7% fewer verified tokens (claimed) but only 32.5% fewer unique experts on Qwen3-30B-A3B. Most of the expert union is driven by the first rows. — [AcceptMoE discussing EVICT](https://arxiv.org/html/2608.02989); [EVICT HTML](https://arxiv.org/html/2605.00342v2)

### Inferences
**Exactness of the proposed per-layer tail-row skip (reasoning, not sourced):**
- In a chain window, row r+1 depends on row r through attention KV and through the GDN recurrent state at every later layer. If row r's experts are skipped at layer l, row r's hidden state is wrong from layer l onward, and so are rows r+1..m at layers > l. The action is therefore not "skip one row's experts" but "truncate the window to rows 1..r−1 from layer l onward". This is consistent only if the dropped set is a **suffix** (tail rows), and the replay numbers (last row 23%, last two rows 47%) are already suffix-shaped.
- Output stays exact because committed tokens come only from rows whose full 48-layer computation was exact. Losing row r means losing the ability to verify draft r−1's successor/bonus at row r. With m rows, dropping the last row caps committed tokens at m−1 instead of m.
- Bookkeeping: the dropped rows' KV / GDN-state writes at layers < l must be discarded. pancake-strata's lazy replay (commit only through the accepted prefix) already handles this if the "committable prefix" is clamped to rows 1..r−1. GPU work for dropped rows is also saved at layers ≥ l, so the decision should be monotone: once dropped, dropped for all later layers.

**Expected value, worked with chain acceptance a (illustrative arithmetic):**
- A 4-row window has E[committed] = 1+a+a²+a³. Dropping row 4 gives 1+a+a². At a = 0.8 that is 2.95 vs 2.44, a 17% token loss, against at most 23% of CPU-missed expert work saved (only layers ≥ l if decided mid-stack). Dropping rows 3-4 gives 1.8 tokens (−39%) against 47% saved.
- *On average* this is roughly break-even with simply choosing a smaller window up front, which the existing cost-aware policy already does. The real gain comes from **conditioning**:
  - (i) **per-row draft confidence**: if row 4's draft token has q = 0.3, the expected loss is a·a·0.3-ish rather than a³;
  - (ii) **realized routing**: at layer l the router has run, so the actual tail-only CPU-miss count at this layer is known, and the miss rate for the remaining layers can be extrapolated from layers seen so far.
- Neither (i) nor (ii) is used by the current per-size policy. (i) could also be exploited before verification, as in EVICT (path-probability benefit). (ii) can only be exploited mid-pass, which is the unique part of the idea.

**Suggested rule (DraftExpert/EVICT form adapted to per-layer):** at each layer l, for the tail suffix S, drop S if

P(S reached and committed) · value_per_token < Σ_{l'≥l} E[CPU_ms(tail-only misses of S at l')]

Here P(S reached and committed) is the product of MTP confidences for drafts 1..r−1 times calibration, and value_per_token is the measured ms per committed token. Estimate the future-layer term as (observed tail-only CPU ms in layers ≤ l / l) × (48 − l), or via a per-layer prior. A layer-0 decision with a predicted miss set is the pre-verify variant. Calibrate the confidences, since MTP acceptance of 70-90% is not uniform per depth (AcceptMoE's p̂_d is the simplest per-depth calibration).

**Expected gain:**
- The upper bound is the CPU time share × fraction of misses that are tail-only × fraction of windows where dropping is right.
- If CPU expert compute is about 60-70% of step time, and conditional dropping captures, say, a third of the 23-47% tail-only misses in low-confidence windows, the result is roughly 5-10% step-time reduction with about 1-3% fewer committed tokens: net about 3-8% tok/s. This is a guess; verify with the offline replay by simulating the rule with logged MTP confidences.
- Implementation effort is moderate: per-layer bitmask of row validity, clamp of commit prefix, union recomputation after dropping. Staged CPU dispatch would be required (compute misses for rows 1..r−1 first, tail-only misses last, so that tail work can be abandoned).

**Lossy opt-in variant (AcceptMoE-style):** instead of invalidating tail rows, reroute their CPU-only experts to the best GPU-resident experts. Tail rows then stay verifiable but the output distribution changes slightly. Published cost: about −0.3 pp accuracy for about 2x under offload, relative to standard SD.

**A cheaper exact cousin:** order CPU expert computation by "min row index that needs it" and let the window's commit prefix be decided by a deadline (cascade-style). It needs the same suffix-truncation machinery.

### Gaps
- No published measurement exists of how much of the CPU-missed expert set is tail-only in real systems, other than the user's own replay. AcceptMoE's per-depth p̂_d tables were not extracted.
- No literature was found on "cascade verification" or "early-exit verification" for MoE specifically. Older early-exit self-speculative work (LayerSkip, Draft&Verify, 2023-2024) uses early exit for drafting, not verification, and was not re-fetched here.

## Q3. Tree vs chain for MoE; adaptive MTP depth; EAGLE-3/HASS/Medusa/MTP; draft vocabulary reduction and dynamic in-context vocab

### Takeaway
For offloaded MoE at batch 1, the evidence favours chains or very narrow trees with cost-aware truncation. Trees widen the expert union, and every MoE paper above either prunes trees aggressively (EVICT, EcoSpec) or uses chains with expansion-aware stopping (DraftExpert). Static draft-vocab reduction (FR-Spec, VocabTrim) gives about 1.1x on the drafter side and loses acceptance on out-of-distribution tokens. Context-aware dynamic vocab (DynaSpec, SpecVocab) fixes this.

### Cited Findings
- Trees: EVICT's motivation is that bigger trees activate more unique experts and raise iteration latency. Its cost-aware truncation verifies far fewer nodes than EAGLE-3. — [EVICT HTML](https://arxiv.org/html/2605.00342v2)
- EcoSpec uses only a 3-step, top-2 tree and reports no chain-vs-tree comparison. It proves "prefix consistency": a child can never outscore its parent, so truncation keeps a valid tree. — [EcoSpec HTML](https://arxiv.org/html/2607.12696v1)
- AcceptMoE notes that verification "can activate the union of experts chosen by every tree node, even though only a few nodes are accepted". — [AcceptMoE abs](https://arxiv.org/abs/2608.02989)
- Limits of Speculation finds the oracle-optimal draft length on Qwen3-Coder-30B-A3B + EAGLE-3 saturates near 2.8, with accepted length about 2.1, even when 12 is allowed. Per-row cost per expert jumps about 3x at m ≥ 5 rows on A100/vLLM. — [arXiv HTML](https://arxiv.org/html/2609.22156v1)
- DeepSeek-V3.1's MTP drafter gains least from expert-aware selection (EcoSpec 1.10x → 1.15x, experts 31.4 → 31.2). Plausible reasons: MTP is a narrow chain-like drafter and DSV3 has a large active-expert union. — [EcoSpec HTML](https://arxiv.org/html/2607.12696v1)
- llama.cpp gained native MTP support in PR #22673 (one source says merged 16 May 2026, unconfirmed). PR #23269 (May 2026, ggerganov) fixes parallel decoding with recurrent-state rollback, allows chaining multiple speculative methods, and notes that parallel MTP decoding is "highly suboptimal". Issue #23302 reports that draft-mtp with n_max = 3 produced different committed tokens from non-speculative decoding (exactness bug), while n_max = 2 matched. — [PR 23269](https://github.com/ggml-org/llama.cpp/pull/23269), [Issue 23302](https://github.com/ggml-org/llama.cpp/issues/23302), [mer.vin guide](https://mer.vin/2026/05/run-qwen-3-6-mtp-in-llama-cpp-faster-local-inference-with-built-in-speculative-decoding/)
- FR-Spec (2502.14856, Feb 2025) restricts draft candidates to a frequency-ranked subset. It reports a 75% cut in LM-head compute (unverified), output equivalence, and 1.12x over EAGLE-2. — [HF papers](https://huggingface.co/papers/2502.14856)
- VocabTrim uses a static 32K vocab picked from target-output token frequencies. DynaSpec argues static subsets drop acceptance and offset the gains. SpecVocab picks a per-step context-specific subset and reports 2.53x over AR on Qwen3-8B, beating EAGLE-3 and VocabTrim. — [DynaSpec PDF](https://arxiv.org/pdf/2510.13847), [SpecVocab (alphaxiv)](https://www.alphaxiv.org/abs/2602.13836)
- qwen38-megakernel (RTX 3090, Qwen3.x MTP): up to 7 drafts (8-row verify; 15 drafts in an experimental build). A verify cycle costs about 29 ms for 8 tokens vs 22 ms for 1. Greedy output is identical at 0/4/7 drafts. The whole cycle (main pass, lm_head, verify, MTP catch-up, next drafts) runs in one persistent cooperative kernel. Reported 166.5 vs 87.7 tok/s (llama.cpp) on short greedy prompts, of which about 11% is non-speculative kernel gains. The README mentions "the draft vocabulary" from the open-jet megakernel lineage but does **not** document dynamic addition of in-context tokens. — [README](https://raw.githubusercontent.com/cutlerbenjamin1-cmd/qwen38-megakernel/main/README.md)

### Inferences
- For pancake-strata, chain MTP with expansion-aware stopping is the right shape. Tree branching should be limited to cases where the alternate branch's experts are predicted to be GPU-resident (EcoSpec-style "zero marginal miss" branches), since a GPU-only branch costs little.
- Reduced draft vocab: the MTP head's lm_head is GPU-side and cheap relative to CPU expert misses, so vocab reduction mainly matters for acceptance loss, not speed. Adding the prompt's / recent-context token IDs to the reduced draft vocab (SpecVocab/DynaSpec direction) directly attacks acceptance loss on code identifiers and names. The suffix drafter covers part of the same niche.
- Adaptive MTP depth (stopping the chain when the MTP head's top-1 probability falls below a threshold) is the minimal form of DraftExpert's rule. Making the threshold depend on predicted marginal CPU misses is the offload-specific upgrade.

### Gaps
- No primary source was fetched for HASS, EAGLE-3 numbers, Medusa, Qwen3-Next/MiMo MTP training details, or an "llamAmpere draft-mtp-adaptive" implementation. Searches did not surface the latter.
- Dynamic in-context draft-vocab augmentation in qwen38-megakernel could not be confirmed. The GitHub page fetch timed out and the README does not mention it.

## Q4. Speculative decoding for hybrid linear-attention / Gated DeltaNet models: state rollback

### Takeaway
There are three approaches: (1) snapshot the recurrent state per draft position (default in current servers, memory-heavy, cannot share across tree branches); (2) TreeWY: compute all draft-node outputs with one WY triangular solve and reconstruct only the accepted state; (3) lazy replay: commit state only through the surely-accepted prefix and recompute the rest. For chain drafting at batch 1, lazy replay is sufficient and cheap. TreeWY matters mostly for wide trees and memory-bound serving.

### Cited Findings
- TreeWY (arXiv 2608.20961, Aug 2026, rev. Oct 2026): existing systems "snapshot the full recurrent state at every draft position", which cannot be shared across branches, so wide trees become memory-infeasible. TreeWY uses a tree-structured WY transform of the gated delta rule: one triangular solve gives every node's output, and only the accepted state is reconstructed from a small pseudo-value matrix. On Qwen3.5 35B and 397B it frees memory at the same acceptance length. Where not memory-bound the cost is "a few percent", and wider trees are "not yet a throughput gain". — [arXiv abs](https://arxiv.org/abs/2608.20961)
- qwen38-megakernel lazy replay: "the recurrent state commits only through the surely-accepted token; rejected drafts cost no state snapshots". — [README](https://raw.githubusercontent.com/cutlerbenjamin1-cmd/qwen38-megakernel/main/README.md)
- llama.cpp PR #23269 fixed a parallel-decoding bug with recurrent rollback. Issue #23302 shows a mismatch at 3 drafts, which shows rollback correctness is fragile in practice. — [PR](https://github.com/ggml-org/llama.cpp/pull/23269), [Issue](https://github.com/ggml-org/llama.cpp/issues/23302)

### Inferences
- The per-layer tail-row drop idea composes cleanly with lazy replay: clamp the "surely-accepted" boundary to the shortened committable prefix. Dropped rows' GDN updates at layers < l are simply never committed.
- TreeWY is relevant only if pancake-strata moves to trees. Under the offload bottleneck, trees are discouraged (Q3), so its expected gain on the target is about 0.

### Gaps
- No numbers on FlashInfer `gated_delta_rule_mtp` were retrieved, and TreeWY's reconstruction cost and hardware were not extracted from the full text.

## Q5. Relaxed / lenient acceptance and its quality cost

### Takeaway
Little 2025-2026 primary evidence was gathered in this pass. The only quantified lossy MoE-specific trade found is AcceptMoE (restrict/reroute verifier experts): −0.27 pp mean accuracy (worst −1.22 pp) for 1.29x resident and 2.06x offload over standard SD. "Speculating Experts" (executing predicted instead of routed experts) is lossy at an unquantified "generally maintained" accuracy, for −14% TPOT.

### Cited Findings
- AcceptMoE accuracy cost: mean −0.27 pp over 12 model-task pairs; per model −0.26 / −0.32 / −0.24; worst −1.22 pp. — [arXiv HTML](https://arxiv.org/html/2608.02989)
- Speculating Experts: executing speculated experts "generally maintains downstream task accuracy", with lightweight estimators added for MoEs where it degrades. — [arXiv abs](https://arxiv.org/abs/2603.19289)

### Inferences
- In pancake-strata, any lenient option should be a strict opt-in flag, because the user's stated requirement is exact output. AcceptMoE-style rerouting of tail-row CPU misses to resident experts is the lossy variant with the best evidence for offloaded MoE.

### Gaps
- No sources were fetched for Medusa typical acceptance, mb-MTP, top-k lenient acceptance, or their quality costs. These need a separate pass.
