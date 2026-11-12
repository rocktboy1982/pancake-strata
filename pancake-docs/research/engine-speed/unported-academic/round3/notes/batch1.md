# Round 3, batch 1: A1 cache-conditional routing, A2 depth-graded verify prior, A4 rank guard, A6 time model, A7 chunked promotions

Read-only pass, 2026-11-12. No code edits, no builds, no GPU. Read level per item is given in each section. Repo checks were grep only, on branch port/rdna3-decode, in `src/`, `tools/`, `bench/specs/`. Worktree copies under `.claude/worktrees/` were excluded.

Summary of the batch:

| Item | Paper read level | Exact or lossy | Already in repo | Verdict for the next GPU step |
| --- | --- | --- | --- | --- |
| A1 | full text (HTML, Sec. 3 and 4 via summary) | lossy | yes: `STRATA_ROUTE_PRIOR` in `src/core/verify.cpp`, `src/kernels/cuda/route_prior.cu` | keep as a gated opt-in; re-measure jointly with deferral |
| A2 | full text (HTML, method and offload eval via summary) | lossy | yes: `STRATA_ROUTE_PRIOR_DEPTH` in `src/core/verify.cpp` | the MOD_PLAN row 23 mechanism is NOT AcceptMoE; the "+5-15%" has no source |
| A4 | full text (HTML, Sec. 3.2, Table 3, Sec. 5.5.1 via summary) | lossy-small | yes: `STRATA_VERIFY_DEFER_CPU_TOP`, `_FOLD`, `_BELOW` | cut on the GPU already (no gain over deferral); HOBBIT's T1/T2 is a different rule |
| A6 | abstract only (Vidur); the 3.4-point model error is our own offline result | exact (offline) | yes: `tools/pancake/time_ceiling.py --validate` | the 3.4-point error is for the 4 fitted arms; the rank-guard miss (-8.1) shows it is not a general 5-9% |
| A7 | full text (HTML, Sec. 4-7 via summary) | exact | no `src` hit; `bench/specs/22-a7-promo-chunk.json` only | the paper's "cancellable" is weaker than our plan wording; the simulator bound is < 5% |

## A1 - Cache-conditional routing (Cache-Prior)

### Takeaway
Cache-Prior adds a cache-residency bonus to router logits used only for ranking; the paper reports miss-rate cuts of about half at a small perplexity cost, on CPU llama.cpp on phones, not on a GPU offload engine. Our `STRATA_ROUTE_PRIOR` is the same family, lossy, and its end-to-end gain with deferral on is still unmeasured.

### Cited Findings
- Claim and mechanism: a training-free, cache-aware routing method raises expert cache hits and cuts miss rates by over 50% at a small accuracy cost; the bias applies to cached experts' router logits, scaled by the per-layer logit range and a trade-off lambda in [0,1]; the original weights still produce the output; a top-J guarantee keeps the top experts loaded regardless of cache state. — [Cache-Conditional Experts, arXiv HTML 2412.00099](https://arxiv.org/html/2412.00099) (Abstract, Sec. 3.3 Eqs. 9-10)
- Evaluation hardware and model: Qwen1.5-MoE-A2.7B on two Snapdragon Android devices (12 GB and 16 GB RAM), llama.cpp on CPU, 30-expert cache (4-bit) and 45-expert cache (8-bit). — [arXiv HTML 2412.00099](https://arxiv.org/html/2412.00099) (Sec. 4.5)
- Baseline: original routing with LRU caching. On-device throughput over 2x vs LRU; throughput rises nearly linearly with hit rate over lambda 0.1-0.9. — [arXiv HTML 2412.00099](https://arxiv.org/html/2412.00099) (Sec. 4.5, Fig. 1, Fig. 8)
- Hit rate at lambda 0.5 on Wikitext: miss rate 35% to 16% (Qwen), 28% to 7% (DeepSeek-V2), 22% to 9% (Phi-3.5-MoE), 40% to 21% (Mixtral). — [arXiv HTML 2412.00099](https://arxiv.org/html/2412.00099) (Sec. 4.6, Fig. 9)
- Quality: Wikitext perplexity rises 0.1% to 3%; the paper reports MMLU and GSM8K accuracy drops under 0.1% in the conclusion, but Sec. 4.4 calls these "no accuracy loss" in some settings, so the wording varies. Lossy by the paper's own framing. — [arXiv HTML 2412.00099](https://arxiv.org/html/2412.00099) (Sec. 4.3, 4.4, 4.7, Conclusion)
- Bias-strength behaviour: larger lambda lowers misses until a floor set by the top-J guarantee; at lambda 0.8 the model over-uses cached experts, which may hurt prediction. A learned bias variant (Appendix E) showed no notable gain over the training-free one. — [arXiv HTML 2412.00099](https://arxiv.org/html/2412.00099) (Sec. 3.3, 4.8, App. E)
- Repo status: `STRATA_ROUTE_PRIOR` is read in `src/core/verify.cpp` (6 hits), with a CUDA kernel `src/kernels/cuda/route_prior.cu` and a test `src/kernels/route_prior_test.cpp`. Specs `bench/specs/06c-turbo.json`, `06-turbo-depth.json`, `19-turbo-guards.json` and `taskgate-lossy.json` use it. (grep, read-only)
- Plan status (MOD_PLAN row A1): "built, gate pending"; measured +30-50% before deferral (111.7 to 149-168 tok/s); "with deferral on the sum is lower - re-measure jointly". The F13 guards are on `work/f13-prior-guards`. Note: the 2026-11-09 summary in MOD_PLAN (line 1543) and the rank-guard review also apply to this switch's guard logic. — MOD_PLAN.md rows A1, F13
- Cross-check of the existing round-2 note: a lossy-only option; "2.9% tokens differ" is from our own plan row, not from the paper.

### Inferences
- The paper's gain is for an on-device CPU llama.cpp setting where a miss is very expensive and the hit rate is the whole cost. On our box the CPU pool already computes misses while the GPU runs, so the gain is bounded by the window time the misses cost after deferral (A6 model: deferral already takes most of the gain). A joint measurement is therefore the only decisive test, which the MOD_PLAN row already says.
- The paper's own accuracy evidence is perplexity plus MMLU/GSM8K, not agent or code-edit tasks. Our task gate (A55) is the relevant check, not the paper's number.
- The paper's lambda floor (top-J) matters for us: the lossy effect will concentrate on edits where the routed set is most important; a per-window opt-in (A44 bandit) is the natural fit.

### Gaps
- No paper number exists for an offload engine on a GPU with a 256-expert top-8 model; Qwen1.5-MoE is 60 experts. Transfer of the hit-rate gain to Flash-Next is untested.
- Paper's Table of exact accuracy per lambda not read (summary only); the 0.1% accuracy claim is therefore from the summary, read level: summary of full HTML.
- Our "2.9% tokens differ" figure is not from the paper and I did not verify where it was measured.

### Smallest test on our engine
- Arms: base fast-hip, `STRATA_ROUTE_PRIOR 0.5` alone, and the same with deferral on (already the fast-hip preset). Use `bench/specs/06c-turbo.json` shape with the current preset; add the F13 guards spec `19-turbo-guards.json` only if the bare prior fails the gate.
- Gate: task gate `task_gate.py` plus `replay_turns.py` (run by hand; see the promotion test-protocol note in MOD_PLAN) and token forcing; PROMOTE requires 95% lower bound of gain >= 3% for lossy items (promote.py rule).
- GPU time: not timed (I did not read the spec durations). Count: 3 arms x short + edit x repeated A/B/B/A runs, plus the task-gate runs. Time it from the first night's log before scheduling.

## A2 - Depth-graded verify prior (AcceptMoE)

### Takeaway
The AcceptMoE paper does not describe a depth-graded logit bonus. Its depth weight multiplies router probabilities inside a utility used to rank experts, and it reroutes non-resident experts. Our `STRATA_ROUTE_PRIOR_DEPTH` is a different mechanism, so the plan's "+5-15%" for row 23 has no source number and must not be cited as AcceptMoE.

### Cited Findings
- Commitment probability per draft depth: p_hat_d = c_d / m_d (proposed nodes at depth d that lie on the accepted prefix), estimated offline; token weight alpha_t = p_hat_d^beta * N_d^(-1/2) with beta = 0.5; utility u_e = sum_t alpha_t * router probability over tokens whose natural top-k includes e. — [AcceptMoE, arXiv HTML 2608.02989](https://arxiv.org/html/2608.02989) (Method, Eqs. 3-4, "Commitment-Weighted Demand")
- Expert set: root token's natural top-k always kept (anchor, Eq. 5); other experts ranked by utility; set size = effective rank (exp of entropy) of normalized utilities, rounded up, minimum k (Eq. 6). Eligibility is a 0 / -infinity mask on logits (Eq. 2), not position-dependent; excluded tokens are rerouted to eligible experts. — [arXiv HTML 2608.02989](https://arxiv.org/html/2608.02989) (Method)
- Offload evaluation: one RTX 5090 (32 GB), 48 expert slots per MoE layer, SGLang 0.5.12.post1, batch size 1, 256 tokens per prompt; models Qwen3-30B-A3B-Instruct-2507, Qwen3-Coder-30B-A3B-Instruct, GPT-OSS-120B; EAGLE-3 draft; baseline Standard SD (natural routing). Mean 2.06x over Standard SD, and H2D traffic down 73.6% to 77.1%. — [arXiv HTML 2608.02989](https://arxiv.org/html/2608.02989) (Experiments, Table 2)
- Accuracy: offload average 90.25% vs 90.90% for Standard SD (Table 1); residency-aware pruning changes mean accuracy by -0.27 points over 12 model-task pairs, per pair -1.83 to +1.22. Lossy. — [arXiv HTML 2608.02989](https://arxiv.org/html/2608.02989) (Table 1, Ablation)
- Repo status: `STRATA_ROUTE_PRIOR_DEPTH` is in `src/core/verify.cpp` (4 hits) and `src/core/conversation_disk_tier.cpp`; specs `06-turbo-depth.json` and `taskgate-lossy.json` use it. MOD_PLAN row 23 says "coded `STRATA_ROUTE_PRIOR_DEPTH` (4f56ce535)" and MOD_PLAN line 1544 says it is not AcceptMoE. Confirmed by the paper text.
- Plan status: MOD_PLAN row A2 "+5-15% over fast-hip (est.)"; source of this number is not stated in the plan. The round-1 note (`moe-speedup/notes/moe_speculative_decoding.md`) records AcceptMoE's 2.06x as relative to Standard SD on a 5090, not to a fast-hip baseline.

### Inferences
- The valid transfer from AcceptMoE is the residency-aware pruning idea (drop non-resident experts in increasing utility until a budget), which is closer to our deferral + rank guard than to a depth prior. That is A4's territory, not A2's.
- Our depth prior (fixed geometric logit bonus per position) is untested against a utility-based set, so the decision in row 23 ("pick A1 or A2") should compare A1 and the depth prior, not AcceptMoE.
- AcceptMoE's gains are relative to Standard SD (verify with a draft tree), which we run with MTP. The baseline is different.

### Gaps
- The depth prior's (STRATA_ROUTE_PRIOR_DEPTH) own measured result is not in the MOD_PLAN rows I read; I did not find a GPU number.
- AcceptMoE per-depth p_hat tables were not extracted (not in the HTML text I fetched).
- Whether AcceptMoE has code: not checked in this pass (round 1 found none).

### Smallest test on our engine
- Decide by name, not by the paper: keep A2 as "our geometric depth prior" and test it with the same spec as A1 (`06-turbo-depth.json`), one arm each, under the same gate.
- If the owner wants the actual AcceptMoE mechanism, a separate row is needed: utility-based set with an anchor, measured only as "residency-aware pruning on verify" (A4-style). That is a new item, not a re-test of A2.
- GPU time: not timed. Same shape as A1 (2 arms x short + edit, plus gate).

## A4 - Rank guard (top-2 or weight threshold)

### Takeaway
HOBBIT's rule is a cumulative-weight importance score with thresholds T1 = 0.6 and T2 = 0.9 that decides precision and skip on misses, measured only on Mixtral and Phi-MoE. Our rank guard (`_TOP`, `_BELOW`, `_FOLD`) is not that rule, and the plan's own GPU screen (rows around MOD_PLAN line 1533) found no gain over deferral alone, so the plan's "fix coded, screen pending" status in row A4 is out of date.

### Cited Findings
- HOBBIT selection rule: experts sorted by normalized gate weight; score of expert e_i = cumulative sum of higher-ranked gate weights (s_e0 = 0); T1 = 0.6 loads high precision, above T1 loads low precision, T2 = 0.9 skips (the explicit T2 comparison is implied, not written); Mixtral split about 67% high, 30% low, 3% skipped. — [HOBBIT, arXiv HTML 2411.01433v2](https://arxiv.org/html/2411.01433v2) (Sec. 3.2, Eq. 2)
- Low-precision copy source: the paper uses pre-existing int4 (for fp16 models) and int2 (for int8 models) versions; it does not describe the conversion. — [HOBBIT arXiv HTML](https://arxiv.org/html/2411.01433v2) (Sec. 4, 5.1)
- Accuracy: Table 3, Mixtral GSM8K 0.52 to 0.51 under fp16+int4 and int8+int2; Phi-MoE unchanged; paper says accuracy drops "no more than 1%". — [HOBBIT arXiv HTML](https://arxiv.org/html/2411.01433v2) (Sec. 5.3, Table 3)
- Dynamic-loading ablation: 1.19x to 1.57x under different configs (per-config values only in Fig. 16). Hardware RTX 4090 and Jetson AGX Orin; baselines Transformers, DeepSpeed-Inference, Llama.cpp, MoE-Infinity, Fiddler. — [HOBBIT arXiv HTML](https://arxiv.org/html/2411.01433v2) (Sec. 5.1, 5.5.1)
- Repo status: `STRATA_VERIFY_DEFER_CPU_BELOW` (19 hits in `src/core/verify.cpp`), `STRATA_VERIFY_DEFER_CPU_TOP` (`src/core/verify.cpp`, `src/program/generate.cpp`), `STRATA_VERIFY_DEFER_CPU_FOLD` (branch `work/a4-rank-guard`, spec `bench/specs/10-rank-guard-fold.json`).
- Plan result, read in MOD_PLAN (line 1533): `10-rank-guard-fold` on the fast-hip base, n=2: top2 short +0.1% / edit -2.6%, top2-fold1 -0.0% / -3.0%, top2-fold2 +0.5% / -4.2%; verdict drop; "A4 rank guard and FOLD HOLD (no gain over deferral alone)". The earlier +20.1% edit was against a base without deferral. MOD_PLAN line 1543: with 10 weights summing to 1 the top weight is at least 0.1, so `_TOP=1` is a no-op at `_BELOW 0.1`.
- Inconsistency to fix in the plan: row A4 (line 1791) still says "fix coded, branch work/a4-rank-guard, screen pending"; the ledger row (line 1533) says the screen ran and was cut.

### Inferences
- HOBBIT's cumulative score differs from our `_BELOW` weight threshold. A cumulative rule could skip the second and third experts when the top two already carry most of the weight, which is a different set from "defer when weight < 0.1". It is untested here and would need the routing trace with weights (the existing 219-window trace has zero weights, per MOD_PLAN row A6).
- The HOBBIT accuracy claim is on 8-expert top-2 models; our model is 256 experts top-8. Transfer is unverified (the same caveat the plan already records).

### Gaps
- No weight-bearing routing trace exists, so the cumulative-score rule cannot be tested offline; a `--dump-routing` run with weights is needed first.
- The paper's threshold sweep for T1 is not read (summary only); the two values above are for Mixtral.

### Smallest test on our engine
- Not worth a new GPU run as a rank guard: the plan's own screen already says no gain over deferral.
- The only new test: one `--dump-routing` run with gate weights (no GPU time beyond one decode run), then an offline replay that scores the cumulative HOBBIT score on the trace. Build a GPU arm only if the replay predicts >= 5% over deferral.
- GPU time: one routing dump run, duration not measured.

## A6 - Recalibrated overlap time model (Vidur method)

### Takeaway
Vidur's abstract claims latency error under 9% across several LLMs, with profiling plus predictive modelling and no hardware named in what I could read. The paper does not cover our overlap model. Our own recalibration (`time_ceiling.py --validate`) fitted the deferral and k-th-miss arms within 3.4 points on the points it was fitted to, but the rank-guard arm was missed by 8.1 points, so the model is not validated to the 5-9% band.

### Cited Findings
- Vidur: a simulator that combines experimental profiling with predictive modelling of operators to estimate latency and throughput; "less than 9% error across the range" on several LLMs (abstract only); no per-model breakdown; no overlap or pipelining model on the abstract page. — [Vidur, arXiv 2405.05465](https://arxiv.org/abs/2405.05465) (abstract, summary of the page)
- Our recalibration: deferral model +15.8% vs measured +15.8% (fitted to the same number); k-th miss predicted +2.7% vs measured +2.0%; FUSE_PHASES +1.1% vs +2.5%; mean error 3.4 points on n = 3 (selftune row). Rank guard predicted +12.0% vs measured +20.1% (model miss -8.1). — MOD_PLAN ledger rows selftune and A6; `bench/results/2026-11-10-offline/A6_A17.md`
- The fitted deferral cost (91 us per deferred layer) is the least certain term by the owner's own notes; it is one fitted number absorbing unmodelled costs. — `bench/results/2026-11-10-offline/A6_A17.md` (Why the model over-predicted)
- Repo: `tools/pancake/time_ceiling.py --validate` exists and runs the engine's layer chain (`window_ms2`); `finemoe_replay.py` and `selftune.py` use the same model.

### Inferences
- The "validate to 5-9% error" in the plan's A6 row cites Vidur's headline number for a different system (GPU cluster simulator); it is not a target our model has met. Our measured error is 3.4 points on the fitted set and 8.1 points on the held-out rank guard arm.
- A7, A8 and A46 were rejected because the model says they are all under 5% at the bench's hit rate; the model error on the rank-guard arm is larger than that bound, so these rejections are only as good as the model's behaviour on non-fitted arms.

### Gaps
- The Vidur full text (HTML 2405.05465v2) was not read; the overlap-model question in the plan is not answered by Vidur.
- Baleen (named in the plan) has no arXiv id in the plan rows I read; not researched here.

### Smallest test on our engine
- Offline only: add the rank-guard arm (short and edit, at least 2 held-out measurements) to the validate fit before using the model to reject A7/A8/A46 again. No GPU.
- If the rank-guard arm stays unfitted, state the model's error in the plan as 3.4 points on the fitted arms and 8.1 on the held-out one.

## A7 - Chunked, cancellable promotions (ProMoE)

### Takeaway
ProMoE's chunking limits a high-priority miss to waiting for one chunk and drops queued low-priority speculative work; it does not cancel an in-flight copy ("an ongoing copy operation cannot be preempted midway"). Its speedups are against LRU and a static cache on a 4090 with PCIe 4.0, and the LRU baseline is per-layer reactive whole-expert LRU, which the paper itself says thrashes. Our own simulator bound is +4.6% (upper), below the plan's 5% bar.

### Cited Findings
- Claim: ProMoE predicts future expert use from intermediate results and prefetches so that "ProMoE eliminates passive cache misses"; abstract reports 2.20x (up to 3.21x) prefill and 2.07x (up to 5.02x) decode average speedups over existing offloading. — [ProMoE, arXiv HTML 2410.22134](https://arxiv.org/html/2410.22134) (Abstract)
- Mechanism: per-layer MLP predictor on CPU (average accuracy 84.7%, Sec. 4.3); stride prefetch raises the prediction distance by one layer at about 5% accuracy cost (Sec. 4.4); chunked prefetch splits each expert into three chunks (Sec. 5.1); early preemption enqueues required experts as high-priority and clears pending low-priority speculative tasks (Sec. 5.2); reordered inference runs cached experts first, in-flight later (Sec. 5.3). — [arXiv HTML 2410.22134](https://arxiv.org/html/2410.22134) (Sec. 4-5)
- Caveat stated by the paper: "Due to the limitations of CUDA's asynchronous copy mechanism, an ongoing copy operation cannot be preempted midway." — [arXiv HTML 2410.22134](https://arxiv.org/html/2410.22134) (Sec. 5.2)
- Evaluation: RTX 4090 (24 GB), i9-14900K, 128 GB host RAM, PCIe 4.0 (stated 32 GB/s, measured 23.9 GB/s, Sec. 2); models DS-1, DS-2, QW-1 (FP16), QW-2 and Mixtral (INT4); ShareGPT, batch 1; transformers and llama.cpp backends. — [arXiv HTML 2410.22134](https://arxiv.org/html/2410.22134) (Sec. 7.1)
- Speedups (Sec. 7.2): transformers prefill 1.42x vs static, 2.21x vs LRU; transformers decode 1.47x vs static, 1.31x vs LRU; llama.cpp prefill 1.36x vs static, 2.12x vs LRU; llama.cpp decode 1.49x vs static, 1.09x vs LRU (1.17x max). Mixtral llama.cpp decode: the on-CPU variant is faster. — [arXiv HTML 2410.22134](https://arxiv.org/html/2410.22134) (Sec. 7.2)
- Baseline caveat: the LRU is the authors' per-layer reactive LRU, and the paper attributes much of the prefill gain over LRU to "cache thrashing caused by LRU" (Sec. 6.1, 7.2). No variance or run counts reported (Sec. 7). — [arXiv HTML 2410.22134](https://arxiv.org/html/2410.22134)
- Repo status: `STRATA_PROMO_CHUNK` appears only in `bench/specs/22-a7-promo-chunk.json` (4 hits); no hit in `src/`. `STRATA_ADAPT_ASYNC`, `_WAIT` and `_BATCH` are in `src/program/generate.cpp`, and `STRATA_ADAPT_SWAPS_AUTO` (the A8 swap-budget controller) is coded.
- Our simulator: A7 "+4.6% upper bound" and A8 "+3.0% whole / +12.2% edit" (finemoe row, offline replay, 219 windows, no weights). MOD_PLAN ranks A46 > A8 > A7 and parks A7.

### Inferences
- The ProMoE speedup over static at decode (1.09x to 1.49x) is from prefetch plus chunking together, on a PCIe 4.0 card with a high-contention baseline; on our PCIe 3.0 box the paper's own cost model says promotions are more expensive (~115 us vs ~42-60 us CPU compute per expert per our notes), which the simulator already charges.
- Our plan row says "chunked, cancellable". The paper gives chunking and dropping of queued speculative work, not cancellation of in-flight copies. The plan wording should say "chunked, queued speculative work dropped".
- The paper's reordered execution (cached first, in-flight later) is exact in output (same experts), so A7 remains an exact candidate; the question is only time.

### Gaps
- No ProMoE number on PCIe 3.0 or on a GPU with a CPU expert pool; the 4090 numbers are with GPU-resident execution on a PCIe 4.0 link.
- Whether the speculative predictor (84.7%) holds on Flash-Next's 256-expert top-8 is not tested; our own trace has no weights, and the finemoe result (prefetch -14.1%) is negative for our trace.
- Paper's "LO" vs "UM" details beyond the summary table were not read.

### Smallest test on our engine
- Keep it in the simulator first, as the plan says: add the chunk size (~256 KB, per the plan's item 53 text) and the drop-queued rule to `time_ceiling.py`. No GPU until the simulator bound is >= 5%.
- If the bound clears 5%, the GPU arm is the existing spec `22-a7-promo-chunk.json` against the async arm (`--adapt-async 1`), short and edit, A/B/B/A. The number of GPU runs is not timed here.

## Open items found during this pass (for the plan owner)

1. MOD_PLAN row A4 says "screen pending" but the ledger (line 1533) says the screen ran and was cut; update row A4.
2. MOD_PLAN row A2 and its "+5-15%" have no source. AcceptMoE is a utility-based set with rerouting, not a depth-graded logit prior; rename row 23 or add a separate AcceptMoE row.
3. MOD_PLAN row A7 says "cancellable"; ProMoE does not cancel in-flight copies.
4. MOD_PLAN row A6 "validate to 5-9% error" is Vidur's headline for another system. Our model: 3.4 points on the fitted arms, 8.1 on the rank-guard arm.
5. No weight-bearing routing trace exists (MOD_PLAN row A6 and this pass's A4 section). One `--dump-routing` run would unblock the HOBBIT score test and the HOBBIT-style replay.
6. GPU durations for the proposed screens were not read from the spec files; time them before scheduling.
