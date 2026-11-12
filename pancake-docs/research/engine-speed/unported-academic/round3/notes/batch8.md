# Batch 8 (round 3): A53, A54, A55, A57

Scope: MOD_PLAN rows A53, A54, A55, A57 (MOD_PLAN.md lines 1948, 1957, 1958, 1960). Research only: no code edits, no builds, no GPU. Repo checks are read-only (grep, sysfs reads, `os_tune.sh` source). Read levels are stated per item: the two arXiv papers were read as full HTML text through an extraction step (not verbatim), the SOSP 2024 paper from its abstract record only, and BFCL from the v3 blog because the v4 leaderboard text did not load.

## A53: IOMMU passthrough guardrail at start (Fast & Safe IO Memory Protection, SOSP 2024)

### Takeaway
The paper is about making strict (translated) IOMMU protection cheap, which is a different mode from the one this box runs. On this box the GPU is already in passthrough (identity), so the paper's mechanism does not apply; the only item worth keeping is the start-up check, and the main HIP engine does not have one yet.

### Cited Findings
- Claim (abstract): IO memory protection stops faulty or malicious devices from making errant transfers; current mechanisms cause "significant throughput loss, large tail latency increases, and isolation violations"; the authors say the fix is to "reduce the cost of each IOTLB miss rather than only minimizing miss rates"; no host hardware change, minimal OS change. [Illinois record](https://experts.illinois.edu/en/publications/fast-amp-safe-io-memory-protection/); DOI 10.1145/3694715.3695943, SOSP 2024, pp. 95-109, Austin, Nov 4-6 2024. Authors: Benny Rubin, Saksham Agarwal, Qizhe Cai, Rachit Agarwal.
- The abstract gives no throughput, latency or overhead percentage. The only magnitude is the qualitative "orders-of-magnitude tail latency inflation". [Illinois record](https://experts.illinois.edu/en/publications/fast-amp-safe-io-memory-protection/)
- Mechanism (artifact README, a fork of the authors' repo): contiguous IOVAs per Tx/Rx ring, invalidate IOMMU page-table caches only when the table structure changes, batch DMA unmaps per descriptor; kernel patch for Linux 6.0.3; the README tells the user to run with `iommu.strict=1` (strict mode, i.e. translated). [xlab-uiuc/vfree](https://github.com/xlab-uiuc/vfree)
- The artifact README does not mention passthrough (`iommu=pt`) at all, and it shows no figures; the results are in its `results_summary/` directory, which I did not read. [xlab-uiuc/vfree](https://github.com/xlab-uiuc/vfree)
- Live box: `/proc/cmdline` has no `iommu=` option; the dGPU's IOMMU group (0000:12:00.0, group 11) has type `identity`. [local: /sys/kernel/iommu_groups/11/type]
- The repo's own measurement with translated DMA (a third-party report, not F&S): pinned copies 7.7-9.2 GB/s with the IOMMU translating (`DMA-FQ`), 12.3 GB/s with `iommu=pt`, about 30% prompt speed. [local: /home/dan/clone-of-d/src/Strata/docs/DETAILS.md lines 1551-1552 (reporter: 2x Xeon E5-2670 v2, 2x CMP 50HX)]

### Inferences
- Exact or lossy: exact (a protection and placement item; it changes no output). The gain here is 0 (the guard prevents a regression, it does not add speed).
- Relevance: F&S optimizes the translated mode. Under identity there are no IOTLB misses to make cheaper, so F&S is not a candidate for this box. It would matter only if someone turned on strict translation for security, which is not planned.
- Already in the repo:
  - `tools/pancake/os_tune.sh` function `iommu_check` (line 77) reads the dGPU's group type from sysfs and prints a warning for `DMA`/`DMA-FQ`. It is a check run by hand (`os_tune.sh check`), not at engine start.
  - `src/split/src/model.cpp` lines 54-76 `require_identity_iommu`: the split (CUDA, multi-card) engine refuses to start when a card's group is not `identity`. That is a refusal, not the "warn" that MOD_PLAN item 54 asks for.
  - The main HIP engine has no start-up IOMMU check (grep for `iommu_group` / `DMA-FQ` in the engine sources finds only the split engine).
  - Status drift: row A53 says "planned (folded into 63)" (line 1948) while line 1727 says "GPU already in passthrough".
- Smallest test (cost: about 20 lines of code and a sysfs read, no GPU): add one start-up line to the HIP engine that reads the same sysfs file and prints `IOMMU group <type>` with a warning for `DMA`/`DMA-FQ`, plus a unit test that points the reader at a fake sysfs directory. The reader cannot be run today on this box without a build, and the user asked for no builds in this round, so the test is deferred. Check first: `bash /home/dan/clone-of-d/src/Strata/tools/pancake/os_tune.sh check` (read-only) already prints the current state.
- Re-check cost: none. A GPU run is not needed for this item.

### Gaps
- The full text of the SOSP 2024 paper was not available, so I have no measured strict-mode overhead for F&S and cannot say how large the translated-mode penalty is on a 7900 XT. The repo's 30% figure comes from another machine.
- The artifact `results_summary/` directory was not read.

## A54: KLD and top-1 rejection of large deviations; fixed task set to choose between close variants (Displacement Is Not Direction, arXiv 2606.19558)

### Takeaway
KLD and top-1 agreement track the volume of disagreement with the reference (silent-zone correlation with the downstream composite is about zero on Qwen3.6-35B-A3B and -0.24 on Devstral-Small-2-24B). So they are fine for rejecting large deviations, and a fixed task set should choose between close variants. The paper's "no information" claim rests on 16-17 models per zone with wide intervals, so it is a strong hint, not a settled result.

### Cited Findings
- Authors: Milos Nikolic, Ali Hadi Zadeh, Enrique Torres Sanchez, Andreas Moshovos; arXiv cs.LG (cross-listed cs.CL), submitted 17 Jun 2026; a DOI in the EMNLP 2026 industry track is listed, venue not confirmed on the page. [arXiv abstract](https://arxiv.org/abs/2606.19558)
- Full-cohort correlation of KLD with the downstream score: rho = -0.72 on Qwen3.6-35B-A3B (28 GGUF quants, 2.17-6.14 bpw) and -0.86 on Devstral-Small-2-24B (41 quants, 1.88-9.84 bpw), both p < 0.001. [arXiv HTML](https://arxiv.org/html/2606.19558); [arXiv abstract](https://arxiv.org/abs/2606.19558)
- Silent zone (near-baseline variants): KLD mean rho = +0.00 on Qwen (17 silent / 11 lossy models, threshold tau = 0.064) and -0.24 on Devstral (16 silent / 25 lossy, tau = 0.027). Bootstrap 95% CI of the Qwen silent-zone correlation: [-0.53, +0.54]; Devstral: [-0.63, +0.33]. [arXiv HTML](https://arxiv.org/html/2606.19558)
- Top-1 agreement in the silent zone: -0.04 (Qwen) and +0.25 (Devstral) on WikiText 512-token context; the other variants (KLD median, p90, p99, log-PPL, 8k context) are all within |rho| <= 0.25 and none significant at p < 0.10. [arXiv HTML](https://arxiv.org/html/2606.19558)
- Mechanism (paper's words, via the extraction): KLD "mainly measures the volume of disagreement"; its relation to the direction of the disagreement is "weak and task-conditional". Per-prompt LiveCodeBench failure prediction: failed-vs-passed geometric-mean ratio in [1.08, 1.22] over five models; cross-model routing on disagreement prompts: 42.3%-49.4% accuracy. [arXiv abstract](https://arxiv.org/abs/2606.19558)
- Downstream composite = unweighted mean of normalized HumanEval, LiveCodeBench v6, IFEval, BFCL v3, GSM8K, GSM8K-v, MMLU (Devstral adds MATH-500), run with EvalScope + llama.cpp, non-thinking mode. [arXiv HTML](https://arxiv.org/html/2606.19558)
- Caveat in the method: the silent-zone cutoff is "the point below which the within-zone Spearman correlation is non-significant (p > 0.10)", i.e. the zone is chosen where the correlation is null. That is a selection on the outcome and inflates the null result. The robustness check with a quality-defined zone (>= 99% of BF16 composite; 18 Qwen and 22 Devstral quants, Appendix C) is the cleaner test; I did not read its numbers. [arXiv HTML](https://arxiv.org/html/2606.19558)
- The paper does not define its top-1 agreement formula and does not give the total token count of the KLD set. [arXiv HTML](https://arxiv.org/html/2606.19558)

### Inferences
- Exact or lossy: this is a measurement rule. It changes no output and gains nothing by itself; it decides which lossy arms are kept.
- Already in the repo:
  - Token forcing (large-deviation rejection): `tools/pancake/quality_gate.py` and `tools/force_check.py` (rate rule + event rule, SpecPilot rules; `check --ear` adds EAR from A56). These come from SpecPilot, not from this paper.
  - Fixed task set and repeated runs: `tools/pancake/task_gate.py` (A55; see below), which the docstring says is motivated by this paper.
  - The docstring of `task_gate.py` says "rho = 0.00 on Qwen3.6-A3B" and that "KLD / top-1 agreement carry no information about the task score among near-baseline variants". That overstates the source: the Qwen 0.00 is the headline KLD value only, Devstral is -0.24, the intervals are about +/-0.5, the zone is chosen post hoc, and the paper did not test a within-zone task gap. The docstring should say that.
  - README line 409 lists the paper as "Adopted: revised quality-gate method". MOD_PLAN row A54 says "done (rule)". Both are accurate for the task gate, not for the token-forcing rules.
- Smallest test (cost: GPU time, not measured): to check the claim on our own variants you need the same kind of population. The paper used 17 silent-zone models; a sample of 6 of our lossy arms (turbo, AcceptMoE depth prior, VERIFY_ECAP, deferral guard, a quant) gives a correlation interval wider than +/-0.5, so it cannot test the claim either way. The smallest useful test is: score each arm with token forcing, run `task_gate.py` on it, and report the rank agreement across all arms the team already has, as a side table, without using it as a gate. Cost: one task-gate run per arm (see A55 for the per-arm count).
- Transfer caveat: the paper's variants are GGUF quants of two dense/MoE models, and its tasks are the public benchmarks. Our variants are engine switches (turbo, deferral, ECAP), which can fail on a different path (tool-call JSON, long context). The paper does not test those.

### Gaps
- Appendix C (quality-defined zone) numbers not read.
- Top-1 formula and KLD token count not stated in the paper.
- No evidence on engine-level variants (speculative decoding switches), only quantization variants.

## A55: Task gate runner (BFCL v4 multi-turn, LiveCodeBench subset, own-repo checkpoints, repeated runs)

### Takeaway
A runner exists in the repo (`tools/pancake/task_gate.py`, 18 offline tasks, unit-tested, first live run pending), but it does not yet contain the two public subsets it is named for. The bigger problem is power: at the current 18 tasks x 5 reps, a 5-point non-inferiority margin cannot be resolved; the design needs about 492 runs per arm for 5 points and about 123 for 10 points.

### Cited Findings
- LiveCodeBench (Jain et al., arXiv 2403.07974, v1 12 Mar 2024, v2 6 Jun 2024): "contamination-free" code benchmark; problems from LeetCode, AtCoder and Codeforces; "four hundred high-quality coding problems" published May 2023 to May 2024; 18 base and 34 instruction-tuned models; scenarios beyond generation: self-repair, code execution, test output prediction. The abstract gives no pass@1 definition. [arXiv abstract](https://arxiv.org/abs/2403.07974)
- The paper's own composite (Displacement paper) uses LiveCodeBench v6 and BFCL v3, so the same benchmark families are the ones a quality gate is most likely to share with the literature. [arXiv HTML](https://arxiv.org/html/2606.19558)
- BFCL multi-turn (v3 blog; v4 not verified): base multi-turn 200 entries; augmented multi-turn 800 entries in four 200-entry sets (missing parameters, missing functions, long context, composite); the page says its leaderboard composition is outdated and lists only three augmented sets. Two checks per turn: state-based (backend state equals ground truth, excluding private attributes) and response-based (the execution path contains the minimal ground-truth path as a subset). An entry passes only if both checks pass in all turns; more than 20 steps in one turn is a failure. [BFCL v3 multi-turn blog](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html)
- BFCL leaderboard overall accuracy is "the unweighted average of all the sub-categories". The v4 leaderboard text did not load, so I have no v4 multi-turn numbers. [BFCL leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
- Repo state (read-only): `tools/pancake/task_gate.py`; task set `bench/tasks/default.json` with 18 tasks: 8 code_edit, 7 tool_call, 3 qa; no BFCL or LiveCodeBench entries; 13 unit tests in `tools/pancake/test_task_gate.py` and `test_task_gate_effort.py`; no `runs-*.jsonl` exists yet under `bench/` (so no live result). Spec defaults: reps 5, temperature 0.2, max_tokens 1024, margin 0.05, validity_margin 0.03; base arm runs twice unless `--no-base-repeat`. [local: /home/dan/clone-of-d/src/Strata/tools/pancake/task_gate.py; /home/dan/clone-of-d/src/Strata/bench/tasks/default.json]
- Power (my calculation, unpaired two-arm, pass rate 0.8, 95% normal interval; not from a source): 90 runs per arm gives a half-width of 11.7 points; 180 gives 8.3; 360 gives 5.8; 500 gives 5.0. A 5-point non-inferiority margin needs about 492 runs per arm; a 10-point margin needs about 123. The task-level bootstrap in `task_gate.py` is wider than this, because the 18 tasks are fixed, so these are lower bounds on the width.

### Inferences
- Exact or lossy: measurement only.
- Already in the repo: the runner, the Wilson and bootstrap statistics, the verdict rule, and `--from-runs` recompute (`task_gate.py`). Not in the repo: BFCL multi-turn adapter, LiveCodeBench subset loader, own-repo checkpoint grading (the plan's "own-repo checkpoints" item).
- Plan status: MOD_PLAN item 67 says the first PR is "a 30-task BFCL v4 multi-turn subset on two presets"; that is not built. Row A55 says "18 offline tasks, not the BFCL / LiveCodeBench subsets yet". The plan's own note "A55 needs 100+ tasks or a 10-15 pt margin" (row agentloop, line 1513) agrees with the power numbers above.
- Smallest test (cost: GPU time plus a new adapter, both unmeasured):
  1. Dry run, no GPU, no server: `python tools/pancake/task_gate.py bench/specs/taskgate-lossy.json --out <dir> --dry-run` to validate the spec and checkers (already documented; deferred, not run here).
  2. The first live run on base and one lossy arm, 5 reps, 18 tasks: about 90 runs per arm plus the base repeat. That run cannot resolve a 5-point margin; use it to measure the median request time and the failure rate, which sets the real per-run cost. Estimate, not measured: at 60 s per run, 270 runs is about 4.5 hours of GPU.
  3. The 30-entry BFCL base subset (base multi-turn, 200 entries; take 30 with a fixed seed) on the same two arms, one rep, as a harness check, not a gate.
- Recommendation: report the interval width with every verdict, and do not declare non-inferiority at 18 tasks; add the BFCL base subset before the first decision on turbo vs AcceptMoE.

### Gaps
- BFCL v4 (not v3) multi-turn counts and scoring were not read; the v4 leaderboard text did not load.
- The LiveCodeBench release/version used for a subset (and its window relative to the model cutoff) is not chosen; v1 covers May 2023 to May 2024 per the abstract, which may already be in the training data of the models under test.
- Per-run cost of the task gate is unmeasured.

## A57: Area under the time-to-success curve for agent A/Bs (SWE-Effi, arXiv 2509.09853)

### Takeaway
SWE-Effi normalizes a cumulative resolved-issues curve against a cap on tokens, cost or time, and reports the area as a single 0-1 score next to the plain resolve rate. Our `time_auc()` uses the same idea (success weighted by how early it came, over a fixed horizon), and it is already in `task_gate.py`, unit-tested and information-only. Its weak point is the horizon: it defaults to the longest run in the batch, so two batches are not comparable unless the horizon is pinned.

### Cited Findings
- SWE-Effi (Fan et al., first author Zhiyu Fan; arXiv cs.SE, submitted 11 Sep 2025, v2 18 Sep 2025; no venue listed). It re-ranks agent systems on SWE-bench-Verified subsets by effectiveness under budget, not just solve rate. [arXiv abstract](https://arxiv.org/abs/2509.09853); [arXiv HTML v2](https://arxiv.org/html/2509.09853v2)
- Definition (paper's words via extraction): for each metric, "we plot the cumulative number of resolved tasks"; "the resulting AUC is then normalized to a score between 0 and 1". Four metrics with caps: EuTB (tokens, cap 2 million per issue), EuCB (cost, cap $1.00 per issue), EuCTB (CPU time, cap 30 minutes), EuITB (inference time, normalized, cap 30 minutes). No closed-form formula is given in the text read. [arXiv HTML v2](https://arxiv.org/html/2509.09853v2)
- Table 1 (%, subset of 50 issues from SWE-bench-Verified, 5 scaffolds x 3 models): examples, EuTB / resolve rate: OpenHands + Qwen3-32B 22.7 / 34.0; Agentless + Qwen3-32B 46.7 / 48.0; SWE-agent + GPT-4o-mini 5.1 / 10.0; Agentless + GPT-4o-mini 25.3 / 26.0. The AUC tracks the solve rate and falls below it as the budget binds. [arXiv HTML v2](https://arxiv.org/html/2509.09853v2)
- Repo implementation: `time_auc(runs, horizon)` = mean over all runs of `ok * (horizon - elapsed) / horizon` (so a failure counts 0, an instant success 1, a success at the horizon 0); horizon = the longest run in the batch unless `--auc-horizon`; paired bootstrap of the difference vs base, information only. Unit tests: `test_time_auc_definition` and `test_auc_ranks_a_faster_equal_success_arm_higher_and_is_not_part_of_the_verdict` in `tools/pancake/test_task_gate.py`. [local: /home/dan/clone-of-d/src/Strata/tools/pancake/task_gate.py lines 513-530]
- The two definitions are the same up to scaling: the integral of the cumulative-success curve over [0, T] equals the sum over successes of (T - t_i), which is the repo's formula times the number of runs. The differences are that SWE-Effi caps tokens and cost as well as time, and uses one run per issue, where we repeat runs and bootstrap.

### Inferences
- Exact or lossy: measurement only. It is the right metric for A18 (effort per task) and A38-A43 (agent loop), as the plan says.
- Already in the repo: `time_auc`, the `time_curve` spec key (success within 10/30/60/120/300 s), and the paired bootstrap. Not in the repo: any cost or token budget (only seconds). Status drift: row A57 says "planned" (line 1960) while the summary table (line 1728) says "used by the quality gates"; in fact it lives in `task_gate.py`, not in `quality_gate.py`.
- Horizon problem: the repo's default horizon is the longest run of the batch, so AUCs from two batches with different slowest runs are not comparable. Fix: pin `time_horizon_s` (or `--auc-horizon`) in the spec and report the AUC at two or three horizons, the way SWE-Effi's caps are chosen per budget.
- The metric is dominated by the success rate when the horizon is long; the interesting part is the gap between AUC and success rate. Report both.
- Smallest test (cost: one A/B of the existing 18-task set, GPU time, not measured): run A18 (`xhigh` vs `medium`) with `"time_curve": true` and a pinned horizon, 5 reps; the output already has the AUC difference with its bootstrap interval. The question it answers: is the AUC difference larger than its interval at the horizon chosen before the run. Note the 18-task, 5-rep power limit from A55 applies: a time-AUC difference of about 0.05 needs more runs than the 90 per arm available.
- Offline check, no GPU: `python -m pytest tools/pancake/test_task_gate.py -k auc` (not run here; the tests are the existing ones).

### Gaps
- SWE-Effi's exact normalization constant and its treatment of unresolved issues (a fail vs censored at the cap) are not in the text read; the repo's choice (failures count 0) is ours.
- No measured AUC exists yet on any Strata batch (no `runs-*.jsonl` in `bench/`).
- The venue of SWE-Effi is not stated on the arXiv page.

## Cross-cutting notes for the plan owner

1. A53: the plan's "warn at startup" for the HIP engine is still not built; the split engine refuses instead. Decide which behaviour the HIP engine should have (warn is the plan's wording).
2. A54/A55 wording: the `task_gate.py` docstring's "rho = 0.00 ... carry no information" should be narrowed to what the paper shows (silent-zone correlations within +/-0.25 with intervals of about +/-0.5, zone chosen post hoc).
3. The SOSP 2024 citation in MOD_PLAN row A53 points to the Illinois record; the DOI is 10.1145/3694715.3695943.
4. Statuses to reconcile: A53 (planned vs "GPU already in passthrough"), A55 (first PR: BFCL subset not built), A57 (planned vs used by the quality gates).
5. Power: nothing on the 18-task set can support a 5-point non-inferiority verdict; the first live run should be read as a cost and failure-rate measurement.
