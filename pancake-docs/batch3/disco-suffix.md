# disco-suffix: speculative decoding items of Phase 16.4 (A21, A22/A25/A26, A24, F6, F8, EcoSpec)

Branch `dev/p16/disco-suffix` (from `dev/phase16-academic` 727eb8a62), 2026-11-11. CPU and offline only: no GPU, no engine, no
model was started. MOD_PLAN rows: A21, A22, A24, A25, A26 (16.4), F6, F8 (Phase 15), EcoSpec (research order 3).

## Summary

| Item | Source check | Code | Verdict |
| --- | --- | --- | --- |
| A22 DISCO v2 | method adapted, see below (no entropy, logistic not FFN, rows cost experts) | compiled and tested; engine <-> fitter feature parity test added | exact; gain unmeasured (needs a trace); ceiling below |
| A25 / A26 `draft_report.py` | A25 ok; A26 is an analogy (trained bandit over several draft models) | tests pass; constants checked against `draft_policy.cpp` | report only |
| A24 suffix index | partly matches; chain not tree; paper's gain does not transfer (1.7x over PLD there, -2.4% rounds here) | compiled, unit + TSan + ASan, C++ vs Python on 2,045 real requests | exact; GPU screen pending |
| A21 | ok (Leviathan / Chen), finding confirmed by code | nothing needed in the engine; rule named and pinned by a test | env only: `STRATA_SPEC_MIN_TEMP=0.7` |
| F6 exact speculative sampling | ok (accept `min(1, p/q)`, residual `(p-q)+`, point mass for lookup) | `spec_prob_test` 78 cases pass | env only; combine with `..._GATE=top` |
| F8 `STRATA_SPEC_CUM` | sound as a marginal-benefit rule | ported as a DISCO model on the chain feature (about 20 lines) | exact; replay arms in `disco_fit.py`; spec `b3-f8-spec-cum.json` |
| EcoSpec | tree and trained predictor do not exist here; the chain part is DISCO / DraftPolicy | simulator `ecospec_sim.py` + tests | **no engine switch**: ceiling too small, see below |

Commits: 4c6cbc6b3 (A22 parity test), 3ef33aa7e (A24 score rule in the simulator), d5864d6d1 (A21), bf10834be (F8),
f28fb0a89 (EcoSpec simulator, draft_costs.build_tables), plus the notes / spec commit.

## A22 / A25 / A26

**Source (arXiv 2405.04304, Mamou et al., Intel Labs; main text read, appendices not).** A 2-layer FFN on [top-10 draft probabilities, entropy,
position] is trained against draft-target agreement; after every draft token it is compared with a threshold tau (tuned on a
validation set) and the draft stops. Dense models (StarCoder 15B target / 168M draft, Vicuna 13B / 68M), greedy, latency in ms
on a GPU: DISCO 1.84x vs 1.64x for the best static lookahead on MBPP (oracle 2.28x), 10.3% over the best static lookahead
on average, 31.4% over the "dynHeur" heuristic. Mismatches with the code:

- The classifier is a logistic regression over 9 features, not an FFN, and has no entropy (the MTP selection kernel returns the
  top-1 probability only). Check it with `disco_fit.py`'s held-out AUC against the draft probability alone.
- In the paper a stop saves draft forwards and the stopped token is still verified. Here the draft is already computed and
  the decision is whether to verify its row, which costs CPU-missed experts (SpecFit): a stopped draft is left out of the
  window, and the thresholds are per position. This is an adaptation, not the paper's rule; exactness is unchanged.
- Training labels: the paper uses agreement with a TV-distance loss; here hard accepted / rejected labels from a trace taken
  with `--spec-min-p 0` (every draft verified, labels of later drafts conditional on the earlier ones holding).
- The paper's headroom (oracle over best static: +39%) is for a draft model with SL up to 5+; at `--spec 4` (3 drafts) the
  oracle cut here is much smaller (below).

**Ceiling here.** `bench/results/2026-11-10-offline/tail_ceiling.txt` (oracle that drops exactly the rejected rows on the real
routing trace `bench/prefetch/routing.bin`; modeled): oracle cut saves 1.0% (short chat), 7.9% (code edit), 3.4% (long) of
window time. No stop rule (DISCO, SPEC_CUM, EcoSpec) can beat it; the fork's SPEC_CUM captured about a third of its headroom.
Expect 0.3% short, 1-3% edit.

**Compiled and run.**

- `src/program/generate.cpp` (A22 hooks in both loops, A24 hooks) compiles with the HIP configure (gfx1100, clang 23,
  `build-hip`, target `CMakeFiles/strata.dir/src/program/generate.cpp.o`), and `src/core/mtp.cpp`. **Needs one line that is not
  mine:** `generate.cpp` uses `strata::CalibMeasure` / `calib_*` (A45) but never includes
  `strata/core/calibrate_ratio.hpp`, so the unpatched file stops with 21 errors (all A45). I compiled with that include added
  locally and did not commit it (the compile-gate agent owns it).
- CPU-only CMake configure (`build-cpu/cmake`): `ctest` of `disco_test`, `suffix_index_test`, `row_miss_test`,
  `spec_prob_test`, `suffix_drafter_test`, `draft_policy_test`, `window_rule_test`, `controller_test`, `coupled_draft_test`,
  `gated_proposal_test`: 10 of 10 pass with the GPUs hidden. Also built by plain `g++ -std=c++20 -O2 -Wall -Wextra`: no warnings.
- `disco_test` and `suffix_index_test` also pass under ASan + UBSan; `suffix_index_test` (concurrent insert + draft) passes
  under TSan.
- New `tools/pancake/test_disco_parity.py` + `disco_parity_driver.cpp`: 400 random MTP rounds through `DiscoPolicy::end_round`
  (its trace CSV) and through `disco_fit.DiscoState`: all nine features equal (rel 2e-4, the CSV prints `%.5g`). The fitter and
  the engine cannot drift apart silently now.
- Python: `test_disco_fit.py` (6), `test_draft_report.py` (3), `wtrace.py --selftest`: pass.
- A25 constants (`kShape`, `1 + 0.7 (t-1)`, `draft_rate`) match `draft_policy.cpp`.

**A25 / A26 sources.** A25 (arXiv 2601.11580, vLLM, four SD variants): acceptance length varies by position and verification
dominates time: consistent with what `draft_report.py` measures. A26 (arXiv 2408.08470, EMNLP'24): a contextual bandit trained
offline chooses among several independent draft models; ours is a counterfactual report over lookup / MTP / suffix drafts, no
learning yet. Keep A26 parked as the plan says.

**Left.** The trace run (`STRATA_WINDOW_TRACE`, `STRATA_DISCO_TRACE`, `--spec-min-p 0`), the fit, the screen
(`bench/specs/24-a22-disco.json`). None of it ran (no GPU). Skip the screen if `disco_fit.py` prints under +2% held out.

## A24 suffix index

**Source (arXiv 2411.04975, Oliaro et al., NeurIPS'25 spotlight; method section, ablation read).** Two suffix trees: a global
one over previous outputs and a per-request one over prompt + output. For each tree and each pattern length p = 1..P the
matching node's subtree is expanded greedily by `D(N) = D(parent) x COUNT(N) / COUNT(siblings)` to at most `MAX_SPEC = alpha p`
nodes (alpha 1-4); the candidate with the highest SCORE (sum of D = expected accepted tokens) is the speculation TREE that is
verified (tree attention). Hybrid: use it if SCORE > tau, else a model-based drafter. Measured with vLLM: SWE-Bench (OpenHands)
and AgenticSQL (proprietary), 4 x H100 TP4 for the end-to-end run: 2.5x over vanilla and 1.7x over PLD on SWE-Bench (7.8
accepted tokens a step vs 3.2), 5.3x on AgenticSQL; the ablation shows both trees beat either one and the global tree beats
the per-request tree except on Combine.

Differences from `SuffixIndex`:

- One chain (the most frequent path), not a tree: `--spec` windows here are chains, tree verify needs GDN state forks (rejected
  in 16.4). The index is the global tree; `--lookup` is the per-request one, as in the paper's two-tree split.
- Match rule: three hash chains (24 / 8 / 3 tokens) and the deepest match, not the best SCORE over every p. **Tested:**
  `suffix_sim.py --rule score` implements Alg. 1's pattern choice (candidates p in {3,4,6,8,12,16,24,32,48,64}); on the 2,045
  recorded requests the rounds are the same as the deepest match (-2.45% vs -2.44% against `--lookup`; factor 2: -2.69% vs
  -2.67%). The simpler rule stays.
- alpha 1 (paper: 1-4): factor 2-4 gains 0.3% more rounds but accepts less (earlier note), so alpha 1 stays.
- Hybrid decision: "longer match wins" instead of SCORE vs tau. The policy (DraftPolicy) and the MTP first-token agreement
  still decide on top.
- The gain does not transfer. The paper's 1.7x over PLD rests on very repetitive SWE-Bench rollouts, trees of dozens of tokens
  and free verification compute; here, replaying 1.27M tokens of real opencode output (25 sessions, Qwen3.8-Flash-Next) the index
  saves 2.44% of rounds over `--lookup` (main agent 1.58%, sub-agents 7.53%), counted at one-token-step cost; the MTP drafter is
  not modelled, so the real gain is a fraction. Honest expectation: 0-2% on agent traffic.

**Verified.**

- `suffix_sim.py` reproduces the batch-2 numbers exactly (rounds 560,038 vs 574,016 for lookup alone).
- The C++ `SuffixIndex` replayed on the same 2,045 requests (`suffix_index_test --simulate`): rounds 777,282, offered
  1,361,216, accepted 493,990 against the Python model's 777,281 / 1,361,261 / 493,991 (hash collisions only); 0.64 us a
  lookup at -O2 on this host.
- Unit tests, TSan (concurrent insert + draft), ASan + UBSan: clean. The serve-loop hooks compile (see above).
- Reviewed by reading: the hook runs only inside `o.suffix_draft > 0 && !first_window`, the MTP first-token agreement and
  `graph_fits` still apply, a cancelled request's output is not inserted.

**Left.** The GPU screen and the replay (`bench/specs/24-a24-suffix-index.json`; judge the first request of each case), then
`STRATA_SUFFIX_INDEX_FILE` on the daily server. SYCL / Metal: `sycl/CMakeLists.txt` lists `suffix_index.cpp`; the SYCL and Metal
builds use their own `generate.cpp` copy without the hooks (not built here).

## A21 and F6: sampled requests

By code: `greedy = temperature <= 0` (`generate.cpp` `req_sp.greedy`), so a request at T 0.1 (opencode sub-agents) IS sampled:
the verifier uses the sampled path (a draft is kept only if it equals the token the target sampled for that row; at T 0.1 that
is the argmax almost always), and only with `STRATA_SPEC_COUPLED` / `STRATA_SPEC_PROB` does the drafter sample as well.
`STRATA_SPEC_MIN_TEMP=0.7` already exists (`MtpDrafter::set_draft_sampling`): requests under it keep the argmax drafter.
New: `strata::core::spec_sampled_drafts(greedy, T, min_temp)` in `spec_prob.hpp` names that rule (mtp.cpp calls it, same
behaviour) and `spec_prob_test` pins the six cases (T 0.1 sampled by default; kept argmax under 0.7).

F6 (Leviathan et al., arXiv 2211.17192; Chen et al., 2302.01318): the code is the standard rule: the draft is drawn from q on
its own Philox stream, accepted with `min(1, p/q)`, replaced by a draw from the normalised `(p - q)+` on a reject, a drafter
with no list (lookup, suffix) is a point mass (accept prob `p(d)`); `p` is the sampler's own post-chain list, not the raw softmax.
`spec_prob_test` (78 cases: chi-square per case, p == q accepts everything, drafts outside the support are always replaced)
passes. Forks' +5.5% (B70), +5.8% / +13.4% (5090): SYCL / CUDA, T 1.0, dense-ish agent turns; no HIP or hybrid-CPU number.
The T < 0.7 loss is measured upstream (3060, 5070), hence MIN_TEMP.

Interplay (read from `sampler.cu:981` and `gated_proposal_usable`): with the default pick gate the draft's reported
probability is that of the DRAWN token, so a window rule that reads it (DISCO, SPEC_CUM, `--spec-min-p`) sees the draw and is
inexact in distribution; `gated_proposal_usable` falls back to the older rule with a warning when a window rule is on. With
`STRATA_SPEC_PROB_GATE=top` the probability is q's largest entry (draw independent). Use `top` with DISCO / SPEC_CUM on sampled
requests. Greedy is exact either way (`disco_test` checks greedy equality).

Screen: `bench/specs/a21-spec-lowtemp.json` (T 0.1) and `12-spec-sampling.json` (T 1.0) exist; they need the GPU.

## F8 STRATA_SPEC_CUM

**Source** (android-imdad/Strata-7900xtx `f1bd09212a`, read through a summarising fetch of the commit patch; the later
`ef42c2fde9` lookup-agreement arm was not read and is not ported: the repo's fork notes call it "a wash with lookup chain"). Rule: keep draft j while `p_1 x .. x p_j >= S` (and `p_j >= --spec-min-p`); the MTP drafter loop stops
early, so the later steps are not even computed. An optional calibrated variant (`STRATA_SPEC_CAL`: per depth and probability
bin hit rates, prior weight 4, decay 0.995) was not better (+0.3-1.3% vs +1.8%). Measured on a 7900 XTX (Flash-Next IQ3_S):
+1.8% pinned, +6.4% short chat, +2.1% agent, `S` 0.5.

**Sound?** Yes as a marginal-benefit rule: adding row j costs one row for sure and gains one token with probability
`P(path_j)`, so keep it while `P(path_j) x (ms per token) >= row cost`, i.e. a threshold on the chain product, and the best S
is about row cost over cost per token (a lower S on a box where a row is cheap). The raw product assumes the MTP's softmax
equals acceptance, which the fork's own data (2nd drafts at p 0.8-0.9 held 57%) and DraftPolicy's `draft_rate` comment
contradict; S is tuned, which absorbs part of it. `--spec-min-p` alone ignores the earlier drafts.

**Port.** `DiscoModel::cumulative(s)`: weight 1 on feature 2 (the chain), bias 0, threshold sigmoid(s) (monotone, so the same cut
as `chain >= s`). `STRATA_SPEC_CUM=<s in (0,1]>` builds it in `DiscoPolicy::init_from_env`; `STRATA_DISCO` wins when both are
set; a bad value is ignored with a message. No new hook, no new engine path: it rides on the DISCO chain rule, the trace
logger and the feature state, so `STRATA_DISCO_TRACE` also works under it. Differences from the fork: the per-draft
`--spec-min-p` floor is not applied (S >= min-p implies it; for S < 0.5 the floor is gone), no calibration. Tests: `disco_test`
(decisions equal the product rule on 8,000 random chains, env parse, exactness against greedy on the mock target for S
0.1 / 0.3 / 0.5: 60 runs in all), `test_disco_fit.py::test_spec_cum_arm_is_the_chain_threshold`. `disco_fit.py` replays the
SPEC_CUM arms (0.1-0.7, and 0.2 / 0.3 + min-p 0.5) next to the static lengths, `--spec-min-p` and the oracle cut.

Screen spec: `bench/specs/b3-f8-spec-cum.json` (base / 0.2 / 0.3 / 0.5, text hash must equal base). Replay on a real trace
first; skip if no arm is >= 2% over `--spec-min-p 0.5`.

## EcoSpec cost-aware draft selection

**Source (arXiv 2607.12696, Xie, Liu et al., JD AI Infra / Tsinghua, July 2026; method read).** Score `S = P(path) / (marginal
new-expert cost + eps)` over the candidates of a draft TREE, with a buffer B of the experts already chosen; the expert set of a
candidate comes from a TRAINED lightweight predictor (a small decoder-only network trained with a cross-entropy against the
target router's top-k); the verifier is unchanged. Evidence: DeepSeek-V3.1 671B, Qwen3-235B-A22B (1.22x -> 1.36x, unique
experts per step 23.7 -> 20.5), GPT-OSS-120B, up to 1.62x on MTBench; NVIDIA, HBM-resident experts. The refuted "about 11%
relative" figure is not used.

Why it does not become an engine switch:

1. No tree: the engine verifies one MTP / lookup chain; "which gamma candidates" reduces to "how many drafts", which is DISCO's
   per-position threshold with a per-row miss cost (A22/A23) and DraftPolicy's cost-aware rule (`STRATA_SPEC_COST`).
2. No expert predictor: the plan measured next-layer expert recall at 19-21%; the paper's predictor is trained. The engine
   does have the cheap stand-in (an EMA of each row's new CPU misses = DISCO feature 6).
3. Headroom: the oracle cut above (1.0% short, 7.9% edit, 3.4% long) bounds every chain-stop rule. And the premise (expert
   scattering) is mild on this model: on the real routing trace (219 windows, 48 layers, 512 experts, top-10), row j adds
   68 / 60 / 53 / 51% new experts for rows 1-4 (vs rows before it; a window of 4 rows holds 70% of 4 x 10, of 12 rows 53%) and
   the new-expert count of a given row varies only 13-33% window to window (CV; e.g. row 3 of a 4-row window 253 +- 57 of 480).
   A row's cost is mostly a function of its index, which the EMA already has; a perfect predictor adds the +-20%.

**Simulator `tools/pancake/ecospec_sim.py`** (`test_ecospec_sim.py`, 5 tests): fits the DISCO logistic on the first 75% of a
window trace, then on the last 25% chooses the prefix length `k* = argmax (1 + sum q_j) / cost(k)` with `q_j` the product of
the model's per-draft scores and `cost(k)` from (eco-table) the `draft_costs.py` per-size table (the exact
`STRATA_SPEC_COST_TABLE` the engine loads; `draft_costs.build_tables()` is now a function, same output verified identical on a
layer-record trace, plus a record-5 fallback), (eco-ema) the one-row window plus the fitted row price and the expected new
misses, (eco-oracle) the true cost of this window cut at k (a perfect predictor). It prints them beside static lengths,
`--spec-min-p`, SPEC_CUM, DISCO's table and the oracle cut, the best baseline and the verdict (engine switch only if eco-ema
beats the best baseline by >= 2%). On synthetic windows with random +14 ms burst rows, the oracle-cost arm beats all others by
8.6% and eco-ema ties DISCO (-0.2%): the mechanism works; the synthetic costs say nothing about the real gain. **No real
window trace exists yet** (`STRATA_WINDOW_TRACE` was never set), so there is no real number; with the ceiling and the CVs
above the call is: **not built**. Re-run it on the DISCO trace run (same trace); build only if eco-ema >= +2%.

Not built, as the plan says: naive shared-expert + top-r self-drafting, DraftExpert (training).

## Tests run (all offline)

- C++ via CMake CPU-only (`build-cpu/cmake`, Ninja, -j2, 6 GB cap): 10 spec tests pass (above). Via plain g++: also
  `coupled_draft_test` (907,257 checks), `gated_proposal_test` (266), `spec_prob_test` (78 cases, 0 failed).
- Sanitizers: ASan + UBSan (`disco_test`, `suffix_index_test`), TSan (`suffix_index_test`): clean.
- HIP compile-only: `generate.cpp.o` and `mtp.cpp.o`, gfx1100: no errors (generate.cpp with the one A45 include added locally).
- Python (`python3 -m pytest`): `test_disco_fit.py` 6, `test_disco_parity.py` 1, `test_draft_report.py` 3, `test_suffix_sim.py`
  4 (+ `--self-test`), `test_ecospec_sim.py` 5, `test_abtest_cases.py` 6; `wtrace.py --selftest`.
- Real data: `suffix_sim.py` on 2,045 opencode requests (20 s); `suffix_index_test --simulate` on the same; the routing trace
  statistics above.
- Not run: any GPU / engine run, CUDA / SYCL / Metal compiles, abtest, the fit on a real window trace (none exists).

## Risk

All switches are opt-in and off by default; the default path is unchanged (`STRATA_SPEC_CUM` and `STRATA_DISCO` unset: the
policy is inactive, one `on()` branch). The only edits to engine sources are the `spec_sampled_drafts` rename in `mtp.cpp`
(same expression) and the new `DiscoModel::cumulative` (reached only through the env var). `STRATA_SPEC_CUM` replaces
`--spec-min-p` for MTP windows (like DISCO), so with S < 0.5 the per-draft floor is gone. DISCO / SPEC_CUM with
`STRATA_SPEC_PROB` need `STRATA_SPEC_PROB_GATE=top` to stay exact in distribution.

## Next

1. One GPU window: a trace run with `STRATA_WINDOW_TRACE` + `STRATA_DISCO_TRACE` + `--spec-min-p 0` over agent replays; then
   `disco_fit.py`, `draft_report.py`, `ecospec_sim.py` on it (all offline). That one trace decides A22, F8 and EcoSpec.
2. If a SPEC_CUM or DISCO arm is >= 2% over `--spec-min-p 0.5`: screen `b3-f8-spec-cum.json` / `24-a22-disco.json`.
3. A24: the screen and the replay; if it promotes, give index windows their own acceptance stats in DraftPolicy.
