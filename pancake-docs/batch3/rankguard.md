# rankguard - verify-path items: A4 rank guard + FOLD, F5 VERIFY_ECAP, F13 turbo guards (+ item 23 depth prior), HOBBIT proxy, ROUTER_FUSE audit

Branch `dev/p16/rankguard` (worktree `Strata-p16-rankguard`, from `dev/phase16-academic` 727eb8a62). 2026-11-11. CPU and
compile-only work: no GPU, no engine run, no model loaded. Every claim about engine behaviour below is from reading the
code (unconfirmed static analysis) unless a test or a compile is named.

## Commits

| commit | what |
| --- | --- |
| dba7e841e | A4: the pool's heavy/light split + FOLD decision moved into `defer_rank.hpp` (`defer_order_jobs`, same order and count as the inline code), new `defer_rank_test` (label cpu) |
| 73a8224dd | verify: one stderr line when `STRATA_ROUTER_FUSE=1` will not act, naming the first blocker |
| 802513d1a | `tools/pancake/gate_importance.py` (HOBBIT proxy, offline) + `test_gate_importance.py` |
| 6acd5400a | Metal stub: `route_prior_top10` with the item-23 parameters (the old stub was a different overload) |

## Builds and tests run (2026-11-11, this branch)

Passed:
- CPU configure (`-DSTRATA_ENABLE_HIP=OFF -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=ON`), targets `defer_rank_test`,
  `route_prior_scope_test`, `verify_ecap_test`: `ctest -R "defer_rank_test|route_prior_scope_test|verify_ecap_test"`
  3/3 pass (verify_ecap_test: 4000 random windows, 50974 entries dropped, all promises hold).
- `defer_rank_test` fails on three hand mutations of the header (fold bound +1: 718 failed checks; heavy>0 guard
  removed: 228; tie order reversed: 2), so it tests the rule, not just compiles.
- `conversation_disk_tier_test` (F5/F13 add switches to the disk key): built with g++ outside CMake (the test sits in a
  GPU-only CMake block and the HIP configure has tests off): 97 checks pass.
- `python -m pytest tools/pancake/test_gate_importance.py`: 7 pass.
- HIP configure (`-DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DCMAKE_HIP_ARCHITECTURES=gfx1100`, ROCm core-10.0),
  objects only: `strata_engine` `src/core/verify.cpp.o` and `src/core/expert_source.cpp.o`, `strata_kernels`
  `src/kernels/cuda/route_prior.cu.o` build with no error (after this branch's changes too).

Failed (not my items, not fixed here):
- HIP `strata` `src/program/generate.cpp.o`: 21 errors, all `CalibMeasure` / `calib_*` (A45): generate.cpp lacks
  `#include "strata/core/calibrate_ratio.hpp"` (lost in the batch-2 merges). With that header force-included
  (`-include`), generate.cpp passes `-fsyntax-only` in the HIP configuration, so the F5 / F13 parts of generate.cpp
  compile. Owner: A45 / the compile-gate agent.

Not run: CUDA and SYCL builds (no toolkit here), Metal (no Mac), any GPU test, any screen.

## A4 / item 28: rank guard `STRATA_VERIFY_DEFER_CPU_TOP=<n>` and its fix `STRATA_VERIFY_DEFER_CPU_FOLD=<n>`

What. With `STRATA_VERIFY_DEFER_CPU=1` the CPU rows of layer l join one layer late; `_BELOW` (0.1) keeps rows of weight
>= w on time; `_TOP=n` also keeps a CPU entry on time when it is among its token's top n by weight (rank over all k
entries, ties to the lower index; host `defer_heavy_entry` and GPU `copy_rows_by_weight_kernel` use the same rule, checked
by reading both). `_FOLD=n` (43b384aa1): a layer with at least one heavy job and at most n light jobs runs them in one pool
pass, the partial flag after it.

Source check (KTransformers, SOSP'25, Chen et al., PDF read): **ok in spirit, with differences**.
- Rule in the paper: the immediate experts are the top-I by routing score, the rest are deferred; a fixed count per model
  (DS-3 BF16 5+3, Int4 2+6; DS-2 2+4; QW-2 BF16 6+2, Int8 4+4), heuristic "defer the fewest that saturate the CPU, at
  least 2 immediate per layer"; never the last layer; decode only (not prefill: a batch covers almost every expert).
  Strata's `_TOP=2` is that "at least the top 2 immediate" floor; "never the last layer" exists (`l + 1 >= le_`).
- Differences: (1) KT defers by rank only; Strata defers only CPU misses (VRAM hits are always on time) and by weight
  (`_BELOW`) plus the rank floor. (2) KT adds a deferred output after the next layer's MoE (it misses layer k+1's attention
  and MoE input); Strata adds it before layer l+1's FFN read (it misses only the mixer): a smaller deviation and a
  smaller overlap window. (3) KT ran single-token decode at batch 1; Strata defers per row inside verify windows of up to
  8+ rows, where the rows' union of experts grows (KT's own reason for not deferring in prefill).
- What KT measured: dual Xeon 8452Y (AMX, DDR5, 220 GB/s per socket), A100 40 GB and RTX 4080 16 GB on PCIe 4.0;
  DeepSeek-V3 (256 experts, top-8), DeepSeek-V2.5 (160, top-6), Qwen2-57B-A14B (64, top-8); BF16/FP16 and Int4/Int8.
  Deferral alone: +33% decode on DS-3 BF16, up to +45% across setups. Quality: HumanEval / MBPP / GSM8K / StrategyQA within
  2 points (Table 2); DS-3 LiveBench at 2+6: -0.5% mean vs -13.3% for skipping the same experts. Nothing on 512 x top-10,
  IQ2_XS/Q2_0, RDNA3, PCIe 3.0 or speculative verify windows.
- New finding (arithmetic, tested in `defer_rank_test`): with k = 10 weights renormalised to 1 the largest is >= 0.1, so at
  the default `_BELOW 0.1` rank 0 is always heavy: `_TOP=1` is a no-op and `_TOP=2` only adds the rank-1 entry of a token
  whose second weight is below 0.1 (flat routing). The A4 measurement (edit +20.1%, short -6.9%) therefore comes from a
  minority of rows; a trace with weights would show how many.

FOLD check. Exact w.r.t. the same `_TOP/_BELOW` run, as claimed: post(l) and late_add choose rows by the weight rule on
the GPU, the pool writes every row before the flags whichever pass computes it, and folding only moves light jobs before
the partial flag (the second `run_jobs` gets 0 jobs). `defer_rank_test` checks on 4000 random windows that every heavy
entry's expert runs before the partial flag for fold 0/1/2/3/64 and that the order equals the old inline stable
partition. The helper (`defer_order_jobs`) replaces 10 lines of `expert_pool_dispatch_multi` with one call: same order,
same count, fold unset = the old count, so the default path is unchanged (by construction and by the reference check in
the test; not measured on the GPU). Minor: the start-up line "FOLD ignored (... needs STRATA_VERIFY_DEFER_CPU=1 ...)" is
also printed when deferral is on but the split was turned off by the device plan or a helper GPU (those print their own
line first).

Risk. Lossy-small (the rank guard); FOLD itself exact, timing only (post(l) waits for the folded light rows). Next: the
screen `bench/specs/10-rank-guard-fold.json` (text hashes of top2 and top2-fold* must match), then the w sweep.

## F5: `STRATA_VERIFY_ECAP=<n>` (2d44d31ec, lossy)

Source check (jmnargi/Strata-V100 24894e676b, 376355453f, GitHub commit pages read): **adapted, different parameter**.
jmnargi caps the whole speculative tail on the GPU: token 0 keeps K, the tail shares N-K entries for the window (dropped
picks get weight 0). Measured on 2x V100-PCIE-16GB, IQ3_S, 32K prompts, spec 4: decode median 91.8 -> 118.1 tok/s (N=20,
+17.7% paired) and 111.9 (N=16); no KL / quality measurement. 376355453f: N=16 with spec 8 (0.86 kept entries per draft
row) left 12 of 20 tool-calling turns truncated (0 of 20 without), while acceptance ROSE 56.7% -> 62.3%: acceptance is no
quality signal for this switch. Our port caps each draft row at n new CPU-missed experts and always keeps the VRAM
experts and every expert row 0 or an earlier row reads, so it is much milder than jmnargi at any n; `n` is not jmnargi's N.
Code check: the rule (`verify_ecap.hpp`) and its use in the dispatch read correctly; dropped entries become kind -2 zero
rows written before any flag (composes with the deferral; `defer_heavy_entry` only counts kind -1 entries, ranks over
all k on both host and GPU). With `--spec-split` each group caps against its own reads (consistent with per-call cost).
Tests: `verify_ecap_test` passes; HIP objects build. Risk: lossy; the tool-call gate first. Next: screen
`bench/specs/11-verify-ecap.json`, token forcing, task gate.

## F13: `STRATA_ROUTE_PRIOR_PROTECT/KEEP/KEEP_P/SCOPE` (3dd7c27b8) and item 23 `STRATA_ROUTE_PRIOR_DEPTH` (4f56ce535)

F13 source check (zsariboga/Strata 8d7042ad97, GitHub page read): **ok**. Same switches and meaning (protect the router's
first N picks; protect picks with p >= x of the top pick; a mapped per-window gate read by the kernel, prompt reads off
unless `_ON_PROMPT`). zsariboga writes lambda or 0 into the gate, we write a 1/0 scale: equivalent. The commit has no
measurements. lighttransport 2ea0c1e042 (from research_notes): residency-aware pruning and cache-aware substitution "do not
beat uniform top-3" on their quality gate, so spec 11-turbo-guards' uniform tail-skip arms at equal KL are the right
control. Code: guards off -> `own_phase` false from the start, the selection loop is the pre-F13 one (with the later NaN
guard). `route_prior.cu` builds for gfx1100; scope test passes; disk-tier test passes.

AcceptMoE source check (arXiv 2608.02989, abstract + HTML read): **mismatch; ours is "inspired by", not the method**.
- AcceptMoE builds one eligible expert SET per verification block: utility u_e = sum_t alpha_t r_{t,e} 1[e in K_t] with
  token weight alpha_t = p_d^0.5 N_d^-0.5 (p_d = offline commitment probability at depth d), S = the root's natural top-k
  (anchor) + the top n_er by utility (n_er = exp of the demand entropy, self-sizing); every token then routes by masked
  top-k inside S. Under offloading, non-resident members are pruned by increasing utility (rerouting budget, at least k
  left), and the anchor's non-resident experts may be pruned too.
- Ours (`STRATA_ROUTE_PRIOR_DEPTH=a`): a logit bonus lambda * (1 - a^pos) per row, 0 at position 0, fixed geometric a (not
  DraftPolicy's measured acceptance as item 23 says). No set, no self-sizing, no commitment weights.
- What it measured: Qwen3-30B-A3B-Instruct-2507, Qwen3-Coder-30B-A3B, GPT-OSS-120B (128 experts, top-8 / top-4 per their
  model cards); EAGLE-3 trees (5 steps, 64 draft tokens); SGLang batch 1; RTX PRO 6000 (all resident) 1.290x, RTX 5090
  with 48 expert slots per layer, BF16 / MXFP4, 2.06x; -0.27 accuracy points over 12 model-task pairs. No ablation isolates
  the depth weighting, so the plan's "+5-15% over fast-hip" for item 23 has no source number behind it.
- Code findings (static): (1) the depth is the row index `tb + row`; in a batch window (`--parallel`, rows = different
  slots' committed tokens) slot rows 1.. get a bonus although each is a committed row (still less than plain turbo, which
  gives the full bonus everywhere); (2) in a prompt read every row is committed, same remark. (3) Metal: the stub kept the
  7-parameter `route_prior_top10`, a separate overload, so the 9-parameter function verify.cpp calls had no Metal
  definition; fixed in 6acd5400a (g++ -Wmissing-declarations confirms the new stub matches the header).
Risk: lossy. Next: token forcing + task gate for F13 (spec 11-turbo-guards); for item 23 decide A1 vs A2 in the joint
quality batch; consider gating the depth bonus off in batch windows.

## HOBBIT gate-magnitude importance proxy (phase 16 order 2): offline only

Source check (arXiv 2411.01433 v2, via the deep-research findings and the HTML): proxy = normalised gate weight of the
selected experts; score s_i = w_0 + ... + w_{i-1} (s_0 = 0); s <= T1 high precision, T1 < s <= T2 low, above T2 skipped
(the skip inequality is our reading); T1 0.6 / T2 0.9 -> 67 / 30 / 3 on Mixtral-8x7B (8 experts, top-2); Pearson 0.99
between gate weight and ||G(x) E(x)||. The precision swap does not transfer (no copy below IQ2_XS / Q2_0).

Two structural points (no trace needed):
- At top-k the T2 cut is not comparable: the last entry has s = 1 - w_min >= 1 - 1/k, so at k = 10 T2 = 0.9 skips at least
  one entry per token by construction (>= 10% of entries vs HOBBIT's 3%). Synthetic Gaussian routers (512 experts, top-10,
  logit sigma 1 / 2): split 52/37/10% and 38/43/19%, ~7% of the weight mass skipped; 8 experts top-2 at sigma 2: 63/30/7%.
  Synthetic only, to show the mechanics.
- HOBBIT's correlation is Pearson(w, w * ||E||): w is on both sides, so it is high even when ||E|| is independent of w
  (test: independent lognormal norms give Pearson(w, ||E||) ~ 0 and Pearson(w, w*||E||) > 0.5). The tool reports both.

Tool: `tools/pancake/gate_importance.py` (TRACE.stwt | --dump-routing FILE | --npz FILE): correlation (needs per-entry
output norms), split and skip share per rank, smallest T2 per loss budget (1/3/5%), CPU-missed entries and distinct CPU
experts the refit T2 removes per window-layer, modeled decode saving = removed x --us-per-miss / window ms against the
plan's 2% gate. Existing `STRATA_ROUTE_TAIL_SKIP=R` is the uniform-rank version of the same skip; this would be the
cumulative-mass version.

Data: none usable here. The 219-window routing trace behind 2026-11-09-time-ceiling has zero weights and is not in git;
no `*.stwt` exists; `STRATA_ACT_DUMP_RAW` records carry no expert id or weight. So no measurement and **no engine switch**
(the plan's rule: only if the replay models >= 2%). To measure: one `--dump-routing` run (weights: split, refit) or a
`STRATA_WINDOW_TRACE` run (kind + ms: CPU impact and saving); the correlation needs a dump of ||E_e(x)|| per entry, which
the engine does not write today.

## Audit: STRATA_ROUTER_FUSE (item 48 / A33) is a silent no-op outside one narrow setup

`bf16_router_publish_arm` (native_bf16.cu) reads `STRATA_ROUTER_FUSE`, but verify.cpp only calls it when `rpub` holds
(verify.cpp:2505 on this branch, 2413-2425 on the plan's numbering): `STRATA_ROUTE_PUBLISH=1`, a decode window of 2..8
rows with the batched native router (512 experts, top-10), not all-resident, no `STRATA_VERIFY_DEVICE_PLAN`, no
`--remote-expert-opt`, no always-publish mode, **no `STRATA_VERIFY_DEFER_CPU`**, **`STRATA_ROUTE_PRIOR` unset** (so never in
the turbo arms), route-resident margin <= 0, on HIP no `STRATA_DOORBELL_STORE`, and `STRATA_ROUTE_FAST` not taking the step
first (rfast wins when it applies). With the fast-hip preset (deferral on) it never acts. 73a8224dd prints once at
`Verifier::init`, only when `STRATA_ROUTER_FUSE=1` is set, e.g. "STRATA_ROUTER_FUSE=1 will not act: STRATA_VERIFY_DEFER_CPU
is on (item 48 rides on the STRATA_ROUTE_PUBLISH launch)". Conditions known only per window (row count, always-publish set
after init, the lookahead router) are not checked there. Any A33 retest must run without deferral and without the prior,
or measures nothing.
