# Speculative decoding, MTP, drafting, verification and sampling: ideas from all Strata forks (state 2026-11-10)

Scope: every fork commit in the fork KB (`data/fork-kb/forks.sqlite`, `upstream_history = 0`, 2026-09-24 .. 2026-11-10, 43,685 commits) whose message matches drafting / MTP / speculation / lookup / acceptance / verify / sampler / grammar terms. 3,309 distinct subjects were left after removing those already in pancake's own git history (`git log --all`). From those, ~80 full messages were read, and 24 diffs were read with `gh api repos/<fork>/commits/<sha>`. Fork READMEs were grepped for acceptance and spec claims.

Target: pancake-strata (RX 7900 XT gfx1100 + Ryzen 7 5700G, Flash-Next IQ2_XS, preset `flashnext-iq2_xs-fast-hip`).

**Commit URLs.** Each link has the form `https://github.com/<fork>/commit/<sha>`. The KB's `commit_url` column points every commit at `Niko1221/Strata` (upstream's fork network). GitHub serves a fork-network commit from any fork in the network, so both forms open the same commit.

**Speed claims.** Every speed number below is the fork author's own claim on the author's hardware. None has been measured on our box yet.

## Pancake baseline (used for every "gain for pancake" estimate)

- Daily preset `configs/pancake/flashnext-iq2_xs-fast-hip.json` (local file): `--spec 4 --spec-min-p 0.5 --mtp <rt> --mtp-draft-vocab draft_vocab_en.bin --grammar on --pcie-frac 0`.
  - Not set in the preset: STRATA_SPEC_COST, STRATA_SPEC_PROB, STRATA_SPEC_COUPLED.
  - Engine defaults apply for the rest: `suffix_draft = 3` (suffix lookup on) and `--lookup-max-t` defaulting to 12 on the 8-row HIP build (`include/strata/core/lookup_default.hpp`, `src/program/generate.cpp:991, 4359-4374`). The long window can reach `kVerifyLongT` = 16 (`include/strata/kernels/verify_kernels.hpp:31`).
- Live acceptance, from the owner's opencode traffic in `pancake-turns.jsonl` (local file, 605 turns with draft counters; computed here):
  - 253,064 drafts offered, 190,863 accepted: **75.4% acceptance**.
  - Accepted drafts per output token: 0.634, so about **2.73 tokens per verify window**.
  - Per-turn acceptance: median 72.4%, p10 59.7%, p90 94.6%.
  - Median decode 121.8 tok/s.
  - About 0.56 rejected draft rows per window, roughly 17% of all verified rows.
- Window anatomy (README.md section 5, local):
  - Edit window 37.4 ms = GPU 15.8 + CPU expert misses 17.4 + MTP draft 2.5.
  - Short-chat window 24.8 ms = 15.9 + 5.2 + 2.2.
  - An extra verify row mostly costs CPU-missed experts, so a row that is then rejected is expensive here.
- Already measured in pancake (pancake-docs/RESEARCH_AND_FINDINGS.md lines 474-477, MOD_PLAN.md lines 801-805, local):
  - Flat: `--spec 3/5/6` and window thresholds.
  - **STRATA_SPEC_COST** (upstream's cost-aware chain rule): 112.0 / 98.3 against base 111.7 / 92.5, closed.
  - DFlash2: 115.9 short but 84.3 after a long prompt, closed.
  - Draft vocabularies: full and en flat, zh60k worse.
- Already in pancake's tree:
  - STRATA_MTP_PROXY (2-bit draft-head screen, on).
  - STRATA_MTP_EAGER (item 74).
  - STRATA_SPEC_COST + STRATA_SPEC_COST_TABLE (item 43).
  - DraftPolicy (MTP vs suffix by measured cost).
  - SuffixDrafter, `--lookup-chain`, `--lookup-max-t`.
  - STRATA_SPEC_PROB / STRATA_SPEC_COUPLED / STRATA_SPEC_PROB_GATE / STRATA_SPEC_PROB_DT / STRATA_SPEC_MIN_TEMP (`include/strata/core/spec_prob.hpp`).
  - STRATA_SPEC_GUMBEL, STRATA_MTP_TOP2 (diagnostic), STRATA_DF2_GATE.
  - STRATA_ROUTE_PRIOR_DEPTH (AcceptMoE), STRATA_VERIFY_DEFER_CPU(_TOP).
  - Grammar inside speculative decoding (phase 6).
  - The `src/spec/controller.cpp` cost-model controller, present but not wired into the serve loop.

## Q1. Which fork commits could raise accepted tokens per window: window sizing, stop rules, acceptance calibration?

### Takeaway
The only rule that is both new to pancake and measured on our GPU family is android-imdad's cumulative-probability draft stop on an RX 7900 XTX. It is STRATA_SPEC_CUM, optionally boosted by agreement with the suffix lookup (STRATA_SPEC_LOOKUP_Q). It is exact, a ~150-line port, and measured +1.8% to +6.4% on that card. Every other window-sizing commit (Ben1078, MistVVK, zsariboga, svgop W1, dys0re, htbmixbox) is a variant of the cost-aware rule pancake already has as STRATA_SPEC_COST, which measured flat here. Their numbers mainly show that the gain depends on the per-row cost of the machine.

### Cited Findings
- **android-imdad/Strata-7900xtx `f1bd09212a`, 2026-10-14: STRATA_SPEC_CUM=S / STRATA_SPEC_CAL=1 / STRATA_SPEC_LOG=<file>** — [commit](https://github.com/android-imdad/Strata-7900xtx/commit/f1bd09212adbdf431b393b54d527b8bffefea96c)
  - What it does: a draft stays in the window only while the product of the draft-layer probabilities up to it is ≥ S, beside the per-draft `--spec-min-p` floor. The drafter loop in `MtpDrafter::draft` stops at the same point, so no draft steps are wasted. STRATA_SPEC_CAL replaces p with an online-calibrated acceptance per (depth, probability band); see the diff in `src/core/mtp.cpp` and `src/program/generate.cpp`.
  - Size: 5 + 52 lines.
  - Exact: only which drafts are verified changes.
  - Claim (RX 7900 XTX, Flash-Next IQ3_S, six 1,000-token prompts × 300 tokens, pinned text, 3 alternating rounds): `--spec-min-p 0.5` alone 104.09 tok/s, with STRATA_SPEC_CUM=0.5 105.94 (+1.8%). The calibrated variant was not better (+0.3% to +1.3%).
  - Replay on 604 logged windows: verify costs 11.2 ms + 2.4 ms per token, and a draft step 0.74 ms. **An oracle stop would give +21%** (140.6 against 115.8 tok/s modelled), so the draft probabilities are what limits any stop rule.
  - The draft layer is calibrated at depth 1 and overconfident deeper: second drafts at p 0.8-0.9 held 57% of the time.
- android-imdad `4fe0492f53`, 2026-10-14 (README/docs, part 6): with `tools/strata_bench.py` and `--lookup-chain 3`, STRATA_SPEC_CUM=0.5 gave **+6.4% short chat (96.3 → 102.5 tok/s) and +2.1% agent session (97.2 → 99.2)** on the RX 7900 XTX. "Every round with the rule beat every round without it." — [commit](https://github.com/android-imdad/Strata-7900xtx/commit/4fe0492f53f428e8b58bfc99aca318ccb4cc5ca6); docs/RX7900XTX.md at `0633cd52c9`, read via gh api
- **android-imdad `ef42c2fde9`, 2026-10-14: STRATA_SPEC_LOOKUP_Q=q, STRATA_SPEC_DEPTH_DECAY=d, STRATA_SPEC_FEAT=1** — [commit](https://github.com/android-imdad/Strata-7900xtx/commit/ef42c2fde9313e70a5204dab79bdc850dc686fd0)
  - What it does: under SPEC_CUM, a draft that the suffix lookup (prompt + output so far) also proposes at the same place counts q in the product. Every other draft counts p·d^depth. The drafter gets the lookup proposal as a stop hint (`MtpDrafter::set_stop_hint`, `stop_q`). About 100 lines over 3 files.
  - Why lookup agreement: a logistic fit over 1,167 logged windows ranked it the strongest feature (+1.5), the draft log-probability +0.8, depth -0.4. The runner-up probability and the entropy added nothing.
  - Measured: CUM 0.3 + q 0.95 + d 0.9 replays +3.9% over CUM 0.5; measured +1.4% on the logged prompts and **+5.8% on six unseen prompts**. Exact.
  - Caveat in the same fork's docs (`0633cd52c9`): with `--lookup-chain 3` and suffix drafts on, the official benchmark was "a wash (short chat -2.5%, agent sessions +0.8%)", so it was not kept in the recommended settings. — [commit](https://github.com/android-imdad/Strata-7900xtx/commit/0633cd52c95798ddbe6ff0bd03d14ed8e1456cbc)
- **android-imdad `f9fac49321`, 2026-10-13**: `--spec 6` was fastest in every request on the RX 7900 XTX, +5-20% over `--spec 4`. The drafter's acceptance stays ~80% down to the fifth guess. — [commit](https://github.com/android-imdad/Strata-7900xtx/commit/f9fac49321a25a5ef4acc4ffd072b87b2a90aef2). This contradicts pancake's own flat result for `--spec 5/6` (RESEARCH_AND_FINDINGS.md line 474). The two machines differ: the XTX holds about 90% of experts in VRAM, while pancake pays CPU misses per extra row.
- **Ben1078/Strata `b5f98f4c68`, 2026-10-23: `--spec-policy cost`**, plus `--sfx-any M`, STRATA_SPEC_TUNE_FILE and STRATA_SPEC_CALIB — [commit](https://github.com/Ben1078/Strata/commit/b5f98f4c68f4e0af7df2b3d921b34e7d9add4016)
  - What it does: includes draft j while the calibrated probability that drafts 1..j are all accepted is ≥ θ = R*·dV (renewal-reward rule). R* is the decayed decode rate; dV is the online least-squares slope of verify ms over rows. The calibration table is (probability bin × depth), with separate greedy and sampled tables.
  - Its machine has an expensive row, like pancake: forced T=1/2/4/8 costs 26.0/36.1/49.4/88.0 ms.
  - Claim: +3.3% general and **+5.2% agent prompts**, pooled over two starts +1.0% / +4.0%, against `--spec 6 --spec-min-p 0.85`. Lossless. `--sfx-any` was measured a loser.
  - Status: the same idea as pancake's STRATA_SPEC_COST (`include/strata/spec/draft_policy.hpp:98`, "COST-AWARE MTP WINDOW"), which measured flat here. The θ = R*·dV threshold is a different decision rule from SPEC_COST's argmax of E/cost. It could be tried only if SPEC_COST is re-tested on edit windows.
- MistVVK/XeStrata `372bdac4ce`, 2026-10-22: the MTP window length is grown only while tokens per ms rise, from the measured window cost and the calibrated confidence ("DeepInfer confidence-scheduled verification"). The draft chain's stop uses the same predicate. Claim (Intel B70, IQ3_XXS, 1K, one run each): **78.5 → 85.0 tok/s**, the same 2.71 tokens per round with 233 → 217 drafts checked and draft time 5.9 → 4.9 ms/round. — [commit](https://github.com/MistVVK/XeStrata/commit/372bdac4ce7ae9e984e427c6612bb3ec32ce902a)
  - Follow-ups `796aab4461` and `174ada411e`: when experts come from an SSD file tier, window times are noisy and the auto rule picks 1-token windows (71 of 90 rounds at 32K), so the min-p rule became the default there. — [796aab4461](https://github.com/MistVVK/XeStrata/commit/796aab4461d8d2f870297475159a91af411e1f87), [174ada411e](https://github.com/MistVVK/XeStrata/commit/174ada411e342efd635e6af11550654d4aef6c4e)
  - Relevance: pancake's CPU-miss window times are also noisy, so a measured-cost rule can lock onto short windows. This is a plausible reason STRATA_SPEC_COST was flat here.
- zsariboga/Strata `f9667bf4f8`, 2026-10-15: STRATA_MTP_ADAPTIVE=1. DraftPolicy's learned expected commits per ms also picks the MTP window size, with every size measured first (largest first) and a 3% margin. Claim (GTX 1080 Ti, IQ3_S, `--spec 4 --spec-min-p 0.5`, pinned text, ABBA): steady stream 25.12/24.62 → 26.43/26.26 tok/s (~+5%); shifting stream level. Free-running runs "cannot resolve this" because acceptance varied 31-81% on the same prompts. — [commit](https://github.com/zsariboga/Strata/commit/f9667bf4f8d5b78f7d139dc4be7b559fff764a4f)
- svgop/Strata `ee435ae5ab`, 2026-11-02: STRATA_ADAPTIVE_SPEC=1 wires upstream's dead `src/spec/controller.cpp` (pancake has the same file, unwired). argmax_T of E[tokens]/step_ms(T) uses the session's per-depth acceptance EMAs and a measured wall fit of 8.4 + 1.57·rows ms. Cut-only, warm-up 100 tested drafts, min_gain 5%. Claim (RTX 5090): +0.9/-0.4/+3.0%, "battery-NEUTRAL, all inside noise"; window trimmed to T≈3.1-3.9 with acceptance ~85%. — [commit](https://github.com/svgop/Strata/commit/ee435ae5abe2e9a816c8232d2296a8263246a81f)
- svgop `0d8cbe1a60`, 2026-10-22: STRATA_SPEC_KEEP, a calibrated draft_cum keep rule using a 5×6 (position × probability-bin) acceptance table, cut-only and exact. Replay: 0.605 rows/window withheld. — [commit](https://github.com/svgop/Strata/commit/0d8cbe1a60382d2267587259a0c9cb1ac2690e5e). svgop `706a41adf3` later closed the keep-rule refit as "data-empty": the decile curve is "a drafter-model-pair property" and "no threshold family has room". — [commit](https://github.com/svgop/Strata/commit/706a41adf35f316c72fe9fd505870c39be3cc74f)
- svgop `0dfc3ea650` (F-228): per-depth acceptance data at min-p 0.5. Depths 3-5 are offered 35/17/8% of the time and accepted 85/68/75%. Lowering min-p to 0.35 lengthens the chain 2.20 → 2.31 but was negative (-1.75% M1), because "the marginal row costs full-graph T-scaling, not just expert bytes". — [commit](https://github.com/svgop/Strata/commit/0dfc3ea650ceb84c6caf8a83ade18ca94e47dc21). svgop `a52058c7b1`: spec depth 8 was -22%. — [commit](https://github.com/svgop/Strata/commit/a52058c7b1a2b429534871156b0064a6041e0c99)
- tctien342/strata-x `3ad15d54ec` and `7163271373`, 2026-10-19: `--mtp-adaptive` / STRATA_MTP_ADAPTIVE=1 is a per-request depth controller over depths 3-6. It grows above 80% token-weighted acceptance and shrinks below 65%, over three pure-MTP rounds. Its README reports live counters of 69.9% acceptance (249 of 356) and a final depth of 6 on 79% of requests. No A/B speed number. — [3ad15d54ec](https://github.com/tctien342/strata-x/commit/3ad15d54ec7bd8fb508a04046e73548b77a19acf), [7163271373](https://github.com/tctien342/strata-x/commit/7163271373b1e7aafdb56c5af396cb00e4a2a7a9)
- htbmixbox/Strata-890M `ba96856d17`: separate EMA cost curves for lookup and MTP windows in DraftPolicy (`--890m-source-cost`), so a slow lookup round does not inflate the MTP cost table. It has a unit test and no speed claim. — [commit](https://github.com/htbmixbox/Strata-890M/commit/ba96856d170c9f971152df3a49e3ca438745f874). Pancake's DraftPolicy uses one shared cost table per context bucket (`draft_policy.hpp`, `cost_` array).
- dys0re/strata-dcu `089c5b4dc9`: `--mtp-adaptive` with `include/strata/spec/mtp_budget.hpp`, a DeepInfer-style window selection from observed acceptance with costs paired to the draft that built each window. Validated only on a Hygon DCU (gfx936). — [commit](https://github.com/dys0re/strata-dcu/commit/089c5b4dc9cde355374bb9f0c85c1b854234552c)
- Spark-NV `f95f0bf6cb`: `--spec-min-p 0.7` on an RTX 5090. This was already in the earlier report "Forkuri Strata active noiembrie" (item 1.6); xyzzing's README also lists `--spec 3 --spec-min-p 0.70` +9.5% on an RX 7900 XTX. — [commit](https://github.com/Spark-NV/Strata/commit/f95f0bf6cb3f0756cb3aad4b2aca8bb2b4c0e5dc)

### Inferences
- **Cumulative stop (SPEC_CUM) on pancake.** Our live data show about 0.56 rejected rows per window (17% of rows). With CPU-missed experts per row, a perfect stop is worth at most ~5-6% of window time (estimate: 0.56 rows × ~3 ms per edit-window row ÷ ~30 ms). SPEC_CUM captured roughly a third of android-imdad's +21% oracle headroom on short chat and less on agent turns. Estimated pancake gain: **+1-3% decode, more on edit windows**, where a rejected row costs more CPU expert time.
  - Exact. Port size S (~150 lines).
  - Conflict: STRATA_SPEC_COST is off in our preset, so SPEC_CUM can sit beside `--spec-min-p` like at android-imdad.
- **Lookup agreement (SPEC_LOOKUP_Q).** Agreement with the suffix lookup as a stop feature is the one new piece of information. Pancake runs suffix drafts by default, which is where android-imdad found the rule "a wash". Test it as an arm, not as the main bet.
- **Window-sizing controllers (cost rules, adaptive depth, W1).** All of them add 0-5% where rows are cheap and their outcome is noisy. Pancake already measured the upstream member of this family as flat. Do not port more of them. If anything, re-measure STRATA_SPEC_COST on edit windows with a seeded cost table (item 43), since MistVVK showed that noisy window times break measured-cost rules.
- The `--spec 6` result on the XTX (+5-20%) against pancake's flat `--spec 5/6` fits the row-cost picture. SPEC_CUM with a deeper cap (`--spec 6`) could let confident chains run long while the cumulative rule trims the rest. That is a cheap two-arm test once SPEC_CUM exists.

### Gaps
- No fork measured SPEC_CUM on a hybrid CPU/GPU box where most extra-row cost is CPU expert misses. The 1-3% estimate is extrapolated.
- android-imdad's numbers are for IQ3_S with ~90% VRAM hit rate. Our IQ2_XS with a 4B MTP head and the en draft vocabulary may calibrate differently. A STRATA_SPEC_LOG-style per-window log (pancake has the window trace, item 64) would answer this offline before any GPU run.

## Q2. Which commits cut the cost per verify window: fewer experts in draft rows, draft overlap, cheaper draft step?

### Takeaway
The strongest cost cutter is jmnargi's STRATA_VERIFY_ECAP, a cap on the speculative tail's expert set. It is lossy and not in pancake, and fits pancake's main cost of CPU experts per extra row. It claims about +11% (wide error bar) on 2× V100 at 32K, and has a documented failure mode when the cap is set too tight. Draft-step micro-optimizations (DRAFT_EARLY overlap, SUFFIX_SKIP, MTP_HC_Q8, 2-bit draft head) are capped by pancake's 2.2-2.5 ms draft share; most are already present or worth at most ~1%.

### Cited Findings
- **jmnargi/Strata-V100 `24894e676b`, 2026-10-29: STRATA_VERIFY_ECAP=N** ("relaxed MTP", upstream PR #3545) — [commit](https://github.com/jmnargi/Strata-V100/commit/24894e676b1de0364b388cafd347702262885559)
  - What it does: token 0 keeps all K=10 picks. The speculative tail (rows 1..n-1) keeps only the top N-K entries by routing weight. Each dropped entry is redirected round-robin onto a kept expert at weight 0, so no expert outside the kept set is read and dropped rows contribute exactly 0.
  - Implementation: one kernel in `verify_kernels.cu` (+87 lines) and a hook in `verify.cpp` before `resident_plan`, decode windows only, graph-capturable, with an `ecap_parity` test.
  - **Lossy**: a draft row computed from fewer experts can flip acceptance and therefore the text.
- jmnargi `7c5041c358`, 2026-10-29 (docs): the cap acts only when the tail holds more than N-K entries. At the daily avg T 1.82-2.01, N=20 is a no-op and N=16 drops 2-4 entries. Measured on 2× V100-SXM2, 32K, 256 tokens, four requests per arm, service restarted per arm: off 71.5 / 66.2 against N=16 79.7 / 73.9 mean tok/s. That is **+11.5% in two drift-paired comparisons, but pooled medians only +2.8%** ("read that as about +11% with a wide error bar"). — [commit](https://github.com/jmnargi/Strata-V100/commit/7c5041c3584276863faef21c1cff19d6a5b9c256)
- jmnargi `376355453f`, 2026-10-31: refuses N < K + max_t - 1. With `--spec 8` and N=16, a 5-seed × 4-turn tool-calling loop ended 12 of 20 turns as truncated reasoning with no call (0 of 20 with the cap off). Acceptance was 62.3% against 56.7%. — [commit](https://github.com/jmnargi/Strata-V100/commit/376355453f68d1776c55df034ba8c5e2866a23cf)
  - **A too-tight cap breaks agent tool calls**, which is exactly pancake's workload. Any test must run the tool-call quality gate.
- yqYo1/Strata `751524ec68`, 2026-10-09: STRATA_GLM_MTP_KEEP=n fetches only the n best-scored missing experts of the GLM NextN draft's own route. The default of 2 is "never slower than skipping them all" on 2× V100. — [commit](https://github.com/yqYo1/Strata/commit/751524ec68139dbdbd19b98a66083b2575000984). This is GLM-only. The Flash-Next MTP head's experts are already in VRAM in pancake (the draft step is 2.2-2.5 ms GPU work, MOD_PLAN item 10 code study).
- mandoof1/Strata `b8004eb8d7`, 2026-10-20: STRATA_DRAFT_EARLY=1 splits `draft()` into `draft_launch` / `draft_finish`. The draft round is launched as soon as the window's accepted count is known, before the adaptive tier's finalize, the commit graph launch, the non-finite check and the emit, so host work overlaps the GPU draft. It is safe because the round reads nothing the commit graph writes (GDN state, QSA tails and the PLE history are captured by `Verifier::capture_commit`). Disabled with coupled drafts under penalties and on layer splits. No speed number in the commit. — [commit](https://github.com/mandoof1/Strata/commit/b8004eb8d758869d378441f5d371445bb3e386db)
- mandoof1 `868841cb32`, 2026-10-20: STRATA_DRAFT_SUFFIX_SKIP=1 ends the MTP chain after draft 0 when the next window will be a suffix-lookup window. Steps 1..n were "computed and thrown away". It predicts DraftPolicy's choice for every MTP size the chain could still produce and resumes the chain (`draft_resume`) if the forecast missed. Exact. No speed number. — [commit](https://github.com/mandoof1/Strata/commit/868841cb32d16dc9a1ce160a4aee6b3224254bdc)
- zsariboga `01fc2c80fc`, 2026-10-15: STRATA_MTP_HC_Q8=1 serves the draft layer's hyper-connection reads from load-time Q8_0 copies (19.9 MiB): draft step -0.03 ms, acceptance unchanged. STRATA_MTP_PRIO_CUDA gave "no net gain". — [commit](https://github.com/zsariboga/Strata/commit/01fc2c80fc4d00a1dd21bd188afcd72d2dfece04)
- eddoursul/Lamina `2ae47d0f44`, 2026-10-13: draft_proxy, a 2-bit copy of the draft head plus an exact rescore of candidates near the best. Draft head 110 → 51 µs (IQ3_S); round 1.49 → 1.31 ms; decode +0.2..+2.2%. — [commit](https://github.com/eddoursul/Lamina/commit/2ae47d0f444d03640073bc739ba694fd4df6b9c7). **Already in pancake** as STRATA_MTP_PROXY (basalt lineage, on by default).
- svgop `64e6ba455f` (F-294b): a two-stage head on the *full* output head is "DEAD on the real tensor". The flat logit top explodes the rescore band (558 → 65,536 rows per column, ~26% of the vocabulary), and the speed-up collapses from 2.4× to 1.33×. — [commit](https://github.com/svgop/Strata/commit/64e6ba455fbac3be373d1acc5a2f9a0151bd1830). This is a caveat for any extension of pancake's two-stage pick beyond the draft head.
- eddoursul `c463defdfa` (the round catches up only the accepted rows, upstream 3fd0460b) and `282334701a` (the commit graph runs beside the MTP draft, +0.9%): both are upstream ideas that pancake carries. Pancake has STRATA_MTP_CATCHUP_ALL as the opt-out in `src/core/mtp.cpp:91`. — [c463defdfa](https://github.com/eddoursul/Lamina/commit/c463defdfa7361c7854d150c3793c43da275e9d4), [282334701a](https://github.com/eddoursul/Lamina/commit/282334701acd4953e672be0b7f8f287bb0d82426)
- svgop `5f8fc1bf10` (F-229): STRATA_DRAFT_UNGATED, which enqueues all step graphs back to back like pancake's STRATA_MTP_EAGER, was **-2.73%** on an RTX 5090. The gate trims at ~2.2 of 6 steps, so ~3.8 discarded graphs per window (~0.19 ms GPU tail) outweigh the ~0.1 ms of host bubbles saved. — [commit](https://github.com/svgop/Strata/commit/5f8fc1bf1051385b1638c165dcf8c72ee10be316). This is a direct warning for pancake's STRATA_MTP_EAGER A/B (bench spec 07).
- maoyufeng1985/Strata `dc95cf3a82` and `3278b89460`, 2026-10-29 (Qwen3.6, P100 + 2 P104 helpers): the speculative step "pays rows": cost = 18.01 + 6.16·R ms, where R counts the drafts' own MTP rows. More drafts were strictly slower: 42.4 → 13.8 tok/s for N = 0 → 7 on a story prompt. On one card MTP lost even at 81% acceptance (31.4 → 27.7). — [dc95cf3a82](https://github.com/maoyufeng1985/Strata/commit/dc95cf3a8265db6ba26f643d00ff1c04409d6467), [3278b89460](https://github.com/maoyufeng1985/Strata/commit/3278b894607fb46ab70bce35d6d7a69b9a3feaae)
- lighttransport/Strata `9f0c19bc24`, 2026-10-26 (DS4.1-Flash, RX 9070 XT, CPU expert lane): lookup drafting made decode slower, 9.76 → 7.96 tok/s, because "a 5-token pass costs the CPU lane ~100 ms vs 54 ms for one token". Lookup + DEFER_RANK=3 gave 10.13, and DEFER_RANK=3 alone 11.70. — [commit](https://github.com/lighttransport/Strata/commit/9f0c19bc24007eea607b546d858bfc04f327761b)

### Inferences
- **ECAP on pancake.** It attacks pancake's dominant term directly: CPU experts named only by draft rows. It belongs in the same lossy family as pancake's STRATA_VERIFY_DEFER_CPU and STRATA_ROUTE_PRIOR_DEPTH, but works by dropping entries rather than deferring them or biasing the router.
  - Estimated pancake gain: on edit windows (17.4 ms CPU misses), dropping the lowest-weight tail entries at N ≈ K + 2·(avg T - 1) could be **+3-10% on edits**, at a quality cost of the same kind as `turbo` (2.9% of tokens differ).
  - Effort S-M: the kernel is self-contained. Pancake would need to check its CPU pool row layout (cpu::MAXT) and the deferral path for conflicts.
  - Must pass the tool-call gate. jmnargi's 12/20 truncated turns at a too-tight cap is the precedent.
  - Combine with or compare against DEFER_CPU_TOP, not on top of `turbo`.
- **Draft-step overlap (DRAFT_EARLY, SUFFIX_SKIP, MTP_HC_Q8).** The draft is 2.2-2.5 ms of 24.8-37.4 ms, so these are worth ≤1% each. DRAFT_EARLY is the only one that hides host time rather than shortening GPU work, so it is the best of the three (~0.5-1%, exact, M effort because pancake's serve loop has its own commit and emit order). SUFFIX_SKIP saves at most the discarded chain steps on lookup windows (~0.3% est.).
- **STRATA_MTP_EAGER may lose.** svgop's UNGATED result suggests pancake's coded STRATA_MTP_EAGER may lose for the same reason (discarded step graphs). Treat a negative A/B there as expected, not as a bug.
- **Extra rows are expensive with CPU experts.** maoyufeng's and lighttransport's results confirm the pancake-specific rule: when experts are on the CPU, an extra verify row costs real CPU time. Rules that cut low-probability rows (Q1) and caps on tail experts (ECAP) are the levers. Longer drafts and trees are not.

### Gaps
- ECAP has never been measured with a 4B MTP head, IQ2_XS experts or a CPU expert pool; jmnargi's rig keeps experts on two V100s. Its interaction with STRATA_VERIFY_DEFER_CPU (both rewrite the window's routing before the plan) is unknown.
- No fork published a measured number for DRAFT_EARLY or SUFFIX_SKIP.

## Q3. N-gram, lookup and prompt-lookahead drafting for agent and code-edit workloads

### Takeaway
Pancake already has the core of every n-gram idea the forks contain: SuffixDrafter, DraftPolicy arbitration, `--lookup-chain`, and lookup windows up to 12 by default (16 possible). New pieces are limited to three. First, a shared cross-request n-gram pool (tctien342's port of llama.cpp ngram-mod), mostly useful for multi-slot or repeated tasks. Second, `--lookup-max-t 16`, an env-only test. Third, a negative from svgop: on copy-heavy prompts the MTP head already beats the prompt-lookup ceiling in acceptance.

### Cited Findings
- eddoursul/Lamina `37c1d81e71`, 2026-09-29: `--spec-lookup N` builds windows of up to 8 tokens from the text's earlier occurrence of its last 16+ tokens. SuffixDrafter keeps its index across serve requests. Claim (fixed text, RTX 3090 + second GPU): a file returned with a rename went **178.9 → 225.0 t/s (IQ3_XXS) and 134.5 → 166.0 (UD-Q4_K_XL), 7.5 tokens per pass instead of 3.8**. Thinking and post-16K text unchanged. Exact. — [commit](https://github.com/eddoursul/Lamina/commit/37c1d81e7102cdd3b29d6c76f23f04387209a280). **Already in pancake** in its upstream form (SuffixDrafter + `--lookup-max-t`).
- zsariboga `712707942d`, 2026-10-25: `--lookup-max-t N` lets lookup windows grow to N while MTP keeps `--spec`. On a copy-heavy turn (142-line rename, RTX 2060S), 1477 of 1487 lookup drafts were accepted, so decode is bound by the window cap. Each window size is its own graph (~20-35 MiB VRAM); a size that finds no VRAM is dropped (`DraftPolicy::disable`). — [commit](https://github.com/zsariboga/Strata/commit/712707942da9401c0fa1c083de7af552f8a4f4f9)
- zsariboga `8a6b8942c3`, 2026-10-25: serial windows up to 16 tokens (kVerifyLongT), run as two 8-row groups. The drafter catches up past 8 rows through `MtpDrafter::draft_long`. — [commit](https://github.com/zsariboga/Strata/commit/8a6b8942c30ff9f3752300337bc07eab34175f9b). Pancake already has `kVerifyLongT` and `draft_long` (`src/core/mtp.cpp`, `generate.cpp`), and `--lookup-max-t 12 (P2c)` is queued with a measured +1.4..+3.9% (MOD_PLAN.md line 195).
- melchisedekko/Strata `4cfe19b0be`, 2026-10-28: lookup verify windows up to 12 rows, MTP kept at 8, wide windows caught up via prefill. No number. — [commit](https://github.com/melchisedekko/Strata/commit/4cfe19b0bedfe0f95eda783f6bc1ed9ac44d1ef3)
- **tctien342/strata-x `05c5a841e9`, `7163271373`, `44099cd96c`, 2026-10-19: `--ngram-mod` / STRATA_NGRAM_MOD=1** — [05c5a841e9](https://github.com/tctien342/strata-x/commit/05c5a841e910405b8e1e91f04ee32a99b2249a01), [7163271373](https://github.com/tctien342/strata-x/commit/7163271373b1e7aafdb56c5af396cb00e4a2a7a9), [44099cd96c](https://github.com/tctien342/strata-x/commit/44099cd96c487e32de4bec178f27b8bba5285cb4)
  - What it does: one engine-wide 16 MiB CPU hash table, ported from llama.cpp's ngram-mod (MIT). It proposes 48-64-token chains from prompt + committed history; the table stores continuations only, so one slot can serve a chain learned in another. Five windows under 25% acceptance clear a slot's feedback.
  - Solo decode: the n-gram chain replaces the MTP drafts for a window.
  - Claim: a copy fixture accepted 234 of 234 n-gram drafts. No tok/s A/B.
- svgop `b91fde1191` (F-291): "context-n-gram draft candidate DEAD at the falsifier". **The MTP drafter already beats the n-gram oracle on every copy-heavy prompt**: 88.7/89.4/81.4% accept against a 78.7/54.8/64.1% prompt-lookup ceiling; controls have a ~0% ceiling. — [commit](https://github.com/svgop/Strata/commit/b91fde1191a09dae979c07e92feea0a062352f8a)
- lighttransport `1a7a35d0b3`, 2026-10-11: STRATA_GLM_MTP_LOOKUP_POLICY=cost takes a lookup draft only when its expected tokens per ms beat MTP's, cut to the best length, with acceptance learned per match-length bucket. Claim (GLM-5.3 Q2, 512 tokens × 3; MTP only / any match replaces / cost): edit-cpp 20.1/21.5/20.9, **edit-py 22.6/19.9/22.1**, 8K review 14.7/14.3/15.3. — [commit](https://github.com/lighttransport/Strata/commit/1a7a35d0b3de1a098c0f457cdef927ee93c7292b)
  - Takeaway: naive "always take the lookup" can lose 12% on edits; cost arbitration recovers it. Pancake's DraftPolicy already arbitrates by cost.
- lighttransport `5dbfebe93f`, 2026-10-23 (DS4.1, 2× V100): STRATA_DS41_LOOKUP=1 is an adaptive history lookup before DSpark. Wikitext E4 30.3 → 58.0 tok/s; code within noise. — [commit](https://github.com/lighttransport/Strata/commit/5dbfebe93f66db6e5ea0d3254440703ae7d840b7)
- RandoTeam/4gbVramStrata `a74eefeb83`, 2026-10-06 (strata-poly): a gate measures tokens/ms of rounds with and without asking the draft block (exploring every 8th/24th/96th round). Lookup drafts are not gated because "they cost nothing to make". Claim (RTX 5090, all in VRAM): Ornith code 183 → 197, story 154 → 184; Qwen3.6 and Gemma 4 flat. — [commit](https://github.com/RandoTeam/4gbVramStrata/commit/a74eefeb8315075a27e9ebd30cb25e786eae2681). Relevant only to pancake's strata-poly presets (CUDA-only, not on the current 7900 XT box).

### Inferences
- For opencode edits on pancake, the lookup path is already the main lever and is already enabled. Two env-only tests:
  - `--lookup-max-t 16` against the default 12 (on a rename-style edit, android/zsariboga data show near-100% lookup acceptance, so the 12-row cap may bind).
  - `--lookup-chain 3`, which android-imdad's recommended XTX settings use and pancake leaves at 0.
  - Estimated gain: 0-4% on edit turns, nothing on chat.
- **ngram-mod (shared pool)** adds cross-request memory, which pancake's poly path had as `--suffix-drafts 256` ("repeated task 90 → 98-107 tok/s", MOD_PLAN.md line 527). For the main engine it is a port of M size. It pays mainly when the same file or tool output reappears across conversations, which opencode does often (re-reads ~18% of turn time, MOD_PLAN.md line 110). It is worth a trace replay over `pancake-turns.jsonl`-like data before porting.
- svgop's negative means that n-gram drafting does not beat MTP on acceptance. It wins only on window length, when the copy is long and the cap allows more rows. On pancake each extra row costs CPU experts, so very long lookup windows pay only when acceptance is near 100%, which is the case for verbatim copies.

### Gaps
- No fork measured ngram-mod with tok/s against a suffix-drafter baseline.
- The fraction of pancake windows that are lookup windows and how often they hit the 12-row cap are not in `pancake-turns.jsonl`. The window trace (item 64) could answer it.

## Q4. Sampling at temperature > 0: exact speculative sampling, coupled drafts, draft temperature, lossy acceptance

### Takeaway
Pancake already has the full upstream exact speculative-sampling stack (STRATA_SPEC_PROB, STRATA_SPEC_COUPLED, STRATA_SPEC_PROB_GATE=top, STRATA_SPEC_PROB_DT), but its preset turns none of it on. Two forks measured it as default-worthy on sampled agent traffic: riceharvest +5.5% on the Intel B70, and svgop +5.8% plus +13.4% from a 0.85 draft temperature on an RTX 5090. svgop's own earlier run called SPEC_PROB neutral. The test is env-only and matters only if opencode's requests are sampled. Lossy typical or margin acceptance (zsariboga) exists but has no published speed number.

### Cited Findings
- **riceharvest/strata-b70-only `7389a10f2c`, 2026-11-01**: STRATA_SPEC_PROB=1 + STRATA_SPEC_COUPLED=1 + STRATA_SPEC_PROB_GATE=top made the default on the B70. The drafter samples its own q; the verifier accepts with min(1, p/q) and redraws from (p-q)+. "The `top` gate: exact rejection sampling; the default pick gate needs --pcie-frac 0 and no STRATA_SPEC_COST, else it keeps an older, inexact rule." Claim (six agent-shaped turns, thinking on, temperature 1.0, 1,000 tokens each): **82.1/82.9/83.5 → 88.1/88.2 tok/s (+5.5%)**; greedy unchanged. — [commit](https://github.com/riceharvest/strata-b70-only/commit/7389a10f2c093e121db0618fe69f3bc8d7f9d253)
- svgop `0355c3164f` (F-282), 2026-10-23: SPEC_PROB=1 adopted. M1 217.8 against control 205.8 (**+5.8%**), concurrency +4.4%, MMLU mean 94.3. — [commit](https://github.com/svgop/Strata/commit/0355c3164f8d2ec5e73a0cb4a438d07101fd3040). This contradicts svgop `65e4bdae03` (F-235, 2026-10-18): "SPEC_PROB closed negative". — [commit](https://github.com/svgop/Strata/commit/65e4bdae03e07ce16a6f9cd12ec58df4e84069ca). The later, larger battery won, but the conflict shows the effect is within run-to-run noise on some lanes.
- svgop `b59c0e6bbc` (F-201), 2026-10-15: STRATA_SPEC_DRAFT_TEMP=0.85, the coupled drafter sampling at 0.85× the request temperature ("proposals-only/exact-match so no MMLU class"). A sweep over 0.45-0.90 peaked at 0.85: **M1 +13.4%, pair +9.8%, conc +7.5%** (RTX 5090, IQ3_S). — [commit](https://github.com/svgop/Strata/commit/b59c0e6bbc35d37e1bc6cc6495e020f2582f9584). Pancake's equivalent switch is **STRATA_SPEC_PROB_DT** (`include/strata/core/spec_prob.hpp`, "the drafter's chain runs at x * the request's temperature"). Already have it, never measured here.
- svgop `0175c36cb7` (F-184): coupled slot drafts (`--batch-mtp`), pair +11.5%. This applies to batch serving only; pancake serves one slot. — [commit](https://github.com/svgop/Strata/commit/0175c36cb73d14aabcc186419b6599a1dc0a6495)
- Pancake's own note in `include/strata/core/spec_prob.hpp` (STRATA_SPEC_MIN_TEMP): "Measured on the RTX 3060 and 5070: sampled drafting gains at temperature 1.0 and loses below about 0.7, where the draft head's argmax is the better guess" (local file, upstream measurement).
- svgop `8731e33c77` (F-230): "prefill-bits changes pay ~-2.5% decode acceptance tax". Q8 activations in dense MMQ shifted -2.1 pp acceptance, because exact-match coupled MTP flips borderline decisions. — [commit](https://github.com/svgop/Strata/commit/8731e33c770626fb1d31b1bd05f667fa3681c90f). This is a warning that any numerics change in pancake's prompt or verify path can cost acceptance.
- zsariboga `ee9d60f5e6`, 2026-10-07: STRATA_TYPICAL="eps,delta" (Medusa typical acceptance) or STRATA_TYPICAL_MARGIN=r ("keep when p ≥ r·p_top1", arXiv 2609.02897). STRATA_TYPICAL_THINK=1 restricts it to `<think>`; STRATA_TYPICAL_ANSWER=1 makes it strict inside `<tool_call>` and code fences. **Lossy**, with no speed claim in the commit. — [commit](https://github.com/zsariboga/Strata/commit/ee9d60f5e6313b7a0727695db871d4327f7b5568)
  - zsariboga `6c4c468404`: the margin rule can keep a loop forever (a loop token at p=0.4 against an exit at 0.6 is always kept), so the lossy rules are turned off while the reply repeats a 1-16-token pattern. — [commit](https://github.com/zsariboga/Strata/commit/6c4c46840498b8e4b34aac90d91d47d9e4ab8208)
- sebastianrcnt/Strata `2efa2f9096`, 2026-10-04: applies the target's token bans to MTP draft sampling. The bans are mapped onto the reduced draft-head rows before the draft probability and floor are computed, so the drafter stops proposing banned tokens that verification always rejects. — [commit](https://github.com/sebastianrcnt/Strata/commit/2efa2f909673baf1792cb6555e718b33236fd4db)
- riceharvest `8d8d425958`: top_k block lists in the sampled draft sampler and a code-specific draft vocabulary (`draft_vocab_code.bin`). — [commit](https://github.com/riceharvest/strata-b70-only/commit/8d8d4259584bb0cf9a5636c6f4615f5fca41d6d5)

### Inferences
- **Free test for sampled requests.** If opencode sends temperature ≥ ~0.7, `STRATA_SPEC_PROB=1 STRATA_SPEC_COUPLED=1 STRATA_SPEC_PROB_GATE=top` plus `STRATA_SPEC_PROB_DT=0.85` is a free, exact env test.
  - Estimated gain: **+3-8% decode on sampled turns, 0 on greedy**. The B70 and RTX 5090 numbers come from all-GPU boxes; on pancake the extra accepted tokens save CPU expert work per token, so the relative gain could be similar or larger.
  - Pancake already satisfies riceharvest's exactness condition: `--pcie-frac 0` and no STRATA_SPEC_COST.
- RESEARCH_AND_FINDINGS.md line 594 records that the owner "moves to higher temperature". That makes this the cheapest acceptance lever in the whole list.
- Draft-side token bans are a small exactness-neutral port. They matter only if pancake uses bans (`logit_bias` or the loop guard), so the gain is low.
- **Typical/margin acceptance is lossy.** It belongs with the parked "Typical acceptance on thinking only (Medusa)" item (RESEARCH_AND_FINDINGS.md line 662). zsariboga's code is the ready-made implementation of that parked idea, including the tool-call/code strictness and the loop-guard interaction.

### Gaps
- `pancake-turns.jsonl` does not record request temperature, so the share of sampled traffic is unknown. Check the opencode provider config or the serve log's GEN lines.
- No fork measured STRATA_SPEC_PROB on a hybrid CPU/GPU MoE.

## Q5. Other drafters and draft shapes: DFlash2, DSpark/confidence heads, draft trees, deeper chains, top-2

### Takeaway
No fork shows a drafter that beats the MTP head on a consumer card with experts partly on the CPU. DFlash2 lost on several boxes, and pancake closed it (84.3 after a long prompt). Draft trees were priced "arithmetically dead under exact-match" (svgop). Deeper chains and batch draft depth lose whenever a row is expensive. The positive evidence is all on all-GPU boxes.

### Cited Findings
- svgop `af29fb35c6` (T17C): "tree verify arithmetically DEAD under exact-match (delete from all queues)". — [commit](https://github.com/svgop/Strata/commit/af29fb35c65e47e426ad12bc84a3aa3701efc50c). svgop `fa6a1e228a` (T50): "depth/trees/row-batching/MoE-spec all priced dead with mechanisms". — [commit](https://github.com/svgop/Strata/commit/fa6a1e228a710de2e6ebce59cc08896d135c2ab7)
- unimatrix099/Strata `1ad25e4d67`, 2026-10-08: the STRATA_MTP_TOP2=1 diagnostic finds the drafter's second choice is the missed bonus token in 31% of rejections (2× RX 7900 XTX, pipelined windows). — [commit](https://github.com/unimatrix099/Strata/commit/1ad25e4d679a17910473a3c9c4ee4c7271a41919). Pancake has the same diagnostic (`src/core/mtp.cpp:79, 577`). A top-2 branch would add a full row of experts, so it hits the same row-cost wall as a tree.
- svgop `89e63ce8c9` (F-352): batch draft depth D=2 was falsified ("rows +19%, window +20-22%, pair -7, conc -6"). — [commit](https://github.com/svgop/Strata/commit/89e63ce8c9f64d28973812301ef3dc865177e59d)
- dongnh311/Strata `08dd52cb59`, 2026-11-03: on an RTX 3060, "MTP stays faster - 65-68 tok/s warm vs 37-52" for DFlash2, so the run config pins drafting=mtp. — [commit](https://github.com/dongnh311/Strata/commit/08dd52cb598ebe1c2fc83a3148903d649d0e4714). svgop `5a6e6b030e`: the stock UD config's full 106K draft vocabulary caused an "acceptance collapse" (avg T≈1.96) compared with the en 40K subset. — [commit](https://github.com/svgop/Strata/commit/5a6e6b030e399d3b06a0dfffd2227e9e3b5b20bf). Pancake uses the en subset already.
- lighttransport `739100bb56`, 2026-11-01 (MiniMax M3.1): DSpark's confidence head (conf_proj) cuts low-confidence drafts before verification (`pmin=`); best 24.5-25.0 tok/s at draft 6, pmin 0.7. — [commit](https://github.com/lighttransport/Strata/commit/739100bb56e3ccf4837b5cb0ee97dfa841156263). lighttransport `05d3ff25a3`: STRATA_GLM_DFLASH_ADAPTIVE_DEPTH, which stops before the first token whose logit gap to the runner-up is below a threshold; timing "not qualified". — [commit](https://github.com/lighttransport/Strata/commit/05d3ff25a30ef2afec314e3672a1b8211686a8ef)
  - Both are a stop rule using a learned or logit-gap confidence. Our MTP head has no confidence head; android-imdad found the runner-up gap adds nothing (Q1).
- samnolak/Strata `2adacdbc05` (draft chain up to 11 outputs for `--pipeline-windows 3`) and `bfee1bf282` (SwarmSchedule, DSpark's global verify schedule across pipelined slot groups): multi-GPU / pipelined only. — [2adacdbc05](https://github.com/samnolak/Strata/commit/2adacdbc0555fc0c60e9111b2c60fbad9d7a083a), [bfee1bf282](https://github.com/samnolak/Strata/commit/bfee1bf2829dc9a01c8e05480e944ad0d4602374)
- bsorensen110/Strata `15255e4953`: spare verify rows ranked across batch slots by DFlash2 chain probability (batch only). — [commit](https://github.com/bsorensen110/Strata/commit/15255e4953e6d5643210724d2bdd17f54ff94a99). Already covered in "Forkuri Strata active noiembrie" (bsorensen110's DFlash2 wave).
- VlDubr/strata-mi50 `c3d0ca379c`: a script-specific draft vocabulary (Cyrillic subset) gave +14.8% on Cyrillic text; `--spec 2 --spec-min-p 0.8` on 4× MI50. — [commit](https://github.com/VlDubr/strata-mi50/commit/c3d0ca379cd40cda812f446d20ad77becc56d9d6)
  - riceharvest's `draft_vocab_code.bin` (Q4) points the same way for code: a draft vocabulary matched to the workload.
  - Pancake measured en vs full as flat. A code-only subset is untested here and could be cheap: smaller head, same acceptance on code.

### Inferences
- Do not invest in trees, top-2 branches, DFlash2 or deeper fixed chains for pancake. Every fork measurement with expensive rows was negative.
- The only drafter-side change with a plausible gain is a **code-focused draft vocabulary** (riceharvest) for opencode turns, plus SPEC_PROB_DT for sampled turns (Q4).
  - Code vocabulary: estimated 0-2%, because the draft step is 2.2-2.5 ms and the proxy already screens the head. Exact.
- A retrained drafter (svgop's DAgger and FastMTP lane, [`a5d7f2a1f8`](https://github.com/svgop/Strata/commit/a5d7f2a1f824eb0daae791b218121791cf344d68) and others) is the only way past the probability ceiling that android-imdad's oracle (+21%) exposes. svgop's live result was rejected (rt2 0.520/1.47 against live 0.587/2.14), so it is high effort and unproven.

### Gaps
- No fork measured a code-only draft vocabulary on Flash-Next with numbers.

## Q6. Grammar, tool-call, rollback/rewind and sampler-kernel paths

### Takeaway
Pancake's grammar runs inside the engine (phase 6, "0.1-0.2% per window"). The one grammar bug found in a fork (jess-sol: suffix windows wider than one mask reply) belongs to a server-side mask protocol that pancake does not use. Rewind/rollback commits in forks are about conversation parking or GDN snapshots and add no decode speed. Sampler-kernel items were already covered in the earlier fork reports.

### Cited Findings
- jess-sol/Strata `53016163f3`, 2026-11-07: a masked decode window is cut to kVerifyMaxT rows (`mask_window_cut`). Suffix drafts built 12-row windows, and the server's MKN mask reply ended the request on "MKN rows above kVerifyMaxT" (finish_reason=length a few tokens in). The fix keeps the chained drafts that fit and never cuts a merged window's prompt rows. — [commit](https://github.com/jess-sol/Strata/commit/53016163f307943ce8dd5b96245a5c710b1c53f0). Pancake's grammar is engine-side (`src/program/grammar/guard.hpp`, `--grammar on`, with no MKN protocol in `generate.cpp`), so this is not applicable. Still worth a test that a 12-16-row lookup window under `--grammar on` with a tool schema completes.
- zsariboga `eab13fc6af`: STRATA_LOOP_BREAK=1. Three copies of a ≥256-token span enter a 1024-token break mode, with the copy-continuing token at -100 at row 0 of each window. Sampler bans also apply in `sample_tokens_spec`. — [commit](https://github.com/zsariboga/Strata/commit/eab13fc6af338a3dfada385c381c4103b6ffff01). Pancake has `serve/loop_guard.py`; long verbatim loops are exactly where lookup drafting accepts 100%, so a loop costs little decode time but wastes the turn.
- Asofwar/Strata-Turbo `962e795d50` (from the commit list): "a GDN model cannot rewind: 209,688 tokens took 72 s". The resample pass extends the engine's state instead of re-reading. This is a serve-path issue, not decode. (Message only, diff not read.) — [commit](https://github.com/Asofwar/Strata-Turbo/commit/962e795d50ea139d02701c8335690747485c1403)
- Rollback cost in decode: pancake replays GDN on rejected rows (MOD_PLAN item 20, closed by code study: "snapshots per row would be ~2 MB per GDN layer per row - replay stays"). No fork commit proposes a cheaper scheme for Flash-Next; the Cudila nemotron engine's "exact Mamba rollback" is a different model family.
- Sampler kernels: rpcribari's "eight logit loads in flight", the top-k lexicographic threshold, and svgop's STRATA_TYPICAL_DEVICE / VERIFY_ASYNC_LAUNCH (+4.9..+9% on an RTX 5090, svgop `4d81eb48dd`) were already covered or are upstream. Pancake has VERIFY_ASYNC_LAUNCH in `src/core/verify.cpp` / `arch_defaults.cpp`; TYPICAL_DEVICE is not in pancake's tree. — [svgop 4d81eb48dd](https://github.com/svgop/Strata/commit/4d81eb48dd0095aa2687fefaa5d529a87328c2d6)

### Inferences
- Nothing in this group is a speed lever for pancake.
- Add one correctness test to the grammar suite: a long lookup window (12-16 rows) under a strict tool schema. Pancake enabled both defaults independently: lookup_max_t 12 and grammar on in the IQ2_XS presets.

### Gaps
- STRATA_TYPICAL_DEVICE's meaning was not read: the svgop commit is a battery note, not code.

## Q7. Ranked list for pancake (what to test first)

### Takeaway
Two env-only tests come first: sampled-request speculative sampling, and lookup window and chain settings. They are followed by two small ports. One is exact (STRATA_SPEC_CUM with the lookup-agreement arm). The other is the only lossy item aimed at pancake's dominant cost (STRATA_VERIFY_ECAP), and it must pass the tool-call gate. Everything else is ≤1% or measured dead elsewhere.

### Cited Findings
Each row repeats the claim with its source from Q1-Q6.

| # | Idea | Fork / commit | Env / flag | Claim (hardware, workload) | Exact? | In pancake? | Est. gain for pancake | Effort |
|---|---|---|---|---|---|---|---|---|
| 1 | Exact speculative sampling + draft temperature for sampled turns | riceharvest [`7389a10f2c`](https://github.com/riceharvest/strata-b70-only/commit/7389a10f2c093e121db0618fe69f3bc8d7f9d253); svgop [`0355c3164f`](https://github.com/svgop/Strata/commit/0355c3164f8d2ec5e73a0cb4a438d07101fd3040), [`b59c0e6bbc`](https://github.com/svgop/Strata/commit/b59c0e6bbc35d37e1bc6cc6495e020f2582f9584) | `STRATA_SPEC_PROB=1 STRATA_SPEC_COUPLED=1 STRATA_SPEC_PROB_GATE=top STRATA_SPEC_PROB_DT=0.85` | +5.5% (B70, agent turns T=1.0); +5.8% and +13.4% with DT 0.85 (RTX 5090); svgop also once "neutral" | exact | yes, off in preset | +3-8% on sampled turns, 0 greedy | env only |
| 2 | Lookup windows 16 rows; lookup chain | zsariboga [`712707942d`](https://github.com/zsariboga/Strata/commit/712707942da9401c0fa1c083de7af552f8a4f4f9), [`8a6b8942c3`](https://github.com/zsariboga/Strata/commit/8a6b8942c30ff9f3752300337bc07eab34175f9b); android-imdad settings | `--lookup-max-t 16`; `--lookup-chain 3` | 1477/1487 lookup drafts accepted on a rename (2060S); +1.4..+3.9% for 12 on pancake already | exact | yes (default 12, chain 0) | 0-4% on edits | env only |
| 3 | Cumulative draft stop (+ lookup agreement arm) | android-imdad [`f1bd09212a`](https://github.com/android-imdad/Strata-7900xtx/commit/f1bd09212adbdf431b393b54d527b8bffefea96c), [`ef42c2fde9`](https://github.com/android-imdad/Strata-7900xtx/commit/ef42c2fde9313e70a5204dab79bdc850dc686fd0) | `STRATA_SPEC_CUM=0.5` (`STRATA_SPEC_LOOKUP_Q=0.95 STRATA_SPEC_DEPTH_DECAY=0.9 STRATA_SPEC_CUM=0.3`) | +1.8% pinned, +6.4% chat / +2.1% agent (RX 7900 XTX); lookup arm +5.8% unseen prompts but a wash with lookup chain | exact | no | +1-3% (more on edits) | S, ~150 lines |
| 4 | Expert cap on the speculative tail | jmnargi [`24894e676b`](https://github.com/jmnargi/Strata-V100/commit/24894e676b1de0364b388cafd347702262885559), [`7c5041c358`](https://github.com/jmnargi/Strata-V100/commit/7c5041c3584276863faef21c1cff19d6a5b9c256), [`376355453f`](https://github.com/jmnargi/Strata-V100/commit/376355453f68d1776c55df034ba8c5e2866a23cf) | `STRATA_VERIFY_ECAP=N` (N ≥ K + max_t - 1) | +11.5% paired / +2.8% pooled medians (2× V100, 32K); too tight a cap broke 12/20 tool turns | **lossy** | no | +3-10% on edits, quality gate required | S-M |
| 5 | Draft launched before commit/emit | mandoof1 [`b8004eb8d7`](https://github.com/mandoof1/Strata/commit/b8004eb8d758869d378441f5d371445bb3e386db) | `STRATA_DRAFT_EARLY=1` | none published | exact | no (MTP_EAGER is different) | 0.5-1% | M |
| 6 | Stop the MTP chain when the next window is a lookup window | mandoof1 [`868841cb32`](https://github.com/mandoof1/Strata/commit/868841cb32d16dc9a1ce160a4aee6b3224254bdc) | `STRATA_DRAFT_SUFFIX_SKIP=1` | none published | exact | no | ≤0.5% | M |
| 7 | Shared cross-request n-gram pool | tctien342 [`05c5a841e9`](https://github.com/tctien342/strata-x/commit/05c5a841e910405b8e1e91f04ee32a99b2249a01), [`7163271373`](https://github.com/tctien342/strata-x/commit/7163271373b1e7aafdb56c5af396cb00e4a2a7a9) | `--ngram-mod` / `STRATA_NGRAM_MOD=1` | 234/234 accepted on a copy fixture; no tok/s | exact | partly (poly `--suffix-drafts`) | 0-3% on repeated tasks (replay first) | M |
| 8 | Separate lookup vs MTP cost curves in DraftPolicy | htbmixbox [`ba96856d17`](https://github.com/htbmixbox/Strata-890M/commit/ba96856d170c9f971152df3a49e3ca438745f874) | `--890m-source-cost` | unit test only | exact | no | 0-1% (better MTP/lookup choice) | S |
| 9 | Code-only draft vocabulary | riceharvest [`8d8d425958`](https://github.com/riceharvest/strata-b70-only/commit/8d8d4259584bb0cf9a5636c6f4615f5fca41d6d5); VlDubr [`c3d0ca379c`](https://github.com/VlDubr/strata-mi50/commit/c3d0ca379cd40cda812f446d20ad77becc56d9d6) (Cyrillic +14.8%) | `--mtp-draft-vocab draft_vocab_code.bin` | script-matched vocabulary +14.8% (4× MI50) | exact | no (en and full flat) | 0-2% | S (data file) |
| 10 | Typical / margin acceptance, thinking only | zsariboga [`ee9d60f5e6`](https://github.com/zsariboga/Strata/commit/ee9d60f5e6313b7a0727695db871d4327f7b5568), [`6c4c468404`](https://github.com/zsariboga/Strata/commit/6c4c46840498b8e4b34aac90d91d47d9e4ab8208) | `STRATA_TYPICAL_MARGIN=r STRATA_TYPICAL_THINK=1` | no speed number | **lossy** | parked idea | unknown; thinking ~6.6% of turn time limits it | M |
| 11 | Draft-side token bans | sebastianrcnt [`2efa2f9096`](https://github.com/sebastianrcnt/Strata/commit/2efa2f909673baf1792cb6555e718b33236fd4db) | (automatic) | none | exact | no | ~0 unless bans are used | S |

Measured dead or already covered (do not queue):

| Idea | Evidence |
|---|---|
| More window-sizing controllers: Ben1078 `--spec-policy cost`, MistVVK auto window, zsariboga STRATA_MTP_ADAPTIVE, svgop STRATA_ADAPTIVE_SPEC, svgop STRATA_SPEC_KEEP, dys0re `--mtp-adaptive`, tctien342 adaptive depth | same family as pancake's STRATA_SPEC_COST, which was flat here; svgop's own were neutral or "data-empty" |
| Tree / top-2 / deeper chains / batch draft depth | svgop `af29fb35c6`, `89e63ce8c9`, `a52058c7b1`; maoyufeng `dc95cf3a82` |
| DFlash2 | pancake 84.3 after a long prompt; dongnh311 `08dd52cb59` (3060: MTP 65-68 vs DFlash2 37-52) |
| STRATA_DRAFT_UNGATED-style eager step launch | svgop `5f8fc1bf10` -2.73% (a warning for pancake's STRATA_MTP_EAGER A/B) |
| 2-bit draft head proxy | already in pancake (STRATA_MTP_PROXY); eddoursul `2ae47d0f44` +0.2..+2.2% |
| Two-stage pick on the full output head | svgop `64e6ba455f`: rescore band explodes on flat logits |
| Context n-gram as a replacement for MTP | svgop `b91fde1191`: MTP acceptance beats the prompt-lookup ceiling |
| Naive "any lookup match replaces MTP" | lighttransport `1a7a35d0b3`: edit-py -12% |

### Inferences
- Order:
  1. Run 1 and 2 in the next GPU test session (env only).
  2. Log windows to check whether a SPEC_CUM-style stop has headroom on our traffic, using the existing window trace and drafts record (items 64 and 5; `tools/pancake/trace/tail_replay.py` already replays tail-row drops with real draft confidences).
  3. Port 3 if the replay shows ≥2%.
  4. Port 4 as an opt-in lossy arm next to STRATA_VERIFY_DEFER_CPU_TOP, gated by the tool-call quality check.
- Pancake's item 5 (tail-row drop, `tail_replay.py`) is the offline twin of SPEC_CUM, and android-imdad's commit is a ready runtime implementation of it. The two belong in one spec.
- The android-imdad oracle (+21% on a 90%-VRAM XTX) and our 0.56 rejected rows per window put a hard ceiling on all exact window-shaping work at roughly 5-6% on pancake. Bigger gains need either more accepted tokens (a better drafter or sampled acceptance at T ≥ 0.7) or cheaper rows (ECAP, deferral).

### Gaps
- All estimates are unmeasured on the RX 7900 XT + 5700G. None of the cited forks runs a 4B MTP head on IQ2_XS with experts in DDR4.
- Not read due to the API budget: diffs of Ben1078's full cost_policy.cpp, MistVVK's 372bdac4ce generate.cpp changes, dys0re's mtp_budget.hpp beyond its header, and the imanu86/Strata-2080Ti "GPT6-MTP" series (about 40 Italian-language research commits on two-horizon adapters and causal traces, with no speed claims in their subjects).
