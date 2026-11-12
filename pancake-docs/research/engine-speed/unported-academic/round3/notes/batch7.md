# Round-3 batch 7: A45, A46, A48, A50, A51 (read-only)

Date: 2026-11-12. Scope: literature and repo state only. No code edits, no builds, no GPU. Repo checks are read-only greps on /home/dan/clone-of-d/src/Strata.

Read level for every paper below: **abstract only** (WebFetch of the arXiv abstract page). No full text was read in this pass. Numbers quoted from the MOD_PLAN rows were written earlier by other sessions and were not re-checked here.

## A45 - startup probe and calibration of the CPU-vs-promote ratio

### Takeaway
The DALI paper supports runtime workload-aware CPU/GPU assignment, but its abstract gives no numbers, so the 6-13% figure in MOD_PLAN is not confirmed here. The Pancake A45 code is opt-in and lives on a branch, not in the main tree. The probe can be run on the GPU once, cheaply.

### Cited Findings
- DALI (arXiv 2602.03495, submitted 3 Feb 2026): a workload-aware offloading framework for MoE on local PCs, with a runtime CPU/GPU expert-assignment method, residual-based prefetch and a workload-aware GPU cache replacement policy. Abstract claims "significant speedups in the both prefill and decoding phases over the state-of-the-art offloading frameworks"; no figures and no hardware named on the abstract page. [DALI](https://arxiv.org/abs/2602.03495)
- MOD_PLAN (not re-verified here) records DALI as: assignment alone 23% faster than HybriMoE, decode 1.32-3.97x, prefill 2.00-7.62x, on EPYC 7532 + RTX 3090 + PCIe 4.0 x16, with a smallest batch of 4 in the text. These need the full text to confirm. [MOD_PLAN.md row A45 and the DALI correction note, line ~1640](/home/dan/clone-of-d/src/Strata/MOD_PLAN.md)
- Repo: the A45 probe is `calibrate_ratio.hpp` plus `generate.cpp` probe, opt-in via `STRATA_CALIBRATE=1`, on branch `work/a45-calibrate` (from `origin/dev/batch-2`). The header is absent from the main tree; it exists only in `.claude/worktrees/agent-ad25cdf1dc7ea704a/include/strata/core/calibrate_ratio.hpp`. The `generate.cpp` include of it was added on the policy branch (MOD_PLAN compilegate row). Design notes: `pancake-docs/batch2/A45-calibrate.md`.
- Repo: the probe's threshold `STRATA_CALIBRATE_MIN_SHARE` (default 0.35) is stated in its own doc as a heuristic, not a measurement; the doc says the box's expected raw share is about 0.28, so it predicts 0 here.
- MOD_PLAN status (policy row): A45 "matches DALI's warm-up balance only, static share, 'finds 0' true by construction"; run the probe once on the GPU. [MOD_PLAN.md policy row, line ~1511](/home/dan/clone-of-d/src/Strata/MOD_PLAN.md)

### Inferences
- The idea is sound in principle: the ratio `cpu/copy` is measured, not assumed. The DALI abstract supports that runtime workload matters, not that a one-time probe is enough. DALI's per-layer, per-step assignment is a different mechanism from A45's once-per-start static share.
- On this box a static share is expected to be 0, so A45 probably changes nothing. The probe's main value is to confirm that.

### Gaps
- DALI full text not read: hardware, speedup breakdown, and whether it uses a one-time or per-step probe are unverified.
- Repo: no measured probe output on this box exists in the notes I read.

### Mechanism, evaluation, exact/lossy, port status, smallest test
- Mechanism: one-time timing of CPU expert pass vs. host-to-device copy, then a static `pcie_frac`.
- Evaluation: DALI abstract only (no numbers). Pancake: not evaluated on GPU.
- Exact or lossy: exact (policy only; same experts, different lane). A nonzero share changes bits only through lane rounding.
- Already in Pancake? Partly. Code exists on branch `work/a45-calibrate`, opt-in, host unit test only. The main tree does not contain the header.
- Smallest test: merge or cherry-pick the branch, run `STRATA_CALIBRATE=1` once on the 7900 XT, read the printed ratio and the chosen share. Expected GPU time: about 5 min (engine start plus one short decode). Pass criterion: share is 0 with a log, which matches the static `--pcie-frac 0` run.

## A46 - learned cache policy on the simulator (Belady imitation, admission and migration)

### Takeaway
The ADRS paper ("Barbarians at the Gate") is a general method for LLM-driven algorithm search with simulators, with four case studies. Its abstract reports up to 5.0x runtime or 50% cost reduction, but not for cache policies or hit rates. Pancake's own A46 test already closed the learned admission idea on the bench traces.

### Cited Findings
- ADRS (arXiv 2510.06189, v3, title "Barbarians at the Gate: How AI is Upending Systems Research"): systems problems have reliable verifiers, so LLM-evolved candidates can be checked against real implementations or simulators. Four case studies: multi-region cloud load balancing, MoE inference, LLM-based SQL, transaction scheduling. Abstract: "up to 5.0x runtime improvements or 50% cost reductions." No hit-rate figures on the abstract page. [ADRS](https://arxiv.org/abs/2510.06189)
- Repo: `tools/pancake/cache_learn.py` exists (Belady, Hawkeye, imitation). MOD_PLAN records its result as "best online +3.2 hit points but 2-3x promotions, slower than k-th miss -> closed on bench traces." [MOD_PLAN.md row A46 and status line ~1726](/home/dan/clone-of-d/src/Strata/MOD_PLAN.md)
- Repo, finemoe row: A46 on the one bench trace gives +1.0 point; the claim is limited by a trace with zero router weights. [MOD_PLAN.md finemoe row, line ~1512](/home/dan/clone-of-d/src/Strata/MOD_PLAN.md)
- Repo, research notes: the replay ceilings at 10,909 cached experts (44%) are engine tier 79.6%, per-layer LRU 89.5%, Belady 94.7%. [MOD_PLAN.md line ~819](/home/dan/clone-of-d/src/Strata/MOD_PLAN.md)

### Inferences
- The Belady ceiling bounds the gain of any replacement policy: at most about 5 hit points over the engine tier on this trace (94.7 vs 89.5 LRU, 79.6 engine). A learned policy cannot beat 94.7% on that trace.
- The real blocker is promotions and migration cost, not the hit rate. A admission split needs a cost term.

### Gaps
- ADRS full text not read; the policy-evolution method and any hit-rate result are unverified.
- No agent-session trace with nonzero router weights exists to test an admission policy against (the blocker named in MOD_PLAN).

### Mechanism, evaluation, exact/lossy, port status, smallest test
- Mechanism: candidate policies are searched by an LLM against a simulator; in Pancake, Belady imitation plus admission/migration split.
- Evaluation: ADRS abstract only (no cache numbers). Pancake: offline replay on one 219-window trace.
- Exact or lossy: policy only; exact on the bytes served, the cost is in promotions.
- Already in Pancake? Yes, in offline form (`tools/pancake/cache_learn.py`). Closed on bench traces, not on agent traces.
- Smallest test: offline replay only, no GPU. Requires a recorded agent-session trace with router weights (MOD_PLAN item 6 of the next-steps list). GPU time 0 min. Pass criterion: at least +5 hit points over the current engine tier, with promotions not above the k-th-miss baseline.

## A48 - single-thread and pool GB/s micro-benchmark of the Q2_0 kernel versus 2-bit GEMV

### Takeaway
The Intel 2-bit paper's claims are end-to-end, not GEMV GB/s, so it cannot be used as a per-kernel ceiling. Pancake's own repo already measured the Q2_0 floor on DDR4. The remaining work is a CPU-only 8-thread run on an idle box.

### Cited Findings
- "Pushing the Envelope of LLM Inference with Ultra-Low-Bit Quantized Models" (arXiv 2508.06753): 2-bit CPU microkernels; end-to-end "2.2x" over bitnet.cpp and "up to 7x" over 16-bit on CPU; "up to 6.7x" over the 16-bit pipeline on Intel Xe2 GPUs via vLLM. Abstract says CPU microkernels are "close-to-roofline," but no GEMV GB/s on the page. [Intel 2-bit](https://arxiv.org/abs/2508.06753)
- T-MAC (EuroSys 2025, arXiv 2407.00088): listed in the plan as the LUT-based low-bit CPU reference. Not fetched in this pass.
- Repo (MOD_PLAN kern-cpu row): F7 Q2_0 AVX2 is merged with a runtime probe; DDR4 floor per row: gate/up 720 B = 21.8 ns, down 180 B = 5.45 ns, expert 1,382,400 B = 41.9 us at 33 GB/s. One thread from DRAM: legacy 4-6x the floor, fast about 4x. At 1 token/expert the 6+ core pool is DDR4-bound for either kernel (room <= 22%). At 2+ tokens/expert legacy is near compute-bound. [MOD_PLAN.md kern-cpu row, line ~1520](/home/dan/clone-of-d/src/Strata/MOD_PLAN.md)
- Repo (same row): "A48 closed (x86 2-bit GEMV literature needs AVX-VNNI/offline layout: not Zen 3)." [MOD_PLAN.md kern-cpu row](/home/dan/clone-of-d/src/Strata/MOD_PLAN.md)
- Repo: the 8-thread pool number "still to measure on an idle box (`q2_rows_bench --threads 8`)." [MOD_PLAN.md kern-cpu row](/home/dan/clone-of-d/src/Strata/MOD_PLAN.md)

### Inferences
- The gap to a DDR4-bound floor is small at 1 token/expert, which is the common decode case here. A LUT rewrite is unlikely to pay on Zen 3 without VNNI.
- The literature's large speedups are against 16-bit baselines, which Pancake does not run on the CPU lane.

### Gaps
- No single-thread or 8-thread GB/s number from the literature was found on the abstract pages.
- Zen 3 (AVX2, no VNNI) is not the platform of the cited Intel work.

### Mechanism, evaluation, exact/lossy, port status, smallest test
- Mechanism: measure bytes per second per thread and per pool for Q2_0 vs. the DDR4 floor.
- Evaluation: Intel abstract only; T-MAC not read. Pancake: single-thread and 6-thread numbers exist; 8-thread pending.
- Exact or lossy: exact (benchmark only).
- Already in Pancake? Yes (F7 kernel, F12 sign path, floor measured). The 8-thread number is the only open item.
- Smallest test: `q2_rows_bench --threads 8` on an idle box. CPU only, GPU time 0 min. Pass criterion: result within 22% of the 33 GB/s floor at 1 token/expert, which closes the item.

## A50 - verify-window attention that reads the prefix K/V once for all window rows

### Takeaway
LongSpec's abstract reports up to 3.26x over Flash Attention baselines on long-context tasks, but the abstract does not say how the target K/V is read during verification. Pancake already has an F11 window-attention branch in the fork notes; its gain is large at 200K context and near zero at short context.

### Cited Findings
- LongSpec (arXiv 2502.17421): a lighter draft with a constant-size KV cache and position indexing for long context; "up to a 3.26x speedup over strong Flash Attention baselines across five long-context understanding datasets"; "a 2.25x reduction in wall-clock time on the AIME24 long reasoning task with the QwQ model." Target-side KV reading is not described on the abstract. [LongSpec](https://arxiv.org/abs/2502.17421)
- Repo (MOD_PLAN A50 row): fork F11: 200K decode 48.6 -> 88.9 tok/s (on a 5080); about 0 at short context. Flash-Next not built (QSA caps it at about 1%). Qwen3.6 engine: coded on `work/f11-attn-window`, exact by design (Phase 15 "F11 status"). [MOD_PLAN.md A50 row, line ~1940](/home/dan/clone-of-d/src/Strata/MOD_PLAN.md)
- Repo: the fork F11 entry in `research_notes/Active Strata forks November/batch3.md` (atmJava 540df69c04) is a different F11 (concurrent cold prompt reads), marked NOT-APPLICABLE. The two are easy to confuse; the window-attention F11 is the one in MOD_PLAN. [batch3.md](/home/dan/clone-of-d/src/Strata/research_notes/Active Strata forks November/batch3.md)
- Repo: the verify window already routes all window tokens in one projection (`docs/AMD_HIP.md`, verify-window routing). [AMD_HIP.md](/home/dan/clone-of-d/src/Strata/docs/AMD_HIP.md)

### Inferences
- Reading the prefix K/V once for all window rows is exact by construction, so the gain is bounded by how much of verify time is prefix K/V traffic. That share grows with context length, so the item is a long-context item.
- Its value depends on A49 (attention share at 10K/64K/128K). MOD_PLAN marks A50 as "decided by A49."

### Gaps
- Whether the branch `work/f11-attn-window` still builds and matches the main tree was not checked in this read-only pass (no branch access without a checkout, which I did not do).
- LongSpec's target-side mechanism not read (full text needed).

### Mechanism, evaluation, exact/lossy, port status, smallest test
- Mechanism: one K/V read per prefix block shared across all rows of the verify window (int8 mma -> gfx11 WMMA, per MOD_PLAN).
- Evaluation: LongSpec abstract only (speedup vs. Flash Attention, draft-side design). Pancake: fork numbers on a 5080, not on the 7900 XT.
- Exact or lossy: exact (by design, per MOD_PLAN).
- Already in Pancake? Partly, on `work/f11-attn-window` (coded). Not on the main tree.
- Smallest test: the A49 decode run at 10K / 64K / 128K via the existing `longdec32k/64k/128k` abtest cases. GPU time about 45-60 min for the three contexts with A/B order A B B A. Pass criterion: attention share of decode at 64K or above is large enough that the window read saves at least 3% of decode time, per the A49 result.

## A51 - query-aware sparse decode attention

### Takeaway
Quest's abstract reports "up to 2.23x self-attention speedup" and a "7.03x" latency reduction, but the figures are for its own setting, not Pancake's. Twilight reports adaptive top-p pruning with "3.9x" end-to-end per-token latency. Both are lossy, and the claimed gains depend on attention being a large part of decode, which MOD_PLAN says it is not at 16 GB context.

### Cited Findings
- Quest (arXiv 2406.10774, ICML 2024): query-aware page selection using per-page min/max key bounds; top-K pages loaded for attention. Abstract: "up to 2.23x self-attention speedup," "reduces inference latency by 7.03x," and "negligible accuracy loss" (no numeric accuracy). Hardware not on the abstract page. [Quest](https://arxiv.org/abs/2406.10774)
- Twilight (arXiv 2502.02770): top-p (nucleus) pruning on top of sparse attention, adaptive budget; abstract: "adaptively prune at most 98% of redundant tokens" and "3.9x acceleration in end-to-end per token latency"; a 15.4x self-attention figure appears in the same sentence. Hardware not on the abstract page. [Twilight](https://arxiv.org/abs/2502.02770)
- Repo (MOD_PLAN A51 row): parked; "attention must exceed ~15% of decode (A49)"; "a few % at 100K+". Repo grep found no query-aware sparse attention implementation in the main tree (only QSA rows mentioned in docs). [MOD_PLAN.md A51 row, line ~1941](/home/dan/clone-of-d/src/Strata/MOD_PLAN.md)

### Inferences
- Both papers measure attention alone, on GPUs with long contexts. Pancake's decode is CPU-bound for the MoE expert lane (combine waits on CPU, per the fork notes), so an attention speedup transfers only in proportion to attention's decode share.
- Lossy by design: needs the same accuracy gate as the other lossy items (rigorous tier).

### Gaps
- Quest and Twilight full text not read; hardware and per-task accuracy unknown.
- Whether Pancake's QSA (MOD_PLAN "QSA caps it at about 1%") is the same mechanism was not verified.

### Mechanism, evaluation, exact/lossy, port status, smallest test
- Mechanism: select top pages by query-dependent bounds, attend only to those; Twilight adds adaptive top-p budgets.
- Evaluation: abstracts only. Pancake: none.
- Exact or lossy: lossy.
- Already in Pancake? No query-aware sparse decode found. QSA exists (capped at about 1%).
- Smallest test: after A49, if decode attention share at 64K or above is at least 15%, run one Quest-style page-selection build behind a flag with the rigorous accuracy gate. GPU time about 2-3 h including an accuracy suite. If A49 shows attention below 15%, skip (the MOD_PLAN decision).

## Summary of GPU time

| Item | Smallest test | GPU time | Gate |
| --- | --- | --- | --- |
| A45 | Run `STRATA_CALIBRATE=1` once on the branch | about 5 min | Expect share 0 here |
| A46 | Offline replay on agent traces with weights | 0 | Needs a trace |
| A48 | `q2_rows_bench --threads 8` | 0 (CPU) | Within 22% of floor |
| A50 | A49 decode at 10K/64K/128K | about 45-60 min | Depends on A49 |
| A51 | Quest-style build behind flag, if A49 >= 15% | about 2-3 h | Rigorous accuracy gate |

## Notes on sources
- Every paper above is read at abstract level. Several full texts (DALI, ADRS, LongSpec, Quest, Twilight, Intel 2-bit) must be read before any port decision.
- The 2026 DALI and ADRS links were fetched as of 2026-11-12; the repo states were read from the main tree and the branch names given in MOD_PLAN.
