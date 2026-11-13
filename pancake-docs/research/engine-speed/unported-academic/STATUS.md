# Academic items: status and remaining work (2026-11-12)
Source of truth for each item: MOD_PLAN.md (candidate list, remaining-items table, verification corrections). This file is the status view.
Types: DONE, CPU (can run now), GPU (overnight), DATA (waits for logs), DEC (needs owner decision).

| Item | Topic | Status | Remaining | Type |
| --- | --- | --- | --- | --- |
| A1 | Cache-conditional routing (turbo) | Verified: miss rates halved (Cache-Conditional, full PDF) | Joint re-measure with deferral on (GPU) | GPU |
| A2 | Depth-graded prior (AcceptMoE) | Verified: paper's mechanism differs from ours | Measure ours as written (GPU); rename row | GPU |
| A4 | Rank guard (HOBBIT thresholds) | Verified thresholds (score, not CDF); no gain over deferral in screen | Cumulative-score test needs routing weights | GPU |
| A5 | k-th miss admission | Done (opt-in); async rejected | None | DONE |
| A6 | Time model | Held-out error 8.1 pts vs 5-9 target | Recalibrate (CPU) | CPU |
| A7 | Chunked promotions (ProMoE) | Verified 1.09x; 'cancellable' wrong (copies cannot be preempted) | Simulator only (+4.6%) | CPU |
| A8 | Online swap budget, blocking tier | Coded; async rejected -9.6% | --adapt-swaps 64/128 then auto (GPU) | GPU |
| A9 | Pre-stage during tools | Coded opt-in | Agent replay with real tool gaps (CPU), then GPU | CPU |
| A10 | Working-set prefetch | Done opt-in | Item 84 first | CPU |
| A11 | Tool-boundary checkpoints | Tests pass 12/12 (tool-boundary marks) | Targeted check of flag in logs | CPU |
| A12 | Pinned prefix store | Opt-in; -1.3% modelled for Qwen | Pinned-entry eviction fix first | CPU then GPU |
| A14 | Restore equivalence | Code done; paper verified as abstract-level | Teacher-forced restore per boundary (GPU) | GPU |
| A15 | Idle TTL for sub-agent sessions | No gain for single user (paper) | Fix rank-3 eviction bug first (CPU) | CPU |
| A16 | Byte-stable prefix | Done (637a71c23) | None | DONE |
| A17 | Per-effort caps (s1 / interruptible) | Verified: 60% drop when updates arrive late | Needs 200+ turns of logs | DATA |
| A18 | Effort per task | Verified overthinking figures (wording); no time-to-success source | Owner decision; ~100 tasks | DEC |
| A19 | Thinking stop (ThinkBrake / Thought Calibration) | Verified -25% thinking tokens (ThinkBrake v5); turn log has no logprobs; thinking_entropy exists server-side | Engine top-2 margin logging (GPU run) before replay | GPU |
| A20 | Rumination / entropy close | Entropy Sentinel r=0.91 is one model | Log-only entropy, then replay (CPU) | CPU |
| A21 | Exact sampled speculation | Verified (Leviathan); switches exist, off; turn log now records temperature | Read real temperatures from logs; min-temp 0.7 screen (GPU) | CPU then GPU |
| A22 | Chain stop by entropy (DISCO / AdaEDL) | Verified DISCO 10.3/31.4%; AdaEDL +4.4-10.5% vs static | Trace step 1 code done (record 6); identity test + entropy replay | GPU |
| A23 | Draft-position cost (record 5) | Done (rowmiss record) | None | DONE |
| A24 | Suffix index (SuffixDecoding) | Verified 5.3x is Text-to-SQL; SWE-Bench 2.5x | suffix_sim rerun (CPU), GPU screen | CPU then GPU |
| A25 | Position decay check | draft_report.py exists | Run on a window trace (after A22 trace) | GPU then CPU |
| A26 | Draft-source selector | Counterfactual possible on existing traces | Counterfactual (CPU) | CPU |
| A27 | Sub-2-bit expert copies (HOBBIT) | Verified; SliceMoE: 2-bit ppl 10.85 vs 8.09 (not lossless) | Routing trace with gate weights, sub-2-bit source | DATA |
| A28 | ISTA coder variant | Source exists (HF); plan's numbers wrong (RCO prunes 50%); 58 GB | Owner approval for download; disk space | DEC |
| A29 | Expert-byte entropy coding | Measured bound 8-12% (2026-11-10) | Owner decision whether an 8-12% saving is worth it | DEC |
| A30 | CPU share threshold (Fiddler) | Speedups verified; gate is latency-based | Env A/B 512-3000 tokens (GPU) | GPU |
| A31 | GDN chunk 64 | Chunk 64 in code; no speedup in paper | hip_gdn_stage_bench (GPU, minutes) | GPU |
| A32 | Streaming experts for large prefill | Headlines verified (OSDI 1200/1800 tok/s); PCIe 5.0 not 3.0; 'exact' wrong | Distinct-experts count and PCIe read (CPU) then timing (GPU) | CPU then GPU |
| A33 | Small-M fusion | HOLD +0.6%; Qwen3 'gated residual' not in paper | expdec_bench (GPU, 10 min) | GPU |
| A34 | Coordinate-descent sweep | Verified 5.5-8.5% on A100; numerics risk | Bit-exact check then one pass (GPU) | GPU |
| A35 | Infinity Cache warm (PRESERVE) | 1.6x verified but on Ascend; mapping is ours | 0/16/40/80 MiB sweep (GPU night) | GPU |
| A36 | Hit-path layout (MonoMoE) | FP8, lossy, H200 only; ceiling +1.99% | Two gpu_profile traces (GPU, 20 min) | GPU |
| A37 | LDS codebook duplication | Kernel gate: no-go (0.15-0.55% ceiling) | None (closed) | DONE |
| A38 | Behaviour counters | ON (log-only) in daily preset | Judge on real sessions (DATA) | DATA |
| A39 | Compaction share | Blocked: compactions predate turn log | Match new compactions to turn log | DATA |
| A40 | Tool overlap (AsyncFC / TraceLab) | Verified 59.8% tool share; AsyncFC 1.35x vs parallel | Overlap share from logs (CPU) | CPU |
| A41 | Read before edit (ECLoop) | 10% of edits had no prior read (33/339) | Decide on hold hook (owner) | DEC |
| A42 | Externalize tool output (RLM) | Verified median 25.6% over four tasks; lossy | A/B offload arm on real turns (after counters) | DATA |
| A43 | Edit substring constraint | 0 of 339 edits failed; closed | Keep log check | DONE |
| A44 | Bandit (Athena) | Verified: beats DUCB-like MAB 5%; framing corrected | Off / on / margin 0.02 A/B (GPU, 20-30 min) | GPU |
| A45 | Calibration probe (DALI) | Headlines verified (3.97/2.16/1.48/1.32x); 6-13% not in paper | One probe run (branch work/a45-calibrate) | GPU |
| A46 | Learned cache policy (Belady) | Harmonia terms corrected; offline bound 94.7 vs 79.6% | Agent trace with router weights (DATA) | DATA |
| A48 | 2-bit kernel (T-MAC / Intel) | Benchmark built and run: 1.23x single thread, 1.03x at 8 threads (memory-bound) | Ceiling ~3% at decode thread count; no work | DONE |
| A49 | Attention share vs context | No published MoE measurement | Decode profile at 2k-128k (GPU, 1 h) | GPU |
| A50 | Verify-window attention (LongSpec) | 3.26x verified; 2.25x was one row (2.34x); 200K result is our fork | After A49 (GPU) | GPU |
| A51 | Sparse decode attention (Quest / Twilight) | Quest 7.03x attn, 2.23x end-to-end (fixed); Twilight verified | Only if A49 attention >= 15% (GPU, 2-3 h) | GPU |
| A52 | Direct I/O expert loader | Code done (184e68e83) | Measure | GPU |
| A53 | IOMMU start-up check | DONE: start-up note on translated GPU; CPU test passes; engine builds | None | DONE |
| A54 | KLD rule | Silent-zone result post hoc; docstring narrowed | Rank agreement over existing lossy arms (CPU) | CPU |
| A55 | Task gate | Dry run passes on 5 specs (path fixed); 16/16 tests | Live base vs lossy run (GPU) to measure cost | GPU |
| A56 | Expected acceptance rate | Verified Leviathan; trace step 1 code done (record 6) | Identity test and dump run (GPU) | GPU |
| A57 | Time-to-success AUC | Horizon pinned in specs (1800 s) | Measure AUC on a live batch | GPU |

## Update 2026-11-13
| item | change since 2026-11-12 |
| --- | --- |
| A24 | suffix replay done: +2.0% accepted, -2.4% rounds (sub-agents +6.0%, -7.5%); GPU screen still open |
| A26 | done: no per-request selector beats always-lookup; per-window longest match is the only candidate above noise (+2.0%), the same figure as A24 |
| A9 | done as a bound: 2 to 4% of wall; coded path is STRATA_IDLE_ADAPT; GPU replay open |
| A6 | done: mean held-out error 3.4 points; worst arm 8 points; refit not honest |
| A40 | measured as an upper bound (5.4% of span), not an overlap measurement |
| A48 | done: 1.23x one thread, 1.03x at 8 threads, about 3% ceiling |
| A49 | done: decode flat with context within noise; A51 gate not met (drop unless profile says otherwise) |
| A51 | recommended drop (see A49) |
| A41 | hook written and tested (8 of 8); not registered |
| A21, A18 | temperature and effort fields in the turn log; confirm on next run |
| A53 | done |
| A56 | code done; identity GPU run open |
| A55 | dry run passes; live run open |
