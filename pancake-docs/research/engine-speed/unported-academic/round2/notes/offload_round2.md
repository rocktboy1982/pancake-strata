# Expert offload, round 2: Athena bandit, sub-2-bit expert tiers, attention share vs context

Research date: 2026-11-12. Read-only pass: no code edits, no builds, no GPU. Round-1 notes: `/home/dan/clone-of-d/src/Strata/pancake-docs/research/engine-speed/unported-academic/notes/expert_offload_papers.md`.

Read levels used below:
- **full text (partial)**: WebFetch of the arXiv HTML, summary by a small model, covering the first part of the body only where stated.
- **full text (summary)**: WebFetch summary of the full HTML; numbers should be spot-checked against the PDF before quoting.
- **abstract only**: arXiv abstract or search snippet.
- **snippet**: search-result text, not opened.
- **unverified**: claim from round-1 notes or a secondary site that this pass did not confirm.

Corrections to round 1 found in this pass:
- Athena's body gives an action space of 4 (prefetcher on/off x OCP on/off) with a prefetcher-degree knob, four state features, a SARSA rule, and an epoch of 2K retired instructions. Round 1 said the epoch and the state were unconfirmed; they are now confirmed (full text partial).
- Round 1 cited "+5.0 to +7.0% geomean over a stateless MAB". The body reports Athena beats MAB by **5.0%** across the 100 workloads (an average speedup, not a geomean). The "+1.4 to +1.7 from state" and "reward correction +1.0" figures were not found in the part read.
- The q·sqs paper (2603.08960) is an analytical cost model, not a measurement. Round 1 did not cover it.

---

## Q1. A44: Athena (2601.17615) and other online cache/prefetch controllers

### Takeaway
Athena is a tabular SARSA controller that picks among 4 coordination settings per 2K-instruction epoch for a CPU prefetcher and an off-chip predictor. Its reward is cycles-based, measured in ChampSim simulation on 100 memory-intensive CPU traces. Nothing in the primary sources shows an online RL or bandit controller tuning a MoE expert cache or prefetch configuration that has been measured against a stateless bandit on real traffic. The closest MoE work, ExpertCache, is abstract-level only. The Athena reward and window design transfer to Pancake only as a design pattern (per-window choice among opt-in arms, reward from an epoch metric). Its 5% gain over a MAB is a simulator result on CPU workloads, not evidence for Pancake.

### Cited Findings

**Athena, arXiv 2601.17615 (Bera et al., ETH SAFARI, HPCA 2026, submitted 24 Jan 2026) [full text (partial): first 100,000 of ~244,700 characters of the HTML were read; the rest, including most of the evaluation, was not]**
- Action space: four coordination actions, enable only the prefetcher, only the OCP, both, or neither. When the prefetcher is on, its degree is scaled by the Q-value confidence margin (Algorithm 1), with tau = 0.12 (Table 3). [Athena HTML](https://arxiv.org/html/2601.17615v1), §4.2
- State: four features chosen from seven candidates: prefetcher accuracy, OCP accuracy, bandwidth usage, prefetch-induced cache pollution. [Athena HTML](https://arxiv.org/html/2601.17615v1), §4.1, §5.3.1, Table 3
- Reward (Eq. 2): R_t = R_corr − R_uncorr. Correlated (action-driven) terms: cycles (lambda 1.6), LLC misses, LLC miss latency; Table 3 sets the LLC weights to 0, so cycles is the effective term. Uncorrelated (workload-driven) terms: load instructions (lambda 0.6) and mispredicted branches (lambda 1.0). The uncorrelated term is subtracted so that workload noise is not credited to the action. [Athena HTML](https://arxiv.org/html/2601.17615v1), §4.3, Table 2, Table 3
- Epoch: 2,000 retired instructions (not cycles). [Athena HTML](https://arxiv.org/html/2601.17615v1), §5.3.1, Table 3
- Learning rule: SARSA update (Eq. 1) on a tabular QVStore (8 hashed planes, 64 rows, 4 action columns, 8-bit entries). alpha 0.6, gamma 0.6, epsilon 0 (no exploration). Q-update is delayed 50 cycles after each epoch. [Athena HTML](https://arxiv.org/html/2601.17615v1), §3.1, §5.1, §5.3.1, Table 3
- Evaluation: 100 memory-intensive single-core traces (at least 3 LLC MPKI) from SPEC CPU 2006 (29), SPEC 2017 (20), PARSEC (13), Ligra (13), CVP (25); 90 four-core and 90 eight-core mixes; hyperparameters tuned on 20 separate workloads. [Athena HTML](https://arxiv.org/html/2601.17615v1), §6.1, Table 6, §5.3
- Results: default config CD1 (POPET off-chip predictor + Pythia prefetcher at L2C), Athena +10.3% average over no-prefetch/no-OCP baseline vs +11.1% for StaticBest. Across all 100 workloads, Athena beats Naive by 5.7%, HPAC by 7.9%, MAB by 5.0%. These are average speedups; the paper does not give a geomean IPC gain. [Athena HTML](https://arxiv.org/html/2601.17615v1), §7.1.1
- Abstract: outperforms prior coordination policies across configurations, prefetcher/OCP combinations and bandwidths, with modest storage. [arXiv abstract](https://arxiv.org/abs/2601.17615v1)
- Not read (second half of body): the sensitivity to epoch length, the per-feature ablation numbers, and the "+1.4 to +1.7" state-feature figure in round 1. Those remain unverified.

**Other online controllers (all abstract or snippet only; no full text opened)**
- ExpertCache (ICSME 2025 NIER, Univ. of Luxembourg) [snippet, search result](https://orbilu.uni.lu/bitstream/10993/65750/1/ICSME25-ExpertCache.pdf): two-phase RL. A pre-loading controller selects a task-specific expert subset for GPU cache; a runtime controller activates experts per token. Evaluated on Qwen3-235B-A22B with BigCodeBench. Reward signal: not in the snippet. Transfer: not tested beyond one model and one benchmark, per snippet.
- Princeton option-critic work (arXiv 2604.20156) [snippet, search result](https://www.alphaxiv.org/abs/2604.20156.md): a per-layer controller decides when to switch expert sets and which experts to load; applied to gpt-oss-20b; switch rate reportedly drops from above 50% below a threshold. Reward and comparison baseline: not in the snippet.
- CAR-MoE (NeurIPS 2026 listing) [snippet, search result](https://neurips.cc/virtual/2026/164462): RL as reward shaping of the router so cache hits rise over an expert subset; tested on 1B and 7B. This changes the router, so it is not a window-level config choice. Read level: listing only.
- ProMoE (arXiv 2410.22134) [snippet]: learned proactive prefetch, not RL. On DeepSeek-MoE, missing-expert loading is over 60% of inference time (snippet; unverified).
- MoE-Beyond (arXiv 2508.17137) [snippet]: transformer predictor replacing a heuristic prefetch predictor; not an online controller.
- No primary source found for an online RL/bandit that chooses a cache size or prefetch depth per window in MoE serving. The search returned none.

### Inferences
- Pancake's `STRATA_BANDIT` arms (top, kmiss, preread, plus the EXACT/RULE/TYPE/MARGIN variants) map onto Athena's "enable or disable each mechanism" action set more than onto ExpertCache's expert-subset choice. The structural match is real, which supports the A44 research question.
- Athena's reward is an epoch-level cycle count with the workload term subtracted. Pancake's equivalent is decode tokens/s per window, minus a term for the lossy arm's own bias (quality is not in the reward). That quality gap is the main transfer problem: Athena's actions have no output-quality cost, while the A44 arms (A4, A5, pool pre-read) do. A bandit cannot learn that cost from a time-only reward.
- Transfer evidence: Athena's own gain comes from one simulator and one workload family. It does not show transfer to a different domain. The "+5%" over MAB is the paper's number on CPU traces.
- Single-window gain is bounded by the epoch count. Pancake's lossy arm ran in 44.5 to 49.6% of epochs (round 1, offline replay). With a 2K-instruction epoch equivalent of roughly one short decode window, the controller sees few samples per session; Athena's 2K-instruction epoch is far shorter than a Pancake window, so the per-epoch statistics are not comparable.

### Gaps
- Athena's epoch-length sensitivity and feature ablations were in the unread half of the body.
- No primary source found on a bandit or RL controller for MoE expert cache configuration with a measured comparison to a stateless bandit.
- ExpertCache reward design and results: not verified (abstract/snippet only).

### The single measurement that settles A44
Offline replay of the existing bandit decision log is free (CPU only). The decisive check is a GPU A/B on real decode, measured with the A B B A paired-start method:
- Fixed-off, fixed-on (`STRATA_BANDIT=top,kmiss,preread` forced), and bandit with `STRATA_BANDIT_MARGIN=0.02`, on an edit window and a short-chat window.
- Pass criterion: the bandit keeps the best fixed arm's gain on the edit window and does not lose on short chat.
- GPU time estimate (my estimate, not measured): about 20 to 30 minutes for 4 paired runs of 3 to 5 minutes each on a quiet machine, plus warm-up. Replay is 0 GPU.

---

## Q2. A27: sub-2-bit copies of experts on CPU misses (HOBBIT and follow-ups)

### Takeaway
HOBBIT's precision split is confirmed in full text (summary): a score of the sum of higher-ranked normalized gate weights, T1 = 0.6 for high precision, T2 = 0.9 for skip, with the middle band loaded in low precision. The low-precision copies are pre-existing int4 (for float16 models) and int2 (for int8 models), and the paper does not describe how they are made. Accuracy loss on Mixtral and Phi-MoE is at most about 1 point. Pancake's experts are already IQ2_XS (about 2.3 bits per weight, from memory, unverified here), so a "sub-2-bit copy" is a tier below a 2.3-bit base, not a fraction of a 16-bit base. The gain in bytes per miss is roughly 25% (IQ1_M-class, about 1.75 bpw from memory, unverified), not the 2x to 4x of HOBBIT's pairs. Given the round-1 shadow-pack result (IQ1_M expert error 0.55 to 0.91), a per-miss sub-2-bit tier is not a sensible tier on this engine without a gate that predicts which experts can take it.

### Cited Findings

**HOBBIT, arXiv 2411.01433 [full text (summary) of arXiv HTML v2, plus search snippet of v1]**
- Precision split: the K selected experts are sorted by normalized gate weight; expert i's score is the sum of the weights of higher-ranked experts (Eq. 2); the first expert always scores 0. Score <= T1 loads high precision; T1 = 0.6. T2 = 0.9 bypasses the expert. Scores between T1 and T2 are loaded low precision (implied, not stated explicitly). For Mixtral-8x7B the reported split is about 67% high, 30% low, 3% skipped. [HOBBIT HTML v2](https://arxiv.org/html/2411.01433v2), §3.2
- Low-precision copy: the paper says it preloads "versions of the experts with different precision levels" (§3.3); §5.1 says int4 replaces float16 and int2 replaces int8. The generation method is not named. [HOBBIT HTML v2](https://arxiv.org/html/2411.01433v2), §3.3, §5.1
- Accuracy (Table 3, §5.3): Mixtral GSM8K 0.52 (Float16) vs 0.51 (Float16+Int4); 0.52 (Int8) vs 0.51 (Int8+Int2). Phi-MoE GSM8K 0.83 in all configs. TruthfulQA changes are within about 0.02. The paper says accuracy drops "no more than 1%". [HOBBIT HTML v2](https://arxiv.org/html/2411.01433v2), §5.3, Table 3
- Models: Mixtral-8x7B (8 experts, top-2) and Phi-MoE (16 experts, top-2). Hardware: RTX 4090 (24 GB) with 256 GB host RAM, PCIe 4.0; Jetson AGX Orin. [HOBBIT HTML v2](https://arxiv.org/html/2411.01433v2), §5.1, Table 1
- Speedups: as recorded in round 1 (body says the "up to 9.93x" is an average over configurations against one baseline). Read level: full text (summary).
- Search summary says HOBBIT's gating-weight importance is "magnitude-based"; the paper's proxy is gate weight with reported Pearson 0.99 against expert output norm, on Mixtral only (round 1). [WebSearch snippet, 2026-11-12](https://arxiv.org/html/2411.01433v1)

**SliceMoE, arXiv 2512.12990 (Dec 2025, rev. Mar 2026) [full text (summary); complete through conclusion]**
- Mechanism: AMAT, calibration-free Matryoshka-style truncation quantization. A low-bit slice is the high-bit weight right-shifted, so low and high precision share memory with no duplication. Configurations: MAT(4,2), MAT(6,3), MAT(8,4). Expert weights G32 asymmetric INT; non-expert G128 symmetric INT8. [SliceMoE HTML](https://arxiv.org/html/2512.12990), §4.2, §6.1
- Quality (expert-only perplexity, Table 1; lower is better). 2-bit (MAT422) column:
  - DeepSeek-V2-Lite: Sym base 19.67; Asym base 9.46; AMAT 10.18 (4-bit column: 7.06 base, 7.11 AMAT).
  - Qwen1.5-MoE-A2.7B: Sym base 19.75; Asym base 11.51; AMAT 10.85 (4-bit: 8.14 base, 8.09 AMAT).
  - At 3-bit (MAT63): DeepSeek AMAT 7.56 vs base 7.29; Qwen AMAT 8.61 vs base 8.51.
  - The paper's claim that AMAT "consistently ... outperform[s] the baseline low-bit quantization" holds for Qwen at 2 and 4 bits and not for DeepSeek at 2 bits or for 3 bits on either model. Dataset for perplexity not stated. [SliceMoE HTML](https://arxiv.org/html/2512.12990), §4.2, Table 1
- Accuracy: GSM8K 5-shot shown only as a figure (Fig. 8). No numeric accuracy in the text.
- Latency: DBSC with AMAT vs high-bit Cache-Prior baseline, decode: up to 1.81x speed-up (DeepSeek-V2-Lite) and 1.64x (Qwen1.5-MoE). Figures are normalized; no absolute latencies. Bit widths used for the headline numbers are not stated. [SliceMoE HTML](https://arxiv.org/html/2512.12990), §6.3, abstract
- Hardware: simulated mobile SoC (systolic array, LPDDR4, UFS 3.1 flash). Not a GPU measurement. [SliceMoE HTML](https://arxiv.org/html/2512.12990), §5
- Miss rates: Cache-Prior effective at about 10 to 30% miss, collapsing below 5%. [SliceMoE HTML](https://arxiv.org/html/2512.12990), §1

**MoBiLE, arXiv 2510.12357 (round 1, full text summary)**: not a precision method; it halves the expert count for unimportant tokens. Not relevant to the sub-2-bit question. Round 1 notes stand.

**Surveys**: [arXiv 2510.16805](https://arxiv.org/pdf/2510.16805) mixed-precision survey summarizes HOBBIT as a mixed-precision loader, prefetcher and cache. Abstract/snippet only.

### Inferences
- Byte math (estimate): Pancake's per-miss transfer is an IQ2_XS expert (about 2.3 bpw, unverified). A 1.75 bpw IQ1_M-class copy saves about 25% of bytes per miss. HOBBIT's int4/int2 pairs save 50% to 75%. So a 2-bit-class copy on Pancake buys little bandwidth and costs quality. The shadow-pack regime (round 1: IQ1_M expert output error 0.55 to 0.91) is the larger problem.
- SliceMoE's result is the closest evidence for an in-family tier: its 2-bit AMAT slice on Qwen1.5-MoE is within about 1.2 perplexity points of its 4-bit base (10.85 vs 8.09, roughly 34% worse, so not close). The paper's claim of "near-high-bit accuracy" rests on the quality figure in Fig. 8, which was not readable. On DeepSeek-V2-Lite, the 2-bit AMAT is about 44% worse than the 4-bit base in perplexity. These numbers do not support a 2-bit copy as lossless.
- Sensible tier verdict: a sub-2-bit copy of an IQ2_XS expert is not a sensible tier as an unconditional per-miss replacement. It might be sensible only as a gated tier applied to low-gate-weight experts (HOBBIT's score rule), so that the expected error is small. That gating is untested here; the routing trace has no gate weights (MOD_PLAN item 6).
- The useful engine question is whether the gate-score rule (HOBBIT's T1) alone, without a lower-precision copy, decides which misses can be skipped (T2) or served from the already-resident copy. That is closer to A4 than to A27.

### Gaps
- HOBBIT's low-precision copy generation procedure is not described in the paper; no source found.
- No 2025-2026 paper measured a sub-2-bit expert tier on Qwen3 or DeepSeek-V3 with a GPU decode benchmark. SliceMoE is on mobile simulation and DeepSeek-V2-Lite and Qwen1.5-MoE only.
- IQ2_XS and IQ1_M bits-per-weight values are from memory, not checked here.

### The single measurement that settles A27
On a routing trace with gate weights (`--dump-routing`, MOD_PLAN item 6), compute for each CPU miss its HOBBIT score and the output-error of the IQ2_XS read vs an IQ1_M-class read on the same expert, then check whether the top-score bucket error stays under the round-1 threshold. Offline replay of the trace is CPU only. The decisive GPU run is an A/B decode with the rule on, under the task gate (A55).
- GPU time estimate (my estimate): about 1 to 2 GPU hours (a dump-routing run, one sub-2-bit source build on CPU, plus two task-gate runs of about 30 minutes each).

---

## Q3. A49: attention share of MoE decode time vs context, and where expert offload dominates

### Takeaway
No primary measurement (2025-2026) found that gives attention's share of MoE decode time vs context on a consumer or workstation GPU with expert offload. The sources found are either a CPU-attention study with a 128 to 2048 token range (MoE-Lightning, not consumer), an analytical cost model (q·sqs, 2603.08960), or a snippet from MoE-Infinity (2401.14361) that reports computation rising from about 50 ms to 160 ms from 4K to 128K context on DeepSeek-V2-Lite. The last is a snippet only, not opened. Expert and KV transfer over PCIe is reported as dominant at long context in several sources, but none gives an attention-share number. The crossover context length is therefore not known from the literature and must be measured on this engine.

### Cited Findings

**MoE-Lightning, arXiv 2411.11217 [snippet from search, not opened]**
- Decode study: when micro-batch size and context grow, CPU attention becomes the bottleneck; MoE FFN latency is roughly flat (memory-bound). Context range 128 to 2048 tokens. [MoE-Lightning](https://arxiv.org/pdf/2411.11217) (snippet; unverified). Not consumer hardware; CPU attention is a different design from Pancake.

**q·sqs inequality, arXiv 2603.08960 (Adhinarayanan & Jayasena, Mar 2026) [full text (summary) of HTML v1]**
- Analytical per-token cost model (Eq. 19); no hardware runs. Reference config: 180 GB HBM3e, 8 TB/s, 4.5 PFLOPS FP8 (Table 2), a specification, not a measurement.
- Table 4 gives throughput at 1k to 16384k context on 64 GPUs; Table 5 gives a per-token breakdown (HBM, compute, communication, relative units) only at 1k and 128k; it does not isolate attention.
- Section 2.2: attention latency "typically dominated by KV cache traffic." No attention-share number. [q·sqs HTML](https://arxiv.org/html/2603.08960v1), §2.2, §4.1, Tables 2, 4, 5
- Read level: full text (summary). Model, not a measurement.

**MoE-Infinity, arXiv 2401.14361 [snippet only, not opened]**
- On DeepSeek-V2-Lite with 128K context, the expert buffer shrinks as KV cache grows. Reported computation rising from about 50 ms (4K) to 160 ms (128K). The rest of the added latency is offload-related. [MoE-Infinity](https://arxiv.org/pdf/2401.14361) (snippet; unverified). Source for both numbers must be checked in the PDF before use.

**Harvest, arXiv 2602.00328 (Feb 2026) [abstract/snippet]**
- Argues PCIe transfers sit on the critical path of misses for both KV-cache and expert offload; NVLink peer GPU memory gives up to about 10x lower MoE expert transfer latency vs host PCIe; 1.5 to 2.0x higher end-to-end throughput for Qwen and Phi-3.5. [Harvest](https://arxiv.org/html/2602.00328v1) (snippet; unverified). Not an attention-share measurement.

**KV offload, arXiv 2601.19910 [snippet]**: up to 99% of end-to-end latency to PCIe transfers, for KV cache offload (not expert offload). Different measurement; do not transfer.

**ProMoE (arXiv 2410.22134) [snippet]**: expert loading over 60% of inference time on DeepSeek-MoE (unverified).

### Inferences
- The available evidence points one way: as context grows, KV-cache traffic and attention cost rise, and the expert-offload share falls only if the KV cache displaces the expert cache (MoE-Infinity's mechanism). For Pancake the crossover is set by where expert transfer stops being the largest term, which is plausibly in the 16k to 64k context range for a 20 GB card, but this is an inference, not a measurement.
- AGENTS.md lists 32k, 64k and 128k context options for the product; those are the context points that matter for the product question.

### Gaps
- No 2025-2026 primary measurement of attention share vs context for a MoE model with offloaded experts on a 16 to 24 GB consumer GPU.
- MoE-Infinity and MoE-Lightning numbers are not verified against full text.
- The q·sqs paper is analytical only.

### The single measurement that settles A49
On this engine, with deferral on, a decode profile at fixed context points of 2k, 16k, 32k, 64k and 128k (a filled KV cache, not just a prompt length). Per point, record the wall-time split: attention kernel time, expert-pool wait (misses), VRAM-hit expert kernels, and the rest. The crossover is the first context where the expert-pool wait falls below attention time. Report with the A B B A method only if a lossy arm is involved; this is a plain profile.
- GPU time estimate (my estimate): about 1 hour. Each point needs a warm-up decode and a few hundred decode tokens; 5 points at a few minutes each plus model load (the 3-minute load dominates).

---

## Summary

| Question | Answer | Best source | Read level | Settling measurement (GPU time, estimate) |
|---|---|---|---|---|
| Q1 Athena / controllers | Athena: 4-action SARSA, cycles reward, 2K-instr epoch, +5.0% over MAB on CPU sims. No MoE controller with a measured stateless-bandit comparison found | [Athena HTML](https://arxiv.org/html/2601.17615v1) | full text (partial, first ~100k of ~245k chars) | Bandit vs fixed-off vs fixed-on A B B A on edit and chat windows (~20 to 30 min) |
| Q2 sub-2-bit tier | Not sensible unconditionally: 2-bit AMAT is 34 to 44% worse in perplexity than its 4-bit base; a 1.75-bpw copy saves only ~25% of IQ2_XS bytes (estimate) | [HOBBIT HTML v2](https://arxiv.org/html/2411.01433v2); [SliceMoE HTML](https://arxiv.org/html/2512.12990) | full text (summary) | Gate-score-gated copy: routing trace with weights, output error by score bucket, then task gate (~1 to 2 h) |
| Q3 attention share vs context | No primary consumer measurement found; crossover unknown | [q·sqs HTML](https://arxiv.org/html/2603.08960v1) (model only); MoE-Infinity snippet | full text (model); snippet (MoE-Infinity) | Decode profile at 2k/16k/32k/64k/128k with deferral on (~1 h) |

## Unverified items to check before quoting
- Athena: the "+1.4 to +1.7" state gain and "+1.0" reward correction (round 1) are not seen in the read portion.
- SliceMoE: GSM8K accuracy values in Fig. 8 are not readable in text.
- MoE-Infinity: 50 ms to 160 ms context numbers are from a snippet.
- IQ2_XS (about 2.3 bpw) and IQ1_M (about 1.75 bpw) sizes are from memory.
- ExpertCache, Princeton option-critic, CAR-MoE: abstract or listing only.
