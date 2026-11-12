# Round 3, batch 4: A21, A24, A25, A26, A29

Scope: MOD_PLAN rows A21, A24, A25, A26, A29 (speculative decoding and expert-byte entropy). Read-only: no code edits, no builds, no GPU runs. Date of research: 2026-11-12.

Read levels used throughout: the arXiv abstract pages were fetched through WebFetch, and no paper body was readable in this session, so every paper figure below is an abstract figure. Repo facts come from grep and reading the files named. Where a figure comes from a fork's commit message or a local research note, it is labelled as such and is not verified.

Summary for the owner:

- A21 is the only item with an immediate, cheap decision: the repo already has exact sampled speculative decoding, and the open question is which temperature opencode's requests actually carry. The repo's own comment says sampled drafting loses below about 0.7.
- A24 already has an implementation and a simulator. Its remaining step is a GPU screen, and the MOD_PLAN status text is stale (it says "planned" in one table and "HOLD" in another).
- A25 already has its analysis tool (`draft_report.py`). It needs one real window trace.
- A26 is mostly covered by DraftPolicy's measured-cost arbitration. Parked, as the plan says.
- A29 was already measured on 2026-11-10 and closed (best saving 12.0%, below the 20% threshold). MOD_PLAN row A29 (line 1867) still says "parked, run the 5-minute check"; that row is stale and should be updated to closed.

---

## A21 - Exact speculative sampling for sampled requests

### Takeaway
Speculative sampling with modified rejection sampling keeps the target's output distribution (exact in distribution, not bitwise). The repo already has this path (`STRATA_SPEC_PROB`, `_COUPLED`, `_PROB_GATE`, `_PROB_DT`, `_MIN_TEMP`), and the preset leaves it off. The first thing to settle is what temperature opencode's requests actually carry, because the plan's own finding says T=0.1 takes the sampled path, and the repo says sampled drafting loses at low temperature.

### Claim (papers, abstract only)
- Leviathan, Kalman, Matias, "Fast Inference from Transformers via Speculative Decoding", arXiv 2211.17192 (ICML 2023 oral per the arXiv comments). Abstract says the method samples "without changing the distribution" and reports "a 2X-3X acceleration compared to the standard T5X implementation" with T5-XXL as the only named model. [Source](https://arxiv.org/abs/2211.17192)
- Chen, Borgeaud, Irving, Lespiau, Sifre, Jumper, "Accelerating Large Language Model Decoding with Speculative Sampling", arXiv 2302.01318. Abstract says a "modified rejection sampling scheme ... preserves the distribution of the target model within hardware numerics" and reports "a 2-2.5x decoding speedup in a distributed setup" on Chinchilla 70B. The draft model and benchmark tasks are not in the abstract. [Source](https://arxiv.org/abs/2302.01318)

### Mechanism
Draft from q, accept token x with probability min(1, p(x)/q(x)), and on rejection resample from the normalised residual (p - q)+. The output then has distribution p. Chen et al. note the guarantee holds only up to hardware numerics, which matters on ROCm where the sampler is not bit-identical to a reference.

### Evaluation (abstract numbers only)
- 2-3x on T5-XXL (Leviathan); 2-2.5x on Chinchilla 70B across a distributed setup (Chen). Neither is a MoE with CPU-resident experts, so neither transfers as a number.
- Fork numbers recorded in the repo (not verified here): riceharvest/strata-b70-only `7389a10f2c` claims +5.5% on six agent-shaped turns at temperature 1.0 (Intel B70); svgop `b59c0e6bbc` claims +13.4% with draft temperature 0.85 on an RTX 5090. Source: local note [speculative_mtp.md](research_notes/Speed%20ideas%20from%20all%20forks/speculative_mtp.md) lines 168-170.

### Exact or lossy
Exact in distribution. Not bitwise: the same seed gives different tokens than the non-speculative path, so greedy parity tests do not apply. A distribution check is needed (the plan already requires one).

### Read level
Abstract only for both papers. Repo side: read in full, `include/strata/core/spec_prob.hpp` lines 53-72, `src/kernels/cuda/sampler.cu:950`, and `docs/DETAILS.md` lines 570-605.

### Pancake status
Already has it, off by default.
- `STRATA_SPEC_PROB`, `STRATA_SPEC_COUPLED`, `STRATA_SPEC_PROB_GATE=top`, `STRATA_SPEC_PROB_DT` (drafter runs at x times the request temperature), `STRATA_SPEC_MIN_TEMP` (coupled/probabilistic drafting only at temperature >= x; default 0 means every sampled request). Source: [spec_prob.hpp](include/strata/core/spec_prob.hpp) lines 53-72.
- The header comment: "Measured on the RTX 3060 and 5070: sampled drafting gains at temperature 1.0 and loses below about 0.7, where the draft head's argmax is the better guess."
- Not in the preset: [speculative_mtp.md](research_notes/Speed%20ideas%20from%20all%20forks/speculative_mtp.md) line 14.
- Gate caveat from the repo: with DISCO, SPEC_CUM or `--spec-min-p`, `STRATA_SPEC_PROB` needs `STRATA_SPEC_PROB_GATE=top` to stay exact; the runtime warning text names the wrong switches (`generate.cpp:3482`, per MOD_PLAN line 1510).
- Temperature source is unresolved:
  - `serve/runconfig.py:24` says the default for requests without a temperature is the model card's (1.0 with thinking, 0.7 without).
  - MOD_PLAN line 1698 and line 1747 say opencode sends no temperature.
  - MOD_PLAN line 860 says "owner keeps T=0.1", and the A21 finding (line 1715) says T=0.1 requests take the sampled path.
  These three cannot all be the live setting. The request temperature has to be read from the actual opencode store or the serve log.

### Smallest test
1. CPU only, no GPU: read the temperature of recent opencode requests (the opencode session store is what `suffix_sim.py` reads) and confirm which path a T=0.1 request takes (`spec_min_temp()` in `spec_prob.hpp`).
2. If sampled requests are real traffic: one night-queue screen, env only, `STRATA_SPEC_MIN_TEMP=0.7` against default, with `STRATA_SPEC_PROB=1 STRATA_SPEC_COUPLED=1 STRATA_SPEC_PROB_GATE=top`, A B B A order, and the first run discarded (owner rule). GPU time: not measured; it is one night slot of several sessions.
3. Distribution check before any default change: sample the first token and a short continuation over many seeds, compare target-only against speculative-sampled.

---

## A24 - Cross-session suffix index shared by sub-agents

### Takeaway
SuffixDecoding reports large speedups on agentic workloads (up to 5.3x), but on SWE-Bench and Text-to-SQL, not on Pancake's workload, and the abstract does not say whether it is exact. Pancake already has a per-request lookup, a cross-session suffix index behind `STRATA_SUFFIX_INDEX`, and a replay simulator. The replay on 2,045 real opencode requests recorded a small gain, so the next step is one GPU screen.

### Claim (abstract only)
Oliaro, Jia, Campos, Qiao, "SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications", arXiv 2411.04975 (NeurIPS 2025 spotlight per the arXiv comments). Abstract: suffix trees cache token sequences from prompts and previous outputs; speculation length adapts to acceptance likelihood. Reported: up to 5.3x speedup; 2.8x faster than EAGLE-2/3; 1.9x faster than Token Recycling; workloads SWE-Bench and Text-to-SQL. [Source](https://arxiv.org/abs/2411.04975)

### Mechanism
Build a suffix tree over prior tokens (prompt and past outputs). At each step, look up the longest matching suffix of the current context and propose its continuation as draft tokens. The draft is verified by the target as in ordinary speculative decoding.

### Evaluation
- Headline numbers above (abstract). The abstract does not give the global-tree details or frequency scoring; those need the paper body.
- Pancake's own replay (repo, recorded in MOD_PLAN line 1510 and line 1717): on 2,045 real opencode requests, the suffix index gave rounds -2.4% against lookup alone, and sub-agent requests -7.5%. The frequency-scored rule matched the simple rule (-2.45% against -2.44%), and the plan expects 0-2%. The sign convention (fewer rounds is better) must be read from `suffix_sim.py` output before this is quoted as a gain.
- MOD_PLAN line 1576 records that the paper's own figure was previously misquoted as "a few % up to ~10%"; the paper's figure is up to 5.3x on its own workloads.

### Exact or lossy
The abstract does not say. Under greedy verification a suffix draft is exact by construction, and the repo lists it as "exact (drafts)" (MOD_PLAN line 1717). Confirm the same for sampled requests.

### Read level
Abstract only for the paper. Repo side: `tools/pancake/suffix_sim.py` read (header lines 1-20); `MOD_PLAN.md` rows read at lines 1510, 1594, 1717, 1851.

### Pancake status
Partly built.
- `STRATA_SUFFIX_INDEX` (+ `_MB`, `_RULE`, `_FILE`) and `tools/pancake/suffix_sim.py` exist (MOD_PLAN line 1717).
- Status is inconsistent across the plan: line 1851 says "planned", line 1594 says "HOLD (screen pending)", and line 1510 says "A24 HOLD (screen pending)". The screen is named `24-a24-suffix-index` and is not yet run.

### Smallest test
1. CPU only: rerun `suffix_sim.py` on the current opencode store and report rounds per sub-agent and per main-agent request, with the sign stated.
2. One night-queue screen `24-a24-suffix-index` (exists per MOD_PLAN line 1717), with the first request per case judged for output quality. GPU time: not measured here.

---

## A25 - Check DraftPolicy's MTP table against measured position decay

### Takeaway
The paper's abstract says verification dominates the cost and acceptance length varies by position, request and dataset. It gives no numbers. Pancake's `DraftPolicy` uses priors that assume a linear token gain per extra row and no decay, and then replaces them with measured EMAs. The check the plan asks for already exists as `draft_report.py`, and it needs one real window trace to run.

### Claim (abstract only)
Liu, Yu, Park, Stoica, Cheung, "Speculative Decoding: Performance or Illusion?", arXiv 2601.11580 (v2 18 Mar 2026; no peer-reviewed venue listed). The abstract: a systematic study on vLLM of n-gram, EAGLE/EAGLE-3, draft-model and MTP variants across workloads, scales and batch sizes; "verification by the target model dominates the execution"; "acceptance length varies markedly across output token positions, requests, and datasets"; a substantial gap between observed and theoretical speedups. The abstract gives no speedup number and does not state when speedup fails. [Source](https://arxiv.org/abs/2601.11580)

### Mechanism (as the abstract states it)
Acceptance falls with draft position; verification is the dominant cost; so the speedup is set by the accepted-tokens-per-verify ratio, not by drafting speed. The paper's position-decay figure is not in the abstract.

### Evaluation
None in the abstract. Pancake's own priors (repo, read in full in the `draft_report.py` docstring): `mtp_tokens(t) = 1 + 0.7 (t - 1)` before any window of that size is measured (every extra row worth 0.7 tokens, no decay); `draft_rate(p) = p` (the MTP's own softmax as acceptance) before its bucket has data; a fixed prior shape `kShape` for window cost (1, 1.35, 1.7, 2.05, ...). Each prior is replaced by a measured EMA as windows come in, so a wrong prior costs only the first windows of a process, plus any size or bucket that stays rare.

### Exact or lossy
Exact. The check changes no output; it only chooses the window size.

### Read level
Abstract only for the paper. Repo side: `tools/pancake/draft_report.py` read in full (docstring, lines 1-25).

### Pancake status
Tool exists; the measurement does not.
- `tools/pancake/draft_report.py` (with `test_draft_report.py`) reads window traces (`STRATA_WINDOW_TRACE`, "STWT" magic), draft traces (`STRATA_SPEC_TRACE`) and decode timing logs (`STRATA_DECODE_TIMING=1`). It reports per-position acceptance, acceptance by MTP probability bucket, tokens per window, window cost by size, and the verify share. This is exactly the A25 check.
- MOD_PLAN line 1716 lists the window-trace run as the one GPU step shared with A22.

### Smallest test
One window-trace GPU run (`STRATA_WINDOW_TRACE=<file>`, `--spec-min-p 0`, plus `STRATA_DECODE_TIMING=1` for the verify share), then `draft_report.py` offline. This is the same run A22 needs, so it costs no extra night slot. Expected gain stays at the plan's "0-5% (est.)" until the trace is read.

---

## A26 - Draft source (MTP / lookup / suffix) as a learned action

### Takeaway
The paper learns which assistant (draft) model to use per context, as a contextual bandit trained offline. Pancake's DraftPolicy already chooses MTP, suffix or lookup per context bucket by measured cost, and the repo parks a learned selector until the logs show wrong choices. `draft_report.py` has a counterfactual that answers that question.

### Claim (abstract only)
Huang, Parthasarathi, Rezagholizadeh, Chandar, "Context-Aware Assistant Selection for Improved Inference Acceleration with Large Language Models", arXiv 2408.08470 (EMNLP 2024 long paper per the arXiv page; page is v4, Dec 2024). Abstract: a policy picks among several draft models per context, framed as a contextual bandit; trained on an offline dataset built from draft and target outputs; "can speed up performance across multiple domains when the candidate drafts are effective". The abstract gives no numbers, datasets, or exactness statement. [Source](https://arxiv.org/abs/2408.08470)

### Mechanism
Per context, score each candidate assistant and pick the one predicted to be fastest; learn the scorer from logged (context, draft, acceptance) data.

### Evaluation
None in the abstract.

### Exact or lossy
Not stated in the abstract. Verified assisted decoding is exact in distribution for sampling, so the same reasoning as A21 applies, but the paper must be read to confirm.

### Read level
Abstract only.

### Pancake status
Mostly covered by DraftPolicy.
- DraftPolicy arbitrates MTP, suffix and lookup windows by measured cost and acceptance, one shared cost table per context bucket (`draft_policy.hpp`, `cost_` array; MOD_PLAN line 712 and the local note [speculative_mtp.md](research_notes/Speed%20ideas%20from%20all%20forks/speculative_mtp.md) line 76).
- A local note records that naive "always take lookup" can lose 12% on edits, and cost arbitration recovers it (speculative_mtp.md line 146).
- MOD_PLAN line 1320 and line 1853: "Draft source as a bandit - only if DraftPolicy logs show wrong choices. [parked]".
- `draft_report.py` implements the A26 counterfactual: for every draft round, both the drafter's chain and the lookup's proposal are scored against the target's committed tokens.

### Smallest test
CPU only, if draft traces (`STRATA_SPEC_TRACE`) already exist: run `draft_report.py` and count rounds where the other source would have accepted more tokens. Only if that count is material (the owner should set the threshold) does a learned selector get a GPU test. Until then, nothing to run.

---

## A29 - Entropy of Q2_0 expert bytes and whether entropy coding pays

### Takeaway
The measurement is done and the item is closed. Q2_0 expert bytes can be compressed by at most about 12% with a block-aware coder and about 8% with a general compressor, below the plan's 20% threshold. The IQ types are essentially random bytes (about 3% at most, 0% real). The literature (EntroLLM, Huff-LLM) reports savings on uint8/uint4 integer formats and on its own hardware simulations, which do not transfer to a 2-bit block format.

### Claim (literature, abstract/search excerpts only)
- EntroLLM, Sanyal, Datta, Mukherjee, Chinchali, Orshansky, arXiv 2505.02380 (v4, Jan 2026). Huffman coding applied after quantization to integer weights; reports up to 30% storage savings over uint8 and up to 65% over uint4 on smolLM-1.7B, phi3-mini-4k and mistral-7B; adds a parallel decoder for edge devices. Source: [arXiv abstract](https://arxiv.org/abs/2505.02380); summary from search excerpts.
- Huff-LLM, Yubeaton et al., arXiv 2502.00922 (Feb 2025). Lossless compression that keeps weights compressed through memory and on-chip buffers; each weight split into subfields, all but the sign Huffman-coded; reports up to 32% size reduction, 31% latency, 26% energy, from the authors' simulations and an FPGA prototype. SOTAVerified lists the numbers as not independently reproduced. Source: [arXiv](https://arxiv.org/abs/2502.00922); summary from search excerpts.

### Mechanism
Lower-entropy symbols (integer codes after quantization, or sub-fields of floats) can be Huffman-coded, so fewer bytes cross the bus or disk; decoding costs compute on the hot path.

### Evaluation
- The literature numbers above are against dense uint8/uint4 or dense-model baselines, not against a 2-bit block format with a per-block fp16 scale.
- Pancake's measurement (repo, run 2026-11-10, `pancake-docs/batch2/A-small-items.md` lines 32-53):

| file / tensor type | sample | crumb entropy (of 2 bits) | per-position bound saving | best real compressor |
|---|---|---|---|---|
| Q2_0 file, Q2_0 experts | 12 MiB | 1.843 | 11.7% | zstd -19: 8.0% |
| IQ2_XS file, Q2_0 (ffn_down) | 4 MiB | 1.837 | 12.0% | zstd -19: 8.3% |
| IQ2_XS file, IQ2_XXS / IQ2_S / IQ1_M | 2-5 MiB each | 1.999 | 3.3-3.4% | none (0-0.1%) |

  The sample is about 12 MiB across 12 tensors; the estimator bias is below 0.0002 bits. The note's own rerun suggestion is `--tensors 48 --max-mb 48`.

### Exact or lossy
Lossless coding, so exact. The result is about size, not output.

### Read level
Literature: search excerpts and arXiv abstracts only. Repo: `tools/pancake/expert_entropy.py` header read in full (lines 1-30); the result table in `A-small-items.md` read in full.

### Pancake status
Built and run. The tool is `tools/pancake/expert_entropy.py` with `test_expert_entropy.py`. Plan rows conflict: line 1718 and line 1523 say closed; line 1867 still says "parked -> run the 5-minute check" and "~0 expected (2.25 bpw)". Row 1867 should be updated to closed, with the numbers above.

### Smallest test
None needed. Optional CPU-only rerun with more tensors (`--tensors 48 --max-mb 48`) if the owner wants a larger sample before the item is written off. No GPU.

---

## Gaps (batch-wide)
- No paper body was read for any of the five papers; every paper figure is an abstract figure. Full texts (HTML/PDF) are needed before any number from them goes into a design decision.
- The request temperature opencode sends is not established (see A21). This decides whether A21 has any traffic to help.
- The sign convention of the A24 replay (`rounds -2.4%`) needs confirming from `suffix_sim.py` output.
- MOD_PLAN is internally inconsistent on A24 (planned / HOLD) and A29 (closed / parked). These rows should be fixed by the owner or the plan maintainer.
