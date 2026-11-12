# Round 3 batch 6: A37, A38, A39, A42, A43 (2026-11-12)

Read-only research. No code edits, no builds, no GPU. The only local computation was read-only SQL on the opencode session store (`/home/dan/.local/share/opencode/opencode.db`, opened `mode=ro`) and read-only Python over the existing probe. Sources read: arXiv abstracts for all five papers; HTML full text (section-level, through a summarizing fetch) for FLUTE, RLM, Beyond Accuracy, Slipstream and ACE; the Type-Constrained paper has abstract only (HTML 404, PDF not decodable). Where a number comes from an earlier round it is marked "prior round".

Read-level legend: **full** = HTML full text read through the fetch tool (the fetch truncates long pages, noted where it did); **abstract** = abstract page only.

Summary table

| Item | Paper read level | Exact / lossy | Already in Pancake or opencode | Verdict proposed | Smallest test (cost) |
| --- | --- | --- | --- | --- | --- |
| A37 LDS LUT dup | FLUTE full (HTML) | exact (our variant); FLUTE itself approximate | Built (`STRATA_IQ_LUT_DUP`, opt-in), measured not kept (kern-hip) | Close: NO-GO at the 2% bar; keep the revert decision | rocprofv3 on expdec_bench 16/42, 1 night slot (~10 min GPU); not recommended |
| A38 behaviour counters | Beyond Accuracy full text to 100k of 168k chars (section 6 not read) | log exact | Built (`STRATA_AGENT_COUNTERS=1`), log only, default off, not in presets | Turn on for real sessions (log only) | Counters on live traffic, CPU only; opencode.db pilot below (crude, confounded) |
| A39 async / delta compaction | Slipstream full (HTML); ACE full to 100k of 131k chars | lossy | Design note only (batch2); opencode has its own compaction (43 events in the store) | Stay parked; measure compaction latency share first | Share of turn time spent in opencode compaction, from existing turn logs (CPU, minutes) |
| A42 externalize | RLM full (HTML) | lossy | Built (`STRATA_EXTERNALIZE_TOKENS`, head/tail + path, opt-in); opencode truncates big outputs to a file; caveman retrieve already used | Test the arm, but expect the gain to be the offload baseline, not RLM | Share of big tool outputs on opencode.db (done, below); live A/B arm only after counters |
| A43 oldString constraint | Type-Constrained paper abstract only | lossy (generation constraint) | Check + log + note built; constrained decoding not built; opencode already fails on not-found and multiple matches | Close the constrained-decoding part; keep log only | Failure count from opencode.db (done: 0 not-found in 339 edit calls); log switch for a week (CPU) |

---

## A37 LDS bank-conflict check of the HIP IQ kernels; duplicated LUT (FLUTE)

### Takeaway
FLUTE's bank-conflict fix is real for its own NVIDIA kernel: a non-uniform LUT read with 8-way conflicts (4-bit) is removed by duplicating the vectorized table across bank pairs, and the paper reports 2-4x kernel speed-ups. For Pancake the duplicated codebook is already coded and was measured: the kernel is 1.7% of GPU time, the ceiling is 0.15-0.55%, and 16 copies halve occupancy. The item should close as not kept; no new test is justified.

### Cited Findings
- Claim: FLUTE is a LUT-quantized GEMM engine; at batch < 32 and group size 128 it is "2-4x faster than existing GEMM kernels", and it gives "an end-to-end throughput increase of 1.5 to 2 times" with LLaMA-3 in vLLM. Source: [FLUTE abstract](https://arxiv.org/abs/2407.10960); [FLUTE HTML](https://arxiv.org/html/2407.10960) (full, section 4.2).
- Mechanism (section 3.2, "Vectorized Lookup in Shared Memory"): a table of every pair of 4-bit values, slightly more than 1 KB, copied to shared memory at kernel start. Bank conflicts (section 3.2, "Reducing bank conflicts"): "up to 8-way bank conflicts (4-bit) or 2-way bank conflicts (3-bit)"; fix: "we duplicate the 4-bit vectorized lookup table multiple times, placing copies in different memory banks"; the copy count is chosen by per-shape autotuning (section 4.1). Source: [FLUTE HTML](https://arxiv.org/html/2407.10960).
- Exactness: quantization is approximate (weights mapped to nearest quantile; section 4.2); the kernel accumulates in FP32 in registers and reduces across blocks in FP16 (section 3.3). The paper makes no bit-exact claim against `torch.mm`. Source: [FLUTE HTML](https://arxiv.org/html/2407.10960).
- Baseline is 16-bit `torch.mm` on A6000 and A100, shapes from Llama-3-8B and 70B, W4G128. Source: [FLUTE HTML](https://arxiv.org/html/2407.10960).
- Pancake (prior round, code read this round): `STRATA_IQ_LUT_DUP=8|16` is in `src/kernels/cuda/iq_kernels.cu` (`lut_dup_slot`, `expdec_gu_kernel` DUPC); it is opt-in and exact (same words, parity tests in `src/kernels/iq_multi_parity.cpp`). Source: [batch2 A37 note](/home/dan/clone-of-d/src/Strata/pancake-docs/batch2/A37-gpu-kernels.md).
- Bank model (`tools/pancake/probes/lds_lut_conflicts.py`, a model, not a measurement): plain 5.85, x8 3.99, x16 2.00 LDS clocks per wave lookup. The probe names rocprofv3 counters `SQC_LDS_BANK_CONFLICT` and `SQC_LDS_IDX_ACTIVE` as the confirmation; those have not been run.
- Measured (prior round, kern-hip): kernel is 1.7% of GPU time; ceiling 0.15-0.55%; x16 uses 249 VGPRs, 43.25 KB LDS, 4 waves (x8: 179 VGPRs, 27.25 KB, 8 waves). Verdict: "not worth keeping", revert d11a9ca09 unless one `expdec_bench` run shows >= 10% on 16/42 gate/up. Source: `pancake-docs/batch3/kern-hip.md` section A37 (local).
- Kernel gate (prior round): A37 NO-GO even at the optimistic bound. Source: `tools/pancake/kernel_gate.py` candidate A37 and the MOD_PLAN 16.7 status (local).
- Plan row A37 (MOD_PLAN 16.7) says "0-10% of GPU expert time if conflicts exist". The measured ceiling (0.15-0.55%) is far below that; the plan row is stale on the upper end.

### Inferences
- The conflict model is for a 32-bank x 4 B layout. The RDNA3 LDS bank layout used by gfx1100 was not checked against AMD documentation this round, so the 5.85 -> 2.00 clock gap is a model number, not a 7900 XT number.
- Even if the conflict removal were perfect, the LUT read is a fraction of a 1.7%-of-GPU kernel; the x16 layout's occupancy loss matches or exceeds the gain. The only path that could change this is a much larger kernel share (for example after A33 or a new expert kernel), which would need the kern-hip ceiling re-run.

### Gaps
- No measured counter (SQC_LDS_BANK_CONFLICT) exists for the gfx1100 build; the probe is still a model.
- The FLUTE exactness is for NVIDIA with its own table layout; our variant is exact by construction, not by the paper.

### Fields
- Read level: full (FLUTE HTML, section-level).
- Exact / lossy: our variant exact; FLUTE approximate.
- Already has it: yes, coded and measured; not kept.
- Smallest test and cost: `rocprofv3` counters on `expdec_bench 300 0 20 16/42` for `dec` vs `dec:l16`, one night-queue slot, about 10 minutes of GPU. Not recommended: the kernel-gate verdict is NO-GO even at the upper bound. Recommended action is the kern-hip revert decision, not a test.

---

## A38 Behaviour counters: edits without test, reads without edit (Beyond Accuracy)

### Takeaway
The paper is observational: 404 multi-hunk bugs, 1,616 trajectories, four agents. Failed repairs use more tokens and time, and the failure pattern is "over-modification without validation" or "over-exploration without action". It does not define "edits without test" or "reads without edit" as metrics; those are our mapping of its WRITE, TEST and READ categories. Counters are cheap and exact, so logging them is worth doing; no threshold should come from the paper.

### Cited Findings
- Claim and data: 404 multi-hunk bugs from PolyHunk, 1,616 trajectories, four agents (Claude Code, Codex, Gemini-cli, Qwen Code); repair accuracy 26.98% (Qwen Code) to 92.82% (Claude Code). Source: [arXiv 2511.11012 abstract](https://arxiv.org/abs/2511.11012); [HTML](https://arxiv.org/html/2511.11012) (full to 100k of 168k chars).
- Cost of failure: "33%-440% more input tokens and 35%-330% longer execution time" (abstract); Gemini-cli failed repairs +440.4% input tokens (section 4.9.1); mean runtime 723.91 s failures vs 285.57 s successes (section 4.9.2). Source: [HTML](https://arxiv.org/html/2511.11012). Prior-round note corrected the plan's "1.4-4.4x" to 1.33-5.4x (citecheck, `pancake-docs/batch3/citecheck.md`).
- Behaviour categories (section 4.9.3, Table 10): categories WRITE, READ, TEST, BUILD, SEARCH_CONTENT, SEARCH_FILES, NAVIGATE. Gemini-cli "over-modification without validation": WRITE 25.8% -> 42.9% and TEST 23.2% -> 17.5% from pass to fail. Claude Code "over-exploration without action": SEARCH_CONTENT 8.9% -> 14.6% in failed repairs while WRITE falls. Modification-to-comprehension ratio (WRITE:READ, not formally defined) Gemini-cli 0.97:1 pass -> 1.99:1 fail. Source: [HTML](https://arxiv.org/html/2511.11012).
- Statistics: no correlation coefficients, no significance tests; descriptive percentages (section 4.8-4.9). Agent and model are confounded (section 4.3); only 12 bugs unsolved by all agents (section 4.7). Source: [HTML](https://arxiv.org/html/2511.11012).
- Claim wording check: the abstract says "no universal pass/fail pattern" is claimed by the prior-round note; this round's fetch did not confirm that phrase, so it is not cited here.
- Pancake (prior round, batch2 A38 note): `STRATA_AGENT_COUNTERS=1` writes an `agent` object per request (streaks, `code_edits_since_check`, `files_edited`, `edits_unread`, A43 states, `ends_unchecked`), numbers only, default off, log only. Not yet in a preset. Source: `pancake-docs/batch2/A38-agent-loop.md` lines 14, 29-38, 208 (local).

### Inferences (from the opencode store, read-only, this round)
- `/home/dan/.local/share/opencode/opencode.db`: 84 sessions with tool parts, 398 completed edit/write calls, 21 sessions with at least one completed edit. In 11 of those 21 (52%) no bash command matching a test pattern (regex: pytest, ctest, cargo test, npm test, go test, make test, a bare `test`) came after the last edit. This is an upper bound on "edits without test": the regex is crude and short edit-only sessions (docs, notes) are included.
- 57 sessions had reads; 38 of them (67%) had no edit at all. Most are research or Q&A sessions, so this does not measure an agent failure.
- Conclusion: the signal exists in the store, but it is confounded by session type. The counter must be split by repo or task type before any intervention.

### Gaps
- Section 6 (threats to validity) and the rest of the paper beyond 100k chars were not read.
- No paper gives a threshold for "too many reads" or "edits without test"; any threshold must come from our own logs.

### Fields
- Read level: full text, truncated at 100k of 168k chars (section 6 not read).
- Exact / lossy: log exact (counters); any nudge derived from them would be lossy.
- Already has it: yes, log-only counters built; not on by default.
- Smallest test and cost: turn on `STRATA_AGENT_COUNTERS=1` with the turn log for real sessions and read the counters for a week. CPU only, no GPU. The opencode.db pilot above is the zero-cost version and already run.

---

## A39 Asynchronous compaction in tool windows and delta compaction (Slipstream, ACE)

### Takeaway
Slipstream runs the compactor in parallel with the agent and has a judge validate the summary; its gain is latency, up to 39.7% vs synchronous compaction on Qwen3.5-9B / Seed-OSS-36B with vLLM on H100, with accuracy gains of 1.3-8.8 points. "Delta compaction" is not in Slipstream: ACE is the source of incremental delta updates, but ACE updates a playbook, not a transcript window. Pancake and opencode need a spare GPU slot and a compaction trigger they do not control; this stays parked.

### Cited Findings
- Slipstream mechanism (section 1, 4): compaction runs in parallel with continued execution on the uncompacted context; the steps taken meanwhile form a "next-k trajectory" (average overlap 2.1 steps on BrowseComp, 3.7 on SWE-bench Verified; section 4.1). Source: [Slipstream HTML](https://arxiv.org/html/2605.08580) (full to the fetch limit); [abstract](https://arxiv.org/abs/2605.08580).
- Judge (section 4.2, appendix B.1): plan-level and statement-level checks, each 0-10; overall = round(0.5 x plan + 0.5 x information); accept at >= 7; the default judge is the agent's own model. Rejection 1.0-3.5% on BrowseComp and 5.4-8.5% on SWE-bench Verified. Source: [HTML](https://arxiv.org/html/2605.08580).
- Latency: "up to 39.7%" (abstract); section 5.2 gives "11.3-39.7%" across configurations vs the synchronous baseline (Sync). Compaction was 26-44% of Sync per-query latency (section 3). Hardware: Qwen3.5-9B on one H100 80GB; Seed-OSS-36B-Instruct on two; vLLM 0.19.1. Source: [HTML](https://arxiv.org/html/2605.08580).
- Accuracy (Table 1, section 5.2): SWE-bench Verified +2.6 to +8.8 points (Qwen3.5-9B 22.0% -> 30.8% at a 6k threshold); BrowseComp +1.3 to +4.6 (52.7% -> 57.3% at 48k). Source: [HTML](https://arxiv.org/html/2605.08580).
- Ablation (section 5.3): "Async-only" captures most of the latency benefit, but its accuracy stays close to Sync; the accuracy gain is attributed to the judge. Source: [HTML](https://arxiv.org/html/2605.08580).
- Limitations (no dedicated section): errors that surface after the window cannot be recovered; two models and two workloads only; no monotonic accuracy trend across thresholds (section 5.2). Source: [HTML](https://arxiv.org/html/2605.08580).
- ACE (arXiv 2510.04618, ICLR 2026 per comments): "incrementally produces compact _delta contexts_" instead of regenerating (section 3.1). Context collapse example (section 2.2): context of 18,282 tokens at accuracy 66.7 collapses to 122 tokens and accuracy 57.1, below the 63.7 no-adaptation baseline. Source: [ACE HTML](https://arxiv.org/html/2510.04618) (full to 100k of 131k chars).
- ACE numbers: AppWorld offline with ground-truth labels, ReAct base 42.4, ReAct + ACE 59.4 (+17.0), vs ICL 46.0 and GEPA 46.4; FiNER offline with labels 78.3 vs base 70.7 (+7.6). Latency: AppWorld offline vs GEPA 53,898 s -> 9,517 s (-82.3%); FiNER online vs DC (CU) 65,104 s -> 5,503 s (-91.5%), token cost $17.7 -> $2.9 (-83.6%). Source: [ACE HTML](https://arxiv.org/html/2510.04618) section 4.3, 4.4, 4.7 (Table 4).
- ACE limitations (section 5, 4.4, appendix A.1): needs a strong Reflector; degrades without reliable feedback; smaller models (Llama-3.3-70B) show smaller gains. Source: [ACE HTML](https://arxiv.org/html/2510.04618).
- Pancake/opencode status (prior round, batch2 A38 note, section A39): design only. The blockers recorded there hold: one GPU slot is shared with A9; the swap must coincide with opencode's own compaction, which the server cannot see; the delta form needs a stable playbook the client keeps, and opencode has none. Source: `pancake-docs/batch2/A38-agent-loop.md` section A39 (local).
- opencode store (this round, read-only): 43 `compaction` parts across the 191 sessions in `opencode.db`.

### Inferences
- The -39.7% is the top of an 11.3-39.7% range against a synchronous baseline that spends 26-44% of query time compacting, on H100 with a 9B-36B model. On the local 7900 XT with a MoE model and a single GPU slot, the share is unknown; the gain only transfers if compaction is a comparable share of turn time.
- ACE's delta mechanism fits an agent memory, not a tool-window compaction. The MOD_PLAN row combining them mixes two different mechanisms.

### Gaps
- Slipstream's "tool windows" wording is not in the text read; the read did not confirm that the paper uses tool windows as the unit.
- No token-count figures for ACE's delta mechanism were read.

### Fields
- Read level: full text (Slipstream HTML to fetch limit; ACE HTML to 100k of 131k chars).
- Exact / lossy: lossy (summary replaces context).
- Already has it: no. opencode has its own synchronous compaction (43 events in the store); Pancake has no compaction.
- Smallest test and cost: the share of turn time spent in opencode's compaction turns, from the existing turn logs (`STRATA_TURN_LOG`), CPU only, minutes. If it is under 10% of turn time, stop. A real test needs a spare judge slot and is a GPU item for the night queue.

---

## A42 Externalize huge inputs instead of compacting (Recursive Language Models)

### Takeaway
RLM keeps the long prompt in a REPL variable and recurses with sub-calls; its gains (median +26% vs a compaction agent on GPT-5) are real on its own benchmarks, but the comparison that matters for us is "Claude Code + offloading", where RLM's margin is much smaller and mixed. Pancake's built `STRATA_EXTERNALIZE_TOKENS` is head/tail excerpts plus a file path, which is the offload baseline, not RLM. The measured share of large outputs in opencode says where to test it.

### Cited Findings
- Claim: RLM handles inputs "two orders of magnitude beyond model context windows", and outperforms vanilla LLMs and scaffolds at comparable cost (abstract). Source: [arXiv 2512.24601 abstract](https://arxiv.org/abs/2512.24601); [HTML](https://arxiv.org/html/2512.24601) (full to 100k of 133k chars).
- Mechanism (section 2): the prompt is a string variable in a persistent Python REPL; the root model sees only length and a prefix; REPL output returned to history is truncated; sub-calls via `llm_query` or a sub-RLM, results stored in variables; ends with `FINAL()` / `FINAL_VAR()`. Source: [HTML](https://arxiv.org/html/2512.24601).
- Table 1 (GPT-5, depth 1), CodeQA / BrowseComp+ (1K) / OOLONG / OOLONG-Pairs:
  - Compaction agent: 58.0 / 70.5 / 46.0 / 0.1
  - Claude Code (Opus 4.1): 12.0* / 0.0* / 40.2 / 0.1 (* hit input context limits)
  - Claude Code + offloading (Opus 4.1): 62.0 / 84.0 / 48.0 / 6.5
  - RLM (depth 1): 62.0 / 91.3 / 56.0 / 58.0
  Source: [HTML](https://arxiv.org/html/2512.24601), Table 1.
- Median +26% (abstract) reproduced from Table 1 by this round: RLM over compaction per benchmark is +6.9% (CodeQA), +29.5% (BrowseComp+), +21.7% (OOLONG), and very large on OOLONG-Pairs (58.0 vs 0.1); the median of the four ratios is about +25.6%, consistent with the abstract. The arithmetic is ours.
- RLM over "Claude Code + offloading": CodeQA 62.0 vs 62.0 (0), BrowseComp+ 91.3 vs 84.0 (+8.7%), OOLONG 56.0 vs 48.0 (+16.7%), OOLONG-Pairs 58.0 vs 6.5. The offloading arm is the closest analogue to A42 and captures most of the gain on three of four tasks. Source: Table 1 in [HTML](https://arxiv.org/html/2512.24601); arithmetic ours.
- Cost (Table 1, observation 4): GPT-5 RLM depth 1 about $0.11 (CodeQA), $0.99 (BrowseComp+), $0.43 (OOLONG), $0.33 (OOLONG-Pairs); compaction $1.31, $0.57, $0.13, $0.13; Claude Code + offloading $1.25, $2.03, $0.98, $2.99. The paper says RLM is "cheaper or comparable" to most baselines, and the mean is higher than the median because of outliers. Source: [HTML](https://arxiv.org/html/2512.24601).
- Limitations (section 7, appendix B, section 5): sub-call cost can explode; sync sub-calls are slow; smaller models (Qwen3-8B) struggle; Qwen3-Coder made more syntax errors that propagate to sub-calls; the RLM-Qwen3-8B gain is +28.3% over its base (abstract). Source: [HTML](https://arxiv.org/html/2512.24601).
- Pancake (prior round, batch2 A38 note): `STRATA_EXTERNALIZE_TOKENS=N` writes tool results over N tokens to a file and keeps 400 head + 400 tail tokens in the prompt; skips `read` by default; lossy; opt-in. Source: `pancake-docs/batch2/A38-agent-loop.md` lines 75-98 (local).
- opencode store (this round, read-only, `opencode.db`): 12.6M characters of tool output in total. Outputs over 6,400 characters: 499 calls, 73.6% of all output characters. By tool, the total share is caveman_caveman_retrieve 46.8%, bash 19.3%, read 17.7%, webfetch 4.8%. Of the 499 large outputs, 269 are caveman retrieve, 81 bash, 75 read. Caveman retrieve already returns stored originals on request, so it is a form of externalize that the model calls by hand.

### Inferences
- The best-supported A42 arm is the offload baseline with head/tail plus a retrieval tool, which Pancake and opencode both already have in a weaker form. RLM's extra gain comes from programmatic sub-calls over the stored text, which opencode does not do and which costs model calls.
- Read and bash outputs are the candidates for externalization (about 37% of output characters together). caveman retrieve is already the escape hatch, so externalizing it again would double-wrap.
- The RLM paper's tasks are long-context QA and aggregation, not multi-turn coding repair. The +26% does not transfer to a coding agent without a coding benchmark.

### Gaps
- No direct comparison of RLM vs a head/tail excerpt baseline (the closest we have is "Claude Code + offloading", which is not identical to head/tail).
- The paper does not give a per-task token budget comparable to opencode's context limit.

### Fields
- Read level: full text (HTML to 100k of 133k chars; the appendix tail not read).
- Exact / lossy: lossy.
- Already has it: partly. Pancake: `STRATA_EXTERNALIZE_TOKENS` built, opt-in. opencode: big outputs go to a file already (prior round), and caveman retrieve is in use.
- Smallest test and cost: the share analysis above (done, CPU). The live A/B needs the counters from A38 first, then a few days of real opencode traffic with the arm on (per-server A/B through `/metrics`). No GPU needed for the log; the arm changes what the model sees, so quality must be judged on real turns.

---

## A43 oldString substring constraint in the edit tool (Type-Constrained Code Generation)

### Takeaway
The paper constrains LLM output to well-typed programs (TypeScript, HumanEval and MBPP). It does not constrain edits or substrings of existing text, so it gives no evidence for an oldString constraint. The remaining value is the cheap check and log already built. In opencode's own history the oldString-not-found failure does not appear at all, so the expected gain is small and the constrained-decoding part should stay unbuilt.

### Cited Findings
- Claim: LLM output often fails to compile; the authors propose type-constrained decoding with "prefix automata" and a "search over inhabitable types"; compile errors reduced "by more than half" on HumanEval and MBPP; functional correctness "significantly increases" in synthesis, translation and repair; models include open-weight above 30B. Source: [arXiv 2504.09246 abstract](https://arxiv.org/abs/2504.09246). Read level: **abstract only**. The HTML URL returned 404 and the PDF was not decodable by the fetch tool, so section-level figures (per-model numbers, overhead) are not available here.
- Scope: the constraint is on whole-program generation through next-token decoding; applicability to edits is not addressed in the abstract (Source: same).
- Pancake (prior round, batch2 A38 note): `STRATA_EDIT_OLDSTRING_CHECK=log|note` classifies each edit's oldString as exact, multi-match, or unseen, using the history the server knows; log, metric and a user-role note; default off. Constrained decoding not built. Source: `pancake-docs/batch2/A38-agent-loop.md` lines 100-130 (local).
- Failure rate: MOD_PLAN row A43 says "parked: 1 failure in 180 edits" (local, prior round).
- opencode store (this round, read-only): 402 completed-or-error edit/write calls (339 `edit`). Failed calls: 4 total: 2 "No changes to apply: oldString and newString are identical." and 2 write-tool schema errors. Zero output texts containing "Could not find", "Found multiple" or "not found".
- Edit-failure texts "Could not find oldString", "Found multiple matches for oldString", "oldString not found in content" were checked against the installed opencode by the batch2 note (not re-verified this round: the binary search did not find them).

### Inferences
- In this store, the substring failure rate is 0 of 339 edits. This contradicts the MOD_PLAN figure of 1 in 180 only in sample size: 0 of 339 is consistent with 1 in 180, so the store cannot tell them apart. The upper bound at 95% for 0 of 339 is about 0.9% (rule of three: 3/339).
- The whole-failed-turn cost avoided is at most one failure per 180 edits times the cost of a failed turn (batch2 A38 gives 1.33-5.4x tokens for failed multi-hunk repairs, a different task). The expected benefit is below the noise of any session A/B.

### Gaps
- Full paper text not read; the claim that the constraint does not cover edits rests on the abstract.
- The "read-before-edit" enforcement that makes the check redundant is recorded in the batch2 note; not re-verified this round.

### Fields
- Read level: abstract only.
- Exact / lossy: lossy (the generation constraint changes outputs; the edit check is exact and only logs or notes).
- Already has it: partly. Check + log + note built; opencode already fails loudly on not-found and multi-match; `edits_unread` counter exists.
- Smallest test and cost: the store count above (done, CPU). A log-only week with `STRATA_EDIT_OLDSTRING_CHECK=log` gives more samples; cost is nothing beyond the normal session. Constrained decoding: not worth building on this evidence.

---

## Open questions for the owner

1. A37: close as NO-GO (kern-hip revert decision), with no new GPU run. Confirm, or name the case where the kernel share would change.
2. A38: turn `STRATA_AGENT_COUNTERS=1` on for real opencode sessions (log only). The DB pilot is crude; the counters are the better measurement.
3. A39: measure opencode compaction turn share from existing turn logs before any design work.
4. A42: the right comparison is against an offload-plus-retrieve arm, not against compaction; that is what the built switch approximates.
5. A43: close constrained decoding; keep the log.

Sources (all URLs used):
- https://arxiv.org/abs/2407.10960 ; https://arxiv.org/html/2407.10960
- https://arxiv.org/abs/2511.11012 ; https://arxiv.org/html/2511.11012
- https://arxiv.org/abs/2605.08580 ; https://arxiv.org/html/2605.08580
- https://arxiv.org/abs/2510.04618 ; https://arxiv.org/html/2510.04618
- https://arxiv.org/abs/2512.24601 ; https://arxiv.org/html/2512.24601
- https://arxiv.org/abs/2504.09246 (html 404; pdf not decodable)
- Local: `pancake-docs/batch2/A37-gpu-kernels.md`, `pancake-docs/batch2/A38-agent-loop.md`, `pancake-docs/batch3/kern-hip.md`, `pancake-docs/batch3/agentloop.md`, `pancake-docs/batch3/citecheck.md`, `tools/pancake/probes/lds_lut_conflicts.py`, `tools/pancake/kernel_gate.py`, `/home/dan/.local/share/opencode/opencode.db` (read-only)
