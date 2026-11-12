# T2 ID-level source check (7 arXiv papers)

Read-only check. PDFs fetched from arxiv.org/pdf/<id> and read with pdftotext. Page = PDF page.

| arXiv | Cited claim (where in repo) | Paper value (page) | Verdict | Plan row (status) |
|---|---|---|---|---|
| 2510.04618 ACE | +10.6% agents, +8.6% finance (batch3/agentloop.md; citecheck 16.8) | +10.6% on agents, +8.6% on finance (abstract p1); "8.6% on domain-specific benchmarks" (p2); 10.6% avg (p7) | VERIFIED | A39 (parked) |
| 2510.11713 LRMs Interruptible | Accuracy drops up to 60% when updates arrive late (berkeley notes) | "dropping by up to 60% when new information is introduced" (p1, p2) | VERIFIED | A17 (partly done: grace 512 + close_retry shipped; per-level caps still waiting for agent-turn data) |
| 2601.09001 Entropy Sentinel | Judge-score correlation r = 0.91 for some models on chat traces (avalon report) | r = 0.91 for PHI-3.5-MINI, 95% CI [0.68, 0.98] (p7); r = 0.79-0.91 for PHI, LLAMA-3.1-8B, both GEMMA-3 on OOD categories (p7-8); paper says other models do not correlate as well | VERIFIED (number is one model's point estimate, not a general result) | A20 (open: log + opt-in close coded on feat/item-34-rumination; entropy log new) |
| 2602.03495 DALI | "6-13% on other platforms"; static PcieAuto lost that much here (MOD_PLAN A45) | No 6-13% figure and no PcieAuto baseline found in text. Paper's headline speedups are larger: 3.97x / 2.16x / 1.48x / 1.32x vs baselines (p3) | NOT FOUND | A45 (planned) |
| 2604.18556 GSQ | SWE-bench Verified 95.7% of BF16 (card); 256 of 512 experts, ~3.5 bpw LUT types (MOD_PLAN A28) | No SWE-bench result, no 512-expert model, no LUT types. Paper quantizes Llama-3.1-8B/70B-Instruct and Kimi-K2.5, at 2-3 bpp and mixed 2.37/2.62 avg bits (p1, p7, p8). Matches for "LUT" were only "GPTQ" substrings | NOT FOUND (both numbers) | A28 (planned, D space) |
| 2605.00649 RCO | "256 of 512 experts, ~3.5 bpw LUT types" for the Coder variant (MOD_PLAN A28) | 512 experts/layer is Qwen3-Coder-Next; 256 kept = 50% expert PRUNING with uniform allocation (p21), not quantization. 3.5 bit is mixed-precision on Qwen1.5-MoE-A2.7B (p8), not Coder. No LUT types | DISCREPANT: 256/512 is a pruning setting; 3.5 bit is a different model; LUT not in paper | A28 (planned, D space) |
| 2609.03114 Coordinate-Based Hill Climbing | Plan gain "1-3% (est.)" (A34); repo selftune.md: +5.5-8.5% on A100 thread blocks | "5.5-8.5% improvement over default configurations" on NVIDIA A100 for thread-block allocation (p1, p3, p18). CPU result is 1.06-1.28x geomean (p1) | VERIFIED for 5.5-8.5%. The "1-3%" in the plan is the plan's own estimate, not a paper claim | A34 (planned: sweep.py, 7861dc8c1) |

## Notes
- Plan row status column read from /home/dan/clone-of-d/src/Strata/MOD_PLAN.md. All rows are open except A17, which is partly done.
- A28 cites two papers for one "Coder variant" line. The numbers come from neither paper as stated. Fix the row before any D-space work: drop the 95.7% (card, not paper) and the "LUT" claim, and re-derive the 256/512 setting from RCO pruning if that is what is meant.
- A45's "6-13% on other platforms" should be marked as an unmeasured estimate, not a DALI result.
- Triage: TRIAGE.md lists all seven IDs on its open-plan list; this check covers only the ID-level claims.
