# Citation check: MOD_PLAN Phase 16 and Phase 15, README 4C (2026-11-11)

Owner: citecheck. Branch dev/p16/citecheck (worktree Strata-p16-citecheck, from dev/phase16-academic 727eb8a62).
Read-only check. MOD_PLAN.md and README.md were not edited.

## Result in short

- 85 arXiv ids are cited in scope (Phase 16 sections, the Top 10 note, the Method-validity section that is on the
  branch tip 3f4ba97cf, and README 4C). All 85 resolve. Each was checked against its arXiv record (title, date,
  comments) and its abstract (numbers and key names).
- 12 non-arXiv URLs were fetched. Venues not on an arXiv record were checked with web search.
- Phase 15 cites no arXiv id, no URL and no paper venue. Its sources are fork commit hashes, fork names and local
  reports. The local report it names (`pancake-docs/research/engine-speed/speed-ideas-all-forks/report.md`) exists.
- Section A: 7 items are wrong, misquoted, or cite a source that does not make the claim.
- Section B: 19 items have a figure or claim that is not in the cited abstract. Check the body or drop them.
- Section C: the rest are OK, some with a note.
- The Phase 16 Batch 2 and status tables cite no papers.

## Decision: XAttention id

**XAttention is arXiv 2503.16428.** The arXiv record is "XAttention: Block Sparse Attention with Antidiagonal Scoring"
(Ruyi Xu et al., 2025-03-20). Its abstract says "up to 13.5x acceleration in attention computation".

**arXiv 2502.17421 is LongSpec** ("LongSpec: Long-Context Lossless Speculative Decoding with Efficient Drafting and
Verification", Penghui Yang et al., 2025-02-24, ACL 2025 main).

- The MIT report (`pancake-docs/research/avalon/literature/mit-harvard/report.md`, research_notes attention_kernels.md) gives 2503.16428 for
  XAttention. That is right.
- The Asia report (`pancake-docs/research/avalon/literature/asia/report.md`, P8 and hk_taiwan_singapore.md) gives 2502.17421 to LongSpec.
  That is right.
- The plan note at the end of Top 10 says "README section 4C lists XAttention at arXiv 2502.17421". That is not true for
  `README.md`. The `README.md` section 4C row (line 273 on dev/phase16-academic; line 272 in the main checkout on
  port/rdna3-decode) already says 2503.16428.
- The file that really has the wrong id is `pancake-docs/README.md` line 180 (its section 4C), and
  `pancake-docs/RESEARCH_AND_FINDINGS.md` line 208 (the XAttention row, with the link arxiv.org/html/2502.17421).
  Fix those two. `README.md` needs no XAttention change.
- The plan note should be corrected to name `pancake-docs/README.md` (not `README.md`) and to say the id is 2503.16428.

Also in the same README 4C table: the SpargeAttn row is wrong (section A, item A1).

## A. Wrong or misquoted (fix before the plan or README is used)

| # | Item | Cited id / URL | Problem | What the source says | Fix |
| --- | --- | --- | --- | --- | --- |
| A1 | README 4C, SpargeAttn row (not in the plan) | arXiv 2411.10958 | Wrong paper. 2411.10958 is "SageAttention2". | SpargeAttention is arXiv 2502.18137, "SpargeAttention: Accurate and Training-free Sparse Attention Accelerating Any Model Inference", 2025-02-25. The row says "Tsinghua, 2024". | Id 2502.18137, year 2025. The Asia report (pancake-docs/research/avalon/literature/asia/report.md, line 68) gives the same wrong id for SageAttention/SpargeAttn. |
| A2 | 16.8 A38, Beyond Accuracy | arXiv 2511.11012 | Figure misquoted. Plan: "failed runs cost 1.4-4.4x tokens (authors)". | Abstract: "failed repairs consume substantially more resources (33%-440% more input tokens) and require longer execution time (35%-330%)". 440% more is 5.4x, not 4.4x, and the range is 1.33x-5.4x. | Quote the abstract range, in the abstract's own words. |
| A3 | 16.4 A24, SuffixDecoding | arXiv 2411.04975 | Figure and label do not match the source. Plan: "a few % up to ~10% on edits (authors)". | Authors report "speedups of up to 5.3x" on SWE-Bench and Text-to-SQL, 2.8x faster than EAGLE-2/3 and 1.9x faster than model-free methods. | Remove "(authors)" from the few-% estimate, or quote 5.3x with its benchmark. Mark the few-% number as our estimate. |
| A4 | 16.3 A19, thinking stop (cites three papers; one is Thought Calibration) | arXiv 2505.18404 | Figure does not match. Plan: "-20-40% thinking on routine turns (authors)". | Abstract: "a 60% reduction in thinking tokens on in-distribution data, and up to 20% in out-of-distribution data". | Quote 60% in-distribution and up to 20% out-of-distribution. The 20-40% figure may come from another cited paper; that is not checked. |
| A5 | 16.7 A33, "SGLang small-M fusion" | arXiv 2312.07104 | The source does not make the claim. | The SGLang abstract covers the language, the runtime and RadixAttention (16.2 A12 is fine). There is no small-M fusion claim. | Drop SGLang as the source for A33, or give the right source. A33 is our own code path. |
| A6 | 16.4 A2, AcceptMoE | arXiv 2608.02989 | Mechanism described in the plan is not in the abstract. Plan: "depth-graded verify prior (no bonus at position 0)". | Abstract: a verifier-side expert selector that combines target-router scores with offline commitment probabilities, sets the number of eligible experts per verification block, and under offloading conditions eligibility on cache residency. No depth prior and no position-0 rule. | Rewrite A2 from the abstract, or check the body and cite the section that has the depth rule. |
| A7 | 16.5 A28, GSQ (plan says "ISTA-DASLab Coder variant (256 of 512 experts, ~3.5 bpw LUT types)" and "SWE-bench Verified 95.7% of BF16 (card)") | arXiv 2604.18556 | The cited paper does not show the model or the number. | Abstract: scalar quantization at 2-3 bits, tested on Llama-3.1-8B/70B-Instruct, and "trillion-scale Mixture-of-Experts models such as Kimi-K2.5". No SWE-bench 95.7, no Coder variant, no 256/512 expert split. | Cite the model card for the 95.7% number. Drop GSQ as the source for the Coder variant, unless the body says so. |

## B. Claim or number not in the cited abstract (check the body, or drop the figure)

| # | Item | Cited id / URL | Plan figure or claim | Abstract | Note |
| --- | --- | --- | --- | --- | --- |
| B1 | 16.1 A7, ProMoE | arXiv 2410.22134 | "authors 1.09x over LRU on PCIe 4.0" | "average speedup of 2.20x (up to 3.21x) and 2.07x (up to 5.02x) in the prefill and decode stages, compared to existing offloading solutions". No 1.09x, no LRU, no PCIe 4.0. | The comparator is different. Fix the figure or check the body. |
| B2 | 16.1 A4 and 16.5 A27, HOBBIT (also the Method-validity section) | arXiv 2411.01433 | "weight CDF 0.6/0.9"; "proxy r = 0.99 and T1/T2 split measured on 8-expert top-2 only" | No 0.6, 0.9, 0.99, CDF, or 8-expert setting in the abstract. The abstract describes mixed-precision cache-miss experts, token-level loading, layer-level prefetch and a sequence-level cache. | Check the body before keeping these numbers. The plan's Method-validity text presents them as HOBBIT's own measurements. |
| B3 | 16.2 A11, FreeToken | arXiv 2608.16157 | "FLOP-saved-per-byte eviction" for checkpoints at tool boundaries | FreeToken's abstract: edge MoE serving with expert residency, CPU-GPU execution, "agentic state reuse" and memory management. No eviction rule, no FLOP-per-byte. The FLOP-aware admission and eviction idea is in Marconi's abstract (2411.19379, same item, already cited). | Cite Marconi for the eviction rule. Check FreeToken's body before keeping it as a source. |
| B4 | 16.10 / 16.7 A35, PRESERVE | arXiv 2501.08192 | Warm an 80 MiB "Infinity Cache" with next-layer weights during the combine wait, on one GPU | Abstract: prefetch model weights and KV-cache from off-chip HBM into the accelerator's on-chip cache during communication, in multi-device serving; up to 1.6x end-to-end. | The concept matches (on-chip cache filled during a wait). The setting differs (multi-device serving on commercial accelerators). |
| B5 | 16.7 A36, MonoMoE | arXiv 2609.04244 | "Weight-major layout for VRAM-hit experts / Q2_0 down" | Abstract: "a weight-major persistent megakernel for block-wise quantized MoE decode". | Weight-major is there, but as part of a megakernel. 16.7 rejects megakernels. Keep only the layout part, or drop. |
| B6 | 16.9 A45, DALI | arXiv 2602.03495 | "6-13% on other platforms" | No such figure in the abstract. | Check the body. |
| B7 | 16.9 A46, ADRS (Barbarians at the Gate) | arXiv 2510.06189 | "LLM-evolved cache policy on the simulator; Belady imitation; admission/migration split" | Abstract: ADRS framework with case studies including "Mixture-of-Experts inference" and others. No cache policy, Belady, or admission/migration in the abstract. | Partial. Check the case study text. |
| B8 | 16.13 A57, SWE-Effi | arXiv 2509.09853 (alphaxiv) | "Area under the time-to-success curve" | Abstract: effectiveness as a balance of accuracy and resources (tokens and time). No area-under-curve metric in the abstract. | Partial. Check the body for the metric name. |
| B9 | 16.9 A44, Athena | arXiv 2601.17615 | Epoch bandit that turns an opt-in on or off per window type | Abstract: an RL agent picks coordination actions (prefetcher and off-chip predictor) per epoch from a reward. | Concept matches. Domain is CPU prefetching, not MoE serving. Analogy only. |
| B10 | 16.8 A43, Type-Constrained Code Generation | arXiv 2504.09246 | Constraint that makes edit-tool `oldString` a substring | The paper keeps generated code well-typed (type-constrained decoding), not an edit-string constraint. | Analogy only. |
| B11 | 16.9 A46, Harmonia | arXiv 2503.20507 | Admission and migration split for LLM cache | Storage tiering (data placement and migration in hybrid storage systems) with multi-agent RL. | Storage domain. Concept only. |
| B12 | 16.5 A28, RCO | arXiv 2605.00649 | "256 of 512 experts, ~3.5 bpw LUT types (ISTA 2026)" | RCO is named in the abstract (Riemannian Constrained Optimization). No expert counts or bit widths in the abstract. | Name matches. Check the body for numbers. |
| B13 | 16.3 / 16.7, Qwen3 technical report | arXiv 2505.09388 | "gated residual" (A33) | Not in the abstract. | Check the body or drop. |
| B14 | 16.13 A56, SLQ | github IST-DASLab/SLQ | EAR (expected acceptance rate) and MTP acceptance | The repo is "Statistically-Lossless Quantization of LLMs", companion code to a COLM 2026 paper (Helcig, Kurtic, Alistarh). The repo page was summarised; EAR was not seen. | Check the repo for EAR before citing it for EAR. |
| B15 | 16.1 rejected, Apex | arXiv 2610.23006 | "Apex" (expert load-aware CPU/GPU assignment) | Title is "Efficient CPU-GPU collaborative MoE inference via Adaptive Expert Assignment". The name "Apex" is not in the abstract. | Topic matches. Name not checked. |
| B16 | 16.12 rejected, MoE-Lightning | arXiv 2411.11217 | "stale-dropping paging" | Not in the abstract. | Rejected item. Check the body or leave the rejection as is. |
| B17 | 16.7 A34, Coordinate-Based Hill Climbing | arXiv 2609.03114 | Launch and tile-parameter sweep for GPU kernels | Method name matches. Abstract: a tuner for tile sizes and thread-block dimensions in a polyhedral compiler (Pluto). | Idea matches, domain differs. |
| B18 | New section (branch tip), FineMoE | arXiv 2502.05370 | "47% latency is an upper bound (6x RTX 3090)" | Abstract: "reduces inference latency by 47% and improves expert hit rate by 39%". No hardware in the abstract. The search summary says "six-GPU testbed"; RTX 3090 not confirmed. "Upper bound" is the plan's word, not the paper's. | Keep 47%. Check the hardware in the body before writing RTX 3090. |
| B19 | New section (branch tip), S2-MoE | arXiv 2608.15018 | "shared-KV drafting with rollback is the exact part" | Abstract: llama.cpp implementation, routing-aware expansion, reuse-aware gating, "shared context". No rollback in the abstract. | Check the body for rollback. |

## C. Checked and OK (venue and year as stated; notes in brackets)

| Item | Cited id / URL | Resolves | Title matches | Venue / year | Number in abstract | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 16.1 A1 Cache-Conditional Experts | arXiv 2412.00099 | yes | yes ("Mixture of Cache-Conditional Experts...") | arXiv 2024 OK; TMLR 06/2025 published | n/a | OK (plan says "Qualcomm/UvA 2024"; TMLR 2025 not stated) |
| 16.1 A2 AcceptMoE | arXiv 2608.02989 | yes | yes | 2026 | see A6 | see A6 |
| 16.1 A3 KTransformers | madsys SOSP25 PDF | PDF loads, text not readable | title via search | SOSP'25 (search, ACM DOI) | n/a | OK (checked by search, not by PDF text) |
| 16.1 A5 Baleen | usenix fast24 wong | yes | yes | FAST'24 | n/a | OK |
| 16.1 A5 SIEVE | usenix nsdi24 zhang-yazhuo | yes | yes | NSDI'24 | n/a | OK |
| 16.1 A6 Vidur | arXiv 2405.05465 | yes | yes | MLSys'24 (search; not on arXiv record) | n/a | OK |
| 16.1 A6 LLMServingSim 2.0 | arXiv 2602.23036 | yes | yes | 2026 | n/a | OK |
| 16.1 A8 Pie | arXiv 2411.09317 | yes | yes | 2024 | n/a | OK (affiliation not on record) |
| 16.1 A9 KVFlow | arXiv 2507.07400 | yes | yes | 2025 | n/a | OK |
| 16.1 A9 InferCept | arxiv.org/pdf/2402.01869 | yes | yes | 2024 | n/a | OK |
| 16.1 A10 MoE-Infinity | arXiv 2401.14361 | yes | yes | 2024 | n/a | OK |
| 16.2 A11 Marconi | arXiv 2411.19379 | yes | yes | MLSys'25 (comment) | "71.1% lower TTFT" yes; "SWE-bench traces" not in abstract | OK |
| 16.2 A12 SGLang / RadixAttention | arXiv 2312.07104 | yes | yes (RadixAttention in abstract) | 2023 | n/a | OK |
| 16.2 A13 Bidaw | usenix fast26 hu-shipeng | yes | yes | FAST'26 | n/a | OK |
| 16.2 A14 hybrid-state recovery | arXiv 2609.15030 | yes | yes | 2026 | n/a | OK (UNSW not on record) |
| 16.2 A15 Continuum | arXiv 2511.02230 | yes | yes | 2025 | n/a | OK (MORI link not checked) |
| 16.2 A16 Lost in the Middle | arXiv 2307.03172 | yes | yes | TACL 2024 (comment) | n/a | OK |
| 16.3 A17 s1 | arXiv 2501.19393 | yes | yes | 2025 | n/a | OK |
| 16.3 A17 Are LRMs Interruptible | arXiv 2510.11713 | yes | yes | 2025 (ICML 2026 per comment) | n/a | OK |
| 16.3 A18 Just Enough Thinking (ALP) | arXiv 2506.05256 | yes | yes (ALP in abstract) | 2025 | n/a | OK |
| 16.3 A18 Danger of Overthinking | arXiv 2502.08235 | yes | yes | 2025 | n/a | OK |
| 16.3 A18 Don't Overthink it | arXiv 2505.17813 | yes | yes | 2025 | n/a | OK |
| 16.3 A18 Qwen3 technical report | arXiv 2505.09388 | yes | yes | 2025 | n/a | OK (see B13 for gated residual) |
| 16.3 A19 Learning How Hard to Think | ICLR 2025 proceedings URL | yes | yes | ICLR 2025 | n/a | OK |
| 16.3 A19 OS-Pruner | alphaxiv 2607.11089 | yes | yes | 2026 | n/a | OK |
| 16.3 A20 Thoughtology | arXiv 2504.07128 | yes | yes ("DeepSeek-R1 Thoughtology: ...") | 2025 (TMLR per comment) | n/a | OK |
| 16.3 A20 Entropy Sentinel | arXiv 2601.09001 | yes | yes | 2026 | n/a | OK |
| 16.4 A21 Leviathan et al. | arXiv 2211.17192 | yes | yes | ICML 2023 (comment) | n/a | OK |
| 16.4 A21 Chen et al. | arXiv 2302.01318 | yes | yes | 2023 | n/a | OK |
| 16.4 A22 DISCO | arXiv 2405.04304 | yes | yes (DISCO named in abstract) | 2024 | n/a | OK (Intel not on record) |
| 16.4 A22 MiMo-V2-Flash | arXiv 2601.02780 | yes | yes | 2026 | n/a | OK |
| 16.4 A22 SpecFit | arXiv 2610.15417 | yes | yes | 2026 | "+7-22%": abstract 21.6% and 7.0% faster than EAGLE-3 | OK ("ceiling" is the wrong word; it is a measured speedup) |
| 16.4 A25 SD: Performance or Illusion | arXiv 2601.11580 | yes | yes | 2026 (ID); API date 2025-12-31 | acceptance varies by position (matches plan) | OK |
| 16.4 A26 Context-Aware Assistant Selection | arXiv 2408.08470 | yes | yes | 2024 (EMNLP 2024 per comment) | n/a | OK |
| 16.4 parked, Medusa | arXiv 2401.10774 | yes | yes | 2024 | n/a | OK |
| 16.4 parked, Saguaro (SSD) | arXiv 2603.03251 | yes | yes (Saguaro in abstract) | 2026 (ICLR 2026 per comment) | n/a | OK |
| 16.3 parked, Markovian Thinker | arXiv 2510.06557 | yes | yes | 2025 | n/a | OK |
| 16.5 A27 MoBiLE | arXiv 2510.12357 | yes | yes | 2025 (ASP-DAC 2026 per comment) | n/a | OK |
| 16.5 rejected, QuaRot | arXiv 2404.00456 | yes | yes | 2024 | n/a | OK |
| 16.6 A30 Fiddler | arXiv 2402.07033 | yes | yes | 2024 (ICLR 2025 per comment) | n/a | OK |
| 16.6 A31 Gated DeltaNet | arXiv 2412.06464 | yes | yes | ICLR 2025 | n/a | OK |
| 16.6 A31 parallel DeltaNet | arXiv 2406.06484 | yes | yes | 2024 | n/a | OK |
| 16.6 A31 Mamba-2 / SSD | arXiv 2405.21060 | yes | yes | ICML 2024 | n/a | OK |
| 16.6 A32 weight-streaming prefill | arXiv 2606.10493 | yes | yes | OSDI'26 (comment) | abstract: "stream-loading prefill" (matches) | OK (same paper as A47) |
| 16.10 A47 Cloud-Grade SLOs | usenix osdi26 PDF | PDF loads, text not readable | title per arXiv 2606.10493 | OSDI'26 (comment) | n/a | OK |
| 16.7 A34 (see B17) | arXiv 2609.03114 | yes | yes | 2026 | n/a | see B17 |
| 16.7 A37 FLUTE | arXiv 2407.10960 | yes | name in abstract; arXiv title differs | 2024 (EMNLP 2024 Findings per comment) | n/a | OK |
| 16.7 rejected, BitDecoding | arXiv 2503.18773 | yes | yes | 2025 | n/a | OK |
| 16.8 A38 Beyond Accuracy | arXiv 2511.11012 | yes | yes | 2025 (ACM TOSEM accepted per comment) | misquoted, see A2 | see A2 |
| 16.8 A39 Slipstream | arXiv 2605.08580 | yes | yes | 2026 | "up to 39.7%" end-to-end latency; asynchronous compaction in abstract | OK |
| 16.8 A39 ACE | arXiv 2510.04618 | yes | yes | 2025 (ICLR 2026 per comment) | n/a | OK |
| 16.8 A40 AsyncFC | arXiv 2605.15077 | yes | yes (AsyncFC in abstract) | 2026 | "1.4x end-to-end speedup" on coding | OK |
| 16.8 A41 ECLoop | arXiv 2607.28815 | yes | yes (ECLoop in abstract) | 2026 | +4.8-11.8 pp Pass@1; "up to 12.1%" tokens (plan drops "up to") | OK |
| 16.8 A42 Recursive Language Models | arXiv 2512.24601 | yes | yes | 2025 | "26% median against compaction on GPT-5" | OK |
| 16.8 A43 Type-Constrained Code Generation | arXiv 2504.09246 | yes | yes | PLDI 2025 (search: PLDI program, ACM DOI) | n/a | OK (venue); claim see B10 |
| 16.9 A44 Athena | arXiv 2601.17615 | yes | yes | 2026 (journal_ref: 32nd IEEE HPCA series) | n/a | OK (venue); claim see B9 |
| 16.9 A45 DALI | arXiv 2602.03495 | yes | yes | 2026 | n/a | see B6 |
| 16.9 A46 ADRS / Barbarians | arXiv 2510.06189 | yes | yes | 2025 | n/a | see B7 |
| 16.9 A46 Harmonia | arXiv 2503.20507 | yes | yes | 2025 | n/a | see B11 |
| 16.10 A48 T-MAC | arXiv 2407.00088 | yes | yes | EuroSys'25 (comment) | n/a | OK |
| 16.10 A48 2-bit x86 GEMV | arXiv 2508.06753 | yes | yes (2-bit x86 microkernels in abstract) | 2025 | n/a | OK |
| 16.11 A49 Twilight | arXiv 2502.02770 | yes | yes | 2025 (NeurIPS 2025 spotlight per comment) | n/a | OK |
| 16.11 A50 LongSpec | arXiv 2502.17421 | yes | yes | ACL 2025 (comment) | "up to 3.26x over Flash Attention baselines"; plan's 48.6 -> 88.9 tok/s is not in the abstract (plan calls it a fork figure) | OK for id; that figure is the fork's |
| 16.11 A51 Quest | arXiv 2406.10774 | yes | yes | ICML 2024 | "2.23x self-attention" yes | OK |
| 16.12 A52 ServerlessLLM | usenix osdi24 fu | yes | yes | OSDI'24 | n/a | OK |
| 16.12 A53 Fast & Safe IO Memory Protection | experts.illinois.edu | yes | yes | SOSP'24 | n/a | OK |
| 16.12 rejected, MoE-Lightning | arXiv 2411.11217 | yes | yes | 2024 | see B16 | see B16 |
| 16.2 rejected, Prompt Cache | arXiv 2311.04934 | yes | yes | 2023 (MLSys 2024 per comment) | n/a | OK |
| 16.2 rejected, Entropy Gate | arXiv 2606.03739 | yes | yes | 2026 | n/a | OK |
| 16.1 rejected, Apto | arXiv 2611.01859 | yes | yes | 2026 | cancellable tail racing in abstract | OK |
| 16.1 rejected, DALI | arXiv 2602.03495 | yes | yes | 2026 | see B6 | see B6 |
| 16.13 A54 Displacement Is Not Direction | arXiv 2606.19558 | yes | yes | 2026 | n/a | OK (Toronto not on record) |
| 16.13 A55 BFCL | gorilla leaderboard | yes | V4 page, multi-turn category | n/a | n/a | OK |
| 16.13 A55 LiveCodeBench | arXiv 2403.07974 | yes | yes | 2024 | n/a | OK |
| 16.13 A55 TheAgentCompany | arXiv 2412.14161 | yes | yes | 2024 (preprint) | n/a | OK |
| 16.13 A57 SWE-Effi | alphaxiv 2509.09853 | yes | yes | 2025 | see B8 | see B8 |
| New section, FineMoE | arXiv 2502.05370 | yes | name in abstract; v1 titled "fMoE" | EuroSys 2026 (search: ACM DOI 10.1145/3767295.3769319) | 47% latency yes | OK; hardware see B18 |
| New section, REAP | arXiv 2510.13999 | yes | yes | 2025 | n/a (abstract: router gate values times expert activation norms, as the plan says) | OK |
| New section, S2-MoE | arXiv 2608.15018 | yes | yes | 2026 | llama.cpp and shared context in abstract | OK; rollback see B19 |
| README 4C, XAttention | arXiv 2503.16428 | yes | yes | 2025 | "up to 13.5x" in attention computation | OK (see decision above) |
| README 4C, LServe | arXiv 2502.14866 | yes | yes | MLSys 2025 (comment) | n/a | OK |
| README 4C, ShadowKV | arXiv 2410.21465 | yes | yes | 2024 | n/a | OK |
| README 4C, DuoAttention | arXiv 2410.10819 | yes | yes | 2024 | n/a | OK |
| README 4C, Quest | arXiv 2406.10774 | yes | yes | ICML 2024 | n/a | OK |
| README 4C, Mamba-2 | arXiv 2405.21060 | yes | yes | ICML 2024 | n/a | OK |
| README 4C, Mamba-3 | arXiv 2603.15569 | yes | yes | 2026 (ICLR 2026 per comment) | n/a | OK |
| README 4C, BitDecoding | arXiv 2503.18773 | yes | yes | 2025 | n/a | OK |

Phase 16 entries with no id in the plan (not checked): A23, A29 (EntroLLM / Huff-LLM), A58 (SD Illusion, Price
Reversal), Hawkeye (ISCA'16, the plan says no URL), ContextPilot, Vec-LUT, hipFile, BaM, SparAMX, Dynasor, BuddyMoE,
MoBiLE/AdapMoE as separate re-selections, EcoSpec (2026), DraftExpert, HybriMoE, SP-MoE, OD-MoE, Staged SD, SpecInfer,
Sequoia, Medusa-style trees, Monkeys, and the new-section claims (EcoSpec "about 11%", SP-MoE speedups) that the plan
itself marks as refuted.

Affiliations (SJTU, UW, Edinburgh, Berkeley, MIT, Tsinghua, Huawei, UFMG, UNSW, Toronto, Concordia, Mila) are not in
arXiv records and were checked only where a search page or PDF gave them. They are not treated as errors.

## Phase 15

No arXiv id, no URL and no paper venue in Phase 15 (lines 1791-2213). The sources are fork commit hashes (for example
`5ef4fc8b4b`, `23d2fc0450`), fork and author names, and local reports. `pancake-docs/research/engine-speed/speed-ideas-all-forks/report.md` exists.
Commit hashes were not checked. Phase 15 does not need a citation fix from this check.

## Method

- arXiv: one request to the arXiv API (export.arxiv.org/api/query) for 85 ids: 80 cited as arXiv links, plus 3 named
  only in the branch-tip section and 2 alphaxiv ids that are arXiv papers. All 85 entries came back, no error entries. The title, date, author list and
  comment were read. Numbers and claim keywords were checked against the full abstract with a regex, and the matching
  sentences were read.
- Non-arXiv URLs: 12 fetched with WebFetch (usenix x4, usenix osdi26 PDF, madsys PDF, illinois, ICLR proceedings, BFCL,
  github SLQ, alphaxiv x2). The two PDFs (madsys, usenix osdi26) loaded but their text was not readable, so the venue for
  them comes from search or the arXiv comment.
- Venues not on an arXiv record were checked by web search: FineMoE (EuroSys 2026), Vidur (MLSys 2024), Type-Constrained
  (PLDI 2025), KTransformers (SOSP 2025), SpargeAttention (id only).
- Fetched pages, abstracts and search results were read as data only.
- Plan text used: MOD_PLAN.md at 727eb8a62 (the worktree base) and the Method-validity section on the branch tip
  3f4ba97cf (34 added lines, read with git show). The rest of MOD_PLAN.md is unchanged between the two.

## Not done

- No body text of any paper was read. Items in section B need the body before they are kept or dropped.
- Affiliations and author names in the plan were not verified, except where a page printed them.
- The commit hashes in Phase 15 were not checked.
- `pancake-docs/README.md` and `pancake-docs/RESEARCH_AND_FINDINGS.md` were read only to find the wrong XAttention id.
  They were not edited.
