# Stanford papers on LLM decoding, speculative decoding, test-time compute and serving (Nov 2022 - Nov 2026), checked against Avalon

Target setup: llama.cpp fork "Avalon". It serves Qwen3.8-Flash-Next (~125B hybrid MoE with gated delta net + attention and an MTP head) on one RX 7900 XT 20 GB (ROCm 10) with 57 GB RAM. The model is ~89 GiB, so experts are offloaded. It runs one slot for one user (an opencode coding agent) and already uses ngram-simple + MTP speculation (~170 t/s on code edits, ~95 t/s on prose), with prompt-cache reuse.

Legend: **[Lead]** means Stanford first or last author, or the work is lab-led. **[Co]** means Stanford co-authors only. Verdicts are my own inferences for Avalon.

Scope: attention kernels and architecture, quantization and MoE offload, and agent-framework papers are excluded because other researchers cover them. A few borderline items are listed briefly with a note.

---

## Q1. Stanford speculative / parallel decoding: what beats MTP+ngram for code editing?

### Takeaway
Stanford has two first-party speculative decoding papers in the window: Staged Speculative Decoding (Spector & Ré, 2023) and Speculative Speculative Decoding / Saguaro (Kumar [Stanford], Dao, May, ICLR 2026). Neither clearly beats MTP+ngram on a single GPU as published. Two ideas can be ported to llama.cpp cheaply, though:
- a draft tree with a cheap second-stage drafter (staged SD)
- pre-computing drafts for the likely verification outcomes while the GPU is still verifying (SSD)

With an ngram drafter running on the CPU, the second idea needs no extra GPU. My estimate is roughly +5-20% on code edits (an inference, not measured).

### Cited Findings
**1. Accelerating LLM Inference with Staged Speculative Decoding** [Lead]
- Authors: Benjamin Spector, Christopher Ré (Stanford / Hazy Research).
- Venue: arXiv 2308.04623, Aug 2023; ICML 2023 ES-FoMo workshop.
- Links: [arXiv](https://arxiv.org/abs/2308.04623), [ICML listing](https://icml.cc/virtual/2023/28273).
- Core idea: built for small-batch, on-device decoding, which is bound by memory bandwidth. (1) The draft is a tree of candidate continuations rather than a single sequence, so each verification batch carries more tokens. The draft model runs only at internal nodes, so leaves are cheap. (2) The draft model is itself sped up by a second speculative stage. — [arXiv](https://arxiv.org/abs/2308.04623)
- Numbers: 3.16x lower single-batch latency on a 762M GPT-2-L target with quality preserved, and about 1.36x faster than standard speculative decoding (deterministic sampling). — [arXiv](https://arxiv.org/abs/2308.04623); [summary](https://liner.com/review/accelerating-llm-inference-with-staged-speculative-decoding)
- Code: none found.
- Verdict for Avalon:
  - The idea is already partly present. llama.cpp's MTP plus ngram is effectively a two-drafter setup, and upstream llama.cpp has draft-tree support only in limited form.
  - The portable piece is a small tree, e.g. top-2 at the first 1-2 draft positions. Verify all branches in one batch using a tree attention mask.
  - Expected gain is small to moderate, roughly 5-15% (inference). In Avalon the cost of verifying extra tokens is dominated by the MoE expert reads, which are partly on the CPU. Every additional token can activate different experts, so wide trees cost more on an expert-offloaded MoE than on a dense model.
  - The gated-delta-net recurrent state also has to be forked for each branch, which needs state checkpointing per branch. That is a real engineering cost in the hybrid architecture.

**2. Speculative Speculative Decoding (SSD; algorithm "Saguaro")** [Co-led; first author at Stanford]
- Authors and affiliations: Tanishq Kumar (Stanford), Tri Dao (Princeton / Together AI), Avner May (Together AI).
- Venue: arXiv 2603.03251, v1 3 Mar 2026; listed as ICLR 2026.
- Links: [arXiv](https://arxiv.org/abs/2603.03251), [HTML](https://arxiv.org/html/2603.03251).
- Code: [github.com/tanishqkumar/ssd](https://github.com/tanishqkumar/ssd).
- Core idea: removes the serial draft→verify dependency.
  - While the target verifies, the draft model predicts likely verification outcomes (how many tokens get accepted, plus the bonus token). It pre-computes the next speculation for each outcome and stores it in a "speculation cache".
  - On a cache hit, drafting latency drops to zero.
  - Fan-out per position follows a capped geometric series. Saguaro sampling down-weights the draft's top tokens by a constant C, which makes the bonus token more likely to land in the cache, at some cost in acceptance rate.
  - — [arXiv HTML](https://arxiv.org/html/2603.03251)
- Numbers: about 30% faster on average than the strongest optimized SD baselines, and up to 5x faster than autoregressive decoding. It stays about 20% ahead of SD at larger batch sizes. The bonus token is predicted with up to 90% accuracy. — [arXiv](https://arxiv.org/abs/2603.03251), [HTML](https://arxiv.org/html/2603.03251)
- Setup: target Llama-3.1-70B on 4xH100 (TP=4). The draft is Llama-3.2-1B on a **separate dedicated H100**. Batch 1, greedy decoding. Datasets: Alpaca, GSM8k, UltraFeedback, HumanEval. Qwen-3 results are in the appendix. — [HTML](https://arxiv.org/html/2603.03251)
- Stated limitations:
  - A dedicated draft device is required.
  - Adding draft devices gives diminishing returns.
  - Combining SSD with EAGLE or token trees is unexplored.
  - The cache-miss fallback returns random tokens, and the authors note that n-gram speculators could do better.
  - — [HTML](https://arxiv.org/html/2603.03251)
- Verdict for Avalon:
  - The paper's setup (a separate draft GPU) does not match ours. MTP runs on the same GPU as the target.
  - **Port that fits us:** run an SSD-lite version with the CPU-side ngram drafter. While the GPU runs a verify batch, idle CPU threads pre-build ngram drafts for the top-k candidate bonus tokens (and for the accept-all and accept-n-1 outcomes). When verification returns, the next batch is ready immediately, cutting the per-step host overhead.
  - Gain is limited to whatever host time sits between GPU steps. On a 7900 XT with experts partly on the CPU, the CPU is also busy with MoE FFNs, so free cores are scarce. Expect **roughly 0-10%** (inference). Measure the inter-step gap with a profiler before investing.
  - The full Saguaro method with MTP as the drafter would need MTP to run concurrently with verification on the same GPU. That is not useful on one device.

**3. Next-Concept Modeling (NCM): Fast, Parallel Coarse-to-Fine Text Generation** [Lead]
- Authors: Hermann Kumbong, Ishan S. Khare et al. (Stanford Hazy / Scaling Intelligence).
- Venue: arXiv 2610.14162, Oct 2026 preprint.
- Links: [arXiv](https://arxiv.org/abs/2610.14162), [blog](https://hazyresearch.stanford.edu/blog/2026-10-12-ncm).
- Listed for completeness. It is a new generation paradigm that needs a model trained for it, so it falls under "architecture" and is **not applicable** to an existing Qwen checkpoint. I did not read the details. — [Scaling Intelligence pubs](https://scalingintelligence.stanford.edu/pubs/)

**4. SPRINT: Interleaved Planning and Parallelized Execution in Reasoning Models** [Lead]
- Authors: Emil Biju, Shayan Talaei et al.
- Venue: NeurIPS 2025; arXiv 2506.05745.
- Links: [arXiv](https://arxiv.org/abs/2506.05745).
- Core idea: train reasoning models to plan and then run independent sub-steps in parallel, cutting sequential reasoning length. — [Scaling Intelligence pubs](https://scalingintelligence.stanford.edu/pubs/)
- Verdict: needs fine-tuning, and it only pays off with batch >1 (parallel branches). With one slot it is **not applicable** without multi-sequence decoding.

**5. Low-latency megakernel for Llama-1B ("Look Ma, No Bubbles!")** [Lead] (a kernel and engine engineering result, not attention, so kept here)
- Authors: Spector*, Juravsky*, Sul, Dugan, Lim, Fu, Arora, Ré (Stanford Hazy).
- Venue: blog post, 27 May 2025.
- Links: [blog](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles).
- Code: [HazyResearch/Megakernels](https://github.com/HazyResearch/Megakernels).
- Core idea: at batch 1, decoding is limited by memory bandwidth, and the ~100 small kernels per forward pass leave stalls ("bubbles") between kernels. Fusing the whole forward pass into one persistent kernel removes them.
- Numbers:
  - Llama-3.2-1B forward pass under 1 ms on H100, about 2.5x faster than vLLM and over 1.5x faster than SGLang. Bandwidth use is 78% versus at most 50% for existing engines.
  - Under 680 µs on B200.
  - Launch overhead is about 2.1 µs per kernel, or 1.3 µs with CUDA graphs.
  - — [blog](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
- Verdict for Avalon:
  - A full megakernel is impractical on ROCm for a 125B hybrid MoE with CPU-offloaded experts.
  - The lesson still transfers: in speculative verify steps of only a few tokens, per-kernel launch and sync overhead plus GPU↔CPU expert hand-offs can dominate.
  - Make sure HIP graphs are used in Avalon for the decode and verify graphs, and count sync points per token.

### Inferences
- I found **no Stanford paper on lookahead / Jacobi decoding or other draft-free methods**. Lookahead decoding comes from LMSYS/UCSD, Medusa from Princeton/UIUC/CMU, EAGLE from PKU/Waterloo, SuffixDecoding from Snowflake/CMU. None are Stanford.
- For code editing in particular, Stanford offers nothing that beats ngram+MTP. The most productive non-Stanford direction remains suffix-tree / longer-match ngram drafting over the whole session history, which is outside this scope.
- The best Stanford-derived options ranked by expected value per hour of work:
  1. SSD-lite: pre-compute ngram drafts for candidate bonus tokens while the GPU verifies (cheap, small gain).
  2. A small draft tree at the first positions (moderate work, because the recurrent-state forking in the gated delta net is the hard part).
  3. HIP-graph and sync audit, inspired by the megakernel post.

### Gaps
- Hardware details and the Qwen-3 appendix numbers of SSD were not read (the HTML was truncated at 100k characters).
- I found no public code for Staged SD.
- I have no measurements of Avalon's inter-step host gap, which is needed to estimate the SSD-lite gain.

---

## Q2. Serving, scheduling and KV-reuse systems with Stanford involvement: what fits a single-GPU, single-user server?

### Takeaway
Stanford people co-created most of the foundational serving work: FlexGen, SGLang/RadixAttention, H2O, Hydragen, Tokasaurus, Cartridges, and the Minions local-cloud protocol. Almost all of it targets throughput (large batches, many users). For one slot and one user, the ideas that transfer are:
- a RadixAttention-style **tree of cached prefixes**: several cached prompt states instead of one, so agent sub-conversations do not evict each other
- FlexGen-style **offload placement tuned by cost model or LP**
- possibly H2O-style KV eviction for very long contexts (quality risk)

Hydragen, Tokasaurus and Cartridges do not fit.

### Cited Findings
**FlexGen: High-Throughput Generative Inference of LLMs with a Single GPU** [Lead: Ying Sheng (Stanford, first author); Liang and Ré co-authors]
- Authors: Sheng, Zheng, Yuan, Li, Ryabinin, Fu, Xie, Chen, Barrett, Gonzalez, Liang, Ré, Stoica, Zhang.
- Venue: arXiv 2303.06865, Mar 2023. The ICML 2023 venue is from my own knowledge; the abstract page lists none.
- Links: [arXiv](https://arxiv.org/abs/2303.06865).
- Code: [FMInference/FlexGen](https://github.com/FMInference/FlexGen).
- Core idea: pool GPU, CPU and disk memory and compute. A linear program chooses where each tensor (weights, KV, activations) lives and how it is accessed. Large effective batches amortize the cost of offloading.
- Numbers: OPT-175B on one 16 GB GPU at 1 token/s with effective batch 144. Weights and KV compressed to 4-bit with negligible loss. — [arXiv](https://arxiv.org/abs/2303.06865)
- Verdict for Avalon:
  - The throughput design does not apply to one slot. Its gains come from batching, and at batch 1 offloading stays bandwidth-bound.
  - The transferable idea is **solving for the placement**: which layers' experts and which KV go to VRAM versus RAM, using a measured cost model instead of hand-tuned `-ot` regexes.
  - Expected gain is only what is left beyond the current hand tuning, probably small. (MoE offload as such is covered by another researcher.)

**SGLang: Efficient Execution of Structured LM Programs (RadixAttention)** [Co-led: Ying Sheng (Stanford, last/co-corresponding author); Christos Kozyrakis and Clark Barrett (Stanford) co-authors; first author Lianmin Zheng (Berkeley)]
- Venue: arXiv 2312.07104, Dec 2023. NeurIPS 2024 is from my own knowledge.
- Links: [arXiv](https://arxiv.org/abs/2312.07104).
- Code: [sgl-project/sglang](https://github.com/sgl-project/sglang).
- Core idea: RadixAttention keeps the KV cache of past requests in a radix tree with LRU eviction, so any shared prefix is reused automatically. A compressed finite-state machine speeds up constrained / JSON decoding.
- Numbers: up to 6.4x higher throughput than the state of the art. — [arXiv](https://arxiv.org/abs/2312.07104)
- Verdict for Avalon: **high applicability as a concept**.
  - opencode issues interleaved requests: the main agent, sub-agents, title and summary calls. Each has a different prefix.
  - With one slot and one cached prompt, every switch forces a long re-prefill.
  - A small radix/LRU store of several prompt states, held in RAM, would turn those into prefix hits. llama.cpp's `--cache-reuse` and slot save/restore are partial equivalents.
  - **Hybrid-model caveat:** the gated-delta-net layers have a recurrent state that cannot be truncated to an arbitrary prefix. You need **state snapshots at chosen boundaries**, e.g. at the end of the system prompt and at each user turn, not per token.
  - Expected gain: very large time-to-first-token savings on agent context switches (seconds to tens of seconds of prefill avoided per switch). There is no decode-speed gain. This is an inference, not a measurement.

**H2O: Heavy-Hitter Oracle for Efficient Generative Inference** [Co: Ying Sheng, Christopher Ré, Clark Barrett (Stanford); lead UT Austin / Beidi Chen]
- Venue: arXiv 2306.14048, Jun 2023. NeurIPS 2023 is from my own knowledge.
- Links: [arXiv](https://arxiv.org/abs/2306.14048).
- Code: [FMInference/H2O](https://github.com/FMInference/H2O).
- Core idea: keep the recent tokens plus the ~20% "heavy-hitter" tokens with the highest accumulated attention, and evict the rest of the KV cache.
- Numbers: up to 29x throughput over DeepSpeed Zero-Inference and HF Accelerate, 3x over FlexGen, and up to 1.9x lower latency (OPT-6.7B/30B). — [arXiv](https://arxiv.org/abs/2306.14048)
- Verdict for Avalon:
  - Only the attention layers carry a KV cache in the hybrid model, so the KV is already small and H2O's benefit is lower.
  - Lossy eviction is risky for exact-recall coding: file contents and line numbers.
  - It is **not recommended** except as a last resort for contexts beyond the VRAM limit.

**Hydragen: High-Throughput LLM Inference with Shared Prefixes** [Lead]
- Authors: Juravsky, Brown, Ehrlich, Fu, Ré, Mirhoseini.
- Venue: arXiv 2402.05099, Feb 2024; ES-FoMo-II @ ICML 2024.
- Links: [arXiv](https://arxiv.org/abs/2402.05099).
- Core idea: split attention into the shared prefix and the unique suffixes, so the prefix KV is read once per batch.
- Numbers: up to 32x end-to-end throughput on CodeLlama-13b. Prefix growth from 1K to 16K costs under 15% throughput versus over 90% for the baselines. — [arXiv](https://arxiv.org/abs/2402.05099)
- Verdict: **not applicable** to one sequence. It helps only when many sequences share a prefix. Tree-verify in speculative decoding is a weak analogue, but the prefix there is a single sequence anyway. It is also an attention kernel, which is out of scope.

**Tokasaurus** [Lead: Stanford Scaling Intelligence lab]
- Venue: blog post and code, Jun 2025.
- Links: [code](https://github.com/ScalingIntelligence/tokasaurus); the lab post is at scalingintelligence.stanford.edu/blogs/tokasaurus/ (I did not open it).
- Claim: up to 3x+ over vLLM/SGLang on throughput benchmarks. Modal reproduced >80k tok/s with Llama-3.2-1B. — [HN](https://nextjs-hn-feed.vercel.app/post/44195961), [Modal](https://frontend.modal.com/docs/examples/tokasaurus_throughput)
- Verdict: **not applicable**. It is a throughput engine (Async-TP, large batches).

**Cartridges: Lightweight long-context representations via self-study** [Lead]
- Authors: Eyuboglu, Ehrlich, Arora et al.
- Venue: arXiv 2506.06266, Jun 2025; ES-FoMo III oral.
- Links: [blog](https://hazyresearch.stanford.edu/blog/2025-06-08-cartridges), [pubs](https://scalingintelligence.stanford.edu/pubs/).
- Core idea: train a small KV cache (the "cartridge") offline for a fixed corpus, such as a code repository, and load it in place of the full context.
- Numbers: matches ICL quality with 38.6x less memory and 26.4x higher throughput; extends effective context from 128k to 484k tokens on MTOB. Authors include J. Zou, A. Mirhoseini, C. Ré. — [arXiv](https://arxiv.org/abs/2506.06266)
- Verdict: interesting for a fixed repository, but it needs offline training per corpus and per model, and the hybrid recurrent layers complicate it. **Not practical** for a live coding agent whose files change every turn.

**Auditing Prompt Caching in Language Model APIs** [Lead]
- Authors: Chenchen Gu, Xiang Lisa Li, Rohith Kuditipudi, Percy Liang, Tatsunori Hashimoto.
- Venue: ICML 2025; arXiv 2502.07776.
- Links: [arXiv](https://arxiv.org/abs/2502.07776).
- Findings: the timing difference between cache hits and misses is a side channel. The authors detected global cache sharing across users at 7 API providers, including OpenAI, and inferred that OpenAI's embedding model is a decoder-only Transformer. — [arXiv](https://arxiv.org/abs/2502.07776)
- Verdict: no speed gain. It is a reminder that if Avalon is ever exposed to more than one user, prompt-cache state should be partitioned per user.

**Minions: Cost-efficient collaboration between on-device and cloud LMs** [Lead: Narayan, Biderman, Eyuboglu, ... Ré (Hazy)]
- Venue: arXiv 2502.15964, Feb 2025. ICML 2025 is from my own knowledge.
- Links: [arXiv](https://arxiv.org/abs/2502.15964), [retrospective](https://hazyresearch.stanford.edu/blog/2026-05-15-minions-to-openjarvis-retrospective).
- Code: [HazyResearch/minions](https://github.com/HazyResearch/minions).
- Core idea:
  - Minion: the local model chats with a cloud model that never reads the long context.
  - MinionS: the cloud model splits the task into sub-tasks, local models run them in parallel over chunks of the context, and the cloud model aggregates.
- Numbers on long-document QA (FinanceBench, ConvFinQA, MuSiQue, QASPER, ...):
  - Minion keeps 87.9% of GPT-4o accuracy at 30.4x lower cost.
  - MinionS keeps 97.9% at 5.7x lower cost.
  - A follow-up adds H100 confidential-computing "Secure Minions".
  - — [retrospective](https://hazyresearch.stanford.edu/blog/2026-05-15-minions-to-openjarvis-retrospective)
- Verdict: a protocol layer on top of the server, with no server changes needed. Avalon could play the "local" role behind an OpenAI-compatible API. MinionS's parallel sub-tasks need several slots, so with one slot they serialize. This is an agent-workflow option, not a decoding speedup.

**Archon: architecture search over inference-time techniques** [Lead: Saad-Falcon et al.]
- Venue: arXiv 2409.15254, 2024.
- Links: [pubs](https://scalingintelligence.stanford.edu/pubs/).
- Core idea: search over ensembles of inference techniques (generate, rank, fuse, critique) across several models.
- Verdict: it multiplies LLM calls, which is the opposite of what a one-slot, latency-bound server wants. **Not applicable.**

### Inferences
- For Avalon, the biggest serving-side win from this literature is **multi-entry prefix caching with recurrent-state snapshots** (the RadixAttention idea adapted to a hybrid model). It attacks prefill latency, which dominates in agent workloads with 30-100k-token prompts, rather than decode speed.
- The RAM budget (57 GB with most of it holding experts) limits how many snapshots fit. A disk tier on the D drive could hold older snapshots. KV for attention layers plus the delta-net state per snapshot is likely hundreds of MB to a few GB, depending on context length; I did not compute this.

### Gaps
- The ICML / NeurIPS venues for FlexGen, SGLang, H2O and Minions are from my own knowledge, not from the fetched pages.
- I did not open the Tokasaurus blog, so its numbers are secondary.
- I found no Stanford paper on KV offload or prompt caching designed specifically for hybrid SSM/attention models.

---

## Q3. Test-time compute papers: relevance to reasoning-effort handling and latency for an agent

### Takeaway
Stanford leads this area: Large Language Monkeys, CodeMonkeys, Weaver, the power-laws follow-up, s1 budget forcing, and Think-Prune-Train. For a one-slot, latency-bound coding agent:
- **s1 budget forcing is directly implementable** in a llama.cpp server. Cap the thinking tokens by forcing `</think>`, or extend them by suppressing end-of-think and appending "Wait". That gives a principled reasoning-effort control.
- Repeated sampling only pays off with a verifier (tests) and parallel slots, so it is mostly irrelevant at one slot. One exception: cheap best-of-2 on hard edits that come with a test command.

### Cited Findings
**s1: Simple test-time scaling** [Lead]
- Authors: Muennighoff, Yang, Shi, X. L. Li, Fei-Fei Li, Hajishirzi, Zettlemoyer, Liang, Candès, Hashimoto. Stanford-led, with UW / AI2 co-authors.
- Venue: arXiv 2501.19393, Jan 2025.
- Links: [arXiv](https://arxiv.org/abs/2501.19393).
- Code: [simplescaling/s1](https://github.com/simplescaling/s1).
- Core idea: "budget forcing".
  - To stop early, force the end-of-thinking delimiter.
  - To extend, suppress the delimiter and append "Wait", which often makes the model re-check its work.
  - The model is fine-tuned on only 1,000 curated reasoning traces (s1K) on top of Qwen2.5-32B-Instruct.
- Numbers: s1-32B beats o1-preview by up to 27% on MATH/AIME24. With budget forcing, AIME24 rises from 50% to 57%. — [arXiv](https://arxiv.org/abs/2501.19393)
- Verdict for Avalon: **directly applicable, low effort**.
  - Implement a per-request `thinking_budget` (max reasoning tokens): when it is reached, inject the `</think>` token(s) and continue in answer mode.
  - Optionally add a `min_thinking` setting that suppresses `</think>` until N tokens and appends "Wait".
  - Map opencode's reasoning-effort levels (low/medium/high) to budgets.
  - Latency gain: hard caps on runaway thinking. At about 95 t/s for prose, every 1,000 avoided thinking tokens saves about 10.5 s.
  - Quality cost: cutting thinking short lowers accuracy. s1 shows monotone scaling with budget, so pick the budgets empirically on your own tasks.
  - Upstream llama.cpp has a `--reasoning-budget` flag; check what Avalon's fork supports. Only 0 / unlimited is the classic behavior, so a true token budget may need adding.

**Large Language Monkeys: Scaling Inference Compute with Repeated Sampling** [Lead]
- Authors: B. Brown, J. Juravsky, R. Ehrlich, R. Clark, Q. V. Le, C. Ré, A. Mirhoseini. Stanford with Oxford and Google DeepMind co-authors.
- Venue: arXiv 2407.21787, Jul 2024.
- Links: [arXiv](https://arxiv.org/abs/2407.21787).
- Numbers and findings:
  - Coverage (solved by at least one sample) grows log-linearly over 4 orders of magnitude of sample count.
  - On SWE-bench Lite, DeepSeek-Coder-V2 goes from 15.9% with 1 sample to 56% with 250 samples, against a single-sample state of the art of 43%.
  - Without an automatic verifier, majority vote and reward models plateau after several hundred samples.
  - — [arXiv](https://arxiv.org/abs/2407.21787)
- Verdict: one slot at about 100 t/s makes 250 samples absurd. The **principle is still useful**: when a cheap verifier exists (unit tests, compilation), 2-4 sequential attempts beat one long attempt.
  - That belongs in the agent policy, not the server.
  - A server-side enabler is to keep the shared prefix cached so each retry costs only decode time.

**CodeMonkeys: Scaling Test-Time Compute for Software Engineering** [Lead]
- Authors: Ehrlich, Brown, Juravsky et al.
- Venue: arXiv 2501.14723, Jan 2025.
- Links: [arXiv](https://arxiv.org/abs/2501.14723), [pubs](https://scalingintelligence.stanford.edu/pubs/).
- Core idea: repeated sampling plus model-generated tests and selection for SWE-bench, sharing a cached codebase context across samples.
- Authors: Ehrlich, Brown, Juravsky, Clark, Ré, Mirhoseini. Numbers: 57.4% on SWE-bench Verified for about $2,300 total; selecting among edits from existing top submissions reaches 66.2%, beating the best single member. Code and data: [project page](https://scalingintelligence.stanford.edu/pubs/codemonkeys). — [arXiv](https://arxiv.org/abs/2501.14723)
- Verdict: same as Monkeys. Its relevance is the **amortization of a shared context across samples**, which again argues for strong prefix caching.

**How Do Large Language Monkeys Get Their Power (Laws)?** [Co/Lead: Schaeffer, Kazdan et al., incl. Stanford]
- Venue: ICML 2025 oral; arXiv 2502.17578.
- Links: [arXiv](https://arxiv.org/abs/2502.17578).
- Content: theory of why coverage follows power laws. No serving implication.

**Weaver: Shrinking the Generation-Verification Gap with Weak Verifiers** [Lead: Saad-Falcon, Buchanan et al.]
- Venue: NeurIPS 2025; arXiv 2506.18203.
- Links: [arXiv](https://arxiv.org/abs/2506.18203).
- Core idea: combine several weak verifiers (reward models, LM judges) to select among repeated samples.
- Verdict: needs many samples and many verifier calls. **Not applicable** at one slot.

**Think, Prune, Train, Improve: Scaling Reasoning Without Scaling Models** [Lead]
- Venue: arXiv 2504.18116, 2025.
- Links: [arXiv](https://arxiv.org/abs/2504.18116).
- Content: a training-time method. Not applicable to serving.

### Inferences
- Priority for Avalon: (1) a token-level thinking budget with forced `</think>` (s1-style), exposed per request. (2) Prefix caching that makes retries cheap. Everything else in this cluster targets throughput or offline evaluation.
- The "Wait" extension is probably counter-productive for a latency-focused coding agent. Use the cap side of budget forcing, not the extend side.

### Gaps
- I found no Stanford paper on **adaptive** reasoning length (models deciding their own budget) that runs without fine-tuning. The known adaptive-length papers I am aware of (L1 from CMU, others) are not Stanford.
- I did not verify the Weaver numbers this session.

---

## Q4. Stanford work on local / on-device LLM efficiency

### Takeaway
Stanford Hazy Research / Scaling Intelligence (Ré, Mirhoseini) run an explicit "local AI" program: Minions → Intelligence per Watt → hybrid inference / "Gross Domestic Intelligence" → OpenJarvis. The results are measurement and protocol papers, not kernel speedups. Their own position is that local hardware favors **dense, small, well-quantized models over sparse MoE** because of memory bandwidth. Avalon's 125B MoE with offloaded experts sits at the opposite end of that recommendation, though this is the authors' argument rather than a tested claim.

### Cited Findings
**Intelligence per Watt: Measuring Intelligence Efficiency of Local AI** [Lead]
- Authors: Saad-Falcon, Narayan, et al.; Ré and Mirhoseini; Together AI co-authors.
- Venue: arXiv 2511.07885, Nov 2025.
- Links: [arXiv](https://arxiv.org/abs/2511.07885), [project page](https://scalingintelligence.stanford.edu/pubs/ipw), [Hazy blog](https://hazyresearch.stanford.edu/blog/2025-11-18-intelligence-per-watt).
- Code: [HazyResearch/intelligence-per-watt](https://github.com/HazyResearch/intelligence-per-watt).
- Metric: intelligence per watt (IPW) = task accuracy / power draw. The study covers over 1M queries, 20+ local models (≤20B active parameters) and 8 accelerators from 2023 to 2025.
- Numbers:
  - Local models answer 88.7% of single-turn chat and reasoning queries correctly.
  - IPW improved 5.3x in two years: about 3.1-3.2x from models and 1.7x from hardware.
  - The best single local model's coverage rose from 23.2% to 71.3%.
  - Local accelerators are 1.4-7.4x less efficient per query than cloud ones.
  - A local-cloud router cuts energy, compute and cost by 60-80%.
  - — [retrospective](https://hazyresearch.stanford.edu/blog/2026-05-15-minions-to-openjarvis-retrospective), [The Batch](https://www.deeplearning.ai/the-batch/stanford-and-together-ai-researchers-chart-edge-models-performance-in-intelligence-per-watt)
- Minor discrepancy: models' share of the gain is given as 3.2x in one source and 3.1x in another. — [search summary of arXiv/IPW](https://scalingintelligence.stanford.edu/pubs/ipw) vs [retrospective](https://hazyresearch.stanford.edu/blog/2026-05-15-minions-to-openjarvis-retrospective)
- Verdict: the profiling harness could be pointed at Avalon to measure joules per token and accuracy per watt for configuration choices (MTP on/off, draft length, expert placement). That is useful for evaluation, but there is no speed gain.

**OpenJarvis: Personal AI, On Personal Devices** [Lead]
- Authors: Saad-Falcon, Narayan et al.
- Venue: arXiv 2605.17172, May 2026.
- Links: [arXiv](https://arxiv.org/abs/2605.17172).
- Code: [open-jarvis/OpenJarvis](https://github.com/open-jarvis/OpenJarvis).
- Content: a local-first stack with 4 local engines (Ollama, vLLM, SGLang, llama.cpp) and 5 cloud engines. A frontier model reads on-device traces and proposes configuration edits, which are accepted only if a held-out gate shows no regression. — [retrospective](https://hazyresearch.stanford.edu/blog/2026-05-15-minions-to-openjarvis-retrospective)
- Verdict: it is agent-framework territory, which another researcher covers. The "LLM-guided spec search with a regression gate" idea could be borrowed to auto-tune Avalon launch flags.

**Related lab posts:**
- "Maximizing American Gross Domestic Intelligence with Hybrid Inference", Nov 28, 2025 — [Hazy blog index](https://hazyresearch.stanford.edu/blog)
- "Mind the Trust Gap: Fast, Private Local-to-Cloud LLM Chat", May 12, 2025 — [blog](https://hazyresearch.stanford.edu/blog/2025-05-12-security)
- "Smoothie: label-free inference-time LLM routing", Dec 10, 2024 — [blog](https://hazyresearch.stanford.edu/blog/2024-12-10-smoothie)
- All three are about routing and hybrid inference, not single-server speed.

The retrospective states that local memory bandwidth (~500 GB/s on Apple Silicon) is far below datacenter HBM3e, and argues that dense, small, well-quantized models suit local serving better than sparse MoE. — [retrospective](https://hazyresearch.stanford.edu/blog/2026-05-15-minions-to-openjarvis-retrospective)

### Inferences
- None of the Stanford local-AI work offers a drop-in decoding speedup for llama.cpp. Its value for Avalon is (a) an energy and accuracy measurement methodology (IPW harness) and (b) a hybrid policy: route the hardest agent steps to a cloud model and keep routine edits local.

### Gaps
- IPW did not evaluate AMD RDNA3 consumer GPUs as far as the sources show, and did not evaluate a >100B MoE with CPU-offloaded experts. Its conclusions do not directly cover Avalon's configuration.

---

### Borderline / out-of-scope items noted, not analyzed
- Contrastive Decoding (Xiang Lisa Li et al., Stanford-led, arXiv 2210.15097, Oct 2022, ACL 2023): a quality-oriented decoding method using an amateur model. It is slightly before the window and costs a second model per token, so it is not a latency win. I did not fetch it this session.
- CATS: contextual sparsity thresholding (COLM 2024) and Deja Vu (Stanford co-authors): activation-sparsity / architecture work, out of scope. — [pubs](https://scalingintelligence.stanford.edu/pubs/)
- Discrete diffusion LMs (Ermon group, Stanford; Inception Labs Mercury): a different model family, not applicable to Qwen checkpoints. Not verified this session.
