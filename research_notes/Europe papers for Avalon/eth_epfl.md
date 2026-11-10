# ETH Zurich and EPFL AI/systems papers (Nov 2022 - Nov 2026) for pancake / Avalon

Scope note. Target: pancake (private Strata fork, C++/HIP + Python) and Avalon (llama.cpp fork). Both serve Qwen3.8-Flash-Next (~125B MoE, GDN + attention, MTP) as 2-3-bit GGUF on an RX 7900 XT 20 GB, 5700G, DDR4, 57 GB RAM, NVMe. Experts are tiered across VRAM, pinned RAM and NVMe, and the CPU computes missed experts.

Method and limits. This session ran about 18 searches and fetches. Prior reports cover almost no ETH/EPFL-led work: SpQR, FlexGen and TLT appear only as UW-, Stanford- and MIT-led. I verified entries tagged [verified] against an arXiv, proceedings or search-result page in this session. Entries tagged [not re-fetched] come from my prior knowledge with their known arXiv IDs. Treat their numbers as approximate and check the abstract before relying on them.

Main negative finding. Targeted searches found no ETH or EPFL paper on single-node MoE expert offloading, expert caching or expert prefetching. I searched for ETH plus MoE offloading, SPCL 2025 inference, SAFARI LLM inference, and Jaggi/Cevher plus MoE. The MoE offloading literature is led by Edinburgh, Yandex, KAIST, SJTU, NTU and others. ETH/EPFL contribute three things instead: (a) learned prefetching and placement from SAFARI, transferable as ideas; (b) rotation, quantization and slicing (SPCL with ISTA); (c) agent, constrained-decoding and eval work (SRI Lab, SPY Lab, SPCL).

## Q1. Which ETH/EPFL ideas could cut expert-read bytes or improve expert caching and prefetching?

### Takeaway
No ETH/EPFL paper directly targets MoE expert caching. The most transferable idea is SAFARI's line of small online-learned controllers: Pythia, Hermes, Athena, Sibyl and Harmonia. Each is a tiny RL agent or perceptron that learns, within one run, when to prefetch, whether a load will miss, and which tier to place data in.

Mapped to pancake, these give three cheap experiments:
- a per-layer miss predictor (Hermes-style) that starts the CPU expert path early;
- an RL coordinator (Athena-style) that adapts prefetch aggressiveness to RAM/PCIe bandwidth headroom;
- a placement-plus-migration policy split (Harmonia-style) for the VRAM / pinned RAM / NVMe tiers.

Realistic upside is closing part of the 79.6% (tier) to 89.5% (LRU) hit-rate gap, not passing Belady's 94.7%. The previously tested ClassLRU scoring and reuse-aware admission were heuristic, not online-learned. So this differs from them, but it may still give no gain if the access stream is already near LRU-optimal.

### Cited Findings

**SAFARI (Onur Mutlu, ETH Systems Group). Learned prefetch, placement and miss prediction**

- **Athena: Synergizing Data Prefetching and Off-Chip Prediction via Online Reinforcement Learning.** SAFARI LEAD. Authors: Rahul Bera (first) and six co-authors, including Onur Mutlu. HPCA 2026, arXiv 2601.17615 (Jan 2026). [verified]
  - Idea: the coordination between a prefetcher and an off-chip-load predictor is cast as RL. At the end of each epoch, the agent reads prefetch accuracy, predictor accuracy and bandwidth use. It then switches the prefetcher or predictor on or off, or changes prefetcher aggressiveness. Reward is the change in system metrics such as cycles per epoch.
  - Finding: naively combining prefetching and miss prediction often underperforms, especially when bandwidth is constrained. Athena beats prior coordination policies across configurations with modest storage overhead.
  - Code: github.com/CMU-SAFARI/Athena.
  - Sources: [arXiv 2601.17615](https://arxiv.org/abs/2601.17615v1); [search summary](https://www.emergentmind.com/papers/2601.17615). Bera's 2026 dissertation, which combines Pythia, Hermes and Athena: [arXiv 2603.07683](https://arxiv.org/abs/2603.07683).
- **Harmonia: A Multi-Agent RL Approach to Data Placement and Migration in Hybrid Storage Systems.** SAFARI LEAD: Rakesh Nadig et al. (ETH, AMD Research, POSTECH), with Mutlu. arXiv 2503.20507 (2025, preprint). [verified via search result]
  - Idea: two lightweight agents, one for placement (extends Sibyl) and one for migration. A single agent doing both performs worse than a placement-only agent.
  - Numbers: +49.5% over the best prior method on performance-optimized two-device systems, and +31.7% on cost-optimized ones. With three or four devices, +37.0% and +42.0%. Inference costs about 240 ns and about 206 KiB of DRAM. The v2 and v3 PDFs differ on some two-device figures.
  - Code: the authors say it will be open-sourced.
  - Source: [arXiv 2503.20507](https://arxiv.org/abs/2503.20507)
- **Sibyl** (RL data placement in hybrid SSD/HDD storage; Singh et al., ISCA 2022, arXiv 2205.07394). **Pythia** (RL hardware prefetcher; Bera et al., MICRO 2021, arXiv 2109.12021). **Hermes** (perceptron off-chip-load predictor; Bera et al., MICRO 2022, arXiv 2209.00188). All three are SAFARI-led, predate or sit on the edge of the window, and are the base of Athena and Harmonia. [not re-fetched] [Pythia](https://arxiv.org/abs/2109.12021), [Hermes](https://arxiv.org/abs/2209.00188), [Sibyl](https://arxiv.org/abs/2205.07394)
- **PIM is All You Need: A CXL-Enabled GPU-Free System for LLM Inference.** Gu, Khadem, Umesh, Liang, Servot, Mutlu, Iyer, Das. ASPLOS 2025. Seen only as a citation, so lead lab is unconfirmed (likely U. Michigan; Mutlu co-author). Requires PIM/CXL hardware and does not apply here. — [citation seen in arXiv 2506.21901](https://arxiv.org/html/2506.21901v1)
- Searches for a SAFARI paper on MoE SSD offloading found none. The closest SSD-MoE work is non-ETH:
  - KAIST "FlashMoE": ML-based cache replacement mixing recency and frequency, up to 51% higher hit rate than LRU/LFU and up to 2.6x over Fiddler/DAOP, no code. [arXiv 2601.17063](https://arxiv.org/abs/2601.17063)
  - SNU energy study: SSD-resident experts cost about 4.9x the energy per token of HBM and about 3.1x of CPU-RAM offload. [arXiv 2508.06978](https://arxiv.org/pdf/2508.06978)

**EPFL**

- **Accelerating MoE Model Inference with Expert Sharding (MoEShard).** EPFL LEAD (Kermarrec's SaCS lab): Balmau, Kermarrec, Pires, Loureiro Espírito Santo, de Vos, Vujasinovic. EuroMLSys '25, arXiv 2503.08467.
  - Idea: tensor-shard experts across GPUs for perfect load balance, and fuse decomposed expert computations to cut kernel launches.
  - Multi-GPU only; does not apply to one GPU. The kernel-fusion point is already handled by grouped MoE kernels.
  - Source: [arXiv 2503.08467](https://web3.arxiv.org/pdf/2503.08467)

**ETH Klimovic (EASL). Delta compression**

- **DeltaZip: Efficient Serving of Multiple Full-Model-Tuned LLMs.** ETH LEAD: Xiaozhe Yao, Qinghao Hu, Ana Klimovic. EuroSys '25, arXiv 2312.05215.
  - Idea: store fine-tuned models as compressed deltas from a shared base, up to 10x delta compression. Throughput is 2-12x over state-of-the-art serving (1.5-3x over HF in v1).
  - Code: deltazip.readthedocs.io.
  - Source: [arXiv 2312.05215](https://arxiv.org/abs/2312.05215v3)

### Inferences

- **SAFARI to pancake, experiment A: Hermes-style expert-miss predictor.**
  - Map a cache line to an expert slice, an off-chip load to a VRAM miss, and the prefetcher to the expert prefetcher.
  - A hashed perceptron over (layer, previous-layer top-k ids, router logit margin, token position class) predicts "this expert will miss VRAM". The CPU expert job and the RAM/NVMe read can then be issued one layer earlier.
  - This is a latency-hiding change, not a hit-rate change. It complements the tested DraftExpert windows because it is per-token and online.
  - Expected gain (my estimate): 0-5% decode, and only if CPU expert jobs currently start late. Profile first.
- **SAFARI to pancake, experiment B: Athena-style bandwidth-aware throttle.**
  - Use a tiny epoch-based bandit (per 64 tokens) to choose prefetch depth and whether to speculatively fetch MTP-draft experts, rewarded by tokens per second.
  - Pancake is RAM-bandwidth bound on CPU experts. Speculative prefetch competes for the same DDR4 bandwidth, which is exactly the regime where Athena reports that naive combinations lose.
  - Expected gain (my estimate): avoids regressions more than it adds speed, about 0-3%.
- **SAFARI to pancake, experiment C: Harmonia split.**
  - Separate the "where to admit on first touch" policy from the "promote/demote between VRAM and pinned RAM" policy.
  - Pancake's tier policy is 10 points below LRU on replay, which suggests the admission/migration coupling is part of the loss.
  - Cheapest version: replay-trace offline, with a two-agent tabular Q-learner against the existing trace.
  - Expected gain (my estimate): if the tier policy reaches LRU's 89.5%, misses fall from 20.4% to 10.5%. With CPU experts at about half of decode, that could be worth roughly 15-25% decode. That is the upper bound for any caching work, not a prediction for Harmonia specifically.
- **DeltaZip:** no direct use with a single model. Its delta-compression kernels show that sparse+quantized deltas decompress fast on GPU, which is only relevant if two GGUF variants of the same model were ever served together.

### Gaps
- No ETH/EPFL paper found on MoE expert offloading, caching or prefetching for consumer GPUs. SPCL and Cevher/Jaggi publication pages were not exhaustively crawled; a check of spcl.inf.ethz.ch/Publications and the LIONS page is needed to rule this out.
- Athena's headline speedup numbers were not captured; only the qualitative claims were.
- None of the SAFARI mechanisms has been evaluated on MoE expert streams. Transfer is inference, not evidence.

## Q2. Quantization and sparsity work usable with GGUF on CPU+GPU

### Takeaway
ETH SPCL (Ashkboos/Hoefler, with ISTA's Alistarh and EPFL's Jaggi/Mohtashami) produced the main rotation line: QuaRot, SliceGPT and QUIK. QuaRot's offline Hadamard rotation is the transferable piece. It is computational invariance, so rotated weights can be re-quantized to GGUF IQ2/IQ3 with fewer outliers, at no runtime cost when the rotation folds into adjacent weights.

For a hybrid GDN model, folding needs care: norms, the GDN gates and shared vs routed experts. The gain would be quality at a fixed bit-width, or the same quality at fewer bits, which means fewer expert bytes per token. Separately, ETH SRI's "Mind the Gap" shows GGUF quantization can be weaponized, which matters because pancake runs community-quantized GGUFs.

### Cited Findings

- **QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs.** ETH SPCL LEAD (Saleh Ashkboos first). Co-authors: Mohtashami (EPFL MLO), Croci, Li, Cameron, Jaggi (EPFL), Alistarh (ISTA), Hoefler (ETH), Hensman (Microsoft). arXiv 2404.00456 (Mar 2024, v2 Oct 2024); NeurIPS 2024 (venue from my knowledge; the arXiv page lists none). [verified]
  - Idea: rotate the residual stream, FFN activations, parts of attention and the KV cache. Outputs are unchanged, outliers disappear, and every matmul runs at 4 bits.
  - Numbers: 4-bit LLaMa2-70B loses at most 0.47 WikiText-2 perplexity and keeps 99% of zero-shot accuracy. 6-bit and 8-bit with round-to-nearest are lossless without calibration.
  - Code: github.com/spcl/QuaRot.
  - Source: [arXiv 2404.00456](https://arxiv.org/abs/2404.00456)
- **SliceGPT: Compress LLMs by Deleting Rows and Columns.** ETH LEAD (Ashkboos, Croci, Nascimento, Hoefler, Hensman). ICLR 2024, arXiv 2401.15024. Uses PCA after an orthogonal rotation to delete whole embedding dimensions. Up to about 25% of parameters removed on Llama-2 70B / OPT 66B / Phi-2 while keeping about 90-99% of zero-shot accuracy (approximate). [not re-fetched] — [arXiv 2401.15024](https://arxiv.org/abs/2401.15024)
- **QUIK: Towards End-to-End 4-Bit Inference on Generative LLMs.** ETH/ISTA (Ashkboos et al.). arXiv 2310.09259, EMNLP 2024. Most weights and activations at 4 bits with outlier channels in higher precision; up to about 3.4x end-to-end speedup on GPU (approximate). [not re-fetched] — [arXiv 2310.09259](https://arxiv.org/abs/2310.09259)
- **GPTQ** (Frantar, Ashkboos, Hoefler, Alistarh; ICLR 2023, arXiv 2210.17323, Oct 2022, on the window edge). ISTA LEAD, ETH co-author. GGUF's imatrix quantization is a simpler cousin of this, so it is not new for pancake. **MARLIN** (Frantar, Castro, Chen, Hoefler, Alistarh; PPoPP 2025, arXiv 2408.11743). ISTA LEAD, ETH co-author. A mixed-precision INT4xFP16 GPU kernel close to ideal 4x at batch 16-32; CUDA/Ampere-specific and does not apply to gfx1100 batch-1. [not re-fetched] — [GPTQ](https://arxiv.org/abs/2210.17323), [MARLIN](https://arxiv.org/abs/2408.11743)
- **SpQR** (UW-led; ETH/ISTA co-authors) is already covered in research_notes/US university papers for Avalon/cmu_uw.md. Not repeated.
- **Mind the Gap: A Practical Attack on GGUF Quantization.** ETH SRI LEAD: Egashira, Staab, Vero, Jingxuan He (now UC Berkeley), Vechev. ICML 2025 (PMLR 267), arXiv 2505.23786. [verified]
  - Idea: the quantization-error interval leaves room to train a model that is benign in full precision but malicious once GGUF-quantized.
  - Numbers: attack-success gaps of 88.7% for insecure code generation, 85.0% for content injection and 30.1% for refusal, across 3 LLMs and 9 GGUF types.
  - Code: github.com/eth-sri/llm-quantization-attack.
  - Sources: [PMLR](https://proceedings.mlr.press/v267/egashira25a.html), [arXiv 2505.23786](https://arxiv.org/abs/2505.23786)
- **EPFL MLO (Jaggi), sparse attention:**
  - **Faster Causal Attention Over Large Sequences Through Sparse Flash Attention** (Pagliardini, Paliotta, Jaggi, Fleuret; NeurIPS 2023, arXiv 2306.01160). Extends FlashAttention to dynamic key/query dropping and hash-based sparsity. Reported 2.0x/3.3x faster training on 8k/16k sequences (approximate). [not re-fetched]
  - **Landmark Attention** (Mohtashami & Jaggi; NeurIPS 2023, arXiv 2305.16300). Block-level landmark tokens let attention retrieve relevant KV blocks, enabling random-access long context. Needs training. [not re-fetched]
  - Sources: [arXiv 2306.01160](https://arxiv.org/abs/2306.01160), [arXiv 2305.16300](https://arxiv.org/abs/2305.16300)
- **ISTA work with no ETH/EPFL author (out of strict scope; listed only because it targets expert bytes directly):**
  - QMoE (Frantar & Alistarh, MLSys 2024, arXiv 2310.16795) compresses SwitchTransformer-c2048 experts to under 1 bit per parameter with custom decode kernels. [not re-fetched] — [arXiv 2310.16795](https://arxiv.org/abs/2310.16795)

### Inferences

- **QuaRot-style rotation for GGUF experts (Avalon first, then pancake).**
  - Fold a random Hadamard into the residual stream and the expert up/gate input side offline, then re-quantize to IQ2_XXS/IQ3. Outlier-free weights quantize better at 2-3 bits.
  - QuIP# (Cornell, already in prior notes) showed the same effect with incoherence processing.
  - For Qwen3.8-Flash-Next the blockers are:
    - RMSNorm weights must be fused first;
    - the GDN and attention input projections share the residual, so both must absorb the rotation;
    - the MTP head must be rotated consistently;
    - the down_proj output rotation needs an online Hadamard unless it folds into the next layer's inputs, which it does in QuaRot's residual-stream scheme.
  - Runtime cost: zero for residual rotations, which fold into weights. Online Hadamards (QuaRot's FFN/attention-internal ones) would add a small kernel and can be skipped for weight-only quantization.
  - Expected gain (my estimate): roughly IQ3-level quality at IQ2-level bytes for routed experts. If expert bytes drop about 15-25%, CPU-bound miss time drops about proportionally: about 7-12% decode, plus more experts fit in VRAM.
  - This is the most concrete byte-cutting lever from ETH. It needs the original BF16 checkpoint and an imatrix re-quant pass, which is heavy on this disk (about 250 GB of BF16; would need the D drive).
- **SliceGPT:** deleting about 10-25% of hidden dimensions would shrink every expert, but quality loss at 2-3 bits on top is unknown and likely compounding. Lower priority than rotation.
- **MARLIN/QUIK:** CUDA kernels, not usable on gfx1100. The batch-1 decode kernels in pancake already target memory bandwidth.
- **Mind the Gap (security action):** verify any third-party GGUF by comparing FP-dequantized behaviour to a self-quantized reference, or quantize from the official BF16 yourself. The 88.7% insecure-code gap is a direct risk for a coding agent that runs shell commands. Cost is zero except disk.
- **Sparse flash attention and Landmark attention:** only the attention layers of the hybrid use KV, and the user already has a parked-conversation cache. Both need training or kernel work, so expected gain is small for decode. They might help 131-262k prefill. Low priority.

### Gaps
- No ETH/EPFL paper evaluates rotation plus GGUF k-quants/i-quants specifically. The rotation-into-GGUF result above is extrapolated, not measured.
- Whether rotation composes with llama.cpp imatrix weighting (rotation changes activation statistics) is untested in the literature I found.
- Cevher (LIONS) quantization or MoE-routing papers were not located in this session.

## Q3. Agent, reasoning and evaluation work for opencode

### Takeaway
ETH is strong here. SRI Lab (Vechev) offers type-constrained decoding, which more than halves compile errors, plus LMQL, SWT-Bench and BaxBench. SPY Lab (Tramèr) offers AgentDojo, a prompt-injection benchmark for tool-using agents. SPCL (Besta/Hoefler) offers Graph of Thoughts and its successors. The cheapest wins for a single-user opencode agent are grammar/type-constrained decoding for tool-call JSON and edits (pancake already has tool-call recovery; constraining prevents failures instead) and running AgentDojo-style injection tests against the local agent.

### Cited Findings

- **Type-Constrained Code Generation with Language Models.** ETH SRI LEAD: Niels Mündler (first), Jingxuan He, Hao Wang, Koushik Sen, Dawn Song (Berkeley), Martin Vechev. PLDI 2025, arXiv 2504.09246. [verified]
  - Idea: prefix automata plus type inference and a search over inhabitable types enforce well-typedness during decoding. Formalized on a simply-typed language and extended to TypeScript.
  - Numbers: about 94% of compilation errors are type-check failures. The method cuts compilation errors by more than half and raises functional correctness on HumanEval/MBPP synthesis, translation and repair for 2B-34B models.
  - Code: github.com/eth-sri/type-constrained-code-generation.
  - Sources: [arXiv 2504.09246](https://arxiv.org/abs/2504.09246), [PLDI listing](https://pldi25.sigplan.org/details/pldi-2025-papers/25/Type-Aware-Constraining-for-LLM-Code-Generation)
- **Graph of Thoughts: Solving Elaborate Problems with LLMs.** ETH SPCL LEAD: Maciej Besta et al., Hoefler. AAAI 2024 (vol. 38(16), pp. 17682-17690), arXiv 2308.09687. Arbitrary graph of LLM thoughts with aggregation and refinement. Sorting quality +62% over Tree of Thoughts at more than 31% lower cost. Code: github.com/spcl/graph-of-thoughts (from my knowledge). [verified] — [arXiv 2308.09687](https://arxiv.org/abs/2308.09687v1), [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/29720), [SPCL page](https://www.unixer.de/publications/index.php?pub=495)
- **SPCL follow-ups**, all ETH SPCL LEAD (Besta et al.) [not re-fetched]:
  - "Demystifying Chains, Trees, and Graphs of Thoughts" (survey, arXiv 2401.14295);
  - "Multi-Head RAG" (arXiv 2406.05085), which uses multiple attention-head activations as retrieval keys for multi-aspect queries;
  - "Reasoning Language Models: A Blueprint" (arXiv 2501.11223);
  - "Affordable AI Assistants with Knowledge Graph of Thoughts" (KGoT, arXiv 2504.02670), where small models plus a knowledge graph of task state reach GAIA scores at a fraction of GPT-4o cost.
  - Sources: [2401.14295](https://arxiv.org/abs/2401.14295), [2406.05085](https://arxiv.org/abs/2406.05085), [2501.11223](https://arxiv.org/abs/2501.11223), [2504.02670](https://arxiv.org/abs/2504.02670)
- **AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents.** ETH SPY Lab LEAD: Debenedetti, Zhang, Balunović, Beurer-Kellner, Fischer, Tramèr. NeurIPS 2024 Datasets & Benchmarks, arXiv 2406.13352. About 97 realistic tool-use tasks and about 629 security test cases; extensible; code github.com/ethz-spylab/agentdojo. [not re-fetched] — [arXiv 2406.13352](https://arxiv.org/abs/2406.13352)
- **LMQL: Prompting Is Programming.** ETH SRI LEAD: Beurer-Kellner, Fischer, Vechev. PLDI 2023, arXiv 2212.06094. A query language with token-level constraints compiled to decoding masks; reported 26-85% fewer model calls/tokens in the paper (approximate). [not re-fetched] — [arXiv 2212.06094](https://arxiv.org/abs/2212.06094)
- **SWT-Bench: Testing and Validating Real-World Bug-Fixes with Code Agents.** ETH SRI LEAD: Mündler, Müller, He, Vechev. NeurIPS 2024, arXiv 2406.12952. Turns SWE-bench into a test-generation benchmark; code agents generate reproducing tests better than dedicated test tools. **BaxBench** (ETH SRI, 2025, arXiv 2502.11844): backend-generation benchmark with security exploits; most "correct" LLM backends were exploitable. [not re-fetched] — [SWT-Bench](https://arxiv.org/abs/2406.12952), [BaxBench](https://arxiv.org/abs/2502.11844)
- **ETH NLP (Cotterell):** "Locally Typical Sampling" (Meister, Pimentel, Wiher, Cotterell; TACL 2023, arXiv 2202.00666) reduces degenerate repetition versus nucleus sampling. It is related to pancake's loop guard, but pancake already has one. [not re-fetched] — [arXiv 2202.00666](https://arxiv.org/abs/2202.00666)
- **Ce Zhang / DS3Lab (pre-move):** FlexGen and Deja Vu were co-authored but led by Stanford and Rice respectively. Both are already covered in the US reports. No new ETH-led inference work in the window.

### Inferences

- **Tool-call JSON grammar plus type-aware constraint (pancake server):**
  - pancake (Python server) and Avalon (llama.cpp GBNF / json-schema grammars) can constrain tool-call arguments to the opencode tool schema. Mündler et al. show that going from syntax to types removes most remaining compile errors.
  - For opencode, the analogue is constraining `edit` tool `oldString` to substrings present in the file. That is a constrained-decoding check against a trie of file content, and removes the "oldString not found" retry loops.
  - Expected gain (my estimate): fewer failed tool turns. Each avoided retry saves a full turn, often more than 500 tokens of decode plus re-prefill. That can matter more than a 5% kernel gain.
  - Cost: a grammar-mask overhead per token on CPU, which is small at 126 t/s if the mask is incremental.
  - Interaction: masks must also apply to MTP and prompt-lookup draft verification. Reject drafts that violate the mask before verification.
- **AgentDojo as a regression suite:** run its injection cases through the local opencode stack to check that the loop guard and tool-call recovery do not amplify injected instructions. Zero performance gain; a safety check.
- **SWT-Bench / BaxBench as eval:** usable as quality gates when changing quantization (for example after a QuaRot-style re-quant). SWT-Bench Lite is smaller than SWE-bench and runs locally in Docker. This gives a coding-specific accuracy signal, better than perplexity, for 2-3-bit decisions.
- **Graph of Thoughts / KGoT:** these are orchestration patterns that increase LLM calls. On a single-user 126 t/s local engine, more calls cost wall-clock time. The useful part of KGoT is externalizing task state into a structured store so prompts stay short, which complements the parked conversation cache and thinking budgets.

### Gaps
- No ETH/EPFL work found on thinking-budget control or on speculative decoding specifically. Searches returned no Swiss-affiliated spec-decoding papers.
- EPFL NLP (Bosselut), DCL and DIAS: nothing directly applicable surfaced. Their 2023-2026 output is mostly knowledge and reasoning (MEDITRON, RECKONING), distributed/Byzantine learning, and databases. Not exhaustively checked.
- Exact AgentDojo, LMQL and SliceGPT numbers were not re-fetched this session.
