# UK university AI papers (Nov 2022 - Nov 2026) applicable to pancake / Avalon

Scope note: this covers UK-university-led or co-led work. Prior reports (`/home/dan/reports/`, `research_notes/Lucrări universități SUA pentru Avalon/`, pancake's `research_notes/Pancake MoE engine speedup/`) mention MoE-Infinity only in one line (3.1-16.7x headline) and do not analyse its mechanism. They do not cover MoE-Gen/BatchGen, ContextPilot, BitDecoding, AcceptMoE, DMC/DMS, Sparse Frontier, or LQER/QERA. So everything below is new except that one-line MoE-Infinity mention.

Research budget: about 20 searches and fetches. The Edinburgh (Luo Mai, Ponti) and Imperial (Constantinides/Zhao, Luk) coverage is solid. Cambridge, Oxford, UCL, Manchester, Bristol and Warwick are thin: see Gaps.

## Q1. Edinburgh's MoE-Infinity line: does it beat pancake's tier + per-conversation working sets, and what would porting need?

### Takeaway
MoE-Infinity is close to what pancake already has: request-level expert-frequency traces that steer prefetch and eviction. It predates pancake's design and is weaker on two points. It never computes on the CPU; it fetches every miss over PCIe. And it reports no hit rates, only latency against weak baselines (vLLM, Ollama, DeepSpeed). It is unlikely to beat pancake's LRU at 89.5%. Only two parts are worth an offline replay test: the cosine-matched "nearest past conversation" predictor and the early-layer eviction bias. The ceiling is the 5.2-point LRU-to-Belady gap. The successors (MoE-Gen, BatchGen) target throughput and big batches, not batch-1 decode.

### Cited Findings
**MoE-Infinity** (lead: Edinburgh, Luo Mai's group; all six authors are Edinburgh). Authors: Leyang Xue, Yao Fu, Zhan Lu, Chuanhao Sun, Luo Mai, Mahesh Marina. arXiv 2401.14361: v1 Jan 2024, v3 12 Mar 2025, retitled "Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache". No conference venue is listed on arXiv. Code: github.com/EfficientMoE/MoE-Infinity — [arXiv abs](https://arxiv.org/abs/2401.14361); [v3 HTML](https://arxiv.org/html/2401.14361v3)
- **Premise:** personal machines serve one user at batch size 1. Decode then shows high activation sparsity, with a small set of experts reused often — [arXiv abs](https://arxiv.org/abs/2401.14361)
- **Expert Activation Matrix (EAM):** an L×E matrix whose entry [i][j] counts tokens routed to expert j at layer i. An iEAM covers one iteration; an rEAM accumulates over a request, with prefill and decode kept separate — [v3 HTML](https://arxiv.org/html/2401.14361v3)
- **EAMC:** a fixed-capacity collection of past rEAMs. The current request's partial EAM is matched to past ones by cosine distance over flattened EAMs. Several matches are summed and row-normalised, then weighted by a layer-proximity factor (1−(i−l)/L) to give a predicted EAM. When the collection is full, the most similar stored EAM is replaced, which keeps it diverse — [v3 HTML](https://arxiv.org/html/2401.14361v3)
- **Eviction:** score p = (pEAM[layer]+ε)·(1−layer_idx/L)/n_token, and the lowest score is evicted. Early layers get higher caching priority because their predictions are less reliable. Prefetch uses the predicted EAM for the next layers. On a miss with a full cache the system evicts and fetches on demand — [v3 HTML](https://arxiv.org/html/2401.14361v3)
- **Results:**
  - DeepSeek-V2-Lite TPOT: MoE-Infinity 155 ms, vLLM 485 ms, DeepSpeed 737 ms, Mixtral-Offloading 1250 ms, Ollama 2590 ms.
  - Mixtral TPOT is 836 ms.
  - Hardware: one RTX A5000 24 GB on PCIe 4.0.
  - The paper is internally inconsistent: "3.1–16.7×" in the abstract and evaluation, "2.7–13.7×" in the conclusion — [v3 HTML](https://arxiv.org/html/2401.14361v3)
- **No hit-rate numbers are reported.** The comparison with LRU (llama.cpp, vLLM) and with BrainStorm-style counting is a worked example (Fig. 6), not a measurement — [v3 HTML](https://arxiv.org/html/2401.14361v3)
- **No CPU compute:** misses are fetched to the GPU. The paper notes that llama.cpp computes offloaded parameters on the CPU, but MoE-Infinity does not — [v3 HTML](https://arxiv.org/html/2401.14361v3)
- **Stated limits:**
  - At long context the KV cache squeezes the expert buffer, down to about 1 GB at 128K, and performance degrades.
  - Mixtral, with few large experts, is the worst case.
  - Clustering-based EAMC is left as future work — [v3 HTML](https://arxiv.org/html/2401.14361v3)
- **Independent check:** MoE-Beyond (not UK) reports MoE-Infinity's heuristic at only 17% hit rate with a 10%-of-experts cache in simulation, against 72% for its learned predictor — local pancake notes citing [MoE-Beyond](https://arxiv.org/html/2508.17137v1)

**MoE-Gen** (lead: Edinburgh). Authors: Tairan Xu, Leyang Xue, Zhan Lu, Adrian Jackson (EPCC), Luo Mai. arXiv 2503.09716, Mar 2025. Code: github.com/EfficientMoE/MoE-Gen — [arXiv](https://arxiv.org/abs/2503.09716)
- **Idea:** module-based batching. Tokens accumulate in host memory and are launched in large per-module batches, with a batch size per module chosen to overlap compute and transfer.
- **Results:** 8-31x throughput over model-based offloading (FlexGen, MoE-Lightning, DeepSpeed); the HTML version says 16-33x decode throughput. It targets offline throughput — [arXiv HTML](https://arxiv.org/html/2503.09716v1); [search summary](https://arxiv.org/pdf/2503.09716)

**BatchGen** (lead: Edinburgh, with Tencent). Authors: Tairan Xu, Leyang Xue, Zhan Lu, Jinfu Deng, Hongyang Xiao, Luo Mai. OSDI 2026, arXiv 2606.21712. Code: github.com/batchgen-project/batchgen — [USENIX](https://www.usenix.org/conference/osdi26/presentation/xu-tairan); [arXiv](https://arxiv.org/abs/2606.21712)
- **Idea:** a "sequence coroutine" compute model.
- **Results:** up to 2.3x lower batch completion time on 128 GPUs, and up to 9.6x over the strongest offloading baseline on memory-constrained accelerators.
- **Models and hardware:** Mixtral, DeepSeek-R1 and Kimi-K2 on H20/H200 — [USENIX](https://www.usenix.org/conference/osdi26/presentation/xu-tairan)

**ServerlessLLM** (lead: Edinburgh). Authors: Yao Fu, Leyang Xue, Yeqi Huang, Andrei-Octavian Brabete, Dmitrii Ustiugov, Yuvraj Patel, Luo Mai. OSDI 2024, arXiv 2401.14351 — [arXiv](https://arxiv.org/abs/2401.14351)
- **Idea:** a loading-optimised checkpoint format plus a multi-tier loader that saturates the server's storage hierarchy, live migration, and locality-aware scheduling.
- **Results:** 10-200x lower latency than serverless baselines — [arXiv](https://arxiv.org/abs/2401.14351)

**MoE-CAP** (lead: Edinburgh). Authors: Dayou Du, Tairan Xu, Kai Zou, Edoardo Ponti, Luo Mai. NeurIPS 2025 Datasets & Benchmarks. A cost/accuracy/performance benchmark for sparse MoE systems — [Luo Mai homepage](https://luomai.github.io/); [search summary](https://www.proceedings.com/content/085/085713-2717open.pdf)

**Luo Mai's 2026 list:** no new batch-1 expert-offload paper. The 2026 entries are BatchGen (OSDI), ContextPilot (MLSys), BitDecoding (HPCA), SwarmPilot (EuroSys 2027, agentic scheduling), and wafer-scale work (WaferLLM OSDI'25, Wavel and MeshRT SOSP'26) — [Luo Mai homepage](https://luomai.github.io/)

### Inferences
- **pancake already covers MoE-Infinity.** Per-conversation and per-domain working sets are the rEAM idea. Tiered VRAM, pinned RAM and NVMe plus CPU compute for misses go beyond it, because MoE-Infinity serialises every miss over PCIe.
- **Its eviction score is a frequency score.** The score is a per-request predicted frequency times a layer-depth weight. That is close to the ClassLRU scoring and reuse-aware admission already tested in pancake with no gain, so expect little from a straight port.
- **Port recipe, offline replay only, about a day:**
  1. Build rEAMs from existing pancake replay traces.
  2. Keep an EAMC of about 100-1000 past conversations.
  3. Each decode step (or every N tokens), match the current partial EAM by cosine and form the pEAM.
  4. Evict by the score above.
  5. Report the hit rate against LRU 89.5% and Belady 94.7%.
  6. Add the early-layer bias (1−l/L) separately on top of LRU. FATE's "shallow-favoring" cache reached 99% in its own setting, so this term is the cheapest one to test alone.
  7. Go further only if it gains at least 1.5 points over LRU.
- **Expected gain (rough estimate, not measured):**
  - At an 89.5% hit rate, misses are about 10.5%, and CPU-side miss compute is about half of decode.
  - Raising the hit rate to 91-92% cuts misses by 15-25%. That might give 5-10% decode (126 → about 132-138 t/s) if CPU miss time scales linearly.
  - Likely result: 0-3%.
- **MoE-Gen and BatchGen do not help batch-1 decode.** Their module-batching idea applies only to long-prompt prefill (load each expert once per ubatch). llama.cpp-style ubatched prefill already does this.

### Gaps
- No public per-step hit-rate numbers for MoE-Infinity at batch 1, so it cannot be compared directly with pancake's 89.5%.
- I did not confirm whether the MoE-Infinity GitHub repo has a llama.cpp/GGUF or ROCm backend. The paper describes a PyTorch/CUDA system.
- ServerlessLLM's loader microbenchmarks (GB/s from NVMe) were not extracted.

## Q2. Other UK work on expert offload, CPU+GPU hybrid MoE, NVMe streaming, and MoE speculative decoding

### Takeaway
The only other UK-affiliated batch-1 MoE-offload paper found is Imperial's AcceptMoE (Aug 2026). It shrinks the expert set the verifier touches during speculative decoding, favouring cached experts and weighting by acceptance probability. That maps directly onto pancake's MTP + prompt-lookup verification and is the most promising UK idea for pancake. No Cambridge, Oxford, UCL, Manchester, Bristol or Warwick paper on expert offload, CPU/GPU hybrid MoE or NVMe streaming turned up.

### Cited Findings
**AcceptMoE** (Imperial College London, Wayne Luk's group, with Tile-AI; Imperial is probably the lead, but the author-to-affiliation mapping is not shown). Authors: Shuang Liang, Hao Chen, Zhiwen Mo, Qianzhou Wang, Guoyu Li, Lingxiao Ma, Wayne Luk. arXiv 2608.02989, 4 Aug 2026. No code link found — [alphaXiv](https://www.alphaxiv.org/abs/2608.02989)
- **Problem:** verifying a draft tree activates the union of experts across all nodes.
- **Idea:** restrict verification to a smaller expert set.
  - Each expert is weighted by the estimated acceptance probability of the tokens that need it.
  - The set size comes from the entropy of the demand distribution, so there is no manual budget.
  - Under offloading, non-resident experts are pruned in favour of cached ones.
- **Results:**
  - Batch-1 throughput in SGLang: 1.29x over EAGLE-3 with all experts resident, 2.06x under offloading.
  - Host-to-device expert traffic −73.6 to −77.1%; cache hit rate +10.8 to +14.2 points.
  - Mean accuracy −0.27 points over 12 model-task pairs.
  - Models: Qwen3-30B and GPT-OSS-120B. Tasks: GSM8K, MATH500, HumanEval, MBPP.
  - Source: [alphaXiv](https://www.alphaxiv.org/abs/2608.02989)
- **Not UK, for context only:** an OSDI 2026 paper, "Achieving Cloud-Grade SLOs for Local Mixture-of-Experts Inference through CPU–GPU Hybrid Design" (Wang Wenxin et al.), is directly on topic. Its affiliation was not verified and is probably not UK — [USENIX](https://www.usenix.org/conference/osdi26/presentation/wang-wenxin)

### Inferences
- **AcceptMoE fits pancake.** pancake verifies MTP and prompt-lookup drafts at batch 1. Each extra draft token can pull in uncached experts that the CPU must then compute, which is the dominant cost.
- **Port sketch:**
  1. During verification, for each layer, take the union of the top-k experts across draft positions.
  2. Drop or substitute low-weight, non-resident experts. Weight = router probability × the draft position's acceptance probability, which falls with depth. Renormalise the remaining gates.
  3. Never prune for position 0, the token that is guaranteed to be accepted. This keeps the output exact for the committed token. Later draft tokens become slightly lossy, which matches the paper's −0.27 accuracy.
- **Expected gain:** if CPU miss work during verification is a large share, a 10-25% decode gain is plausible. That is a guess from the 2.06x figure scaled by pancake's already-high hit rate.
- **Risk:** overlap with the failed "DraftExpert miss-aware windows" experiment. AcceptMoE differs in that it changes the verifier's expert set, not the drafting window. It needs quality checks on opencode tasks because routing becomes lossy.
- **Possible NVMe tier improvement:** ServerlessLLM's loading-optimised format (large aligned chunks, parallel multi-tier loader) could inform pancake's NVMe expert layout, for example contiguous per-expert blobs with O_DIRECT/io_uring. The gain only matters for NVMe-tier misses, so expect it to be low to modest.

### Gaps
- **Cambridge:** no on-topic paper found from Nic Lane's group or from Cambridge MLSys on MoE offload or on-device LLM inference in the window. Searches returned non-Cambridge papers (EdgeMoE, D²MoE, Qualcomm's cache-conditional experts). Searching Lane's DBLP or Google Scholar directly might find something.
- **Oxford, UCL, Manchester, Bristol, Warwick:** no relevant inference-systems papers surfaced in two broad searches. That is a coverage gap, not evidence that none exist.
- **AcceptMoE details:** hardware, the full affiliation mapping and code availability are unconfirmed.

## Q3. UK quantization, decoding, KV/attention, agent and evaluation work that fits

### Takeaway
The most useful UK work for pancake falls into four areas:
- **Quality at 2-3 bits:** Imperial's LQER/QERA low-rank error correction.
- **Long-context attention sparsity:** Edinburgh's Sparse Frontier finding that decode-time page selection (Quest-style) tolerates high sparsity.
- **Context reuse:** Edinburgh's ContextPilot.
- **Evaluation:** MoE-CAP.

KV-compression retrofits (DMC/DMS) need fine-tuning that is impractical for a 125B model. BitDecoding is a CUDA tensor-core design, and TurboQuant KV already showed no gain.

### Cited Findings
**LQER** (lead: Imperial; Cheng Zhang, Jianyi Cheng, George Constantinides, Yiren Zhao). ICML 2024 — [PMLR](https://proceedings.mlr.press/v235/zhang24j.html)
- Low-rank reconstruction of quantization error using an activation-induced scale matrix; nearly lossless W4A8 — [PMLR](https://proceedings.mlr.press/v235/zhang24j.html)

**QERA** (lead: Imperial; Cheng Zhang, Jeffrey T. H. Wong, Can Xiao, George Constantinides, Yiren Zhao). ICLR 2025, arXiv 2410.06040 — [arXiv](https://arxiv.org/html/2410.06040v2)
- A closed-form solution for the low-rank error term that minimises output (activation) error.
- Results: +2.97% accuracy on 4-bit Llama-3.1-70B over ZeroQuant-V2; WikiText2 Δppl −0.28 versus LQER.
- Sources: [arXiv](https://arxiv.org/html/2410.06040v2); [ICLR slides](https://iclr.cc/media/iclr-2025/Slides/30014.pdf)

**The Sparse Frontier** (lead: Edinburgh, Ponti's group, with Cohere). Authors: Nawrot, Li, Huang, Ruder, Marchisio, Ponti. arXiv 2504.17768, ACL 2026 Findings — [arXiv](https://arxiv.org/html/2504.17768v2); [ACL](https://aclanthology.org/2026.findings-acl.1926.pdf)
- A study of training-free sparse attention.
- Longer sequences tolerate higher sparsity.
- At decode, token-to-page selection (Quest) is cheap and tolerates more compression than sparse prefill — [search summary of arXiv/ACL](https://arxiv.org/html/2504.17768v2)

**Dynamic Memory Compression (DMC)** (Edinburgh, Ponti, as co-author or last author; Nawrot was at Edinburgh, with NVIDIA and Wrocław). ICML 2024, arXiv 2403.09636 — [arXiv](https://arxiv.org/abs/2403.09636)
- Learned per-head append-or-merge of KV entries.
- Retrofitting costs about 2% (2x compression) to 8% (8x) of the original pretraining data.
- Up to about 7x throughput on H100 in v2; 3.4-3.7x in other configurations — [arXiv v2](https://arxiv.org/html/2403.09636v2)

**Inference-Time Hyper-Scaling / DMS** (Łańcucki, Nawrot, Ponti et al.; NVIDIA + Edinburgh). NeurIPS 2025, arXiv 2506.05345 — [arXiv](https://arxiv.org/html/2506.05345v1)
- 8x KV compression after 1K training steps via delayed eviction — [arXiv](https://arxiv.org/html/2506.05345v1)

**ContextPilot** (lead: Edinburgh). Authors: Yinsicheng Jiang, Yeqi Huang, Liang Cheng, Cheng Deng, Xuan Sun, Luo Mai. MLSys 2026 oral, arXiv 2511.03475. Code: github.com/EfficientContext/ContextPilot (Apache-2.0; the README claims vLLM, SGLang and llama.cpp support) — [MLSys proceedings](https://proceedings.mlsys.org/paper_files/paper/2026/hash/b0131b6ee02a00b03fc3320176fec8f5-Abstract-Conference.html); [PyPI](https://pypi.org/project/contextpilot/)
- A context index finds overlapping context blocks across requests and turns. Alignment and de-duplication raise KV reuse, and annotations keep quality.
- Up to 3x lower prefill latency — [MLSys proceedings](https://proceedings.mlsys.org/paper_files/paper/2026/hash/b0131b6ee02a00b03fc3320176fec8f5-Abstract-Conference.html)

**BitDecoding** (lead: Edinburgh, with Microsoft Research). Authors: Dayou Du, Shijie Cao, Jianyi Cheng, Luo Mai, Ting Cao, Mao Yang. arXiv 2503.18773 (v3 Jan 2026); the homepage lists HPCA 2026. Code: github.com/DD-DuDa/BitDecoding — [arXiv v3](https://arxiv.org/html/2503.18773v3); [Luo Mai homepage](https://luomai.github.io/)
- Tensor cores do the matmul while CUDA cores dequantize the KV cache.
- 7.5x average over FP16 FlashDecoding-v2; 3x lower single-batch latency on Llama-3.1-8B at 128K. NVIDIA only — [arXiv v3](https://arxiv.org/html/2503.18773v3)

**MoE-CAP** (Edinburgh, NeurIPS 2025 D&B): see Q1. **SwarmPilot** (Edinburgh, EuroSys 2027): agentic scheduling for agentic systems; details not read — [Luo Mai homepage](https://luomai.github.io/)

### Inferences
- **LQER/QERA: medium value for quality, not speed.** A rank-16 to rank-32 low-rank correction on 2-3-bit expert weights could recover quality lost to the aggressive GGUF quantization. The cost is an extra small GEMV per expert, which is cheap next to an expert matmul but adds bytes to the VRAM cache.
  - The fair test: an IQ2/IQ3 model plus low-rank correction, against the next-larger quant at equal VRAM.
  - The quality and speed tradeoff for pancake is unmeasured.
- **Sparse Frontier: low to medium.** Qwen3.8-Flash-Next is a Gated DeltaNet hybrid, so only the full-attention layers have growing KV. Quest-style page selection at decode would cut attention reads at 64K+ contexts typical of opencode sessions. The gain depends on attention's share of decode time; likely a few percent at long context, near zero at short.
- **ContextPilot: low.** pancake already has a conversation cache with message-boundary checkpoints, which covers prefix reuse. ContextPilot's non-prefix block reuse conflicts with the GDN recurrent state, which cannot be spliced.
- **DMC/DMS: not practical.** Both need retraining on a 125B MoE, and attention KV is a small share in a GDN hybrid.
- **BitDecoding: not applicable.** It is a CUDA design (its RDNA3 WMMA analogue is unexplored), and TurboQuant KV already showed no gain.
- **MoE-CAP: useful as methodology.** Report pancake results with sparsity-aware utilisation metrics so they can be compared with academic systems. The exact metric names were not verified.

### Gaps
- No UK-led agent or coding-agent evaluation paper relevant to opencode-style single-user agents was found within budget. SwarmPilot targets multi-agent cluster scheduling; its details were not read.
- No UK-led speculative decoding paper (MTP, prompt lookup, tree verification) other than AcceptMoE was found.
- HPCA 2026 acceptance for BitDecoding is listed on the homepage but not independently confirmed.
- No UK-led kernel/compiler work targeting AMD RDNA3 was found. Edinburgh's compiler work (Wavel, MeshRT) is wafer-scale only.
