# Decoding-speed measurement papers for Pancake/Strata (rows A28, A32, A34, A49, A56)

Scope: read-only literature pass. No code edits, no builds, no GPU use. Target hardware: AMD RX 7900 XT (20 GB VRAM), MoE model, speculative decoding with an MTP draft head.

Tags: [paper, full text read] = fetched HTML/PDF body; [paper, abstract only] = arXiv abstract page only; [unverified] = search snippet or secondary news only.

Tool-call budget: about 15 calls (5 searches, 5 abstract fetches, 3 full-text fetches, plus this write). Several primary papers were not reachable in full text. Those gaps are listed per item.

---

## A28 - ISTA-DASLab Coder variant (256 of 512 experts, ~3.5 bpw LUT types)

### Takeaway
The only source for the DASLab "Coder" variant is secondary news coverage. It describes half the routed experts removed and about 3.5 bits per weight. No DASLab repo, model card, or paper was found, so nothing here can be called verified. The result is lossy (pruning plus quantization), and the claimed resident size (29.6 GB) is larger than this card's 20 GB VRAM.

### Cited Findings
- Claim (secondary): DASLab released "Qwen3.8-Flash-Next GSQ-RCO Coder", a 176.9B-parameter MoE. Half of the 512 routed experts per layer were removed, the remaining weights were quantized to about 3.5 bits per weight, and part of the model was packaged as disk-backed lookup data. Reported: 58.4 GB two-shard GGUF package, 29.6 GB that must stay memory-resident, 91.3% of BF16 SWE-bench Verified, 98.7% of LiveCodeBench v6. [unverified] — [AlphaSignal](https://alphasignal.ai/news/daslab-squeezes-a-176-9b-coding-model-down-to-32gb-hardware); same story at [Hermes AI](https://hermes-ai.net/news/daslab-squeezes-a-176-9b-coding-model-down-to-32gb-hardware/). Search snippets only; neither page was fetched.
- Related DASLab work, abstract/snippet only: QMoE, sub-1-bit compression of trillion-parameter MoE models ([arXiv 2310.16795](https://arxiv.org/abs/2310.16795)); IST-DASLab/MoE-Quant, GPTQ 4-bit DeepSeek-R1 variants, no 3.5 bpw mention ([GitHub, IST-DASLab/MoE-Quant](https://github.com/IST-DASLab/MoE-Quant); the search returned a mirror at gh.r-mj.com). [paper, abstract only] for QMoE; MoE-Quant [unverified] from search snippet.
- Separate one-shot domain pruning paper for coding MoE models, "Half the Experts, All the Code" ([arXiv 2607.16721](https://arxiv.org/pdf/2607.16721)). Snippet says its importance estimates were noise-dominated at its sample size. Not linked to DASLab by the search results. [unverified]

### Inferences
- Claim (A28 as written): "256 of 512 experts at ~3.5 bpw LUT types." Mechanism: halving routed experts halves expert weight bytes; a LUT-based 3.5 bpw format lowers bytes per weight further. Both are lossy. Quality must be checked on a coding eval, not on decode speed alone.
- Size check (inference from the secondary numbers): 29.6 GB resident is more than the 20 GB on the RX 7900 XT, so a card-only resident design is not possible. Any test would need offload, which changes the decode path.
- Smallest test that does not need a GPU: compute expert-tensor bytes per layer and per token at 3.5 bpw with 256 experts, versus the current quant format on this engine, from the model's tensor shapes. Pure arithmetic, CPU only.
- Making the 3.5 bpw LUT type fast on RDNA3 is a kernel project. The measurement that needs the GPU is decode tok/s with the new format, which is out of scope until a DASLab artifact is found.

### Gaps
- No primary source (DASLab repo, model card, arXiv paper) found for the Coder variant. Next step: search the IST-DASLab GitHub org and Hugging Face for the release, then fetch it.
- Quality numbers (91.3%, 98.7%) are unverified and depend on the eval harness.
- Did not check whether the engine already has a 3.5 bpw or LUT-based expert kernel. Not done in this pass.

---

## A32 - Stream experts to the GPU for large prefill ubatches

### Takeaway
Prefill is the phase where expert streaming works best: many tokens activate most experts, so each PCIe transfer is spread over many tokens. The literature supports streaming for large batches and warns that chunked prefill repeats transfers. The specific throughput numbers found are from abstracts or snippets. The engine's current prefill path was not checked in this pass, so the comparison is open.

### Cited Findings
- MoE-Lightning: CPU-GPU-I/O pipelining schedule (CGOPipe) with paged weights, up to 10.3x throughput over offloading systems for Mixtral 8x7B on one 16 GB T4 in batch inference; batch runs on 2-4 T4s for Mixtral 8x22B and DBRX. [paper, abstract only] — [arXiv 2411.11217](https://arxiv.org/abs/2411.11217). The abstract does not say which tensors are streamed or at what granularity.
- OSDI '26 "stream-loading prefill" for consumer GPUs: about 1,200 tokens/s on one node and about 1,800 tokens/s with a distributed variant on two RTX 5090s, 32K prompts within 30 s. [unverified] — search snippet only from [OSDI '26 PDF](https://www.usenix.org/system/files/osdi26-wang-wenxin.pdf). Not read in full.
- SeqMoE: chunked prefill is poorly suited to offloading because it repeats expert transfers across chunks; prefill still rarely reaches full-load performance under offloading. [unverified] — snippet from [arXiv 2609.12978](https://arxiv.org/pdf/2609.12978).
- TriMoE (snippet attributes these claims to it; the mapping of the arXiv link 2603.01058 to that title is unconfirmed): during prefill, experts are widely activated and GPU compute largely hides I/O; PCIe 5.0 at 64 GB/s is the bottleneck. [unverified] — [arXiv 2603.01058](https://arxiv.org/html/2603.01058).
- MoE-Infinity: activation-aware expert prefetching and caching. [paper, abstract only] — [arXiv 2401.14361](https://arxiv.org/abs/2401.14361).
- PreScope (snippet): prefetch and on-demand loading compete for PCIe bandwidth. [unverified], no link verified.

### Inferences
- Mechanism: with E experts activated per layer across ubatch of T tokens, the expected number of distinct experts touched grows with T, so bytes streamed per token fall as T rises. The break-even point is where per-layer expert compute hides the PCIe transfer.
- Exact vs lossy: scheduling only. Output should be bit-identical if the kernel path is unchanged; verify this.
- Smallest measurement: for ubatch sizes T in a sweep (for example 64, 256, 1024, 4096), log bytes copied host-to-device per ubatch and prefill tok/s. Needs the GPU. Pure accounting (distinct experts per ubatch from routing logs) can be done on CPU if routing traces are dumped.
- Do not assume a PCIe link speed for this box; read it from the GPU link state before using any bandwidth number.

### Gaps
- Did not read the engine's prefill code or MOD_PLAN A32 row in this pass. The "check what prefill does now" part of the assignment is therefore not done. Next step: read the prefill path and the A32 row.
- OSDI '26, SeqMoE, TriMoE, and PreScope were not read in full. Their numbers need a full read before use.
- No paper found that targets RDNA3 or ROCm specifically.

---

## A34 - Coordinate-descent sweep of launch and tile parameters

### Takeaway
A coordinate-descent refinement of tuned kernel parameters is established in compiler autotuning: AutoTVM's Droplet Search is cited by a 2024 Ansor patch paper, which reports acceptance into Ansor. Kernel Tuner and CLTune are the general tools. The found sources are abstracts or snippets, so the reported speedups are unverified.

### Cited Findings
- "Explore as a Storm, Exploit as a Raindrop": Ansor's exploration is capped at the first 300 kernels (vs 10,000 in baseline Ansor), then the best candidate is refined by coordinate descent borrowed from AutoTVM's Droplet Search. Evaluated on 20 DL models across AMD Ryzen 7, NVIDIA A100, RTX 3080, and ARM A64FX. [paper, abstract only] — [arXiv 2406.20037](https://arxiv.org/abs/2406.20037). The abstract says "usually" better kernels in less time and gives no speedup figures. The budget split between exploration and coordinate descent is not in the abstract.
- Kernel Tuner (van Werkhoven, Future Generation Computer Systems, 2018): auto-tuner for CUDA, OpenCL, and C with many search strategies; snippet claims a GEMM tuned 71.2x faster than brute force on an AMD Vega Frontier Edition. [unverified] — title and claim from search snippet only; no URL verified.
- CLTune: auto-tuner for OpenCL kernels over parameters such as workgroup size, tile sizes, and unrolling. [paper, abstract only] — [arXiv 1703.06503](https://arxiv.org/pdf/1703.06503).
- Lim, Norris, Malony: static plus predictive analysis to reduce empirical search for GPU kernels. [unverified] — [arXiv 1701.08547](https://arxiv.org/pdf/1701.08547).

### Inferences
- Mechanism: sweep one parameter at a time (block/workgroup size, tile M/N/K, unroll, LDS use, waves per EU), keep the best, move to the next, and repeat until no change. Cheap: roughly sum of the grid sizes per pass, not the product.
- Exact vs lossy: a launch or tile change alters the reduction order in GEMM and attention, so float output can change in the last bits. Treat as lossy unless the logits are checked bit-for-bit or to a fixed tolerance.
- Smallest measurement: one pass of the sweep over a single kernel (for example the decode GEMV) with a fixed shape, measured with the project's A B B A order and discard-first-run rule. Needs the GPU. Benchmarks belong in the night queue under the owner's rule.
- Overfitting risk: a sweep tuned at one context length or batch size can be wrong at another. Tie this to A49.

### Gaps
- No paper found that gives coordinate-descent sweep results specifically for RDNA3/ROCm LLM kernels.
- Droplet Search original and Kernel Tuner full text not read.
- Did not read the Phase 15 notes in MOD_PLAN.md, which were the assignment's pointer for the existing sweep method. Next step: read those notes and compare with the sweep order above.

---

## A49 - Decode at 10K vs 64K vs 128K: attention share vs context length

### Takeaway
Attention's share of decode time grows with active context, and the effect is stronger when FFN bytes per token are small. The best full-text source (SLIM) gives this as a qualitative trend, not a table of percentages, and it covers dense models only. No paper found reports the attention share for an MoE model across 10K/64K/128K. That measurement has to be made on this engine.

### Cited Findings
- SLIM (full text read in part; the HTML was truncated at 100,000 characters, but Section V was read): attention takes a "progressively larger share of decode time" as active context grows, whether from batch size, input length, or output length. For batch sizes above 64 attention exceeds the FFN (GEMM) share. Fig. 5 uses Mistral-7B; Granite-8B appears only in the roofline. Single H100 per experiment. [paper, full text read, partly] — [arXiv 2607.29575 HTML](https://arxiv.org/html/2607.29575). The paper gives no numeric attention-share percentages. Its hardware text says "64GB HBM2" for H100, which does not match published H100 memory and should not be trusted for hardware specs.
- SLIM abstract: throughput saturation comes from decode attention kernels; arithmetic intensity stays nearly constant as active context grows, so DRAM bandwidth saturates. [paper, abstract only] — [arXiv 2607.29575](https://arxiv.org/abs/2607.29575).
- Dense-vs-MoE decode latency split (FFN, attention, exposed communication): attention cost is about the same for dense and MoE because each token reads the same KV state, and MoE differences mostly come from FFN weight reuse; admissible batch shrinks with context. [unverified] — snippet from [arXiv 2603.08960](https://arxiv.org/html/2603.08960v1). Not read in full.
- Long-context study: decode latency rises about linearly with context, and the KV cache is still smaller than the weights at 50K context. [unverified] — snippet from [arXiv 2405.08944](https://arxiv.org/pdf/2405.08944).
- MoE offloading with up to 128K context on DeepSeek-V2-Lite: decode latency rises with context, and a growing KV cache shrinks the GPU buffer left for experts. [unverified] — snippet from [arXiv 2401.14361](https://arxiv.org/abs/2401.14361).

### Inferences
- Mechanism: per decode step, expert bytes read are roughly fixed (only top-k experts), while KV bytes read grow linearly with context. So attention share should rise with context, and faster in an MoE with small active FFN bytes.
- On this card, 128K KV plus expert weights may not fit in 20 GB VRAM. If so, the 128K point measures offload, not attention. Check the KV size formula for this model's attention type (GQA, MLA, or other) before choosing the points.
- Smallest measurement: decode tok/s and per-kernel time at 10K, 64K, and 128K, with a profiler (rocprof) to get the attention share directly. Keep batch and weights fixed. Needs the GPU. Use A B B A order.
- Use the attention share to decide whether A34 sweeps should target attention kernels or GEMMs at each context length.

### Gaps
- No paper found with attention share measured for an MoE model across these context lengths. The MoE link is inferred.
- SLIM Fig. 5 numbers are not in the text. Could not extract them.
- Did not read the engine's KV layout or the model's attention config in this pass.

---

## A56 - Expected acceptance rate (EAR) beside top-token agreement

### Takeaway
EAR is the expected per-token acceptance probability alpha: average over prefixes of the per-prefix acceptance probability beta. The source defining alpha and the expected-tokens formula is Leviathan et al. (full text read). The scaling-laws paper (2603.11053) defines EAR in those terms and was read in part. Top-token agreement is a different quantity (argmax match). They coincide only under greedy verification (inference).

### Cited Findings
- Leviathan et al., speculative decoding. Acceptance rule (Section 2.3): draft token x ~ q is kept with probability min(1, p/q); on rejection a replacement is drawn from an adjusted distribution. With alpha = E(beta) and i.i.d. per-token acceptance, expected tokens per target call is (1 - alpha^(gamma+1)) / (1 - alpha) (Section 3.1, Eq. 1). [paper, full text read] — [arXiv 2211.17192 HTML v2](https://arxiv.org/html/2211.17192v2). The original abstract page was also fetched: [arXiv 2211.17192](https://arxiv.org/abs/2211.17192) (abstract only).
- Speculative Decoding Scaling Laws (SDSL): defines beta as the acceptance probability of a draft token given a prefix, and alpha as its average over prefixes, "the expected acceptance rate across prefixes" (Section 2.1). Fits alpha = Ax + By + C with draft perplexity x and target perplexity y (Section 4.2, Eq. 5); R^2 = 0.602, MSE = 0.00128; draft perplexity dominant. [paper, full text read in part] — [arXiv 2603.11053 HTML](https://arxiv.org/html/2603.11053v1). The text was truncated at about 100K characters, and the appendices were only partly visible. The sign of B (positive in Table 3) appears to conflict with a remark in Appendix E about lower-perplexity targets; not resolved here.
- NVIDIA AIPerf docs: per-request draft acceptance rate is accepted/proposed (j/l); "acceptance length" tau is the average tokens accepted per round. [unverified] — [NVIDIA AIPerf docs](https://docs.nvidia.com/aiperf/tutorials/datasets-inputs/per-request-speculative-decoding-metrics). Snippet only.
- Tree-draft acceptance across cognitive domains: expected accepted length for trees is a sum over depths of products of per-depth acceptance probabilities. [unverified] — snippet from [arXiv 2604.14682](https://arxiv.org/abs/2604.14682).

### Inferences
- Naming is inconsistent across sources: "acceptance rate" (alpha), "acceptance length" (tau), and "EAR" are not interchangeable. Any table must state which one it reports.
- Exact vs lossy: a metric, not a speed change. Adding EAR does not alter output.
- Greedy verification: acceptance reduces to "draft token equals target argmax," so EAR and top-token agreement match in greedy mode. Under sampling they differ. This is an inference from the acceptance rule above.
- Smallest measurement: on the MTP draft head, dump draft and target probabilities at each draft position and compute the mean of min(1, p/q) per position (a lower-variance estimate of alpha than counting sampled accepts). Compare the measured tokens per step with the i.i.d. formula to test the i.i.d. assumption. Needs the GPU to produce the logits; the metric itself can run on CPU from dumped probabilities.

### Gaps
- Did not find an original paper that introduces the exact term "expected acceptance rate" with a name-only definition. The concept comes from Leviathan et al. and Chen et al.; the name EAR appears in later papers. Not verified in full text.
- SDSL fit sign conflict and R^2 = 0.60 are unresolved.
- Did not check whether the engine already logs per-position acceptance.
