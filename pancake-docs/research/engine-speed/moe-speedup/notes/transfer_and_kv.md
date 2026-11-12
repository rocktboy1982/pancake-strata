# Host-to-GPU MoE expert transfer (A) and KV cache / attention for hybrid models at long context (B), for pancake-strata on 1x RX 7900 XT / PCIe 3.0

Target recap, used for the applicability judgements: Qwen3.8-Flash-Next 125B MoE, 512 experts top-10, IQ2_XS experts of about 1.37 MiB (1.437 MB) each, 48 layers, so 480 expert activations per decoded token. The engine has hybrid Gated DeltaNet plus QSA top-k attention layers with int8 KV, 32,768 cells in VRAM and older KV in pinned RAM. Hardware: 7900 XT 20 GB, H2D 13.4 GB/s measured, CPU expert reads about 33 GB/s, DDR4-3200 dual channel (51.2 GB/s theoretical peak).

Research budget note: about 15 web calls. Several items below are marked as gaps that need follow-up. Arithmetic in "Inferences" is mine, built from the stated target numbers and the cited mechanisms.

## A1. CPU compute vs PCIe copy for cache misses: cost models and when copying wins on PCIe 3.0

### Takeaway
Every published hybrid system (Fiddler, kTransformers, HybriMoE, llama.cpp's new expert cache) computes misses on the CPU at batch=1 and copies weights only when the copy is reused (caching) or the batch is large (prefill). On this machine a decode miss costs about 44 us on the CPU and about 107 us to copy. DMA reads also compete with CPU reads for the same DDR4 bandwidth. Copying a miss only to compute it once is therefore a net loss, which matches the A/B result that turning PCIe share off was about 10% faster. Copies should be driven by cache admission (expected reuse), not by load-balancing the current token.

### Cited Findings
- Fiddler (ICLR 2024, older): it moves activations (batch x 4096 for Mixtral) to the CPU instead of moving weights to the GPU, because at small batch the activations are far smaller than the weights. It picks CPU or GPU per expert layer from the input count and estimated CPU vs GPU latency, and places popular experts on the GPU using offline profiling. It reports 8.2x (Quadro RTX 6000) and 10.1x (L4) single-batch latency improvement over offloading baselines, and more than 3 tok/s for uncompressed Mixtral-8x7B on a 24 GB GPU. Code: github.com/efeslab/fiddler — [arXiv 2402.07033](https://arxiv.org/pdf/2402.07033); [UW SyFI](https://syfi.cs.washington.edu/publications/fiddler)
- A secondary summary reports Fiddler's speedup over llama.cpp as only 1.26x — [liner review](https://liner.com/ko/review/fiddler-cpugpu-orchestration-for-fast-inference-mixtureofexperts-models)
- HybriMoE (DAC 2025): built on kTransformers. It uses dynamic intra-layer CPU/GPU scheduling, impact-driven inter-layer prefetching, and a score-based "Minus Recent Score" cache. It reports 1.33x prefill and 1.70x decode on average over kTransformers. It notes that Fiddler and kTransformers compute misses on the CPU instead of transferring them. Code: PKU-SEC-Lab/HybriMoE — [arXiv 2504.05897](https://www.arxiv.org/pdf/2504.05897); [EmergentMind summary](https://www.emergentmind.com/papers/2504.05897)
- kTransformers: Expert Deferral reorders expert execution across layers so CPU expert compute overlaps GPU attention. It switches from AVX-512 to AMX kernels when each expert sees more than about 4 tokens on average. The SGLang integration reports 2.42-4.09x over Fiddler without deferral — [LMSYS blog 2025-10-22](https://www.lmsys.org/blog/2025-10-22-KTransformers)
- llama.cpp mainline `--moe-cache-mib` (PR #29887, reported merged Oct 7): an LRU GPU cache of host-resident experts used for batches of 32 tokens or fewer. Only misses are uploaded over PCIe. The suggested size is about 10% of total expert bytes. On Qwen3.8-Flash-Next Q4_0 (65.4 GiB of experts), EPYC 7742, PCIe 4.0 x16: RTX 4090 goes from 25.0 to 40.7 tok/s at a 77% hit rate, and RTX 5090 from 30.8 to 67.8 tok/s at 89%. A Radeon AI PRO R9700 (ROCm) goes from 11.0 to 27.6 tok/s at 16,000 MiB (94.6% hit). An RTX 5060 Ti with IQ3_XXS goes from 14.95 to 19.26 tok/s. A tester saw a slowdown at a 58% hit rate, and reviewers point to 75% or more as the break-even. Large caches cut prompt speed on 2x4090 (440.7 to 321.5 tok/s) — [Developer's Digest summary of PR #29887/#30112](https://www.developersdigest.tech/blog/llama-cpp-moe-offload-n-cpu-moe-vs-moe-cache-2026). This is a secondary source and the PR text was not verified.
- llama.cpp op offload: for batches of 32 or more tokens, CPU-resident weights are copied to the GPU instead of computed on the CPU (`GGML_OP_OFFLOAD_MIN_BATCH`). ik_llama.cpp scales the threshold by total/active experts. The guide author says 32 is likely too low for slow PCIe — [Doctor-Shotgun HF guide](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide); [gist](https://gist.github.com/DocShotgun/a02a4c0c0a57e43ff4f038b46ca66ae0)
- PreScope/LayerScope (2025): on-demand PCIe transfers often take several times longer than the matching GPU compute. PreScope reports 141% higher throughput than Klotski through predictive prefetch — [arXiv 2509.23638](https://arxiv.org/html/2509.23638v1)

### Inferences
- Per-miss cost model at batch=1 on the target. Copying costs 1.437 MB / 13.4 GB/s = 107 us. CPU compute is memory-bound, 1.437 MB / 33 GB/s = 44 us. The CPU is about 2.5x cheaper per miss unless the copied expert is hit again before eviction. Break-even: a copied expert must save at least about 0.6 future CPU computes, about 1.5 total uses, within its cache lifetime. That makes it a cache admission decision based on predicted reuse, not a parallelism decision.
- Parallel split, if the two paths were independent: equal finish time at a CPU fraction f where f/33 = (1-f)/13.4, giving f ≈ 0.71 and a PCIe share of about 0.29. The auto-tuner's 0.20-0.25 sits near that. Both paths read the same DDR4, though: 33 + 13.4 = 46.4 GB/s is about 90% of the 51.2 GB/s theoretical peak and likely above the sustainable dual-channel rate. The split probably slows the CPU path, which explains why PCIe share off measured about 10% faster. Recommendation: set PCIe share to about 0, or let it admit only experts whose predicted reuse is at least 2. Run the CPU expert path with DMA active to measure the contention directly.
- Literature hit-rate threshold: llama.cpp testers see gains only above about 75% hit. If Strata's adaptive tier hit rate is below that, sizing and policy matter more than transfer mechanics. Candidate policies: HybriMoE's score-based cache, and llama.cpp's LRU as the baseline.
- Expected gain on the target: about 0-10% decode, mostly from removing the contention, and the current A/B already shows this. More only comes from a higher hit rate, which depends on cache policy.

### Gaps
- "Apto" and "Apex" as MoE offload systems: no sources were found in this pass. Names may be mistyped or unpublished.
- No published measurement of CPU and DMA bandwidth contention on DDR4 dual-channel APUs was found.

## A2. Pinned memory, hipHostRegister, zero-copy kernel reads, Large BAR, SDMA vs blit copies on AMD

### Takeaway
Nothing changes the 13.4 GB/s ceiling. A PCIe 3.0 x16 link is about 15.75 GB/s raw (my number), and 13.4 GB/s is already about 85% of raw, near the practical maximum. Zero-copy kernel reads, SDMA vs blit, and ReBAR can only remove latency and staging overhead or free compute queues. They cannot add bandwidth. The useful levers are fewer, larger copies issued early on a separate engine and stream.

### Cited Findings
- HSA_ENABLE_SDMA (default on) controls DMA-engine use for H2D, D2H and D2D in hsa_memory_copy, hsa_amd_memory_async_copy and async_copy_on_engine. Setting it to 0 forces the shader blit-kernel path — [ROCR env vars](https://rocm.docs.amd.com/projects/ROCR-Runtime/en/master/api-reference/environment_variables.html)
- AMD docs: kernels reading host-pinned memory traverse PCIe and are much slower than on-device bandwidth (more than 40x on MI200). The docs call this generally not recommended. Pinned allocations are coherent (fine-grained, not cached in the GPU L2) by default. hipHostMallocNonCoherent lets the GPU cache host data in L2. The device pointer comes from hipHostGetDevicePointer. Default HIP copies from pinned memory "run at the limit of the interconnect" — [ROCm GPU memory](https://rocm.docs.amd.com/en/docs-6.2.1/conceptual/gpu-memory.html); [GPUOpen MI200 memory space](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-mi200-memory-space-overview)

### Inferences
- Zero-copy (a GEMV kernel reading IQ2_XS experts directly from non-coherent pinned host memory): for decode misses this would be a third compute path between CPU and copy. It reads each byte once at no more than PCIe rate, so about 107 us per expert, the same as a copy plus compute but without launch and sync latency. It does not beat CPU compute at 44 us, and it competes for DDR4 bandwidth in the same way. It is low value for decode. For prefill, a kernel reading the expert once per chunk equals the DMA path in bytes. Expected gain: about 0. It also loses async overlap unless run on a separate low-priority queue.
- SDMA vs blit: blit kernels consume CUs and SDMA does not. For prefill streaming overlapped with compute, keep SDMA. Batch many 1.4 MB experts into large contiguous copies (lay experts of one layer out contiguously in pinned RAM so one copy covers a layer's streamed set) to amortize per-copy overhead. Expected gain: low single digits % unless per-copy overhead is currently visible in traces.
- Large BAR / ReBAR mainly lets the CPU map all of VRAM. It does not raise device-initiated H2D bandwidth. That is general knowledge, not sourced in this pass. One possible use: CPU threads write CPU-computed expert outputs straight into VRAM, saving a small copy. The gain is negligible because activations are KB-sized.
- A DirectStorage-like path (NVMe to GPU) is irrelevant: experts already sit in RAM, and NVMe is slower than 13.4 GB/s.

### Gaps
- No measured AMD numbers for SDMA vs blit H2D bandwidth, or for zero-copy kernel read bandwidth, on RDNA3 over PCIe 3.0 were found. A 10-minute local test with rocm-bandwidth-test / hipMemcpyAsync and a custom read kernel would settle it, once the benchmarks are done.

## A3. Compressing experts for transfer (lossless entropy coding, GPU decompress)

### Takeaway
Already-quantized GGUF weights are close to incompressible losslessly. ZipNN found GGUF checkpoints essentially incompressible, and other quantized formats only reached 85-91% of their size. IQ2_XS (lattice-codebook indices plus scales) is likely the worst case. Expected effective-bandwidth gain is at most about 10%, with real GPU decode cost. Not worth it.

### Cited Findings
- A 2026 survey cites ZipNN (Hershcovitch et al., 2025): GGUF checkpoints essentially incompressible, other quantized checkpoints at about 85-91%, BF16 shrinking by about a third — [arXiv 2608.25188](https://arxiv.org/pdf/2608.25188)
- DFloat11 (2025): Huffman-codes BF16 exponents down to about 11 bits per weight (about 30% smaller, bit-exact). It decodes with SRAM lookup tables per transformer block on the GPU. A secondary summary claims up to 24.87x faster than CPU-to-GPU BF16 transfer and 15.12x faster than nvCOMP ANS. NeuZip uses ANS in the same pre-GEMM decode slot. Code: LeanModels/DFloat11 — [arXiv 2504.11651](https://arxiv.org/html/2504.11651v3)

### Inferences
- For transfer compression to pay on the target, the compressed ratio r must satisfy r x 107 us + GPU decode time < 107 us. With r ≈ 0.9 at best, savings are 10 us per expert minus decode time. Prefill decode would contend with the GEMM CUs. Expected gain: about 0-5%, high effort. A lossy alternative (re-quantizing streamed experts lower, for example IQ1-class, only for prefill) changes outputs and needs a separate quality study.

### Gaps
- No direct entropy measurements of IQ2_XS tensors were found. A local zstd test on a GGUF expert tensor would quantify this quickly.

## A4. Prefill: expert streaming pipelines and how others reach high prefill with offloaded MoE

### Takeaway
Offloaded MoE prefill is bounded by bytes streamed per chunk divided by PCIe bandwidth. The main lever is tokens per pass (chunk / ubatch size), along with keeping as many experts resident as possible and full overlap of the stream with compute. Best published consumer-class prefill: about 1,200 tok/s with one RTX 5090 on PCIe 5.0 and more than 1,800 tok/s with two. Strata's 1,500-1,800 tok/s on PCIe 3.0 is already competitive. Remaining gain comes mostly from larger chunks.

### Cited Findings
- arXiv 2606.10493 (2026, unnamed engine), "SLP" stream-loading prefill: loader, model and unloader threads on three CUDA streams synchronized at sub-layer granularity by events. Experts pass through a ring buffer sized to one layer's expert count for short prompts (maximum lookahead), shrinking to 2 (ping-pong) beyond about 50K tokens. Other tensors get a fixed budget of about 1 GB. The paper says short prefills are transfer-dominated and compute dominates beyond about 50K tokens. Results: about 1,200 tok/s on FP8 with one RTX 5090 (32K prompt under 30 s TTFT), and more than 1,800 tok/s with two RTX 5090s on PCIe 5.0 with dual EPYC 9355. It claims 2.8x the estimated kTransformers AMX prefill at 20-32K. Not bit-exact (FP8 with BF16 intermediates) — [arXiv 2606.10493](https://arxiv.org/html/2606.10493)
- kTransformers AMX prefill: more than 500 tok/s at 2K on its Xeon platform (per 2606.10493). A secondary source reports 347 tok/s on Qwen3-235B with a Xeon 4th gen and RTX 4090. The v0.3 preview reached 286 tok/s on DeepSeek R1 with dual Xeon — [LMSYS](https://www.lmsys.org/blog/2025-10-22-KTransformers); [arXiv 2606.10493](https://arxiv.org/html/2606.10493)
- MoE-Gen (2025): accumulates a large batch per expert (average 8,192 tokens/expert, versus about 115-153 for baselines) before loading the expert. It reports 7.7-11x prefill throughput on DeepSeek-class models on an A5000 — [arXiv 2503.09716](https://arxiv.org/html/2503.09716v1)
- Klotski (2025): an expert-aware multi-batch pipeline that fills bubbles when per-expert compute is shorter than I/O, with gains of up to 85x over prior offloading (best case, setting not verified) — [arXiv 2502.06888](https://www.arxiv.org/abs/2502.06888)
- llama.cpp: large `-b/-ub` (for example 2048-4096) amortizes streaming. Reported prefill on Qwen3.8-Flash-Next ranges from about 89 tok/s (5060 Ti) to about 1,300 tok/s (RTX 5090 with DDR5, Gerganov preset) — [Developer's Digest](https://www.developersdigest.tech/blog/llama-cpp-moe-offload-n-cpu-moe-vs-moe-cache-2026); [HF guide](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide)

### Inferences
- Streaming cost on the target: 48 x 512 x 1.437 MB ≈ 35.3 GB of experts. A full pass at 13.4 GB/s takes about 2.63 s. If R GB stay resident in VRAM, a pass is (35.3 - R)/13.4 s. Observed 1,800 tok/s implies about 4,700 tokens per full-stream pass, or proportionally fewer if part of the experts stay resident.
- Prefill throughput is roughly chunk_tokens / max(stream_time, compute_time). Doubling the chunk from about 4-5K to 8-10K tokens should come close to doubling prefill until compute-bound. Compute ceiling estimate: an assumed ~5-6B active params gives about 12 GFLOP/token; at about 40 TFLOPS effective FP16/WMMA on the 7900 XT that is about 3,000+ tok/s. This is my estimate; active params were not given. Costs: activation and intermediate VRAM for the larger chunk, and less VRAM for KV/experts. For 12-32K prompts, one or two passes over the whole prompt (chunk = prompt length, as MoE-Gen does) would be ideal if activation memory allows.
- Skip experts that no token in the chunk routes to. With 512 experts top-10 and a 4K chunk, almost all experts are hit (expected unused fraction ≈ (1-10/512)^4096 ≈ 0), so there is no gain at large chunks. It only matters for short tail chunks.
- Resident experts: every GB held resident during prefill saves about 75 ms per pass. Making the decode cache double as the prefill-resident set (no eviction during prefill) helps both.
- Avalon's 1,771-1,805 tok/s on the same card is the same PCIe-bound regime. Expected further gain: about 1.3-1.8x at 16-32K from larger chunks if memory allows, and about 0 from transfer-mechanism tweaks.

### Gaps
- Avalon fork internals were not found publicly in this pass.
- I did not get numbers on how chunk size trades against activation VRAM for this model.

## B1. KV cache compression for hybrid models (int8/int4/mixed windows) to free VRAM for experts

### Takeaway
In a hybrid GDN plus QSA model only the QSA layers hold growing KV, so KV is already a small share of VRAM. Moving older KV from int8 to about 4-6 bit (a recent window kept at 8-bit, the KIVI/QServe-style pattern) roughly halves KV bytes, both resident and streamed. The VRAM freed for experts is modest. The bigger payoff is in streamed-KV bandwidth beyond the 32K resident cells.

### Cited Findings
- LServe (MIT Han lab, 2025): combining block-sparse attention with KV quantization in one kernel gives additive decode gains. Reported speedups versus vLLM: prefill up to 2.9x, decode 1.3-2.1x, up to 512K context. Code: mit-han-lab/omniserve — [arXiv 2502.14866](https://arxiv.org/html/2502.14866v2)
- ShadowKV: values move to CPU memory, keys are kept as low-rank pre-RoPE projections on the GPU, and selected keys are rebuilt while the selected values are fetched (overlapped on separate streams). With a 1.56% sparse budget it reports up to 6x batch and 3.04x throughput on A100, and it beats Quest at equal budget — [arXiv 2410.21465](https://arxiv.org/html/2410.21465v2)

### Inferences
- VRAM freed: 32,768 cells x (number of QSA layers) x 2 x kv_heads x head_dim x 1 byte. Halving to 4-bit frees half of that. Example with assumed shapes (12 attention layers, 2 KV heads x 256 dims): 32,768 x 12 x 1,024 B ≈ 0.4 GB at int8, saving about 0.2 GB, about 140 experts, under 1% of 24,576. Exact numbers need the real shapes. Decode gain from extra cached experts: small, about 1-2%.
- Better use: keep int8 resident and quantize the streamed (older than 32K) KV to 4-bit to halve PCIe bytes at 64-131K contexts. Or, with a 4-bit resident KV, the 32,768-cell window doubles to 65K and avoids streaming entirely up to that length.
- Exactness: lossy. KIVI-style per-channel keys and per-token values at 2-4 bit are usually reported near-lossless, but this was not verified in this pass.

### Gaps
- The cutlerbenjamin1-cmd/qwen38-megakernel repo fetch timed out, so its mixed-window scheme (last N tokens 8-bit, older 5-6 bit) is unverified.
- KIVI, QServe and SJ-KVaRN primary sources were not retrieved this pass.

## B2. Sparse / top-k attention decode at long context, and KV offload/streaming policies

### Takeaway
QSA already does top-k selection. The literature's main lesson is to keep selection cheap (page-level metadata like Quest's min/max keys, or a constant number of pages as in LServe) and to stream only the selected pages from RAM, not the whole old KV. Streaming dense old KV over PCIe at 13.4 GB/s would dominate decode at 64-131K. Fetching only the top-k pages (ShadowKV pattern) makes cost independent of context length.

### Cited Findings
- Quest (ICML 2024, older): stores per-page min/max keys and uses the query to upper-bound page criticality. Only the top-K pages are loaded. Its motivation is that long-context decode slowdown is caused by reading the KV cache. Reported speedups vary between sources: up to 2.23x end-to-end and up to 7.03x self-attention, see the conflict note in the search summaries — [HF papers 2406.10774](https://huggingface.co/papers/2406.10774); [Moonlight review](https://www.themoonlight.io/en/review/quest-query-aware-sparsity-for-efficient-long-context-llm-inference)
- LServe: a constant number of KV pages preserves long-context ability regardless of context length. It uses hierarchical page selection and DuoAttention-style streaming heads (sinks plus a local window) for most heads — [arXiv 2502.14866](https://arxiv.org/html/2502.14866v1)
- DuoAttention: a full KV is needed only for retrieval heads (under 5% of heads in the studied models). Streaming heads keep a fixed sink plus recent window — [LServe paper, background section](https://arxiv.org/html/2502.14866v2)
- ShadowKV: naive sparse attention with KV on the CPU is slow because of fetch latency. It hides this by overlapping key reconstruction with value fetch — [arXiv 2410.21465](https://arxiv.org/html/2410.21465v1)
- 2606.10493: decode for Q4_K_M DeepSeek-R1 drops from 28 tok/s (1-8K) to 19 tok/s at 128K. kTransformers drops from 22 to 15 tok/s on the same task — [arXiv 2606.10493](https://arxiv.org/html/2606.10493)

### Inferences
- For contexts above 32K: keep page-level selection metadata (min/max or a low-rank key summary) for all pages in VRAM. It is small. Fetch only the selected pages from pinned RAM, issued one layer ahead (the selection for layer L+1 can't be known early, but the query for layer L can trigger the fetch while GDN layers run). With a top-k budget of B tokens per QSA layer, bytes per token are about B x kv_bytes x QSA_layers, independent of context.
- Expected gain: depends on how Strata streams today. If it streams the full older KV, page-selective fetch could make decode at 64-131K near flat. If it already fetches only top-k, gains come from page-granular, coalesced copies (fewer small DMAs).

### Gaps
- No public details on "QSA" were found. MagicPIG was not retrieved.

## B3. Decode slowdown after long prompts (112 -> 90 tok/s after 12K) in hybrid GDN plus attention models

### Takeaway
At 12K all KV is resident, so streaming is not the cause. The likely causes are (1) the QSA top-k selection or indexer scanning all keys, which is linear in context per QSA layer, (2) attention reads of int8 KV, and (3) expert-cache disturbance from prefill. GDN state is constant-size and cannot cause it. Each needs a profile split before acting.

### Cited Findings
- Quest identifies KV reading during self-attention as the main long-context decode slowdown — [HF papers 2406.10774](https://huggingface.co/papers/2406.10774)
- LServe: a constant page budget plus hierarchical selection makes selection cost sublinear — [arXiv 2502.14866](https://arxiv.org/html/2502.14866v1)
- Published offload engines also slow down with context: 28 to 19 tok/s from 1K to 128K for one engine, and 22 to 15 for kTransformers — [arXiv 2606.10493](https://arxiv.org/html/2606.10493)

### Inferences
- Slowdown budget: 1/90 - 1/112 = 2.18 ms extra per token. If it comes from KV reads, 2.18 ms at about 600 GB/s effective VRAM bandwidth is about 1.3 GB per token, far more than the resident int8 KV at 12K. So raw KV reads are unlikely to be the main cost. Selection-kernel inefficiency (many small launches, a full-context top-k sort per head) or cache disturbance is more likely. This is an inference from the arithmetic; verify with rocprof per-kernel timing at 0 vs 12K.
- Candidate fixes:
  - (a) Page-level (Quest min/max) or two-level selection: score pages and only then tokens. This cuts the scan by about the page size (16-64x).
  - (b) A radix or partial top-k instead of a full sort.
  - (c) Make the adaptive expert tier resistant to prefill pollution: don't let prefill routing statistics evict the decode working set, or re-warm right after prefill.
  - (d) Fuse selection plus attention into one kernel per layer.
- Expected gain: up to recovering most of the 20% gap (90 to about 105-110 tok/s at 12K) if selection overhead dominates. Exact (no accuracy change) for (b), (c) and (d). (a) changes which tokens are attended unless it is used only as a conservative upper-bound filter.

### Gaps
- No public profile of Qwen3.8-Flash-Next / QSA decode cost vs context was found. The Gated DeltaNet decode cost is assumed constant per token from the architecture; no external measurement was retrieved.
