# Distributing MoE decode across slow consumer devices: pipeline vs expert vs tensor parallelism

Conventions. [M] = measurement from a cited source. [D] = my derivation/arithmetic from cited numbers. [E] = estimate/assumption (no source measured it). Date of research: 2026-11-10. About 15 tool calls were used, and several papers were only read via abstracts or fetched summaries (flagged in Gaps).

Model constants used below
- Qwen3-30B-A3B: hidden H=2048, **48 layers (not 40)**, 128 experts, top-8, moe_intermediate 768, 4 KV heads, head_dim 128 [M] — [HF config](https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json)
- Qwen3.6-35B-A3B: H=2048, 40 layers (10 x [3 Gated-DeltaNet+MoE, 1 Gated-Attention+MoE], so only 10 full-attention layers), 256 experts, 8 routed + 1 shared, expert dim 512 [M, secondary summaries of the model card] — [HF mirror](https://huggingface.co/Wudiam/Qwen3.6-35B-A3B), [architecture overview](https://huggingface.co/blog/EXDai/qwen36-35b-a3b-architecture-overview)
- Derived [D]: expert = 3·H·I params. Qwen3-30B: 3·2048·768 = 4.72 M params (about 2.65 MB at ~4.5 bpw). Qwen3.6: 3·2048·512 = 3.15 M params (about 1.77 MB at Q4). Active experts per token: 48·8·4.72 M = 1.81 B params (Qwen3-30B). Attention (about 0.9 B) plus embeddings and lm_head (about 0.3 B) are about 1.2 B and must live on whichever device runs them, so they are not parallelisable across experts.
- Size at quant [D]: 35 B x 4.85 bpw / 8 = about 21 GB (Q4_K_M-class); 35 B x 2.7 bpw / 8 = about 12 GB (Q2-class). Bytes read per token about 3 B x 0.6 B/param = about 1.8 GB (Q4), about 1.0 GB (Q2).

---

## 1. Bytes per token and latency floor: pipeline vs expert-parallel vs tensor-parallel vs hybrid

### Takeaway
Batch-1 MoE decode is latency-bound, not bandwidth-bound, on any network. Pipeline split needs only D one-way hops per token (KB-scale), so it is nearly free on LAN/Wi-Fi and tolerable on WAN (about 4 hops x one-way delay). Expert-parallel needs about L sequential round trips per token (48 or 40), which is fine on 1 GbE (about 10-25 ms) but ruinous on Wi-Fi (0.1-0.5 s) and impossible on WAN (1-5 s). Tensor parallel needs 2L all-reduces and is the worst of the three.

### Cited Findings
- prima.cpp (pipeline "piped-ring" over a home Wi-Fi cluster, 320-610 Mbps, 3-7 ms link latency): each device sends one hidden-state vector per layer-window, in FP32 (4e bytes); dllama-style tensor parallelism needs about 64 all-reduces/token for 8B and 160 for 70B, which the authors estimate adds about 24 s/token at 70B over such Wi-Fi — [prima.cpp arXiv v2](https://arxiv.org/html/2504.08791v2)
- prima.cpp measured TPOT on a 4-device heterogeneous home cluster (Wi-Fi): 30B 72 ms, 45B 233 ms, 70B 674 ms. Baselines on the same cluster: exo 8B 263 ms (llama.cpp 15 ms), dllama 8B 459 ms. exo/dllama are 5-8x slower than prima.cpp. These are dense models, no MoE — [prima.cpp arXiv v2](https://arxiv.org/html/2504.08791v2)
- OD-MoE (expert-parallel decode on 8 worker RTX 3090 nodes + main + shadow node, **1 Gbps Ethernet**, Mixtral-8x7B top-2, 32 layers, batch 1): 3.69 tok/s vs 4.89 tok/s for the all-experts-cached single-server Transformers baseline (75.5%). Main-to-worker traffic is about 16 KB/token/layer (about 512 KB/token over 32 layers; the 512 KB is the fetcher's own derivation). The authors call comms delay "tiny" but do not quantify it — [OD-MoE HTML](https://arxiv.org/html/2512.03927), [abstract](https://arxiv.org/abs/2512.03927)
- Another paper argues expert parallelism across multi-node systems gives inefficient expert distribution for single-batch decode and proposes tensor parallelism on a workstation (GPU-NDP) — surfaced via search snippet only; see [search hit list](https://arxiv.org/pdf/2512.03927) context and [DATE 2026 paper](https://past.date-conference.com/proceedings-archive/2026/DATA/274.pdf) (not opened).
- A separate system notes that concurrent decode requests, even at small batch sizes, degrade per-request TPOT on MoE because expert activations effectively multiply — search snippet from [OSDI'26 paper](https://www.usenix.org/system/files/osdi26-wang-wenxin.pdf) (not opened).

### Inferences (formulas and worked numbers; all [D] unless stated)

Hidden-vector sizes at H=2048: fp32 8192 B; fp16/bf16 4096 B; int8 2048 B (+scales); int4 1024 B.

(a) Layer-pipeline, D devices in a ring (head holds embeddings and lm_head, so the last hop returns to the head):
- hops/token = D; bytes/token = D · H · b (b = bytes/element).
- D=4, fp16: 4 x 4096 = **16 KB/token** (same for Qwen3 48 L and Qwen3.6 40 L; it does not depend on L). fp32 (prima.cpp default): 32 KB.
- T_token = T_compute_total + D · (RTT/2 + ser + sw), where ser = bytes·8/BW and sw = software stack overhead per hop (about 0.05-0.3 ms [E]).
- ser at 1 GbE: 4096·8/1e9 = 33 µs. At 50 Mbps WAN: 0.66 ms. Neither matters next to latency.
- Network floor (compute = 0), D=4: 1 GbE (RTT 0.2-0.5 ms): 4·(0.1-0.25) = 0.4-1.0 ms, so over 1000 tok/s ceiling. Wi-Fi 5/6 (RTT 2-10 ms): 4·(1-5) = 4-20 ms, so 50-250 tok/s ceiling. WAN (RTT 20-100 ms): 4·(10-50) = 40-200 ms, so **5-25 tok/s ceiling**, before compute.
- Compute is NOT reduced by pipelining a single stream: devices work one after another, so T_compute_total equals single-device time. Sharding only buys memory capacity.

(b) Expert-parallel (attention+router+shared on a head device, experts partitioned over D devices; per layer scatter x, then gather partial sums):
- Distinct devices hit per layer with top-k=8 and balanced random placement: D·(1-(1-1/D)^8). D=4: 4·(1-0.75^8) = 4·0.900 = **3.60**; D=3: 3·(1-0.667^8)=2.92; D=2: 1.99.
- Bytes/layer = 2 · 3.6 · 4096 = 29.5 KB (fp16). **Qwen3-30B (48 L): 1.42 MB/token. Qwen3.6 (40 L): 1.18 MB/token.** About 70-90x the pipeline traffic, still tiny vs 1 GbE (about 11 ms of pure serialisation at the head NIC: 14.7 KB out + 14.7 KB in per layer = 0.24 ms, x48 = 11.3 ms).
- Latency: L sequential round trips: T_net = L·(RTT + ser_layer + sw). 1 GbE (RTT 0.2-0.5 ms + 0.24 ms ser): 48·(0.44-0.74) = **21-36 ms**. Wi-Fi: 48·(2-10) = **96-480 ms**. WAN: 48·(20-100) = **0.96-4.8 s/token**. For Qwen3.6, 40 layers: 18-30 ms (LAN), 80-400 ms (Wi-Fi), 0.8-4 s (WAN).
- Cross-check against measurement [D from M]: OD-MoE, 32 layers, 1 GbE: (1/3.69 - 1/4.89) = 271 ms - 204 ms = 66 ms extra per token, or about 2.1 ms/layer. That is an upper bound for comms (includes prediction, loading stalls and Python/torch overhead) and is 3-4x my floor of about 0.5-0.7 ms/layer, which fits "floor vs real" expectations.
- Compute side: expert FLOPs/bytes split across about 3.6 devices; the head keeps about 1.2 B params (about 0.68 GB Q4) serial. Example, effective 15 GB/s per old PC [E]: head 45 ms + experts about 1.0 GB / 3.6 hit devices, worst device holds about 3 experts, so about 20 ms, so about 65 ms compute versus 124 ms single device. Adding 21-36 ms LAN comms gives about 90-100 ms/token (about 10-11 tok/s), only about 25% better than a lone device that could hold everything, so EP is a memory-fit tool at 1 GbE, not a speed tool.
- EP memory advantage: each device holds 1/D of the experts (about 5 GB of a 21 GB Q4 model at D=4) plus the head holds attention/embeddings (about 0.7 GB Q4), which works for 8 GB-RAM boxes.

(c) Tensor-parallel (attention heads + FFN sharded): 2 all-reduces per layer, so 96/token (48 L) or 80/token (40 L), each at least one RTT in a star topology (more in ring all-reduce: 2(D-1) steps). Network floor: 1 GbE 96·(0.2-0.5) = **19-48 ms**; Wi-Fi 96·(2-10) = **0.2-1.0 s**; WAN 96·(20-100) = **2-10 s**. Bytes small (96 x 4 KB x about (D-1)). With a 3B-active MoE, the compute per token is already tiny, so TP gains nothing: fixed latency dominates. Qwen3 has only 4 KV heads and 128 experts, which limits TP degree for attention to 4. Measured: exo claims 1.8x on 2 devices / 3.2x on 4 devices for TP, but that is on RDMA/Thunderbolt-5 Macs with dense or big models, not weak LAN boxes — [exo README mirror via search](https://noqta.tn/en/blog/exo-labs-local-ai-cluster-frontier-models-guide-2026) (secondary, project's own claim); Qwen3-235B-A22B 19.5 to 31.9 tok/s on 1 to 4 Macs over TB5 RDMA — same secondary source. RDMA-class latency (tens of µs) is exactly what makes TP/EP viable, and 1 GbE TCP does not offer it.

(d) Hybrid designs (what makes sense on weak hardware):
- H1 "pipeline across sites, EP within a LAN group": inter-group traffic 4 KB x (#groups) per token; intra-group EP only in LAN. Floor = (#groups)·(WAN RTT/2) + sum over layers (RTT_LAN + ser). Example, 2 groups over WAN: 2·(10-50) = 20-100 ms plus 48·0.5 ms (about 24 ms) of intra-group EP.
- H2 "head holds attention/router + remote experts with predictive prefetch" (OD-MoE style): moves expert weights, not activations; needs a good predictor. OD-MoE reports 99.94% activation prediction accuracy with a shadow model (abstract). HOBBIT reports about 96% next-layer top-1 prediction, about 90% for 2-3 layers ahead — [HOBBIT](https://arxiv.org/html/2411.01433v2). Prediction lets you issue the dispatch before the layer finishes, which can overlap RTT with compute; this is the only way I found to mitigate EP's per-layer round trip. Gains are unquantified on LAN/Wi-Fi.
- H3 "pipeline + replicas": see section 3/4.

### Gaps
- No measured EP or TP numbers for Qwen3-30B-A3B / Qwen3.6-35B-A3B over 1 GbE, Wi-Fi or WAN found.
- Wi-Fi 5/6 RTT, 1 GbE RTT, and WAN RTT are from the task brief, not independently sourced by me; stack overhead per hop is my assumption.
- Real routing is skewed/correlated (not uniform random), so the 3.6 devices-hit figure is an upper-ish estimate; placement by co-activation can reduce it.

---

## 2. What papers and projects report

### Takeaway
Offloading papers (Mixtral-offloading, MoE-Infinity, HOBBIT, ProMoE, AdapMoE, Fiddler) fight the PCIe/SSD bandwidth wall on a single machine and report mostly relative speedups. OD-MoE is the only item found that does network-distributed decode with a cacheless expert-sharded design at 1 GbE: it reaches 75% of cached speed. No source reports measured MoE-over-WAN decode at the layer granularity; WAN-aware systems (BloomBee, SpecEdge, WANSpec) use pipeline plus speculation.

### Cited Findings
- OD-MoE (1 GbE, Mixtral-8x7B, batch 1, tok/s): Transformers all-cached 4.89; OD-MoE 3.69; AdapMoE 3.13; Mixtral-Offloading 2.24; HOBBIT 0.79; llama.cpp CPU 0.82; MoE-Infinity 0.69. About 1/3 of the GPU memory of the cached deployment; under 1 GB GPU per worker — [OD-MoE](https://arxiv.org/html/2512.03927). Note the baselines ran on one RTX 3090 and offload over PCIe, so this compares network sharding to PCIe offload on very strong GPUs; caveat for weak devices.
- HOBBIT (llama.cpp-based, RTX 4090 with PCIe 4 and Jetson AGX Orin): loading one Mixtral layer over PCIe 4.0 takes about 80 ms vs about 3 ms compute (expert loading is 85.5% of time on 4090, 94.5% on Orin); mixed-precision substitution (fp16 to int4, int8 to int2) speeds loading up to 4x; up to 9.93x decode speedup vs MoE-Infinity (Phi-MoE, Orin); 2.30-3.92x vs MoE-Infinity and 3.21-3.29x vs MoE-Offloading on 4090. Only relative speedups reported — [HOBBIT](https://arxiv.org/html/2411.01433v2)
- ProMoE (proactive caching, integrated in transformers and llama.cpp): avg decode speedup 1.34x (up to 1.79x) vs handcrafted caching, 2.07x (up to 5.02x) vs open-source offloading — [ProMoE](https://arxiv.org/html/2410.22134v2)
- FineMoE (EuroSys '26) compares to ProMoE, MoE-Infinity, Mixtral-offload, DeepSpeed (no throughput numbers retrieved) — [FineMoE](https://arxiv.org/pdf/2502.05370v2)
- Petals (pipeline-over-internet, BLOOM-176B on consumer GPUs): about 1 step/s reported in the abstract; bandwidth/latency conditions not given in what I read — [Petals arXiv](https://arxiv.org/abs/2209.01188). I found no Mixtral-on-Petals measurements.
- BloomBee (Apr 2026 preprint): decentralized consumer-GPU pipeline over the internet combining layer assignment, KV offload, micro-batch pipelining, activation compression and speculative decoding; reports speculative decoding can hurt over WAN unless draft candidates are pruned (more candidates means more traffic) — [summary](https://www.emergentmind.com/papers/2604.21072) (secondary)
- SpecEdge (NeurIPS 2025): consumer edge GPU drafts, server verifies, only token IDs cross the network; 1.91x cost efficiency, 2.22x server throughput, 11.24% lower inter-token latency vs server-only — [SpecEdge](https://arxiv.org/abs/2505.17052v2)
- WANSpec (Feb 2026): speculative decoding's overlap lets a pipeline absorb WAN latency — [WANSpec](https://arxiv.org/html/2602.18931v1) (abstract-level)
- prima.cpp with speculative decoding: 70B 674 to 442 ms/token; 32B at 26 tok/s [M] — [prima.cpp](https://arxiv.org/html/2504.08791v2)
- Not found/not verified: Fiddler, DAOP, PowerInfer absolute numbers; Megablocks/DeepSpeed-MoE are datacenter systems (all-to-all over NVLink/IB), not applicable at consumer-LAN scale and not researched here.

### Inferences
- The per-layer-sync design (EP) is only viable when RTT is under about 1 ms; every successful "distributed MoE" number found is at 1 GbE or faster (OD-MoE) or uses RDMA (exo on TB5).
- Wire quantisation: hop payloads at batch 1 are 1-8 KB, so serialisation (33 µs at 1 GbE for fp16; 0.66 ms at 50 Mbps) is negligible against RTT. Quantising activations to int8/int4 therefore buys almost nothing at decode; it matters only for prefill (e.g. 1000-token prompt x 4 KB = 4 MB per hop: 33 ms at 1 GbE, 0.11 s at 300 Mbps Wi-Fi, 0.64 s at 50 Mbps WAN [D]). Residual-stream activations in LLMs often have outlier channels, so fp16/bf16 is the safe default; int8 would need a test. This is my prior, not sourced here.
- Expert prefetch/co-activation predictors help in two ways: (1) hide weight transfers (HOBBIT/ProMoE/OD-MoE), (2) enable early dispatch in EP. In a cluster where each device already holds its experts permanently in RAM, the predictor is only needed for (2).

### Gaps
- No full-text read of ProMoE, AdapMoE, Fiddler, DAOP, PowerInfer, MoE-Infinity; absolute tok/s only from OD-MoE.
- OD-MoE hardware is much stronger than "weak consumer" (RTX 3090 workers, R7-7700, 192 GB DRAM).
- No measured source for activation quantisation of MoE hidden states on the wire.

---

## 3. Pipeline decode throughput scaling: single stream vs concurrent streams

### Takeaway
A single stream gets no speedup from more devices (it gets slightly slower, +D·hop latency), and more devices only enable bigger models. With M concurrent streams you can overlap the stages, up to about D x aggregate throughput, but MoE expert reads grow quickly with batch size, which erodes the benefit on bandwidth-bound weak devices.

### Cited Findings
- prima.cpp: for models that fit on one device, it degrades to llama.cpp speed (8B: 15 ms on both; 14B 20 ms on both) — [prima.cpp](https://arxiv.org/html/2504.08791v2). Pipeline adds no speed when memory is not the bottleneck; its big wins (70B: 674 ms vs llama.cpp 10,120 ms) come from avoiding swap/mmap thrashing and from overlapping disk I/O.
- exo 8B: 263 ms vs llama.cpp 15 ms in the same testbed (implementation overhead, not physics) — [prima.cpp](https://arxiv.org/html/2504.08791v2)
- Petals pipelines many clients concurrently over volunteer servers; the paper's number is about 1 step/s per BLOOM-176B client — [Petals](https://arxiv.org/abs/2209.01188).
- SpecEdge and BloomBee interleave multiple requests/micro-batches to hide WAN latency — see section 2 sources.

### Inferences (formulas; [D]/[E])
- Single stream: tok/s = 1 / (T_c + D·h), T_c = whole-model compute time on one device (sum over stages), h = per-hop cost (RTT/2 + ser + sw).
- M in-flight streams, stage time t_s = T_c/D: steady-state aggregate = min(M / (T_c + D·h), 1 / max(t_s, h)), at most D / T_c. Needs M ≥ about 1 + D·h/t_s streams to saturate.
- Worked, Q4 35B-A3B, T_c = 124 ms [E: 1.8 GB / 15 GB/s ≈ 120 ms + 4 ms], D=4:
  - 1 GbE (h ≈ 0.3 ms): single = 1/(0.124+0.0012) = **8.0 tok/s**; ceiling D/T_c = 32 tok/s with M≥4 (ignoring batch expert growth).
  - Wi-Fi (h ≈ 3 ms): single = 1/(0.124+0.012) = 7.4 tok/s (range 6.5-7.8 for h=1-5 ms).
  - WAN (h = 10-50 ms): single = 1/(0.124+0.04..0.2) = **3.1-6.1 tok/s**. With M=4: aggregate 12-24 tok/s but each user still sees 3-6 tok/s. M must be ≥ 1 + 4·h/31 ms (about 2.3-7.4) to saturate.
- Batch cost for MoE (uniform-random routing assumption, [E]): distinct experts touched per layer = N·(1-(1-k/N)^b). Qwen3-30B (N=128, k=8): b=1: 8; b=2: 15.5; b=4: **28.9**; b=8: 51.5; b=16: 82. Qwen3.6 (N=256): b=4: 30.5, b=8: 57. So expert bytes read per step grow about 3.6x at b=4 and 6.4x at b=8, vs 4x/8x token output: aggregate gain per device is only about 1.1x at b=4 once the expert term dominates. Real routing is correlated, so reality is somewhat better; OSDI'26 snippet above says it directly: concurrent requests degrade MoE TPOT. Pipeline concurrency (distinct micro-batches at different stages) avoids the union growth since each stage handles one micro-batch at a time, so prefer many single-token micro-batches in flight over big batches on bandwidth-bound devices.
- Speculative decoding on MoE (draft k=4, acceptance 0.7): expected tokens/round = (1-0.7^5)/(1-0.7) = 2.77 [D]. Verifying 5 tokens touches 128·(1-(15/16)^5) = 35 experts/layer (4.4x expert bytes) so, in the single-device compute model, verify ≈ 45 ms (head) + 297 ms (experts: 48·35·2.65 MB = 4.45 GB / 15 GB/s) = about 340 ms [E]. On WAN with h=25 ms: round = 340+100 = 440 ms / 2.77 = 159 ms/token = 6.3 tok/s vs 4.5 tok/s non-speculative (224 ms) — a modest +40% [E]. On 1 GbE: (340+1.2)/2.77 = 123 ms vs 125 ms with no gain. So draft-based speculation is only worth it when RTT is large; this matches BloomBee's caution about WAN candidate traffic. Measured dense analogue: prima.cpp 674 to 442 ms (1.5x).

### Gaps
- No measured multi-stream pipeline scaling for MoE on weak devices.
- Effective bandwidth of "old PCs" is assumed (15 GB/s), not measured by me.

---

## 4. Model routing (one model per device) vs sharding

### Takeaway
If a model fits in a single device's RAM, replicating it (or routing requests among small models) gives about N x aggregate throughput with zero network coupling and is strictly better than pipelining the same model. Sharding is justified only when a model larger than any single device's RAM is required (or quality demands it).

### Cited Findings
- prima.cpp: when the model fits on one device its latency equals llama.cpp's, so sharding gives no speedup there (8B 15 ms vs 15 ms; 14B 20 vs 20) — [prima.cpp](https://arxiv.org/html/2504.08791v2)
- Single-host Qwen3-30B-A3B Q4_K_M on a Strix Halo (strong iGPU, not weak): 86.1 tok/s — [strix-halo-llm-perf](https://github.com/visorcraft/strix-halo-llm-perf) (reference point only).
- llama.cpp RPC splits layers across machines (pipeline) and one community post reports a 122B MoE at about 96 tok/s over two RTX 5090 rigs on 2.5GbE (reddit snapshot, unverified) — [reddit snapshot](https://reddit.sentinel-team.org/posts/1r6mwsd/snapshots/2026-02-17T00%3A30%3A13.01368Z)

### Inferences
- Decision rule [D]: compare per-device RAM R to model size S. If S ≤ R·(about 0.8): replicate/route (aggregate N·v1, no net). Else shard into the fewest devices possible, D_min = ceil(S / (0.8R)), and use leftover devices as replicas of the pipeline (e.g. 8 devices = 2 pipelines of 4). Minimising D minimises both hop latency and exposure to jitter.
- Examples: 35B-A3B Q4 = 21 GB. Devices with 8 GB RAM need D_min = ceil(21/6.4) = 4; with 16 GB: D=2 (or 1 at tight fit); with 32 GB: 1. Q2 = 12 GB: 16 GB boxes hold it alone (single device ≈ 13-22 tok/s [E]).
- Model routing wins when: many concurrent users, requests are heterogeneous (small model suffices for most), device churn/unreliability (a pipeline dies if any stage dies), WAN scattered devices (no per-token coupling). Sharding wins when: you need the single biggest model's quality, memory is the binding constraint, devices share a LAN.
- MoE caveat: a 30B-A3B MoE is already "small compute but big memory". Q2 of it on one 16 GB device is often a better use than a dense 8B pair, but a quality loss at Q2 should be tested (no source retrieved).
- Routing can also use a speculative-decoding style arrangement: small model on a weak device drafts, bigger sharded model verifies (SpecEdge pattern; only token IDs cross the network).

### Gaps
- No source comparing model-routing vs sharding in home clusters quantitatively; reasoning above is mine.
- No benchmark on quality of Q2 35B-A3B vs smaller dense models.

---

## 5. Realistic tok/s: 35B-A3B Q2/Q4 on 3-4 old PCs

### Takeaway
No measurement was found for this exact setup. Using memory-bandwidth arithmetic, expect about 5-10 tok/s at Q4 over 1 GbE with a layer pipeline across 3-4 old PCs (almost the same as one device if it fit), about 8-14 tok/s at Q2 if split, and about 3-5 tok/s over the open internet (single stream). Expert-parallel over 1 GbE might reach 8-11 tok/s at best but it is more complex and fragile; over Wi-Fi or internet it is not viable (about 2-5 tok/s and below 1 tok/s respectively).

### Cited Findings
- Closest measured pipeline data (dense, 4 mixed home devices, Wi-Fi 320-610 Mbps): 30B 72 ms/token (about 14 tok/s), 70B 674 ms/token (about 1.5 tok/s) — [prima.cpp](https://arxiv.org/html/2504.08791v2) (that cluster has a 3070/2080 Ti GPU; not old CPU-only PCs).
- Closest measured MoE-over-1 GbE: OD-MoE 3.69 tok/s Mixtral-8x7B (14 B active) on RTX 3090s — [OD-MoE](https://arxiv.org/html/2512.03927)
- Closest measured Qwen3 MoE over a network: a 235B-A22B on 4 Macs over TB5 RDMA, 31.9 tok/s (19.5 on 1 node) — [noqta summary](https://noqta.tn/en/blog/exo-labs-local-ai-cluster-frontier-models-guide-2026) (secondary).
- Community llama.cpp RPC data are mostly for fast links (2.5-10 GbE, USB4 at 9.4 Gbps) — [DEV post](https://dev.to/ljkunal/distributed-llm-inference-across-nvidia-blackwell-and-apple-silicon-over-10gbe-2feg) (projection of 4-8 tok/s for 200B+ MoE over 10 GbE, flagged as an estimate by its author).

### Inferences (all [E], assuming CPU-only DDR4 desktops, effective 15-25 GB/s, not compute-limited)
- Per-device ceiling: bytes/token / BW. Q4: 1.8 GB / 15-25 GB/s = 72-120 ms = **8-14 tok/s**. Q2: 1.0 GB / 15-25 = 40-67 ms = 15-25 tok/s. Old CPUs (no AVX2, 2-4 cores) may be compute-bound instead and fall to 3-6 tok/s; llama.cpp quantised kernels typically reach 50-70% of the bandwidth ceiling, so apply x0.6-0.8 as a realism factor.
- Pipeline, 1 GbE, D=3-4: add D·h = 4·(0.15-0.5 ms) + overhead (llama.cpp RPC is chattier than the ideal; exo showed 17x overhead vs llama.cpp for implementation reasons) so tok/s about 0.9-0.97 x the per-device figure. Q4: about 5-11 tok/s, midpoint about 7. Q2: about 9-18, midpoint about 12 [E].
- Pipeline, Wi-Fi (h 1-5 ms): minus 3-10% vs LAN, jitter spikes (10 ms+) cause visible stutter but average stays within about 10%: Q4 about 5-9 tok/s.
- Pipeline, internet (h 10-50 ms, D=4): Q4 3.1-6.1 tok/s (section 3 math); Q2 (T_c about 70 ms): 1/(0.07+0.04..0.2) = 3.7-9 tok/s. D=3 improves slightly. A speculative draft adds about +40% only at WAN RTTs [E].
- EP over 1 GbE: about 90-100 ms/token Q4 (about 10-11 tok/s) per section 1 [E]; much more sensitive to jitter (48 syncs/token): one 10 ms stall per layer in 10% of layers adds about 48 ms.
- Multiple users: with 3-4 concurrent users the pipeline's aggregate can approach 3-4x single-stream (up to about 25-30 tok/s Q4 aggregate on LAN), limited by expert-union growth (section 3).

### Gaps
- No direct measurement for Qwen3.6-35B-A3B or Qwen3-30B-A3B distributed across several weak PCs at any link speed. All numbers in this section are estimates.
- Old-PC memory bandwidth and CPU compute throughput were not measured; recommend a 30-minute calibration run (llama-bench on one box, iperf3/ping between boxes) before committing to a design.
- Wi-Fi/WAN jitter distributions were not found in sources; mean-latency math understates tail effects.
