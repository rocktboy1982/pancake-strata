# Whole-pipeline scheduling and cost models for heterogeneous MoE inference (CPU / PCIe / GPU / NVMe), Nov 2022 - Nov 2026

Scope: hardware-aware cost models, rooflines for hybrid MoE decode, cross-device pipelining, overlap scheduling, draft-driven lookahead, online auto-tuning, simulators and contention models, critical-path tools. Excludes repeats already covered in the series (Fiddler, KTransformers deferral, FreeToken, ProMoE, MoE-Lightning HRM, FlexGen LP, MoE-Infinity, HOBBIT, Fate, PreScope only as a placement idea, Harmonia/Athena, Baleen, SP-MoE). Where an already-covered paper matters for the simulator, this is flagged.

Pancake baseline used in every verdict (from /home/dan/reports, not re-measured): edit window 37.4 ms = GPU 15.8 + CPU experts 17.4 + draft 2.5 (357 CPU experts/window, ~48.8 us each, ~27 of 33 GB/s DDR4); short window 24.8 ms = 15.9 + 5.2 + 2.2; 38.3% of GPU kernel time is `combine_cpu_k10_vec4` spinning on CPU rows; PCIe 3.0 promotion ~115 us vs ~42-49 us CPU compute, so `--pcie-frac 0` wins; floor of the edit window with perfect overlap ~20 ms (grouping report, F1-F3).

Repo facts read for this note (read-only, no builds or runs):
- `/home/dan/clone-of-d/src/Strata/tools/pancake/time_ceiling.py` (item 14): replays a `--dump-routing` trace; per window computes `window = max(gpu_ms + cpu + alpha*dma, dma)`, with `cpu = distinct_misses * c_miss` calibrated so the engine tier reproduces `--cpu-ms` (5.2 / 17.4), `dma = promotions * 1.4 MB / 12 GB/s`, contention as `alpha` in {0, 0.25, 0.5} "CPU ms lost per ms of DMA". It is a serial-additive model: GPU and CPU time are summed, never overlapped. Policies: tier, LRU, Belady, Belady-with-bypass horizon, online k-th miss.
- `/home/dan/clone-of-d/src/Strata/src/spec/draft_policy.cpp` (338 lines): per context bucket and per window size t it keeps an EMA of measured round time (`kCostAlpha 0.1`), falls back to a fixed shape table `kShape[t]` for unmeasured sizes, EMA of MTP acceptance, a lookup acceptance table with priors, forced probes (`kProbes 3`) and stale re-measurement (`kStaleRounds 300`); `choose()` maximises expected accepted tokens / `cost_ms(t)`; `seed_costs()` can pre-load a cost table per bucket. This is already a greedy contextual-bandit-like controller with a measured cost model.

## Q1. Hardware-aware cost models and rooflines that decide per expert / layer / token where to compute and what to move

### Takeaway
The 2025-2026 hybrid-MoE schedulers (HybriMoE, PreScope, DALI) all use the same small cost model: CPU time linear in tokens routed to the expert (beta*m + C), PCIe time a constant per expert, GPU time ~constant, objective = minimise max(T_cpu, T_gpu) per layer, solved greedily online. For pancake at batch 1 with windows of at most 5 tokens this model reproduces what pancake already found (`--pcie-frac 0`): the crossover where PCIe beats the CPU needs many tokens per expert, so the direct gain is small; the real value is the max() form, which the current simulator lacks.

### Cited Findings
- HybriMoE (Peking University, Meng Li's group; DAC 2025, lead): dynamic intra-layer CPU/GPU scheduling, impact-driven inter-layer prefetching, score-based caching; 1.33x prefill and 1.70x decode on average over a state-of-the-art hybrid framework (built on kTransformers); code at github.com/PKU-SEC-Lab/HybriMoE — [arXiv 2504.05897](https://arxiv.org/abs/2504.05897). A third-party summary says it simulates candidate schedules before execution by filling CPU-compute, GPU-compute and transfer timelines — [search summary of HybriMoE review](https://www.themoonlight.io/review/hybrimoe-hybrid-cpu-gpu-scheduling-and-cache-management-for-efficient-moe-inference) (not confirmed on the arXiv abstract page).
- DALI (Institute of Automation CAS + UCAS + SJTU, lead CAS; arXiv Feb 2026, no venue): per-expert 0-1 integer program, T_cpu = sum t_cpu(w_i)*C_i, T_gpu = sum max(Trans, compute(w_i))*G_i with Trans a fixed per-expert PCIe time when w_i>0, objective min max(T_gpu, T_cpu), GPU-memory constraint; exact solving costs ~55% of inference time, so a greedy (sort experts by |t_gpu - t_cpu|, place each where cumulative latency is lower) reaches ~92% of optimal at ~4.5% overhead; timings from warm-up profiling; also residual-based next-layer prefetch and a windowed workload-score cache swap (w_size/u_size e.g. 4/8); RTX 3090, EPYC 7532, DDR4, PCIe 4.0; decode 3.97x vs llama.cpp, 2.16x vs KTransformers, 1.48x vs MoE-Lightning, 1.32x vs HybriMoE (Mixtral, Qwen3-30B-A3B, DeepSeek-V2-Lite) — [arXiv 2602.03495](https://arxiv.org/html/2602.03495v1).
- PreScope (NUDT + Tsinghua; arXiv Sep 2025; already in the series as a prefetch idea, listed here for its cost model): CPU time t_c = beta*m + C, PCIe a serial queue of fixed t_io, GPU fixed t_g; an expert goes to the GPU queue if the cross-layer GPU finish time (including earlier prefetch delay alpha) is below the CPU finish time; prefetch count f = (T_gap + t_attn)/t_io from the CPU gap; O(n) scheduler; V100 on PCIe 3.0 and A100 on PCIe 4.0; up to 70.8% throughput over HybriMoE and 141% over Klotski at batch 4-64 — [arXiv 2509.23638](https://arxiv.org/html/2509.23638v1).
- MoE-Lens (University of Michigan, lead; AMD-funded; arXiv Apr 2025): two-stage analytic model for CPU-offloaded MoE serving; stage 1 bounds throughput by min(PME*M/delta, T_GPU) with delta = model size / I/O bandwidth and MoE GEMM intensity ~ n*N_k/N_e; stage 2 adds bounded batch, paged KV and prefill/decode overlap; a profiler fits GPU time vs token count to find the token count where GEMM becomes the bottleneck; predicts throughput with ~94% average accuracy; 4.6x average (up to 25.5x) over MoE-Lightning; throughput-oriented (offline batches of tens of thousands of requests), not latency — [arXiv 2504.09345](https://arxiv.org/html/2504.09345v1).
- MoE-Lens reports its model overestimates for Mixtral with a 210 GB KV cache, which the authors attribute to CPU attention contending with weight transfers — [arXiv 2504.09345](https://arxiv.org/html/2504.09345v1).
- LIA (UIUC Nam Sung Kim group with Google DeepMind co-author Yazdanbakhsh; ISCA 2025, lead UIUC): systematic sublayer offload to CPU using Intel AMX plus a CXL+DDR memory policy; up to 5.1x lower latency and 3.7x higher throughput vs the latest single-GPU offloading framework on Sapphire Rapids + H100; artifact on GitHub — [dblp ISCA 2025](https://dblp1.uni-trier.de/rec/conf/isca/KimWX0YK25.html); [artifact mirror](https://kelaenderkoe.pages.dev/Using-https-github.com/MachineLearningSystem/25ISCA-LIA_AMXGPU).
- MoE-SpeQ defines an "Amortization Roofline Model": throughput (accepted tokens per cycle) vs accepted work per byte of synchronous I/O, with the I/O roof slope = PCIe bandwidth and a compute roof that depends on draft length k — [arXiv 2511.14102](https://arxiv.org/html/2511.14102v1).

### Inferences
- Roofline for pancake's edit window (three roofs, built from the numbers above): GPU roof 15.8 ms (+2.5 draft), CPU roof = CPU expert bytes / ~27 GB/s = 17.4 ms, PCIe roof ~0 at `--pcie-frac 0`. The bound with perfect overlap is max(18.3, 17.4) + sync ~ 20 ms; measured 37.4 ms is almost exactly the sum. So pancake today sits on the "sum" line, not the "max" line: about 17 ms per edit window is non-overlap, i.e. a critical-path problem (F3), not a bandwidth problem (F1). This is the single most useful output a roofline gives pancake, and it agrees with the 38.3% spin in `combine_cpu`.
- Per-expert placement (DALI/PreScope rule) applied to pancake: with CPU ~45 us per expert largely bandwidth-bound (weights read once per window, cost weakly dependent on m <= 5 tokens) and PCIe 115 us flat, the rule "GPU if t_gpu < t_cpu" almost never fires in decode. Expected gain from porting it: <=1-2% on edits; it would only matter in prefill or if a second GPU / PCIe 4.0 appears. Verdict: model it in the simulator (one line), do not build in the engine. Family F2 (placement), substitutes `--pcie-frac` / FreeToken split (item 22).
- The DALI windowed workload-score swap (w_size, u_size) is the same knob as pancake's `--adapt-swaps` / tier every window; nothing new to build, but DALI's best values (swap every 4 iterations, u=8 for 64-128-expert models) are a sanity reference for the 64/96/160 sweep.
- MoE-Lens and LIA target batch throughput or AMX/CXL hardware; not applicable to a single-user Ryzen box except as modelling method (stage-1 bound, profiler fitting GPU time vs tokens).

### Gaps
- HybriMoE institutional affiliation and timeline-simulation details were not visible on the arXiv abstract page; the simulation claim comes from a secondary summary.
- No paper found that fits such a cost model on an AMD consumer GPU over PCIe 3.0 with 512-expert layers; all evaluated models have 8-128 experts.

## Q2. Overlap and pipelining across CPU, PCIe, GPU and NVMe

### Takeaway
Published overlap schedulers hide I/O behind compute by reordering work inside a layer (resident experts first, CPU-cold experts in parallel, PCIe for the rest) or across layers/batches. For batch-1 decode the relevant pattern is "GPU runs the next independent work while CPU rows finish", which in pancake is the deferral path (item 8, KTransformers, already covered); multi-batch pipelines (Klotski) and phone-NVMe pipelines (PowerInfer-2) do not transfer.

### Cited Findings
- Klotski (Sun Yat-sen University + Huawei, lead SYSU; arXiv Feb 2025): observes per-expert compute is shorter than its load, leaving pipeline bubbles; uses an expert-aware multi-batch pipeline with a constraint-sensitive I/O-compute planner and an expert-correlation table for prefetch — [arXiv 2502.06888](https://www.arxiv.org/abs/2502.06888). Reported speedups (e.g. "85.12x") appear only on an aggregator page that mixes text from another paper — [catalyzex listing](https://catalyzex.com/author/Yufeng%20Lyu); treat as unverified. PreScope reports beating Klotski by up to 141% throughput — [arXiv 2509.23638](https://arxiv.org/html/2509.23638v1).
- PowerInfer-2 (SJTU IPADS, lead; arXiv Jun 2024): an offline planner profiles CPU FLOPS, I/O throughput, memory bandwidth and hot/cold neuron split per phone, then a neuron-cluster pipeline overlaps flash reads for uncached neurons with compute on cached clusters; 3.94x average over LLM in a Flash and 25.4x over llama.cpp; 11.68 tok/s on TurboSparse-Mixtral-47B — [arXiv 2406.06282](https://arxiv.org/html/2406.06282v2). A mirror claims the planner uses Z3 SMT solving — [reprint](https://rss.boorghani.com/?p=341310) (unverified against the paper).
- PreScope schedules prefetches only into the idle PCIe window created while the CPU computes cold experts, and never interrupts a started prefetch — [arXiv 2509.23638](https://arxiv.org/html/2509.23638v1).
- HybriMoE's intra-layer balancing fills CPU, GPU and transfer timelines per layer — [arXiv 2504.05897](https://arxiv.org/abs/2504.05897).

### Inferences
- For pancake the overlap ceiling is ~17 ms per edit window (Q1 roofline). Of the overlap techniques, only those that break the per-layer dependency (CPU rows of layer l consumed later, i.e. deferral) can reach it; intra-layer reordering helps only if GPU has independent work in the same layer, which at batch 1 is just the resident experts (~3 ms VRAM-hit phase per the grouping report). So: intra-layer reorder (HybriMoE/ProMoE style "cache -> in flight -> CPU") is worth at most a few percent; deferral remains the big lever. Family F3.
- NVMe: with the model resident in the ~34 GB RAM arena, NVMe is not on the decode path; a PowerInfer-2-style planner only matters if the arena is shrunk (RAM_KEEP / parked conversations conflict). Verdict: park.
- Klotski's multi-batch pipeline needs batch > 1; parallel sub-agents in opencode are sequential on one GPU today. Park.

### Gaps
- No university paper found that pipelines CPU expert compute, PCIe and GPU kernels for batch-1 speculative verify windows on a discrete PCIe 3.0 GPU.

## Q3. Lookahead scheduling with drafted tokens, and draft-length cost models

### Takeaway
2025 work uses the draft pass to predict which experts the verify pass will need (Expert Lookahead Buffer) and picks draft length with a roofline-style cycle-time model whose I/O term grows with new experts per extra position. Pancake's DraftPolicy already picks k by measured cost per size; what it lacks is an explicit term for "new CPU experts added by position k", which is exactly the MoE-SpeQ cycle formula with PCIe replaced by CPU expert time.

### Cited Findings
- MoE-SpeQ (SJTU, lead; HKUST co-author; arXiv Nov 2025): a small quantized draft model predicts experts for future tokens; the CPU builds a k x L Expert Lookahead Buffer of (expert_id, confidence) non-blocking while the GPU drafts; three prefetch phases (cache priming, bandwidth-guided mid-buffer prefetch, aggressive tail prefetch); a Speculative Governor picks k in [k_min, k_SLO] from cycle time = max(T_draft(k), T_pcie,init) + T_pcie,new(k) + T_verify(k+1), with T_pcie,new = new experts * size / PCIe bandwidth and acceptance from warm-up updated by EMA; A100 40 GB, PCIe 4.0; "at most 2.34x" over the best offloading framework on Phi-MoE in the abstract but 3.3x in Section 4.2 (inconsistent in the paper); no code link seen — [arXiv 2511.14102](https://arxiv.org/html/2511.14102v1).
- SP-MoE (already covered in the series) uses the draft model to anticipate verify-time experts, 1.07-3.5x TPOT — [arXiv 2510.10302](https://arxiv.org/html/2510.10302v2).
- DraftExpert (arXiv 2607.24434, seen only via an aggregator): per-layer lightweight draft experts for self-speculation under offloading, 1.45x decode, 86-88% prefetch hit on DeepSeek-V2-Lite / Moonlight — [aggregator](https://tldr.takara.ai/p/2607.24434) (not verified on arXiv).
- Sequoia (CMU, Beidi Chen's group, lead; with Together/Meta co-authors; NeurIPS 2024): a hardware-aware optimizer treats verification time as a hardware-dependent function of tokens verified and chooses tree size and depth from it; in offloading on L40, Llama2-70B at 0.56 s/token, 9.96x over their offloading baseline (HF page says up to 10.33x, version difference); code github.com/Infini-AI-Lab/Sequoia — [arXiv 2402.12374](https://arxiv.org/html/2402.12374v2).

### Inferences
- Concrete DraftPolicy change (exact, low effort): replace the fixed `kShape[t]` fallback with a model cost(t) = GPU(t) + max(0, CPU(U(t)) - overlap) + draft(t), where U(t) = expected distinct missed experts in the union of t+1 positions. U(t) comes from the simulator replay of routing traces (Q5), per context bucket. Then call `DraftPolicy::seed_costs()` with the simulated table so cold buckets start from a trace-derived curve instead of the RTX 5070 shape. This is SpecFit (item 36) made quantitative; family F4, complements DISCO (item 15). Expected gain: 0-5% (only where the current shape mis-prices sizes before measurements accumulate; the EMA already corrects later). Risk: none on output (exact).
- Expert lookahead from draft tokens: in pancake the MTP draft's hidden state could predict verify-window experts (as SP-MoE/MoE-SpeQ do), but on PCIe 3.0 there is no promotion budget (~115 us/expert) and the CPU pool is already DDR-bound, so prefetch into VRAM or L3 gains little (L3 prefetch already tested without gain). The useful use of the lookahead is informational: a predicted union-miss count per candidate k feeds the cost term above. Verdict: log-only first (recall of MTP-predicted experts vs actual in the item-30 recall column of trace run 2a).
- Sequoia's tree optimizer is excluded (draft trees conflict with GDN state), but its principle "verify cost is a measured hardware function of tokens verified" is what DraftPolicy already implements.

### Gaps
- No paper evaluates draft-driven expert lookahead where misses are computed on CPU rather than transferred; all assume PCIe transfer to GPU.
- MoE-SpeQ's headline vs body numbers conflict (2.34x vs 3.3x) and no code was found.

## Q4. Online auto-tuning and calibration of scheduling policies (bandits, RL, Bayesian optimization)

### Takeaway
The proven online method for speculation parameters is a UCB/EXP3 bandit over discrete configurations (BanditSpec, ICML 2025), with near-oracle throughput; engine-config tuning uses offline Bayesian optimization (SCOOT). Given pancake's ~20% run-to-run variance and an existing simulator, the cheaper path is: tune scheduler knobs offline on the simulator, keep online adaptation only for draft length (already done by DraftPolicy).

### Cited Findings
- BanditSpec (NUS-affiliated authors per author page host vyftan.github.io; ICML 2025, PMLR v267): treats speculative-decoding configurations as bandit arms maximising accepted tokens per step; UCBSpec (stochastic) and EXP3Spec (adversarial); training-free; selects speculation length 0-4 with Eagle-1 across batch 1-50; UCBSpec matches the oracle-best length in simulated serving; logarithmic regret bound under stationary means — [PMLR](https://proceedings.mlr.press/v267/hou25h.html); [arXiv 2505.15141](https://arxiv.org/html/2505.15141v1).
- SCOOT (Nanjing University + Ant Group; WWW 2025): single- and multi-objective Bayesian optimization of inference-engine parameters under SLOs, random-forest model of hidden constraints to avoid invalid configs, parallel suggestions, search-space pruning; tested on vLLM and TensorRT-LLM; deployed at Ant Group — [arXiv 2408.04323](https://arxiv.org/html/2408.04323v2).
- MoE-SpeQ updates acceptance probabilities online by EMA after a warm-up phase — [arXiv 2511.14102](https://arxiv.org/html/2511.14102v1).
- DALI calibrates all per-expert timings once by warm-up profiling and reuses them — [arXiv 2602.03495](https://arxiv.org/html/2602.03495v1).
- MoE-Lens calibrates its model with a pipeline profiler (linear fit of GPU time vs token count, per-layer transfer time) — [arXiv 2504.09345](https://arxiv.org/html/2504.09345v1).

### Inferences
- DraftPolicy vs BanditSpec: DraftPolicy is effectively a greedy contextual bandit (EMA payoff = tokens/cost, forced exploration via `kProbes` and stale re-measurement). BanditSpec's UCB term would replace the ad-hoc probe/stale rules with a principled exploration bonus; expected gain small (0-2%) because pancake has only `--spec 4` sizes and the measured `--spec 3/5/6` were flat. Verdict: optional cleanup on `feat/disco`, low priority. Family F4.
- Scheduler knobs (`--adapt-swaps`, tier period, k-th-miss k/W, `--pcie-frac`, pool pre-read on/off): an online bandit in the engine would fight the 20% noise and adapt-tier non-stationarity. Better: grid or BO on `time_ceiling.py` (cheap, deterministic), confirm the winner with one I5 A/B. Athena (covered) stays the fallback only if PREREAD A/B is mixed. Family F2/F9.
- A calibration loop worth building: at startup or every N windows, fit c_miss (us per CPU expert), GPU floor per window size and alpha (DMA contention) from the engine's own timers, the way DALI and MoE-Lens profile once; feed both DraftPolicy seed costs and simulator parameters from the same JSON. Effort: small once I3 logging exists.

### Gaps
- No paper found that applies RL or BO to expert placement on a single consumer box online; the cited online tuners address speculation parameters or engine flags.

## Q5. Simulators / digital twins for inference pipelines, and contention modelling (DDR shared by CPU compute and DMA)

### Takeaway
Trace-driven, operator-profiled simulators reach <2-9% latency error (Vidur, LLMServingSim 2.0), which is the accuracy bar pancake's `time_ceiling.py` should be validated against. The main structural gap in pancake's simulator is that it sums GPU and CPU time instead of modelling per-layer overlap (max-plus critical path), so it cannot price deferral, intra-layer reordering or per-expert placement. No 2022-2026 paper gives a model of DDR contention between CPU compute and PCIe DMA on discrete systems; MoE-Lens observed the effect as model overestimate.

### Cited Findings
- Vidur (Georgia Tech + Microsoft Research; MLSys 2024): operator-level profiling plus predictive models for unprofiled sizes; inference latency error <9% (an earlier draft says <5%); Vidur-Search finds the best deployment for LLaMA2-70B in one CPU-hour; open source — [MLSys poster](https://mlsys.org/virtual/2024/poster/2667); [arXiv 2405.05465](https://arxiv.org/html/2405.05465v1).
- LLMServingSim (KAIST, lead; 2024): iteration-level HW/SW co-simulation reusing results across iterations, <14.7% error vs real GPU serving, 91.5x faster than accelerator simulators — [arXiv 2408.05499](https://arxiv.org/html/2408.05499v1). LLMServingSim2.0 (IEEE CAL, Nov 2025): trace-driven operator-level profiler, MoE with expert offloading, 1.9% error; a Feb 2026 revision reports 0.97% average error and models batching, routing, offloading, memory and power in one runtime loop — [arXiv 2511.07229](https://arxiv.org/pdf/2511.07229); [arXiv 2602.23036 summary](https://www.opentrain.ai/papers/llmservingsim-2-0-a-unified-simulator-for-heterogeneous-and-disaggregated-llm-se--arxiv-2602.23036/).
- HybriMoE reportedly chooses schedules by simulating CPU/GPU/transfer timelines before execution — [review summary](https://www.themoonlight.io/review/hybrimoe-hybrid-cpu-gpu-scheduling-and-cache-management-for-efficient-moe-inference).
- MoE-Lens: model overestimates throughput when CPU attention contends with weight transfers (Mixtral, 210 GB KV) — [arXiv 2504.09345](https://arxiv.org/html/2504.09345v1).
- Contention literature is mostly pre-2022 and on integrated SoCs: host task WCET up to 5x with a memory-heavy co-running GPU app; GPU DMA contention lands on system DRAM — [Unimore RTNS'17](https://hipert.unimore.it/people/marko/papers/RTNS17.pdf); BWLOCK++ throttles CPU bandwidth while the GPU holds a lock — [arXiv 1712.08738](https://arxiv.org/pdf/1712.08738.pdf). Select-N (2025) monitors shared-PCIe-bus bandwidth instead of FlexGen's worst-case 1/n assumption — [arXiv 2502.08182](https://arxiv.org/pdf/2502.08182).

### Inferences
- Proposed simulator upgrade (owner: the session that owns items 14/27/30; this note is only a spec):
  1. Per-layer max-plus model instead of the sum: for each layer l and window, `T_l = max(G_l, C_l + s) ` for the non-deferred path and, for deferral, `C_l` overlapped with `G_{l+1}`; window = sum_l T_l + draft. G_l comes from the rocprofv3 profile (I7), C_l = misses_l * c_miss (already calibrated), s = sync cost. Calibrate against two observables at once: window ms (37.4 / 24.8) and spin share in `combine_cpu` (38.3%). This lets one trace answer items 8 (deferral ceiling), 28 (rank guard: which rows may defer), 29 (barrier sync s), 22/`--pcie-frac` (per-expert placement with DALI's greedy), and ProMoE reordering, none of which the current additive formula can price.
  2. Window-size dimension: replay with the union of experts over t+1 positions to emit cost(t) per context bucket for `DraftPolicy::seed_costs()` (Q3).
  3. Contention: keep `alpha`, but measure it once with a micro-benchmark (CPU pool running while hipMemcpy host->device streams), since no published model exists; also apply it to pool pre-read (same DDR).
  4. Validation target: predicted vs measured window within ~5-9% (Vidur bar) on at least three configs (preset, `--pcie-frac` 0.2, tier off) before any item-14 decision uses the >=5% rule; with 20% run variance, compare against I5 distributions.
  Effort: ~1-2 days Python for 1-2 given traces and a per-layer GPU profile; risk none (offline). Family F9 (measurement) serving F2/F3/F4.
- Vidur/LLMServingSim themselves are cluster-serving simulators (batching, routing) and are not worth adopting; their method (operator profiles + interpolation + error validation) is what to copy.

### Gaps
- No 2022-2026 university paper models DDR bandwidth contention between CPU expert GEMV and PCIe DMA on a discrete GPU system; the alpha coefficient must be measured locally.
- Vidur's per-operator estimator accuracy was not verified in the full text.

## Q6. Critical-path analysis tools for heterogeneous inference

### Takeaway
Existing tools build kernel/operator dependency graphs from PyTorch/CUPTI traces (SKIP, TaxBreak, HTA) and attribute idle time to host vs device, but none does what-if analysis for a CPU+GPU MoE pipeline, and none supports HIP/rocprof traces of a C++ engine. For pancake a small custom critical-path pass over its own per-layer timestamps is the practical route.

### Cited Findings
- SKIP (arXiv Apr 2025): builds operator-to-kernel dependency graphs from PyTorch Profiler traces via CUPTI correlation IDs, with a TKLQT metric to locate CPU-bound vs GPU-bound transitions; studies CPU-GPU coupled architectures — [arXiv 2504.11750](https://arxiv.org/html/2504.11750v1).
- TaxBreak (arXiv Mar 2026): trace-driven split of host overhead into framework, library and launch time; notes that GPU utilization cannot distinguish host delays, device dependency chains and memory stalls — [arXiv 2603.12465](https://arxiv.org/html/2603.12465v1).
- PyTorch Holistic Trace Analysis has a `critical_path_analysis` function aimed at training steps — [vendor guide](https://www.spheron.network/blog/gpu-profiling-ai-workloads-nsight-compute-pytorch-profiler-guide/) (secondary source).

### Inferences
- Pancake-specific critical-path pass: per window, timestamp per layer (GPU route publish, CPU pool start/end, `combine_cpu` wait start/end, PCIe flag wait). The critical path alternates GPU->CPU->GPU; slack per layer = G_l - C_l. Output: ms of spin per layer, which layers are CPU-critical, and the "virtual speedup" answer (Coz-style: how much the window shrinks if C_l shrinks by x% vs G_l by x%). This is the same data that calibrates the max-plus simulator (Q5) and directly tells whether F8 kernel work is worth anything on edits (it is not while CPU is critical). Effort: small (timers mostly exist via `STRATA_*_TIMING`-style switches; not verified in code here). Family F9, enables F3/F8 decisions.

### Gaps
- Institutions of SKIP and TaxBreak were not established from the search results.
- No published critical-path/what-if tool targets HIP + CPU-thread-pool MoE engines.

## Summary table for the report writer (verdicts are estimates from the cited mechanisms and pancake's measured budget)

| Technique (lead institution, venue) | Pancake use | Plug-in point | Expected gain | Effort | Risk | Family |
|---|---|---|---|---|---|---|
| Per-layer max-plus timeline (HybriMoE PKU DAC'25, PreScope NUDT/Tsinghua, DALI CAS) | replace additive window formula; price deferral/reorder/placement | `time_ceiling.py` | indirect (decides items 8/22/28/29/30) | 1-2 days | none | F9 -> F2/F3 |
| DALI greedy per-expert CPU/GPU placement | per-expert `--pcie-frac` by token count | simulator only | <=1-2% edits | small | none | F2 |
| Three-roof roofline (MoE-Lightning HRM idea, MoE-Lens UMich, MoE-SpeQ SJTU) | shows edit window is on the sum line: ~17 ms non-overlap | report / dashboard | — (diagnostic) | hours | none | F3/F9 |
| MoE-SpeQ cycle-time governor (SJTU) | cost(t) with union-miss CPU term | `DraftPolicy::seed_costs`, `kShape` | 0-5% | small | none (exact) | F4 (SpecFit 36) |
| Draft-driven expert lookahead (MoE-SpeQ, SP-MoE) | log recall only | trace run 2a | ~0 direct on PCIe 3.0 | small | none | F2/F4 |
| BanditSpec UCB (NUS, ICML'25) | exploration rule in DraftPolicy | `draft_policy.cpp` | 0-2% | small | low | F4 |
| SCOOT BO (NJU/Ant, WWW'25) | tune tier/k-miss/swaps on the simulator, not online | offline script | via F2 band (+5-16%) | small | none | F2/F9 |
| Vidur (GaTech/MSR, MLSys'24) / LLMServingSim (KAIST) | validation methodology (<9% / <2% error) | simulator validation | — | small | none | F9 |
| Contention micro-benchmark for alpha (no paper; MoE-Lens evidence) | measure DDR contention CPU pool vs DMA | simulator alpha, PREREAD | — | hours | none | F9 |
| Custom critical-path pass (SKIP/TaxBreak method) | per-layer slack and virtual speedup | engine timers -> offline | — (decides F3 vs F8) | small | none | F9 |
| Klotski (SYSU), PowerInfer-2 (SJTU), LIA (UIUC), MoE-Lens scheduler | multi-batch, phone NVMe, AMX/CXL, batch throughput | — | not applicable | — | — | park |
