# Control plane vs execution plane in LLM inference runtimes (2026)

Scope: the split between scheduling, KV/memory management and request lifecycle (control plane) and model kernels / forward pass (execution plane). Layers covered: survey -> techniques -> mechanisms -> implementations -> applicability. Research date: 2026-11-08. All benchmark figures are vendor/project self-reported unless noted.

## 1. lightseekorg / TokenSpeed: does it exist, and is the "C++ FSM control plane with compile-time checks, Python execution plane" claim accurate?

### Takeaway
**Verified, with one overstatement.** TokenSpeed exists: an MIT-licensed open-source inference engine from the LightSeek Foundation at github.com/lightseekorg/tokenspeed, aimed at agentic workloads. Its scheduler does split a C++ control plane, built as a finite-state machine working with the type system, from a Python execution plane. The phrase "zero runtime errors" is **not found** in the primary sources. The project claims that safe KV reuse is "enforced by the type system at compile time" and checked by "a verifiable control system rather than convention", but it shows no formal guarantee, proof or mechanism.

### Cited Findings
- The repo exists at github.com/lightseekorg/tokenspeed. It is MIT-licensed, and at fetch time it showed about 2.9k stars, 402 forks, about 2,395 commits and no formal releases. Its news log runs to 2026/10–11, and it joined the PyTorch Ecosystem in 2026/08. — [GitHub lightseekorg/tokenspeed](https://github.com/lightseekorg/tokenspeed)
- README wording: "Request lifecycle, KV cache ownership, and overlap timing are encoded as a finite-state machine", with "safe KV resource reuse enforced by the type system at compile time". — [GitHub lightseekorg/tokenspeed](https://github.com/lightseekorg/tokenspeed)
- Launch blog (May 6, 2026, "TokenSpeed Team"): the scheduler "decouples the control plane from the execution plane". The control plane is "implemented in C++ as a finite-state machine that works with the type system". Request lifecycle, KV cache resources and overlap timing are explicit FSM transitions with ownership semantics. Safe KV handling is enforced "at compile time rather than at runtime". Correctness comes from "a verifiable control system rather than convention". The Python execution plane exists to "maintain development efficiency". — [LightSeek blog](https://lightseek.org/blog/lightseek-tokenspeed.html)
- The blog does **not** claim "zero runtime errors" or any formal verification results, and it does not describe the type-level mechanism (for example typestate or move-only handles). — [LightSeek blog](https://lightseek.org/blog/lightseek-tokenspeed.html)
- Scheduler contributors named: Yulei Qian, Fengcun Li and Bo Wang. AsyncLLM and SMG (Rust gateway) integration is credited to Simo Lin, Keyang Ru and Xipeng Guan. — [LightSeek blog](https://lightseek.org/blog/lightseek-tokenspeed.html)
- Other layers:
  - Front-end: SMG (SGLang Model Gateway) and NVIDIA Dynamo drive the engine from Rust over a msgpack wire, "no Python on the request path".
  - Kernels: a pluggable, layered kernel system with a portable API, a central registry and plugins for heterogeneous accelerators (AMD and NPU directories exist).
  - Modeling layer: local SPMD, with a static compiler that generates the collectives.
  — [GitHub](https://github.com/lightseekorg/tokenspeed); [BestHub / Machine Heart article](https://www.besthub.dev/articles/nvidia-endorses-tokenspeed-a-light-speed-agent-inference-engine-built-in-two-months-419dd7b1bf68)
- Maturity: development began in mid-March 2026. The May 2026 blog calls the release a "performance preview", says production hardening was planned for the following month, and says PD disaggregation and KV storage were still being merged. — [LightSeek blog](https://lightseek.org/blog/lightseek-tokenspeed.html); [BestHub](https://www.besthub.dev/articles/nvidia-endorses-tokenspeed-a-light-speed-agent-inference-engine-built-in-two-months-419dd7b1bf68)
- Benchmarks (self-reported). Setup: Kimi K2.5 on B200, SWE-smith coding-agent traces (evaluated with the EvalScope team), single non-disaggregated deployment, compared against TensorRT-LLM. Best config at >70 TPS/user was attention TP4 + MoE TP4.
  - About 9% lower latency than TensorRT-LLM at batch size 1.
  - About 11% higher throughput near 100 TPS/user.
  - The MLA decode kernel "nearly halves latency" versus TRT-LLM with speculative decoding at batch sizes 4, 8 and 16.
  - The MLA prefill kernel beats TRT-LLM on 5 prefill workloads.
  — [LightSeek blog](https://lightseek.org/blog/lightseek-tokenspeed.html)
- The README news log lists later figures: 580 TPS on agentic workloads (2026/05, PyTorch blog) and 685 tok/s/user on GB300 (2026/10). The model names come from the fetch summary and were not checked character by character. — [GitHub](https://github.com/lightseekorg/tokenspeed)
- vLLM reportedly adopted TokenSpeed's MLA kernel. — [BestHub](https://www.besthub.dev/articles/nvidia-endorses-tokenspeed-a-light-speed-agent-inference-engine-built-in-two-months-419dd7b1bf68)
- The headline "Nvidia endorses" appears only in an aggregator headline, and the article body does not substantiate it. — [BestHub](https://www.besthub.dev/articles/nvidia-endorses-tokenspeed-a-light-speed-agent-inference-engine-built-in-two-months-419dd7b1bf68)
- The PyTorch Foundation write-up, as quoted in the README, calls TokenSpeed "the first to separate the control plane from the execution plane". That is marketing: SGLang, vLLM and TensorRT-LLM already separated scheduling processes and threads from execution (see section 2). — [GitHub](https://github.com/lightseekorg/tokenspeed)

### Inferences
- Rate the overview's claim as **partially accurate**. The architecture description (C++ FSM control plane, compile-time type checks on KV and request lifecycle, Python execution plane) matches the primary sources. "Zero runtime errors" or a "guarantee" overstates them: the sources claim certain misuse is impossible to compile, not that the system is free of runtime errors, and no verification artifact has been published.
- The likely mechanism is typestate-style C++ (states as distinct types, move-only ownership tokens for KV blocks). This is an inference and is not documented.
- "Novelty" means a strongly typed control plane in a different language from the execution plane. It does not mean being the first to separate the two planes.

### Gaps
- Exact C++ types and FSM states, plus code paths for overlap timing, were not inspected in the source tree (only the README and blog were read).
- No independent benchmark: all results compare against a single baseline (TRT-LLM) on one model and one GPU.
- The figures behind "580 TPS" and "685 tok/s/user" (hardware and concurrency) were not verified.

## 2. How vLLM V1, SGLang, TensorRT-LLM and LMDeploy TurboMind split control and execution; CPU overhead and how they hide it

### Takeaway
All four engines converged on the same pattern. A CPU-side scheduler/KV manager runs in its own process or thread and prepares step N+1 while the GPU runs step N (the "overlap" or "async" scheduler, which traces back to NanoFlow). Separately, I/O such as tokenization and detokenization is moved out of the hot loop. The engines differ mainly in implementation language (Python vs C++) and in how much input preparation has moved onto the GPU.

### Cited Findings
**Survey / origin**
- NanoFlow (arXiv 2408.12757) found that when the GPU is highly utilized, CPU work (KV-cache management, batch formation, retired-request selection) takes >10% of inference time. It hides this with asynchronous scheduling that runs batch formation in parallel with GPU execution. It reports 1.91x average throughput over vLLM, DeepSpeed-FastGen and TRT-LLM (68.5% of optimal), and an online request rate up to 1.64x TRT-LLM's at similar latency. The >10% quote was seen on a secondary page. — [arXiv 2408.12757](https://arxiv.org/html/2408.12757v1)
- The SGLang v0.4 blog says an unoptimized engine can spend "up to half" its time on CPU overhead (scheduling, memory allocation, prefix matching). — [LMSYS SGLang v0.4](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4)

**vLLM V1** (V1 alpha Jan 2025; Model Runner V2 Mar 2026)
- `EngineCore` is an isolated execution loop "that focuses exclusively on the scheduler and model executor". It builds on the v0.6.0 multiprocess API server, which uses ZeroMQ IPC. Tokenization, multimodal preprocessing, detokenization and streaming overlap with the core loop. — [vLLM V1 alpha blog](https://vllm.ai/blog/2025-01-27-v1-alpha-release)
- `EngineCoreProc` wraps `EngineCore` with ZMQ I/O. Messages use msgpack, and background threads bridge the sockets to queues. Running the API server event loop and the GPU loop in separate processes means they don't contend for the GIL. — [vLLM EngineCore IPC write-up](https://hungchun0201.github.io/agentic-ai-survey/ai-infra-basics/vllm/engine-core.html); [Infomaniak guide](https://engineering.infomaniak.com/vllm-ultimate-guide-part-1/)
- The unified scheduler has no prefill/decode distinction. Decisions are a dict `{request_id: num_tokens}`, which covers chunked prefill, prefix caching and speculative decoding. — [vLLM V1 alpha blog](https://vllm.ai/blog/2025-01-27-v1-alpha-release)
- Prefix caching is hash-based with constant-time LRU eviction. It is on by default, with <1% throughput loss at a 0% hit rate. — [vLLM V1 alpha blog](https://vllm.ai/blog/2025-01-27-v1-alpha-release)
- Persistent batch: input tensors are cached and updated with diffs each step, using NumPy. Under TP, workers cache request state and receive only incremental diffs, and the scheduler is no longer co-located with worker 0. Execution uses torch.compile plus piecewise CUDA graphs. Result: up to 1.7x throughput over V0. For context, Llama-8B on H100 runs only about 5 ms of GPU time per step. — [vLLM V1 alpha blog](https://vllm.ai/blog/2025-01-27-v1-alpha-release)
- Earlier, vLLM v0.6.0 overlapped output processing and added multi-step scheduling, for 28% throughput on Llama-70B on 4xH100. — [vLLM v0.6.0 blog](https://vllm.ai/blog/2024-09-05-perf-update)
- Model Runner V2 (March 2026) makes async scheduling "a design constraint, not a retrofit". Step N+1 is prepared while the GPU runs step N, and outputs come back on a separate CUDA stream. Persistent request state lives in a fixed-size table, gathered into per-step inputs. Inputs (`input_ids`, `positions`, `seq_lens`…) are built on the GPU with Triton, with "zero synchronization" as the goal. Results:
  - 25K vs 16K output tok/s (+56%) on Qwen3-0.6B on 1×GB200, a small model chosen to expose host overhead.
  - 6.3% lower TPOT on GLM-4.7-FP8 with MTP on 4×GB200.
  — [vLLM MRV2 blog](https://vllm.ai/blog/2026-03-24-mrv2)

**SGLang**
- The zero-overhead batch scheduler (v0.4, Dec 2024) runs one batch ahead: it prepares the next batch's metadata, including radix-cache operations, while the GPU computes. Dependencies are resolved with "future tokens" plus CUDA events. Nsight showed no GPU idle time over 5 decode batches with Triton attention, and a small gap with FlashInfer.
  - Throughput: about 1.1x from overlap alone; 1.9x decode throughput for v0.4 vs v0.3 with all optimizations.
  - Biggest gains on small models and large TP.
  - On by default; `--disable-overlap` turns it off.
  - Benchmark: Llama-3.2-3B, 4096 in / 2048 out.
  — [LMSYS SGLang v0.4](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4); [beam.cloud](https://www.beam.cloud/blog/sglang)

**TensorRT-LLM**
- The overlap scheduler is "enabled by default". At step n it launches GPU work for step n+1 before step n's CPU work (stop criteria, response updates, scheduling) finishes. The cost is one extra decoding step. `disable_overlap_scheduler=True` turns it off. The docs cite NanoFlow and the SGLang blog. — [TRT-LLM overlap scheduler](https://nvidia.github.io/TensorRT-LLM/1.2.0/torch/features/overlap_scheduler.html)
- The PyTorch backend (now the main path) is built around a Python `PyExecutor` that mirrors the C++ Executor API. Per step it fetches requests, schedules, runs the forward pass, samples and retires requests.
  - Scheduling has two stages, `CapacityScheduler` and `MicroBatchScheduler`.
  - `KVCacheManager` and the schedulers are C++-bound but have Python-customizable interfaces.
  - Execution is `PyTorchModelEngine`.
  - The control plane is therefore a hybrid: a Python orchestration loop around C++ cores, plus CUDA-graph execution.
  — [TRT-LLM Architecture Overview](https://nvidia.github.io/TensorRT-LLM/latest/torch/arch_overview.html)

**LMDeploy TurboMind**
- Persistent batch: N pre-configured slots that requests occupy and release (that is, continuous batching). The KV cache manager is an LRU "cache of KV caches", and evicted sequences are kept as token IDs and recomputed on a miss. — [LMDeploy TurboMind docs](https://lmdeploy.readthedocs.io/en/latest/inference/turbomind.html)
- The current docs (per a search snippet) describe a fully C++ engine with an asynchronous model "to decouple host-side scheduling from device-side execution":
  - An Engine Thread handles admission, validation, cancellation, `Scheduler::Schedule()` and retirement.
  - A `BatchData` carrier passes phase information and CUDA events to the executor threads.
  - A third-party source reading describes a single engine thread fed from a Gateway via queues.
  — [LMDeploy docs search snippet](https://lmdeploy.readthedocs.io/en/latest/inference/turbomind.html); [pyshine walkthrough](https://pyshine.com/lmdeploy-2/)
- Claimed up to 1.8x request throughput vs vLLM (older official claim, undated baseline). — [LMDeploy docs](https://lmdeploy.readthedocs.io/)

### Inferences
- The "hiding" toolbox is the same across engines:
  1. Process or thread isolation of I/O from the core loop (vLLM ZMQ, SGLang tokenizer/detokenizer managers, TurboMind engine thread).
  2. One-step-ahead overlap scheduling.
  3. CUDA graphs to cut kernel-launch overhead.
  4. Incremental and persistent batch state, diffs instead of rebuilds.
  5. Moving input preparation onto the GPU (vLLM MRV2).
- Measured benefit scales inversely with model size: about 1.1x for SGLang overall, up to 1.56x for vLLM on a 0.6B model. For large models on a single GPU it is usually single-digit percent.
- The overlap approaches cost one extra decode step of latency, and they complicate speculative decoding and structured output because of the dependency on the previous step's tokens.

### Gaps
- No primary-source per-engine "scheduler overhead %" measured under identical conditions, and no 2026 comparison.
- vLLM upstream default for `--async-scheduling` on CUDA was not confirmed. A Neuron doc says it is on by default there.
- The current TurboMind architecture page was only partly retrievable, so the BatchData and Gateway names rest on a search snippet and a secondary walkthrough.

## 3. Rust/C++ serving front-ends (routers/gateways) and why

### Takeaway
Rust has become the standard for the layer in front of the engines. SGLang's router became the SGLang Model Gateway (SMG). vLLM's `vllm-project/router` is a fork of it, and TokenSpeed uses SMG and Dynamo. The reasons: high-concurrency HTTP/gRPC, cache-aware routing, PD orchestration, and moving tokenization and parsing out of Python so there is "no Python on the request path".

### Cited Findings
- The SGLang v0.4 cache-aware load balancer is "pure Rust" and claims about 2x speed over Python alternatives. It keeps an approximate, lazily updated radix tree per worker without syncing with workers. Results: up to 1.9x throughput and 3.8x hit rate; on 8×A100, 82,665 → 158,596 tok/s with hit rate going from 20% to 75%. — [LMSYS SGLang v0.4](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4)
- SGLang Model Gateway (`sgl-model-gateway`, `smg`; renamed Dec 2025 per a non-official source) has a gRPC mode running "fully in Rust". Tokenizer, reasoning parser and tool parser sit in the router, which streams tokenized requests to SRT gRPC workers. It supports PD disaggregation and a multi-tenant gateway mode (`--enable-igw`). — [SGLang Model Gateway docs](https://docs.sglang.io/advanced_features/sgl_model_gateway.html); [PyPI smg](https://pypi.org/project/smg/)
- `vllm-project/router` is a Rust core with Python bindings (PyPI `vllm-router`), "derived from a fork of the SGLang model gateway". It offers load-balancing policies, PD disaggregation and Kubernetes discovery. — [PyPI vllm-router](https://pypi.org/project/vllm-router/); [deepwiki](https://deepwiki.com/vllm-project/router)
- vLLM Semantic Router is a different project: an intent-aware Mixture-of-Models router running as an Envoy ExtProc (Go plus Rust candle bindings), with self-reported +10.2% accuracy and −47.1% latency on MMLU-Pro with Qwen3-30B. — [vLLM Semantic Router blog](https://vllm.ai/blog/2025-09-11-semantic-router); [Red Hat](https://developers.redhat.com/articles/2025/09/11/vllm-semantic-router-improving-efficiency-ai-reasoning)
- TokenSpeed is driven from Rust (SMG and Dynamo) over msgpack, with no Python on the request path. — [GitHub lightseekorg/tokenspeed](https://github.com/lightseekorg/tokenspeed)

### Inferences
- The architecture is now three tiers: a Rust gateway for HTTP, tokenization, parsing and routing; then a control plane (a Python or C++ scheduler with a KV manager); then an execution plane (PyTorch, CUDA graphs and custom kernels). The Rust tier removes GIL contention and per-request Python cost at high QPS.

### Gaps
- TGI's Rust router (continuous batching in Rust, with gRPC to Python shards) was not re-verified in this session. Background knowledge only: TGI went into maintenance mode around late 2025. Confirm from the huggingface/text-generation-inference README before citing.

## 4. Measured effects (summary table, self-reported)

### Takeaway
Overlap and async scheduling typically gives 1.1x throughput on mid-size setups and up to about 1.5x on tiny models where host overhead dominates. Architecture rewrites (V1, MRV2) give 1.5–1.7x. Rust routing gives up to 1.9x through cache hits, not through scheduler speed.

### Cited Findings
- SGLang overlap: about 1.1x throughput (Llama-3.2-3B); v0.4 overall 1.9x decode vs v0.3. — [LMSYS](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4)
- vLLM V1 vs V0: up to 1.7x throughput. — [vLLM V1 blog](https://vllm.ai/blog/2025-01-27-v1-alpha-release)
- vLLM v0.6.0 multi-step scheduling: +28% (Llama-70B, 4×H100). — [vLLM v0.6.0](https://vllm.ai/blog/2024-09-05-perf-update)
- vLLM MRV2: +56% (Qwen3-0.6B, GB200); −6.3% TPOT on GLM-4.7 with MTP. — [vLLM MRV2](https://vllm.ai/blog/2026-03-24-mrv2)
- NanoFlow: CPU overhead >10% of inference time; 1.91x throughput. — [arXiv 2408.12757](https://arxiv.org/html/2408.12757v1)
- TRT-LLM overlap: "significant" throughput gain at the cost of one extra decode step, with no number given. — [TRT-LLM](https://nvidia.github.io/TensorRT-LLM/1.2.0/torch/features/overlap_scheduler.html)
- TokenSpeed vs TRT-LLM: −9% latency at BS1, +11% throughput at about 100 TPS/user. — [LightSeek blog](https://lightseek.org/blog/lightseek-tokenspeed.html)
- SGLang Rust cache-aware router: 1.9x throughput, 3.8x hit rate. — [LMSYS](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4)

### Inferences
- Applicability to a single consumer-GPU setup (for example one RTX 3060 serving a 7–14B model at batch 1–4): per-step GPU time is tens of ms, so CPU scheduling overhead is a small fraction and overlap gains will be modest. Gains appear with small models, high batch counts or very fast GPUs. CUDA graphs and avoiding CPU–GPU syncs matter more there than a typed C++ control plane. TokenSpeed targets Blackwell datacenter MoE agentic serving and has no consumer-GPU evidence.

### Gaps
- No apples-to-apples 2026 measurement of scheduler overhead % across engines. Every figure comes from a different model, GPU and workload.
