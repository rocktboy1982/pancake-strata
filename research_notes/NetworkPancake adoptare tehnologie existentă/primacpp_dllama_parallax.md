# prima.cpp, distributed-llama, Parallax as candidate bases for NetworkPancake (as of 2026-11-10)

Method: GitHub REST API (unauthenticated; it hit the rate limit mid-way, so Parallax commit/contributor numbers come from calls made before the limit), raw.githubusercontent.com source files I downloaded and grepped (not built, not run), WebFetch of GitHub pages. File sizes and line numbers come from the trees/files I fetched. Effort estimates are my inference and are marked. Benchmark tables are in existing_systems.md and are not repeated here.

## 1. prima.cpp: live repo status, license, activity

### Takeaway
Lizonghang/prima.cpp is still 404. The only live repo is OpenCPIL/prima.cpp (MIT), created 2026-06-30, committed by the same person (zonghang-li), and the API says fork:false, so it is the author's new home, not a third-party fork. It is nearly dormant: 3 commits in the last 90 days (README edits), 1 contributor. Treat it as a research snapshot of an old llama.cpp fork.

### Cited Findings
- Lizonghang/prima.cpp returns "Not Found" — [API](https://api.github.com/repos/Lizonghang/prima.cpp)
- OpenCPIL/prima.cpp: fork=false, created 2026-06-30, last push 2026-09-06, 129 stars, 22 forks, 2 open issues, MIT, repo size about 32 MB, not archived — [API](https://api.github.com/repos/OpenCPIL/prima.cpp)
- Commits since 2026-08-12: 3; contributors: 1 — [commits](https://api.github.com/repos/OpenCPIL/prima.cpp/commits?since=2026-08-12T00:00:00Z), [contributors](https://api.github.com/repos/OpenCPIL/prima.cpp/contributors)
- Recent commits (2026-06-30 "Add files via upload", 2026-09-01 README update about a "Hy4 770B experiment") are all by zonghang-li through the GitHub web UI — [commits](https://api.github.com/repos/OpenCPIL/prima.cpp/commits?per_page=10)
- LICENSE is MIT, "Copyright (c) 2023-2024 The ggml authors and 2024-2025 the prima.cpp authors" — [LICENSE](https://raw.githubusercontent.com/OpenCPIL/prima.cpp/main/LICENSE)
- README links a blog about a "Hy4 preview 770B" run; the page is a JS redirect, so I could not read it. Whether that model is MoE and what numbers it gave: UNVERIFIED — [README](https://raw.githubusercontent.com/OpenCPIL/prima.cpp/main/README.md)
- Paper: arXiv 2504.08791, ICLR 2026 (see academic_papers.md).

### Inferences
- "[Official]" resolved: official in the sense of same author. The old account repo was removed or renamed; I found no document explaining why.
- The bulk-upload commit pattern suggests the original development history was not carried over (inference).
- Bus factor 1.

### Gaps
- Release list not queried separately. Reason for the 404 undocumented.

## 2. prima.cpp: architecture, protocol, hardware, MoE, security

### Takeaway
A hard fork of llama.cpp: monolithic `src/llama.cpp` of 24,559 lines (1.05 MB), ZeroMQ push/pull ring between ranks, an in-binary device profiler and an ILP layer-assignment solver (HiGHS) in `common/common.cpp`. All welded into the llama.cpp context, so there is no plug-in point for another engine. Dense models in the README; no MoE-aware scheduling.

### Cited Findings
- Transport is ZeroMQ: `src/llama.cpp` holds `zmq::socket_t` push/pull sockets (send, recv, master, signal) and uses `zmq::send_multipart` for sync metadata, tensors and device info; ranks form a ring (`--world N --rank R --master IP --next IP`); default ports 9000 (data) and 10000 (signal) — [src/llama.cpp](https://raw.githubusercontent.com/OpenCPIL/prima.cpp/main/src/llama.cpp), [README](https://raw.githubusercontent.com/OpenCPIL/prima.cpp/main/README.md)
- No authentication or encryption seen: README advises disabling the firewall or changing ports; SECURITY.md is the stock llama.cpp one. I did not grep for zmq CURVE — [README](https://raw.githubusercontent.com/OpenCPIL/prima.cpp/main/README.md), [SECURITY.md](https://raw.githubusercontent.com/OpenCPIL/prima.cpp/main/SECURITY.md)
- `llama-server` runs on rank 0, `llama-cli` on the others — [README](https://raw.githubusercontent.com/OpenCPIL/prima.cpp/main/README.md)
- Hardware: macOS, Linux, Android, HarmonyOS (via Termux); CUDA only for GPUs; Vulkan on roadmap; AMD only "once we have that device"; Windows not yet — [README FAQ](https://raw.githubusercontent.com/OpenCPIL/prima.cpp/main/README.md)
- Quant: Q4K, Q6K, Q80, IQ1; speculative decoding; dynamic batching; models "Llama, Qwen (and QwQ), DeepSeek series" — [README](https://raw.githubusercontent.com/OpenCPIL/prima.cpp/main/README.md)
- MoE: inherited llama.cpp architecture enums still include qwen2moe, olmoe, granite moe (113 `n_expert` hits in src/llama.cpp), but the README claims only dense families — [src/llama.cpp](https://raw.githubusercontent.com/OpenCPIL/prima.cpp/main/src/llama.cpp)
- Build deps: gcc-9+, make, cmake, fio, libzmq, HiGHS >= 1.9.0 (rank 0 built with `USE_HIGHS=1`) — [README](https://raw.githubusercontent.com/OpenCPIL/prima.cpp/main/README.md)
- UI: only the inherited llama.cpp server page — [tree](https://api.github.com/repos/OpenCPIL/prima.cpp/git/trees/main?recursive=1)

### Inferences
- Hosting Strata as a prima rank would mean re-implementing the ring and sync_meta/device_info exchange inside Strata, because the ring carries per-ubatch ggml tensors, not text.

### Gaps
- Did not read the sync_meta wire layout field by field; did not check zmq security options.

## 3. Halda scheduler: where, how big, reusable?

### Takeaway
The word "Halda" does not appear in the source (grep of src/llama.cpp, common/common.cpp, common/profiler.cpp, README: no hits). The algorithm is `assign_layers_to_device()` in `common/common.cpp` (starts at line 860; the HiGHS ILP block spans roughly lines 889-1530, about 670 lines), plus the profiler `common/profiler.cpp` (2,864 lines, 100 KB) and `profiler.h` (13 KB). It can be lifted as an algorithm (port about 700 lines), not as a library: it takes `llama_model*` and `llama_context_params` and the `device_info` struct.

### Cited Findings
- Signature: `static bool assign_layers_to_device(uint32_t n_world, const device_info* dev_info_set, uint32_t* n_layer_window, uint32_t* n_gpu_layers, llama_model* model, const llama_context_params cparams, float min_disk_read_speed = 0.1f)`, inside `#if defined(USE_HIGHS)`; computes per-device alpha, beta, xi and kappa constants, builds a `HighsModel` with `n_world*2` integer variables, runs HiGHS, outputs layer-window and GPU-layer counts — [common/common.cpp](https://raw.githubusercontent.com/OpenCPIL/prima.cpp/main/common/common.cpp)
- `struct device_info` (profiler.h) holds disk, cpu_props, memory, gpu_props; probes `device_disk_seq_bw`, `device_disk_rnd_bw`, `device_memory_bw`, `device_inp_embd_delay`, backend detectors (`device_has_cuda/metal/vulkan/sycl`), and `serialize()` for sending over the network — [common/profiler.h](https://raw.githubusercontent.com/OpenCPIL/prima.cpp/main/common/profiler.h)
- The profiler runs ggml micro-benchmarks, and the model side uses `model_flops`/`model_bytes` from a loaded llama model — [profiler.h](https://raw.githubusercontent.com/OpenCPIL/prima.cpp/main/common/profiler.h)
- Paper: Halda solves its integer program in about 10-12 ms for 4-32 devices — [arXiv](https://arxiv.org/html/2504.08791v2)
- License MIT, same repo.

### Inferences
- A coordinator-side port is realistic: Python function with MILP (scipy milp or HiGHS bindings, MIT), inputs per-device FLOPS, memory bandwidth, free RAM/VRAM, disk bandwidth, link latency; outputs layer windows. Estimate 2-3 days to port and test from the paper's equations with the C++ as reference, plus about 0.5 day for a Strata-side probe (nvidia-smi/psutil/disk test). Estimates are mine.
- Lifting the profiler as is (2,864 lines) is not worth it; it is bound to ggml and old quant tables.
- Halda models dense layers with mmap page-cache and disk. Strata is MoE with its own expert streaming, so the model needs reformulating. How well it transfers is UNVERIFIED.

### Gaps
- Did not read all 670 lines of the solver; constant semantics come from the paper. Did not check whether the solver has an expert term (likely not, README is dense-only).

## 4. distributed-llama (b4rtaz): license, activity, releases

### Takeaway
The healthiest of the three: MIT, pushed 2026-11-09, 33 commits in 90 days from 14 contributors, regular releases (v0.17.0 on 2026-11-07). A from-scratch C++ codebase (not a llama.cpp fork), small and readable.

### Cited Findings
- MIT, created 2023-12-04, 3,132 stars, 260 forks, 49 open issues, last push 2026-11-09, repo size about 3.9 MB — [API](https://api.github.com/repos/b4rtaz/distributed-llama)
- Commits since 2026-08-12: 33; contributors on first page: 14 — [commits](https://api.github.com/repos/b4rtaz/distributed-llama/commits?since=2026-08-12T00:00:00Z&per_page=100)
- Releases: v0.17.0 (2026-11-07), v0.16.5 (2026-02-02), v0.16.4 (2026-01-17), v0.16.3 (2025-10-26), v0.16.2 (2025-09-20) — [releases](https://api.github.com/repos/b4rtaz/distributed-llama/releases)
- README lists Qwen 3.5 9B Q40 (8.0 GB) beside Qwen3 30B A3B Q40 (17.0 GB); `src/graphs/qwen35-graph.cpp` (37 KB) and a Vulkan gated-delta-net shader exist — [README](https://raw.githubusercontent.com/b4rtaz/distributed-llama/main/README.md), [tree](https://api.github.com/repos/b4rtaz/distributed-llama/git/trees/main?recursive=1)

### Inferences
- The 14 contributors suggest real PR flow, but I did not check how many commits come from the maintainer.

## 5. distributed-llama: architecture and extension points

### Takeaway
Layered design: compute-graph IR (`nn-core`), pluggable `NnDevice`/`NnDeviceSegment` interface (CPU and Vulkan implementations), `NnNodeSynchronizer`, and a raw TCP layer (`nn-network`). `NnDevice` is a real backend extension point, but at op level (matmul slices, rope, norm) of a tensor-parallel graph, which does not match Strata's monolithic engine.

### Cited Findings
- Sizes in bytes: `src/nn/nn-network.cpp` 35,404 + `.hpp` 4,046; `nn-core.cpp` 14,406 + hpp 7,936; `nn-executor.cpp` 7,828; `nn-cpu-ops.cpp` 86,042; `nn-cpu.cpp` 8,802; `nn-vulkan.cpp` 53,875 plus about 25 `.comp` shaders; `graphs/llama-graph.cpp` 32,402; `graphs/qwen35-graph.cpp` 37,022; `app.cpp` 14,308; `dllama.cpp` 10,068; `dllama-api.cpp` 14,146 (+ hpp 15,963); `tokenizer.cpp` 25,631; `http.cpp` 8,137; vendored `json.hpp` 919,975 — [tree](https://api.github.com/repos/b4rtaz/distributed-llama/git/trees/main?recursive=1)
- `nn-executor.hpp`: `NnDevice { maxNThreads(); createSegment(segmentIndex) }`, `NnDeviceSegment { loadWeight(opIndex, offset, nBytes, weight); forward(opIndex, nThreads, threadIndex, batchSize) }`, `NnNodeSynchronizer { sync(segmentIndex, nThreads, threadIndex) }` — [nn-executor.hpp](https://raw.githubusercontent.com/b4rtaz/distributed-llama/main/src/nn/nn-executor.hpp)
- `nn-network.hpp`: `NnNetwork::serve/connect`, blocking `write/read(socketIndex, data, size)`, `writeMany/readMany`, `writeAll`, `setTurbo`, stats; `NnRootWeightLoader` streams weight slices from root to workers; `NnRootConfigWriter`/`NnWorkerConfigReader` ship configs — [nn-network.hpp](https://raw.githubusercontent.com/b4rtaz/distributed-llama/main/src/nn/nn-network.hpp)
- Plain TCP with TCP_NODELAY and TCP_QUICKACK; root connects to workers (README: root plus 2^n - 1 workers) — [nn-network.cpp](https://raw.githubusercontent.com/b4rtaz/distributed-llama/main/src/nn/nn-network.cpp), [README](https://raw.githubusercontent.com/b4rtaz/distributed-llama/main/README.md)
- No authentication, token, magic or version handshake and no encryption found in `nn-network.cpp` (grep hits only TCP options and bind errors) — [nn-network.cpp](https://raw.githubusercontent.com/b4rtaz/distributed-llama/main/src/nn/nn-network.cpp)
- Model format: own `.m` plus `.t` tokenizer from converters; Q40 weights only with Q80 sync, or F32; code throws otherwise — [app.cpp](https://raw.githubusercontent.com/b4rtaz/distributed-llama/main/src/app.cpp)
- API: `dllama-api` binary with HTTP server (`dllama-api.cpp`, `http.cpp`) and `examples/chat-api-client.js`; route list not read, so OpenAI compatibility is UNVERIFIED — [README](https://raw.githubusercontent.com/b4rtaz/distributed-llama/main/README.md)
- UI: none (CLI and API). `python launch.py <model>` downloads pre-converted models — [README](https://raw.githubusercontent.com/b4rtaz/distributed-llama/main/README.md)

### Inferences
- Reusable as a library: `src/nn/nn-network.{cpp,hpp}` (about 39 KB) is a dependency-light blocking-TCP helper. It adds little over Strata's own line protocol, so the value is the idea, not the code. `http.cpp` (8 KB) and `tokenizer.cpp` (25 KB) are self-contained.

### Gaps
- Exact all-reduce topology (root star vs other) not confirmed; `NnNetworkNodeSynchronizer::sync` not read.

## 6. distributed-llama: MoE, work split, tensor-parallel limits, hardware

### Takeaway
Verified: Qwen3-30B-A3B runs (CPU since 5 Sep 2025, Vulkan since 16 Sep 2025); user logs show 13.04 tok/s on 4x Raspberry Pi 5 (existing_systems.md, Discussion #255). It is NOT expert-parallel: each expert's matrices are sliced across all nodes (tensor parallel inside every expert) and layer weights are not assigned per node. Node count must divide sliced dimensions (README: powers of two) and cannot exceed the KV-head count (enforced in code).

### Cited Findings
- `llama-graph.cpp` builds MoE buffers `gt` (gate), `act_exp_ix` (active expert indexes), `moe_y/moe_d/moe_l` sized by `nActiveExperts` and per-node slice dims (`n.w1Slice.d0`, `n.w3Slice.d0`); Qwen3 MoE uses `moeHiddenDim` as ffDim — [llama-graph.cpp](https://raw.githubusercontent.com/b4rtaz/distributed-llama/main/src/graphs/llama-graph.cpp)
- `NnRootWeightLoader::loadRowMatmulSlices(opName, opIndex, expertIndex, slice, weight)` and `loadColMatmulSlices` slice each expert matrix across `nNodes` — [nn-network.hpp](https://raw.githubusercontent.com/b4rtaz/distributed-llama/main/src/nn/nn-network.hpp)
- Asserts `d % nNodes == 0`, `n % nNodes == 0`, `kvDim % nNodes == 0` in `sliceRowMatmul/sliceColMatmul/sliceKvCache` — [nn-core.cpp](https://raw.githubusercontent.com/b4rtaz/distributed-llama/main/src/nn/nn-core.cpp)
- `if (nNodes > header->nKvHeads) throw "...does not support more nodes than the number of KV heads"` with a TODO to issue #70 — [app.cpp](https://raw.githubusercontent.com/b4rtaz/distributed-llama/main/src/app.cpp)
- README: "only on 1, 2, 4... 2^n nodes"; max nodes = KV heads; RAM split across nodes, root needs somewhat more — [README](https://raw.githubusercontent.com/b4rtaz/distributed-llama/main/README.md)
- Hardware: Linux/macOS/Windows; ARM and x86-64 AVX2; experimental Vulkan from v0.13.0 (23 Mar 2025); GPU doc `docs/HOW_TO_RUN_GPU.md` — [README](https://raw.githubusercontent.com/b4rtaz/distributed-llama/main/README.md)
- Android/Termux: no official support. Open issues #286 "add APK Android support" and #89 "What about mobile phones?", closed #154 "build error in termux"; an 8-node Snapdragon 8 Gen 2 log is in Discussion #180 — [issue search](https://api.github.com/search/issues?q=repo:b4rtaz/distributed-llama+termux+OR+android), [#180](https://github.com/b4rtaz/distributed-llama/discussions/180)
- Throughput on 1 GbE and 2.5 GbE: Discussions #41, #255, #266 (existing_systems.md).

### Inferences
- The power-of-two rule is not an explicit check; it follows from the divisibility asserts for typical power-of-two head and hidden dims (my reading). A 3-node run would normally fail an assert.
- Slices are equal (`d / nNodes`); I found no speed weighting, so the slowest node sets the pace on heterogeneous hardware.
- Per-layer tensor-parallel syncs explain the need for a wired switch; this is the opposite of what a heterogeneous, latency-heavy old-PC and phone cluster wants.
- Qwen3-30B-A3B has few KV heads (4 from my memory, UNVERIFIED), consistent with the 4-node demos, so the node cap for it would be 4.

### Gaps
- KV head count of Qwen3-30B-A3B not checked in a config file. No new benchmark numbers collected.

## 7. Parallax (GradientHQ): license, activity, architecture, hardware

### Takeaway
Apache-2.0 but quiet: last push 2026-07-01, zero commits in the last 90 days, last release v0.1.2 (2025-12-02). Python on top of SGLang, vLLM and MLX-LM. Its scheduler design (two phases: layer allocation then request routing) is the most reusable and is pure Python. GPU/server oriented; not a fit for old PCs or phones.

### Cited Findings
- Apache-2.0, created 2025-09-22, 1,428 stars, 153 forks, 23 open issues, last push 2026-07-01; 459 commits total (repo page) — [API](https://api.github.com/repos/GradientHQ/parallax), [repo](https://github.com/GradientHQ/parallax)
- Commits since 2026-08-12: 0; contributors first page: 26 — [API](https://api.github.com/repos/GradientHQ/parallax/commits?since=2026-08-12T00:00:00Z)
- Releases: v0.0.1 (2025-09-29), v0.1.0 (2025-11-11), v0.1.1 (2025-11-26), v0.1.2 (2025-12-02) — [releases](https://api.github.com/repos/GradientHQ/parallax/releases)
- Stack: Lattica P2P, SGLang and vLLM for GPU, MLX LM for Mac; `./install.sh` then `parallax serve -m <model>`; Python >= 3.11 and < 3.14 — [README](https://raw.githubusercontent.com/GradientHQ/parallax/main/README.md), [install.md](https://raw.githubusercontent.com/GradientHQ/parallax/main/docs/user_guide/install.md)
- OS: Windows through a desktop app (source/Docker "not recommended"), Linux via source or Docker (images for Blackwell, Ampere, Hopper, DGX Spark), macOS from source on Apple silicon — [install.md](https://raw.githubusercontent.com/GradientHQ/parallax/main/docs/user_guide/install.md)
- No Android, Vulkan or CPU-only path stated in the pages I read (absence, not proof).
- Layout: `src/backend` (main.py, server/, benchmark/), `src/frontend`, `src/parallax` (models, p2p, server, sglang, vllm, utils; launch.py, launch_chat.py, cli.py), `src/parallax/server` (executor/, cache/, sampling/, shard_loader.py, node_chat_http_server.py, radix_cache.py, scheduler.py), `src/scheduling`, `src/router` — [src](https://github.com/GradientHQ/parallax/tree/main/src), [src/parallax](https://github.com/GradientHQ/parallax/tree/main/src/parallax), [server](https://github.com/GradientHQ/parallax/tree/main/src/parallax/server)
- Scheduler (`src/scheduling/`: scheduler.py, layer_allocation.py, request_routing.py, node.py, node_management.py, model_info.py): phase 1 contiguous layer ranges by greedy or DP (maximize k^2/s*(k)) then water-filling by compute and bandwidth under memory limits, rebalance on a coefficient-of-variation threshold; phase 2 per-request min-latency path DP with RTT edges and per-layer latency vertices; pluggable via `BaseLayerAllocator` and `RequestRoutingStrategy`, custom RTT callback; depends on its own `ModelInfo`, `Node`, `NodeHardwareInfo` (TFLOPS, memory size, bandwidth); no MoE mention — [scheduling README](https://raw.githubusercontent.com/GradientHQ/parallax/main/src/scheduling/README.md)
- Model list includes MoE models (Qwen3.6-35B-A3B, Step-3.5-Flash); MoE execution is inside SGLang/vLLM, the scheduler treats layers as dense — [README](https://raw.githubusercontent.com/GradientHQ/parallax/main/README.md)

### Inferences
- Extension point: `src/parallax/server/executor/` plus the `sglang`/`vllm` packages. A Strata backend would be a third executor. Strata does not expose layer-range shards (assumption from the task description), so it would join as a whole-model node; then the two-phase scheduler degenerates to replica routing and adds little.
- The `src/scheduling` package is the piece worth vendoring (Apache-2.0, pure Python, about 8 files; sizes not obtained).

### Gaps
- Line counts of `src/scheduling/*` (tree API rate-limited); Lattica encryption/auth model; whether the frontend is a usable UI; whether the Windows app is open source.

## 8. Better foundation than llama.cpp RPC or mesh-llm for "old PCs + phones"?

### Takeaway
No. prima.cpp offers ideas (Halda, disk-aware windows) but is dormant, CUDA-only, no Windows. distributed-llama offers a readable codebase and an MoE-on-ARM proof but needs a wired switch, 2^n nodes and a KV-head cap. Parallax offers a clean scheduler but is GPU-class Python and dormant. None has a plug-in point for an external engine like Strata.

### Cited Findings
- Cross-comparison with RPC and mesh-llm: see existing_systems.md sections 3 and 4. Constraints per candidate: sections 2, 6, 7 above.

### Inferences
- Phone evidence: only prima.cpp (Termux, Kirin 9000 in paper) and dllama (Snapdragon logs, build issues) have any, and neither ships an official Android build.
- Strata is a whole-model engine behind a text protocol, so the natural integration is request-level routing, which none of the three cores do. Layer-level integration would need Strata to export per-layer forward, a much larger project and in conflict with its expert streaming (UNVERIFIED, based on the task text only).
- Suggested plan: keep layer-split transport on llama.cpp RPC or mesh-llm; add Strata as a request-level worker behind the coordinator; port Halda (as a Python MILP) and optionally Parallax's routing DP into the coordinator.

### Gaps
- No measurement of any candidate with Strata.

## 9. Reusable components (paths, sizes) and effort to host Strata

### Takeaway
Lift ideas, not code. Worthwhile: Halda (about 670 lines tied to llama types) and Parallax scheduling (pure Python, vendorable). dllama's network layer is too thin to matter. Hosting Strata as a worker is cheap at request level and expensive at layer level.

### Cited Findings
- Halda: `common/common.cpp` `assign_layers_to_device`, about lines 860-1530, MIT, HiGHS dependency — [file](https://raw.githubusercontent.com/OpenCPIL/prima.cpp/main/common/common.cpp)
- Device profiler: `common/profiler.cpp` 2,864 lines / 100,454 bytes; `common/profiler.h` 13,213 bytes; MIT — [file](https://raw.githubusercontent.com/OpenCPIL/prima.cpp/main/common/profiler.cpp)
- dllama network layer: `src/nn/nn-network.cpp` 35,404 bytes and `.hpp` 4,046 bytes; MIT — [dir](https://github.com/b4rtaz/distributed-llama/tree/main/src/nn)
- dllama Vulkan backend: `src/nn/nn-vulkan.cpp` 53,875 bytes plus shaders in `src/nn/vulkan/`; MIT; tied to Q40/Q80 and tensor-parallel slices, so not reusable for Strata — [dir](https://github.com/b4rtaz/distributed-llama/tree/main/src/nn/vulkan)
- Parallax scheduler: `src/scheduling/*.py`; Apache-2.0 — [dir](https://github.com/GradientHQ/parallax/tree/main/src/scheduling)

### Inferences (effort estimates, mine, nothing built)
- prima.cpp as host: re-implement the zmq ring and device_info exchange in Strata and track a 1 MB llama.cpp fork, roughly 3,000-6,000 lines, 3-6 weeks; poor value. Porting Halda only: 300-500 lines Python, 2-3 days.
- distributed-llama as host: an `NnDevice` for Strata makes no sense (op-level, Q40 only). A "remote worker" mode forwarding GEN to `strata-poly --serve`: roughly 300-600 lines in `dllama-api.cpp`, 2-4 days, but it turns dllama into a proxy with no benefit.
- Parallax as host: an executor that spawns `strata-poly --serve`, speaks GEN/T/DONE, registers as a whole-model node, plus `ModelInfo` entry and heartbeat: roughly 400-800 lines Python, 3-5 days.
- Coordinator-side (recommended): Strata adapter for GEN/T/DONE, Halda-style MILP, Parallax-style routing DP: about 600-1,000 lines total, 4-6 days.

### Gaps
- All effort numbers are estimates. Strata protocol details come only from the assignment text.
