# Portable compute backends for LLM inference on old and heterogeneous hardware (as of Nov 2026)

Convention: MEASURED = numbers from benchmark posts/papers (user-submitted, unverified hardware config); CLAIMED = vendor or blog assertions. Researched with ~17 tool calls; several items remain gaps (listed per section).

## 1. Measured llama.cpp Vulkan tok/s (Vega/Ryzen APU, Intel iGPU, RX 480/580, GTX 10xx/16xx, phones, A17/A18)

### Takeaway
Vulkan is a real, vendor-neutral path: RX 480/580 reach ~35-40 tok/s tg on a 7B Q4_0, GTX 10xx/16xx 34-70 tok/s, but Ryzen 3000-series APU iGPUs do only ~8.5 tok/s tg (and ~49 tok/s pp), i.e. roughly CPU-class. Phones: Vulkan is unreliable on Adreno/Mali; Apple A-series via Metal is good (Mistral 7B Q4_0 ~9-12 tok/s).

### Cited Findings
All rows below: Llama 2 7B Q4_0, llama-bench, from the Vulkan scoreboard. MEASURED (community-submitted) — [llama.cpp discussion #10879](https://github.com/ggml-org/llama.cpp/discussions/10879)
- RX 480: pp512 248.66 / tg128 34.71 (no FA); 194.52 / 37.23 (FA). RX 580: 258.03 / 39.32. RX 570: 226.11 / 37.44 (no FA), 229.23 / 40.18 (FA).
- GTX 1660 Ti Mobile: 511.67 / 56.60. GTX 1080 Ti: 585.54 / 67.75 (FA 647.81 / 71.58). GTX 1070 Ti: 297.50 / 42.86. GTX 1070: 321.57 / 41.48. GTX 980: 186.24 / 33.90. No GTX 1060 or 1650 rows were found in the fetched content.
- "Ryzen 3000 Series" APU (Vega iGPU; exact SKU not stated): pp512 48.63 / tg128 8.49 (FA: 47.41 / 8.47). Ryzen 4000: 103.87 / 9.63. Ryzen 5000: 90.55 / 10.98. Ryzen 6000 (RDNA2): 240.89 / 21.26. Ryzen 7000: 281.62 / 19.91. Steam Deck: 144.31 / 17.52. Ryzen AI 300: 479 / 22.4. Ryzen AI Max+ 300: 1487 / 51.8.
- Intel: Core 8000 series iGPU 25.43 / 3.35; Core 1000 series 25.58 / 4.25; Core 1100 series (Xe) 187.20 / 10.39; Core Ultra 100 185.51 / 8.21; Core Ultra 200 865 / 24.4. i7-1185G7 Iris Xe (comments, Jan-Jun 2025, ngl 100): pp512 42-106, tg128 5.9-8.0; with ngl 0 (CPU) pp512 73, tg128 5.24 — i.e. iGPU gave little or no tg gain over CPU there.
- Apple (Metal, not Vulkan): iPhone 15 Pro (A17 Pro): TinyLlama 1.1B Q4_0 pp512 646 / tg128 56.9; Phi-2 2.7B Q4_0 157 / 24.7; Mistral 7B Q4_0 80.6 / 9.01. iPhone 16 Pro (A18): TinyLlama Q4_0 647.5 / 69.8; Mistral 7B Q4_0 94.4 / 11.64. MEASURED — [llama.cpp discussion #4508](https://github.com/ggml-org/llama.cpp/discussions/4508)
- Apple Silicon reference (LLaMA 7B Q4_0, Metal): M1 8-core pp512 118 / tg128 14.2; M2 10-core 179.6 / 21.9. MEASURED — [llama.cpp discussion #4167](https://github.com/ggml-org/llama.cpp/discussions/4167)
- Android Vulkan: Adreno 740 (8 Gen 2) showed corrupted output with Vulkan on Qwen3.5 while CPU was correct; reporter says corruption appeared fixed on master Mar 2026 but with a perf regression — [issue #20603](https://github.com/ggml-org/llama.cpp/issues/20603). 2024 discussion: Vulkan on phones very slow; Adreno failed to load the model, Mali loaded — [discussion #9464](https://github.com/ggml-org/llama.cpp/discussions/9464). PowerVR k-quant Vulkan shaders gave incorrect results — [Imagination forum](https://forums.imgtec.com/t/powervr-dxt-48-1536-incorrect-results-from-k-quant-vulkan-compute-shaders-in-llama-cpp/4291). A Reddit claim (CLAIMED/anecdotal, Feb 2026) says MNN-OpenCL beats llama.cpp Vulkan on Mali and Adreno — [reddit snapshot](https://reddit.sentinel-team.org/posts/1r9aay0/snapshots/2026-02-27T15%3A04%3A59.533315Z).
- Adreno OpenCL backend (Qualcomm): optimized for Q4_0 on Snapdragon 8 Gen 1/2/3/Elite; docs list Adreno 750 and recommend `llama-quantize --pure`. CLAIMED — [Qualcomm blog](https://www.qualcomm.com/developer/blog/2024/11/introducing-new-opn-cl-gpu-backend-llama-cpp-for-qualcomm-adreno-gpu), [OpenCL doc](https://huggingface.co/hekod19045/llama-cuda/blob/main/docs/backend/OPENCL.md). IWOCL 2025 slide decode chart (model/quant not named in snippet): ~5.3->6.4 tok/s on 8 Gen 2, ~11.2->14.2 on 8 Gen 3 (baseline->improved) — [IWOCL slides](https://www.iwocl.org/wp-content/uploads/iwocl-2025-hongqiang-wang-lamacpp-backend-update.pdf). Community: Adreno 750 Phi-4-mini Q8_0 8.9 tok/s (not Q4_0); a single developer anecdote says Q8_0 was faster than Q4_0/Q6_K on Adreno OpenCL (contradicts Qualcomm's Q4_0 focus).
- Hexagon NPU backend exists in llama.cpp (docs/backend/hexagon) — [README](https://sources.debian.org/src/llama.cpp/7593%2Bdfsg-3/docs/backend/hexagon/README.md); not benchmarked here.

### Inferences
- A Ryzen 3200G/3400G (Vega 8/11, DDR4) is likely in the same bracket as the "Ryzen 3000 Series" row (~8-9 tok/s tg on 7B Q4_0), which is about what a decent AVX2 CPU does; Vulkan on it is mostly a pp (prefill) and CPU-offload benefit, not a decode win. Verify with llama-bench on the real SKU (3000-series row may be a mobile 3500U).
- Old discrete cards (RX 580, GTX 1070-class) are the sweet spot: 35-45 tok/s on 7B Q4_0, provided 8 GB VRAM.
- For phones, Metal (iOS) is solid; Android needs Adreno OpenCL (Snapdragon only) or CPU; Vulkan on Android is risky for correctness.

### Gaps
- No measured numbers found for Vega 8/11 specifically (3200G/3400G), GTX 1060, GTX 1650, or Snapdragon Q4_0 in llama-bench form (search for the discussion's raw tables was not exhaustive).
- Discussion #4508 has no iPad rows in fetched content.

## 2. iGPU memory: UMA/GTT/Vulkan heap, and whether it beats CPU-only

### Takeaway
iGPU capacity is bounded by BIOS UMA carve-out, kernel GTT settings and per-allocation driver caps; bandwidth is shared with the CPU so decode gains are small. Evidence for Vega-specific limits is thin.

### Cited Findings
- BIOS can set a fixed UMA frame buffer (e.g. 8G/16G) separate from GTT; GTT (dynamic system RAM) can be raised via amdgpu/ttm kernel params, and llama.cpp reportedly could use the full defined GTT (56 GB reported on a Framework/Strix Halo setup) — [Framework thread](https://community.frame.work/t/tuning-the-framework-shared-memory-being-blocked/79441), [guide](https://www.faceofit.com/the-apu-guide-to-llms/).
- Per-allocation limits can bite: AMDVLK reported "requested buffer size exceeds device memory allocation limit" while RADV worked; Mali G68 32-bit hit max_buffer_size — [llama.cpp issues search results](https://github.com/ggml-org/llama.cpp/issues/15054), [reddit](https://reddit.sentinel-team.org/posts/1sb5jmv/snapshots/2026-04-03T21%3A20%3A24.375333Z).
- Measured: Ryzen 3000 iGPU tg 8.5 vs Ryzen 7000 (DDR5, RDNA3) ~20 and Iris Xe tg ~7.3 (GPU) vs 5.2 (CPU) — [discussion #10879](https://github.com/ggml-org/llama.cpp/discussions/10879). Strix Halo CPU-only ~9 vs Vulkan 17-19 tok/s on one (different) model — from search summary only.
- Arithmetic: tg is bandwidth-bound; DDR4-3200 dual-channel ~51 GB/s theoretical, so a 3.8 GB Q4_0 7B caps at ~13 tok/s; observed 8.5 is ~65% of that.

### Inferences
- iGPU helps prefill and frees CPU cores, but cannot beat the RAM-bandwidth ceiling; for Ryzen 3000 APUs plan CPU or iGPU as roughly equivalent for decode and target 3B-class or smaller (Q4) models for usable speed.
- Use dual-channel RAM; single-channel halves the ceiling.

### Gaps
- No Vega-specific heap size/UMA data found; recommend `vulkaninfo` and llama.cpp startup log on target machines.

## 3. Quantisation formats and Vulkan kernels

### Takeaway
Q4_0/Q8_0 and k-quants have mature Vulkan mat-vec and integer-dot (dp4a/MMQ) paths; coopmat helps only on supporting GPUs (recent NVIDIA, RDNA3+, Intel Xe). IQ-quants have Vulkan kernels but I found no end-to-end benchmark data.

### Cited Findings
- Vulkan has cooperative-matrix support (PR #10597), DP4A MMQ with Q8_1 quantization (#12135), q4_0 x q8_1 int-dot and later k-quant integer-dot, per-vendor tuning (Intel Xe2 uses NVIDIA-style MMVQ override for k-quants) — [commit #10597](https://cdn04132025.gitlink.org.cn/replica/llama.cpp/commit/3df784b3050f657ea681f804187ce5bddb433e88), [commit #12135](https://cdn04132025.gitlink.org.cn/replica/llama.cpp/commit/a8a1f3356786cbf8bcc3422e3c8737fc33b453e7), [PR #11521](https://github.com/ggml-org/llama.cpp/pull/11521).
- iq4_xs matmul microbenchmark +16.7% on RTX 4060 Ti (kernel-level, not tok/s) — [PR #16203](https://github.com/ggml-org/llama.cpp/pull/16203) (as returned by search).
- Q6_K pp512 regression on RTX 3090 with coopmat+int-dot (2428 -> 2062 t/s) — same search results; cause unknown.
- Adreno OpenCL is Q4_0-centric — [Qualcomm blog](https://www.qualcomm.com/developer/blog/2024/11/introducing-new-opn-cl-gpu-backend-llama-cpp-for-qualcomm-adreno-gpu).
- Apple Metal Q4_0 tg ~1.6-1.7x Q8_0 on A17/A18 (TinyLlama 56.9 vs 35.6) — [#4508](https://github.com/ggml-org/llama.cpp/discussions/4508).

### Inferences
- Simplest safe default across all targets: Q4_0 (and Q4_K_M where the shader path is mature); avoid IQ2/IQ3 on old GPUs/phones where dequant compute dominates (unverified here).
- Older GPUs lack coopmat/dp4a, falling back to generic shaders; expect no prefill boost.

### Gaps
- No direct Q4_K vs Q4_0 vs IQ tok/s comparison on Vulkan found.

## 4. Other backends: ROCm/HIP limits, CUDA minimums, WebGPU, CPU

### Takeaway
ROCm no longer covers Polaris/Vega/RDNA1 officially (only RDNA2+ pro and RDNA3/4 consumer); CUDA 13 dropped Pascal offline compilation, so old NVIDIA needs CUDA 12.x or Vulkan. Vulkan is therefore the only practical single backend for old AMD.

### Cited Findings
- ROCm 7.2.4 Linux requirements: supported Radeon = RX 9000, RX 7000 (7700-7900), PRO W7000/W6800/V620 (RDNA2 PRO only); RX 6000 and RX 5000 not listed; Polaris gfx803 not listed; gfx900 (MI25) and gfx906 (MI50, Radeon VII, PRO VII) marked unsupported; unlisted GPUs "not officially supported", may work via TheRock nightlies; Ubuntu 24.04.4/22.04.5, RHEL 9.7/10.1 only — [ROCm 7.2.4 system requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.4/reference/system-requirements.html), [GPU/OS support](https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html). The page does not cover Windows or Ryzen APU Vega.
- CUDA 13.0 removed offline compilation and library support for sm 5.x/6.x/7.0/7.2 (Maxwell, Pascal, Volta); flagged in CUDA 12.9 notes; apps built with CUDA 12.x still run; NVIDIA recommends CUDA 12.9 for pre-7.5 offline compilation and driver branch 580 (support through ~2028) — [CUDA features archive](https://docs.nvidia.com/cuda/cuda-features-archive/index.html), [Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/nvidia-to-drop-cuda-support-for-maxwell-pascal-and-volta-gpus-with-the-next-major-toolkit-release), [Guru3D](https://www.guru3d.com/publish/comments/nvidia-gpu-driver-changes-maxwell-pascal-and-volta-unsupported-after-release-580-series/). Third-party summaries describe this more broadly than NVIDIA's wording (offline compilation/libraries).
- WebLLM: paper claims up to 80% native; MLC blog claims up to 85% vs native Metal (M3 Max); a blog cites 41 tok/s Llama 3.1 8B on M3 Max (~71-80% of native) — [WebLLM arXiv](https://arxiv.org/abs/2412.15803v1), [MLC blog](https://blog.mlc.ai/2024/06/13/webllm-a-high-performance-in-browser-llm-inference-engine), [pinggy](https://pinggy.io/blog/run_llm_in_browser_webgpu/). CLAIMED, not independently measured here.
- wllama is CPU WASM GGUF and needs COOP/COEP headers for threads; WASM heap is grow-only; a 2 GB wllama cap could not be confirmed — [pinggy](https://pinggy.io/blog/run_llm_in_browser_webgpu/), [tianpan](https://tianpan.co/blog/2026-04-17-browser-native-llm-inference-webgpu).
- A llama.cpp WebGPU backend exists (article "ceiling measured across 16 devices") but the page could not be fetched — [youngju.dev](https://www.youngju.dev/transcribe/2026-07-16-llama-cpp-webgpu-browser-inference.en); status/numbers unverified.
- Vulkan vs ROCm: Phoronix finds Vulkan works across vendors and is competitive — [Phoronix Lemonade](https://www.phoronix.com/review/lemonade-vulkan-rocm/3); one blog claims Vulkan ~21% faster tg than ROCm on an AMD setup.

### Inferences
- Pascal (GTX 10xx, sm_61) via llama.cpp CUDA needs a CUDA 12.x build (ship 12.x binaries) or use Vulkan (measured 40-70 tok/s above, probably slower than CUDA but one codebase).
- ROCm should be treated as optional for RDNA3+ only; do not depend on it for Polaris/Vega/RDNA1/RDNA2 consumer cards.
- Browser path (WebLLM) is good zero-install for small (<=3-8B) models on WebGPU-capable devices; limited by buffer limits and download size.

### Gaps
- ROCm Windows/HIP SDK support for RX 6000/5000 and community builds (e.g. rocBLAS gfx803 patches) not researched. CUDA 12.x minimum toolkit/driver details for sm_61 not fetched.
- Nov 2026 CPU AVX2/NEON numbers not collected.

## 5. iOS and Android constraints

### Takeaway
iOS per-app memory is a hard ceiling well below RAM (reports ~6 GB flat on recent iPhones even with the entitlement); Android needs a foreground service and models heavy thermal throttling. Both are best treated as opportunistic, small-model clients.

### Cited Findings
- `com.apple.developer.kernel.increased-memory-limit` may raise the limit only on some models; Apple says to handle absence — [Apple docs](https://developer.apple.com/tutorials/data/documentation/bundleresources/entitlements/com.apple.developer.kernel.increased-memory-limit.md). One developer reports ~6144 MB cap on a 12 GB iPhone with the entitlement — [Apple forums](https://developer.apple.com/forums/thread/834805). Other reports: no change on iPhone 8 / iOS 15 — [forum](https://developer.apple.com/forums/thread/740831); no effect on WebKit WebContent process; iPad dev builds reached ~15 GB but not App Store builds (anecdotal).
- Thermal: an arXiv 2026 sustained-inference benchmark says phones are limited mainly by thermal management; S24 Ultra and iPhone throttle rapidly (single model/prompt) — [arXiv 2603.23640](https://arxiv.org/abs/2603.23640). Blog numbers (unverified): 8 Gen 2 7B 18 -> 11-13 tok/s in 90-120 s; S24 ~15 tok/s for ~4 min then throttle — [dev.to](https://dev.to/software_mvp-factory/thermal-throttling-and-sustained-on-device-llm-inference-on-android-4nh5).
- Android foreground services raise priority and need a persistent notification but process can still be killed under memory pressure — [Android Developers Blog](https://android-developers.googleblog.com/2018/12/effective-foreground-services-on-android_11.html).
- `PowerManager.getThermalHeadroom()` (API 31+) can be used to adapt (blog claim).

### Inferences
- iOS: budget ~3-5 GB for model+KV on 8 GB devices; no meaningful background execution for GPU/Metal work (iOS suspends; Metal work from background is not allowed - from general knowledge, not verified in this research); App Store/sideload rules not researched.
- Treat phones as short-burst workers (small model, e.g. 1-3B Q4) rather than steady nodes.

### Gaps
- No sources found on iOS background execution limits, App Store/sideload/JIT rules, Android 12+ phantom-process killer, or Android per-app memory limits.

## 6. Pragmatic minimum plan

### Takeaway
One llama.cpp-based worker process with Vulkan + CPU builds (plus CUDA 12 optional) speaking a network protocol covers desktops/old GPUs/APUs; phones and browsers join via separate native paths speaking the same protocol.

### Cited Findings
- llama.cpp Vulkan runs across AMD/NVIDIA/Intel and was competitive in Phoronix tests — [Phoronix](https://www.phoronix.com/review/lemonade-vulkan-rocm/3); measured data above show it works on Polaris, Pascal, Turing, Vega iGPU, Intel Xe.
- Metal has strong measured phone/Mac numbers — [#4508](https://github.com/ggml-org/llama.cpp/discussions/4508).
- Adreno OpenCL for Snapdragon — [Qualcomm blog](https://www.qualcomm.com/developer/blog/2024/11/introducing-new-opn-cl-gpu-backend-llama-cpp-for-qualcomm-adreno-gpu).

### Inferences
- Proposed tiers: (a) desktop worker: llama.cpp with Vulkan build (default) + CPU fallback (AVX2) in a separate process, Q4_0/Q4_K_M GGUF; optionally CUDA 12.x build for GTX 10xx and ROCm only for RDNA3+. Select backend at startup by probing (`llama-bench` smoke test); fall back to CPU on Vulkan correctness failures (observed garbage-output bugs on Adreno, Intel Arrow Lake iGPU #19327, PowerVR).
  (b) iOS/Mac: native llama.cpp Metal app, small models; (c) Android: Adreno OpenCL on Snapdragon, CPU NEON elsewhere, Vulkan only after per-device validation; (d) browser: WebLLM (WebGPU) for zero-install small models, wllama as CPU fallback.
- Expose OpenAI-compatible/llama-server-style HTTP or a thin RPC; workers advertise capabilities (backend, free memory, measured tok/s) and a coordinator routes by model size. llama.cpp RPC backend not researched.
- Because old iGPUs are bandwidth-limited, route only small models (<=3B) or prefill-heavy jobs to them.

### Gaps
- Distribution/splitting across devices (llama.cpp RPC, latency over network) was out of scope and not researched.
