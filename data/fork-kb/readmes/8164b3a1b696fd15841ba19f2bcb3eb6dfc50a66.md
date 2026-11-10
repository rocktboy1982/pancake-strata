<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h2 align="center">Strata-Server</h2>
<h3 align="center">Large models. Your hardware. Native CPU and GPU execution.</h3>
<p align="center">Swift Flash Next, Swift 27B and DeepSeek — native model backends and measured local performance.</p>

<p align="center">
  <a href="#what-is-strata-server">Overview</a> ·
  <a href="#models">Models</a> ·
  <a href="#benchmarks"><b>Benchmarks</b></a> ·
  <a href="#test-hardware">Hardware</a> ·
  <a href="#build-and-run">Build</a> ·
  <a href="#guides">Guides</a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/github/license/XopMC/Strata-Server" alt="MIT license"></a>
  <a href="https://huggingface.co/XopMC513"><img src="https://img.shields.io/badge/Hugging_Face-Models-FFD21E" alt="Our models on Hugging Face"></a>
  <a href="https://github.com/Niko1221/Strata"><img src="https://img.shields.io/badge/Based_on-Strata-0ab887" alt="Based on Strata"></a>
</p>

**Free, open source, and yours to run.** Strata-Server brings our native Swift and DeepSeek backends into one source checkout, with explicit choices for the model, GPU count and context. It is built on [Strata](https://github.com/Niko1221/Strata); the original app, Terminal and documentation remain available.

## What is Strata-Server?

The focus is large Mixture-of-Experts models on a CPU with plenty of RAM and one or more NVIDIA GPUs. Keep experts in system memory, retain useful weights on the GPUs, and give each part of the machine a defined role.

| What is included | What it does |
| --- | --- |
| **Native Swift Flash Next and DeepSeek providers** | Keep the matching loaders, kernels, attention state and speculative decoding for each model. |
| **One build entry** | Select the model family, GPU count and context with CMake flags. Numerical implementations keep their own matching headers and ABI. |
| **Shared expert RAM** | Compatible native workers can attach to one sealed, read-only expert bank instead of loading the expert payload again. Request KV and recurrent state remain separate. |
| **Required native P2P** | Communicating GPUs use direct GPU-resident transport. Startup refuses unavailable peer access; it does not silently switch to host staging. |
| **NVFP4 execution on Ada** | Custom CPU and GPU paths run the selected mixed NVFP4 package on RTX 4090. Ada does not acquire Blackwell FP4 Tensor Cores from a weight format. |
| **Verified MTP** | The target model checks draft tokens and owns acceptance and rollback. Draft speed is not presented as final generation speed. |
| **Fresh device and memory discovery** | GPU indices resolve at launch. NUMA interleave follows current allowed memory nodes, including changed NPS layouts, rather than a saved UUID or two-node list. |
| **Measured source reconstruction** | The two-GPU Swift provider has been rebuilt from source and checked against the original backend on a filled 512K context. |

All-expert, full-attention comparison profiles are retained. The selected source profiles do not introduce expert skipping, a shorter attention window or a lower-bit replacement for the published weights. Useful experiments and architecture work remain separate from qualified defaults.

## Models

Our checkpoints are published under **[XopMC513 on Hugging Face](https://huggingface.co/XopMC513)**. Open a model below for its formats, backend, measurements and build commands. Weights are separate downloads; this repository contains source, profiles and documentation.

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Swift 1.5 Qwen3.8 Flash Next Heretic</b></summary>

**[Download on Hugging Face →](https://huggingface.co/XopMC513/Swift-1.5-Qwen3.8-Flash-Next-Heretic)**

Our gentle-Heretic Swift is available as original-precision weights, **Q4_K_M**, and a **custom mixed NVFP4** package. The native work covers CPU experts, one shared RAM bank, GPU expert residency, prefill, attention and target-verified MTP.

### Swift performance

These are our local, single-request measurements. New-prompt speed counts newly encoded input. Long rows use genuinely filled contexts; output is 1,024 real generated tokens.

| GPUs | New input / allocated context | Generation | New prompt | Recorded profile |
| ---: | ---: | ---: | ---: | --- |
| **1** | 260,000 / 262,144 | **173.26 tok/s** | **5,297 tok/s** | Historical adaptive, median of 3 |
| **1** | 520,000 / 524,288 | **181.79 tok/s** | **4,752 tok/s** | Historical adaptive, median of 3 |
| **1** | 1,040,000 / 1,048,576 | **175.29 tok/s** | **3,833 tok/s** | Historical adaptive, median of 3 |
| **2** | 260,000 / 262,144 | **209.04 tok/s** | **9,005 tok/s** | Original shared-bank profile |
| **2** | 520,000 / 524,288 | **192.34 tok/s** | **9,265 tok/s** | **Unified source rebuild** |
| **2** | 1,040,000 / 1,048,576 | **187.64 tok/s** | **7,753 tok/s** | Original selected V454/V458 |
| **3** | 260,000 / 262,144 | **177.89 tok/s** | **11,457 tok/s** | Original prefill profile V444 |
| **3** | 520,000 / 524,288 | **185.38 tok/s** | **12,321 tok/s** | Original prefill profile V444 |
| **3** | 1,040,000 / 1,048,576 | **173.90 tok/s** | **10,778 tok/s** | Historical three-dense V55 |
| **4** | 260,000 / 262,144 | **169.18 tok/s** | **5,939 tok/s** | Experimental shared-bank control |
| 4 | — / 524,288 | — | — | No completed result in the audited set |
| 4 | — / 1,048,576 | — | — | No completed result in the audited set |

The table covers **1–4 GPUs and 256K / 512K / 1M**. One-GPU rows are historical medians of three runs; the other numerical rows are individual completed runs. The profiles and test epochs differ, so this is an inventory of recorded versions, **not a controlled GPU-scaling curve**. A historical 1M run does not qualify a newly rebuilt 1M profile. A dash means no completed result in this receipt set, not zero performance.

**Other preserved trade-offs:**

| GPUs | New input / allocated context | Generation | New prompt | Alternative profile |
| ---: | ---: | ---: | ---: | --- |
| 3 | 260,000 / 262,144 | 205.04 tok/s | 8,725 tok/s | Original generation-oriented profile |
| 4 | 260,000 / 262,144 | 169.38 tok/s | 5,997 tok/s | Later four-GPU V28 experiment, not selected |

The three-GPU generation-oriented profile is faster at generation, while the separate prefill profile reads new prompts faster. They are not one combined configuration. The later four-GPU result remains experimental; more cards did not automatically improve generation.

[Full 1–4 GPU receipt table, including TTFT →](docs/data/owned-swift-1-4-gpu-results.json)

**Small fill in a long allocation:** the unified two-GPU build measured **225.05 generation tok/s** and **3,298 new-prompt tok/s** at **16,384 / 524,288** tokens. This was a fresh short request in the same process, not a filled 512K result.

### Swift source profiles

| GPUs | Context choices | Current qualification |
| ---: | --- | --- |
| 1 | 256K / 512K / 1M | Context-specific source imported; rebuilt-model qualification pending |
| 2 | 512K / 1M | Source build and filled-512K parity checked; rebuilt 1M qualification separate |
| 3 | 256K / 512K | Prefill-oriented source imported; rebuilt-model qualification pending |
| 4 | Experimental | Requires `STRATA_OWNED_EXPERIMENTAL=ON`; not a production winner |

The final two-GPU source build matched the original control's **initial logits, prefill geometry and all 1,024 output token IDs**, on both 520,000-token and 16,384-token inputs. Expert cache maps stayed the same; the shared bank was reused with **zero expert payload bytes reloaded**. Shutdown drained captured resource pools.

The long-fill control measured 192.22 generation / 9,244 new-prompt tok/s; the rebuild measured 192.34 / 9,265. This establishes reconstruction parity on these tests; the small timing difference is not promoted as an optimization gain.

### Build Swift

After cloning the repository, build the selected two-GPU provider with the qualified toolchain:

```sh
cmake -S owned -B build-owned -G Ninja \
  -DCMAKE_C_COMPILER=clang-18 \
  -DCMAKE_CXX_COMPILER=clang++-18 \
  -DCMAKE_CUDA_HOST_COMPILER=clang++-18 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.4/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DSTRATA_OWNED_FAMILY=swift \
  -DSTRATA_OWNED_GPU_COUNT=2 \
  -DSTRATA_OWNED_CONTEXT=524288

cmake --build build-owned --target strata_owned_swift -j8
```

Provide compatible existing model assets, then inspect the native launch plan:

```sh
python3 owned/launch.py \
  --profile swift-nvfp4-two --gpus 0,1 --context 512k \
  --pack /models/swift/native-pack-experts-v1 \
  --common /models/swift/common.gguf \
  --mtp /models/swift/mtp-production-q4 \
  --expert-profile /models/swift/expert-profile.bin \
  --plan
```

`--gpus` selects current NVIDIA system indices. `--plan` loads no CUDA or weights. `--native-server` selects the native stdin protocol; it is not an HTTP listener. A valid `--shared-expert-arena` reuses a sealed expert bank, not request KV or a pre-reboot bank.

[Swift formats and assets](docs/XOPMC_MODELS.md) · [Full measurements](docs/XOPMC_RESULTS.md) · [Source qualification](owned/swift/QUALIFICATION.json) · [Profile registry](owned/profiles/README.md)

</details>

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Swift 1.5 Qwen3.8 27B Heretic</b></summary>

**Our dense 27B variant**, derived from [ukisai/Swift-1.5-Qwen3.8-27b](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b), with its own gentle-Heretic weights and separate optimized native backend.

### Swift 27B formats and backend

| Item | Prepared version |
| --- | --- |
| Original-precision model | Retained **BF16 Heretic** checkpoint |
| Native model | **Q4_K_M**, unchanged during runtime optimization; approximately 18.20 GB, with separate BF16 vision projection |
| Apple format | **MLX 4-bit**, plus native MTP assets and a separate DFlash2 head; strict CPU reload checked, no Apple inference speed claim |
| Selected draft | **DFlash2 Q4**, stock head at 128K and Marathon R32 at 256K, with target verification |
| Attention cache | **Q4_0** in the recorded selected runs |
| Contexts | **128K / 256K** |

The native work includes async packed-Q4 KV prefetch, pairwarp GDN, a compact exact recurrence/rollback journal, fused FFN prefill and separate Main/draft memory budgets. Native same-checkpoint MTP and other draft heads were compared; DFlash2 remained selected on these measured workloads.

### Swift 27B performance

Original selected v19 / API r9 backend, **CPU + one RTX 4090**, one coding request, 1,792 generated tokens:

| GPUs | New input / allocated context | Generation | New prompt | Draft |
| ---: | ---: | ---: | ---: | --- |
| 1 | 129,024 / 131,072 | **122.28 tok/s** | **2,352 tok/s** | DFlash2 Q4 |
| 1 | 260,096 / 262,144 | **108.27 tok/s** | **1,924 tok/s** | DFlash2 Marathon R32 Q4 |

These are fresh, filled-context runs, not sums of parallel requests. Same-geometry output checks against the preceding selected backend passed; these checks do not establish a broad model-quality benchmark.

### Independent per-GPU copies

The original local deployment also supported independent **128K** instances on individual cards, with per-GPU replacement and a two-hour idle TTL. The following historical tests loaded and unloaded each replica **sequentially** on the then-installed PCIe 2.0 ×1 cards. These indices describe the 17 October setup, not a permanent binding to today's cards.

| Historical GPU | New input / allocated context | Generation | New prompt |
| ---: | ---: | ---: | ---: |
| 1 | 129,024 / 131,072 | 101.23 tok/s | 1,097 tok/s |
| 2 | 129,024 / 131,072 | 100.27 tok/s | 1,092 tok/s |
| 3 | 129,024 / 131,072 | 101.10 tok/s | 1,098 tok/s |
| 4 | 129,024 / 131,072 | 100.50 tok/s | 1,093 tok/s |
| 5 | 129,024 / 131,072 | 100.71 tok/s | 1,096 tok/s |
| 6 | 129,024 / 131,072 | 100.73 tok/s | 1,097 tok/s |

These are separate model instances. They are **not** one Swift 27B request accelerated by 2–4 GPUs, and their rates are not added together.

The original local API profiles are `swift27-128k`, `swift27-256k` and `swift27-gpu1-128k` through `swift27-gpu6-128k`. The common `owned/` source selector does **not yet include this separate 27B runtime**; these names describe the retained original deployment, not a ready generic build target in this fork. Own 27B Heretic artifacts are not among the two currently published XopMC513 model repositories; the link above is the unmodified base model.

[Swift 27B backend and formats →](docs/XOPMC_SWIFT27.md) · [Recorded measurements →](docs/data/owned-swift27-results.json)

</details>

<details>
<summary><img src="docs/media/models/deepseek.svg" width="22" alt="" align="middle"> &nbsp;<b>DeepSeek V4.1 Flash 1017 Heretic</b></summary>

**[Download on Hugging Face →](https://huggingface.co/XopMC513/DeepSeek-V4.1-Flash-1017-Heretic)**

Our gentle-Heretic DeepSeek is available as an **original-precision checkpoint** and **mixed NVFP4-based quantization**. The separate native backend covers compressed attention, CPU/GPU experts and target-verified speculative decoding.

Original precision preserves the checkpoint's tensor formats, including FP8/MXFP4 where present. It is not an all-BF16 model. The custom mixed package requires its matching loader; a file extension alone does not establish compatible arithmetic.

### DeepSeek performance

Our original two-GPU native API profile, one request with 512 real output tokens:

| Backend | GPUs | New input / allocated context | Generation | New prompt |
| --- | ---: | ---: | ---: | ---: |
| DeepSeek 1017 mixed, original native API | 2 | 261,568 / 262,144 | **54.69 tok/s** | **4,936 tok/s** |
| DeepSeek 1017 mixed, original native API | 2 | 523,712 / 524,288 | **56.64 tok/s** | **5,590 tok/s** |

These are filled-context measurements of the original native implementation. They are not new measurements of the reconstructed source provider.

### DeepSeek source profile

| GPUs | Context choices | Current qualification |
| ---: | --- | --- |
| 1 | Separate original runtime | Single-GPU work retained locally; not imported into the current public selector |
| 2 | 256K / 512K | Source checks passed; clean build and common serving-factory qualification pending |
| 3 | — | No completed profile in the published evidence set |
| 4 | — | No completed profile in the published evidence set |

The imported provider retains matching numerical headers, original per-source arithmetic flags and explicit current-device locality checks. Its unqualified common serving factory remains disabled.

### Build DeepSeek

After cloning the repository, choose the DeepSeek family and one context:

```sh
cmake -S owned -B build-deepseek -G Ninja \
  -DCMAKE_C_COMPILER=clang-18 \
  -DCMAKE_CXX_COMPILER=clang++-18 \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.4 \
  -DSTRATA_OWNED_FAMILY=deepseek \
  -DSTRATA_OWNED_GPU_COUNT=2 \
  -DSTRATA_OWNED_CONTEXT=262144

cmake --build build-deepseek --target strata_deepseek_classic -j8
```

Use `524288` for the 512K source variant. This builds native source; it does not enable the unqualified serving factory or claim parity of a newly built binary.

[DeepSeek formats and assets](docs/XOPMC_MODELS.md) · [Full measurements](docs/XOPMC_RESULTS.md) · [Native source status](owned/deepseek/STATUS.md) · [Native C ABI](owned/deepseek/C_ABI.md)

</details>

## Benchmarks

All model-specific tables are inside the model blocks above. Measurements use the dual-EPYC / RTX 4090 server below, **new input tokens**, one request and the stated actual fill. Individual long-context runs and historical single-GPU medians are identified separately. Component bandwidth is not model throughput, and these results are not guarantees for another machine.

[Full results, settings and limits →](docs/XOPMC_RESULTS.md)

## Test hardware

Snapshot: **28 October 2026**. Hardware indices are dated observations, not permanent allowlists.

| Component | Development server |
| --- | --- |
| CPU | **2 × AMD EPYC 9754** — 256 physical cores, 512 logical threads |
| Memory | **24 × 32 GB DDR5**, 12 populated channels per socket; all DIMMs report **4,800 MT/s configured speed** |
| Memory visible to Linux | **810.996 GB / 755.299 GiB** |
| GPUs | **5 × GeForce RTX 4090**, 24,564 MiB reported per card |
| GPU links | Current GPU0–3: **PCIe 4.0 ×16**; GPU4: **PCIe 4.0 ×4** |
| Locality | GPU0/1 on socket 0; GPU2/3/4 on socket 1; cross-socket traffic traverses the CPU interconnect |
| GPU transport | Native P2P-capable Ada setup; **no NVLink** |
| GPU power limit | 500 W reported; no power change was made for these measurements |
| Software | Linux 7.0.0-38, NVIDIA 615.71.09, CUDA 13.4, **Clang 18 + mold**, SM89 |

### Measured bandwidth

| Path or workload | Result | Meaning |
| --- | ---: | --- |
| **System RAM, read-only peak** | **699.33 GB/s** | Actual load-byte throughput, locally placed test buffers with hugepage advice |
| **System RAM, tuned STREAM Triad** | **683.47 GB/s** | Validated logical read/write throughput with non-temporal stores |
| System RAM, theoretical aggregate ceiling | 921.6 GB/s | 2 sockets × 12 channels × 4,800 MT/s × 8 bytes; not a sustained application result |
| **GPU0 → GPU1 native P2P** | **26.26 GB/s** | 32 MiB blocks, sequential one-way test |
| **GPU1 → GPU0 native P2P** | **26.25 GB/s** | Same method, reverse direction |
| Pinned host → GPU0 / GPU1 | 26.83 / 26.84 GB/s | Measured host transfers, not system RAM bandwidth |
| RTX 4090 local VRAM, specification | 1,008 GB/s | Theoretical GPU-local bandwidth, not PCIe or peer-copy speed |

RAM tests searched thread counts, L3-domain placement and NUMA policy; the largest observed values are not a proof of an absolute hardware maximum. The copy tests and RAM tests are component measurements, not model token rates. Methods, repeats, variability and sanitized receipts are linked below.

[Hardware, NUMA topology and bandwidth methods →](docs/XOPMC_HARDWARE.md)

## Build and run

The owned providers are opt-in. Start with the shared checkout:

```sh
git clone https://github.com/XopMC/Strata-Server.git
cd Strata-Server
```

Open the matching model block above for its build commands, GPU/context choices and qualification. Native source profiles use **Clang 18 + mold**, CUDA 13.4 and SM89. Model assets remain separate. Building the upstream project alone does not select an owned native provider.

The shared peer-policy API handles up to seven selected compatible devices. This does not mean seven-GPU model execution has been implemented or measured. Unqualified launch profiles are disabled; unsupported choices fail explicitly. Integration of every owned provider with a common HTTP factory remains separate from the retained upstream app and APIs.

After a BIOS NPS change, CPU placement and memory admission must be checked again. The selector discovers current allowed memory nodes instead of assuming `0,1` is the whole machine. Shared RAM seals and request state do not survive reboot.

[Complete source build and runtime contract →](owned/README.md) · [Profile registry →](owned/profiles/README.md) · [P2P admission →](docs/XOPMC_P2P.md)

## Research and retained work

The upstream audit covered **367 commits and 534 changed paths** through Strata 0.2.7. Related work includes Project Maya, 1Cat-vLLM and Uzu. Useful directions include shape-specific kernels, compact activation exchange, exact communication overlap, speculative-tree budgets and explicit recurrent-state commit rules.

Candidates are not silently enabled because another project reports a speedup. Numerical order, supported formats, graph lifetimes and actual whole-model results still matter. Uzu is primarily Metal/Apple work; it does not provide a ready Ada CUDA NVFP4/P2P backend.

[Upstream review and adjacent-project findings →](docs/XOPMC_RESEARCH.md)

## Guides

[Model downloads](docs/XOPMC_MODELS.md) · [Native results](docs/XOPMC_RESULTS.md) · [Hardware and RAM bandwidth](docs/XOPMC_HARDWARE.md) · [Build flags](owned/README.md) · [Native P2P](docs/XOPMC_P2P.md) · [Source status](owned/profiles/README.md)

<details>
<summary><b>Original Strata app, Terminal and platform guides</b></summary>

The original Strata features and platform documentation are preserved. Upstream release packages do not include these newly reconstructed owned native providers.

- [Original Strata releases](https://github.com/Niko1221/Strata/releases/latest)
- [Desktop installation](docs/INSTALL.md) · [Terminal guide](docs/TERMINAL_LAUNCH.md)
- [OpenAI- and Anthropic-compatible API connections](docs/TERMINAL_LAUNCH.md#connect-your-apps-strata-connect)
- [Original source installation](docs/SOURCE_INSTALL.md) · [Docker](docs/DOCKER.md)
- [AMD](docs/AMD_HIP.md) · [Intel Arc](docs/INTEL_ARC.md) · [Apple Silicon](docs/MAC.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md) · [Contributing](CONTRIBUTING.md)

</details>

## Credits and licenses

**Strata-Server builds on [Strata by Niko1221 and contributors](https://github.com/Niko1221/Strata).** The original visual style, app and Terminal are retained. This fork's native work and local measurements are maintained by [XopMC](https://github.com/XopMC); checkpoints are at [Hugging Face](https://huggingface.co/XopMC513).

[MIT license](LICENSE) · [Original credits](docs/HOW_IT_WORKS.md#credits) · [Swift third-party notices](owned/swift/THIRD_PARTY_NOTICES.md) · [DeepSeek third-party notices](owned/deepseek/THIRD_PARTY_NOTICES.md)
