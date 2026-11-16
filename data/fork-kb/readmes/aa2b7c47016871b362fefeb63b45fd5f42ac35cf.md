<p align="center">
  <b>English (README.md)</b> · <a href="README.zh-CN.md"><b>简体中文 (README.zh-CN.md)</b></a>
</p>

> [!IMPORTANT]
> **Hardware-Targeted Optimization Fork (`AMD MI50 16GB` + `Xeon E5-2686 v4`)**
>
> This repository is a **hardware-targeted deep optimization fork** of **[Niko1221/Strata](https://github.com/Niko1221/Strata)** demonstrating how an **≈1,500 CNY (≈\$207 USD) secondhand workstation** (`Intel Xeon E5-2686 v4` + `64 GB DDR3-1866 ECC` + dual-fan cooled `AMD Instinct MI50 16GB HBM2`) can smoothly serve the **125B sparse MoE model `Qwen3.8-Flash-Next` (`IQ2_XS`)** all the way up to **256K (`262,144`) multi-turn Agent context**.
>
> For official desktop installers and general multi-vendor GPU support, please visit the upstream repository: **[github.com/Niko1221/Strata](https://github.com/Niko1221/Strata)**.

---

## Platform Overview & End-to-End Benchmarks (≈1,500 CNY / ≈\$207 Workstation)

### 1. Hardware & System Setup

| Component | Configuration |
| :--- | :--- |
| **Total Cost** | **≈1,500 CNY (≈\$207 USD)** (secondhand X99 server platform + air-cooled MI50 build) |
| **CPU** | **Intel Xeon E5-2686 v4** (14nm Broadwell-EP, **18 Cores / 36 Threads**, `3.00 GHz` all-core `AVX2` + `FMA3`; Uncore ring bus locked at **`2.70 GHz`**; `45 MiB` 20-way L3 cache partitioned via Linux `resctrl`: **`40.5 MiB` (18 ways)** dedicated to CPU compute threads, **`4.5 MiB` (2 ways)** isolating background PCIe transfer threads) |
| **System RAM** | **64 GB (`4 × 16 GB`) Quad-Channel DDR3-1866 ECC REG** (`59.7 GB/s` peak; 2 MiB Transparent Huge Pages `THP [always]` zero-copy pinning **`33.02 GiB`** of sparse expert weights and **`2.39 GiB`** of 256K streaming KV cache) |
| **GPU** | **AMD Instinct MI50 16GB HBM2** (Radeon Pro VII firmware, `Vega 20` / `gfx906`, 7nm, **60 CUs / 3,840 SPs**, **no Tensor / Matrix Cores**, native `INT8` dot-product, `26.8 TFLOPS` FP16, 1-cycle cross-lane DPP; running at **`1700 MHz` core / `1100 MHz` HBM2 (`1,126 GB/s`)** via a Linux runtime PowerPlay table without physical VBIOS flashing; **0 ECC errors**, **`15.96 GiB`** usable VRAM) |
| **Cooling** | Dual-fan shroud mod + automatic fan daemon: holds **`32–53 °C` edge / `42–60 °C` HBM2 / `47–87 °C` junction** under sustained full-load prefill with zero thermal throttling |
| **PCIe Link** | **PCIe 3.0 x16** (`iommu=pt`, motherboard `MaxReadReq` unlocked from `256B` to **`4096 Bytes`**, ASPM disabled → **`13.3 GB/s` sustained host-to-device DMA**) |
| **NVMe Storage** | PCIe NVMe SSD streaming the `26.8 GiB` per-layer embedding shard via `O_DIRECT` at **`38,069 rows/s`** |
| **Software Stack** | Ubuntu 24.04 LTS (Linux 6.8), native **`rocm-gfx906:7.14`** (`rocBLAS 5.5.0`, `-O3 -march=broadwell`) |
| **Model** | **ISTA-DASLab `Qwen3.8-Flash-Next` (`GSQ-RCO-IQ2_XS`)** (≈125B total sparse MoE, 512 experts activating top-10, 36 linear-recurrence GDN + 12 sparse block-attention QSA layers, MTP speculative drafter, `262,144` native context) |

---

### 2. Measured Performance

#### 2A. Upstream Strata vs. Optimized Fork (End-to-End Comparison)

Measured on the same workstation across identical prompts and output lengths:

| Workload (`Qwen3.8-Flash-Next` `IQ2_XS`) | Upstream Strata | Optimized Fork | End-to-End Gain | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Everyday Chat & Coding (Median / Mean)** | `36.6 tok/s` (`39.8` mean) | **`55.6 tok/s`** (`60.0` mean) | **`+51.9%`** | Reaches `56.9 tok/s` median even with 256K max context enabled |
| **Code Generation Peak (Default Speculation)** | `48.8 tok/s` | **`71.6 tok/s`** (`76–78` warm) | **`+46.7%`** | High draft acceptance + high-priority GPU drafting stream |
| **Code Generation Peak (Multi-Branch Tree Speculation)** | `48.8 tok/s` | **`75.4` – `83.8 tok/s`** | **`+54.5%` – `+71.7%`** | Verifies two candidate branches in parallel (`--spec-tree 3 --suffix-draft 2`) |
| **100% Lossless Full-Expert Chat Decode** | `35.8 tok/s` | **`55.2 tok/s`** (`62.2` peak) | **`+54.2%`** | Low-weight expert skipping disabled; **100% bit-identical** greedy output |
| **100% Lossless Full-Expert 4K Decode** | `31.5 tok/s` | **`47.8` / `48.6 tok/s`** | **`+51.7%`** | Pure kernel & CPU–GPU scheduling speedup without approximations |
| **4K Context Prefill (Cold / Warm Cache)** | `308.3 tok/s` | **`550.7` / `614.0 tok/s`** | **`+78.6%` / `+99.2%`** | Nearly **2x faster** prompt processing once hot experts warm up |
| **4K Context Long-Form Decode** | `35.5 tok/s` | **`54.2 tok/s`** | **`+52.7%`** | Hardware L3 cache isolation cuts latency jitter by 16x |
| **32K Long-Context Prefill (`27.8K` tokens)** | `345.7 tok/s` (`80.4 s`) | **`685.7 tok/s` (`40.6 s`)** | **`+98.4%`** | Cuts wait time in half (saves `~40 s`); 100% passcode recall |
| **32K Long-Context Decode (`27.8K` tokens)** | `~38.6 tok/s` | **`55.0 tok/s`** | **`+42.5%`** | 256 tokens of original technical prose after retrieving archive facts |
| **64K Long-Context Prefill (`256K` Max Context)** | `~331.7 tok/s` (`~194 s`) | **`655.7 tok/s` (`98.3 s`)** | **`+97.7%`** | Saves `~96 s` per prompt; 100% passcode recall |
| **64K Long-Context Decode (`256K` Max Context)** | `~33.3 tok/s` | **`54.6 tok/s`** | **`+64.0%`** | Upstream drops to `5,539` VRAM experts at 256K; ours keeps **`7,499`**, zero drop vs. 32K |
| **Full-Vocab Output Divergence (FP64 KL)** | `0.00935` | **`0.00609` (`5/5` Top-1)** | **`34.9%` lower error** | Closer to full-precision reference than upstream defaults |

#### 2B. 64K → 256K Ultra-Long Context & Multi-Turn Agent Stress Test (Single 16GB MI50)

Tested on a single 16GB MI50 with full `262,144` max context enabled across non-repetitive multi-module C++20 codebases (`—` indicates not run on upstream):

| Workload & Active Context Depth | Upstream Strata | Resident VRAM Experts | Prefill / Incremental TTFT | Decode Speed | Accuracy & Agent Task Verification |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **64K Document Analysis (`64,452` tok)** | `~331.7` / `~33.3 tok/s` | **`7,499` (`10.04 GiB`)** | **`655.7 tok/s`** (`98.3 s` cold) | **`54.6 tok/s`** | 100% passcode recall; `92.4%` of sparse attention reads hit VRAM |
| **128K Full Codebase Cold Load (`127,949` tok)** | `—` | **`7,499` (`10.04 GiB`)** | **`646.4 tok/s`** (`198.0 s` cold) | **`56.2 tok/s`** | Recalls both audit keys at `19K` and `108K` verbatim |
| **256K Extreme Cold Load Turn 1 (`251,964` tok)** | `—` | **`7,499` (`10.04 GiB`)** | **`614.5 tok/s`** (`410.0 s` cold) | **`51.8 tok/s`** | Recalls all 3 keys across `12K / 125K / 239K` verbatim; `90.5%` attention VRAM hit |
| **256K Multi-Turn Follow-Up Turn 2 (`255,037` tok)** | `—` | **`7,499` (`10.04 GiB`)** | **TTFT `6.80 s`** (`2,618` new tok, `384.8 tok/s`) | **`55.3 tok/s`** | Reuses `252K` session prefix cache and writes a lock-free C++20 patch; `93.6%` attention VRAM hit |
| **256K Reasoning + Tool Call Turn 3 (`255,653` tok)** | `—` | **`7,499` (`10.04 GiB`)** | **`614.3 tok/s`** (`415.8 s` cold) | **`55.9 tok/s`** | Extracts keys `243K` tokens apart into a valid JSON tool call; `94.9%` attention VRAM hit |

---

### 3. Optimization Highlights

1. **System Interconnect, CPU Cache Isolation & Binary-Patched GPU PowerPlay Table (Zero Hardware Flashing)**
   - Unlocked motherboard PCIe `4096B` read requests (`13.3 GB/s` sustained DMA) and 2 MiB huge-page pinning for the `33 GiB` RAM expert pool; locked the Xeon Uncore ring bus to `2.70 GHz` and used Intel `resctrl` hardware cache partitioning so background PCIe transfers cannot evict L3 cache lines used by the 18 CPU cores (**16x lower decode latency jitter**); dumped the GPU's binary PowerPlay table from Linux `sysfs`, reverse-engineered the hidden HBM2 clock clamps (`PowerSavingClockMax` / `DcModeMaxFreq`), and hot-injected a patched binary table unlocking **`1700 MHz` core and `1100 MHz` HBM2 (`1,126 GB/s`)** without touching physical VBIOS flash.
2. **GPU Kernel Binary Disassembly (552 → 0 Register Spills) & Custom Kernels for Matrix-Core-Less Hardware**
   - Because Vega 20 has no Tensor Cores, we went straight into the compiled `.hip_fatbin` GPU ELF binaries (`clang-offload-bundler` + `llvm-readobj` / `llvm-objdump`) and caught a hidden ROCm compiler trap: HIP translates CUDA's `__launch_bounds__` occupancy hint per-SIMD instead of per-CU, silently spilling **552 registers to slow VRAM** in core quantized kernels. After eliminating every spill across the entire engine (**`552 → 0` VGPR spills**), we rewrote Wave64 GPU kernels for fused MoE experts, bank-conflict-free shared-memory bitplane transposes, expert routing (**`3.05x` faster**), and 248K-vocabulary token selection (**`7.84x` faster**), plus register-blocked `AVX2` kernels speeding up 17 Xeon CPU cores by **`1.73x`**.
3. **Hand-Written GCN Inline Assembly (Direct HBM2-to-Shared-Memory DMA & 1-Cycle Cross-Lane Ops)**
   - Seeing in disassembled machine code that the compiler still staged activations through registers, we hand-wrote AMD GCN inline assembly (`buffer_load_dword ... lds` — **308 hardware async DMA instructions** verified directly in the final ELF binary) to stream input activations **straight from HBM2 into on-chip shared memory (LDS) without touching general-purpose registers**, overlapping memory transfers with weight compute. Across 7 core GPU modules, slow shared-memory shuffles were replaced with 1-cycle hardware cross-lane instructions (VALU DPP) while keeping greedy outputs **100% bit-identical**.
4. **CPU–GPU–PCIe Load Balancing & High-Priority Speculative Stream**
   - Tuned heterogeneous dispatch so 17 CPU cores compute 82% of non-VRAM experts while 18% stream over PCIe DMA, paired with background hot-expert promotion (`93%–98%` active VRAM hit rate). Isolating the MTP speculative drafter onto a high-priority GPU hardware stream prevented queuing behind background weight transfers, cutting draft latency by **40%**.
5. **112-Shape `rocBLAS` Tuning Table & Custom `<32 KB` FlashAttention (Nearly 2x Faster Prefill)**
   - Benchmarked and shipped optimal Vega 20 `rocBLAS` FP16 kernels across all 112 model matrix shapes, and wrote a custom `INT8` dot-product online FlashAttention kernel fitting strictly under `32 KB` shared memory (doubling GPU occupancy per compute unit) — sustaining **`614–685 tok/s`** prompt prefill across 32K–256K context depths.
6. **256K Pinned-RAM Streaming KV Cache (Near-Zero Decode Drop from 32K to 256K)**
   - Fixed an excess `263 MiB` VRAM allocation in the speculative draft ring and placed cold 256K KV blocks in `2.39 GiB` of pinned system RAM while keeping the hottest `16K` window per layer in HBM2 (even at `255K` depth, **`90.5%–94.9%`** of sparse attention reads hit VRAM directly). With `max_context=262144` enabled, resident VRAM experts jump from `5,489` back to **`7,499` (`10.04 GiB`)**, keeping decode speed at **`51.8–56.2 tok/s`** all the way from 32K to 256K.
7. **Multi-Branch Tree Speculative Decoding (Up to `83.8 tok/s` on Code)**
   - Added optional causal-tree speculative verification (`--spec-tree 1/2/3`): when draft confidence splits, the engine verifies two candidate token branches in a single forward pass across both recurrent and attention layers, reaching **`75.4–83.8 tok/s`** on code generation and refactoring.

---

### 4. Building & Running

```bash
# 1. Build optimized engine for gfx906 (MI50) + Broadwell (Xeon v4)
cmake -B build-opt -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_HIP=ON \
  -DSTRATA_HIP_GFX906=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DCMAKE_CXX_FLAGS="-O3 -march=broadwell -mtune=broadwell"
cmake --build build-opt -j"$(nproc)"

# 2. Start 256K Server (for 32K daily mode, use --max-context 32768 --kv int8 --kv-grow --vram-reserve-mib 704;
#    for code-heavy workloads, optionally append --spec-tree 3 --suffix-draft 2)
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export STRATA_Q2_BITPLANE=1 STRATA_CPU_IQ_ALL_EXACT1=1
export STRATA_ADAPT_OVERLAP=1 STRATA_ADAPT_JOB_CPU=35 STRATA_SPEC_COST=1
export STRATA_HIP_PROMPT_F16=1 STRATA_HC_UPMIX=1 STRATA_GFX906_PA=1
export STRATA_MMQ_GROUP_SORT=1 STRATA_MTP_HIP_PRIO=1
export STRATA_ROCBLAS_TUNING="$PWD/tools/hip/gfx906-rocblas-5.5.0.cd957402309.txt"

python3 serve/server.py --engine strata --host 127.0.0.1 --port 8090 -- \
  ./build-opt/strata --serve \
  --pack /path/to/packs/iq2_xs \
  --native /path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
  --ple-gguf /path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf \
  --expert-profile data/expert-profile.bin --expert-cache auto --prefill auto \
  --spec 4 --spec-min-p 0.65 --mtp /path/to/mtp/rt \
  --max-context 262144 --kv k8v4 --kv-resident 16384 --pcie-frac 0.18 \
  --pool-workers 17 --host-core last --aux-cpus auto --vram-reserve-mib 576 \
  --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
```

<hr>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h3 align="center">Run powerful AI on your own computer.</h3>
<p align="center">Chat, code and work with images, locally.</p>

<p align="center">
  <a href="#models">Models</a> · <a href="#install"><b>Install</b></a> ·
  <a href="#hardware">Hardware</a> · <a href="#tools">Tools</a> ·
  <a href="#help">Help</a> · <a href="#support-strata">Support</a>
</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?label=download&color=0ab887" alt="Download Strata"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT license"></a>
  <a href="https://github.com/sponsors/Niko1221"><img src="https://img.shields.io/badge/Sponsor-EA4AAA?logo=githubsponsors&logoColor=fff" alt="Sponsor Strata on GitHub Sponsors"></a>
  <a href="https://buymeacoffee.com/strataengine"><img src="https://img.shields.io/badge/Buy_me_a_coffee-FFDD00?logo=buymeacoffee&logoColor=000" alt="Support Strata on Buy Me a Coffee"></a>
</p>

<p align="center"><sub><b>English</b> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.pl.md">Polski</a></sub></p>

**Free, open source, and yours to run.** Use Strata on your desktop or with your favorite coding tools. No account needed.

## Models

We recommend **ISTA-DASLab first**, then **Unsloth** and other providers, where a compatible model fits.

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.8-Flash-Next</b></summary>

| Provider | Choices |
| --- | --- |
| **ISTA-DASLab · recommended** | Q2\_0, IQ2\_XS, IQ3\_XXS, IQ3\_S. [Coder: IQ1\_M](docs/MODELS.md#coder) for coding. |
| **Unsloth** | UD-IQ4\_XS, UD-Q4\_K\_XL, UD-Q5\_K\_XL, UD-Q6\_K\_XL, Q8\_0 |
| **Gyro · agentionai** | [Gyro-S, Gyro-M](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF). Lower memory use but slower. |
| **Other options** | [Swift 1.5](docs/MODELS.md#swift-15), [NVFP4](docs/NVFP4.md), [OrcaRouter](docs/ORCA.md) |

[Compare sizes, memory needs and compatibility](docs/MODELS.md).

</details>

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.6-35B-A3B</b></summary>

**Unsloth:** UD-Q4_K_XL, UD-Q6_K_XL, Q8_0. These are the currently documented packs for this model.

[Model guide and compatibility](docs/QWEN36.md).

</details>

<details>
<summary><img src="docs/media/models/deepseek.svg" width="22" alt="" align="middle"> &nbsp;<b>DeepSeekV4.1-Flash-1017</b></summary>

In the **desktop app** and the **Terminal**. What does not fit in memory is read from your SSD, so it needs a lot of free
disk space.

| Set (ISTA-DAS) | Download | Free disk needed |
| --- | ---: | ---: |
| 1.75 bpw | 239 GB | about 245 GB |
| 2.00 bpw | 257 GB | about 265 GB |
| 2.25 bpw | 274 GB | about 280 GB |
| ISTA-DAS Coder (half of the experts removed) | 191 GB | about 197 GB |

Runs on NVIDIA RTX 30, 40 and 50 series (Windows and Linux), AMD Radeon AI PRO R9700 (Linux), AMD Ryzen AI Max — Strix Halo
(Windows), and Apple Silicon Macs with 64 GB or more ([Mac](docs/DEEPSEEK_MAC.md)). Setup downloads the files and the DeepSeek engine, then starts the model.

```sh
strata launch --family deepseek41 --model ist-das-2.00bpw --plan
```

Measured with the 2.00 bpw set on Strata 0.2.5 (Strix Halo: 0.2.7). Output is the speed of the answer, prompt is how fast Strata reads your input.

| Hardware | Output | Prompt |
| --- | ---: | ---: |
| RTX 5070 12 GB, 64 GB RAM | 4 to 6 tokens/s | about 110 tokens/s (4K) |
| RTX 3060 12 GB, 46 GB RAM | 9 to 11 tokens/s | about 155 tokens/s (4K) |
| 1x Radeon AI PRO R9700, 251 GB RAM | 21 to 25 tokens/s | 320 to 360 tokens/s (8K) |
| 4x Radeon AI PRO R9700 | 74 to 77 tokens/s | 400 to 450 tokens/s (8K) |
| Ryzen AI Max+ 395 (Strix Halo), 128 GB unified | 19 to 21 tokens/s (code), 11 (chat) | 87 tokens/s (4K), 125 tokens/s (8K) |

[Memory, disk and details](docs/DEEPSEEK41.md) · [Release notes](https://github.com/Niko1221/Strata/releases)

</details>

<a id="the-other-sizes"></a>

Not sure what fits? [Choose a model for your computer](docs/MODELS.md#pick-by-memory).

## Install

<a id="option-a-the-strata-app"></a>

### Desktop app

**[Download Strata for Windows, Linux or Mac →](https://github.com/Niko1221/Strata/releases/latest)**

Install the app, choose a model, and start chatting. There is one installer for each system. On first start the app
downloads the engine for your graphics card. [Help with the installer](docs/INSTALL.md#the-app).

<sub>Files: `Strata-0.2.13-windows-x64-setup.exe` · `Strata-0.2.13-linux-x86_64.AppImage` · `Strata-0.2.13-macos-arm64.dmg`</sub>

<p align="center">
  <img src="docs/media/app-chat.png" width="900" alt="Strata desktop app with a chat and live code preview side by side">
  <br><sub>Chat, files and a live preview in one window.</sub>
</p>

<a id="option-b-the-terminal-one-line"></a>

### Terminal app

Prefer the terminal? Install Strata, then run **`strata launch`**.

<details open>
<summary><b>Install commands</b></summary>

**Linux / Mac**

```sh
curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

**Windows PowerShell**

```powershell
irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

On Windows, open a new terminal afterwards. Then:

```sh
strata launch
```

Follow the prompts to download and start a model. [Terminal guide](docs/TERMINAL_LAUNCH.md).

</details>

<p align="center">
  <img src="docs/media/terminal-monitor.png" width="900" alt="Strata terminal app showing model activity, speed, memory and recent requests">
  <br><sub>Watch your model and hardware while it runs.</sub>
</p>

<a id="build-from-source"></a>

[Build from source](docs/SOURCE_INSTALL.md) · [Docker](docs/DOCKER.md) · [Full install guide](docs/INSTALL.md)

<a id="what-you-need"></a>

## Hardware

| Supported hardware | Guide |
| --- | --- |
| **NVIDIA GeForce RTX 20 / 30 / 40 / 50 series** | [NVIDIA setup](docs/INSTALL.md#what-you-need) |
| **AMD Radeon RX 7700 / 7800 / 7900, RX 9060 / 9070, AI PRO R9700** | [AMD setup](docs/AMD_HIP.md) |
| **AMD Radeon RX 6800 / 6900 · experimental** | [RX 6000 setup](docs/AMD_HIP.md#rdna2-gfx1030) |
| **Intel Arc · Linux** | [Intel setup](docs/INTEL_ARC.md) |
| **Apple Silicon · M1 or newer** | [Mac setup](docs/MAC.md) |
| **AMD Ryzen AI Max · Strix Halo** | [Strix Halo setup](docs/STRIX_HALO.md) |
| **NVIDIA DGX Spark · Linux** | [DGX Spark setup](docs/DGX_SPARK_LINUX.md) |

<a id="using-it"></a>

## Tools

**Claude Code · Codex CLI · OpenCode · oh-my-pi · pi.dev · Hermes Agent · OpenClaw**, and more through the OpenAI- and Anthropic-compatible APIs.

Run **`strata connect`** to see setup instructions for your tool. [Connection guide](docs/TERMINAL_LAUNCH.md#connect-your-apps-strata-connect).

<a id="guides"></a>
<a id="how-fast-is-it"></a>
<a id="how-does-it-work"></a>
<a id="something-went-wrong"></a>

## Help

[All guides](docs) · [Troubleshooting](docs/TROUBLESHOOTING.md) · [Report a bug](https://github.com/Niko1221/Strata/issues/new/choose) ·
[Benchmarks](docs/COMMUNITY_BENCHMARKS.md) · [How it works](docs/HOW_IT_WORKS.md) · [Contribute](CONTRIBUTING.md)

## Support Strata

**Help keep local AI open to everyone.** A GitHub sponsorship or a coffee helps fund development and testing on more hardware.

<a href="https://github.com/sponsors/Niko1221"><img src="https://img.shields.io/badge/Sponsor_on_GitHub-EA4AAA?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="GitHub Sponsors: sponsor Strata" height="48"></a>
<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee: support Strata" height="48"></a>

### Loyal supporters

Thank you to everyone who supports Strata.

<!-- supporters:start (one line per public supporter, newest last; copy a line and change the GitHub name in all four places) -->
<a href="https://github.com/ryan4yin"><img src="https://github.com/ryan4yin.png?size=96" width="48" height="48" alt="ryan4yin" title="ryan4yin"></a>
<a href="https://github.com/themactep"><img src="https://github.com/themactep.png?size=96" width="48" height="48" alt="themactep" title="themactep"></a>
<a href="https://github.com/macintog"><img src="https://github.com/macintog.png?size=96" width="48" height="48" alt="macintog" title="macintog"></a>
<a href="https://github.com/bisonbet"><img src="https://github.com/bisonbet.png?size=96" width="48" height="48" alt="bisonbet" title="bisonbet"></a>
<a href="https://github.com/vizv"><img src="https://github.com/vizv.png?size=96" width="48" height="48" alt="vizv" title="vizv"></a>
<a href="https://github.com/kod88vn"><img src="https://github.com/kod88vn.png?size=96" width="48" height="48" alt="kod88vn" title="kod88vn"></a>
<a href="https://github.com/seoulworker"><img src="https://github.com/seoulworker.png?size=96" width="48" height="48" alt="seoulworker" title="seoulworker"></a>
<a href="https://github.com/jackyini"><img src="https://github.com/jackyini.png?size=96" width="48" height="48" alt="jackyini" title="jackyini"></a>
<a href="https://github.com/router0mail"><img src="https://github.com/router0mail.png?size=96" width="48" height="48" alt="router0mail" title="router0mail"></a>
<!-- supporters:end -->

[MIT license](LICENSE) · [Credits and model licenses](docs/HOW_IT_WORKS.md#credits) · [Privacy](docs/SAFETY.md)

<a id="option-c-let-your-ai-set-it-up"></a>

<sub>For AI tools and automated readers: <a href="AGENTS.md">technical reference</a> · <a href="docs/AI_SETUP.md">setup instructions</a>.</sub>

**Sponsorships, hardware partnerships or collaborations:** [niko.veit@outlook.com](mailto:niko.veit@outlook.com).

