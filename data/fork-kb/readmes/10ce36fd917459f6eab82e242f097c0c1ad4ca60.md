# 复活你的 MI50，真的可以干活，Qwen3.8 Flash 走起

> 本仓库是 [Niko1221/Strata](https://github.com/Niko1221/Strata) 的 fork（基线 `main` @ `99f3dbd`，2026-10 上）。
> upstream 的 HIP 后端本是 wave32 的 RDNA3/RDNA4 卡专用（gfx1100/1101/1200/1201）；本 fork 把 Strata 完整移植到
> 老数据中心卡 **AMD Mi50 的 32 GB 版**（gfx906，CDNA1/Vega，HBM2——原版 Mi50 只有 16 GB，装不下这个模型的
> VRAM 专家缓存，32 GB 版是关键），在**单张 Mi50** 上实测跑通 **Qwen3.8-Flash-Next IQ3_XXS、131072 上下文
> （11 万 prompt 实测 decode 48 tok/s）**，31 GB 内存的低配主机也能常驻服务。
> （upstream 于 2026-10-03 起也合入了社区 gfx906 支持——`STRATA_HIP_GFX906` opt-in 兼容构建，
> 其社区基准是 2×Mi50 **16 GB**；对照见 §7 末。）这一节是 fork 的全部改动、依赖、编译、运行方式与实测性能，upstream 原文保留在后面。

## 1. 移植改动（相对 upstream 的全部差异）

地基结论：**gfx906 不存在 wave32 执行模式**。GCN5/CDNA1 的 SIMD 只有 64 宽（RDNA 起才有 32 宽），
各版本 LLVM 对 wave32 code object 要么崩溃、要么忽略、要么拒绝——`tools/hip/gfx906_wave32_probe.{hip,sh}`
就是这个穷举验证本身（探针输出：无现成 clang 能为 gfx906 产出 wave32 对象）。所以 fork 走 llama.cpp 式的
**虚拟 warp32**：按 wave64 编译 code object，kernel 内部用 32 车道逻辑 warp——所有 shuffle 显式指定
width=32（HIP `__shfl*_sync` 原生支持且组内隔离），`__ballot` 取 64 位结果中本组那一半。gfx906 +
`HIP_VISIBLE_DEVICES` 下实测：wave=64、ballot 低半 = `0xffffffff`、lane32 半字 = `0xffffff00`，语义坐实。

| 文件 | 改动 |
| --- | --- |
| `cmake/hip_backend.cmake` | 新增 `_strata_hip_fork` 架构档：gfx906 允许构建，警告注明"wave64 对象 + 32 车道虚拟 warp" |
| `include/strata/hip_compat/intrinsics.hpp` | `ballot_sync` 半字修正（按 `__AMDGCN_WAVEFRONT_SIZE__` 与线程在 wave 内的位置取高/低 32 位）；补 ROCm 6.3 缺失的 `__syncwarp`（→ `__syncthreads`，全部 kernel 站点审计为 block-convergent 才安全）与 `__match_any_sync`（width-32 shuffle 模拟，硬件 `v_permblane` 是 gfx10+）；`dp4a` 的 CDNA1 分支（`v_dot4_i32_i8`） |
| `include/strata/hip_compat/cublas_v2.h` | 定义 `HIPBLAS_V2`，让 `hipDataType` 直连 `hipblasGemmEx_v2`；ROCm 6.3 hipBLAS 没有 `SetWorkspace`，映射成 no-op（真正的 workspace 路由在 hipBLASLt） |
| `src/kernels/cuda/qsa_prompt_attn.cu` | 一处裸 `__shfl` 补 width=32（此前默认整 wave，会跨组拿错数据） |
| `src/core/device.cu` | `arch_problem` 接受 gfx906 的 `warpSize == 64` |
| `tests/hip/intrinsics.cpp` | 测试同时接受 wave32 与 wave64 目标卡 |
| `serve/telemetry.py` | AMD 卡从 PCI sysfs 读 `current/max_link_speed`、`current_link_width`，Monitor 可见链路 Gen/宽（amdgpu 没有 NVML 式吞吐计数器） |
| `serve/server.py` | `/metrics` 输出最近一次请求的专家层级计数（`engine.last`：generated/hits/file_mb…） |
| `serve/web/app.js` | Monitor：sparkline 不再把 null 画成 0（AMD 的 PCIe 假直线消失，缺口断线）；PCIe 卡在无吞吐计数器的后端改显示引擎自报的每请求层级流量 |
| `tools/hip/gfx906_wave32_probe.{hip,sh}` | 新增探针：穷举 wave32 编译标志写法并上卡验证 |
| `tools/mspget.sh` | 新增：多连接 Range 并行下载 + 断点续传（ModelScope 拉大权重用；与移植无关，随仓库带上） |

验证：全套 ctest 42 项 **41 绿**（hip_intrinsics / qsa / iq / kv / sampler / gemm / dequant parity）。
唯一红项 `ple_parity` 只是缺 28.8 GB PLE GGUF 测试夹具，不是移植缺陷。

## 2. 依赖

**编译（引擎）**

- **ROCm 6.3.3**。用其 llvm 的 clang 18 直接驱动 HIP 编译（不走 hipcc）。注意 **ROCm 7.x 的 libhsa 与 6.8 内核
  不兼容**（`hsa_init` 失败），运行时必须停在 6.x。
- **device bitcode**：ROCm 6.3 apt 包的 bitcode 不在 clang 默认搜索路径。本机做法：`apt download` 对应 llvm 包 +
  `dpkg-deb -x` 解到 `~/c/devlib/opt/rocm-6.3.3/lib/llvm/lib/clang/18/lib/amdgcn/bitcode`（不侵入系统），
  用 `--rocm-device-lib-path` 指过去。
- CMake ≥ 3.28（实测 3.28.3）、系统 g++ 作 host 编译器。
- llama.cpp 走仓库自带 `third_party/llama.cpp`（`-DSTRATA_GGML_DIR=…`），避免 FetchContent 拉网络。

**运行（serve 服务端 + Monitor 页面）**：Python ≥ 3.11（实测 3.12.3），仅三个第三方包：
**jinja2**（模板）、**regex**（`tools/strata_tokenizer.py` 的分词器）、**psutil**（Monitor 的磁盘/RAM 读数）。
HTTP 层就是 stdlib `http.server`，没有 FastAPI。

## 3. 编译

```sh
cmake -S . -B build-gfx906 -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DCMAKE_HIP_COMPILER=/opt/rocm-6.3.3/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_FLAGS="--rocm-path=/opt/rocm-6.3.3 --rocm-device-lib-path=$HOME/c/devlib/opt/rocm-6.3.3/lib/llvm/lib/clang/18/lib/amdgcn/bitcode" \
  -DSTRATA_GGML_DIR=$HOME/strata/third_party/llama.cpp
cmake --build build-gfx906 -j"$(nproc)"
```

跑测试再带 `-DSTRATA_BUILD_TESTS=ON` 重新 configure，然后：

```sh
cd build-gfx906 && HIP_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/opt/rocm-6.3.3/lib ctest
```

**双卡主机注意**（本机是 RX580 + Mi50）：gfx803 在场时 gfx906 二进制 `hipSetDevice(Mi50)` 也报
`invalid device function`——**运行/测试一律 `HIP_VISIBLE_DEVICES=1`**（Mi50 在 device 1）；
缺 `LD_LIBRARY_PATH` 会得到 39/42 的假失败（libamdhip64.so.6 找不到）。

## 4. 运行

服务端（OpenAI 兼容 API + Monitor），配置样例 `strata-mi50.json`（在本仓库根目录，路径按你的机器改）：

```json
{
 "exe": "/home/fett/c/strata/build-gfx906/strata",
 "args": [
  "--pack", "/c/strata-data/packs/iq3_xxs",
  "--native", "/c/strata/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
  "--ple-gguf", "/c/strata/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf",
  "--expert-profile", "data/expert-profile.bin",
  "--expert-cache", "auto",
  "--resident-budget-gib", "12",
  "--prefill", "auto:32768",
  "--spec", "4", "--spec-min-p", "0.5",
  "--mtp", "/c/strata-data/mtp-rt",
  "--max-context", "131072", "--kv", "int8"
 ],
 "cwd": "/home/fett/c/strata",
 "tokenizer": "/c/strata-data/packs/iq3_xxs/tokenizer",
 "model_name": "qwen3.8-flash-next-iq3_xxs",
 "log": "strata-mi50.log",
 "lib_dirs": ["/opt/rocm-6.3.3/lib", "/usr/lib64"],
 "port": 8080, "host": "0.0.0.0", "gpu": 1, "backend": "hip", "hip_ordinal": 1
}
```

```sh
.venv/bin/python serve/server.py --engine strata --config strata-mi50.json --port 8080
```

- `"backend": "hip"` + `"hip_ordinal": 1`：server 自己以 `HIP_VISIBLE_DEVICES=1` 拉起引擎；
  `"gpu": 1` 让 Monitor 监视 Mi50 而不是 RX580（否则你会看到 8 GB 的卡、负载恒 0）。
- 不配 `api_key` 时 server 照样有 Host 校验：`host: "0.0.0.0"` 下局域网地址自动放行（如 `192.168.3.6:8080`），
  跨源浏览器页面仍被 Origin 检查拦住。
- 引擎 CLI 细节：`strata` 二进制就是 generate 程序，选项直接跟在 argv[1] 后，**没有 `generate` 子命令**
  （多写一个会报 `unknown argument: generate`）；serve 拉起时自动加 `--serve`。
- native pack 必须 `--spec`（≥2）与 `--prefill CHUNK` 同时给（如 `--prefill auto:32768`）。
- Qwen3.8 是思考模型：`max_tokens` 给小了会被思考吃光、正文为空——需要 `reasoning_budget_tokens` 或足够大的
  `max_tokens`。
- 用 Claude Code 挂这个大上下文服务时：把 `CLAUDE_CODE_AUTO_COMPACT_THRESHOLD` 设到 **0.5 以下**、多试几档，
  否则自动压缩可能触发过晚、长会话受影响。

## 5. 32 GB 内存怎么跑（实测机 31 GB）

默认模式把全部 experts（42.9 GB）装进匿名常驻 arena——31 GB 机上 `MAP_FAILED`，引擎报
"the arena could not be reserved" 后静默退出。**低内存模式 = 加 `--resident-budget-gib 12`**
（隐含 `--mmap-experts --resident-cpu-experts --resident-pin`）：

- experts 不进匿名 arena，改为从 `experts.bin` 文件层经 OS page cache 读；
- 按 profile 热度把 GPU cache 装不下的那部分里最热的 12.00 GiB（7,357 个 expert）常驻 RAM——
  内核页锁定被拒（`mlock failed (raise ulimit -l)`），实际是 pageable 常驻；想要真 pin 先提 `ulimit -l`；
- `--expert-cache auto` 吃 24.4 GiB VRAM、14,976 slots，启动就从 expert-profile 预填满
  （PROFILE 策略、不驱逐），路由漂移时与 RAM 层自适应互换；
- 整机 steady-state RAM ~17.7 GB / 33.5 GB，VRAM 用满 31.4 / 32.0 GiB（剩 592 MiB）。

引擎日志里的层级样貌：

```text
FileExpertSource: RAM budget 12.00 GiB: 7357 of the 15.60 GiB of experts the GPU cache does not hold, by profile rank
resident RAM mode: 12.00 GiB of experts in RAM (pageable), 14976 in the GPU cache; adaptive swaps exchange them (no file reads)
```

冷缓存注意：重启后第一个请求慢一个数量级（实测首个 54-token 请求读取只有 ~52 tok/s 量级，热态 300+），
属 expert/页缓存预热，属正常。

## 6. 更大内存怎么跑

去掉 `--resident-budget-gib` 就是 upstream 48–64 GB 主机的原生路径：全部 experts 进常驻 arena
（匿名内存，`--resident-pin` 尝试 huge pages / mlock），GPU cache 不变，steady state 零文件层读。
upstream 的基准机就是 47–64 GB RAM；48 GB 起跑 IQ3_XXS 无压力，64 GB 有余量（还想要 conversation cache
更大就再加）。

## 7. 当前环境实测性能

机器：Mi50 **32 GB 版**（gfx906，PCIe sysfs 报 Gen4 x16）· Ryzen 9 3950X 16 核 · 31 GB **DDR4-3200**（X570 主板）·
IQ3_XXS · 上下文 131,072 · kv int8 · draft `--spec 4` + MTP · prefill chunk 32,768 ·
expert 三层：VRAM 14,976 slots / RAM 12 GiB / 文件层。

| 上下文（prompt tokens） | 新增读取 | 输出 | 读取速度 | 生成速度 | draft 接受 | expert 命中 | 该请求文件层读入 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 54（重启后首请求，冷缓存） | 54 / 1.0 s | 16 | ~52 tok/s | 34.5 tok/s | 69% | 92.3% | 121 blob / 208 MB |
| 41,339（全新冷读） | 41,339 / 164 s | 2,649* | 252 tok/s | 29.6 tok/s | 69% | 97.3% | 10,403 blob / 17.3 GB |
| 43,872（复用 41,334） | 2,538 / 15.3 s | 6,868 | 165 tok/s | 36.2 tok/s | 59% | 98.8% | 12,768 blob / 20.8 GB |
| 50,274（复用 43,867） | 6,407 / 20.8 s | 5,428 | 307 tok/s | 38.7 tok/s | 61% | 98.9% | 9,550 blob / 15.3 GB |
| 55,487（复用 50,269） | 5,218 / 17.0 s | 9,839 | 307 tok/s | 42.2 tok/s | 71% | 98.5% | 21,612 blob / 35.1 GB |
| 65,030（复用 55,482，最长一次生成） | 9,548 / 32.2 s | 20,426 | 297 tok/s | 45.6 tok/s | 80% | 98.2% | 49,560 blob / 79.5 GB |
| 76,890（复用 65,025） | 11,865 / 38.6 s | 18,993 | 307 tok/s | 47.9 tok/s | 90% | 97.1% | 73,053 blob / 119.6 GB |
| 125,039（128K 证明跑，更早的服务实例） | 125,039 / 405 s | 32 | ~309 tok/s | 17.7 tok/s（与分块 prefill 重叠） | ~70% | 96–97% | — |
| 76,854（第二轮对话全新冷读） | 76,854 / 282 s | 6,243 | 272 tok/s | 38.3 tok/s | 86% | 96.9% | 60,690 blob / 43.0 GB |
| 110,475（复用 96,546，**接近跑满 128K**） | 13,929 / 42.2 s | 11,463 | 330 tok/s | **48.4 tok/s** | **92%** | 97.3% | 128,312 blob / 64.9 GB |

\* 41,339 行是客户端在生成中途断开后的统计；65,030 行的 20,426 tokens 是模型自然结束（finish stop），
没碰到 66 K 的输出上限，请求全程 480 s。"文件层读入"是引擎发起的读入量——page cache 吸收了大部分，
块设备实测均值 ~7、峰值 ~27 MB/s。

**速度的真正瓶颈是专家能否快速进入显存，不是上下文长度**。只要被路由选中的 expert 已经在 VRAM 缓存
（14,976 slots）或 12 GiB RAM 层里等着（实测命中 97–99%），单张 Mi50 上 11 万提示 token 也能跑 **48.4 tok/s**
——110,475 那行 draft 接受 92%、96,546 个 token 走 conversation cache 复用，只新增读取 13,929。
反过来，当 expert 必须从文件层现读再进显存时速度立刻掉下来：125,039 行在分块 prefill 期间只有 17.7 tok/s，
因为 prompt 流水线临时借走 8,285 个 cache slots（引擎日志 "the prompt path borrows 8285 cache slots"），
decode 的 expert 供应被挤占，17.7 tok/s 的 decode 只能与后续 prefill 重叠进行。

同一对话里还会**越跑越快**：expert 缓存随路由热集自适应收敛、MTP draft 对正在生成的文风命中率上升——
两轮连续对话 11 次请求里 draft 接受率从 59% 一路爬到 **92%**，decode 均速从 29.6 tok/s 一路爬到 **48.4 tok/s**，
上下文最深的 110,475 请求反而是全程最快的一次；conversation cache 又让每轮新增读取越来越少。
同一个 Mi50，聊得越深跑得越快。

**瞬时峰值**：表中 decode 列是每个请求的全程均速，Monitor 的实时曲线比它跳得高——速度历史的采样峰值
≥65 tok/s，draft 连击时见过 **75 tok/s** 的瞬间 burst：`--spec 4` + MTP 一次前推可吐出多个 token，
连续全收时瞬时速度翻倍。能跑多快由 draft 接受率决定（上下文越深接受率越高，65,030 行 80%、76,890 行 90%、110,475 行 92%），而不是
Mi50 的原始算力——稳态均速受限于接受率与 expert 换页，算力上还有余量。

**单卡对照：相对 upstream 社区基准**

本 fork 的成绩全部来自**单张 Mi50 32 GB**（host6，插一张 PCIe 卡）。upstream 2026-10-03 合入的社区基准
`bench/results/2026-10-03-community-2x-mi50`（JeanP00l）为 **2× Mi50 16 GB**（layer split）、Coder IQ1_M、
HIP 7.14 社区容器：4K prompt 50.1 / 32K 47.8 / 128K prompt 45.7 tok/s，host→device 读取 13.8 GB/s。
这台 32 GB 单卡的同规模成绩：

| | upstream 社区基准 | 本 fork（单卡 32 GB） |
| --- | --- | --- |
| 卡 | 2× Mi50 16 GB | 1× Mi50 32 GB |
| 模型 | Coder IQ1_M | Qwen3.8-Flash-Next IQ3_XXS（71 GB pack） |
| 上下文 128K | 45.7 tok/s（4K 时 50.1） | 43.6–48.4 tok/s @ 83–110K prompt，瞬时峰值 75 tok/s |
| 专家权重驻留 | 双卡分摊 | 24.6 GiB 全驻单卡显存 + 12 GiB RAM 层 |

（模型与量化不同，速度数字不宜直接对比，列在一起只为说明规模。）结论：上游的 16G 双卡路线证明了 gfx906
跑得动 Strata；本 fork 的证明是**单张 32 GB 就够**——IQ3_XXS + 128K 上下文 + 接近满上下文仍 48 tok/s，
而且跑在 ROCm 6.3.3 + 6.8 内核、31 GB RAM 的低配主机上，不需要社区容器。

**温度 / 功率 / 磁盘 / 占用**（同一台机器，60 个采样窗口，按 idle 与负载分段）：

| 状态 | GPU 占用 | 温度 | 功率（上限 178 W） | 磁盘读 | CPU | RAM 已用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 空闲 | 0% | 54–80 °C | 21–24 W | 0–27 MB/s | 1–35% | 17.7 GB |
| 负载（生成中） | 16–100%（均 97） | 77–83 °C（均 ~80） | 74–198 W（均 ~168） | 0.8–52 MB/s（均 ~12） | 38–49%（均 ~43） | 17.7 GB |

温度随负载深度爬升：短负载稳在 78–79 °C，480 秒的满上下文长生成把它推到 81–83 °C 并稳住；功率贴着
178 W 上限跑（瞬时峰值 198 W 是 boost 脉冲）。长生成对文件层的压力也最大（见 65,030 行的 79.5 GB
读入——绝大部分被 page cache 吸收，块设备实测均值 ~12、峰值 ~52 MB/s，NTFS 盘随机读余量充足）。
Mi50 的 32 GB HBM 全程占用 ~31.4/32.0 GiB。Monitor 页（`http://<host>:8080/`）对每路都有实时
sparkline 曲线；PCIe 卡因为 amdgpu 没有吞吐计数器，显示链路规格 + 引擎自报的每请求层级流量，而不是假曲线。

---

<h1 align="center">Strata</h1>

**English** · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

<p align="center"><b>Run a 125-billion-parameter AI model on your own gaming PC</b><br>
NVIDIA or AMD graphics card (12 GB or more) · Windows or Linux · free and open source</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="A voxel pagoda garden that Strata's model wrote, running in the browser"></a><br>
<sub>A voxel pagoda garden, 1 shot prompt running on an RTX 5070 with Strata (IQ3_S, 128K context) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">full video (49 s)</a></sub></p>

Strata runs **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** on a normal PC. This is a
large, smart AI model that usually needs a server. It chats, writes code, reads pictures and works with your apps
and coding agents. Nothing leaves your PC.

## How fast is it?

We measured it on two ordinary gaming PCs. A token is about ¾ of a word.

- **Writes answers:** how fast the reply appears in a short chat. 60 tokens per second is faster than you can read.
- **Reads your prompt:** how fast it takes in what you send (here a 32K-token document, code or chat history).

<table>
<tr><th>NVIDIA: RTX 5070 (12 GB), Ryzen 5 7600, 64 GB RAM</th><th>AMD: RX 9070 XT (16 GB), Ryzen 9 3900X, 47 GB RAM</th></tr>
<tr><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **Q2_0** | 94 tokens/s | 2,650 tokens/s |
| **IQ2_XS** | 79 tokens/s | 2,090 tokens/s |
| **IQ3_XXS** | 62 tokens/s | 1,750 tokens/s |
| **IQ3_S** | 53 tokens/s | 1,620 tokens/s |
| **Coder** | 55 tokens/s | 2,180 tokens/s |

</td><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **Q2_0** | 60 tokens/s | 1,160 tokens/s |
| **IQ2_XS** | 52 tokens/s | 1,110 tokens/s |
| **Coder** | 44 tokens/s | 1,420 tokens/s |

</td></tr>
</table>

NVIDIA: Q2_0 with engine 0.1.36, the other rows with 0.1.26 (4K answers, 32K prompts). The full tables are in
[DETAILS.md](docs/DETAILS.md#speed-measured). A card with more VRAM is faster: an RTX 3090 (24 GB) should write
about 100-140 tokens per second. Long chats and other cards: [speed of each model](docs/MODELS.md#how-fast-is-each-size),
[community results](docs/COMMUNITY_BENCHMARKS.md).

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a><br>
<sub>Strata is free. If it runs well on your PC, a coffee keeps the work on it going.</sub></p>

## What you need

| | |
| --- | --- |
| **Graphics card** | **NVIDIA** GeForce RTX 20, 30, 40 or 50 series, or **AMD** Radeon RX 7900 XT / XTX, RX 7800 XT / 7700 XT, RX 9060 XT, RX 9070 / 9070 XT, Radeon AI PRO R9700 or RX 6800 / 6900 series. It needs **12 GB of VRAM or more**. |
| **RAM** | 32 GB or more. Your RAM decides [which model](#which-model-should-i-pick) fits. 64 GB runs every size. |
| **Disk** | About 80 GB free. Use an SSD if you can: the first start is much faster. |
| **System** | Windows 10 / 11 or Linux, and a current graphics driver from NVIDIA or AMD. |

The installer sets up everything else. Two or three cards can share the model ([multi-GPU](docs/MULTI_GPU.md)).

Experimental, written and tested by community members on their own machines:

- **Older graphics cards** (Tesla P40 / V100, GTX 10, Radeon VII / MI50, RX 6700 XT, RX 5500 XT): [Older GPUs](docs/OLDER_GPUS.md).
- **Intel Arc**, built from source on Linux: [Intel Arc](docs/INTEL_ARC.md).
- **AMD Ryzen AI Max (Strix Halo)**, built from source on Linux: [Strix Halo](docs/STRIX_HALO.md).
- **Older processors without AVX2**: they work, but slowly. [Older CPUs](docs/INSTALL.md#older-cpus-experimental).

The full list: [docs/INSTALL.md](docs/INSTALL.md#what-you-need).

## Install

### Let your AI set it up

Do you use an AI coding assistant (Claude Code, Cursor, Codex, GitHub Copilot, ...)? Paste this into it:

```text
Set up Strata on this PC for me: https://github.com/Niko1221/Strata - follow docs/AI_SETUP.md in that repository.
```

It checks your graphics card, RAM and disk and picks the model that fits. Then it installs and starts it and tells
you how to connect your apps. AI tools can also install, start and stop Strata through its
[MCP server](docs/MCP_SERVER.md).

### Or do it yourself

[Download Strata](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip) and unzip it (or `git clone` it).
**Windows:** double-click **`START-HERE.bat`**. **Linux:** run **`./setup.sh`** in the Strata folder.

The steps are the same for NVIDIA and AMD. The installer finds your card and sets up the right engine for it. It
asks you a few questions:

- which model and which size,
- how much context (how much text the model keeps in mind),
- whether it should read pictures.

Press Enter each time for the recommended answer. Then it downloads the model (about 70 GB) and starts it. If the
download stops, run it again: it continues where it left off. Your browser opens the Strata app at
`http://127.0.0.1:8080`.

> **While the model starts, your PC can be slow or stop responding for 1-3 minutes** (longest the first time).
> Strata loads 35-55 GB into your RAM and locks part of it for the graphics card. This is normal. Wait, and don't
> close the window. The window shows what Strata is doing.

**Next time**, run `START-HERE.bat` (or `./setup.sh`) again. It starts right away and downloads nothing twice. Close
its window to stop the model. `UPDATE.bat` (`./update.sh`) updates Strata without starting it. Updating, Docker,
several cards, where the files go and every option: [docs/INSTALL.md](docs/INSTALL.md).

## Which model should I pick?

The installer recommends one for your RAM. The same model comes in several sizes, compressed more or less. Smaller
sizes are faster. Larger sizes are a bit smarter.

| Your RAM | Take | Why |
| --- | --- | --- |
| **32 GB** | **Coder** | it fits 32 GB, and it is made for code (with a 24 GB card, Q2_0 and IQ2_XS run too) |
| **48 GB** | **IQ2_XS** (or Q2_0, the fastest) | the larger sizes do not fit |
| **64 GB** | **IQ2_XS** (recommended), or IQ3_XXS / IQ3_S | every size fits; IQ3_S is the best and the slowest |
| **96 GB or more** | **IQ3_S**, or Unsloth's UD-IQ4_XS (~4-bit) | room for the largest sizes with everything else open |

- **[Coder](docs/MODELS.md#coder):** a coding version with half of the experts removed. It reaches 91% of the full
  model's SWE-bench Verified score (measured by its authors) and fits 32 GB of RAM. It is weaker outside code,
  including Chinese and other CJK text (#438). For those, take Q2_0, IQ2_XS or IQ3_S, which keep every expert.
- **[Swift 1.5](docs/MODELS.md#swift-15):** a fine-tune that thinks for a much shorter time before it answers. You
  get the answer sooner, at about the same quality.
- **[Unsloth UD-IQ4_XS](docs/MODELS.md#unsloth-ud-iq4_xs):** Unsloth's ~4-bit version, between IQ3_S and
  UD-Q4_K_XL in quality. A 94 GB download. With less than ~80 GB of RAM, Strata reads part of it from the SSD
  while it answers, so it is slower there (an NVMe SSD helps).
- **[Unsloth UD-Q4_K_XL](docs/MODELS.md#unsloth-ud-q4_k_xl-experimental)** (experimental): the closest to the full
  model. But Strata reads most of it from the SSD while it answers, so it writes only 7-8.5 tokens/s on a 64 GB PC.
- **[OrcaRouter's Uncensored IQ3_XXS](docs/MODELS.md#orcarouter-uncensored-iq3_xxs):** you set it up by hand. It is
  not in the installer's menu.

Sizes, downloads and what fits where: [docs/MODELS.md](docs/MODELS.md). To add another model later, run
`SETUP.bat` (Linux: `./setup.sh --setup`).

## Using it

<p align="center"><img src="docs/media/runpagoda.png" width="900" alt="The Strata app's Monitor tab next to a coding agent"><br>
<sub>The Strata app's <b>Monitor</b> (left) while a coding agent writes the pagoda garden from the video (right)</sub></p>

- **In the browser:** open `http://127.0.0.1:8080`. It has **Chat**, a live **Monitor** of the model and your
  GPU/CPU/RAM, and **About** with the settings and addresses.
- **Your apps and coding agents:** add an "OpenAI-compatible" provider with the base URL
  **`http://127.0.0.1:8080/v1`**. Any API key and any model name work.
  - Apps that use Anthropic's API: `http://127.0.0.1:8080/v1/messages` (Claude Code:
    `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`).
  - Codex CLI and other apps that use the OpenAI Responses API: `/v1/responses`
    ([setup](docs/DETAILS.md#the-responses-api-and-codex-cli)).
- **Thinking:** choose **off, low, medium or high** in the chat menu or in your app's "reasoning effort". Off is the
  fastest. High is best for hard questions.
- **Pictures:** say yes to "Images?" in setup. Then click **Picture** in the chat, or attach pictures in your app.
  AMD cards read pictures on Linux through the processor; on Windows they can't yet.
- **From your phone or another PC:** `START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>`. Always set a key.
- **One request at a time:** by default Strata answers one request, and the others wait. To answer several at once,
  set `"parallel": 2` ([BATCHING.md](docs/BATCHING.md)). On a 12 GB card this makes each answer slower.
- **Long prompts:** Strata reads the first message of a chat in full, about 1 minute per 30,000 tokens. Follow-up
  messages start in seconds.

More: [where your chats are stored](docs/INSTALL.md#where-things-are-stored), [the API](docs/DETAILS.md#using-it).

## Something went wrong?

- **My PC froze the first time Strata started.** This is normal while it loads the model. Wait, and don't close the
  window. Still frozen after 10 minutes? Restart the PC, close other programs and try again, or pick a smaller size.
- **It stopped while downloading or installing.** Run `START-HERE.bat` (or `./setup.sh`) again. It continues where
  it stopped.
- **It's very slow and the disk light keeps blinking, or it says "the engine stopped unexpectedly".** Your PC does
  not have enough free RAM. Close other programs (browsers use a lot), or pick a smaller size (Q2_0 or IQ2_XS).
- **It says port 8080 is already in use.** Strata is already running. Look for its window.

More problems and their fixes: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Still stuck? Open an
[issue](https://github.com/Niko1221/Strata/issues) and attach `strata-<model>.log` from the Strata folder. Found a
security problem? Report it privately: [SECURITY.md](SECURITY.md).

## How does it work?

Models like this one usually run on servers with hundreds of gigabytes of graphics memory. Your graphics card has
12-24 GB. Strata makes the model fit by **sharing the work across your whole PC**. Think of a kitchen: the things
you use all the time stay on the counter, and the rest waits in the pantry.

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="The model's 24,576 experts: the busiest on the graphics card, all of them in RAM, a lookup table on the SSD"></p>

- **The model is a team of 24,576 small specialists ("experts").** Each word needs only 10 of them.
- **Your graphics card** keeps the few thousand experts that are used most often. **Your RAM** holds all of them,
  and **your processor** works on the rest at the same time. **Your SSD** holds a big lookup table.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="A small helper guesses the next words; the big model checks them all at once and keeps the right ones"></p>

- **Guess, then check:** a small helper guesses the next few words. The big model checks them all at once. You get
  the same answer, 1.6-1.8x sooner.
- **Long texts are read in big pieces** (up to 8,192 tokens at a time), at over 1,000 tokens per second.

The longer explanation: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). Every part and its numbers:
[the details](docs/DETAILS.md#how-it-works) and the [paper](docs/paper/Strata-Paper.pdf).

## Credits and license

The model is [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team. It was
compressed by [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5)
and Unsloth. Strata uses parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). All credits:
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits). Strata is open source under the [MIT License](LICENSE). A few
parts and every model have their own licenses ([which ones](docs/HOW_IT_WORKS.md#license)).

## Support Strata

Strata is free and open source. If it is useful to you, you can support its development:

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a></p>
