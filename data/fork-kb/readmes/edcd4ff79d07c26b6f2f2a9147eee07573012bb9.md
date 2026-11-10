<h1 align="center">Strata</h1>

**[English](README.md)** · **简体中文** · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

<p align="center"><b>在你自己的游戏电脑上运行一个 1250 亿参数的 AI 模型</b><br>
NVIDIA 或 AMD 显卡（12 GB 或以上）· Windows 或 Linux · 免费且开源</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="Strata 模型写出的体素宝塔庭院，在浏览器中运行"></a><br>
<sub>一座体素宝塔庭院，1 shot 提示词，在 RTX 5070 上用 Strata 运行（IQ3_S，128K context）·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">完整视频（49 秒）</a></sub></p>

Strata 在普通电脑上运行 **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**。这是一个庞大的
聪明 AI 模型，通常需要一个服务器才行。它能聊天、写代码、读图片，并与你的应用和 coding agent 协作。
所有内容都不会离开你的电脑。

## 它有多快？

我们在两台普通的游戏电脑上对它做了实测。一个 token 大约相当于 3/4 个单词。

- **写回答（Writes answers）：** 在一段短聊天里，回复出现得有多快。每秒 60 个 token 就比你阅读的速度还快。
- **读你的提示（Reads your prompt）：** 它接收你发送内容（这里是一份 32K-token 的文档、代码或聊天记录）的速度有多快。

<table>
<tr><th>NVIDIA：RTX 5070（12 GB）、Ryzen 5 7600、64 GB RAM</th><th>AMD：RX 9070 XT（16 GB）、Ryzen 9 3900X、47 GB RAM</th></tr>
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

NVIDIA：Q2_0 使用引擎 0.1.36，其余各行使用 0.1.26（4K 回答，32K 提示）。完整表格见
[DETAILS.md](docs/DETAILS.md#speed-measured)。显存（VRAM）更多的显卡会更快：一张 RTX 3090（24 GB）
大约应能每秒写出 100-140 个 token。长对话与其他显卡：[每个模型的速度](docs/MODELS.md#how-fast-is-each-size)、
[社区实测结果](docs/COMMUNITY_BENCHMARKS.md)。

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a><br>
<sub>Strata 是免费的。如果它在你的电脑上跑得不错，一杯咖啡能让对它的开发继续下去。</sub></p>

## 你需要什么

| | |
| --- | --- |
| **显卡（Graphics card）** | **NVIDIA** GeForce RTX 20、30、40 或 50 系列，或 **AMD** Radeon RX 7900 XT / XTX、RX 7800 XT / 7700 XT、RX 9060 XT、RX 9070 / 9070 XT、Radeon AI PRO R9700 或 RX 6800 / 6900 系列。需要 **12 GB 或以上的 VRAM**。 |
| **内存（RAM）** | 32 GB 或以上。你的 RAM 决定了[哪个模型](#我该选哪个模型)装得下。64 GB 可以运行每一种尺寸。 |
| **磁盘（Disk）** | 约 80 GB 可用空间。尽可能用 SSD：首次启动会快得多。 |
| **系统（System）** | Windows 10 / 11 或 Linux，以及一个来自 NVIDIA 或 AMD 的较新显卡驱动。 |

安装程序会帮你配置其余一切。两到三张显卡可以共享这个模型（[multi-GPU](docs/MULTI_GPU.md)）。

实验性功能，由社区成员在自己的机器上编写并测试：

- **较旧的显卡**（Tesla P40 / V100、GTX 10、Radeon VII / MI50、RX 6700 XT、RX 5500 XT）：[Older GPUs](docs/OLDER_GPUS.md)。
- **Intel Arc**，在 Linux 上从源码构建：[Intel Arc](docs/INTEL_ARC.md)。
- **AMD Ryzen AI Max (Strix Halo)**，在 Linux 上从源码构建：[Strix Halo](docs/STRIX_HALO.md)。
- **不带 AVX2 的较旧处理器**：可以工作，但很慢。[Older CPUs](docs/INSTALL.md#older-cpus-experimental)。

完整列表：[docs/INSTALL.md](docs/INSTALL.md#what-you-need)。

## 安装

### 让你的 AI 来配置

你在用某个 AI 编程助手吗（Claude Code、Cursor、Codex、GitHub Copilot，……）？把下面这段粘贴给它：

```text
Set up Strata on this PC for me: https://github.com/Niko1221/Strata - follow docs/AI_SETUP.md in that repository.
```

它会检查你的显卡、RAM 和磁盘，并挑选装得下的那个模型。然后它会安装并启动，并告诉你如何连接你的应用。
AI 工具也可以通过它的 [MCP server](docs/MCP_SERVER.md) 来安装、启动和停止 Strata。

### 或者自己动手

[下载 Strata](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip) 并解压（或 `git clone` 它）。
**Windows：** 双击 **`START-HERE.bat`**。**Linux：** 在 Strata 文件夹里运行 **`./setup.sh`**。

NVIDIA 和 AMD 的步骤是一样的。安装程序会找到你的显卡并为它配置合适的引擎。它会问你几个问题：

- 选哪个模型、哪个尺寸，
- 多大的 context（模型在"脑中"保留多少文本），
- 是否要读取图片。

每次都按 Enter 采用推荐答案。然后它会下载模型（约 70 GB）并启动。如果下载中断了，再运行一次：
它会从中断处继续。你的浏览器会在 `http://127.0.0.1:8080` 打开 Strata 应用。

> **在模型启动期间，你的电脑可能会变慢或停止响应 1-3 分钟**（第一次最久）。
> Strata 会把 35-55 GB 载入你的 RAM，并锁定其中一部分给显卡。这是正常的。等一下，别关那个窗口。
> 窗口会显示 Strata 正在做什么。

**下一次**，再运行 `START-HERE.bat`（或 `./setup.sh`）。它会立即启动，也不会重复下载任何东西。
关掉它的窗口即可停止模型。`UPDATE.bat`（`./update.sh`）会更新 Strata 而不启动它。更新、Docker、多张显卡、
文件放在哪里以及每一个选项：[docs/INSTALL.md](docs/INSTALL.md)。

**它在你的电脑上跑得快吗？** 运行 `BENCH.bat`（`./bench.sh`）：大约 5-10 分钟，它会用你自己的设置测量
聊天、4K/16K 提示以及一段 agent 风格的会话，并写出一份可分享的报告（[方法](docs/COMMUNITY_BENCHMARKS.md#run-the-benchmark)）。

## 我该选哪个模型？

安装程序会根据你的 RAM 推荐一个。同一个模型有几种尺寸，压缩程度各不相同。更小的尺寸更快，
更大的尺寸稍微更聪明一点。

| Your RAM | Take | Why |
| --- | --- | --- |
| **32 GB** | **Coder** | 它装得进 32 GB，而且是为代码打造的（配一张 24 GB 的显卡时，Q2_0 和 IQ2_XS 也能跑） |
| **48 GB** | **IQ2_XS**（或 Q2_0，最快） | 更大的尺寸装不下 |
| **64 GB** | **IQ2_XS**（推荐），或 IQ3_XXS / IQ3_S | 每种尺寸都装得下；IQ3_S 最好，也最慢 |
| **96 GB 或以上** | **IQ3_S**，或 Unsloth 的 UD-IQ4_XS（约 4-bit） | 在其他东西都开着的情况下，仍有空间容纳最大的尺寸 |

- **[Coder](docs/MODELS.md#coder)：** 一个移除了半数 experts 的编程版本。它达到完整模型 SWE-bench Verified
  分数的 91%（由其作者测量），并装得进 32 GB 的 RAM。它在代码之外的能力较弱，
  包括中文与其他 CJK 文本（#438）。对于这些场景，选 Q2_0、IQ2_XS 或 IQ3_S，它们保留了每一个 expert。
- **[Swift 1.5](docs/MODELS.md#swift-15)：** 一个微调版本，它在回答前"思考"的时间大大缩短。
  你会更快拿到答案，质量大致相当。
- **[Unsloth UD-IQ4_XS](docs/MODELS.md#unsloth-ud-iq4_xs)：** Unsloth 的约 4-bit 版本，质量介于 IQ3_S 与
  UD-Q4_K_XL 之间。一个 94 GB 的下载。当 RAM 少于约 80 GB 时，Strata 会在回答的同时从 SSD 读取其中一部分，
  所以在那里会慢一些（一块 NVMe SSD 有帮助）。
- **[Unsloth UD-Q4_K_XL](docs/MODELS.md#unsloth-ud-q4_k_xl-experimental)**（实验性）：最接近完整模型。
  但 Strata 会在回答的同时从 SSD 读取其中大部分，所以在 64 GB 的电脑上它每秒只写 7-8.5 个 token。
- **[OrcaRouter's Uncensored IQ3_XXS](docs/MODELS.md#orcarouter-uncensored-iq3_xxs)：** 由你手动配置。
  它不在安装程序的菜单里。

尺寸、下载以及哪里装得下：[docs/MODELS.md](docs/MODELS.md)。要在之后添加另一个模型，运行
`SETUP.bat`（Linux：`./setup.sh --setup`）。

## 使用它

<p align="center"><img src="docs/media/runpagoda.png" width="900" alt="Strata 应用的 Monitor 标签页，旁边是一个 coding agent"><br>
<sub>Strata 应用的 <b>Monitor</b>（左），同时一个 coding agent 正在编写视频里那座宝塔庭院（右）</sub></p>

- **在浏览器里：** 打开 `http://127.0.0.1:8080`。它带有 **Chat**、一个对模型和你的
  GPU/CPU/RAM 实时监测的 **Monitor**，以及带有设置和地址的 **About**。
- **你的应用和 coding agent：** 添加一个"OpenAI-compatible"的 provider，base URL 用
  **`http://127.0.0.1:8080/v1`**。任意 API key 和任意模型名都可以。
  - 使用 Anthropic API 的应用：`http://127.0.0.1:8080/v1/messages`（Claude Code：
    `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`）。
  - Codex CLI 及其他使用 OpenAI Responses API 的应用：`/v1/responses`
    （[配置](docs/DETAILS.md#the-responses-api-and-codex-cli)）。
- **思考（Thinking）：** 在聊天菜单或你应用的"reasoning effort"里选择 **off、low、medium 或 high**。
  off 最快，high 最适合难题。
- **图片：** 在设置里对"Images?"回答 yes。然后在聊天里点 **Picture**，或在你的应用里附上图片。
  AMD 显卡在 Linux 上通过处理器读取图片；在 Windows 上目前还不能。
- **从你的手机或另一台电脑：** `START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>`。务必设置一个 key。
- **一次一个请求：** 默认情况下 Strata 一次回答一个请求，其余的排队等待。要同时回答多个，
  设置 `"parallel": 2`（[BATCHING.md](docs/BATCHING.md)）。在一张 12 GB 的显卡上，这会让每个回答都变慢。
- **长提示：** Strata 会完整读取一次聊天的第一条消息，大约每 30,000 个 token 用 1 分钟。
  后续消息几秒内就开始。

更多：[你的聊天存在哪里](docs/INSTALL.md#where-things-are-stored)、[API](docs/DETAILS.md#using-it)。

## 出了问题？

- **我第一次启动 Strata 时电脑卡死了。** 在它加载模型期间这是正常的。等一下，别关那个窗口。
  10 分钟后还卡着？重启电脑，关掉其他程序再试，或者选一个更小的尺寸。
- **它在下载或安装过程中停了。** 再运行一次 `START-HERE.bat`（或 `./setup.sh`）。它会从中断处继续。
- **它非常慢，硬盘指示灯一直闪，或者它说"engine stopped unexpectedly"。** 你的电脑没有足够的空闲 RAM。
  关掉其他程序（浏览器很吃内存），或者选一个更小的尺寸（Q2_0 或 IQ2_XS）。
- **它说端口 8080 已被占用。** Strata 已经在运行了。去找它的窗口。

更多问题及其解决办法：[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)。还是卡住了？开一个
[issue](https://github.com/Niko1221/Strata/issues) 并附上 Strata 文件夹里的 `strata-<model>.log`。
发现了安全问题？请私下上报：[SECURITY.md](SECURITY.md)。

## 它是怎么工作的？

这类模型通常运行在带有数百 GB 显存的服务器上。你的显卡只有 12-24 GB。Strata 通过
**把工作量分摊到你整台电脑上** 让模型装得下。想象一间厨房：你随时要用的东西放在台面上，
其余的则在食品柜里等着。

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="模型的 24,576 个 experts：最繁忙的那些在显卡上，全部在 RAM 里，一张查找表在 SSD 上"></p>

- **模型是一个由 24,576 个小型专家（"experts"）组成的团队。** 每个词只需要其中的 10 个。
- **你的显卡** 保留最常使用的那几千个 experts。**你的 RAM** 存放全部 experts，
  而 **你的处理器** 同时处理其余的部分。**你的 SSD** 存放一张很大的查找表。

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="一个小助手猜测接下来的词；大模型一次性检查它们并保留正确的"></p>

- **先猜，再核对：** 一个小助手猜出接下来的几个词。大模型一次性核对它们。你得到同样的答案，
  却快 1.6-1.8 倍。
- **长文本会被成块地读取**（一次最多 8,192 个 token），速度超过每秒 1,000 个 token。

更详细的解释：[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md)。每个部分及其数字：
[细节](docs/DETAILS.md#how-it-works) 以及那篇 [论文](docs/paper/Strata-Paper.pdf)。

## 致谢与许可证

该模型是由 Qwen 团队出品的 [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)。它由
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)、UkisAI（Swift 1.5）
和 Unsloth 做了压缩。Strata 使用了 [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) 的部分内容。
所有致谢：[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits)。Strata 以 [MIT License](LICENSE) 开源。
少数几个部分以及每个模型各自有其许可证（[是哪些](docs/HOW_IT_WORKS.md#license)）。

## 支持 Strata

Strata 是免费且开源的。如果它对你有用，你可以支持它的开发：

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a></p>
