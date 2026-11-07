# 🥞 pancake-strata

**A fast local LLM engine, built one layer at a time.**

Why *pancake*? A good stack of pancakes is built one at a time: each one goes on top of the last, warm, and the stack
only gets better as it grows. This project is built the same way. It starts from **[Strata](https://github.com/Niko1221/Strata)**,
a fast mixture-of-experts engine for home GPUs. On top of it go the newest speed techniques the open-source
community has published, one after another: an engine from FastStrata, kernels from llama.cpp, ideas from basalt,
colibri, SGLang and others, and our own measured kernels. Every layer is checked before it goes on the stack: it has
to be **faster on a real benchmark** and keep the **same answers**. The result is a full load of pancakes: a stack of
the latest LLM inference technologies, each adding speed.

The target is single-user agent serving (long, growing prompts and tool calls) on ordinary hardware: an
**RTX 3060 12 GB** (CUDA) and an **RX 7900 XT 20 GB** (HIP), a Ryzen 7 5700G and about 57 GB of RAM.

> The original Strata README is kept as [README.strata.md](README.strata.md); its translations are the `README.*.md` files.

## Results so far

Qwen3.6-35B-A3B (UD-Q4_K_XL with its MTP layer) on an RTX 3060 12 GB, greedy, cold runs:

| | Strata (`strata`) | pancake-strata (`strata-poly`) | Gain |
| --- | ---: | ---: | ---: |
| Short chat, writing speed | 64.6 tok/s | **100.7 tok/s** | x1.56 |
| Code edit (5.3K prompt), writing speed | 40.1 tok/s | **151.1 tok/s** | x3.8 |
| 42K-token prompt: reading | 42.1 s | **32.9 s** | x1.28 |
| 42K-token prompt: writing speed after it | 9.7 tok/s | **71.7 tok/s** | x7.4 |
| Largest context on the 12 GB card | 64K (the 131K setup default runs out of VRAM) | **131K** | |

Every test, profile and dead end: [docs/PANCAKE_FINDINGS.md](docs/PANCAKE_FINDINGS.md). The plan:
[MOD_PLAN.md](MOD_PLAN.md).

## The layers of the stack, and what each does

### From FastStrata: the `strata-poly` engine

A second engine beside Strata's, tuned for Qwen3.6-35B-A3B on small NVIDIA cards and measured on an RTX 3060 Laptop.
Ported whole (`src/poly/`), with the kernels it needs merged into Strata's newer ones.

| Technology | What it does | Benefit |
| --- | --- | --- |
| Hybrid expert placement with swaps | Keeps the most-used experts in VRAM and computes the rest on the CPU; swaps experts in while it writes | Runs a 23 GB model on a 12 GB card |
| `--kv-q8` | The attention cache in 8 bits instead of 16 (11.7 KB a token), growing with the conversation | 131K context on 12 GB; more VRAM left for experts |
| `--attn-split` | Decode attention as one pass over the cache on the tensor cores (split-K, flash-decoding style) | Writing stays fast after long prompts (FastStrata: +30% at 37K) |
| `--graph-experts` | Each layer of a decode step recorded as one CUDA graph, card and CPU taking turns through pinned memory | Fewer launch costs: 7-10% less time a step |
| MTP drafting + verification windows | The model's own multi-token-prediction layer guesses 3 tokens ahead; one window verifies them | More tokens a step, same output |
| `--suffix-drafts` | Guesses from every earlier prompt and answer (SuffixDecoding), in a 160 MiB index | Repeated tasks and code edits written much faster |
| `--cpu-multi` | The CPU's K-quant experts read once for all tokens of a window that use them (AVX2) | CPU experts at 83-98% of RAM bandwidth |
| `--elastic-prefill`, `--lend-staging`, `--stage-next` | Prompt batches borrow VRAM from the expert cache, and the next layer's experts are copied while a layer runs | Faster prompt reading on a small card |
| `--conversation-disk`, conversation cache, turn checkpoints | Conversations parked in RAM or on disk, resumed at the last turn | An agent's next turn does not re-read its history |

### From Strata and llama.cpp / ggml

Strata gives the whole base: model loading, the OpenAI/Anthropic-compatible server, setup, the CUDA and HIP builds, the
Flash-Next and DeepSeek engines, and the multi-GPU, router and caching work. The prompt path multiplies the experts
with **llama.cpp's MMQ kernels** (ggml-cuda, quantized matrix products on int8 tensor cores), compiled from the
pinned llama.cpp checkout Strata already uses.

### Our own layers (pancake-strata)

| Technology | What it does | Benefit |
| --- | --- | --- |
| DeltaNet recurrence, 4 columns a warp | Each warp walks 4 state columns of a head through the prompt, sharing their loads; ~1 wave of blocks instead of ~5 on a 28-SM card | `gdn_step` x1.4, the same bits |
| Register-tiled small GEMM | The router and gate products computed 4x4 per thread from registers, each output's sum in the same order | Router x2.0-2.3, the same bits |
| Register softmax for prompt attention | Each thread keeps its scores in registers: a score row is read once, not twice | ~1.5% faster long reads, the same bits |
| Prompt reads, together | The two above | 9-12% faster prompt reading (16K: 10.7 -> 9.6 s, 42K: 35.8 -> 32.9 s) |
| `turn_at` in the server | The server tells `strata-poly` where the last chat turn starts | The next turn resumes from the cache |
| Bench harness `tools/pancake/bench_engines.py` | The same prompts through any engine, every run kept as JSON lines | Every layer is measured before it stays |
| Bitwise kernel tests | `poly_gdn_cols_test`, `poly_gemm_small_test` | A speedup only stays when it gives the same bits |

### Studied, and planned or rejected

basalt (two-stage output head, grammar-constrained decoding, drafting), colibri (cache-aware routing, multitier
memory), SGLang (tree drafting, GPU grammar masks), TensorRT-LLM's blogs (grammar with speculative decoding),
Aviary and lilt's colibri fork: what was taken, planned or rejected, and why, is in
[docs/PANCAKE_FINDINGS.md](docs/PANCAKE_FINDINGS.md#5-tried-and-rejected-with-the-reason).

## Build and run

```sh
export PATH=/usr/local/cuda/bin:$PATH
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86-real
cmake --build build-cuda --target strata strata-poly -j8
./setup.sh --family qwen36 --model UD-Q4_K_XL --build --no-start   # model, tokenizer, a config
```

Then point the server at `strata-poly` with a config whose `"exe"` is `build-cuda/strata-poly` and whose `"args"` are
`--model <gguf> --max-context 131072 --threads <cores> --kv-q8 --attn-split --graph-experts --elastic-prefill 1024
--lend-staging --stage-next --suffix-drafts 256 --cpu-multi`, and start it with
`.venv/bin/python serve/server.py --engine strata --config <that config> --port 8091`.

## Thank you 🙏

This stack exists because these people shared their work openly. Thank you!

| Project | Who | License | What pancake-strata uses |
| --- | --- | --- | --- |
| **[Strata](https://github.com/Niko1221/Strata)** | Niko1221 and the Strata contributors | MIT | The base of everything: engines, server, setup, builds |
| **[FastStrata](https://github.com/Jarson1408/FastStrata)** | Jarson1408 | MIT | The `strata-poly` engine for Qwen3.6, its speed options and its measured optimization log |
| **[PolyStrata](https://github.com/simplexai-labs/PolyStrata)** | simplexai-labs | MIT | FastStrata's parent: the `src/poly` multi-architecture engine |
| **[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)** | Georgi Gerganov and the ggml contributors | MIT | MMQ quantized matrix kernels, the GGUF format, the CPU expert dot products |
| **[basalt](https://github.com/jesdga95/basalt)** | Jesús Gulfo | MIT | Studied: two-stage head, grammar, drafting ideas (planned) |
| **[colibri](https://github.com/JustVugg/colibri)** | JustVugg and contributors | Apache-2.0 | Studied: cache-aware routing, multitier memory ideas |
| **[Aviary](https://github.com/SensAI-PT/aviary-hy3)** | SensAI-PT | Apache-2.0 | Studied: distributed expert placement |
| **[llamAmpere](https://github.com/JakeATX/llamAmpere)** | JakeATX | MIT | Studied: adaptive MTP depth, SJ-KVaRN KV cache, mb-MTP (candidates) |
| **[gpu-unlock](https://github.com/Zagitta/gpu-unlock)** | Zagitta | GPL-2.0 | Recommended (not bundled): full-rate FP16 tensor cores on GeForce, which speeds up attention |
| **[Strata-rdna3](https://github.com/nicejunjie/Strata-rdna3)** | nicejunjie | MIT | Studied: RDNA3 (7900 XT/XTX) kernels and tensor parallel, for the HIP half of the plan |
| **[Speedy](https://github.com/DevDock-AI/Speedy)** | DevDock-AI | MIT | Studied: RTX 3080 (sm_86) Flash-Next presets and the upstream PRs it measured |
| **[StrataFast](https://github.com/eddoursul/StrataFast)** | eddoursul | MIT | Studied: two-stage prompts, a self-trained draft layer |
| **[SGLang](https://github.com/sgl-project/sglang)** | the SGLang team | Apache-2.0 | Ideas: tree drafting, speculative sampling, GPU token masks (planned) |
| **[Unsloth](https://huggingface.co/unsloth)** | Unsloth AI | - | The Qwen3.6 GGUFs with the MTP layer used for every measurement |
| **[Qwen](https://huggingface.co/Qwen)** | the Qwen team, Alibaba | Apache-2.0 | Qwen3.6-35B-A3B |

Code taken from another project keeps its license and credit in the commit that brought it in. pancake-strata itself
is MIT, as Strata is.
