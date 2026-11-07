# pancake-strata: tests and findings

Everything measured while building pancake-strata, in the order it was done. Hardware for every number:
**RTX 3060 12 GB** (GA106, sm_86, PCIe 3.0 x16 behind a Ryzen 7 5700G), **Ryzen 7 5700G** (8 cores / 16 threads,
AVX2, 16 MiB L3), **57 GB DDR4** dual channel, Linux (kernel 7.0), NVIDIA driver 595.99, CUDA 13.1, gcc 15.2.
Model: **Qwen3.6-35B-A3B UD-Q4_K_XL** with its MTP layer (Unsloth), greedy decoding.

Raw results (one JSON line per run, the generated tokens included) are in `bench/results/2026-11-07-pancake-*.jsonl`.
The harness is `tools/pancake/bench_engines.py`; the kernel tests are `poly_gdn_cols_test` and `poly_gemm_small_test`.

## How to reproduce

```sh
# build (CUDA, RTX 30 series)
export PATH=/usr/local/cuda/bin:$PATH
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86-real
cmake --build build-cuda --target strata strata-poly poly_gdn_cols_test poly_gemm_small_test -j8

# kernel tests: bit-for-bit against the previous kernels, and their times
./build-cuda/poly_gdn_cols_test 4096
./build-cuda/poly_gemm_small_test

# end-to-end bench (the same prompts through any engine that speaks --serve)
M=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf; T=<pack>/tokenizer
.venv/bin/python tools/pancake/bench_engines.py --name poly --runs 1 --tokenizer $T --out bench/results/x.jsonl -- \
  build-cuda/strata-poly --model $M --max-context 131072 --threads 8 --kv-q8 --attn-split --graph-experts \
  --elastic-prefill 1024 --lend-staging --stage-next --suffix-drafts 256 --cpu-multi
```

Cases: `read1k`..`read16k` (fresh prompts of growing length, 16 new tokens), `short` (34-token chat, 256 new),
`edit` (5.3K-token code rewrite, 512 new, copy-heavy), `long` (41.8K-token prompt, 128 new). "Cold" = the first run of
a prompt in a fresh engine; later runs hit the conversation cache and the suffix drafter and are not comparable.

## 1. Base Strata vs strata-poly (FastStrata's engine)

| Case | `strata`, 16K ctx | `strata`, 64K ctx | `strata-poly`, 131K ctx |
| --- | ---: | ---: | ---: |
| short, decode | 64.6 tok/s | 53.2 tok/s | **100.7 tok/s** |
| edit, decode | 40.1 tok/s | 39.4 tok/s | **151.1 tok/s** |
| 41.8K prompt: read / decode | - | 42.1 s / 9.7 tok/s | **35.8 s / 71.7 tok/s** |

- `strata` keeps its KV cache in fp32 (45 KB a token). The 131,072 context Strata's setup writes for a 12 GB card runs
  out of VRAM at start (`cudaMalloc ... out of memory`, q36_engine.cu:1128): an upstream setup bug. At 64K no routed
  expert fits in VRAM ("0 of 256 per layer"), at 16K 26 of 256.
- `strata-poly` keeps a Q8 cache (11.7 KB a token) that grows with the conversation, and ~4,500 of 10,496 experts in VRAM.
- The two engines' greedy texts part after 25 to 81 tokens: different kernels and cache precision. Expected.

## 2. Where the time goes (strata-poly, daily flags)

**Prompt reading is GPU-compute bound, not PCIe bound.** Nsight Systems, a 15K-token prompt (10.4 s):
GPU kernels 9.7 s. `gdn_state_reg_kernel` (Gated DeltaNet recurrence) 21.6%, `gemm_small_kernel` 12.9%, MMQ expert
products ~30%, attention ~16%. The read rate (~1,400 tok/s past 4K tokens, about 1 s fixed per read) does not change
with the prompt batch knobs (`--prompt-batch 2048` +5-8% read and -4% decode; `--elastic-prefill 512`, `--cpu-share 0`:
no change; `--elastic-batch` is capped at 4096).

**At 42K tokens attention is the largest cost:** Q.K^T GEMM 4.04 s, P.V GEMM 4.13 s, softmax 3.10 s, rescale and
Q8->fp16 0.81 s: about 12 s of 32 s (38%).

**Decode: the CPU experts are about half of it.** `STRATA_CPU_STATS=1`: 4.1 s of 8.75 s of writing, at 33.4 GB/s,
close to what dual-channel DDR4 delivers. They can only go faster by reading fewer bytes.

## 3. CPU threads: per thread or per core?

`--threads N`, decode tok/s (short / edit):

| N | 4 | 6 | 8 | 12 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| short | 91.5 | 100.3 | 98.0, 93.5 | 99.0 | 92.9 |
| edit | 139.5 | 150.4 | 154.7, 150.7 | 153.5 | 150.1 |

6 to 12 are equal within noise; 4 and 16 are slower. The experts are RAM-bound, so a core's second hardware thread
adds no bandwidth. Keep **one thread per physical core** (8). strata-poly does not pin its threads.

## 4. Kernel changes (exact: the same bits)

| Change | Test | Result |
| --- | --- | --- |
| DeltaNet: 4 state columns of a head per warp (`gdn_state_cols_kernel`) | `poly_gdn_cols_test` | gdn_step x1.38 (1K tokens), x1.44 (4K); same bits for out, state and snapshots. 2 columns x1.27, 8 columns x1.05 |
| Small fp32 GEMM register-tiled 4x4 (`gemm_small_tiled_kernel`) | `poly_gemm_small_test` | router 2048->256: x2.26 (4K tokens), x1.99 (1K); same bits. Slower for 32 outputs at few tokens, so used only for >= 64 outputs and >= 256 tokens |

End to end, interleaved old/new/new/old, daily flags (prompt read time):

| | 4K | 16K | 42K | edit (5.3K) |
| --- | ---: | ---: | ---: | ---: |
| before | 3.12 s | 10.74 s | 35.76 s | 3.97 s |
| after | 2.86 s | 9.60 s | 32.89 s | 3.63 s |

**Exactness:** with the expert cache on, strata-poly is *not* run-to-run deterministic (old vs old already differs
after 9-81 tokens): the cache resizes and swaps experts by timing and free VRAM, and CPU and GPU round differently.
With a fixed placement (`--no-gpu-experts --expert-swaps 0`) the greedy tokens of read4k, read16k, short and edit are
identical old/old, new/new and old/new. Exactness checks must use a fixed placement.

## 5. Tried and rejected (with the reason)

| Idea | Finding |
| --- | --- |
| basalt's exact two-stage output head | The head is Q8_0, 2048 x 248,320 (540 MB, ~1.5 ms a window at 360 GB/s); screening by 6 of 8 bits saves ~0.4 ms of a 20-30 ms window: ~1-2%. basalt's kernel is also written for 2560-wide rows. Skipped. |
| `kAttendResults` tuned to the 3060's 3 MiB L2 | 42K read: 1 MiB 42.0 s, 2 MiB 35.7, 4 MiB 34.9, **8 MiB (default) 32.7**, 16 MiB 33.3. The default is best; the knob was removed. |
| ggml's flash attention (`fattn`) for prompt reading | Needs a causal mask buffer and fp16 copies of a Q8 cache from its pool: ~130 MB at 42K tokens, with ~186 MiB of VRAM free. Not taken; an own fused kernel reading the Q8 cache is the plan. |
| `--attn-split` for prompt batches | It is split-K decode attention, at most 2 tokens a block: at 42K it would read the cache once per 2 tokens. Not suitable. |
| colibri (JustVugg/colibri) | Built for models larger than RAM (disk streaming, pin set, LRU). Its Qwen3.6 decode is ~30 tok/s on an RTX 3070, strata-poly 100+ on the 3060. Idea kept: `CACHE_ROUTE` (cache-aware routing; lossy, opt-in candidate). |
| Aviary (SensAI-PT/aviary-hy3) | Multi-machine cluster on colibri, activations over the LAN. No gain on one PC; 40 layers of LAN round trips would cost more than a whole token takes now. |
| lilt/colibri-fork | A mirror of colibri v1.5.0, no own features. |
| TensorRT-LLM, vLLM, SGLang, LMDeploy, MLC-LLM, PowerInfer, KTransformers, ExLlamaV2/V3, TGI, LightLLM | Keep the model in VRAM, need far more VRAM/RAM, special model files, or are archived. See MOD_PLAN.md. |
| IPEX-LLM, FPGA projects (LUT-LLM, llm-oneapi-fpga) | Other hardware. |

## 6. Next

1. A fused flash-attention prompt kernel for D=256 / GQA 8 that reads the Q8 cache and masks causally inline
   (target: the 38% at 42K).
2. Cache-aware routing (colibri's `CACHE_ROUTE`) as a lossy opt-in, with agreement/KL against the true router.
3. The MMQ expert products (~30% of a read).
4. basalt's drafting items and grammar inside speculative decoding (MOD_PLAN.md phases 3 and 6).
