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

| Attention softmax with a thread's scores in registers (`attn_softmax_reg_kernel`) | end to end, fixed placement | read once instead of twice; identical tokens (read16k, edit, long). Read 16K 9.58 -> 9.47 s, 42K 32.78 -> 32.30 s (x1.015), 4K unchanged. `STRATA_SOFTMAX_REG=0` restores the old kernel |

End to end, interleaved old/new/new/old, daily flags (prompt read time, the first two changes):

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

## 6. gpu-unlock on the RTX 3060

[gpu-unlock](https://github.com/Zagitta/gpu-unlock) (Zagitta, GPL-2.0) is a kernel module that loads before the NVIDIA
driver and removes GeForce's half-rate cap on FP16/BF16/TF32 tensor math **with FP32 accumulation** (RTX 3060:
27 -> 51 TFLOPS BF16, per its README). Results are bit-identical; only speed changes. INT8/INT4 math, FP32 CUDA cores and
memory bandwidth are unchanged.

What it speeds up in strata-poly (from the 42K-token profile):

| Part | Math | With gpu-unlock |
| --- | --- | --- |
| Prompt attention, Q.K^T and P.V (cuBLAS, fp16 in, fp32 sums): 8.2 s of 32 s | FP16 MMA, FP32 accumulate | up to 2x faster: about 3-4 s off a 42K read |
| Decode attention `--attn-split` (mma.sync f16 -> f32) | FP16 MMA, FP32 accumulate | faster writing after long prompts |
| A future fused flash-attention prompt kernel | FP16 MMA, FP32 accumulate | its whole tensor-core part doubles |
| MMQ expert and dense products (~30% of a read) | INT8 MMA | no change |
| DeltaNet recurrence, small fp32 GEMMs, CPU experts, decode bandwidth | FP32 / memory / CPU | no change |

To get the most from it here: (1) install it (it needs root: `make install` in the clone, `gpu_unlock.targets=0000:10:00.0`
on the kernel command line, reboot, check `journalctl -k -b | grep gpu_unlock`); (2) keep attention on FP16 tensor
cores with FP32 sums (prompt and decode), which pancake-strata already does and the flash-attention kernel will too;
(3) it makes moving more of the prompt path onto FP16 tensor cores worth more than before (attention first). It does
not change decode speed much: decode is bound by the CPU experts and memory.

## 7. llamAmpere (JakeATX/llamAmpere v0.6.2, MIT): what fits a MoE engine

A llama.cpp fork tuned for RTX 30 cards, mostly for the dense Qwen 27B (the r/LocalLLaMA 2.3 bpw post). Read at v0.6.2:

| Technique | What it is | For pancake-strata |
| --- | --- | --- |
| Adaptive MTP depth (`draft-mtp-adaptive`, 3-4) | The MTP drafter's depth chosen per round instead of fixed | **Candidate, exact.** strata-poly's block always guesses 3; Strata's DraftPolicy already prices windows by measured cost |
| mb-MTP | Keeps a guessed token when the model rates it at least as likely as its own pick | **Lossy opt-in candidate** (as basalt's relaxed drafting); exact drafting stays the default |
| SJ-KVaRN KV cache | fp16 "sink" (first 128 tokens) + fp16 recent tail (4-8K) + a 3-4 bit body | **Candidate:** halves the Q8 cache's VRAM again; on a MoE the room goes to experts (131K context: ~0.9 GB, ~500 more experts on the card). Changes attention numerics: opt-in, KL measured |
| `--kv-gpu-tokens` | Only the recently used part of the context in VRAM | Idea for very long contexts on 12 GB |
| `--moe-skip` | Skips low router weight experts within a 15% budget, first 5 routes protected | Lossy, the same family as FastStrata's `--tail-skip`: opt-in only |
| AI+ / AmperInteger+ trellis weight formats, 2.3 bpw 27B | Weight formats and fine-tuned quants for the dense 27B | Not for this engine (dense model, own file format) |

## 8. Community reports read (r/LocalLLaMA, November 2026)

| Report | What it says | For pancake-strata |
| --- | --- | --- |
| llama.cpp MoE expert cache PR (am17an) | Cache-missed experts computed on the CPU instead of waiting for a PCIe copy; hot experts kept in VRAM from a calibration profile | strata-poly already does both (hybrid CPU/GPU experts, swaps, `--expert-profile`). New test to run: a profile learned on our own agent/coding prompts, loaded at start, against today's start without one |
| KV cache quantization KLD, Qwen 27B (35 quants, 3 engines) | q8_0 KV is essentially free (< 0.00012 KL); 4-bit adds +0.0016-0.0027 KL and grows 15-30% from 4K to 32K context; keys are about twice as sensitive as values; a ~2K recent-token window at higher precision makes 5-6 bit nearly lossless | Confirms `--kv-q8` as the default. Any lower-bit KV (llamAmpere's SJ-KVaRN style) stays opt-in, with keys kept at 8 bits (strata-poly's `--kv-q4` already keeps keys Q8) and a recent high-precision window |
| DSpine (Tencent, arXiv 2609.36173) | A parallel drafter (DFlash family) with causal conditioning injected at every layer: acceptance length 3.77 -> 4.82 on Qwen3-8B, +23% serving throughput in SGLang | Needs a drafter trained for the target model; released only for Qwen3-4B/8B. None exists for Qwen3.6-35B-A3B, and training one is out of scope now. Watch for a release for our models |

## 9. Other Strata forks (read 2026-11-07)

All three tune Strata's **main engine** (Qwen3.8-Flash-Next, also DeepSeek), not the Qwen3.6 `strata-poly` path measured above.

| Fork | Hardware | What it adds | For pancake-strata |
| --- | --- | --- | --- |
| [Strata-rdna3](https://github.com/nicejunjie/Strata-rdna3) (nicejunjie, MIT) | 1-2x RX 7900 XTX (gfx1100) | 987 commits on Strata 8183c6b6 (2026-10-24), ~14.7K lines in src: RDNA3 kernels (`rdna_gr.cu`, `rdna_mmvq.cu`), a fused DeltaNet megakernel, fused IQ MoE prompt kernels, 2-GPU tensor parallel (`tp.cpp`), SDMA copies, idle-queue parking (98+91 W -> 5+13 W idle). 2-3x stock Strata on RDNA3; prefill 3,500+ tok/s one card | **The source for the 7900 XT (gfx1100) half of the plan** once that card is detected again. Ported by hand (upstream moved 1,938 commits since its base). Its MTP draft vocabulary (59.6K tokens, ~12% cheaper draft step, lossless) also applies on CUDA |
| [Speedy](https://github.com/DevDock-AI/Speedy) (DevDock-AI, MIT) | 1-2x RTX 3080 10 GB (sm_86, unlocked) | A squashed import of Strata v0.2.7.1 plus measured defaults and upstream PRs turned on: smaller prompt scratch (#4039: +22-33% prefill), routed prompt stream (#4283: +8-10% on IQ4_XS), INT8-KV prompt attention on IMMA (+2%), KV prefetch, light CPU expert rows a layer late in verify (#2689/#2690), PSI-driven shedding; RDNA3 merged from Strata-rdna3 | Our base (v0.2.12) already has most of these as opt-ins (`STRATA_PF_GROUP_SCRATCH`, `STRATA_PREFILL_ROUTED_STREAM`, `STRATA_PROMPT_ATTN_IMMA`, `STRATA_KV_PREFETCH`, ...): its presets are a ready Flash-Next config for a 3060. Missing here: `STRATA_VERIFY_DEFER_CPU`, `STRATA_KV_RING_ADAPT`, `STRATA_PSI_SHED`. Idea for strata-poly: Q8 KV prompt attention on INT8 tensor cores (IMMA) |
| [StrataFast / "Lamina"](https://github.com/eddoursul/StrataFast) (eddoursul, MIT) | RTX 3090 + RTX 5070 Ti | 812 commits on Strata e88ee39d (2026-10-19): two-stage prompts (`--prompt-split`, 1.24-1.55x at 32K-128K), a draft layer trained on the model's own output, Q5_0 experts. Reports 1.35-1.7x Strata's prompt reads and 1.3-2.1x its answers (against Strata 0.1.43.1) | Ideas for the Flash-Next phase: two-stage prompts, a self-trained draft layer |

Community reports alongside: a 5070 Ti user finds Flash-Next IQ2_XS the best speed/quality point (~150 tok/s, Q2_0 loops
about once a day) and gains 10-20% decode from a VRAM overclock; others insist on Q4 for real work. Speedy's own
comments report gains from 0% (other cards, untuned) to +10% (after tuning) outside its reference machine: forks tuned
on one machine need re-measuring on ours.

## 10. Flash-Next IQ2_XS on the RTX 3060 (main engine, 2026-11-07)

ISTA-DASLab GSQ-RCO IQ2_XS (shard 1 37 GB, n-gram table 27 GB on the SSD), `--expert-cache auto --prefill auto --spec 4
--spec-min-p 0.5 --kv int8 --kv-resident 32768 --max-context 131072`, single cold runs (`2026-11-07-pancake-flashnext*.jsonl`):

| | base (upstream defaults: `STRATA_ROUTE_TAIL_SKIP=7`, lossy) | exact (`TAIL_SKIP=0`) | exact + Speedy's switches |
| --- | ---: | ---: | ---: |
| short, decode | 67.0 | 54.9 | 54.4 |
| edit, decode | 87.7 | 72.8 | 74.9 |
| 42K prompt: read / decode | 38.5 s / 52.8 | 38.5 s / 43.4 | 37.1 s / 47.3 |
| read 4K / 16K | 5.8 / 15.2 s | 5.9 / 15.2 s | 5.5 / 15.7 s |

- Upstream's default tail skip buys 18-22% decode by dropping low-weight experts (answers change). pancake-strata's
  presets use `STRATA_ROUTE_TAIL_SKIP=0`; the tail skip stays an opt-in.
- Speedy's switches (prompt scratch, IMMA prompt attention, KV prefetch, pin-by-need, exchange rotation, HC Q8, adapt
  async/swaps) on this card: within noise to +9%, not the +22-33% prefill measured on 10 GB RTX 3080s. More runs needed.

Second pass, two interleaved rounds each (exact vs exact + Speedy's switches, both with the English draft vocabulary;
`2026-11-08-pancake-flashnext-preset.jsonl`), decode tok/s and read time, mean of 2:

| | exact | + Speedy's switches |
| --- | ---: | ---: |
| short | 49.2 | 54.6 |
| edit | 74.7 | 73.9 |
| after 42K: decode / read | 41.4 / 38.39 s | **46.8 / 37.19 s** (+13% / -3%) |
| read 16K | 16.98 s | 18.65 s |

Speedy's switches pay after long prompts; short answers are within noise (the exact arm's first short run was an
outlier at 41.9). Some of them (`STRATA_PROMPT_ATTN_IMMA` INT8 attention, `STRATA_HC_Q8`) change the numbers slightly,
so they make the **fast** preset, not the default (`run-pancake.sh fast`).

**Correction (2026-11-08):** the engine prints "experimental native Q5_K head", but for IQ2_XS that head is the GGUF's
**IQ4_XS** `output.weight` (337,715,200 bytes = 248,320 rows x 1,360 B; `native_head.hpp`: Q5_K in the Q2_0 / IQ3_XXS
files, IQ4_XS in IQ2_XS), and so is the MTP draft head subset (52.6 MiB = 40,525 x 1,360 B). An earlier note here said
Q5_K; a first A/B of basalt's Q5_K kernels on this model therefore measured unchanged code (noise). See section 10b.

MTP draft vocabulary (exact config, lossless):

| `--mtp-draft-vocab` | short | edit |
| --- | ---: | ---: |
| default (106K tokens, with CJK) | 53.2 | 74.4 |
| `data/draft_vocab_en.bin` (40.5K) | **55.7** | **78.2** |
| `data/draft_vocab_zh60k.bin` (59.6K, from Strata-rdna3) | 53.2 | 72.7 |

The English subset is the best choice for English and code on a 12 GB card (+5%, ~215 MiB less VRAM for the draft
head); the 59.6K list helps Chinese text, not this test.

`STRATA_VERIFY_DEFER_CPU` (Strata PRs #2689/#2690 by lask3802, as re-ported by Speedy; applied 3-way onto v0.2.12):
short 57.7 (+8%), edit 69.0 (-7%) against the exact config; it changes the output. Kept as an opt-in, off by default.

## 10b. basalt's head kernels, and their IQ4_XS ports (2026-11-08)

Ported from basalt (`q5k_head.cu`, `head_proxy.cuh`, MIT) into `src/kernels/cuda/`: the Q5_K head GEMV (bitwise
`native_mmvq`, one warp a row, activations in shared memory), its fused top-1, and the 2-bit proxy two-stage top-1.
They engage on Q5_K heads (Flash-Next Q2_0 / IQ3_XXS files). For the IQ2_XS file in daily use, pancake-strata adds the
same designs for IQ4_XS (`iq4xs_head.cu`):

- **IQ4_XS head GEMV**, bitwise `native_mmvq(23)` and `native_mmvq_il` for 1-8 columns (`iq4xs_head_test`, full head
  2560 x 248,320): 1 column 1.079 -> 1.006 ms (x1.07), 2: 1.462 -> 1.152 (x1.27), 5: 2.788 -> 2.250 (x1.24),
  6-8: x1.17-1.02. At 3-4 columns `native_mmvq_il` is faster (1.13 / 1.25 ms) and stays. Used for the verify head and
  the MTP draft head; `STRATA_Q5K_HEAD=0` / `STRATA_MTP_PROXY=0` turn them off. End to end: greedy tokens identical
  off / on; the speed difference is below this run's noise (a browser was open, load ~5).
- **IQ4_XS 2-bit proxy** for the MTP draft head (31.0 MiB): every row scored from the 2-bit copy (basalt's packing
  and scoring, unchanged), the near-best rescored exactly with the IQ4_XS arithmetic above, the exact best is the
  pick. Greedy tokens identical off / on in both rounds; drafts accepted the same (short 172 of 231-242, edit
  456-463 of 466-483); decode short 52.3 -> 54.0 tok/s (+3%), edit 72.8 -> 76.8 (+5%), mean of 2, noisy (load ~5).

## 10c. Specialized head kernels for every head format the runnable models use (2026-11-08)

The heads, read from each file's GGUF header on Hugging Face (every Qwen3.8-Flash-Next file: 2560 x 248,320):

| Head format | Files | In 57 GB RAM | Kernels |
| --- | --- | --- | --- |
| IQ4_XS | ISTA / Swift IQ2_XS (daily) | yes | `head_fmt.cu` |
| Q5_K | ISTA / Swift Q2_0, IQ3_XXS | yes | basalt's `q5k_head.cu` |
| Q6_K | ISTA / Swift IQ3_S, Coder IQ1_M, Unsloth UD-IQ4_XS | IQ3_S, Coder | `head_fmt.cu` |
| Q4_K | Unsloth UD-IQ1_M | likely | `head_fmt.cu` |
| Q8_0 | Unsloth UD-Q4_K_XL ... Q8_0 | no (72-136 GB) | - |

`head_fmt.cu`: a traits struct per format (native_mmvq.cu's arithmetic, unchanged) and kernels instantiated per format
at compile time - no run-time generic path (owner: "no general port needed"). `head_fmt_test`, 2560 x 248,320:

| Format | GEMV, bitwise native_mmvq and native_mmvq_il, 1-8 columns | Draft pick, 40,525 rows: = exact argmax / time vs the GEMV alone |
| --- | --- | --- |
| IQ4_XS | x1.07 / x1.27 at 1 / 2 columns, x1.24-1.02 at 5-8 | 200/200, 0.113 vs 0.179 ms (x1.59) |
| Q6_K | x1.03 / x1.18, x1.17-1.09 | 200/200, 0.114 vs 0.263 ms (x2.31) |
| Q4_K | x1.02 ..., x1.19-1.07 | 200/200, 0.114 vs 0.182 ms (x1.60) |

At 3-4 columns `native_mmvq_il` is faster and stays. The picks are tested on peaked heads (an activation = a row plus
noise), as a draft head is; on a flat random head no 2-bit copy can tell the rows apart (basalt's own pick then agrees
with the exact argmax about half the time, like this one). End to end on IQ2_XS: both kernels engaged, greedy tokens
identical on / off.

## 10d. More files on this PC: Flash-Next Q2_0 and Qwen3.6 UD-Q3_K_M (2026-11-08)

- Flash-Next **Q2_0** (ISTA-DASLab): shard 1 37.6 GB; its n-gram table is the same file as IQ2_XS's (same SHA-256),
  hard-linked, so it took no extra space. Head Q5_K: basalt's head kernels apply.
- Qwen3.6 **UD-Q3_K_M** (Unsloth MTP GGUF, 17.1 GB, SHA-256 checked): experts IQ3_XXS, so FastStrata's preset leaves
  out `--graph-experts` and `--cpu-multi` (neither reads IQ3_XXS).

Presets (`./run-pancake.sh iq2_xs|q2_0|qwen36|qwen36-q3 [fast]`), single cold runs, decode tok/s (load ~4):

| Preset | short | edit | after 42K | read 42K |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.6 UD-Q3_K_M (poly) | 82.8 | 153.9 | 83.0 | 32.5 s |
| Qwen3.6 UD-Q4_K_XL (poly) | 36.8 (outlier; ~100 before) | 120.6 | 66.5 | 32.6 s |
| Flash-Next Q2_0 | 56.8 | 73.3 | 45.4 | 36.8 s |
| Flash-Next IQ2_XS | 52.0 | 75.1 | 42.6 | 39.2 s |

| Qwen3.8-35B-A3B-Distill Q4_K_M (poly, `--draft-model` = Qwen3.6 Q4_K_XL) | 97.9 | 151.6 | 74.3 | 32.3 s |

The **Qwen3.8-35B-A3B-Distill** (empero-ai, Apache-2.0) is Qwen3.6-35B-A3B fine-tuned on Qwen3.8 2.4T / Flash-Next
answers (not an official Qwen model); its GGUFs (Q4_K_M 21.7 GB, SHA-256 checked) carry no MTP layer, so strata-poly
borrows Qwen3.6's (`--draft-model`; the text stays exact, drafts hold less often). Trained on 8K-token examples: its
card warns of weaker long context. To make room the Avalon Flash-Next model (90 GB, the 7900 XT engine's) was deleted
(owner's choice); `avalon.service` needs it downloaded again to run.

Q2_0 is the faster Flash-Next here (writing +5-9%, long reads -6%); a 5070 Ti user reported Q2_0 looping about once a
day, IQ2_XS not. Qwen3.6 Q3_K_M writes 25-28% faster than Q4_K_XL after long prompts and in edits, at 3-bit quality.

## 11. CUDA and ROCm learning projects and links (read 2026-11-07)

**NVIDIA / RTX 3060 (sm_86):** inference-cuda-kernels (step-by-step LLM ops measured on a 3060), cuda-gemm-benchmark (FP32
/ FP16 tensor-core / BF16 GEMM on a 3060), CUDA_performance_optimization (profiling-driven method), rkinas/cuda-learning
(matching cuBLAS: tiling, vectorized loads, bank-conflict-free shared memory), 100-days-cuda (WMMA, double buffering,
reductions, FlashAttention). These are teaching material, not engines: nothing to port, but they are the reference
techniques for item 8.8 (the fused flash-attention prompt kernel: ldmatrix + mma.sync m16n8k16, cp.async double
buffering, swizzled shared memory). pancake-strata already builds for `sm_86-real` and keeps attention on FP16 tensor
cores with FP32 sums (what gpu-unlock speeds up).

**AMD / RX 7900 XT (gfx1100), for phase 8.10:**

| Project / link | What it is | For pancake-strata |
| --- | --- | --- |
| chelokot/flash-attention-rdna3; ROCm flash-attention (navi branches, e.g. howiejay/navi_support) | FlashAttention-2 for gfx1100 (Triton; CK/WMMA C++) | Algorithm and WMMA layout references for a HIP flash-attention prompt kernel; Triton itself cannot go into this C++/HIP engine |
| uulong950/qingming-z-image-turbo (ROCm/TheRock discussion 8006) | Native HIP/C++ image model on a 7900 XTX: BF16 WMMA kernels, persistent GEMM scheduling, the whole graph on the GPU | Technique reference (persistent GEMM, GPU-resident graph) for RDNA3; not an LLM |
| st1vms/bitsandbytes-rocm-gfx1100 | bitsandbytes (LLM.int8, fine-tuning) fixed for RDNA3 | Not relevant: no PyTorch in this engine |
| charlie12345/vLLM_for_AMD; cadamcat/dual-radeon-vllm (github topic rx7900xt) | vLLM on 7900 XT/XTX, tensor parallel; root cause and fix for an RCCL / PCIe-atomics crash | vLLM is not our engine; the PCIe-atomics finding matters if the 7900 XT shares a PC with another card |
| PyTorch tips (AOTriton, `TORCH_BLAS_PREFER_HIPBLASLT=0`, TunableOp) | PyTorch env vars | Not for this engine, but the claim that hipBLASLt has poor RDNA3 tables is worth an A/B of Strata's HIP BLAS targets (rocBLAS vs hipBLASLt) on the 7900 XT |
| 0xShug0/audio.cpp discussion 317 | 7900 XTX, ROCm 7.2.1: sampling moved to the GPU, no per-token readback: -14.5%, bit-identical | Strata already picks greedy tokens on the GPU; keep sampled decoding on the device too (check when profiling the HIP path) |
| ROCm/amd-rocm-optimization-guide | AMD's HIP optimization guide (coalescing, reductions, multi-GPU) | Reference for the HIP kernels |
| ROCm/rocm-systems | The ROCm systems super-repo: HIP, ROCr, rocprofiler-sdk / -compute / -systems, roctracer, amdsmi | rocprofiler-compute / -systems are the 7900 XT's equivalent of Nsight for phase 8.10 |
| jitllm (jitllm/jitllm, Go, Apache-2.0) | "An operating system for LLM inference": own kernels emitted at load for the model's shapes, experts paged across disk / RAM / VRAM, CUDA PTX, Vulkan, Metal; ROCm untested | No Qwen3.8 Flash-Next support, only ratios against other engines published (no tokens/s). Its expert paging is what Strata already does; a Go engine cannot be merged into this C++/CUDA one. Not taken |
| danielrosehill gist | An AI-written overview of AMD compute stacks (ROCm, Vulkan, ...), for a 7700 XT | General background; its `HSA_ENABLE_SDMA=0` stability tip conflicts with Strata-rdna3's measured SDMA gains: measure, do not copy |

## 12. Next

0. Adaptive MTP depth (llamAmpere), a learned expert profile at start (llama.cpp MoE cache PR), and an SJ-KVaRN-style KV body (opt-in, keys 8-bit, recent window), each measured.
1. A fused flash-attention prompt kernel for D=256 / GQA 8 that reads the Q8 cache and masks causally inline
   (target: the 38% at 42K).
2. Cache-aware routing (colibri's `CACHE_ROUTE`) as a lossy opt-in, with agreement/KL against the true router.
3. The MMQ expert products (~30% of a read).
4. basalt's drafting items and grammar inside speculative decoding (MOD_PLAN.md phases 3 and 6).
5. basalt's two-stage output head, ported for Qwen3.6's 2048-wide Q8_0 head (asked for by the owner; expected ~1-2%).
