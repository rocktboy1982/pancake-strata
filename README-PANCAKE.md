# updated-strata-v0.2.17: measured enhancements from pancake-strata, on top of Strata v0.2.17

This branch (`updated-strata-v0.2.17`) is the upstream release `v0.2.17` of [Niko1221/Strata](https://github.com/Niko1221/Strata) plus the
code that the fork [pancake-strata](https://github.com/rocktboy1982/pancake-strata) (work line `port/rdna3-decode`) ran in its tested
"updated" configuration, and that configuration as a preset. Ported to v0.2.17 on 2026-11-20 (the earlier branch `updated-strata` is the same
work on v0.2.16 and is not changed). It is written as the description of a pull request back to upstream. Every number below is copied
from the fork's test ledger (`TEST_LEDGER.md`, ids `T-nnn`) and names the hardware and the release it was measured on. A number without a
ledger id is not in this file.

**Every performance number in this file was measured on v0.2.16 (2026-11-19).** The code has not been compiled with HIP on v0.2.17 and has
not been measured there; the benchmark against stock v0.2.17 is still to be run. Treat the v0.2.16 numbers as a direction, not as a promise
for v0.2.17 (which has its own changes, see section 5).

Hardware of all ledger numbers: AMD Radeon RX 7900 XT 20 GB (gfx1100), Ryzen 7 5700G, ROCm 10.0, Linux, model Qwen3.8-Flash-Next
(`GSQ-RCO-IQ2_XS` GGUF).

## 1. What this branch contains

Every engine switch is default off; with it unset the code path is v0.2.17's. The branch adds only what the tested configuration needs.

| item | what it is | switch |
|---|---|---|
| RDNA multi-column decode GEMVs | Q4_K / Q5_K / Q6_K / IQ4_XS decode GEMV for 4 to 16 columns on gfx11 (`rdna_mmvq.cu`) | `STRATA_RDNA_Q6K=1` (2.1) |
| Hyper-connection read kernels | gfx1100 read, int8 read, A8, inject-up, fused (`rdna_gr.cu`) | `STRATA_GR_RDNA`, `_Q8`, `_A8`, `STRATA_GR_INJ_UP`, `STRATA_GR_FUSED` (2.4) |
| One-wave MMVQ | gfx1100 one wave per row, emulating the generic four-warp sums | `STRATA_RDNA_MMVQ_WAVE=3` (2.5) |
| Fused combine and wait | CPU-row combine with the shared gate in the router launch; plan wait + copy in one launch | `STRATA_COMBINE_CPU`, `STRATA_WAIT_FUSE` (2.6) |
| Router publish | router top-10 and doorbell publish in one launch | `STRATA_ROUTE_PUBLISH` (2.7) |
| Verify-window expert cap (lossy) | cap on the CPU experts that only draft rows need | `STRATA_VERIFY_ECAP`, `STRATA_VERIFY_ECAP_MIN_ROWS` (2.8) |
| Learned expert profile | `data/expert-profile-pancake.bin` | `--expert-profile` (2.3) |
| Build script | `tools/pancake/build_updated.sh`: Release gfx1100 HIP (or CUDA) build | (2.9) |
| Preset | `configs/pancake/flashnext-iq2_xs-updated-hip.json` | section 3 |

Not ported, and why: `STRATA_POOL_FUSE_PHASES` (v0.2.17 has an equal feature on by default, section 2.10), the status page, the host tuning
scripts, `flag_diag`, the tuned-flags mechanism, the benchmark chain scripts and the experiments. They were not part of the tested
configuration.

## 2. The changes, one section each

### 2.1 RDNA multi-column decode GEMVs (`STRATA_RDNA_Q6K`)

What it does. During decode with speculative drafting the engine multiplies the model's K-quant weights by 4 to 16 activation
columns at once (the verify window). On gfx11 the generic `native_mmvq` kernels read the weights again for every column group.
The new kernels use one wave per output row, decode the weights once, and apply them to every column with `v_dot4` (sudot4),
reducing with DPP and `permlanex16`. Formats: Q4_K and Q5_K (taken from 4 columns up; below that the native kernels are
faster), Q6_K and IQ4_XS (all column counts, 1 to 16).

Enable: `STRATA_RDNA_Q6K=1`. Default off, the unset path is the code of v0.2.17 byte for byte (the dispatch only asks
`rdna_q6k_enabled()` first). CUDA and gfx906 builds compile `rdna_stub.cpp`, whose functions answer "not taken".

Files changed:
`engine/src/kernels/cuda/rdna_mmvq.cu` (new), `engine/src/kernels/cuda/rdna_stub.cpp` (new),
`engine/src/kernels/rdna_mmvq_parity.cpp` (new test, GPU, exits 77 off wave32),
`engine/include/strata/kernels/native_mmvq.hpp` (three declarations),
`engine/src/kernels/cuda/native_mmvq.cu` (four dispatch lines: types 12, 13, 14, 23),
`engine/cmake/strata_gpu_engine.cmake` (sources, test), `terminal/data/settings-register.json` and `docs/SETTINGS.md` (the setting is
registered as `opt-in`).
Left out of the fork's version on purpose (no gain or unproven): the matrix-core variants (`STRATA_RDNA_WMMA_*`) and the bench
knobs `STRATA_RDNA_KQ_ALL`, `STRATA_RDNA_KQ_MAXCOLS`.

Measured (RX 7900 XT, gfx1100):

* Micro benchmark `b6_mmvq_bench`, 300 launches, median, kernel against the native one, 4 / 8 / 12 / 16 columns (T-079): head
  Q5_K +74.6 / +126.3 / +118.8 / +147.8 %; gdn qkv Q4_K +73.6 / +117.9 / +98.0 / +123.3 %; Q6_K +59.0 / +122.1 / +122.4 /
  +98.6 %; IQ4_XS +51..66 / +73..75 / +46..49 / +67 %. Archive: `pancake-docs/research/bench/results-2026-11-15/rdna-16col/b6-rdna0.out`
  and `b6-rdna1.out`. Few-column cost of the native kernels (T-078).
* End to end, confirm n = 12, `bench/specs/51-rdna-16col-confirm.json` (T-081): short 108.5 -> 110.4 tok/s (+1.8 % +-7.8),
  edit 108.3 -> 113.0 tok/s (+4.3 % +-5.0), both "no difference" at the 95 % interval; the edit text is identical in all 12 runs.
  Archive: `pancake-docs/research/bench/results-2026-11-15/rdna-16col/51-rdna-16col-confirm.{stdout,jsonl}`.
  The 16-column limit against the 8-column limit (T-699, spec 104, confirm): 8 columns not faster (short -4.1 % +-5.4,
  edit -1.4 % +-2.1), so 16 stays.
* The port of the fork's RDNA kernel bundle as a whole raised decode from 87-101 to 104-108 tok/s (T-009, no raw archive;
  the bundle also holds the hyper-connection read kernels, which are not part of this branch, see section 6).
  The honest summary: the kernel is much faster in isolation; the end-to-end gain of this one kernel is inside the noise of the
  confirm run. the flag-by-flag screen on the benchmark machine re-checks it.

Exactness. The products and per-16 scales are those of ggml's `vec_dot_*_q8_1`, summed in another order, so the kernels are
"bounded, not bitwise": `rdna_mmvq_parity` requires every output within 1e-4 of the column's largest reference magnitude and two
runs bitwise equal (the fork's 192 of 192 cases, T-081 context).

Checked on this branch: `native_mmvq.cu` and the stub compile with nvcc 12.8 (sm_75, CUDA path); `rdna_mmvq.cu` type-checks with nvcc
using stand-ins for the five gfx11 builtins. It has not been compiled with the ROCm compiler in this change; the benchmark
chain's first step builds it on the box.


### 2.2 hipBLASLt solution table for gfx1100 (selected, not added)

The fork's prompt-path table `gfx1100-hipblaslt-100401.txt` is the same file upstream already ships under `terminal/tools/hip/`
(upstream commit by miskahm, 2026-10-04). The fork measured it as: 11.7K-token prompt read 1,345 -> 1,850-1,920 tok/s (+38 %,
"provisional, noisy machine", T-096, no raw archive). This branch adds nothing here. The preset (section 3) sets
`STRATA_HIPBLASLT_TUNING` to that file. On the benchmark machine's ROCm 10.0 the hipBLASLt runtime is version 100401, so this is the table that
matches: the engine logs `prefill gemm: hipBLASLt tuning enabled (43 rows, gfx1100, version 100401)`, while a 100200 table (the one the
Strata-rdna3 setups name) is refused with `hipBLASLt version mismatch: file=100200 runtime=100401; using hipBLASEx` (engine logs of
T-998). The benchmark preset uses it too (`<REPO>/terminal/tools/hip/gfx1100-hipblaslt-100401.txt`). The engine checks the table's
version header against the installed hipBLASLt and ignores a mismatch, so other ROCm versions fall back safely.


### 2.3 Learned expert profile (data file)

`data/expert-profile-pancake.bin` (196,632 bytes) is the profile the fork learned for the Flash-Next IQ2_XS files with
`tools/pancake/learn_profile.py` (not part of this branch). Passed as `--expert-profile`, it was measured against the profile
shipped with the model: decode 52.8 / 63.4 / 75.8 / 90.0 -> 57.8 / 70.8 / 80.9 / 91.4 tok/s (+6.7 %, every run faster, T-194,
archive `bench/results/2026-11-08-pancake-profile-ab.jsonl`, `bench/prefetch/prof-learned.jsonl`). **That measurement is from an
RTX 3060 12 GB (the fork's first card), not from the RX 7900 XT; it has not been repeated on this build.** The profile belongs
to one model family, so it is used by the preset only.


### 2.4 Hyper-connection read kernels (`STRATA_GR_RDNA`, `_Q8`, `_A8`, `STRATA_GR_INJ_UP`, `STRATA_GR_FUSED`)

What it does. The model's two hyper-connection reads per layer (a down and an up projection of a 4 x 2560 state) run in one decode
window of up to 8 tokens. `rdna_gr.cu` reads them on wave32 gfx11 with a split-K down pass and a partial-folding up pass in another
summation order than `fused_gr_read_multi`. Four optional layers on top: `GR_RDNA_Q8` keeps int8 + fp16-scale copies of the two
projections (made at load in `NativeDense` from the GGUF's Q8_0 or BF16 rows) and leaves the BF16 copies off the card;
`GR_RDNA_A8` uses int8 activations and dot4; `GR_INJ_UP` gives the four inject rows their own blocks; `GR_FUSED` runs both passes in one
launch with a grid barrier. The geometry is the fork's Flash-Next one (n_embd 2560, hc 4, hc_lr 320) as in `rdna_gr.cu`.

Enable: `STRATA_GR_RDNA=1` (BF16 read); add `STRATA_GR_RDNA_Q8=1` for the int8 weights, then `STRATA_GR_RDNA_A8=1`,
`STRATA_GR_INJ_UP=1`, `STRATA_GR_FUSED=1` in that order (each needs the one before). Default off; the unset path is v0.2.17's.

Files: `engine/src/kernels/cuda/rdna_gr.cu` (new), `rdna_stub.cpp` (CUDA / gfx906 answers), `engine/src/kernels/rdna_gr_parity.cpp` (new test,
five variants), `fused_gr.hpp` / `fused_gr.cu` (the `r8_down` / `r8_up` fields and one call in `fused_gr_read_multi`), `weights.hpp`
(`hc_r8`), `native_dense.cpp` (load of the int8 copies, pricing, `hc_bf16_view` returns a BF16 image from them for the prompt path and the
single-token read), `prefill.cpp` (two conditions), `verify.cpp` (two pointers), `generate.cpp` (the BF16 copies are not loaded under
`GR_RDNA` + `GR_RDNA_Q8`, single GPU only). Left out of the fork's version: its debug check, bench selectors and `STRATA_GR_UPPRE`.

Measured (RX 7900 XT, Flash-Next IQ2_XS): decode 87-101 tok/s with the stock Strata kernels, 104-108 with the ported RDNA kernels, about 112 with
the fusions (T-009, no raw archive; Strata-rdna3 109-119, Avalon 95-100, stock Strata 75-77 in the same row); the round that chose the preset
stack (T-378: 128.7 short / 103.6 edit, one run). The ledger has no row that isolates `GR_RDNA` from the rest of that bundle, and no passing
quality gate for the int8 variants (`GR_RDNA_Q8`, `_A8` are in the fork's list of lossy switches); that is why only `GR_RDNA=1` is in the
shipped profile and the rest is left to the benchmark. `rdna_gr_parity` bounds the BF16 and int8 reads against the default read (1e-4, and 3e-2 for A8).


### 2.5 `STRATA_RDNA_MMVQ_WAVE=3`

One wave per output row that emulates the generic four-warp layout in registers: the same partial sums in the same order, no LDS exchange and
no block barrier, one kernel for 1 to 16 columns (so a column's result does not depend on how many verify tokens ride along). Upstream's
wave kernel exists for gfx906 only; this is a new gfx11 branch of the same macro in `native_mmvq.cu`. Short projections take it only for
`n_in <= 1024` or `(n_in <= 2560 and n_out >= 1024)`, as in the fork. Measured: mode 1 (all shapes, another summation order) was 4.5 % slower
than mode 3 on short and 2.5 % on edit (T-072, spec 40, screen n = 2); the bundle number is T-009. Enable `STRATA_RDNA_MMVQ_WAVE=3`.


### 2.6 `STRATA_COMBINE_CPU` and `STRATA_WAIT_FUSE`

With CPU experts, the verify window copies the CPU rows from the mapped buffer, adds the GPU hit rows, then combines. `COMBINE_CPU` reads the
CPU rows straight from the mapped buffer and the GPU's hit rows in one combine kernel and carries the shared expert's gate as an extra block of
the router GEMV launch: bitwise the launches it replaces. `WAIT_FUSE` (needs `COMBINE_CPU` for its second half) puts the plan wait and the plan
copy in one launch and the CPU-share wait inside the combine, with a system-scope acquire in every wave. Files: `native_moe.cu` / `.hpp`,
`native_bf16.cu` (`bf16_f32_mmvf_multi_kernel` gets the aux block), `verify_kernels.cu` / `.hpp`, `verify.cpp`.
Measured by the fork: `WAIT_FUSE` 21.85 to 21.60 ms per window, byte-identical output (the commit's own measurement on an IQ3 model, nicejunjie);
turning the fused wait off lost (T-077); the bundle is T-009. v0.2.17 already has a different plan wait + copy (`wait_copy_i32_from_mapped`), which
stays the default.


### 2.7 `STRATA_ROUTE_PUBLISH`

The router's top-10 and the doorbell publish in one launch for 1 to 8 rows (`native_router_top10_publish_res`). Measured by the fork: 21.60 to
21.51 ms per window, identical output (0.4 %: below the 2 % bar on its own; T-009). The fork found it unsafe together with expert deferral
(T-317), which this branch does not have. Not taken with `STRATA_ROUTE_PRIOR`, a resident-route margin, a device plan, `--remote-expert-opt` or
`STRATA_DOORBELL_STORE`.


### 2.8 `STRATA_VERIFY_ECAP` and `STRATA_VERIFY_ECAP_MIN_ROWS` (lossy)

In a decode verify window each draft row (rows 1 to T-1, never row 0) reads at most N CPU-missed experts that the committed row and the earlier rows
do not read anyway, kept by routing rank; the others contribute zero (host side only, no graph change). `STRATA_VERIFY_ECAP=0` with
`STRATA_VERIFY_ECAP_MIN_ROWS=9` is the fork's daily setting: the cap acts only in windows of 9 or more rows (the long lookup windows).
Measured (RX 7900 XT, spec 60, 6 clean runs per arm and case, T-588): decode edit 165.3 +-6.2 tok/s = +41.7 % +-5.6 over no ECAP, short +2.8 % +-2.6;
token forcing PASS (mismatch 1.583 % against 1.467 / 1.722 % for the stock arms, -0.011 points, T-589); tool-call task gate PASS (99.4 % success,
tool-call validity 100 %, T-591). **The two gates ran with the fork's grammar guard, which exempted a tool call or JSON reply in progress from the cap;
this branch has no such guard, so on this branch the cap also acts inside a tool call. That is not covered by T-591**; run `quality_gate.py` and the
task gate on the release build before enabling it by default. Files: `verify_ecap.hpp` (the rule) and `verify_ecap_test.cpp` (built and run on
the PC: ok), `expert_source.cpp` / `.hpp`, `verify.cpp` / `.hpp`, `generate.cpp`. The fork's ECAP v2 (per-row lists, probability gate, window
policy) is not ported.


### 2.9 Optimised build: `tools/pancake/build_updated.sh`

`tools/pancake/build_updated.sh [SRC] [BUILD]` (environment: `BACKEND=hip|cuda`, `ROCM`, `GFX`, `LTO=1`, `TESTS=1`). The
header of the script lists each flag with the code that justifies it. In short: Release with `-O3 -DNDEBUG`; HIP on, CUDA off;
`gfx1100` only; `STRATA_AMD_COMMUNITY_TUNING=ON`; `STRATA_WINDOW_MAX_T=16` (the 16-row verify window the 4..16 column kernels
serve); `STRATA_NATIVE_EXPERTS=ON`. Deliberately not added: a global `-march=native` (the CPU quant layer keeps AVX-only files with
AVX2 off), `-ffast-math` / `--use_fast_math` on HIP (the tree sets `-ffp-contract=off` on files whose rounding ties matter), and
LTO (opt-in `LTO=1`, no claim that it is faster). `BACKEND=cuda` keeps the CUDA build available. The effect of these flags is
not measured in isolation; the benchmark of section 4 measured the whole updated build against the plain release.


### 2.10 `STRATA_POOL_FUSE_PHASES` (skipped: v0.2.17 has it)

The fork's switch ran a layer's gate/up and down phases of the CPU expert pool as one phase (ledger T-224). v0.2.17 contains the same
idea: `STRATA_POOL_DEP` (mode 7 of `ExpertPool`, `DepExpert` in `pool.hpp`, `dep_task`): one batch per native layer call, a down task waits
only for its own expert, and it is **on by default** (`STRATA_POOL_DEP=0` is the way back). So the fork's switch was not ported; the tested
configuration's `STRATA_POOL_FUSE_PHASES=1` is replaced by v0.2.17's default. The fork's T-224 number was measured with its own
implementation on v0.2.16 and does not carry over to upstream's.

## 3. The published configuration

The configuration is the difference between the plain release configuration (model arguments only, no environment) and the tested
"updated" arm of specs 220 / 222. It is the preset `configs/pancake/flashnext-iq2_xs-updated-hip.json` (placeholders `<REPO>` and
`<STRATA_DATA>` like the other presets in `configs/`). Everything else is the release's own setting: v0.2.17's defaults stay.

Arguments added to the model arguments (`--pack`, `--native`, `--ple-gguf`, `--expert-cache auto`, `--prefill auto`, `--spec 4`,
`--spec-min-p 0.5`, `--mtp`, `--max-context 131072`, `--kv int8`, `--kv-resident 32768` are the same in both arms):

* `--expert-profile <REPO>/data/expert-profile-pancake.bin (replaces the release profile terminal/data/expert-profile.bin)`
* `--adapt-swaps 32`
* `--pool-affinity measure`
* `--pcie-frac 0`
* `--adapt-every 1`
* `--lookup-max-t 12`
* `--conversation-cache-mib 2048`
* `--conversation-cache-min-tokens 8192`

Environment of the updated arm (the plain arm sets none):

| variable | value |
|---|---|
| `STRATA_ROUTE_TAIL_SKIP` | `0` |
| `STRATA_PIN_BY_NEED` | `1` |
| `STRATA_EXPDEC` | `1` |
| `STRATA_EXCHANGE_ROTATE` | `1` |
| `STRATA_ATTN_DAILY` | `1` |
| `STRATA_PROMPT_ATTN_IMMA` | `1` |
| `STRATA_KV_PREFETCH` | `1` |
| `STRATA_PF_GROUP_SCRATCH` | `1` |
| `STRATA_PF_HC_REGION` | `1` |
| `STRATA_HC_Q8` | `0` |
| `STRATA_PREFILL_CPU_SHARE` | `0.6` |
| `STRATA_HIP_WMMA` | `1` |
| `STRATA_SELECT_WMMA` | `1` |
| `STRATA_SH_STREAM` | `0` |
| `STRATA_PREFILL_AUTO_MAX` | `32768` |
| `STRATA_QFUSE` | `1` |
| `STRATA_VERIFY_QDEDUP` | `1` |
| `STRATA_PF_FUSED` | `1` |
| `STRATA_STAGER_RING` | `64` |
| `STRATA_PREFILL_CPU_SHARE_MAX` | `4096` |
| `GPU_MAX_HW_QUEUES` | `6` |
| `STRATA_GDN_HEAD` | `1` |
| `STRATA_GDN_CHUNKED` | `1` |
| `STRATA_GDN_PP` | `1` |
| `STRATA_GDN_CONVL2` | `1` |
| `STRATA_GDN_NOY` | `1` |
| `STRATA_PF_GEMM` | `1` |
| `STRATA_PF_PAD` | `1` |
| `STRATA_HC_UPMIX` | `1` |
| `STRATA_PF_HCDOWN` | `1` |
| `STRATA_FUSE_CONVAB` | `1` |
| `STRATA_HEAD_MIX_MULTI` | `1` |
| `STRATA_PA_FAST` | `1` |
| `STRATA_PF_F16` | `1` |
| `STRATA_PREFILL_RING` | `192` |
| `STRATA_KV_HOST_DMA` | `1` |
| `STRATA_ADAPT_OVERLAP` | `1` |
| `STRATA_QSA_ROWS` | `1` |
| `STRATA_HIPBLASLT_TUNING` | `<REPO>/terminal/tools/hip/gfx1100-hipblaslt-100401.txt` |
| `STRATA_RDNA_MMVQ_WAVE` | `3` |
| `STRATA_GR_RDNA` | `1` |
| `STRATA_GR_RDNA_Q8` | `1` |
| `STRATA_RDNA_Q6K` | `1` |
| `STRATA_VERIFY_ECAP` | `0` |
| `STRATA_VERIFY_ECAP_MIN_ROWS` | `9` |
| `STRATA_GR_RDNA_A8` | `1` |
| `STRATA_GR_INJ_UP` | `1` |
| `STRATA_GR_FUSED` | `1` |
| `STRATA_COMBINE_CPU` | `1` |
| `STRATA_WAIT_FUSE` | `1` |
| `STRATA_ROUTE_PUBLISH` | `1` |
| `STRATA_ENGINE_NICE` | `0` |
| `STRATA_EXPDEC_DOWN` | `1` |

Every variable above exists in the v0.2.17 code (engine or terminal) or is added by this branch. Variables already on by default in v0.2.17
are listed anyway because the tested arm set them explicitly. `STRATA_PREFILL_EQUAL_TAIL`, `STRATA_AGENT_COUNTERS`, `--grammar` and
`--mtp-draft-vocab` of the fork's daily preset are not part of it. The values come from the spec files of the fork repository
(`bench/specs/220` and `222`); the arm's configuration file on the benchmark machine was not read for this port (no access).

## 4. Results (v0.2.16, 2026-11-19, ledger T-991 to T-996)

Ran on the benchmark machine with `abtest.py`: plain release v0.2.16 with the model arguments only against the "updated" arm (this
configuration on the v0.2.16 version of this branch, `updated-strata`). Numbers are mean +- 95 % interval in tok/s. Noise caveat: every start
was flagged noisy (foreign CPU load 7-23 %, `kcompactd0` at the top); screens kept the quietest attempt. Only the confirm of spec 220 has a
tight interval; the long-prompt rows are single requests (n = 1): a direction, not a size.

Lossy arm `updated` (text differs from plain):

| case | plain v0.2.16 | updated | difference | n, tier | ledger |
|---|---|---|---|---|---|
| short, decode tok/s | 69.1 +-5.4 | 102.0 +-4.4 | +47.6 % +-11.0 (better) | n = 6 each, confirm | T-991 |
| edit, decode tok/s | 83.7 +-2.5 | 177.7 +-6.2 | +112.4 % +-9.6 (better) | n = 6 each, confirm | T-991 |
| 10K prompt tok/s / decode | 608 / 67.8 | 1,086 / 66.7 | +78.6 % / -1.6 % | n = 1, screen | T-992 |
| 32K prompt tok/s / decode | 688 / 67.4 | 1,580 / 100.5 | +129.7 % / +49.1 % | n = 1, screen | T-992 |
| 64K prompt tok/s / decode | 680 / 52.4 | 2,400 / 63.1 | +252.9 % / +20.4 % | n = 1, screen | T-992 |
| 100K prompt tok/s / decode | 660 / 52.0 | 1,996 / 89.0 | +202.4 % / +71.2 % | n = 1, screen | T-992 |

Exact arm `updated-exact` (the lossy switches off, still `--lookup-max-t 12`; screen only, spec 223 was cut off): short 74.1 +-39.4 -> 101.8 +-62.9
(+37.4 %, n = 2), edit 86.1 +-99.8 -> 135.8 +-195.1 (+57.8 %, no difference), prompt speed 10K to 100K +139 % to +195 % (n = 1, partial) (T-994, T-996).

Quality gate (token forcing, 48 requests, 9,410 forced tokens, plain run twice with 1.562 % / 1.360 % mismatch; T-995): both arms are **REJECT**
on the rate rule: updated-exact 3.379 % (+1.92 points, 90 % interval [+1.23, +2.74], limit 2.625), updated 3.964 % (+2.50 points [+1.67, +3.43]);
the event rule passes for both. The speed gain is real on v0.2.16, but the output changes more than the release changes against itself, so by the
fork's rules neither arm may be a default. Text identity cannot tell exact from lossy on this model (the plain release against itself also differs).
A flag-by-flag screen of the configuration is in progress on the benchmark machine; its results are not in this file.

## 5. Differences between v0.2.16 and v0.2.17 that touch this work

* v0.2.17 sets `STRATA_ADAPT_FETCH=2` by default for 12.x NVIDIA cards of 16 GiB or less (`adapt_fetch_default_test`); it does not apply to gfx1100.
* `STRATA_POOL_DEP` is on by default (section 2.10).
* `verify.cpp`, `expert_source.cpp`, `prefill.cpp`, `generate.cpp` and the CPU pool changed upstream; this branch is merged on top of them.
  The performance of the combined code is not measured.

## 6. Compatibility, risks and what is not included

* Defaults. With no switch set the engine path is v0.2.17's.
* Unsupported hardware. The RDNA kernels exist only in a wave32 HIP build (gfx10 / gfx11 / gfx12); CUDA and gfx906 builds take the stub. Tuned for and tested on gfx1100 only.
* Exactness. `STRATA_RDNA_Q6K` changes the summation order (bounded error); `STRATA_VERIFY_ECAP` and `--lookup-max-t 12` change the output; the quality gate says REJECT (section 4).
* ECAP and tool calls. The fork's gates ran with a grammar guard that exempted a tool call in progress from the cap; this branch has no such guard.
* Build. **None of the HIP code of this branch has been compiled by the ROCm compiler on v0.2.17.** It was type-checked with nvcc through stand-ins for the gfx11 builtins and with clang HIP syntax checks; the HIP link and the parity tests (`rdna_mmvq_parity`, `rdna_gr_parity`) run on the benchmark machine later.
* Not ported: the fork's ECAP v2, `STRATA_GR_UPPRE`, the debug and bench selectors of `rdna_gr.cu`, `STRATA_AGENT_COUNTERS`, `STRATA_PREFILL_EQUAL_TAIL`, the fork's grammar guard.
* Licence compatibility. All sources are MIT (section 7); no AGPL code is included.

## 7. Credits and authorship

Licence of this work: MIT, like upstream (`LICENSE`: Copyright (c) 2026 Niko1221 and the Strata contributors).

| item | source | licence | notes |
|---|---|---|---|
| item | source | licence | notes |
|---|---|---|---|
| RDNA multi-column GEMVs (2.1) | **nicejunjie** (Junjie Li), fork Strata-rdna3: commits `863804599` (2026-10-11, "multi-column Q6_K and IQ4_XS decode GEMV") and `ceb14be4e` (2026-10-11, Q4_K / Q5_K from 4 columns), as they appear in the pancake-strata history | MIT as a fork of MIT Strata; the fork's own licence file was not re-checked for this change: confirm before merging | The kernel design is theirs. pancake-strata (rocktboy1982), original work on top: HIP-only guards and CUDA stubs (`413e1b968`, 2026-11-08), the parity tests (`76f114fa4`, 2026-11-09), the stub file (`20733fb77`, 2026-11-09), the 9 to 16 column extension (`82687a7bd`, 2026-11-15); removal of the unproven variants and the port to upstream's layout in this branch |
| hyper-connection read kernels (2.4) | **nicejunjie** (Junjie Li), Strata-rdna3: `89da64b93` (2026-10-10, read + DPP shuffles), `042a9ef7b` (2026-10-11, int8 read), `9044c7724` (2026-10-11, A8), `e56ac57e5` (2026-10-11, INJ_UP), `06d959d7a` (2026-10-11, FUSED), as they appear in the pancake-strata history | MIT as a fork of MIT Strata (the fork's licence file not re-checked: confirm before merging) | kernel design is theirs. pancake-strata (rocktboy1982): the q8_1 image for QFUSE (`6d9a1a43b`, 2026-11-08), HIP-only guards and stubs (`413e1b968`, `20733fb77`), the parity test (`76f114fa4`), and the port to upstream's layout in this branch (loader, `hc_bf16_view`, generate.cpp skip) |
| `STRATA_RDNA_MMVQ_WAVE=3` (2.5) | **nicejunjie**, Strata-rdna3 (the wave32 exact kernel), as in the pancake-strata history | MIT as above | mode selection and the shape rule are the fork's; port by hand into upstream's macro |
| `STRATA_COMBINE_CPU` (2.6) | **nicejunjie**, Strata-rdna3, ported into pancake-strata as `2745ea818` (rocktboy1982, 2026-11-08) | MIT as above | `STRATA_WAIT_FUSE`: nicejunjie `297243a5e` (2026-10-11), cherry-picked in the fork |
| `STRATA_ROUTE_PUBLISH` (2.7) | **nicejunjie**, Strata-rdna3: `50ec2ade3` (2026-10-11) | MIT as above | re-applied by hand |
| `STRATA_VERIFY_ECAP` (2.8) | idea and rule adapted from **jmnargi** (Strata-V100 `24894e676b`, `7c5041c358`, `376355453f`; upstream PR 3545, MIT as read in the fork's README section 9); the per-draft-row, CPU-missed-only form is pancake-strata's | MIT | `2d44d31ec` (2026-11-10, rocktboy1982), the MIN_ROWS gate `9a4cd7568` (2026-11-15, rocktboy1982); ledger T-588, T-589, T-591 |
| hipBLASLt table (2.2) | **miskahm**, upstream commit `7984688e3` (2026-10-04) | MIT (upstream) | already in upstream; measured by the fork (T-096) |
| learned expert profile (2.3) | pancake-strata (rocktboy1982), original work | MIT | first commit `3638b8cee` (2026-11-08); learner `a4f305b82` (2026-11-08); ledger T-194 |
| build script (2.9) | pancake-strata (rocktboy1982), original work | MIT | `tools/pancake/build_updated.sh` |
| adopted values `--pcie-frac 0`, `--lookup-max-t 12` | pancake-strata (rocktboy1982): measured settings of existing upstream options | MIT | T-197, T-268, T-126, T-127 |

Proof of authorship of the "original work" rows: the commits are in the public history of https://github.com/rocktboy1982/pancake-strata
(branch `port/rdna3-decode`), authored by `rocktboy1982` on the dates given, and every measurement has a ledger row in `TEST_LEDGER.md` of that
repository. The commits of this branch carry the trailer `Co-Authored-By: Claude Sonnet 5.5`.

Ideas read but not used as code: **jdominick05/caminus** is licensed AGPL-3.0-or-later; the fork took ideas only, nothing from it is in this branch.

## 8. Commits of this branch

The first commit of the branch is a snapshot of the tree of the upstream tag `v0.2.17` (upstream commit
`028e6826bdb152be2d0579818c2ede6fe7da984c`), parented on this fork's `main`, with `.github/workflows` left as in `main` (the account used to push
has no `workflow` scope). The commits after it are the changes described above; each commit message names the ledger ids it relies on. The
real history (the same commits on the tag itself) is the local tag `updated-strata-v0.2.17-realhistory`. A pull request to upstream should be made
from the commits rebased onto the tag `v0.2.17`.
