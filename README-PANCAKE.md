# updated-strata: measured enhancements from pancake-strata, on top of Strata v0.2.16

This branch (`updated-strata`) is the upstream release `v0.2.16` of [Niko1221/Strata](https://github.com/Niko1221/Strata) plus a small
set of changes taken from the fork [pancake-strata](https://github.com/rocktboy1982/pancake-strata) (work line `port/rdna3-decode`).
It is written as the description of a pull request back to upstream. Every performance number below is copied from the fork's
test ledger (`TEST_LEDGER.md`, ids `T-nnn`) and names the hardware it was measured on. A number without a ledger id is not in this file.
Every ported switch is default off and its unset path is v0.2.16's code. The switches of sections 2.8 to 2.13 are the adopted core of the
fork's daily preset; for most of them the ledger holds the bundle's number (T-009, T-378) or a single run, not a confirmed per-switch gain,
and each section says which. The benchmark of section 4 and `flag_diag.sh` are how this branch settles them on the release build.

Hardware of all ledger numbers unless stated: AMD Radeon RX 7900 XT 20 GB (gfx1100), Linux, ROCm 10.0, model
Qwen3.8-Flash-Next (`GSQ-RCO-IQ2_XS` GGUF), the fork's `fast-hip` preset. The ledger and the raw result archives live in the
pancake-strata repository (`TEST_LEDGER.md`, `bench/results/`, `pancake-docs/research/bench/`); the archive path of every row is
given below where one exists.

## 1. Overview and capabilities

| capability | what it is | how to run it |
|---|---|---|
| RDNA decode GEMVs | multi-column GEMV kernels for Q4_K, Q5_K, Q6_K and IQ4_XS on gfx11 (`engine/src/kernels/cuda/rdna_mmvq.cu`), opt-in `STRATA_RDNA_Q6K=1` | `STRATA_RDNA_Q6K=1 strata ...` (see section 2.1) |
| Hyper-connection read, wave and fusion kernels | `STRATA_GR_RDNA*`, `STRATA_RDNA_MMVQ_WAVE=3`, `STRATA_COMBINE_CPU`, `STRATA_WAIT_FUSE`, `STRATA_ROUTE_PUBLISH` (gfx1100 decode) | env switches, default off (sections 2.8 to 2.11) |
| CPU pool and verify-window switches | `STRATA_POOL_FUSE_PHASES`, `STRATA_VERIFY_ECAP` + `STRATA_VERIFY_ECAP_MIN_ROWS` | env switches, default off (sections 2.12, 2.13) |
| Per-hardware flag diagnostics | `tools/pancake/flag_diag.sh`: detects the hardware and, flag by flag, answers "can it run here, and does it help here" | `tools/pancake/flag_diag.sh hw`, `list`, `run CONFIG OUT_DIR` (section 2.5) |
| Tuned flags at the normal start | `terminal/strata_control/tuned_flags.py`: measured flags for the detected hardware are added to the run configuration at every start; your own settings always win | automatic; off with `STRATA_TUNED_FLAGS=0` or `--no-tuned-flags` (section 2.4) |
| Host tuning scripts | `tools/pancake/host_tune.sh`, `host_tune_ab.sh`, `os_tune.sh`, `pcie_link.py`: reversible Linux host settings with a state file | `host_tune.sh --check` (read-only), see section 2.6 |
| Optimised build script | `tools/pancake/build_updated.sh`: Release build for gfx1100 with every flag justified from the code | `tools/pancake/build_updated.sh [SRC] [BUILD]` (section 2.7) |
| Benchmark against the plain release | `tools/pancake/box_chains/bench-updatedAA.sh` and specs 220 / 221 | on the Linux benchmark machine (section 4) |

Data files: `data/expert-profile-pancake.bin` (a learned expert profile, section 2.3). The hipBLASLt table for gfx1100 and
hipBLASLt 100401 is already in upstream (`terminal/tools/hip/gfx1100-hipblaslt-100401.txt`); this branch only selects it
(section 2.2).

## 2. The changes, one section each

### 2.1 RDNA multi-column decode GEMVs (`STRATA_RDNA_Q6K`)

What it does. During decode with speculative drafting the engine multiplies the model's K-quant weights by 4 to 16 activation
columns at once (the verify window). On gfx11 the generic `native_mmvq` kernels read the weights again for every column group.
The new kernels use one wave per output row, decode the weights once, and apply them to every column with `v_dot4` (sudot4),
reducing with DPP and `permlanex16`. Formats: Q4_K and Q5_K (taken from 4 columns up; below that the native kernels are
faster), Q6_K and IQ4_XS (all column counts, 1 to 16).

Enable: `STRATA_RDNA_Q6K=1`. Default off, the unset path is the code of v0.2.16 byte for byte (the dispatch only asks
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
  confirm run. `tools/pancake/flag_diag.sh` re-checks it on any machine in a few minutes.

Exactness. The products and per-16 scales are those of ggml's `vec_dot_*_q8_1`, summed in another order, so the kernels are
"bounded, not bitwise": `rdna_mmvq_parity` requires every output within 1e-4 of the column's largest reference magnitude and two
runs bitwise equal (the fork's 192 of 192 cases, T-081 context).

Checked on this branch: `native_mmvq.cu` and the stub compile with nvcc 12.8 (sm_75, CUDA path); `rdna_mmvq.cu` type-checks with nvcc
using stand-ins for the five gfx11 builtins. It has not been compiled with the ROCm compiler in this change; the benchmark
chain's first step builds it on the box.

### 2.2 hipBLASLt solution table for gfx1100 (selected, not added)

The fork's prompt-path table `gfx1100-hipblaslt-100401.txt` is the same file upstream already ships under `terminal/tools/hip/`
(upstream commit by miskahm, 2026-10-04). The fork measured it as: 11.7K-token prompt read 1,345 -> 1,850-1,920 tok/s (+38 %,
"provisional, noisy machine", T-096, no raw archive). This branch adds nothing here. The tuned-flags profile for gfx1100 set
`STRATA_HIPBLASLT_TUNING` to that file until 2026-11-20; the box diagnostics of this branch found it UNSUPPORTED (section 2.4a), so the
profile no longer sets it. The benchmark preset still uses it (`<REPO>/terminal/tools/hip/gfx1100-hipblaslt-100401.txt`). The engine
checks the table's version header against the installed hipBLASLt and ignores a mismatch.

### 2.3 Learned expert profile (data file)

`data/expert-profile-pancake.bin` (196,632 bytes) is the profile the fork learned for the Flash-Next IQ2_XS files with
`tools/pancake/learn_profile.py` (not part of this branch). Passed as `--expert-profile`, it was measured against the profile
shipped with the model: decode 52.8 / 63.4 / 75.8 / 90.0 -> 57.8 / 70.8 / 80.9 / 91.4 tok/s (+6.7 %, every run faster, T-194,
archive `bench/results/2026-11-08-pancake-profile-ab.jsonl`, `bench/prefetch/prof-learned.jsonl`). **That measurement is from an
RTX 3060 12 GB (the fork's first card), not from the RX 7900 XT; it has not been repeated on this build.** The profile belongs
to one model family, so it is not part of the automatic tuned flags; it is used by the benchmark preset only.

### 2.4 Tuned flags at the normal start

Where. The normal start of the Terminal, `runtime start` and the app all call `apply_saved_tuning` in
`terminal/serve/server_parts/http_api.py` before the saved-preset overlay. Two lines were added there; the logic is the new
module `terminal/strata_control/tuned_flags.py` (tests: `terminal/tests/tuning/test_tuned_flags.py`, 12 tests with fake
hardware and fake verdict files).

Rules:

1. The profile is chosen by graphics architecture, ROCm version prefix and VRAM range. The shipped profile
   (`terminal/data/tuned-flags.json`) is for gfx1100, ROCm 10.x, 16-24 GiB. A file written by `flag_diag.sh --write-recommended`
   (`~/.strata/tuned-flags.json` or `$STRATA_TUNED_FLAGS_FILE`) is read first.
2. An explicit user setting always wins: a variable in the configuration's `env` or in the process environment, an argument
   already in the configuration's `args`.
3. `STRATA_TUNED_FLAGS=0` or `--no-tuned-flags` gives plain upstream behaviour. `STRATA_NO_PRESET` (the tuner's own workers) also
   skips it.
4. Only measured gains enter a profile: exact (bitwise equal text) and at least 2 % end to end, or lossy with a passing quality
   gate and at least 3 %. The rule is one function (`accepted`) applied when the file is read, so a hand-edited file cannot add a
   flag with another verdict.
5. Unknown hardware (no profile matches, or no card could be read) keeps the plain defaults and prints one line saying so.

The shipped profile holds four flags since 2026-11-20: `STRATA_RDNA_Q6K=1` (section 2.1), `STRATA_GR_RDNA=1` and `STRATA_RDNA_MMVQ_WAVE=3`
(2.8, 2.9: the T-009 kernel bundle, 3.0 % is the bundle's bound 104 against 101 tok/s, not a per-flag figure) and `--pcie-frac 0` (+6 % short,
+13 % edit, one run each, T-197; 0.15 / 0.3 / 0.55 were all slower, T-268). It held six before; two were removed after the box
measurements of this branch (section 2.4a).
`--adapt-every` / `--adapt-swaps` are not in it: the ledger shows no gain without `--pcie-frac 0` (T-199). Also not in it, each for a
reason in the ledger: `STRATA_GR_RDNA_Q8` / `_A8` / `GR_INJ_UP` / `GR_FUSED` (lossy, no passing quality gate on record),
`STRATA_ROUTE_PUBLISH` (21.60 to 21.51 ms per window, 0.4 %), `STRATA_WAIT_FUSE` (+1.2 %), `STRATA_COMBINE_CPU` (no single number),
`STRATA_POOL_FUSE_PHASES` (+2.5 % +-1.7, short -1.3 %), `STRATA_VERIFY_ECAP` (its gates ran with a grammar guard v0.2.16 lacks).
None of the four has a valid single-flag verdict on a HIP build of this branch yet; regenerate the profile from the benchmark's verdicts
(`flag_diag.sh --write-recommended`) before relying on it; `STRATA_TUNED_FLAGS=0` removes it.

### 2.4a Flags removed from the shipped profile (2026-11-20) and the data files the settings use

Owner order 2026-11-19 23:05: remove from the shipped settings every flag that did not work on this branch. The rule: a FAIL, NO-GAIN
or UNSUPPORTED verdict from the box diagnostics of this branch, or a lossy flag whose quality gate did not pass, leaves the shipped
profile. Each removal is also recorded in `terminal/data/tuned-flags.json` (key `removed`, with the evidence).

| flag | verdict on this branch | evidence |
|---|---|---|
| `STRATA_HIPBLASLT_TUNING` (100401 table) | UNSUPPORTED | flag_diag rerun, plain upstream build (`.../diag-rerun/diag-plain/verdicts.json`): text differs |
| `--lookup-max-t 12` | lossy, gate did not pass | quality gate of this branch (`/home/dan/auto-runs/benchexact/gate-us`, spec 222, 48 requests, token forcing against plain upstream v0.2.16): `updated-exact` (keeps `--lookup-max-t 12`) REJECT on the rate rule, mismatch 3.379 % against 1.562 % / 1.360 % for two stock runs, +1.92 points, 90 % interval [+1.23, +2.74]; `updated` REJECT (3.964 %, +2.50 points); event rule PASS for both. AGENTS rule of the pancake line: never lossy without a passing gate |

Restored the same day (owner decision 2026-11-20): `STRATA_RDNA_Q6K=1`, `STRATA_GR_RDNA=1`, `STRATA_RDNA_MMVQ_WAVE=3`. The rerun
(`/home/dan/auto-runs/winbenchus/diag-rerun/diag-updated/verdicts.json`) called them UNSUPPORTED because their text differed from the
default, but on this model plain upstream against itself (`base-late`) also gave different text in every run that day, so text
identity cannot judge exactness here. Their parity ctests (`rdna_mmvq_parity`, `rdna_gr_parity`; both bounded by a tolerance, not
bitwise, by their own description) are the judge; they have not run on the box yet. `STRATA_HIPBLASLT_TUNING` was removed for the
same text-identity reason and stays out pending an owner decision. `tools/pancake/flag_diag.py` now starts the default a second time
when a claimed-exact flag's text differs: if the repeat differs too, the flag keeps its speed verdict with exactness
"EXACT-UNVERIFIED (base vs base also differs)"; `--ctest BUILD_DIR` runs the flag's parity ctest (FAIL is the verdict, PASS is
recorded; "Not Run" is no result). A text difference alone never gives UNSUPPORTED.

Not in the shipped profile before and so not removed from it: `STRATA_VERIFY_ECAP` (NO-GAIN -15.0 % on short, same updated-build run)
and `STRATA_PF_F16=1` (NO-GAIN -21.2 % prompt, plain build). The benchmark preset `configs/pancake/flashnext-iq2_xs-updated-hip.json`
and specs 220-223 are the measurement arms of the whole updated preset against plain upstream; they keep their flags (a single-flag
negative on short is not a measurement of the arm, and ECAP's case is the code edit), so the next box runs measure the same arm.

Data files. Every file a setting of this branch names is in this tree; nothing points into the pancake tree. The benchmark specs of
the pancake harness now read them from the updated-strata tree the box chain builds (`/mnt/pancake/benchus/src-us/...`), not from
the harness checkout. No file had to be copied in: both were already here, byte-identical to the pancake copies (same git blob).

| file | size | origin | used by |
|---|---|---|---|
| `data/expert-profile-pancake.bin` | 196,632 bytes | ours: learned by the pancake line for the Flash-Next IQ2_XS files (T-194), committed here in `10480acdb` | `--expert-profile` of the updated preset and specs 220-223 |
| `terminal/tools/hip/gfx1100-hipblaslt-100401.txt` | 1,401 bytes | upstream Strata v0.2.16 (commit by miskahm, 2026-10-04) | `STRATA_HIPBLASLT_TUNING` of the updated preset and specs 220-223 (the preset said `<REPO>/tools/hip/...`, a folder this tree does not have; now `<REPO>/terminal/tools/hip/...`) |

Model weights, packs and the MTP head stay in the data folder (`<STRATA_DATA>`), not in the repository.

### 2.5 Per-flag diagnostics: `tools/pancake/flag_diag.sh`

Read-only on the system (sysfs, proc and rocminfo reads; the engine is started as the user; files are written only under the
output folder; no sudo, no sysctl writes).

* `flag_diag.sh hw`: gfx architecture, ROCm version, VRAM, PCIe link speed and width, CPU, RAM, hipBLASLt library, VMM hint.
* `flag_diag.sh list`: the candidate flags (sections 2.1 to 2.13 plus `STRATA_PF_F16`, `STRATA_HIP_WMMA`, `STRATA_SELECT_WMMA`,
  `STRATA_PROMPT_ATTN_IMMA`, `STRATA_KV_HOST_DMA`, `STRATA_PREFILL_RING`, `STRATA_STAGER_RING`, `GPU_MAX_HW_QUEUES`) with exact / lossy
  status and ledger id; a candidate that only acts together with others (`GR_FUSED`, `COMBINE_CPU` + `WAIT_FUSE`, `VERIFY_ECAP` +
  `MIN_ROWS`) carries them in its arm and in the recommended profile.
* `flag_diag.sh run CONFIG.json OUT_DIR [--only A,B] [--budget-min 15]`: per flag, first a hardware check without starting
  anything (UNSUPPORTED with the reason), then one engine start with the flag next to two starts of the default (the second is the
  noise floor), two short prompts each, through the A/B harness `tools/pancake/abtest.py` of the pancake-strata line
  (`FLAG_DIAG_ABTEST=path` points at it). The verdict is UNSUPPORTED (cannot run, failed to start, or claimed exact and the text
  differs), NO-GAIN, GAIN (exact >= 2 %, lossy >= 3 %, and above the noise floor) or UNTESTED (budget ended). A lossy flag enters the
  recommended file only with a passing quality gate.
* `flag_diag.sh --write-recommended OUT_DIR/verdicts.json [FILE] [--merge]` writes the tuned-flags profile from the verdicts.
* Time: about 2.5 minutes per flag, 15 minutes at the default budget (four flags); the benchmark chain runs it on both builds (10 minutes
  for the two flags the plain release knows, 20 minutes for six of the updated build's).

Tests: `python -m pytest -q tools/pancake/test_flag_diag.py` (16 tests: fake sysfs / proc tree, fake rocminfo, fake A/B output,
the verdict rules, the recommended-file rule). Reproduce the real run with `flag_diag.sh run <config> <out>` on the target machine.

### 2.6 Host tuning scripts

`tools/pancake/host_tune.sh` applies one reversible Linux host step at a time (NVMe read-ahead, transparent huge pages,
swappiness, memlock, CPU governor, amd_pstate, GPU DPM, ASPM, C-states), saves each previous value in a state file, checks every
write by reading it back, and refuses unknown names. `--check` is read-only and needs no sudo. `host_tune_ab.sh` measures one step
against a baseline (it needs the harness `abtest.py`). `os_tune.sh` keeps free RAM in large blocks for the expert arena.
`pcie_link.py` reads the PCIe link of the card. Tests: `tools/pancake/test_host_tune.py`, `test_pcie_link.py` (24 tests, fake
trees). No gain is claimed for these scripts; the fork's host-tuning A/B (spec 125) is a measurement tool, not a result.

### 2.7 Optimised build: `tools/pancake/build_updated.sh`

`tools/pancake/build_updated.sh [SRC] [BUILD]` (environment: `BACKEND=hip|cuda`, `ROCM`, `GFX`, `LTO=1`, `TESTS=1`). The
header of the script lists each flag with the code that justifies it. In short: Release with `-O3 -DNDEBUG`; HIP on, CUDA off;
`gfx1100` only; `STRATA_AMD_COMMUNITY_TUNING=ON`; `STRATA_WINDOW_MAX_T=16` (the 16-row verify window the 4..16 column kernels
serve); `STRATA_NATIVE_EXPERTS=ON`. Deliberately not added: a global `-march=native` (the CPU quant layer keeps AVX-only files with
AVX2 off), `-ffast-math` / `--use_fast_math` on HIP (the tree sets `-ffp-contract=off` on files whose rounding ties matter), and
LTO (opt-in `LTO=1`, no claim that it is faster). `BACKEND=cuda` keeps the CUDA build available. The effect of these flags is
not measured in isolation; the benchmark (section 4) measures the whole updated build against the plain release.

### 2.8 Hyper-connection read kernels (`STRATA_GR_RDNA`, `_Q8`, `_A8`, `STRATA_GR_INJ_UP`, `STRATA_GR_FUSED`)

What it does. The model's two hyper-connection reads per layer (a down and an up projection of a 4 x 2560 state) run in one decode
window of up to 8 tokens. `rdna_gr.cu` reads them on wave32 gfx11 with a split-K down pass and a partial-folding up pass in another
summation order than `fused_gr_read_multi`. Four optional layers on top: `GR_RDNA_Q8` keeps int8 + fp16-scale copies of the two
projections (made at load in `NativeDense` from the GGUF's Q8_0 or BF16 rows) and leaves the BF16 copies off the card;
`GR_RDNA_A8` uses int8 activations and dot4; `GR_INJ_UP` gives the four inject rows their own blocks; `GR_FUSED` runs both passes in one
launch with a grid barrier. The geometry is the fork's Flash-Next one (n_embd 2560, hc 4, hc_lr 320) as in `rdna_gr.cu`.

Enable: `STRATA_GR_RDNA=1` (BF16 read); add `STRATA_GR_RDNA_Q8=1` for the int8 weights, then `STRATA_GR_RDNA_A8=1`,
`STRATA_GR_INJ_UP=1`, `STRATA_GR_FUSED=1` in that order (each needs the one before). Default off; the unset path is v0.2.16's.

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

### 2.9 `STRATA_RDNA_MMVQ_WAVE=3`

One wave per output row that emulates the generic four-warp layout in registers: the same partial sums in the same order, no LDS exchange and
no block barrier, one kernel for 1 to 16 columns (so a column's result does not depend on how many verify tokens ride along). Upstream's
wave kernel exists for gfx906 only; this is a new gfx11 branch of the same macro in `native_mmvq.cu`. Short projections take it only for
`n_in <= 1024` or `(n_in <= 2560 and n_out >= 1024)`, as in the fork. Measured: mode 1 (all shapes, another summation order) was 4.5 % slower
than mode 3 on short and 2.5 % on edit (T-072, spec 40, screen n = 2); the bundle number is T-009. Enable `STRATA_RDNA_MMVQ_WAVE=3`.

### 2.10 `STRATA_COMBINE_CPU` and `STRATA_WAIT_FUSE`

With CPU experts, the verify window copies the CPU rows from the mapped buffer, adds the GPU hit rows, then combines. `COMBINE_CPU` reads the
CPU rows straight from the mapped buffer and the GPU's hit rows in one combine kernel and carries the shared expert's gate as an extra block of
the router GEMV launch: bitwise the launches it replaces. `WAIT_FUSE` (needs `COMBINE_CPU` for its second half) puts the plan wait and the plan
copy in one launch and the CPU-share wait inside the combine, with a system-scope acquire in every wave. Files: `native_moe.cu` / `.hpp`,
`native_bf16.cu` (`bf16_f32_mmvf_multi_kernel` gets the aux block), `verify_kernels.cu` / `.hpp`, `verify.cpp`.
Measured by the fork: `WAIT_FUSE` 21.85 to 21.60 ms per window, byte-identical output (the commit's own measurement on an IQ3 model, nicejunjie);
turning the fused wait off lost (T-077); the bundle is T-009. v0.2.16 already has a different plan wait + copy (`wait_copy_i32_from_mapped`), which
stays the default.

### 2.11 `STRATA_ROUTE_PUBLISH`

The router's top-10 and the doorbell publish in one launch for 1 to 8 rows (`native_router_top10_publish_res`). Measured by the fork: 21.60 to
21.51 ms per window, identical output (0.4 %: below the 2 % bar on its own; T-009). The fork found it unsafe together with expert deferral
(T-317), which v0.2.16 does not have. Not taken with `STRATA_ROUTE_PRIOR`, a resident-route margin, a device plan, `--remote-expert-opt` or
`STRATA_DOORBELL_STORE`.

### 2.12 `STRATA_POOL_FUSE_PHASES`

The CPU expert pool runs a layer's gate/up and down phases as one phase: down tasks wait only for their own expert, whose last gate/up task
quantized the down input. Same row ranges and kernels, output identical. Measured (T-224, spec 03b, confirm 3 reps, RX 7900 XT): edit +2.5 % +-1.7,
short -1.3 % (not significant); the fork adopted it; by its own interval it is under the 2 % bar. Not with the NUMA split. File: `pool.cpp` / `.hpp`.

### 2.13 `STRATA_VERIFY_ECAP` and `STRATA_VERIFY_ECAP_MIN_ROWS` (lossy)

In a decode verify window each draft row (rows 1 to T-1, never row 0) reads at most N CPU-missed experts that the committed row and the earlier rows
do not read anyway, kept by routing rank; the others contribute zero (host side only, no graph change). `STRATA_VERIFY_ECAP=0` with
`STRATA_VERIFY_ECAP_MIN_ROWS=9` is the fork's daily setting: the cap acts only in windows of 9 or more rows (the long lookup windows).
Measured (RX 7900 XT, spec 60, 6 clean runs per arm and case, T-588): decode edit 165.3 +-6.2 tok/s = +41.7 % +-5.6 over no ECAP, short +2.8 % +-2.6;
token forcing PASS (mismatch 1.583 % against 1.467 / 1.722 % for the stock arms, -0.011 points, T-589); tool-call task gate PASS (99.4 % success,
tool-call validity 100 %, T-591). **The two gates ran with the fork's grammar guard, which exempted a tool call or JSON reply in progress from the cap;
v0.2.16 has no such guard, so on this branch the cap also acts inside a tool call. That is not covered by T-591**; run `quality_gate.py` and the
task gate on the release build before enabling it by default. Files: `verify_ecap.hpp` (the rule) and `verify_ecap_test.cpp` (built and run on
the PC: ok), `expert_source.cpp` / `.hpp`, `verify.cpp` / `.hpp`, `generate.cpp`. The fork's ECAP v2 (per-row lists, probability gate, window
policy) is not ported.

## 3. Diagnostic tests and results

| test | command | result in this change |
|---|---|---|
| tuned flags | `python -m pytest -q terminal/tests/tuning/test_tuned_flags.py` | 11 passed (PC, fake hardware) |
| flag diagnostics | `python -m pytest -q tools/pancake/test_flag_diag.py` | 16 passed |
| ECAP drop rule | `verify_ecap_test` (built and run with cl on the PC) | ok (4000 random windows, 50,278 entries dropped) |
| host compile checks of the pass-2 files | `cl /Zs` of `verify.cpp`, `expert_source.cpp`, `pool.cpp`, `native_dense.cpp`; nvcc sm_75 of `fused_gr.cu`, `native_moe.cu`, `native_bf16.cu`, `native_router.cu`, `verify_kernels.cu` | no errors; the compile checks found two real mistakes of the port (`peer_rows_`, a launch argument order), fixed |
| host tuning | `python -m pytest -q tools/pancake/test_host_tune.py tools/pancake/test_pcie_link.py` | 24 passed |
| settings register | `python tools/test_settings_register.py` | 8 passed |
| CUDA path compile | `nvcc -arch=sm_75 ... native_mmvq.cu`, `rdna_stub.cpp` | compiled (CUDA 12.8) |
| RDNA kernel parity | `rdna_mmvq_parity`, `rdna_gr_parity` (five variants) (build with `-DSTRATA_BUILD_TESTS=ON`, HIP) | TO BE MEASURED on the box (not run: no gfx1100 card in the PC that prepared the change) |
| per-flag diagnostics on gfx1100 | `flag_diag.sh run ...` (step 3 of the benchmark chain) | TO BE MEASURED |

`terminal/tests/tuning/test_preset_persistence.py` has 3 failures on the Windows PC used for this change, with and without these
changes (checked by stashing the edit); it was not investigated further.

## 4. Benchmark: plain upstream release against updated-strata

Plan (not yet run): `tools/pancake/box_chains/bench-updatedAA.sh`. It builds (a) the plain `v0.2.16` release with its own
default flags (the documented HIP command of `docs/AMD_HIP.md`, nothing else) and (b) this branch with `build_updated.sh`, both on
`/mnt/pancake`; runs `flag_diag.sh` on both builds; then the same workloads on both through `abtest.py` (screen first, then `--tier confirm --block 3
--warmup 1 --outlier robust --reclaim-max 3 --reps 3 --no-early-stop` for an arm that earned it), serialized, with no other job on the machine:

* spec `220`: `short` and `edit` (code rewrite) decode tok/s; text identity against the plain release is printed per case;
* spec `221`: cold prompts of 10K, 32K, 64K and 100K tokens followed by a 256-token reply: prompt tok/s and decode tok/s of every case.

The plain arm gets the model arguments only (`--pack`, `--native`, `--ple-gguf`, `--mtp`, `--max-context 131072`); the updated arm
gets the same plus the adopted preset (`configs/pancake/flashnext-iq2_xs-updated-hip.json`: the fork's daily fast-hip preset without
`STRATA_AGENT_COUNTERS` (log only), `STRATA_PREFILL_EQUAL_TAIL` (a trial under the 2 % bar), `--grammar` and `--mtp-draft-vocab`). Estimated box time about 2 hours (short plan: screen first, confirm 3 reps only for an arm with a screen benefit, tune-up included, budget 7200 s; `bench-sched.sh` runs it before the rest of the box queue).

Results (RX 7900 XT 20 GB, Qwen3.8-Flash-Next IQ2_XS): **TO BE MEASURED**

| case | plain upstream v0.2.16 | updated-strata | difference | text identical |
|---|---|---|---|---|
| short, decode tok/s | TO BE MEASURED | TO BE MEASURED | TO BE MEASURED | TO BE MEASURED |
| edit, decode tok/s | TO BE MEASURED | TO BE MEASURED | TO BE MEASURED | TO BE MEASURED |
| 10K, prompt / decode tok/s | TO BE MEASURED | TO BE MEASURED | TO BE MEASURED | not compared |
| 32K, prompt / decode tok/s | TO BE MEASURED | TO BE MEASURED | TO BE MEASURED | not compared |
| 64K, prompt / decode tok/s | TO BE MEASURED | TO BE MEASURED | TO BE MEASURED | not compared |
| 100K, prompt / decode tok/s | TO BE MEASURED | TO BE MEASURED | TO BE MEASURED | not compared |

After the run the recommended-flags profile is regenerated from the measured verdicts
(`flag_diag.sh --write-recommended`), and the shipped profile is replaced by it where the verdicts differ.

## 5. Credits and authorship

Licence of this work: MIT, like upstream (`LICENSE`: Copyright (c) 2026 Niko1221 and the Strata contributors). The fork keeps that
licence and credits (pancake-strata `README.md` section 9).

| item | source | licence | notes |
|---|---|---|---|
| RDNA multi-column GEMVs (2.1) | **nicejunjie** (Junjie Li), fork Strata-rdna3: commits `863804599` (2026-10-11, "multi-column Q6_K and IQ4_XS decode GEMV") and `ceb14be4e` (2026-10-11, Q4_K / Q5_K from 4 columns), as they appear in the pancake-strata history | MIT as a fork of MIT Strata; the fork's own licence file was not re-checked for this change: confirm before merging | The kernel design is theirs. pancake-strata (rocktboy1982), original work on top: HIP-only guards and CUDA stubs (`413e1b968`, 2026-11-08), the parity tests (`76f114fa4`, 2026-11-09), the stub file (`20733fb77`, 2026-11-09), the 9 to 16 column extension (`82687a7bd`, 2026-11-15); removal of the unproven variants and the port to upstream's layout in this branch |
| hyper-connection read kernels (2.8) | **nicejunjie** (Junjie Li), Strata-rdna3: `89da64b93` (2026-10-10, read + DPP shuffles), `042a9ef7b` (2026-10-11, int8 read), `9044c7724` (2026-10-11, A8), `e56ac57e5` (2026-10-11, INJ_UP), `06d959d7a` (2026-10-11, FUSED), as they appear in the pancake-strata history | MIT as a fork of MIT Strata (the fork's licence file not re-checked: confirm before merging) | kernel design is theirs. pancake-strata (rocktboy1982): the q8_1 image for QFUSE (`6d9a1a43b`, 2026-11-08), HIP-only guards and stubs (`413e1b968`, `20733fb77`), the parity test (`76f114fa4`), and the port to upstream's layout in this branch (loader, `hc_bf16_view`, generate.cpp skip) |
| `STRATA_RDNA_MMVQ_WAVE=3` (2.9) | **nicejunjie**, Strata-rdna3 (the wave32 exact kernel), as in the pancake-strata history | MIT as above | mode selection and the shape rule are the fork's; port by hand into upstream's macro |
| `STRATA_COMBINE_CPU` (2.10) | **nicejunjie**, Strata-rdna3, ported into pancake-strata as `2745ea818` (rocktboy1982, 2026-11-08) | MIT as above | `STRATA_WAIT_FUSE`: nicejunjie `297243a5e` (2026-10-11), cherry-picked in the fork |
| `STRATA_ROUTE_PUBLISH` (2.11) | **nicejunjie**, Strata-rdna3: `50ec2ade3` (2026-10-11) | MIT as above | re-applied by hand |
| `STRATA_POOL_FUSE_PHASES` (2.12) | pancake-strata (rocktboy1982), original work | MIT | first commit `35c512984` (2026-11-09, author rocktboy1982), ledger T-224 |
| `STRATA_VERIFY_ECAP` (2.13) | idea and rule adapted from **jmnargi** (Strata-V100 `24894e676b`, `7c5041c358`, `376355453f`; upstream PR 3545, MIT as read in the fork's README section 9); the per-draft-row, CPU-missed-only form is pancake-strata's | MIT | `2d44d31ec` (2026-11-10, rocktboy1982), the MIN_ROWS gate `9a4cd7568` (2026-11-15, rocktboy1982); ledger T-588, T-589, T-591 |
| hipBLASLt table (2.2) | **miskahm**, upstream commit `7984688e3` (2026-10-04) | MIT (upstream) | already in upstream; measured by the fork (T-096) |
| learned expert profile (2.3) | pancake-strata (rocktboy1982), original work | MIT | first commit `3638b8cee` (2026-11-08); learner `a4f305b82` (2026-11-08); ledger T-194 |
| tuned flags (2.4), flag_diag (2.5), build script (2.7), benchmark chain and specs (4) | pancake-strata (rocktboy1982) line and this branch, original work | MIT | the benchmark chain follows `redo-chainAA.sh` of the fork; commits of this branch below |
| host tuning scripts (2.6) | pancake-strata (rocktboy1982), original work; the list of host settings was read from **bramvanoploo/strata-performance-setup** (MIT, https://github.com/bramvanoploo/strata-performance-setup), no code copied | MIT | `host_tune.sh` and `host_tune_ab.sh` first commit `cf6726c20` (2026-11-16, author rocktboy1982); `os_tune.sh` first commit `9ff4bdc4c` (2026-11-10, author rocktboy1982) |
| adopted values `--pcie-frac 0`, `--lookup-max-t 12` | pancake-strata (rocktboy1982): measured settings of existing upstream options | MIT | T-197, T-268, T-126, T-127 |

Proof of authorship of the "original work" rows: the commits above are in the public history of
https://github.com/rocktboy1982/pancake-strata (branch `port/rdna3-decode`), authored by `rocktboy1982` on the dates given, and every
measurement has a ledger row in `TEST_LEDGER.md` of that repository. `git log --format='%h %an %ad' -- <file>` on that repository
reproduces each hash and date. The commits of this branch carry the trailer `Co-Authored-By: Claude Sonnet 5.5`.

Ideas read but not used as code: the fork's `LEADS.md` and `README.md` section 9 list the other projects it studied. In
particular **jdominick05/caminus** is licensed AGPL-3.0-or-later; the fork took ideas only, copied nothing, and nothing from it is in
this branch.

## 6. Compatibility, risks and what is not included

* Defaults. Nothing changes unless a flag is set or the tuned-flags profile matches a gfx1100 card with ROCm 10.x and 16-24 GiB of
  VRAM; `STRATA_TUNED_FLAGS=0` or `--no-tuned-flags` removes even that. The engine's unset path is unchanged by the kernel
  change.
* Unsupported hardware. The RDNA kernels exist only in a wave32 HIP build (gfx10 / gfx11 / gfx12 family); CUDA and gfx906
  builds take the stub. The kernels are tuned for and tested on gfx1100 only.
* Exactness. `STRATA_RDNA_Q6K` changes the summation order (bounded error, section 2.1); `--lookup-max-t 12` changes the text
  (quality gate PASS, T-127). The tuned-flags rule accepts a lossy flag only with a passing gate.
* Evidence strength. `--pcie-frac 0` rests on one run per case (T-197) plus a confirming rejection of the other values (T-268);
  the hipBLASLt gain is marked provisional in the ledger (T-096); the kernel's end-to-end gain is inside the noise (T-081). The
  benchmark in section 4 is meant to settle these on the release build.
* Not ported (so a reader does not look for them): the fork's ECAP v2 (per-row lists, probability gate, window policy), `STRATA_GR_UPPRE`, the
  debug and bench selectors of `rdna_gr.cu`, `STRATA_AGENT_COUNTERS`, `STRATA_PREFILL_EQUAL_TAIL` and the fork's grammar guard. The fork's
  per-request knob trial (`STRATA_KNOB_TRIAL`) needs engine-side per-request keys upstream's engine does not have and has no measured gain.
* Where v0.2.16 differs from the fork and the port had to change: no `peer_rows_`, no expert deferral, no grammar guard (so ECAP is not exempted
  inside a tool call), a different `bf16_f32_mmvf_multi_kernel` signature (the aux launch passes `mmvf_fast()`), and a wave-kernel macro that was
  gfx906-only.
* Build. **None of the HIP code of this branch has been compiled by the ROCm compiler yet** (`rdna_mmvq.cu`, `rdna_gr.cu`, the gfx11 wave
  kernel). It was type-checked with nvcc through stand-ins for the gfx11 builtins, and the host code was compiled with MSVC and nvcc; the
  `rdna_gr_down_a8` kernel needs 52,736 bytes of shared memory, which fits HIP's 64 KiB and not CUDA's 48 KiB (nvcc's ptxas says so, only in
  that check). The benchmark chain builds it first, logs the compiler's error lines and stops if the build fails.
* Licence compatibility. All sources are MIT (see section 5); no AGPL code is included.

## 7. Commits of this branch

The first commit of the branch is a snapshot of the tree of the upstream tag `v0.2.16` (upstream commit
`64524fb9fc82222ddce554c0643b64f056ec8366`), parented on this fork's `main`, with `.github/workflows` left as in `main` (the account
used to push has no `workflow` scope, so GitHub refuses a push that changes workflow files). The tree outside `.github/workflows` is
identical to the tag. The commits after it are the changes described above; each commit message names the ledger ids it relies on.
A pull request to upstream should be made from the same commits rebased onto the tag `v0.2.16` itself.

## 8. Status page (`/status`, `/pancake`)

Open `http://127.0.0.1:<port>/status` in a browser, or `/pancake`. The chat page's header and the API monitor link to it. `GET /status.json`
returns the same data as JSON. `/status` without `Accept: text/html` (or `?html=1`) is still the old API answer. The page is read-only: GET
only, no file from outside the tree, no new hardware probe. The server draws the page, so it reads with JavaScript off; with JavaScript it
updates every 2 s. With an API key set, the values need the key (the page asks for it).

What it shows:

* Project: branch and head commit of the checkout (read from `.git`), engine version and build id (the engine's `BUILD.json`), backend,
  model, context, run config, uptime.
* Hardware: gfx architecture, ROCm version, VRAM, PCIe link, CPU, RAM. It uses `tuned_flags.detect_hardware` (rocminfo, sysfs, `/proc`; no
  privileges), read once in the background.
* Tuned flags: the profile of `terminal/strata_control/tuned_flags.py` that matches this machine (`$STRATA_TUNED_FLAGS_FILE` or
  `~/.strata/tuned-flags.json`, then `terminal/data/tuned-flags.json`). Each flag shows its state at start: applied, yours (run config or
  environment), changed by a saved tuning, or not applied. `STRATA_TUNED_FLAGS=0` or `--no-tuned-flags` shows "off".
* Your settings: the run config's env and args, and the `STRATA_` / `HIP_` / `HSA_` variables of the environment.
* Flag diagnostics: the verdicts of `tools/pancake/flag_diag.sh run`, when `$STRATA_FLAG_DIAG_VERDICTS` (or
  `~/.strata/flag-diag/verdicts.json`) names a verdicts file.
* Speed and caches: decode and prompt tok/s (now, run average, last request), prefix reuse, expert cache hit rate (last request, mean),
  PCIe share, drafts accepted, the conversation cache from the engine's log, every INFO fact the engine reported (expert slots,
  windows, ...), and the live GPU and RAM readings.

A value that is not known prints as an em dash. Code: `terminal/serve/status_page.py`, `terminal/serve/web/status.html`,
`terminal/serve/web/status.js`. Tests: `terminal/serve/test_status_page.py` (mock engine, no GPU). Credits: ours (pancake-strata
`214c8cd31`, `19ffe37ab`, extended for this branch). The pancake line's two other web changes were already in this tree, so they were not
ported again: `/?q=` fills in the question without sending it (`faf3db296`), and the speed cards show the run average when idle
(`6acd50efe`).
