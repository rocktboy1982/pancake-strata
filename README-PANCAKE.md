# updated-strata: measured enhancements from pancake-strata, on top of Strata v0.2.16

This branch (`updated-strata`) is the upstream release `v0.2.16` of [Niko1221/Strata](https://github.com/Niko1221/Strata) plus a small
set of changes taken from the fork [pancake-strata](https://github.com/rocktboy1982/pancake-strata) (work line `port/rdna3-decode`).
It is written as the description of a pull request back to upstream. Every performance number below is copied from the fork's
test ledger (`TEST_LEDGER.md`, ids `T-nnn`) and names the hardware it was measured on. A change without a measured gain is not in
this branch, and a number without a ledger id is not in this file.

Hardware of all ledger numbers unless stated: AMD Radeon RX 7900 XT 20 GB (gfx1100), Linux, ROCm 10.0, model
Qwen3.8-Flash-Next (`GSQ-RCO-IQ2_XS` GGUF), the fork's `fast-hip` preset. The ledger and the raw result archives live in the
pancake-strata repository (`TEST_LEDGER.md`, `bench/results/`, `pancake-docs/research/bench/`); the archive path of every row is
given below where one exists.

## 1. Overview and capabilities

| capability | what it is | how to run it |
|---|---|---|
| RDNA decode GEMVs | multi-column GEMV kernels for Q4_K, Q5_K, Q6_K and IQ4_XS on gfx11 (`engine/src/kernels/cuda/rdna_mmvq.cu`), opt-in `STRATA_RDNA_Q6K=1` | `STRATA_RDNA_Q6K=1 strata ...` (see section 2.1) |
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
"provisional, noisy machine", T-096, no raw archive). This branch adds nothing here except that the tuned-flags profile for
gfx1100 sets `STRATA_HIPBLASLT_TUNING` to that file. The engine checks the table's version header against the installed
hipBLASLt and ignores a mismatch.

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
module `terminal/strata_control/tuned_flags.py` (tests: `terminal/tests/tuning/test_tuned_flags.py`, 15 tests with fake
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

The shipped profile holds four flags: `STRATA_RDNA_Q6K=1` (section 2.1), `STRATA_HIPBLASLT_TUNING` (2.2), `--pcie-frac 0`
(+6 % short, +13 % edit, one run each, T-197; 0.15 / 0.3 / 0.55 were all slower, T-268) and `--lookup-max-t 12` (edit +8.8 % +-6.8,
text differs, token-forcing quality gate PASS, T-126 / T-127, archive `bench/results/2026-11-10-test-phase/05c-lookup12*`).
`--adapt-every` / `--adapt-swaps` are not in it: the ledger shows no gain without `--pcie-frac 0` (T-199).

### 2.5 Per-flag diagnostics: `tools/pancake/flag_diag.sh`

Read-only on the system (sysfs, proc and rocminfo reads; the engine is started as the user; files are written only under the
output folder; no sudo, no sysctl writes).

* `flag_diag.sh hw`: gfx architecture, ROCm version, VRAM, PCIe link speed and width, CPU, RAM, hipBLASLt library, VMM hint.
* `flag_diag.sh list`: the candidate flags (the ones in section 2.1-2.4 plus `STRATA_PF_F16`, `STRATA_HIP_WMMA`,
  `STRATA_SELECT_WMMA`, `STRATA_PROMPT_ATTN_IMMA`, `STRATA_KV_HOST_DMA`, `STRATA_PREFILL_RING`, `STRATA_STAGER_RING`,
  `GPU_MAX_HW_QUEUES`) with exact / lossy status and ledger id.
* `flag_diag.sh run CONFIG.json OUT_DIR [--only A,B] [--budget-min 15]`: per flag, first a hardware check without starting
  anything (UNSUPPORTED with the reason), then one engine start with the flag next to two starts of the default (the second is the
  noise floor), two short prompts each, through the A/B harness `tools/pancake/abtest.py` of the pancake-strata line
  (`FLAG_DIAG_ABTEST=path` points at it). The verdict is UNSUPPORTED (cannot run, failed to start, or claimed exact and the text
  differs), NO-GAIN, GAIN (exact >= 2 %, lossy >= 3 %, and above the noise floor) or UNTESTED (budget ended). A lossy flag enters the
  recommended file only with a passing quality gate.
* `flag_diag.sh --write-recommended OUT_DIR/verdicts.json [FILE] [--merge]` writes the tuned-flags profile from the verdicts.
* Time: about 2.5 minutes per flag, 15 minutes at the default budget (four flags); the benchmark chain runs it on both builds.

Tests: `python -m pytest -q tools/pancake/test_flag_diag.py` (13 tests: fake sysfs / proc tree, fake rocminfo, fake A/B output,
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

## 3. Diagnostic tests and results

| test | command | result in this change |
|---|---|---|
| tuned flags | `python -m pytest -q terminal/tests/tuning/test_tuned_flags.py` | 15 passed (PC, fake hardware) |
| flag diagnostics | `python -m pytest -q tools/pancake/test_flag_diag.py` | 13 passed |
| host tuning | `python -m pytest -q tools/pancake/test_host_tune.py tools/pancake/test_pcie_link.py` | 24 passed |
| settings register | `python tools/test_settings_register.py` | 8 passed |
| CUDA path compile | `nvcc -arch=sm_75 ... native_mmvq.cu`, `rdna_stub.cpp` | compiled (CUDA 12.8) |
| RDNA kernel parity | `rdna_mmvq_parity` (build with `-DSTRATA_BUILD_TESTS=ON`, HIP) | TO BE MEASURED on the box (not run: no gfx1100 card in the PC that prepared the change) |
| per-flag diagnostics on gfx1100 | `flag_diag.sh run ...` (step 3 of the benchmark chain) | TO BE MEASURED |

`terminal/tests/tuning/test_preset_persistence.py` has 3 failures on the Windows PC used for this change, with and without these
changes (checked by stashing the edit); it was not investigated further.

## 4. Benchmark: plain upstream release against updated-strata

Plan (not yet run): `tools/pancake/box_chains/bench-updatedAA.sh`. It builds (a) the plain `v0.2.16` release with its own
default flags (the documented HIP command of `docs/AMD_HIP.md`, nothing else) and (b) this branch with `build_updated.sh`, both on
`/mnt/pancake`; runs `flag_diag.sh` on both builds; then the same workloads on both through `abtest.py --tier confirm --block 3
--warmup 1 --outlier robust --reclaim-max 3 --reps 5 --no-early-stop`, serialized, with no other job on the machine:

* spec `220`: `short` and `edit` (code rewrite) decode tok/s; text identity against the plain release is printed per case;
* spec `221`: cold prompts of 10K, 32K, 64K and 100K tokens followed by a 256-token reply: prompt tok/s and decode tok/s of every case.

The plain arm gets the model arguments only (`--pack`, `--native`, `--ple-gguf`, `--mtp`, `--max-context 131072`); the updated arm
gets the same plus the adopted preset (`configs/pancake/flashnext-iq2_xs-updated-hip.json`: the preset's flags that exist in this
branch). Estimated box time 5.5-7.5 hours.

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
* Not ported in this change (so a reader does not look for them): the fork's hyper-connection read kernels
  (`STRATA_GR_RDNA*`, `rdna_gr.cu`), the verify-window expert cap (`STRATA_VERIFY_ECAP`, T-588 / T-589 / T-591), the fused
  combine and wait switches (`STRATA_COMBINE_CPU`, `STRATA_WAIT_FUSE`, `STRATA_ROUTE_PUBLISH`, T-009) and
  `STRATA_POOL_FUSE_PHASES` (T-224). They touch `generate.cpp`, `verify.cpp`, `fused_gr.cu` and the weight loader, which differ
  strongly from upstream's layout, and they cannot be compiled on the PC used for this port (no ROCm). They are candidates for a
  follow-up with a HIP toolchain. The fork's per-request knob trial (`STRATA_KNOB_TRIAL`) needs engine-side per-request keys that
  upstream's engine does not have, and has no measured gain, so it is not ported either.
* Build. The HIP code of this branch has not been compiled by the ROCm compiler yet; the benchmark chain builds it first and stops
  with an error line if the build fails.
* Licence compatibility. All sources are MIT (see section 5); no AGPL code is included.

## 7. Commits of this branch

See `git log v0.2.16..updated-strata`; each commit message names the ledger ids it relies on.
