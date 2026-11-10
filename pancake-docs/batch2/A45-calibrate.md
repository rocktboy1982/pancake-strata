# A45 - startup probe and calibration of the CPU-vs-promote ratio

Branch: `work/a45-calibrate` (from `origin/dev/batch-2`). Status: implemented, opt-in, host-only unit test only; the engine was not built here.

## What

The static decode miss split (`PcieAuto`, `pcie_share_rule.hpp`) gives the link a share `link / (link + k R)` of a layer's missed experts, with `R` the
CPU pool's rate. At start `R` is *assumed* (4 GB/s of expert bytes per worker, capped at 65), `k` is an empirical 1.85 (falling to 0.8 for a weak
pool), and on HIP outside gfx12 the refinement is off, so a non-gfx12 native pack gets the fixed 0.55 start value. On this box the daily engine
runs `--pcie-frac 0` because every static share lost (the row quotes 6-13%).

`STRATA_CALIBRATE=1` replaces the assumption by a measurement, made once per start (or once per machine with the cache file), after the CPU pool is
up and before the GPU expert cache is filled (nothing else is using VRAM then). It times, per expert:

| figure | how |
|---|---|
| `cpu_us` | the decode's own call, `run_split_multi_native` (2 distinct missed experts x 4 tokens per layer, all layers, 150 us host spin between layers like the GPU flags); same set-up as `--pool-affinity measure` |
| `cpu_c_us` | the same while host->device copies keep the link busy (the DDR reads compete: the pool's real cost, which `k` stood for) |
| `copy_us` | 32 blob-sized copies from pinned memory, one synchronize, host-timed |
| `copy_c_us` | link bytes done (chunks of 8 copies, event-polled) over the wall time of the contended CPU pass |
| `gpu_us` | device-to-device copy of the same blobs (a VRAM read-bound stand-in for a hit's expert kernel; also writes, so an upper bound) |

Each is the best of two passes (a disturbance only slows). Total probe time is under a second. All timing is host-side (no event brackets: the
Windows HIP events are not trusted, see `probe_pcie_h2d_gbps`).

Derivation (`include/strata/core/calibrate_ratio.hpp`, pure functions): the contended pair is used when both were measured, else the quiet pair.
`ratio = cpu/copy`; balance share `f = cpu / (cpu + copy)` (both lanes finish together; this is `link/(link+R)`, the static rule with k = 1; it is
also the fraction of the CPU-only miss time that the split saves). `f` below `STRATA_CALIBRATE_MIN_SHARE` (default 0.35) means the saving does not
pay for the PCIe path's fixed cost (flag-B wait, launches of the share), so the share is 0 and the verify windows drop that path exactly as
`--pcie-frac 0` does. Otherwise the share is `f` rounded to 0.05 and kept in [0.10, ceiling] (ceiling = `pcie_share_max()`). It also prints the
promotion break-even `copy / (cpu - gpu)`: how many hits a promoted expert needs to repay its copy (advisory only, not fed to the adaptive tier).

The result sets `o.pcie_frac` before the verifier decides on `set_pcie_off` and turns `pcie_frac_auto` off, so the measured share is the one the
run uses (PcieAuto's start-up and refinement would otherwise override it on CUDA/gfx12). Applies only where the static share would be chosen: one GPU,
native pack, no `--pcie-frac`. With `--pcie-frac` given, a layer split, a peer GPU or a canonical pack it prints "ignored" and does nothing.

The threshold 0.35 is a heuristic, not a measurement: it is placed so that this box's expected figures (CPU about 45 us per expert at 16 threads,
PCIe 3.0 x16 about 12 GB/s, 1.4 MB expert = about 117 us, raw share about 0.28) give 0. If the screen shows the probe picks a nonzero share on this
box, raise `STRATA_CALIBRATE_MIN_SHARE`; if it picks 0 on a box where a share wins, lower it. The probe prints every figure so this can be judged from
one log.

## Not lossy

Policy only: the same experts are computed, only the lane differs. A share of 0 is the same run as `--pcie-frac 0`. A nonzero share changes bits only
as every PCIe share does (the CPU lane rounds FP32 rows differently from the GPU lane).

## How to enable

```
STRATA_CALIBRATE=1                       # needs no --pcie-frac on the command line
STRATA_CALIBRATE_CACHE=/path/calib.txt   # optional: read when the key (GPU name, pool workers, host thread, layers, blob size) matches, else probe and write
STRATA_CALIBRATE_MIN_SHARE=0.35          # optional threshold, 0.0..0.9
```
Log lines: `strata generate: STRATA_CALIBRATE (...)` with all figures, then `cpu/copy ratio ..., balance share ... -> pcie_frac ...`.

## Files changed

- `include/strata/core/calibrate_ratio.hpp` (new): `CalibMeasure`, `calib_derive`, `calib_cache_format/parse`, env helpers.
- `src/core/calibrate_ratio_test.cpp` (new) + `CMakeLists.txt` (test registered next to `pcie_share_rule_test`, and in the second test list).
- `src/program/generate.cpp`: `calibrate_probe()` after `measure_pool_affinity`; the call and the override right after the pool/aux-cpu set-up
  (before the expert cache). Default path untouched (the whole block is behind `STRATA_CALIBRATE`).
- `bench/specs/24-a45-calibrate.json` (new screen spec).

## Tests run

None executed: `wait_quiet.sh` returned 1 (a GPU benchmark, `abtest.py 09-prefill`, was running) in all three calls, so by the batch rules nothing was compiled.
`src/core/calibrate_ratio_test.cpp` (host-only, header-only dependency, registered in CMake next to `pcie_share_rule_test`) and the new code in `generate.cpp` were reviewed by hand
(asserts re-derived by hand, one fix: the probe no longer names `cudaMemcpyKind`, which `hip_compat/cuda_runtime.h` does not define). First thing to run in the coordinator build:
`ctest -R calibrate_ratio_test`, then the HIP build of `generate.cpp`.

## Spec to run

`bench/specs/24-a45-calibrate.json`: arms `base` (explicit `--pcie-frac 0`), `calibrate` (flag removed, `STRATA_CALIBRATE=1`), `static-default` (flag removed only: the loss
control). Screen tier, cases short and edit. Pass criterion from the row: `calibrate` must find 0 alone, i.e. its log says `pcie_frac 0.00` and its speed and text hash match
`base`; `static-default` should reproduce the 6-13% loss.

## Expected gain and risk

Gain: 0% here if it finds 0 (it only removes the need to know the flag); 6-13% on other platforms where the static start value is wrong, but that is not
verifiable on this box. Risks: (1) the threshold heuristic above; (2) the probe's CPU pass contends with other processes on the machine (best of two passes
mitigates, and the cache file lets one clean reading serve later starts); (3) the contended copy rate is bytes done over wall time of a CPU-bound loop, so it
can read low if the loop polls too rarely (two chunks of 8 experts are kept in flight, polled twice per layer); (4) the GPU figure is a copy proxy, not the
expert kernel, and only feeds the advisory break-even.

## Left

- Build and run the unit test in the coordinator build; build `generate.cpp` (HIP) - the new code uses only calls already used elsewhere in the file
  (`cudaStreamCreateWithFlags`, `cudaMallocHost`, `cudaEventQuery`, `cudaMemcpyDeviceToDevice`, `cudaGetDeviceProperties`).
- Feed the break-even to `STRATA_ADAPT_KMISS` (k of the k-th-miss admission) once A5 is confirmed; today it is printed only.
- Per-stage probes for a layer split; the canonical Q2_0 pack (needs its own CPU call).
- A per-window refinement from `ms_run / multi_misses` (PcieAuto's counters) around the calibrated start, if the start alone is not stable.
