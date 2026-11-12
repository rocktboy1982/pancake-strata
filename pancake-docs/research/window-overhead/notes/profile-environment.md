# Phase 18 item 1: environment of the profiled run (2026-11-12)

Profile: `pancake-docs/research/bench/profile/` (summary.txt and CSV stats). Run 15:39-15:41, one request, short case, rocprofv3 kernel trace.
Config used: `flashnext-iq2_xs-fast-hip-nodefer.json`, created at 15:38 from `configs/pancake/flashnext-iq2_xs-fast-hip.json` with the deferral line removed. So the profiled run equals the current fast preset.
Engine: `build-hip/strata` (worktree agent-ad25cdf1dc7ea704a, branch dev/p17/integration).

## Switch states in the profiled run
Label: [config] = set in the config file; [default] = unset, engine default read from code.

| Switch | State | How set |
|---|---|---|
| STRATA_WAIT_FUSE | 1 | [config] |
| STRATA_FUSE_WAIT | off | [default] `verify.cpp:460-470`: on HIP, unset means on only when `gfx_arch_is_gfx12` (gfx1100 is not gfx12, so off) |
| STRATA_VERIFY_DEFER_CPU | off | [default] `verify.cpp:977`: only "1" or "check" turns it on |
| STRATA_COMBINE_POLL1 | off | [default] `native_moe.cu:268`: only "1" turns it on |
| STRATA_WAIT_COPY256 | off | [default] `verify.cpp:1625`, `verify_kernels.cu:1798` |
| STRATA_ROUTE_FAST | off | [default] `native_router.cu:491` |
| STRATA_VERIFY_DEVICE_PLAN | off | [default] `verify.cpp:1457`, `expert_source.cpp:5274` |
| STRATA_MTP_EAGER | off | [default] `mtp.cpp:1863` |
| STRATA_EXPDEC_DOWN | 1 | [config] |
| STRATA_ROUTE_PUBLISH | 1 | [config] |
| STRATA_POOL_FUSE_PHASES | 1 | [config] |
| STRATA_GR_RDNA_A8 / _Q8 | 1 / 1 | [config] (lossy class in the env table) |
| STRATA_PF_F16 | 2 | [config] (lossy class in the env table) |
| STRATA_ROUTE_TAIL_SKIP | 0 | [config] |

CLI: `--spec 4 --spec-min-p 0.5 --lookup-max-t 12 --kv int8 --kv-resident 32768 --max-context 131072 --adapt-swaps 32 --adapt-every 1 --pcie-frac 0` (full list in the config).

## What this settles
- `wait_flag_copy_i32` (5.3%) is the WAIT_FUSE plan copy: WAIT_FUSE is on, FUSE_WAIT is off.
- The CPU-row deferral was off, so `wait_flag_ge` and `combine_cpu` are the non-deferred waits.
- FUSE_WAIT is off on the 7900 XT by default. Item 4 (FUSE_WAIT with WAIT_FUSE off) is therefore a real A/B: FUSE_WAIT=1 vs unset.

## What remains open
- The engine log does not print these switches, so this table comes from config and code, not from a runtime dump. A runtime check needs the engine to print its switch state.
- Re-profile with stream IDs (the rest of item 1) needs the GPU. The GPU is booked overnight (queue from 23:00).
