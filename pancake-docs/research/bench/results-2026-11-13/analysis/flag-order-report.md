# hip_flag_order_test root-cause analysis (read-only, static)

Saved by the main session from the agent's final text (the agent's own write was refused). No GPU run, no HIP binary executed, no repo edits.

## Summary
- Top hypothesis, about 60% (derived): the consumer kernel reads host-mapped staging right after a volatile spin on the flag, with no system-scope fence between them. The engine's wait kernel has `__threadfence_system()` after its spin (`verify_kernels.cu:1330-1333`). Host-side ordering (`raise_flag_b_ordered`) is correct: all flag-value checks pass. Only the staging reads fail, and the counts show one device failing all staging checks.
- Segfault in a direct run: the binary is built against ROCm core-10.0 (libamdhip64 7.15). ctest sets `LD_LIBRARY_PATH` to core-10.0. A direct run loads the system libamdhip64 7.1.52801 with HSA 1.18. A mixed runtime can crash before any output (derived, unverified).
- Not a regression from a later commit: no change to the helper or the call sites since 40849df0a.

## Ordering contract (layer L, want = L+1)
1. `fetch_dma` (`src/core/verify.cpp:4390-4415`) enqueues uploads on the copy stream, then `raise_flag_b_ordered(want)` (line 4414).
2. The helper (`include/strata/core/flag_order.hpp:46-57`) enqueues a host function on the copy stream. That function calls `flag_raise_max`, which only raises. It runs after all earlier uploads.
3. Empty plans use the same ordered raise.
4. The graph waits on the counter (`verify.cpp:2859-2862`), then reads staging.

Host side: the flag rises only after the bytes are written (copy-stream FIFO + host function + monotonic counter). Device side: the GPU must observe the bytes after seeing the flag, which needs a system-scope acquire after the spin. The engine's wait kernel has it. The test's consumer (`tests/core/flag_order_test.cu:26-29`) does not:

    while (*flag < want) {}
    *seen = *staging;

## Check by check (ctest log of 2026-11-12 22:20)
- Line 54 (flag == 0 while copy in flight): pass.
- Line 57 (seen == 0xAA): fail x1.
- Lines 59, 73, 79, 85 (flag values): pass.
- Line 67 (negative control): pass.
- Line 101 (seen == 0x55+rep, graph replay): fail x2.

Counts: 57 x1 and 101 x2 means one device fails, not both (both would give 57 x2 and 101 x4). Attribution is inferred, since FAIL lines carry no device index.

## Segfault in a direct run
- Missing device: no. HIP sees two devices (RX 7900 XT and the Vega iGPU); the test loops over both.
- ctest sets only `LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib` for this test. No device is hidden.
- The binary has NEEDED libamdhip64.so.7 and no RUNPATH. Under ctest it loads the matched 7.15 runtime. Directly, it loads the system 7.1.52801 with HSA 1.18: a mixed ABI (derived, unverified).
- Wrong-device pointer: no. `cudaSetDevice(dev)` comes first. Buffer lifetimes are drained before going out of scope.

## Most likely cause (H1)
Test consumer reads staging after the spin with no system-scope fence. The engine's equivalent has one. The allocation is the same default mapped memory in both.

H1b (less likely, unverified): the store `*seen = *staging` may not reach host memory before `cudaStreamSynchronize` returns, because `seen_h` is non-coherent mapped memory. The GPU run below separates H1 from H1b.

H3 (rejected): copy-stream ordering. The same host-function ordering passes lines 54, 59, 73, 79, 85.

Reporting defect: line 120 prints "device N ok" regardless of failures, and `CHECK` does not stop the run.

## Proposed fix (diff text, NOT applied)
Test side (`tests/core/flag_order_test.cu`):

    __global__ void consumer(volatile uint32_t* flag, uint32_t want, volatile uint32_t* staging, volatile uint32_t* seen) {
        while (*flag < want) {}
    +   __threadfence_system();   // system-scope acquire after the spin, as wait_flag_ge_kernel
        *seen = *staging;
    +   __threadfence_system();   // publish the store to the host
    }

Reporting: print the failure count per device rather than "ok" regardless.

Direct runs: set `LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib`. A RUNPATH for all HIP binaries would fix it everywhere; that is an owner decision.

## Single GPU run to confirm (queue after the overnight chain; not run)
- Build A (current test) and B (with the diff above), in a scratch build directory outside the repo.
- For each: `LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib HIP_VISIBLE_DEVICES=0 <bin>` and the same with `=1`.
- Decision:
  - A fails on one device, B passes there: H1 confirmed. Apply the diff.
  - B still fails there, device value 0xAA but host reads 0: H1b. Try coherent host allocation for the test buffers.
  - B fails on both: copy-stream host-function path is wrong on this runtime. Check with an event on the copy stream.
  - A segfaults unset and passes set: H2 (runtime mismatch) confirmed.

## Open points
- The worktree `build-hip/CMakeCache.txt` names build-int as its source dir, so confirm which build directory is authoritative before the run.
- Not verified: the engine's own staging readers after `wait_flag_ge_or` (around `verify_kernels.cu:1619-1650`) may lack the fence. If so, the engine has the same defect and the fix is larger.
- Static analysis only. The 7.15 runtime behaviour on the two devices was not reproduced.

## Correction to the earlier fence experiment
My earlier scratch test added one fence after the spin and ran the binary directly. The direct run segfaults for the runtime-mismatch reason above, so that experiment did not test the hypothesis. It should be repeated with `LD_LIBRARY_PATH` set, as in the run plan.
