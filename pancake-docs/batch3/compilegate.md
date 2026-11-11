# Batch 3, compilegate: first compile of dev/batch-2 (+ the Phase 16 plan commit)

Branch `dev/p16/compilegate` (from `dev/phase16-academic` 727eb8a62; the source tree equals `dev/batch-2` d0f638306, the plan
commit only touches MOD_PLAN.md). dev/batch-2 was coded in a night with no C++ built. This note records the first build.
No source file is changed on this branch. Nothing here ran on a GPU: every test ran with the GPUs hidden, every GPU object was
compiled only.

## Switch, source check

None. This is a build and test baseline, not a method; there is no paper to check and no engine switch.

## What was built (every command under `systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 nice -n 19`, ninja -j2)

| Config (build dir in the worktree) | Configure | Built |
| --- | --- | --- |
| CPU only (`build-cpu`) | `-DSTRATA_ENABLE_HIP=OFF -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=ON` (+ `-DSTRATA_BUILD_CONVERSATION_TESTS=ON` second pass) | target `strata_tests` (92 executables, then 97) |
| HIP gfx1100 (`build-hip`) | `-DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DSTRATA_PREFILL_MMQ=ON -DSTRATA_BUILD_TESTS=ON`, ROCm `/opt/rocm/core-10.0` | the 31 objects of the files changed since port/rdna3-decode (30 ok, generate.cpp failed), one `ninja <obj>` at a time; then the 108 executables of the `cpu` tests; then the other 38 objects that include a changed header (not linked) |
| CUDA sm_86 (`build-cuda`), extra | `-DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86-real`, nvcc 13.1 | the same objects as the HIP pass, plus 96 more objects that include a changed header (compile only, no link) |

llama.cpp was not fetched: `-DFETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP=<copy of the pinned 3cf03257 tree>` (the copy sits in
`build-cpu/llama.cpp-src`, not committed). The ninja binary is `Strata/.venv/bin/ninja`.

## Result: compile errors

One file does not compile on dev/batch-2:

| File:line | Error | Introduced by | Owner |
| --- | --- | --- | --- |
| `src/program/generate.cpp:2865` (`strata::CalibMeasure`), then 8278-8329 (`calib_enabled`, `CalibMeasure`, `calib_cache_parse`, `calib_cache_format`, `CalibResult`, `calib_min_share`) and 2894-3018 (`CalibProbeOut::m`, follow-on) | 21 errors in the HIP build: `include/strata/core/calibrate_ratio.hpp` is never included (the CUDA build has the same missing header; I only compiled it there with the patch) | b0abbc5cc "A45: STRATA_CALIBRATE=1 start-up probe ..." (merged by 7ec8e29c2); its 233 added lines use the header but the commit adds no `#include` | policy (A45) |

Everything else compiled with no error. With one line added (`#include "strata/core/calibrate_ratio.hpp"` next to the other
`strata/core/*` includes) generate.cpp compiles clean in HIP (77 s) and CUDA (86 s), so the 21 errors are all one cause. That
one-line patch is commit 1a06bb1c9 on the side branch `dev/p16/compilegate-calib-include-patch` and is NOT on this branch:
the lead asked that policy fixes it. `git cherry-pick 1a06bb1c9` applies it.

Warnings: the HIP builds print the usual `-Wunused-result` / nodiscard `hipError_t` warnings (before this work too); the only
other ones are unused variables in `src/prefill/prefill.cpp:6986,7065`, `src/kernels/gdn_parity.cpp:248`,
`src/kernels/iq_parity.cpp:47,52` (unchecked fread) and dangling-reference warnings in `src/poly/conversation_disk.cpp:184`.
None is in a batch-2 line.

## Result: files changed since port/rdna3-decode (`git diff --stat d251be8f5 dev/batch-2 -- src include`, 62 files)

d251be8f5 itself has one commit that dev/batch-2 does not (mtp --mtp-q4 head size); it only touches src/core/mtp.cpp, so the diff
still lists mtp.cpp. "-" = not part of that configuration.

| File | CPU-only | HIP gfx1100 | CUDA sm_86 |
| --- | --- | --- | --- |
| src/program/generate.cpp | - | **FAIL, 21 errors (A45 include; owner policy)** | not compiled without the patch (the header is missing the same way); ok with it, 86 s |
| src/core/verify.cpp | - | ok | ok |
| src/core/expert_source.cpp | - | ok | ok |
| src/core/mtp.cpp | - | ok | ok |
| src/core/conversation_disk_tier.cpp (+ its test target) | - | ok | ok |
| src/core/calibrate_ratio_test.cpp, epoch_bandit_test.cpp, row_miss_test.cpp, verify_ecap_test.cpp, ple_residency_test.cpp | ok (ple_residency: -) | ok | ok |
| src/core/conversation_cache_test.cpp | ok (needs STRATA_BUILD_CONVERSATION_TESTS=ON) | - (not built there) | - |
| src/spec/disco.cpp, suffix_index.cpp, disco_test.cpp, suffix_index_test.cpp | ok | ok | ok |
| src/kernels/cpu/q2_avx2.cpp (+ q2_avx2_rows.inl) | ok | ok | ok |
| src/kernels/cpu/portable.cpp (aarch64 stand-in) | not built on x86; `g++ -fsyntax-only` ok | - | - |
| src/kernels/cuda/iq_kernels.cu, native_bf16.cu, native_router.cu, qsa_prompt_attn.cu, route_prior.cu, verify_kernels.cu | - | ok (iq_kernels 86 s) | ok (iq_kernels 151 s) |
| src/qwen35/q36_engine.cu | - | ok | ok |
| src/kernels/expdec_bench.cpp, iq_multi_parity.cpp, native_multi_parity.cpp, qsa_prompt_attn_parity.cpp | - | ok | ok |
| src/poly/fattn.cu, fattn.hpp, fattn_mma.cuh | - | ok, but only stubs: fattn.cu is `#if !defined(GGML_USE_HIP)` around the kernel, so HIP does not check the kernel | ok (fattn.cu 11 s, includes fattn_mma.cuh) |
| src/poly/fattn_test.cu | - | - (CUDA only) | ok |
| src/poly/main.cpp, qwen35.cpp (+ qwen35.hpp) | - | ok | ok |
| 29 changed header-like files (include/strata/**, src/poly/*.hpp, fattn_mma.cuh, q2_avx2_rows.inl) | - | every header is included by at least one built object: ok | ok |

`src/poly/engine.cpp` (not changed in batch-2, `#include <cuda.h>` since the port commit 53d9b0925) does not compile in the HIP
config; the CMake comment already says strata-poly is CUDA only. It compiles in the CUDA config.

## Result: ctest

All with the GPUs hidden (`ROCR_VISIBLE_DEVICES=HIP_VISIBLE_DEVICES=CUDA_VISIBLE_DEVICES=-1`, set by the test labels).

| Config | `ctest -L cpu` | Skipped (reason) |
| --- | --- | --- |
| CPU only | 109 tests: 102 passed, 0 failed, 7 skipped | io_priority_test (Windows only); expert_multi_test (this CPU has no AVX-512); native_hadamard_test, tq_cpu_parity, tq_cpu_parity_avx2 (the pinned llama.cpp has no trellis types); expert_parity, pool_test (need pack/full/experts.bin, not in a worktree) |
| CPU only + conversation tests | 114 tests: 107 passed, 0 failed, 7 skipped (same 7) | the 5 added: conversation_cache/split_failure/memory/file, session_save_reclaim: all pass |
| HIP gfx1100 | 132 tests: 124 passed, 0 failed, 8 skipped | the 7 above plus platform_memory_test (lock_resident: `ulimit -l` 8192 KiB < 256 MiB; its other cases pass) |

PANCAKE_STATUS (item 93) says 117/0/6 on the earlier base. The `_strata_tests_cpu` list grew from 131 to 140 names since
d251be8f5; in the HIP configure that is 123 -> 132 registered cpu tests. The nine new ones (calibrate_ratio_test,
disco_test, epoch_bandit_test, promo_chunk_test, route_prior_scope_test, row_miss_test, suffix_index_test, think_probe_test,
verify_ecap_test) all pass. 124 passed = 117 + 9 - 2 and 8 skipped = 6 + 2, so two tests that passed before now skip. The skip
messages name expert_parity and pool_test (need `pack/full/experts.bin`, which is in neither this worktree nor the main
checkout) and platform_memory_test (`ulimit -l`, a limit of the shell it ran in); which two of those ran before I cannot tell
from PANCAKE_STATUS, and none of them is a batch-2 file. I did not load any model or pack. Failed: none.
The first HIP ctest run showed 9 failures that were only unbuilt executables (spec_admission_test, iq512_gather_parity,
nvfp4_avx512_parity, ple_reader_test, ple_reader_stress_test); building them cleared all nine.

## Tools

`tools/pancake/check_stubs.py`: ok (4 stub files, 561 files scanned).

## Risk

Low: this branch has no source change. The one broken file is on the A45 path; the fix is one line (see above). The CUDA and HIP
passes are compile only: no kernel was run, so the parity tests of the changed kernels (iq_kernels, native_router,
qsa_prompt_attn, fattn_mma) are still unrun.

## Next

1. policy cherry-picks 1a06bb1c9 (or adds the include itself).
2. Run `ctest -L gpu` on a quiet card (the kernel parity tests) when the GPU is free.
3. Make the test classification independent of the backend: 23 host-only tests are only registered when HIP or CUDA is on, so a
   CPU-only configure runs 109 cpu tests where the HIP one runs 132.
