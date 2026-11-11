# kern-hip: Phase 16 kernel wave, HIP side (F21, F9, A37, A33/A35) - 2026-11-11

Branch `dev/p16/kern-hip` (from `dev/phase16-academic` 3f4ba97cf). No GPU on this machine for this agent: everything
below is compile-only (gfx1100 HIP and CUDA sm_86, one object at a time, `-j1`, 6 GB cap), static analysis of the
compiled code (clang `-Rpass-analysis=kernel-resource-usage`, `cuobjdump -res-usage`, disassembly of the gfx1100 code
objects) and the CPU tests that exist. Gains are estimates from the 2026-11-09 profiles
(`bench/results/2026-11-09-gpuprof-defer/summary.txt`, `2026-11-09-7900xt-gpuprof/`), not measurements.

Profile figures used: deferral profile, 120 windows of ~4.3 tokens, 48 MoE layers, one group per window; a real short
window is ~32 ms (4.3 tokens at ~132 tok/s). The profiler inflates launch gaps.

## Summary

| Item | Finding | Code state | Static (gfx1100) | Expected gain | GPU test |
| --- | --- | --- | --- | --- | --- |
| F21 `STRATA_WAIT_COPY256` | **inert on the preset** (it only changed the STRATA_FUSE_WAIT copy; fast-hip runs STRATA_WAIT_FUSE's) | fixed (424b3fce0) + `hip_wait_copy` test | wide kernel 10 VGPRs; default kernels' ISA identical | <= ~0.6% short, ~0 edits | `ctest -R hip_wait_copy`, spec 25 |
| F9 `STRATA_ROUTE_FAST` | port sound; spec 17 had the wrong base | note + spec fix (bc3a48c41) | `route_publish_fast` 67 VGPRs, no LDS/spill; router GEMV: no VGPR/LDS/occupancy change in 192 instantiations | +1-2% (est.) | `native_multi_parity` (passed 11-10), specs 25 / 17 |
| A37 `STRATA_IQ_LUT_DUP` | ceiling 0.2-0.5% of decode; x16 halves occupancy | unchanged; recommend revert after one microbench | x8: 179 VGPRs, 27.25 KB, 8 waves; x16: 249 VGPRs, 43.25 KB, 4 waves | <= 0.5%, unmeasurable in a screen | `expdec_bench` + counters only |
| A33 `STRATA_EXPDEC_SWIGLU` | works with deferral, never run vs the old kernels | parity CTest added (2fd76cf86) | same occupancy as the kernels it replaces (8 waves) | ~1% (one launch gap per EXPDEC layer) | `iq_multi_parity_expdec_swiglu`, spec 25 |
| A33 `STRATA_ROUTER_FUSE` | inert under deferral (known); F9 is its deferral-compatible successor | none | - | - | skip `a33-router-fuse-nodefer` |
| A35 `STRATA_IC_WARM_MB` | **inert under deferral** (needs the fused CPU combine, which deferral turns off) | spec arms moved to no-deferral (2fd76cf86) | - | unknown | `a33-a35-fusion-retest` nodefer arms |

## F21: STRATA_WAIT_COPY256 (plan copy with 256 threads and 16-byte reads)

Source: android-imdad/Strata-7900xtx `caa9b00115` (read via the GitHub API). There `STRATA_WAIT_FUSE=1` sends the plan
copy through `wait_copy_i32_from_mapped`, whose kernel already read 16 bytes; COPY256 raised it from 128 to 256 threads
("one round of reads instead of two", ping-pong 4.47 -> 3.57 us), with every wave making its own acquire after the
barrier (`block_wait_flag_ge`).

Mismatch in our port (0573c36b6): our `STRATA_WAIT_FUSE` is a different kernel (`wait_flag_copy_i32`), and
`record_window` takes it before the `fuse_wait_on()` branch. The fast-hip preset sets `STRATA_WAIT_FUSE=1`, so
`wait_copy_i32_from_mapped` never runs there and COPY256 did nothing, with or without `STRATA_FUSE_WAIT=1` - the
`fusewait-copy256` arms of specs 18 and 23 measured nothing. The port also dropped the per-wave acquire.

Fix (424b3fce0, `src/kernels/cuda/verify_kernels.cu` only): `wait_flag_copy_i32` takes the wide kernel too when the
switch is on and the plan has more than 512 words; the wide kernel makes a system-scope acquire in every wave after
the barrier (its 16-byte reads are plain loads: `global_load_b128` without glc, so `buffer_gl0_inv / buffer_gl1_inv`
in every wave keep it from reading a line older than the flag). Default path: `wait_flag_copy_i32_kernel` and
`wait_copy_i32_kernel` disassemble identical to before; all 76 functions of the object identical except the wide one.

Why it pays more here than in the fork: our 128-thread loops read 4 bytes and wait for each read before the next
(`flat_load_b32 ... glc dlc; s_waitcnt vmcnt(0); global_store`), so a 16-row plan (1,320 words: `plan_i32_` at
`max_t` 16, K 10) is 11 dependent PCIe round trips; the wide kernel needs 2 (330 int4 over 256 threads).
`wait_flag_copy_i32_kernel`'s minimum in the pre-deferral profile is 5.44 us (flag read + the copy), i.e. ~0.5 us a
round: ~4 us saved per copy when the plan is already there, 48 copies a window, <= ~0.2 ms = <= ~0.6% of a short
window, less where the GPU later waits for CPU rows anyway (edits). Exact (same words).

Test added: `tests/hip/wait_copy.cpp` - the host rewrites the SAME mapped plan words each round after the GPU is
spinning (as the engine reuses the plan buffer every layer); 5 cases (the engine's aligned 1,320 words on both entry
points, a word tail, a misaligned source, a plan under 512 words) x 200 rounds; any stale word fails. CTest
`hip_wait_copy` (default kernels) and `hip_wait_copy_256` (`STRATA_WAIT_COPY256=1`). Compiled, not linked or run.

## F9: STRATA_ROUTE_FAST (router publish fast path)

Sources: android-imdad `caa9b00115` (ROUTE_FAST = DPP router exchanges + the router GEMV copying the MoE rows; its
publish fused with `doorbell_publish_res`, RMW ring) and Ben1078 `23d2fc0450` (top-10 + doorbell in one launch, ring
stored). Neither gives a per-item number; the fork measured on IQ3_S with ~90% VRAM hits and no deferral. Our port
(9409a228c) combines both and goes further: it publishes the weights and runs under deferral. Checked here:

- Arithmetic: `route_publish_fast` is `route_multi` step for step (same loads, trees, NaN rule, exchange order); the
  one change - lane r writes rank r's id from its copy of the butterfly winner - is exact because the exchange is a
  strict total order on (probability, -id) with distinct ids, so every lane ends with the same pair.
- Publish under the preset: deferral's default `STRATA_VERIFY_DEFER_CPU_BELOW` is 0.1, so `heavy` is on and the
  preset publishes with `doorbell_publish` (rows, ids, weights, RMW ring). That also confirms the commit's reading of the
  11-09 non-finite bisect: with ROUTE_PUBLISH and heavy, `route_publish_res` rang and `doorbell_publish` rang again.
- Ring: stored as `(l - lb) * G + grp + 1` = k + 1, what the host waits for at step k after resetting it per window.
- Ordering: the GEMV's row copy into `m_x_` comes after `late_add(l - 1)` (which waits for layer l - 1's CPU flag), so
  the pool never reads rows being overwritten - the same order as the old publish.
- Static: `route_publish_fast` 67 VGPRs, 0 LDS, no spill (CUDA: 78 regs, `route_multi` 79). The router GEMV
  (`bf16_f32_mmvf_multi_kernel`, all 192 instantiations) has the same VGPRs, LDS and occupancy as before F9; only SGPRs
  rose (by 0-10, no effect on RDNA occupancy). The default GEMV pays one uniform branch.
- GPU parity already passed on 2026-11-10 (`bench/results/2026-11-10-batch1/RESULTS.md`: native_multi_parity OK);
  the speed screens of that day were too noisy to decide.

Changes here (bc3a48c41): `verify.cpp` says once when the switch will not act and why (route prior - i.e. every turbo
arm -, route-resident margin, device plan, remote opt, always-publish, lookahead router); `bench/specs/17-f9-route-fast.json`
pointed at the rdna3test preset (no deferral: ROUTE_PUBLISH is the base path there, contrary to its note) - now fast-hip.

Expected gain: per layer `route_multi` 9.7 us + `doorbell_publish` 6.9 us + a 6.8 us gap become one launch of about
route_multi's length: ~12 us a layer, ~0.6 ms a window, ~+1.8% of a short window; the doorbell also rings ~10 us earlier,
so the CPU experts start earlier (edits). Estimate +1-2%: at the screen's resolution, so confirm tier.

Not built: putting `route_publish_fast` into the router GEMV's last block (ROUTER_FUSE-style, under deferral) would
remove one more launch gap (~5 us x 48 = ~0.25 ms, ~0.7%): below the 2% bar; revisit after spec 25.

## A37: STRATA_IQ_LUT_DUP (duplicated IQ2_XXS codebook in LDS) - decision

Source: FLUTE (arXiv 2407.10960) duplicates and vectorizes the LUT to relieve shared-memory bandwidth for 3/4-bit
non-uniform LUT GEMMs on NVIDIA. Its tables are small; IQ2 codebooks are 2 KB (IQ2_XXS) to 8 KB (IQ2_S) of 8-byte
entries, so only IQ2_XXS can be copied 16 times (32 KB). The port (d11a9ca09) matches its description; I re-derived the
layout (copy g in bank pairs [gB, (g+1)B), lane t reads slot t - t%B + lut_dup_slot(i) = entry i) and the bank model
(`lds_lut_conflicts.py --uniform`: plain 5.85, x8 3.99, x16 2.00 clocks a wave lookup; its test: 5 passed).

Static (preset kernel `expdec_gu_kernel<16,1,4>`, 256 threads, 128 KB LDS per WGP):

| Copies | VGPRs | LDS / block | waves / SIMD |
| --- | ---: | ---: | ---: |
| 1 (today) | 188 | 13.25 KB | 8 (VGPR-bound) |
| 8 | 179 | 27.25 KB | 8 |
| 16 | 249 | 43.25 KB | 4 (LDS-bound: 2 blocks per WGP) |

With the switch off, all 32 `expdec_gu` instantiations disassemble identical to the pre-A37 build (one PC-relative
constant offset differs). Non-default trap: `STRATA_EXPDEC_MINB=2` + `STRATA_IQ_SIGN_FAST` + 8 copies spills 8 VGPRs.

Numbers for the decision: the IQ2_XXS gate/up kernel is 1.7% of GPU kernel time in the deferral profile (67 ms of
3,947; 1.2% of the profiled wall). The lookups are ~23 of 33-60 LDS clocks per wave step and DRAM is ~43, so the
kernel is LDS-bound only at 3-4 entries per group. Best case x8 saves ~7 clocks a step (12-22% of the kernel) =
0.15-0.26% of decode; x16 saves ~15 (25-45%) = 0.3-0.55% before the halved occupancy, which hurts a streaming kernel.
On edits the GPU waits for the CPU: ~0. `STRATA_EXPDEC_SWIGLU` (A33) runs its own kernel, which ignores the copies:
if A33 is promoted, A37 is dead code. **Not worth keeping as an engine switch.** Keep the code only for one
`expdec_bench` + rocprof counter run (it is the conflict-free reference that tells whether LDS conflicts cost
anything on gfx1100, which decides the IQ2_S options); revert d11a9ca09 if the 16/42 gate/up gain there is < 10%.
Do not spend a screen (spec 24) on it. For IQ2_S (5.8-8.8%), `STRATA_EXPDEC_R=2` costs no occupancy
(`<22,2,4>`: 184 VGPRs, 8 waves, same as R 1) and is the cheaper lever.

## A33 / A35 retests (16.7 fusion items)

Source check: the plan cites SGLang (arXiv 2312.07104) for "small-M fusion"; that paper is SGLang's frontend and
runtime (RadixAttention, compressed FSM for structured output), not decode-kernel fusion: there is no published gain
to transfer; the fusions come from fork work (Ben1078 23d2fc0450) and our own launch-gap profile. PRESERVE (arXiv
2501.08192) prefetches weights/KV into on-chip L2 during inter-device communication in distributed serving ("up to
1.6x"); here the wait is a single GPU waiting for the host CPU and the cache is the 80 MB Infinity Cache: the method
transfers in principle, the gain does not.

- `STRATA_EXPDEC_SWIGLU` acts with deferral (independent of the CPU-row path). Static: `expdec_gu_swiglu<16|22,1,4>`
  191 / 177 VGPRs, 14.3 / 20.3 KB, 8 waves/SIMD (= the kernels it replaces; `swiglu_q8_1_entries` 16 VGPRs). It was
  never run against the old kernels: CTest `iq_multi_parity_expdec_swiglu` (the existing parity under the switch).
  Gain: the expdec_gu -> swiglu gap is 9.2 us x ~34 IQ2_S layers a window in the profile plus the SwiGLU kernel, ~1%.
- `STRATA_ROUTER_FUSE`: inert under deferral (rpub needs !defer_), as the 11-11 audit and rankguard's start-up note say;
  F9 does the router-publish fusion under deferral. `a33-router-fuse-nodefer` only measures a world without deferral.
- `STRATA_IC_WARM_MB`: **also inert under deferral** - `warm_now` needs `cfuse_on(n)`, and `cfuse_on` is false with
  `defer_` (verify.cpp), so the preset prints the start-up line and never launches the warm kernel. The spec's ic-warm
  arms were no-ops; they now run as nodefer vs nodefer+ic-warm32/64. Only a gain >= 2% there justifies moving the
  fork to the deferral's own wait (post(l)'s heavy-row wait, `wait_flag_ge(m_pflag_)`), a code change not made here.

## Tests run here

- HIP gfx1100 objects (build-hip, Release, `-j1`, 6 GB cap): native_router.cu, native_bf16.cu, verify_kernels.cu,
  iq_kernels.cu, core/verify.cpp, native_multi_parity.cpp, iq_multi_parity.cpp, expdec_bench.cpp, tests/hip/wait_copy.cpp:
  all compile (warnings only, pre-existing). CUDA sm_86 (build-cuda): native_router.cu, native_bf16.cu,
  verify_kernels.cu, core/verify.cpp compile. Nothing linked (strata_kernels is not built whole here).
- Resource/ISA comparisons against pre-change objects (git show of 9409a228c^ / d11a9ca09^ compiled with the same
  command): numbers above.
- `pytest tools/pancake/test_lds_lut_conflicts.py` 5 passed; `lds_lut_conflicts.py --uniform` reproduced the A37 table;
  `pytest tools/pancake/test_abtest_cases.py` 6 passed (specs load).
- Not run: every GPU test and screen (no GPU).

## GPU queue for the lead (in this order)

1. `ninja strata native_multi_parity iq_multi_parity hip_wait_copy expdec_bench` (build-hip).
2. `ctest -R 'native_multi_parity|iq_multi_parity|hip_wait_copy' --output-on-failure` - includes `hip_wait_copy_256`
   (F21) and `iq_multi_parity_expdec_swiglu` (A33); all must pass before any screen.
3. A37: `build-hip/expdec_bench 300 0 20 16/42,22/42 def,dec:l0,dec:l8,dec:l16,dec:r2` plus the rocprofv3 counter line
   of `pancake-docs/batch2/A37-gpu-kernels.md`; revert d11a9ca09 if the 16/42 gu gain is < 10%.
4. `python tools/pancake/abtest.py bench/specs/25-kern-hip.json --tier confirm --out bench/results/<day>-25-kern-hip`
   (copy256, route-fast, expdec-swiglu, all3 vs fast-hip; quiet machine, hugepages as in RESULTS of 11-10).
5. `python tools/pancake/abtest.py bench/specs/a33-a35-fusion-retest.json --tier screen` (A35 in the no-deferral world).
6. Optional: spec 17 (adds the `doorbell-store` arm: the RMW part of F9 alone).
Skip: `24-a37-lut-dup` (screen cannot resolve <= 0.5%), `a33-router-fuse-nodefer` (superseded by F9).

Risk: F21 changes how plan words are read (cache policy), covered by `hip_wait_copy_256`; F9's GEMV row copy adds
PCIe writes inside the router GEMV (the old publish did the same writes in its own launch). All exact.
