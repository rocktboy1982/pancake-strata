# kern-cpu: 16.10 CPU expert path, Q2_0 / IQ kernels (F7, F12, A48, CPU pool switches)

Branch `dev/p16/kern-cpu` (from `dev/phase16-academic` 3f4ba97cf), 2026-11-11.  CPU only: no GPU, no engine run.
Microbenches pinned to cores 6,7 at `nice -n 19`; the other cores (and the SMT siblings 14/15) carried other agents'
work, so the 2-thread numbers are noisy (medians and best runs are both given).

| Commit | What |
| --- | --- |
| 3bafb6ab9 | F7: `STRATA_Q2_FAST` bitwise the legacy kernel under any compiler flags (corr-form probe), stronger parity test |
| f9e0d5644 | F12: the sign step in `include/strata/kernels/iq_signs.hpp` (host + device), exhaustive host test, ctest labels |
| ee4fda5e4 | A48: `q2_rows_bench` prints the DDR4 floor per row and runs the pool shape on N threads |

## The DDR4 bound

5700G, dual-channel DDR4-3200 (51.2 GB/s theoretical).  The plan's figure is 33 GB/s measured read, the pool runs at
~27 of 33 GB/s (MOD_PLAN 16.10).  A kernel cannot stream a row faster than row bytes / bandwidth.

- Q2_0 rows: gate/up 40 blocks x 18 B = 720 B, down 10 x 18 B = 180 B.  One expert = 1,280 gate/up rows + 2,560 down
  rows = 1,382,400 B.
- Floor at 33 GB/s: gate/up 21.8 ns/row, down 5.45 ns/row, one expert 41.9 us (at the pool's 27 GB/s: 51 us).

One thread, rows from DRAM (256 MB), median of 5 runs (`q2_rows_bench --seconds 0.5 --reps 7`, core 6/7):

| nt | shape | legacy ns/row (x floor) | fast ns/row (x floor) | fast / legacy |
| --- | --- | --- | --- | --- |
| 1 | down | 31.9 (5.9) | 21.8 (4.0) | x1.47 |
| 1 | gate/up | 111.3 (5.1) | 89.3 (4.1) | x1.25 |
| 2 | down | 41.3 (7.6) | 34.5 (6.3) | x1.20 |
| 2 | gate/up | 151.6 (6.9) | 129.2 (5.9) | x1.17 |

L2-hot rows give the same per-row times (down 32.3 / 20.5, gate/up 109.3 / 86.2 at nt 1): one core is compute-bound,
not memory-bound.  A plain read of the same bytes runs at 22-24 GB/s on one core, 25-29 GB/s on two.

Per expert on one thread: nt 1 legacy 224 us, fast 170 us; nt 2 legacy 300 us, fast 254 us.  Divided by the 41.9 us
floor, that is the number of cores a kernel needs to reach 33 GB/s:

| | nt 1 | nt 2 |
| --- | --- | --- |
| legacy | 5.4 cores | 7.2 cores |
| fast (F7) | 4.1 cores | 6.1 cores |

Reading: at one token per expert both kernels saturate DDR4 with fewer cores than the pool has (6-12 threads give the
same result, research_notes cpu_ram.md), so the lever there is the 27 -> 33 GB/s gap: at most ~22% of CPU-expert
time, for any kernel.  At two or more tokens per expert (verify windows) the legacy kernel needs ~7 of the 8 cores
before pool overheads, i.e. it is close to compute-bound, and F7 has real room.  This is a model from single-thread
numbers; the pool number with 8 threads is the open A48 measurement (below).

Two threads (pool shape, 64-row calls, DRAM): best runs scale linearly (gate/up nt 1: fast 44.95 ns/row = 16.0 GB/s,
legacy 56.1 = 12.8 GB/s; down nt 1: fast 10.55, legacy 15.53).  Medians are disturbed by the shared cores (gate/up nt 1:
fast 12.1 vs legacy 11.9 GB/s; down nt 1: 15.7 vs 11.1 GB/s).

## F7 `STRATA_Q2_FAST=1` (fast bit-exact Q2_0 AVX2 kernel)

Source check (imanu86 c3fc8f34d3, read through the upstream URL): matches - 8-op unpack, d broadcast once per block,
4 rows (1 token) / 2 rows (2 tokens) at a time on short rows, long rows one at a time.  Mismatches:
1. The source writes `corr += d * (hx0 + hx1)` plainly, so its bit-exactness depends on the compiler contracting both
   kernels the same way.  Our port hard-coded `std::fma`, which matches the legacy kernel only under GCC's default
   `-ffp-contract=fast` at -O2+.  **Bug found and fixed:** in a `STRATA_HOST_NATIVE` build (`-ffp-contract=off`, the
   `build-hip-native` used by bench/specs/08-build.json) 114,605 of 584,676 outputs differed; GCC -O0/-O1 and MSVC
   /fp:precise do not contract either.  Now the fast kernel has both forms spelled out (`std::fma`, or a product rounded
   on its own behind an empty asm) and `q2_fast_corr_form()` picks the one the binary's legacy kernel gives on a probe
   row where they differ (at 1-4 tokens, per variant); neither form -> the legacy rows and a note.  Probed only when the
   fast kernel is asked for; the default path is unchanged.
2. Claimed gains: author 1.11-1.45x on one thread (ours: x1.17-1.47, above), pool "all Q2_0 -17% per layer" with 7
   workers and ~2 CPU experts per layer, CPU not named.  Our edit window computes ~7.4 experts per layer on the CPU and
   is DDR4-bound at one token per expert, so the author's pool gain does not transfer as is (bound above).
3. Rows grouped on long rows (tried here, not kept): gate/up R=4 at nt 1 is 8% faster from L2 (78.2 vs 84.6 ns/row) but
   0.66x from DRAM (132.4 vs 87.5); R=2 at nt 2 is 145.5 vs 128.9 from DRAM.  This agrees with the source's note
   (interleaved long rows break the hardware prefetcher's stream, 0.89x).

Tests (all passed):
- `q2_fast_parity_test`, default build (Release, GCC 15.2): 2,592 cases, 1,902,900 outputs, 957,424 of them with edge
  inputs (fp16 +-Inf / NaN / max scales, code bytes swept over all 256 values per position, raw `scale` / `hx` with float
  subnormals, -0, +-Inf, NaN, FLT_MAX, random bits, int8 -128), 0 differ, 0 NaN payload differences, form fma.
- Same test, `STRATA_HOST_NATIVE` build: 0 differ, form "rounded product + add"; 253 NaN outputs differ in payload
  only (both NaN: x86 passes on the first NaN operand and the compiler orders a commutative add / FMA form freely).
  The test counts these and does not fail on them; nothing reads a payload.
- Manual -O0 / -O1 / -O2 builds of the kernel file: PASS (forms rounded / rounded / fma; -O0 11,024 and -O1 4 NaN
  payload differences).
- `q2_bitplane_parity` PASS; `ctest -L cpu`: 85 of 85 passed (4 skipped: no AVX-512 / environment), also with
  `STRATA_Q2_FAST=1` set; `q2_rows_bench`: "bitwise equal on all shapes: yes" in both builds, dispatcher picks FAST.
- Not run: the AVX-VNNI copy (Zen 3 has no AVX-VNNI); MSVC build.

Risk: low (opt-in; the probe makes a mismatch fall back to legacy).  Next: engine screen of `STRATA_Q2_FAST=1` on
short and edit traffic (GPU); expect a few % at most on one-token windows, more where verify windows put 2+ tokens on
an expert.  Then the 8-thread A48 run below decides the default.

## A48 single-thread and pool GB/s against the best x86 2-bit GEMV

Sources: Intel "Pushing the Envelope of LLM Inference on AI-PC" (arXiv 2508.06753, HTML v1): 2-bit weights in an
offline VNNI4-interleaved layout, expanded to int8 with ~1 shift, 2 AND, 4 vpshufb per 128 weights, multiplied with
int8 activations by vpdpbssd (AVX-VNNI-INT8) on Arrow Lake / Arrow Lake H / Lunar Lake (98 / 75 / 97 GB/s measured);
2-bit GEMV within 10-20% of the int8 roofline (Lunar Lake 20-25%), 1-bit unpack-bound.  T-MAC (arXiv 2407.00088): LUT
mpGEMV without dequantization, throughput linear in bits, ARM devices on the abstract page (BitNet 3B 30 tok/s on one
M2-Ultra core, 71 on eight).  Mismatches for us: Zen 3 has no AVX-VNNI / VNNI-INT8 (maddubs + madd instead), the GGUF
Q2_0 layout is fixed (an interleaved layout needs a repack), and a 64-value block carries one fp16 scale plus two
activation scales.  Our pool at ~82% of measured bandwidth sits inside the Intel paper's 80-90% band.  Verdict:
LUT / T-MAC stays rejected (cuts compute, not bytes; the pool is DDR4-bound at one token); no arithmetic rewrite
beyond F7 is worth building.

Code: `q2_rows_bench` now prints the floor per row (`--bw`, default 33 GB/s) and "x floor" per arm, and `--threads N`
runs the DRAM rows in the pool's shape.  Numbers above.  Open: `taskset -c 0-7 q2_rows_bench --threads 8 --nt 1` and
`--nt 2` on an idle box (needs all 8 cores, not allowed in this session) - that is the pool GB/s A48 asks for.

## F12 `STRATA_IQ_SIGN_FAST=1` (IQ2 sign expansion, GPU EXPDEC gate/up)

Source check: llama.cpp's `__vcmpne4` / `__vsub4` sign step replaced by the multiply-free form (RX5950XT b9a2111d3f,
llama.cpp `ggml_cuda_apply_signs` without its multiply); boxwrench's -29% / -35% kernel time is from another engine (GLM,
7900 XT) and is not assumed here.  The method is exact when every byte of t has at most one bit and no codebook byte
is 0; both hold.  This is GPU code, so here only the identity and the build are checked:
- `iq_sign_fast_test` (new; the 22,229-case check named in 3e1583773 was not in the tree): every word of the five
  codebooks (4,352, none with a zero byte) x every one-bit-per-byte sign word (6,561, a superset of the 768 format
  sign fields) against the CUDA semantics and the HIP SWAR emulation, plus a negative control: 28,553,472 cases,
  0 failures.  It tests `iq_signs_fast`, the function `iq_signed<true>` now calls.
- HIP gfx1100 object of `iq_kernels.cu` builds (0 errors); device code before vs after the refactor: `.text` (5.8 MB),
  `.rodata`, `.data` byte-identical, disassembly identical (only `.dynstr` differs: the source path in internal
  symbol names).  CUDA sm_86 object builds (0 errors).
- Register cost (gfx1100 metadata, same object): at the default `STRATA_EXPDEC_MINB=4`, R=1 the SF=1 kernels use
  182-189 VGPRs vs 177-191 for SF=0 (no occupancy change at <= 192); with MINB=2 the fast form adds up to 72 VGPRs
  (IQ2_S R=1 swiglu 178 -> 250), so keep it a per-format A/B as planned.
- Not run: expdec_bench, iq_multi_parity, the screen (GPU).

## Q2_PREFETCH / POOL_TAIL4 / MWAITX / LAZY_PARK (not rebuilt)

Confirmed from bench/results/2026-11-09-test-phase/03-cpu-pool.jsonl: the summary's huge intervals come from two
memory-compaction runs (base r2, q2-prefetch r1, ~25% slow).  Without them every arm is within about +-1.5% of base
(short: base 131.2 / 128.9, mwaitx 131.0-130.7, tail4 128.8-131.7, lazy-park 128.7-131.5, q2-prefetch 131.3-131.5;
edit: base 135.5 / 130.7, mwaitx 135.6-137.1, tail4 131.0-136.4, lazy-park 135.6-136.2, q2-prefetch 134.8-135.9).
Closed as n.s.; consistent with the microbench (one core is compute-bound, a prefetch hint has nothing to hide).

## Other 16.10 / 16.5 items

- A47 done (in fast-hip).  A48: above.  16.5 (A27 HOBBIT / MoBiLE low-precision miss copy) is lossy and owned by the
  shadow-pack work; A29 closed.  Nothing else in 16.10 is uncoded and sound for 8-core Zen 3 + DDR4:
  `row_multi<1>` chain splitting changes the summation order (lossy; F7's row groups are the exact form), an offline
  VNNI4 repack would remove at most the unpack (~9 of ~33 vector ops per block-row) and still meets the DDR4 floor at
  one token, and long-row grouping measured slower from DRAM (F7 item 3).
