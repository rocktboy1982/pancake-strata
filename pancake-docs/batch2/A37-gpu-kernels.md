# A37 / A31 / A36: GPU kernel items of Phase 16.7 and 16.6 (batch 2, 2026-11-10)

Branch `work/a37-gpu-kernels` (from `origin/dev/batch-2`). Nothing here was built or run on a GPU: the coordinator builds
at 02:00. Every new behavior is opt-in and default off; the default kernels are unchanged (the new template parameters
default to the old code path, textually the same expressions).

| Item | Outcome | Switch | Exact? |
| --- | --- | --- | --- |
| A37 LDS bank conflicts in the HIP IQ kernels (FLUTE) | static analysis on real model rows: the codebook lookups run at ~3x the conflict-free LDS time; duplicated-LUT variant coded for IQ2_XXS; IQ2_S cannot be fixed by duplication | `STRATA_IQ_LUT_DUP=16` / `8` | exact (same words) |
| A31 GDN prefill chunk 64 | not coded: design, numbers and the measurement that decides it | none | (rounding class when built) |
| A36 weight-major hit experts / Q2_0 down | not coded: decision rule implemented as a tool | `tools/pancake/probes/hit_critical.py` | n/a |

## A37: LDS bank-conflict check of the HIP IQ expert kernels

Paper: FLUTE, Guo et al. (MIT), EMNLP 2024 Findings, https://arxiv.org/abs/2407.10960 - a LUT GEMM that duplicates the
lookup table in shared memory so that lanes reading random entries hit different banks.

### Which kernels and which LDS arrays

The decode profile (bench/results/2026-11-09-7900xt-gpuprof, fast-hip, and 2026-11-09-gpuprof-defer) puts the VRAM-hit
expert kernels at: `expdec_gu_kernel<22,1,4>` (IQ2_S gate/up) 5.8% / 8.8% (defer) of GPU kernel time,
`native_down_multi_kernel<42>` (Q2_0 down) 5.8% / 6.1%, `expdec_gu_kernel<16,1,4>` (IQ2_XXS gate/up) 1.6% / 1.7%.
The preset now also sets `STRATA_EXPDEC_DOWN=1`, so the down runs `expdec_down_kernel<42>`. The other large kernels have
no LDS lookup table: `rdna_kq_kernel<23>` (IQ4_XS) decodes its 16-entry table in registers with `v_perm`
(`iq4_lut4`, src/kernels/cuda/rdna_mmvq.cu), and Q2_0 has no codebook.

LDS accesses of `expdec_gu_kernel` (src/kernels/cuda/iq_kernels.cu), per 16-lane sub-warp (lane t, k = t + 32j and
t + 16 + 32j; the two sub-warps of a wave32 take rows r and r + 1 of the same k):

| Array | Access | Banks (32 x 4 B) | Verdict |
| --- | --- | --- | --- |
| `s_u` activation planes, `ed_load_u` | two 16-byte reads per (entry, k), 16 consecutive k | consecutive int4: conflict-free; the other sub-warp reads the same addresses (served again in its own passes: 2x redundant traffic, not a conflict) | ok |
| `s_ds` scales | one float per (entry, k), consecutive | conflict-free, broadcast across sub-warps | ok |
| `s_u` staging writes, `ed_stage_entry<80, true>` | thread = (k, half) | plane stride 80 int4 = 320 words = 0 mod 32: the two halves of 4 consecutive k hit the same 16 banks, 2-way | minor (once per 4 entries); not changed |
| codebook `s_grid`, `Split<22/16>::load<true>` | 4 x 8-byte lookups per (row, k), index = the weight's code | data-dependent random: see below | **conflicted** |
| `expdec_down_kernel<42>` `s_u` | plane stride 20 int4 = 80 words = 16 mod 32 | reads and staging writes conflict-free | ok |

### Bank model and numbers (static analysis, unconfirmed until the counter run below)

gfx1100 LDS as the brief states it: 32 banks of 4 bytes; a 64-bit `ds_load_b64` of a wave32 is served 16 lanes per clock
(128 B/clock), so the two 16-lane sub-warps are two passes; a pass costs as many clocks as the most distinct 8-byte slots
that fall in one bank pair (same slot = broadcast). wave64 would be four 16-lane passes, the same per-sub-warp result
(the lane-to-copy mapping below uses `t = tid & 15`, so it holds for both). Entry i of the plain table sits in bank pair
i mod 16; an IQ2_S index is a 10-bit code (1024 entries, 8 KB table), IQ2_XXS an 8-bit one (256, 2 KB).

Modeled on real rows (`tools/pancake/probes/lds_lut_conflicts.py --gguf <IQ2_XS gguf> --rows 128`: reads
`blk.0.ffn_gate_exps` (IQ2_S) and `blk.1.ffn_gate_exps` (IQ2_XXS), 128 rows of expert 0, and replays the exact index
sequence of `ed_gu_rows` through the bank model): every codebook entry is used, entropy 9.46 bits (IQ2_S) / 7.73 bits
(IQ2_XXS), so the indices are close to uniform and the uniform model (`--uniform`) agrees to within 3%. These are clock
counts of a bank model, not measurements.

| Table layout | LDS per block | IQ2_S clocks per wave lookup | IQ2_XXS clocks per wave lookup |
| --- | ---: | ---: | ---: |
| plain (today) | 8 KB / 2 KB | **5.98** | **5.83** |
| 2 copies | 16 / 4 KB | 5.80 | 5.74 |
| 4 copies | 32 / 8 KB | 5.31 | 5.26 |
| 8 copies | (64 KB) / 16 KB | - | 4.00 |
| 16 copies (one bank pair per lane) | (128 KB) / 32 KB | - | **2.00** |
| ideal (no conflict) | | 2.00 | 2.00 |

So the lookups cost ~3x their conflict-free LDS time. Partial duplication barely helps (balls in bins: the busiest bank
pair of 16/G lanes in 16/G bank pairs, maximized over G groups, stays near the single-group value); only one copy per lane
removes the conflicts. For IQ2_S that is 128 KB (the whole WGP's LDS): impossible. For IQ2_XXS it is 32 KB.

(Other bank models: if gfx1100 LDS in WGP mode behaves as 64 banks, the plain table costs ~4.7 clocks instead of ~6; if a
64-bit read is two dword passes of 32 lanes, ~9.4. The counter run below settles it.)

Is LDS the limit at all? Per wave step (2 rows x 16 k) the kernel reads ~328 B of IQ2_S weights from DRAM (~43 WGP clocks
at 800 GB/s over 42 WGPs) and spends 4 x ~6 = 24 clocks on lookups plus 9 x NC clocks on activations (NC = entries of the
group staged at once, 1-4): about 33 clocks at NC = 1 and 60 at NC = 4. LDS and DRAM are of the same order, so the
conflicts can matter for groups with several entries (verify windows), but the counters must confirm it; if the WGP's LDS
serves two such streams per clock (two CU halves), LDS has 2x headroom and A37 is worth ~0.

### What was coded (opt-in, exact)

`STRATA_IQ_LUT_DUP=16` (or `1`) / `=8` - `expdec_gu_kernel<16, 1, MINB, SF, DUPC>` stages the IQ2_XXS codebook as
`DUPC` copies (`ed_grid_copy_dup_iq2xxs`): slot s of each 16-slot (128-byte) row holds entry `((s >> 4) * B) | (s & (B-1))`
of copy `(s & 15) / B`, B = 16 / DUPC, so copy g occupies only bank pairs [gB, (g+1)B). Lane t's base pointer is offset by
its copy (`s_grid_lane`), and `Split<16>::load<true, SF, DUPB>` reads entry i at `lut_dup_slot<B>(i) = ((i / B) << 4) |
(i & (B - 1))`. With 16 copies a sub-warp's 16 lanes have 16 private bank pairs. The copy writes consecutive slots
(conflict-free) and reads each global entry once per 16 / 8 threads (same address). The values loaded are the same 8
bytes as from the plain table, so W, the products and the sums are bitwise identical (it composes with the F12 sign form).

Cost: LDS per block 13.5 KB -> 43.5 KB (16 copies) / 27.5 KB (8 copies), i.e. 2 / 4 blocks per WGP instead of up to 8
(the kernel's launch bounds target 4); the copy is 16 / 8 LDS stores per thread per block instead of 1 (per block of
32 rows: ~256 LDS clocks of copy against ~1,850 -> ~640 clocks of lookups saved). Only the screen can say whether 2.0 vs
5.8 clocks per lookup beats the lost occupancy; 8 copies keeps the target occupancy, and larger blocks
(`STRATA_EXPDEC_RB=128`, bitwise) amortize the copy (spec arm `lut16-rb128`).

Scope limits (logged): only R = 1 (the default; at `STRATA_EXPDEC_R=2` the log says "ignored at R 2"), not the item-48
SwiGLU kernel, not the default (non-EXPDEC) kernels; with EXPDEC off a one-time note says it has no effect. IQ2_S and
IQ2_XS keep the plain table. API: `NativeExpertExpdecParams::lut_dup` (0, 8, 16; 1 = 16; < 0 keeps).

Files: src/kernels/cuda/iq_kernels.cu (`lut_dup_slot`, `Split<16>::load` DUPB, `ed_grid_copy_dup_iq2xxs`, `ed_gu_rows`
DUPB, `expdec_gu_kernel` DUPC, `iq_lut_dup`, launcher, log line, params), include/strata/kernels/iq_kernels.hpp
(`lut_dup`), src/kernels/iq_multi_parity.cpp (two more EXPDEC runs, 16 and 8 copies, bitwise vs the old kernels, all
formats), src/kernels/expdec_bench.cpp (choice field `l0|l8|l16`), docs/EXPERTDEC.md, bench/specs/24-a37-lut-dup.json.

### IQ2_S (the kernel that matters, 5.8-8.8%): what would work instead

Duplication cannot reach it. Options, in order of cost, for the case the counters show the conflicts matter:
1. `STRATA_EXPDEC_R=2` (exists, bitwise): two rows per lane halve the activation LDS reads per row (the lookups stay), so
   the LDS budget per wave step drops from ~33 to ~28 clocks at NC = 1 and from ~60 to ~42 at NC = 4. Measured only on
   NVIDIA (R = 1 won on the 2080 Ti); it is an arm of the spec.
2. A 2-byte codebook for IQ2_S: the grid bytes are {8, 25, 43}, so an entry is eight 2-bit codes (16 bits); 1024 x 2 B =
   2 KB, and 16 copies fit in 32 KB (copy t in bank pair t; the two sub-warps' lane t share it, so at most 2-way: ~2
   clocks per wave lookup instead of ~6). A lookup becomes a `ds_load_u16` plus ~5 VALU per dword to expand
   (v = 8 + 17c + (c >> 1) per byte on byte-spread codes), ~10-11 VALU per lookup, 4 lookups per (row, k): a large VALU
   increase, so only worth trying if the counters show the LDS pipe saturated. Bitwise (same integers).
3. Remove the redundant activation reads: the two sub-warps of a wave read the same `s_u` words (16 lanes per row);
   a 32-lane row layout would halve them but changes `row_dot_80_sub16`'s sum tree, i.e. loses bitwise parity with the
   default kernel (it would be rounding class).

### Tests run here

- Bank model on real rows and uniform indices: numbers above (`lds_lut_conflicts.py`, python, no GPU); its test
  `python tools/pancake/test_lds_lut_conflicts.py` passed.
- No compile at all: the 09-prefill screen kept the machine busy through four wait_quiet.sh calls (23:26-00:04), so per
  the brief nothing was compiled, not even `-fsyntax-only`. The C++/HIP changes were reviewed by hand (types, template
  defaults, `if constexpr` discards, static LDS 44,288 B < 48 KB CUDA / 64 KB HIP, format strings). First build step for
  the coordinator: `ninja strata_kernels iq_multi_parity expdec_bench hip_gdn_stage_bench` in build-hip.
- `python tools/pancake/test_hit_critical.py` (A36) passed.
- Not run: iq_multi_parity, expdec_bench, the engine (GPU busy, no builds allowed).

### For the coordinator (night run, GPU free)

1. Build; run `ctest -R iq_multi_parity` (must print "bitwise equal" for every gate/up x down pair; the FAIL line names
   "expdec + A37 codebook copies" if the new kernels differ).
2. `build-hip/expdec_bench 300 0 20 16/42,22/42 def,dec:l0,dec:l8,dec:l16,dec:r2` - every cell `identical`; compare
   gu_us of 16/42 across l0/l8/l16 and of both pairs for r2 (expdec_bench is EXCLUDE_FROM_ALL: `ninja expdec_bench`).
3. Bank-conflict counters (gfx11 names from /opt/rocm/core-10.0/share/rocprofiler-sdk/derived_counters.xml:
   `LDSBankConflict = 100 * SQC_LDS_BANK_CONFLICT / SQC_LDS_IDX_ACTIVE`):

       HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib /opt/rocm/core-10.0/bin/rocprofv3 \
         --pmc SQC_LDS_BANK_CONFLICT SQC_LDS_IDX_ACTIVE SQ_INSTS_LDS --kernel-include-regex 'expdec_(gu|down)' \
         --output-format csv -d /tmp/a37prof -o lds -- build-hip/expdec_bench 20 0 20 16/42,22/42 dec,dec:l16,dec:r2

   (if the three counters cannot share a pass, split them: `--pmc SQC_LDS_BANK_CONFLICT SQC_LDS_IDX_ACTIVE --pmc
   SQ_INSTS_LDS`). Read per kernel: conflict share = SQC_LDS_BANK_CONFLICT / SQC_LDS_IDX_ACTIVE. Expected: ~0.3-0.5
   for expdec_gu<22> and <16> (2/3 of the lookup clocks are conflict clocks: ~16 of ~33 LDS clocks per wave step at one
   entry, ~16 of ~60 at four, the rest conflict-free activation reads), ~0 for `expdec_gu_kernel<16,1,4,false,16>`
   and for expdec_down<42>. The same counters on the engine: add `--pmc ... --kernel-include-regex expdec_gu` to
   tools/pancake/gpu_profile.py's rocprofv3 line (one short request). Note: the kernels' demangled names now carry the
   fifth template argument (`expdec_gu_kernel<22, 1, 4, false, 1>`); match them by substring, not by the old name.
4. Screen: `abtest.py bench/specs/24-a37-lut-dup.json --tier screen` (exact arms: hashes must match base).

Decision: keep `STRATA_IQ_LUT_DUP` only if expdec_bench shows a gate/up gain for 16/42 at T = 2-4 and the screen does
not regress; build option 2 above only if the counters show conflict share > 0.5 on expdec_gu<22> AND the LDS pipe busy
(SQC_LDS_IDX_ACTIVE close to the kernel's cycles), since otherwise IQ2_S is DRAM-bound and the conflicts are hidden.

Expected gain: <= 0.5% of decode GPU time from IQ2_XXS (its kernel is 1.7%); the IQ2_S share (up to ~3% of GPU time if
its LDS time is exposed) needs option 1 or 2. On edits the GPU mostly waits for the CPU (see A36), so ~0 there. Risk:
occupancy loss (16 copies), i.e. a slowdown that the screen catches; correctness risk low (same values, parity test).

## A31: GDN prefill chunk 64 + blockwise inverse + more blocks per head (design, not coded)

Papers: Gated DeltaNet (ICLR'25) https://arxiv.org/abs/2412.06464; parallel DeltaNet (WY form, chunked)
https://arxiv.org/abs/2406.06484; Mamba-2/SSD https://arxiv.org/abs/2405.21060.

### Today's HIP path (gfx1100, preset `STRATA_GDN_CHUNKED=1`)

src/prefill/kernels.cu `gdn_rec_wmma`: `gdn_wmma_prep_kernel` (grid HK x T/16, a wave per value head; per 16-token chunk
the WMMA products K K^T and Q K^T, A, T = (I + A)^-1 by per-lane forward substitution in FP32, T and P stored FP16,
1 KB per chunk and value head) and `gdn_wmma_scan_kernel` (grid HV = 48 blocks of 8 waves; per chunk: staging, barrier,
X = K S0 and Q S0 as 8 + 8 chained WMMAs, Y, D = T Y, o = exp(gamma) Q S0 + P D, the state update K^T dd as 8 WMMAs,
barrier, the fused output norm). Geometry S = 128, HK = 16, HV = 48.

Per chunk and wave: 26 WMMAs; per block 208. At ~32 clocks per 16x16x16 WMMA on one SIMD32 and 4 SIMDs per WGP, a head
alone on a WGP needs >= ~1,660 clocks per chunk, ~1.4 ms per GDN layer for a 32K chunk (2,048 chunks) at ~2.5 GHz. But
48 heads on 42 WGPs (84 CUs) puts two heads on 6 WGPs, so if the scan is WMMA-bound its floor is ~2.7 ms per layer at 32K
(estimate, static).

### Design of the chunk-64 variant (`STRATA_GDN_CHUNK64=1`, to be written after the timing)

- Chunk of 64 tokens = 4 sub-tiles of 16. gamma is the running gate sum over the 64 tokens (exponents still <= 0).
- Prep (fused, one block per key head and 64-chunk, 3 waves): the 4 x 4 lower blocks of K K^T and Q K^T by WMMA
  (10 + 10 products), A blocks, then the blockwise inverse: T_bb = (I + A_bb)^-1 by the existing per-lane forward
  substitution (FP32), off-diagonal T_ba = -T_bb * sum_{a <= c < b} A_bc T_ca in FP32 (or FP16 WMMA: rounding class). Store
  only the 10 lower blocks of T and of P (FP16): 10 KB per 64-chunk and value head, T x 7.5 KB in all, which fits the
  FP32 scratch `y` (T x 24 KB) that holds T and P today.
- Scan: per 64-chunk, X_b = K_b S0 and Q_b S0 for the 4 sub-tiles as 4 independent accumulator chains (interleaved: the
  WMMA latency of one chain hides behind the others), Y_b, D = T Y as 10 block products (chains of 1-4), o_b = exp(gamma)
  Q_b S0 + sum_{a <= b} P_ba D_a (10 products), the state update S = exp(gamma_last) S0 + sum_b K_b^T dd_b (8 x 4 products,
  8 independent chains). 116 WMMAs per 64 tokens vs 104 for four 16-chunks (+12%), but 2 barriers instead of 8 and a
  dependent chain of ~20 WMMA latencies instead of ~44.
- LDS per block (estimate): k, q (FP16, 64 x 136: 17 KB each), k^T (128 x 64 FP16: 16 KB), v and o (FP32 64 x 128:
  32 KB each), T and P blocks (10 KB): ~125 KB - above the 64 KB a block may take. Plan: keep v and o in registers (each
  wave owns 16 columns: 64 x 16 floats = 32 VGPRs per lane for each), LDS then ~61 KB. Register risk: state 64 VGPRs +
  v 32 + o 32 + four X / QS accumulators 64 + operands: close to the 256-VGPR wave32 limit, i.e. one block per WGP-half;
  the alternative is staging 32 tokens at a time (two halves per 64-chunk), which gives back part of the barrier saving.
- More blocks per head: a column split (2 blocks of 4 waves per head, 96 blocks) would cut the busiest WGP's load from
  2 heads to 1.5 (-25% if WMMA-bound), but it breaks the fused norm, which needs sum(o^2) over all 128 columns of a
  token. Writing o pre-norm costs T x 24 KB of FP32 traffic per layer (~2 ms at 32K: more than the scan); exchanging
  per-chunk partial sums between the two blocks needs co-resident blocks and a global flag (deadlock risk if the
  dispatcher does not co-schedule them). So the split is a second step, only if the timing says WMMA-bound (where
  chunk 64 cannot help) and the 25% is worth that risk.
- Parity: the existing tests/hip/gdn_chunked.cpp (FP16-class bounds of its Error::ok against the FP32 kernels) with the
  switch on; plus a chunk-64 vs chunk-16 comparison at T = 64, 1000, 4096 (unaligned tails: rows past T get gate 0 and
  beta 0 as today).

### Why it is not coded tonight, and the measurement that decides it

If the scan runs near its WMMA floor (~2.7 ms per layer at 32K on the 7900 XT), chunk 64 is ~12% MORE WMMA work and a
loss; it pays only if the scan is latency- or barrier-bound (the V100 chunk-64 attempt was 4-6x slower,
bench/results/2026-10-02-gdn-chunk). The kernel is ~300 lines of lane-layout code that could not be compiled for a GPU or
run here, so writing it blind before this number exists was the wrong trade. The timing needs almost no new code (item
82's instrumentation exists); the one addition is a printed floor in the stage bench:

1. Per-layer recurrence time: `HIP_VISIBLE_DEVICES=0 STRATA_GDN_CHUNKED=1 build-hip/hip_gdn_stage_bench 4096 16384 32768`
   (and once with `STRATA_GDN_CHUNKED=0`). New (tests/hip/gdn_stage_bench.cpp): on gfx11 it prints
   `A31 T=... wmma_floor_ms=... rec_out_over_floor=...` after each STAGE line (the floor above, from the device's clock
   and CU count).
2. Share of cold prefill: the server with `STRATA_PREFILL_TIMING=1`, cold prompts of 4K / 16K / 32K; the "gdn
   recurrence" phase of the `strata prefill timing:` line (PfTimer, src/prefill/prefill.cpp).
3. Optional: `rocprofv3 --kernel-trace --stats` over one cold 32K prompt; gdn_wmma_scan_kernel + gdn_wmma_prep_kernel.

Decision rule: build A31 (chunk 64) only if (a) `rec_out_over_floor` at 32K is >= 2 (latency/barrier-bound) AND (b)
"gdn recurrence" is >= 6% of the cold-prefill GPU timeline at 16-32K (then halving it gives >= 3%). If (a) fails but
(b) holds, the scan is WMMA-bound: consider only the column split above. Otherwise close A31 (the 5-11% estimate
assumed a recurrence share like the RDNA2-era sequential one, 4.3% at 8.9K on a 6900 XT,
bench/results/2026-10-05-rdna2-rocblas-table/timing-on.txt).

Spec: none (nothing to A/B until the variant exists); the stage bench line and the PREFILL_TIMING run are the screen.

## A36: weight-major layout for the VRAM-hit experts / Q2_0 down (MonoMoE 2026)

Paper: MonoMoE 2026, https://arxiv.org/abs/2609.04244.

Code + profiles: per MoE layer the GPU publishes the routing (doorbell), runs the hit experts (expdec_gu, SwiGLU, down),
then spins in `wait_flag_ge_kernel` / `combine_cpu_*` until the CPU rows land. With deferral (2026-11-09-gpuprof-defer)
the waits are 31% (wait_flag_ge) + 7% (wait_flag_copy) of kernel time against 16.6% for the hit kernels; without it
combine_cpu (which spins) is 38%. In every layer where the GPU still waits after its hits, a faster hit kernel only
lengthens the wait. The aggregate stats cannot tell how many layers have no wait (combine_cpu_k10_vec4 in the fast-hip
stats: 4,128 calls, mean 254 us, min 5.6 us, sd 363 us: some layers sit at the floor, the share is unknown), so the
rule is per layer:

`tools/pancake/probes/hit_critical.py TRACE.csv` (new; trace from `tools/pancake/gpu_profile.py CONFIG OUT --case short`
and `--case edit`): segments the decode trace per layer (routing kernel to routing kernel), sums hit-kernel time H and
wait time W, calls a layer GPU-bound when W <= the wait kernel's 5th-percentile duration + 2 us, and reports the exposed
hit time (H over GPU-bound layers, with the down's part) and the ceiling of an X% faster hit kernel (default 25%) as a
share of the decode wall. Test: `python tools/pancake/test_hit_critical.py` (3 synthetic cases) - passed here.

Decision rule: write A36's weight-major kernels only if the ceiling at 25% is >= 2% of the decode wall on both short and
edit traces; else keep it parked and re-run the tool after changes that move CPU time (deferral, k-th-miss admission,
swaps). Expected from the existing stats: short chats may expose some hit time (the GPU floor sets the short window),
edits ~none. A36 itself would be M-L work (new layout at load time + kernels), so it was not started.

## What is left

- A37: build + parity + counters + screen (above); IQ2_S options 1-3 depending on the counters.
- A31: the two timings, then build or close by the rule.
- A36: two traces through hit_critical.py, then build or keep parked.
