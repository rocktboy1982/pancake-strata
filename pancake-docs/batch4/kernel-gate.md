# kernel-gate: go / no-go for kernel items before an end-to-end screen (MOD_PLAN project feature 9) - 2026-11-11

Agent `kernel-gate`, branch `dev/p16b/kernel-gate` (from `dev/phase16-academic` 0dd7ef2e8). New files only:
`tools/pancake/kernel_gate.py`, `tools/pancake/test_kernel_gate.py`, this note. Python only; no benchmark, no GPU, no model was run.
No engine switch, no existing file edited.

## What it does

Evidence: the F21 / F9 / A33 end-to-end tests cost about 3 hours for a +-1% answer (paired SD of one A/B pair 3.9%), while a kernel
microbench takes minutes. `kernel_gate.py` turns a kernel item's kernel-level number plus where the kernel sits in the decode window into
a verdict, the expected end-to-end gain on the short and the edit window, and the engine starts an A/B would need.

    python tools/pancake/kernel_gate.py table                      # every item decided so far, from the notes + the 11-09 profile
    python tools/pancake/kernel_gate.py item F7 --q2rows OUT.txt   # one item with your measurement (see below)

Inputs it reads (the formats were taken from the sources, not from memory): `expdec_bench` cell lines (`dev T shape gu/d choice all_us
gu_us down_us GB/s pct identical|DIFFERENT`, expdec_bench.cpp:341; a DIFFERENT cell is refused), `q2_rows_bench` output (arm lines, the
`-- N threads` pool block, the bitwise-equal flag; a non-equal kernel is refused), gpu_profile.py `summary.txt` (top 20 kernels) or a
rocprofv3 `--stats` csv (kernel ms and calls per window; windows = wait_flag_copy_i32 calls / 48), `hip_gdn_stage_bench` STAGE / A31
lines (parser reused from `probes/gdn_chunk64_rule.py`), or a saving worked out by hand (`--saved-ms`, `--saved-frac`).

Verdicts (bar 2% end to end, MOD_PLAN 16 row 8, the same bar as `hit_critical.py`):
- GO: a **measured** kernel saving gives >= 2% on a window.
- NO-GO: even the optimistic bound is < 2% (nothing the missing number could be rescues it), or the measured saving is.
- MEASURE: only an estimate / bound reaches 2%; the "missing" column says which kernel number to take first (minutes), and the item
  output gives the break-even saving (ms per window, or as a % of the item's kernels).

## The lane model (lead's request: GPU speedup under a CPU-critical path buys ~0)

Decode items go through `time_ceiling.window_ms2` (the A6 engine chain) at the fast-hip preset (deferral + FUSE_PHASES). A kernel is on
one lane: `router` / `mixer` (GPU work before the doorbell), `hit` (VRAM expert kernels, hidden behind the CPU rows), `cpu` (the CPU
pool). The model says how much of the saving lands in the window (probe: 1 ms on each lane, 200 synthetic windows):

| Lane | edit window | short window |
| --- | ---: | ---: |
| router (F9) | 100% lands | 100% |
| mixer | 79% | 99% |
| hit (expdec, down, SwiGLU gap, plan copy) | 38% | 91% |
| cpu (22% of the CPU experts' time) | 2.3 ms = +7.1% | 0.11 ms = +0.5% |

Removing **all** the VRAM-hit GPU work (E 4.3 -> 0 ms) buys only 0.67 ms = +1.99% on an edit window: no hit-lane kernel item can reach
the bar on edits (this agrees with the A36 note). Prefill items (A31, BFLA) are Amdahl on the GPU timeline.

Calibration: the A6 `--validate` parameters (pre 15.75, mu 0.65, E 4.3, 48 x 40 us barriers, c 44.6 us per missed expert, alpha 0.5,
sdef 91 us). A6 needs the per-layer miss rows of a routing trace; the trace is not in the repo, so the per-layer misses are lognormal around
the window's mean with **one fitted number** (sigma 1.02) so that deferral on the edit window gives the A6 33.76 ms. Not fitted, and
reproduced: serial 39.2 ms (A6 39.26), deferral + FUSE_PHASES 33.4 ms (A6 33.37). Percentages are on the measured preset window: edit
34.42 ms (= 33.37 + the A6 bench constant 1.05 = 4.70 tok/window at 136.5 tok/s, the measured 135-138), short 24.8 ms (the plan's).

## Table of the items decided so far (`kernel_gate.py table`, 11-09 deferral profile, 120 windows, bar 2%)

| Item | Lane | Saving point / bound | Short % | Edit % | Starts 95% (80%) | Verdict | Missing number |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| F7 Q2_FAST | cpu | - / 22% of CPU-expert time | <=0.5 | <=7.1 | 8 (10) | MEASURE | `q2_rows_bench --threads 8 --nt 1` and `--nt 2` on an idle box (A48): the pool ns/row; break-even 6% of the CPU time |
| F12 IQ sign | hit | - / 1.21 ms (35% of gate/up 3.45 ms) | <=4.6 | <=1.3 | 12 (16) | MEASURE | `expdec_bench ... 16/42,22/42 def,dec:s0,dec:s1` gu_us; break-even 15% of gate/up; short window only |
| F21 COPY256 | hit | 0.19 ms (4 us x 48) / 0.26 ms | 0.7 (1.0) | 0.2 (0.3) | 230 (464) | **NO-GO** | none needed: the whole 5.44 us copy cannot reach the bar; (MinNs of wait_flag_copy_i32 with the switch, if wanted) |
| F9 ROUTE_FAST | router | 0.58 ms (12 us x 48) / 0.66 ms | 2.4 (2.7) | 1.7 (1.9) | 26 (46) | MEASURE | `gpu_profile.py --env STRATA_ROUTE_FAST=1`: route_publish_fast us vs route_multi 9.7 + doorbell_publish 6.9 + 6.8 gap; break-even 0.49 ms short / 0.67 ms edit |
| A33 SWIGLU | hit | 0.31 ms (gap 9.2 us x 34) / 0.63 ms | 1.2 (2.4) | 0.4 (0.7) | 90 (176) | MEASURE (NO-GO if the short window is 32.6 ms) | expdec_bench `all_us` or gpu_profile with SWIGLU on vs off; e2e so far short +0.6 +-1.8, edit +1.1 +-4.7 (n=6) |
| A37 LUT dup | hit | - / 0.25 ms (45% of gu<16>, 0.56 ms) | <=0.9 | <=0.3 | 136 (272) | **NO-GO** | none: break-even is 95% of the kernel. (expdec_bench l0/l16 only to justify reverting d11a9ca09) |
| flat loads | hit | - / 1.36 ms (25% of hit kernels 5.44 ms) | <=5.2 | <=1.4 | 10 (14) | MEASURE (short only) | expdec_bench gu_us + down_us of a `-DSTRATA_HIP_GLOBAL_WEIGHTS=1` build vs default (both `identical`); break-even 10% of the hit kernels |
| A36 (extra row) | hit | - / 1.36 ms | <=5.2 | <=1.4 | 10 (14) | MEASURE (short only) | `hit_critical.py` on `gpu_profile.py` traces with deferral on; same lane and kernels as flat loads |
| A31 GDN chunk 64 | prefill | - / 3% of the prompt (6% share x halving) | - | - | 18 (30) | MEASURE | `hip_gdn_stage_bench 4096 16384 32768` + the 32K prompt speed; break-even rec_out = 2% of the prompt |
| BFLA dense | prefill | 50% of the 128K prompt (386.9 -> 771.2 tok/s, measured) | - | - | 4 (4) | **GO** | 64K off baseline only; (the sparse part is lossy and not gated here) |

Reading:
- **No kernel item has a kernel-level measurement yet** (only BFLA has an end-to-end number): every "point" except BFLA is a static
  estimate from the notes. What the gate says about each missing number is in the last column; each is minutes, not hours.
- F21 and A37 are closed without a screen. F9, A33 and F12 live or die on the **short** window, whose model is not validated (A6: no
  short routing trace).
- **Resolving +-1% end to end is out of reach**. Engine starts at SD 3.9% (95% interval excludes 0 / 80% power): 1% needs 122 / 244 (the
  95% figure is ~4 h of starts at 2.1 min each), 1.5% 58 / 110, 2% 36 / 64, 3% 18 / 32, 5% 10 / 14. Above 60 starts the tool says "an end-to-end screen cannot decide this" (F21, A33, A37). Exact items
  that pass the kernel number and the bitwise parity test can ship on that evidence; lossy ones still need the gate.
- BFLA: the kernel model (283 s -> 96 s of a 338 s prompt) would say +124%, the measured +99%: 19 s of non-attention time.

## Flags: numbers that are missing, inconsistent or assumed

1. **Short window**: the plan's 24.8 ms (profile with 2.29 tokens per window at 92 tok/s with timers on; GPU 15.9, CPU 5.2, draft 2.2)
   against kern-hip's 32.6 ms (4.3 tokens per window at 132 tok/s, deferral profile). The tool takes 24.8 (the brief's) and `--short-ms 32.6`
   switches. With 32.6: A33 flips to NO-GO (bound 1.8%), F9 stays MEASURE (bound 2.1% short / 1.9% edit), F12 3.4%, flat loads 3.9%.
   The model itself is not validated for short windows (A6), so every short percentage is an order of magnitude.
2. **Paired SD 3.9%**: could not be reproduced from the committed jsonl. Of the 62 attempts only `25-kern-hip-confirm.jsonl` has >= 4
   pairs (6 reps, abtest's kept attempts): SD of a pair 1.7% (expdec-swiglu short), 4.5% (expdec-swiglu edit), 5.9% (all3 short), 5.3%
   (all3 edit), median **4.9%**; the SD of the 42 clean short requests of all five files is 4.1%. `table` prints the recomputed figure. Starts scale with SD squared: at 4.9% every count above is 1.6x (2% needs 52 / 98 starts
   instead of 36 / 64; 1% 190 / 382). The prefill SD is not measured at all (no `long*` attempt in the 62); 3.9% is assumed there.
3. **F7 bound**: kern-cpu says "at most ~22% of CPU-expert time" (27 -> 33 GB/s is +22% bandwidth = 18% less time). 22% is used; at 18.2%
   the edit gain is 5.9% instead of 7.1%, verdict unchanged. The single-thread numbers (x1.25 gate/up, x1.47 down: 224 -> 170 us per expert,
   reproduced by the parser test) are not the pool; the pool number is the open A48 run.
4. **A33**: the bound adds the whole SwiGLU kernel, which is not in the top-20 table (it is below the 20th row, 38.0 ms / 120 = 0.32 ms per
   window). Whether `STRATA_EXPDEC_SWIGLU` is already active in the preset (research_notes batch6 says fast-hip runs `expdec_gu_swiglu_kernel`;
   the 11-09 profile still shows the separate SwiGLU kernel) was not checked.
5. **Profile**: `summary.txt` lists only the top 20 kernels, so a kernel outside it raises `KeyError` (use a rocprofv3 `--stats` csv; the csv
   covers the whole run, not the decode phase). The down kernel regex matches `native_down_multi_kernel<42` and `expdec_down_kernel<42`
   (the preset's EXPDEC_DOWN renamed it after the profile).
6. **Hit-lane items**: where a per-dispatch trace exists, `hit_critical.py` (per-layer exposure) supersedes the model's exposure.

## Tests

| Command | Result |
| --- | --- |
| `python -m pytest tools/pancake/test_kernel_gate.py -q` | 24 passed (about 8 s) |
| `python tools/pancake/test_kernel_gate.py` | all ok (same tests, plain runner) |
| `python tools/pancake/test_hit_critical.py`, `test_gdn_chunk64_rule.py` (modules reused) | pass, unchanged |
| `python tools/pancake/kernel_gate.py table` / `item F21` / `item A31 --stage ... --prefill-tok-s 2580` | run, output as above |

What the tests pin: the profile fixture (copied from `bench/results/2026-11-09-gpuprof-defer/summary.txt`: 120 windows, 34 / 11 / 48 calls per
window of gu<22> / gu<16> / down, 2.89 / 0.56 / 1.99 ms), a rocprofv3 csv with the five-argument A37 kernel name, `q2_rows_bench` text with the
kern-cpu numbers, the 6-pair `25-kern-hip-confirm` A33 arm (abtest's kept-attempt rule and summary line: short +0.6 +-1.8, edit +1.1 +-4.7), the A6
windows (33.76 fitted, 39.26 and 33.37 reproduced), lane exposures, t quantiles and pair counts (2% at SD 3.9% = 18 pairs), every verdict rule,
the table's ten verdicts and their note-derived numbers, evidence-driven GO / NO-GO for F7, F12, A37, flat loads and A31, the refusal of a
DIFFERENT cell or a non-bitwise-equal Q2 kernel, and the CLI (`--short-ms`, `--saved-ms`, wrong evidence flag). **Synthetic fixtures**: the
`expdec_bench` microseconds and the 8-thread block of `q2_rows_bench` (the notes have neither; the line formats are exact) and the A31 stage
numbers.

## Risk and next

- The verdict table is only as good as the notes' static estimates; the point of the tool is that the next GPU session fills the "missing"
  column in minutes (`expdec_bench`, `q2_rows_bench --threads 8`, two `gpu_profile.py` tables) and re-runs `item` before any screen.
- The lane model for short windows needs a short-chat routing trace (corpus recorder) and one deferral-on `gpu_profile.py --case short` to validate;
  until then read short percentages as an order of magnitude. A per-layer miss matrix from a real trace would replace the fitted dispersion
  (`LaneModel._misses`).
- Not done: no `--json` output, no block-mode start counts (block mode would divide the starts by roughly the reps per start), no
  CUDA-side (RTX 3060) items (F14 / F17 / F18 / F22).
