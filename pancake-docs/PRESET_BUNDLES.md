# Preset bundles and the interaction table (MOD_PLAN 17.6)

2026-11-12. Rule (17.6): ship **bundles**, not switches; every switch stays default-off until a paired test on a quiet machine
promotes it. This page defines the three bundles from what is in the repo today and lists what substitutes for what. Labels:
**[meas.]** measured on this box; **[model]** simulator / analysis; **[code]** read from code or the engine's env table;
**[pending]** will be filled by the night queue (`NIGHT_QUEUE_RUNNER.md`, `bench/results/night-queue/RESULTS.md`).
Nothing below is a new measurement.

## 1. The three bundles (IQ2_XS; the Q2_0 files are the same with `flashnext-q2_0-*-adopted.json`)

| Bundle | Definition | Config | Status |
| --- | --- | --- | --- |
| **fast** (the daily preset) | the 53 env switches + args of `flashnext-iq2_xs-fast-hip.json` as they are | `configs/pancake/flashnext-iq2_xs-fast-hip.json` | in use; measured 2026-11-11 vs Avalon [meas.] |
| **lossy** (turbo) | `fast` + `STRATA_ROUTE_PRIOR=0.5`, nothing else (the diff of the two files is exactly that one line) | `configs/pancake/flashnext-iq2_xs-turbo-hip.json` | turbo -4..-8% on 2026-11-11, not a default [meas.]; turbo +9% on code edits, -15..-25% on prose (MOD_PLAN 17.1) |
| **exact** | `fast` with the 12 switches below left unset / 0, so the output does not move except for noise | not written as a file yet | [pending]: whether unsetting each one is enough (some defaults differ per GPU family) has not been tried |

Switches of the daily preset that the engine's own table marks as not bit-identical (`src/core/conversation_disk_env.def`) [code];
these are what separates `exact` from `fast`:

| Switch (value in `fast`) | Engine table says |
| --- | --- |
| `STRATA_GDN_CONVL2=1` | conv + L2 norm kernel: not bitwise on CUDA sm_120 |
| `STRATA_GR_RDNA_A8=1` | int8 activations + dot4 in the Q8 read (lossy) |
| `STRATA_GR_RDNA_Q8=1` | int8 HC read on RDNA (lossy) |
| `STRATA_HC_UPMIX=1` | HC up projection + mix in one kernel (bit-changing, KL gated) |
| `STRATA_PA_FAST=1` | single FP16 q and p prompt attention (rounding level) |
| `STRATA_PF_F16=2` | prompt experts on F16 matrix-core kernels (rounding differs from int8) |
| `STRATA_PF_HCDOWN=1` | HC down by WMMA kernel (first-token KL doubled at 16K) |
| `STRATA_PREFILL_AUTO_MAX=32768` | chunk geometry changes the rounding |
| `STRATA_PREFILL_CPU_SHARE=0.6` | prompt experts on the CPU pool (mean KL ~0.004-0.011) |
| `STRATA_ROUTE_TAIL_SKIP=0` | already 0 (the CUDA serve default of 7 changes the output) |
| `STRATA_SELECT_WMMA=1` | matrix-core block scorer (not bitwise, cutoff cells flip) |
| `STRATA_VERIFY_DEFER_CPU=1` | CPU rows deferred a window: lossy |

Not yet in any bundle (all default-off, coded on `dev/p17/integration`): `STRATA_ROUTE_PRIOR_AUTO`, `STRATA_VERIFY_DEFER_AUTO`,
`STRATA_KV_RING_ADAPT`, `STRATA_PARK_REWIND_MAX`, `STRATA_PARK_DIVERGENCE_LOG`, `STRATA_PREFILL_EQUAL_TAIL`
(+ `_MIN`), `STRATA_ADAPT_KMISS`. A switch is promoted into `fast` only after its paired night-queue arm.

## 2. Interaction table: what substitutes for what

| A | B | Relation | Evidence |
| --- | --- | --- | --- |
| `ROUTE_PRIOR` (turbo) | `VERIFY_DEFER_CPU` (deferral) | substitutes for the same time; must be measured together (MOD_PLAN 13) | turbo -4..-8% with deferral on [meas.]; paired arm in group `1-defer` [pending] |
| `VERIFY_DEFER_CPU` | six fused paths | deferral turns six fused paths off for the whole process | [code], MOD_PLAN 17.1 finding |
| `VERIFY_DEFER_CPU` | model file | off = +7-8% on Q2_0 (reproduced), -3% on IQ2_XS (overlapped with agent work): **unsettled** | [meas.], retest on a quiet machine in `1-defer` [pending] |
| `ROUTE_PRIOR_AUTO` | `ROUTE_PRIOR` | AUTO applies the prior only in lookup windows; the fixed prior applies it always; never both | [code] unit test; speed [pending] |
| `VERIFY_DEFER_AUTO` | `VERIFY_DEFER_CPU` | AUTO defers only in lookup windows | [code] `defer_auto_test`; speed [pending] |
| `STRATA_ADAPT_KMISS` | `--adapt-async` | kmiss needs `--adapt-async 1`: the blocking tier stalls the GPU ~5.6 ms per swap round | [meas.] profile; kmiss+async probe never finished [pending] |
| `KV_RING_ADAPT` | `--max-context` | under `--serve` the rule uses `--max-context`, not the request depth: with 131072 the ring is the cap for every request (arm R behaves like static 49152) | [code], NIGHT_QUEUE_RUNNER.md; speed [pending] |
| `PREFILL_EQUAL_TAIL` | `PREFILL_EQUAL` | EQUAL_TAIL acts only on a staged short last chunk and takes precedence there; EQUAL evens out otherwise | [code] unit test; speed [pending] (100K trace run in `2-ttft`) |
| `PREFILL_CPU_SHARE` | prompt length | the CPU share can only take experts that <= 8 tokens route to: it moves warm turns of up to ~400 new tokens, almost nothing above ~700 | [est.] PREFILL_LEVERS.md section 5 |
| `PARK_REWIND_MAX` | conversation cache | skips the park when resuming from the live conversation's own checkpoint; helps only if cache admission is the miss | [code] `conv_cache_test`; TTFT effect [pending] (group `3-lever`) |
| draft vocabulary (`--mtp-draft-vocab`) | decode speed / acceptance | no effect (closed, 17.8) | [meas.] 2026-11-11 |
| `--spec 2` | `--spec 4` | spec 2 is slower on both engines | [meas.] 2026-11-11 |

## 3. What happens next

1. The night queue (group order `1-defer`, `2-ttft`, `3-lever`, `4-kvring`, `5-prose`) fills the **[pending]** cells in
   `bench/results/night-queue/RESULTS.md`; this table is then updated with the numbers and a verdict per switch.
2. A switch with a paired win and a quality check moves into `fast`; one with a loss is closed; `exact` is written as a config only
   after its 12 unsets are checked on the engine.
3. 17.3 (greedy ablation over the coded-but-untested items) runs alone afterwards.


## 4. Results and decisions, 2026-11-12 (owner decisions; see NIGHT_RESULTS_2026-11-12.md)
- **fast** now has expert deferral OFF (IQ2_XS and Q2_0, owner decision). The line `STRATA_VERIFY_DEFER_CPU` was removed from both daily presets (`configs/pancake/flashnext-iq2_xs-fast-hip.json`, `flashnext-q2_0-fast-hip-adopted.json`).
- Deferral on vs off, measured: IQ2_XS on 105.7 / off 101.4-107.5 (no difference); Q2_0 on 91.2 (one run) / off 107-113 (about 19% faster off).
- Lever runs (17.0): none beats the control; cache-admission levers are slower (24.8-28.5 s session TTFT).
- KV ring (17.4): no measurable effect at 1K-120K.
- Prose gap to Avalon (17.2): Avalon 3.4-8.4% faster on prose, not established. Controls (single starts): Avalon tail-skip off 97.3, Avalon IQ3_KS main 90.7, Pancake repeat 95.0. Full set (12 starts per arm, A B B A) queued after tonight's runs.
- **exact** bundle: still unbuilt. The exact-vs-exact control (A-A) is needed first because greedy text varies between starts of the same config.
- 17.3 ablation: queued after the full prose set (round 1 is 69 engine starts, roughly 6-7 hours).
