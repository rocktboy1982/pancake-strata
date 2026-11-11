# presets-specs: presets, spec bases, the stack spec (Phase 16 round 2, 2026-11-11)

Branch `dev/p16b/presets-specs` (from `dev/phase16-academic` 0dd7ef2e8). JSON, one shell script and this note. No GPU, no
engine, no model: the checks are `spec_lint.py`, pytest under `tools/pancake` and reading the engine source.

## Changes

| What | Files |
| --- | --- |
| Q2_0 presets with the four adopted items (`STRATA_VERIFY_DEFER_CPU`, `STRATA_POOL_FUSE_PHASES`, `STRATA_EXPDEC_DOWN`, `--lookup-max-t 12`) as **separate** files; the existing q2_0 presets are unchanged. After a name swap (`iq2_xs` -> `q2_0`, `IQ2_XS` -> `Q2_0`) each equals its iq2_xs twin byte for byte (`diff` clean) | `configs/pancake/flashnext-q2_0-fast-hip-adopted.json`, `flashnext-q2_0-turbo-hip-adopted.json` |
| `15-q2_0` base -> `pancake-flashnext-iq2_xs-fast-hip.json`; note says which preset each arm equals | `bench/specs/15-q2_0.json` |
| Six specs named the **template** `configs/pancake/flashnext-iq2_xs-fast-hip.json` as base: nothing expands `<REPO>` / `<STRATA_DATA>` in it (only `restore_check.py` does), so the engine would start `<REPO>/build-hip/strata` with `cwd <REPO>`. Base -> the generated `pancake-flashnext-iq2_xs-fast-hip.json` | `19-turbo-guards`, `taskgate-a19-probe`, `-caps`, `-effort`, `-lossy`, `-rumination` |
| `"env": {}` added to the four poly / qwen36 templates, so `abtest.make_config` no longer raises `KeyError: 'env'` (checked: every arm of `18-f11-attn-window` and `18-f17-poly-fattn-3060` builds a config; the server reads `cfg.get("env") or {}`, so `{}` changes nothing for `run-pancake.sh`) | `configs/pancake/qwen36-ud-q4_k_xl.json`, `-q4_k_xl-q36.json`, `-q3_k_m.json`, `qwen38-35b-distill-q4_k_m.json` |
| `25-f-env-rerun`: note now opens with "ARENA TIER ONLY for the lookahead0 arm" and says to use the sibling on the mmap tier. Arms unchanged | `bench/specs/25-f-env-rerun.json` |
| New sibling: the same arms without `lookahead0`, for the mmap tier (`queue_run.sh` default `STRATA_RAM_KEEP_GIB=20`) | `bench/specs/25-f-env-rerun-mmap.json` |
| New stack spec (below) | `bench/specs/26-stack-small-exact.json` |
| `tools/pancake/gen_bases.sh` (new): writes every `pancake-*.json` base the specs name | `tools/pancake/gen_bases.sh` |
| `24-a44-bandit` note said the preset has no deferral: it has, since adoption. Text only | `bench/specs/24-a44-bandit.json` |

## How the base files are made (and why 15-q2_0 looked wrong)

A spec's `"base": "pancake-X.json"` is a **generated, git-ignored** file in the repo root (`/pancake-*.json`); abtest, quality_gate,
task_gate and replay_turns read `ROOT/base`. Who writes it:

- `run-pancake.sh`: `sed` of `configs/pancake/$PRESET.json` (`<STRATA_DATA>`, `<REPO>`, `<MODELS>`, and `/build-cuda/` -> `/$BUILD/`;
  on a box with no NVIDIA card `BUILD=build-hip` and `$PRESET-hip` is taken where it exists) into `pancake-$PRESET.json`.
- `test_phase.sh`: the same `sed` from `flashnext-iq2_xs-fast-hip.json` into `pancake-flashnext-iq2_xs-rdna3test.json`, and nothing else.
  It does this on every run, so after a `test_phase.sh` the rdna3test base **is** fast-hip (the root files in Strata-p16 are byte
  identical).
- `queue_run.sh`: writes nothing. A queue run uses whatever root files exist.

So the real 15-q2_0 runs used a file with the four items when `test_phase.sh` had regenerated it. What was wrong: (1) `spec_lint`
(and `selftune.py`) fall back to `configs/pancake/<name>.json` when the root file is missing, and the template called
`flashnext-iq2_xs-rdna3test.json` is an **older, different preset** (50 env keys, none of the four items, no `--lookup-max-t`), so
lint and anyone reading the spec saw a base without them and `q2_0-stock` equal to `q2_0`; (2) a queue run on a stale or missing root
file would silently measure the wrong base. The base now names the file `run-pancake.sh` writes, and
`gen_bases.sh` regenerates it. Arm check (merged config against the sed-expanded presets): arm `q2_0` = `flashnext-q2_0-fast-hip-adopted`,
arm `q2_0-stock` = `flashnext-q2_0-fast-hip`, differing only in `model_name` and `tokenizer` strings (the two tokenizer directories
hash identical). There is no `15c` spec file; 15c is the redo of `15-q2_0`.

`tools/pancake/gen_bases.sh [SPEC ...]`: reads the `base` of the given specs (default all), and for each `pancake-X.json` writes it
from `configs/pancake/X.json` with the `run-pancake.sh` substitutions. `rdna3test` is written from fast-hip (what `test_phase.sh` does).
Without an NVIDIA card a template that runs `build-cuda` (the 3060 presets and `strata-poly`) is skipped, because `run-pancake.sh`
never writes it there. On this box it writes `pancake-flashnext-iq2_xs-fast-hip.json`, `-rdna3test.json` and
`pancake-qwen36-ud-q4_k_xl-q36.json` and skips `-fast.json` (3060 specs) and `-qwen36-ud-q4_k_xl.json` (CUDA only). Run it before a queue.

## spec_lint

Before: 70 specs, **2 errors, 9 warnings**. After: 72 specs, **0 errors, 8 warnings** (presets: 0 errors, every `configs/pancake/*.json`
through `load_base` + `check_base`).

Fixed: the two `KeyError 'env'` errors; the `15-q2_0` identical-arm warning. Left, none is a spec problem:

| Spec | Warning | Why left |
| --- | --- | --- |
| 08-build | `host-native` exe `build-hip-native/strata` not built | runtime: build it first |
| 11-conv-cache, 24-a15-park-ttl, 24-a24-suffix-index, 25-a11-a13-cache, 25-f1-park-truncate | `pancake-turns.jsonl` not here | the turn log is a run output |
| 18-batch1-exact | `port` exe in the main checkout | by design: "the daily binary (sanity)" |
| 24-a44-bandit | `defer` changes nothing against the base | by design: arm 0 is the reference; it equals the base now that deferral is in the preset |

Found while doing this, not fixed (not mine or a decision):

- **The lint is blind to the rdna3test alias.** With `gen_bases.sh` run, lint resolves the real base and warns, for 8 arms of 7 closed specs, that
  the arm changes nothing: `01-deferral defer`, `03-cpu-pool fuse-phases` and `fuse+tail4`, `04-gpu-launches` and `04c-gpu-retest expdec-down`,
  `05-small-flags`, `05c-lookup12` and `05s-screen lookup12`. They measured the adoption of items that are in the preset now. Do not re-run
  them (a bare `test_phase.sh` runs every spec). 26 specs still name the rdna3test base; the live ones are `20-verify-ecap`, `22-f12-sign-fast`,
  `25-a4-w-sweep` (and the done `10-rank-guard-fold`). Switching them to `pancake-flashnext-iq2_xs-fast-hip.json` is safe (identical file
  after `test_phase.sh`) and removes the alias; the lead's call.
- `tools/pancake/test_task_gate_effort.py::test_the_new_specs_are_well_formed` asserts `ROOT/spec["base"]` exists. With the six
  specs on the generated base it passes only where `gen_bases.sh` (or `run-pancake.sh`) has run (it passes in Strata-p16); in a clean
  checkout it fails. Conversely `test_spec_lint.py::test_base_config_problems` fails where the root file exists (already the case in
  Strata-p16). The clean fix is in `abtest.make_config` (harness-core): expand `<REPO>`, `<STRATA_DATA>`, `<MODELS>` in a base that is
  a template, then specs can name `configs/pancake/*.json` and neither test depends on a generated file.
- The `q2fast` arms of `18-batch1-exact` and `23-batch1-confirm` were **not** inert (see below), the `fusewait-copy256` arms were (kern-hip).

## 26-stack-small-exact (stack, then bisect)

`base` vs `stack`, cases short + edit, reps 8, base = fast-hip. Run with `ABTEST_ARGS="--no-early-stop --reps 8"` (or `:--no-early-stop` in
`queue_run.sh`). `stack` sets: `STRATA_EXPDEC_SWIGLU`, `STRATA_ROUTE_FAST`, `STRATA_WAIT_COPY256`, `STRATA_IQ_SIGN_FAST` (GPU) and
`STRATA_Q2_FAST`, `STRATA_POOL_TAIL4`, `STRATA_POOL_LAZY_PARK` (CPU), all `1`. A member goes in only if it exists in source, is
documented exact, is opt-in and **acts on the fast-hip preset** (read in the source, not assumed):

| Member | Evidence it acts / is exact |
| --- | --- |
| EXPDEC_SWIGLU | works with deferral; parity ctest `iq_multi_parity_expdec_swiglu`; ~1% (kern-hip.md) |
| ROUTE_FAST | acts under deferral (verify.cpp prints when it will not); `native_multi_parity` passed 11-10; est. +1-2% |
| WAIT_COPY256 | acts since the kern-hip fix (it acts on `wait_flag_copy_i32`, the copy `STRATA_WAIT_FUSE` takes); `hip_wait_copy_256`; <= 0.6% |
| IQ_SIGN_FAST | `iq_sign_fast_test` 28.5M cases, 0 failures; both the plain and the swiglu gate/up kernels take it; no VGPR/occupancy change at the default `EXPDEC_MINB` (kern-cpu.md). No kernel-level timing yet |
| Q2_FAST | bit-exact (`q2_fast_parity_test`, probe falls back to the legacy kernel and prints "ignored"). **Acts here**: the IQ2_XS GGUF has 48 expert tensors of type 42 = Q2_0 (read from the file header: `IQ2_S` 68, `IQ2_XXS` 22, `IQ1_M` 6, and type 42 x 48, all `ffn_down_exps`), i.e. the CPU down rows; microbench x1.2-1.5 per row |
| POOL_TAIL4, POOL_LAZY_PARK | pool code in `pool.cpp`, act with `STRATA_POOL_FUSE_PHASES` (mode 7); exact; 03-cpu-pool: every arm within +-1.5% of base |

Left out, with the reason: `STRATA_COMBINE_POLL1`, `STRATA_ROUTER_FUSE`, `STRATA_IC_WARM_MB` (inert under deferral: they need `cfuse_on`, which
requires `!defer_`); `STRATA_IQ_LUT_DUP` (<= 0.5%, x16 halves occupancy, SWIGLU ignores the copies); `STRATA_MMVF_AUX` (already on through
`STRATA_COMBINE_CPU`); `STRATA_Q2_PREFETCH` (same rows, but one core is compute-bound and it needs a tuned byte count); `STRATA_IQ2S_PAIR`
(measured neutral); `STRATA_POOL_MWAITX` (a wake-latency trade, needs a tuned cycle count); `STRATA_MTP_EAGER` (wastes the draft steps past
the cut; 06c / 07 measured +1.6% / 0%). A member with a plausible cost would hide the others in a stack, so those stay out.

Decision rule (in the spec note): promote the group only if the 95% lower bound of the paired gain is >= 1% on short or edit
(`promote.py`); a decided gain is bisected by halves (GPU four, CPU three); no gain closes all seven. Preconditions before the run:
`ctest -R 'native_multi_parity|iq_multi_parity|hip_wait_copy|iq_sign_fast|q2_fast'`, and `expdec_bench 300 -1 20 16/42,17/42,22/42
def,dec:s0,dec:s1` for IQ_SIGN_FAST (keep it only if `dec:s1` is not slower). Log proof for the stack arm: `STRATA_ROUTE_FAST=1: top-10`,
`STRATA_EXPDEC_SWIGLU: gate/up, SwiGLU and q8_1 in one launch`, `STRATA_WAIT_COPY256 on`, `F12 sign form`; the CPU switches print a line only
when ignored. Text parity: the base text itself differs between runs on this box, so a text hash difference between stack and base is not
by itself a failure (25-kern-hip-confirm saw the same).

## Risk and next

- Nothing here changes an engine default or a source file. The two adopted presets are not selected by `run-pancake.sh` (its preset
  names are fixed), so the owner picks the default after the 15c redo: copy the file over `flashnext-q2_0-fast-hip.json` or add a model alias.
- `26-stack-small-exact` has no run result. `25-f-env-rerun-mmap` has none either.
- Next: run `tools/pancake/gen_bases.sh`, then 15-q2_0 (15c), 26 and 25-f-env-rerun-mmap through `queue_run.sh`; switch the live rdna3test specs
  to fast-hip; fix `abtest.make_config` template expansion (harness-core) and then `test_task_gate_effort.py` needs no generated file.
