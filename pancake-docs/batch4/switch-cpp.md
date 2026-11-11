# Batch 4, switch-cpp: every audited opt-in switch says once that it acted, or why it will not act

Branch `dev/p16b/switch-cpp` (from `dev/phase16-academic` 0dd7ef2e8). MOD_PLAN "Project features discovered" item 1. Nothing
here ran on a GPU and no engine was started: compile-only checks and one CPU unit test.

## What and why

Earlier GPU arms measured nothing because opt-in switches were silent no-ops: F21 STRATA_WAIT_COPY256 never acted on the
preset, STRATA_IC_WARM_MB never launches under deferral (the fused CPU combine is off there), STRATA_ROUTER_FUSE is inert
under deferral. `include/strata/core/switch_report.hpp` (new, header-only) gives each switch two once-only stderr lines, and
the six audited switches use it. switch-harness (`tools/pancake/switch_report.py`) reads the lines; abtest can mark an arm VOID
when its switch never wrote "acted".

Source check: none. This is a project feature (MOD_PLAN item 1), not a paper method.

## The line format (for switch-harness)

Exactly one of these two forms, one line, on stderr:

    strata switch: NAME acted: DETAIL
    strata switch: NAME will not act: REASON

- Regex: `^strata switch: (STRATA_[A-Z0-9_]+) (acted|will not act): (.*)$`. The prefix is `strata switch: ` (one space after
  the colon), NAME is the environment variable or build macro, then one space, the kind, `: ` and free text.
- DETAIL / REASON never contain a line break (the helper writes `\n` / `\r` as a space) and are at most 511 bytes.
- Each kind is written **at most once per process and switch**. A switch can write both lines, in either order: a reason
  that held for a while (a short window, a fallback, the first layer) and then it acted. "acted" means its path ran at least
  once. The verify window is captured into a graph, so for the verify.cpp switches "acted" is written while the window is
  captured: the path is in the graph that every replay runs.
- Nothing is written for a switch that is unset (or set to a value that means off, e.g. `0`). With every switch unset the
  output is byte-identical to before.
- Only "acted" counts as proof. "will not act" is the reason for a human; a missing line means neither happened (for example
  the code that would act was never reached).
- Other free-form lines about the same switches still exist and are not part of the format: `strata: STRATA_ROUTER_FUSE: ...`
  and `strata: STRATA_ROUTE_FAST: ...` (native_bf16.cu, armed, not acted), `strata verify: STRATA_IC_WARM_MB=N: ...` (setup
  done, not acted). Parse only lines that start with `strata switch: `.

The six names: `STRATA_WAIT_COPY256`, `STRATA_ROUTE_FAST`, `STRATA_ROUTER_FUSE`, `STRATA_IC_WARM_MB`, `STRATA_EXPDEC_SWIGLU`,
`STRATA_HIP_GLOBAL_WEIGHTS` (the list in the header; the CPU test checks the names).

## Per switch: when each line is written

| Switch (set when) | acted | will not act (first reason that holds) |
| --- | --- | --- |
| STRATA_WAIT_COPY256 (`=1`), verify_kernels.cu `wait_copy_wide` + verify.cpp plan copy | `plans of N words use the 256-thread wide copy` | `a plan of N words (it takes plans of more than 512)`; `every expert is resident in VRAM: no plan copy`; `STRATA_VERIFY_DEVICE_PLAN copies the plan in its own launches`; `the plan's wait and copy are two launches (STRATA_VERIFY_PROFILE is set / STRATA_PCIE_BALANCE is on / it needs STRATA_WAIT_FUSE=1 or STRATA_FUSE_WAIT=1)` |
| STRATA_ROUTE_FAST (`=1`), verify.cpp router step | `the top-10, ids + weights and the step's ring in one launch, the rows copied by the router GEMV` (or `publish launch`) ` (F9)` | the shared router-step list below, then `the lookahead router is on` |
| STRATA_ROUTER_FUSE (`=1`), verify.cpp router step | `the router GEMV's last block makes the top-10 and publishes (item 48)` | `it needs STRATA_ROUTE_PUBLISH=1`; the shared list; `STRATA_VERIFY_DEFER_CPU is on`; `STRATA_DOORBELL_STORE is on` (HIP); `STRATA_ROUTE_FAST=1 takes the router step`; `no route-publish slot is left for this stream`; `the router GEMV's launch is not the 256-thread kernel that takes it` |
| STRATA_IC_WARM_MB (`> 0`), verify.cpp | `layer L+1's pre-doorbell weights read on a side stream during layer L's CPU wait (item 47)` | `its setup failed (no weights, or the side stream's allocation)`; `the fused CPU combine is off (<why>): the reads ride in its CPU wait`, <why> one of `it needs STRATA_COMBINE_CPU=1`, `STRATA_VERIFY_DEFER_CPU is on`, `every expert is resident in VRAM`, `STRATA_VERIFY_DEVICE_PLAN is on`, `--remote-expert-opt is set`, `the model's top-k is not 10`, `the native MoE combine is off`, `STRATA_DEC_BATCH=0`, `the shared expert is not native BF16`, `the CPU experts are not in the native layout`, `the model width is not a multiple of 4`; `the side stream's fork failed` |
| STRATA_EXPDEC_SWIGLU (`=1`), iq_kernels.cu `expdec_gu_swiglu` | `gate/up, SwiGLU and q8_1 in one launch (item 48)` (+ `, F12 sign form ...` with STRATA_IQ_SIGN_FAST) | `EXPDEC is off (STRATA_EXPDEC)`; `the old IQ kernels are on (STRATA_OLD_IQ_MMVQ)`; `EXPDEC does not take this gate/up layout`; `n_ff is not a multiple of 32`; `the layer's down input is Hadamard-rotated` |
| STRATA_HIP_GLOBAL_WEIGHTS (build macro, `-DCMAKE_HIP_FLAGS=-DSTRATA_HIP_GLOBAL_WEIGHTS=1`), iq_kernels.cu at start-up | `built with it: the grouped expert kernels read the weights through global-address-space pointers (Phase 15)` | `built with it, but only a HIP build that is not gfx906 reads the weights that way` |

Shared router-step list (ROUTE_FAST and ROUTER_FUSE give the same text for the same cause): `STRATA_DEC_BATCH=0`, `the layer
has no router weight`, `the native router is off`, `the router is not 512 experts x top-10`, `every expert is resident in VRAM`,
`STRATA_VERIFY_DEVICE_PLAN is on`, `--remote-expert-opt is set`, `the windows always publish (pipelined)`, `STRATA_ROUTE_PRIOR is
set`, `the STRATA_ROUTE_RESIDENT margin is above 0`. A window of 1 row or of more than 8 rows is not a reason: nothing is said
for it (the next window may act).

HIP_GLOBAL_WEIGHTS is a build setting, so its line comes at process start (a static initializer in iq_kernels.cu, written only
when the build has the macro). It reports the build, not a kernel launch.

## Code

- `include/strata/core/switch_report.hpp`: `struct Switch {name, acted_said, refused_said}`; one inline `Switch` per audited
  switch (one object per process whichever file reports it); `switch_report::acted(sw, fmt, ...)` and
  `switch_report::will_not_act(sw, fmt, ...)` (printf format, return true when that call wrote the line); `format_line`. After
  the first line of a kind a call is one relaxed atomic load; the first caller wins an atomic exchange, so exactly one thread
  writes; the line goes out in one `fprintf`.
- verify.cpp: the router step's ten shared conditions are one ternary list (`publish_off`) that now both gates the two
  one-launch publishes and names the reason; `cfuse_on(n)` is `cfuse_off(n) == nullptr` with the same order of tests. A brute
  force over all inputs (2^14 booleans x 4 margins incl. NaN x 7 row counts x NE x K, 1,835,008 cases, throwaway program) gives
  rfast, rpub and cfuse_on identical to the old expressions. The init-time ROUTER_FUSE note moved to the router step.
- verify_kernels.cu: the ad-hoc COPY256 line is replaced.
- iq_kernels.cu (not in this agent's file list; own commit so the lead can take or drop it): EXPDEC_SWIGLU's ad-hoc line is
  replaced; the HIP_GLOBAL_WEIGHTS start-up line.

## Tests and checks (all with `systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 nice -n 19`)

Passed:
- `switch_report_test` (new, label cpu): `cmake -DSTRATA_ENABLE_HIP=OFF -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=ON`
  (build-cpu in the worktree), `ninja switch_report_test`, `ctest -L cpu -R switch_report`: 1/1 passed (116 cpu tests
  registered). Also built by hand with g++ C++17/C++20 and clang++ with `-fsanitize=thread`: ok, no TSan report. Checks: the
  exact text of both lines, once per kind, the kinds independent, line breaks flattened, an unreported switch writes nothing,
  16 threads x 2,000 calls of each kind write exactly two lines and exactly one caller of each kind is told it wrote, the six
  names.
- HIP gfx1100 (compile commands of Strata-p16-compilegate/build-hip, rewritten to this worktree): verify.cpp, verify_kernels.cu,
  iq_kernels.cu with `-fsyntax-only`; verify.cpp also a full object (strings in the object checked); verify.cpp also with
  `-DSTRATA_HIP_GFX906=1`; iq_kernels.cu also with `-DSTRATA_HIP_GLOBAL_WEIGHTS=1` (`-fsyntax-only` and a full object, 92 s; the
  object holds the "acted" text). No new warning (only the usual nodiscard / sign-compare ones in other lines).
- CUDA sm_86 (nvcc 13.1, build-cuda commands): verify.cpp full object, verify_kernels.cu and iq_kernels.cu (`-fsyntax-only`,
  iq_kernels with `-DSTRATA_HIP_GLOBAL_WEIGHTS=1`, the will-not-act branch): ok, no new warning.

Not run: anything on a GPU or with a model (the lines themselves have not been seen from a real engine run yet); the full
`ctest -L cpu` (no CPU-built source changed besides the new header and test).

## Risk

- Default path: every new call sits behind the switch's own flag (a static bool read once, or `warm_segs_ != nullptr`), so
  with the switches unset nothing is written and no new work runs. The two refactored conditions are proven equal.
- With a switch set, the router step evaluates the ten-condition list on every layer even for 1-row windows (pure loads, no
  GPU work).
- Log texts changed (only with the switch set): the old `strata: STRATA_WAIT_COPY256 on: ...`, `strata verify:
  STRATA_ROUTE_FAST=1: ...` / `... will not act: ...`, `strata verify: STRATA_ROUTER_FUSE=1 will not act: ...`, `strata verify:
  STRATA_IC_WARM_MB: setup failed, off` and `strata expdec: STRATA_EXPDEC_SWIGLU: ...` lines are now `strata switch:` lines. The
  "log proof" notes in bench/specs/17-f9-route-fast.json, 25-kern-hip.json and 25-kern-hip-confirm.json still quote the old
  texts (presets-specs owns them; no code parses them).

## Next

- switch-harness: parse the lines above; an arm is VOID when its switch never wrote "acted" in that run's engine log.
- More switches: add one `inline Switch` to the header and call `acted` / `will_not_act` where the switch decides.
- First GPU night run: check that each arm's log has its `strata switch: NAME acted:` line.
