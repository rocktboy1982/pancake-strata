# speclint: static lint of bench/specs/*.json (batch 3)

Branch dev/p16/speclint, worktree Strata-p16-speclint, base 3f4ba97cf (dev/phase16-academic).

## What

- `tools/pancake/spec_lint.py`: validates every spec in `bench/specs/` against what the harness reads.
  `python3 tools/pancake/spec_lint.py [SPEC ...] [-v]`. Exit 1 on errors; warnings and notes do not fail.
- `tools/pancake/test_spec_lint.py`: 19 unit tests on synthetic good and bad specs, plus the real specs
  (no crash). `python3 -I tools/pancake/test_spec_lint.py`.

Static only. It parses JSON, reads harness source with `ast` and regex, reads `CMakeLists.txt` and the
engine / server source as text. It never imports or runs abtest.py, quality_gate.py, task_gate.py,
replay_turns.py, the engine, a server or a model.

## Source check (what the lint takes from where)

- abtest.py: `LONG_TARGET`, `MAXTOK` (ast literals); the case names are the keys of the `P` table in
  `prompts()`; the prompt sources (`src/spec/draft_policy.cpp`, `src/spec`); arm keys (`arm.get(...)`:
  env, args, drafting, config, exe); top-level keys (`spec.get(...)`: cases, reps, prefill, base, arms and
  the sampling keys temperature, top_p, seed).
- `make_config` merge rules: mirrored in `merged()` (not imported). A test checks the mirror on a
  hand-written arm. If abtest changes `make_config`, change `merged()` too.
- task_gate.py: `DEFAULTS`, `REQUEST_KEYS`, `DEFAULT_TASKS` (ast); arm key `request`.
- quality_gate.py: the default `--set` file. replay_turns.py: the `replay` keys (`rp.get(...)`).
- Engine switches: a quoted name in `src/`, `include/`, `metal/`, `sycl/`, `serve/`, `strata_cli/`,
  `strata_control/` (test files excluded). `include/` is needed: most `getenv` calls live in headers
  (e.g. `include/strata/program/kmiss_tier.hpp` reads STRATA_ADAPT_KMISS). A src-only search reports 12
  valid names as unknown.
- Config keys of an arm's `config`: quoted in `serve/`, `strata_cli/`, `strata_control/` (Python).
- Engine flags: quoted `"--flag"` literals in the parser of the exe: `strata` -> `src/program/generate.cpp`
  plus `src/qwen35` (generate.cpp dispatches Qwen3.5/3.6 models to it); `strata-q36` -> `src/qwen35`;
  `strata-poly` -> `src/poly`. The base config's exe and an arm's exe pick the parser. The flags of
  `configs/pancake/qwen36-ud-q4_k_xl.json` (strata-poly) are not in generate.cpp, so one parser per exe is needed.
- Engine targets: `add_executable` names in CMakeLists.txt (strata, strata-q36, strata-poly).
- Tier names: no spec has a tier key; `abtest.py --tier screen|confirm` is CLI only. Not checked.

## Checks

Per spec: JSON parses; no duplicate key (JSON keeps the last one, so duplicate arm names are caught);
`base` and `arms` present; every top-level, arm, request and replay key is one the harness reads; value
types; case names; arm names are file names (abtest writes files named after them); `base` exists (a
generated `pancake-X.json` that is not here is linted through `configs/pancake/X.json`, a note only);
the base has `args`; its exe is an engine target; its flags and env names are known; each arm's env names,
flags (against the arm's exe), config keys and exe; `tasks`, the harness default files and the replay turn
log exist (turn log: warning, a run writes it); replay `arms` name real arms and `add_arms` do not collide.
Warnings: an arm that changes nothing against the base or an earlier arm (effective config and request
compared); a long case whose prompt (+8%, +reply) does not fit the arm's `--max-context`; an exe that is
not built here or lies outside this checkout. Base without `env` is an error only when an arm sets env
(abtest then fails with KeyError 'env').

## Result on dev/p16/speclint (2026-11-11)

`python3 -I tools/pancake/spec_lint.py`: 60 specs, 2 errors, 7 warnings. Unit tests: 19 ok, exit 0.

Errors (not fixed: they are config or measurement decisions, not spec typos):

| file | arm | issue |
|---|---|---|
| bench/specs/18-f11-attn-window.json | window | base `configs/pancake/qwen36-ud-q4_k_xl-q36.json` has no `env` object; the arm sets env, so abtest.make_config fails (KeyError 'env'). Fix in the config template (add `"env": {}`), not in the spec. |
| bench/specs/18-f17-poly-fattn-3060.json | fattn, fdec, both, fdec-from0 | base `configs/pancake/qwen36-ud-q4_k_xl.json` has no `env` object: same KeyError. Same fix. |

Warnings:

| file | arm | issue |
|---|---|---|
| bench/specs/15-q2_0.json | q2_0-stock | identical to arm q2_0. Its base `pancake-flashnext-iq2_xs-rdna3test.json` (template `configs/pancake/flashnext-iq2_xs-rdna3test.json`) sets none of STRATA_VERIFY_DEFER_CPU, STRATA_POOL_FUSE_PHASES, STRATA_EXPDEC_DOWN and has no --lookup-max-t, so the stock arm's removals do nothing. The spec note says base = iq2_xs fast-hip with the adopted items; `configs/pancake/flashnext-iq2_xs-fast-hip.json` sets all four. The base looks wrong (q2_0 is not "the same preset"). Decide the base; not a typo. |
| bench/specs/24-a44-bandit.json | defer | identical to the base: base fast-hip already sets STRATA_VERIFY_DEFER_CPU=1. It is arms[0], the reference the other arms are compared with. Intended? Keep or drop. |
| bench/specs/18-batch1-exact.json | port | exe `/home/dan/clone-of-d/src/Strata/build-hip/strata` is the main checkout's build (that checkout is on branch port/rdna3-decode), not this worktree's. The comparison mixes builds. |
| bench/specs/08-build.json | host-native | exe `build-hip-native/strata` is not built in this checkout (runtime; build it first). |
| bench/specs/11-conv-cache.json, 24-a15-park-ttl.json, 24-a24-suffix-index.json | replay | turn log `pancake-turns.jsonl` is not in this worktree (run output). The replay cannot run from here. |

No spec typo was found (every env name and flag is read or accepted), so no spec file was changed.

Spot checks that passed: STRATA_SPEC_MIN_TEMP (a21) is read in `include/strata/core/spec_prob.hpp`;
STRATA_SPEC_PROB_DT, STRATA_SPEC_PROB_GATE, STRATA_PROMO_CHUNK, STRATA_SPEC_COUPLED are read in
`include/strata/core/*.hpp`, `include/strata/program/promo_chunk.hpp`; the 17 flags used in arms are all in
generate.cpp; `lossy` is documentation only (no harness reads it; 53 specs carry it).

## Bugs found and fixed while testing the tool

- An arm's own exe was passed as a path to the flag-parser lookup (the arm's flags were never checked for
  `strata-poly`). Fixed by using the exe name; the poly unit test now asserts a flag error.
- A non-object `request` reported an error and then crashed; a non-list replay `arms` and an unhashable
  case name had the same problem. All guarded.

## Risk and limits

- The env and flag checks confirm that a quoted name exists in the reader source, not that the reader is live
  (a name left only in a message string would pass). The `strata` flag set is generate.cpp plus qwen35.
- Harness key sets come from regex over the harness source; a key read in a new form (for example
  `spec[k]` in a loop) is not seen. SAMPLING_KEYS and DOC_KEYS are hard-coded.
- LONG_SLACK = 1.08 is abtest's 8% warning, hard-coded.
- `test_abtest_cases.py` checks only the batch-2 specs; spec_lint checks all of them.

## Next

- Fix the two qwen36 templates (add `"env": {}`) or drop the env arms; choose the base for 15-q2_0; keep or drop
  `defer` in 24-a44; point `port` (18-batch1-exact) at this worktree's build.
- A getenv-level census with tree-sitter (call expressions with a literal argument) would tighten the env check.
- Run `tools/pancake/spec_lint.py` before a spec is committed (not wired into test_phase.sh or pre-push: not asked).
