# AGENTS.md

`pancake-strata` is a performance fork of [Strata](https://github.com/Niko1221/Strata): measured kernels, cache and
scheduling changes, and a benchmark suite on top of the upstream engine, app and terminal. Work here is engine-first;
the app and terminal are inherited and change rarely. Target: single-user agent serving on an RX 7900 XT 20 GB
(HIP/gfx1100, Linux) and an RTX 3060 12 GB (CUDA/sm_86). Daily models: Qwen3.8-Flash-Next (IQ2_XS, Q2_0) and
Qwen3.6-35B-A3B; see README §8 and `docs/MODELS.md`.

## Where the truth lives

- `README.md` - measured results (§2), how it works (§5), `STRATA_*`/flag census (§6), subsystem walkthrough (§7),
  build/run (§8), credits (§9).
- `PANCAKE_STATUS.md` - what is built and tested now, and what runs next; `MOD_PLAN.md` - the authoritative plan,
  the night queue, and the "not adopted" ledger (closed decisions, plus the trigger to reopen parked ones). The plan
  can list an item and still forbid starting it (Phase 18: no start until researched and the plan is reviewed).
- `pancake-docs/` - the fork's long-form record; start at `pancake-docs/research/README.md` for the topic reports.
  `docs/` is mostly inherited upstream, but fork findings live there too (`docs/PANCAKE_FINDINGS.md`,
  `docs/QUANTS.md`), as do the shared references (`docs/AI_REFERENCE.md`, `docs/DETAILS.md`, `docs/AMD_HIP.md`,
  `docs/WINDOWS_CUDA_SM75.md`).
- `reports/` holds root-relative copies of the research `report.md` files; when you edit one, sync its copy (same
  text, links differ).
- `bench/results/<date>-<name>/` - raw evidence (`README.md`, `REPORT.md`, `.jsonl`/`.json` rows); the running test
  and findings ledger is `docs/PANCAKE_FINDINGS.md`. Do not claim a number that is not there or that you measured.

## Layout

- `src/`, `include/` - the C++ engine (CUDA/HIP/SYCL/Metal). `src/program/generate.cpp` holds generation, scheduling
  and cache decisions; `src/core/verify.cpp` the verify/combine path; kernels under `src/kernels/{cuda,cpu}/`.
- `src/poly/` - the second engine, `strata-poly` (CUDA only): long-context path with q8 KV and a conversation disk
  cache.
- `serve/` - Python server: OpenAI and Anthropic APIs, agent loop, loop guard, conversation cache, tool parsing.
- `strata.py`, `strata_cli/`, `strata_control/` - the `strata` terminal; `apps/` - desktop app and Node host
  (inherited).
- `configs/pancake/` - serving presets; `bench/specs/` + `bench/results/` - A/B specs and evidence.
- `tools/pancake/` - the measurement suite (A/B, quality gates, night queue, TTFT, profiling).

## Branches

`main` is the default/release line. Engine work lives on topic branches (`dev/*`, `work/*`, `feat/*`, for example
`dev/p17/integration`). New work starts from the base named by the active MOD_PLAN phase (Phase 18 workers branch from
`port/rdna3-decode` and merge back there), not blindly from `main`. Run `git branch -r` and read `PANCAKE_STATUS.md`
before assuming a switch or feature exists.

## Build

HIP (Linux/ROCm; the benchmark platform):

```sh
cmake -S . -B build-hip -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build build-hip --target strata -j2
```

CUDA:

```sh
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86-real
cmake --build build-cuda --target strata strata-poly -j8
```

- Add `-DSTRATA_BUILD_TESTS=ON` to build the test targets.
- Keep the conventional `build-hip` / `build-cuda` directory names: `tools/pancake/test_phase.sh` rewrites preset
  paths to `/build-hip/`.
- CMake fetches the pinned llama.cpp revision; for an offline build pass `-DSTRATA_GGML_DIR=<checkout>`.
- Never enable CUDA and HIP in one build. `strata-poly` is CUDA-only.
- Windows: the documented path is CUDA (`docs/WINDOWS_CUDA_SM75.md`). `build_matrix.sh`, `abtest.py` (reads
  `/proc`), `test_phase.sh`, `night_queue.py` and the other `.sh` gates are Linux-only; use WSL for measurement work.

## Tests

C++ (build `strata_tests` first):

```sh
cmake --build build-hip --target strata_tests
ctest --test-dir build-hip -L cpu --output-on-failure     # no GPU or model files needed
ctest --test-dir build-hip -L gpu --output-on-failure     # needs the card alone
ctest --test-dir build-hip -L model --output-on-failure   # needs model fixtures
```

- The documented full-suite command excludes two known environment failures:
  `ctest --test-dir build-hip --output-on-failure --timeout 60 -E '^(ple_parity|platform_memory_test)$'`
  (`docs/AMD_HIP.md`).
- A CPU-only test missing from `_strata_tests_cpu` in `CMakeLists.txt` silently gets the `gpu` label and never runs
  in `ctest -L cpu`. Add it to that list (the Phase 17 review caught `defer_auto_test` this way).
- `tools/pancake/build_matrix.sh --tests hip-tests` builds every test executable and runs `ctest -L cpu`; `--head`
  builds the committed HEAD from a detached worktree, so other sessions' edits cannot break the check.

Python (fake servers/engines; no GPU and no downloads):

```sh
python -m pytest -q tools/pancake/test_<name>.py
python -m pytest -q serve/test_<name>.py
python -m unittest serve.test_<name> -v        # serve tests are unittest-style; pytest also runs them
python -m pytest tests/terminal
```

- `conftest.py` redirects `APPDATA`/`LOCALAPPDATA`/`XDG_*` to a temp dir so tests never touch the real user config.
  Do not bypass it.
- Some tool e2e tests skip while a real engine/server is running; a skip is not a pass.
- `apps/host`: `npm install && npm test` for the Node host/UI tests.

## Measurement rules

- Never benchmark while other work uses the machine (agents, builds, another server). `abtest.py` waits for a quiet
  machine, marks noisy runs and repeats them; it refuses to start beside a running engine/server (exit 3).
- Judge speed changes only by paired A/B, never one run:
  `python tools/pancake/abtest.py SPEC.json --out bench/results/<day>-<name>.jsonl`. Run 1 is warm-up and is
  discarded; keep the second pass. Promotion bar: >= 2% end to end (>= 3% for lossy). Start with `--tier screen`;
  confirm with `--block 3 --warmup 1 --outlier robust --reclaim-max 3 --reps 9`.
- Lossy arms also need `tools/pancake/quality_gate.py` (token forcing); for close variants use
  `tools/pancake/task_gate.py` (KLD/top-1 agreement alone does not predict task score).
- Specs live in `bench/specs/`; results go to `bench/results/<date>-<name>/`, and night outcomes plus owner
  decisions to `pancake-docs/NIGHT_RESULTS_<date>.md` / `pancake-docs/PRESET_BUNDLES.md`. Update
  `PANCAKE_STATUS.md` / `MOD_PLAN.md` when a result changes the plan; daily presets can differ from engine defaults
  by owner decision.
- GPU-heavy plans run through `tools/pancake/night_queue.py` (plan in `night_queue_plan.json`; `--dry-run` first) or
  `tools/pancake/test_phase.sh` on a quiet night, not during the day.
- The tools parse the engine's own log lines (prompt/decode tok/s, park/restore/skip). Do not remove or reword them
  without updating `tools/pancake/`.

## Switches, presets and credits

- New features are opt-in; the unset/default path must stay byte-identical, and say how you checked. Change a
  default only with a paired measurement.
- Every new `STRATA_*` name that affects engine state or cache must be classified in
  `src/core/conversation_disk_env.def` (`KeyIfSet`, `InConfig`, `Speed`, ...). A missing entry breaks `ctest -L cpu`
  and can reuse a disk-cache image made under different routing/KV settings. Check with
  `python tools/pancake/test_disk_env_table.py`.
- Switch census and exact/lossy status: README §6. Presets: `configs/pancake/*.json` keep the `<REPO>`,
  `<STRATA_DATA>` and `<MODELS>` placeholders. `run-pancake.sh [model] [fast|exact|turbo]` renders them and defaults
  to the owner's `/home/dan/...` paths, so set `STRATA_DATA` and `MODELS` on another machine.
- Model-pack recommendation policy: ISTA-DASLab -> Unsloth -> Gyro -> others, only a pack that fits the model and
  card; honor an explicit user choice.
- When adopting code or an idea from another project, add the credit in README §9 and
  `pancake-docs/RESEARCH_AND_FINDINGS.md` §4 in the same commit.

## Claims and docs

- Docs style: plain words, short sentences, non-technical reader first. Every performance claim names the hardware,
  model/quant, preset and measurement; no claim without a measurement.
- Label conclusions drawn from reading code (not reproduced) as "unconfirmed static analysis"
  (`CONTRIBUTING.md`); never present fake-engine or fixture results as hardware measurements.
- In `pancake-docs/research/`, label claims `[code]`, `[measured]`, `[paper]`, `[derived]` or `[unverified]`.
- Keep passed / failed / not run separate when reporting validation.

## Contributing

- Issue/PR titles need an area prefix: `[Engine]`, `[GUI]`, `[TUI]`, `[Benchmark]`, `[Docs]`, `[Other]` (CI fails
  without it).
- One problem per PR; agree scope in an issue before a large feature, a refactor or a default change.
- Build every backend a file touches (CUDA/HIP/SYCL, Metal for Metal paths) before review. A compile is not a
  hardware result.
- `tools/pancake/pre-push.sh` runs `build_matrix.sh --head --configure-only` (~2 min); install it as the git
  pre-push hook. `PANCAKE_PUSH_FULL=1 git push` does the full link check; `SKIP_BUILD_MATRIX=1` skips it once.

## Upstream parts and installers

No fork-specific installer or release is defined in this repo. The inherited `packaging/install.*` scripts and the
installer links in upstream docs point at `Niko1221/Strata` releases and do not contain pancake's engine changes; do
not hand them out as this fork. Build from source (above), or follow `docs/SOURCE_INSTALL.md` for the inherited
terminal/app.
