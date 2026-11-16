# AGENTS.md

`pancake-strata` is a performance fork of [Strata](https://github.com/Niko1221/Strata): measured kernels, cache and
scheduling changes, and a benchmark suite on top of the upstream engine, app and terminal. Work here is engine-first;
the app and terminal are inherited and change rarely. Target: single-user agent serving on an RX 7900 XT 20 GB
(HIP/gfx1100, Linux) (the RTX 3060 12 GB, CUDA/sm_86, was removed 2026-11-08; CUDA stays buildable). Daily models: Qwen3.8-Flash-Next (IQ2_XS, Q2_0) and
Qwen3.6-35B-A3B; see README §8 and `docs/MODELS.md`.

## Documentation rules (owner, 2026-11-15) - read first, apply always

1. **Three files hold all fork knowledge.** Only these are maintained, at the repository root:

   | file | what goes in it | when to update |
   |---|---|---|
   | `MOD_PLAN.md` | what the daily preset runs (with test ids), decisions waiting for the owner, the GPU queue, code to write, defects, parked items, the do-not-repeat list, architecture facts | whenever a decision, status or priority changes |
   | `TEST_LEDGER.md` | one row per test ever run: next free id `T-nnn` in its area table; date, item, spec, hw, tier/reps, numbers per case, exact/text, verdict, archive path | in the same commit as every run, including no-gain, aborted and offline runs |
   | `LEADS.md` | every unported idea (fork commit, upstream PR, paper, hardware note) with the author number and whether our tree has it | when a new idea is found; a chosen lead moves to MOD_PLAN.md |

2. **Never create another document**, in any folder, from any agent or tool: no reports, notes, status, findings,
   night results, analyses, plans, design notes or summaries (no `*_REPORT.md`, `NIGHT_RESULTS_*.md`, `pancake-docs/*.md`,
   `docs/*` fork notes). If you are about to write one, put its content into the right one of the three files instead.
3. **Allowed besides the three:** `AGENTS.md` (these rules), `README.md` (reader-facing summary that points to the three
   files), specs in `bench/specs/`, raw machine output in `bench/results/<date>-<name>/` and
   `pancake-docs/research/bench/` (files the tools write: jsonl, json, stdout, summaries; no hand-written prose), and code
   comments. Inherited upstream `docs/` stays as upstream has it. One generated view is allowed: `plan.html` (the
   filterable plan board: plan items by status, every test by verdict, the full plan). Never edit it by hand; rebuild it
   with `python3 tools/pancake/plan_page.py` and commit it in the same commit as every change to MOD_PLAN.md or
   TEST_LEDGER.md, so it never goes stale.
4. **Cite only the three files** (and specs / raw archives) in code comments, spec notes, commit messages and docs:
   tests by id (`TEST_LEDGER.md T-123`), plan items by section (`MOD_PLAN.md section 5`), ideas as `LEADS.md`. Never cite a
   deleted document or an old commit's file.
5. **Every claim needs a ledger row.** Do not state a number that is not in TEST_LEDGER.md or in a raw archive.
6. **Keep the three in step:** a test result goes to TEST_LEDGER.md, its decision to MOD_PLAN.md (sections 3-5), and a
   lead that got tested moves out of LEADS.md into MOD_PLAN.md / TEST_LEDGER.md.
7. **Ledger ids are taken at commit time.** Several agents append rows (Windows box, Linux box). Right before committing,
   `git fetch` and merge `port/rdna3-decode`, then give each new row the next free `T-nnn` after the highest id in the
   merged TEST_LEDGER.md; on a merge conflict in the ledger keep both sides in id order and renumber your own rows, never
   the other agent's.
8. **Every lead is followed (owner, 2026-11-15).** Port each lead from LEADS.md and from the forks as a default-off
   `STRATA_*` switch (byte-identical default, classified in `conversation_disk_env.def`), build, `ctest`, add an A/B spec
   for the night queue, and record the result. Never drop a lead because reasoning says it will not help here ("our preset
   does not use that path", "attention is < 1%", "pcie-frac is 0", "large port"): test, never judge by reading. Only
   hardware this box does not have (two GPUs, CUDA sm_86) is parked, and it stays in LEADS.md.

## Where the truth lives

- `MOD_PLAN.md`, `TEST_LEDGER.md`, `LEADS.md` - see the rules above.
- `README.md` - reader-facing: measured results (section 2), how it works, switch census (section 6), build/run
  (section 8), credits (section 9).
- Raw evidence: `bench/specs/`, `bench/results/<date>-<name>/`, `pancake-docs/research/bench/` (result archives, run
  scripts, night-queue logs), `bench/quality/`, `bench/prefetch/`.
- `docs/` is inherited upstream; the kept shared references are `docs/QUANTS.md`, `docs/AI_REFERENCE.md`,
  `docs/DETAILS.md`, `docs/AMD_HIP.md`, `docs/MODELS.md`.

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
`port/rdna3-decode` and merge back there), not blindly from `main`. Run `git branch -r` and read `MOD_PLAN.md`
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
- Windows: the documented path is CUDA (README section 8 (recipe: MOD_PLAN.md section 10)). `build_matrix.sh`, `abtest.py` (reads
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
  discarded; keep the second pass. Screen: promote from 0% up, drop if any case is below -3% (owner rule 2026-11-13); decide at confirm with the bar
  >= 2% end to end (>= 3% for lossy). Start with `--tier screen`;
  confirm with `--block 3 --warmup 1 --outlier robust --reclaim-max 3 --reps 5` (owner, 2026-11-16: 5 reps, not 9;
  9 reps made the night confirm round 10+ hours).
- Lossy arms also need `tools/pancake/quality_gate.py` (token forcing); for close variants use
  `tools/pancake/task_gate.py` (KLD/top-1 agreement alone does not predict task score).
- Before any A/B, run `python tools/pancake/spec_lint.py` against `bench/specs/`: it flags arm `STRATA_*` env keys and
  `--flags` that this fork does not read (a 2026-11-12 audit dropped `STRATA_MTP_DRAFT_ALL` from spec 34 as an inert
  arm this way). An arm that is not in `conversation_disk_env.def` or parsed in `generate.cpp`/`serve` is a no-op, not
  a test.
- Specs live in `bench/specs/`; raw results go to `bench/results/<date>-<name>/`. Every run (including no-gain and
  aborted ones) gets a row in `TEST_LEDGER.md`; a result that changes a decision updates `MOD_PLAN.md` (sections 3-5).
  Daily presets can differ from engine defaults by owner decision.
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
- When adopting code or an idea from another project, add the credit in README §9 in the same commit (fork, commit,
  author numbers), and mark the lead in `LEADS.md`.

## Claims and docs

- Docs style: plain words, short sentences, non-technical reader first. Every performance claim names the hardware,
  model/quant, preset and measurement; no claim without a measurement.
- Label conclusions drawn from reading code (not reproduced) as "unconfirmed static analysis"
  (`CONTRIBUTING.md`); never present fake-engine or fixture results as hardware measurements.
- In `LEADS.md`, label claims `[code]`, `[measured]`, `[paper]`, `[derived]` or `[unverified]`.
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
