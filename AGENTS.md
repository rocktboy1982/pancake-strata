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
   files), `README-PANCAKE.md` (on `updated-strata` and `main` only: the contribution text), specs in `bench/specs/`, raw machine output in `bench/results/<date>-<name>/` and
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
- `tools/pancake/box_chains/` - the Linux-box chains and their tools: `sweep.py` (arms, triage, two-run rule),
  `sweep_candidates.json` (registry: measured / untested / backlog / pruned), `retest_groups.json`, `bench-sweep.sh`
  (screen and rescreen), `bench-retest.sh` (gates), `bench-final-us.sh` (stock against updated), `bench-sched.sh`
  (scheduler: freezes the AA shell, runs one chain), `run-queue-us*.sh` (sequencers; the highest number is current),
  `bench-updatedAA.sh` (first upstream benchmark), `bench-exact.sh`, `bench-diagUS.sh`, `chainT-decide.py` (two-pass
  decision), older `win-chain*.sh` / `*tail-chain*.sh`. Strata-rdna3 files (`bench-updatedRD.sh`,
  `bench_updatedRD_conf.py`, `bench-rdgate.sh`, specs 250-252) are history: RD is paused.
- `tools/pancake/apply_triage.py`, `flag_diag.sh` / `flag_diag.py`, `host_tune*.sh`, `os_tune.sh` - apply per-flag
  verdicts to the shipped tuned-flags files, per-flag hardware diagnostics, host tuning (read-only checks).
- Shipped per-hardware flags: `terminal/data/tuned-flags.json` (read by `terminal/strata_control/tuned_flags.py`;
  applied at start; `STRATA_TUNED_FLAGS=0` or `--no-tuned-flags` turns it off); status page `/status`
  (`terminal/serve/status_page.py`, `terminal/serve/web/status.html`).
- On the branches `updated-strata` and `main` the upstream layout applies (`engine/` for the C++ engine, `terminal/`
  for serve, CLI and web); the paths above (`src/`, `serve/`) are the pancake line's. `README-PANCAKE.md` (branch root,
  per ported item: what, files, ledger ids, credits) lives on `updated-strata` and `main` only.

## Branches

- `port/rdna3-decode` - OUR WORK LINE (the pancake tree: engine work, specs, raw results in `bench/results/`, the
  three docs, `plan.html`). New work starts here and merges back here. The name is historical; it is not the
  Strata-rdna3 repo.
- `main` (fork rocktboy1982/pancake-strata) = `updated-strata` (upstream Strata v0.2.16 plus our hand-ported commits)
  plus MOD_PLAN.md, TEST_LEDGER.md, LEADS.md, plan.html, README.md (stock-against-updated comparison) and
  README-PANCAKE.md. It holds no `bench/results/`: ledger archive paths resolve on `port/rdna3-decode`. Only normal
  fast-forward commits, never force. The previous main is kept as `main-backup-2026-11-20`.
- `updated-strata` - the upstream-Strata port on v0.2.16 (worktree on the PC, push origin; the box queue uses it, do not change it).
  `updated-strata-v0.2.17` - the same port reduced to the tested arm, on upstream v0.2.17 (worktree `pancake-strata-v0217`; first commit = snapshot of the tag
  without workflow changes, real history on the local tag `updated-strata-v0.2.17-realhistory`). `updated-strata-rdna3` - the
  Strata-rdna3 port, published and paused. `NetworkPancake` is never deleted.
- Topic branches (`dev/*`, `work/*`, `feat/*`) as before. Run `git branch -r` and read `MOD_PLAN.md` before assuming a
  switch or feature exists. The push token has no `workflow` scope: keep `.github/workflows` as on the remote.

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
  discarded; keep the second pass. Screen and rescreen of single flags follow the FLAG TESTING PROTOCOL below (owner,
  2026-11-20; it replaces the old rule 'drop if any case is below -3%': that rule dropped flags that had a big gain on
  one case). Whole-set confirms (stock against shipped set) use the bar >= 2% end to end (>= 3% for lossy), start with
  `--tier screen`, confirm with `--block 3 --warmup 1 --outlier robust --reclaim-max 3 --reps 5` (owner, 2026-11-16:
  5 reps, not 9; 9 reps made the night confirm round 10+ hours).
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

## Flag testing protocol (owner, 2026-11-20) - applies to every flag test

Why: on 2026-11-19/20 flags were dropped on one noisy run, on text differences and as part of a bundle (ledger T-1002,
T-1003). These rules came out of that. Each flag is judged on its own numbers.

1. **Individual, never batches.** One arm per flag (`minus-X` for a flag the preset sets, `plus-X` for an off-by-default
   one; a flag whose preset value equals the engine default still gets the arm that flips it). A bundle result (whole
   preset, spec 250 `updated`, a whole-arm gate) never adds or removes a flag; it only triggers per-flag arms.
2. **Screen = run 1 of every flag**, one rep, cases short, edit and long 64K (prompt and decode). Blocks of at most 4 arms with a base start after every 2 arms
   (`updated`, a, a, `base-m1`, a, a, `base-late`; one warm-up start per slice; T-1006): every arm is judged against the MEAN OF THE TWO BASE STARTS AROUND IT, never against
   a base from the other end of a block or run. Drift guard = 5% on the LAST request of each case between those two bases (a pair up to 8% passes only when both
   bases show the same box state: long-prompt prefill within 12%); an arm whose pair disagrees is INCONCLUSIVE (never DROP or KEEP) and runs again alone (queue
   `rerun1`), there is no whole-block re-run. The box switches between a fast and a slow state for 10-15 min at a time (prefill -30%, decode -5..-12%); nothing in the
   harness can prevent it, so only the local base pair is trusted. Cause (T-1006): the pack (33 GB) is evicted from the page cache (60 GB RAM, engine 39 GB), the slow
   start reads it from the SSD at 0.45 GiB/s instead of 1.2 GiB/s. A base start in that state (engine log load rate < 0.8 GiB/s or cold 64K prefill < 1800 tok/s) is followed by a
   120 s idle pause with the pack read into the cache and runs once more (REGIME-PAUSE in the log); an arm is never judged against a base from the other state.
   Prevention (T-1006): `ABTEST_PREREAD=1` (set by the queue scripts, default 0) reads the model files of the config (`--pack` dir, `--native`, `--ple-gguf`) sequentially into the page cache
   before EVERY engine start and before the quiet gate (`PREREAD` lines; skipped at >= 95% cached; 90 s cap); the pause stays the fallback.
   `ABTEST_PREREAD=1` did not change the load (the engine pins its 33 GiB arena before it reads the file, which evicts the pre-read: TEST_LEDGER.md T-1006); `ABTEST_PREREAD=2` replaces it with a prefetch beside the engine's own load (`PREFETCH` lines, never more than 4 GiB ahead of the engine's reads, stops at `ready: http`).
3. **Classify by per-case effect (gain of having the flag ON):** HIGH = any case >= +2% even if another case is
   negative (short -1% with long +10% is a big gain on a case); LOW = best case above 0% and below 2%: not retested this
   round, goes to the backlog file; NEGATIVE = every case <= 0%: an off-by-default flag is NO-GAIN and dropped, a preset
   flag (having it ON hurts) is rescreened once before anything is removed.
4. **Rescreen = run 2, only for HIGH (and HURTS preset flags).** Independent (other seed, other block neighbours). The
   verdict comes from runs 1 and 2 together (effect = mean, spread = difference): KEEP (benefit above 0% in both runs on
   the carrying case), OPTIONAL (a case >= +10% and a case <= -3% in both runs: opt-in, listed under `optional` in the
   tuned-flags file, not default), MIXED (gain repeats but a case lost >= 3% in both: nothing ships), RETEST-GATE (lossy,
   two-run mean >= 3%: quality gate before it ships), DROP (run 2 does not repeat the benefit, or both runs <= 0%),
   INCONCLUSIVE (drift only). There is no separate 3-rep confirm stage. One run 1 below the bar is never a reason to
   remove a shipped flag.
5. **Priority:** highest gain first; low-gain flags wait in the backlog (`us-backlog-low.json`, `us-backlog.json`).
   Switches that cannot act in this build (hardware this box does not have, aliases) are the only ones never tested;
   everything else is tested or in the backlog, never judged by reading (rule 8).
6. **Exactness is not judged by text.** On this model plain Strata against itself gives different text in every run:
   'text differs' is never UNSUPPORTED. Judge exactness with the bitwise parity ctests; no result if they are 'Not Run'.
7. **Arms = their flags + ours added, never override.** When benchmarking a project's own default config (stock Strata,
   Strata-rdna3), the arm starts from that config exactly (all env and args, their values, their sampling); only what
   they do not set is added. Where our value differs it runs as a separate arm. Substitutions forced by a model file
   (IQ2_XS instead of IQ3_S, `--mtp-q4 proj` for a small head) are logged as `CONFIG SUBST` with the reason.
8. **Stock Strata appears only in the final comparison** (5 reps, `--block 3 --warmup 1 --outlier robust
   --reclaim-max 3`, exact and lossy, short / edit and long 10K / 32K / 64K / 100K, with the quality gate). No stock arm
   inside flag screens. Strata-rdna3 tests are stopped (owner, 2026-11-20); its branch stays published, paused.
9. **Gates:** `quality_gate.py` for lossy arms, `task_gate.py` for close variants; a gate that rejects a whole arm does
   not remove a flag; attribute with per-flag arms. A gate result is 'no measurable loss', not proof: the default task
   set scores 100% on our base (T-819).
10. **Save results as they come.** Commit and push the raw rows of every flag block (a loop copies new result files from
    the box to `bench/results/<date>-flag-tests/` every 5 min and pushes); results also live on the box under
    `/home/dan/auto-runs/`. Every run still gets a ledger row; large files (> 1.5 MB, engine logs, traces, build logs)
    stay on the box and the row names their path.

## Box operations (Linux box 192.168.100.82, RX 7900 XT)

- One measured job at a time, everything serialized; no CPU work beside it. Every chain waits until the box is idle
  (30 s quick idle between steps, 3 min before a chain's first step), stops `pancake.service` only when no client is
  connected and restarts it at the end, and ends with a DONE marker line.
- `pancake.service` is never stopped while it serves. Box files are never deleted; scripts are never edited while they
  run (deploy a new versioned copy: `tmp` file + `mv`). Never edit sudoers or permissions.
- Start background work with `ssh -n host 'cd dir && setsid -f sh ./script >/dev/null 2>&1 < /dev/null'`: a leftover
  `bash -c` wrapper whose command line holds a chain name blocks the schedulers' busy checks.
- A chain must check build exit codes and stop with an `ERROR` line; a failed link must never delete the previous
  binary (back it up first). The harness python on the box is `Strata-win/.venv/bin/python` (`ABTEST_PYTHON`);
  `ctest` is `/home/linuxbrew/.linuxbrew/bin/ctest`; builds go outside `/` (`/mnt/pancake`), stop under 2 GB free.
- The AA chain (`redo-chainAA.sh`) is frozen with SIGSTOP by owner order: only benchmarks run; never resume it without
  the owner. A running job is stopped only on an owner order.
- Always check processes (CPU, memory, load), not only the log, when reporting status.
- The page cache cannot hold the model files (36.5 + 26.8 GiB gguf) beside a 40 GB engine (the engine pins its 33 GiB arena before it reads the file), so every start reads them from the NVMe. `ABTEST_PREREAD=2` (abtest.py, exported by `run-queue-us13.sh`) runs a prefetch thread beside the engine's own load
  (`PREFETCH <file> <GiB> <seconds>` log line, never more than 4 GiB ahead of the engine's reads, stops at `ready: http`; mode 1, a pre-read before the start, was evicted by the arena and does not help). It does not fix the slow starts either (T-1006). `ABTEST_SETTLE_AFTER_LONG_S=120` (exported by `run-queue-us13.sh`, default 0): after an engine run of >= 130 s the harness idles that long before the next engine start (`SETTLE <s> after a <run> s run`; nothing of ours runs meanwhile); REGIME-PAUSE stays the fallback.
  Every engine start prints `STARTDIAG` (nvme0n1 sectors and io_ticks, meminfo Cached / Mlocked / MemAvailable, nvme temperature, compact_stall delta, previous run length, gap) and `STARTLOAD` (the load rate); read them with `python3 tools/pancake/box_chains/startdiag_report.py <stdout files>` (slow share by gap bucket and SETTLE on / off, diag correlations). Never read the files by hand beside a running benchmark; check cache state read-only with `fincore`; no sudo, no drop_caches.

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
