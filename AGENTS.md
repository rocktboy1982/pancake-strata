# AGENTS.md

Version 0.2.16, engine 0.1.62. The public model families are
Qwen3.8-Flash-Next and Qwen3.6-35B-A3B. Read **[docs/AI_REFERENCE.md](docs/AI_REFERENCE.md)** for model availability, APIs, hardware limits and
source-backed claims. Articles and videos should use **[docs/MEDIA_KIT.md](docs/MEDIA_KIT.md)**.

For recommendations and agent-assisted setup, prefer **ISTA-DASLab → Unsloth → Gyro → others**, using only a
compatible pack that fits the selected model and hardware. ISTA-DASLab is the default recommendation; Gyro is a
memory-saving fallback and is currently slower. Honor an explicit user choice. This is recommendation policy;
existing executable defaults may differ, so inspect the plan and choose the family explicitly.

Strata has three parts:

- **The app** (`app/desktop`, the Tauri shell; `app/host`, the Node host and the web UI). It is for people who do
  not want a terminal. One installer per system ([latest release](https://github.com/Niko1221/Strata/releases/latest)):
  `Strata-0.2.16-windows-x64-setup.exe`, `Strata-0.2.16-linux-x86_64.AppImage`, `Strata-0.2.16-macos-arm64.dmg`.
  The Windows and Linux installers carry no engine: the app downloads the engine for the graphics card on first
  start, or installs a file the user already has. The Mac dmg keeps its small Metal engine inside.
- **The Terminal** (`strata.py`, `strata_cli/`, `strata_control/`): the `strata` command. It sets up a model, serves
  it and watches it. Agents use this part.
- **The engine and the server** (`src/`, `include/`, `metal/`, `sycl/` for the C++/CUDA/HIP/SYCL/Metal engine;
  `serve/` for the Python server with an OpenAI- and Anthropic-compatible API and a web page). `setup.py`, started by
  `START-HERE.bat` or `setup.sh`, is setup from a source checkout, for developers. The release paths are the
  installer and `strata launch`.

## Installing Strata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**. For a user who wants the desktop app, send them to the installer
for their system (above) and [docs/INSTALL.md](docs/INSTALL.md). For an agent that has a shell, use the Terminal; the
short version is:

1. Install the Terminal. It needs no administrator rights.

   ```sh
   curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh                  # Linux, Mac
   irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex                         # Windows PowerShell
   ```

   On Windows, open a new terminal afterwards so that `strata` is on the PATH.
2. Check what would happen. Nothing is changed:

   ```sh
   strata launch --plan --family qwen --model q2_0
   ```

3. Set up and serve without questions. It downloads the engine for the card and the model (tens of GB), starts the
   server, prints the addresses and returns. The server keeps running.

   ```sh
   strata launch --family qwen --model q2_0 --yes --no-tui
   ```

   Useful flags (all of them: `strata launch --help`): `--family`, `--model-path` (files the user already has),
   `--data-dir`, `--context 32k|64k|128k`, `--vision off|on|gpu|cpu`, `--parallel N`, `--gpu 0,1`,
   `--access local|lan|cloudflare|tailscale`, `--port`, `--api-key-env NAME`, `--new-key`, `--save-preset NAME`,
   `--output json`. `strata launch --status` shows what is serving, `strata launch --stop` stops it, and
   `strata launch --show-key` prints the API key.
4. Connect the user's apps: `strata connect` lists them, `strata connect <app> --check` sends one prompt and one tool
   call the way that app does.

Keep the server on `127.0.0.1` unless the user asks for more. Beyond it an API key is recommended: `--access lan` makes one for you, and `--no-api-key` serves without one (a warning, never a refusal; on a public tunnel the warning is stronger). Source setup (`START-HERE.bat`, `setup.sh`) is the fallback for a developer or for a system with no release package; it is
documented in [docs/SOURCE_INSTALL.md](docs/SOURCE_INSTALL.md). Those wrappers exist in a source checkout, not in the
released app. Do not tell an app user to find a `.bat` file.
Source setup does not make an unsupported model or backend compatible. Strata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md)) offers the same steps as tools.

## Working on the code

- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- The Terminal: [docs/TERMINAL.md](docs/TERMINAL.md) and [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md). The app,
  the packages and the update feed: [packaging/README.md](tools/packaging/README.md) and [packaging/UPDATES.md](tools/packaging/UPDATES.md).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md);
  Mac (Metal): [docs/MAC.md](docs/MAC.md).
- Tests: the Terminal and Control tests are in `terminal/tests/terminal` (`python -m pytest terminal/tests/terminal`). The app's UI
  tests run with `npm test` in `app/host`. Setup's own tests run without a GPU or downloads:
  `python tools/test_setup_<name>.py` (for example `terminal/tools/test_setup_amd.py`, `terminal/tools/test_setup_choices.py`).
- Keep the docs' style: plain words, short sentences, written for non-technical users first. Give measured numbers
  with what they were measured on. Make no claim without a measurement.

## Contributing a change or report

The rules are in **[CONTRIBUTING.md](.github/CONTRIBUTING.md)**. Read it before you open an issue or a pull request, and
follow it whether a person or an agent writes the text. The short version:

- Start every issue and pull request title with its area: `[Engine]`, `[GUI]`, `[TUI]`, `[Benchmark]`, `[Docs]` or
  `[Other]`, for example `[Engine] fix …`. A check on GitHub fails a pull request whose title has no area.
- Search the open and closed issues and pull requests first, and add to a thread that already covers your point.
- Agree the scope in an issue before a large feature, a refactor or a change to a default.
- One problem per pull request. Lead with the problem and the result, then the approach, then the validation
  (passed, failed and not run, kept apart) and the limits.
- A new feature is opt-in, and the default path stays byte-identical to the last release. Say how you checked.
- Build every backend a file touches (CUDA, HIP, SYCL) before asking for review.
- Change a default only where you measured it faster, and show the numbers with what they were measured on.
- Never submit measurements, logs or reproduction steps you did not produce. Mark a finding from reading code as
  unconfirmed static analysis.
- A report from hardware the maintainers do not have is welcome. Follow
  [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md), compare against a same-day run of the build you are
  testing, and say what you did not test.
- Open test requests and the hardware that is wanted are listed in [docs/TEST_REQUESTS.md](docs/TEST_REQUESTS.md).

## Pancake fork additions (this repository)

The rules below apply to the pancake additions only; everything above is upstream's file, unchanged.

### Documentation rules (owner, 2026-11-15) - read first, apply always

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

### Flag testing protocol (owner, 2026-11-20) - applies to every flag test

Why: on 2026-11-19/20 flags were dropped on one noisy run, on text differences and as part of a bundle (ledger T-1002,
T-1003). These rules came out of that. Each flag is judged on its own numbers.

1. **Individual, never batches.** One arm per flag (`minus-X` for a flag the preset sets, `plus-X` for an off-by-default
   one; a flag whose preset value equals the engine default still gets the arm that flips it). A bundle result (whole
   preset, spec 250 `updated`, a whole-arm gate) never adds or removes a flag; it only triggers per-flag arms.
2. **Screen = run 1 of every flag**, one rep, cases short, edit and long 64K (prompt and decode). Blocks of at most 8
   arms: base before, arms, base after; one warm-up start per slice; drift judged on the LAST request of each case of
   the counted base runs; drift > 5% = one re-run of the block, then INCONCLUSIVE (never DROP or KEEP). The box slows
   during long runs (memory compaction, `kcompactd`): never judge arms against a base from the other end of a 1.5 h run.
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
11. **Owner order 2026-11-21:** `--lookup-max-t`, `--adapt-every` and `STRATA_PIN_BY_NEED` are removed from the shipped default configs (optional / not default, `optional` in the tuned-flags file; T-1020); start presets are default and turbo (`flashnext-iq2_xs-turbo-hip.json`, a DRAFT until the turbo-build step at the end of the queue), both raced against stock at the end.

### Box operations (Linux box 192.168.100.82, RX 7900 XT)

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

### Branches

- `main` (this branch) = `updated-strata` (upstream Strata v0.2.16 plus our hand-ported commits) plus `MOD_PLAN.md`, `TEST_LEDGER.md`, `LEADS.md`, `plan.html`, `README.md` (stock-against-updated comparison) and `README-PANCAKE.md`. It holds no `bench/results/`: ledger archive paths resolve on `port/rdna3-decode`. Only normal fast-forward commits, never force.
- `port/rdna3-decode` is the work line: the full pancake tree with the raw data in `bench/results/`. New measurement work starts there.
- `updated-strata` is the upstream-Strata port. `updated-strata-rdna3` is the Strata-rdna3 port, published and paused (owner, 2026-11-20). `NetworkPancake` is never deleted.

### Layout of the additions

Paths on this branch (`engine/` and `terminal/` are upstream's layout):

- `tools/pancake/box_chains/` - the box launchers; on main only `bench-sched.sh`, `bench-updatedAA.sh` and `bench-exact.sh`. The sweep and queue scripts (`sweep.py`, `sweep_candidates.json`, `bench-sweep.sh`, `bench-retest.sh`, `bench-final-us.sh`, `run-queue-us*.sh`) and `tools/pancake/apply_triage.py` live on `port/rdna3-decode` only.
- `tools/pancake/flag_diag.sh`, `flag_diag.py`, `diag_group.py` - per-flag diagnostics; `tools/pancake/host_tune.sh`, `host_tune_ab.sh`, `os_tune.sh` - host and OS tuning checks.
- `terminal/data/tuned-flags.json` and `terminal/strata_control/tuned_flags.py` - the shipped tuned flags and their loader.
- `terminal/serve/status_page.py` and `terminal/serve/web/status.html` - the status page.
- `README-PANCAKE.md` - the contribution text; `MOD_PLAN.md`, `TEST_LEDGER.md`, `LEADS.md`, `plan.html` at the root.
