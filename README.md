# pancake-strata

pancake-strata is a performance fork of [Niko1221/Strata](https://github.com/Niko1221/Strata) for one machine class: an AMD Radeon
RX 7900 XT 20 GB (gfx1100) on Linux with ROCm, serving one user's coding agent. It adds measured kernels, settings and tools on top of
the upstream engine, terminal and app, and a benchmark suite that decides what stays.

**This `main` branch is `updated-strata`:** a snapshot of the upstream release **Strata v0.2.16** with the fork's value-bringing changes
ported on top by hand (one commit per change, every change default off, each commit message names its ledger ids), plus the fork's
tracking documents. Every number in this file has a test id (`T-nnn`) in [TEST_LEDGER.md](TEST_LEDGER.md), and every test row names its
hardware, model, settings, repetitions and raw-result archive. A number without an id is not in this file.

## Branches

| branch | what it is |
|---|---|
| `main` (this one) | upstream Strata v0.2.16 + the pancake ports (= `updated-strata`) + the tracking documents and this README |
| [`updated-strata`](https://github.com/rocktboy1982/pancake-strata/tree/updated-strata) | the same code without the tracking documents; written as a pull request to upstream ([README-PANCAKE.md](README-PANCAKE.md)) |
| [`updated-strata-rdna3`](https://github.com/rocktboy1982/pancake-strata/tree/updated-strata-rdna3) | the same kind of port onto [nicejunjie/Strata-rdna3](https://github.com/nicejunjie/Strata-rdna3) `main` (`8b01b0c85`), with its own README-PANCAKE.md |
| [`port/rdna3-decode`](https://github.com/rocktboy1982/pancake-strata/tree/port/rdna3-decode) | the fork's work line: the full history, the benchmark harness (`tools/pancake/`), specs and raw results (`bench/`) |
| `main-backup-2026-11-20` | the previous `main` (before it became `updated-strata`), kept unchanged |

## Where things are

| file | what is in it |
|---|---|
| [MOD_PLAN.md](MOD_PLAN.md) | status, decisions, what the daily preset runs, the GPU queue, what was dropped (section 5 has the 2026-11-19 benchmark summary) |
| [TEST_LEDGER.md](TEST_LEDGER.md) | one row per test ever run (ids `T-nnn`): date, hardware, settings, repetitions, numbers, exact or lossy, verdict, archive |
| [LEADS.md](LEADS.md) | ideas not ported yet (other forks, upstream pull requests, papers), with authors |
| [plan.html](plan.html) | a generated, filterable view of the plan and the ledger (open it in a browser) |
| [README-PANCAKE.md](README-PANCAKE.md) | the detailed description of every ported change: what it does, files, switches, measurements, credits |

The raw results named by the ledger rows (`bench/results/...`, `pancake-docs/research/bench/...`) live on the
[`port/rdna3-decode`](https://github.com/rocktboy1982/pancake-strata/tree/port/rdna3-decode) branch.

## Measured comparison

How these were measured. Machine: RX 7900 XT 20 GB (gfx1100), AMD Ryzen 7 5700G, 58 GB RAM, PCIe 4.0 x16, ROCm 10.0, Linux. Model:
Qwen3.8-Flash-Next, `GSQ-RCO-IQ2_XS` GGUF, with its MTP draft head and `--spec 4`. Harness: `tools/pancake/abtest.py` of the work line,
one engine at a time, plain and updated builds made the same day. "short" is a short chat answer, "edit" a code rewrite (both decode
tok/s); the long cases are one cold prompt of the given size followed by a 256-token answer. Values are mean +- 95 % interval; n is the
number of measured requests per case. "Screen" is the quick first pass (one engine start per arm); "confirm" is the repeated pass
(two starts x three passes) that decides.

**Noise warning.** Every start on 2026-11-19 was flagged noisy by the harness: other CPU load of 7-23 % in the upstream runs and 21-58 %
in the Strata-rdna3 runs, with the kernel's memory-compaction thread `kcompactd0` on top. The screens keep the quietest of three noisy
attempts. Only the confirm rows carry a usable interval; the long-prompt rows are single requests and show a direction, not a size.

### Plain upstream Strata v0.2.16 against updated-strata (this branch)

Plain = the release built with its documented HIP command, started with the model arguments only. Updated = this branch built with
`tools/pancake/build_updated.sh`, started with `configs/pancake/flashnext-iq2_xs-updated-hip.json` (the fork's daily preset, adapted).

Lossy arm (the full preset; its text differs from plain):

| case | plain v0.2.16 | updated-strata (lossy) | difference | n, pass | ledger |
|---|---|---|---|---|---|
| short, decode tok/s | 69.1 +-5.4 | 102.0 +-4.4 | **+47.6 % +-11.0** | 6 / 6, confirm | T-991 |
| edit, decode tok/s | 83.7 +-2.5 | 177.7 +-6.2 | **+112.4 % +-9.6** | 6 / 6, confirm | T-991 |
| 10K-token prompt, prompt / decode tok/s | 608 / 67.8 | 1,086 / 66.7 | +78.6 % / -1.6 % | 1, screen | T-992 |
| 32K-token prompt, prompt / decode tok/s | 688 / 67.4 | 1,580 / 100.5 | +129.7 % / +49.1 % | 1, screen | T-992 |
| 64K-token prompt, prompt / decode tok/s | 680 / 52.4 | 2,400 / 63.1 | +252.9 % / +20.4 % | 1, screen | T-992 |
| 100K-token prompt, prompt / decode tok/s | 660 / 52.0 | 1,996 / 89.0 | +202.4 % / +71.2 % | 1, screen | T-992 |

Across all three noisy attempts of the long screen the updated prompt speed stayed above plain at every size (1,086-1,575 against
605-608 tok/s at 10K, 2,400-2,614 against 676-681 at 64K; T-992).

Exact arm (the preset without its lossy switches; it keeps `--lookup-max-t 12`):

| case | plain v0.2.16 | updated-exact | difference | n, pass | ledger |
|---|---|---|---|---|---|
| short, decode tok/s | 74.1 +-39.4 | 101.8 +-62.9 | +37.4 % +-31.7 | 2 / 2, screen | T-994 |
| edit, decode tok/s | 86.1 +-99.8 | 135.8 +-195.1 | +57.8 % +-110.8 (no difference) | 2 / 2, screen | T-994 |
| 10K-token prompt, prompt tok/s | 542 | 1,294 | +138.7 % | 1, screen (run cut off) | T-996 |
| 32K-token prompt, prompt tok/s | 690 | 2,035 | +194.9 % | 1, screen (run cut off) | T-996 |
| 64K-token prompt, prompt tok/s | 680 | 1,934 | +184.4 % | 1, screen (run cut off) | T-996 |
| 100K-token prompt, prompt tok/s | 662 | 1,831 | +176.6 % | 1, screen (run cut off) | T-996 |

Quality gate (token forcing, `quality_gate.py`: 48 requests, 9,410 forced tokens, plain run twice with 1.562 % and 1.360 % mismatch; T-995):

| arm | mismatch | difference, 90 % interval | rate rule (limit 2.625 points) | event rule | verdict |
|---|---|---|---|---|---|
| updated-exact | 3.379 % | +1.92 points [+1.23, +2.74] | REJECT | PASS | **REJECT** |
| updated (lossy) | 3.964 % | +2.50 points [+1.67, +3.43] | REJECT | PASS | **REJECT** |

What this means. updated-strata is clearly faster than the plain release on this machine: about one and a half times on short answers,
about twice on code edits (confirmed), and about 1.8 to 3.5 times faster at reading long prompts (single requests). But neither
preset passes the token-forcing gate: both change the output more than the release changes against itself. By the fork's own rules
neither preset may be a default until a variant passes, and the lossy `--lookup-max-t 12` is already out of the shipped tuned flags.
Note: on this model the plain release run twice also gives different text, so "identical text" cannot tell exact from lossy here.

### Plain Strata-rdna3 against updated-strata-rdna3

Plain = Strata-rdna3 `8b01b0c85` built with its documented flags and started with its own setup config `rdna3/configs/iq3-1gpu.json`
exactly (only the model is the box's IQ2_XS file; `--mtp-q4 proj` because this model's draft head is stored below 4.5 bits, T-987).
Updated = [`updated-strata-rdna3`](https://github.com/rocktboy1982/pancake-strata/tree/updated-strata-rdna3) with the same config plus
only the pancake settings plain does not set. Updated-exact = updated without `STRATA_VERIFY_ECAP` (its one lossy addition); both arms
keep Strata-rdna3's own lossy defaults, so "exact" is relative to plain.

Confirm, 2 starts x 3 passes (other CPU load 29-58 % in every start; T-999):

| case | plain Strata-rdna3 | updated-exact | updated (lossy) | n | verdict |
|---|---|---|---|---|---|
| short, decode tok/s | 82.6 +-9.9 | 87.5 +-6.1 (+5.3 % +-16.2) | 88.6 +-9.7 (+9.9 % +-16.2) | 5 / 6 / 3 | no difference |
| edit, decode tok/s | 87.7 +-10.3 | 93.8 +-4.9 (+6.1 % +-12.7) | 119.0 +-53.9 (+35.9 % +-36.8) | 5 / 6 / 3 | no difference |

Screens (quietest attempt, direction only):

| case | plain | updated-exact | updated (lossy) | n, spec | ledger |
|---|---|---|---|---|---|
| short, decode tok/s | 70.3 +-10.2 | 112.0 +-56.6 (+59.2 %) | 113.6 +-25.4 (+61.6 % +-50.6) | 2, spec 252 | T-999 |
| edit, decode tok/s | 84.1 +-160.1 | 109.5 +-280.9 (+30.2 %) | 118.1 +-188.7 (+40.4 % +-34.0) | 2, spec 252 | T-999 |
| 32K-token prompt, prompt tok/s | 1,072.9 | 1,604.5 (+49.5 %) | 1,592.6 (+48.4 %) | 1, spec 252 | T-999 |
| short, decode tok/s | 93.6 +-10.2 | not run | 73.9 +-68.0 (-21.0 %, no difference) | 2, spec 250 | T-997 |
| edit, decode tok/s | 96.4 +-114.4 | not run | 141.9 +-617.1 (+47.3 %, no difference) | 2, spec 250 | T-997 |
| 10K / 32K / 64K / 100K prompt, prompt tok/s | 1,442 / 1,901 / 2,222 / 2,413 | not run | 940 / 1,273 / 2,188 / 2,486 (-34.8 / -33.0 / -1.5 / +3.0 %) | 1, spec 250 | T-997 |

Task gate (`task_gate.py`, 18 tasks: 8 code edits, 7 tool calls, 3 questions; 5 repetitions, temperature 0.2, plain run twice; T-1000):
plain 180 of 180 (100.0 %), updated 90 of 90 (100.0 %), tool-call validity 100 % for both, difference +0.0 points: **PASS**. Caveat: the
same task set also sat at 100 % on the fork's own line (T-819), so these tasks may be too easy to separate the arms.

What this means. On Strata-rdna3, which already carries the RDNA3 kernels the fork once took from it, the pancake additions are **not
shown faster**: the confirm finds no difference that this noisy machine can resolve, and two screens of the same plain config disagree
by 1,901 against 1,073 tok/s on the same 32K prompt. The task gate finds no quality loss. A quiet-machine rerun is the open item
(MOD_PLAN.md section 5).

## What this branch adds over Strata v0.2.16

Every switch is off by default and its unset path is v0.2.16's code. Details, files and the full evidence per item: README-PANCAKE.md.

| item | what it does | evidence (ledger) | credit |
|---|---|---|---|
| `STRATA_RDNA_Q6K` | multi-column Q4_K / Q5_K / Q6_K / IQ4_XS decode GEMVs for gfx11: weights decoded once for 4-16 verify columns | micro benchmark +46..+148 % at 4-16 columns (T-079); end to end edit +4.3 % +-5.0, inside noise (T-081); bundle T-009 | nicejunjie (Strata-rdna3), extended to 16 columns by pancake-strata |
| `STRATA_GR_RDNA` (+ `_Q8`, `_A8`, `GR_INJ_UP`, `GR_FUSED`) | hyper-connection read kernels for wave32 gfx11 | RDNA kernel bundle: decode 87-101 -> 104-108 tok/s (T-009); preset round T-378 | nicejunjie (Strata-rdna3) |
| `STRATA_RDNA_MMVQ_WAVE=3` | exact wave32 GEMV mode, one kernel for 1-16 columns | mode 1 was 4.5 % slower than mode 3 on short (T-072); bundle T-009 | nicejunjie (Strata-rdna3) |
| `STRATA_COMBINE_CPU`, `STRATA_WAIT_FUSE` | CPU expert rows combined straight from the mapped buffer; plan wait + copy in one launch | 21.85 -> 21.60 ms per window (T-009); fused wait off lost (T-077) | nicejunjie (Strata-rdna3) |
| `STRATA_ROUTE_PUBLISH` | router top-10 and doorbell publish in one launch | 21.60 -> 21.51 ms per window, 0.4 % (T-009) | nicejunjie (Strata-rdna3) |
| `STRATA_POOL_FUSE_PHASES` | CPU expert pool runs gate/up and down as one phase | edit +2.5 % +-1.7 (T-224) | pancake-strata |
| `STRATA_VERIFY_ECAP` + `_MIN_ROWS` (lossy) | caps CPU-missed experts per draft row in long verify windows | edit +41.7 % +-5.6 (T-588); token forcing PASS (T-589); task gate PASS (T-591), both with the fork's grammar guard that v0.2.16 lacks | idea from jmnargi (Strata-V100, upstream PR 3545); per-row form by pancake-strata |
| hipBLASLt table selected | `STRATA_HIPBLASLT_TUNING` = upstream's `gfx1100-hipblaslt-100401.txt`; matches the ROCm 10.0 runtime (100401); a 100200 table falls back to hipBLASEx (T-998 engine logs) | 11.7K prompt 1,345 -> 1,850-1,920 tok/s, provisional (T-096) | table by miskahm (upstream) |
| learned expert profile | `data/expert-profile-pancake.bin` for the Flash-Next IQ2_XS files (`--expert-profile`) | +6.7 % decode, measured on an RTX 3060, not on this card (T-194) | pancake-strata |
| adopted option values | `--pcie-frac 0`, `--adapt-every 1`, `--adapt-swaps 32`, `--conversation-cache-mib 2048 --conversation-cache-min-tokens 8192` (in the updated preset) | `--pcie-frac 0` +6 % short / +13 % edit, one run (T-197, T-268); the adapt values show no gain without it (T-199); conversation cache -21.6 % +-14.8 prompt time on a recorded session (T-183) | pancake-strata |
| tuned flags at the normal start | measured flags for the detected hardware added at every start (below) | rule: exact >= 2 % or lossy >= 3 % with a passing gate | pancake-strata |
| `tools/pancake/flag_diag.sh` | per-flag "can it run here, does it help here" (below) | 2026-11-19 runs T-993, T-1000 | pancake-strata |
| status page `/status` | project, hardware, tuned flags, settings, diagnostics and live speed in one page (below) | GUI, no speed claim (T-989) | pancake-strata |
| host tuning scripts | `host_tune.sh`, `host_tune_ab.sh`, `os_tune.sh`, `pcie_link.py`: reversible Linux host settings | no gain claimed | pancake-strata; settings list from bramvanoploo/strata-performance-setup (no code copied) |
| optimised build | `tools/pancake/build_updated.sh` | the whole build is measured in the tables above | pancake-strata |

## Tuned flags at the normal start

Every normal start of the terminal, `runtime start` and the app run `apply_saved_tuning`, which first calls
`terminal/strata_control/tuned_flags.py`. It picks a profile by graphics architecture, ROCm version and VRAM size and adds its flags to
the run. Your own setting always wins (a variable in the config's `env` or in the environment, an argument already in `args`).
`STRATA_TUNED_FLAGS=0` or `--no-tuned-flags` gives plain upstream behaviour; unknown hardware keeps the plain defaults and prints one line.
Only exact flags with >= 2 % or lossy flags with >= 3 % and a passing quality gate are accepted, also from a hand-edited file.
The shipped profile (`terminal/data/tuned-flags.json`, gfx1100, ROCm 10.x, 16-24 GiB) holds five flags: `STRATA_HIPBLASLT_TUNING`
(100401 table, T-096), `STRATA_RDNA_Q6K=1` (T-081), `STRATA_GR_RDNA=1` and `STRATA_RDNA_MMVQ_WAVE=3` (bundle T-009) and `--pcie-frac 0`
(T-197). None of the five has a confirmed single-flag verdict on this build yet; their exactness is judged by parity ctests, not by
identical text (T-990). A file written by `flag_diag.sh --write-recommended` (`~/.strata/tuned-flags.json` or `$STRATA_TUNED_FLAGS_FILE`)
is read first.

## Per-flag diagnostics: `tools/pancake/flag_diag.sh`

Read-only on the system (no sudo). `flag_diag.sh hw` prints the gfx architecture, ROCm, VRAM, PCIe link, CPU, RAM and hipBLASLt;
`flag_diag.sh list` the candidate flags with their exact / lossy status and ledger id; `flag_diag.sh run CONFIG.json OUT_DIR` starts the
engine once per flag next to the default (through the work line's A/B harness, `FLAG_DIAG_ABTEST=path`) and gives each flag a verdict:
GAIN, NO-GAIN, UNSUPPORTED (cannot run here), FAIL (its parity ctest failed, `--ctest BUILD_DIR`) or UNTESTED. A flag claimed exact whose
text differs gets one repeat of the default first; if the default also differs, the flag keeps its speed verdict as "EXACT-UNVERIFIED".
`flag_diag.sh --write-recommended OUT_DIR/verdicts.json` writes a tuned-flags profile from the verdicts.

## Status page: `/status`

Open `http://127.0.0.1:<port>/status` (or `/pancake`) in a browser; `GET /status.json` returns the same data. It shows the checkout's
branch and head, engine version and build, the hardware, the tuned flags with each flag's state at start (applied, yours, changed, not
applied), your settings, the flag_diag verdicts when a verdicts file is named (`$STRATA_FLAG_DIAG_VERDICTS`), and live counters: decode
and prompt speed (now, run average, last request), prefix reuse, expert cache hits, PCIe share, drafts accepted, the conversation cache,
GPU and RAM. Read-only, drawn on the server, works without JavaScript.

## Build and run

Build the engine (Linux, ROCm 10.0, gfx1100; the script's header justifies every flag):

```sh
tools/pancake/build_updated.sh                 # this checkout -> build-updated/ ; BACKEND=cuda for an NVIDIA build
ROCM=/opt/rocm GFX=gfx1100 tools/pancake/build_updated.sh . build-hip
```

Then either use it through the upstream terminal (`strata components add-build <build folder>`, then `strata launch`; see
`docs/SOURCE_INSTALL.md`), or start the benchmark preset `configs/pancake/flashnext-iq2_xs-updated-hip.json` with the work line's harness
(`<REPO>`, `<STRATA_DATA>` placeholders; `exe` = `<REPO>/build-hip/strata`). Remember the gate result above before using the lossy preset
for real work. The upstream installers and releases (`Niko1221/Strata`) do not contain these changes; build from source.

## Credits

Licence: MIT, like upstream (`LICENSE`: Copyright (c) 2026 Niko1221 and the Strata contributors).

- **Niko1221** and the Strata contributors: the engine, terminal and app this branch is built on.
- **nicejunjie** (Junjie Li) and the Strata-rdna3 contributors: the RDNA3 wave32 kernels (`RDNA_Q6K`, `GR_RDNA*`, `RDNA_MMVQ_WAVE`,
  `COMBINE_CPU`, `WAIT_FUSE`, `ROUTE_PUBLISH`); commits are listed in README-PANCAKE.md section 5.
- **jmnargi** (Strata-V100, MIT; upstream PR 3545): the verify-window expert cap that `STRATA_VERIFY_ECAP` adapts.
- **miskahm** (upstream commit `7984688e3`): the gfx1100 hipBLASLt 100401 table.
- **bramvanoploo** (strata-performance-setup, MIT): the list of host settings behind `host_tune.sh` (no code copied).
- **jesdga95** (basalt, MIT), **merbanan** (upstream PRs 4040, 4039) and Niko1221's commits `9d3c8c366`, `03627326d`: the grammar path,
  the 16-row windows and `--lookup-max-t`, ported onto `updated-strata-rdna3`.
- **pancake-strata** (rocktboy1982): the pool phase fusion, the per-row ECAP form and its window gate, the learned profile, the tuned
  flags, flag_diag, the status page, the host scripts, the build script and the measurements.

The work line credits every other project it studied or took from (Strata forks, upstream pull requests, papers) in section 9 of the
[`port/rdna3-decode` README](https://github.com/rocktboy1982/pancake-strata/blob/port/rdna3-decode/README.md) and in [LEADS.md](LEADS.md).
Ideas only, no code, were taken from AGPL projects (jdominick05/caminus); no AGPL code is in this branch.
