# updated-strata-rdna3: measured speed work from pancake-strata, ported onto Strata-rdna3

This branch carries a small, selected set of changes from the pancake-strata fork (rocktboy1982/pancake-strata, branch `port/rdna3-decode`) on top of
the head of [nicejunjie/Strata-rdna3](https://github.com/nicejunjie/Strata-rdna3) `main` (`8b01b0c8512d388552e8601b3aff6753d05286e0`, 2026-11-15; the
repository has no tags and no releases). It is written as the description of a pull request: what each piece does, how to switch it on, which files it
touches, what was measured and where the raw results are, who the ideas come from, and what is not proven yet.

Publication note: this branch is published from a clone whose push token cannot write `.github/workflows`, so its first commit is a snapshot of that head without the
workflow files, and the commits above it are plain patches against the head: `git format-patch` them and `git am` them onto the real `nicejunjie/Strata-rdna3`
`main`, or cherry-pick them, and they apply unchanged.

The two repositories started from the same Strata source and Strata-rdna3 is where our RDNA3 kernels came from, so this is a return. The pancake line has
3,317 commits since the fork point and Strata-rdna3 has 1,025: a merge was not attempted. Every piece below was ported by hand, adapted to this tree, and
kept only where the pancake line measured a gain (at least 2% end to end for an exact change, at least 3% for a lossy one that passed its quality gate) or
fixed a real defect. Everything the pancake line dropped, left unmeasured or never wired was left out.

All numbers below were measured on the pancake line: RX 7900 XT 20 GB (gfx1100), Ryzen 7 5700G, ROCm 10.0, model Qwen3.8-Flash-Next (GSQ-RCO IQ2_XS),
greedy decoding unless stated. They are not yet measured on this branch. Section 4 holds the benchmark of this branch against plain Strata-rdna3; its
tables are marked TO BE MEASURED until the box run exists. The ledger rows (`T-nnn`) are rows of `TEST_LEDGER.md` on the pancake line.

## 1. What this branch adds

| capability | how to use it | files |
|---|---|---|
| tuned flags at startup: the normal start (`rdna3/run.py`) exports the measured flags for the detected GPU | `python rdna3/run.py iq4-1gpu --data ...` (default on); `--no-tuned-flags` or `STRATA_TUNED_FLAGS=0` for plain behaviour; a variable you export yourself always wins | `rdna3/run.py`, `rdna3/tuned_flags.py`, `rdna3/recommended-flags/gfx1100.json`, `rdna3/test_tuned_flags.py` |
| per-flag diagnostics: "can this flag run here, does it help here?" | `tools/pancake/flag_diag.sh --hardware`, `--list`, `run --config CFG.json`, `--write-recommended` | `tools/pancake/flag_diag.sh`, `flag_diag.py`, `test_flag_diag.py` |
| host tuning scripts (reversible, `--check` is read only) | `tools/pancake/host_tune.sh --list / --check / --apply / --revert`; `os_tune.sh`; `pcie_link.py` | `tools/pancake/host_tune.sh`, `host_tune_ab.sh`, `os_tune.sh`, `pcie_link.py` and their tests |
| optimised build for gfx1100 | `tools/pancake/build_updated_rdna3.sh SRC BUILD_DIR [--print-flags]` | `tools/pancake/build_updated_rdna3.sh` |
| benchmark of plain against updated on the box | `tools/pancake/box_chains/bench-updatedRD.sh` and spec `bench/specs/250-rdna3-updated-vs-plain.json` | `tools/pancake/box_chains/` |
| engine changes (section 2): pool phases, ECAP, grammar, two fixes, F16 value; optional table and profile | environment switches and `--grammar on`, default off | see section 2 |

How the tuned flags work. `rdna3/tuned_flags.py` detects the GPU (`rocm_agent_enumerator`, ROCm version, VRAM from sysfs), finds the most specific file
`rdna3/recommended-flags/<gfx>[-rocm<M.m>].json` whose key matches, and puts its `env` and `args` into the server config before the engine starts. Order of
precedence: your exported environment, then the file, then the setup's own config (`rdna3/configs/*.json`); an argument the config already carries is
never changed. On hardware with no matching file the setup runs unchanged and one line says so. Only a measured GAIN enters a file: exact (bitwise equal
text) at 2% or more, or lossy at 3% or more with a passing quality gate. `flag_diag.sh` prints OK, UNSUPPORTED, NO-GAIN, GAIN or FAIL for each candidate
and writes the file for the hardware; `--write-recommended` copies the verdicts of the last run into `rdna3/recommended-flags/`. The shipped file for
gfx1100 holds the two ported flags that have a measured gain on the pancake base (`STRATA_POOL_FUSE_PHASES=1`, `STRATA_PF_F16=1`) and is to be regenerated
from the benchmark of section 4.

The tuning scripts are ported from the pancake line unchanged, except that `host_tune_ab.sh` takes the A/B harness paths from `AB_ABTEST`, `AB_LOADTIME`
and `AB_SPEC` (the harness is not part of this tree). The list of host settings was taken from reading
[bramvanoploo/strata-performance-setup](https://github.com/bramvanoploo/strata-performance-setup) (MIT); no code was copied (section 5).

## 2. The ported enhancements

### 2.1 `STRATA_POOL_FUSE_PHASES` (exact, default off)
What: the CPU expert pool runs a layer's gate/up tasks and down tasks in one phase; a down task waits only for its own expert, whose last gate/up task
has quantized its input. The row ranges and kernels are those of the two-phase path, so the output is identical. Enable: `STRATA_POOL_FUSE_PHASES=1`.
Files: `src/kernels/cpu/pool.cpp`, `include/strata/kernels/cpu/pool.hpp`. Measured: code edit +2.5% +-1.7 over the two-phase path, 3 repeats, short chat
neutral (T-224; `bench/results/2026-11-09-test-phase/03b-fuse-min.summary.txt`, `03-cpu-pool.summary.txt`, quality `03b-fuse-min.quality.txt`: gate PASS). The first
version had a defect found by the first test (a stale task count cut the down rows wrongly or divided by zero); the version here has the fix.

### 2.2 `STRATA_VERIFY_ECAP`, `STRATA_VERIFY_ECAP_MIN_ROWS` (lossy, default off)
What: in a decode verify window each draft row (rows 1 to T-1, never row 0, whose output is always committed) reads at most n CPU-resident experts that row 0
and the earlier rows do not read anyway, kept by routing rank; the other draft-only experts are skipped like `STRATA_ROUTE_TAIL_SKIP` experts (neither copied nor
computed, a zero row, routing weights as routed). `STRATA_VERIFY_ECAP_MIN_ROWS=N` lets the cap act only in windows of N rows or more. Host side only; no captured
graph changes. Enable: `STRATA_VERIFY_ECAP=0` with `STRATA_VERIFY_ECAP_MIN_ROWS=<N>`.
Files: `include/strata/core/verify_ecap.hpp` (the rule), `src/core/verify_ecap_test.cpp`, `src/core/expert_source.cpp` and `.hpp`, `src/core/verify.cpp` and
`.hpp`, `src/program/generate.cpp`, `CMakeLists.txt`. Test: `verify_ecap_test` (4,000 random windows against the rule's promises; built and passed on a PC).
Measured on the pancake base (6 clean runs per arm): cap 0 from 9-row windows, edit 165.3 +-6.2 tok/s = +41.7% +-5.6 over no cap, short +2.8% +-2.6 (T-588,
`/home/dan/auto-runs/60b.jsonl` on the box); token forcing PASS: mismatch 1.583% against 1.467% and 1.722% for two stock runs, difference -0.011 points
(T-589, `bench/quality-hip/2026-11-15-ecap-gated`); tool-call task gate PASS: 99.4% success, tool-call validity 100% over 180 runs (T-591,
`bench/results/2026-11-16-taskgate-ecap`). An ungated cap 0 was rejected by the same gate (+0.95 points, T-271).
Two limits stated plainly. First, the gain was measured on windows of 9 or more rows (prompt-lookup windows of the pancake line); this tree's windows are at most
8 rows, so the 9-row gate never opens here. It is therefore not in the shipped flags file; spec 250 has an arm with a 6-row gate (the pancake base measured
+36.2% +-17.8 for it, T-588) and the diagnostics decide. Second, the tool-call gate ran with the grammar guard on (section 2.5); this tree has it now, but `--grammar` is off by default, so enable both together (`--grammar on` and the ECAP variables) if tool calls matter.

### 2.3 Two fixes
`sampler.cu` (T-573): a NaN temperature fell through the `t <= 0` test into a uniform draw over the top-k list; `sampler_parity` measured 4 of 4 picks wrong. One
rule at all four sites, `!(t >= kSamplerMinTemperature)` with a floor of 1e-4 (greedy). Normal sampling is unchanged. `elementwise.cu` (T-576): `gdn_gate_kernel` was
bounded by the head count and computed only the first token (96 of 144 values differed from three single-token calls; 0 after the fix); every current caller passes one
token, so decode is not affected. Files: `include/strata/kernels/sampler.hpp`, `src/kernels/cuda/sampler.cu`, `src/kernels/sampler_parity.cpp`,
`src/kernels/cuda/elementwise.cu`, `src/kernels/elementwise_parity.cpp`. Both compile for sm_75 (nvcc 12.8). No archive was kept for T-573 and T-576 (the ledger rows
are the record).

### 2.4 `STRATA_PF_F16=1` as a shipped value (lossy)
The F16 prompt-expert kernels are already in Strata-rdna3; its setup configs set `STRATA_PF_F16` to 2, which leaves them off. The pancake line measured 1: prompt
tok/s +15.1% +-0.4 at 14K, +16.7% +-0.2 at 32K, +17.3% +-0.5 at 64K tokens, no effect at 3K (+0.3%) or on decode (T-820, `bench/results/2026-11-17-chainC/winC-161.*`);
token-forcing gate PASS on rate and event rules (mismatch 1.700% against 1.637% and 1.605%, +0.08 points, 90% interval [-0.075, +0.234], T-828,
`bench/results/2026-11-17-chainQ/winQ-163.stdout`), while the stricter near-tie column failed (98.3% top-1 agreement; reported as information). It enters through
the recommended-flags file, not through the code.

### 2.5 Grammar-constrained decoding, `--grammar on` (default off)
What: a request's `grammar` key (a GBNF tool-call or JSON-schema grammar) is enforced on the sampled tokens inside speculative decoding. After each verify window the first kept
pick the grammar refuses is drawn again under the grammar's mask (`Verifier::resample_row`) and the window ends there; a merged window checks its output row only. The vocabulary is
read from the pack's tokenizer files at start. On the serve side, tool calls and JSON response formats of `/v1/chat/completions`, `/v1/responses` and `/v1/messages` carry the grammar
when the engine runs `--grammar on` (or the request says `strict: true`). `Guard::enforcing()` tells the verifier a tool call or JSON reply is in progress, so ECAP does not cap the draft
rows there. With `--grammar off` (the default) nothing is read, drawn or printed. Enable: `--grammar on`.
Files: `include/strata/program/grammar/{grammar,guard,schema}.hpp`, `src/program/grammar/{grammar,guard,schema}.cpp`, `json_value.hpp` (CPU only, library `strata_grammar`),
`src/program/generate.cpp`, `src/core/verify.cpp` and `.hpp`, `serve/grammar.py`, `serve/server.py`. Tests: `tests/program/grammar_test.cpp`, `grammar_schema_test.cpp`,
`grammar_guard_test.cpp` (CPU; built and passed on a PC), `serve/test_grammar.py`, `serve/test_grammar_routes.py` (7 pass).
Measured on the pancake line (T-315; no raw archive, the ledger row is the record): RTX 3060, Flash-Next IQ2_XS: strict json_schema (enum and required, off-topic prompts too) and a tool call with an
enum argument all valid; 0.3 to 0.8 ms of grammar code per request, 35 to 110 us per 50 to 130 ms window (0.1 to 0.2%). With ECAP the tool-call task gate passed with the guard on: 99.4% success,
tool-call validity 100% (T-591). It is a correctness feature, not a speed one; it was not measured on this tree.

### 2.6 hipBLASLt tuning table 100401 (optional; not in the shipped flags)
This tree already ships `tools/hip/gfx1100-hipblaslt-100401.txt` (byte-identical to the pancake line's table) and the `STRATA_HIPBLASLT_TUNING` switch, and its setup configs use the 100200
table. The pancake line measured the 100401 table only against no table: prompt 1,345 to 1,850 to 1,920 tok/s, +38%, provisional on a noisy machine, no raw file kept (T-096). Against
the 100200 table nothing is measured, so it is a diagnostic candidate: `flag_diag.sh` compares the two (base arm keeps 100200) and spec 251 has the arm `hipblaslt-100401`. It enters
`rdna3/recommended-flags/` only with a measured GAIN.

### 2.7 Learned expert profile (optional; not in the shipped flags)
`--expert-profile` already exists here. `data/expert-profile-pancake.bin` (196,632 bytes, the same `STRP` format as `data/expert-profile.bin`) is the profile the pancake line learned for
Qwen3.8-Flash-Next IQ2_XS: it is specific to that model and to the pancake line's workload, and T-194 measured it on an RTX 3060, not on a 7900 XT (+6.7% decode against another shipped
profile, 2 runs of short and edit, every run faster). Use: `--expert-profile data/expert-profile-pancake.bin`. `flag_diag.sh` (candidate `EXPERT_PROFILE`) and spec 251 (arm `profile-pancake`)
compare it with `data/expert-profile.bin`. A recommended argument replaces the setup config's value, so a measured GAIN can ship through the flags file.

### 2.8 Considered and not ported
Prompt-lookup windows of 12 rows (`--lookup-max-t 12`, T-126 and T-127, edit +8.8% +-6.8) need the window machinery of upstream Strata that this tree lacks; log-only counters; the Speedy
opt-ins `PF_GROUP_SCRATCH` and `PF_HC_REGION` (no HIP measurement); `PREFILL_EQUAL_TAIL` (under 2%); the BF16 router kernels for 7 and 8 columns (micro-benchmark only). Everything the
pancake line dropped or never measured stays out. The first candidate of a round after this one is the 12-row lookup windows.

## 3. Diagnostic tests and how to reproduce
Host checks run for this branch (no GPU needed): `python -m pytest rdna3/test_tuned_flags.py tools/pancake/test_flag_diag.py tools/pancake/test_host_tune.py
tools/pancake/test_pcie_link.py` (fake hardware, fake verdicts, fake sysfs); `verify_ecap_test` (CTest, CPU). The HIP engine files compile only on a Linux
ROCm host (they include POSIX headers): `tools/pancake/build_updated_rdna3.sh SRC BUILD --target strata` builds them, `ctest --test-dir BUILD -R
"verify_ecap_test|sampler_parity|elementwise_parity"` runs the parity tests. On the target machine: `tools/pancake/flag_diag.sh run --config <server config>
--flags POOL_FUSE_PHASES,VERIFY_ECAP,PF_F16,HIPBLASLT_TUNING,EXPERT_PROFILE` (a few minutes per flag; it starts engines only through the A/B harness). Results of a diagnostic
run on the box: TO BE MEASURED.

## 4. Benchmark: plain Strata-rdna3 against this branch (TO BE MEASURED)
Plan: `tools/pancake/box_chains/bench-updatedRD.sh` builds plain Strata-rdna3 `8b01b0c85` with its documented flags and this branch with
`build_updated_rdna3.sh`, runs the parity tests, the flag diagnostics, spec 250 (screen, then 5 repeats in blocks of 3, one engine at a time, text compared with
the plain arm) and spec 251 (screen only: the 100401 table and the learned profile against the updated build's own). Workloads: short chat and code edit (decode tok/s), cold prompts of 10,000 / 32,000 / 64,000 / 100,000 tokens (prompt tok/s). Short plan (screen first, confirm only for arms with a screen benefit, 3 reps, tune-up included): about 1.9 hours of box time, budget 2 hours (`bench-sched.sh` runs it before the rest of the box queue).

| case | plain | updated | change | text identical |
|---|---|---|---|---|
| short (decode tok/s) | TO BE MEASURED | TO BE MEASURED | TO BE MEASURED | TO BE MEASURED |
| edit (decode tok/s) | TO BE MEASURED | TO BE MEASURED | TO BE MEASURED | TO BE MEASURED |
| 10,000 tokens (prompt tok/s) | TO BE MEASURED | TO BE MEASURED | TO BE MEASURED | n/a (unique first line) |
| 32,000 tokens (prompt tok/s) | TO BE MEASURED | TO BE MEASURED | TO BE MEASURED | n/a |
| 64,000 tokens (prompt tok/s) | TO BE MEASURED | TO BE MEASURED | TO BE MEASURED | n/a |
| 100,000 tokens (prompt tok/s) | TO BE MEASURED | TO BE MEASURED | TO BE MEASURED | n/a |

Build flags of the optimised build (each justified in the script header): Release, -O3, gfx1100 only, HIP on and CUDA off, `-mtune=native` (no change of
instruction set, same bits), no fast-math; `-march=native` and LTO stay opt-in until measured.

## 5. Credits and authorship
Outside sources (ideas or code), from the pancake README section 9 and LEADS.md:
- **nicejunjie and the Strata-rdna3 contributors** (MIT): the base of this branch, and the RDNA3 wave32 kernels that the pancake line ported from it in the first place.
- **Niko1221 and the Strata contributors** (MIT, copyright holders of the shared base).
- **jmnargi**, Strata-V100 (MIT), commits `24894e676b`, `7c5041c358`, `376355453f` (upstream PR #3545): the verify-window expert cap that `STRATA_VERIFY_ECAP` adapts, and
  the measurement of how a too-tight cap breaks tool calls. The per-draft-row form that counts only CPU-resident experts is the pancake adaptation.
- **jesdga95**, basalt (MIT), commits `d64768699`, `02e1006fe`, `c7813a6c3`, `011e13d72`: the grammar path (window guard, masked redraw, `--grammar` and the request key); the GBNF engine and the schema builder in it are llama.cpp's (MIT). Adapted and extended on the pancake line.
- **bramvanoploo**, strata-performance-setup (MIT): the list of host settings behind `host_tune.sh`; no code copied.
- **caminus** (jdominick05, AGPL-3.0-or-later): ideas only, none copied; the note that the PCIe link stays awake behind `pcie_link.py`, which is our own code.

Original work of pancake-strata (rocktboy1982), with the record that supports it. Git history of the pancake line: first commit `346709879`, 2026-11-07 17:10 +0200,
author rocktboy1982; `STRATA_POOL_FUSE_PHASES` `35c512984` (2026-11-09) and its fix `50c6464cd` (2026-11-09), both rocktboy1982, ledger T-224; the ECAP adaptation `2d44d31ec`
(2026-11-10, rocktboy1982), ledger T-588, T-589, T-591; the two fixes `5735ae7ac` and `bcfe5ba08` (2026-11-15, rocktboy1982), ledger T-573, T-576; `host_tune.sh`
`cf6726c20` (2026-11-16) and `os_tune.sh` `9ff4bdc4c` (2026-11-10), rocktboy1982; `pcie_link.py` `3d8d516a9` (2026-11-18, local account `iancu`, the same owner). The grammar port is `3d3eef53e` (2026-11-08) with the serve routes `bcbdeb5d2` (2026-11-09), rocktboy1982, ledger T-315; the learned profile is `3638b8cee` (2026-11-08, rocktboy1982), T-194. The
sources of the diagnostics, the tuned-flags startup, the build script and the benchmark chain are new in this branch (pancake-strata, rocktboy1982, original work). The
commits of this branch carry the trailer `Co-Authored-By: Claude Sonnet 5.5`.

## 6. Compatibility and risks
- Everything engine-side is off by default; with no switch set the default path is byte-identical to Strata-rdna3 `8b01b0c85` apart from the two fixes in 2.3
  (NaN temperature, multi-token gate), which do not change normal runs. `--no-tuned-flags` (or `STRATA_TUNED_FLAGS=0`) restores the setup's plain behaviour in the startup script.
- The shipped flags (`STRATA_POOL_FUSE_PHASES=1`, `STRATA_PF_F16=1`) were measured on the pancake base, not on this tree; `STRATA_PF_F16=1` is lossy (gate passed, near-tie
  column failed). Regenerate the file from section 4 before relying on it. `flag_diag.sh` reports a flag UNSUPPORTED on hardware that lacks what it needs (WMMA, VMM, hipBLASLt).
- `STRATA_VERIFY_ECAP` is lossy and was measured with the grammar guard on; it is not enabled by the startup script. `--grammar on` is off by default and was measured for validity on another GPU, not for speed here.
- The learned profile is specific to Qwen3.8-Flash-Next IQ2_XS and was measured on another GPU; the 100401 table was measured only against no table. Neither is in the shipped flags file.
- What was compiled locally: `verify_ecap_test` and the three grammar tests (MSVC), `pool.cpp` syntax, `sampler.cu` and `elementwise.cu` for sm_75. The HIP engine files were not compiled before the box build.
- Licence: Strata-rdna3 is MIT; every ported piece is MIT-compatible (our own code, or MIT sources credited above); AGPL sources were read for ideas only.
