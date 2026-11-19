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
first kept only where the pancake line measured a gain or fixed a real defect. Owner order 2026-11-19 (night): the updated arm must be Strata-rdna3 plus the
WHOLE pancake daily preset (`configs/pancake/flashnext-iq2_xs-fast-hip.json` and its build), so the pancake gains stack on Strata-rdna3's own. Rule: start
from Strata-rdna3's own setup config (`rdna3/configs/iq3-1gpu.json`) and engine defaults; every setting Strata-rdna3 does not have gets the pancake value;
where both have one and the values differ, Strata-rdna3's value stays and the pancake value is a separate spec arm (section 2.15). Everything the pancake line
dropped or never wired stays out.

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
| engine changes (section 2): pool phases, ECAP, grammar, two fixes, 16-row windows and `--lookup-max-t`, RDNA 9-16 column kernels, prompt-path switches, equal tail, persistent adapt worker | environment switches, `--grammar on`, `--lookup-max-t`, build option `STRATA_WINDOW_MAX_T`; the shipped flags file turns on the whole pancake preset (2.15) | see section 2 |

How the tuned flags work. `rdna3/tuned_flags.py` detects the GPU (`rocm_agent_enumerator`, ROCm version, VRAM from sysfs), finds the most specific file
`rdna3/recommended-flags/<gfx>[-rocm<M.m>].json` whose key matches, and puts its `env` and `args` into the server config before the engine starts. Order of
precedence: your exported environment, then the file, then the setup's own config (`rdna3/configs/*.json`); an argument the config already carries is
never changed. On hardware with no matching file the setup runs unchanged and one line says so. Only a measured GAIN enters a file: exact (bitwise equal
text) at 2% or more, or lossy at 3% or more with a passing quality gate. `flag_diag.sh` prints OK, UNSUPPORTED, NO-GAIN, GAIN or FAIL for each candidate
and writes the file for the hardware; `--write-recommended` copies the verdicts of the last run into `rdna3/recommended-flags/`. The shipped file for
gfx1100 now holds the whole pancake preset on top of Strata-rdna3 (section 2.15): `verdicts` lists the entries with a measured gain at or over the bar on the
pancake base, `preset` the rest of the pancake daily preset (enabled as the preset has them, not separately measured), `theirs_kept` the settings where
Strata-rdna3's own value stays. It is to be regenerated from the benchmark of section 4.

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
The gain was measured on windows of 9 or more rows (the prompt-lookup windows of `--lookup-max-t 12` in a 16-row build). This branch now has those windows
(section 2.9), so the shipped file carries the pancake preset's gated form, `STRATA_VERIFY_ECAP=0` with `STRATA_VERIFY_ECAP_MIN_ROWS=9`, together with
`--lookup-max-t 12` and `--grammar on` (the tool-call gate ran with the grammar guard on, section 2.5). Before this round the windows were at most 8 rows and the
gate never opened (the first box screen of spec 250 saw no ECAP effect for that reason).

### 2.3 Two fixes
`sampler.cu` (T-573): a NaN temperature fell through the `t <= 0` test into a uniform draw over the top-k list; `sampler_parity` measured 4 of 4 picks wrong. One
rule at all four sites, `!(t >= kSamplerMinTemperature)` with a floor of 1e-4 (greedy). Normal sampling is unchanged. `elementwise.cu` (T-576): `gdn_gate_kernel` was
bounded by the head count and computed only the first token (96 of 144 values differed from three single-token calls; 0 after the fix); every current caller passes one
token, so decode is not affected. Files: `include/strata/kernels/sampler.hpp`, `src/kernels/cuda/sampler.cu`, `src/kernels/sampler_parity.cpp`,
`src/kernels/cuda/elementwise.cu`, `src/kernels/elementwise_parity.cpp`. Both compile for sm_75 (nvcc 12.8). No archive was kept for T-573 and T-576 (the ledger rows
are the record).

### 2.4 `STRATA_PF_F16=1` (lossy; NOT shipped since 2026-11-19 night, spec arm `updated+pf16-1`)
The F16 prompt-expert kernels are already in Strata-rdna3; its setup configs set `STRATA_PF_F16` to 2, which leaves them off. The pancake line measured 1: prompt
tok/s +15.1% +-0.4 at 14K, +16.7% +-0.2 at 32K, +17.3% +-0.5 at 64K tokens, no effect at 3K (+0.3%) or on decode (T-820, `bench/results/2026-11-17-chainC/winC-161.*`);
token-forcing gate PASS on rate and event rules (mismatch 1.700% against 1.637% and 1.605%, +0.08 points, 90% interval [-0.075, +0.234], T-828,
`bench/results/2026-11-17-chainQ/winQ-163.stdout`), while the stricter near-tie column failed (98.3% top-1 agreement; reported as information). On this tree the
value 1 replaces Strata-rdna3's own F16 prompt path (2), and the first box screen of spec 250 saw the updated arm (then with PF_F16=1) slower on prompts (10K 1110-1462
against 1367-1501 tok/s, preliminary, noisy). Owner rule: Strata-rdna3's PF_F16=2 stays in the shipped file; spec 250 arm `updated+pf16-1` measures 1 against it.

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

### 2.6 hipBLASLt tuning table 100401 (Strata-rdna3's 100200 table stays; spec 251 arm `hipblaslt-100401`)
This tree already ships `tools/hip/gfx1100-hipblaslt-100401.txt` (byte-identical to the pancake line's table) and the `STRATA_HIPBLASLT_TUNING` switch, and its setup configs use the 100200
table. The pancake line measured the 100401 table only against no table: prompt 1,345 to 1,850 to 1,920 tok/s, +38%, provisional on a noisy machine, no raw file kept (T-096). Against
the 100200 table nothing is measured, so it is a diagnostic candidate: `flag_diag.sh` compares the two (base arm keeps 100200) and spec 251 has the arm `hipblaslt-100401`. It enters
`rdna3/recommended-flags/` only with a measured GAIN.

### 2.7 Learned expert profile (Strata-rdna3's profile stays; spec 251 arm `profile-pancake`)
`--expert-profile` already exists here. `data/expert-profile-pancake.bin` (196,632 bytes, the same `STRP` format as `data/expert-profile.bin`) is the profile the pancake line learned for
Qwen3.8-Flash-Next IQ2_XS: it is specific to that model and to the pancake line's workload, and T-194 measured it on an RTX 3060, not on a 7900 XT (+6.7% decode against another shipped
profile, 2 runs of short and edit, every run faster). Use: `--expert-profile data/expert-profile-pancake.bin`. `flag_diag.sh` (candidate `EXPERT_PROFILE`) and spec 251 (arm `profile-pancake`)
compare it with `data/expert-profile.bin`. A recommended argument replaces the setup config's value, so a measured GAIN can ship through the flags file.

### 2.8 Not ported, and why
- `STRATA_AGENT_COUNTERS=1`: a log-only counter of the pancake server's agent loop (`serve/agent_loop.py`, T-513); this tree's server has no such module. No speed effect.
- The GPU router lookahead (T-192, +10%): it runs only where a look-ahead thread exists (a file tier that warms pages, `STRATA_FS_SLOTS` or `STRATA_LOOKAHEAD_STATS`);
  the pancake daily preset runs none of these on the 7900 XT, so the daily preset does not use it there either.
- (Since the 2026-11-19 night round `STRATA_AMD_COMMUNITY_TUNING` IS ported, for the users of other AMD cards: section 2.16.)
- The BF16 router kernels for exactly 7 and 8 columns (micro-benchmark only, not part of the preset).
TO BE MEASURED on this branch: everything in sections 2.9 to 2.15.

### 2.9 Verify windows of up to 16 rows and `--lookup-max-t N` (build option + engine; pancake T-126, T-127)
What: the verify / batch window holds up to 16 rows in the HIP build (CMake `STRATA_WINDOW_MAX_T` = auto | 8 | 16, auto = 16 for HIP and 8 for CUDA;
`include/strata/kernels/window_limits.hpp`), and `--lookup-max-t N` lets prompt-lookup windows grow to N tokens (at most 16) while the MTP keeps `--spec`; each
lookup size is its own window graph (`--lookup-sizes A,B`; a size whose graph finds no VRAM is dropped). Hand port of upstream Strata `9d3c8c366` and `03627326d`
(Niko1221; the latter a port of PR 4040 by merbanan), adapted to this tree's per-size graph sets (the long-context variants), resident plan, BF16 aux rows and
tensor-parallel draft hook; the batch / pipeline parts this tree lacks (`STRATA_PIPED_MTP`, the P4 probe calls) are left out. Windows of up to 8 rows run the
kernels they ran before. Enable: build with `-DSTRATA_WINDOW_MAX_T=16` (the build script does) and run `--lookup-max-t 12`.
Files: `CMakeLists.txt`, `include/strata/kernels/{window_limits,verify_kernels,fused_gr,s2_expert_grouped}.hpp`, `include/strata/core/{verify,mtp}.hpp`,
`include/strata/spec/draft_policy.hpp`, `include/strata/program/pipe_probe.hpp`, `src/core/{verify,mtp,expert_source,peer_experts,remote_experts}.cpp`,
`src/core/remote_expert_opt.cu`, `src/kernels/cuda/{fused_gr,native_bf16,native_mmvq,native_moe,s2_expert_grouped,shared_expert,verify_kernels}.cu`,
`src/kernels/cuda/q8_sm60.cuh`, `src/program/generate.cpp`, `src/spec/draft_policy.cpp`; tests `src/core/window_limits_test.cpp`, `src/spec/draft_policy_test.cpp`
(pass on a PC at 16 and 8 rows). Measured on the pancake line: `--lookup-max-t 12` edit +8.8% +-6.8, quality PASS (T-126, T-127). This is what lets the ECAP gate
of 2.2 open.

### 2.10 RDNA K-quant decode kernels for 9 to 16 columns (pancake T-079, T-080, T-081, T-372)
The RDNA Q4_K / Q5_K / Q6_K / IQ4_XS / IQ3_S multi-column kernels took at most 8 columns, so a 12-row window fell back to the native MMVQ. They take 9 to 16 now
(unroll 1 above 8); `STRATA_RDNA_KQ_MAXCOLS=8` restores the old cap. Pancake: parity 192/192 against native_mmvq (T-372), 1.5-2.5x faster than the fallback at
12/16 columns (T-079), engine +3.7% short, +3.8% edit (T-080), adopted at confirm (T-081). File: `src/kernels/cuda/rdna_mmvq.cu`.

### 2.11 `STRATA_PF_GROUP_SCRATCH`, `STRATA_PF_HC_REGION` (prompt path, default off, bitwise by the authors' checks)
Hand port of upstream `fbf6e1c75`, `3cc47f029`, `c94010648` (Niko1221, port of PR 4039 by merbanan): the MMQ path's gate/up output, SwiGLU and q8_1 copy hold one
expert group's rows (GROUP_SCRATCH; only where every layer takes MMQ, otherwise ignored with a line), and the hyper-connection read's scratch lives in the shared
region (HC_REGION), so `--prefill auto` finds a bigger chunk. Adapted: no `STRATA_MMQ_WS` here; HC_REGION keeps this tree's own-rows sizing and
`STRATA_PF_NO_GATED`. The pancake fault with the fused `PF_COMBINE_WRITE` kernel (T-793) cannot happen here (no such kernel, no layer-major path). In the pancake
preset from Speedy's numbers, no HIP A/B on record. File: `src/prefill/prefill.cpp`.

### 2.12 `STRATA_PREFILL_EQUAL_TAIL` (default off; pancake T-679, T-712)
A prompt segment longer than its chunk whose last chunk would be staged reads in the same number of equal chunks, so every chunk streams (not taken when the evened
chunks would be shorter than 8192 or than the staged limit); one log line per changed plan. The idea is this tree's own `5906629d`; the rule is the pancake line's
`7639dbf76`. Files: `include/strata/program/prefill_plan.hpp`, `src/program/generate.cpp`, `tests/program/prefill_plan_test.cpp` (passes on a PC).

### 2.13 `STRATA_ADAPT_PERSIST` (default on; =0 restores the old behaviour)
With `--adapt-every 1` the serve loop made and joined a thread every window; one persistent worker (this tree's `JobThread`) runs the adaptive tier now, waited for
on every path that joined before. Exact (the same work on another thread). Pancake item 75 (`8780d9e2b`), part of the pancake daily build; no separate A/B on record.
File: `src/program/generate.cpp`.

### 2.14 Switches that already existed here and are set as the pancake preset sets them
`STRATA_PIN_BY_NEED=1`, `STRATA_EXPDEC=1`, `STRATA_EXPDEC_DOWN=1` (T-037, T-040), `STRATA_EXCHANGE_ROTATE=1`, `STRATA_KV_PREFETCH=1`, `STRATA_PROMPT_ATTN_IMMA=1`
(a CUDA sm_80+ kernel; inert on HIP as far as read), `STRATA_ATTN_DAILY=1` (returns false in a HIP build: inert on gfx1100, kept for parity), `--pool-affinity
measure`, `--conversation-cache-min-tokens 8192`. The rest of the pancake env block is already Strata-rdna3's own setup env or gfx1100 engine default with the same
value (section 2.15).

### 2.15 The updated preset: Strata-rdna3 + the whole pancake daily preset
Gap list: the pancake daily preset entry by entry against Strata-rdna3 `8b01b0c85` (its setup config `rdna3/configs/iq3-1gpu.json`, its gfx1100 engine defaults
`src/core/arch_defaults.cpp` `gfx1100_default_env`, its code). "theirs" = Strata-rdna3's value is what the updated arm runs; "ours added" = the pancake value is added.

| entry | Strata-rdna3 | status | updated arm |
|---|---|---|---|
| `GR_RDNA`, `GR_RDNA_Q8`, `GR_RDNA_A8`, `GR_INJ_UP`, `GR_FUSED`, `RDNA_Q6K`, `RDNA_MMVQ_WAVE=3`, `COMBINE_CPU`, `WAIT_FUSE`, `ROUTE_PUBLISH`, `QFUSE`, `VERIFY_QDEDUP`, `HEAD_MIX_MULTI`, `FUSE_CONVAB`, `HIP_WMMA`, `SELECT_WMMA`, `PA_FAST`, `PF_GEMM`, `PF_PAD`, `PF_FUSED`, `PF_HCDOWN`, `HC_UPMIX`, `GDN_HEAD`, `GDN_CHUNKED`, `GDN_PP=1`, `GDN_CONVL2`, `GDN_NOY`, `QSA_ROWS`, `KV_HOST_DMA`, `ADAPT_OVERLAP`, `STAGER_RING=64`, `PREFILL_RING=192`, `PREFILL_CPU_SHARE=0.6`, `PREFILL_CPU_SHARE_MAX=4096`, `PREFILL_AUTO_MAX=32768`, `SH_STREAM=0`, `GPU_MAX_HW_QUEUES=6`, `ENGINE_NICE=0` | in iq3-1gpu.json env (most also in gfx1100_default_env) | PRESENT-EQUAL | theirs (same value) |
| `ROUTE_TAIL_SKIP=0`, `HC_Q8=0` | gfx1100_default_env (ROUTE_TAIL_SKIP 0); `native_dense.cpp` (HC_Q8 auto = off on gfx1100) | PRESENT-EQUAL | theirs (same value) |
| `PF_F16=1` | 2 | PRESENT-DIFFERENT | theirs kept (2); arm `updated+pf16-1` (spec 250) |
| `HIPBLASLT_TUNING` 100401 table | 100200 table | PRESENT-DIFFERENT | theirs kept; arm `hipblaslt-100401` (spec 251) |
| `--expert-profile` pancake | `data/expert-profile.bin` | PRESENT-DIFFERENT | theirs kept; arm `profile-pancake` (spec 251) |
| `--conversation-cache-mib 2048`, `--max-context 131072`, `--mtp-draft-vocab` en, temperature 0.1 | 8192, 262144, zh60k, 0.6 | PRESENT-DIFFERENT | theirs kept in the shipped file (the benchmark runs the pancake values in both arms) |
| `--spec 4`, `--spec-min-p 0.5`, `--kv int8`, `--kv-resident 32768`, `--pcie-frac 0`, `--adapt-every 1`, `--adapt-swaps 32`, `--prefill auto`, `--expert-cache auto`, `--mtp` | same | PRESENT-EQUAL | theirs |
| `PIN_BY_NEED`, `EXPDEC`, `EXPDEC_DOWN`, `EXCHANGE_ROTATE`, `KV_PREFETCH`, `PROMPT_ATTN_IMMA`, `ATTN_DAILY`, `--pool-affinity measure`, `--conversation-cache-min-tokens 8192` | the switch / option exists, not set | PRESENT, not set | ours added (2.14) |
| `POOL_FUSE_PHASES`, `VERIFY_ECAP=0` + `_MIN_ROWS=9`, `--grammar on` | missing | ported earlier (2.1, 2.2, 2.5) | ours added |
| `--lookup-max-t 12` + 16-row windows (`STRATA_VERIFY_MAX_T=16`) | missing (8-row windows) | ported (2.9) | ours added (+ build flag) |
| RDNA 9-16 column kernels, `ADAPT_PERSIST` (code defaults of the pancake build) | missing | ported (2.10, 2.13) | ours added (code default) |
| `PF_GROUP_SCRATCH`, `PF_HC_REGION`, `PREFILL_EQUAL_TAIL` | missing | ported (2.11, 2.12) | ours added |
| `AGENT_COUNTERS`, GPU router lookahead | missing / inert here | not ported (2.8) | - |
| `AMD_COMMUNITY_TUNING` (build option, ON in both builds) | missing | ported (2.16); no effect on gfx1100 by construction | same as the pancake build |

Build flags: Strata-rdna3's CMake defaults plus `-DSTRATA_WINDOW_MAX_T=16` (ours added: the plain tree has no window option and runs 8 rows; "auto" would also give
16 for HIP, stated explicitly in `tools/pancake/build_updated_rdna3.sh`).

### 2.16 `STRATA_AMD_COMMUNITY_TUNING` (on by default, as on the pancake line; for gfx1010 and gfx103x, nothing for gfx1100)
Four AMD changes that contributors made and measured on cards we do not have, under one switch. Each applies only to the card or the exact library build it was
made for. Ported for the people who run Strata-rdna3 on those cards; it is part of what this branch gives back. Not measured on gfx1100: no effect by construction.

| Change | Applies to | Contributor's numbers (theirs, not reproduced here) |
| --- | --- | --- |
| Integer dot product through the SDWA byte-select sequence that gfx1012 already used (#4855, **Keoian**) | gfx1010 only (RX 5700 / 5700 XT), device code | 2x RX 5700 XT, Qwen3.8-Flash-Next Q2_0, decode 22.38 -> 28.77 tok/s (12 of 12 pairs faster), identical greedy text in 6 of 6 fixed-placement runs |
| `tools/hip/gfx1030-rocblas-5.4.0.ad84daa81c.txt` (#3343, **elightcap**) | gfx1030 with the rocBLAS build `5.4.0.ad84daa81c` (TheRock `gfx103X-all` 7.13.0a20260515 wheels) | RX 6900 XT, UD-Q4_K_XL, `STRATA_HIP_PROMPT_F16=1`: 268-395-token follow-up on a cached conversation 1,851 -> 1,635 ms (median), fresh 745-1,569 tokens -3.3%, fresh 4,859-6,690 tokens -1.0% |
| `tools/hip/gfx1030-rocblas-5.2.0.dabb6df2b98-dirty.txt` (#3924, **schmittcn**) | gfx1030 with the rocBLAS build `5.2.0.dabb6df2b98-dirty` (Arch Linux ROCm 7.2.4) | RX 6900 XT, Coder IQ1_M, GPU timeline of prompts below 1,152 tokens 12-18% shorter, 1.3K-1.5K tokens 2-4%, 11K-12K tokens 4-9%; 11 of 12 answers identical (one differs by FP16 rounding) |
| `STRATA_SELECT_SIMT=2` (the simt2 QSA block scorer) as the default, from 28,672 tokens of reach (#3073, **Vop3p**) | gfx103x (RX 6000) only; any `STRATA_SELECT_SIMT` value you set wins | 2x RX 6900 XT, IQ3_S, prompt time -1.35% at 31K and -6.53% at 119K tokens (10 of 10 interleaved cold-start pairs each); another summation order, so values change in the last bits (their KL against main 0.0256) |

How to switch it: `STRATA_AMD_COMMUNITY_TUNING=0` (or off / false / no) at run time turns the two tables and the selector default off. The dot product is device
code, so only the build can drop it: `-DSTRATA_AMD_COMMUNITY_TUNING=OFF` (CMake, default ON) defines `STRATA_AMD_COMMUNITY_DOT_OFF` and gfx1010 compiles the
portable loop again. On a gfx1010 or gfx103x card the engine prints one line saying what is active; on any other card it prints nothing. A table loads only when
the card and the full rocBLAS version string match its header, with `STRATA_HIP_PROMPT_F16=1` and `STRATA_ROCBLAS_TUNING` pointing at `tools/hip`.
Adapted to this tree: Strata-rdna3's HIP selector had only `STRATA_SELECT_SIMT=1`, so the simt2 launch is added for gfx103x only; an explicit `=2` on any other
HIP card keeps what this tree ran before (the pancake line runs simt2 there too), and the generic `STRATA_SELECT_ALT_MIN_TOKENS` of the pancake line is not
ported (only the 28,672-token floor of the community default is).

Why gfx1100 is not affected (read from the code, not measured): (1) the build define is read only inside `#elif (defined(__gfx1012__) || (defined(__gfx1010__)
&& ...))` of `include/strata/hip_compat/intrinsics.hpp`, which a gfx1100 compile never enters (it takes the `__gfx1100__` sudot4 branch above it), and in
`dot4_gfx1010_built()`, used only by the gfx1010 part of the log line; the box build passes `ON`, the default, so it adds no define at all. (2) The log line
returns an empty string unless the card is gfx1010 or gfx103x. (3) The tables are refused by the loader on any architecture but gfx1030, and the new check only
refuses more. (4) The selector default needs `sel_gfx103_device()`; on gfx1100 `STRATA_SELECT_SIMT` unset, `=1`, `=2` or anything else picks the same scorer
as before. The one difference in a gfx1100 build is that the simt2 kernel is now also compiled into the HIP code object (never launched on gfx1100), the same
with the option on or off. Files: `CMakeLists.txt`, `include/strata/core/amd_community.hpp` (new), `include/strata/hip_compat/intrinsics.hpp`,
`src/core/arch_defaults.cpp`, `src/prefill/gemm.cu`, `src/kernels/cuda/qsa_select.cu`, the two tables in `tools/hip`, `tools/pancake/build_updated_rdna3.sh`;
test `src/core/amd_community_test.cpp` (CTest `amd_community_test` and `amd_community_test_dot_off`, CPU). Pancake commits `3eb91676d`, `b130965ca`, `55ab701e1`.

## 3. Diagnostic tests and how to reproduce
Host checks run for this branch (no GPU needed): `python -m pytest rdna3/test_tuned_flags.py tools/pancake/test_flag_diag.py tools/pancake/test_host_tune.py
tools/pancake/test_pcie_link.py` (fake hardware, fake verdicts, fake sysfs); `verify_ecap_test` (CTest, CPU). The HIP engine files compile only on a Linux
ROCm host (they include POSIX headers): `tools/pancake/build_updated_rdna3.sh SRC BUILD --target strata` builds them, `ctest --test-dir BUILD -R
"verify_ecap_test|sampler_parity|elementwise_parity|window_limits_test|draft_policy_test|prefill_plan_test"` runs the parity and window tests. On the target machine: `tools/pancake/flag_diag.sh run --config <server config>
--flags POOL_FUSE_PHASES,VERIFY_ECAP,PF_F16,HIPBLASLT_TUNING,EXPERT_PROFILE` (a few minutes per flag; it starts engines only through the A/B harness). Results of a diagnostic
run on the box: TO BE MEASURED.

## 4. Benchmark: plain Strata-rdna3 against this branch (TO BE MEASURED)
Plan: `tools/pancake/box_chains/bench-updatedRD.sh` builds plain Strata-rdna3 `8b01b0c85` with its documented flags and this branch with
`build_updated_rdna3.sh`, runs the parity tests, the flag diagnostics, spec 250 (screen, then 5 repeats in blocks of 3, one engine at a time, text compared with
the plain arm; arms `plain`, `updated` = plain + the whole pancake preset of 2.15, `updated+pf16-1`) and spec 251 (screen first: the 100401 table and the learned
profile against the updated arm with Strata-rdna3's own table and profile). Workloads: short chat and code edit (decode tok/s), cold prompts of 10,000 / 32,000 / 64,000 / 100,000 tokens (prompt tok/s). Short plan (screen first, confirm only for arms with a screen benefit, 3 reps, tune-up included): about 1.9 hours of box time, budget 2 hours (`bench-sched.sh` runs it before the rest of the box queue).

| case | plain | updated | change | text identical |
|---|---|---|---|---|
| short (decode tok/s) | TO BE MEASURED | TO BE MEASURED | TO BE MEASURED | TO BE MEASURED |
| edit (decode tok/s) | TO BE MEASURED | TO BE MEASURED | TO BE MEASURED | TO BE MEASURED |
| 10,000 tokens (prompt tok/s) | TO BE MEASURED | TO BE MEASURED | TO BE MEASURED | n/a (unique first line) |
| 32,000 tokens (prompt tok/s) | TO BE MEASURED | TO BE MEASURED | TO BE MEASURED | n/a |
| 64,000 tokens (prompt tok/s) | TO BE MEASURED | TO BE MEASURED | TO BE MEASURED | n/a |
| 100,000 tokens (prompt tok/s) | TO BE MEASURED | TO BE MEASURED | TO BE MEASURED | n/a |

Build flags of the optimised build (each justified in the script header): Release, -O3, gfx1100 only, HIP on and CUDA off, `-mtune=native` (no change of
instruction set, same bits), no fast-math, `-DSTRATA_WINDOW_MAX_T=16` (2.9); `-march=native` and LTO stay opt-in until measured.

## 5. Credits and authorship
Outside sources (ideas or code), from the pancake README section 9 and LEADS.md:
- **nicejunjie and the Strata-rdna3 contributors** (MIT): the base of this branch, and the RDNA3 wave32 kernels that the pancake line ported from it in the first place.
- **Niko1221 and the Strata contributors** (MIT, copyright holders of the shared base).
- **jmnargi**, Strata-V100 (MIT), commits `24894e676b`, `7c5041c358`, `376355453f` (upstream PR #3545): the verify-window expert cap that `STRATA_VERIFY_ECAP` adapts, and
  the measurement of how a too-tight cap breaks tool calls. The per-draft-row form that counts only CPU-resident experts is the pancake adaptation.
- **jesdga95**, basalt (MIT), commits `d64768699`, `02e1006fe`, `c7813a6c3`, `011e13d72`: the grammar path (window guard, masked redraw, `--grammar` and the request key); the GBNF engine and the schema builder in it are llama.cpp's (MIT). Adapted and extended on the pancake line.
- **Niko1221** (Strata, MIT): commits `9d3c8c366` (16-row windows), `03627326d` (`--lookup-max-t`), `fbf6e1c75`, `3cc47f029`, `c94010648` (`PF_GROUP_SCRATCH`,
  `PF_HC_REGION`), hand-ported here.
- **merbanan** (Strata PRs 4040 and 4039, MIT): the lookup windows past `--spec` and the prompt-path scratch savings those commits port.
- **bramvanoploo**, strata-performance-setup (MIT): the list of host settings behind `host_tune.sh`; no code copied.
- **Keoian** (Strata PR #4855, the gfx1010 SDWA dot product), **elightcap** (#3343) and **schmittcn** (#3924) (the two gfx1030 rocBLAS tables), **Vop3p**
  (#3073, the simt2 selector default on gfx103x): the community AMD changes of 2.16, measured by them on cards we do not have; ported to the pancake line
  by Niko1221's Strata commits as pancake `3eb91676d`, `b130965ca`, `55ab701e1`, and from there to this branch.
- **caminus** (jdominick05, AGPL-3.0-or-later): ideas only, none copied; the note that the PCIe link stays awake behind `pcie_link.py`, which is our own code.

Original work of pancake-strata (rocktboy1982), with the record that supports it. Git history of the pancake line: first commit `346709879`, 2026-11-07 17:10 +0200,
author rocktboy1982; `STRATA_POOL_FUSE_PHASES` `35c512984` (2026-11-09) and its fix `50c6464cd` (2026-11-09), both rocktboy1982, ledger T-224; the ECAP adaptation `2d44d31ec`
(2026-11-10, rocktboy1982), ledger T-588, T-589, T-591; the two fixes `5735ae7ac` and `bcfe5ba08` (2026-11-15, rocktboy1982), ledger T-573, T-576; `host_tune.sh`
`cf6726c20` (2026-11-16) and `os_tune.sh` `9ff4bdc4c` (2026-11-10), rocktboy1982; `pcie_link.py` `3d8d516a9` (2026-11-18, local account `iancu`, the same owner). The grammar port is `3d3eef53e` (2026-11-08) with the serve routes `bcbdeb5d2` (2026-11-09), rocktboy1982, ledger T-315; the learned profile is `3638b8cee` (2026-11-08, rocktboy1982), T-194. The
sources of the diagnostics, the tuned-flags startup, the build script and the benchmark chain are new in this branch (pancake-strata, rocktboy1982, original work). The
commits of this branch carry the trailer `Co-Authored-By: Claude Sonnet 5.5` (Claude Opus 5.5 from the 2026-11-19 night round on). The RDNA 9-16
column kernels (`82687a7bd`, `b9b34ca1e`, 2026-11-15), the equal-tail rule (`7639dbf76`) and the persistent adapt worker (`8780d9e2b`, 2026-11-09) are
rocktboy1982's, ledger T-079..T-081, T-372, T-679, T-712.

## 6. Compatibility and risks
- Engine changes are off by default except two code defaults taken from the pancake build: the RDNA kernels take 9-16 columns (only reached by windows of 9+ rows,
  i.e. `--lookup-max-t` past 8 or a wide batch) and `STRATA_ADAPT_PERSIST` (the same work on a persistent thread). With windows of up to 8 rows and no switch set the
  default path runs Strata-rdna3 `8b01b0c85`'s kernels, apart from the two fixes in 2.3 (NaN temperature, multi-token gate), which do not change normal runs. `--no-tuned-flags` (or `STRATA_TUNED_FLAGS=0`) restores the setup's plain behaviour in the startup script.
- The shipped flags (the whole pancake preset, 2.15) were measured on the pancake base, not on this tree, and some were never A/B'd on HIP even there
  (`PF_GROUP_SCRATCH`, `PF_HC_REGION`, `KV_PREFETCH`: Speedy's numbers). `STRATA_VERIFY_ECAP` is lossy (gates passed on the pancake base). Regenerate the file from
  section 4 before relying on it. `flag_diag.sh` reports a flag UNSUPPORTED on hardware that lacks what it needs (WMMA, VMM, hipBLASLt).
- `STRATA_VERIFY_ECAP` is lossy and was measured with the grammar guard on; the shipped file enables both (with `--lookup-max-t 12`). `--grammar on` was measured for validity on another GPU, not for speed here.
- The learned profile is specific to Qwen3.8-Flash-Next IQ2_XS and was measured on another GPU; the 100401 table was measured only against no table. Neither is in the shipped flags file (Strata-rdna3's own stay; spec 251 arms).
- What was compiled locally: `verify_ecap_test` and the three grammar tests (MSVC), `pool.cpp` syntax, `sampler.cu` and `elementwise.cu` for sm_75. For the 2026-11-19 night round: a HIP clang syntax check (ROCm 6.2 clang, gfx1100,
  `STRATA_VERIFY_MAX_T` 16 and 8) of every touched engine file, and `window_limits_test`, `draft_policy_test`, `prefill_plan_test` built and run on a PC (pass). The HIP
  build and every run happen on the box.
- Licence: Strata-rdna3 is MIT; every ported piece is MIT-compatible (our own code, or MIT sources credited above); AGPL sources were read for ideas only.
