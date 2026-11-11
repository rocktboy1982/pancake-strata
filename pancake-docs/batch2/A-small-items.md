# Batch 2, group A: small Phase 16 items (A21, A29, A30, A33/A34/A35, A49, A52, A53, A56, A57)

Branch `work/a-small-items` (from `origin/dev/batch-2`). Nothing here was built or run on the GPU; everything that needs the
engine is a spec or a tool for the 02:00 window. Python tests: 52 passed (see "Tests run" at the end).

All new behavior is opt-in or information only; the engine's default path is unchanged (no C++ was touched).

## A21 - does T = 0.1 take the sampled speculative path?

**Answer, by reading the code: yes, T = 0.1 is a SAMPLED request everywhere.** The only greedy test in the stack is
`temperature <= 0`:

- `src/program/generate.cpp` (serve request parse, ~18432): `req_sp.greedy = req_temperature <= 0.0f`.
- `src/core/verify.cpp:4060` and 5901: `sampled = !sampling_.greedy && sampling_.temperature > 0.0f`; the sampled head draws
  every row with Philox keyed by position (`sampler.cu` tail), the host accepts a draft only if it EQUALS the token the target
  drew for that row. At T 0.1 that token is the argmax almost always, so the plain MTP argmax drafts keep their acceptance.
- `serve/server.py:1799` forwards any non-zero temperature (`temperature=0.1`); only 0 is dropped.
- The coupled / probabilistic drafter is a separate opt-in: `MtpDrafter::set_draft_sampling` turns it on for a request when
  `STRATA_SPEC_COUPLED` or `STRATA_SPEC_PROB` is set AND `temperature > 0 && temperature >= STRATA_SPEC_MIN_TEMP`
  (default MIN_TEMP 0). So with those switches on and no MIN_TEMP, opencode sub-agents at T 0.1 DO take the sampled drafter.
- Earlier 3060 / 5070 numbers (docs/DETAILS.md, `spec_prob.hpp`): sampled drafting gains at T 1.0 and loses below ~0.7.

**Consequence / recommendation (untested on the 7900 XT):** if `STRATA_SPEC_PROB=1` or `STRATA_SPEC_COUPLED=1` is turned on for
the main agent (T 1.0), add `STRATA_SPEC_MIN_TEMP=0.7` so the T 0.1 sub-agents keep the argmax drafter.

**Spec:** `bench/specs/a21-spec-lowtemp.json` (new top-level `temperature: 0.1`; abtest already supports it). Arms: base,
coupled, gumbel, f6-dt100, f6-dt050, f6-min07. `f6-min07` is the control: it must equal base within noise (if it does not,
the MIN_TEMP guard is not working). Expected: base best at T 0.1; coupled / f6-dt100 slower. Text differs run to run
(ignore the hash line); compare tok/s and "drafts accepted". `python tools/pancake/abtest.py bench/specs/a21-spec-lowtemp.json
--tier screen --out bench/results/<day>-a21`. Risk: none (env only). Existing `12-spec-sampling.json` (T 1.0) left as is.

## A29 - entropy check of the expert bytes (RUN, closes the item)

Tool: `tools/pancake/expert_entropy.py` (`--help`; mmap, reads only block-aligned windows, default <= 8 MB, run with
`nice -n 19`). It reports byte / nibble / 2-bit-crumb / order-1 / per-block-position entropy and real compressor ratios
(zlib -9, lzma, zstd -19, lz4 -9; zstd/lz4 through the python module or the CLI, both CLIs exist on this box).

Run on 2026-11-10 (12 expert tensors spread over layers 0-47, 1 MiB each; 12 MiB from the Q2_0 file, 12 MiB from the IQ2_XS file):

| file / tensor type | sampled | byte entropy (of 8) | crumb 2-bit (of 2) | per-position bound saving | best real compressor |
|---|---|---|---|---|---|
| Q2_0 file, all Q2_0 | 12 MiB | 7.32 | 1.843 | 11.7% (order-0: 8.5%) | zstd -19: 8.0% |
| IQ2_XS file, Q2_0 (ffn_down) | 4 MiB | 7.30 | 1.837 | 12.0% (order-0: 8.8%) | zstd -19: 8.3% |
| IQ2_XS file, IQ2_XXS (gate) | 2 MiB | 7.948 | 1.9986 | 3.3% | none (0.1%) |
| IQ2_XS file, IQ2_S (gate/up) | 5 MiB | 7.964 | 1.9987 | 3.3% | none (0%) |
| IQ2_XS file, IQ1_M (up) | 1 MiB | 7.971 | 1.9998 | 3.4% | none (0%) |

(The file called IQ2_XS is a MIXED file: Q2_0 down tensors, IQ2_XXS / IQ2_S / IQ1_M gate and up. The second shard holds no experts.)
Findings: the IQ types are random bytes (the per-position figure is only the constant fp16 scale high byte). Q2_0 payload is
mildly skewed (zero symbol is most common): 1.84 of 2 bits, so a block-aware coder could save at most ~12%, a general
compressor 8%. **Best saving 12.0% < 20%: an entropy-coded or compressed expert miss path is not worth building. A29 closes.**
Numbers are from this one sample (estimator bias < 0.0002 bits); rerun with `--tensors 48 --max-mb 48` for more layers.
Files: `tools/pancake/expert_entropy.py`, `test_expert_entropy.py`.

## A30 - Fiddler token-count threshold for the prefill CPU share

Reading: the preset has `STRATA_PREFILL_CPU_SHARE=0.6`, `_MAX=4096` (chunks below 4096 tokens stage experts on the decode pool).
Spec `bench/specs/a30-fiddler-cpu-share.json` (`prefill: true`), arms base, share0 (`STRATA_PREFILL_CPU_SHARE=0`), max1024,
max2048. New abtest cases `long1k` / `long3k` (cold 1000 / 3000-token prompts, judged on prompt tok/s; `LONG_TARGET`, `MAXTOK`
32) cover agent-sized deltas; `short` first as warm-up, `edit` (~2K tokens) kept. Lossy (the staged CPU experts round
differently from the GPU path, KL ~0.004-0.011 per the engine's own note), so a faster arm needs token forcing. Expected: up to
~0.4 s per agent turn per the row; risk: it can be a loss at 1-2K. Caveat: one request per server per long case (abtest design),
so use `--reps 4` and `--tier confirm` for a verdict. Run: `abtest.py bench/specs/a30-fiddler-cpu-share.json --tier screen --out ...`.

## A33 / A35 retests and the A34 sweep

- `bench/specs/a33-a35-fusion-retest.json`: base (fast-hip, deferral on), expdec-swiglu, ic-warm32, ic-warm64, swiglu+warm32; cases
  short, edit; exact (same text expected). SWIGLU is only taken where the layer's gate/up type is IQ2_XXS/XS/IQ3/IQ2_S: this model
  has IQ1_M layers that keep the old path, so expect a partial effect.
- **Finding (code reading): `STRATA_ROUTER_FUSE` is INERT on the current preset.** `verify.cpp` computes `rpub = ... && !defer_ ...`; the
  fast-hip preset has `STRATA_VERIFY_DEFER_CPU=1`, so `bf16_router_publish_arm` is never called (the start-up line "STRATA_ROUTER_FUSE:
  the router GEMV's last block routes and publishes" will be missing). `bench/specs/a33-router-fuse-nodefer.json` compares inside the
  no-deferral world (nodefer vs nodefer+router-fuse, plus the unmodified preset for scale); making the fusion work WITH the
  deferral is a code change (the deferred rows would have to use the fused publish), not done here.
- A34: `tools/pancake/sweep.py` exists (coordinate descent, UFMG recipe). Its built-in knob list starts at `STRATA_EXPDEC_MINBLK=400` (engine
  default 800) and its default `--base` is the old rdna3test preset, so the first point is not the preset. Added
  `tools/pancake/sweep-knobs-fast-hip.json` whose first value of every knob is the engine default, so point p0 == the preset. Run line:
  `nice -n 19 python tools/pancake/sweep.py --knobs tools/pancake/sweep-knobs-fast-hip.json --base pancake-flashnext-iq2_xs-fast-hip.json
  --cases short,edit --reps 2 --out bench/results/<day>-sweep-fasthip` (about 17 server starts per pass, repeated while a pass gains > 1%;
  plan ~1 h). Apply the winner via `abtest.py` confirm before changing the preset (the sweep picks on tok/s alone, noisy).

## A49 - decode at 10K vs 32K vs 64K vs 128K

Spec `bench/specs/a49-decode-context.json`: one arm, cases short, longdec10k (new case, 10000-token cold prompt, 256-token reply),
longdec32k, longdec64k, longdec128k (existing). Metric is the engine's decode tok/s of the reply. Convert to ms/token; the
slope over context is the attention cost. Decision rule written in the spec note: (ms@128K - ms@10K) < ~10% of ms@10K means the
attention share is too small for A50/A51 (sparse attention); > ~25% means it is worth it. Single arm: use `--no-early-stop --reps 3`
(`decided()` would otherwise stop after 2 reps). Expect about an hour of GPU time (each 120K read is minutes).

## A52 - direct loader start time (184e68e83)

184e68e83 is on `feat/direct-loader`; dev/batch-2 already carries the same change as 5c071d364 (`load_experts_gguf_direct_linux`
is in `expert_source.cpp`), switch `STRATA_UNBUFFERED_LOAD` (default: the engine picks direct when `file_cache_keeps` says the page cache
cannot hold the arena). New tool `tools/pancake/load_time.py`: runs default / direct (`=1`) / buffered (`=0`), each with the model's
page cache advised away (cold, `posix_fadvise DONTNEED`, no root) and warm, interleaved, and records: seconds to "ready: http" (0.5 s
poll), the engine's own "expert arena load: mode ..., GiB/s" line, memory PSI some/full deltas (full = the oomd trigger), peak page-cache
growth (the double copy), lowest MemAvailable, and the `STRATA_VERIFY_ARENA` checksum (must be identical across modes; the tool warns).
`python tools/pancake/load_time.py --out bench/results/<day>-load-time --reps 3` (~18 starts, ~20 min, GPU must be free; it stops
`server.py --engine strata` processes like abtest does). `--from-rows rows.jsonl` reprints the table. Expected per README: 28-65 s -> 10-14 s cold.
Tests cover parsing / summary only (`test_load_time.py`); the run itself is untested.

## A53 - IOMMU translated-mode note

`serve/iommu_check.py` (stdlib, Linux, sysfs + `/proc/cmdline`; reads the group type of the display-class PCI devices that are not the
APU's graphics: `identity` = passthrough, `DMA` / `DMA-FQ` = translated; falls back to `iommu=pt` / `iommu=off` on the command line;
no IOMMU unit = off). `serve/server.py` prints ONE line before the engine starts, only when translated (default on, `STRATA_IOMMU_CHECK=0`
hides it; wrapped so it can never stop a start). `tools/pancake/os_tune.sh check` now also prints the GPU's IOMMU group type and the
`iommu=pt` hint. No behaviour change. On this machine (kernel 7.0.0-43, no iommu= option) the 7900 XT at 0000:12:00.0 is already in
an `identity` group (passthrough), so the note does not fire here; of 35 groups, 33 are DMA-FQ (translated) and 2 identity. Tests:
`serve/test_iommu_check.py` (10, fake sysfs trees: translated DMA-FQ and DMA warn, identity / VFIO / APU graphics / no IOMMU do not,
cmdline fallback, switch off, missing sysfs). The text says translation "can cost PCIe throughput": not measured here.

## A56 - EAR next to "same top token" in token forcing

SLQ's definition, read from `IST-DASLab/SLQ` `src/metrics/evaluation.py::compute_kl_and_ear`: per position, p = softmax of the baseline's cached
top-K logits (K = 10), q = the other model's logits gathered at the SAME K indices, renormalised; EAR = sum_k min(p_k, q_k); the
reported value is the token-weighted mean over positions. Implemented in `tools/force_check.py`: `ear --stock A.tsv --cand C.tsv [--topk 10]`,
`check ... --ear` (cand vs stock arm 1, plus stock 1 vs stock 2 as the noise floor), `score --topk K` (default 2 = byte-identical to before; use
10 so EAR has SLQ's K), and `quality_gate.py --ear` (scores with top-10 and adds the numbers to every verdict). Output: EAR and the same-top-token
rate with 95% bootstrap intervals over requests. Limitation: the TSV holds only the top-K ids; an id of the reference's list that the other arm
did not list gets that arm's smallest listed probability (an upper bound, counted in `substituted_ids`); with K = 2 EAR is a top-2 figure.
Tests: `tools/pancake/test_force_ear.py` (8; known overlaps, renormalisation, substitution, synthetic TSVs end to end); `force_check.py selftest` still ok.
Use: EAR predicts the MTP draft acceptance of an approximation (sample drafts from p, accept with min(1, q/p)); same-top-token cannot.

## A57 - area under the time-to-success curve (task_gate)

`tools/pancake/task_gate.py`: `time_auc(runs, horizon)` = mean over ALL runs of `ok * (horizon - elapsed) / horizon` = the area under
S(t) (share of runs succeeded within t) on [0, horizon], 0..1; failures and errors count 0. One horizon for every arm: the longest run of
the batch, or `--auc-horizon SECONDS`. The report prints per arm "time-AUC x (horizon, success within 25/50/75/100% of it) vs base d [lo, hi]" (paired
bootstrap over tasks like the success diff) and `summary.json` has `auc`, `auc_diff`, `auc_horizon_s`. Information only: it never changes
a verdict. Tests (2 new in `test_task_gate.py`): the definition against the integrated step curve, clipping, and a faster arm with identical
successes ranks higher with an identical verdict. Caveat: elapsed is per request (the whole agent task for multi-turn tasks), and includes machine noise;
the `noisy` warning still applies.

## Files changed

`tools/pancake/`: `expert_entropy.py`, `load_time.py`, `sweep-knobs-fast-hip.json` (new); `abtest.py` (cases long1k, long3k, longdec10k),
`task_gate.py`, `quality_gate.py`, `os_tune.sh`; tests `test_expert_entropy.py`, `test_load_time.py`, `test_abtest_cases.py`, `test_force_ear.py`,
`test_task_gate.py`. `tools/force_check.py`. `serve/iommu_check.py`, `serve/test_iommu_check.py`, `serve/server.py` (3 lines).
`bench/specs/`: `a21-spec-lowtemp`, `a30-fiddler-cpu-share`, `a33-a35-fusion-retest`, `a33-router-fuse-nodefer`, `a49-decode-context`.

## Tests run (nice -n 19)

`python3 -m pytest -q tools/pancake/test_abtest_cases.py tools/pancake/test_expert_entropy.py tools/pancake/test_load_time.py
tools/pancake/test_force_ear.py tools/pancake/test_task_gate.py serve/test_iommu_check.py` -> 52 passed. `python3 tools/force_check.py selftest` -> ok.
`python3 -m py_compile serve/server.py tools/pancake/quality_gate.py tools/pancake/task_gate.py` ok. `sh tools/pancake/os_tune.sh check` runs, prints the IOMMU line.
`expert_entropy.py` run on the two real GGUFs (numbers above).

## What is left / not verified

- Every spec (a21, a30, a33/35, a33-router, a49, A34 sweep line) and `load_time.py` itself: not run (GPU busy). The `long1k/long3k` prompt size accuracy
  (LONG_CPT, warns if > 8% off) is only checked by length, not by the server's token count.
- A33 router fusion with the deferral is not possible without a code change (see above).
- A52: if direct is already the default here (page cache cannot keep a 33 GiB arena next to the GGUF on 57 GB RAM?), the default arm and the
  direct arm are the same loader; the tool shows the `loader` column to tell.
- Commits end with `Co-Authored-By: Claude Opus 5.5` as the brief asked.
