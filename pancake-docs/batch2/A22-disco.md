# A22 DISCO learned chain stop (with A25 and A26)

Branch `work/a22-disco` (from `origin/dev/batch-2`). MOD_PLAN Phase 16.4, rows A22, A25 and A26. A23 (the measured
cost table, 663107a56) is used as it is.

## What

**A22. A learned stop rule for the MTP draft chain.** After each draft, a small logistic model guesses whether the
target will accept it. The chain stops at the first draft the model does not trust. A table of thresholds, one per
draft position, also lets each position carry its own price. That price is the number of new CPU-missed experts its row
wakes (SpecFit). The rule only decides how many drafts get verified. The verifier still accepts exactly the target's
tokens, so the output does not change (exact).

The feat/disco branch already had the engine side (`STRATA_DISCO`, `STRATA_DISCO_TRACE`) and an older fitter, but it
was built on an old base. This branch brings it onto dev/batch-2 and adds the following:

1. **Per-row miss counts in the window trace.** `include/strata/core/row_miss.hpp` adds `RowMissCounter`. For each
   layer of a verify window it counts the experts that row r runs on the CPU and that no earlier row in the window had
   already put on the CPU. It sums these over the layers. With `STRATA_WINDOW_TRACE` set, every kept window gets a new
   **record 5** (`u8 5, u8 n, u16 0, n x u16`) right after its window record. `wtrace.py` reads it (`Window.row_miss`,
   `row_new_miss(w)`). When record 5 is missing, it falls back to the existing `new_cpu_miss()` on the layer records.
   Those give the same numbers.
2. **DISCO weights version 2.** Feature 6 used to be a window-level miss rate. It is now the expected new CPU misses
   of the draft's own row: one EMA per window row, learned from earlier MTP windows. The engine feeds the same counter
   from the verify pool, but only while `STRATA_DISCO` or `STRATA_DISCO_TRACE` is set. The weights file can now carry
   an optional `thresholds n t0 .. t(n-1)` line, one threshold per draft position (0 keeps every draft at that
   position, 1 keeps none). A `disco 1` file is refused with a message that asks for a refit. `STRATA_DISCO_TRACE`
   gains a trailing `nmiss` column.
3. **Drafter entropy is not available cheaply.** The MTP selection kernel (`mtp_select`) returns only the top-1
   probability. Its two-stage proxy pick never builds the full softmax. Entropy would need a GPU reduction change in
   both HIP and CUDA, and the proxy pick would have to be turned off. So it was not done. The top-1 probability,
   -log p and the chain product carry the confidence signal. The probability is already in record 4.
4. **`tools/pancake/disco_fit.py`** (rewritten, numpy only):
   - It reads window traces and/or DISCO CSV traces.
   - For a window trace it replays the engine's own feature state exactly: p, surprise, chain, position, recent
     acceptance, the row's expected misses and context. The two features a window trace cannot replay (5 lookup
     match, 7 DraftPolicy's dcost) are fitted at weight 0.
   - It fits an IRLS logistic regression.
   - It fits a window cost model `ms = c0 + c1 (rows-1) + c2 (new misses of the draft rows) + c3 (row 0's misses)` on
     windows that directly follow their predecessor. The per-row term c1 is fitted only when the window sizes vary (at
     least two sizes with 5% of the windows each). A trace taken with every draft verified has nearly one size, so
     there c1 is set to `--row-ms` (default 0.7 ms, about one MTP draft step), and the report says so.
   - It picks a global threshold, then a per-position table by coordinate ascent. The ascent starts from the better
     of the global threshold and every static length, so on the training windows the table is never worse than any
     static length.
   - It reports on the last 25% of the windows: the AUC against the draft probability alone, and for each arm the
     tokens per window, ms per window and tokens per ms against the logged rule. The arms are DISCO (global and
     table), every static length, `--spec-min-p` 0.3 / 0.5 / 0.7 / 0.85, and the oracle cut (verify exactly the drafts
     that held, which is the ceiling for any rule that only stops earlier).
   - `--simulate WEIGHTS` replays an existing file.

**A25. `tools/pancake/draft_report.py`.** It compares DraftPolicy's built-in MTP numbers with what the traces measure:

- the prior `mtp_tokens(t) = 1 + 0.7 (t-1)`, which says every extra row is worth 0.7 tokens with no decay;
- the prior `draft_rate(p) = p`, which takes the MTP's softmax as the acceptance;
- `kShape`, the prior cost by window size.

It prints:

- the acceptance at each position (conditional and survival) next to the prior and the `mtp_probe` numbers;
- tokens per window by size;
- acceptance in the 8 probability buckets of `draft_rate`;
- the median window cost by size against `kShape`;
- the **in-engine verify share** from the engine's `strata decode timing:` lines (`STRATA_DECODE_TIMING=1`):
  verify, draft, commit and adapt as shares of a window, weighted by windows.

**A26. The same tool, draft source section.** It shows the realised tokens per window and tokens per ms for each draft
source, from the window trace. From a `STRATA_SPEC_TRACE` file it also shows the **counterfactual**. Each round there
logs the drafter's chain, the lookup proposal and the committed tokens. Both proposals are scored against the committed
tokens. This shows, for lookup windows, whether the drafter's chain would have held longer, and for drafter windows,
whether the lookup would have. It also flags whether the lookup was eligible (its first token equal to the drafter's,
which the engine requires before it takes a lookup), and it compares each round per ms at median costs. The decision
rule printed: A26 stays parked while the other source looks better in only a few % of rounds. A learned source choice
is worth building above about 10%.

## Papers

- DISCO: Mamou et al., Intel Labs, https://arxiv.org/abs/2405.04304
- MiMo-V2-Flash: https://arxiv.org/abs/2601.02780 (entropy as the draft signal; not available here, see point 3)
- SpecFit: https://arxiv.org/abs/2610.15417 (draft position cost = new missed experts)
- A25: SD: Performance or Illusion?, https://arxiv.org/abs/2601.11580
- A26: Context-Aware Assistant Selection, https://arxiv.org/abs/2408.08470

## How to enable

All of it is opt-in. With none of these set, the code path is unchanged. The only cost is one `on()` branch per layer
in the verify pool, next to the existing window-trace branch.

- `STRATA_DISCO=<weights file>`: the learned rule replaces `--spec-min-p` / `STRATA_SPEC_COST` for MTP windows (serve
  and CLI). DraftPolicy still chooses lookup vs MTP. DFlash2 requests keep their own rule.
- `STRATA_DISCO_TRACE=<csv>`: DISCO's training lines, holding the features exactly as the engine saw them.
- `STRATA_WINDOW_TRACE=<file>`: the window trace, now with record 5.
- `STRATA_SPEC_TRACE=<file>` (existing): the per-round proposals that A26's counterfactual needs.
- `STRATA_DECODE_TIMING=1` (existing): the timing lines that A25's verify share needs.

Workflow:

```
# 1. one trace run on the new build: every draft verified, so every draft has a label
STRATA_WINDOW_TRACE=/home/dan/clone-of-d/disco/w.stwt STRATA_DISCO_TRACE=/home/dan/clone-of-d/disco/d.csv \
STRATA_SPEC_TRACE=/home/dan/clone-of-d/disco/s.txt STRATA_DECODE_TIMING=1 \
  <serve with the preset, --spec-min-p 0 instead of 0.5>   # then replay agent turns (replay_turns.py) or abtest prompts
# 2. A25 / A26 report and the DISCO fit (offline, no GPU)
.venv/bin/python tools/pancake/draft_report.py /home/dan/clone-of-d/disco/w.stwt /home/dan/clone-of-d/disco/s.txt <server log>
.venv/bin/python tools/pancake/disco_fit.py /home/dan/clone-of-d/disco/w.stwt -o data/disco-weights.txt
#    (or fit on d.csv: all 9 features, including the lookup match and DraftPolicy's dcost)
# 3. the A/B
.venv/bin/python tools/pancake/abtest.py bench/specs/24-a22-disco.json --tier screen --out bench/results/<day>-a22-disco.jsonl
```

`WindowTrace::open` opens its file with "wb", so every server start overwrites the trace. Collect it in one long
server run, not across abtest arms.

## Files changed

- `include/strata/core/row_miss.hpp` (new): RowMissCounter, `row_misses()`.
- `include/strata/core/window_trace.hpp`: record 5, its counter, the format comment.
- `include/strata/spec/disco.hpp`, `src/spec/disco.cpp`: version 2, feature 6, per-position thresholds, `Round.row_miss`,
  the `nmiss` column. (These two files and the generate.cpp hooks come from feat/disco bf53e9f68.)
- `src/program/generate.cpp`: feat/disco's hooks (applied without conflicts), plus feeding the row counter in
  `drive_pool_multi_impl` and enabling it, passing it and resetting it in the serve and CLI loops.
- `src/spec/disco_test.cpp` (updated), `src/core/row_miss_test.cpp` (new), `CMakeLists.txt` (row_miss_test registered;
  disco_test and row_miss_test in the cpu label list).
- `tools/pancake/disco_fit.py` (rewritten), `tools/pancake/test_disco_fit.py` (rewritten),
  `tools/pancake/draft_report.py` + `tools/pancake/test_draft_report.py` (new), `tools/pancake/trace/wtrace.py` (record 5).
- `bench/specs/24-a22-disco.json` (new).

## Tests run

- `python3 tools/pancake/test_disco_fit.py`: 5 passed. These cover the replayed features against the engine's state
  machine, the cost model (the prior on a one-size trace, the fitted per-row ms of 1.5 recovered on mixed sizes), a fit
  that beats the logged rule and every static length on held-out synthetic windows, a CSV with `nmiss` and appended
  segments, and the replay arithmetic.
- `python3 tools/pancake/test_draft_report.py`: 3 passed (a synthetic window trace, a draft trace, timing lines).
- `python3 tools/pancake/trace/wtrace.py --selftest`: ok (record 5 read; `row_new_miss` falls back to the layer records).
- C++ `disco_test` and `row_miss_test`: see "C++ build" below.
- On a synthetic 20K-window trace, the fit and report took 10 s.

## Spec to run

`bench/specs/24-a22-disco.json`: arms base (the preset's `--spec-min-p 0.5`), disco (`STRATA_DISCO=.../data/disco-weights.txt`)
and spec-cost (`STRATA_SPEC_COST=1`, DraftPolicy's existing cost-aware rule). Cases short and edit, `lossy: false`.
The text hash must equal base. It needs the fitted weights first (workflow steps 1 and 2). Skip the GPU A/B if
`disco_fit.py`'s held-out gain is under about 2%.

## Expected gain and risk

- Gain: 0-10% decode (estimate). The SpecFit authors' ceiling is +7-22%. The synthetic numbers in the tests are not an
  estimate: the synthetic costs are made up. The real ceiling for any stop-only rule is the oracle-cut line in the
  `disco_fit.py` report on a real trace. At `--spec 4` (at most 3 drafts) the room is small.
- Exactness: the rule only changes the window length. `disco_test` checks greedy equality on a mock target across many
  thresholds and with the per-position table.
- Risks:
  - The replay sees the trajectory of the rule that wrote the trace. With DISCO on, windows are shorter (a
    distribution shift).
  - A stop-only rule cannot learn to draft deeper than the logged chains, which is why the trace run uses
    `--spec-min-p 0`.
  - The window cost model is a linear fit.
  - Feeding the row counter turns on the pool's per-entry placement copy (host work, at most 2,550 ints per layer
    on prompt windows) while DISCO is on.
- Format: with `STRATA_WINDOW_TRACE` on, the trace now carries record 5. Any reader that is not this branch's
  `wtrace.py` stops at the unknown record type. Every reader in the repo goes through `wtrace.read`.

## What is left

- GPU: the trace run, the fit on real traces, and the screen. None of these has run; the GPU was busy.
- The C++ build of generate.cpp (HIP and CUDA): not built here (see below). The hooks were reviewed by reading.
- Entropy or top-k entropy as a feature would need an `mtp_select` kernel change (HIP and CUDA). Only worth it if the
  real-trace AUC of the classifier is close to the probability-alone AUC.
- A26 decision: run `draft_report.py` on a real `STRATA_SPEC_TRACE`. Keep A26 parked unless the counterfactual shows
  more than about 10% of rounds better served by the other source.

## C++ build

**Nothing in C++ was compiled or run here.** A GPU benchmark (abtest `09-prefill` screen) kept the machine busy
through all three `wait_quiet.sh` calls, so under the batch rules nothing was built. The coordinator's 02:00 build
needs to:

- build `disco_test` and `row_miss_test` (both host-only, in the cpu label list) and run `ctest -L cpu -R "disco|row_miss"`;
- build the HIP engine (generate.cpp includes `strata/core/row_miss.hpp` and uses `DiscoPolicy`; `strata_spec` now
  compiles `src/spec/disco.cpp`).

What was checked by reading:

- includes and types (`const int32_t*` placements from `Drive::d.trace_kind`; `std::clamp` on the static constexpr
  `kMaxT`, as the existing code already does);
- `drive` is in scope at both init points (main's `Drive drive` at line ~9032 encloses the serve and CLI blocks);
- every byte offset in `row_miss_test` against the writer;
- the expected values in `disco_test`, recomputed by hand (EMA, thresholds, CSV columns).

`disco_test` also parses a weights file exactly as `disco_fit.py` writes it. A small cross-check (the engine's
`STRATA_DISCO_TRACE` features against `disco_fit.py`'s replay of the same windows) is ready in the scratchpad but was
not run, because it needs a compile.

Note for the plan: MOD_PLAN says "spec only on feat/disco", but feat/disco (bf53e9f68) already had the full engine
side, its tests and a CSV fitter. Its hunks applied to dev/batch-2 without conflicts and are the first commit on this
branch.
