# A19 - learned thinking stop: a linear probe on the head's input

Branch `work/a19-thinking-probe` (from `origin/dev/batch-2`). MOD_PLAN Phase 16 item A19 (item 21). The per-effort caps (A17) and
the rumination close (A20) are another agent's work on `work/a17-thinking`; this item adds the engine hook, the probe tool and
one more reason for the server's existing soft close.

## What

Papers:
- Thought Calibration (MIT 2025), https://arxiv.org/abs/2505.18404 - a linear probe on hidden states at reasoning-step
  boundaries decides when the thinking can stop; the threshold is calibrated so that the risk of a changed answer stays bounded.
- Learning How Hard to Think (ICLR'25),
  https://proceedings.iclr.cc/paper_files/paper/2025/hash/ff414825df833edb8b1839e3d5d495e9-Abstract-Conference.html - a cheap
  predictor of how much thinking a query needs.

Three parts, all opt-in:

1. **`STRATA_HIDDEN_DUMP=<dir>[,every=N]`** (engine, exact). While the reply is inside `<think> ... </think>`, the vector the LM
   head reads is copied to the host at the check points and appended to one file per request. The check points are: every N
   thinking tokens (default 64), the state right after a `"\n\n"` in the thinking (also when the two newlines come in two
   tokens), and the model's own `</think>` (an END record). On Flash-Next that vector is `head_mixed_` in the verify window
   (`src/core/verify.cpp`): the final hyper-connection read with `output_hc_norm`, n_embd = 2560 floats. All four head paths
   there write it (the Q8 mixer, the fused head read, the multi-token mixer and the per-row `lm_head_mix`), on CUDA and HIP
   alike (the same source). The copy is one `cudaMemcpy` of one row, only on a check point
   (`Verifier::copy_head_row`, the same pattern as `copy_logits`). With the variable unset no code on the decode path runs.
2. **`tools/pancake/thinking_probe.py`** (offline). It reads the dumps, makes labels, trains a logistic (or ridge) probe,
   reports calibration on requests it held out, replays the engine's stop rule for a threshold grid, picks a threshold and
   exports the weights (SPRB). It has `--help` and tests.
3. **`STRATA_THINK_PROBE=<probe.sprb>,<threshold>[,k=K][,min=M]`** (engine + server, **lossy**). At the same check points the
   engine computes `sigmoid(w.h + b)` on the host (2560 multiply-adds on the row it copied). K checks in a row at or above
   the threshold (default 3), once M thinking tokens are written (default 0), make the engine print
   `TPROBE <thinking tokens> <score>` after that window's `T` lines. The server reads it (`StrataEngine.generate` sets
   `think_probe_hit`) and closes the thinking with the soft close it already has for the budget and the rumination close:
   it waits for the next sentence or line end (at most `reasoning_budget_grace` tokens, else `THINK_GRACE` 512, and at most
   half the room left), then writes `REASONING_WRAP_UP` and `</think>` and lets the model answer. Once per request. The
   per-effort budget (`reasoning_budget_by_effort`) is untouched and stays the safety net. The turn log gets
   `think_probe_closed` (true/false).

Lossy: the probe close changes the text (a shorter thinking, possibly a different answer). The dump is exact.

## How to enable

Collect dumps (exact; add to the preset's `"env"`):

    "STRATA_HIDDEN_DUMP": "/home/dan/clone-of-d/Strata-data/a19-dumps,every=64"

Each thinking request writes `hd-<unix ms>-<pid>-<seq>.bin` (records) and `.ids` (every thinking token id). A request that does
not think writes nothing. Size: about 10 KB per check point, so roughly 0.3-0.6 MB for a 1,000-token thinking.

Train and export:

    python tools/pancake/thinking_probe.py info /home/dan/clone-of-d/Strata-data/a19-dumps
    python tools/pancake/thinking_probe.py train /home/dan/clone-of-d/Strata-data/a19-dumps --k 3 --out a19.sprb --report a19.json
    # with ground-truth labels instead of the proxy:
    python tools/pancake/thinking_probe.py train <dumps> --labels gt.jsonl --k 3 --risk 0.05 --out a19.sprb

Use it (lossy):

    "STRATA_THINK_PROBE": "/path/a19.sprb,0.9,k=3,min=256"

A malformed value is ignored with a log line; a probe whose dimension is not n_embd is refused at start ("the probe is off").
Only the serial serve loop is hooked (the daily single-card preset). `--batch` slots and `--pipeline-windows` are not, and the
engine says so at start. A layer split asks the last stage for the row (`next_`), not tested.

## File formats

Dump, little-endian, version 1. Header 32 bytes: `"SHDM"`, u32 version, u32 dim, u32 every, u64 start_unix_ms,
u32 prompt_tokens, u32 request_seq. Then fixed-size records of 16 + 4*dim bytes: u32 think_index (the thinking tokens written
before the token this state produced; stopping "here" keeps exactly that many), i32 token, u32 flags (1 periodic,
2 paragraph, 4 END = the produced token is the model's own `</think>`, 8 = the probe fired here), f32 score (NaN without a
probe), f32 h[dim]. A record cut off at the end is dropped by the reader.

Probe (SPRB) version 1: `"SPRB"`, u32 version, u32 dim, u32 link (0 logistic, 1 identity), f32 bias, f32 w[dim]. The tool folds
the standardisation into the weights (w' = w/std, b' = b - sum(w*mean/std)), so the engine needs nothing else.

## Labels

The label of a check point at t thinking tokens is "the final answer would already be correct / unchanged if the thinking
stopped here".

**Ground truth (forced-stop replay).** `thinking_probe.py labels-plan <dumps> --out plan.jsonl` writes one line per check point
with the thinking prefix to force (`.ids[:t]`). For each line, send the same request again with the assistant turn prefilled as
`<think>\n` + detokenize(prefix) + `\n</think>\n\n` (greedy, the same sampling as the full run) and compare the answer with the
full run's: label 1 when it is the same tool call (name and arguments, JSON-normalised), or the same final text (whitespace
normalised), or - for tools/pancake/task_gate.py tasks - when the task's checker passes. Write
`{"file": "hd-....bin", "think_index": t, "label": 0|1}` lines and train with `--labels`. The dumps do not store the prompt (the
turn log stores no text either), so the replay needs the prompts from where the requests came from: the task gate's task file
is the practical source (run the task gate with `STRATA_HIDDEN_DUMP` set, then replay its tasks). A dump is joined to its turn
log line by `start_unix_ms` and `prompt_tokens` (the turn log's `time` and `prompt_total`). The replay driver itself is not
written yet (it needs a running server).

**Cheap proxy (default, no replay).** Only requests whose thinking ended at the model's own `</think>` (an END record) are used;
capped or closed ones are left out. Label 1 when the point is within `--proxy-tail` tokens (default 128) of that close or past
`--proxy-frac` (default 0.9) of the thinking. This learns "the model is about to close". It is safe but its savings are
bounded by the tail (about 10% of the thinking at the defaults). The early "the answer is already decided" points that the
papers report (-20-40% of the thinking on routine turns) need the ground truth.

## Calibration report

On held-out requests (split by request, never by row): AUC, Brier, log loss, ECE (10 bins) and the reliability table; then the
engine's stop rule (K in a row, min tokens) replayed per request for a threshold grid: fires, premature fires (a fire on a
label-0 point), thinking tokens saved. The recommended threshold is the lowest whose premature-fire rate's Hoeffding upper
bound stays under `--risk` (default 0.1, at `--delta` 0.1). This is a simple form of the paper's risk control (it uses
Learn-then-Test).

## Files changed

- `include/strata/program/think_probe.hpp` (new): settings, check points, paragraph tracker, stop rule, probe file, dump writer,
  `TPROBE` line. Header-only, no GPU.
- `tests/program/think_probe_test.cpp` (new) + `CMakeLists.txt` (registered next to `route_prior_scope_test`, in the CPU list).
- `include/strata/core/verify.hpp`, `src/core/verify.cpp`: `Verifier::copy_head_row(t, host)`.
- `src/program/generate.cpp`: startup parse (vocabulary from the pack's tokenizer files, or `--grammar`'s), per-request state,
  the scan before the commit, the `TPROBE` line after the T lines.
- `serve/server.py`: `TPROBE` parse in `StrataEngine.generate`, the probe close in `release()` (beside the rumination close),
  the wrap-up log line, `think_probe_closed` in the history / turn log.
- `serve/test_think_probe.py` (new), `tools/pancake/thinking_probe.py` (new), `tools/pancake/test_thinking_probe.py` (new),
  `bench/specs/16-a19-hidden-dump.json` (new).

## Tests run

See the coordinator report for the exact results of this branch:
- `think_probe_test` (host-only C++, built with g++ alone outside CMake).
- `python -m pytest tools/pancake/test_thinking_probe.py`.
- `python -m pytest serve/test_think_probe.py serve/test_rumination.py`.

Not built: the engine (`strata`, HIP and CUDA). `generate.cpp` and `verify.cpp` were only re-read for syntax, types and
includes. Not run: any GPU test, the spec.

## Spec to run

`bench/specs/16-a19-hidden-dump.json`: base vs the dump arm (exact). The text hashes must be equal, the speed the same (one row
copy per check point). If the cases think, the dump directory must hold `hd-*.bin` files afterwards; delete it after the run.
The probe arm is not in the spec: it needs weights trained on real agent dumps, and it is judged with the task gate (A55),
not with a speed spec.

## Expected gain and risk

- Gain: the papers report -20-40% thinking on routine turns with ground-truth probes; untested on agents. With the proxy labels
  expect at most about the tail (~10% of the thinking). The A17 analysis (34 agent turns, p99 1,061 thinking tokens, thinking
  22% of turn time) bounds the turn-time gain at a few percent today.
- Risk: a premature close changes the answer (lossy); the risk bound and K checks in a row limit it, the effort cap stays.
  The dump's cost is one synchronous device-to-host copy (10 KB) per check point. A non-finite window (#879) whose tokens never
  go out may still leave its records in that request's dump file (the retry writes a new file).

## What is left

1. Build and run `think_probe_test` and the spec on the 7900 XT; check the dump arm's hashes equal base.
2. Collect dumps from a day of agent turns plus a task-gate run.
3. Write the forced-stop replay driver (labels-plan -> server -> labels JSONL) and train on ground truth.
4. Screen the probe close with the task gate (A55) against the effort caps alone (A17).
5. If it helps: the batch-slot path (per-slot state) and a GPU-side score are not needed for one card.
