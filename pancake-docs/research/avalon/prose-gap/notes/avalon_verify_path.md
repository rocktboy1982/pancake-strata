# Avalon MTP / speculative verify path: what it does on prose (read-only, from local code and files)

Scope and method: read-only. Nothing built, run or measured; no GPU use. Avalon source at `/home/dan/clone-of-d/Avalon`, HEAD `8b9a0f4` (confirmed with `git rev-parse`, same commit as the 2026-11-11 bench build per `matrix.sh`). Labels: **[code]** = read from source, file:line given; **[file]** = read from a saved log, json, README or script; **[derived]** = arithmetic on saved numbers; **[unverified]** = not checked.

Source notes used for cross-checking:
- `/home/dan/clone-of-d/src/Strata/pancake-docs/PROSE_GAP_AVALON.md` does **not exist** at that path. The note lives on branch `dev/p17/prosegap` and was found in worktrees, e.g. `/home/dan/clone-of-d/src/Strata/.claude/worktrees/agent-ad25cdf1dc7ea704a/pancake-docs/PROSE_GAP_AVALON.md`. I read that copy and used it only as a cross-check; its claims are restated here with my own file:line checks.
- `/home/dan/clone-of-d/src/Strata/bench/results/2026-11-11-vs-avalon/REPORT.md` exists. Its medians differ from the numbers in the brief (see the Gaps under KQ6).
- The brief's prose numbers (Pancake 100.6 / 109.7 / 106.7; Avalon 104.2 / 114.6 / 115.7) and the acceptance and tokens-per-window figures match `/home/dan/night-queue-run/RESULTS.md` lines 425-434 **[file]**. That is a separate night-queue run, not the REPORT.md run.

## Baseline command (the one used for the prose numbers)

**[file]** `/home/dan/clone-of-d/src/Strata/bench/results/2026-11-11-vs-avalon/raw/matrix.sh:36`:

    llama-server -m <IQ2_XS main> -c 131072 -b 4096 -ub 4096 -fa on -ctk q8_0 -ctv q4_0 --jinja --parallel 1
      -md .../mtp-Qwen3.8-Flash-Next-shared-IQ3K-IQ4NL-FR65k.gguf --spec-type ngram-simple,draft-mtp
      --spec-draft-n-max 2 -ubd 256 -ctkd q8_0 -ctvd q8_0 --chat-template-file ... --port 8081

The shipped launcher `/home/dan/.local/bin/avalon-server` differs in two places: `-c 262144 -b 9536 -ub 9536`. The REPORT.md note says 9536 ran out of VRAM with the smaller quants, so the bench used 4096.

**[file]** The night-queue log `/home/dan/night-queue-run/raw/srv-prose-avalon.log:8` shows the Avalon server ran with the **IQ2_XS Strata main model** (`Strata-data/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf`) and the **Avalon IQ3_KS-hybrid MTP head**. Log line `0.31.906` reads "MoE expert cache shared with the draft context". So the head was not paired with the model it came from. Its effect on acceptance is unmeasured.

---

## KQ1. How many draft tokens per window, and how are they chosen?

### Takeaway
In the benchmark command the MTP drafter proposes exactly **2 tokens per window**, one token per draft decode step, greedy argmax, with **no confidence cut** (`--spec-draft-p-min` defaults to 0). The 2 is set by the flag and overrides the code default of 3. The ngram-simple drafter runs first and can override MTP on a window, and its length is not capped by `--spec-draft-n-max`.

### Cited Findings
- Code default is 3 and p_min is 0: `n_max = 3`, `p_min = 0.0f` — [common.h:330, 334](/home/dan/clone-of-d/Avalon/common/common.h) **[code]**.
- Flag sets `draft.n_max`; `--spec-draft-p-min` sets `p_min` — [arg.cpp:4183-4195, 4240-4245](/home/dan/clone-of-d/Avalon/common/arg.cpp) **[code]**.
- MTP draft loop stops when `result.size() >= params.n_max` — [speculative.cpp:2060-2064](/home/dan/clone-of-d/Avalon/common/speculative.cpp) **[code]**.
- The only early cut is `cur_p->data[0].p < params.p_min`, which never fires at p_min = 0 — [speculative.cpp:2042-2046](/home/dan/clone-of-d/Avalon/common/speculative.cpp) **[code]**.
- Per-request target-side cap is the context budget, `dp.n_max = n_draft_max`, from `get_n_draft_max()` (ctx minus prompt minus 2, and n_remaining - 1). It is not `--spec-draft-n-max` — [server-context.cpp:558-576, 3549-3585](/home/dan/clone-of-d/Avalon/tools/server/server-context.cpp) **[code]**.
- Draft truncation to `dp.n_max` happens in the dispatcher after each impl — [speculative.cpp:3208-3270](/home/dan/clone-of-d/Avalon/common/speculative.cpp) **[code]** (approximate line range; the truncation block is inside it).
- Measured windows: `drafted_per_window` = 2.00 and 1.99 on the MoE prompt, in `/home/dan/night-queue-run/raw/prose-avalon-p3.json` **[file]** (derived from draft_n / windows). Night-queue prose Avalon: draft acceptance 58.7 / 64.8 / 67.4 % on story / tips / moe — RESULTS.md:430-432 **[file]**.

### Inferences
- On prose, Avalon's windows are effectively always "1 sampled + 2 drafted" (3 rows). The 1.98-2.00 drafted per window on prose is consistent with ngram-simple almost never firing on these prompts (an ngram window of length L would push the average away from 2). This is an inference from averages, not per-window attribution.
- Because the MTP chain never stops early, the second draft is always sent even when its own probability is low. The Pancake note's `--spec-min-p 0.5` cut is the opposite policy (see the KQ6 gap about what that means for comparison).

### Gaps
- No per-window log of which drafter (ngram or MTP) produced each window. The "ngram rarely fires" statement is inferred from averages only.
- `nextn_predict_layers` (the MTP block count) is present in the MTP GGUF header, but its value was not read (only the key string was seen). The chain-head branch (`chain_heads = n_mtp_layers > 1`, [speculative.cpp:1461-1504](/home/dan/clone-of-d/Avalon/common/speculative.cpp)) therefore could not be confirmed as taken or not.

---

## KQ2. How does the verify step batch tokens? One forward per window? Which batch sizes?

### Takeaway
Verify is **one target decode per window**: the sampled token and its k drafts are appended to the normal server batch as one contiguous run (1 + k rows), and then sampled and matched in a single pass. `-b`/`-ub` only size the chunks of large batches. A 3-row window fits in any of the batch sizes used, so they do not change window cost. `-ubd 256` only sizes the draft context, whose steps are 1-row decodes.

### Cited Findings
- Window construction: `spec_i_batch` records the row index of the sampled token and each draft; the sampled token and drafts are added with `batch.add(...)` in one batch — [server-context.cpp:600-615](/home/dan/clone-of-d/Avalon/tools/server/server-context.cpp) **[code]**.
- Verify and accept: `common_sampler_sample_and_accept_n(slot.smpl, ctx_tgt, spec_i_batch, spec_draft)` — [server-context.cpp:4730-4748](/home/dan/clone-of-d/Avalon/tools/server/server-context.cpp) **[code]**.
- Sample-and-match loop: sample target row i, accept it, break on the first mismatch with `draft[i]`; if all drafts match, sample the bonus row — [sampling.cpp:852-882](/home/dan/clone-of-d/Avalon/common/sampling.cpp) **[code]**.
- Draft context physical batch: `-ubd` / `--spec-draft-ubatch-size` = `draft.n_ubatch` — [arg.cpp:4279-4286](/home/dan/clone-of-d/Avalon/common/arg.cpp) **[code]**.
- Target prompt chunking uses `n_batch` / `n_ubatch` — [server-context.cpp:3663-3670](/home/dan/clone-of-d/Avalon/tools/server/server-context.cpp) **[code]**.
- MTP draft steps are separate decodes on the draft context: one `llama_process(ctx_dft, DECODE, batch)` per step, `i` steps up to n_max — [speculative.cpp:1992-2010, 2085-2095](/home/dan/clone-of-d/Avalon/common/speculative.cpp) **[code]**.
- Benchmark command uses `-b 4096 -ub 4096 -ubd 256` — [matrix.sh:36](/home/dan/clone-of-d/src/Strata/bench/results/2026-11-11-vs-avalon/raw/matrix.sh) **[file]**.
- Per-window ms on prose is ~20-22 ms for Avalon (REPORT.md note, derived from saved json) — [PROSE_GAP cross-check, section 1] **[derived]**.

### Inferences
- The target-side cost of a window is one forward over 3 rows. The `-b`/`-ub` values in the bench (4096) and the shipped launcher (9536) give the same 3-row window; only prompt-side throughput and VRAM are affected. So `-b`/`-ub` cannot explain the prose decode gap.
- Each window also costs 2 draft decodes on the draft context. Their cost is part of the ~20 ms per window measured, but not separated out.

### Gaps
- The exact kernel-level cost of a 1-row MTP step vs the 3-row verify on the 7900 XT was not measured.
- Whether the draft context's 256-row ubatch affects the 1-row steps was not checked.

---

## KQ3. Acceptance rule: greedy vs sampling, min-p

### Takeaway
The benchmark is greedy. `--spec-draft-sampling` defaults to **greedy** (argmax draft, sample-and-match verify). The rejection-sampling path is used only when `temp > 0`. **min-p is not part of the speculative path at all**: `speculative.cpp` has no min-p reference. The only draft threshold is `p_min` (default 0).

### Cited Findings
- Default draft sampling is greedy: `probabilistic = false` — [common.h:338](/home/dan/clone-of-d/Avalon/common/common.h) **[code]**; the flag help prints the default from this field — [arg.cpp:4256-4270](/home/dan/clone-of-d/Avalon/common/arg.cpp) **[code]**.
- In greedy mode the draft's result_q is cleared and the draft token is `cur_p->data[0].id`: [speculative.cpp:1951-1957, 2034-2038](/home/dan/clone-of-d/Avalon/common/speculative.cpp) **[code]**.
- Rejection path only when `temp > 0`: `use_spec_rejection()` — [server-context.cpp:545-547](/home/dan/clone-of-d/Avalon/tools/server/server-context.cpp) **[code]**.
- Target sampling defaults: `min_p = 0.05`, `temp = 0.80` — [common.h:235, 239](/home/dan/clone-of-d/Avalon/common/common.h) **[code]**. These apply only if the client does not request greedy. The bench uses greedy **[file]** (REPORT.md "greedy decoding").
- `grep -n "min_p" common/speculative.cpp` returns nothing **[code]**.
- Draft top-k 10 is applied on the draft sampler chain (backend, when enabled): [speculative.cpp:1484-1493](/home/dan/clone-of-d/Avalon/common/speculative.cpp) **[code]**.
- README: MTP acceptance "0.55 – 0.65 on prose, up to ~0.95 on code"; head agrees with the model on "80.7% of the positions of a fixed text (teacher-forced)" — [README.md:160-162](/home/dan/clone-of-d/Avalon/README.md) **[file, self-reported, unverified]**.

### Inferences
- Under greedy, a draft token is accepted iff it equals the target's argmax at that row. Acceptance is therefore a plain agreement rate between the head's argmax and the target's argmax, with no probability threshold on either side.
- Min-p does not affect acceptance in this configuration.

### Gaps
- No measured teacher-forced agreement for this head on prose (the 80.7% figure is the README's).

---

## KQ4. ngram-simple + draft-mtp combination (`--spec-type`) and its prose contribution

### Takeaway
Priority is set by the order of impls, not by the flag order: **ngram-simple is tried first**, and if it returns any draft for a sequence, MTP is skipped for that window. ngram-simple matches the last 12 tokens against history and copies up to 48 following tokens. It has no minimum-hit check on this path, and it is **not capped at 2**. On prose its contribution is expected to be small (an exact 12-token repeat is needed); the saved drafted-per-window averages suggest it rarely fires.

### Cited Findings
- Priority order: NGRAM_SIMPLE before DRAFT_MTP — [speculative.cpp:3036-3055](/home/dan/clone-of-d/Avalon/common/speculative.cpp) **[code]**.
- Dispatcher: each impl drafts only for sequences still `drafting`; a non-empty result sets `drafting = false` (so later impls, including MTP, skip that sequence) — [speculative.cpp:3208-3260](/home/dan/clone-of-d/Avalon/common/speculative.cpp) **[code]**.
- ngram-simple draft: needs at least 12 + 48 + 1 tokens of history; searches backwards for the most recent exact match of the last 12 tokens; copies `min(48, cur_len - (match_pos + 12))` tokens — [ngram-map.cpp:49-108](/home/dan/clone-of-d/Avalon/common/ngram-map.cpp) **[code]**.
- ngram-simple's `min_hits` is not read in `common_ngram_simple_draft`; `min_hits` is used only in the map-based path at [ngram-map.cpp:401](/home/dan/clone-of-d/Avalon/common/ngram-map.cpp) **[code]**.
- ngram-simple config: size_n = 12 (`ngram_simple.size_n`), size_m = 48 — [common.h:367-369](/home/dan/clone-of-d/Avalon/common/common.h) **[code]**; impl at [speculative.cpp:2136-2175, 3081-3095](/home/dan/clone-of-d/Avalon/common/speculative.cpp) **[code]**.
- `--spec-draft-n-max` sets only the MTP cap; ngram-simple's cap is its own `size_m` — [speculative.cpp:2713-2722](/home/dan/clone-of-d/Avalon/common/speculative.cpp) **[code]** (`common_speculative_n_max`).
- Measured: Avalon `drafted_per_window` 1.98-2.00 on the three prose prompts and 2.0 in `prose-avalon-p3.json` **[derived]**; on code (JS / LRU) 1.99-2.12 **[derived, PROSE_GAP cross-check]**. An ngram window of length L would raise this average above 2 if it happened often.

### Inferences
- On prose the ngram-simple component contributes close to nothing in these runs. Its effect is at most on the few windows that hit an exact 12-token repeat (a repeated phrase or a copied passage). Those windows would also skip MTP for that step.
- The `ngram-simple,draft-mtp` combination therefore behaves like MTP-only with depth 2 on prose and code alike, with occasional long ngram windows on copied text. The README's "up to ~0.95 on code or copied passages" acceptance is consistent with that, but I did not verify the copied-text path.

### Gaps
- No per-window trace separating ngram and MTP windows. The "rarely fires" statement is an inference from averages.
- The long ngram-window path (up to 48 drafts, which can exceed the recurrent-state snapshot size `n_rs_seq`, [server-context.cpp:4722](/home/dan/clone-of-d/Avalon/tools/server/server-context.cpp)) was not traced to its checkpoint behaviour.

---

## KQ5. MTP head handling: draft vocabulary, shared tensors, LM head

### Takeaway
The head is a `qwen4exp` architecture (the same arch as the main model in this family), with an FR-Spec-style **cut LM head over 65k tokens**. Its logits are scattered to the full target vocabulary with -INF elsewhere, so the drafter can only propose ids in the 65k set, and greedy verification remains target-exact. It shares the MoE expert cache with the target. The head has its own LM head in this file, so the target's output is not taken.

### Cited Findings
- Architecture of both files is `qwen4exp` (GGUF header strings: `general.architecture = qwen4exp` in the MTP file and in the main IQ3_KS shard) **[file]** (read with `strings` on the first bytes; not a full GGUF parse).
- The MTP file carries the key `qwen4exp.nextn_shared_target_tensors` and `qwen4exp.nextn_predict_layers` **[file]**.
- Cut head: when the MTP-only file has a `d2t` tensor, the output is `n_vocab_out = d2t.ne[0]` (65k per the file name and README) — [qwen4exp.cpp:287-296](/home/dan/clone-of-d/Avalon/src/models/qwen4exp.cpp) **[code]**.
- Scatter to full vocab with -INF outside the cut, via `ggml_set_rows` on `d2t` — [qwen4exp.cpp:889-905](/home/dan/clone-of-d/Avalon/src/models/qwen4exp.cpp) **[code]**.
- MTP context uses a plain attention KV cache holding only the nextn layer(s) — [llama-model.cpp:2585-2588](/home/dan/clone-of-d/Avalon/src/llama-model.cpp) **[code]**.
- Shared IO helper: takes the target's token embedding and output only when the draft lacks its own — [llama-model.cpp:3155-3171](/home/dan/clone-of-d/Avalon/src/llama-model.cpp) **[code]**. With a cut head present, the draft keeps its own output.
- Vocabulary compatibility check: text must match for ids from 5 up to min(vocab), with max size difference 128 — [speculative.cpp:31-32, 136-160](/home/dan/clone-of-d/Avalon/common/speculative.cpp) **[code]**.
- "MoE expert cache shared with the draft context" — [srv-prose-avalon.log:0.31.906](/home/dan/night-queue-run/raw/srv-prose-avalon.log) **[file]**; the code path is `GGML_MOE_SHARE_DRAFT` per PROSE_GAP cross-check **[unverified here]**.

### Inferences
- A 65k head gives a larger drafting vocabulary than Pancake's 40.5k English draft vocab (figure from PROSE_GAP cross-check, not verified here). The head's argmax can differ from the target's when the target's top token is outside the 65k cut. That is a plausible source of lower per-position agreement, but no measurement here supports the size of the effect.
- The head's weights (IQ3_K/IQ4_NL) and its vocabulary cut are both fixed by the file. Their combined effect on agreement cannot be separated from code alone.

### Gaps
- The 65k token list (which ids are kept) was not inspected.
- Whether the head's tensor precision (IQ3_K / IQ4_NL, 1.48 GB) vs Pancake's Q2_0 (0.71 GB) moves per-position acceptance is unmeasured here.
- The head was paired with the IQ2_XS main model in the night-queue run, not with the IQ3_KS hybrid it was quantized for; the effect of this mismatch on acceptance was not measured.

---

## KQ6. Prose-specific logic, and the numbers the brief quoted

### Takeaway
I found **no prose-specific branch** in the speculative, ngram, server or sampler code: no text-type detection, no per-prompt policy, no language heuristics. Avalon's prose advantage in the saved numbers is not explained by a prose mode in the code. The explanation has to come from the fixed policy (always 2 drafts, no cut) and the head.

### Cited Findings
- Search for prose, english or language detection in `common/speculative.cpp`, `common/ngram-map.cpp`, `tools/server/server-context.cpp`, `src/llama-context.cpp`: no matches **[code]**.
- Tail skip default 7 in the expert cache (lossy, changes which experts are computed for main-model layers): [moe-cache.cu:9677-9684](/home/dan/clone-of-d/Avalon/ggml/src/ggml-cuda/moe-cache.cu) **[code]**. The bench did not set `GGML_CUDA_MOE_TAIL_SKIP`, so the default 7 applies **[file: matrix.sh env + code]**; this is a confound for the prose comparison, not a prose-specific rule.
- Saved prose numbers, night-queue run: Pancake `prose-base-01A` 100.6 / 109.7 / 106.7 tok/s, Avalon `prose-avalon` 104.2 / 114.6 / 115.7; tokens per window Pancake 1.84 / 2.04 / 1.99, Avalon 2.17 / 2.29 / 2.34; acceptance Pancake 52.8 / 62.8 / 58.6 %, Avalon 58.7 / 64.8 / 67.4 % — [RESULTS.md:425-434](/home/dan/night-queue-run/RESULTS.md) **[file]**.
- Separate 2026-11-11 run (REPORT.md): Avalon prose 102.2, pancake fast-hip prose 98.8 on IQ2_XS — [REPORT.md table](/home/dan/clone-of-d/src/Strata/bench/results/2026-11-11-vs-avalon/REPORT.md) **[file]**.

### Inferences
- The fixed draft policy (2 drafts, no cut, greedy) gives Avalon 2.2-2.3 tokens per window on prose against Pancake's 1.8-2.0 in the night-queue run. That gap (about +0.2 to +0.3 tokens per window) is the most direct code-level explanation I found. Whether it is also the head's quality is not settled by code alone.
- The two sets of numbers disagree in size (REPORT.md shows a much smaller prose gap). The gap should be treated as run-dependent until both are re-measured on one day.

### Gaps
- The brief's Avalon numbers come from a run whose launcher and tail-skip environment are not recorded in RESULTS.md. The night-queue `srv-prose-avalon.log` shows the IQ2_XS main model and the IQ3_KS head, not the bench flags.
- REPORT.md and RESULTS.md give different prose medians for the same engines; the cause (run date, prompts, number of requests) was not reconciled.

---

## Summary of differences from Pancake's MTP verify path (code-level only)

| Area | Avalon (this config) | Pancake (per PROSE_GAP cross-check, not re-read here) |
| --- | --- | --- |
| Draft depth | Fixed 2, always sent (`--spec-draft-n-max 2`, p_min 0) [speculative.cpp:2060, 2042] | Up to 4, chain cut at probability < 0.5 |
| Draft selection | Greedy argmax; no cut | Greedy with per-step probability check |
| Window | 1 target decode over 1 + k rows | Same shape, 1 + kept drafts |
| Acceptance | Sample-and-match (greedy target) [sampling.cpp:852] | Not re-read here |
| min-p | Not used in the speculative path | `--spec-min-p` flag exists in Pancake |
| Ngram drafter | ngram-simple first, 12-gram match, up to 48 tokens, not capped by n-max | Suffix / lookup drafter, `--lookup-max-t 12` (per PROSE_GAP) |
| Head vocab | 65k cut head with -INF scatter [qwen4exp.cpp:287-296, 889-905] | 40.5k English draft vocab (per PROSE_GAP) |
| Head precision | IQ3_K / IQ4_NL, 1.48 GB (file name) | Q2_0, 0.71 GB (per PROSE_GAP) |
| Batch flags | `-b 4096 -ub 4096 -ubd 256` (bench); shipped launcher 9536 | Not compared here |

## Next checks that would close the gaps (not run)

1. Per-window trace for Avalon that tags each window as ngram or MTP (the server already logs per-slot draft stats; the tag needs a patch or a debug build, so it is not free).
2. Read `nextn_predict_layers` and the `d2t` vocabulary from the MTP GGUF with the repo's `gguf-py` (read-only).
3. Run Avalon with `GGML_CUDA_MOE_TAIL_SKIP=0` on the same prose prompts (the control the PROSE_GAP note asks for).
4. Re-measure Avalon's head on its own IQ3_KS hybrid main model, to separate head quality from the IQ2_XS pairing.

## Credits

Nothing here is adopted into Pancake, so no README section 9 or `RESEARCH_AND_FINDINGS.md` section 4 rows are needed yet.
