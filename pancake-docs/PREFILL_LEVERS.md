# Prompt chunk plan and the staged-path floor (MOD_PLAN 17.0 lever 3, 17.5)

2026-11-11, branch `dev/p17/prefilllevers` (from `origin/dev/p17/integration` `17b11ef85`). Code only: nothing was built,
no engine was started, no GPU was used. Sources: the code at that commit, the RX 7900 XT engine log of real traffic
(`pancake-flashnext.log`, read only), `pancake-docs/TTFT_LEVERS.md`, three Strata-rdna3 commits (`791cca4b`, `5906629d`,
`2b19d510`), upstream issue #5539 and the 0xluk notes in `research_notes/Active Strata forks November/batch4.md`.

Labels, as in TTFT_LEVERS.md: **[meas.]** benchmark on this box; **[log]** engine log of real traffic on this box;
**[code]** unconfirmed (read from code); **[est.]** estimate (the arithmetic is given); **[ext.]** another machine.

## 0. The answer in short

1. **The plan is a pure function of the segment length and the start's auto chunk** [code]. The serve loop cuts a prompt
   at the system prompt's end (when it is 2,048 tokens or more), at the last turn boundary and before the last token;
   each segment reads in chunks of `min(chunk, segment rounded up to 256)`, full chunks first, the rest last. The auto
   chunk was 32,768 on 359 of 374 starts in the log, 31,488 on 4 of them (11-09, the day of the 70K read) [log].
2. **Short-tail risk: real** [code]. A single-message prompt of 100,000 tokens reads `32,768 x 3 + 1,693`; ttft_bench's
   "100K" (102,400) reads `32,768 x 3 + 4,093`. Those last chunks are below the staged limit (4,096), so they take the
   staged walk while the segment still holds the loan of its 32,768-token chunk (9,413 of 10,733 cache slots [log]).
   Est. ~2.8 s of serial PCIe copy for that tail, of which 1.5-2.5 s could go: 4-6% of a 100K read [est., section 3.1].
3. **The 3x32K handoff (#5539): geometry reachable, trigger not set** [code]. A single-message segment of 65,537-98,304
   tokens reads in 3 chunks with the third at 65,536 (prompt about 65,540-98,307 tokens, the assistant header is a few
   tokens), 98,305 and up in 4. The reported fault needed `STRATA_KVA=1` (off in our preset; KVA also declines any read
   that does not start at token 0) and `--kv-resident 49152` (ours 32,768). The KV staging pool holds every page of one
   layer, so no pool size limit is reached at 98,304 [code]. Not shown from the code; the 100K run with `STRATA_TRACE=1`
   is the test (section 3.3).
4. **Switch added, coded, not built, not measured: `STRATA_PREFILL_EQUAL_TAIL=1`** (+ `_MIN`, default 256). A segment
   whose last chunk would be staged reads in the same number of equal chunks instead (100,000: `25,088 x 3 + 24,733`).
   Pure function `prefill_plan::equal_tail_chunk` with tests. Default off, the default path unchanged.
5. **The "40% PCIe share" is not a fixed 40%** [code]. The CPU share can only take experts that 8 tokens or fewer route
   to (`kernels::cpu::MAXT`). That covers almost every routed expert at 100-200 new tokens, about 60% at 400 and almost
   none from 700 on [est.]. So `STRATA_PREFILL_CPU_SHARE` 0.65 / 0.7 / `auto` can only move warm turns of up to about
   400 new tokens; from about 500 new tokens the staged walk copies nearly every non-resident routed expert over PCIe,
   one layer at a time, after that layer's routing (section 6).
6. **PF_AHEAD and REFILL_ASYNC are in no local fork clone** (only in notes; the code is in 0xluk/Strata on GitHub).
   Not ported. The streamed path already overlaps copies with the attention, so the cheaper first arm is an env A/B of
   `STRATA_PREFILL_CPU_SHARE_MAX` 4096 vs 2048 vs 1024, which sends 1-4K-token deltas to the streamed path (section 6.4).

## 1. How a prompt is cut into chunks today [code]

- **Segments** (`src/program/generate.cpp`, serve loop, `cuts`): a prompt read from token 0 has the cut points
  `root_at` (the first `<|im_start|>` after token 0, when it is at `--prompt-cache-root` = 2,048 tokens or more and before
  the turn boundary), `turn_at` (the last `<|im_start|>`, so the assistant header after it is the next part) and `n - 1`
  (the last token goes through the first decode window). `STRATA_CACHE_MESSAGE_BOUNDARY`, `STRATA_CACHE_TOOL_BOUNDARY` and
  `pin=` add cuts when they are on (off in the preset). A part of `--short-read` (64) tokens or fewer goes through
  decode windows.
- **The chunk of a segment** (`lend` -> `planned_chunk` -> `equal_chunk` -> `request_chunk`): `min(o.prefill_chunk,
  segment rounded up to 256)`. `planned_chunk` uses the host-aware plan (`balanced_chunk`) only when that plan chose a
  smaller chunk than the VRAM scan; the log shows "prefill plan: ... kept" on every start that printed one, so it is
  not in play here [log]. `STRATA_PREFILL_EQUAL=1` evens the chunks out only when the last one would stream anyway.
  The segment's buffers are laid out for that chunk (`Prefill::relayout`) and the loan stays out until the segment
  ends (the refill runs before the next windows).
- **The chunk loop** (`Prefill::run_impl`, `for (c0 = 0; c0 < n; c0 += C)`): full chunks, the last one the rest.
  `STRATA_TRACE=1` prints `strata trace: prompt chunk <c0> of <n> ...` per chunk and `lent <slots> slots for <chunk>
  tokens` per segment.
- **Paths by chunk length T** (fast-hip preset): `STRATA_PREFILL_CPU_SHARE=0.6` arms the CPU share, which makes
  `stream_all_min()` equal to `STRATA_PREFILL_CPU_SHARE_MAX` = 4,096 (`Prefill::arm_cpu_share`). T of 4,096 or more:
  **streamed**, every non-resident expert of every layer in id order through the 192-slot ring (`STRATA_PREFILL_RING`),
  the copies running through the attention halves. T below 4,096: **staged**, each layer waits for its routing
  (`cudaStreamSynchronize`), then copies its routed non-resident experts through the 8-slot staging ring (32 with MMQ
  groups) while the CPU pool computes its share.
- **The auto chunk** is the largest chunk on the 256-token grid whose loan takes at most 90% of the expert cache slots
  (85% when less of the RAM copy is pinned) and leaves 128, at most `STRATA_PREFILL_AUTO_MAX` (32,768; values above it
  are cut to 32,768). It depends on the VRAM free at start: 32,768 on 359 starts, 31,488 on 4, and 31,744 / 30,208 /
  20,992 / 17,152 once each [log, "prompt chunk auto" lines; a few 8,192 / 9,472 / 256 lines come from other runs].
- **The loan at 32,768**: 9,413 of 10,733 cache slots, 12.59 GiB of 14.37 GiB (latest IQ2_XS start) [log]. During such
  a chunk about 1,320 cached experts stay resident: 5% of the 24,576.

## 2. The chunk plans

Single-message prompt (ttft_bench scenario A, one user message): the segment is the prompt minus the assistant header,
taken here as 3 tokens (`<|im_start|>assistant\n`; with `--think off` the template adds a few more) [code]. Chunk
32,768, staged limit 4,096. "s" = staged chunk. Reproduce with `python tools/pancake/chunk_plan.py <tokens> [--equal]
[--equal-tail] [--root R] [--cap C]`.

| Prompt | Segment | Today | `STRATA_PREFILL_EQUAL=1` | `STRATA_PREFILL_EQUAL_TAIL=1` | Short tail today | A chunk at p0 >= 65,536? |
| ---: | ---: | --- | --- | --- | --- | --- |
| 32,768 (bench "32K") | 32,765 | 32,765 | same | same | no | no |
| 35,000 | 34,997 | 32,768 + 2,229 s | same | 17,664 + 17,333 | **staged, 2,229** | no |
| 49,152 | 49,149 | 32,768 + 16,381 | 24,576 + 24,573 | same as today | no (streams) | no |
| 57,344 | 57,341 | 32,768 + 24,573 | 28,672 + 28,669 | same as today | no | no |
| 65,536 (bench "64K") | 65,533 | 32,768 + 32,765 | same | same | no | **no: 2 chunks** |
| 65,537 | 65,534 | 32,768 + 32,766 | same | same | no | no |
| 65,540 | 65,537 | 32,768 x 2 + 1 s | same | same (1 < 256) | 1 token (cheap) | yes, 1 token |
| 66,000 | 65,997 | 32,768 x 2 + 461 s | same | 22,016 x 2 + 21,965 | **staged, 461** | yes / no |
| 70,000 | 69,997 | 32,768 x 2 + 4,461 | 23,552 x 2 + 22,893 | same as today | **streamed, 4,461** | yes / no |
| 98,305 | 98,302 | 32,768 x 2 + 32,766 | same | same | no | yes (the #5539 shape) |
| 98,308 | 98,305 | 32,768 x 3 + 1 s | same | same | 1 token | yes |
| 100,000 | 99,997 | 32,768 x 3 + 1,693 s | same | 25,088 x 3 + 24,733 | **staged, 1,693** | yes |
| 102,400 (bench "100K") | 102,397 | 32,768 x 3 + 4,093 s | same | 25,600 x 3 + 25,597 | **staged, 4,093** | yes |
| 128,000 | 127,997 | 32,768 x 3 + 29,693 | 32,000 x 3 + 31,997 | same as today | no | yes |
| 131,000 | 130,997 | 32,768 x 3 + 32,693 | same | same | no | yes |

With both switches on, a segment is evened out whatever its tail (except tails under 256 tokens). Bands with a staged
tail at chunk 32,768: segments of 32,769-36,863, 65,537-69,631 and 98,305-102,399 tokens (tails of 256 tokens and more
are what the new switch takes). Bands with a streamed tail too short to hide its pass's copy (EQUAL's case; a tail
under ~8.5K tokens computes for less than the ~2.8 s copy, see 3.2) [est.]: about 36,864-41,300, 69,632-74,000 and
102,400-106,900.

**Agent shape.** A conversation with a system prompt (plus tools) of 2,048 tokens or more is cut there first, so the
grid moves: a 100,000-token prompt with a 12,000-token system prompt reads `12,000 | 32,768 + 32,768 + 22,461`, no short
tail, and its third chunk starts at 77,536. "3 chunks from 65,537 tokens" holds for single-message prompts only.

**Another start's chunk.** At 31,488 (11-09) the 70,082-token read was `31,488 x 2 + 7,103` if it was one segment, its
third chunk at 62,976 [code + log]; 99,997 reads `31,488 x 3 + 5,533` (the tail streams).

## 3. Risks

### 3.1 A staged short tail inside a long segment (real from the code)

What happens [code]: the tail chunk keeps the 32,768-token loan, so about 95% of the experts are not resident while it
runs. A tail of 1-4K tokens routes to nearly every expert of every layer (1 - e^-33 at 1,693 tokens). The CPU share
takes only experts with 8 tokens or fewer (none at that size, section 6.2). So each of the 48 layers waits for its
routing, then copies about 485 experts (about 690 MB at 1.42 MB each) over PCIe before its expert compute.

Cost [est.]: 690 MB at ~12 GB/s is ~57 ms per layer, ~2.8 s for the tail, against ~0.6-1.4 s for the same tokens inside
streamed chunks (0.33 ms per token, copies hidden behind ~8 s of compute per chunk). About 1.5-2.5 s saved on a
100-102K prompt (~40 s): 4-6%. A 461-token tail (66,000) leaves the CPU about 45% of the experts: ~1.5 s, about 5% of
~25 s. At 35,000 (k = 1) the two evened chunks of ~17.6K still hide their copies (est. 2.3 s copy vs 5.8 s compute):
est. up to 2.5 s of ~13.5 s.

One observation that fits [log]: on 11-09 (chunk 31,488 that start) a cold 34,261-token read took 17.1 s (2,000 tok/s,
only 177.6 MB read from storage), while 32,049-token reads ran at 2,520-2,591 tok/s and 50-62K reads at 2,388-2,647.
If it was one segment, its plan was `31,488 + 2,770 s`; at 2,550 tok/s it would have taken ~13.4 s, so ~3.7 s more. One
sample, and the log does not say whether it had a system-prompt cut: consistent, not proof.

### 3.2 A streamed short tail

A 70,000-token prompt's third chunk of 4,461 tokens streams the whole non-resident set (~31 GiB with the 32,768 loan out,
~2.8 s at 12 GB/s) behind ~1.5 s of compute: est. ~1.3 s exposed, ~5% of a 70K read [est.]. `STRATA_PREFILL_EQUAL=1`
(exists, PR #693) already covers it (`23,552 x 2 + 22,893`). TTFT_LEVERS' "~19 GB per chunk" left the loan out; with the
loan the streamed set per 32K chunk is ~31 GiB [est. from the log's slot counts].

### 3.3 The 3x32K chunk handoff, upstream #5539

The report [ext.]: RX 7900 XTX, engine 0.1.55, IQ3_S, `STRATA_KVA=1` (split 36, suffix 2048), `--prefill auto` (32,768),
`--kv int8`, `--max-context 131072`, `--kv-resident 49152`; one 93,821-token document read as 32,768 + 32,768 + 28,285;
"Memory access fault ... Page not present" about 90 s into the read, 3 of 3; single-chunk prompts and a 4,700-token
KVA request were clean; no multi-chunk run without KVA was reported; no maintainer reply.

Our side [code]: the same chunk geometry happens for single-message segments of 65,537-98,304 tokens. KVA is off unless
`STRATA_KVA=1` (`kva.hpp`, "off (STRATA_KVA is not 1)") and declines a read that does not start at token 0. The KV staging
pool (`take_stage`) is sized for `n_pages` of one layer, i.e. the whole context, and every streamed QSA layer stages
`[0, p0)` into it per chunk, so p0 = 65,536 or 98,304 is inside it. What has run [log]: the 70,082-token read of 11-09
(31,488-chunk start, three chunks, KV streaming on), clean; a 131,071-token prefill in a worktree build, chunk not recorded,
clean (`docs/PANCAKE_FINDINGS.md`). No preset run with a chunk starting at 65,536 or later is in the log.

What a test must show: one single-message prompt of 100,000 tokens (and one of ~98,400, the exact 3-chunk shape) on the
exact preset with `STRATA_TRACE=1`. The trace must list `prompt chunk 0 / 32768 / 65536 / 98304 of 99997` (4 chunks
today), the `lent ... for 32768 tokens` line, a completed read and a reply. A fault prints the chunk it was in. A second
run with `STRATA_PREFILL_EQUAL_TAIL=1` must list chunks at 0 / 25088 / 50176 / 75264.

### 3.4 A trap for the measurements

ttft_bench calibrates its prompts to about 8,192 / 32,768 / 65,536 / 102,400 tokens. "64K" therefore lands right at the
2-to-3-chunk boundary (2 chunks at 65,536, 3 with a 1-token tail a few tokens above it), and "100K" lands 3 tokens under
the staged limit of its tail. A calibration error of a few tokens flips the plan between runs. For the chunk study use
sizes away from the boundaries (below), and recompute each run's plan from its recorded prompt tokens with
`chunk_plan.py`.

## 4. Strata-rdna3's grid and equal-tail rule, compared

- `791cca4b` keeps an explicit `STRATA_PREFILL_AUTO_MAX` above 32,768 on the 8,192 grid up to 49,152 (ours cuts it to
  32,768). A 40-48K prompt then reads in one pass (their 1-GPU 32K -> 40K dip was 4-7% [ext.]).
- `5906629d` adds `STRATA_PREFILL_EQUAL_TAIL_MAX` to their `equal_chunk`: with `STRATA_PREFILL_EQUAL=1`, the "keep a short
  tail" exception holds only for chunks up to that size; past it a segment is evened out whatever its tail. Their case:
  a 57K prompt at 49,152 read `49,152 + 8,185`, the 49,152 loan took 81% of a Q4 1-GPU cache, 18.6 s against 15.35 s
  in two 28,672 chunks [ext.].
- `2b19d510` (`STRATA_REFILL_HEAD_BIG`) makes a part lent for a chunk above 32,768 refill more rows before the first
  token; it sits on their late-refill machinery (`STRATA_REFILL_HEAD*`, rows joining after the first token), which our
  tree does not have.
- For us: our ceiling is 32,768, so their 49,152 trap cannot happen. The analogue at 32,768 is the staged tail of 3.1:
  their loan problem (81%) is our 88%. Our rule differs in three ways: it needs no `STRATA_PREFILL_EQUAL`; it is keyed on
  the tail being staged (not on the chunk size); and it leaves tails under 256 tokens and evened chunks under 8,192
  alone (upstream measured a prompt 25 tokens past 8,192 read 25% slower evened out, two RTX 4090s [ext.,
  `prefill.cpp` `pipeline_chunk`]). The 49,152 grid stays a separate lever (TTFT_LEVERS lever 7; est. 2-4% at
  33-49K and 98-147K).

## 5. The switch added: `STRATA_PREFILL_EQUAL_TAIL` (coded, not built, not measured)

- `STRATA_PREFILL_EQUAL_TAIL=1` turns it on; `STRATA_PREFILL_EQUAL_TAIL_MIN=N` (default 256) is the shortest last chunk
  it evens out. Unset: nothing changes.
- Rule (`include/strata/program/prefill_plan.hpp`, `equal_tail_chunk`): a segment longer than its chunk whose last chunk
  would be shorter than `stream_all_min()` and at least `_MIN` tokens reads in the same number of chunks of equal size
  (rounded up to 256, at most the chunk). Not applied when the evened chunks would be shorter than 8,192 or than the
  staged limit, or the evened last chunk would be staged.
- Hook (`src/program/generate.cpp`, `equal_chunk`): checked first, only when the switch is on; the existing
  `STRATA_PREFILL_EQUAL` code below it is untouched. `equal_chunk` is the chunk choice of the serve path's `lend` and of
  `strata generate`'s prompt. One stderr line at start when it is on.
- What it changes: chunk geometry (rounding: KL-level differences, like `STRATA_PREFILL_EQUAL`), a smaller loan for the
  evened segment (est. ~9.6 GiB instead of 12.59 at 25,088 tokens, a shorter refill), less KV staging (100K: the chunks'
  p0 sum 150,528 instead of 196,608 cells), and the positions of the periodic checkpoints (25,088 / 50,176 / 75,264).
  Classed `KeyIfSet` in `src/core/conversation_disk_env.def` (a parked or disk state read with other chunks is a
  different rounding). Not in `sycl/` (pancake does not build it).
- Default path: when unset, `equal_tail_min` is 0 and the `if` is not taken; nothing else moved. Checked by reading the
  diff only; not built.
- Tests: `tests/program/prefill_plan_test.cpp` (the existing `prefill_plan_test`, registered in CMakeLists.txt): the
  table's cases, the 31,488-chunk cases, the guards, the env parsing, and invariants over 86K (segment, chunk, staged
  limit) triples. Syntax-checked with `g++ -fsyntax-only` only. The same cases in Python:
  `tools/pancake/test_chunk_plan.py` (6 tests, PASS, `nice -n 19`).

## 6. The staged-path floor of a warm turn (lever 3)

### 6.1 What a warm turn does [code]

A warm turn of 65-4,095 new tokens is one segment, one staged chunk. It lends only what its own chunk needs, so most of
the cache stays resident (~44% of the experts at decode [meas., Phase 11]). Per MoE layer: attention and router on the
GPU; the host waits for the routing (`cudaStreamSynchronize`) and downloads the routed ids; it splits the non-resident
routed experts between the CPU pool and the GPU; the GPU's ones are copied through the staging ring and computed as
they land; the CPU's rows join at the combine. The copy engine is idle through the attention half, because a layer's
copies can only be issued after its routing.

### 6.2 What the "40% PCIe share" really is

`STRATA_PREFILL_CPU_SHARE=0.6` makes the CPU take `min(candidates, round(0.6 x non-resident routed))`. A candidate is a
non-resident routed expert with **8 tokens or fewer** (`c <= kernels::cpu::MAXT`, the CPU expert kernel's row limit) and
a stable RAM blob; the ones with the fewest tokens go first. So 40% is the GPU's minimum, not its share. Share of the
routed experts with 8 tokens or fewer, uniform top-10 routing over 512 experts, per layer [est., Poisson with
lambda = 10 T / 512]:

| New tokens T | Experts routed | Of those, 8 tokens or fewer | CPU share 0.6 binds? |
| ---: | ---: | ---: | --- |
| 100 | 86% | 100% | yes |
| 200 | 98% | 98% | yes |
| 300 | 99.7% | 86% | yes |
| 400 | 100% | 62% | about |
| 500 | 100% | 36% | no: the CPU gets ~36% |
| 700 | 100% | 7% | no |
| 1,000 and up | 100% | ~0% | no: PCIe carries nearly all |

Non-resident experts are colder than average (the cache holds the hot ones), so their counts are lower and the real
numbers sit somewhat higher. The warm-turn bands of TTFT_LEVERS fit this [log]: 301-800 tokens 1.0 s, 801-2,000 1.7 s,
2,001-4,095 2.65 s. At 2K+ the GPU copies all ~285 non-resident experts per layer, ~405 MB, ~34 ms at 12 GB/s, ~1.6 s
for 48 layers, serial with the attention [est.].

TTFT_LEVERS' balance arithmetic (share ~0.68 balances ~13 ms of copy against ~9 ms of CPU per layer at 300 tokens)
holds only where the cap does not bind, i.e. up to about 400 new tokens.

### 6.3 `STRATA_PREFILL_CPU_SHARE` and its `auto` mode [code]

- A number x: the fixed share above, every layer (the gate below is not used).
- `auto`: running means (weight 0.25) of the CPU's ms per expert (c) and the GPU's ms per streamed expert (g, CUDA events
  around its expert work); share = clamp(g / (c + g), 0.05, 0.9). A gate first alternates shared and unshared layers
  until 3 wall-time ratios are in, then shares only while the shared layers cost less per non-resident expert, probing the
  other arm every 29th layer. Upstream added the gate after #1282 (RX 7900 GRE + 5700X3D: every share lost, auto's
  0.47 by 4.5% [ext.]). The same MAXT cap applies.
- HIP default: off except on a single gfx12 card; our preset sets 0.6 by hand, and `STRATA_PREFILL_CPU_SHARE_MAX=4096`
  makes every delta below 4,096 staged (MOD_PLAN item 78 asked for the `SHARE_MAX=1024` A/B; not run yet).

### 6.4 What follows for the arms

1. **Share 0.6 vs 0.7 vs `auto`** moves only deltas of up to ~400 tokens: est. -0.05 to -0.15 s there, nothing on bigger
   deltas [est.].
2. **`STRATA_PREFILL_CPU_SHARE_MAX` 4096 vs 2048 vs 1024** (env only, rounding class unchanged: the share is already on):
   deltas from the limit to 4,095 take the streamed path, where the copies run through the attention halves in a fixed
   order and no layer waits for its routing. At 1-4K tokens the CPU takes almost nothing anyway, and nearly every
   expert is routed, so the bytes are about the same. Est. -0.1 to -0.4 s on 1-4K deltas, possibly a loss on
   1,000-1,500 [est.]. This is the cheapest test of what `PF_AHEAD` would buy.
3. **`PF_AHEAD` port** only if (2) leaves a gap on 1-4K deltas.

### 6.5 PF_AHEAD and REFILL_ASYNC

`grep -rn 'PF_AHEAD\|REFILL_ASYNC'` over every clone in `/home/dan/clone-of-d/src` (C++, headers, CUDA) finds only
`STRATA_IO_PF_AHEAD` in `expert_source.cpp` (the I/O read-ahead queue, a different thing) and notes and readmes. The
code is in 0xluk/Strata, branch `flashnext-pcie3` (PF_AHEAD: `896481c943`, `09a550bc66`, `cd90f646ff`, `4a3faee927`,
~290 lines of `prefill.cpp`; REFILL_ASYNC: `f9b14a4965`), on an older upstream base. Their numbers are a bundle with
`PARK_REWIND_MAX`, `REFILL_ASYNC` and `ARENA_REUSE` on 2x RTX 3060 (agent turns of ~2.5K tokens: TTFT 4.97 -> 3.67 s)
[ext.]. Strata-rdna3 has a related refill idea (`STRATA_REFILL_HEAD*`, rows joining after the first token), tied to
its multi-GPU paths.

Not ported: no local code to port from, and a ~300-line change in the staged walk that cannot be built or measured
today. Sketch for a later port (credit 0xluk in README section 9 and RESEARCH_AND_FINDINGS section 4 then): after layer
l's routing, issue the copies of layer l + 1's predicted experts (last pass's counts, most tokens first, minus the
resident ones and the ones the CPU would take) into an N-slot pool carved from the loan, on the copy stream, so they run
during l + 1's attention; at l + 1's routing, use the pool's copies that were routed and copy the rest as today. The
CPU/GPU split must read the residency without the pool so the output stays the same. A pure function
`prefetch_order(prev_counts, resident, cpu_takes, pool_slots)` can carry the decision and its test.

REFILL_ASYNC: our refill queues the copies and waits once before the first window (`refill`, `STRATA_REFILL_BLOCKING`
keeps the old blocking form). After a long cold read the loan is ~12.6 GiB: est. ~1 s of refill inside that request's
TTFT [est.]; `STRATA_TRACE=1` prints `refilled <slots> slots ... in <ms>`, which measurement M1 should record before any
port. A warm turn lends little, so little to win there.

## 7. Measurement plan (night queue, one engine, nothing else running)

Bar: at least 0.1 s per warm turn, or at least 2% on long prompts, paired, run 1 discarded (as TTFT_BENCH.md).

1. **Scenario A, baseline and chunk arms.** `tools/pancake/ttft_bench.py --scenario a --runs 4` on the exact preset,
   default sizes `8k,32k,64k,100k`, then the chunk study `--sizes 30000,34000,60000,66500,70000,100000` (1 chunk;
   32,768 + ~1.2K staged; 2 chunks; 2 + ~960 staged; 2 + ~4.5K streamed; 3 + ~1.7K staged). Arms, one server each:
   base; `STRATA_PREFILL_EQUAL=1`; `STRATA_PREFILL_EQUAL_TAIL=1`; both. Record `prompt_ms` and client TTFT per size;
   with `STRATA_TRACE=1` on one run per size, the chunk lines, the `lent` line and the `refilled` line. Recompute each
   plan from the recorded prompt tokens (`chunk_plan.py`). Expected [est.]: EQUAL_TAIL -1.5 to -2.5 s at 100K, ~-1 s at
   66.5K, up to -2.5 s at 34K; EQUAL ~-1.3 s at 70K; nothing at 30K, 60K.
2. **The #5539 check** (3.3): 100,000 and ~98,400 tokens, base preset, `STRATA_TRACE=1`; then 100,000 with
   `STRATA_PREFILL_EQUAL_TAIL=1`. Pass = 4 (and 3) chunk lines at the expected p0, a reply, no "Memory access fault".
3. **Scenario E, warm-delta sweep** (the other agent's branch `dev/p17/ttftscen`): deltas of 100 / 300 / 1,000 / 3,000
   tokens at 32K and 64K depth, `STRATA_PREFILL_TIMING=1`. Arms: share 0.6 / 0.7 / `auto`; then
   `STRATA_PREFILL_CPU_SHARE_MAX` 4096 / 2048 / 1024 with share 0.6. Expected [est.]: the share arms move only the
   100-300 deltas; SHARE_MAX moves only the 1,000-3,000 deltas.
4. Only then: the 49,152 grid (lever 7) and a PF_AHEAD port.

## 8. Unverified

- Every [code] statement: the cuts, the chunk loop, the loan held through the tail, the staged walk's order of events,
  the KV staging pool's size, KVA's decline.
- The header length of the assistant turn (3 tokens assumed; more with `--think off` or an effort tail turn).
- Every [est.] cost: the PCIe rate (~12 GB/s, the engine's probe), the per-token compute (0.33 ms), the expert blob
  size (1.42 MB), the Poisson model of routing.
- The 34,261-token observation is one read, its cuts unknown.
- `STRATA_PREFILL_EQUAL_TAIL` is not built and not run; its effect on quality is assumed to be the rounding class of
  `STRATA_PREFILL_EQUAL` (KL-level), not checked.

## 9. Build and test commands for later

    # unit tests (CPU only; the build directory is a placeholder, never build-int while it is in use)
    cmake --build <build-dir> --target prefill_plan_test conversation_disk_env_test
    ctest --test-dir <build-dir> -R 'prefill_plan_test|conversation_disk_env_test' --output-on-failure
    # Python mirror (runs today)
    nice -n 19 python3 tools/pancake/test_chunk_plan.py
    python3 tools/pancake/chunk_plan.py 34000 66500 70000 100000 102400 [--equal] [--equal-tail]
    # engine: HIP build of strata (generate.cpp changed), then the preset with and without the switch
    STRATA_TRACE=1 STRATA_PREFILL_EQUAL_TAIL=1 <preset launch>   # expect the start line "STRATA_PREFILL_EQUAL_TAIL=1: ..."
