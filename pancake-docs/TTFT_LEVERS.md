# Time to first token in long-context agent turns: what it is made of and what can cut it (MOD_PLAN 17.0)

2026-11-11, branch `dev/p17/ttftlevers` (from `dev/phase16-academic` `a4500666e`). Analysis only: no build, no engine,
no GPU. Sources: the code at that commit, the RX 7900 XT logs of real traffic (`pancake-turns.jsonl`, 1,578 requests,
2026-11-09 16:15 to 11-11 15:50, and `pancake-flashnext.log`, both read only), the 2026-11-11 vs-Avalon report
(`bench/results/2026-11-11-vs-avalon/` on `origin/port/rdna3-decode`), the 11-08 final suites, earlier pancake notes
(`pancake-docs/batch3/percache.md`, `reports/Active Strata forks November.md`), three Strata-rdna3 commits and upstream
issue #5539.

How each claim is labelled:
- **[meas.]** measured on this box by a benchmark (source named);
- **[log]** taken from the engine log or turn log of real traffic on this box (numbers, no text);
- **[code]** unconfirmed (read from code);
- **[est.]** my estimate or extrapolation (the arithmetic is given);
- **[ext.]** another machine or a paper.

The owner's rationale: an agent workflow of 10 sequential calls pays every second of TTFT 10 times. This note asks
where those seconds are on this box and which ones can go.

## 0. The answer in short

1. **When the conversation cache holds the conversation, pancake already re-reads only about 1% of the prompt.**
   46 warm agent turns at 34-82K context re-read 27,404 of 2,699,749 prompt tokens (1.02%) and took 0.26-3.4 s to the
   first token, median about 1 s [log]. That is the ideal: a turn should re-read only the new tool result or user
   message.
2. **The large TTFT losses are cache misses.** On 11-09 one agent conversation of 54-62K tokens was re-read from token 0
   on 7 turns in a row, 21-25 s each [log], because the engine refused to park it ("skip parking (snapshot 1574 MiB
   exceeds available budget)") while short side requests ran between its turns. One miss costs 20-40 s at 50-100K; that
   is the only lever in the "5 s per request" class. It is exact and mostly configuration (see lever 1).
3. **A warm turn has a floor of about 0.55-0.75 s** before the prompt length matters: every prompt pass of 65 or more new
   tokens touches nearly every expert of every layer, so the non-resident ones go over PCIe 3.0 or to the CPU once per
   pass [log + code]. Reads of 64 tokens or fewer go through decode windows at about 9 ms per token [log + code].
4. **Parking is synchronous on HIP** and sits inside the next request's TTFT: about 0.1 s for a short conversation,
   0.2 s at 25K, 0.3-1.4 s at 50-77K [log]; HIP has no async park (`park_async_on()` returns false) [code].
   13% of warm turns (6 of 46) also missed the live state and fell back to the turn checkpoint, re-reading
   0.2-1.8K tokens and paying a park of a state they never use again [log].
5. **Prompt speed at depth is not the main problem.** Fresh 54-62K prompts read at 2,388-2,647 tok/s on 11-09 [log],
   about 10% below the 25K rate (2,776 tok/s on 11-11 [meas.]). Extrapolated: 64K about 25-26 s, 100K about
   38-45 s [est.]. BFLA does not apply to Flash-Next (its QSA attention is already sparse).
6. **The first long prompt after a start or a page-cache flush can be 30-40% slower**: a 57K prompt read
   53.7 GB from the NVMe and took 34.9 s instead of about 23 s, because the experts that are not in the RAM arena
   (28.3-30.5 GiB of about 33 GiB are) are read from the GGUF through the page cache on every pass [log].

Top three to test first: (1) conversation-cache admission (`--conversation-cache-min-tokens 8192`, a budget floor,
`STRATA_PARK_TRUNCATE=1`); (2) parks off the critical path plus a diagnosis of the checkpoint fallbacks; (3) the
staged-path floor (CPU share balance, then `PF_AHEAD`). Section 6 has the reasons, section 7 the measurements that
must come first.

## 1. TTFT anatomy of one agent turn

### 1.1 Path of a request

| Step | Where | What it costs on this box | Scales with |
| --- | --- | --- | --- |
| HTTP, `prepare()`: images, loop guard (`_guarded_messages`), agent loop notes, context dedup (off), template render, tokenization | `serve/server.py` 6347-6420, `frontend.ChatTemplate` | not measured. The template is rendered over the whole history every request; tokenization is incremental (#567: only the text after the last special token shared with a recent prompt) [code]. Est. tens of ms at 100K | history length (render), new text (tokenize) |
| Queue and pipe to the engine, first window after the prompt | `Service._run` (`first_engine_token_s` starts when the request enters the engine queue) | `first_engine_token_s - prompt_ms`: p50 17 ms, p90 23 ms, p99 103 ms over 1,576 requests [log]. A few requests waited seconds behind another one (4.8 s on a 76,505-token agent turn; the longest 36 s) | fixed, unless requests overlap |
| Prefix lookup | `generate.cpp` 17338-17346: the live state if the prompt starts with all of it, else the deepest checkpoint that is a prefix | token compare, negligible [code] | prompt length |
| Park of the outgoing live state when the request does not continue it | `generate.cpp` 17566-17590 (`park_current`, after the prompt timer starts at 17215) | mean 111 ms below 4K tokens (1,938 parks), 212 ms at 25K (94), 328-1,417 ms at 50-77K (10) [log]. Synchronous on HIP [code] | conversation size |
| Restore of a parked conversation or checkpoint | conversation cache | 130-164 ms for 57-76K parked conversations, 44-101 ms for 2-4K checkpoints [log] | conversation size (small) |
| Reading the new tokens: up to 64 through decode windows (`--short-read 64`), 65-4,095 through the **staged** path (routed experts only, 60% of the missing ones on the CPU pool, `STRATA_PREFILL_CPU_SHARE=0.6`, `_MAX=4096`), chunks of 4,096 and more through the **streamed** path (every expert the GPU does not hold goes through the 192-slot ring, once per chunk) | `prefill.cpp` 260-330, 400-460 | see 1.2 | tokens read; passes; depth |
| Checkpoints while reading | at chunk ends at least 16,384 tokens apart and at the turn boundary, up to 6 kept, ~113 MiB each | small [code; size fit from percache.md] | number of chunks |
| Refill of the expert-cache slots lent to the prompt path | "refilled before any window reads" with queued copies and one wait (`generate.cpp` 11197, 449) [code] | not isolated; est. 0.1-0.3 s after a 25-32K chunk (1-3 GB at ~12 GB/s), small after a staged delta (a request lends only what its chunk needs). Strata-rdna3 lets most rows join after the first token (`REFILL_HEAD*`), 0xluk refills in the background (`REFILL_ASYNC`, ~143 ms on a 3060 [ext.]); neither is in our tree. One unexplained case: a 33-token turn right after a 34K cold read took 3.4 s [log] | loan size |

### 1.2 Fixed part and growing part

**Fresh prompts** (11-11, `fast-hip`, IQ2_XS, `cmp_bench.py`; TTFT there is the engine's `prompt_ms`) [meas.]:
3,259 tokens 1.84 s, 12,737 tokens 4.88 s, 25,360 tokens 9.13 s. A straight line through them:
TTFT = 0.76-0.79 s + 0.32-0.34 ms per token; the slope rises a little with depth (0.321 ms per token from 3K to 13K,
0.337 from 13K to 25K). Hidden costs sit inside these numbers: each fresh prompt first parks the previous one
(25,383-token parks, 137-365 ms, mean 212 ms [log]), the slots lent to the prompt path are refilled before the first
window [code], and the 3K prompt (below 4,096) takes the staged path [code].

**Warm agent turns.** The 46 turns with at least 8,192 reused tokens (11-09 16:19-16:31 and 11-11 10:39-10:43, 34-82K
context) split into 35 that continued the live state, 6 that fell back to the turn checkpoint and 5 that restored a
parked copy after a side request (28-60 new tokens at 57-76K: 344-554 ms, one 1,932 ms) [log,
`tools/pancake/ttft_audit.py`]. The 35 live ones by tokens read [log]:

| Tokens read | Turns | Median TTFT | Range | Path |
| --- | ---: | ---: | --- | --- |
| 1-40 | 5 | 325 ms | 261-3,423 | decode windows |
| 41-100 | 3 | 558 ms | 416-599 | decode windows, then staged |
| 101-300 | 7 | 761 ms | 615-1,014 | staged |
| 301-800 | 15 | 1,003 ms | 852-1,318 | staged |
| 801-2,000 | 4 | 1,708 ms | 1,243-1,709 | staged |
| 2,001-4,095 | 1 | 2,654 ms | - | staged |

So a typical warm turn (300-400 new tokens) costs about 1 s, and more than half of it is the floor of a prompt pass.
A switch back to a parked conversation is cheap on the restore side (130-164 ms for 57-76K [log]); the cost is the park
when the other conversation goes out.

**Why there is a floor** [code + est.]. Flash-Next has 48 layers of 512 experts, top-10 routing, ~1.4 MB per IQ2_XS
expert (~33 GiB in all). About 44% of the experts sit in VRAM (10,909 of 24,576 at decode [meas., Phase 11]); fewer
during a prompt, because the prompt path borrows cache slots for its buffers. With n tokens, the share of a layer's
experts that at least one token routes to is about 1 - exp(-10 n / 512): 42% at 28 tokens, 86% at 100, all of them by
~300. On the staged path the CPU computes 60% of the missing ones and 40% go over PCIe: per layer about
0.4 x 285 x 1.4 MB = 160 MB, ~13 ms at ~12 GB/s, against ~9 ms of CPU work for the other 60% at ~26 GB/s of DDR4.
48 layers x ~14 ms is about 0.67 s, the floor seen in the table. If this is right, the floor is bound by PCIe, and a
slightly larger CPU share would balance it (lever 4).

On the streamed path (chunks of 4,096 and more) every missing expert is copied once per chunk (~19 GB, ~1.6 s at
12 GB/s), but a 25-32K chunk computes for 9-12 s, so the copy hides behind the compute; it shows only when a prompt
just passes a chunk boundary (Strata-rdna3: 32K -> 40K prompts dipped 4-7% on their box [ext.]).

**What grows with depth** [code]:
- QSA selection: the indexer scores every completed 4-cell block for every query (4 heads x 128), so its total cost
  grows as n squared; the attention itself reads at most ~2,051 cells per query, so it grows linearly. The block scorer
  runs on matrix cores (`STRATA_SELECT_WMMA=1` in the preset). On a Strix Halo the switch gained x1.008 at 8K, x1.037 at
  32K and x1.077 at 64K [ext., `arch_defaults.cpp`], so the selector's share rises with depth.
- KV streaming in the prompt path (`--kv-resident 32768`): every QSA layer stages its cells [0, p0) from host memory
  into one staging pool for every chunk (`kv_stage_from_host`, overlapped by `STRATA_KV_PREFETCH`), and writes the chunk
  back. About 1,056 B per token per layer, 12 QSA layers: at p0 = 98,304 that is ~104 MB per layer and ~1.2 GB per chunk,
  ~0.1 s at PCIe 3.0 if nothing hides it. Small against a 32K chunk's ~10 s.
- The number of streamed passes: one per chunk of `STRATA_PREFILL_AUTO_MAX` = 32,768 (see 3.2).

The one profile of a long prompt is stale: 41,818 tokens on an early 11-08 build (976 tok/s): expert GEMMs 29%, GDN
~12%, PLE 10.6% (host), QSA attention 6.7% + projections 5.5% + select 2.6% [meas., MOD_PLAN "Prompt reading on the
daily preset"]. Today's build reads 2.5-2.7x faster, so the shares have moved; a new `STRATA_PREFILL_TIMING=1` profile
at 64K and 100K is measurement M1.

### 1.3 Expected fresh-prompt TTFT at depth

| Prompt | Source | TTFT |
| --- | --- | --- |
| 25,360 | [meas.] 11-11 | 9.13 s (2,776 tok/s) |
| 32,049 | [meas.] 11-08 final suites | 12.45 s (2,580 tok/s) |
| 50,756 | [meas.] 11-08 needle test | 19.4 s (2,616 tok/s) |
| 54,313-61,565 | [log] 11-09, 7 cold re-reads, warm page cache | 21.2-25.0 s (2,388-2,647 tok/s) |
| 57,453 | [log] 11-11 10:30, 53.7 GB read from storage | 34.9 s (1,648 tok/s) |
| 70,082 | [log] 11-09, first long request after a start | 39.3 s (1,784 tok/s) |
| 64K | [est.] t = 0.329 s/K x n + 0.00123 s/K^2 x n^2 through 25K and 58K | ~26 s |
| 100K | [est.] same fit (upper) / straight line from 25K-58K slope (lower) | ~41-45 s; ~38 s if the rate holds at 2,600 |
| 128K | [est.] same fit | ~50-62 s |

For scale [ext., `bench/results/2026-10-01-ctx-ladder/matrix.md`]: upstream Strata on an RTX 4070 Ti SUPER with
IQ2_XS, int8 KV and KV streaming read 2,695 / 2,678 / 2,670 / 2,484 tok/s at 30K / 60K / 121K / 241K, so depth cost
little there; the same table's repeated requests (served from the cache, ~26 tokens read) answered in 183-725 ms
at 30-241K.

## 2. Cache reuse across turns

### 2.1 How it works [code]

- **Live state**: after a reply the engine holds prompt + generated tokens. The next request continues it when the
  new prompt starts with all of it (up to n - 1: the last prompt token is always read by the first window).
- **Checkpoints** (`--prompt-cache 6`, ~118 MB of RAM each): the turn boundary (just before the
  `<|im_start|>assistant` header; the server's `effort_end` and `--tail-role-token` keep it there), periodic ones at
  chunk ends every 16,384 tokens or more, optional tool-boundary marks (A11 / item 16, `STRATA_CACHE_TOOL_BOUNDARY`,
  off). GDN is recurrent, so a restore can only start from a checkpoint that is a prefix of the new prompt.
- **Parked conversations** in RAM (`--conversation-cache-mib`, auto-sized at start from the free RAM: 1,280-6,656 MiB
  in this log, most often 4,352; 4 slots). Snapshot size = 111 MiB + 14.9 KiB per token + 113 MiB per checkpoint
  (percache fit), so a 53K conversation with 6 checkpoints is ~1.57 GiB and a 100K one ~2.2 GiB.
- **Opt-ins**: disk tier (A13 / item 39, `--conversation-disk-dir`), `STRATA_PARK_TRUNCATE` (F1: park the deepest
  prefix that fits instead of nothing), `--conversation-cache-min-tokens N` (do not park short side requests),
  `--conversation-cache-park-gain N` (do not re-park a conversation that grew less than N tokens), `keep_reasoning`
  (item 31: the server puts back the reasoning a client dropped), pins (items 71/90, `STRATA_AUTO_PIN`: do NOT enable
  for Qwen, percache.md). On in the `fast-hip` preset: `env_to_user` (opencode's `<env>` block moves out of the system
  prompt) and `effort_position: end` (a non-default effort goes into a tail turn, so the prompt start does not change).
  Not in the preset: min-tokens, park-gain, PARK_TRUNCATE, disk tier, keep_reasoning.

### 2.2 What makes a prefix not reusable

| Cause | Effect | Handled? |
| --- | --- | --- |
| The client edits earlier messages (opencode prune of old tool outputs, compaction) | re-read from the first edited message | no; 2.6% of request time on the old opencode history [percache] |
| The system prompt changes (date, folder) | re-read of everything | opencode's `<env>` block: yes (`env_to_user`, A16); other clients: no |
| The tool list changes (tools are rendered in the system block) | re-read of everything | no [code]. Matters for clients whose tool set varies per request (MCP servers connecting, possibly Copilot); unknown |
| Effort changes between turns | re-read from the system block | yes (`effort_end`) |
| The re-rendered last assistant turn differs from the tokens the model generated (whitespace removed by the template's `trim`, tool-call formatting, text the server inserted: loop-guard notes, a forced continuation) | fall back to the turn checkpoint: the last assistant turn is read again (+0.2-1.8K tokens) and the live state is parked for nothing | partly: 6 of 46 warm turns did this [log]; cause not known (needs M3) |
| The client drops `reasoning_content` | with `preserve_thinking` on by default, only the LAST assistant turn renders differently (empty think block), so the cost equals a fallback, not a full re-read [code] | yes with `keep_reasoning` (off) |
| The conversation was not parked or was evicted (budget, short side requests, sub-agents) | full re-read | partly; this is lever 1 |
| The loop guard collapses a looping conversation | re-read from the collapsed part | rare, logged |
| Context dedup (lossy, off) | none: prefix-stable (1,482 checks, 0 mismatches) | - |

### 2.3 The clients

- **opencode**: resends reasoning (2,033 of 2,036 recorded steps), and quiet in-session steps reuse 96.5% of the
  previous state [percache]. The 46 warm turns in this log (likely opencode or a similar agent; the log has no client
  field) re-read 1.02% of their prompt tokens [log]. Sub-agents (explore, general) switch the live state and so park
  the main conversation each time.
- **Copilot through the OpenAI endpoint**: no data in the repository. Plain OpenAI chat clients usually do not send
  `reasoning_content` back; if Copilot does not, `keep_reasoning` should be on for it, and any per-request system or
  tool text would make every turn a full re-read. One captured pair of consecutive request bodies settles it (M4).

### 2.4 The ideal and the gap

An ideal agent turn re-reads the new tool result or user message plus ~10 framing tokens: about 1% of a 50-80K prompt.
That is what pancake does on a live hit. The gap is elsewhere: (a) misses, which re-read 100%; (b) the fixed ~0.6 s of a
prompt pass; (c) parks; (d) the 13% fallbacks.

Per 10-call workflow at ~60K context [est. from the log numbers]: all warm about 10 x 1 s = 10 s of TTFT; each lost
conversation +20-25 s; each sub-agent switch +0.3-1.4 s of park; a cold session start +5-15 s (a 15-40K system + tools
prefix).

## 3. Prompt speed at depth

### 3.1 QSA, WMMA prompt attention and BFLA

- Flash-Next's prompt attention is QSA: indexer, block top-k (~2,051 cells per query), then sparse attention on matrix
  cores (`STRATA_PROMPT_ATTN_IMMA=1`, `STRATA_PA_FAST=1`, `STRATA_HIP_WMMA=1`) [code].
- **BFLA does not apply.** It replaces a dense prompt attention with a block-sparse one. On the qwen35 engine it took
  128K prompts from 387 to 1,220 tok/s (3.15x, Qwen3.6 UD-Q4_K_XL [meas.]), but against our materialised GEMM path;
  its own fused dense mode is 2.0x of that. Flash-Next already attends sparsely, and the BFLA paper's own Flash-Next row
  gives BFLA 1.273x against QSA's 1.319x at 128K (both vs FA3) [ext., `pancake-docs/batch4/bfla-dense.md`]. The
  tile-shared form of QSA was closed (a shared set holds only 42-74% of a query's cells). What a port would need:
  BFLA's selection replacing QSA's selector, not the attention; that is the 2.6% "select" share of the stale 42K
  profile, before any quality question. Not worth it.

### 3.2 Chunk size and grid

- Our clamp: `STRATA_PREFILL_AUTO_MAX` above 32,768 is cut to 32,768 (`generate.cpp` 3868) [code]. The prompt path
  takes the largest chunk whose buffers fit in 90% of the expert-cache slots; a request lends only what its own chunk
  needs [code].
- Strata-rdna3 (nicejunjie, new commits, so in scope):
  - `791cca4b` keeps an explicit ceiling on the 8,192 grid up to 49,152 (one chunk for a 40-48K prompt instead of two;
    their 1-GPU 32K -> 40K dip was 4-7%); 49,152 is the last step before the 40 KB-per-token R buffer passes 2 GiB.
  - `5906629d` adds `STRATA_PREFILL_EQUAL_TAIL_MAX`: with 49,152 a 57K prompt read 49,152 + 8,185, the loan took 81% of
    a Q4 1-GPU cache, so both passes streamed nearly every expert (18.6 s vs 15.35 s in two 28,672 chunks).
  - `2b19d510` adds `STRATA_REFILL_HEAD_BIG`: a bigger loan needs more refilled rows before the first token
    (Q4 2-GPU 72-80K: +3.4% ms per window without it).
- For us [est.]: a 49,152 grid saves one pass on 32,769-49,152-token prompts and on 98,305-147K ones; at ~0.5-1 s
  of visible pass cost that is 2-4% on those sizes and nothing at 64K (two chunks either way). It costs ~0.6 GiB more
  loan, needs the equal-tail rule and the refill head, and changes chunk geometry (rounding, and the disk-cache key
  marks `STRATA_PREFILL_AUTO_MAX` "Unsure"). Measure M1 first.
- `STRATA_PREFILL_EQUAL=1` (opt-in, exists) splits a multi-chunk prompt evenly instead of leaving a small tail; today a
  tail below 4,096 tokens takes the staged path [code]. `STRATA_PREFILL_LAYER_MAJOR=1` (exists; the group-scratch fix
  `cfcde96a3` is in) streams a layer's experts once per super-chunk; unmeasured at 64-100K. Both change rounding.

### 3.3 The device fault at a 3x32K chunk handoff (upstream issue #5539)

- The report (xyzzing, RX 7900 XTX, engine 0.1.55, Qwen3.8-Flash-Next IQ3_S): `STRATA_KVA=1`, `--prefill auto` (chunk
  32,768), `--kv int8`, `--max-context 131072`, **`--kv-resident 49152`**; a 93,821-token request (32,768 + 32,768 +
  28,285) dies ~90 s into the read with "Memory access fault ... Page not present", 3 of 3; single-chunk prompts are
  clean; no multi-chunk run without KVA was reported; no maintainer reply or fix [ext.]. The ledger lists it as the
  "KVA fault".
- **Can our chunking reach it? Yes** [code]: with `STRATA_PREFILL_AUTO_MAX=32768` and no `STRATA_PREFILL_EQUAL`, a prompt
  of 65,537-98,304 tokens reads in 3 chunks (the third starts at 65,536) and 98,305-131,072 in 4 (a 4th at 98,304).
  Differences: our preset has KVA off (`STRATA_KVA` is research, off by default) and `--kv-resident 32768`.
- What this box has run: a 70,082-token prompt on 11-09 with the preset, no fault [log] (by the chunk rule: 32,768 +
  32,768 + 4,546 [code]); a 131,071-token prefill on 11-09 in a worktree build (`STRATA_SEL_OVERLAP=all`, chunk size not
  recorded), no fault [meas., `docs/PANCAKE_FINDINGS.md`]. No 4-chunk read with the preset is in the logs. Risk: open
  but unconfirmed; the 100K harness case is the test (run it with `STRATA_TRACE=1` so the chunk of a fault is in the
  log).

## 4. Other levers

- **CPU share and ring.** The share is fixed at 0.6 for chunks below 4,096; `auto` measures each side's time per expert
  and hands the CPU g / (c + g) [code]. If the floor is PCIe-bound (1.2), 0.65-0.7 or `auto` should balance it:
  est. -0.1 to -0.15 s per warm turn. The ring is set to 192 for every chunk size; on an R9700 192 cost 1.3% at 32K
  chunks vs 96 [ext., `prefill.cpp` 422-440]; a ~1% knob.
- **Next layer's expert copy during its attention** (`STRATA_PF_AHEAD`, 0xluk, not in our tree): staged path only. On
  a 300-token delta the attention half of a layer is ~1 ms against ~13 ms of copy, so little to hide; on 2-4K deltas
  it can hide most of it: est. up to 0.3-0.6 s per such turn. The author's +35% (TTFT 4.97 -> 3.67 s) was a bundle with
  `PARK_REWIND_MAX`, `REFILL_ASYNC` and `ARENA_REUSE` on 2x RTX 3060, PCIe 3.0 [ext.]. `REFILL_ASYNC`: ~143 ms on the
  same box [ext.]. `STRATA_PREFILL_ROUTED_STREAM` (exists, off) only matters for chunks just above 4,096.
- **First request after idle or restart.** (a) The file tier: experts outside the RAM arena are read from the GGUF in
  place on every pass; 11-11 10:30 read 53.7 GB from storage for a 57K prompt [log]. The arena held 28.26 GiB that day,
  30.50 GiB on 11-09 [log]. (b) The first long prompt after a start: 70K at 1,784 tok/s, 11-09 [log]. (c) Conversations
  do not survive a restart unless the disk tier is on (opt-in; est. ~1-1.5 s to restore a 70K state vs 25-30 s to
  re-read; D is 91-94% full, so the tier needs a cap). (d) Engine start itself is 28-65 s (item 38).
- **Fewer prompt tokens.** ContextPilot dedup (lossy, off) removes 1.6% of the characters of 40 opencode sessions
  [meas. offline, MOD_PLAN]: 1-2% of a cold read, nothing on a warm turn. Not a TTFT lever.
- **Prompt-level speculation.** The tool result cannot be read before the tool returns, so the only exact work to move
  into the tool gap is housekeeping: park the live state during the gap (an "idle park"), finish refills and
  checkpoint copies. Token-dropping prefill (SpecPrefill-style) is lossy and breaks the GDN state; two-stage prompts
  need a second GPU.
- **Avalon's prompt path.** Avalon is not faster at 3K: 2.13 s vs pancake's 1.84 s [meas.]. Its tok/s is flat (1,531 /
  1,583 / 1,617) because its cost is almost all per token: fit 0.13 s + 0.61 ms per token, against pancake's
  0.77 s + 0.33 ms. It ran with 4,096-token ubatches and streams host-resident experts per ubatch
  (`ggml-cuda/moe-cache.cu`: "prefill_streamed", the lending ring) [code, Avalon], so its pass cost is spread over the
  tokens. Nothing to port for TTFT.
- **`--short-read 64`.** Decode windows cost ~9 ms per token at 67K depth (28 tokens 261 ms, 64 tokens ~550 ms) and
  the staged path starts at ~0.6 s, so the crossover is near 64 already [log].

## 5. Risks found in long-prompt handling

1. **Lost conversations.** On starts where the budget came out at 1,280 MiB, no conversation above ~40K tokens could be
   parked; the log has 10 "skip parking ... exceeds available budget" lines (snapshots 1,286-1,857 MiB) [log]. Even at
   4,352 MiB a 100K conversation (~2.2 GiB) leaves room for one more long one, and short side requests are parked too
   (1,938 parks below 4K tokens) and can evict it.
2. **4-chunk prompts** (98,305 tokens and more) have never run with the preset here (3.3).
3. **Synchronous parks on HIP**: est. 0.6-2 s at 100K, inside the next request's TTFT.
4. **Fallback parks**: a request that resumes from its own turn checkpoint still parks the live state it diverged
   from (1,417 ms at 67,711 tokens on 11-11 10:40 [log]); `PARK_REWIND_MAX` (0xluk) is not in our tree.
5. **File tier on cold prompts** (section 4): +10-12 s on a 57K read (53.7 GB from storage), probably also on the first
   70K read after a start.
6. Already fixed or out of reach: the KV-stream miss list is bounded (`cfcde96a3`, kllmagn); `iq_embed_rows` with
   grid.y above 65,535 cannot happen with chunks up to 49,152; the 3x32K fault needs the test above.
7. **The benchmark's prompt numbers include a park** of the previous prompt (~0.2 s at 25K); Avalon's do not.

## 6. Ranked levers

Ranked by expected seconds saved per 10-call workflow (size of the effect x how often it happens).

| # | Lever | Mechanism | Expected TTFT effect, reasoning | Effort | Risk, exactness | How to measure | Depends on (unmeasured) |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | Conversation-cache admission: `--conversation-cache-min-tokens 8192`, a budget floor (`--conversation-cache-mib` >= 2048 or an auto floor), then `STRATA_PARK_TRUNCATE=1` | short side requests no longer take slots; a long conversation that does not fit parks its deepest fitting prefix instead of nothing | -20 to -25 s per lost turn at 54-62K [log], est. -35 to -45 s at 100K; offline replay of 11-09: 158 of 162.8 s saved by the two flags, 39-52 s by PARK_TRUNCATE alone [MOD_PLAN 4.0.1] | XS (preset args), S (truncate is in the code; its unit tests not run, MOD_PLAN F1) | exact; RAM headroom (arena 28-33 GiB of 57 GB): a larger cache budget competes with the expert arena for RAM | `replay_turns.py bench/specs/11-conv-cache.json`; M2 with a side request and a sub-agent between turns | how often real sessions interleave (old history: switches 3.5%, unexplained misses 3.9% of request time [percache]) |
| 2 | Parks off the critical path: `--conversation-cache-park-gain 4096` (exists), a `PARK_REWIND_MAX` port (0xluk `53cecdeee9`, ~27 lines), later an idle park in the tool gap or async park on HIP | no park when the request resumes from its own checkpoint; no re-park of a conversation that barely grew; or do the copy while the GPU is idle | -0.1 to -0.2 s per switch below 32K, -0.3 to -1.4 s at 50-77K [log], est. -0.6 to -2 s at 100K | XS / S / M-L | exact (rewind skip loses only the generated tail the client re-rendered differently) | engine log "parked N tokens in X ms" per turn in M2 | switches and fallbacks per workflow |
| 3 | Checkpoint fallbacks (13% of warm turns) | find where the prompt leaves the live state, then fix that render (server) or turn on `keep_reasoning` for clients that drop reasoning | -0.5 to -1 s per fallback (re-read of the last assistant turn), plus the park of lever 2 | XS (diagnostic), S-M (fix) | exact | M3 divergence line; M4 request bodies | the cause (unknown) |
| 4 | Staged-path floor: `STRATA_PREFILL_CPU_SHARE=auto` / 0.65 / 0.7 arms, then `PF_AHEAD` and `REFILL_ASYNC` ports | balance PCIe and CPU on 65-4,095-token reads; overlap the next layer's copy | est. -0.1 to -0.3 s per warm turn (floor 0.55-0.75 s); more on 2-4K deltas | XS (env) / M (ports, ~290 lines for PF_AHEAD) | the share is already in the preset (rounding class unchanged); ports exact | `STRATA_PREFILL_TIMING=1` on 100 / 300 / 1,000 / 3,000-token deltas at 32K and 64K depth | whether the floor is PCIe-bound (1.2 is arithmetic) |
| 5 | Cold-read file tier | keep the experts a prompt pass needs in RAM or in a warm page cache (arena size from the RAM budget, idle pre-read of the file-tier experts, `os_tune.sh`) | -10 to -12 s on a 57-70K cold read after a start or a cache flush (one clear case: 57K with 53.7 GB from storage; one probable: the first 70K prompt after a start at 1,784 tok/s [log]) | S-M | exact; RAM pressure (oomd history) | "file tier I/O" line per request in M1, cold and warm page cache | how often the page cache is cold on a daily start |
| 6 | Session start and sub-agent first call: disk tier (item 39) for restarts; a real multi-entry prefix store for system + tools (A12) | restore instead of re-reading a 10-40K shared prefix | -4 to -15 s per new session that misses; offline model +0.05-0.68% of request time while running (parked entries already cover most) [percache]; restart survival not modeled | S (disk tier exists) / M-L (prefix store) | exact; D drive is 91-94% full | M2 after a restart; percache_sim roots arm | restarts per day |
| 7 | Depth: 49,152 grid + equal tail + refill head (Strata-rdna3 3 commits), `STRATA_PREFILL_EQUAL=1`, `STRATA_PREFILL_LAYER_MAJOR=1` | fewer streamed passes on multi-chunk prompts, no short tail | est. -0.5 to -1.5 s at 64-100K (2-4%), 0 on warm turns | S (port) / XS (env) | rounding changes; the 57K trap without the tail rule | M1 at 33K, 40K, 57K, 64K, 100K | the visible cost of one pass |
| 8 | QSA selector at depth | already on matrix cores; `STRATA_SELECT_TWO_PASS` exists (not checked on HIP) | est. < 3% at 100K | - | - | M1 profile | selector share at 100K |
| 9 | Ring 96 vs 192 on 32K chunks | less loan per chunk | ~1% [ext.] | XS | rounding | M1 | - |
| 10 | Context dedup | fewer tokens | 1-2% of cold reads only, lossy | - | lossy | - | - |
| 11 | BFLA on Flash-Next; Avalon's prompt path; `--short-read` | - | none (sections 3.1, 4) | - | - | - | - |

## 7. What to test first, and what must be measured before any code

**The three levers to test first:**

1. **Cache admission (lever 1).** It is the only one in the "seconds per request" class, it needs no code (flags, then
   an existing env switch), and the failure is in the log: 7 consecutive 21-25 s re-reads of one 54-62K conversation
   after "skip parking (snapshot 1574 MiB exceeds available budget)" on 11-09 16:23-16:30.
2. **Parks and fallbacks together (levers 2 and 3).** Every switch and every fallback pays a synchronous park (0.1-1.4 s
   measured); the fallback case is pure waste (11-11 10:40: 2,816 ms for a 1,265-token read, of which the park was
   1,417 ms). `--conversation-cache-park-gain` exists today; the divergence diagnostic tells whether the fallbacks
   can be removed at the source.
3. **The staged-path floor (lever 4).** It is on every warm turn (0.55-0.75 s of a ~1 s median); the first arms are env
   only (share `auto` / 0.65 / 0.7).

**Measurements that must exist before any code is written** (the lead's harness covers M1 and M2):

- **M1, single prompt at 8K / 32K / 64K / 100K**: per size record `prompt_ms`, the client's time to first byte, the park
  of the previous live state (engine log), the chunk plan (`STRATA_TRACE=1`), the "file tier I/O" and "KV streaming"
  lines, and a `STRATA_PREFILL_TIMING=1` breakdown at 64K and 100K (expert GEMMs, GDN, PLE, QSA attention, select, KV
  stage). Run each size twice and keep both (the first after a start shows the file tier). Start each size from a
  fresh conversation state or report the park apart. The 100K case is also the 4-chunk / #5539 check.
- **M2, a 10-turn agent session**: per turn TTFT (client), `prompt_ms`, prompt tokens, reused, read, the class (live hit,
  turn-checkpoint fallback, parked restore, cold), park and restore ms. Three variants: the session alone; with a short
  side request (a title or a token count) and one sub-agent switch between turns; and with a tight budget
  (`--conversation-cache-mib 1280`) to reproduce the skip. Then the same with lever 1's flags.
- **M3, divergence on fallbacks**: for every request that does not continue the live state although one exists, log
  the first token index where the prompt differs from the live tokens, the live length, and ~16 decoded tokens on each
  side (opt-in; a few lines next to the prefix lookup at `generate.cpp` 17338, or in the server, which keeps recent
  prompts for its incremental tokenizer and sees the generated ids [code]). Without it, lever 3 is guessing.
- **M4, two consecutive request bodies from opencode and from Copilot** (OpenAI endpoint): does the client resend
  `reasoning_content`, the tool calls as sent, a byte-stable system prompt and tool list?
  Tooling written (not yet run on a real client): `tools/pancake/prefix_check.py`, `STRATA_REQUEST_CAPTURE`, `pancake-docs/PREFIX_CHECK.md`.
- **M5, the warm-delta sweep for lever 4**: 100 / 300 / 1,000 / 3,000 new tokens at 32K and 64K depth, with
  `STRATA_PREFILL_TIMING=1`, share 0.6 vs `auto`.

## 8. Unconfirmed points

- Every [code] statement, in particular: park inside the timed prompt (the log agrees: 2,816 ms for a 1,265-token read
  with a 1,417 ms park), periodic checkpoints that never split a chunk, the KV staging cost, the staged path being
  PCIe-bound, and the chunk plan at 64K and 100K.
- The 64K / 100K / 128K TTFT values are extrapolations from 25K and 54-62K.
- The 46 warm turns are one client on two days; the client is not recorded in the turn log.
- Copilot's behaviour (section 2.3) is unknown.
- The causes of the 3.4 s first warm turn after a cold read, of the 4.8 s outlier outside the engine timer, and of the
  13% fallbacks.
- Strata-rdna3's numbers are from their boxes (Q4 files, one and two GPUs); the issue #5539 fault is unconfirmed without
  KVA.

## 9. Reproduce the log numbers

Read-only, CPU only (added with this note; 6 tests in `tools/pancake/test_ttft_audit.py`):

    python tools/pancake/ttft_audit.py --turns pancake-turns.jsonl --engine-log pancake-flashnext.log [--json OUT]

It classifies every request of the turn log (live, fallback, switch, lost, cold; the rules are in its header),
prints the agent-like re-read share, the live-turn TTFT bands and the time outside the engine's prompt timer, and
summarises the engine log's park, skip, restore and file-tier lines. On the 11-11 15:50 logs its "lost" class finds
exactly the 7 cold re-reads of 11-09 (162.7 s, the same total as the replay dry run of MOD_PLAN 4.0.1). It is meant
for the harness's 10-turn session as well (M2): point it at that run's turn log and engine log.
`tools/pancake/conv_cache_sim.py` and `replay_turns.py` model and replay the 11-09 session.
