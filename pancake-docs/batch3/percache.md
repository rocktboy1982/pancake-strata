# Per-turn cache (percache): A11, A12, A13, A14, items 16, 31, 39, 71, 90 (2026-11-11, branch dev/p16/percache)

Scope: MOD_PLAN 16.2, prompt re-reads in agent turns. Everything here is exact (same greedy text) and off by default.
No GPU and no engine were used. What ran: compile-only checks, CPU unit tests, Python tests, the replay dry run, and a new
offline model over the real opencode history (`tools/pancake/percache_sim.py`).

Label note: the task text calls the tool-boundary checkpoints "A16". In the plan table that is A11 (item 16). A16 there is
the byte-stable prefix (`env_to_user`, done). This file uses the plan table's labels.

## Result in one table

Share of summed request time (38,911 s over 2,036 recorded Qwen requests, 25 sessions, 2026-10-15 to 11-10).
The history re-reads 12.2% of that time (12.7 M tokens at 0.375 ms per token). Causes of the re-read: no event in the store
3.9%, a switch to another session 3.5%, after a new user message 1.6%, compaction or prune 2.6%, first request of a session 0.5%.

| Item | What the source says | What the code does | Modeled saving here | Verdict |
| --- | --- | --- | --- | --- |
| A14 restore_check | UNSW 2609.15030: paired greedy continuations vs a recompute control with matched chunking; found a complete-hit off-by-one | same method (`STRATA_CKPT_REREAD` re-reads in the saving read's chunks); was able to pass without ever using the cache | 0 (it is the gate) | fixed, GPU run queued |
| A11 tool-boundary marks | FreeToken 2608.16157 section 3.1: checkpoints at thinking, tool-call and turn boundaries, LRU slots; no isolated number | `qwen_tool_marks` + `tool_checkpoint_marks`; correct and tested | about 0 in steady agent growth; negative for edits far back (see below); upper bound 0.5% | keep off |
| A11 FLOP eviction | Marconi 2411.19379: S = recency + alpha x flop efficiency, alpha tuned by replay; 19-220% hit rate over LRU | `flops_victim`; alpha is a fixed env value | micro model: 17-30% fewer tokens re-read for edits far back; 0.5-0.8% of request time if all partial re-reads were such edits | try `--prompt-cache-spread` first (exists, same gain) |
| A12 / 90 / 71 pin | RadixAttention 2312.07104 keeps a multi-entry prefix tree | one pin at a time; a parked entry holding a pin is never evicted | as coded: **-1.3%** (worse than no pin); a real multi-entry prefix store: +0.68% (base), +0.05% (with the 4.0.1 flags) | do not enable `STRATA_AUTO_PIN` for Qwen |
| A13 / 39 disk tier | Bidaw FAST'26: eviction guided by the model's last answer; human chat, transformer KV, 1.5 GB/s SATA RAID | disk keeps what RAM evicted; tool-call turns stay in RAM | +0.11% beyond the 4.0.1 flags (43 s); restart survival not measurable from the store | exactness test queued, little to gain here |
| 31 preserve_thinking | - | the pack's template already keeps earlier thinking by default | 0, nothing to build | closed |
| the 4.0.1 flags (not mine, the model's base) | - | `--conversation-cache-mib 2048 --conversation-cache-min-tokens 8192` | **+1.89%** (734 s) on this history; ceiling of any parked store 2.0% | the real gain of this area |

No item reaches the 2% bar for an engine switch except the already known flags. Nothing was added to the engine.

## Source checks

**Marconi (Pan et al., MLSys 2025, 2411.19379; read the PDF).**
- Admission: it keeps at most two SSM states per sequence: the last decoded token and branch-off points found by a speculative
  radix insert. It does NOT checkpoint tool boundaries. That idea is FreeToken's.
- Eviction: `S(n) = recency(n) + alpha * flop_efficiency(n)`, both normalised to (0,1) over the nodes; flop efficiency is the
  FLOPs saved per byte of the node's state; nodes with at most one child can be evicted, branch nodes are protected; alpha is
  fitted by a grid search that replays 5-15x the requests seen before the first eviction.
- Measured on four A100-40GB with a 7B hybrid ({4 attention, 24 SSM, 28 MLP} layers), LMSys, ShareGPT, SWE-Bench traces.
  "71.1% (617 ms) lower P95 TTFT" and "34.4x token hit rate" compare admission plus eviction with a fine-grained vLLM
  baseline. The eviction rule alone gives 19.0-219.7% more hit rate than LRU. The plan row ("Marconi TTFT up to -71% on SWE-bench") mixes these.
- Match with the code: `flops_victim` follows eq. 2 with the node bytes constant (the arena holds the K/V, a checkpoint is only
  the recurrent state), pinned root and newest never leave. Difference: alpha is fixed, not fitted. Difference: the FLOP term
  assumes attention cost grows with position (`attn_tokens` 32768). The 11-09 cold reads show linear cost (ms = 1583 + 0.375 x tokens, R2 0.99; per-token cost falls from 0.67 at 3K to 0.39 at 60K), so on this card the
  term is not visible. The micro model gives the same ranking with a linear cost, so this does not change the verdict.
- Not built, and the reason the Marconi gain does not transfer: the engine already keeps a checkpoint at every turn boundary
  and every 16,384 tokens in a 6-slot chain. That is the fine-grained thrash Marconi's admission avoids. Its effect shows in the
  recorded data: re-reads after a deep change fall back to the root or to the last few turn checkpoints.

**FreeToken (2608.16157, section 3.1 on the HTML version).** Checkpoints at special-token boundaries (thinking, tool calls and
outputs, turns), restore from the deepest one that survives an edit, LRU slots. Matches `STRATA_CACHE_TOOL_BOUNDARY` and `=2`.
The paper gives no hit rate or time for the state cache alone (only end-to-end agent TTFT on an RTX 5090), so the plan's
gain for A11 has no source number.

**UNSW hybrid-state recovery (2609.15030, abstract).** vLLM + LMCache, GLM-5.3-Flash, TP4. The bug: a complete-hit restore
covered the whole prompt but the scheduler credited one token less. 34/36 generations matched before the fix, 36/36 after.
The test: paired 256-token continuations vs a recompute control with matched checkpoint scheduling. Our engine guards the
same case (prefix only if <= n-1 tokens, generate.cpp 16161-16166). `restore_check.py` had no complete-hit turn: added.

**Bidaw (FAST'26, read the PDF).** Predicts when a user comes back from the length of the previous answer (correlation
0.94-0.98) to choose what leaves host memory for SSD. Human chat, several users, OPT/Qwen-7B/14B transformers, one A800,
200 GB RAM, 4 SATA SSD in RAID-5 at 1.5 GB/s. Latency -3.58x, throughput 1.83x at high load. "Tool call stays, final answer
goes first" is an adaptation (an agent returns at once after a tool call, a person after a pause), not what Bidaw measured.

**RadixAttention (SGLang).** Multi-entry prefix tree. The code has one pin at a time, so it is not that.

## Correctness of a restore (why it can be exact)

- A checkpoint holds the recurrent (GDN) state, the PLE history, the indexer tails and the drafter snapshot. The K/V cells
  live in one positional arena, so the engine keeps only checkpoints whose tokens are a prefix of the current prompt (a chain).
  Restoring one at position p and reading on gives the same state as reading from 0 as long as the chunk geometry is the same.
- A mark splits the read of a prompt at p. That changes chunk geometry against a run without marks, so bit equality with
  "no marks" is not claimed (the docs say so). The check that matters compares restore with `STRATA_CKPT_REREAD`, which
  re-reads [0, resume) in the same chunks.
- Disk tier: the key holds the model file, the session identity and the lossy env switches, so a lossy state is not resumed
  by an exact run. `STRATA_EXP_MODE` / `STRATA_EXP9_DOWN` (F14, kernel wave) are not in the key. If they change values they must be.
- The replay boundary of a complete hit is n-1: the last prompt token is always read by the verify window.

## What I changed

- `tools/pancake/restore_check.py`: a new last turn `repeat` (the previous prompt again: a complete hit); the run is
  refused (exit 2) when a turn that should hit the cache reports 0 cached tokens, because a cache that does nothing matches a
  re-read; `repeat` must equal `tool-retry` inside every arm; with `--tool-boundary` the engine log must show a
  tool-boundary checkpoint (`STRATA_TRACE=1` is set for it); 96 tokens compared (was 48; the UNSW paper compared 64-256).
- `tools/pancake/test_restore_check.py` (13 tests). One test renders the script with the pack's real chat template and
  tokenizer: the call boundary is token 1245, both retry prompts share their first 1250 tokens, so the retry can resume at
  the mark. Skipped without a pack tokenizer folder.
- `tools/pancake/percache_sim.py` and `test_percache_sim.py` (7 tests + self-test); results in
  `bench/results/2026-11-11-percache/` (text and JSON, numbers only).
- Cherry-picked `1a06bb1c9` from dev/p16/compilegate (generate.cpp lacked `#include "strata/core/calibrate_ratio.hpp"`, 31 compile errors). Same commit, it merges clean.

## The model (percache_sim.py)

Input is opencode's own store, read only. Each assistant message of the served model has the server's `tokens`
(input, `cache.read` = reused). So the history says what each request read again. No text is used.
- Causes: a re-read is the input beyond the growth since the session's previous request; each is labelled by what came before it.
- Park model: all requests replayed in time order through the conversation cache rules (one live state, parked snapshots under
  a MiB budget and slot count, oldest out first, snapshot 111 MiB + 14.9 KiB/token + 113 MiB per checkpoint as in
  park_replay.py). A family's shared prefix is estimated from repeated `reused` values (Sisyphus-Junior 39,534, explore
  18,001, build about 10.7K). Check against the record: the base arm agrees with the recorded hit or miss on 1811 of 1937 continuing requests (93.5%).
- Chain micro model: one 120K conversation in 6 checkpoint slots, LRU vs Marconi vs spread, with and without tool marks.

| Arm | tokens read (continuing requests) | saved s | % of request time |
| --- | ---: | ---: | ---: |
| base (1280 MiB, 4 slots) | 3,787,707 | 0 | 0 |
| flags (2048 MiB, min 8192) | 1,749,377 | 734 | 1.89 |
| flags + disk tier | 1,617,424 | 777 | 2.00 |
| prefix store of every family (A12 as the plan means it) | 3,080,064 | 264 | 0.68 |
| prefix store + flags | 1,692,279 | 756 | 1.94 |
| `STRATA_AUTO_PIN` as coded | 5,214,099 | -513 | -1.32 |
| pin, parked pin kept only for the family in use | 4,205,608 | -151 | -0.39 |
| unlimited store (ceiling) | 1,617,424 | 781 | 2.01 |

Chain micro model, tokens read again per edit (120K conversation, 6 slots):

| Policy | edit at a random place | edit at a tool boundary | edit in the last 8K |
| --- | ---: | ---: | ---: |
| LRU (now) | 105,524 | 102,409 | 30,662 |
| tool marks on LRU, steady growth | same | same | same |
| FLOP alpha 1 | 87,727 | 87,282 | 7,300 |
| FLOP alpha 2 | 73,253 | 70,322 | 9,663 |
| spread (`--prompt-cache-spread`) | 78,506 | 76,217 | 7,780 |
| after one cold read: LRU / LRU + marks | 72,780 / 88,511 | 70,588 / 87,840 | 11,196 / 7,911 |

Reading: in steady growth every tool boundary lies within 1,024 tokens of the previous turn checkpoint, so the marks never
qualify. After a long read they help edits near the end and take two slots from the periodic checkpoints, which makes the
other edits dearer. opencode has no retried tool step, and it prunes old tool outputs, which is far from the last boundaries.

## What the model cannot say

- The text is not in the store: the 3.9% "no event" re-reads and the 1.6% after a new user message have no known cause. The pack's
  template keeps thinking by default (checked by rendering: default equals `preserve_thinking=true`; only `false`
  drops it). Reasoning is present in 2033 of 2036 steps, and the quiet in-session steps (1806: no switch, user message,
  compaction or prune) reuse 96.5% of the previous state by tokens, most of them all of it; the 169 "no event" steps are the
  misses. So a dropped thinking block is not the cause. The guess is per-turn text that the agent harness adds to earlier
  messages; one logged request body would settle it.
- The history mixes engine settings of 2026-10-15 to 18. The conversation cache did less then.
- The micro model assumes edits spread over the history. It shows the direction, not the size.
- Seconds use one marginal rate (0.375 ms per token, fitted on cold reads from 0).

## GPU checks to queue (nothing run here)

1. A14, first, alone on the GPU: `python tools/pancake/restore_check.py --tool-boundary --disk --out bench/results/<day>-restore-check.json`
   (preset `flashnext-iq2_xs-hip`, `--adapt-swaps 0`, greedy, 96 tokens). Pass is exit 0: every turn MATCH for arms restore and
   disk vs reread, `repeat` equals `tool-retry`, at least 2 disk writes and read-backs, at least 1 tool-boundary checkpoint line.
   Exit 2 means the cache was not exercised (a vacuous pass); exit 1 is a real difference. Run it once without
   `--tool-boundary --disk` as well, for the default path. The MTP drafter is in the preset, so `STRATA_SNAPSHOT_VERIFY` covers its snapshot.
2. Pin hazard, one confirm (the model says it loses): 6 sessions, same short system text, 12K private tokens each, order
   A B C D E F A B; arms base and `STRATA_AUTO_PIN=1`. Expect with the pin: "skip parking" lines for E and F and 0 cached
   tokens on the second A and B; without it parked hits. Run it before anyone sets the pin for Qwen.
3. 4.0.1 flags on the replay (already queued by the plan): `replay_turns.py bench/specs/11-conv-cache.json --out <dir>`
   then add `--prompt-cache-spread` as an arm. The recorded session s0 has no deep edits, so it cannot rank spread against LRU; use an
   opencode session with edits or leave it.
4. Disk tier timing only if the owner wants restart survival: resume of a 70K state from disk vs `STRATA_CKPT_REREAD`.

## Risks found

- `STRATA_AUTO_PIN=1` on Qwen: `make_room` never evicts a parked entry that holds a pin, so with several long sessions the
  first ones stay forever and new ones are not parked. Fix needs a separate prefix store (the plan's real A12), not
  immovable whole conversations. Not edited: the lines are in `conversation_cache.hpp` next to the policy agent's park-TTL code.
- A cheap variant (protect a pin only for the family in use) still loses 0.39% in the model, so it is not a fix.
- `STRATA_CACHE_TOOL_BOUNDARY` can displace periodic checkpoints after a long read.
- The disk key lacks `STRATA_EXP_MODE` / `STRATA_EXP9_DOWN`.

## Tests run

- HIP configure (gfx1100, compile only), built: conversation_disk_tier_test (97 checks), message_boundary_test (31),
  conv_cache_test, conversation_cache_test, conversation_file_test, conversation_memory_test, conversation_split_failure_test,
  session_save_reclaim_test: 8 of 8 passed with the GPUs hidden. conversation_validation_test and conversation_snapshot_test need the
  engine library and a GPU: not built.
- generate.cpp: `-fsyntax-only` with the HIP flags, 0 errors after the include fix (4 old unused-variable warnings, none in these paths).
- Python: `serve/test_tool_boundary_marks.py`, `serve/test_keep_reasoning.py`, `test_dsv41.py`, `test_ds4_restore_points.py`,
  `test_frontend.py` (120 passed, 1 skipped); `tools/pancake/test_restore_check.py` (13), `test_percache_sim.py` (7), `test_replay_turns.py`.
- `replay_turns.py bench/specs/11-conv-cache.json --session s0 --dry-run` on the recorded log: 75 requests, prompt 303 s,
  turn time 701 s, 7 cold re-reads = 162.7 s (23% of turn time), 8.7 min with every agent turn warm vs 11.7 min. Sessions s1 to s6 are
  benchmark traffic (no request above 12K tokens that a replay can use).

## Next

- Run GPU checks 1 and 2.
- If the owner wants A12, build a prefix store beside the parked set (about 0.8 GiB for the 39.5K-token family) and
  re-run `percache_sim.py --arms roots`; the model puts it at 0.05% over the flags.
- Log one opencode request body to name the 3.9% / 1.6% re-reads.
