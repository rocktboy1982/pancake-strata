# Offline tier-0 tests (TEST_PLAN 4.0.1-4.0.5), 2026-11-10

No GPU, no builds. Every script ran under `nice -n 19`, each under 20 s of CPU. Branch `work/offline-tier0`
(from `origin/port/rdna3-decode` a08ca2a1d). Scripts and raw outputs are in this directory.

Data used (read only):
- `/home/dan/clone-of-d/src/Strata/pancake-turns.jsonl` (714 turns, 2026-11-09 16:18 to 2026-11-10 12:54) and
  `/home/dan/clone-of-d/src/Strata/pancake-flashnext.log` (engine log, 34,933 lines).
- Routing trace: `routing.bin` + `expert-profile-pancake.bin` in the session scratchpad
  (`.../scratchpad/tr/`, sha256 of routing.bin `37b87e16...aa9240`; identical to the 2026-11-08 21:02 trace-bench
  run: 219 decode windows, requests read4k / short chat / code edit / long, cuts 11 / 96 / 157 / 219). It is the
  trace behind `bench/results/2026-11-09-time-ceiling` (the numbers below reproduce `runs.txt`). It is not in git.
- GGUF headers of `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS` (daily model; header bytes only).

Summary

| # | Answer | Unlocks / kills |
| --- | --- | --- |
| 4.0.1 | Cause found: not PARK_TRUNCATE-only territory. 5 of 7 cold turns = "skip parking, snapshot 1286-1342 MiB > 1280 MiB budget" (0.5-4.8% over), 1 = 1574 MiB (6 checkpoints), 1 = parked then evicted by the next side request. PARK_TRUNCATE alone saves 39-52 s of the 162.8 s (24-32%); with `--conversation-cache-min-tokens 8192` 131-143 s (80-88%); config only (cache 2048 MiB + min-tokens 8192) 158 s (97%) | F1 keep, but second to two config flags; F3 neutral |
| 4.0.2 | opencode 1.19.5 sends NO temperature for Qwen ids, so Strata applies the model card (thinking: T 1.0, top_p 0.95, top_k 20) = sampled. Plugin sub-agents send 0.1 | F6 stays alive (main agent is sampled) |
| 4.0.3 | Cannot run: no window trace exists anywhere (and the engine never writes the drafts record). Oracle ceiling from the routing trace: short chat 1.0%, long 3.4%, code edit 7.9% of window time | F8 undecided; plausible only on code edit |
| 4.0.4 | k-th-miss admission (already built as `STRATA_ADAPT_KMISS`): hit rate edit 76.7% -> 86.5%, short 85.9% -> 89.1%, whole trace 81.2% -> 86.0%. Prompt-fed counts: bound +2.9 points (own-counts oracle 84.03 vs 81.17), but seeds are inert under KMISS | F10 low value; A5 admission confirmed on edit |
| 4.0.5 | CPU expert bytes now ~65-105 MB per token; IQ1_M shadow saves 18.2% (gate/up) or 25.3% (all), IQ2_XXS only 9.8% / 12.5% | F4 passes the 15% gate only with IQ1_M; blocked by RAM (+17.6 / +26.4 GB) |

---

## 4.0.1 Why were 7 agent turns at 54-61K tokens re-read from token 0?

Method. `big.py`-style scan of `pancake-turns.jsonl` for turns with `reused = 0` and a long prompt: 7 consecutive
cold turns of conversation B (prompt 54,313 / 56,264 / 57,763 / 58,710 / 59,395 / 60,233 / 61,565 tokens),
`prompt_ms` 21.2 + 22.5 + 22.9 + 24.6 + 23.3 + 25.0 + 23.3 = **162.8 s** (the 162 s of the plan). The matching
engine run is log lines 16,769-17,630 (engine started 2026-11-09 16:13:07, flags: `--prompt-cache` default 6,
`--conversation-cache-mib` not set, `--conversation-cache-min-tokens` not set; log line 80: "conversation cache
sized from the free RAM: 1280 MiB for up to 4 parked conversations"). The log has no timestamps per line, so
turn and log events were matched by the order of the `prompt N tokens = ... reused` lines (same token counts).

Cause per cold turn (log line numbers within the run):

| cold turn | what the log shows just before it |
| --- | --- |
| 54,313 (21.2 s) | `skip parking (snapshot 1574 MiB exceeds available budget)`: the 53.9K conversation carried 6 checkpoints; then a 69-token request took the live session |
| 56,264 (22.5 s) | parked fine (55,424 tokens, 1257 MiB, evictions=3) but the next side request's park (226 MiB, evictions=4) pushed the total over 1280 MiB and evicted it, then 9 more side parks |
| 57,763 / 58,710 / 59,395 / 60,233 / 61,565 (22.9 + 24.6 + 23.3 + 25.0 + 23.3 = 119.1 s) | each preceded by `skip parking (snapshot 1286 / 1303 / 1314 / 1321 / 1342 MiB exceeds available budget)` with a budget of 1280 MiB: 0.5% to 4.8% over |

Between the agent turns the same server received other requests: 55-75 token decodes (max_new 400-1000), 3058- and
4084-4093-token prefill probes with max_new 1: these look like the benchmark harness (abtest-style), not opencode. The first 25 agent turns of the
same conversation (no interleaving) were warm (0.3-1.5 s prompt time) and so were the last two (62,572 and 62,943
tokens, reused 62,027 and 62,910): the cold turns appear only when another client switched the live session away.
All 10 "skip parking (snapshot" lines of the whole 35K-line log are in this run.

Snapshot size model (fits the log's park/skip lines within 1 MiB): `size = 111 MiB + 14.9 KiB x tokens + 113 MiB x
checkpoints` (checkpoints: every 16384 tokens from where the request resumed, plus tail ones; a cold 54-60K read
leaves 3).

Replay (`park_replay.py`): the real request sequence of that run through a model of the cache (budget, 4 slots,
oldest-first eviction, incoming copy taken before the outgoing one is parked, as in generate.cpp ~16,750), prefill
cost 0.41 ms/token (log: 0.39-0.42), restore 0.9 s per 1.2 GiB. The baseline reproduces the log (8 cold reads of
conversation B including its first; 7 repeated ones). Seconds of conversation-B prompt time saved vs baseline:

| scenario | cold re-reads | saved |
| --- | --- | --- |
| S0 baseline (1280 MiB, 4 slots) | 8 | 0 |
| S1 `STRATA_PARK_TRUNCATE` only (deepest checkpoint prefix that fits) | 5-6 | 38.6-51.5 s (24-32% of 162.8 s) |
| S2 `--conversation-cache-min-tokens 8192` only | 7 | 21.8 s |
| S3 PARK_TRUNCATE + min-tokens 8192 | 1 | 131.3-142.8 s (80-88%) |
| S4 `--conversation-cache-mib 2048` only | 5 | 67.6 s |
| S5 cache 2048 MiB + min-tokens 8192 (no code) | 1 | 158.1 s (97%) |
| S6 cache 1600 MiB + min-tokens 8192 | 3 | 112.3 s |

The two S1 numbers bracket the unknown checkpoint positions (periodic only vs an extra tail checkpoint 2K tokens
before the end). The remaining cold read in S3/S5 is the first turn (34K, unavoidable). Side requests lose their own
warm hits under min-tokens (a 69-token request costs 0.7 s instead of 0.2 s); not counted, negligible.

Verdict. Yes, "park the prefix that fits" would have rescued 3 of the 7 turns (about 40-50 s), but it does not stop
the real mechanism: with 4 slots and 1280 MiB one 1.2-1.3 GiB agent snapshot leaves no room, and every side request
that parks evicts it. Two existing flags remove most of the loss with no code: `--conversation-cache-min-tokens 8192`
(side requests are not parked) and a larger `--conversation-cache-mib` (2048 if RAM allows; the auto value 1280 came
from the RAM free at start, with the 33 GiB arena loaded). PARK_TRUNCATE is still worth porting as insurance for
prompts whose snapshot exceeds any budget (>= ~80K tokens at 2 GiB), but it is no longer the first fix.
- F1: keep, demote behind the two flags; run S5 as a real A/B only if agent + bench traffic share a server again.
- F3 (extra checkpoint cuts): no effect measurable offline (S1 vs S1b move in opposite directions); leave to its own test.
- Not tested: the disk tier (`conversation_disk_tier`), which can take evicted images; it was not enabled in this run.

## 4.0.2 Does opencode send sampled requests?

Method. (a) Read `~/.config/opencode/opencode.jsonc`: provider `strata` (`@ai-sdk/openai-compatible`, baseURL
`http://127.0.0.1:11437/v1`), model `qwen3.8-flash-next-iq2_xs` has only `reasoningEffort` options; no temperature
anywhere for it. The only `temperature` in the config (0.1) is the `strata-coder` sub-agent, which uses
`qwen3.6-35b-a3b`. (b) Read opencode 1.19.5's own default-temperature function (strings of the binary
`/snap/opencode/current/bin/opencode`): it returns a value only for north-mini-code, gemini, glm-4.6/4.7, minimax-m2,
kimi-k2; **no branch for qwen**, so nothing is sent. (c) Measured: a stub OpenAI server on 127.0.0.1:18437 (killed
afterwards) and `OPENCODE_CONFIG` pointing the strata provider at it; `opencode run -m strata/qwen3.8-flash-next-iq2_xs
"say hi"` (default agent Sisyphus, 3 runs) sent exactly: `model, max_tokens 32000, tool_choice auto, stream true,
stream_options, 2 messages, 56 tools` and **no temperature, top_p, top_k or reasoning field**. (d) The oh-my-openagent
plugin sets `temperature: 0.1` for its Oracle, Librarian, Explore, Multimodal-Looker, Atlas, Momus, Hephaestus agents
(0.3 for Metis); those are sub-agents.

Server side (`serve/server.py` `_run`, `MODEL_CARD_SAMPLING`): a request without sampling fields gets the model
card of its family, and the pancake presets set no `sampling` block and no `arch` (so `qwen4exp`): thinking turns
(the agent turns in the log carry 1,000+ reasoning tokens) are served at **T 1.0, top_p 0.95, top_k 20**.

Verdict. The main agent's turns are sampled at T = 1.0 (>= 0.7), not greedy; sub-agents of the plugin run at T 0.1,
the cloud-style local coder at 0.1. The turn log does not record sampling parameters, and the 2026-11-09 agent turns
show max_new 32768 against the 32000 this opencode version sends, so that day's client may have been another build;
the server-side default applies to either. Acceptance on those sampled agent turns is already 0.694 per offered draft
row (34 turns >= 20K prompt) against 0.757 on the bench turns (682 turns), so sampled traffic does not collapse
drafting.
- F6 (sampled-aware speculation): not killed. The arms of 4.1.2 are worth running; the target workload is the main
  agent at T 1.0.
- Side effect to know: the 3 stub runs added 3 "say hi" sessions to opencode's local session store.

## 4.0.3 Headroom of a cumulative-probability draft stop

Method. `tools/pancake/trace/tail_replay.py` (note: under `trace/`, the plan says `tools/pancake/`) needs a window
trace (`*.stwt`) with drafts records. `find /` for `*.stwt`, `*wtrace*`, `*window*trace*`: none exists, and
`STRATA_WINDOW_TRACE` has never been set on a recorded run. Worse, the drafts record needs a `window_trace().drafts()`
call from the draft area; `grep` finds no caller (include/strata/core/window_trace.hpp defines it, MOD_PLAN item 74
open). So the replay cannot run, now or after only setting the variable. A window trace WITHOUT drafts records
would still carry rows and accepted counts per window, enough for an exact oracle ceiling (needs a short GPU run).

Fallback (`tail_ceiling.py`, reuses `tools/pancake/tail_rows.py` logic on the routing trace): an oracle that drops
exactly the rejected rows saves the CPU-missed experts only those rows needed. Rejected rows per window = rows -
tokens per window (accepted + 1) of each request. No real stop rule beats the oracle (and a real one also drops
accepted rows):

| request | windows | tokens/window | CPU-missed experts saved by the oracle | CPU share of window (time_ceiling calibration) | oracle gain of window time |
| --- | --- | --- | --- | --- | --- |
| short chat | 85 | 3.8 | 4.9% | 21% (5.2 of 24.8 ms) | 1.0% |
| long | 62 | 2.7 | 16.2% | 21% | 3.4% |
| code edit | 61 | 9.1 | 17.0% | 46.5% (17.4 of 37.4 ms) | 7.9% |

Verdict. Inconclusive by the plan's own rule; the bound says F8 cannot reach 2% on short chat even with perfect
foresight, and could reach it only on code-edit-like traffic (a predictor capturing 30-60% of the oracle gives
2.4-4.7% there). Whether the drafter's confidences separate accepted from rejected rows is exactly what is missing.
- F8: do not port yet. Cheapest next step (deferred, needs the GPU): one short run per workload with
  `STRATA_WINDOW_TRACE` set (accepted counts, exact oracle), and a 5-line `drafts()` call to record confidences.
- Item 5 / `tail_replay.py`: blocked on the same two things.

## 4.0.4 Frequency-gated admission and prompt-fed tier counts

Method. `tools/pancake/time_ceiling.py` (policies tier / LRU / Belady / online k-th miss) and
`tools/pancake/replay_seeding.py` (seeding the tier's usage counts at a request's start, including the oracle of the
request's own counts) on the routing trace above; 10,909 slots, 32 swaps per window every window, decay 0.7. The
numbers reproduce `2026-11-09-time-ceiling/runs.txt` and the header of `replay_seeding.py`
(plain tier 81.17%). Raw: `time_ceiling_{all,short,edit}.txt`, `replay_seeding.txt`.

Frequency-gated admission (admit an expert after its k-th miss in the last W windows, evict LRU), vs the engine's tier
(share of routed entries served from VRAM; window gain at DMA contention alpha 0 / 0.25 / 0.5):

| trace | tier hit rate | 2nd miss in 4 windows | 2nd miss in 8 windows | window gain of 2nd-in-4 |
| --- | --- | --- | --- | --- |
| all 219 windows | 81.2% | 86.0% | 86.5% | +4.8% / +0.1% / -4.8% |
| short chat (85 w) | 85.9% | n/a | 89.1% | 2nd-in-8: +5.5% / +1.8% / -1.5% |
| code edit (61 w) | 76.7% | 86.5% | 86.9% | +22.0% / +11.9% / +3.7% |

Prompt-fed tier counts (F10). The trace holds decode windows only; there is no prefill routing trace, so F10 itself
cannot be replayed. Its bound is the seeding replay: seeding the tier with the request's own counts gives 84.03% vs
81.17% plain (+2.9 points; per request read4k +5.2, short +3.0, edit +2.5, long +3.1); a domain profile learned from
other requests gives 82.9% (+1.7). Prompt routing is a noisier version of the request's own counts, so expect between
those. In window time: +2.9 points of hit rate removes ~15% of the CPU-missed experts = about 0.8 ms of a 24.8 ms window
(3%) at best, and only until the tier adapts. MOD_PLAN item 84 (still open): seeds only raise usage, so they are inert
under `STRATA_ADAPT_KMISS`, which needs real misses.

Verdict. k-th-miss admission is the big lever on edit-like traffic (+10 points of hit rate, +12% window at
alpha 0.25) and costs nothing on short chat at alpha <= 0.25; it is already implemented (`STRATA_ADAPT_KMISS`, plan A5,
test 2.2) and this replay says to run it. F10 is worth at most 1-3% and conflicts with KMISS unless seeded pairs get
virtual miss bits.
- F10: low priority, port only together with the item-84 fix; cannot be validated offline without a prefill routing dump.
- A5 / `STRATA_ADAPT_KMISS` (needs `--adapt-async`): confirmed by replay on edit; test 2.2 goes ahead.

## 4.0.5 CPU expert bytes a sub-2-bit shadow pack would save per token

Method. `gguf_table.py` reads only the GGUF header (metadata + tensor-info table) of both shards. Daily model:
`qwen4exp`, 48 blocks, 512 experts, 10 used, expert FFN 640, hidden 2560. Per expert per layer: gate and up are
2560 x 640 weights each, down 640 x 2560. Types per layer: gate/up IQ2_S in 34 layers (524,800 B per expert tensor),
IQ2_XXS in 11 (422,400 B), IQ1_M in 3 (358,400 B); down is `Q2_0` (type 42, 2.25 bpw, 460,800 B) in all 48. Mean per
expert (gate+up+down) 1,443 KB. Expert tensors total 33.0 GiB of the 39.2 GB shard 1 (shard 2 is one 26.8 GiB IQ4_NL
tensor).
Miss rate: `shadow_bytes.py` replays the routing trace through the tier and counts, per window, the distinct
(layer, expert) the CPU computes, priced with each layer's actual bytes; shadow variants use min(current, shadow)
per tensor.

| request | CPU MB per decoded token now | gate/up IQ2_XXS | all IQ2_XXS | gate/up IQ1_M | all IQ1_M |
| --- | --- | --- | --- | --- | --- |
| read4k | 71.9 | -9.9% | -12.6% | -18.3% | -25.4% |
| short chat | 92.5 | -9.7% | -12.4% | -18.1% | -25.2% |
| code edit | 120.5 | -9.8% | -12.5% | -18.2% | -25.3% |
| long | 89.7 | -10.0% | -12.7% | -18.4% | -25.5% |
| all | 105.2 | -9.8% | -12.5% | -18.2% | -25.3% |

Cross-check from the logs (hit rate of the decode expert cache over the turn log, token-weighted): agent turns
>= 20K prompt 0.906 (34 turns), others 0.891 (700 turns); quartiles 0.862 / 0.912 / 0.943. That is a miss rate of
9-11%, i.e. at most 480 routed entries x 0.094-0.109 x 1.443 MB = 65-76 MB per token before the within-window
de-duplication the replay applies; the replay's 105 MB comes from its lower hit rate (81% vs the live tier's 89-91%).
Use 65-105 MB per token. The saving fractions barely move with the miss rate (they depend only on the type mix).

Time translation (bytes are not time): CPU experts are 21% (short chat) to 46.5% (code edit) of the window in the
time_ceiling calibration, so gate/up IQ1_M (-18.2% bytes) is worth at most 3.8-8.5% of decode time and all-IQ1_M
(-25.3%) 5.3-11.8%, if CPU time is proportional to bytes (the calibration implies ~66 GB/s at 23 us per 1.5 MB expert,
above the 51 GB/s of dual-channel DDR4-3200, so part of the CPU time is not bandwidth-bound and the real figure is
lower). The plan's estimate (edit +7-10%, chat +3-5%) is consistent with the upper end.

Verdict. By the plan's rule (requant only if >= 15% of CPU expert bytes) F4 passes with IQ1_M shadows (gate/up 18.2%,
all 25.3%) and fails with IQ2_XXS shadows (9.8% / 12.5%). Two blockers the byte count exposes: (1) RAM: reads happen
from RAM, and the exact experts must stay for VRAM promotions, so a shadow adds 17.6 GB (gate/up IQ1_M, 48 x 2 x 512
x 358,400 B) or 26.4 GB (all IQ1_M) to the 30.5 GiB of experts already resident, on a 57 GB box; (2) down is already
Q2_0 at 2.25 bpw and 3 layers are already IQ1_M, so the quality headroom is thin (lossy, needs the KL gate).
- F4: conditional go for a gate/up IQ1_M shadow only if the RAM question has an answer (shadow replaces the exact copy
  of the layers it covers and promotions read from disk, or fewer resident experts); otherwise defer. Do not
  requant to IQ2_XXS.

## Deferred (needs the GPU or code, not run)

- A real window trace with accepted counts (`STRATA_WINDOW_TRACE`) for 4.0.3, and the `drafts()` call for confidences.
- A prefill routing dump for a direct F10 replay.
- S5 as a live A/B on the server (cache 2048 MiB + min-tokens 8192 with agent + bench traffic).
- Per-line timestamps in the engine log would make 4.0.1 an exact join instead of an order match.
