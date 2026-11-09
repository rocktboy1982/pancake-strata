# Pancake: server, agent and operations pain points mapped to the research series

Scope: the serve/ layer, the agent loop with opencode, and operations of pancake-strata (repo /home/dan/clone-of-d/src/Strata, branch port/rdna3-decode, HEAD c87ef4d27). This was a read-only pass: nothing was built, run, tested or modified. The sources are local files. Links are file paths. "opencode DB" means aggregate queries run read-only on /home/dan/.local/share/opencode/opencode.db (tables message and part, assistant messages with providerID "strata" and a qwen3.8 model). No user content was read or quoted, only counts, sizes, tool names and error classes.

Important caveat for every opencode number below: the 2,063 local Flash-Next turns date from 2026-10-15 to 2026-11-02 (1,552 of them on 10-16, 418 on 10-18, only 2 after 11-02). They were served on port 11437 by the older Strata checkout (/home/dan/Strata via strata.service) or by Avalon, not by pancake. No opencode session has run on pancake yet. The figures describe the workload shape and the cache behaviour that pancake inherits from Strata's serve layer. They are not pancake measurements.

## Q1. What drives TTFT and per-turn latency in agent turns (cache hits and misses, compaction, thinking in history, env_to_user, tool-schema size)?

### Takeaway
Most turns hit the conversation cache. About 90.8% have cache_read > 0, and the median uncached input is 226 tokens. The time is lost in a tail of re-reads. Three causes account for roughly 13M re-read tokens, about 2 h of prefill out of about 11.4 h of local turn time:
- same-session misses that go beyond the prompt's growth (207 turns, 6.05M tokens),
- switches between sub-agents and sessions (about 103 large misses, about 5.5M tokens),
- compaction (38 compaction requests that re-read 1.51M tokens, plus 45 turns where the prompt shrank, with 1.87M uncached tokens).

None of the per-turn cache levers in pancake is enabled in any preset (message-boundary checkpoint, prompt-cache tail, conversation working set, env_to_user). Items 11, 16, 31 and 35 are all still test arms.

### Cited Findings
- Workload shape (opencode DB): 2,063 local Flash-Next assistant turns in 25 sessions. Median 5 turns per session, maximum 1,017.
  - Prompt (input + cache_read) p10 33.9K, p50 64.5K, p90 93.0K, p99 107K, max 121.9K tokens. 42 turns were above 100K.
  - Uncached input p50 226, p90 21.8K, p99 81.8K.
  - Turn duration p50 6.2 s, p90 50.4 s, p99 224 s, mean 19.9 s (about 11.4 h in total).
  - finish: tool-calls 1,906, stop 112, length 15, none 30.
  - Total uncached input 15.23M tokens against 118.7M tokens read from cache. — opencode DB
- Same-session re-reads: on consecutive turns of the same session with no other session in between, the uncached input matched the prompt's growth (a hit) in 1,690 cases. It exceeded the growth by more than 2K tokens in 207 cases: median 16.9K, p90 77.2K, max 111.5K, 6.05M tokens in total. The prompt shrank (compaction or pruning) in 45 cases, with 1.87M uncached tokens. — opencode DB
- Sub-agent and session interleaving: a turn after a different session's turn had more than 4,096 uncached tokens 83 times (4.77M tokens, about 57K per event). Another 20 events came after compaction, a model switch or a gap over 10 minutes (about 0.70M tokens). — opencode DB
- Compaction: 38 compaction-agent requests carried 1.51M uncached tokens (about 40K each). opencode's compaction is a separate request with its own system prompt, so it shares no prefix with the agent's conversation. 28 same-session turns right after a compaction had large misses (1.03M tokens). ~/opencode.json has "compaction": {"prune": true}, which removes old tool outputs and so also rewrites the history. — opencode DB; /home/dan/opencode.json
- Prefill rate on the 7900 XT: 1,850-1,920 tok/s at 11.7K with the matching hipBLASLt table (1,345 without it). Per MOD_PLAN, "per-turn cache and thinking length weigh as much as tok/s". — /home/dan/clone-of-d/src/Strata/MOD_PLAN.md (Phase 13)
- Conversation cache (parked states):
  - Sized from free RAM at start: "5376 MiB for up to 4 parked conversations" (other runs: 5888 and 6144 MiB).
  - Snapshot size: 540 tokens = 243.8 MB, 16,081 tokens = 482 MB, 32,072 tokens = 726 MB, 50,876 tokens = 1.13 GB.
  - Park takes 105-462 ms. Restoring a checkpoint takes 33 ms.
  - Evictions reach 33 in bench runs. The log has 748 parks, 461 restores and 34 lines with evictions above 0. — /home/dan/clone-of-d/src/Strata/pancake-flashnext.log
  - Budget rule: 40% of the RAM left above --conversation-cache-min-free-mib after load, at most 8192 MiB, off under 1024. Default 4 slots. --conversation-cache-min-tokens defaults to 0 (park everything). — docs/DETAILS.md:730-733; src/program/generate.cpp:933-939
- Single slot: "the daily preset serves solo (no --batch, no admission queue)". Requests are serialized, and alternation between conversations depends entirely on the parked cache. — MOD_PLAN ContextPilot entry (branch feat/contextpilot, commit 5f03010f3)
- Message-boundary checkpoint (STRATA_CACHE_MESSAGE_BOUNDARY=1): opt-in, only for prefixes of at least 8,192 tokens with a tail of at most 1,024. The author measured the first edit 33.1% faster (17.24 -> 11.53 s), but that was a combined policy and chunk-size comparison on other hardware. The cost is about 0.2 s on cold 8K requests. — docs/MESSAGE_BOUNDARY_CACHE.md
- --prompt-cache-tail: opt-in, one extra checkpoint near the prompt's end. --prompt-cache-every does not cut the prefill pieces. — docs/PROMPT_CACHE_TAIL.md
- The server's slot picker knows about turn checkpoints at the last <|im_start|> and about message-boundary branches (BRANCH_BOUNDARY_MIN = 8192, #2518 fix). — serve/server.py:2731-2800
- No preset enables any of these switches. A grep for reasoning_budget, env_to_user, MESSAGE_BOUNDARY, prompt-cache-tail, conversation-cache, CONV_WORKSET, DOMAIN_PROFILES and loop_guard over configs/pancake/*.json and the generated pancake-*.json found nothing. MOD_PLAN calls item 11 "existing opt-ins ... -> test arms". — configs/pancake/flashnext-iq2_xs-fast-hip.json; MOD_PLAN.md (development pass 2026-11-08)
- Thinking in history: the template keeps <think> for every earlier assistant turn when preserve_thinking is undefined or true. When the client sends no reasoning_content, it renders an empty "<think>\n\n</think>". Item 31 says: "check that opencode resends earlier reasoning, else the conversation cache re-reads from the first dropped block every turn". — serve/chat_template.jinja:113-122; MOD_PLAN item 31
- The frontend passes an incoming reasoning_content through (frontend.py:591-592). Whether opencode's @ai-sdk/openai-compatible provider sends it back was not checked (see Gaps).
- env_to_user (opt-in, off by default) moves the environment block from the system prompt into the first user message, so a changing date or cwd does not invalidate the prefix. — serve/frontend.py:413-437; serve/server.py:10956-10960
- Tool-schema and system-prompt size (first turn of each session):
  - build agent 11.0-15.2K tokens;
  - oh-my-openagent agents: Sisyphus-Junior about 29.9-40.9K, Sisyphus-ultraworker about 38K, librarian 30.1K, Prometheus 28.8K, strata-coder 28.5K, explore about 19.1K.
  - Turns by agent: build 1,632, Sisyphus-Junior 254, Sisyphus-ultraworker 78, compaction 36, Prometheus 28, plan 14, explore 14. — opencode DB
- Research mapping:
  - Marconi (Princeton/AWS, MLSys 2025): 34.4x token hit rate over vLLM+, TTFT down up to 71.1% on SWE-bench traces.
  - FreeToken: checkpoints at think, tool-call, tool-output and turn boundaries.
  - Continuum and MORI: TTL and idleness ordering for sub-agent switches.
  - Report verdict: "câștigul este de secunde pe fiecare tură editată" (the gain is seconds on each edited turn).
  - The report also warns that a checkpoint after </think> is only valid if the next prompt keeps the thinking text. — /home/dan/reports/Lucrări universități SUA pentru Avalon.md (section "Checkpoint-urile la granițele de tool")
- RadixAttention (Stanford): a multi-state LRU store, saved at chosen boundaries for hybrid models. "Opencode alternează agentul principal, sub-agenții și cererile de rezumat sau compactare, fiecare cu alt prefix" (opencode alternates the main agent, sub-agents and summary or compaction requests, each with its own prefix). — /home/dan/reports/Lucrări Stanford utile pentru Avalon.md (Serving)
- UNSW restore-equivalence test (R1): a full-hit restore must equal a cold recompute, with cache_n/prompt_n checked for off-by-one. Precondition for items 11, 16, 31 and 3. — /home/dan/reports/Lucrări Pacific Africa America Latină.md (R1); MOD_PLAN item 35
- Slipstream (async compaction in tool windows, +8.8 pp and -39.7% latency on SWE-bench Verified with Qwen3.5-9B) and ACE (delta compaction, +10.6% on AppWorld) target the compaction re-read. They are harness-side. — Lucrări universități SUA (Agentul câștigă din asincronie); Lucrări Stanford (Contextul agentului)
- Matching llama.cpp work: #30745 (last-user-end checkpoint, reuse 91% -> 94%), #24891 (checkpoint invalidation after tool requests, "exactly opencode's pattern"), #28092 (disk cache, avoids about 70 s of cold prefill at 131K). — /home/dan/reports/Avalon upgrades from upstream projects.md
- feat/contextpilot: block dedup saved 1.6% of characters over 40 real opencode sessions. Verdict: keep off. — MOD_PLAN series ranking; commit 449d5325c

### Inferences
- At 1,850 tok/s, the three re-read buckets cost about 55 min (same-session), 49 min (interleaving) and 14 min (compaction requests). That is roughly 18% of all local turn time. Prefill is slower at 60-120K than at 11.7K, so this is a lower bound.
- The same-session bucket (median 16.9K, p90 77K beyond growth) is what items 11, 16 and 31 target. It is consistent with checkpoints that are too far back after a tool result or a pruned tool output, and possibly with the thinking-block mismatch.
- The interleaving bucket follows from the RAM budget. A 64-100K conversation needs about 1.4-2.0 GB (about 17.6 KB per token on top of a 244 MB base, extrapolated from the log). A 5.4-6.1 GiB budget therefore holds only 2-3 long conversations, while one oh-my-openagent run uses build + Sisyphus-Junior + explore + compaction. Each sub-agent's own 19-40K system prompt is a cold read the first time.
- Parking older states on the D drive (RadixAttention tiers, Marconi FLOP-per-byte eviction) is the researched fix.
- Compaction re-reads are not solvable in the engine because the compaction prompt shares no prefix. Fixing them needs a harness change: compacting during tool windows (Slipstream), delta compaction (ACE), or a compaction prompt that reuses the agent's prefix.
- Expected benefit, if items 35 -> 31/16/11 work as the author measured: about 33% off edited-turn TTFT, and seconds to tens of seconds on the 11% of turns that re-read.

### Gaps
- Not verified whether opencode resends reasoning_content in assistant history. The DB does not store the HTTP payloads. One logged request is needed (item 31's precondition).
- No pancake measurement of a real opencode session exists. PANCAKE_FINDINGS.md has no "opencode", "TTFT" or "agent turn" entries, and Phase 14.1 (turn timeline) is not done. All the numbers above are from the older backend.
- Prefill tok/s at 64K and 120K on pancake is not documented in the files read, so the minute estimates use the 11.7K rate.
- The cause of each individual same-session re-read (pruning versus thinking mismatch versus checkpoint distance) cannot be separated from aggregate token counts.

## Q2. Thinking-budget behaviour: global budget, reasoning_budget_by_effort, grace soft close, answer_room, reasoning_close_retry, empty answers

### Takeaway
Thinking is uncapped in daily use. opencode sends reasoningEffort "xhigh" by default (mapped to the template's xhigh). No preset sets reasoning_budget_by_effort, reasoning_budget_grace or reasoning_close_retry. Only answer_room is on by default.

On the older backend, runaway generations hit the 8K, 16K or 32K output cap 15 times (0.7% of turns) and burned about 45 minutes. opencode also never sees thinking-token counts, because chat-completions usage omits reasoning_tokens.

### Cited Findings
- opencode model options: "reasoningEffort": "xhigh" by default. Variants low, medium, xhigh and none exist but were never selected (variant = None in all 2,063 turns). There is a "-quick" model with no thinking. — ~/.config/opencode/opencode.jsonc; opencode DB
- Effort mapping: low -> low, medium -> medium, high/xhigh/max -> xhigh, none/minimal -> thinking off. — serve/frontend.py:132-135
- reasoning_budget_by_effort (5dd620fb9): a per-level cap, used only when the request carries no budget of its own. The example in the code is {"low": 2048, "medium": 8192, "high": 0}. Opt-in. Not present in any preset. — serve/server.py:5223-5250, 10986-10992
- reasoning_budget_grace (Maya #613): opt-in, default 0. Closes thinking at the next sentence or line end, at most N tokens later (Maya uses 512). — serve/server.py:10979-10985; MOD_PLAN "Agent robustness from Project Maya"
- answer_room: on by default. From max_tokens >= 1024, thinking without a budget is closed early enough to leave room for an answer. — serve/server.py:225, 4886, 6520-6528
- reasoning_close_retry (#1053): opt-in and off. It closes the thinking once when a reply ends on its stop token while still inside <think>. Maya's resume-the-thinking was not ported because "the empty-answer case is already #1053". — serve/server.py:230, 4887, 7091-7101; MOD_PLAN Maya section
- Runaway turns on the older backend (opencode DB): 15 turns ended with "length". Output was 8,192 (2 turns), 16,384 (10) or 32,000 (3), taking 84-340 s each, 2,726 s in total.
  - Reasoning text per turn: p50 300 chars, p90 2.4K, p99 17.9K, max 135K chars.
  - No turn had "stop" with an empty visible answer (output <= reasoning + 5 with finish stop: 0). But tokens.reasoning is 0 everywhere, so the empty-answer check is weak.
- Usage reporting: reasoning_tokens appears only in the Responses API usage (responses.py:349, 663). The server counts thinking_n (server.py:6552, 7310) but does not put it in chat-completions usage. That is why opencode records reasoning = 0 for every turn. — serve/responses.py; serve/server.py; opencode DB
- Research mapping:
  - s1 budget forcing: a hard cap is principled, and accuracy saturates with budget.
  - ALP: short thinking on easy turns.
  - Berkeley "Danger of Overthinking": about 30% better and 43% less compute picking the less-thinking solution, over 4,018 SWE-Bench trajectories.
  - "Reasoning Models Can Be Effective Without Thinking": NoThinking 51.3 vs Thinking 28.9 at 700 tokens.
  - "Are LRMs Interruptible?": up to 60% accuracy loss on abrupt cuts, hence the soft close.
  - Policy that follows: low or medium by default, escalate on an external signal.
  - The reports note that no paper evaluated effort levels on agentic coding turns. — /home/dan/reports/Lucrări Stanford utile pentru Avalon.md; /home/dan/reports/Lucrări universități SUA pentru Avalon.md
- Thought Calibration (MIT): linear probes on hidden states, up to 60% fewer thinking tokens in distribution and up to 20% out of distribution. It removes the "forked probe" objection to Dynasor (item 21). — /home/dan/reports/Lucrări MIT Harvard pentru Avalon.md
- Thoughtology (Mila/McGill): a rolling 5-gram repetition rate as a rumination signal feeding the soft close (item 34). Entropy Sentinel (UBA): top-k entropy, logged only (item 36). — Lucrări Canada; Lucrări Pacific Africa America Latină
- MOD_PLAN P3c: "20-40% fewer thinking tokens on routine turns", "tens of seconds per turn", changes the text. — MOD_PLAN Phase 13

### Inferences
- The cheapest win is configuration, not code. Possible actions:
  - set reasoning_budget_by_effort, plus reasoning_budget_grace 512 and reasoning_close_retry true, in the daily preset or config;
  - switch opencode's default variant to medium, with xhigh used explicitly for planning.
- The owner's temperature 0.1 stays as it is. The loop guard and the budgets work at that temperature.
- Expected effect: the 15 runaway turns (45 min) would mostly become bounded turns of about 1-2 min. Routine-turn thinking would shrink 20-40% (taken from the papers, not measured on agents).
- Risk: quality loss on hard turns. Use the fixed task set (revised quality gates) and log the cut-off rate.
- Adding completion_tokens_details.reasoning_tokens to chat-completions usage is a small serve change. It would make every budget A/B visible in opencode's own DB. No research covers it; it is plumbing.

### Gaps
- Whether the 15 length-terminated turns were thinking loops or answer loops cannot be told from token counts. tokens.reasoning is 0, and reading the reasoning text was out of scope.
- The real distribution of thinking tokens per effort level on pancake (Flash-Next IQ2_XS) is unmeasured.

## Q3. Reliability: tool-call parse failures and recovery, loops, the failing test

### Takeaway
Tool-call reliability is already good on this workload:
- 22 failed tool calls out of 2,362 (0.9%), plus 23 calls to opencode's "invalid" placeholder (about 1.0%, malformed or unknown calls);
- only 1 edit failed with "oldString not found".

Item 26 (semantic oldString constraint) therefore has little to fix here. The Maya port (loop guard on by default, tolerant tool calls, soft budget close) targets the real tail, the runaway turns. The failing test is a pre-existing, Windows-only keep-awake test that does not affect Linux use.

### Cited Findings
- Tool parts on local Flash-Next turns (opencode DB): 2,362 calls, 2,339 completed, 22 error, 1 running.
  - Top tools: bash 1,369, caveman_retrieve 318, read 231, edit 180, webfetch 51, write 48, glob 29, todowrite 25, task 24, invalid 23.
  - Error classes:
    - caveman_retrieve "unavailable or unknown": 9;
    - read permission or other: 4;
    - webfetch: 3;
    - aborted: 3;
    - edit "oldString not found": 1;
    - bash "unknown": 1.
  - Tool calls per message: 1 (1,590), 2 (251), 3 (47), 4 or more (25).
  - Tool output length: p50 377 chars, p90 4.3K, p99 31K, max 51K.
- Message errors: 18 MessageAbortedError. 1 APIError 500 from the Jinja template ("System message must be at the beginning", chat_template.jinja:110). Pancake now turns late system messages into user messages (_late_system_to_user, serve/frontend.py:439-444), so that error is probably addressed. — opencode DB; serve/frontend.py
- Tool-call recovery is on by default since 0.2.8 (tool_call_recovery). Tolerant handling of an unclosed <parameter= tag: a malformed call never raises. The Avalon report cites Strata's tolerant recovery as 98.3% vs 92.9% of turns read correctly. — serve/server.py:4990, 10451-10456; MOD_PLAN Maya section; /home/dan/reports/Lucrări Stanford utile pentru Avalon.md
- Grammar inside speculative decoding is on in the presets ("--grammar on"). "tool calls / strict schemas always valid", at a cost of 0.1-0.2% per window. — configs/pancake/flashnext-iq2_xs-fast-hip.json; MOD_PLAN status table (phase 6)
- Loop guard (on by default, d1e73edfc):
  - in-loop repeat detection with up to 64 tokens of holdback;
  - repetition penalty 1.15 on the first loop;
  - </think> on the third thinking loop;
  - a cross-turn note after the same tool result 3 times, then blocking;
  - "keep going" at most 3 times;
  - counters in /metrics and Prometheus strata:loop_guard_*_total.
  - Fixtures with an exact loop needed the guard turned off (24c67f2b3), which shows it acts on scripted filler. — MOD_PLAN Maya section; serve/loop_guard.py (690 lines); git log
- Behaviour counters (item 33, UBC/Queen's "Beyond Accuracy": failed runs use 39-343% more tokens and 43-427% more time; Qwen Code fixed 25.81% vs Claude Code 93.28%): N edits without a build or test, M reads without an edit. Log-only first. Not implemented. — /home/dan/reports/Lucrări Canada pentru Avalon.md; MOD_PLAN item 33
- ECLoop (Concordia): require read-before-edit and test-before-done, +4.8 to +11.8 pp Pass@1 on SWE-bench Verified. A harness-side wrapper. — Lucrări Canada
- Type-constrained decoding (ETH, PLDI 2025) and the oldString suffix-automaton idea (item 26). The report itself says to "count how often these failures appear in the opencode logs" before coding. — /home/dan/reports/Lucrări Europa pentru Avalon.md
- presence_penalty must stay 0 for tool calling (DASLab card: SWE-bench 74.7 -> 70.6 with it on). The opencode build agent sets only temperature 0.1 and top_p 0.95. — MOD_PLAN (Europe additions); ~/opencode.json
- Failing test: serve/test_keepawake.py:80 test_service_holds_for_the_whole_request_including_early_close. The ContextPilot commit message records "1604 tests, 1 failure (test_keepawake early_close), which also fails on port/rdna3-decode without this change".
  - The feature is "prevent_sleep", Windows only (KeepAwake.create returns None off Windows).
  - The mechanism is an edge-triggered thread (_loop waits while (count > 0) == held), checked by a polling wait_for with a 3 s timeout. — serve/keepawake.py:25-94; serve/server.py:6495-6506; commit 449d5325c

### Inferences
- The tool-reliability items (26, the oldString constraint) rank low on this data: 1 oldString failure in 180 edits. The research ranking already puts them last (place 10).
- The "invalid" tool rate (about 1%) and the caveman_retrieve "unavailable" errors (9) point at tool-set mismatch more than parse failure. The model called tools that were not enabled in that session (the caveman MCP is "enabled": false in the current config). This is a harness configuration issue, not one for the engine.
- The keep-awake failure is very likely a timing or edge-trigger artefact in a Windows-only feature. Possible fixes: skip it off Windows, or make it deterministic (inject a fake clock or join the thread). Effort is small and there is no user impact on Linux. It does block a "green suite" A/B gate.
- The loop guard's real-workload hit rate is unknown until /metrics is scraped during opencode use.

### Gaps
- Loop-guard trigger counts on real sessions: none recorded, since the guard was added 2026-11-09 and there are no pancake opencode sessions.
- The keep-awake failure was not reproduced (running tests was out of scope), so its root cause is inferred, not confirmed.
- How opencode turns a malformed call into the "invalid" tool, and whether pancake's recovery would have saved those 23 calls, was not checked.

## Q4. Operations: startup, restarts, RAM and disk pressure, preset and config consistency, monitoring, A/B testing, documentation drift

### Takeaway
Operations are the weakest area:
- There is no service unit for pancake. The only units run the older Strata or Avalon on port 11437, while run-pancake.sh defaults to port 8080.
- A cold start is about 1-1.5 min of expert loading, after capped.sh waits for up to 10 min for 48 GiB to be free. The arena build can trip systemd-oomd.
- The conversation cache gets only about 5.4-6.1 GiB of RAM.
- Metrics exist only in memory.
- Several documents disagree about which file is the daily one and about the model size.

### Cited Findings
- Services:
  - strata.service runs /home/dan/Strata/run-iq2_xs.sh, a separate older checkout, with strata-iq2_xs.json on port 11437.
  - avalon.service also uses 11437 (Conflicts=strata.service).
  - Both are inactive and disabled. No pancake unit exists.
  - run-pancake.sh uses PORT default 8080 and execs tools/pancake/capped.sh with serve/server.py. — ~/.config/systemd/user/strata.service, avalon.service; /home/dan/Strata/run-iq2_xs.sh; /home/dan/clone-of-d/src/Strata/run-pancake.sh
- Startup:
  - "experts loaded: 33.02 GiB at 0.69 GiB/s (66 s so far)" (pancake-final-run.out).
  - "mapped pinned cache complement ready: resident 30.62 GiB, pinned 30.62 GiB in 44 s" (pancake-flashnext.log).
  - capped.sh polls until MemAvailable >= WAIT_GIB (48) for up to 120 x 5 s = 10 min, then starts anyway. MemoryMax=54G, swap 0, STRATA_RAM_KEEP_GIB=12. — tools/pancake/capped.sh
- oomd: "building the 34 GB expert arena now often trips systemd-oomd ... STRATA_RAM_KEEP_GIB=20 (mmap tier) loads every time but decodes ~8% slower (69 vs 75 tok/s)". Listed as an open decision. — MOD_PLAN (Status, next session #1); docs/PANCAKE_FINDINGS.md:312, 345
- RAM split: about 33 GiB of experts + 12 GiB kept free leaves 5.4-6.1 GiB for parked conversations. At the time of reading, `free -g` showed 16 GiB available with the other session's benchmark running. — logs; free
- Disk: / 23 GB free (79%), D 74 GB free (91%). Avalon's model needs about 20 GB more on D. The Coder variant (item 25) competes for D. Disk-saved sessions (#28092-style) would also have to go on D. — df; MOD_PLAN items 25 and Phase 11 index
- Preset and config drift:
  - The repo-root strata-iq2_xs.json still points to engine/strata with CUDA lib_dirs and data/expert-profile.bin. The pancake presets use build-hip, data/expert-profile-pancake.bin and the 100401 hipBLASLt table.
  - run-pancake.sh defaults to iq2_xs and calls q2_0 "a lower-quality quant (80.6% same top token as Q8_0)".
  - MOD_PLAN's series ranking says "Item 6 closed (Q2_0 is the daily file)". The Europe index row says "Q2_0 confirmed" (-0.09 task average).
  - opencode only knows the model "qwen3.8-flash-next-iq2_xs". — strata-iq2_xs.json; run-pancake.sh; MOD_PLAN.md:443, 556; ~/.config/opencode/opencode.jsonc
- Model description drift:
  - README.md:56 and MOD_PLAN.md:27/244 say "125B MoE, ~6B active".
  - The Europe research row says "base is 176.9B incl. the 51B n-gram table".
  - The "IQ2_XS" file's experts are actually IQ2_S/IQ2_XXS/IQ1_M with Q2_0 down (MOD_PLAN.md:443, item 24).
  - The preset is still named flashnext-iq2_xs*. — README.md; MOD_PLAN.md
- BFLA scope mismatch: item 19 says BFLA "benchmarks Flash-Next itself, 1.34-1.44x TTFT at 128-256K". The feat/bfla-hip port (e0988bf54) touches src/qwen35/q36_engine.cu, the Qwen3.6-35B engine, not the main Flash-Next engine. — MOD_PLAN item 19; git show --stat e0988bf54
- Monitoring:
  - /metrics JSON and Prometheus (prometheus.py: TTFT histogram, strata:* facts, llamacpp:prompt_tokens_total);
  - the conversation-cache card parsed from the engine log (ConvCacheLog, server.py:947-1027);
  - per-request "[strata] done: ... expert cache X% hit" lines on stdout.
  - Nothing persists per-turn records, and there is no scraper or TSDB in the configs read. Phase 14.1 (turn timeline) and Phase 14's regression watch and automatic A/B harness are plans only. — serve/prometheus.py; MOD_PLAN Phase 14
- A/B and quality gates: same-binary interleaved pairs, 3 runs, median, plus greedy exactness over 900+ tokens (Bench). For lossy items: token forcing rejects only large deviations, and close variants are chosen by a fixed task set (LiveCodeBench subset, BFCL v4 multi-turn, own-repo checkpoints) with repeated runs ("Displacement Is Not Direction", Toronto). The test-phase queue is manual (scratchpad queue.sh). — MOD_PLAN Bench, Canada additions, development pass; /home/dan/reports/Lucrări Canada pentru Avalon.md
- Branches: four chains are unmerged (port/faststrata-poly -> exp/expert-prefetch -> feat/grammar -> port/rdna3-decode) plus side branches feat/disco (+1), feat/contextpilot (+2), feat/bitdecoding-hip (+2) and feat/bfla-hip (+4). The working tree has uncommitted changes to src/prefill/prefill.cpp and src/program/generate.cpp. — git; MOD_PLAN "Owner's call"

### Inferences
- The highest-value ops fix needs no research: add a pancake.service with Conflicts= for strata and avalon, PORT=11437, the ExecStartPre mount wait, and capped.sh semantics.
- MOD_PLAN's next-session #1 load decision (RAM_KEEP 20 versus pinning the arena in steps) also sets the conversation-cache budget. A larger keep-free reserve buys parked conversations at the cost of about 8% decode. Decide it together with the interleaving data from Q1.
- Persisting the per-request "done" record (TTFT, cache_n, prompt_n, thinking_n, loop-guard events, expert hit) to a JSONL file on D is the prerequisite for Phase 14.1. It is also the prerequisite for any A/B of items 11, 16, 31 or 12 on real sessions.
- The doc drift (125B vs 176.9B, "IQ2_XS", which file is daily, stale root config) is cheap to fix. It matters because the A/B harness and future sessions read these files as truth.

### Gaps
- Exact end-to-end cold-start time (process start to "ready") was not in the logs read. Only the expert-load and pin phases were found.
- Restart frequency and crash history under opencode use are unknown (no service log for pancake).
- Whether opencode's ~/opencode.json (a llama-dgpu provider at 127.0.0.1:8080/v1) is ever active is uncertain. It depends on opencode's project-config lookup from the cwd. If it is active, it collides with pancake's default port.

## Q5. Integration gaps with opencode

### Takeaway
The opencode side was set up for the older Strata and Avalon on port 11437, with a cloud default model. For pancake to be used daily, opencode needs:
- the right port;
- a context limit for the sub-agent that matches the server;
- reasoning-token reporting;
- an effort default that agrees with the research;
- a decision about the oh-my-openagent sub-agents, whose 19-40K system prompts multiply cold reads on a single slot.

### Cited Findings
- Provider "strata": baseURL http://127.0.0.1:11437/v1, models qwen3.8-flash-next-iq2_xs (context 131072, output 32768, reasoningEffort xhigh), "-quick" (none, output 2048) and qwen3.6-35b-a3b (context 163840).
  - The global default model is the cloud "opencode/claude-sonnet-5-5". small_model is "opencode/glm-5.3-flash-free", so title requests go to the cloud and not to the local slot. — ~/.config/opencode/opencode.jsonc
- The strata-coder sub-agent uses "strata/qwen3.6-35b-a3b". Pancake's model_for answers any unknown name with its own model (server.py:5211-5215), so the sub-agent gets Flash-Next silently. opencode, however, sizes that model's context as 163,840 against a server maximum of 131,072. — opencode.jsonc; serve/server.py; configs/pancake/flashnext-iq2_xs-fast-hip.json (--max-context 131072)
- Plugin oh-my-openagent@latest (unpinned): Sisyphus, Prometheus, explore and librarian agents produced 374+ local turns, and their first prompts are 19-41K tokens. Background tasks (background_output 12, background_cancel 5 tool calls) imply concurrent requests to a single-slot server. — opencode.jsonc; opencode DB
- MCP servers enabled: graphify and token-savior. Their tool schemas add to every prompt. caveman is disabled, yet 318 caveman_retrieve calls appear in the local turns (from when it was enabled). — opencode.jsonc; opencode DB
- Agent-harness items from the reports (outside the engine): low or medium default effort with escalation on an external signal; ACE delta compaction or Slipstream async compaction in tool-idle windows; a Recursive-LM tool for huge inputs; AsyncFC (1.4x end to end on code tasks, needs an opencode client fork); GEPA on the compaction prompt. — MOD_PLAN (Agent-harness items); reports listed above
- Prefix layout: "un prompt de sistem stabil la nivel de octet ... fără timestamp-uri" (a byte-stable system prompt, with no timestamps); env_to_user already implements the cache-side part. A fixed state block at the end of the prompt (MEM1 pattern) and the STAIR or Dynamic Cheatsheet suggestions must go after the shared prefix. — Lucrări Stanford; Lucrări MIT Harvard; Lucrări Canada

### Inferences
- A sub-agent request above 131,072 tokens would hit the server's context check, which has a test, serve/test_context_overflow.py. Set strata-coder's limit to 131072, or point it at the Flash-Next model name.
- Item 11's "one system/tool prompt prefix shared across sessions" only helps if the sub-agents share a byte-identical start. With Qwen's template, tools are rendered inside the system message after the agent's own system text, and each oh-my-openagent agent has a different system text. So there is probably no shared prefix today.
- No researched technology covers reordering the prompt as tools first, then agent text. ContextPilot's reordering is for RAG documents.

### Gaps
- opencode's openai-compatible provider behaviour was not verified for: resending reasoning_content, sending reasoning_effort for variants, and session headers (for llama.cpp #29738-style slot keys).
- Whether oh-my-openagent actually issues parallel requests (versus queued ones) on the local provider was not verified.

## Q6. Ranked pain-point map (user-visible impact x fixability)

### Takeaway
Rank order:
1. Make pancake reachable and configured for agent use: service, port, budgets, cache switches.
2. Bound thinking: per-effort caps and soft close; consider a medium default.
3. Fix the per-turn re-read tail, after the restore-equivalence test.
4. Make turns observable: reasoning_tokens in usage, a per-turn log.
5. Handle sub-agent interleaving: RAM budget, disk tier, prefix layout.
6. Compaction re-reads (harness-side).

Tool-call parse reliability and the keep-awake test are low impact.

### Cited Findings
Ranking (each row's evidence is in Q1-Q5 above. "Addressed?" means present in code and enabled by default):

| # | Pain point | Evidence | Cause | Technologies that close it (source) | Expected benefit | Effort | Risk | Addressed? |
|---|---|---|---|---|---|---|---|---|
| 1 | Pancake is not wired into daily use | No unit; run-pancake.sh on port 8080 vs opencode on 11437; only 2 local turns since 11-02 | Launcher written for benches | None needed (ops): pancake.service with Conflicts=, PORT=11437, mount wait | Enables everything else | S | Low | No |
| 2 | Uncapped thinking, runaway turns | xhigh default; 15 length cut-offs (8K-32K tokens) = 2,726 s; no budget keys in presets | Budgets opt-in and unset | s1 + ALP + Overthinking + NoThinking (Stanford, US reports) -> reasoning_budget_by_effort (5dd620fb9); Interruptible LRMs -> reasoning_budget_grace 512 (Maya); reasoning_close_retry (#1053); later item 21 (Thought Calibration), item 34 (Thoughtology) | Bounded worst case (minutes to about 1-2 min); 20-40% fewer thinking tokens (papers, est.) | S (config) / L (item 21) | Medium (quality, so gate with the fixed task set) | Code yes, enabled no |
| 3 | Same-session re-reads | 207 turns (11%), median 16.9K, p90 77K tokens beyond growth, 6.05M total (about 55 min at 1,850 tok/s) | Checkpoint distance after tool results or pruning; possible thinking mismatch | UNSW restore-equivalence (item 35) -> preserve_thinking end to end (31), FreeToken/Marconi tool-boundary checkpoints (16), message-boundary + tail (11; docs/MESSAGE_BOUNDARY_CACHE.md: first edit -33%) | Seconds to tens of seconds on 11% of turns | S (switches) / M (16) | Low (exact, if 35 passes) | Opt-in only (11); 16, 31, 35 open |
| 4 | Turns not observable | tokens.reasoning = 0 in all 2,063 turns; metrics in memory only; Phase 14.1 not done; no pancake opencode data | Chat usage lacks reasoning_tokens; no per-turn log | No paper: plumbing. Phase 14 harness; Stanford "measure first" | Makes items 2, 3, 5 testable | S | None | No |
| 5 | Sub-agent and session interleave misses | About 103 events, about 5.5M tokens (about 50 min); 4 slots and 5.4-6.1 GiB budget; snapshot about 1.1 GB at 51K tokens; agents' first prompts 19-41K | Single slot plus a small RAM budget; distinct system prompts | RadixAttention multi-state store (Stanford); Marconi FLOP/byte eviction (16); Continuum/MORI TTL; disk tier (llama.cpp #28092, phase 7 "automatic disk sessions"); --conversation-cache-min-tokens; item 3 conversation working set | Tens of seconds per switch | M | Low-medium (RAM vs the oomd decision) | Partly (parking exists, sized small) |
| 6 | Compaction re-reads | 38 compaction requests, 1.51M tokens; +28 large misses after compaction; prune:true | Compaction prompt has its own prefix | Slipstream async compaction, ACE delta compaction, GEPA prompt (harness); Marconi checkpoint at the compaction boundary | Tens of seconds per compaction | M-L (opencode side) | Medium (quality) | No |
| 7 | RAM, load and oomd fragility | Arena 33 GiB at 0.69 GiB/s; capped.sh waits up to 10 min for 48 GiB; oomd kills | 57 GB box, 34 GB arena | None in research; MOD_PLAN next-session #1 (RAM_KEEP 20 = -8% decode, or stepwise pinning); phase 7 sleep-in-place (basalt) | Reliable restarts; more RAM for parking | S-M | Low | Open decision |
| 8 | Config, preset and doc drift | 125B vs 176.9B; "IQ2_XS" really IQ2_S/IQ2_XXS/IQ1_M + Q2_0; Q2_0 "daily" vs iq2_xs default; stale strata-iq2_xs.json; strata-coder 163,840 ctx vs 131,072; BFLA ported to the qwen35 engine, not Flash-Next | Fast iteration | None (housekeeping) | Fewer wrong A/Bs and wrong decisions | S | None | No |
| 9 | Long-prompt TTFT at 64-131K | Prompt p50 64.5K, p90 93K; 1,850-1,920 tok/s measured only at 11.7K | Prefill GPU-bound; GDN and attention | Item 2 (keep the decode cache through long prompts, +20% decode after long prompts), item 18 (GDN chunk 64, 5-11% prefill), item 19 BFLA (1.34-1.44x TTFT at 128-256K, needs a main-engine port), hipBLASLt table (already in presets) | 5-40% on cold reads (est.) | M-L | Medium (19 is lossy) | Table yes; others open |
| 10 | Behavioural loops (not text repeats) | Failed runs cost 39-343% more tokens (UBC); guard catches only exact repeats | Guard is lexical | Item 33 counters, ECLoop read-before-edit / test-before-done, item 36 entropy (log-only), fresh-context review (MIT Self-Repair) | Fewer wasted sessions (unmeasured) | S-M | Low (log-only first) | Text loops yes (guard on); behaviour no |
| 11 | Tool-call failures | 22 errors / 2,362 (0.9%); 23 "invalid"; 1 oldString miss in 180 edits | Mostly tool-set mismatch | Already: grammar in spec, tolerant recovery, late-system folding; item 26 (ETH type-constrained) has little to fix | Small | M (26) | Low | Mostly yes |
| 12 | Failing keep-awake test | test_keepawake.py:80 fails on base and branch | Windows-only edge-triggered thread plus a 3 s poll | None needed: skip off Windows or make it deterministic | Green suite for A/B gates | S | None | No |

- Series-wide ranking puts "correct, cheap per-turn cache with the restore test (35, 31, 16, 11)" 4th and "effort and thinking (12, 21, 33, 34)" 8th, below engine work (GPU profile, turbo-hip, sparsity masks). — /home/dan/reports/Lucrări Pacific Africa America Latină.md (Închiderea seriei); MOD_PLAN series-wide ranking

### Inferences
- From the user's side of an agent session, the order above differs from the series ranking.
  - The series ranks by decode-window gains. The opencode data says a median turn is only about 6 s, but the p90 is 50 s and the p99 224 s.
  - That tail comes from re-reads, runaway thinking and interleaving, not from tok/s.
  - So rows 1-5 (mostly configuration and small serve changes) likely give more user-visible time per hour of work than places 1-3 of the series ranking. They should be validated with row 4's logging before and after.
- Gaps that no research covers:
  1. service, port and lifecycle integration;
  2. reasoning_tokens in chat-completions usage and a persistent per-turn log;
  3. reordering the prompt so the tool schemas form a shared prefix across opencode sub-agents with different system prompts;
  4. matching the opencode limits to the server (context and output caps per model name);
  5. a RAM-budget policy that trades expert-arena RAM against parked-conversation RAM;
  6. a cold-start shortcut (snapshotting the pinned arena, or a faster D-drive read) beyond basalt's sleep-in-place;
  7. a pinned oh-my-openagent version and a tool-set audit, to cut "invalid" and unavailable-tool calls.

### Gaps
- No user-visible impact figures are measured on pancake. All agent-side numbers come from the older backend's opencode history, and the gains are the authors' or estimates.
- Thinking-token counts per turn and loop-guard trigger rates are unknown until row 4 is done.
