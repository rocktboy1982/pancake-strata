# TTFT bench: time to first token at long context and over an agent session

`tools/pancake/ttft_bench.py` (stdlib only) measures the time from sending a request to the first streamed token.
Why: an agent workflow of 10 sequential calls pays every second of TTFT ten times, so TTFT can matter more than a 5-10%
decode gain. This page says how to run it. **No result is recorded here yet: the harness is written and unit-tested
against a fake server only (MOD_PLAN Phase 17 item 17.0).** Scenarios C, D, E and the engine-log attach (below) were added
for the measurements of `TTFT_LEVERS.md` section 7.

It uses only `POST /v1/chat/completions` with `stream=true`, so it works unchanged against pancake's Strata server,
Avalon's llama-server and Strata-rdna3. It never starts or stops an engine; start one first, alone, and keep the GPU
and the CPU otherwise idle (it prints a warning and records `load.warning` when the 1-minute load average is above 2.0).

## What it measures

- **Scenario A, single long prompts.** Sizes 8K, 32K, 64K, 100K tokens (K = 1024; `--sizes 8k,32k,...` or plain
  numbers). Each prompt is `tools/ab_engine.py long_prompt(words)` with a unique random first line and a random rotation
  of its word list, so neither a prefix cache nor a content cache can help. Words per token are estimated and then
  calibrated on the first run of each size; the tables list the ACTUAL prompt tokens the server reports. 4 runs per size,
  run 1 discarded, medians over the other three. `max_tokens` 16.
- **Scenario B, agent session.** The system prompt and four tools of `tools/agent_turns_bench.py`, then 10 turns (`--turns`).
  Every turn adds about 1,500 words of a project file (consecutive chunks of this checkout's own documents) and resends
  the whole conversation including the model's own earlier replies. Per turn: TTFT, prompt tokens, tokens reused from the
  server's cache, tokens read, and tokens of the earlier conversation that had to be read again. The summary has the
  per-turn table, the total TTFT over the session and the share of prompt tokens reused. Each session starts with a
  random session id in the system prompt, so a session never reuses an earlier one.
- **TTFT** is the client's clock: from just before the request is sent to the first delta with content, reasoning or a
  tool call. Empty role-only deltas do not count. The engine's own `prompt_ms` / `prompt_per_second` from `timings` are
  recorded next to it when the server sends them; `ttft_minus_engine_prompt_s` is the difference (queueing, tokenization,
  the first decode step, the network), derived, not measured by the engine.
- `--think off` (default) sends `chat_template_kwargs {"enable_thinking": false}`; `--think on` leaves it out.

## Scenarios C, D, E and the engine-log attach (Phase 17 item 17.0, measurements M1, M2, M5)

These were added because `TTFT_LEVERS.md` section 7 says the measurements below must exist before any code is changed.
They are written and unit-tested against the fake server only; **no result is recorded yet**, and nothing here says how
long a run takes (not measured).

- **Scenario C, session with side requests (`--scenario c`).** The 10-turn session of scenario B (the same system prompt,
  tools and file chunks, so the main conversation is the same as B's apart from the session id), plus requests that are
  unrelated to it, sent after the main turn's reply is in the conversation and before the next main turn:
  - **Side requests** at `--side-after 3,6` (default; empty string for none): its own fixed system prompt (title and
    one-line summaries), no tools, 1 user message or, with `--side-messages 2`, user / assistant / user, about
    `--side-words 60` words of unique text. This is the shape of the title or summary request an agent client sends
    on the side; it is a guess at that shape, not a recording of any client (TTFT_LEVERS M4 would give the real one).
  - **One sub-agent switch** with `--switch-after N` (0 = none): after main turn N the harness runs `--subagent-turns 2`
    turns of a second long conversation (another system prompt, two other tools, other file chunks, each turn adding
    `--agent-words` words and resending its own whole conversation), then returns to the main conversation. When a side
    request and the switch sit after the same turn, the side request goes first.
  - Positions must be before the last turn (a request after the last turn disturbs nothing); the harness refuses them.
  - Per main turn it records what B records plus `after_side` (side requests since the previous main turn) and
    `after_switch`. The summary table has an `after` column (`side`, `switch`, `-`), and turns 2+ are grouped into
    `plain`, `after_side` and `after_switch` with the median TTFT, tokens read and tokens re-read per group, which is the
    number that says what a disturbance cost. Side and sub-agent requests are recorded separately
    (`other_requests`, with `kind`, `after_turn`, TTFT, reuse). A failed side or sub-agent request is recorded and the
    session goes on; a failed main turn ends the session as in B.
  - Reading it: the conversation cache of the server decides whether the turn after a side request resumes the live
    conversation, restores a parked one, or re-reads it. `reused_tokens` and `reread_tokens` of those turns show which.
- **Scenario D, tight budget (nothing to implement in the engine).** Run scenario C again against a server started with
  a small conversation cache, `--conversation-cache-mib 1280` in TTFT_LEVERS M2 (the flag is named in the server's startup
  message in `src/program/generate.cpp`; this harness does not set it). The harness cannot read the budget from the server, so the operator
  records it: `--note-cache-budget-mib 1280` stores `cache_budget_mib` in the JSON, prints it under the load line, and
  `--compare A.json B.json` prints `conversation-cache budget (noted by the operator): A ..., B ...` (`not noted` for a
  file without it; no line when neither has one). The engine log lines `conversation cache: skip parking (...)` and
  `evicted` for the run are attached with `--prefill-timing-log` (below) and show whether the budget bit.
- **Scenario E, warm-delta sweep (`--scenario e`, TTFT_LEVERS M5).** One conversation of about `--delta-base 30k`
  tokens (the same system prompt and tools as B, one user message of the scenario A word list, unique first line) is built
  once. Then, in `--delta-repeats 3` rounds, each size of `--deltas 50,100,300,1000,3000` (tokens, ascending) is appended
  as a new user message of unique random words and the reply is kept in the conversation, so each request is the warm
  path of an agent turn. Order is round by round, every size once per round, so sizes are compared at nearly the same
  depth; the conversation grows by the deltas and replies (about 13K tokens over a full sweep) and `prompt_tokens`
  (depth) is recorded per request. Words per token start from `--tokens-per-word` and are corrected from the base
  request and from each delta of 200 words or more; `new_tokens_est` is the client-side count (this request's prompt
  tokens minus the previous prompt tokens and the previous reply), which is the new message plus a few chat-template
  tokens and works for every server; `read_tokens` is the server's own count when it reports one.
  - The summary has, per delta size, the median, min and max TTFT, new tokens, tokens read, depth and engine prompt ms,
    and two least-squares fits `TTFT = intercept + slope * new tokens`, over all requests and over deltas of 1000+. The
    staged-path floor (0.55-0.75 s for 300-400 new tokens in the logs) is read from the small-delta rows; the fit
    intercept is the floor only if the small deltas lie on the line, so look at the table first. The slope per extra
    token is the second number.
  - M5 asks for 32K and 64K depth: run it twice, `--delta-base 32k` and `--delta-base 64k` (the server needs a context
    of about base + 14K tokens or the sweep ends with a recorded `skipped: context limit`; the rows measured so far are
    kept). For "share 0.6 vs auto" start the engine twice with `STRATA_PREFILL_CPU_SHARE` set differently and compare
    the two JSON files with `--compare`.
- **Engine-log attach (`--prefill-timing-log PATH`, M1).** PATH is the file the engine's stderr (and, for the `RESUME`
  line, stdout) goes to. Before each request the harness notes the file size; after it (`--log-settle-s 0.3` seconds
  later, so the last lines reach the file) it reads what was added and stores, in the request's record as `engine_log`,
  the matching lines (at most 120) and their parsed numbers. It works for every scenario's requests (A, B, C including
  side and sub-agent requests, E). Without the option no record changes. The matched lines, from the engine code:

  | Line (engine code) | Parsed into |
  | --- | --- |
  | `strata prefill timing: N tokens, GPU timeline X ms, wall Y ms, host staging Z ms: <phase> ms (pct%) ...` (`STRATA_PREFILL_TIMING=1`, `src/prefill/prefill.cpp`) | `prefill_timing[]`: tokens, gpu_ms, wall_ms, host_staging_ms, phases (ms and %) |
  | `strata prefill timing: host: chunk setup ... A ms, waiting for each chunk B ms, after each chunk ... C ms, PLE D ms` | `host` of the previous timing entry |
  | `strata prefill timing (layer-major): N tokens, GPU timeline X ms: ...` | a `prefill_timing[]` entry with `layer_major: true` |
  | `strata prefill: N tokens in M-token chunks over S stages (--prefill-pipe ...)` (the chunk plan, printed only when the plan is chunked) | `chunk_plan[]` |
  | `strata ple wait: chunk at ...` | counted (`ple_wait_lines`), kept in `lines` |
  | `conversation cache: parked N tokens in X ms`, `skip parking (...)`, `restored N tokens ... in X ms`, `evicted N tokens` | `parks`, `skips`, `restores`, `evicted` |
  | `strata serve: file tier I/O this request: the OS read X MB from storage` | `file_tier_mb` |
  | `RESUME N` (the engine's stdout) | `resume` |

  Kept as text in `lines` but not parsed: the peer-GPU timing lines. Not matched at all, so not in the record: the
  `strata serve: KV streaming: ...` line (it exists in `generate.cpp`) and any other `STRATA_TRACE=1` output. The engine log has no timestamps, so lines are tied to a request by file
  position only. On HIP the park of the previous conversation is printed while the NEXT request is served (that is what
  `ttft_audit.py` says), so a park line appears on the request after the one that caused it. If the file is shorter
  than the offset (rotated), the harness reads it from the start and sets `truncated_file`; if it cannot be read the
  record has `error`. The text summary lists, per Scenario A size, the largest `prefill timing` line of the last kept run
  (GPU, wall, host staging, top four phases) and says so when no `prefill timing` line was found (then the engine was
  probably not started with `STRATA_PREFILL_TIMING=1`, or the file is not where its stderr goes).

### Night-run commands

One engine at a time, nothing else heavy running, one command per measurement. Port 8080 is `./run-pancake.sh`'s default
(the systemd unit uses 11437). For every run that attaches the log, start the engine with `STRATA_PREFILL_TIMING=1` in its
environment and its stderr (and stdout) in `ENGINE.log`; the file may be shared by all runs because each request reads
only its own part.

```sh
# M1: single prompts, all four sizes, the breakdown attached (the first run of each size is kept in the JSON, marked discarded)
python3 tools/pancake/ttft_bench.py --port 8080 --scenario a --label m1-pancake --out ttft-m1.json --prefill-timing-log ENGINE.log

# M2 variant 1: the session alone (scenario B)
python3 tools/pancake/ttft_bench.py --port 8080 --scenario b --label m2-alone --out ttft-m2-alone.json --prefill-timing-log ENGINE.log

# M2 variant 2: the session with a side request after turn 3 and one sub-agent switch after turn 6
python3 tools/pancake/ttft_bench.py --port 8080 --scenario c --side-after 3 --switch-after 6 --label m2-side --out ttft-m2-side.json --prefill-timing-log ENGINE.log

# M2 variant 3 (D): restart the server with a tight budget (--conversation-cache-mib 1280), then the same command, noting the budget
python3 tools/pancake/ttft_bench.py --port 8080 --scenario c --side-after 3 --switch-after 6 --note-cache-budget-mib 1280 --label m2-tight --out ttft-m2-tight.json --prefill-timing-log ENGINE.log

# M5: warm-delta sweep at 32K and at 64K depth
python3 tools/pancake/ttft_bench.py --port 8080 --scenario e --delta-base 32k --label m5-32k --out ttft-m5-32k.json --prefill-timing-log ENGINE.log
python3 tools/pancake/ttft_bench.py --port 8080 --scenario e --delta-base 64k --label m5-64k --out ttft-m5-64k.json --prefill-timing-log ENGINE.log

# two runs side by side (same scenario in both files: variants 2 and 3 here, or two builds)
python3 tools/pancake/ttft_bench.py --compare ttft-m2-side.json ttft-m2-tight.json
```

Variant 1 is stored as `scenario_b`, variants 2 and 3 as `scenario_c`, with the same main conversation turn by turn;
`--compare` compares B with B and C with C, so read variant 1 against the others from the per-turn tables of each
summary. Run `ttft_audit.py` on the server's turn log for the same period as a cross-check.

## Which fields report cache reuse (read from the code, not measured)

| Server | Reuse field the harness reads | Where |
| --- | --- | --- |
| pancake / Strata server | `usage.prompt_tokens_details.cached_tokens` and `timings.cache_n` (`timings.prompt_n` = tokens read), both in the final streamed chunk | `serve/server.py` (final chunk of the streamed reply), asserted in `serve/test_server.py` `UsageAndStatus.test_openai_stream` and `serve/test_request_accounting.py` |
| Avalon's llama-server | `timings.cache_n` / `timings.prompt_n` / `timings.prompt_ms` / `timings.prompt_per_second` on the last chunk; `usage.prompt_tokens_details.cached_tokens` only on the usage chunk, which the harness requests with `stream_options.include_usage` | `Avalon/tools/server/server-task.cpp` (`usage_json_oaicompat`, the streamed final chunk), `server-common.cpp` (`server_slot_stats::to_json`) |
| Strata-rdna3 | not checked: its server source was not read for this page | run it once and look at the result JSON: `reused_source` is `null` when neither field exists, and the summary then says "not reported" instead of guessing |

The harness prefers `usage.prompt_tokens_details.cached_tokens` and falls back to `timings.cache_n`; the field it used
is stored per request as `reused_source`.

## How to run it

One engine at a time, nothing else heavy running. `--label` names the run in the output; `--out` is the result JSON.

```sh
# pancake (Strata server, ./run-pancake.sh keeps the default port 8080; the systemd unit uses 11437)
./run-pancake.sh iq2_xs fast                       # in another terminal, wait until it answers
python3 tools/pancake/ttft_bench.py --port 8080 --label pancake-iq2_xs-fast --out ttft-pancake.json

# Avalon's llama-server (start it the way its own guide says; the port is whatever you gave --port)
python3 tools/pancake/ttft_bench.py --port 11437 --label avalon --out ttft-avalon.json

# Strata-rdna3 (its server on whatever port you started it on; 8081 is only an example)
python3 tools/pancake/ttft_bench.py --port 8081 --label strata-rdna3 --out ttft-rdna3.json

# side by side (negative change = the second file is faster)
python3 tools/pancake/ttft_bench.py --compare ttft-pancake.json ttft-avalon.json
```

Useful options: `--sizes 8k,32k` (fewer or smaller sizes), `--scenario a|b|both|c|e|all` (`both` is still A and B; C, D and E
are described above), `--runs 4`, `--turns 10`,
`--sessions 3` (scenario B medians per turn), `--api-key KEY` or `TTFT_BENCH_API_KEY` (never written to the JSON),
`--host`, `--no-calibrate`, `--timeout`. Run time is dominated by the 64K and 100K prompts: at 4 runs per size that is
eight reads of 64K-100K tokens, so plan for a long run and start with `--sizes 8k,32k --scenario a`.

## Reading the result

- A size is **skipped, not failed** when the server refuses it for the context window (HTTP 400 with a message about
  context, "exceeds", "too long" and similar): the refusal is recorded, and larger sizes are skipped without being tried.
  100K tokens needs a context of at least about 102.5K tokens (prompt plus the 16 new tokens).
- Different engines may tokenize the same text into different token counts. `--compare` prints both actual prompt token
  counts and flags a difference above 5%; compare engine tok/s as well as seconds when it does.
- Scenario B turn 1 has no reuse by design (a fresh session id sits at the start of the system prompt); a server with a
  conversation cache should reuse almost the whole earlier conversation from turn 2 on, so "re-read" tokens (the earlier
  conversation read again) show what its cache missed.
- A load average above 2.0 at the start or the end marks the whole file `load.warning: true`.

Tests (fake streaming server, no engine): `nice -n 19 python3 -I -m pytest -q tools/pancake/test_ttft_bench.py`.
