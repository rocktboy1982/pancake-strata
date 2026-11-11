# TTFT bench: time to first token at long context and over an agent session

`tools/pancake/ttft_bench.py` (stdlib only) measures the time from sending a request to the first streamed token.
Why: an agent workflow of 10 sequential calls pays every second of TTFT ten times, so TTFT can matter more than a 5-10%
decode gain. This page says how to run it. **No result is recorded here yet: the harness is written and unit-tested
against a fake server only (MOD_PLAN Phase 17 item 17.0).**

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

Useful options: `--sizes 8k,32k` (fewer or smaller sizes), `--scenario a|b|both`, `--runs 4`, `--turns 10`,
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
