# Prefix check: does the client resend a byte-identical prefix between turns? (MOD_PLAN 17.0, measurement M4)

2026-11-11, branch `dev/p17/prefixcheck` (from `dev/p17/integration`). Code only: nothing was run against a real client,
no engine, no GPU. Everything below that says what a divergence costs is read from the code (`serve/chat_template.jinja`,
`serve/frontend.py`) and from `pancake-docs/TTFT_LEVERS.md`; none of it is measured yet. The 28 tests use synthetic
requests.

## 1. The question

The engine's conversation cache continues a conversation only when the new prompt starts with ALL the tokens it holds: the
last prompt plus the reply it generated. A full re-read of a 55K prompt costs 21-25 s in our logs
(`TTFT_LEVERS.md`, section 0). So the first thing to know about a client is whether, in turn N+1, the messages array starts
with the messages of turn N plus the assistant reply the client received, byte for byte after rendering:

- does it resend the reply with its `reasoning_content` (OpenAI) or thinking block (Anthropic)?
- does it resend the tool calls as received (ids, argument text, key order)?
- is the tool list identical, in the same order? Is the system prompt byte-stable (no date, no per-request id)?
- are old tool results left alone (opencode prunes old outputs; other clients truncate)?

`tools/pancake/prefix_check.py` answers this from two or more captured request bodies. It needs no engine. The server can
write the capture itself (`STRATA_REQUEST_CAPTURE`, section 3).

## 2. Privacy (read first)

A request body is the user's prompts, code and tool output; the reply is the model's answer. A capture holds all of it as
plain text.

- The capture goes to the local directory YOU name in `STRATA_REQUEST_CAPTURE`, and nowhere else. The code opens no socket
  and sends nothing (`serve/request_capture.py`, `tools/pancake/prefix_check.py`). The directory is created with mode
  0700 and the file with 0600.
- No `Authorization` header, API key or cookie is recorded: only `User-Agent` and any header whose name contains
  `session`, `conversation` or `thread` (a client's conversation id, used to tell sessions apart).
- The capture is bounded: `STRATA_REQUEST_CAPTURE_MAX_MB` (default 512). The dev box's root disk is small; point the
  directory at the D drive (`/home/dan/clone-of-d/...`) and delete the file when the measurement is done.
- To share a capture, run it through `--redact` first (section 5). A redacted file has every message text, system text,
  tool-call argument, tool result and reply replaced by its length and hashes, and each tool by its name and hashes.
  The comparison on a redacted file gives the same verdicts. Look at the redacted file before sending it.
- The report itself holds no capture text unless you pass `--snippets N`.

## 3. Capturing

### 3.1 What already existed, and why it is not enough

| Existing | Where | Why not |
| --- | --- | --- |
| `turn_log` / `STRATA_TURN_LOG` (item 73) | `serve/server.py` `_write_turn_log` | numbers only, by design (no prompt text); it cannot say what the client sent. It cannot be converted to a capture. |
| API monitor (`--api-monitor`, `/api/requests?id=...`) | `serve/server.py` `begin_request` | memory only, last 100 requests, each input cut at 262,144 characters (a 55K-token agent prompt plus tools is larger), the reply's tool calls are not kept. A record that fits can still be read: `prefix_check.py` reads a monitor record (`input` text) and skips a truncated one, but then has no reply to compare. |
| `STRATA_PARK_DIVERGENCE_LOG` (M3) | `serve/park_divergence.py` | token level, in the engine's view, after the server's own fixes; it tells where the prompt left the cache, not what the client changed. The two are complementary (section 6). |

So a minimal opt-in was added.

### 3.2 `STRATA_REQUEST_CAPTURE=DIR`

Unset (the default): no behaviour change, nothing is recorded or printed. Set: one JSON line per `/v1/chat/completions` or
`/v1/messages` request, appended to `DIR/requests-<date>-<pid>.jsonl`:

```
{"ts", "t", "path", "session", "client": {"user-agent", ...}, "stream",
 "request": <the body exactly as parsed, key order kept, taken before the server edits it>,
 "response": {"message": <the assistant message in the API's own format: content, reasoning_content, tool_calls with the
                           streamed argument text; or Anthropic content blocks>, "finish_reason", "usage"},
 "http_status"}
```

The start prints one line with the file name and the privacy note. A line is written when the request handler ends, a moment after the client has its answer. `/v1/responses` is not captured (the checker does not
read that format).

Start the server as usual with the variable added. With `run-pancake.sh` (port 8080 by default; the owner's Copilot test
used 11437, `PORT=11437`):

```
STRATA_REQUEST_CAPTURE=/home/dan/clone-of-d/captures PORT=11437 ./run-pancake.sh
```

(an engine must be running for a real request; the capture itself needs no engine: any request that reaches
`/v1/chat/completions` is recorded, including one answered by a mock or refused with a 400).

### 3.3 Capture steps for opencode

1. Start the server with the line above (a night-time or idle-machine job; the daytime rule bans starting an engine).
2. Point opencode at it: the `tabby` provider in opencode's config already has the server's `/v1` base URL
   (`reference_opencode-headless` in the owner's notes). Use a throw-away directory with a small repository, so the prompt is
   not full of private code.
3. One short run that makes a tool call, then continues after the tool result, gives two consecutive requests:
   `opencode run "Read README.md with your read tool and tell me its first line." < /dev/null`
   Request 1 is the user message; the model calls the read tool; request 2 carries the assistant turn and the tool
   result. A second user message in the same session (for example typed in the TUI) adds a third request that checks
   behaviour across user turns.
4. Stop the server when the run ends (or restart it without the variable). The session also makes side requests (a
   title); the checker pairs each request with the earlier one it continues (`--link auto`), so they do not hide the
   result.

How long: one run of 2-6 requests is enough for the verdict. Do not leave the capture on for a working day.

### 3.4 Capture steps for GitHub Copilot (VS Code, OpenAI-compatible endpoint)

Copilot's behaviour is unknown (`TTFT_LEVERS.md` 2.3). Use the setup of the owner's existing Copilot test on port 11437
(the custom OpenAI-compatible model entry that points at `http://127.0.0.1:<port>/v1`); this note does not say how to
configure that entry because it was not verified here.

1. Start the server with `STRATA_REQUEST_CAPTURE` as above, on that port.
2. In agent mode, in a scratch workspace, send a prompt that needs one tool call ("Open README.md and tell me its first
   line"). The next request, after the tool result, is the pair to look at. Then send a second prompt in the same chat.
3. Stop the server. Run the check (section 4). If Copilot also issues side requests (title, intent detection) they are
   different conversations and are skipped by `--link auto`; the report lists them as "share no leading message".

## 4. Checking

```
python3 tools/pancake/prefix_check.py /home/dan/clone-of-d/captures/requests-*.jsonl --json /tmp/prefix.json
```

Options: `--link auto|consecutive` (auto pairs each request with the earlier request of its session that shares the most
leading messages, so a title request in between does not matter; `consecutive` compares strictly consecutive requests of a
session, the way a clean two-request capture reads), `--session KEY`, `--snippets N` (N characters of text around a
system-prompt difference; off by default), `--redact OUT` (section 5). Input: the server's capture lines, a JSON/JSONL of
bare request bodies (OpenAI or Anthropic), or API-monitor records. Sessions are told apart by the `session` header value,
else `prompt_cache_key`, else Anthropic `metadata.user_id`, else `user`; with none, all requests are one session.

### 4.1 Reading the report

One block per pair, for example (synthetic data from the tests, not real traffic):

```
request 1 -> 2  [openai; session s1; best prefix]  ->  PREFIX BREAKS
  messages  : earlier 1 + reply -> sent 3; new: tool
  tools     : DIFFERENT (tools_reordered); 4 -> 4 tools; first difference at tool 0
  system    : DIFFERENT system_timestamp (first difference at char 47 of 2,182)
  reasoning : the reply had 860 chars; resent: NO - dropped
  history   : 0 of 1 assistant messages carry reasoning
  first break: tools_reordered at tools  [yes]
  further breaking differences: 2 (kinds: reasoning_dropped, system_timestamp)
  shared prefix: 0 of 3,719 chars = 0.0%  (a perfect client: 76.3%)
```

- **Verdict**: `PREFIX REUSABLE` (the rendered prompt of request N+1 starts with request N plus the reply), `PREFIX BREAKS`
  (a difference that changes the rendered prompt), `UNCLEAR` (a difference that may or may not; see `content_shape`).
- **messages**: the earlier request's message count plus the reply (when the capture has it), against the messages sent.
  `new:` are the roles after the resent reply (a tool result or a user message).
- **tools / system**: identical or not, with the kind. Tools are compared as the template renders them (the function object,
  `json.dumps` in the key order sent).
- **reasoning**: whether the reply's reasoning came back, and how many assistant turns of the resent history carry it
  (counted as the server reads it: `reasoning_content`, or a thinking block).
- **first break**: the first difference, in the order the template renders: tool list, system prompt, then message by message
  (reasoning, text, tool calls, tool results). It gives the message index and role, the field, and the characters
  expected, sent and shared at that point.
- **neutral**: differences the render ignores (listed apart, they do not break the verdict).
- **shared prefix**: characters of the new request that sit before the first break (in render order), against the
  characters of the whole request, and against what a perfect client would give (everything except the new messages).
  Characters, not tokens: a rough guide. Redacted captures know the share to 512 characters.

The JSON has the same per pair, plus up to 12 breaking divergences and the neutral ones, with `chars_expected`,
`chars_sent`, `common_prefix_chars`.

### 4.2 What each kind means and what fixes it

`breaks`: yes = the rendered prompt changes, so the engine's cache is left at that point.

| Kind | Breaks | What it means | Setting that fixes it |
| --- | --- | --- | --- |
| `reasoning_dropped` | yes | The client resent the assistant reply without its reasoning (OpenAI: no `reasoning_content`; Anthropic: no thinking block). The cache holds the thinking tokens; the template renders the resent turn with an empty think block (`preserve_thinking` is on by default, so every assistant turn renders its think block). The prompt leaves the cache at that reply: only that last turn is re-read (a fallback to the turn checkpoint, `TTFT_LEVERS.md` 2.2) [code], not the whole conversation. | `keep_reasoning` (config, or `STRATA_KEEP_REASONING=1`): the server puts its own copy back. Leave `preserve_thinking` at its default: `false` makes the template drop old reasoning from the render at every new user message, which changes the old part of the prompt [code]. |
| `reasoning_other_key` | yes | The client sends the reasoning under `reasoning` / `reasoning_text` / `thinking`; the server reads only `reasoning_content` (`openai_to_messages`), so for the server it is dropped. | `keep_reasoning`; or teach the server the other key (not done). |
| `reasoning_changed` / `reasoning_added` | yes | The reasoning sent is not what the model wrote, or the model wrote none. | none known (`keep_reasoning` fills in only reasoning the client did not send). |
| `tools_reordered` | yes | Same tools, other order. The tool list is rendered before the system prompt, so the cache is left at the very start. | Client: stable order. Server: `_stable_tool_order` puts a repeated set back in the order it first came in, so the loss remains only where the server never saw the first order (e.g. after a restart). This tool shows the client's bytes, not the server's fix. |
| `tools_added` / `tools_removed` / `tools_changed` | yes | The tool set changes between requests (an MCP server connecting, per-request tools). Everything is re-read. | Client: a fixed tool list. No server setting. |
| `tool_schema_changed` / `tool_schema_key_order` | yes | A tool's description or schema differs (the template renders `json.dumps` in the sent key order). | Client: stable descriptions and key order. |
| `system_timestamp` / `system_volatile_id` / `system_changed` | yes | A date/time, an id-like token or other text in the system prompt differs; it starts the prompt, so everything after it is re-read. `system_timestamp` means both changed regions look like a date or a time. | opencode: `env_to_user` (on in `fast-hip`) moves its `<env>` block into the first user message. Claude Code's billing stamp is pinned by the server (`system_billing_stamp`, neutral). Others: remove the date in the client's settings. |
| `text_changed` / `text_inner_whitespace` | yes | A message text (or the reply as resent) differs from before: the client edited it, or changed inner whitespace or line endings. | Client behaviour. |
| `tool_result_changed` / `tool_result_reformatted` | yes | An old tool result differs (pruned, truncated, re-rendered). opencode prunes old tool outputs (`TTFT_LEVERS.md` 2.2). | Client behaviour. |
| `tool_args_key_order` | yes | The tool-call arguments have the same content in another key order; the template renders them in key order. | Client: resend the arguments as received. |
| `tool_args_changed`, `tool_call_name_changed`, `tool_call_added`/`removed`, `tool_result_added`/`removed`, `media_changed`, `role` | yes | The history is not what the client received. | Client behaviour. |
| `history_shorter` | yes | The new request has fewer messages than the earlier one plus its reply: compaction, pruning, or another conversation. | Client behaviour. |
| `not_a_continuation` | yes | The new request has nothing after the earlier request's history: a retry or a different conversation. | none |
| `content_shape` | maybe | Content is a string in one request and a list of parts in the other. The server joins text parts, so probably neutral; confirm with a replay. | none if neutral |
| `text_edge_whitespace`, `reasoning_edge_whitespace`, `tool_result_edge_whitespace`, `system_edge_whitespace` | no | The text differs only at its ends; the template trims every text. | none |
| `tool_args_whitespace` | no | Argument JSON differs only in whitespace; the server parses the arguments. | none |
| `tool_call_id_changed`, `tool_result_id_changed` | no | Ids are not rendered. (`keep_reasoning` finds a turn by its ids first, then by a hash of its text and calls.) | none |
| `system_billing_stamp`, `extra_fields` | no | The server pins the stamp / ignores the field. | none |

`cache_control` markers in Anthropic bodies are ignored (clients move them to the newest block each turn). Everything in
this table is read from the code. The actual loss for a case is whatever `park_divergence` and `ttft_audit.py` report on
the engine side (next section).

## 5. Redaction and sharing

```
python3 tools/pancake/prefix_check.py CAPTURE.jsonl --redact /tmp/shareable.jsonl --no-report
python3 tools/pancake/prefix_check.py /tmp/shareable.jsonl          # same verdicts, same kinds
```

Each text becomes `[[redacted n=<chars> sha=... st=... ws=... pre=...]]`: its length, the hash of the text, of the text with
edge whitespace removed, of the text without any whitespace, of its JSON re-dump (for tool-call arguments), and a hash
for each 512-character prefix. Equal texts give equal markers, so every comparison above still works; what is lost is the
text of a difference (a changed system prompt is `system_changed`, not `system_timestamp`) and the exact length of a shared
prefix inside one text (to 512 characters). Roles, call ids, tool names, numbers and the keys of tool arguments stay: look
at them before sharing. The session header value is replaced by a hash.

## 6. Reading it with the engine logs

- `STRATA_PARK_DIVERGENCE_LOG=1` (M3) prints, per request that did not continue the live conversation, the token where the
  prompt left it (`reply_offset`, `where`, `reasoning_dropped_guess`). A `reasoning_dropped` here should show as a
  divergence inside the previous reply (`reply-start`) in the same request; a `system_timestamp` or `tools_reordered` as a
  divergence deep inside the history (`where: history`, a small `diverge_at`). Agreement between the two logs on one
  session would confirm the code-reading in section 4.2; disagreement would show a case this tool misses (for example
  something the server's own fixes change).
- `tools/pancake/ttft_audit.py` classes every request of the turn log as live / fallback / switch / lost / cold. Match the
  capture to it by time and request order.

## 7. Limits and unconfirmed points

- Nothing here was run on traffic from a real client; the tests are synthetic. Which clients really break the prefix is
  what the capture is for.
- The render rules (every text trimmed, tools rendered with `json.dumps` in sent order before the system prompt, only
  `reasoning_content` / thinking blocks read as reasoning, ids not rendered, tool arguments parsed as JSON) are read from
  `serve/chat_template.jinja` and `serve/frontend.py`. The tool compares the client's bytes; it does not apply the
  server's fixes (`_stable_tool_order`, `pin_billing_stamp`, `env_to_user`, `_late_system_to_user`), so it can report a
  break the server already repairs. It does not render the template, so a case where two different messages render to the
  same prompt is called a difference.
- The comparison is on characters, not tokens, and on the history. It cannot see the engine's state: a request that is a
  perfect continuation in text can still miss the cache (budget, eviction, side requests between turns; lever 1 of
  `TTFT_LEVERS.md`). The share is only a guide.
- `content_shape` (string vs list of parts) is a guess of "probably neutral"; `str` -> `[text part]` is the case.
- A capture taken while a client also sends side requests needs `--link auto`; if the FIRST message differs on every turn,
  auto-link finds no parent and says so (use `--link consecutive`).
- The streamed argument text is exactly what the client received, but a client may re-serialise it; the capture shows what
  it sent back, which is the point.

## 8. Files

- `tools/pancake/prefix_check.py`: the checker and redaction (stdlib only); `tools/pancake/test_prefix_check.py` (19 tests).
- `serve/request_capture.py`: the opt-in capture; hooks in `serve/server.py` (`_post`, `_capture`/`_tee_capture`, `_sse`,
  `_json`, start-up line); `serve/test_request_capture.py` (9 tests, mock engine, no GPU).
- Run the tests: `nice -n 19 python3 -I -m pytest -q tools/pancake/test_prefix_check.py serve/test_request_capture.py`.
