# pancake server / agent layer: deep code study (2026-11-09)

Repo: /home/dan/clone-of-d/src/Strata (branch port/rdna3-decode). Read-only study (Serena + grep); nothing run.
All line numbers are 1-based, serve/server.py unless another file is named.
**Legend:** [code] = read in the code (certain). [est.] = estimate, not measured. [meas.] = a number from MOD_PLAN/FINDINGS, measured earlier by someone else.
Nothing in this file was benchmarked in this session (the GPU box was busy).

---

## 0. Headline findings (short)

1. **The daily preset sets none of the server-side agent settings.** configs/pancake/flashnext-iq2_xs-fast-hip.json has only engine args and env. So the defaults apply: `reasoning_budget_by_effort` {} , `reasoning_budget_grace` 0, `reasoning_close_retry` off, `effort_position` "start", `env_to_user` off, loop guard ON, `answer_room` ON. [code]
2. **opencode always gets an implicit thinking cap of 24 000 tokens.** opencode sends `max_tokens` 32000 (its output limit is 32768, capped by OUTPUT_TOKEN_MAX) and `reasoningEffort: xhigh`. With no budget configured, `answer_room` (6520-6528) sets budget = max_new − max(512, max_new//4) = 24 000. That is item 12's "cap", only at ~24K instead of a p99 value. It also prints one log line on every request. [code + opencode config]
3. **reasoning_tokens is missing from chat usage** even though `_run` already counts it. `done["reasoning_tokens"]` exists (7310), but `openai_chunks` builds usage at 7767-7770 without `completion_tokens_details`. The fix is one line (item 73). [code]
4. **Switching an opencode variant (xhigh / medium / low) re-reads the whole prompt.** The Qwen3.8 template writes the effort sentence at the top of the system turn, before the tools. `effort_position: "end"` (#458, `effort_end_args` 3729) fixes this, but the preset does not set it. When it is on, `render_prompt` renders the template 3 times for non-default efforts (5952-5967). [code]
5. **The `<env>` block in opencode's system prompt (cwd, date) breaks the shared prefix across days and folders.** The fix `env_to_user` (frontend.py:405-438; the comment there says "~14K tokens, 8-11 s") exists but is off and not in the preset. [code; the 8-11 s figure comes from the code comment]
6. **Grammar-constrained tool calls apply only to /v1/chat/completions.** `req["strata_grammar"]` is set only in `_openai` (9174). `_anthropic` (9438) and `_responses_prepare` (9369) never set it, so Claude-Code-style clients and Codex lose the phase-6 guarantee. This is a real bug that comes from the three duplicated API paths. [code]
7. **The pin=N shared-prefix primitive (item 71's store) already exists in the main engine** (generate.cpp:15881, `strata_prefix` → `resolve_prefix` 6247). But the server auto-pins only for DeepSeek (`deepseek_shared_start`, 6229; the gate is in prepare ~6180). Also, the engine's `pin_sibling` rule (generate.cpp:16463) does **not park** the branch it leaves when a request resumes exactly at the pin. The comment there assumes "a few dozen tokens" of private suffix. For opencode sub-agents that branch is the parent's whole conversation, so turning auto-pin on for Qwen as-is would make every return to the parent a full re-read. The rule needs a size threshold first. [code]
8. **The loop guard's hold-back adds no per-turn wall time.** It holds up to HOLD_MAX=64 tokens (~0.5 s at 125 tok/s), and the held tail is released at the stop token (6918-6924). Tool calls are executed by opencode only after the stream ends anyway. It only makes streaming less smooth, most of all for edit-tool `newString` that copies `oldString` (a long period-p run). The exception: in a conversation where the guard has blocked calls, tool-call events are held until the answer ends (`pending_evs`, 6815). [code]
9. **There is no per-turn log on disk.** The only per-request records are `self.history` (in memory, 7193), `api_requests` (only with api_monitor, which json.dumps the whole body: 5763), stdout prints, and ConvCacheLog parsing of the engine log. Item 73's JSONL has to be added. [code]
10. **capped.sh cannot be used as a systemd service's ExecStart.** It `exec systemd-run --user --scope`, which moves the engine out of the service's cgroup. Then Conflicts= and stop no longer reach the engine, and Restart= sees an empty cgroup. The unit must carry MemoryMax itself (see the item 73 sketch). [code + systemd semantics]
11. **The strata-coder agent points at `strata/qwen3.6-35b-a3b`** (context 163840 in opencode.jsonc). Pancake serves `qwen3.8-flash-next-iq2_xs`, context 131072. Wiring = change the agent model, or add an alias. [config]
12. **preserve_thinking cannot be set by a client.** `openai_to_messages` (frontend.py:605-614) forwards only `enable_thinking` and `reasoning_effort` from chat_template_kwargs. The template defaults `preserve_thinking` to true, so earlier `reasoning_content` is rendered if the client sends it. Whether opencode resends it decides item 31 (see §2). [code]

---

## 1. Request path of one opencode turn (HTTP → engine → back)

An opencode "turn" from the user's view is N HTTP requests (one per model step; every tool result is a new POST with the whole conversation). Each request:

| # | Step | Where | Latency notes |
|---|------|-------|---------------|
| 1 | ThreadingHTTPServer thread, `do_POST` → `_post` | 8735, 8741 | auth, path aliases, `_foreign_page` |
| 2 | `svc.reserve_inference()` admission | 8783, 5048 | lock only |
| 3 | body read + `json.loads` | 8797 (`_body` 8151) | 0.3-1.5 MB at 30-100K-token prompts: ~5-15 ms [est.] |
| 4 | `begin_request` | 8843 → 5759 | returns None unless api_monitor; with it on, it does `json.dumps(indent=2)` of the whole body (~10-30 ms [est.]) |
| 5 | `_openai`: `with_shared`, logit_bias normalize | 9161-9163 | trivial |
| 6 | `openai_to_messages` (+ `_late_system_to_user`, `_env_to_user` if on, `_order_chat_tool_results`, `_stable_tool_order` via `_tool_list`) | frontend.py:577 | O(messages); ~1-5 ms [est.] |
| 7 | `grammar_api.grammar_spec(...)` | 9174 | builds the tool grammar spec from the tools each request (cached? not checked) |
| 8 | `svc.load()` | 9177 | no-op when loaded |
| 9 | validations: `refuse_for_split`, logprobs, `bias_api.validate_request`, `reasoning_budget(req)` (also called again in `_run`), `stop_strings`, `check_request_sampling` | 9183-9213 | trivial |
| 10 | `_watch_client(cancel)` thread | 8249 | one thread per request |
| 11 | **`svc.prepare`** | 6040-6196 | see 11a-11g |
| 11a | tool-image reference check over all parts | 6047-6063 | O(parts) |
| 11b | **loop guard across turns** `_guarded_messages` → `loop_guard.tool_blocked`, `echoes`, `tool_repeat`, `collapse` | 6474-6493, loop_guard.py:405-650 | regex/gist scans over all messages (COLLAPSE_MAX_CHARS=32000 bounds part of it): ~2-20 ms [est.]. **Note: `collapse` rewrites earlier history, so while it is active the prompt start diverges and the engine re-reads from there.** |
| 11c | `_note_unreadable_tool_images` | 6000 | O(messages) |
| 11d | **`encode_prompt`**: `mark_think_literals` (deep walk), `render_prompt` (sandboxed Jinja; 1 render, or 3 with effort_end and a non-default effort), `unmark_think_literals` | 5970-5983, 5952-5967 | Jinja sandbox over 50-300 messages + ~20 tool schemas through `tojson`: ~20-80 ms per render [est.]; ×3 on the effort_end path |
| 11e | **tokenize**: `PromptEncoder.encode` (#567) reuses ids up to the last special token shared with a kept prompt, then pure-Python regex+BPE on the rest | tools/strata_tokenizer.py:396-460 | prefix compare on ~400K chars is cheap. The new suffix (a tool result of 1-20K tokens) goes through pure-Python BPE: ~10-150 ms [est.]. Its docstring: a full encode costs "hundreds of milliseconds at 100K tokens" |
| 11f | context-room checks, `fit_max_tokens` shortening | 6143-6170 | trivial |
| 11g | `strata_prefix`/auto-pin (DS41 only), `set_restore_points` (DS41 only), `classify_domain` (only with STRATA_DOMAIN_PROFILES) | 6180-6196 | `resolve_prefix` renders + encodes again when used |
| 12 | `openai_chunks` generator; first chunk (role) yielded before the engine is touched | 7703-7716 | the SSE headers go out right away (`_sse`, 9226) |
| 13 | `svc.run` → `_run`: sampling merge (model card, config, shared, request) | 6496-6517 | trivial |
| 14 | thinking budget: `reasoning_budget` (by_effort / request / config), else `answer_room` (24 000 for opencode), clamp to max_new−reserve | 6519-6545, 5223-5248 | prints a line every request when max_new ≤ 32768 |
| 15 | terminal_requests bookkeeping, **FIFO wait** (single turn; ping every QUEUE_BEAT_S=10 s) | 6600-6653 | zero with one client. A sub-agent running in parallel with its parent queues here |
| 16 | `ensure_loaded` (restarts a dead engine) | 6669 | normally a no-op |
| 17 | `ToolSafeGuard` set up (loop guard on by default), grace computed | 6697-6705 | — |
| 18 | **`engine.generate(prompt, ...)`**: builds `GEN <n> <sampling keys> <ids comma-joined>` and writes it to the engine's stdin | 6898 → StrataEngine.generate 2835-3008 (head/write ~2870-2880) | 100K ids ≈ 0.7 MB string: ~10-30 ms to build + write + engine parse [est.]. The engine finds its own prefix match (conversation cache / turn checkpoint at the last `<|im_start|>`); the server does not tell it more on Qwen (marks/tail/shared only for DS41) |
| 19 | engine prefill (PP lines → `ping`) | engine | **the dominant cost on re-reads** (MOD_PLAN: re-reads ~18% of turn time [meas.]) |
| 20 | **token loop**: for each id: stop check, `raw_ids`/`seg`, guard `push` (hold ≤ 64), `release` → detok, `parser.feed`, `cut` (stop strings), `_note` (status lock + 600-char tail concat), `_progress`, budget/force/loop-recovery checks, yield | 6915-6956, `release` 6788-6851 | Python per token ~50-300 µs [est.]; the engine writes ~8 ms/token in bursts of ≤5 (spec 4), so Python keeps up. Not on the critical path, but the GIL is shared with the pump thread. `reasoning_repeat_coverage` every 512 tokens only with reasoning_loop_recovery (off) |
| 21 | per chunk: `json.dumps` + `wfile.write` + `flush` | 9227-9233 | one syscall per token; fine |
| 22 | budget wrap (if hit): `gen.close()` (STOP + drain to DONE), then a new GEN with prompt+seg+REASONING_WRAP_UP | 7120-7150 | one round trip + ~20-token prefill from the live state: ~50-200 ms [est.] |
| 23 | end: `parser.finish`, held-back calls flushed, `history.append` with timings, status reset (inside the fifo), then `done` | 7175-7316 | under the fifo: the next queued request waits for this bookkeeping (small) |
| 24 | final chunk with `finish_reason` + `usage` + `timings` | 7760-7774 | **no completion_tokens_details.reasoning_tokens** |

**Server-side overhead per request, total:** ~60-300 ms [est.] (parse + guard scans + render + encode + GEN write), against a median opencode turn of 6.2 s and p90 50 s [meas., old backend]. So about 1-5% per request. It grows with conversation length (render is O(history), not incremental) and with effort_end ×3. **This should be measured first with the per-turn log (t_parse, t_render, t_encode fields), not guessed.**

The big per-turn items are not in Python. They are (a) re-reads the engine has to do because the prompt start changed: effort switch, `<env>` block, sub-agent switch, loop-guard collapse, dropped thinking; and (b) thinking length.

---

## 2. Conversation cache interplay (what makes the engine re-read)

| Cause | Mechanism | Fix in code | State |
|---|---|---|---|
| reasoning effort changed (opencode variants) | effort sentence at the top of the system turn | `effort_position: "end"` → `--tail-role-token`, `render_prompt` effort tail | exists, not in preset |
| new day / new cwd | `<env>` block in opencode's system prompt | `env_to_user: true` | exists, off |
| tool order changes (MCP registration order) | tools render before the system prompt | `_stable_tool_order` (frontend.py:470) | on (per process; lost on restart: not persisted) |
| late system messages | template refuses mid-conversation system messages | `_late_system_to_user` | on |
| dropped thinking (item 31) | template keeps `reasoning_content` (preserve_thinking default true); if opencode does not resend it, the earlier assistant turn renders as `<think>\n\n</think>` while the engine holds the real thinking tokens | engine restores at its turn checkpoint (last `<|im_start|>` of the previous prompt) and re-reads the previous assistant message + tool result | **unknown: needs one logged request.** If reasoning is dropped, the loss is the restore plus re-reading the assistant's content/tool-call text (hundreds of tokens), not the whole history. If reasoning IS resent, check the rendering matches byte for byte (template `|trim` vs model output) |
| empty assistant turns | `ChatTemplate.render` drops assistant messages with no text, no image, no calls (#843), **including ones that carried only reasoning_content** | — | rare edge: changes history vs what the engine generated |
| loop-guard collapse | `_collapse` removes earlier model text from the prompt | — | only in looping conversations; costs a deep re-read every turn while active |
| sub-agent switch | different system prompt + tool set: no common prefix past `<|im_start|>system` | items 71 / 39 | open |
| loop-guard tool note | `_tool_repeat` appends a user message at the end only | — | cheap (tail only), by design |

PromptEncoder keeps `conversation_slots(engine)` prompts (default 4). A sub-agent plus its parent plus compaction fits.

---

## 3. Ranked opportunities

Gain columns: share of agent wall time unless stated. "exact" = same tokens as today for the same prompt.

### Rank 1: Item 73 wiring (prerequisite; 0 gain itself, unlocks everything)

**a) reasoning_tokens in chat usage**. `openai_chunks`, server.py:7767-7770:
```python
last["usage"] = {"prompt_tokens": pt, "completion_tokens": x["completion_tokens"],
                 "total_tokens": pt + x["completion_tokens"],
                 "prompt_tokens_details": {"cached_tokens": x.get("reused") or 0},
                 "completion_tokens_details": {"reasoning_tokens": x.get("reasoning_tokens", 0)}}
```
`openai_collect` (7797) copies `last["usage"]`, so non-stream gets it too. The ai-sdk openai-compatible provider maps `completion_tokens_details.reasoning_tokens` → `reasoningTokens`, so opencode's session stats show it. Risk: none (an extra field). Add a test next to test_responses' usage test. Note: with MCP rounds (`run_with_mcp`) check that the last round's `reasoning_tokens` is the sum.

**b) systemd unit** `~/.config/systemd/user/pancake.service`. Sketch; do not use capped.sh inside it:
```ini
[Unit]
Description=pancake-strata LLM server (port 11437, Qwen3.8-Flash-Next IQ2_XS fast-hip)
After=network-online.target
# one server holds the 7900 XT and port 11437; Conflicts= works both ways
Conflicts=strata.service avalon.service qwen35b.service

[Service]
Type=simple
WorkingDirectory=/home/dan/clone-of-d/src/Strata
Environment=PORT=11437
Environment=HIP_VISIBLE_DEVICES=0
Environment=STRATA_RAM_KEEP_GIB=20
# per-turn log (item 73), on D
Environment=STRATA_TURN_LOG=/home/dan/clone-of-d/pancake-logs/turns.jsonl
# ntfs3 partition at /home/dan/clone-of-d (fstab nofail, 5 s device timeout): a user unit cannot order itself after a
# system mount, so wait for the file (same pattern as strata.service.d/wait-model.conf) and check it is ntfs3, not the
# empty mount point
ExecStartPre=/bin/sh -c 'for i in $(seq 90); do findmnt -n -t ntfs3 /home/dan/clone-of-d >/dev/null && [ -f /home/dan/clone-of-d/Strata-data/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf ] && exit 0; sleep 2; done; echo "D (ntfs3) not mounted" >&2; exit 1'
# capped.sh's RAM wait, without its systemd-run (a scope would move the engine out of this unit's cgroup)
ExecStartPre=/bin/sh -c 'for i in $(seq 120); do [ $(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo) -ge 40 ] && exit 0; sleep 5; done; exit 0'
ExecStartPre=/bin/mkdir -p /home/dan/clone-of-d/pancake-logs
# run-pancake.sh minus its last line: generate the config, then run server.py directly (a NOCAP=1 switch in run-pancake.sh
# that execs python instead of capped.sh is the smallest change)
ExecStart=/usr/bin/env NOCAP=1 /home/dan/clone-of-d/src/Strata/run-pancake.sh iq2_xs fast
MemoryMax=54G
MemorySwapMax=0
ManagedOOMMemoryPressure=kill
ManagedOOMMemoryPressureLimit=80%
OOMPolicy=stop
Restart=on-failure
RestartSec=20
TimeoutStartSec=15min
TimeoutStopSec=60
KillMode=mixed
StandardOutput=append:/home/dan/clone-of-d/pancake-logs/pancake-service.log
StandardError=append:/home/dan/clone-of-d/pancake-logs/pancake-service.log

[Install]
WantedBy=default.target
```
Also: (1) give the unit its own ManagedOOM limit so user@.service's 50%/30 s oomd rule does not kill the arena build (MOD_PLAN "next session" item 1). (2) `run-pancake.sh` needs `NOCAP` support. (3) opencode: `agent.strata-coder.model` → `strata/qwen3.8-flash-next-iq2_xs` (context 131072 is already in its model entry), or add `"aliases": ["qwen3.6-35b-a3b"]` to the preset. (4) Disable strata.service so it does not start at boot and take the port first.

**c) per-turn JSONL log**. Write one line per `_run` end, appended by a background writer thread (a queue; ntfs3 write latency must not sit under the fifo). Hook: next to `self.history.append` (7193). Prepare-side timings go in `req["_turn"]` (the underscore keys survive the sampling merge at 6515-6517). Fields:
- identity: `ts`, `request_id`, `api` (openai/anthropic/responses), `model`, `session` (hash of system text + first user message), `agent` (hash of system text only: tells sub-agents apart), `turn_index` (count of assistant messages), `stream`
- request shape: `n_messages`, `n_tools`, `tools_hash`, `last_role`, `last_tool_names` (the tool results in this request), `effort`, `thinking`, `max_tokens`, `asst_with_reasoning` / `asst_total` (**answers item 31**), `env_block_in_system` (bool: what env_to_user would move)
- prompt/cache: `prompt_tokens`, `reused` (cache_n), `encoder_reused_ids` (PromptEncoder hit), `prev_prompt_lcp` (common token prefix with the same session's previous prompt: the ideal reuse; `prev_prompt_lcp − reused` = what the engine re-read that it need not have), `conv_cache` (parked/restored from ConvCacheLog)
- thinking: `budget`, `budget_source` (request / by_effort / config / answer_room), `thought`, `reasoning_tokens`, `wrapped`, `grace_used`, `close_retried`, `loop_guard` (guard.summary()), `rumination_5gram` (item 34, log-only)
- output: `completion_tokens`, `finish`, `tool_calls` (names), `rejected`, `args_repaired`, `stray_think`
- timing: `t_parse`, `t_guard`, `t_render`, `t_encode`, `t_queue`, `t_first_token`, `prompt_ms`/`prompt_tok_s` and decode tok/s (engine DONE `timings`), `t_total`, `hit_rate`, `pcie_share`
- item 33 counters computed from messages (log-only): `edits_since_test`, `reads_since_edit`, `same_call_repeats`

Size ~1 KB per request; D is 91% full, so rotate at 50 MB. Gain 0; it unblocks 12 (p99), 21, 31, 33/34, 66, 71 and the overhead measurement in §1.

### Rank 2: config-only wins (no code, exact or near-exact; put them in the preset)
| Change | Function | Gain | Risk |
|---|---|---|---|
| `"effort_position": "end"` | `effort_end_args` 3729 + `render_prompt` | a variant switch no longer re-reads the ~15-25K-token system+tools prefix (~8-15 s each switch [est.]) | changes prompt text for non-default efforts (a system turn at the end): quality A/B through item 67; costs 3 renders on those requests |
| `"env_to_user": true` | frontend `_env_to_user` | new day/folder: ~14K tokens, 8-11 s not re-read (code comment; [meas. by upstream?]) | the model sees env one turn later; opencode-specific regex |
| item 12: `"reasoning_budget_by_effort": {"xhigh": P99, "medium": .., "low": ..}`, `"reasoning_budget_grace": 512`, `"reasoning_close_retry": true` | `reasoning_budget` 5223, `_run` grace 6703 | worst turn 84-340 s → ~1-2 min, ~6.6% of turn time [MOD_PLAN est.]; today's implicit cap is 24 000 | text changes on the tail only; p99 needs the 73 log. Setting by_effort switches off answer_room for that level (by design); keep a cap ≤ max_new−8000 |

### Rank 3: Item 66 agent-turn pipeline, staged (15-25% of agent wall time [MOD_PLAN est.])
Concrete server stages, in order:
1. **35 restore-equivalence** (GPU test, first PR of 66): needed before any checkpoint change.
2. **31**: from the 73 log, `asst_with_reasoning`. If opencode drops reasoning: an opt-in `preserve_thinking` config that (a) forwards `chat_template_kwargs.preserve_thinking` (today dropped at frontend.py:605-614), and (b) keeps the server's own copy of each answer's reasoning keyed by the tool-call id / content hash and re-inserts it into the history. That makes turn n+1's prompt continue the engine's live state token for token. Exact (same as if the client had sent it). Gain: restore + re-read of the last assistant message per request, ~0.2-1 s [est.]. If opencode does send it: verify the rendering is byte-identical (trim, `\n</think>\n\n`).
3. **16/11 restore marks for Qwen**: `set_restore_points` (6211) and `restore_keys` (1815) already plumb `marks=/tail=/shared=` for DS41 only. Generalize: marks at each `<|im_start|>user\n<tool_response>` and after each `</tool_call>` of the prompt (the tool boundaries). This needs the main engine (generate.cpp) to accept `marks=` (today only `--prompt-cache-tail` and the turn token). This is the "turn plan" in its smallest form.
4. **12** caps (Rank 2) and **17** pre-stage hint as more keys on the same GEN line.

### Rank 4: Item 71 shared-prefix store for sub-agents (up to ~7% of turn time [MOD_PLAN est.])
- The primitive exists: `strata_prefix` → `pin=N` (resolve_prefix 6247; engine generate.cpp:15881). Auto-pin exists for DS41 (`deepseek_shared_start` 6229).
- **Blocking issue:** `pin_sibling` (generate.cpp:16463-16470) skips parking the branch it leaves. For sub-agents that branch is the parent conversation (19-41K tokens). Engine change: set pin_sibling only when the leaving branch's suffix past the pin is < ~2K tokens. Then enable auto-pin for Qwen: in `prepare` (~6180), drop the `family == "deepseek41"` gate behind `STRATA_AUTO_PIN=1`.
- The pin only helps if agents share the prefix: today the system turn = [effort line] + tools + system text. Tools differ per agent. The server-side reorder (put the common tool schemas first in a stable order, agent-specific ones after) has to be gated by item 67 (behaviour change). Measure first: common-prefix length across the agents' first requests (the 73 log's `agent` + `tools_hash` give it).
- Gain [est.]: a switch to a known agent reads its suffix instead of 19-41K tokens: ~10-25 s per switch at ~1.5-2.5K tok/s prefill.

### Rank 5: Item 21 learned thinking stop / 34 rumination (−20-40% thinking on routine turns [MOD_PLAN est.])
- Hook point: `release()` (6788) already has `think_tail`, `thought`, and a clean-boundary check (`parser.clean and not detok.pending()`). Item 34: compute a rolling 5-gram repetition rate over `reasoning_text` (already collected when reasoning_loop_recovery is on; keep a bounded deque instead) and log it per turn first (73). Then let it lower the effective budget (soft close via the same `wrap=True` path, 6843-6848). Item 21 needs hidden states from the engine (an I2 hook), so it is not server-only.
- Risk: lossy (text). Gate with item 67.

### Rank 6: Items 33 behaviour counters (unmeasured; failed runs cost 1.4-4.4x tokens)
- Pure server, log-only first: in `_guarded_messages` (6474) the messages are already scanned per turn. Add counters (edits without build/test, reads without edit) to `req["_turn"]` → JSONL. Later, a note like `tool_note`. Exact (log), cheap.

### Rank 7: Item 26 (parked)
- Grammar already constrains tool-call JSON when `--grammar on` (chat only, see finding 6). An oldString must-exist check needs file access the server does not have; it stays parked. **Do fix the grammar parity first** (NEW-1 below).

### NEW items
| ID | Change | Function | Gain | Risk |
|---|---|---|---|---|
| NEW-1 grammar parity | set `req["strata_grammar"]` in `_anthropic` and `_responses_prepare` exactly as `_openai` 9174 (best: one shared `_prepare_generation()`, see §4) | 9438, 9369 | tool calls always valid for Claude-Code/Codex clients (phase 6's guarantee) | low; same engine path |
| NEW-2 stage timers | `time.perf_counter()` around parse / guard / render / encode / GEN-write / first token into `req["_turn"]` | `_post`, `prepare`, `_run` | turns §1's estimates into measurements | none |
| NEW-3 incremental render | cache the rendered text per (session, message index) where the template is append-only: render only messages past the last cached boundary and check `full.startswith(cached)` once per N requests (or keep the full render as a debug cross-check) | `encode_prompt` 5970 | 20-80 ms ×(1-3) per request [est.] → ~5 ms | the template's `last_query_index` logic makes old turns depend on later ones (preserve_thinking false, multi_step_tool) → fall back to a full render when kwargs/tool set change; needs a golden test (chat_golden.json exists) |
| NEW-4 effort_end single render | render `history` once and derive `mine` from it: `mine` = history with the effort kwargs. Today 3 renders | `render_prompt` 5952 | 2 renders saved on non-default effort requests | low |
| NEW-5 answer_room log spam + meaning | print the answer_room line once per value, not per request; count it in `budget_source` | 6527 | log hygiene | none |
| NEW-6 persist `_TOOL_ORDERS` | write the first-seen tool orders to `<data>/tool-orders.json` so a server restart does not lose the order (a restart then a reordered tool list re-reads the whole prefix) | frontend `_stable_tool_order` 470 | avoids one full re-read after restarts | stale orders: keep LRU 64 |
| NEW-7 api_monitor cost | `begin_request` dumps the whole body with indent=2 under the GIL; dump lazily (on view) or without indent | 5763 | 10-30 ms per request when the monitor is on [est.] | none |
| NEW-8 empty-assistant filter keeps reasoning | in `ChatTemplate.render` keep an assistant message that has `reasoning_content` (when preserve_thinking applies) | frontend.py ChatTemplate.render | exact prefix match for thinking-only replies | template-specific; test |
| NEW-9 BPE suffix in native code | the tokenizer is pure-Python `regex` + BPE; move `_encode_matching`/`_bpe` behind a C extension or the engine's tokenizer (TOKENIZE line) | tools/strata_tokenizer.py | 10-150 ms per big tool result [est.] | ids must be identical (test_prompt_encoder covers it) |
| NEW-10 grammar_spec cache | memoize `grammar_api.grammar_spec` by (tools hash, tool_choice, response_format) | 9174 | small [est.]; not checked how heavy it is | none |

---

## 4. Structure and maintainability (server.py, 11 138 lines)

**Measured sizes (line counts, [code]):** Service 2 443 lines (4878-7320), of which `_run` is ~810 (6508-7317) with nested closures `server_turn`, `flush_pending`, `discard`, `release`, `guard_defers` sharing ~25 `nonlocal`s. StrataEngine 2 066 (1173-3238). `make_handler` 1 414 (8080-9493: one closure class with all routes). `_main` 650. frontend.OutputParser 908. Service.__init__ has ~95 attributes. There are 98 module constants plus ~40 `os.environ` reads at import time (COALESCE_*, ADMIT_*), and config keys spread across `_main` (10862-10975+), `*_setting()` helpers, and per-request overrides.

**Problems**
1. **God object `Service`:** admission/queue, model load/unload/VRAM, vision lifecycle, metrics/prometheus, prompt rendering/encoding, the loop guard across turns, the generation loop, history/terminal events. Every feature lands in it; `_run` is the hardest place to change safely, and it is exactly where items 12/21/34/66 must go.
2. **Three API paths duplicate the pre-generation steps** (`_openai` 9161, `_anthropic` 9438, `_responses_prepare` 9369): parse → with_shared → messages → force → validators → watch → prepare. They have already drifted: grammar only in chat (finding 6), and `svc.load()` before vs after `with_shared`. Every new per-request feature (the turn log, turn plan, preserve_thinking) would otherwise need three copies.
3. **Config sprawl:** settings come from the JSON config, CLI args, env vars at import, `/settings` shared state, and the request, in four merge orders: `sampling_defaults`/`shared`/model card in `_run` 6515, `with_shared`, `effort_default`, and request fields. No single schema: the preset silently lacked every agent setting and nothing warned.
4. **Request-scoped state travels as underscore keys in the request dict** (`_loop_blocked`, `_loop_keep`, `_restore`, `_reply_from`, `strata_img_keys`) and as `setattr(cancel, ...)`. It works, but nothing is typed.
5. **Bookkeeping under the fifo** (history, stats) and `print` as the log: fine at one client, but it is noise for parsing.

**Pragmatic refactor (does not block current work; each step is one PR with the existing 140+ tests as the net)**
1. **`serve/turnlog.py`** (new, item 73): `TurnRecord` dataclass + a background JSONL writer. `_run` fills it. This is the first "typed request state" and costs nothing to the rest.
2. **`GenerationRequest` dataclass** (new `serve/request.py`): messages, tools, kwargs, force, max_new, sampling, cancel, grammar, budget, turn record, plus the underscore keys as fields. One function `build_generation(svc, api, req) -> GenerationRequest` holds the common steps (validators, grammar_spec, reasoning_budget, watch, prepare). The three handlers keep only their format conversion (`openai_to_messages` / `anthropic_to_messages` / `responses_api`) and output encoding. This fixes NEW-1 by construction.
3. **Split `_run` without changing behaviour**: move the closures into a `TurnLoop` class (serve/turn_loop.py) whose `nonlocal`s become attributes. Methods: `budget_plan()` (6519-6545), `queue()` (6600-6653), `one_pass()`, `release()`, `server_turn()`, `wrap_or_retry()` (7091-7150), `account()` (7175-7255). `Service._run` becomes `yield from TurnLoop(self, req).run()`. Do it mechanically, behind the existing tests (test_stop_holdback, test_loop_guard, test_budget_by_effort, test_reasoning_*). Items 12/21/34/66 then change one method each.
4. **`Service` split by responsibility, gradually:** `ModelLifecycle` (load/unload/vram/vision: ~5264-5666), `Metrics` (5722-5949), `PromptBuilder` (render/encode/prepare/resolve_prefix/loop-guard-across-turns: 5951-6493). Keep `Service` as a façade with the same attribute names (properties) so the handler and tests do not change in the same PR.
5. **One config schema** (`serve/runconfig.py` already exists, 191 lines): declare every server key with type, default and doc. `_main` validates against it and prints the effective agent settings at start ("thinking caps: none; effort_position: start; env_to_user: off"). That would have shown finding 1 at the first run.
6. Leave StrataEngine (the engine protocol) and OutputParser alone for now: they are large but cohesive and well tested.

Order relative to the plan: 1 goes with 73 (now). 2 goes with NEW-1/31 (next). 3 goes before 66/21 code. 4-5 whenever convenient.

---

## 5. What to measure first (once the 73 log runs)
1. `t_render + t_encode + t_parse` per request vs `t_first_token`: confirms or refutes §1's 1-5%.
2. `asst_with_reasoning / asst_total`: decides item 31's form.
3. `prev_prompt_lcp − reused` per cause (effort change, env, agent switch, collapse): sizes S5 per cause.
4. The p99 of `reasoning_tokens` by effort: the values for `reasoning_budget_by_effort` (item 12).
5. The common-prefix length across agents' first requests: item 71's ceiling.
