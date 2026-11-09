# Upstream llama.cpp server / agent-facing feature work relevant to Avalon (as of 2026-11-08)

Method: `gh pr list/view` and `gh issue list/view` against ggml-org/llama.cpp (searches on server, tool, reasoning, cache, slot, responses, anthropic, router, parser, qwen; created/merged since ~2026-08), plus the local Avalon checkout (/home/dan/clone-of-d/Avalon, last commit c24c03d, 2026-11-02). GPU kernels and speculative decoding excluded per scope. All PR/issue facts below come from the PR/issue page bodies and metadata; "OPEN" = not merged as of 2026-11-08. Nearly every open PR listed is `REVIEW_REQUIRED` (no maintainer approval yet), and most disclose AI-assisted authorship.

Avalon baseline (inference from local checkout): Avalon's server already has `--cache-ram`, `--cache-idle-slots`, `--cache-reuse`, `--ctx-checkpoints`, `--kv-unified`, `--slot-save-path`, `--sleep-idle-seconds`, `--reasoning-budget`/`--reasoning-budget-message`, router (`--models-preset`/`--models-dir`), built-in `--tools` + MCP stdio tools, `/v1/responses` and `/v1/messages`. It contains #28554 (id_slot on /v1/messages, merged 2026-10-20) but not #30856 (merged 2026-10-21), #31256 (2026-10-27), /v1/realtime (2026-10-28) or #30137 (2026-11-05). So Avalon tracks upstream to roughly 2026-10-20; everything merged after that is a candidate cherry-pick.

## 1. Tool calling / structured output

### Takeaway
The most relevant work for a Qwen-family agent is the cluster of open PRs that make tool-argument grammars accept parameters/JSON keys in any order (fixes dropped/garbage args when the model reorders), plus named `tool_choice` support and `--json-schema` reasoning-aware grammar. None is merged yet; #30193 and #31040 are the most active.

### Cited Findings
- **#30193 "chat : accept JSON tool argument keys in any order for Qwen3-Coder/Nemotron"** (OPEN, created 2026-10-08, updated 2026-11-08, +512/-105, 10 comments). Makes every grammar generated from a JSON schema accept object keys in any order; keys tracked as a set within a 1024-state budget per object (≤10 properties exact-once), grammar-wide budget 4096. Motivation: declared-order enforcement causes "garbage keys, dropped fields, and tool call failures" with Qwens/Nemotrons. Usefulness: HIGH for opencode tool calls if Avalon's Qwen3.8 template uses JSON args. — [PR #30193](https://github.com/ggml-org/llama.cpp/pull/30193)
- **#31040 "common/autoparser : exact bitmask permutation for mixed required/optional tagged tool args"** (OPEN, 2026-10-23, upd 2026-11-06, +215/-24, 5 comments). Fixes rigid definition-order enforcement for XML-tagged args (`<parameter=name>` in Qwen3.5/3.6, also Bailing/Ling/Flint) in `build_tool_parser_tag_tagged`. Usefulness: HIGH if Qwen3.8 uses Qwen3-Coder-style XML tool calls. — [PR #31040](https://github.com/ggml-org/llama.cpp/pull/31040)
- **#30382 "chat : accept Qwen tool arguments in any order"** (OPEN, 2026-10-12, 4 comments, +82/-6). Notes the Qwen parser only accepts optional args after required ones while Qwen's own vLLM `qwen3coder_tool_parser.py` accepts any order. Overlaps #31040/#26472. — [PR #30382](https://github.com/ggml-org/llama.cpp/pull/30382)
- Older overlapping attempts: #26472 "enforce required tagged tool args in any order via permute()" (OPEN, 2026-08-02) — [PR #26472](https://github.com/ggml-org/llama.cpp/pull/26472); #26833 "fix qwen tool parser anyOf object args" (draft, 2026-08-10) — [PR #26833](https://github.com/ggml-org/llama.cpp/pull/26833); #28327 "capture non-string args raw to fix array<object> parsing" (OPEN, 2026-09-03) — [PR #28327](https://github.com/ggml-org/llama.cpp/pull/28327).
- **#30476 "common : interleave optional properties in json-schema-to-grammar"** (OPEN, 2026-10-13, by aldehir, a frequent chat/parser contributor). Fixes issue #30321 where an optional field listed before a required one gets dropped. — [PR #30476](https://github.com/ggml-org/llama.cpp/pull/30476), [Issue #30321](https://github.com/ggml-org/llama.cpp/issues/30321)
- **#31834** "common/json-schema : fix allOf with plain types generating an empty object grammar" (OPEN, 2026-11-06). — [PR #31834](https://github.com/ggml-org/llama.cpp/pull/31834)
- **#31862 "server : pass in --json-schema to chat parsing"** (OPEN, 2026-11-07, aldehir). `--json-schema` currently builds a grammar unaware of the generation prompt and without reasoning rules; this passes it into the chat parser so a valid grammar (including reasoning) is built. Supersedes #31809 and #29066; fixes #29006. Usefulness: medium (only if a server-wide schema is used). — [PR #31862](https://github.com/ggml-org/llama.cpp/pull/31862)
- **Named tool_choice**: #31564 "common : implement named tool choice" (OPEN, 2026-11-02) — object `tool_choice` currently "silently sets tool_choice to auto"; PR forces `required` with a single named function (autoparser + Qwen3-Coder) — [PR #31564](https://github.com/ggml-org/llama.cpp/pull/31564). #30800 (draft, 2026-10-19) does the same in server parsing and fixes Anthropic `{"type":"tool","name":...}` being mapped to plain `required` — [PR #30800](https://github.com/ggml-org/llama.cpp/pull/30800). #28806 rejects invalid named choices — [PR #28806](https://github.com/ggml-org/llama.cpp/pull/28806).
- **#30434 "withhold tool definitions when tool_choice: none"** (OPEN, 2026-10-13, +3/-1). Today `none` still passes tools to the template, so models emit unparsed tool calls. — [PR #30434](https://github.com/ggml-org/llama.cpp/pull/30434)
- **#31256 "common : support deferred tools and tool_reference content parts"** (MERGED 2026-10-27; not in Avalon). Deferred tool loading (Anthropic-style `defer_loading`/`tool_reference`); listed supported models are LFM3 and GLM-5.1, so only useful if the Qwen3.8 template supports it. — [PR #31256](https://github.com/ggml-org/llama.cpp/pull/31256)
- **#31861 "sanitize invalid UTF-8 in PEG tool call arguments"** (MERGED 2026-11-08) — robustness fix, cheap cherry-pick. — [PR #31861](https://github.com/ggml-org/llama.cpp/pull/31861)
- **#29876 "accumulate generated text and tokens as parse input"** (MERGED 2026-10-07) is groundwork for token-aware parsing in #29307 (not yet ready). — [PR #29876](https://github.com/ggml-org/llama.cpp/pull/29876)
- **#28504 "allow logprobs with tools + stream"** (draft, 2026-09-06) removes a 400 rejection. Low relevance. — [PR #28504](https://github.com/ggml-org/llama.cpp/pull/28504)
- Feature request **#29022 "Fast Tool Gating & Single-Pass Selection via Prefill Logit Slicing"** (OPEN issue, 12 comments): proposes choosing the tool name in one pass instead of autoregressively spelling `<tool_call>{"name":...`. No PR found. — [Issue #29022](https://github.com/ggml-org/llama.cpp/issues/29022)
- Issue #30246 (OPEN): K2 Horizon at low/medium effort leaves final text and tool calls inside `reasoning_content` with `finish_reason: stop` — the same failure class to watch for with Qwen3.8 low effort. — [Issue #30246](https://github.com/ggml-org/llama.cpp/issues/30246)

### Inferences
- Opencode tools frequently have mixed required/optional params; if Avalon ever logs tool-call parse failures or missing args, #31040 (XML) or #30193 (JSON) are the first things to port. Which one applies depends on the Qwen3.8 template's tool format (not verified here).
- Parallel tool calls and streaming tool-call deltas already exist upstream; I found no new open PR adding them in the window.

### Gaps
- Did not verify which tool-call format Avalon's custom Qwen3.8 template uses, so the choice between #30193 and #31040 is left open.
- No JSON-schema/grammar speedup PRs (as opposed to correctness) were found in the window.

## 2. Reasoning handling

### Takeaway
Upstream merged reasoning-effort-level discovery from the chat template (#31271) and several reasoning-budget fixes; open PRs add soft wrap-up / fractional budgets and let `reasoning_effort` override `--reasoning on|off`. Directly relevant to Avalon's low/medium/xhigh template.

### Cited Findings
- **#31271 "server: expose the reasoning effort levels of the chat template"** (MERGED 2026-10-27). Probe renders a user turn for minimal/low/medium/high/xhigh/max plus an unknown level; a level is honored if it renders differently from the unknown fallback; aliases collapse to the highest. Follow-up #31273 (MERGED 2026-10-28) builds the web UI reasoning menu from those levels. Usefulness: HIGH — could replace Avalon's template hack mapping high/max to xhigh, if the server also maps requests to the nearest supported level (not confirmed from the PR text). — [PR #31271](https://github.com/ggml-org/llama.cpp/pull/31271), [PR #31273](https://github.com/ggml-org/llama.cpp/pull/31273)
- **#27221 "chat: allow reasoning_effort to control enable_thinking"** (OPEN, 2026-08-17, +1/-2). With `--reasoning on|off`, `enable_thinking` is also put into `template_kwargs`, which overrides the per-request `reasoning_effort`; PR stops that. — [PR #27221](https://github.com/ggml-org/llama.cpp/pull/27221)
- **#27578 "reasoning-budget: opt-in soft wrap-up hint and bounded grace with hard termination cap"** (OPEN, 2026-08-22, +432/-27): `--reasoning-budget-soft-ratio` injects a hint at a newline once reasoning reaches a fraction of the budget, then a bounded grace period before hard cut. Based on draft #25961. — [PR #27578](https://github.com/ggml-org/llama.cpp/pull/27578)
- **#28189** (draft, 2026-09-01) `--reasoning-budget-warn-offset` reserves tokens for wrap-up before hard cutoff. — [PR #28189](https://github.com/ggml-org/llama.cpp/pull/28189)
- **#27592** (draft, 2026-08-23) fractional budgets (`0.5`, `0.5+4000`) and `--reasoning-budget-current`. — [PR #27592](https://github.com/ggml-org/llama.cpp/pull/27592)
- **#28356** (OPEN, 2026-09-04) `--reasoning-budget 0` with Qwen 3.8 still emits reasoning; fix so FORCING does not consume prefill tokens. — [PR #28356](https://github.com/ggml-org/llama.cpp/pull/28356)
- Merged fixes: #31374 end budget forcing on an end tag already in the prompt (2026-10-30) — [PR #31374](https://github.com/ggml-org/llama.cpp/pull/31374); #28869 force `\n</think>` on budget end for qwen3-coder (2026-09-16) — [PR #28869](https://github.com/ggml-org/llama.cpp/pull/28869); #30557 stop qwen3-coder reasoning/content before template separators (2026-10-28) — [PR #30557](https://github.com/ggml-org/llama.cpp/pull/30557); #31380 strip think block for qwen3-coder with reasoning_format none (2026-10-31) — [PR #31380](https://github.com/ggml-org/llama.cpp/pull/31380); #28174 preserve_reasoning kwarg enabled by default (2026-09-02) — [PR #28174](https://github.com/ggml-org/llama.cpp/pull/28174); #26045 `reasoning_effort: "none"` (2026-07-24) — [PR #26045](https://github.com/ggml-org/llama.cpp/pull/26045).
- Feature requests: #29372 split timings into reasoning vs answer phases (rate and draft acceptance per phase) — [Issue #29372](https://github.com/ggml-org/llama.cpp/issues/29372); #31353 inject text right after `<think>` to steer thinking — [Issue #31353](https://github.com/ggml-org/llama.cpp/issues/31353).

### Inferences
- The qwen3-coder reasoning fixes (#30557 merged after Avalon's base, #31380) are low-risk cherry-picks if Qwen3.8 is routed through the qwen3-coder parser path.
- A thinking budget with soft wrap-up (#27578) would let xhigh effort be capped for latency-sensitive turns without hard-cutting mid-sentence; still unreviewed.

### Gaps
- Could not confirm whether #31271 also clamps an unsupported requested effort (e.g. "high") to a supported one server-side, or only reports levels.

## 3. Prompt/KV cache, slots and scheduling

### Takeaway
This is the most active area. Open PRs add a persistent disk prompt cache (#28092), session-keyed automatic slot save/restore (#29738), cross-slot prefix sharing without copies (#30970), chunked prefill with shortest-job-first while busy (#30992 — directly addresses the title-request timeout), and cache-preservation across sleep. Merged items since Avalon's base reduce checkpoint copy cost (#30137).

### Cited Findings
- **#30992 "server : add --prefill-budget-busy"** (OPEN, 2026-10-22, upd 2026-11-06, +126/-3). While a user is receiving output, long prompts are prefilled in chunks of N per step instead of all at once; "If multiple requests are waiting, the shortest one is processed first", with an anti-starvation mechanism. Tested on Strix Halo. Usefulness: HIGH — with `-np 2` it would let opencode's small title request run while a 100k agent prompt prefills. — [PR #30992](https://github.com/ggml-org/llama.cpp/pull/30992)
- **#28532 "server: prioritise throughput over latency"** (OPEN, 2026-09-07, +502/-4, 5 comments). `--slot-linger-ms`: when a slot frees, briefly reserve it for the session that just ended a turn (agent turns often end in a millisecond tool call) to avoid KV swaps. Usefulness: medium-high for multi-session; opposite trade-off to #30992. — [PR #28532](https://github.com/ggml-org/llama.cpp/pull/28532)
- **#30970 "server : share prompt prefix cells between slots (--slot-attach)"** (OPEN, 2026-10-22, +544/-17). With unified KV, a new request can start from another (even busy) slot's cells holding the same prefix — no copy, no recompute; for recurrent/hybrid models it attaches at a checkpoint and the server adds a checkpoint where prompts diverge (≥1,024 tokens in). Usefulness: HIGH if running 2 slots (main agent + subagent/title requests sharing the system prompt + tool defs). — [PR #30970](https://github.com/ggml-org/llama.cpp/pull/30970)
- **#26204 "/slots action=clone_to (KV clone between slots)"** (OPEN, 2026-07-27): copies KV between slots; 4 branches on a 7,403-token trunk ran 2.12x faster (42.9 s to 20.2 s) on M3 Max. — [PR #26204](https://github.com/ggml-org/llama.cpp/pull/26204)
- **#28092 "server : add --cache-disk for a persistent prompt cache"** (OPEN, 2026-08-31, +1706/-71, 9 comments). Persistent prompt cache supporting hybrid/SWA checkpoints, auto-restore across restarts, corrupt-entry cleanup, LRU bounded by disk size; closes #20697. Usefulness: HIGH for Avalon (first session after restart is slow; 131k prompts take ~70 s+ to prefill at ~1.8k t/s), but disk on / is tight — point at the D drive. — [PR #28092](https://github.com/ggml-org/llama.cpp/pull/28092)
- **#29738 "feat(server): session-keyed automatic slot save/restore"** (OPEN, 2026-09-30, +914/-55, 7 comments) implementing issue #29678. Uses client session-id headers to save/restore slots to files automatically, with lifecycle cleanup; tested with Pi (`sendSessionAffinityHeaders`). Usefulness: HIGH if opencode sends a session header (unverified). — [PR #29738](https://github.com/ggml-org/llama.cpp/pull/29738), [Issue #29678](https://github.com/ggml-org/llama.cpp/issues/29678)
- **#29578 "Server/pinned prefix preload"** (OPEN, 2026-09-28, +586/-77): `cache-prompt-dir` to preload pinned prefixes into RAM. — [PR #29578](https://github.com/ggml-org/llama.cpp/pull/29578)
- Issue **#29201** export/import slot state over HTTP (for proxies routing across instances). — [Issue #29201](https://github.com/ggml-org/llama.cpp/issues/29201)
- **Sleep + cache**: #28022 `--sleep-preserve-cache` (OPEN, 2026-08-30) — [PR #28022](https://github.com/ggml-org/llama.cpp/pull/28022); #29408 save slots into prompt cache before sleep, keep cache on wake (OPEN, 2026-09-25, fixes #29322) — [PR #29408](https://github.com/ggml-org/llama.cpp/pull/29408); #31166 (draft, 2026-10-26, tested on Qwen3.8-27B with HIP) — [PR #31166](https://github.com/ggml-org/llama.cpp/pull/31166). Relevant only if `--sleep-idle-seconds` is used.
- **#30997 "retry cached prompt restoration after clearing idle slots"** (OPEN, 2026-10-22, fixes #30961): with unified KV, a RAM-cached prompt can fail to restore because an idle slot holds the cells, causing full reprocessing; PR clears idle slots and retries. — [PR #30997](https://github.com/ggml-org/llama.cpp/pull/30997)
- **#30866 "Pin empty slot"** (OPEN, 2026-10-20, +22): with a pinned slot, a 0/0 NaN comparison makes an evicted conversation reprocess cold instead of loading from RAM cache (fixes #28139). — [PR #30866](https://github.com/ggml-org/llama.cpp/pull/30866)
- **#28046 "per-entry prompt-cache context limit"** (`-kvcpc`, OPEN, 2026-08-30): aggregate token limit can evict independent conversations before `--cache-ram` is full. — [PR #28046](https://github.com/ggml-org/llama.cpp/pull/28046)
- **#30188 "discard prompt state for cache_prompt=false tasks"** (draft, upd 2026-11-08): a `cache_prompt:false` request (e.g. a title request) saves the displaced conversation to RAM and does not pollute caches. Usefulness: medium — a way to keep title requests from evicting the agent prompt. — [PR #30188](https://github.com/ggml-org/llama.cpp/pull/30188)
- **#31740 "[WIP] throttle unified KV work at free-cell watermarks"** (draft, 2026-11-05): pause sequences at 20%/10% free KV to avoid "context size exceeded". — [PR #31740](https://github.com/ggml-org/llama.cpp/pull/31740)
- **#27451** share checkpoint storage and handle `bad_alloc` in prompt cache (OPEN, 2026-08-20). — [PR #27451](https://github.com/ggml-org/llama.cpp/pull/27451)
- Merged since Avalon's base: **#30137 reference-counted checkpoints instead of deep copies** (MERGED 2026-11-05) — [PR #30137](https://github.com/ggml-org/llama.cpp/pull/30137); #30567 prefer empty slots / save states overwritten by child tasks (MERGED 2026-10-15, may already be in Avalon) — [PR #30567](https://github.com/ggml-org/llama.cpp/pull/30567); #30498 save only restorable slot states (MERGED 2026-10-15) — [PR #30498](https://github.com/ggml-org/llama.cpp/pull/30498); #26004 preserve checkpoints across slot save/restore (MERGED 2026-10-08) — [PR #26004](https://github.com/ggml-org/llama.cpp/pull/26004).
- Known open bug to be aware of: **#27148** RAM prompt cache (`--cache-ram`/`--cache-idle-slots`, both default on) can restore an unrelated finished conversation into a new slot under concurrent load with `--kv-unified` (6 comments). — [Issue #27148](https://github.com/ggml-org/llama.cpp/issues/27148)
- Others: #29918 `--cache-reuse-hybrid` for recurrent/hybrid models (draft) — [PR #29918](https://github.com/ggml-org/llama.cpp/pull/29918); #29569 "Server radix and thinking disable" (draft, +1579, empty description) — [PR #29569](https://github.com/ggml-org/llama.cpp/pull/29569).

### Inferences
- For the single-user opencode workload the best ROI candidates are: #30992 (lets a second slot serve title/subagent requests without stalling), #30970 (shared prefix across those slots), #28092 or #29738 (avoid cold prefill after restart/eviction). All are unreviewed and large; #30992 is the smallest (+126).
- If Qwen3.8-Flash-Next is hybrid/recurrent (checkpoint-based reuse), the checkpoint PRs (#30137, #30970's checkpoint-at-divergence, #28092's checkpoint support) matter more than for pure-attention models.

### Gaps
- Did not verify whether opencode sends session-affinity headers (needed for #29738).
- No upstream context-compaction/summarisation feature found; context shift remains the only built-in mechanism.

## 4. API compatibility (Responses, Anthropic Messages, router, metrics)

### Takeaway
Responses and Messages endpoints exist and keep getting compatibility fixes (Codex CLI drives most); new merged APIs are /v1/realtime (audio) and router LRU. For opencode via chat completions most of this is low priority, except metrics and connection-reuse fixes.

### Cited Findings
- Responses API open fixes: #31638 truncated tool call reported as `completed` and missing `function_call_arguments.done` (OPEN, 2026-11-04, fixes #30773) — [PR #31638](https://github.com/ggml-org/llama.cpp/pull/31638); #31568 namespaced tools (Codex CLI, MCP hosts; flattens to `ns_{namespace}___{tool}`) (OPEN, 2026-11-03) — [PR #31568](https://github.com/ggml-org/llama.cpp/pull/31568); issue #30859 unknown input item types (Codex `additional_tools`) abort every request — [Issue #30859](https://github.com/ggml-org/llama.cpp/issues/30859); #29346 `text.format` not translated to `response_format` — [PR #29346](https://github.com/ggml-org/llama.cpp/pull/29346); #27751 reasoning items/Codex compat — [PR #27751](https://github.com/ggml-org/llama.cpp/pull/27751); long-open umbrella #19720 "add OpenAI Responses API compliance" (OPEN since 2026-02-19) — [PR #19720](https://github.com/ggml-org/llama.cpp/pull/19720). Merged: #30959 `extra_params` on /v1/responses (2026-10-31) — [PR #30959](https://github.com/ggml-org/llama.cpp/pull/30959); #30863 output_index in SSE (2026-10-20) — [PR #30863](https://github.com/ggml-org/llama.cpp/pull/30863).
- Anthropic Messages: merged #30856 forwards all 24 llama.cpp sampling params on /v1/messages (2026-10-21) — [PR #30856](https://github.com/ggml-org/llama.cpp/pull/30856); #28554 id_slot pass-through (2026-10-20) — [PR #28554](https://github.com/ggml-org/llama.cpp/pull/28554). Open: #30244 timings in /v1/messages — [PR #30244](https://github.com/ggml-org/llama.cpp/pull/30244); #31554 stop_sequence as stop_reason — [PR #31554](https://github.com/ggml-org/llama.cpp/pull/31554); #29482 tool results before user text (draft) — [PR #29482](https://github.com/ggml-org/llama.cpp/pull/29482).
- Router mode: merged LRU scheduler #26572 and "do not evict busy models" #26567 (2026-08-07), eviction-race fix #29217 (2026-09-22), disconnect propagation to children #29707 (2026-10-20). — [PR #26572](https://github.com/ggml-org/llama.cpp/pull/26572), [PR #29217](https://github.com/ggml-org/llama.cpp/pull/29217), [PR #29707](https://github.com/ggml-org/llama.cpp/pull/29707). Open #27027 require API key for non-GET /models — [PR #27027](https://github.com/ggml-org/llama.cpp/pull/27027).
- Metrics: #27760 report live generation throughput in /metrics during a request (OPEN, 2026-08-26) — [PR #27760](https://github.com/ggml-org/llama.cpp/pull/27760); #30543 flush metrics after every batch decode (OPEN) — [PR #30543](https://github.com/ggml-org/llama.cpp/pull/30543); #27280 optional OTLP/HTTP tracing (draft) — [PR #27280](https://github.com/ggml-org/llama.cpp/pull/27280). Merged #27041 /metrics and /slots accessible during `llama_decode()` (2026-08-14) — [PR #27041](https://github.com/ggml-org/llama.cpp/pull/27041).
- Connection handling: #30492 fixes server closing the connection after a successful streamed response despite Keep-Alive (OPEN, fixes #29680). Usefulness: medium — saves a TCP reconnect per agent turn. — [PR #30492](https://github.com/ggml-org/llama.cpp/pull/30492); #28059 gzip request/response compression (SSE excluded) — [PR #28059](https://github.com/ggml-org/llama.cpp/pull/28059).
- Architecture/control: #31489 service mode with control endpoint over a Unix `.sock` (OPEN, ngxson, +1434) — [PR #31489](https://github.com/ggml-org/llama.cpp/pull/31489); #30689 split inference core into `tools/engine` library (draft, ngxson, +6047/-2385) — [PR #30689](https://github.com/ggml-org/llama.cpp/pull/30689). Both are by a core server maintainer, so likely to land eventually and would conflict heavily with fork patches.
- Other new merged APIs (low relevance): /v1/realtime conversation + tool calls (#31269, #31287, 2026-10-28) — [PR #31287](https://github.com/ggml-org/llama.cpp/pull/31287); `x-tool-cwd` header for built-in tools (#26420, 2026-08-03) — [PR #26420](https://github.com/ggml-org/llama.cpp/pull/26420); `--connect` P2P (#28277) — [PR #28277](https://github.com/ggml-org/llama.cpp/pull/28277). Open #28053 `-sysf` server-wide system prompt file — [PR #28053](https://github.com/ggml-org/llama.cpp/pull/28053).
- Privacy note: issue #31573 shows `/slots` token counters leak what a local tool-using agent did to co-resident processes. — [Issue #31573](https://github.com/ggml-org/llama.cpp/issues/31573)

### Inferences
- #30689 (engine split) is the biggest upcoming rebase risk for Avalon; worth tracking before porting big server PRs.

### Gaps
- No upstream discussion found for "model hot-swap" beyond router mode.

## 5. Other TTFT / robustness items

### Takeaway
Few items beyond the cache/scheduling work; most robustness is small parser/grammar hardening.

### Cited Findings
- #31621 prompt cache reuse with a multimodal projector loaded (only skip reuse for prompts with media) — [PR #31621](https://github.com/ggml-org/llama.cpp/pull/31621).
- Grammar hardening merged: recursion-depth bound #29755 (2026-10-25) — [PR #29755](https://github.com/ggml-org/llama.cpp/pull/29755); reject JSON nested >128 levels #30242 — [PR #30242](https://github.com/ggml-org/llama.cpp/pull/30242).
- Issue #30887 (OPEN): "Qwen3.8-Flash-Next + DFlash2 crashes on first prompt in extract_layer_inputs()" — spec-dec, out of scope, but names Avalon's model. — [Issue #30887](https://github.com/ggml-org/llama.cpp/issues/30887)
- Issue #30238 asks for offloading agentic subtasks to another llama-server instance from the UI. — [Issue #30238](https://github.com/ggml-org/llama.cpp/issues/30238)

### Inferences
- Shortlist to port, in order: (1) #30992 prefill budget + shortest-first (small, fixes the title-timeout class with `-np 2`); (2) #31040 / #30193 tool-arg order; (3) #30137 refcounted checkpoints (merged, low risk); (4) #28092 disk cache or #29738 session restore; (5) #30970 slot-attach; (6) #30492 keep-alive; (7) #27221 effort overrides `--reasoning`.

### Gaps
- No benchmarks found for most PRs on ROCm or on 100k+ prompts; usefulness ratings are inferred from descriptions.
- GitHub Discussions were not searched separately (gh search covered PRs/issues only).
