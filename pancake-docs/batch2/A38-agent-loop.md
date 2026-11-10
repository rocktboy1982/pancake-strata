# A38 / A39 / A41 / A42 / A43 - agent-loop helpers (batch 2)

Branch `work/a38-agent-loop`.  Code: `serve/agent_loop.py` (new), wiring in `serve/server.py`
(`Service.prepare`, `Service.run` / `server_turn`, `_write_turn_log`, `/metrics`), tests `serve/test_agent_loop.py`,
spec `bench/specs/a38-agent-loop.json`.  Everything is opt-in and default off: with no switch set,
`Service.agent_loop.active` is False, `prepare` and `run` take exactly the old path, no new key appears in the turn
log, `/metrics` or `/metrics/prometheus`.

The server only ever sees the request's messages, i.e. the model's tool calls and the tool results the client sends
back.  Every statement below is about "what the conversation shows", never about the disk.

| Item | What | Switch | Status |
|---|---|---|---|
| A38 | exact per-request behaviour counters in the turn log | `STRATA_AGENT_COUNTERS=1` | built, log only (exact, not lossy) |
| A41 | read-before-edit: already enforced by opencode; test-before-done: one nudge at the end of the answer | `STRATA_TEST_BEFORE_DONE=1` | built, LOSSY, opt-in |
| A42 | externalize huge tool results to a file, head/tail excerpt + path in the prompt | `STRATA_EXTERNALIZE_TOKENS=N` | built, LOSSY, opt-in |
| A43 | flag edit calls whose `oldString` is in nothing the model read of that file | `STRATA_EDIT_OLDSTRING_CHECK=log\|note` | built as check + log + note; constrained decoding NOT built |
| A39 | asynchronous compaction in tool windows | - | design note only (below) |

## A38 - behaviour counters (log first)

Paper: "Beyond Accuracy" (UBC / Queen's 2025, arXiv 2511.11012): failed agent runs cost 1.4-4.4x the tokens; the
useful signals are edits that are never tested and reads that never lead to an edit.

Item 33 (`loop_guard.behaviour`, already in the tree) counts the two streaks, but only reports them when they cross
a threshold (window line, `/metrics`).  A38 adds the full per-request numbers to the turn log, so a session can be
analysed afterwards and the A/B arms (A57, time-to-success) can be compared on behaviour, not only on pass rate.

With `STRATA_AGENT_COUNTERS=1` (and the turn log on, `STRATA_TURN_LOG`) every line gets an `"agent"` object, computed
from the request's messages since the user's last message (a bare "continue" does not start over, as in item 33):

`tool_calls, rounds, edits, checks, reads, other` (kinds from `loop_guard.call_kind`, so the same opencode / Claude
Code name table and the same bash classifier as item 33, including the config's `loop_guard_behaviour.tools` table),
`edits_since_check, reads_since_edit` (the streaks at the request's end; equal to item 33's, tested),
`max_edits_since_check, max_reads_since_edit`, `code_edits_since_check` (the streak without doc files `.md .txt
.rst ...`), `files_edited`, `edits_unread` (edit calls with an oldString on a file that no read result of the whole
conversation shows and no write created), `edit_old_exact / _loose / _miss / _unknown` and `edit_failed`
(A43, below), `last_round_old_miss, last_round_failed`, `ends_unchecked`.  No text, only numbers.  Key set is fixed
(all keys always present), the request has no tool call in its turn -> no `agent` key.

The log has one line per request (not per user turn), so the counters at the last request of a user turn are the
turn's totals; `max_*` are the turn's maxima.

## A41 - read-before-edit, test-before-done

Paper: ECLoop (Concordia 2026, arXiv 2607.28815).

**Read-before-edit: nothing to build.**  Checked in the installed opencode binary (`/snap/opencode/current/bin/opencode`,
build of 2026-10-28; read with `grep -a`, not modified): the edit tool's own description says "You must use your Read
tool at least once in the conversation before editing. This tool will error if you attempt an edit without reading the
file", the write tool's "If this is an existing file, you MUST use the Read tool first ... This tool will fail if you
did not read the file first".  So a server-side check would only repeat what the client already refuses.  A38's
`edits_unread` counts, from the history, edit calls on files with no read result - if a client version stops
enforcing this, the number shows it.  (A write that creates a new file counts as "known".)

**Test-before-done: built, opt-in.**  When the model is about to end its turn (no tool call in the answer) and the
request showed code edits with no build, test or run since (`ends_unchecked`: the request ends with tool results and
`code_edits_since_check > 0`; doc-only edits do not count), the server continues the same answer once, with a short
user-role note (`agent_loop.FINISH_NOTE`: run the check that fits and read what it prints, or say in one sentence why
none applies).  It reuses the loop guard's keep-going mechanism (`_keep_suffix` renders the template's turn end + the
note + the next assistant opening; `server_turn` returns it as `server_next`), so no new generation path.
- Prefix (A16): the request's prompt is untouched; the note is appended after the model's own words in this answer.
  The client never sees the note.  Its next request carries the assistant message (text + the call to run the tests)
  and so differs from the server's cache from the end of the model's first text on: only that short tail is read
  again, the same cost as the loop guard's keep-going.
- Once per answer (`nudged`); the answer that ends the turn after the test ends normally.  An answer that already
  makes a tool call is never nudged.
- Needs the loop guard on (it is the default) and a template whose turn is a prefix of the next (the same condition as
  keep-going; otherwise `_keep_suffix` returns None and nothing happens).
- What counts as a test is item 33's classifier (bash by command regex: make, pytest, cargo test, ./x ...; tools in
  `loop_guard_behaviour.tools` mapped to "check").  An MCP test runner must be added to that table.
- LOSSY: the answer is longer and the model reacts to the note.  Risk: nudging a model whose edits really need no
  test (it is allowed to say so); one extra continuation (a few hundred decode tokens) per finished edit turn.

## A42 - externalize huge inputs

Paper: Recursive Language Models (MIT 2025, arXiv 2512.24601): keep the huge input outside the prompt and let the
model look into it, instead of compacting it.

`STRATA_EXTERNALIZE_TOKENS=N` (0 = off): in `prepare`, every tool result of more than N tokens (estimate: 4 characters
a token, no tokenizer call) is written to `STRATA_EXTERNALIZE_DIR/<sha256 prefix>-<tool>.txt` (default
`<tmp>/strata-externalized`, directory mode 0700) and replaced, in the prompt only, by its first
`STRATA_EXTERNALIZE_HEAD_TOKENS` (400) and last `STRATA_EXTERNALIZE_TAIL_TOKENS` (400) tokens cut at line ends,
and one line "N of M characters (~T tokens) are left out here. The whole output is saved in the file PATH - read it in
ranges (offset and limit) or search it with grep instead of running the command again."
- Byte-stable prefix: the excerpt and the path depend on the result alone, and the client resends the same result
  every request, so every later request of the conversation carries the same bytes at the same place (tested:
  second request writes nothing new and gives an identical message).  The client's own conversation is not touched.
- Results of the tools in `STRATA_EXTERNALIZE_SKIP` (default `read,read_file`: a file read is already a file, and
  A43 needs read results), results with pictures, and excerpts are left alone.  A file that cannot be written leaves
  the result as it is.
- LOSSY: the model sees 800 tokens of a result and must read the rest itself.  This only works when its tools can
  read the path: opencode's `read` / `grep` of a path outside the project may ask for permission
  (`external_directory`), which blocks a headless run, so for opencode set `STRATA_EXTERNALIZE_DIR` to a directory
  inside the project (add it to `.gitignore`).  The files contain tool output (possibly secrets); there is no
  clean-up (names are content hashes, so the same output is stored once; delete the directory when done).
- The counters of A38/A43 are computed from the original messages, before the replacement.
- Not built: truncation by relevance, a per-tool N, an automatic summary.  Compaction (opencode's own) is untouched.

## A43 - oldString substring constraint

Paper: Type-Constrained Code Generation (ETH, PLDI 2025, arXiv 2504.09246).  Parked in the plan: 1 failure in 180
edits, so the upside is small.

Built (`STRATA_EDIT_OLDSTRING_CHECK=log|note`, default off): from the history the server knows, per file, the texts the
conversation showed - the file's read results (line numbers and wrapper lines removed: opencode `00001| `, `1: `, Claude
Code `1<tab>` / `1->`), the `newString` of earlier successful edits and the content of earlier writes (a write replaces
what was read).  Every edit call (also each entry of a multi-edit) is classified: `exact` (oldString is a substring),
`loose` (only after white space is collapsed - opencode's edit tool also does fuzzy matching, so this usually works),
`miss` (in none of them), `unknown` (no text of that file known: e.g. read through bash).  `edit_failed` counts results
that carry an edit tool's "not found / multiple matches" wording (opencode, Claude Code), so the log shows how often
`miss` really failed.
- `log`: the counters in the turn log (with `STRATA_AGENT_COUNTERS=1`), `strata:agent_loop_oldstring_flagged_total`.
- `note`: after the failed result of the model's last round, a user-role note (re-read the lines, copy oldString
  exactly, without line numbers); only when the round has a `miss` and a failed result.  Appended after the loop
  guard's own notes, tail only.
- False positives: a read with `offset/limit` that missed the region the model edits later via grep output only;
  a file changed on disk by a bash command between read and edit.  They are flagged `miss` although valid.  The log of
  `edit_old_miss` against `edit_failed` over real sessions is the measurement that decides whether anything stronger
  is worth building.

**Not built: constrained decoding.**  The server cannot restrict the sampler to substrings of a file: the engine's
grammar is built from the tool's JSON schema (`serve/grammar.py`, `_grammar` request key) and `logit_bias` is static per
request.  A per-step "next token must continue a substring of T" mask needs a suffix automaton of the read text
inside the engine's decode loop (a per-step host callback or an engine-side trie), and a JSON-Schema `pattern` for
"substring of T" is quadratic in |T|.  Not cheap, and the measured failure rate (1 in 180) does not pay for it.  A
cheaper next step if the counters show a real failure rate: a generation-time check at the end of an edit call (the
`tool_call` event, next to `server_turn`'s blocked-call machinery) that answers a flagged call with a short result in
the same answer, like the loop guard's blocked calls, instead of letting the client run it.

## A39 - asynchronous compaction in tool windows (design note only)

Papers: Slipstream (arXiv 2605.08580, -39.7% latency), ACE (Stanford 2025, arXiv 2510.04618).  Not built; nothing
small fits.  What it would be here: while the model decodes, a background worker prepares the compacted form of the
older tool window (A42's excerpts are the cheapest compaction; a summary needs a model call), and the next request
swaps it in at a turn boundary.  Why it is not small:
1. The engine runs one request at a time on one GPU slot (parked with A9: "one GPU slot shared"); a summariser call
   would compete with the decode that it is meant to speed up.  A CPU-only summary of tool output (head/tail,
   deduplication of repeated lines) is possible but is what A42 already does at no cost.
2. The compaction must not move the prefix: the swap is only free when it happens exactly where the client
   itself compacts (opencode's `compaction` rewrites the history on its own; the server cannot know when).  Swapping
   in the middle of a conversation re-reads everything after the changed message (62K tokens = minutes with
   experts on the CPU), which costs more than the compaction saves.
3. "Delta compaction" (ACE) needs a stable playbook the client keeps: opencode has none.
Recommendation: first measure with A42 on and the A38 counters how much of the prompt is old tool output; build
the swap only if that share is large AND the compaction can be tied to opencode's own compaction events (a plugin
that tells the server, see below).

## Optional opencode plugin

Not added: opencode's `tool.execute.after` hook could do A42 client-side (and for any provider), but the install under
`~/.config/opencode` must not be touched and the server-side version works for every client.  If A42 shows value, the
same `agent_loop.excerpt()` is a ten-line plugin.

## Files changed

- `serve/agent_loop.py` (new): `settings`, `counters`, `edit_old_state`, `known_text`, `externalize`, `excerpt`,
  `FINISH_NOTE`, `oldstring_note`.
- `serve/server.py`: import; `Service.agent_loop` / `agent_loop_counts`; `prepare` (counters + externalize before the
  loop guard, A43 note after it, `_loop_finish` for A41, `_agent_counters` for the log); `run` (`nudge` in
  `server_turn`, the stop-token check `(blocked or keep or nudge)`); `_write_turn_log(..., agent=)`; `/metrics`
  `agent_loop` and `strata:agent_loop_*_total` only when a switch is on; start-up line and parsing in `main`.
- `serve/test_agent_loop.py`, `bench/specs/a38-agent-loop.json`, this note.

## Tests run

**Not run.**  `tools/pancake/abtest.py` (the 09-prefill screen) held the machine for the whole session and
`wait_quiet.sh` returned 1 (busy) four times between 23:11 and 23:52, so by the brief the tests were not run.  Only
`python3 -m py_compile serve/agent_loop.py serve/test_agent_loop.py serve/server.py` was run (OK).  The code was
re-read for syntax, types and the nonlocal / variable scoping of `server_turn`.  To run:

    nice -n 19 python3 -m unittest serve.test_agent_loop -v          # 25 cases, mock engines, no GPU
    nice -n 19 python3 -m unittest serve.test_loop_guard serve.test_turn_log serve.test_prometheus   # regression

`serve/test_agent_loop.py` covers: every counter (kinds, streaks, max streaks, turn reset, bare "continue", doc-only
edits, unread edits, Claude Code names, agreement with item 33's `loop_guard.behaviour`), the oldString states with
the three read-output line-number styles, text written by earlier edits / writes, multi-edit, settings parsing
(including the bad values), `externalize` (stored file equals the result, excerpt shape, determinism and idempotence,
skip list, limit boundary, unwritable directory), and the server wiring: default path untouched (no keys, no
metrics), the `agent` object in the turn log only with the switch, the A43 log and note modes, the A41 continuation
(prompt prefix kept, the note in the next engine prompt, once per answer, not when checked / docs only / switch off
/ the answer already calls a tool), A42 in the prompt only.  If a case fails, the most likely suspects are the
template-dependent ones (`_keep_suffix` rendering for the A41 test; the exact text the mock template gives).

## Spec to run

`bench/specs/a38-agent-loop.json` is a regression screen only (arms start, no speed cost; the `short` case has no
tool calls).  The verdict needs agent traffic: `tools/pancake/task_gate.py` with a multi-turn edit-then-test task set
(time-to-success area, A57), or a replay of a real opencode session with the arms of that spec.  First, run the
daily opencode traffic with `STRATA_AGENT_COUNTERS=1` + `STRATA_EDIT_OLDSTRING_CHECK=log` (both log only, no
behaviour change) for a few sessions: it gives the real `edits_unread`, `edit_old_miss` vs `edit_failed`, and
`ends_unchecked` rates, which decide whether A41 / A43 are worth enabling.

## Expected gain and risk

- A38: none by itself (measurement).  Cost: one scan of the history per request (linear in the history text; a few ms).
- A41: authors +4.8..11.8 pp, -12.1% tokens.  Here: only for sessions that end after edits without a test; each costs
  one extra continuation.  Risk: lossy, a needless test run for edits that need none.
- A42: authors +26% median vs compaction (GPT-5).  Here: the prompt shrinks by whatever the huge results were
  (prefill on this box is slow: 62K tokens cold re-read is minutes).  Risk: the model does not follow the pointer, or
  its read tool cannot reach the path.
- A43: small (1 failure in 180 edits).  Log first.

## What is left

- Real-session numbers (above) and a task-gate verdict for A41 / A42.
- A43 generation-time check / constrained decoding (above), A39 (above).
- `STRATA_AGENT_COUNTERS=1` is not yet in a preset; adding it next to `STRATA_TURN_LOG` in the Flash-Next presets is
  one line once the coordinator agrees.
- Untested here: a live opencode session (no server was run); the Anthropic `/v1/messages` path shares
  `prepare` and the template-message shape, but only the Chat Completions shape is covered by the tests.
