# A24: cross-session suffix index with frequency scoring, shared by sub-agents

MOD_PLAN Phase 16 item A24. Paper: SuffixDecoding (Oliaro et al., CMU, "SuffixDecoding: Extreme Speculative
Decoding for Emerging AI Applications", https://arxiv.org/abs/2411.04975). Branch `work/a24-suffix-index` (from
`origin/dev/batch-2`).

## What

`--lookup` (SuffixDrafter) drafts from a repeat inside the one request it serves: that request's prompt and its output
so far. An agent also repeats what it wrote in other requests: the same tool-call skeletons, the same reasoning
openers, the same edit shapes. A sub-agent writes much of what its siblings and its parent wrote. The paper keeps a
global suffix tree over previous outputs, scores each continuation by how often it followed (count(child) /
count(node)), and drafts the most frequent path.

This item adds that global index to the engine, beside `--lookup`:

- `SuffixIndex` (`include/strata/spec/suffix_index.hpp`, `src/spec/suffix_index.cpp`): a ring of the finished
  outputs of every request (main agent and sub-agents), each after a lead of its prompt's last 64 tokens. Three hash
  chains (the places that end with the same 24 / 8 / 3 tokens, as an LZ77 matcher chains them) find the deepest
  match of the current context (back to 64 tokens, 64 places walked at most). What followed those places is voted on
  token by token: the most frequent next token is drafted, its share is its probability, the places that disagree
  drop out, and the draft ends where the product of shares falls under `min_prob` (0.1) or after `factor` x the match
  length (1.0) tokens. These are the paper's `min_token_prob` and `max_spec_factor` defaults. The engine verifies one
  window, so the draft is the single most frequent path of the paper's speculation tree, not the tree itself.
- The ring evicts the oldest output first. Memory is bounded: 40 bytes a token (ring 4, per level a chain entry 4 and
  a table slot 8). The capacity is a power of two: 64 MiB gives 1,048,576 tokens, which held the whole 10-day
  recorded history below.
- `SharedSuffixIndex`: the index behind a `std::shared_mutex`. Proposals take a shared lock and keep their scratch on
  the stack (`propose()` is const). Inserts take the exclusive lock. An output goes in whole when its request ends, so
  two requests that decode at the same time can never interleave their tokens in the ring. The serve loop serves one
  request at a time today, so the lock is never contended.
- The structure is ported from strata-poly's `SuffixIndex` (`src/poly/suffix_index.hpp`, from the FastStrata fork's
  poly engine). I changed three things: the query takes an explicit context, the index holds outputs with a prompt
  lead, and the lock is new. strata-poly is untouched. **Credit row needed** (memory rule
  `feedback_pancake-credits`): README §9 and RESEARCH_AND_FINDINGS §4 should credit FastStrata's strata-poly
  SuffixIndex as the base of A24. I did not edit those files (brief rule 5).

**Exact.** The index only proposes drafts, and the verify window checks them, so the output cannot change. Window
shapes can change, as with `--lookup-max-t 12`, so a rounding-level text difference against base is possible. That
is why the screen spec has `lossy: true`: run the quality gate only if the hash differs.

## How to enable

| env | default | meaning |
|---|---|---|
| `STRATA_SUFFIX_INDEX=1` | off | turns the index on (serve mode). Off: `shared_suffix_index()` returns null and every hook is skipped. |
| `STRATA_SUFFIX_INDEX_MB` | 64 | memory cap in MiB (1..16384), rounded down to a power of two of tokens |
| `STRATA_SUFFIX_INDEX_MIN_PROB` | 0.1 | a draft ends where its cumulative probability falls under this |
| `STRATA_SUFFIX_INDEX_FACTOR` | 1.0 | a draft is at most factor x match-length tokens |
| `STRATA_SUFFIX_INDEX_RULE` | longest | `longest`: the index's draft replaces `--lookup`'s when its match is longer (ties keep lookup). `fallback`: the index drafts only where `--lookup` has none. |
| `STRATA_SUFFIX_INDEX_FILE` | unset | a log file that keeps the index across restarts (see below); unset: memory only |

**Restart log (`STRATA_SUFFIX_INDEX_FILE=<path>`, a second opt-in).** The simulator shows that much of the gain comes
from outputs of earlier days (numbers below), but the index lives in memory and the server restarts often. With a
path set, `SharedSuffixIndex::attach_log()` works like this at startup:

- It reads the log: an append-only file, `SXI1`, then per output `uint32 lead, uint32 out` and the int32 tokens.
- It inserts the log's newest outputs that fit in the ring.
- After that, every insert is appended to the log, with one `fwrite` and `fflush` at the end of each request (a few
  KB).
- A record cut off by a kill is dropped. A log above twice the ring is compacted to the outputs that fit. Both are
  written to `<path>.tmp` and then renamed over the log.
- A file without the `SXI1` header is refused and never touched.
- If a write fails, the index goes on in memory only.

The file holds token ids of the agent's outputs (the user's own data on the user's disk). Its size stays at most
twice the ring: 8 MiB at the default 64 MiB. Do not use it in the abtest screen: a shared log would carry outputs
across arms and reps.

At startup, a line on stderr confirms the index is on:
`strata: STRATA_SUFFIX_INDEX=1: cross-session suffix index (A24), 1048576 tokens in 40 MiB, ...`.
After each request, another line reports the index's work:
`strata serve: suffix index: N lookups, D drafts, W windows took one, A of X drafts accepted, U us a lookup; holds H of C tokens (O outputs)`.

## Where it hooks (src/program/generate.cpp, serve loop only)

1. Serial serve round, the `--lookup` block (`o.suffix_draft > 0 && !first_window`): after `sfx.propose()`,
   `sidx_offer()` asks the index with the last 64 committed tokens (`sfx.history()`). If the index's draft wins by the
   rule, it goes into `sbuf` and `sfx_match` becomes the index's match. The policy (`policy.choose` /
   `source_policy.choose`), the MTP-first-token agreement check and `graph_fits` then apply unchanged. `from_sidx`
   marks the round. It is cleared again when the deferred-DFlash2 path (`dfr_lookup`) reloads `--lookup`'s draft.
2. Commit: windows that took an index draft are counted (`sidx_windows / sidx_drafts / sidx_ok`).
3. Request end, before `live.swap(consumed)`: `consumed[0, out0)` is the lead and `consumed[out0, end)` is the output.
   Both are inserted. A cancelled request's partial output is not inserted.
4. `--lookup-chain`: at serve start, the index is registered once in `extra_draft_sources()` as `SuffixIndexSource`.
   `propose_from_sources()` then also asks it after the MTP drafts (longest match wins, as for every source).

Not hooked: the gated-proposal path (`prepare_gated_window`, the probabilistic spec-gate mode), the
`--pipeline-windows 2` path, the CLI (non-serve) loop (one request: nothing to share across) and strata-poly (it
already has its own `--suffix-drafts`).

## Files changed

- `include/strata/spec/suffix_index.hpp`, `src/spec/suffix_index.cpp`: new (index, shared wrapper, restart log, env
  switch, DraftSource).
- `src/spec/suffix_index_test.cpp`: new unit tests.
- `src/program/generate.cpp`: the hooks above, all behind `sidx != nullptr` / `shared_suffix_index() != nullptr`.
- `CMakeLists.txt`: `suffix_index.cpp` in `strata_spec`; `suffix_index_test` registered next to
  `suffix_drafter_test` and in the cpu test list.
- `metal/CMakeLists.txt`, `sycl/CMakeLists.txt`: `suffix_index.cpp` in their `strata_spec` source lists (both build
  `generate.cpp` too).
- `tools/pancake/suffix_sim.py`, `tools/pancake/test_suffix_sim.py`: the simulator and its tests.
  `suffix_sim.py --dump FILE` writes each request's lead and output. `suffix_index_test --simulate FILE` replays the
  index-alone arm through the C++ `SuffixIndex`, to check it against the Python model and to time a lookup on real
  traffic.
- `bench/specs/24-a24-suffix-index.json`: the screen spec (plus a replay block).

## Simulator (tools/pancake/suffix_sim.py) and its numbers

The per-turn log (`pancake-turns.jsonl`, read by `replay_turns.py`) holds numbers only, so it cannot feed a drafter.
The simulator instead reads opencode's own session store, `~/.local/share/opencode/opencode.db`. It opens the file
read-only and reads only the session, message and part tables. The store holds the text of every agent step the
engine served.

How the simulator replays it:

- It rebuilds each step's prompt and output in the chat template's shape, with tool calls as
  `<tool_call><function=..><parameter=..>`. The system prompt and tool definitions are not stored, so they are left
  out.
- It tokenizes with the server's tokenizer (`Strata-data/packs/iq2_xs/tokenizer`).
- It replays the decode. Each round, a source drafts up to 11 tokens (`--lookup-max-t 12`). The round takes the
  drafted prefix that equals the recorded output, plus one model token.

The `lookup` arm emulates `SuffixDrafter` exactly: trigram to the 4 newest places, longest match >= 3 back to 64
tokens, newest on ties, the whole continuation.

Data: 2045 requests of `strata/qwen3.8-flash-next-iq2_xs` in 25 sessions, 10 of them sub-agent sessions. Output is
1,269,645 tokens. A full replay takes about 15 s.

Default run (k 11, 64 MiB = 1,048,576 tokens, min_prob 0.1, factor 1):

| arm | tok/round | rounds vs lookup | draft tokens offered | accepted | acceptance |
|---|---|---|---|---|---|
| lookup (today) | 2.212 | - | 2,517,195 | 697,198 | 27.7% |
| index alone | 1.633 | +35.4% | 1,361,261 | 493,991 | 36.3% |
| lookup + index, fallback | 2.261 | -2.16% | 2,574,595 | 709,564 | 27.6% |
| **lookup + index, longest** (default rule) | **2.267** | **-2.44%** | 2,476,066 | 711,176 | 28.7% |

Split by session type (rule longest): **main agent rounds -1.58%** (2.205 -> 2.240 tok/round); **sub-agents rounds
-7.53%** (2.254 -> 2.438 tok/round). With rule longest, the system offers 1.6% fewer draft tokens than lookup alone
and gets 2.0% more of them accepted, so the verify does less wasted work too. With rule fallback, 25,302 windows had
only an index draft, and they accepted 0.70 tokens a window.

Sweep (rounds of rule longest vs lookup alone; same data):

| setting | all | main | sub |
|---|---|---|---|
| default (64 MiB, min_prob 0.1, factor 1) | -2.44% | -1.58% | -7.53% |
| min_prob 0.3 | -2.41% | -1.56% | -7.46% |
| factor 2 | -2.67% | -1.77% | -8.05% |
| factor 4 | -2.73% | -1.81% | -8.18% |
| 4 MiB (65,536 tokens) | -1.00% | -0.60% | -3.38% |
| 16 MiB | -1.71% | -1.09% | -5.42% |
| 256 MiB | -2.47% | -1.61% | -7.61% |
| k 7 (window 8) | -2.49% | -1.67% | -7.38% |
| k 15 (window 16) | -2.35% | -1.45% | -7.75% |

Factor 2-4 gains about 0.3% more rounds, but its acceptance falls from 28.7% to 26.4-27.5% (more wasted verify
rows), so the paper's factor 1 stays the default. 64 MiB is enough: 256 MiB adds nothing on 10 days of traffic.

What the numbers are and are not:

- They are a ceiling for the lookup regime. They count rounds when every window costs what a one-token step costs.
- The MTP drafter is not modelled. In the engine, a lookup draft is taken only when the policy prefers it to the MTP
  window, and (greedy) only when the MTP's first guess agrees. So the real gain is a fraction of these numbers.
- The whole 10-day history sits in one index. A real server restarts. `--reset-gap MIN` starts the index empty after
  every pause longer than MIN minutes between two requests, as a stand-in for a restart:

  | index lifetime | resets | rounds (longest) all | main | sub |
  |---|---|---|---|---|
  | whole history (with `STRATA_SUFFIX_INDEX_FILE`) | 0 | -2.44% | -1.58% | -7.53% |
  | reset after 30-min pauses | 10 | -1.38% | -1.04% | -3.41% |
  | reset after 5-min pauses | 31 | -0.62% | -0.37% | -2.06% |

  So about half of the gain comes from outputs older than the last 30-min pause. That is why the restart log exists.
- The Python index uses exact dict keys. The C++ table hashes keys into `capacity` slots, so collisions use up some
  of the 64-place walk budget. The difference should be small, but I did not measure it. `suffix_index_test
  --simulate` (below) measures it.

## Tests run

- `python3 tools/pancake/suffix_sim.py --self-test`: `suffix_sim self-test: ok`.
- `python3 -m pytest -q tools/pancake/test_suffix_sim.py`: `4 passed, 1 warning` (the warning is not about this
  code).
- Full replays: the default run, a 10-point sweep, `--reset-gap 30` and `--reset-gap 5`. Numbers are above.
- **C++: NOT COMPILED, NOT RUN.** A GPU benchmark (`abtest.py bench/specs/09-prefill.json --tier screen`) kept
  `wait_quiet.sh` busy on 3 calls in a row (23:30-23:52), so I followed brief rule 3: no compiling, careful review
  instead. Things fixed during the review:
  - a nested `Params` struct with member initializers used as a default argument in its own class (moved to
    namespace scope as `SuffixIndexParams`)
  - `#include <mutex>` for `std::unique_lock` (the coordinator's clangd note)
  - a `reinterpret_cast` aliasing read replaced by `memcpy`
  - Windows `rename` over an existing file

  I also brace-checked the scope of the new `generate.cpp` code with a script: all the new locals are in the request
  block (depth 5, which never dips between their declaration and their last use).

Commands for the coordinator (host only, seconds):

```sh
cmake --build build-hip --target suffix_index_test && ./build-hip/suffix_index_test        # expect "suffix_index unit tests: OK"
# or without CMake:
g++ -std=c++20 -O2 -Wall -Wextra -Iinclude src/spec/suffix_index_test.cpp src/spec/suffix_index.cpp \
    src/spec/draft_source.cpp src/spec/suffix_drafter.cpp -pthread -o /tmp/suffix_index_test && /tmp/suffix_index_test
# a thread-sanitizer pass over the concurrency test:
g++ -std=c++20 -O1 -g -fsanitize=thread -Iinclude src/spec/suffix_index_test.cpp src/spec/suffix_index.cpp \
    src/spec/draft_source.cpp src/spec/suffix_drafter.cpp -o /tmp/suffix_index_tsan && /tmp/suffix_index_tsan
# the C++ index against the Python model on the recorded traffic (Python index arm: rounds 777281, offered 1361261,
# accepted 493991 at k 11 / 64 MiB / 0.1 / 1.0):
python3 tools/pancake/suffix_sim.py --dump /tmp/a24-turns.dump && /tmp/suffix_index_test --simulate /tmp/a24-turns.dump
```

## Spec to run

`bench/specs/24-a24-suffix-index.json`: arms `base`, `sidx` (`STRATA_SUFFIX_INDEX=1`) and `sidx-fallback`, on the
cases short and edit.

- Screen: `python tools/pancake/abtest.py bench/specs/24-a24-suffix-index.json --tier screen --out bench/results/<day>-a24-screen`.
- **Abtest caveat.** Each case sends 2 identical requests per server. The 2nd request is drafted whole from the 1st
  one's output: an exact repeat, which is a ceiling and not agent traffic. Judge on the first request of each case
  (the index then holds only the other case's output), or better on the replay.
- Replay (the real test):
  `python tools/pancake/replay_turns.py bench/specs/24-a24-suffix-index.json --out bench/results/<day>-a24-replay`
  replays the 11-09 agent session per arm. The index fills with the server's own replies turn by turn.

## Expected gain and risk

- Gain: a few % of decode rounds on agent traffic at best (offline ceiling -2.4% overall, -7.5% on sub-agent turns).
  About 0 on single short/edit requests. The paper reports up to ~10% on agentic edits.
- Host cost: one index lookup per lookup round. My estimate is a few microseconds (64 places at most, an O(m^2) vote
  over at most 64), but it is **not measured**. `suffix_index_test --simulate` prints it on the recorded traffic, and
  the per-request stderr line prints it in the engine. Memory is 40 MiB at the default cap, plus the log file (at most
  8 MiB) when `STRATA_SUFFIX_INDEX_FILE` is set.
- Risks:
  - The draft policy learns the acceptance of index windows together with `--lookup`'s, since both pass `sfx_match`
    as their confidence. If index drafts are accepted differently, the policy's lookup estimates shift a little.
  - Index windows bypass nothing: the MTP-agreement check and `graph_fits` still apply.
  - Rounding-level text change, as noted above.
- Default path: byte-identical when `STRATA_SUFFIX_INDEX` is unset. The only change is `const int k` becoming `int k`
  in the lookup block, plus dead locals.

## What is left

- Compile and run `suffix_index_test` (unit tests, TSan, `--simulate` against the Python numbers), then build the
  engine (HIP, and the CUDA / SYCL / Metal lists I touched), then run the screen and the replay (coordinator, 02:00).
- If it promotes: give index windows their own acceptance stats in `DraftPolicy` (a separate source id), and try
  `STRATA_SUFFIX_INDEX_FILE` on the daily server (run-pancake.sh), since restarts cost about half of the offline gain.
- Consider tree verification of the top-2 branches (the paper's speculation tree) if a tree-verify window ever
  exists.
- Coordinator: add the README §9 / RESEARCH_AND_FINDINGS §4 credit for strata-poly's SuffixIndex (FastStrata), and
  flip the MOD_PLAN A24 row from "planned" to "sim done, code done, untested on GPU".
