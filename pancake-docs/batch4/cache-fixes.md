# cache-fixes: two conversation-cache defects in opt-in paths (2026-11-11, branch dev/p16b/cache-fixes)

Files: `include/strata/core/conversation_cache.hpp`, `src/core/conversation_cache_test.cpp`, plus one arm in
`tools/pancake/percache_sim.py` and its test. No GPU and no engine run. The default path is unchanged: both fixes act only
when their switch is on (`STRATA_PARK_TTL_S`, `set_pin_evict_last`).

## 1. A15 park TTL: rank 3 short of bytes

**Defect (review, medium).** Rank 3 holds the entries whose last reply ended in a tool call and that are still inside the hold.
The shortest goes first. When make_room is short of BYTES, it calls pick_victim again and again. It drops the short held
entries one by one and then the long one too. Reproduced with the old header: a 60K-byte main agent parked first, three held
10K sub-agents, 8 slots, 100K budget, a new 60K image. Rule off: 1 eviction. Rule on: 4 evictions, 1 entry left.

**Fix.** make_room computes the shortfall (`bytes_ - (budget - held - incoming)`, 0 when only a slot is short). It passes it
through `evict_oldest(size_t shortfall = 0)` to `pick_victim(int* rank, size_t shortfall = 0)`. The defaults keep every
other caller as before (the RAM gate and QUIT in generate.cpp call `evict_oldest()`). In rank 3 the shortest entry whose
bytes cover the shortfall by itself goes. When none covers it, or the shortfall is 0, the shortest goes, as before.
Same case now: 1 eviction, the three sub-agents stay.
- "Smallest" is read as fewest tokens, the measure rank 3 already uses for the re-read cost. Bytes follow tokens.
- One difference from the task text: when slots AND bytes are short at the same time, the cover rule applies. Any eviction
  frees the one slot, so the covering entry fixes both in one eviction.

**Source check (Continuum, arXiv 2511.02230, section 5.2).** Under memory pressure Continuum unpins the pinned request of the
program with the LATEST arrival time, one at a time, until the new request fits. Size and reload cost set the TTL only, not
the victim. So the shortest-first rule and this cover rule are our adaptations, not Continuum's. Continuum's own order (newest
program first) would also drop the sub-agents one by one in the review's case.

**Tests.** `conversation_cache_test`:
- the review's case (rule off 1, rule on 1; the new test fails on the old header);
- an 85K image that no single entry covers: 2 short entries, then the main agent: 3 evictions, rule off also 3;
- slots only: the shortest goes and the main agent stays;
- property test: 1000 random put sequences (random budget, slots, sizes and lengths, all entries held). Before every put a
  rule-off copy gets the same image. When one entry covers the shortfall (3856 cases): exactly 1 eviction, never more than
  oldest-first. No byte shortfall (3024): the same evictions as rule-off. Neither (743): not asserted. There shortest-first
  can still evict more than oldest-first, as before.

`park_replay_sim --synthetic 6,4,3,1024` (the real ConversationCache, scaled size model). The default rows are identical.

| budget, rule | agent prompt | all prompts | evictions | restores |
| --- | --- | --- | --- | --- |
| 2048, A15 (ttl 30, hold 60), before | 23.7 s | 278.7 s | 56 | 19 |
| 2048, A15, after | 23.7 s | 257.0 s | 48 | 26 |
| 1280, A15, before / after | 52.0 s | 310.9 / 309.8 s | 59 / 59 | 16 / 16 |
| 1024, A15 | 78.9 s | 333.4 s (same) | 60 | 16 |

Synthetic, not a measurement. At 2048 MiB, A15's extra cost on all prompts (default 237.7 s) falls from +41.0 s to +19.3 s.
The agent's own prompt time stays the same.

**Not fixed (outside the task; tell me to do it).** The same 1-vs-4 pattern happens across ranks. A 60K main agent (rank 2) and
three 10K branches (rank 1 not idle, or rank 0 idle), a new 60K image: rule off 1 eviction, rule on 4. The ranks go strictly
lowest first, so the branches go one by one before the main agent, which goes anyway. A general fix would be: when the lower
ranks together cannot cover the shortfall, take the single entry that covers it. That changes the A15 order for ranks 0-2, so
it is a design decision.

## 2. Pin hazard: a pinned parked entry is evicted last, not never

**Defect (percache).** A parked conversation that holds a pinned shared prefix is never evicted. With `STRATA_AUTO_PIN=1` each
agent kind pins its own system prompt. A few sessions fill the slots or the budget with pinned entries, and every new
conversation hits "skip parking".

**Options modeled** (`percache_sim.py` on the real opencode history: 2,036 requests, 25 sessions; tokens read on continuing
requests; base 3,787,707):

| arm | tokens | saved | % of request time |
| --- | ---: | ---: | ---: |
| pin as coded (never evicted) | 5,214,099 | -513 s | -1.32% |
| pin-last (one pin per prefix family stays) | 5,214,099 | -513 s | -1.32% |
| pin-newest (only the family in use stays) | 4,205,608 | -151 s | -0.39% |
| **pin-evict-last (new)** | 3,784,039 | +1 s | +0.00% |
| pin-evict-last + the 4.0.1 flags | 1,749,377 | +734 s | +1.89% (= flags) |

"One pin per family" does not help: the families differ, so each family's last holder still stays for good. Evicting pinned
entries last is the simplest rule that is not worse than no pin. It is also no gain: here AUTO_PIN becomes safe, not useful.

**Fix.** `ConversationCache::set_pin_evict_last(bool)`, off by default (the old rule exactly). When on:
- make_room / evict_oldest take the unpinned entries first, in the usual order (the A15 ranks and the keep-hot rule included),
  then the oldest pinned one;
- the guarded donor still stays, and the guard's pre-check counts pinned entries as movable;
- truncate_oldest still never shrinks a pinned entry;
- the RAM gate's evict_oldest and QUIT's hand-off to the disk tier take pinned entries too, last.

**Switch: generate.cpp needs this change (not committed, the file is not mine).** It was compiled with the HIP configure:
`generate.cpp.o` built, no new warnings.

```diff
@@ -12701,6 +12701,13 @@ int main(int argc, char** argv) {
                                  "%lld ms (exact: eviction order only)\n", (long long) pt.ttl_ms, (long long) pt.hold_ms);
         }
+        // MOD_PLAN 71 / 90 (STRATA_AUTO_PIN=1, the server's opt-in auto-pin for Qwen): a parked conversation that holds a pinned
+        // shared prefix is evicted last, not never (ConversationCache::set_pin_evict_last).  Each agent kind pins its own system
+        // prompt, so with the old rule a few sessions filled the cache with pinned entries and no new one was parked.  Unset: as before.
+        if (const char* ap = std::getenv("STRATA_AUTO_PIN"); ap != nullptr && std::strcmp(ap, "1") == 0 && conversations.enabled()) {
+            conversations.set_pin_evict_last(true);
+            std::fprintf(stderr, "strata serve: conversation cache: a pinned shared prefix is evicted last, not never (STRATA_AUTO_PIN)\n");
+        }
```

DeepSeek auto-pins by default (`STRATA_DS4_AUTO_PIN`) and does not read `STRATA_AUTO_PIN`. So its default keeps the old rule.

**Tests.** `conversation_cache_test`: off by default; the hazard (old rule: the 4th pinned session is not parked; evict-last:
the oldest pinned one goes, the 4th is parked, the other pins are still found); unpinned before an older pinned one; bytes
(the old tiny-cache case now parks); evict_oldest alone; guard (old rule refuses, new rule evicts the other pinned entry);
truncation; with the park TTL on. `test_percache_sim.py`: the hazard and "pinned outlives unpinned" for the new arm.

**Careful with the planned GPU confirm** (MOD_PLAN "6 sessions ... order A B C D E F A B"). In that order the OLD rule wins.
It keeps A-D for good, so the returns to A and B hit. Evict-last behaves like base: A and B were evicted, so they are re-read.
The hazard shows when the NEWER sessions come back (A B C D E F E F) and on the real history (-1.32%). Use one of those, or
both orders, before reading the result.

## Runs

- CPU build (`-DSTRATA_BUILD_CONVERSATION_TESTS=ON`, pinned llama.cpp source): `ctest -R "conversation|session_save_reclaim|conv_cache"`
  6/6 passed (conversation_cache_test: 27,051 checks). The test also builds clean with `-Wall -Wextra`.
- `conversation_disk_tier_test` (built by hand, it needs STRATA_BUILD_TESTS): 97 checks passed. `park_replay_sim --selftest`: passed.
- `python -m pytest tools/pancake/test_percache_sim.py`: 8 passed; `percache_sim.py --self-test` OK.
- HIP compile-only (`-DSTRATA_ENABLE_HIP=ON`, gfx1100) of `CMakeFiles/strata.dir/src/program/generate.cpp.o`, with the patch above:
  built, no warning from the changed code.

## Risk

- Both fixes change eviction ORDER only. What an entry holds and what a restore reads stay the same, so outputs are exact.
- Fix 1 computes `bytes()` of each rank-3 entry per decision, only with A15 on (a few entries).
- Fix 2 lets a pinned prefix leave under pressure. A pinned prefix that leaves is re-read once by its next query, as for any
  eviction. That is what the model scores as +0.00%.
