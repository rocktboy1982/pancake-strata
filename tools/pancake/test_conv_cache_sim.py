"""conv_cache_sim.py: the budget rules against src/core/conversation_memory_test.cpp, the cache rules on tiny synthetic sessions.
No model, no GPU, no server.

Run: python -m pytest tools/pancake/test_conv_cache_sim.py"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conv_cache_sim as S  # noqa: E402

GIB = 1024


def req(conv, tot, gen=100, agent=True):
    return {"conv": conv, "tot": tot, "gen": gen, "agent": agent, "rec_reused": 0, "rec_s": 0.0}


def hows(reqs, p):
    return [r["how"] for r in S.simulate(reqs, p)["rows"]]


# ------------------------------------------------------------------ the budget rules (same numbers as the C++ test)
def test_ram_rule_matches_the_cpp_test():
    f = S.ram_rule_budget_mib
    assert f(16 * GIB, 12 * GIB, 2560, 512, 8192) == 3584
    assert f(13 * GIB, 12 * GIB, 2560, 512, 8192) == 0        # 512 MiB left: off
    assert f(12 * GIB, 12 * GIB, 2560, 512, 8192) == 0        # below keep, no underflow
    assert f(8 * GIB, 0, 2560, 512, 8192) == 5120             # no keep: the floor is kept
    assert f(8 * GIB, 1 * GIB, 2560, 512, 8192) == 5120
    assert f(40 * GIB, 12 * GIB, 2560, 512, 8192) == 8192     # capped
    assert f(40 * GIB, 12 * GIB, 2560, 512, 0) == 28160       # cap 0 = none
    assert f(16 * GIB, 12 * GIB, 2560, -5, 8192) == 4096      # a negative margin is 0
    assert f(18 * GIB, 12 * GIB) == 5632                      # this box after the arena: 18 GiB available, keep 12


def test_legacy_rule_matches_the_cpp_test():
    f = S.legacy_budget_mib
    assert f(2 * GIB) == 0 and f(4 * GIB) == 0
    assert f(8 * GIB) == 2048 and f(20 * GIB) == 7168 and f(40 * GIB) == 8192


# ------------------------------------------------------------------ the cache
def test_one_conversation_is_warm_after_its_first_turn():
    reqs = [req("A", 20000), req("A", 20300), req("A", 20700)]
    assert hows(reqs, S.Params(4096)) == ["cold", "warm", "warm"]


def test_a_side_request_parks_the_conversation_and_it_comes_back():
    reqs = [req("A", 30000), req("s0", 70, agent=False), req("A", 30300)]
    res = S.simulate(reqs, S.Params(4096))
    assert [r["how"] for r in res["rows"]] == ["cold", "cold", "restore"]
    assert res["rows"][2]["read"] == 30300 - (30000 + 100)    # only the new tail is read (the reply was part of the parked prefix)
    assert res["rows"][2]["cost_s"] < 0.2 * res["rows"][0]["cost_s"]


def test_a_snapshot_over_the_budget_is_skipped_and_the_next_turn_is_cold():
    reqs = [req("A", 60000), req("s0", 70, agent=False), req("A", 60400)]
    p = S.Params(1280)                                        # a 60K snapshot is ~1.4 GiB
    assert p.snapshot_mib(60100, 3) > 1280
    res = S.simulate(reqs, p)
    assert res["skipped"] and hows(reqs, p) == ["cold", "cold", "cold"]
    assert hows(reqs, S.Params(2048)) == ["cold", "cold", "restore"]


def test_short_side_requests_evict_the_agent_conversation_unless_min_tokens():
    sides = [req("s%d" % i, 100 + i, agent=False) for i in range(5)]
    reqs = [req("A", 30000)] + sides + [req("A", 30300)]
    assert hows(reqs, S.Params(8192, slots=4))[-1] == "cold"                       # 4 slots: the 4th side copy pushes out A, the oldest
    assert hows(reqs, S.Params(8192, slots=4, min_tokens=8192))[-1] == "restore"   # the side copies are never made
    assert hows(reqs, S.Params(8192, slots=8))[-1] == "restore"                    # or the slots are enough


def test_the_incoming_copy_is_held_while_the_outgoing_one_is_parked():
    # B (parked) comes back while A is live: A must fit beside B's copy, which is still in RAM
    a, b = 40000, 40000
    p = S.Params(budget_mib=1500)
    one = p.snapshot_mib(a + 100, max(1, a // S.CKPT_EVERY))
    assert one < 1500 < 2 * one                                # each fits alone, both do not fit together
    reqs = [req("A", a), req("B", b), req("A", a + 300), req("B", b + 300)]
    got = hows(reqs, p)
    assert got[:2] == ["cold", "cold"]
    assert got[2] == "restore"                                 # A was parked when B went live (B was not parked yet)
    assert got[3] == "cold"                                    # A's return could not park B beside A's copy


def test_two_large_conversations_need_room_for_both():
    # a main agent and a sub-agent, both over 8K tokens, taking turns (the case 2048 MiB cannot hold)
    reqs = [req("A", 50000), req("B", 45000)]
    for i in range(1, 5):
        reqs += [req("A", 50000 + 400 * i), req("B", 45000 + 400 * i)]
    small = S.simulate(reqs, S.Params(2048, min_tokens=8192))
    big = S.simulate(reqs, S.Params(S.ram_rule_budget_mib(18 * GIB, 12 * GIB), min_tokens=8192))
    assert sum(r["how"] == "cold" for r in small["rows"]) > 2
    assert sum(r["how"] == "cold" for r in big["rows"]) == 2   # only the two first reads
    assert S.summarize(big)["prompt_s"] < 0.5 * S.summarize(small)["prompt_s"]


def test_min_tokens_costs_the_small_conversations_their_reuse():
    side = [req("s0", 3000, agent=False), req("x", 70, agent=False), req("s0", 3000, agent=False)]
    cheap = S.summarize(S.simulate(side, S.Params(4096)))
    strict = S.summarize(S.simulate(side, S.Params(4096, min_tokens=8192)))
    assert strict["side_s"] > cheap["side_s"]


def test_cold_reread_counts_only_non_first_agent_turns_over_the_limit():
    reqs = [req("A", 30000), req("s0", 70, agent=False), req("A", 30300)]
    s = S.summarize(S.simulate(reqs, S.Params(1)))             # a 1 MiB cache parks nothing: the return is cold
    assert s["cold"] == 1 and s["cold_s"] == pytest.approx(30300 * 0.414 / 1000)


# ------------------------------------------------------------------ from a log
def turn_rows(n_cold=3):
    t, rows = 1000.0, []

    def add(tot, reused, out=300, ms=500.0):
        nonlocal t
        rows.append({"time": t, "duration_s": 5.0, "prompt_total": tot, "reused": reused, "prompt_read": tot - reused,
                     "prompt_ms": ms, "output_tokens": out, "max_new": 32768, "reasoning_tokens": 10})
        t += 6.0
    add(40000, 0, ms=17000.0)
    for i in range(n_cold):
        add(70, 0, out=50, ms=700.0)                          # a side request takes the live session
        add(40500 + 600 * i, 0, ms=17500.0)                   # the agent's turn, read from token 0 (recorded)
    return rows


def test_requests_from_log_groups_the_agent_turns():
    reqs = S.requests_from_log(turn_rows())
    agents = [r for r in reqs if r["agent"]]
    assert len(agents) == 4 and {r["conv"] for r in agents} == {"A"}
    assert [r["rec_reused"] for r in agents] == [0, 0, 0, 0] and reqs[0]["rec_s"] == pytest.approx(17.0)


def test_main_prints_the_scenario_table(tmp_path, capsys):
    f = tmp_path / "turns.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in turn_rows()) + "\n")
    S.main(["--turns", str(f), "--session", "s0"])
    out = capsys.readouterr().out
    assert "S0 legacy" in out and "S3 4.0.1 flags" in out and "RAM rule (86), 5632 MiB" in out
    S.main(["--turns", str(f), "--session", "s0", "--budget", "2048", "--min-tokens", "8192"])
    assert "budget 2048 MiB, 4 slots, min-tokens 8192" in capsys.readouterr().out
