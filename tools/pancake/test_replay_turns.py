"""Tests for replay_turns.py without a server: conversation split, side-request repeats, the synthetic session and the
sizing loop (estimate counter, so no tokenizer is needed).  python -m pytest tools/pancake/test_replay_turns.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_turns as RT  # noqa: E402


def row(t, total, reused, out, think=True, max_new=32768, dur=1.0):
    return {"time": t, "duration_s": dur, "prompt_total": total, "reused": reused, "output_tokens": out,
            "max_new": max_new, "reasoning_tokens": out if think else 0, "prompt_ms": 100.0}


def test_split_keeps_cold_rereads_in_their_conversation():
    rows = [row(0, 65, 0, 4, think=False, max_new=60), row(10, 70082, 0, 1397), row(70, 71788, 71478, 273),
            row(80, 34261, 0, 484), row(120, 34777, 34744, 191),
            row(130, 69, 0, 400, max_new=400), row(135, 69, 64, 400, max_new=400),       # a side text and its repeat
            row(140, 35900, 0, 300),                                                       # cold re-read of B
            row(150, 73, 0, 158, think=False, max_new=700), row(155, 71, 66, 380, max_new=700)]   # off/on pair
    plan = RT.plan_from_rows(rows, 8192, 16384, 2.0, "output")
    assert [e["kind"] for e in plan] == ["side", "agent", "agent", "agent", "agent", "side", "side", "agent", "side", "side"]
    assert [e["conv"] for e in plan if e["kind"] == "agent"] == ["A", "A", "B", "B", "B"]
    assert [e["first"] for e in plan if e["kind"] == "agent"] == [True, False, True, False, False]
    assert plan[6]["repeat"] and plan[6]["conv"] == plan[5]["conv"]
    assert plan[9]["repeat"] and plan[9]["conv"] == plan[8]["conv"]      # thinking off counts 2 tokens more
    assert plan[1]["max_new"] == 1397 and plan[1]["gap_s"] == 2.0 and not plan[0]["thinking"]


def test_synthetic_shape():
    plan = RT.plan_synthetic(gap_s=0.5)
    agent = [e for e in plan if e["kind"] == "agent"]
    assert len(agent) == 18 and agent[0]["target"] == 40000 and agent[-1]["target"] == 62000
    assert len(plan) == 48 and all(e["gap_s"] == 0.5 for e in plan)
    assert sum(e["kind"] == "side" and e["repeat"] for e in plan) == 15


def test_sizing_hits_the_recorded_prompt():
    C = RT.Counter("/nonexistent")              # the estimate counter
    assert not C.exact
    convs = RT.Conversations(C, RT.Corpus(RT.ROOT))
    plan = RT.plan_synthetic()
    for e in plan[:12]:
        msgs, n = convs.build(e)
        assert abs(n - e["target"]) <= RT.TOL + 1, (e, n)
        convs.placeholder_reply(e)
    # an agent turn extends the previous one: same messages, then the reply and the tool result
    first = convs.msgs["S"]
    assert first[0]["role"] == "system" and first[-1]["content"].startswith("<tool_response>")


def test_engine_lines():
    text = ("strata serve: conversation cache: skip parking (snapshot 1286 MiB exceeds available budget)\n"
            "strata serve: conversation cache: restored 66 tokens (checkpoint) in 38.5 ms; parked=4 bytes=1\n"
            "strata serve: prompt 4084 tokens = 4079 reused + 5 read in 200 ms (25.0 tok/s), 1 generated in 43 ms "
            "(23.3 tok/s), drafts accepted 0 of 0, 1 checkpoints\n"
            "strata serve: conversation cache: parked 4084 tokens in 129.8 ms; parked=4 bytes=1 evictions=19 "
            "snapshot_bytes=298508196 reused_kv_bytes=62216064\n")
    p = RT.parse_engine(text)
    assert p["prompt_lines"][0]["reused"] == 4079 and p["prompt_lines"][0]["read_ms"] == 200
    assert p["events"] == ["skip (snapshot 1286 MiB exceeds available budget)", "restored 66 (checkpoint, 38 ms)",
                           "parked 4084 (284 MiB)"] and p["evictions"] == 19


def test_shared_system_prompt_is_opt_in():
    C = RT.Counter("/nonexistent")
    plan = RT.plan_synthetic()
    side = max((e for e in plan if e["kind"] == "side" and not e["repeat"]), key=lambda e: e["target"])   # room for the shared text
    agent = plan[[e["kind"] for e in plan].index("agent")]
    try:
        RT.SHARED_SYSTEM = 0
        off = RT.Conversations(C, RT.Corpus(RT.ROOT))
        off_agent, _ = off.build(agent)
        off_side, _ = off.build(side)
        assert off.shared is None and off_side[0]["role"] == "user"          # the default: nothing changed
        RT.SHARED_SYSTEM = min(600, side["target"] // 2)   # the shared system text must fit in the shortest side request
        on = RT.Conversations(C, RT.Corpus(RT.ROOT))
        on_agent, n1 = on.build(agent)
        on_side, n2 = on.build(side)
        assert on_agent[0] == on_side[0] and on_agent[0]["role"] == "system" and on.shared
        assert abs(n1 - agent["target"]) <= RT.TOL + 1 and abs(n2 - side["target"]) <= RT.TOL + 1
    finally:
        RT.SHARED_SYSTEM = 0
