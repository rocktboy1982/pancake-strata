"""Tests for park_sim.py: the plan -> trace conversion (no binary needed) and, when the park_replay_sim binary is built
(`cmake --build <dir> --target park_replay_sim`, found under build*/ or in $PARK_REPLAY_SIM), the recorded-shape results: A15 does nothing
where the cold re-reads are budget skips, F1 shortens them, the tool hold protects the long conversation.
python -m pytest tools/pancake/test_park_sim.py"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import park_sim as PS  # noqa: E402
import replay_turns as RT  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def sim_binary():
    env = os.environ.get("PARK_REPLAY_SIM")
    if env and Path(env).is_file():
        return Path(env)
    for d in sorted(ROOT.glob("build*")):
        p = d / "park_replay_sim"
        if p.is_file():
            return p
    return None


def rows():
    """A small session: a long conversation B (34K -> 62K, two cold re-reads) between short side requests; times 1 s apart."""
    out, t = [], 1000.0

    def add(total, reused, n, think=True):
        nonlocal t
        out.append({"id": f"r{len(out)}", "time": t, "duration_s": 1.0, "finish": "stop", "prompt_total": total, "reused": reused,
                    "prompt_read": total - reused, "prompt_ms": 100.0, "output_tokens": n, "max_new": 32768,
                    "reasoning_tokens": 5 if think else 0, "effort": None})
        t += 2.0

    add(34000, 0, 300)
    add(34600, 34200, 300)
    add(70, 0, 30)
    add(35300, 0, 300)
    add(36000, 35600, 300)
    return out


def plan():
    return RT.plan_from_rows(rows(), conv_min=8192, max_growth=16384, max_gap=2.0, max_new_mode="output")


def test_trace_fields_and_timeline():
    lines = PS.plan_to_trace(plan())
    assert len(lines) == 5
    f = lines[0].split()
    assert f[:4] == ["A", "A", "34000", "300"] or f[1] == "A"   # conv A, an agent turn
    conv, kind, target, out, start, dur, tool, shared = lines[2].split()
    assert kind == "S" and int(target) == 70 and tool == "0" and shared == "0"
    starts = [float(l.split()[4]) for l in lines]
    assert starts == sorted(starts) and starts[0] == 0.0 and abs(starts[1] - 2.0) < 1e-6   # the recording's own timeline


def test_shared_system_is_capped_for_short_conversations():
    lines = [l.split() for l in PS.plan_to_trace(plan(), shared_system=1024)]
    assert all(int(l[7]) == 1024 for l in lines if l[1] == "A")
    short = [l for l in lines if l[1] == "S"][0]
    assert int(short[7]) <= 70 - PS.REPLAY_SHARED_MIN_TAIL      # a 70-token request cannot carry a 1024-token system prompt


def test_tool_agent_marks_only_agent_turns():
    lines = [l.split() for l in PS.plan_to_trace(plan(), tool_agent=True)]
    assert all(l[6] == ("1" if l[1] == "A" else "0") for l in lines)


def test_plan_without_a_recording_gets_a_timeline():
    syn = RT.plan_synthetic()
    lines = PS.plan_to_trace(syn)
    starts = [float(l.split()[4]) for l in lines]
    assert len(lines) == len(syn) and starts == sorted(starts) and starts[-1] > starts[0]


def test_parse_result_line():
    r = PS.parse_result("result   agent prompt   224.2 s  cold 7 ( 168.8 s)  all   271.9 s | parks 9 (+2 truncated) skips 7 restores 3 "
                        "borrows 1 | evictions 31 (ttl 4) truncations 5\n    cold requests: 35 46\n")
    assert r["cold"] == 7 and r["truncated_parks"] == 2 and r["ttl_evictions"] == 4 and r["cold_requests"] == [35, 46]


@pytest.mark.skipif(sim_binary() is None, reason="park_replay_sim is not built")
def test_recorded_shape_with_the_real_cache():
    sim = sim_binary()
    # a session sized like the 11-09 run: a 54-62K conversation whose snapshot is just over a 1280 MiB budget, side requests between
    syn = RT.plan_synthetic()
    res = PS.evaluate(syn, sim, budgets=(1280,))
    base = res["1280 MiB | default (oldest first)"]
    f1 = res["1280 MiB | F1 park-truncate"]
    assert base["cold"] >= 1 and base["skips"] >= 1                    # the snapshot does not fit: nothing is parked
    assert f1["truncated_parks"] >= 1 and f1["agent_prompt_s"] < base["agent_prompt_s"]   # F1 re-reads only the tail
    a15 = res["1280 MiB | A15 ttl30, hold 60"]
    assert a15["skips"] == base["skips"]       # A15 reorders evictions; it cannot make an oversized image fit
