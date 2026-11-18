"""Tests of the box-hours tools: chain_time.py, stop_rule_replay.py, power_calc.py, spec_cost.py and abtest.py's --bar-stop
rule (arm_status / decided_bar) and the 10K / 100K long-set cases.  No server, no GPU.

Run: python3 -I -m pytest -q tools/pancake/test_box_hours.py"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abtest as A  # noqa: E402
import chain_time as CT  # noqa: E402
import power_calc as PC  # noqa: E402
import spec_cost as SC  # noqa: E402
import stop_rule_replay as SR  # noqa: E402


def rec(arm, rep, vals, t="2026-11-18 01:00:00", tier="confirm"):
    return {"t": t, "rep": rep, "arm": arm, "attempt": 0, "clean": True, "noise": 1.0, "reclaim": 0.0,
            "rows": [{"case": "short", "decode_tps": v, "prompt_tps": 500.0, "prompt": 70, "gen": 512, "accept": None} for v in vals],
            "texts": {}, "opts": {"test_tier": tier, "outlier": "gate", "block": 0}}


def results(base, arm):
    return {"base": [rec("base", i, [v]) for i, v in enumerate(base)], "arm": [rec("arm", i, [v]) for i, v in enumerate(arm)]}


def test_arm_status_promote_drop_undecided():
    base = [100.0, 100.5, 99.5, 100.2]
    assert A.arm_status(results(base, [110.0, 110.3, 109.6, 110.1]), ["base", "arm"], ["short"], "arm") == "promote"
    assert A.arm_status(results(base, [100.1, 100.4, 99.7, 100.3]), ["base", "arm"], ["short"], "arm") == "drop"
    assert A.arm_status(results(base, [90.0, 90.2, 89.9, 90.1]), ["base", "arm"], ["short"], "arm") == "drop"      # below the -3% floor
    assert A.arm_status(results(base, [96.0, 106.0, 98.0, 104.0]), ["base", "arm"], ["short"], "arm") is None      # noisy
    assert A.arm_status(results(base[:1], [110.0]), ["base", "arm"], ["short"], "arm") is None                      # one value: no interval


def test_decided_bar_needs_every_arm():
    r = results([100.0, 100.5, 99.5, 100.2], [110.0, 110.3, 109.6, 110.1])
    assert A.decided_bar(r, ["base", "arm"], ["short"])
    r["arm2"] = [rec("arm2", i, [v]) for i, v in enumerate([96.0, 106.0, 98.0, 104.0])]
    assert not A.decided_bar(r, ["base", "arm", "arm2"], ["short"])


def test_bar_stop_flag_parses():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bar-stop", type=float, nargs="?", const=-1.0, default=None)
    assert ap.parse_args([]).bar_stop is None
    assert ap.parse_args(["--bar-stop"]).bar_stop == -1.0
    assert ap.parse_args(["--bar-stop", "3"]).bar_stop == 3.0


def test_long_set_cases():
    for c, n in (("long10k", 10000), ("long32k", 32000), ("long64k", 64000), ("long100k", 100000), ("longdec100k", 100000)):
        assert A.LONG_TARGET[c] == n and A.is_long(c)
    assert A.metric("longdec100k") == "decode_tps" and A.metric("long100k") == "prompt_tps"
    assert A.MAXTOK["long100k"] == 32 and A.MAXTOK["longdec100k"] == 256


def test_chain_steps_and_split(tmp_path):
    log = ["START 2026-11-18 x", "START 9-spec 2026-11-17_23:00:00", "END 9-spec exit=0 00:30:00", "START 10-spec 2026-11-18_00:33:00",
           "END 10-spec exit=1 00:40:00", "noise"]
    steps = CT.chain_steps(log)
    assert [(s[0], s[3]) for s in steps] == [("9-spec", 0), ("10-spec", 1)]
    assert (steps[0][2] - steps[0][1]).total_seconds() == 5400
    row = {"prompt": 100, "prompt_tps": 100.0, "gen": 100, "decode_tps": 50.0}
    assert CT.measured_seconds(row) == pytest.approx(3.0)
    (tmp_path / "win-chainZ.log").write_text("\n".join(log))
    rows = [rec("a", 0, [100.0], t="2026-11-17 23:10:00"), rec("a", 1, [100.0], t="2026-11-17 23:20:00")]
    (tmp_path / "win1-9-spec.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    d = CT.breakdown(str(tmp_path))["Z"]
    assert d["steps"] == 2 and d["starts"] == 2 and d["idle"] == 180 and d["abtest_wall"] > 0
    assert "ALL" in CT.fmt({"Z": d})


def test_stop_rule_replay_full_run_and_early(tmp_path):
    recs = []
    for i in range(6):
        recs += [rec("base", i, [100.0 + (i % 2) * 0.2]), rec("arm", i, [110.0 + (i % 2) * 0.2])]
    p = tmp_path / "x.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs))
    r = SR.replay_file(str(p), 2.0, 3.0)
    assert r["full"] == {"arm": "promote"}
    assert r["bar"]["stop_round"] == 2 and r["bar"]["same_as_full"] and r["bar"]["starts_saved"] == 8
    assert SR.main([str(p), "--min-reps", "2"]) == 0


def test_power_calc_pools_and_rounds(tmp_path):
    recs = []
    for i in range(6):
        recs += [rec("base", i, [100.0 + (i % 2)]), rec("arm", i, [105.0 - (i % 2)])]
    p = tmp_path / "x.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs))
    assert PC.main([str(p)]) == 0
    assert PC.rounds_needed(1.0, 2.0) is not None and PC.rounds_needed(10.0, 0.5) is None
    assert PC.family("longdec32k") == "longdec" and PC.family("long64k") == "long-prompt" and PC.family("pep8") == "edit"


def test_spec_cost_matches_measured_screens():
    spec = {"arms": {"a": {}, "b": {}, "c": {}, "d": {}}, "cases": ["short", "edit", "long1k", "long3k"]}
    h = SC.screen_hours(spec, drift=True)
    assert 0.14 < h < 0.2       # spec 126 (4 arms + drift, short / edit / long1k / long3k): modelled 0.15 h, measured 0.151 h
    assert SC.screen_hours({"arms": {"a": {}}, "cases": ["short", "long32k"]}) < SC.screen_hours(
        {"arms": {"a": {}}, "cases": ["short", "long32k"], "prefill": True})
    assert SC.confirm_hours(spec, 1) > 0


def test_spec_cost_64k_is_per_model():
    """T-943: long64k 27 s (flashnext IQ2_XS, 2405.7 tok/s) and longdec64k 101 s prefill (Qwen3.6 Q4_K_XL, 643 tok/s) are two models."""
    flash = {"base": "configs/pancake/flashnext-iq2_xs-fast-hip.json"}
    q36 = {"base": "configs/pancake/qwen36-ud-q4_k_xl-q36.json"}
    assert abs(SC.case_s(flash, "long64k") - 64000 / 2405.7) < 0.5
    assert abs(SC.case_s(q36, "long64k") - 65149 / 643.0) < 0.5
    assert abs(SC.case_s(q36, "longdec64k") - (65149 / 643.0 + 256 / 21.0)) < 0.5
    assert SC.case_s(flash, "longdec64k") < SC.case_s(q36, "longdec64k") / 2
    assert SC.case_s(flash, "short") == SC.case_s(q36, "short")
