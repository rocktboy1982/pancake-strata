"""Unit tests for task_gate.py: schema checks, argument matching, code extraction, the default task set's checkers, the
statistics and the verdict, and the report from stored runs.  No server, no GPU.
Run: python -m pytest tools/pancake/test_task_gate.py   (or python tools/pancake/test_task_gate.py)"""
import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task_gate as T  # noqa: E402

DEFAULT = T.ROOT / T.DEFAULT_TASKS


def test_default_task_set_is_valid_and_selftests():
    _, tasks = T.load_tasks(DEFAULT)
    assert 10 <= len(tasks) <= 20
    assert {t["kind"] for t in tasks} == set(T.KINDS)
    assert T.validate_tasks(tasks) == []
    assert T.selftest(tasks) == []


def test_validate_tasks_finds_mistakes():
    _, tasks = T.load_tasks(DEFAULT)
    bad = json.loads(json.dumps(tasks))
    bad[0]["expect"].pop("test")
    bad.append(dict(bad[1]))                                   # duplicate id
    tc = next(t for t in bad if t["kind"] == "tool_call" and t["expect"]["calls"])
    tc["expect"]["calls"][0]["args"]["nope"] = 1               # argument that is not a parameter
    probs = "\n".join(T.validate_tasks(bad))
    assert "expect.test missing" in probs and "duplicated" in probs and "not a parameter" in probs


def test_selftest_catches_a_broken_checker():
    _, tasks = T.load_tasks(DEFAULT)
    t = json.loads(json.dumps(next(t for t in tasks if t["id"] == "qa-capital")))
    t["expect"]["answer_re"] = "Sydney"
    assert T.selftest([t])


def test_schema():
    s = {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1, "maximum": 3}, "u": {"enum": ["a", "b"]},
                                          "xs": {"type": "array", "items": {"type": "string"}}},
         "required": ["n"], "additionalProperties": False}
    assert T.validate_schema({"n": 2, "u": "a", "xs": ["x"]}, s) == []
    assert T.validate_schema({"n": 2.0}, s) == []
    assert T.validate_schema({}, s)
    assert T.validate_schema({"n": 4}, s)
    assert T.validate_schema({"n": True}, s)
    assert T.validate_schema({"n": 1, "u": "c"}, s)
    assert T.validate_schema({"n": 1, "extra": 1}, s)
    assert T.validate_schema({"n": 1, "xs": [1]}, s)
    assert T.schema_problems({"type": "object", "properties": {}, "required": ["x"]})


def test_arg_match():
    assert T.arg_match("Lisbon", " lisbon ")
    assert not T.arg_match("Lisbon", "Porto")
    assert T.arg_match({"re": "^gym$"}, "Gym")
    assert T.arg_match({"min": 1, "max": 3}, 2) and not T.arg_match({"min": 1, "max": 3}, 4)
    assert T.arg_match({"set": ["a", "b"]}, ["b", "a"]) and not T.arg_match({"set": ["a", "b"]}, ["a"])
    assert T.arg_match({"any_of": [1, 2]}, 2)
    assert T.arg_match({"hour": 6}, {"hour": 6, "minute": 30}) and not T.arg_match({"hour": 6}, "06:30")
    assert T.arg_match(45, 45.0) and not T.arg_match(45, "45")
    assert T.arg_match(True, True) and not T.arg_match(True, 1)


def test_tool_call_grading():
    _, tasks = T.load_tasks(DEFAULT)
    by = {t["id"]: t for t in tasks}
    t = by["tc-single"]
    ok = T.as_message({"calls": [{"name": "get_weather", "args": {"city": "lisbon", "unit": "celsius"}}]})
    assert T.grade(t, ok) == {"ok": True, "valid": True, "reason": ""}
    wrong_city = T.as_message({"calls": [{"name": "get_weather", "args": {"city": "Porto", "unit": "celsius"}}]})
    g = T.grade(t, wrong_city)
    assert not g["ok"] and g["valid"]                      # well formed but the wrong call
    broken = {"tool_calls": [{"function": {"name": "get_weather", "arguments": "{\"city\": "}}]}
    g = T.grade(t, broken)
    assert not g["ok"] and not g["valid"]
    text_call = {"content": "<tool_call>{\"name\": \"get_weather\"}</tool_call>"}
    assert not T.grade(t, text_call)["valid"]
    assert not T.grade(t, {"content": "It is sunny."})["valid"]
    nocall = by["tc-no-call"]
    assert T.grade(nocall, {"content": "144"})["ok"]
    assert not T.grade(nocall, {"content": "no idea"})["ok"]
    assert T.grade(by["tc-parallel"], T.as_message({"calls": [{"name": "get_weather", "args": {"city": "Madrid"}},
                                                              {"name": "get_weather", "args": {"city": "Oslo"}}]}))["ok"]


def test_extract_code():
    assert T.extract_code("hi\n```python\nx = 1\n```\n") == "x = 1\n"
    two = "old\n```python\ndef f(): return 1\n```\nnew\n```python\ndef f(): return 2\n```"
    assert "return 2" in T.extract_code(two)
    assert T.extract_code("no code here") is None
    assert "def g" in T.extract_code("def g():\n    return 1\n")
    assert T.extract_code("<think>```python\ndef a(): pass\n```</think>nothing") is None


def test_run_python_limits():
    rc, _ = T.run_python("print(1)")
    assert rc == 0
    rc, _ = T.run_python("assert False")
    assert rc not in (0, None)
    rc, _ = T.run_python("while True: pass", timeout=1)
    assert rc != 0                                          # CPU limit kills it (or the wall timeout: None)


def test_wilson():
    lo, hi = T.wilson(0, 0)
    assert lo != lo
    lo, hi = T.wilson(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60
    assert T.wilson(10, 10)[1] == 1.0


def rates(p_by_task, reps, seed):
    rng = random.Random(seed)
    return {t: [1 if rng.random() < p else 0 for _ in range(reps)] for t, p in p_by_task.items()}


def test_verdict_rule():
    assert T.verdict(0.01, 0.10, 0.05) == "BETTER"
    assert T.verdict(-0.04, 0.08, 0.05) == "PASS"
    assert T.verdict(-0.20, -0.07, 0.05) == "REJECT"
    assert T.verdict(-0.15, 0.02, 0.05) == "INCONCLUSIVE"
    assert T.worst("PASS", "REJECT") == "REJECT" and T.worst("BETTER", "INCONCLUSIVE") == "INCONCLUSIVE"


def test_boot_diff_separates_real_loss_from_noise():
    tasks = {f"t{i}": 0.7 for i in range(18)}
    same_a, same_b = rates(tasks, 20, 1), rates(tasks, 20, 2)
    d, lo, hi, n = T.boot_diff(same_a, same_b, 1000, 0)
    assert n == 18 and lo < 0 < hi and T.verdict(lo, hi, 0.05) != "REJECT"
    worse = rates({t: 0.3 for t in tasks}, 20, 3)
    d, lo, hi, n = T.boot_diff(worse, same_b, 1000, 0)
    assert d < -0.3 and hi < -0.05 and T.verdict(lo, hi, 0.05) == "REJECT"
    # deterministic runs (every task 0 or 1): the task-resampling interval stays honest, the repetitions-only one collapses
    det_base = {f"t{i}": [1] * 5 for i in range(16)}
    det_arm = {f"t{i}": [1] * 5 for i in range(15)}
    det_arm["t15"] = [0] * 5
    d, lo, hi, _ = T.boot_diff(det_arm, det_base, 1000, 0, resample_tasks=True)
    assert lo < d < hi and hi >= 0 - 1e-12 and T.verdict(lo, hi, 0.05) == "INCONCLUSIVE"
    d2, lo2, hi2, _ = T.boot_diff(det_arm, det_base, 1000, 0, resample_tasks=False)
    assert lo2 == hi2 == d2
    assert T.boot_diff({"a": [1]}, {"b": [1]}) is None


def mkruns(arm, p_by_task, reps, seed, kind="qa", valid=None):
    rng = random.Random(seed)
    out = []
    for r in range(reps):
        for t, p in p_by_task.items():
            ok = rng.random() < p
            out.append({"arm": arm, "rep": r, "task": t, "kind": kind, "ok": ok, "valid": (ok if valid else None),
                        "reason": "", "elapsed": 2.0 + (0 if ok else 5.0), "completion_tokens": 10, "error": None, "message": {}})
    return out


def _boot_diff_two_stage_as_it_was(arm, base, boots, seed, resample_tasks=True):
    """boot_diff before resample_reps existed (the default path must keep giving these numbers)."""
    tasks = [t for t in arm if t in base and arm[t] and base[t]]
    rng = random.Random(seed)
    mean = lambda xs: sum(xs) / len(xs)
    d = mean([mean(arm[t]) - mean(base[t]) for t in tasks])
    ds = []
    for _ in range(boots):
        pick = [rng.choice(tasks) for _ in tasks] if resample_tasks else tasks
        s = 0.0
        for t in pick:
            a, b = arm[t], base[t]
            s += mean(rng.choices(a, k=len(a))) - mean(rng.choices(b, k=len(b)))
        ds.append(s / len(pick))
    ds.sort()
    return d, ds[int(0.025 * boots)], ds[min(boots - 1, int(0.975 * boots))], len(tasks)


def test_boot_diff_default_is_unchanged_and_tasks_only_is_the_narrower_one_stage_interval():
    rng = random.Random(3)
    ps = [0.97] * 10 + [0.75] * 5 + [0.4] * 3
    base = {f"t{i}": [1 if rng.random() < p else 0 for _ in range(5)] for i, p in enumerate(ps)}
    arm = {f"t{i}": [1 if rng.random() < p else 0 for _ in range(5)] for i, p in enumerate(ps)}
    for rt in (True, False):
        assert T.boot_diff(arm, base, 300, 5, rt) == _boot_diff_two_stage_as_it_was(arm, base, 300, 5, rt)
    two = T.boot_diff(arm, base, 1000, 5)
    one = T.boot_diff(arm, base, 1000, 5, resample_reps=False)
    assert one[0] == two[0] and one[3] == two[3]                       # the same point estimate
    assert one[2] - one[1] < two[2] - two[1]                           # repetition noise counted once, not twice
    assert one[1] <= one[0] <= one[2]
    # tasks held fixed: the repetitions are the only noise, so resample_reps=False cannot switch them off
    assert T.boot_diff(arm, base, 200, 5, resample_tasks=False, resample_reps=False) == \
        T.boot_diff(arm, base, 200, 5, resample_tasks=False)
    runs = [{"task": f"t{i}", "ok": bool(v), "valid": None, "elapsed": 1.0, "kind": "qa"}
            for i, p in enumerate(ps) for v in base[f"t{i}"]]
    rep = T.build_report(["a"], {"a": runs}, 0.05, 0.03, resample_reps=False)
    assert "over tasks only" in T.format_report(rep)
    assert "over tasks and repetitions" in T.format_report(T.build_report(["a"], {"a": runs}, 0.05, 0.03))


def test_report_from_runs_dir():
    tasks = {f"t{i}": 0.8 for i in range(18)}
    with tempfile.TemporaryDirectory() as d:
        spec = {"base": "x.json", "arms": {"base": {}, "same": {}, "worse": {}}}
        Path(d, "spec.json").write_text(json.dumps(spec))
        for lab, p, seed in (("base#a", 0.8, 1), ("base#b", 0.8, 2), ("same", 0.8, 3), ("worse", 0.35, 4)):
            rs = mkruns(lab, {t: p for t in tasks}, 10, seed, "tool_call", valid=True)
            Path(d, f"runs-{lab}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
        runs = T.load_runs(d)
        by, drift = T.arrange(list(spec["arms"]), runs)
        assert len(by["base"]) == 360 and drift is not None
        rep = T.build_report(list(spec["arms"]), by, 0.05, 0.03, boots=500)
        assert rep["arms"]["worse"]["verdict"] == "REJECT"
        assert rep["arms"]["same"]["verdict"] in ("PASS", "INCONCLUSIVE")
        assert rep["arms"]["same"]["success_diff"]["lo"] < 0 < rep["arms"]["same"]["success_diff"]["hi"]
        text = T.format_report(rep)
        assert "worse" in text and "REJECT" in text and "Wilson" in text


def test_validity_guard_overrides_success():
    base = mkruns("base", {f"t{i}": 0.9 for i in range(18)}, 10, 1, "tool_call", valid=True)
    arm = mkruns("arm", {f"t{i}": 0.9 for i in range(18)}, 10, 2, "tool_call", valid=True)
    for r in arm[:90]:                                      # same success draw, but half a rep of invalid calls
        r["ok"], r["valid"] = False, False
    rep = T.build_report(["base", "arm"], {"base": base, "arm": arm}, 0.05, 0.03, boots=500)
    assert rep["arms"]["arm"]["verdict"] in ("REJECT", "INCONCLUSIVE")


def run(ok, elapsed, task="t0"):
    return {"arm": "x", "rep": 0, "task": task, "kind": "qa", "ok": ok, "valid": None, "reason": "", "elapsed": elapsed,
            "completion_tokens": 1, "error": None, "message": {}}


def test_time_auc_definition():
    # A57: AUC = mean over all runs of ok * (horizon - elapsed) / horizon
    assert T.time_auc([run(True, 0.0)], 10.0) == 1.0                       # instant success
    assert T.time_auc([run(True, 10.0)], 10.0) == 0.0                      # success at the horizon
    assert T.time_auc([run(False, 1.0)], 10.0) == 0.0                      # a failure never counts
    assert abs(T.time_auc([run(True, 2.0), run(False, 3.0)], 10.0) - 0.4) < 1e-12   # (0.8 + 0) / 2
    assert T.time_auc([run(True, 15.0)], 10.0) == 0.0                      # beyond the horizon: clipped, not negative
    assert T.time_auc([], 10.0) is None and T.time_auc([run(True, 1.0)], 0.0) is None
    # step curve S(t): the area equals the integral of the share of runs that succeeded within t
    rs = [run(True, 2.0), run(True, 6.0), run(False, 9.0), run(True, 4.0)]
    ts = [i * 0.001 for i in range(10001)]
    area = sum(T.success_within(rs, t) * 0.001 for t in ts[:-1]) / 10.0
    assert abs(area - T.time_auc(rs, 10.0)) < 2e-3
    assert T.success_within(rs, 4.0) == 0.5 and T.success_within(rs, 100.0) == 0.75


def test_auc_ranks_a_faster_equal_success_arm_higher_and_is_not_part_of_the_verdict():
    tasks = {f"t{i}": 0.8 for i in range(18)}
    slow = mkruns("slow", tasks, 10, 5, "qa")                               # same success draw as fast (same seed)
    fast = mkruns("fast", tasks, 10, 5, "qa")
    base = mkruns("base", tasks, 10, 6, "qa")
    for r in fast:
        r["elapsed"] = r["elapsed"] * 0.5
    arms = ["base", "fast", "slow"]
    rep = T.build_report(arms, {"base": base, "fast": fast, "slow": slow}, 0.05, 0.03, boots=300)
    a_fast, a_slow = rep["arms"]["fast"]["auc"]["auc"], rep["arms"]["slow"]["auc"]["auc"]
    assert a_fast > a_slow
    assert rep["arms"]["fast"]["success_verdict"] == rep["arms"]["slow"]["success_verdict"]   # identical successes: same verdict
    assert rep["arms"]["fast"]["auc_diff"]["diff"] > rep["arms"]["slow"]["auc_diff"]["diff"]
    assert rep["auc_horizon_s"] == 7.0                                      # the longest run of all arms (a failed base run)
    text = T.format_report(rep)
    assert "time-AUC" in text and "horizon 7 s" in text
    rep2 = T.build_report(arms, {"base": base, "fast": fast, "slow": slow}, 0.05, 0.03, boots=300, auc_horizon=20.0)
    assert rep2["auc_horizon_s"] == 20.0 and rep2["arms"]["slow"]["auc"]["auc"] > a_slow   # a wider axis rewards both successes


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
