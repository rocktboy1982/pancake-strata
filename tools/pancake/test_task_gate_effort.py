"""Tests for the A18 / A17 / A20 additions of task_gate.py and abtest.make_config: a per-arm request override (thinking level),
a per-arm server config, and the time-to-success curve.  No server, no GPU.
python -m pytest tools/pancake/test_task_gate_effort.py"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abtest  # noqa: E402
import task_gate as T  # noqa: E402

SPECS = T.ROOT / "bench/specs"


def run(ok, elapsed, task="t1", rep=0, kind="qa"):
    return {"arm": "x", "rep": rep, "task": task, "kind": kind, "ok": ok, "valid": None, "reason": "", "elapsed": elapsed,
            "completion_tokens": 100, "error": None}


def test_time_curve_counts_only_successes_and_weights_speed():
    runs = [run(True, 10), run(True, 60), run(False, 5), run(True, 200)]
    c = T.time_curve(runs, 120.0, [10, 30, 60, 300])
    assert c["within"] == {"10": 0.25, "30": 0.25, "60": 0.5, "300": 0.75}
    # (1 - 10/120) + (1 - 60/120) + 0 (failure) + 0 (200 s is past the horizon), over 4 runs
    assert abs(c["auc"] - ((1 - 10 / 120) + 0.5) / 4) < 1e-12
    assert T.time_curve([], 120.0, [10]) is None


def test_report_has_the_curve_only_when_asked():
    runs = {"a": [run(True, 20, task=f"t{i}") for i in range(4)], "b": [run(True, 40, task=f"t{i}") for i in range(4)]}
    plain = T.format_report(T.build_report(["a", "b"], runs, 0.05, 0.03, boots=50))
    assert "area under the curve" not in plain and "time_curve" not in T.build_report(["a"], runs, 0.05, 0.03)["arms"]["a"]["stats"]
    rep = T.build_report(["a", "b"], runs, 0.05, 0.03, boots=50, curve=(120.0, [10, 30, 60]))
    text = T.format_report(rep)
    assert "succeeded within: 10s 0%  30s 100%  60s 100%" in text and "area under the curve to 120 s" in text
    assert rep["arms"]["a"]["stats"]["time_curve"]["auc"] > rep["arms"]["b"]["stats"]["time_curve"]["auc"]


def test_arm_settings_override_only_known_request_keys():
    s = {**T.DEFAULTS}
    a = T.arm_settings(s, {"request": {"thinking": True, "reasoning_effort": "xhigh", "max_tokens": 8192, "evil": 1}})
    assert a["thinking"] is True and a["reasoning_effort"] == "xhigh" and a["max_tokens"] == 8192 and "evil" not in a
    assert T.arm_settings(s, {}) == s and s["thinking"] is False                  # the spec's own settings are untouched


def test_chat_body_carries_the_effort(monkeypatch=None):
    seen = {}

    class Resp:
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}], "usage": {"completion_tokens": 3}}).encode()

        def __iter__(self):
            return iter([self.read()])

    def fake(req, timeout=0):
        seen["body"] = json.loads(req.data)
        return __import__("io").BytesIO(Resp().read())

    real = T.urllib.request.urlopen
    T.urllib.request.urlopen = fake
    try:
        T.chat("http://x", {"id": "q", "kind": "qa", "prompt": "hi"}, {**T.DEFAULTS, "thinking": True, "reasoning_effort": "xhigh"}, 0, 5)
        assert seen["body"]["reasoning_effort"] == "xhigh" and seen["body"]["chat_template_kwargs"] == {"enable_thinking": True}
        T.chat("http://x", {"id": "q", "kind": "qa", "prompt": "hi"}, {**T.DEFAULTS}, 0, 5)
        assert "reasoning_effort" not in seen["body"]                              # default requests are unchanged
    finally:
        T.urllib.request.urlopen = real


def test_make_config_sets_server_config_keys_and_leaves_old_arms_alone():
    base = T.ROOT / "configs/pancake/flashnext-iq2_xs-fast-hip.json"
    with tempfile.TemporaryDirectory() as d:
        plain = abtest.make_config(str(base), {"env": {}}, str(Path(d) / "a.json"))
        withc = abtest.make_config(str(base), {"config": {"reasoning_budget_by_effort": {"high": 512}, "rumination_entropy": 8,
                                                          "reasoning_budget_grace": None}}, str(Path(d) / "b.json"))
    assert "reasoning_budget_by_effort" not in plain and plain["reasoning_budget_grace"] == 512
    assert withc["reasoning_budget_by_effort"] == {"high": 512} and withc["rumination_entropy"] == 8
    assert "reasoning_budget_grace" not in withc                                    # null removes the key
    assert {k: v for k, v in withc.items() if k not in ("reasoning_budget_by_effort", "rumination_entropy", "reasoning_budget_grace")} \
        == {k: v for k, v in plain.items() if k != "reasoning_budget_grace"}


def test_the_new_specs_are_well_formed():
    for name in ("effort", "caps", "rumination"):
        spec = json.load(open(SPECS / f"taskgate-{name}.json"))
        assert list(spec["arms"]) and spec["time_curve"] is True and spec["max_tokens"] >= 4096
        for arm in spec["arms"].values():
            assert set(arm.get("request", {})) <= set(T.REQUEST_KEYS)
            assert isinstance(arm.get("config", {}), dict)
        assert (T.ROOT / spec["base"]).exists() and (T.ROOT / spec["tasks"]).exists()
    # every config key an arm sets is one the server reads
    server = (T.ROOT / "serve/server.py").read_text()
    for name in ("caps", "rumination"):
        for arm in json.load(open(SPECS / f"taskgate-{name}.json"))["arms"].values():
            for key in arm.get("config", {}):
                assert f'"{key}"' in server, key


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
