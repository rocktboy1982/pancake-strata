"""abtest.py case tables and the batch-2 specs (A21, A30, A33, A35, A49): every spec parses, every arm / case is valid, the new cases are
consistent.  No server, no GPU.

Run: python -m pytest tools/pancake/test_abtest_cases.py   (or python tools/pancake/test_abtest_cases.py)"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abtest as A  # noqa: E402

SPECS = A.ROOT / "bench/specs"
BATCH2 = ["a21-spec-lowtemp", "a30-fiddler-cpu-share", "a33-a35-fusion-retest", "a33-router-fuse-nodefer", "a49-decode-context"]


def test_new_cases_are_in_every_table():
    for c in ("long1k", "long3k", "longdec10k"):
        assert c in A.LONG_TARGET and c in A.MAXTOK and A.is_long(c)
    assert A.metric("long1k") == "prompt_tps" and A.metric("long3k") == "prompt_tps"
    assert A.metric("longdec10k") == "decode_tps" and "longdec10k" in A.LONG_DEC
    assert A.MAXTOK["long3k"] == 32 and A.MAXTOK["longdec10k"] == 256
    assert A.LONG_TARGET["longdec10k"] < A.LONG_TARGET["longdec32k"] < A.LONG_TARGET["longdec64k"] < A.LONG_TARGET["longdec128k"]
    assert set(A.LONG_DEC) <= set(A.LONG_TARGET) and set(A.MAXTOK) <= set(A.LONG_TARGET)


def test_prompts_for_the_new_cases_have_the_target_size():
    P = A.prompts(["long1k", "long3k", "longdec10k"])
    for c in ("long1k", "long3k", "longdec10k"):
        want = int(A.LONG_TARGET[c] * A.LONG_CPT) + len(A.SUMMARIZE)
        assert abs(len(P[c]) - want) <= 2, (c, len(P[c]), want)
    assert len(P["long1k"]) < len(P["long3k"]) < len(P["longdec10k"])


def test_batch2_specs_are_valid():
    known = ["short", "edit", "long"] + list(A.LONG_TARGET)
    for name in BATCH2:
        spec = json.loads((SPECS / f"{name}.json").read_text())
        assert spec["base"].endswith(".json") and spec["arms"] and spec["note"]
        assert all(c in known for c in spec["cases"]), name
        for arm, body in spec["arms"].items():
            assert set(body) <= {"env", "args", "drafting", "exe"}, (name, arm)
            for k, v in (body.get("env") or {}).items():
                assert k.startswith(("STRATA_", "GPU_", "HIP_")) and (v is None or isinstance(v, str)), (name, arm, k)
        assert "lossy" in spec


def test_a21_spec_samples_at_low_temperature_and_a30_keeps_its_prefill_cases():
    s21 = json.loads((SPECS / "a21-spec-lowtemp.json").read_text())
    assert 0 < s21["temperature"] <= 0.1 and "base" in s21["arms"]
    assert any(a["env"].get("STRATA_SPEC_MIN_TEMP") for a in s21["arms"].values() if a.get("env"))
    s30 = json.loads((SPECS / "a30-fiddler-cpu-share.json").read_text())
    assert s30["prefill"] is True and "long3k" in s30["cases"]
    assert s30["arms"]["share0"]["env"] == {"STRATA_PREFILL_CPU_SHARE": "0"}
    assert s30["arms"]["max1024"]["env"] == {"STRATA_PREFILL_CPU_SHARE_MAX": "1024"}


def test_make_config_removes_a_null_env_key(tmp_path=None):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        base = Path(d) / "b.json"
        base.write_text(json.dumps({"env": {"STRATA_VERIFY_DEFER_CPU": "1", "X": "y"}, "args": ["--spec", "4"]}))
        cfg = A.make_config(str(base), {"env": {"STRATA_VERIFY_DEFER_CPU": None, "STRATA_ROUTER_FUSE": "1"}}, str(Path(d) / "o.json"))
        assert "STRATA_VERIFY_DEFER_CPU" not in cfg["env"] and cfg["env"]["STRATA_ROUTER_FUSE"] == "1" and cfg["env"]["X"] == "y"


def test_a49_spec_measures_one_arm_over_growing_context():
    s = json.loads((SPECS / "a49-decode-context.json").read_text())
    ctx = [c for c in s["cases"] if c.startswith("longdec")]
    assert ctx == ["longdec10k", "longdec32k", "longdec64k", "longdec128k"] and list(s["arms"]) == ["base"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
