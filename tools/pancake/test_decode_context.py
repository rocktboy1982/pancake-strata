"""Unit tests for decode_context.py (A49): the line fit, the attention share and the verdict on synthetic abtest rows.
Run: python -m pytest tools/pancake/test_decode_context.py   (or python tools/pancake/test_decode_context.py)"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decode_context as D  # noqa: E402

CTX = {"short": 200, "longdec10k": 10000, "longdec32k": 32000, "longdec64k": 64000, "longdec128k": 120000}


def run_lines(a_ms, b_us, arm="base", clean=True, noisy_extra=0.0):
    """One abtest record per case: decode tok/s from ms = a + b x context (+ a constant on the longest case)."""
    out = []
    for case, ctx in CTX.items():
        ms = a_ms + b_us * 1e-3 * (ctx + 128) + (noisy_extra if case == "longdec128k" else 0.0)
        out.append(json.dumps({"arm": arm, "clean": clean, "rows": [
            {"case": case, "prompt": ctx, "gen": 256, "decode_tps": 1000.0 / ms, "accept": 0.8}]}))
    return out


def test_fit_recovers_a_line():
    a, b = D.fit([(0, 10.0), (100, 12.0), (200, 14.0), (300, 16.0)])
    assert abs(a - 10.0) < 1e-9 and abs(b - 0.02) < 1e-9


def test_attention_share_and_verdict():
    big = D.analyse(D.requests(run_lines(10.0, 0.15), "base"))    # 0.15 ms per 1K tokens of context: 18 ms at 120K
    assert abs(big["b"] - 0.15e-3) < 1e-6 and abs(big["a"] - 10.019) < 0.05
    assert 60 < big["top_share"] < 68 and big["verdict"] == "build A50 / A51"
    small = D.analyse(D.requests(run_lines(10.0, 0.005), "base"))
    assert small["top_share"] < 10 and small["verdict"] == "close A50 / A51"
    mid = D.analyse(D.requests(run_lines(10.0, 0.014), "base"))
    assert 10 <= mid["top_share"] < 15 and mid["verdict"].startswith("borderline")
    assert big["share"]["short"] < 1.0 and big["cases"]["longdec128k"]["accept"] == 0.8


def test_only_the_clean_runs_of_the_asked_arm_count():
    lines = run_lines(10.0, 0.15) + run_lines(10.0, 0.0, arm="other") + run_lines(10.0, 0.0, clean=False)
    assert len(D.requests(lines, "base")) == 5
    assert D.analyse(D.requests(lines, "base"))["top_share"] > 50


def test_too_few_contexts_is_refused():
    try:
        D.analyse(D.requests(run_lines(10.0, 0.1)[:2], "base"))
    except SystemExit:
        return
    raise AssertionError("two contexts cannot fit a slope worth a verdict")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
