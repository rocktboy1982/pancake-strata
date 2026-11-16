"""turn_pipeline.py on a synthetic turn log.  Run: python -m pytest -q tools/pancake/test_turn_pipeline.py"""
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import turn_pipeline as P  # noqa: E402


def turn(t, dur, read, prompt_ms, ft):
    return {"time": t, "duration_s": dur, "prompt_read": read, "prompt_ms": prompt_ms, "first_engine_token_s": ft,
            "max_new": 8192}


def test_gaps_and_no_gap_saves_nothing():
    rows = [turn(100, 5, 0, 0, 1.0), turn(105.5, 5, 2000, 800, 1.0)]       # next turn starts 0.5 s after the first ended: start 100.5
    seq = P.turns_with_gaps(rows)
    assert seq[0][1] is None and abs(seq[1][1] - 0.5) < 1e-9
    rows = [turn(100, 5, 0, 0, 1.0), turn(105, 5, 2000, 800, 1.0)]         # back to back: gap 0
    assert P.simulate(rows, avail=1.0)["mean_after"] == P.simulate(rows, avail=1.0)["mean_before"]


def test_preread_is_bounded_by_gap_avail_and_prompt_ms():
    # 400 ms per 1k tokens (default; < 3 fitting points).  Gap 1 s allows 2500 tokens; the turn reads 2000 and avail 0.5 gives 1000.
    rows = [turn(100, 5, 0, 0, 1.0), turn(106, 5, 2000, 800, 1.0)]     # start 106 -> gap 1.0 s... end 111: start 106, gap 1 s
    rep = P.simulate(rows, avail=0.5, ms_per_1k=400.0)
    assert abs(rep["read_saved_s"] - 0.4) < 1e-9 and rep["read_staged_turns"] == 1
    rep = P.simulate(rows, avail=1.0, ms_per_1k=400.0)                      # 2000 tokens = 0.8 s = prompt_ms
    assert abs(rep["read_saved_s"] - 0.8) < 1e-9
    rows[1]["prompt_ms"] = 300                                              # cannot save more than the prompt took
    assert abs(P.simulate(rows, avail=1.0)["read_saved_s"] - 0.3) < 1e-9
    short = [turn(100, 5, 0, 0, 1.0), turn(105.2, 5, 2000, 800, 1.0)]  # gap 0.2 s: 500 tokens = 0.2 s at most
    assert abs(P.simulate(short, avail=1.0)["read_saved_s"] - 0.2) < 1e-9


def test_rewarm_only_after_long_gaps_and_only_if_it_fits():
    rows = [turn(100, 5, 0, 0, 1.0), turn(165, 5, 0, 0, 1.0)]     # gap 60 s
    rep = P.simulate(rows, avail=0.0, rewarm_ms=500.0, rewarm_gap=30.0)
    assert abs(rep["mean_before"] - 1.25) < 1e-9 and abs(rep["mean_after"] - 1.0) < 1e-9 and rep["warm_staged_turns"] == 1
    short = [turn(100, 5, 0, 0, 1.0), turn(115, 5, 0, 0, 1.0)]    # gap 10 s < 30 s: no cold start, nothing to save
    rep = P.simulate(short, avail=0.0, rewarm_ms=500.0, rewarm_gap=30.0)
    assert rep["mean_before"] == rep["mean_after"] == 1.0
    tiny = [turn(100, 5, 0, 0, 1.0), turn(105.3, 5, 0, 0, 1.0)]    # gap 0.3 s but rewarm-gap 0.1: cold, 500 ms does not fit
    rep = P.simulate(tiny, avail=0.0, rewarm_ms=500.0, rewarm_gap=0.1)
    assert rep["warm_staged_turns"] == 0 and rep["mean_after"] > 1.0


def test_fitted_slope_is_used_when_the_log_has_points():
    rows = [turn(1000 * i, 5, 1000 * (i + 1), 500 * (i + 1), 1.0) for i in range(5)]   # exactly 500 ms per 1k
    mpt, fitted = P.ms_per_token(rows, 400.0)
    assert fitted and abs(mpt - 0.5) < 1e-9


def test_main_json():
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "t.jsonl"
        log.write_text("\n".join(json.dumps(r) for r in [turn(100, 5, 0, 0, 1.0), turn(111, 5, 2000, 800, 1.0)]) + "\n")
        out = Path(d) / "o.json"
        assert P.main([str(log), "--avail", "1.0", "--rewarm-ms", "0", "--json", str(out)]) == 0
        assert json.loads(out.read_text())[0]["turns"] == 2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_turn_pipeline: ok")
