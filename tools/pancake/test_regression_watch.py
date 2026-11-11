"""Unit tests for regression_watch.py: collecting clean runs, the noise-aware comparison, the exit codes and the stored
last-good record, with abtest.py replaced by a fake that writes its --out file.  No server, no GPU.
Run: python -m pytest tools/pancake/test_regression_watch.py   (or python tools/pancake/test_regression_watch.py)"""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regression_watch as rw  # noqa: E402


def run_record(clean, short, edit, long_):
    return {"clean": clean, "rows": [{"case": "short", "decode_tps": short, "prompt_tps": 1.0},
                                     {"case": "edit", "decode_tps": edit, "prompt_tps": 1.0},
                                     {"case": "long", "decode_tps": 1.0, "prompt_tps": long_}]}


def lines(*recs):
    return [json.dumps(r) for r in recs]


def test_collect_drops_noisy_runs_and_uses_the_right_metric():
    got = rw.collect(lines(run_record(True, 100, 120, 900), run_record(False, 1, 1, 1), run_record(True, 102, 118, 910)))
    assert got == {"short": [100, 102], "edit": [120, 118], "long": [900, 910]}   # long is prompt tok/s
    assert rw.collect(lines(run_record(False, 1, 1, 1))) == {}


def test_summarize_gives_mean_and_interval():
    s = rw.summarize({"short": [100.0, 104.0, 96.0, 100.0], "edit": [120.0]})
    assert s["short"]["mean"] == 100.0 and 4.0 < s["short"]["half"] < 8.0 and s["short"]["n"] == 4
    assert s["edit"]["half"] is None   # one run has no interval


def rec(mean, half, n=4):
    return {"mean": mean, "half": half, "n": n}


def test_compare_statuses():
    good = {"short": rec(100, 2), "edit": rec(100, 2), "long": rec(100, 2), "x": rec(100, 2)}
    now = {"short": rec(99, 2),    # -1%: ok
           "edit": rec(80, 2),     # -20%, intervals 2.8%: regression
           "long": rec(94, 6)}     # -6% above the tolerance but inside the +-6.3% noise: suspect
    res = rw.compare(now, good, 5.0)
    assert res["short"]["status"] == "ok" and res["edit"]["status"] == "regression" and res["long"]["status"] == "suspect"
    assert res["x"]["status"] == "missing"
    assert rw.compare({"a": rec(90, None, 1)}, {"a": rec(100, 2)}, 5.0)["a"]["status"] == "suspect"   # no interval: never "sure"
    assert rw.compare({"a": rec(120, 1)}, {"a": rec(100, 1)}, 5.0)["a"]["status"] == "ok"             # faster is fine
    assert [rw.exit_code({"a": {"status": s}}) for s in ("ok", "regression", "missing", "suspect")] == [0, 1, 2, 3]
    assert rw.exit_code({"a": {"status": "suspect"}, "b": {"status": "regression"}}) == 1


def fake_abtest(tps):
    """subprocess.run replacement: abtest.py writes its --out file with `tps` records, git prints a revision."""
    def run(cmd, **kw):
        if "--out" in cmd:
            Path(cmd[cmd.index("--out") + 1]).write_text("\n".join(lines(*tps)))
        return SimpleNamespace(stdout="abc1234\n", returncode=0)
    return run


def test_main_stores_first_run_compares_later_ones_and_refuses_empty_runs():
    saved = rw.ROOT, rw.GOOD, rw.HISTORY, rw.subprocess.run   # subprocess is the shared module: put it back
    try:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rw.ROOT, rw.GOOD, rw.HISTORY = root, root / "good.json", root / "history.jsonl"
            steady = [run_record(True, 100 + i, 120 - i, 900 + i) for i in range(4)]
            rw.subprocess.run = fake_abtest(steady)
            assert rw.main(["--accept"]) == 0 and rw.GOOD.exists()                    # the first run stores itself
            assert rw.main([]) == 0                                                     # the same numbers again
            slow = [run_record(True, 70 + i, 120 - i, 900 + i) for i in range(4)]      # short -30%
            rw.subprocess.run = fake_abtest(slow)
            assert rw.main([]) == 1
            before = rw.GOOD.read_text()
            rw.subprocess.run = fake_abtest([run_record(False, 1, 1, 1)])
            assert rw.main(["--accept"]) == 2 and rw.GOOD.read_text() == before         # no clean data: nothing compared or stored
            assert len(rw.HISTORY.read_text().splitlines()) == 3                         # the empty run is not recorded
    finally:
        rw.ROOT, rw.GOOD, rw.HISTORY, rw.subprocess.run = saved


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
