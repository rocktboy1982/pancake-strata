"""Unit tests for sweep.py (item 37 / A34): the Mann-Whitney test, the start point, the noise-robust coordinate descent and
its confirmation on a simulated noisy landscape, and the knob file.  No server, no GPU.
Run: python -m pytest tools/pancake/test_sweep.py   (or python tools/pancake/test_sweep.py)"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep as S  # noqa: E402

KNOBS = {"A": ["0", "1", "2"], "B": ["x", "y"]}


def landscape(effect, sd=0.03, seed=0, calls=None, fail=None):
    """run_once of a fake engine: decode tok/s of short and edit, a relative `effect` per (knob, value), Gaussian noise."""
    rng = random.Random(seed)

    def run_once(point):
        if calls is not None:
            calls.append(S.key(point))
        if fail is not None and fail(point):
            return None
        gain = 1.0
        for k, v in point.items():
            gain *= 1.0 + effect.get((k, v), 0.0)
        return [("short", 100.0 * gain * (1 + rng.gauss(0, sd))), ("edit", 120.0 * gain * (1 + rng.gauss(0, sd)))]
    return run_once


def test_mann_whitney_exact_values():
    assert S.p_faster([1, 2], [0, 0]) == 1 / 6          # only one of C(4, 2) splits separates them completely
    assert S.p_faster([1, 2, 3], [0, 0.5, 0.7]) == 1 / 20
    assert S.p_faster([5, 5], [5, 5]) == 1.0            # ties: nothing to see
    assert S.p_faster([0, 0], [1, 2]) == 1.0            # slower, not faster


def test_mann_whitney_sampled_branch_is_deterministic():
    new, old = [1.0 + 0.01 * i for i in range(13)], [0.7 + 0.01 * i for i in range(13)]   # C(26, 13) > 20000: permutations sampled
    p = S.p_faster(new, old)
    assert p == S.p_faster(new, old) and p < 0.001
    same = [1.0 + 0.01 * i for i in range(13)]
    assert 0.3 < S.p_faster(same, [x + 0.005 for x in same]) < 0.7


def test_faster_needs_both_the_margin_and_significance():
    new, old = [1.02, 1.03, 1.025, 1.04, 1.03, 1.02], [1.0, 1.01, 0.99, 1.0, 1.005, 0.995]
    assert S.faster(new, old, 0.05, 0.5)
    assert not S.faster(new, old, 0.05, 5.0)             # significant but below the margin
    assert not S.faster([1.01, 0.99, 1.02, 0.98, 1.0, 1.0], old, 0.05, 0.5)   # a higher mean inside the noise


def test_start_point_prefers_the_base_config():
    knobs = {"STRATA_EXPDEC_MINBLK": ["800", "400"], "STRATA_EXPDEC_R": ["1", "2"], "--pool-tasks": ["0", "32"],
             "--other": ["a", "b"]}
    cfg = {"env": {"STRATA_EXPDEC_MINBLK": "400"}, "args": ["--pool-tasks", "32", "--spec", "4"]}
    assert S.start_point(knobs, cfg) == {"STRATA_EXPDEC_MINBLK": "400", "STRATA_EXPDEC_R": "1", "--pool-tasks": "32", "--other": "a"}


def test_a_real_effect_is_found_and_confirmed():
    effect = {("A", "2"): 0.10}
    found = 0
    for seed in range(10):
        run_once = landscape(effect, seed=seed)
        start = {"A": "0", "B": "x"}
        best, hist, ref = S.search(KNOBS, start, run_once, reps=3, alpha=0.05, min_gain=0.5)
        found += best == {"A": "2", "B": "x"}
        if best != start:
            gain, p, ok = S.confirm(start, best, landscape(effect, seed=100 + seed), ref, 3, 0.05, 0.5)
            assert ok and 6 < gain < 14 and p < 0.01
    assert found >= 9   # a 10% knob among 3% noise is found nearly every time


def test_pure_noise_is_rarely_confirmed():
    """The winner of ~20 noisy points looks better than the start (winner's curse); the fresh A B B A runs must not agree."""
    moved = confirmed = 0
    for seed in range(40):
        start = {"A": "0", "B": "x"}
        best, hist, ref = S.search(KNOBS, start, landscape({}, seed=seed), reps=3, alpha=0.05, min_gain=0.5)
        if best != start:
            moved += 1
            confirmed += S.confirm(start, best, landscape({}, seed=500 + seed), ref, 3, 0.05, 0.5)[2]
    assert moved >= 1   # the search does follow noise now and then (several comparisons) ...
    assert confirmed <= 3   # ... and the confirmation throws it out (5% of the moves in expectation)


def test_points_are_measured_once_and_failed_servers_are_skipped():
    calls = []
    run_once = landscape({}, seed=1, calls=calls, fail=lambda p: p["A"] == "1")
    best, hist, ref = S.search(KNOBS, {"A": "0", "B": "x"}, run_once, reps=2, alpha=0.05, min_gain=0.5)
    for k in set(calls):   # every point: exactly `reps` servers; a failing one stops at its first server
        assert calls.count(k) == (1 if dict(k)["A"] == "1" else 2)
    assert best["A"] != "1" and hist[S.key({"A": "1", "B": "x"})] is None
    assert abs(sum(hist[S.key({"A": "0", "B": "x"})]) / 4 - 1.0) < 0.05   # the start's samples are the unit


def test_a_failing_start_ends_the_search():
    best, hist, ref = S.search(KNOBS, {"A": "0", "B": "x"}, landscape({}, fail=lambda p: True), 3, 0.05, 0.5)
    assert best == {"A": "0", "B": "x"} and ref == {}


def test_knob_file_lists_the_engine_defaults_first():
    knobs = json.loads((Path(__file__).resolve().parent / "sweep-knobs-fast-hip.json").read_text())
    defaults = {"STRATA_EXPDEC_RB": "0", "STRATA_EXPDEC_MINBLK": "800", "STRATA_EXPDEC_R": "1", "STRATA_EXPDEC_MINB": "4",
                "STRATA_EXPDEC_DRB": "0", "STRATA_EXPDEC_DR": "4", "STRATA_Q2_PREFETCH": "0", "STRATA_IQ_PREFETCH": "2048",
                "--pool-tasks": "0"}   # src/kernels/cuda/iq_kernels.cu, q2_avx2.cpp, iq_avx2.cpp, generate.cpp (checked 2026-11-11)
    assert {k: v[0] for k, v in knobs.items()} == defaults
    assert all(len(set(v)) == len(v) for v in knobs.values())


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
