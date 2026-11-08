"""Tests for predict_experts.py on synthetic --dump-routing traces.  Run: python tools/pancake/test_predict_experts.py"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import predict_experts as P  # noqa: E402

R = P.R
EMPTY = np.zeros((0, 2), np.int64)  # nothing cached: every used expert is a miss


def write_trace(path, windows, k=2):
    """windows: list of per-window lists of NL expert-id tuples (one token per layer)."""
    recs = []
    for w in windows:
        for l, ids in enumerate(w):
            recs.append(np.concatenate([[l, k], ids, np.zeros(k, np.float32).view(np.int32)]).astype(np.int32))
    np.concatenate(recs).tofile(path)


def run(windows, budgets=(2,)):
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "t.bin"
        write_trace(f, windows)
        loaded, k = R.load_trace(str(f))
    assert k == 2 and len(loaded) == len(windows)
    return P.evaluate(loaded, EMPTY, list(budgets))[1]


def test_repeating_pattern_is_predicted_by_history():
    w = [[(l % 500, l % 500 + 1) for l in range(R.NL)]]
    out = run(w * 20)
    rc, mc, ws = out[("prev-window", 2)]
    assert rc > 0.9 and mc > 0.9 and ws == 0.0, out[("prev-window", 2)]


def test_transition_follows_the_layer_below():
    rng = np.random.default_rng(0)
    pats = [(0, 1), (2, 3)]
    windows = [[pats[p]] * R.NL for p in rng.integers(0, 2, 60)]
    out = run(windows)
    tr, prev = out[("transition", 2)], out[("prev-window", 2)]
    assert tr[0] > 0.9 and tr[2] < 0.1, tr
    assert tr[0] > prev[0] + 0.2, (tr, prev)


def test_coact_proposes_partners():
    # windows alternate {0,1} / {1,2}: after {1,2} only the partner graph (0 fired with 1) proposes expert 0
    windows = [[(0, 1)] * R.NL, [(1, 2)] * R.NL] * 20
    out = run(windows, budgets=(4,))
    assert out[("coact", 4)][0] > 0.9, out[("coact", 4)]
    assert out[("prev-window", 4)][0] < 0.6, out[("prev-window", 4)]


def test_cross_layer_swaps_move_slots_to_the_layer_that_misses():
    # every layer uses expert 0 (cached); layer 0 also uses 7, which only a slot from another layer can hold
    w = [(0, 7)] + [(0, 0)] * (R.NL - 1)
    windows = [[(l, np.array([ids])) for l, ids in enumerate(w)]] * 40
    init = np.array([(l, 0) for l in range(R.NL)] + [(1, 5)])
    same = R.simulate(windows, init, R.Policy())
    cross = R.simulate(windows, init, R.Policy(cross=True))
    assert same["share"] < cross["share"], (same, cross)
    assert cross["slots_max"] == 2 and cross["slots_min"] == 1, cross


def test_rank_drops_zero_scores():
    assert P.rank(np.array([0.0, 2.0, 1.0, 0.0])).tolist() == [1, 2]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
