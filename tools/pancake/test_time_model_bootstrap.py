"""Unit tests for time_model_bootstrap.py: the predictions on a small synthetic trace, the refit of the deferred-layer cost and
the block resampling.  No GPU, no real trace.
Run: python -m pytest tools/pancake/test_time_model_bootstrap.py   (or python tools/pancake/test_time_model_bootstrap.py)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import replay_cache_policy as rp  # noqa: E402
import time_model_bootstrap as B  # noqa: E402


def trace(n_win, seed=0, hot=40):
    rng = np.random.default_rng(seed)
    hots = [rng.choice(rp.NE, hot, replace=False) for _ in range(rp.NL)]
    return [[(l, np.stack([rng.choice(hots[l], 10, replace=False) for _ in range(4)]).astype(np.int64)) for l in range(rp.NL)]
            for _ in range(n_win)]


def profile(slots):   # the first `slots` pairs, interleaved over layers, expert ids 0.. (not the trace's hot sets)
    return np.array([(l, e) for e in range(rp.NE) for l in range(rp.NL)][:slots])


def test_predictions_fit_the_deferral_gain_exactly():
    r = B.predictions(trace(30), profile(48 * 20))
    assert abs(r["defer"] - 15.8) < 0.05          # the deferred-layer cost is fitted to it
    assert 0 < r["sdef_us"] < 500 and {"kmiss", "fuse", "guard"} <= set(r)
    assert 0 < r["fuse"] < 5                       # halving the pool barriers is a small gain


def test_bootstrap_resamples_blocks_and_varies():
    w = trace(40, seed=1)
    out = B.bootstrap(w, profile(48 * 20), boot=3, block=10)
    assert len(out) == 3 and all(abs(o["defer"] - 15.8) < 0.05 for o in out)
    assert len({round(o["sdef_us"], 3) for o in out}) > 1   # different resamples fit different costs
    again = B.bootstrap(w, profile(48 * 20), boot=3, block=10)
    assert [o["kmiss"] for o in out] == [o["kmiss"] for o in again]   # fixed seed


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
