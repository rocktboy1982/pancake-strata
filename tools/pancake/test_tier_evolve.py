"""pytest for tools/pancake/tier_evolve.py: the optimiser on a known function, a synthetic trace with a known optimum."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cache_learn as cl  # noqa: E402
import tier_evolve as te  # noqa: E402
import tier_laws as tl  # noqa: E402

NL, NE = 2, 8


def test_evolve_finds_the_minimum_of_a_sphere():
    target = np.array([0.2, 0.8, 0.5, 0.35])
    u, f, hist, evals = te.evolve(lambda x: float(np.sum((x - target) ** 2)), 4, 12, 16, seed=3)
    assert f < 0.01 and np.max(np.abs(u - target)) < 0.12
    assert hist == sorted(hist, reverse=True) and evals > 16


def test_fixed_seed_is_deterministic():
    fn = lambda x: float(np.sum((x - 0.3) ** 2))
    a = te.evolve(fn, 3, 5, 8, seed=7)
    b = te.evolve(fn, 3, 5, 8, seed=7)
    assert a[1] == b[1] and np.array_equal(a[0], b[0])


def test_defaults_round_trip_and_box():
    for fam, d in te.DEFAULTS.items():
        p = te.decode(fam, te.encode(fam, d))
        for k, v in d.items():
            assert abs(p[k] - v) < 1e-6 * max(1.0, abs(v)), (fam, k, p[k], v)
        lo = te.decode(fam, np.zeros(len(d)))
        hi = te.decode(fam, np.ones(len(d)))
        assert all(lo[k] <= hi[k] for k in d)


def test_kmiss_optimum_on_a_synthetic_trace_is_admit_on_first_miss():
    # the hot pair 4,5 is not resident: k = 1 admits it one window earlier than the default k = 2, nothing else matters
    cl_tr = cl.Layered([[np.array([4, 5])] * 40 for _ in range(NL)], NL)
    D = tl.dense(cl_tr, NE)
    init = np.array([(l, e) for e in (0, 1, 2, 3) for l in range(NL)], dtype=np.int64)
    a = argparse.Namespace(generations=6, population=10, seed=0, promo_penalty=0.005)
    o = te.run_family("kmiss", ("synthetic", D, None), init, 8, {"ranked": init, "seed": None}, a)
    assert o["params"]["k"] == 1
    assert o["holdout_gain_ms"] >= -1e-9 and o["evolved"]["fit_half"]["fitness"] <= o["defaults"]["fit_half"]["fitness"] + 1e-12
    assert "BETTER" in te.fmt(o) or "NOT" in te.fmt(o)
