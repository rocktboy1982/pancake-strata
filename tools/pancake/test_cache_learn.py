"""Unit tests for cache_learn.py (A46): the Belady-with-bypass oracle against a brute-force optimum, OPTgen labels on
hand-made traces, the online Hawkeye learner, past-only features, the synthetic interleave and the command line on a tiny
synthetic trace.  No GPU, no engine, no real trace.
Run: python -m pytest tools/pancake/test_cache_learn.py   (or python tools/pancake/test_cache_learn.py)"""
import itertools
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cache_learn as C  # noqa: E402
import replay_cache_policy as rp  # noqa: E402


def mk(windows_per_layer):
    """windows_per_layer[l][t] = list of routed ids (entries, repeats allowed)."""
    seqs = [[np.array(w, dtype=np.int64) for w in layer] for layer in windows_per_layer]
    return C.Layered(seqs, len(seqs))


def dp_min_misses(windows, b, init):
    """Brute force: the fewest distinct misses over every admission / eviction choice between windows."""
    states = {frozenset(init[:b]): 0}
    for w in windows:
        req = set(w)
        nxt = {}
        for s, cost in states.items():
            c = cost + len(req - s)
            pool = sorted(s | req)
            for k in range(0, min(b, len(pool)) + 1):
                for sub in itertools.combinations(pool, k):
                    f = frozenset(sub)
                    if nxt.get(f, 1 << 30) > c:
                        nxt[f] = c
        states = nxt
    return min(states.values())


def test_opt_is_the_optimum_on_small_traces():
    rnd = random.Random(7)
    for trial in range(60):
        ne, b, n = 6, rnd.choice([1, 2, 3]), rnd.randint(3, 8)
        wins = []
        for _ in range(n):
            k = rnd.randint(1, 3)
            ids = rnd.sample(range(ne), k)
            wins.append([e for e in ids for _ in range(rnd.randint(1, 3))])
        init = rnd.sample(range(ne), b)
        tr = mk([wins])
        r = C.run_opt(tr, [b], [init])
        got = round(r["miss_w"] * n)
        want = dp_min_misses(wins, b, init)
        assert got == want, (trial, wins, b, init, got, want)


def test_opt_never_worse_than_the_other_policies():
    rnd = random.Random(11)
    old = rp.NL
    try:
        rp.NL = 2   # time_ceiling.replay walks rp.NL layers
        for trial in range(8):
            ne, n = 40, 30
            hot = rnd.sample(range(ne), 8)
            layers = []
            for _l in range(2):
                wins = []
                for _ in range(n):
                    ids = [rnd.choice(hot) if rnd.random() < 0.6 else rnd.randrange(ne) for _ in range(12)]
                    wins.append(ids)
                layers.append(wins)
            tr = mk(layers)
            budgets = [6, 4]
            inits = [rnd.sample(range(ne), 6), rnd.sample(range(ne), 4)]
            init_pairs = np.array([(0, e) for e in inits[0]] + [(1, e) for e in inits[1]], dtype=np.int64)
            res = C.run_all(tr, budgets, inits, init_pairs, with_tier=False,
                            imit_w=np.array([0.0, 1, 1, 0, 0, 0, 0]))
            opt = res["opt"]["miss_w"]
            for k, r in res.items():
                assert opt <= r["miss_w"] + 1e-9, (trial, k, opt, r["miss_w"])
    finally:
        rp.NL = old


def test_optgen_labels_by_hand():
    # one slot: A at 0 and 2, B at 1 and 3 -> A's reuse fits, B's overlaps it (window 2 is full)
    tr = mk([[[0], [1], [0], [1]]])   # A = 0, B = 1
    lab = C.optgen_labels(tr, [1])[0]
    assert lab[(0, 0)] is True
    assert lab[(1, 1)] is False
    assert lab[(2, 0)] is False and lab[(3, 1)] is False   # last uses: no reuse
    # two slots: both fit
    lab2 = C.optgen_labels(tr, [2])[0]
    assert lab2[(0, 0)] is True and lab2[(1, 1)] is True
    # zero slots: nothing is ever kept
    lab0 = C.optgen_labels(tr, [0])[0]
    assert not any(lab0.values())


def test_label_oracle_and_hawkeye_beat_lru_on_a_scan_with_a_hot_set():
    # a hot pair reused every window plus a scan of one-off experts that floods an LRU of 3 slots
    wins = []
    nxt = 100
    for t in range(60):
        w = [1, 1, 2, 2]
        for _ in range(3):
            w.append(nxt)
            nxt += 1
        wins.append(w)
    tr = mk([wins])
    old = rp.NL
    try:
        rp.NL = 1
        res = C.run_all(tr, [3], [[50, 51, 52]], np.array([(0, 50), (0, 51), (0, 52)]), with_tier=False)
    finally:
        rp.NL = old
    assert res["label"]["hits"] > res["lru"]["hits"]
    assert res["hawkeye-pc"]["hits"] > res["lru"]["hits"]
    assert res["opt"]["hits"] >= res["hawkeye-pc"]["hits"] - 1e-9
    assert res["hawkeye-pc"]["promo_w"] < res["lru"]["promo_w"]   # the one-offs are bypassed, not copied


def test_features_use_the_past_only():
    rnd = random.Random(3)
    wins = [[rnd.randrange(8) for _ in range(6)] for _ in range(20)]
    tr = mk([wins])
    f = C.features(tr, 0, wins[10][0], 10)
    wins2 = [list(w) for w in wins]
    for t in range(11, 20):
        wins2[t] = [rnd.randrange(8) for _ in range(6)]
    tr2 = mk([wins2])
    assert np.allclose(f, C.features(tr2, 0, wins[10][0], 10))
    assert f.shape == (C.NF,)


def test_fit_logistic_separates():
    X = np.array([[1.0, x] for x in np.linspace(-1, 1, 50)])
    y = (X[:, 1] > 0).astype(float)
    w = C.fit_logistic(X, y, iters=2000, lr=1.0)
    p = 1 / (1 + np.exp(-(X @ w)))
    assert ((p >= 0.5) == (y > 0.5)).mean() > 0.95


def test_interleave_keeps_every_window():
    a = mk([[[0]] * 5])
    b = mk([[[1]] * 3])
    tr = C.interleave([a, b], 2)
    assert tr.n == 8
    order = [int(w[0]) for w in tr.seqs[0]]
    assert order == [0, 0, 1, 1, 0, 0, 1, 0]


def _write_synthetic(d: Path, n_windows=6, k=4, tokens=2, slots=96):
    rnd = np.random.default_rng(5)
    recs = []
    for _w in range(n_windows):
        for l in range(rp.NL):
            for _t in range(tokens):
                ids = rnd.choice(16, size=k, replace=False).astype(np.int32)
                recs.append(np.concatenate([np.array([l, k], np.int32), ids, np.zeros(k, np.float32).view(np.int32)]))
    tr = d / "routing.bin"
    np.concatenate(recs).astype(np.int32).tofile(tr)
    pairs = np.array([(l, e) for e in range(2) for l in range(rp.NL)], dtype=np.uint16)
    prof = d / "profile.bin"
    with open(prof, "wb") as f:
        f.write(b"STRP")
        f.write(np.array([1, rp.NL, rp.NE, 0, len(pairs)], dtype=np.uint32).tobytes())
        f.write(pairs.tobytes())
    return tr, prof, min(slots, len(pairs))


def test_command_line_on_a_synthetic_trace():
    with tempfile.TemporaryDirectory() as td:
        tr, prof, slots = _write_synthetic(Path(td))
        out = Path(td) / "r.json"
        rep = C.main([str(prof), str(slots), str(tr), "--cuts", "3", "--names", "a,b", "--interleave", "1",
                      "--json", str(out)])
        assert set(rep) == {"a", "b", "whole", "interleave1"}
        for res in rep.values():
            assert {"tier", "lru", "belady", "opt", "label", "kmiss 2/4", "kmiss 2/8", "hawkeye-pc", "imit"} <= set(res)
            for r in res.values():
                assert 0.0 <= r["hits"] <= 1.0
            assert res["opt"]["miss_w"] <= min(r["miss_w"] for r in res.values()) + 1e-9
        assert out.exists()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
