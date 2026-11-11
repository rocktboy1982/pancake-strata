"""Unit tests for finemoe_replay.py (MOD_PLAN 16 order item 1): FineMoE's delta selection, trajectory / semantic search,
store deduplication, PRI / LRU / LFU eviction, the prefetch timing, the MoE-Infinity EAM baseline, the cross-check of the
cache replay against time_ceiling.replay, the time_ceiling.parse / fit_model refactor, cache_learn's per-layer misses and
the command line on a tiny synthetic trace.  No GPU, no engine, no real trace.
Run: python -m pytest tools/pancake/test_finemoe_replay.py   (or python tools/pancake/test_finemoe_replay.py)"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cache_learn as cl  # noqa: E402
import finemoe_replay as F  # noqa: E402
import replay_cache_policy as rp  # noqa: E402
import time_ceiling as tc  # noqa: E402

NL, NE = F.NL, F.NE
A, B = list(range(0, 10)), list(range(10, 20))


def pattern_windows(pats):
    """One token per window, top-10: every layer routes the pattern's ten experts."""
    return [[(l, np.array([p], dtype=np.int64)) for l in range(NL)] for p in pats]


def small_cache(b=10, init=range(100, 110)):
    budgets = np.full(NL, b)
    inits = [list(init)[:b] for _ in range(NL)]
    keep = np.zeros((NL, NE))
    return budgets, inits, keep


def chain_prev(n):
    prev = np.full(n, -1)
    prev[1:] = np.arange(n - 1)
    return prev


def test_select_delta_rule():
    P = np.zeros(NE, np.float32)
    P[[5, 7, 9]] = [0.5, 0.3, 0.2]
    assert list(F.select(P, 0.6, 2)) == [5, 7]          # delta 0.4: one expert reaches it, at least K = 2
    assert list(F.select(P, 0.1, 2)) == [5, 7, 9]       # delta 0.9: 0.5 + 0.3 < 0.9 -> three
    assert list(F.select(P, 0.25, 1)) == [5, 7]         # delta 0.75: 0.8 >= 0.75 -> two
    assert list(F.select(P, -1.0, 1)) == [5, 7, 9]      # delta clipped to 1: every expert with p > 0, never a p = 0 one
    assert list(F.select(P, 1.0, 1)) == [5]             # delta 0: K experts
    assert list(F.select(P, 1.0, 8)) == [5, 7, 9]       # K is capped by the experts with p > 0
    assert list(F.select(P, 0.0, 2, use_delta=False)) == [5, 7]


def test_expert_maps_from_counts_and_weights():
    w = [[(l, np.array([[1, 2], [2, 3]], dtype=np.int64)) for l in range(NL)]]
    counts, maps = F.expert_maps(w)
    assert counts[0, 0, 2] == 2 and counts[0, 0, 1] == 1 and np.isclose(maps[0, 5, 2], 0.5)
    wts = [[np.array([[0.6, 0.4], [0.5, 0.5]], np.float32)] * NL]
    _c, maps = F.expert_maps(w, wts)
    assert np.allclose(maps[0, 3, [1, 2, 3]], [0.3, 0.45, 0.25]) and np.isclose(maps[0].sum(axis=1), 1).all()


def test_trajectory_and_semantic_search():
    counts, maps = F.expert_maps(pattern_windows([A, B, B]))
    d = 3
    fm = F.FineMoE(maps, chain_prev(3), d, "none", store=[0, 1])
    out = fm.predict(2)
    assert all(out[l] is None for l in range(d))                       # Map(T): nothing below d
    P, sc, S, ahead = out[d]
    assert np.isclose(sc, 1.0) and sorted(S) == B and ahead == d       # the stored B window, delta 0 -> K = 10
    # oracle semantic: the current window's own first d layers; prev semantic: the window before (0 = A for window 1)
    fm = F.FineMoE(maps, chain_prev(3), d, "oracle", store=[0, 1])
    assert sorted(fm.predict(2)[0][2]) == B and fm.predict(2)[0][3] is None
    prev = np.array([-1, 0, 0])            # window 2 follows window 0 (A), like stored window 1 did
    fm = F.FineMoE(maps, prev, d, "prev", store=[1])
    P, sc, S, ahead = fm.predict(2)[1]
    assert np.isclose(sc, 1.0) and sorted(S) == B


def test_store_deduplicates_at_capacity():
    counts, maps = F.expert_maps(pattern_windows([A, B, A]))
    fm = F.FineMoE(maps, chain_prev(3), 3, "none", cap=2, store=[0, 1])
    fm.update(2)                           # identical to window 0: it replaces window 0, the most redundant
    assert fm.ids == [2, 1]
    fm = F.FineMoE(maps, chain_prev(3), 3, "none", cap=3, store=[0, 1])
    fm.update(2)
    assert fm.ids == [0, 1, 2]


def test_eviction_orders():
    budgets, inits, keep = small_cache(b=3, init=[1, 2, 3])
    keep[0, [1, 2, 3]] = [3, 2, 1]
    excl = np.zeros(NE, bool)
    ca = F.Cache(budgets, inits, keep, "pri")
    assert ca.victim(0, excl) == 3                          # all keys equal: the lowest profile rank
    ca.freq[0, [1, 2, 3]] = [5, 1, 2]
    ca.p[0, [1, 2, 3]] = [0.0, 0.5, 0.0]
    assert ca.victim(0, excl) == 3                          # (0 + eps) * 3 < (0 + eps) * 6 < 0.5 * 2
    ca.p[0, 3] = 0.9
    assert ca.victim(0, excl) == 1                          # eps * 6 is the smallest now
    excl[1] = True
    assert ca.victim(0, excl) == 2                          # never an excluded expert
    ca.evict = "lfu"
    assert ca.victim(0, np.zeros(NE, bool)) == 2
    ca.evict = "lru"
    ca.stamp[0, [1, 2, 3]] = [4, 7, 2]
    assert ca.victim(0, np.zeros(NE, bool)) == 3
    assert ca.put(0, 9, np.zeros(NE, bool)) and ca.res[0, 9] and not ca.res[0, 3] and ca.nres[0] == 3
    b0 = np.zeros(NL, int)
    assert not F.Cache(b0, inits, keep, "lru").put(0, 9, np.zeros(NE, bool))   # no slots: never admitted


def test_fine_beats_lru_on_alternating_patterns():
    """A B A B ...: one pattern fits a layer.  LRU (admit every miss) thrashes; FineMoE finds the stored window of the same
    pattern and prefetches it, PRI evicts the other one."""
    n = 8
    wins = pattern_windows([A if t % 2 == 0 else B for t in range(n)])
    counts, maps = F.expert_maps(wins)
    budgets, inits, keep = small_cache()
    order = list(range(n))
    lru, _ = F.simulate(counts, order, budgets, inits, keep, None, "miss", "lru")
    assert lru[0] == 0.0
    fm = F.FineMoE(maps, chain_prev(n), 3, "oracle")
    r, st = F.simulate(counts, order, budgets, inits, keep, fm, "none", "pri")
    assert np.isclose(r[0], (n - 2) / n) and r[1][2:].sum() == 0 and st["late"] == 0.0
    # cold misses (not resident before the prefetch) in windows 0, 1, 3..7; window 1's prefetch (A: score 0 -> delta 1)
    # misses B, windows 3..7 cover theirs
    assert np.isclose(st["cover"], (n - 3) / (n - 1))
    # copies that cannot land before the layer's experts: misses this window, residents afterwards
    slow = {"layer": 1.0, "pre": 0.5, "dma": 1.0}
    fm = F.FineMoE(maps, chain_prev(n), 3, "oracle")
    r2, st2 = F.simulate(counts, order, budgets, inits, keep, fm, "none", "pri", timing=slow)
    assert st2["late"] > 0.5 and r2[0] < r[0]
    fast = {"layer": 1.0, "pre": 0.5, "dma": 0.001}
    fm = F.FineMoE(maps, chain_prev(n), 3, "oracle")
    r3, st3 = F.simulate(counts, order, budgets, inits, keep, fm, "none", "pri", timing=fast)
    assert st3["late"] == 0.0 and np.isclose(r3[0], r[0])


def test_eam_matches_the_request_with_the_same_pattern():
    n = 3
    wins = pattern_windows([A] * n + [B] * n + [A] * n)
    counts, maps = F.expert_maps(wins)
    req = np.repeat([0, 1, 2], n)
    budgets, inits, keep = small_cache()
    order = list(range(3 * n))
    r, st = F.simulate(counts, order, budgets, inits, keep, F.EAM(counts, req, 10, False, order), "none", "eam")
    assert r[1][:2 * n].sum() > 0 and r[1][2 * n:].sum() == 0      # request 2 is served from request 0's EAM
    off, _ = F.simulate(counts, order, budgets, inits, keep, F.EAM(counts, req, 10, True, order), "none", "eam")
    assert off[0] > r[0]                                          # offline: requests 0 and 1 see each other too


def zipf_trace(n, seed=3, k=4):
    rnd = np.random.default_rng(seed)
    w = 1.0 / np.arange(1, 81) ** 1.1
    w /= w.sum()
    out = []
    for _ in range(n):
        out.append([(l, rnd.choice(80, size=(3, k), p=w).astype(np.int64)) for l in range(NL)])
    return out


def test_eviction_only_replay_tracks_time_ceiling():
    wins = zipf_trace(60)
    counts, _maps = F.expert_maps(wins)
    budgets = np.full(NL, 12)
    inits = [list(range(12)) for _ in range(NL)]
    keep = np.zeros((NL, NE))
    order = list(range(len(wins)))
    seqs, uses = tc.per_layer(wins)
    for adm, kind in (("miss", "lru"), ("kmiss", "kmiss")):
        ref = tc.replay(seqs, uses, budgets, inits, kind, kmiss=(2, 8))
        got, _ = F.simulate(counts, order, budgets, inits, keep, None, adm, "lru")
        assert abs(got[0] - ref[0]) < 0.01, (adm, got[0], ref[0])
        assert abs(got[2].mean() - ref[2].mean()) < 0.05 * max(1.0, ref[2].mean())
        assert (got[3].sum(axis=1) == got[1]).all()


def test_interleave_order_matches_cache_learn():
    lengths = [3, 5, 2]
    s = np.cumsum([0] + lengths)
    parts = [cl.Layered([[np.array([s[i] + t]) for t in range(n)] for _ in range(NL)], NL) for i, n in enumerate(lengths)]
    ref = [int(x[0]) for x in cl.interleave(parts, 2).seqs[0]]
    assert F.interleave_order(lengths, 2) == ref


def test_cache_learn_per_layer_misses():
    wins = zipf_trace(12, seed=5)
    seqs, _uses = tc.per_layer(wins)
    lay = cl.Layered(seqs, NL)
    budgets = np.full(NL, 6)
    inits = [list(range(6)) for _ in range(NL)]
    for run in (lambda m: cl.run_opt(lay, budgets, inits, ml=m),
                lambda m: cl.run_predicted(lay, budgets, inits, lambda l, e, t: True, ml=m)):
        m = np.zeros((lay.n, NL), int)
        r = run(m)
        assert np.isclose(m.sum() / lay.n, r["miss_w"]) and m.sum() > 0
    assert cl.run_opt(lay, budgets, inits) == cl.run_opt(lay, budgets, inits, ml=np.zeros((lay.n, NL), int))


def test_time_ceiling_parse_and_fit():
    a = tc.parse(["t", "p", "10", "--validate"])
    assert a.alpha == 0.5 and a.cpu_ms == 17.41 and a.draft_ms == 2.5
    a2 = tc.parse(["t", "p", "10", "--validate", "--alpha", "0.3"])
    assert a2.alpha == 0.3
    assert tc.parse(["t", "p", "10"]).alpha == 0.25
    n = 20
    rnd = np.random.default_rng(1)
    ml = rnd.integers(2, 12, size=(n, NL))
    base = ("tier", 0.8, ml.sum(axis=1), np.full(n, 20), ml)
    p, sdef, extra = tc.fit_model(a, base)
    assert np.isclose(p["c"], (a.cpu_ms - NL * a.ps_us / 1e3) / ml.sum(axis=1).mean())
    s0 = tc.model_ms(p, base, "serial", a.heavy)
    d0 = tc.model_ms(p, base, "defer", a.heavy, sdef)
    if 0.0 < sdef < 0.5:                   # the fit reproduces --fit-gain when it is reachable
        assert abs(((s0 + extra) / (d0 + extra) - 1) * 100 - a.fit_gain) < 0.01
    _p, sd2, _e = tc.fit_model(tc.parse(["t", "p", "10", "--validate", "--sdef-us", "91"]), base)
    assert np.isclose(sd2, 0.091)


def _write_synthetic(d, n=10, k=4):
    recs = []
    for w in zipf_trace(n, seed=9, k=k):
        for l, ids in w:
            for row in ids:
                recs.append(np.concatenate([[l, k], row, np.zeros(k)]).astype(np.int32))
    tr = d / "routing.bin"
    np.stack(recs).tofile(tr)
    pairs = np.array([(l, e) for e in range(20) for l in range(NL)], dtype=np.uint16)
    prof = d / "profile.bin"
    with open(prof, "wb") as f:
        f.write(b"STRP")
        f.write(np.array([1, NL, NE, 0, len(pairs)], dtype=np.uint32).tobytes())
        f.write(pairs.tobytes())
    return tr, prof, 8 * NL


def test_load_matches_replay_cache_policy():
    with tempfile.TemporaryDirectory() as td:
        tr, _prof, _slots = _write_synthetic(Path(td))
        wins, wts, k = F.load(str(tr))
        ref, k2 = rp.load_trace(str(tr))
        assert k == k2 == 4 and len(wins) == len(ref) == 10 and all(x is None for x in wts[0])
        assert all((a[1] == b[1]).all() and a[0] == b[0] for wa, wb in zip(wins, ref) for a, b in zip(wa, wb))


def test_command_line_on_a_synthetic_trace():
    with tempfile.TemporaryDirectory() as td:
        tr, prof, slots = _write_synthetic(Path(td))
        out = Path(td) / "r.json"
        rep = F.main([str(prof), str(slots), str(tr), "--cuts", "4", "--names", "a,b", "--interleave", "2",
                      "--json", str(out)])
        assert out.exists()
        assert len(rep) == 4                                     # a, b, whole, interleave
        for name, r in rep.items():
            pol = r["policies"]
            assert {"tier (preset)", "kmiss 2/8", "opt (A46 ceiling)", "fine d3 (faithful: prev-sem, miss, pri)",
                    "eam online (MoE-Infinity)", "no prefetch: kmiss + pri"} <= set(pol)
            assert all(0.0 <= v["hits"] <= 1.0 and v["ms"] > 0 for v in pol.values())
            between = [v["miss_w"] for k, v in pol.items() if not k.startswith(("fine", "eam"))]   # no in-window prefetch
            assert pol["opt (A46 ceiling)"]["miss_w"] <= min(between) + 1e-9
            assert set(r["A8"]) == {8, 16, 64, 128} and set(r["A7"]) == {"no contention", "in-flight reorder", "both"}
        assert any("offline store" in k for k in rep[next(iter(rep))]["policies"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
