"""gate_importance.py (HOBBIT gate-magnitude proxy, offline) on synthetic routing.  No model, no GPU.

Run: python -m pytest tools/pancake/test_gate_importance.py   (or python tools/pancake/test_gate_importance.py)"""
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "trace"))
import gate_importance as G  # noqa: E402
import wtrace  # noqa: E402


def routed(rng, rows, n_expert, k, sigma=1.0):
    """A router: Gaussian logits, softmax over all experts, top-k renormalised (Mixtral and Qwen MoE both do this)."""
    logits = rng.normal(0.0, sigma, (rows, n_expert))
    ids = np.argsort(-logits, axis=1)[:, :k]
    p = np.exp(np.take_along_axis(logits, ids, axis=1))
    return ids, p / p.sum(axis=1, keepdims=True)


def test_hand_split_top2():
    w = np.array([[0.95, 0.05], [0.3, 0.7], [0.5, 0.5]])   # the second row unsorted on purpose
    ws, = G.sort_rows(w)
    s = G.cum_score(ws)
    assert np.allclose(s, [[0, 0.95], [0, 0.7], [0, 0.5]])
    hi, lo, sk = G.split(s, 0.6, 0.9)
    assert hi.sum() == 4 and lo.sum() == 1 and sk.sum() == 1
    assert sk[0, 1] and lo[1, 1]
    assert abs(G.loss_share(ws, None, sk) - 0.05 / 3) < 1e-12


def test_correlation_measures():
    rng = np.random.default_rng(1)
    _, w = routed(rng, 4000, 64, 8)
    ws, = G.sort_rows(w)
    c = G.correlations(ws, np.ones_like(ws))               # equal output norms: contribution is the weight
    assert c["pearson_w_contrib"] > 0.999 and c["same_order"] == 1.0
    c = G.correlations(ws, rng.lognormal(0.0, 0.5, ws.shape))   # norms independent of the weight
    assert abs(c["pearson_w_norm"]) < 0.05                 # the honest measure sees no link
    assert c["pearson_w_contrib"] > 0.5                    # HOBBIT's measure still looks strong (w on both sides)
    assert c["same_order"] < 0.5


def test_refit_meets_budget_and_is_tight():
    rng = np.random.default_rng(2)
    _, w = routed(rng, 3000, 512, 10)
    ws, = G.sort_rows(w)
    s = G.cum_score(ws)
    for b, (t2, share) in G.refit_t2(ws, None, (0.01, 0.03, 0.05)).items():
        assert t2 is not None
        assert G.loss_share(ws, None, s > t2) <= b
        if t2 > 0.5:
            assert G.loss_share(ws, None, s > round(t2 - 0.01, 2)) > b
        assert abs(share - np.mean(s > t2)) < 1e-12
    r = G.refit_t2(ws, None, (0.01, 0.05))
    assert r[0.01][0] >= r[0.05][0]                        # a tighter budget never skips more


def test_thresholds_do_not_transfer_from_top2_to_top10():
    rng = np.random.default_rng(3)
    _, w2 = routed(rng, 20000, 8, 2)
    _, w10 = routed(rng, 20000, 512, 10)
    s2 = G.cum_score(G.sort_rows(w2)[0])
    s10 = G.cum_score(G.sort_rows(w10)[0])
    sk2 = G.split(s2, 0.6, 0.9)[2].mean()
    sk10 = G.split(s10, 0.6, 0.9)[2].mean()
    hi10 = G.split(s10, 0.6, 0.9)[0].mean()
    assert sk2 < 0.1 and sk10 > sk2 and hi10 < 0.8         # the same T1/T2 cut a top-10 row very differently


def test_cpu_impact_counts_experts_only_when_every_entry_skips():
    ids = np.array([[1, 2, 3], [4, 2, 5]])
    w = np.array([[0.90, 0.06, 0.04], [0.5, 0.3, 0.2]])
    kind = np.full(ids.shape, -1, np.int8)
    kind[0, 0] = 0                                         # expert 1 in VRAM
    # T2 0.89: row 0 skips experts 2 and 3 (s 0.90, 0.96), row 1 nothing (s 0, 0.5, 0.8)
    ce, se, cx, rm = G.cpu_impact([(ids, kind, w)], 0.89)
    assert (ce, cx) == (5, 4)                              # CPU entries 2,3,4,2,5; experts 2,3,4,5
    assert se == 2 and rm == 1                             # expert 3 removed; expert 2 still needed by row 1


def trace_with(windows_ids, windows_w, kinds, ms):
    wins = []
    for ids, w, kind, m in zip(windows_ids, windows_w, kinds, ms):
        win = wtrace.Window(rows=ids.shape[0], accepted=ids.shape[0] - 1, position=0, ms=m)
        win.layers[0] = wtrace.Layer(ids.astype(np.int16), kind.astype(np.int8), w.astype(np.float32))
        wins.append(win)
    return wtrace.Trace(1, 512, ids.shape[1], wins)


def test_analyse_on_a_window_trace_and_the_gate():
    rng = np.random.default_rng(4)
    ids, w, kinds, ms = [], [], [], []
    for _ in range(50):
        i, ww = routed(rng, 4, 512, 10)
        ids.append(i)
        w.append(ww)
        kinds.append(np.where(rng.random(i.shape) < 0.2, -1, 0))
        ms.append(30.0)
    wr, groups, total_ms = G.from_trace(trace_with(ids, w, kinds, ms))
    assert wr.shape == (200, 10) and len(groups) == 50 and total_ms == 1500.0
    rep = G.analyse(wr, None, groups, total_ms, budgets=(0.03,), us_per_miss=40.0)
    c = rep["cpu"][0.03]
    assert c["cpu_entries"] == sum(int((k == -1).sum()) for k in kinds)
    assert 0 <= c["removed"] <= c["cpu_experts"] and c["skipped"] <= c["cpu_entries"]
    assert abs(c["saving"] - c["removed"] * 0.040 / 1500.0) < 1e-12
    assert rep["correlation"] is None and len(rep["skip_by_rank"]) == 10 and rep["skip_by_rank"][0] == 0.0


def test_inputs_npz_and_dump_routing():
    rng = np.random.default_rng(5)
    ids, w = routed(rng, 96, 512, 10)
    with tempfile.TemporaryDirectory() as d:
        npz = Path(d) / "e.npz"
        np.savez(npz, w=w, out_norm=np.ones_like(w), ids=ids, kind=np.full(ids.shape, -1), group=np.repeat(np.arange(48), 2))
        rep = G.main(["--npz", str(npz), "--budgets", "0.03"])
        assert rep["correlation"]["pearson_w_contrib"] > 0.999 and rep["cpu"][0.03]["cpu_entries"] == 960
        # the engine's --dump-routing records: int32 layer, int32 k, int32 ids[k], float32 weights[k]; 48 layers = one window
        rec = np.zeros((96, 2 + 20), np.int32)
        rec[:, 0] = np.tile(np.arange(48), 2)
        rec[:, 1] = 10
        rec[:, 2:12] = ids
        rec[:, 12:] = w.astype(np.float32).view(np.int32)
        dump = Path(d) / "r.bin"
        rec.tofile(dump)
        rep = G.main(["--dump-routing", str(dump), "--budgets", "0.03"])
        assert rep["rows"] == 96 and rep["cpu"][0.03]["cpu_entries"] == 960   # no placement: every entry counted CPU


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_gate_importance: ok")
