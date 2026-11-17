"""The weighted gate-importance route (MOD_PLAN.md section 6 item 20): a served run's STRATA_WINDOW_TRACE carries the
published router weights, the served --dump-routing file has unit weights.  Synthetic traces, no model, no GPU.

Run: python -m pytest tools/pancake/test_gate_importance_weighted.py"""
import ast
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "trace"))
import gate_importance as G  # noqa: E402
import wtrace  # noqa: E402

NL, NE, K = 48, 512, 10


def write_stwt(path, windows, weighted=True, unweighted_layers=()):
    """windows: list of (ids [NL][n, K], w [NL][n, K]); layer records in the window_trace.hpp layout."""
    with open(path, "wb") as f:
        f.write(b"STWT" + struct.pack("<4I", 1, NL, NE, K))
        for ids, w in windows:
            f.write(struct.pack("<BBBBqf", 2, ids[0].shape[0], 255, 0, 0, 1.0))
            for l in range(NL):
                n = ids[l].shape[0]
                has = weighted and l not in unweighted_layers
                f.write(struct.pack("<BBBB", 1, l, n, 1 if has else 0))
                for i in ids[l].ravel():
                    f.write(struct.pack("<hb", int(i), -1))
                if has:
                    f.write(np.asarray(w[l], "<f4").tobytes())


def synth(rng, nwin=4, n=3):
    wins = []
    for _ in range(nwin):
        ids, ws = [], []
        for _l in range(NL):
            i = np.stack([rng.choice(NE, K, replace=False) for _ in range(n)])
            p = rng.random((n, K)) + 0.05
            ids.append(i)
            ws.append(p / p.sum(axis=1, keepdims=True))
        wins.append((ids, ws))
    return wins


def test_weighted_stwt_emits_a_48x512_table_of_weight_sums():
    rng = np.random.default_rng(7)
    wins = synth(rng)
    with tempfile.TemporaryDirectory() as d:
        t, out = Path(d) / "r.stwt", Path(d) / "imp.txt"
        write_stwt(t, wins)
        G.main([str(t), "--emit-importance", str(out), "--budgets", "0.05"])
        imp = G.read_importance(out)
        cnt = G.expert_importance(wtrace.read(str(t)), counts=True)
    assert imp.shape == cnt.shape == (NL, NE)
    want = np.zeros((NL, NE))
    for ids, ws in wins:
        for l in range(NL):
            np.add.at(want[l], ids[l].ravel(), ws[l].ravel())
    assert np.allclose(imp, want, rtol=1e-6)
    assert np.allclose(imp.sum(axis=1), sum(w[1][0].shape[0] for w in wins))   # normalised weights: one unit per token-layer
    assert not np.allclose(imp, cnt) and np.allclose(cnt.sum(axis=1), imp.sum(axis=1) * K)   # same shape, different content


def test_unweighted_stwt_fails_without_counts_and_partial_skips_unweighted_layers():
    rng = np.random.default_rng(8)
    wins = synth(rng, nwin=2)
    with tempfile.TemporaryDirectory() as d:
        t = Path(d) / "u.stwt"
        write_stwt(t, wins, weighted=False)
        tr = wtrace.read(str(t))
        assert G.weight_coverage(tr) == (0, 2 * NL)
        for partial in (False, True):
            try:
                G.expert_importance(tr, partial=partial)
                raise AssertionError("no weights accepted")
            except ValueError:
                pass
        write_stwt(t, wins, unweighted_layers={5})
        tr = wtrace.read(str(t))
        assert G.weight_coverage(tr) == (2 * (NL - 1), 2 * NL)
        try:
            G.expert_importance(tr)
            raise AssertionError("mixed trace accepted without partial")
        except ValueError:
            pass
        imp = G.expert_importance(tr, partial=True)
        assert imp[5].sum() == 0 and imp[4].sum() > 0


def test_conv_trace_engine_env():
    src = (HERE / "conv_trace.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "engine_env")
    ns = {}
    exec(compile(ast.Module([fn], []), "conv_trace.py", "exec"), ns)
    f = ns["engine_env"]
    assert f([]) == {} and f(None, None) == {}
    assert f(["A=1", "B=x=y"], "/tmp/w.stwt") == {"A": "1", "B": "x=y", "STRATA_WINDOW_TRACE": "/tmp/w.stwt"}
    try:
        f(["nokey"])
        raise AssertionError("bad pair accepted")
    except SystemExit:
        pass


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_gate_importance_weighted: ok")
