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
        assert G.weight_coverage(tr) == (0, 0, 2 * NL)
        for partial in (False, True):
            try:
                G.expert_importance(tr, partial=partial)
                raise AssertionError("no weights accepted")
            except ValueError:
                pass
        write_stwt(t, wins, unweighted_layers={5})
        tr = wtrace.read(str(t))
        assert G.weight_coverage(tr) == (2 * (NL - 1), 2 * (NL - 1), 2 * NL)
        try:
            G.expert_importance(tr)
            raise AssertionError("mixed trace accepted without partial")
        except ValueError:
            pass
        imp = G.expert_importance(tr, partial=True)
        assert imp[5].sum() == 0 and imp[4].sum() > 0


def zero_windows(rng):
    wins = synth(rng, nwin=2)
    return [(ids, [np.zeros_like(w) for w in ws]) for ids, ws in wins]


def test_all_zero_weighted_stwt_is_refused_with_the_cause():
    """chain W's real2.stwt (T-834): every layer record flagged weighted, every weight 0.0 -> the tools must say why."""
    rng = np.random.default_rng(9)
    with tempfile.TemporaryDirectory() as d:
        t, out = Path(d) / "z.stwt", Path(d) / "imp.txt"
        write_stwt(t, zero_windows(rng))
        tr = wtrace.read(str(t))
        assert G.weight_coverage(tr) == (2 * NL, 0, 2 * NL)   # present, nonzero, all
        for call in (lambda: G.check_weights(tr), lambda: G.expert_importance(tr), lambda: G.expert_importance(tr, partial=True),
                     lambda: G.from_trace(tr)):
            try:
                call()
                raise AssertionError("all-zero weights accepted")
            except ValueError as e:
                assert "the trace weights are all 0" in str(e) and "section 6 item 20" in str(e)
        for argv in ([str(t), "--emit-importance", str(out)], [str(t), "--emit-importance", str(out), "--allow-partial"], [str(t)]):
            try:
                G.main(argv)
                raise AssertionError("main accepted all-zero weights")
            except SystemExit as e:
                assert e.code == 2
        assert not out.exists()
        G.main([str(t), "--emit-importance", str(out), "--counts"])   # counts need only the ids
        assert G.read_importance(out).sum() > 0


def test_weights_not_summing_to_one_are_refused_unless_the_check_is_off():
    rng = np.random.default_rng(10)
    wins = [(ids, [w * 3.0 for w in ws]) for ids, ws in synth(rng, nwin=2)]
    with tempfile.TemporaryDirectory() as d:
        t, out = Path(d) / "s.stwt", Path(d) / "imp.txt"
        write_stwt(t, wins)
        tr = wtrace.read(str(t))
        try:
            G.check_weights(tr)
            raise AssertionError("rows summing to 3 accepted")
        except ValueError as e:
            assert "do not sum to 1" in str(e)
        G.check_weights(tr, tol=0)
        G.main([str(t), "--emit-importance", str(out), "--weight-sum-tol", "0"])
        assert out.exists()
        write_stwt(t, synth(rng, nwin=2))
        G.check_weights(wtrace.read(str(t)))   # real normalised weights pass


def test_zero_records_among_real_ones_count_as_unweighted():
    rng = np.random.default_rng(11)
    wins = synth(rng, nwin=2)
    wins[0][1][3][:] = 0.0   # one layer record flagged but all 0
    with tempfile.TemporaryDirectory() as d:
        t = Path(d) / "m.stwt"
        write_stwt(t, wins)
        tr = wtrace.read(str(t))
        assert G.weight_coverage(tr) == (2 * NL, 2 * NL - 1, 2 * NL)
        G.check_weights(tr)
        try:
            G.expert_importance(tr)
            raise AssertionError("mixed trace accepted without partial")
        except ValueError:
            pass
        assert G.expert_importance(tr, partial=True).sum() > 0


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
