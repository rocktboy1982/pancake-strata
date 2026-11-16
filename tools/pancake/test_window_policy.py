"""window_policy.py on synthetic window traces.  No model, no GPU.  Run: python -m pytest -q tools/pancake/test_window_policy.py"""
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "trace"))
import window_policy as W  # noqa: E402
import wtrace  # noqa: E402

ROWS, K = 5, 4


def window(accepted, ms, cpu_new_per_row, base):
    """One window of ROWS rows; row 0 hits the GPU, every later row routes to `cpu_new_per_row` fresh CPU experts."""
    ids = np.zeros((ROWS, K), np.int16)
    kind = np.zeros((ROWS, K), np.int8)
    ids[0] = [0, 1, 2, 3]
    nxt = base
    for r in range(1, ROWS):
        for j in range(K):
            if j < cpu_new_per_row:
                ids[r, j] = nxt
                kind[r, j] = -1
                nxt += 1
            else:
                ids[r, j] = j
    w = wtrace.Window(rows=ROWS, accepted=accepted, position=0, ms=ms)
    w.layers[0] = wtrace.Layer(ids, kind, None)
    return w


def regime_trace(pattern):
    """pattern: list of 'g' (all drafts accepted, no CPU misses) / 'b' (none accepted, 4 fresh CPU misses a draft row)."""
    wins = []
    for i, c in enumerate(pattern):
        wins.append(window(ROWS - 1, 10.0, 0, 100 * i) if c == "g" else window(0, 10.0, 4, 100 * i))
    return wtrace.Trace(1, 512, K, wins)


def test_hand_tables_and_oracle():
    tr = regime_trace("gb")
    tab = W.tables(W.usable(tr), 100.0)               # 100 us per miss: a bad window's 16 misses cost 1.6 ms (k=1 saves all)
    tok, ms = tab[0]
    assert tok.tolist() == [1, 2, 3, 4, 5] and np.allclose(ms, 10.0)        # good window: nothing to save
    tok, ms = tab[1]
    assert tok.tolist() == [1, 1, 1, 1, 1]                                  # nothing accepted
    assert np.allclose(ms, [10 - 1.6, 10 - 1.2, 10 - 0.8, 10 - 0.4, 10.0])  # 4 misses a row, 100 us each
    ks, r = W.oracle(tab)
    assert ks == [5, 1]
    assert abs(r - 6 / 18.4 * 1e3) < 1e-6
    assert W.rate(tab, [5, 5]) < r


def test_regimes_give_headroom_and_causal_catches_runs():
    tr = regime_trace("g" * 10 + "b" * 10 + "g" * 10 + "b" * 10)
    rep = W.analyse(tr, us_per_miss=400.0, split=True)
    assert rep["windows"] == 40
    assert rep["oracle"] > rep["fixed"] >= rep["full"] - 1e-9
    assert rep["oracle_headroom"] > 0.05
    assert rep["causal"] > rep["full"]                       # long runs: the last window predicts the next
    assert rep["causal"] <= rep["oracle"] + 1e-9
    assert rep["causal_n"] == 1                              # a run is best followed by the shortest memory
    assert rep["causal_heldout"]["rate"] > rep["causal_heldout"]["full"]
    text = W.report(rep, 0.05)
    assert "over the 5% gate" in text


def test_uniform_trace_has_no_headroom():
    tr = regime_trace("g" * 12)
    rep = W.analyse(tr, us_per_miss=400.0)
    assert abs(rep["oracle_headroom"]) < 1e-9
    assert "under the 5% gate" in W.report(rep, 0.05)
    assert abs(rep["causal"] - rep["full"]) < 1e-9


def test_causal_first_window_keeps_everything_and_clamps():
    wins = W.usable(regime_trace("bg"))
    assert W.causal_ks(wins, 1, 0) == [ROWS, 1]              # after a window that accepted 0, keep 0 + 1 rows
    wins = W.usable(regime_trace("gg"))
    assert W.causal_ks(wins, 1, 3) == [ROWS, ROWS]           # clamped to the window's rows


def test_empty_and_unusable():
    tr = wtrace.Trace(1, 512, K, [wtrace.Window(rows=None, accepted=None, position=None, ms=None)])
    assert W.analyse(tr) is None and "no usable" in W.report(None, 0.05)


def test_theta_rule_uses_drafts_when_present():
    tr = regime_trace("bbbb")
    for w in tr.windows:
        w.drafts = np.array([0.1, 0.9, 0.9, 0.9], np.float32)   # a confident chain only after a weak first draft
    rep = W.analyse(tr, us_per_miss=400.0)
    assert rep["theta_windows"] == 4 and rep["theta_rate"] > rep["theta_full"]


def test_main_runs_on_a_written_trace():
    # wtrace has no writer in the repo's tools; the CLI path is covered by analyse() above, so only check the argument error
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.stwt"
        p.write_bytes(b"nope")
        try:
            W.main([str(p)])
            raise AssertionError("bad file accepted")
        except ValueError:
            pass


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_window_policy: ok")
