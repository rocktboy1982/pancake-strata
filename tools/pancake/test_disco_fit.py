"""test_disco_fit.py - DISCO's offline fit on a synthetic trace (no GPU, no model).

    python tools/pancake/test_disco_fit.py      (or: python -m pytest tools/pancake/test_disco_fit.py)
"""
import math
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disco_fit  # noqa: E402


def synth_trace(path, n_rounds=3000, T=6, seed=7):
    """rounds of T rows: draft j holds with a chance that follows its probability and falls with depth;
    one more row costs 3 ms on a 20 ms round (an offloaded MoE: each row wakes experts)"""
    rnd = random.Random(seed)
    with open(path, "w") as f:
        f.write("round,idx,label,T,a,round_ms," + ",".join(f"f{i}" for i in range(9)) + "\n")
        recent = 0.7
        for rid in range(n_rounds):
            ps, held = [], []
            for j in range(T - 1):
                p = min(0.999, max(0.01, rnd.betavariate(4, 1.5) - 0.05 * j))
                ps.append(p)
                held.append(rnd.random() < p * (0.95 - 0.03 * j))
            a = 0
            while a < T - 1 and held[a]:
                a += 1
            ms = 20.0 + 3.0 * (T - 1) + rnd.gauss(0, 0.5)
            chain = 1.0
            for j, p in enumerate(ps):
                chain *= p
                lab = 1 if j < a else 0 if j == a else -1
                x = [p, -math.log(p) / 5, chain, j / 8, recent, 0.0, 0.3, 3.0 / (20 + 3 * j), 0.6]
                f.write(f"{rid},{j},{lab},{T},{a},{ms:.3f}," + ",".join(f"{v:.5g}" for v in x) + "\n")
            for j in range(min(a + 1, T - 1)):
                recent = 0.9 * recent + 0.1 * (1.0 if j < a else 0.0)
    return path


def test_fit_writes_a_loadable_file_that_beats_static():
    with tempfile.TemporaryDirectory() as d:
        tr = synth_trace(Path(d) / "t.csv")
        out = Path(d) / "w.txt"
        assert disco_fit.main([str(tr), "-o", str(out)]) == 0
        txt = out.read_text()
        assert "disco 1" in txt and "\nw " in txt and "\nthreshold " in txt
        m = disco_fit.read_weights(out)
        assert 0.0 < m["threshold"] < 1.0
        assert m["w"][0] > 0, "the draft probability must raise the acceptance score"
        rounds = disco_fit.read_traces([str(tr)])
        cost, _ = disco_fit.cost_table(rounds)
        dv = disco_fit.replay(rounds, disco_fit.disco_cut(m, m["threshold"]), cost)[2]
        best_static = max(disco_fit.replay(rounds, (lambda L: lambda r: L - 1)(L), cost)[2] for L in range(1, 7))
        assert dv >= best_static * 0.99, (dv, best_static)
        # replaying the written file gives the same answer
        assert disco_fit.main([str(tr), "--simulate", str(out)]) == 0


def test_appended_file_segments_and_short_traces():
    with tempfile.TemporaryDirectory() as d:
        tr = Path(d) / "t.csv"
        synth_trace(tr, n_rounds=50)
        body = tr.read_text().split("\n", 1)[1]
        with open(tr, "a") as f:   # a second process appending: round ids restart at 0
            f.write(body)
        assert len(disco_fit.read_traces([str(tr)])) == 100
        assert disco_fit.main([str(tr)]) == 2, "fewer than --min-rounds rounds: refused"


def test_replay_arithmetic():
    rounds = [{"T": 4, "a": 2, "ms": 10.0, "rows": []}, {"T": 4, "a": 0, "ms": 10.0, "rows": []}]
    cost = {1: 5.0, 2: 6.0, 3: 7.0, 4: 8.0}
    tok, ms, _ = disco_fit.replay(rounds, lambda r: 3, cost)
    assert (tok, ms) == (3 + 1, 16.0)
    tok, ms, _ = disco_fit.replay(rounds, lambda r: 1, cost)
    assert (tok, ms) == (2 + 1, 12.0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
