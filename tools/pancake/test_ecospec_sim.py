"""ecospec_sim.py: the mechanism on synthetic traces (the costs are made up: these check what the arms do, not the gain).

    python -m pytest -q tools/pancake/test_ecospec_sim.py
"""
import io
import math
import os
import random
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "trace"))
import disco_fit as D  # noqa: E402
import draft_costs  # noqa: E402
import ecospec_sim as E  # noqa: E402
import wtrace  # noqa: E402
from test_disco_fit import write_stwt  # noqa: E402


def burst_windows(n_req=80, per_req=60, T=6, seed=3, burst=True):
    """MTP windows whose rows are cheap (a few new CPU misses each) except that, in half of the windows, one draft row 2..T-1
    wakes 120 new experts (+14 ms), independent of the draft probabilities: only a rule that sees the window's own misses can
    skip it.  ms = 18 + 1.0 (T - 1) + 0.12 x draft-row misses + 0.05 x row 0's."""
    rnd = random.Random(seed)
    out, pos = [], 0
    for _ in range(n_req):
        pos += 200 + rnd.randrange(300)
        out.append({"T": 1, "a": 0, "pos": pos, "ms": 900.0, "probs": None, "src": [], "rm": [70]})
        pos += 1
        for _ in range(per_req):
            probs = [min(0.999, max(0.02, rnd.betavariate(4, 1.6))) for _ in range(T - 1)]
            a = 0
            while a < T - 1 and rnd.random() < 1 / (1 + math.exp(-(7 * (probs[a] - 0.55) - 0.5 * a))):
                a += 1
            rm = [60 + rnd.randrange(20)] + [rnd.randrange(7) for _ in range(1, T)]
            if burst and rnd.random() < 0.5:
                rm[rnd.randrange(2, T)] += 120
            ms = 18 + 1.0 * (T - 1) + 0.12 * sum(rm[1:]) + 0.05 * rm[0] + rnd.gauss(0, 0.4)
            out.append({"T": T, "a": a, "pos": pos, "ms": ms, "probs": probs, "src": [0] * (T - 1), "rm": rm})
            pos += a + 1
    return out


def prepare(wins):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.stwt")
        write_stwt(p, wins)
        rounds, _ = D.read_rounds([p])
        tr = [w for w in wtrace.read(p).windows if w.rows and w.ms and w.position is not None and w.accepted is not None]
    cost = D.CostModel(rounds)
    tables, _ = draft_costs.build_tables(tr, max_t=D.KMAXT)
    cut = int(len(rounds) * 0.75)
    model = D.fit(rounds[:cut], D.REPLAY_FEATURES, 1.0, cost)
    return rounds, cost, tables, rounds[cut:], model


def test_oracle_cost_sees_the_expensive_row_the_others_cannot():
    rounds, cost, tables, held, model = prepare(burst_windows())
    buf = io.StringIO()
    res, verdict = E.run(rounds, cost, tables, held, model, out=buf)
    text = buf.getvalue()
    table, ema, orac = (res["eco-table (DraftPolicy cost table)"], res["eco-ema (expected new misses)"],
                        res["eco-oracle (a perfect expert predictor)"])
    ceiling = res["oracle cut (verify exactly what held)"]
    assert orac > 1.02 * max(table, ema), (table, ema, orac)   # a perfect predictor skips the burst rows: the others cannot see them
    assert orac <= ceiling * (1 + 1e-9), "no chain-stop rule beats the oracle cut"
    assert ceiling >= max(v for k, v in res.items() if not k.startswith("oracle")) * (1 - 1e-9)
    assert verdict["eco_oracle_vs_best"] > E.GAIN_BAR
    assert "best baseline:" in text and "eco-oracle" in text and "STRATA_SPEC_CUM 0.3" in text


def test_without_bursts_the_expected_cost_is_as_good_as_the_oracle():
    rounds, cost, tables, held, model = prepare(burst_windows(burst=False, seed=5))
    res, _ = E.run(rounds, cost, tables, held, model, out=io.StringIO())
    ema, orac = res["eco-ema (expected new misses)"], res["eco-oracle (a perfect expert predictor)"]
    assert abs(orac / ema - 1.0) < 0.03, (ema, orac)   # nothing the predictor could add when every row costs the same


def test_every_arm_returns_a_legal_window():
    rounds, cost, tables, held, model = prepare(burst_windows(n_req=20, per_req=30, seed=9))
    costs = E.Costs(rounds, cost, tables, 6)
    for src in ("table", "ema", "oracle"):
        cut = E.cut_eco(model, costs, src)
        for r in held:
            if r["mtp"]:
                assert 0 <= cut(r) <= r["T"] - 1


def test_main_prints_the_verdict(capsys, tmp_path):
    p = tmp_path / "t.stwt"
    write_stwt(str(p), burst_windows(n_req=40, per_req=40, seed=1))
    assert E.main([str(p), "--min-windows", "100"]) == 0
    text = capsys.readouterr().out
    assert "eco-ema" in text and "engine switch" in text
    assert E.main([str(p), "--min-windows", "100000"]) == 2   # too few windows: refused


def test_draft_costs_falls_back_to_the_engine_row_counts():
    """A trace without layer records (a window trace's record 5 only) still prices the extra rows."""
    wins = burst_windows(n_req=20, per_req=40, seed=2)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.stwt")
        write_stwt(p, wins)
        tr = [w for w in wtrace.read(p).windows if w.rows and w.ms and w.position is not None and w.accepted is not None]
    adds = draft_costs.new_misses_per_row(tr)
    assert adds[0] > 50 and 0 < adds[1] < 70, adds   # row 0 ~ 70 misses; a draft row a few (plus the bursts)
