"""test_disco_fit.py - DISCO's offline fit (MOD_PLAN A22) on synthetic window traces and DISCO csv traces (no GPU, no model).

    python tools/pancake/test_disco_fit.py      (or: python -m pytest tools/pancake/test_disco_fit.py)
"""
import io
import math
import os
import random
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disco_fit  # noqa: E402


def write_stwt(path, windows, nl=2, ne=64, k=4):
    """windows: dicts T, a (None: merged), pos, ms, probs (list or None), src (list), rm (row misses or None)."""
    with open(path, "wb") as f:
        f.write(b"STWT" + struct.pack("<4I", 1, nl, ne, k))
        for w in windows:
            acc = 255 if w["a"] is None else w["a"]
            f.write(struct.pack("<BBBBqf", 2, w["T"], acc, 0, w["pos"], w["ms"]))
            if w.get("rm") is not None:
                f.write(struct.pack("<BBH", 5, len(w["rm"]), 0) + np.array(w["rm"], "<u2").tobytes())
            if w.get("probs") is not None:
                n = len(w["probs"])
                f.write(struct.pack("<BBH", 4, n, 0) + np.array(w["probs"], "<f4").tobytes() + bytes(w["src"]))


def synth_windows(n_req=60, per_req=60, T=6, seed=7, lookup_every=0):
    """Requests of one-row first windows then MTP windows of T rows.  Draft j holds with sigmoid(7 (p - 0.55) - 0.5 j);
    row j wakes ~36 / j new CPU experts; ms = 18 + 1.5 (T - 1) + 0.12 x draft-row misses + 0.05 x row 0's."""
    rnd = random.Random(seed)
    out, pos = [], 0
    for _ in range(n_req):
        pos += 200 + rnd.randrange(300)   # the prompt
        out.append({"T": 1, "a": 0, "pos": pos, "ms": 900.0, "probs": None, "src": [], "rm": [70]})
        pos += 1
        for i in range(per_req):
            if lookup_every and i % lookup_every == lookup_every - 1:
                probs = [float("nan")] * (T - 1)
                a = rnd.randrange(T)
                rm = [60] + [10] * (T - 1)
                ms = 18 + 1.5 * (T - 1) + 0.12 * sum(rm[1:]) + 0.05 * rm[0]
                out.append({"T": T, "a": a, "pos": pos, "ms": ms, "probs": probs, "src": [2] * (T - 1), "rm": rm})
                pos += a + 1
                continue
            probs = [min(0.999, max(0.02, rnd.betavariate(4, 1.6))) for _ in range(T - 1)]
            a = 0
            while a < T - 1 and rnd.random() < 1 / (1 + math.exp(-(7 * (probs[a] - 0.55) - 0.5 * a))):
                a += 1
            rm = [60 + rnd.randrange(20)] + [int(36 / j + rnd.randrange(6)) for j in range(1, T)]
            ms = 18 + 1.5 * (T - 1) + 0.12 * sum(rm[1:]) + 0.05 * rm[0] + rnd.gauss(0, 0.5)
            out.append({"T": T, "a": a, "pos": pos, "ms": ms, "probs": probs, "src": [0] * (T - 1), "rm": rm})
            pos += a + 1
    return out


def test_replayed_features_match_the_engine_state_machine():
    wins = synth_windows(n_req=1, per_req=3)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.stwt")
        write_stwt(p, wins)
        rounds, kinds = disco_fit.read_rounds([p])
    assert kinds == {"stwt"} and len(rounds) == 4
    r0, r1, r2 = rounds[0], rounds[1], rounds[2]
    assert not r0["mtp"] and r1["mtp"] and r2["mtp"]
    assert not r0["ms_ok"] and r1["ms_ok"] and r2["ms_ok"], "the first window's time includes the prompt; the rest follow"
    # first MTP window: no MTP window before it -> recent 0.7, feature 6 = 0; ctx from the one-row window's position
    x = r1["rows"][0][2]
    w1 = wins[1]
    assert abs(x[0] - w1["probs"][0]) < 1e-6 and abs(x[1] - (-math.log(w1["probs"][0]) / 5)) < 1e-6
    assert x[4] == 0.7 and x[6] == 0.0 and abs(x[8] - math.log2(1 + wins[0]["pos"]) / 17) < 1e-12
    assert abs(r1["rows"][1][2][2] - w1["probs"][0] * w1["probs"][1]) < 1e-6 and r1["rows"][1][2][3] == 1 / 8
    # labels: accepted / first rejected / not reached
    a1 = w1["a"]
    assert [lab for _, lab, _ in r1["rows"]] == [1 if j < a1 else 0 if j == a1 else -1 for j in range(5)]
    # second MTP window: feature 6 of draft j = the first MTP window's row j + 1 misses / 64; recent updated
    x2 = [x for _, _, x in r2["rows"]]
    for j in range(5):
        assert abs(x2[j][6] - w1["rm"][j + 1] / 64.0) < 1e-12
    rec = 0.7
    for j in range(min(5, a1 + 1)):
        rec = 0.9 * rec + 0.1 * (1.0 if j < a1 else 0.0)
    assert abs(x2[0][4] - rec) < 1e-12 and abs(x2[0][8] - math.log2(1 + w1["pos"]) / 17) < 1e-12


def test_cost_model_recovers_the_synthetic_costs():
    wins = synth_windows()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.stwt")
        write_stwt(p, wins)
        rounds, _ = disco_fit.read_rounds([p])
    cost = disco_fit.CostModel(rounds)
    assert cost.coef is not None
    c = cost.coef
    # T is constant (6) in the MTP windows: the per-row ms cannot be told from the intercept and takes the prior;
    # the miss terms are identifiable
    assert cost.row_fixed == disco_fit.ROW_MS_PRIOR and c[1] == disco_fit.ROW_MS_PRIOR
    assert abs(c[2] - 0.12) < 0.02 and abs(c[3] - 0.05) < 0.03, c
    assert disco_fit.CostModel(rounds, row_ms=1.5).coef[1] == 1.5
    # sizes that vary: the per-row ms is fitted (true 1.5)
    mixed = []
    for T in (3, 4, 5, 6):
        mixed += synth_windows(n_req=8, per_req=40, T=T, seed=T)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.stwt")
        write_stwt(p, mixed)
        rm, _ = disco_fit.read_rounds([p])
    cm = disco_fit.CostModel(rm)
    assert cm.row_fixed is None and abs(cm.coef[1] - 1.5) < 0.2 and abs(cm.coef[2] - 0.12) < 0.02, cm.coef
    r = next(r for r in rounds if r["mtp"])
    full = cost.cut_ms(r, r["T"] - 1)
    assert full == r["ms"] and cost.cut_ms(r, 0) < full and cost.cut_ms(r, 2) < full


def test_fit_beats_the_logged_rule_and_static_on_held_out_windows():
    wins = synth_windows(seed=11, lookup_every=9)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.stwt")
        out = os.path.join(d, "w.txt")
        write_stwt(p, wins)
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            rc = disco_fit.main([p, "-o", out])
        finally:
            sys.stdout = old
        text = buf.getvalue()
        assert rc == 0, text
        m = disco_fit.read_weights(out)
        assert m["w"][5] == 0.0 and m["w"][7] == 0.0, "the features a window trace cannot replay stay at 0"
        assert m["w"][0] > 0 and m["w"][3] < 0, m["w"]   # sure drafts hold, deep drafts less
        rounds, _ = disco_fit.read_rounds([p])
        held = rounds[int(len(rounds) * 0.75):]
        cost = disco_fit.CostModel(rounds)
        mtp = [r for r in held if r["mtp"]]
        base = disco_fit.replay(mtp, disco_fit.logged, cost)["tok_per_ms"]
        dis = disco_fit.replay(mtp, disco_fit.cut_disco(m), cost)["tok_per_ms"]
        assert m.get("thresholds") and len(m["thresholds"]) == 5, m
        best_static = max(disco_fit.replay(mtp, disco_fit.cut_static(n), cost)["tok_per_ms"] for n in range(6))
        orac = disco_fit.replay(mtp, disco_fit.cut_oracle, cost)["tok_per_ms"]
        assert dis > base and dis >= best_static * 0.995 and orac >= dis, (base, dis, best_static, orac)
        assert "DISCO per-position table" in text and "oracle cut" in text and "AUC classifier" in text
        # --simulate replays the file
        buf = io.StringIO()
        sys.stdout = buf
        try:
            rc = disco_fit.main([p, "--simulate", out])
        finally:
            sys.stdout = old
        assert rc == 0 and "threshold sweep" in buf.getvalue()
        # an old (version 1) file is refused, as the engine refuses it
        Path(out).write_text(Path(out).read_text().replace("disco 2", "disco 1"))
        try:
            disco_fit.read_weights(out)
            assert False, "disco 1 accepted"
        except ValueError:
            pass


def test_disco_csv_with_nmiss_and_appended_segments():
    head = "round,idx,label,T,a,round_ms," + ",".join(f"f{i}" for i in range(9)) + ",nmiss\n"
    feats = lambda p, j: ",".join(f"{v:.5g}" for v in [p, -math.log(p) / 5, p, j / 8, 0.7, 0, 0.5, 0.35, 0.6])
    lines = [head,
             f"0,0,1,3,1,20.0,{feats(0.9, 0)},40\n", f"0,1,0,3,1,20.0,{feats(0.4, 1)},20\n",
             "1,-1,-1,1,0,12.0," + ",".join("0" for _ in range(9)) + ",-1\n",
             # a second process appended: round ids restart
             f"0,0,0,2,0,15.0,{feats(0.3, 0)},33\n"]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.csv")
        Path(p).write_text("".join(lines))
        rounds, kinds = disco_fit.read_rounds([p])
    assert kinds == {"csv"} and len(rounds) == 3
    assert rounds[0]["T"] == 3 and rounds[0]["miss"] == [0, 40, 20] and [l for _, l, _ in rounds[0]["rows"]] == [1, 0]
    assert rounds[1]["T"] == 1 and rounds[1]["rows"] == []
    assert rounds[2]["miss"] == [0, 33] and rounds[2]["key"][0] == 1


def test_replay_arithmetic():
    class Flat:
        def cut_ms(self, r, k):
            return 10.0 + k

        def size_ms(self, t):
            return 10.0 + t - 1

    r = {"T": 4, "a": 2, "ms": 13.0, "ms_ok": True, "mtp": True, "rows": [(j, 1, [0.9] * 9) for j in range(3)], "miss": None}
    other = {"T": 3, "a": 1, "ms": 7.0, "ms_ok": True, "mtp": False, "rows": [], "miss": None}
    v = disco_fit.replay([r, other], disco_fit.cut_static(1), Flat())
    assert v["tokens"] == 2 + 2 and v["ms"] == 11.0 + 7.0 and v["windows"] == 2
    v = disco_fit.replay([r], disco_fit.cut_oracle, Flat())
    assert v["tokens"] == 3 and v["ms"] == 12.0


def test_spec_cum_arm_is_the_chain_threshold():
    """F8 STRATA_SPEC_CUM s keeps the leading drafts whose probability product is >= s; s >= min_p makes the floor redundant."""
    wins = synth_windows(seed=5)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.stwt")
        write_stwt(p, wins)
        rounds, _ = disco_fit.read_rounds([p])
    mtp = [r for r in rounds if r["mtp"]]
    assert mtp
    for s in (0.1, 0.3, 0.5, 0.7):
        for r in mtp:
            want = 0
            for idx, _, x in r["rows"]:
                if x[2] < s:
                    break
                want += 1
            assert disco_fit.cut_cum(s)(r) == want
            if s >= 0.5:
                assert disco_fit.cut_cum(s, 0.5)(r) == want, "chain <= p, so s >= min_p already implies the floor"
    cost = disco_fit.CostModel(rounds)
    buf = io.StringIO()
    disco_fit.report_arms(rounds, cost, out=buf)
    assert "STRATA_SPEC_CUM 0.3" in buf.getvalue() and "STRATA_SPEC_CUM 0.3 + --spec-min-p 0.5" in buf.getvalue()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

