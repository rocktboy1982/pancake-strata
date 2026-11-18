"""Tests for replay_fill_transition.py on small synthetic routing (a 4-layer, 16-expert model patched into
replay_cache_policy).  python -m pytest -q tools/pancake/test_replay_fill_transition.py"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_fill_transition as F  # noqa: E402

R = F.R
NL, NE, K = 4, 16, 2


@pytest.fixture(autouse=True)
def small_model(monkeypatch):
    monkeypatch.setattr(R, "NL", NL)
    monkeypatch.setattr(R, "NE", NE)


def windows_from(pattern):
    """pattern(wi, l) -> (tokens, K) ids."""
    return lambda n: [[(l, pattern(wi, l)) for l in range(NL)] for wi in range(n)]


def init_slots(n_slots):
    """The first n_slots (layer, expert) pairs in layer-major order, as the profile fill would hand them out."""
    return np.array([(i // 8, i % 8) for i in range(n_slots)], np.int64)


def run(windows, name, slots=16, every=2):
    return F.simulate(windows, init_slots(slots), slots, R.Policy(every=every, swaps=96), F.parse(name), nl=NL, ne=NE, k=K)


def test_parse():
    assert F.parse("base")["tr"] == 0.0
    c = F.parse("ff:0.5+tr:2+pm")
    assert c["ff"] == 0.5 and c["tr"] == 2.0 and c["pm"]


def test_base_matches_the_reference_simulation():
    rng = np.random.default_rng(1)
    ws = windows_from(lambda wi, l: rng.integers(0, 12, size=(3, K)))(30)
    ref = R.simulate(ws, init_slots(16), R.Policy(every=2, swaps=96))
    got = run(ws, "base")
    assert abs(got["hits"].sum() / got["ents"].sum() - ref["share"]) < 1e-12
    assert abs(got["swaps"] * len(ws) - ref["swaps"]) < 1e-9


def test_fill_first_without_hot_misses_changes_nothing():
    ws = windows_from(lambda wi, l: np.array([[0, 1], [1, 2]]))(20)   # everything routed is already resident
    a, b = run(ws, "base"), run(ws, "ff:0.5")
    assert a["hits"].sum() == b["hits"].sum() and b["swaps"] == 0


def test_fill_first_admits_on_first_miss_into_an_empty_slot():
    stat = F.Stat(NL, NE, 0.7, core_n=4)
    slot = np.full((NL, NE), -1, np.int32)
    res = np.zeros((NL, NE), bool)
    for i, (l, e) in enumerate(init_slots(8)):
        slot[l, e] = i
        res[l, e] = True
    raw = np.zeros((NL, NE))
    raw[0, 12] = 1.2   # one miss in layer 0 (slots 0..7 are layer 0's: 4 claimed, 4 placeholders)
    raw[0, 13] = 0.5   # below a first miss
    stat.observe(raw, slot, 0.7)
    assert stat.claimed[0, :4].all() and not stat.claimed[0, 4:8].any()
    u = raw.copy()
    stat.apply_fill(u, res)
    assert u[0, 12] >= 2.0 and u[0, 13] == 0.0
    assert (u[0, 4:8] == -1.0).all() and (u[0, :4] == raw[0, :4]).all()
    assert (u[1] == raw[1]).all()   # layer 1 holds nothing: no empty slot, untouched


def test_truly_empty_slots_cost_hits_until_filled():
    ws = windows_from(lambda wi, l: np.array([[0, 1], [2, 3]]))(12)
    full, empty = run(ws, "base", slots=16), run(ws, "ffe:0.25", slots=16)
    assert empty["hits"].sum() < full["hits"].sum()


def test_transition_protects_the_expert_that_follows():
    stat = F.Stat(1, 8, 0.7, trans_w=1.0)
    A, B = [0, 1, 4], [2, 3, 4]
    raw = np.zeros((1, 8))
    slot = np.full((1, 8), -1, np.int32)
    slot[0, [0, 1, 2, 5]] = [0, 1, 2, 3]
    for r in range(12):
        raw = raw * 0.7
        raw[0, A if r % 2 == 0 else B] += 1.0
        stat.observe(raw, slot, 0.7)
    u = raw.copy()
    stat.apply_trans(u, slot >= 0)   # state B: the next set is A
    assert u[0, 0] - raw[0, 0] > 1.5 and u[0, 5] == raw[0, 5]
    assert u[0, 0] - raw[0, 0] > 2 * (u[0, 2] - raw[0, 2])
    assert stat.s < 0.5


def test_disjoint_round_is_a_switch():
    stat = F.Stat(1, 8, 0.7, trans_w=1.0)
    slot = np.zeros((1, 8), np.int32)
    raw = np.zeros((1, 8))
    for r in range(8):
        raw = raw * 0.7
        raw[0, [0, 1, 4] if r % 2 == 0 else [2, 3, 4]] += 1.0
        stat.observe(raw, slot, 0.7)
    before = stat.s
    raw = raw * 0.7
    raw[0, [6, 7]] += 1.0
    stat.observe(raw, slot, 0.7)
    assert stat.s > before + 0.2


def test_bootstrap_of_identical_results_is_zero():
    rng = np.random.default_rng(0)
    h = rng.integers(50, 100, 40).astype(float)
    r = dict(hits=h, ents=np.full(40, 100.0), dh=h, dn=np.full(40, 100.0), swaps=1.0)
    out = F.summarize({"base": r, "x": dict(r)}, nboot=200, block=4)
    assert out["x"]["diff"] == 0.0 and out["x"]["lo"] == 0.0 and out["x"]["hi"] == 0.0


def test_bootstrap_interval_contains_a_real_gain():
    rng = np.random.default_rng(0)
    h = rng.integers(60, 80, 80).astype(float)
    base = dict(hits=h, ents=np.full(80, 100.0), dh=h, dn=np.full(80, 100.0), swaps=1.0)
    better = dict(base, hits=h + 3, dh=h + 3)
    out = F.summarize({"base": base, "x": better}, nboot=400, block=4)["x"]
    assert abs(out["diff"] - 3.0) < 1e-9 and out["lo"] > 2.9 and out["hi"] < 3.1


# ---- multi-conversation trace: sidecar turns, switches, real router weights (chain V) ----

def write_dump(path, wins, wt=None):
    """wins: list of (rows, K) id arrays; layer l routes (ids + l) % NE; wt: one (K,) float32 weight row for every entry, or zeros."""
    w = np.zeros(K, np.float32) if wt is None else np.asarray(wt, np.float32)
    recs = [np.concatenate([[l, K], (ids[r] + l) % NE, w.view(np.int32)]).astype(np.int32)
            for ids in wins for l in range(NL) for r in range(ids.shape[0])]
    np.array(recs, np.int32).tofile(path)


def test_load_dump_matches_the_reference_loader_and_offsets(tmp_path):
    wins = [np.array([[0, 1], [2, 3]]), np.array([[4, 5]]), np.array([[6, 7], [8, 9], [1, 2]])]
    p = tmp_path / "d.bin"
    write_dump(p, wins)
    got, k, starts, weights, total = F.load_dump(str(p))
    ref, kref = R.load_trace(str(p))
    assert k == kref == K and len(got) == len(ref) == 3
    for a, b in zip(got, ref):
        assert [(l, ids.tolist()) for l, ids in a] == [(l, ids.tolist()) for l, ids in b]
    assert starts.tolist() == [0, 2 * NL, 3 * NL] and total == (2 + 1 + 3) * NL
    assert not F.has_real_weights(weights)


def test_has_real_weights_rejects_zero_and_unit():
    z = [{0: np.zeros((2, K), np.float32)}]
    u = [{0: np.ones((2, K), np.float32)}]
    r = [{0: np.tile(np.linspace(0.05, 0.3, K, dtype=np.float32), (2, 1))}]
    assert not F.has_real_weights(z) and not F.has_real_weights(u) and F.has_real_weights(r)


def test_segment_turns_and_switches():
    reqs = [dict(end_record=20, conv="A", domain="code"), dict(end_record=40, conv="A", domain="code"),
            dict(end_record=60, conv="B", domain="prose"), dict(end_record=80, conv="A", domain="code")]
    starts = np.array([0, 8, 16, 20, 30, 40, 52, 60, 70])
    turn, turns = F.segment(reqs, starts, 80)
    assert turn.tolist() == [0, 0, 0, 1, 1, 2, 2, 3, 3]
    assert [t["switch"] for t in turns] == [False, False, True, True]
    assert F.switch_windows(turns) == [5, 7]
    post, other = F.subset_masks(9, [5, 7], 2)
    assert post.tolist() == [False] * 5 + [True] * 4 and not (post & other).any()


def test_real_prob_mass_counts_weight_times_k_and_falls_back_to_one():
    heavy = np.array([[0, 1]])      # expert 0 carries the weight, 1 is the tail
    wts = {l: np.array([[0.9, 0.1]], np.float32) for l in range(NL)}
    ws = windows_from(lambda wi, l: heavy)(8)
    base = run(ws, "base")
    res = F.simulate(ws, init_slots(16), 16, R.Policy(every=2, swaps=96), F.parse("pmw"), nl=NL, ne=NE, k=K,
                     weights=[wts] * 8)
    assert res["hits"].sum() == base["hits"].sum()               # nothing to place differently here: same hits
    none = F.simulate(ws, init_slots(16), 16, R.Policy(every=2, swaps=96), F.parse("pmw"), nl=NL, ne=NE, k=K,
                      weights=[None] * 8)
    assert none["hits"].sum() == base["hits"].sum() and none["swaps"] == base["swaps"]   # no weights: +1 as the engine


def test_real_prob_mass_prefers_the_heavy_expert():
    # layer 0 routes 9 twice (weight .1 each) and 8, 10 once (weight .9): count order puts 9 first, weight mass puts 8 / 10 first
    def w0(l):
        return np.array([[8, 9], [10, 9]]) if l == 0 else np.array([[0, 1], [1, 2]])
    ws = [[(l, w0(l)) for l in range(NL)] for _ in range(12)]
    wt = [{l: np.array([[.9, .1], [.9, .1]], np.float32) for l in range(NL)} for _ in range(12)]
    pol = R.Policy(every=2, swaps=1, margin=0.0, mincand=0.0)
    base = F.simulate(ws, init_slots(8), 8, pol, F.parse("base"), nl=NL, ne=NE, k=2)
    pmw = F.simulate(ws, init_slots(8), 8, pol, F.parse("pmw"), nl=NL, ne=NE, k=2, weights=wt)
    assert base["hits"][2] == 2.0 and pmw["hits"][2] == 1.0      # base admits 9 (two entries a window), pmw admits 8 or 10 (one)
    assert pmw["hits"][-1] == base["hits"][-1] == 4.0


def test_trb_feeds_the_boundary_and_changes_the_switch_state():
    rng = np.random.default_rng(3)
    a = [rng.integers(0, 4, size=(2, K)) for _ in range(12)]
    ws = windows_from(lambda wi, l: a[wi % 12])(24)
    pol = R.Policy(every=2, swaps=96)
    plain = F.simulate(ws, init_slots(16), 16, pol, F.parse("tr:2"), nl=NL, ne=NE, k=K)
    marked = F.simulate(ws, init_slots(16), 16, pol, F.parse("trb:2"), nl=NL, ne=NE, k=K, boundaries=[10])
    assert F.parse("trb:2")["tr"] == 2.0 and F.parse("trb:2")["trb"] and not F.parse("tr:2")["trb"]
    assert marked["tstat_rounds"] == plain["tstat_rounds"] > 0
    unmarked = F.simulate(ws, init_slots(16), 16, pol, F.parse("trb:2"), nl=NL, ne=NE, k=K)
    assert unmarked["switch_s"] == plain["switch_s"]


def test_summarize_subset_uses_only_those_windows():
    h = np.arange(40, dtype=float) + 50
    base = dict(hits=h, ents=np.full(40, 100.0), dh=h, dn=np.full(40, 100.0), swaps=1.0)
    x = dict(base, hits=h + np.where(np.arange(40) < 20, 5.0, 0.0), dh=h)
    mask = np.arange(40) < 20
    out = F.summarize({"base": base, "x": x}, nboot=200, block=4, subset=mask)["x"]
    assert abs(out["diff"] - 5.0) < 1e-9 and out["n"] == 20
    assert abs(F.summarize({"base": base, "x": x}, nboot=200, block=4, subset=~mask)["x"]["diff"]) < 1e-12


def test_attach_stwt_matches_by_ids_and_reports_weights(tmp_path):
    import struct
    wins = [np.array([[0, 1]]), np.array([[2, 3]])]
    p = tmp_path / "d.bin"
    write_dump(p, wins)
    got, k, starts, weights, total = F.load_dump(str(p))
    f = tmp_path / "t.stwt"
    with open(f, "wb") as fh:
        fh.write(b"STWT" + struct.pack("<4I", 1, NL, NE, K))
        for ids in wins:
            fh.write(struct.pack("<BBBBqf", 2, 1, 255, 0, 0, 1.0))
            for l in range(NL):
                fh.write(struct.pack("<BBBB", 1, l, 1, 1))
                for e in (ids[0] + l) % NE:
                    fh.write(struct.pack("<hb", int(e), 0))
                fh.write(np.array([0.7, 0.3], "<f4").tobytes())
    w, m, nz = F.attach_stwt(got, str(f))
    assert m == 2 and nz == 2 and F.has_real_weights([x for x in w if x])
    assert abs(float(w[0][0][0, 0]) - 0.7) < 1e-6


def test_conv_trace_order_option():
    import conv_trace as C
    convs = {"A": ("code", None, ["1", "2", "3"]), "B": ("prose", None, ["1", "2"])}
    assert C.parse_order("", convs) == C.ORDER
    assert C.parse_order("aaabb", convs) == list("AAABB")
    with pytest.raises(SystemExit):
        C.parse_order("AAAA", convs)
    with pytest.raises(SystemExit):
        C.parse_order("AZ", convs)
