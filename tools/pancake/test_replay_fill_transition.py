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
