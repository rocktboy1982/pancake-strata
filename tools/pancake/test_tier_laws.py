"""pytest for tools/pancake/tier_laws.py: synthetic Layered traces with a known answer."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cache_learn as cl  # noqa: E402
import tier_laws as tl  # noqa: E402

NL, NE = 2, 8


def layered(rows):
    """rows[t] = list of routed ids of window t, the same in every layer."""
    return cl.Layered([[np.array(r) for r in rows] for _ in range(NL)], NL)


def init_pairs(per_layer):
    # rank order: the first NL pairs are every layer's best expert (the profile's order), so a protected prefix is a core in every layer
    return np.array([(l, e) for e in per_layer for l in range(NL)], dtype=np.int64)


def go(rows, per_layer, policy, n_prot=0, turns=None):
    d = tl.dense(layered(rows), NE)
    return tl.run(d, init_pairs(per_layer), policy, n_prot, turns)


def test_static_changes_nothing():
    rows = [[0, 1], [5, 6], [5, 6], [0, 5]] * 3
    r, _, pl, _ = go(rows, [0, 1, 2, 3], tl.Static())
    assert r["promo_w"] == 0.0
    # entries 0,1 hit; 5,6 miss; 0 hit, 5 miss
    assert abs(r["hits"] - (2 + 0 + 0 + 1) / 8) < 1e-12
    assert pl.sum() == 0


def test_protect_keeps_core_resident():
    rows = [[6, 7]] * 30                       # a hot outsider pair, the core 0..3 is never used
    free, *_ = go(rows, [0, 1, 2, 3], tl.Tier(mingain=0.0))
    r, _, _, hist = tl.run(tl.dense(layered(rows), NE), init_pairs([0, 1, 2, 3]), tl.Tier(mingain=0.0), 8, None, record=True)
    assert free["promo_w"] > 0 and free["hits"] > 0.5      # unprotected: the tier replaces the cold residents
    assert r["promo_w"] == 0 and r["hits"] == 0.0            # protected core: nothing may leave, nothing enters
    assert hist[-1, :, :4].all()
    # half protected: the two cold unprotected slots are replaced, the two protected ones stay
    r2, _, _, h2 = tl.run(tl.dense(layered(rows), NE), init_pairs([0, 1, 2, 3]), tl.Tier(mingain=0.0), 4, None, record=True)
    assert h2[-1, :, :2].all() and h2[-1, :, 6:8].all() and r2["hits"] > 0.5


def test_protect_one_equals_static():
    rows = [[6, 7], [6, 7], [4, 5]] * 5
    st, *_ = go(rows, [0, 1, 2, 3], tl.Static())
    for pol in (tl.Tier(), tl.KMiss(1, 1), tl.BreakEven(8, 0.25), tl.Survival(4)):
        r, *_ = go(rows, [0, 1, 2, 3], pol, n_prot=tl.protect_slots(1.0, 4 * NL))
        assert r == st


def test_protect_slots_matches_engine_rule():
    assert tl.protect_slots(0.5, 10909) == 5455 and tl.protect_slots(0.75, 10909) == 8182
    assert tl.protect_slots(0.9, 10909) == 9819 and tl.protect_slots(0.0, 10) == 0
    assert tl.protect_slots(1.0, 10) == 10 and tl.protect_slots(0.5, 100) == 50


def test_hot_expert_after_window_3_admitted_by_kmiss_not_static():
    rows = [[0, 1]] * 4 + [[5, 0]] * 8
    km, _, _, hist = tl.run(tl.dense(layered(rows), NE), init_pairs([0, 1, 2, 3]), tl.KMiss(1, 1), 0, None, record=True)
    st, _, _, hs = tl.run(tl.dense(layered(rows), NE), init_pairs([0, 1, 2, 3]), tl.Static(), 0, None, record=True)
    assert not hist[4, :, 5].any() and hist[5, :, 5].all()    # missed in window 4, resident from window 5
    assert not hs[:, :, 5].any()
    assert km["hits"] > st["hits"]


def test_identical_sets_have_jaccard_one():
    d = tl.dense(layered([[0, 1, 2]] * 4), NE)
    a = tl.top_sets(tl.turn_counts(d, 0, 2), 3)
    b = tl.top_sets(tl.turn_counts(d, 2, 4), 3)
    assert tl.jaccard(a, b) == 1.0
    c = tl.top_sets(tl.turn_counts(tl.dense(layered([[4, 5, 6]]), NE), 0, 1), 3)
    assert tl.jaccard(a, c) == 0.0


def test_validation_runs_and_conv_seed_helps_resume():
    # conversation a uses {4,5}, b uses {6,7}; a resumes after b: the seeded tier restores {4,5} faster
    rows = [[4, 5]] * 6 + [[6, 7]] * 6 + [[4, 5]] * 6
    turns = [(0, 6, "a"), (6, 12, "b"), (12, 18, "a")]
    d = tl.dense(layered(rows), NE)
    budgets = np.array([4, 4])
    v = tl.validate_conv(d, init_pairs([0, 1, 2, 3]), budgets, turns, 0.5, "synthetic test")
    assert v["V1"]["same_conv_adjacent"] == 1.0 and v["V1"]["other_conv_adjacent"] == 0.0
    assert v["V2"]["hit_lift_points"] >= 0.0
    assert v["V3"]["resume"]["1"] == 1.0 and v["V3"]["switch"]["1"] == 0.0
    assert set(v["V4"]) == {"1", "2", "4", "8"}


def test_seedfile_machinery_and_policies_run():
    rows = [[4, 5]] * 8 + [[2, 3, 4]] * 4
    seed = np.zeros((NL, NE))
    seed[:, 4] = seed[:, 5] = 5.0
    r, *_ = go(rows, [0, 1, 2, 3], tl.Tier(seed=seed, seed_w=1.0))
    plain, *_ = go(rows, [0, 1, 2, 3], tl.Tier())
    assert r["hits"] >= plain["hits"]
    ranked = np.array([(l, e) for l in range(NL) for e in range(NE)])
    for pol in (tl.BreakEven(4, 0.25), tl.Bayes(8, 4, 1.0, 0.6, 0.9, ranked), tl.Survival(4), tl.Tier(conv_w=1.0)):
        rr, *_ = go(rows, [0, 1, 2, 3], pol)
        assert 0.0 <= rr["hits"] <= 1.0


def test_thrash_metric_and_gap_misses_known_answer():
    rows = [[2], [0], [2], [2]]
    r, *_ = go(rows, [0, 1], tl.KMiss(1, 1))
    th, g = r["thrash"], r["gap"]
    assert th["evictions"] == 4 and th["promotions"] == 4
    assert abs(th["swapback1"] - 0.5) < 1e-12 and abs(th["swapback16"] - 0.5) < 1e-12
    assert abs(th["net_us_mean"] - ((2 * 48.7 - 58.5) + (0 * 48.7 - 58.5)) / 2) < 1e-9
    assert abs(th["neg_share"] - 0.5) < 1e-12
    assert abs(g["gap_per_w"] - 0.5) < 1e-12 and abs(g["saved_per_w"]["16"] - 0.5) < 1e-12


def test_ctx_sequence_and_slice_turns():
    turns = [(0, 2, "a"), (2, 4, "b"), (4, 6, "b")]
    ctx = tl.ctx_sequence(6, turns)
    assert ctx[1] == "switch" and ctx[3] == "continue" and ctx[0] == "continue"
    assert tl.slice_turns(turns, 2, 6) == [(0, 2, "b"), (2, 4, "b")]


def test_ghost_reports_protect_fraction_and_runs():
    rows = [[2], [0], [2], [3]] * 8
    r, *_ = go(rows, [0, 1], tl.Ghost(delta=0.5, quiet=2, k=1, W=1))
    pf = r["protect_frac"]
    assert 0.0 <= pf["min"] <= pf["mean"] <= pf["max"] <= 1.0
    assert r["thrash"]["evictions"] > 0


def test_phase_markov_learns_alternating_phases():
    rows = [[0, 1], [4, 5]] * 10
    d = tl.dense(layered(rows), NE)
    p = tl.Phase(2, H=4).fit(d[:10], None)
    r = tl.run(d[10:], init_pairs([0, 1, 2, 3]), p, 0, None)[0]
    acc = r["phase_acc"]["continue"]
    assert acc["n"] > 0 and acc["top1"] == 1.0


def test_lecar_weights_and_rules_run():
    rows = ([[2, 3]] * 6 + [[4, 5]] * 6 + [[2, 3]] * 6 + [[0, 5]] * 6) * 2
    d = tl.dense(layered(rows), NE)
    pol = tl.LeCaR(lr=2.0, H=4, K=2).fit(d[:24], None)
    r = tl.run(d[24:], init_pairs([0, 1, 2, 3]), pol, 0, None)[0]
    lc = r["lecar"]
    assert abs(sum(lc["mean_weight"]) - 1.0) < 0.02 and abs(sum(lc["leader_share"]) - 1.0) < 0.01
    assert 0.0 <= r["hits"] <= 1.0
    for rule in (tl.KMissRule(1, 2), tl.GhostRule(k=1)):
        rr = go(rows, [0, 1, 2, 3], rule)[0]
        assert rr["thrash"]["promotions"] >= 1 and rr["hits"] > go(rows, [0, 1, 2, 3], tl.Static())[0]["hits"]


def final_residents(rows, rule, per_layer=(0, 1, 2, 3)):
    d = tl.dense(layered(rows), NE)
    _, _, _, hist = tl.run(d, init_pairs(list(per_layer)), rule, 0, None, record=True)
    # hist[t] is the residency at the START of window t; add one empty window so the last decision shows
    d2 = tl.dense(layered(rows + [[]]), NE)
    _, _, _, hist = tl.run(d2, init_pairs(list(per_layer)), rule, 0, None, record=True)
    return set(np.flatnonzero(hist[-1][0]).tolist())


def test_srlru_evicts_the_unrepeated_newcomer_before_the_scan_resistant_residents():
    rows = [[6], [6], [7], [7], [7]]
    # plain LRU-ordered rule: 6 was admitted after window 1 (its last = 1), the initial residents are older, so 1 goes for 7
    assert final_residents(rows, tl.KMissRule(2, 8)) == {6, 7, 2, 3}
    # SR-LRU: 6 sits in R (never routed again while resident), the profile residents are SR: 6 leaves for 7
    assert final_residents(rows, tl.SRLruRule(2, 8)) == {7, 1, 2, 3}


def test_crlfu_breaks_frequency_ties_toward_the_most_recent():
    rows = [[0], [0], [0], [1], [1], [6], [6]]
    # LRU order: expert 2 is the oldest idle resident; CR-LFU: 2 and 3 both have 0 uses, the tie goes to the MRU, expert 3
    assert 2 in final_residents(rows, tl.CRLfuRule(2, 8)) and 3 not in final_residents(rows, tl.CRLfuRule(2, 8))
    assert 3 in final_residents(rows, tl.KMissRule(2, 8)) and 2 not in final_residents(rows, tl.KMissRule(2, 8))


def test_cacheus_runs_adapts_its_rate_and_honours_the_cap():
    rows = ([[2, 3]] * 6 + [[4, 5]] * 6 + [[2, 3]] * 6 + [[0, 5]] * 6) * 3
    d = tl.dense(layered(rows), NE)
    pol = tl.Cacheus(lr=0.5, H=4)
    pol.cap = 1
    r, _, promo, _ = tl.run(d, init_pairs([0, 1, 2, 3]), pol, 0, None)
    assert promo.sum() > 0 and r["promo_w"] <= NL * 1.0 + 1e-9        # cap 1 swap a window, a window has swaps in both layers
    cc = r["cacheus"]
    assert 0.001 <= cc["lr_min"] <= cc["lr_final"] <= cc["lr_max"] <= 1.0 and cc["lr_max"] != cc["lr_min"]
    assert abs(sum(r["lecar"]["mean_weight"]) - 1.0) < 0.02 and r["lecar"]["rules"] == ["srlru", "crlfu"]
    assert r["hits"] > go(rows, [0, 1, 2, 3], tl.Static())[0]["hits"]
    assert tl.parse_policy("cacheus:0.2,4", {}).name == "cacheus lr0.2 H4"


def test_kmiss_with_cap_keeps_its_lists_consistent_with_the_cache():
    # 4 hot experts are missing at once and the cap lets 2 swaps through a window: before the 2026-11-16 fix the policy's own
    # LRU list assumed every proposal was applied and run() tripped its residency assertion
    rows = [[0, 1]] * 3 + [[2, 3, 4, 5]] * 10
    d = tl.dense(layered(rows), NE)
    pol = tl.KMiss(1, 1)
    pol.cap = 1                                              # per layer 1 swap would be 2 swaps in a window of 2 layers
    r, _, promo, hist = tl.run(d, init_pairs([0, 1, 6, 7]), pol, 0, None, record=True)
    assert hist[5, :, 2:6].sum() <= 2                        # one swap a window: windows 3 and 4 admitted at most 2 by window 5
    assert hist[-1, :, 2:6].all()                            # but all four hot experts arrive over the windows
