"""Unit tests for selftune.py (item 70) and chain_model.py (the A6 model as one function): the held-out folds, the candidate
ranking, the decision gate, the A/B spec, decode-window filtering and the command line on small synthetic traces.
No GPU, no engine, no real trace.
Run: python -m pytest tools/pancake/test_selftune.py   (or python tools/pancake/test_selftune.py)"""
import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chain_model as cm  # noqa: E402
import make_profile as mp  # noqa: E402
import replay_cache_policy as rp  # noqa: E402
import selftune as st  # noqa: E402

NL, NE, K = rp.NL, rp.NE, 10


def hot_sets(seed, n_hot=24):
    rng = np.random.default_rng(seed)
    return [rng.choice(NE, n_hot, replace=False) for _ in range(NL)]


def session(n_win, hot, seed, rows=4):
    """n_win decode windows [(layer, ids[rows, K])]: every layer routes inside its hot set (distinct experts per row)."""
    rng = np.random.default_rng(seed)
    return [[(l, np.stack([rng.choice(hot[l], K, replace=False) for _ in range(rows)]).astype(np.int64)) for l in range(NL)]
            for _ in range(n_win)]


def ranking(hot):
    """A complete profile ranking: the hot pairs first (interleaved over layers), then every other pair."""
    first = [(l, int(e)) for i in range(len(hot[0])) for l, h in enumerate(hot) for e in [h[i]]]
    return mp.rank_profile(first, {})[0]


def dump_routing(path, windows):
    """A --dump-routing file: one record per layer group (layer, k, ids of every row ... as the engine writes per token)."""
    with open(path, "wb") as f:
        for w in windows:
            for l, ids in w:
                for row in ids:
                    f.write(struct.pack("<ii", l, K) + struct.pack(f"<{K}i", *row.tolist()) + struct.pack(f"<{K}f", *([0.1] * K)))


def test_candidate_ranking_ranks_every_pair():
    hot = hot_sets(1)
    wins = session(5, hot, 2)
    short_current = [(0, 0), (0, 1), (1, 7)]   # an old profile that ranks almost nothing
    r = st.candidate_ranking(wins, short_current, 0.0)
    assert len(r) == NL * NE and len(set(r)) == NL * NE
    seen = {(l, int(e)) for w in wins for l, ids in w for e in ids.ravel()}
    assert set(r[:len(seen)]) == seen   # blend 0: the pairs the trace used come first


def test_candidate_ranking_blend_keeps_the_prior():
    hot_a, hot_b = hot_sets(1), hot_sets(2)
    cur = ranking(hot_b)
    wins = session(3, hot_a, 3)
    top = lambda blend: set(st.candidate_ranking(wins, cur, blend)[:200])   # noqa: E731
    pure, prior_heavy = top(0.0), top(0.95)
    overlap_cur = lambda s: len(s & set(cur[:200]))   # noqa: E731
    assert overlap_cur(prior_heavy) > overlap_cur(pure)


def test_make_folds_leave_one_session_out():
    w = lambda n: [object() for _ in range(n)]   # noqa: E731
    s = [("a", w(30)), ("b", w(25)), ("c", w(10))]
    folds = st.make_folds(s, k=4, min_fold=20)
    assert [f[0] for f in folds] == ["a", "b"]   # the 10-window session is never judged on
    for name, learn, judge in folds:
        assert not {id(x) for x in learn} & {id(x) for x in judge}
    assert len(folds[0][1]) == 25 + 10 and len(folds[1][1]) == 30 + 10


def test_make_folds_single_session_blocks():
    wins = [object() for _ in range(100)]
    folds = st.make_folds([("one", wins)], k=4, min_fold=20)
    assert len(folds) == 4
    judged = [id(x) for _, _, j in folds for x in j]
    assert sorted(judged) == sorted(id(x) for x in wins)   # disjoint blocks that cover the session
    for _, learn, judge in folds:
        assert len(learn) == 75 and not {id(x) for x in learn} & {id(x) for x in judge}


def test_gain_when_the_current_profile_is_wrong_and_none_when_right():
    slots = NL * 24
    hot_a, hot_b = hot_sets(10), hot_sets(11)
    learn, judge = session(40, hot_a, 4), session(30, hot_a, 5)
    wrong = ranking(hot_b)
    gains, hits = st.fold_gains(learn, judge, wrong, np.array(wrong[:slots]), slots, 0.3, "defer", True, None)
    assert gains[st.BASELINE] == 0.0
    assert gains["candidate tier"] > 5.0 and hits["candidate tier"] > hits[st.BASELINE] + 0.1
    right = ranking(hot_a)
    gains, _ = st.fold_gains(learn, judge, right, np.array(right[:slots]), slots, 0.3, "defer", True, None)
    assert abs(gains["candidate tier"]) < 3.0   # nothing to learn: the candidate is the current ranking


def test_verdict_needs_mean_and_enough_folds():
    v = st.verdict({"steady": [3, 2.5, 4, 3], "one_big_win": [12, -3, -1, -2], "tiny": [0.5, 1, 0.8, 0.9]}, 2.0, 0.75)
    assert v["steady"]["ok"] and v["steady"]["positive"] == 1.0
    assert not v["one_big_win"]["ok"] and v["one_big_win"]["mean"] == 1.5   # the mean is above 0 but 3 of 4 folds lose
    assert not v["tiny"]["ok"]   # every fold gains but the mean is below the gate
    three_of_four = st.verdict({"p": [8, 4, 2, -1]}, 2.0, 0.75)["p"]
    assert three_of_four["ok"] and three_of_four["min"] == -1


def test_spec_arms():
    assert st.make_spec(None, "x.bin", "b.json")["arms"] == {"current": {}}
    kmiss = st.make_spec("current kmiss 2/8", "x.bin", "b.json")["arms"]["selftune"]
    assert kmiss == {"env": {"STRATA_ADAPT_KMISS": "2,8"}}   # blocking tier: no --adapt-async (measured and rejected)
    both = st.make_spec("candidate kmiss 2/8", "x.bin", "b.json")
    assert both["arms"]["selftune"]["env"] == {"STRATA_ADAPT_KMISS": "2,8"}
    assert Path(both["arms"]["selftune"]["args"]["--expert-profile"]).is_absolute()
    prof = st.make_spec("candidate tier", "x.bin", "b.json")["arms"]["selftune"]
    assert "env" not in prof and list(prof["args"]) == ["--expert-profile"]
    assert both["lossy"] is True and both["cases"] == ["short", "edit"]


def test_preset_mode_reads_the_base_config():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cfg.json"
        p.write_text(json.dumps({"env": {"STRATA_VERIFY_DEFER_CPU": "1", "STRATA_POOL_FUSE_PHASES": "1"}}))
        assert st.preset_mode(str(p))[:2] == ("defer", True)
        p.write_text(json.dumps({"env": {}}))
        assert st.preset_mode(str(p))[:2] == ("serial", False)
    assert st.preset_mode("no-such-config-anywhere.json") == ("defer", True, None)
    assert st.preset_mode("pancake-flashnext-iq2_xs-fast-hip.json")[:2] == ("defer", True)   # the repo's own preset


def test_only_decode_windows_are_read():
    hot = hot_sets(3)
    wins = session(3, hot, 6)
    prompt = [(l, np.tile(wins[0][l][1][:1], (300, 1))) for l in range(NL)]   # a 300-row prompt window
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.bin"
        dump_routing(p, wins + [prompt])
        got = st.read_session(str(p), True)
    assert len(got) == 3


def test_main_end_to_end_and_the_gates():
    slots = NL * 24
    hot_a, hot_b = hot_sets(20), hot_sets(21)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for i in (1, 2):
            dump_routing(d / f"s{i}.bin", session(55, hot_a, 30 + i))
        mp.write_profile(d / "cur.bin", ranking(hot_b))
        args = ["--routing", str(d / "s1.bin"), str(d / "s2.bin"), "--profile", str(d / "cur.bin"), "--slots", str(slots),
                "--out", str(d / "out"), "--min-windows", "100"]
        assert st.main(args) == 0
        rep = json.loads((d / "out/report.json").read_text())
        assert [f["fold"] for f in rep["folds"]] == ["s1.bin", "s2.bin"] and rep["best"] == "candidate tier"
        spec = json.loads((d / "out/abtest-spec.json").read_text())
        assert spec["arms"]["selftune"]["args"]["--expert-profile"].endswith("candidate-profile.bin")
        cand = mp.read_profile(d / "out/candidate-profile.bin")
        assert len(cand) == NL * NE
        assert st.main(args[:-1] + ["400"]) == 2   # too few windows: nothing is judged, nothing written for the A/B


def test_chain_model_prices_misses_and_promotions():
    n = 20
    ml = np.full((n, NL), 6)
    run = lambda ml_, promo: (0.8, ml_.sum(1), np.full(n, promo), ml_)   # noqa: E731
    base = cm.window_ms(run(ml, 0))
    assert cm.window_ms(run(ml + 2, 0)) > base                  # more CPU misses: slower
    assert cm.window_ms(run(ml, 40)) > base                     # promotions steal CPU time (alpha 0.5): slower at equal misses
    assert cm.window_ms(run(ml, 0), "serial") > base            # deferral overlaps the CPU with the next layer
    assert cm.window_ms(run(ml, 0), fuse_phases=False) > base   # FUSE_PHASES halves the pool barriers
    assert cm.gain_pct(40.0, 36.0) > 0 > cm.gain_pct(36.0, 40.0)


def test_promotion_traffic_is_priced_unlike_the_2026_11_09_dry_run():
    """The first selftune priced k-th miss with tc.window_ms (serial, promotions free): +23% modeled against +2.0% +-1.6
    measured.  Same 28% fewer CPU misses, 3x the promotions (26.8 -> 81.6 per window, the measured pair): the chain model
    keeps a small gain, because promotion DMA takes DDR4 bandwidth from the CPU pool (alpha 0.5)."""
    import time_ceiling as tc
    n = 20
    tier_ml, kmiss_ml = np.full((n, NL), 7.0), np.full((n, NL), 5.0)
    tier, kmiss = (0.78, tier_ml.sum(1), np.full(n, 26.8), tier_ml), (0.86, kmiss_ml.sum(1), np.full(n, 81.6), kmiss_ml)
    new = cm.gain_pct(cm.window_ms(tier, "serial"), cm.window_ms(kmiss, "serial"))
    old_ms = lambda ml: float(np.mean([tc.window_ms(r, 0.0486, 11.5 / NL, 4.3 / NL, 0.04, 2.5, "serial", 0.5) for r in ml]))   # noqa: E731
    old = cm.gain_pct(old_ms(tier_ml), old_ms(kmiss_ml))
    assert old > 12 and 0 < new < old / 4


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
