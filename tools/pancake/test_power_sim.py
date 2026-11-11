"""power_sim.py: the pure functions (noise fit, promote.py's rule on arrays, outlier rule, designs, minimum n) and one small report run.
No engine, no GPU; a few seconds of CPU.

Run: python -m pytest tools/pancake/test_power_sim.py"""
import contextlib
import io
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import power_sim as P  # noqa: E402
import promote  # noqa: E402

REAL = P.ROOT / "bench/results/2026-11-11-gpu"
FIT = {"case": "short", "sd_between_pct": 1.5, "sd_within_pct": 1.0, "outlier_prob": 0.0, "outlier_factors": [0.8]}


def fit(**kw):
    return {**FIT, **kw}


def synth(rng, groups=30, per=6, sd_b=0.02, sd_w=0.01, p_out=0.0, out_f=0.8, case="short"):
    """abtest-like attempt records: `groups` arms of `per` starts, two requests per start (start effect, slow starts, request noise)."""
    recs = []
    for g in range(groups):
        level = 100 * rng.uniform(0.9, 1.1)
        for r in range(per):
            start = level * (1 + rng.gauss(0, sd_b)) * (out_f if rng.random() < p_out else 1)
            rows = [{"case": case, "decode_tps": start * (1 + rng.gauss(0, sd_w))} for _ in range(2)]
            recs.append({"src": f"f{g}", "arm": "base", "rep": r, "attempt": 0, "clean": True, "noise": 1.0, "reclaim": 0.0, "rows": rows})
    return recs


# ---------------------------------------------------------------------------------------------------- fit
def test_pairwise_sd_recovers_a_normal_and_shrugs_off_outliers():
    rng = random.Random(3)
    groups = [[100 * (1 + rng.gauss(0, 0.02)) * (0.6 if rng.random() < 0.1 else 1) for _ in range(5)] for _ in range(400)]
    assert abs(P.pairwise_sd(groups) - 0.02) < 0.008      # 10% slow starts bias it up a little; the plain sd would be > 0.1
    assert np.std([x for g in groups for x in g] / np.mean([x for g in groups for x in g])) > 0.08
    clean = [[100 * (1 + rng.gauss(0, 0.03)) for _ in range(3)] for _ in range(400)]
    assert abs(P.pairwise_sd(clean) - 0.03) < 0.004          # a group of 3 is not biased low (median centring would be)
    assert abs(P.pairwise_sd([[0.1 * rng.gauss(0, 1) for _ in range(4)] for _ in range(400)], relative=False) - 0.1) < 0.015


def test_fit_case_recovers_the_noise_of_synthetic_attempts():
    f = P.fit_case(synth(random.Random(1), sd_b=0.02, sd_w=0.01, p_out=0.1), "short")
    assert abs(f["sd_total_pct"] - 2.24) < 0.7 and abs(f["sd_within_pct"] - 1.0) < 0.5
    assert abs(f["sd_between_pct"] - 2.0) < 0.8 and 0.04 < f["outlier_prob"] < 0.2
    assert 0.75 < sum(f["outlier_factors"]) / len(f["outlier_factors"]) < 0.85
    quiet = P.fit_case(synth(random.Random(2), p_out=0.0), "short")
    assert quiet["outlier_prob"] < 0.03 and quiet["outlier_factors"] == [0.8]    # no slow start: the placeholder factor is never drawn


def test_fit_case_gates_pick_the_starts_and_too_little_data_raises():
    recs = synth(random.Random(4))
    for r in recs[::2]:
        r["clean"], r["noise"] = False, 80.0          # every second start was discarded by the gates
    assert P.fit_case(recs, "short", "recorded")["starts"] == len(recs) // 2
    assert P.fit_case(recs, "short", "none")["starts"] == len(recs)
    assert P.discard_rates(recs) == {"recorded": 0.5, "relaxed": 0.5, "none": 0.0}
    with pytest.raises(ValueError):
        P.fit_case(synth(random.Random(5), groups=1, per=4), "short")


@pytest.mark.skipif(not REAL.is_dir(), reason="real results not in this checkout")
def test_fit_on_the_real_results_is_sane():
    att = P.read_attempts(sorted(REAL.glob("*.jsonl")))
    for case in ("short", "edit"):
        for gates in P.GATES:
            f = P.fit_case(att, case, gates)
            assert 0.5 < f["sd_total_pct"] < 6 and f["sd_between_pct"] ** 2 + f["sd_within_pct"] ** 2 == pytest.approx(f["sd_total_pct"] ** 2)
            assert 0 <= f["outlier_prob"] < 0.4 and all(0.3 < x < 0.97 for x in f["outlier_factors"])
    assert P.discard_rates(att)["recorded"] > 0.3 > P.discard_rates(att)["relaxed"]      # 48% as run; most were kswapd above 1%


# ---------------------------------------------------------------------------------------------------- promote.py's rule
def test_verdicts_follow_promote_gain_ci():
    rng = random.Random(7)
    for _ in range(300):
        n, need = rng.randint(3, 8), rng.choice([1.0, 3.0])
        base = [100 * (1 + rng.gauss(0, 0.02)) for _ in range(n)]
        arm = [b * (1 + rng.uniform(-0.01, 0.07) + rng.gauss(0, 0.02)) for b in base]
        g = promote.gain_ci(list(zip(base, arm)))
        want = 0 if g["lo"] >= need else 2 if g["hi"] < need else 1
        assert P.verdicts(np.array([base]), np.array([arm]), need)[0] == want


def test_verdicts_alpha_and_pairs():
    rng = random.Random(8)
    base = [100 * (1 + rng.gauss(0, 0.02)) for _ in range(5)]
    arm = [b * 1.06 * (1 + rng.gauss(0, 0.02)) for b in base]
    b, a = np.array([base]), np.array([arm])
    g, g_bonf = promote.gain_ci(list(zip(base, arm))), promote.gain_ci(list(zip(base, arm)), alpha=0.05 / 4)
    assert g_bonf["lo"] < g["lo"]
    need = (g["lo"] + g_bonf["lo"]) / 2                 # reached at alpha 0.05, not at 0.05 / 4
    assert P.verdicts(b, a, need)[0] == 0 and P.verdicts(b, a, need, alpha=0.05 / 4)[0] == 1
    assert P.verdicts(b[:, :2], a[:, :2], 1.0)[0] == 1                       # 2 pairs: never PROMOTE (MIN_PAIRS 3), never REJECT
    nan = np.array([[100, 101, np.nan, 99, 100.5]])                          # a missing run drops its pair
    assert P.verdicts(nan, nan * 1.1, 1.0)[0] == 0


def test_thresholds_come_from_promote():
    assert P.NEED_PCT == {"exact": promote.MIN_GAIN[True], "lossy": promote.MIN_GAIN[False]}


# ---------------------------------------------------------------------------------------------------- the designs
def test_no_gain_never_promotes_and_power_grows_with_gain_and_pairs():
    rng = np.random.default_rng(1)
    assert P.power(rng, FIT, "old", 6, 0, 1.0, 4000) < 0.01
    p = [P.power(rng, FIT, "old", 10, g, 1.0, 4000) for g in (2, 3, 5, 10)]
    assert p == sorted(p) and p[0] < 0.9 and p[-1] > 0.99
    assert P.power(rng, FIT, "old", 3, 5, 1.0, 4000) < P.power(rng, FIT, "old", 6, 5, 1.0, 4000) < P.power(rng, FIT, "old", 20, 5, 1.0, 4000) + 1e-9
    assert P.power(rng, FIT, "old", 40, 2, 3.0, 4000) < 0.01       # a lossy item (need 3%) with a true 2% gain cannot promote
    at_need = P.power(rng, fit(sd_between_pct=0.3, sd_within_pct=0.3), "old", 40, 3.0, 3.0, 4000)
    assert 0.01 < at_need < 0.045            # gain = need: the 95% lower bound clears it alpha/2 = 2.5% of the time (no coin flip)


def test_two_starts_per_arm_can_never_promote():
    rng = np.random.default_rng(6)
    assert P.power(rng, fit(sd_between_pct=0.1, sd_within_pct=0.1), "block", 2, 10, 1.0, 2000, passes=8) == 0   # promote.py needs 3 pairs
    assert P.power(rng, fit(sd_between_pct=0.1, sd_within_pct=0.1), "block", 3, 10, 1.0, 2000, passes=8) > 0.99


def test_block_mode_beats_old_mode_only_when_the_noise_is_inside_the_start():
    rng = np.random.default_rng(2)
    inside = fit(sd_between_pct=0.5, sd_within_pct=4.0)
    assert P.power(rng, inside, "block", 4, 5, 1.0, 5000, passes=8) > P.power(rng, inside, "old", 4, 5, 1.0, 5000) + 0.3
    between = fit(sd_between_pct=3.0, sd_within_pct=0.0)
    old, block = P.power(rng, between, "old", 4, 5, 1.0, 5000), P.power(rng, between, "block", 4, 5, 1.0, 5000, passes=8)
    assert abs(old - block) < 0.04                       # passes inside a start cannot average out the start-to-start variance


def test_reading_every_pass_as_a_pair_promotes_far_above_alpha_and_the_design_effect_only_part_of_the_way():
    rng = np.random.default_rng(3)
    f = fit(sd_between_pct=2.0, sd_within_pct=0.5)       # the passes of a start share 94% of their variance
    honest, deff, naive = (P.power(rng, f, "block", 4, 0, 1.0, 6000, passes=8, analysis=a) for a in ("start", "pass", "naive"))
    assert honest < 0.02 and naive > 0.06                # 32 'pairs' from 4 starts: the interval is far too narrow (nominal: 2.5%)
    assert honest <= deff < naive                        # abtest's ICC 0.3 widening is too little for such a start effect


def test_outlier_flags_follow_abtest_s_rule():
    x = np.array([[100.0, 101, 99, 60], [100, 101, 99, 98.5], [100, 101, 99, 93]])
    assert P.outlier_flags(x).tolist() == [[False, False, False, True], [False] * 4, [False] * 4]   # 93 is 7% off: under the 8.75% floor
    assert not P.outlier_flags(np.array([[100.0, 50.0]])).any()                  # fewer than 3 values: no centre
    tight = np.array([[100.0, 100.1, 99.9, 100.0, 90.0]])                          # MAD ~ 0: the 2.5% floor sets the spread
    assert P.outlier_flags(tight).tolist() == [[False] * 4 + [True]] and not P.outlier_flags(tight, z=5).any()


def test_the_outlier_rule_pulls_slow_starts_out_and_costs_pairs():
    rng = np.random.default_rng(4)
    f = fit(outlier_prob=0.3, outlier_factors=[0.75])
    plain = P.power(rng, f, "old", 8, 5, 1.0, 4000)
    ruled = P.power(rng, f, "old", 8, 5, 1.0, 4000, robust=True)
    assert ruled > plain + 0.15
    assert P.power(rng, FIT, "old", 6, 0, 1.0, 4000, robust=True) < 0.01          # and it does not invent a gain
    x = P.draw_passes(rng, f, 1, 6, 4, 0.0)
    assert x.shape == (1, 6, 4) and np.allclose(x[0].mean() / 100, 1, atol=0.3)


def test_slow_start_rule_reruns_the_slower_of_two_and_costs_starts():
    rng = np.random.default_rng(8)
    f = fit(sd_between_pct=0.5, sd_within_pct=0.5, outlier_prob=0.3, outlier_factors=[0.7])
    x = P.draw_passes(rng, f, 4000, 2, 3, 0.0)
    after, reruns = P.rerun_slow_starts(rng, f, x, 0.0)
    ref = x.mean(axis=2).max(axis=1)
    slow_before = (x.mean(axis=2) < ref[:, None] * (1 - P.SLOW_START)).any(axis=1)
    assert slow_before.mean() > 0.3 and slow_before.mean() <= reruns.mean() < 1.5 * slow_before.mean()   # a rerun per slow start, a second one when the rerun is slow again
    assert (after.mean(axis=2).min(axis=1) < after.mean(axis=2).max(axis=1) * (1 - P.SLOW_START)).mean() < 0.5 * slow_before.mean()
    one, none = P.rerun_slow_starts(rng, f, x[:, :1], 0.0)                                        # a single start has no reference
    assert (one == x[:, :1]).all() and none.sum() == 0
    v, starts = P.simulate(rng, f, "old", 2, 5, 1.0, 3000, robust=True)
    assert starts > 2 and P.simulate(rng, f, "old", 2, 5, 1.0, 3000)[1] == 2                      # reruns are counted, a plain run has none
    assert P.minutes("old", 3, reruns=1) == pytest.approx(2 * 4 * (P.START_S + P.PASS_S) / 60)


def test_verdicts_are_hold_when_every_pair_is_dropped():
    nan = np.full((1, 4), np.nan)
    assert P.verdicts(nan, nan, 1.0)[0] == 1


def test_min_units():
    rng = np.random.default_rng(5)
    assert P.min_units(rng, FIT, "old", 1.0, 1.0, 1000) is None and P.min_units(rng, FIT, "old", 2.0, 3.0, 1000) is None
    quiet = P.min_units(rng, fit(sd_between_pct=0.3, sd_within_pct=0.3), "old", 5.0, 1.0, 2000)
    noisy = P.min_units(rng, fit(sd_between_pct=3.0, sd_within_pct=2.0), "old", 5.0, 1.0, 2000)
    assert quiet == promote.MIN_PAIRS and noisy is not None and noisy > quiet
    assert P.min_units(rng, fit(sd_between_pct=8.0, sd_within_pct=8.0), "old", 2.0, 1.0, 500, cap=10) is None   # not within the cap
    k = P.min_units(rng, fit(sd_between_pct=1.0, sd_within_pct=1.0), "block", 5.0, 1.0, 2000, passes=4)
    assert k is not None and k >= promote.MIN_PAIRS


def test_override_resplits_the_variance_and_keeps_the_total():
    f = {"sd_total_pct": 2.0, "sd_between_pct": 0.0, "sd_within_pct": 2.0, "outlier_prob": 0.1, "outlier_factors": [0.8]}
    g = P.override(f, icc=0.25)
    assert g["sd_between_pct"] == pytest.approx(1.0) and g["sd_within_pct"] == pytest.approx(math.sqrt(3)) and f["sd_between_pct"] == 0.0
    assert math.hypot(g["sd_between_pct"], g["sd_within_pct"]) == pytest.approx(2.0)
    h = P.override(f, between=0.5, within=1.5, outlier_prob=0.0, icc=0.9)
    assert (h["sd_between_pct"], h["sd_within_pct"], h["outlier_prob"]) == (0.5, 1.5, 0.0)      # explicit values win over the split
    assert P.override(f) == f and P.parse(["--icc", "0.3"]).icc == 0.3


def test_shares_minutes_and_labels():
    assert P.shares(np.array([0, 0, 1, 2])) == (0.5, 0.25, 0.25)
    assert P.minutes("old", 3) == pytest.approx(2 * 3 * (P.START_S + P.PASS_S) / 60)
    assert P.minutes("block", 3, 4) == pytest.approx(2 * 3 * (P.START_S + 4 * P.PASS_S) / 60)
    assert P.minutes("old", 3, discard=0.5) == pytest.approx(2 * 3 * (P.START_S + P.PASS_S) / 60 / 0.5)
    assert P.label("old", 6, 1) == "n6" and P.label("block", 3, 4, "start") == "k3x4"
    assert P.label("block", 3, 4, "pass") == "k3x4+p" and P.label("block", 3, 4, "naive") == "k3x4+n"


def test_report_runs_and_prints_every_table(tmp_path):
    recs = synth(random.Random(9), case="short") + synth(random.Random(10), case="edit")
    path = tmp_path / "r.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    out = io.StringIO()
    args = P.parse(["--results", str(path), "--trials", "300", "--gains", "0,5", "--old", "3,6", "--starts", "2,3", "--seed", "2"])
    with contextlib.redirect_stdout(out):
        assert P.report(args) == 0
    text = out.getvalue()
    assert text.count("=== case") == 4 and text.count("P(PROMOTE) %") == 12
    assert "minimum for 80% power, short" in text and "minimum for 80% power, edit" in text
    assert "k2x4 " in text and "k3x4+p" in text and "k3x4+n" in text and "n6" in text
