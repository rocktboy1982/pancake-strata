"""Tests for gate_power.py: the noise models and the two simulations on small trial counts (fast, seeded).  No engine, no model.
Run: python -m pytest tools/pancake/test_gate_power.py   (or python tools/pancake/test_gate_power.py)"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_power as P  # noqa: E402


def test_mix_rates_follow_the_proportions():
    r = P.mix_rates(P.MIX_PROMOTION, 18)
    assert len(r) == 18 and r.count(0.95) == 10 and r.count(0.10) == 2
    assert len(P.mix_rates(P.MIX_BALANCED, 150)) == 150 and abs(sum(1 for x in P.mix_rates(P.MIX_BALANCED, 100) if x == 0.5) - 40) <= 1


def test_the_tie_model_matches_the_real_stock_arms():
    rng = random.Random(3)
    lens = P.OLD_LENS * 8
    w = P.weights(rng, len(lens))
    ties = P.tie_mask(rng, lens, w)
    a, b = P.arm_rows(rng, lens, w, ties, "ties"), P.arm_rows(rng, lens, w, ties, "ties")
    ma = {(r, i) for r, rows in a.items() for i, x in enumerate(rows) if x[0]}
    mb = {(r, i) for r, rows in b.items() for i, x in enumerate(rows) if x[0]}
    n = sum(lens)
    assert 0.008 < len(ma) / n < 0.018 and 0.008 < len(mb) / n < 0.018        # the real arms: 1.17% and 1.30%
    assert len(ma & mb) > 0.3 * len(ma)                                       # real: 36 of 62-69 positions shared (independent arms: 1%)


def test_token_power_identical_arms_pass_and_a_real_deviation_is_rejected_more_at_48_requests():
    rng = random.Random(1)
    same = P.token_power(P.OLD_LENS, "ties", 0.0, 30, rng, boots=300)
    assert abs(sum(same.values()) - 1) < 1e-9 and same["REJECT"] == 0 and same["PASS"] > 0.6
    big = P.token_power(P.OLD_LENS + [240] * 28, "iid", 1.5, 30, rng, boots=300)
    assert big["REJECT"] > 0.8
    small = P.token_power(P.OLD_LENS, "iid", 1.0, 40, random.Random(2), boots=300)["REJECT"]
    grown = P.token_power(P.OLD_LENS + [240] * 28, "iid", 1.0, 40, random.Random(2), boots=300)["REJECT"]
    assert grown > small


def test_task_power_small_sets_cannot_decide_and_big_balanced_sets_can():
    rng = random.Random(1)
    small = P.task_power(P.MIX_PROMOTION, 18, 5, 0.0, 12, rng, boots=300)
    assert small["PASS"] <= 0.25 and small["REJECT"] == 0                        # promotion.md: 3%
    big_same = P.task_power(P.MIX_BALANCED, 100, 10, 0.0, 12, rng, boots=300)
    assert big_same["PASS"] >= 0.5 and big_same["REJECT"] == 0
    big_loss = P.task_power(P.MIX_BALANCED, 100, 10, 10.0, 12, rng, boots=300)
    assert big_loss["PASS"] == 0 and big_loss["REJECT"] > 0.3


def test_runs_are_reproducible_from_the_seed():
    a = P.task_power(P.MIX_PROMOTION, 20, 3, 5.0, 8, random.Random(9), boots=200)
    assert a == P.task_power(P.MIX_PROMOTION, 20, 3, 5.0, 8, random.Random(9), boots=200)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
