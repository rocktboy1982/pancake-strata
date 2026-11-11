"""block_reading_sim.py: the three readings of a --block run and the decision they back (promote.py reads a start as one pair).
No engine, no GPU; about a second of CPU.

Run: python -m pytest tools/pancake/test_block_reading_sim.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import block_reading_sim as B  # noqa: E402
import promote  # noqa: E402

QUIET = {"case": "x", "sd_between_pct": 0.2, "sd_within_pct": 0.2, "outlier_prob": 0.0, "outlier_factors": [0.8]}
EDIT_LIKE = {"case": "edit", "sd_between_pct": 1.9, "sd_within_pct": 0.8, "outlier_prob": 0.14, "outlier_factors": [0.89]}   # most of a pass's variance is the start's


def test_draw_gives_starts_by_passes_and_drops_outlier_passes_as_nan():
    x = B.draw(np.random.default_rng(1), EDIT_LIKE, 500, 3, 4, 0.0)
    assert x.shape == (500, 3, 4) and np.isnan(x).any() and np.isfinite(x).mean() > 0.9


def test_a_big_gain_in_quiet_data_promotes_but_two_starts_only_in_the_reading_that_counts_passes():
    rates = B.promote_rates(np.random.default_rng(3), QUIET, 2, 3, 10.0, 1.0, 400)
    assert rates == {"start": 0.0, "pass": 1.0, "pass, t(k-1)": 0.0}                      # promote.py and the cluster reading need MIN_PAIRS starts
    assert set(B.promote_rates(np.random.default_rng(3), QUIET, promote.MIN_PAIRS, 3, 10.0, 1.0, 400).values()) == {1.0}


def test_the_start_reading_keeps_false_promote_near_nominal_where_the_pass_reading_does_not():
    rates = B.promote_rates(np.random.default_rng(3), EDIT_LIKE, 3, 4, 0.0, 1.0, 6000)     # a true 0%: any PROMOTE is false
    assert rates["start"] < 0.02 and rates["pass"] > 0.05 and rates["start"] <= rates["pass, t(k-1)"] < rates["pass"], rates
