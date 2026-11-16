"""Tests for head_cluster.py on synthetic data (no model)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import head_cluster as hc  # noqa: E402


def blobs(v=600, d=16, k=12, noise=0.15, seed=1):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(k, d)) * 2
    lab = rng.integers(0, k, v)
    return (centres[lab] + rng.normal(size=(v, d)) * noise).astype(np.float32), lab


def test_cluster_is_deterministic_and_pure_on_separated_blobs():
    rows, lab = blobs()
    a1, c1 = hc.cluster(rows, 12, seed=3)
    a2, c2 = hc.cluster(rows, 12, seed=3)
    assert (a1 == a2).all() and np.allclose(c1, c2)
    pur = sum(np.bincount(lab[a1 == c]).max() for c in range(12) if (a1 == c).any()) / len(lab)
    assert pur > 0.85


def test_table_round_trip(tmp_path):
    rows, _ = blobs(v=100, d=8, k=5)
    a, c = hc.cluster(rows, 5)
    hc.save_table(tmp_path / "t.shl", a, c)
    a2, c2 = hc.load_table(tmp_path / "t.shl")
    assert (a == a2).all() and np.allclose(c, c2)


def test_full_fraction_equals_full_argmax_and_small_fraction_scores_fewer_rows():
    rows, _ = blobs()
    a, c = hc.cluster(rows, 12)
    rng = np.random.default_rng(7)
    h = (rows[rng.integers(0, len(rows), 80)] + rng.normal(size=(80, 16)) * 0.1).astype(np.float32)
    same, share = hc.agreement(rows, h, a, c, 1.0)
    assert same == 1.0 and share > 1.0                 # every row scored, plus the centres
    same, share = hc.agreement(rows, h, a, c, 0.25)
    assert same > 0.8 and share < 0.5                  # 3 of 12 clusters


def test_a_low_scored_centre_loses_the_true_top_token():
    # row 3 is the true best for h, but its cluster's centre (mean of rows 3..5) scores below cluster 0's
    rows = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, -1], [0, -1]], np.float32)
    assign = np.array([0, 0, 0, 1, 1, 1], np.int32)
    centres = np.array([[1, 0], [0, -1 / 3]], np.float32)
    h = np.array([-0.2, 1.0], np.float32)
    assert int((rows @ h).argmax()) == 3
    tok, n = hc.shortlist_argmax(h, rows, assign, centres, 0.5)    # one cluster: the one with the higher centre score
    assert tok in (0, 1, 2) and n == 3
    assert hc.shortlist_argmax(h, rows, assign, centres, 1.0)[0] == 3
