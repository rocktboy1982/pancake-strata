"""Tests for head_bound.py on synthetic data (MOD_PLAN.md section 6 item 46): the two-stage search is exact, the block bound is never
looser than the whole-row bound, the oracle floor is no worse than a realised first stage, and the dump reader round-trips."""
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import head_bound as hb  # noqa: E402


def synth(v=400, d=32, steps=20, seed=1):
    rng = np.random.default_rng(seed)
    norms = np.exp(rng.normal(0, 0.6, v))   # a spread of row norms: what makes the bound bite
    rows = rng.normal(size=(v, d)) * norms[:, None]
    hidden = []
    for _ in range(steps):
        t = int(rng.integers(v))
        hidden.append(rows[t] / np.linalg.norm(rows[t]) * 3.0 + rng.normal(size=d) * 0.3)   # points at one row
    return rows.astype(np.float32), np.array(hidden, np.float32)


def test_upper_bound_holds_for_every_row():
    rows, hidden = synth()
    for nb in (1, 3, 8):
        wn, edges = hb.block_norms(rows, nb)
        for h in hidden:
            ub = hb.upper_bounds(wn, edges, h)
            assert np.all(ub >= rows.astype(np.float64) @ h.astype(np.float64) - 1e-6)


def test_block_bound_is_never_looser_than_the_row_bound():
    rows, hidden = synth()
    w1, e1 = hb.block_norms(rows, 1)
    w8, e8 = hb.block_norms(rows, 8)
    for h in hidden:
        assert np.all(hb.upper_bounds(w8, e8, h) <= hb.upper_bounds(w1, e1, h) + 1e-9)


def test_two_stage_is_exact_and_prunes():
    rows, hidden = synth()
    res = hb.evaluate(rows, hidden, blocks=(1, 4), ms=(8, 32), max_steps=20)
    assert all(r["exact"] for r in res)
    assert any(r["pruned_mean"] > 0.05 for r in res if r["m"] > 0)   # the synthetic spread must let the bound skip rows


def test_oracle_floor_is_no_worse_than_a_realised_first_stage():
    rows, hidden = synth()
    res = hb.evaluate(rows, hidden, blocks=(4,), ms=(8, 64), max_steps=20)
    by = {r["m"]: r for r in res}
    assert by[0]["scored_mean"] <= by[8]["scored_mean"] + 1e-12
    assert by[0]["scored_mean"] <= by[64]["scored_mean"] + 1e-12


def test_more_blocks_never_score_more_rows():
    rows, hidden = synth()
    res = hb.evaluate(rows, hidden, blocks=(1, 8), ms=(32,), max_steps=20)
    s = {(r["blocks"], r["m"]): r["scored_mean"] for r in res}
    assert s[(8, 32)] <= s[(1, 32)] + 1e-12
    assert s[(8, 0)] <= s[(1, 0)] + 1e-12


def test_flat_rows_prune_nothing():
    rng = np.random.default_rng(3)
    rows = rng.normal(size=(300, 16)).astype(np.float32)
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)   # equal norms and a random hidden: the bound cannot separate
    hidden = rng.normal(size=(10, 16)).astype(np.float32)
    res = hb.evaluate(rows, hidden, blocks=(1,), ms=(16,), max_steps=10)
    r = [x for x in res if x["m"] == 16][0]
    assert r["exact"] and r["pruned_mean"] < 0.3


def test_dump_roundtrip(tmp_path):
    p = tmp_path / "h.hdmp"
    a = np.arange(3 * 4, dtype="<f4").reshape(3, 4)
    b = (np.arange(2 * 4, dtype="<f4") + 100).reshape(2, 4)
    with open(p, "wb") as f:
        f.write(struct.pack("<IIII", hb.MAGIC, 1, 4, 99))
        for w, pos in ((a, 7), (b, 10)):
            f.write(struct.pack("<Iq", w.shape[0], pos))
            f.write(w.tobytes())
        f.write(struct.pack("<Iq", 5, 20))   # a window cut short at the end of the run
        f.write(b"\x00" * 8)
    h, nv = hb.read_dump(str(p))
    assert nv == 99 and h.shape == (5, 4)
    assert np.array_equal(h, np.concatenate([a, b]))


def test_bad_dump_is_refused(tmp_path):
    p = tmp_path / "x"
    p.write_bytes(b"\x00" * 32)
    with pytest.raises(ValueError):
        hb.read_dump(str(p))


def test_main_end_to_end(tmp_path, capsys):
    rows, hidden = synth(v=200, d=16, steps=6)
    np.save(tmp_path / "r.npy", rows)
    np.save(tmp_path / "h.npy", hidden)
    rc = hb.main(["eval", str(tmp_path / "r.npy"), str(tmp_path / "h.npy"), "--blocks", "1,2", "--ms", "8"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "oracle floor" in out and "best realised" in out
