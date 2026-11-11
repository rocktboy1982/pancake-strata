"""Tests for probes/lds_lut_conflicts.py (MOD_PLAN A37 bank model).  Run: python tools/pancake/test_lds_lut_conflicts.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "probes"))
import lds_lut_conflicts as L  # noqa: E402


def test_slot_layout_matches_the_kernel():
    # the copy routine writes slot s <- entry ((s >> 4) * B) | (s & (B - 1)); a lane of copy g reads entry i at
    # (t - t % B) + lut_dup_slot<B>(i): every lane must find its entry where the copy put it, inside its own bank pairs
    for copies in (8, 16):
        b = 16 // copies
        table = {s: ((s >> 4) * b) | (s & (b - 1)) for s in range(256 * copies)}
        for t in range(16):
            for i in range(256):
                s = L.slot(i, copies, t)
                assert table[s] == i
                assert (t - t % b) <= s % 16 < (t - t % b) + b


def test_sixteen_copies_never_conflict():
    rng = np.random.default_rng(3)
    for _ in range(200):
        a = rng.integers(0, 256, 16).tolist()
        b = rng.integers(0, 256, 16).tolist()
        assert L.wave_clocks(a, b, 16) == 2


def test_plain_table_worst_and_best_cases():
    same_pair = [16 * t for t in range(16)]        # 16 distinct entries in bank pair 0
    spread = list(range(16))                       # one entry per bank pair
    assert L.wave_clocks(same_pair, spread, 1) == 16 + 1
    assert L.wave_clocks([5] * 16, [5] * 16, 1) == 2   # broadcast


def test_uniform_model_numbers():
    rng = np.random.default_rng(1)
    v1 = L.uniform(1024, 1, 4000, rng)
    v16 = L.uniform(256, 16, 500, rng)
    assert 5.6 < v1 < 6.4, v1      # ~6.07 (pancake-docs/batch2/A37-gpu-kernels.md)
    assert v16 == 2.0


def test_block_indices():
    blk = np.zeros(82, np.uint8)
    blk[2 + 4 * 3 + 1] = 0x7F      # qs of q8_1 block 3, lookup 1
    blk[66 + 3] = 0b00001100       # qh of block 3: lookup 1's high bits = 0b11
    assert L.block_indices(22, blk, 3)[1] == 0x37F
    blk2 = np.zeros(66, np.uint8)
    blk2[2 + 8 * 5 + 2] = 200
    assert L.block_indices(16, blk2, 5)[2] == 200


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
