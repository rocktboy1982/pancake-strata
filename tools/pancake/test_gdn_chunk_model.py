"""Tests for probes/gdn_chunk_model.py (MOD_PLAN A31 numerical model).
Run: python tools/pancake/test_gdn_chunk_model.py   (or pytest)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "probes"))
import gdn_chunk_model as M  # noqa: E402


def test_blockwise_inverse_is_exact_in_fp64():
    rng = np.random.default_rng(3)
    a = np.tril(0.3 * rng.standard_normal((64, 64)) / 8, -1)
    assert np.abs(M.inverse(a, "block") - M.inverse(a, "exact")).max() < 1e-12


def test_chunked_forms_equal_the_recurrence_in_fp64():
    for decay in (0.5, 0.01):
        e = M.compare(n=256, decay=decay)
        for name in ("c16 fp64", "c64 block fp64"):
            assert max(e[name]) < 1e-12, (decay, name, e[name])


def test_chunk64_stays_in_the_chunk16_fp16_error_class():
    for decay in (0.5, 0.01):
        e = M.compare(n=256, decay=decay)
        base = e["c16 fp16"]
        assert 1e-4 < base[0] < 2e-3, base   # operand rounding (2^-11), not an algebra error
        for name in ("c64 block fp16", "c64 block16 fp16"):
            assert e[name][0] < 1.5 * base[0] and e[name][1] < 1.5 * base[1], (decay, name, e[name], base)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
