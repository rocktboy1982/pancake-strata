"""Unit tests for bfla_recall.py (the fused prompt attention's quality gate): the streamed logits comparison, the gate, the
arms' environment and the "acted" check, on synthetic files.  No engine, no GPU.
Run: python -m pytest tools/pancake/test_bfla_recall.py   (or python tools/pancake/test_bfla_recall.py)"""
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bfla_recall as B  # noqa: E402


def write(path, rows):
    np.asarray(rows, dtype=np.float32).tofile(path)


def kl(p_logits, q_logits):
    lp = p_logits - np.log(np.exp(p_logits).sum())
    lq = q_logits - np.log(np.exp(q_logits).sum())
    return float((np.exp(lp) * (lp - lq)).sum())


def test_identical_logits_agree_with_zero_kl():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(5, 11)).astype(np.float32)
    ids = [int(r.argmax()) for r in ref]
    with tempfile.TemporaryDirectory() as d:
        write(Path(d) / "ref", ref)
        write(Path(d) / "forced", np.vstack([ref, rng.normal(size=(1, 11))]))   # the forced run has one more row
        agree, n, kls = B.compare_logits(Path(d) / "ref", Path(d) / "forced", ids)
    assert (agree, n) == (5, 5) and max(abs(x) for x in kls) < 1e-9


def test_kl_and_agreement_on_a_perturbed_row():
    ref = np.array([[2.0, 0.0, -1.0], [0.0, 3.0, 0.5]], dtype=np.float32)
    test = np.array([[0.0, 2.5, -1.0], [0.0, 3.0, 0.5]], dtype=np.float32)
    with tempfile.TemporaryDirectory() as d:
        write(Path(d) / "ref", ref)
        write(Path(d) / "test", test)
        agree, n, kls = B.compare_logits(Path(d) / "ref", Path(d) / "test", [0, 1])
    assert (agree, n) == (1, 2)
    assert math.isclose(kls[0], kl(ref[0].astype(np.float64), test[0].astype(np.float64)), rel_tol=1e-6)
    assert abs(kls[1]) < 1e-12


def test_a_file_of_the_wrong_width_is_refused():
    with tempfile.TemporaryDirectory() as d:
        write(Path(d) / "ref", np.zeros((3, 4)))
        write(Path(d) / "bad", np.zeros(10))
        try:
            B.compare_logits(Path(d) / "ref", Path(d) / "bad", [0, 0, 0])
        except ValueError:
            return
    raise AssertionError("a 10-float file is not 4-wide rows")


def res(**kw):
    r = {"acted": True, "n_tokens": 320, "mean_kl": 2e-4, "top1_agree": 0.997, "recall_test": 6, "recall_ref": 6}
    r.update(kw)
    return r


def test_gate():
    assert B.gate(res()) == (True, "pass")
    assert not B.gate(res(acted=False))[0] and "VOID" in B.gate(res(acted=False))[1]
    assert not B.gate(res(n_tokens=48))[0]                 # the batch-2 run: 48 tokens cannot show 99%
    assert not B.gate(res(mean_kl=0.0014))[0]              # the batch-2 64K KL
    assert not B.gate(res(top1_agree=0.985))[0]
    assert not B.gate(res(recall_test=5))[0]


def test_clean_env_keeps_the_ram_cap_only():
    e = B.clean_env({"PATH": "/bin", "STRATA_BFLA_SPARSE": "1", "STRATA_ATTN_WINDOW": "1", "STRATA_RAM_KEEP_GIB": "12"})
    assert e == {"PATH": "/bin", "STRATA_RAM_KEEP_GIB": "12"}


def test_arm_env_and_acted():
    e = B.arm_env({"HIP_VISIBLE_DEVICES": "0"}, "dense", {"STRATA_BFLA_MIN": "4096"})
    assert e == {"HIP_VISIBLE_DEVICES": "0", "STRATA_BFLA_DENSE": "1", "STRATA_BFLA_MIN": "4096"}
    assert B.arm_env({}, "sparse", {}) == {"STRATA_BFLA_SPARSE": "1"}
    log = "strata-q36: STRATA_BFLA_SPARSE acted: prompt attention of positions 32768..36863 by the fused kernel\n"
    assert B.acted(log, "sparse") and not B.acted(log, "dense") and not B.acted("", "sparse")


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f()
            print("ok", name)
