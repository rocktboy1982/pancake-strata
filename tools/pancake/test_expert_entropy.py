"""A29: expert_entropy.py estimators and the whole tool over a synthetic Q2_0 GGUF.  No model, no GPU.

Run: python -m pytest tools/pancake/test_expert_entropy.py   (or python tools/pancake/test_expert_entropy.py)"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "tools"))
import expert_entropy as E  # noqa: E402
from gguf_writer import GGUFWriter  # noqa: E402


def test_entropy_of_known_distributions():
    rng = np.random.default_rng(1)
    uniform = rng.integers(0, 256, 1 << 20, dtype=np.uint8)
    assert abs(E.byte_entropy(uniform) - 8.0) < 0.01
    assert abs(E.nibble_entropy(uniform) - 4.0) < 0.01
    assert abs(E.crumb_entropy(uniform) - 2.0) < 0.01
    assert E.byte_entropy(np.zeros(1000, dtype=np.uint8)) == 0.0
    two = np.array([0, 255] * 5000, dtype=np.uint8)
    assert abs(E.byte_entropy(two) - 1.0) < 1e-9
    assert abs(E.nibble_entropy(two) - 1.0) < 1e-9
    assert E.entropy_bits([0, 0]) == 0.0


def test_order1_entropy_sees_dependence():
    rng = np.random.default_rng(2)
    iid = rng.integers(0, 256, 1 << 20, dtype=np.uint8)
    assert E.order1_entropy(iid) > 7.9
    seq = np.arange(1 << 16, dtype=np.uint32).astype(np.uint8)      # the next byte is determined by the previous one
    assert E.order1_entropy(seq) < 0.01


def test_position_entropy_separates_scale_and_payload():
    rng = np.random.default_rng(3)
    n = 20000
    blocks = np.zeros((n, 18), dtype=np.uint8)
    blocks[:, 0] = rng.integers(0, 256, n)           # random scale low byte
    blocks[:, 1] = 0x3C                              # constant scale high byte
    blocks[:, 2:] = rng.integers(0, 256, (n, 16))
    pos = E.position_entropies(blocks.reshape(-1), 18)
    assert pos[1] == 0.0 and pos[0] > 7.9 and pos[5] > 7.9


def test_compressors_find_redundancy_and_leave_noise_alone():
    rng = np.random.default_rng(4)
    noise = rng.integers(0, 256, 1 << 18, dtype=np.uint8).tobytes()
    rep = (b"abcd" * (1 << 16))
    n = E.compressed_sizes([noise])
    r = E.compressed_sizes([rep])
    assert n["zlib9"] >= len(noise) * 0.99 and n["lzma"] >= len(noise) * 0.99
    assert r["zlib9"] < len(rep) * 0.05 and r["lzma"] < len(rep) * 0.05


def test_whole_tool_on_a_synthetic_q2_0_gguf():
    rng = np.random.default_rng(5)
    with tempfile.TemporaryDirectory() as d:
        w = GGUFWriter()
        w.add("general.architecture", "qwen4exp")
        for blk in range(3):
            # a skewed 2-bit code distribution (Q2_0 symbols are not uniform): entropy below 2 bits per symbol
            a = rng.standard_normal((64, 256)).astype(np.float32)
            w.add_q2_0(f"blk.{blk}.ffn_down_exps.weight", a, [256, 64])
        w.add_f32("blk.0.ssm_a", rng.standard_normal((4, 8)).astype(np.float32))   # not an expert tensor: ignored
        p = w.write(Path(d) / "tiny.gguf")
        rep = E.run(p, tensors=3, window_kb=4, windows=2, max_mb=1, pattern="_exps")
        assert list(rep["types"]) == ["Q2_0"] and len(rep["tensors_sampled"]) == 3
        r = rep["types"]["Q2_0"]
        assert r["block_bytes"] == 18 and r["bytes_sampled"] % 18 == 0 and r["bytes_sampled"] > 0
        assert 0 < r["crumb_entropy_bits"] < 2.0 and r["byte_entropy_bits"] < 8.0
        assert r["saving_pct"]["zlib9"] is not None
        closed, best = E.verdict(rep, 20.0)
        assert isinstance(closed, bool) and best == r["best_saving_pct"]
        out = Path(d) / "r.json"
        assert E.main([str(p), "--tensors", "3", "--window-kb", "4", "--json", str(out)]) == 0
        assert json.loads(out.read_text())[0]["file"] == "tiny.gguf"


def test_a_file_without_expert_tensors_is_skipped_not_fatal():
    with tempfile.TemporaryDirectory() as d:
        w = GGUFWriter()
        w.add("general.architecture", "x")
        w.add_f32("blk.0.ssm_a", np.ones((4, 8), dtype=np.float32))
        p = w.write(Path(d) / "none.gguf")
        rep = E.run(p, 3, 4, 2, 1, "_exps")
        assert rep["types"] == {} and "skipped" in rep
        E.print_report(rep, 20.0)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
