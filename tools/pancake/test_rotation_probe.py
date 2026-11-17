"""Tests for probes/rotation_probe.py and hip_resources.py on SYNTHETIC data (no engine, no GPU, no model).
The hip_resources fixture is the shape of real clang -Rpass-analysis=kernel-resource-usage output for gfx1100 (compiled offline from
src/kernels/cuda/qsa_decode_attn.cu with the ROCm 6.2 SDK on the Windows PC; nothing was run on a GPU)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "probes"))
import hip_resources as hr  # noqa: E402
import rotation_probe as rp  # noqa: E402


def test_rotation_is_orthogonal_and_preserves_the_product():
    r = rp.rotation(256)
    assert np.allclose(r @ r.T, np.eye(256), atol=1e-12)
    w, x = rp.synthetic(rows=16, tokens=8, k=512)
    assert rp.rel_err(x @ w.T, rp.rotate(x) @ rp.rotate(w).T) < 1e-12


def test_sign_hash_matches_the_paper_formula():
    s = rp.sign_diag(8)
    expect = [(-1.0 if ((j * 2654435761) % 2 ** 32) >> 31 else 1.0) for j in range(8)]
    assert list(s) == expect


def test_outlier_channels_make_rotation_win_at_4_bit_activations():
    w, x = rp.synthetic(rows=64, tokens=32, k=1024, seed=1)
    res = rp.probe(w, x, wbits=(4,), abits=(4, 8))
    a4 = res["acts"][4]
    assert a4["rot"]["err_best"] < a4["plain"]["err_best"] * 0.7
    assert res["kurtosis"]["acts_rot"] < res["kurtosis"]["acts_plain"]
    assert res["rotation_check"] < 1e-12
    assert res["product"]["w4a4"]["rot"] < res["product"]["w4a4"]["plain"]


def test_gaussian_data_gets_no_claimed_benefit():
    rng = np.random.default_rng(3)
    w, x = rng.standard_normal((32, 512)), rng.standard_normal((32, 512))
    res = rp.probe(w, x, wbits=(4,), abits=(8,))
    r = res["product"]["w4a8"]
    assert abs(r["rot"] / r["plain"] - 1.0) < 0.15   # about 1: rotation helps outliers, not Gaussian data


def test_best_clip_is_never_worse_than_no_clip():
    w, x = rp.synthetic(rows=16, tokens=16, k=512, seed=2)
    res = rp.probe(w, x, wbits=(4,), abits=(4,), clips=(1.0, 0.9, 0.8))
    for tag in ("plain", "rot"):
        e = res["acts"][4][tag]
        assert e["err_best"] <= e["err_clip1"] + 1e-15


def test_quantize_levels_and_zero_rows():
    x = np.zeros((2, 64))
    assert np.array_equal(rp.quantize(x, 4), x)
    y = rp.quantize(np.linspace(-1, 1, 64)[None, :], 2, group=32)
    assert len(np.unique(np.round(y[0, :32] / (np.abs(y[0, :32]).max() or 1), 6))) <= 4


FIXTURE = """\
k.cu:1:1: remark: Function Name: _Z6k_a [-Rpass-analysis=kernel-resource-usage]
k.cu:1:1: remark:     SGPRs: 41 [-Rpass-analysis=kernel-resource-usage]
k.cu:1:1: remark:     VGPRs: 124 [-Rpass-analysis=kernel-resource-usage]
k.cu:1:1: remark:     ScratchSize [bytes/lane]: 0 [-Rpass-analysis=kernel-resource-usage]
k.cu:1:1: remark:     Occupancy [waves/SIMD]: 10 [-Rpass-analysis=kernel-resource-usage]
k.cu:1:1: remark:     SGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
k.cu:1:1: remark:     VGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
k.cu:1:1: remark:     LDS Size [bytes/block]: 15872 [-Rpass-analysis=kernel-resource-usage]
k.cu:1:1: remark: Function Name: _Z6k_b [-Rpass-analysis=kernel-resource-usage]
k.cu:1:1: remark:     SGPRs: 38 [-Rpass-analysis=kernel-resource-usage]
k.cu:1:1: remark:     VGPRs: 120 [-Rpass-analysis=kernel-resource-usage]
k.cu:1:1: remark:     ScratchSize [bytes/lane]: 32 [-Rpass-analysis=kernel-resource-usage]
k.cu:1:1: remark:     Occupancy [waves/SIMD]: 12 [-Rpass-analysis=kernel-resource-usage]
k.cu:1:1: remark:     SGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
k.cu:1:1: remark:     VGPRs Spill: 30 [-Rpass-analysis=kernel-resource-usage]
k.cu:1:1: remark:     LDS Size [bytes/block]: 15872 [-Rpass-analysis=kernel-resource-usage]
k.cu:1:1: remark: Function Name: _Z6k_c [-Rpass-analysis=kernel-resource-usage]
k.cu:1:1: remark:     SGPRs: 40 [-Rpass-analysis=kernel-resource-usage]
k.cu:1:1: remark:     VGPRs: 46 [-Rpass-analysis=kernel-resource-usage]
k.cu:1:1: remark:     ScratchSize [bytes/lane]: 0 [-Rpass-analysis=kernel-resource-usage]
k.cu:1:1: remark:     Occupancy [waves/SIMD]: 8 [-Rpass-analysis=kernel-resource-usage]
k.cu:1:1: remark:     SGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
k.cu:1:1: remark:     VGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
k.cu:1:1: remark:     LDS Size [bytes/block]: 31744 [-Rpass-analysis=kernel-resource-usage]
"""


def test_hip_resources_flags_cliff_spill_and_lds_limit():
    ks, s = hr.report(hr.parse(FIXTURE.splitlines()), cliff=8)
    by = {k["name"]: k for k in ks}
    assert by["_Z6k_a"]["cliff"] == (12, 4) and by["_Z6k_a"]["relation"] == "match"
    assert by["_Z6k_b"]["spill"] and by["_Z6k_b"]["cliff"] is None   # 120 VGPRs = 12 waves; 16 waves needs 96, 24 fewer: beyond --cliff 8
    assert by["_Z6k_c"]["relation"] == "other-limit" and by["_Z6k_c"]["cliff"] is None
    assert s["contradiction"] == 0 and s["spills"] == 1


def test_hip_resources_model_matches_measured_pairs():
    # pairs taken from the compiler's remarks for gfx1100: (VGPRs, waves)
    for v, w in ((9, 16), (72, 16), (96, 16), (124, 10), (126, 10), (120, 12), (109, 12), (110, 12)):
        assert hr.model_waves(v) == w, (v, w)


def test_hip_resources_exit_codes(tmp_path):
    f = tmp_path / "r.txt"
    f.write_text(FIXTURE)
    assert hr.main([str(f)]) == 0
    assert hr.main([str(f), "--fail-on-spill"]) == 1
    e = tmp_path / "e.txt"
    e.write_text("nothing here\n")
    assert hr.main([str(e)]) == 2
    bad = FIXTURE.replace("Occupancy [waves/SIMD]: 10", "Occupancy [waves/SIMD]: 16")
    b = tmp_path / "b.txt"
    b.write_text(bad)
    assert hr.main([str(b)]) == 3


# ---- rotate_gptq.py (LEADS.md Part L L1) -------------------------------------------------------------------------------------------
import rotate_gptq as rg  # noqa: E402


def test_gptq_beats_rtn_on_the_layer_output_in_the_rotated_basis():
    rng = np.random.default_rng(5)
    w = rng.standard_normal((16, 256))
    x = rng.standard_normal((512, 256)) @ rng.standard_normal((256, 256)) * 0.1   # correlated activations: GPTQ's case
    wr, xr = rp.rotate(w, 256), rp.rotate(x, 256)
    codes, scales, deq = rg.gptq(wr, xr, bits=4, group=32)
    assert codes.min() >= -8 and codes.max() <= 7
    assert scales.shape == (16, 8)
    assert np.allclose(deq, codes.astype(np.float64) * np.repeat(scales, 32, axis=1), atol=1e-5)
    assert rg.out_err(wr, deq, xr) < rg.out_err(wr, rg.rtn(wr, 4, 32), xr)


def test_gptq_cli_roundtrip(tmp_path):
    rng = np.random.default_rng(6)
    np.save(tmp_path / "w.npy", rng.standard_normal((8, 256)))
    np.save(tmp_path / "x.npy", rng.standard_normal((256, 256)))
    assert rg.main(["--weights", str(tmp_path / "w.npy"), "--acts", str(tmp_path / "x.npy"), "--out", str(tmp_path / "q.npz")]) == 0
    z = np.load(tmp_path / "q.npz")
    assert z["codes"].shape == (8, 256) and z["deq"].dtype == np.float32
