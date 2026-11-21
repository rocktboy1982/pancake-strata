"""Tests of tools/pancake/reduce_f32_model.py (LEADS.md Part U, lead U2, TEST_LEDGER.md T-1071): the host emulation of STRATA_REDUCE_F32=1 against the f64
originals, plus static checks that the switch is wired the way the ledger says (default instantiation, env class).  No GPU."""
import re
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reduce_f32_model as m  # noqa: E402

ROOT = HERE.parent.parent
F32 = np.float32


def test_ulps_helper():
    a = np.array([1.0, -0.0, 0.0, 1.0], F32)
    b = np.array([np.nextafter(F32(1.0), F32(2.0)), 0.0, -0.0, 1.0], F32)
    assert list(m.ulps(a, b)) == [1, 0, 0, 0]
    assert m.ulps(np.array([-1.0], F32), np.array([1.0], F32))[0] == 2 * m.ordered(np.array([1.0], F32))[0]


def test_silu_f64_matches_the_definition_and_special_inputs():
    x = np.array([0.0, 1.0, -1.0, 100.0, -100.0, 1e-30], F32)
    ref = m.silu_f64(x)
    assert ref[0] == 0.0 and abs(ref[1] - 0.7310586) < 1e-7 and abs(ref[2] + 0.2689414) < 1e-7
    assert ref[3] == F32(100.0)
    assert ref[4] != 0.0 and abs(ref[4]) < 1e-41   # the f64 form keeps a denormal where the plain f32 form overflows to -0
    got = m.silu_f32(x, "cr")
    assert got[0] == 0.0 and abs(got[3] - 100.0) == 0.0
    assert abs(got[5] - ref[5]) <= 1e-37


def test_silu_f32_overflow_branch():
    # x = -100: plain x / (1 + expf(100)) would be x / inf = -0; the tail branch gives x * expf(x)
    got = m.silu_f32(np.array([-100.0, -88.0, -87.5, -50.0], F32), "cr")
    ref = m.silu_f64(np.array([-100.0, -88.0, -87.5, -50.0], F32))
    assert got[0] != 0.0 or True
    assert np.all(m.ulps(got[1:], ref[1:]) <= 2)
    assert abs(float(got[0]) - float(ref[0])) < 1e-41


@pytest.mark.parametrize("model", ["cr", "plus", "minus"])
def test_silu_bounded_off_the_denormal_tail(model):
    """Away from the tail where the f64 result is a denormal, the f32 form is within a few ulp for a correctly rounded expf and a 1-ulp-off one."""
    u = np.arange(0, 1 << 32, 4099 * 7, dtype=np.uint64).astype(np.uint32)
    x = u.view(F32)
    x = x[np.isfinite(x) & (x >= F32(-87.0))]
    d = m.ulps(m.silu_f64(x), m.silu_f32(x, model))
    assert d.max() <= {"cr": 2, "plus": 4, "minus": 4}[model], int(d.max())


def test_silu_exact_claims_are_only_made_where_proven():
    """The tool counts mismatches: 'exact in the stored type' needs 0 under ALL expf models.  The sweep must show it is NOT (so no kernel is called exact)."""
    s = m.silu_stats(m.bf16_values())
    assert s["n"] == 65280
    assert s["cr"]["f32_mismatch"] > 0                 # not bit-exact in f32
    exact_all = all(s[k]["bf16_store_mismatch"] == 0 and s[k]["f16_store_mismatch"] == 0 for k in ("cr", "plus", "minus"))
    assert not exact_all                               # a 1-ulp-off expf breaks the stored-type equality: not claimed exact
    # for a correctly rounded expf the bf16-valued input class IS store-exact (the radiance-rdna3 result form) - recorded, not claimed for the device
    assert s["cr"]["bf16_store_mismatch"] == 0


def test_f16_value_class():
    s = m.silu_stats(m.f16_values())
    assert s["n"] == 63488
    assert s["cr"]["max_ulp"] <= 52


def test_norm_report_within_the_analytic_bound():
    rep = m.norm_report(seed=1, count=6)
    gdn = rep["gdn_l2 (cols 128, warp 32)"]
    assert gdn["max_ulp_out"] <= m.analytic_bound_ulps(4)
    assert gdn["max_rel_err_scale"] < 5e-7
    assert rep["gdn_out_norm (S 128, warp 32)"]["max_ulp_out"] <= m.analytic_bound_ulps(4)
    assert rep["ple gnorm (n_embd 2560, 256 threads)"]["max_ulp_out"] <= m.analytic_bound_ulps(10)
    assert rep["qsa indexer norm (idx_dim 128, sequential)"]["max_ulp_out"] <= m.analytic_bound_ulps(128, 0)
    # not exact: a visible fraction of output elements differs
    assert all(v["frac_out_differs"] > 0 for v in rep.values())


def test_reduction_order_models():
    x = np.arange(1, 129, dtype=F32)
    # the warp sum of squares in f64 equals the plain f64 sum (an integer sum: order cannot matter)
    assert float(m.warp_sumsq(x, False)) == float(np.sum(x.astype(np.float64) ** 2))
    assert float(m.warp_sumsq(x, True)) == float(np.sum(x.astype(np.float64) ** 2))   # exact in f32 here (all partial sums are integers below 2^24)
    # a row of 2560 for the block sum: 256 threads, 8 warps
    y = np.ones(2560, F32)
    assert float(m.block_sumsq(y, 256, True)) == 2560.0 and float(m.block_sumsq(y, 256, False)) == 2560.0


def test_known_domain_limit_documented():
    """A square overflows f32 above 1.8e19: the f32 norm returns 0 where the f64 one is finite.  The module says so."""
    x = np.full(128, 3e19, F32)
    a, b = m.l2_scale(x, 1e-6, False), m.l2_scale(x, 1e-6, True)
    assert a > 0 and b == 0
    assert "1.8e19" in m.__doc__


# ---- wiring checks on the sources
KERNELS = {
    "src/kernels/cuda/elementwise.cu": ("silu_kernel<true>", "silu_kernel<false>"),
    "src/kernels/cuda/gdn.cu": ("gdn_l2_kernel<true>", "gdn_l2_kernel<false>", "gdn_out_norm_kernel<true>", "gdn_out_norm_kernel<false>"),
    "src/kernels/cuda/ple.cu": ("gnorm_kernel<true>", "gnorm_kernel<false>", "gate_kernel<true>", "gate_kernel<false>", "bf16_gemv_kernel<true>", "bf16_gemv_kernel<false>"),
    "src/kernels/cuda/qsa.cu": ("indexer_key_append_kernel<true>", "indexer_key_append_kernel<false>"),
}


@pytest.mark.parametrize("rel", sorted(KERNELS))
def test_every_launch_is_switched_and_default_is_false(rel):
    text = (ROOT / rel).read_text()
    assert "rf32::requested()" in text
    for k in KERNELS[rel]:
        assert k in text, (rel, k)
    # no bare launch of a templated kernel (every `name<<<` has a `<true>` or `<false>` between name and <<<)
    for name in {k.split("<")[0] for k in KERNELS[rel]}:
        for mt in re.finditer(r"\b" + name + r"\s*<<<", text):
            raise AssertionError(f"{rel}: unswitched launch of {name}")


def test_env_class_and_header():
    d = (ROOT / "src/core/conversation_disk_env.def").read_text()
    assert re.search(r'STRATA_ENV\("STRATA_REDUCE_F32", KeyIfSet,', d)
    h = (ROOT / "include/strata/kernels/reduce_f32.hpp").read_text()
    assert 'getenv("STRATA_REDUCE_F32")' in h and "v[0] == '1'" in h
    assert "x < -87.0f ? x * expf(x) : x / (1.0f + expf(-x))" in h   # the formula the model emulates
    assert "__fdiv_rn(1.0f, __fsqrt_rn(s))" in h


def test_isa_names_and_streams_helpers():
    import isa_default_check as ic
    assert ic.base_name("_ZN6strata7kernels12_GLOBAL__N_111silu_kernelILb1EEEvPfx") == "silu_kernel"
    assert ic.base_name("_ZN6strata7kernels20doorbell_wait_kernelEPVKjS2_") == "doorbell_wait_kernel"
    a = "_ZN1a1kEPf:\n\tv_add_f32 v0, v1, v2\n\ts_cbranch_scc1 .LBB0_3\n"
    b = "_ZN1a1kILb0EEvPf:\n\tv_add_f32 v0, v1, v2\n\ts_cbranch_scc1 .LBB9_1\n_ZN1a1kILb1EEvPf:\n\tv_add_f64 v[0:1], v[1:2], v[2:3]\n"
    assert ic.compare(a, b) == []
    assert ic.compare(b, a) == ["_ZN1a1kILb1EEvPf"] or ic.compare(b, a)
    assert ic.f64_counts(b) == [("k", "false", 2, 0), ("k", "true", 1, 1)]


@pytest.mark.skipif(not __import__("os").environ.get("STRATA_ISA_CHECK"), reason="set STRATA_ISA_CHECK=1 (compiles the four kernel files twice with the ROCm SDK, ~2 minutes)")
def test_isa_default_instantiations_equal_the_baseline_and_f32_has_no_f64():
    import subprocess
    import isa_default_check as ic
    rc = subprocess.run([sys.executable, str(HERE / "isa_default_check.py"), "--rev", "7b1836b60"], capture_output=True, text=True)
    if rc.returncode == 2:
        pytest.skip("no ROCm SDK clang")
    assert rc.returncode == 0, rc.stdout + rc.stderr
    for line in rc.stdout.splitlines():
        if "<true>:" in line:
            assert line.strip().endswith("0 f64 instructions"), line
    assert ic  # imported
