"""Fixture tests for tools/pancake/probes/valu_rate_report.py (LEADS.md Part U U4): the parser, the percentages, the verdict
ratios and the ISA check, on stored text.  No GPU, no box."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "probes"))
import valu_rate_report as V  # noqa: E402

# a fabricated probe output (numbers are NOT measurements: they only exercise the arithmetic)
OUT = '''DEVICE name="AMD Radeon RX 7900 XT" arch=gfx1100 cus=84 reported_clock_mhz=2400 iters=20000 reps=5
RESULT mode=fma waves_per_simd=4 unit=flop ops_per_clk_cu=120.000 inst_per_clk_cu=60.000 ops_per_s=2.0e+13 implied_clock_ghz=1.9800 best_ops_per_clk_cu=121.000
RESULT mode=fma waves_per_simd=8 unit=flop ops_per_clk_cu=126.000 inst_per_clk_cu=63.000 ops_per_s=2.1e+13 implied_clock_ghz=1.9800 best_ops_per_clk_cu=127.000
RESULT mode=dot4 waves_per_simd=8 unit=op ops_per_clk_cu=252.000 inst_per_clk_cu=31.500 ops_per_s=4.2e+13 implied_clock_ghz=1.9800 best_ops_per_clk_cu=253.000
RESULT mode=dot2 waves_per_simd=8 unit=flop ops_per_clk_cu=252.000 inst_per_clk_cu=63.000 ops_per_s=4.2e+13 implied_clock_ghz=1.9800 best_ops_per_clk_cu=253.000
RESULT mode=vopd waves_per_simd=8 unit=flop ops_per_clk_cu=250.000 inst_per_clk_cu=125.000 ops_per_s=4.1e+13 implied_clock_ghz=1.9800 best_ops_per_clk_cu=251.000
RESULT mode=lds_r128 waves_per_simd=8 unit=byte ops_per_clk_cu=96.000 inst_per_clk_cu=6.000 ops_per_s=1.6e+13 implied_clock_ghz=1.9800 best_ops_per_clk_cu=97.000
RESULT mode=wmma_f16 waves_per_simd=8 unit=flop ops_per_clk_cu=460.800 inst_per_clk_cu=0.056 ops_per_s=7.7e+13 implied_clock_ghz=1.9800 best_ops_per_clk_cu=461.000
garbage line
RESULT mode=broken waves_per_simd=notanumber
'''


def test_parse_skips_bad_lines():
    dev, rows = V.parse(OUT)
    assert dev["arch"] == "gfx1100" and dev["name"] == "AMD Radeon RX 7900 XT" and dev["cus"] == "84"
    assert len(rows) == 7 and {r["mode"] for r in rows} == {"fma", "dot4", "dot2", "vopd", "lds_r128", "wmma_f16"}
    assert rows[0]["waves"] == 4 and abs(rows[1]["inst_per_clk_cu"] - 63.0) < 1e-9


def test_percent_of_theory():
    _, rows = V.parse(OUT)
    by = {(r["mode"], r["waves"]): r for r in rows}
    th, pct = V.percent_of_theory(by[("fma", 8)], 128.0, 128.0)
    assert th == 64.0 and abs(pct - 100.0 * 63 / 64) < 1e-9
    th, pct = V.percent_of_theory(by[("dot4", 8)], 128.0, 128.0)
    assert abs(pct - 100.0 * 31.5 / 64) < 1e-9
    th, pct = V.percent_of_theory(by[("vopd", 8)], 128.0, 128.0)
    assert th == 128.0 and abs(pct - 100.0 * 125 / 128) < 1e-9
    th, pct = V.percent_of_theory(by[("lds_r128", 8)], 128.0, 128.0)
    assert th == 128.0 and abs(pct - 75.0) < 1e-9
    th, pct = V.percent_of_theory(by[("wmma_f16", 8)], 128.0, 128.0)
    assert th == 512.0 and abs(pct - 90.0) < 1e-9


def test_report_text_and_verdicts():
    text, ok = V.report(OUT, 84, 128.0, 128.0)
    assert ok
    assert "dot4   / fma = 0.50" in text                      # 31.5 / 63 (the best fma occupancy)
    assert "VERDICT dot4: reduced rate" in text
    assert "VERDICT vopd: pairs issue" in text
    assert "lds_r128 96.0 bytes per clock per CU" in text
    # the chip rate and the theoretical chip rate use the clock the kernel measured (1.98 GHz), not the reported 2400 MHz
    line = [ln for ln in text.splitlines() if ln.startswith("fma") and " 8 " in ln][0]
    assert abs(float(line.split()[-1]) - 64 * 2 * 84 * 1.98e9 / 1e12) < 0.01   # theory column: 64 lane-inst x 2 flop x 84 CUs x 1.98 GHz = 21.3 T


def test_verdict_thresholds():
    def mk(dot4):
        return ('RESULT mode=fma waves_per_simd=8 unit=flop ops_per_clk_cu=128 inst_per_clk_cu=64 ops_per_s=1e13 implied_clock_ghz=2 best_ops_per_clk_cu=128\n'
                'RESULT mode=dot4 waves_per_simd=8 unit=op ops_per_clk_cu=%f inst_per_clk_cu=%f ops_per_s=1e13 implied_clock_ghz=2 best_ops_per_clk_cu=1\n' % (dot4 * 8, dot4))
    assert "full rate" in V.report(mk(64), 84, 128, 128)[0]
    assert "reduced rate" in V.report(mk(32), 84, 128, 128)[0]
    assert "quarter rate or worse" in V.report(mk(16), 84, 128, 128)[0]


def test_empty_input_fails():
    text, ok = V.report("nothing here\n", 84, 128.0, 128.0)
    assert not ok and "no RESULT lines" in text


ASM = '''_Z11rate_kernelILi0EEvjPiPy:
\tv_fma_f32 v1, v1, v2, v3
\ts_endpgm
.Lfunc_end0:
_Z11rate_kernelILi3EEvjPiPy:
\tv_dual_fmac_f32 v1, v2, v3 :: v_dual_mul_f32 v4, v5, v6
\ts_endpgm
.Lfunc_end1:
_Z11rate_kernelILi5EEvjPiPy:
\tv_dot4_i32_iu8 v1, v2, v3, v1 neg_lo:[1,1,0]
\ts_endpgm
.Lfunc_end2:
_Z11rate_kernelILi6EEvjPiPy:
\tv_mul_i32_i24 v1, v2, v3
\tv_add_nc_u32 v1, v1, v4
\ts_endpgm
.Lfunc_end3:
'''


def test_isa_check_flags_missing_and_wrong_kernels():
    text, ok = V.isa_check(ASM)
    assert not ok                                              # most modes have no body; mad24 became mul + add
    lines = {ln.split()[0]: ln for ln in text.splitlines()}
    assert "ok" in lines["fma"] and "ok" in lines["dot4"] and "ok" in lines["vopd"]
    assert "NOT EMITTED" in lines["mad24"]
    assert "MISSING" in lines["add"]


def test_isa_check_forbids_pairing_in_asm_modes():
    asm = '_Z11rate_kernelILi0EEvjPiPy:\n\tv_fma_f32 v1, v1, v2, v3\n\tv_dual_fmac_f32 v4, v5, v6 :: v_dual_mul_f32 v7, v8, v9\n.Lfunc_end0:\n'
    text, _ = V.isa_check(asm)
    assert "UNEXPECTED v_dual_" in text.splitlines()[0]


def test_cli(tmp_path, capsys):
    f = tmp_path / "out.txt"
    f.write_text(OUT, encoding="utf-8")
    assert V.main([str(f)]) == 0
    assert "VERDICT dot4" in capsys.readouterr().out
    g = tmp_path / "x.s"
    g.write_text(ASM, encoding="utf-8")
    assert V.main(["--isa", str(g)]) == 1
