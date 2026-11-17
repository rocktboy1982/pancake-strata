"""Tests for ablate_expdec.py on small synthetic expdec_bench outputs."""
import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ablate_expdec as A  # noqa: E402

DEF = "# device 0: x\n0 1 real 16/42 dec 100.0 50.0 40.0 300.0 60.0% identical\n0 4 real 16/42 dec 200.0 120.0 60.0 250.0 50.0% identical\n"
NOMAC = "# STRATA_KERNEL_ABLATE=1: the gate/up kernel is an ablation build\n0 1 real 16/42 dec 90.0 40.0 40.0 300.0 60.0% DIFFERENT\n0 4 real 16/42 dec 150.0 90.0 60.0 250.0 50.0% DIFFERENT\n"
SAME = "# STRATA_KERNEL_ABLATE=2: the gate/up kernel is an ablation build\n0 1 real 16/42 dec 70.0 20.0 40.0 300.0 60.0% DIFFERENT\n"


def test_parse_and_report():
    d = A.parse(DEF)
    assert d[(0, 1, "real", "16/42", "dec")] == 50.0 and len(d) == 2
    rows = A.report(d, A.parse(NOMAC), A.parse(SAME))
    assert len(rows) == 1                                          # only the cell present in all three
    k, gd, ga, gb, sm, sd = rows[0]
    assert (gd, ga, gb) == (50.0, 40.0, 20.0) and abs(sm - 0.2) < 1e-9 and abs(sd - 0.6) < 1e-9
    assert "share_dram" in A.format_rows(rows)


def test_macro_detection_and_cli_check(tmp_path):
    assert A.ablate_macro(DEF) == 0 and A.ablate_macro(NOMAC) == 1 and A.ablate_macro(SAME) == 2
    for n, t in (("d", DEF), ("n", NOMAC), ("s", SAME)):
        (tmp_path / n).write_text(t)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert A.main(["report", str(tmp_path / "d"), str(tmp_path / "n"), str(tmp_path / "s")]) == 0
    assert "20.0%" in out.getvalue() and "60.0%" in out.getvalue()
    with contextlib.redirect_stderr(io.StringIO()):
        assert A.main(["report", str(tmp_path / "n"), str(tmp_path / "d"), str(tmp_path / "s")]) == 2    # files in the wrong order
