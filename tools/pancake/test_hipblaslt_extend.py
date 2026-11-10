"""Tests for hipblaslt_extend.py (no GPU).  Run: python tools/pancake/test_hipblaslt_extend.py"""
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hipblaslt_extend as X  # noqa: E402

SHIPPED = HERE.parent / "hip" / "gfx1100-hipblaslt-100401.txt"


def write(tmp, name, text):
    p = Path(tmp) / name
    p.write_text(text)
    return p


class Extend(unittest.TestCase):
    def test_shipped_table_loads_and_plans(self):
        arch, version, rows = X.load_table(SHIPPED)
        self.assertEqual((arch, version), ("gfx1100", 100401))
        cases, skipped = X.plan(rows, X.DEFAULT_TARGETS)
        self.assertIn(("f16", 1280, 2560, 1280), skipped)      # decode-sized buckets (8..999)
        self.assertIn(("bf16", 2048, 48, 2560, 96), cases)
        self.assertIn(("f16", 32768, 12288, 2560, 12288), cases)
        # 10240x2560 bf16 already has 1024 and 2048
        self.assertNotIn(("bf16", 1024, 10240, 2560, 10240), cases)
        self.assertIn(("bf16", 3072, 10240, 2560, 10240), cases)
        self.assertFalse(any(c[1] not in X.DEFAULT_TARGETS for c in cases))

    def test_command_matches_tuner_case_syntax(self):
        cmd = X.tuner_command("t", [("bf16", 1024, 48, 2560, 96)], "o.txt", 64)
        self.assertEqual(cmd, ["t", "--workspace-mib", "64", "--case", "bf16,1024,48,2560,96", "--tuning-out", "o.txt"])

    def test_merge_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = write(tmp, "base.txt", "STRATA_HIPBLASLT_TUNING_V1 gfx1100 100401\nbf16 48 2560 96 4096 1035\n")
            tuned = write(tmp, "tuned.txt", "# c\nSTRATA_HIPBLASLT_TUNING_V1 gfx1100 100401\n"
                                             "bf16 48 2560 96 1024 1034\nbf16 48 2560 96 4096 1036\n")
            out = Path(tmp) / "out.txt"
            self.assertEqual(X.main(["merge", "--table", str(base), "--tuned", str(tuned), "--out", str(out)]), 0)
            _, _, rows = X.load_table(out)
            self.assertEqual(rows[("bf16", 48, 2560, 96, 4096)], 1035)   # base kept without --replace
            self.assertEqual(rows[("bf16", 48, 2560, 96, 1024)], 1034)
            X.main(["merge", "--table", str(base), "--tuned", str(tuned), "--out", str(out), "--replace"])
            self.assertEqual(X.load_table(out)[2][("bf16", 48, 2560, 96, 4096)], 1036)

    def test_merge_refuses_other_version(self):
        base = ("gfx1100", 100401, {})
        with self.assertRaises(X.TableError):
            X.merge(base, ("gfx1100", 100500, {}))

    def test_fallback_log_adds_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = write(tmp, "e.log", "prefill gemm: fallback shape dtype=bf16 T=1500 N=96 K=2560 ldy=96\n"
                                      "prefill gemm: Lt fallback; no calibration for dtype=f16 T=7 N=64 K=64 ldy=64\n")
            found = X.fallback_shapes(log)
            self.assertEqual(found, {("bf16", 96, 2560, 96), ("f16", 64, 64, 64)})
            cases, _ = X.plan({}, (1024,), extra_shapes=found)
            self.assertEqual(sorted(cases), [("bf16", 1024, 96, 2560, 96), ("f16", 1024, 64, 64, 64)])

    def test_bad_rows_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for text in ("bf16 1 1 1 1 1\n", "STRATA_HIPBLASLT_TUNING_V1 g 1\nbf16 4 1 2 1 1\n",
                         "STRATA_HIPBLASLT_TUNING_V1 g 1\nbf16 4 1 4 1 1\nbf16 4 1 4 1 2\n"):
                with self.assertRaises(X.TableError):
                    X.load_table(write(tmp, "t.txt", text))


if __name__ == "__main__":
    unittest.main()
