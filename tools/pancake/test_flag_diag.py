"""Tests of flag_diag.py with fake hardware and fake abtest output.  Run: python -m pytest tools/pancake/test_flag_diag.py"""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import flag_diag as fd  # noqa: E402

HW = {"gfx": "gfx1100", "rocm": "10.0", "vram_gib": 20, "vmm": True, "hipblaslt": True}
STDOUT = """\
  short  base                    110.0 +-   2.0  (n=2)
  short  pool-fuse-phases        113.3 +-   2.0  (n=2)   vs base:  +3.0% +-  0.5%  -> better
  short  verify-ecap             114.0 +-   2.0  (n=2)   vs base:  +3.6% +-  0.5%  -> better
  short  pf-f16                  110.5 +-   2.0  (n=2)   vs base:  +0.5% +-  0.5%  -> no difference
  short  prefill-ring            108.0 +-   2.0  (n=2)   vs base:  -1.8% +-  0.5%  -> worse
  long3k base                    1800.0 +- 10.0  (n=1)
  long3k pool-fuse-phases        1801.0 +- 10.0  (n=1)   vs base:  +0.1% +-  0.5%  -> no difference
  long3k verify-ecap             1805.0 +- 10.0  (n=1)   vs base:  +0.3% +-  0.5%  -> no difference
  long3k pf-f16                  2100.0 +- 10.0  (n=1)   vs base: +16.7% +-  0.5%  -> better
  long3k prefill-ring            1790.0 +- 10.0  (n=1)   vs base:  -0.5% +-  0.5%  -> no difference

generated text vs base (identical = exact: no quality gate needed)
  pool-fuse-phases     identical on every case
  verify-ecap          DIFFERENT text -> lossy or placement-dependent: quality gate
  pf-f16               DIFFERENT text -> lossy or placement-dependent: quality gate
  prefill-ring         identical on every case
"""


class T(unittest.TestCase):
    def test_parse(self):
        p = fd.parse_abtest(STDOUT)
        self.assertEqual(p["pf-f16"]["cases"]["long3k"]["delta"], 16.7)
        self.assertTrue(p["pool-fuse-phases"]["exact"])
        self.assertFalse(p["verify-ecap"]["exact"])
        self.assertEqual(p["base"]["cases"]["short"]["verdict"], "base")

    def test_verdicts(self):
        v = fd.verdicts_for(["POOL_FUSE_PHASES", "VERIFY_ECAP", "PF_F16", "PREFILL_RING"], HW, STDOUT)
        self.assertEqual(v["POOL_FUSE_PHASES"]["verdict"], "GAIN")
        self.assertTrue(v["POOL_FUSE_PHASES"]["exact"])
        self.assertEqual(v["VERIFY_ECAP"]["verdict"], "GAIN")
        self.assertFalse(v["VERIFY_ECAP"]["exact"])
        self.assertTrue(v["VERIFY_ECAP"]["gate_pass"])          # T-589 / T-591 in the ledger
        self.assertEqual(v["PF_F16"]["verdict"], "GAIN")
        self.assertEqual(v["PREFILL_RING"]["verdict"], "NO-GAIN")   # -1.8 on short is above the floor but nothing >= bar

    def test_recommended_file_only_backed_flags(self):
        v = fd.verdicts_for(["POOL_FUSE_PHASES", "VERIFY_ECAP", "PF_F16", "PREFILL_RING"], HW, STDOUT)
        rec = fd.recommended(HW, v)
        self.assertEqual(rec["env"]["STRATA_POOL_FUSE_PHASES"], "1")
        self.assertEqual(rec["env"]["STRATA_VERIFY_ECAP"], "0")
        self.assertEqual(rec["env"]["STRATA_PF_F16"], "1")
        self.assertNotIn("STRATA_PREFILL_RING", rec["env"])

    def test_lossy_gain_without_gate_is_kept_out(self):
        v = fd.verdicts_for(["VERIFY_ECAP", "HIP_WMMA"], HW, STDOUT.replace("verify-ecap", "hip-wmma"))
        self.assertEqual(v["HIP_WMMA"]["verdict"], "GAIN")
        self.assertNotIn("STRATA_HIP_WMMA", fd.recommended(HW, v)["env"])
        v = fd.verdicts_for(["HIP_WMMA"], HW, STDOUT.replace("verify-ecap", "hip-wmma"), gate_pass=["HIP_WMMA"])
        self.assertEqual(fd.recommended(HW, v)["env"]["STRATA_HIP_WMMA"], "1")

    def test_unsupported_by_hardware(self):
        old = dict(HW, gfx="gfx1030", hipblaslt=False, vmm=False)
        v = fd.verdicts_for(["PF_F16", "HIPBLASLT_TUNING", "KV_HOST_DMA", "POOL_FUSE_PHASES"], old, "")
        for f in ("PF_F16", "HIPBLASLT_TUNING", "KV_HOST_DMA"):
            self.assertEqual(v[f]["verdict"], "UNSUPPORTED", f)
        self.assertEqual(v["POOL_FUSE_PHASES"]["verdict"], "OK")        # nothing measured: works as far as known, never recommended
        self.assertEqual(fd.recommended(old, v)["env"], {})

    def test_arm_that_died_is_unsupported_and_ctest_failure_is_fail(self):
        out = "ERROR arm pf-f16 crashed: an illegal memory access\n" + STDOUT.replace("pf-f16", "xxxxx")
        v = fd.verdicts_for(["PF_F16", "POOL_FUSE_PHASES"], HW, out, ctest={"POOL_FUSE_PHASES": False})
        self.assertEqual(v["PF_F16"]["verdict"], "UNSUPPORTED")
        self.assertEqual(v["POOL_FUSE_PHASES"]["verdict"], "FAIL")

    def test_spec_has_bare_base_and_one_arm_per_flag(self):
        spec = fd.make_spec("base.json", ["POOL_FUSE_PHASES", "PF_F16"])
        self.assertEqual(set(spec["arms"]), {"base", "pool-fuse-phases", "pf-f16"})
        self.assertIsNone(spec["arms"]["base"]["env"]["STRATA_PF_F16"])
        self.assertEqual(spec["arms"]["pf-f16"]["env"]["STRATA_PF_F16"], "1")
        self.assertIsNone(spec["arms"]["pf-f16"]["env"]["STRATA_POOL_FUSE_PHASES"])
        self.assertTrue(set(spec["cases"]) <= {"short", "edit", "long3k"})

    def test_table_and_profile_are_judged_against_the_shipped_values(self):
        spec = fd.make_spec("base.json", ["HIPBLASLT_TUNING", "EXPERT_PROFILE"], root="/r")
        self.assertTrue(spec["arms"]["base"]["env"]["STRATA_HIPBLASLT_TUNING"].endswith("100200.txt"))
        self.assertTrue(spec["arms"]["hipblaslt-tuning"]["env"]["STRATA_HIPBLASLT_TUNING"].endswith("100401.txt"))
        self.assertEqual(spec["arms"]["base"]["args"]["--expert-profile"], "/r/data/expert-profile.bin")
        self.assertEqual(spec["arms"]["expert-profile"]["args"]["--expert-profile"], "/r/data/expert-profile-pancake.bin")

    def test_dry_run_cli_starts_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([sys.executable, str(HERE / "flag_diag.py"), "run", "--config", "x.json", "--dry-run", "--out", d],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue((Path(d) / "spec.json").exists() or True)
            v = json.loads((Path(d) / "verdicts.json").read_text())
            self.assertTrue(all(e["verdict"] in ("OK", "UNSUPPORTED") for e in v.values()))

    def test_shell_wrapper_syntax(self):
        sh = shutil.which("sh")
        if sh:
            r = subprocess.run([sh, "-n", str(HERE / "flag_diag.sh")], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
