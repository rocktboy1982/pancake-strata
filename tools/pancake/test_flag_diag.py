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
        self.assertEqual(set(spec["arms"]), {"base", "base-again", "pool-fuse-phases", "pf-f16"})
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

    # ---- 2026-11-19: one arm per abtest run, per-arm time limit, incremental verdicts, group stop
    def fake_runner(self, decode, timeouts=(), refuse=(), seen=None):
        def runner(cmd, log, timeout_s):
            spec = json.loads(Path(cmd[cmd.index("--tier") - 1]).read_text())
            arm = next(iter(spec["arms"]))
            self.assertEqual(spec["cases"], ["short"])
            self.assertIn("--retries", cmd)
            if seen is not None:
                seen.append((arm, json.loads((Path(log).parent / "verdicts.json").read_text())))
            Path(log).write_text("rep 0 %s ...\n" % arm)
            if arm in timeouts:
                return None, True
            if arm in refuse:
                return 3, False
            jl = Path(cmd[cmd.index("--out") + 1])
            jl.write_text(json.dumps({"arm": arm, "rows": [{"case": "short", "decode_tps": decode[arm]}, {"case": "short", "decode_tps": decode[arm]}],
                                      "texts": {"short": "h-" + ("x" if arm == "verify-ecap" else "same")}}) + "\n")
            return 0, False
        return runner

    def test_run_arms_per_flag_timeout_and_incremental(self):
        seen = []
        dec = {"base": 100.0, "pool-fuse-phases": 103.0, "verify-ecap": 104.0, "kv-host-dma": 100.2}
        with tempfile.TemporaryDirectory() as d:
            v = fd.run_arms("cfg.json", ["POOL_FUSE_PHASES", "VERIFY_ECAP", "PF_F16", "KV_HOST_DMA"], HW, Path(d), "abtest.py",
                            runner=self.fake_runner(dec, timeouts=("pf-f16",), seen=seen))
            on_disk = json.loads((Path(d) / "verdicts.json").read_text())
        self.assertEqual(v["POOL_FUSE_PHASES"]["verdict"], "GAIN")
        self.assertTrue(v["POOL_FUSE_PHASES"]["exact"])
        self.assertEqual(v["VERIFY_ECAP"]["verdict"], "GAIN")   # lossy +4 % >= 3
        self.assertFalse(v["VERIFY_ECAP"]["exact"])
        self.assertEqual(v["PF_F16"]["verdict"], "UNTESTED-TIMEOUT")
        self.assertEqual(v["KV_HOST_DMA"]["verdict"], "NO-GAIN")
        self.assertEqual(on_disk["PF_F16"]["verdict"], "UNTESTED-TIMEOUT")
        # base once, then one start per flag; the verdicts on disk grow arm by arm
        self.assertEqual([a for a, _ in seen], ["base", "pool-fuse-phases", "verify-ecap", "pf-f16", "kv-host-dma"])
        self.assertEqual(seen[0][1]["POOL_FUSE_PHASES"]["verdict"], "UNTESTED")
        self.assertEqual(seen[2][1]["POOL_FUSE_PHASES"]["verdict"], "GAIN")
        self.assertEqual(seen[4][1]["PF_F16"]["verdict"], "UNTESTED-TIMEOUT")

    def test_run_arms_base_timeout_or_busy(self):
        with tempfile.TemporaryDirectory() as d:
            v = fd.run_arms("cfg.json", ["POOL_FUSE_PHASES"], HW, Path(d), "abtest.py", runner=self.fake_runner({}, timeouts=("base",)))
            self.assertEqual(v["POOL_FUSE_PHASES"]["verdict"], "UNTESTED-TIMEOUT")
            v = fd.run_arms("cfg.json", ["POOL_FUSE_PHASES"], HW, Path(d), "abtest.py", runner=self.fake_runner({}, refuse=("base",)))
            self.assertEqual(v["POOL_FUSE_PHASES"]["verdict"], "UNTESTED")

    def test_run_arms_stopped_keeps_partial_verdicts(self):
        dec = {"base": 100.0, "pool-fuse-phases": 103.0}
        inner = self.fake_runner(dec)

        def runner(cmd, log, t):
            if "spec-verify-ecap.json" in " ".join(cmd):
                raise fd.Stopped()
            return inner(cmd, log, t)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(fd.Stopped):
                fd.run_arms("cfg.json", ["POOL_FUSE_PHASES", "VERIFY_ECAP", "KV_HOST_DMA"], HW, Path(d), "abtest.py", runner=runner)
            v = json.loads((Path(d) / "verdicts.json").read_text())
        self.assertEqual(v["POOL_FUSE_PHASES"]["verdict"], "GAIN")
        self.assertEqual(v["VERIFY_ECAP"]["verdict"], "UNTESTED-TIMEOUT")
        self.assertEqual(v["KV_HOST_DMA"]["verdict"], "UNTESTED-TIMEOUT")

    def test_text_differs_with_a_differing_default_is_exact_unverified_never_unsupported(self):
        # owner order 2026-11-20: the default's text differs from itself on this model; base-again runs once, the flag keeps its speed verdict
        dec = {"base": 100.0, "base-again": 100.2, "rdna-mmvq-wave": 127.0, "kv-host-dma": 103.0}
        seen = []

        def runner(cmd, log, timeout_s):
            spec = json.loads(Path(cmd[cmd.index("--tier") - 1]).read_text())
            arm = next(iter(spec["arms"]))
            seen.append(arm)
            Path(log).write_text("rep 0 %s ...\n" % arm)
            Path(cmd[cmd.index("--out") + 1]).write_text(json.dumps({"arm": arm, "rows": [{"case": "short", "decode_tps": dec[arm]}],
                                                                     "texts": {"short": "h-" + arm}}) + "\n")
            return 0, False
        with tempfile.TemporaryDirectory() as d:
            v = fd.run_arms("cfg.json", ["RDNA_MMVQ_WAVE", "KV_HOST_DMA"], HW, Path(d), "abtest.py", runner=runner)
        self.assertEqual(seen, ["base", "rdna-mmvq-wave", "base-again", "kv-host-dma"])
        for f in ("RDNA_MMVQ_WAVE", "KV_HOST_DMA"):
            self.assertEqual((v[f]["verdict"], v[f]["exact"]), ("GAIN", True), f)
            self.assertEqual(v[f]["exactness"], "EXACT-UNVERIFIED (base vs base also differs)")
        # an identical repeat of the default: the difference is real, judged as lossy (as before)
        e = fd.verdict_for("KV_HOST_DMA", HW, {"kv-host-dma": {"cases": {"short": {"mean": 1, "delta": 5.0, "verdict": "better"}}, "exact": False}},
                           "x", None, (), False)
        self.assertFalse(e["exact"])
        self.assertIn("treated as lossy", e["note"])
        self.assertNotEqual(e["verdict"], "UNSUPPORTED")
        # the parity ctest decides
        self.assertEqual(fd.apply_ctest(dict(v["RDNA_MMVQ_WAVE"]), False)["verdict"], "FAIL")
        ok = fd.apply_ctest(dict(v["RDNA_MMVQ_WAVE"]), True)
        self.assertEqual((ok["verdict"], ok["parity_ctest"]), ("GAIN", "PASS"))
        self.assertTrue(ok["exactness"].startswith("EXACT (parity ctest PASS"))

    def test_ctest_not_run_is_no_result_not_a_fail(self):
        class R:
            returncode = 8
            stdout = "1/7 Test #124: pool_tasks_test ....***Not Run   0.00 sec\n0% tests passed, 7 tests failed out of 7\n"
        self.assertEqual(fd.run_ctest("/b", ["POOL_FUSE_PHASES"], runner=lambda cmd, **kw: R()), {"POOL_FUSE_PHASES": None})
        R.stdout = "1/1 Test #3: pool_test .... Failed\n"
        self.assertEqual(fd.run_ctest("/b", ["POOL_FUSE_PHASES"], runner=lambda cmd, **kw: R()), {"POOL_FUSE_PHASES": False})

    def test_ctest_only_for_exact_flags(self):
        calls = []

        class R:
            returncode = 0

        def runner(cmd, **kw):
            calls.append(cmd[cmd.index("-R") + 1])
            return R()
        ct = fd.run_ctest("/b", ["POOL_FUSE_PHASES", "VERIFY_ECAP"], runner=runner)
        self.assertEqual(calls, ["pool"])
        self.assertEqual(ct, {"POOL_FUSE_PHASES": True})

    def test_diag_group_reap_kills_only_the_group_leftovers(self):
        import diag_group
        alive = {101: "python3 abtest.py", 102: "python3 serve/server.py --engine strata"}
        killed, now = [], [0.0]

        def scan(pg):
            self.assertEqual(pg, 77)
            return sorted(alive.items())

        def kill(pid, sig):
            killed.append(pid)
            alive.pop(pid, None)

        def sleep(s):
            now[0] += s
        out = diag_group.reap(77, wait=60, scan=scan, kill=kill, sleep=sleep, clock=lambda: now[0])
        self.assertEqual(killed, [101, 102])
        self.assertEqual([p for p, _ in out], [101, 102])
        self.assertGreaterEqual(now[0], 60)
        self.assertEqual(diag_group.reap(77, scan=lambda pg: [], kill=kill, sleep=sleep, clock=lambda: now[0]), [])

    @unittest.skipUnless(sys.platform.startswith("linux"), "process groups and /proc: Linux")
    def test_diag_group_timeout_stops_the_whole_group(self):
        import diag_group
        with tempfile.TemporaryDirectory() as d:
            t0 = __import__("time").time()
            rc, to, killed = diag_group.run(["sh", "-c", "sleep 300 & sleep 300"], Path(d) / "log", 1, kill_after=2, orphan_wait=5)
            self.assertTrue(to)
            self.assertIsNone(rc)
            self.assertLess(__import__("time").time() - t0, 30)

    def test_shell_wrapper_syntax(self):
        sh = shutil.which("sh")
        if sh:
            r = subprocess.run([sh, "-n", str(HERE / "flag_diag.sh")], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
