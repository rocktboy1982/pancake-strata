"""Tests of tools/pancake/flag_diag.py with fake hardware (a fake sysfs / proc tree and fake rocminfo) and fake A/B results.
Run: python -m pytest -q tools/pancake/test_flag_diag.py"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import flag_diag as fd  # noqa: E402

ROCMINFO = "Agent 2\n  Name:                    gfx1100\n  Marketing Name:          AMD Radeon RX 7900 XT\n"


def fake_root(vram=20 * 2**30, hipblaslt=True, link=True, rocm="10.0.0-1"):
    root = Path(tempfile.mkdtemp())
    dev = root / "sys/class/drm/card1/device"
    dev.mkdir(parents=True)
    (dev / "vendor").write_text("0x1002\n")
    (dev / "mem_info_vram_total").write_text(f"{vram}\n")
    if link:
        (dev / "current_link_speed").write_text("16.0 GT/s PCIe\n")
        (dev / "current_link_width").write_text("16\n")
    (root / "opt/rocm-10.0/.info").mkdir(parents=True)
    (root / "opt/rocm-10.0/.info/version").write_text(rocm + "\n")
    if hipblaslt:
        (root / "opt/rocm-10.0/lib").mkdir(parents=True)
        (root / "opt/rocm-10.0/lib/libhipblaslt.so.1").write_text("x")
    (root / "dev").mkdir()
    (root / "dev/kfd").write_text("")
    (root / "proc").mkdir()
    (root / "proc/cpuinfo").write_text("model name\t: AMD Ryzen 7 5800X\n")
    (root / "proc/meminfo").write_text("MemTotal:       65536000 kB\n")
    return str(root)


def row(arm, case, dec, pr=1000.0):
    return {"arm": arm, "rep": 0, "rows": [{"case": case, "decode_tps": dec, "prompt_tps": pr}], "texts": {case: "h1"}}


def jsonl(*recs):
    return "\n".join(json.dumps(r) for r in recs)


class Detect(unittest.TestCase):
    def test_gfx1100(self):
        hw = fd.detect(fake_root(), lambda cmd: ROCMINFO)
        self.assertEqual((hw["gfx"], hw["wave"], hw["hipblaslt"], hw["vram_gib"]), ("gfx1100", 32, True, 20.0))
        self.assertEqual(hw["rocm"], "10.0.0")
        self.assertIn("x16", hw["pcie_link"])
        self.assertEqual(hw["ram_gib"], 62.5)

    def test_no_card(self):
        self.assertIsNone(fd.detect(fake_root(), lambda cmd: ""))

    def test_supported(self):
        hw = fd.detect(fake_root(hipblaslt=False, link=False), lambda cmd: ROCMINFO)
        by = {c["name"]: c for c in fd.CANDIDATES}
        self.assertFalse(fd.supported(by["STRATA_HIPBLASLT_TUNING"], hw)[0])
        self.assertFalse(fd.supported(by["--pcie-frac"], hw)[0])
        self.assertTrue(fd.supported(by["STRATA_RDNA_Q6K"], hw)[0])
        hw["gfx"], hw["wave"] = "gfx906", 64
        self.assertIn("gfx11", fd.supported(by["STRATA_RDNA_Q6K"], hw)[1])
        self.assertFalse(fd.supported(by["STRATA_HIP_WMMA"], hw)[0])


class Classify(unittest.TestCase):
    def arms(self, cand_dec, texts="same", again=100.5, again_texts=None):
        t = json.dumps({"short": "h1"}, sort_keys=True)
        return {"base": {"decode": 100.0, "prompt": 1000.0, "texts": t, "ok": True},
                "base-again": {"decode": again, "prompt": 1000.0, "texts": again_texts or t, "ok": True},
                "F": {"decode": cand_dec, "prompt": 1000.0, "texts": t if texts == "same" else "other", "ok": True}}

    def flag(self, exact=True, gate=None, metric="decode"):
        return {"name": "F", "kind": "env", "value": "1", "exact": exact, "metric": metric, "ledger": "T-1", "quality_gate": gate}

    def test_exact_gain(self):
        self.assertEqual(fd.classify(self.flag(), self.arms(103.0))["verdict"], "GAIN")

    def test_exact_below_bar(self):
        self.assertEqual(fd.classify(self.flag(), self.arms(101.5))["verdict"], "NO-GAIN")

    def test_inside_noise(self):
        self.assertEqual(fd.classify(self.flag(), self.arms(102.5, again=104.0))["verdict"], "NO-GAIN")

    def test_exact_text_differs_never_unsupported(self):
        # owner order 2026-11-20: the default differs from itself on this model; then a claimed-exact flag keeps its speed verdict, EXACT-UNVERIFIED
        v = fd.classify(self.flag(), self.arms(120.0, texts="other", again_texts="third"))
        self.assertEqual((v["verdict"], v["exact"]), ("GAIN", True))
        self.assertEqual(v["exactness"], "EXACT-UNVERIFIED (base vs base also differs)")
        # the default's repeat is identical: a real difference, judged as lossy (lossy bar, needs a gate), still not UNSUPPORTED
        v = fd.classify(self.flag(), self.arms(120.0, texts="other"))
        self.assertEqual((v["verdict"], v["exact"]), ("GAIN", False))
        self.assertIn("treated as lossy", v["exactness"])
        self.assertIn("quality gate", v["reason"])
        self.assertEqual(fd.classify(self.flag(), self.arms(102.5, texts="other"))["verdict"], "NO-GAIN")   # 2.5 % < the lossy bar

    def test_parity_ctest_is_the_judge(self):
        c = self.flag()
        c["ctest"] = "^x$"
        v = fd.classify(c, self.arms(120.0, texts="other", again_texts="third"))
        self.assertEqual(fd.apply_ctest(v, False, c)["verdict"], "FAIL")
        ok = fd.apply_ctest(v, True, c)
        self.assertEqual((ok["verdict"], ok["parity_ctest"]), ("GAIN", "PASS"))
        self.assertTrue(ok["exactness"].startswith("EXACT (parity ctest PASS"))

    def test_lossy_bar_is_3(self):
        self.assertEqual(fd.classify(self.flag(exact=False), self.arms(102.5))["verdict"], "NO-GAIN")
        self.assertEqual(fd.classify(self.flag(exact=False, gate="PASS"), self.arms(103.5, texts="other"))["verdict"], "GAIN")

    def test_did_not_start(self):
        a = self.arms(0)
        a["F"]["ok"] = False
        self.assertEqual(fd.classify(self.flag(), a)["verdict"], "UNSUPPORTED")
        del a["F"]
        self.assertEqual(fd.classify(self.flag(), a)["verdict"], "UNSUPPORTED")


class Run(unittest.TestCase):
    def test_text_differs_runs_the_default_again_once_and_the_ctests(self):
        hw = fd.detect(fake_root(), lambda cmd: ROCMINFO)
        out = Path(tempfile.mkdtemp())
        arms, cmds = [], []

        def run(cmd, log, timeout_s):
            if cmd[0] == "ctest":
                cmds.append(cmd)
                return (0 if "rdna_gr" in cmd[cmd.index("-R") + 1] else 8), False
            spec = json.loads(Path(cmd[cmd.index("--tier") - 1]).read_text())
            arm = next(iter(spec["arms"]))
            arms.append(arm)
            text = {"base": "h1"}.get(arm, "h-" + arm)     # every start gives other text, the default too
            rec = json.loads(jsonl(row(arm, "short", {"base": 100.0, "base-again": 100.5}.get(arm, 110.0))))
            rec["texts"] = {"short": text}
            Path(cmd[cmd.index("--out") + 1]).write_text(json.dumps(rec) + "\n")
            return 0, False
        res = fd.run("cfg.json", out, ["STRATA_RDNA_Q6K", "STRATA_GR_RDNA"], 60.0, Path("abtest.py"), runner=run, hw=hw, ctest="/b")
        self.assertEqual(arms, ["base", "STRATA_RDNA_Q6K", "base-again", "STRATA_GR_RDNA"])
        v = {x["name"]: x for x in res["verdicts"]}
        self.assertEqual((v["STRATA_GR_RDNA"]["verdict"], v["STRATA_GR_RDNA"]["parity_ctest"]), ("GAIN", "PASS"))
        self.assertEqual((v["STRATA_RDNA_Q6K"]["verdict"], v["STRATA_RDNA_Q6K"]["parity_ctest"]), ("FAIL", "FAIL"))
        self.assertEqual(len(cmds), 2)
        self.assertTrue(fd.ctest_not_run("7/7 Test #154: pool_stress ....***Not Run   0.00 sec"))
        self.assertFalse(fd.ctest_not_run("1/1 Test #3: rdna_gr_parity .... Failed"))

    def test_parse_and_run(self):
        hw = fd.detect(fake_root(), lambda cmd: ROCMINFO)
        out = Path(tempfile.mkdtemp())

        res = fd.run("cfg.json", out, ["STRATA_RDNA_Q6K", "STRATA_HIPBLASLT_TUNING", "--adapt-every", "STRATA_PF_F16"],
                     60.0, Path("abtest.py"), runner=self.runner({"base": 100.0, "STRATA_RDNA_Q6K": 110.0}), hw=hw)
        v = {x["name"]: x for x in res["verdicts"]}
        self.assertEqual(v["STRATA_RDNA_Q6K"]["verdict"], "GAIN")
        self.assertEqual(v["--adapt-every"]["verdict"], "NO-GAIN")
        self.assertEqual(v["STRATA_PF_F16"]["verdict"], "NO-GAIN")
        self.assertTrue((out / "verdicts.json").exists())
        self.assertTrue(res["complete"])

    def runner(self, dec, timeouts=(), refuse=(), seen=None, out=None):
        """A fake abtest run of ONE arm: the spec has one arm, case short; writes that arm's jsonl."""
        def run(cmd, log, timeout_s):
            spec = json.loads(Path(cmd[cmd.index("--tier") - 1]).read_text())
            self.assertEqual(len(spec["arms"]), 1)
            self.assertEqual(spec["cases"], ["short"])
            self.assertEqual(cmd[cmd.index("--retries") + 1], "0")
            arm = next(iter(spec["arms"]))
            if seen is not None:
                seen.append((arm, timeout_s, json.loads((Path(log).parent / "verdicts.json").read_text())))
            if arm in timeouts:
                return None, True
            if arm in refuse:
                return 3, False
            Path(cmd[cmd.index("--out") + 1]).write_text(jsonl(row(arm, "short", dec.get(arm, 100.5))))
            return 0, False
        return run

    def test_budget_and_unsupported(self):
        hw = fd.detect(fake_root(hipblaslt=False), lambda cmd: ROCMINFO)
        out = Path(tempfile.mkdtemp())
        res = fd.run("cfg.json", out, None, 6.0, Path("abtest.py"), runner=lambda c, o, t: (1, False), hw=hw)
        v = {x["name"]: x for x in res["verdicts"]}
        self.assertEqual(v["STRATA_HIPBLASLT_TUNING"]["verdict"], "UNSUPPORTED")
        self.assertGreater(sum(1 for x in res["verdicts"] if x["verdict"] == "UNTESTED"), 0)

    def test_per_flag_timeout_and_incremental_verdicts(self):
        """2026-11-19: a flag that does not answer within the per-flag limit is UNTESTED-TIMEOUT and the next flags still run;
        verdicts.json grows arm by arm (base once, one start per flag)."""
        hw = fd.detect(fake_root(), lambda cmd: ROCMINFO)
        out, seen = Path(tempfile.mkdtemp()), []
        res = fd.run("cfg.json", out, ["STRATA_RDNA_Q6K", "STRATA_GR_RDNA", "STRATA_COMBINE_CPU"], 60.0, Path("abtest.py"),
                     runner=self.runner({"base": 100.0, "STRATA_RDNA_Q6K": 103.0}, timeouts=("STRATA_GR_RDNA",), seen=seen),
                     hw=hw, flag_timeout=240.0)
        v = {x["name"]: x for x in res["verdicts"]}
        self.assertEqual(v["STRATA_GR_RDNA"]["verdict"], "UNTESTED-TIMEOUT")
        self.assertEqual(v["STRATA_RDNA_Q6K"]["verdict"], "GAIN")
        self.assertEqual(v["STRATA_COMBINE_CPU"]["verdict"], "NO-GAIN")
        self.assertEqual([a for a, _, _ in seen], ["base", "STRATA_RDNA_Q6K", "STRATA_GR_RDNA", "STRATA_COMBINE_CPU"])
        self.assertTrue(all(t == 240.0 for _, t, _ in seen))
        first = {x["name"]: x["verdict"] for x in seen[0][2]["verdicts"]}
        self.assertEqual(first["STRATA_RDNA_Q6K"], "UNTESTED")
        self.assertFalse(seen[0][2]["complete"])
        third = {x["name"]: x["verdict"] for x in seen[3][2]["verdicts"]}
        self.assertEqual((third["STRATA_RDNA_Q6K"], third["STRATA_GR_RDNA"]), ("GAIN", "UNTESTED-TIMEOUT"))

    def test_base_timeout_and_busy_harness(self):
        hw = fd.detect(fake_root(), lambda cmd: ROCMINFO)
        res = fd.run("cfg.json", Path(tempfile.mkdtemp()), ["STRATA_RDNA_Q6K"], 60.0, Path("abtest.py"),
                     runner=self.runner({}, timeouts=("base",)), hw=hw)
        self.assertEqual(res["verdicts"][0]["verdict"], "UNTESTED-TIMEOUT")
        res = fd.run("cfg.json", Path(tempfile.mkdtemp()), ["STRATA_RDNA_Q6K"], 60.0, Path("abtest.py"),
                     runner=self.runner({}, refuse=("base",)), hw=hw)
        self.assertEqual(res["verdicts"][0]["verdict"], "UNTESTED")
        self.assertIn("exited 3", res["verdicts"][0]["reason"])

    def test_stopped_keeps_partial_verdicts(self):
        hw = fd.detect(fake_root(), lambda cmd: ROCMINFO)
        out = Path(tempfile.mkdtemp())
        inner = self.runner({"base": 100.0, "STRATA_RDNA_Q6K": 103.0})

        def run(cmd, log, t):
            if "spec-STRATA_GR_RDNA.json" in " ".join(cmd):
                raise fd.Stopped()
            return inner(cmd, log, t)
        with self.assertRaises(fd.Stopped):
            fd.run("cfg.json", out, ["STRATA_RDNA_Q6K", "STRATA_GR_RDNA", "STRATA_COMBINE_CPU"], 60.0, Path("abtest.py"), runner=run, hw=hw)
        v = {x["name"]: x["verdict"] for x in json.loads((out / "verdicts.json").read_text())["verdicts"]}
        self.assertEqual(v, {"STRATA_RDNA_Q6K": "GAIN", "STRATA_GR_RDNA": "UNTESTED-TIMEOUT", "STRATA_COMBINE_CPU": "UNTESTED-TIMEOUT"})

    def test_budget_counts_the_per_flag_limit(self):
        hw = fd.detect(fake_root(), lambda cmd: ROCMINFO)
        res = fd.run("cfg.json", Path(tempfile.mkdtemp()), ["STRATA_RDNA_Q6K", "STRATA_GR_RDNA", "STRATA_COMBINE_CPU"], 12.0,
                     Path("abtest.py"), runner=self.runner({"base": 100.0}), hw=hw, flag_timeout=240.0)
        v = {x["name"]: x["verdict"] for x in res["verdicts"]}
        self.assertEqual(sum(1 for x in v.values() if x == "UNTESTED"), 1)   # 12 min = base + 2 flags of 4 min

    def test_diag_group_reap_kills_only_the_group_leftovers(self):
        import diag_group
        alive = {101: "python3 abtest.py", 102: "python3 serve/server.py --engine strata"}
        killed, now = [], [0.0]

        def kill(pid, sig):
            killed.append(pid)
            alive.pop(pid, None)

        def sleep(s):
            now[0] += s
        out = diag_group.reap(77, wait=60, scan=lambda pg: sorted(alive.items()) if pg == 77 else [], kill=kill, sleep=sleep,
                              clock=lambda: now[0])
        self.assertEqual(killed, [101, 102])
        self.assertEqual([p for p, _ in out], [101, 102])
        self.assertEqual(diag_group.reap(78, scan=lambda pg: [], kill=kill, sleep=sleep, clock=lambda: now[0]), [])

    @unittest.skipUnless(sys.platform.startswith("linux"), "process groups and /proc: Linux")
    def test_diag_group_timeout_stops_the_whole_group(self):
        import diag_group
        import time as _t
        t0 = _t.time()
        rc, to, _ = diag_group.run(["sh", "-c", "sleep 300 & sleep 300"], Path(tempfile.mkdtemp()) / "log", 1, kill_after=2, orphan_wait=5)
        self.assertTrue(to)
        self.assertIsNone(rc)
        self.assertLess(_t.time() - t0, 30)


class Recommended(unittest.TestCase):
    def result(self):
        hw = {"gfx": "gfx1100", "rocm": "10.0.0", "vram_gib": 20.0}

        def mk(n, verdict, exact, gain, gate=None):
            return {"name": n, "kind": "env", "value": "1", "verdict": verdict, "exact": exact, "gain_pct": gain,
                    "quality_gate": gate, "ledger": "T"}
        return {"hardware": hw, "verdicts": [
            mk("A_EXACT", "GAIN", True, 2.5), mk("B_LOSSY_PASS", "GAIN", False, 3.2, "PASS"), mk("C_LOSSY_NOGATE", "GAIN", False, 9.0),
            mk("D_NOGAIN", "NO-GAIN", True, 0.4), mk("E_UNSUP", "UNSUPPORTED", True, None), mk("F_LOW", "GAIN", True, 1.0),
            mk("G_LOSSY_LOW", "GAIN", False, 2.9, "PASS")]}

    def test_only_accepted_enter(self):
        prof = fd.recommended(self.result())
        self.assertEqual([f["name"] for f in prof["flags"]], ["A_EXACT", "B_LOSSY_PASS"])
        self.assertEqual(prof["match"]["gfx"], "gfx1100")
        self.assertEqual(prof["match"]["rocm"], ["10.0."])

    def test_write_and_merge(self):
        f = Path(tempfile.mkdtemp()) / "t.json"
        fd.write_recommended(self.result(), f)
        fd.write_recommended(self.result(), f, merge=True)
        doc = json.loads(f.read_text())
        self.assertEqual(len(doc["profiles"]), 1)
        self.assertEqual(doc["schema"], "strata.tuned-flags/1")


class Groups(unittest.TestCase):
    def test_spec_arm_carries_the_extra_switches(self):
        by = {c["name"]: c for c in fd.CANDIDATES}
        spec = fd.make_spec("cfg.json", [by["STRATA_VERIFY_ECAP"], by["STRATA_GR_FUSED"], by["--pcie-frac"]], ["short"])
        self.assertEqual(spec["arms"]["STRATA_VERIFY_ECAP"]["env"], {"STRATA_VERIFY_ECAP": "0", "STRATA_VERIFY_ECAP_MIN_ROWS": "9"})
        self.assertEqual(spec["arms"]["STRATA_GR_FUSED"]["env"]["STRATA_GR_INJ_UP"], "1")
        self.assertEqual(spec["arms"]["--pcie-frac"], {"args": {"--pcie-frac": "0"}})

    def test_recommended_expands_the_extra_switches_of_an_accepted_group(self):
        v = {"name": "STRATA_VERIFY_ECAP", "kind": "env", "value": "0", "verdict": "GAIN", "exact": False, "gain_pct": 9.0,
             "quality_gate": "PASS", "ledger": "T-588", "extra": {"STRATA_VERIFY_ECAP_MIN_ROWS": "9"}}
        bad = dict(v, name="STRATA_X", verdict="NO-GAIN", extra={"STRATA_Y": "1"})
        prof = fd.recommended({"hardware": {"gfx": "gfx1100", "rocm": "10.0.0", "vram_gib": 20.0}, "verdicts": [v, bad]})
        self.assertEqual([(f["name"], f["value"]) for f in prof["flags"]],
                         [("STRATA_VERIFY_ECAP", "0"), ("STRATA_VERIFY_ECAP_MIN_ROWS", "9")])

    def test_every_candidate_is_unique_and_the_ported_switches_are_listed(self):
        names = [c["name"] for c in fd.CANDIDATES]
        self.assertEqual(len(names), len(set(names)))
        for n in ("STRATA_GR_RDNA", "STRATA_GR_RDNA_Q8", "STRATA_GR_FUSED", "STRATA_RDNA_MMVQ_WAVE", "STRATA_COMBINE_CPU",
                  "STRATA_ROUTE_PUBLISH", "STRATA_POOL_FUSE_PHASES", "STRATA_VERIFY_ECAP", "STRATA_RDNA_Q6K"):
            self.assertIn(n, names)


if __name__ == "__main__":
    unittest.main()
