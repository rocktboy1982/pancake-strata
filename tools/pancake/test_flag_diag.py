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
    def arms(self, cand_dec, texts="same", again=100.5):
        t = json.dumps({"short": "h1"}, sort_keys=True)
        return {"base": {"decode": 100.0, "prompt": 1000.0, "texts": t, "ok": True},
                "base-again": {"decode": again, "prompt": 1000.0, "texts": t, "ok": True},
                "F": {"decode": cand_dec, "prompt": 1000.0, "texts": t if texts == "same" else "other", "ok": True}}

    def flag(self, exact=True, gate=None, metric="decode"):
        return {"name": "F", "kind": "env", "value": "1", "exact": exact, "metric": metric, "ledger": "T-1", "quality_gate": gate}

    def test_exact_gain(self):
        self.assertEqual(fd.classify(self.flag(), self.arms(103.0))["verdict"], "GAIN")

    def test_exact_below_bar(self):
        self.assertEqual(fd.classify(self.flag(), self.arms(101.5))["verdict"], "NO-GAIN")

    def test_inside_noise(self):
        self.assertEqual(fd.classify(self.flag(), self.arms(102.5, again=104.0))["verdict"], "NO-GAIN")

    def test_exact_parity_broken(self):
        v = fd.classify(self.flag(), self.arms(120.0, texts="other"))
        self.assertEqual(v["verdict"], "UNSUPPORTED")
        self.assertIn("text differs", v["reason"])

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
    def test_parse_and_run(self):
        hw = fd.detect(fake_root(), lambda cmd: ROCMINFO)
        out = Path(tempfile.mkdtemp())

        def runner(cmd, outdir):
            spec = json.loads(Path(cmd[2]).read_text())
            self.assertEqual(spec["arms"]["base"], {})
            recs = [row("base", "short", 100.0), row("base-again", "short", 100.4)]
            for name in spec["arms"]:
                if name.startswith("base"):
                    continue
                recs.append(row(name, "short", 110.0 if name == "STRATA_RDNA_Q6K" else 100.5))
            (outdir / "abtest.jsonl").write_text(jsonl(*recs))
            return 0

        res = fd.run("cfg.json", out, ["STRATA_RDNA_Q6K", "STRATA_HIPBLASLT_TUNING", "--adapt-every", "STRATA_PF_F16"],
                     60.0, Path("abtest.py"), runner=runner, hw=hw)
        v = {x["name"]: x for x in res["verdicts"]}
        self.assertEqual(v["STRATA_RDNA_Q6K"]["verdict"], "GAIN")
        self.assertEqual(v["--adapt-every"]["verdict"], "NO-GAIN")
        self.assertEqual(v["STRATA_PF_F16"]["verdict"], "NO-GAIN")
        self.assertTrue((out / "verdicts.json").exists())

    def test_budget_and_unsupported(self):
        hw = fd.detect(fake_root(hipblaslt=False), lambda cmd: ROCMINFO)
        out = Path(tempfile.mkdtemp())
        res = fd.run("cfg.json", out, None, 6.0, Path("abtest.py"), runner=lambda c, o: 1, hw=hw)
        v = {x["name"]: x for x in res["verdicts"]}
        self.assertEqual(v["STRATA_HIPBLASLT_TUNING"]["verdict"], "UNSUPPORTED")
        self.assertGreater(sum(1 for x in res["verdicts"] if x["verdict"] == "UNTESTED"), 0)


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
