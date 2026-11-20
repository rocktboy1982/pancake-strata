"""strata_control/tuned_flags.py: fake hardware, fake verdict files.  Run: python -m pytest -q terminal/tests/tuning/test_tuned_flags.py"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from strata_control import tuned_flags as tf  # noqa: E402

HW = {"gfx": "gfx1100", "rocm": "10.0.2", "vram_gib": 20.0, "pcie_link": "16.0 GT/s x16", "cpu": "x", "ram_gib": 64.0}


def flag(kind, name, value, verdict="GAIN", exact=True, gain=5.0, gate=None):
    d = {"kind": kind, "name": name, "value": value, "verdict": verdict, "exact": exact, "gain_pct": gain}
    if gate:
        d["quality_gate"] = gate
    return d


def write(flags, match=None):
    p = Path(tempfile.mkdtemp()) / "t.json"
    p.write_text(json.dumps({"schema": tf.SCHEMA, "profiles": [
        {"match": match or {"gfx": "gfx1100", "rocm": ["10."], "vram_gib": [16, 24]}, "flags": flags}]}))
    return p


class Rule(unittest.TestCase):
    def test_accepted(self):
        self.assertTrue(tf.accepted(flag("env", "A", "1", exact=True, gain=2.0)))
        self.assertFalse(tf.accepted(flag("env", "A", "1", exact=True, gain=1.9)))
        self.assertTrue(tf.accepted(flag("env", "A", "1", exact=False, gain=3.0, gate="PASS")))
        self.assertFalse(tf.accepted(flag("env", "A", "1", exact=False, gain=9.0)))
        self.assertFalse(tf.accepted(flag("env", "A", "1", exact=False, gain=2.9, gate="PASS")))
        for bad in ("NO-GAIN", "UNSUPPORTED", "UNTESTED", "OK"):
            self.assertFalse(tf.accepted(flag("env", "A", "1", verdict=bad, gain=50)))
        self.assertFalse(tf.accepted({"kind": "env", "name": "A", "verdict": "GAIN", "exact": True}))

    def test_file_drops_rejected_records(self):
        p = write([flag("env", "KEEP", "1"), flag("env", "NOGAIN", "1", verdict="NO-GAIN"), flag("env", "LOSSY", "1", exact=False)])
        self.assertEqual([f["name"] for f in tf.load_profiles(p)[0]["flags"]], ["KEEP"])


class Match(unittest.TestCase):
    def test_match(self):
        self.assertTrue(tf.matches({"gfx": "gfx1100", "rocm": ["10."], "vram_gib": [16, 24]}, HW))
        self.assertFalse(tf.matches({"gfx": "gfx1201"}, HW))
        self.assertFalse(tf.matches({"gfx": "gfx1100", "rocm": ["7."]}, HW))
        self.assertFalse(tf.matches({"gfx": "gfx1100", "vram_gib": [24, 48]}, HW))
        self.assertTrue(tf.matches({"gfx": "gfx1100"}, HW))


class Apply(unittest.TestCase):
    def go(self, cfg, files, env=None, argv=None, hw=HW):
        lines = []
        out = tf.apply(cfg, env=env or {}, argv=argv or [], hw=hw, files=files, say=lines.append)
        return out, lines

    def test_adds_flags(self):
        p = write([flag("env", "STRATA_X", "<ROOT>/t.txt"), flag("arg", "--pcie-frac", "0"), flag("arg", "--bare", "")])
        out, lines = self.go({"args": ["--spec", "4"], "env": {"K": "v"}}, [p])
        self.assertEqual(out["env"]["K"], "v")
        self.assertTrue(out["env"]["STRATA_X"].endswith("t.txt") and "<ROOT>" not in out["env"]["STRATA_X"])
        self.assertEqual(out["args"], ["--spec", "4", "--pcie-frac", "0", "--bare"])
        self.assertEqual(len(lines), 1)
        self.assertIn("gfx1100", lines[0])

    def test_user_setting_wins(self):
        p = write([flag("env", "STRATA_X", "1"), flag("env", "STRATA_Y", "1"), flag("arg", "--pcie-frac", "0"), flag("arg", "--adapt", "1")])
        cfg = {"args": ["--pcie-frac", "0.3", "--adapt=2"], "env": {"STRATA_X": "0"}}
        out, _ = self.go(cfg, [p], env={"STRATA_Y": "0"})
        self.assertEqual(out["env"], {"STRATA_X": "0"})
        self.assertEqual(out["args"], ["--pcie-frac", "0.3", "--adapt=2"])

    def test_off_switches(self):
        p = write([flag("env", "STRATA_X", "1")])
        cfg = {"env": {}}
        for env, argv in (({"STRATA_TUNED_FLAGS": "0"}, []), ({}, ["strata", "--no-tuned-flags"])):
            out, lines = self.go(cfg, [p], env=env, argv=argv)
            self.assertEqual(out, cfg)
            self.assertEqual(lines, [])
        out, _ = self.go(cfg, [p], env={"STRATA_TUNED_FLAGS": "1"})
        self.assertIn("STRATA_X", out["env"])

    def test_unknown_hardware_plain_with_one_line(self):
        p = write([flag("env", "STRATA_X", "1")], match={"gfx": "gfx1201"})
        out, lines = self.go({"env": {}}, [p])
        self.assertEqual(out, {"env": {}})
        self.assertEqual(len(lines), 1)
        self.assertIn("plain defaults", lines[0])

    def test_no_hardware_read(self):
        lines = []
        out = tf.apply({"env": {}}, env={}, argv=[], files=[], say=lines.append, root="/nonexistent-root")
        self.assertEqual(out, {"env": {}})
        self.assertEqual(len(lines), 1)
        self.assertIn("plain defaults", lines[0])

    def test_user_file_first_then_shipped(self):
        user = write([flag("env", "FROM_USER", "1")])
        shipped = write([flag("env", "FROM_SHIPPED", "1")])
        out, _ = self.go({"env": {}}, [user, shipped])
        self.assertIn("FROM_USER", out["env"])
        self.assertNotIn("FROM_SHIPPED", out["env"])
        out, _ = self.go({"env": {}}, [Path("/nonexistent.json"), shipped])
        self.assertIn("FROM_SHIPPED", out["env"])

    def test_shipped_default_is_valid_and_all_accepted(self):
        raw = json.loads(tf.SHIPPED.read_text(encoding="utf-8"))
        n = sum(len(p["flags"]) for p in raw["profiles"])
        self.assertGreater(n, 0)
        self.assertEqual(n, sum(len(p["flags"]) for p in tf.load_profiles(tf.SHIPPED)))
        out, _ = self.go({"args": [], "env": {}}, [tf.SHIPPED])
        self.assertIn("--pcie-frac", out["args"])
        # the three RDNA kernels stay (owner decision 2026-11-20: text identity cannot judge exactness on this model, the parity ctests do)
        self.assertEqual((out["env"]["STRATA_RDNA_Q6K"], out["env"]["STRATA_GR_RDNA"], out["env"]["STRATA_RDNA_MMVQ_WAVE"]), ("1", "1", "3"))
        # the 100401 hipBLASLt table is back (owner decision 2026-11-20: its text-identity UNSUPPORTED is void here; it matches the ROCm 10.0 runtime)
        self.assertTrue(out["env"]["STRATA_HIPBLASLT_TUNING"].endswith("gfx1100-hipblaslt-100401.txt"))
        self.assertNotIn("<ROOT>", out["env"]["STRATA_HIPBLASLT_TUNING"])
        # removed 2026-11-20 (owner order 2026-11-19 23:05): --lookup-max-t 12 (lossy, gate-us REJECT)
        gone = {r["name"]: r for r in raw["removed"]}
        self.assertEqual(set(gone), {"--lookup-max-t"})
        for name, r in gone.items():
            self.assertNotIn(name, out["env"])
            self.assertNotIn(name, out["args"])
            self.assertIn(r["verdict"], ("FAIL", "NO-GAIN", "UNSUPPORTED", "GATE-REJECT"))
            self.assertTrue(r["evidence"])
        for p in raw["profiles"]:
            for f in p["flags"]:   # a lossy flag only with a passing gate
                self.assertTrue(f["exact"] is True or f.get("quality_gate") == "PASS", f["name"])

    def test_no_shipped_setting_points_outside_this_tree(self):
        """Owner order 2026-11-19 23:05: a shipped setting names only files of this tree (<ROOT> / <REPO> = the checkout), never the pancake tree;
        the model data folder (<STRATA_DATA>) and the build output (build-hip/) live elsewhere by design."""
        root = Path(__file__).resolve().parents[3]

        def strings(v):
            if isinstance(v, dict):
                return [x for w in v.values() for x in strings(w)]
            if isinstance(v, list):
                return [x for w in v for x in strings(w)]
            return [v] if isinstance(v, str) else []
        raw = json.loads(tf.SHIPPED.read_text(encoding="utf-8"))
        for p in raw["profiles"]:
            for f in p["flags"]:
                v = str(f.get("value", ""))
                if "/" in v:
                    self.assertTrue(v.startswith("<ROOT>/"), v)
                    self.assertTrue((root / v[len("<ROOT>/"):]).is_file(), v)
        for rel in ("configs/pancake/flashnext-iq2_xs-updated-hip.json", "configs/pancake/flashnext-iq2_xs-plain-upstream.json",
                    "bench/specs/220-updated-vs-upstream-short.json", "bench/specs/221-updated-vs-upstream-long.json"):
            d = json.loads((root / rel).read_text(encoding="utf-8"))
            for v in strings({k: d[k] for k in ("args", "env", "arms") if k in d}):
                self.assertNotIn("clone-of-d/src/Strata/", v, rel)
                self.assertNotIn("Strata-win", v, rel)
                if v.startswith("<REPO>/") and not v.startswith("<REPO>/build-hip/"):
                    self.assertTrue((root / v[len("<REPO>/"):]).is_file(), (rel, v))

    def test_never_raises(self):
        lines = []
        self.assertEqual(tf.apply({"env": {}}, env={}, argv=[], hw=HW, files=[object()], say=lines.append), {"env": {}})


if __name__ == "__main__":
    unittest.main()
