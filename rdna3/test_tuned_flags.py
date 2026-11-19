"""Tests of rdna3/tuned_flags.py with fake hardware and fake verdicts (no GPU, no sysfs, no engine).
Run: python -m pytest rdna3/test_tuned_flags.py   (or python rdna3/test_tuned_flags.py)"""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tuned_flags as tf

FILES = {
    "/sys/class/drm/card0/device/mem_info_vram_total": str(20 * (1 << 30)),
    "/sys/class/drm/card0/device/current_link_speed": "8.0 GT/s PCIe",
    "/sys/class/drm/card0/device/current_link_width": "16",
    "/proc/cpuinfo": "processor : 0\nmodel name\t: AMD Ryzen 7 5700G\n",
    "/proc/meminfo": "MemTotal:       58720256 kB\n",
    "/opt/rocm/.info/version": "10.0.0-123\n",
}


def fake_read(path):
    p = str(path).replace("\\", "/")
    for k, v in FILES.items():
        if p.endswith(k):
            return v
    raise OSError(p)


def fake_run(cmd):
    return "gfx000\ngfx1100\n" if cmd[0] == "rocm_agent_enumerator" else ""


HW = {"gfx": "gfx1100", "rocm": "10.0", "vram_gib": 20}


class T(unittest.TestCase):
    def cfg(self):
        return {"args": ["--spec", "4"], "env": {"STRATA_PF_F16": "2", "GPU_MAX_HW_QUEUES": "6"}}

    def test_detect(self):
        hw = tf.detect_hardware(read=fake_read, run=fake_run, env={})
        self.assertEqual((hw["gfx"], hw["rocm"], hw["vram_gib"], hw["ram_gib"]), ("gfx1100", "10.0", 20, 56))
        self.assertIn("5700G", hw["cpu"])
        self.assertIn("16", hw["pcie"])

    def test_detect_no_gpu(self):
        hw = tf.detect_hardware(read=fake_read, run=lambda c: "", env={})
        self.assertEqual(hw["gfx"], "")

    def test_shipped_default_for_gfx1100(self):
        rec, path = tf.load_recommended(HW)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["env"]["STRATA_POOL_FUSE_PHASES"], "1")
        self.assertNotIn("STRATA_PF_F16", rec["env"])   # owner rule: Strata-rdna3's own PF_F16=2 stays; the pancake value is a spec arm
        self.assertEqual(rec["env"]["STRATA_VERIFY_ECAP"], "0")          # the pancake daily preset's gated ECAP (16-row windows now)
        self.assertEqual(rec["env"]["STRATA_VERIFY_ECAP_MIN_ROWS"], "9")
        self.assertEqual(rec["args"]["--lookup-max-t"], "12")
        self.assertIn("STRATA_PF_F16", rec["theirs_kept"])
        for k in rec["env"]:   # every exported key is either backed by a verdict or listed as a preset entry
            self.assertTrue(k in rec["preset"] or any(k in (e.get("env") or {}) for e in rec["verdicts"].values()), k)

    def test_shipped_flags_are_all_backed_by_a_gain_verdict(self):
        rec, _ = tf.load_recommended(HW)
        for name, e in rec["verdicts"].items():
            self.assertTrue(tf.eligible(e), name)

    def test_small_card_gets_nothing(self):
        rec, _ = tf.load_recommended({"gfx": "gfx1100", "rocm": "10.0", "vram_gib": 8})
        self.assertIsNone(rec)

    def test_apply_sets_and_user_env_wins(self):
        cfg = self.cfg()
        msgs = []
        ch = tf.apply_startup(cfg, env={"STRATA_POOL_FUSE_PHASES": "0"}, hw=HW, say=msgs.append)
        self.assertNotIn("STRATA_POOL_FUSE_PHASES", cfg["env"])      # the user's variable is theirs, the file stays out
        self.assertEqual(cfg["env"]["STRATA_PF_F16"], "2")          # the setup's own value stays (the file does not carry PF_F16)
        self.assertEqual(cfg["env"]["STRATA_VERIFY_ECAP"], "0")      # the file's additions are set
        self.assertEqual(cfg["env"]["GPU_MAX_HW_QUEUES"], "6")
        self.assertTrue(ch and len(msgs) == 1 and "tuned flags from" in msgs[0])

    def test_recommended_argument_replaces_the_configs_value_or_is_appended(self):
        rec = {"env": {}, "args": {"--spec": "9", "--stats": "", "--expert-profile": "@STRATA@/p.bin"}}
        cfg = self.cfg()
        cfg["args"] += ["--expert-profile", "old.bin", "--kv", "int8"]
        ch = tf.apply_to_config(cfg, rec, env={}, root="/r")
        self.assertEqual(cfg["args"], ["--spec", "9", "--expert-profile", "/r/p.bin", "--kv", "int8", "--stats"])
        self.assertEqual({c[1] for c in ch}, {"--spec", "--stats", "--expert-profile"})
        cfg2 = self.cfg()
        self.assertEqual(tf.apply_to_config(cfg2, {"env": {}, "args": {"--spec": "4"}}, env={}), [])   # same value: nothing changes

    def test_off_switches_are_byte_plain(self):
        for kw in ({"no_tuned": True, "env": {}}, {"no_tuned": False, "env": {"STRATA_TUNED_FLAGS": "0"}}):
            cfg = self.cfg(); before = copy.deepcopy(cfg); msgs = []
            self.assertEqual(tf.apply_startup(cfg, hw=HW, say=msgs.append, **kw), [])
            self.assertEqual(cfg, before)
            self.assertEqual(len(msgs), 1)

    def test_unknown_hardware_is_plain_with_one_line(self):
        for hw in ({"gfx": "gfx1030", "rocm": "6.2", "vram_gib": 16}, {"gfx": "", "rocm": "", "vram_gib": 0}):
            cfg = self.cfg(); before = copy.deepcopy(cfg); msgs = []
            self.assertEqual(tf.apply_startup(cfg, env={}, hw=hw, say=msgs.append), [])
            self.assertEqual(cfg, before)
            self.assertEqual(len(msgs), 1)
            self.assertIn("plain defaults", msgs[0])

    def test_eligibility(self):
        g = lambda **k: dict({"verdict": "GAIN", "exact": True, "gain_pct": 2.5}, **k)
        self.assertTrue(tf.eligible(g()))
        self.assertFalse(tf.eligible(g(gain_pct=1.9)))
        self.assertFalse(tf.eligible(g(exact=False)))                           # lossy, no gate
        self.assertFalse(tf.eligible(g(exact=False, gate_pass=True, gain_pct=2.5)))   # lossy bar is 3
        self.assertTrue(tf.eligible(g(exact=False, gate_pass=True, gain_pct=3.2)))
        for v in ("UNSUPPORTED", "NO-GAIN", "OK", "FAIL"):
            self.assertFalse(tf.eligible(g(verdict=v)))

    def test_write_recommended_keeps_only_eligible_and_round_trips(self):
        ver = {
            "a": {"env": {"STRATA_A": "1"}, "verdict": "GAIN", "exact": True, "gain_pct": 4.0},
            "b": {"env": {"STRATA_B": "1"}, "verdict": "NO-GAIN", "exact": True, "gain_pct": 0.3},
            "c": {"env": {"STRATA_C": "1"}, "verdict": "UNSUPPORTED"},
            "d": {"env": {"STRATA_D": "1"}, "verdict": "GAIN", "exact": False, "gain_pct": 9.0},
            "e": {"args": {"--x": "7"}, "verdict": "GAIN", "exact": False, "gate_pass": True, "gain_pct": 5.0},
        }
        with tempfile.TemporaryDirectory() as d:
            p = tf.write_recommended(HW, ver, directory=d)
            rec = json.loads(p.read_text())
            self.assertEqual(rec["env"], {"STRATA_A": "1"})
            self.assertEqual(rec["args"], {"--x": "7"})
            self.assertEqual(set(rec["verdicts"]), set(ver))
            got, _ = tf.load_recommended(HW, directory=d)
            self.assertEqual(got["env"], {"STRATA_A": "1"})


if __name__ == "__main__":
    unittest.main()
