"""abtest.render_preset fills the preset placeholders (regression: the server died before ready on a literal <STRATA_DATA>)."""
import sys
import importlib.util
import json
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("abtest", os.path.join(HERE, "abtest.py"))
abtest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(abtest)


class RenderPreset(unittest.TestCase):
    def test_placeholders_are_filled(self):
        text = '{"a": "<REPO>/x", "b": "<STRATA_DATA>/packs", "c": "<MODELS>/m"}'
        out = abtest.render_preset(text, repo="/r")
        self.assertNotIn("<", out)
        d = json.loads(out)
        self.assertEqual(d["a"], "/r/x")
        self.assertTrue(d["b"].endswith("Strata-data/packs"))
        self.assertTrue(d["c"].endswith("models/m"))

    @unittest.skipIf(not sys.platform.startswith("linux"), "render_preset splices the raw repo path into JSON text: Linux paths only")
    def test_default_repo_is_the_checkout(self):
        out = abtest.render_preset('{"a": "<REPO>"}')
        self.assertTrue(os.path.isdir(json.loads(out)["a"]))


class ArmValuesAreRendered(unittest.TestCase):
    """Regression (spec 126, 2026-11-17): an arm's --expert-profile "<REPO>/data/..." reached the engine literally and the server died before ready."""
    def test_arm_args_and_env_carry_no_placeholder(self):
        import tempfile
        d = tempfile.mkdtemp(prefix="mc-")
        base = os.path.join(d, "base.json")
        json.dump({"exe": "/x/strata", "args": ["--kv", "int8"], "env": {"A": "1"}}, open(base, "w"))
        arm = {"args": {"--expert-profile": "<REPO>/data/p.bin"}, "env": {"STRATA_W": "<REPO>/data/w.lkw", "STRATA_D": "<STRATA_DATA>/x"}}
        out = os.path.join(d, "out.json")
        c = abtest.make_config(base, arm, out)
        text = json.dumps(c)
        self.assertNotIn("<REPO>", text)
        self.assertNotIn("<STRATA_DATA>", text)
        i = c["args"].index("--expert-profile")
        self.assertTrue(c["args"][i + 1].endswith("/data/p.bin"))
        self.assertTrue(os.path.isabs(c["args"][i + 1]))
        self.assertTrue(c["env"]["STRATA_W"].endswith("/data/w.lkw"))


if __name__ == "__main__":
    unittest.main()
