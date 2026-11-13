"""abtest.render_preset fills the preset placeholders (regression: the server died before ready on a literal <STRATA_DATA>)."""
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

    def test_default_repo_is_the_checkout(self):
        out = abtest.render_preset('{"a": "<REPO>"}')
        self.assertTrue(os.path.isdir(json.loads(out)["a"]))


if __name__ == "__main__":
    unittest.main()
