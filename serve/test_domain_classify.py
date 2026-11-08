"""MOD_PLAN 11 item 4 (pancake-strata): the request's domain for the engine's expert-tier seed (STRATA_DOMAIN_PROFILES).
classify_domain picks code / tool / prose from the last messages; sampling_keys sends it as domain=<name> only when set.

    python -m serve.test_domain_classify
"""
import unittest

from serve.server import StrataEngine, classify_domain, domain_profiles_on


class DomainClassify(unittest.TestCase):
    def test_code_fence(self):
        msgs = [{"role": "user", "content": "Fix this:\n```python\ndef f(x):\n    return x\n```"}]
        self.assertEqual(classify_domain(msgs, None), "code")

    def test_code_lines_in_a_tool_result(self):
        src = "\n".join(["#include <cstdio>", "static int a;", "int main() {", "    return 0;", "}"] * 3)
        msgs = [{"role": "user", "content": "read the file"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
                {"role": "tool", "content": src}]
        self.assertEqual(classify_domain(msgs, [{"type": "function"}]), "code")

    def test_anthropic_parts(self):
        msgs = [{"role": "user", "content": [{"type": "tool_result", "content": "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n"
                                                                                  "-a\n+b\n" * 3}]}]
        self.assertEqual(classify_domain(msgs, None), "code")

    def test_tool_without_code(self):
        msgs = [{"role": "user", "content": "What is the weather in Paris tomorrow?"}]
        self.assertEqual(classify_domain(msgs, [{"type": "function", "function": {"name": "weather"}}]), "tool")

    def test_prose(self):
        msgs = [{"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Write a short story about a lighthouse keeper.\nMake it sad."}]
        self.assertEqual(classify_domain(msgs, None), "prose")
        self.assertEqual(classify_domain([], None), "prose")

    def test_switch(self):
        self.assertFalse(domain_profiles_on({}))
        self.assertFalse(domain_profiles_on({"STRATA_DOMAIN_PROFILES": " "}))
        self.assertTrue(domain_profiles_on({"STRATA_DOMAIN_PROFILES": "/tmp/p"}))

    def test_key(self):
        self.assertIn("domain=code", StrataEngine.sampling_keys({"strata_domain": "code"}).split())
        self.assertNotIn("domain", StrataEngine.sampling_keys({}))
        for bad in ("a b", "x=1", "", 3, "../etc"):
            self.assertNotIn("domain", StrataEngine.sampling_keys({"strata_domain": bad}))


if __name__ == "__main__":
    unittest.main()
