"""STRATA_TOOL_SNIP (opt-in; ContextSniper-style tool-output filtering, LEADS.md Part E)."""
import unittest
from unittest import mock

from serve import frontend, tool_snip


def _long_output(n=500):
    lines = [f"line {i} ok" for i in range(n)]
    lines[250] = "src/foo.py:12: error: name 'x' is not defined"
    return "\n".join(lines)


class ToolSnipTest(unittest.TestCase):
    def test_off_by_default_leaves_messages_identical(self):
        text = _long_output()
        req = {"messages": [{"role": "user", "content": "hi"},
                            {"role": "assistant", "content": None,
                             "tool_calls": [{"id": "a", "function": {"name": "f", "arguments": "{}"}}]},
                            {"role": "tool", "tool_call_id": "a", "content": text}]}
        with mock.patch.object(tool_snip, "_CFG", None), mock.patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("STRATA_TOOL_SNIP", None)
            msgs, _, _ = frontend.openai_to_messages(req)
        tool = [m for m in msgs if m["role"] == "tool"][0]
        self.assertEqual(tool["content"], text)

    def test_snip_keeps_head_tail_and_diagnosis(self):
        text = _long_output()
        out = tool_snip.snip_text(text, 1000, 5, 5, 200)
        self.assertIn("line 0 ok", out)
        self.assertIn("line 499 ok", out)
        self.assertIn("src/foo.py:12: error", out)
        self.assertNotIn("line 100 ok", out)
        self.assertIn("lines omitted (STRATA_TOOL_SNIP)", out)
        self.assertLess(len(out), len(text))
        self.assertEqual(out, tool_snip.snip_text(text, 1000, 5, 5, 200))   # deterministic: the prefix cache holds

    def test_short_result_unchanged(self):
        self.assertEqual(tool_snip.snip_text("a\nb", 1000, 1, 1, 10), "a\nb")
        self.assertEqual(tool_snip.snip_text("x" * 5000, 10, 40, 40, 10), "x" * 5000)   # one line: nothing to omit

    def test_on_filters_openai_tool_message(self):
        text = _long_output()
        req = {"messages": [{"role": "user", "content": "hi"},
                            {"role": "assistant", "content": None,
                             "tool_calls": [{"id": "a", "function": {"name": "f", "arguments": "{}"}}]},
                            {"role": "tool", "tool_call_id": "a", "content": text}]}
        with mock.patch.object(tool_snip, "_CFG", (1000, 5, 5, 200)):
            msgs, _, _ = frontend.openai_to_messages(req)
        tool = [m for m in msgs if m["role"] == "tool"][0]
        self.assertIn("lines omitted", tool["content"])
        user = [m for m in msgs if m["role"] == "user"][0]
        self.assertEqual(user["content"], "hi")


if __name__ == "__main__":
    unittest.main()
