"""STRATA_MCP_EARLY_STOP (opt-in; agent-run early termination after EarlyEval / EET, LEADS.md).

    python -m unittest serve.test_early_stop -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import early_stop  # noqa: E402
from serve.test_mcp import ToolLoop, call_script  # noqa: E402


def c(name, **args):
    return SimpleNamespace(name=name, arguments=args)


class ProgressTest(unittest.TestCase):
    def test_default_env_is_off(self):
        with mock.patch.object(early_stop, "_N", None), mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(early_stop.patience(), 0)

    def test_new_call_or_new_result_is_progress(self):
        p = early_stop.Progress(2)
        self.assertFalse(p.round([c("a", x=1)], ["r1"]))
        self.assertFalse(p.round([c("a", x=1)], ["r2"]))      # same call, new result: progress
        self.assertEqual(p.stale, 0)
        self.assertFalse(p.round([c("a", x=1)], ["r2"]))      # nothing new: 1
        self.assertTrue(p.round([c("a", x=1)], ["r1"]))       # nothing new: 2 -> stop
        self.assertFalse(early_stop.Progress(0).round([], []))


class EarlyStopLoop(ToolLoop):
    def test_repeated_call_goes_to_the_final_answer(self):
        self.hub.settings["max_rounds"] = 6
        same = call_script("fake__echo", text="same")
        with mock.patch.object(early_stop, "_N", 1):
            self.start(same, same, same, same, "</think>\n\nDone from the evidence.")
            code, text = self.post({"strata_mcp": True})
        self.assertEqual(code, 200, text)
        mcp = [ch["strata_mcp"] for ch in self.chunks(text) if "strata_mcp" in ch]
        self.assertEqual(sum(m["event"] == "call" for m in mcp), 2)    # the second round repeats: stop
        self.assertIn("early_stop", [m["event"] for m in mcp])
        self.assertIn("returned no new information", self.engine.prompt_text(2))
        self.assertNotIn("<tools>", self.engine.prompt_text(2))
        self.assertEqual(len(self.engine.prompts), 3)


if __name__ == "__main__":
    unittest.main()
