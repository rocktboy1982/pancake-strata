"""serve/test_budget_by_effort.py - pancake: the config's "reasoning_budget_by_effort" caps the thinking per level.

    python -m unittest serve.test_budget_by_effort -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, budget_by_effort_setting  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class BudgetByEffort(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "ok", max_context=8192), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.reasoning_budget_by_effort = budget_by_effort_setting({"Low": 2048, "medium": 8192, "high": 0})

    def test_each_api_form_picks_its_level(self):
        b = self.svc.reasoning_budget
        self.assertEqual(b({"reasoning_effort": "low"}), 2048)                              # OpenAI Chat
        self.assertEqual(b({"reasoning": {"effort": "medium"}}), 8192)                      # Responses style
        self.assertEqual(b({"chat_template_kwargs": {"reasoning_effort": "low"}}), 2048)    # vLLM / llama.cpp
        self.assertEqual(b({"output_config": {"effort": "medium"}}), 8192)                  # Anthropic

    def test_high_names_the_same_level_as_xhigh_and_max(self):
        for word in ("high", "xhigh", "max"):                          # all one template level: 0 = no cap
            self.assertIsNone(self.svc.reasoning_budget({"reasoning_effort": word}))

    def test_a_request_budget_wins_and_the_global_one_fills_in(self):
        self.svc.reasoning_budget_tokens = 500
        self.assertEqual(self.svc.reasoning_budget({"reasoning_effort": "low", "reasoning_budget_tokens": 99}), 99)
        self.assertIsNone(self.svc.reasoning_budget({"reasoning_effort": "low", "reasoning_budget_tokens": 0}))
        self.assertIsNone(self.svc.reasoning_budget({"reasoning_effort": "xhigh"}))       # its level says no cap
        self.svc.reasoning_budget_by_effort = {"low": 2048}
        self.assertEqual(self.svc.reasoning_budget({"reasoning_effort": "medium"}), 500)  # unlisted level: global
        self.assertEqual(self.svc.reasoning_budget({}), 500)

    def test_the_default_level_applies_to_a_request_without_one(self):
        self.svc.config_effort = "low"
        self.assertEqual(self.svc.reasoning_budget({}), 2048)
        self.svc.shared = {"reasoning_effort": "medium"}                                   # Chat settings win
        self.assertEqual(self.svc.reasoning_budget({}), 8192)

    def test_bad_config_is_refused(self):
        for bad in ([1], {"low": -1}, {"low": "2k"}, {"low": True}, {"": 5}):
            with self.assertRaises(SystemExit):
                budget_by_effort_setting(bad)


if __name__ == "__main__":
    unittest.main()
