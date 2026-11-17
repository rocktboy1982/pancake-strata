"""serve/test_prompt_normalize.py - STRATA_PROMPT_NL_NORMALIZE (LEADS.md Part P L2): the pure function, the switch,
and that an unset switch leaves render_prompt byte-identical.

    python -m unittest serve.test_prompt_normalize -v
"""
from __future__ import annotations

import os
import sys
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.prompt_normalize import normalize_prompt_newlines, prompt_nl_normalize_enabled  # noqa: E402


class NormalizeTest(unittest.TestCase):
    def test_collapse(self):
        self.assertEqual(normalize_prompt_newlines("a\n\n\nb"), "a\n\nb")
        self.assertEqual(normalize_prompt_newlines("a\n\n\n\n\n\nb\n\n\nc"), "a\n\nb\n\nc")

    def test_two_and_one_untouched(self):
        for t in ("a\nb", "a\n\nb", "", "\n", "\n\n", "no newline"):
            self.assertEqual(normalize_prompt_newlines(t), t)

    def test_edges(self):
        self.assertEqual(normalize_prompt_newlines("\n\n\nx\n\n\n"), "\n\nx\n\n")

    def test_idempotent(self):
        t = "def f():\n    pass\n\n\n\ndef g():\n    pass\n\n\n"
        once = normalize_prompt_newlines(t)
        self.assertEqual(normalize_prompt_newlines(once), once)
        self.assertNotIn("\n\n\n", once)

    def test_crlf_is_not_a_run(self):
        t = "a\r\n\r\n\r\nb"           # no 3 consecutive '\n': left alone
        self.assertEqual(normalize_prompt_newlines(t), t)

    def test_switch(self):
        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STRATA_PROMPT_NL_NORMALIZE", None)
            self.assertFalse(prompt_nl_normalize_enabled())
        for v, want in (("1", True), (" 1 ", True), ("0", False), ("", False), ("true", False)):
            with unittest.mock.patch.dict(os.environ, {"STRATA_PROMPT_NL_NORMALIZE": v}):
                self.assertEqual(prompt_nl_normalize_enabled(), want, v)


class RenderHookTest(unittest.TestCase):
    """Service.render_prompt: off = the raw render, on = the raw render normalized."""

    def _svc(self):
        from serve.server import Service
        svc = object.__new__(Service)
        svc._render_prompt_raw = lambda messages, tools, kwargs: "sys\n\n\n\nuser\n\n\nx"
        return svc, Service

    def test_off_is_raw(self):
        svc, S = self._svc()
        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STRATA_PROMPT_NL_NORMALIZE", None)
            self.assertEqual(S.render_prompt(svc, [], None, {}), "sys\n\n\n\nuser\n\n\nx")

    def test_on_normalizes(self):
        svc, S = self._svc()
        with unittest.mock.patch.dict(os.environ, {"STRATA_PROMPT_NL_NORMALIZE": "1"}):
            self.assertEqual(S.render_prompt(svc, [], None, {}), "sys\n\nuser\n\nx")


if __name__ == "__main__":
    unittest.main()
