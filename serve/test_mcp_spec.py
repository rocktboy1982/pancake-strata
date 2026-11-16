"""STRATA_MCP_SPEC (opt-in; speculative read-only tool execution after PASTE / Atomix, LEADS.md).

    python -m unittest serve.test_mcp_spec -v
"""
from __future__ import annotations

import collections
import re
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import mcp_spec  # noqa: E402
from serve.test_mcp import ToolLoop  # noqa: E402


def cfg(**kw):
    return {"min": 2, "tools": re.compile(mcp_spec._DEFAULT_TOOLS, re.I), "ttl": 30.0, "max": 64, **kw}


class Hub:
    settings = {"timeout_s": 5}

    def __init__(self):
        self.calls = []

    def call(self, name, arguments, cancel=None):
        self.calls.append((name, dict(arguments)))
        return {"ok": True, "text": f"{name} {arguments}", "chars": 1, "truncated": False, "ms": 0}


class SpeculatorTest(unittest.TestCase):
    def setUp(self):
        self._t = mock.patch.object(mcp_spec, "SUCCESSORS", collections.OrderedDict())
        self._t.start()

    def tearDown(self):
        self._t.stop()

    def test_default_env_is_off(self):
        with mock.patch.object(mcp_spec, "_CFG", None), mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(mcp_spec.enabled())

    def test_read_only_rule(self):
        self.assertTrue(mcp_spec.read_only("fs__read_file"))
        self.assertTrue(mcp_spec.read_only("git__list"))
        self.assertFalse(mcp_spec.read_only("fs__write_file"))
        self.assertFalse(mcp_spec.read_only("shell__run"))

    def test_learned_successor_runs_ahead_and_is_taken(self):
        hub = Hub()
        for _ in range(2):                                 # two runs: A then B
            s = mcp_spec.Speculator(hub, threading.Event(), cfg())
            s.observe("fs__list_dir", {"path": "."})
            s.observe("fs__read_file", {"path": "a.py"})
            s.close()
        hub.calls.clear()
        s = mcp_spec.Speculator(hub, threading.Event(), cfg())
        s.observe("fs__list_dir", {"path": "."})           # predicts read_file a.py and starts it
        self.assertEqual(list(s.pending), [mcp_spec.key("fs__read_file", {"path": "a.py"})])
        r = s.take("fs__read_file", {"path": "a.py"})
        self.assertIsNotNone(r)
        self.assertEqual(hub.calls, [("fs__read_file", {"path": "a.py"})])
        self.assertIsNone(s.take("fs__read_file", {"path": "b.py"}))   # a different call: not speculative

    def test_below_min_count_and_writes_never_run_ahead(self):
        hub = Hub()
        s = mcp_spec.Speculator(hub, threading.Event(), cfg())
        s.observe("fs__list_dir", {})
        s.observe("fs__write_file", {"path": "x"})
        for _ in range(3):
            s2 = mcp_spec.Speculator(hub, threading.Event(), cfg(min=1))
            s2.observe("fs__list_dir", {})
            self.assertEqual(s2.pending, {})                # write_file is not read-only
            s2.observe("fs__write_file", {"path": "x"})

    def test_stale_result_is_not_used(self):
        hub = Hub()
        mcp_spec.learn(mcp_spec.key("a__get", {}), mcp_spec.key("a__get", {"n": 1}), 64)
        s = mcp_spec.Speculator(hub, threading.Event(), cfg(min=1, ttl=-1.0))
        s.observe("a__get", {})
        self.assertIsNone(s.take("a__get", {"n": 1}))


class SpecToolLoop(ToolLoop):
    """The MCP tool-loop suite with the switch on (min count 1): same events, prompts and results."""

    def setUp(self):
        self._c = mock.patch.object(mcp_spec, "_CFG", cfg(min=1))
        self._t = mock.patch.object(mcp_spec, "SUCCESSORS", collections.OrderedDict())
        self._c.start()
        self._t.start()
        super().setUp()

    def tearDown(self):
        super().tearDown()
        self._t.stop()
        self._c.stop()


if __name__ == "__main__":
    unittest.main()
