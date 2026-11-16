"""STRATA_MCP_ASYNC (opt-in; AsyncFC-style calls started during the decode, LEADS.md A40).

The whole MCP tool-loop suite (serve/test_mcp.py ToolLoop) runs again with the switch on: the client-visible events,
the prompts and the results must be the same as with calls started after the answer.  Plus the futures themselves.

    python -m unittest serve.test_mcp_async -v
"""
from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import mcp_async  # noqa: E402
from serve.test_mcp import ToolLoop  # noqa: E402


class AsyncToolLoop(ToolLoop):
    def setUp(self):
        self._on = mock.patch.object(mcp_async, "_ON", True)
        self._on.start()
        super().setUp()

    def tearDown(self):
        super().tearDown()
        self._on.stop()


class FakeHub:
    def __init__(self, delay=0.05):
        self.delay, self.started = delay, []

    def call(self, name, arguments, cancel=None):
        self.started.append((name, time.monotonic()))
        time.sleep(self.delay)
        return {"ok": True, "text": f"{name}:{arguments}", "chars": 1, "truncated": False, "ms": 0}


class FuturesTest(unittest.TestCase):
    def test_default_env_is_off(self):
        with mock.patch.object(mcp_async, "_ON", None), mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(mcp_async.enabled())

    def test_calls_run_side_by_side_and_are_taken_once(self):
        hub = FakeHub(delay=0.2)
        f = mcp_async.Futures(hub, threading.Event())
        a, b = SimpleNamespace(id="a", name="x", arguments={}), SimpleNamespace(id="b", name="y", arguments={})
        t0 = time.monotonic()
        f.start(a)
        f.start(b)
        f.start(a)                                     # a second event for the same id does not run it twice
        for c in (a, b):
            worker, box = f.take(c)
            worker.join()
            self.assertTrue(box["r"]["ok"])
        self.assertLess(time.monotonic() - t0, 0.38)   # in parallel, not 0.4 s in a row
        self.assertEqual(len(hub.started), 2)
        self.assertIsNone(f.take(a))

    def test_clear(self):
        f = mcp_async.Futures(FakeHub(delay=0.0), threading.Event())
        f.start(SimpleNamespace(id="a", name="x", arguments={}))
        self.assertEqual(f.clear(), 1)
        self.assertEqual(f.jobs, {})


if __name__ == "__main__":
    unittest.main()
