"""STRATA_TOOL_IDLE / STRATA_TOOL_IDLE_PREWARM (opt-in; MORI idleness prediction + TokenCake predictive upload,
LEADS.md).

    python -m unittest serve.test_tool_idle -v
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
from serve import tool_idle  # noqa: E402


def cfg(**kw):
    return {"on": True, "prewarm": False, "alpha": 0.5, "default_s": 5.0, "lead_s": 0.0, "min_s": 0.0, **kw}


def ev(kind, **kw):
    return "event", SimpleNamespace(kind=kind, text=kw.get("text", ""), call=kw.get("call"))


class FakeSvc:
    def __init__(self):
        self.fifo = threading.Lock()
        self.runs = []
        self.ran = threading.Event()

    def run(self, ids, thinking, tools, max_new, sampling, cancel):
        self.runs.append((list(ids), max_new, tool_idle.background()))
        self.ran.set()
        yield "done", {"finish": "length", "completion_tokens": 1}


class PredictorTest(unittest.TestCase):
    def test_ema_and_default(self):
        p = tool_idle.Predictor(alpha=0.5, default_s=5.0)
        self.assertEqual(p.predict(["bash"]), 5.0)
        p.observe(["bash"], 10.0)
        p.observe(["bash"], 20.0)
        self.assertEqual(p.predict(["bash"]), 15.0)
        p.observe(["read"], 1.0)
        self.assertEqual(p.predict(["read", "bash"]), 15.0)      # the slowest decides

    def test_result_ids_from_every_api(self):
        req = {"messages": [{"role": "tool", "tool_call_id": "a"},
                            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "b"}]},
                            {"type": "function_call_output", "call_id": "c"}]}
        self.assertEqual(tool_idle.tool_result_ids(req), {"a", "b", "c"})


class TrackerTest(unittest.TestCase):
    def test_default_env_is_off(self):
        with mock.patch.object(tool_idle, "_CFG", None), mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(tool_idle.enabled())
            self.assertIsNone(tool_idle.tracker())

    def test_wrap_passes_events_and_observes_the_window(self):
        t = tool_idle.Tracker(cfg())
        call = SimpleNamespace(id="call_1", name="bash")
        events = [ev("content", text="ok"), ev("tool_call", call=call), ("done", {"finish": "stop"})]
        self.assertEqual(list(t.wrap(FakeSvc(), [1, 2, 3], iter(events))), events)
        self.assertEqual(t.pending["ids"], {"call_1"})
        time.sleep(0.05)
        t.on_request({"messages": [{"role": "tool", "tool_call_id": "call_1", "content": "x"}]})
        self.assertIsNone(t.pending)
        self.assertEqual(t.stats["observed"], 1)
        self.assertGreater(t.pred.predict(["bash"]), 0.0)
        self.assertLess(t.pred.predict(["bash"]), 5.0)

    def test_other_request_marks_interleaved(self):
        t = tool_idle.Tracker(cfg())
        t.on_turn_end(FakeSvc(), [1], [("a", "bash")])
        t.on_request({"messages": [{"role": "user", "content": "title please"}]})
        self.assertTrue(t.pending["interleaved"])

    def test_prewarm_runs_only_after_an_interleaved_window(self):
        svc = FakeSvc()
        t = tool_idle.Tracker(cfg(prewarm=True, default_s=0.2))
        t.on_turn_end(svc, [7, 8, 9], [("a", "bash")])
        self.assertFalse(svc.ran.wait(0.4))                      # not interleaved: the KV never left
        svc = FakeSvc()
        t.on_turn_end(svc, [7, 8, 9], [("b", "bash")])
        t.on_request({"messages": [{"role": "user", "content": "side request"}]})
        # the side request cancelled the timer; a fresh turn end then a side request with the timer re-armed:
        t.on_turn_end(svc, [7, 8, 9], [("c", "bash")])
        t.pending["interleaved"] = True
        self.assertTrue(svc.ran.wait(2.0))
        self.assertEqual(svc.runs[0], ([7, 8, 9], 1, True))

    def test_prewarm_skipped_when_the_engine_is_busy(self):
        svc = FakeSvc()
        svc.fifo.acquire()
        t = tool_idle.Tracker(cfg(prewarm=True, default_s=0.05))
        t.on_turn_end(svc, [1], [("a", "bash")])
        t.pending["interleaved"] = True
        time.sleep(0.3)
        self.assertEqual(svc.runs, [])
        self.assertEqual(t.stats["prewarm_skipped"], 1)


if __name__ == "__main__":
    unittest.main()
