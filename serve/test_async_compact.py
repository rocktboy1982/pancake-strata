"""STRATA_ASYNC_COMPACT (opt-in; Slipstream-style compaction in the idle window, LEADS.md A39).

    python -m unittest serve.test_async_compact -v
"""
from __future__ import annotations

import collections
import json
import re
import sys
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import async_compact, tool_idle  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, Service, serve  # noqa: E402
from serve.test_mcp import ScriptedEngine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INSTR = "Provide a detailed prompt for continuing our conversation above."


class Unit(unittest.TestCase):
    def test_default_env_is_off(self):
        with mock.patch.object(async_compact, "_CFG", None), mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(async_compact.enabled())
        with mock.patch.object(tool_idle, "_CFG", None), mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(tool_idle.enabled())

    def test_marks_and_key(self):
        self.assertEqual(async_compact.is_compaction([{"role": "user", "content": INSTR}]), INSTR)
        self.assertIsNone(async_compact.is_compaction([{"role": "user", "content": "fix the bug"}]))
        a = [{"role": "user", "content": "hi  there"}, {"role": "assistant", "content": "ok",
                                                         "reasoning_content": "hmm"}]
        b = [{"role": "user", "content": "hi there"}, {"role": "assistant", "content": "ok"}]
        self.assertEqual(async_compact.conversation_key(a), async_compact.conversation_key(b))


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.patches = [
            mock.patch.object(async_compact, "_CFG", {"tokens": 1, "mark": re.compile(async_compact._DEFAULT_MARK,
                                                                                      re.I), "keep": 4}),
            mock.patch.object(async_compact, "CACHE", collections.OrderedDict()),
            mock.patch.object(async_compact, "STATE", {"instruction": None, "job": None,
                                                       "stats": {"precomputed": 0, "served": 0, "cancelled": 0,
                                                                 "misses": 0}}),
            mock.patch.object(tool_idle, "_CFG", {"on": True, "prewarm": False, "alpha": 0.3, "default_s": 5.0,
                                                  "lead_s": 1.0, "min_s": 2.0}),
            mock.patch.object(tool_idle, "TRACKER", None),
        ]
        for p in self.patches:
            p.start()
        tok = ByteTokenizer()
        self.engine = ScriptedEngine(tok, ["</think>\n\nFirst summary.", "</think>\n\nHi there.",
                                           "</think>\n\nIdle-window summary.", "</think>\n\nnever"])
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        for p in reversed(self.patches):
            p.stop()

    def post(self, messages):
        body = {"model": "m", "messages": messages, "stream": False}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())["choices"][0]["message"]["content"]

    def test_summary_made_in_the_idle_window_is_served(self):
        self.assertEqual(self.post([{"role": "user", "content": "old"}, {"role": "user", "content": INSTR}]),
                         "First summary.")                      # learns the instruction, generated as usual
        self.assertEqual(self.post([{"role": "user", "content": "hello"}]), "Hi there.")
        for _ in range(100):                                   # the background run takes the next script
            if async_compact.STATE["stats"]["precomputed"]:
                break
            time.sleep(0.05)
        self.assertEqual(async_compact.STATE["stats"]["precomputed"], 1)
        n = len(self.engine.prompts)
        self.assertIn(INSTR, self.engine.prompt_text(n - 1))
        got = self.post([{"role": "user", "content": "hello"}, {"role": "assistant", "content": "Hi there."},
                         {"role": "user", "content": INSTR}])
        self.assertEqual(got, "Idle-window summary.")
        self.assertEqual(len(self.engine.prompts), n)          # served without the engine
        self.assertEqual(async_compact.STATE["stats"]["served"], 1)


if __name__ == "__main__":
    unittest.main()
