"""serve/test_prefill_warm.py - STRATA_PREFILL_TYPING (LEADS.md Part Q1): read the prompt while the user types.
No GPU, no pack: the pure logic (serve/prefill_warm.py) and the server's wiring against the mock engine.

    python -m unittest serve.test_prefill_warm -v
"""
from __future__ import annotations

import json
import os
import sys
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import prefill_warm  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class Logic(unittest.TestCase):
    def test_default_off(self):
        self.assertFalse(prefill_warm.enabled({}))
        self.assertFalse(prefill_warm.enabled({"STRATA_PREFILL_TYPING": ""}))
        self.assertFalse(prefill_warm.enabled({"STRATA_PREFILL_TYPING": "0"}))
        self.assertTrue(prefill_warm.enabled({"STRATA_PREFILL_TYPING": "1"}))

    def test_min_tokens(self):
        self.assertEqual(prefill_warm.min_tokens({}), 512)
        self.assertEqual(prefill_warm.min_tokens({"STRATA_PREFILL_TYPING_MIN": "100"}), 100)
        self.assertEqual(prefill_warm.min_tokens({"STRATA_PREFILL_TYPING_MIN": "x"}), 512)
        self.assertEqual(prefill_warm.min_tokens({"STRATA_PREFILL_TYPING_MIN": "0"}), 512)

    def test_is_warm(self):
        self.assertFalse(prefill_warm.is_warm({"messages": []}))
        self.assertTrue(prefill_warm.is_warm({"strata_prefill": True}))
        self.assertFalse(prefill_warm.is_warm({"strata_prefill": "yes"}))
        self.assertTrue(prefill_warm.is_warm({}, {"X-Strata-Prefill": "1"}))
        self.assertFalse(prefill_warm.is_warm({}, {"X-Strata-Prefill": "0"}))

    def test_stable_prefix_is_the_common_start_minus_one(self):
        self.assertEqual(prefill_warm.stable_prefix([1, 2, 3, 4], [1, 2, 3, 9, 9]), 2)
        self.assertEqual(prefill_warm.stable_prefix([1], [1, 2]), 0)
        self.assertEqual(prefill_warm.stable_prefix([], [1]), 0)
        self.assertEqual(prefill_warm.stable_prefix([5], [6]), 0)

    def test_probe_messages(self):
        m = [{"role": "system", "content": "s"}, {"role": "user", "content": "draft"}]
        p = prefill_warm.probe_messages(m)
        self.assertEqual(p[-1]["content"], "draft" + prefill_warm.PROBE)
        self.assertEqual(m[-1]["content"], "draft")                          # the input is not changed
        parts = [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]
        self.assertTrue(prefill_warm.probe_messages(parts)[-1]["content"][-1]["text"].startswith("b"))
        self.assertIsNone(prefill_warm.probe_messages([{"role": "assistant", "content": "x"}]))
        self.assertIsNone(prefill_warm.probe_messages([{"role": "user", "content": [
            {"type": "text", "text": "a"}, {"type": "image_url", "image_url": {"url": "u"}}]}]))
        self.assertIsNone(prefill_warm.probe_messages([]))

    def test_plan_pins_the_stable_prefix(self):
        # a tokenizer that makes one token per character, plus a twin that appends the probe: the draft is a prefix
        # (plus the template's fixed tail after the user text, so the draft is not a prefix of its probe twin)
        enc = lambda msgs, tools, kw: list("".join(str(x["content"]) for x in msgs) + "<|im_end|>")   # noqa: E731
        msgs = [{"role": "user", "content": "x" * 800}]
        p = prefill_warm.plan(msgs, None, {}, enc, min_tok=512)
        self.assertIsNone(p.skip)
        self.assertEqual(p.prompt_tokens, 810)
        self.assertEqual(p.tokens, 799)                                    # the whole draft is shared, minus one token
        # under the floor: not warmed
        short = prefill_warm.plan([{"role": "user", "content": "x" * 100}], None, {}, enc, min_tok=512)
        self.assertIsNotNone(short.skip)
        self.assertFalse(short.body()["warmed"])
        # a body that pins by itself keeps its pin and is still read
        pinned = prefill_warm.plan(msgs, None, {}, enc, min_tok=512, already_pinned=True)
        self.assertIsNone(pinned.skip)
        req = prefill_warm.apply({"messages": msgs, "stream": True, "max_tokens": 900, "max_completion_tokens": 9,
                                  "strata_prefix": {"messages": 1}}, pinned)
        self.assertEqual(req["strata_prefix"], {"messages": 1})
        req = prefill_warm.apply({"messages": msgs, "stream": True, "max_tokens": 900, "max_completion_tokens": 9}, p)
        self.assertEqual((req["max_tokens"], req["stream"], req["strata_prefix"]), (1, False, {"tokens": 799}))
        self.assertNotIn("max_completion_tokens", req)

    def test_gate_cancels_queued_warms_only(self):
        g = prefill_warm.Gate()
        a, b, c = threading.Event(), threading.Event(), threading.Event()
        g.begin(a)
        g.begin(b)                                  # a newer warm: the queued older one gives its place up
        self.assertTrue(a.is_set())
        self.assertFalse(b.is_set())
        running = {id(b): False}
        g.probe = lambda ev: running.get(id(ev), True)   # b has begun reading: it is left to end
        g.note_real()
        self.assertFalse(b.is_set())
        g.begin(c)
        self.assertFalse(b.is_set())                # a running warm is never cancelled by a newer one
        g.probe = lambda ev: ev is not b
        g.note_real()                               # a real request: the queued c goes
        self.assertTrue(c.is_set())
        g.end(b)
        self.assertEqual(g.stats["cancelled_by_real"], 1)


class RecordingEngine(MockEngine):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.calls = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.calls.append({"n": len(ids), "max_new": max_new, "sampling": dict(sampling or {})})
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class Wiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = RecordingEngine(tok, "</think>\n\nok", max_context=8192)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.svc.loop_guard = False
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        self.engine.calls.clear()

    def post(self, path, body, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def test_default_off_never_touches_the_new_code(self):
        boom = mock.Mock(side_effect=AssertionError("the prefill warm code ran with the switch off"))
        with mock.patch.dict(os.environ, {}, clear=False), \
                mock.patch.object(prefill_warm, "plan", boom), mock.patch.object(prefill_warm, "apply", boom), \
                mock.patch.object(prefill_warm.GATE, "begin", boom), mock.patch.object(prefill_warm.GATE, "note_real", boom):
            os.environ.pop("STRATA_PREFILL_TYPING", None)
            s, _ = self.post("/v1/prefill", {"messages": [{"role": "user", "content": "x" * 900}]})
            self.assertEqual(s, 404)                                          # no such endpoint, as before
            self.assertEqual(self.engine.calls, [])
            # a chat completion that carries the key is a normal answer, and the key is not acted on
            s, b = self.post("/v1/chat/completions", {"model": "m", "max_tokens": 8, "strata_prefill": True,
                                                      "messages": [{"role": "user", "content": "hi"}]},
                             {"X-Strata-Prefill": "1"})
            self.assertEqual(s, 200)
            self.assertNotIn("strata_prefill", b)
            self.assertEqual(self.engine.calls[0]["max_new"], 8)
            self.assertNotIn("strata_prefix", self.engine.calls[0]["sampling"])

    def test_on_a_draft_is_read_as_one_token_with_the_prefix_pinned(self):
        with mock.patch.dict(os.environ, {"STRATA_PREFILL_TYPING": "1", "STRATA_PREFILL_TYPING_MIN": "100"}):
            s, b = self.post("/v1/prefill", {"model": "m", "stream": True, "max_tokens": 400,
                                             "messages": [{"role": "user", "content": "x" * 900}]})
            self.assertEqual(s, 200)
            self.assertTrue(b["strata_prefill"]["warmed"])
            n = b["strata_prefill"]["stable_tokens"]
            self.assertGreaterEqual(n, 100)
            self.assertLess(n, b["strata_prefill"]["prompt_tokens"])
            call = self.engine.calls[-1]
            self.assertEqual(call["max_new"], 1)
            self.assertEqual(call["sampling"].get("strata_prefix"), {"tokens": n})
            # a short draft is not read at all (the template's own 227 tokens come before it: the floor counts them)
            self.engine.calls.clear()
            os.environ["STRATA_PREFILL_TYPING_MIN"] = "400"
            s, b = self.post("/v1/prefill", {"model": "m", "messages": [{"role": "user", "content": "hello"}]})
            self.assertEqual(s, 200)
            self.assertFalse(b["warmed"])
            self.assertEqual(self.engine.calls, [])
            # the same through a chat completion body with the header
            s, b = self.post("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "y" * 900}]},
                             {"X-Strata-Prefill": "1"})
            self.assertEqual(s, 200)
            self.assertTrue(b["strata_prefill"]["warmed"])
            self.assertEqual(self.engine.calls[-1]["max_new"], 1)

    def test_on_a_real_request_is_not_changed(self):
        with mock.patch.dict(os.environ, {"STRATA_PREFILL_TYPING": "1"}):
            s, b = self.post("/v1/chat/completions", {"model": "m", "max_tokens": 8,
                                                      "messages": [{"role": "user", "content": "z" * 900}]})
            self.assertEqual(s, 200)
            self.assertNotIn("strata_prefill", b)
            self.assertEqual(self.engine.calls[-1]["max_new"], 8)
            self.assertNotIn("strata_prefix", self.engine.calls[-1]["sampling"])


if __name__ == "__main__":
    unittest.main()
