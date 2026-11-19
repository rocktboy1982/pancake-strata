"""serve/test_grammar_routes.py - item 89: with --grammar on, every API route hands the engine the tool-call grammar
(it was only set on /v1/chat/completions; /v1/messages and /v1/responses sent none).  Mock engine, no GPU.

    python -m unittest serve.test_grammar_routes -v
"""
from __future__ import annotations

import json
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}


class RecordingEngine(MockEngine):
    grammar = True                                       # as the engine started with --grammar on

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_sampling = dict(sampling or {})
        return super().generate(ids, max_new, sampling, cancel, embeddings)


class GrammarRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = RecordingEngine(tok, "ok", max_context=8192)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()

    def grammar_sent(self):
        spec = self.engine.last_sampling.get("strata_grammar")
        self.assertIsNotNone(spec, "the route sent no grammar to the engine")
        self.assertEqual(spec["kind"], "tools")
        self.assertEqual([t["function"]["name"] for t in spec["tools"]], ["read"])
        return spec

    def test_chat_completions(self):
        status, raw = self.post("/v1/chat/completions", {
            "model": "m", "max_tokens": 32, "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "read", "description": "d", "parameters": SCHEMA}}]})
        self.assertEqual(status, 200, raw)
        self.grammar_sent()

    def test_anthropic_messages(self):
        status, raw = self.post("/v1/messages", {
            "model": "m", "max_tokens": 32, "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "read", "description": "d", "input_schema": SCHEMA}],
            "tool_choice": {"type": "tool", "name": "read"}})
        self.assertEqual(status, 200, raw)
        spec = self.grammar_sent()
        self.assertEqual(spec["tool_choice"], {"type": "tool", "name": "read"})   # the engine reads the name

    def test_responses(self):
        status, raw = self.post("/v1/responses", {
            "model": "m", "max_output_tokens": 32, "input": "hi",
            "tools": [{"type": "function", "name": "read", "description": "d", "parameters": SCHEMA}]})
        self.assertEqual(status, 200, raw)
        self.grammar_sent()

    def test_no_tools_no_grammar(self):
        status, raw = self.post("/v1/messages", {"model": "m", "max_tokens": 32,
                                                 "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200, raw)
        self.assertIsNone(self.engine.last_sampling.get("strata_grammar"))


if __name__ == "__main__":
    unittest.main()
