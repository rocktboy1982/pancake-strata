"""serve/test_canonical_tools.py - config "canonical_tools" (ported from DevDock-AI/Speedy 2bc7a7e06f71, LEADS.md):
the same tools in another order or key order render the same prompt (so the cached prompt start is kept); off by
default, the request's order is rendered as sent.

    python -m unittest serve.test_canonical_tools -v
"""
from __future__ import annotations

import os
import sys
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate, anthropic_to_messages, anthropic_tools  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, canonical_tools, canonical_tools_setting  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MSGS = [{"role": "user", "content": "hi"}]
READ = {"type": "function", "function": {"name": "read_file", "description": "Read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
                       "required": ["path"]}}}
READ_REORDERED = {"function": {"parameters": {"required": ["path"], "properties": {"limit": {"type": "integer"},
                  "path": {"type": "string"}}, "type": "object"}, "description": "Read a file", "name": "read_file"},
                  "type": "function"}
WRITE = {"type": "function", "function": {"name": "write_file", "description": "Write a file",
         "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "text": {"type": "string"}}}}}


def service(on):
    tok = ByteTokenizer()
    svc = Service(MockEngine(tok, "ok", max_context=8192), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
    if on is not None:
        svc.canonical_tools = on
    return svc


def render(svc, tools, messages=MSGS, kw=None):
    return svc.tok.decode(svc.encode_prompt(messages, tools, kw or {}, store=False))


class Canonical(unittest.TestCase):
    def test_default_is_off(self):
        self.assertFalse(service(None).canonical_tools)

    def test_off_is_the_old_render(self):
        # the default Service renders exactly what an explicit off does, for any order
        for tools in ([READ, WRITE], [WRITE, READ_REORDERED]):
            self.assertEqual(render(service(None), tools), render(service(False), tools))

    def test_on_same_prompt(self):
        svc = service(True)
        self.assertEqual(render(svc, [READ, WRITE]), render(svc, [WRITE, READ_REORDERED]))

    def test_off_order_kept(self):
        svc = service(False)
        a, b = render(svc, [READ, WRITE]), render(svc, [WRITE, READ])
        self.assertNotEqual(a, b)
        self.assertLess(b.index("write_file"), b.index("read_file"))

    def test_only_order_changes(self):
        out = canonical_tools([WRITE, READ_REORDERED])
        self.assertEqual([t["function"]["name"] for t in out], ["read_file", "write_file"])
        self.assertEqual(out[0], READ)             # dict equality: same content, only key order differs
        self.assertEqual(list(out[0]["function"]["parameters"]["properties"]), ["limit", "path"])
        self.assertEqual(canonical_tools(None), None)
        flat = [{"name": "b"}, {"name": "a", "x": 1}, {"name": "a", "x": 2}]   # stable for equal names
        self.assertEqual(canonical_tools(flat), [{"name": "a", "x": 1}, {"name": "a", "x": 2}, {"name": "b"}])

    def test_request_tools_untouched(self):
        tools = [WRITE, READ]
        render(service(True), tools)
        self.assertEqual(tools, [WRITE, READ])

    def test_anthropic_route(self):
        def req(tools):
            return {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}], "tools": tools}
        a = [{"name": "read_file", "description": "r", "input_schema": {"type": "object", "properties": {"p": {"type": "string"}}}},
             {"name": "write_file", "description": "w", "input_schema": {"properties": {"q": {"type": "string"}}, "type": "object"}}]
        b = [{"input_schema": {"type": "object", "properties": {"q": {"type": "string"}}}, "name": "write_file", "description": "w"},
             {"description": "r", "input_schema": {"properties": {"p": {"type": "string"}}, "type": "object"}, "name": "read_file"}]
        outs = {}
        for on in (True, False):
            svc = service(on)
            for key, tools in (("a", a), ("b", b)):
                messages, t, kw = anthropic_to_messages(req(tools))
                self.assertIsNotNone(anthropic_tools(req(tools))[0])
                outs[on, key] = render(svc, t, messages, kw)
        self.assertEqual(outs[True, "a"], outs[True, "b"])
        self.assertNotEqual(outs[False, "a"], outs[False, "b"])

    def test_setting(self):
        with unittest.mock.patch.dict(os.environ, {"STRATA_CANONICAL_TOOLS": ""}):
            self.assertFalse(canonical_tools_setting({}))
            self.assertTrue(canonical_tools_setting({"canonical_tools": True}))
        with unittest.mock.patch.dict(os.environ, {"STRATA_CANONICAL_TOOLS": "1"}):
            self.assertTrue(canonical_tools_setting({}))
            self.assertFalse(canonical_tools_setting({"canonical_tools": False}))   # the config wins


if __name__ == "__main__":
    unittest.main()
