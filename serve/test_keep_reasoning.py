"""serve/test_keep_reasoning.py - item 31: chat_template_kwargs.preserve_thinking reaches the template on the OpenAI,
Anthropic and Responses routes, and the opt-in reasoning keeper ("keep_reasoning") puts the server's own reasoning back
into a history whose client dropped it, so the next prompt starts with exactly what the engine cached.  Mock engine,
no GPU.

    python -m unittest serve.test_keep_reasoning -v
"""
from __future__ import annotations

import json
import sys
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import responses as responses_api  # noqa: E402
from serve.frontend import (ChatTemplate, ReasoningKeeper, anthropic_to_messages,  # noqa: E402
                            openai_to_messages)
from serve.server import IM_END, ByteTokenizer, MockEngine, Service, keeper_of, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OLD_THOUGHT = "an old line of thought"


def post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


class Passthrough(unittest.TestCase):
    """preserve_thinking from chat_template_kwargs into the template kwargs (all three request readers)."""

    def chat(self, ctk=None):
        req = {"messages": [{"role": "user", "content": "hi"}]}
        if ctk is not None:
            req["chat_template_kwargs"] = ctk
        return openai_to_messages(req)[2]

    def test_openai(self):
        self.assertNotIn("preserve_thinking", self.chat())
        self.assertNotIn("preserve_thinking", self.chat({"reasoning_effort": "low"}))
        self.assertIs(self.chat({"preserve_thinking": False})["preserve_thinking"], False)
        self.assertIs(self.chat({"preserve_thinking": True})["preserve_thinking"], True)
        # enable_thinking=false replaced the kwargs; preserve_thinking still comes through
        kw = self.chat({"preserve_thinking": False, "enable_thinking": False})
        self.assertEqual(kw, {"enable_thinking": False, "preserve_thinking": False})
        with self.assertRaises(ValueError):
            self.chat({"preserve_thinking": "no"})

    def test_anthropic(self):
        req = {"messages": [{"role": "user", "content": "hi"}]}
        self.assertNotIn("preserve_thinking", anthropic_to_messages(dict(req))[2])
        kw = anthropic_to_messages({**req, "chat_template_kwargs": {"preserve_thinking": False}})[2]
        self.assertIs(kw["preserve_thinking"], False)
        with self.assertRaises(ValueError):
            anthropic_to_messages({**req, "chat_template_kwargs": {"preserve_thinking": 0}})

    def test_responses(self):
        self.assertNotIn("preserve_thinking", responses_api.template_kwargs({"input": "hi"}, {}))
        kw = responses_api.template_kwargs({"input": "hi", "chat_template_kwargs": {"preserve_thinking": False}}, {})
        self.assertIs(kw["preserve_thinking"], False)
        with self.assertRaises(responses_api.ResponsesError):
            responses_api.template_kwargs({"input": "hi", "chat_template_kwargs": {"preserve_thinking": None}}, {})


class PassthroughRendered(unittest.TestCase):
    """End to end: an earlier turn's reasoning (before the last user query) is in the prompt by default and left out
    with preserve_thinking false, on each route."""

    @classmethod
    def setUpClass(cls):
        cls.tok = ByteTokenizer()
        cls.engine = MockEngine(cls.tok, "hm\n</think>\n\nok", max_context=16384)
        cls.svc = Service(cls.engine, cls.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def prompt(self, path, body):
        post(self.base, path, body)
        return self.tok.decode(self.engine.last_prompt)

    def check(self, path, body):
        self.assertIn(OLD_THOUGHT, self.prompt(path, body))
        self.assertNotIn(OLD_THOUGHT, self.prompt(path, {**body, "chat_template_kwargs": {"preserve_thinking": False}}))
        self.assertIn(OLD_THOUGHT, self.prompt(path, {**body, "chat_template_kwargs": {"preserve_thinking": True}}))

    def test_openai(self):
        self.check("/v1/chat/completions", {"model": "m", "max_tokens": 50, "messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "reasoning_content": OLD_THOUGHT},
            {"role": "user", "content": "q2"}]})

    def test_anthropic(self):
        self.check("/v1/messages", {"model": "m", "max_tokens": 50, "messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": OLD_THOUGHT},
                                              {"type": "text", "text": "a1"}]},
            {"role": "user", "content": "q2"}]})

    def test_responses(self):
        self.check("/v1/responses", {"model": "m", "max_output_tokens": 50, "input": [
            {"type": "message", "role": "user", "content": "q1"},
            {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": OLD_THOUGHT}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "a1"}]},
            {"type": "message", "role": "user", "content": "q2"}]})


class KeeperUnit(unittest.TestCase):
    def test_fill_by_ids_then_body(self):
        k = ReasoningKeeper()
        k.remember("", [{"id": "call_1", "name": "read", "arguments": {"path": "a"}}], "think A")
        # ids match even when the client re-serialized the arguments differently
        m = [{"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "read", "arguments": {"path": "a"}}, "_call_id": "call_1"}]}]
        self.assertEqual(k.fill(m), 1)
        self.assertEqual(m[0]["reasoning_content"], "think A")
        # no ids (a client that renumbers them): found by text + calls
        m = [{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read", "arguments": {"path": "a"}}}]}]
        self.assertEqual(k.fill(m), 1)
        # other arguments: not this turn
        m = [{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read", "arguments": {"path": "b"}}}]}]
        self.assertEqual(k.fill(m), 0)
        self.assertNotIn("reasoning_content", m[0])

    def test_client_reasoning_wins_and_text_turns(self):
        k = ReasoningKeeper()
        k.remember("  The answer.\n", [], "mine")
        m = [{"role": "assistant", "content": "The answer.", "reasoning_content": "client's"},
             {"role": "assistant", "content": "The answer."},
             {"role": "assistant", "content": "Another answer."},
             {"role": "user", "content": "The answer."}]
        self.assertEqual(k.fill(m), 1)
        self.assertEqual(m[0]["reasoning_content"], "client's")
        self.assertEqual(m[1]["reasoning_content"], "mine")
        self.assertNotIn("reasoning_content", m[2])
        self.assertNotIn("reasoning_content", m[3])

    def test_nothing_kept_for_empty(self):
        k = ReasoningKeeper()
        k.remember("", [], "thought")           # no text, no calls: nothing tells it apart
        k.remember("answer", [], "   ")         # no reasoning
        self.assertEqual(len(k), 0)

    def test_lru_bounds(self):
        k = ReasoningKeeper(max_entries=3)
        for i in range(5):
            k.remember(f"a{i}", [], f"r{i}")
        self.assertEqual(len(k), 3)
        m = [{"role": "assistant", "content": f"a{i}"} for i in range(5)]
        self.assertEqual(k.fill(m), 3)
        self.assertEqual([x.get("reasoning_content") for x in m], [None, None, "r2", "r3", "r4"])
        k = ReasoningKeeper(max_chars=10)
        k.remember("a", [], "x" * 6)
        k.remember("b", [], "y" * 6)              # 12 chars > 10: the oldest goes
        self.assertEqual(len(k), 1)
        self.assertEqual(k._chars, 6)
        self.assertEqual(len(k._keys), 1)

    def test_same_body_other_ids(self):
        k = ReasoningKeeper()
        call = {"name": "ls", "arguments": {}}
        k.remember("", [{**call, "id": "c1"}], "first")
        k.remember("", [{**call, "id": "c2"}], "second")
        m = [{"role": "assistant", "content": "", "tool_calls": [{"function": call, "_call_id": "c1"}]},
             {"role": "assistant", "content": "", "tool_calls": [{"function": call, "_call_id": "c2"}]},
             {"role": "assistant", "content": "", "tool_calls": [{"function": call}]}]
        k.fill(m)
        self.assertEqual([x["reasoning_content"] for x in m], ["first", "second", "second"])

    def test_config(self):
        self.assertIsNone(keeper_of(None, None))
        self.assertIsNone(keeper_of(None, False))
        self.assertEqual(keeper_of(None, True).max_entries, 512)
        self.assertEqual(keeper_of(None, 64).max_entries, 64)
        self.assertEqual(keeper_of("1", None).max_entries, 512)
        self.assertIsNone(keeper_of("0", True))
        self.assertEqual(keeper_of("100", None).max_entries, 100)
        with self.assertRaises(SystemExit):
            keeper_of(None, "yes")


THOUGHT = "The user wants 2+2. That is 4."
ANSWER_SCRIPT = THOUGHT + "\n</think>\n\nThe answer is 4."
CALL_SCRIPT = ("I should read the file first.\n</think>\n\n<tool_call>\n<function=read>\n<parameter=path>\na.txt\n"
               "</parameter>\n</function>\n</tool_call>")
TOOLS = [{"type": "function", "function": {"name": "read", "description": "read a file", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]


class KeeperEndToEnd(unittest.TestCase):
    """The second request sends the first reply back without its reasoning.  With the keeper, its prompt starts with
    the first prompt + the exact text the engine generated (what the conversation cache holds); without it, the prompt
    leaves that prefix at the dropped <think> block."""

    def start(self, script, keep):
        self.tok = ByteTokenizer()
        self.engine = MockEngine(self.tok, script, max_context=16384)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.reasoning_keeper = ReasoningKeeper() if keep else None
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def cached(self, script):
        """What the engine holds after the first request: its prompt and the reply it generated, end token included."""
        return self.tok.decode(self.engine.last_prompt) + script + IM_END

    def openai_answer(self, keep):
        self.start([ANSWER_SCRIPT, "fine"], keep)
        first = [{"role": "user", "content": "what is 2+2?"}]
        out = post(self.base, "/v1/chat/completions", {"model": "m", "max_tokens": 200, "messages": first})
        msg = out["choices"][0]["message"]
        self.assertEqual(msg["reasoning_content"].strip(), THOUGHT)
        cached = self.cached(ANSWER_SCRIPT)
        # the client sends the turn back as opencode would without reasoning: content only
        post(self.base, "/v1/chat/completions", {"model": "m", "max_tokens": 200, "messages": first + [
            {"role": "assistant", "content": msg["content"]}, {"role": "user", "content": "thanks"}]})
        return cached, self.tok.decode(self.engine.last_prompt)

    def test_openai_answer_with_keeper(self):
        cached, second = self.openai_answer(True)
        self.assertTrue(second.startswith(cached), (cached[-120:], second[len(cached) - 120:len(cached) + 40]))
        self.assertEqual(self.svc.reasoning_keeper.hits, 1)

    def test_openai_answer_without_keeper(self):
        cached, second = self.openai_answer(False)
        self.assertFalse(second.startswith(cached))
        self.assertIn("<think>\n\n</think>\n\nThe answer is 4.", second)

    def test_openai_tool_call(self):
        self.start([CALL_SCRIPT, "done"], True)
        first = [{"role": "user", "content": "show a.txt"}]
        out = post(self.base, "/v1/chat/completions", {"model": "m", "max_tokens": 300, "messages": first,
                                                        "tools": TOOLS})
        msg = out["choices"][0]["message"]
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "read")
        cached = self.cached(CALL_SCRIPT)
        back = {"role": "assistant", "content": msg.get("content") or "", "tool_calls": msg["tool_calls"]}
        post(self.base, "/v1/chat/completions", {"model": "m", "max_tokens": 300, "tools": TOOLS, "messages": first + [
            back, {"role": "tool", "tool_call_id": msg["tool_calls"][0]["id"], "content": "hello"}]})
        second = self.tok.decode(self.engine.last_prompt)
        self.assertTrue(second.startswith(cached), (cached[-160:], second[len(cached) - 160:len(cached) + 40]))
        # the same history with the reasoning sent by the client renders the same prompt
        back_r = {**back, "reasoning_content": "I should read the file first."}
        post(self.base, "/v1/chat/completions", {"model": "m", "max_tokens": 300, "tools": TOOLS, "messages": first + [
            back_r, {"role": "tool", "tool_call_id": msg["tool_calls"][0]["id"], "content": "hello"}]})
        self.assertEqual(self.tok.decode(self.engine.last_prompt), second)

    def test_openai_streamed(self):
        """opencode streams: the keeper sees the same events on the streamed path."""
        self.start([ANSWER_SCRIPT, "fine"], True)
        first = [{"role": "user", "content": "what is 2+2?"}]
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(
            {"model": "m", "max_tokens": 200, "stream": True, "messages": first}).encode(),
            headers={"Content-Type": "application/json"})
        content = []
        with urllib.request.urlopen(req, timeout=20) as r:
            for line in r.read().decode().splitlines():
                if line.startswith("data: {"):
                    for ch in json.loads(line[6:]).get("choices") or []:
                        content.append((ch.get("delta") or {}).get("content") or "")
        cached = self.cached(ANSWER_SCRIPT)
        post(self.base, "/v1/chat/completions", {"model": "m", "max_tokens": 200, "messages": first + [
            {"role": "assistant", "content": "".join(content)}, {"role": "user", "content": "thanks"}]})
        self.assertTrue(self.tok.decode(self.engine.last_prompt).startswith(cached))

    def test_anthropic_tool_call(self):
        self.start([CALL_SCRIPT, "done"], True)
        tools = [{"name": "read", "description": "read a file", "input_schema": TOOLS[0]["function"]["parameters"]}]
        first = [{"role": "user", "content": "show a.txt"}]
        out = post(self.base, "/v1/messages", {"model": "m", "max_tokens": 300, "messages": first, "tools": tools})
        use = next(b for b in out["content"] if b["type"] == "tool_use")
        cached = self.cached(CALL_SCRIPT)
        # Claude Code-like history without the thinking block
        post(self.base, "/v1/messages", {"model": "m", "max_tokens": 300, "tools": tools, "messages": first + [
            {"role": "assistant", "content": [use]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": use["id"], "content": "hello"}]}]})
        second = self.tok.decode(self.engine.last_prompt)
        self.assertTrue(second.startswith(cached), (cached[-160:], second[len(cached) - 160:len(cached) + 40]))

    def test_responses_tool_call(self):
        self.start([CALL_SCRIPT, "done"], True)
        tools = [{"type": "function", "name": "read", "description": "read a file",
                  "parameters": TOOLS[0]["function"]["parameters"]}]
        first = [{"type": "message", "role": "user", "content": "show a.txt"}]
        out = post(self.base, "/v1/responses", {"model": "m", "max_output_tokens": 300, "input": first, "tools": tools})
        call = next(i for i in out["output"] if i["type"] == "function_call")
        cached = self.cached(CALL_SCRIPT)
        # a client that sends the call back without the reasoning item
        post(self.base, "/v1/responses", {"model": "m", "max_output_tokens": 300, "tools": tools, "input": first + [
            {"type": "function_call", "call_id": call["call_id"], "name": "read", "arguments": call["arguments"]},
            {"type": "function_call_output", "call_id": call["call_id"], "output": "hello"}]})
        second = self.tok.decode(self.engine.last_prompt)
        self.assertTrue(second.startswith(cached), (cached[-160:], second[len(cached) - 160:len(cached) + 40]))

    def test_off_by_default(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "x"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.assertIsNone(svc.reasoning_keeper)


if __name__ == "__main__":
    unittest.main()
