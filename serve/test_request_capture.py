"""serve/test_request_capture.py - M4: the opt-in STRATA_REQUEST_CAPTURE=DIR record of raw chat requests and the replies,
and tools/pancake/prefix_check.py reading it.  Mock engine, no GPU.

    nice -n 19 python3 -I -m unittest serve.test_request_capture -v
"""
from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "pancake"))
import prefix_check  # noqa: E402
from serve import request_capture  # noqa: E402
from serve.frontend import ChatTemplate, ReasoningKeeper  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

THOUGHT = "I should read the file first."
CALL_SCRIPT = THOUGHT + "\n</think>\n\n<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>"
TOOLS = [{"type": "function", "function": {"name": "read", "description": "read a file", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
A_TOOLS = [{"name": "read", "description": "read a file", "input_schema": TOOLS[0]["function"]["parameters"]}]


def post(base, path, body, headers=None, stream=False):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=20) as r:
        raw = r.read().decode()
    return raw if stream else json.loads(raw)


def lines_of(directory, want=None):
    """The captured lines.  The server writes a line when the request handler ends, which can be a moment after the client
    has its answer: with `want`, wait (up to 3 s) for that many."""
    deadline = time.time() + 3
    while True:
        out = _read_lines(directory)
        if want is None or len(out) >= want or time.time() > deadline:
            return out
        time.sleep(0.02)


def _read_lines(directory):
    files = sorted(Path(directory).glob("requests-*.jsonl"))
    out = []
    for f in files:
        out += [json.loads(x) for x in f.read_text().splitlines() if x]
    return out


class Unit(unittest.TestCase):
    def test_off_by_default(self):
        self.assertIsNone(request_capture.from_env({}))
        self.assertIsNone(request_capture.from_env({"STRATA_REQUEST_CAPTURE": "  "}))

    def test_directory_is_private_and_headers_are_filtered(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap = request_capture.from_env({"STRATA_REQUEST_CAPTURE": os.path.join(tmp, "cap")})
            self.assertEqual(stat.S_IMODE(os.stat(os.path.join(tmp, "cap")).st_mode), 0o700)
            entry = cap.begin("/v1/chat/completions", {"b": 1, "a": 2, "messages": []},
                              {"Authorization": "Bearer SECRET", "Cookie": "c=1", "User-Agent": "opencode/1.0",
                               "X-Session-Affinity": "ses_42", "x-api-key": "SECRET", "Accept": "*/*"})
            self.assertEqual(entry["session"], "ses_42")
            self.assertEqual(entry["client"], {"user-agent": "opencode/1.0", "x-session-affinity": "ses_42"})
            self.assertIsNone(cap.begin("/v1/responses", {}, None))
            self.assertIsNone(cap.begin("/v1/messages/count_tokens", {}, None))
            cap.finish(entry)
            self.assertEqual(stat.S_IMODE(os.stat(cap.path).st_mode), 0o600)
            text = Path(cap.path).read_text()
            self.assertNotIn("SECRET", text)
            self.assertIn('"request":{"b":1,"a":2,"messages":[]}', text)      # the client's key order

    def test_size_limit_stops_the_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap = request_capture.from_env({"STRATA_REQUEST_CAPTURE": tmp, "STRATA_REQUEST_CAPTURE_MAX_MB": "0.0005"})
            for _ in range(5):
                entry = cap.begin("/v1/chat/completions", {"messages": [{"role": "user", "content": "x" * 200}]})
                cap.finish(entry)
            self.assertTrue(cap.off)
            self.assertLess(len(lines_of(tmp)), 5)

    def test_observe_openai_and_anthropic(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap = request_capture.RequestCapture(tmp)
            e = cap.begin("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
            for chunk in ({"choices": [{"delta": {"reasoning_content": "th"}}]},
                          {"choices": [{"delta": {"reasoning_content": "ink"}}]},
                          None,
                          {"choices": [{"delta": {"content": "ok"}}]},
                          {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_a", "function": {"name": "read", "arguments": ""}}]}}]},
                          {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"path": '}}]}}]},
                          {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"a"}'}}]}}]},
                          {"choices": [{"delta": {}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 5}}):
                cap.observe(e, "openai", chunk)
            msg = cap.response_message(e)
            self.assertEqual(msg["reasoning_content"], "think")
            self.assertEqual(msg["content"], "ok")
            self.assertEqual(msg["tool_calls"][0], {"id": "call_a", "type": "function",
                                                    "function": {"name": "read", "arguments": '{"path": "a"}'}})
            self.assertEqual((e["finish"], e["usage"]), ("tool_calls", {"prompt_tokens": 5}))
            e = cap.begin("/v1/messages", {"messages": [{"role": "user", "content": "hi"}]})
            for item in (("message_start", {"message": {}}),
                         ("content_block_start", {"content_block": {"type": "thinking", "thinking": "", "signature": ""}}),
                         ("content_block_delta", {"delta": {"type": "thinking_delta", "thinking": "hmm"}}),
                         ("content_block_start", {"content_block": {"type": "tool_use", "id": "t1", "name": "read", "input": {}}}),
                         ("content_block_delta", {"delta": {"type": "input_json_delta", "partial_json": '{"path":'}}),
                         ("content_block_delta", {"delta": {"type": "input_json_delta", "partial_json": ' "a"}'}}),
                         ("message_delta", {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 3}})):
                cap.observe(e, "anthropic", item)
            msg = cap.response_message(e)
            self.assertEqual([b["type"] for b in msg["content"]], ["thinking", "tool_use"])
            self.assertEqual(msg["content"][1]["input"], {"path": "a"})
            self.assertEqual(msg["content"][0]["thinking"], "hmm")

    def test_a_bad_item_never_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap = request_capture.RequestCapture(tmp)
            e = cap.begin("/v1/chat/completions", {"messages": []})
            cap.observe(e, "openai", {"choices": "not a list"})
            cap.observe(e, "anthropic", "not a pair")
            self.assertTrue(e.get("observe_error"))


class EndToEnd(unittest.TestCase):
    def start(self, scripts, capture_dir, keep=False):
        self.tok = ByteTokenizer()
        self.engine = MockEngine(self.tok, scripts, max_context=16384)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.reasoning_keeper = ReasoningKeeper() if keep else None
        if capture_dir:
            self.svc.request_capture = request_capture.RequestCapture(capture_dir)
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def test_default_writes_nothing_and_answers_the_same(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.start([CALL_SCRIPT, "done"], None)
            self.assertIsNone(self.svc.request_capture)
            out = post(self.base, "/v1/chat/completions", {"model": "m", "max_tokens": 300, "tools": TOOLS,
                                                            "messages": [{"role": "user", "content": "show a.txt"}]})
            self.assertEqual(out["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "read")
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def conversation(self, tmp, resend_reasoning, stream=False):
        self.start([CALL_SCRIPT, "done"], tmp)
        first = [{"role": "user", "content": "show a.txt"}]
        body = {"model": "m", "max_tokens": 300, "tools": TOOLS, "messages": first}
        hdr = {"User-Agent": "test-client/1", "Authorization": "Bearer SECRET-KEY", "X-Session-Id": "s-1"}
        if stream:
            raw = post(self.base, "/v1/chat/completions", {**body, "stream": True}, hdr, stream=True)
            msg = {"role": "assistant", "content": "", "tool_calls": []}
            reasoning = ""
            for line in raw.splitlines():
                if line.startswith("data: {"):
                    for ch in json.loads(line[6:]).get("choices") or []:
                        d = ch.get("delta") or {}
                        reasoning += d.get("reasoning_content") or ""
                        for c in d.get("tool_calls") or []:
                            if c.get("id"):
                                msg["tool_calls"].append({"id": c["id"], "type": "function",
                                                          "function": {"name": c["function"]["name"], "arguments": ""}})
                            msg["tool_calls"][c["index"]]["function"]["arguments"] += c["function"].get("arguments") or ""
        else:
            msg = post(self.base, "/v1/chat/completions", body, hdr)["choices"][0]["message"]
            reasoning = msg.get("reasoning_content") or ""
        back = {"role": "assistant", "content": msg.get("content") or "", "tool_calls": msg["tool_calls"]}
        if resend_reasoning:
            back["reasoning_content"] = reasoning
        post(self.base, "/v1/chat/completions", {**body, "messages": first + [
            back, {"role": "tool", "tool_call_id": msg["tool_calls"][0]["id"], "content": "hello"}]}, hdr)
        return reasoning

    def test_capture_lines_and_the_checker(self):
        for stream in (False, True):
            for resend in (True, False):
                with self.subTest(stream=stream, resend_reasoning=resend), tempfile.TemporaryDirectory() as tmp:
                    reasoning = self.conversation(tmp, resend, stream)
                    self.assertTrue(reasoning.strip().startswith(THOUGHT[:10]))
                    recs = lines_of(tmp, 2)
                    self.assertEqual(len(recs), 2)
                    self.assertEqual(recs[0]["session"], "s-1")
                    self.assertEqual(recs[0]["client"], {"user-agent": "test-client/1", "x-session-id": "s-1"})
                    self.assertEqual(recs[0]["http_status"], 200)
                    self.assertNotIn("SECRET-KEY", Path(next(Path(tmp).glob("*.jsonl"))).read_text())
                    self.assertEqual(recs[0]["request"]["messages"], [{"role": "user", "content": "show a.txt"}])
                    reply = recs[0]["response"]["message"]
                    self.assertEqual(reply["tool_calls"][0]["function"]["name"], "read")
                    self.assertEqual(reply["reasoning_content"].strip(), reasoning.strip())
                    self.assertIn("a.txt", reply["tool_calls"][0]["function"]["arguments"])
                    self.assertEqual(recs[0]["response"]["finish_reason"], "tool_calls")
                    # the checker reads the file the server wrote
                    files = [str(p) for p in Path(tmp).glob("*.jsonl")]
                    _, results = prefix_check.run(prefix_check.load(files))
                    self.assertEqual(len(results), 1)
                    res = results[0][0]
                    if resend:
                        self.assertEqual(res["verdict"], "reusable", res["first_divergence"])
                    else:
                        self.assertEqual(res["first_divergence"]["kind"], "reasoning_dropped")

    def test_anthropic_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.start([CALL_SCRIPT, "done"], tmp)
            first = [{"role": "user", "content": "show a.txt"}]
            body = {"model": "m", "max_tokens": 300, "tools": A_TOOLS, "messages": first,
                    "system": [{"type": "text", "text": "You are an agent."}]}
            out = post(self.base, "/v1/messages", body)
            use = next(b for b in out["content"] if b["type"] == "tool_use")
            thinking = next(b for b in out["content"] if b["type"] == "thinking")
            post(self.base, "/v1/messages", {**body, "messages": first + [
                {"role": "assistant", "content": [thinking, use]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": use["id"], "content": "hello"}]}]})
            recs = lines_of(tmp, 2)
            self.assertEqual([r["path"] for r in recs], ["/v1/messages", "/v1/messages"])
            reply = recs[0]["response"]["message"]
            self.assertEqual([b["type"] for b in reply["content"]], ["thinking", "tool_use"])
            self.assertEqual(reply["content"][1]["input"], {"path": "a.txt"})
            _, results = prefix_check.run(prefix_check.load([str(p) for p in Path(tmp).glob("*.jsonl")]), link="consecutive")
            self.assertEqual(results[0][0]["verdict"], "reusable", results[0][0]["first_divergence"])

    def test_other_routes_and_errors_are_not_recorded_or_do_not_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.start(["hi"], tmp)
            try:
                post(self.base, "/v1/chat/completions", {"model": "m", "messages": 5})
            except urllib.error.HTTPError as e:
                self.assertEqual(e.code, 400)
            lines = lines_of(tmp, 1)
            self.assertEqual(len(lines), 1)                              # the bad request is recorded with its status
            self.assertEqual(lines[0]["http_status"], 400)
            self.assertIsNone(lines[0]["response"]["message"])


if __name__ == "__main__":
    unittest.main()
