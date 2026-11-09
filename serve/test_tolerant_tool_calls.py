"""serve/test_tolerant_tool_calls.py - a tool call the parser cannot read never fails or swallows the answer.

    python -m unittest serve.test_tolerant_tool_calls -v

Adapted from Project Maya (https://github.com/mw00/project-maya, MIT) serve/test_tools.py and serve/test_server.py
(#613: a call that did not parse failed the whole request, 24K tokens of work lost; #706: an unclosed `<parameter=`
tag).  pancake already returned an unannounced malformed call as text (tool_call_recovery); what it lacked: a call
announced to a streaming client whose body then did not parse kept the parser waiting for a `</parameter>` that never
came, so every word after it was lost - and with "tool_call_recovery": false it raised.  Now such a call stays
unfinished for the client (#211's convention: no tool_call follows its tool_start) and the answer goes on.
"""
from __future__ import annotations

import json
import sys
import threading
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import CALL_END, CALL_START, ChatTemplate, MalformedToolCall, OutputParser, parse_tool_call  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOOLS = [{"name": "write", "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                                                                           "text": {"type": "string"}}}}]
# an unclosed <parameter= tag (#706): `<parameter=text` runs into `</function>`
BAD = "\n<function=write>\n<parameter=path>\nPLAN.md\n</parameter>\n<parameter=text\n</function>\n"


def feed(text, stream, recover=True, chunk=1):
    p = OutputParser(thinking=False, tools=TOOLS, stream_tools=stream, recover=recover)
    evs = []
    for i in range(0, len(text), chunk):
        evs += p.feed(text[i:i + chunk])
    return p, evs + p.finish("stop")


class ParseToolCall(unittest.TestCase):
    def test_malformed(self):
        for body in ("write junk", "<function=f><parameter=a>1</parameter><parameter=b",
                     "<function=f><parameter=a", "<function=f>\n<parameter=a\n<parameter=b>\nx\n</parameter>"):
            with self.assertRaises(ValueError, msg=body):
                parse_tool_call(body)

    def test_malformed_carries_the_whole_body(self):   # (the log shows all of it, the message 80 characters)
        body = "<function=f><parameter=a" + "x" * 300
        with self.assertRaises(MalformedToolCall) as cm:
            parse_tool_call(body)
        self.assertEqual(cm.exception.body, body)
        self.assertLess(len(str(cm.exception)), 120)

    def test_a_good_call_still_parses(self):
        c = parse_tool_call("<function=write>\n<parameter=path>\na<b>.md\n</parameter>\n</function>")
        self.assertEqual((c.name, c.arguments), ("write", {"path": "a<b>.md"}))


class TheParserNeverRaisesOrSwallows(unittest.TestCase):
    def test_streamed_and_whole(self):
        text = "Let me write it.\n" + CALL_START + BAD + CALL_END + " Then more."
        for recover in (True, False):
            for stream in (False, True):
                for chunk in (1, 7, len(text)):
                    with self.subTest(recover=recover, stream=stream, chunk=chunk):
                        p, evs = feed(text, stream, recover, chunk)
                        out = "".join(e.text for e in evs if e.kind == "content")
                        self.assertEqual([e for e in evs if e.kind == "tool_call"], [])
                        self.assertIn("Then more.", out)            # the answer goes on after it
                        self.assertTrue(p.malformed)
                        if stream and any(e.kind == "tool_start" for e in evs):
                            self.assertNotIn(CALL_START, out)       # announced: unfinished, its text not repeated
                        else:
                            self.assertIn(CALL_START + BAD + CALL_END, out)   # never announced: the text as written

    def test_a_good_call_after_a_bad_one(self):
        good = CALL_START + "\n<function=write>\n<parameter=path>\nB.md\n</parameter>\n</function>\n" + CALL_END
        p, evs = feed(CALL_START + BAD + CALL_END + "\n" + good, True)
        self.assertEqual([e.call.arguments for e in evs if e.kind == "tool_call"], [{"path": "B.md"}])


class TheRequestAnswers(unittest.TestCase):
    def test_chat_completions(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "</think>\n\nLet me write it.\n" + CALL_START + BAD + CALL_END + " Done."), tok,
                      ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.tool_call_recovery = False                  # (it used to raise there: the request failed)
        httpd = serve(svc, port=0)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            base = f"http://127.0.0.1:{httpd.server_address[1]}"
            for stream in (False, True):
                req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps({
                    "model": "m", "max_tokens": 512, "stream": stream,
                    "messages": [{"role": "user", "content": "Write the plan."}],
                    "tools": [{"type": "function", "function": TOOLS[0]}]}).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    self.assertEqual(r.status, 200)
                    body = r.read().decode()
                self.assertNotIn('"error"', body)
                if stream:
                    chunks = [json.loads(line[6:]) for line in body.splitlines()
                              if line.startswith("data: {")]
                    text = "".join((c["choices"][0]["delta"].get("content") or "") for c in chunks if c.get("choices"))
                else:
                    text = json.loads(body)["choices"][0]["message"]["content"]
                self.assertIn("Let me write it.", text)
                self.assertIn("Done.", text)
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
