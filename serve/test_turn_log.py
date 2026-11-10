"""serve/test_turn_log.py - item 73: the "turn_log" config writes one JSON line per request, numbers only (no prompt,
answer or tool text).  Mock engine, no GPU.

    python -m unittest serve.test_turn_log -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SECRET = "the secret prompt words"


class TurnLog(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.engine = MockEngine(tok, "<think>\nhm\n</think>\n\nfine answer", max_context=8192)
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.dir = tempfile.TemporaryDirectory()
        self.svc.turn_log = str(Path(self.dir.name) / "turns.jsonl")
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.dir.cleanup()

    def chat(self, **extra):
        body = {"model": "m", "max_tokens": 200, "messages": [{"role": "user", "content": SECRET}], **extra}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())

    def test_one_line_per_request_without_text(self):
        self.chat()
        self.chat()
        raw = Path(self.svc.turn_log).read_text()
        lines = [json.loads(x) for x in raw.splitlines()]
        self.assertEqual(len(lines), 2)
        for rec in lines:
            for key in ("time", "duration_s", "finish", "prompt_tokens", "reused", "output_tokens",
                        "reasoning_tokens", "max_new"):
                self.assertIn(key, rec)
            self.assertGreater(rec["reasoning_tokens"], 0)
        self.assertNotIn(SECRET, raw)
        self.assertNotIn("fine answer", raw)

    def test_effort_is_logged_as_a_new_key(self):
        self.chat(reasoning_effort="low")
        self.chat(reasoning_effort="XHigh")
        self.chat()
        lines = [json.loads(x) for x in Path(self.svc.turn_log).read_text().splitlines()]
        self.assertEqual([r["effort"] for r in lines], ["low", "xhigh", None])
        for key in ("time", "duration_s", "finish", "reasoning_tokens", "max_new", "model"):   # the old keys stay
            self.assertIn(key, lines[0])

    def test_unwritable_path_turns_the_log_off(self):
        self.svc.turn_log = str(Path(self.dir.name) / "missing-dir" / "turns.jsonl")
        self.chat()                                      # the request still succeeds
        self.assertIsNone(self.svc.turn_log)


if __name__ == "__main__":
    unittest.main()
