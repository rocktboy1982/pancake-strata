"""serve/test_status_page.py - the engine status page: Service.pancake_status(), GET /status.json, and the HTML
content negotiation on /status and /pancake (a browser gets serve/web/status.html, an API caller the old JSON).
Mock engine, no GPU.

    python -m pytest serve/test_status_page.py -v
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
CTX = 8192


def make_svc(script="ok", api_key=""):
    tok = ByteTokenizer()
    svc = Service(MockEngine(tok, script, max_context=CTX), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
    svc.api_key = api_key
    return svc


def make_httpd(svc):
    httpd = serve(svc, port=0)
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, dict(r.headers), r.read()


class PancakeStatus(unittest.TestCase):
    def setUp(self):
        self.svc = make_svc()
        self.httpd, self.base = make_httpd(self.svc)

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def chat(self):
        body = {"model": "m", "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}]}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())

    def test_shape_after_a_turn(self):
        self.chat()
        status, headers, raw = get(self.base + "/status.json")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/json")
        d = json.loads(raw)
        for key in ("engine", "hardware", "model", "switches", "live", "stats", "loop_guard", "turns", "time"):
            self.assertIn(key, d)
        self.assertEqual(d["engine"]["max_context"], CTX)
        self.assertIsNone(d["engine"]["binary"])          # the mock engine has no spawn
        self.assertIsInstance(d["engine"]["info"], dict)
        self.assertEqual(d["model"]["model"], self.svc.model)
        self.assertEqual(d["live"]["busy"], False)
        self.assertEqual(d["live"]["queued"], 0)
        self.assertEqual(d["loop_guard"]["enabled"], True)
        self.assertEqual(sorted(d["loop_guard"]["counts"]),
                         ["collapsed", "detected", "kept_going", "penalised", "stopped", "thinking_closed",
                          "tool_blocked", "tool_repeats"])

    def test_stats_and_turns_from_one_request(self):
        self.chat()
        d = json.loads(get(self.base + "/status.json")[2])
        s = d["stats"]
        self.assertEqual(s["requests"], 1)
        self.assertGreater(s["prompt_tokens"], 0)
        self.assertGreater(s["output_tokens"], 0)
        self.assertEqual(s["drafts_offered"], 0)
        self.assertEqual(s["drafts_accepted"], 0)
        self.assertEqual(s["prefix_reuse_pct"], 0.0)      # nothing was reused yet
        self.assertIsNone(s["decode_tok_s_mean"])         # the mock engine reports no decode_ms
        turns = d["turns"]
        self.assertEqual(len(turns), 1)
        t = turns[0]
        self.assertEqual(t["finish"], "stop")
        self.assertGreater(t["prompt_tokens"], 0)
        self.assertGreater(t["output_tokens"], 0)
        self.assertLessEqual(t["age_s"], 60)

    def test_switches_carry_strata_env_and_server_flags(self):
        import os
        self.svc.reasoning_budget_grace = 7
        self.svc.turn_log = "/tmp/turns.jsonl"
        with unittest.mock.patch.dict(os.environ, {"STRATA_TEST_SWITCH": "1", "MAYA_TEST": "2",
                                                                 "HOME": os.environ.get("HOME", "")}):
            d = json.loads(get(self.base + "/status.json")[2])
        env = d["switches"]["env"]
        self.assertEqual(env.get("STRATA_TEST_SWITCH"), "1")
        self.assertEqual(env.get("MAYA_TEST"), "2")
        self.assertEqual(list(env), sorted(env))
        server = d["switches"]["server"]
        self.assertEqual(server["reasoning_budget_grace"], 7)
        self.assertEqual(server["turn_log"], "/tmp/turns.jsonl")
        self.assertIn("loop_guard", server)

    def test_turn_log_tail_when_history_is_empty(self):
        import tempfile
        self.svc.history.clear()
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "turns.jsonl"
            rec = {"time": self.svc.started_at, "finish": "stop", "prompt_tokens": 11, "reused": 3,
                   "prompt_read": 8, "output_tokens": 5, "reasoning_tokens": 2, "first_engine_token_s": 0.25,
                   "prompt_ms": 120.0, "decode_ms": 500.0, "decode_tok_s": 10.0, "hit_rate": 0.5,
                   "drafts_offered": 4, "drafts_accepted": 2}
            log.write_text(json.dumps(rec) + "\n", encoding="utf-8")
            self.svc.turn_log = str(log)
            d = json.loads(get(self.base + "/status.json")[2])
        self.assertEqual(len(d["turns"]), 1)
        t = d["turns"][0]
        self.assertEqual((t["prompt_tokens"], t["reused"], t["prompt_read"]), (11, 3, 8))
        self.assertEqual(t["ttft_ms"], 250)
        self.assertEqual(t["decode_tok_s"], 10.0)
        self.assertEqual(t["drafts_offered"], 4)
        self.assertEqual(t["finish"], "stop")
        self.assertEqual(d["stats"]["reasoning_tokens"], 0)   # the history is empty: no reasoning to sum


class StatusJsonEndpoint(unittest.TestCase):
    def setUp(self):
        self.svc = make_svc()
        self.httpd, self.base = make_httpd(self.svc)

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def test_status_json_needs_the_key_when_one_is_set(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.svc.api_key = "k3y"
        self.httpd, self.base = make_httpd(self.svc)
        with self.assertRaises(urllib.error.HTTPError) as e:
            get(self.base + "/status.json")
        self.assertEqual(e.exception.code, 401)
        e.exception.close()
        status, _, raw = get(self.base + "/status.json", headers={"Authorization": "Bearer k3y"})
        self.assertEqual(status, 200)
        self.assertIn("engine", json.loads(raw))

    def test_status_json_without_a_key_needs_no_key(self):
        status, _, raw = get(self.base + "/status.json")
        self.assertEqual(status, 200)
        self.assertIn("engine", json.loads(raw))


class StatusHtmlNegotiation(unittest.TestCase):
    def setUp(self):
        self.svc = make_svc()
        self.httpd, self.base = make_httpd(self.svc)

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def page(self):
        return (ROOT / "serve/web/status.html").read_bytes()

    def test_pancake_serves_the_page(self):
        status, headers, body = get(self.base + "/pancake")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
        self.assertEqual(headers.get("Content-Security-Policy"), "frame-src 'none'")
        self.assertEqual(body, self.page())

    def test_status_with_accept_html_serves_the_page(self):
        status, headers, body = get(self.base + "/status", headers={"Accept": "text/html,application/xhtml+xml"})
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
        self.assertEqual(body, self.page())

    def test_status_with_html_query_serves_the_page(self):
        status, headers, body = get(self.base + "/status?html=1")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
        self.assertEqual(body, self.page())

    def test_status_without_html_stays_the_old_json(self):
        status, headers, raw = get(self.base + "/status")     # urllib sends no Accept header
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/json")
        d = json.loads(raw)
        self.assertIn("busy", d)
        self.assertIn("queued", d)
        self.assertNotIn("engine", d)                         # the old payload, not pancake_status()

    def test_status_json_mode_still_needs_the_key(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.svc.api_key = "k3y"
        self.httpd, self.base = make_httpd(self.svc)
        with self.assertRaises(urllib.error.HTTPError) as e:
            get(self.base + "/status")
        self.assertEqual(e.exception.code, 401)
        e.exception.close()
        # a browser still gets the page without a key, like / and /health
        status, headers, body = get(self.base + "/status", headers={"Accept": "text/html"})
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
        self.assertEqual(body, self.page())


if __name__ == "__main__":
    unittest.main()
