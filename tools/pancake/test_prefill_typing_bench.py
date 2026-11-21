"""Unit tests for prefill_typing_bench.py with a fake server (stdlib only, no engine, no GPU).
Run: python -m pytest -q tools/pancake/test_prefill_typing_bench.py"""
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prefill_typing_bench as P  # noqa: E402
import ttft_bench as T  # noqa: E402


class Fake:
    """A server with a one-slot word-prefix cache; TTFT = per_word * words not in the cache."""

    def __init__(self, advertise=True, per_word=0.0004):
        self.cache, self.per_word, self.advertise = [], per_word, advertise
        self.prefill_posts = 0
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == "/health":
                    body = json.dumps({"status": "ok", **({"prefill_typing": True} if fake.advertise else {})}).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                toks = " ".join(m["content"] for m in body["messages"]).split()
                common = 0
                for x, y in zip(fake.cache, toks):
                    if x != y:
                        break
                    common += 1
                if self.path == "/v1/prefill":
                    fake.prefill_posts += 1
                    time.sleep(fake.per_word * (len(toks) - common))
                    fake.cache = toks
                    out = json.dumps({"strata_prefill": {"warmed": True, "stable_tokens": len(toks) - 1}}).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                time.sleep(fake.per_word * (len(toks) - common))
                fake.cache = toks

                def send(obj):
                    self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                    self.wfile.flush()
                send({"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]})
                send({"choices": [], "usage": {"prompt_tokens": len(toks), "completion_tokens": 1,
                                               "prompt_tokens_details": {"cached_tokens": common}}})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def test_draft_is_a_word_boundary_prefix():
    text = "alpha beta gamma delta epsilon zeta eta theta"
    d = P.draft_of(text, 0.5)
    assert text.startswith(d) and not d.endswith(" ") and 0 < len(d) < len(text)


def test_warm_arm_reuses_the_draft_and_is_faster():
    f = Fake()
    try:
        cl = T.Client("127.0.0.1", f.port, timeout=30)
        res = P.run(cl, [2048], 3, 0.9, 0.0, 1, False, log=lambda *a: None)
    finally:
        f.close()
    s = res["sizes"][0]["summary"]
    assert s["n"] == 2                                   # run 1 is discarded
    assert s["warm_reused_tokens"] and s["warm_reused_tokens"] > 0.5 * s["prompt_tokens"]
    assert s["warm_ttft_s"] < s["cold_ttft_s"]
    assert f.prefill_posts == 3
    assert "cold TTFT" in P.text_summary(res)


def test_refuses_a_server_without_the_switch(capsys):
    f = Fake(advertise=False)
    try:
        assert P.main(["--port", str(f.port), "--sizes", "1k"]) == 2
    finally:
        f.close()
    assert "STRATA_PREFILL_TYPING" in capsys.readouterr().err
