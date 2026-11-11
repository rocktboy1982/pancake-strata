"""Unit tests for ttft_bench.py with a fake streaming OpenAI-compatible server (stdlib only, no engine, no GPU).
Run: nice -n 19 python3 -I -m pytest -q tools/pancake/test_ttft_bench.py"""
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ttft_bench as T  # noqa: E402


class Fake:
    """style: 'strata' (usage.prompt_tokens_details.cached_tokens + timings), 'llama' (timings cache_n/prompt_n only),
    'none' (usage without any reuse field).  A prefix cache over whitespace tokens, one slot."""

    def __init__(self, style="strata", first_delay=0.05, tail_delay=0.2, slow_first=0.0, max_prompt=None, tokens_per_word=1.0):
        self.style, self.first_delay, self.tail_delay = style, first_delay, tail_delay
        self.slow_first, self.max_prompt, self.tpw = slow_first, max_prompt, tokens_per_word
        self.requests, self.bodies, self.prev = 0, [], []
        self.lock = threading.Lock()
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(404)
                self.end_headers()

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with fake.lock:
                    fake.requests += 1
                    n = fake.requests
                    fake.bodies.append(body)
                toks = []
                for m in body["messages"]:
                    toks += (m["role"] + ": " + m["content"]).split()
                n_prompt = int(len(toks) * fake.tpw)
                if fake.max_prompt and n_prompt > fake.max_prompt:
                    err = json.dumps({"error": {"code": 400, "type": "exceed_context_size_error",
                                                "message": "the request exceeds the available context size"}}).encode()
                    self.send_response(400)
                    self.send_header("Content-Length", str(len(err)))
                    self.end_headers()
                    self.wfile.write(err)
                    return
                common = 0
                for x, y in zip(fake.prev, toks):
                    if x != y:
                        break
                    common += 1
                fake.prev = toks
                cached = int(common * fake.tpw)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()

                def send(obj):
                    self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                    self.wfile.flush()
                send({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]})   # must not count as a token
                time.sleep(fake.slow_first if n == 1 and fake.slow_first else fake.first_delay)
                send({"choices": [{"index": 0, "delta": {"content": "ok"}}]})
                time.sleep(fake.tail_delay)
                send({"choices": [{"index": 0, "delta": {"content": " done"}, "finish_reason": "stop"}]})
                usage = {"prompt_tokens": n_prompt, "completion_tokens": 2, "total_tokens": n_prompt + 2}
                last = {"choices": [], "usage": usage}
                if fake.style == "strata":
                    usage["prompt_tokens_details"] = {"cached_tokens": cached}
                if fake.style in ("strata", "llama"):
                    last["timings"] = {"cache_n": cached, "prompt_n": n_prompt - cached,
                                       "prompt_ms": 10.0 * (n_prompt - cached) / 100, "prompt_per_second": 10000.0,
                                       "predicted_n": 2}
                send(last)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def client(f, think=False):
    return T.Client("127.0.0.1", f.port, think=think, timeout=30)


def first_line(body):
    return body["messages"][-1]["content"].split("\n", 1)[0]


def test_ttft_is_first_content_not_first_event_nor_total():
    f = Fake(first_delay=0.3, tail_delay=0.4)
    try:
        r = client(f).chat([{"role": "user", "content": "hi there"}], 16)
    finally:
        f.close()
    assert r["err"] is None and r["text"] == "ok done"
    assert r["first_event"] < 0.2            # the empty role delta arrives at once and is not a token
    assert 0.3 <= r["ttft"] < 0.6            # the first content delta, after first_delay
    assert r["total"] >= r["ttft"] + 0.4     # the tail is after the first token
    assert r["timings"]["prompt_n"] > 0


def test_think_off_sends_enable_thinking_false_and_stream_usage():
    f = Fake(first_delay=0.0, tail_delay=0.0)
    try:
        client(f).chat([{"role": "user", "content": "x"}], 16)
        client(f, think=True).chat([{"role": "user", "content": "x"}], 16)
    finally:
        f.close()
    off, on = f.bodies
    assert off["chat_template_kwargs"] == {"enable_thinking": False}
    assert "chat_template_kwargs" not in on
    assert off["stream"] is True and off["stream_options"] == {"include_usage": True} and off["max_tokens"] == 16


def test_scenario_a_discards_run_1_uses_medians_and_unique_prompts():
    f = Fake(first_delay=0.05, tail_delay=0.0, slow_first=0.6)    # request 1 is slow; it must not count
    try:
        res = T.run_a(client(f), [1500], runs=4, max_tokens=16, calibrate=True, tpw=1.0, seed=5, log=lambda *_: None)
    finally:
        f.close()
    e = res["sizes"][0]
    assert [r["discarded"] for r in e["runs"]] == [True, False, False, False]
    assert e["runs"][0]["ttft_s"] >= 0.6
    s = e["summary"]
    assert s["n"] == 3
    assert 0.05 <= s["ttft_s"]["median"] < 0.3 and s["ttft_s"]["max"] < 0.6
    assert s["engine_prompt_tps"] == 10000.0 and s["engine_prompt_ms"] is not None
    firsts = [first_line(b) for b in f.bodies]
    assert len(set(firsts)) == 4 and all(x.startswith("Run id ") for x in firsts)
    bodies = [b["messages"][0]["content"].split("\n")[2] for b in f.bodies]
    assert len(set(bodies)) > 1               # the word list is rotated too: no content cache can match later runs
    assert e["status"] == "measured"


def test_scenario_a_calibrates_to_the_actual_prompt_tokens():
    # this fake counts 2 tokens per word: the first guess (1.06 tokens per word) is far too long, runs 2+ are corrected
    f = Fake(first_delay=0.0, tail_delay=0.0, tokens_per_word=2.0)
    try:
        res = T.run_a(client(f), [1200], runs=3, max_tokens=16, calibrate=True, tpw=1.0, seed=1, log=lambda *_: None)
    finally:
        f.close()
    runs = res["sizes"][0]["runs"]
    assert runs[0]["prompt_tokens"] > 1.5 * 1200
    for r in runs[1:]:
        assert abs(r["prompt_tokens"] - 1200) < 0.05 * 1200, r["prompt_tokens"]
    assert abs(res["sizes"][0]["summary"]["prompt_tokens"] - 1200) < 60


def test_context_limit_skips_the_size_and_larger_ones_cleanly():
    f = Fake(first_delay=0.0, tail_delay=0.0, max_prompt=2000)
    try:
        res = T.run_a(client(f), [700, 5000, 9000], runs=3, max_tokens=16, calibrate=False, tpw=1.0, seed=2, log=lambda *_: None)
        served = f.requests
    finally:
        f.close()
    s0, s1, s2 = res["sizes"]
    assert s0["status"] == "measured" and s0["summary"]["n"] == 2
    assert s1["status"].startswith("skipped: context limit") and "exceeds the available context" in s1["status"]
    assert len(s1["runs"]) == 1 and s1["runs"][0]["context_limit"] is True and s1["summary"]["n"] == 0
    assert s2["status"].startswith("skipped") and s2["runs"] == []
    assert served == 3 + 1                     # 3 runs of the first size, one refused request, nothing for the third size


def test_other_errors_are_failures_not_skips():
    r = T.digest({"status": 500, "err": "boom", "ttft": None, "first_event": None, "total": 0.1})
    assert r["ok"] is False and r["context_limit"] is False
    r = T.digest({"status": 400, "err": '{"message": "prompt is too long: 9 tokens"}', "ttft": None, "total": 0.1})
    assert r["context_limit"] is True


def run_session(style, turns=4):
    f = Fake(style=style, first_delay=0.0, tail_delay=0.0)
    try:
        s = T.agent_session(client(f), turns, 120, 64, "nonce1", T.agent_chunks(turns, 120), log=lambda *_: None)
    finally:
        f.close()
    return s, f


def test_reuse_accounting_strata_style_fields():
    s, f = run_session("strata")
    rows = s["turns"]
    assert [r["turn"] for r in rows] == [1, 2, 3, 4] and all(r["ok"] for r in rows)
    assert {r["reused_source"] for r in rows} == {"usage.prompt_tokens_details.cached_tokens"}
    assert rows[0]["reused_tokens"] == 0
    for prev, cur in zip(rows, rows[1:]):
        assert cur["reused_tokens"] == prev["prompt_tokens"]          # the whole earlier prompt is a cache hit
        assert cur["read_tokens"] == cur["prompt_tokens"] - cur["reused_tokens"]
        assert cur["reread_tokens"] == prev["generated_tokens"] - 0   # only the model's own reply was not in the cache
    sm = T.summarize_b([s])
    assert sm["reuse_reported"] and sm["reuse_source"] == ["usage.prompt_tokens_details.cached_tokens"]
    exp = 100.0 * sum(r["reused_tokens"] for r in rows) / sum(r["prompt_tokens"] for r in rows)
    assert abs(sm["reuse_pct_all_turns"] - exp) < 0.1
    # the conversation is resent whole and grows with the model's own replies
    last = f.bodies[-1]["messages"]
    assert [m["role"] for m in last] == ["system", "user", "user", "assistant", "user", "assistant", "user", "assistant", "user"]
    assert last[3]["content"] == "ok done" and last[0]["content"].startswith("(session nonce1)")
    assert f.bodies[-1]["tools"] == T.TOOLS


def test_reuse_accounting_llama_style_fields():
    s, _ = run_session("llama")
    rows = s["turns"]
    assert {r["reused_source"] for r in rows} == {"timings.cache_n"}
    assert [r["reused_tokens"] for r in rows[1:]] == [r["prompt_tokens"] for r in rows[:-1]]
    sm = T.summarize_b([s])
    assert sm["reuse_source"] == ["timings.cache_n"] and sm["reuse_pct_follow_up_turns"] > 60


def test_reuse_not_reported_is_not_guessed():
    s, _ = run_session("none")
    rows = s["turns"]
    assert all(r["reused_tokens"] is None and r["reused_source"] is None and r["read_tokens"] is None for r in rows)
    sm = T.summarize_b([s])
    assert sm["reuse_reported"] is False and sm["reuse_pct_all_turns"] is None
    res = fake_result("none", sm)
    assert "not reported by this server" in T.summary_text(res)


def fake_result(label, sm=None, a=None, load=1.0, warning=False):
    res = {"label": label, "base": "http://x", "model": "m", "args": {"think": "off"},
           "load": {"start": load, "end": load, "warning": warning}}
    if sm is not None:
        res["scenario_b"] = {"turns": len(sm["per_turn"]), "words_per_turn": 1500, "sessions": [], "summary": sm}
        res["scenario_b"]["summary"]["sessions"] = 1
    if a is not None:
        res["scenario_a"] = a
    return res


def a_block(ttft8, ttft32, tok8=8200, tok32=32800, skipped=False):
    def e(label, target, t, tok):
        s = {"n": 3, "ttft_s": {"median": t, "min": t, "max": t}, "prompt_tokens": tok, "reused_tokens": 0,
             "engine_prompt_ms": t * 900, "engine_prompt_tps": tok / (t * 0.9), "ttft_minus_engine_prompt_s": 0.1, "total_s": t}
        return {"label": label, "target_tokens": target, "runs": [], "status": "measured", "summary": s}
    sizes = [e("8K", 8192, ttft8, tok8)]
    if skipped:
        sizes.append({"label": "32K", "target_tokens": 32768, "runs": [], "status": "skipped: context limit: x",
                      "summary": {"n": 0, "ttft_s": None, "prompt_tokens": None, "reused_tokens": None,
                                  "engine_prompt_ms": None, "engine_prompt_tps": None, "ttft_minus_engine_prompt_s": None, "total_s": None}})
    else:
        sizes.append(e("32K", 32768, ttft32, tok32))
    return {"sizes": sizes}


def test_compare_two_files(tmp_path, capsys):
    s_a, _ = run_session("strata")
    s_b, _ = run_session("none")
    ra = fake_result("pancake", T.summarize_b([s_a]), a_block(4.0, 20.0))
    rb = fake_result("avalon", T.summarize_b([s_b]), a_block(2.0, 25.0), load=3.5, warning=True)
    # force known per-turn TTFTs so the change column is checkable
    for r, v in ((ra, 2.0), (rb, 1.0)):
        for p in r["scenario_b"]["summary"]["per_turn"]:
            p["ttft_s"] = v
        r["scenario_b"]["summary"]["total_ttft_s"] = v * 4
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(ra))
    pb.write_text(json.dumps(rb))
    assert T.main(["--compare", str(pa), str(pb)]) == 0
    out = capsys.readouterr().out
    assert "A = pancake   B = avalon" in out
    assert "-50.0%" in out                      # 8K: 4.0 s -> 2.0 s
    assert "+25.0%" in out                      # 32K: 20.0 s -> 25.0 s
    assert "total TTFT over the session: A 8.00 s, B 4.00 s (-50.0%)" in out
    assert "B 0.0 %" not in out and "(not reported)" in out   # B's server reported no reuse
    assert "run B (avalon) had a 1-minute load average above 2.0" in out


def test_compare_with_a_skipped_size_and_no_common_scenario():
    ra = fake_result("a", None, a_block(4.0, 20.0))
    rb = fake_result("b", None, a_block(4.0, 0.0, skipped=True))
    out = T.compare_text(ra, rb)
    assert "B: skipped: context limit" in out
    sm = T.summarize_b([run_session("strata")[0]])
    assert "nothing to compare" in T.compare_text(fake_result("a", None, a_block(1, 2)), fake_result("b", sm))


def test_sizes_and_chunks():
    assert T.parse_size("8k") == 8192 and T.parse_size("100K") == 102400 and T.parse_size("5000") == 5000
    assert T.size_label(8192) == "8K" and T.size_label(1200) == "1200"
    ch = T.agent_chunks(10, 1500)
    assert len(ch) == 10 and len({c[1] for c in ch}) == 10
    gen = T.agent_chunks(3, 50, root=Path("/nonexistent-dir-for-test"))
    assert [c[0] for c in gen] == ["generated/part0.txt", "generated/part1.txt", "generated/part2.txt"]
    assert len({c[1] for c in gen}) == 3


def test_main_end_to_end_writes_json_and_load_warning(tmp_path, monkeypatch, capsys):
    f = Fake(first_delay=0.0, tail_delay=0.0)
    monkeypatch.setattr(T, "load_avg", lambda: 3.0)
    out = tmp_path / "r.json"
    try:
        rc = T.main(["--port", str(f.port), "--label", "fake", "--out", str(out), "--sizes", "600,900", "--runs", "2",
                     "--turns", "3", "--agent-words", "100", "--seed", "3"])
    finally:
        f.close()
    assert rc == 0
    res = json.loads(out.read_text())
    assert res["load"]["warning"] is True and res["label"] == "fake" and "api_key" not in res["args"]
    assert [e["label"] for e in res["scenario_a"]["sizes"]] == ["600", "900"]
    assert len(res["scenario_b"]["sessions"][0]["turns"]) == 3
    text = capsys.readouterr()
    assert "WARNING: 1-minute load average is 3.00" in text.err
    assert "A. single long prompts" in text.out and "B. agent session" in text.out
    assert len(text.out.splitlines()) < 45      # one screen
