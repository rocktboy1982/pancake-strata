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

    def __init__(self, style="strata", first_delay=0.05, tail_delay=0.2, slow_first=0.0, max_prompt=None, tokens_per_word=1.0,
                 log_path=None, per_token=0.0):
        self.style, self.first_delay, self.tail_delay = style, first_delay, tail_delay
        self.slow_first, self.max_prompt, self.tpw = slow_first, max_prompt, tokens_per_word
        self.requests, self.bodies, self.prev = 0, [], []
        self.per_token = per_token    # extra seconds before the first token per token read (a known slope)
        self.log_path = log_path     # an engine log the fake appends to, the way the engine prints while serving
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
                if fake.log_path:
                    with open(fake.log_path, "a") as lf:
                        lf.write(f"RESUME {cached}\nnoise line {n} nobody wants\n")
                        lf.write(f"strata prefill timing: {n_prompt - cached} tokens, GPU timeline {100 + n} ms, wall {120 + n} ms, "
                                 f"host staging 7 ms: gemm gate/up {50 + n} ({50.0 + n:.1f}%) hc read: down+silu 20 (20.0%) "
                                 f"wait copy 5 (5.0%)\n")
                        lf.write("strata prefill timing: host: chunk setup (PLE rows, the expert stream plan) 3 ms, waiting for "
                                 "each chunk 90 ms, after each chunk (the draft layer, progress) 1 ms, PLE 2 ms\n")
                        lf.write(f"strata prefill timing (peer GPU, ms): idle 1; MoE layers 2\n")
                        lf.write(f"conversation cache: parked {n_prompt} tokens in 12 ms, parked=1 bytes=99\n")
                        if n % 2 == 0:
                            lf.write("strata serve: conversation cache: skip parking (4000 tokens, below the 4096 floor)\n")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()

                def send(obj):
                    self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                    self.wfile.flush()
                send({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]})   # must not count as a token
                time.sleep((fake.slow_first if n == 1 and fake.slow_first else fake.first_delay) + fake.per_token * (n_prompt - cached))
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


# ------------------------------------------------------------------------------------------ scenarios C, D, E and the log tap
def kind_of(body):
    """'main', 'side' or 'sub' from the system prompt the request carries."""
    sys_text = body["messages"][0]["content"]
    if sys_text == T.SIDE_SYSTEM:
        return "side"
    if sys_text.startswith("(sub-agent"):
        return "sub"
    assert sys_text.startswith("(session"), sys_text[:40]
    return "main"


def quiet(*_):
    pass


def run_c(f, turns, side_after, switch_after=0, sub_turns=2, side_messages=1, sessions=1):
    return T.run_c(client(f), turns, 60, 64, sessions, side_after, side_messages, 20, switch_after, sub_turns, 11, log=quiet)


def test_parse_positions():
    assert T.parse_positions("3,6") == [3, 6] and T.parse_positions("") == [] and T.parse_positions("6, 3,3") == [3, 6]
    try:
        T.parse_positions("0")
    except ValueError:
        pass
    else:
        raise AssertionError("position 0 must be refused")


def test_scenario_c_side_requests_go_out_at_the_right_positions_and_are_flagged():
    f = Fake(first_delay=0.0, tail_delay=0.0)
    try:
        c = run_c(f, 5, [2, 4])
    finally:
        f.close()
    assert [kind_of(b) for b in f.bodies] == ["main", "main", "side", "main", "main", "side", "main"]
    rows = c["sessions"][0]["turns"]
    assert [r["turn"] for r in rows] == [1, 2, 3, 4, 5]
    assert [r["after_side"] for r in rows] == [0, 0, 1, 0, 1] and not any(r["after_switch"] for r in rows)
    # a side request is unrelated and short: system + user, no tools, and never inside a main conversation
    side = [b for b in f.bodies if kind_of(b) == "side"]
    assert all([m["role"] for m in b["messages"]] == ["system", "user"] for b in side)
    assert side[0]["messages"][1]["content"] != side[1]["messages"][1]["content"]
    assert all("title" not in m["content"] for b in f.bodies if kind_of(b) == "main" for m in b["messages"])
    assert "tools" not in side[0]
    # the fake has one cache slot, so a main turn after a side request re-reads everything: the flag marks exactly those
    assert rows[2]["reused_tokens"] <= 2 and rows[4]["reused_tokens"] <= 2   # at most the shared "system:" word
    assert rows[1]["reused_tokens"] > 0 and rows[3]["reused_tokens"] > 0
    g = c["summary"]["groups"]
    assert g["plain"]["n"] == 2 and g["after_side"]["n"] == 2 and g["after_switch"]["n"] == 0
    assert g["after_side"]["reread_tokens_median"] > 10 * g["plain"]["reread_tokens_median"]
    assert [p["after_side"] for p in c["summary"]["per_turn"]] == [0, 0, 1, 0, 1]
    assert c["summary"]["other_requests"]["side_n"] == 2 and c["summary"]["other_requests"]["failed"] == 0
    assert c["side_after"] == [2, 4] and c["switch_after"] is None


def test_scenario_c_two_message_side_request_and_main_conversation_matches_scenario_b():
    f = Fake(first_delay=0.0, tail_delay=0.0)
    try:
        c = run_c(f, 3, [1], side_messages=2)
        mains_c = [b for b in f.bodies if kind_of(b) == "main"]
        side = [b for b in f.bodies if kind_of(b) == "side"][0]
    finally:
        f.close()
    assert [m["role"] for m in side["messages"]] == ["system", "user", "assistant", "user"]
    f2 = Fake(first_delay=0.0, tail_delay=0.0)
    try:
        T.agent_session(client(f2), 3, 60, 64, "n", T.agent_chunks(3, 60), log=quiet)
        mains_b = f2.bodies
    finally:
        f2.close()
    # same roles and the same user messages as scenario B (only the session nonce differs)
    assert [[m["role"] for m in b["messages"]] for b in mains_c] == [[m["role"] for m in b["messages"]] for b in mains_b]
    assert [b["messages"][-1]["content"] for b in mains_c] == [b["messages"][-1]["content"] for b in mains_b]
    assert c["summary"]["groups"]["after_side"]["n"] == 1


def test_scenario_c_subagent_switch_order_and_flags():
    f = Fake(first_delay=0.0, tail_delay=0.0)
    try:
        c = run_c(f, 4, [], switch_after=2, sub_turns=2)
    finally:
        f.close()
    assert [kind_of(b) for b in f.bodies] == ["main", "main", "sub", "sub", "main", "main"]
    subs = [b for b in f.bodies if kind_of(b) == "sub"]
    assert [t["function"]["name"] for t in subs[0]["tools"]] == ["search_symbols", "list_dir"]
    assert subs[0]["tools"] != f.bodies[0]["tools"] and subs[0]["messages"][0]["content"] != f.bodies[0]["messages"][0]["content"]
    assert len(subs[1]["messages"]) == len(subs[0]["messages"]) + 2     # the second sub-agent turn continues its own conversation
    rows = c["sessions"][0]["turns"]
    assert [r["after_switch"] for r in rows] == [False, False, True, False] and not any(r["after_side"] for r in rows)
    assert rows[2]["reused_tokens"] <= 2                                 # the sub-agent displaced the main conversation
    assert c["summary"]["groups"]["after_switch"]["n"] == 1 and c["summary"]["other_requests"]["subagent_n"] == 2
    others = c["sessions"][0]["other_requests"]
    assert [(o["kind"], o["after_turn"], o["sub_turn"]) for o in others] == [("subagent", 2, 1), ("subagent", 2, 2)]


def test_scenario_c_side_request_goes_before_the_switch_at_the_same_turn():
    f = Fake(first_delay=0.0, tail_delay=0.0)
    try:
        c = run_c(f, 4, [2], switch_after=2, sub_turns=1)
    finally:
        f.close()
    assert [kind_of(b) for b in f.bodies] == ["main", "main", "side", "sub", "main", "main"]
    r3 = c["sessions"][0]["turns"][2]
    assert r3["after_side"] == 1 and r3["after_switch"] is True
    assert c["summary"]["groups"]["after_switch"]["n"] == 1 and c["summary"]["groups"]["after_side"]["n"] == 0   # the switch wins


def test_scenario_c_refuses_positions_at_or_after_the_last_turn(capsys):
    f = Fake()
    try:
        T.run_c(client(f), 4, 60, 64, 1, [4], 1, 20, 0, 2, 1, log=quiet)
    except ValueError as e:
        assert "before the last turn" in str(e)
    else:
        raise AssertionError("expected ValueError")
    finally:
        f.close()
    assert T.main(["--scenario", "c", "--turns", "4", "--switch-after", "4"]) == 2
    assert "before the last turn" in capsys.readouterr().err


def test_scenario_c_failed_main_turn_stops_but_side_request_result_is_kept():
    f = Fake(first_delay=0.0, tail_delay=0.0, max_prompt=900)   # the main conversation grows past what this fake accepts
    try:
        c = T.run_c(client(f), 8, 100, 64, 1, [1], 1, 20, 0, 2, 1, log=quiet)
    finally:
        f.close()
    s = c["sessions"][0]
    assert s["other_requests"][0]["ok"] is True and s["stopped"] and "context" in s["stopped"]
    assert s["turns"][-1]["ok"] is False and all(t["ok"] for t in s["turns"][:-1])   # the failed turn ends the session


def test_scenario_e_sizes_repeats_order_and_unique_content():
    f = Fake(first_delay=0.0, tail_delay=0.0)
    try:
        e = T.run_e(client(f), 700, [50, 100, 300], 3, 1.0, 16, 4, log=quiet)
    finally:
        f.close()
    assert e["status"] == "measured" and e["base"]["ok"] and e["base"]["prompt_tokens"] > 600
    rows = e["rows"]
    assert len(rows) == 9
    assert [r["delta_target"] for r in rows] == [50, 100, 300] * 3 and [r["repeat"] for r in rows] == [1] * 3 + [2] * 3 + [3] * 3
    users = [m["content"] for m in f.bodies[-1]["messages"] if m["role"] == "user"]
    assert len(users) == 10 and len(set(users)) == 10                   # the base plus nine deltas, all different
    assert f.bodies[-1]["tools"] == T.TOOLS and f.bodies[-1]["messages"][0]["content"].startswith("(session ")
    for r in rows:                                                       # new tokens are close to the target (this fake: 1 token per word)
        assert abs(r["new_tokens_est"] - r["delta_target"]) <= 25, (r["delta_target"], r["new_tokens_est"])
        assert r["reused_tokens"] > 600                                  # the whole earlier conversation stays a cache hit
    per = e["summary"]["per_delta"]
    assert [p["delta"] for p in per] == [50, 100, 300] and all(p["n"] == 3 for p in per)
    assert e["summary"]["fit_ge_1000"] is None and e["summary"]["fit_all"] is not None


def test_scenario_e_fit_recovers_floor_and_slope():
    f = Fake(first_delay=0.05, tail_delay=0.0, per_token=0.0001)      # 0.05 s floor + 0.1 ms per token read
    try:
        e = T.run_e(client(f), 1500, [100, 1000, 3000], 2, 1.0, 16, 9, log=quiet)
    finally:
        f.close()
    s = e["summary"]
    fit = s["fit_all"]
    assert fit and 0.07 <= fit["slope_ms_per_token"] <= 0.13, fit
    assert 0.03 <= fit["intercept_s"] <= 0.15, fit
    big = s["fit_ge_1000"]
    assert big and 0.07 <= big["slope_ms_per_token"] <= 0.13
    t = {p["delta"]: p["ttft_s"]["median"] for p in s["per_delta"]}
    assert t[100] < t[1000] < t[3000]


def test_scenario_e_context_limit_is_recorded_not_a_crash():
    f = Fake(first_delay=0.0, tail_delay=0.0, max_prompt=900)
    try:
        e = T.run_e(client(f), 700, [50, 1000], 2, 1.0, 16, 3, log=quiet)
    finally:
        f.close()
    assert e["status"].startswith("skipped: context limit") and e["summary"]["per_delta"][1]["n"] == 0
    assert e["summary"]["per_delta"][0]["n"] >= 1


def test_linfit():
    assert T.linfit([(0, 1.0), (1000, 1.5), (2000, 2.0)]) == {"n": 3, "slope_ms_per_token": 0.5, "intercept_s": 1.0}
    assert T.linfit([(5, 1.0)]) is None and T.linfit([(5, 1.0), (5, 2.0)]) is None


ENGINE_LINES = (
    "strata prefill timing: 41811 tokens, GPU timeline 42000 ms, wall 43100 ms, host staging 900 ms: embed+steps 10 (0.0%) "
    "hc read 500 (1.2%) hc read: down+silu 200 (0.5%) gdn recurrence 2515 (6.0%) qsa: norm/rope/kv append/q prep 90 (0.2%) "
    "gemm gate/up 9000 (21.4%)\n"
    "strata prefill timing: host: chunk setup (PLE rows, the expert stream plan) 12 ms, waiting for each chunk 40000 ms, "
    "after each chunk (the draft layer, progress) 30 ms, PLE 55 ms\n"
    "strata prefill timing (layer-major): 100 tokens, GPU timeline 7 ms: gdn 3 (42.9%) ple 4 (57.1%)\n"
    "strata prefill timing (peer GPU, ms): idle 1 moe in 2; MoE layers 2, rows/layer 1 of 2 cap, experts/layer 1, over the cap 0\n"
    "strata ple wait: chunk at 0 (4096 tokens): gather 1.0 ms (FP32, 1 thread(s), 1 slice(s)), host waited 0.5 ms at layer 1 (first slice 0.5 ms), PLE sub-batch 2\n"
    "strata serve: conversation cache: skip parking (31 tokens, below the 4096 floor)\n"
    "conversation cache: parked 41811 tokens in 2100 ms, parked=2 bytes=123\n"
    "conversation cache: restored 20000 tokens in 800 ms, parked=1 bytes=5\n"
    "conversation cache: evicted 9000 tokens\n"
    "strata serve: file tier I/O this request: the OS read 1500.5 MB from storage (3 major faults) for x\n"
    "RESUME 1234\n"
    "strata prefill: 41811 tokens in 16384-token chunks over 2 stages (--prefill-pipe auto)\n"
    "strata prefill: warmed the BF16 and FP16 GEMM libraries on GPU 0 in 5 ms\n"
    "unrelated: INFO served GET /health\n")


def test_parse_engine_lines_reads_the_formats_the_code_prints():
    p = T.parse_engine_lines([x for x in ENGINE_LINES.splitlines() if T.INTEREST_RE.search(x)])
    pt = p["prefill_timing"]
    assert [x["tokens"] for x in pt] == [41811, 100] and pt[1]["layer_major"] is True and pt[0]["layer_major"] is False
    assert pt[0]["gpu_ms"] == 42000 and pt[0]["wall_ms"] == 43100 and pt[0]["host_staging_ms"] == 900
    assert pt[0]["phases"]["gdn recurrence"] == {"ms": 2515, "pct": 6.0}
    assert pt[0]["phases"]["hc read: down+silu"]["ms"] == 200 and pt[0]["phases"]["qsa: norm/rope/kv append/q prep"]["ms"] == 90
    assert len(pt[0]["phases"]) == 6 and pt[0]["host"] == {"chunk_setup_ms": 12.0, "waiting_for_chunks_ms": 40000.0,
                                                           "after_each_chunk_ms": 30.0, "ple_ms": 55.0}
    assert pt[1]["wall_ms"] is None and pt[1]["host"] is None and pt[1]["phases"]["ple"]["ms"] == 4
    assert p["parks"] == [{"tokens": 41811, "ms": 2100.0}] and p["restores"] == [{"tokens": 20000, "ms": 800.0}]
    assert p["skips"] == ["31 tokens, below the 4096 floor"] and p["evicted"] == [9000] and p["resume"] == [1234]
    assert p["file_tier_mb"] == [1500.5] and p["ple_wait_lines"] == 1
    assert p["chunk_plan"] == [{"tokens": 41811, "chunk": 16384, "stages": 2}]


def test_log_tap_attaches_only_the_lines_added_during_each_run(tmp_path):
    log = tmp_path / "engine.log"
    log.write_text("strata prefill timing: 5 tokens, GPU timeline 1 ms, wall 1 ms, host staging 0 ms: gdn 1 (100.0%)\nold\n")
    f = Fake(first_delay=0.0, tail_delay=0.0, log_path=str(log))
    cl = client(f)
    cl.tap = T.LogTap(str(log), settle_s=0.0)
    try:
        a = T.run_a(cl, [400], runs=3, max_tokens=16, calibrate=False, tpw=1.0, seed=2, log=quiet)
    finally:
        f.close()
    runs = a["sizes"][0]["runs"]
    assert len(runs) == 3
    for i, r in enumerate(runs, 1):
        el = r["engine_log"]
        assert len(el["prefill_timing"]) == 1 and el["prefill_timing"][0]["tokens"] == r["read_tokens"]
        assert el["prefill_timing"][0]["gpu_ms"] == 100 + i and el["prefill_timing"][0]["host"]["waiting_for_chunks_ms"] == 90.0
        assert set(el["prefill_timing"][0]["phases"]) == {"gemm gate/up", "hc read: down+silu", "wait copy"}
        assert el["resume"] == [r["reused_tokens"]] and len(el["parks"]) == 1 and el["parks"][0]["tokens"] == r["prompt_tokens"]
        assert len(el["skips"]) == (1 if i % 2 == 0 else 0)
        assert not any("noise" in x or x == "old" for x in el["lines"]) and el["lines_total"] > el["lines_matched"]
    assert all(r["engine_log"]["prefill_timing"][0]["tokens"] != 5 for r in runs)   # the line that was there before is attached to nothing
    res = {"label": "x", "base": "b", "model": "m", "args": {"think": "off", "prefill_timing_log": str(log)},
           "load": {"start": 1.0, "end": 1.0, "warning": False}, "scenario_a": a}
    out = T.summary_text(res)
    assert "engine log lines attached per request" in out and "A   400:" in out and "gemm gate/up" in out
    assert "no 'strata prefill timing' line found" not in out


def test_log_tap_missing_file_empty_file_and_truncation(tmp_path):
    t = T.LogTap(str(tmp_path / "nope.log"), settle_s=0.0)
    assert "not readable" in t.since(t.mark())["error"]
    empty = tmp_path / "empty.log"
    empty.write_text("")
    f = Fake(first_delay=0.0, tail_delay=0.0)                    # an engine that writes nothing to that file
    cl = client(f)
    cl.tap = T.LogTap(str(empty), settle_s=0.0)
    try:
        a = T.run_a(cl, [300], runs=2, max_tokens=16, calibrate=False, tpw=1.0, seed=1, log=quiet)
    finally:
        f.close()
    assert a["sizes"][0]["runs"][1]["engine_log"]["prefill_timing"] == []
    res = {"label": "x", "base": "b", "model": "m", "args": {"think": "off", "prefill_timing_log": str(empty)},
           "load": {"start": 1.0, "end": 1.0, "warning": False}, "scenario_a": a}
    assert "no 'strata prefill timing' line found" in T.summary_text(res) and "STRATA_PREFILL_TIMING=1" in T.summary_text(res)
    big = tmp_path / "rot.log"
    big.write_text("x" * 500)
    off = T.LogTap(str(big), settle_s=0.0).mark()
    big.write_text("RESUME 7\n")                                  # rotated: smaller than the offset, read from the start
    rec = T.LogTap(str(big), settle_s=0.0).since(off)
    assert rec["truncated_file"] is True and rec["resume"] == [7]


def test_no_log_option_leaves_records_unchanged():
    f = Fake(first_delay=0.0, tail_delay=0.0)
    try:
        a = T.run_a(client(f), [300], runs=2, max_tokens=16, calibrate=False, tpw=1.0, seed=1, log=quiet)
    finally:
        f.close()
    assert all("engine_log" not in r for r in a["sizes"][0]["runs"])
    assert T.engine_log_text({"args": {}, "scenario_a": a}) == []


def test_log_tap_also_reaches_scenarios_b_c_and_e(tmp_path):
    log = tmp_path / "e.log"
    log.write_text("")
    f = Fake(first_delay=0.0, tail_delay=0.0, log_path=str(log))
    cl = client(f)
    cl.tap = T.LogTap(str(log), settle_s=0.0)
    try:
        s = T.agent_session(cl, 2, 60, 64, "n", T.agent_chunks(2, 60), log=quiet)
        c = T.run_c(cl, 3, 60, 64, 1, [1], 1, 20, 2, 1, 3, log=quiet)
        e = T.run_e(cl, 400, [50], 1, 1.0, 16, 3, log=quiet)
    finally:
        f.close()
    assert all(t["engine_log"]["prefill_timing"] for t in s["turns"])
    assert all(o["engine_log"]["resume"] for o in c["sessions"][0]["other_requests"])
    assert e["base"]["engine_log"]["prefill_timing"] and e["rows"][0]["engine_log"]["parks"]
    res = {"label": "x", "args": {"prefill_timing_log": "p"}, "scenario_c": c, "scenario_e": e}
    reqs = list(T.iter_requests(res))
    assert len(reqs) == 3 + 2 + 1 + 1       # C: 3 main + 1 side + 1 sub-agent turn... plus E: base + 1 row
    assert all(r.get("engine_log") for r in reqs)


def test_cache_budget_note_is_recorded_and_compare_prints_it(tmp_path, capsys):
    f = Fake(first_delay=0.0, tail_delay=0.0)
    out = tmp_path / "r.json"
    try:
        rc = T.main(["--port", str(f.port), "--label", "tight", "--out", str(out), "--scenario", "c", "--turns", "3",
                     "--agent-words", "60", "--side-after", "1", "--seed", "3", "--note-cache-budget-mib", "2048"])
    finally:
        f.close()
    assert rc == 0
    res = json.loads(out.read_text())
    assert res["cache_budget_mib"] == 2048 and res["args"]["note_cache_budget_mib"] == 2048
    assert "conversation-cache budget noted by the operator: 2048 MiB" in capsys.readouterr().out
    other = dict(res, label="roomy", cache_budget_mib=20000)
    none = dict(res, label="unnoted", cache_budget_mib=None)
    pa, pb, pc = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "c.json"
    pa.write_text(json.dumps(res))
    pb.write_text(json.dumps(other))
    pc.write_text(json.dumps(none))
    assert T.main(["--compare", str(pa), str(pb)]) == 0
    assert "conversation-cache budget (noted by the operator): A 2048 MiB, B 20000 MiB" in capsys.readouterr().out
    assert T.main(["--compare", str(pa), str(pc)]) == 0
    assert "A 2048 MiB, B not noted" in capsys.readouterr().out
    assert "budget" not in T.compare_text(none, dict(none, label="also"))   # neither noted: no line


def test_compare_prints_scenarios_c_and_e():
    f = Fake(first_delay=0.0, tail_delay=0.0)
    try:
        ra = {"label": "A", "load": {"warning": False}, "scenario_c": run_c(f, 4, [1], switch_after=2, sub_turns=1),
              "scenario_e": T.run_e(client(f), 500, [50, 100], 2, 1.0, 16, 5, log=quiet)}
        rb = {"label": "B", "load": {"warning": False}, "scenario_c": run_c(f, 4, [1], switch_after=2, sub_turns=1),
              "scenario_e": T.run_e(client(f), 500, [50, 100], 2, 1.0, 16, 6, log=quiet)}
    finally:
        f.close()
    for r, v in ((ra, 2.0), (rb, 1.0)):
        for p in r["scenario_c"]["summary"]["per_turn"]:
            p["ttft_s"] = v
        r["scenario_c"]["summary"]["total_ttft_s"] = 4 * v
        for p in r["scenario_e"]["summary"]["per_delta"]:
            p["ttft_s"]["median"] = v
    out = T.compare_text(ra, rb)
    assert "C. session with side requests" in out and "-50.0%" in out and "switch" in out and "side" in out
    assert "total TTFT over the main turns: A 8.00 s, B 4.00 s (-50.0%)" in out
    assert "E. warm-delta sweep" in out and "fit, all deltas" in out
    assert out.count("-50.0%") >= 4 + 1 + 2


def test_summary_text_c_and_e_show_flags_and_fit():
    f = Fake(first_delay=0.0, tail_delay=0.0)
    try:
        c = run_c(f, 4, [1], switch_after=3, sub_turns=1)
        e = T.run_e(client(f), 500, [50, 100], 2, 1.0, 16, 5, log=quiet)
    finally:
        f.close()
    res = {"label": "x", "base": "b", "model": "m", "args": {"think": "off"}, "load": {"start": 1.0, "end": 1.0, "warning": False},
           "scenario_c": c, "scenario_e": e}
    out = T.summary_text(res)
    assert "C. session with side requests" in out and "after turn(s) 1" in out and "sub-agent switch after turn 3 for 1 turns" in out
    rows = [x.split() for x in out.splitlines() if x.strip()[:1].isdigit()]
    flags = {int(r[0]): r[1] for r in rows if len(r) >= 8 and r[1] in ("side", "switch", "-")}
    assert flags[2] == "side" and flags[4] == "switch" and flags[1] == "-" and flags[3] == "-"
    assert "E. warm-delta sweep" in out and "fit (all deltas)" in out and "turns 2+ after_side" in out


def test_main_all_scenarios_end_to_end(tmp_path, monkeypatch, capsys):
    f = Fake(first_delay=0.0, tail_delay=0.0)
    monkeypatch.setattr(T, "load_avg", lambda: 0.5)
    out = tmp_path / "r.json"
    try:
        rc = T.main(["--port", str(f.port), "--label", "all", "--out", str(out), "--scenario", "all", "--sizes", "400", "--runs", "2",
                     "--turns", "4", "--agent-words", "60", "--side-after", "2", "--switch-after", "3", "--subagent-turns", "1",
                     "--delta-base", "500", "--deltas", "50,100", "--delta-repeats", "2", "--seed", "3"])
    finally:
        f.close()
    assert rc == 0
    res = json.loads(out.read_text())
    assert {"scenario_a", "scenario_b", "scenario_c", "scenario_e"} <= set(res) and res["cache_budget_mib"] is None
    assert res["scenario_c"]["side_after"] == [2] and res["scenario_c"]["switch_after"] == 3
    assert [r["delta_target"] for r in res["scenario_e"]["rows"]] == [50, 100, 50, 100]
    text = capsys.readouterr().out
    for head in ("A. single long prompts", "B. agent session", "C. session with side requests", "E. warm-delta sweep"):
        assert head in text


def test_main_default_scenario_is_still_a_plus_b(tmp_path, monkeypatch):
    f = Fake(first_delay=0.0, tail_delay=0.0)
    monkeypatch.setattr(T, "load_avg", lambda: 0.5)
    out = tmp_path / "r.json"
    try:
        T.main(["--port", str(f.port), "--out", str(out), "--sizes", "300", "--runs", "2", "--turns", "2", "--agent-words", "50"])
    finally:
        f.close()
    res = json.loads(out.read_text())
    assert "scenario_c" not in res and "scenario_e" not in res and "scenario_a" in res and "scenario_b" in res


def test_main_prefill_timing_log_option(tmp_path, monkeypatch, capsys):
    log = tmp_path / "engine.log"
    log.write_text("")
    f = Fake(first_delay=0.0, tail_delay=0.0, log_path=str(log))
    monkeypatch.setattr(T, "load_avg", lambda: 0.5)
    out = tmp_path / "r.json"
    try:
        T.main(["--port", str(f.port), "--out", str(out), "--scenario", "a", "--sizes", "300", "--runs", "2",
                "--prefill-timing-log", str(log), "--log-settle-s", "0"])
    finally:
        f.close()
    res = json.loads(out.read_text())
    assert res["scenario_a"]["sizes"][0]["runs"][1]["engine_log"]["prefill_timing"][0]["tokens"] > 0
    assert "engine log lines attached per request" in capsys.readouterr().out
