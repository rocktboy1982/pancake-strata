"""Unit tests for prose_gap.py (MOD_PLAN 17.2): a fake OpenAI-compatible server on localhost (stdlib http.server), no
engine, no GPU.  It plays two servers: "pancake" (/health says service=strata, has /metrics and /v1/status whose totals
grow with every request) and "llama" (no /metrics, no /v1/status).
Run: nice -n 19 python3 -I -m pytest -q tools/pancake/test_prose_gap.py"""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prose_gap as pg  # noqa: E402


class Fake:
    """Answers with fixed timings; the first request is slower (a cold run)."""

    def __init__(self, pancake, tps_list, tok=512, accepted=300, drafted=470):
        self.pancake, self.tps_list, self.tok, self.accepted, self.drafted = pancake, list(tps_list), tok, accepted, drafted
        self.n = 0
        self.totals = {"requests": 0, "output_tokens": 0, "decode_ms": 0.0, "drafts_offered": 0, "drafts_accepted": 0}
        self.prompts = []
        self.bodies = []

        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                raw = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path == "/health":
                    self._send(200, {"service": "strata", "status": "ok"} if fake.pancake else {"status": "ok"})
                elif self.path == "/metrics" and fake.pancake:
                    self._send(200, {"engine": {"model": "fake"}, "totals": dict(fake.totals), "hardware": {"cpu_pct": 10.0},
                                     "live": {"state": "idle"}})
                elif self.path == "/v1/status" and fake.pancake:
                    self._send(200, {"model": "fake", "vram": {"free_mib": 100}})
                else:
                    self._send(404, {"error": "no"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.bodies.append(body)
                fake.prompts.append(body["messages"][0]["content"])
                tps = fake.tps_list[min(fake.n, len(fake.tps_list) - 1)]
                fake.n += 1
                ms = 1000.0 * fake.tok / tps
                fake.totals["requests"] += 1
                fake.totals["output_tokens"] += fake.tok
                fake.totals["decode_ms"] += ms
                fake.totals["drafts_offered"] += fake.drafted
                fake.totals["drafts_accepted"] += fake.accepted
                self._send(200, {"usage": {"completion_tokens": fake.tok},
                                 "choices": [{"message": {"content": "Once upon a time"}}],
                                 "timings": {"predicted_n": fake.tok, "predicted_ms": ms, "predicted_per_second": tps,
                                             "draft_n": fake.drafted, "draft_n_accepted": fake.accepted}})

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()


def test_derive_arithmetic():
    r = pg.derive({"predicted_n": 512, "predicted_per_second": 100.0, "draft_n": 470, "draft_n_accepted": 300})
    assert r["windows"] == 212
    assert abs(r["tok_per_window"] - 512 / 212) < 1e-9
    assert abs(r["accepted_per_window"] - 300 / 212) < 1e-9
    assert abs(r["drafted_per_window"] - 470 / 212) < 1e-9
    assert abs(r["accept_pct"] - 100 * 300 / 470) < 1e-9
    assert abs(r["ms_per_window"] - (1000 * 512 / 100.0) / 212) < 1e-9


def test_derive_without_draft_counters():
    r = pg.derive({"predicted_n": 100, "predicted_ms": 1000.0})
    assert r["tps"] == 100.0
    assert "tok_per_window" not in r and "accept_pct" not in r


def test_numeric_diff_nested_and_skips():
    d = pg.numeric_diff({"a": 1, "b": {"c": 2.0, "z": "x"}, "t": True}, {"a": 4, "b": {"c": 5.5, "z": "y"}, "t": False, "new": 3})
    assert d == {"a": 3, "b": {"c": 3.5}}


def test_series_pancake_discards_run1_and_snapshots(tmp_path, capsys):
    with Fake(True, [50.0, 100.0, 110.0, 90.0, 100.0]) as f:
        out = tmp_path / "p.json"
        rc = pg.main(["--port", str(f.port), "--label", "pc", "--runs", "5", "--out", str(out)])
        assert rc == 0
        res = json.loads(out.read_text())
        assert res["engine_kind"] == "pancake"
        assert [r["discarded"] for r in res["runs"]] == [True, False, False, False, False]
        assert res["summary"]["n_kept"] == 4 and res["summary"]["n_discarded"] == 1
        assert res["summary"]["tps"]["median"] == 100.0           # the 50 tok/s warm-up is not in it
        assert res["summary"]["tps"]["min"] == 90.0
        assert len(set(f.prompts)) == 1 and f.prompts[0] == pg.PROMPTS["story"]   # one fixed prompt, every time
        assert f.bodies[0]["temperature"] == 0 and f.bodies[0]["max_tokens"] == 512
        assert f.bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}
        # snapshots: before and after, and the counters' difference covers all five requests (the discarded one too)
        assert res["snapshot_before"]["totals"]["requests"] == 0 and res["snapshot_after"]["totals"]["requests"] == 5
        assert res["snapshot_diff"]["totals"]["requests"] == 5
        assert res["snapshot_diff"]["totals"]["drafts_accepted"] == 5 * 300
        assert res["snapshot_after"]["status"]["vram"]["free_mib"] == 100
        text = capsys.readouterr().out
        assert "tokens / window" in text and "server /metrics totals" in text and "run 1 (discarded)" in text
        assert "answer text: 1 distinct of 4 kept runs" in text
        assert res["runs"][0]["text_head"] == "Once upon a time" and len(res["runs"][0]["text_sha256"]) == 16
        assert len(text.strip().splitlines()) < 40                # one screen


def test_series_llama_takes_no_snapshot(tmp_path):
    with Fake(False, [100.0]) as f:
        out = tmp_path / "a.json"
        assert pg.main(["--port", str(f.port), "--label", "av", "--runs", "3", "--prompt-name", "tips", "--out", str(out)]) == 0
        res = json.loads(out.read_text())
        assert res["engine_kind"] == "llama"
        assert "snapshot_before" not in res and "snapshot_diff" not in res
        assert res["prompt_name"] == "tips" and f.prompts[0] == pg.PROMPTS["tips"]


def test_engine_flag_overrides_detection(tmp_path):
    with Fake(False, [100.0]) as f:                           # a server that does not say "strata", forced to pancake mode
        out = tmp_path / "x.json"
        assert pg.main(["--port", str(f.port), "--runs", "2", "--engine", "pancake", "--out", str(out)]) == 0
        res = json.loads(out.read_text())
        assert res["engine_kind"] == "pancake" and res["snapshot_after"]["totals"] is None   # endpoints missing: empty, no crash


def test_unreachable_server_exit_2(capsys):
    with Fake(False, [100.0]) as f:
        port = f.port
    assert pg.main(["--port", str(port), "--runs", "2", "--timeout", "2"]) == 2
    assert "no request succeeded" in capsys.readouterr().err


def test_compare(tmp_path, capsys):
    with Fake(False, [100.0, 100.0], accepted=300) as a, Fake(True, [90.0, 90.0], accepted=250) as b:
        pa, pb = tmp_path / "a.json", tmp_path / "b.json"
        assert pg.main(["--port", str(a.port), "--label", "avalon", "--runs", "3", "--out", str(pa)]) == 0
        assert pg.main(["--port", str(b.port), "--label", "pancake", "--runs", "3", "--out", str(pb)]) == 0
        capsys.readouterr()
        assert pg.main(["--compare", str(pa), str(pb)]) == 0
        text = capsys.readouterr().out
        assert "A=avalon" in text and "B=pancake" in text and "tok_per_window" in text
        ja, jb = json.loads(pa.read_text()), json.loads(pb.read_text())
        assert ja["summary"]["tok_per_window"]["median"] > jb["summary"]["tok_per_window"]["median"]


def test_server_log_mean_len_parsed(tmp_path):
    log = tmp_path / "svc.log"
    old = "0.1 I slot print_timing: id  0 | task 1 | draft acceptance = 0.10000 (    1 accepted /    10 generated), mean len =  1.10\n"
    new = "9.9 I slot print_timing: id  0 | task 9 | draft acceptance = 0.57234 (  275 accepted /  471 generated), mean len =  2.15\n"
    log.write_text("old line\n" + old)
    with Fake(False, [100.0]) as f:
        # a server that appends its line to the log for every request, as llama-server does
        orig = f.httpd.RequestHandlerClass.do_POST

        def do_POST(self):
            orig(self)
            with open(log, "a") as fh:
                fh.write(new)

        f.httpd.RequestHandlerClass.do_POST = do_POST
        out = tmp_path / "l.json"
        assert pg.main(["--port", str(f.port), "--runs", "3", "--server-log", str(log), "--out", str(out)]) == 0
        res = json.loads(out.read_text())
        assert len(res["server_log_requests"]) == 3                     # the old line before the series is not counted
        assert res["server_log_requests"][0] == {"acceptance": 0.57234, "accepted": 275, "generated": 471, "mean_len": 2.15}


def test_parse_server_log_missing_file():
    assert pg.parse_server_log("/nonexistent/none.log", 0) == []


def test_runs_below_two_rejected():
    try:
        pg.main(["--port", "1", "--runs", "1"])
    except SystemExit as e:
        assert e.code == 2
    else:
        raise AssertionError("expected SystemExit")
