"""tools/pancake/mcp_bench_server.py (read-only MCP server for bench/tasks/mcp-repo.json) and task_gate.add_mcp_servers."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mcp_bench_server as srv  # noqa: E402


def _tree(d):
    (Path(d) / "sub").mkdir()
    (Path(d) / "sub" / "a.txt").write_text("one\nkTarget = 7\nthree\n")
    (Path(d) / "b.py").write_text("x = 1\n")


def call(root, name, args):
    return srv.handle(root, {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": args}})["result"]


def test_tools_read_list_grep_and_confinement():
    with tempfile.TemporaryDirectory() as d:
        _tree(d)
        assert call(d, "list_dir", {"path": "."})["content"][0]["text"] == "b.py\nsub/"
        assert call(d, "read_file", {"path": "sub/a.txt", "start": 2, "lines": 1})["content"][0]["text"] == "2: kTarget = 7"
        assert call(d, "grep", {"pattern": "kTarget"})["content"][0]["text"] == "sub/a.txt:2: kTarget = 7"
        bad = call(d, "read_file", {"path": "../" + os.path.basename(d) + "x/secret"})
        assert bad["isError"] is True
        assert call(d, "nope", {})["isError"] is True


def test_protocol_over_stdio():
    with tempfile.TemporaryDirectory() as d:
        _tree(d)
        lines = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
                 {"jsonrpc": "2.0", "method": "notifications/initialized"},
                 {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                 {"jsonrpc": "2.0", "id": 3, "method": "bogus"}]
        out = subprocess.run([sys.executable, "-I", str(HERE / "mcp_bench_server.py"), d], input="\n".join(map(json.dumps, lines)) + "\n",
                             capture_output=True, text=True, timeout=30).stdout.splitlines()
        got = [json.loads(x) for x in out]
        assert [g["id"] for g in got] == [1, 2, 3]          # the notification gets no reply
        assert {t["name"] for t in got[1]["result"]["tools"]} == {"list_dir", "read_file", "grep"}
        assert got[2]["error"]["code"] == -32601


def test_task_gate_adds_mcp_servers(monkeypatch):
    if not hasattr(os, "sysconf"):
        monkeypatch.setattr(os, "sysconf", lambda k: 100, raising=False)   # abtest reads the clock tick at import (Linux)
    import task_gate as tg
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "arm.json"
        p.write_text(json.dumps({"args": []}))
        tg.add_mcp_servers(str(p), {"mcp_servers": {"r": {"command": "<PYTHON>", "args": ["<REPO>/x.py"]}}, "mcp": {"max_rounds": 8}})
        cfg = json.loads(p.read_text())
        assert cfg["mcp"] == {"max_rounds": 8}
        assert "<" not in json.dumps(cfg["mcp_servers"]) and cfg["mcp_servers"]["r"]["args"][0].endswith("/x.py")
        p.write_text(json.dumps({"args": []}))
        tg.add_mcp_servers(str(p), {})                      # no block: untouched
        assert json.loads(p.read_text()) == {"args": []}
