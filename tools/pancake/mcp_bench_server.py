#!/usr/bin/env python3
"""mcp_bench_server: a small read-only MCP server (stdio, newline-delimited JSON-RPC 2.0) over one directory tree.

    python3 tools/pancake/mcp_bench_server.py ROOT

It exists so the server's own MCP tool loop (serve/server.py run_with_mcp; the STRATA_MCP_ASYNC / STRATA_MCP_SPEC /
STRATA_MCP_EARLY_STOP switches act only there) can be measured with tools/pancake/task_gate.py and the task set
bench/tasks/mcp-repo.json (MOD_PLAN.md section 5 item 2).  Tools, all read-only and confined to ROOT:
  list_dir(path)                       names in a directory (directories end in "/")
  read_file(path, start=1, lines=80)   numbered lines of a text file
  grep(pattern, path=".", max=40)      "file:line: text" matches of a regular expression under path
Names the speculator treats as read-only (read/list/grep) on purpose.  Stdlib only; never writes.
"""
import json
import os
import re
import sys

PROTOCOL = "2025-06-18"
TOOLS = [
    {"name": "list_dir", "description": "List the entries of a directory of the repository (directories end in /).",
     "inputSchema": {"type": "object", "properties": {"path": {"type": "string", "description": "directory, relative to the repository root"}},
                     "required": ["path"]}},
    {"name": "read_file", "description": "Read numbered lines of a text file of the repository.",
     "inputSchema": {"type": "object", "properties": {
         "path": {"type": "string", "description": "file, relative to the repository root"},
         "start": {"type": "integer", "description": "first line (1-based, default 1)"},
         "lines": {"type": "integer", "description": "number of lines (default 80, at most 400)"}}, "required": ["path"]}},
    {"name": "grep", "description": "Search files of the repository for a regular expression; returns file:line: text.",
     "inputSchema": {"type": "object", "properties": {
         "pattern": {"type": "string", "description": "Python regular expression"},
         "path": {"type": "string", "description": "file or directory to search, relative to the root (default .)"},
         "max": {"type": "integer", "description": "most matches to return (default 40)"}}, "required": ["pattern"]}},
]
SKIP_DIRS = {".git", "build", "node_modules", "__pycache__", ".venv"}


def inside(root, rel):
    """The absolute path of `rel` under root, or None when it escapes root."""
    p = os.path.realpath(os.path.join(root, rel or "."))
    r = os.path.realpath(root)
    return p if p == r or p.startswith(r + os.sep) else None


def list_dir(root, a):
    p = inside(root, a.get("path", "."))
    if p is None or not os.path.isdir(p):
        raise ValueError("no such directory")
    names = sorted(os.listdir(p))
    return "\n".join(n + ("/" if os.path.isdir(os.path.join(p, n)) else "") for n in names)


def read_file(root, a):
    p = inside(root, a.get("path", ""))
    if p is None or not os.path.isfile(p):
        raise ValueError("no such file")
    start = max(1, int(a.get("start") or 1))
    n = min(400, max(1, int(a.get("lines") or 80)))
    out = []
    with open(p, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            if i >= start + n:
                break
            if i >= start:
                out.append(f"{i}: {line.rstrip()}")
    return "\n".join(out) if out else "(no lines in that range)"


def grep(root, a):
    rx = re.compile(a["pattern"])
    p = inside(root, a.get("path", "."))
    if p is None or not os.path.exists(p):
        raise ValueError("no such path")
    cap = min(200, max(1, int(a.get("max") or 40)))
    files = [p] if os.path.isfile(p) else []
    if not files:
        for d, dirs, fs in os.walk(p):
            dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS and not x.startswith("build-"))
            files += [os.path.join(d, f) for f in sorted(fs)]
    out = []
    for f in files:
        try:
            with open(f, encoding="utf-8", errors="strict") as h:
                for i, line in enumerate(h, 1):
                    if rx.search(line):
                        out.append(f"{os.path.relpath(f, root).replace(os.sep, '/')}:{i}: {line.strip()[:200]}")
                        if len(out) >= cap:
                            return "\n".join(out)
        except (UnicodeDecodeError, OSError):
            continue
    return "\n".join(out) if out else "(no matches)"


HANDLERS = {"list_dir": list_dir, "read_file": read_file, "grep": grep}


def handle(root, msg):
    """One JSON-RPC message -> the reply dict, or None for a notification."""
    mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
    if mid is None:
        return None
    if method == "initialize":
        result = {"protocolVersion": params.get("protocolVersion") or PROTOCOL, "capabilities": {"tools": {}},
                  "serverInfo": {"name": "pancake-mcp-bench", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        fn = HANDLERS.get(params.get("name"))
        try:
            if fn is None:
                raise ValueError("unknown tool")
            text, err = fn(root, params.get("arguments") or {}), False
        except (ValueError, KeyError, re.error, OSError) as e:
            text, err = f"error: {e}", True
        result = {"content": [{"type": "text", "text": text}], "isError": err}
    elif method == "ping":
        result = {}
    else:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    root = os.path.abspath(argv[0] if argv else ".")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        reply = handle(root, msg)
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
