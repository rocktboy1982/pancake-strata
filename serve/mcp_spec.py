"""STRATA_MCP_SPEC=N (opt-in; unset, empty or 0 = no call ever runs before the model writes it).

Speculative read-only tool execution for the tools this server runs itself (serve/mcp.py; LEADS.md "Speculative
read-only tool execution (Atomix, PASTE, speculative actions)").  PASTE (pattern-aware speculative tool execution)
learns which call tends to follow which and runs the predicted next call while the model is still thinking; Atomix
and "speculative actions" add the rule that only side-effect-free calls may run ahead.  First version here:

  * A process-wide successor table: (tool, arguments) -> how often each (tool, arguments) was the very next call in
    the same request's tool loop.  Learned from every finished MCP call while the switch is on.
  * After a call finishes, the most frequent successor is started at once when it was seen at least N times
    (STRATA_MCP_SPEC=N) and its tool name looks read-only (STRATA_MCP_SPEC_TOOLS, a regular expression on the tool
    part of `<server>__<tool>`; default: read/get/list/search/find/grep/glob/view/fetch/stat/ls/cat/head/tail/show/
    describe/query).
  * When the model then writes exactly that call (same tool, same arguments), its result is the speculative one if
    it finished no more than STRATA_MCP_SPEC_TTL_S seconds ago (default 30); otherwise the call runs as usual.  A
    prediction the model never makes is dropped.

Lossy in the PASTE sense: a read-only tool's answer can change between the speculative run and the moment the model
asks (a file written in between), and the name rule can misjudge a tool.  The successor table is bounded
(STRATA_MCP_SPEC_MAX entries, default 4096, oldest first out).
"""
from __future__ import annotations

import collections
import json
import os
import re
import threading
import time

from serve.mcp import McpCancelled

_CFG: dict | None = None
_DEFAULT_TOOLS = (r"(^|_)(read|get|list|search|find|grep|glob|view|fetch|stat|ls|cat|head|tail|show|describe|"
                  r"query)($|_)")

_LOCK = threading.Lock()
SUCCESSORS: "collections.OrderedDict[str, collections.Counter]" = collections.OrderedDict()
STATS = {"started": 0, "hits": 0, "stale": 0, "misses": 0}


def config() -> dict:
    """{"min": int (0 = off), "tools": compiled regex, "ttl": float, "max": int}; read once."""
    global _CFG
    if _CFG is None:
        try:
            n = max(0, int(os.environ.get("STRATA_MCP_SPEC", "").strip() or 0))
        except ValueError:
            n = 0
        try:
            rx = re.compile(os.environ.get("STRATA_MCP_SPEC_TOOLS", "").strip() or _DEFAULT_TOOLS, re.IGNORECASE)
        except re.error:
            rx = re.compile(_DEFAULT_TOOLS, re.IGNORECASE)
        try:
            ttl = float(os.environ.get("STRATA_MCP_SPEC_TTL_S", "").strip() or 30.0)
        except ValueError:
            ttl = 30.0
        try:
            cap = max(16, int(os.environ.get("STRATA_MCP_SPEC_MAX", "").strip() or 4096))
        except ValueError:
            cap = 4096
        _CFG = {"min": n, "tools": rx, "ttl": ttl, "max": cap}
    return _CFG


def enabled() -> bool:
    return config()["min"] > 0


def key(name: str, arguments) -> str:
    try:
        args = json.dumps(arguments if isinstance(arguments, dict) else {}, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        args = repr(arguments)
    return f"{name}\x00{args}"


def read_only(name: str, rx=None) -> bool:
    rx = rx or config()["tools"]
    tool = name.split("__", 1)[1] if "__" in name else name
    return rx.search(tool) is not None


def learn(prev: str, nxt: str, cap: int) -> None:
    with _LOCK:
        c = SUCCESSORS.pop(prev, None) or collections.Counter()
        c[nxt] += 1
        SUCCESSORS[prev] = c                           # most recently used at the end
        while len(SUCCESSORS) > cap:
            SUCCESSORS.popitem(last=False)


def predict(prev: str, min_count: int) -> str | None:
    with _LOCK:
        c = SUCCESSORS.get(prev)
        if not c:
            return None
        nxt, n = c.most_common(1)[0]
    return nxt if n >= min_count else None


class Speculator:
    """One request's tool loop: the previous call, and the speculative calls in flight or done."""

    def __init__(self, hub, cancel, cfg: dict | None = None):
        self.hub, self.cancel = hub, cancel
        self.cfg = cfg or config()
        self.prev: str | None = None
        self.pending: dict = {}                        # key -> (thread, box)

    def take(self, name: str, arguments):
        """The speculative result of this exact call, or None (then the caller runs it)."""
        k = key(name, arguments)
        job = self.pending.pop(k, None)
        if job is None:
            return None
        worker, box = job
        worker.join(max(0.0, float(self.hub.settings.get("timeout_s", 60.0))) if hasattr(self.hub, "settings")
                    else None)
        r, done_at = box.get("r"), box.get("t")
        if r is None or done_at is None:
            return None
        if time.monotonic() - done_at > self.cfg["ttl"]:
            STATS["stale"] += 1
            return None
        STATS["hits"] += 1
        return r

    def observe(self, name: str, arguments) -> None:
        """A call finished: learn it as the successor of the previous one, then start the predicted next call."""
        k = key(name, arguments)
        if self.prev is not None:
            learn(self.prev, k, self.cfg["max"])
        self.prev = k
        nxt = predict(k, self.cfg["min"])
        if nxt is None or nxt in self.pending or self.cancel.is_set():
            return
        tname, _, args = nxt.partition("\x00")
        if not read_only(tname, self.cfg["tools"]):
            return
        try:
            arguments2 = json.loads(args)
        except ValueError:
            return
        box: dict = {}

        def work():
            try:
                box["r"] = self.hub.call(tname, arguments2, self.cancel)
                box["t"] = time.monotonic()
            except McpCancelled:
                pass

        t = threading.Thread(target=work, daemon=True)
        t.start()
        self.pending[nxt] = (t, box)
        STATS["started"] += 1

    def close(self) -> None:
        STATS["misses"] += len(self.pending)
        self.pending.clear()
