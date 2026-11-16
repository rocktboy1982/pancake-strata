"""STRATA_MCP_ASYNC=1 (opt-in; unset, empty or 0 = every MCP call starts after the model's answer ended, as before).

AsyncFC-style "decode during tools" for the tools this server runs itself (serve/mcp.py, run_with_mcp in
serve/server.py; LEADS.md "A40 decode during tools (AsyncFC futures)").  AsyncFC (paper "Asynchronous LLM Function
Calling") lets the model keep decoding while a function runs and treats the result as a future.  The first version
here is the part our loop can do without a model change: an MCP call is started the moment its tool_call event is
complete, while the engine is still decoding the rest of the answer (more calls, text); after the answer ends the
round waits for those futures instead of starting the calls then.  With one call per answer the overlap is the
decode of the answer's tail; with several, the calls also run side by side.

Lossy in one way (side effects): a call is started before the loop knows whether the round runs (an answer that
also calls a client tool, a cancel or the last round skips the round).  Such a call still ran; its result is
dropped and the client's view says "not run" as before.  Read-only tools are unaffected.

`start_call()` is also the one place the speculative read-only path (serve/mcp_spec.py, STRATA_MCP_SPEC) plugs in.
"""
from __future__ import annotations

import os
import threading

from serve.mcp import McpCancelled

_ON: bool | None = None


def enabled() -> bool:
    """STRATA_MCP_ASYNC=1, read once."""
    global _ON
    if _ON is None:
        _ON = os.environ.get("STRATA_MCP_ASYNC", "").strip() == "1"
    return _ON


def start_call(hub, call, cancel, spec=None) -> tuple[threading.Thread, dict]:
    """Run `call` on a thread -> (thread, box); box gets "r" (hub.call's dict) or "cancelled".  With `spec`
    (serve/mcp_spec.Speculator), a speculative result of the same call is taken instead of running it again, and the
    finished call is reported to it (which may start the predicted next call)."""
    box: dict = {}

    def work():
        try:
            r = spec.take(call.name, call.arguments) if spec is not None else None
            if r is None:
                r = hub.call(call.name, call.arguments, cancel)
            elif isinstance(r, dict):
                r = {**r, "speculative": True}
            box["r"] = r
            if spec is not None:
                spec.observe(call.name, call.arguments)
        except McpCancelled:
            box["cancelled"] = True

    t = threading.Thread(target=work, daemon=True)
    t.start()
    return t, box


class Futures:
    """The calls of one answer, started as their tool_call events arrive (call id -> (thread, box))."""

    def __init__(self, hub, cancel, spec=None):
        self.hub, self.cancel, self.spec = hub, cancel, spec
        self.jobs: dict = {}

    def start(self, call) -> None:
        if call.id not in self.jobs:
            self.jobs[call.id] = start_call(self.hub, call, self.cancel, self.spec)

    def take(self, call):
        """(thread, box) of a started call (removed), or None."""
        return self.jobs.pop(call.id, None)

    def clear(self) -> int:
        """Forget the calls of a round that does not run (they finish on their own; results dropped)."""
        n = len(self.jobs)
        self.jobs.clear()
        return n
