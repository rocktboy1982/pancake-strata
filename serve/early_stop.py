"""STRATA_MCP_EARLY_STOP=N (opt-in; unset, empty or 0 = the tool loop runs up to max_rounds as before).

Agent-run early termination for the tool loop this server runs itself (run_with_mcp in serve/server.py; LEADS.md
"Agent-run early termination (EarlyEval, EET)").  EarlyEval stops runs a predictor judges hopeless; EET (early exit
for agent trajectories) cuts runs that stopped making progress.  Neither predictor is available here, so the first
version is the progress rule from their descriptions: a round makes progress when at least one of its calls is new
(tool + arguments never called before in this request) or returns a result never seen before.  After N rounds in a
row without progress the loop goes to its final answer the same way it does at max_rounds (no tools offered, a
system note asks for the answer from the evidence so far).

Lossy: a run that would have found something on a repeated call is cut.  Deterministic: the same rounds give the same
decision.
"""
from __future__ import annotations

import hashlib
import json
import os

_N: int | None = None


def patience() -> int:
    """STRATA_MCP_EARLY_STOP=N, read once; 0 = off."""
    global _N
    if _N is None:
        try:
            _N = max(0, int(os.environ.get("STRATA_MCP_EARLY_STOP", "").strip() or 0))
        except ValueError:
            _N = 0
    return _N


NOTE = ("The last tool calls returned no new information. No tools are available for this final reply. Answer the "
        "user's original request in their language using the results already obtained. Explain any gaps in the "
        "available evidence. Do not request or emit further tool calls.")


def _digest(obj) -> str:
    try:
        s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        s = repr(obj)
    return hashlib.blake2b(s.encode("utf-8", "replace"), digest_size=12).hexdigest()


class Progress:
    """Rounds without progress, for one request's tool loop."""

    def __init__(self, n: int):
        self.n = n
        self.calls: set = set()
        self.results: set = set()
        self.stale = 0

    def round(self, calls, results) -> bool:
        """Record one round (calls with .name/.arguments, their results) -> True when the loop should stop."""
        new = False
        for c in calls:
            k = _digest([getattr(c, "name", None), getattr(c, "arguments", None)])
            if k not in self.calls:
                self.calls.add(k)
                new = True
        for r in results:
            k = _digest(r)
            if k not in self.results:
                self.results.add(k)
                new = True
        self.stale = 0 if new else self.stale + 1
        return self.n > 0 and self.stale >= self.n
