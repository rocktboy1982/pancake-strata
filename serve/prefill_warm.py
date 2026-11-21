"""STRATA_PREFILL_TYPING=1 (opt-in; unset, empty or 0 = no /v1/prefill, no warm requests, the server as before).

Read the prompt while the user types (LEADS.md Part Q1; idea from Strata-HotRod commit d0aa072f6 and 7b392a7a5, MIT,
which took it from Stream2LLM, MLSys 2026).  The switch is not named STRATA_PREFILL_WARM: the engine already reads that
name (the prompt path's GEMM warm-up, src/prefill/prefill.cpp).

A client posts the draft it has typed so far to `POST /v1/prefill` (the body of a chat completion), or sends a normal
`/v1/chat/completions` body with `"strata_prefill": true` or the header `X-Strata-Prefill: 1`.  The server then:

  1. renders the draft and a probe-extended draft (the last user message plus PROBE) and keeps the tokens they share,
     minus one (a recurrent state cannot rewind, and the tokenizer may merge across the boundary): the STABLE prefix.
     A draft whose stable prefix has fewer than STRATA_PREFILL_TYPING_MIN tokens (default 512) is not warmed.
  2. runs it as a `max_tokens 1` request with `strata_prefix: {"tokens": N}`: the engine reads the stable prefix and keeps
     the checkpoint there pinned (pin=N; needs --prompt-cache 3 or more), so the real request that follows reads only what
     lies past it.  The answer of the warm is never used.
  3. a NEWER warm, or a real request, cancels a warm that is still queued (a warm that has begun reading is left to end,
     one token past the prompt).  A real request is never delayed by more than the one warm that already runs.

The answers are unchanged by construction: the engine reads the same tokens, only earlier (the same rule as any pinned
prefix).  The gain is the time to first token of a long first message (up to the whole prefill); nothing for API and
agent clients, none for decode.  Pure logic here (no engine, no network), tested by serve/test_prefill_warm.py; the
wiring is in serve/server.py (Handler._openai) and serve/web/app.js (warms on a 1.2 s typing pause or a paste).
"""
from __future__ import annotations

import copy
import os
import threading

PROBE = " · and then"       # appended to the last user message to find where the draft's tokens stop being stable
DEFAULT_MIN_TOKENS = 512


def enabled(env=None) -> bool:
    """STRATA_PREFILL_TYPING: unset, empty or 0 is off."""
    v = (os.environ if env is None else env).get("STRATA_PREFILL_TYPING", "")
    return v not in ("", "0")


def min_tokens(env=None) -> int:
    v = (os.environ if env is None else env).get("STRATA_PREFILL_TYPING_MIN", "")
    try:
        n = int(v)
    except (TypeError, ValueError):
        return DEFAULT_MIN_TOKENS
    return n if n > 0 else DEFAULT_MIN_TOKENS


def is_warm(req, headers=None) -> bool:
    """A chat body that asks to be warmed, not answered: `strata_prefill: true` or the X-Strata-Prefill: 1 header."""
    if isinstance(req, dict) and req.get("strata_prefill") is True:
        return True
    return headers is not None and str(headers.get("X-Strata-Prefill", "")).strip() == "1"


def common_prefix(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def stable_prefix(draft_ids, probe_ids) -> int:
    """Tokens the draft and its probe-extended twin share, minus one (0 when nothing is left)."""
    return max(0, common_prefix(draft_ids, probe_ids) - 1)


def probe_messages(messages):
    """`messages` with PROBE added to the end of the last message's text, or None when that message cannot take it (it is
    not a plain-text user turn, or carries pictures: those drafts are not warmed)."""
    if not messages or not isinstance(messages[-1], dict):
        return None
    last = messages[-1]
    if last.get("role") != "user":
        return None
    content = last.get("content")
    out = copy.deepcopy(messages)
    if isinstance(content, str):
        out[-1]["content"] = content + PROBE
        return out
    if isinstance(content, list) and content and all(isinstance(p, dict) and p.get("type") == "text" for p in content):
        out[-1]["content"][-1]["text"] = str(content[-1].get("text", "")) + PROBE
        return out
    return None


class Plan:
    """What to do with a warm request: `skip` is the reason it is not run (or None), `tokens` the stable prefix."""

    def __init__(self, skip=None, tokens=0, prompt_tokens=0):
        self.skip, self.tokens, self.prompt_tokens = skip, tokens, prompt_tokens

    def body(self) -> dict:
        """The answer of a warm that is not run."""
        return {"object": "prefill", "warmed": False, "reason": self.skip, "prompt_tokens": self.prompt_tokens}


def plan(messages, tools, kwargs, encode, min_tok=None, already_pinned=False) -> Plan:
    """`encode(messages, tools, kwargs) -> ids` renders and tokenizes like a real request (Service.encode_prompt with
    store=False).  `already_pinned`: the body names a strata_prefix of its own, which is kept (the warm still reads)."""
    floor = min_tokens() if min_tok is None else min_tok
    probe = probe_messages(messages)
    if probe is None:
        return Plan("the last message is not a plain-text user turn")
    draft_ids = encode(messages, tools, kwargs)
    n = stable_prefix(draft_ids, encode(probe, tools, kwargs))
    if already_pinned:
        return Plan(None, 0, len(draft_ids))
    if n < floor:
        return Plan(f"the stable prefix is {n} tokens, under the {floor} that are worth reading ahead", n, len(draft_ids))
    if n >= len(draft_ids) - 1:                     # the same bound Service.resolve_prefix applies
        return Plan("no usable prefix", n, len(draft_ids))
    return Plan(None, n, len(draft_ids))


def apply(req: dict, p: Plan) -> dict:
    """The warm request: one token, not streamed, the stable prefix pinned.  Returns `req` changed in place."""
    req["max_tokens"] = 1
    req.pop("max_completion_tokens", None)
    req["stream"] = False
    req.pop("stream_options", None)
    req.pop("tools_stream", None)
    # strata_mcp stays: the MCP tools are part of the real request's prompt, so the warm must render them too (it reads
    # one token, so no tool is ever called)
    req["strata_prefill"] = True
    if p.tokens > 0 and req.get("strata_prefix") is None:
        req["strata_prefix"] = {"tokens": p.tokens}
    return req


class Gate:
    """Which warm requests are still queued, so a newer warm or a real request can cancel them.  `is_queued(cancel)`
    says whether the warm owning that cancel event has not begun reading (the server answers it from the request table)."""

    def __init__(self, is_queued=None):
        self._lock = threading.Lock()
        self._queued = []                           # cancel events of the warms that may still be queued, oldest first
        self.probe = is_queued or (lambda cancel: True)   # the server sets it per warm (see Handler._openai)
        self.stats = {"warms": 0, "cancelled_by_warm": 0, "cancelled_by_real": 0}

    def _cancel_queued(self, why: str) -> None:
        keep = []
        for c in self._queued:
            if c.is_set():
                continue
            if self.probe(c):
                c.set()
                self.stats[why] += 1
            else:
                keep.append(c)                      # reading already: left to end
        self._queued = keep

    def begin(self, cancel: threading.Event) -> None:
        """A warm starts: the older queued ones are cancelled, this one is registered."""
        with self._lock:
            self._cancel_queued("cancelled_by_warm")
            self._queued.append(cancel)
            self.stats["warms"] += 1

    def end(self, cancel: threading.Event) -> None:
        with self._lock:
            self._queued = [c for c in self._queued if c is not cancel]

    def note_real(self) -> None:
        """A real request arrived: the warms still queued give their place up."""
        with self._lock:
            self._cancel_queued("cancelled_by_real")


GATE = Gate()
