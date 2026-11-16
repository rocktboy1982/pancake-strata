"""STRATA_ASYNC_COMPACT=N (opt-in; unset, empty or 0 = a compaction request is generated when it arrives, as before).

Asynchronous context compaction after Slipstream (LEADS.md "A39 async compaction (Slipstream)"): the summary an agent
asks for when its context is full is produced in the background, in an idle window, instead of on the request's clock.
Slipstream runs compaction beside the agent; with one GPU the window here is the idle time after an answer.

How it works (first version):
  * The client's compaction instruction is learned from the first compaction request it sends (the last user message
    matches STRATA_ASYNC_COMPACT_MARK, a regular expression; the default knows opencode's and Claude Code's wording).
  * When a client turn ends with a plain answer (no tool calls) and its prompt has at least N tokens, a background run
    renders the conversation + that answer + the learned instruction and generates the summary (greedy, the
    request's own template settings).  A request that arrives meanwhile cancels it (it never delays a client).
  * When the compaction request for exactly that conversation arrives (same messages, reasoning ignored), its reply
    is the stored summary: no prefill, no decode.  Anything else is generated as before.

Lossy: the stored summary was decoded greedily with the server's limits, not with the compaction request's own
sampling and max_tokens; and a precomputation costs engine time in the idle window.  At most
STRATA_ASYNC_COMPACT_KEEP (default 4) summaries are kept.  Runs on serve/tool_idle.py's hooks (it turns them on).
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import threading

from serve import tool_idle

_CFG: dict | None = None
_DEFAULT_MARK = (r"provide a detailed prompt for continuing our conversation|create a detailed summary of the "
                 r"conversation|summari[sz]e (?:this|the|our) (?:entire )?conversation|compact(?:ion)? (?:the )?"
                 r"(?:conversation|context)")

_LOCK = threading.Lock()
CACHE: "collections.OrderedDict[str, str]" = collections.OrderedDict()
STATE = {"instruction": None, "job": None, "stats": {"precomputed": 0, "served": 0, "cancelled": 0, "misses": 0}}


def config() -> dict:
    global _CFG
    if _CFG is None:
        try:
            n = max(0, int(os.environ.get("STRATA_ASYNC_COMPACT", "").strip() or 0))
        except ValueError:
            n = 0
        try:
            mark = re.compile(os.environ.get("STRATA_ASYNC_COMPACT_MARK", "").strip() or _DEFAULT_MARK, re.I)
        except re.error:
            mark = re.compile(_DEFAULT_MARK, re.I)
        try:
            keep = max(1, int(os.environ.get("STRATA_ASYNC_COMPACT_KEEP", "").strip() or 4))
        except ValueError:
            keep = 4
        _CFG = {"tokens": n, "mark": mark, "keep": keep}
    return _CFG


def enabled() -> bool:
    return config()["tokens"] > 0


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content
                         if isinstance(p, (dict, str)))
    return ""


def conversation_key(messages) -> str:
    """The conversation as the client resends it: role, text, tool calls; reasoning ignored, whitespace folded."""
    rows = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        calls = [[(c.get("function") or c).get("name"), (c.get("function") or c).get("arguments")]
                 for c in (m.get("tool_calls") or []) if isinstance(c, dict)]
        rows.append([m.get("role"), " ".join(_text(m.get("content")).split()), calls])
    s = json.dumps(rows, ensure_ascii=False, default=str)
    return hashlib.blake2b(s.encode("utf-8", "replace"), digest_size=16).hexdigest()


def is_compaction(messages) -> str | None:
    """The instruction text when the last message is a user turn asking for a compaction summary, else None."""
    if not messages or not isinstance(messages[-1], dict) or messages[-1].get("role") != "user":
        return None
    t = _text(messages[-1].get("content"))
    return t if t and config()["mark"].search(t) else None


def on_prepare(messages, tools, kwargs) -> None:
    """Service.prepare: remember this request's messages for the turn-end hook; learn the instruction; put a stored
    summary in place for Service.run when this is the compaction request it was made for."""
    tool_idle._LOCAL.serve = None
    if tool_idle.background():
        return
    tool_idle._LOCAL.prepared = {"messages": list(messages or []), "tools": tools, "kwargs": dict(kwargs or {})}
    instr = is_compaction(messages)
    if instr is None:
        return
    STATE["instruction"] = instr
    with _LOCK:
        text = CACHE.get(conversation_key(messages[:-1]) + "\x00" + instr)
    if text is None:
        STATE["stats"]["misses"] += 1
        return
    tool_idle._LOCAL.serve = text


def take_cached(prompt_len: int):
    """Service.run: the events of a stored summary for this thread's request, or None."""
    text = getattr(tool_idle._LOCAL, "serve", None)
    if text is None:
        return None
    tool_idle._LOCAL.serve = None
    STATE["stats"]["served"] += 1
    print(f"[strata] async compaction: summary served from the idle-window run ({len(text):,} characters)",
          flush=True)
    return _events(text, prompt_len)


def _events(text: str, prompt_len: int):
    from serve.frontend import Event
    yield "event", Event("content", text)
    yield "done", {"finish": "stop", "completion_tokens": 0, "reused": prompt_len, "timings": None,
                   "reasoning_tokens": 0, "reasoning_recoveries": 0, "prompt_tokens": prompt_len}


def on_request(req) -> None:
    job = STATE.get("job")
    if job is not None and not job.is_set():
        job.set()
        STATE["stats"]["cancelled"] += 1


def on_turn_end(svc, info: dict) -> None:
    cfg = config()
    instr = STATE.get("instruction")
    if info.get("calls") or instr is None or len(info.get("prompt") or []) < cfg["tokens"]:
        return
    msgs = info.get("messages")
    if not msgs or is_compaction(msgs):
        return
    fifo = getattr(svc, "fifo", None)
    convo = list(msgs) + [{"role": "assistant", "content": info.get("content") or ""}]
    key = conversation_key(convo) + "\x00" + instr
    with _LOCK:
        if key in CACHE:
            return
    cancel = threading.Event()
    STATE["job"] = cancel

    def work():
        if fifo is not None:                    # the turn that just ended still holds it for a moment
            while fifo.locked() and not cancel.is_set():
                cancel.wait(0.05)
        if cancel.is_set():
            return
        tool_idle._LOCAL.background = True
        try:
            ids, thinking, max_new = svc.prepare(convo + [{"role": "user", "content": instr}], None,
                                                 dict(info.get("kwargs") or {}))
            out, finish = [], None
            for kind, obj in svc.run(ids, thinking, None, max_new, {"temperature": 0.0}, cancel):
                if kind == "event" and getattr(obj, "kind", None) == "content":
                    out.append(obj.text)
                elif kind == "done":
                    finish = obj.get("finish") if isinstance(obj, dict) else None
            if finish == "stop" and not cancel.is_set():
                with _LOCK:
                    CACHE[key] = "".join(out)
                    while len(CACHE) > cfg["keep"]:
                        CACHE.popitem(last=False)
                STATE["stats"]["precomputed"] += 1
                print(f"[strata] async compaction: summary ready in the idle window ({len(ids):,}-token prompt)",
                      flush=True)
        except Exception as e:                  # noqa: BLE001 - background work must never take the server down
            print(f"[strata] async compaction: failed ({type(e).__name__}: {e})", flush=True)
        finally:
            tool_idle._LOCAL.background = False

    threading.Thread(target=work, daemon=True).start()


def install(tracker) -> None:
    """Register on the tool-idle tracker once."""
    if tracker is not None and on_turn_end not in tracker.turn_end_hooks:
        tracker.turn_end_hooks.append(on_turn_end)     # info carries this request's "messages" and "kwargs"
        tracker.request_hooks.append(on_request)
