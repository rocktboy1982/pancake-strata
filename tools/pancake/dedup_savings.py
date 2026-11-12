"""dedup_savings.py - pancake-strata: how much would ContextPilot's block dedup (serve/context_dedup.py) cut from real
opencode prompts?  Offline: no server, no GPU.  Reads opencode's local database READ-ONLY (only the message and part
tables), rebuilds each session's last prompt as template messages (user / assistant text, assistant tool calls,
tool results; no system prompt - opencode does not store it, so the system-prompt matches are not counted), runs
dedup_messages on it and prints totals.  Prints numbers only, never prompt text.

    python tools/pancake/dedup_savings.py [--db ~/.local/share/opencode/opencode.db] [--min-chars 20000]

--ab URL (a running pancake server, e.g. http://127.0.0.1:8080): a prefill A/B on the K sessions that save the most.  Each
prompt is sent with max_tokens 1, once with the request's "context_dedup": false and once true, interleaved, R rounds;
a fresh nonce at the start of the first user message keeps the conversation cache from serving either arm.  Prints
the median wall time per arm (dominated by the prompt read).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
import time
import urllib.request
import uuid
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from serve import context_dedup as cd  # noqa: E402


def sessions(db: str):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    roles = {}
    order = defaultdict(list)
    for mid, sid, t, data in c.execute("select id, session_id, time_created, data from message order by time_created, id"):
        roles[mid] = json.loads(data).get("role")
        order[sid].append(mid)
    parts = defaultdict(list)
    for mid, data in c.execute("select message_id, data from part order by time_created, id"):
        parts[mid].append(json.loads(data))
    for sid, mids in order.items():
        msgs = []
        for mid in mids:
            role = roles.get(mid)
            ps = parts.get(mid, [])
            if any(p.get("type") == "compaction" for p in ps):
                msgs = []                                  # the prompt starts again after a compaction
            text = "\n".join(p.get("text", "") for p in ps if p.get("type") == "text" and not p.get("synthetic"))
            tools = [p for p in ps if p.get("type") == "tool" and (p.get("state") or {}).get("status") == "completed"]
            if role == "user":
                if text:
                    msgs.append({"role": "user", "content": text})
                continue
            if role != "assistant":
                continue
            if text or tools:
                msgs.append({"role": "assistant", "content": text,
                             "tool_calls": [{"function": {"name": p.get("tool") or "tool", "arguments": {}}}
                                            for p in tools]})
            for p in tools:
                out = (p.get("state") or {}).get("output")
                msgs.append({"role": "tool", "content": out if isinstance(out, str) else ""})
        yield sid, msgs


def ab(url: str, picked: list, rounds: int) -> None:
    def send(msgs, dedup):
        nonce = f"[run {uuid.uuid4().hex}]\n"
        msgs = [dict(m) for m in msgs]
        first = next(i for i, m in enumerate(msgs) if m["role"] == "user")
        msgs[first]["content"] = nonce + msgs[first]["content"]
        for m in msgs:                                      # OpenAI wire form
            if m.get("tool_calls"):
                m["tool_calls"] = [{"id": f"c{k}", "type": "function",
                                    "function": {"name": c["function"]["name"], "arguments": "{}"}}
                                   for k, c in enumerate(m["tool_calls"])]
        k = {}
        for i, m in enumerate(msgs):
            if m.get("tool_calls"):
                k = {"ids": [c["id"] for c in m["tool_calls"]], "n": 0}
            elif m["role"] == "tool":
                ids = k.get("ids") or ["c0"]
                m["tool_call_id"] = ids[min(k.get("n", 0), len(ids) - 1)]
                k["n"] = k.get("n", 0) + 1
        body = json.dumps({"messages": msgs, "max_tokens": 1, "temperature": 0, "context_dedup": dedup,
                           "chat_template_kwargs": {"enable_thinking": False}}).encode()
        req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", body, {"Content-Type": "application/json"})
        t = time.monotonic()
        with urllib.request.urlopen(req, timeout=1800) as r:
            usage = json.loads(r.read()).get("usage", {})
        return time.monotonic() - t, usage.get("prompt_tokens")
    for sid, msgs, frac in picked:
        times = {False: [], True: []}
        toks = {}
        for _ in range(rounds):
            for arm in (False, True):
                dt, pt = send(msgs, arm)
                times[arm].append(dt)
                toks[arm] = pt
        off, on = statistics.median(times[False]), statistics.median(times[True])
        print(f"session {sid[:12]}: saved {100 * frac:.1f}% chars; prompt tokens {toks[False]} -> {toks[True]}; "
              f"median {off:.2f} s -> {on:.2f} s ({100 * (off - on) / off:+.1f}%)", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default=str(Path.home() / ".local/share/opencode/opencode.db"))
    ap.add_argument("--min-chars", type=int, default=20000, help="count sessions whose last prompt is at least this long")
    ap.add_argument("--ab", metavar="URL", help="prefill A/B against a running server (see the module text)")
    ap.add_argument("--top", type=int, default=3, help="--ab: this many sessions, the ones that save the most")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-chars", type=int, default=300000, help="--ab: skip prompts longer than this (context)")
    a = ap.parse_args(argv)
    n = before = saved = blocks = results = hit_sessions = 0
    per, picked = [], []
    for sid, msgs in sessions(a.db):
        size = sum(len(m["content"]) for m in msgs if isinstance(m.get("content"), str))
        if size < a.min_chars:
            continue
        _, res = cd.dedup_messages(msgs)
        n += 1
        before += size
        saved += res.chars_saved
        blocks += res.blocks_deduped
        results += res.results_deduped
        hit_sessions += bool(res.chars_saved)
        per.append(res.chars_saved / size)
        if res.chars_saved and size <= a.max_chars:
            picked.append((sid, msgs, res.chars_saved / size))
    if not n:
        print("no session long enough")
        return 0
    per.sort()
    print(f"sessions >= {a.min_chars:,} chars: {n}; with something replaced: {hit_sessions}")
    print(f"characters in their last prompts: {before:,}; dedup removes {saved:,} ({100 * saved / before:.1f}%)")
    print(f"blocks replaced: {blocks:,}; whole tool results replaced: {results:,}")
    print(f"per session saved: median {100 * per[len(per) // 2]:.1f}%, p90 {100 * per[int(len(per) * 0.9)]:.1f}%, "
          f"max {100 * per[-1]:.1f}%")
    if a.ab:
        picked.sort(key=lambda x: -x[2])
        ab(a.ab, picked[:a.top], a.rounds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
