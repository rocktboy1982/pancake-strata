#!/usr/bin/env python3
"""nq_clients.py - the three small clients the night queue (tools/pancake/night_queue.py) runs against a server it started.

Standard library only (plus ttft_bench.py and ab_engine.py of this checkout).  They start nothing, build nothing and touch no
GPU: they only send HTTP requests to the port they are given.  One invocation = one "pass" of the queue.

  decode   one round over the five prompts of bench/results/2026-11-11-vs-avalon/cmp_bench.py (thinking off, greedy, 512
           tokens, the same request body), per request: decode tok/s, draft counters, prompt tokens / tok/s / ms.  Each
           prompt carries its class (code or prose).  Output: {"kind": "decode", "rows": [...]}.
  session  the ttft_bench scenario B agent session (10 turns, about 1,500 words added per turn, the whole conversation
           resent) with SIDE requests between the turns: short, unrelated conversations of their own (a sub-agent or a
           title request).  Without them a conversation cache never has to choose, and the admission / park levers of
           pancake-docs/TTFT_LEVERS.md would show nothing.  Output: {"kind": "session", "sessions": [...], "summary": {...}}.
  depth    single prompts of a given depth (--sizes 1k,32k,...), then a decode of --max-tokens: TTFT, prompt tok/s and
           decode tok/s / draft acceptance AT that depth (what the K/V ring changes).  Run 1 of each size is discarded.
           Output: {"kind": "depth", "sizes": [...]}.

    python3 tools/pancake/nq_clients.py decode  --port 8080 --out decode.json
    python3 tools/pancake/nq_clients.py session --port 8080 --out session.json --sessions 3 --side 1
    python3 tools/pancake/nq_clients.py depth   --port 8080 --out depth.json --sizes 1k,32k,64k,120k --runs 3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import ttft_bench as tb  # noqa: E402

# the five prompts of cmp_bench.py (bench/results/2026-11-11-vs-avalon/cmp_bench.py), with the class each belongs to
DECODE_PROMPTS = [
    ("story", "prose", "Write a short story (about 400 words) about a lighthouse keeper."),
    ("js", "code", "Write a JavaScript debounce function with JSDoc and a usage example."),
    ("tips", "prose", "List 15 practical tips for reducing cloud costs, one line each."),
    ("moe", "prose", "Explain in about 300 words how a mixture-of-experts language model works."),
    ("lru", "code", "Write a Python class implementing an LRU cache with type hints and tests."),
]


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def post_json(host: str, port: int, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(f"http://{host}:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        j = json.load(r)
    j["_wall_s"] = time.time() - t0
    return j


def run_decode(a) -> dict:
    rows = []
    for rnd in range(1, a.rounds + 1):
        for name, cls, prompt in DECODE_PROMPTS:
            body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": a.max_tokens, "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False}}
            try:
                j = post_json(a.host, a.port, body, a.timeout)
            except Exception as e:                                   # noqa: BLE001
                print(f"decode {name}: ERROR {e!r}", file=sys.stderr, flush=True)
                rows.append({"round": rnd, "prompt": name, "class": cls, "error": repr(e)[:200]})
                continue
            t = j.get("timings") or {}
            row = {"round": rnd, "prompt": name, "class": cls, "tok": t.get("predicted_n"),
                   "tps": t.get("predicted_per_second"), "accepted": t.get("draft_n_accepted"), "drafted": t.get("draft_n"),
                   "prompt_n": t.get("prompt_n"), "prompt_tps": t.get("prompt_per_second"), "prompt_ms": t.get("prompt_ms"),
                   "wall_s": round(j["_wall_s"], 3)}
            rows.append(row)
            tps = row["tps"]
            print(f"decode r{rnd} {name} ({cls}): {row['tok']} tok @ {tps if tps is None else round(tps, 1)} t/s", flush=True)
    return {"kind": "decode", "rows": rows, "max_tokens": a.max_tokens}


def side_prompt(rng: random.Random) -> list:
    """A short unrelated conversation of its own: a fresh id and about 150 words, so it never shares a prefix."""
    words = tb.long_prompt(rng.randrange(100, 160))
    return [{"role": "system", "content": f"(side task {uuid.uuid4().hex[:12]}) You answer in one short sentence."},
            {"role": "user", "content": words}]


def run_session(a) -> dict:
    cl = tb.Client(a.host, a.port, a.api_key, think=False, timeout=a.timeout, model=a.model)
    rng = random.Random(a.seed or None)
    chunks = tb.agent_chunks(a.turns, a.words)
    sessions = []
    for s in range(a.sessions):
        nonce = uuid.uuid4().hex[:12]
        msgs = [{"role": "system", "content": f"(session {nonce})\n{tb.SYSTEM}"},
                {"role": "user", "content": "I will show you files from a project one at a time. After each file, answer in "
                                            "one or two sentences what it is for. Do not call tools."}]
        rows, side, stopped = [], [], None
        for t in range(1, a.turns + 1):
            name, body = chunks[t - 1]
            msgs.append({"role": "user", "content": f"File {name}:\n{body}\n\nWhat is this file for? Two sentences."})
            r = cl.chat(msgs, a.max_tokens, tools=tb.TOOLS)
            d = tb.digest(r)
            d["turn"] = t
            if rows and rows[-1]["prompt_tokens"] is not None and d["reused_tokens"] is not None:
                d["reread_tokens"] = max(0, rows[-1]["prompt_tokens"] + (rows[-1]["generated_tokens"] or 0) - d["reused_tokens"])
            rows.append(d)
            if not d["ok"]:
                stopped = ("context limit: " if d["context_limit"] else "failed: ") + (d["err"] or "")[:160]
                break
            print(f"session {s + 1} turn {t}: prompt {d['prompt_tokens']}, reused {d['reused_tokens']}, TTFT {d['ttft_s']} s", flush=True)
            msgs.append({"role": "assistant", "content": r["text"] or "Understood."})
            if t < a.turns:
                for _ in range(a.side):
                    sd = tb.digest(cl.chat(side_prompt(rng), 16))
                    sd["after_turn"] = t
                    side.append(sd)
        sessions.append({"turns": rows, "side": side, "stopped": stopped})
    summary = tb.summarize_b([{"turns": s["turns"], "stopped": s["stopped"]} for s in sessions])
    side_ok = [x for s in sessions for x in s["side"] if x.get("ok")]
    summary["side_requests"] = {"n": len(side_ok), "median_ttft_s": tb.rnd(med(x["ttft_s"] for x in side_ok))}
    return {"kind": "session", "turns": a.turns, "side_per_gap": a.side, "sessions": sessions, "summary": summary}


def run_depth(a) -> dict:
    cl = tb.Client(a.host, a.port, a.api_key, think=False, timeout=a.timeout, model=a.model)
    rng = random.Random(a.seed or None)
    out = {"kind": "depth", "max_tokens": a.max_tokens, "sizes": []}
    ratio = tb.DEFAULT_TOKENS_PER_WORD
    refused = None
    for target in [tb.parse_size(s) for s in a.sizes.split(",") if s.strip()]:
        entry = {"label": tb.size_label(target), "target_tokens": target, "runs": [], "status": "measured"}
        out["sizes"].append(entry)
        if refused is not None:
            entry["status"] = f"skipped: {tb.size_label(refused)} was already refused for the context"
            continue
        words = max(10, round((target - tb.PROMPT_OVERHEAD_TOKENS) / ratio))
        for run in range(a.runs):
            r = cl.chat([{"role": "user", "content": tb.unique_variant(words, rng)}], a.max_tokens)
            d = tb.digest(r)
            t = r.get("timings") or {}
            d.update(run=run + 1, discarded=run == 0, words=words, decode_tps=t.get("predicted_per_second"),
                     accepted=t.get("draft_n_accepted"), drafted=t.get("draft_n"), decode_tokens=t.get("predicted_n"))
            entry["runs"].append(d)
            if not d["ok"]:
                entry["status"] = ("skipped: context limit: " if d["context_limit"] else "failed: ") + (d["err"] or "")[:160]
                if d["context_limit"]:
                    refused = target
                break
            print(f"depth {entry['label']} run {run + 1}{' (discarded)' if run == 0 else ''}: {d['prompt_tokens']} tokens, TTFT "
                  f"{d['ttft_s']} s, decode {d['decode_tps']}", flush=True)
            if d["prompt_tokens"] and words:
                ratio = d["prompt_tokens"] / words
                if run == 0 and abs(d["prompt_tokens"] - target) > 0.03 * target:
                    words = max(10, round(words * target / d["prompt_tokens"]))
        kept = [r for r in entry["runs"] if r["ok"] and not r["discarded"]]
        acc, dr = sum(r["accepted"] or 0 for r in kept), sum(r["drafted"] or 0 for r in kept)
        entry["summary"] = {"n": len(kept), "prompt_tokens": med(r["prompt_tokens"] for r in kept),
                            "ttft_s": tb.rnd(med(r["ttft_s"] for r in kept)), "prompt_tps": tb.rnd(med(r["engine_prompt_tps"] for r in kept), 1),
                            "decode_tps": tb.rnd(med(r["decode_tps"] for r in kept), 1),
                            "accept_pct": tb.rnd(100.0 * acc / dr, 1) if dr else None}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("decode", "session", "depth"):
        p = sub.add_parser(name)
        p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--port", type=int, default=8080)
        p.add_argument("--out", required=True)
        p.add_argument("--timeout", type=float, default=3600)
        p.add_argument("--api-key", default=os.environ.get("TTFT_BENCH_API_KEY", ""))
        p.add_argument("--model", default="default")
        p.add_argument("--seed", type=int, default=0)
        if name == "decode":
            p.add_argument("--rounds", type=int, default=1, help="rounds over the five prompts per pass (default 1)")
            p.add_argument("--max-tokens", type=int, default=512)
        if name == "session":
            p.add_argument("--turns", type=int, default=10)
            p.add_argument("--sessions", type=int, default=3)
            p.add_argument("--side", type=int, default=1, help="side requests between two turns (default 1)")
            p.add_argument("--words", type=int, default=1500)
            p.add_argument("--max-tokens", type=int, default=64)
        if name == "depth":
            p.add_argument("--sizes", default="1k,32k,64k,120k")
            p.add_argument("--runs", type=int, default=3, help="requests per size, run 1 discarded (default 3)")
            p.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args(argv)
    res = {"decode": run_decode, "session": run_session, "depth": run_depth}[a.cmd](a)
    res["port"] = a.port
    res["time"] = time.time()
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"written: {a.out}", flush=True)
    bad = [r for r in res.get("rows", []) if r.get("error")]
    return 2 if bad and len(bad) == len(res["rows"]) else 0


if __name__ == "__main__":
    sys.exit(main())
