#!/usr/bin/env python3
"""ttft_bench: time to first token (TTFT) at long context and over an agent session, for any OpenAI-compatible server.

    python3 tools/pancake/ttft_bench.py --port 8080 --label pancake --out ttft-pancake.json
    python3 tools/pancake/ttft_bench.py --compare ttft-pancake.json ttft-avalon.json

Why: an agent that makes 10 sequential calls pays every second of TTFT ten times, so TTFT can matter more than a few
percent of decode speed.  This harness measures it two ways and works against pancake's Strata server, Avalon's
llama-server and Strata-rdna3, because it only uses POST /v1/chat/completions with stream=true.  The engine is never
started or stopped here: point it at one that is already running.

Scenario A, single long prompts.  Sizes 8K, 32K, 64K, 100K tokens by default (K = 1024; --sizes).  Each prompt is
tools/ab_engine.py long_prompt(words) with a unique random first line and a random rotation of its word list, so no
prefix cache and no content cache can help.  Words per token are estimated, then calibrated from the first run of each
size (which is discarded anyway), and the ACTUAL prompt tokens the server reports are what the result lists.  Four runs
per size, run 1 discarded, medians over the rest.  max_tokens is 16.

Scenario B, agent session.  The system prompt and the four tools of tools/agent_turns_bench.py, then N turns (default
10).  Each turn adds about 1,500 words of a project file and resends the whole conversation so far, including the
model's own earlier replies, as agent clients do (the same shape as the agent test of tools/strata_bench.py).  Per turn:
TTFT, prompt tokens, tokens reused from the server's cache and tokens read again.

TTFT is measured on the client: the time from just before the request is sent to the first streamed delta with
content, reasoning, or a tool call (empty keep-alive and role-only deltas are ignored).  The engine's own timings
(prompt_ms, prompt_per_second) are recorded next to it when the server reports them.  Cache reuse comes from
usage.prompt_tokens_details.cached_tokens, else from timings.cache_n; a server that reports neither is shown as "not
reported", never guessed.  --think off (default) sends chat_template_kwargs {"enable_thinking": false}.

Stdlib only.  Needs tools/ab_engine.py and tools/agent_turns_bench.py (both stdlib) beside this folder.
"""
from __future__ import annotations

import argparse
import datetime
import http.client
import json
import os
import random
import re
import statistics
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from ab_engine import long_prompt  # noqa: E402  the word list the other benches use
from agent_turns_bench import SYSTEM, TOOLS  # noqa: E402  the same agent system prompt and tool list

SCHEMA = 1
DEFAULT_SIZES = "8k,32k,64k,100k"
DEFAULT_TOKENS_PER_WORD = 1.06   # long_prompt's vocabulary: 3000 words = 3.2K tokens, 24000 = 25.3K (Qwen tokenizer)
PROMPT_OVERHEAD_TOKENS = 40      # template + the fixed question around the word list
LOAD_WARN = 2.0
CONTEXT_RE = re.compile(r"context|exceed|too long|too many tokens|n_ctx|maximum length|input length|max_position",
                        re.IGNORECASE)
DOC_CANDIDATES = ("README.md", "docs/HOW_IT_WORKS.md", "docs/DETAILS.md", "docs/INSTALL.md", "docs/MODELS.md",
                  "docs/COMMUNITY_BENCHMARKS.md", "docs/AI_REFERENCE.md", "docs/TROUBLESHOOTING.md",
                  "docs/TERMINAL.md", "docs/AMD_HIP.md", "docs/QUANTS.md", "docs/BATCHING.md")


# ------------------------------------------------------------------------------------------------ small helpers
def median(values):
    v = [x for x in values if isinstance(x, (int, float))]
    return statistics.median(v) if v else None


def rnd(x, digits=3):
    return None if x is None else round(x, digits)


def parse_size(text: str) -> int:
    """'8k' -> 8192, '100K' -> 102400, '5000' -> 5000 (K = 1024)."""
    t = text.strip().lower()
    if not t:
        raise ValueError("empty size")
    if t.endswith("k"):
        return int(float(t[:-1]) * 1024)
    return int(t)


def size_label(tokens: int) -> str:
    return f"{tokens // 1024}K" if tokens >= 1024 and tokens % 1024 == 0 else str(tokens)


def unique_variant(words: int, rng: random.Random) -> str:
    """long_prompt(words) with a unique first line and its word list rotated by a random offset: the first token
    already differs from every earlier run and so does every later chain of tokens."""
    text = long_prompt(words)
    head, body, tail = text.split("\n", 2)
    items = body.split()
    k = rng.randrange(max(1, len(items)))
    body = " ".join(items[k:] + items[:k])
    return f"Run id {uuid.UUID(int=rng.getrandbits(128)).hex}\n{head}\n{body}\n{tail}"


def agent_chunks(n: int, words: int, root: Path = ROOT) -> list[tuple[str, str]]:
    """n distinct pieces of text of about `words` words: consecutive chunks of this checkout's own documents when they
    exist (nothing personal), else generated filler of the same size."""
    out: list[tuple[str, str]] = []
    for rel in DOC_CANDIDATES:
        p = root / rel
        if not p.is_file():
            continue
        items = p.read_text(encoding="utf-8", errors="replace").split()
        for i in range(0, len(items), words):
            piece = items[i:i + words]
            if len(piece) >= words // 2:
                out.append((f"{rel}#{i // words + 1}", " ".join(piece)))
            if len(out) >= n:
                return out
    i = 0
    while len(out) < n:
        out.append((f"generated/part{i}.txt", unique_variant(words, random.Random(1000 + i))))
        i += 1
    return out[:n]


# ------------------------------------------------------------------------------------------------------ client
class Client:
    def __init__(self, host: str, port: int, key: str = "", think: bool = False, timeout: float = 3600,
                 model: str = "default", path: str = "/v1/chat/completions"):
        self.host, self.port, self.key, self.think = host, port, key, think
        self.timeout, self.model, self.path = timeout, model, path

    def get_json(self, path: str, timeout: float = 10):
        try:
            c = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
            c.request("GET", path, headers={"Authorization": f"Bearer {self.key}"} if self.key else {})
            r = c.getresponse()
            body = r.read()
            c.close()
            return json.loads(body) if r.status == 200 else None
        except Exception:                                                 # noqa: BLE001
            return None

    def chat(self, messages: list, max_tokens: int, tools=None) -> dict:
        """One streamed request -> {ttft, first_event, total, text, usage, timings, finish, err, status}."""
        body = {"model": self.model, "stream": True, "stream_options": {"include_usage": True},
                "max_tokens": max_tokens, "temperature": 0, "messages": messages}
        if tools:
            body["tools"] = tools
        if not self.think:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        res = {"ttft": None, "first_event": None, "total": None, "text": "", "usage": None, "timings": None,
               "finish": None, "err": None, "status": None}
        text = []
        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        t0 = time.perf_counter()
        try:
            conn.request("POST", self.path, body=json.dumps(body).encode(), headers=headers)
            r = conn.getresponse()
            res["status"] = r.status
            if r.status != 200:
                res["err"] = r.read(4000).decode("utf-8", "replace")
            else:
                while True:
                    raw = r.readline()
                    if not raw:
                        break
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    d = line[5:].strip()
                    if d == "[DONE]":
                        break
                    try:
                        j = json.loads(d)
                    except ValueError:
                        continue
                    if res["first_event"] is None:
                        res["first_event"] = time.perf_counter() - t0
                    if j.get("error"):
                        res["err"] = json.dumps(j["error"])[:400]
                    for ch in j.get("choices") or []:
                        dl = ch.get("delta") or {}
                        piece = (dl.get("content") or "") + (dl.get("reasoning_content") or "") + (dl.get("reasoning") or "")
                        if (piece or dl.get("tool_calls")) and res["ttft"] is None:
                            res["ttft"] = time.perf_counter() - t0
                        text.append(dl.get("content") or "")
                        if ch.get("finish_reason"):
                            res["finish"] = ch["finish_reason"]
                    if j.get("usage"):
                        res["usage"] = j["usage"]
                    if j.get("timings"):
                        res["timings"] = j["timings"]
        except Exception as e:                                           # noqa: BLE001
            res["err"] = res["err"] or repr(e)[:300]
        finally:
            res["total"] = time.perf_counter() - t0
            conn.close()
        res["text"] = "".join(text)
        return res


def short_err(text: str) -> str:
    """The message of an OpenAI-style error body when it parses, else the first 300 characters."""
    try:
        j = json.loads(text)
        e = j.get("error", j) if isinstance(j, dict) else j
        if isinstance(e, dict) and e.get("message"):
            return str(e["message"])[:300]
    except ValueError:
        pass
    return text[:300]


def digest(r: dict) -> dict:
    """The numbers of one request.  Reuse: usage.prompt_tokens_details.cached_tokens, else timings.cache_n, else None
    (the server does not report it)."""
    u, t = r.get("usage") or {}, r.get("timings") or {}
    prompt = u.get("prompt_tokens")
    if prompt is None and t.get("prompt_n") is not None and t.get("cache_n") is not None:
        prompt = t["prompt_n"] + t["cache_n"]
    reused, src = (u.get("prompt_tokens_details") or {}).get("cached_tokens"), "usage.prompt_tokens_details.cached_tokens"
    if reused is None:
        reused, src = t.get("cache_n"), "timings.cache_n"
    if reused is None:
        src = None
    read = t.get("prompt_n")
    if read is None and prompt is not None and reused is not None:
        read = prompt - reused
    ok = r.get("err") is None and r.get("ttft") is not None
    ctx_hit = (not ok) and bool(CONTEXT_RE.search(r.get("err") or ""))
    return {"ok": ok, "context_limit": ctx_hit, "err": None if ok else short_err(r.get("err") or "no token streamed"),
            "status": r.get("status"), "ttft_s": rnd(r.get("ttft")), "first_event_s": rnd(r.get("first_event")),
            "total_s": rnd(r.get("total")), "prompt_tokens": prompt, "reused_tokens": reused, "reused_source": src,
            "read_tokens": read, "generated_tokens": u.get("completion_tokens") or t.get("predicted_n"),
            "engine_prompt_ms": t.get("prompt_ms"), "engine_prompt_tps": t.get("prompt_per_second"),
            "finish": r.get("finish")}


# ------------------------------------------------------------------------------------------------ scenario A
def run_a(cl: Client, sizes: list[int], runs: int, max_tokens: int, calibrate: bool, tpw: float, seed: int,
          log=print) -> dict:
    rng = random.Random(seed)
    out = {"sizes": [], "tokens_per_word_start": tpw, "calibrated": calibrate, "k_is": 1024,
           "note": "run 1 of each size is discarded; TTFT is the client's, stream=true"}
    ratio = tpw
    refused_at = None
    for target in sizes:
        label = size_label(target)
        entry = {"label": label, "target_tokens": target, "runs": [], "status": "measured"}
        out["sizes"].append(entry)
        if refused_at is not None:
            entry["status"] = f"skipped: {size_label(refused_at)} was already refused for the context"
            log(f"A {label:>5}: {entry['status']}")
            continue
        words = max(10, round((target - PROMPT_OVERHEAD_TOKENS) / ratio))
        for run in range(runs):
            prompt = unique_variant(words, rng)
            d = digest(cl.chat([{"role": "user", "content": prompt}], max_tokens))
            d.update(run=run + 1, discarded=run == 0, words=words)
            entry["runs"].append(d)
            if d["context_limit"] and not d["ok"]:
                entry["status"] = "skipped: context limit: " + (d["err"] or "")[:160]
                refused_at = target
                log(f"A {label:>5}: {entry['status']}")
                break
            if not d["ok"]:
                entry["status"] = f"failed at run {run + 1}: {d['err']}"
                log(f"A {label:>5}: {entry['status']}")
                break
            log(f"A {label:>5} run {run + 1}{' (discarded)' if run == 0 else ''}: {d['prompt_tokens']} tokens, "
                f"TTFT {d['ttft_s']} s, engine prompt {d['engine_prompt_ms']} ms")
            if d["prompt_tokens"] and words:
                ratio = d["prompt_tokens"] / words
                if run == 0 and calibrate and abs(d["prompt_tokens"] - target) > 0.03 * target:
                    words = max(10, round(words * target / d["prompt_tokens"]))
        kept = [r for r in entry["runs"] if r["ok"] and not r["discarded"]]
        entry["summary"] = {
            "n": len(kept),
            "ttft_s": {"median": rnd(median(r["ttft_s"] for r in kept)), "min": min((r["ttft_s"] for r in kept), default=None),
                       "max": max((r["ttft_s"] for r in kept), default=None)} if kept else None,
            "prompt_tokens": median(r["prompt_tokens"] for r in kept),
            "reused_tokens": median(r["reused_tokens"] for r in kept),
            "engine_prompt_ms": rnd(median(r["engine_prompt_ms"] for r in kept), 1),
            "engine_prompt_tps": rnd(median(r["engine_prompt_tps"] for r in kept), 1),
            "ttft_minus_engine_prompt_s": rnd(median(
                r["ttft_s"] - r["engine_prompt_ms"] / 1000 for r in kept if r["engine_prompt_ms"] is not None)),
            "total_s": rnd(median(r["total_s"] for r in kept)),
        }
        if not kept and entry["status"] == "measured":
            entry["status"] = "no usable run (only run 1, which is discarded)" if entry["runs"] else "no run"
    return out


# ------------------------------------------------------------------------------------------------ scenario B
def agent_session(cl: Client, turns: int, words: int, max_tokens: int, rng_nonce: str, chunks, log=print) -> dict:
    msgs = [{"role": "system", "content": f"(session {rng_nonce})\n{SYSTEM}"},
            {"role": "user", "content": "I will show you files from a project one at a time. After each file, answer in "
                                        "one or two sentences what it is for. Do not call tools."}]
    rows, stopped = [], None
    for t in range(1, turns + 1):
        name, body = chunks[t - 1]
        msgs.append({"role": "user", "content": f"File {name}:\n{body}\n\nWhat is this file for? Two sentences."})
        r = cl.chat(msgs, max_tokens, tools=TOOLS)
        d = digest(r)
        d["turn"] = t
        if rows and rows[-1]["prompt_tokens"] is not None and d["reused_tokens"] is not None:
            # earlier conversation read again: what the previous turn held (prompt + its reply) minus what this turn reused
            d["reread_tokens"] = max(0, rows[-1]["prompt_tokens"] + (rows[-1]["generated_tokens"] or 0) - d["reused_tokens"])
        rows.append(d)
        if not d["ok"]:
            stopped = ("context limit: " if d["context_limit"] else "failed: ") + (d["err"] or "")[:160]
            log(f"B turn {t}: {stopped}")
            break
        log(f"B turn {t}: prompt {d['prompt_tokens']}, reused {d['reused_tokens']}, read {d['read_tokens']}, TTFT {d['ttft_s']} s")
        msgs.append({"role": "assistant", "content": r["text"] or "Understood."})
    return {"turns": rows, "stopped": stopped}


def share(rows: list[dict]) -> tuple[float | None, int]:
    """Percentage of prompt tokens that were reused, over the rows that report both numbers; and how many rows."""
    pairs = [(r["reused_tokens"], r["prompt_tokens"]) for r in rows if r.get("ok") and r.get("reused_tokens") is not None
             and r.get("prompt_tokens")]
    if not pairs:
        return None, 0
    return 100.0 * sum(a for a, _ in pairs) / sum(b for _, b in pairs), len(pairs)


def summarize_b(sessions: list[dict]) -> dict:
    n_turns = max((len(s["turns"]) for s in sessions), default=0)
    per_turn = []
    for i in range(n_turns):
        rows = [s["turns"][i] for s in sessions if len(s["turns"]) > i and s["turns"][i]["ok"]]
        per_turn.append({"turn": i + 1, "n": len(rows), "ttft_s": rnd(median(r["ttft_s"] for r in rows)),
                         "prompt_tokens": median(r["prompt_tokens"] for r in rows),
                         "reused_tokens": median(r["reused_tokens"] for r in rows),
                         "read_tokens": median(r["read_tokens"] for r in rows),
                         "reread_tokens": median(r.get("reread_tokens") for r in rows),
                         "engine_prompt_ms": rnd(median(r["engine_prompt_ms"] for r in rows), 1)})
    complete = [s for s in sessions if s["turns"] and all(t["ok"] for t in s["turns"])]
    totals = [sum(t["ttft_s"] for t in s["turns"]) for s in complete]
    all_rows = [t for s in sessions for t in s["turns"]]
    follow = [t for s in sessions for t in s["turns"] if t["turn"] > 1]
    reported = [r for r in all_rows if r.get("reused_source")]
    sources = sorted({r["reused_source"] for r in reported})
    return {"per_turn": per_turn, "complete_sessions": len(complete), "sessions": len(sessions),
            "total_ttft_s": rnd(median(totals)), "total_ttft_s_all": [rnd(x) for x in totals],
            "reuse_pct_all_turns": rnd(share(all_rows)[0], 1), "reuse_pct_follow_up_turns": rnd(share(follow)[0], 1),
            "reuse_reported": bool(reported), "reuse_source": sources,
            "median_follow_up_ttft_s": rnd(median(t["ttft_s"] for t in follow if t.get("ok")))}


def run_b(cl: Client, turns: int, words: int, max_tokens: int, sessions: int, log=print) -> dict:
    chunks = agent_chunks(turns, words)
    ss = []
    for k in range(sessions):
        ss.append(agent_session(cl, turns, words, max_tokens, uuid.uuid4().hex[:12], chunks, log))
    return {"turns": turns, "words_per_turn": words, "max_tokens": max_tokens, "chunks": [c[0] for c in chunks],
            "sessions": ss, "summary": summarize_b(ss)}


# ------------------------------------------------------------------------------------------------ text summary
def fmt(x, digits=2, unit=""):
    return "-" if x is None else f"{x:.{digits}f}{unit}"


def summary_text(res: dict) -> str:
    L = [f"TTFT bench: {res['label']}  ({res['base']}, model {res.get('model_reported') or res.get('model')}, think "
         f"{'on' if res['args']['think'] == 'on' else 'off'})",
         f"load average 1 min: start {fmt(res['load']['start'], 2)}, end {fmt(res['load']['end'], 2)}"
         + ("  WARNING: above 2.0, the numbers are not clean" if res["load"]["warning"] else "")]
    a = res.get("scenario_a")
    if a:
        L += ["", "A. single long prompts (run 1 discarded, median of the rest; TTFT = client, stream=true)",
              f"{'size':>6} {'prompt tok':>10} {'TTFT s':>8} {'min-max s':>13} {'engine ms':>10} {'engine tok/s':>12} {'reused':>7}  status"]
        for e in a["sizes"]:
            s = e.get("summary")
            if s and s["ttft_s"]:
                t = s["ttft_s"]
                L.append(f"{e['label']:>6} {fmt(s['prompt_tokens'], 0):>10} {t['median']:>8.2f} "
                         f"{t['min']:>5.2f}-{t['max']:<7.2f} {fmt(s['engine_prompt_ms'], 0):>10} "
                         f"{fmt(s['engine_prompt_tps'], 0):>12} {fmt(s['reused_tokens'], 0):>7}  {e['status']}")
            else:
                L.append(f"{e['label']:>6} {'-':>10} {'-':>8} {'-':>13} {'-':>10} {'-':>12} {'-':>7}  {e['status']}")
    b = res.get("scenario_b")
    if b:
        s = b["summary"]
        L += ["", f"B. agent session: {b['turns']} turns x about {b['words_per_turn']} words, {s['sessions']} session(s), "
                  "whole conversation resent each turn",
              f"{'turn':>4} {'prompt':>8} {'reused':>8} {'read':>8} {'re-read':>8} {'reuse %':>8} {'TTFT s':>8} {'engine ms':>10}"]
        for p in s["per_turn"]:
            pct = None if p["reused_tokens"] is None or not p["prompt_tokens"] else 100 * p["reused_tokens"] / p["prompt_tokens"]
            L.append(f"{p['turn']:>4} {fmt(p['prompt_tokens'], 0):>8} {fmt(p['reused_tokens'], 0):>8} "
                     f"{fmt(p['read_tokens'], 0):>8} {fmt(p['reread_tokens'], 0):>8} {fmt(pct, 1):>8} "
                     f"{fmt(p['ttft_s']):>8} {fmt(p['engine_prompt_ms'], 0):>10}")
        L.append(f"total TTFT over the session (median of {s['complete_sessions']} complete): {fmt(s['total_ttft_s'])} s; "
                 f"median follow-up TTFT {fmt(s['median_follow_up_ttft_s'])} s")
        if s["reuse_reported"]:
            L.append(f"prompt tokens reused: {fmt(s['reuse_pct_all_turns'], 1)} % of all turns, "
                     f"{fmt(s['reuse_pct_follow_up_turns'], 1)} % of turns 2+ (from {', '.join(s['reuse_source'])}); "
                     "re-read = earlier conversation the server read again")
        else:
            L.append("cache reuse: not reported by this server (neither usage.prompt_tokens_details.cached_tokens "
                     "nor timings.cache_n); reused/read/re-read columns are empty")
        stops = [x["stopped"] for x in b["sessions"] if x["stopped"]]
        if stops:
            L.append("stopped early: " + "; ".join(stops[:2]))
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------------ compare
def pct_change(a, b):
    if a is None or b is None or a == 0:
        return "-"
    return f"{100 * (b - a) / a:+.1f}%"


def compare_text(a: dict, b: dict) -> str:
    la, lb = a["label"], b["label"]
    L = [f"TTFT compare: A = {la}   B = {lb}   (negative change = B is faster)"]
    for r, tag in ((a, "A"), (b, "B")):
        if r["load"]["warning"]:
            L.append(f"WARNING: run {tag} ({r['label']}) had a 1-minute load average above 2.0")
    if a.get("scenario_a") and b.get("scenario_a"):
        L += ["", "A. single long prompts: median TTFT in seconds",
              f"{'size':>6} {'A tok':>8} {'B tok':>8} {'A TTFT':>8} {'B TTFT':>8} {'change':>8} {'A eng tok/s':>11} {'B eng tok/s':>11}"]
        bm = {e["label"]: e for e in b["scenario_a"]["sizes"]}
        for e in a["scenario_a"]["sizes"]:
            o = bm.get(e["label"])
            sa_, sb_ = e.get("summary") or {}, (o or {}).get("summary") or {}
            ta, tb = (sa_.get("ttft_s") or {}).get("median"), (sb_.get("ttft_s") or {}).get("median")
            L.append(f"{e['label']:>6} {fmt(sa_.get('prompt_tokens'), 0):>8} {fmt(sb_.get('prompt_tokens'), 0):>8} "
                     f"{fmt(ta):>8} {fmt(tb):>8} {pct_change(ta, tb):>8} {fmt(sa_.get('engine_prompt_tps'), 0):>11} "
                     f"{fmt(sb_.get('engine_prompt_tps'), 0):>11}"
                     + ("" if o else "  (size missing in B)"))
            pa, pb = sa_.get("prompt_tokens"), sb_.get("prompt_tokens")
            if pa and pb and abs(pa - pb) > 0.05 * max(pa, pb):
                L.append(f"       note: actual prompt tokens differ by more than 5% (tokenizers differ); compare tok/s, not only seconds")
            for r, tag in ((e, "A"), (o, "B")):
                if r and r["status"] != "measured":
                    L.append(f"       {tag}: {r['status']}")
    if a.get("scenario_b") and b.get("scenario_b"):
        pa, pb = a["scenario_b"]["summary"], b["scenario_b"]["summary"]
        L += ["", "B. agent session: median TTFT per turn in seconds",
              f"{'turn':>4} {'A prompt':>9} {'B prompt':>9} {'A TTFT':>8} {'B TTFT':>8} {'change':>8} {'A reuse %':>10} {'B reuse %':>10}"]

        def pr(p):
            return None if p is None or p["reused_tokens"] is None or not p["prompt_tokens"] else 100 * p["reused_tokens"] / p["prompt_tokens"]
        for i in range(max(len(pa["per_turn"]), len(pb["per_turn"]))):
            x = pa["per_turn"][i] if i < len(pa["per_turn"]) else None
            y = pb["per_turn"][i] if i < len(pb["per_turn"]) else None
            L.append(f"{i + 1:>4} {fmt(x and x['prompt_tokens'], 0):>9} {fmt(y and y['prompt_tokens'], 0):>9} "
                     f"{fmt(x and x['ttft_s']):>8} {fmt(y and y['ttft_s']):>8} {pct_change(x and x['ttft_s'], y and y['ttft_s']):>8} "
                     f"{fmt(pr(x), 1):>10} {fmt(pr(y), 1):>10}")
        L.append(f"total TTFT over the session: A {fmt(pa['total_ttft_s'])} s, B {fmt(pb['total_ttft_s'])} s "
                 f"({pct_change(pa['total_ttft_s'], pb['total_ttft_s'])})")
        L.append(f"prompt tokens reused (turns 2+): A {fmt(pa['reuse_pct_follow_up_turns'], 1)} %"
                 f"{'' if pa['reuse_reported'] else ' (not reported)'}, B {fmt(pb['reuse_pct_follow_up_turns'], 1)} %"
                 f"{'' if pb['reuse_reported'] else ' (not reported)'}")
    if len(L) == 1:
        L.append("nothing to compare: the two files share no scenario")
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------------ main
def load_avg() -> float | None:
    try:
        return os.getloadavg()[0]
    except (OSError, AttributeError):
        return None


def server_info(cl: Client) -> dict:
    """Best effort: the model and window the server reports (llama-server's /props, Strata's /v1/status)."""
    info: dict = {}
    props = cl.get_json("/props")
    if isinstance(props, dict):
        n = (props.get("default_generation_settings") or {}).get("n_ctx")
        if n:
            info["n_ctx_reported"], info["n_ctx_source"] = n, "/props"
    if "n_ctx_reported" not in info:
        st = cl.get_json("/v1/status")
        if isinstance(st, dict) and isinstance(st.get("context"), dict) and st["context"].get("native"):
            info["n_ctx_reported"], info["n_ctx_source"] = st["context"]["native"], "/v1/status"
            info["model_reported"] = st.get("model")
    models = cl.get_json("/v1/models")
    if isinstance(models, dict) and models.get("data"):
        info.setdefault("model_reported", models["data"][0].get("id"))
    return info


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"), help="compare two result files and exit")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--label", default="engine", help="name of this run in the output (the engine and build)")
    ap.add_argument("--out", default="", help="result JSON file (default ttft_<label>.json)")
    ap.add_argument("--api-key", default=os.environ.get("TTFT_BENCH_API_KEY", ""), help="bearer key (never written to the JSON)")
    ap.add_argument("--model", default="default", help="model name sent in the request")
    ap.add_argument("--scenario", choices=("a", "b", "both"), default="both")
    ap.add_argument("--sizes", default=DEFAULT_SIZES, help="scenario A prompt sizes in tokens, K = 1024 (default %(default)s)")
    ap.add_argument("--runs", type=int, default=4, help="scenario A runs per size; run 1 is discarded (default 4)")
    ap.add_argument("--max-tokens", type=int, default=16, help="scenario A max_tokens (default 16)")
    ap.add_argument("--tokens-per-word", type=float, default=DEFAULT_TOKENS_PER_WORD)
    ap.add_argument("--no-calibrate", action="store_true", help="keep the words per size from --tokens-per-word")
    ap.add_argument("--turns", type=int, default=10, help="scenario B turns (default 10)")
    ap.add_argument("--agent-words", type=int, default=1500, help="words added per scenario B turn (default 1500)")
    ap.add_argument("--agent-max-tokens", type=int, default=64)
    ap.add_argument("--sessions", type=int, default=1, help="scenario B sessions (medians per turn when more than 1)")
    ap.add_argument("--think", choices=("off", "on"), default="off", help="off sends chat_template_kwargs enable_thinking=false")
    ap.add_argument("--timeout", type=float, default=3600, help="seconds per request")
    ap.add_argument("--seed", type=int, default=0, help="seed of the prompt variation (0: random)")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.compare:
        ra, rb = (json.loads(Path(p).read_text(encoding="utf-8")) for p in a.compare)
        print(compare_text(ra, rb))
        return 0
    sizes = [parse_size(s) for s in a.sizes.split(",") if s.strip()]
    if a.runs < 2:
        print("--runs must be at least 2: run 1 is discarded", file=sys.stderr)
        return 2
    cl = Client(a.host, a.port, a.api_key, think=a.think == "on", timeout=a.timeout, model=a.model)
    try:
        import socket
        socket.create_connection((a.host, a.port), timeout=5).close()
    except OSError as e:
        print(f"cannot connect to {a.host}:{a.port}: {e}", file=sys.stderr)
        return 2
    load0 = load_avg()
    if load0 is not None and load0 > LOAD_WARN:
        print(f"WARNING: 1-minute load average is {load0:.2f} (> {LOAD_WARN}); other work is running, so the "
              "timings are not clean", file=sys.stderr)
    info = server_info(cl)
    res = {"schema": SCHEMA, "tool": "ttft_bench", "label": a.label, "base": f"http://{a.host}:{a.port}",
           "started": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "args": {k: v for k, v in vars(a).items() if k not in ("api_key", "compare")}, "server": info,
           "model_reported": info.get("model_reported"), "model": a.model, "load": {"start": rnd(load0, 2), "end": None, "warning": False}}
    seed = a.seed or random.SystemRandom().randrange(1, 2**31)
    res["args"]["seed"] = seed
    if a.scenario in ("a", "both"):
        res["scenario_a"] = run_a(cl, sizes, a.runs, a.max_tokens, not a.no_calibrate, a.tokens_per_word, seed)
    if a.scenario in ("b", "both"):
        res["scenario_b"] = run_b(cl, a.turns, a.agent_words, a.agent_max_tokens, a.sessions)
    load1 = load_avg()
    res["load"]["end"] = rnd(load1, 2)
    res["load"]["warning"] = any(x is not None and x > LOAD_WARN for x in (load0, load1))
    out = Path(a.out or f"ttft_{re.sub(r'[^A-Za-z0-9_.-]+', '_', a.label)}.json")
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print()
    print(summary_text(res))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
