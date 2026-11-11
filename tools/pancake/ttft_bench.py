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

Scenario C (--scenario c), session with side requests: the scenario B session, with short UNRELATED side requests
(own system prompt, 1-2 short messages, like a title request) sent between turns at --side-after 3,6 and optionally one
sub-agent switch (--switch-after N: a second long conversation with another system prompt and tools, then back).  Each
main turn records TTFT, reused and re-read tokens and whether it followed a side request or the switch.  Scenario D
(tight budget) is scenario C against a server started with a small conversation cache; --note-cache-budget-mib N records
that budget in the JSON and --compare prints it.  Scenario E (--scenario e), warm-delta sweep: one conversation of about
30K tokens (--delta-base) built once, then 50, 100, 300, 1000 and 3000 new tokens appended, 3 repeats each, unique
content; TTFT per delta size plus a least-squares floor and slope.  --prefill-timing-log PATH attaches, to every
request, the engine-log lines added while it ran (STRATA_PREFILL_TIMING=1 breakdown, conversation-cache lines, RESUME).

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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ttft_audit import PARK_RE, SKIP_RE, FILE_RE  # noqa: E402  the conversation-cache log lines the audit already parses

SCHEMA = 1
DEFAULT_SIZES = "8k,32k,64k,100k"
DEFAULT_TOKENS_PER_WORD = 1.06   # long_prompt's vocabulary: 3000 words = 3.2K tokens, 24000 = 25.3K (Qwen tokenizer)
PROMPT_OVERHEAD_TOKENS = 40      # template + the fixed question around the word list
LOAD_WARN = 2.0
CONTEXT_RE = re.compile(r"context|exceed|too long|too many tokens|n_ctx|maximum length|input length|max_position",
                        re.IGNORECASE)
DEFAULT_DELTAS = "50,100,300,1000,3000"
DEFAULT_DELTA_BASE = "30k"
SIDE_SYSTEM = "You write short titles and one-line summaries for notes. Answer with the title only unless asked for more."
SUB_SYSTEM = ("You are a read-only code-search sub-agent. You are given one question about a repository and the files "
              "found so far; answer with the file names and one line of evidence each. You never edit anything.")
SUB_TOOLS = [
    {"type": "function", "function": {
        "name": "search_symbols", "description": "Find where a function, class or constant is defined.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "list_dir", "description": "List the files of a directory of the repository.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
]
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
        self.tap = None          # a LogTap when --prefill-timing-log is given

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


# ------------------------------------------------------------------------------------------------ engine log tap
# What the engine prints (read from the code, src/prefill/prefill.cpp, src/split/src/session.cpp, src/program/generate.cpp):
#   strata prefill timing: N tokens, GPU timeline X ms, wall Y ms, host staging Z ms: <phase> ms (pct%) ...   (STRATA_PREFILL_TIMING=1)
#   strata prefill timing: host: chunk setup (...) A ms, waiting for each chunk B ms, after each chunk (...) C ms, PLE D ms
#   strata prefill timing (layer-major): N tokens, GPU timeline X ms: <phase> ms (pct%) ...
#   strata ple wait: chunk at ...                      (STRATA_PLE_WAIT_TRACE=1, or any STRATA_PREFILL_TIMING)
#   conversation cache: parked N tokens in X ms, ... / skip parking (...) / restored N tokens in X ms / evicted N tokens
#   strata serve: file tier I/O this request: the OS read X MB from storage ...
#   RESUME N                                           (stdout of the engine process: N prompt tokens are reused)
#   strata prefill: N tokens in M-token chunks over S stages (--prefill-pipe ...)   (the chunk plan; only when the plan is chunked)
INTEREST_RE = re.compile(r"strata prefill timing|strata ple wait|conversation cache|file tier I/O|^RESUME \d+|"
                         r"strata prefill: \d+ tokens in \d+-token chunks")
PLAN_RE = re.compile(r"strata prefill: (\d+) tokens in (\d+)-token chunks over (\d+) stages")
TIMING_RE = re.compile(r"strata prefill timing( \(layer-major\))?: (\d+) tokens, GPU timeline (\d+) ms"
                       r"(?:, wall (\d+) ms, host staging (\d+) ms)?:(.*)$")
PHASE_RE = re.compile(r"([A-Za-z][^()]*?) (\d+) \(([0-9.]+)%\)")
HOST_RE = re.compile(r"strata prefill timing: host: chunk setup \([^)]*\) ([0-9.]+) ms, waiting for each chunk ([0-9.]+) ms, "
                     r"after each chunk \([^)]*\) ([0-9.]+) ms, PLE ([0-9.]+) ms")
RESTORE_RE = re.compile(r"conversation cache: restored (\d+) tokens.*? in ([0-9.]+) ms")
EVICT_RE = re.compile(r"conversation cache: evicted (\d+) tokens")
RESUME_RE = re.compile(r"^RESUME (\d+)")


def parse_engine_lines(lines: list[str]) -> dict:
    """The structured numbers of the engine-log lines of one request (see the format list above); a field stays empty
    when its line is not there, nothing is guessed."""
    out: dict = {"prefill_timing": [], "chunk_plan": [], "parks": [], "skips": [], "restores": [], "evicted": [],
                 "resume": [], "file_tier_mb": [], "ple_wait_lines": 0}
    for line in lines:
        m = PLAN_RE.search(line)
        if m:
            out["chunk_plan"].append({"tokens": int(m.group(1)), "chunk": int(m.group(2)), "stages": int(m.group(3))})
            continue
        m = TIMING_RE.search(line)
        if m:
            out["prefill_timing"].append({
                "layer_major": bool(m.group(1)), "tokens": int(m.group(2)), "gpu_ms": int(m.group(3)),
                "wall_ms": int(m.group(4)) if m.group(4) else None,
                "host_staging_ms": int(m.group(5)) if m.group(5) else None,
                "phases": {n.strip(): {"ms": int(ms), "pct": float(pc)} for n, ms, pc in PHASE_RE.findall(m.group(6))},
                "host": None})
            continue
        m = HOST_RE.search(line)
        if m:
            if out["prefill_timing"]:
                out["prefill_timing"][-1]["host"] = {
                    "chunk_setup_ms": float(m.group(1)), "waiting_for_chunks_ms": float(m.group(2)),
                    "after_each_chunk_ms": float(m.group(3)), "ple_ms": float(m.group(4))}
            continue
        m = PARK_RE.search(line)
        if m:
            out["parks"].append({"tokens": int(m.group(1)), "ms": float(m.group(2))})
            continue
        m = SKIP_RE.search(line)
        if m:
            out["skips"].append(m.group(1))
            continue
        m = RESTORE_RE.search(line)
        if m:
            out["restores"].append({"tokens": int(m.group(1)), "ms": float(m.group(2))})
            continue
        m = EVICT_RE.search(line)
        if m:
            out["evicted"].append(int(m.group(1)))
            continue
        m = RESUME_RE.search(line)
        if m:
            out["resume"].append(int(m.group(1)))
            continue
        m = FILE_RE.search(line)
        if m:
            out["file_tier_mb"].append(float(m.group(1)))
            continue
        if "strata ple wait" in line:
            out["ple_wait_lines"] += 1
    return out


class LogTap:
    """Reads the part of an engine log file that was added between two points (a byte offset taken before a request
    and the end of the file after it).  The engine log has no timestamps, so this is by position; a settle pause lets
    the last lines of a request reach the file.  On HIP the conversation-cache park of a conversation is printed while
    the NEXT request is being served, so those lines land on the next request, which is where their time is spent."""

    def __init__(self, path: str, settle_s: float = 0.3, max_lines: int = 120):
        self.path, self.settle_s, self.max_lines = str(path), settle_s, max_lines

    def mark(self) -> int:
        try:
            return os.path.getsize(self.path)
        except OSError:
            return 0

    def since(self, offset: int) -> dict:
        if self.settle_s > 0:
            time.sleep(self.settle_s)
        try:
            with open(self.path, "rb") as f:
                size = f.seek(0, os.SEEK_END)
                truncated = size < offset
                f.seek(0 if truncated else offset)
                raw = f.read().decode("utf-8", "replace")
        except OSError as e:
            return {"error": f"engine log not readable: {e}"}
        lines = [x.rstrip() for x in raw.splitlines()]
        kept = [x for x in lines if INTEREST_RE.search(x)]
        rec = {"bytes": len(raw.encode("utf-8")), "lines_total": len(lines), "lines_matched": len(kept),
               "lines": kept[:self.max_lines], "truncated_file": truncated or None}
        rec.update(parse_engine_lines(kept))
        return rec


def tapped(cl: "Client", messages: list, max_tokens: int, tools=None) -> tuple[dict, dict]:
    """cl.chat + digest, with the engine-log lines of this request attached as d['engine_log'] when cl.tap is set."""
    tap = getattr(cl, "tap", None)
    off = tap.mark() if tap else None
    r = cl.chat(messages, max_tokens, tools=tools)
    d = digest(r)
    if tap is not None:
        d["engine_log"] = tap.since(off)
    return r, d


def random_words(n: int, rng: random.Random) -> str:
    """n random words of the long_prompt vocabulary (about 1.06 tokens each)."""
    vocab = sorted(set(long_prompt(400).split("\n")[1].split()))
    return " ".join(rng.choice(vocab) for _ in range(max(1, n)))


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
            _, d = tapped(cl, [{"role": "user", "content": prompt}], max_tokens)
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
def agent_session(cl: Client, turns: int, words: int, max_tokens: int, rng_nonce: str, chunks, log=print,
                  between=None) -> dict:
    """between(t) -> dict, when given, runs after main turn t (not after the last) and its dict is merged into the NEXT
    main turn's row (scenario C uses it for side requests and the sub-agent switch)."""
    pending: dict = {}
    msgs = [{"role": "system", "content": f"(session {rng_nonce})\n{SYSTEM}"},
            {"role": "user", "content": "I will show you files from a project one at a time. After each file, answer in "
                                        "one or two sentences what it is for. Do not call tools."}]
    rows, stopped = [], None
    for t in range(1, turns + 1):
        name, body = chunks[t - 1]
        msgs.append({"role": "user", "content": f"File {name}:\n{body}\n\nWhat is this file for? Two sentences."})
        r, d = tapped(cl, msgs, max_tokens, tools=TOOLS)
        d["turn"] = t
        d.update(pending)
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
        pending = between(t) if between and t < turns else {}
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


# ------------------------------------------------------------------------------------------------ scenario C
def parse_positions(text: str) -> list[int]:
    """'3,6' -> [3, 6]; '' -> []."""
    out = sorted({int(x) for x in text.split(",") if x.strip()})
    if any(x < 1 for x in out):
        raise ValueError("positions count from 1")
    return out


def side_messages(rng: random.Random, words: int, n_messages: int) -> list:
    """A short UNRELATED request of the kind an agent client sends between turns (a title / a summary): its own system
    prompt, 1 or 2 short user messages, unique text every time."""
    nonce = uuid.UUID(int=rng.getrandbits(128)).hex[:12]
    msgs = [{"role": "system", "content": SIDE_SYSTEM},
            {"role": "user", "content": f"Write a title of at most six words for this note ({nonce}):\n{random_words(words, rng)}"}]
    if n_messages >= 2:
        msgs += [{"role": "assistant", "content": "Understood."},
                 {"role": "user", "content": "Now give a one-sentence summary of the same note."}]
    return msgs


def subagent_run(cl: Client, rng: random.Random, turns: int, chunks, max_tokens: int, log=print) -> list[dict]:
    """A second long conversation (own system prompt and tools, own files) run for `turns` turns, then left."""
    msgs = [{"role": "system", "content": f"(sub-agent {uuid.UUID(int=rng.getrandbits(128)).hex[:12]})\n{SUB_SYSTEM}"},
            {"role": "user", "content": "Question: which files describe how the server answers a request? I will show you "
                                        "candidate files one at a time; after each, say in one sentence if it is relevant."}]
    rows = []
    for k in range(1, turns + 1):
        name, body = chunks[k - 1]
        msgs.append({"role": "user", "content": f"Candidate {name}:\n{body}\n\nRelevant? One sentence."})
        r, d = tapped(cl, msgs, max_tokens, tools=SUB_TOOLS)
        d["sub_turn"] = k
        rows.append(d)
        log(f"C sub-agent turn {k}: prompt {d['prompt_tokens']}, reused {d['reused_tokens']}, TTFT {d['ttft_s']} s"
            + ("" if d["ok"] else f"  FAILED {d['err']}"))
        if not d["ok"]:
            break
        msgs.append({"role": "assistant", "content": r["text"] or "Understood."})
    return rows


def session_side(cl: Client, turns: int, words: int, max_tokens: int, nonce: str, chunks, side_after: list[int],
                 side_messages_n: int, side_words: int, switch_after: int, sub_turns: int, seed: int, log=print) -> dict:
    """Scenario C: the 10-turn agent session of scenario B with, between turns, unrelated side requests (after the
    turns in side_after) and optionally ONE sub-agent switch (after turn switch_after; 0 = none).  Both go out after the
    main turn's reply is in the conversation and before the next main turn; when both sit after the same turn, the side
    request goes first.  The main conversation itself is byte-identical to scenario B's, so the two are comparable.
    Each main row says whether it followed a side request (after_side = how many) or the switch (after_switch)."""
    rng = random.Random(seed)
    extra: list[dict] = []

    def between(t: int) -> dict:
        flags = {"after_side": 0, "after_switch": False}
        if t in side_after:
            r, d = tapped(cl, side_messages(rng, side_words, side_messages_n), 32)
            d.update(kind="side", after_turn=t)
            extra.append(d)
            flags["after_side"] += 1
            log(f"C side request after turn {t}: prompt {d['prompt_tokens']}, reused {d['reused_tokens']}, "
                f"TTFT {d['ttft_s']} s" + ("" if d["ok"] else f"  FAILED {d['err']}"))
        if t == switch_after:
            for d in subagent_run(cl, rng, sub_turns, chunks[turns:turns + sub_turns], 32, log):
                d.update(kind="subagent", after_turn=t)
                extra.append(d)
            flags["after_switch"] = True
        return flags

    s = agent_session(cl, turns, words, max_tokens, nonce, chunks, log, between=between)
    for r in s["turns"]:
        r.setdefault("after_side", 0)
        r.setdefault("after_switch", False)
    s["other_requests"] = extra
    return s


def turn_group(r: dict) -> str:
    return "after_switch" if r.get("after_switch") else "after_side" if r.get("after_side") else "plain"


def summarize_c(sessions: list[dict]) -> dict:
    base = summarize_b(sessions)
    first = sessions[0]["turns"] if sessions else []
    for p in base["per_turn"]:
        row = next((r for r in first if r["turn"] == p["turn"]), None)
        p["after_side"] = (row or {}).get("after_side", 0)
        p["after_switch"] = bool((row or {}).get("after_switch"))
    groups = {}
    for g in ("plain", "after_side", "after_switch"):
        rows = [t for s in sessions for t in s["turns"] if t["turn"] > 1 and t["ok"] and turn_group(t) == g]
        groups[g] = {"n": len(rows), "ttft_s_median": rnd(median(t["ttft_s"] for t in rows)),
                     "reused_tokens_median": median(t["reused_tokens"] for t in rows),
                     "read_tokens_median": median(t["read_tokens"] for t in rows),
                     "reread_tokens_median": median(t.get("reread_tokens") for t in rows)}
    others = [d for s in sessions for d in s.get("other_requests", [])]
    side = [d for d in others if d.get("kind") == "side" and d["ok"]]
    sub = [d for d in others if d.get("kind") == "subagent" and d["ok"]]
    base["groups"] = groups
    base["other_requests"] = {"side_n": len(side), "side_ttft_s_median": rnd(median(d["ttft_s"] for d in side)),
                              "subagent_n": len(sub), "subagent_ttft_s_median": rnd(median(d["ttft_s"] for d in sub)),
                              "failed": sum(1 for d in others if not d["ok"])}
    return base


def run_c(cl: Client, turns: int, words: int, max_tokens: int, sessions: int, side_after: list[int], side_messages_n: int,
          side_words: int, switch_after: int, sub_turns: int, seed: int, log=print) -> dict:
    if any(t >= turns for t in side_after) or switch_after >= turns:
        raise ValueError(f"--side-after / --switch-after must be before the last turn ({turns}): a request after the last "
                         "turn has no main turn to disturb")
    chunks = agent_chunks(turns + (sub_turns if switch_after else 0), words)
    ss = [session_side(cl, turns, words, max_tokens, uuid.uuid4().hex[:12], chunks, side_after, side_messages_n, side_words,
                       switch_after, sub_turns, seed + k, log) for k in range(sessions)]
    return {"turns": turns, "words_per_turn": words, "max_tokens": max_tokens, "side_after": side_after,
            "side_messages": side_messages_n, "side_words": side_words, "switch_after": switch_after or None,
            "subagent_turns": sub_turns if switch_after else 0, "chunks": [c[0] for c in chunks[:turns]],
            "sessions": ss, "summary": summarize_c(ss)}


# ------------------------------------------------------------------------------------------------ scenario E
def linfit(points: list[tuple[float, float]]) -> dict | None:
    """Least squares y = a + b*x over (x, y); None with fewer than 2 distinct x."""
    n = len(points)
    if n < 2 or len({x for x, _ in points}) < 2:
        return None
    mx, my = sum(x for x, _ in points) / n, sum(y for _, y in points) / n
    sxx = sum((x - mx) ** 2 for x, _ in points)
    b = sum((x - mx) * (y - my) for x, y in points) / sxx
    return {"n": n, "slope_ms_per_token": rnd(b * 1000, 4), "intercept_s": rnd(my - b * mx, 3)}


def run_e(cl: Client, base_tokens: int, deltas: list[int], repeats: int, tpw: float, max_tokens: int, seed: int,
          log=print) -> dict:
    """Scenario E, warm-delta sweep (TTFT_LEVERS M5).  One long conversation (about base_tokens) is built once; then, in
    `repeats` rounds, each delta size (ascending) is appended as a new user message with unique content and the reply
    is kept, so every request is the live warm path of an agent turn.  The conversation grows by the deltas and
    replies, so depth drifts upward over rounds (prompt_tokens is recorded per request); a round runs every size once, so
    sizes are compared at nearly the same depth.  new_tokens_est = prompt_tokens - (previous prompt_tokens + previous
    generated tokens): the new message plus a few chat-template tokens, client-side, valid for every server."""
    rng = random.Random(seed)
    nonce = uuid.uuid4().hex[:12]
    words = max(10, round((base_tokens - PROMPT_OVERHEAD_TOKENS) / tpw))
    msgs = [{"role": "system", "content": f"(session {nonce})\n{SYSTEM}"},
            {"role": "user", "content": unique_variant(words, rng)}]
    out: dict = {"base_target_tokens": base_tokens, "deltas": deltas, "repeats": repeats, "max_tokens": max_tokens,
                 "status": "measured", "base": None, "rows": [],
                 "note": "new_tokens_est includes a few chat-template tokens; depth grows by the deltas and replies"}
    r, d = tapped(cl, msgs, max_tokens, tools=TOOLS)
    d["words"] = words
    out["base"] = d
    if not d["ok"]:
        out["status"] = ("skipped: context limit: " if d["context_limit"] else "failed building the base: ") + (d["err"] or "")[:160]
        log(f"E base: {out['status']}")
        out["summary"] = summarize_e(out)
        return out
    log(f"E base: {d['prompt_tokens']} tokens, TTFT {d['ttft_s']} s")
    msgs.append({"role": "assistant", "content": r["text"] or "Understood."})
    ratio = (d["prompt_tokens"] - PROMPT_OVERHEAD_TOKENS) / words if d["prompt_tokens"] else tpw
    prev_prompt, prev_gen = d["prompt_tokens"], d["generated_tokens"] or 0
    stop = False
    for rep in range(1, repeats + 1):
        for size in deltas:
            w = max(1, round(size / ratio))
            msgs.append({"role": "user", "content": f"Further notes {uuid.UUID(int=rng.getrandbits(128)).hex[:12]}: "
                                                    f"{random_words(w, rng)}\nReply with one word."})
            r, d = tapped(cl, msgs, max_tokens, tools=TOOLS)
            d.update(delta_target=size, repeat=rep, words=w)
            if d["prompt_tokens"] is not None and prev_prompt is not None:
                d["new_tokens_est"] = d["prompt_tokens"] - (prev_prompt + prev_gen)
            out["rows"].append(d)
            if not d["ok"]:
                out["status"] = ("skipped: context limit: " if d["context_limit"] else f"failed at delta {size} repeat {rep}: ") \
                    + (d["err"] or "")[:160]
                log(f"E delta {size} repeat {rep}: {out['status']}")
                stop = True
                break
            log(f"E delta {size:>5} repeat {rep}: prompt {d['prompt_tokens']}, new ~{d.get('new_tokens_est')}, "
                f"read {d['read_tokens']}, TTFT {d['ttft_s']} s")
            if w >= 200 and d.get("new_tokens_est"):
                ratio = max(0.5, d["new_tokens_est"] / w)
            msgs.append({"role": "assistant", "content": r["text"] or "Understood."})
            prev_prompt, prev_gen = d["prompt_tokens"], d["generated_tokens"] or 0
        if stop:
            break
    out["summary"] = summarize_e(out)
    return out


def summarize_e(e: dict) -> dict:
    rows = [r for r in e["rows"] if r["ok"]]
    per = []
    for size in e["deltas"]:
        rr = [r for r in rows if r["delta_target"] == size]
        per.append({"delta": size, "n": len(rr), "ttft_s": {"median": rnd(median(r["ttft_s"] for r in rr)),
                                                            "min": min((r["ttft_s"] for r in rr), default=None),
                                                            "max": max((r["ttft_s"] for r in rr), default=None)} if rr else None,
                    "new_tokens_est": median(r.get("new_tokens_est") for r in rr),
                    "read_tokens": median(r["read_tokens"] for r in rr),
                    "reused_tokens": median(r["reused_tokens"] for r in rr),
                    "prompt_tokens": median(r["prompt_tokens"] for r in rr),
                    "engine_prompt_ms": rnd(median(r["engine_prompt_ms"] for r in rr), 1)})
    pts = [(r["new_tokens_est"], r["ttft_s"]) for r in rows if r.get("new_tokens_est") is not None]
    return {"per_delta": per, "fit_all": linfit(pts), "fit_ge_1000": linfit([p for p in pts if p[0] >= 1000]),
            "fit_note": "TTFT s = intercept_s + slope * new tokens, least squares over the kept requests; the staged-path "
                        "floor is the intercept only if the small deltas sit on the line, so read the table too",
            "base_prompt_tokens": (e["base"] or {}).get("prompt_tokens"), "base_ttft_s": (e["base"] or {}).get("ttft_s")}


# ------------------------------------------------------------------------------------------------ text summary
def fmt(x, digits=2, unit=""):
    return "-" if x is None else f"{x:.{digits}f}{unit}"


def budget_text(res: dict) -> str | None:
    b = res.get("cache_budget_mib")
    return None if b is None else f"conversation-cache budget noted by the operator: {b} MiB (the harness did not read it from the server)"


def iter_requests(res: dict):
    """Every request digest of a result file that may carry an engine_log."""
    for e in ((res.get("scenario_a") or {}).get("sizes") or []):
        yield from e.get("runs", [])
    for key in ("scenario_b", "scenario_c"):
        for s in ((res.get(key) or {}).get("sessions") or []):
            yield from s.get("turns", [])
            yield from s.get("other_requests", [])
    e = res.get("scenario_e") or {}
    if e.get("base"):
        yield e["base"]
    yield from e.get("rows", [])


def engine_log_text(res: dict) -> list[str]:
    if not (res.get("args") or {}).get("prefill_timing_log"):
        return []
    reqs = [r for r in iter_requests(res) if r.get("engine_log") is not None]
    L = ["", "engine log lines attached per request (--prefill-timing-log; the prefill breakdown needs STRATA_PREFILL_TIMING=1)"]
    n_timing = sum(len(r["engine_log"].get("prefill_timing") or []) for r in reqs)
    errs = {r["engine_log"]["error"] for r in reqs if r["engine_log"].get("error")}
    if errs:
        L.append("  " + "; ".join(sorted(errs)))
    if not n_timing:
        L.append("  no 'strata prefill timing' line found in the log lines added during the requests: was the engine started "
                 "with STRATA_PREFILL_TIMING=1, and is this the file its stderr goes to?")
    for e in ((res.get("scenario_a") or {}).get("sizes") or []):
        kept = [r for r in e.get("runs", []) if r.get("ok") and not r.get("discarded")
                and (r.get("engine_log") or {}).get("prefill_timing")]
        if not kept:
            continue
        pt = max(kept[-1]["engine_log"]["prefill_timing"], key=lambda x: x["tokens"])
        top = sorted(pt["phases"].items(), key=lambda kv: -kv[1]["ms"])[:4]
        L.append(f"  A {e['label']:>5}: {pt['tokens']} tok, GPU {pt['gpu_ms']} ms, wall {fmt(pt['wall_ms'], 0)} ms, host staging "
                 f"{fmt(pt['host_staging_ms'], 0)} ms; top: " + ", ".join(f"{n} {v['ms']} ms ({v['pct']:.0f}%)" for n, v in top))
    parks = sum(len(r["engine_log"].get("parks") or []) for r in reqs)
    skips = sum(len(r["engine_log"].get("skips") or []) for r in reqs)
    rest = sum(len(r["engine_log"].get("restores") or []) for r in reqs)
    L.append(f"  over {len(reqs)} requests: {n_timing} prefill timing line(s), {parks} park(s), {skips} skipped park(s), "
             f"{rest} restore(s); full lines are in the JSON (engine_log per request)")
    return L


def summary_c_text(c: dict) -> list[str]:
    s = c["summary"]
    side = ",".join(map(str, c["side_after"])) or "none"
    L = ["", f"C. session with side requests: {c['turns']} turns x about {c['words_per_turn']} words, side requests after turn(s) "
             f"{side} ({c['side_messages']} message(s), {c['side_words']} words), sub-agent switch "
             f"{('after turn %d for %d turns' % (c['switch_after'], c['subagent_turns'])) if c['switch_after'] else 'none'}; "
             f"{s['sessions']} session(s)",
         f"{'turn':>4} {'after':>6} {'prompt':>8} {'reused':>8} {'read':>8} {'re-read':>8} {'reuse %':>8} {'TTFT s':>8}"]
    for p in s["per_turn"]:
        pct = None if p["reused_tokens"] is None or not p["prompt_tokens"] else 100 * p["reused_tokens"] / p["prompt_tokens"]
        flag = "switch" if p["after_switch"] else "side" if p["after_side"] else "-"
        L.append(f"{p['turn']:>4} {flag:>6} {fmt(p['prompt_tokens'], 0):>8} {fmt(p['reused_tokens'], 0):>8} "
                 f"{fmt(p['read_tokens'], 0):>8} {fmt(p['reread_tokens'], 0):>8} {fmt(pct, 1):>8} {fmt(p['ttft_s']):>8}")
    L.append(f"total TTFT over the main turns (median of {s['complete_sessions']} complete): {fmt(s['total_ttft_s'])} s")
    for g, v in s["groups"].items():
        if v["n"]:
            L.append(f"  turns 2+ {g:<12} n={v['n']:>2}: median TTFT {fmt(v['ttft_s_median'])} s, read {fmt(v['read_tokens_median'], 0)}, "
                     f"re-read {fmt(v['reread_tokens_median'], 0)} tokens")
    o = s["other_requests"]
    L.append(f"  side requests: n={o['side_n']} median TTFT {fmt(o['side_ttft_s_median'])} s; sub-agent turns: n={o['subagent_n']} "
             f"median TTFT {fmt(o['subagent_ttft_s_median'])} s; failed {o['failed']}")
    if not s["reuse_reported"]:
        L.append("cache reuse: not reported by this server; reused/read/re-read columns are empty")
    stops = [x["stopped"] for x in c["sessions"] if x["stopped"]]
    if stops:
        L.append("stopped early: " + "; ".join(stops[:2]))
    return L


def summary_e_text(e: dict) -> list[str]:
    s = e["summary"]
    L = ["", f"E. warm-delta sweep: base about {size_label(e['base_target_tokens'])} tokens built once (actual "
             f"{fmt(s['base_prompt_tokens'], 0)}, TTFT {fmt(s['base_ttft_s'])} s), deltas {e['deltas']} x {e['repeats']} repeats; "
             f"status: {e['status']}",
         f"{'delta':>6} {'n':>2} {'new tok':>8} {'read':>8} {'depth':>8} {'TTFT s':>8} {'min-max s':>13} {'engine ms':>10}"]
    for p in s["per_delta"]:
        t = p["ttft_s"]
        L.append(f"{p['delta']:>6} {p['n']:>2} {fmt(p['new_tokens_est'], 0):>8} {fmt(p['read_tokens'], 0):>8} "
                 f"{fmt(p['prompt_tokens'], 0):>8} " + (f"{t['median']:>8.2f} {t['min']:>5.2f}-{t['max']:<7.2f}" if t else f"{'-':>8} {'-':>13}")
                 + f" {fmt(p['engine_prompt_ms'], 0):>10}")
    for key, name in (("fit_all", "all deltas"), ("fit_ge_1000", "deltas of 1000+")):
        f = s[key]
        L.append(f"fit ({name}): " + (f"TTFT = {f['intercept_s']:.2f} s + {f['slope_ms_per_token']:.3f} ms per new token (n={f['n']})"
                                      if f else "not enough points"))
    return L


def summary_text(res: dict) -> str:
    L = [f"TTFT bench: {res['label']}  ({res['base']}, model {res.get('model_reported') or res.get('model')}, think "
         f"{'on' if res['args']['think'] == 'on' else 'off'})",
         f"load average 1 min: start {fmt(res['load']['start'], 2)}, end {fmt(res['load']['end'], 2)}"
         + ("  WARNING: above 2.0, the numbers are not clean" if res["load"]["warning"] else "")]
    if budget_text(res):
        L.append(budget_text(res))
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
    if res.get("scenario_c"):
        L += summary_c_text(res["scenario_c"])
    if res.get("scenario_e"):
        L += summary_e_text(res["scenario_e"])
    L += engine_log_text(res)
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
    if a.get("cache_budget_mib") is not None or b.get("cache_budget_mib") is not None:
        def bud(r):
            return "not noted" if r.get("cache_budget_mib") is None else f"{r['cache_budget_mib']} MiB"
        L.append(f"conversation-cache budget (noted by the operator): A {bud(a)}, B {bud(b)}")
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
    if a.get("scenario_c") and b.get("scenario_c"):
        pa, pb = a["scenario_c"]["summary"], b["scenario_c"]["summary"]
        L += ["", "C. session with side requests: median TTFT per main turn in seconds (side = followed a side request, "
                  "switch = followed the sub-agent run)",
              f"{'turn':>4} {'A after':>8} {'B after':>8} {'A TTFT':>8} {'B TTFT':>8} {'change':>8} {'A re-read':>10} {'B re-read':>10}"]
        for i in range(max(len(pa["per_turn"]), len(pb["per_turn"]))):
            x = pa["per_turn"][i] if i < len(pa["per_turn"]) else None
            y = pb["per_turn"][i] if i < len(pb["per_turn"]) else None

            def fl(p):
                return "-" if p is None else "switch" if p.get("after_switch") else "side" if p.get("after_side") else "-"
            L.append(f"{i + 1:>4} {fl(x):>8} {fl(y):>8} {fmt(x and x['ttft_s']):>8} {fmt(y and y['ttft_s']):>8} "
                     f"{pct_change(x and x['ttft_s'], y and y['ttft_s']):>8} {fmt(x and x['reread_tokens'], 0):>10} "
                     f"{fmt(y and y['reread_tokens'], 0):>10}")
        L.append(f"total TTFT over the main turns: A {fmt(pa['total_ttft_s'])} s, B {fmt(pb['total_ttft_s'])} s "
                 f"({pct_change(pa['total_ttft_s'], pb['total_ttft_s'])})")
        for g in ("plain", "after_side", "after_switch"):
            ga, gb = pa["groups"][g], pb["groups"][g]
            if ga["n"] or gb["n"]:
                L.append(f"  turns 2+ {g:<12} median TTFT: A {fmt(ga['ttft_s_median'])} s (n={ga['n']}), B {fmt(gb['ttft_s_median'])} s "
                         f"(n={gb['n']}) {pct_change(ga['ttft_s_median'], gb['ttft_s_median'])}")
    if a.get("scenario_e") and b.get("scenario_e"):
        pa, pb = a["scenario_e"]["summary"], b["scenario_e"]["summary"]
        L += ["", "E. warm-delta sweep: median TTFT in seconds by delta size",
              f"{'delta':>6} {'A new':>7} {'B new':>7} {'A TTFT':>8} {'B TTFT':>8} {'change':>8}"]
        bm = {p["delta"]: p for p in pb["per_delta"]}
        for p in pa["per_delta"]:
            o = bm.get(p["delta"])
            ta, tb = (p["ttft_s"] or {}).get("median"), ((o or {}).get("ttft_s") or {}).get("median")
            L.append(f"{p['delta']:>6} {fmt(p['new_tokens_est'], 0):>7} {fmt(o and o['new_tokens_est'], 0):>7} {fmt(ta):>8} "
                     f"{fmt(tb):>8} {pct_change(ta, tb):>8}" + ("" if o else "  (delta missing in B)"))
        for key, name in (("fit_all", "fit, all deltas"), ("fit_ge_1000", "fit, deltas of 1000+")):
            fa, fb = pa[key], pb[key]
            L.append(f"  {name}: A " + (f"{fa['intercept_s']:.2f} s + {fa['slope_ms_per_token']:.3f} ms/token" if fa else "-")
                     + ", B " + (f"{fb['intercept_s']:.2f} s + {fb['slope_ms_per_token']:.3f} ms/token" if fb else "-"))
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
    ap.add_argument("--scenario", choices=("a", "b", "both", "c", "e", "all"), default="both",
                    help="a: long prompts, b: agent session, both: a+b (default), c: session with side requests and "
                         "an optional sub-agent switch, e: warm-delta sweep, all: a+b+c+e")
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
    ap.add_argument("--side-after", default="3,6", help="scenario C: send a short unrelated side request after these main "
                                                       "turns, e.g. 3,6 (default %(default)s; empty: none)")
    ap.add_argument("--side-messages", type=int, choices=(1, 2), default=1, help="scenario C: messages in a side request (1 or 2)")
    ap.add_argument("--side-words", type=int, default=60, help="scenario C: words of text in a side request (default 60)")
    ap.add_argument("--switch-after", type=int, default=0, help="scenario C: after this main turn run a sub-agent (a second "
                                                               "long conversation, other system prompt and tools), then return; 0: none")
    ap.add_argument("--subagent-turns", type=int, default=2, help="scenario C: sub-agent turns (default 2)")
    ap.add_argument("--delta-base", default=DEFAULT_DELTA_BASE, help="scenario E: size of the conversation built once, K = 1024 "
                                                                    "(default %(default)s)")
    ap.add_argument("--deltas", default=DEFAULT_DELTAS, help="scenario E: new tokens per request (default %(default)s)")
    ap.add_argument("--delta-repeats", type=int, default=3, help="scenario E: repeats of each delta (default 3)")
    ap.add_argument("--delta-max-tokens", type=int, default=16, help="scenario E max_tokens (default 16)")
    ap.add_argument("--note-cache-budget-mib", type=int, default=None, metavar="N",
                    help="scenario D (tight budget): record the server's conversation-cache budget in MiB in the JSON, "
                         "and print it in --compare; the harness cannot set or read it")
    ap.add_argument("--prefill-timing-log", default="", metavar="PATH",
                    help="the engine's log file: after each request the lines added since it started (STRATA_PREFILL_TIMING=1 "
                         "prefill breakdown, conversation-cache park/skip/restore, RESUME) are attached to its record")
    ap.add_argument("--log-settle-s", type=float, default=0.3, help="seconds to wait before reading the engine log after a request")
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
    try:
        side_after = parse_positions(a.side_after)
        deltas = [parse_size(x) for x in a.deltas.split(",") if x.strip()]
        delta_base = parse_size(a.delta_base)
    except ValueError as e:
        print(f"bad option: {e}", file=sys.stderr)
        return 2
    if a.scenario in ("c", "all") and (any(t >= a.turns for t in side_after) or a.switch_after >= a.turns):
        print(f"--side-after / --switch-after must be before the last turn ({a.turns})", file=sys.stderr)
        return 2
    cl = Client(a.host, a.port, a.api_key, think=a.think == "on", timeout=a.timeout, model=a.model)
    if a.prefill_timing_log:
        cl.tap = LogTap(a.prefill_timing_log, settle_s=a.log_settle_s)
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
           "model_reported": info.get("model_reported"), "model": a.model, "cache_budget_mib": a.note_cache_budget_mib,
           "load": {"start": rnd(load0, 2), "end": None, "warning": False}}
    seed = a.seed or random.SystemRandom().randrange(1, 2**31)
    res["args"]["seed"] = seed
    if a.scenario in ("a", "both", "all"):
        res["scenario_a"] = run_a(cl, sizes, a.runs, a.max_tokens, not a.no_calibrate, a.tokens_per_word, seed)
    if a.scenario in ("b", "both", "all"):
        res["scenario_b"] = run_b(cl, a.turns, a.agent_words, a.agent_max_tokens, a.sessions)
    if a.scenario in ("c", "all"):
        res["scenario_c"] = run_c(cl, a.turns, a.agent_words, a.agent_max_tokens, a.sessions, side_after, a.side_messages,
                                  a.side_words, a.switch_after, a.subagent_turns, seed)
    if a.scenario in ("e", "all"):
        res["scenario_e"] = run_e(cl, delta_base, deltas, a.delta_repeats, a.tokens_per_word, a.delta_max_tokens, seed)
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
