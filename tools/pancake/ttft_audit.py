#!/usr/bin/env python3
"""ttft_audit.py - where an agent turn's time to first token went, from the logs a pancake server already writes
(MOD_PLAN 17.0, pancake-docs/TTFT_LEVERS.md).  Read only: no model, no GPU, no server.

    python tools/pancake/ttft_audit.py [--turns pancake-turns.jsonl] [--engine-log pancake-flashnext.log]
                                       [--min-reused 8192] [--since 2026-11-11T10:00] [--json OUT]

From the per-turn log (item 73; numbers only) every request is put in one class:
  live      it continued the live state: reused is the previous prompt_tokens + previous engine_generated, less 0-2
            (the last prompt token is always read by the first window);
  fallback  it continued the previous request's conversation, but from a checkpoint at or after that request's
            prompt end (previous prompt_tokens - 64 <= reused < the live length): the re-rendered last reply did
            not match the generated tokens, so the last assistant turn was read again (and the live state parked);
  switch    reused > 0 from anywhere else (a parked conversation, a shallower checkpoint, a side request's prefix);
  lost      reused == 0 although the request looks like the next turn of a long conversation seen shortly before it
            (an earlier request of >= --min-reused tokens, at most --max-gap-s earlier, with
            previous_prompt + max(16, previous_output / 2) <= prompt <= previous_prompt + previous_output +
            --max-growth): a full re-read the conversation cache should have avoided;
  cold      reused == 0 otherwise (a new conversation, a side request, a benchmark prompt).
It prints per class the count, the summed and median prompt_ms and the tokens read; for the agent-like turns
(reused >= --min-reused) the share of prompt tokens read again; the median TTFT by tokens read of the agent-like live
turns (the warm-turn floor); and `first_engine_token_s - prompt_ms`, the time outside the engine's prompt timer
(queue, pipe, first window).

From the engine log (optional) it summarises the conversation-cache lines, which sit inside the next request's
prompt time on HIP (no async park there): "parked N tokens in X ms" by size band, "skip parking (...)",
"restored N tokens (...) in X ms", and the "file tier I/O this request: the OS read X MB from storage" lines (experts
read from the NVMe during a request: a cold page cache).

Limits: the turn log does not name the client or the conversation, so "lost" and "switch" are heuristics; the engine
log has no timestamps, so its lines are counted, not tied to requests.
"""
import argparse
import json
import re
import sys
import time

BANDS = ((1, 40), (41, 100), (101, 300), (301, 800), (801, 2000), (2001, 4095), (4096, 1 << 30))
PARK_BANDS = (0, 4096, 16384, 32768, 49152, 65536, 98304, 1 << 30)
CLASSES = ("live", "fallback", "switch", "lost", "cold")

PARK_RE = re.compile(r"conversation cache: parked (\d+) tokens in ([0-9.]+) ms")
SKIP_RE = re.compile(r"conversation cache: skip parking \(([^)]*)\)")
RESTORE_RE = re.compile(r"conversation cache: restored (\d+) tokens \((\w+)\) in ([0-9.]+) ms")
FILE_RE = re.compile(r"file tier I/O this request: the OS read ([0-9.]+) MB from storage")


def _num(r, k, default=0):
    v = r.get(k)
    return default if v is None else v


def _out(r):
    return _num(r, "engine_generated", _num(r, "output_tokens"))


def load_turns(path, since=None):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("prompt_ms") is None or r.get("time") is None:
                continue
            if since is not None and r["time"] < since:
                continue
            recs.append(r)
    recs.sort(key=lambda r: r["time"])
    return recs


def classify(recs, min_reused=8192, max_growth=8192, max_gap_s=900.0):
    """-> list of (record, class), the classes of the module text."""
    out = []
    prev = None
    recent = []          # (time, prompt_tokens, output) of earlier long requests, newest last
    for r in recs:
        reused = _num(r, "reused")
        prompt = _num(r, "prompt_tokens")
        if reused > 0 and prev is not None:
            p_prompt = _num(prev, "prompt_tokens")
            live_len = p_prompt + _out(prev)
            if live_len - 2 <= reused <= live_len:
                cls = "live"
            elif p_prompt - 64 <= reused < live_len - 2:
                cls = "fallback"
            else:
                cls = "switch"
        elif reused > 0:
            cls = "switch"
        else:
            cls = "cold"
            for t, p_tok, p_out in reversed(recent[-8:]):
                if r["time"] - t > max_gap_s:
                    break
                if p_tok + max(16, p_out // 2) <= prompt <= p_tok + p_out + max_growth:
                    cls = "lost"
                    break
        out.append((r, cls))
        if prompt >= min_reused:
            recent.append((r["time"], prompt, _out(r)))
        prev = r
    return out


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def summarize_turns(rows, min_reused=8192):
    s = {"classes": {}, "agent_like": {}, "live_bands": [], "outside_timer_ms": {}}
    for cls in CLASSES:
        rr = [r for r, c in rows if c == cls]
        ms = [r["prompt_ms"] for r in rr]
        s["classes"][cls] = {"n": len(rr), "prompt_s": round(sum(ms) / 1000, 1), "median_prompt_ms": _median(ms),
                             "tokens_read": sum(_num(r, "prompt_read") for r in rr),
                             "prompt_tokens": sum(_num(r, "prompt_tokens") for r in rr)}
    agent = [(r, c) for r, c in rows if _num(r, "reused") >= min_reused]
    if agent:
        rd = sum(_num(r, "prompt_read") for r, _ in agent)
        pt = sum(_num(r, "prompt_tokens") for r, _ in agent)
        s["agent_like"] = {"n": len(agent), "tokens_read": rd, "prompt_tokens": pt,
                           "share": round(rd / pt, 4) if pt else 0.0,
                           "by_class": {c: sum(1 for _, k in agent if k == c) for c in ("live", "fallback", "switch")}}
    live = [r for r, c in agent if c == "live"]
    for lo, hi in BANDS:
        ms = [r["prompt_ms"] for r in live if lo <= _num(r, "prompt_read") <= hi]
        if ms:
            s["live_bands"].append({"read": [lo, hi], "n": len(ms), "median_ms": _median(ms),
                                    "min_ms": min(ms), "max_ms": max(ms)})
    gaps = sorted(1000 * r["first_engine_token_s"] - r["prompt_ms"] for r, _ in rows
                  if r.get("first_engine_token_s") is not None)
    if gaps:
        def q(p):
            return round(gaps[min(len(gaps) - 1, int(p * (len(gaps) - 1) + 0.5))], 1)
        s["outside_timer_ms"] = {"n": len(gaps), "p50": q(0.5), "p90": q(0.9), "p99": q(0.99), "max": round(gaps[-1], 1)}
    return s


def summarize_engine_log(lines):
    parks, skips, restores, files = [], [], [], []
    for line in lines:
        m = PARK_RE.search(line)
        if m:
            parks.append((int(m.group(1)), float(m.group(2))))
            continue
        m = SKIP_RE.search(line)
        if m:
            skips.append(m.group(1))
            continue
        m = RESTORE_RE.search(line)
        if m:
            restores.append((int(m.group(1)), m.group(2), float(m.group(3))))
            continue
        m = FILE_RE.search(line)
        if m:
            files.append(float(m.group(1)))
    bands = []
    for lo, hi in zip(PARK_BANDS, PARK_BANDS[1:]):
        t = [ms for n, ms in parks if lo <= n < hi]
        if t:
            bands.append({"tokens": [lo, hi - 1], "n": len(t), "mean_ms": round(sum(t) / len(t), 1),
                          "min_ms": min(t), "max_ms": max(t)})
    return {"parks": len(parks), "park_s": round(sum(ms for _, ms in parks) / 1000, 1), "park_bands": bands,
            "skips": len(skips), "skip_reasons": skips[-10:], "restores": len(restores),
            "restore_ms_16k_plus": [round(x[2], 1) for x in restores if x[0] >= 16384][-10:],
            "requests_with_storage_reads_over_1gb": sum(1 for mb in files if mb > 1024),
            "storage_read_gb_max": round(max(files) / 1024, 1) if files else 0.0}


def report(ts, es, min_reused=8192):
    lines = ["TTFT audit (pancake-docs/TTFT_LEVERS.md)", "",
             "class        n   prompt s  median ms  tokens read / prompt tokens"]
    for cls, v in ts["classes"].items():
        med = "-" if v["median_prompt_ms"] is None else f"{v['median_prompt_ms']:.0f}"
        share = v["tokens_read"] / v["prompt_tokens"] if v["prompt_tokens"] else 0.0
        lines.append(f"{cls:9s} {v['n']:5d} {v['prompt_s']:9.1f} {med:>10s}  {v['tokens_read']:,} / {v['prompt_tokens']:,}"
                     f" ({100 * share:.2f}%)")
    a = ts["agent_like"]
    if a:
        lines += ["", f"agent-like turns (reused >= {min_reused}): n={a['n']} {a['by_class']}; tokens read again "
                      f"{a['tokens_read']:,} of {a['prompt_tokens']:,} ({100 * a['share']:.2f}%)"]
    if ts["live_bands"]:
        lines += ["", "agent-like live turns, TTFT by tokens read (the warm-turn floor):"]
        for b in ts["live_bands"]:
            hi = b["read"][1] if b["read"][1] < 1 << 29 else "more"
            lines.append(f"  read {b['read'][0]:5d}-{hi:>5}: n={b['n']:3d} median {b['median_ms']:7.0f} ms"
                         f"  range {b['min_ms']:.0f}-{b['max_ms']:.0f}")
    o = ts["outside_timer_ms"]
    if o:
        lines += ["", f"outside the engine's prompt timer (queue, pipe, first window): p50 {o['p50']} ms, p90 {o['p90']},"
                      f" p99 {o['p99']}, max {o['max']} (n={o['n']})"]
    if es is not None:
        lines += ["", f"engine log: {es['parks']} parks ({es['park_s']} s, synchronous on HIP), {es['skips']} skipped parks,"
                      f" {es['restores']} restores"]
        for b in es["park_bands"]:
            lines.append(f"  park {b['tokens'][0]:6d}-{b['tokens'][1]:6d} tokens: n={b['n']:4d} mean {b['mean_ms']:6.0f} ms"
                         f"  range {b['min_ms']:.0f}-{b['max_ms']:.0f}")
        for why in es["skip_reasons"]:
            lines.append(f"  skip parking: {why}")
        if es["restore_ms_16k_plus"]:
            lines.append(f"  restores of 16K+ tokens (ms): {es['restore_ms_16k_plus']}")
        lines.append(f"  requests that read > 1 GB of experts from storage: {es['requests_with_storage_reads_over_1gb']}"
                     f" (max {es['storage_read_gb_max']} GB)")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--turns", default="pancake-turns.jsonl")
    ap.add_argument("--engine-log", default=None)
    ap.add_argument("--min-reused", type=int, default=8192, help="a long conversation / an agent-like turn")
    ap.add_argument("--max-growth", type=int, default=8192, help="tokens a turn may add and still be the next turn")
    ap.add_argument("--max-gap-s", type=float, default=900.0, help="seconds between two turns of one conversation, at most")
    ap.add_argument("--since", default=None, help="local time, e.g. 2026-11-11T10:00")
    ap.add_argument("--json", default=None, help="also write the summary as JSON")
    a = ap.parse_args(argv)
    since = time.mktime(time.strptime(a.since, "%Y-%m-%dT%H:%M")) if a.since else None
    rows = classify(load_turns(a.turns, since), a.min_reused, a.max_growth, a.max_gap_s)
    ts = summarize_turns(rows, a.min_reused)
    es = None
    if a.engine_log:
        with open(a.engine_log, errors="replace") as f:
            es = summarize_engine_log(f)
    print(report(ts, es, a.min_reused))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"turns": ts, "engine_log": es}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
