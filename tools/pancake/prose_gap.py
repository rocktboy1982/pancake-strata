#!/usr/bin/env python3
"""prose_gap.py - MOD_PLAN 17.2: measure one engine on one fixed prose prompt, the same way for pancake and Avalon.

Background (measured 2026-11-11, bench/results/2026-11-11-vs-avalon/ on port/rdna3-decode): on the same IQ2_XS GGUF
Avalon decoded prose 5-10% faster than pancake.  Read from that file's draft counters (pancake-docs/PROSE_GAP_AVALON.md,
"what the old numbers already say"): Avalon commits more tokens per verify window on prose; the cost per window is
about the same.  This harness measures exactly those quantities, per request, on any OpenAI-compatible server:

    tokens per window   = predicted_n / (predicted_n - draft_n_accepted)    (assumes a window commits 1 + accepted tokens)
    ms per window       = predicted_ms / windows
    accepted per window, drafted per window, acceptance %  (draft_n_accepted / draft_n)

It sends ONE fixed prompt N times (run 1 is the warm-up and is discarded), greedy, thinking off, 512 tokens by default.
For pancake's Strata server (detected by /health service == "strata") it also snapshots /metrics and /v1/status before
and after the series and stores the difference of the numeric counters (totals: drafts_offered, drafts_accepted,
decode_ms, ...), reusing tools/strata_bench.try_json.  Avalon's llama-server has none of these endpoints; nothing
is snapshotted there (no /metrics unless the server was started with --metrics, which the harness does not assume).

    python3 tools/pancake/prose_gap.py --port 8080 --label pancake --out pancake.json      # pancake
    python3 tools/pancake/prose_gap.py --port 11437 --label avalon --out avalon.json       # Avalon
    python3 tools/pancake/prose_gap.py --compare avalon.json pancake.json                  # side by side, offline

It starts nothing, builds nothing and touches no GPU by itself: it only sends HTTP requests to a server you started.
Run it on a quiet machine (the harness records the load average before and after and warns above 2.0).  Standard library only.
Exit codes: 0 ok, 2 server unreachable or no request succeeded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

# the same five prompts as bench/results/2026-11-11-vs-avalon/cmp_bench.py (so the numbers compare with that file)
PROMPTS = {
    "story": "Write a short story (about 400 words) about a lighthouse keeper.",
    "tips": "List 15 practical tips for reducing cloud costs, one line each.",
    "moe": "Explain in about 300 words how a mixture-of-experts language model works.",
    "js": "Write a JavaScript debounce function with JSDoc and a usage example.",
    "lru": "Write a Python class implementing an LRU cache with type hints and tests.",
}
SCHEMA = 1
LOAD_WARN = 2.0


def _fallback_try_json(base, path, key="", timeout=20):
    try:
        req = urllib.request.Request(base + path, headers={"Authorization": "Bearer " + key} if key else {})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:  # noqa: BLE001
        return None


try:  # reuse the bench's helper when it imports cleanly (it pulls in sibling tools); the fallback is the same few lines
    from strata_bench import try_json  # type: ignore  # noqa: E402
except Exception:  # noqa: BLE001
    try_json = _fallback_try_json


def ask(base: str, prompt: str, max_tokens: int, key: str = "", timeout: float = 1800) -> dict:
    """One non-streamed chat request, the way cmp_bench.py sends it.  Returns the answer's `timings` plus wall_s."""
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    t0 = time.time()
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    t = dict(resp.get("timings") or {})
    t["wall_s"] = time.time() - t0
    usage = resp.get("usage") or {}
    t.setdefault("predicted_n", usage.get("completion_tokens"))
    try:
        t["text"] = resp["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError, AttributeError):
        t["text"] = None
    return t


def derive(t: dict) -> dict:
    """The per-request numbers.  `windows` = predicted_n - draft_n_accepted is an ASSUMPTION (each verify window commits
    one token plus its accepted drafts); the raw counters are kept next to it."""
    n = t.get("predicted_n") or 0
    tps = t.get("predicted_per_second")
    ms = t.get("predicted_ms")
    if ms is None and tps:
        ms = 1000.0 * n / tps
    if tps is None and ms:
        tps = 1000.0 * n / ms
    acc, dn = t.get("draft_n_accepted"), t.get("draft_n")
    row = {"tok": n, "tps": tps, "decode_ms": ms, "draft_n": dn, "draft_n_accepted": acc, "wall_s": t.get("wall_s")}
    text = t.get("text")
    if text is not None:   # the same text on two engines means the acceptance numbers compare like for like
        row["text_sha256"] = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]
        row["text_head"] = text[:80]
    if acc is not None and dn:
        windows = n - acc
        row["accept_pct"] = 100.0 * acc / dn
        if windows > 0:
            row["windows"] = windows
            row["tok_per_window"] = n / windows
            row["accepted_per_window"] = acc / windows
            row["drafted_per_window"] = dn / windows
            if ms:
                row["ms_per_window"] = ms / windows
    return row


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _sd(xs):
    xs = [x for x in xs if x is not None]
    return statistics.stdev(xs) if len(xs) > 1 else None


FIELDS = ("tps", "tok_per_window", "accepted_per_window", "drafted_per_window", "accept_pct", "ms_per_window")


def summarize(rows: list[dict]) -> dict:
    kept = [r for r in rows if not r.get("discarded")]
    out = {"n_kept": len(kept), "n_discarded": len(rows) - len(kept)}
    for f in FIELDS:
        vals = [r.get(f) for r in kept]
        out[f] = {"median": _med(vals), "mean": statistics.fmean([v for v in vals if v is not None]) if any(v is not None for v in vals) else None,
                  "sd": _sd(vals), "min": min([v for v in vals if v is not None], default=None),
                  "max": max([v for v in vals if v is not None], default=None)}
    return out


# ------------------------------------------------------------------------------------------------ pancake snapshots
def is_pancake(base: str, key: str = "") -> bool:
    h = try_json(base, "/health", key, 5)
    return bool(isinstance(h, dict) and h.get("service") == "strata")


def snapshot(base: str, key: str = "") -> dict:
    """/metrics and /v1/status as the server gives them (None where an endpoint is missing).  Only the parts that
    help: engine facts, the running totals and hardware at this moment, the status block."""
    m = try_json(base, "/metrics", key, 30) or {}
    s = try_json(base, "/v1/status", key, 20) or {}
    return {"time": time.time(), "engine": m.get("engine"), "totals": m.get("totals"), "hardware": m.get("hardware"),
            "live": m.get("live"), "status": s}


def numeric_diff(before, after, _depth=0):
    """after - before for every number both sides have (nested dicts, depth-limited); other values are skipped."""
    out = {}
    if not isinstance(before, dict) or not isinstance(after, dict) or _depth > 4:
        return out
    for k, a in after.items():
        b = before.get(k)
        if isinstance(a, bool) or isinstance(b, bool):
            continue
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            out[k] = a - b
        elif isinstance(a, dict) and isinstance(b, dict):
            d = numeric_diff(b, a, _depth + 1)
            if d:
                out[k] = d
    return out


# --------------------------------------------------------------------------------------- llama-server's own log
# llama-server prints per request: "draft acceptance = 0.57 ( 275 accepted / 471 generated), mean len =  2.15" where
# mean len = 1 + accepted / verify steps (tools/server/server-context.cpp in Avalon), i.e. the MEASURED tokens per
# window.  Given the server's log file (Avalon's service writes one), the lines added during the series are parsed.
MEAN_LEN = re.compile(r"draft acceptance = ([0-9.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([0-9.]+)")


def log_size(path) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def parse_server_log(path, start: int) -> list[dict]:
    try:
        with open(path, "rb") as f:
            f.seek(start)
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    return [{"acceptance": float(m.group(1)), "accepted": int(m.group(2)), "generated": int(m.group(3)), "mean_len": float(m.group(4))}
            for m in MEAN_LEN.finditer(text)]


# ------------------------------------------------------------------------------------------------------------ report
def fmt(v, nd=1):
    return "-" if v is None else f"{v:.{nd}f}"


def text_summary(res: dict) -> str:
    s = res["summary"]
    L = [f"prose_gap  label={res['label']}  engine={res['engine_kind']}  port={res['port']}  prompt={res['prompt_name']}  "
         f"max_tokens={res['max_tokens']}",
         f"runs: {s['n_kept']} kept + {s['n_discarded']} discarded (run 1 = warm-up)   load avg {res['loadavg_before'][0]:.2f} -> "
         f"{res['loadavg_after'][0]:.2f}" + ("   WARNING: machine not quiet" if res.get("noisy") else "")]
    L.append(f"{'':22s}{'median':>9s}{'mean':>9s}{'sd':>8s}{'min':>9s}{'max':>9s}")
    names = {"tps": "decode tok/s", "tok_per_window": "tokens / window", "accepted_per_window": "accepted / window",
             "drafted_per_window": "drafted / window", "accept_pct": "acceptance %", "ms_per_window": "ms / window"}
    for f in FIELDS:
        d = s[f]
        nd = 2 if "window" in f and f != "ms_per_window" else 1
        L.append(f"{names[f]:22s}{fmt(d['median'], nd):>9s}{fmt(d['mean'], nd):>9s}{fmt(d['sd'], nd):>8s}{fmt(d['min'], nd):>9s}{fmt(d['max'], nd):>9s}")
    L.append("per run (tok/s, tok/window): " + "  ".join(
        f"{'[' if r.get('discarded') else ''}{fmt(r.get('tps'))},{fmt(r.get('tok_per_window'), 2)}{']' if r.get('discarded') else ''}"
        for r in res["runs"]))
    if res.get("snapshot_diff"):
        d = res["snapshot_diff"].get("totals") or {}
        keep = {k: d[k] for k in ("requests", "output_tokens", "decode_ms", "drafts_offered", "drafts_accepted") if k in d}
        if keep:
            L.append("server /metrics totals over the series: " + ", ".join(f"{k} +{v:g}" for k, v in keep.items()))
    if res.get("server_log_requests"):
        ml = [x["mean_len"] for x in res["server_log_requests"]]
        L.append(f"server log 'mean len' (measured tokens/window, {len(ml)} requests incl. the discarded one): "
                 + " ".join(f"{x:.2f}" for x in ml))
    shas = [r.get("text_sha256") for r in res["runs"] if not r.get("discarded") and r.get("text_sha256")]
    if shas:
        L.append(f"answer text: {len(set(shas))} distinct of {len(shas)} kept runs (sha {shas[0]}); greedy on one server should repeat")
    L.append("tokens/window and ms/window are derived (window = predicted_n - draft_n_accepted); nothing here is a conclusion by itself.")
    return "\n".join(L)


def compare_text(a: dict, b: dict) -> str:
    L = [f"prose_gap compare: A={a['label']} ({a['engine_kind']})  B={b['label']} ({b['engine_kind']})  prompt A={a['prompt_name']} B={b['prompt_name']}"]
    if a["prompt_name"] != b["prompt_name"] or a["max_tokens"] != b["max_tokens"]:
        L.append("WARNING: prompts or max_tokens differ, the rows are not comparable")
    L.append(f"{'median':22s}{'A':>10s}{'B':>10s}{'B/A':>8s}")
    for f in FIELDS:
        x, y = a["summary"][f]["median"], b["summary"][f]["median"]
        ratio = (y / x) if (x and y) else None
        L.append(f"{f:22s}{fmt(x, 2):>10s}{fmt(y, 2):>10s}{fmt(ratio, 3):>8s}")
    sda, sdb = a["summary"]["tps"]["sd"], b["summary"]["tps"]["sd"]
    L.append(f"run-to-run sd of tok/s: A {fmt(sda)}  B {fmt(sdb)}  (a gap below ~1 sd of a series is not a result)")
    return "\n".join(L)


# -------------------------------------------------------------------------------------------------------------- main
def run_series(args) -> dict:
    base = f"http://{args.host}:{args.port}"
    prompt = args.prompt or PROMPTS[args.prompt_name]
    kind = {"auto": None, "pancake": "pancake", "llama": "llama"}[args.engine]
    if kind is None:
        kind = "pancake" if is_pancake(base, args.key) else "llama"
    load0 = os.getloadavg()
    log0 = log_size(args.server_log) if args.server_log else 0
    snap0 = snapshot(base, args.key) if kind == "pancake" else None
    rows, errors = [], []
    for i in range(1, args.runs + 1):
        try:
            t = ask(base, prompt, args.max_tokens, args.key, args.timeout)
        except (urllib.error.URLError, OSError, ValueError) as e:
            errors.append(f"run {i}: {e}")
            print(f"run {i}: ERROR {e}", file=sys.stderr, flush=True)
            continue
        r = derive(t)
        r["run"] = i
        r["discarded"] = i == 1
        rows.append(r)
        print(f"run {i}{' (discarded)' if i == 1 else ''}: {r['tok']} tok @ {fmt(r['tps'])} tok/s"
              + (f", {fmt(r.get('tok_per_window'), 2)} tok/window, accept {fmt(r.get('accept_pct'))}%" if "tok_per_window" in r else ""),
              flush=True)
    load1 = os.getloadavg()
    snap1 = snapshot(base, args.key) if kind == "pancake" else None
    res = {"schema": SCHEMA, "label": args.label, "engine_kind": kind, "host": args.host, "port": args.port, "time": time.time(),
           "prompt_name": "custom" if args.prompt else args.prompt_name, "prompt": prompt, "max_tokens": args.max_tokens,
           "runs": rows, "errors": errors, "summary": summarize(rows), "loadavg_before": list(load0), "loadavg_after": list(load1),
           "noisy": max(load0[0], load1[0]) > LOAD_WARN, "windows_assumption": "windows = predicted_n - draft_n_accepted"}
    if args.server_log:
        res["server_log"] = args.server_log
        res["server_log_requests"] = parse_server_log(args.server_log, log0)
    if snap0 is not None:
        res["snapshot_before"], res["snapshot_after"] = snap0, snap1
        res["snapshot_diff"] = {"totals": numeric_diff(snap0.get("totals") or {}, snap1.get("totals") or {}),
                                "hardware": numeric_diff(snap0.get("hardware") or {}, snap1.get("hardware") or {})}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--label", default="engine")
    ap.add_argument("--runs", type=int, default=6, help="requests sent, run 1 discarded (default 6)")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--prompt-name", choices=sorted(PROMPTS), default="story")
    ap.add_argument("--prompt", help="a custom prompt (overrides --prompt-name)")
    ap.add_argument("--engine", choices=["auto", "pancake", "llama"], default="auto",
                    help="pancake = also snapshot /metrics and /v1/status; auto asks /health")
    ap.add_argument("--server-log", help="llama-server's log file (Avalon: avalon-service.log): the 'mean len' lines the series adds "
                    "are stored as the measured tokens per window")
    ap.add_argument("--key", default="", help="API key if the server needs one")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--out", help="write the full JSON here")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"), help="offline: print two saved series side by side")
    args = ap.parse_args(argv)
    if args.compare:
        a, b = (json.load(open(p)) for p in args.compare)
        print(compare_text(a, b))
        return 0
    if args.port is None:
        ap.error("--port is required (or use --compare)")
    if args.runs < 2:
        ap.error("--runs must be at least 2 (run 1 is discarded)")
    res = run_series(args)
    if not any(not r["discarded"] for r in res["runs"]):
        print("no request succeeded: " + "; ".join(res["errors"] or ["server did not answer"]), file=sys.stderr)
        return 2
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))
    print()
    print(text_summary(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
