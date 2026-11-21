#!/usr/bin/env python3
"""prefill_typing_bench: does reading the draft while the user types cut the time to first token of a long message?

    python3 tools/pancake/prefill_typing_bench.py --port 8080 --sizes 4k,8k,16k --out typing.json

LEADS.md Part Q1 (STRATA_PREFILL_TYPING=1, serve/prefill_warm.py).  Run it against a server started with
STRATA_PREFILL_TYPING=1 and --prompt-cache 3 or more (the pinned checkpoint needs it).  Per size, alternating pairs on
unique content (so no cache can help either arm):

  cold: one long user message, sent as a real request                       -> TTFT cold
  warm: the first --draft-frac of the same kind of message posted to /v1/prefill (what the page sends after a typing
        pause), then the whole message sent as a real request after --gap seconds -> TTFT warm, tokens reused

The warm request itself takes about the prefill time of the draft; that time is the user's typing time and is reported as
`warm_post_s`, never added to the warm TTFT.  Run 1 of each size is discarded.  Stdlib only; the engine is never started or
stopped here.  Exactness is by construction (the engine reads the same tokens, only earlier); the ledger row says so.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ttft_bench as T  # noqa: E402

QUESTION = "\n\nQuestion: name the first word of the list above."


def warm_post(cl: "T.Client", draft: str, think: bool) -> dict:
    """POST the draft to /v1/prefill -> {status, seconds, body}."""
    body = {"model": cl.model, "messages": [{"role": "user", "content": draft}]}
    if not think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Content-Type": "application/json"}
    if cl.key:
        headers["Authorization"] = f"Bearer {cl.key}"
    conn = http.client.HTTPConnection(cl.host, cl.port, timeout=cl.timeout)
    t0 = time.perf_counter()
    res = {"status": None, "seconds": None, "body": None}
    try:
        conn.request("POST", "/v1/prefill", body=json.dumps(body).encode(), headers=headers)
        r = conn.getresponse()
        res["status"] = r.status
        raw = r.read()
        try:
            res["body"] = json.loads(raw)
        except ValueError:
            res["body"] = raw[:300].decode("utf-8", "replace")
    except Exception as e:                                                  # noqa: BLE001
        res["body"] = repr(e)[:300]
    finally:
        res["seconds"] = round(time.perf_counter() - t0, 3)
        conn.close()
    return res


def draft_of(text: str, frac: float) -> str:
    """The first `frac` of the message, cut at a word boundary (what a user has typed so far)."""
    cut = text[: max(1, int(len(text) * frac))]
    sp = cut.rfind(" ")
    return cut[:sp] if sp > 0 else cut


def one_pair(cl, words: int, frac: float, gap: float, rng, think: bool, order_warm_first: bool) -> dict:
    out = {}
    for arm in (("warm", "cold") if order_warm_first else ("cold", "warm")):
        text = T.unique_variant(words, rng)
        if arm == "warm":
            w = warm_post(cl, draft_of(text, frac), think)
            out["warm_post_s"] = w["seconds"]
            out["warm_post_status"] = w["status"]
            out["warm_post_body"] = w["body"] if isinstance(w["body"], dict) and "strata_prefill" not in (w["body"] or {}) \
                else (w["body"].get("strata_prefill") if isinstance(w["body"], dict) else w["body"])
            time.sleep(gap)
        d = T.digest(cl.chat([{"role": "user", "content": text + QUESTION}], 8))
        out[arm] = d
    return out


def run(cl, sizes, runs, frac, gap, seed, think, log=print) -> dict:
    rng = random.Random(seed)
    res = {"frac": frac, "gap_s": gap, "sizes": []}
    ratio = 1.35
    for target in sizes:
        words = max(10, round((target - T.PROMPT_OVERHEAD_TOKENS) / ratio))
        entry = {"label": T.size_label(target), "target_tokens": target, "pairs": []}
        res["sizes"].append(entry)
        for i in range(runs):
            p = one_pair(cl, words, frac, gap, rng, think, order_warm_first=bool(i % 2))
            p["run"] = i + 1
            p["discarded"] = i == 0
            entry["pairs"].append(p)
            c, w = p["cold"], p["warm"]
            if c["ok"] and c["prompt_tokens"] and i == 0:
                ratio = c["prompt_tokens"] / words
                if abs(c["prompt_tokens"] - target) > 0.03 * target:
                    words = max(10, round(words * target / c["prompt_tokens"]))
            log(f"{entry['label']:>5} run {i + 1}{' (discarded)' if i == 0 else ''}: cold TTFT {c['ttft_s']} s, warm TTFT "
                f"{w['ttft_s']} s (reused {w['reused_tokens']} of {w['prompt_tokens']}), warm post {p.get('warm_post_s')} s "
                f"status {p.get('warm_post_status')}")
        kept = [p for p in entry["pairs"] if not p["discarded"] and p["cold"]["ok"] and p["warm"]["ok"]]
        entry["summary"] = {
            "n": len(kept),
            "cold_ttft_s": T.rnd(T.median(p["cold"]["ttft_s"] for p in kept)),
            "warm_ttft_s": T.rnd(T.median(p["warm"]["ttft_s"] for p in kept)),
            "saved_s": T.rnd(T.median(p["cold"]["ttft_s"] - p["warm"]["ttft_s"] for p in kept)),
            "warm_reused_tokens": T.median(p["warm"]["reused_tokens"] for p in kept),
            "prompt_tokens": T.median(p["warm"]["prompt_tokens"] for p in kept),
            "warm_post_s": T.rnd(T.median(p.get("warm_post_s") for p in kept)),
        } if kept else None
    return res


def text_summary(res: dict) -> str:
    rows = ["size   n  cold TTFT s  warm TTFT s  saved s  reused/prompt tokens  warm post s"]
    for e in res["sizes"]:
        s = e.get("summary")
        if not s:
            rows.append(f"{e['label']:>5}  no usable pair")
            continue
        rows.append(f"{e['label']:>5} {s['n']:>2}  {s['cold_ttft_s']:>10}  {s['warm_ttft_s']:>10}  {s['saved_s']:>7}  "
                    f"{s['warm_reused_tokens']}/{s['prompt_tokens']}  {s['warm_post_s']}")
    return "\n".join(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--api-key", default=os.environ.get("TTFT_BENCH_API_KEY", ""))
    ap.add_argument("--model", default="default")
    ap.add_argument("--sizes", default="4k,8k,16k", help="prompt sizes in tokens, K = 1024")
    ap.add_argument("--runs", type=int, default=4, help="pairs per size; run 1 is discarded (default 4)")
    ap.add_argument("--draft-frac", type=float, default=0.9, help="share of the message already typed when the warm is sent")
    ap.add_argument("--gap", type=float, default=0.3, help="seconds between the warm's answer and the real request")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--think", choices=("off", "on"), default="off")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if not 0.05 <= a.draft_frac < 1.0:
        ap.error("--draft-frac must be in 0.05 .. 0.99")
    cl = T.Client(a.host, a.port, a.api_key, think=a.think == "on", timeout=a.timeout, model=a.model)
    h = cl.get_json("/health")
    if not h or not h.get("prefill_typing"):
        print("this server does not advertise prefill_typing in /health: start it with STRATA_PREFILL_TYPING=1", file=sys.stderr)
        return 2
    sizes = [T.parse_size(s) for s in a.sizes.split(",") if s.strip()]
    res = run(cl, sizes, max(2, a.runs), a.draft_frac, a.gap, a.seed or random.randrange(1 << 30), a.think == "on")
    print(text_summary(res))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
