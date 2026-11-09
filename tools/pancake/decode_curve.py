"""decode_curve.py - pancake-strata MOD_PLAN item 2 (phase 13 P3b): decode speed ALONG a reply, after short and long prompts.

The question item 2 asks: after a long prompt, is decode slower because the prompt path's borrowed cache slots are
still being refilled (then the first part of the reply is slow and the rest recovers), or because the reply itself
is harder to draft (then it is slow all along)?  This streams each reply, timestamps every streamed piece, and prints
tok/s per segment of the reply.  Cases (temperature 0, thinking off):
  edit        a code edit, ~2K-token prompt (the same task as the A/B harness's)
  edit-long   the SAME edit task with ~11.7K tokens of unrelated source code in front of it: same kind of reply, long prompt
  long        the A/B harness's summary of the long source text
Compare edit vs edit-long: a refill cost shows as early segments of edit-long below edit's, late segments equal.

    python tools/pancake/decode_curve.py [--url http://127.0.0.1:8080] [--cases edit,edit-long,edit] [--seg 64] [--max 512]
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def prompts():
    src = (ROOT / "src/spec/draft_policy.cpp").read_text()
    files = sorted((ROOT / "src/spec").glob("*.cpp"))[:4]
    long_text = "".join(f.read_text()[:20000] for f in files)[:64000]
    edit = "Rewrite the following C++ file, renaming every variable to snake_case. Output only the full code.\n\n" + src[:6000]
    return {
        "edit": edit,
        "edit-long": "Reference material (not part of the task):\n\n" + long_text + "\n\nTask:\n" + edit,
        "long": "Summarize what each of these source files does, one paragraph per file.\n\n" + long_text,
    }


def stream(url, text, max_tokens):
    body = {"model": "x", "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    pieces, usage, first = [], None, None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            j = json.loads(data)
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices", []):
                c = (ch.get("delta") or {}).get("content") or ""
                if c:
                    now = time.time()
                    if first is None:
                        first = now
                    pieces.append((now, len(c)))
    return t0, first, pieces, usage


def curve(first, pieces, n_tok, seg):
    """tok/s per segment of `seg` tokens; tokens are spread over the streamed characters (pieces are a few tokens)."""
    total = sum(n for _, n in pieces)
    if total == 0 or n_tok == 0:
        return []
    out, cum, last_t, last_tok, bound = [], 0, first, 0.0, seg
    for t, n in pieces:
        cum += n
        tok = cum / total * n_tok
        while tok >= bound and bound <= n_tok:
            dt = t - last_t
            out.append((bound, (bound - last_tok) / dt if dt > 0 else float("nan")))
            last_t, last_tok, bound = t, bound, bound + seg
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--cases", default="edit,edit-long,edit")
    ap.add_argument("--seg", type=int, default=64)
    ap.add_argument("--max", type=int, default=512)
    a = ap.parse_args()
    P = prompts()
    for name in a.cases.split(","):
        t0, first, pieces, usage = stream(a.url, P[name], a.max)
        n_tok = (usage or {}).get("completion_tokens", 0)
        p_tok = (usage or {}).get("prompt_tokens", 0)
        c = curve(first, pieces, n_tok, a.seg)
        dec = n_tok / (pieces[-1][0] - first) if pieces and pieces[-1][0] > first else 0
        print(f"{name:10s} prompt {p_tok:6d}  ttft {first - t0 if first else 0:6.2f}s  {n_tok} tokens  decode {dec:6.1f} tok/s  "
              "per segment: " + " ".join(f"{v:5.0f}" for _, v in c))


if __name__ == "__main__":
    main()
