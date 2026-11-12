#!/usr/bin/env python3
"""The chunk plan pancake's serve loop gives a prompt (MOD_PLAN 17.5), as a pure function: no engine, no GPU.

A mirror of the C++ rules, read from the code (unconfirmed until a STRATA_TRACE=1 run shows the same chunks):
  - the cuts (src/program/generate.cpp, serve loop, `cuts`): a prompt read from token 0 is read in segments
    [0, root) [root, turn) [turn, n - 1) and the last token goes through the first window.  `turn` is the last
    <|im_start|> (the assistant header that follows is a few tokens); `root` is the first <|im_start|> after token 0
    (the end of the system prompt) when it is at least --prompt-cache-root (2048) tokens in, and before `turn`.
    A segment of --short-read (64) tokens or fewer goes through decode windows.
  - each segment lends one chunk (`lend` -> planned_chunk -> equal_chunk -> request_chunk): min(cap, tokens rounded
    up to 256); STRATA_PREFILL_EQUAL=1 evens the chunks out when the last one would stream anyway; the opt-in
    STRATA_PREFILL_EQUAL_TAIL=1 (include/strata/program/prefill_plan.hpp equal_tail_chunk) evens them out when the
    last one would be staged.
  - Prefill::run_impl reads a segment in chunks of that size, the last one the rest.  A chunk of stream_min tokens
    or more streams every expert the GPU does not hold; a shorter one is staged (routed experts, after the routing).

The pancake fast-hip preset: cap 32768 (STRATA_PREFILL_AUTO_MAX, the log says "prompt chunk auto: 32768 tokens"),
stream_min 4096 (STRATA_PREFILL_CPU_SHARE=0.6 arms the share, STRATA_PREFILL_CPU_SHARE_MAX=4096 is its limit).

    python tools/pancake/chunk_plan.py 32768 65537 100000 [--header 3] [--root 12000] [--equal] [--equal-tail [MIN]]
"""
from __future__ import annotations

import argparse
import sys

CAP = 32768
STREAM_MIN = 4096
SHORT_READ = 64
ROOT_MIN = 2048
EQUAL_TAIL_MIN_TAIL = 256      # prefill_plan.hpp kEqualTailMinTail
EQUAL_TAIL_MIN_CHUNK = 8192    # prefill_plan.hpp kEqualTailMinChunk


def round256(t: int) -> int:
    return (t + 255) // 256 * 256


def request_chunk(tokens: int, cap: int) -> int:
    """generate.cpp request_chunk: the segment rounded up to 256, at most the cap."""
    if tokens <= 0 or cap <= 0:
        return 0
    return min(cap, round256(tokens))


def equal_tail_chunk(tokens: int, cap: int, stream_min: int, min_tail: int = EQUAL_TAIL_MIN_TAIL,
                     min_chunk: int = EQUAL_TAIL_MIN_CHUNK) -> int:
    """prefill_plan.hpp equal_tail_chunk: 0 where the rule does not apply."""
    if tokens <= 0 or cap <= 0 or min_tail < 1:
        return 0
    n = -(-tokens // cap)
    if n < 2:
        return 0
    last = tokens - (n - 1) * cap
    if last >= stream_min or last < min_tail:
        return 0
    c = min(cap, round256(-(-tokens // n)))
    last2 = tokens - (n - 1) * c
    if c < min_chunk or c < stream_min or last2 < stream_min:
        return 0
    return c


def equal_chunk(tokens: int, cap: int, stream_min: int = STREAM_MIN, equal: bool = False, tail_min: int = 0) -> int:
    """generate.cpp equal_chunk with the opt-in hook: tail_min > 0 is STRATA_PREFILL_EQUAL_TAIL on."""
    plain = request_chunk(tokens, cap)
    if tail_min > 0:
        c = equal_tail_chunk(tokens, cap, stream_min, tail_min)
        if c > 0:
            return c
    if not equal or tokens <= 0 or cap <= 0:
        return plain
    n = -(-tokens // cap)
    last = tokens - (n - 1) * cap
    if n <= 1 or last < stream_min:
        return plain
    return min(cap, round256(-(-tokens // n)))


def chunk_lengths(tokens: int, chunk: int) -> list[int]:
    """prefill_plan.hpp chunk_lengths / Prefill::run_impl: full chunks, then the rest."""
    if tokens <= 0 or chunk <= 0:
        return []
    return [min(chunk, tokens - c0) for c0 in range(0, tokens, chunk)]


def segments(n: int, header: int = 3, root: int | None = None) -> list[tuple[int, int]]:
    """The serve loop's parts of a prompt of n tokens read from 0: [0, root) [root, turn) [turn, n - 1)."""
    turn = n - header if 0 < header < n else -1
    cuts = []
    if root is not None and ROOT_MIN <= root < (turn if turn > 0 else n):
        cuts.append(root)
    if turn > 0:
        cuts.append(turn)
    cuts.append(n - 1)
    out, at = [], 0
    for to in cuts:
        if to > at:
            out.append((at, to))
            at = to
    return out


def plan(n: int, header: int = 3, root: int | None = None, cap: int = CAP, stream_min: int = STREAM_MIN,
         equal: bool = False, tail_min: int = 0) -> list[dict]:
    """Every part of the prompt: its range, path (windows / batched) and chunks with their path."""
    out = []
    for a, b in segments(n, header, root):
        seg = b - a
        if seg <= SHORT_READ:
            out.append({"from": a, "to": b, "path": "windows", "chunks": []})
            continue
        c = equal_chunk(seg, cap, stream_min, equal, tail_min)
        chunks = chunk_lengths(seg, c)
        out.append({"from": a, "to": b, "path": "batched", "chunk": c,
                    "chunks": [(t, "streamed" if t >= stream_min else "staged") for t in chunks]})
    return out


def describe(n: int, parts: list[dict]) -> str:
    rows = []
    for p in parts:
        if p["path"] == "windows":
            rows.append(f"[{p['from']},{p['to']}) {p['to'] - p['from']} tokens through windows")
        else:
            ch = " + ".join(f"{t}{'' if k == 'streamed' else ' (staged)'}" for t, k in p["chunks"])
            rows.append(f"[{p['from']},{p['to']}) chunk {p['chunk']}: {ch}")
    return f"{n} tokens: " + "; ".join(rows) + "; last token: first window"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tokens", type=int, nargs="+", help="prompt lengths in tokens")
    ap.add_argument("--header", type=int, default=3, help="tokens from the last <|im_start|> to the end (default 3)")
    ap.add_argument("--root", type=int, default=None, help="end of the system prompt (agent shape), if any")
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--stream-min", type=int, default=STREAM_MIN)
    ap.add_argument("--equal", action="store_true", help="STRATA_PREFILL_EQUAL=1")
    ap.add_argument("--equal-tail", type=int, nargs="?", const=EQUAL_TAIL_MIN_TAIL, default=0,
                    help="STRATA_PREFILL_EQUAL_TAIL=1 (optional value: STRATA_PREFILL_EQUAL_TAIL_MIN)")
    a = ap.parse_args(argv)
    for n in a.tokens:
        print(describe(n, plan(n, a.header, a.root, a.cap, a.stream_min, a.equal, a.equal_tail)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
