#!/usr/bin/env python3
"""mid_edit_bench: what a request that edits the MIDDLE of an earlier prefix reuses (MOD_PLAN.md section 6 item 42,
STRATA_CKPT_STRIDE; LEADS.md Part J).

    python3 tools/pancake/mid_edit_bench.py --port 8080 --label base --out mid-base.json
    python3 tools/pancake/mid_edit_bench.py --port 8080 --label stride4k --out mid-stride.json   (engine started with STRATA_CKPT_STRIDE=4096)
    python3 tools/pancake/mid_edit_bench.py --compare mid-base.json mid-stride.json

Per round: one conversation of --tokens tokens (one system + one long user message of numbered paragraphs, unique per
round) is read cold, then the SAME conversation is sent again with ONE paragraph rewritten at each of --fractions of the
way in (same length), so the cache can reuse only what lies before the edit.  Recorded per edit: the edit's token
position (estimated from the base prompt's actual tokens), the tokens the server reused, the tokens it read again, TTFT,
and 'reused_vs_edit' = reused / edit position (1.0 = everything below the edit was reused; the root or the turn
checkpoint alone gives a low value).  The engine is never started or stopped here.  The numbers are what the server says;
only the edit position is estimated: the share of the words before the edit times the prompt's tokens.  Stdlib only;
uses ttft_bench.
"""
import argparse
import json
import random
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ttft_bench as T  # noqa: E402

SCHEMA = 1


def make_paragraphs(n: int, words: int, rng: random.Random) -> list[str]:
    """n numbered paragraphs of `words` words each, unique text."""
    return [f"Paragraph {i + 1}: " + T.random_words(words, rng) for i in range(n)]


def edit_paragraph(paras: list[str], frac: float, rng: random.Random) -> tuple[list[str], int]:
    """A copy with the paragraph at fraction `frac` (0 < frac < 1) rewritten (same word count, different words);
    returns (paragraphs, index)."""
    idx = min(len(paras) - 1, max(1, int(frac * len(paras))))
    words = len(paras[idx].split()) - 2
    out = list(paras)
    out[idx] = f"Paragraph {idx + 1}: " + T.random_words(words, rng)
    return out, idx


def messages(nonce: str, paras: list[str]) -> list[dict]:
    return [{"role": "system", "content": f"(session {nonce})\n{T.SYSTEM}"},
            {"role": "user", "content": "\n".join(paras) + "\nReply with one word."}]


def edit_token_position(prompt_tokens: int, paras: list[str], idx: int) -> int:
    """Estimated token position of paragraph idx's start: the share of the words before it times the prompt's tokens."""
    total = sum(len(p.split()) for p in paras)
    before = sum(len(p.split()) for p in paras[:idx])
    return int(prompt_tokens * before / total) if total else 0


def run(cl, tokens: int, fractions: list[float], rounds: int, para_words: int, max_tokens: int, seed: int, log=print) -> dict:
    rng = random.Random(seed)
    n_paras = max(8, tokens // max(1, para_words))
    res = {"tokens_target": tokens, "fractions": fractions, "rounds": rounds, "rows": [], "status": "measured"}
    for rd in range(1, rounds + 1):
        nonce = uuid.uuid4().hex[:12]
        paras = make_paragraphs(n_paras, para_words, rng)
        _, base = T.tapped(cl, messages(nonce, paras), max_tokens)
        if not base["ok"]:
            res["status"] = "failed building the base: " + (base["err"] or "")
            log(res["status"])
            break
        log(f"round {rd} base: {base['prompt_tokens']} tokens, TTFT {base['ttft_s']} s")
        for fr in fractions:
            edited, idx = edit_paragraph(paras, fr, rng)
            _, d = T.tapped(cl, messages(nonce, edited), max_tokens)
            pos = edit_token_position(base["prompt_tokens"], paras, idx)
            d.update(round=rd, fraction=fr, paragraph=idx, edit_token=pos)
            d["reused_vs_edit"] = None if d.get("reused_tokens") is None or not pos else T.rnd(d["reused_tokens"] / pos, 3)
            res["rows"].append(d)
            log(f"  edit at {fr:.2f} (~token {pos}): reused {d.get('reused_tokens')}, read {d.get('read_tokens')}, "
                f"TTFT {d['ttft_s']} s, reused/edit {d['reused_vs_edit']}")
            paras = edited   # the next edit builds on the text the cache now holds
    res["summary"] = summarize(res)
    return res


def summarize(res: dict) -> list[dict]:
    out = []
    for fr in res["fractions"]:
        rows = [r for r in res["rows"] if r["fraction"] == fr and r["ok"]]
        out.append({"fraction": fr, "n": len(rows),
                    "ttft_s": T.rnd(T.median(r["ttft_s"] for r in rows)),
                    "reused_tokens": T.median(r["reused_tokens"] for r in rows),
                    "read_tokens": T.median(r["read_tokens"] for r in rows),
                    "reused_vs_edit": T.rnd(T.median(r["reused_vs_edit"] for r in rows))})
    return out


def compare_text(a: dict, b: dict) -> str:
    L = [f"{'edit at':>8} {'reused A':>9} {'reused B':>9} {'read A':>8} {'read B':>8} {'TTFT A':>8} {'TTFT B':>8} {'TTFT B/A':>9}"]
    sa = {s["fraction"]: s for s in a["scenario"]["summary"]}
    for s in b["scenario"]["summary"]:
        x = sa.get(s["fraction"])
        if not x:
            continue
        ratio = "-" if not x["ttft_s"] or not s["ttft_s"] else f"{s['ttft_s'] / x['ttft_s']:.3f}"
        L.append(f"{s['fraction']:>8.2f} {T.fmt(x['reused_tokens'], 0):>9} {T.fmt(s['reused_tokens'], 0):>9} "
                 f"{T.fmt(x['read_tokens'], 0):>8} {T.fmt(s['read_tokens'], 0):>8} {T.fmt(x['ttft_s'], 2):>8} "
                 f"{T.fmt(s['ttft_s'], 2):>8} {ratio:>9}")
    L.append("A = first file, B = second; medians over the rounds; every number is the server's own, none is a claim until "
             "it has a TEST_LEDGER.md row")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--label", default="engine")
    ap.add_argument("--out", default="")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--model", default="default")
    ap.add_argument("--tokens", default="32k", help="conversation size, K = 1024 (default %(default)s)")
    ap.add_argument("--fractions", default="0.25,0.5,0.75")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--para-words", type=int, default=150)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=3600)
    a = ap.parse_args(argv)
    if a.compare:
        ra, rb = (json.loads(Path(p).read_text(encoding="utf-8")) for p in a.compare)
        print(compare_text(ra, rb))
        return 0
    fractions = [float(x) for x in a.fractions.split(",") if x.strip()]
    if not fractions or any(not 0 < f < 1 for f in fractions):
        print("--fractions must be numbers between 0 and 1", file=sys.stderr)
        return 2
    cl = T.Client(a.host, a.port, a.api_key, think=False, timeout=a.timeout, model=a.model)
    seed = a.seed or random.SystemRandom().randrange(1, 2**31)
    res = {"schema": SCHEMA, "tool": "mid_edit_bench", "label": a.label, "seed": seed,
           "scenario": run(cl, T.parse_size(a.tokens), fractions, a.rounds, a.para_words, a.max_tokens, seed)}
    out = Path(a.out or f"mid_{a.label}.json")
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for s in res["scenario"]["summary"]:
        print(s)
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
