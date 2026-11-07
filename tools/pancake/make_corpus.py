"""make_corpus.py - pancake-strata: a held corpus for tools/force_check.py from data/quant_quality/v1 (code, docs,
English, multilingual, agent): per kind, N chat requests over different stretches of the text.
    .venv/bin/python tools/pancake/make_corpus.py --tokenizer DIR --out bench/quality/held.jsonl [--per-kind 4]
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "pancake"))
from bench_engines import chat, load_tok  # noqa: E402

ASK = {"code": "Explain what this code does, then continue it with the next function.",
       "docs": "Summarize this documentation and then continue it.",
       "english": "Continue this text in the same style.",
       "multilingual": "Continue this text in the same language.",
       "agent": "Continue this agent session with the next step."}

ap = argparse.ArgumentParser()
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--per-kind", type=int, default=4)
ap.add_argument("--chars", type=int, default=3000)
a = ap.parse_args()
tok = load_tok(a.tokenizer)
with open(a.out, "w") as out:
    for kind, ask in ASK.items():
        text = (ROOT / "data/quant_quality/v1" / f"{kind}.txt").read_text(encoding="utf-8")
        step = max(1, (len(text) - a.chars) // max(1, a.per_kind))
        for i in range(a.per_kind):
            part = text[i * step:i * step + a.chars]
            ids = tok.encode(chat(part + "\n\n" + ask), parse_special=True)
            out.write(json.dumps({"id": f"{kind}-{i}", "kind": kind, "ids": ids}) + "\n")
print("ok", a.out)
