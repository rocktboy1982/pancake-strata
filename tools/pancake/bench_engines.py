"""bench_engines.py - pancake-strata: the same greedy prompts through engines that speak the --serve protocol
(strata, strata-poly), one engine at a time, results appended to a jsonl.

    .venv/bin/python tools/pancake/bench_engines.py --name poly --tokenizer DIR --out bench/results/x.jsonl \
        --runs 3 -- build-cuda/strata-poly --model M.gguf --max-context 131072 ...

Cases: short (new text), edit (copy-heavy code rewrite), long (~16K-token prompt).  Each case runs --runs times
in the same process; the first run of a case is a cold start for that prompt, later ones may hit the engine's caches
(both engines keep conversation checkpoints), so the jsonl keeps every run.
"""
import argparse, json, os, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "qwen35"))
sys.path.insert(0, str(ROOT / "tools"))
from q36_harness import Eng  # noqa: E402
import strata_tokenizer as ST  # noqa: E402


def load_tok(d):
    d = Path(d)
    vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (d / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((d / "token_type.json").read_text())
    cfg = json.loads((d / "tokenizer.json").read_text(encoding="utf-8")) if (d / "tokenizer.json").exists() else {}
    return ST.Tokenizer(tokens, merges, types, cfg.get("pre") or "qwen35", cfg.get("special_ids") or None)


def chat(user):
    # Qwen3.6's template with thinking off
    return "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def cases():
    src = (ROOT / "src/spec/draft_policy.cpp").read_text(encoding="utf-8")
    long_parts, n = [], 0
    for f in ["src/spec/draft_policy.cpp", "src/spec/suffix_drafter.cpp", "src/spec/controller.cpp",
              "src/core/conversation_memory.cpp", "src/core/conversation_file.cpp", "src/poly/suffix_index.hpp",
              "src/poly/conversation_disk.cpp", "src/poly/guesses.hpp"]:
        t = (ROOT / f).read_text(encoding="utf-8")
        long_parts.append(f"=== {f} ===\n{t}")
    # read-time sweep: fresh prompts of growing length (a unique tag first, so no cache or checkpoint is reused)
    corpus = "\n\n".join(long_parts)
    sweep = []
    for k in (1, 2, 4, 8, 16):
        chars = k * 1000 * 3          # ~3 characters a token for source code
        sweep.append((f"read{k}k", chat(f"[{k}k] " + corpus[:chars] + "\n\nWhat language is this?"), 16))
    return sweep + [
        ("short", chat("Write a Python function that reverses a singly linked list, with a short docstring and two "
                       "unit tests."), 256),
        ("edit", chat("Rename the class DraftPolicy to WindowPolicy everywhere in this file and return the whole "
                      "file, nothing else.\n\n" + src), 512),
        ("long", chat("Here are some source files.\n\n" + "\n\n".join(long_parts) +
                      "\n\nSummarize in five bullet points what these files do together."), 128),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--only", default="")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    tok = load_tok(a.tokenizer)
    log = str(Path(a.out).with_suffix("")) + f".{a.name}.engine.log"
    e = Eng(cmd[0], cmd[1:], log=log)
    print(f"[{a.name}] loaded in {e.load_s:.1f}s: {e.ready}", flush=True)
    out = open(a.out, "a")
    for name, text, n in cases():
        if a.only and name not in a.only.split(","):
            continue
        ids = tok.encode(text, parse_special=True)
        rows = []
        for r in range(a.runs):
            toks, done, ttft, wall = e.solo(ids, n)
            dec = (len(toks) - 1) / (wall - ttft) if len(toks) > 1 and wall > ttft else 0.0
            row = dict(engine=a.name, case=name, run=r, prompt=len(ids), new=len(toks), ttft_s=round(ttft, 3),
                       wall_s=round(wall, 3), decode_tok_s=round(dec, 2), prefill_tok_s=round(len(ids) / ttft, 1),
                       done=done, tokens=toks, cmd=cmd, t=time.strftime("%Y-%m-%dT%H:%M:%S"))
            out.write(json.dumps(row) + "\n")
            out.flush()
            rows.append(row)
            print(f"[{a.name}] {name} run{r}: prompt {len(ids)} ttft {ttft:.2f}s ({row['prefill_tok_s']} tok/s) "
                  f"decode {dec:.1f} tok/s, {len(toks)} new", flush=True)
        print(f"[{a.name}] {name} median decode {statistics.median(x['decode_tok_s'] for x in rows):.1f} tok/s",
              flush=True)
    e.p.stdin.close()
    e.p.wait(timeout=120)


if __name__ == "__main__":
    main()
