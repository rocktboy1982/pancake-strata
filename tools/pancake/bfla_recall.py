#!/usr/bin/env python3
"""bfla_recall.py: the long-context quality gate of STRATA_BFLA (lossy) on the qwen35 engine (Qwen3.6-35B-A3B family).

One N-token prompt: filler text from this repository (docs and sources) with NEEDLES access codes hidden at even depths,
then a question that asks for all of them.  Three engine runs (strata-q36 entry point of build-hip/strata):
  dense   STRATA_BFLA unset: greedy continuation (the reference) and its full logits per step
  bfla    STRATA_BFLA=1:     greedy continuation (recall of the needles on its own)
  forced  STRATA_BFLA=1:     teacher-forced on the dense continuation, full logits per step
Reported: needles recalled per arm, top-1 agreement of forced vs dense over the continuation, mean KL(dense || bfla),
and each arm's prompt time (the TTFT A/B).

  bfla_recall.py --exe build-hip/strata --model GGUF --tokenizer PACK/tokenizer --len 65536 --out DIR [--needles 6]
                 [--new 48] [--env K=V ...] [--skip-dense]
Needs ~2 x new x vocab x 4 bytes in DIR for the logits (48 steps: ~95 MB).
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import strata_tokenizer as ST  # noqa: E402

NAMES = ["AMBER", "BASALT", "CEDAR", "DELTA", "EMBER", "FJORD", "GRANITE", "HARBOR", "IVORY", "JUNIPER", "KESTREL", "LUMEN"]


def load_tok(d: Path):
    vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (d / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((d / "token_type.json").read_text())
    tcfg = {}
    try:
        tcfg = json.loads((d / "tokenizer.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    return ST.Tokenizer(tokens, merges, types, tcfg.get("pre") or "qwen35", tcfg.get("special_ids") or None)


def filler_texts(repo: Path):
    files = sorted(repo.glob("docs/*.md")) + sorted(repo.glob("src/core/*.cpp")) + sorted(repo.glob("src/program/*.cpp"))
    for f in files:
        try:
            yield f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue


def build_prompt(tok, n_tokens: int, n_needles: int, seed: int):
    rng = random.Random(seed)
    needles = [(NAMES[i], f"{rng.randrange(100000, 999999)}") for i in range(n_needles)]
    question = ("\n\n---\nQuestion: the text above hides several lines of the form 'The access code for vault NAME is NUMBER.' "
                "List every vault and its access code, one per line.\nAnswer:\nThe access code for vault")
    q_ids = tok.encode(question)
    budget = n_tokens - len(q_ids)
    ids = []
    for text in filler_texts(HERE.parent.parent):
        ids += tok.encode(text)
        if len(ids) >= budget:
            break
    ids = ids[:budget]
    # needles at even depths (by token position), each on its own line
    for k, (name, code) in reversed(list(enumerate(needles))):
        at = int(len(ids) * (k + 1) / (n_needles + 1))
        ids[at:at] = tok.encode(f"\nThe access code for vault {name} is {code}.\n")
    ids = ids[: n_tokens - len(q_ids)] + q_ids
    return ids, needles


def run(exe, model, ids_file, ctx, n_new, env_extra, force_file=None, dump=None, log=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("STRATA_")}
    env.update(env_extra)
    cmd = [exe, "--model", model, "--max-context", str(ctx), "--tokens", "@" + str(ids_file), "--n", str(n_new)]
    if force_file:
        cmd += ["--force", "@" + str(force_file)]
    if dump:
        cmd += ["--dump-full-logits", str(dump)]
    t0 = time.time()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if log:
        Path(log).write_text(p.stderr)
    if p.returncode != 0:
        sys.exit(f"engine failed ({p.returncode}): {' '.join(cmd)}\n{p.stderr[-3000:]}")
    gen = []
    for line in p.stdout.splitlines():
        if line.startswith("gen:"):
            gen = [int(x) for x in line[4:].split()]
    m = re.search(r"prompt (\d+) tokens in ([0-9.]+) ms", p.stderr)
    return gen, (float(m.group(2)) / 1000.0 if m else float("nan")), time.time() - t0


def read_logits(path, n_steps):
    import numpy as np
    a = np.fromfile(path, dtype=np.float32)
    return a.reshape(n_steps, -1).astype(np.float64)


def logsoftmax(x):
    import numpy as np
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--len", type=int, default=65536)
    ap.add_argument("--needles", type=int, default=6)
    ap.add_argument("--new", type=int, default=48)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    ap.add_argument("--env", action="append", default=[], help="K=V for every arm (e.g. HIP_VISIBLE_DEVICES=0)")
    ap.add_argument("--bfla-env", action="append", default=[], help="K=V for the BFLA arms only (STRATA_BFLA_MIN=...)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = load_tok(Path(a.tokenizer))
    ids, needles = build_prompt(tok, a.len, a.needles, a.seed)
    ids_file = out / "prompt.ids"
    ids_file.write_text(" ".join(map(str, ids)))
    ctx = ((len(ids) + a.new + 1024) + 1023) // 1024 * 1024
    base = dict(kv.split("=", 1) for kv in a.env)
    bfla = dict(base, STRATA_BFLA="1", **dict(kv.split("=", 1) for kv in a.bfla_env))
    print(f"prompt {len(ids)} tokens, {a.needles} needles, context {ctx}", flush=True)

    def recall(gen):
        text = tok.decode(gen)
        return sum(1 for _, code in needles if code in text), text

    g_dense, t_dense, _ = run(a.exe, a.model, ids_file, ctx, a.new, base, dump=out / "dense.logits", log=out / "dense.log")
    (out / "dense.ids").write_text(" ".join(map(str, g_dense)))
    g_bfla, t_bfla, _ = run(a.exe, a.model, ids_file, ctx, a.new, bfla, log=out / "bfla.log")
    _, t_forced, _ = run(a.exe, a.model, ids_file, ctx, a.new, bfla, force_file=out / "dense.ids", dump=out / "forced.logits",
                         log=out / "forced.log")
    rd, td = recall(g_dense)
    rb, tb = recall(g_bfla)
    L1 = read_logits(out / "dense.logits", a.new)
    L2 = read_logits(out / "forced.logits", a.new)
    import numpy as np
    n = min(len(g_dense), L1.shape[0], L2.shape[0])
    agree = int((L2[:n].argmax(axis=1) == np.array(g_dense[:n])).sum())
    lp, lq = logsoftmax(L1[:n]), logsoftmax(L2[:n])
    kl = float((np.exp(lp) * (lp - lq)).sum())
    res = {
        "len": len(ids), "needles": a.needles, "new": a.new,
        "recall_dense": rd, "recall_bfla": rb,
        "top1_agree": agree / n, "mean_kl": kl / n,
        "first_token_same": bool(g_bfla[:1] == g_dense[:1]),
        "prompt_s_dense": t_dense, "prompt_s_bfla": t_bfla, "prompt_s_forced": t_forced,
        "text_dense": td, "text_bfla": tb, "bfla_env": bfla,
    }
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("text")}, indent=1))
    print("dense:", td.replace("\n", " | ")[:300])
    print("bfla: ", tb.replace("\n", " | ")[:300])


if __name__ == "__main__":
    main()
