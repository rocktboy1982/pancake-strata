#!/usr/bin/env python3
"""bfla_recall.py: the long-context quality gate of the fused prompt attention on the qwen35 engine (Qwen3.6-35B-A3B family).

--mode dense:  STRATA_BFLA_DENSE=1 (the fused kernel over every block; exact up to rounding: expect KL near 0, top-1 ~100%)
--mode sparse: STRATA_BFLA_SPARSE=1 (BFLA block-sparse, LOSSY: this is its gate)

One N-token prompt: filler text from this repository (docs and sources) with NEEDLES access codes hidden at even depths,
then a question that asks for all of them.  Three engine runs (build-hip/strata, which hands qwen35moe files to strata-q36):
  ref     no switch: greedy continuation (the reference) and its full logits per step
  test    the mode's switch: greedy continuation (recall of the needles on its own)
  forced  the mode's switch: teacher-forced on the reference continuation, full logits per step
Reported: needles recalled per arm, top-1 agreement of forced vs ref, mean / max KL(ref || forced), each arm's prompt time
(the TTFT A/B), whether the switch logged "acted" (else the arm measured nothing: VOID), and MOD_PLAN's lossy gate
(mean KL <= 1e-3, top-1 >= 99%, on >= 300 tokens, recall not below the reference).

  bfla_recall.py --mode dense|sparse --exe build-hip/strata --model GGUF --tokenizer PACK/tokenizer --len 65536 --out DIR
                 [--needles 6] [--new 320] [--env K=V ...] [--test-env K=V ...] [--reuse-ref]
--reuse-ref takes the reference from DIR when it is there for the same prompt (run dense and sparse on one reference).
Needs ~2 x new x vocab x 4 bytes in DIR for the logits (320 steps x 248K vocab: ~0.64 GB); the comparison streams them.
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
SWITCH = {"dense": "STRATA_BFLA_DENSE", "sparse": "STRATA_BFLA_SPARSE"}
GATE = {"mean_kl_max": 1e-3, "top1_min": 0.99, "min_tokens": 300}


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


def arm_env(base, mode, extra):
    """The test arms' environment: the base, the mode's switch, then --test-env (STRATA_BFLA_MIN=... etc.)."""
    return dict(base, **{SWITCH[mode]: "1"}, **extra)


def acted(log_text, mode):
    """True when the engine log has the switch's once-only "acted" line (q36_engine.cu)."""
    return f"{SWITCH[mode]} acted" in log_text


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
    return gen, (float(m.group(2)) / 1000.0 if m else float("nan")), p.stderr


def compare_logits(ref_path, test_path, ref_ids):
    """Top-1 agreement of the test logits with the reference picks, and KL(ref || test) per step.

    Row i of each file is the logits that picked token i (the reference file has exactly len(ref_ids) rows, which gives
    the vocab size; the forced file has one more).  Streamed a row at a time, so 300+ steps of a 248K vocab fit."""
    import numpy as np
    n_ref = len(ref_ids)
    a = np.memmap(ref_path, dtype=np.float32, mode="r")
    if n_ref == 0 or a.size % n_ref:
        raise ValueError(f"{ref_path}: {a.size} floats is not {n_ref} rows")
    V = a.size // n_ref
    b = np.memmap(test_path, dtype=np.float32, mode="r")
    if b.size % V:
        raise ValueError(f"{test_path}: {b.size} floats is not a whole number of {V}-wide rows")
    a, b = a.reshape(n_ref, V), b.reshape(-1, V)
    n = min(n_ref, b.shape[0])

    def logsoftmax(x):
        x = x.astype(np.float64)
        m = x.max()
        return x - m - np.log(np.exp(x - m).sum())

    agree, kls = 0, []
    for i in range(n):
        lp, lq = logsoftmax(a[i]), logsoftmax(b[i])
        kls.append(float((np.exp(lp) * (lp - lq)).sum()))
        agree += int(int(b[i].argmax()) == ref_ids[i])
    return agree, n, kls


def gate(res):
    """MOD_PLAN's lossy-proposal gate (mean KL <= 1e-3, top-1 >= 99%) on >= 300 tokens, recall not below the reference,
    and a switch that acted.  Returns (pass, reason)."""
    if not res["acted"]:
        return False, "VOID: the switch never logged 'acted' (prompt below STRATA_BFLA_MIN, or no gfx11 GPU)"
    why = []
    if res["n_tokens"] < GATE["min_tokens"]:
        why.append(f"{res['n_tokens']} tokens < {GATE['min_tokens']}")
    if res["mean_kl"] > GATE["mean_kl_max"]:
        why.append(f"mean KL {res['mean_kl']:.2e} > {GATE['mean_kl_max']:.0e}")
    if res["top1_agree"] < GATE["top1_min"]:
        why.append(f"top-1 {res['top1_agree']:.4f} < {GATE['top1_min']}")
    if res["recall_test"] < res["recall_ref"]:
        why.append(f"recall {res['recall_test']} < {res['recall_ref']}")
    return not why, "; ".join(why) or "pass"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=sorted(SWITCH))
    ap.add_argument("--exe", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--len", type=int, default=65536)
    ap.add_argument("--needles", type=int, default=6)
    ap.add_argument("--new", type=int, default=320)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    ap.add_argument("--env", action="append", default=[], help="K=V for every arm (e.g. HIP_VISIBLE_DEVICES=0)")
    ap.add_argument("--test-env", action="append", default=[], help="K=V for the test arms only (STRATA_BFLA_MIN=...)")
    ap.add_argument("--reuse-ref", action="store_true", help="take ref.ids / ref.logits from --out for the same prompt")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = load_tok(Path(a.tokenizer))
    ids, needles = build_prompt(tok, a.len, a.needles, a.seed)
    ids_file = out / "prompt.ids"
    prompt_text = " ".join(map(str, ids))
    same_prompt = ids_file.exists() and ids_file.read_text() == prompt_text
    ids_file.write_text(prompt_text)
    ctx = ((len(ids) + a.new + 1024) + 1023) // 1024 * 1024
    base = dict(kv.split("=", 1) for kv in a.env)
    tenv = arm_env(base, a.mode, dict(kv.split("=", 1) for kv in a.test_env))
    print(f"prompt {len(ids)} tokens, {a.needles} needles, context {ctx}, mode {a.mode} ({SWITCH[a.mode]}=1)", flush=True)

    def recall(gen):
        text = tok.decode(gen)
        return sum(1 for _, code in needles if code in text), text

    ref_ids_file, ref_logits = out / "ref.ids", out / "ref.logits"
    reuse = a.reuse_ref and same_prompt and ref_ids_file.exists() and ref_logits.exists() and (out / "ref.json").exists()
    if reuse:
        g_ref = [int(x) for x in ref_ids_file.read_text().split()]
        t_ref = json.loads((out / "ref.json").read_text())["prompt_s"]
        reuse = len(g_ref) == a.new
    if not reuse:
        g_ref, t_ref, _ = run(a.exe, a.model, ids_file, ctx, a.new, base, dump=ref_logits, log=out / "ref.log")
        ref_ids_file.write_text(" ".join(map(str, g_ref)))
        (out / "ref.json").write_text(json.dumps({"prompt_s": t_ref}))
    g_test, t_test, log_test = run(a.exe, a.model, ids_file, ctx, a.new, tenv, log=out / f"{a.mode}.log")
    _, t_forced, log_forced = run(a.exe, a.model, ids_file, ctx, a.new, tenv, force_file=ref_ids_file,
                                  dump=out / f"{a.mode}.forced.logits", log=out / f"{a.mode}.forced.log")
    rr, tr = recall(g_ref)
    rt, tt = recall(g_test)
    agree, n, kls = compare_logits(ref_logits, out / f"{a.mode}.forced.logits", g_ref)
    res = {
        "mode": a.mode, "switch": SWITCH[a.mode], "len": len(ids), "needles": a.needles, "new": a.new,
        "recall_ref": rr, "recall_test": rt,
        "top1_agree": agree / n if n else 0.0, "n_tokens": n,
        "mean_kl": sum(kls) / n if n else float("nan"), "max_kl": max(kls) if kls else float("nan"),
        "first_token_same": bool(g_test[:1] == g_ref[:1]),
        "acted": acted(log_test, a.mode) and acted(log_forced, a.mode),
        "prompt_s_ref": t_ref, "prompt_s_test": t_test, "prompt_s_forced": t_forced, "ref_reused": bool(reuse),
        "text_ref": tr, "text_test": tt, "test_env": tenv,
    }
    res["gate_pass"], res["gate"] = gate(res)
    (out / f"result-{a.mode}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("text")}, indent=1))
    print("ref: ", tr.replace("\n", " | ")[:300])
    print("test:", tt.replace("\n", " | ")[:300])


if __name__ == "__main__":
    main()
