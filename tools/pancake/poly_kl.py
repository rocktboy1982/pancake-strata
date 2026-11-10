"""poly_kl.py - pancake-strata: the KL gate for a rounding-level change in strata-poly (MOD_PLAN F17: STRATA_FATTN,
STRATA_FDEC).  tools/force_check.py drives the main engine (STRATA_LOGPOS / STRATA_FOLLOW); strata-poly has its own
teacher forcing, `--tokens @PROMPT --follow @ANSWER --top-probs N`, which prints the N most likely ids of every step
of the answer to stderr ("top id:p id:p ...").  This runs it.

  1. The prompt: the A/B harness's long* text (abtest.corpus(), src/**/*.cpp, cut to --tokens tokens) with a question,
     in Qwen3.6's template (thinking off), tokenized with the pack's tokenizer.
  2. The answer: one greedy run of the base engine (no arm's variables), --new tokens.
  3. Teacher-forced top-N of that answer: the base twice (the noise floor; strata-poly's kernels are deterministic,
     so it is usually 0) and every arm once.
  4. Per arm against the base's first forced run: mean and largest KL(base || arm) per step and the top-1 agreement.
     KL is taken on the base's top-N ids plus one bucket for the rest of the vocabulary (a coarsening of the two
     distributions, so a lower bound of the full KL when both lists hold those ids; an id of the base's list that the
     arm's list lacks gets the smaller of the arm's N-th probability and its rest mass shared out).
  Verdict per arm: PASS when mean KL <= --max-kl and top-1 agreement >= --min-top1, else HOLD (the switch stays opt-in).
  The defaults (KL 1e-3, top-1 0.99) are a proposal: q8_0 KV costs < 0.00012 KL (docs/PANCAKE_FINDINGS.md, the KV
  study), BFLA's sparse prompt attention measured 0.0014 / 0.0045 at 64K / 128K; the owner sets the bar.

    python tools/pancake/poly_kl.py --config pancake-qwen36-ud-q4_k_xl.json --tokenizer <pack>/tokenizer \\
        --tokens 32000 --new 256 --arm fattn:STRATA_FATTN=1 --arm fdec:STRATA_FDEC=1 \\
        --arm both:STRATA_FATTN=1,STRATA_FDEC=1 --out bench/quality/<day>-f17-32k
    python tools/pancake/poly_kl.py --selftest      (the arithmetic, no engine)

Needs the GPU alone (one engine at a time; each run loads the model).  The config is a rendered pancake config
(run-pancake.sh writes pancake-<preset>.json at the repo root): its exe, args, env and lib_dirs are used; --tokens
mode reads one prompt and exits, so the serve-only arguments do nothing there.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUESTION = "\n\nIn three sentences: what do these files have in common?"


def chat(user: str) -> str:
    # Qwen3.6's template with thinking off (as tools/pancake/bench_engines.py)
    return "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def load_tok(d: str):
    sys.path.insert(0, str(ROOT / "tools"))
    import strata_tokenizer as ST  # noqa: E402
    d = Path(d)
    vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (d / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((d / "token_type.json").read_text())
    cfg = json.loads((d / "tokenizer.json").read_text(encoding="utf-8")) if (d / "tokenizer.json").exists() else {}
    return ST.Tokenizer(tokens, merges, types, cfg.get("pre") or "qwen35", cfg.get("special_ids") or None)


def parse_top(stderr: str) -> list[dict[int, float]]:
    """The --top-probs lines of a run: one {id: p} a step."""
    steps = []
    for line in stderr.splitlines():
        if not line.startswith("top "):
            continue
        d = {}
        for item in line.split()[1:]:
            i, p = item.split(":")
            d[int(i)] = float(p)
        steps.append(d)
    return steps


def kl_step(p: dict[int, float], q: dict[int, float]) -> float:
    """KL(p || q) on p's ids plus the rest of the vocabulary (see the module's note)."""
    if not p:
        return 0.0
    q_listed = sum(q.values())
    q_rest_all = max(1e-12, 1.0 - q_listed)
    missing = [i for i in p if i not in q]
    fill = min(min(q.values()) if q else 1e-12, q_rest_all / (len(missing) + 1)) if missing else 0.0
    kl = 0.0
    q_used = 0.0
    for i, pi in p.items():
        qi = q.get(i, fill)
        q_used += qi
        if pi > 0:
            kl += pi * math.log(pi / max(qi, 1e-12))
    p_rest = max(0.0, 1.0 - sum(p.values()))
    q_rest = max(1e-12, 1.0 - q_used)
    if p_rest > 0:
        kl += p_rest * math.log(p_rest / q_rest)
    return max(0.0, kl)


def compare(base: list[dict[int, float]], arm: list[dict[int, float]]) -> dict:
    n = min(len(base), len(arm))
    if n == 0:
        return {"steps": 0, "mean_kl": float("nan"), "max_kl": float("nan"), "top1": float("nan")}
    kls = [kl_step(base[i], arm[i]) for i in range(n)]
    top1 = sum(max(base[i], key=base[i].get) == max(arm[i], key=arm[i].get) for i in range(n) if base[i] and arm[i])
    return {"steps": n, "steps_base": len(base), "steps_arm": len(arm), "mean_kl": sum(kls) / n, "max_kl": max(kls),
            "top1": top1 / n}


def run(cfg: dict, extra: list[str], env_add: dict[str, str], log: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    for k, v in (cfg.get("env") or {}).items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = str(v)
    env.update(env_add)
    libs = [d for d in cfg.get("lib_dirs") or [] if d]
    if libs:
        env["LD_LIBRARY_PATH"] = ":".join(libs + [env.get("LD_LIBRARY_PATH", "")]).rstrip(":")
    cmd = [cfg["exe"]] + list(cfg.get("args") or []) + extra
    r = subprocess.run(cmd, cwd=cfg.get("cwd") or ROOT, env=env, capture_output=True, text=True)
    log.write_text(" ".join(cmd) + "\n" + json.dumps(env_add) + "\n\n" + r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"poly_kl: {cmd[0]} exited with {r.returncode}; log {log}")
    return r


def selftest() -> int:
    p = {1: 0.7, 2: 0.2, 3: 0.05}
    assert kl_step(p, dict(p)) == 0.0
    q = {1: 0.6, 2: 0.3, 3: 0.05}
    want = 0.7 * math.log(0.7 / 0.6) + 0.2 * math.log(0.2 / 0.3)   # the rest buckets are equal (0.05 each)
    assert abs(kl_step(p, q) - want) < 1e-12, (kl_step(p, q), want)
    assert kl_step(p, {1: 0.7, 2: 0.2, 4: 0.05}) > 0.0          # id 3 missing from q's list
    assert kl_step({}, q) == 0.0
    c = compare([p, p, q], [p, q, q])
    assert c["steps"] == 3 and abs(c["top1"] - 1.0) < 1e-12 and c["max_kl"] > 0 and c["mean_kl"] > 0
    c = compare([{5: 0.9}], [{6: 0.9}])
    assert c["top1"] == 0.0
    steps = parse_top("strata-poly: x\ntop 5:0.9 7:0.05\nnoise\ntop 6:0.5 5:0.4\n")
    assert steps == [{5: 0.9, 7: 0.05}, {6: 0.5, 5: 0.4}], steps
    print("poly_kl selftest: ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--config", help="a rendered pancake config (pancake-<preset>.json)")
    ap.add_argument("--tokenizer", help="the pack's tokenizer directory (vocab.json, merges.txt, token_type.json)")
    ap.add_argument("--tokens", type=int, default=32000, help="prompt tokens, about (default 32000)")
    ap.add_argument("--new", type=int, default=256, help="answer tokens forced and compared (default 256)")
    ap.add_argument("--top", type=int, default=20, help="--top-probs N (default 20)")
    ap.add_argument("--arm", action="append", default=[], help="name:KEY=V[,KEY=V...] (repeat)")
    ap.add_argument("--max-kl", type=float, default=1e-3)
    ap.add_argument("--min-top1", type=float, default=0.99)
    ap.add_argument("--out", help="directory for the ids, logs and result.json")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.config and a.tokenizer and a.out and a.arm):
        ap.error("--config, --tokenizer, --out and at least one --arm are needed (or --selftest)")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import abtest  # noqa: E402  (the long* text)
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = load_tok(a.tokenizer)
    text = chat(abtest.SUMMARIZE + abtest.corpus(int(a.tokens * abtest.LONG_CPT)) + QUESTION)
    ids = tok.encode(text, parse_special=True)
    (out / "prompt.ids").write_text(",".join(map(str, ids)))
    print(f"poly_kl: prompt {len(ids)} tokens, answer {a.new}, top {a.top}", flush=True)
    base_run = run(cfg, ["--tokens", f"@{out / 'prompt.ids'}", "--max-new", str(a.new)], {}, out / "answer.log")
    answer = base_run.stdout.strip().splitlines()[-1] if base_run.stdout.strip() else ""
    if not answer:
        raise SystemExit("poly_kl: the base run printed no answer ids")
    (out / "answer.ids").write_text(answer)
    forced = ["--tokens", f"@{out / 'prompt.ids'}", "--follow", f"@{out / 'answer.ids'}", "--max-new", str(a.new),
              "--top-probs", str(a.top)]
    tops = {}
    for name, env in [("base", {}), ("base2", {})] + [
            (s.split(":", 1)[0], dict(kv.split("=", 1) for kv in s.split(":", 1)[1].split(",") if kv)) for s in a.arm]:
        r = run(cfg, forced, env, out / f"{name}.log")
        tops[name] = parse_top(r.stderr)
        print(f"poly_kl: {name}: {len(tops[name])} steps", flush=True)
    result = {"prompt_tokens": len(ids), "new": a.new, "top": a.top, "max_kl": a.max_kl, "min_top1": a.min_top1,
              "arms": {}}
    floor = compare(tops["base"], tops["base2"])
    result["noise_floor"] = floor
    print(f"noise floor (base twice): mean KL {floor['mean_kl']:.2e}, max {floor['max_kl']:.2e}, "
          f"top-1 {floor['top1']:.4f} over {floor['steps']} steps")
    for name in tops:
        if name in ("base", "base2"):
            continue
        c = compare(tops["base"], tops[name])
        c["verdict"] = "PASS" if c["mean_kl"] <= a.max_kl and c["top1"] >= a.min_top1 else "HOLD"
        result["arms"][name] = c
        print(f"{name}: mean KL {c['mean_kl']:.2e}, max {c['max_kl']:.2e}, top-1 {c['top1']:.4f} over "
              f"{c['steps']} steps -> {c['verdict']}")
    (out / "result.json").write_text(json.dumps(result, indent=1))
    return 0 if all(c["verdict"] == "PASS" for c in result["arms"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
