"""quality_gate.py - pancake-strata quality gate as a process (MOD_PLAN phase 14): token forcing for every arm of an A/B spec.

Same spec as tools/pancake/abtest.py ({"base": CFG, "arms": {"name": {"env": {...}, "args": {...}}, ...}}).  The first arm
is the stock: it is scored twice (the run-to-run noise); every other arm is scored once and judged against the two
stock arms with tools/force_check.py's rules (rate rule + large-margin event rule, SpecPilot): PASS / REJECT /
INCONCLUSIVE.  The fixed text is bench/quality/set.json (or --set).  An arm whose output is exact by construction (a
placement-only switch) does not need this; lossy ones (route prior, deferral rank guard, depth-graded prior) do.

    python tools/pancake/quality_gate.py SPEC.json --out bench/quality-hip/<day>-<name> [--set bench/quality/set.json]
    python tools/pancake/quality_gate.py SPEC.json --out ... --set bench/quality/set-v2.json --fill     first use of a grown set
    python tools/pancake/quality_gate.py SPEC.json --out ... --set bench/quality/set-v2.json --dry-run  the plan and the set, no engine

The 20-request set (5,322 tokens) is INCONCLUSIVE 22% of the time for an arm identical to stock (pancake-docs/batch3/promotion.md);
bench/quality/set-v2.json (tools/pancake/build_quality_set.py: the 20 old requests + 28 agent-shaped ones, ~11K scored tokens) is the grown set.  Its
new requests have "cont": null: the continuation is the STOCK arm's greedy output (tools/force_check.py gen), so a set with pending
requests refuses to run unless --fill is given; --fill generates them with the first arm (one engine start, before the scoring),
writes the complete set to <out>/set.filled.json and scores with it.  Commit that file as the set and use it from then on (the same
tokens every run: the old runs stay comparable on the 20 common requests, force_check.py check pairs the requests it finds in both).
--dry-run checks the spec and the set and prints what would run (engine starts, commands, a time estimate); it starts nothing.
"""
import argparse
import collections
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abtest  # noqa: E402

ROOT = abtest.ROOT
PY = abtest.PY
VOCAB = 248320                  # Qwen3.x tokenizer: every prompt / continuation id must be below this
MS_START, MS_PER_PROMPT_TOKEN, MS_PER_NEW_TOKEN = 1600.0, 0.375, 10.0   # prefill: percache_sim.py's fit (1583 ms + 0.375 ms/token); decode: ~100 tok/s, the median of the 2026-11-11 short/edit rows


# ------------------------------------------------------------------------------------------------ the set (pure)
def pending(st):
    """The requests of a set that still need their continuation."""
    return [it for it in st["items"] if it.get("cont") is None]


def set_problems(st, old=None, min_requests=0, vocab=VOCAB):
    """Problems of a token-forcing set (empty list = fine); `old` = the set whose items must come first, unchanged."""
    probs, items = [], st.get("items") or []
    if len(items) < max(1, min_requests):
        probs.append(f"{len(items)} requests (need at least {max(1, min_requests)})")
    seen = set()
    for it in items:
        i = it.get("id")
        if not isinstance(i, str) or not i or i in seen:
            probs.append(f"{i!r}: id missing or duplicated")
        seen.add(i)
        if not isinstance(it.get("kind"), str):
            probs.append(f"{i}: kind missing")
        for key, may_be_null in (("prompt", False), ("cont", True)):
            v = it.get(key)
            if v is None and may_be_null:
                continue
            if not isinstance(v, list) or not v or not all(isinstance(x, int) and not isinstance(x, bool) and 0 <= x < vocab for x in v):
                probs.append(f"{i}: {key} must be a non-empty list of token ids below {vocab}")
    if old:
        for a, b in zip(old["items"], items):
            if a != b:
                probs.append(f"{a['id']}: the old request is not copied verbatim (the old results would not be comparable)")
        if len(items) < len(old["items"]):
            probs.append(f"fewer requests than the old set ({len(old['items'])})")
    return probs


def set_summary(st):
    items = st["items"]
    prompts = sorted(len(it["prompt"]) for it in items)
    scored = sum(len(it["cont"]) for it in items if it.get("cont"))
    kinds = collections.Counter(it["kind"] for it in items)
    return (f"{len(items)} requests ({len(pending(st))} without continuation), prompts {prompts[0]} / {prompts[len(prompts) // 2]} / {prompts[-1]} "
            f"tokens (min / median / max), {scored} scored tokens; kinds " + ", ".join(f"{k} {n}" for k, n in sorted(kinds.items())))


def merge_conts(st, gen):
    """The set with every pending continuation taken from `gen` (force_check.py gen's output: items with id and cont)."""
    by_id = {it["id"]: it.get("cont") for it in gen["items"]}
    items = []
    for it in st["items"]:
        if it.get("cont") is None:
            if not by_id.get(it["id"]):
                raise ValueError(f"{it['id']}: the generation produced no continuation")
            it = {**it, "cont": by_id[it["id"]]}
        items.append(it)
    return {**st, "items": items}


def estimate_seconds(st):
    """Rough time to score the set once: prefill by the fitted line, decode at ~100 tok/s (an estimate, not a measurement of this set)."""
    return sum(MS_START + MS_PER_PROMPT_TOKEN * len(it["prompt"]) + MS_PER_NEW_TOKEN * len(it.get("cont") or [300]) for it in st["items"]) / 1000.0


def score(cfg_path, set_path, tsv, topk=2):
    cfg = json.load(open(cfg_path))
    envs = []
    for k, v in (cfg.get("env") or {}).items():
        envs += ["--env", f"{k}={v}"]
    env = dict(os.environ, LD_LIBRARY_PATH="/opt/rocm/core-10.0/lib", HIP_VISIBLE_DEVICES="0")
    cmd = [str(ROOT / "tools/pancake/capped.sh"), PY, "tools/force_check.py", "score", cfg_path, set_path, tsv] + (["--topk", str(topk)] if topk != 2 else []) + envs
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-2000:] + r.stderr[-2000:])
        return False
    return True


def fill(st, cfg_path, out):
    """Generates the pending continuations with the stock config (force_check.py gen, greedy) -> the complete set, or None on a failure."""
    todo = pending(st)
    rows = out / "pending.jsonl"
    rows.write_text("".join(json.dumps({"id": it["id"], "kind": it["kind"], "ids": it["prompt"]}) + "\n" for it in todo))
    gen = str(out / "gen.json")
    cmd = [str(ROOT / "tools/pancake/capped.sh"), PY, "tools/force_check.py", "gen", cfg_path, gen, "--corpus", str(rows), "--per-kind", str(len(todo)), "--offset", "0"]
    env = dict(os.environ, LD_LIBRARY_PATH="/opt/rocm/core-10.0/lib", HIP_VISIBLE_DEVICES="0")
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-2000:] + r.stderr[-2000:])
        return None
    full = merge_conts(st, json.load(open(gen)))
    json.dump(full, open(out / "set.filled.json", "w"), separators=(",", ":"))
    return full


def dry_run(spec, a, st, set_path):
    """What a run would do, from the spec and the set alone."""
    arms = list(spec["arms"])
    base = spec["base"] if os.path.isabs(spec["base"]) else str(ROOT / spec["base"])
    probs = set_problems(st)
    if not os.path.exists(base):
        probs.append(f"base preset {base} not found")
    if len(arms) < 1:
        probs.append("the spec needs at least one arm")
    print(f"set {set_path}: " + set_summary(st))
    print(f"spec: base {spec['base']}, arms {arms} (the first is the stock, scored twice)")
    todo = pending(st)
    if todo and not a.fill:
        probs.append(f"{len(todo)} requests have no continuation: run once with --fill (the stock arm generates them)")
    starts = len(arms) + 1 + (1 if todo and a.fill else 0)
    per = estimate_seconds(st)
    print(f"plan: {starts} engine starts" + (f" (first: force_check.py gen of {len(todo)} continuations with {arms[0]}, <= 300 tokens each)" if todo and a.fill else "")
          + f", then scoring: {arms[0]} a, {arms[0]} b" + "".join(f", {x}" for x in arms[1:]))
    print(f"estimate: ~{per / 60:.0f} min of scoring per arm (prefill + decode, engine start not counted; an estimate), ~{per * (len(arms) + 1) / 60:.0f} min for all arms")
    for p in probs:
        print("ERROR: " + p)
    print("dry run: " + ("FAILED" if probs else "ok"))
    return 1 if probs else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec")
    ap.add_argument("--out", required=True)
    # once --fill has run and a complete bench/quality/set-v2.json is committed, make that the default (test_phase.sh passes no --set);
    # keep it a plain string literal: spec_lint.py reads it from this line
    ap.add_argument("--set", default="bench/quality/set.json")
    ap.add_argument("--fill", action="store_true", help="generate the missing continuations of the set with the stock arm first (see above)")
    ap.add_argument("--dry-run", action="store_true", help="check the spec and the set, print the plan; no engine")
    ap.add_argument("--max-load", type=float, default=0.0)
    ap.add_argument("--ear", action="store_true", help="A56: also report EAR (SLQ) and the same-top-token rate of every arm against stock 1; "
                    "scores with the top-10 ids per position (--topk 10) so EAR has SLQ's K")
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    arms = list(spec["arms"])
    base = str(ROOT / spec["base"]) if not os.path.isabs(spec["base"]) else spec["base"]
    out = Path(a.out)
    set_path = str(ROOT / a.set) if not os.path.isabs(a.set) else a.set
    st = json.load(open(set_path))
    if a.dry_run:
        return dry_run(spec, a, st, set_path)
    probs = set_problems(st)
    if probs:
        print("\n".join("ERROR: " + p for p in probs))
        return 2
    if pending(st) and not a.fill:
        print(f"{len(pending(st))} requests of {a.set} have no continuation: run once with --fill (the stock arm generates them), then commit <out>/set.filled.json")
        return 2
    out.mkdir(parents=True, exist_ok=True)
    abtest.stop_servers()
    if pending(st):
        stock_cfg = str(out / f"{arms[0]}.json")
        abtest.make_config(base, spec["arms"][arms[0]], stock_cfg)
        abtest.quiet_gate(a.max_load, 50.0, [])
        print(f"generating {len(pending(st))} continuations with {arms[0]} ...", flush=True)
        st = fill(st, stock_cfg, out)
        if st is None:
            print("generation FAILED")
            return 2
        set_path = str(out / "set.filled.json")
        print(f"complete set: {set_path} ({set_summary(st)}); commit it as the set", flush=True)
    tsv = {}
    plan = [(arms[0], "a"), (arms[0], "b")] + [(x, "") for x in arms[1:]]
    for arm, tag in plan:
        abtest.quiet_gate(a.max_load, 50.0, [])
        cfg_path = str(out / f"{arm}.json")
        abtest.make_config(base, spec["arms"][arm], cfg_path)
        path = str(out / f"{arm}{'-' + tag if tag else ''}.tsv")
        print(f"scoring {arm}{' (' + tag + ')' if tag else ''} ...", flush=True)
        if not score(cfg_path, set_path, path, 10 if a.ear else 2):
            print(f"  {arm}: scoring FAILED", flush=True)
            continue
        tsv.setdefault(arm, []).append(path)
    stock = tsv.get(arms[0], [])
    if len(stock) < 2:
        print("stock not scored twice: no verdicts")
        return 2
    verdicts = {}
    for arm in arms[1:]:
        if arm not in tsv:
            verdicts[arm] = "not scored"
            continue
        vj = str(out / f"{arm}.verdict.json")
        r = subprocess.run([PY, "tools/force_check.py", "check", "--stock", *stock, "--cand", tsv[arm][0], "--json", vj] + (["--ear"] if a.ear else []),
                           cwd=ROOT, capture_output=True, text=True)
        verdicts[arm] = {0: "PASS", 1: "REJECT", 2: "INCONCLUSIVE"}.get(r.returncode, f"error {r.returncode}")
        (out / f"{arm}.verdict.txt").write_text(r.stdout)
        print(f"{arm:24s} {verdicts[arm]}\n" + "\n".join("    " + x for x in r.stdout.strip().splitlines()[-6:]), flush=True)
    json.dump(verdicts, open(out / "verdicts.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
