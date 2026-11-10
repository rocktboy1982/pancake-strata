"""quality_gate.py - pancake-strata quality gate as a process (MOD_PLAN phase 14): token forcing for every arm of an A/B spec.

Same spec as tools/pancake/abtest.py ({"base": CFG, "arms": {"name": {"env": {...}, "args": {...}}, ...}}).  The first arm
is the stock: it is scored twice (the run-to-run noise); every other arm is scored once and judged against the two
stock arms with tools/force_check.py's rules (rate rule + large-margin event rule, SpecPilot): PASS / REJECT /
INCONCLUSIVE.  The fixed text is bench/quality/set.json (or --set).  An arm whose output is exact by construction (a
placement-only switch) does not need this; lossy ones (route prior, deferral rank guard, depth-graded prior) do.

    python tools/pancake/quality_gate.py SPEC.json --out bench/quality-hip/<day>-<name> [--set bench/quality/set.json]
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abtest  # noqa: E402

ROOT = abtest.ROOT
PY = abtest.PY


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="bench/quality/set.json")
    ap.add_argument("--max-load", type=float, default=0.0)
    ap.add_argument("--ear", action="store_true", help="A56: also report EAR (SLQ) and the same-top-token rate of every arm against stock 1; "
                    "scores with the top-10 ids per position (--topk 10) so EAR has SLQ's K")
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    arms = list(spec["arms"])
    base = str(ROOT / spec["base"]) if not os.path.isabs(spec["base"]) else spec["base"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    set_path = str(ROOT / a.set) if not os.path.isabs(a.set) else a.set
    abtest.stop_servers()
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
