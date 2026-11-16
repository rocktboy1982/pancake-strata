#!/usr/bin/env python3
"""stack_select: build one preset stack greedily from confirm winners (MOD_PLAN.md section 6 item 24).

Gains inside one group do not add (MOD_PLAN.md section 2 rule 4), so a stack is grown one arm at a time and an arm stays
only when the stack with it beats the stack without it at the same tier.

    python3 tools/pancake/stack_select.py init STATE.json WINNERS.json --base PRESET.json --exe EXE [--cases short,edit]
    python3 tools/pancake/stack_select.py next STATE.json --out-dir bench/specs     # writes the spec of the next step
    python3 tools/pancake/stack_select.py record STATE.json RUN.stdout              # reads that step's abtest summary

WINNERS.json: a list of {"arm": name, "env": {...}, "args": [...], "tag": "KERNEL|ENGINE|WORKLOAD|MODEL", "gain": pct}
(confirm winners; the best "gain" goes first).  A step's spec has two arms: `stack` (the arms kept so far) and
`stack+<arm>`; both run the same binary.  `record` keeps the arm when the candidate's mean change over the cases is at least
--bar (default +1.0%) and no case is below -1.0%.  An arm whose env key is already set to another value in the stack is
skipped as a conflict.  Stdlib only; STATE.json holds the whole history, so the tool can be stopped between steps.
"""
import argparse
import json
import re
import sys
from pathlib import Path

LINE = re.compile(r"^\s+(\S+)\s+(\S+)\s+[\d.]+ \+-\s*[\d.nan]+\s+\(n=\d+\)\s+vs base:\s+([+-][\d.]+)%")


def merge(stack, arm):
    """The stack's env / args with `arm` added; None on a conflict (same env key, another value)."""
    env = dict(stack.get("env", {}))
    for k, v in (arm.get("env") or {}).items():
        if k in env and env[k] != v:
            return None
        env[k] = v
    have = stack.get("args", [])
    return {"env": env, "args": list(have) + [x for x in (arm.get("args") or []) if x not in have]}


def order(winners):
    return sorted(winners, key=lambda w: -float(w.get("gain", 0)))


def parse_step(text, cand):
    """{case: pct of arm `cand` against the first arm} from an abtest summary."""
    out = {}
    for line in text.splitlines():
        m = LINE.match(line)
        if m and m.group(2) == cand:
            out[m.group(1)] = float(m.group(3))
    return out


def decide(pcts, bar):
    if not pcts:
        return False, "no numbers for the candidate"
    mean = sum(pcts.values()) / len(pcts)
    low = min(pcts.values())
    if low < -1.0:
        return False, "case at %+.1f%%" % low
    return mean >= bar, "mean %+.1f%%" % mean


def save(path, st):
    Path(path).write_text(json.dumps(st, indent=2) + "\n", encoding="utf-8")


def cmd_init(a):
    winners = json.loads(Path(a.winners).read_text(encoding="utf-8"))
    st = {"base": a.base, "exe": a.exe, "cases": a.cases.split(","), "bar": a.bar,
          "queue": [w["arm"] for w in order(winners)], "winners": {w["arm"]: w for w in winners},
          "stack": {"env": {}, "args": []}, "kept": [], "history": [], "pending": None}
    save(a.state, st)
    print("%d winners queued" % len(st["queue"]))
    return 0


def cmd_next(a):
    st = json.loads(Path(a.state).read_text(encoding="utf-8"))
    while st["queue"]:
        name = st["queue"].pop(0)
        merged = merge(st["stack"], st["winners"][name])
        if merged is None:
            st["history"].append({"arm": name, "kept": False, "why": "env conflict"})
            continue
        step = len(st["history"]) + 1
        arms = {"stack": {"exe": st["exe"], **{k: v for k, v in st["stack"].items() if v}},
                "stack+" + name: {"exe": st["exe"], **{k: v for k, v in merged.items() if v}}}
        spec = {"base": st["base"], "lossy": any(st["winners"][k].get("lossy") for k in st["kept"] + [name]),
                "cases": st["cases"], "reps": 5, "arms": arms,
                "note": "stack_select step %d (MOD_PLAN.md section 6 item 24): kept so far %s; candidate %s. Run with --tier "
                        "confirm --block 3 --warmup 1 --outlier robust --reclaim-max 3 --reps 5 --no-early-stop."
                        % (step, st["kept"] or "none", name)}
        path = Path(a.out_dir) / ("stack-%s-step%02d.json" % (Path(a.state).stem, step))
        path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        st["pending"] = {"arm": name, "merged": merged, "spec": str(path)}
        save(a.state, st)
        print(path)
        return 0
    save(a.state, st)
    print("queue empty; stack: %s" % json.dumps(st["stack"]))
    return 0


def cmd_record(a):
    st = json.loads(Path(a.state).read_text(encoding="utf-8"))
    pend = st.get("pending")
    if not pend:
        print("no pending step", file=sys.stderr)
        return 1
    pcts = parse_step(Path(a.stdout).read_text(encoding="utf-8", errors="replace"), "stack+" + pend["arm"])
    keep, why = decide(pcts, st["bar"])
    st["history"].append({"arm": pend["arm"], "kept": keep, "why": why, "pcts": pcts, "spec": pend["spec"]})
    if keep:
        st["stack"] = pend["merged"]
        st["kept"].append(pend["arm"])
    st["pending"] = None
    save(a.state, st)
    print("%s %s (%s)" % (pend["arm"], "KEPT" if keep else "dropped", why))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init")
    i.add_argument("state")
    i.add_argument("winners")
    i.add_argument("--base", required=True)
    i.add_argument("--exe", required=True)
    i.add_argument("--cases", default="short,edit")
    i.add_argument("--bar", type=float, default=1.0)
    n = sub.add_parser("next")
    n.add_argument("state")
    n.add_argument("--out-dir", default="bench/specs")
    r = sub.add_parser("record")
    r.add_argument("state")
    r.add_argument("stdout")
    a = ap.parse_args(argv)
    return {"init": cmd_init, "next": cmd_next, "record": cmd_record}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
