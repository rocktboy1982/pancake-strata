#!/usr/bin/env python3
"""promote_confirm: turn the "PROMOTE to confirm" arms of screen runs into confirm specs.

    python3 tools/pancake/promote_confirm.py --out-dir bench/specs --prefix 107-auto-confirm
        --pair RUN.stdout bench/specs/SPEC.json [--pair RUN.stdout bench/specs/SPEC.json ...]

Each --pair pairs an abtest.py screen stdout with the spec it ran.  The arms the screen verdict marked PROMOTE are
copied (exe, env, args) into one confirm spec per (base preset, cases, base arm) group, with the group's base arm, reps 9
and "lossy": true when any copied arm's text differed from base (abtest.py's text line), so quality_gate.py is due
before adoption (AGENTS.md measurement rules).  Prints the specs it wrote; writes nothing when no arm was promoted.
Stdlib only.
"""
import argparse
import json
import re
import sys
from pathlib import Path

PROMOTE = re.compile(r"^\s+(\S+)\s+.*->\s*PROMOTE to confirm\s*$")
DIFFERENT = re.compile(r"^\s+(\S+)\s+DIFFERENT text")


def promoted(stdout_text):
    """(promoted arm names in order, arms whose text differed from base) from one screen stdout."""
    arms, diff, in_verdict = [], set(), False
    for line in stdout_text.splitlines():
        if line.startswith("screen verdict"):
            in_verdict = True
            continue
        m = DIFFERENT.match(line)
        if m:
            diff.add(m.group(1))
        if in_verdict:
            m = PROMOTE.match(line)
            if m:
                arms.append(m.group(1))
    return arms, diff


def build(pairs):
    """pairs: [(stdout_text, spec_dict, spec_name)] -> {group key: spec dict}."""
    groups = {}
    for text, spec, name in pairs:
        arms, diff = promoted(text)
        if not arms:
            continue
        sarms = spec.get("arms", {})
        base_name = next(iter(sarms), "base")
        key = (spec.get("base"), tuple(spec.get("cases", [])), json.dumps(sarms.get(base_name, {}), sort_keys=True))
        g = groups.setdefault(key, {"base": spec.get("base"), "lossy": False, "cases": list(spec.get("cases", [])),
                                    "reps": 9, "arms": {"base": dict(sarms.get(base_name, {}))}, "from": []})
        for a in arms:
            if a not in sarms or a == base_name:
                continue
            label = a if a not in g["arms"] else "%s-%s" % (name.split("-")[0], a)
            g["arms"][label] = dict(sarms[a])
            g["lossy"] = g["lossy"] or a in diff or bool(spec.get("lossy"))
        g["from"].append("%s (%s)" % (name, ", ".join(arms)))
    return groups


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pair", nargs=2, action="append", required=True, metavar=("RUN.stdout", "SPEC.json"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", required=True, help="file name prefix of the written specs (a number per group is added)")
    a = ap.parse_args(argv)
    pairs = []
    for out, spec in a.pair:
        try:
            text = Path(out).read_text(encoding="utf-8", errors="replace")
            pairs.append((text, json.loads(Path(spec).read_text(encoding="utf-8")), Path(spec).stem))
        except (OSError, ValueError) as e:
            print("skip %s %s: %s" % (out, spec, e), file=sys.stderr)
    written = []
    for i, g in enumerate(build(pairs).values(), 1):
        src = g.pop("from")
        g["note"] = ("Confirm of screen PROMOTE arms written by tools/pancake/promote_confirm.py from: %s. Run with --tier "
                     "confirm --block 3 --warmup 1 --outlier robust --reclaim-max 3 --reps 9; bar >= 2%% end to end "
                     "(>= 3%% lossy); 'lossy': true means an arm's text differed from base, so quality_gate.py is due "
                     "before adoption. Stop pancake.service first." % "; ".join(src))
        if len(g["arms"]) < 2:
            continue
        path = Path(a.out_dir) / ("%s-%d.json" % (a.prefix, i))
        path.write_text(json.dumps(g, indent=2) + "\n", encoding="utf-8")
        written.append(str(path))
    print("\n".join(written) if written else "no promoted arms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
