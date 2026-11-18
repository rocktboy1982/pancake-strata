#!/usr/bin/env python3
"""chainT-decide.py - decision rule of chain T (owner order 2026-11-18): from the summary of an abtest.py SCREEN run, pick the arms that earn a
confirm run and write the confirm spec.  Stdlib only; reads text, starts nothing.

  python3 chainT-decide.py SCREEN.stdout SCREEN_SPEC.json OUT_CONFIRM_SPEC.json [--confirm-spec SIBLING.json] [--ref ARM] [--bar 2.0]
                           [--lossy-bar 3.0] [--floor -3.0] [--exclude a,b]

Rule: an arm is a screen winner when, against the reference arm (default: the spec's first arm; with a base-late arm the reference mean
is taken over both), its mean is >= bar % better on at least one case and not worse than floor % on any case, AND its generated text is identical
to base (abtest's "generated text vs base" block) -- or the spec is lossy / the text differs, then the bar is lossy-bar and the arm is flagged
"quality gate due".  A spec that is not lossy whose arm shows DIFFERENT text is held to the lossy bar and flagged (a bug, or the short case's own run-to-run text noise).
Output: one line "CONFIRM <spec> <arm,arm>" on stdout and OUT written (arms: first arm, the reference, the winners; the arm definitions come from the
sibling confirm spec when it has them, else from the screen spec), or "NONE <reason>" and no file.  Exit code 0 always, 2 on unreadable input.
"""
import argparse
import copy
import json
import re
import sys

LINE = re.compile(r"^\s+(\S+)\s+(\S+)\s+(-?[\d.]+) \+-\s+(-?[\d.]+)\s+\(n=(\d+)\)")
TEXT = re.compile(r"^\s+(\S+)\s+(identical on every case|DIFFERENT text)")
LATE = "base-late"


def parse(text):
    means, texts, in_text = {}, {}, False
    for ln in text.splitlines():
        if ln.startswith("generated text vs base"):
            in_text = True
            continue
        if in_text:
            m = TEXT.match(ln)
            if m:
                texts[m.group(1)] = m.group(2).startswith("identical")
            continue
        m = LINE.match(ln)
        if m and not ln.lstrip().startswith("drafts"):
            means.setdefault(m.group(1), {})[m.group(2)] = float(m.group(3))
    return means, texts


def decide(text, spec, ref=None, bar=2.0, lossy_bar=3.0, floor=-3.0, exclude=()):
    """-> (winners [(arm, {case: pct}, note)], log lines)."""
    means, texts = parse(text)
    arms = list(spec["arms"])
    first = arms[0]
    ref = ref or first
    log = []
    if not means:
        return [], ["no summary lines parsed"]
    cases = [c for c in means]
    winners = []
    for a in arms:
        if a in (first, ref, LATE) or a in exclude:
            continue
        deltas = {}
        for c in cases:
            xs = means[c].get(a)
            rs = [means[c][r] for r in ([ref, LATE] if ref == first else [ref]) if r in means[c]]
            if xs is None or not rs:
                continue
            deltas[c] = xs / (sum(rs) / len(rs)) * 100 - 100
        if not deltas:
            log.append(f"{a}: no clean runs, not judged")
            continue
        exact_text = texts.get(a, True)
        lossy = bool(spec.get("lossy")) or not exact_text
        need = lossy_bar if lossy else bar
        best, worst = max(deltas.values()), min(deltas.values())
        row = " ".join(f"{c} {d:+.1f}%" for c, d in deltas.items())
        if not spec.get("lossy") and not exact_text:
            row += "  [TEXT DIFFERS in a spec flagged exact: bug or run-to-run noise, held to the lossy bar]"
        if best >= need and worst > floor:
            note = "quality gate due (lossy / text differs)" if lossy else "exact"
            log.append(f"{a}: {row}  -> WIN (bar {need:g}%), {note}")
            winners.append((a, deltas, note))
        else:
            log.append(f"{a}: {row}  -> no (bar {need:g}%, floor {floor:g}%)")
    return winners, log


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stdout")
    ap.add_argument("spec")
    ap.add_argument("out")
    ap.add_argument("--confirm-spec")
    ap.add_argument("--ref")
    ap.add_argument("--bar", type=float, default=2.0)
    ap.add_argument("--lossy-bar", type=float, default=3.0)
    ap.add_argument("--floor", type=float, default=-3.0)
    ap.add_argument("--exclude", default="")
    a = ap.parse_args(argv)
    try:
        text = open(a.stdout, errors="replace").read()
        spec = json.load(open(a.spec))
        sib = json.load(open(a.confirm_spec)) if a.confirm_spec else None
    except (OSError, ValueError) as e:
        print(f"NONE unreadable input: {e}")
        return 2
    winners, log = decide(text, spec, a.ref, a.bar, a.lossy_bar, a.floor, [x for x in a.exclude.split(",") if x])
    for l in log:
        print("  " + l)
    if not winners:
        print("NONE no screen winner")
        return 0
    first = list(spec["arms"])[0]
    keep = [first] + ([a.ref] if a.ref and a.ref != first else []) + [w[0] for w in winners]
    conf = copy.deepcopy(sib if sib else spec)
    conf["arms"] = {k: copy.deepcopy((sib or {}).get("arms", {}).get(k) or spec["arms"][k]) for k in keep}
    if sib:
        conf["lossy"] = bool(sib.get("lossy")) or any("quality" in w[2] for w in winners)
    conf["note"] = (conf.get("note") or "")[:400] + " [chain T: arms cut to the screen winners " + ",".join(w[0] for w in winners) + "]"
    json.dump(conf, open(a.out, "w"), indent=1)
    print("CONFIRM " + (a.confirm_spec or a.spec) + " " + ",".join(keep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
