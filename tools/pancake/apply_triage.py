#!/usr/bin/env python3
"""apply_triage.py - apply a bench-sweep.sh triage file to the shipped settings of an updated branch (owner order 2026-11-20).

  apply_triage.py --triage BRANCH-triage.v2.json --branch us|rd --shipped FILE [--preset FILE] [--spec FILE ...] [--dry-run]

  The v2 triage files (schema pancake.sweep-triage/2, the drift-guarded ones: box_chains/sweep.py triage / remark) are read: given BRANCH-triage.json,
  the BRANCH-triage.v2.json beside it is used instead; a triage file without the v2 schema is refused (its verdicts may come from drifted data:
  TEST_LEDGER.md, the 2026-11-20 sweep slices with a late base up to 44 % below the early one).

  us: --shipped terminal/data/tuned-flags.json (profiles[0].flags, top-level "removed"); --preset configs/pancake/flashnext-iq2_xs-updated-hip.json
  rd: --shipped rdna3/recommended-flags/gfx1100.json (env / args / verdicts / preset / removed)
  --spec: bench specs whose updated-family arms (every arm but plain* / base*) follow the shipped settings (us 220-223, rd 250-252).

Rules:
  DROP   only with a clean pass 2 (3 reps, base drift <= the file's drift_max, "pass2_clean": true) that is negative or below the bar: then the flag
         is taken out of the shipped file (recorded under "removed" with the pass-1 / pass-2 numbers), out of the preset and out of the
         updated-family arms of the specs.  A PROTECTED flag (the registry's "protected": ECAP judged on edit, the three RDNA kernels on decode) is
         dropped only when its pass 2 exists and is negative on its target case(s) (the mean of them); otherwise it stays, reported as PROTECTED.
         A DROP from pass 1 alone (no pass 2, or a drifted one) never removes a shipped / preset flag: it is reported as HOLD.
  KEEP   (clean pass 2 only, else HOLD) kept; a plus flag (a switch the updated arm did not set) is added to the shipped file, the preset and the updated-family spec arms.
  RETEST-GATE, RETEST, INCONCLUSIVE, UNTESTED: nothing changes (a lossy gain needs a quality gate first; an unconfirmed flag is not touched).
A flag the shipped file / preset does not carry (e.g. a setting of Strata-rdna3's own setup on rd) is never written; it is only reported.
Prints one line per change and a JSON summary; --dry-run writes nothing."""
import argparse
import json
import os
import sys

SCHEMA = "pancake.sweep-triage/2"


def evidence(rec):
    return ("bench-sweep.sh triage %s: pass 1 %s%s; %s" % (rec.get("verdict"), json.dumps(rec.get("pass1")),
            ("; pass 2 " + json.dumps(rec["pass2"])) if rec.get("pass2") else "", rec.get("reason", "")))


def protected_stays(flag, rec, protected):
    """True when a protected flag must stay despite a DROP verdict."""
    targets = protected.get(flag)
    if not targets:
        return False
    p2 = rec.get("pass2") or {}
    vals = [p2[t] for t in targets if t in p2]
    return not vals or sum(vals) / len(vals) >= 0.0


def clean_pass2(rec):
    """True only for a pass-2 result (3 reps) whose base drift was within the limit (sweep.py: "pass2_clean")."""
    return bool(rec.get("pass2")) and rec.get("pass2_clean") is True


def plan(tri):
    """[(flag, action, rec)] with action drop / add / protected / hold / none.  hold: a DROP or KEEP without a clean pass 2 (nothing changes)."""
    prot = tri.get("protected", {})
    out = []
    for flag, rec in tri["flags"].items():
        v = rec.get("verdict")
        if v == "DROP":
            if not clean_pass2(rec):
                out.append((flag, "hold", rec))
            else:
                out.append((flag, "protected" if protected_stays(flag, rec, prot) else "drop", rec))
        elif v == "KEEP" and rec.get("kind") == "plus":
            out.append((flag, "add" if clean_pass2(rec) else "hold", rec))
        else:
            out.append((flag, "none", rec))          # RETEST, RETEST-GATE, INCONCLUSIVE, UNTESTED, a KEEP of a minus flag (already shipped)
    return out


def v2_path(path):
    """BRANCH-triage.json -> BRANCH-triage.v2.json when that exists (the drift-guarded copy); a .v2.json path stays."""
    if path.endswith(".v2.json"):
        return path
    cand = path[:-5] + ".v2.json" if path.endswith(".json") else path + ".v2.json"
    return cand if os.path.exists(cand) else path


def keys_of(rec):
    return list((rec.get("env") or {}).keys()), list((rec.get("args") or {}).keys())


def apply_us_shipped(doc, actions):
    changes = []
    prof = doc["profiles"][0]
    removed = doc.setdefault("removed", [])
    for flag, act, rec in actions:
        env, args = keys_of(rec)
        names = set(env) | set(args)
        if act == "drop":
            gone = [f for f in prof["flags"] if f["name"] in names]
            if gone:
                prof["flags"] = [f for f in prof["flags"] if f["name"] not in names]
                for f in gone:
                    removed.append({"kind": f["kind"], "name": f["name"], "value": f.get("value"), "verdict": "SWEEP-DROP", "evidence": evidence(rec)})
                changes.append("shipped: removed %s" % ", ".join(f["name"] for f in gone))
        elif act == "add":
            have = {f["name"] for f in prof["flags"]}
            gain = max((rec.get("pass2") or {}).values() or [0.0])
            for k, v in list((rec.get("env") or {}).items()) + list((rec.get("args") or {}).items()):
                if k in have:
                    continue
                prof["flags"].append({"kind": "arg" if k.startswith("--") else "env", "name": k, "value": v, "verdict": "GAIN", "exact": True,
                                      "gain_pct": round(gain, 1), "ledger": evidence(rec)})
                changes.append("shipped: added %s=%s" % (k, v))
    return changes


def apply_rd_shipped(doc, actions):
    changes = []
    removed = doc.setdefault("removed", {})
    for flag, act, rec in actions:
        env, args = keys_of(rec)
        if act == "drop":
            gone_e = [k for k in env if k in doc.get("env", {})]
            gone_a = [k for k in args if k in doc.get("args", {})]
            if not gone_e and not gone_a:
                continue
            was = {k: doc["env"].pop(k) for k in gone_e}
            was.update({k: doc["args"].pop(k) for k in gone_a})
            for sect in ("verdicts", "preset"):
                for name in list(doc.get(sect, {})):
                    if any(name == k or name.startswith(k + " ") or name.startswith(k + "=") or name.startswith(k + ",") for k in gone_e + gone_a):
                        doc[sect].pop(name)
            removed[flag] = {"was": was, "verdict": "SWEEP-DROP", "evidence": evidence(rec)}
            changes.append("shipped: removed %s" % ", ".join(gone_e + gone_a))
        elif act == "add":
            add_e = {k: v for k, v in (rec.get("env") or {}).items() if doc.get("env", {}).get(k) != v}
            add_a = {k: v for k, v in (rec.get("args") or {}).items() if doc.get("args", {}).get(k) != v}
            if not add_e and not add_a:
                continue
            doc.setdefault("env", {}).update(add_e)
            doc.setdefault("args", {}).update(add_a)
            gain = max((rec.get("pass2") or {}).values() or [0.0])
            doc.setdefault("verdicts", {})[flag] = {"env": add_e, "args": add_a, "verdict": "GAIN", "exact": True, "gain_pct": round(gain, 2),
                                                    "note": evidence(rec)}
            changes.append("shipped: added %s" % ", ".join("%s=%s" % kv for kv in list(add_e.items()) + list(add_a.items())))
    return changes


def apply_preset(doc, actions):
    """A server config: env dict, args list."""
    changes = []
    for flag, act, rec in actions:
        env, args = keys_of(rec)
        if act == "drop":
            for k in env:
                if k in doc.get("env", {}):
                    doc["env"].pop(k)
                    changes.append("preset: removed env %s" % k)
            a = doc.get("args", [])
            for k in args:
                if k in a:
                    i = a.index(k)
                    n = 2 if i + 1 < len(a) and not str(a[i + 1]).startswith("--") else 1
                    del a[i:i + n]
                    changes.append("preset: removed arg %s" % k)
        elif act == "add":
            for k, v in (rec.get("env") or {}).items():
                if doc.setdefault("env", {}).get(k) != v:
                    doc["env"][k] = v
                    changes.append("preset: added env %s=%s" % (k, v))
            a = doc.setdefault("args", [])
            for k, v in (rec.get("args") or {}).items():
                if k not in a:
                    a += [k] + ([v] if v != "" else [])
                    changes.append("preset: added arg %s %s" % (k, v))
    return changes


def apply_spec(doc, actions):
    changes = []
    for name, arm in doc.get("arms", {}).items():
        if name.startswith("plain") or name.startswith("base"):
            continue
        for flag, act, rec in actions:
            env, args = keys_of(rec)
            if act == "drop":
                for sect, ks in (("env", env), ("args", args)):
                    for k in ks:
                        if k in (arm.get(sect) or {}):
                            arm[sect].pop(k)
                            changes.append("spec arm %s: removed %s" % (name, k))
            elif act == "add":
                for sect in ("env", "args"):
                    for k, v in (rec.get(sect) or {}).items():
                        if k not in (arm.get(sect) or {}):
                            arm.setdefault(sect, {})[k] = v
                            changes.append("spec arm %s: added %s" % (name, k))
    return changes


def main(argv=None):
    ap = argparse.ArgumentParser(description="apply a bench-sweep.sh triage to the shipped settings")
    ap.add_argument("--triage", required=True)
    ap.add_argument("--branch", required=True, choices=["us", "rd"])
    ap.add_argument("--shipped", required=True)
    ap.add_argument("--preset")
    ap.add_argument("--spec", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    path = v2_path(a.triage)
    if path != a.triage:
        print("apply_triage: reading %s (the v2, drift-guarded copy of %s)" % (path, a.triage))
    tri = json.load(open(path, encoding="utf-8"))
    if tri.get("schema") != SCHEMA:
        raise SystemExit("apply_triage: %s is not a v2 triage file (schema %r, need %s): its verdicts are not drift-guarded; "
                         "write the v2 copy first (box_chains/sweep.py remark)" % (path, tri.get("schema"), SCHEMA))
    if tri.get("branch") not in (None, a.branch):
        raise SystemExit("apply_triage: the triage is of branch %s, not %s" % (tri.get("branch"), a.branch))
    actions = plan(tri)
    files = [(a.shipped, apply_us_shipped if a.branch == "us" else apply_rd_shipped)]
    if a.preset:
        files.append((a.preset, apply_preset))
    files += [(s, apply_spec) for s in a.spec]
    summary = {"protected": [f for f, act, _ in actions if act == "protected"], "hold": [f for f, act, _ in actions if act == "hold"], "changes": {}}
    for path, fn in files:
        doc = json.load(open(path, encoding="utf-8"))
        ch = fn(doc, actions)
        summary["changes"][path] = ch
        for c in ch:
            print("%s: %s" % (path, c))
        if ch and not a.dry_run:
            open(path, "w", encoding="utf-8", newline="\n").write(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    for f in summary["protected"]:
        print("PROTECTED %s: DROP verdict, but no negative pass 2 on its target case: it stays" % f)
    for f in summary["hold"]:
        print("HOLD %s: %s without a clean pass 2 (3 reps, base drift within the limit): nothing changes" % (f, tri["flags"][f].get("verdict")))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
