#!/usr/bin/env python3
"""apply_triage.py - apply a bench-sweep.sh triage file to the shipped settings of an updated branch (owner order 2026-11-20).

  apply_triage.py --triage BRANCH-triage.v2.json --branch us|rd --shipped FILE [--preset FILE] [--spec FILE ...] [--readme README-PANCAKE.md] [--dry-run]

  The v2 triage files (schema pancake.sweep-triage/2, the drift-guarded ones: box_chains/sweep.py triage / remark) are read: given BRANCH-triage.json,
  the BRANCH-triage.v2.json beside it is used instead; a triage file without the v2 schema is refused (its verdicts may come from drifted data:
  TEST_LEDGER.md, the 2026-11-20 sweep slices with a late base up to 44 % below the early one).

  us: --shipped terminal/data/tuned-flags.json (profiles[0].flags, top-level "removed"); --preset configs/pancake/flashnext-iq2_xs-updated-hip.json
  rd: --shipped rdna3/recommended-flags/gfx1100.json (env / args / verdicts / preset / removed)
  --spec: bench specs whose updated-family arms (every arm but plain* / base*) follow the shipped settings (us 220-223, rd 250-252).

Rules (owner rules 2026-11-20: flags are judged ONE BY ONE, each from its own arm against its own block base; the merged triage file holds the
sweep slices AND the RETEST slices of bench-retest.sh, same v2 schema):
  BUNDLE a record with "bundle": true (a whole arm or preset, e.g. RD `updated` against `plain`, or a whole-arm quality gate) NEVER changes a flag:
         it is reported only (it can trigger per-flag arms, not remove one).
  NO-GAIN-P1 (pass 1 without a benefit), PROMISING / PROMISING-LOW (stage-1 screen results, not confirmed), INCONCLUSIVE (base drift > 5 %), RETEST, RETEST-GATE, UNTESTED: nothing changes.
  OPTIONAL (clean pass 2 only, else HOLD: a metric >= +10 % and one <= -3 %): the flag leaves the default flags (shipped file, preset, spec
         arms) and is listed in the "optional" section of the shipped file with the cases it helps and hurts, and in README-PANCAKE.md (--readme:
         the block between the markers <!-- optional-flags:begin --> / <!-- optional-flags:end -->), so a user can turn it on for that workload.
  DROP   only with a clean pass 2 (3 reps, base drift <= the file's drift_max, "pass2_clean": true) that is negative or below the bar: then the flag
         is taken out of the shipped file (recorded under "removed" with the pass-1 / pass-2 numbers), out of the preset and out of the
         updated-family arms of the specs.  A PROTECTED flag (the registry's "protected": ECAP judged on edit, the three RDNA kernels on decode) is
         dropped only when its pass 2 exists and is negative on its target case(s) (the mean of them); otherwise it stays, reported as PROTECTED.
         A DROP from pass 1 alone (no pass 2, or a drifted one) never removes a shipped / preset flag: it is reported as HOLD.
  KEEP   (clean pass 2 only, else HOLD) kept; a flag the shipped file does not carry (a plus flag, or a minus flag taken out pending its retest, e.g.
         --lookup-max-t on us) is added to the shipped file, the preset and the updated-family spec arms.  A box path in the arm is written as
         the record's "ship" value (<ROOT> = the branch tree).
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
    """[(flag, action, rec)] with action drop / add / optional / protected / hold / bundle / none.  hold: a DROP / KEEP / OPTIONAL without a
    clean pass 2 (nothing changes).  bundle: a whole-arm record (never changes a flag)."""
    prot = tri.get("protected", {})
    out = []
    for flag, rec in tri["flags"].items():
        v = rec.get("verdict")
        if rec.get("bundle") or flag.startswith("BUNDLE"):
            out.append((flag, "bundle", rec))
        elif v == "OPTIONAL":
            out.append((flag, "optional" if clean_pass2(rec) else "hold", rec))
        elif v == "DROP":
            if not clean_pass2(rec):
                out.append((flag, "hold", rec))
            else:
                out.append((flag, "protected" if protected_stays(flag, rec, prot) else "drop", rec))
        elif v == "KEEP":
            out.append((flag, "add" if clean_pass2(rec) else "hold", rec))    # add: only what the shipped file does not carry yet
        else:
            out.append((flag, "none", rec))          # NO-GAIN-P1, PROMISING(-LOW), RETEST, RETEST-GATE, INCONCLUSIVE, UNTESTED, REFERENCE
    return out


def v2_path(path):
    """BRANCH-triage.json -> BRANCH-triage.v2.json when that exists (the drift-guarded copy); a .v2.json path stays."""
    if path.endswith(".v2.json"):
        return path
    cand = path[:-5] + ".v2.json" if path.endswith(".json") else path + ".v2.json"
    return cand if os.path.exists(cand) else path


def keys_of(rec):
    return list((rec.get("env") or {}).keys()), list((rec.get("args") or {}).keys())


def values_of(rec):
    """(env, args) to WRITE: the record's "ship" values (a box path replaced by <ROOT>) over its arm values."""
    ship = rec.get("ship") or {}
    return ({**(rec.get("env") or {}), **(ship.get("env") or {})}, {**(rec.get("args") or {}), **(ship.get("args") or {})})


def opt_entry(flag, rec):
    p2 = rec.get("pass2") or {}
    env, args = values_of(rec)
    return {"flag": flag, "env": env, "args": args, "exact": bool(rec.get("exact", True)),
            "helps": {k: v for k, v in p2.items() if v > 0}, "hurts": {k: v for k, v in p2.items() if v < 0},
            "verdict": "OPTIONAL", "evidence": evidence(rec),
            "note": "opt-in (owner rule 2026-11-20): turn it on for the workload it helps; not in the default flags" +
                    ("" if rec.get("exact", True) else "; lossy: a quality gate is due before it is offered")}


def apply_us_shipped(doc, actions):
    changes = []
    prof = doc["profiles"][0]
    removed = doc.setdefault("removed", [])
    for flag, act, rec in actions:
        env, args = keys_of(rec)
        names = set(env) | set(args)
        if act in ("drop", "optional"):
            gone = [f for f in prof["flags"] if f["name"] in names]
            if gone:
                prof["flags"] = [f for f in prof["flags"] if f["name"] not in names]
                if act == "drop":
                    for f in gone:
                        removed.append({"kind": f["kind"], "name": f["name"], "value": f.get("value"), "verdict": "SWEEP-DROP", "evidence": evidence(rec)})
                changes.append("shipped: %s %s" % ("removed" if act == "drop" else "moved to optional", ", ".join(f["name"] for f in gone)))
            if act == "optional":
                opt = [o for o in doc.setdefault("optional", []) if o.get("flag") != flag]
                opt.append(opt_entry(flag, rec))
                doc["optional"] = opt
                changes.append("shipped: optional %s (helps %s; hurts %s)" % (flag, json.dumps(opt[-1]["helps"]), json.dumps(opt[-1]["hurts"])))
        elif act == "add":
            have = {f["name"] for f in prof["flags"]}
            gain = max((rec.get("pass2") or {}).values() or [0.0])
            env, args = values_of(rec)
            for k, v in list(env.items()) + list(args.items()):
                if k in have:
                    continue
                prof["flags"].append({"kind": "arg" if k.startswith("--") else "env", "name": k, "value": v, "verdict": "GAIN",
                                      "exact": bool(rec.get("exact", True)), "gain_pct": round(gain, 1), "ledger": evidence(rec)})
                changes.append("shipped: added %s=%s" % (k, v))
    return changes


def apply_rd_shipped(doc, actions):
    changes = []
    removed = doc.setdefault("removed", {})
    for flag, act, rec in actions:
        env, args = keys_of(rec)
        if act == "optional":
            gone_e = [k for k in env if k in doc.get("env", {})]
            gone_a = [k for k in args if k in doc.get("args", {})]
            for k in gone_e:
                doc["env"].pop(k)
            for k in gone_a:
                doc["args"].pop(k)
            o = opt_entry(flag, rec)
            doc.setdefault("optional", {})[flag] = o
            changes.append("shipped: optional %s%s (helps %s; hurts %s)" % (flag, " (out of the defaults: %s)" % ", ".join(gone_e + gone_a) if gone_e or gone_a else "",
                                                                          json.dumps(o["helps"]), json.dumps(o["hurts"])))
        elif act == "drop":
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
            ve, va = values_of(rec)
            add_e = {k: v for k, v in ve.items() if doc.get("env", {}).get(k) != v}
            add_a = {k: v for k, v in va.items() if doc.get("args", {}).get(k) != v}
            if not add_e and not add_a:
                continue
            doc.setdefault("env", {}).update(add_e)
            doc.setdefault("args", {}).update(add_a)
            gain = max((rec.get("pass2") or {}).values() or [0.0])
            doc.setdefault("verdicts", {})[flag] = {"env": add_e, "args": add_a, "verdict": "GAIN", "exact": bool(rec.get("exact", True)),
                                                    "gain_pct": round(gain, 2), "note": evidence(rec)}
            changes.append("shipped: added %s" % ", ".join("%s=%s" % kv for kv in list(add_e.items()) + list(add_a.items())))
    return changes


def apply_preset(doc, actions):
    """A server config: env dict, args list."""
    changes = []
    for flag, act, rec in actions:
        env, args = keys_of(rec)
        if act in ("drop", "optional"):
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
            ve, va = values_of(rec)
            for k, v in ve.items():
                if doc.setdefault("env", {}).get(k) != v:
                    doc["env"][k] = v
                    changes.append("preset: added env %s=%s" % (k, v))
            a = doc.setdefault("args", [])
            for k, v in va.items():
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
            if act in ("drop", "optional"):
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


README_BEGIN, README_END = "<!-- optional-flags:begin -->", "<!-- optional-flags:end -->"


def readme_block(shipped_doc, branch):
    opt = shipped_doc.get("optional") or ([] if branch == "us" else {})
    items = opt if isinstance(opt, list) else list(opt.values())
    lines = [README_BEGIN, "Optional flags (owner rule 2026-11-20: a big gain on one workload and a loss on another; not in the default flags, turn",
             "them on for the workload they help; generated by tools/pancake/apply_triage.py from the confirm results):", ""]
    if not items:
        lines.append("- none yet")
    for o in items:
        kv = ", ".join(["%s=%s" % x for x in o.get("env", {}).items()] + ["%s %s" % x for x in o.get("args", {}).items()])
        lines.append("- `%s`: helps %s; hurts %s%s" % (kv, ", ".join("%s %+.1f %%" % x for x in o.get("helps", {}).items()) or "-",
                                                     ", ".join("%s %+.1f %%" % x for x in o.get("hurts", {}).items()) or "-",
                                                     "" if o.get("exact", True) else " (lossy: quality gate due)"))
    lines.append(README_END)
    return "\n".join(lines)


def apply_readme(text, block):
    if README_BEGIN in text and README_END in text:
        a, b = text.index(README_BEGIN), text.index(README_END) + len(README_END)
        return text[:a] + block + text[b:]
    return text.rstrip("\n") + "\n\n" + block + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="apply a bench-sweep.sh triage to the shipped settings")
    ap.add_argument("--triage", required=True)
    ap.add_argument("--branch", required=True, choices=["us", "rd"])
    ap.add_argument("--shipped", required=True)
    ap.add_argument("--preset")
    ap.add_argument("--spec", action="append", default=[])
    ap.add_argument("--readme", help="README-PANCAKE.md: the optional-flags block is rewritten from the shipped file")
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
    summary = {"protected": [f for f, act, _ in actions if act == "protected"], "hold": [f for f, act, _ in actions if act == "hold"],
               "bundle": [f for f, act, _ in actions if act == "bundle"], "optional": [f for f, act, _ in actions if act == "optional"], "changes": {}}
    for path, fn in files:
        doc = json.load(open(path, encoding="utf-8"))
        ch = fn(doc, actions)
        summary["changes"][path] = ch
        for c in ch:
            print("%s: %s" % (path, c))
        if ch and not a.dry_run:
            open(path, "w", encoding="utf-8", newline="\n").write(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    if a.readme:
        shipped = json.load(open(a.shipped, encoding="utf-8"))
        txt = open(a.readme, encoding="utf-8").read()
        new = apply_readme(txt, readme_block(shipped, a.branch))
        if new != txt:
            print("%s: optional-flags block rewritten" % a.readme)
            if not a.dry_run:
                open(a.readme, "w", encoding="utf-8", newline="\n").write(new)
    for f in summary["bundle"]:
        print("BUNDLE %s: %s (a whole arm): reported only, changes no flag (flags are judged one by one)" % (f, tri["flags"][f].get("verdict")))
    for f in summary["protected"]:
        print("PROTECTED %s: DROP verdict, but no negative pass 2 on its target case: it stays" % f)
    for f in summary["hold"]:
        print("HOLD %s: %s without a clean pass 2 (3 reps, base drift within the limit): nothing changes" % (f, tri["flags"][f].get("verdict")))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
