#!/usr/bin/env python3
"""sweep.py - the full flag sweep of an updated branch (bench-sweep.sh; owner order 2026-11-20: "test all the coded flags once, then triage which to
retest, don't keep low performers").

  sweep.py spec    --branch us|rd --plain PLAIN.json (--arm-from SPEC.json [--arm-name updated] | --updated-config UPDATED.json) --src SRC_TREE --out SPEC1.json
                   [--slice K/N] [--registry sweep_candidates.json]          the pass-1 spec (+ SPEC1.json.meta.json); prints arms and the estimate
                   [--drop-env KEY]... [--set-env KEY=VALUE]...                 (with --updated-config) fix the updated config first; each change is
                                                                                 printed as an ARMFIX line
  sweep.py triage  --meta SPEC1.json.meta.json --pass1 ROWS1.jsonl [--pass2 ROWS2.jsonl] --out TRIAGE.json
  sweep.py pass2   --spec SPEC1.json --triage TRIAGE.json --out SPEC2.json [--max N]   the 3-rep confirm spec of the promising flags (exit 3: none)
  sweep.py merge   --out BRANCH-triage.json PART.json...                                  one triage file of all slices of a branch
  sweep.py estimate --arms N [--pass2-max P]

PASS 1: base = the plain config of the branch; arm `updated` = the updated arm (US: spec 220's `updated`; RD: config-updated.json against
config-plain.json, i.e. only what the updated config adds); one arm per NEVER-TESTED coded switch of the branch (sweep_candidates.json
"untested": no gfx1100 ledger row, no 2026-11-19 box verdict; "measured" lists the skipped ones), one engine start each (abtest --tier screen,
1 rep, the base again at the end: --drift-check / `base-late`):
  minus-X  an untested switch the updated arm turns on, taken out (the plain config's value or the engine default; grouped keys together)
  plus-X   an untested switch the updated arm does not set, at its value (the non-default side; @SRC@ = the branch tree)
Cases short, edit (decode tok/s) and longdec32k (prompt and decode tok/s of a cold 32K prompt).  The effect of a flag on a metric is the gain of
having it ON: plus-X against the base, the base against minus-X (the base = the mean of `updated` and `base-late`).
TRIAGE (the owner's two-pass rule): promising = effect >= 1.0 % (exact) / >= 1.5 % (lossy) on a metric and no metric below -3 %; else a low
performer (DROP).  PASS 2: only the promising flags, abtest --tier confirm, 3 reps, the same cases; KEEP = >= 2 % (exact) on a metric and none below
-3 %; a lossy flag at >= 3 % is RETEST-GATE (a quality gate is due before it is shipped); everything else DROP.  A flag without rows is UNTESTED.
PRIORITY ORDER (owner order 2026-11-20): the arms run in this order: first the [arm] switches (source "updated arm"), then the [named] ones
(source "owner list ..."), then the [code] ones; slice 1 of a branch holds ALL [arm] + [named] arms (filled up with [code] arms to the even slice
size when they are fewer), the other slices split the rest evenly.  Every slice is triaged, confirmed (pass 2) and merged on its own, so the
results of slice 1 are usable before the other slices run.
Estimate: 150 s per engine start (the 2026-11-19 box runs: ~1.5 min per start of short + edit + long32k, plus the 32K decode)."""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.path.join(HERE, "sweep_candidates.json")
CASES = ["short", "edit", "longdec32k"]
METRICS = ["short", "edit", "longdec32k", "longdec32k-prompt"]
START_S = 150
PROMISE = {True: 1.0, False: 1.5}        # exact / lossy: pass-1 bar
KEEP = {True: 2.0, False: 3.0}           # exact / lossy: pass-2 bar
FLOOR = -3.0
BASES = ("updated", "base-late")


def argmap(args):
    out, i = {}, 0
    while i < len(args):
        a = str(args[i])
        if a.startswith("--"):
            if i + 1 < len(args) and not str(args[i + 1]).startswith("--"):
                out[a] = str(args[i + 1])
                i += 2
                continue
            out[a] = ""
        i += 1
    return out


def arm_from_configs(plain, upd):
    """The updated config as an abtest arm over the plain one: exe, every env key / argument it adds or changes (None: one it drops)."""
    pe, ue = plain.get("env", {}), upd.get("env", {})
    pa, ua = argmap(plain.get("args", [])), argmap(upd.get("args", []))
    env = {k: v for k, v in ue.items() if pe.get(k) != v}
    env.update({k: None for k in pe if k not in ue})
    args = {k: v for k, v in ua.items() if pa.get(k, object()) != v}
    args.update({k: None for k in pa if k not in ua})
    return {"exe": upd.get("exe"), "env": env, "args": args}


def load_arm(path, name="updated"):
    return json.load(open(path, encoding="utf-8"))["arms"][name]


def flags_of(arm, registry, untested, src, plain=None):
    """[(arm name, meta, arm)]: one arm per never-tested switch of the branch (registry "untested"): minus-X when the updated arm sets it (taken
    out, its group with it), plus-X otherwise (its value, @SRC@ = the branch tree).  Already-measured switches get no arm."""
    groups = registry.get("groups", {})
    base_env, base_args = arm.get("env") or {}, arm.get("args") or {}
    out = []
    for c in untested:
        env = {k: str(v).replace("@SRC@", src) for k, v in (c.get("env") or {}).items()}
        args = {k: str(v).replace("@SRC@", src) for k, v in (c.get("args") or {}).items()}
        on_e = [k for k in env if base_env.get(k) is not None]
        on_a = [k for k in args if base_args.get(k) is not None]
        nm = c["name"].replace("STRATA_", "").lstrip("-")
        if on_e or on_a:
            keys = [x for k in on_e for x in groups.get(k, [k]) if x in base_env]
            a = {**arm, "env": {x: y for x, y in base_env.items() if x not in keys}, "args": {x: y for x, y in base_args.items() if x not in on_a}}
            out.append(("minus-" + nm, {"flag": c["name"], "kind": "minus", "keys": keys + on_a, "exact": c.get("exact", True),
                                        "env": {x: base_env[x] for x in keys}, "args": {x: base_args[x] for x in on_a},
                                        "why": c.get("why", ""), "source": c.get("source", "")}, a))
        else:
            pe, pa = (plain or {}).get("env", {}), argmap((plain or {}).get("args", []))
            if all(pe.get(k) == v for k, v in env.items()) and all(pa.get(k) == v for k, v in args.items()) and (env or args):
                continue                               # the plain config (theirs) already runs this value: the arm would equal updated
            a = {**arm, "env": {**base_env, **env}, "args": {**base_args, **args}}
            out.append(("plus-" + nm, {"flag": c["name"], "kind": "plus", "keys": list(env) + list(args), "exact": c.get("exact", True),
                                       "env": env, "args": args, "why": c.get("why", ""), "source": c.get("source", "")}, a))
    return out


def priority(meta):
    """0 = [arm] (set by the updated arm), 1 = [named] (owner list), 2 = [code] (any other coded switch)."""
    src = meta.get("source", "")
    return 0 if src == "updated arm" else 1 if src.startswith("owner list") else 2


def take_slice(items, sl, first=0):
    """Part K of N.  The first `first` items (the priority ones) all go to slice 1, which is at least the even size ceil(len / N); the other
    slices split the rest evenly."""
    k, n = (int(x) for x in sl.split("/"))
    if not 1 <= k <= n:
        raise SystemExit("sweep: --slice K/N needs 1 <= K <= N")
    if n == 1:
        return list(items)
    s1 = max(first, math.ceil(len(items) / n))
    if k == 1:
        return items[:s1]
    rest = items[s1:]
    size = math.ceil(len(rest) / (n - 1))
    return rest[(k - 2) * size:(k - 1) * size]


def fix_config(upd, drop=(), sets=()):
    """The updated config with the keys in `drop` taken out of its env and KEY=VALUE of `sets` put in; returns (config, ARMFIX lines)."""
    upd = json.loads(json.dumps(upd))
    env, notes = upd.setdefault("env", {}), []
    for k in drop:
        if k in env:
            notes.append("ARMFIX dropped %s=%s from the updated config" % (k, env.pop(k)))
        else:
            notes.append("ARMFIX %s is not in the updated config: nothing dropped" % k)
    for kv in sets:
        k, _, v = kv.partition("=")
        if env.get(k) == v:
            notes.append("ARMFIX %s=%s already in the updated config" % (k, v))
        else:
            notes.append("ARMFIX set %s=%s in the updated config (was %s)" % (k, v, env.get(k, "unset")))
            env[k] = v
    return upd, notes


def estimate(n_arms, pass2_max=6):
    p1 = (n_arms + 2) * START_S                         # + updated + base-late
    p2 = (min(pass2_max, n_arms) + 1) * 3 * START_S     # 3 starts per arm, the base included
    return p1, p2


def build_spec(branch, plain_path, arm, registry, src, sl="1/1", plain=None):
    if plain is None and os.path.exists(plain_path):
        plain = json.load(open(plain_path, encoding="utf-8"))
    items = flags_of(arm, registry, registry.get("untested", {}).get(branch, []), src, plain)
    items.sort(key=lambda it: priority(it[1]))                 # stable: [arm], then [named], then [code], registry order inside each
    items = take_slice(items, sl, sum(priority(m) < 2 for _, m, _ in items))
    arms = {"updated": arm}
    meta = {"branch": branch, "slice": sl, "flags": {}, "protected": registry.get("protected", {}),
            "measured": registry.get("measured", {}).get(branch, {})}
    for name, m, a in items:
        arms[name] = a
        meta["flags"][name] = m
    spec = {"base": plain_path, "drift_check": True, "lossy": True, "prefill": True, "cases": CASES, "reps": 1, "arms": arms,
            "note": "bench-sweep.sh pass 1 (%s, slice %s): the updated arm, then one arm per flag (minus-X: taken out, plus-X: added); "
                    "tools/pancake/box_chains/sweep.py" % (branch, sl)}
    return spec, meta


def metrics(lines):
    """{arm: {metric: mean}} from abtest jsonl lines."""
    acc = {}
    for ln in lines:
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            rec = json.loads(ln)
        except ValueError:
            continue
        arm = rec.get("arm")
        for r in rec.get("rows") or []:
            c = r.get("case")
            if not arm or c not in CASES:
                continue
            if r.get("decode_tps"):
                acc.setdefault(arm, {}).setdefault(c, []).append(float(r["decode_tps"]))
            if c.startswith("longdec") and r.get("prompt_tps"):
                acc.setdefault(arm, {}).setdefault(c + "-prompt", []).append(float(r["prompt_tps"]))
    return {a: {m: sum(v) / len(v) for m, v in ms.items()} for a, ms in acc.items()}


def effects(m, arm, kind):
    """{metric: % gain of having the flag ON} or {} without data."""
    base = {}
    for met in METRICS:
        vals = [m[b][met] for b in BASES if met in m.get(b, {})]
        if vals:
            base[met] = sum(vals) / len(vals)
    got = m.get(arm, {})
    out = {}
    for met, b in base.items():
        x = got.get(met)
        if not x or not b:
            continue
        out[met] = round(((x / b) if kind == "plus" else (b / x)) * 100.0 - 100.0, 2)
    return out


def judge(eff, bar):
    return bool(eff) and max(eff.values()) >= bar and min(eff.values()) >= FLOOR


def triage(meta, pass1_lines, pass2_lines=None):
    m1 = metrics(pass1_lines)
    m2 = metrics(pass2_lines) if pass2_lines is not None else None
    flags = {}
    for name, f in meta["flags"].items():
        e1 = effects(m1, name, f["kind"])
        promising = judge(e1, PROMISE[f["exact"]])
        rec = {**f, "arm": name, "pass1": e1, "promising": promising, "pass2": None}
        if not e1:
            rec["verdict"], rec["reason"] = "UNTESTED", "no pass-1 rows for this arm"
        elif not promising:
            rec["verdict"], rec["reason"] = "DROP", "low performer in pass 1 (bar %.1f %%, floor %.0f %%)" % (PROMISE[f["exact"]], FLOOR)
        elif m2 is None:
            rec["verdict"], rec["reason"] = "RETEST", "promising; pass 2 not run (budget or no rows)"
        else:
            e2 = effects(m2, name, f["kind"])
            rec["pass2"] = e2
            if not e2:
                rec["verdict"], rec["reason"] = "RETEST", "promising; no pass-2 rows for this arm"
            elif judge(e2, KEEP[f["exact"]]):
                rec["verdict"] = "KEEP" if f["exact"] else "RETEST-GATE"
                rec["reason"] = "confirmed >= %.0f %%%s" % (KEEP[f["exact"]], "" if f["exact"] else " (lossy: a quality gate is due before it ships)")
            else:
                rec["verdict"], rec["reason"] = "DROP", "not confirmed in pass 2 (bar %.0f %%, floor %.0f %%)" % (KEEP[f["exact"]], FLOOR)
        flags[f["flag"]] = rec
    return {"schema": "pancake.sweep-triage/1", "branch": meta["branch"], "slices": [meta["slice"]], "protected": meta.get("protected", {}),
            "flags": flags}


def pass2_spec(spec, tri, max_flags=6):
    prom = [r for r in tri["flags"].values() if r.get("promising")]
    prom.sort(key=lambda r: -max(r["pass1"].values()))
    prom = prom[:max_flags]
    arms = {"updated": spec["arms"]["updated"]}
    for r in prom:
        arms[r["arm"]] = spec["arms"][r["arm"]]
    return {**spec, "arms": arms, "drift_check": False, "reps": 3,
            "note": "bench-sweep.sh pass 2: the promising flags of pass 1 (%d, the strongest first, at most %d), 3 reps against updated" % (len(prom), max_flags)}, \
        [r["flag"] for r in prom]


def merge(parts):
    out = None
    for p in parts:
        d = json.load(open(p, encoding="utf-8"))
        if out is None:
            out = {**d, "flags": {}, "slices": []}
        out["flags"].update(d["flags"])
        out["slices"] += d.get("slices", [])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="bench-sweep.sh helper")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("spec")
    s.add_argument("--branch", required=True, choices=["us", "rd"])
    s.add_argument("--plain", required=True)
    s.add_argument("--arm-from")
    s.add_argument("--arm-name", default="updated")
    s.add_argument("--updated-config")
    s.add_argument("--src", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--slice", default="1/1")
    s.add_argument("--registry", default=REGISTRY)
    s.add_argument("--pass2-max", type=int, default=6)
    s.add_argument("--drop-env", action="append", default=[])
    s.add_argument("--set-env", action="append", default=[])
    t = sub.add_parser("triage")
    t.add_argument("--meta", required=True)
    t.add_argument("--pass1", required=True)
    t.add_argument("--pass2")
    t.add_argument("--out", required=True)
    p = sub.add_parser("pass2")
    p.add_argument("--spec", required=True)
    p.add_argument("--triage", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max", type=int, default=6)
    g = sub.add_parser("merge")
    g.add_argument("--out", required=True)
    g.add_argument("parts", nargs="+")
    e = sub.add_parser("estimate")
    e.add_argument("--arms", type=int, required=True)
    e.add_argument("--pass2-max", type=int, default=6)
    a = ap.parse_args(argv)
    if a.cmd == "spec":
        plain = json.load(open(a.plain, encoding="utf-8"))
        if a.arm_from:
            arm = load_arm(a.arm_from, a.arm_name)
        elif a.updated_config:
            upd, notes = fix_config(json.load(open(a.updated_config, encoding="utf-8")), a.drop_env, a.set_env)
            for ln in notes:
                print(ln)
            arm = arm_from_configs(plain, upd)
        else:
            raise SystemExit("sweep spec: --arm-from or --updated-config")
        spec, meta = build_spec(a.branch, a.plain, arm, json.load(open(a.registry, encoding="utf-8")), a.src, a.slice)
        json.dump(spec, open(a.out, "w"), indent=1)
        json.dump(meta, open(a.out + ".meta.json", "w"), indent=1)
        p1, p2 = estimate(len(meta["flags"]), a.pass2_max)
        pr = [priority(f) for f in meta["flags"].values()]
        print("SWEEP %s slice %s: %d flag arms (%d minus, %d plus; %d arm, %d named, %d code) + updated + base-late; estimate pass 1 %d s, pass 2 <= %d s, total <= %d s"
              % (a.branch, a.slice, len(meta["flags"]), sum(f["kind"] == "minus" for f in meta["flags"].values()),
                 sum(f["kind"] == "plus" for f in meta["flags"].values()), pr.count(0), pr.count(1), pr.count(2), p1, p2, p1 + p2))
        print("ESTIMATE %d %d" % (p1, p2))
    elif a.cmd == "triage":
        meta = json.load(open(a.meta, encoding="utf-8"))
        rd = lambda f: open(f, encoding="utf-8").read().splitlines() if f and os.path.exists(f) else None
        tri = triage(meta, rd(a.pass1) or [], rd(a.pass2) if a.pass2 else None)
        json.dump(tri, open(a.out, "w"), indent=1)
        for fl, r in sorted(tri["flags"].items(), key=lambda kv: kv[1]["verdict"]):
            print("%-12s %-36s pass1 %s%s" % (r["verdict"], fl, json.dumps(r["pass1"]), "  pass2 " + json.dumps(r["pass2"]) if r["pass2"] else ""))
    elif a.cmd == "pass2":
        tri = json.load(open(a.triage, encoding="utf-8"))
        spec2, names = pass2_spec(json.load(open(a.spec, encoding="utf-8")), tri, a.max)
        if not names:
            print("PASS2 none: no promising flag")
            return 3
        json.dump(spec2, open(a.out, "w"), indent=1)
        print("PASS2 %d flag(s): %s" % (len(names), ", ".join(names)))
    elif a.cmd == "merge":
        json.dump(merge(a.parts), open(a.out, "w"), indent=1)
    else:
        p1, p2 = estimate(a.arms, a.pass2_max)
        print("ESTIMATE %d %d" % (p1, p2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
