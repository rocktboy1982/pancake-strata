#!/usr/bin/env python3
"""sweep.py - the full flag sweep of an updated branch (bench-sweep.sh; owner order 2026-11-20: "test all the coded flags once, then triage which to
retest, don't keep low performers").

  sweep.py spec    --branch us|rd --plain PLAIN.json (--arm-from SPEC.json [--arm-name updated] | --updated-config UPDATED.json) --src SRC_TREE --out SPEC1.json
                   [--slice K/N] [--registry sweep_candidates.json]          the pass-1 spec (+ SPEC1.json.meta.json); prints arms and the estimate
                   [--drop-env KEY]... [--set-env KEY=VALUE]...                 (with --updated-config) fix the updated config first; each change is
                                                                                 printed as an ARMFIX line
                   [--split priority|even] [--only-meta META.json] [--exclude-meta META.json] [--tag NAME] [--block-size B]
  sweep.py blocks  --spec SPEC1.json --prefix OUT [--size B] [--seed S]      the pass-1 blocks: OUT-b01.json ... (prints one BLOCK line each)
  sweep.py drift   --rows ROWS.jsonl [--pass2] [--max PCT]                    the base drift of one run (exit 0 clean, 4 drift, 5 no base rows)
  sweep.py triage  --meta SPEC1.json.meta.json --pass1 ROWS1.jsonl... [--pass2 ROWS2.jsonl] --out TRIAGE.json [--drift-max PCT]
  sweep.py remark  --triage OLD.json --pass1 ROWS1.jsonl [--pass2 ROWS2.jsonl] --out NEW.v2.json     the drift guard on a one-run (v1) triage
  sweep.py pass2   --spec SPEC1.json --triage TRIAGE.json --out SPEC2.json [--max N]   the 3-rep confirm spec of the promising flags (exit 3: none)
  sweep.py merge   --out BRANCH-triage.v2.json PART.json...                               one triage file of all slices of a branch (oldest first)
  sweep.py estimate --arms N [--pass2-max P] [--block-size B]

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
Estimate: 150 s per engine start (the 2026-11-19 box runs: ~1.5 min per start of short + edit + long32k, plus the 32K decode).
BLOCKED PASS 1 (2026-11-20, TEST_LEDGER.md: the first sweep slices ran 33-36 arms in ONE abtest run of ~1.5 h with one base at the start and one
at the end; the box slowed down during the run and the late base was up to 44 % below the early one on edit, so every arm late in the run was
judged against the wrong base): pass 1 runs as blocks of at most 6 arms; each block is its own abtest run (base `updated`, the block's arms in a
random order, the base again: `base-late`), so every arm is judged against the two base runs of its own block (<= ~20 min apart).  A block whose
base drift (late vs early) is above 5 % on any case is run once more; still above 5 %: its arms are INCONCLUSIVE (never DROP / KEEP / RETEST).
TRIAGE GUARD (schema pancake.sweep-triage/2): a verdict computed on data with a base drift above 5 % is INCONCLUSIVE (pass 1: the block's late vs
early base; pass 2: the largest deviation of a base rep from the first one).  A pass-1-clean promising flag whose pass 2 drifted stays RETEST.
A flag with a clean pass 2 keeps its pass-2 verdict.  `remark` applies the same guard to a triage file of the old one-run design (RETEST stays
RETEST).  apply_triage.py ignores INCONCLUSIVE and RETEST and removes a shipped flag only on a clean pass-2 result."""
import argparse
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.path.join(HERE, "sweep_candidates.json")
CASES = ["short", "edit", "longdec32k"]
METRICS = ["short", "edit", "longdec32k", "longdec32k-prompt"]
START_S = 150
BLOCK = 6                                 # arms per pass-1 block (owner order 2026-11-20: <= 6)
DRIFT_MAX = 5.0                           # % base drift above which a verdict is INCONCLUSIVE
WEAK = ("INCONCLUSIVE", "UNTESTED")       # a merge never lets these replace a decided verdict
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


def block_sizes(n, size=BLOCK):
    """Balanced block sizes: ceil(n / size) blocks, none larger than size, sizes differ by at most 1."""
    if n <= 0:
        return []
    nb = math.ceil(n / size)
    return [n // nb + (1 if i < n % nb else 0) for i in range(nb)]


def estimate(n_arms, pass2_max=6, block=BLOCK):
    """(pass 1, one block re-run, pass 2) in seconds: every block = its arms + updated + base-late; one re-run of the largest block is budgeted."""
    bs = block_sizes(n_arms, block)
    p1 = (n_arms + 2 * len(bs)) * START_S
    rr = ((max(bs) + 2) * START_S) if bs else 0
    p2 = (min(pass2_max, n_arms) + 1) * 3 * START_S     # 3 starts per arm, the base included
    return p1, rr, p2


def block_specs(spec, size=BLOCK, seed="0"):
    """[(block id, spec, arm order)]: the flag arms of a pass-1 spec in blocks of <= size (spec order), each block's arms shuffled (a seeded
    random order: the same seed gives the same order); every block spec = updated, the block's arms, then abtest's base-late (drift_check)."""
    names = [a for a in spec["arms"] if a != "updated"]
    sizes = block_sizes(len(names), size)
    out, i = [], 0
    for k, n in enumerate(sizes, 1):
        part = names[i:i + n]
        i += n
        bid = "b%02d" % k
        random.Random("%s:%s" % (seed, bid)).shuffle(part)
        arms = {"updated": spec["arms"]["updated"]}
        arms.update({a: spec["arms"][a] for a in part})
        note = "%s; block %s of %d: %d arms, random order, base again at the end" % (spec.get("note", ""), bid, len(sizes), n)
        out.append((bid, {**spec, "arms": arms, "drift_check": True, "note": note.lstrip("; ")}, part))
    return out


def build_spec(branch, plain_path, arm, registry, src, sl="1/1", plain=None, split="priority", only=None, exclude=None, tag=None, block=BLOCK):
    """only / exclude: arm names (e.g. the arms of an earlier slice's meta): keep only those / leave those out, before the slice is cut.
    split="even": the slices split the list evenly (no priority-first slice 1)."""
    if plain is None and os.path.exists(plain_path):
        plain = json.load(open(plain_path, encoding="utf-8"))
    items = flags_of(arm, registry, registry.get("untested", {}).get(branch, []), src, plain)
    items.sort(key=lambda it: priority(it[1]))                 # stable: [arm], then [named], then [code], registry order inside each
    if only is not None:
        items = [it for it in items if it[0] in set(only)]
    if exclude:
        items = [it for it in items if it[0] not in set(exclude)]
    items = take_slice(items, sl, 0 if split == "even" else sum(priority(m) < 2 for _, m, _ in items))
    arms = {"updated": arm}
    meta = {"branch": branch, "slice": tag or sl, "flags": {}, "protected": registry.get("protected", {}),
            "measured": registry.get("measured", {}).get(branch, {}), "design": "blocked", "block_size": block}
    for name, m, a in items:
        arms[name] = a
        meta["flags"][name] = m
    spec = {"base": plain_path, "drift_check": True, "lossy": True, "prefill": True, "cases": CASES, "reps": 1, "arms": arms,
            "note": "bench-sweep.sh pass 1 (%s, slice %s): the updated arm, then one arm per flag (minus-X: taken out, plus-X: added), run as "
                    "blocks of <= %d arms (sweep.py blocks); tools/pancake/box_chains/sweep.py" % (branch, tag or sl, block)}
    return spec, meta


def records(lines):
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue
    return out


def metrics(lines, rep=None):
    """{arm: {metric: mean}} from abtest jsonl lines (rep: only the records of that rep)."""
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
        if rep is not None and rec.get("rep", 0) != rep:
            continue
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


def drift1(lines):
    """Pass 1 (one block): {case: % of the late base run (base-late) against the early one (updated)}; {} without both."""
    m = metrics(lines)
    e, l = m.get("updated", {}), m.get("base-late", {})
    return {c: round(l[c] / e[c] * 100.0 - 100.0, 2) for c in CASES if e.get(c) and l.get(c)}


def drift2(lines):
    """Pass 2 (confirm, A B B A over the reps): {case: the largest deviation (signed, %) of a later base rep from the first one}; {} with < 2 reps."""
    reps = sorted({r.get("rep", 0) for r in records(lines) if r.get("arm") == "updated"})
    if len(reps) < 2:
        return {}
    first = metrics(lines, reps[0]).get("updated", {})
    later = [metrics(lines, k).get("updated", {}) for k in reps[1:]]
    out = {}
    for c in CASES:
        devs = [x[c] / first[c] * 100.0 - 100.0 for x in later if x.get(c) and first.get(c)]
        if devs:
            out[c] = round(max(devs, key=abs), 2)
    return out


def clean(d, mx=DRIFT_MAX):
    """True when the drift is known (base rows on both ends) and no case is beyond mx %."""
    return bool(d) and max(abs(v) for v in d.values()) <= mx


def worst(d):
    return ("%s %+.1f %%" % max(d.items(), key=lambda kv: abs(kv[1]))) if d else "no base rows on both ends"


def _sets(pass1):
    """[(label, lines)]: the lines of one run (one-run design) or the blocks [(label, lines), ...] (oldest first)."""
    if pass1 and isinstance(pass1[0], str):
        return [("p1", pass1)]
    return list(pass1 or [])


def pass2_verdict(rec, f, m2, d2, mx):
    """(verdict, reason) from the pass-2 rows of this arm, or None without them."""
    e2 = effects(m2, rec["arm"], f["kind"])
    rec["pass2"] = e2 or None
    if not e2:
        return None
    if not clean(d2, mx):
        if rec.get("clean1"):
            return "RETEST", "promising; pass 2 base drift %s > %.0f %%: not judged" % (worst(d2), mx)
        return "INCONCLUSIVE", "pass-1 and pass-2 base drift > %.0f %% (pass 2 %s)" % (mx, worst(d2))
    if judge(e2, KEEP[f["exact"]]):
        return ("KEEP" if f["exact"] else "RETEST-GATE",
                "confirmed >= %.0f %%%s (clean pass 2)" % (KEEP[f["exact"]], "" if f["exact"] else " (lossy: a quality gate is due before it ships)"))
    return "DROP", "not confirmed in pass 2 (bar %.0f %%, floor %.0f %%; clean pass 2)" % (KEEP[f["exact"]], FLOOR)


def triage(meta, pass1, pass2_lines=None, drift_max=DRIFT_MAX):
    """pass1: the lines of one run, or [(block label, lines)] of the blocks, oldest first (a block's re-run after its first run: per arm the
    latest clean run is used, else the latest run)."""
    sets = [(lab, metrics(ls), drift1(ls)) for lab, ls in _sets(pass1)]
    m2 = metrics(pass2_lines) if pass2_lines is not None else None
    d2 = drift2(pass2_lines) if pass2_lines is not None else {}
    ok2 = clean(d2, drift_max) if pass2_lines is not None else None
    flags = {}
    for name, f in meta["flags"].items():
        have = [(lab, m, d) for lab, m, d in sets if m.get(name)]
        lab, m1, d1 = ([x for x in have if clean(x[2], drift_max)] or have or [(None, {}, {})])[-1]
        e1 = effects(m1, name, f["kind"])
        ok1 = clean(d1, drift_max)
        promising = judge(e1, PROMISE[f["exact"]])
        rec = {**f, "arm": name, "pass1": e1, "promising": promising and ok1, "pass2": None, "block": lab, "drift1": d1, "clean1": ok1,
               "blocks_run": [x[0] for x in have], "pass2_drift": d2 or None, "pass2_clean": ok2}
        v = pass2_verdict(rec, f, m2, d2, drift_max) if m2 is not None else None
        if not e1:
            v = ("UNTESTED", "no pass-1 rows for this arm")
        elif v is not None:
            pass                                    # a pass-2 verdict (RETEST / INCONCLUSIVE when pass 2 drifted)
        elif not ok1:
            v = ("INCONCLUSIVE", "pass-1 block %s base drift %s > %.0f %% (after its re-run): no verdict" % (lab, worst(d1), drift_max))
        elif not promising:
            v = ("DROP", "low performer in pass 1 (bar %.1f %%, floor %.0f %%; clean block %s)" % (PROMISE[f["exact"]], FLOOR, lab))
        else:
            v = ("RETEST", "promising; pass 2 not run (budget or no rows)" if m2 is None else "promising; no pass-2 rows for this arm")
        rec["verdict"], rec["reason"] = v
        flags[f["flag"]] = rec
    return {"schema": "pancake.sweep-triage/2", "branch": meta["branch"], "slices": [meta["slice"]], "protected": meta.get("protected", {}),
            "drift_max": drift_max, "flags": flags}


def remark(tri, pass1_lines, pass2_lines=None, drift_max=DRIFT_MAX):
    """The drift guard on a triage file of the one-run design (pass 1 = one abtest run, the base once at the start and once at the end): a
    verdict from drifted data becomes INCONCLUSIVE; RETEST stays RETEST; a pass-2 verdict stays only when pass 2 was clean.  The old verdict is
    kept as prior_verdict."""
    d1 = drift1(pass1_lines or [])
    d2 = drift2(pass2_lines) if pass2_lines else {}
    ok1, ok2 = clean(d1, drift_max), clean(d2, drift_max)
    out = {**tri, "schema": "pancake.sweep-triage/2", "drift_max": drift_max, "remarked": "sweep.py remark (one-run design, drift guard)",
           "flags": {}}
    for flag, r in tri["flags"].items():
        v = r.get("verdict")
        r = {**r, "prior_verdict": v, "block": "one-run", "drift1": d1, "clean1": ok1, "pass2_drift": d2 or None,
             "pass2_clean": ok2 if pass2_lines else None}
        if v in ("RETEST", "UNTESTED", "INCONCLUSIVE"):
            pass
        elif r.get("pass2"):
            if not ok2:
                r["verdict"], r["reason"] = "INCONCLUSIVE", "was %s: pass 2 base drift %s > %.0f %% (pass 1 %s)" % (v, worst(d2), drift_max, worst(d1))
        elif not ok1:
            r["verdict"], r["reason"] = "INCONCLUSIVE", "was %s: pass-1 base drift %s > %.0f %% (one run, ORDER EFFECT)" % (v, worst(d1), drift_max)
        r["promising"] = bool(r.get("promising")) and ok1
        out["flags"][flag] = r
    return out


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
    """Oldest first: a later verdict replaces an earlier one, except that INCONCLUSIVE / UNTESTED never replace a decided one (KEEP, DROP,
    RETEST-GATE, RETEST)."""
    out = None
    for p in parts:
        d = json.load(open(p, encoding="utf-8"))
        if out is None:
            out = {**d, "flags": {}, "slices": [], "protected": {}}
        out["protected"].update(d.get("protected", {}))
        for fl, r in d["flags"].items():
            old = out["flags"].get(fl)
            if old is not None and r.get("verdict") in WEAK and old.get("verdict") not in WEAK:
                continue
            out["flags"][fl] = r
        out["slices"] += d.get("slices", [])
    if out is not None:
        out["schema"] = "pancake.sweep-triage/2"
    return out


def read_lines(f):
    return open(f, encoding="utf-8").read().splitlines() if f and os.path.exists(f) else None


def meta_arms(path):
    return list(json.load(open(path, encoding="utf-8"))["flags"])


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
    s.add_argument("--split", choices=["priority", "even"], default="priority")
    s.add_argument("--only-meta", help="only the arms of this (earlier slice's) meta file")
    s.add_argument("--exclude-meta", action="append", default=[], help="leave out the arms of this meta file")
    s.add_argument("--tag", help="the slice name in the meta and the file names (default: --slice)")
    s.add_argument("--block-size", type=int, default=BLOCK)
    b = sub.add_parser("blocks")
    b.add_argument("--spec", required=True)
    b.add_argument("--prefix", required=True)
    b.add_argument("--size", type=int, default=BLOCK)
    b.add_argument("--seed", default="0")
    d = sub.add_parser("drift")
    d.add_argument("--rows", required=True)
    d.add_argument("--pass2", action="store_true")
    d.add_argument("--max", type=float, default=DRIFT_MAX)
    t = sub.add_parser("triage")
    t.add_argument("--meta", required=True)
    t.add_argument("--pass1", required=True, nargs="+", help="one jsonl (one-run design) or the block jsonl files, oldest first")
    t.add_argument("--pass2")
    t.add_argument("--out", required=True)
    t.add_argument("--drift-max", type=float, default=DRIFT_MAX)
    r = sub.add_parser("remark")
    r.add_argument("--triage", required=True)
    r.add_argument("--pass1", required=True)
    r.add_argument("--pass2")
    r.add_argument("--out", required=True)
    r.add_argument("--drift-max", type=float, default=DRIFT_MAX)
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
    e.add_argument("--block-size", type=int, default=BLOCK)
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
        only = meta_arms(a.only_meta) if a.only_meta else None
        excl = [x for m in a.exclude_meta for x in meta_arms(m)]
        spec, meta = build_spec(a.branch, a.plain, arm, json.load(open(a.registry, encoding="utf-8")), a.src, a.slice, split=a.split, only=only,
                                exclude=excl, tag=a.tag, block=a.block_size)
        json.dump(spec, open(a.out, "w"), indent=1)
        json.dump(meta, open(a.out + ".meta.json", "w"), indent=1)
        n = len(meta["flags"])
        p1, rr, p2 = estimate(n, a.pass2_max, a.block_size)
        pr = [priority(f) for f in meta["flags"].values()]
        print("SWEEP %s slice %s: %d flag arms (%d minus, %d plus; %d arm, %d named, %d code) in %d blocks of <= %d (%s), each with updated + base-late; "
              "estimate pass 1 %d s, one block re-run %d s, pass 2 <= %d s, total <= %d s"
              % (a.branch, a.tag or a.slice, n, sum(f["kind"] == "minus" for f in meta["flags"].values()),
                 sum(f["kind"] == "plus" for f in meta["flags"].values()), pr.count(0), pr.count(1), pr.count(2), len(block_sizes(n, a.block_size)),
                 a.block_size, "/".join(map(str, block_sizes(n, a.block_size))), p1, rr, p2, p1 + rr + p2))
        print("ESTIMATE %d %d %d" % (p1, p2, rr))
    elif a.cmd == "blocks":
        spec = json.load(open(a.spec, encoding="utf-8"))
        for bid, sp, order in block_specs(spec, a.size, a.seed):
            path = "%s-%s.json" % (a.prefix, bid)
            json.dump(sp, open(path, "w"), indent=1)
            print("BLOCK %s %d %s order: %s" % (bid, (len(order) + 2) * START_S, path, " ".join(order)))
    elif a.cmd == "drift":
        lines = read_lines(a.rows) or []
        dd = drift2(lines) if a.pass2 else drift1(lines)
        ok = clean(dd, a.max)
        print("DRIFT %s max %s -> %s" % ("  ".join("%s %+.1f%%" % kv for kv in dd.items()) or "no base rows on both ends",
                                          ("%.1f%%" % max(abs(v) for v in dd.values())) if dd else "-",
                                          "CLEAN" if ok else ("DRIFT (> %g%%)" % a.max if dd else "UNKNOWN")))
        return 0 if ok else 4 if dd else 5
    elif a.cmd == "triage":
        meta = json.load(open(a.meta, encoding="utf-8"))
        sets = [(os.path.basename(f).rsplit(".", 1)[0], read_lines(f)) for f in a.pass1]
        sets = [(lab, ls) for lab, ls in sets if ls is not None]
        tri = triage(meta, sets, read_lines(a.pass2) if a.pass2 else None, a.drift_max)
        json.dump(tri, open(a.out, "w"), indent=1)
        for fl, rc in sorted(tri["flags"].items(), key=lambda kv: kv[1]["verdict"]):
            print("%-12s %-36s pass1 %s%s  [%s]" % (rc["verdict"], fl, json.dumps(rc["pass1"]), "  pass2 " + json.dumps(rc["pass2"]) if rc["pass2"] else "",
                                                 rc["reason"]))
    elif a.cmd == "remark":
        tri = remark(json.load(open(a.triage, encoding="utf-8")), read_lines(a.pass1) or [], read_lines(a.pass2) if a.pass2 else None, a.drift_max)
        json.dump(tri, open(a.out, "w"), indent=1)
        cnt = {}
        for rc in tri["flags"].values():
            cnt[rc["verdict"]] = cnt.get(rc["verdict"], 0) + 1
        print("REMARK %s -> %s: %s" % (a.triage, a.out, ", ".join("%s %d" % kv for kv in sorted(cnt.items()))))
    elif a.cmd == "pass2":
        tri = json.load(open(a.triage, encoding="utf-8"))
        spec2, names = pass2_spec(json.load(open(a.spec, encoding="utf-8")), tri, a.max)
        if not names:
            print("PASS2 none: no promising flag")
            return 3
        json.dump(spec2, open(a.out, "w"), indent=1)
        print("PASS2 %d flag(s): %s" % (len(names), ", ".join(names)))
        print("ESTIMATE2 %d" % ((len(names) + 1) * 3 * START_S))
    elif a.cmd == "merge":
        json.dump(merge(a.parts), open(a.out, "w"), indent=1)
    else:
        p1, rr, p2 = estimate(a.arms, a.pass2_max, a.block_size)
        print("ESTIMATE %d %d %d" % (p1, p2, rr))
    return 0


if __name__ == "__main__":
    sys.exit(main())
