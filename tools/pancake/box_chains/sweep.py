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
  sweep.py pass2   --spec SPEC1.json --triage TRIAGE.json --out SPEC2.json [--max N] [--carry CARRY.json] [--tag T]   the 3-rep confirm spec of
                   the promising flags, ALL of them by default (exit 3: none); --max N (the budget) runs the N strongest, the rest goes to CARRY.json
  sweep.py retest  --groups retest_groups.json --slice NAME --out-dir DIR [--rd-plain P --rd-updated U --rd-src D --us-plain P --us-arm-spec S
                   --us-exact-spec S --us-src D] [--carry C.json...] [--drop-env K]... [--set-env K=V]...   the blocks of a RETEST slice
  sweep.py retest-triage --blocks BLOCKS.json --run BID=ROWS.jsonl... --out-dir DIR   per-flag verdicts of a RETEST slice (one v2 file per
                   branch: DIR/<branch>-triage-rt-<slice>.v2.json)
  sweep.py final-specs --triage us-triage.v2.json --us-arm-spec S220 --us-exact-spec S222 --out-dir DIR [--shipped tuned-flags.json] [--reps N --prefix P --no-triage]   the FINAL
                   COMPARISON of the US-only queue: stock upstream against updated-strata with the shipped flags AFTER apply_triage (final-short.json,
                   final-long.json, final-gate.json, final-plan.txt; ESTIMATE line)
  sweep.py paired-drift --rows ROWS.jsonl --base NAME --arm NAME [--cases a,b]   the base drift of one paired block (exit 0 clean, 4 drift, 5 none)
  sweep.py carry   --spec SPEC1.json --meta META --triage TRI.v2.json --tag T --out CARRY.json   the carried flags of a slice (with their arms)
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
TRIAGE (the owner's two-pass rule, restated 2026-11-20): an arm goes to pass 2 when ANY metric shows a benefit >= 1.0 % (exact) / >= 1.5 %
(lossy); a metric below -3 % does NOT block pass 2 (it is reported; the confirm decides).  An arm without such a benefit in a clean block is
NO-GAIN-P1 ("dropped after pass 1: no benefit observed (n=1, not a refutation)"): it changes no shipped flag.  DROP exists ONLY as a pass-2
verdict (3 reps) from a clean run (base drift <= 5 %).  PASS 2: ALL promising flags (no cap; when the budget is short the rest is written to a
carry file and runs in a follow-up RETEST slice, bench-retest.sh), abtest --tier confirm, 3 reps, the same cases:
  KEEP        >= 2 % (exact) on a metric and none below -3 %;  RETEST-GATE  a lossy flag at >= 3 % and none below -3 % (a quality gate is due);
  OPTIONAL    any metric >= +10 % AND any metric <= -3 % (owner rule 2026-11-20): kept as an opt-in, not in the default flags, listed with the
              workload it helps and the one it hurts (apply_triage.py: the "optional" section of the shipped file and README-PANCAKE.md);
  DROP        everything else.  A flag without rows is UNTESTED.
PER FLAG (owner rule 2026-11-20): every verdict is of ONE flag, from its own arm against its own block's base.  A bundle arm (a whole preset,
e.g. RD `updated` against `plain`) is measured and recorded with "bundle": true, but never changes a flag (apply_triage.py skips it); it can only
trigger per-flag arms.
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
WARM-UP + DRIFT ESTIMATOR (T-1004, owner-delegated 2026-11-20): every block (pass-1 block, paired confirm block) starts with ONE extra engine start of the
base, arm `warmup` (the base config; run first; never counted, never judged: it brings the box, the page cache and kcompactd to a steady state;
in the first real block the first start ran edit at 158.5 tok/s and every later one at 137-140).  The drift is judged on the counted base runs
only, with the LAST request of each case in a start (the second request of short / edit, the single one of longdec32k): it varies 0.3 % (short) /
0.5 % (edit) from start to start, the first request 3-6 %.  The effects (the flag against the base) still use the mean of both requests.  The 5 %
limit, the one re-run and INCONCLUSIVE after it are unchanged.
TRIAGE GUARD (schema pancake.sweep-triage/2): a verdict computed on data with a base drift above 5 % is INCONCLUSIVE (pass 1: the block's late vs
early base; pass 2: the largest deviation of a base rep from the first one), also when pass 1 was clean.  An INCONCLUSIVE or RETEST flag with
promising numbers is marked "carry": true: the follow-up RETEST slice (bench-retest.sh, slice carry-<branch>) runs it again.  A flag with a
clean pass 2 keeps its pass-2 verdict.  `remark` applies the same guard to a triage file of the old one-run design (RETEST stays RETEST, a pass-1
DROP becomes NO-GAIN-P1); `merge` relabels any DROP without a clean pass 2 as NO-GAIN-P1.  apply_triage.py ignores NO-GAIN-P1, INCONCLUSIVE,
RETEST and bundles and changes a shipped flag only on a clean pass-2 result of that flag's own arm."""
import argparse
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.path.join(HERE, "sweep_candidates.json")
CASE_S = {"short": 30, "edit": 40, "longdec10k": 30, "longdec32k": 50, "longdec64k": 90, "longdec100k": 140, "long10k": 20, "long32k": 45,
          "long64k": 70, "long100k": 110}
START_OVH = 30                             # per engine start (load, warm-up); short + edit + longdec32k = 150 s
CASES = os.environ.get("SWEEP_CASES", "short,edit,longdec32k").split(",")   # the US queue (owner 2026-11-20): SWEEP_CASES=short,edit,longdec64k (190 s a start)
METRICS = ["short", "edit", "longdec32k", "longdec32k-prompt", "longdec64k", "longdec64k-prompt"]
START_S = START_OVH + sum(CASE_S.get(c, 60) for c in CASES)
BLOCK = 6                                 # arms per pass-1 block (owner order 2026-11-20: <= 6)
DRIFT_MAX = 5.0                           # % base drift above which a verdict is INCONCLUSIVE
WEAK = ("INCONCLUSIVE", "UNTESTED", "NO-GAIN-P1", "PROMISING", "PROMISING-LOW", "HURTS")   # a merge never lets these replace a decided verdict (PROMISING = a stage-1 screen result: any later stage-2 verdict replaces it)
NOGAIN = "NO-GAIN-P1"                     # pass 1, clean block, no benefit: dropped after pass 1 (n=1, not a refutation); no shipped change
OPT_GAIN, OPT_LOSS = 10.0, -3.0           # OPTIONAL: any metric >= +10 % AND any metric <= -3 % after the confirm (owner rule 2026-11-20)
PROMISE = {True: 1.0, False: 1.5}        # exact / lossy: pass-1 bar
HIGH_GAIN = 2.0                          # owner 2026-11-20: ANY case >= +2 % in run 1 (the screen) = HIGH (rescreened); 0 < best < 2 % = LOW (backlog); all cases <= 0: NO-GAIN (plus arm) / HURTS (minus arm, rescreened)
KEEP = {True: 2.0, False: 3.0}           # exact / lossy: pass-2 bar
FLOOR = -3.0
BASES = ("updated", "base-late")
MID = "base-m"                            # T-1006: a base start between the arms of a block (base-m1, base-m2 ...): every arm is judged against the two bases around it
PAIR_MAX = 8.0                            # T-1006: with mid bases a pair of single-start bases may differ up to this much (start-to-start noise, 8 % passes 79 % of drift-free pairs) when both bases are in the same box state
PREFILL_SAME = 12.0                       # T-1006: two base starts are in the same box state when their long-prompt prefill tok/s differ by less than this % (fast state ~2170, slow ~1500-1600 tok/s; the two bases run the same config, so a flag cannot move it; reclaim share was tried and misclassifies)
WARM = "warmup"                          # the extra first engine start of every block (T-1004): base config, not counted, not judged


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


EXTRA_SOURCE = "owner list (stage 1: the named retest flags that are not in the untested registry)"


def extra_items(keys, groups, registry, arm, src, base_path):
    """[(arm name, meta, arm)] for named flags the registry does not list as untested (measured before: --lookup-max-t, GR_RDNA, RDNA_MMVQ_WAVE,
    RDNA_Q6K, HIPBLASLT_TUNING): built by resolve_entry from the retest groups' entries, so the stage-1 screen and the stage-2 confirm run the same arm."""
    out = []
    for key in keys:
        br = key.split(":", 1)[0]
        d = resolve_entry(key, groups["entries"].get(key, {}), {br: {"path": base_path, "arms": {"updated": arm}}}, registry, {br: src})
        if d["type"] != "paired" or d.get("bundle"):
            raise SystemExit("sweep spec: --extra %s is not a plain flag entry" % key)
        meta = {"flag": d["flag"], "kind": d["kind"], "keys": list(d["env"]) + list(d["args"]), "exact": d["exact"], "env": d["env"],
                "args": d["args"], "why": d.get("why", ""), "source": EXTRA_SOURCE}
        if d.get("ship"):
            meta["ship"] = d["ship"]
        out.append((d["arm_name"], meta, d["arm"]))
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


def estimate(n_arms, pass2_max=6, block=BLOCK, warm="each", mid=0):
    """(pass 1, one block re-run, pass 2) in seconds: every block = its arms + updated + base-late; one re-run of the largest block is budgeted;
    pass 2 = pass2_max paired blocks of 8 starts (warm-up, base, flag x 3 interleaved, base); every block has one extra warm-up start (T-1004)."""
    bs = block_sizes(n_arms, block)
    p1 = (n_arms + (3 if warm == "each" else 2) * len(bs) + (0 if warm == "each" else 1) + (sum((n - 1) // mid for n in bs) if mid else 0)) * START_S      # warm = first: ONE warm-up start per slice; mid: one more base start after every `mid` arms of a block (T-1006)
    rr = ((max(bs) + 2) * START_S) if bs and warm != "each" else ((max(bs) + 3) * START_S) if bs else 0
    p2 = min(pass2_max, n_arms) * 8 * START_S          # planning: one paired block (8 starts) per flag; pass 2 itself has no cap
    #                                                    (bench-sweep.sh runs blocks while the 2.5 h budget holds, the rest is carried)
    return p1, rr, p2


def block_specs(spec, size=BLOCK, seed="0", warm="each", mid=0):
    """[(block id, spec, arm order)]: the flag arms of a pass-1 spec in blocks of <= size (spec order), each block's arms shuffled (a seeded
    random order: the same seed gives the same order); every block spec = the warm-up start (the base, not counted), updated, the block's arms, then abtest's base-late (drift_check)."""
    names = [a for a in spec["arms"] if a != "updated"]
    sizes = block_sizes(len(names), size)
    out, i = [], 0
    for k, n in enumerate(sizes, 1):
        part = names[i:i + n]
        i += n
        bid = "b%02d" % k
        random.Random("%s:%s" % (seed, bid)).shuffle(part)
        arms = {WARM: spec["arms"]["updated"]} if (warm == "each" or k == 1) else {}     # warm-up start first (T-1004: not counted, not judged); warm = first: only the slice's first block (rows of rdC-1 k01r / k02: a block that follows another one starts steady)
        arms["updated"] = spec["arms"]["updated"]
        for j, a in enumerate(part):
            if mid and j and j % mid == 0:
                arms["%s%d" % (MID, j // mid)] = spec["arms"]["updated"]      # T-1006: a base start after every `mid` arms (judge each arm against the bases around it)
            arms[a] = spec["arms"][a]
        note = "%s; block %s of %d: %s%d arms, random order, base again at the end" % (spec.get("note", ""), bid, len(sizes), "warm-up start, " if WARM in arms else "", n)
        out.append((bid, {**spec, "arms": arms, "drift_check": True, "note": note.lstrip("; ")}, part))
    return out


def build_spec(branch, plain_path, arm, registry, src, sl="1/1", plain=None, split="priority", only=None, exclude=None, tag=None, block=BLOCK, extra=(), source="untested"):
    """only / exclude: arm names (e.g. the arms of an earlier slice's meta): keep only those / leave those out, before the slice is cut.
    split="even": the slices split the list evenly (no priority-first slice 1)."""
    if plain is None and os.path.exists(plain_path):
        plain = json.load(open(plain_path, encoding="utf-8"))
    items = flags_of(arm, registry, registry.get(source, {}).get(branch, []), src, plain)      # source = "backlog": the deferred switches (run-queue-us-backlog.sh)
    have = {m["flag"] for _, m, _ in items}
    items += [it for it in extra if it[1]["flag"] not in have]          # the named flags measured before (stage 1: screened again, one arm each)
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


def metrics(lines, rep=None, cases=None, last=False):
    """{arm: {metric: mean}} from abtest jsonl lines (rep: only the records of that rep).  short / edit / longdecN: decode tok/s (longdecN also
    "-prompt": prompt tok/s); longN (a cold prompt, 32 tokens out): prompt tok/s only ("longN-prompt").
    last=True: only the LAST request of each case in a start counts (the second one of short / edit): the drift estimator (T-1004)."""
    cases = cases or CASES
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
        rows = rec.get("rows") or []
        if last:
            rows = list({r.get("case"): r for r in rows}.values())     # the last row of each case (dict order: the first sighting; value: the last row)
        for r in rows:
            c = r.get("case")
            if not arm or c not in cases:
                continue
            prompt_only = c.startswith("long") and not c.startswith("longdec")
            if r.get("decode_tps") and not prompt_only:
                acc.setdefault(arm, {}).setdefault(c, []).append(float(r["decode_tps"]))
            if c.startswith("long") and r.get("prompt_tps"):
                acc.setdefault(arm, {}).setdefault(c + "-prompt", []).append(float(r["prompt_tps"]))
    return {a: {m: sum(v) / len(v) for m, v in ms.items()} for a, ms in acc.items()}


def effects(m, arm, kind, bases=BASES):
    """{metric: % gain of having the flag ON} or {} without data (bases: the arm names whose mean is the base)."""
    base = {}
    mets = sorted({k for b in bases for k in m.get(b, {})}, key=lambda k: (METRICS.index(k) if k in METRICS else 99, k))
    for met in mets:
        vals = [m[b][met] for b in bases if met in m.get(b, {})]
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
    """The pass-2 (confirm) KEEP rule: >= bar on a metric and none below the floor."""
    return bool(eff) and max(eff.values()) >= bar and min(eff.values()) >= FLOOR


def promising_of(eff, bar):
    """The pass-1 rule (owner 2026-11-20): ANY metric >= bar earns pass 2; a metric below -3 % does not block it (reported, the confirm decides)."""
    return bool(eff) and max(eff.values()) >= bar


def optional_of(eff):
    """OPTIONAL (owner rule 2026-11-20): a big gain on one metric and a loss on another, after the confirm."""
    return bool(eff) and max(eff.values()) >= OPT_GAIN and min(eff.values()) <= OPT_LOSS


def helps_hurts(eff):
    return ({k: v for k, v in eff.items() if v > 0}, {k: v for k, v in eff.items() if v < 0})


def confirm_verdict(e2, exact):
    """(verdict, reason) of a CLEAN confirm of one flag: OPTIONAL (checked first: a metric >= +10 % and one <= -3 %) / KEEP / RETEST-GATE / DROP."""
    if optional_of(e2):
        hp, ht = helps_hurts(e2)
        return "OPTIONAL", ("opt-in: helps %s; hurts %s (a metric >= +%.0f %% and one <= %.0f %%; clean pass 2)%s"
                            % (", ".join("%s %+.1f %%" % kv for kv in hp.items()), ", ".join("%s %+.1f %%" % kv for kv in ht.items()),
                               OPT_GAIN, OPT_LOSS, "" if exact else "; lossy: a quality gate is due before it is offered"))
    if judge(e2, KEEP[exact]):
        return ("KEEP" if exact else "RETEST-GATE",
                "confirmed >= %.0f %%%s (clean pass 2)" % (KEEP[exact], "" if exact else " (lossy: a quality gate is due before it ships)"))
    return "DROP", "not confirmed in pass 2 (bar %.0f %%, floor %.0f %%; clean pass 2)" % (KEEP[exact], FLOOR)


def drift_keys(cases):
    """The metric of each case the drift guard watches: decode tok/s, or prompt tok/s for a longN case (prompt only)."""
    return [c + "-prompt" if c.startswith("long") and not c.startswith("longdec") else c for c in cases]


def drift1(lines, base="updated", cases=None):
    """Pass 1 (one block): {case: % of the late base run (base-late) against the early one (base)}; {} without both.  Drift estimator (T-1004): the
    LAST request of each case in a start (the second one of short / edit: the steady one; the first one wobbles 4-9 % from start to start)."""
    cases = cases or CASES
    m = metrics(lines, cases=cases, last=True)
    e, l = m.get(base, {}), m.get("base-late", {})
    return {c: round(l[c] / e[c] * 100.0 - 100.0, 2) for c in drift_keys(cases) if e.get(c) and l.get(c)}


def drift2(lines, base="updated", cases=None):
    """Pass 2 (confirm, A B B A over the reps): {case: the largest deviation (signed, %) of a later base rep from the first one}; {} with < 2 reps."""
    cases = cases or CASES
    reps = sorted({r.get("rep", 0) for r in records(lines) if r.get("arm") == base})
    if len(reps) < 2:
        return {}
    first = metrics(lines, reps[0], cases, last=True).get(base, {})
    later = [metrics(lines, k, cases, last=True).get(base, {}) for k in reps[1:]]
    out = {}
    for c in drift_keys(cases):
        devs = [x[c] / first[c] * 100.0 - 100.0 for x in later if x.get(c) and first.get(c)]
        if devs:
            out[c] = round(max(devs, key=abs), 2)
    return out


def is_base(name):
    return name in BASES or str(name).startswith(MID)


def _start_metrics(rec, cases, last=False):
    return metrics([json.dumps(rec)], cases=cases, last=last).get(rec.get("arm"), {})


def local_view(lines, arm, kind, mx=DRIFT_MAX, cases=None):
    """T-1006 (drift fix, ledger): one arm judged against the TWO BASE STARTS AROUND IT (updated / base-mN / base-late), not the block ends.  The usS1 screen
    showed the box switch to a slow state after 2-8 starts of a block (a step, not a slope), so the ends of a block are never the same box.
    Returns (effects, drift, ok, info): effects = % gain against the mean of the two bounding bases (all requests); drift = {case: % of the later base against
    the earlier one} on the LAST request of each case (T-1004); ok = the pair agrees within mx % (the owner's 5 % guard) -- or, with mid bases, within PAIR_MAX % when
    both bases are in the same box state by their long-prompt prefill speed (start-to-start noise, not drift).  No base on one side: ({}, {}, False)."""
    cases = cases or CASES
    last_at = {}
    recs = records(lines)
    for i, r in enumerate(recs):
        if r.get("arm") and r["arm"] != WARM:
            last_at[r["arm"]] = i
    if arm not in last_at:
        return {}, {}, False, {}
    at = last_at[arm]
    before = [a for a, i in last_at.items() if i < at and is_base(a)]
    after = [a for a, i in last_at.items() if i > at and is_base(a)]
    if not before or not after:
        if "updated" in last_at and "base-late" in last_at:         # rows not in run order (an old file, a hand-made one): the block ends, as before T-1006
            b, c = "updated", "base-late"
        else:
            return {}, {}, False, {}
    else:
        b = max(before, key=lambda a: last_at[a])
        c = min(after, key=lambda a: last_at[a])
    rb, rc, ra = recs[last_at[b]], recs[last_at[c]], recs[at]
    mb, mc = _start_metrics(rb, cases), _start_metrics(rc, cases)
    base = {k: (mb[k] + mc[k]) / 2.0 for k in mb if k in mc}
    eff = effects({"__b__": base, arm: _start_metrics(ra, cases)}, arm, kind, bases=("__b__",))
    lb, lc = _start_metrics(rb, cases, True), _start_metrics(rc, cases, True)
    drift = {k: round(lc[k] / lb[k] * 100.0 - 100.0, 2) for k in drift_keys(cases) if lb.get(k) and lc.get(k)}
    ok = clean(drift, mx)
    pb = {k: v for k, v in mb.items() if k.endswith("-prompt")}
    pc = {k: v for k, v in mc.items() if k.endswith("-prompt")}
    keys = [k for k in pb if k in pc]
    same = bool(keys) and all(abs(pc[k] / pb[k] * 100.0 - 100.0) <= PREFILL_SAME for k in keys)
    mid = any(str(a).startswith(MID) for a in last_at)
    if not ok and mid and same and clean(drift, PAIR_MAX):
        ok = True
    return eff, drift, ok, {"bases": [b, c], "prefill": [round(pb[k]) for k in keys], "same_state": same}


def local_report(lines, kinds=None, mx=DRIFT_MAX, cases=None):
    """[(arm, bases, drift, ok, states)] of every non-base arm of one run (the DRIFT lines of bench-sweep)."""
    out = []
    seen = []
    for r in records(lines):
        a = r.get("arm")
        if a and a != WARM and not is_base(a) and a not in seen:
            seen.append(a)
    for a in seen:
        kind = "plus" if a.startswith("plus-") else "minus"
        e, d, ok, info = local_view(lines, a, kind, mx, cases)
        out.append((a, info.get("bases"), d, ok, info.get("same_state")))
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
        rec["carry"] = True
        return "INCONCLUSIVE", "pass 2 base drift %s > %.0f %%: no verdict (pass 1 %s); carried to the follow-up RETEST slice" % (
            worst(d2), mx, "clean" if rec.get("clean1") else "drifted too")
    return confirm_verdict(e2, f["exact"])


def two_run(p1, e2, exact):
    """The TWO-RUN rule (owner 2026-11-20, replaces the 3-rep confirm): run 1 = the screen, run 2 = the rescreen.  Effect = the mean of the two runs per case, spread = their difference.
    OPTIONAL: a case >= +10 % in BOTH runs and a case <= -3 % in BOTH runs.  DROP (no gain): no case above 0 in both runs (run 2 does not repeat the benefit, or both <= 0).
    MIXED (nothing ships): the gain repeats but a case lost >= 3 % in both runs without a +10 % case.  KEEP: the benefit repeats in the same direction (both runs > 0 on the case(s) that
    carried it); a lossy flag whose two-run mean on such a case is >= 3 % is RETEST-GATE (the quality gate runs before it ships).  Returns (verdict, reason, mean, spread)."""
    common = [m for m in e2 if m in p1]
    mean = {m: round((p1[m] + e2[m]) / 2.0, 2) for m in common}
    spread = {m: round(abs(p1[m] - e2[m]), 2) for m in common}
    big = [m for m in common if p1[m] >= OPT_GAIN and e2[m] >= OPT_GAIN]
    loss = [m for m in common if p1[m] <= OPT_LOSS and e2[m] <= OPT_LOSS]
    gain = [m for m in common if p1[m] > 0 and e2[m] > 0]
    if common and max(p1[m] for m in common) <= 0:                  # a HURTS flag (run 1: every case <= 0 with the preset flag ON): dropped only when run 2 agrees, else nothing changes
        if max(e2[m] for m in common) <= 0:
            return "DROP", "two-run rule: every case <= 0 in both runs (having the flag ON hurts: runs agree): no gain", mean, spread
        return "UNSTABLE", "two-run rule: run 1 had every case <= 0 but run 2 shows a gain (%s): the runs disagree, nothing changes" % ", ".join("%s %+.2f %%" % (m, e2[m]) for m in common if e2[m] > 0), mean, spread
    fmt = lambda ms: ", ".join("%s %+.2f %% (runs %+.2f / %+.2f)" % (m, mean[m], p1[m], e2[m]) for m in ms)
    if big and loss:
        return "OPTIONAL", "two runs agree: helps %s; hurts %s: opt-in, not default%s" % (fmt(big), fmt(loss), "" if exact else "; lossy: a quality gate is due before it is offered"), mean, spread
    if not gain:
        return "DROP", "two-run rule: no case above 0 in both runs (run 2 does not repeat the benefit, or both <= 0): no gain", mean, spread
    if loss:
        return "MIXED", "two-run rule: gain repeats (%s) but %s lost >= %.0f %% in both runs without a +%.0f %% case: nothing ships" % (fmt(gain), ", ".join(loss), -OPT_LOSS, OPT_GAIN), mean, spread
    if not exact and max(mean[m] for m in gain) >= 3.0:
        return "RETEST-GATE", "two-run rule: the benefit repeats (%s); lossy and >= 3 %%: a quality gate is due before it ships" % fmt(gain), mean, spread
    return "KEEP", "two-run rule: the benefit repeats in the same direction (%s)" % fmt(gain), mean, spread


def triage(meta, pass1, pass2_lines=None, drift_max=DRIFT_MAX, screen_only=False, prior=None):
    """pass1: the lines of one run, or [(block label, lines)] of the blocks, oldest first (a block's re-run after its first run: per arm the
    latest clean run is used, else the latest run).  pass2_lines: {arm: [(label, lines), ...]} = the PAIRED confirm block(s) of each flag
    (owner order 2026-11-20: one flag per confirm block), or the lines of one old multi-flag confirm run (A B B A, kept for old files)."""
    sets = [(lab, metrics(ls), drift1(ls), ls) for lab, ls in _sets(pass1)]
    paired2 = isinstance(pass2_lines, dict)
    legacy = pass2_lines is not None and not paired2
    m2 = metrics(pass2_lines) if legacy else None
    d2 = drift2(pass2_lines) if legacy else {}
    ok2 = clean(d2, drift_max) if legacy else None
    flags = {}
    for name, f in meta["flags"].items():
        have = []
        for lab, m, d, ls in sets:
            if m.get(name):
                e, dl, okl, _ = local_view(ls, name, f["kind"], drift_max)       # T-1006: against the two bases around the arm (the block ends in the old layout)
                have.append((lab, e, dl, okl))
        lab, e1, d1, ok1 = ([x for x in have if x[3]] or have or [(None, {}, {}, False)])[-1]
        promising = promising_of(e1, PROMISE[f["exact"]])
        p1 = pm = None
        if screen_only:                          # owner 2026-11-20: ANY case above 0 % (clean block) is "to be considered": PROMISING; ranked by gain afterwards (sweep.py rank)
            promising = bool(e1) and max(e1.values()) >= HIGH_GAIN     # run 1: HIGH = ANY case >= +2 % (per-case effects), even when another case is negative
            pm = None
            pr_rec = (prior or {}).get(f["flag"]) or {}
            p1 = pr_rec.get("run1") or pr_rec.get("pass1")                                 # run 1 (a drifted run 2 keeps it as "run1" for the next rescreen)
            if p1:                                                                         # the RESCREEN (second 1-run test): the two-run rule decides (two_run)
                pm = max(p1, key=lambda k: p1[k])
                promising = bool(e1) and e1.get(pm, 0) > 0
        rec = {**f, "run1": (p1 if screen_only and pm else None), "run1_best": pm if screen_only else None, "arm": name, "slice": meta.get("slice"), "pass1": e1, "promising": promising and ok1, "pass2": None, "block": lab, "drift1": d1, "clean1": ok1,
               "blocks_run": [x[0] for x in have], "pass2_drift": d2 or None, "pass2_clean": ok2, "carry": False}
        if promising and min(e1.values()) < FLOOR:
            rec["floor_note"] = "a metric below %.0f %% in pass 1 (%s): reported, the confirm decides" % (FLOOR, worst(e1))
        v = pass2_verdict(rec, f, m2, d2, drift_max) if m2 is not None else None
        if paired2 and pass2_lines.get(name):
            v = paired_record(rec, pass2_lines[name], "updated", name, f["kind"], f["exact"], CASES, drift_max)
        if not e1:
            v = ("UNTESTED", "no pass-1 rows for this arm")
        elif v is not None:
            pass                                    # a pass-2 verdict (RETEST / INCONCLUSIVE when pass 2 drifted)
        elif not ok1:
            rec["carry"] = promising and not screen_only        # a screen-only INCONCLUSIVE is re-screened once (sweep.py select), not confirmed
            v = ("INCONCLUSIVE", "pass-1 block %s base drift %s > %.0f %% (after its re-run): no verdict%s" % (
                lab, worst(d1), drift_max, "; promising numbers: carried to the follow-up RETEST slice" if promising else ""))
        elif screen_only and not pm and not promising and e1 and max(e1.values()) > 0:
            rec["carry"] = False
            v = ("PROMISING-LOW", "run 1 (clean block %s): best case %+.2f %% (0 < gain < %.1f %%): LOW gain, not rescreened this round: backlog (us-backlog-low.json, run-1 numbers kept)" % (
                lab, max(e1.values()), HIGH_GAIN))
        elif screen_only and not pm and not promising and e1 and f["kind"] == "minus":
            rec["carry"] = True
            v = ("HURTS", "run 1 (clean block %s): every case <= 0 %% with the preset flag ON (best %+.2f %%): having it ON hurts; rescreened once (a shipped change is at stake: no removal from one run)" % (
                lab, max(e1.values())))
        elif not promising and pm:
            v = (NOGAIN, "RESCREEN (second 1-run test, clean block %s): the run-1 benefit on %s did not repeat (%+.2f %%): dropped, no shipped change" % (lab, pm, e1.get(pm, 0)))
        elif not promising:
            v = (NOGAIN, "dropped after pass 1: no benefit observed (bar %.1f %%; clean block %s; n=1, not a refutation): no shipped change"
                 % (PROMISE[f["exact"]], lab))
        elif screen_only:
            rec["carry"] = True
            v = ("PROMISING", ("RESCREEN (second 1-run test, clean block %s): the benefit repeats on %s (%+.2f %%): the 3-rep confirm follows" % (lab, pm, e1.get(pm, 0)))
                 if pm else "stage-1 screen (n=1, clean block %s): a case >= +%.1f %% (best %+.2f %%): HIGH gain, goes to the rescreen" % (lab, HIGH_GAIN, max(e1.values())))
        else:
            rec["carry"] = True
            v = ("RETEST", "promising; pass 2 not run (budget): carried to the follow-up RETEST slice" if m2 is None
                 else "promising; no pass-2 rows for this arm (budget): carried to the follow-up RETEST slice")
        if screen_only and pm and ok1 and e1:                      # both runs known, clean block: the two-run rule gives the verdict (owner 2026-11-20: no separate 3-rep stage)
            v2, why2, mean, spread = two_run(p1, e1, f["exact"])
            rec.update({"pass2": mean, "pass2_clean": True, "spread": spread, "runs": [p1, e1], "carry": False,
                        "pass2_design": "two-run: run 1 (screen) + run 2 (rescreen), mean effect, spread = their difference"})
            v = (v2, why2)
        rec["verdict"], rec["reason"] = v
        flags[f["flag"]] = rec
    return {"schema": "pancake.sweep-triage/2", "branch": meta["branch"], "slices": [meta["slice"]], "protected": meta.get("protected", {}),
            "drift_max": drift_max, "flags": flags}


# ------------------------------------------------------------------ the PAIRED confirm block (owner order 2026-11-20: confirm the flags one by one)
PAIRED_REPS = 3


def rep_name(arm, k):
    return arm if k == 1 else "%s~%d" % (arm, k)


def paired_spec(base_path, base_name, base_arm, name, arm, cases=None, note="", reps=PAIRED_REPS, layout="interleaved", warm=True):
    """ONE flag's confirm block: warm-up (the base, not counted), base, flag, base, flag, base, flag, base -- abtest --tier screen --drift-check runs the arms in spec order, one
    engine start each, and the base again at the end (base-late): 3 interleaved reps of the flag, each between two base runs of the same block."""
    cases = cases or CASES
    arms = {WARM: base_arm} if warm else {}      # the warm-up start (T-1004): first, not counted, not judged
    if layout == "compact":                      # base, flag x reps, then abtest's base-late (owner 2026-11-20: 5 counted starts, ~12 min)
        arms[base_name] = base_arm
        for k in range(1, reps + 1):
            arms[rep_name(name, k)] = arm
    else:
        for k in range(1, reps + 1):
            arms[rep_name(base_name, k)] = base_arm
            arms[rep_name(name, k)] = arm
    return {"base": base_path, "drift_check": True, "lossy": True, "prefill": any(c.startswith("long") for c in cases), "cases": list(cases),
            "reps": 1, "arms": arms,
            "note": ("paired confirm block of ONE flag (%s against %s): one warm-up start of the base (not counted), base, flag x %d interleaved, base again at the end; tools/pancake/box_chains/"
                     "sweep.py paired_spec. %s" % (name, base_name, reps, note)).strip()}


def paired_eval(lines, base_name, name, kind, cases=None, reps=PAIRED_REPS, layout="interleaved"):
    """(effects {metric: % gain of the flag ON, the mean of the per-rep paired gains: flag rep k against the mean of the base runs before and
    after it}, drift {metric: the largest deviation (signed, %) of a later base run from the first one}, per-rep gains)."""
    cases = cases or CASES
    m = metrics(lines, cases=cases)
    compact = layout == "compact"
    bases = [base_name, "base-late"] if compact else [rep_name(base_name, k) for k in range(1, reps + 1)] + ["base-late"]
    flags = [rep_name(name, k) for k in range(1, reps + 1)]
    per = []
    for k, fa in enumerate(flags):
        fx = m.get(fa)
        if not fx:
            continue
        br = [m.get(b) for b in (bases if compact else bases[k:k + 2]) if m.get(b)]
        e = {}
        for met, x in fx.items():
            bv = [b[met] for b in br if b.get(met)]
            if not bv or not x:
                continue
            b = sum(bv) / len(bv)
            e[met] = ((x / b) if kind == "plus" else (b / x)) * 100.0 - 100.0
        if e:
            per.append(e)
    eff = {}
    for met in {k for e in per for k in e}:
        vals = sorted(e[met] for e in per if met in e)
        eff[met] = round(vals[len(vals) // 2] if compact and len(vals) % 2 else sum(vals) / len(vals), 2)     # compact: the median of the 3 reps (a disturbed start does not move it)
    eff = {k: eff[k] for k in sorted(eff, key=lambda k: (METRICS.index(k) if k in METRICS else 99, k))}
    md = metrics(lines, cases=cases, last=True)        # the drift is judged on the last request of each case per start (T-1004), the effects on the mean
    first = md.get(bases[0], {})
    drift = {}
    for c in drift_keys(cases):
        devs = [md[b][c] / first[c] * 100.0 - 100.0 for b in bases[1:] if md.get(b, {}).get(c) and first.get(c)]
        if devs:
            drift[c] = round(max(devs, key=abs), 2)
    return eff, drift, [{k: round(v, 2) for k, v in e.items()} for e in per]


def paired_record(rec, runs, base_name, name, kind, exact, cases, mx=DRIFT_MAX, layout="interleaved"):
    """Fill rec from the paired block run(s) [(label, lines)] of one flag (the latest clean run, else the latest) -> (verdict, reason)."""
    evs = [(lab, paired_eval(ls, base_name, name, kind, cases, layout=layout)) for lab, ls in runs if ls]
    evs = [x for x in evs if x[1][0]]
    if not evs:
        rec.update({"pass2": None, "pass2_drift": None, "pass2_clean": None})
        return None
    lab, (eff, drift, per) = ([x for x in evs if clean(x[1][1], mx)] or evs)[-1]
    ok = clean(drift, mx)
    rec.update({"pass2": eff, "pass2_reps": per, "pass2_drift": drift or None, "pass2_clean": ok, "pass2_block": lab,
                "pass2_runs": [x[0] for x in evs], "pass2_design": ("compact paired block: base, flag x 3, base (median of the 3 paired gains)" if layout == "compact" else "paired block: warm-up (not counted), base, flag x 3 interleaved, base"), "drift_estimator": "last request of each case per start"})
    if not ok:
        rec["carry"] = True
        return "INCONCLUSIVE", "paired block %s base drift %s > %.0f %% (after its re-run): no verdict; carried to the follow-up RETEST slice" % (
            lab, worst(drift), mx)
    return confirm_verdict(eff, exact)


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
        elif v == "DROP":
            r["verdict"], r["reason"] = NOGAIN, "was DROP from pass 1 alone (clean): dropped after pass 1, n=1, not a refutation; no shipped change"
        r["promising"] = bool(r.get("promising")) and ok1
        out["flags"][flag] = r
    return out


def pass2_spec(spec, tri, max_flags=None):
    """The 3-rep confirm spec of the promising flags: ALL of them (owner order 2026-11-20: no cap), the strongest first; max_flags (only the
    budget sets it) cuts the list -- the cut flags are returned as the carry list for the follow-up RETEST slice."""
    prom = [r for r in tri["flags"].values() if r.get("promising")]
    prom.sort(key=lambda r: -max(r["pass1"].values()))
    run = prom if max_flags is None else prom[:max(0, max_flags)]
    carry = prom[len(run):]
    arms = {"updated": spec["arms"]["updated"]}
    for r in run:
        arms[r["arm"]] = spec["arms"][r["arm"]]
    return {**spec, "arms": arms, "drift_check": False, "reps": 3,
            "note": "bench-sweep.sh pass 2: the promising flags of pass 1 (%d of %d, the strongest first%s), 3 reps against updated"
                    % (len(run), len(prom), "; %d carried to the follow-up RETEST slice (budget)" % len(carry) if carry else "")}, \
        [r["flag"] for r in run], carry


def carry_doc(spec, meta, carry, tag):
    """The carry file: the arms (definitions included) the budget left out of pass 2, for bench-retest.sh (slice carry-<branch>)."""
    return {"schema": "pancake.sweep-carry/1", "branch": meta["branch"], "slice": tag, "base": spec.get("base"), "cases": spec.get("cases", CASES),
            "base_arm": spec["arms"]["updated"],
            "arms": {r["arm"]: {"arm": spec["arms"][r["arm"]], "meta": meta["flags"][r["arm"]], "pass1": r.get("pass1")} for r in carry}}


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
            if r.get("verdict") == "DROP" and not (r.get("pass2") and r.get("pass2_clean") is True):
                r = {**r, "prior_verdict": "DROP", "verdict": NOGAIN,
                     "reason": "was DROP without a clean pass 2: dropped after pass 1 (n=1, not a refutation); no shipped change"}
            old = out["flags"].get(fl)
            if old is not None and r.get("verdict") in WEAK and old.get("verdict") not in WEAK:
                continue
            out["flags"][fl] = r
        out["slices"] += d.get("slices", [])
    if out is not None:
        out["schema"] = "pancake.sweep-triage/2"
    return out


# ------------------------------------------------------------------ RETEST slices (bench-retest.sh): one paired block per flag


def start_s(cases):
    return START_OVH + sum(CASE_S.get(c, 60) for c in cases)


def resolve_entry(key, ent, bases, registry, src):
    """One retest entry -> a block description {bid, branch, kind, flag, exact, bundle, base_name, base, arm, env, args, ship, cases, est, ...}.
    bases[branch] = {"path": base config, "arms": {"updated": .., "plain": .., "exact": ..}}."""
    br, name = key.split(":", 1)
    B = bases[br]
    upd = B["arms"]["updated"]
    d = {"key": key, "branch": br, "flag": name, "why": ent.get("why", ""), "cases": list(ent.get("cases") or CASES), "bundle": bool(ent.get("bundle"))}
    if ent.get("gate"):
        flag = ent["gate"]
        reg = next((c for c in registry.get("untested", {}).get(br, []) if c["name"] == flag), {})
        keys = list((reg.get("env") or {})) + list((reg.get("args") or {})) or [flag]
        arm = {**upd, "env": {k: v for k, v in (upd.get("env") or {}).items() if k not in keys},
               "args": {k: v for k, v in (upd.get("args") or {}).items() if k not in keys}}
        d.update(type="gate", base_name="updated", base=B["path"], arms={"updated": upd, "minus-" + flag.lstrip("-"): arm}, est=int(ent.get("est", 2700)))
        return d
    if ent.get("spec"):
        d.update(type="spec", spec=ent["spec"], exact=bool(ent.get("exact")), est=int(ent.get("est", 1700)))
        return d
    if d["bundle"]:
        base_arm, arm = B["arms"][ent["base"]], json.loads(json.dumps(B["arms"][ent["arm_from"]]))
        arm.setdefault("env", {}).update(ent.get("env") or {})
        d.update(type="paired", kind="plus", exact=bool(ent.get("exact")), base_name=ent["base"], arm_name=ent["arm_from"], base_arm=base_arm, arm=arm,
                 env={}, args={})
    else:
        reg = next((c for c in registry.get("untested", {}).get(br, []) if c["name"] == name), None) or {}
        env = {k: str(v).replace("@SRC@", src[br]) for k, v in (ent.get("env") or reg.get("env") or {}).items()}
        args = {k: str(v).replace("@SRC@", src[br]) for k, v in (ent.get("args") or reg.get("args") or {}).items()}
        if not env and not args:
            raise SystemExit("sweep retest: entry %s has no env / args and is not in the registry" % key)
        ue, ua = upd.get("env") or {}, upd.get("args") or {}
        kind = ent.get("kind") or ("minus" if any(ue.get(k) is not None for k in env) or any(ua.get(k) is not None for k in args) else "plus")
        exact = ent.get("exact", reg.get("exact", True))
        if kind == "minus":
            groups = registry.get("groups", {})
            ke = [x for k in env for x in groups.get(k, [k]) if x in ue]
            ka = [k for k in args if k in ua]
            if not ke and not ka:
                raise SystemExit("sweep retest: %s is a minus entry but the updated arm does not set it" % key)
            arm = {**upd, "env": {k: v for k, v in ue.items() if k not in ke}, "args": {k: v for k, v in ua.items() if k not in ka}}
            env, args = {k: ue[k] for k in ke}, {k: ua[k] for k in ka}
        else:
            arm = {**upd, "env": {**ue, **env}, "args": {**ua, **args}}
        d.update(type="paired", kind=kind, exact=bool(exact), base_name="updated", arm_name="%s-%s" % (kind, name.replace("STRATA_", "").lstrip("-")),
                 base_arm=upd, arm=arm, env=env, args=args)
        if ent.get("ship"):
            d["ship"] = ent["ship"]
    d["base"] = B["path"]
    d["est"] = 8 * start_s(d["cases"])
    return d


def carry_entries(docs):
    """Block descriptions of the carried arms (carry files of bench-sweep.sh): one paired block each, against the carry file's updated arm."""
    out = []
    for doc in docs:
        for an, x in doc.get("arms", {}).items():
            m = x["meta"]
            cases = list(doc.get("cases") or CASES)
            out.append({"key": "%s:%s" % (doc["branch"], m["flag"]), "branch": doc["branch"], "flag": m["flag"], "type": "paired", "kind": m["kind"],
                        "exact": bool(m.get("exact", True)), "bundle": False, "base_name": "updated", "arm_name": an, "base_arm": doc["base_arm"],
                        "arm": x["arm"], "env": m.get("env") or {}, "args": m.get("args") or {}, "base": doc["base"], "cases": cases,
                        "est": 8 * start_s(cases), "why": "carried from slice %s (pass 1 %s)" % (doc.get("slice"), json.dumps(x.get("pass1"))),
                        "gain": max((x.get("pass1") or {0: 0}).values())})
    return out


def carried(docs, br=None):
    """The carried flags of the carry files, deduplicated (a flag carried twice runs once: the later file wins), the strongest pass-1 gain first
    (stage 2 of the US queue confirms the most promising flags first)."""
    seen, uniq = set(), []
    for e in carry_entries(docs):
        if br not in (None, e["branch"]):
            continue
        if e["key"] in seen:
            uniq = [u for u in uniq if u["key"] != e["key"]]
        seen.add(e["key"])
        uniq.append(e)
    return sorted(uniq, key=lambda e: -e["gain"])




def rank_doc(ents, high=HIGH_GAIN):
    """The ranked queue of the PROMISING flags (best screen case effect first, OPTIONAL-pattern flags by their big case): gain >= high (2.0 %) = HIGH, confirmed this round
    in descending order (6 per slice, the tail spills into the next slice); 0 < gain < 2 % = PROMISING-LOW: skipped for now, confirmed in stage 2b after the final
    comparison in the same order.  Nothing is dropped here (flags <= 0 % never get this far: NO-GAIN-P1)."""
    m = len(ents)
    cut = sum(1 for e in ents if e["gain"] >= high)
    rule = "%d of %d promising flag(s) have a best screen case >= %.1f %% (HIGH, confirmed this round, highest first); %d are PROMISING-LOW (0 < gain < %.1f %%, stage 2b after the final comparison)" % (
        cut, m, high, m - cut, high)
    flags = [{"rank": i + 1, "flag": e["flag"], "arm": e["arm_name"], "kind": e["kind"], "exact": e["exact"], "gain": e["gain"],
              "pass1": json.loads(e["why"].split("(pass 1 ", 1)[1][:-1]) if "(pass 1 " in e["why"] else None,
              "verdict": "PROMISING" if i < cut else "PROMISING-LOW", "tier": "high" if i < cut else "low"} for i, e in enumerate(ents)]
    return {"schema": "pancake.sweep-ranked/1", "cut": cut, "n": m, "high": high, "rule": rule, "flags": flags}


def gate_entries(docs, base_path, upd, prefix="", bar=3.0):
    """Quality-gate blocks of flags whose CLEAN confirm shows a lossy gain >= bar (verdict RETEST-GATE, or an OPTIONAL lossy one) in the triage docs
    `docs`, only records of a retest slice whose name starts with `prefix` (the stage-2 slices usS2-*): the gate compares updated with updated MINUS the flag (a flag the
    updated arm sets) or PLUS the flag (one it does not set)."""
    out = []
    ue, ua = upd.get("env") or {}, upd.get("args") or {}
    for doc in docs:
        for flag, r in doc.get("flags", {}).items():
            p2 = r.get("pass2") or {}
            if r.get("bundle") or r.get("exact", True) or r.get("verdict") not in ("RETEST-GATE", "OPTIONAL") or r.get("pass2_clean") is not True \
                    or not p2 or max(p2.values()) < bar                     or not (str(r.get("source", "")).startswith("bench-retest.sh " + prefix) or (prefix and str(r.get("slice") or "").startswith(prefix))):
                continue
            env, args = r.get("env") or {}, r.get("args") or {}
            if r.get("kind") == "minus":
                arm = {**upd, "env": {k: v for k, v in ue.items() if k not in env}, "args": {k: v for k, v in ua.items() if k not in args}}
            else:
                arm = {**upd, "env": {**ue, **env}, "args": {**ua, **args}}
            out.append({"key": "us:GATE-" + flag, "branch": "us", "flag": flag, "type": "gate", "base_name": "updated", "base": base_path,
                        "arms": {"updated": upd, "%s-%s" % (r.get("kind", "plus"), flag.replace("STRATA_", "").lstrip("-")): arm}, "est": 2700,
                        "cases": list(CASES), "bundle": False,
                        "why": "quality gate: the confirm of %s shows a lossy gain (%s)" % (flag, json.dumps(p2))})
    return out


def retest_blocks(groups, slice_name, bases, registry, src, carry_docs=(), offset=0, limit=None, gate_docs=(), gate_prefix=""):
    """offset / limit: a carry slice takes the carried flags [offset:offset+limit] (strongest first): the stage-2 slices of the US queue."""
    if slice_name.startswith("carry"):
        parts = slice_name.split("-")
        br = parts[1] if len(parts) > 1 and parts[1] in ("us", "rd") else None
        ents = carried(carry_docs, br)
        ents = ents[offset:] if limit is None else ents[offset:offset + limit]
    elif slice_name.startswith("gates"):
        ents = [resolve_entry(k, groups["entries"].get(k, {}), bases, registry, src) for k in groups["slices"]["gates-us"]]    # every gates-* slice: the static gate(s) + the dynamic ones, cut by offset / limit
        have = {e["key"] for e in ents}
        ents += [e for e in gate_entries(gate_docs, bases["us"]["path"], bases["us"]["arms"]["updated"], gate_prefix) if e["key"] not in have]
        ents = ents[offset:] if limit is None else ents[offset:offset + limit]
    else:
        if slice_name not in groups["slices"]:
            raise SystemExit("sweep retest: no slice %s in the groups file (have: %s)" % (slice_name, ", ".join(groups["slices"])))
        ents = [resolve_entry(k, groups["entries"].get(k, {}), bases, registry, src) for k in groups["slices"][slice_name]]
    for i, e in enumerate(ents, 1):
        e["bid"] = "k%02d" % i
    return ents


def block_spec_of(e):
    if e["type"] == "paired":
        return paired_spec(e["base"], e["base_name"], e["base_arm"], e["arm_name"], e["arm"], e["cases"], e.get("why", ""), layout=e.get("layout", "interleaved"),
                           warm=e.get("warm", True))
    if e["type"] == "gate":
        return {"base": e["base"], "lossy": True, "prefill": False, "cases": ["short", "edit"], "reps": 1, "arms": e["arms"],
                "note": "quality gate of updated against updated minus %s (bench-retest.sh): %s" % (e["flag"], e.get("why", ""))}
    return None


def retest_verdicts(blocks, runs, drift_max=DRIFT_MAX):
    """{branch: v2 triage doc} from the paired blocks' runs {bid: [(label, lines), ...]}.  Bundles get "bundle": true (they change no flag)."""
    out = {}
    for e in blocks:
        if e["type"] == "gate":
            continue
        doc = out.setdefault(e["branch"], {"schema": "pancake.sweep-triage/2", "branch": e["branch"], "slices": [], "protected": {}, "drift_max": drift_max,
                                           "flags": {}})
        rs = runs.get(e["bid"]) or []
        if e["type"] == "spec":
            for lab, ls in rs[-1:]:
                m = metrics(ls, cases=e.get("cases_run") or ["longdec10k", "longdec32k", "longdec64k", "longdec100k"])
                arms = [r.get("arm") for r in records(ls)]
                first = arms[0] if arms else None
                d1 = drift1(ls, first, e.get("cases_run") or ["longdec10k", "longdec32k", "longdec64k", "longdec100k"]) if first else {}
                for a in dict.fromkeys(arms):
                    if a in (first, "base-late"):
                        continue
                    eff = effects(m, a, "plus", (first, "base-late"))
                    doc["flags"]["BUNDLE %s %s" % (e["flag"], a)] = {"flag": "BUNDLE %s %s" % (e["flag"], a), "bundle": True, "kind": "plus",
                        "exact": e.get("exact", False), "arm": a, "pass1": eff, "pass2": None, "drift1": d1, "clean1": clean(d1, drift_max),
                        "verdict": "REFERENCE", "reason": "bundle reference (one screen rep, base again at the end): changes no flag", "block": lab,
                        "source": "bench-retest.sh %s" % e["bid"]}
            continue
        rec = {"flag": e["flag"], "kind": e["kind"], "exact": e["exact"], "env": e.get("env") or {}, "args": e.get("args") or {}, "arm": e["arm_name"],
               "bundle": e["bundle"], "pass1": None, "promising": None, "carry": False, "source": "bench-retest.sh %s %s" % (e.get("slice", ""), e["bid"]),
               "why": e.get("why", ""), "cases": e["cases"]}
        if e.get("ship"):
            rec["ship"] = e["ship"]
        v = paired_record(rec, rs, e["base_name"], e["arm_name"], e["kind"], e["exact"], e["cases"], drift_max, e.get("layout", "interleaved"))
        if v is None:
            v = ("UNTESTED", "no rows of the paired block")
        if e["bundle"]:
            v = (v[0], v[1] + "; BUNDLE (a whole arm): recorded as a reference, changes no flag (flags are judged one by one)")
        rec["verdict"], rec["reason"] = v
        key = ("BUNDLE %s" % e["flag"]) if e["bundle"] else e["flag"]
        doc["flags"][key] = rec
    return out


# ------------------------------------------------------------------ FINAL COMPARISON (bench-final-us.sh): stock upstream against the final shipped flags
FINAL_SHORT = ["short", "edit"]             # edit = the code-edit case ("code")
FINAL_LONG = ["longdec10k", "longdec32k", "longdec64k", "longdec100k"]
FINAL_REPS = 5


def final_arms(tri, s220, s222):
    """(arms, changes): spec 220's updated arm and spec 222's updated-exact arm after apply_triage's rules (tools/pancake/apply_triage.py plan +
    apply_spec: a clean pass-2 DROP / OPTIONAL leaves the arm, a clean KEEP of a flag the arm lacks is added; HOLD, INCONCLUSIVE, RETEST-GATE,
    bundles change nothing)."""
    import sys as _sys
    up = os.path.dirname(HERE)
    if up not in _sys.path:
        _sys.path.insert(0, up)
    import apply_triage as AT
    actions = AT.plan(tri)
    doc = {"arms": {"updated": json.loads(json.dumps(s220["updated"])), "updated-exact": json.loads(json.dumps(s222["updated-exact"]))}}
    changes = AT.apply_spec(doc, actions)
    return doc["arms"], changes, actions


def final_specs(tri, s220_doc, s222_doc, shipped=None, reps=FINAL_REPS, long_cases=None):
    """{name: spec} of the final comparison: final-short (short + edit), final-long (10K / 32K / 64K / 100K), final-gate (the quality-gate spec: plain-upstream
    first = scored twice as the noise, updated-exact, updated); each with the three arms stock plain-upstream, updated (lossy), updated-exact; plus the plan lines."""
    arms, changes, actions = final_arms(tri, s220_doc["arms"], s222_doc["arms"])
    plain = s220_doc["arms"]["plain-upstream"]
    three = {"plain-upstream": plain, "updated": arms["updated"], "updated-exact": arms["updated-exact"]}
    note = ("FINAL COMPARISON (owner order 2026-11-20 09:19: stop the Strata-rdna3 tests, compare only to stock Strata): stock upstream Strata v0.2.16 (plain-upstream, "
            "its own default flags) against updated-strata with the final shipped flags: `updated` (lossy variant: the shipped flags after apply_triage on the US "
            "triage) and `updated-exact` (exact variant: the 11 lossy switches out). Confirm standard: abtest.py --tier confirm --block 3 --warmup 1 --outlier robust "
            "--reclaim-max 3 --reps %d --no-early-stop. Built by tools/pancake/box_chains/bench-final-us.sh (sweep.py final-specs) from specs 220 / 222 and the merged "
            "US triage; the flag changes are listed in the chain log." % reps)
    base = s220_doc["base"]
    mk = lambda cases, prefill, reps: {"base": base, "lossy": True, "prefill": prefill, "cases": list(cases), "reps": reps,
                                       "arms": json.loads(json.dumps(three)), "note": note}
    specs = {"final-short": mk(FINAL_SHORT, False, reps), "final-long": mk(long_cases or FINAL_LONG, True, reps)}
    gate = mk(FINAL_SHORT, False, 1)
    gate["arms"] = {k: json.loads(json.dumps(three[k])) for k in ("plain-upstream", "updated-exact", "updated")}     # the order of spec 222
    gate["note"] = "QUALITY GATE of the final comparison: " + note
    specs["final-gate"] = gate
    shipped_changes = []
    if shipped:
        import apply_triage as AT
        shipped_changes = AT.apply_us_shipped(shipped, actions)
    lines = ["FINALARM %s" % c for c in changes] + ["FINALSHIPPED %s" % c for c in shipped_changes]
    verdicts = {}
    for flag, act, rec in actions:
        verdicts.setdefault("%s/%s" % (rec.get("verdict"), act), []).append(flag)
    lines += ["FINALVERDICTS %s: %s" % (k, ", ".join(v)) for k, v in sorted(verdicts.items())]
    return specs, lines


def final_estimate(specs, reps=FINAL_REPS):
    """seconds: short = 15 starts + 2 warm-ups per block of 3 reps, long = the same for the four long cases (one start holds the cases of a rep), gate = quality_gate.py (T-1003 bench-retest: 2700 s)."""
    n_arms = 3
    blocks = -(-reps // 3)
    starts = n_arms * reps + blocks * n_arms * 1
    return {"final-short": starts * start_s(FINAL_SHORT), "final-long": starts * start_s(specs["final-long"]["cases"]), "final-gate": 2700}


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
    s.add_argument("--pass2-max", type=int, default=6, help="planning only: the pass-2 share of the slice estimate (pass 2 itself has no cap)")
    s.add_argument("--drop-env", action="append", default=[])
    s.add_argument("--set-env", action="append", default=[])
    s.add_argument("--split", choices=["priority", "even"], default="priority")
    s.add_argument("--only-meta", help="only the arms of this (earlier slice's) meta file")
    s.add_argument("--exclude-meta", action="append", default=[], help="leave out the arms of this meta file")
    s.add_argument("--tag", help="the slice name in the meta and the file names (default: --slice)")
    s.add_argument("--block-size", type=int, default=BLOCK)
    s.add_argument("--warm", choices=["each", "first"], default="each")
    s.add_argument("--mid-base", type=int, default=0, help="T-1006: a base start after every K arms of a block (0: base only at both ends)")
    s.add_argument("--source", choices=["untested", "backlog"], default="untested")
    s.add_argument("--extra", action="append", default=[], help="KEY of retest_groups.json (e.g. us:--lookup-max-t): a named flag measured before, screened again (one arm)")
    s.add_argument("--groups", default=os.path.join(HERE, "retest_groups.json"))
    b = sub.add_parser("blocks")
    b.add_argument("--spec", required=True)
    b.add_argument("--prefix", required=True)
    b.add_argument("--size", type=int, default=BLOCK)
    b.add_argument("--seed", default="0")
    b.add_argument("--warm", choices=["each", "first"], default="each")
    b.add_argument("--mid-base", type=int, default=0)
    lc = sub.add_parser("local")
    lc.add_argument("--rows", required=True)
    lc.add_argument("--max", type=float, default=DRIFT_MAX)
    d = sub.add_parser("drift")
    d.add_argument("--rows", required=True)
    d.add_argument("--pass2", action="store_true")
    d.add_argument("--max", type=float, default=DRIFT_MAX)
    t = sub.add_parser("triage")
    t.add_argument("--meta", required=True)
    t.add_argument("--pass1", required=True, nargs="+", help="one jsonl (one-run design) or the block jsonl files, oldest first")
    t.add_argument("--pass2", help="the old multi-flag confirm run (A B B A)")
    t.add_argument("--pass2-run", action="append", default=[], help="ARM=FILE.jsonl: a paired confirm block of one flag (its re-run after it)")
    t.add_argument("--out", required=True)
    t.add_argument("--drift-max", type=float, default=DRIFT_MAX)
    t.add_argument("--prior", help="the merged triage of run 1: this is the RESCREEN, a flag stands only when its run-1 best case is above 0 again")
    t.add_argument("--screen-only", action="store_true", help="stage 1: a promising flag is PROMISING (no pass 2), an INCONCLUSIVE one is not carried")
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
    p.add_argument("--max", type=int, default=None, help="only the budget sets it: the N strongest run, the rest is carried")
    c = sub.add_parser("carry")
    c.add_argument("--spec", required=True)
    c.add_argument("--meta", required=True)
    c.add_argument("--triage", required=True)
    c.add_argument("--tag", required=True)
    c.add_argument("--out", required=True)
    rt = sub.add_parser("retest")
    rt.add_argument("--groups", default=os.path.join(HERE, "retest_groups.json"))
    rt.add_argument("--registry", default=REGISTRY)
    rt.add_argument("--slice", required=True)
    rt.add_argument("--out-dir", required=True)
    rt.add_argument("--rd-plain")
    rt.add_argument("--rd-updated")
    rt.add_argument("--rd-src", default="/mnt/pancake/benchRD/src-updated")
    rt.add_argument("--us-plain")
    rt.add_argument("--us-arm-spec")
    rt.add_argument("--us-exact-spec")
    rt.add_argument("--us-src", default="/mnt/pancake/benchus/src-us")
    rt.add_argument("--drop-env", action="append", default=[])
    rt.add_argument("--set-env", action="append", default=[])
    rt.add_argument("--carry", nargs="*", default=[])
    rt.add_argument("--layout", choices=["interleaved", "compact"], default="interleaved", help="paired block: base,flag x3 interleaved (8 starts) or compact base,flag x3,base-late (5)")
    rt.add_argument("--warm", choices=["each", "first"], default="each", help="warm-up start in every paired block or only the first one of the slice")
    rt.add_argument("--carry-offset", type=int, default=0)
    rt.add_argument("--carry-max", type=int, default=None)
    rt.add_argument("--gate-from", nargs="*", default=[], help="triage files whose clean lossy confirms >= 3 %% get a quality-gate block (slice gates-*)")
    rt.add_argument("--gate-prefix", default="", help="only records whose source starts with this")
    rk = sub.add_parser("rank")
    rk.add_argument("--carry", nargs="*", default=[])
    rk.add_argument("--branch", default="us")
    rk.add_argument("--high", type=float, default=2.0)
    rk.add_argument("--out", required=True)
    cc = sub.add_parser("carry-count")
    cc.add_argument("--carry", nargs="*", default=[])
    cc.add_argument("--branch", default="us")
    cc.add_argument("--gate-from", nargs="*", default=[], help="also print GATECOUNT: the quality-gate blocks of the clean lossy confirms (+1: the static --lookup-max-t gate)")
    cc.add_argument("--gate-prefix", default="")
    cc.add_argument("--gate-base-spec", help="spec 220 (its updated arm is the gate base)")
    sl = sub.add_parser("select")
    sl.add_argument("--triage", required=True)
    sl.add_argument("--verdict", default="INCONCLUSIVE,UNTESTED")
    sl.add_argument("--slice-prefix", default="")
    sl.add_argument("--min-gain", type=float, default=None, help="only flags whose best screen case is >= this (percent)")
    sl.add_argument("--max-gain", type=float, default=None, help="only flags whose best screen case is < this (percent)")
    sl.add_argument("--out", required=True)
    rv = sub.add_parser("retest-triage")
    rv.add_argument("--blocks", required=True)
    rv.add_argument("--run", action="append", default=[], help="BID=FILE.jsonl (a block's run; its re-run after it)")
    rv.add_argument("--out-dir", required=True, help="writes <branch>-triage-rt-<slice>.v2.json per branch")
    rv.add_argument("--drift-max", type=float, default=DRIFT_MAX)
    fs = sub.add_parser("final-specs")
    fs.add_argument("--triage", required=True)
    fs.add_argument("--us-arm-spec", required=True)
    fs.add_argument("--us-exact-spec", required=True)
    fs.add_argument("--shipped", help="terminal/data/tuned-flags.json (a copy is changed in memory only: the FINALSHIPPED lines)")
    fs.add_argument("--out-dir", required=True)
    fs.add_argument("--reps", type=int, default=FINAL_REPS)
    fs.add_argument("--long", default="", help="comma list of long cases (default: all four sizes); the PRELIMINARY comparison: longdec64k,longdec100k")
    fs.add_argument("--prefix", default="final", help="file name prefix (prelim: the PRELIMINARY comparison)")
    fs.add_argument("--no-triage", action="store_true", help="PRELIMINARY: apply no triage, the arms of specs 220 / 222 as shipped now")
    pd = sub.add_parser("paired-drift")
    pd.add_argument("--rows", required=True)
    pd.add_argument("--base", required=True)
    pd.add_argument("--arm", required=True)
    pd.add_argument("--cases", default=",".join(CASES))
    pd.add_argument("--max", type=float, default=DRIFT_MAX)
    pd.add_argument("--layout", choices=["interleaved", "compact"], default="interleaved")
    g = sub.add_parser("merge")
    g.add_argument("--out", required=True)
    g.add_argument("parts", nargs="+")
    e = sub.add_parser("estimate")
    e.add_argument("--arms", type=int, required=True)
    e.add_argument("--pass2-max", type=int, default=6)
    e.add_argument("--block-size", type=int, default=BLOCK)
    e.add_argument("--warm", choices=["each", "first"], default="each")
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
        registry = json.load(open(a.registry, encoding="utf-8"))
        extra = extra_items(a.extra, json.load(open(a.groups, encoding="utf-8")), registry, arm, a.src, a.plain) if a.extra else ()
        spec, meta = build_spec(a.branch, a.plain, arm, registry, a.src, a.slice, split=a.split, only=only,
                                exclude=excl, tag=a.tag, block=a.block_size, extra=extra, source=a.source)
        json.dump(spec, open(a.out, "w"), indent=1)
        json.dump(meta, open(a.out + ".meta.json", "w"), indent=1)
        n = len(meta["flags"])
        p1, rr, p2 = estimate(n, a.pass2_max, a.block_size, a.warm, a.mid_base)
        pr = [priority(f) for f in meta["flags"].values()]
        print("SWEEP %s slice %s: %d flag arms (%d minus, %d plus; %d arm, %d named, %d code) in %d blocks of <= %d (%s), each with updated + base-late; "
              "estimate pass 1 %d s, one block re-run %d s, pass 2 <= %d s, total <= %d s"
              % (a.branch, a.tag or a.slice, n, sum(f["kind"] == "minus" for f in meta["flags"].values()),
                 sum(f["kind"] == "plus" for f in meta["flags"].values()), pr.count(0), pr.count(1), pr.count(2), len(block_sizes(n, a.block_size)),
                 a.block_size, "/".join(map(str, block_sizes(n, a.block_size))), p1, rr, p2, p1 + rr + p2))
        print("ESTIMATE %d %d %d" % (p1, p2, rr))
    elif a.cmd == "blocks":
        spec = json.load(open(a.spec, encoding="utf-8"))
        for bid, sp, order in block_specs(spec, a.size, a.seed, a.warm, a.mid_base):
            path = "%s-%s.json" % (a.prefix, bid)
            json.dump(sp, open(path, "w"), indent=1)
            nstarts = len(sp["arms"]) + 1                         # every arm + base-late (abtest appends it)
            print("BLOCK %s %d %s order: %s" % (bid, nstarts * START_S, path, " ".join(order)))
    elif a.cmd == "local":                                 # T-1006: per arm, the base starts around it, their spread and box state
        rep = local_report(read_lines(a.rows) or [], mx=a.max)
        for arm, bs, d, ok, st in rep:
            print("LOCAL %-28s bases %s spread %s same-state %s -> %s" % (arm, "/".join(bs) if bs else "-", "  ".join("%s %+.1f%%" % kv for kv in d.items()) or "-", st, "CLEAN" if ok else "UNCLEAN"))
        print("LOCALSUM %d of %d arms clean" % (sum(1 for r in rep if r[3]), len(rep)))
        return 0
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
        p2 = read_lines(a.pass2) if a.pass2 else None
        if a.pass2_run:
            p2 = {}
            for x in a.pass2_run:
                an, _, f = x.partition("=")
                ls = read_lines(f)
                if ls is not None:
                    p2.setdefault(an, []).append((os.path.basename(f).rsplit(".", 1)[0], ls))
        prior = json.load(open(a.prior, encoding="utf-8"))["flags"] if a.prior and os.path.exists(a.prior) else None
        tri = triage(meta, sets, p2, a.drift_max, a.screen_only, prior)
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
        spec = json.load(open(a.spec, encoding="utf-8"))
        spec2, names, carry = pass2_spec(spec, tri, a.max)
        if not names and not carry:
            print("PASS2 none: no promising flag")
            return 3
        # one PAIRED block per promising flag (owner order 2026-11-20): <out>-<arm>.json, one PAIR line each, strongest first
        print("PASS2 %d flag(s), one paired block each: %s" % (len(names), ", ".join(names)))
        by_flag = {r["flag"]: r for r in tri["flags"].values()}
        for fl in names:
            an = by_flag[fl]["arm"]
            ps = paired_spec(spec["base"], "updated", spec["arms"]["updated"], an, spec["arms"][an], spec.get("cases") or CASES,
                             "bench-sweep.sh pass 2 of %s" % fl)
            path = "%s-%s.json" % (a.out[:-5] if a.out.endswith(".json") else a.out, an)
            json.dump(ps, open(path, "w"), indent=1)
            print("PAIR %s %d %s" % (an, 8 * start_s(spec.get("cases") or CASES), path))
        if carry:
            print("CARRY %d flag(s) over the budget: %s" % (len(carry), ", ".join(r["flag"] for r in carry)))
        print("ESTIMATE2 %d" % (len(names) * 8 * start_s(spec.get("cases") or CASES)))
    elif a.cmd == "carry":
        spec, meta = json.load(open(a.spec, encoding="utf-8")), json.load(open(a.meta, encoding="utf-8"))
        tri = json.load(open(a.triage, encoding="utf-8"))
        cr = [r for r in tri["flags"].values() if r.get("carry") and r.get("arm") in spec["arms"]]
        if not cr:
            print("CARRY none")
            return 0
        json.dump(carry_doc(spec, meta, cr, a.tag), open(a.out, "w"), indent=1)
        print("CARRY %d flag(s) -> %s: %s" % (len(cr), a.out, ", ".join(r["flag"] for r in cr)))
    elif a.cmd == "retest":
        groups, registry = json.load(open(a.groups, encoding="utf-8")), json.load(open(a.registry, encoding="utf-8"))
        bases, src = {}, {"rd": a.rd_src, "us": a.us_src}
        if a.rd_plain and a.rd_updated and os.path.exists(a.rd_plain) and os.path.exists(a.rd_updated):
            rp = json.load(open(a.rd_plain, encoding="utf-8"))
            upd, notes = fix_config(json.load(open(a.rd_updated, encoding="utf-8")), a.drop_env, a.set_env)
            for ln in notes:
                print(ln)
            bases["rd"] = {"path": a.rd_plain, "arms": {"updated": arm_from_configs(rp, upd), "plain": {"exe": rp.get("exe")}}}
        if a.us_plain and a.us_arm_spec and os.path.exists(a.us_arm_spec):
            s220 = json.load(open(a.us_arm_spec, encoding="utf-8"))["arms"]
            ex = json.load(open(a.us_exact_spec, encoding="utf-8"))["arms"] if a.us_exact_spec and os.path.exists(a.us_exact_spec) else {}
            bases["us"] = {"path": a.us_plain, "arms": {"updated": s220["updated"], "plain": s220.get("plain-upstream", {}),
                                                         "exact": ex.get("updated-exact", {})}}
        keys = groups["slices"].get(a.slice, []) if not a.slice.startswith("carry") else []
        miss = sorted({k.split(":", 1)[0] for k in keys} - set(bases))
        if miss:
            raise SystemExit("sweep retest: no base configs for branch(es) %s" % ", ".join(miss))
        docs = [json.load(open(f, encoding="utf-8")) for f in a.carry if os.path.exists(f)]
        gdocs = [json.load(open(f, encoding="utf-8")) for f in a.gate_from if os.path.exists(f)]
        blocks = retest_blocks(groups, a.slice, bases, registry, src, docs, a.carry_offset, a.carry_max, gdocs, a.gate_prefix)
        first_paired = True
        for e in blocks:
            e["slice"] = a.slice
            if e["type"] == "paired":
                e["layout"] = a.layout
                e["warm"] = a.warm == "each" or first_paired
                first_paired = False
                e["est"] = ((1 if e["warm"] else 0) + (5 if a.layout == "compact" else 7)) * start_s(e["cases"])
            sp = block_spec_of(e)
            e["path"] = os.path.join(a.out_dir, "retest-%s-%s.json" % (a.slice, e["bid"])) if sp else None
            if sp:
                json.dump(sp, open(e["path"], "w"), indent=1)
            print("BLOCK %s %s %d %s %s %s %s %s %s" % (e["bid"], e["type"], e["est"], e["branch"], e["path"] or e.get("spec"), e.get("base_name", "-"),
                                                  e.get("arm_name", "-"), ",".join(e.get("cases") or CASES), e["flag"]))
        json.dump(blocks, open(os.path.join(a.out_dir, "retest-%s.blocks.json" % a.slice), "w"), indent=1)
        print("RETEST %s: %d blocks, estimate %d s" % (a.slice, len(blocks), sum(e["est"] for e in blocks)))
        print("ESTIMATE %d" % sum(e["est"] for e in blocks))
    elif a.cmd == "rank":
        docs = [json.load(open(f, encoding="utf-8")) for f in a.carry if os.path.exists(f)]
        ents = carried(docs, a.branch)
        doc = rank_doc(ents, a.high)
        json.dump(doc, open(a.out, "w"), indent=1)
        print("RANKCUT %d %d" % (doc["cut"], len(ents)))
        print("RANKRULE %s" % doc["rule"])
        for r in doc["flags"]:
            print("RANKFLAG %d %s %+.2f %s" % (r["rank"], r["flag"], r["gain"], r["verdict"]))
    elif a.cmd == "carry-count":
        docs = [json.load(open(f, encoding="utf-8")) for f in a.carry if os.path.exists(f)]
        ents = carried(docs, a.branch)
        print("CARRYCOUNT %d" % len(ents))
        if a.gate_from and a.gate_base_spec:
            gd = [json.load(open(f, encoding="utf-8")) for f in a.gate_from if os.path.exists(f)]
            print("GATECOUNT %d" % (1 + len(gate_entries(gd, "-", load_arm(a.gate_base_spec), a.gate_prefix))))
        for e in ents:
            print("CARRYFLAG %s %+.2f" % (e["flag"], e["gain"]))
    elif a.cmd == "select":
        tri = json.load(open(a.triage, encoding="utf-8"))
        vs = set(a.verdict.split(","))
        fl = {r["arm"]: r for r in tri["flags"].values() if r.get("verdict") in vs and str(r.get("slice") or "").startswith(a.slice_prefix) and r.get("arm")
              and (a.min_gain is None or max((r.get("pass1") or {0: -1e9}).values()) >= a.min_gain) and (a.max_gain is None or max((r.get("pass1") or {0: -1e9}).values()) < a.max_gain)}
        json.dump({"schema": "pancake.sweep-select/1", "verdicts": sorted(vs), "flags": fl}, open(a.out, "w"), indent=1)
        print("SELECT %d flag(s) with verdict %s: %s" % (len(fl), "/".join(sorted(vs)), ", ".join(r["flag"] for r in fl.values())))
    elif a.cmd == "retest-triage":
        blocks = json.load(open(a.blocks, encoding="utf-8"))
        runs = {}
        for x in a.run:
            bid, _, f = x.partition("=")
            ls = read_lines(f)
            if ls is not None:
                runs.setdefault(bid, []).append((os.path.basename(f).rsplit(".", 1)[0], ls))
        docs = retest_verdicts(blocks, runs, a.drift_max)
        for br, doc in docs.items():
            doc["slices"] = [blocks[0].get("slice", "retest")] if blocks else []
            path = os.path.join(a.out_dir, "%s-triage-rt-%s.v2.json" % (br, doc["slices"][0] if doc["slices"] else "retest"))
            json.dump(doc, open(path, "w"), indent=1)
            for fl, rc in doc["flags"].items():
                print("%-12s %-36s %s  [%s]" % (rc["verdict"], fl, json.dumps(rc.get("pass2") or rc.get("pass1")), rc["reason"]))
            print("TRIAGEFILE %s %s" % (br, path))
    elif a.cmd == "final-specs":
        tri = json.load(open(a.triage, encoding="utf-8")) if not a.no_triage else {"schema": "pancake.sweep-triage/2", "branch": "us", "protected": {}, "flags": {}}
        if tri.get("schema") != "pancake.sweep-triage/2" or tri.get("branch") not in (None, "us"):
            raise SystemExit("sweep final-specs: %s is not a v2 triage of branch us" % a.triage)
        sh = json.load(open(a.shipped, encoding="utf-8")) if a.shipped and os.path.exists(a.shipped) else None
        specs, lines = final_specs(tri, json.load(open(a.us_arm_spec, encoding="utf-8")), json.load(open(a.us_exact_spec, encoding="utf-8")), sh, a.reps, a.long.split(",") if a.long else None)
        est = final_estimate(specs, a.reps)
        for name, sp in specs.items():
            fn = os.path.join(a.out_dir, name.replace("final", a.prefix, 1) + ".json")
            json.dump(sp, open(fn, "w"), indent=1)
            print("FINALSPEC %s %s arms %s cases %s reps %s estimate %d s" % (name, fn, ",".join(sp["arms"]),
                                                                              ",".join(sp["cases"]), sp["reps"], est[name]))
        for ln in lines:
            print(ln[:600])
        print("ESTIMATE %d" % sum(est.values()))
    elif a.cmd == "paired-drift":
        lines = read_lines(a.rows) or []
        eff, dd, per = paired_eval(lines, a.base, a.arm, "plus", a.cases.split(","), layout=a.layout)
        ok = clean(dd, a.max)
        print("DRIFT %s max %s -> %s" % ("  ".join("%s %+.1f%%" % kv for kv in dd.items()) or "no base rows",
                                          ("%.1f%%" % max(abs(v) for v in dd.values())) if dd else "-",
                                          "CLEAN" if ok else ("DRIFT (> %g%%)" % a.max if dd else "UNKNOWN")))
        return 0 if ok else 4 if dd else 5
    elif a.cmd == "merge":
        json.dump(merge(a.parts), open(a.out, "w"), indent=1)
    else:
        p1, rr, p2 = estimate(a.arms, a.pass2_max, a.block_size, a.warm)
        print("ESTIMATE %d %d %d" % (p1, p2, rr))
    return 0


if __name__ == "__main__":
    sys.exit(main())
