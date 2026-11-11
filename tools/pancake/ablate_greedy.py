"""ablate_greedy.py - greedy ablation over untested speed items, tested as ONE growing bundle (MOD_PLAN Phase 17, item 17.3).

Testing the coded-but-untested items one by one cannot find interactions (a switch that pays only after another one, two
switches that substitute for each other) and costs one full A/B per item.  This driver builds the bundle greedily instead:

  round 1   arm "base" = the start bundle (the preset, plus the plan's "bundle_start"); one arm per remaining candidate =
            base + that candidate.  One abtest run (a generated spec file, abtest.py as a subprocess: this tool measures nothing).
  judge     every candidate arm against "base" with promote.py's own arm judge (promote.judge_arm): paired gain, 95% interval
            (Student t, Bonferroni over arms x claim cases), MIN_PAIRS pairs, regression and switch-never-acted checks.
            A candidate "clears the bar" when promote's verdict for it is PROMOTE (lower bound of the gain on a claim case
            >= min_gain: 2% for exact items = MOD_PLAN 17.3, 3% for lossy ones = promote.MIN_GAIN).
  accept    the clearing candidate with the highest lower bound joins the bundle; candidates that substitute for it are
            dropped and noted; the next round starts from the new bundle with the remaining candidates.
  stop      no candidate clears, nothing remains, or --max-rounds / plan "max_rounds" is reached.
  then      (unless final_check is off) one more run: the start preset against the final bundle, so the stack is judged as
            a whole, not as a sum of rounds.

Plan (JSON; the real one is bench/specs/ablation-17-3-plan.json):
  {"name": "ablation-17-3",
   "base": "pancake-flashnext-iq2_xs-fast-hip.json",      the abtest spec "base" (a preset), or "base_spec": an abtest spec file
                                                          whose base and first-arm delta are the start (its cases / reps are defaults)
   "bundle_start": {"env": {...}, "args": {...}},         optional: a delta every arm carries (default: none)
   "cases": ["short", "edit"], "reps": 9, "block": 3, "warmup": 1, "tier": "confirm", "outlier": "robust", "reclaim_max": 3.0,
   "claim": ["short", "edit"],                            cases a gain is claimed on (default: all cases); others only checked
   "min_gain": 2.0, "min_gain_lossy": 3.0, "min_pairs": 3,
   "max_rounds": 8, "lossy_policy": "exact-first",        exact-first | mixed | skip, see below
   "retest_benched": false, "final_check": true, "spec_extra": {"temperature": 0},   (spec keys copied as they are)
   "candidates": [{"name": "poll1", "env": {"STRATA_COMBINE_POLL1": "1"}, "args": {"--spec": "5"},
                   "lossy": false,                        true = output may differ: never promoted by this tool, see below
                   "substitutes_of": ["other-name"],      symmetric: accepting either drops the other
                   "interacts_with": ["other-name"],      a benched candidate is tested again when one of these is accepted
                   "disabled": "reason",                  listed, never tested (a switch that cannot act on this preset)
                   "group": "core",                       free label: --group core tests only the candidates with it
                   "note": "...", "source": "file or spec it comes from"}]}
  env / args follow abtest: "KEY": null removes, "--flag": "value" sets, "--flag": "" is a bare switch, "--flag": null removes.

Rules (see the module docstring of abtest.py and promote.py for the reasons):
  - every arm of a round runs the same tier and the same abtest options; a switch that never logged "acted" voids its arm
    (promote's switch check), so a candidate that cannot act is never accepted; parity and the quality gates are NOT judged
    here: text identity cannot judge exactness on this engine (it is not run-to-run deterministic), so an item declared
    exact is checked with the noise-aware equivalence gate, and a lossy one goes to the quality gate (below).
  - benched: a candidate whose arm is REJECT (no gain worth having), shows a regression, or never acted is not run in the
    next rounds (each round costs one engine-start set per arm), unless "retest_benched" is true or one of its
    "interacts_with" candidates was accepted since.  An undecided candidate (interval straddles the bar) stays in.
  - lossy_policy: "exact-first" takes the best clearing exact candidate and a lossy one only in a round where no exact one
    clears (the exact items form a prefix of the bundle); "mixed" takes the best clearing candidate of either kind; "skip"
    drops every lossy candidate from the plan.  A lossy candidate that clears the SPEED bar is not promoted: it is written to
    quality-gate-spec.json for tools/pancake/quality_gate.py, and the exact-only preset is written next to the full one.
  - tier "screen" judges one pass by abtest's own screen rule (>= +min_gain on a claim case, no case below -3%) and says
    so: a screen bundle is a lead, to be confirmed with a confirm-tier run of the final bundle.

Output (--out-dir, default bench/results/<day>-ablate-<name>/): RESULTS.md (bundle order, a table per round with the gain and
its interval per case, the interaction tables, the lossy queue), round<k>.spec.json / round<k>.jsonl (abtest's files),
final-preset.json (the start preset with the accepted bundle applied; written when the preset file is readable),
final-preset-exact.json (exact items only, when a lossy one was accepted), quality-gate-spec.json, state.json.

    python tools/pancake/ablate_greedy.py PLAN.json --dry-run                 the plan, the name check, round 1's spec, the run order
    python tools/pancake/ablate_greedy.py PLAN.json [--out-dir DIR] [--resume]    the real run (hours: needs the whole machine)
    python tools/pancake/ablate_greedy.py PLAN.json --fake --out-dir DIR      the whole loop on the fake engine (seconds, no GPU)

--fake builds a fake repo root (tools/pancake/probes/fake_engine.py) and runs the same loop against it; a candidate's "fake" key
({"gain_pct": 5.0}) is its synthetic effect (gains add up in a bundle, a lossy candidate changes the text).  The numbers are
synthetic: RESULTS.md says so.  Nothing real is measured, built or started in that mode.
"""
import argparse
import copy
import glob
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import abtest  # noqa: E402   (schedule(), make_config(), select(): reads sysfs only)
import promote  # noqa: E402   (judge_arm and its statistics: the rule this tool applies)

ROOT = HERE.parents[1]
NAME = re.compile(r"[A-Za-z0-9._+-]+")
POLICIES = ("exact-first", "mixed", "skip")
SPEC_EXTRA_KEYS = ("temperature", "top_p", "seed", "prefill")
SRC_GLOBS = ("*.cpp", "*.hpp", "*.cu", "*.cuh", "*.h", "*.inc", "*.def", "*.py")
PARITY_WAIVED = "ablation driver: exactness is checked with the equivalence gate, a lossy item goes to the quality gate"


class PlanError(ValueError):
    pass


# ------------------------------------------------------------------------------------------------ deltas
def norm_delta(d):
    """{"env": {...}, "args": {...}} with values as strings (None stays None: remove)."""
    out = {}
    for sect in ("env", "args"):
        cur = {}
        for k, v in ((d or {}).get(sect) or {}).items():
            cur[str(k)] = None if v is None else str(v)
        if cur:
            out[sect] = cur
    return out


def merge(a, b):
    """The delta of a bundle `a` with `b` added: b wins on a shared key (abtest applies a delta over the preset in that order)."""
    out = copy.deepcopy(norm_delta(a))
    for sect, items in norm_delta(b).items():
        out.setdefault(sect, {}).update(items)
    return out


def bundle_delta(start, members):
    d = norm_delta(start)
    for c in members:
        d = merge(d, c)
    return d


# ------------------------------------------------------------------------------------------------ plan
def load_plan(path):
    plan = json.loads(Path(path).read_text())
    plan["_path"] = str(path)
    return check_plan(plan)


def check_plan(plan):
    """Fills the defaults and validates; returns the plan.  PlanError says what is wrong."""
    plan = copy.deepcopy(plan)
    plan.setdefault("name", Path(plan.get("_path", "ablation")).stem)
    if not NAME.fullmatch(plan["name"]):
        raise PlanError(f"plan name {plan['name']!r} must match {NAME.pattern}")
    base_spec = None
    if plan.get("base_spec"):
        p = Path(plan["base_spec"])
        base_spec = json.loads((p if p.is_absolute() else ROOT / p).read_text())
        plan.setdefault("base", base_spec["base"])
        first = next(iter(base_spec["arms"]))
        plan.setdefault("bundle_start", base_spec["arms"][first])
        plan.setdefault("cases", base_spec.get("cases"))
        plan.setdefault("reps", base_spec.get("reps"))
    if not plan.get("base"):
        raise PlanError("plan has neither 'base' (a preset) nor 'base_spec'")
    plan["bundle_start"] = norm_delta(plan.get("bundle_start"))
    plan["cases"] = plan.get("cases") or ["short", "edit"]
    if plan.get("reps") is None:
        plan["reps"] = 9
    plan.setdefault("block", 3)
    plan.setdefault("warmup", 1)
    plan.setdefault("tier", "confirm")
    plan.setdefault("outlier", "robust")
    plan.setdefault("reclaim_max", 3.0)
    plan["claim"] = plan.get("claim") or list(plan["cases"])
    plan.setdefault("min_gain", 2.0)
    plan.setdefault("min_gain_lossy", promote.MIN_GAIN[False])
    plan.setdefault("min_pairs", promote.MIN_PAIRS)
    plan.setdefault("max_rounds", 8)
    plan.setdefault("lossy_policy", "exact-first")
    plan.setdefault("retest_benched", False)
    plan.setdefault("final_check", True)
    plan.setdefault("spec_extra", {})
    if plan["tier"] not in ("screen", "confirm"):
        raise PlanError("tier must be screen or confirm")
    if plan["lossy_policy"] not in POLICIES:
        raise PlanError(f"lossy_policy must be one of {', '.join(POLICIES)}")
    for k in plan["claim"]:
        if k not in plan["cases"]:
            raise PlanError(f"claim case {k!r} is not in cases")
    for k in plan["spec_extra"]:
        if k not in SPEC_EXTRA_KEYS:
            raise PlanError(f"spec_extra key {k!r} is not one of {', '.join(SPEC_EXTRA_KEYS)}")
    if plan["tier"] == "screen" and (plan["block"] or plan["outlier"] == "robust"):
        plan["block"], plan["outlier"], plan["warmup"] = 0, "gate", 0   # abtest refuses --block / --outlier robust on a screen
    cands, seen = [], set()
    for raw in plan.get("candidates") or []:
        c = dict(raw)
        n = c.get("name", "")
        if not NAME.fullmatch(n) or n == "base":
            raise PlanError(f"candidate name {n!r} must match {NAME.pattern} and not be 'base'")
        if n in seen:
            raise PlanError(f"candidate {n!r} appears twice")
        seen.add(n)
        c["lossy"] = bool(c.get("lossy", False))
        c.update(norm_delta(c))
        c.setdefault("env", {})
        c.setdefault("args", {})
        c["substitutes_of"] = list(c.get("substitutes_of") or [])
        c["interacts_with"] = list(c.get("interacts_with") or [])
        if not c.get("disabled") and not (c["env"] or c["args"]):
            raise PlanError(f"candidate {n!r} changes nothing (no env, no args)")
        cands.append(c)
    names = {c["name"] for c in cands}
    for c in cands:
        for k in ("substitutes_of", "interacts_with"):
            for o in c[k]:
                if o not in names or o == c["name"]:
                    raise PlanError(f"candidate {c['name']!r}: {k} names {o!r}, which is not another candidate")
    active = [c for c in cands if not c.get("disabled")]
    for i, a in enumerate(active):   # two candidates that set one key differently can only meet if they substitute for each other
        for b in active[i + 1:]:
            if a["name"] in b["substitutes_of"] or b["name"] in a["substitutes_of"]:
                continue
            for sect in ("env", "args"):
                for k in a[sect].keys() & b[sect].keys():
                    if a[sect][k] != b[sect][k]:
                        raise PlanError(f"candidates {a['name']!r} and {b['name']!r} set {k} to {a[sect][k]!r} and {b[sect][k]!r}: "
                                        "declare them substitutes_of each other, or one bundle would silently overwrite the other")
    plan["candidates"] = cands
    return plan


def plan_sha(plan):
    keep = {k: v for k, v in plan.items() if not k.startswith("_")}
    return hashlib.sha1(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:12]


def pool_of(plan, only=None, exclude=None, group=None):
    """The candidates the loop works with (disabled ones, lossy ones under 'skip', --exclude, --only and --group left out)."""
    out = []
    for c in plan["candidates"]:
        if c.get("disabled") or (c["lossy"] and plan["lossy_policy"] == "skip"):
            continue
        if group and c.get("group") != group:
            continue
        if only and c["name"] not in only or exclude and c["name"] in exclude:
            continue
        out.append(c)
    return out


# ------------------------------------------------------------------------------------------------ name check
def name_check(root, cands):
    """{(kind, name): [files]} for every STRATA_* env key and every --flag the candidates use: does it exist in src/ include/ serve/?"""
    found = {}
    for c in cands:
        names = [("env", k) for k in c["env"] if k.startswith("STRATA_")] + [("flag", k) for k in c["args"] if k.startswith("--")]
        for kind, n in names:
            if (kind, n) in found:
                continue
            cmd = ["grep", "-rlF", *[f"--include={g}" for g in SRC_GLOBS], "--", n, "src", "include", "serve"]
            r = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
            files = r.stdout.split()
            found[(kind, n)] = sorted(files, key=lambda f: (f.endswith(".def") or "_test" in f, f))   # code first: a registry line or a test is weaker proof
    return found


# ------------------------------------------------------------------------------------------------ specs
def fake_env(members, cand=None):
    """The synthetic effect of a bundle on the fake engine: gains add up; any lossy member changes the text."""
    allm = list(members) + ([cand] if cand else [])
    env = {"FAKE_GAIN_PCT": f"{sum(float((c.get('fake') or {}).get('gain_pct', 0.0)) for c in allm):g}"}
    if any(c["lossy"] for c in allm):
        env["FAKE_TEXT_SALT"] = "+".join(sorted(c["name"] for c in allm if c["lossy"]))
    return env


def build_spec(plan, members, tested, base_path=None, fake=False):
    """The abtest spec of one round: arm base = start bundle + members, one arm per tested candidate = base + candidate."""
    start = bundle_delta(plan["bundle_start"], members)
    arms = {"base": copy.deepcopy(start)}
    if fake:
        arms["base"] = merge(arms["base"], {"env": fake_env(members)})
    for c in tested:
        arm = merge(bundle_delta(plan["bundle_start"], members + [c]), {})
        if fake:
            arm = merge(arm, {"env": fake_env(members, c)})
        arms[c["name"]] = arm
    spec = {"base": base_path or plan["base"], "arms": arms, "cases": list(plan["cases"]), "reps": plan["reps"],
            "lossy": any(c["lossy"] for c in tested),
            "note": f"generated by ablate_greedy.py ({plan['name']}): base = {', '.join(c['name'] for c in members) or 'the start bundle'}; "
                    "one arm per remaining candidate"}
    spec.update(copy.deepcopy(plan["spec_extra"]))
    return spec


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def abtest_flags(plan, fake=False, force_busy=False, port=None):
    f = ["--tier", plan["tier"]]
    if plan["tier"] == "confirm":
        f += ["--no-early-stop"]   # an early stop can leave fewer than promote's MIN_PAIRS starts: every round runs all its reps
        if plan["block"]:
            f += ["--block", str(plan["block"])]
        if plan["warmup"]:
            f += ["--warmup", str(plan["warmup"])]
        f += ["--outlier", plan["outlier"]]
    f += ["--reps", str(plan["reps"]), "--reclaim-max", str(1e6 if fake else plan["reclaim_max"])]
    if fake:
        f += ["--foreign-cpu", "1000000", "--max-wait", "1"]
    if force_busy:
        f += ["--force-busy"]
    if port:
        f += ["--port", str(port)]
    return f + [str(x) for x in plan.get("abtest_args", [])]


def starts_per_arm(plan):
    if plan["tier"] == "screen":
        return 1
    return len(abtest.schedule(["x"], plan["reps"], plan["block"])[0])


# ------------------------------------------------------------------------------------------------ judging
def fin(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else x


def case_table(plan, jsonl, arms, m):
    """{arm: {case: {gain, lo, hi, n}}} paired against "base", with the Bonferroni alpha promote uses on claim cases."""
    records, _ = promote.read_records(jsonl)
    runs, _ = promote.select_runs(records)
    out = {}
    for arm in arms:
        out[arm] = {}
        if "base" not in runs or arm not in runs:
            continue
        for c in plan["cases"]:
            pairs = promote.case_pairs(runs["base"], runs[arm], c)
            if pairs:
                g = promote.gain_ci(pairs, promote.ALPHA / m if c in plan["claim"] else promote.ALPHA)
                out[arm][c] = {k: fin(g[k]) for k in ("gain", "lo", "hi", "n")}
    return out


def judge_round(plan, root, spec_path, jsonl, tested):
    """{candidate: {verdict, gain_lo, cases, reject, hold, lines, void}} by promote.judge_arm (confirm) or abtest's screen rule."""
    spec = json.loads(Path(spec_path).read_text())
    m = len(tested) * len(plan["claim"])
    table = case_table(plan, jsonl, [c["name"] for c in tested], max(m, 1))
    res = {}
    if plan["tier"] == "screen":
        for c in tested:
            d = {k: v["gain"] for k, v in table[c["name"]].items()}
            need = plan["min_gain_lossy"] if c["lossy"] else plan["min_gain"]
            ok = bool(d) and max(d.get(k, -1e9) for k in plan["claim"]) >= need and min(d.values()) > -promote.REGRESS_POINT
            res[c["name"]] = {"verdict": "PROMOTE" if ok else "REJECT", "gain_lo": max(d.values()) if d else None, "cases": table[c["name"]],
                              "reject": [] if ok else ["screen: no claim case at +%g%% or a case below -%g%%" % (need, promote.REGRESS_POINT)],
                              "hold": [], "lines": ["screen tier: one pair per case, a lead only"], "void": False}
        return res
    exempt = promote.sw.load_exempt(root, {})
    for c in tested:
        item = {"name": f"{plan['name']}:{c['name']}", "spec": str(spec_path), "exact": not c["lossy"], "branch": "ablate", "commits": [],
                "claim": plan["claim"], "gates": [], "parity": [{"waived": PARITY_WAIVED}], "runs": str(jsonl)}
        need = plan["min_gain_lossy"] if c["lossy"] else plan["min_gain"]
        found = promote.locate(Path(root), item, Path(jsonl).stem)
        a = promote.judge_arm(item, spec, found, c["name"], need, plan["min_pairs"], max(m, 1), exempt)
        g = a.gain
        res[c["name"]] = {"verdict": a.verdict(), "gain_lo": fin(g), "cases": table[c["name"]], "reject": list(a.reject), "hold": list(a.hold),
                          "lines": list(a.lines), "void": any("switch never acted" in h for h in a.hold)}
    return res


def regressed(r):
    return r["void"] or any(h.startswith("regression on") for h in r["hold"])


# ------------------------------------------------------------------------------------------------ the loop
class State:
    """What the loop knows: the bundle, the pool, and a note for every candidate that left it."""

    def __init__(self, plan, pool):
        self.plan, self.pool = plan, list(pool)
        self.initial = {c["name"] for c in pool}
        self.bundle, self.rounds, self.out = [], [], {}   # out: name -> {"state": dropped|benched|rejected, "why": str, "round": k}
        self.benched = {}   # name -> accepted count when it was benched
        self.accepted_round = {}

    def eligible(self):
        res = []
        for c in self.pool:
            if c["name"] in self.benched and not self.plan["retest_benched"]:
                since = [m["name"] for m in self.bundle[self.benched[c["name"]]:]]
                if not any(x in c["interacts_with"] for x in since):
                    continue
                self.benched.pop(c["name"])   # an interacting candidate was accepted since: test it again
            res.append(c)
        return res

    def choose(self, results, tested):
        clears = [c for c in tested if results[c["name"]]["verdict"] == "PROMOTE"]
        if self.plan["lossy_policy"] == "exact-first" and any(not c["lossy"] for c in clears):
            clears = [c for c in clears if not c["lossy"]]
        if not clears:
            return None
        return max(clears, key=lambda c: (results[c["name"]]["gain_lo"] or -math.inf, c["name"]))

    def accept(self, c, k):
        self.bundle.append(c)
        self.accepted_round[c["name"]] = k
        self.pool = [p for p in self.pool if p["name"] != c["name"]]
        for p in list(self.pool):
            if c["name"] in p["substitutes_of"] or p["name"] in c["substitutes_of"]:
                self.pool.remove(p)
                self.out[p["name"]] = {"state": "dropped", "round": k, "why": f"substitute of {c['name']} (accepted in round {k})"}

    def note_round(self, k, tested, results, n_before):
        """n_before: bundle size when the round started (a candidate benched now was measured without what the round accepted)."""
        for c in tested:
            r = results[c["name"]]
            if c["name"] in self.accepted_round or not any(p["name"] == c["name"] for p in self.pool):   # accepted, or dropped as a substitute
                continue
            if r["verdict"] == "REJECT" or regressed(r):
                self.benched[c["name"]] = n_before
                self.out[c["name"]] = {"state": "benched", "round": k, "why": first_reason(r)}
            elif c["name"] in self.out:
                self.out.pop(c["name"])   # undecided now


def first_reason(r):
    for k in ("reject", "hold"):
        if r[k]:
            return r[k][0]
    return ""


def describe_bundle(plan, members):
    return ", ".join(c["name"] for c in members) or "the start bundle"


class Runner:
    """Runs one abtest per round (real or against the fake root) and keeps the files."""

    def __init__(self, plan, out_dir, fake=False, force_busy=False, resume=False, log=print):
        self.plan, self.out, self.fake, self.force_busy, self.resume, self.log = plan, Path(out_dir), fake, force_busy, resume, log
        self.root = ROOT
        self.py = str(ROOT / ".venv/bin/python") if (ROOT / ".venv/bin/python").exists() else sys.executable
        self.base_path = None
        if fake:
            sys.path.insert(0, str(HERE / "probes"))
            import fake_engine as F
            self.root = F.make_fake_root(self.out / "fake-root", abtest=str(HERE / "abtest.py"), scale=float(os.environ.get("FAKE_TIME_SCALE", "0.05")))
            self.py = str(self.root / ".venv/bin/python")
            prof = {"startup_s": 2.0, "sd_start_pct": 0.3, "sd_request_pct": 0.3, **(plan.get("fake_profile") or {})}
            base = self.root / "configs/ablate-base.json"
            base.write_text(json.dumps({"exe": str(self.root / "bin/strata"), "args": ["--spec", "4"], "env": {"STRATA_RAM_KEEP_GIB": "12"},
                                        "log": str(self.root / "logs/ablate.log"), "fake": prof}))
            self.base_path = str(base)

    def run(self, k, members, tested, tag=None):
        """Writes the spec, runs abtest (unless a finished run is resumed), returns (spec_path, jsonl_path, returncode)."""
        tag = tag or f"round{k}"
        spec = build_spec(self.plan, members, tested, self.base_path, self.fake)
        spec_path, jsonl = self.out / f"{tag}.spec.json", self.out / f"{tag}.jsonl"
        done = self.out / f"{tag}.done"
        if self.resume and done.exists() and jsonl.exists():
            self.log(f"{tag}: finished run found, judged again without running")
            return spec_path, jsonl, 0
        if jsonl.exists():
            raise PlanError(f"{jsonl} exists without a finished marker: a run was cut off. Move it away, or use another --out-dir "
                            "(abtest appends to its file, so a partial run would be counted twice)")
        spec_path.write_text(json.dumps(spec, indent=1) + "\n")
        cmd = [self.py, str(self.root / "tools/pancake/abtest.py"), str(spec_path), "--out", str(jsonl),
               *abtest_flags(self.plan, self.fake, self.force_busy, free_port() if self.fake else self.plan.get("port"))]
        self.log(f"{tag}: {len(spec['arms'])} arms x {starts_per_arm(self.plan)} starts: " + " ".join(cmd))
        with open(self.out / f"{tag}.abtest.log", "w") as lf:
            p = subprocess.Popen(cmd, cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                lf.write(line)
            rc = p.wait()
        if rc == 0:
            done.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        return spec_path, jsonl, rc


def run_loop(plan, out_dir, pool, fake=False, force_busy=False, resume=False, no_final=False, log=print):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_file = out_dir / "state.json"
    if state_file.exists() and not resume:
        raise PlanError(f"{out_dir} already holds a run (state.json): use --resume to continue it, or another --out-dir")
    if state_file.exists() and json.loads(state_file.read_text()).get("plan_sha") != plan_sha(plan):
        raise PlanError("the plan changed since this --out-dir was started: judging old runs under a new plan is not allowed")
    runner = Runner(plan, out_dir, fake, force_busy, resume, log)
    st = State(plan, pool)
    status = "done"
    for k in range(1, plan["max_rounds"] + 1):
        tested = st.eligible()
        if not tested:
            status = "pool empty"
            break
        spec_path, jsonl, rc = runner.run(k, list(st.bundle), tested)
        if rc != 0:
            status = f"abtest failed in round {k} (exit {rc}: 3 = another test or engine is running, 2 = the server died twice)"
            st.rounds.append({"round": k, "bundle": [c["name"] for c in st.bundle], "tested": [c["name"] for c in tested], "failed": rc})
            break
        results = judge_round(plan, runner.root, spec_path, jsonl, tested)
        pick = st.choose(results, tested)
        n_before = len(st.bundle)
        rec = {"round": k, "bundle": [c["name"] for c in st.bundle], "tested": [c["name"] for c in tested], "results": results,
               "accepted": pick["name"] if pick else None, "spec": spec_path.name, "jsonl": jsonl.name, "dropped": []}
        if pick:
            before = set(st.out)
            st.accept(pick, k)
            rec["dropped"] = [{"name": n, **st.out[n]} for n in st.out if n not in before and st.out[n]["state"] == "dropped"]
        st.note_round(k, tested, results, n_before)
        st.rounds.append(rec)
        state_file.write_text(json.dumps({"plan_sha": plan_sha(plan), "rounds": st.rounds}, indent=1, default=str) + "\n")
        log(f"round {k}: " + (f"accepted {pick['name']} (lower bound {results[pick['name']]['gain_lo']:+.1f}%)" if pick else "no candidate clears the bar: stop"))
        if not pick:
            status = "no candidate clears the bar"
            break
    else:
        status = f"round limit ({plan['max_rounds']}) reached"
    final = None
    if plan["final_check"] and not no_final and st.bundle and not status.startswith("abtest failed"):
        final = final_check(plan, runner, st, log)
    state_file.write_text(json.dumps({"plan_sha": plan_sha(plan), "rounds": st.rounds, "status": status, "final": final}, indent=1, default=str) + "\n")
    return st, runner, status, final


def final_check(plan, runner, st, log):
    """One run: the start bundle against the final bundle (and the exact-only prefix when a lossy item was accepted)."""
    members = list(st.bundle)
    exact = exact_members(members)
    arms = [("bundle", members)] + ([("exact-only", exact)] if exact and len(exact) != len(members) else [])
    cands = []
    for name, ms in arms:
        cands.append({"name": name, "env": bundle_delta({}, ms).get("env", {}), "args": bundle_delta({}, ms).get("args", {}),
                      "lossy": any(c["lossy"] for c in ms), "substitutes_of": [], "interacts_with": [], "fake": {"gain_pct": sum(float((c.get("fake") or {}).get("gain_pct", 0)) for c in ms)}})
    spec_path, jsonl, rc = runner.run(0, [], cands, tag="final")
    if rc != 0:
        return {"failed": rc}
    res = judge_round(plan, runner.root, spec_path, jsonl, cands)
    log("final check: " + ", ".join(f"{n} vs start {fmt_cases(r['cases'])}" for n, r in res.items()))
    return {"results": res}


def exact_members(members):
    """The exact items before the first lossy one, plus the exact items after it (those were measured WITH the lossy one: noted in RESULTS.md)."""
    return [c for c in members if not c["lossy"]]


# ------------------------------------------------------------------------------------------------ output
def fmt_g(g):
    if not g or g.get("gain") is None:
        return "n/a"
    lo = "n/a" if g["lo"] is None else f"{g['lo']:+.1f}"
    hi = "n/a" if g["hi"] is None else f"{g['hi']:+.1f}"
    return f"{g['gain']:+.1f}% [{lo}, {hi}] n={g['n']}"


def fmt_cases(cases):
    return "; ".join(f"{c} {fmt_g(g)}" for c, g in cases.items()) or "no data"


def apply_preset(plan, runner, members, path):
    """The start preset with the bundle applied (abtest.make_config), written to `path`; None when the preset cannot be read."""
    base = runner.base_path or (plan["base"] if os.path.isabs(plan["base"]) else str(ROOT / plan["base"]))
    if not Path(base).is_file():
        return None
    abtest.make_config(base, bundle_delta(plan["bundle_start"], members), str(path))
    return path


def md_cell(s, n=140):
    s = str(s).replace("|", "/").replace("\n", " ")
    return s if len(s) <= n else s[: n - 3] + "..."


def render(plan, st, runner, status, final, fake, command, pool_all, out_dir="."):
    day = time.strftime("%Y-%m-%d")
    L = [f"# Greedy ablation {plan['name']} ({day})", ""]
    if fake:
        L += ["**FAKE ENGINE: every number below is synthetic (tools/pancake/probes/fake_engine.py). Nothing was measured.**", ""]
    L += [f"Status: {status}.  Rounds run: {len([r for r in st.rounds if 'results' in r])}.", ""]
    L += ["## Setup", "",
          f"- base preset: `{plan['base']}`" + (f", start delta `{json.dumps(plan['bundle_start'])}`" if plan["bundle_start"] else ""),
          f"- cases {plan['cases']}, claim cases {plan['claim']}, tier {plan['tier']}, reps {plan['reps']}, block {plan['block']}, "
          f"warm-up {plan['warmup']}, outlier {plan['outlier']}, reclaim-max {plan['reclaim_max']:g}, starts per arm {starts_per_arm(plan)}",
          f"- bar: lower bound of the 95% interval (Student t, Bonferroni over arms x claim cases, promote.judge_arm) >= {plan['min_gain']:g}% "
          f"for exact items, {plan['min_gain_lossy']:g}% for lossy ones; at least {plan['min_pairs']} pairs per claim case; lossy policy {plan['lossy_policy']}",
          "- parity and quality are not judged here: `exact` is the item's declaration (check it with the equivalence gate); lossy items go to the quality gate",
          f"- command: `{command}`", ""]
    L += ["## Bundle (in the order accepted)", ""]
    if st.bundle:
        L += ["| # | Item | Kind | Round | Lower bound on its round | Env / args |", "| ---: | --- | --- | ---: | ---: | --- |"]
        for i, c in enumerate(st.bundle, 1):
            k = st.accepted_round[c["name"]]
            r = next(x for x in st.rounds if x["round"] == k)["results"][c["name"]]
            L.append(f"| {i} | {c['name']} | {'lossy' if c['lossy'] else 'exact'} | {k} | {r['gain_lo']:+.1f}% | `{md_cell(json.dumps({s: c[s] for s in ('env', 'args') if c[s]}), 200)}` |")
    else:
        L.append("Nothing was accepted: the start bundle stays as it is.")
    L.append("")
    for k, r in enumerate(st.rounds):
        if "results" not in r:
            L += [f"## Round {r['round']}", "", f"abtest failed (exit {r['failed']}); no result.", ""]
            continue
        L += [f"## Round {r['round']}", "", f"base = {describe_bundle(plan, [c for c in st.bundle if st.accepted_round[c['name']] < r['round']])}; "
              f"{len(r['tested'])} candidates ({r['spec']}, {r['jsonl']}).  Gain = paired, in % of base; interval is the 95% one used for the verdict.", "",
              "| Candidate | Kind | Verdict | " + " | ".join(plan["cases"]) + " | Note |", "| --- | --- | --- | " + " | ".join("---" for _ in plan["cases"]) + " | --- |"]
        for n in r["tested"]:
            c = next(x for x in pool_all if x["name"] == n)
            x = r["results"][n]
            note = ("ACCEPTED. " if r["accepted"] == n else "") + ("switch never acted. " if x["void"] else "") + first_reason(x)
            L.append(f"| {n} | {'lossy' if c['lossy'] else 'exact'} | {x['verdict']} | " + " | ".join(fmt_g(x["cases"].get(cs)) for cs in plan["cases"]) + f" | {md_cell(note)} |")
        L.append("")
        L.append(f"Decision: " + (f"accept **{r['accepted']}**" if r["accepted"] else "no candidate clears the bar: stop") + ".")
        for d in r["dropped"]:
            L.append(f"- dropped without a test: {d['name']} ({d['why']})")
        L.append("")
    L += ["## Interactions", ""]
    declared = [(c, o) for c in pool_all for o in c["substitutes_of"]]
    L += ["Declared substitutes (symmetric: accepting one drops the other):", ""]
    if declared:
        L += ["| Pair | Outcome |", "| --- | --- |"]
        seen = set()
        for c, o in declared:
            key = tuple(sorted((c["name"], o)))
            if key in seen:
                continue
            seen.add(key)
            a, b = key
            acc = [x for x in key if x in st.accepted_round]
            fate = (f"{acc[0]} accepted (round {st.accepted_round[acc[0]]}), the other dropped untested" if acc else
                    "neither accepted: " + "; ".join(f"{x}: {st.out.get(x, {}).get('why', 'still in the pool or never tested')}" for x in key))
            L.append(f"| {a} vs {b} | {md_cell(fate, 220)} |")
    else:
        L.append("none declared.")
    L += ["", "Measured: each candidate's gain the first round it ran (alone on the start bundle) and the last round it ran (on the bundle then).  "
          "A difference far outside both intervals is an interaction; overlapping intervals say nothing.", ""]
    tested_in = {}
    for r in st.rounds:
        for n, x in (r.get("results") or {}).items():
            tested_in.setdefault(n, []).append((r["round"], x))
    rows = [(n, v) for n, v in tested_in.items() if len(v) > 1]
    if rows:
        L += ["| Candidate | first round (alone) | last round (on bundle) | Bundle then |", "| --- | --- | --- | --- |"]
        for n, v in rows:
            (k1, x1), (k2, x2) = v[0], v[-1]
            cs = plan["claim"][0]
            L.append(f"| {n} | r{k1} {fmt_g(x1['cases'].get(cs))} ({cs}) | r{k2} {fmt_g(x2['cases'].get(cs))} | {describe_bundle(plan, [c for c in st.bundle if st.accepted_round[c['name']] < k2])} |")
    else:
        L.append("No candidate ran in more than one round.")
    L += ["", "## Not in the bundle", "", "| Item | Fate | Why |", "| --- | --- | --- |"]
    for c in pool_all:
        if c["name"] in st.accepted_round:
            continue
        o = st.out.get(c["name"])
        if c.get("disabled"):
            L.append(f"| {c['name']} | disabled | {md_cell(c['disabled'], 220)} |")
        elif o:
            L.append(f"| {c['name']} | {o['state']} (round {o['round']}) | {md_cell(o['why'], 220)} |")
        elif c["lossy"] and plan["lossy_policy"] == "skip":
            L.append(f"| {c['name']} | skipped | lossy_policy is skip |")
        elif c["name"] not in st.initial:
            L.append(f"| {c['name']} | left out | not in this run's pool (--only / --exclude / --group) |")
        elif c["name"] in tested_in:
            x = tested_in[c["name"]][-1][1]
            L.append(f"| {c['name']} | undecided | {md_cell(first_reason(x) or 'never cleared the bar', 220)} |")
        else:
            L.append(f"| {c['name']} | not tested | the loop ended first |")
    lossy_in = [c for c in st.bundle if c["lossy"]]
    L += ["", "## Lossy items: quality-gate queue", ""]
    if lossy_in:
        L += ["These cleared the SPEED bar but change the output.  They are NOT promoted by this tool: run the quality gate before any preset uses them.", "",
              "| Item | Round | Why lossy |", "| --- | ---: | --- |"]
        for c in lossy_in:
            L.append(f"| {c['name']} | {st.accepted_round[c['name']]} | {md_cell(c.get('note', ''), 200)} |")
        after = [c["name"] for c in st.bundle if not c["lossy"] and st.accepted_round[c["name"]] > min(st.accepted_round[x["name"]] for x in lossy_in)]
        if after:
            L += ["", f"Exact items accepted AFTER a lossy one were measured with it ({', '.join(after)}): confirm them again on the exact-only preset."]
        L += ["", "```", f"python tools/pancake/quality_gate.py {out_dir}/quality-gate-spec.json --out bench/quality-hip/<day>-ablation --set bench/quality/set-v2.json --dry-run   (then without --dry-run)", "```", ""]
    else:
        L += ["No lossy item was accepted.", ""]
    L += ["## Final check: start bundle against the final bundle", ""]
    if final and "results" in final:
        L += ["| Arm | Kind | " + " | ".join(plan["cases"]) + " |", "| --- | --- | " + " | ".join("---" for _ in plan["cases"]) + " |"]
        for n, x in final["results"].items():
            L.append(f"| {n} | {'contains lossy items' if n == 'bundle' and lossy_in else 'exact only'} | " + " | ".join(fmt_g(x["cases"].get(cs)) for cs in plan["cases"]) + " |")
        L.append("")
        L.append("A stack that gains less than the sum of its rounds shows an interaction the rounds missed: look at the table above.")
    elif final:
        L.append(f"The final run failed (exit {final['failed']}).")
    else:
        L.append("Not run (no bundle, final_check off, --no-final, or an earlier failure).")
    L += ["", "## Files", "", "`final-preset.json` (start preset + the whole bundle), `final-preset-exact.json` (exact items only, when a lossy one was accepted), "
          "`quality-gate-spec.json` (lossy items, for quality_gate.py), `round<k>.spec.json` / `.jsonl` (abtest), `state.json`.", ""]
    return "\n".join(L)


def write_outputs(plan, st, runner, status, final, fake, command, out_dir, pool_all):
    out_dir = Path(out_dir)
    (out_dir / "RESULTS.md").write_text(render(plan, st, runner, status, final, fake, command, pool_all, out_dir))
    written = []
    if st.bundle:
        if apply_preset(plan, runner, st.bundle, out_dir / "final-preset.json"):
            written.append("final-preset.json")
        ex = exact_members(st.bundle)
        if len(ex) != len(st.bundle) and apply_preset(plan, runner, ex, out_dir / "final-preset-exact.json"):
            written.append("final-preset-exact.json")
        (out_dir / "final-delta.json").write_text(json.dumps({"start": plan["bundle_start"], "bundle": [c["name"] for c in st.bundle],
                                                              "delta": bundle_delta(plan["bundle_start"], st.bundle),
                                                              "exact_only_delta": bundle_delta(plan["bundle_start"], exact_members(st.bundle))}, indent=1) + "\n")
    lossy = [c for c in st.bundle if c["lossy"]]
    if lossy:
        ex = [c for c in st.bundle if not c["lossy"]]
        arms = {"base": bundle_delta(plan["bundle_start"], ex)}
        for c in lossy:
            arms[c["name"]] = bundle_delta(plan["bundle_start"], ex + [c])
        spec = {"base": runner.base_path or plan["base"], "arms": arms, "cases": plan["cases"], "reps": plan["reps"], "lossy": True,
                "note": "generated by ablate_greedy.py: the lossy items of the bundle, each on top of the exact-only bundle; first arm = stock for the gate"}
        (out_dir / "quality-gate-spec.json").write_text(json.dumps(spec, indent=1) + "\n")
    return written


# ------------------------------------------------------------------------------------------------ dry run
def dry_run(plan, pool, root, out=print):
    out(f"plan {plan['name']}: base {plan['base']}, tier {plan['tier']}, cases {plan['cases']}, claim {plan['claim']}")
    out(f"  abtest flags: {' '.join(abtest_flags(plan, port=plan.get('port')))}")
    out(f"  bar: lower bound >= {plan['min_gain']:g}% exact, {plan['min_gain_lossy']:g}% lossy, >= {plan['min_pairs']} pairs; "
        f"lossy policy {plan['lossy_policy']}; max rounds {plan['max_rounds']}")
    out("candidates:")
    for c in plan["candidates"]:
        why = c.get("disabled")
        kind = "DISABLED" if why else "lossy" if c["lossy"] else "exact"
        tag = "" if not why else f" ({why})"
        if not why and c not in pool:
            tag = " (left out: lossy_policy skip, --only, --exclude or --group)"
        out(f"  {c['name']:22s} {kind:8s} env={json.dumps(c['env'])} args={json.dumps(c['args'])}"
            + (f" substitutes_of={c['substitutes_of']}" if c["substitutes_of"] else "")
            + (f" interacts_with={c['interacts_with']}" if c["interacts_with"] else "") + tag)
    found = name_check(root, plan["candidates"])
    out("name check (grep in src/ include/ serve/):")
    for (kind, n), files in sorted(found.items()):
        weak = bool(files) and (files[0].endswith(".def") or "_test" in files[0])
        out(f"  {'MISSING' if not files else 'weak   ' if weak else 'FOUND  '} {kind:4s} {n}" + (f"   {files[0]}" if files else ""))
    if not pool:
        out("nothing to run: the pool is empty")
        return 0
    spec = build_spec(plan, [], pool)
    arms = list(spec["arms"])
    order, passes = abtest.schedule(arms, plan["reps"], plan["block"]) if plan["tier"] == "confirm" else ([(0, a) for a in arms], 1)
    out(f"round 1 spec (round k > 1 starts from the bundle accepted so far and drops what was benched or substituted):")
    out(json.dumps(spec, indent=1))
    out(f"round 1 run order ({len(order)} engine starts, {passes} pass(es) per start): " + " ".join(f"r{r}:{a}" for r, a in order))
    spa = starts_per_arm(plan)
    n = len(pool)
    worst = sum((n - i + 1) * spa for i in range(min(plan["max_rounds"], n)))
    out(f"engine starts: round 1 = {len(arms)} arms x {spa} = {len(arms) * spa}; worst case (one candidate accepted per round, none benched) {worst} "
        f"in {min(plan['max_rounds'], n)} rounds, plus {2 * spa} for the final check; each round stops the loop if nothing clears.")
    return 0


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("plan")
    ap.add_argument("--out-dir")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, the name check, round 1's spec and the run order; start nothing")
    ap.add_argument("--fake", action="store_true", help="run the loop on the fake engine (synthetic numbers, no GPU, seconds)")
    ap.add_argument("--force-busy", action="store_true", help="pass --force-busy to abtest (fake runs only: it starts although another test runs)")
    ap.add_argument("--resume", action="store_true", help="continue an --out-dir: finished rounds are judged again from their files, not rerun")
    ap.add_argument("--only", help="comma separated candidate names: test only these")
    ap.add_argument("--exclude", help="comma separated candidate names to leave out")
    ap.add_argument("--group", help="test only the candidates whose plan 'group' is this (the example plan has core and extra)")
    ap.add_argument("--max-rounds", type=int)
    ap.add_argument("--no-final", action="store_true", help="skip the final start-vs-bundle run")
    a = ap.parse_args(argv)
    if a.force_busy and not a.fake:
        ap.error("--force-busy is for --fake runs only: a real run beside another test would give wrong numbers")
    try:
        plan = load_plan(a.plan)
        if a.max_rounds:
            plan["max_rounds"] = a.max_rounds
        only = set(a.only.split(",")) if a.only else None
        exclude = set(a.exclude.split(",")) if a.exclude else None
        known = {c["name"] for c in plan["candidates"]}
        for n in (only or set()) | (exclude or set()):
            if n not in known:
                raise PlanError(f"--only / --exclude names {n!r}, which is not a candidate")
        pool = pool_of(plan, only, exclude, a.group)
        if a.dry_run:
            return dry_run(plan, pool, ROOT)
        day = time.strftime("%Y-%m-%d")
        out_dir = Path(a.out_dir) if a.out_dir else ROOT / "bench/results" / f"{day}-ablate-{plan['name']}"
        command = "python tools/pancake/ablate_greedy.py " + " ".join(sys.argv[1:] if argv is None else argv)
        st, runner, status, final = run_loop(plan, out_dir, pool, a.fake, a.force_busy, a.resume, a.no_final)
        written = write_outputs(plan, st, runner, status, final, a.fake, command, out_dir, plan["candidates"])
        print(f"\n{status}.  bundle: {describe_bundle(plan, st.bundle)}.  wrote {out_dir / 'RESULTS.md'}" + "".join(f", {w}" for w in written))
        return 2 if status.startswith("abtest failed") else 0
    except PlanError as e:
        print(f"ablate_greedy: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
