"""promote.py - which tested items go into port/rdna3-decode (owner rule, 2026-11-11: only items that show value).

Reads what the test tools already wrote under bench/results and prints PROMOTE / HOLD / REJECT per item, with the numbers.
It only reads: it never runs git, never writes a file and never touches port/rdna3-decode.  For PROMOTE items it prints the
`git cherry-pick` command; the owner runs it in the checkout that has port/rdna3-decode checked out.

    python tools/pancake/promote.py ITEMS.json [--only NAME,NAME] [--root REPO]
    python tools/pancake/promote.py --noise bench/results/2026-11-10-test-phase/*.jsonl     (noise and pairs needed, see below)

ITEMS.json: {"items": [ITEM, ...]}.  ITEM keys (the first five are required; the catalog is bench/promotion/phase16.json):
  name, spec (bench/specs/*.json), exact (true / false: the owner's flag), branch, commits (oldest first, may be empty until merged)
  arms         arms of the spec that implement the item (default: every arm but the first; the first arm is the reference)
  claim        cases the item claims a gain on (default: every case of the spec).  The other cases are only checked for regressions.
  vs           the arm the gain is measured against (default: the first arm).  Parity and token forcing always compare with the first arm
               (stock); use vs for an item that sits on top of another switch (F21 over FUSE_WAIT) or for a spec with a non-base reference
  decides      "abtest" (default: abtest.py jsonl), "replay" (replay_turns.py: cache items; abtest only checks side effects) or
               "taskgate" (task_gate.py time-to-success curve: thinking and agent-loop items)
  gates        lossy items, default ["token_forcing", "task_gate"] ("task_gate" only when decides is "taskgate"); exact: []
  parity       exact items, default ["text"]; also {"restore": FILE}, {"ctest": FILE, "tests": [names]}, {"waived": "reason"}
  min_gain, min_pairs   per-item override of the thresholds below (printed when used)
  tested_rev   short hash the results must come from (rev.txt of the result directory); runs, quality, taskgate, replay: paths or
               globs relative to the repo (default: the newest bench/results/*/<spec stem>.jsonl, -quality, -taskgate, -replay)

Result layout read (tools/pancake/test_phase.sh writes the first two; run task_gate.py / replay_turns.py with
--out <results dir>/<stem>-taskgate and <stem>-replay):  <stem>.jsonl (abtest), <stem>-quality/verdicts.json and <arm>.verdict.json
(quality_gate), <stem>-taskgate/summary.json, <stem>-replay/summary.json, rev.txt, restore-check*.txt, ctest-gpu.txt.

Verdict.  Any REJECT reason -> REJECT; else any HOLD reason -> HOLD; else PROMOTE.  An item with several arms gets its best arm.
Thresholds (constants below; each cites where it comes from):
  gain        the 95% lower bound of the paired gain on at least one claim case must reach MIN_GAIN: 1% of base for exact items,
              3% for lossy ones (SpecPilot, arXiv 2611.02501 sec. 3.2: promote only when the gain's 95% lower bound exceeds 1%, "~3%"
              when the outputs diverge).  SpecPilot adds twice its stock-vs-stock noise (< 0.1% on its H100); the paired interval is
              this tool's noise term (a 0.1% offset is not measurable on this box: the interval half-width is 2-7%).  The interval
              is Student-t over runs paired by repetition, Bonferroni over (arms x claim cases).
  no gain     every claim case has an upper bound below MIN_GAIN -> REJECT (abtest's own stop rule: "no effect worth having").
  open        otherwise HOLD, with the number of pairs that would settle it at the observed paired SD, or a note that it is too many.
  pairs       at least MIN_PAIRS (3) clean pairs per claim case; a run with foreign CPU or memory reclaim noise (abtest's flag) is not clean.
  regression  on every case: upper bound below -1% (REGRESS_CI) or point estimate <= -3% (REGRESS_POINT, abtest --tier screen's rule)
              -> HOLD (mixed result: fix it or the owner decides).
  parity      exact item: the generated text hash must equal base on every run (abtest texts) -> else REJECT ("declared exact, text
              differs": fix it or flag it lossy and pass the lossy gates).  Sampled specs have no text parity: give another parity.
  token forcing   (tools/force_check.py, SpecPilot sec. 3.3: rate rule +0.5 points at 95%, large-margin event rule) PASS needed,
              REJECT -> REJECT, INCONCLUSIVE -> HOLD.  The paper uses it on lossless systems (bugs, reward hacking); on a lossy arm a
              PASS only means "no large deviation".  HOLD when the stock-vs-stock spread exceeds the 0.5-point margin.
  task gate   (task_gate.py) BETTER or PASS needed (success non-inferior within its margin and tool-call validity): "Displacement Is Not
              Direction" (arXiv 2606.19558) shows KLD / top-1 do not rank close variants, so lossy arms are judged by task score too.
  --noise     per case: base CV, paired SD per arm, and the pairs needed to see a 1% / 3% gain, from existing abtest jsonl files.
"""
import argparse
import glob
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abtest  # noqa: E402   (metric(): which engine figure judges a case; importing it reads sysfs only)

ROOT = Path(__file__).resolve().parents[2]
ALPHA = 0.05
MIN_PAIRS = 3
MIN_GAIN = {True: 1.0, False: 3.0}   # % of base, by item.exact
REGRESS_CI = 1.0
REGRESS_POINT = 3.0
STOCK_SPREAD = 0.5                   # points: token forcing's rate-rule margin
HEX = re.compile(r"[0-9a-f]{7,40}")
NAME = re.compile(r"[A-Za-z0-9._/+-]+")
ORDER = ["REJECT", "HOLD", "PROMOTE"]   # worst first


# ------------------------------------------------------------------------------------------------ statistics
def _betacf(a, b, x):
    """Continued fraction of the incomplete beta function (Numerical Recipes betacf)."""
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        for aa in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)), -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-14:
            break
    return h


def t_cdf(t, df):
    """P(T <= t) for Student's t with df degrees of freedom, through the regularized incomplete beta function."""
    x = df / (df + t * t)
    lead = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - math.lgamma(0.5) + (df / 2) * math.log(x) + 0.5 * math.log1p(-x))
    if x < (df / 2 + 1) / (df / 2 + 0.5 + 2):
        tail = lead * _betacf(df / 2, 0.5, x) / (df / 2)
    else:
        tail = 1.0 - lead * _betacf(0.5, df / 2, 1 - x) / 0.5
    tail /= 2
    return 1 - tail if t > 0 else tail


def t_quantile(p, df):
    """The t with P(T <= t) = p (p > 0.5), by bisection on t_cdf (abtest.T95 stops at df 10 and then uses 2.0)."""
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < p:
        hi *= 2
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if t_cdf(mid, df) < p else (lo, mid)
    return (lo + hi) / 2


def gain_ci(pairs, alpha=ALPHA, lower_is_better=False):
    """pairs [(base, arm)] paired by repetition -> {n, gain, lo, hi, sd} in % of the base mean (positive = better), t interval."""
    n = len(pairs)
    bm = sum(b for b, _ in pairs) / n
    d = [(b - x if lower_is_better else x - b) / bm * 100 for b, x in pairs]
    m = sum(d) / n
    if n < 2:
        return {"n": n, "gain": m, "lo": -math.inf, "hi": math.inf, "sd": math.nan}
    sd = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1))
    h = t_quantile(1 - alpha / 2, n - 1) * sd / math.sqrt(n)
    return {"n": n, "gain": m, "lo": m - h, "hi": m + h, "sd": sd}


def pairs_needed(sd, target, alpha=ALPHA, cap=200):
    """Smallest number of pairs whose interval half-width is <= target (same unit as sd); None above cap."""
    for n in range(2, cap + 1):
        if t_quantile(1 - alpha / 2, n - 1) * sd / math.sqrt(n) <= target:
            return n
    return None


def fmt(g, unit="%"):
    return f"{g['gain']:+.1f}{unit} [{g['lo']:+.1f}, {g['hi']:+.1f}] n={g['n']}"


# ------------------------------------------------------------------------------------------------ readers
def read_runs(path):
    """abtest jsonl -> ({arm: {rep: record}}, discarded, problems): the record abtest used per (arm, rep) = the first clean
    attempt, else the quietest (tools/pancake/abtest.py main)."""
    attempts, problems = {}, []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            slot = attempts.setdefault((r["arm"], r["rep"]), {})
            if r["attempt"] in slot:
                problems.append(f"{r['arm']} rep {r['rep']} attempt {r['attempt']} appears twice (a rerun was appended to the file)")
            slot[r["attempt"]] = r
    runs, discarded = {}, 0
    for (arm, rep), slot in attempts.items():
        recs = [slot[k] for k in sorted(slot)]
        used = next((r for r in recs if r["clean"]), None) or min(recs, key=lambda r: r["noise"])
        runs.setdefault(arm, {})[rep] = used
        discarded += len(recs) - 1
    return runs, discarded, problems


def case_value(rec, case):
    v = [r[abtest.metric(case)] for r in rec["rows"] if r["case"] == case]
    return sum(v) / len(v) if v else None


def case_pairs(base, arm, case):
    """[(base, arm)] for the repetitions both arms have; base, arm: {rep: record}."""
    out = [(case_value(base[r], case), case_value(arm[r], case)) for r in sorted(base.keys() & arm.keys())]
    return [(b, x) for b, x in out if b is not None and x is not None]


def load_json(path):
    return json.loads(Path(path).read_text()) if path and Path(path).is_file() else None


def read_replay(path):
    """replay_turns.py summary.json -> (base arm, {arm: {rep: {prompt_s, clean}}})."""
    s = load_json(Path(path) / "summary.json")
    if not s:
        return None, {}
    return s["arms"][0], {a: {r["rep"]: {"prompt_s": r["totals"]["prompt_s"], "clean": r["clean"]} for r in rs} for a, rs in s["runs"].items()}


def read_restore(path):
    """restore_check.py output -> (matches, differences, exit code of the 'restore check ... exit N' line test_phase.sh appends)."""
    text = Path(path).read_text()
    rows = re.findall(r"^(\S+ .+): (MATCH|DIFFERENT)$", text, re.MULTILINE)
    code = re.findall(r"^restore check.* exit (\d+)$", text, re.MULTILINE)
    return [r for r, v in rows if v == "MATCH"], [r for r, v in rows if v == "DIFFERENT"], int(code[-1]) if code else None


def read_ctest(path):
    """ctest output -> ({test: passed}, exit code of the 'ctest ... exit N' line test_phase.sh appends or None)."""
    text = Path(path).read_text()
    tests = {m[1]: "Passed" in m[2] for m in re.finditer(r"Test\s+#\d+:\s+(\S+)\s+\.+(.*)", text)}
    code = re.findall(r"exit (\d+)\s*$", text.strip(), re.MULTILINE)
    return tests, int(code[-1]) if code else None


# ------------------------------------------------------------------------------------------------ one arm
class Arm:
    def __init__(self, name):
        self.name, self.lines, self.reject, self.hold, self.gain = name, [], [], [], None

    def verdict(self):
        return "REJECT" if self.reject else "HOLD" if self.hold else "PROMOTE"


def settle_gain(a, labelled, claim, need, min_pairs, alpha, unit="%"):
    """labelled: {label: gain_ci dict}.  Records the gain finding on arm a from the claim labels, the regressions from all."""
    for label, g in labelled.items():
        a.lines.append(f"{'claim ' if label in claim else 'other '} {label:14s} {fmt(g, unit)}")
        if g["n"] >= 2 and (g["hi"] < -REGRESS_CI or g["gain"] <= -REGRESS_POINT):
            a.hold.append(f"regression on {label}: {g['gain']:+.1f}{unit} [{g['lo']:+.1f}, {g['hi']:+.1f}] "
                          f"(a loss below -{REGRESS_CI:g}{unit} is certain or the point estimate is <= -{REGRESS_POINT:g}{unit})")
    cl = {k: g for k, g in labelled.items() if k in claim and g["n"] >= min_pairs}
    a.gain = max((g["lo"] for g in cl.values()), default=-math.inf)
    if not cl:
        have = ", ".join(f"{k} n={labelled[k]['n']}" for k in claim if k in labelled) or "no runs"
        a.hold.append(f"fewer than {min_pairs} clean pairs on every claim case ({have})")
    elif any(g["lo"] >= need for g in cl.values()):
        a.lines.append(f"gain   lower bound >= {need:g}{unit}: shown")
    elif all(g["hi"] < need for g in cl.values()):
        a.reject.append(f"no gain worth having: the upper bound is below {need:g}{unit} on every claim case "
                        f"({', '.join(k + ' ' + format(g['hi'], '+.1f') for k, g in cl.items())})")
    else:
        k, g = max(cl.items(), key=lambda kv: kv[1]["gain"])
        far = abs(g["gain"] - need)
        if g["sd"] != g["sd"]:   # a task gate interval: no paired SD to extrapolate from
            todo = "more tasks or repetitions"
        else:
            n = pairs_needed(g["sd"], far, alpha) if far > 0 else None
            todo = (f"about {n} pairs would settle it at the observed paired SD {g['sd']:.1f}{unit}" if n else
                    f"more than 200 pairs at the observed paired SD {g['sd']:.1f}{unit}: too close to the threshold, the noise is too high, or both")
        a.hold.append(f"gain open on {k}: {fmt(g, unit)} straddles {need:g}{unit}; {todo}")


def noisy_runs(a, recs):
    bad = [r for r in recs if not r["clean"]]
    if bad:
        a.hold.append(f"{len(bad)} of {len(recs)} runs used were noisy (foreign CPU / memory reclaim: noise "
                      f"{max(r.get('noise', 0) for r in bad):.0f}%): rerun on a quiet machine")


def judge_text(a, spec, runs, base, arm):
    """Exact item: the text hash of every non-prefill case equals base's on every run."""
    cases = [c for c in spec["cases"] if any(c in r.get("texts", {}) for r in runs[arm].values())]
    if spec.get("temperature") or not cases:
        a.hold.append("text parity not checkable (" + ("sampled spec" if spec.get("temperature") else "no text hashes in the jsonl: an abtest "
                      "older than 2026-11-10, or only prefill cases") + "): rerun, or give the item another parity (restore / ctest / waived)")
        return
    bad, unstable = [], []
    for c in cases:
        ref = {r["texts"][c] for r in runs[base].values() if c in r.get("texts", {})}
        if len(ref) != 1:
            unstable.append(c)
        elif any(r["texts"].get(c) != next(iter(ref)) for r in runs[arm].values()):
            bad.append(c)
    if unstable:
        a.hold.append(f"base text on {', '.join(unstable)} is not the same on every run: text parity not checkable")
    if bad:
        a.reject.append(f"declared exact but the text differs from base on {', '.join(bad)}: fix it, or flag the item lossy and pass the lossy gates")
    if not (bad or unstable):
        a.lines.append(f"parity text identical to base on {', '.join(cases)} ({len(runs[arm])} runs)")


def judge_parity_file(a, kind, p, base_dir):
    path = base_dir / p[kind] if base_dir else None
    if not path or not path.is_file():
        a.hold.append(f"parity {kind}: {p[kind]} not found next to the results")
    elif kind == "restore":
        ok, diff, code = read_restore(path)
        if diff:
            a.reject.append(f"parity restore: {len(diff)} turns DIFFERENT ({', '.join(diff[:4])})")
        elif not ok or code == 2:
            a.hold.append("parity restore: no MATCH lines or a setup error (exit 2): not run")
        else:
            a.lines.append(f"parity restore {len(ok)} turns MATCH")
    else:
        tests, code = read_ctest(path)
        want = p.get("tests") or []
        missing = [t for t in want if t not in tests]
        failed = [t for t in (want or tests) if t in tests and not tests[t]]
        if failed or (not want and code not in (None, 0)):
            a.reject.append(f"parity ctest: failed {', '.join(failed) or 'exit ' + str(code)}")
        elif missing or (not want and code is None):
            a.hold.append(f"parity ctest: {', '.join(missing) or 'no exit line'} not in {p[kind]}")
        else:
            a.lines.append(f"parity ctest {len(want) or len(tests)} tests passed")


def judge_token_forcing(a, qdir, arm):
    verdicts = load_json(Path(qdir) / "verdicts.json") if qdir else None
    if verdicts is None:
        a.hold.append("token forcing not run (no verdicts.json): quality_gate.py")
        return
    v = verdicts.get(arm, "not scored")
    d = load_json(Path(qdir) / f"{arm}.verdict.json") or {}
    if d:
        s = d["mismatch_rate_stock_arms_pct"]
        a.lines.append(f"token forcing {v}: mismatch {d['mismatch_rate_cand_pct']:.2f}% vs stock {s[0]:.2f}% / {s[-1]:.2f}%, diff {d['diff_pts']:+.2f} pts "
                       f"[{d['diff_ci90_pts'][0]:+.2f}, {d['diff_ci90_pts'][1]:+.2f}] (90% two-sided = 95% one-sided), large-margin events "
                       f"{d['events_cand']} vs {d['events_stock_arms']}, {d['requests']} requests / {d['tokens_cand']} tokens")
        if max(s) - min(s) > STOCK_SPREAD:
            a.hold.append(f"token forcing: the two stock runs differ by {max(s) - min(s):.2f} pts, more than the {STOCK_SPREAD} pt margin: "
                          "the gate cannot resolve")
    else:
        a.lines.append(f"token forcing {v}")
    if v == "REJECT":
        a.reject.append("token forcing REJECT")
    elif v != "PASS":
        half = (d["diff_ci90_pts"][1] - d["diff_ci90_pts"][0]) / 2 if d else None
        extra = ""
        if v == "INCONCLUSIVE" and half:
            extra = f": interval half-width {half:.2f} pts at {d['requests']} requests; halving it takes 4x the requests (bench/quality/set.json)"
        a.hold.append(f"token forcing {v}{extra}")


def judge_task_gate(a, summary, arm):
    e = ((summary or {}).get("arms") or {}).get(arm)
    if not e or "verdict" not in e:
        a.hold.append("task gate not run for this arm (task_gate.py)")
        return
    d = e.get("success_diff") or {}
    a.lines.append(f"task gate {e['verdict']}: success {d.get('diff', 0) * 100:+.1f} pts [{d.get('lo', 0) * 100:+.1f}, {d.get('hi', 0) * 100:+.1f}] "
                   f"over {d.get('tasks', 0)} tasks (margin {summary['margin'] * 100:.0f} pts)"
                   + (f", tool-call validity {e['validity_verdict']}" if "validity_verdict" in e else ""))
    if e["verdict"] == "REJECT":
        a.reject.append("task gate REJECT")
    elif e["verdict"] not in ("BETTER", "PASS"):
        a.hold.append(f"task gate {e['verdict']}: the interval straddles the margin; more tasks help more than repetitions "
                      f"({d.get('tasks', '?')} tasks here; a simulation in pancake-docs/batch3/promotion.md: 18 tasks x 5 reps passes an identical arm "
                      "in about 3% of runs, 100+ tasks x 10 reps in about 85%)")
    for w in summary.get("warnings", []):
        if "base runs differ" in w:
            a.hold.append("task gate: " + w)


def judge_arm(item, spec, found, arm, need, min_pairs, m):
    """One arm of the item -> Arm.  found: the located result paths; m: number of comparisons for the Bonferroni correction."""
    a = Arm(arm)
    base = next(iter(spec["arms"]))         # stock: parity and token forcing compare with it
    ref = item.get("vs", base)              # the gain is measured against it
    claim = item.get("claim") or spec.get("cases") or []
    alpha = ALPHA / m
    decides = item.get("decides", "abtest")
    gates = item.get("gates", ["task_gate"] if decides == "taskgate" else [] if item["exact"] else ["token_forcing", "task_gate"])
    runs = None
    if found["runs"]:
        runs, discarded, problems = read_runs(found["runs"])
        a.hold += problems
        missing = [x for x in {base, ref, arm} if x not in runs]
        if missing:
            a.hold.append(f"no abtest runs of {', '.join(sorted(missing))} in {found['runs'].name}")
            runs = None
        elif discarded:
            a.lines.append(f"abtest   {discarded} noisy attempts were discarded and repeated")
    if decides == "abtest":
        if runs is None:
            a.hold.append("no abtest result" if not found["runs"] else "abtest result incomplete")
        else:
            labelled = {}
            for c in spec["cases"]:
                pairs = case_pairs(runs[ref], runs[arm], c)
                if pairs:
                    labelled[c] = gain_ci(pairs, alpha if c in claim else ALPHA)
            settle_gain(a, labelled, claim, need, min_pairs, alpha)
            noisy_runs(a, list(runs[ref].values()) + list(runs[arm].values()))
    elif decides == "replay":
        rbase, rruns = read_replay(found["replay"]) if found["replay"] else (None, {})
        rbase = item.get("vs", rbase)
        if rbase is None or arm not in rruns or rbase not in rruns:
            a.hold.append(f"no replay result (replay_turns.py --out <results>/<stem>-replay) for {arm} against {rbase}")
        else:
            pairs = [(rruns[rbase][r]["prompt_s"], rruns[arm][r]["prompt_s"]) for r in sorted(rruns[rbase].keys() & rruns[arm].keys())]
            settle_gain(a, {"replay prompt s": gain_ci(pairs, alpha, True)} if pairs else {}, ["replay prompt s"], need, min_pairs, alpha)
            noisy_runs(a, [v for rs in (rruns[rbase], rruns[arm]) for v in rs.values()])
            if runs is not None:   # abtest on the same spec only checks that decode speed does not drop
                for c in spec["cases"]:
                    pairs = case_pairs(runs[ref], runs[arm], c)
                    if pairs:
                        g = gain_ci(pairs, ALPHA)
                        a.lines.append(f"side   {c:14s} {fmt(g)}")
                        if g["n"] >= 2 and (g["hi"] < -REGRESS_CI or g["gain"] <= -REGRESS_POINT):
                            a.hold.append(f"regression on {c} (abtest side effect check): {fmt(g)}")
    else:   # taskgate: time to success
        summary = load_json(Path(found["taskgate"]) / "summary.json") if found["taskgate"] else None
        e = ((summary or {}).get("arms") or {}).get(item.get("taskgate_arm", arm)) or {}
        g = e.get("auc_diff")
        if not g:
            a.hold.append("no time-to-success curve in the task gate summary (spec key time_curve: true)")
        else:
            lab = {"time AUC pts": {"n": e["success_diff"]["tasks"], "gain": g["diff"] * 100, "lo": g["lo"] * 100, "hi": g["hi"] * 100,
                                    "sd": math.nan}}
            settle_gain(a, lab, ["time AUC pts"], need, 1, alpha, " pts")
    if item["exact"]:
        for p in item.get("parity", ["text"]):
            if p == "text":
                if runs is not None:
                    judge_text(a, spec, runs, base, arm)
                elif decides != "abtest":
                    a.hold.append("no abtest runs: text parity unavailable")
            elif isinstance(p, dict) and "waived" in p:
                a.lines.append("parity waived by the item: " + p["waived"])
            elif isinstance(p, dict):
                judge_parity_file(a, "restore" if "restore" in p else "ctest", p, found["dir"])
    if "token_forcing" in gates:
        judge_token_forcing(a, found["quality"], arm)
    if "task_gate" in gates:
        summary = load_json(Path(found["taskgate"]) / "summary.json") if found["taskgate"] else None
        judge_task_gate(a, summary, item.get("taskgate_arm", arm))
    return a


# ------------------------------------------------------------------------------------------------ items
def newest(root, pattern):
    hits = sorted(glob.glob(str(root / pattern)))
    return Path(hits[-1]) if hits else None


def validate(item, root):
    for k in ("name", "spec", "exact", "branch", "commits"):
        if k not in item:
            raise ValueError(f"item {item.get('name', '?')}: key {k!r} missing")
    if not isinstance(item["exact"], bool) or not NAME.fullmatch(item["branch"]) or not all(HEX.fullmatch(c) for c in item["commits"]):
        raise ValueError(f"item {item['name']}: exact must be true/false, branch a branch name, commits hex hashes")
    spec = json.loads((root / item["spec"]).read_text())
    arms = list(spec["arms"])
    if "vs" in item and item["vs"] not in arms:
        raise ValueError(f"item {item['name']}: vs {item['vs']!r} is not an arm of {item['spec']}")
    for k in item.get("arms", []):
        if k not in arms[1:]:
            raise ValueError(f"item {item['name']}: arm {k!r} is not a non-reference arm of {item['spec']} ({', '.join(arms[1:])})")
    for k in item.get("claim", []):
        if k not in spec.get("cases", []):
            raise ValueError(f"item {item['name']}: claim case {k!r} is not a case of {item['spec']}")
    return spec


def locate(root, item, stem):
    runs = newest(root, item.get("runs") or f"bench/results/*/{stem}.jsonl")
    where = {"runs": runs, "dir": runs.parent if runs else None}
    for key in ("quality", "taskgate", "replay"):
        where[key] = newest(root, item.get(key) or f"bench/results/*/{stem}-{key}")
    if where["dir"] is None:   # a taskgate / replay only item: the folder that holds its summary
        owner = where["taskgate"] or where["replay"]
        where["dir"] = owner.parent if owner else None
    return where


def evaluate(item, root):
    """Item -> {name, verdict, best (Arm), arms: [Arm], ...}."""
    spec = validate(item, root)
    found = locate(root, item, Path(item["spec"]).stem)
    arms = item.get("arms") or list(spec["arms"])[1:]
    if found["dir"] is None:
        return {"name": item["name"], "verdict": "HOLD", "best": None, "arms": [], "item": item, "spec": item["spec"], "found": found,
                "rev": None, "need": item.get("min_gain", MIN_GAIN[item["exact"]]), "overrides": [], "arm_spec": None, "command": None,
                "why": [f"not run: no result for {Path(item['spec']).stem} under bench/results"]}
    decides = item.get("decides", "abtest")
    claim = item.get("claim") or spec.get("cases") or []
    m = len(arms) * (len(claim) if decides == "abtest" else 1)   # comparisons the Bonferroni correction counts
    need = item.get("min_gain", MIN_GAIN[item["exact"]])
    min_pairs = item.get("min_pairs", MIN_PAIRS)
    res = [judge_arm(item, spec, found, arm, need, min_pairs, m) for arm in arms]
    rev_file = found["dir"] / "rev.txt" if found["dir"] else None
    rev = rev_file.read_text().strip() if rev_file and rev_file.is_file() else None
    want = item.get("tested_rev")
    if want and (not rev or not (rev.startswith(want) or want.startswith(rev))):
        for a in res:
            a.hold.append(f"results are from rev {rev or '(no rev.txt)'}, the item was tested at {want}: stale or unverifiable")
    best = max(res, key=lambda a: (ORDER.index(a.verdict()), -math.inf if a.gain is None else a.gain))
    return {"name": item["name"], "verdict": best.verdict(), "best": best, "arms": res, "item": item, "spec": item["spec"], "found": found,
            "rev": rev, "need": need, "overrides": [k for k in ("min_gain", "min_pairs") if k in item],
            "arm_spec": spec["arms"][best.name],
            "command": f"git cherry-pick -x {' '.join(item['commits'])}" if item["commits"] else None}


def render(results):
    out = []
    for r in results:
        it = r["item"]
        out.append(f"{r['verdict']:8s} {r['name']}  [{'exact' if it['exact'] else 'lossy'}, {r['spec']}, branch {it['branch']}]")
        if r["found"]["dir"]:
            out.append(f"           results {r['found']['dir']}" + (f", rev {r['rev']}" if r["rev"] else "")
                       + f"; needs lower bound >= {r['need']:g}%" + (f" (item override: {', '.join(r['overrides'])})" if r["overrides"] else ""))
        out += [f"      HOLD: {x}" for x in r.get("why", [])]
        for a in r["arms"]:
            out.append(f"  arm {a.name}: {a.verdict()}")
            out += [f"      {x}" for x in a.lines]
            out += [f"      REJECT: {x}" for x in a.reject] + [f"      HOLD: {x}" for x in a.hold]
        if r["verdict"] == "PROMOTE":
            out.append(f"  winning arm {r['best'].name}: {json.dumps(r['arm_spec'])}")
    counts = {v: sum(r["verdict"] == v for r in results) for v in ORDER[::-1]}
    out.append("\n" + ", ".join(f"{n} {v}" for v, n in counts.items()))
    cmds = [r for r in results if r["verdict"] == "PROMOTE"]
    if cmds:
        out.append("\n# PROMOTE: run in the checkout that has port/rdna3-decode checked out (this tool does not run git)")
        for r in cmds:
            out.append(f"# {r['name']} (arm {r['best'].name})" + (f": {r['item']['note']}" if r["item"].get("note") else ""))
            out.append(r["command"] or "# no commits listed for this item: fill them into the catalog first (see its note)")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ noise
def noise_report(paths):
    """Per file and case: base mean and CV, per arm paired SD, and the pairs needed to see a 1% / 3% gain at 95%."""
    out = []
    for p in paths:
        runs, _, _ = read_runs(p)
        base = json.loads(Path(p).read_text().splitlines()[0])["arm"]   # the reference arm runs first in rep 0
        cases = sorted({r["case"] for rs in runs.values() for rec in rs.values() for r in rec["rows"]})
        out.append(Path(p).name)
        for c in cases:
            bv = [case_value(rec, c) for rec in runs[base].values()]
            bv = [v for v in bv if v is not None]
            if len(bv) < 2:
                continue
            m = sum(bv) / len(bv)
            cv = math.sqrt(sum((v - m) ** 2 for v in bv) / (len(bv) - 1)) / m * 100
            line = f"  {c:11s} base {m:7.1f} CV {cv:4.1f}% n={len(bv)}"
            for arm in runs:
                g = gain_ci(case_pairs(runs[base], runs[arm], c)) if arm != base and len(case_pairs(runs[base], runs[arm], c)) >= 2 else None
                if g:
                    need = [pairs_needed(g["sd"], t) for t in (1.0, 3.0)]
                    line += f" | {arm} SD {g['sd']:.1f}% pairs for 1%/3%: {need[0] or '>200'}/{need[1] or '>200'}"
            out.append(line)
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("items", nargs="*", help="ITEMS.json (or, with --noise, abtest jsonl files)")
    ap.add_argument("--only", help="comma separated item names")
    ap.add_argument("--root", default=str(ROOT), help="repo root (default: this repo)")
    ap.add_argument("--noise", action="store_true")
    a = ap.parse_args(argv)
    if a.noise:
        print(noise_report(a.items))
        return 0
    if len(a.items) != 1:
        ap.error("one ITEMS.json")
    root = Path(a.root)
    items = json.loads(Path(a.items[0]).read_text())["items"]
    if a.only:
        items = [i for i in items if i["name"] in a.only.split(",")]
    try:
        results = [evaluate(i, root) for i in items]
    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"promote: {e}", file=sys.stderr)
        return 2
    print(render(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
