"""abtest_replay.py - feed recorded abtest attempts (jsonl) through abtest.select() and print what each selection rule keeps.

    python tools/pancake/abtest_replay.py bench/results/2026-11-11-gpu/*.jsonl [--reclaim-max 3.0] [--block 3] [--components]

It only reads (the jsonl files and the <stem>/<arm>-r<rep>-a<attempt>.out files beside them); it never starts anything.
Per file: the attempts each rule would have started for the recorded arm x rep slots, the requests it keeps, and the
summary in abtest's own format, under
  gate        the old proxy gate as recorded ("clean" of the run: per arm and rep the first clean attempt, else the quietest)
  gate R      the same gate with --reclaim-max R (default 3.0): the first recorded attempt that passes it
  robust      --outlier robust: the first attempt of every arm and rep (a robust run never repeats one), then the robust rule
and the wall time: as recorded, and modeled for --block N --warmup 1 --outlier robust (start overhead and pass time taken
from the recorded timestamps and engine lines).
--components: from the screen files (two requests per case in one engine start) the within-start and between-start spread
of the short case (how much of the noise the passes of one start share: the design effect of a block), the second
request's shift, and from the .out files the shift between the warm-up request and the first measured one.
"""
import argparse
import contextlib
import io
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abtest as A  # noqa: E402

DONE = re.compile(r"done: \d+ tokens in [\d.]+ s \(([\d.]+) tok/s\)")


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def arms_cases(records):
    """Arms and cases in the order the run used them (first seen)."""
    arms = list(dict.fromkeys(r["arm"] for r in records))
    cases = list(dict.fromkeys(x["case"] for r in records for x in r["rows"]))
    return arms, cases


slots = A.slots_of


def attempts_run(records, mode, foreign_cpu=None, reclaim_max=None):
    """Attempts a run under this rule would have started for the recorded slots.  robust: up to and including the pick
    (the recorded later attempts stand in for the re-runs of slow starts; all of them when every one was slow).  gate: up
    to and including the first clean one (all recorded ones when none was); thresholds None: the recorded "clean"."""
    if mode == "robust":
        s = slots(records)
        return sum(len(s[k]) if p is None else next(i for i, r in enumerate(s[k]) if r is p) + 1
                   for k, p in A.robust_picks(records).items())
    clean = (lambda r: r["clean"]) if foreign_cpu is None else (lambda r: r["noise"] <= foreign_cpu and r.get("reclaim", 0.0) <= reclaim_max)
    return sum(next((i for i, r in enumerate(recs) if clean(r)), len(recs) - 1) + 1 for recs in slots(records).values())


def captured(fn, *args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args)
    return buf.getvalue()


def requests_used(used):
    return sum(len(rec["rows"]) for recs in used.values() for rec in recs)


def stamp(r):
    return time.mktime(time.strptime(r["t"], "%Y-%m-%d %H:%M:%S"))


def seconds(rec, cold_only=False):
    """Engine time of an attempt's requests: prompt read + decode; a case's later requests in the same start (a screen's
    second one) read their prompt from the conversation cache, so only their decode counts (cold_only: left out)."""
    t, seen = 0.0, set()
    for x in rec["rows"]:
        if x["case"] in seen and cold_only:
            continue
        t += x["gen"] / x["decode_tps"] + (0.0 if x["case"] in seen else x["prompt"] / x["prompt_tps"])
        seen.add(x["case"])
    return t


def timing(records):
    """(median seconds between consecutive attempts, median seconds of one cold pass, median seconds of the rest, wall
    seconds of the file): an attempt's time is the gap to the next record; the rest is that minus its requests (quiet-gate
    wait, start, warm-up request, stop)."""
    ts = [stamp(r) for r in records]
    gaps = [(b - a, r) for a, b, r in zip(ts, ts[1:], records) if b > a]
    if not gaps:
        return (float("nan"),) * 4
    gap = statistics.median(g for g, _ in gaps)
    rest = statistics.median(g - seconds(r) for g, r in gaps)
    return gap, statistics.median(seconds(r, True) for r in records), rest, ts[-1] - ts[0] + gap


def is_screen(records):
    """Two requests of a case in one start: abtest --tier screen (per_run 2)."""
    return any(len([x for x in r["rows"] if x["case"] == c]) > 1 for r in records for c in {x["case"] for x in r["rows"]})


def model_seconds(n_arms, reps, block, overhead, pass_s, warmup=1):
    """Wall time of a robust block run (no repeats): starts x (overhead + (warmup + passes) x pass time)."""
    order, passes = A.schedule(list(range(n_arms)), reps, block)
    return len(order) * (overhead + (warmup + passes) * pass_s)


def replay_file(path, reclaim_max, block):
    records = load(path)
    arms, cases = arms_cases(records)
    n_slots = len(slots(records))
    reps = max(r["rep"] for r in records) + 1
    print(f"\n==== {Path(path).name}: {len(records)} attempts recorded, {n_slots} arm x rep slots ({len(arms)} arms x {reps} reps), "
          f"cases {', '.join(cases)}")
    screen = is_screen(records)
    robust_name = "robust" + (" (screen: not allowed)" if screen else "")
    rules = [("gate (as run)", dict(outlier="gate")),
             (f"gate reclaim {reclaim_max:g}", dict(outlier="gate", foreign_cpu=25.0, reclaim_max=reclaim_max)),
             (robust_name, dict(outlier="robust"))]
    print(f"  {'rule':30s} {'attempts':>8s} {'requests used':>14s} {'dropped':>8s}")
    out = []
    for name, kw in rules:
        used, dropped = A.select(records, **kw)
        run = attempts_run(records, kw["outlier"], kw.get("foreign_cpu"), kw.get("reclaim_max"))
        print(f"  {name:30s} {run:8d} {requests_used(used):14d} {sum(len(d['rows']) for d in dropped):8d}")
        out.append((name, used, dropped, run))
    gap, pass_s, rest, wall = timing(records)
    print(f"  time: {wall / 60:.0f} min recorded, median {gap:.0f} s per attempt (one cold pass {pass_s:.0f} s; the rest "
          f"{rest:.0f} s: quiet gate, start, warm-up request, stop)")
    if not screen:   # same power: as many passes as make reps independent starts' worth (reps x the design effect)
        for n in sorted({block, 2 * block}):
            for label, rp in (("same reps", reps), ("same power", math.ceil(reps * A.design_effect(n)))):
                print(f"  modeled --block {n} --warmup 1 --outlier robust, {label} ({len(arms)} arms x {rp} passes): "
                      f"{model_seconds(len(arms), rp, n, rest, pass_s) / 60:.0f} min, {len(A.schedule(arms, rp, n)[0])} starts")
    passes = next((r["opts"].get("passes", 1) for r in records if "opts" in r), 1)
    for name, used, dropped, _ in out:
        if name.startswith("gate reclaim"):
            continue
        print(f"\n  ---- {name}")
        text = (captured(A.print_selection, used, dropped, name.startswith("robust"), passes)
                + captured(A.summary, used, arms, cases, A.design_effect(passes)))
        print("\n".join("  " + line for line in text.strip("\n").splitlines()))
    return out


def starts_with_repeats(records, case):
    """{arm: [[value per slot]]}: attempts with two or more measured requests of the case (a screen's two requests, or
    the passes of a --block start), cut to the arm's shortest; slow starts (abtest.slow_starts: a robust run repeats
    them, so they are not part of the noise a block's passes share) left out."""
    by = {}
    for r in records:
        if sum(1 for x in r["rows"] if x["case"] == case) >= 2:
            by.setdefault(r["arm"], []).append(r)
    out = {}
    for arm, recs in by.items():
        slow = A.slow_starts(recs)
        groups = [[x[A.metric(case)] for x in r["rows"] if x["case"] == case] for i, r in enumerate(recs) if i not in slow]
        m = min((len(g) for g in groups), default=0)
        out[arm] = [g[:m] for g in groups]
    return out


def components(paths, case):
    """One-way ANOVA of the case's metric with the starts as groups (pooled over arms and files, in % of each arm's mean,
    each slot's own mean removed first: the warm-up trend and a screen's cached second request are position effects, not
    noise): within-start and between-start SD, the start share (ICC) and the design effect of a block."""
    ssw = dfw = ssb = dfb = 0.0
    sizes, shift = [], []
    for p in paths:
        for groups in starts_with_repeats(load(p), case).values():
            if len(groups) < 2:
                continue
            m = len(groups[0])
            slot = [statistics.mean(g[s] for g in groups) for s in range(m)]
            gm = statistics.mean(slot)
            res = [[(g[s] - slot[s]) / gm * 100 for s in range(m)] for g in groups]
            ssw += sum((x - statistics.mean(g)) ** 2 for g in res for x in g)
            dfw += (len(res) - 1) * (m - 1)   # (the m slot means taken out cost m - 1 of the k (m - 1))
            ssb += m * sum(statistics.mean(g) ** 2 for g in res)
            dfb += len(res) - 1
            sizes += [m] * len(res)
            shift += [(g[1] / g[0] - 1) * 100 for g in groups]
    if dfb < 2:
        print(f"\ncomponents ({case}): no file with two or more requests of it in one start")
        return
    msw, msb, m = ssw / dfw, ssb / dfb, statistics.mean(sizes)
    icc = max(0.0, (msb - msw) / (msb + (m - 1) * msw))
    print(f"\ncomponents ({case}, {len(sizes)} starts with {m:g} requests each): within-start SD {msw ** 0.5:.2f}%, between-start "
          f"SD {max(0.0, (msb - msw) / m) ** 0.5:.2f}%, start share (ICC) {icc:.2f}; design effect of a block of 3 passes "
          f"{1 + 2 * icc:.2f}, of 6 {1 + 5 * icc:.2f} (abtest assumes BLOCK_ICC {A.BLOCK_ICC})")
    print(f"  second request vs first in the same start: {statistics.median(shift):+.1f}% (median)"
          + ("  (a screen sends the same prompt again: it comes from the conversation cache; --block passes get a '[pass k]' line)"
             if case == "edit" else ""))


def warmup_shift(paths):
    """From the .out files: decode tok/s of the first measured short vs the warm-up short request of the same start."""
    first, second = [], []
    for p in paths:
        work = Path(p).with_suffix("")
        for r in load(p):
            f = work / f"{r['arm']}-r{r['rep']}-a{r['attempt']}.out"
            v = [float(m[1]) for m in DONE.finditer(f.read_text(errors="replace"))] if f.exists() else []
            if len(v) >= 2:
                first.append((v[1] / v[0] - 1) * 100)
            if len(v) >= 3 and sum(1 for x in r["rows"] if x["case"] == "short") == 2:
                second.append((v[2] / v[1] - 1) * 100)
    if first:
        print(f"  first measured short vs the warm-up short request: {statistics.median(first):+.1f}% (median of {len(first)} starts)"
              + (f"; screens' second short vs first: {statistics.median(second):+.1f}% ({len(second)})" if second else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--reclaim-max", type=float, default=3.0)
    ap.add_argument("--block", type=int, default=3, help="block size of the modeled run (and twice it)")
    ap.add_argument("--components", action="store_true")
    a = ap.parse_args()
    totals = {}
    for p in a.jsonl:
        for name, used, dropped, run in replay_file(p, a.reclaim_max, a.block):
            t = totals.setdefault(name.split(" (")[0], [0, 0, 0])
            t[0] += run
            t[1] += requests_used(used)
            t[2] += sum(len(d["rows"]) for d in dropped)
    print("\n==== all files: rule, attempts started, requests used, requests dropped")
    for name, (run, req, drop) in totals.items():
        print(f"  {name:30s} {run:8d} {req:14d} {drop:8d}")
    if a.components:
        for case in ("short", "edit"):
            components(a.jsonl, case)
        warmup_shift(a.jsonl)


if __name__ == "__main__":
    sys.exit(main())
