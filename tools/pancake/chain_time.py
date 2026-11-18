"""chain_time.py - where the box hours go (owner order 2026-11-18).

Reads finished box chain logs (win-chain*.log: "START <name> <date>_<time>" / "END <name> exit=N <time>" lines) and the
abtest result files (jsonl, one row per engine start with "t" = the end of the start and the engine's own per-request
numbers) and splits the wall time into
  - measured   : the seconds the engine spent on the measured and warm-up requests (prompt / prompt_tps + gen / decode_tps),
  - start      : everything else inside an abtest (engine start + model load + warm-up request + stop + settle between starts),
  - idle       : the gaps between two chain steps (3 idle minutes, service stop, compaction),
  - other      : chain steps that are not an abtest (replays, gates, builds, probes).

    python3 tools/pancake/chain_time.py LOGDIR [--json OUT]

LOGDIR holds win-chain*.log and win*.jsonl (copied from the box).  Nothing here touches a server.
"""
import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime, timedelta

START_RE = re.compile(r"^START (\S+(?: \S+)*?) (\d{4}-\d\d-\d\d)_(\d\d:\d\d:\d\d)\s*$")
END_RE = re.compile(r"^END (\S+(?: \S+)*?) exit=(\d+) (\d\d:\d\d:\d\d)\s*$")


def chain_steps(lines):
    """[(name, start datetime, end datetime, exit)] of the START / END pairs of one chain log; the END carries only a time of
    day, so the day is the START's (plus one when the time went backwards)."""
    out, open_ = [], {}
    for ln in lines:
        m = START_RE.match(ln)
        if m:
            open_[m[1]] = datetime.strptime(m[2] + " " + m[3], "%Y-%m-%d %H:%M:%S")
            continue
        m = END_RE.match(ln)
        if m and m[1] in open_:
            s = open_.pop(m[1])
            e = datetime.strptime(s.strftime("%Y-%m-%d") + " " + m[3], "%Y-%m-%d %H:%M:%S")
            if e < s:
                e += timedelta(days=1)
            out.append((m[1], s, e, int(m[2])))
    return out


def measured_seconds(row):
    """Engine seconds of one request row: prompt / prompt_tps + gen / decode_tps."""
    s = 0.0
    if row.get("prompt_tps"):
        s += row["prompt"] / row["prompt_tps"]
    if row.get("decode_tps"):
        s += row["gen"] / row["decode_tps"]
    return s


def abtest_split(path, t_start=None):
    """(wall seconds, measured seconds, starts) of one abtest jsonl.  The wall time of a start is the gap to the previous start's
    "t" (the first one: to t_start, the chain step's START, when known; else it is skipped from wall and measured)."""
    rows = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln.startswith("{"):
                continue
            try:
                rows.append(json.loads(ln))
            except ValueError:
                pass
    rows = [r for r in rows if "t" in r and "rows" in r]
    rows.sort(key=lambda r: r["t"])
    prev, wall, meas, n = t_start, 0.0, 0.0, 0
    for r in rows:
        t = datetime.strptime(r["t"], "%Y-%m-%d %H:%M:%S")
        m = sum(measured_seconds(x) for x in r["rows"]) + sum(measured_seconds(x) for x in r.get("warmup_rows", []))
        if prev is not None:
            wall += (t - prev).total_seconds()
            meas += m
            n += 1
        prev = t
    return wall, meas, n


def breakdown(logdir):
    """Per chain: dict(steps, abtest wall / measured / starts, other seconds, idle seconds)."""
    res = {}
    for lp in sorted(glob.glob(os.path.join(logdir, "win-chain*.log"))):
        chain = os.path.basename(lp)[len("win-chain"):-len(".log")]
        steps = chain_steps(open(lp, errors="replace").read().splitlines())
        if not steps:
            continue
        steps.sort(key=lambda x: x[1])
        d = {"steps": len(steps), "abtest_wall": 0.0, "abtest_measured": 0.0, "starts": 0, "other": 0.0, "idle": 0.0}
        last_end = None
        for name, s, e, _rc in steps:
            if last_end is not None and s > last_end:
                d["idle"] += (s - last_end).total_seconds()
            last_end = max(last_end, e) if last_end else e
            dur = (e - s).total_seconds()
            cand = [p for p in glob.glob(os.path.join(logdir, "win*-" + name.split()[0] + "*.jsonl"))
                    if os.path.getmtime(p) and datetime.fromtimestamp(os.path.getmtime(p)).date() >= s.date()]
            if not cand:
                cand = glob.glob(os.path.join(logdir, "win*-" + name.split()[0] + "*.jsonl"))
            if cand and "replay" not in name:
                w, m, n = abtest_split(sorted(cand)[0], s)
                w = min(w, dur) if w else dur
                d["abtest_wall"] += dur
                d["abtest_measured"] += min(m, dur)
                d["starts"] += n
            else:
                d["other"] += dur
        res[chain] = d
    return res


def fmt(res):
    lines = [f"{'chain':8} {'steps':>5} {'starts':>6} {'abtest h':>9} {'measured h':>11} {'start h':>8} {'other h':>8} {'idle h':>7} {'total h':>8}"]
    tot = dict(abtest_wall=0, abtest_measured=0, other=0, idle=0, starts=0, steps=0)
    for ch, d in res.items():
        st = d["abtest_wall"] - d["abtest_measured"]
        total = d["abtest_wall"] + d["other"] + d["idle"]
        lines.append(f"{ch:8} {d['steps']:5d} {d['starts']:6d} {d['abtest_wall']/3600:9.2f} {d['abtest_measured']/3600:11.2f} "
                     f"{st/3600:8.2f} {d['other']/3600:8.2f} {d['idle']/3600:7.2f} {total/3600:8.2f}")
        for k in tot:
            tot[k] += d[k]
    st = tot["abtest_wall"] - tot["abtest_measured"]
    total = tot["abtest_wall"] + tot["other"] + tot["idle"]
    lines.append(f"{'ALL':8} {tot['steps']:5d} {tot['starts']:6d} {tot['abtest_wall']/3600:9.2f} {tot['abtest_measured']/3600:11.2f} "
                 f"{st/3600:8.2f} {tot['other']/3600:8.2f} {tot['idle']/3600:7.2f} {total/3600:8.2f}")
    if total:
        lines.append(f"share of wall: measured {100*tot['abtest_measured']/total:.1f}%  engine start/stop {100*st/total:.1f}%  "
                     f"non-abtest steps {100*tot['other']/total:.1f}%  idle waits {100*tot['idle']/total:.1f}%")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("logdir")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    res = breakdown(a.logdir)
    print(fmt(res))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
