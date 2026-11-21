#!/usr/bin/env python3
"""dispatch_report.py - the served-path census of a run, per arm, and the INERT verdicts (LEADS.md Part U U3; stdlib only).

The engine counts the kernel path each launch took (include/strata/core/dispatch_stats.hpp: served() / served_if(), by dotted names such as
`pf.gemm.f16.hipblaslt`, `rdna.gr.a8.fused`, `attn.decode.chunk_split`) and, with STRATA_DISPATCH_REPORT=<path>, writes a JSON report when the engine
exits ({"launches": ..., "counts": [...], "served": {"name": {"n": hits, "declined": misses}}}).  abtest.py with ABTEST_DISPATCH_REPORT=1 sets that
variable for every engine start (<work>/<arm>-r<rep>-a<attempt>.dispatch.json) and keeps a digest of the report in the attempt's jsonl row ("dispatch").

An arm whose switch never changed a served path is INERT, not "no gain" (T-044, T-048, T-498, T-803, T-984 and the INCONCLUSIVE flags of the screen were that).
tools/pancake/dispatch_expect.json maps a flag to the counters its switch must move: "up" counters rise when the flag takes the tested value, "down" counters fall.
Per mapped flag of an arm, against the base arm (the first non-warm-up arm, or --base):
  ACTIVE     an expected counter moved the expected way (by at least max(1, 2 %) of the larger value; "mode": "cross" = appeared / vanished, for counters
             that are counted once per graph capture, whose totals depend on how many windows were captured);
  DECLINED   no expected counter moved and the path that should have started declined (the `declined` side of served_if() went up): the switch asked, the path said no;
  INERT      no expected counter moved: the switch did not act (an exact path counter that moved the other way is shown);
  NO-REPORT  the arm or the base has no report (the engine did not exit cleanly, or the census was off): nothing is concluded;
  UNMAPPED   the flag is in the unmapped list or in neither (no counter at its launch site yet): nothing is concluded.
An arm named minus-X (sweep.py: switch X taken out) reverses the expectations of X.  The arm's verdict: ACTIVE when any mapped flag is ACTIVE, else DECLINED /
INERT when every mapped flag is, else NO-REPORT / UNMAPPED.

  dispatch_report.py --rows ROWS.jsonl [--spec SPEC.json] [--base NAME] [--expect dispatch_expect.json] [--strict] [--top N]
  dispatch_report.py --dir WORKDIR [--spec SPEC.json] ...            reads WORKDIR/<arm>-r<rep>-a<attempt>.dispatch.json (the report file, or an engine log holding one)
  dispatch_report.py --report ARM=FILE ... [--base NAME]             explicit files (a report, or an engine log / stdout with the report embedded)
--rows and --dir / --report can be combined.  --strict: exit 4 when an arm is INERT.  Exit 0 otherwise.
"""
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECT_PATH = HERE / "dispatch_expect.json"
TOL = 0.02          # relative move that counts (prefill counters are proportional to the work, and the arms run the same requests)
WARM_ARMS = ("warmup",)
FILE_RE = re.compile(r"^(?P<arm>.+)-r(?P<rep>\d+)-a(?P<att>\d+)\.dispatch\.json$")


# ------------------------------------------------------------------------------------------------------------------ reading a report
def parse_report(text):
    """The report dict inside `text`: the whole text when it is the JSON, else the first `{` line of an engine log / stdout that starts a block
    whose second line is ` "launches"` (the dump is written with that layout), up to the next line that is exactly `}`.  None when there is none."""
    t = text.strip()
    if t.startswith("{"):
        try:
            d = json.loads(t)
            return d if isinstance(d, dict) and ("served" in d or "counts" in d or "launches" in d) else None
        except ValueError:
            pass
    lines = text.splitlines()
    for i in range(len(lines) - 1):
        if lines[i].strip() == "{" and lines[i + 1].lstrip().startswith('"launches"'):
            for j in range(i + 1, len(lines)):
                if lines[j].rstrip() == "}":   # the closing brace at column 0 (the served block's own ` }` is indented)
                    try:
                        return json.loads("\n".join(lines[i:j + 1]))
                    except ValueError:
                        break
            break
    return None


def read_report(path):
    try:
        return parse_report(Path(path).read_text(errors="replace"))
    except OSError:
        return None


def digest(report):
    """A small, jsonl-friendly form of a report: the served map as {name: [hits, declined]}, the graph launches, and the free-text notes (path / reason
    strings of the older counters) as {text: n}.  None for no report."""
    if not isinstance(report, dict):
        return None
    served = {k: [int(v.get("n", 0)), int(v.get("declined", 0))] for k, v in (report.get("served") or {}).items() if isinstance(v, dict)}
    notes = {}
    for c in report.get("counts") or []:
        if c.get("cat") == "str":
            notes[c.get("key", "?")] = notes.get(c.get("key", "?"), 0) + int(c.get("n", 0))
    return {"served": served, "launches": report.get("launches") or {}, "notes": notes}


# ------------------------------------------------------------------------------------------------------------------ the expectation file
def load_expect(path=None):
    return json.loads(Path(path or EXPECT_PATH).read_text(encoding="utf-8"))


def lint_expect(exp, registry_names=None, source_names=None):
    """Problems of an expectation file (empty list = fine): the shape, no flag both mapped and unmapped, every counter a dotted lowercase name, a known mode, and
    (when given) every registry flag covered and every counter present as a served name in the sources."""
    bad = []
    flags, unm = exp.get("flags") or {}, exp.get("unmapped") or {}
    if not isinstance(flags, dict) or not isinstance(unm, dict):
        return ["flags and unmapped must be objects"]
    for f in sorted(set(flags) & set(unm)):
        bad.append(f"{f}: both mapped and unmapped")
    name_re = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)+$")
    for f, e in flags.items():
        if not isinstance(e, dict) or not (e.get("up") or e.get("down")):
            bad.append(f"{f}: needs a non-empty up or down list")
            continue
        for k in e:
            if k not in ("up", "down", "mode", "why"):
                bad.append(f"{f}: unknown key {k}")
        if e.get("mode", "delta") not in ("delta", "cross"):
            bad.append(f"{f}: mode must be delta or cross")
        for side in ("up", "down"):
            for c in e.get(side) or []:
                if not name_re.match(c):
                    bad.append(f"{f}: bad counter name {c!r}")
                elif source_names is not None and c not in source_names:
                    bad.append(f"{f}: counter {c} is not a served() name in the sources")
        if not e.get("why"):
            bad.append(f"{f}: no why")
    for f, why in unm.items():
        if not isinstance(why, str) or not why:
            bad.append(f"{f}: unmapped needs a reason")
    if registry_names is not None:
        for f in sorted(set(registry_names) - set(flags) - set(unm)):
            bad.append(f"{f}: in the sweep registry but neither mapped nor unmapped")
    return bad


# ------------------------------------------------------------------------------------------------------------------ the INERT rule
def total(dg, names):
    """(hits, declined) of the listed counters in a digest."""
    s = (dg or {}).get("served") or {}
    h = sum(s.get(n, [0, 0])[0] for n in names)
    d = sum(s.get(n, [0, 0])[1] for n in names)
    return h, d


def moved(base, arm, how):
    """Signed move class of one total: +1 up, -1 down, 0 none."""
    if how == "cross":
        return 1 if (base == 0 and arm > 0) else -1 if (base > 0 and arm == 0) else 0
    d = arm - base
    return 0 if abs(d) < max(1.0, TOL * max(base, arm)) else (1 if d > 0 else -1)


def classify_flag(entry, base_dg, arm_dg, reverse=False):
    """(verdict, detail) of one mapped flag.  entry: {"up": [...], "down": [...], "mode": "delta"|"cross"}."""
    if base_dg is None or arm_dg is None:
        return "NO-REPORT", "no report" + (" for the base" if base_dg is None else " for the arm")
    how = entry.get("mode", "delta")
    want_up, want_down = (entry.get("down") or [], entry.get("up") or []) if reverse else (entry.get("up") or [], entry.get("down") or [])
    parts, active, opposite = [], False, False
    for names, want in ((want_up, 1), (want_down, -1)):
        if not names:
            continue
        bh, _ = total(base_dg, names)
        ah, ad = total(arm_dg, names)
        m = moved(bh, ah, how)
        parts.append(f"{'+'.join(names) if len(names) < 3 else names[0] + '+' + str(len(names) - 1) + ' more'}: {bh} -> {ah}")
        if m == want:
            active = True
        elif m == -want:
            opposite = True
    if active:
        return "ACTIVE", "; ".join(parts)
    up_names = want_up
    if up_names:
        _, bd = total(base_dg, up_names)
        _, ad = total(arm_dg, up_names)
        if ad > bd:
            return "DECLINED", "; ".join(parts) + f"; declined {bd} -> {ad}"
    return "INERT", "; ".join(parts) + ("; moved the other way" if opposite else "")


def flags_of_arm(arm, spec_arm=None):
    """(flags, reverse): the switches an arm sets (env keys with a value, args keys), or the one named by a plus-X / minus-X arm."""
    m = re.match(r"^(plus|minus)-(.+)$", arm)
    if m:
        return [m.group(2)], m.group(1) == "minus"
    fl = []
    for k, v in ((spec_arm or {}).get("env") or {}).items():
        if v is not None:
            fl.append(k)
    for k, v in ((spec_arm or {}).get("args") or {}).items():
        if v is not None:
            fl.append(k)
    return fl, False


def classify_arm(arm, flags, reverse, expect, base_dg, arm_dg):
    """{"arm", "verdict", "flags": {flag: (verdict, detail)}}."""
    mapped = expect.get("flags") or {}
    out, fl = {}, {}
    for f in flags:
        if f in mapped:
            fl[f] = classify_flag(mapped[f], base_dg, arm_dg, reverse)
        else:
            fl[f] = ("UNMAPPED", (expect.get("unmapped") or {}).get(f, "not in the expectation file"))
    vs = [v for v, _ in fl.values()]
    real = [v for v in vs if v != "UNMAPPED"]
    if not flags:
        v = "UNMAPPED"
    elif not real:
        v = "UNMAPPED"
    elif "ACTIVE" in real:
        v = "ACTIVE"
    elif all(x == "NO-REPORT" for x in real):
        v = "NO-REPORT"
    elif "INERT" in real:
        v = "INERT"
    elif "DECLINED" in real:
        v = "DECLINED"
    else:
        v = real[0]
    out.update(arm=arm, verdict=v, flags=fl)
    return out


# ------------------------------------------------------------------------------------------------------------------ collecting digests per arm
def digests_from_rows(rows):
    """{arm: first digest} from abtest jsonl rows ("arm", "dispatch"); the first attempt with a report counts."""
    d = {}
    for r in rows:
        a = r.get("arm")
        if a is None or a in d and d[a] is not None:
            continue
        dg = r.get("dispatch")
        if dg is not None and "served" in dg:
            d[a] = dg
        else:
            d.setdefault(a, None)
    return d


def digests_from_dir(path):
    d = {}
    for f in sorted(glob.glob(os.path.join(path, "*.dispatch.json"))):
        m = FILE_RE.match(os.path.basename(f))
        if not m:
            continue
        a = m.group("arm")
        dg = digest(read_report(f))
        if d.get(a) is None:
            d[a] = dg
    return d


def pick_base(arms, spec=None, base=None):
    if base:
        return base
    names = list((spec or {}).get("arms") or arms)
    return next((a for a in names if a not in WARM_ARMS), None)


def report(digs, expect, spec=None, base=None, top=0):
    """(lines, results): per arm the verdict, its flags with details, and optionally the top counters."""
    arms = [a for a in (list((spec or {}).get("arms") or []) + [x for x in digs if x not in ((spec or {}).get("arms") or [])]) if a in digs]
    b = pick_base(arms, spec, base)
    bdg = digs.get(b)
    lines, res = [f"DISPATCH census: base arm {b}" + ("" if bdg is not None else " (NO REPORT)")], []
    for a in arms:
        if a in WARM_ARMS or a == b:
            continue
        flags, rev = flags_of_arm(a, ((spec or {}).get("arms") or {}).get(a))
        r = classify_arm(a, flags, rev, expect, bdg, digs.get(a))
        res.append(r)
        lines.append(f"DISPATCH {a}: {r['verdict']}" + (" (switch reversed: a minus arm)" if rev else ""))
        for f, (v, why) in r["flags"].items():
            lines.append(f"    {f}: {v}  {why}")
        if top and digs.get(a):
            s = digs[a].get("served") or {}
            for n, (h, dcl) in sorted(s.items(), key=lambda kv: -kv[1][0])[:top]:
                bh = ((bdg or {}).get("served") or {}).get(n, [0, 0])[0]
                lines.append(f"      {n:40s} {h:>10d} (base {bh:>10d}){'  declined ' + str(dcl) if dcl else ''}")
    n_in = sum(1 for r in res if r["verdict"] == "INERT")
    lines.append(f"DISPATCH summary: {sum(1 for r in res if r['verdict'] == 'ACTIVE')} ACTIVE, {n_in} INERT, "
                 f"{sum(1 for r in res if r['verdict'] == 'DECLINED')} DECLINED, {sum(1 for r in res if r['verdict'] == 'NO-REPORT')} NO-REPORT, "
                 f"{sum(1 for r in res if r['verdict'] == 'UNMAPPED')} UNMAPPED of {len(res)} arms")
    return lines, res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", action="append", default=[], help="abtest jsonl rows (with the dispatch digest)")
    ap.add_argument("--dir", action="append", default=[], help="a work directory holding <arm>-r<rep>-a<attempt>.dispatch.json files")
    ap.add_argument("--report", action="append", default=[], metavar="ARM=FILE", help="a report file, or an engine log / stdout holding one")
    ap.add_argument("--spec", help="the abtest spec (arm switches); without it a plus-X / minus-X arm name names its switch")
    ap.add_argument("--base", help="the base arm (default: the first non-warm-up arm of the spec)")
    ap.add_argument("--expect", default=str(EXPECT_PATH))
    ap.add_argument("--top", type=int, default=0, help="also print the N most served counters of every arm")
    ap.add_argument("--strict", action="store_true", help="exit 4 when an arm is INERT")
    a = ap.parse_args(argv)
    digs = {}
    for p in a.rows:
        rows = []
        for line in Path(p).read_text(errors="replace").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
        for k, v in digests_from_rows(rows).items():
            if digs.get(k) is None:
                digs[k] = v
    for p in a.dir:
        for k, v in digests_from_dir(p).items():
            if digs.get(k) is None:
                digs[k] = v
    for item in a.report:
        k, _, f = item.partition("=")
        if not f:
            ap.error("--report wants ARM=FILE")
        digs[k] = digest(read_report(f))
    if not digs:
        print("dispatch_report: no reports given (--rows, --dir or --report)", file=sys.stderr)
        return 2
    spec = json.loads(Path(a.spec).read_text()) if a.spec else None
    lines, res = report(digs, load_expect(a.expect), spec, a.base, a.top)
    print("\n".join(lines))
    return 4 if a.strict and any(r["verdict"] == "INERT" for r in res) else 0


if __name__ == "__main__":
    sys.exit(main())
