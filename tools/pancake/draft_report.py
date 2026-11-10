"""draft_report.py - pancake-strata MOD_PLAN A25 / A26: DraftPolicy's MTP table against the measured position decay, the
in-engine verify share, and which draft source was chosen against which would have accepted more.

    .venv/bin/python tools/pancake/draft_report.py FILE [FILE ...]

Each FILE is recognised by its content:
  * a window trace (STRATA_WINDOW_TRACE=<file>, "STWT" magic; item 64): per-position acceptance of plain MTP windows,
    acceptance by MTP probability bucket, tokens per window, window cost by size, realised acceptance per draft source;
  * a draft trace (STRATA_SPEC_TRACE=<file>; "R pos T a lookup ms k match | 7 probs | 7 drafter tokens | lookup tokens |
    committed tokens" lines): the same acceptance numbers, plus the A26 counterfactual - for every round both proposals
    (the drafter's chain and the prompt lookup's) are scored against the tokens the target committed, so a lookup window
    shows whether the drafter would have held longer and a drafter window whether the lookup would have;
  * a server / engine log with "strata decode timing:" lines (STRATA_DECODE_TIMING=1): the verify share of a window
    (verify / (verify + draft + commit + the rest)), weighted by windows.

A25 (SD: Performance or Illusion?, arXiv 2601.11580) asks whether DraftPolicy's built-in MTP numbers match the box:
  - mtp_tokens(t) before any window of that size = 1 + 0.7 (t - 1)  (every extra row worth 0.7 tokens, no decay);
  - draft_rate(p) before its bucket has data = p (the MTP's own softmax as the acceptance);
  - kShape: the prior shape of the window cost by size (1, 1.35, 1.7, 2.05, ...).
The engine replaces each prior with measured EMAs as windows come in, so a wrong prior costs the first windows of a
process (and every size or bucket that stays rare).  A26 (Context-Aware Assistant Selection, arXiv 2408.08470) stays
parked unless the counterfactual shows wrong source choices that cost tokens.

Counterfactual limits: the target's tokens are known only up to the committed ones (accepted + 1), so a proposal that
matches all of them is counted as ">= that many" (censored); the alternative's window cost is the median measured cost of
a window of its size; rounds whose drafter tokens were not computed this round (DFlash2's deferred lookup) cannot be told
apart and are scored as they are.
"""
import argparse
import math
import os
import re
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "trace"))
import wtrace  # noqa: E402

KSHAPE = [0.0, 1.0, 1.35, 1.7, 2.05, 2.45, 2.85, 3.25, 3.6, 4.0, 4.4, 4.8, 5.2, 5.6, 6.0, 6.4, 6.8]   # draft_policy.cpp
PROBE_COND = {1: 0.89, 2: 0.86, 3: 0.85}   # tools/mtp_probe.py: greedy acceptance at draft steps 1-3 (include/strata/core/mtp.hpp)
TIMING_RE = re.compile(r"strata decode timing: (\d+) windows, avg T ([\d.]+), ([\d.]+) tokens/window, ([\d.]+) ms/window = (.*)")


def mtp_prior_tokens(t):
    return 1.0 + 0.7 * (t - 1)


# ---------------------------------------------------------------------------------------------------------------------
# inputs -> a common round list: dict(src, T, a, ms, ms_ok, probs, drafter, lookup, out, eligible)
def kind_of(path):
    with open(path, "rb") as f:
        head = f.read(4)
    if head == b"STWT":
        return "stwt"
    with open(path, "r", errors="replace") as f:
        for line in f:
            if line.startswith("R ") and line.count("|") >= 4:
                return "spec"
            if "strata decode timing:" in line:
                return "log"
    return "unknown"


def rounds_from_stwt(path):
    tr = wtrace.read(path)
    out, prev = [], None
    for w in tr.windows:
        if w.rows is None:
            continue
        a = w.accepted
        if a is None:
            src = "merged"
        elif w.drafts is None or w.draft_src is None or len(w.draft_src) == 0:
            src = "one-row"
        else:
            s = set(int(v) for v in w.draft_src)
            src = "lookup" if 2 in s else "mtp+chain" if 3 in s else "dflash2" if 1 in s else "mtp"
        follows = prev is not None and prev[0] is not None and w.position is not None and \
            w.position == prev[0] + (prev[1] if prev[2] is None else prev[2] + 1)
        probs = None
        if src in ("mtp", "dflash2", "mtp+chain") and w.drafts is not None:
            probs = [float(v) for v in w.drafts]
        out.append({"src": src, "T": int(w.rows), "a": a if a is not None else int(w.rows) - 1, "ms": float(w.ms or 0.0),
                    "ms_ok": bool(follows and a is not None), "probs": probs, "drafter": None, "lookup": None, "out": None,
                    "eligible": None, "file": "stwt"})
        prev = (w.position, int(w.rows), a)
    return out


def rounds_from_spec(path):
    out = []
    df2 = False
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("# request"):
                m = re.search(r"df2 (\d)", line)
                df2 = bool(m and m.group(1) == "1")
                continue
            if not line.startswith("R "):
                continue
            parts = line.rstrip("\n").split("|")
            if len(parts) < 5:
                continue
            try:
                h = parts[0].split()
                pos, T, a, sfx = int(h[1]), int(h[2]), int(h[3]), int(h[4])
                ms = float(h[5])
                probs = [float(v) for v in parts[1].split()]
                dtok = [int(v) for v in parts[2].split()]
                ltok = [int(v) for v in parts[3].split()]
                otok = [int(v) for v in parts[4].split()]
            except (ValueError, IndexError):
                continue
            n_d = 0
            while n_d < len(probs) and n_d < len(dtok) and probs[n_d] > 0.0:
                n_d += 1
            src = "lookup" if sfx else ("dflash2" if df2 else "mtp")
            out.append({"src": src, "T": T, "a": a, "ms": ms, "ms_ok": ms > 0, "pos": pos,
                        "probs": None if sfx else probs[:max(0, T - 1)], "drafter": dtok[:n_d], "lookup": ltok, "out": otok,
                        "eligible": bool(ltok) and bool(dtok[:n_d]) and ltok[0] == dtok[0], "file": "spec"})
    return out


def timing_from_log(path):
    rows = []
    with open(path, errors="replace") as f:
        for line in f:
            m = TIMING_RE.search(line)
            if not m:
                continue
            n, _, tok, ms, rest = int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)), m.group(5)
            def grab(name):
                mm = re.search(name + r" ([\d.]+)", rest)
                return float(mm.group(1)) if mm else 0.0
            rows.append({"windows": n, "tok": tok, "ms": ms, "verify": grab(r"verify"), "draft": grab(r"\+ draft"),
                         "commit": grab(r"commit/emit"), "adapt_wait": grab(r"adapt wait"), "adapt_join": grab(r"adapt join")})
    return rows


# ---------------------------------------------------------------------------------------------------------------------
# A25
def position_table(rounds, src="mtp"):
    """-> list of (row j, reached, accepted, conditional, survival) for draft rows of plain `src` windows."""
    rs = [r for r in rounds if r["src"] == src and r["T"] > 1]
    max_t = max((r["T"] for r in rs), default=1)
    out, surv = [], 1.0
    for j in range(1, max_t):
        reached = sum(1 for r in rs if r["T"] - 1 >= j and r["a"] >= j - 1)
        acc = sum(1 for r in rs if r["T"] - 1 >= j and r["a"] >= j)
        cond = acc / reached if reached else float("nan")
        if reached:
            surv *= cond
        out.append((j, reached, acc, cond, surv if reached else float("nan")))
    return out


def calibration(rounds, src="mtp"):
    """-> per probability bucket (8, as DraftPolicy::draft_rate): (lo, hi, reached, accepted share, mean p)."""
    n = [0] * 8
    ok = [0] * 8
    ps = [0.0] * 8
    for r in rounds:
        if r["src"] != src or not r["probs"]:
            continue
        for j, p in enumerate(r["probs"][:r["T"] - 1]):
            if j > r["a"] or not math.isfinite(p):
                break
            b = min(7, max(0, int(p * 8.0)))
            n[b] += 1
            ok[b] += 1 if j < r["a"] else 0
            ps[b] += p
    return [(b / 8, (b + 1) / 8, n[b], ok[b] / n[b] if n[b] else float("nan"), ps[b] / n[b] if n[b] else float("nan"))
            for b in range(8)]


def tokens_by_size(rounds, src="mtp"):
    by = {}
    for r in rounds:
        if r["src"] == src:
            by.setdefault(r["T"], []).append(r["a"] + 1)
    return {t: (len(v), sum(v) / len(v)) for t, v in sorted(by.items())}


def cost_by_size(rounds):
    by = {}
    for r in rounds:
        if r["ms_ok"] and r["ms"] > 0 and r["src"] != "merged":
            by.setdefault(r["T"], []).append(r["ms"])
    return {t: (len(v), statistics.median(v)) for t, v in sorted(by.items())}


def verify_share(timing):
    w = sum(t["windows"] for t in timing)
    if not w:
        return None
    avg = lambda k: sum(t[k] * t["windows"] for t in timing) / w
    ms = avg("ms")
    return {"requests": len(timing), "windows": w, "ms": ms, "verify": avg("verify"), "draft": avg("draft"),
            "commit": avg("commit"), "adapt": avg("adapt_wait") + avg("adapt_join"), "tok": avg("tok")}


# ---------------------------------------------------------------------------------------------------------------------
# A26
def prefix_match(prop, out):
    """Drafts of `prop` that equal the committed tokens, from the first; censored when every known token matched."""
    n = 0
    while n < len(prop) and n < len(out) and prop[n] == out[n]:
        n += 1
    return n, (n == len(out) and n < len(prop))


def source_counterfactual(rounds, cost):
    """-> dict with the lookup-chosen and drafter-chosen comparisons (spec-trace rounds only)."""
    def c(t):
        if t in cost:
            return cost[t][1]
        if not cost:
            return 1.0 + 0.35 * (t - 1)
        ts = sorted(cost)
        near = min(ts, key=lambda u: abs(u - t))
        return cost[near][1] * KSHAPE[min(t, 16)] / KSHAPE[min(near, 16)]

    res = {}
    lk = [r for r in rounds if r["file"] == "spec" and r["src"] == "lookup" and r["drafter"]]
    more = same = fewer = 0
    tok_ch = tok_alt = 0.0
    v_better = 0
    for r in lk:
        cf, cens = prefix_match(r["drafter"], r["out"])
        a = r["a"]
        tok_ch += a + 1
        tok_alt += cf + 1
        if cf > a:
            more += 1
        elif cf == a:
            same += 1
        else:
            fewer += 1
        t_alt = min(len(r["drafter"]), 15) + 1   # the drafter's own chain as its window
        if (cf + 1) / c(t_alt) > (a + 1) / c(r["T"]):
            v_better += 1
    res["lookup"] = {"n": len(lk), "drafter_more": more, "same": same, "drafter_fewer": fewer,
                     "tok_chosen": tok_ch, "tok_drafter": tok_alt, "drafter_better_per_ms": v_better}
    md = [r for r in rounds if r["file"] == "spec" and r["src"] in ("mtp", "dflash2") and r["lookup"]]
    more = same = fewer = more_el = n_el = 0
    tok_ch = tok_alt = 0.0
    v_better = 0
    for r in md:
        cf, cens = prefix_match(r["lookup"], r["out"])
        a = r["a"]
        tok_ch += a + 1
        tok_alt += cf + 1
        n_el += 1 if r["eligible"] else 0
        if cf > a:
            more += 1
            more_el += 1 if r["eligible"] else 0
        elif cf == a:
            same += 1
        else:
            fewer += 1
        t_alt = min(len(r["lookup"]), 15) + 1
        if (cf + 1) / c(t_alt) > (a + 1) / c(r["T"]):
            v_better += 1
    res["drafter"] = {"n": len(md), "eligible": n_el, "lookup_more": more, "lookup_more_eligible": more_el, "same": same,
                      "lookup_fewer": fewer, "tok_chosen": tok_ch, "tok_lookup": tok_alt, "lookup_better_per_ms": v_better}
    n_all = sum(1 for r in rounds if r["file"] == "spec")
    res["rounds"] = n_all
    res["tokens"] = sum(r["a"] + 1 for r in rounds if r["file"] == "spec")
    return res


def realised_by_source(rounds):
    by = {}
    for r in rounds:
        if r["src"] in ("merged", "one-row"):
            continue
        d = by.setdefault(r["src"], {"n": 0, "rows": 0, "tok": 0, "ms": 0.0, "n_ms": 0, "tok_ms": 0})
        d["n"] += 1
        d["rows"] += r["T"]
        d["tok"] += r["a"] + 1
        if r["ms_ok"] and r["ms"] > 0:
            d["ms"] += r["ms"]
            d["n_ms"] += 1
            d["tok_ms"] += r["a"] + 1
    return by


# ---------------------------------------------------------------------------------------------------------------------
def report(rounds, timing, out=None):
    out = out if out is not None else sys.stdout
    p = lambda *a: print(*a, file=out)
    src = "mtp" if any(r["src"] == "mtp" for r in rounds) else "dflash2"
    p(f"{len(rounds)} windows read ({sum(1 for r in rounds if r['file'] == 'stwt')} from window traces, "
      f"{sum(1 for r in rounds if r['file'] == 'spec')} from draft traces); {len(timing)} decode-timing lines")

    p(f"\n== A25.1 per-position acceptance, plain {src} windows (DraftPolicy's prior: every row worth 0.70, no decay)")
    p(f"{'row':>4s} {'reached':>8s} {'held':>8s} {'P(held|reached)':>16s} {'survival':>9s} {'prior':>6s} {'mtp_probe':>9s}")
    pt = position_table(rounds, src)
    for j, reached, acc, cond, surv in pt:
        probe = f"{PROBE_COND[j]:.2f}" if j in PROBE_COND else ""
        p(f"{j:4d} {reached:8d} {acc:8d} {cond:16.3f} {surv:9.3f} {0.70:6.2f} {probe:>9s}")
    if pt:
        p("  survival = P(rows 1..j all held); the prior's mtp_tokens(t) = 1 + 0.7 (t - 1) assumes survival 0.70 at every row")

    p(f"\n== A25.2 tokens per {src} window by size: measured vs mtp_tokens' prior 1 + 0.7 (t - 1)")
    for t, (n, tok) in tokens_by_size(rounds, src).items():
        pr = mtp_prior_tokens(t)
        p(f"  T{t:<2d} windows {n:6d}  tokens {tok:6.3f}  prior {pr:5.2f}  prior/measured {pr / tok:5.2f}")

    p(f"\n== A25.3 acceptance by draft probability (draft_rate's 8 buckets; prior = the probability itself)")
    p(f"{'bucket':>10s} {'reached':>8s} {'held':>7s} {'mean p':>7s} {'held - p':>9s}")
    for lo, hi, n, acc, mp in calibration(rounds, src):
        if n:
            p(f"  {lo:.3f}-{hi:.3f} {n:8d} {acc:7.3f} {mp:7.3f} {acc - mp:+9.3f}")

    cost = cost_by_size(rounds)
    p("\n== A25.4 window cost by size (median ms of windows that follow their predecessor) vs kShape")
    if cost:
        t0 = min(cost)
        base = cost[t0][1]
        for t, (n, ms) in cost.items():
            p(f"  T{t:<2d} n {n:6d}  {ms:8.2f} ms  shape {ms / base * KSHAPE[min(t0, 16)]:5.2f}  kShape {KSHAPE[min(t, 16)]:5.2f}")
    else:
        p("  (no window with a trusted time)")

    p("\n== A25.5 in-engine verify share (\"strata decode timing\" lines, STRATA_DECODE_TIMING=1)")
    vs = verify_share(timing)
    if vs:
        p(f"  {vs['requests']} requests, {vs['windows']} windows, {vs['ms']:.2f} ms/window, {vs['tok']:.2f} tokens/window:"
          f" verify {vs['verify']:.2f} ({vs['verify'] / vs['ms']:.1%}), draft {vs['draft']:.2f} ({vs['draft'] / vs['ms']:.1%}),"
          f" commit/emit {vs['commit']:.2f} ({vs['commit'] / vs['ms']:.1%}), adapt {vs['adapt']:.2f} ({vs['adapt'] / vs['ms']:.1%})")
    else:
        p("  (no timing lines: run the engine with STRATA_DECODE_TIMING=1 and pass its log)")

    p("\n== A26 draft source: realised per source (window traces and draft traces)")
    for s, d in sorted(realised_by_source(rounds).items()):
        tpm = d["tok_ms"] / d["ms"] if d["ms"] > 0 else float("nan")
        p(f"  {s:10s} windows {d['n']:6d}  rows/window {d['rows'] / d['n']:5.2f}  tokens/window {d['tok'] / d['n']:5.2f}"
          f"  tokens/ms {tpm:7.4f}")
    cf = source_counterfactual(rounds, cost)
    if cf["rounds"]:
        lk, md = cf["lookup"], cf["drafter"]
        p(f"  counterfactual on {cf['rounds']} draft-trace rounds ({cf['tokens']} committed tokens):")
        if lk["n"]:
            p(f"    lookup chosen {lk['n']}: the drafter's chain would have held longer in {lk['drafter_more']}"
              f" ({lk['drafter_more'] / lk['n']:.1%}), as long in {lk['same']}, shorter in {lk['drafter_fewer']};"
              f" tokens {lk['tok_chosen']:.0f} chosen vs >= {lk['tok_drafter']:.0f};"
              f" better per ms (median costs) in {lk['drafter_better_per_ms']}")
        if md["n"]:
            p(f"    drafter chosen with a lookup proposal {md['n']} (eligible: the lookup's first token = the drafter's,"
              f" {md['eligible']}): the lookup would have held longer in {md['lookup_more']} ({md['lookup_more'] / md['n']:.1%};"
              f" eligible {md['lookup_more_eligible']}), as long in {md['same']}, shorter in {md['lookup_fewer']};"
              f" tokens {md['tok_chosen']:.0f} chosen vs >= {md['tok_lookup']:.0f}; better per ms in {md['lookup_better_per_ms']}")
        wrong = lk["drafter_better_per_ms"] + md["lookup_better_per_ms"]
        p(f"    rounds where the other source looks better per ms: {wrong} of {cf['rounds']} ({wrong / max(1, cf['rounds']):.1%})."
          " A26 stays parked while this is a few % of rounds; a learned source choice is worth building above ~10%.")
    else:
        p("  (no draft-trace rounds: the counterfactual needs STRATA_SPEC_TRACE=<file> with --suffix-draft on)")
    return {"positions": pt, "verify": vs, "counterfactual": cf}


def main(argv=None):
    ap = argparse.ArgumentParser(description="A25 / A26 report: DraftPolicy's MTP table vs measured acceptance, verify "
                                             "share, draft source choice.", epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="window traces (.stwt), STRATA_SPEC_TRACE files, engine/server logs")
    a = ap.parse_args(argv)
    rounds, timing = [], []
    for f in a.files:
        k = kind_of(f)
        if k == "stwt":
            rounds += rounds_from_stwt(f)
        elif k == "spec":
            rounds += rounds_from_spec(f)
        elif k == "log":
            timing += timing_from_log(f)
        else:
            print(f"draft_report: {f}: not a window trace, a STRATA_SPEC_TRACE file or a log with decode timing lines",
                  file=sys.stderr)
    if not rounds and not timing:
        return 2
    report(rounds, timing)
    return 0


if __name__ == "__main__":
    sys.exit(main())
