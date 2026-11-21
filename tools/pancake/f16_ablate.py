#!/usr/bin/env python3
"""f16_ablate.py - stage ablation ladder of the F16 prompt-expert kernel (LEADS.md Part U, lead U5; the method is the per-stage "remove one stage and
read the time" ladder of the radiance-rdna3 study, our own code).

The kernel is native_f16_kernel (src/prefill/moe_fused_iq.cu; the shared staging / WMMA / epilogue parts of src/prefill/moe_f16_common.cuh also act
on the K-quant kernels of moe_f16_kq.cu).  src/prefill/moe_f16_ablate.cuh defines seven BENCH-ONLY macros that each remove one stage (results wrong by
design; a macro without STRATA_BENCH_ABLATION is a compile error).  One binary per variant, all with STRATA_PF_F16=1, run the prompt cases; this tool
turns the prompt tok/s of every variant into the share of the prompt time each stage owns.

  python3 tools/pancake/f16_ablate.py variants [--set singles|full]          name<TAB>macros;..  per line (the box driver reads it)
  python3 tools/pancake/f16_ablate.py spec --exe-root DIR [--set ..] [--out FILE]   the abtest spec (bench/specs/276-f16-ablation-ladder.json)
  python3 tools/pancake/f16_ablate.py report RUN.jsonl [RUN2.jsonl ...] [--json]    the ladder

How the report reads the data (every number is a prompt-time share of THIS run, not a claim about another card or model):
  t(arm, case) = 1 / prompt tok/s (prompt lengths are equal across arms); base time = mean of the arms `base` and `base-late` (the drift arm: a gap
  above DRIFT_MAX is printed and the case is marked DRIFTED).
  single share of a stage  = 1 - t(base) / ... = (t_base - t_arm) / t_base     (the stage removed alone; stages overlap, so singles need not add up)
  cumulative step of stage k = (t_cum(k-1) - t_cum(k)) / t_base                (stages removed one after another in LADDER order: they add up)
  floor = t_cum(last) / t_base: what is left with every stage removed (barriers, loop, launch, and everything outside the expert kernel)
  gain_if_free = share / (1 - share) - 1: the most prompt tok/s the stage could return if it cost nothing (an upper bound, never a forecast)
  The ablated binaries route later layers on wrong hidden states, so expert counts change a little: the ladder is approximate (see the header).
Verdict per case: OPEN when a single stage or cumulative step owns at least BAR (5 %) of the prompt time, else CLOSED (the lead's rule: under 5 % closes the
kernel).  Tests: python -m pytest -q tools/pancake/test_f16_ablate.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

BAR = 0.05
DRIFT_MAX = 0.03
CASES = ["long3k", "long", "long32k", "long64k"]
BASE_CONFIG = "configs/pancake/flashnext-iq2_xs-fast-hip.json"

# the ladder order: staging reads first, then the decode of the weights, the LDS traffic, the WMMA, the epilogue
LADDER = [
    ("noload", "STRATA_ABL_NO_GLOBAL_LOAD", "global loads of the weight unit and the staged activations"),
    ("nodeq", "STRATA_ABL_NO_DEQUANT", "format decode of the weight sub-block into int8 codes + scales (native_f16_kernel)"),
    ("nowiden", "STRATA_ABL_NO_WIDEN", "int8 -> F16 widening and scale multiply of the A fragment (native_f16_kernel)"),
    ("nostore", "STRATA_ABL_NO_LDS_STORE", "LDS writes of the weight tile, scales and activation tile"),
    ("noread", "STRATA_ABL_NO_LDS_READ", "LDS reads of the weight fragment, scales and B fragments"),
    ("nowmma", "STRATA_ABL_NO_WMMA", "v_wmma_f32_16x16x16_f16"),
    ("noepi", "STRATA_ABL_NO_EPILOGUE", "gate/up (SwiGLU + int8 H) and down epilogues"),
]


def variants(which: str = "full") -> list:
    """[(name, [macros], kind)]: base, the single removals, then (full) the cumulative ladder cum2..cumN (cum1 is `noload`)."""
    out = [("base", [], "base")]
    for name, macro, _ in LADDER:
        out.append((name, [macro], "single"))
    if which == "full":
        for k in range(2, len(LADDER) + 1):
            out.append((f"cum{k}", [m for _, m, _ in LADDER[:k]], "cum"))
    return out


def spec(exe_root: str, which: str = "full") -> dict:
    arms = {}
    for name, macros, _ in variants(which):
        arms[name] = {"exe": f"{exe_root.rstrip('/')}/{name}/strata", "env": {"STRATA_PF_F16": "1"}}
    return {
        "drift_check": True,
        "base": BASE_CONFIG,
        "lossy": True,
        "prefill": True,
        "cases": list(CASES),
        "reps": 1,
        "arms": arms,
        "note": (
            "LEADS.md Part U U5, TEST_LEDGER.md T-1070: stage ablation ladder of the F16 prompt-expert kernel. BENCH ONLY: every arm except base is a "
            "binary built with -DSTRATA_BENCH_ABLATION and STRATA_ABL_* macros (src/prefill/moe_f16_ablate.cuh) that REMOVE ONE STAGE, so the text is wrong "
            "by design and only prompt tok/s is read (no drift check of text, no quality gate). Arms: base = the engine unchanged; noload, nodeq, nowiden, "
            "nostore, noread, nowmma, noepi = one stage removed; cum2..cum7 = the stages removed one after another in ladder order (cum1 is noload, "
            "cum7 has every stage removed). Build and run with tools/pancake/box_chains/bench-ablate-f16.sh (one build directory, two translation units "
            "recompiled per variant, binaries copied to /mnt/pancake/ablate-f16/bin/<arm>/strata); read with python3 tools/pancake/f16_ablate.py report "
            "<run1.jsonl> <run2.jsonl>. STRATA_PF_F16=1 is set in every arm (the preset's PF_F16=2 is the F16-off path, T-820). long3k is the control "
            "(T-820: no F16 gain there), long / long32k / long64k are where the kernel is worth +15-17%. Two runs, never one (owner protocol 2026-11-20)."
        ),
    }


def load(path: Path) -> dict:
    """{(arm, case): [prompt_tps...]} from an abtest jsonl (clean records only; rows of the run)."""
    got = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if "rows" not in rec or "arm" not in rec or rec.get("clean") is False:
            continue
        for row in rec["rows"]:
            tps = row.get("prompt_tps")
            if tps and row.get("case"):
                got[(rec["arm"], row["case"])].append(float(tps))
    return got


def mean(xs):
    return sum(xs) / len(xs)


def ladder(data: dict, case: str, which: str = "full") -> dict:
    """The ladder of one case from {(arm, case): [tps]}."""
    tps = {}
    for (arm, c), xs in data.items():
        if c == case and xs:
            tps[arm] = mean(xs)
    if "base" not in tps:
        return {"case": case, "error": "no base arm"}
    base_tps = tps["base"]
    drift = 0.0
    if "base-late" in tps:
        drift = abs(tps["base-late"] - tps["base"]) / tps["base"]
        base_tps = (tps["base"] + tps["base-late"]) / 2
    t_base = 1.0 / base_tps

    def share(arm):
        return (t_base - 1.0 / tps[arm]) / t_base if arm in tps else None

    singles, steps = [], []
    for name, _, what in LADDER:
        s = share(name)
        singles.append({"arm": name, "what": what, "share": s, "gain_if_free": None if s is None else (s / (1 - s) if s < 1 else None)})
    prev = 0.0
    chain = ["noload"] + [f"cum{k}" for k in range(2, len(LADDER) + 1)]
    for (name, _, what), arm in zip(LADDER, chain):
        cum = share(arm)
        step = None if cum is None else cum - prev
        if cum is not None:
            prev = cum
        steps.append({"arm": arm, "stage": name, "what": what, "cum_share": cum, "step": step,
                      "gain_if_free": None if step is None or step >= 1 else step / (1 - step)})
    floor = None
    last = chain[-1]
    if last in tps:
        floor = (1.0 / tps[last]) / t_base
    kernel_total = None if floor is None else 1.0 - floor
    bests = [x["share"] for x in singles if x["share"] is not None] + [x["step"] for x in steps if x["step"] is not None]
    best = max(bests) if bests else None
    verdict = "NO DATA" if best is None else ("OPEN" if best >= BAR else "CLOSED")
    return {"case": case, "base_tps": base_tps, "drift": drift, "drifted": drift > DRIFT_MAX, "singles": singles, "steps": steps,
            "floor": floor, "kernel_total": kernel_total, "best": best, "verdict": verdict,
            "arms_missing": [a for a, _, k in variants(which) if a not in tps and k != "base"]}


def pct(x):
    return "   n/a" if x is None else f"{x * 100:5.1f}%"


def format_report(results: list) -> str:
    out = []
    for r in results:
        if "error" in r:
            out.append(f"case {r['case']}: {r['error']}")
            continue
        out.append(f"== case {r['case']}: base {r['base_tps']:.1f} prompt tok/s, drift base vs base-late {r['drift'] * 100:.1f}%"
                   + ("  DRIFTED (above 3%: read with care)" if r["drifted"] else ""))
        out.append(f"{'stage (removed alone)':26} {'single':>7} {'gain_if_free':>12}   what")
        for x in r["singles"]:
            g = "" if x["gain_if_free"] is None else f"{x['gain_if_free'] * 100:+.1f}%"
            out.append(f"{x['arm']:26} {pct(x['share'])} {g:>12}   {x['what']}")
        out.append(f"{'stage (cumulative ladder)':26} {'step':>7} {'cum':>7} {'gain_if_free':>12}")
        for x in r["steps"]:
            g = "" if x["gain_if_free"] is None else f"{x['gain_if_free'] * 100:+.1f}%"
            out.append(f"{x['arm'] + ' (' + x['stage'] + ')':26} {pct(x['step'])} {pct(x['cum_share'])} {g:>12}")
        if r["floor"] is not None:
            out.append(f"floor with every stage removed: {r['floor'] * 100:.1f}% of the base prompt time; the expert kernel's whole stage cost = {r['kernel_total'] * 100:.1f}% of it")
        if r["arms_missing"]:
            out.append("missing arms: " + ", ".join(r["arms_missing"]))
        out.append(f"verdict {r['verdict']}: largest stage share {pct(r['best'])} of the prompt time (bar {BAR * 100:.0f}%; under it the kernel is closed)")
        out.append("")
    return "\n".join(out)


def merge(paths: list) -> dict:
    data = defaultdict(list)
    for p in paths:
        for k, v in load(Path(p)).items():
            data[k].extend(v)
    return data


def report(paths: list, which: str = "full") -> list:
    """One ladder per case of every run alone, then of all runs together (a stage seen in one run only is not a result)."""
    results = []
    for p in paths:
        d = load(Path(p))
        for case in CASES:
            if any(c == case for (_, c) in d):
                r = ladder(d, case, which)
                r["run"] = str(p)
                results.append(r)
    if len(paths) > 1:
        d = merge(paths)
        for case in CASES:
            if any(c == case for (_, c) in d):
                r = ladder(d, case, which)
                r["run"] = "all runs"
                results.append(r)
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("variants")
    v.add_argument("--set", choices=("singles", "full"), default="full")
    s = sub.add_parser("spec")
    s.add_argument("--exe-root", required=True)
    s.add_argument("--set", choices=("singles", "full"), default="full")
    s.add_argument("--out")
    r = sub.add_parser("report")
    r.add_argument("runs", nargs="+")
    r.add_argument("--json", action="store_true")
    r.add_argument("--set", choices=("singles", "full"), default="full")
    a = ap.parse_args(argv)
    if a.cmd == "variants":
        for name, macros, _ in variants(a.set):
            print(f"{name}\t{';'.join(macros)}")
        return 0
    if a.cmd == "spec":
        text = json.dumps(spec(a.exe_root, a.set), indent=1) + "\n"
        if a.out:
            Path(a.out).write_text(text, encoding="utf-8", newline="\n")
        else:
            sys.stdout.write(text)
        return 0
    res = report(a.runs, a.set)
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        for x in res:
            print(f"# {x.get('run', '')}")
            print(format_report([x]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
