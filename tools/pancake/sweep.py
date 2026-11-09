"""sweep.py - pancake-strata MOD_PLAN item 37: coordinate-descent sweep of launch / tile / CPU block knobs (UFMG recipe).

One knob at a time over its candidate values, keeping the best value of each before moving to the next; a pass over
all knobs, repeated while a pass still improved by more than --min-gain %.  Every measurement is one abtest.py run of
the current point (quiet-machine gate, noisy runs repeated, engine-log decode rates); the objective is the mean decode
tok/s of the chosen cases (default short + edit).  The result is printed and written as an env block for the preset.

    python tools/pancake/sweep.py [--knobs default] [--cases short,edit] [--reps 1] --out bench/results/<day>-sweep
Knob sets (--knobs): default = the 7900 XT GPU expert kernels and the CPU pool; or a JSON file {"ENV": ["v1", ...]}
(a key starting with "--" is an engine argument).  The first value of each knob is the starting point.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import abtest  # noqa: E402

DEFAULT = {
    "STRATA_EXPDEC_RB": ["0", "32", "64", "128"],          # gate/up rows per block (0 = auto)
    "STRATA_EXPDEC_MINBLK": ["400", "800", "1600"],        # auto rows-per-block target block count
    "STRATA_EXPDEC_R": ["1", "2"],                         # gate/up rows per sub-warp
    "STRATA_EXPDEC_MINB": ["2", "4"],                      # __launch_bounds__ min blocks
    "STRATA_EXPDEC_DRB": ["0", "64", "128"],               # down rows per block
    "STRATA_Q2_PREFETCH": ["0", "512", "1024", "2048"],    # CPU Q2_0 row prefetch distance (bytes)
    "STRATA_EXPDEC_DR": ["4", "2", "1"],                   # down rows per lane
    "--pool-tasks": ["48", "32", "64", "96"],              # CPU pool tasks per phase (an engine argument)
}


def measure(base, env, cases, reps, work, tag, max_load, foreign):
    arm = {"env": {k: v for k, v in env.items() if not k.startswith("--")},
           "args": {k: v for k, v in env.items() if k.startswith("--")}}
    spec = {"base": base, "arms": {tag: arm}, "cases": cases, "reps": reps}
    sp = work / f"{tag}.spec.json"
    sp.write_text(json.dumps(spec))
    vals = []
    for r in range(reps):
        for attempt in range(3):
            abtest.quiet_gate(max_load, foreign, [])
            cfg = str(work / f"{tag}.json")
            c = abtest.make_config(base, spec["arms"][tag], cfg)
            res, err = abtest.run_arm(tag, cfg, c, cases, abtest.prompts(), str(work / f"{tag}-r{r}-a{attempt}.out"), 8080)
            if err:
                print(f"  {tag}: {err}")
                return None
            if res["noise"] <= foreign:
                vals += [x["decode_tps"] for x in res["rows"]]
                break
            print(f"  {tag}: noisy run (foreign CPU {res['noise']:.0f}%), repeated")
    return sum(vals) / len(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knobs", default="default")
    ap.add_argument("--cases", default="short,edit")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--min-gain", type=float, default=1.0)
    ap.add_argument("--max-load", type=float, default=1.0)
    ap.add_argument("--foreign-cpu", type=float, default=25.0)
    ap.add_argument("--base", default="pancake-flashnext-iq2_xs-rdna3test.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    knobs = DEFAULT if a.knobs == "default" else json.load(open(a.knobs))
    cases = a.cases.split(",")
    work = Path(a.out)
    work.mkdir(parents=True, exist_ok=True)
    base = str(abtest.ROOT / a.base)
    abtest.stop_servers()
    point = {k: v[0] for k, v in knobs.items()}
    n = 0
    best = measure(base, point, cases, a.reps, work, f"p{n}", a.max_load, a.foreign_cpu)
    print(f"start {point}: {best}")
    log = [{"point": dict(point), "tps": best}]
    improved = True
    while improved and best is not None:
        improved = False
        start = best
        for k, cand in knobs.items():
            for v in cand:
                if v == point[k]:
                    continue
                trial = dict(point, **{k: v})
                n += 1
                t = measure(base, trial, cases, a.reps, work, f"p{n}", a.max_load, a.foreign_cpu)
                log.append({"point": trial, "tps": t})
                print(f"  {k}={v}: {t} (best {best:.1f})")
                if t is not None and t > best:
                    best, point = t, trial
            print(f"{k} -> {point[k]} ({best:.1f} tok/s)")
        improved = best > start * (1 + a.min_gain / 100)
    (work / "sweep.json").write_text(json.dumps({"best": point, "tps": best, "log": log}, indent=1))
    print("best:", " ".join(f"{k}={v}" for k, v in point.items()), f"-> {best} tok/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
