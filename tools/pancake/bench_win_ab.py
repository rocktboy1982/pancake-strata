"""Windows A/B for strata-poly: runs tools/pancake/bench_win.py once per arm of a spec and writes one summary.
bench_win.py takes ONE arm (extra strata-poly args on its command line, env from the process); this wrapper only reads
a spec (bench/specs/159-*.json, 160-*.json), runs the arms one after the other in the spec's order (base first and
base-again last, so drift shows), and keeps each arm's stdout, bench_rows.json and bench.err in <out>/<arm>/.
It measures nothing itself and starts no process but bench_win.py (one at a time).  Windows counterpart of abtest.py for
the poly engine on this PC; abtest.py needs /proc and Linux.
usage: bench_win_ab.py SPEC.json --out DIR [--runs N] [--warmup N] [--measured N] [--max-new N] [--only a,b] [--dry]
env:   STRATA_POLY_EXE, STRATA_MODEL, STRATA_TOKENIZER, CUDA_BIN (as bench_win.py)
spec:  {"runner": "bench_win_ab", "runs": 3, "warmup": 3, "measured": 3, "max_new": 96,
        "arms": {"base": {}, "x": {"args": {"--threads": "4", "--cpu-multi": ""}, "env": {"STRATA_X": "1"}}}}
       an arg whose value is "" is a flag without a value"""
import argparse, json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
LINE = re.compile(r"^(\w+)\s+decode\s+([\d.]+) \+/- \s*([\d.nan]+) tok/s \(95%, n=(\d+) runs\).*time-to-first-token ([\d.]+) s")


def arm_argv(body):
    out = []
    for flag, val in (body.get("args") or {}).items():
        out.append(flag)
        if val != "":
            out.append(str(val))
    return out


def parse_summary(text):
    res = {}
    for ln in text.splitlines():
        m = LINE.match(ln.strip())
        if m:
            res[m.group(1)] = {"decode_tps": float(m.group(2)), "ci95": m.group(3), "n_runs": int(m.group(4)),
                               "ttft_s": float(m.group(5))}
    return res


def plan(spec, a):
    runs = a.runs or spec.get("runs", 3)
    warm = a.warmup if a.warmup is not None else spec.get("warmup", 3)
    meas = a.measured or spec.get("measured", 3)
    new = a.max_new or spec.get("max_new", 96)
    names = list(spec["arms"])
    if a.only:
        keep = set(a.only.split(","))
        names = [n for n in names if n in keep]
    return [(n, [sys.executable, os.path.join(HERE, "bench_win.py"), str(runs), str(warm), str(meas), str(new),
                 *arm_argv(spec["arms"][n])], dict(spec["arms"][n].get("env") or {})) for n in names]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int)
    ap.add_argument("--warmup", type=int)
    ap.add_argument("--measured", type=int)
    ap.add_argument("--max-new", type=int)
    ap.add_argument("--only")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    spec = json.load(open(a.spec, encoding="utf-8"))
    steps = plan(spec, a)
    if a.dry:
        for n, cmd, env in steps:
            print(n, " ".join(cmd[1:]), json.dumps(env) if env else "")
        return 0
    for k in ("STRATA_POLY_EXE", "STRATA_MODEL", "STRATA_TOKENIZER"):
        if k not in os.environ:
            print("bench_win_ab: missing env " + k, file=sys.stderr)
            return 2
    os.makedirs(a.out, exist_ok=True)
    summary = {"spec": os.path.basename(a.spec), "arms": {}}
    rc_all = 0
    for n, cmd, env in steps:
        d = os.path.join(a.out, n)
        os.makedirs(d, exist_ok=True)
        e = dict(os.environ, BENCH_OUT=d)
        e.update(env)
        t0 = time.time()
        with open(os.path.join(d, "stdout.txt"), "w", encoding="utf-8") as so:
            rc = subprocess.call(cmd, env=e, stdout=so, stderr=subprocess.STDOUT)
        text = open(os.path.join(d, "stdout.txt"), encoding="utf-8", errors="replace").read()
        summary["arms"][n] = {"exit": rc, "wall_s": round(time.time() - t0), "cases": parse_summary(text),
                              "args": arm_argv(spec["arms"][n]), "env": env}
        print(f"arm {n}: exit {rc}, {summary['arms'][n]['cases']}", flush=True)
        rc_all = rc_all or rc
        json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    return 1 if rc_all else 0


if __name__ == "__main__":
    sys.exit(main())
