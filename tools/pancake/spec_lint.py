#!/usr/bin/env python3
"""Validate bench/specs/*.json arm references against this repo's engine.

Since:
- What: future spec maintenance must not silently invert an arm's env/flag to something our engine never reads.
- Why: the 2026-11-12 spec audit caught `STRATA_MTP_DRAFT_ALL` armed after our cherry-picks, despite our engine not recognising it (inert arm, engine ignores it = baseline).
- How: scan every bench/specs/*.json, extract every arm's env STRATA_* keys and args --flags, and verify each is recognised by this repo (defined in conversation_disk_env.def, read in src/include, or parsed as a CLI flag in generate.cpp/serve).

Does:
- walks bench/specs/*.json;
- for each arm, checks each STRATA_* env key and each --flag reference against this repo's sources;
- prints NON-PHANTOM wrnings for anything not found (ambiguous cache miss, not a delete);
- exit code: 1 if any phantom found, 0 otherwise, 2 on usage/read errors.

Interface:
```
python tools/pancake/spec_lint.py [--dir bench/specs] [--quiet] [--require-predict-from N]
python tools/pancake/spec_lint.py --score SPEC.json OBSERVED.json
```

Pre-registered predictions and stop rules (LEADS.md Part O O13; the idea of writing the expected numbers down before the first timed run and scoring them
afterwards without rescoring; tool only, no document).  A spec may carry two OPTIONAL top-level keys (abtest.py ignores them):
  "predict": [{"arm": "x", "case": "short" (optional), "metric": "pct" (optional, default "pct"), "lo": 1.0, "hi": 4.0, "why": "..." (optional)}, ...]
  "stop":    ["text of a stop rule", ...]      (or [{"if": "...", "then": "..."}])
The lint checks their shape (arm and case names exist, lo <= hi, numbers) and, with --require-predict-from N, demands both keys on every spec numbered N or
more (a new spec after N must say what it expects and when to stop).  --score reads an OBSERVED json ({"<arm>/<case>": value} or {"<arm>": value} per
metric key "pct"; {"<metric>": {"<arm>/<case>": value}} for another) and prints hit / MISS per prediction; the table is the record, nothing is rescored.
Requires `git` on PATH for stale reference checks; exitable no-phantom on empty env.

Reversible:
- read-only diagnostics; deletes no repo state and can be rerun any time.

Caveats:
- a PHANTOM finding is a bisection prompt: a not-found env key may mean "our engine reads it from an unindexed path" (i.e. correct the spec, not the finding);
- acceptable but with real, the intended ink (see NAME), so nothing is written.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Where a STRATA_* key or --flag is truly intended to be found.
ENV_DEF = ROOT / "src" / "core" / "conversation_disk_env.def"
CLI_DIRS = [ROOT / "src" / "program", ROOT / "serve"]


def git_ls() -> list[str]:
    """Files containing sources for the grep. This is the second source of truth."""
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "src", "include", "serve"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        return []
    return [os.path.join(str(ROOT), p) for p in out.stdout.splitlines()]


_TEXT_CACHE: dict = {}


def read_text(path) -> str:
    key = str(path)
    if key not in _TEXT_CACHE:
        try:
            _TEXT_CACHE[key] = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            _TEXT_CACHE[key] = ""
    return _TEXT_CACHE[key]


def check_arm(spec: Path, arm: str, key: str, kind: str, sources: list[str], quiet: bool) -> bool:
    """Return True if `key` is recognised by this repo (not a phantom)."""
    if kind == "env":
        if key == "STRATA_GGML_DIR":
            return True
        if re.search(re.escape(key), read_text(ENV_DEF)):
            return True
    needle = key.lstrip("-") if kind == "args" else key
    for src in sources:
        if needle in read_text(src):
            return True
    return False


def walk_specs(spec_dir: Path) -> list[tuple[Path, str, dict]]:
    for spec_file in sorted(spec_dir.glob("*.json")):
        try:
            data = json.loads(spec_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for arm, body in (data.get("arms") or {}).items():
            yield spec_file, arm, body


def effective_args(preset_args: list, arm_args: dict) -> list:
    """The preset's argv with an arm's `args` applied the way abtest.py does it: a key replaces the flag (and its value),
    "" is a bare switch, null removes."""
    a = [str(x) for x in preset_args]
    for k, v in (arm_args or {}).items():
        if k in a:
            i = a.index(k)
            n = 2 if i + 1 < len(a) and not a[i + 1].startswith("--") else 1
            del a[i:i + n]
        if v is not None:
            a += [k] + ([str(v)] if v != "" else [])
    return a


def flag_value(a: list, flag: str):
    """The value after `flag` in argv `a` (None: the flag is absent; "": a bare switch)."""
    if flag not in a:
        return None
    i = a.index(flag)
    return a[i + 1] if i + 1 < len(a) and not a[i + 1].startswith("--") else ""


def inert_gate(preset_args: list, arm: dict) -> str | None:
    """A reason when the arm turns on a switch that the engine silently leaves off under the arm's own other flags.
    Rule 1 (T-391 found it late): `--kv-grow` / STRATA_KV_GROW=1 needs the whole K/V in VRAM, and the engine says
    "--kv-grow is off" under `--kv-resident N` (generate.cpp, `o.kv_resident <= 0`); an arm that adds it to the daily
    preset (which carries --kv-resident 32768) measures nothing."""
    a = effective_args(preset_args, arm.get("args"))
    env = arm.get("env") or {}
    ev = env.get("STRATA_KV_GROW")
    asked = (ev not in (None, "", "0")) if "STRATA_KV_GROW" in env else ("--kv-grow" in a and "--no-kv-grow" not in a)
    res = flag_value(a, "--kv-resident")
    if asked and res not in (None, "", "0"):
        return f"--kv-grow is off under --kv-resident {res} (the elastic K/V needs the whole K/V in VRAM): the arm is the base"
    return None


def predict_problems(spec: dict) -> list:
    """Shape errors of the optional "predict" / "stop" keys of one parsed spec (empty list: fine or absent)."""
    bad = []
    arms = set((spec.get("arms") or {}).keys())
    cases = set()
    c = spec.get("cases")
    if isinstance(c, dict):
        cases = set(c.keys())
    elif isinstance(c, list):
        cases = {x.get("name") if isinstance(x, dict) else x for x in c}
    pr = spec.get("predict")
    if pr is not None:
        if not isinstance(pr, list) or not pr:
            bad.append('"predict" must be a non-empty list')
        else:
            for i, e in enumerate(pr):
                if not isinstance(e, dict):
                    bad.append(f"predict[{i}] is not an object")
                    continue
                if e.get("arm") not in arms:
                    bad.append(f"predict[{i}] arm {e.get('arm')!r} is not an arm of the spec")
                if e.get("case") is not None and cases and e["case"] not in cases:
                    bad.append(f"predict[{i}] case {e['case']!r} is not a case of the spec")
                lo, hi = e.get("lo"), e.get("hi")
                if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (lo, hi)):
                    bad.append(f"predict[{i}] needs numeric lo and hi")
                elif lo > hi:
                    bad.append(f"predict[{i}] lo {lo} is above hi {hi}")
    st = spec.get("stop")
    if st is not None:
        if not isinstance(st, list) or not st:
            bad.append('"stop" must be a non-empty list')
        else:
            for i, e in enumerate(st):
                ok = (isinstance(e, str) and e.strip()) or (isinstance(e, dict) and e.get("if") and e.get("then"))
                if not ok:
                    bad.append(f"stop[{i}] must be a text or an object with if and then")
    return bad


def score_predictions(spec: dict, observed: dict) -> list:
    """[(label, lo, hi, value or None, 'hit' | 'MISS' | 'no data')] for every prediction of the spec against OBSERVED values."""
    out = []
    for e in spec.get("predict") or []:
        metric = e.get("metric", "pct")
        table = observed.get(metric) if isinstance(observed.get(metric), dict) else (observed if metric == "pct" else {})
        key = e["arm"] + ("/" + e["case"] if e.get("case") else "")
        v = table.get(key)
        label = f"{key} [{metric}]"
        if not isinstance(v, (int, float)):
            out.append((label, e["lo"], e["hi"], None, "no data"))
        else:
            out.append((label, e["lo"], e["hi"], v, "hit" if e["lo"] <= v <= e["hi"] else "MISS"))
    return out


def spec_number(name: str):
    m = re.match(r"(\d+)-", name)
    return int(m.group(1)) if m else None


def preset_argv(spec_file: Path):
    """The `args`-style argv list of the spec's base preset, or None when it cannot be read."""
    try:
        data = json.loads(spec_file.read_text(encoding="utf-8"))
        base = data.get("base")
        if not base:
            return None
        pp = Path(base) if os.path.isabs(base) else ROOT / base
        preset = json.loads(pp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for key in ("args", "argv", "engine_args"):
        if isinstance(preset.get(key), list):
            return preset[key]
    return preset if isinstance(preset, list) else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(ROOT / "bench" / "specs"))
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--require-predict-from", type=int, default=None, metavar="N",
                    help='specs numbered N or more must carry "predict" and "stop" (O13)')
    ap.add_argument("--score", nargs=2, metavar=("SPEC", "OBSERVED"), help="score the spec's predictions against an observed json and exit")
    args = ap.parse_args(argv)

    if args.score:
        try:
            spec = json.loads(Path(args.score[0]).read_text(encoding="utf-8"))
            obs = json.loads(Path(args.score[1]).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as ex:
            print(f"spec_lint: {ex}", file=sys.stderr)
            return 2
        rows = score_predictions(spec, obs)
        for label, lo, hi, v, verdict in rows:
            print(f"{verdict:8s} {label}: predicted [{lo}, {hi}], observed {v}")
        print(f"spec_lint: {sum(r[4] == 'hit' for r in rows)} hit, {sum(r[4] == 'MISS' for r in rows)} missed, {sum(r[4] == 'no data' for r in rows)} without data")
        return 0 if all(r[4] == "hit" for r in rows) else 1

    spec_dir = Path(args.dir)
    if not spec_dir.is_dir():
        print(f"spec_lint: not a directory: {spec_dir}", file=sys.stderr)
        return 2

    sources = git_ls()
    if not sources:
        print("spec_lint: no sources (run inside the repo with git ls-files working)", file=sys.stderr)
        return 2

    missing = 0
    inert = 0
    pbad = 0
    for sf in sorted(spec_dir.glob("*.json")):
        try:
            sd = json.loads(sf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(sd, dict):
            continue
        probs = predict_problems(sd)
        n = spec_number(sf.name)
        if args.require_predict_from is not None and n is not None and n >= args.require_predict_from:
            probs += [f'"{k}" is missing (required from spec {args.require_predict_from})' for k in ("predict", "stop") if k not in sd]
        for pm in probs:
            pbad += 1
            print(f"PREDICT      {sf.name}: {pm}")
    missing += pbad
    presets: dict = {}
    for spec_file, arm, body in walk_specs(spec_dir):
        if spec_file not in presets:
            presets[spec_file] = preset_argv(spec_file)
        if presets[spec_file] is not None:
            why = inert_gate(presets[spec_file], body)
            if why:
                inert += 1
                print(f"INERT-GATE   {spec_file.name} arm({arm}): {why}")
        env = body.get("env") or {}
        flags = body.get("args") or {}
        for key in env:
            if not key.startswith("STRATA_"):
                continue
            if not check_arm(spec_file, arm, key, "env", sources, args.quiet):
                missing += 1
                print(f"PHANTOM-ENV  {spec_file.name} arm({arm}) {key}: not in disk_env.def or src/include")
        for flag in flags:
            if not flag.startswith("--"):
                continue
            if not check_arm(spec_file, arm, flag, "args", sources, args.quiet):
                missing += 1
                print(f"PHANTOM-FLAG {spec_file.name} arm({arm}) {flag}: not parsed in generate.cpp/serve")
    if inert and not args.quiet:
        print(f"spec_lint: {inert} arm(s) turn on a switch the engine leaves off under the arm's own flags (warning only)",
              file=sys.stderr)
    if missing:
        if not args.quiet:
            print(f"spec_lint: {missing} phantom arm reference(s) found (see above)", file=sys.stderr)
        return 1
    if not args.quiet:
        print("spec_lint: all arm env/flag references resolve against this repo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
