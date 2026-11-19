#!/usr/bin/env python3
"""flag_diag.py - "can this flag run on this hardware, and does it help here?"  A few minutes per flag, read-only on the system.

  tools/pancake/flag_diag.sh --hardware                     detect and print the hardware (gfx arch, ROCm, VRAM, PCIe link, CPU, RAM, VMM, hipBLASLt)
  tools/pancake/flag_diag.sh --list                         the candidate flags and what each needs
  tools/pancake/flag_diag.sh run --config BASE.json [--flags a,b] [--abtest PATH] [--out DIR] [--ctest BUILD_DIR] [--dry-run]
  tools/pancake/flag_diag.sh --write-recommended [--out DIR] [--gate-pass FLAG,FLAG]

`run`: (a) detects the hardware; (b) for each candidate whose hardware need is met builds ONE abtest.py screen spec: the base arm is the
config with every candidate key removed (bare), each flag is its own arm (a bare config + that flag's env / args), cases `short` and
`long3k` (decode and prompt tok/s, text hash vs base); an arm that does not start or dies is UNSUPPORTED, a failed parity ctest
(--ctest, the flag's own tests of a build tree) is FAIL; (c) the verdict per flag:

  UNSUPPORTED  the hardware lacks what it needs, or the engine did not start / faulted with it
  FAIL         a bitwise parity ctest of a flag claimed exact failed
  NO-GAIN      it runs, and its best case is under the bar (or worse)
  GAIN         >= 2.0 % (text identical to base: exact) or >= 3.0 % (lossy) on a case, and not worse than -3 % on another
  OK           it starts and runs; nothing was measured (--dry-run / no abtest result)

and writes DIR/hardware.json, DIR/verdicts.json and DIR/recommended-flags.json (the preset env / args block for this hardware: only
GAIN flags that are exact, or lossy with a passing quality gate; the gate verdicts recorded in the pancake ledger are listed in
CANDIDATES["gate"], any other lossy flag needs --gate-pass).  `--write-recommended` copies that into rdna3/recommended-flags/, where
rdna3/run.py reads it (rdna3/tuned_flags.py).  The measurements are screens: confirm a GAIN (5 reps) before relying on it.
Tests: python -m pytest tools/pancake/test_flag_diag.py (fake hardware and fake abtest output)."""
import argparse
import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rdna3"))
import tuned_flags as tf  # noqa: E402

# name: env / args the flag sets, exact (bitwise) or lossy, the hardware it needs (predicate names below), the ledger gate
# that already passed for a lossy flag, parity ctests of a build tree (regex for ctest -R), the case that shows it
CANDIDATES = {
    "POOL_FUSE_PHASES": {"env": {"STRATA_POOL_FUSE_PHASES": "1"}, "exact": True, "needs": [], "ctest": "pool"},
    "VERIFY_ECAP": {"env": {"STRATA_VERIFY_ECAP": "0", "STRATA_VERIFY_ECAP_MIN_ROWS": "6"}, "exact": False, "needs": [],
                    "gate": "T-589,T-591", "ctest": "verify_ecap_test"},
    "PF_F16": {"env": {"STRATA_PF_F16": "1"}, "exact": False, "needs": ["wmma"], "gate": "T-828"},
    "HIP_WMMA": {"env": {"STRATA_HIP_WMMA": "1"}, "exact": False, "needs": ["wmma"]},
    "SELECT_WMMA": {"env": {"STRATA_SELECT_WMMA": "1"}, "exact": False, "needs": ["wmma"]},
    "KV_HOST_DMA": {"env": {"STRATA_KV_HOST_DMA": "1"}, "exact": True, "needs": ["vmm"]},
    "PREFILL_RING": {"env": {"STRATA_PREFILL_RING": "192"}, "exact": True, "needs": []},
    # the table is judged against the 100200 table this tree ships (base_env: what the base arm runs with instead of "no value")
    "HIPBLASLT_TUNING": {"env": {"STRATA_HIPBLASLT_TUNING": "@STRATA@/tools/hip/gfx1100-hipblaslt-100401.txt"}, "exact": False,
                         "base_env": {"STRATA_HIPBLASLT_TUNING": "@STRATA@/tools/hip/gfx1100-hipblaslt-100200.txt"},
                         "needs": ["hipblaslt", "gfx1100"]},
    # the learned profile of the pancake line (Qwen3.8-Flash-Next IQ2_XS only; T-194 measured it on another GPU) against the profile this tree ships
    "EXPERT_PROFILE": {"args": {"--expert-profile": "@STRATA@/data/expert-profile-pancake.bin"}, "exact": False,
                       "base_args": {"--expert-profile": "@STRATA@/data/expert-profile.bin"}, "needs": []},
    "GPU_MAX_HW_QUEUES": {"env": {"GPU_MAX_HW_QUEUES": "6"}, "exact": True, "needs": []},
    "RDNA_MMVQ_WAVE": {"env": {"STRATA_RDNA_MMVQ_WAVE": "3"}, "exact": True, "needs": ["rdna"]},
}
DEFAULT_FLAGS = ["POOL_FUSE_PHASES", "VERIFY_ECAP", "PF_F16", "KV_HOST_DMA", "PREFILL_RING", "HIPBLASLT_TUNING", "RDNA_MMVQ_WAVE"]
CASES = ["short", "long3k"]
BAR_EXACT, BAR_LOSSY, FLOOR = 2.0, 3.0, -3.0


def predicate(name, hw):
    g = hw.get("gfx", "")
    if name == "wmma":
        return bool(re.match(r"gfx1[12]", g))      # RDNA3 / RDNA4 matrix cores
    if name == "rdna":
        return bool(re.match(r"gfx1[0-2]", g))
    if name == "vmm":
        return bool(hw.get("vmm"))
    if name == "hipblaslt":
        return bool(hw.get("hipblaslt"))
    return g == name


def unmet(flag, hw):
    return [n for n in CANDIDATES[flag]["needs"] if not predicate(n, hw)]


# ---------------------------------------------------------------------------------------------- the spec
def keys_of(flags):
    env, args = set(), set()
    for f in flags:
        env |= set(CANDIDATES[f].get("env", {}))
        args |= set(CANDIDATES[f].get("args", {}))
    return env, args


def make_spec(base_cfg_path, flags, root=ROOT):
    """One abtest spec: base = every candidate key removed; one arm per flag."""
    env_keys, arg_keys = keys_of(flags)
    bare = {"env": {k: None for k in env_keys}, "args": {k: None for k in arg_keys}}
    for f in flags:   # a flag judged against another value (not against "unset") keeps that value in the base arm
        bare["env"].update({k: v.replace("@STRATA@", str(root)) for k, v in CANDIDATES[f].get("base_env", {}).items()})
        bare["args"].update({k: v.replace("@STRATA@", str(root)) for k, v in CANDIDATES[f].get("base_args", {}).items()})
    arms = {"base": copy.deepcopy(bare)}
    for f in flags:
        a = copy.deepcopy(bare)
        for k, v in CANDIDATES[f].get("env", {}).items():
            a["env"][k] = v.replace("@STRATA@", str(root))
        for k, v in CANDIDATES[f].get("args", {}).items():
            a["args"][k] = v.replace("@STRATA@", str(root))
        arms[f.lower().replace("_", "-")] = a
    return {"base": str(base_cfg_path), "lossy": True, "prefill": True, "cases": CASES, "reps": 1, "arms": arms,
            "note": "flag_diag.py: bare base (every candidate key removed) against one arm per flag; screen tier"}


# ---------------------------------------------------------------------------------------------- parsing abtest output
LINE = re.compile(r"^\s+(\S+)\s+(\S+)\s+(-?[\d.]+) \+-\s+(-?[\d.]+|nan)\s+\(n=(\d+)\)\s*(?:vs base:\s+([-+]?[\d.]+)%.*->\s+(better|worse|no difference))?")
TEXT = re.compile(r"^\s+(\S+)\s+(identical on every case|DIFFERENT text)")


def parse_abtest(text):
    """{arm: {"cases": {case: {"mean", "delta", "verdict"}}, "exact": bool|None}}; arms without a row are absent."""
    out = {}
    in_text = False
    for ln in text.splitlines():
        if ln.startswith("generated text vs base"):
            in_text = True
            continue
        if in_text:
            m = TEXT.match(ln)
            if m:
                out.setdefault(m.group(1), {"cases": {}, "exact": None})["exact"] = m.group(2).startswith("identical")
            continue
        m = LINE.match(ln)
        if m:
            case, arm = m.group(1), m.group(2)
            d = out.setdefault(arm, {"cases": {}, "exact": None})
            d["cases"][case] = {"mean": float(m.group(3)), "delta": float(m.group(6)) if m.group(6) else 0.0,
                                "verdict": m.group(7) or "base"}
    return out


def crashed(text, arm):
    return any(arm in ln and re.search(r"crash|died|no clean runs|never ready|failed to start", ln, re.I) for ln in text.splitlines())


# ---------------------------------------------------------------------------------------------- verdicts
def verdict_for(flag, hw, parsed, stdout="", ctest_ok=None, gate_pass=()):
    c = CANDIDATES[flag]
    arm = flag.lower().replace("_", "-")
    miss = unmet(flag, hw)
    e = {"env": {k: v for k, v in c.get("env", {}).items()}, "args": dict(c.get("args", {})), "claimed_exact": c["exact"]}
    if miss:
        e.update(verdict="UNSUPPORTED", note="hardware lacks: " + ",".join(miss))
        return e
    if ctest_ok is False:
        e.update(verdict="FAIL", note="a parity ctest of this flag failed")
        return e
    a = parsed.get(arm)
    if a is None or not a["cases"]:
        if not stdout:
            e.update(verdict="OK", note="nothing measured")
        else:
            e.update(verdict="UNSUPPORTED", note="the arm gave no result rows" + (" (crash line in the abtest output)" if crashed(stdout, arm) else ""))
        return e
    deltas = {k: v["delta"] for k, v in a["cases"].items()}
    best = max(deltas.values())
    worst = min(deltas.values())
    exact = bool(a["exact"]) if a["exact"] is not None else False
    e["exact"] = exact
    e["gain_pct"] = round(best, 2)
    e["cases"] = deltas
    bar = BAR_EXACT if exact else BAR_LOSSY
    better = any(v["verdict"] == "better" and v["delta"] >= bar for v in a["cases"].values())
    if better and worst >= FLOOR:
        e["verdict"] = "GAIN"
    else:
        e["verdict"] = "NO-GAIN"
    gates = [g for g in (c.get("gate", "") or "").split(",") if g]
    if not exact and (gates or flag in gate_pass):
        e["gate_pass"] = True
        e["note"] = "quality gate: " + (",".join(gates) if gates else "--gate-pass")
    if c["exact"] and not exact:
        e["note"] = (e.get("note", "") + " claimed exact but the text differs: treated as lossy").strip()
    return e


def verdicts_for(flags, hw, stdout, ctest=None, gate_pass=()):
    parsed = parse_abtest(stdout) if stdout else {}
    return {f: verdict_for(f, hw, parsed, stdout, (ctest or {}).get(f), gate_pass) for f in flags}


def recommended(hw, verdicts):
    return tf.build_recommended(hw, verdicts)


# ---------------------------------------------------------------------------------------------- running
def run_ctest(build_dir, flags, runner=subprocess.run):
    out = {}
    for f in flags:
        rx = CANDIDATES[f].get("ctest")
        if not rx or not build_dir:
            continue
        r = runner(["ctest", "--test-dir", str(build_dir), "-R", rx, "--output-on-failure"], capture_output=True, text=True, timeout=600)
        out[f] = r.returncode == 0
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", default="run", choices=["run"])
    ap.add_argument("--hardware", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--write-recommended", action="store_true")
    ap.add_argument("--config", help="a server config JSON (pancake format: exe, args, env, ...) to run the arms on")
    ap.add_argument("--flags", help="comma list of candidate names (default: %s)" % ",".join(DEFAULT_FLAGS))
    ap.add_argument("--abtest", default=str(ROOT / "tools" / "pancake" / "abtest.py"))
    ap.add_argument("--out", default=str(ROOT / "flag-diag-out"))
    ap.add_argument("--ctest", help="a build directory whose ctest parity tests are run per flag")
    ap.add_argument("--gate-pass", default="", help="lossy flags whose quality gate passed (besides CANDIDATES[...]['gate'])")
    ap.add_argument("--dry-run", action="store_true", help="write the spec and hardware, start nothing (verdict OK / UNSUPPORTED by hardware only)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.list:
        for k, c in CANDIDATES.items():
            print("%-18s %-6s needs=%-18s %s" % (k, "exact" if c["exact"] else "lossy", ",".join(c["needs"]) or "-", json.dumps(c.get("env", {}))))
        return 0
    hw = tf.detect_hardware()
    if a.hardware:
        print(json.dumps(hw, indent=1))
        return 0
    if a.write_recommended:
        v = json.loads((out / "verdicts.json").read_text())
        p = tf.write_recommended(hw, v)
        print("wrote", p)
        return 0
    flags = [f.strip().upper() for f in a.flags.split(",")] if a.flags else DEFAULT_FLAGS
    bad = [f for f in flags if f not in CANDIDATES]
    if bad:
        sys.exit("unknown flag(s): " + ",".join(bad))
    if not a.config:
        sys.exit("run needs --config BASE.json")
    out.mkdir(parents=True, exist_ok=True)
    (out / "hardware.json").write_text(json.dumps(hw, indent=1) + "\n")
    runnable = [f for f in flags if not unmet(f, hw)]
    spec = make_spec(a.config, runnable)
    (out / "spec.json").write_text(json.dumps(spec, indent=1) + "\n")
    stdout = ""
    if not a.dry_run and runnable:
        r = subprocess.run([sys.executable, a.abtest, str(out / "spec.json"), "--tier", "screen", "--out", str(out / "rows.jsonl")],
                           capture_output=True, text=True)
        stdout = r.stdout + r.stderr
        (out / "abtest.stdout").write_text(stdout)
    ct = run_ctest(a.ctest, flags) if a.ctest and not a.dry_run else {}
    v = verdicts_for(flags, hw, stdout, ct, [g.strip().upper() for g in a.gate_pass.split(",") if g.strip()])
    (out / "verdicts.json").write_text(json.dumps(v, indent=1) + "\n")
    (out / "recommended-flags.json").write_text(json.dumps(recommended(hw, v), indent=1) + "\n")
    for f, e in v.items():
        print("%-18s %-12s %s" % (f, e["verdict"], e.get("note", "") or ("gain %.1f%%" % e["gain_pct"] if "gain_pct" in e else "")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
