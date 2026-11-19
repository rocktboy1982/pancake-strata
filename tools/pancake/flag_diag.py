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

Exactness (owner order 2026-11-20): on this model the default's text differs from ITSELF from run to run (placement), so text
identity cannot judge a flag claimed exact.  When such a flag's text differs, `run` starts the default once more (arm base-again):
if that repeat differs too, the flag keeps its speed verdict (exact bar) with exactness "EXACT-UNVERIFIED (base vs base also
differs)"; if the repeat is identical, it is judged as lossy (as before).  The judge of exactness is the flag's parity ctest
(--ctest): FAIL is the verdict, PASS is recorded as parity_ctest.  A text difference alone never makes a flag UNSUPPORTED.

and writes DIR/hardware.json, DIR/verdicts.json and DIR/recommended-flags.json (the preset env / args block for this hardware: only
GAIN flags that are exact, or lossy with a passing quality gate; the gate verdicts recorded in the pancake ledger are listed in
CANDIDATES["gate"], any other lossy flag needs --gate-pass).  `--write-recommended` copies that into rdna3/recommended-flags/, where
rdna3/run.py reads it (rdna3/tuned_flags.py).  The measurements are screens: confirm a GAIN (5 reps) before relying on it.
Since 2026-11-19 (the US chain's diagnostic ran 18 min for 2 flags and left its abtest / server / engine running after the chain's
timeout): `run` starts ONE abtest.py screen per arm (base once, then each flag; case `short` only, no retries, quiet wait <= 30 s),
each in its own process group with a hard limit (--flag-timeout, default 240 s; past it: UNTESTED-TIMEOUT and the group is stopped,
tools/pancake/diag_group.py), rewrites DIR/verdicts.json after every arm, runs the parity ctest only for flags claimed exact, and
on TERM stops the running arm and marks the rest UNTESTED-TIMEOUT (exit 124).
Tests: python -m pytest tools/pancake/test_flag_diag.py (fake hardware and fake abtest output)."""
import argparse
import copy
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rdna3"))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # diag_group
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
    arms = {"base": copy.deepcopy(bare), "base-again": copy.deepcopy(bare)}   # base-again: the default's own repeat (exactness, noise)
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
def verdict_for(flag, hw, parsed, stdout="", ctest_ok=None, gate_pass=(), base_differs=None):
    """BASE_DIFFERS: True when a repeat of the default gave another text than the default (text identity unusable here)."""
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
    if c["exact"] and not exact and base_differs:
        exact = True
        e["exactness"] = "EXACT-UNVERIFIED (base vs base also differs)"
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


# ---------------------------------------------------------------------------------------------- rows -> verdict input
def parse_rows(base_jsonl, arm_jsonl, arm):
    """The parse_abtest structure for one arm from two abtest.py jsonl files (base run, arm run): per case the mean decode
    (prompt for a long* case) tok/s, its delta % to base, and exact = the generated text hashes equal base's on every case."""
    def load(text):
        rows, texts = {}, {}
        for ln in (text or "").splitlines():
            ln = ln.strip()
            if not ln.startswith("{"):
                continue
            try:
                rec = json.loads(ln)
            except ValueError:
                continue
            for r in rec.get("rows") or []:
                key = "prompt_tps" if str(r.get("case", "")).startswith("long") else "decode_tps"
                if r.get(key):
                    rows.setdefault(r["case"], []).append(float(r[key]))
            texts.update(rec.get("texts") or {})
        return {c: sum(v) / len(v) for c, v in rows.items()}, texts
    b, bt = load(base_jsonl)
    a, at = load(arm_jsonl)
    if not a:
        return {}
    cases = {}
    for c, m in a.items():
        d = (m / b[c] - 1.0) * 100.0 if b.get(c) else 0.0
        cases[c] = {"mean": m, "delta": round(d, 2), "verdict": "better" if d > 0 else "worse" if d < 0 else "no difference"}
    common = [c for c in at if c in bt]
    return {arm: {"cases": cases, "exact": (all(at[c] == bt[c] for c in common) if common else None)}}


# ---------------------------------------------------------------------------------------------- running

def ctest_not_run(text):
    """True when ctest ran nothing it could judge: the test binaries are not built ("Not Run") or no test matched.  Not a FAIL:
    the 2026-11-19 RD diagnostic called POOL_FUSE_PHASES FAIL from 7 pool tests that were all "Not Run" (no test build)."""
    return bool(text) and ("Not Run" in text or "No tests were found" in text)


def run_ctest(build_dir, flags, runner=subprocess.run, timeout=600, out_dir=None):
    """Parity ctests of the flags CLAIMED EXACT only (a lossy flag's text differs by design; its quality gate is the ledger's).
    With out_dir: each ctest runs in its own process group with `timeout` (diag_group); a timeout counts as not run (None)."""
    out = {}
    for f in flags:
        rx = CANDIDATES[f].get("ctest")
        if not rx or not build_dir or not CANDIDATES[f]["exact"]:
            continue
        cmd = ["ctest", "--test-dir", str(build_dir), "-R", rx, "--output-on-failure"]
        if out_dir is not None:
            import diag_group
            log = Path(out_dir) / ("ctest-%s.log" % f.lower())
            rc, to, _ = diag_group.run(cmd, log, timeout, say=lambda s: print(s, flush=True))
            text = log.read_text(errors="replace") if log.exists() else ""
            out[f] = None if to or ctest_not_run(text) else rc == 0
        else:
            r = runner(cmd, capture_output=True, text=True, timeout=timeout)
            out[f] = None if ctest_not_run(getattr(r, "stdout", "") or "") else r.returncode == 0
    return out


class Stopped(Exception):
    """The diagnostic got TERM (the chain's time limit): the arm running now is stopped, the rest is UNTESTED-TIMEOUT."""


def _arm_runner(cmd, log, timeout_s):
    import diag_group
    rc, to, _ = diag_group.run(cmd, log, timeout_s, say=lambda s: print(s, flush=True))
    return rc, to


def apply_ctest(e, ok):
    """The parity ctest is the judge of exactness: FAIL is the verdict, PASS is recorded (and settles an unverified text check)."""
    if ok is False:
        e.update(verdict="FAIL", parity_ctest="FAIL", note="a parity ctest of this flag failed")
    elif ok is True:
        e["parity_ctest"] = "PASS"
        if str(e.get("exactness", "")).startswith("EXACT-UNVERIFIED"):
            e["exactness"] = "EXACT (parity ctest PASS; text identity not usable: base vs base also differs)"
    elif ok is None:
        e["note"] = (e.get("note", "") + " parity ctest: not run (no result in time, or the test binaries are not built)").strip()
    return e


def run_arms(cfg, flags, hw, out, abtest, flag_timeout=240.0, runner=None, ctest=None, ctest_timeout=300.0, gate_pass=(),
             write=None):
    """One abtest.py screen per arm (base first, then each flag), case `short` only, one start each, no retries, a hard
    per-arm time limit (`flag_timeout`, in its own process group: diag_group).  After every arm DIR/verdicts.json is written
    again, so a chain that stops the diagnostic still finds the verdicts so far.  Verdicts: as verdict_for, plus
    UNTESTED-TIMEOUT (no result within flag_timeout, or the whole diagnostic was stopped) and UNTESTED (no base, or abtest refused)."""
    runner = runner or _arm_runner
    out = Path(out)
    spec = make_spec(cfg, flags)
    v = {f: dict(verdict_for(f, hw, {}, ""), verdict="UNTESTED", note="pending: not measured yet") if not unmet(f, hw)
         else verdict_for(f, hw, {}, "") for f in flags}
    runnable = [f for f in flags if not unmet(f, hw)]

    def save(final=False):
        doc = dict(v)
        tmp = out / "verdicts.json.tmp"
        tmp.write_text(json.dumps(doc, indent=1) + "\n")
        os.replace(tmp, out / "verdicts.json")
        if write:
            write(doc, final)

    def one(arm):
        sub = dict(spec, arms={arm: spec["arms"][arm]}, cases=["short"], prefill=False, reps=1,
                   note="flag_diag.py: one arm, one start, case short (screen)")
        sp = out / ("spec-%s.json" % arm)
        sp.write_text(json.dumps(sub, indent=1) + "\n")
        jl = out / ("rows-%s.jsonl" % arm)
        if jl.exists():
            jl.unlink()
        cmd = [sys.executable, "-u", str(abtest), str(sp), "--tier", "screen", "--retries", "0", "--max-wait", "30",
               "--out", str(jl)]
        t0 = time.time()
        rc, to = runner(cmd, out / ("abtest-%s.stdout" % arm), flag_timeout)
        print("arm %-20s exit=%s%s %.0f s" % (arm, rc, " TIMEOUT" if to else "", time.time() - t0), flush=True)
        log = out / ("abtest-%s.stdout" % arm)
        return rc, to, (jl.read_text() if jl.exists() else ""), (log.read_text(errors="replace") if log.exists() else "")

    save()
    current = None
    again = {}
    try:
        if not runnable:
            return v
        current = runnable
        rc, to, base_rows, base_log = one("base")
        if not parse_rows(base_rows, base_rows, "base"):
            why = ("no base result within %g s (engine start + short prompt)" % flag_timeout if to else
                   "abtest refused to start (another test or engine is running)" if rc == 3 else
                   "the base arm gave no result rows (abtest exit %s)" % rc)
            for f in runnable:
                v[f].update(verdict="UNTESTED-TIMEOUT" if to else "UNTESTED", note=why)
            save(True)
            return v
        for f in runnable:
            current = runnable[runnable.index(f):]
            arm = f.lower().replace("_", "-")
            rc, to, rows, log = one(arm)
            parsed = parse_rows(base_rows, rows, arm)
            if not parsed and to:
                v[f].update(verdict="UNTESTED-TIMEOUT", note="no result within %g s per flag (engine start + short prompt); its process group was stopped" % flag_timeout)
            elif not parsed and rc == 3:
                v[f].update(verdict="UNTESTED", note="abtest refused to start (another test or engine is running)")
            else:
                if CANDIDATES[f]["exact"] and parsed.get(arm, {}).get("exact") is False and "differs" not in again:
                    # does the default differ from itself here?  one more start of the default, once per diagnostic
                    _, _, arows, _ = one("base-again")
                    ap = parse_rows(base_rows, arows, "base-again").get("base-again")
                    again["differs"] = (ap["exact"] is False) if ap else None
                v[f] = verdict_for(f, hw, parsed, log or "x", None, gate_pass, again.get("differs"))
            save()
        current = None
        if ctest:
            ct = run_ctest(ctest, runnable, timeout=ctest_timeout, out_dir=out)
            for f, ok in ct.items():
                apply_ctest(v[f], ok)
        save(True)
        return v
    except (Stopped, KeyboardInterrupt):
        for f in current or []:
            if v[f]["verdict"] == "UNTESTED":
                v[f].update(verdict="UNTESTED-TIMEOUT", note="the diagnostic was stopped (its time limit) before this flag finished")
        save(True)
        raise


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
    ap.add_argument("--flag-timeout", type=float, default=240.0,
                    help="seconds per arm (one engine start + the short case); past it the arm's process group is stopped: UNTESTED-TIMEOUT")
    ap.add_argument("--ctest-timeout", type=float, default=300.0, help="seconds per parity ctest (flags claimed exact only)")
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
    gp = [g.strip().upper() for g in a.gate_pass.split(",") if g.strip()]
    if a.dry_run:
        v = verdicts_for(flags, hw, "", {}, gp)
        (out / "verdicts.json").write_text(json.dumps(v, indent=1) + "\n")
    else:
        def on_term(*_):
            raise Stopped()
        signal.signal(signal.SIGTERM, on_term)
        try:
            v = run_arms(a.config, flags, hw, out, a.abtest, a.flag_timeout, ctest=a.ctest, ctest_timeout=a.ctest_timeout, gate_pass=gp)
        except (Stopped, KeyboardInterrupt):
            print("flag_diag: stopped; partial verdicts in %s" % (out / "verdicts.json"), flush=True)
            return 124
    (out / "recommended-flags.json").write_text(json.dumps(recommended(hw, v), indent=1) + "\n")
    for f, e in v.items():
        print("%-18s %-12s %s" % (f, e["verdict"], e.get("note", "") or ("gain %.1f%%" % e["gain_pct"] if "gain_pct" in e else "")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
