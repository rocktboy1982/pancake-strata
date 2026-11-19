#!/usr/bin/env python3
"""flag_diag.py - "can this flag run on this hardware, and does it help here?" (helper of tools/pancake/flag_diag.sh).

Read-only on the system: it reads sysfs / proc / rocminfo, starts the engine as your user through the A/B harness and writes
only into --out.  No sudo, no sysctl writes, no change to the installed files.

    flag_diag.py hw                                       print the detected hardware as JSON
    flag_diag.py list                                     the candidate flags, what each needs, claimed exact or lossy
    flag_diag.py run --config PRESET.json --out DIR [--only A,B] [--budget-min 15] [--harness PATH]
    flag_diag.py write-recommended --verdicts DIR/verdicts.json --out FILE [--merge]

Per flag: (a) hardware check without starting anything (UNSUPPORTED with the reason: wrong architecture, no hipBLASLt, ...);
(b) one engine start with the flag next to two starts of the default (the second is the noise floor), 2 short prompts each,
through tools/pancake/abtest.py: the engine must start and answer (smoke); a flag claimed exact must produce the same text
as the default (parity); decode / prompt tok/s against the default (quick screen);
(c) verdict:  UNSUPPORTED (cannot run here, or failed to start)
              FAIL        (a parity ctest of a flag claimed exact failed: run --ctest BUILD_DIR)
              NO-GAIN     (runs, below the bar or inside the noise floor)
              GAIN        (exact and >= 2 % end to end, or lossy and >= 3 %)
              UNTESTED    (budget ended before this flag)
Exactness (owner order 2026-11-20): on this model the text of the default against ITSELF differs from run to run (placement), so
text identity cannot judge a flag claimed exact.  When such a flag's text differs, the default is started once more (arm
base-again): if that repeat differs too, the flag keeps its speed verdict with `exactness: EXACT-UNVERIFIED (base vs base also
differs)`; if the repeat is identical, the flag is treated as lossy (its speed verdict against the lossy bar).  The judge of
exactness is the flag's parity ctest (`ctest` in CANDIDATES, run with --ctest BUILD_DIR): PASS is recorded, FAIL is the verdict.
A text difference alone never makes a flag UNSUPPORTED.
A lossy GAIN enters the recommended file only with a passing quality gate (`quality_gate: PASS`, from the ledger or a run of
tools/pancake/quality_gate.py); the quick screen cannot judge quality.  The file format and the rule are
terminal/strata_control/tuned_flags.py (`accepted`), which the normal start path applies.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))   # diag_group
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "terminal"))
from strata_control import tuned_flags  # noqa: E402

GFX11 = ("gfx11",)
GFX11_12 = ("gfx11", "gfx12")
ANY = ()


def C(name, kind, value, exact, metric, needs, ledger, gate=None, note="", extra=None, ctest=None):
    """`extra`: more environment settings the flag needs in the same arm (a switch that only acts together with others); a
    recommended profile gets them as flags too.  `ctest`: the regex (ctest -R) of the flag's parity tests in a build tree."""
    return {"name": name, "kind": kind, "value": value, "exact": exact, "metric": metric, "needs": needs,
            "ledger": ledger, "quality_gate": gate, "note": note, "extra": extra or {}, "ctest": ctest}


# The candidates: the adopted flags ported to this branch, plus the hardware-dependent ones.  `needs`: gfx prefixes
# (empty = any AMD card) and extra requirements ("hipblaslt": the library and a table for this architecture).
CANDIDATES: list[dict[str, Any]] = [
    C("STRATA_RDNA_Q6K", "env", "1", True, "decode", {"gfx": GFX11, "wave": 32}, "T-009, T-079", ctest="^rdna_mmvq_parity$"),
    C("STRATA_HIPBLASLT_TUNING", "env", "<ROOT>/terminal/tools/hip/gfx1100-hipblaslt-100401.txt", True, "prompt",
      {"gfx": ("gfx1100",), "hipblaslt": True}, "T-096"),
    C("--pcie-frac", "arg", "0", True, "decode", {"gfx": ANY, "pcie_link": True}, "T-197, T-268"),
    C("--adapt-every", "arg", "1", True, "decode", {"gfx": ANY}, "T-199 (no gain without --pcie-frac 0)"),
    C("--lookup-max-t", "arg", "12", False, "decode", {"gfx": ANY}, "T-126, T-127", gate="PASS"),
    C("STRATA_PF_F16", "env", "1", False, "prompt", {"gfx": GFX11}, "T-820, T-828", gate="PASS",
      note="rate and event rules of the gate PASS; the near-tie column FAILS (information)"),
    C("STRATA_HIP_WMMA", "env", "1", True, "decode", {"gfx": GFX11_12}, "T-072 (no-op in the preset)"),
    # the ported hyper-connection read, wave and fusion switches (updated-strata): the ledger has the kernel bundle (T-009) and
    # single numbers where noted; the diagnostic measures each here
    C("STRATA_GR_RDNA", "env", "1", True, "decode", {"gfx": GFX11, "wave": 32}, "T-009 (kernel bundle)", ctest="^rdna_gr_parity$"),
    C("STRATA_GR_RDNA_Q8", "env", "1", False, "decode", {"gfx": GFX11, "wave": 32}, "T-009, T-378",
      extra={"STRATA_GR_RDNA": "1"}),
    C("STRATA_GR_FUSED", "env", "1", False, "decode", {"gfx": GFX11, "wave": 32}, "T-378",
      extra={"STRATA_GR_RDNA": "1", "STRATA_GR_RDNA_Q8": "1", "STRATA_GR_RDNA_A8": "1", "STRATA_GR_INJ_UP": "1"}),
    C("STRATA_RDNA_MMVQ_WAVE", "env", "3", True, "decode", {"gfx": GFX11, "wave": 32}, "T-009, T-072 (mode 1 is 4.5% slower than 3)"),
    C("STRATA_COMBINE_CPU", "env", "1", True, "decode", {"gfx": ANY}, "T-009", extra={"STRATA_WAIT_FUSE": "1"}),
    C("STRATA_ROUTE_PUBLISH", "env", "1", True, "decode", {"gfx": ANY}, "T-009 (21.60 to 21.51 ms per window)"),
    C("STRATA_POOL_FUSE_PHASES", "env", "1", True, "decode", {"gfx": ANY}, "T-224 (edit +2.5% +-1.7)"),
    C("STRATA_VERIFY_ECAP", "env", "0", False, "decode", {"gfx": ANY}, "T-588, T-589, T-591", gate="PASS",
      note="the gate was run with the fork's grammar guard; v0.2.16 has none, so tool-call output is not covered",
      extra={"STRATA_VERIFY_ECAP_MIN_ROWS": "9"}),
    C("STRATA_SELECT_WMMA", "env", "1", False, "decode", {"gfx": GFX11_12}, "MOD_PLAN section 3 (lossy list)"),
    C("STRATA_PROMPT_ATTN_IMMA", "env", "1", False, "prompt", {"gfx": GFX11_12}, "Speedy opt-in, no HIP A/B"),
    C("STRATA_KV_HOST_DMA", "env", "1", True, "decode", {"gfx": ANY}, "MOD_PLAN section 3"),
    C("STRATA_PREFILL_RING", "env", "192", True, "prompt", {"gfx": ANY}, "MOD_PLAN section 3"),
    C("STRATA_STAGER_RING", "env", "64", True, "prompt", {"gfx": ANY}, "MOD_PLAN section 3"),
    C("GPU_MAX_HW_QUEUES", "env", "6", True, "decode", {"gfx": ANY}, "MOD_PLAN section 3"),
]


# --------------------------------------------------------------------------- hardware

def detect(root: str = "/", runner: Callable[[list[str]], str] | None = None) -> dict[str, Any] | None:
    hw = tuned_flags.detect_hardware(root, runner)
    if hw is None:
        return None
    base = Path(root)
    libs = []
    for pat in ("opt/rocm*/lib/libhipblaslt.so*", "opt/rocm*/core-*/lib/libhipblaslt.so*", "usr/lib*/libhipblaslt.so*"):
        libs += list(base.glob(pat))
    hw["hipblaslt"] = bool(libs)
    hw["wave"] = 32 if hw["gfx"].startswith(("gfx10", "gfx11", "gfx12")) else 64
    hw["kfd"] = (base / "dev/kfd").exists()
    hw["vmm"] = "unknown" if not hw["kfd"] else "kfd-present (the engine's own check decides at start)"
    return hw


def supported(c: dict[str, Any], hw: dict[str, Any]) -> tuple[bool, str]:
    n = c["needs"]
    gfx = n.get("gfx") or ()
    if gfx and not hw["gfx"].startswith(tuple(gfx)):
        return False, f"needs {'/'.join(gfx)}, this card is {hw['gfx']}"
    if n.get("wave") and hw.get("wave") != n["wave"]:
        return False, f"needs wave{n['wave']}, this card runs wave{hw.get('wave')}"
    if n.get("hipblaslt") and not hw.get("hipblaslt"):
        return False, "hipBLASLt library not found"
    if n.get("pcie_link") and not hw.get("pcie_link"):
        return False, "PCIe link not readable (an APU or a virtual card has none)"
    return True, ""


# --------------------------------------------------------------------------- measurement through abtest.py

def make_spec(config: str, flags: list[dict[str, Any]], cases: list[str]) -> dict[str, Any]:
    arms: dict[str, Any] = {"base": {}, "base-again": {}}
    for c in flags:
        key = {"env": "env", "arg": "args"}[c["kind"]]
        arms[c["name"]] = {key: {c["name"]: str(c["value"]).replace("<ROOT>", str(ROOT))}}
        if c.get("extra"):
            arms[c["name"]]["env"] = {**arms[c["name"]].get("env", {}), **c["extra"]}
    return {"base": config, "arms": arms, "cases": cases, "reps": 1, "lossy": True, "prefill": "long1k" in cases,
            "note": "flag_diag.py quick diagnostic: base twice (noise floor) and one arm per candidate"}


def parse_abtest(jsonl: str) -> dict[str, dict[str, Any]]:
    """{arm: {"decode": mean decode tok/s, "prompt": mean prompt tok/s, "texts": canonical text record, "ok": bool}}"""
    per: dict[str, dict[str, list]] = {}
    for line in jsonl.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        rec = json.loads(line)
        arm = rec.get("arm")
        if arm is None or not rec.get("rows"):
            continue
        d = per.setdefault(arm, {"decode": [], "prompt": [], "texts": [], "n": 0})
        d["n"] += 1
        for r in rec["rows"]:
            if r.get("decode_tps"):
                d["decode"].append(float(r["decode_tps"]))
            if r.get("prompt_tps"):
                d["prompt"].append(float(r["prompt_tps"]))
        d["texts"].append(json.dumps(rec.get("texts"), sort_keys=True))
    out = {}
    for arm, d in per.items():
        out[arm] = {"decode": statistics.fmean(d["decode"]) if d["decode"] else None,
                    "prompt": statistics.fmean(d["prompt"]) if d["prompt"] else None,
                    "texts": d["texts"][0] if d["texts"] else None, "ok": d["n"] > 0 and bool(d["decode"])}
    return out


def classify(c: dict[str, Any], arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """One verdict from the measured arms (pure: the tests feed it fake numbers)."""
    base, again, cand = arms.get("base"), arms.get("base-again"), arms.get(c["name"])
    v: dict[str, Any] = {"name": c["name"], "kind": c["kind"], "value": c["value"], "exact": c["exact"],
                         "ledger": c["ledger"], "metric": c["metric"]}
    if c.get("extra"):
        v["extra"] = c["extra"]
    if c.get("quality_gate"):
        v["quality_gate"] = c["quality_gate"]
    if cand is None or not cand.get("ok"):
        return {**v, "verdict": "UNSUPPORTED", "reason": "the engine did not start or answer with this flag"}
    if not base or not base.get("ok"):
        return {**v, "verdict": "UNTESTED", "reason": "no base measurement"}
    exact = bool(c["exact"])
    if c["exact"] and base.get("texts") != cand.get("texts"):
        if again and again.get("ok") and again.get("texts") is not None and again.get("texts") != base.get("texts"):
            v["exactness"] = "EXACT-UNVERIFIED (base vs base also differs)"
        else:
            exact = False
            v["exact"] = False
            v["exactness"] = ("TEXT-DIFFERS: claimed exact, the text differs from the default and a repeat of the default is "
                              + ("identical" if again and again.get("ok") else "not measured") + ": treated as lossy")
    m = c["metric"]
    b, k = base.get(m), cand.get(m)
    if not b or not k:
        return {**v, "verdict": "UNTESTED", "reason": f"no {m} tok/s measured"}
    gain = (k / b - 1.0) * 100.0
    noise = abs((again.get(m) / b - 1.0) * 100.0) if again and again.get(m) else 0.0
    bar = tuned_flags.EXACT_MIN_GAIN if exact else tuned_flags.LOSSY_MIN_GAIN
    v.update({"gain_pct": round(gain, 1), "noise_pct": round(noise, 1), "bar_pct": bar})
    if gain >= bar and gain > noise:
        v["verdict"] = "GAIN"
        if not exact and v.get("quality_gate") != "PASS":
            v["reason"] = "lossy: needs a passing quality gate before it is recommended"
    else:
        v["verdict"] = "NO-GAIN"
        v["reason"] = f"{gain:+.1f} % against a bar of {bar:g} % and a noise floor of {noise:.1f} %"
    return v


class Stopped(Exception):
    """The diagnostic got TERM (the chain's time limit): the arm running now is stopped, the rest is UNTESTED-TIMEOUT."""


def _row(c: dict[str, Any], verdict: str, reason: str) -> dict[str, Any]:
    return {"name": c["name"], "kind": c["kind"], "value": c["value"], "exact": c["exact"], "ledger": c["ledger"],
            "metric": c["metric"], "verdict": verdict, "reason": reason}


def ctest_not_run(text):
    """True when ctest ran nothing it could judge: the test binaries are not built ("Not Run") or no test matched.  Not a FAIL:
    the 2026-11-19 RD diagnostic called POOL_FUSE_PHASES FAIL from 7 pool tests that were all "Not Run" (no test build)."""
    return bool(text) and ("Not Run" in text or "No tests were found" in text)


def run_ctest(build_dir: str, flags: list[dict[str, Any]], out: Path, timeout_s: float = 300.0,
              runner: Callable[[list[str], Path, float], tuple] | None = None) -> dict[str, bool | None]:
    """The parity ctests of the flags claimed exact that name one: {name: True (pass) / False (fail) / None (timeout)}."""
    runner = runner or _run_harness
    res: dict[str, bool | None] = {}
    for c in flags:
        if not c.get("ctest") or not c["exact"]:
            continue
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", c["name"])
        log = out / f"ctest-{slug}.log"
        rc, to = runner(["ctest", "--test-dir", str(build_dir), "-R", c["ctest"], "--output-on-failure"], log, timeout_s)
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        res[c["name"]] = None if to or ctest_not_run(text) else rc == 0
    return res


def apply_ctest(v: dict[str, Any], ok: bool | None, c: dict[str, Any]) -> dict[str, Any]:
    """The parity ctest is the judge of exactness: FAIL is the verdict, PASS is recorded (and settles an unverified text check)."""
    if ok is False:
        return {**v, "verdict": "FAIL", "parity_ctest": "FAIL", "reason": f"the parity ctest ({c['ctest']}) of this flag claimed exact failed"}
    if ok is True:
        v = {**v, "parity_ctest": "PASS"}
        if str(v.get("exactness", "")).startswith("EXACT-UNVERIFIED"):
            v["exactness"] = "EXACT (parity ctest PASS; text identity not usable: base vs base also differs)"
    elif ok is None:
        v = {**v, "parity_ctest": "NOT-RUN"}
    return v


def run(config: str, out: Path, only: list[str] | None, budget_min: float, harness: Path,
        runner: Callable[[list[str], Path, float], tuple] | None = None, hw: dict[str, Any] | None = None,
        flag_timeout: float = 240.0, ctest: str | None = None, ctest_timeout: float = 300.0) -> dict[str, Any]:
    """Since 2026-11-19 (the chain's diag plain ran 18 min for 2 flags, then its timeout left abtest / server / engine running):
    ONE abtest.py screen per arm (base once, then each flag), case `short` only, no retries, quiet wait <= 30 s, each arm in its
    own process group with a hard limit (`flag_timeout` s; past it the group is stopped and the flag is UNTESTED-TIMEOUT:
    tools/pancake/diag_group.py).  OUT/verdicts.json is rewritten after every arm ("complete": false until the end), so a chain
    that stops the diagnostic still has the verdicts so far.  The parity check (text identical to the base) is judged only for
    flags claimed exact.  `budget_min` bounds the worst case: base + kept flags, each at most `flag_timeout`.
    runner(cmd, log_path, timeout_s) -> (exit code or None, timed_out)."""
    hw = hw or detect()
    if hw is None:
        raise SystemExit("flag_diag: no AMD graphics card could be read (rocminfo / sysfs)")
    out.mkdir(parents=True, exist_ok=True)
    runner = runner or _run_harness
    verdicts: list[dict[str, Any]] = []
    todo = []
    for c in CANDIDATES:
        if only and c["name"] not in only:
            continue
        ok, why = supported(c, hw)
        if not ok:
            verdicts.append(_row(c, "UNSUPPORTED", why))
        else:
            todo.append(c)
    t0 = time.time()
    per_flag_min = flag_timeout / 60.0   # worst case per arm: one engine start and the short case, cut at flag_timeout
    keep, dropped = [], []
    for c in todo:
        (keep if (len(keep) + 2) * per_flag_min <= budget_min else dropped).append(c)   # + 1 start of the default
    for c in dropped:
        verdicts.append(_row(c, "UNTESTED", f"budget of {budget_min:g} minutes ended"))
    slot = {c["name"]: len(verdicts) + i for i, c in enumerate(keep)}
    verdicts += [_row(c, "UNTESTED", "pending: not measured yet") for c in keep]

    def save(complete: bool) -> dict[str, Any]:
        result = {"schema": "strata.flag-diag/1", "hardware": hw, "minutes": round((time.time() - t0) / 60, 1),
                  "config": config, "complete": complete, "flag_timeout_s": flag_timeout, "verdicts": verdicts}
        tmp = out / "verdicts.json.tmp"
        tmp.write_text(json.dumps(result, indent=1), encoding="utf-8")
        os.replace(tmp, out / "verdicts.json")
        return result

    full = make_spec(config, keep, ["short"])

    def one(arm: str) -> tuple:
        spec = dict(full, arms={arm: full["arms"][arm]}, prefill=False,
                    note="flag_diag.py: one arm, one engine start, case short (quick screen)")
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", arm)
        sp, jl = out / f"spec-{slug}.json", out / f"abtest-{slug}.jsonl"
        sp.write_text(json.dumps(spec, indent=1), encoding="utf-8")
        if jl.exists():
            jl.unlink()
        cmd = [sys.executable, "-u", str(harness), str(sp), "--tier", "screen", "--retries", "0", "--max-wait", "30", "--out", str(jl)]
        t1 = time.time()
        rc, to = runner(cmd, out / f"abtest-{slug}.stdout", flag_timeout)
        print(f"arm {arm:28} exit={rc}{' TIMEOUT' if to else ''} {time.time() - t1:.0f} s", flush=True)
        return rc, to, parse_abtest(jl.read_text(encoding="utf-8") if jl.exists() else "")

    save(False)
    current: list[dict[str, Any]] = []
    try:
        if keep:
            current = list(keep)
            rc, to, arms = one("base")
            if not (arms.get("base") or {}).get("ok"):
                why = (f"no base result within {flag_timeout:g} s (engine start + short prompt)" if to else
                       f"the harness exited {rc} before any measurement" + (" (another test or engine is running)" if rc == 3 else ""))
                for c in keep:
                    verdicts[slot[c["name"]]] = _row(c, "UNTESTED-TIMEOUT" if to else "UNTESTED", why)
                return save(True)
            for i, c in enumerate(keep):
                current = keep[i:]
                rc, to, got = one(c["name"])
                cand = got.get(c["name"])
                if (c["exact"] and cand and cand.get("ok") and cand.get("texts") != arms["base"].get("texts")
                        and "base-again" not in arms):
                    # the text check needs to know whether the default differs from itself here (one more start, once)
                    _, _, again = one("base-again")
                    arms["base-again"] = again.get("base-again") or {"ok": False}
                if c["name"] not in got and to:
                    verdicts[slot[c["name"]]] = _row(c, "UNTESTED-TIMEOUT", f"no result within {flag_timeout:g} s per flag (engine start + "
                                                     "short prompt); its process group was stopped")
                elif c["name"] not in got and rc == 3:
                    verdicts[slot[c["name"]]] = _row(c, "UNTESTED", "the harness exited 3 (another test or engine is running)")
                else:
                    verdicts[slot[c["name"]]] = classify(c, {**arms, **got})
                save(False)
            current = []
        if ctest:
            for name, ok in run_ctest(ctest, keep, out, ctest_timeout, runner).items():
                verdicts[slot[name]] = apply_ctest(verdicts[slot[name]], ok, next(c for c in keep if c["name"] == name))
        return save(True)
    except (Stopped, KeyboardInterrupt):
        for c in current:
            if verdicts[slot[c["name"]]]["verdict"] == "UNTESTED":
                verdicts[slot[c["name"]]] = _row(c, "UNTESTED-TIMEOUT", "the diagnostic was stopped (its time limit) before this flag finished")
        save(True)
        raise


def _run_harness(cmd: list[str], log: Path, timeout_s: float) -> tuple:
    import diag_group
    rc, to, _ = diag_group.run(cmd, log, timeout_s, cwd=str(ROOT), say=lambda s: print(s, flush=True))
    return rc, to


# --------------------------------------------------------------------------- the recommended file

def recommended(result: dict[str, Any], *, source: str = "flag_diag.py") -> dict[str, Any]:
    """The profile for the measured hardware: only flags the entry rule accepts (tuned_flags.accepted)."""
    hw = result["hardware"]
    flags = []
    for v in result["verdicts"]:
        f = {k: v[k] for k in ("kind", "name", "value", "verdict", "exact", "gain_pct", "quality_gate", "ledger") if k in v}
        if tuned_flags.accepted(f):
            flags.append(f)
            for en, ev in (v.get("extra") or {}).items():   # the switches it needs in the same arm
                flags.append({**f, "kind": "env", "name": en, "value": ev})
    vram = float(hw.get("vram_gib") or 0)
    rocm = hw.get("rocm") or ""
    return {"match": {"gfx": hw["gfx"], "rocm": [".".join(rocm.split(".")[:2]) + "."] if rocm else [],
                      "vram_gib": [max(0, int(vram) - 2), int(vram) + 2]},
            "source": f"{source}, measured {time.strftime('%Y-%m-%d')}", "flags": flags}


def write_recommended(result: dict[str, Any], out: Path, merge: bool = False) -> dict[str, Any]:
    prof = recommended(result)
    doc: dict[str, Any] = {"schema": tuned_flags.SCHEMA, "profiles": []}
    if merge and out.exists():
        try:
            doc = json.loads(out.read_text(encoding="utf-8"))
        except ValueError:
            pass
    doc["profiles"] = [p for p in doc.get("profiles", []) if p.get("match") != prof["match"]] + [prof]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("hw")
    sub.add_parser("list")
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--only", default="")
    r.add_argument("--budget-min", type=float, default=15.0)
    r.add_argument("--harness", default=os.environ.get("FLAG_DIAG_ABTEST", str(HERE / "abtest.py")))
    r.add_argument("--flag-timeout", type=float, default=float(os.environ.get("FLAG_DIAG_FLAG_TIMEOUT", "240")),
                   help="seconds per arm (engine start + short case); past it the arm's process group is stopped: UNTESTED-TIMEOUT")
    r.add_argument("--ctest", default=os.environ.get("FLAG_DIAG_CTEST", ""),
                   help="a build tree: run the parity ctests of the flags claimed exact (the judge of exactness)")
    r.add_argument("--ctest-timeout", type=float, default=300.0)
    w = sub.add_parser("write-recommended")
    w.add_argument("--verdicts", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--merge", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "hw":
        print(json.dumps(detect(), indent=1))
    elif a.cmd == "list":
        for c in CANDIDATES:
            print(f"{c['name']:28} {c['kind']:3} {str(c['value'])[:30]:30} {'exact' if c['exact'] else 'lossy':5} {c['metric']:6} {c['ledger']}")
    elif a.cmd == "run":
        if not Path(a.harness).exists():
            print(f"flag_diag: the A/B harness {a.harness} is missing (set FLAG_DIAG_ABTEST or run from a port/rdna3-decode checkout)", file=sys.stderr)
            return 2
        def on_term(*_):
            raise Stopped()
        signal.signal(signal.SIGTERM, on_term)
        try:
            res = run(a.config, Path(a.out), [s for s in a.only.split(",") if s] or None, a.budget_min, Path(a.harness),
                      flag_timeout=a.flag_timeout, ctest=a.ctest or None, ctest_timeout=a.ctest_timeout)
        except (Stopped, KeyboardInterrupt):
            print(f"flag_diag: stopped; partial verdicts in {Path(a.out) / 'verdicts.json'}", flush=True)
            return 124
        for v in res["verdicts"]:
            print(f"{v['verdict']:11} {v['name']:28} {v.get('gain_pct', ''):>6} {v.get('reason', '')} {v.get('exactness', '')}".rstrip())
        print(f"{res['minutes']} minutes; verdicts in {Path(a.out) / 'verdicts.json'}")
    else:
        doc = write_recommended(json.loads(Path(a.verdicts).read_text(encoding="utf-8")), Path(a.out), a.merge)
        print(f"wrote {a.out}: {sum(len(p['flags']) for p in doc['profiles'])} flag(s) in {len(doc['profiles'])} profile(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
