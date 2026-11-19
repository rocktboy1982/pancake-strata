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
(c) verdict:  UNSUPPORTED (cannot run here, failed to start, or claimed exact and not identical)
              NO-GAIN     (runs, below the bar or inside the noise floor)
              GAIN        (exact and >= 2 % end to end, or lossy and >= 3 %)
              UNTESTED    (budget ended before this flag)
A lossy GAIN enters the recommended file only with a passing quality gate (`quality_gate: PASS`, from the ledger or a run of
tools/pancake/quality_gate.py); the quick screen cannot judge quality.  The file format and the rule are
terminal/strata_control/tuned_flags.py (`accepted`), which the normal start path applies.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "terminal"))
from strata_control import tuned_flags  # noqa: E402

GFX11 = ("gfx11",)
GFX11_12 = ("gfx11", "gfx12")
ANY = ()


def C(name, kind, value, exact, metric, needs, ledger, gate=None, note=""):
    return {"name": name, "kind": kind, "value": value, "exact": exact, "metric": metric, "needs": needs,
            "ledger": ledger, "quality_gate": gate, "note": note}


# The candidates: the adopted flags ported to this branch, plus the hardware-dependent ones.  `needs`: gfx prefixes
# (empty = any AMD card) and extra requirements ("hipblaslt": the library and a table for this architecture).
CANDIDATES: list[dict[str, Any]] = [
    C("STRATA_RDNA_Q6K", "env", "1", True, "decode", {"gfx": GFX11, "wave": 32}, "T-009, T-079"),
    C("STRATA_HIPBLASLT_TUNING", "env", "<ROOT>/terminal/tools/hip/gfx1100-hipblaslt-100401.txt", True, "prompt",
      {"gfx": ("gfx1100",), "hipblaslt": True}, "T-096"),
    C("--pcie-frac", "arg", "0", True, "decode", {"gfx": ANY, "pcie_link": True}, "T-197, T-268"),
    C("--adapt-every", "arg", "1", True, "decode", {"gfx": ANY}, "T-199"),
    C("--lookup-max-t", "arg", "12", False, "decode", {"gfx": ANY}, "T-126, T-127", gate="PASS"),
    C("STRATA_PF_F16", "env", "1", False, "prompt", {"gfx": GFX11}, "T-820, T-828", gate="PASS",
      note="rate and event rules of the gate PASS; the near-tie column FAILS (information)"),
    C("STRATA_HIP_WMMA", "env", "1", True, "decode", {"gfx": GFX11_12}, "T-072 (no-op in the preset)"),
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
    if c.get("quality_gate"):
        v["quality_gate"] = c["quality_gate"]
    if cand is None or not cand.get("ok"):
        return {**v, "verdict": "UNSUPPORTED", "reason": "the engine did not start or answer with this flag"}
    if not base or not base.get("ok"):
        return {**v, "verdict": "UNTESTED", "reason": "no base measurement"}
    if c["exact"] and base.get("texts") != cand.get("texts"):
        return {**v, "verdict": "UNSUPPORTED", "reason": "claimed exact but the text differs from the default here"}
    m = c["metric"]
    b, k = base.get(m), cand.get(m)
    if not b or not k:
        return {**v, "verdict": "UNTESTED", "reason": f"no {m} tok/s measured"}
    gain = (k / b - 1.0) * 100.0
    noise = abs((again.get(m) / b - 1.0) * 100.0) if again and again.get(m) else 0.0
    bar = tuned_flags.EXACT_MIN_GAIN if c["exact"] else tuned_flags.LOSSY_MIN_GAIN
    v.update({"gain_pct": round(gain, 1), "noise_pct": round(noise, 1), "bar_pct": bar})
    if gain >= bar and gain > noise:
        v["verdict"] = "GAIN"
        if not c["exact"] and v.get("quality_gate") != "PASS":
            v["reason"] = "lossy: needs a passing quality gate before it is recommended"
    else:
        v["verdict"] = "NO-GAIN"
        v["reason"] = f"{gain:+.1f} % against a bar of {bar:g} % and a noise floor of {noise:.1f} %"
    return v


def run(config: str, out: Path, only: list[str] | None, budget_min: float, harness: Path,
        runner: Callable[[list[str], Path], int] | None = None, hw: dict[str, Any] | None = None) -> dict[str, Any]:
    hw = hw or detect()
    if hw is None:
        raise SystemExit("flag_diag: no AMD graphics card could be read (rocminfo / sysfs)")
    out.mkdir(parents=True, exist_ok=True)
    verdicts: list[dict[str, Any]] = []
    todo = []
    for c in CANDIDATES:
        if only and c["name"] not in only:
            continue
        ok, why = supported(c, hw)
        if not ok:
            verdicts.append({"name": c["name"], "kind": c["kind"], "value": c["value"], "exact": c["exact"],
                             "ledger": c["ledger"], "metric": c["metric"], "verdict": "UNSUPPORTED", "reason": why})
        else:
            todo.append(c)
    t0 = time.time()
    per_flag_min = 3.0   # one engine start (1-3 minutes) and two short prompts
    keep, dropped = [], []
    for c in todo:
        (keep if (len(keep) + 2) * per_flag_min <= budget_min else dropped).append(c)   # 2 starts of the default
    for c in dropped:
        verdicts.append({"name": c["name"], "kind": c["kind"], "value": c["value"], "exact": c["exact"], "ledger": c["ledger"],
                         "metric": c["metric"], "verdict": "UNTESTED", "reason": f"budget of {budget_min:g} minutes ended"})
    arms: dict[str, Any] = {}
    if keep:
        cases = ["short", "edit"] + (["long1k"] if any(c["metric"] == "prompt" for c in keep) else [])
        spec_path, jsonl = out / "spec.json", out / "abtest.jsonl"
        spec_path.write_text(json.dumps(make_spec(config, keep, cases), indent=1), encoding="utf-8")
        cmd = [sys.executable, str(harness), str(spec_path), "--tier", "screen", "--out", str(jsonl)]
        rc = (runner or _run_harness)(cmd, out)
        text = jsonl.read_text(encoding="utf-8") if jsonl.exists() else ""
        arms = parse_abtest(text)
        if rc != 0 and not arms:
            for c in keep:
                verdicts.append({"name": c["name"], "kind": c["kind"], "value": c["value"], "exact": c["exact"],
                                 "ledger": c["ledger"], "metric": c["metric"], "verdict": "UNTESTED",
                                 "reason": f"the harness exited {rc} before any measurement"})
        else:
            verdicts += [classify(c, arms) for c in keep]
    result = {"schema": "strata.flag-diag/1", "hardware": hw, "minutes": round((time.time() - t0) / 60, 1),
              "config": config, "verdicts": verdicts}
    (out / "verdicts.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def _run_harness(cmd: list[str], out: Path) -> int:
    with open(out / "abtest.stdout", "w", encoding="utf-8") as fh:
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=str(ROOT)).returncode


# --------------------------------------------------------------------------- the recommended file

def recommended(result: dict[str, Any], *, source: str = "flag_diag.py") -> dict[str, Any]:
    """The profile for the measured hardware: only flags the entry rule accepts (tuned_flags.accepted)."""
    hw = result["hardware"]
    flags = []
    for v in result["verdicts"]:
        f = {k: v[k] for k in ("kind", "name", "value", "verdict", "exact", "gain_pct", "quality_gate", "ledger") if k in v}
        if tuned_flags.accepted(f):
            flags.append(f)
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
        res = run(a.config, Path(a.out), [s for s in a.only.split(",") if s] or None, a.budget_min, Path(a.harness))
        for v in res["verdicts"]:
            print(f"{v['verdict']:11} {v['name']:28} {v.get('gain_pct', ''):>6} {v.get('reason', '')}")
        print(f"{res['minutes']} minutes; verdicts in {Path(a.out) / 'verdicts.json'}")
    else:
        doc = write_recommended(json.loads(Path(a.verdicts).read_text(encoding="utf-8")), Path(a.out), a.merge)
        print(f"wrote {a.out}: {sum(len(p['flags']) for p in doc['profiles'])} flag(s) in {len(doc['profiles'])} profile(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
