"""Tests of apply_triage.py with fake triage files and copies of the shipped-file shapes.  Run: python -m pytest -q tools/pancake/test_apply_triage.py"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import apply_triage as at  # noqa: E402

PROT = {"STRATA_VERIFY_ECAP": ["edit"], "STRATA_RDNA_Q6K": ["short", "edit", "longdec32k"], "STRATA_GR_RDNA": ["short", "edit", "longdec32k"],
        "STRATA_RDNA_MMVQ_WAVE": ["short", "edit", "longdec32k"]}


def rec(verdict, kind="minus", env=None, args=None, p1=None, p2=None):
    return {"verdict": verdict, "kind": kind, "env": env or {}, "args": args or {}, "pass1": p1 or {"short": 0.1}, "pass2": p2, "reason": "r"}


def tri(branch, flags):
    return {"branch": branch, "protected": PROT, "flags": flags}


US_SHIPPED = {"schema": "strata.tuned-flags/1", "profiles": [{"match": {"gfx": "gfx1100"}, "flags": [
    {"kind": "env", "name": "STRATA_RDNA_Q6K", "value": "1", "verdict": "GAIN", "exact": True, "gain_pct": 4.3},
    {"kind": "env", "name": "STRATA_GR_RDNA", "value": "1", "verdict": "GAIN", "exact": True, "gain_pct": 3.0},
    {"kind": "arg", "name": "--pcie-frac", "value": "0", "verdict": "GAIN", "exact": True, "gain_pct": 6.0}]}], "removed": []}
US_PRESET = {"args": ["--pack", "p", "--pcie-frac", "0", "--lookup-max-t", "12"], "env": {"STRATA_RDNA_Q6K": "1", "STRATA_KV_PREFETCH": "1", "STRATA_GR_RDNA": "1"}}
RD_SHIPPED = {"schema": 1, "env": {"STRATA_PF_F16": "1", "STRATA_VERIFY_ECAP": "0", "STRATA_VERIFY_ECAP_MIN_ROWS": "9", "STRATA_EXPDEC": "1",
                                   "STRATA_EXPDEC_DOWN": "1"},
              "args": {"--lookup-max-t": "12", "--grammar": "on"},
              "verdicts": {"STRATA_VERIFY_ECAP=0,MIN_ROWS=9": {}, "--lookup-max-t 12": {}},
              "preset": {"STRATA_EXPDEC": "x", "STRATA_EXPDEC_DOWN": "y", "--grammar on": "z"}, "theirs_kept": {"STRATA_HIPBLASLT_TUNING": "100200"},
              "removed": {}}
SPEC = {"arms": {"plain": {"exe": "@PLAIN@/strata"}, "updated": {"env": {"STRATA_EXPDEC": "1", "STRATA_KV_PREFETCH": "1"}, "args": {"--lookup-max-t": "12"}},
                 "updated-exact": {"env": {"STRATA_EXPDEC": "1"}, "args": {}}}}


def write(tmp, name, doc):
    p = tmp / name
    p.write_text(json.dumps(doc))
    return p


def run(tmp, t, branch, shipped, preset=None, spec=None, dry=False):
    tp = write(tmp, "t.json", t)
    argv = ["--triage", str(tp), "--branch", branch, "--shipped", str(shipped)]
    if preset:
        argv += ["--preset", str(preset)]
    if spec:
        argv += ["--spec", str(spec)]
    if dry:
        argv.append("--dry-run")
    assert at.main(argv) == 0


def test_us_drop_removes_with_evidence_keep_plus_adds_and_the_kernels_are_protected(tmp_path):
    sh, pr = write(tmp_path, "s.json", US_SHIPPED), write(tmp_path, "p.json", US_PRESET)
    t = tri("us", {
        "--pcie-frac": rec("DROP", args={"--pcie-frac": "0"}, p1={"short": -0.5}),
        "STRATA_KV_PREFETCH": rec("DROP", env={"STRATA_KV_PREFETCH": "1"}),
        "STRATA_RDNA_Q6K": rec("DROP", env={"STRATA_RDNA_Q6K": "1"}),                                          # pass-1 low performer, no pass 2: protected
        "STRATA_GR_RDNA": rec("DROP", env={"STRATA_GR_RDNA": "1"}, p2={"short": -2.0, "edit": -1.0, "longdec32k": 0.5}),   # negative on decode: dropped
        "STRATA_AMD_COMMUNITY_TUNING": rec("KEEP", kind="plus", env={"STRATA_AMD_COMMUNITY_TUNING": "0"}, p2={"short": 2.5}),
        "--lookup-max-t": rec("RETEST-GATE", args={"--lookup-max-t": "12"}),
    })
    run(tmp_path, t, "us", sh, pr)
    s, p = json.loads(sh.read_text()), json.loads(pr.read_text())
    names = [f["name"] for f in s["profiles"][0]["flags"]]
    assert names == ["STRATA_RDNA_Q6K", "STRATA_AMD_COMMUNITY_TUNING"]
    assert {r["name"] for r in s["removed"]} == {"--pcie-frac", "STRATA_GR_RDNA"} and all("bench-sweep.sh triage DROP" in r["evidence"] for r in s["removed"])
    assert p["args"] == ["--pack", "p", "--lookup-max-t", "12"]                               # RETEST-GATE: untouched
    assert p["env"] == {"STRATA_RDNA_Q6K": "1", "STRATA_AMD_COMMUNITY_TUNING": "0"}


def test_rd_drop_prunes_verdicts_and_preset_exactly_ecap_protected_unless_edit_negative_theirs_untouched(tmp_path):
    sh, sp = write(tmp_path, "s.json", RD_SHIPPED), write(tmp_path, "sp.json", SPEC)
    t = tri("rd", {
        "STRATA_EXPDEC": rec("DROP", env={"STRATA_EXPDEC": "1"}),
        "STRATA_VERIFY_ECAP": rec("DROP", env={"STRATA_VERIFY_ECAP": "0", "STRATA_VERIFY_ECAP_MIN_ROWS": "9"}, p2={"short": -4.0, "edit": 3.0}),
        "STRATA_HIPBLASLT_TUNING": rec("DROP", kind="plus", env={"STRATA_HIPBLASLT_TUNING": "x"}),          # a value of theirs: never written
        "--grammar": rec("UNTESTED", args={"--grammar": "on"}),
    })
    run(tmp_path, t, "rd", sh, spec=sp)
    s, spec = json.loads(sh.read_text()), json.loads(sp.read_text())
    assert "STRATA_EXPDEC" not in s["env"] and s["env"]["STRATA_EXPDEC_DOWN"] == "1"            # EXPDEC_DOWN is not EXPDEC
    assert "STRATA_EXPDEC" not in s["preset"] and "STRATA_EXPDEC_DOWN" in s["preset"] and "--grammar on" in s["preset"]
    assert s["env"]["STRATA_VERIFY_ECAP"] == "0"                                                  # edit +3 %: protected
    assert set(s["removed"]) == {"STRATA_EXPDEC"} and s["theirs_kept"] == RD_SHIPPED["theirs_kept"]
    assert "STRATA_EXPDEC" not in spec["arms"]["updated"]["env"] and "STRATA_EXPDEC" not in spec["arms"]["updated-exact"]["env"]
    assert spec["arms"]["plain"] == {"exe": "@PLAIN@/strata"}
    # ECAP negative on edit in pass 2: dropped, its verdict entry too
    sh2 = write(tmp_path, "s2.json", RD_SHIPPED)
    run(tmp_path, tri("rd", {"STRATA_VERIFY_ECAP": rec("DROP", env={"STRATA_VERIFY_ECAP": "0", "STRATA_VERIFY_ECAP_MIN_ROWS": "9"},
                                                         p2={"short": 4.0, "edit": -1.0})}), "rd", sh2)
    s2 = json.loads(sh2.read_text())
    assert "STRATA_VERIFY_ECAP" not in s2["env"] and "STRATA_VERIFY_ECAP_MIN_ROWS" not in s2["env"] and not s2["verdicts"].get("STRATA_VERIFY_ECAP=0,MIN_ROWS=9")


def test_rd_keep_plus_is_added_with_a_verdict_and_dry_run_writes_nothing(tmp_path):
    sh = write(tmp_path, "s.json", RD_SHIPPED)
    t = tri("rd", {"STRATA_POOL_FUSE_PHASES": rec("KEEP", kind="plus", env={"STRATA_POOL_FUSE_PHASES": "1"}, p2={"short": 2.4, "edit": 0.5})})
    run(tmp_path, t, "rd", sh, dry=True)
    assert json.loads(sh.read_text()) == RD_SHIPPED
    run(tmp_path, t, "rd", sh)
    s = json.loads(sh.read_text())
    assert s["env"]["STRATA_POOL_FUSE_PHASES"] == "1" and s["verdicts"]["STRATA_POOL_FUSE_PHASES"]["gain_pct"] == 2.4


def test_protected_rule():
    assert at.protected_stays("STRATA_VERIFY_ECAP", {"pass2": None}, PROT)
    assert at.protected_stays("STRATA_VERIFY_ECAP", {"pass2": {"edit": 0.0}}, PROT)
    assert not at.protected_stays("STRATA_VERIFY_ECAP", {"pass2": {"edit": -0.1}}, PROT)
    assert not at.protected_stays("STRATA_KV_PREFETCH", {"pass2": None}, PROT)
    assert at.protected_stays("STRATA_RDNA_MMVQ_WAVE", {"pass2": {"short": -1.0, "edit": 2.0, "longdec32k": 0.0}}, PROT)   # mean of decode cases >= 0


def test_branch_mismatch_stops(tmp_path):
    import pytest
    sh = write(tmp_path, "s.json", US_SHIPPED)
    with pytest.raises(SystemExit):
        run(tmp_path, tri("rd", {}), "us", sh)
