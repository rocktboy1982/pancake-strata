"""Unit tests of the RIGOR step (T-1020, owner 2026-11-21): sweep.py rigor-select / rigor-verdict / turbo-build, the removal of the three owner-removed flags from the FINAL arms, the
abtest settle minimum and the queue script run-queue-us15.sh.  No box, no GPU.   python -m pytest -q tools/pancake/test_rigor.py"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
CHAINS = HERE / "box_chains"
os.environ.setdefault("SWEEP_CASES", "short,edit,longdec64k")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CHAINS))
import sweep as S  # noqa: E402
try:
    import abtest  # noqa: E402   (Linux only: it reads /proc at import)
except (AttributeError, ImportError):
    abtest = None

FAST, SLOW = 2180.0, 1530.0


def rec(arm, short=100.0, edit=150.0, dec=80.0, pf=FAST, t="2026-11-21 10:00:00", regime=None):
    r = {"t": t, "rep": 0, "arm": arm, "attempt": 0, "clean": True,
         "rows": [{"case": "short", "decode_tps": short, "prompt_tps": 150.0}, {"case": "short", "decode_tps": short, "prompt_tps": 150.0},
                  {"case": "edit", "decode_tps": edit, "prompt_tps": 150.0}, {"case": "edit", "decode_tps": edit, "prompt_tps": 150.0},
                  {"case": "longdec64k", "decode_tps": dec, "prompt_tps": pf}]}
    if regime is not None:
        r["regime"] = regime
    return json.dumps(r)


def block(arm, gain=1.0, arm_pf=FAST, base_pf=(FAST, FAST), base=100.0, mult=None):
    """warm, updated, ARM, base-late (a plus arm: gain = factor of the decode speed)."""
    return [rec("warmup"), rec("updated", short=base, edit=base * 1.5, dec=base * 0.8, pf=base_pf[0]),
            rec(arm, short=base * gain, edit=base * 1.5 * (mult or gain), dec=base * 0.8 * gain, pf=arm_pf),
            rec("base-late", short=base, edit=base * 1.5, dec=base * 0.8, pf=base_pf[1])]


def entry(flag, verdict, runs=None, run1=None, pass1=None, kind="plus", exact=True, slice_="usS1-1of5v"):
    return {"flag": flag, "arm": "plus-" + flag, "kind": kind, "exact": exact, "env": {flag: "1"}, "args": {}, "verdict": verdict, "runs": runs, "run1": run1,
            "pass1": pass1, "slice": slice_}


def eff_of(lines, arm, kind="plus"):
    return S.local_view(lines, arm, kind)[0]


def test_start_regime_uses_the_cold_64k_prefill_and_the_load_rate():
    assert S.start_regime(json.loads(rec("x", pf=FAST)))[0] is True
    assert S.start_regime(json.loads(rec("x", pf=SLOW)))[0] is False
    assert S.start_regime(json.loads(rec("x", pf=FAST)), load=0.3)[0] is False              # fast prefill, slow pack load: slow
    assert S.start_regime(json.loads(rec("x", regime={"slow": True, "prefill": FAST, "load": 1.0})))[0] is False   # the record's own verdict
    assert S.load_rates("STARTLOAD t loaded 33 GiB at 1.0 GiB/s rate=1.00\nSTARTLOAD t loaded 33 GiB at 0.2 GiB/s rate=0.20\n", 2) == [1.0, 0.2]
    assert S.load_rates("STARTLOAD t x rate=1.00\n", 2) is None


def test_trusted_needs_every_run_clean_and_agreeing():
    a, b = block("plus-F", 1.05), block("plus-F", 1.04)
    files = {"us-p1-usS1-1of5v-b01.jsonl": (a, None), "us-p1-usS2r-1-1of1-b01.jsonl": (b, None)}
    e = entry("F", "KEEP", runs=[eff_of(a, "plus-F"), eff_of(b, "plus-F")])
    views = S.gather_views(files, "plus-F", "plus")
    cl = S.classify_flag(e, views)
    assert cl["tier"] == "TRUSTED" and cl["clean_runs"] == 2 and not cl["keep_fails_strict"]


def test_a_slow_arm_or_slow_base_start_makes_the_flag_untrusted():
    a, b = block("plus-F", 1.05), block("plus-F", 1.04, arm_pf=SLOW)
    files = {"x1.jsonl": (a, None), "x2.jsonl": (b, None)}
    e = entry("F", "KEEP", runs=[eff_of(a, "plus-F"), eff_of(b, "plus-F")])
    cl = S.classify_flag(e, S.gather_views(files, "plus-F", "plus"))
    assert cl["tier"] == "UNTRUSTED" and any("arm start SLOW" in r for r in cl["reasons"])
    c = block("plus-F", 1.04, base_pf=(FAST, SLOW))
    files = {"x1.jsonl": (a, None), "x2.jsonl": (c, None)}
    e = entry("F", "KEEP", runs=[eff_of(a, "plus-F"), eff_of(c, "plus-F")])
    cl = S.classify_flag(e, S.gather_views(files, "plus-F", "plus"))
    assert cl["tier"] == "UNTRUSTED" and any("base start SLOW" in r for r in cl["reasons"])


def test_runs_that_disagree_by_more_than_ten_points_with_opposite_sign_are_untrusted():
    a, b = block("plus-F", 1.0), block("plus-F", 1.0)
    files = {"x1.jsonl": (a, None), "x2.jsonl": (b, None)}
    e = entry("F", "KEEP", runs=[{"short": 23.2, "edit": 3.0}, {"short": -5.2, "edit": 3.0}])
    e["runs"] = [eff_of(a, "plus-F"), eff_of(b, "plus-F")]
    e["runs"][0] = {**e["runs"][0], "short": 23.2}
    e["runs"][1] = {**e["runs"][1], "short": -5.2}
    views = [{"eff": e["runs"][0], "clean": True, "why": ""}, {"eff": e["runs"][1], "clean": True, "why": ""}]
    cl = S.classify_flag(e, views)
    assert cl["tier"] == "UNTRUSTED" and any(r.startswith("DISAGREE") for r in cl["reasons"])


def test_one_clean_run_only_is_untrusted_and_a_verdict_without_rows_is_not_found():
    a = block("plus-F", 1.0)
    e = entry("F", "NO-GAIN-P1", pass1=eff_of(a, "plus-F"))
    cl = S.classify_flag(e, S.gather_views({"x1.jsonl": (a, None)}, "plus-F", "plus"))
    assert cl["tier"] == "UNTRUSTED" and any(r.startswith("ONE-RUN") for r in cl["reasons"])
    cl = S.classify_flag(entry("G", "INCONCLUSIVE"), [])
    assert cl["tier"] == "UNTRUSTED" and any(r.startswith("NO-RUN") for r in cl["reasons"])


def test_keep_that_fails_the_strict_rule_is_flagged():
    a, b = block("plus-F", 1.0), block("plus-F", 1.0)
    views = S.gather_views({"x1.jsonl": (a, None), "x2.jsonl": (b, None)}, "plus-F", "plus")
    views[0]["eff"], views[1]["eff"] = {"short": 3.0, "edit": 0.0, "longdec64k": 0.0}, {"short": 1.0, "edit": 5.0, "longdec64k": 0.0}
    e = entry("F", "KEEP", runs=[views[0]["eff"], views[1]["eff"]])
    assert S.classify_flag(e, views)["keep_fails_strict"] is True      # +3 on short in run 1, +5 on edit in run 2: no case >= +2 in BOTH
    e["runs"] = [{"short": 3.0, "edit": 0.0, "longdec64k": 0.0}, {"short": 2.5, "edit": 5.0, "longdec64k": 0.0}]
    views[0]["eff"], views[1]["eff"] = e["runs"]
    assert S.classify_flag(e, views)["keep_fails_strict"] is False


def make_triage(n_untrusted=6):
    tri = {"schema": "pancake.sweep-triage/2", "branch": "us", "protected": {}, "flags": {}}
    files = {}
    ok = block("plus-OK", 1.03)
    files["us-p1-usS1-2of5-b01.jsonl"] = (ok, None)
    ok2 = block("plus-OK", 1.03)
    files["us-p1-usS2r-2-1of1-b01.jsonl"] = (ok2, None)
    tri["flags"]["OK"] = entry("OK", "KEEP", runs=[eff_of(ok, "plus-OK"), eff_of(ok2, "plus-OK")])
    tri["flags"]["OK"]["runs"][0]["short"] = tri["flags"]["OK"]["runs"][1]["short"] = 3.0              # strict rule passed
    for i in range(n_untrusted):
        nm = "U%d" % i
        b = block("plus-" + nm, 1.0 + 0.01 * i, arm_pf=SLOW)
        files["us-p1-usS1-3of5-b0%d.jsonl" % i] = (b, None)
        tri["flags"][nm] = entry(nm, "PROMISING-LOW", pass1=eff_of(b, "plus-" + nm))
    return tri, files


def test_rigor_select_takes_the_untrusted_and_leaves_the_trusted():
    tri, files = make_triage()
    sel, back, table = S.rigor_select(tri, files, budget_s=10 ** 9)
    names = [r["flag"] for r in sel]
    assert "OK" not in names and len(names) == 6 and not back
    assert {r["flag"]: r["tier"] for r in table}["OK"] == "TRUSTED"
    assert [r["flag"] for r in sel] == ["U5", "U4", "U3", "U2", "U1", "U0"]        # same class: the higher gain first


def test_forced_and_added_flags_are_always_candidates_and_first():
    tri, files = make_triage()
    sel, back, table = S.rigor_select(tri, files, forced=["OK"], added=["U0"], budget_s=10 ** 9)
    assert [r["flag"] for r in sel][:2] == ["U0", "OK"] or [r["flag"] for r in sel][:2] == ["OK", "U0"]
    assert {r["flag"]: r["label"] for r in sel}["U0"] == "owner-added" and {r["flag"]: r["label"] for r in sel}["OK"] == "owner-forced"
    assert all(r["rank"] > 0 for r in sel if r["flag"] not in ("OK", "U0"))


def test_the_budget_cuts_the_list_and_the_rest_is_kept():
    tri, files = make_triage(12)
    n = S.rigor_fit(S.RIGOR_BUDGET_S)
    assert 7 <= n <= 11                                                          # about 6 h: 14 valid starts per pass for 9 arms
    sel, back, _ = S.rigor_select(tri, files, budget_s=S.RIGOR_BUDGET_S)
    assert len(sel) == n and len(sel) + len(back) == 12
    assert S.rigor_estimate(len(sel))[0] <= S.RIGOR_BUDGET_S < S.rigor_estimate(len(sel) + 1)[0]


def test_owner_added_flags_enter_even_when_trusted():
    tri, files = make_triage(0)
    sel, back, _ = S.rigor_select(tri, files, added=S.OWNER_ADDED + ("OK",), budget_s=10 ** 9)
    assert [r["flag"] for r in sel] == ["OK"]                                    # the named ones that exist in the triage


def pairs(*effs, clean=True):
    return [{"eff": e, "clean": clean, "why": "" if clean else "arm start SLOW"} for e in effs]


def test_rigor_verdict_keep_add_and_lossy_gate():
    e = {"kind": "plus", "exact": True}
    v = S.rigor_verdict(e, pairs({"short": 3.0, "edit": 0.5, "longdec64k": 0.0}, {"short": 2.5, "edit": 0.0, "longdec64k": -1.0}, {"short": 2.2, "edit": 0.2, "longdec64k": 0.5}))
    assert v[0] == "KEEP" and "ADD" in v[1]
    assert S.rigor_verdict({"kind": "minus", "exact": True}, pairs(*[{"short": 3.0, "edit": 0.5}] * 3))[1].find("KEEP") > 0
    v = S.rigor_verdict({"kind": "plus", "exact": False}, pairs(*[{"short": 4.0, "edit": 0.5}] * 3))
    assert v[0] == "RETEST-GATE"


def test_rigor_verdict_optional_mixed_drop_inconclusive():
    e = {"kind": "plus", "exact": True}
    assert S.rigor_verdict(e, pairs(*[{"short": 1.0, "edit": 25.0, "longdec64k": -8.0}] * 3))[0] == "OPTIONAL"
    assert S.rigor_verdict(e, pairs({"short": 3.0, "edit": -4.0}, {"short": 3.0, "edit": 0.0}, {"short": 3.0, "edit": 0.0}))[0] == "MIXED"
    assert S.rigor_verdict(e, pairs(*[{"short": 0.2, "edit": -0.5, "longdec64k": 0.1}] * 3))[0] == "DROP"
    assert S.rigor_verdict(e, pairs(*[{"short": -4.0, "edit": 0.0, "longdec64k": 0.0}] * 3))[0] == "DROP"       # a repeated loss
    assert S.rigor_verdict(e, pairs({"short": 1.5, "edit": 0.0}, {"short": 0.0, "edit": 0.0}, {"short": 1.0, "edit": 0.0}))[0] == "INCONCLUSIVE"
    v = S.rigor_verdict(e, pairs({"short": 9.0}, {"short": 9.0}) + pairs({"short": 9.0}, clean=False))
    assert v[0] == "INCONCLUSIVE" and "2 clean pair" in v[1]                      # < 3 clean pairs: never decided, the invalid pair is reported


def test_rigor_triage_counts_only_clean_pairs_of_the_rigor_rows():
    files = {}
    for k, pf in ((1, FAST), (2, FAST), (3, FAST), (4, SLOW)):
        files["us-p1-usSR-%d-b01.jsonl" % k] = (block("plus-F", 1.04, arm_pf=pf), None)
    files["us-p1-usS1-1of5v-b01.jsonl"] = (block("plus-F", 0.5), None)           # a screen row: ignored by the RIGOR verdict
    meta = {"flags": {"plus-F": entry("F", "KEEP")}}
    doc = S.rigor_triage(meta, files)
    r = doc["flags"]["F"]
    assert r["n_pairs"] == 4 and r["n_clean"] == 3 and r["verdict"] == "KEEP" and r["pass2_clean"] is True and r["slice"].startswith("usS2r")
    assert S.rigor_topup(meta, files) == {}
    del files["us-p1-usSR-3-b01.jsonl"]
    assert "plus-F" in S.rigor_topup(meta, files)


def test_overlay_and_the_final_arms_drop_the_three_removed_flags():
    s220 = {"base": {}, "arms": {"plain-upstream": {"env": {}, "args": {}},
                                "updated": {"env": {"STRATA_PIN_BY_NEED": "1", "STRATA_KEEPME": "1"}, "args": {"--lookup-max-t": "12", "--adapt-every": "1", "--x": "2"}}}}
    s222 = {"arms": {"updated-exact": {"env": {"STRATA_PIN_BY_NEED": "1"}, "args": {"--lookup-max-t": "12"}}}}
    tri = {"schema": "pancake.sweep-triage/2", "branch": "us", "protected": {}, "flags": {}}
    rigor = {"flags": {"NEWF": {"flag": "NEWF", "kind": "plus", "exact": True, "env": {"STRATA_NEWF": "1"}, "args": {}, "verdict": "KEEP", "pass2": {"short": 3.0}, "pass2_clean": True, "slice": "usS2rR"}}}
    arms, changes, _ = S.final_default(tri, s220, s222, rigor)
    assert arms["updated"]["env"] == {"STRATA_KEEPME": "1", "STRATA_NEWF": "1"} and arms["updated"]["args"] == {"--x": "2"}
    assert "--lookup-max-t" not in arms["updated-exact"]["args"] and "STRATA_PIN_BY_NEED" not in arms["updated-exact"]["env"]
    assert any("removed --lookup-max-t" in c for c in changes)


def test_turbo_build_default_plus_rigorous_positive_flags_with_gates():
    tri = {"schema": "pancake.sweep-triage/2", "branch": "us", "protected": {}, "flags": {
        "--lookup-max-t": {"flag": "--lookup-max-t", "arm": "minus-lookup-max-t", "kind": "minus", "exact": True, "env": {}, "args": {"--lookup-max-t": "12"}, "verdict": "KEEP"},
        "STRATA_HC_UPMIX": {"flag": "STRATA_HC_UPMIX", "arm": "minus-HC_UPMIX", "kind": "minus", "exact": False, "env": {"STRATA_HC_UPMIX": "1"}, "args": {}, "verdict": "RETEST-GATE"},
        "STRATA_PF_HCDOWN": {"flag": "STRATA_PF_HCDOWN", "arm": "minus-PF_HCDOWN", "kind": "minus", "exact": False, "env": {"STRATA_PF_HCDOWN": "1"}, "args": {}, "verdict": "RETEST-GATE"},
        "STRATA_TRUSTED_OPT": {"flag": "STRATA_TRUSTED_OPT", "arm": "plus-T", "kind": "plus", "exact": True, "env": {"STRATA_TRUSTED_OPT": "1"}, "args": {}, "verdict": "OPTIONAL"},
        "STRATA_UNTRUSTED_OPT": {"flag": "STRATA_UNTRUSTED_OPT", "arm": "plus-U", "kind": "plus", "exact": True, "env": {"STRATA_UNTRUSTED_OPT": "1"}, "args": {}, "verdict": "OPTIONAL"}}}
    rigor = {"flags": {"--lookup-max-t": {**tri["flags"]["--lookup-max-t"], "verdict": "OPTIONAL", "pass2": {"edit": 25.0}, "reason": "RIGOR"},
                       "STRATA_HC_UPMIX": {**tri["flags"]["STRATA_HC_UPMIX"], "verdict": "RETEST-GATE", "pass2": {"short": 9.0}, "reason": "RIGOR"},
                       "STRATA_PF_HCDOWN": {**tri["flags"]["STRATA_PF_HCDOWN"], "verdict": "RETEST-GATE", "pass2": {"short": 9.0}, "reason": "RIGOR"}}}
    default = {"env": {"STRATA_BASE": "1"}, "args": {}}
    doc = S.turbo_build(tri, rigor, {"STRATA_HC_UPMIX": "PASS", "STRATA_PF_HCDOWN": "REJECT"}, default, trusted={"STRATA_TRUSTED_OPT"})
    got = {t["flag"] for t in doc["flags"]}
    assert got == {"--lookup-max-t", "STRATA_HC_UPMIX", "STRATA_TRUSTED_OPT"}                 # the untrusted screen verdict and the REJECTED lossy flag stay out
    assert doc["arm"]["args"] == {"--lookup-max-t": "12"} and doc["arm"]["env"]["STRATA_HC_UPMIX"] == "1" and "STRATA_PF_HCDOWN" not in doc["arm"]["env"]
    assert [t["flag"] for t in doc["excluded"]] == ["STRATA_PF_HCDOWN"]
    assert default == {"env": {"STRATA_BASE": "1"}, "args": {}}                               # the input is not changed


def test_gate_results_map_the_arm_names_to_flags(tmp_path):
    d = tmp_path / "g"
    d.mkdir()
    (d / "verdicts.json").write_text(json.dumps({"minus-HC_UPMIX": "PASS", "plus-OTHER": "REJECT"}))
    tri = {"flags": {"STRATA_HC_UPMIX": {"arm": "minus-HC_UPMIX"}}}
    assert S.gate_results([str(d)], tri) == {"STRATA_HC_UPMIX": "PASS"}


def test_final_specs_race_has_the_turbo_arm_and_a_longer_estimate():
    s220 = {"base": {}, "arms": {"plain-upstream": {"env": {}, "args": {}}, "updated": {"env": {"STRATA_PIN_BY_NEED": "1"}, "args": {"--adapt-every": "1"}}}}
    s222 = {"arms": {"updated-exact": {"env": {}, "args": {}}}}
    tri = {"schema": "pancake.sweep-triage/2", "branch": "us", "protected": {}, "flags": {}}
    specs, lines = S.final_specs(tri, s220, s222, turbo={"arm": {"env": {"T": "1"}, "args": {}}, "flags": [], "excluded": []})
    assert list(specs["final-short"]["arms"]) == ["plain-upstream", "updated", "updated-exact", "updated-turbo"]
    assert list(specs["final-gate"]["arms"])[0] == "plain-upstream" and "updated-turbo" in specs["final-gate"]["arms"]
    assert "STRATA_PIN_BY_NEED" not in specs["final-short"]["arms"]["updated"]["env"]
    three, _ = S.final_specs(tri, s220, s222)
    assert S.final_estimate(specs)["final-long"] > S.final_estimate(three)["final-long"]


def test_block_specs_mix_and_reverse_make_other_neighbours_and_the_reverse_order():
    spec = {"arms": {"updated": {}, **{"a%d" % i: {"env": {"A": str(i)}} for i in range(8)}}, "note": ""}
    fw = S.block_specs(spec, 12, "s1", "first", 4, mix=True)
    rv = S.block_specs(spec, 12, "s1", "first", 4, mix=True, reverse=True)
    other = S.block_specs(spec, 12, "s2", "first", 4, mix=True)
    assert fw[0][2] == list(reversed(rv[0][2])) and fw[0][2] != other[0][2]
    assert [a for a in fw[0][1]["arms"] if a.startswith("base-m")] == ["base-m1"]


@pytest.mark.skipif(abtest is None, reason="abtest.py needs Linux")
def test_abtest_settle_min_applies_to_every_start():
    waited = []
    say = []
    abtest.DIAG["run_s"] = None
    assert abtest.settle_after_long(say.append, waited.append, {"ABTEST_SETTLE_MIN_S": "120"}) == 120 and waited == [120.0] and "NA s run" in say[0]
    abtest.DIAG["run_s"] = 50.0
    assert abtest.settle_after_long(say.append, waited.append, {"ABTEST_SETTLE_MIN_S": "120", "ABTEST_SETTLE_AFTER_LONG_S": "60"}) == 120
    abtest.DIAG["run_s"] = 200.0
    assert abtest.settle_after_long(say.append, waited.append, {"ABTEST_SETTLE_MIN_S": "30", "ABTEST_SETTLE_AFTER_LONG_S": "120"}) == 120
    assert abtest.settle_after_long(say.append, waited.append, {}) == 0
    abtest.DIAG["run_s"] = None


@pytest.mark.skipif(not shutil.which("sh"), reason="needs a POSIX sh")
def test_run_queue_us15_is_us14_with_rigor_and_turbo():
    t = (CHAINS / "run-queue-us15.sh").read_text(encoding="utf-8")
    assert subprocess.run([shutil.which("sh"), "-n", str(CHAINS / "run-queue-us15.sh")]).returncode == 0 and "\r" not in t
    body = t[t.index("# ---- us15 (T-1020)"):]
    order = ["leftover\n", "rigor\n", "gates\n", "turbo\n", "final final"]
    pos = [body.index("\n" + x) for x in order]
    assert pos == sorted(pos)                                                     # LEFTOVER, RIGOR, gates, TURBO, FINAL
    assert body.index("rescreen usS2r-5") < body.index("\nleftover\n") and "rerun1 5" not in body and "screen 1/1 usS1-5of5" not in body.replace("usS1-5of5x", "")
    for must in ("bench-sweep.v8.sh", "bench-retest.v5.sh", "bench-final-us.v3.sh", "ABTEST_SETTLE_MIN_S=120", "SWEEP_REGIME_ALL=1", "SWEEP_NO_TRIAGE=1", "us-triage-rigor.json", "ABTEST_PREREAD:-2",
                 "ABTEST_SETTLE_AFTER_LONG_S:-120", "FINAL_TURBO"):
        assert must in t
    for old in ("bench-sweep.v7.sh", "bench-retest.v4.sh", "bench-final-us.v2.sh"):
        assert old not in t.split("# US-ONLY queue, FOURTEENTH")[0] + body
    for script in ("bench-sweep.sh", "bench-retest.sh", "bench-final-us.sh"):
        assert subprocess.run([shutil.which("sh"), "-n", str(CHAINS / script)]).returncode == 0
    assert "SWEEP_NO_TRIAGE" in (CHAINS / "bench-sweep.sh").read_text(encoding="utf-8") and "us-triage-rigor.json" in (CHAINS / "bench-retest.sh").read_text(encoding="utf-8")


@pytest.mark.skipif(not shutil.which("sh"), reason="needs a POSIX sh")
def test_run_queue_us16_is_us15_with_only_the_new_pin_ready_cap_and_names():
    a = (CHAINS / "run-queue-us15.sh").read_text(encoding="utf-8")
    p = CHAINS / "run-queue-us16.sh"
    b = p.read_text(encoding="utf-8")
    assert subprocess.run([shutil.which("sh"), "-n", str(p)]).returncode == 0 and "\r" not in b
    a_lines, b_lines = a.split("\n"), b.split("\n")
    import difflib
    changed = [(x[0], x[2:]) for x in difflib.ndiff(a_lines, b_lines) if x[0] in "+-"]
    kept = [x for x in changed if not x[1].lstrip().startswith("#")]    # comment lines (the new header) aside, only these differ:
    assert sorted(kept) == sorted([("-", "L=/home/dan/clone-of-d/src/run-queue-us15.log"), ("+", "L=/home/dan/clone-of-d/src/run-queue-us16.log"),
                                   ("-", "  hp=$O/work/us15-harness.tgz"), ("+", "  hp=$O/work/us16-harness.tgz"),
                                   ("-", 'say "QUEUE_US15_DONE $(date +%F_%T)"'), ("+", 'say "QUEUE_US16_DONE $(date +%F_%T)"'),
                                   ("-", next(x for x in a_lines if x.startswith('say "QUEUE_US15 START'))),
                                   ("+", next(x for x in b_lines if x.startswith('say "QUEUE_US16 START'))),
                                   ("+", next(x for x in b_lines if x.startswith("export ABTEST_READY_MAX_S=${ABTEST_READY_MAX_S:-900}")))])
    assert next(x for x in a_lines if x.startswith('say "QUEUE_US15 START')).replace("US15", "US16") == next(x for x in b_lines if x.startswith('say "QUEUE_US16 START'))
    body = b[b.index("# ---- us15 (T-1020)"):]
    order = ["leftover\n", "rigor\n", "gates\n", "turbo\n", "final final"]
    pos = [body.index("\n" + x) for x in order]
    assert pos == sorted(pos) and body.index("rescreen usS2r-5") < body.index("\nleftover\n") and "SWEEP_HARNESS_SHA:?" in b   # the pin comes from the caller's environment
