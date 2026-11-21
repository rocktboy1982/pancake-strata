"""T-1024 (owner 2026-11-21): block limits + continuation (the real numbers of b02 / b03 of usS1-L1of5x), the FINAL ROUND selection and verdict, the developed-flag gate selection, the us17 order."""
import difflib
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
from test_rigor import FAST, SLOW, block, entry, eff_of, rec  # noqa: E402

# --- the real b02 / b03 (rows of /home/dan/auto-runs/benchsweep/us-p1-usS1-L1of5x-b0[23].jsonl and the STARTDIAG times of their stdout)
B02_START_OFFSETS = [0, 117, 225, 343, 731, 859, 1227, 1351, 1871]       # s after 14:06:46; the 9th start (the redo of base-late) began at 14:37:57 and was cut inside its 64K request
B02_ELAPSED = 2134                                                       # 14:06:41 -> 14:42:15 would have been its end (the 9th start needs ~ 260 s); the watchdog TERM came at 1930 s
B02_PLANNED_STARTS = 7
B02_ROWS = [("updated", False), ("minus-KV_HOST_DMA", None), ("minus-HEAD_MIX_MULTI", None), ("base-m1", True), ("base-m1", False), ("minus-STAGER_RING", None),
            ("minus-PREFILL_CPU_SHARE_MAX", None), ("base-late", True)]      # (arm, regime.discarded)
B03_ROWS = [("updated", True), ("updated", False), ("plus-GDN_CHUNKED_MIN", None), ("plus-IQ_STAGE_TABLES", None), ("base-m1", True), ("base-m1", False),
            ("plus-FUSE_HEAD_GR", None), ("plus-IQ_STAGE_GRID", None)]
OLD_LIMIT = 1930


def lines_of(rows):
    out = []
    for arm, disc in rows:
        r = json.loads(rec(arm))
        if disc is not None:
            r["regime"] = {"slow": bool(disc) or False, "prefill": 1500 if disc else 2180, "load": 1.0, "discarded": disc}
        out.append(json.dumps(r))
    return out


def spec_of(arms, mid=2):
    d = {"updated": {"env": {"A": "1"}}}
    for j, a in enumerate(arms):
        if mid and j and j % mid == 0:
            d["base-m%d" % (j // mid)] = d["updated"]
        d[a] = {"env": {a: "1"}}
    return {"base": "cfg.json", "arms": d, "cases": ["short", "edit", "longdec64k"]}


def test_old_limit_cut_b02_and_the_new_limits_let_it_finish():
    est = B02_PLANNED_STARTS * S.START_S
    assert est + 600 == OLD_LIMIT and OLD_LIMIT < B02_ELAPSED                    # the old rule cut it (exit 124, inside the 64K request)
    soft, hard = S.block_limits(est)
    assert soft > B02_START_OFFSETS[-1]                                          # the last start (the redo of base-late) still begins: no soft stop before it
    assert hard > B02_ELAPSED + 300 and hard > soft                              # the watchdog does not cut the 64K request of that start
    assert soft == est + 600 + 3 * (S.SETTLE_S + S.PAUSE_S + S.START_S)          # 3 = ceil(0.3 x 7) slow starts allowed
    assert S.block_limits(0, 8)[0] > S.block_limits(0, 6)[0]                     # grows with the planned starts


def test_unfinished_arms_of_the_real_b02_and_b03_rows():
    sp2 = spec_of(["minus-KV_HOST_DMA", "minus-HEAD_MIX_MULTI", "minus-STAGER_RING", "minus-PREFILL_CPU_SHARE_MAX"])
    # b02: every arm has rows, but the closing base-late start was a discarded slow one and its redo was cut: the two arms after the last valid base (base-m1) cannot be judged
    assert S.unfinished_arms(sp2, lines_of(B02_ROWS)) == ["minus-STAGER_RING", "minus-PREFILL_CPU_SHARE_MAX"]
    sp3 = spec_of(["plus-GDN_CHUNKED_MIN", "plus-IQ_STAGE_TABLES", "plus-FUSE_HEAD_GR", "plus-IQ_STAGE_GRID"])
    assert S.unfinished_arms(sp3, lines_of(B03_ROWS)) == ["plus-FUSE_HEAD_GR", "plus-IQ_STAGE_GRID"]       # b03 never reached base-late
    done = lines_of(B03_ROWS + [("base-late", False)])
    assert S.unfinished_arms(sp3, done) == []
    assert S.unfinished_arms(sp3, lines_of(B03_ROWS[:3])) == ["plus-GDN_CHUNKED_MIN", "plus-IQ_STAGE_TABLES", "plus-FUSE_HEAD_GR", "plus-IQ_STAGE_GRID"]   # no rows at all: all of them


def test_continuation_block_holds_the_unfinished_arms_with_bases():
    sp3 = spec_of(["plus-GDN_CHUNKED_MIN", "plus-IQ_STAGE_TABLES", "plus-FUSE_HEAD_GR", "plus-IQ_STAGE_GRID"])
    cs = S.continuation_spec(sp3, lines_of(B03_ROWS), mid=2)
    assert list(cs["arms"]) == ["updated", "plus-FUSE_HEAD_GR", "plus-IQ_STAGE_GRID"] and cs["drift_check"] is True
    assert S.continuation_spec(sp3, lines_of(B03_ROWS + [("base-late", False)]), mid=2) is None
    many = spec_of(["a%d" % i for i in range(5)], mid=0)
    cs = S.continuation_spec(many, [], mid=2)
    assert list(cs["arms"]) == ["updated", "a0", "a1", "base-m1", "a2", "a3", "base-m2", "a4"]


def test_the_shell_wires_the_soft_stop_the_continuation_and_the_limits():
    sh = (CHAINS / "bench-sweep.v9.sh").read_text(encoding="utf-8")
    old = (CHAINS / "bench-sweep.sh").read_text(encoding="utf-8")
    assert subprocess.run([shutil.which("sh") or "sh", "-n", str(CHAINS / "bench-sweep.v9.sh")]).returncode == 0 or not shutil.which("sh")
    assert "ABTEST_SOFT_STOP_S" in sh and "block-limits" in sh and "continue-block" in sh and "est + 600" in sh          # the old formula is only quoted in the log line
    assert "$((est + 600)) $bsp" not in sh and old.count("$((est + 600)) $bsp") == 2
    changed = [x[2:] for x in difflib.ndiff(old.split("\n"), sh.split("\n")) if x[0] in "+-" and not x[2:].lstrip().startswith("#") and x[2:].strip()]
    assert all(("est" in x or "HARD" in x or "blim" in x or "cont" in x or "ABSOFT" in x or "p1f=" in x or "csp" in x or "crc" in x or "csrc" in x or "rows=" in x or "cn=" in x or "cs=" in x or "cr=" in x
                or "LIMITS" in x or "ABTEST_SOFT" in x or x.strip() in ("}", "done", "fi", "for sfx in c cc; do", "wait_idle", "r=$(sw block-limits") or "left_ok" in x or "SKIP continuation" in x
                or "say \"CONTINUE" in x or "set -- $r" in x or "ab $B-p1" in x) for x in changed), changed
    ab = (HERE / "abtest.py").read_text(encoding="utf-8")
    assert "ABTEST_SOFT_STOP_S" in ab and "SOFT-STOP" in ab and "first_attempt" in ab


def test_cli_block_limits_and_continue_block(tmp_path, capsys):
    assert S.main(["block-limits", "--est", "1330"]) == 0
    assert capsys.readouterr().out.split() == ["BLOCKLIMITS"] + [str(x) for x in S.block_limits(1330)]
    spec = tmp_path / "b.json"
    spec.write_text(json.dumps(spec_of(["plus-X", "plus-Y"], mid=0)))
    rows = tmp_path / "b.jsonl"
    rows.write_text("\n".join(lines_of([("updated", False), ("plus-X", None), ("base-m1", False)])))
    assert S.main(["continue-block", "--spec", str(spec), "--rows", str(rows), "--mid", "0", "--out", str(tmp_path / "c.json")]) == 0
    out = capsys.readouterr().out
    assert out.startswith("CONTINUE 1 1 arm(s): plus-Y") and list(json.loads((tmp_path / "c.json").read_text())["arms"]) == ["updated", "plus-Y"]
    rows.write_text("\n".join(lines_of([("updated", False), ("plus-X", None), ("plus-Y", None), ("base-late", False)])))
    assert S.main(["continue-block", "--spec", str(spec), "--rows", str(rows), "--out", str(tmp_path / "d.json")]) == 3


# ---------------------------------------------------------------- the FINAL ROUND
def tri_of(*es):
    return {"schema": "pancake.sweep-triage/2", "branch": "us", "flags": {e["flag"]: e for e in es}}


def test_final_round_takes_only_undecided_flags_lost_first_then_by_run1_evidence():
    a = block("plus-A", 1.0)
    files = {"us-p1-usS1-1of5v-b01.jsonl": (a, None)}
    es = [entry("KEPT", "KEEP", runs=[{"short": 3.0}, {"short": 3.0}]),                                         # decided: never retested
          entry("DROPPED", "DROP", runs=[{"short": -3.0}, {"short": -3.0}]),
          entry("MIX", "MIXED", runs=[{"short": 3.0}, {"short": 3.0}]),
          entry("LOWGAIN", "PROMISING-LOW", pass1={"short": 1.0}),                                              # not undecided (a clean one-run verdict)
          entry("LOST1", "INCONCLUSIVE", pass1={"short": 1.0, "edit": 2.0}),                                    # undecided, no clean run: lost
          entry("UNT", "UNTESTED"),                                                                             # undecided, no rows: lost, first
          entry("STRONG", "INCONCLUSIVE", run1={"short": 9.0, "edit": 1.0}, pass1={"short": 9.0, "edit": 1.0}),
          entry("WEAK", "INCONCLUSIVE", run1={"short": 0.5, "edit": 0.1}, pass1={"short": 0.5, "edit": 0.1})]
    files = {"us-p1-usS1-1of5v-b01.jsonl": (block("plus-STRONG", 1.09) + block("plus-WEAK", 1.005) + block("plus-LOST1", 1.0), None)}
    for e in es:
        if e["flag"] in ("STRONG", "WEAK"):
            e["run1"] = e["pass1"] = eff_of(files["us-p1-usS1-1of5v-b01.jsonl"][0], "plus-" + e["flag"])
    sel, back, info = S.final_round_select(tri_of(*es), files, budget_s=7200)
    names = [r["flag"] for r in sel]
    assert names[0] == "UNT" and set(names[:2]) == {"UNT", "LOST1"} and names[2:] == ["STRONG", "WEAK"] and not back
    assert all(r["flag"] not in names for r in [{"flag": x} for x in ("KEPT", "DROPPED", "MIX", "LOWGAIN")])
    assert info["n_lost"] == 2 and info["over"] is False and info["undecided"] == 4


def clean_flag(name, gain, files):
    """An INCONCLUSIVE flag WITH a clean run 1 (arm and bases fast): ranked by its run-1 gain, never `lost`."""
    lines = block("plus-" + name, gain)
    files["us-p1-usS1-1of5v-%s.jsonl" % name] = (lines, None)
    eff = eff_of(lines, "plus-" + name)
    return entry(name, "INCONCLUSIVE", run1=eff, pass1=eff)


def test_final_round_budget_fit_lost_never_dropped_and_ranked_rest_to_the_backlog():
    files = {}
    es = [entry("L%d" % i, "UNTESTED") for i in range(3)] + [clean_flag("R%d" % i, 1.10 - 0.01 * i, files) for i in range(10)]
    fit = S.rigor_fit(7200, 12, 6, 2, S.FINAL_START_S)
    assert 9 <= fit <= 12 and S.rigor_estimate(fit, 12, 6, 2, S.FINAL_START_S)[0] <= 7200 < S.rigor_estimate(fit + 1, 12, 6, 2, S.FINAL_START_S)[0]
    sel, back, info = S.final_round_select(tri_of(*es), files, budget_s=7200)
    assert [r["flag"] for r in sel[:3]] == ["L0", "L1", "L2"] and len(sel) == fit and len(back) == 13 - fit
    assert [r["flag"] for r in sel[3:]] == ["R%d" % i for i in range(fit - 3)]                              # best run-1 gain first
    assert [r["flag"] for r in back] == ["R%d" % i for i in range(fit - 3, 10)]                              # the lowest-ranked go to the backlog
    assert info["eta_s"] <= 7200 and not info["over"]
    many = [entry("L%d" % i, "UNTESTED") for i in range(fit + 3)] + [clean_flag("R0", 1.05, files)]
    sel, back, info = S.final_round_select(tri_of(*many), files, budget_s=7200)
    assert len(sel) == fit + 3 and info["over"] is True and [r["flag"] for r in back] == ["R0"]              # the lost ones are never dropped, the round runs over


def views(*effs, clean=True):
    return [{"eff": e, "clean": clean, "why": "", "file": "f%d" % i, "arm_pf": FAST, "base_pf": FAST} for i, e in enumerate(effs)]


def test_final_round_verdict_two_pairs():
    e = entry("F", "INCONCLUSIVE")
    v = S.rigor_verdict(e, views({"short": 3.0, "edit": 0.5, "longdec64k": 0.0}, {"short": 2.5, "edit": 0.0, "longdec64k": 0.2}), need=2, drop_mean=0.0, every=True)
    assert v[0] == "KEEP" or v[0] == "ADD" or v[0] == "KEEP"                                                    # a plus arm of an exact flag: KEEP (ADD is the label)
    v = S.rigor_verdict(e, views({"short": 3.0, "edit": 0.5, "longdec64k": 0.0}, {"short": 1.0, "edit": 0.0, "longdec64k": 0.2}), need=2, drop_mean=0.0, every=True)
    assert v[0] == "INCONCLUSIVE"                                                                              # the gain must be >= +2 % in BOTH reps (mean 2.0 but one rep 1.0)
    v = S.rigor_verdict(e, views({"short": -1.0, "edit": 0.0, "longdec64k": -0.5}, {"short": 0.0, "edit": -0.2, "longdec64k": -1.0}), need=2, drop_mean=0.0, every=True)
    assert v[0] == "DROP"
    v = S.rigor_verdict(e, views({"short": -1.0, "edit": 0.0, "longdec64k": -0.5, "longdec64k-prompt": 4.0}, {"short": 0.0, "edit": -0.2, "longdec64k": -1.0, "longdec64k-prompt": 3.0}), need=2, drop_mean=0.0, every=True)
    assert v[0] == "INCONCLUSIVE"                                                                              # a gain on a prompt case: not <= 0 on every case: reported, not dropped
    v = S.rigor_verdict(e, views({"short": 3.0, "edit": 0.5, "longdec64k": 0.0}), need=2, drop_mean=0.0, every=True)
    assert v[0] == "INCONCLUSIVE" and "need 2" in v[1]                                                          # one clean pair only
    v = S.rigor_verdict(e, views({"short": 3.0, "edit": -4.0, "longdec64k": 0.0}, {"short": 2.5, "edit": 0.0, "longdec64k": 0.2}), need=2, drop_mean=0.0, every=True)
    assert v[0] == "MIXED"
    # the RIGOR defaults are unchanged
    assert S.rigor_verdict(e, views({"short": 3.0}, {"short": 2.5}))[0] == "INCONCLUSIVE"                       # 2 pairs < 3


def test_final_round_cli_writes_meta_backlog_and_preview(tmp_path, capsys):
    tri = tri_of(entry("UNT", "UNTESTED"), entry("R", "INCONCLUSIVE", run1={"short": 5.0}, pass1={"short": 5.0}), entry("K", "KEEP", runs=[{"short": 3.0}, {"short": 3.0}]))
    (tmp_path / "t.json").write_text(json.dumps(tri))
    assert S.main(["final-round-select", "--triage", str(tmp_path / "t.json"), "--rows-dir", str(tmp_path), "--out", str(tmp_path / "m.json"), "--backlog", str(tmp_path / "b.json"),
                   "--md", str(tmp_path / "p.md")]) == 0
    out = capsys.readouterr().out
    assert "FINALARM 1 UNT" in out and "FINALARM 2 R" in out and "FINALETA 2 arm(s)" in out and "SELECT 2 flag(s) with verdict FINAL-ROUND: UNT, R" in out
    meta = json.loads((tmp_path / "m.json").read_text())
    assert set(meta["flags"]) == {"plus-UNT", "plus-R"} and meta["flags"]["plus-UNT"]["final_lost"] is True
    assert (tmp_path / "p.md").read_text().startswith("| rank |")


# ---------------------------------------------------------------- the developed-flag gates
def cfg(env, args):
    return {"env": env, "args": args}


def test_developed_flags_are_the_diff_of_plain_and_updated_plus_the_turbo_additions():
    plain = cfg({}, ["--pack", "p", "--expert-profile", "x.bin"])
    upd = cfg({"A": "1", "OFFSW": "0", "SAME": "1"}, ["--pack", "p", "--expert-profile", "y.bin", "--adapt-swaps", "32"])
    turbo = cfg({"A": "1", "OFFSW": "0", "SAME": "1", "T": "1"}, ["--pack", "p", "--expert-profile", "y.bin", "--adapt-swaps", "32", "--lookup-max-t", "12"])
    dev = S.developed_flags(plain, upd, turbo)
    got = {(d["flag"], d["source"]): d for d in dev}
    assert set(got) == {("A", "preset"), ("OFFSW", "preset"), ("SAME", "preset"), ("--adapt-swaps", "preset"), ("T", "turbo"), ("--lookup-max-t", "turbo")}      # --expert-profile is data, not a flag
    assert got[("OFFSW", "preset")]["off"] is True and got[("A", "preset")]["off"] is False


def test_gate_dev_select_groups_budget_and_attribution():
    plain = cfg({}, [])
    env = {"STRATA_HC_UPMIX": "1", "STRATA_PF_HCDOWN": "1", "LOSSY2": "1", "LOSSY3": "1", "EXACT": "1", "OFF": "0", "STRATA_PIN_BY_NEED": "1"}
    upd = cfg(env, [])
    turbo = cfg(env, ["--lookup-max-t", "12"])
    arm = {"exe": "/x/strata", "env": dict(env), "args": {"--lookup-max-t": "12"}}
    tri = tri_of({"flag": "EXACT", "exact": True}, {"flag": "LOSSY2", "exact": False}, {"flag": "STRATA_HC_UPMIX", "exact": False})
    r = S.gate_dev_select(plain, upd, turbo, tri, ["LOSSY3", "STRATA_PF_HCDOWN", "--lookup-max-t"], "base.json", arm)
    assert r["neutral"] == ["OFF"] and "EXACT" in r["exact"] and set(r["lossy"]) == {"STRATA_HC_UPMIX", "STRATA_PF_HCDOWN", "LOSSY2", "LOSSY3", "--lookup-max-t"}
    assert [g["flag"] for g in r["gates"]] == ["G1-hc-upmix-pf-hcdown", "G2-other-lossy-preset-flags"]               # 2 x 2700 s = the 1.5 h
    g1, g2 = r["gates"]
    assert list(g1["arms"]) == ["minus-G1-hc-upmix-pf-hcdown", "updated"] and g1["verdict_arm"] == "updated"        # the reference (scored twice) is updated minus the group
    ref = g1["arms"]["minus-G1-hc-upmix-pf-hcdown"]
    assert "STRATA_HC_UPMIX" not in ref["env"] and "STRATA_PF_HCDOWN" not in ref["env"] and ref["env"]["LOSSY2"] == "1" and ref["exe"] == "/x/strata"
    assert "STRATA_PIN_BY_NEED" not in g1["arms"]["updated"]["env"] and "--lookup-max-t" not in g1["arms"]["updated"]["args"]    # the owner-removed flags are out of the default arm
    assert [g["flag"] for g in r["backlog"] if not g.get("attribution_of")] == ["G3-turbo-lookup-max-t"]           # did not fit: backlog, not dropped
    assert sorted(g["flag"] for g in r["backlog"] if g.get("attribution_of") == "G2-other-lossy-preset-flags") == ["G2-LOSSY2", "G2-LOSSY3"]
    big = S.gate_dev_select(plain, upd, turbo, tri, ["LOSSY3", "STRATA_PF_HCDOWN", "--lookup-max-t"], "base.json", arm, budget_s=3 * 2700)
    g3 = big["gates"][2]
    assert g3["flags"] == ["--lookup-max-t"] and list(g3["arms"]) == ["updated", "plus-G3-turbo-lookup-max-t"] and g3["verdict_arm"] == "plus-G3-turbo-lookup-max-t"
    assert g3["arms"]["plus-G3-turbo-lookup-max-t"]["args"]["--lookup-max-t"] == "12" and "--lookup-max-t" not in g3["arms"]["updated"]["args"]


def test_gate_dev_collect_maps_group_verdicts_to_flags(tmp_path):
    sel = {"gates": [{"flag": "G1", "flags": ["A", "B"], "arm_name": "minus-G1", "verdict_arm": "updated"}, {"flag": "G2", "flags": ["C"], "arm_name": "plus-G2", "verdict_arm": "plus-G2"}],
           "backlog": [{"flag": "G3", "flags": ["D"]}, {"flag": "G1-A", "flags": ["A"], "attribution_of": "G1"}]}
    d = tmp_path / "g1"
    d.mkdir()
    (d / "verdicts.json").write_text(json.dumps({"updated": "PASS"}))
    out = S.gate_dev_collect(sel, [str(d)])
    assert out["flags"] == {"A": "PASS", "B": "PASS"} and out["not_run"] == ["G3"] and out["groups"][1]["verdict"] == "NOT-RUN"


def test_gate_dev_on_the_real_configs():
    root = HERE.parent.parent
    ld = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
    plain = ld(root / "configs/pancake/flashnext-iq2_xs-plain-upstream.json")
    upd, turbo = ld(CHAINS / "gate-dev/updated-hip.json"), ld(CHAINS / "gate-dev/turbo-hip.json")
    spec = ld(root / "bench/specs/220-updated-vs-upstream-short.json")
    reg = ld(CHAINS / "sweep_candidates.json")
    r = S.gate_dev_select(plain, upd, turbo, {"flags": {}}, reg["lossy"], spec["base"], spec["arms"]["updated"])
    assert {"STRATA_HC_UPMIX", "STRATA_PF_HCDOWN"} <= set(r["lossy"]) and "--lookup-max-t" in r["lossy"] and "STRATA_PIN_BY_NEED" not in [d["flag"] for d in r["developed"]]
    assert r["gates"][0]["flags"] == ["STRATA_HC_UPMIX", "STRATA_PF_HCDOWN"] and len(r["gates"]) == 2
    assert "STRATA_ROUTE_TAIL_SKIP" in r["neutral"]                                                              # the preset switches it OFF (=0): not a developed behaviour


# ---------------------------------------------------------------- the us17 order
@pytest.mark.skipif(not shutil.which("sh"), reason="needs a POSIX sh")
def test_run_queue_us17_is_us16_with_only_the_new_order_steps_and_names():
    a = (CHAINS / "run-queue-us16.sh").read_text(encoding="utf-8")
    p = CHAINS / "run-queue-us17.sh"
    b = p.read_text(encoding="utf-8")
    assert subprocess.run([shutil.which("sh"), "-n", str(p)]).returncode == 0 and "\r" not in b
    a_lines, b_lines = a.split("\n"), b.split("\n")
    changed = [(x[0], x[2:]) for x in difflib.ndiff(a_lines, b_lines) if x[0] in "+-"]
    removed = [x[1] for x in changed if x[0] == "-" and not x[1].lstrip().startswith("#")]
    assert set(removed) <= {"L=/home/dan/clone-of-d/src/run-queue-us16.log", "  hp=$O/work/us16-harness.tgz", 'say "QUEUE_US16_DONE $(date +%F_%T)"', "n=10",
                            next(x for x in a_lines if x.startswith('say "QUEUE_US16 START')),
                            next(x for x in a_lines if "BENCH_EXTRA=./bench-sweep.v8.sh" in x), next(x for x in a_lines if "BENCH_EXTRA=./bench-retest.v5.sh" in x),
                            next(x for x in a_lines if "check bench-retest.v5.log" in x), next(x for x in a_lines if x.startswith("  gd=$(ls -d $O/retest-gates-us-")),
                            next(x for x in a_lines if "--gate-dirs $gd" in x),
                            'check bench-sweep.log WINBENCHSWEEP_DONE "screen usS1-5of5x" 0', 'say "END 10 screen usS1-5of5x (started by run-queue-us14.sh) $(date +%F_%T): $(tail -n 1 bench-sweep.log)"',
                            "lostreport usS1-5of5x", "rescreen usS2r-5 --triage $PRIOR --verdict PROMISING,HURTS --slice-prefix usS1-5of5",
                            "rescreen usS2rr-5 --triage $PRIOR --verdict INCONCLUSIVE,UNTESTED --slice-prefix usS2r-5-", "leftover", "rigor", "gates",
                            "sw select --triage $PRIOR --verdict PROMISING-LOW --slice-prefix usS1- --out $O/us-backlog-low.json >> $L 2>&1"}
    for keep in ("export ABTEST_SETTLE_AFTER_LONG_S=${ABTEST_SETTLE_AFTER_LONG_S:-120}", "export ABTEST_READY_MAX_S=${ABTEST_READY_MAX_S:-900}", "export ABTEST_PREREAD=${ABTEST_PREREAD:-2}"):
        assert next(x for x in a_lines if x.startswith(keep)) == next(x for x in b_lines if x.startswith(keep))           # the same exports
    assert "SWEEP_HARNESS_SHA:?" in b and "bench-sweep.v9.sh" in b and "bench-retest.v6.sh" in b and "bench-sweep.v8.sh" not in b.replace("bench-sched.v8.sh", "")
    body = b[b.index("\nn=13\n"):]
    order = ["harness_sync\n", "lostreport usS1-L1of5x\n", "final_round\n", "hwq\n", "gates_dev\n", "rocm102\n", "turbo\n", "final final", "sw select --triage $PRIOR --verdict PROMISING-LOW", 'say "QUEUE_US17_DONE']
    pos = [body.index("\n" + x) for x in order]
    assert pos == sorted(pos) and len(set(pos)) == len(pos)
    for gone in ("\nrigor\n", "\ngates\n", "\nleftover\n", "rescreen usS2r"):
        assert gone not in body                                                                                      # the rescreens, RIGOR and the old gates are no longer called
    assert "final_pass 1 $m 0" in b and "final_pass 2 $m 1" in b and "ABTEST_SETTLE_MIN_S=" not in b[b.index("FINAL_ENV="):b.index("final_pass()")]   # settle only after long runs
    assert "rocm102-install.v1.sh" in b and '[ ! -f ./rocm102-install.v1.sh ]' in b and "sh -n ./rocm102-install.v1.sh" in b and "ROCM102_INSTALL=1 sh" in b
    assert "BENCH_BUDGET_S=${HWQ_BUDGET_S:-3600}" in b and "--budget-s ${FINAL_ROUND_BUDGET_S:-7200}" in b and "--budget-s ${GATE_DEV_BUDGET_S:-5400}" in b
    assert "FINAL_PREFIX=final FINAL_REPS=5 FINAL_RIGOR=$O/us-triage-rigor.json FINAL_TURBO=$O/us-turbo.json" in b
    assert "--gate-dev $O/us-gate-dev.json" in b


def test_s1_files_do_not_collide_with_other_spec_numbers_and_the_hwq_script_names_them():
    specs = sorted(p.name for p in (HERE.parent.parent / "bench/specs").glob("*.json"))
    nums = [n.split("-")[0] for n in specs]
    assert nums.count("290") == 1 and nums.count("291") == 1
    h = (CHAINS / "bench-hwq.v2.sh").read_text(encoding="utf-8")
    assert "290-hwq-decode" in h and "291-hwq-long64k" in h and "253-hwq" not in h and "WINBENCHHWQ_DONE" in h.split("\n")[2] + h
    for f in ("290-hwq-decode.json", "291-hwq-long64k.json"):
        d = json.loads((HERE.parent.parent / "bench/specs" / f).read_text(encoding="utf-8"))
        assert list(d["arms"]) == ["base", "hwq1", "hwq2"] and d["arms"]["hwq1"]["env"] == {"GPU_MAX_HW_QUEUES": "1"} and d["arms"]["hwq2"]["env"] == {"GPU_MAX_HW_QUEUES": "2"}
