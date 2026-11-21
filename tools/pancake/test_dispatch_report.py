"""LEADS.md Part U U3: the served-path census reader (dispatch_report.py), the switch -> counter map (dispatch_expect.json) and the abtest
ABTEST_DISPATCH_REPORT option.  No server, no GPU: a fixture engine log with a counter dump, the INERT / ACTIVE rule, default off, the map's lint.

Run: python3 -I -m pytest -q tools/pancake/test_dispatch_report.py"""
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abtest as A  # noqa: E402
import dispatch_report as D  # noqa: E402
from test_abtest_core import MMAP_LOG, fake_engine  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def dump(served, launches=None, notes=()):
    """The report as include/strata/core/dispatch_stats.hpp json() lays it out."""
    o = "{\n \"launches\": {" + ", ".join(f'"{k}": {v}' for k, v in (launches or {}).items()) + "},\n \"counts\": [\n"
    o += ",\n".join(f'  {{"scope": [-1, -1], "cat": "str", "key": "{k}", "n": {n}}}' for k, n in notes)
    o += "\n ]"
    if served:
        o += ",\n \"served\": {" + ",".join(f'\n  "{k}": {{"n": {h}, "declined": {d}}}' for k, (h, d) in sorted(served.items())) + "\n }"
    return o + "\n}\n"


ENGINE_LOG = ("strata: loading the experts into RAM\nstrata serve: ready: http://127.0.0.1:8080\n" + dump({"pf.gemm.f16.hipblaslt": (40, 2), "pf.gemm.pf.bk64": (0, 5)},
              {"T4/k0": 12}, [("win groups=1", 3)]) + "strata: bye\n")


# ------------------------------------------------------------------------------------------------------------------ reading a dump
def test_parse_report_from_a_plain_file_and_from_an_engine_log():
    plain = D.parse_report(dump({"a.b": (3, 1)}))
    assert plain["served"]["a.b"] == {"n": 3, "declined": 1}
    in_log = D.parse_report(ENGINE_LOG)
    assert in_log["served"]["pf.gemm.f16.hipblaslt"] == {"n": 40, "declined": 2} and in_log["launches"] == {"T4/k0": 12}


def test_parse_report_refuses_text_without_a_dump_and_broken_json():
    assert D.parse_report("") is None
    assert D.parse_report("strata serve: ready\nplain log\n") is None
    assert D.parse_report('{"unrelated": 1}') is None
    assert D.parse_report("{\n \"launches\": {\n garbage\n}\n") is None


def test_digest_keeps_served_launches_and_notes():
    dg = D.digest(D.parse_report(ENGINE_LOG))
    assert dg["served"] == {"pf.gemm.f16.hipblaslt": [40, 2], "pf.gemm.pf.bk64": [0, 5]}
    assert dg["launches"] == {"T4/k0": 12} and dg["notes"] == {"win groups=1": 3}
    assert D.digest(None) is None
    old = D.digest(D.parse_report(dump({})))   # a default-off engine's report has no "served" key
    assert old["served"] == {}


# ------------------------------------------------------------------------------------------------------------------ the INERT rule
def dg(**served):
    return {"served": {k.replace("__", "."): list(v) for k, v in served.items()}, "launches": {}, "notes": {}}


E_UP = {"up": ["x.on"], "down": ["x.off"], "why": "t"}
E_X = {"up": ["x.on"], "mode": "cross", "why": "t"}


def test_flag_active_when_the_up_counter_rises_and_the_down_counter_falls():
    v, why = D.classify_flag(E_UP, dg(x__off=(100, 0)), dg(x__on=(100, 0), x__off=(0, 0)))
    assert v == "ACTIVE" and "x.on: 0 -> 100" in why


def test_flag_inert_when_nothing_moved_and_one_run_noise_is_not_a_move():
    v, _ = D.classify_flag(E_UP, dg(x__off=(100, 0)), dg(x__off=(100, 0)))
    assert v == "INERT"
    v, _ = D.classify_flag(E_UP, dg(x__on=(1000, 0)), dg(x__on=(1010, 0)))   # +1%: under the 2% tolerance
    assert v == "INERT"
    v, _ = D.classify_flag(E_UP, dg(x__on=(1000, 0)), dg(x__on=(1030, 0)))
    assert v == "ACTIVE"


def test_flag_declined_when_the_path_asked_and_said_no():
    v, why = D.classify_flag(E_UP, dg(x__off=(100, 0)), dg(x__on=(0, 7), x__off=(100, 0)))
    assert v == "DECLINED" and "declined 0 -> 7" in why


def test_flag_the_other_way_is_inert_and_says_so():
    v, why = D.classify_flag(E_UP, dg(x__on=(100, 0)), dg(x__on=(10, 0)))
    assert v == "INERT" and "other way" in why


def test_cross_mode_ignores_a_count_that_only_changed_in_size():
    assert D.classify_flag(E_X, dg(x__on=(10, 0)), dg(x__on=(14, 0)))[0] == "INERT"   # counted per graph capture: the number of windows differs
    assert D.classify_flag(E_X, dg(), dg(x__on=(1, 0)))[0] == "ACTIVE"


def test_a_minus_arm_reverses_the_expectation():
    base, arm = dg(x__on=(100, 0)), dg(x__off=(100, 0), x__on=(0, 0))
    assert D.classify_flag(E_UP, base, arm, reverse=False)[0] == "INERT"
    assert D.classify_flag(E_UP, base, arm, reverse=True)[0] == "ACTIVE"   # the switch taken out: x.on falls, x.off rises


def test_no_report_concludes_nothing():
    assert D.classify_flag(E_UP, None, dg())[0] == "NO-REPORT"
    assert D.classify_flag(E_UP, dg(), None)[0] == "NO-REPORT"


EXP = {"flags": {"STRATA_A": E_UP, "STRATA_B": {"up": ["y.on"], "why": "t"}}, "unmapped": {"STRATA_U": "host policy"}}


def test_arm_verdicts_aggregate_the_flags_of_an_arm():
    base = dg(x__off=(10, 0), y__on=(0, 0))
    both_inert = D.classify_arm("a", ["STRATA_A", "STRATA_B"], False, EXP, base, dg(x__off=(10, 0), y__on=(0, 0)))
    assert both_inert["verdict"] == "INERT"
    one_active = D.classify_arm("a", ["STRATA_A", "STRATA_B"], False, EXP, base, dg(x__on=(10, 0), y__on=(0, 0)))
    assert one_active["verdict"] == "ACTIVE" and one_active["flags"]["STRATA_B"][0] == "INERT"
    assert D.classify_arm("a", ["STRATA_U"], False, EXP, base, base)["verdict"] == "UNMAPPED"
    assert D.classify_arm("a", ["STRATA_ZZ"], False, EXP, base, base)["flags"]["STRATA_ZZ"][0] == "UNMAPPED"
    assert D.classify_arm("a", [], False, EXP, base, base)["verdict"] == "UNMAPPED"
    assert D.classify_arm("a", ["STRATA_A"], False, EXP, base, None)["verdict"] == "NO-REPORT"
    mixed = D.classify_arm("a", ["STRATA_A", "STRATA_U"], False, EXP, base, base)
    assert mixed["verdict"] == "INERT"   # the unmapped flag does not hide the mapped one's verdict


def test_flags_of_an_arm_from_its_name_or_its_spec():
    assert D.flags_of_arm("plus-STRATA_PF_GEMM") == (["STRATA_PF_GEMM"], False)
    assert D.flags_of_arm("minus-STRATA_PF_GEMM") == (["STRATA_PF_GEMM"], True)
    arm = {"env": {"STRATA_A": "1", "STRATA_C": None}, "args": {"--x": "2", "--y": None}}
    assert D.flags_of_arm("bundle", arm) == (["STRATA_A", "--x"], False)


def test_report_lines_and_the_base_choice():
    spec = {"arms": {"warmup": {}, "base": {}, "plus-STRATA_A": {}, "plus-STRATA_B": {}, "plus-STRATA_U": {}}}
    digs = {"warmup": dg(), "base": dg(x__off=(10, 0)), "plus-STRATA_A": dg(x__on=(10, 0)), "plus-STRATA_B": dg(), "plus-STRATA_U": dg()}
    lines, res = D.report(digs, EXP, spec)
    txt = "\n".join(lines)
    assert "base arm base" in txt and "DISPATCH plus-STRATA_A: ACTIVE" in txt and "DISPATCH plus-STRATA_B: INERT" in txt
    assert "DISPATCH plus-STRATA_U: UNMAPPED" in txt and "1 ACTIVE, 1 INERT" in txt and [r["arm"] for r in res] == ["plus-STRATA_A", "plus-STRATA_B", "plus-STRATA_U"]
    assert "warmup:" not in txt


# ------------------------------------------------------------------------------------------------------------------ files and the CLI
def test_digests_from_rows_take_the_first_attempt_with_a_report():
    rows = [{"arm": "base", "dispatch": None}, {"arm": "base", "dispatch": dg(a__b=(1, 0))}, {"arm": "x"}, {"arm": "base", "dispatch": dg(a__b=(9, 0))}]
    d = D.digests_from_rows(rows)
    assert d["base"]["served"]["a.b"] == [1, 0] and d["x"] is None


def test_cli_reads_a_work_directory_and_report_files(tmp_path, capsys):
    (tmp_path / "base-r0-a0.dispatch.json").write_text(dump({"x.off": (10, 0)}))
    (tmp_path / "plus-STRATA_A-r0-a0.dispatch.json").write_text(ENGINE_LOG.replace("pf.gemm.f16.hipblaslt", "x.on"))   # an engine log holding the dump
    (tmp_path / "plus-STRATA_B-r0-a0.dispatch.json").write_text("no report here\n")
    exp = tmp_path / "exp.json"
    exp.write_text(json.dumps(EXP))
    assert D.main(["--dir", str(tmp_path), "--base", "base", "--expect", str(exp)]) == 0
    out = capsys.readouterr().out
    assert "plus-STRATA_A: ACTIVE" in out and "plus-STRATA_B: NO-REPORT" in out
    rows = tmp_path / "rows.jsonl"
    rows.write_text("\n".join(json.dumps(r) for r in [{"arm": "base", "dispatch": dg(x__off=(10, 0))}, {"arm": "plus-STRATA_A", "dispatch": dg(x__off=(10, 0))}]))
    assert D.main(["--rows", str(rows), "--expect", str(exp), "--strict"]) == 4   # INERT arm, --strict
    assert "plus-STRATA_A: INERT" in capsys.readouterr().out
    assert D.main(["--rows", str(rows), "--expect", str(exp)]) == 0
    capsys.readouterr()
    assert D.main(["--report", f"base={tmp_path / 'base-r0-a0.dispatch.json'}", "--report", f"plus-STRATA_A={tmp_path / 'plus-STRATA_A-r0-a0.dispatch.json'}",
                   "--expect", str(exp), "--top", "3"]) == 0
    assert "x.on" in capsys.readouterr().out
    assert D.main([]) == 2


# ------------------------------------------------------------------------------------------------------------------ the map's lint
def src_served_names():
    """Every dotted name given to strata::dispatch::served / served_if in src/ (one line each; ternaries included)."""
    names = set()
    for f in list((ROOT / "src").rglob("*.cu")) + list((ROOT / "src").rglob("*.cpp")):
        for line in f.read_text(errors="replace").splitlines():
            if "dispatch::served" in line:
                names |= set(re.findall(r'"([a-z0-9_]+(?:\.[a-z0-9_]+)+)"', line))
    return names


def registry_names():
    reg = json.loads((ROOT / "tools/pancake/box_chains/sweep_candidates.json").read_text())
    names = {e["name"] for b in ("us", "rd") for e in reg["untested"][b]} | {e["name"] for e in reg["backlog"]["us"]} | set(reg["protected"])
    for g in reg["groups"].values():
        names |= set(g)
    return names


def test_the_expectation_file_lints_against_the_sources_and_the_sweep_registry():
    exp = D.load_expect()
    assert D.lint_expect(exp, registry_names(), src_served_names()) == []
    assert len(exp["flags"]) >= 10 and len(exp["unmapped"]) > len(exp["flags"])   # the clean ones mapped, the rest listed


def test_lint_catches_the_mistakes_it_is_for():
    base = {"flags": {"STRATA_A": {"up": ["a.b"], "why": "w"}}, "unmapped": {"STRATA_U": "r"}}
    assert D.lint_expect(base, {"STRATA_A", "STRATA_U"}, {"a.b"}) == []
    assert any("both mapped and unmapped" in p for p in D.lint_expect({"flags": base["flags"], "unmapped": {"STRATA_A": "r"}}))
    assert any("not a served() name" in p for p in D.lint_expect(base, None, {"other.name"}))
    assert any("neither mapped nor unmapped" in p for p in D.lint_expect(base, {"STRATA_A", "STRATA_U", "STRATA_NEW"}, {"a.b"}))
    assert any("bad counter name" in p for p in D.lint_expect({"flags": {"F": {"up": ["Bad Name"], "why": "w"}}, "unmapped": {}}))
    assert any("mode" in p for p in D.lint_expect({"flags": {"F": {"up": ["a.b"], "mode": "x", "why": "w"}}, "unmapped": {}}))
    assert any("non-empty" in p for p in D.lint_expect({"flags": {"F": {"why": "w"}}, "unmapped": {}}))
    assert any("no why" in p for p in D.lint_expect({"flags": {"F": {"up": ["a.b"]}}, "unmapped": {}}))


def test_every_counter_site_in_src_is_a_dotted_lowercase_literal_and_unique_per_meaning():
    names = src_served_names()
    assert len(names) >= 40
    assert all(re.match(r"^(pf|rdna|attn|qsa|gdn)\.[a-z0-9_.]+$", n) for n in names), sorted(names)


# ------------------------------------------------------------------------------------------------------------------ abtest: default off, and on
def test_dispatch_option_is_off_by_default_and_changes_nothing(monkeypatch, tmp_path):
    monkeypatch.delenv("ABTEST_DISPATCH_REPORT", raising=False)
    assert A.dispatch_report_enabled() is False
    assert A.dispatch_report_enabled({"ABTEST_DISPATCH_REPORT": "0"}) is False and A.dispatch_report_enabled({"ABTEST_DISPATCH_REPORT": "1"}) is True
    log, sent = fake_engine(monkeypatch, tmp_path, MMAP_LOG + "\n", lambda t: 124.0)
    cfg = {"log": str(log), "env": {"STRATA_X": "1"}}
    res, err = A.run_arm("base", str(tmp_path / "c.json"), cfg, ["short"], {"short": "S"}, str(tmp_path / "base-r0-a0.out"), 8080)
    assert err is None and "dispatch" not in res
    assert cfg == {"log": str(log), "env": {"STRATA_X": "1"}} and not (tmp_path / "c.json").exists()   # no STRATA_DISPATCH_REPORT, no config rewrite
    assert not list(tmp_path.glob("*.dispatch.json"))


def test_dispatch_option_on_sets_the_report_variable_and_keeps_the_digest(monkeypatch, tmp_path):
    monkeypatch.setenv("ABTEST_DISPATCH_REPORT", "1")
    log, sent = fake_engine(monkeypatch, tmp_path, MMAP_LOG + "\n", lambda t: 124.0)
    seen = {}
    plain_start = A.start_server

    def start(cfg_path, log_out, port, ready_wait=900):
        c = json.loads(Path(cfg_path).read_text())
        seen["path"] = c["env"]["STRATA_DISPATCH_REPORT"]
        Path(seen["path"]).write_text(dump({"pf.gemm.f16.hipblaslt": (7, 1)}))   # what the engine writes when it exits
        return plain_start(cfg_path, log_out, port, ready_wait)

    monkeypatch.setattr(A, "start_server", start)
    cfg = {"log": str(log), "env": {}}
    res, err = A.run_arm("base", str(tmp_path / "c.json"), cfg, ["short"], {"short": "S"}, str(tmp_path / "base-r0-a0.out"), 8080)
    assert err is None and seen["path"] == str(tmp_path / "base-r0-a0.dispatch.json")
    assert res["dispatch"]["served"] == {"pf.gemm.f16.hipblaslt": [7, 1]}
    json.dumps(res)   # the row stays jsonl-able
    # an engine that left no report: the digest is None (NO-REPORT downstream), never a crash
    (tmp_path / "base-r1-a0.dispatch.json").write_text("stale")
    monkeypatch.setattr(A, "start_server", plain_start)
    res2, err2 = A.run_arm("base", str(tmp_path / "c.json"), {"log": str(log), "env": {}}, ["short"], {"short": "S"}, str(tmp_path / "base-r1-a0.out"), 8080)
    assert err2 is None and res2["dispatch"] is None   # the stale file was removed before the start


def test_print_dispatch_verdicts_marks_an_inert_arm(capsys):
    spec = {"arms": {"base": {}, "plus-STRATA_PF_GEMM": {}, "plus-STRATA_HCD_EXACT": {}}}
    records = [{"arm": "base", "dispatch": dg(pf__gemm__f16__hipblaslt=(50, 0))},
               {"arm": "plus-STRATA_PF_GEMM", "dispatch": dg(pf__gemm__f16__hipblaslt=(50, 0))},   # the arm's counters equal the base's: the switch never acted
               {"arm": "plus-STRATA_HCD_EXACT", "dispatch": dg(pf__gemm__f16__hipblaslt=(50, 0), pf__gemm__hcdown_exact=(6, 0))}]
    A.print_dispatch_verdicts(records, spec, "base")
    out = capsys.readouterr().out
    assert "DISPATCH plus-STRATA_PF_GEMM: INERT" in out and "DISPATCH plus-STRATA_HCD_EXACT: ACTIVE" in out
