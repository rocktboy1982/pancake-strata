"""abtest.py block mode, warm-up passes, robust outlier rule, selection, keyed pairing, tier recording and the per-attempt
engine log.  No server, no GPU: run_arm and main run against fakes (monkeypatched server start, requests and noise monitor).

Run: python3 -I -m pytest -q tools/pancake/test_abtest_core.py"""
import contextlib
import io
import json
import math
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abtest as A  # noqa: E402
import abtest_replay as R  # noqa: E402


def rec(arm, rep, attempt=0, clean=True, noise=2.0, reclaim=0.0, rows=None, **kw):
    return {"rep": rep, "arm": arm, "attempt": attempt, "clean": clean, "noise": noise, "reclaim": reclaim,
            "rows": rows if rows is not None else [], "texts": {}, **kw}


def row(case, tps, **kw):
    return {"case": case, "decode_tps": tps, "prompt_tps": 600.0, "prompt": 100, "gen": 512, "accept": None, **kw}


def out(fn, *args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args)
    return buf.getvalue()


# ---------------------------------------------------------------------------------------------------- block scheduler
def test_schedule_without_block_is_the_old_abba_order():
    order, passes = A.schedule(["a", "b", "c"], 4)
    assert passes == 1
    assert order == [(0, "a"), (0, "b"), (0, "c"), (1, "c"), (1, "b"), (1, "a"),
                     (2, "a"), (2, "b"), (2, "c"), (3, "c"), (3, "b"), (3, "a")]


@pytest.mark.parametrize("reps,block,rounds", [(6, 3, 2), (6, 4, 2), (7, 2, 4), (3, 6, 2), (12, 3, 4), (1, 1, 2)])
def test_schedule_block_runs_n_passes_per_start_and_at_least_two_starts_per_arm(reps, block, rounds):
    arms = ["base", "x"]
    order, passes = A.schedule(arms, reps, block)
    assert passes == block
    assert len(order) == rounds * len(arms)
    for arm in arms:
        assert sum(1 for _, x in order if x == arm) == rounds >= 2
    assert rounds * passes >= reps
    assert order[:4] == [(0, "base"), (0, "x"), (1, "x"), (1, "base")]   # A B B A


def test_schedule_block_gives_every_arm_early_and_late_slots_equally():
    order, _ = A.schedule(["a", "b", "c"], 12, 3)   # 4 rounds
    pos = {}
    for i, (r, x) in enumerate(order):
        pos.setdefault(x, []).append(i - 3 * r)   # slot within its round
    assert len({sum(v) for v in pos.values()}) == 1


def test_case_text_prefixes_later_passes_and_keeps_long_cases_unique():
    P = {"short": "S", "edit": "E", "long1k": "L"}
    assert A.case_text(P, "short") == "S" and A.case_text(P, "short", 0) == "S"
    assert A.case_text(P, "edit", 2) == "[pass 2]\nE"
    assert A.case_text(P, "short", 1) == A.case_text(P, "short", 1)   # the same text in every start (comparable hashes)
    a, b = A.case_text(P, "long1k", 1), A.case_text(P, "long1k", 1)
    assert a != b and a.endswith("\nL") and a.startswith("[run ")


# ---------------------------------------------------------------------------------------------------- robust outlier rule
def test_robust_bounds_needs_three_values_and_uses_the_floor():
    assert A.robust_bounds([100.0, 120.0]) is None
    lo, hi = A.robust_bounds([100.0, 100.0, 100.0, 100.0])          # MAD 0: the floor decides
    assert lo == pytest.approx(100 - 3.5 * 2.5) and hi == pytest.approx(100 + 3.5 * 2.5)
    lo, hi = A.robust_bounds([100.0, 110.0, 90.0, 100.0, 120.0])    # MAD 10 > floor
    assert lo == pytest.approx(100 - 3.5 * 1.4826 * 10) and hi == pytest.approx(100 + 3.5 * 1.4826 * 10)


def test_robust_keep_drops_a_slow_attempt_whole_and_keeps_normal_spread():
    # 25-kern-hip-confirm all3: short 120.9 (-5%, inside the bounds) and edit 101.4 (-29%) in the same start
    used = {"all3": [rec("all3", 0, rows=[row("short", 120.9), row("edit", 101.4)])] +
                    [rec("all3", r, rows=[row("short", 127.3), row("edit", e)]) for r, e in zip(range(1, 6), (145.5, 131.0, 142.5, 144.8, 143.1))],
            "base": [rec("base", r, rows=[row("short", 124.0), row("edit", e)]) for r, e in enumerate((143.9, 136.7, 133.4, 143.8, 143.3, 143.5))]}
    kept, dropped = A.robust_keep(used)
    assert [(d["arm"], d["rep"], d["pass"]) for d in dropped] == [("all3", 0, 0)]
    assert [c for c, _, b in dropped[0]["rows"] if b] == ["edit"] and len(dropped[0]["rows"]) == 2
    assert kept["all3"][0]["rows"] == [] and all(len(r["rows"]) == 2 for r in kept["all3"][1:])
    assert all(len(r["rows"]) == 2 for r in kept["base"])          # base edit 133.4 (-7%, a quiet attempt) is kept


def test_robust_keep_judges_a_screens_second_request_in_its_own_group():
    # the second edit in the same server is ~35% faster (cached prompt): it must not make the first one an outlier
    used = {"a": [rec("a", r, rows=[row("edit", 140.0 + r), row("edit", 190.0 + r)]) for r in range(4)]}
    kept, dropped = A.robust_keep(used)
    assert not dropped and all(len(r["rows"]) == 2 for r in kept["a"])


def test_robust_keep_drops_only_the_outlier_pass_of_a_block():
    rows = [row(c, v, **{"pass": k, "warm": True}) for k, (s, e) in enumerate([(124, 143), (95, 110), (125, 144)], 1)
            for c, v in (("short", s), ("edit", e))]
    used = {"a": [rec("a", 0, rows=rows), rec("a", 1, rows=[dict(r) for r in rows if r["pass"] != 2])]}
    kept, dropped = A.robust_keep(used)
    assert [(d["rep"], d["pass"]) for d in dropped] == [(0, 2)]
    assert {r["pass"] for r in kept["a"][0]["rows"]} == {1, 3} and len(kept["a"][1]["rows"]) == 4


def test_robust_keep_uses_prompt_tps_for_prefill_cases():
    used = {"a": [rec("a", r, rows=[{"case": "long3k", "decode_tps": 50.0 if r else 10.0, "prompt_tps": p, "prompt": 3000,
                                     "gen": 32, "accept": None}]) for r, p in enumerate((900.0, 905.0, 600.0, 898.0))]}
    _, dropped = A.robust_keep(used)
    assert [d["rep"] for d in dropped] == [2]   # prompt tok/s 600 is the outlier; decode tok/s 10 of rep 0 does not count


def test_slow_starts_against_the_faster_of_two_or_the_median_of_three():
    two = [rec("a", 0, rows=[row("short", 127.0), row("edit", 143.0)]), rec("a", 1, rows=[row("short", 111.2), row("edit", 130.6)])]
    assert A.slow_starts(two) == {1}                                            # -12.4% on short
    assert A.slow_starts([two[0], rec("a", 1, rows=[row("short", 122.0), row("edit", 136.0)])]) == set()   # -3.9% / -4.9%: normal spread
    three = two + [rec("a", 2, rows=[row("short", 140.0), row("edit", 158.0)])]  # one fast start does not make the others slow
    assert A.slow_starts(three) == {1}
    block = [rec("a", 0, rows=[row("short", v, **{"pass": k}) for k, v in enumerate((124, 125, 126))]),
             rec("a", 1, rows=[row("short", v, **{"pass": k}) for k, v in enumerate((98, 99, 100))])]
    assert A.slow_starts(block) == {1}                                          # judged on the start's mean
    assert A.slow_starts(block[:1]) == set()


def test_robust_picks_replaces_a_slow_start_by_its_slot_s_next_attempt():
    fast = lambda arm, rep, attempt=0: rec(arm, rep, attempt, rows=[row("short", 125.0 + rep)])
    slow = lambda arm, rep, attempt=0: rec(arm, rep, attempt, rows=[row("short", 100.0)])
    recs = [fast("a", 0), slow("a", 1), fast("b", 0), fast("b", 1), slow("a", 1, 1), fast("a", 1, 2)]
    picks = A.robust_picks(recs)
    assert picks[("a", 1)]["attempt"] == 2 and picks[("a", 0)]["attempt"] == 0 and picks[("b", 1)]["attempt"] == 0
    assert A.robust_picks(recs[:2])[("a", 1)] is None                         # every attempt so far slow: run it again
    used, dropped = A.select(recs, outlier="robust")
    assert [(d["arm"], d["rep"], d["attempt"]) for d in dropped] == [("a", 1, 0), ("a", 1, 1)]
    assert [r["attempt"] for r in used["a"]] == [0, 2]


# ---------------------------------------------------------------------------------------------------- selection
def test_select_gate_takes_the_first_clean_attempt_else_the_quietest():
    recs = [rec("base", 0, 0, False, noise=30), rec("base", 0, 1, True, noise=3), rec("x", 0, 0, False, noise=9),
            rec("x", 0, 1, False, noise=4), rec("x", 0, 2, False, noise=6), rec("base", 1, 0, True)]
    used, dropped = A.select(recs)
    assert dropped == [] and [r["attempt"] for r in used["base"]] == [1, 0] and [r["attempt"] for r in used["x"]] == [1]
    assert [r["rep"] for r in used["base"]] == [0, 1]


def test_select_gate_replays_other_thresholds():
    recs = [rec("a", 0, 0, False, noise=3, reclaim=2.1), rec("a", 0, 1, True, noise=3, reclaim=0.0)]
    assert A.select(recs)[0]["a"][0]["attempt"] == 1
    assert A.select(recs, foreign_cpu=25.0, reclaim_max=3.0)[0]["a"][0]["attempt"] == 0


def test_select_robust_takes_the_first_attempt_and_reads_the_mode_from_the_records():
    recs = [rec("a", r, 0, False, noise=40, rows=[row("short", 124.0 + r)], opts={"outlier": "robust"}) for r in range(3)]
    recs.insert(1, rec("a", 0, 1, True, rows=[row("short", 1.0)], opts={"outlier": "robust"}))
    used, _ = A.select(recs)
    assert [r["attempt"] for r in used["a"]] == [0, 0, 0]
    assert A.select(recs, outlier="gate")[0]["a"][0]["attempt"] == 1


# ---------------------------------------------------------------------------------------------------- pairing and summary
def test_paired_matches_list_order_when_complete_and_keys_by_rep_when_not():
    res = {"base": [rec("base", r, rows=[row("short", 100.0 + r)]) for r in range(4)],
           "x": [rec("x", r, rows=[row("short", 110.0 + r)]) for r in (0, 2, 3)]}
    assert A.paired(res, "x", "base", "short") == [10.0, 10.0, 10.0]   # list order would give 10, 9, 9
    res["x"].insert(1, rec("x", 1, rows=[row("short", 111.0)]))
    assert A.paired(res, "x", "base", "short") == [10.0] * 4


def test_paired_pairs_block_passes_by_round_and_pass():
    mk = lambda arm, r, vals: rec(arm, r, rows=[row("short", v, **{"pass": k + 1}) for k, v in enumerate(vals)])
    res = {"base": [mk("base", 0, [100, 101, 102]), mk("base", 1, [90, 91, 92])],
           "x": [mk("x", 0, [105, 106, 107]), mk("x", 1, [95, 96, 97])]}
    assert A.paired(res, "x", "base", "short") == [5.0] * 6


def summary_numbers(text, arm, case="short"):
    m = re.search(rf"{case}\s+{arm}\s+([\d.]+) \+-\s+([\d.]+)\s+\(n=(\d+)\)(?:\s+vs \w+:\s+([+-][\d.]+)% \+-\s+([\d.]+)%)?", text)
    return m.groups()


def test_summary_format_is_unchanged_and_block_widens_the_intervals():
    res = {"base": [rec("base", r, rows=[row("short", v)]) for r, v in enumerate((120.0, 122.0, 121.0, 123.0))],
           "x": [rec("x", r, rows=[row("short", v)]) for r, v in enumerate((126.0, 125.0, 127.0, 128.0))]}
    text = out(A.summary, res, ["base", "x"], ["short"])
    assert text.splitlines()[1] == ("summary (decode tok/s; prompt tok/s for long* but longdec*; mean +- 95% interval over "
                                    "clean runs, or the quietest attempt)")
    assert "  short  base                    121.5 +-   2.1  (n=4)\n" in text
    assert "  short  x                       126.5 +-   2.1  (n=4)   vs base:  +4.1% +-  1.9%  -> better\n" in text
    wide = out(A.summary, res, ["base", "x"], ["short"], A.design_effect(3))
    assert float(summary_numbers(wide, "x")[1]) == pytest.approx(A.ci([126.0, 125.0, 127.0, 128.0])[1] * math.sqrt(1.9), abs=0.051)
    d = A.ci([6.0, 3.0, 6.0, 5.0])[1] * math.sqrt(1.9) / 121.5 * 100
    assert float(summary_numbers(wide, "x")[4]) == pytest.approx(d, abs=0.051)
    assert summary_numbers(wide, "x")[0] == summary_numbers(text, "x")[0]


def test_summary_without_common_reps_prints_no_comparison():
    res = {"base": [rec("base", 0, rows=[row("short", 120.0)])], "x": [rec("x", 1, rows=[row("short", 125.0)])]}
    text = out(A.summary, res, ["base", "x"], ["short"])
    assert "short  x" in text and "vs base" not in text


def test_decided_needs_least_values_and_respects_the_design_effect():
    res = {"base": [rec("base", r, rows=[row("short", 100.0 + (r % 2) * 0.2)]) for r in range(4)],
           "x": [rec("x", r, rows=[row("short", 103.0 + (r % 2) * 0.4)]) for r in range(4)]}
    assert A.decided(res, ["base", "x"], ["short"])
    assert not A.decided(res, ["base", "x"], ["short"], least=5)
    assert A.design_effect(1) == 1 and A.design_effect(3) == pytest.approx(1.9)


# ---------------------------------------------------------------------------------------------------- tier
MMAP_LOG = """strata generate: --ram-keep-gib 20.00: the expert arena is used only if 20.00 GiB of RAM stay available after it
strata generate: not enough RAM for the expert arena (it needs 53.02 GiB ...): this run reads the experts from the file
strata generate: experts via mmap (not enough RAM for the expert arena: the same source as --mmap-experts)"""


def test_tier_of_prefers_the_engine_then_capped_then_the_rule():
    assert A.tier_of(MMAP_LOG) == ("mmap", "engine")
    assert A.tier_of("strata generate: expert arena: 33.1 GiB pinned\n") == ("arena", "engine")
    assert A.tier_of("[strata] experts loaded: 38.20 GiB at 1.40 GiB/s (40 s so far)") == ("arena", "engine")
    assert A.tier_of("[strata] loading the experts into RAM (about 39 GB) and locking part of them") == ("unknown", "none")
    assert A.tier_of("capped: tier=arena keep_gib=12 big_free_gib=35.2\n") == ("arena", "capped")
    assert A.tier_of("capped: tier=arena keep_gib=12 big_free_gib=35\n" + MMAP_LOG) == ("mmap", "engine")
    assert A.tier_of("", keep=12, big=34.0) == ("arena", "guess")
    assert A.tier_of("", keep=20, big=34.0) == ("mmap", "guess")
    assert A.tier_of("", keep=12, big=0.4) == ("mmap", "guess")
    assert A.tier_of("", keep=12, big=-1.0) == ("unknown", "none")


def test_keep_gib_order(monkeypatch):
    monkeypatch.setenv("STRATA_RAM_KEEP_GIB", "16")
    assert A.keep_gib(MMAP_LOG, {"STRATA_RAM_KEEP_GIB": "8"}) == 20.0
    assert A.keep_gib("capped: tier=mmap keep_gib=24 big_free_gib=1\n", {}) == 24.0
    assert A.keep_gib("", {"STRATA_RAM_KEEP_GIB": "8"}) == 8.0
    assert A.keep_gib("", {}) == 16.0
    monkeypatch.delenv("STRATA_RAM_KEEP_GIB")
    assert A.keep_gib("", {}) == 12.0


def test_print_selection_warns_on_mixed_tiers_only():
    used = {"base": [rec("base", 0, tier="mmap"), rec("base", 1, tier="unknown")], "x": [rec("x", 0, tier="mmap")]}
    assert out(A.print_selection, used, [], False) == ""
    used["x"].append(rec("x", 1, tier="arena"))
    assert "mix memory tiers" in out(A.print_selection, used, [], False)


# ---------------------------------------------------------------------------------------------------- run_arm against a fake server
class FakeMonitor:
    reclaim = 0.0

    def __init__(self, roots):
        pass

    def start(self):
        pass

    def result(self):
        return 2.0, []

    def gpu_summary(self):
        return {}


def fake_engine(monkeypatch, tmp_path, start_lines, speed):
    """The shared engine log gets the start lines on start and one engine line per request (decode tok/s from speed(text))."""
    log = tmp_path / "engine.log"
    log.write_text("old run line 1\nold run line 2\n")
    sent = []

    def start_server(cfg_path, log_out, port, ready_wait=900):
        Path(log_out).write_text("capped: tier=arena keep_gib=12 big_free_gib=1.0\n[strata] loading the experts into RAM\n")
        with open(log, "a") as f:
            f.write(start_lines)
        return type("P", (), {"pid": 1})(), None

    def request(url, text, max_tokens=512):
        sent.append(text)
        with open(log, "a") as f:
            f.write(f"strata serve: prompt 71 tokens, read in 1600 ms (44.5 tok/s), 512 generated in 4100 ms ({speed(text):.1f} tok/s)\n")
        return "answer " + text[:12]

    monkeypatch.setattr(A, "start_server", start_server)
    monkeypatch.setattr(A, "request", request)
    monkeypatch.setattr(A, "stop_servers", lambda: None)
    monkeypatch.setattr(A, "token_check", lambda *a: None)
    monkeypatch.setattr(A, "NoiseMonitor", FakeMonitor)
    monkeypatch.setattr(A, "big_free_gib", lambda: 0.4)
    return log, sent


def test_run_arm_default_mode_rows_are_unchanged(monkeypatch, tmp_path):
    log, sent = fake_engine(monkeypatch, tmp_path, MMAP_LOG + "\n", lambda t: 124.0)
    res, err = A.run_arm("base", "c.json", {"log": str(log), "env": {}}, ["short", "edit"], {"short": "S", "edit": "E"},
                         str(tmp_path / "base-r0-a0.out"), 8080)
    assert err is None and sent == ["S", "S", "E"]   # the single warm-up request, then one pass
    assert [set(r) for r in res["rows"]] == [{"prompt", "prompt_tps", "gen", "decode_tps", "accept", "case"}] * 2
    assert "warmup_rows" not in res
    assert (res["tier"], res["tier_source"], res["keep_gib"], res["big_free_gib_start"]) == ("mmap", "engine", 20.0, 0.4)
    eng = (tmp_path / "base-r0-a0.engine.log").read_text()
    assert "old run line" not in eng and eng.startswith("strata generate: --ram-keep-gib 20.00") and eng.count("strata serve: prompt") == 3


def test_run_arm_block_with_warmup_measures_only_the_passes_after_it(monkeypatch, tmp_path):
    speed = lambda t: 100.0 if not t.startswith("[pass") else 120.0 + int(t[6])   # pass 0 slow, pass k: 120 + k
    log, sent = fake_engine(monkeypatch, tmp_path, "strata generate: expert arena: ok\n", speed)
    res, err = A.run_arm("x", "c.json", {"log": str(log), "env": {}}, ["short", "edit"], {"short": "S", "edit": "E"},
                         str(tmp_path / "x-r1-a0.out"), 8080, passes=3, warmup=1)
    assert err is None
    assert sent == ["S", "E", "[pass 1]\nS", "[pass 1]\nE", "[pass 2]\nS", "[pass 2]\nE", "[pass 3]\nS", "[pass 3]\nE"]
    assert [(r["pass"], r["warm"], r["case"], r["decode_tps"]) for r in res["warmup_rows"]] == [(0, False, "short", 100.0), (0, False, "edit", 100.0)]
    assert [(r["pass"], r["warm"], r["decode_tps"]) for r in res["rows"]] == [(k, True, 120.0 + k) for k in (1, 1, 2, 2, 3, 3)]
    assert res["texts"] == {"short": A.hashlib.sha1(b"answer [pass 3]\nS").hexdigest()[:16],
                            "edit": A.hashlib.sha1(b"answer [pass 3]\nE").hexdigest()[:16]}
    assert (res["tier"], res["tier_source"]) == ("arena", "engine")
    assert (tmp_path / "x-r1-a0.engine.log").read_text().count("strata serve: prompt") == 8


def test_run_arm_block_without_warmup_keeps_the_short_warmup_request(monkeypatch, tmp_path):
    log, sent = fake_engine(monkeypatch, tmp_path, "", lambda t: 124.0)
    res, _ = A.run_arm("x", "c.json", {"log": str(log), "env": {}}, ["short"], {"short": "S"}, str(tmp_path / "x.out"), 8080,
                       passes=2)
    assert sent == ["S", "S", "[pass 1]\nS"]
    assert [(r["pass"], r["warm"]) for r in res["rows"]] == [(0, False), (1, True)]
    assert (res["tier"], res["tier_source"], res["keep_gib"]) == ("arena", "capped", 12.0)


def test_run_arm_writes_the_engine_log_of_a_start_that_died(monkeypatch, tmp_path):
    log = tmp_path / "engine.log"
    log.write_text("before\n")

    def died(cfg_path, log_out, port, ready_wait=900):
        with open(log, "a") as f:
            f.write("strata generate: out of memory\n")
        return None, "server died before ready"

    monkeypatch.setattr(A, "start_server", died)
    monkeypatch.setattr(A, "big_free_gib", lambda: 0.4)
    res, err = A.run_arm("x", "c.json", {"log": str(log)}, ["short"], {"short": "S"}, str(tmp_path / "x-r0-a0.out"), 8080)
    assert res is None and "died" in err
    assert (tmp_path / "x-r0-a0.engine.log").read_text() == "strata generate: out of memory\n"


# ---------------------------------------------------------------------------------------------------- main against a fake run_arm
def run_main(monkeypatch, tmp_path, argv, speeds):
    """speeds(arm, rep, attempt) -> (decode tok/s, noise, reclaim); returns (jsonl records, printed text, run_arm calls)."""
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"env": {}, "args": [], "log": str(tmp_path / "engine.log")}))
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"base": str(base), "arms": {"base": {}, "x": {"env": {"STRATA_X": "1"}}}, "cases": ["short"]}))
    calls = []

    def run_arm(name, cfg_path, cfg, cases, P, log_out, port, per_run=1, passes=1, warmup=0):
        r, a = re.search(r"-r(\d+)-a(\d+)\.out$", log_out).groups()
        calls.append((name, int(r), int(a), passes, warmup))
        tps, noise, reclaim = speeds(name, int(r), int(a))
        rows = [row("short", tps + k * 0.1, **({"pass": k + warmup, "warm": k + warmup > 0} if passes > 1 or warmup else {}))
                for k in range(passes)]
        return {"rows": rows, "texts": {"short": "h"}, "noise": noise, "noise_top": [], "reclaim": reclaim, "gpu": {},
                "big_free_gib": 0.4, "tier": "mmap", "tier_source": "engine", "keep_gib": 20.0, "big_free_gib_start": 0.4}, None

    monkeypatch.setattr(A, "run_arm", run_arm)
    monkeypatch.setattr(A, "quiet_gate", lambda *a, **k: True)
    monkeypatch.setattr(A, "stop_servers", lambda: None)
    jsonl = tmp_path / "res.jsonl"
    monkeypatch.setattr(sys, "argv", ["abtest.py", str(spec), "--out", str(jsonl), "--no-early-stop"] + argv)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        A.main()
    return [json.loads(line) for line in jsonl.read_text().splitlines()], buf.getvalue(), calls


def noisy_first(arm, rep, attempt):
    return (124.0 if arm == "base" else 127.0) + rep * 0.01, (40.0 if attempt == 0 else 2.0), 0.0


def test_main_gate_repeats_a_noisy_attempt(monkeypatch, tmp_path):
    recs, text, calls = run_main(monkeypatch, tmp_path, ["--reps", "2"], noisy_first)
    assert len(calls) == 8 and all(c[3:] == (1, 0) for c in calls)   # 4 slots x (noisy + clean)
    assert [r["clean"] for r in recs] == [False, True] * 4
    assert recs[0]["opts"] == {"test_tier": "confirm", "reps": 2, "block": 0, "passes": 1, "warmup": 0, "outlier": "gate",
                               "foreign_cpu": 25.0, "reclaim_max": 1.0, "retries": 2, "early_stop": False, "per_run": 1}
    assert recs[0]["tier"] == "mmap" and "NOISY -> repeat" in text and "outlier rule" not in text
    assert "  short  x                       127.0 +-   0.1  (n=2)   vs base:  +2.4%" in text


def test_main_robust_block_never_repeats_and_runs_two_starts_per_arm(monkeypatch, tmp_path):
    recs, text, calls = run_main(monkeypatch, tmp_path, ["--reps", "6", "--block", "3", "--warmup", "1", "--outlier", "robust"],
                                 noisy_first)
    assert [(c[0], c[1], c[2]) for c in calls] == [("base", 0, 0), ("x", 0, 0), ("x", 1, 0), ("base", 1, 0)]
    assert all(c[3:] == (3, 1) for c in calls)
    assert all(not r["clean"] for r in recs) and all(r["opts"]["outlier"] == "robust" for r in recs)
    assert "noisy by the proxies: kept" in text and "0 slow starts left out" in text and "0 passes dropped" in text
    assert "design effect 1.90" in text and "(n=6)" in text and "vs base:" in text


def test_main_robust_runs_a_slow_start_again_in_its_round(monkeypatch, tmp_path):
    def speeds(arm, rep, attempt):   # x's first start (round 0) is a slow engine state, its re-run is normal
        return (100.0 if (arm, rep, attempt) == ("x", 0, 0) else 127.0 if arm == "x" else 124.0), 2.0, 0.0
    recs, text, calls = run_main(monkeypatch, tmp_path, ["--reps", "4", "--block", "2", "--outlier", "robust"], speeds)
    # x round 0 looks fine alone; x round 1 shows it slow -> run again right after, as round 0 attempt 1
    assert [(c[0], c[1], c[2]) for c in calls] == [("base", 0, 0), ("x", 0, 0), ("x", 1, 0), ("x", 0, 1), ("base", 1, 0)]
    assert "rep 0 x: slow start" in text and "1 slow starts left out" in text
    assert "short  x                       127.0" in text and "(n=4)   vs base:  +2.4%" in text


def test_main_rejects_block_and_robust_on_a_screen(monkeypatch, tmp_path):
    for extra in (["--block", "2"], ["--outlier", "robust"]):
        with pytest.raises(SystemExit):
            run_main(monkeypatch, tmp_path, ["--tier", "screen"] + extra, noisy_first)


# ---------------------------------------------------------------------------------------------------- replay of the real 2026-11-11 data
GPU = A.ROOT / "bench/results/2026-11-11-gpu"


def test_replay_kern_hip_confirm():
    recs = R.load(GPU / "25-kern-hip-confirm.jsonl")
    assert len(recs) == 27 and len(R.slots(recs)) == 18 and not R.is_screen(recs)
    assert R.attempts_run(recs, "gate") == 27
    assert R.attempts_run(recs, "gate", 25.0, 3.0) == 20
    assert R.attempts_run(recs, "robust") == 20   # 18 slots + the all3 rep 0 re-runs (its 3 recorded attempts were all slow)
    used, dropped = A.select(recs, outlier="robust")
    # all3 rep 0: 120.9 / 101.4, 96.5 / 111.2, 111.2 / 130.6 against 127 / 143 in reps 1-5: slow starts, the slot is left out
    assert [(d["arm"], d["rep"], d["attempt"], d["pass"]) for d in dropped] == [("all3", 0, a, None) for a in range(3)]
    assert R.requests_used(used) == 34 and len(used["all3"]) == 5
    gate, _ = A.select(recs)
    assert [r["attempt"] for r in gate["all3"]][0] == 2   # the gate kept a proxy-clean attempt that decoded 111 / 131


def test_replay_marks_screens_and_counts_attempts():
    recs = R.load(GPU / "10-rank-guard-fold.jsonl")
    assert R.is_screen(recs) and R.attempts_run(recs, "gate") == 10 and R.attempts_run(recs, "robust") == 4
    r0 = recs[0]
    assert R.seconds(r0, True) < R.seconds(r0) < R.seconds(r0, True) * 2   # the cached second requests add decode only


def test_model_seconds_counts_starts_and_passes():
    # 3 arms x 6 reps, block 3: 6 starts of (overhead + (1 warm-up + 3) passes)
    assert R.model_seconds(3, 6, 3, 100.0, 10.0) == 6 * (100 + 4 * 10)
