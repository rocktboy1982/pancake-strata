"""test_draft_report.py - the A25 / A26 report (draft_report.py) on synthetic traces and log lines (no GPU, no model).

    python tools/pancake/test_draft_report.py      (or: python -m pytest tools/pancake/test_draft_report.py)
"""
import io
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import draft_report  # noqa: E402
from test_disco_fit import write_stwt  # noqa: E402

LOG = ("strata decode timing: 100 windows, avg T 4.00, 2.50 tokens/window, 40.00 ms/window = adapt wait 1.00 + verify 30.00 "
       "(GPU-reach wait 3.00 + per-layer host 2.00 [plan 0.10 actq 0.10 jobs 0.10 CPU 1.00] + stage 0.50) + commit/emit 2.00 + "
       "draft 5.00 + adapt join 0.00 + other 2.00; per layer-window: CPU experts 3.00 (9.00 entries), VRAM hits 20.00, "
       "PCIe 1.00; 0.0 swaps/window\n")
LOG2 = LOG.replace("100 windows", "300 windows").replace("verify 30.00", "verify 20.00").replace("40.00 ms/window", "30.00 ms/window")


def spec_line(pos, T, a, sfx, ms, probs, drafts, look, out):
    return (f"R {pos} {T} {a} {sfx} {ms:.3f} {len(look)} 5 |" + "".join(f" {p:.4f}" for p in probs) + " |"
            + "".join(f" {d}" for d in drafts) + " |" + "".join(f" {t}" for t in look) + " |"
            + "".join(f" {t}" for t in out) + "\n")


def test_positions_calibration_costs_from_a_window_trace():
    # 4 MTP windows of 4 rows: accepted 3, 1, 0, 2; probabilities high for held drafts
    wins, pos = [], 100
    for a in (3, 1, 0, 2):
        probs = [0.95 if j < a else 0.2 for j in range(3)]
        wins.append({"T": 4, "a": a, "pos": pos, "ms": 30.0 + a, "probs": probs, "src": [0, 0, 0], "rm": [5, 4, 3, 2]})
        pos += a + 1
    wins.append({"T": 3, "a": 2, "pos": pos, "ms": 50.0, "probs": [float("nan")] * 2, "src": [2, 2], "rm": None})
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.stwt")
        write_stwt(p, wins)
        assert draft_report.kind_of(p) == "stwt"
        rounds = draft_report.rounds_from_stwt(p)
    assert [r["src"] for r in rounds] == ["mtp"] * 4 + ["lookup"]
    pt = draft_report.position_table(rounds)
    # row 1: reached 4 (a >= 0), held 3 (a >= 1); row 2: reached 3, held 2; row 3: reached 2, held 1
    assert [(j, n, k) for j, n, k, _, _ in pt] == [(1, 4, 3), (2, 3, 2), (3, 2, 1)]
    assert abs(pt[2][4] - 0.75 * (2 / 3) * 0.5) < 1e-12
    cal = draft_report.calibration(rounds)
    hi = [c for c in cal if c[0] == 0.875][0]
    lo = [c for c in cal if c[0] == 0.125][0]
    assert hi[2] == 6 and hi[3] == 1.0 and lo[2] == 3 and lo[3] == 0.0   # 6 held drafts at 0.95; 3 first-rejected at 0.2
    tb = draft_report.tokens_by_size(rounds)
    assert tb[4] == (4, (4 + 2 + 1 + 3) / 4)
    cost = draft_report.cost_by_size(rounds)
    assert cost[4][0] == 3 and cost[3] == (1, 50.0), cost   # the first window's time is not trusted
    by = draft_report.realised_by_source(rounds)
    assert by["mtp"]["n"] == 4 and by["lookup"]["tok"] == 3


def test_counterfactual_from_a_draft_trace():
    lines = ["# request prompt 50 df2 0 spec 4\n",
             # lookup chosen, 2 of 4 held; the drafter's chain [7 8 9] matches the committed [7 8 5]: 2 -> as long
             spec_line(50, 5, 2, 1, 40.0, [0.9, 0.8, 0.7, 0, 0, 0, 0], [7, 8, 9, 0, 0, 0, 0], [7, 8, 1, 1], [7, 8, 5]),
             # lookup chosen, 0 held; the drafter's [3 4] matches [3]: censored, 1 > 0 -> drafter longer
             spec_line(53, 3, 0, 1, 25.0, [0.9, 0.6, 0, 0, 0, 0, 0], [3, 4, 0, 0, 0, 0, 0], [2, 2], [3]),
             # drafter chosen, 1 held; the lookup [6 6 6] matches [6 2]: 1 -> as long; eligible (first tokens equal)
             spec_line(54, 4, 1, 0, 30.0, [0.9, 0.3, 0.2, 0, 0, 0, 0], [6, 1, 1, 0, 0, 0, 0], [6, 6, 6], [6, 2]),
             # drafter chosen, 0 held; the lookup [5 5] matches [5]: lookup longer; not eligible (first tokens differ)
             spec_line(56, 3, 0, 0, 20.0, [0.5, 0.4, 0, 0, 0, 0, 0], [4, 4, 0, 0, 0, 0, 0], [5, 5], [5]),
             # drafter chosen, no lookup proposal
             spec_line(57, 3, 2, 0, 22.0, [0.9, 0.9, 0, 0, 0, 0, 0], [1, 2, 0, 0, 0, 0, 0], [], [1, 2, 3])]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "spec.txt")
        Path(p).write_text("".join(lines))
        assert draft_report.kind_of(p) == "spec"
        rounds = draft_report.rounds_from_spec(p)
    assert len(rounds) == 5 and rounds[0]["src"] == "lookup" and rounds[0]["drafter"] == [7, 8, 9]
    assert rounds[2]["eligible"] and not rounds[3]["eligible"]
    assert draft_report.prefix_match([7, 8, 9], [7, 8, 5]) == (2, False)
    assert draft_report.prefix_match([3, 4], [3]) == (1, True)
    cf = draft_report.source_counterfactual(rounds, draft_report.cost_by_size(rounds))
    lk, md = cf["lookup"], cf["drafter"]
    assert lk["n"] == 2 and lk["drafter_more"] == 1 and lk["same"] == 1 and lk["drafter_fewer"] == 0
    assert md["n"] == 2 and md["eligible"] == 1 and md["lookup_more"] == 1 and md["lookup_more_eligible"] == 0 and md["same"] == 1
    assert cf["rounds"] == 5 and cf["tokens"] == 3 + 1 + 2 + 1 + 3
    pt = draft_report.position_table(rounds)   # the drafter windows of the draft trace: T 4 a 1, T 3 a 0, T 3 a 2
    assert [(j, n, k) for j, n, k, _, _ in pt] == [(1, 3, 2), (2, 2, 1), (3, 0, 0)]


def test_verify_share_from_log_lines_and_the_full_report():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "server.log")
        Path(p).write_text("noise\n" + LOG + "more noise\n" + LOG2)
        assert draft_report.kind_of(p) == "log"
        tm = draft_report.timing_from_log(p)
        assert len(tm) == 2 and tm[0]["verify"] == 30.0 and tm[0]["draft"] == 5.0 and tm[0]["commit"] == 2.0
        vs = draft_report.verify_share(tm)
        assert vs["windows"] == 400 and abs(vs["ms"] - (100 * 40 + 300 * 30) / 400) < 1e-9
        assert abs(vs["verify"] - (100 * 30 + 300 * 20) / 400) < 1e-9
        s = os.path.join(d, "spec.txt")
        Path(s).write_text(spec_line(50, 3, 1, 0, 20.0, [0.9, 0.4] + [0] * 5, [1, 2] + [0] * 5, [1, 9], [1, 3]))
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            rc = draft_report.main([p, s])
        finally:
            sys.stdout = old
        text = buf.getvalue()
        assert rc == 0 and "A25.1" in text and "A25.5" in text and "verify 22.50 (" in text and "A26" in text, text


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
