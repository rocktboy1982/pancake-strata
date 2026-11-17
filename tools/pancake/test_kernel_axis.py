"""kernel_axis.py (LEADS.md Part P L10): the conjunctive verdict, the exhaustion rule, the trace parser and the wrapper writer.
Synthetic data only: no GPU, no rocprof.  Run: python -m pytest -q tools/pancake/test_kernel_axis.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kernel_axis as K  # noqa: E402

BASE_TPS = [100.0, 101.0, 99.5, 100.5, 100.2]
FAST_TPS = [106.0, 105.5, 106.5, 105.8, 106.2]
BASE_K = [10.0, 10.1, 9.9, 10.05, 10.0]
FAST_K = [8.0, 8.1, 7.9, 8.05, 8.0]


def test_win_needs_both_axes():
    assert K.verdict(BASE_TPS, FAST_TPS, BASE_K, FAST_K)["verdict"] == K.WIN
    # the kernel got faster, the run did not follow: DEAD (the kernel is not the bottleneck)
    r = K.verdict(BASE_TPS, BASE_TPS[::-1], BASE_K, FAST_K)
    assert r["kernel"] == K.WIN and r["tps"] == K.DEAD and r["verdict"] == K.DEAD
    # the run got faster, the kernel did not move: DEAD (noise or another cause)
    r = K.verdict(BASE_TPS, FAST_TPS, BASE_K, BASE_K[::-1])
    assert r["tps"] == K.WIN and r["kernel"] == K.DEAD and r["verdict"] == K.DEAD


def test_floor_and_few_starts():
    tiny = [x * 1.002 for x in BASE_TPS]   # +0.2%: under the 1% floor even though consistent
    assert K.axis(tiny, BASE_TPS, True, 1.0, 0.1) == K.DEAD
    assert K.axis(FAST_TPS[:2], BASE_TPS, True, 1.0, 0.1) == K.INCONCLUSIVE
    assert K.verdict(BASE_TPS, FAST_TPS, BASE_K[:2], FAST_K[:2])["verdict"] == K.INCONCLUSIVE


def test_kernel_time_axis_is_lower_is_better():
    assert K.axis(FAST_K, BASE_K, False, 1.0, 0.1) == K.WIN
    assert K.axis(BASE_K, FAST_K, False, 1.0, 0.1) == K.DEAD   # slower kernel: not a win


def test_exhaustion_after_five_and_win_resets():
    h = [{"kernel": "a", "verdict": K.DEAD}] * 4 + [{"kernel": "b", "verdict": K.INCONCLUSIVE}] * 5
    assert K.exhausted(h) == ["b"]
    h2 = [{"kernel": "a", "verdict": K.DEAD}] * 4 + [{"kernel": "a", "verdict": K.WIN}] + [{"kernel": "a", "verdict": K.DEAD}] * 4
    assert K.exhausted(h2) == []
    assert K.exhausted(h2 + [{"kernel": "a", "verdict": K.INCONCLUSIVE}]) == ["a"]


def test_trace_parser_and_share(tmp_path):
    p = tmp_path / "1_kernel_trace.csv"
    p.write_text("Kernel_Name,Start_Timestamp,End_Timestamp\nattn_chunk,0,2000000\nattn_chunk,3000000,4000000\nmoe_gate,5000000,9000000\n")
    t = K.parse_kernel_trace(p)
    assert t["attn_chunk"] == [2.0, 1.0] and t["moe_gate"] == [4.0]
    assert K.kernel_ms(t, "attn") == 3.0 and K.kernel_ms(t, None) == 7.0
    assert round(K.share_of_wall(t, "attn", 100.0), 3) == 3.0   # exactly at the 3% candidate bar
    assert K.share_of_wall(t, "attn", 100.0) >= K.MIN_SHARE


def test_wrap_writes_wrappers_and_spec(tmp_path):
    spec = tmp_path / "s.json"
    spec.write_text(json.dumps({"base": "x.json", "arms": {"base": {"exe": "/e/strata"}, "arm": {"exe": "/e/strata2", "env": {"STRATA_A": "1"}}}, "note": "n"}))
    out = tmp_path / "w"
    p = K.wrap(spec, out, rocprof="/rp")
    d = json.load(open(p))
    assert d["arms"]["arm"]["exe"].endswith("arm.sh") and d["arms"]["arm"]["env"] == {"STRATA_A": "1"}
    sh = (out / "arm.sh").read_text()
    assert "exec /rp --kernel-trace" in sh and "/e/strata2" in sh and '"$@"' in sh
    assert "kernel_axis wrap" in d["note"]
    spec.write_text(json.dumps({"base": "x.json", "arms": {"base": {}}}))
    try:
        K.wrap(spec, out)
        raise AssertionError("an arm without an exe must be refused")
    except SystemExit:
        pass


def test_judge_reads_jsonl_and_trace_dirs(tmp_path):
    jl = tmp_path / "r.jsonl"
    rows = []
    for arm, tps in (("base", BASE_TPS), ("arm", FAST_TPS)):
        for i, v in enumerate(tps):
            rows.append({"arm": arm, "rep": i, "rows": [{"case": "edit", "decode_tps": v}, {"case": "short", "decode_tps": 1.0}]})
    jl.write_text("\n".join(json.dumps(r) for r in rows))
    for arm, ks in (("base", BASE_K), ("arm", FAST_K)):
        d = tmp_path / "w" / arm / "host"
        d.mkdir(parents=True)
        for i, ms in enumerate(ks):
            (d / f"{i}_kernel_trace.csv").write_text(f"Kernel_Name,Start_Timestamp,End_Timestamp\nk,0,{int(ms * 1e6)}\n")
    assert K.main(["judge", "--jsonl", str(jl), "--dir", str(tmp_path / "w"), "--arm", "arm", "--case", "edit"]) == 0
