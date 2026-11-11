"""Tests for probes/gdn_chunk64_rule.py (MOD_PLAN A31): the stage bench and timing-line parsers and the decision.
Run: python tools/pancake/test_gdn_chunk64_rule.py   (or pytest)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "probes"))
import gdn_chunk64_rule as R  # noqa: E402

# The shape of hip_gdn_stage_bench output (tests/hip/gdn_stage_bench.cpp); the numbers are made up.
STAGE = """device=gfx1100 CUs=84
env STRATA_GDN_CHUNKED=1
STAGE T=4096 gates_conv_ms=0.2000 rec_out_ms=0.5000 sum_ms=0.7000 y_hash=0000000000000001 state_hash=0000000000000002 split_same=1
A31 T=4096 wmma_floor_ms=0.3550 (2 head(s) per WGP, 2400 MHz) rec_out_over_floor=1.41
STAGE T=32768 gates_conv_ms=1.6000 rec_out_ms=%s sum_ms=9.0000 y_hash=0000000000000001 state_hash=0000000000000002 split_same=1
A31 T=32768 wmma_floor_ms=2.8400 (2 head(s) per WGP, 2400 MHz) rec_out_over_floor=%s
"""

# The 41,818-token line of bench/prefetch/long-timing (RTX 3060, CUDA, 2026-11-08), cut after the recurrence entry.
TIMING_3060 = ("strata generate: GPU 0: NVIDIA GeForce RTX 3060, compute capability 8.6\n"
               "strata prefill timing: 41811 tokens, GPU timeline 42172 ms, wall 42260 ms, host staging 245 ms: "
               "embed+steps 355 (0.8%) gdn 5012 (11.9%) gdn conv+gates 546 (1.3%) gdn recurrence 2515 (6.0%) gdn out proj 1850 (4.4%)\n")


def stage(rec_ms, ratio):
    return R.parse_stage(STAGE % (rec_ms, ratio))


def test_parse_stage():
    s = stage("5.5000", "1.94")
    assert set(s) == {4096, 32768}
    assert s[32768]["rec_out_ms"] == 5.5 and s[32768]["floor_ms"] == 2.84 and s[32768]["over_floor"] == 1.94


def test_parse_timing_marks_a_cuda_card_not_applicable():
    t = R.parse_timing(TIMING_3060)
    assert t["tokens"] == 41811 and t["rec_pct"] == 6.0 and t["rec_ms"] == 2515 and not t["hip"]
    lines, verdict = R.decide({}, t, None, 36)
    assert "not applicable" in lines[0] and verdict.startswith("need both")


def test_wmma_bound_scan_closes_chunk64():
    lines, verdict = R.decide(stage("3.4000", "1.20"), None, 2580.0, 36)
    assert verdict.startswith("close chunk 64"), verdict
    assert "fails" in lines[0] and "fails" in lines[1]


def test_latency_bound_but_small_share_closes_a31():
    # ratio 2.1 (a holds), but 6 ms x 36 layers is 1.7% of a 12.7 s prompt (b fails)
    lines, verdict = R.decide(stage("6.0000", "2.11"), None, 2580.0, 36)
    assert verdict.startswith("close A31"), verdict


def test_both_hold_builds():
    # 25 ms a layer = 7.1% of a 32K prompt at 2,580 tok/s over 36 layers, 8.8 x the floor
    lines, verdict = R.decide(stage("25.0000", "8.80"), None, 2580.0, 36)
    assert verdict.startswith("build chunk 64"), verdict


def test_share_threshold_matches_the_documented_21_ms():
    # the 6% line of the 7900 XT at 32K (2,580 tok/s, 36 layers) is 21.2 ms a layer
    assert abs(R.share_from_stage(21.2, 32768, 2580.0, 36) - 6.0) < 0.05


def test_wmma_bound_with_share_met_keeps_the_column_split():
    t = {"tokens": 32768, "gpu_ms": 12700, "rec_ms": 900, "rec_pct": 7.1, "gpu_name": "AMD Radeon RX 7900 XT", "hip": True}
    lines, verdict = R.decide(stage("4.0000", "1.40"), t, None, 36)
    assert "column split" in verdict, verdict


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
