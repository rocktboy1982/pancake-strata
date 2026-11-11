"""Tests for kernel_gate.py (MOD_PLAN project feature 9): the parsers, the lane model against the A6 numbers, the start counts,
the verdicts and the table of the items decided so far.  Fixtures are copied from bench/results/2026-11-09-gpuprof-defer/summary.txt,
bench/results/2026-11-11-gpu/25-kern-hip-confirm.jsonl and the batch2 / batch3 notes; the expdec_bench and q2_rows_bench texts
have the exact printf format of their sources with the notes' numbers where the notes have them (marked synthetic where not).
Run: python tools/pancake/test_kernel_gate.py   (or pytest)"""
import contextlib
import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kernel_gate as K  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# gpu_profile.py summary.txt of the deferral profile (2026-11-09): header, top kernels (cut after the 20th) and the first idle line
SUMMARY = """decode phase 5741 ms wall, kernels 3947 ms (69%), 210856 kernels, 412 per token
top kernels:
   31.1%   1227.4 ms   11280x   108.8 us  wait_flag_ge_kernel
    8.8%    346.9 ms    4080x    85.0 us  expdec_gu_kernel<22, 1, 4>
    7.0%    278.2 ms    5760x    48.3 us  wait_flag_copy_i32_kernel
    6.1%    239.3 ms    5760x    41.5 us  native_down_multi_kernel<42, true, false>
    5.5%    216.4 ms   11520x    18.8 us  rdna_kq_kernel<23, 4, 3>
    3.6%    142.9 ms    6144x    23.3 us  rdna_gr_fused<4>
    3.3%    132.1 ms    6120x    21.6 us  rdna_kq_kernel<23, 5, 2>
    2.8%    108.7 ms    1790x    60.7 us  attn_chunk_kernel<1, false, false, false>
    2.2%     85.7 ms    3264x    26.3 us  rdna_gr_fused<5>
    2.1%     84.1 ms   11040x     7.6 us  copy_rows_by_weight_kernel
    1.7%     67.0 ms    1320x    50.8 us  expdec_gu_kernel<16, 1, 4>
    1.7%     66.4 ms    3059x    21.7 us  rdna_kq_kernel<23, 6, 2>
    1.5%     60.4 ms    4320x    14.0 us  gdn_step_norm_multi_kernel<true, true>
    1.4%     54.7 ms    5616x     9.7 us  route_multi
    1.2%     46.7 ms    1632x    28.6 us  rdna_gr_fused<6>
    1.1%     45.0 ms     119x   378.1 us  gdn_step_commit_layers_kernel
    1.1%     42.3 ms    1790x    23.6 us  attn_merge_kernel
    1.0%     39.5 ms    7808x     5.1 us  bf16_f32_mmvf_multi_kernel<256, 4, true>
    1.0%     39.5 ms    5760x     6.9 us  doorbell_publish_kernel
    1.0%     38.0 ms     890x    42.7 us  rdna_kq_kernel<23, 1, 5>
idle: launch gaps (<50 us) 1111 ms, host stalls (>1 ms) 550 ms of 5741 ms
largest idle by kernel pair:
    453.9 ms     81x avg  5604.0 us  __amd_rocclr_copyBuffer -> copy_i32_from_mapped_kernel
"""

# two rows of bench/results/2026-11-09-7900xt-gpuprof/run_kernel_stats.csv (rocprofv3 --stats, whole run)
STATS_CSV = '''"Name","Calls","TotalDurationNs","AverageNs","Percentage","MinNs","MaxNs","StdDev"
"void strata::kernels::(anonymous namespace)::combine_cpu_k10_vec4<true>(HIP_vector_type<float, 4u> const volatile*, unsigned int)",4128,1049049787,254130.277859,38.26,5600,7831467,363407.266359
"void strata::kernels::(anonymous namespace)::wait_flag_copy_i32_kernel(int const volatile*, int*)",4128,100000000,24224.0,4.1,5440,900000,2000.0
"void strata::kernels::(anonymous namespace)::expdec_gu_kernel<22, 1, 4, false, 1>(unsigned char const*)",3000,255000000,85000.0,9.5,70000,120000,5000.0
'''

# expdec_bench cell lines: the format of src/kernels/expdec_bench.cpp:341; the microseconds are synthetic (the notes have none)
EXPDEC = """# device 0: AMD Radeon RX 7900 XT, 10 reps
# dev T shape gu/d choice all_us gu_us down_us GB/s pct identical|DIFFERENT
0 4 real 16/42 def 400.0 250.0 140.0 300.0 38.0% identical
0 4 real 16/42 dec:s0 300.0 190.0 100.0 400.0 50.0% identical
0 4 real 16/42 dec:s1 285.0 175.0 100.0 421.0 52.0% identical
0 4 real 22/42 dec:s0 500.0 330.0 150.0 300.0 38.0% identical
0 4 real 22/42 dec:s1 470.0 300.0 150.0 319.0 40.0% identical
0 4 real 16/42 dec:l16 300.0 180.0 100.0 400.0 50.0% identical
0 4 real 16/42 dec:l0 300.0 190.0 100.0 400.0 50.0% identical
0 2 real 16/42 dec:s0 200.0 120.0 70.0 300.0 38.0% identical

expdec_bench: ok (0 choice comparisons differ)
"""

# q2_rows_bench, nt=1: the single-thread DRAM / L2 rows of pancake-docs/batch3/kern-cpu.md (down 31.9 -> 21.8, gate/up 111.3 -> 89.3
# ns/row, floors 5.45 / 21.82 ns at 33 GB/s); the 8-thread block is synthetic (the open A48 measurement)
Q2_ROWS = """Q2_0 row kernels, one thread, nt=1, chunk=64 rows, AVX-VNNI no; dispatcher picks FAST (STRATA_Q2_FAST / STRATA_Q2_LEGACY)

down  (640 values, 10 blocks, 180 B/row), L2-hot, 256 KB: 1456 rows, fast vs legacy bitwise equal: yes
  floor        5.45 ns/row at 33.0 GB/s (row bytes / DDR4 bandwidth)
  legacy      32.30 ns/row    30.96 Mrows/s    5.57 GB/s of weights   5.93 x floor
  fast        20.50 ns/row    48.78 Mrows/s    8.78 GB/s of weights   3.76 x floor   x1.58 vs legacy
  dispatch    20.50 ns/row    48.78 Mrows/s    8.78 GB/s of weights   3.76 x floor   x1.58 vs legacy
  read only    3.00 ns/row   333.33 Mrows/s   60.00 GB/s of weights   0.55 x floor

down  (640 values, 10 blocks, 180 B/row), DRAM, 256 MB: 1456 rows, fast vs legacy bitwise equal: yes
  floor        5.45 ns/row at 33.0 GB/s (row bytes / DDR4 bandwidth)
  legacy      31.90 ns/row    31.35 Mrows/s    5.64 GB/s of weights   5.85 x floor
  fast        21.80 ns/row    45.87 Mrows/s    8.26 GB/s of weights   4.00 x floor   x1.46 vs legacy
  dispatch    21.80 ns/row    45.87 Mrows/s    8.26 GB/s of weights   4.00 x floor   x1.46 vs legacy
  read only    7.50 ns/row   133.33 Mrows/s   24.00 GB/s of weights   1.38 x floor
  -- 8 threads, 64 rows per call:
  legacy       7.20 ns/row   138.89 Mrows/s   25.00 GB/s of weights   1.32 x floor
  fast         6.60 ns/row   151.52 Mrows/s   27.27 GB/s of weights   1.21 x floor   x1.09 vs legacy
  read only    6.00 ns/row   166.67 Mrows/s   30.00 GB/s of weights   1.10 x floor

gate/up (2560 values, 40 blocks, 720 B/row), DRAM, 256 MB: 1456 rows, fast vs legacy bitwise equal: yes
  floor       21.82 ns/row at 33.0 GB/s (row bytes / DDR4 bandwidth)
  legacy     111.30 ns/row     8.98 Mrows/s    6.47 GB/s of weights   5.10 x floor
  fast        89.30 ns/row    11.20 Mrows/s    8.06 GB/s of weights   4.09 x floor   x1.25 vs legacy
  dispatch    89.30 ns/row    11.20 Mrows/s    8.06 GB/s of weights   4.09 x floor   x1.25 vs legacy
  read only   30.00 ns/row    33.33 Mrows/s   24.00 GB/s of weights   1.37 x floor
  -- 8 threads, 64 rows per call:
  legacy      27.00 ns/row    37.04 Mrows/s   26.67 GB/s of weights   1.24 x floor
  fast        25.00 ns/row    40.00 Mrows/s   28.80 GB/s of weights   1.15 x floor   x1.08 vs legacy
  read only   24.00 ns/row    41.67 Mrows/s   30.00 GB/s of weights   1.10 x floor

bitwise equal on all shapes: yes
"""

# 25-kern-hip-confirm.jsonl: (arm, rep, attempt, clean, noise, short decode tok/s, edit decode tok/s), every attempt of base and
# expdec-swiglu (the A33 arm)
CONFIRM = [
    ("base", 0, 0, False, 3.7, 124.3, 143.9), ("base", 0, 1, True, 2.2, 125.1, 139.2),
    ("expdec-swiglu", 0, 0, False, 3.8, 125.9, 142.7), ("expdec-swiglu", 0, 1, True, 2.2, 125.9, 142.6),
    ("expdec-swiglu", 1, 0, False, 3.5, 123.5, 144.3), ("expdec-swiglu", 1, 1, True, 2.6, 124.3, 138.8),
    ("base", 1, 0, True, 2.8, 122.8, 136.7), ("base", 2, 0, True, 2.8, 123.8, 133.4),
    ("expdec-swiglu", 2, 0, True, 1.8, 123.7, 145.5), ("expdec-swiglu", 3, 0, True, 2.8, 125.8, 144.9),
    ("base", 3, 0, True, 3.4, 124.5, 143.8), ("base", 4, 0, True, 2.6, 122.3, 143.3),
    ("expdec-swiglu", 4, 0, False, 7.9, 120.1, 144.6), ("expdec-swiglu", 4, 1, True, 2.5, 126.0, 140.7),
    ("expdec-swiglu", 5, 0, False, 3.3, 123.8, 143.5), ("expdec-swiglu", 5, 1, True, 2.2, 121.5, 137.0),
    ("base", 5, 0, True, 1.7, 124.3, 143.5),
]


def confirm_records():
    return [{"arm": a, "rep": rep, "attempt": at, "clean": cl, "noise": n,
             "rows": [{"case": "short", "decode_tps": s}, {"case": "edit", "decode_tps": e}]}
            for a, rep, at, cl, n, s, e in CONFIRM]


def table():
    return K.parse_kernel_table(SUMMARY)


def model():
    global _MODEL
    if "_MODEL" not in globals():
        _MODEL = K.LaneModel()
    return _MODEL


# ---- parsers ----------------------------------------------------------------------------------------------------------
def test_kernel_table_summary():
    t = table()
    assert t.windows == 120                       # 5760 wait_flag_copy_i32 calls / 48 layers
    assert abs(t.ms(K.GU22) - 346.9 / 120) < 1e-9 and abs(t.calls(K.GU22) - 34) < 1e-9
    assert abs(t.ms(K.GU16) - 67.0 / 120) < 1e-9 and abs(t.calls(K.GU16) - 11) < 1e-9
    assert abs(t.ms(K.DOWN) - 239.3 / 120) < 1e-9 and abs(t.calls(K.DOWN) - 48) < 1e-9
    assert abs(t.unlisted_ms() - 38.0 / 120) < 1e-9     # the 20th row bounds every kernel the summary does not list
    try:
        t.ms("route_publish_fast")
        raise AssertionError("a kernel missing from the table must raise")
    except KeyError as e:
        assert "not in the table" in str(e)


def test_kernel_table_rocprof_csv():
    t = K.parse_kernel_table(STATS_CSV)
    assert t.windows == 86                        # 4128 combine_cpu / 48 ... wait_flag_copy comes first: 4128 / 48
    assert abs(t.kernels["wait_flag_copy_i32_kernel"]["min_us"] - 5.44) < 1e-9
    assert "expdec_gu_kernel<22, 1, 4, false, 1>" in t.kernels     # the five-argument name of the A37 build
    assert abs(t.ms(K.GU22) - 255.0 / 86) < 1e-9 and t.unlisted_ms() == 0.0


def test_expdec_bench_cells():
    rows = K.parse_expdec_bench(EXPDEC)
    assert len(rows) == 8 and rows[0]["pair"] == "16/42" and rows[0]["choice"] == "def" and rows[0]["identical"]
    assert K.expdec_cell(rows, "22/42", "dec:s1")["gu_us"] == 300.0
    assert K.expdec_cell(rows, "16/42", "dec:s0", T=2)["all_us"] == 200.0
    try:
        K.expdec_cell(rows, "21/20", "def")
        raise AssertionError("an absent cell must raise")
    except ValueError:
        pass
    try:
        K.parse_expdec_bench("nothing here")
        raise AssertionError("text without cells must raise")
    except ValueError:
        pass


def test_q2_rows_bench_and_expert_time():
    rows = K.parse_q2_rows_bench(Q2_ROWS)
    one = K.q2_expert_saving(rows, threads=1)
    # kern-cpu.md: per expert on one thread legacy 224 us, fast 170 us
    assert abs(one["legacy_us"] - 224.1) < 0.2 and abs(one["fast_us"] - 170.1) < 0.2 and one["nt"] == 1
    assert abs(one["frac"] - 0.241) < 0.002
    pool = K.q2_expert_saving(rows)               # default: the largest thread count present
    assert pool["threads"] == 8
    assert abs(pool["legacy_us"] - (1280 * 27.0 + 2560 * 7.2) / 1e3) < 1e-9
    assert 0.05 < pool["frac"] < 0.10
    bad = Q2_ROWS.replace("fast vs legacy bitwise equal: yes", "fast vs legacy bitwise equal: NO", 1)
    try:
        K.q2_expert_saving(K.parse_q2_rows_bench(bad))
        raise AssertionError("a non-bitwise-equal fast kernel must be refused")
    except ValueError as e:
        assert "not bitwise equal" in str(e)


# ---- end-to-end evidence ----------------------------------------------------------------------------------------------
def test_kept_runs_first_clean_else_quietest():
    kept = K.kept_runs(confirm_records())
    assert kept[("base", 0)]["attempt"] == 1                      # a0 was noisy, a1 clean
    noisy = [{"arm": "a", "rep": 0, "attempt": i, "clean": False, "noise": n, "rows": []} for i, n in enumerate((9.0, 4.0, 6.0))]
    assert K.kept_runs(noisy)[("a", 0)]["attempt"] == 1           # nothing clean: the quietest attempt
    assert len(kept) == 12


def test_paired_gain_is_the_abtest_summary_line_of_spec_25():
    recs = confirm_records()
    m, h, n, sd = K.paired_gain(recs, "expdec-swiglu", "short")
    assert n == 6 and abs(m - 0.6) < 0.1 and abs(h - 1.8) < 0.15 and abs(sd - 1.7) < 0.15
    m, h, n, sd = K.paired_gain(recs, "expdec-swiglu", "edit")
    assert n == 6 and abs(m - 1.1) < 0.2 and abs(h - 4.7) < 0.3 and abs(sd - 4.5) < 0.3
    assert K.paired_gain(recs, "expdec-swiglu", "edit", base="nothing") is None
    med, groups = K.paired_sd_from_jsonl({"x": recs}, min_pairs=4)
    assert len(groups) == 2 and 1.0 < med < 5.0


def test_committed_jsonl_gives_the_recorded_numbers():
    d = ROOT / "bench/results/2026-11-11-gpu"
    if not d.exists():
        return
    ev = K.e2e_evidence(d)
    assert "short +0.6" in ev["A33"] and "no difference" in ev["A33"]
    med, groups = K.paired_sd_from_jsonl(K.load_jsonl_dir(d))
    assert len(groups) == 4 and 1.5 < min(g[4] for g in groups) and max(g[4] for g in groups) < 6.5 and 3.0 < med < 6.0


# ---- the lane model ------------------------------------------------------------------------------------------------------
def test_lane_model_reproduces_the_a6_windows():
    m = model()
    assert abs(m.window_ms("edit", ps_cut=0.0) - 33.76) < 0.02                      # the calibration point (deferral, edit)
    assert abs(m.window_ms("edit", mode="serial", ps_cut=0.0) - 39.26) < 0.25       # A6: serial 39.26 (not fitted)
    assert abs(m.base["edit"] - 33.37) < 0.1                                        # A6: deferral + FUSE_PHASES 33.37 (not fitted)
    bench = K.WINDOWS["edit"]["bench_ms"]
    assert abs(bench - 34.42) < 1e-9 and abs(4.70 / bench * 1e3 - 136.5) < 0.1      # the measured edit base, 135-138 tok/s


def test_lane_exposure_follows_the_critical_path():
    m = model()
    for w in ("short", "edit"):
        assert abs(m.exposure(w, "router", 0.5) - 1.0) < 0.01     # work before the doorbell is always on the chain
    assert m.exposure("edit", "mixer", 0.5) < m.exposure("short", "mixer", 0.5) <= 1.0   # the mixer overlaps the CPU of the layer before
    # the VRAM expert kernels hide behind the CPU rows on edits (A36) and show on short windows, where the CPU is done first
    assert m.exposure("edit", "hit", 0.5) < 0.5 and m.exposure("short", "hit", 0.5) > 0.8
    # even removing every GPU expert kernel buys < 2% of an edit window
    assert m.gain_pct("edit", "hit", K.A6_P["E"]) < 2.0
    # a CPU kernel gain lands on edits, hardly on short windows
    assert m.gain_pct("edit", "cpu", 0.1) > 5 * m.gain_pct("short", "cpu", 0.1) > 0


def test_lane_gain_arithmetic_and_edges():
    m = model()
    assert m.gain_pct("edit", "router", 0.0) == 0.0 and m.delta_ms("short", "hit", 0.0) == 0.0
    d = m.delta_ms("edit", "router", 0.58)
    assert abs(d - 0.58) < 0.01
    assert abs(m.gain_pct("edit", "router", 0.58) - (34.42 / (34.42 - d) - 1) * 100) < 1e-9
    assert m.gain_pct("edit", "hit", -0.5) < 0               # a slower kernel is a negative gain
    assert m.gain_pct("edit", "cpu", 0.2) > m.gain_pct("edit", "cpu", 0.1) > 0
    try:
        m.delta_ms("edit", "nowhere", 1.0)
        raise AssertionError("an unknown lane must raise")
    except ValueError:
        pass
    assert abs(K.prefill_gain_pct(0.5) - 100.0) < 1e-9 and K.prefill_gain_pct(0.0) == 0.0


# ---- starts -------------------------------------------------------------------------------------------------------------
def test_t_crit_and_pairs_needed():
    assert K.t_crit(5) == 2.571 and K.t_crit(30) == 2.042
    assert abs(K.t_crit(60) - 2.000) < 0.01 and abs(K.t_crit(120) - 1.980) < 0.01     # the large-df formula against the table
    assert K.pairs_needed(2.0, 3.9) == 18 and K.starts_for(2.0, 3.9)[0] == 36        # half-width t(17) * 3.9 / sqrt(18) = 1.94
    assert K.pairs_needed(99.0, 3.9) == 2                                           # abtest needs at least 2 reps
    assert K.pairs_needed(0.0, 3.9) is None and K.pairs_needed(-1.0, 3.9) is None
    assert K.pairs_needed(0.001, 3.9) is None                                        # beyond the cap
    a, b = K.pairs_needed(1.0, 3.9), K.pairs_needed(1.0, 7.8)
    assert 3.6 < b / a < 4.4                                                        # starts grow with SD squared
    assert K.pairs_needed(2.0, 3.9, power=True) > K.pairs_needed(2.0, 3.9)
    # the lesson of F21 / F9 / A33: +-1% cannot be resolved end to end
    assert K.starts_for(1.0, 3.9)[0] > 100


# ---- verdicts -------------------------------------------------------------------------------------------------------------
def cand(point, bound, measured=False, lane="router", kind="decode"):
    return K.Candidate("X", "test item", "STRATA_X=1", kind, lane, point, bound, measured, "test", "something")


def test_verdict_rules():
    m = model()
    go = K.evaluate(cand(1.0, 1.0, measured=True), m)                  # 1 ms before the doorbell = 3% of an edit window
    assert go.verdict == "GO" and go.starts[0] <= 20
    small = K.evaluate(cand(0.3, 0.3, measured=True), m)
    assert small.verdict == "NO-GO" and "measured saving" in small.reason
    est = K.evaluate(cand(1.0, 1.2, measured=False), m)                # an estimate that passes is not enough: measure the kernel
    assert est.verdict == "MEASURE"
    low = K.evaluate(cand(None, 0.3), m)                               # the bound fails: no kernel number can rescue it
    assert low.verdict == "NO-GO" and "even the bound" in low.reason
    open_ = K.evaluate(cand(None, 1.2), m)
    assert open_.verdict == "MEASURE" and open_.gains["edit"][0] is None
    slower = K.evaluate(cand(-0.5, 0.0, measured=True), m)             # a measured slowdown
    assert slower.verdict == "NO-GO" and slower.starts is None
    # a point above the stated bound raises the bound (point <= bound always holds in the table)
    assert K.evaluate(cand(1.0, 0.1), m).gains["edit"][1] >= K.evaluate(cand(1.0, 0.1), m).gains["edit"][0]
    pre = K.evaluate(cand(0.5, 0.5, measured=True, lane="gpu", kind="prefill"), m)
    assert pre.verdict == "GO" and set(pre.gains) == {"prefill"} and pre.starts[0] == 4


def test_short_only_is_said():
    m = model()
    r = K.evaluate(cand(None, 1.0, lane="hit"), m)                     # 1 ms of hit kernels: >= 2% on short, < 2% on edit
    assert r.verdict == "MEASURE" and "short window only" in r.reason


# ---- the table of the decided items -----------------------------------------------------------------------------------------
def results():
    return {r.cand.key: r for r in (K.evaluate(c, model()) for c in K.build_candidates(table(), e2e={"A33": "x"}))}


def test_table_has_every_item_and_its_verdict():
    r = results()
    assert list(r) == ["F7", "F12", "F21", "F9", "A33", "A37", "FLAT", "A36", "A31", "BFLA"]
    verdict = {k: v.verdict for k, v in r.items()}
    assert verdict == {"F7": "MEASURE", "F12": "MEASURE", "F21": "NO-GO", "F9": "MEASURE", "A33": "MEASURE", "A37": "NO-GO",
                       "FLAT": "MEASURE", "A36": "MEASURE", "A31": "MEASURE", "BFLA": "GO"}
    # every item that is not measured says what to measure; the measured one says nothing is missing
    assert all(v.cand.missing for k, v in r.items() if k != "BFLA") and r["BFLA"].cand.missing.startswith("64K")
    # no GPU item is measured yet: only BFLA (an end-to-end number) is
    assert [k for k, v in r.items() if v.cand.measured] == ["BFLA"]


def test_table_numbers_from_the_notes():
    r = results()
    # F21 kern-hip.md: <= ~0.6% of a short window, ~0 on edits; the bound (the whole 5.44 us) stays under the bar
    assert r["F21"].gains["short"][0] < 1.0 and r["F21"].gains["edit"][0] < 0.5 and r["F21"].gains["short"][1] < 2.0
    # F9 kern-hip.md: ~12 us a layer = ~0.6 ms: +1-2%; all of it lands (router lane)
    assert abs(r["F9"].cand.point - 0.576) < 1e-9 and 1.5 < r["F9"].gains["edit"][0] < 2.0 and r["F9"].exposure["edit"] > 0.99
    # A33 kern-hip.md: ~1% of a window; the unlisted SwiGLU kernel bounds the rest
    assert 1.0 < r["A33"].gains["short"][0] < 1.5 and r["A33"].gains["short"][1] < 3.0 and r["A33"].cand.e2e == "x"
    # A37 kern-hip.md: <= 0.5% of decode, unmeasurable in a screen
    assert r["A37"].gains["short"][1] < 1.0 and r["A37"].starts[0] > K.E2E_MAX_STARTS
    # F7 kern-cpu.md: <= ~22% of the CPU-expert time, which lands on edits
    assert r["F7"].cand.bound == 0.22 and r["F7"].gains["edit"][1] > 5 > r["F7"].gains["short"][1]
    # flat loads and A36 are the same kernels at the same 25% bound; the hit lane hardly reaches an edit window
    assert r["FLAT"].gains == r["A36"].gains and r["FLAT"].gains["edit"][1] < 2.0 < r["FLAT"].gains["short"][1]
    # A31: the bound is halving a 6% share = 3.1%; BFLA: 386.9 -> 771.2 tok/s
    assert abs(r["A31"].gains["prefill"][1] - 3.09) < 0.05 and abs(r["BFLA"].gains["prefill"][0] - 99.3) < 0.1
    # the break-even of the router lane is the bar itself: 2% of a 34.4 ms window ~ 0.67 ms
    assert abs(r["F9"].breakeven["edit"] - 0.67) < 0.03


# ---- measurements replace the missing numbers --------------------------------------------------------------------------------
def with_evidence(key, **kw):
    ev = K.Evidence(item=key, **kw)
    return {r.cand.key: r for r in (K.evaluate(c, model()) for c in K.build_candidates(table(), ev))}


def test_f12_from_expdec_bench():
    r = with_evidence("F12", expdec=K.parse_expdec_bench(EXPDEC))["F12"]
    # 16/42: 11 layers x (190 - 175) us + 22/42: 34 layers x (330 - 300) us
    assert r.cand.measured and abs(r.cand.point - (11 * 15 + 34 * 30) / 1e3) < 1e-9
    assert r.verdict == "GO" and r.cand.missing == ""                  # 1.185 ms: 4.4% of a short window


def test_a37_reads_the_iq2_xxs_pair_only():
    r = with_evidence("A37", expdec=K.parse_expdec_bench(EXPDEC))["A37"]
    assert abs(r.cand.point - 11 * (190 - 180) / 1e3) < 1e-9           # dec:l0 -> dec:l16 on 16/42
    assert r.verdict == "NO-GO"                                         # 0.11 ms: under the bar on both windows
    r = with_evidence("A37", expdec=K.parse_expdec_bench(EXPDEC), base_choice="dec:l0", new_choice="dec:l16")["A37"]
    assert r.cand.measured


def test_expdec_refuses_a_cell_that_is_not_bitwise_identical():
    bad = EXPDEC.replace("0 4 real 22/42 dec:s1 470.0 300.0 150.0 319.0 40.0% identical",
                         "0 4 real 22/42 dec:s1 470.0 300.0 150.0 319.0 40.0% DIFFERENT")
    try:
        with_evidence("F12", expdec=K.parse_expdec_bench(bad))
        raise AssertionError("a DIFFERENT cell must be refused")
    except ValueError as e:
        assert "DIFFERENT" in str(e)


def test_flat_loads_from_two_builds_gate_up_and_down():
    base = "0 4 real 16/42 dec 300.0 190.0 100.0 400.0 50.0% identical\n0 4 real 22/42 dec 480.0 330.0 150.0 300.0 38.0% identical\n"
    new = "0 4 real 16/42 dec 280.0 170.0 100.0 420.0 52.0% identical\n0 4 real 22/42 dec 440.0 290.0 140.0 330.0 41.0% identical\n"
    r = with_evidence("FLAT", expdec=K.parse_expdec_bench(base), expdec_new=K.parse_expdec_bench(new),
                      base_choice="dec", new_choice="dec")["FLAT"]
    # gate/up: 11 x 20 + 34 x 40 us; down: the mean of (0, 10) us x 48 layers
    assert r.cand.measured and abs(r.cand.point - (11 * 20 + 34 * 40 + 48 * 5) / 1e3) < 1e-9
    assert r.verdict == "GO" and r.gains["short"][0] > 5 > r.gains["edit"][0]      # 1.8 ms of hit kernels: on short, hardly on edits
    same = with_evidence("FLAT", expdec=K.parse_expdec_bench(base), base_choice="dec", new_choice="dec")["FLAT"]
    assert same.cand.point == 0.0 and same.verdict == "NO-GO"


def test_f7_from_q2_rows_bench():
    rows = K.parse_q2_rows_bench(Q2_ROWS)
    pool = with_evidence("F7", q2rows=rows)["F7"]
    assert pool.cand.measured and abs(pool.cand.point - (1 - (1280 * 25.0 + 2560 * 6.6) / (1280 * 27.0 + 2560 * 7.2))) < 1e-9
    assert pool.gains["edit"][0] > 2.0 and pool.verdict == "GO"        # ~8% of the CPU time lands ~4% on edits
    one_thread = [r for r in rows if r["threads"] == 1]
    single = with_evidence("F7", q2rows=one_thread)["F7"]
    assert not single.cand.measured and single.cand.point is None      # one thread is not the pool: still a bound
    assert single.cand.bound == 0.22 and single.verdict == "MEASURE"


def test_a31_from_the_stage_bench():
    stage = ("STAGE T=32768 gates_conv_ms=1.6000 rec_out_ms=%s sum_ms=9.0000 y_hash=0000000000000001 state_hash=0000000000000002 split_same=1\n"
             "A31 T=32768 wmma_floor_ms=2.8400 (2 head(s) per WGP, 2400 MHz) rec_out_over_floor=%s\n")
    near = K.g64.parse_stage(stage % ("3.5000", "1.23"))
    r = with_evidence("A31", stage=near, prefill_tok_s=2580)["A31"]
    assert r.verdict == "NO-GO"                    # a scan near its WMMA floor: chunk 64 adds work, the rule closes A31
    far = K.g64.parse_stage(stage % ("25.0000", "8.80"))
    r = with_evidence("A31", stage=far, prefill_tok_s=2580)["A31"]
    assert r.verdict == "MEASURE" and r.gains["prefill"][1] > 2.0 and "built chunk-64" in r.cand.missing
    assert abs(r.cand.bound - K.g64.share_from_stage(25.0, 32768, 2580, 36) / 100 * (1 - 1.12 * 2.84 / 25.0)) < 1e-9


# ---- command line -----------------------------------------------------------------------------------------------------------
def run_main(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = K.main(argv)
    return rc, out.getvalue()


def test_cli_table_and_item():
    rc, text = run_main(["table"])
    assert rc == 0 and "BFLA" in text and "MEASURE" in text and "NO-GO" in text and "paired SD recomputed" in text
    rc, text = run_main(["item", "F21"])
    assert rc == 0 and "verdict:  NO-GO" in text and "break-even saving" in text
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "q2.txt"
        f.write_text(Q2_ROWS)
        rc, text = run_main(["item", "F7", "--q2rows", str(f)])
        assert rc == 0 and "verdict:  GO" in text and "8 thread(s)" in text
        e = Path(d) / "expdec.txt"
        e.write_text(EXPDEC)
        rc, text = run_main(["item", "F12", "--expdec", str(e), "--t", "4"])
        assert "verdict:  GO" in text
        rc, text = run_main(["item", "F9", "--saved-ms", "0.3"])          # a measured saving under the break-even
        assert "verdict:  NO-GO" in text and "measured" in text
        rc, text = run_main(["item", "F9", "--saved-ms", "0.8", "--bar", "1.0"])
        assert "verdict:  GO" in text


def test_short_window_denominator_can_be_changed():
    c = [c for c in K.build_candidates(table()) if c.key == "F9"][0]
    base = K.evaluate(c, model())
    long_short = K.evaluate(c, K.LaneModel(bench_ms={"short": 32.6}))
    # the same 0.58 ms is a smaller share of a 32.6 ms window (kern-hip.md: "+1.8% of a short window"); edit is untouched
    assert long_short.gains["short"][0] < base.gains["short"][0] and abs(long_short.gains["short"][0] - 1.8) < 0.05
    assert long_short.gains["edit"] == base.gains["edit"]
    rc, text = run_main(["table", "--short-ms", "32.6"])
    assert rc == 0 and "short 32.6 ms" in text


def test_cli_rejects_evidence_the_item_does_not_read():
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        try:
            K.main(["item", "F7", "--expdec", "x.txt"])
            raise AssertionError("F7 does not read expdec_bench output")
        except SystemExit as e:
            assert e.code == 2
    assert "F7 does not use --expdec" in err.getvalue()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
