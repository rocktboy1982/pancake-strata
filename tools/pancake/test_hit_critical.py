"""Tests for probes/hit_critical.py (MOD_PLAN A36) on a synthetic rocprofv3 kernel trace.
Run: python tools/pancake/test_hit_critical.py   (or pytest)"""
import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "probes"))
import hit_critical as H  # noqa: E402
import trace_common as tc  # noqa: E402

NS = "strata::kernels::(anonymous namespace)::"


def layer(t, hit_us, wait_us, down_us=0):
    """One MoE layer: route + doorbell, a gate/up hit kernel, a down hit kernel, the CPU wait, the combine copy."""
    out = []

    def k(name, us):
        nonlocal t
        out.append((NS + name + "(int)", t, t + int(us * 1000)))
        t += int(us * 1000) + 1000   # 1 us launch gap

    k("route_multi", 5)
    k("doorbell_publish_kernel", 3)
    if hit_us:
        k("void " + NS + "expdec_gu_kernel<22, 1, 4>", hit_us)
    if down_us:
        k("void " + NS + "native_down_multi_kernel<42, true, false>", down_us)
    k("wait_flag_ge_kernel", wait_us)
    k("copy_rows_by_weight_kernel", 4)
    return out, t


def write(path, layers):
    t, rows = 1_000_000, []
    for hit_us, wait_us, down_us in layers:
        r, t = layer(t, hit_us, wait_us, down_us)
        rows += r
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Kind", "Kernel_Name", "Start_Timestamp", "End_Timestamp"])
        for n, s, e in rows:
            w.writerow(["KERNEL_DISPATCH", n, s, e])


def analyze(layers, **kw):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "run_kernel_trace.csv"
        write(p, layers)
        rows = tc.load(str(p))
    return H.analyze(rows, **kw), rows


def test_all_waiting_layers_hide_the_hits():
    # every layer waits 100 us for the CPU (floor 2 us given): the hit kernels are fully hidden
    res, _ = analyze([(80, 100, 30)] * 10, floor_ns=2000, eps_ns=2000)
    assert res["segments"] == 10 and res["gpu_bound"] == 0
    assert abs(res["hit_ms"] - 10 * 0.110) < 1e-9 and abs(res["down_ms"] - 10 * 0.030) < 1e-9
    assert res["exposed_hit_ms"] == 0 and res["ceiling_ms"] == 0
    assert "parked" in res["verdict"]


def test_gpu_bound_layers_expose_their_hits():
    # 4 of 10 layers wait only the 2 us floor: their hits (80 + 30 us) are exposed
    layers = [(80, 2, 30)] * 4 + [(80, 150, 30)] * 6
    res, rows = analyze(layers, speedup=0.5)   # floor: the 5th percentile of the waits = 2 us
    assert res["segments"] == 10 and res["gpu_bound"] == 4
    assert abs(res["floor_us"] - 2.0) < 1e-9
    assert abs(res["exposed_hit_ms"] - 4 * 0.110) < 1e-9
    assert abs(res["exposed_down_ms"] - 4 * 0.030) < 1e-9
    assert abs(res["exposed_share_of_hit"] - 0.4) < 1e-9
    assert abs(res["ceiling_ms"] - 0.5 * 4 * 0.110) < 1e-9
    wall = (rows[-1]["end"] - rows[0]["start"]) / 1e6
    assert abs(res["ceiling_pct_wall"] - 100 * res["ceiling_ms"] / wall) < 1e-6
    assert "build" in res["verdict"]   # 0.22 ms of a ~2.3 ms synthetic wall


def test_segments_need_a_route_kernel():
    # kernels before the first routing kernel are ignored; a layer without a wait kernel counts as GPU-bound
    rows = [{"name": "expdec_gu_kernel<22>", "dur": 50, "start": 0, "end": 50},
            {"name": "route_multi", "dur": 5, "start": 60, "end": 65},
            {"name": "expdec_gu_kernel<22>", "dur": 70, "start": 70, "end": 140},
            {"name": "route_multi", "dur": 5, "start": 150, "end": 155},
            {"name": "expdec_gu_kernel<22>", "dur": 30, "start": 160, "end": 190},
            {"name": "wait_flag_ge_kernel", "dur": 900, "start": 200, "end": 1100}]
    segs = H.segments(rows)
    assert [(s["h"], s["w"], s["nwait"]) for s in segs] == [(70, 0, 0), (30, 900, 1)]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
