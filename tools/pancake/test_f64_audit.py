"""Tests of tools/pancake/f64_audit.py (LEADS.md Part U, lead U2, TEST_LEDGER.md T-1071): the device-code detector on fixture sources, no repository scan needed."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import f64_audit as fa  # noqa: E402

FIXTURE = r'''
// a host function: double here is not device code
double host_mean(const float* x, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; ++i) acc += x[i];
    return acc / n;
}

__device__ float helper(float v) { return expf(v); }   // f32 libm: not listed

__global__ void silu_like(float* x, int n) {
    const int i = threadIdx.x;
    if (i >= n) return;
    // double exp(-v) in a comment does not count
    const double v = (double) x[i];
    x[i] = (float) (v / (1.0 + exp(-v)));
    const char* s = "tanh( in a string";
}

template <bool F32>
__global__ void norm_like(float* p, int cols) {
    double acc = 0.0;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        acc += (double) p[i] * (double) p[i];
    }
    const double inv = __ddiv_rn(1.0, __dsqrt_rn(acc));
    p[0] = (float) tanh(inv);
}

namespace {
__device__ double block_total(double v) { return v + 1.0; }
}

#if STRATA_F16_DEVICE
static_assert(sizeof(double) == 8, "a guarded region counts as device code");
#endif
'''


def test_device_only_and_comments_and_strings_ignored():
    hits = fa.scan_text(FIXTURE, "fx.cu")
    kernels = {h["kernel"] for h in hits}
    assert "host_mean" not in kernels and "" not in kernels
    assert {"silu_like", "norm_like", "block_total"} <= kernels
    lines = {h["line"] for h in hits if h["kernel"] == "silu_like"}
    text = FIXTURE.split("\n")
    for ln in lines:
        assert "comment" not in text[ln - 1] and '"' not in text[ln - 1]
    # expf is f32: no libm hit for helper
    assert not [h for h in hits if h["kernel"] == "helper"]


def test_kinds_and_loop_flags():
    hits = fa.scan_text(FIXTURE, "fx.cu")
    sl = [h for h in hits if h["kernel"] == "silu_like"]
    assert sorted({h["kind"] for h in sl}) == ["double", "libm"]
    assert [h["what"] for h in sl if h["kind"] == "libm"] == ["exp("]
    nl = [h for h in hits if h["kernel"] == "norm_like"]
    kinds = {h["kind"] for h in nl}
    assert kinds == {"double", "libm", "rn-double"}
    loop_lines = [h for h in nl if h["loop"]]
    assert loop_lines and all("(double) p[i]" in h["text"] for h in loop_lines)
    assert not any(h["loop"] for h in sl)
    assert {h["what"] for h in nl if h["kind"] == "rn-double"} == {"__ddiv_rn", "__dsqrt_rn"}
    assert [h["what"] for h in nl if h["kind"] == "libm"] == ["tanh("]


def test_guarded_region_counts_as_device():
    hits = fa.scan_text(FIXTURE, "fx.cu")
    assert any(h["kernel"] == "(guarded region)" and h["kind"] == "double" for h in hits)


def test_summary_score_and_candidates():
    hits = fa.scan_text(FIXTURE, "src/kernels/cuda/gdn.cu")
    # pretend the file is gdn.cu: no kernel of the fixture is a candidate by name
    summ = fa.summarize(hits)
    assert summ and not any(a["u2_candidate"] for a in summ)
    a = {x["kernel"]: x for x in summ}
    assert a["norm_like"]["score"] > a["silu_like"]["score"]   # libm and loop hits weigh more
    hits2 = fa.scan_text("__global__ void gdn_l2_kernel(float* x) { double a = exp(1.0); }", "src/kernels/cuda/gdn.cu")
    assert fa.summarize(hits2)[0]["u2_candidate"]


def test_scan_paths_on_a_temp_tree(tmp_path):
    d = tmp_path / "src"
    d.mkdir()
    (d / "a.cu").write_text(FIXTURE)
    (d / "b.cpp").write_text("__global__ void ignored_ext() { double x; }")   # .cpp is not scanned
    hits = fa.scan_paths(tmp_path, ["src", "missing"])
    assert hits and all(h["file"] == "src/a.cu" for h in hits)
    text = fa.format_text(hits, fa.summarize(hits))
    assert "norm_like" in text and "silu_like" in text


def test_the_real_candidates_are_found():
    root = HERE.parent.parent
    hits = fa.scan_paths(root, ["src/kernels/cuda"])
    names = {(Path(a["file"]).name, a["kernel"]) for a in fa.summarize(hits) if a["u2_candidate"]}
    # after the U2 change the originals stay as the F32 = false branch, so the audit still names them
    assert ("elementwise.cu", "silu_kernel") in names
    assert ("gdn.cu", "gdn_l2_kernel") in names and ("gdn.cu", "gdn_out_norm_kernel") in names
    assert ("qsa.cu", "indexer_key_append_kernel") in names
    assert ("ple.cu", "gnorm_kernel") in names
