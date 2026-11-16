"""MOD_PLAN.md section 6 item 26: the chain-16a tooling end to end on a tiny SYNTHETIC trace, no model, no GPU, no engine.

conv_trace.py needs a running engine, so only its argument parser and the file formats it writes are checked here: a --dump-routing
file (records of [layer, k, ids x k, weights x k] int32, one record per layer per window) and a requests.json ([{end_record, conv,
domain, ...}]).  tier_laws.py --turns F.json --validate-conv then has to read both and print V1..V4.  The numbers of a synthetic trace
say nothing about the model; the test only proves the pipeline runs and that the expected structure shows (same-conversation turns overlap
more than different-conversation turns).

Run: python -m pytest -q tools/pancake/test_conv_validate_e2e.py
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import replay_cache_policy as rp  # noqa: E402
import tier_laws as tl  # noqa: E402

K = 10
ORDER = ["A", "B", "A", "C", "B", "A", "C", "B"]      # conv_trace.py's interleaved style
POOL = {"A": 0, "B": 100, "C": 200}                   # each conversation draws from its own 40 experts
WIN_PER_TURN = 6
SLOTS_PER_LAYER = 16


def sub_env():
    """conftest.py moves APPDATA to a temp dir, which hides a user-site numpy from a child process: hand it this process's sys.path."""
    return {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}


def write_fixture(d):
    rng = np.random.default_rng(7)
    rec = []
    ends = []
    for turn, conv in enumerate(ORDER):
        for _ in range(WIN_PER_TURN):
            for l in range(rp.NL):
                ids = POOL[conv] + rng.choice(40, K, replace=False)
                w = np.full(K, 1.0 / K, np.float32).view(np.int32)
                rec.append(np.concatenate([[l, K], ids, w]).astype(np.int32))
        ends.append(len(rec))
    trace = Path(d) / "conv.trace"
    np.array(rec, np.int32).tofile(trace)
    reqs = [{"end_record": e, "conv": c, "domain": {"A": "code", "B": "prose", "C": "tool"}[c], "prompt": 100, "new": 10,
             "decode_tok_s": 0.0} for e, c in zip(ends, ORDER)]
    (Path(d) / "conv.requests.json").write_text(json.dumps(reqs))
    # profile: STRP, 5 x uint32 (hdr[4] = pairs), then (layer, expert) uint16 pairs in rank order (every layer's rank 0 first)
    pairs = [(l, e) for e in range(SLOTS_PER_LAYER) for l in range(rp.NL)]
    prof = Path(d) / "profile.bin"
    with open(prof, "wb") as f:
        f.write(b"STRP")
        f.write(np.array([1, 0, 0, 0, len(pairs)], np.uint32).tobytes())
        f.write(np.array(pairs, np.uint16).tobytes())
    return trace, Path(d) / "conv.requests.json", prof, len(pairs)


def test_validate_conv_runs_end_to_end_on_a_tiny_synthetic_trace():
    with tempfile.TemporaryDirectory() as d:
        trace, reqs, prof, n_pairs = write_fixture(d)
        out = Path(d) / "out.json"
        rep = tl.main([str(prof), str(n_pairs), str(trace), "--turns", str(reqs), "--validate-conv", "--protect", "0",
                       "--policies", "tier", "conv:1", "--no-holdout", "--json", str(out)])
        v = rep["validation"]
        assert len(v) == 1
        data = json.loads(out.read_text())
        assert "validation" in data and data["slots"] == n_pairs
        v1 = v[0]
        for key in ("V1", "V2", "V3", "V4"):
            assert key in v1, key
        # structure: same-conversation adjacent turns share more experts than different-conversation ones
        same, diff = v1["V1"]["same_conv_adjacent"], v1["V1"]["other_conv_adjacent"]
        assert v1["V1"]["n_same"] > 0 and same is not None and diff is not None and same > diff
        assert set(v1["V4"]) == {"1", "2", "4", "8"}


def test_cli_prints_v1_to_v4():
    with tempfile.TemporaryDirectory() as d:
        trace, reqs, prof, n_pairs = write_fixture(d)
        r = subprocess.run([sys.executable, str(HERE / "tier_laws.py"), str(prof), str(n_pairs), str(trace), "--turns", str(reqs),
                            "--validate-conv", "--policies", "tier", "--no-holdout"], capture_output=True, text=True, timeout=300, env=sub_env())
        assert r.returncode == 0, r.stderr[-800:]
        for key in ("V1", "V2", "V3", "V4"):
            assert key in r.stdout, key


def test_conv_trace_parser_and_formats():
    import ast
    tree = ast.parse((HERE / "conv_trace.py").read_text(encoding="utf-8"))
    assert any(isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body)
    try:
        import regex  # noqa: F401  (conv_trace.py imports the tokenizer, which needs it)
        have_tok = True
    except ImportError:
        have_tok = False
    if have_tok:
        r = subprocess.run([sys.executable, str(HERE / "conv_trace.py"), "--help"], capture_output=True, text=True, timeout=120,
                           env=sub_env())
        assert r.returncode == 0 and "--domain-keys" in r.stdout
    with tempfile.TemporaryDirectory() as d:
        trace, reqs, _, _ = write_fixture(d)
        _, complete = __import__("replay_seeding").window_records(str(trace))
        assert len(complete) == len(ORDER) * WIN_PER_TURN                     # every window is complete (48 layers)
        assert json.loads(reqs.read_text())[0]["end_record"] == WIN_PER_TURN * rp.NL


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_conv_validate_e2e: ok")
