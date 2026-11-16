"""Tests for fit_lookahead.py on a SYNTHETIC trace: a known random teacher routes the rows, the fit must recover it.
No engine trace is involved; the layout constants are the ones src/program/lookahead_weights_test.cpp pins."""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_lookahead as fl  # noqa: E402

H, E, K, NL = 16, 8, 2, 3


def make_trace(path, windows=200, ntok=4, seed=1, teachers=None):
    rng = np.random.default_rng(seed)
    if teachers is None:
        teachers = {L: rng.normal(size=(E, H)).astype(np.float32) * 1.5 for L in range(1, NL)}
    with open(path, "wb") as f:
        for w in range(windows):
            xs = {L: rng.normal(size=(ntok, H)).astype(np.float32) for L in range(NL)}
            for L in range(NL):
                if L == 0:
                    ids = rng.integers(0, E, size=(ntok, K))
                else:   # layer L routes by a teacher applied to the PREVIOUS layer's input (what the fit must learn)
                    ids = np.argsort(-(fl.from_bf16(fl.to_bf16(xs[L - 1])) @ teachers[L].T), axis=1)[:, :K]
                resmap = np.zeros(E, bool)
                resmap[:3] = True
                res = resmap[ids]
                fl.write_record(f, L, w, xs[L], ids, res, resmap, E)
    return teachers


def test_layout_constants_match_the_cpp_test():
    assert fl.LKW_MAGIC == 0x31574B4C and fl.RPD_MAGIC == 0x32445250
    assert fl.record_bytes(3, 2, 10, 8) == 28 + 3 * 8 * 2 + 3 * 2 * 2 + 3 * 2 + 2   # pred_record_bytes in the C++ test


def test_bf16_rounding():
    assert fl.to_bf16(np.array([1.0, -2.0, 0.0], np.float32)).tolist() == [0x3F80, 0xC000, 0]
    assert fl.to_bf16(np.array([1.0 + 1 / 256], np.float32))[0] == 0x3F80       # tie, even: down
    assert fl.to_bf16(np.array([1.0 + 3 / 256], np.float32))[0] == 0x3F82       # tie, even: up
    assert np.isnan(fl.from_bf16(fl.to_bf16(np.array([np.nan], np.float32)))[0])


def test_trace_roundtrip_and_pairing(tmp_path):
    p = str(tmp_path / "trace.bin")
    make_trace(p, windows=5)
    recs = list(fl.read_trace(p, H))
    assert len(recs) == 5 * NL and recs[0][0] == 0 and recs[1][0] == 1
    assert recs[1][2].shape == (4, H) and recs[1][3].shape == (4, K) and recs[1][5].sum() == 3
    pairs = fl.collect_pairs(p, H, NL)
    assert sorted(pairs) == [1, 2] and len(pairs[1]) == 5
    # the pair for layer 2 uses layer 1's input rows
    assert np.array_equal(pairs[2][0][1], recs[1][2])


def test_a_cut_last_record_is_ignored(tmp_path):
    p = str(tmp_path / "trace.bin")
    make_trace(p, windows=3)
    with open(p, "ab") as f:
        f.write(struct.pack("<7I", fl.RPD_MAGIC, 0, 4, K, 9, 0, E) + b"\0" * 10)
    assert len(list(fl.read_trace(p, H))) == 3 * NL


def test_wrong_n_embd_is_reported(tmp_path):
    p = str(tmp_path / "trace.bin")
    make_trace(p, windows=3)
    try:
        list(fl.read_trace(p, H + 8))
        assert False, "must fail"
    except ValueError as e:
        assert "bad record magic" in str(e)


def test_fit_recovers_a_teacher_and_beats_popularity(tmp_path):
    p = str(tmp_path / "trace.bin")
    out = str(tmp_path / "pred.lkw")
    make_trace(p, windows=400)
    rep = fl.run(p, H, NL, E, out, epochs=25, lr=2e-2, holdout=0.25, budget=3, min_pairs=32, quiet=True)
    for L in (1, 2):
        assert rep[L]["fitted"] and rep[L]["overlap"] > 0.9, rep[L]
        assert rep[L]["overlap"] > rep[L]["baseline_overlap"] + 0.3, rep[L]
        assert rep[L]["nonresident_recall"] is not None and rep[L]["nonresident_recall"] > 0.9, rep[L]
    # the file: size, header, layer 0 zeros, and the exported weights reproduce the in-script predictions
    assert os.path.getsize(out) == 16 + NL * (E * H * 2 + E * 4)
    nl, ne, nh, layers = fl.read_lkw(out)
    assert (nl, ne, nh) == (NL, E, H)
    assert not layers[0][0].any() and not layers[0][1].any()
    pairs = fl.collect_pairs(p, H, NL)
    X, Y = pairs[1][-1][1], pairs[1][-1][2]
    W, b = layers[1]
    pred = np.argsort(-(X @ W.T + b), axis=1)[:, :K]
    assert np.mean([np.isin(Y[t], pred[t]).mean() for t in range(len(X))]) > 0.85   # bf16 weights keep the picks


def test_layer_with_too_few_pairs_gets_bias_only(tmp_path):
    p = str(tmp_path / "trace.bin")
    out = str(tmp_path / "pred.lkw")
    make_trace(p, windows=3)
    rep = fl.run(p, H, NL, E, out, min_pairs=1000, quiet=True)
    assert not rep[1]["fitted"]
    _, _, _, layers = fl.read_lkw(out)
    assert not layers[1][0].any() and layers[1][1].any()


def test_cli(tmp_path, capsys):
    p = str(tmp_path / "trace.bin")
    make_trace(p, windows=80)
    out = str(tmp_path / "o.lkw")
    assert fl.main([p, "--n-embd", str(H), "--n-layers", str(NL), "--n-expert", str(E), "--out", out, "--epochs", "2", "--min-pairs", "8",
                    "--json", str(tmp_path / "r.json")]) == 0
    assert "wrote" in capsys.readouterr().out and os.path.exists(str(tmp_path / "r.json"))
