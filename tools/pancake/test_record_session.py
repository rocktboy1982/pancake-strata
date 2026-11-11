"""record_session.py and trace/recording.py on synthetic recordings.  No model, no GPU, no engine is started.

The recordings are written here byte for byte as the engine writes them (window_trace.hpp, think_probe.hpp, server.py's turn
log), not with the module's own writer, so a reader that drifts from the engine's layout fails.

Run: python -m pytest tools/pancake/test_record_session.py   (or python tools/pancake/test_record_session.py)"""
import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "trace"))
sys.path.insert(0, str(HERE.parent))
import cache_learn  # noqa: E402,F401  (finemoe_replay's import chain)
import finemoe_replay  # noqa: E402
import gate_importance  # noqa: E402
import reap_saliency  # noqa: E402
import record_session as rs  # noqa: E402
import recording as rec  # noqa: E402
import replay_cache_policy as rp  # noqa: E402
import replay_turns  # noqa: E402
import wtrace  # noqa: E402

NL, NE, K = 48, 512, 10


# ------------------------------------------------------------------------------------------------ the engine's bytes

def layer_rec(layer, ids, kind, weights=None, norms=None):
    nt = ids.shape[0]
    b = struct.pack("<BBBB", 1, layer, nt, 0 if weights is None else 1)
    for i, kd in zip(ids.ravel(), kind.ravel()):
        b += struct.pack("<hb", int(i), int(kd))
    if weights is not None:
        b += np.asarray(weights, "<f4").tobytes()
    if norms is not None:   # the proposed record 6, written right after its layer record
        b += struct.pack("<BBBB", 6, layer, nt, 0) + np.asarray(norms, "<f4").tobytes()
    return b


def window_bytes(rows, accepted, position, ms, layers, drafts=None, row_miss=None):
    """layers: list of (layer, ids, kind, weights|None[, norms]) in file order."""
    b = struct.pack("<BBBBqf", 2, rows, 255 if accepted is None else accepted, 0, position, ms)
    if row_miss is not None:
        b += struct.pack("<BBH", 5, len(row_miss), 0) + np.asarray(row_miss, "<u2").tobytes()
    if drafts is not None:
        b += struct.pack("<BBH", 4, len(drafts), 0) + np.asarray(drafts, "<f4").tobytes() + bytes(len(drafts))
    for lr in layers:
        b += layer_rec(*lr)
    return b


def routed(rng, rows):
    """A router's top-k per row: distinct ids, weights renormalised over the k (sum 1)."""
    ids = np.stack([rng.choice(NE, K, replace=False) for _ in range(rows)]).astype(np.int16)
    w = rng.random((rows, K)).astype(np.float32) + 0.05
    return ids, w / w.sum(axis=1, keepdims=True)


def session(rng, n_windows=30, turn_every=10, weights="real", merged_first=True, norms=False, bad_windows=()):
    """(file bytes, list of window dicts).  Windows 0, turn_every, ... start a turn: the position jumps and ms is large; the
    first window of a turn is a merged prompt window when merged_first.  weights: real | zero | stale | none."""
    out = [b"STWT" + struct.pack("<4I", 1, NL, NE, K)]
    wins, pos, prev_w = [], 100, None
    for i in range(n_windows):
        starts = i % turn_every == 0
        merged = starts and merged_first
        rows = int(rng.integers(2, 7)) if not merged else int(rng.integers(20, 30))
        acc = None if merged else int(rng.integers(0, rows))
        if starts:
            pos += 500
        layers = []
        for l in range(NL):
            ids, w = routed(rng, rows)
            kind = rng.choice([0, 0, 0, 0, 1, -1], (rows, K)).astype(np.int8)
            if weights == "zero" or (i in bad_windows):
                w = np.zeros_like(w)
            elif weights == "none":
                w = None
            elif weights == "stale" and prev_w is not None and prev_w.shape == w.shape:
                w = prev_w
            if w is not None:
                prev_w = w
            n = np.abs(rng.normal(3.0, 1.0, (rows, K))).astype(np.float32) if norms else None
            layers.append((l, ids, kind, w, n) if norms else (l, ids, kind, w))
        d = None if merged else rng.random(rows - 1).astype(np.float32)
        out.append(window_bytes(min(rows, 255), acc, pos, 2000.0 if starts else 30.0 + i % 7, layers, d,
                                None if merged else [3] * rows))
        wins.append(dict(rows=rows, accepted=acc, position=pos, layers=layers, merged=merged))
        pos += rows if merged else acc + 1
    return b"".join(out), wins


def write(path, data):
    Path(path).write_bytes(data)
    return str(path)


def tmp():
    return Path(tempfile.mkdtemp())


# ------------------------------------------------------------------------------------------------ the reader

def test_reader_agrees_with_wtrace():
    data, _ = session(np.random.default_rng(0), 12)
    p = write(tmp() / "s.stwt", data)
    ref = wtrace.read(p)
    got = list(rec.iter_windows(p))
    assert len(got) == len(ref.windows) == 12
    for g, r in zip(got, ref.windows):
        assert (g.rows, g.accepted, g.position, g.ms) == (r.rows, r.accepted, r.position, r.ms)
        assert (g.drafts is None) == (r.drafts is None) and (g.row_miss is None) == (r.row_miss is None)
        if g.drafts is not None:
            assert np.array_equal(g.drafts, r.drafts) and list(g.row_miss) == list(r.row_miss)
        for layer, m in rec.merged(g).items():
            assert np.array_equal(m.ids, r.layers[layer].ids) and np.array_equal(m.kind, r.layers[layer].kind)
            assert np.array_equal(m.weights, r.layers[layer].weights)


def test_token_groups_are_concatenated_in_row_order():
    rng = np.random.default_rng(1)
    ids_a, w_a = routed(rng, 3)
    ids_b, w_b = routed(rng, 2)
    kd = lambda n: np.zeros((n, K), np.int8)   # noqa: E731
    layers = []
    for l in range(NL):   # the order the doorbell rings: layer l group 0, layer l group 1, layer l+1 group 0, ...
        layers += [(l, ids_a, kd(3), w_a), (l, ids_b, kd(2), w_b)]
    p = write(tmp() / "g.stwt", b"STWT" + struct.pack("<4I", 1, NL, NE, K) + window_bytes(5, 4, 7, 40.0, layers))
    w, = list(rec.iter_windows(p))
    m = rec.merged(w)
    assert len(w.recs) == 2 * NL and len(m) == NL
    assert np.array_equal(m[5].ids, np.concatenate([ids_a, ids_b])) and np.array_equal(m[5].weights, np.concatenate([w_a, w_b]))
    st = rec.trace_stats(p)
    assert st["grouped_windows"] == 1 and st["complete_windows"] == 1 and st["weights"]["ok"] == 2 * NL


def test_file_cut_in_a_record_is_reported_not_raised():
    data, _ = session(np.random.default_rng(2), 6)
    p = write(tmp() / "c.stwt", data[:-5000])
    state = {}
    wins = list(rec.iter_windows(p, state))
    assert state["tail_bytes"] > 0 and len(wins) == 6
    st = rec.trace_stats(p)
    assert st["tail_bytes"] > 0 and st["incomplete_windows"] == 1 and st["windows"] == 6
    with pytest.raises(ValueError):
        rec.trace_stats(write(tmp() / "x.stwt", b"NOPE" + bytes(40)))
    with pytest.raises(ValueError):
        list(rec.iter_windows(write(tmp() / "y.stwt", b"STWT" + struct.pack("<4I", 1, NL, NE, K) + bytes([9, 0, 0, 0]))))


# ------------------------------------------------------------------------------------------------ the weights verdict

@pytest.mark.parametrize("mode,state", [("real", "ok"), ("zero", "zero"), ("none", "missing")])
def test_weights_verdict_per_record(mode, state):
    data, _ = session(np.random.default_rng(3), 4, weights=mode)
    st = rec.trace_stats(write(tmp() / "w.stwt", data))
    w = st["weights"]
    assert w["records"] == 4 * NL and w[state] == 4 * NL and w["ok_share"] == (1.0 if mode == "real" else 0.0)
    assert (w["layers_not_ok"] == []) == (mode == "real")
    if mode == "real":
        assert w["rows_sum_to_one_share"] == 1.0 and w["windows_all_ok"] == 4


def test_stale_weights_are_the_previous_records():
    """A host buffer the doorbell did not rewrite holds the previous layer's weights: bit for bit equal."""
    data, _ = session(np.random.default_rng(4), 3, weights="stale")
    w = rec.trace_stats(write(tmp() / "s.stwt", data))["weights"]
    assert w["stale"] > 0 and w["ok"] < w["records"]
    assert w["ok"] + w["stale"] == w["records"] or w["ok"] + w["stale"] + w["missing"] <= w["records"]


def test_bad_weights_nan_negative_zero_row():
    rng = np.random.default_rng(5)
    chk = rec.WeightCheck()
    ok = np.full((2, K), 0.1, np.float32)
    assert chk.state(ok) == "ok"
    for bad in (np.where(np.arange(K) == 0, np.nan, 0.1), np.where(np.arange(K) == 0, -0.1, 0.1)):
        assert chk.state(np.tile(bad.astype(np.float32), (2, 1))) == "bad"
    half = ok.copy()
    half[1] = 0
    assert chk.state(half) == "bad" and chk.state(None) == "missing" and chk.state(np.zeros((2, K), np.float32)) == "zero"
    assert chk.state(ok) == "ok" and chk.state(ok.copy()) == "stale"
    assert rng is not None


# ------------------------------------------------------------------------------------------------ numbers

def test_stats_numbers_and_turn_starts():
    data, wins = session(np.random.default_rng(6), 30, turn_every=10)
    st = rec.trace_stats(write(tmp() / "n.stwt", data))
    assert st["windows"] == 30 and st["merged_prompt_windows"] == 3 and st["decode_windows"] == 27
    assert st["complete_windows"] == 30 and st["incomplete_windows"] == 0
    assert st["turns"]["count"] == 3 and st["turns"]["first_windows"] == [0, 10, 20]
    dec = [w for w in wins if not w["merged"]]
    assert st["drafts"]["offered"] == sum(w["rows"] - 1 for w in dec)
    assert st["drafts"]["accepted"] == sum(w["accepted"] for w in dec)
    kinds = np.concatenate([lr[2].ravel() for w in wins for lr in w["layers"]])
    assert abs(st["placement"]["hit_rate"] - float(np.mean(kinds == 0))) < 1e-3   # no skipped entries here
    assert st["ids"]["out_of_range"] == 0 and st["ids"]["rows_with_duplicate_expert"] == 0
    assert st["header"] == {"n_layers": NL, "n_expert": NE, "k": K}


def test_a_gap_alone_starts_a_turn_and_a_position_jump_alone_too():
    f = rec.turn_start_flags
    # (position, ms, rows, accepted): 100 +3 -> 103 follows; 103 + 1 -> 104 follows
    assert f([(100, 30, 4, 2), (103, 30, 3, 0), (104, 31, 2, 1)]) == [True, False, False]
    assert f([(100, 30, 4, 2), (103, 5000, 3, 0)]) == [True, True]          # same position chain, long gap
    assert f([(100, 30, 4, 2), (900, 30, 3, 0)]) == [True, True]            # a new request: the position jumped
    assert f([(100, 30, 6, None), (106, 30, 3, 0)]) == [True, False]        # a merged window advances by its rows


# ------------------------------------------------------------------------------------------------ convert

def exported(weights="auto", n=30, **kw):
    d = tmp()
    data, wins = session(np.random.default_rng(7), n, **kw)
    p = write(d / "s.stwt", data)
    return d, p, wins, rec.export(p, d / "replay", weights)


def test_export_routing_bin_reads_back_in_the_replay_tools():
    d, p, wins, info = exported()
    dec = [w for w in wins if not w["merged"]]
    assert info["windows_routing_bin"] == len(dec) == 27 and info["weights"] == "valid" and info["cuts"] == [9, 18]
    b = str(d / "replay" / "routing.bin")
    windows, k = rp.load_trace(b)
    assert k == K and len(windows) == 27
    for w, ref in zip(windows, dec):
        assert [l for l, _ in w] == list(range(NL))
        for (l, ids), lr in zip(w, ref["layers"]):
            assert np.array_equal(ids, lr[1])
    tr = wtrace.from_dump_routing(b)
    assert len(tr.windows) == 27
    w0 = tr.windows[0]
    assert np.array_equal(w0.layers[3].weights, dec[0]["layers"][3][3])   # the weights survive (the serve dump has zeros)
    ref_all = reap_saliency.stats(reap_saliency.load(b))
    assert ref_all[0].shape == (NL, NE) and ref_all[1] is not None            # weighted: REAP, not the frequency baseline
    maps = finemoe_replay.load(b)
    assert len(maps[0]) == 27 and maps[2] == K and maps[1][0][0] is not None


def test_export_session_stwt_is_a_cleaned_window_trace():
    d, p, wins, info = exported()
    tr = wtrace.read(str(d / "replay" / "session.stwt"))
    assert len(tr.windows) == 30 == info["windows_stwt"]
    for w, ref in zip(tr.windows, wins):
        assert (w.rows, w.accepted, w.position) == (min(ref["rows"], 255), ref["accepted"], ref["position"])
        assert np.array_equal(w.layers[7].weights, ref["layers"][7][3]) and np.array_equal(w.layers[7].kind, ref["layers"][7][2])
    assert tr.windows[1].drafts is not None and list(tr.windows[1].row_miss) == [3] * tr.windows[1].rows
    all_w = rec.export(p, d / "all", "auto", decode_only=False)
    assert all_w["windows_routing_bin"] == 30 and all_w["cuts"] == [10, 20]


def test_export_merges_token_groups():
    rng = np.random.default_rng(8)
    ids_a, w_a = routed(rng, 3)
    ids_b, w_b = routed(rng, 2)
    kd = lambda n: np.full((n, K), -1, np.int8)   # noqa: E731
    layers = []
    for l in range(NL):
        layers += [(l, ids_a, kd(3), w_a), (l, ids_b, kd(2), w_b)]
    d = tmp()
    p = write(d / "g.stwt", b"STWT" + struct.pack("<4I", 1, NL, NE, K) + window_bytes(5, 4, 7, 40.0, layers))
    info = rec.export(p, d / "r", "auto")
    tr = wtrace.read(str(d / "r" / "session.stwt"))
    assert tr.windows[0].layers[9].ids.shape == (5, K) and info["windows_routing_bin"] == 1
    windows, _ = rp.load_trace(str(d / "r" / "routing.bin"))
    assert windows[0][9][1].shape == (5, K)


def test_weights_modes_zero_trace_and_one_bad_window():
    d = tmp()
    data, _ = session(np.random.default_rng(9), 30, weights="zero")
    p = write(d / "z.stwt", data)
    info = rec.export(p, d / "r", "auto")
    assert info["weights"] == "none" and "exported windows have a layer record without router weights" in info["weights_reason"]
    assert wtrace.read(str(d / "r" / "session.stwt")).windows[2].layers[0].weights is None
    assert not np.fromfile(str(d / "r" / "routing.bin"), np.int32).reshape(-1, 22)[:, 12:].any()
    # one decode window with zero weights: auto drops all, valid keeps the others and only that window loses them
    data, _ = session(np.random.default_rng(9), 30, bad_windows=(5,))
    p = write(d / "b.stwt", data)
    auto = rec.export(p, d / "ba", "auto")
    val = rec.export(p, d / "bv", "valid")
    assert auto["weights"] == "none" and val["weights"] == "valid" and val["exported_windows_without_weights"] == 1
    tr = wtrace.read(str(d / "bv" / "session.stwt"))
    assert tr.windows[5].layers[2].weights is None and tr.windows[6].layers[2].weights is not None
    assert rec.export(p, d / "bn", "none")["weights"] == "none"


def test_export_without_48x512_has_no_routing_bin():
    nl, ne, k = 4, 64, 3
    rng = np.random.default_rng(10)
    b = b"STWT" + struct.pack("<4I", 1, nl, ne, k)
    for i in range(3):
        layers = []
        for l in range(nl):
            ids = np.stack([rng.choice(ne, k, replace=False) for _ in range(2)]).astype(np.int16)
            w = np.full((2, k), 1 / k, np.float32) + l * 0.001
            layers.append((l, ids, np.zeros((2, k), np.int8), w))
        b += window_bytes(2, 1, 10 + 2 * i, 30.0, layers)
    d = tmp()
    info = rec.export(write(d / "m.stwt", b), d / "r")
    assert "routing_bin_skipped" in info and not (d / "r" / "routing.bin").exists() and info["windows_stwt"] == 3


def test_act_norm_from_record_6():
    d, p, wins, info = exported(norms=True)
    assert info["act_norm_coverage"] > 0
    m = np.load(d / "replay" / "act_norm.npy")
    assert m.shape == (NL, NE) and m.dtype == np.float32 and np.isfinite(m).all()
    # the mean of the measured norms of one (layer, expert), by hand
    l, e = 4, int(wins[1]["layers"][4][1][0, 0])
    vals = [lr[4][r, j] for w in wins for lr in [w["layers"][l]] for r in range(lr[1].shape[0]) for j in range(K) if lr[1][r, j] == e]
    assert abs(m[l, e] - np.mean(vals)) < 1e-4
    assert rec.trace_stats(p)["norms"] == {"records": 30 * NL, "finite_share": 1.0}
    score, what = reap_saliency.saliency(*reap_saliency.stats(reap_saliency.load(str(d / "replay" / "session.stwt"))), act_norm=m)
    assert "measured mean output norm" in what   # and the cleaned stwt has no record 6, which wtrace.read refuses


# ------------------------------------------------------------------------------------------------ the other recorders

def turn_line(t, total=1000, reused=0, **kw):
    return json.dumps({"id": f"r{t:.0f}", "time": t, "duration_s": 5.0, "prompt_total": total, "reused": reused, "output_tokens": 40,
                       "engine_generated": 40, "reasoning_tokens": 0, "effort": None, **kw})


def test_turn_log_stats_clean_and_replay_turns_agree():
    d = tmp()
    lines = [turn_line(1000), "not json", turn_line(1010, 2000, 1000), json.dumps({"time": 1020}), turn_line(900, 500),
             json.dumps({"prompt_total": 5}), turn_line(5000, 3000, 0, effort="high", agent={"edits": 1}), ""]
    src = d / "turns.jsonl"
    src.write_text("\n".join(lines) + "\n")
    st = rec.turn_stats(str(src), gap_s=600)
    assert st["turns"] == 4 and st["malformed_lines"] == 3 and st["sessions"] == 2 and st["thinking"] == 1
    assert st["prompt_tokens"] == 500 + 1000 + 2000 + 3000 and st["with_agent_counters"] == 1 and st["max_prompt"] == 3000
    assert st["reused_share"] == round(1000 / 6500, 4)
    info = rec.clean_turns(str(src), str(d / "clean.jsonl"))
    assert info == {"turns": 4, "dropped_lines": 3}
    ref = replay_turns.load_turns(str(src))
    got = replay_turns.load_turns(str(d / "clean.jsonl"))
    assert [r["id"] for r in got] == [r["id"] for r in ref] == ["r900", "r1000", "r1010", "r5000"]
    assert len(replay_turns.split_sessions(got, 600)) == st["sessions"]


def dump_bytes(windows):
    """--dump-routing: per token and layer int32 layer, int32 k, k int32 ids, k float32 weights (zeros: the serve path)."""
    out = b""
    for w in windows:
        for lr in w["layers"]:
            for r in range(lr[1].shape[0]):
                out += np.array([lr[0], K], np.int32).tobytes() + lr[1][r].astype(np.int32).tobytes() + bytes(4 * K)
    return out


def test_routing_dump_stats_and_id_comparison():
    d = tmp()
    data, wins = session(np.random.default_rng(11), 12)
    tp = write(d / "s.stwt", data)
    bp = write(d / "routing.bin", dump_bytes(wins))
    rt = rec.routing_stats(bp)
    assert rt["windows"] == 12 and rt["complete_windows"] == 12 and rt["k"] == K and rt["tail_bytes"] == 0
    assert rt["nonzero_weight_records"] == 0 and rt["k_consistent"]
    assert rec.compare_ids(tp, bp) is True
    other, _ = session(np.random.default_rng(12), 12)
    assert rec.compare_ids(write(d / "o.stwt", other), bp) is False
    cut = write(d / "cut.bin", dump_bytes(wins)[:-30])
    assert rec.routing_stats(cut)["tail_bytes"] == (len(dump_bytes(wins)) - 30) % 88


def test_hidden_dump_directory():
    d = tmp() / "hidden"
    d.mkdir()
    dim = 6
    head = b"SHDM" + struct.pack("<IIIQII", 1, dim, 64, 1700000000000, 120, 1)
    rec_b = struct.pack("<IiIf", 3, 7, 1, float("nan")) + np.zeros(dim, "<f4").tobytes()
    (d / "hd-1-2-3.bin").write_bytes(head + rec_b * 3)
    (d / "hd-1-2-3.ids").write_bytes(bytes(12))
    (d / "hd-1-2-4.bin").write_bytes(head + rec_b + b"xx")        # a cut record
    (d / "other.txt").write_text("x")
    assert rec.hidden_stats(str(d)) == {"files": 2, "records": 4, "bad_files": 1, "dims": [dim], "id_files": 1}


# ------------------------------------------------------------------------------------------------ plan

def fake_repo(env=None, exe="build-hip/strata", args=None):
    repo = tmp() / "repo"
    (repo / "configs/pancake").mkdir(parents=True)
    (repo / "build-hip").mkdir()
    (repo / "build-hip/strata").write_text("")
    (repo / "build-hip/strata-poly").write_text("")
    cfg = {"exe": f"<REPO>/{exe}", "args": args if args is not None else ["--pack", "<STRATA_DATA>/packs/x", "--spec", "4"],
           "cwd": "<REPO>", "log": "<REPO>/eng.log", "turn_log": "<REPO>/pancake-turns.jsonl", "port": 8080, "gpu": 0,
           "env": {"STRATA_VERIFY_DEFER_CPU": "1", "STRATA_ROUTE_TAIL_SKIP": "0"} if env is None else env}
    (repo / "configs/pancake/p.json").write_text(json.dumps(cfg))
    return repo


def plan_args(repo, out, *extra):
    ap_args = ["plan", "--preset", "p", "--out", str(out), "--repo", str(repo), "--data", str(repo / "data"), "--models", str(repo / "m")]
    return ap_args + list(extra)


def test_plan_builds_the_config_and_prints_the_command(monkeypatch, capsys):
    repo = fake_repo()
    called = []
    monkeypatch.setattr(rs.subprocess, "run", lambda *a, **k: called.append(a) or type("R", (), {"stdout": "abc1234\n"})())
    out = repo.parent / "rec1"
    assert rs.main(plan_args(repo, out, "--port", "8123", "--hidden-dump", "32")) == 0
    text = capsys.readouterr().out
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["env"]["STRATA_WINDOW_TRACE"] == str(out / "session.stwt") and cfg["env"]["STRATA_WINDOW_TRACE_MAX_MB"] == "4096"
    assert cfg["env"]["STRATA_HIDDEN_DUMP"] == f"{out / 'hidden'},every=32" and cfg["env"]["STRATA_VERIFY_DEFER_CPU"] == "1"
    assert "STRATA_WINDOW_TRACE_EVERY" not in cfg["env"] and "STRATA_WINDOW_TRACE_NORMS" not in cfg["env"]
    assert cfg["args"][-2:] == ["--dump-routing", str(out / "routing.bin")] and cfg["args"][:2] == ["--pack", f"{repo}/data/packs/x"]
    assert cfg["log"] == str(out / "engine.log") and cfg["turn_log"] == str(out / "turns.jsonl") and cfg["port"] == 8123
    assert cfg["exe"] == f"{repo}/build-hip/strata" and "<" not in json.dumps(cfg)
    plan = json.loads((out / "plan.json").read_text())
    assert plan["preset"] == "p" and plan["args_added"] == ["--dump-routing", str(out / "routing.bin")]
    assert plan["server_env"]["STRATA_RAM_KEEP_GIB"] == "12" and plan["server_env"]["STRATA_AGENT_COUNTERS"] == "1"
    assert "HIP_VISIBLE_DEVICES" in plan["server_env"]
    line = plan["command"]
    assert "tools/pancake/capped.sh" in line and "serve/server.py --engine strata" in line and f"--config {out}/config.json" in line
    assert "--port 8123" in line and "nothing was started" in text and line in text
    assert not [c for c in called if c[0] and c[0][0] != "git"], "plan without --run must not start anything"
    # the original preset is untouched (deep copy)
    assert "STRATA_WINDOW_TRACE" not in json.loads((repo / "configs/pancake/p.json").read_text())["env"]


def test_plan_options_and_refusals(monkeypatch, capsys):
    monkeypatch.setattr(rs.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": ""})())
    repo = fake_repo(env={"STRATA_ROUTE_TAIL_SKIP": "0"})
    out = repo.parent / "r2"
    assert rs.main(plan_args(repo, out, "--every", "2", "--max-mb", "100", "--no-routing-dump", "--no-agent-counters", "--norms",
                             "--publish-weights")) == 0
    cfg = json.loads((out / "config.json").read_text())
    plan = json.loads((out / "plan.json").read_text())
    assert cfg["env"]["STRATA_WINDOW_TRACE_EVERY"] == "2" and cfg["env"]["STRATA_WINDOW_TRACE_MAX_MB"] == "100"
    assert cfg["env"]["STRATA_WINDOW_TRACE_NORMS"] == "1" and cfg["env"]["STRATA_VERIFY_DEFER_CPU"] == "1"
    assert "--dump-routing" not in cfg["args"] and "STRATA_AGENT_COUNTERS" not in plan["server_env"]
    assert "--norms: STRATA_WINDOW_TRACE_NORMS is read only by an engine with the record-6 patch" in capsys.readouterr().out
    # a directory that holds a recording is never reused (the turn log appends)
    (out / "turns.jsonl").write_text("x")
    with pytest.raises(SystemExit) as e:
        rs.main(plan_args(repo, out))
    assert "turns.jsonl" in str(e.value)
    # the weights note: no deferral -> a WARN; with it -> ok
    rs.main(plan_args(repo, repo.parent / "r3"))
    assert "router weights: this preset's doorbell publishes none" in capsys.readouterr().out
    # the engine variants that change the routing stop the plan; strata-poly and --no-pool too
    for env, exe, args, why in (({"STRATA_ROUTE_PRIOR": "0.5"}, "build-hip/strata", None, "STRATA_ROUTE_PRIOR=0.5"),
                                ({"STRATA_ROUTE_TAIL_SKIP": "2"}, "build-hip/strata", None, "STRATA_ROUTE_TAIL_SKIP=2"),
                                ({}, "build-hip/strata-poly", None, "strata-poly"),
                                ({}, "build-hip/strata", ["--no-pool"], "--no-pool")):
        r = fake_repo(env=env, exe=exe, args=args)
        assert rs.main(plan_args(r, r.parent / "rx")) == 2
        err = capsys.readouterr().err
        assert "not started" in err and why in err
    r = fake_repo(env={"STRATA_ROUTE_PRIOR": "0.5"})
    assert rs.main(plan_args(r, r.parent / "ry", "--allow-biased-routing")) == 0
    assert "cache-biased router's" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        rs.main(plan_args(repo, repo.parent / "rz")[:2] + ["nosuchpreset"] + plan_args(repo, repo.parent / "rz")[3:])


def test_plan_run_starts_the_server_once_then_checks(monkeypatch, capsys):
    repo = fake_repo()
    calls = []

    def fake_run(cmd, **kw):
        if cmd[0] == "git":
            return type("R", (), {"stdout": "abc\n"})()
        calls.append((cmd, kw))
        out = Path(cmd[cmd.index("--config") + 1]).parent   # the "session": the recorders' files appear
        data, wins = session(np.random.default_rng(13), 20)
        (out / "session.stwt").write_bytes(data)
        (out / "turns.jsonl").write_text("\n".join(turn_line(1000 + 10 * i) for i in range(6)) + "\n")
        return type("R", (), {"returncode": 0})()
    monkeypatch.setattr(rs.subprocess, "run", fake_run)
    out = repo.parent / "run1"
    code = rs.main(plan_args(repo, out, "--run", "--min-windows", "5", "--no-routing-dump"))
    assert len(calls) == 1 and calls[0][0][0].endswith("capped.sh") and calls[0][1]["cwd"] == repo
    assert calls[0][1]["env"]["STRATA_RAM_KEEP_GIB"] == "12"
    text = capsys.readouterr().out
    assert "== running" in text and "== check" in text and (out / "manifest.json").exists() and code == 0


# ------------------------------------------------------------------------------------------------ check / convert

def recording(weights="real", n=30, turns=8, routing=True, log=True, plan=True, **kw):
    d = tmp() / "rec"
    d.mkdir()
    data, wins = session(np.random.default_rng(14), n, weights=weights, **kw)
    (d / "session.stwt").write_bytes(data)
    base = 1_700_000_000.0
    (d / "turns.jsonl").write_text("\n".join(turn_line(base + 20 * i, 1000 + 500 * i, 800 * (i > 0), engine_generated=30) for i in range(turns)) + "\n")
    if routing:
        (d / "routing.bin").write_bytes(dump_bytes(wins))
    if log:
        (d / "engine.log").write_text("starting\nstrata: STRATA_WINDOW_TRACE: verify windows -> x (every 1, cap 4096 MiB)\n")
    if plan:
        (d / "plan.json").write_text(json.dumps({"name": "t", "preset": "p", "git": "abc", "max_mb": 4096}))
    return d, wins


def test_check_a_good_recording(capsys):
    d, _ = recording(norms=True)
    assert rs.main(["check", str(d), "--min-windows", "20", "--min-turns", "5", "--gap-ms", "1500"]) == 0
    m = json.loads((d / "manifest.json").read_text())
    assert m["verdict"] == "ok" and m["problems"] == [] and all(m["usable_for"].values()), m["warnings"]
    assert m["window_trace"]["decode_windows"] == 27 and m["window_trace"]["weights"]["ok_share"] == 1.0
    assert m["routing_dump"]["ids_match_trace"] is True and m["turn_log"]["turns"] == 8
    assert len(m["files"]["session.stwt"]["sha256"]) == 64 and m["files"]["routing.bin"]["bytes"] > 0
    assert m["name"] == "t" and m["git"] == "abc" and "_flags" not in json.dumps(m)
    assert "== check" in capsys.readouterr().out


def test_check_zero_weights_is_usable_but_not_for_weights(capsys):
    d, _ = recording(weights="zero")
    assert rs.main(["check", str(d), "--min-windows", "20"]) == 0
    m = json.loads((d / "manifest.json").read_text())
    assert m["verdict"] == "usable, not for everything" and m["problems"] == []
    assert m["usable_for"]["ids (time_ceiling, finemoe_replay, cache_learn, replay_seeding, selftune)"]
    assert not m["usable_for"]["router weights (reap_saliency gate part, gate_importance, finemoe maps)"]
    assert any("router weights: 0 of" in w and "zero" in w for w in m["warnings"])
    assert rs.main(["check", str(d), "--min-windows", "20", "--require-weights"]) == 2


def test_check_problems(capsys):
    d, _ = recording()
    assert rs.main(["check", str(d)]) == 2                       # 27 decode windows < the default 200
    assert "fewer than --min-windows 200" in capsys.readouterr().out
    d, _ = recording(turns=2)
    assert rs.main(["check", str(d), "--min-windows", "5"]) == 2
    assert "turns in the turn log" in capsys.readouterr().out
    d, _ = recording()
    (d / "turns.jsonl").unlink()
    assert rs.main(["check", str(d), "--min-windows", "5"]) == 2
    d, _ = recording()
    (d / "session.stwt").write_bytes(b"")
    assert rs.main(["check", str(d)]) == 2 and "missing or empty" in capsys.readouterr().out
    d, _ = recording()
    (d / "engine.log").write_text("strata: STRATA_WINDOW_TRACE: cannot write /nonexistent/x.stwt\n")
    assert rs.main(["check", str(d), "--min-windows", "5"]) == 2


def test_check_warnings(capsys):
    d, _ = recording(routing=False, log=False)
    rs.main(["check", str(d), "--min-windows", "5", "--json"])
    m = json.loads(capsys.readouterr().out)
    assert any("routing.bin: absent" in w for w in m["warnings"]) and any("engine.log: absent" in w for w in m["warnings"])
    d, _ = recording()
    (d / "engine.log").write_text("starting\n")
    rs.main(["check", str(d), "--min-windows", "5", "--json"])
    assert any("no 'STRATA_WINDOW_TRACE: verify windows" in w for w in json.loads(capsys.readouterr().out)["warnings"])
    d, _ = recording()
    data = (d / "session.stwt").read_bytes()
    (d / "session.stwt").write_bytes(data[:-3000])
    rs.main(["check", str(d), "--min-windows", "5", "--json"])
    m = json.loads(capsys.readouterr().out)
    assert any("ends in the middle of a record" in w for w in m["warnings"])
    assert not any("routing.bin holds" in w for w in m["warnings"])        # the cut window is still a window
    d, _ = recording()
    (d / "engine.log").write_text("STRATA_WINDOW_TRACE: verify windows -> x\nstrata: STRATA_WINDOW_TRACE reached its cap (4096 MiB): no more windows written\n")
    rs.main(["check", str(d), "--min-windows", "5", "--json"])
    assert any("reached its cap" in w for w in json.loads(capsys.readouterr().out)["warnings"])


def test_convert_cli_writes_the_replay_inputs_and_prints_commands(capsys):
    d, _ = recording(norms=True)
    assert rs.main(["convert", str(d)]) == 0
    text = capsys.readouterr().out
    r = d / "replay"
    for name in ("routing.bin", "session.stwt", "turns.jsonl", "act_norm.npy", "convert.json"):
        assert (r / name).exists(), name
    info = json.loads((r / "convert.json").read_text())
    assert info["weights"] == "valid" and info["cuts"] == [9, 18] and info["turns"]["turns"] == 8
    assert f"time_ceiling.py {r}/routing.bin data/expert-profile-pancake.bin 10909 --validate" in text
    assert f"finemoe_replay.py data/expert-profile-pancake.bin 10909 {r}/routing.bin --cuts 9,18" in text
    assert f"reap_saliency.py {r}/session.stwt" in text and f"--act-norm {r}/act_norm.npy" in text
    assert f"replay_turns.py SPEC.json --out DIR --turns {r}/turns.jsonl" in text
    d2, _ = recording(weights="zero")
    assert rs.main(["convert", str(d2), "--weights", "auto", "--slots", "5000"]) == 0
    text = capsys.readouterr().out
    assert "weights: none (" in text and " 5000 " in text and "act-norm" not in text


def test_the_real_tools_run_on_the_converted_files():
    d, _ = recording()
    rs.main(["convert", str(d)])
    r = d / "replay"
    count, gate = reap_saliency.stats(reap_saliency.load(str(r / "session.stwt")))
    assert gate is not None and count.sum() == sum(w.rows * K * NL for w in wtrace.read(str(r / "session.stwt")).windows)
    tr = wtrace.read(str(r / "session.stwt"))
    ws, = gate_importance.sort_rows(np.concatenate([w.layers[0].weights for w in tr.windows]))
    assert np.allclose(ws.sum(axis=1), 1.0, atol=1e-5)
    assert len(finemoe_replay.load(str(r / "routing.bin"))[0]) == 27


def test_session_without_a_turn_start_marks_one_turn_and_empty_export_is_clean():
    d = tmp()
    data, _ = session(np.random.default_rng(15), 6, turn_every=100, merged_first=False)
    p = write(d / "one.stwt", data)
    st = rec.trace_stats(p)
    assert st["turns"]["count"] == 1 and st["merged_prompt_windows"] == 0
    assert rec.export(p, d / "r")["cuts"] == []
    only_header = write(d / "h.stwt", b"STWT" + struct.pack("<4I", 1, NL, NE, K))
    st = rec.trace_stats(only_header)
    assert st["windows"] == 0 and st["weights"]["ok_share"] is None
    info = rec.export(only_header, d / "e")
    assert info["windows_routing_bin"] == 0 and info["weights"] == "none"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
