"""Unit tests for thinking_probe.py (MOD_PLAN A19): the dump format reader against hand-packed bytes, the .ids companion,
the probe file, proxy and ground-truth labels, the replay plan, training on a synthetic signal, the folded export (the
engine's score equals the model's), the stop-rule replay and the CLI.  No server, no GPU.
Run: python -m pytest tools/pancake/test_thinking_probe.py   (or python tools/pancake/test_thinking_probe.py)"""
import json
import math
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thinking_probe as P  # noqa: E402


def test_reader_matches_hand_packed_bytes():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "hd-1234567890123-4242-5.bin"
        raw = b"SHDM" + struct.pack("<IIIQII", 1, 4, 64, 1234567890123, 777, 5)
        raw += struct.pack("<IiIf", 64, 99, 3, float("nan")) + struct.pack("<4f", 1.0, -2.0, 3.5, 0.0)
        raw += struct.pack("<IiIf", 70, 248069, 12, 0.875) + struct.pack("<4f", 0.25, 0.5, 0.75, 1.0)
        path.write_bytes(raw)
        (Path(d) / "hd-1234567890123-4242-5.ids").write_bytes(struct.pack("<3i", 11, 12, 13))
        dm = P.read_dump(path)
        assert (dm.dim, dm.every, dm.start_ms, dm.prompt_tokens, dm.seq) == (4, 64, 1234567890123, 777, 5)
        assert dm.think_index.tolist() == [64, 70] and dm.token.tolist() == [99, 248069]
        assert dm.flags.tolist() == [P.PERIODIC | P.PARAGRAPH, P.END | P.FIRED]
        assert math.isnan(dm.score[0]) and dm.score[1] == np.float32(0.875)
        assert dm.h.tolist() == [[1.0, -2.0, 3.5, 0.0], [0.25, 0.5, 0.75, 1.0]]
        assert dm.ids.tolist() == [11, 12, 13]
        assert dm.end_index == 70
        # a record cut in the middle (the engine stopped while writing) is dropped, the rest kept
        path.write_bytes(raw[:-6])
        assert len(P.read_dump(path).think_index) == 1


def test_reader_refuses_other_files():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.bin"
        p.write_bytes(b"XXXX" + bytes(28))
        for bad in (b"XXXX" + bytes(28), b"SHDM" + struct.pack("<IIIQII", 2, 4, 64, 0, 0, 0), b"SHD"):
            p.write_bytes(bad)
            try:
                P.read_dump(p)
                raise AssertionError("accepted a bad file")
            except ValueError:
                pass


def test_write_read_round_trip_and_find():
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as d:
        recs = [(64 * (i + 1), 100 + i, P.PERIODIC, float("nan"), rng.standard_normal(8)) for i in range(5)]
        P.write_dump(Path(d) / "hd-1-1-0.bin", 8, recs, ids=list(range(330)))
        (Path(d) / "other.bin").write_bytes(b"")            # not a dump name: not picked up from a directory
        found = P.find_dumps([d])
        assert [Path(f).name for f in found] == ["hd-1-1-0.bin"]
        dm = P.read_dump(found[0])
        assert np.allclose(dm.h, np.array([r[4] for r in recs], np.float32))
        assert dm.end_index is None and len(dm.ids) == 330


def test_probe_file_round_trip_and_engine_score():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "p.sprb"
        P.write_probe(p, [0.5, -1.0, 2.0], 0.25, P.LOGISTIC)
        raw = p.read_bytes()
        assert raw[:4] == b"SPRB" and struct.unpack_from("<IIIf", raw, 4) == (1, 3, 0, 0.25) and len(raw) == 32
        w, b, link = P.read_probe(p)
        h = np.array([1.0, 2.0, 0.5])
        assert abs(P.probe_score(w, b, link, h) - 1 / (1 + math.exp(0.25))) < 1e-7
        P.write_probe(p, [1, 1, 1], -1.0, P.IDENTITY)
        assert abs(P.probe_score(*P.read_probe(p), h) - 2.5) < 1e-7


def make_requests(n_req=40, dim=16, seed=0, every=16, tail=128, frac=0.9):
    """Synthetic requests: the thinking closes itself at `end`; coordinate 3 of h carries the proxy label (+1 / -1 plus
    noise) and coordinate 5 the progress through the thinking; the rest is noise."""
    rng = np.random.default_rng(seed)
    out = []
    for r in range(n_req):
        end = int(rng.integers(200, 800))
        recs = []
        for t in range(every, end, every):
            pos = (end - t <= tail) or (t >= frac * end)
            h = rng.standard_normal(dim) * 0.5
            h[3] += 1.5 if pos else -1.5
            h[5] += t / end
            recs.append((t, 1, P.PERIODIC, float("nan"), h))
        h = rng.standard_normal(dim) * 0.5
        h[3] += 1.5
        recs.append((end, 248069, P.END, float("nan"), h))
        out.append((f"hd-{r}-1-{r}.bin", recs, end))
    return out


def write_requests(d, reqs, dim=16):
    for name, recs, end in reqs:
        P.write_dump(Path(d) / name, dim, recs, ids=list(range(end)))


def test_proxy_labels():
    recs = [(t, 1, P.PERIODIC, 0.0, np.zeros(2)) for t in (64, 128, 448, 512, 576)] + [(600, 248069, P.END, 0.0, np.zeros(2))]
    with tempfile.TemporaryDirectory() as d:
        P.write_dump(Path(d) / "hd-0-0-0.bin", 2, recs)
        dm = P.read_dump(Path(d) / "hd-0-0-0.bin")
    # end 600: tail 128 -> t >= 472; frac 0.9 -> t >= 540
    assert P.proxy_labels(dm, tail=128, frac=0.9).tolist() == [0, 0, 0, 1, 1, 1]
    assert P.proxy_labels(dm, tail=0, frac=0.9).tolist() == [0, 0, 0, 0, 1, 1]
    dm.flags = dm.flags & ~np.uint32(P.END)
    assert P.proxy_labels(dm) is None                       # no natural close: left out


def test_ground_truth_labels_and_replay_plan():
    with tempfile.TemporaryDirectory() as d:
        reqs = make_requests(2, dim=8)
        write_requests(d, reqs, dim=8)
        dumps = [P.read_dump(f) for f in P.find_dumps([d])]
        lab = Path(d) / "gt.jsonl"
        lab.write_text("\n".join(json.dumps({"file": dumps[0].name, "think_index": int(t), "label": int(t > 100)})
                                 for t in dumps[0].think_index[:5]) + "\n")
        table = P.load_label_file(lab)
        fl = P.file_labels(dumps[0], table)
        assert fl[:5].tolist() == [int(t > 100) for t in dumps[0].think_index[:5]] and (fl[5:] == -1).all()
        assert (P.file_labels(dumps[1], table) == -1).all()
        plan = P.replay_plan(dumps)
        assert len(plan) == sum(len(x.think_index) - 1 for x in dumps)   # every point but the END records
        first = plan[0]
        assert first["prefix_ids"] == list(range(first["think_index"]))


def test_train_finds_the_signal_and_export_matches():
    with tempfile.TemporaryDirectory() as d:
        write_requests(d, make_requests(80))
        dumps = [P.read_dump(f) for f in P.find_dumps([d])]
        m, rep = P.train(dumps, model="logreg", l2=1.0, holdout=0.5, seed=0, k=2, risk=0.25)
        assert rep["test_requests"] == 40 and rep["train_requests"] == 40
        assert rep["auc"] > 0.95
        assert rep["ece"] < 0.15
        assert rep["recommended_threshold"] is not None
        best = [r for r in rep["stop_rule"]["sweep"] if r["threshold"] == rep["recommended_threshold"]][0]
        assert best["fires"] > 0 and best["saved_tokens"] > 0
        # the folded weights (what the engine reads) give the model's own scores on raw vectors
        p = Path(d) / "probe.sprb"
        wf, bf = m.folded()
        P.write_probe(p, wf, bf, m.link)
        w, b, link = P.read_probe(p)
        X = dumps[0].h
        assert np.allclose(P.probe_score(w, b, link, X), m.predict(X), atol=1e-4)
        # ridge: the identity link, the same signal
        mr, rr = P.train(dumps, model="ridge", l2=10.0, k=2)
        assert mr.link == P.IDENTITY and rr["auc"] > 0.95 and "ece" not in rr


def test_stop_rule_replay_matches_the_engine_rule():
    t = np.array([64, 128, 192, 256, 300])
    s = np.array([0.9, 0.4, 0.9, 0.95, 0.0])
    lab = np.array([0, 0, 1, 1, 1])
    fl = np.array([1, 1, 1, 1, P.END])
    r = P.simulate_stop([(t, s, lab, fl, 300)], 0.8, k=2)
    assert (r["fires"], r["premature"], r["saved_tokens"], r["thinking_tokens"]) == (1, 0, 300 - 256, 300)
    r = P.simulate_stop([(t, s, lab, fl, 300)], 0.8, k=1)
    assert (r["fires"], r["premature"], r["saved_tokens"]) == (1, 1, 300 - 64)
    r = P.simulate_stop([(t, s, lab, fl, 300)], 0.8, k=2, min_tokens=1000)
    assert r["fires"] == 0 and r["saved_share"] == 0.0
    assert P.auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == 0.75
    assert P.hoeffding_upper(0, 0) == 1.0


def test_cli_train_info_and_plan():
    with tempfile.TemporaryDirectory() as d:
        write_requests(d, make_requests(12))
        out, rep, plan = Path(d) / "p.sprb", Path(d) / "r.json", Path(d) / "plan.jsonl"
        assert P.main(["info", d]) == 0
        assert P.main(["train", d, "--k", "2", "--out", str(out), "--report", str(rep)]) == 0
        w, b, link = P.read_probe(out)
        assert len(w) == 16 and link == P.LOGISTIC
        assert json.loads(rep.read_text())["requests"] == 12
        assert P.main(["labels-plan", d, "--out", str(plan)]) == 0
        assert len(plan.read_text().splitlines()) > 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
