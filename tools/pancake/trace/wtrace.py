"""wtrace.py - reader for the shared window trace (MOD_PLAN item 64, include/strata/core/window_trace.hpp).

    from wtrace import read, from_dump_routing
    tr = read("session.stwt")            # Trace(n_layers, n_expert, k, windows=[Window, ...])
    for w in tr.windows:                 # Window(rows, accepted, position, ms, layers={layer: Layer})
        lay = w.layers[12]               # Layer(ids [n_tok, k] int16, kind [n_tok, k] int8, weights [n_tok, k] or None)

kind: 0 VRAM, 1 PCIe, 2 peer/remote, -1 CPU, -2 skipped tail.  accepted is None for a merged prompt window.
`from_dump_routing(path)` converts today's --dump-routing file (layer, k, ids, weights records) to the same objects
(windows split where the layer index goes back; rows/accepted/position/kind unknown -> None / CPU).
`to_replay_windows(tr)` gives the [(layer, ids[n_tok, k])...] lists tools/replay_cache_policy.py simulates.

    python tools/pancake/trace/wtrace.py FILE        summary (windows, acceptance, placement shares per layer band)
"""
import struct
import sys
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Layer:
    ids: np.ndarray
    kind: np.ndarray
    weights: np.ndarray | None


@dataclass
class Window:
    rows: int | None
    accepted: int | None
    position: int | None
    ms: float | None
    layers: dict = field(default_factory=dict)
    drafts: np.ndarray | None = None   # the drafter's probability per draft position (record 3, optional)


@dataclass
class Trace:
    n_layers: int
    n_expert: int
    k: int
    windows: list


def read(path):
    raw = open(path, "rb").read()
    if raw[:4] != b"STWT":
        raise ValueError(f"{path}: not a window trace")
    ver, nl, ne, k = struct.unpack_from("<4I", raw, 4)
    if ver != 1:
        raise ValueError(f"{path}: version {ver}")
    off, wins, cur = 20, [], None
    n = len(raw)
    while off < n:
        typ = raw[off]
        if typ == 2:
            if off + 16 > n:
                break
            _, rows, acc, _, pos, ms = struct.unpack_from("<BBBBqf", raw, off)
            off += 16
            cur = Window(rows, None if acc == 255 else acc, pos, ms)
            wins.append(cur)
        elif typ == 1:
            _, layer, nt, flags = struct.unpack_from("<BBBB", raw, off)
            off += 4
            m = nt * k
            if off + 3 * m > n:
                break
            rec = np.frombuffer(raw, dtype=np.dtype([("id", "<i2"), ("kind", "i1")]), count=m, offset=off)
            off += 3 * m
            w = None
            if flags & 1:
                if off + 4 * m > n:
                    break
                w = np.frombuffer(raw, dtype="<f4", count=m, offset=off).reshape(nt, k)
                off += 4 * m
            if cur is not None:
                cur.layers[layer] = Layer(rec["id"].reshape(nt, k).astype(np.int16), rec["kind"].reshape(nt, k), w)
        elif typ == 3:
            _, nd, _ = struct.unpack_from("<BBH", raw, off)
            off += 4
            if off + 4 * nd > n:
                break
            if cur is not None:
                cur.drafts = np.frombuffer(raw, dtype="<f4", count=nd, offset=off).copy()
            off += 4 * nd
        else:
            raise ValueError(f"{path}: bad record type {typ} at {off}")
    return Trace(nl, ne, k, wins)


def from_dump_routing(path, n_layers=48, n_expert=512):
    raw = np.fromfile(path, dtype=np.int32)
    k = int(raw[1])
    rec = 2 + 2 * k
    raw = raw[: (len(raw) // rec) * rec].reshape(-1, rec)
    layer = raw[:, 0]
    ids = raw[:, 2:2 + k]
    wts = raw[:, 2 + k:].view(np.float32)
    wins, cur, last, s = [], None, -1, 0
    chg = list(np.flatnonzero(np.diff(layer) != 0) + 1) + [len(layer)]
    for e in chg:
        l = int(layer[s])
        if cur is None or l <= last:
            cur = Window(None, None, None, None)
            wins.append(cur)
        w = wts[s:e]
        cur.layers[l] = Layer(ids[s:e].astype(np.int16), np.full((e - s, k), -1, np.int8), None if not w.any() else w.copy())
        last, s = l, e
    return Trace(n_layers, n_expert, k, [w for w in wins if len(w.layers) == n_layers])


def to_replay_windows(tr):
    return [[(l, w.layers[l].ids.astype(np.int64)) for l in sorted(w.layers)] for w in tr.windows]


def summary(tr):
    print(f"{len(tr.windows)} windows, {tr.n_layers} layers, {tr.n_expert} experts, top-{tr.k}")
    acc = [w.accepted for w in tr.windows if w.accepted is not None and w.rows]
    rows = [w.rows for w in tr.windows if w.accepted is not None and w.rows]
    if acc:
        print(f"  drafts accepted {sum(acc)} of {sum(r - 1 for r in rows)} ({sum(acc) / max(1, sum(r - 1 for r in rows)):.2f}),"
              f" mean window {np.mean([w.ms for w in tr.windows if w.ms]):.1f} ms")
    names = {0: "VRAM", 1: "PCIe", 2: "peer", -1: "CPU", -2: "skip"}
    band = max(1, tr.n_layers // 4)
    for b0 in range(0, tr.n_layers, band):
        ks = [w.layers[l].kind.ravel() for w in tr.windows for l in w.layers if b0 <= l < b0 + band]
        if not ks:
            continue
        kk = np.concatenate(ks)
        print(f"  layers {b0:2d}-{b0 + band - 1:2d}: " + "  ".join(f"{names[v]} {np.mean(kk == v):.1%}" for v in names if np.any(kk == v)))


def _selftest():
    import os
    import tempfile
    k, nl = 10, 3
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"STWT" + struct.pack("<4I", 1, nl, 512, k))
        f.write(struct.pack("<BBBBqf", 2, 2, 1, 0, 100, 25.0))
        for l in range(nl):
            f.write(struct.pack("<BBBB", 1, l, 2, 1))
            for i in range(2 * k):
                f.write(struct.pack("<hb", l * 20 + i, -1 if i % 3 else 0))
            f.write(np.arange(2 * k, dtype="<f4").tobytes())
        f.write(struct.pack("<BBBBqf", 2, 1, 255, 0, 102, 20.0))
        path = f.name
    tr = read(path)
    os.unlink(path)
    assert len(tr.windows) == 2 and tr.windows[0].accepted == 1 and tr.windows[1].accepted is None
    lay = tr.windows[0].layers[2]
    assert lay.ids.shape == (2, k) and lay.ids[0, 1] == 41 and lay.kind[0, 0] == 0 and lay.kind[0, 1] == -1
    assert lay.weights is not None and lay.weights[1, 0] == 10.0
    print("selftest ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _selftest()
    elif len(sys.argv) > 1:
        p = sys.argv[1]
        summary(read(p) if open(p, "rb").read(4) == b"STWT" else from_dump_routing(p))
    else:
        print(__doc__)
