"""wtrace.py - reader for the shared window trace (MOD_PLAN item 64, include/strata/core/window_trace.hpp).

    from wtrace import read, from_dump_routing
    tr = read("session.stwt")            # Trace(n_layers, n_expert, k, windows=[Window, ...])
    for w in tr.windows:                 # Window(rows, accepted, position, ms, layers={layer: Layer})
        lay = w.layers[12]               # Layer(ids [n_tok, k] int16, kind [n_tok, k] int8, weights [n_tok, k] or None)

kind: 0 VRAM, 1 PCIe, 2 peer/remote, -1 CPU, -2 skipped tail.  accepted is None for a merged prompt window.
Window.row_miss (record 5, A22): the CPU-missed experts each row added over the rows before it, as the engine counted them
(None in traces written before A22); `row_new_miss(w)` gives it, or new_cpu_miss(w) from the layer records when absent.
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
    drafts: np.ndarray | None = None   # prob per draft row 1..T-1 (float32, NaN: lookup / chained row, no confidence)
    draft_src: np.ndarray | None = None   # source per draft row (uint8): 0 MTP, 1 DFlash2, 2 lookup window, 3 chained lookup; None in a legacy record 3
    row_miss: np.ndarray | None = None    # record 5 (A22): new CPU-missed experts per row 0..T-1 (uint16), None before A22
    target_p: np.ndarray | None = None    # record 6 (A56): the target's probability of each drafted token (f32), None unless STRATA_WINDOW_TRACE_TARGET=1


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
        elif typ == 4:   # drafts, written when the window is built: n f32 probs, then n u8 sources
            _, nd, _ = struct.unpack_from("<BBH", raw, off)
            off += 4
            if off + 5 * nd > n:
                break
            if cur is not None:
                cur.drafts = np.frombuffer(raw, dtype="<f4", count=nd, offset=off).copy()
                cur.draft_src = np.frombuffer(raw, dtype="u1", count=nd, offset=off + 4 * nd).copy()
            off += 5 * nd
        elif typ == 5:   # row misses (A22): n u16, the new CPU misses of rows 0..n-1
            _, nr, _ = struct.unpack_from("<BBH", raw, off)
            off += 4
            if off + 2 * nr > n:
                break
            if cur is not None:
                cur.row_miss = np.frombuffer(raw, dtype="<u2", count=nr, offset=off).astype(np.int64)
            off += 2 * nr
        elif typ == 6:   # target probabilities (A56): n f32, the target's p of window[j] for draft row j = 1..n
            _, nt, _ = struct.unpack_from("<BBH", raw, off)
            off += 4
            if off + 4 * nt > n:
                break
            if cur is not None:
                cur.target_p = np.frombuffer(raw, dtype="<f4", count=nt, offset=off).copy()
            off += 4 * nt
        elif typ == 3:   # legacy: n f32 probs only
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


def new_cpu_miss(w):
    """CPU-missed experts (kind -1) each row of window `w` adds over the rows before it, summed over layers (distinct
    expert per layer).  Index 0 is the committed row, index j draft row j.  Exact: the verifier records the placement of
    every entry, so the per-row count is read back from the layer records (A22 DISCO prices a draft row with it)."""
    n = int(w.rows or 0)
    out = np.zeros(max(n, 1), dtype=np.int64)
    for lay in w.layers.values():
        seen = set()
        for r in range(min(n, lay.ids.shape[0])):
            cpu = set(lay.ids[r][lay.kind[r] == -1].tolist())
            out[r] += len(cpu - seen)
            seen |= cpu
    return out


def row_new_miss(w):
    """The new CPU misses per row of window `w`: the engine's record 5 when the trace has it, else new_cpu_miss(w)."""
    if w.row_miss is not None and len(w.row_miss) > 0:
        return w.row_miss
    return new_cpu_miss(w)


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
        nd = sum(1 for w in tr.windows if w.drafts is not None)
        if nd:
            print(f"  {nd} windows carry draft confidences (see tail_replay.py)")
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
        f.write(struct.pack("<BBH", 5, 2, 0) + np.array([14, 6], "<u2").tobytes())
        f.write(struct.pack("<BBBB", 4, 1, 0, 0) + np.array([0.75], "<f4").tobytes() + bytes([0]))
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
    d = tr.windows[0].drafts
    assert d is not None and d.shape == (1,) and d[0] == 0.75 and tr.windows[0].draft_src[0] == 0 and tr.windows[1].drafts is None
    nm = new_cpu_miss(tr.windows[0])   # ids are distinct per entry, so every CPU-kind entry (i % 3 != 0) is a new miss
    assert nm.shape == (2,) and nm.sum() == sum(1 for _ in range(nl) for i in range(2 * k) if i % 3), nm
    assert tr.windows[0].row_miss is not None and list(tr.windows[0].row_miss) == [14, 6] and tr.windows[1].row_miss is None
    assert list(row_new_miss(tr.windows[0])) == [14, 6]
    assert list(row_new_miss(Window(2, 1, 0, 1.0, {0: Layer(np.array([[1, 2], [2, 3]], np.int16), np.full((2, 2), -1, np.int8), None)}))) == [2, 1]
    print("selftest ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _selftest()
    elif len(sys.argv) > 1:
        p = sys.argv[1]
        summary(read(p) if open(p, "rb").read(4) == b"STWT" else from_dump_routing(p))
    else:
        print(__doc__)
