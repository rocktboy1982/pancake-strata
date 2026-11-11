"""recording.py - what a recording session leaves behind, read raw, checked and rewritten for the offline tools.

The engine writes a window trace (STRATA_WINDOW_TRACE, include/strata/core/window_trace.hpp), the server a per-turn log
(item 73), the engine optionally a --dump-routing file and a STRATA_HIDDEN_DUMP directory.  tools/pancake/record_session.py
drives this module; it is also usable alone:

    from recording import trace_stats, export, turn_stats, clean_turns
    st = trace_stats("session.stwt")                       # counts, weights verdict, placement, turn starts
    export("session.stwt", "replay", weights="auto")       # routing.bin (legacy dump format) + session.stwt (cleaned)

Why a second reader (wtrace.py stays the one the replay tools use): the check needs the records as written.
  - a layer can have more than one record in a window (the engine's two token groups: --spec-split, or a window past
    kVerifyMaxT rows); wtrace.read keeps the last one only.  Here the records are concatenated in row order.
  - the weights of a layer record are only the router's when the doorbell published them (see the weights verdict below);
    wtrace.read hands over whatever bytes are there.
  - the file may end in the middle of a record (the engine was killed): the cut is reported, not an error.
  - record 6 (the proposed per-entry expert output norms, pancake-docs/batch4/corpus-recorder.md) is read when present.

Weights verdict of a layer record (state): ok | missing (flag bit 0 clear) | zero (all zero: the doorbell published none) |
stale (bit for bit the previous record's weights: the host buffer was not rewritten for this layer) | bad (NaN, Inf,
negative, or a row of zeros).  Only `ok` weights are the router's.
"""
import hashlib
import json
import mmap
import os
import struct
from dataclasses import dataclass, field

import numpy as np

MAGIC = b"STWT"
ENTRY = np.dtype([("id", "<i2"), ("kind", "i1")])
KIND_NAMES = {0: "VRAM", 1: "PCIe", 2: "peer", -1: "CPU", -2: "skip"}
GAP_MS = 1500.0            # a window this long after its predecessor starts a new turn (decode windows are ~30-60 ms)


@dataclass
class Rec:
    layer: int
    ids: np.ndarray                 # [n_tok, k] int16
    kind: np.ndarray                # [n_tok, k] int8
    weights: np.ndarray | None = None   # [n_tok, k] float32
    norms: np.ndarray | None = None     # [n_tok, k] float32 (record 6)


@dataclass
class RawWindow:
    rows: int
    accepted: int | None
    position: int
    ms: float
    recs: list = field(default_factory=list)
    drafts: np.ndarray | None = None
    draft_src: np.ndarray | None = None
    row_miss: np.ndarray | None = None


@dataclass
class Header:
    n_layers: int
    n_expert: int
    k: int


# ------------------------------------------------------------------------------------------------ window trace: read

def iter_windows(path, state=None):
    """Yield the RawWindow of a window trace in file order.  `state` receives "header" (Header) and, when the file ends in
    the middle of a record, "tail_bytes" (the bytes after the last whole record).  The arrays view the file's mmap."""
    state = {} if state is None else state
    with open(path, "rb") as f:
        n = os.fstat(f.fileno()).st_size
        if n < 20:
            raise ValueError(f"{path}: {n} bytes, not a window trace")
        raw = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    if raw[:4] != MAGIC:
        raise ValueError(f"{path}: not a window trace (no STWT magic)")
    ver, nl, ne, k = struct.unpack_from("<4I", raw, 4)
    if ver != 1:
        raise ValueError(f"{path}: window trace version {ver}")
    state["header"] = Header(nl, ne, k)
    state["tail_bytes"] = 0
    off, cur = 20, None

    def cut(at):
        state["tail_bytes"] = n - at

    while off < n:
        typ = raw[off]
        if typ == 2:
            if off + 16 > n:
                cut(off)
                break
            _, rows, acc, _, pos, ms = struct.unpack_from("<BBBBqf", raw, off)
            if cur is not None:
                yield cur
            cur = RawWindow(rows, None if acc == 255 else acc, pos, ms)
            off += 16
        elif typ == 1:
            _, layer, nt, flags = struct.unpack_from("<BBBB", raw, off)
            m = nt * k
            if off + 4 + 3 * m + (4 * m if flags & 1 else 0) > n:
                cut(off)
                break
            off += 4
            rec = np.frombuffer(raw, dtype=ENTRY, count=m, offset=off)
            off += 3 * m
            w = None
            if flags & 1:
                w = np.frombuffer(raw, dtype="<f4", count=m, offset=off).reshape(nt, k)
                off += 4 * m
            if cur is not None:
                cur.recs.append(Rec(layer, rec["id"].reshape(nt, k), rec["kind"].reshape(nt, k), w))
        elif typ in (3, 4, 5, 6):
            if off + 4 > n:
                cut(off)
                break
            _, a, b, _ = struct.unpack_from("<BBBB", raw, off)
            size = {3: 4 * a, 4: 5 * a, 5: 2 * a, 6: 4 * b * k}[typ]
            if off + 4 + size > n:
                cut(off)
                break
            body = off + 4
            if cur is not None:
                if typ == 4:
                    cur.drafts = np.frombuffer(raw, dtype="<f4", count=a, offset=body)
                    cur.draft_src = np.frombuffer(raw, dtype="u1", count=a, offset=body + 4 * a)
                elif typ == 3:
                    cur.drafts = np.frombuffer(raw, dtype="<f4", count=a, offset=body)
                elif typ == 5:
                    cur.row_miss = np.frombuffer(raw, dtype="<u2", count=a, offset=body)
                else:   # record 6: the expert output norms of the layer record just before it
                    for r in reversed(cur.recs):
                        if r.layer == a and r.norms is None:
                            r.norms = np.frombuffer(raw, dtype="<f4", count=b * k, offset=body).reshape(b, k)
                            break
            off = body + size
        else:
            raise ValueError(f"{path}: bad record type {typ} at {off}")
    if cur is not None:
        yield cur


def merged(win):
    """{layer: Rec}: the records of a layer concatenated in file order (token group 0 first = row order)."""
    out = {}
    for r in win.recs:
        o = out.get(r.layer)
        if o is None:
            out[r.layer] = r
            continue

        def cat(a, b):
            return None if a is None or b is None else np.concatenate([a, b])
        out[r.layer] = Rec(r.layer, np.concatenate([o.ids, r.ids]), np.concatenate([o.kind, r.kind]),
                           cat(o.weights, r.weights), cat(o.norms, r.norms))
    return out


# ------------------------------------------------------------------------------------------------ weights verdict

class WeightCheck:
    """State of each layer record's weights, in file order (the previous record is what `stale` compares with)."""

    def __init__(self):
        self.prev = None

    def state(self, w):
        prev, self.prev = self.prev, w
        if w is None:
            return "missing"
        if not w.any():
            return "zero"
        if not np.isfinite(w).all() or (w < 0).any() or not w.any(axis=1).all():
            return "bad"
        if prev is not None and prev.shape == w.shape and np.array_equal(prev, w):
            return "stale"
        return "ok"


def turn_start_flags(wins, gap_ms=GAP_MS):
    """wins: iterable of (position, ms, rows, accepted).  A window starts a turn when it is the first, when its position
    does not continue the previous window's commit (disco_fit.py's rule), or when it came gap_ms after it (idle time and
    the prompt read are in the window's ms)."""
    out, prev = [], None
    for pos, ms, rows, acc in wins:
        follows = prev is not None and pos == prev[0] + (prev[1] if prev[2] is None else prev[2] + 1)
        out.append(prev is None or not follows or ms > gap_ms)
        prev = (pos, rows, acc)
    return out


# ------------------------------------------------------------------------------------------------ window trace: check

def trace_stats(path, gap_ms=GAP_MS):
    """One pass over a window trace.  JSON-able numbers; keys starting with "_" are per-window arrays for export()."""
    state = {}
    chk = WeightCheck()
    hdr = None
    nW = nDecode = nMerged = nComplete = nGrouped = 0
    rows_dec, ms_all, ms_dec = [], [], []
    offered = accepted = 0
    place = np.zeros(5, np.int64)
    bad_range = dup_rows = n_entries = 0
    w_states = {s: 0 for s in ("ok", "missing", "zero", "stale", "bad")}
    layer_ok, layer_n = {}, {}
    sum1_rows = sum1_ok = 0
    norm_recs = norm_fin = norm_n = 0
    drafts_w = rowmiss_w = 0
    flags, meta = [], []
    for w in iter_windows(path, state):
        hdr = state["header"]
        nW += 1
        decode = w.accepted is not None
        nDecode += decode
        nMerged += not decode
        layers = {r.layer for r in w.recs}
        complete = len(layers) == hdr.n_layers and all(l < hdr.n_layers for l in layers)
        nComplete += complete
        nGrouped += len(w.recs) > len(layers)
        drafts_w += w.drafts is not None
        rowmiss_w += w.row_miss is not None
        ms_all.append(w.ms)
        if decode:
            rows_dec.append(w.rows)
            ms_dec.append(w.ms)
            offered += max(w.rows - 1, 0)
            accepted += min(w.accepted, max(w.rows - 1, 0))
        win_ok = bool(w.recs)
        for r in w.recs:
            s = chk.state(r.weights)
            w_states[s] += 1
            layer_n[r.layer] = layer_n.get(r.layer, 0) + 1
            layer_ok[r.layer] = layer_ok.get(r.layer, 0) + (s == "ok")
            win_ok &= s == "ok"
            if s == "ok":
                sum1_rows += r.weights.shape[0]
                sum1_ok += int((np.abs(r.weights.sum(axis=1) - 1.0) < 1e-3).sum())
            place += np.bincount(r.kind.ravel().astype(np.int64) + 2, minlength=5)[:5]
            ids = r.ids.astype(np.int64)
            n_entries += ids.size
            bad_range += int(((ids < 0) | (ids >= hdr.n_expert)).sum())
            srt = np.sort(ids, axis=1)
            dup_rows += int((srt[:, 1:] == srt[:, :-1]).any(axis=1).sum()) if hdr.k > 1 else 0
            if r.norms is not None:
                norm_recs += 1
                norm_n += r.norms.size
                norm_fin += int(np.isfinite(r.norms).sum())
        flags.append((1 if complete else 0) | (2 if decode else 0) | (4 if win_ok else 0))
        meta.append((w.position, w.ms, w.rows, w.accepted))
    if hdr is None:
        hdr = state["header"]
    starts = turn_start_flags(meta, gap_ms)
    n_rec = sum(w_states.values())
    served = int(place.sum() - place[0])   # entries not skipped (index 0 = kind -2)
    pct = lambda x, n: round(float(x) / n, 4) if n else None   # noqa: E731
    ms_dec_a = np.array(ms_dec, float) if ms_dec else np.zeros(0)
    st = {
        "header": {"n_layers": hdr.n_layers, "n_expert": hdr.n_expert, "k": hdr.k},
        "bytes": os.path.getsize(path), "tail_bytes": state["tail_bytes"],
        "windows": nW, "decode_windows": nDecode, "merged_prompt_windows": nMerged,
        "complete_windows": nComplete, "incomplete_windows": nW - nComplete, "grouped_windows": nGrouped,
        "rows": {"mean": round(float(np.mean(rows_dec)), 2) if rows_dec else None, "max": int(max(rows_dec)) if rows_dec else None},
        "ms": {"median": round(float(np.median(ms_dec_a)), 2) if len(ms_dec_a) else None,
               "p99": round(float(np.percentile(ms_dec_a, 99)), 2) if len(ms_dec_a) else None,
               "first_window": round(ms_all[0], 1) if ms_all else None, "over_gap": int((np.array(ms_all) > gap_ms).sum()) if ms_all else 0},
        "drafts": {"offered": offered, "accepted": accepted, "rate": pct(accepted, offered),
                   "windows_with_confidences": drafts_w, "windows_with_row_miss": rowmiss_w},
        "placement": {KIND_NAMES[i - 2]: pct(place[i], int(place.sum())) for i in range(5)} | {"hit_rate": pct(place[2], served)},
        "ids": {"entries": n_entries, "out_of_range": bad_range, "rows_with_duplicate_expert": dup_rows},
        "weights": {"records": n_rec, **w_states, "ok_share": pct(w_states["ok"], n_rec),
                    "rows_sum_to_one_share": pct(sum1_ok, sum1_rows),
                    "layers_not_ok": sorted(l for l in layer_n if layer_ok[l] < layer_n[l]),
                    "windows_all_ok": int(sum(1 for f in flags if f & 4))},
        "norms": {"records": norm_recs, "finite_share": pct(norm_fin, norm_n)},
        "turns": {"gap_ms": gap_ms, "count": int(sum(starts)), "first_windows": [i for i, s in enumerate(starts) if s][:20]},
        "_flags": flags, "_turn_start": starts,
    }
    return st


def public(st):
    """The JSON-able part of trace_stats()."""
    return {k: v for k, v in st.items() if not k.startswith("_")}


# ------------------------------------------------------------------------------------------------ window trace: rewrite

def stwt_header(h):
    return MAGIC + struct.pack("<4I", 1, h.n_layers, h.n_expert, h.k)


def stwt_window(w, recs):
    """One window as the engine writes it (record 2, 5, then 4 and the layers in ascending order)."""
    b = bytearray(struct.pack("<BBBBqf", 2, min(w.rows, 255), 255 if w.accepted is None else min(w.accepted, 254), 0, w.position, w.ms))
    if w.row_miss is not None and len(w.row_miss):
        b += struct.pack("<BBH", 5, len(w.row_miss), 0) + np.asarray(w.row_miss, "<u2").tobytes()
    if w.drafts is not None and len(w.drafts):
        n = len(w.drafts)
        src = w.draft_src if w.draft_src is not None else np.zeros(n, "u1")
        b += struct.pack("<BBH", 4, n, 0) + np.asarray(w.drafts, "<f4").tobytes() + np.asarray(src, "u1").tobytes()
    for r in recs:
        nt = r.ids.shape[0]
        b += struct.pack("<BBBB", 1, r.layer, nt, 1 if r.weights is not None else 0)
        e = np.empty(r.ids.size, ENTRY)
        e["id"], e["kind"] = r.ids.ravel(), r.kind.ravel()
        b += e.tobytes()
        if r.weights is not None:
            b += np.asarray(r.weights, "<f4").tobytes()
        if r.norms is not None:
            b += struct.pack("<BBBB", 6, r.layer, nt, 0) + np.asarray(r.norms, "<f4").tobytes()
    return bytes(b)


def dump_records(layer, ids, weights):
    """--dump-routing records of one layer: per token int32 layer, int32 k, k int32 ids, k float32 weights."""
    n, k = ids.shape
    rec = np.zeros((n, 2 + 2 * k), np.int32)
    rec[:, 0], rec[:, 1], rec[:, 2:2 + k] = layer, k, ids
    if weights is not None:
        rec[:, 2 + k:] = np.ascontiguousarray(weights, np.float32).view(np.int32)
    return rec.tobytes()


def export(path, out_dir, weights="auto", decode_only=True, stats=None):
    """Rewrite a window trace for the replay tools into `out_dir`:
       routing.bin  the --dump-routing format (time_ceiling.py, finemoe_replay.py, cache_learn.py, reap_saliency.py,
                    gate_importance.py --dump-routing ...): complete windows, token rows in order, decode windows only unless
                    decode_only is False; weights as `weights` says (a record without them is written with zeros, which
                    the readers take as "no weights")
       session.stwt the same windows as a window trace (every complete window, merged prompt windows too) with the layer
                    records merged and weights per `weights`: for the tools that read placement, drafts and ms
       act_norm.npy [n_layers, n_expert] mean expert output norm (record 6; only when the trace has them)
    weights: "auto" keep the weights only if every exported window has all-ok records (all or nothing, as reap_saliency
             treats a trace), "valid" keep each record that is ok and drop the others, "none" drop all.
    Returns a dict (windows written, weights mode used, cuts for finemoe_replay --cuts, the number of dropped windows)."""
    st = stats or trace_stats(path)
    h = Header(**st["header"])
    os.makedirs(out_dir, exist_ok=True)
    flags = st["_flags"]
    exported = [bool(f & 1) and (not decode_only or bool(f & 2)) for f in flags]
    bad = sum(1 for f, e in zip(flags, exported) if e and not f & 4)
    used = weights
    if weights == "auto":
        used = "valid" if exported.count(True) and bad == 0 else "none"
    reason = None
    if weights == "auto" and used == "none":
        reason = f"{bad} of {exported.count(True)} exported windows have a layer record without router weights"
    bin_ok = h.n_layers == 48 and h.n_expert == 512
    chk = WeightCheck()
    sums = np.zeros((h.n_layers, h.n_expert)), np.zeros((h.n_layers, h.n_expert))
    wins_bin = wins_stwt = 0
    cuts, pending, starts = [], False, st["_turn_start"]
    with open(os.path.join(out_dir, "session.stwt"), "wb") as fs, open(os.path.join(out_dir, "routing.bin"), "wb") as fb:
        fs.write(stwt_header(h))
        for i, w in enumerate(iter_windows(path)):
            ok_by_rec = {}
            for r in w.recs:   # the stale chain sees every record, exported or not
                ok_by_rec[id(r)] = chk.state(r.weights) == "ok"
            if not flags[i] & 1:
                pending |= starts[i]
                continue
            keep_w = used == "valid"
            recs, norm_recs = [], []
            for layer, m in sorted(merged(w).items()):
                good = keep_w and m.weights is not None and all(ok_by_rec[id(r)] for r in w.recs if r.layer == layer)
                recs.append(Rec(layer, m.ids, m.kind, m.weights if good else None))   # no record 6: wtrace.read refuses it
                norm_recs.append(m)
            fs.write(stwt_window(w, recs))
            wins_stwt += 1
            if exported[i]:
                if bin_ok:
                    for r in recs:
                        fb.write(dump_records(r.layer, r.ids.astype(np.int32), r.weights))
                if (pending or starts[i]) and wins_bin > 0:
                    cuts.append(wins_bin)
                pending = False
                wins_bin += 1
            else:
                pending |= starts[i]
            for r in norm_recs:
                if r.norms is not None:
                    ok = np.isfinite(r.norms)
                    ids = r.ids.astype(np.int64)
                    np.add.at(sums[0][r.layer], ids[ok], r.norms[ok].astype(np.float64))
                    np.add.at(sums[1][r.layer], ids[ok], 1.0)
    info = {"windows_routing_bin": wins_bin if bin_ok else 0, "windows_stwt": wins_stwt, "weights": used,
            "weights_requested": weights, "weights_reason": reason, "decode_only": decode_only, "cuts": cuts,
            "exported_windows_without_weights": bad}
    if not bin_ok:
        os.remove(os.path.join(out_dir, "routing.bin"))
        info["routing_bin_skipped"] = (f"the replay tools are written for 48 layers x 512 experts, this trace has "
                                       f"{h.n_layers} x {h.n_expert}")
    if st["norms"]["records"]:
        tot, cnt = sums
        seen = cnt > 0
        mean = np.where(seen, tot / np.maximum(cnt, 1), 0.0)
        for l in range(h.n_layers):   # an expert whose entries never ran on the host: its layer's mean (coverage in info)
            if seen[l].any():
                mean[l, ~seen[l]] = mean[l, seen[l]].mean()
        np.save(os.path.join(out_dir, "act_norm.npy"), mean.astype(np.float32))
        info["act_norm_coverage"] = round(float(seen.mean()), 4)
    return info


# ------------------------------------------------------------------------------------------------ the other recorders

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_turns(path):
    """(valid, malformed): [(time, line, record)] sorted by time of the log's usable lines (replay_turns.load_turns' rule: a
    JSON object with prompt_total and time), and the count of the others."""
    rows, bad = [], 0
    with open(path, errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if isinstance(r, dict) and r.get("prompt_total") is not None and r.get("time") is not None:
                rows.append((float(r["time"]), line.rstrip("\n"), r))
            else:
                bad += 1
    rows.sort(key=lambda t: t[0])
    return rows, bad


def turn_stats(path, gap_s=600.0):
    rows, bad = read_turns(path)
    recs = [r for _, _, r in rows]
    sess = 0
    for i, r in enumerate(recs):
        if i == 0 or r["time"] - (recs[i - 1]["time"] + (recs[i - 1].get("duration_s") or 0)) > gap_s:
            sess += 1
    tot = sum(int(r["prompt_total"]) for r in recs)
    reused = sum(int(r.get("reused") or 0) for r in recs)
    return {"bytes": os.path.getsize(path), "turns": len(recs), "malformed_lines": bad, "sessions": sess,
            "span_s": round(recs[-1]["time"] + (recs[-1].get("duration_s") or 0) - recs[0]["time"], 1) if recs else 0,
            "prompt_tokens": tot, "reused_share": round(reused / tot, 4) if tot else None,
            "output_tokens": sum(int(r.get("output_tokens") or 0) for r in recs),
            "generating": sum(1 for r in recs if (r.get("engine_generated") or r.get("output_tokens") or 0) > 0),
            "thinking": sum(1 for r in recs if r.get("effort") or (r.get("reasoning_tokens") or 0) > 0),
            "max_prompt": max((int(r["prompt_total"]) for r in recs), default=0),
            "with_agent_counters": sum(1 for r in recs if r.get("agent"))}


def clean_turns(src, dst):
    """The usable lines of a turn log, sorted by time, to `dst` (the file replay_turns.py --turns reads)."""
    rows, bad = read_turns(src)
    with open(dst, "w", encoding="utf-8") as f:
        for _, line, _ in rows:
            f.write(line + "\n")
    return {"turns": len(rows), "dropped_lines": bad}


def ints(path):
    """A --dump-routing file as int32 (a file cut in the middle of a word loses its last bytes)."""
    return np.memmap(path, dtype=np.int32, mode="r", shape=(os.path.getsize(path) // 4,))


def routing_stats(path, n_layers=48):
    """A --dump-routing file: records, k, windows (a window starts where the layer does not rise), weights (the serve path
    writes zeros on purpose), a record cut in two at the end."""
    size = os.path.getsize(path)
    if size < 8:
        return {"bytes": size, "records": 0, "windows": 0}
    raw = ints(path)
    k = int(raw[1])
    rec = 2 + 2 * k
    n = len(raw) // rec
    m = raw[: n * rec].reshape(n, rec)
    layer = np.array(m[:, 0])
    run_start = np.concatenate([[0], np.flatnonzero(np.diff(layer) != 0) + 1])
    rl = layer[run_start]
    win_start = np.concatenate([[True], rl[1:] <= rl[:-1]]) if len(rl) else np.zeros(0, bool)
    per_win = np.diff(np.concatenate([np.flatnonzero(win_start), [len(rl)]])) if len(rl) else np.zeros(0, int)
    return {"bytes": size, "k": k, "records": int(n), "tail_bytes": int(size - 4 * n * rec),
            "k_consistent": bool((np.asarray(m[:, 1]) == k).all()), "windows": int(win_start.sum()),
            "complete_windows": int((per_win == n_layers).sum()),
            "nonzero_weight_records": sum(int(np.asarray(m[a:a + (1 << 20), 2 + k:]).any(axis=1).sum())
                                          for a in range(0, n, 1 << 20))}


def hidden_stats(directory):
    """STRATA_HIDDEN_DUMP's directory: hd-*.bin files (32-byte header "SHDM", then 16 + 4 * dim byte records)."""
    files = sorted(f for f in os.listdir(directory) if f.startswith("hd-") and f.endswith(".bin"))
    recs = bad = 0
    dims = set()
    for f in files:
        p = os.path.join(directory, f)
        with open(p, "rb") as fh:
            head = fh.read(32)
        if len(head) < 32 or head[:4] != b"SHDM":
            bad += 1
            continue
        dim = struct.unpack_from("<I", head, 8)[0]
        dims.add(dim)
        body = os.path.getsize(p) - 32
        recs += body // (16 + 4 * dim)
        bad += body % (16 + 4 * dim) != 0
    return {"files": len(files), "records": recs, "bad_files": bad, "dims": sorted(dims),
            "id_files": sum(1 for f in files if os.path.exists(os.path.join(directory, f[:-4] + ".ids")))}


def compare_ids(trace_path, dump_path, n=50):
    """The first `n` windows of the trace against the dump: the same rows and ids in the same order?  False also when the
    two were not written for the same windows (STRATA_WINDOW_TRACE_EVERY above 1, a window dropped before its commit)."""
    raw = ints(dump_path)
    k = int(raw[1])
    rec = 2 + 2 * k
    m = raw[: (len(raw) // rec) * rec].reshape(-1, rec)
    pos = 0
    for i, w in enumerate(iter_windows(trace_path)):
        if i >= n:
            break
        for layer, r in sorted(merged(w).items()):
            rows = r.ids.shape[0]
            got = np.asarray(m[pos:pos + rows])
            if len(got) < rows or (got[:, 0] != layer).any() or not np.array_equal(got[:, 2:2 + k], r.ids.astype(np.int32)):
                return False
            pos += rows
    return True
