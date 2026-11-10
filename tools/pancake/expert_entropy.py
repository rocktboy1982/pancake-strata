#!/usr/bin/env python3
"""expert_entropy.py - offline entropy check of the routed-expert bytes of a GGUF (MOD_PLAN A29).

Question: are the Q2_0 / IQ2_XS expert bytes compressible enough to justify an entropy-coded or compressed
expert path (EntroLLM / Huff-LLM line: move fewer bytes over the bus / from the disk, decode on the fly)?
The row closes the item when the saving is below 20%.  A 2-bit quant of a Gaussian-like weight is close to
incompressible, so the expected answer is a few percent; this tool measures it on the real file.

What it does (host only, CPU-light, no GPU, no engine):
  * parses the GGUF header (tools/gguf_reader.py; tensor data is never read as a whole);
  * picks --tensors expert tensors (name contains --pattern, default "_exps", e.g. blk.N.ffn_gate_exps.weight),
    spread evenly over the layers, and from each one reads --window-kb kilobytes at a few block-aligned offsets
    through mmap (page cache sees only those pages), at most --max-mb in total;
  * reports, over the sample, per tensor type:
      byte entropy      order-0 Shannon entropy of the bytes, bits per byte (8.0 = incompressible)
      nibble entropy    the same over 4-bit halves, bits per nibble (4.0 = incompressible)
      crumb entropy     over 2-bit symbols (the Q2_0 code width), bits per symbol (2.0 = incompressible)
      per-position      mean over the block's byte positions of the byte entropy at that position (the fp16
                        scale bytes and the payload bytes have different statistics; this is the entropy a
                        coder that knows the block layout could use), and the scale-byte positions alone
      order-1 byte      conditional entropy H(byte | previous byte), bits per byte
      compressors       compressed / raw size of the sample windows for zlib -9, lzma, zstd -19 and lz4 -9
                        (zstd / lz4 through the python module when installed, else the CLI if on PATH, else
                        "n/a"); the saving = 1 - ratio
  * prints a verdict line: the best saving (entropy bound and real compressors) against --threshold (20%).

    python tools/pancake/expert_entropy.py MODEL.gguf [MODEL2.gguf ...] [--tensors 6] [--window-kb 512]
                                          [--windows 2] [--max-mb 8] [--pattern _exps] [--json OUT.json]

The sample is a few MB: the entropy estimates have a small positive bias (about 255 / (2 N ln 2) bits for N samples:
0.001 bits at 1M bytes, ~0.01 for the per-position histograms), printed per run.  Run it niced:
nice -n 19 python tools/pancake/expert_entropy.py ...
"""
from __future__ import annotations

import argparse
import json
import lzma
import mmap
import shutil
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from gguf_reader import BLOCK_GEOMETRY, GGUFFile  # noqa: E402


# ------------------------------------------------------------------ entropy estimators
def entropy_bits(counts) -> float:
    """Shannon entropy in bits of a histogram."""
    c = np.asarray(counts, dtype=np.float64)
    n = c.sum()
    if n <= 0:
        return 0.0
    p = c[c > 0] / n
    return float(-(p * np.log2(p)).sum())


def byte_entropy(a: np.ndarray) -> float:
    return entropy_bits(np.bincount(a.reshape(-1), minlength=256))


def nibble_entropy(a: np.ndarray) -> float:
    a = a.reshape(-1)
    return entropy_bits(np.bincount(np.concatenate([a & 0x0F, a >> 4]), minlength=16))


def crumb_entropy(a: np.ndarray) -> float:
    a = a.reshape(-1)
    return entropy_bits(np.bincount(np.concatenate([(a >> s) & 3 for s in (0, 2, 4, 6)]), minlength=4))


def order1_entropy(a: np.ndarray) -> float:
    """H(X_t | X_{t-1}) in bits per byte (joint entropy minus the entropy of the previous byte)."""
    a = a.reshape(-1).astype(np.int64)
    if a.size < 2:
        return 0.0
    joint = np.bincount(a[:-1] * 256 + a[1:], minlength=65536)
    prev = np.bincount(a[:-1], minlength=256)
    return entropy_bits(joint) - entropy_bits(prev)


def position_entropies(a: np.ndarray, block_bytes: int) -> np.ndarray:
    """Byte entropy at every position inside the block, over all blocks (a is a whole number of blocks)."""
    n = a.size // block_bytes
    m = a[: n * block_bytes].reshape(n, block_bytes)
    return np.array([entropy_bits(np.bincount(m[:, j], minlength=256)) for j in range(block_bytes)])


# ------------------------------------------------------------------ compressors
def _zstd_module():
    try:
        import zstandard  # type: ignore
        return zstandard
    except ImportError:
        return None


def _lz4_module():
    try:
        import lz4.frame  # type: ignore
        return lz4.frame
    except ImportError:
        return None


def _cli(cmd: list[str], data: bytes) -> int | None:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        r = subprocess.run(cmd, input=data, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return len(r.stdout)


def compressed_sizes(chunks: list[bytes]) -> dict[str, int | None]:
    """Compressed size summed over the chunks (each compressed on its own, as a per-expert coder would)."""
    out: dict[str, int | None] = {"zlib9": 0, "lzma": 0, "zstd19": 0, "lz4_9": 0}
    zm, lm = _zstd_module(), _lz4_module()
    for ch in chunks:
        out["zlib9"] += len(zlib.compress(ch, 9))
        out["lzma"] += len(lzma.compress(ch, preset=6))
        for key, fn in (("zstd19", lambda d: len(zm.ZstdCompressor(level=19).compress(d)) if zm else _cli(["zstd", "-19", "-q", "-c"], d)),
                        ("lz4_9", lambda d: len(lm.compress(d, compression_level=9)) if lm else _cli(["lz4", "-9", "-q", "-c"], d))):
            if out[key] is None:
                continue
            n = fn(ch)
            out[key] = None if n is None else out[key] + n
    return out


# ------------------------------------------------------------------ sampling
def pick_tensors(g: GGUFFile, pattern: str, count: int):
    """`count` tensors whose name contains `pattern`, evenly spread over the sorted list (so over layers and
    over gate / up / down); only encodings with a known block geometry (a known byte size)."""
    cand = [t for t in g.tensors if pattern in t.name and t.type_name in BLOCK_GEOMETRY and t.expected_bytes()]
    cand.sort(key=lambda t: t.offset)
    if count >= len(cand):
        return cand
    idx = sorted({round(i * (len(cand) - 1) / max(count - 1, 1)) for i in range(count)})
    return [cand[i] for i in idx]


def sample_windows(g: GGUFFile, mm, t, window_kb: int, windows: int, budget: int):
    """Block-aligned windows of one tensor, spread over its extent; `budget` bytes at most.  Returns arrays (uint8)."""
    be = BLOCK_GEOMETRY[t.type_name][1]
    total = t.expected_bytes()
    win = max(be, (window_kb * 1024 // be) * be)
    win = min(win, total, max(be, (budget // be) * be))
    out = []
    for i in range(windows):
        if total <= win:
            off = 0
        else:
            span = (total - win) // be
            off = (span * i // max(windows - 1, 1)) * be if windows > 1 else (span // 2) * be
        start = g.data_start + t.offset + off
        if start + win > len(mm):
            break
        out.append(np.frombuffer(mm, dtype=np.uint8, count=win, offset=start).copy())
        budget -= win
        if budget < be:
            break
    return out


def analyse(arrays_by_type: dict[str, list[np.ndarray]]) -> dict:
    res = {}
    for tname, arrs in arrays_by_type.items():
        be = BLOCK_GEOMETRY[tname][1]
        a = np.concatenate(arrs)
        pos = position_entropies(a, be)
        sizes = compressed_sizes([x.tobytes() for x in arrs])
        raw = int(a.size)
        res[tname] = {
            "bytes_sampled": raw,
            "block_bytes": be,
            "byte_entropy_bits": round(byte_entropy(a), 4),
            "nibble_entropy_bits": round(nibble_entropy(a), 4),
            "crumb_entropy_bits": round(crumb_entropy(a), 4),
            "order1_byte_entropy_bits": round(order1_entropy(a), 4),
            "position_entropy_mean_bits": round(float(pos.mean()), 4),
            "position_entropy_first2_bits": [round(float(x), 4) for x in pos[:2]],
            "position_entropy_rest_mean_bits": round(float(pos[2:].mean()), 4) if be > 2 else None,
            "bias_bits_order0": round(255 / (2 * raw * np.log(2)), 5),
            "compressed_bytes": sizes,
            "ratio": {k: (None if v is None else round(v / raw, 4)) for k, v in sizes.items()},
            "saving_pct": {k: (None if v is None else round(100 * (1 - v / raw), 2)) for k, v in sizes.items()},
            "saving_pct_entropy_order0": round(100 * (1 - byte_entropy(a) / 8), 2),
            "saving_pct_entropy_position": round(100 * (1 - float(pos.mean()) / 8), 2),
            "saving_pct_entropy_order1": round(100 * (1 - order1_entropy(a) / 8), 2),
        }
        res[tname]["best_saving_pct"] = max([v for v in res[tname]["saving_pct"].values() if v is not None]
                                            + [res[tname]["saving_pct_entropy_position"], res[tname]["saving_pct_entropy_order1"]])
    return res


def run(path: Path, tensors: int, window_kb: int, windows: int, max_mb: float, pattern: str) -> dict:
    g = GGUFFile(path)
    picked = pick_tensors(g, pattern, tensors)
    if not picked:
        return {"file": path.name, "tensors_sampled": [], "types": {},
                "skipped": f"no tensor named *{pattern}* with a known block geometry (types in the file: {g.by_type()})"}
    budget = int(max_mb * 1024 * 1024)
    per_tensor = max(1, budget // len(picked))
    by_type: dict[str, list[np.ndarray]] = {}
    names = []
    with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for t in picked:
            arrs = sample_windows(g, mm, t, window_kb, windows, per_tensor)
            if arrs:
                by_type.setdefault(t.type_name, []).extend(arrs)
                names.append(f"{t.name} [{t.type_name}] {sum(a.size for a in arrs) // 1024} KiB")
    return {"file": path.name, "tensors_sampled": names, "types": analyse(by_type)}


def verdict(report: dict, threshold: float) -> tuple[bool, float]:
    best = max(v["best_saving_pct"] for v in report["types"].values())
    return best < threshold, best


def print_report(rep: dict, threshold: float) -> None:
    if not rep["types"]:
        print(f"\n{rep['file']}: skipped, {rep.get('skipped')}")
        return
    print(f"\n{rep['file']}: {len(rep['tensors_sampled'])} tensors sampled")
    for n in rep["tensors_sampled"]:
        print("   ", n)
    for tname, r in rep["types"].items():
        print(f"  type {tname} ({r['block_bytes']} B blocks), {r['bytes_sampled'] / 1048576:.2f} MiB sampled "
              f"(order-0 bias ~{r['bias_bits_order0']} bits)")
        print(f"    byte entropy    {r['byte_entropy_bits']:.4f} / 8   nibble {r['nibble_entropy_bits']:.4f} / 4   "
              f"crumb(2-bit) {r['crumb_entropy_bits']:.4f} / 2   order-1 byte {r['order1_byte_entropy_bits']:.4f} / 8")
        print(f"    per-position    mean {r['position_entropy_mean_bits']:.4f} / 8   scale bytes {r['position_entropy_first2_bits']}"
              f"   payload mean {r['position_entropy_rest_mean_bits']}")
        print("    compressors (saving %): " + "  ".join(
            f"{k} {'n/a' if v is None else f'{v:+.2f}'}" for k, v in r["saving_pct"].items()))
        print(f"    entropy bounds (saving %): order-0 {r['saving_pct_entropy_order0']:+.2f}  "
              f"per-position {r['saving_pct_entropy_position']:+.2f}  order-1 {r['saving_pct_entropy_order1']:+.2f}")
    closed, best = verdict(rep, threshold)
    print(f"  VERDICT: best saving {best:.2f}% vs threshold {threshold:g}% -> "
          + ("below the threshold: an entropy-coded / compressed expert path is not worth building (A29 closes)"
             if closed else "ABOVE the threshold: look closer (more tensors, other layers) before deciding"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gguf", nargs="+", type=Path)
    ap.add_argument("--pattern", default="_exps", help="tensor-name substring that selects the routed experts")
    ap.add_argument("--tensors", type=int, default=6, help="expert tensors to sample, spread over the file")
    ap.add_argument("--window-kb", type=int, default=512, help="size of one block-aligned window")
    ap.add_argument("--windows", type=int, default=2, help="windows per tensor")
    ap.add_argument("--max-mb", type=float, default=8.0, help="total bytes read from the file")
    ap.add_argument("--threshold", type=float, default=20.0, help="saving %% below which the item closes")
    ap.add_argument("--json", type=Path, help="write the reports here")
    a = ap.parse_args(argv)
    reps = []
    for p in a.gguf:
        rep = run(p, a.tensors, a.window_kb, a.windows, a.max_mb, a.pattern)
        print_report(rep, a.threshold)
        reps.append(rep)
    if a.json:
        a.json.write_text(json.dumps(reps, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
