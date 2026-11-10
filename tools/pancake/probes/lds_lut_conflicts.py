#!/usr/bin/env python3
"""lds_lut_conflicts.py - MOD_PLAN A37: LDS bank-conflict model of the EXPDEC gate/up codebook lookups (static analysis).

QUESTION   expdec_gu_kernel (src/kernels/cuda/iq_kernels.cu) looks up the IQ2_S (1024 x 8 B) / IQ2_XXS (256 x 8 B)
           codebook in LDS with an index taken from the weights: 4 lookups per (row, k), lane t of a 16-lane sub-warp at
           k = t + 32 j and t + 16 + 32 j, the two sub-warps of a wave32 on rows r and r + 1.  How many LDS clocks does a
           wave's 8-byte lookup take, plain and with the STRATA_IQ_LUT_DUP layouts (FLUTE's duplicated LUT)?
MODEL      32 banks x 4 B; a 64-bit read is served 16 lanes per clock (one sub-warp per pass); a pass costs the largest
           number of distinct 8-byte slots in one bank pair (the same slot is a broadcast).  Copies: G copies, copy g in
           bank pairs [g B, (g + 1) B), B = 16 / G, lane t reads copy t / B (lut_dup_slot in iq_kernels.cu).
           A model, not a measurement: the rocprofv3 counters SQC_LDS_BANK_CONFLICT / SQC_LDS_IDX_ACTIVE confirm it.
BUILD      nothing (Python 3 + numpy).
RUN        python tools/pancake/probes/lds_lut_conflicts.py --uniform [--waves 20000]
           python tools/pancake/probes/lds_lut_conflicts.py --gguf MODEL-00001-of-0000N.gguf [--rows 128]
           (--gguf reads the first ffn_gate_exps tensor of type IQ2_S (22) and of IQ2_XXS (16): `rows` rows of expert 0)
READ       "clocks per wave lookup" per layout; ideal 2.00 (two conflict-free passes).
"""
import argparse
import struct
import sys

import numpy as np

QK = 256
BLOCK_BYTES = {22: 82, 16: 66}
ENTRIES = {22: 1024, 16: 256}
NAMES = {22: "IQ2_S", 16: "IQ2_XXS"}


def slot(i, copies, t):
    """uint2 slot of entry i read by lane t (0..15) of a sub-warp; copies = 1 is the plain table."""
    if copies == 1:
        return i
    b = 16 // copies
    return (i // b) * 16 + (t - t % b) + (i % b)


def pass_clocks(slots):
    """one 16-lane pass: the most distinct slots in one bank pair"""
    per = {}
    for s in set(slots):
        per.setdefault(s % 16, set()).add(s)
    return max(len(v) for v in per.values())


def wave_clocks(idx_sw0, idx_sw1, copies):
    """one wave32 lookup: the two sub-warps' 16 indices each"""
    return (pass_clocks([slot(i, copies, t) for t, i in enumerate(idx_sw0)])
            + pass_clocks([slot(i, copies, t) for t, i in enumerate(idx_sw1)]))


def uniform(n_entries, copies, waves, rng):
    tot = 0
    for _ in range(waves):
        a = rng.integers(0, n_entries, 16)
        b = rng.integers(0, n_entries, 16)
        tot += wave_clocks(a.tolist(), b.tolist(), copies)
    return tot / waves


# ------------------------------------------------------------------------------------------------ GGUF rows
def _gguf_tensors(f):
    def rd(fmt):
        return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))

    def rstr():
        (n,) = rd("Q")
        return f.read(n).decode("utf-8", "replace")

    size = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def skip(t):
        if t in size:
            return f.read(size[t])
        if t == 8:
            return rstr()
        if t == 9:
            (at,) = rd("I")
            (n,) = rd("Q")
            if at in size:
                f.seek(size[at] * n, 1)
            else:
                for _ in range(n):
                    skip(at)
            return None
        raise ValueError(f"gguf value type {t}")

    magic, _ver = rd("II")
    if magic != 0x46554747:
        sys.exit("not a GGUF file")
    n_t, n_kv = rd("QQ")
    align = 32
    for _ in range(n_kv):
        k = rstr()
        (t,) = rd("I")
        v = skip(t)
        if k == "general.alignment":
            align = struct.unpack("<I", v)[0]
    tens = []
    for _ in range(n_t):
        name = rstr()
        (nd,) = rd("I")
        dims = rd("Q" * nd)
        (ty,) = rd("I")
        (off,) = rd("Q")
        tens.append((name, dims, ty, off))
    base = (f.tell() + align - 1) // align * align
    return tens, base


def gguf_rows(path, ty, rows):
    with open(path, "rb") as f:
        tens, base = _gguf_tensors(f)
        for name, dims, t, off in tens:
            if t == ty and "ffn_gate_exps" in name:
                nb = dims[0] // QK
                f.seek(base + off)
                raw = np.frombuffer(f.read(nb * BLOCK_BYTES[ty] * rows), np.uint8)
                return name, raw.reshape(rows, nb, BLOCK_BYTES[ty])
    return None, None


def block_indices(ty, blk, ib):
    """the 4 codebook indices of q8_1 block ib (0..7) of one IQ block (Split<22/16>::load)"""
    if ty == 22:   # qs[0..31] low 8 bits, qh[ib] the high 2 bits of each of the 4
        qs, qh = blk[2:66], blk[66:74]
        return [int(qs[4 * ib + l]) | ((int(qh[ib]) << (8 - 2 * l)) & 0x300) for l in range(4)]
    qs = blk[2:66]   # IQ2_XXS: 8 bytes per q8_1 block, the first 4 are the indices
    return [int(qs[8 * ib + l]) for l in range(4)]


def real(raw, ty, copies):
    """clocks per wave lookup over the kernel's exact access order on these rows; plus the index histogram"""
    hist = np.zeros(ENTRIES[ty], np.int64)
    tot = cnt = 0
    nrows = raw.shape[0]
    steps = [[t + 32 * j for t in range(16)] for j in range(3)] + [[t + 16 + 32 * j for t in range(16)] for j in range(2)]
    for r in range(0, nrows - 1, 2):
        for ks in steps:
            idx = [[block_indices(ty, raw[rr, k // 8], k % 8) for k in ks] for rr in (r, r + 1)]
            for l in range(4):
                a = [idx[0][t][l] for t in range(16)]
                b = [idx[1][t][l] for t in range(16)]
                for i in a + b:
                    hist[i] += 1
                tot += wave_clocks(a, b, copies)
                cnt += 1
    return tot / max(cnt, 1), hist


def layouts(ty):
    return (1, 2, 4) if ty == 22 else (1, 2, 4, 8, 16)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--uniform", action="store_true", help="uniform random indices")
    g.add_argument("--gguf", help="model file (first split) with IQ2_S / IQ2_XXS ffn_gate_exps tensors")
    ap.add_argument("--rows", type=int, default=128, help="--gguf: rows of expert 0 to replay (default 128)")
    ap.add_argument("--waves", type=int, default=20000, help="--uniform: wave lookups to draw (default 20000)")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    for ty in (22, 16):
        if a.uniform:
            rng = np.random.default_rng(a.seed)
            res = {c: uniform(ENTRIES[ty], c, a.waves, rng) for c in layouts(ty)}
            head = f"{NAMES[ty]} uniform indices"
        else:
            name, raw = gguf_rows(a.gguf, ty, a.rows)
            if raw is None:
                print(f"{NAMES[ty]}: no ffn_gate_exps tensor of type {ty} in this file")
                continue
            res, hist = {}, None
            for c in layouts(ty):
                res[c], h = real(raw, ty, c)
                hist = h if hist is None else hist
            p = hist / hist.sum()
            ent = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
            head = (f"{NAMES[ty]} {name}, {a.rows} rows: {int((hist > 0).sum())}/{ENTRIES[ty]} entries used, "
                    f"entropy {ent:.2f} bits")
        print(head)
        print("   clocks per wave lookup (ideal 2.00): " + ", ".join(
            f"{c} cop{'y' if c == 1 else 'ies'} ({ENTRIES[ty] * 8 * c // 1024} KB) {v:.2f}" for c, v in res.items()))


if __name__ == "__main__":
    main()
